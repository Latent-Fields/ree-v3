"""Experiment coordinator -- pure stdlib (http.server + sqlite3).

Zero third-party dependencies by design: the deploy target is a minimal
CX22 and a venv/dependency is itself a fragility source. Low-QPS internal
service (a handful of workers polling ~every 30s) so ThreadingHTTPServer is
more than adequate.

Auth: every endpoint except /health requires `Authorization: Bearer <tok>`.
Tokens live in a JSON file mapping token -> machine label (per-worker, so a
single worker can be revoked without rotating everyone). The socket should
bind to the WireGuard interface IP only (set COORDINATOR_BIND_HOST); the
token is defense-in-depth, not the only control.

All printed text is ASCII-only (Windows cp1252 safety).

Config (env):
  COORDINATOR_DB           sqlite path (default ./coordinator.db)
  COORDINATOR_BIND_HOST    default 127.0.0.1 (set to the WG IP in prod)
  COORDINATOR_BIND_PORT    default 8787
  COORDINATOR_TOKENS_FILE  JSON {token: machine} (default ./tokens.json)
  COORDINATOR_STALE_HOURS  stale-claim cutoff, default 6
  COORDINATOR_HEARTBEAT_FRESH_SECONDS live-owner grace window, default 3600
                           (see db.HEARTBEAT_FRESH_DEFAULT_SECONDS docstring
                           for why this is not the same number as the one
                           below)
  COORDINATOR_CLAIM_REAP_QUIET_SECONDS departed-owner silence before its
                           claims are reapable, default 900 (0 disables)
  COORDINATOR_MODE         shadow (default) | coordinator
"""

import base64
import datetime
import gzip
import hashlib
import hmac
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import db
import manifest_spool

# machine_identity.py lives in ree-v3/, one directory up from coordinator/.
# Same import shim phase3_preflight.py and phase3_verify.py already use.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from machine_identity import canonical_machine_name  # noqa: E402

MAX_BODY = 32 * 1024 * 1024  # 32MB; observed max manifest ~7MB


def _canon(machine):
    """Canonicalise a machine name arriving at the ingest boundary.

    THE INGEST BOUNDARY IS THE ONLY PLACE THIS HAPPENS. Every mutable
    coordination key -- the `heartbeats` PRIMARY KEY, `experiments`
    .claimed_by_machine, `commands.machine` -- is resolved here, once, so
    db.py can go on comparing plain strings and every downstream reader sees
    one identity per physical box.

    Why it is needed: macOS re-suffixes LocalHostName on a Bonjour collision,
    so this fleet's Mac has reported `DLAPTOP-4.local` and `DLAPTOP-5.local`
    for the same laptop, and `experiment_runner._get_machine_name` now
    resolves both to `DLAPTOP`. Against a coordinator that string-matches,
    that rename SPLITS the box in two: a second heartbeat row (PK is
    `machine`), a claim its owner can no longer release (`release_claim`),
    and -- worst -- a claim that `_has_fresh_owner_heartbeat` reads as
    ownerless and hands to a second worker while the first is still running
    it. Full analysis: REE_assembly/evidence/planning/
    coordinator_canonical_machine_identity_staged_20260816.md.

    NOT a `-<digits>` strip. `canonical_machine_name` is an ALLOWLIST
    (machine_identity.SUFFIX_BLIND_BASES, one entry), so `ree-cloud-1` ..
    `-5` and `ree-worker-1` .. `-4` pass through untouched. Collapsing those
    would give the whole fleet one heartbeat row and one colliding claim
    space -- a fleet-wide outage, not a cosmetic bug. That property is
    pinned coordinator-side by test_machine_identity_ingest.py, not merely
    inherited from tests/contracts/test_machine_identity.py.

    Falls back to the raw value when canonicalisation yields "" (empty or
    None input), and passes non-strings through untouched, so this can never
    turn a present name into a blank NOR raise on a malformed body -- the
    callers' own "machine required" guards keep deciding what is valid, and
    they still see exactly what the client sent.
    """
    if not isinstance(machine, str):
        return machine
    return canonical_machine_name(machine) or machine

# Remote-control command kinds the coordinator accepts on POST /commands/issue.
# Mirrors ree-v3/runner_remote_control.VALID_COMMAND_KINDS exactly so a command
# issued through the coordinator is indistinguishable from one issued through
# the legacy git command-file.
VALID_COMMAND_KINDS = (
    "stop", "force_stop", "pause", "resume", "suspend", "resume_run",
    "kick", "release_claim", "reclassify",
)
# Kinds that are meaningless without a target queue_id (mirrors serve.py
# append_machine_command's guard).
_COMMAND_KINDS_REQUIRING_QUEUE_ID = ("kick", "release_claim")

DB_PATH = os.environ.get("COORDINATOR_DB", os.path.join(
    os.path.dirname(__file__), "coordinator.db"))
BIND_HOST = os.environ.get("COORDINATOR_BIND_HOST", "127.0.0.1")
BIND_PORT = int(os.environ.get("COORDINATOR_BIND_PORT", "8787"))
TOKENS_FILE = os.environ.get("COORDINATOR_TOKENS_FILE", os.path.join(
    os.path.dirname(__file__), "tokens.json"))
STALE_HOURS = float(os.environ.get("COORDINATOR_STALE_HOURS", "6"))
HEARTBEAT_FRESH_SECONDS = int(
    os.environ.get("COORDINATOR_HEARTBEAT_FRESH_SECONDS",
                   str(db.HEARTBEAT_FRESH_DEFAULT_SECONDS))
)
# Lifecycle state thresholds (Phase 3 readiness). The "live" window is
# generous (covers brief network blips between heartbeats). The watchdog
# window bounds how long a graceful shutdown stays "gracefully_offline"
# before escalating to "stale" -- a machine that announced it was going
# down and never came back IS a stale machine, regardless of intent.
LIFECYCLE_LIVE_SECONDS = int(
    os.environ.get("COORDINATOR_LIFECYCLE_LIVE_SECONDS", "300")
)
LIFECYCLE_STALE_AFTER_DAYS = float(
    os.environ.get("COORDINATOR_LIFECYCLE_STALE_AFTER_DAYS", "7")
)
# Drain-window claim fence: how long a pending shutdown notice makes /claim
# refuse a machine with verdict "draining". Must exceed the ACPI drain
# window the scaler's `hcloud server shutdown` opens (~26 min measured
# 2026-07-30). Set to 0 to disable. See db.claim_fence_active.
CLAIM_FENCE_SECONDS = int(
    os.environ.get("COORDINATOR_CLAIM_FENCE_SECONDS",
                   str(db.CLAIM_FENCE_DEFAULT_SECONDS))
)
# Stale-claim reaper: how long a machine must be silent AFTER announcing a
# machine shutdown before another worker may take over its claims, without
# waiting out COORDINATOR_STALE_HOURS. Set to 0 to disable (recovery then
# falls back to the 6h floor alone). See db.claim_orphaned_by_departure.
CLAIM_REAP_QUIET_SECONDS = int(
    os.environ.get("COORDINATOR_CLAIM_REAP_QUIET_SECONDS",
                   str(db.CLAIM_REAP_QUIET_DEFAULT_SECONDS))
)
MODE = os.environ.get("COORDINATOR_MODE", "shadow")

# /writer-health reads this file, written by sync_daemon on every tick.
# sync_daemon + app.py run as separate systemd units (ree-sync-daemon /
# ree-coordinator), so in-process state sharing is not available -- the
# file is the shared channel. Same env var on both processes ensures they
# agree on the path; default sibling-of-script keeps the deploy template
# simple.
WRITER_HEALTH_FILE = os.environ.get(
    "PHASE3_WRITER_HEALTH_FILE",
    os.path.join(os.path.dirname(__file__), "writer_health.json"))

# Errors stamped onto a writer record age out after this window. A writer
# that errored, recovered, and then ran cleanly for >10 min has demonstrably
# self-healed; surfacing the stale error in the explorer would false-alarm.
WRITER_HEALTH_ERROR_TTL_S = 600

_tokens_lock = threading.Lock()
_tokens = {}


def _utc_iso_now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_writer_health():
    """Read the snapshot persisted by sync_daemon. Returns None when the
    file is missing or malformed (writer hasn't ticked yet, or a partial
    write). Never raises."""
    try:
        with open(WRITER_HEALTH_FILE, "r", encoding="utf-8") as fh:
            doc = json.loads(fh.read())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    return doc


def _age_out_errors(snapshot, cutoff_iso):
    """Mutate snapshot in place: drop last_error entries older than the
    cutoff. cutoff_iso is a UTC ISO string (`YYYY-MM-DDTHH:MM:SSZ`);
    ISO-8601 sorts lexicographically so string compare is safe."""
    writers = snapshot.get("writers") or {}
    if not isinstance(writers, dict):
        return
    for rec in writers.values():
        if not isinstance(rec, dict):
            continue
        err = rec.get("last_error")
        if isinstance(err, dict) and err.get("at", "") < cutoff_iso:
            rec["last_error"] = None


def load_tokens():
    global _tokens
    with _tokens_lock:
        if os.path.exists(TOKENS_FILE):
            with open(TOKENS_FILE, "r", encoding="utf-8") as fh:
                _tokens = json.load(fh)
        else:
            _tokens = {}


def auth_machine(header_value):
    """Return the machine label for a valid bearer token, else None.
    Constant-time compare against each known token."""
    if not header_value or not header_value.startswith("Bearer "):
        return None
    presented = header_value[len("Bearer "):].strip()
    with _tokens_lock:
        for tok, machine in _tokens.items():
            if hmac.compare_digest(presented, tok):
                return machine
    return None



# ---------------------------------------------------------------------------
# PHASE-2 POST handlers for TASK_CLAIMS.json / TASK_CHIPS.json.
#
# Kept as plain module-level functions in one dispatch table rather than
# another eleven `if path == ...` arms inside do_POST: do_POST is already ~500
# lines and each new arm there has to re-do the body/auth boilerplate.
#
# Each returns (http_status, payload_dict). Every payload carries "verdict"
# verbatim from db.py so a client never has to infer the outcome from the
# status code alone -- the CLI branches on the verdict string, the status code
# is for anything speaking plain HTTP.
#
# ASCII-only in every string, per this module's header.
# ---------------------------------------------------------------------------

def _tc_open(conn, body, machine_tok):
    session_id = body.get("session_id")
    if not session_id:
        return 400, {"error": "session_id is required"}
    resources = body.get("resources") or []
    if not isinstance(resources, list):
        return 400, {"error": "resources must be a list"}
    verdict, payload = db.try_open_task_claim(
        conn,
        session_id=session_id,
        session_label=body.get("session_label") or "",
        task=body.get("task") or "",
        resources=resources,
        allow_overlap=bool(body.get("allow_overlap")),
        spawned_by=body.get("spawned_by"),
        claimed_at=body.get("claimed_at"),
        stale_hours=float(body.get("stale_after_hours")
                          or db.TASK_CLAIM_STALE_HOURS_DEFAULT),
    )
    out = dict(payload)
    out["verdict"] = verdict
    if verdict == "owned_by_other":
        return 409, out
    if verdict == "error":
        return 500, out
    return 200, out


def _tc_close(conn, body, machine_tok):
    session_id = body.get("session_id")
    if not session_id:
        return 400, {"error": "session_id is required"}
    closed_at = body.get("closed_at")
    note = body.get("completion_note")
    if not closed_at or note is None:
        # Mirrors the CLI's own refusal: a bare status:done cannot be told
        # apart from an abandoned claim (root CLAUDE.md).
        return 400, {"error": "closed_at and completion_note are both required"}
    verdict, payload = db.close_task_claim(
        conn, session_id=session_id, closed_at=closed_at,
        completion_note=note, claimed_at=body.get("claimed_at"))
    out = dict(payload)
    out["verdict"] = verdict
    if verdict == "ambiguous":
        return 409, out
    if verdict == "not_found":
        return 404, out
    if verdict == "error":
        return 500, out
    return 200, out


def _tc_renew(conn, body, machine_tok):
    session_id = body.get("session_id")
    if not session_id:
        return 400, {"error": "session_id is required"}
    verdict, payload = db.renew_task_claim(
        conn, session_id=session_id, claimed_at=body.get("claimed_at"),
        new_claimed_at=body.get("new_claimed_at"),
        stale_hours=float(body.get("stale_after_hours")
                          or db.TASK_CLAIM_STALE_HOURS_DEFAULT))
    out = dict(payload)
    out["verdict"] = verdict
    if verdict in ("ambiguous", "would_lose_ownership"):
        return 409, out
    if verdict == "not_found":
        return 404, out
    if verdict == "error":
        return 500, out
    return 200, out


def _tc_amend(conn, body, machine_tok):
    session_id = body.get("session_id")
    note = body.get("completion_note")
    if not session_id or note is None:
        return 400, {"error": "session_id and completion_note are required"}
    verdict, payload = db.amend_task_claim(
        conn, session_id=session_id, completion_note=note,
        claimed_at=body.get("claimed_at"))
    out = dict(payload)
    out["verdict"] = verdict
    if verdict == "ambiguous":
        return 409, out
    if verdict == "not_found":
        return 404, out
    if verdict == "error":
        return 500, out
    return 200, out


def _tc_dedupe(conn, body, machine_tok):
    # Accepted no-op -- D3. The duplicate class it cleans up cannot be created
    # under the composite primary key.
    session_id = body.get("session_id")
    if not session_id:
        return 400, {"error": "session_id is required"}
    verdict, payload = db.dedupe_task_claim(
        conn, session_id=session_id, claimed_at=body.get("claimed_at"))
    out = dict(payload)
    out["verdict"] = verdict
    return 200, out


def _chip_record(conn, body, machine_tok):
    chip = body.get("chip")
    if not isinstance(chip, dict):
        return 400, {"error": "chip object is required"}
    verdict, payload = db.record_chip(conn, chip)
    out = dict(payload)
    out["verdict"] = verdict
    if verdict == "missing_marker":
        return 400, out
    if verdict == "ref_collision":
        return 409, out
    if verdict == "error":
        return 500, out
    return 200, out


def _chip_claim(conn, body, machine_tok):
    if not body.get("chip_ref") and not body.get("task_id"):
        return 400, {"error": "chip_ref or task_id is required"}
    verdict, payload = db.try_claim_chip(
        conn, chip_ref=body.get("chip_ref"), task_id=body.get("task_id"),
        claimed_by=body.get("claimed_by"),
        claimed_host=body.get("claimed_host"), note=body.get("note"),
        claimed_at=body.get("claimed_at"),
        stale_hours=float(body.get("stale_after_hours")
                          or db.CHIP_CLAIM_STALE_HOURS_DEFAULT))
    out = dict(payload)
    out["verdict"] = verdict
    if verdict in ("already_claimed", "not_open"):
        return 409, out
    if verdict == "not_found":
        return 404, out
    if verdict == "error":
        return 500, out
    return 200, out


def _chip_unclaim(conn, body, machine_tok):
    if not body.get("chip_ref") and not body.get("task_id"):
        return 400, {"error": "chip_ref or task_id is required"}
    verdict, payload = db.unclaim_chip(
        conn, chip_ref=body.get("chip_ref"), task_id=body.get("task_id"),
        note=body.get("note"))
    out = dict(payload)
    out["verdict"] = verdict
    if verdict == "not_found":
        return 404, out
    if verdict == "error":
        return 500, out
    return 200, out


def _chip_resolve(conn, body, machine_tok):
    if not body.get("chip_ref") and not body.get("task_id"):
        return 400, {"error": "chip_ref or task_id is required"}
    verdict, payload = db.resolve_chip(
        conn, status=body.get("status"), chip_ref=body.get("chip_ref"),
        task_id=body.get("task_id"), note=body.get("note"),
        resolved_by_session_id=body.get("resolved_by_session_id"),
        note_auto=bool(body.get("note_auto")),
        force=bool(body.get("force")), resolved_at=body.get("resolved_at"))
    out = dict(payload)
    out["verdict"] = verdict
    if verdict == "bad_status":
        return 400, out
    if verdict == "terminal_conflict":
        return 409, out
    if verdict == "not_found":
        return 404, out
    if verdict == "error":
        return 500, out
    return 200, out


def _chip_attach(conn, body, machine_tok):
    chip_ref = body.get("chip_ref")
    task_id = body.get("task_id")
    if not chip_ref or not task_id:
        return 400, {"error": "chip_ref and task_id are both required"}
    verdict, payload = db.attach_chip(
        conn, chip_ref=chip_ref, task_id=task_id,
        attached_by_session_id=body.get("attached_by_session_id"))
    out = dict(payload)
    out["verdict"] = verdict
    if verdict in ("task_id_taken", "already_attached"):
        return 409, out
    if verdict == "not_found":
        return 404, out
    if verdict == "error":
        return 500, out
    return 200, out


def _chip_amend_prompt(conn, body, machine_tok):
    chip_ref = body.get("chip_ref")
    prompt = body.get("prompt")
    if not chip_ref or prompt is None:
        return 400, {"error": "chip_ref and prompt are both required"}
    verdict, payload = db.amend_chip_prompt(
        conn, chip_ref=chip_ref, prompt=prompt, reason=body.get("reason"))
    out = dict(payload)
    out["verdict"] = verdict
    if verdict == "missing_marker":
        return 400, out
    if verdict == "not_found":
        return 404, out
    if verdict == "error":
        return 500, out
    return 200, out


def _ws_append(conn, body, machine_tok):
    """POST /workspace_state/append (PHASE-4 first slice). Append-ONLY: there
    is deliberately no edit or delete verb for WORKSPACE_STATE entries --
    the file's git copy stays authoritative for everything already in it;
    this endpoint only spools NEW closing entries for the materializer to
    splice in (byte-preserving). client_git_write=true declares the client
    will also land the entry itself (dual-write soak); the materializer then
    only watches for it instead of splicing -- see
    db.submit_workspace_state_entry."""
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return 400, {"error": "text is required and must be non-empty",
                     "verdict": "empty_text"}
    verdict, payload = db.submit_workspace_state_entry(
        conn,
        text=text,
        ts=body.get("timestamp"),
        session_id=body.get("session_id") or "",
        client_git_write=bool(body.get("client_git_write")),
        host=machine_tok)
    out = dict(payload)
    out["verdict"] = verdict
    if verdict in ("empty_text", "bad_timestamp"):
        return 400, out
    if verdict == "error":
        return 500, out
    return 200, out


def _reclog_append(conn, body, machine_tok):
    """POST /recommendation_log/append (PHASE-4 slice, 2026-08-29). Append-
    ONLY, the jsonl-trivial clone of _ws_append: no edit or delete verb --
    the file's git copy stays authoritative for every line already in it;
    this endpoint only spools NEW records for the materializer to append
    verbatim. client_git_write=true declares the client will also land the
    line itself (dual-write soak); the materializer then only watches for it
    instead of appending -- see db.submit_recommendation_log_entry."""
    record = body.get("record")
    verdict, payload = db.submit_recommendation_log_entry(
        conn,
        record=record,
        session_id=body.get("session_id") or "",
        client_git_write=bool(body.get("client_git_write")),
        host=machine_tok)
    out = dict(payload)
    out["verdict"] = verdict
    if verdict in ("empty", "bad_json"):
        return 400, out
    if verdict == "error":
        return 500, out
    return 200, out


# /chip/archive is DELIBERATELY ABSENT and must stay that way in Phase 2 --
# plan doc D7. Its correctness gate is that the archive file has actually
# reached ORIGIN (cmd_archive fetches and verifies at origin_ref() before
# stripping, after the 2026-08-19 first-run failure). That is inherently a git
# fact with no DB equivalent, so archiving stays a git-side operation reading
# the materialized file. Revisit only in Phase 3.
_TASK_CLAIM_CHIP_POST = {
    "/task_claim/open": _tc_open,
    "/task_claim/close": _tc_close,
    "/task_claim/renew": _tc_renew,
    "/task_claim/amend": _tc_amend,
    "/task_claim/dedupe": _tc_dedupe,
    "/chip/record": _chip_record,
    "/chip/claim": _chip_claim,
    "/chip/unclaim": _chip_unclaim,
    "/chip/resolve": _chip_resolve,
    "/chip/attach": _chip_attach,
    "/chip/amend-prompt": _chip_amend_prompt,
    # PHASE-4 (append-only; the name of this dict predates it -- same
    # dispatch, same auth, same body plumbing, so it rides here rather than
    # growing a parallel table).
    "/workspace_state/append": _ws_append,
    "/recommendation_log/append": _reclog_append,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "REECoordinator/1.0"

    # ---- helpers -------------------------------------------------------
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > MAX_BODY:
            return None
        raw = self.rfile.read(length)
        if (self.headers.get("Content-Encoding") or "").lower() == "gzip":
            try:
                raw = gzip.decompress(raw)
            except OSError:
                return None
        return raw

    def _json_body(self):
        raw = self._read_body()
        if raw is None:
            return None
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def _authed(self):
        m = auth_machine(self.headers.get("Authorization"))
        if m is None:
            self._send(401, {"error": "unauthorized"})
            return None
        return m

    def log_message(self, fmt, *args):
        # ASCII-only, single line, to stderr.
        sys.stderr.write("[coordinator] %s - %s\n" % (
            self.address_string(), (fmt % args)))

    # ---- routing -------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"ok": True, "mode": MODE})
            return
        if self._authed() is None:
            return
        if path == "/writer-health":
            # phase3 writer-health snapshot. Source of truth = the JSON
            # file sync_daemon writes on every tick (see WRITER_HEALTH_FILE
            # on both processes). Replaces the explorer's SSH + journal +
            # git-log scrape for "is the writer alive" -- one HTTP call
            # over WireGuard tells the explorer everything it needs.
            #
            # 503 when the file is missing or malformed: that means the
            # sync_daemon process hasn't published a snapshot yet (cold
            # start, deploy gap, or it's wedged). Explorer interprets
            # 503 as "fall back to SSH probe".
            snapshot = _read_writer_health()
            if snapshot is None:
                self._send(503, {
                    "error": "writer_health snapshot not available",
                    "path": WRITER_HEALTH_FILE,
                })
                return
            # Defensive copy: don't mutate disk-derived state in place.
            snapshot = json.loads(json.dumps(snapshot))
            cutoff_dt = (
                datetime.datetime.utcnow()
                - datetime.timedelta(seconds=WRITER_HEALTH_ERROR_TTL_S))
            cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            _age_out_errors(snapshot, cutoff_iso)
            # Always stamp now_utc fresh: the file's now_utc is from the
            # last sync_daemon tick. A caller computing "how recently did
            # the writer tick" wants the file's last_tick_at vs THIS
            # response's now_utc.
            snapshot["now_utc"] = _utc_iso_now()
            self._send(200, snapshot)
            return
        if path == "/commands":
            qs = parse_qs(urlparse(self.path).query)
            # Canonicalised on the FETCH side, and POST /commands/issue is
            # canonicalised on the ISSUE side. Those two must move together
            # or not at all: canonicalising one alone converts a deliverable
            # command into a permanently pending one. Round-trip pinned by
            # test_machine_identity_ingest.CommandChannelDriftTest.
            machine = _canon((qs.get("machine") or [""])[0])
            conn = db.connect(DB_PATH)
            try:
                cmds = db.fetch_pending_commands(conn, machine)
            finally:
                conn.close()
            self._send(200, {"machine": machine, "commands": cmds})
            return
        if path == "/queue/active":
            # Active worklist from the coordinator mirror (Phase 3 authority).
            conn = db.connect(DB_PATH)
            try:
                rows = conn.execute(
                    "SELECT item_json FROM experiments "
                    "WHERE status IN ('pending', 'claimed', 'suspended') "
                    "ORDER BY priority DESC, queue_id"
                ).fetchall()
                items = []
                for row in rows:
                    raw = row["item_json"]
                    try:
                        items.append(json.loads(raw))
                    except (TypeError, ValueError):
                        continue
            finally:
                conn.close()
            self._send(200, {
                "source": "coordinator_db",
                "now_utc": _utc_iso_now(),
                "items": items,
            })
            return
        if path == "/shadow/divergence":
            conn = db.connect(DB_PATH)
            try:
                rows = conn.execute(
                    "SELECT queue_id, machine, git_verdict, coord_verdict, "
                    "detail, logged_at FROM claim_log WHERE diverged=1 "
                    "ORDER BY id DESC LIMIT 200").fetchall()
                total = conn.execute(
                    "SELECT COUNT(*) c FROM claim_log").fetchone()["c"]
                ndiv = conn.execute("SELECT COUNT(*) c FROM claim_log "
                                    "WHERE diverged=1").fetchone()["c"]
                dstat = db.divergence_stats(conn)
            finally:
                conn.close()
            self._send(200, {"total_claims": total, "divergences": ndiv,
                             "divergences_explained": dstat["explained"],
                             "divergences_blocking": dstat["blocking"],
                             "rows": [dict(r) for r in rows]})
            return
        if path == "/shadow/status":
            # One-call operator soak snapshot: traffic seen, divergence
            # count, per-machine heartbeat freshness. Backs check_shadow.py.
            conn = db.connect(DB_PATH)
            try:
                total = conn.execute(
                    "SELECT COUNT(*) c FROM claim_log").fetchone()["c"]
                ndiv = conn.execute("SELECT COUNT(*) c FROM claim_log "
                                    "WHERE diverged=1").fetchone()["c"]
                dstat = db.divergence_stats(conn)
                nexp = conn.execute(
                    "SELECT COUNT(*) c FROM experiments").fetchone()["c"]
                machines = []
                # Rows are reported by their STORED key, not re-canonicalised
                # on read. Since every writer of this table now resolves at
                # ingest (_canon), new rows are already canonical -- and the
                # one-off residue of the transition (an orphaned
                # `DLAPTOP-4.local` row beside a new `DLAPTOP` one) SHOULD
                # stay visible here rather than be merged away: that readout
                # is how an operator knows the old row is safe to delete.
                # See machine_identity.py's closing note on telemetry
                # filenames -- let the stale pair age out, do not hide it.
                stale_after_seconds = (
                    LIFECYCLE_STALE_AFTER_DAYS * 86400.0)
                for r in conn.execute(
                    "SELECT machine, last_seen, state, current_exq, "
                    "progress_json, seconds_elapsed, seconds_remaining, "
                    "last_shutdown_at, shutdown_reason, "
                    "expected_wake_condition, claim_fence_cleared_at, "
                    "last_process_exit_at, process_exit_reason "
                    "FROM heartbeats ORDER BY machine"
                ).fetchall():
                    row = dict(r)
                    raw_pj = row.pop("progress_json", None)
                    try:
                        row["progress"] = (
                            json.loads(raw_pj)
                            if isinstance(raw_pj, str) and raw_pj else {})
                    except (TypeError, ValueError):
                        row["progress"] = {}
                    # announced_offline_at, not last_shutdown_at directly:
                    # a runner process exit is stored in its own column
                    # since the 2026-07-30 split, and it has always counted
                    # as "announced going away" for THIS readout (the runner
                    # announces runner_drain_complete precisely to get
                    # gracefully_offline on a manual systemctl stop). The
                    # fence and the reaper below deliberately do NOT use the
                    # helper -- they mean a machine shutdown specifically.
                    row["lifecycle_state"] = db.lifecycle_state(
                        row.get("last_seen"),
                        db.announced_offline_at(
                            row.get("last_shutdown_at"),
                            row.get("last_process_exit_at")),
                        live_threshold_seconds=LIFECYCLE_LIVE_SECONDS,
                        stale_after_seconds=stale_after_seconds,
                    )
                    # Surfaced because the 2026-07-30 orphan was hard to
                    # diagnose precisely BECAUSE nothing showed that a
                    # machine was mid-drain: lifecycle_state reads "live"
                    # throughout the window (the runner keeps heartbeating),
                    # so the drain is invisible to every existing readout.
                    row["claim_fenced"] = db.claim_fence_active(
                        row.get("last_shutdown_at"),
                        row.get("claim_fence_cleared_at"),
                        row.get("shutdown_reason"),
                        fence_seconds=CLAIM_FENCE_SECONDS,
                    )
                    # Reaper visibility. Because recovery happens inside the
                    # claim predicate, an orphaned claim stays visibly
                    # `claimed` until some worker next asks for it -- this
                    # flag is what says "and its owner is gone", so the
                    # orphan is diagnosable from the status readout rather
                    # than only from the claim that eventually succeeds.
                    row["departed"] = db.machine_departed(
                        row.get("last_seen"),
                        row.get("last_shutdown_at"),
                        row.get("claim_fence_cleared_at"),
                        row.get("shutdown_reason"),
                        quiet_seconds=CLAIM_REAP_QUIET_SECONDS,
                    )
                    machines.append(row)
                recent = [dict(r) for r in conn.execute(
                    "SELECT queue_id, machine, git_verdict, coord_verdict, "
                    "logged_at FROM claim_log WHERE diverged=1 "
                    "ORDER BY id DESC LIMIT 20").fetchall()]
            finally:
                conn.close()
            self._send(200, {"mode": MODE, "total_claims": total,
                             "divergences": ndiv,
                             "divergences_explained": dstat["explained"],
                             "divergences_blocking": dstat["blocking"],
                             "adjusted_divergences": dstat["blocking"],
                             "experiments_in_mirror": nexp,
                             "machines": machines,
                             "recent_divergences": recent})
            return
        if path == "/task_claim/list":
            # PHASE-1 read-only observability only -- see
            # task_claim_chip_shadow_sync.py's module docstring. No mutating
            # /task_claim/* or /chip/* endpoint exists yet; that is PHASE-2
            # (plan doc section 5.2.5), not yet built.
            qs = parse_qs(urlparse(self.path).query)
            status = (qs.get("status") or [None])[0]
            conn = db.connect(DB_PATH)
            try:
                if status:
                    rows = conn.execute(
                        "SELECT entry_json FROM task_claims WHERE status=? "
                        "ORDER BY claimed_at", (status,)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT entry_json FROM task_claims "
                        "ORDER BY claimed_at").fetchall()
                claims = []
                for r in rows:
                    try:
                        claims.append(json.loads(r["entry_json"]))
                    except (TypeError, ValueError):
                        continue
            finally:
                conn.close()
            self._send(200, {"claims": claims, "stale_after_hours": STALE_HOURS})
            return
        if path == "/task_claim/check":
            # Mirrors task_claim.py's own arbitration predicate: does any
            # ACTIVE claim already own one of the named resources? GET and
            # writes nothing (D10 in the plan doc) -- this is the START-TIME
            # predicate a chip STOP-CHECK can carry once something actually
            # calls it; nothing does yet in Phase 1.
            qs = parse_qs(urlparse(self.path).query)
            resources = qs.get("resource") or []
            conn = db.connect(DB_PATH)
            try:
                rivals = []
                if resources:
                    q = ",".join("?" * len(resources))
                    rivals = conn.execute(
                        "SELECT c.session_id, c.claimed_at, c.session_label, "
                        "       c.task, r.resource "
                        "FROM task_claim_resources r "
                        "JOIN task_claims c ON c.session_id=r.session_id "
                        "                  AND c.claimed_at=r.claimed_at "
                        "WHERE r.resource IN (%s) AND c.status='active' "
                        "ORDER BY c.claimed_at" % q,
                        resources,
                    ).fetchall()
            finally:
                conn.close()
            rivals = [dict(r) for r in rivals]
            self._send(200, {"owned": bool(rivals), "rivals": rivals})
            return
        if path == "/chip/list":
            qs = parse_qs(urlparse(self.path).query)
            status = (qs.get("status") or [None])[0]
            origin = (qs.get("origin") or [None])[0]
            chip_ref = (qs.get("chip_ref") or [None])[0]
            task_id = (qs.get("task_id") or [None])[0]
            try:
                limit = int((qs.get("limit") or ["500"])[0])
            except ValueError:
                limit = 500
            limit = max(1, min(limit, 5000))
            clauses = []
            params = []
            if status:
                clauses.append("status=?")
                params.append(status)
            if origin:
                clauses.append("origin=?")
                params.append(origin)
            # Exact-match lookups for the suppressed-record lag window: a chip
            # recorded with the git write suppressed exists only in this DB
            # until the hub materializer renders it, so same-session verbs
            # (resolve/claim) need a direct fetch by identity.
            if chip_ref:
                clauses.append("chip_ref=?")
                params.append(chip_ref)
            if task_id:
                clauses.append("task_id=?")
                params.append(task_id)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            conn = db.connect(DB_PATH)
            try:
                rows = conn.execute(
                    "SELECT entry_json FROM chip_ledger%s "
                    "ORDER BY spawned_at DESC LIMIT ?" % where,
                    params + [limit],
                ).fetchall()
                chips = []
                for r in rows:
                    try:
                        chips.append(json.loads(r["entry_json"]))
                    except (TypeError, ValueError):
                        continue
            finally:
                conn.close()
            self._send(200, {"chips": chips})
            return
        if path == "/task_claim/drift":
            # PHASE-1 soak-status readout: the evidence a human reads to
            # judge the "N days of zero drift" exit criterion (plan doc
            # frontmatter, PHASE-1 node). Mirrors /shadow/divergence's
            # shape for the existing experiment-claim shadow audit.
            qs = parse_qs(urlparse(self.path).query)
            try:
                limit = int((qs.get("limit") or ["50"])[0])
            except ValueError:
                limit = 50
            limit = max(1, min(limit, 1000))
            # ?since=<iso> windows the counts. The unwindowed totals are
            # cumulative and therefore permanently carry the 2026-08-27
            # pruned-done false positive, so a windowed read is the only way
            # the soak's exit criterion stays checkable. See
            # db.task_claim_chip_drift_summary.
            since = (qs.get("since") or [None])[0]
            conn = db.connect(DB_PATH)
            try:
                summary = db.task_claim_chip_drift_summary(
                    conn, limit=limit, since=since)
            finally:
                conn.close()
            self._send(200, summary)
            return
        if path == "/recommendation_log/pending":
            # PHASE-4 observability, RECLOG sibling of the WS spool below.
            conn = db.connect(DB_PATH)
            try:
                rows = db.pending_recommendation_log_entries(conn)
            finally:
                conn.close()
            pending = []
            for r in rows:
                pending.append({
                    "entry_id": r["entry_id"],
                    "session_id": r["session_id"],
                    "client_git_write": bool(r["client_git_write"]),
                    "submitted_at": r["submitted_at"],
                    "record_head": (r["record"] or "")[:120],
                })
            self._send(200, {"n_pending": len(pending), "pending": pending})
            return
        if path == "/workspace_state/pending":
            # PHASE-4 observability: the not-yet-materialized WS entry spool.
            # awaiting_client (client_git_write=1, dual-write soak) rows are
            # the ones to watch for staleness -- the materializer never
            # splices them, so one whose client crashed pre-push sits here
            # until a human (or a future takeover rule) intervenes.
            conn = db.connect(DB_PATH)
            try:
                rows = db.pending_workspace_state_entries(conn)
            finally:
                conn.close()
            pending = []
            for r in rows:
                pending.append({
                    "entry_id": r["entry_id"],
                    "ts": r["ts"],
                    "session_id": r["session_id"],
                    "client_git_write": bool(r["client_git_write"]),
                    "submitted_at": r["submitted_at"],
                    "first_line": (r["text"] or "").split("\n", 1)[0][:120],
                })
            self._send(200, {
                "n_pending": len(pending),
                "n_awaiting_client": sum(
                    1 for p in pending if p["client_git_write"]),
                "pending": pending})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        machine_tok = self._authed()
        if machine_tok is None:
            return
        body = self._json_body() if path != "/result" else None

        if path == "/claim":
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            qid = body.get("queue_id")
            # TWO names on purpose, and they must not be collapsed:
            #   machine     -- canonical; decides and RECORDS claim ownership
            #                  (evaluate_claim / try_claim / apply_git_outcome
            #                  write experiments.claimed_by_machine).
            #   machine_raw -- exactly what the client reported; goes only to
            #                  log_claim. claim_log is the shadow AUDIT TRAIL,
            #                  append-only, and its whole value is being a
            #                  faithful record of what each box said at the
            #                  time. Canonicalising it would erase the only
            #                  evidence an identity split ever happened --
            #                  which is precisely the evidence the "10533 vs
            #                  0" argument in machine_identity.py rests on.
            #                  Same reasoning as results.machine under /result.
            machine_raw = body.get("machine") or machine_tok
            machine = _canon(machine_raw)
            git_verdict = body.get("git_verdict")  # may be None
            if not qid:
                self._send(400, {"error": "queue_id required"})
                return
            conn = db.connect(DB_PATH)
            try:
                # Shadow mode leaves the fence off: git is authoritative
                # there, so a "draining" verdict the git path never
                # produces would register as a spurious divergence and
                # break the apples-to-apples comparison evaluate_claim
                # exists for. Production runs MODE=coordinator.
                fence = 0 if MODE == "shadow" else CLAIM_FENCE_SECONDS
                # Same reasoning for the reaper: it widens eligibility, so
                # in shadow mode it would produce 'ok' where the git path
                # says 'already_claimed' and register as a spurious
                # divergence. Off in shadow, on under MODE=coordinator.
                reap = 0 if MODE == "shadow" else CLAIM_REAP_QUIET_SECONDS
                coord_verdict = db.evaluate_claim(
                    conn, qid, machine, STALE_HOURS,
                    HEARTBEAT_FRESH_SECONDS, fence_seconds=fence,
                    reap_quiet_seconds=reap)
                if MODE == "shadow":
                    diverged = db.log_claim(
                        conn, qid, machine_raw, git_verdict, coord_verdict)
                    db.apply_git_outcome(conn, qid, machine, git_verdict)
                    self._send(200, {"verdict": coord_verdict,
                                     "diverged": bool(diverged),
                                     "authoritative": False})
                else:
                    real = db.try_claim(
                        conn, qid, machine, STALE_HOURS,
                        HEARTBEAT_FRESH_SECONDS,
                        fence_seconds=CLAIM_FENCE_SECONDS,
                        reap_quiet_seconds=CLAIM_REAP_QUIET_SECONDS)
                    # Phase 3: claim_log is the audit trail for who tried to
                    # claim what + the coordinator's verdict. The shadow path
                    # above logs evaluate_claim's predicted verdict; under
                    # MODE=coordinator the actually-applied verdict from
                    # try_claim is what we want to record. git_verdict is
                    # None (no git-side mutex consulted on the authoritative
                    # path; the legacy runner-side `claim:` commit is a
                    # separate, advisory channel). detail='phase3_only'
                    # marks rows that came in via the Phase 3 endpoint so
                    # mixed-mode periods are distinguishable post-hoc.
                    db.log_claim(conn, qid, machine_raw, None, real,
                                 detail="phase3_only")
                    self._send(200, {"verdict": real,
                                     "authoritative": True})
            finally:
                conn.close()
            return

        if path == "/claim/release":
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            qid = body.get("queue_id")
            machine = _canon(body.get("machine") or machine_tok)
            if not qid:
                self._send(400, {"error": "queue_id required"})
                return
            if MODE == "shadow":
                self._send(200, {"ok": True, "applied": False,
                                 "note": "shadow mode"})
                return
            conn = db.connect(DB_PATH)
            try:
                ok, note = db.release_claim(conn, qid, machine)
            finally:
                conn.close()
            self._send(200 if ok else 409,
                       {"ok": ok, "applied": ok, "note": note})
            return

        if path == "/heartbeat":
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            # PLAN.md step 6: clients may include `payload` (the full
            # runner-side dict that gets written to runner_heartbeats/
            # <machine>.json) so sync_daemon can materialise the file
            # from the coordinator DB. Old clients omit it -- the column
            # stays whatever it was (None on first insert, unchanged on
            # update) and the structured fields cover lifecycle state.
            payload = body.get("payload")
            payload_json = (json.dumps(payload, sort_keys=False)
                            if isinstance(payload, dict) else None)
            conn = db.connect(DB_PATH)
            try:
                db.upsert_heartbeat(
                    conn, _canon(body.get("machine") or machine_tok),
                    body.get("state"), body.get("current_exq"),
                    body.get("progress"), body.get("gpu"),
                    body.get("seconds_elapsed"),
                    body.get("seconds_remaining"),
                    payload_json=payload_json)
            finally:
                conn.close()
            self._send(200, {"ok": True})
            return

        if path == "/status":
            # PLAN.md step 6: store the full status payload so
            # sync_daemon can materialise runner_status/<machine>.json
            # from the DB. Body may be the bare payload dict OR
            # {machine, status: {...}} (coordinator_client's existing
            # report_status wrapper).
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            machine = _canon(body.get("machine") or machine_tok)
            payload = body.get("status") if "status" in body else body
            if not isinstance(payload, dict):
                self._send(400, {"error": "status payload must be dict"})
                return
            conn = db.connect(DB_PATH)
            try:
                db.record_status_payload(
                    conn, machine,
                    json.dumps(payload, sort_keys=False))
            finally:
                conn.close()
            self._send(200, {"ok": True})
            return

        if path == "/shutdown_notify":
            # A machine (or the scaler workflow on its behalf) announces an
            # intentional shutdown. Lets /shadow/status return
            # lifecycle_state="gracefully_offline" instead of "stale" until
            # the watchdog window expires (LIFECYCLE_STALE_AFTER_DAYS).
            #
            # Auth: bearer token. body.machine is REQUIRED -- the token's
            # machine label is never substituted. Empty bodies were
            # creating stray heartbeat rows for the token label (e.g. a
            # probe with no body via the scaler token wrote a "scaler"
            # row), so the endpoint now demands an explicit machine
            # field. The scaler workflow uses a coordinator-side token
            # tagged for that purpose and MAY post on behalf of any
            # machine (no per-machine restriction beyond having a valid
            # token -- the trust model is "GitHub Actions secrets =
            # coordinator-level trust", same as the cloud-scaler.yml
            # already enjoys via HCLOUD_TOKEN).
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            if not isinstance(body, dict):
                self._send(400, {"error": "machine required"})
                return
            machine = body.get("machine")
            if not machine or not isinstance(machine, str):
                self._send(400, {"error": "machine required"})
                return
            # Canonicalised AFTER the guard, so a malformed body still gets
            # the same 400. record_shutdown_notice upserts the heartbeats
            # row, and the claim fence + departure reaper both re-read it by
            # that key -- a drifted spelling here arms a fence nobody checks.
            machine = _canon(machine)
            reason = body.get("reason")
            wake = body.get("expected_wake_condition")
            conn = db.connect(DB_PATH)
            try:
                db.record_shutdown_notice(
                    conn, machine, reason=reason,
                    expected_wake_condition=wake)
            finally:
                conn.close()
            self._send(200, {"ok": True, "machine": machine,
                             "reason": reason,
                             "expected_wake_condition": wake})
            return

        if path == "/claim_fence/clear":
            # Disarm the drain-window claim fence for a machine. The scaler
            # calls this immediately after `hcloud server poweron` -- a
            # freshly-woken worker must be able to claim at once rather than
            # sitting out the remainder of its own shutdown's fence window.
            #
            # Same auth + explicit-machine contract as /shutdown_notify: the
            # token's own label is never substituted, so an empty body
            # cannot create a stray heartbeat row for the token name.
            # Idempotent -- re-clearing an already-clear machine is a no-op
            # in effect, so the scaler can call it unconditionally on every
            # poweron without tracking state.
            if body is None or not isinstance(body, dict):
                self._send(400, {"error": "bad body"})
                return
            machine = body.get("machine")
            if not machine or not isinstance(machine, str):
                self._send(400, {"error": "machine required"})
                return
            # Must resolve the same way /shutdown_notify does, or the scaler
            # clears the fence on one spelling's row while the armed one
            # stays armed and the freshly-woken worker keeps getting
            # "draining".
            machine = _canon(machine)
            conn = db.connect(DB_PATH)
            try:
                db.record_claim_fence_clear(conn, machine)
            finally:
                conn.close()
            self._send(200, {"ok": True, "machine": machine})
            return

        if path == "/commands/issue":
            # Issue a remote-control command for a machine. The Phase-3
            # replacement for serve.py writing runner_commands/<machine>.json.
            # Auth: bearer token (trust model = token == command-issue access,
            # same as the legacy git command-file). body.machine is REQUIRED
            # (the issuer token's label is never substituted -- serve.py
            # issues on behalf of cloud-2/3/4 with the operator token).
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            if not isinstance(body, dict):
                self._send(400, {"error": "machine required"})
                return
            machine = body.get("machine")
            if not machine or not isinstance(machine, str):
                self._send(400, {"error": "machine required"})
                return
            # ISSUE side. Paired with the FETCH side in GET /commands and the
            # ACK side in POST /commands/ack -- all three canonicalise, or
            # none may. serve.py's _coordinator_issue_command resolves before
            # it posts too, so the explorer and a hand-rolled curl land the
            # command in the same place.
            machine = _canon(machine)
            kind = body.get("kind")
            if kind not in VALID_COMMAND_KINDS:
                self._send(400, {"error": "unknown command kind",
                                 "kind": kind,
                                 "valid_kinds": list(VALID_COMMAND_KINDS)})
                return
            args = body.get("args") or {}
            if not isinstance(args, dict):
                self._send(400, {"error": "args must be an object"})
                return
            if kind in _COMMAND_KINDS_REQUIRING_QUEUE_ID and \
                    not args.get("queue_id"):
                self._send(400, {"error": "%s requires args.queue_id" % kind})
                return
            issued_by = body.get("issued_by") or "unknown"
            conn = db.connect(DB_PATH)
            try:
                row = db.insert_command(
                    conn, machine, kind, json.dumps(args), issued_by)
            finally:
                conn.close()
            self._send(200, {"ok": True, "command": row})
            return

        if path == "/commands/ack":
            # A runner acks a command it has executed. Stamps acked_at +
            # terminal result_status (done|failed) + result_note. After ack
            # the command no longer appears in GET /commands. Idempotent on
            # repeat (returns ok). machine defaults to the token's label;
            # ack is owner-guarded in db.ack_command.
            if body is None or not isinstance(body, dict):
                self._send(400, {"error": "bad body"})
                return
            cmd_id = body.get("id")
            if not isinstance(cmd_id, int):
                self._send(400, {"error": "integer command id required"})
                return
            # ACK side of the command channel -- see POST /commands/issue.
            # ack_command is owner-guarded (`row["machine"] != machine`), so
            # without this a runner cannot ack a command addressed to its
            # other spelling and the command never leaves the pending list.
            machine = _canon(body.get("machine") or machine_tok)
            result_status = body.get("result_status") or "done"
            if result_status not in ("done", "failed"):
                self._send(400, {"error": "result_status must be "
                                          "done or failed"})
                return
            result_note = body.get("result_note")
            conn = db.connect(DB_PATH)
            try:
                ok, note = db.ack_command(
                    conn, cmd_id, machine, result_status, result_note)
            finally:
                conn.close()
            self._send(200 if ok else 409,
                       {"ok": ok, "applied": ok, "note": note})
            return

        if path == "/result":
            raw = self._read_body()
            if raw is None:
                self._send(413, {"error": "body too large or bad gzip"})
                return
            try:
                manifest = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._send(400, {"error": "result not valid json"})
                return
            run_id = manifest.get("run_id")
            if not run_id:
                self._send(400, {"error": "run_id required"})
                return
            sha = hashlib.sha256(raw).hexdigest()
            conn = db.connect(DB_PATH)
            try:
                # DELIBERATELY NOT _canon()'d, and this is the one ingest
                # point that stays raw. `results.machine` is APPEND-ONLY
                # PROVENANCE -- the DB form of the result manifest, which
                # CLAUDE.md is explicit is "aliased on READ, deliberately
                # never rewritten", because recording a run under a name the
                # box was not reporting at the time is falsifying provenance.
                # Nothing keys coordination behaviour off this column; it is
                # a record of what happened, not a decision input.
                #
                # The manifest bytes spooled below carry the same raw name,
                # so DB and evidence file agree -- canonicalising here would
                # silently make them disagree.
                #
                # Same rule applies to claim_log.machine (see POST /claim).
                fresh = db.record_result(
                    conn, run_id, manifest.get("queue_id"),
                    manifest.get("machine") or machine_tok,
                    manifest.get("outcome"), sha, len(raw))
            finally:
                conn.close()
            # Phase 3 prep: when COORDINATOR_SPOOL_DIR is set, persist the
            # raw manifest bytes so sync_daemon's phase3_git_writer can
            # commit them into REE_assembly later. Unset by default ->
            # bit-identical to Phase 2. Spool failures do not fail the
            # POST; the runner's own evidence/ checkout is still the
            # authoritative copy under Phase 2 semantics.
            if fresh:
                manifest_spool.write_manifest(
                    run_id, raw,
                    manifest_relpath=manifest.get("manifest_relpath") if
                        isinstance(manifest, dict) else None,
                    received_at=db.utcnow(),
                    sha256_hex=sha,
                )
            self._send(200, {"ok": True, "run_id": run_id,
                             "idempotent_noop": (not fresh)})
            return

        if path == "/result/sidefile":
            # Phase 3 side-file sync: spool a run's COMPANION artifact (e.g.
            # an *_episode_log.json) so sync_daemon's phase3_git_writer
            # materialises it into REE_assembly/evidence/experiments/ in the
            # same commit as the run's manifest. Body is a JSON envelope
            # {run_id, relpath, content_b64[, sha256]} (gzip-encoded in
            # transit). Spooling is a no-op when COORDINATOR_SPOOL_DIR is
            # unset; the runner only POSTs here when PHASE3_SPOOL_SIDEFILES is
            # set, so this endpoint is dormant by default.
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            run_id = body.get("run_id")
            relpath = body.get("relpath")
            content_b64 = body.get("content_b64")
            if not run_id or not relpath or content_b64 is None:
                self._send(400, {"error":
                                 "run_id, relpath, content_b64 required"})
                return
            try:
                raw = base64.b64decode(content_b64)
            except (ValueError, TypeError):
                self._send(400, {"error": "content_b64 not valid base64"})
                return
            client_sha = body.get("sha256")
            sha = hashlib.sha256(raw).hexdigest()
            if client_sha and client_sha != sha:
                self._send(400, {"error": "sha256 mismatch"})
                return
            spooled = manifest_spool.write_sidefile(
                run_id, relpath, raw,
                received_at=db.utcnow(), sha256_hex=sha)
            self._send(200, {"ok": spooled is not None,
                             "run_id": run_id, "relpath": relpath,
                             "spooled": spooled is not None,
                             "bytes": len(raw)})
            return

        if path == "/queue/add":
            # Authoritative DB ingress for a NEW queue item. Under Phase 3 the
            # DB owns the queue and experiment_queue.json is a derived view
            # (phase3_queue_writer re-materialises it from the experiments
            # table). A producer that only `git commit`s the queue file can
            # have its addition ERASED by the next snapshot-writer tick if
            # reconcile loses the writer race (or the hub is wedged). POSTing
            # the item here upserts it directly into the experiments table, so
            # it survives re-materialisation; the git commit is kept only as a
            # legacy/fallback worklist signal. Mirror of /queue/remove on the
            # add side.
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            item = body.get("item")
            if item is None and body.get("queue_id"):
                # Convenience: allow the bare item dict as the body.
                item = {k: v for k, v in body.items()}
            if not isinstance(item, dict) or not item.get("queue_id"):
                self._send(400, {"error": "item with queue_id required"})
                return
            if not item.get("script"):
                self._send(400, {"error": "item.script required"})
                return
            qid = item["queue_id"]
            force_rerun = bool(item.get("force_rerun"))
            # Shadow mode: git is authoritative, so a direct DB add would be
            # reconciled away (reconcile_once deletes DB rows missing from the
            # git file under shadow's upsert_only=False). Do not mutate; the
            # caller's git commit is the ingress in shadow. Mirror /queue/remove.
            if MODE == "shadow":
                self._send(200, {"ok": True, "applied": False,
                                 "queue_id": qid, "note": "shadow mode; "
                                 "git is authoritative"})
                return
            conn = db.connect(DB_PATH)
            try:
                existing = db.get_queue_status(conn, qid)
                if existing in db.TERMINAL_STATUSES and not force_rerun:
                    # Re-adding a terminal id silently re-runs a done
                    # experiment. Refuse (mirrors validate_queue + the runner
                    # skip-completed guard). Caller should use a new
                    # letter/number, or set force_rerun:true deliberately.
                    self._send(409, {"ok": False, "applied": False,
                                     "queue_id": qid, "existing_status":
                                     existing, "error": "queue_id already has "
                                     "a terminal DB row; use a new id or "
                                     "force_rerun"})
                    return
                # A fresh add is always pending and unclaimed; never let a
                # caller inject a claimed/completed state. preserve_claim=True
                # keeps the DB's own claim/status for an in-flight (non-
                # terminal) re-add so a running experiment is not clobbered;
                # for a brand-new row (existing is None) it inserts pending.
                # A deliberate force_rerun RESETS the row, so preserve_claim
                # is False there (otherwise the old terminal status survives
                # and the rerun never re-enters the worklist).
                add = dict(item)
                add["status"] = "pending"
                add.pop("claimed_by", None)
                db.upsert_experiment(
                    conn, add, preserve_claim=not force_rerun)
            finally:
                conn.close()
            self._send(200, {"ok": True, "applied": True, "queue_id": qid,
                             "existed": existing is not None})
            return

        if path == "/queue/remove":
            if body is None or not body.get("queue_id"):
                self._send(400, {"error": "queue_id required"})
                return
            # Shadow: do not mutate the mirror destructively; sync_daemon
            # reconciles removals from experiment_queue.json (git is
            # authoritative in Phase 1). Coordinator mode marks the item
            # completed immediately so no other coordinator-mode worker can
            # reclaim it while the git queue removal propagates.
            applied = False
            if MODE != "shadow":
                conn = db.connect(DB_PATH)
                try:
                    applied = db.mark_queue_removed(
                        conn, body.get("queue_id"), body.get("reason"))
                finally:
                    conn.close()
            self._send(200, {"ok": True, "applied": applied})
            return

        # ---- PHASE-2 mutating task_claim / chip verbs -------------------
        #
        # Plan doc section 5.2.5. Namespaced under /task_claim/* and /chip/*,
        # NOT under /claim/* -- D1: /claim and /claim/release are already the
        # EXPERIMENT claim endpoints and mean something entirely different.
        #
        # Status-code convention, matching the plan: 409 for a refusal that is
        # a legitimate VERDICT (a rival owns the resource, a chip is already
        # claimed) so a client can branch on it; 400 for a malformed request;
        # 404 for an unknown key. A 409 is not an error to retry.
        #
        # D6: the bearer token identifies a MACHINE (machine_tok), but claims
        # and chips are per-SESSION -- session_id always comes from the body.
        # The token still gates access; it just is not the actor.
        if (path.startswith("/task_claim/") or path.startswith("/chip/")
                or path.startswith("/workspace_state/")
                or path.startswith("/recommendation_log/")):
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            handler = _TASK_CLAIM_CHIP_POST.get(path)
            if handler is None:
                self._send(404, {"error": "not found"})
                return
            conn = db.connect(DB_PATH)
            try:
                code, payload = handler(conn, body, machine_tok)
            finally:
                conn.close()
            self._send(code, payload)
            return

        self._send(404, {"error": "not found"})


def main():
    db.init_db(DB_PATH)
    load_tokens()
    srv = ThreadingHTTPServer((BIND_HOST, BIND_PORT), Handler)
    sys.stderr.write(
        "[coordinator] mode=%s listening on %s:%d db=%s tokens=%d\n" % (
            MODE, BIND_HOST, BIND_PORT, DB_PATH, len(_tokens)))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
