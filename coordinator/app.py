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
  COORDINATOR_HEARTBEAT_FRESH_SECONDS live-owner grace window, default 900
  COORDINATOR_MODE         shadow (default) | coordinator
"""

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

MAX_BODY = 32 * 1024 * 1024  # 32MB; observed max manifest ~7MB

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
    os.environ.get("COORDINATOR_HEARTBEAT_FRESH_SECONDS", "900")
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
            machine = (qs.get("machine") or [""])[0]
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
                stale_after_seconds = (
                    LIFECYCLE_STALE_AFTER_DAYS * 86400.0)
                for r in conn.execute(
                    "SELECT machine, last_seen, state, current_exq, "
                    "progress_json, seconds_elapsed, seconds_remaining, "
                    "last_shutdown_at, shutdown_reason, "
                    "expected_wake_condition "
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
                    row["lifecycle_state"] = db.lifecycle_state(
                        row.get("last_seen"),
                        row.get("last_shutdown_at"),
                        live_threshold_seconds=LIFECYCLE_LIVE_SECONDS,
                        stale_after_seconds=stale_after_seconds,
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
            machine = body.get("machine") or machine_tok
            git_verdict = body.get("git_verdict")  # may be None
            if not qid:
                self._send(400, {"error": "queue_id required"})
                return
            conn = db.connect(DB_PATH)
            try:
                coord_verdict = db.evaluate_claim(
                    conn, qid, machine, STALE_HOURS,
                    HEARTBEAT_FRESH_SECONDS)
                if MODE == "shadow":
                    diverged = db.log_claim(
                        conn, qid, machine, git_verdict, coord_verdict)
                    db.apply_git_outcome(conn, qid, machine, git_verdict)
                    self._send(200, {"verdict": coord_verdict,
                                     "diverged": bool(diverged),
                                     "authoritative": False})
                else:
                    real = db.try_claim(
                        conn, qid, machine, STALE_HOURS,
                        HEARTBEAT_FRESH_SECONDS)
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
                    db.log_claim(conn, qid, machine, None, real,
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
            machine = body.get("machine") or machine_tok
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
                    conn, body.get("machine") or machine_tok,
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
            machine = body.get("machine") or machine_tok
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
            machine = body.get("machine") or machine_tok
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
