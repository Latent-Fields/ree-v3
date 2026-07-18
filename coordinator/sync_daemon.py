"""Sync daemon.

PHASE 1 (shadow) behaviour -- the default behaviour:
  * Periodically read experiment_queue.json (git is authoritative in
    shadow) and reconcile the coordinator DB mirror to match it, so the
    coordinator's claim logic always evaluates against fresh state.
  * Claim-level shadow checks happen via runner POST /claim (git_verdict
    vs coordinator evaluate_claim). State-level pre-upsert reconcile was
    removed 2026-05-20 (false positives; see SOAK_LOG.md E1/E2).
  * It does NOT write git. Read-only on the queue file. No autostash, no
    rebase -- this daemon is structurally incapable of the failure class
    the whole project exists to remove.

PHASE 3 (authoritative) behaviour is present but guarded OFF: becoming the
sole git writer (commit result manifests, push, snapshot queue) only
activates when SYNC_MODE=authoritative AND --i-understand-phase3 is passed.
Stubbed deliberately; do not enable until Phase 1 has proven out.

PHASE 2 (claim cutover) behaviour is selected by SYNC_MODE=coordinator:
git remains the queue worklist and result/status transport, but the DB is
the claim authority. Reconciliation refreshes metadata and removals from
git without overwriting coordinator claim state.

All printed text is ASCII-only.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time

import db
import manifest_spool

DEFAULT_QUEUE = os.path.join(
    os.path.dirname(__file__), "..", "experiment_queue.json")


def _load_queue_json(queue_path):
    """Return the parsed queue dict from the AUTHORITATIVE git ref
    (SYNC_QUEUE_REF, default origin/main), fetched read-only.

    Why not just read queue_path: the local working-tree copy is only as
    fresh as this box's last `git pull`. When this box's runner is drained
    nothing pulls, so the file goes stale and every other machine's
    git-claim looks like a state-divergence (mirror=claimed vs stale
    file=pending) -- a false positive, not a coordinator-logic fault.
    `git fetch` + `git show <ref>:file` never touches the working tree
    (no autostash risk, consistent with sync_daemon being git-read-only).
    Degrades to the local file if git is unavailable, logging that it is
    running on a possibly-stale source."""
    repo = os.path.dirname(os.path.abspath(queue_path))
    rel = os.path.basename(queue_path)
    ref = os.environ.get("SYNC_QUEUE_REF", "origin/main")
    try:
        subprocess.run(["git", "-C", repo, "fetch", "--quiet", "origin"],
                        check=True, capture_output=True, timeout=30)
        out = subprocess.run(
            ["git", "-C", repo, "show", "%s:%s" % (ref, rel)],
            check=True, capture_output=True, timeout=15)
        return json.loads(out.stdout.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 -- degrade, never crash
        if not os.path.exists(queue_path):
            sys.stderr.write(
                "[sync] no git queue and no local file: %r\n" % exc)
            return None
        sys.stderr.write(
            "[sync] WARN authoritative git queue unavailable (%r); "
            "falling back to STALE local file\n" % exc)
        with open(queue_path, "r", encoding="utf-8") as fh:
            return json.load(fh)


def reconcile_once(conn, queue_path, *, claim_authority="git",
                   upsert_only=False):
    """Make the mirror match the AUTHORITATIVE git queue (not this box's
    possibly-stale local file). Returns (n_items, n_state_divergences).

    claim_authority='git' is Phase 1 shadow: git claims are truth and
    state-level mismatches are logged. claim_authority='coordinator' is
    Phase 2: git is only the worklist, so existing DB claim state is
    preserved and git-vs-DB claim mismatches are not divergence rows.

    upsert_only=False (default): items missing from the git queue but
    present in the DB are DELETEd (Phase 1/2 semantics: git is the
    worklist; what's gone from the file has been completed/removed by
    the authoritative path).

    upsert_only=True is Phase 3 authoritative: the DB owns the queue,
    `experiment_queue.json` is a DERIVED view written back by
    `phase3_queue_writer`. Operator hand-edits to the file MUST be
    additions only (use `POST /queue/remove` to drop items via the
    coordinator); items missing from the file are NOT deleted from the
    DB, because the DB row's status='completed' is the authoritative
    "this was done" record and must survive the writeback round-trip.
    """
    qdata = _load_queue_json(queue_path)
    if qdata is None:
        return (0, 0)
    items = {it["queue_id"]: it for it in qdata.get("items", [])
             if it.get("queue_id")}

    divergences = 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        mirror = {r["queue_id"]: r for r in conn.execute(
            "SELECT queue_id, status, claimed_by_machine FROM experiments"
        ).fetchall()}

        for qid, item in items.items():
            # Phase 1 (git authority): upsert mirror from authoritative git
            # queue each tick. Pre-upsert state-reconcile logged false
            # divergences when the mirror was briefly ahead of origin/main
            # (harness E1 in SOAK_LOG.md) -- claim-level shadow /claim
            # compares git_verdict vs evaluate_claim instead.
            db.upsert_experiment(
                conn, item, preserve_claim=(claim_authority == "coordinator"))

        stale = set(mirror) - set(items)
        if not upsert_only:
            # Items no longer in the queue file have been completed/removed
            # by the authoritative path; drop them from the mirror so the
            # coordinator does not hand them out.
            for qid in stale:
                conn.execute(
                    "DELETE FROM experiments WHERE queue_id=?", (qid,))
        elif stale:
            # Phase 3 authoritative path: file is a DERIVED view, so items
            # missing from the file must NOT be deleted from the DB (the
            # row may be claimed/running/completed and the file just hasn't
            # caught up yet). But surface the silent revert: an operator
            # hand-edit removing a pending item will reappear in the file
            # on the next phase3_queue_writer tick with no signal. WARN
            # cheaply lists what was preserved so the operator can spot
            # an unintended revert in the log.
            non_terminal = [
                qid for qid in stale
                if mirror[qid]["status"] not in ("completed", "failed")
            ]
            if non_terminal:
                sys.stderr.write(
                    "[sync] upsert_only: %d non-terminal item(s) missing "
                    "from queue file are PRESERVED in DB (use "
                    "POST /queue/remove to drop): %s\n" % (
                        len(non_terminal), non_terminal[:5]))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return (len(items), divergences)


# Set True only after phase3_git_writer steps 1-6 are implemented and
# phase3_preflight.py + phase3_verify.py pass on the live fleet.
PHASE3_GIT_WRITER_READY = True

# Hub paths (override via env when deploying). PHASE3_REE_ASSEMBLY and
# PHASE3_REE_V3 are validated below, AFTER the _validate_* helpers are
# defined (top-to-bottom Python module execution).
PHASE3_QUEUE_FILE = os.environ.get(
    "COORDINATOR_QUEUE_FILE", DEFAULT_QUEUE)


# Max manifests to write+commit per writer tick. Bounds tick latency and
# limits worst-case rollback if a push fails (we ROLLBACK committed_at for
# the unpushed batch, then retry). Must be >0; an operator setting it to
# 0 or negative via env override would produce silently misleading
# refusals (writer slices an empty batch out of a non-empty spool, hits
# "no manifests staged" and refuses for the wrong reason). Validate at
# module load and fall back to the default with a loud log.
#
# LOW-B note: env validation runs at MODULE IMPORT. Setting PHASE3_*
# env vars (this one, and the branch / relpath / interval validators
# below) after `import sync_daemon` is silently ignored. The daemon's
# main() reads the validated module-level constants. Operators must
# set the env in the systemd unit / launcher, before the Python
# process starts.
def _validate_batch_size(raw, default=32):
    try:
        v = int(raw)
    except (TypeError, ValueError):
        sys.stderr.write(
            "[phase3] PHASE3_BATCH_SIZE=%r is not an integer; "
            "using default %d\n" % (raw, default))
        return default
    if v <= 0:
        sys.stderr.write(
            "[phase3] PHASE3_BATCH_SIZE=%d is not > 0; using default %d\n"
            % (v, default))
        return default
    return v


PHASE3_BATCH_SIZE = _validate_batch_size(
    os.environ.get("PHASE3_BATCH_SIZE", "32"))


# LOW-C: same fail-loudly-then-default pattern for the other env knobs
# that can silently misdirect the writer when malformed. Empty string,
# whitespace-only, or values containing whitespace / path separators get
# rejected back to the documented default with a stderr warning.
def _validate_branch_name(raw, env_name, default):
    """Branch names: non-empty, no whitespace, no path separators."""
    if not isinstance(raw, str) or not raw.strip():
        sys.stderr.write(
            "[phase3] %s=%r is empty/blank; using default %r\n"
            % (env_name, raw, default))
        return default
    s = raw.strip()
    if any(c.isspace() for c in s) or "/" in s or "\\" in s:
        sys.stderr.write(
            "[phase3] %s=%r contains whitespace or path separator; "
            "using default %r\n" % (env_name, raw, default))
        return default
    return s


def _validate_abs_repo_path(raw, env_name, default):
    """Repo checkout absolute paths: non-empty, no embedded whitespace,
    absolute (POSIX) or default. The writer runs `git` with cwd=this; an
    empty or whitespace-only env override would silently target the
    daemon's cwd and write to whatever happened to be checked out there.
    Falls back to the documented default with a stderr warning.

    Existence is NOT checked here -- the writer's `_hub_working_tree_clean`
    + `git fetch` give clearer per-tick failures with the actual git error
    output, and a path that doesn't exist YET (e.g. fresh deploy before
    the checkout is created) shouldn't refuse module import."""
    if not isinstance(raw, str) or not raw.strip():
        sys.stderr.write(
            "[phase3] %s=%r is empty/blank; using default %r\n"
            % (env_name, raw, default))
        return default
    s = raw.strip()
    if any(c.isspace() for c in s):
        sys.stderr.write(
            "[phase3] %s=%r contains whitespace; using default %r\n"
            % (env_name, raw, default))
        return default
    if not os.path.isabs(s):
        sys.stderr.write(
            "[phase3] %s=%r is not absolute; using default %r\n"
            % (env_name, raw, default))
        return default
    return s


def _validate_repo_relpath(raw, env_name, default):
    """Repo-internal relative paths: non-empty, no leading slash, no
    `..` segment escape. The writer writes files to repo/<relpath>;
    accepting absolute paths or parent-dir escapes would let an env
    override write outside the managed checkout."""
    if not isinstance(raw, str) or not raw.strip():
        sys.stderr.write(
            "[phase3] %s=%r is empty/blank; using default %r\n"
            % (env_name, raw, default))
        return default
    s = raw.strip()
    # `os.path.isabs` covers both POSIX "/foo" and Windows drive paths.
    if os.path.isabs(s) or s.startswith("/") or s.startswith("\\"):
        sys.stderr.write(
            "[phase3] %s=%r is absolute; using default %r\n"
            % (env_name, raw, default))
        return default
    parts = s.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        sys.stderr.write(
            "[phase3] %s=%r contains '..' segment; using default %r\n"
            % (env_name, raw, default))
        return default
    return s


def _validate_float(raw, env_name, default):
    """Float env knobs (SYNC_INTERVAL): parse with default fallback.
    Same shape as _validate_batch_size but for floats. Negative and
    zero are rejected -- a zero interval would spin the daemon."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        sys.stderr.write(
            "[phase3] %s=%r is not a number; using default %s\n"
            % (env_name, raw, default))
        return default
    if v <= 0:
        sys.stderr.write(
            "[phase3] %s=%s is not > 0; using default %s\n"
            % (env_name, v, default))
        return default
    return v


# Truthy vocabulary shared with the legacy PHASE3_AUTO_RESET_ON_REBASE_CONFLICT
# env check (see _sync_to_origin). Keep the two in sync so a value that turns
# on one path turns on the other.
_TRUTHY = ("1", "true", "TRUE", "yes", "YES")
_FALSEY = ("0", "false", "FALSE", "no", "NO", "")


def _validate_bool(raw, env_name, default=False):
    """Parse a boolean env knob using the same truthy set the existing
    PHASE3_AUTO_RESET_ON_REBASE_CONFLICT check uses. Anything outside the
    recognised truthy/falsey vocabulary falls back to `default` with a loud
    stderr warning -- the same fail-loud-then-default shape as
    _validate_batch_size / _validate_float."""
    if raw is None:
        return default
    s = str(raw).strip()
    if s in _TRUTHY:
        return True
    if s in _FALSEY:
        return False
    sys.stderr.write(
        "[phase3] %s=%r is not a recognised boolean; using default %s\n"
        % (env_name, raw, default))
    return default


# PLAN.md step 5: queue snapshot writeback. Materialises the canonical
# experiment_queue.json from the coordinator DB and pushes ree-v3. Gated
# by its OWN flag (independent of the result writer) so result-cutover
# and queue-cutover can be staged separately. Default False until the
# implementation is reviewed AND the runner-side
# PHASE3_DISABLE_RUNNER_QUEUE_PUSH flag is set on every worker.
PHASE3_QUEUE_WRITER_READY = True

# Shadow/canary flag for the queue-writer conflict-recovery self-heal (see
# _sync_to_origin + phase3_queue_writer). When True, a rebase CONFLICT in the
# queue writer's _sync_to_origin -- which happens when an operator/IGW/session
# commit edits experiment_queue.json on origin while the writer holds a
# retained (push-rejected) snapshot commit -- is recovered losslessly via
# `git reset --hard origin/<branch>` + re-materialise-from-DB instead of
# refusing the tick (the historic ~4.5h push-rejected/conflict wedge,
# 2026-06-02). Safe because the queue is DB-authoritative: reconcile_once
# absorbs operator file additions into the DB BEFORE the writer materialises,
# and _check_ahead_writer_authored guarantees only writer-authored ahead
# commits are ever dropped (operator commits, already on origin, are preserved
# by the reset). ADDITIVE over the legacy PHASE3_AUTO_RESET_ON_REBASE_CONFLICT
# env: when this flag is set the queue writer forces recovery on; when unset it
# falls back to the env, so this flag never DISABLES recovery the env already
# provides -- it only adds a queue-scoped, observable (n_conflict_recoveries)
# control. Default False + legacy env unset = the pre-flag refuse-on-conflict
# behaviour. Scoped to the queue writer ONLY -- the result and heartbeat writers
# touch writer-exclusive paths that do not conflict, so they stay on the bare
# env-fallback policy. Module-load validated: set it in the systemd unit env
# BEFORE the process starts.
PHASE3_QUEUE_CONFLICT_RECOVERY = _validate_bool(
    os.environ.get("PHASE3_QUEUE_CONFLICT_RECOVERY", "0"),
    "PHASE3_QUEUE_CONFLICT_RECOVERY", False)

# Belt-and-suspenders to the 2026-06-06 flat-only silent-drop fix
# (REE_assembly commit c92458c731, which made the LOCAL governance.sh
# converter robust). When True, phase3_git_writer ALSO materialises the
# canonical runs/<run_id>/{manifest.json,metrics.json,summary.md} pack for
# each flat evidence result it commits -- reusing the EXACT field mapping in
# sync_v3_results.build_runpack_docs -- so a cloud worker's result is
# immediately scoreable on origin without waiting for the next local
# governance.sh run (which is when the converter would otherwise back-fill
# the pack). Default False = shadow / bit-identical: no pack written, no
# cross-checkout import attempted, the writer commits only the flat manifest
# exactly as before. Skip-if-pack-exists is enforced at the write site so a
# runner-synced pack is NEVER clobbered -- this only back-fills the gap where
# a worker's runs/ pack failed to sync and only the flat manifest landed.
# Module-load validated: set it in the systemd unit env BEFORE the process
# starts (same constraint as the other PHASE3_* env knobs).
PHASE3_MATERIALIZE_RUNPACK = _validate_bool(
    os.environ.get("PHASE3_MATERIALIZE_RUNPACK", "0"),
    "PHASE3_MATERIALIZE_RUNPACK", False)

# Phase 3 side-file sync (2026-06-10). When True, phase3_git_writer ALSO
# materialises each run's spooled COMPANION artifacts (e.g. an
# *_episode_log.json that fishtank_viz.html reads) into
# REE_assembly/evidence/experiments/ and git-adds them into the SAME commit as
# the run's manifest -- closing the gap where a side-file-emitting experiment
# (a fishtank showcase) running on a cloud worker stranded its episode_log on
# the worker's disk (confirmed for V3-EXQ-664 on ree-cloud-3, 2026-06-10). The
# worker side is gated by PHASE3_SPOOL_SIDEFILES too (runner POSTs companions
# only when set); enable both together. Default False = bit-identical: the
# writer ignores the companion spool, committing only the flat manifest exactly
# as before. Module-load validated -- set it in the systemd unit env BEFORE the
# process starts (same constraint as the other PHASE3_* env knobs).
PHASE3_SPOOL_SIDEFILES = _validate_bool(
    os.environ.get("PHASE3_SPOOL_SIDEFILES", "0"),
    "PHASE3_SPOOL_SIDEFILES", False)

# Hub paths for the queue writer. ree-v3 checkout is separate from the
# REE_assembly checkout used by the result writer.
PHASE3_REE_V3 = _validate_abs_repo_path(
    os.environ.get("PHASE3_REE_V3", "/home/ree/REE_Working/ree-v3"),
    "PHASE3_REE_V3", "/home/ree/REE_Working/ree-v3")
PHASE3_REE_V3_BRANCH = _validate_branch_name(
    os.environ.get("PHASE3_REE_V3_BRANCH", "main"),
    "PHASE3_REE_V3_BRANCH", "main")
# Relative path inside the ree-v3 checkout for the canonical queue file.
PHASE3_QUEUE_RELPATH = _validate_repo_relpath(
    os.environ.get("PHASE3_QUEUE_RELPATH", "experiment_queue.json"),
    "PHASE3_QUEUE_RELPATH", "experiment_queue.json")

# REE_assembly checkout path (result writer + heartbeat writer share it).
# Same validation as PHASE3_REE_V3; deferred to here so the helper exists.
PHASE3_REE_ASSEMBLY = _validate_abs_repo_path(
    os.environ.get("PHASE3_REE_ASSEMBLY",
                   "/home/ree/REE_Working/REE_assembly"),
    "PHASE3_REE_ASSEMBLY", "/home/ree/REE_Working/REE_assembly")


# PLAN.md step 6: derived heartbeats + runner_status writeback. The writer
# materialises evidence/experiments/runner_heartbeats/<machine>.json AND
# runner_status/<machine>.json from rows in the heartbeats table, replacing
# the per-runner runner_remote_control.push_heartbeat git push (the original
# autostash-war bug source). Gated by its OWN flag so the heartbeat cutover
# can stage separately from result + queue cutovers.
PHASE3_HEARTBEAT_WRITER_READY = True

# Subdirectories (relative to REE_assembly) where the writer materialises
# the per-machine files. Match the existing legacy layout that explorer
# + scaler workflow + governance scripts already read.
PHASE3_HEARTBEATS_RELDIR = "evidence/experiments/runner_heartbeats"
PHASE3_STATUS_RELDIR = "evidence/experiments/runner_status"

# Default branch on the hub's REE_assembly checkout. Override via env if the
# hub is ever moved to a non-master deploy layout.
PHASE3_ASSEMBLY_BRANCH = _validate_branch_name(
    os.environ.get("PHASE3_ASSEMBLY_BRANCH", "master"),
    "PHASE3_ASSEMBLY_BRANCH", "master")

# State-change-triggered commit knobs (2026-05-31 redesign). Replace the
# previous every-N-seconds commit storm with: commit only when meaningful
# fleet state changes (queue_id / idle / runner_pid / machine added / went
# silent); debounce stacking; liveness floor so external viewers see "fleet
# alive" via fresh git timestamps even when no events fire.
#
#   PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL  -- minimum seconds between commits
#       (default 300 = 5 min). When a state change fires inside the
#       debounce window, the writer holds; the next post-window tick
#       commits whatever has accumulated.
#   PHASE3_HEARTBEAT_LIVENESS_INTERVAL  -- maximum seconds between commits
#       (default 1800 = 30 min). When no state change has fired for this
#       long, the writer forces a "liveness tick" commit batching the
#       current state.
#   PHASE3_HEARTBEAT_STALE_AFTER        -- machine considered "silent" if
#       its heartbeat.last_tick_utc has not advanced for this many
#       seconds (default 600 = 10 min). silent <-> active is a tracked
#       state-change axis.
PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL = _validate_float(
    os.environ.get("PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL", "300"),
    "PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL", 300.0)
PHASE3_HEARTBEAT_LIVENESS_INTERVAL = _validate_float(
    os.environ.get("PHASE3_HEARTBEAT_LIVENESS_INTERVAL", "1800"),
    "PHASE3_HEARTBEAT_LIVENESS_INTERVAL", 1800.0)
PHASE3_HEARTBEAT_STALE_AFTER = _validate_float(
    os.environ.get("PHASE3_HEARTBEAT_STALE_AFTER", "600"),
    "PHASE3_HEARTBEAT_STALE_AFTER", 600.0)


# Module-level state for the state-change-triggered writer. Persists across
# ticks within a single daemon process; reset between test fixtures via
# _reset_phase3_heartbeat_state(). Module-level (rather than passed-in
# context) because the writer entry point is a free function called from
# main()'s while-loop, matching the queue writer / git writer pattern.
_PHASE3_HEARTBEAT_LAST_COMMITTED_STATE = {}   # machine -> state dict
_PHASE3_HEARTBEAT_LAST_TICK_UTC = {}          # machine -> (utc_str, monotonic_seen_at)
_PHASE3_HEARTBEAT_LAST_COMMIT_TS = 0.0        # monotonic time of last commit
_PHASE3_HEARTBEAT_INITIALIZED = False         # False until first commit lands


# ---- Writer-health snapshot (read by coordinator app.py /writer-health) ----
#
# Each writer updates last_tick_at on entry (post-READY-gate) and last_commit_at
# on successful push. main() wraps each writer call in try/except to record
# last_error. The snapshot is persisted to a JSON file shared with the
# coordinator process: sync_daemon + app.py run as separate systemd units, so
# in-process import is not available; a file is the simplest shared channel.
# Explorer probes via the coordinator's HTTP plane (chip 2026-05-31), which
# reads this file at request time.
_WRITER_HEALTH_LOCK = threading.Lock()
_WRITER_HEALTH = {
    "git_writer":       {"last_tick_at": None, "last_commit_at": None,
                         "last_error": None, "last_commit_sha": None,
                         "last_commit_subject": None,
                         "n_conflict_recoveries": 0,
                         "last_conflict_recovery_at": None},
    "queue_writer":     {"last_tick_at": None, "last_commit_at": None,
                         "last_error": None, "last_commit_sha": None,
                         "last_commit_subject": None,
                         "n_conflict_recoveries": 0,
                         "last_conflict_recovery_at": None},
    "heartbeat_writer": {"last_tick_at": None, "last_commit_at": None,
                         "last_error": None, "last_commit_sha": None,
                         "last_commit_subject": None,
                         "n_conflict_recoveries": 0,
                         "last_conflict_recovery_at": None},
}
WRITER_HEALTH_FILE = os.environ.get(
    "PHASE3_WRITER_HEALTH_FILE",
    os.path.join(os.path.dirname(__file__), "writer_health.json"))


def _utc_iso_now():
    # Timezone-aware UTC (matches db.utcnow()); the literal "Z" is hardcoded
    # and %z is not used, so the output string is byte-identical to the old
    # naive datetime.utcnow() form -- this is a deprecation cleanup only.
    return datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_iso_from_unix(ts):
    """Convert a unix timestamp (seconds since epoch) to a UTC ISO-8601
    stamp matching _utc_iso_now()'s format. Used by the writer-health
    bootstrap to render git author-time into the same shape the live tick
    path writes."""
    return datetime.datetime.fromtimestamp(
        int(ts), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record_writer_tick(name):
    with _WRITER_HEALTH_LOCK:
        rec = _WRITER_HEALTH.get(name)
        if rec is not None:
            rec["last_tick_at"] = _utc_iso_now()


def _record_writer_commit(name, repo_path=None, subject=None):
    """Stamp the last successful commit. Fetches HEAD sha via a single
    `git rev-parse HEAD` when repo_path is given; failure to fetch the sha
    is non-fatal (the tick still counts as a commit)."""
    now = _utc_iso_now()
    sha = None
    if repo_path:
        try:
            cp = subprocess.run(
                ["git", "-C", repo_path, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5)
            if cp.returncode == 0:
                sha = cp.stdout.strip()[:40] or None
        except (subprocess.TimeoutExpired, OSError):
            sha = None
    with _WRITER_HEALTH_LOCK:
        rec = _WRITER_HEALTH.get(name)
        if rec is not None:
            rec["last_commit_at"] = now
            if sha is not None:
                rec["last_commit_sha"] = sha
            if subject is not None:
                rec["last_commit_subject"] = subject[:200]
            # A successful commit clears any prior error -- if the writer
            # ticked again and pushed, whatever broke is no longer breaking.
            rec["last_error"] = None


def _record_writer_conflict_recovery(name):
    """Stamp a successful conflict-recovery self-heal: the queue writer hit a
    rebase conflict in _sync_to_origin and recovered by `git reset --hard
    origin/<branch>` + re-materialise-from-DB (PHASE3_QUEUE_CONFLICT_RECOVERY).
    Bumps a counter so /writer-health surfaces how often the self-heal fired --
    the canary observable that the wedge no longer requires manual operator
    intervention. A recovery is a successful event, so it does NOT set
    last_error."""
    now = _utc_iso_now()
    with _WRITER_HEALTH_LOCK:
        rec = _WRITER_HEALTH.get(name)
        if rec is not None:
            rec["n_conflict_recoveries"] = rec.get(
                "n_conflict_recoveries", 0) + 1
            rec["last_conflict_recovery_at"] = now


def _record_writer_error(name, exc):
    with _WRITER_HEALTH_LOCK:
        rec = _WRITER_HEALTH.get(name)
        if rec is not None:
            rec["last_error"] = {
                "at": _utc_iso_now(),
                "message": repr(exc)[:240],
            }


def _persist_writer_health():
    """Atomically write the writer-health snapshot to WRITER_HEALTH_FILE.
    Best-effort; failures do not break the writer tick."""
    try:
        with _WRITER_HEALTH_LOCK:
            snapshot = {
                "writers": {k: dict(v) for k, v in _WRITER_HEALTH.items()},
                "sync_daemon_pid": os.getpid(),
                "now_utc": _utc_iso_now(),
                # Surfaced on GET /writer-health for the explorer panel
                # (replaces SSH `ls coordinator-spool/pending | wc -l`).
                "spool_pending": manifest_spool.count_pending(),
            }
        text = json.dumps(snapshot, indent=2) + "\n"
        tmp = WRITER_HEALTH_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, WRITER_HEALTH_FILE)
    except OSError as exc:
        sys.stderr.write(
            "[writer-health] persist failed: %r\n" % exc)


def _reset_writer_health_state():
    """Test helper. Mirrors _reset_phase3_heartbeat_state()."""
    with _WRITER_HEALTH_LOCK:
        for rec in _WRITER_HEALTH.values():
            rec["last_tick_at"] = None
            rec["last_commit_at"] = None
            rec["last_error"] = None
            rec["last_commit_sha"] = None
            rec["last_commit_subject"] = None
            rec["n_conflict_recoveries"] = 0
            rec["last_conflict_recovery_at"] = None


def _phase3_writer_bootstrap_targets():
    """Return (writer_name, repo_path, branch, commit_prefix) tuples for
    each phase3 writer. Resolved lazily so tests can monkey-patch the
    module-level PHASE3_* constants before bootstrap fires."""
    return (
        ("git_writer",
         PHASE3_REE_ASSEMBLY, PHASE3_ASSEMBLY_BRANCH,
         _PHASE3_COMMIT_PREFIX),
        ("queue_writer",
         PHASE3_REE_V3, PHASE3_REE_V3_BRANCH,
         _PHASE3_QUEUE_COMMIT_PREFIX),
        ("heartbeat_writer",
         PHASE3_REE_ASSEMBLY, PHASE3_ASSEMBLY_BRANCH,
         _PHASE3_HEARTBEAT_COMMIT_PREFIX),
    )


def _bootstrap_writer_health_from_git():
    """Seed _WRITER_HEALTH's commit fields (sha / at / subject) from each
    writer's most recent matching commit on origin/<branch>, so a fresh
    sync_daemon process surfaces a meaningful history snapshot via the
    /writer-health endpoint before any writer has ticked.

    Failures must not crash the daemon: a git failure for any writer logs
    a one-line warning and leaves that writer's commit fields untouched
    (same as the pre-bootstrap state). The function NEVER touches
    last_tick_at -- tick-age is a this-process signal that must reflect
    reality, not git history. Idempotent: a second invocation with the
    same git state is a no-op because the timestamp-newer guard skips
    when the existing value is at or after the bootstrap value (matters
    if the function is ever called mid-lifecycle, e.g. signal-handler
    reload)."""
    for name, repo, branch, prefix in _phase3_writer_bootstrap_targets():
        try:
            cp = subprocess.run(
                ["git", "-C", repo, "log", "-1",
                 "--grep=^" + prefix,
                 "--pretty=%H %at %s",
                 "origin/" + branch],
                capture_output=True, text=True, timeout=10)
            if cp.returncode != 0:
                # No matching commit (or fetch missing) is not an error
                # -- many fresh deploys won't have any phase3 commits yet.
                continue
            line = cp.stdout.strip().splitlines()[0] if cp.stdout.strip() else ""
            if not line:
                continue
            parts = line.split(" ", 2)
            if len(parts) < 3:
                continue
            sha = parts[0][:40] or None
            try:
                commit_unix = int(parts[1])
            except (TypeError, ValueError):
                continue
            subject = parts[2]
            commit_at = _utc_iso_from_unix(commit_unix)
            with _WRITER_HEALTH_LOCK:
                rec = _WRITER_HEALTH.get(name)
                if rec is None:
                    continue
                existing_at = rec.get("last_commit_at")
                # Only fill nulls or values strictly older than the
                # bootstrap stamp. ISO-8601 fixed-width Zulu stamps
                # compare lexically.
                if existing_at is not None and existing_at >= commit_at:
                    continue
                rec["last_commit_at"] = commit_at
                rec["last_commit_sha"] = sha
                rec["last_commit_subject"] = subject[:200]
        except (subprocess.TimeoutExpired, OSError) as exc:
            sys.stderr.write(
                "[writer-health] bootstrap failed for %s: %r\n"
                % (name, exc))
        except Exception as exc:  # noqa: BLE001 -- never crash the daemon
            sys.stderr.write(
                "[writer-health] bootstrap failed for %s: %r\n"
                % (name, exc))


def _phase3_heartbeat_now():
    """Monotonic clock indirection so tests can drive virtual time."""
    return time.monotonic()


def _reset_phase3_heartbeat_state():
    """Test helper: clear all module-level writer state for a fresh
    fixture. Production code should never call this."""
    global _PHASE3_HEARTBEAT_LAST_COMMIT_TS, _PHASE3_HEARTBEAT_INITIALIZED
    _PHASE3_HEARTBEAT_LAST_COMMITTED_STATE.clear()
    _PHASE3_HEARTBEAT_LAST_TICK_UTC.clear()
    _PHASE3_HEARTBEAT_LAST_COMMIT_TS = 0.0
    _PHASE3_HEARTBEAT_INITIALIZED = False


def _extract_heartbeat_machine_state(status_doc, heartbeat_doc):
    """Pull the minimum set of state-change-relevant fields from the
    (status, heartbeat) payload pair. Always returns a dict; missing
    fields default to None / False so absent payloads do not falsely
    register as state changes."""
    state = {
        "queue_id": None,
        "idle": False,
        "runner_pid": None,
        "last_completed_queue_id": None,
        "last_completed_result": None,
        "last_tick_utc": None,
        "silent": False,
    }
    if isinstance(status_doc, dict):
        current = status_doc.get("current") or {}
        if isinstance(current, dict):
            state["queue_id"] = current.get("queue_id")
        state["idle"] = bool(status_doc.get("idle"))
        state["runner_pid"] = status_doc.get("runner_pid")
        completed = status_doc.get("completed") or []
        if isinstance(completed, list) and completed:
            last = completed[-1]
            if isinstance(last, dict):
                state["last_completed_queue_id"] = last.get("queue_id")
                state["last_completed_result"] = last.get("result")
    if isinstance(heartbeat_doc, dict):
        state["last_tick_utc"] = heartbeat_doc.get("last_tick_utc")
    return state


def _phase3_heartbeat_is_silent(machine, current_utc, now, stale_after):
    """Update the per-machine last_tick_utc cache and return True iff the
    machine's last_tick_utc has been unchanged for >= stale_after seconds.
    First observation of a (machine, utc) pair seeds the timer without
    declaring silent."""
    prev = _PHASE3_HEARTBEAT_LAST_TICK_UTC.get(machine)
    if prev is None or prev[0] != current_utc:
        _PHASE3_HEARTBEAT_LAST_TICK_UTC[machine] = (current_utc, now)
        return False
    return (now - prev[1]) >= stale_after


def _phase3_heartbeat_compute_changes(last_committed, current_states):
    """Build the list of state-change events relative to the last
    committed state. Each event is a dict naming what changed; the
    commit-message builder turns the list into a subject line.

    Detected kinds: added, started, finished, released, switched, restart,
    idle_changed, went_silent, came_back.
    """
    changes = []
    for machine, curr in current_states.items():
        prev = last_committed.get(machine)
        if prev is None:
            changes.append({"machine": machine, "kind": "added"})
            continue
        prev_silent = bool(prev.get("silent"))
        curr_silent = bool(curr.get("silent"))
        # queue_id transitions
        if curr["queue_id"] != prev["queue_id"]:
            if prev["queue_id"] is None and curr["queue_id"] is not None:
                changes.append({"machine": machine, "kind": "started",
                                "queue_id": curr["queue_id"]})
            elif prev["queue_id"] is not None and curr["queue_id"] is None:
                # Attribute a result from the completed list when the last
                # completed entry matches the queue_id that was just running
                # AND is fresher than what we had at last-commit time.
                if (curr.get("last_completed_queue_id") == prev["queue_id"]
                        and curr.get("last_completed_queue_id") !=
                            prev.get("last_completed_queue_id")):
                    changes.append({
                        "machine": machine, "kind": "finished",
                        "queue_id": prev["queue_id"],
                        "result": curr.get("last_completed_result") or "DONE",
                    })
                else:
                    changes.append({"machine": machine, "kind": "released",
                                    "queue_id": prev["queue_id"]})
            else:
                changes.append({
                    "machine": machine, "kind": "switched",
                    "prev_queue_id": prev["queue_id"],
                    "queue_id": curr["queue_id"],
                })
        elif curr["idle"] != prev["idle"]:
            changes.append({"machine": machine, "kind": "idle_changed",
                            "now_idle": curr["idle"]})
        # Restart: runner_pid changed AND both values known.
        if (curr["runner_pid"] != prev["runner_pid"]
                and curr["runner_pid"] is not None
                and prev["runner_pid"] is not None):
            changes.append({"machine": machine, "kind": "restart",
                            "pid": curr["runner_pid"]})
        # Silent transitions (one event per crossing, not every tick).
        if curr_silent and not prev_silent:
            changes.append({"machine": machine, "kind": "went_silent"})
        elif prev_silent and not curr_silent:
            changes.append({"machine": machine, "kind": "came_back"})
    return changes


def _phase3_heartbeat_phrase(ev):
    """Render a single state-change event as a human-readable phrase."""
    m = ev["machine"]
    k = ev["kind"]
    if k == "started":
        return "%s -> %s started" % (m, ev["queue_id"])
    if k == "finished":
        return "%s ran %s -> %s" % (m, ev["queue_id"], ev["result"])
    if k == "released":
        return "%s released %s" % (m, ev["queue_id"])
    if k == "switched":
        return "%s %s -> %s" % (m, ev["prev_queue_id"], ev["queue_id"])
    if k == "restart":
        return "%s restart pid=%s" % (m, ev["pid"])
    if k == "idle_changed":
        return "%s idle=%s" % (m, "True" if ev["now_idle"] else "False")
    if k == "added":
        return "%s added" % m
    if k == "went_silent":
        return "%s went silent" % m
    if k == "came_back":
        return "%s came back" % m
    return "%s ?%s" % (m, k)


def _phase3_heartbeat_commit_message(changes, n_machines_active):
    """Build the commit subject. Empty changes -> liveness fallback."""
    prefix = _PHASE3_HEARTBEAT_COMMIT_PREFIX
    if not changes:
        return "%sliveness tick, %d machine(s) active" % (
            prefix, n_machines_active)
    if len(changes) == 1:
        return "%s%s" % (prefix, _phase3_heartbeat_phrase(changes[0]))
    if len(changes) <= 3:
        return "%s%s" % (
            prefix,
            ", ".join(_phase3_heartbeat_phrase(c) for c in changes))
    head = ", ".join(_phase3_heartbeat_phrase(c) for c in changes[:2])
    return "%s%d state changes: %s, ..." % (prefix, len(changes), head)


def _fsync_dir(path):
    """MED-B: after `os.replace(tmp, target)` the rename is not crash-
    durable until the containing directory's metadata journal entry is
    flushed. Open the dir read-only and fsync its descriptor. Linux:
    standard pattern; macOS: the open() succeeds, the fsync is a no-op
    but harmless. Windows: O_DIRECTORY is unsupported, so we swallow
    the EINVAL/ENOTDIR (the hub deploy target is Linux; this guard is
    for the smoke harness which runs on macOS / dev machines).
    Best-effort: any failure is swallowed -- the rename itself already
    succeeded, the worst case is a non-durable directory entry across a
    power loss, and the writer's spool retains the source-of-truth bytes
    until committed_at is set."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _revert_target_to_head(repo, relpath, target):
    """Best-effort rollback after a `git add` failure.

    The atomic write already replaced the working-tree file with the new
    content. If `git add` then fails (out of disk, EAGAIN, permission
    glitch on the .git/index lockfile, ...) the working tree is dirty and
    blocks the next tick's clean-tree check, stalling the writer until
    operator intervention.

    First try `git checkout HEAD -- <relpath>` to restore the pre-write
    content (the common case: we were updating an existing tracked file).
    If that fails (e.g. brand-new file with no HEAD blob to restore), fall
    back to `os.unlink(target)` to leave the working tree clean.

    Both steps are best-effort: if BOTH fail, the leak is the same as
    pre-fix behaviour. Never raises."""
    try:
        result = _git(
            repo, "checkout", "HEAD", "--", relpath,
            check=False, timeout=10)
        if result.returncode == 0:
            return
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    try:
        os.unlink(target)
    except OSError:
        pass


def _git(repo, *args, timeout=30, check=True):
    """Run git in repo. Returns CompletedProcess. capture_output=True so
    nothing leaks to stdout/stderr unless we choose to log it."""
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, timeout=timeout, check=check,
    )


def _porcelain_relpath(line):
    """Parse one `git status --porcelain` line into a repo-relative path."""
    # Short format is XY<space>PATH; X may be a space when unstaged-only.
    rest = line[2:].lstrip() if len(line) >= 2 else line.strip()
    if " -> " in rest:
        rest = rest.split(" -> ", 1)[1]
    return rest.replace("\\", "/")


def _rel_path_under_heartbeat_writer_guards(relpath):
    """True when relpath is owned by phase3_heartbeat_writer only."""
    for prefix in _WRITER_GUARD_PATHS["phase3_heartbeat_writer"]:
        p = prefix.rstrip("/")
        if relpath == p or relpath.startswith(prefix):
            return True
    return False


def _maybe_revert_exclusive_telemetry_dirt(repo, log_prefix):
    """Recover hub checkout when ONLY heartbeat-writer paths are dirty.

    On the hub VM, runner_heartbeats/ and runner_status/ on disk are
    materialised exclusively by phase3_heartbeat_writer from the coordinator
    DB (runners POST /heartbeat; hub runner must not write these files --
    PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1). A partial writer tick
    (atomic write + failed git add/commit) can still leave telemetry dirt
    that blocks phase3_git_writer and stalls result spool commits.

    When every porcelain entry is under the heartbeat writer's guard
    prefixes, revert those paths to HEAD (or remove untracked telemetry
    files) and return True. Any other path (manifests, planning docs,
    ...) leaves the tree unchanged and returns False. Never raises.
    """
    try:
        out = _git(repo, "status", "--porcelain", check=True).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            OSError) as exc:
        sys.stderr.write(
            "[%s] telemetry dirt recovery: git status failed: %r\n" % (
                log_prefix, exc))
        return False
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    if not lines:
        return True
    entries = []
    for line in lines:
        relpath = _porcelain_relpath(line)
        if not _rel_path_under_heartbeat_writer_guards(relpath):
            return False
        entries.append((line[:2], relpath))
    reverted = []
    for xy, relpath in entries:
        target = os.path.join(repo, relpath)
        if xy == "??":
            try:
                os.unlink(target)
                reverted.append(relpath)
            except OSError:
                return False
            continue
        try:
            result = _git(
                repo, "restore", "--source=HEAD", "--staged",
                "--worktree", relpath, check=False, timeout=15)
            if result.returncode != 0:
                _revert_target_to_head(repo, relpath, target)
            reverted.append(relpath)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            _revert_target_to_head(repo, relpath, target)
            reverted.append(relpath)
    if reverted:
        sys.stderr.write(
            "[%s] auto-reverted exclusive telemetry dirt (%d path(s)): %s\n"
            % (log_prefix, len(reverted), ", ".join(reverted[:5])
               + (" ..." if len(reverted) > 5 else "")))
    clean, _reason = _hub_working_tree_clean(repo)
    return clean


def _hub_working_tree_clean(repo):
    """Phase 3 explicitly retires autostash, so the writer refuses to
    operate on a dirty tree -- any uncommitted edit on the hub checkout
    must be resolved by a human, not silently stashed. Returns (clean,
    reason). reason is a one-line string when clean=False."""
    try:
        out = _git(repo, "status", "--porcelain", check=True).stdout
    except subprocess.CalledProcessError as exc:
        return (False, "git status failed: %r" % exc)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (False, "git status error: %r" % exc)
    if out.strip():
        first = out.strip().splitlines()[0]
        return (False, "dirty working tree: %s" % first[:120])
    return (True, "")


def _restore_derived_path(repo, relpath, log_prefix):
    """Restore a WRITER-DERIVED path to HEAD (index + working tree).

    Only ever call this on a path the writer materialises from the
    coordinator DB (today: experiment_queue.json). Such a path is a
    derived view, never operator input, so discarding an uncommitted
    edit is lossless -- the next tick re-materialises it from the DB.

    Why this exists: `_hub_working_tree_clean` treats ANY porcelain
    output as dirty, so a writer that leaves its own half-finished edit
    in the tree -- staged OR unstaged -- trips its own precondition on
    every subsequent tick and deadlocks itself permanently. Confirmed
    2026-07-18: a staged experiment_queue.json (git add succeeded, the
    commit did not) wedged phase3_queue_writer for 5h31m, silently, with
    last_error=None and the tick loop still running.

    Best-effort: a failure here is logged and the tick still reports its
    original outcome. `git checkout HEAD -- <path>` resets index AND
    working tree for the path in one call.
    """
    try:
        res = _git(repo, "checkout", "HEAD", "--", relpath,
                   check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write(
            "%s WARN could not restore %s after a failed tick (%r); the "
            "next tick may refuse on a dirty tree.\n"
            % (log_prefix, relpath, exc))
        return False
    if res.returncode != 0:
        sys.stderr.write(
            "%s WARN could not restore %s after a failed tick (%s); the "
            "next tick may refuse on a dirty tree.\n"
            % (log_prefix, relpath, res.stderr.strip()[:240]))
        return False
    return True


def _hub_working_tree_clean_for_writer(repo, log_prefix):
    """Clean-tree check with one-shot telemetry-only auto-recovery."""
    clean, reason = _hub_working_tree_clean(repo)
    if clean:
        return (True, "")
    if _maybe_revert_exclusive_telemetry_dirt(repo, log_prefix):
        return (True, "")
    return (False, reason)


# Every writer-authored commit's message starts with one of these prefixes;
# see commit_msg construction inside phase3_git_writer (results),
# phase3_queue_writer (queue snapshot), and phase3_heartbeat_writer
# (telemetry). The foreign-commit check uses prefix membership to gate
# pushes -- the writer refuses to publish any commit no phase3 writer
# authored. Each writer has its own prefix for log readability + audit
# (`git log --grep=phase3-heartbeats:` answers "what did the heartbeat
# writer do today?").
_PHASE3_COMMIT_PREFIX = "phase3: "
_PHASE3_QUEUE_COMMIT_PREFIX = "phase3-queue: "
_PHASE3_HEARTBEAT_COMMIT_PREFIX = "phase3-heartbeats: "

# The foreign-commit check accepts any of these prefixes -- the result
# writer and heartbeat writer share REE_assembly, so each must tolerate
# the other's unpushed commit (which can happen after a transient push
# failure). Treating sibling-writer commits as "foreign" caused a permanent
# deadlock: each writer rejected the other's leftover commit on every
# subsequent tick. ADDING A NEW PHASE3 WRITER? Add its prefix here, or the
# deadlock comes back.
_PHASE3_WRITER_PREFIXES = (
    _PHASE3_COMMIT_PREFIX,
    _PHASE3_QUEUE_COMMIT_PREFIX,
    _PHASE3_HEARTBEAT_COMMIT_PREFIX,
)


# Per-writer guard paths (repo-relative). Mirror of the runner-side
# _active_claim_on_evidence_dir pattern in runner_remote_control.py, but
# path-prefix matched (more surgical than substring) and scoped per
# writer so a claim on evidence/planning/ does not pause the heartbeat
# writer that only touches evidence/experiments/runner_*.
#
# The repo prefix layer is how we translate TASK_CLAIMS.json resource
# strings (which use REE_Working-rooted paths like
# "REE_assembly/evidence/experiments/runner_status.json") into the
# repo-relative paths each writer actually touches.
_WRITER_GUARD_REPO_PREFIXES = {
    "phase3_git_writer":        ("REE_assembly/",),
    "phase3_queue_writer":      ("ree-v3/",),
    "phase3_heartbeat_writer":  ("REE_assembly/",),
}
_WRITER_GUARD_PATHS = {
    "phase3_git_writer":        ("evidence/experiments/",),
    "phase3_queue_writer":      ("experiment_queue.json",),
    "phase3_heartbeat_writer":  (
        "evidence/experiments/runner_heartbeats/",
        "evidence/experiments/runner_status/",
        "evidence/experiments/runner_status.json",
    ),
}


def _active_claim_blocks_writer(writer_name, ree_working_path):
    """Best-effort guard. True iff REE_Working/TASK_CLAIMS.json holds an
    active claim whose `resources` list matches this writer's write-path
    prefixes (per `_WRITER_GUARD_PATHS`).

    Mirrors runner_remote_control._active_claim_on_evidence_dir but
    (a) path-prefix matched, so the guard is per-writer rather than
    "any evidence/ touch", and (b) scoped to the umbrella TASK_CLAIMS
    file at REE_Working root (ree_working_path).

    Returns False on every error path -- the guard never blocks the
    writer from running because the guard itself misbehaved. A stale
    TASK_CLAIMS snapshot (the hub may not pull the umbrella on every
    tick) just means the guard activates one tick late, which is fine:
    the session that opens the claim can wait a tick to confirm the
    writer has paused before editing.

    `ree_working_path` is the directory containing TASK_CLAIMS.json --
    typically `os.path.dirname(PHASE3_REE_ASSEMBLY)` on the hub.
    """
    try:
        repo_prefixes = _WRITER_GUARD_REPO_PREFIXES[writer_name]
        write_prefixes = _WRITER_GUARD_PATHS[writer_name]
    except KeyError:
        return False
    try:
        claims_path = os.path.join(ree_working_path, "TASK_CLAIMS.json")
        if not os.path.exists(claims_path):
            return False
        with open(claims_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("claims", []):
            if entry.get("status") != "active":
                continue
            for res in entry.get("resources", []):
                if not isinstance(res, str):
                    continue
                for repo_prefix in repo_prefixes:
                    if not res.startswith(repo_prefix):
                        continue
                    rel = res[len(repo_prefix):]
                    if any(rel.startswith(wp) for wp in write_prefixes):
                        return True
        return False
    except Exception:
        return False


def _check_ahead_writer_authored(repo, branch,
                                 prefixes=_PHASE3_WRITER_PREFIXES):
    """Inspect every commit reachable from HEAD but not origin/<branch>.

    Returns (ok, detail). ok=True iff every such commit's subject starts
    with one of `prefixes` (i.e. was authored by SOME phase3 writer).
    detail is a list of up to three foreign subject lines when ok=False,
    or a single-element list with the git-log failure message when the
    log itself errored.

    Caller is expected to have refreshed origin/<branch> via `git fetch`
    immediately before this call -- a stale ref would over-count ahead
    commits (false positives) for the foreign check.

    `prefixes` defaults to the full writer-authored set so that two
    writers sharing a repo (result + heartbeat both push to REE_assembly)
    do not treat each other's unpushed commits as foreign. Pass a
    narrower tuple only when the caller deliberately wants to scope
    "authored by THIS writer" -- e.g. an audit tool.
    """
    log = _git(
        repo, "log", "--format=%s",
        "origin/" + branch + "..HEAD",
        check=False, timeout=10)
    if log.returncode != 0:
        return (False, ["<git-log-failed: %s>"
                        % log.stderr.strip()[:120]])
    foreign = [
        line for line in log.stdout.splitlines()
        if line and not any(line.startswith(p) for p in prefixes)
    ]
    return (not foreign, foreign[:3])


def _sync_to_origin(repo, branch, log_prefix, auto_reset_on_conflict=None,
                    writer_name=None):
    """Fetch origin/<branch>, then if local HEAD is BEHIND origin, rebase
    the writer-authored ahead commits on top of refreshed origin. Returns
    (ok, reason); ok=False means the caller must refuse the tick.

    auto_reset_on_conflict selects the rebase-CONFLICT policy:
      None  (default): fall back to the legacy
             PHASE3_AUTO_RESET_ON_REBASE_CONFLICT env check -- the
             behaviour the result + heartbeat writers keep (refuse on
             conflict unless the env is set globally).
      True:  recover losslessly via `git reset --hard origin/<branch>`
             (the queue writer passes PHASE3_QUEUE_CONFLICT_RECOVERY here;
             safe only when the caller can re-materialise the dropped
             commit from an authoritative source, e.g. the DB-backed
             queue snapshot).
      False: refuse on conflict (explicit opt-out, ignores the env).
    writer_name, when given, names the _WRITER_HEALTH record to stamp on a
    successful self-heal (the /writer-health canary observable).

    Why this helper exists:
    Between two writer ticks, an unrelated commit can land on origin --
    a Claude session closing a TASK_CLAIMS entry, a lit-pull synthesis
    push, a runner claim during the legacy-push transition window. The
    next writer tick's local HEAD is then both AHEAD (its own
    retained-for-retry phase3-* commits) AND BEHIND (origin has new
    work). Without this absorb step `git push` rejects non-FF on every
    tick and the spool grows forever. That is exactly the failure mode
    we observed live on 2026-05-28 ~16:25Z..17:36Z, where REE_assembly
    sat ahead 7 / behind 2 and ree-v3 ahead 4 / behind 6 for ~70
    minutes while phase3_git_writer + phase3_queue_writer + phase3_
    heartbeat_writer all logged push REJECTED on a fixed cadence.

    Why this is safe -- and is NOT the `git pull --rebase --autostash`
    pattern Phase 3 retires:
      1. The caller has already passed `_hub_working_tree_clean`, so
         there is nothing to autostash. `git rebase` without
         --autostash refuses on a dirty tree, double-enforcing the
         clean-tree precondition.
      2. `_check_ahead_writer_authored` ensures every commit being
         rebased was authored by some phase3 writer. A foreign commit
         (operator hand-edit, accidental tooling commit) refuses the
         tick instead of getting rebased under the writer's authority.
      3. On rebase failure (content conflict against an origin commit
         that touched the same files), we `git rebase --abort` and
         refuse. The tree is restored. The next tick re-materialises
         from DB / replays from spool and tries again. We never silently
         resolve conflicts or drop work.

    Idle no-op path: when behind_count == 0 (we're up to date with
    origin, or ahead-only), the helper returns ok immediately. The
    existing fetch + ahead-of-origin guard in the caller's push block
    handles the ahead-only case (a rejected push followed by a
    no-operator-action tick is the case the 2026-05-27 HIGH-1 guard
    already covered).
    """
    fetched = _git(
        repo, "fetch", "--quiet", "origin", branch,
        check=False, timeout=30)
    if fetched.returncode != 0:
        return (False, "fetch origin %s failed: %s" % (
            branch, fetched.stderr.strip()[:240]))
    behind = _git(
        repo, "rev-list", "--count", "HEAD..origin/" + branch,
        check=False, timeout=10)
    if behind.returncode != 0:
        return (False, "rev-list behind-count failed: %s" % (
            behind.stderr.strip()[:240]))
    behind_count = behind.stdout.strip()
    if not behind_count or behind_count == "0":
        return (True, "")
    # Behind > 0. Validate ahead commits are writer-authored BEFORE
    # touching HEAD -- a foreign commit ahead means the operator has
    # work that the writer must not rebase under its own authority.
    ok, foreign = _check_ahead_writer_authored(repo, branch)
    if not ok:
        return (False,
                "behind origin/%s by %s commit(s) AND %d foreign "
                "commit(s) ahead (%s); operator must resolve before the "
                "writer can rebase" % (
                    branch, behind_count, len(foreign), foreign))
    # Record the writer-authored ahead count BEFORE the rebase so the
    # success log line states what was rebased rather than what was
    # absorbed. (behind_count is the count of new origin commits we're
    # catching up to; ahead_count is the count of writer-authored
    # commits being replayed on top.)
    ahead = _git(
        repo, "rev-list", "--count", "origin/" + branch + "..HEAD",
        check=False, timeout=10)
    ahead_count = ahead.stdout.strip() if ahead.returncode == 0 else "?"
    rebased = _git(
        repo, "rebase", "origin/" + branch,
        check=False, timeout=60)
    if rebased.returncode != 0:
        # Conflict against an origin commit that touched the same path
        # the writer's commit modified. Abort cleanly. Then either
        # auto-recover (opt-in via env) or surface and refuse.
        _git(repo, "rebase", "--abort", check=False, timeout=10)
        # Opt-in self-heal: PHASE3_AUTO_RESET_ON_REBASE_CONFLICT=1 lets
        # the writer drop its own writer-authored stale ahead commits
        # via `git reset --hard origin/<branch>` and recover on the next
        # tick by re-materialising the same content from DB/spool. This
        # is the behaviour the abort message has always recommended; the
        # env gates it because trust-in-regeneration is a per-deployment
        # operator decision, not a structural property of the writer.
        # Safe preconditions already satisfied at this call site:
        #   - clean tree (caller passed _hub_working_tree_clean)
        #   - all ahead commits writer-authored (foreign-check above)
        # so the hard-reset only drops content this writer can re-emit.
        # Motivating incident: 2026-05-29 06:06-06:49 UTC, 42 consecutive
        # rebase-conflict aborts on ree-v3 main against operator commits
        # landing every minute; writer wedged until the operator stream
        # quieted. With auto-reset the wedge is at-most-one tick.
        #
        # Policy resolution (PHASE3_QUEUE_CONFLICT_RECOVERY, 2026-06-03):
        # an explicit auto_reset_on_conflict argument overrides the legacy
        # env. The queue writer passes True (DB-authoritative; reset +
        # re-materialise is lossless); result/heartbeat writers pass None
        # and keep the env-fallback (refuse unless globally enabled).
        if auto_reset_on_conflict is not None:
            auto_reset = bool(auto_reset_on_conflict)
        else:
            auto_reset = os.environ.get(
                "PHASE3_AUTO_RESET_ON_REBASE_CONFLICT", "0").strip() in (
                    "1", "true", "TRUE", "yes", "YES")
        if auto_reset:
            reset = _git(
                repo, "reset", "--hard", "origin/" + branch,
                check=False, timeout=30)
            if reset.returncode != 0:
                return (False,
                        "rebase onto origin/%s failed (%s); "
                        "conflict-recovery reset requested but "
                        "`git reset --hard origin/%s` also failed (%s); "
                        "tree may be in an unexpected state, operator "
                        "must investigate." % (
                            branch, rebased.stderr.strip()[:240],
                            branch, reset.stderr.strip()[:240]))
            if writer_name:
                _record_writer_conflict_recovery(writer_name)
            sys.stderr.write(
                "%s rebase onto origin/%s conflicted; conflict-recovery "
                "dropped %s writer-authored commit(s) via `git reset "
                "--hard origin/%s`; this tick re-materialises from DB / "
                "spool (self-heal, no operator intervention). (rebase "
                "stderr: %s)\n" % (
                    log_prefix, branch, ahead_count, branch,
                    rebased.stderr.strip()[:240]))
            # Successful reset == we are now at origin/<branch>. Return
            # ok so the caller proceeds; the materialise step then
            # re-emits the dropped content.
            return (True, "")
        return (False,
                "rebase onto origin/%s failed (%s); rebase aborted, "
                "spool / DB state intact. Operator may need to resolve "
                "conflicts by hand or `git reset --hard origin/%s` to "
                "drop the stale writer commits (they will be "
                "regenerated on the next tick). Set "
                "PHASE3_AUTO_RESET_ON_REBASE_CONFLICT=1 to make the "
                "writer perform the same reset automatically on "
                "conflict." % (
                    branch, rebased.stderr.strip()[:240], branch))
    sys.stdout.write(
        "%s absorbed %s origin commit(s) by rebasing %s "
        "writer-authored commit(s) onto refreshed origin/%s\n" % (
            log_prefix, behind_count, ahead_count, branch))
    return (True, "")


def _load_runpack_builder(asm):
    """Lazy cross-checkout import of sync_v3_results.runpack_for_flat from the
    hub's REE_assembly scripts dir. Returns the callable, or None on any
    failure (run-pack materialisation is strictly best-effort -- the flat
    manifest commit must NEVER depend on it). The import is lazy + cached via
    sys.modules so the cross-repo path insert happens at most once per process,
    and only when PHASE3_MATERIALIZE_RUNPACK is on."""
    try:
        scripts_dir = os.path.join(
            asm, "evidence", "experiments", "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import sync_v3_results
        return sync_v3_results.runpack_for_flat
    except Exception as exc:  # noqa: BLE001 -- degrade, never crash the tick
        sys.stderr.write(
            "[phase3] runpack materialise: import of "
            "sync_v3_results.runpack_for_flat from %s failed: %r; "
            "committing flat manifest(s) only\n" % (asm, exc))
        return None


def _materialize_runpacks(asm, staged, log_prefix="[phase3]"):
    """For each just-staged flat manifest, ALSO write + git-add the canonical
    runs/<run_id>/{manifest.json,metrics.json,summary.md} pack so the result is
    immediately scoreable on origin (build_experiment_indexes consumes runs/
    packs, not flat manifests). Returns the number of run-pack FILES staged.

    Reuses sync_v3_results.build_runpack_docs (via runpack_for_flat) for the
    exact field mapping, so the pack is byte-identical to one the local
    governance.sh converter would produce.

    SAFETY / reversibility:
      - Best-effort: any per-run failure logs a WARN and is skipped; the flat
        manifest commit (already staged by the caller) is never blocked.
      - Skip-if-pack-exists: if runs/<run_id>/manifest.json already exists, the
        pack was synced by the runner -- do NOT clobber it. This makes the
        writer a pure back-fill for the missing-pack gap.
      - Atomic write (tmp + os.replace) mirrors the flat-manifest write site;
        a git-add failure reverts the working-tree write so the next tick's
        clean-tree check still passes.
    """
    builder = _load_runpack_builder(asm)
    if builder is None:
        return 0
    evidence_dir = os.path.join(asm, "evidence", "experiments")
    n_files = 0
    for run_id, relpath in staged:
        flat_target = os.path.join(asm, relpath)
        try:
            result = builder(flat_target, evidence_dir)
        except Exception as exc:  # noqa: BLE001 -- per-run isolation
            sys.stderr.write(
                "%s runpack materialise: build failed for %s: %r\n" % (
                    log_prefix, run_id, exc))
            continue
        if result is None:
            # Not an eligible V3 evidence/diagnostic flat manifest (the same
            # gate the converter applies); nothing to back-fill.
            continue
        run_dir, manifest_doc, metrics_doc, summary = result
        run_dir = str(run_dir)
        if os.path.exists(os.path.join(run_dir, "manifest.json")):
            # Runner already synced the pack -- never overwrite it.
            continue
        docs = {
            "manifest.json": json.dumps(manifest_doc, indent=2) + "\n",
            "metrics.json": json.dumps(metrics_doc, indent=2) + "\n",
            "summary.md": summary,
        }
        try:
            os.makedirs(run_dir, exist_ok=True)
        except OSError as exc:
            sys.stderr.write(
                "%s runpack materialise: mkdir failed for %s: %r\n" % (
                    log_prefix, run_dir, exc))
            continue
        for fname, content in docs.items():
            target = os.path.join(run_dir, fname)
            tmp_target = target + ".phase3.tmp"
            target_replaced = False
            try:
                with open(tmp_target, "w", encoding="utf-8") as fh:
                    fh.write(content)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_target, target)
                target_replaced = True
                _fsync_dir(run_dir)
            except OSError as exc:
                sys.stderr.write(
                    "%s runpack materialise: write failed %s: %r\n" % (
                        log_prefix, target, exc))
                try:
                    os.unlink(tmp_target)
                except OSError:
                    pass
                continue
            rel = os.path.relpath(target, asm)
            try:
                _git(asm, "add", rel, timeout=15, check=True)
                n_files += 1
            except (subprocess.CalledProcessError,
                    subprocess.TimeoutExpired) as exc:
                sys.stderr.write(
                    "%s runpack materialise: git add failed %s: %r. "
                    "Reverting working-tree write.\n" % (
                        log_prefix, rel, exc))
                if target_replaced:
                    _revert_target_to_head(asm, rel, target)
    return n_files


def _materialize_sidefiles(asm, run_ids, log_prefix="[phase3]"):
    """For each run_id, write + git-add its spooled COMPANION side-files under
    REE_assembly/<relpath>, so they land in the same commit as the run's
    manifest. Returns (n_files_staged, processed_run_ids) where processed
    run_ids are those that had at least one companion successfully staged (and
    whose companion spool dir should be dropped after the batch commits).

    SAFETY / reversibility mirrors _materialize_runpacks:
      - Best-effort: any per-file failure logs a WARN and is skipped; the
        manifest commit (already staged by the caller) is never blocked.
      - Each destination relpath was validated under evidence/experiments/ at
        spool time AND is re-validated by list_sidefiles_for_run.
      - Atomic write (tmp + os.replace); a git-add failure reverts the
        working-tree write so the next tick's clean-tree check still passes.
      - Idempotent: re-committing a companion already on origin is a no-diff
        git-add (harmless); the diff-cached short-circuit handles it.
    """
    n_files = 0
    processed = []
    for run_id in run_ids:
        entries = manifest_spool.list_sidefiles_for_run(run_id)
        if not entries:
            continue
        any_staged = False
        for relpath, bin_path in entries:
            try:
                with open(bin_path, "rb") as fh:
                    raw = fh.read()
            except OSError as exc:
                sys.stderr.write(
                    "%s sidefile materialise: read failed %s: %r\n" % (
                        log_prefix, bin_path, exc))
                continue
            target = os.path.join(asm, relpath)
            target_dir = os.path.dirname(target)
            tmp_target = target + ".phase3sf.tmp"
            target_replaced = False
            try:
                os.makedirs(target_dir, exist_ok=True)
                with open(tmp_target, "wb") as fh:
                    fh.write(raw)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_target, target)
                target_replaced = True
                _fsync_dir(target_dir)
            except OSError as exc:
                sys.stderr.write(
                    "%s sidefile materialise: write failed %s -> %s: %r\n" % (
                        log_prefix, run_id, relpath, exc))
                try:
                    os.unlink(tmp_target)
                except OSError:
                    pass
                continue
            try:
                _git(asm, "add", relpath, timeout=15, check=True)
                n_files += 1
                any_staged = True
            except (subprocess.CalledProcessError,
                    subprocess.TimeoutExpired) as exc:
                sys.stderr.write(
                    "%s sidefile materialise: git add failed %s -> %s: %r. "
                    "Reverting working-tree write.\n" % (
                        log_prefix, run_id, relpath, exc))
                if target_replaced:
                    _revert_target_to_head(asm, relpath, target)
        if any_staged:
            processed.append(run_id)
    return n_files, processed


def phase3_git_writer(
    conn,
    queue_path,
    *,
    ree_assembly_path=None,
    dry_run=False,
):
    """Sole git writer tick (Phase 3).

    Reads pending manifests from the filesystem spool, writes them under
    REE_assembly/evidence/experiments/, commits, and pushes. Marks
    `results.committed_at` only after a successful push so a crash midway
    leaves the manifest available for the next tick (idempotent retry).

    SAFETY:
      - PHASE3_GIT_WRITER_READY is checked at every entry. While False the
        writer logs intent and returns False; main()'s authoritative-mode
        loop then refuses to advance, so no git writes can happen even if
        the operator flips SYNC_MODE prematurely.
      - Refuses to operate on a dirty REE_assembly working tree (the whole
        point of Phase 3 is to retire the autostash war; a human must
        clean up unexpected dirt).
      - Never calls `git pull --rebase --autostash`. A non-fast-forward
        push fails the tick loudly and leaves the spool intact for retry.
      - Before marking results.committed_at on a "no new diff" tick, runs
        `git fetch --quiet origin <branch>` then checks
        `git rev-list --count origin/<branch>..HEAD`. The diff-cached
        short-circuit only fires when ahead==0 (bytes truly on origin);
        ahead>0 forces a push of the unpushed local commit first, or
        refuses the tick if that push is still rejected. The pre-fetch
        is load-bearing: rev-list reads the local remote-tracking ref,
        and a stale ref would let ahead==0 lie (writer would drain the
        spool against a stale view of origin while bytes never reach
        the remote). Fetch failure refuses the tick rather than
        proceeding with a possibly-stale ref. Without this guard a
        rejected-push tick followed by a no-operator-action tick would
        silently drain the spool without origin ever receiving the
        bytes.
      - Before EVERY push, verifies that every commit in
        origin/<branch>..HEAD was writer-authored (subject starts with
        the `phase3: ` prefix). If any foreign commit is found (operator
        hand-edit on the hub, accidental tooling commit, force-push
        residue), the writer refuses to push and retains the spool. The
        post-cutover invariant "all REE_assembly commits attributable to
        sync_daemon" depends on this check; without it, an unpushed
        operator commit between origin/<branch> and HEAD would be
        published silently by the writer.
      - Batched to PHASE3_BATCH_SIZE manifests per tick; the rest land in
        subsequent ticks.

    Returns True only when a full tick completed (or dry_run simulated).
    Returns False when the writer stub guard is active, the tree is dirty,
    or the spool is empty (nothing to do -- the daemon is idle).

    Out-of-scope for this sketch (deferred TODO):
      - Step 5: snapshot completed queue items from `experiments` table
        into the ree-v3 checkout's experiment_queue.json and push.
      - Step 6: write derived runner_heartbeats/*.json + runner_status/
        from the heartbeats table (replaces the per-runner git heartbeat
        push that runner_remote_control.push_heartbeat does today).
      Both extensions live in this same function once the results path is
      validated under a test fleet.
    """
    asm = ree_assembly_path or PHASE3_REE_ASSEMBLY
    if not PHASE3_GIT_WRITER_READY:
        sys.stderr.write(
            "[phase3] git writer stub (PHASE3_GIT_WRITER_READY=False); "
            "no git writes performed\n")
        return False

    _record_writer_tick("git_writer")

    # Claim-guard: pause when an operator session has an active
    # TASK_CLAIMS entry covering paths this writer touches. Returns
    # True (idle, not a failure) so the main loop schedules another
    # tick when the claim closes. See _active_claim_blocks_writer.
    if _active_claim_blocks_writer(
            "phase3_git_writer", os.path.dirname(asm)):
        sys.stderr.write(
            "[phase3] claim-guard active on evidence/experiments/; "
            "skipping result-writer tick\n")
        return True

    # Spool is the prerequisite. Without it /result has no bytes to
    # commit; refusing is louder than producing empty ticks forever.
    if manifest_spool.spool_root() is None:
        sys.stderr.write(
            "[phase3] COORDINATOR_SPOOL_DIR unset; refusing -- /result "
            "is not persisting manifest bytes, so nothing to commit\n")
        return False

    pending_ids = list(manifest_spool.list_pending_run_ids())
    # Companion side-files (default OFF -> empty -> bit-identical). A run's
    # companions may arrive AFTER its manifest was already committed+drained
    # (POST ordering: manifest then companions), so process companion run_ids
    # that are NOT in the manifest batch too -- otherwise a late companion
    # would never be picked up.
    sidefile_ids = (list(manifest_spool.list_pending_sidefile_run_ids())
                    if PHASE3_SPOOL_SIDEFILES else [])
    if not pending_ids and not sidefile_ids:
        return True  # idle tick is a successful no-op

    batch = pending_ids[:PHASE3_BATCH_SIZE]
    _batch_set = set(batch)
    sidefile_batch = (
        list(batch) + [s for s in sidefile_ids if s not in _batch_set]
    )[:PHASE3_BATCH_SIZE] if PHASE3_SPOOL_SIDEFILES else []

    if dry_run:
        sys.stdout.write(
            "[phase3] dry_run tick: %d pending, would commit %d "
            "(+%d run(s) with side-files)\n" % (
                len(pending_ids), len(batch), len(sidefile_ids)))
        return True

    clean, reason = _hub_working_tree_clean_for_writer(asm, "phase3")
    if not clean:
        sys.stderr.write(
            "[phase3] refusing tick: REE_assembly at %s is %s. Phase 3 "
            "does NOT autostash -- resolve the dirt by hand, then the "
            "next tick will retry.\n" % (asm, reason))
        return False

    # Absorb any unrelated commits that landed on origin between writer
    # ticks. The clean-tree check above is the precondition; without
    # this the writer wedges as soon as origin advances by even one
    # commit (see _sync_to_origin docstring).
    synced, reason = _sync_to_origin(asm, PHASE3_ASSEMBLY_BRANCH, "[phase3]")
    if not synced:
        sys.stderr.write(
            "[phase3] refusing tick: %s. Spool retained.\n" % reason)
        return False

    # Stage 1: write manifests onto the working tree and stage them.
    staged = []  # list of (run_id, relpath) successfully written
    for run_id in batch:
        raw = manifest_spool.read_manifest(run_id)
        meta = manifest_spool.read_meta(run_id) or {}
        if raw is None:
            sys.stderr.write(
                "[phase3] WARN missing manifest bytes for %s; skipping\n"
                % run_id)
            continue
        try:
            manifest_doc = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            manifest_doc = {}
        # Prefer the meta sidecar's hint (runner-supplied at /result time);
        # fall back to the manifest body; finally to the run_id default.
        hint = meta.get("manifest_relpath") or manifest_doc.get(
            "manifest_relpath")
        try:
            relpath = manifest_spool.derive_evidence_relpath(
                run_id, {"manifest_relpath": hint} if hint else manifest_doc)
        except ValueError as exc:
            sys.stderr.write(
                "[phase3] WARN derive_evidence_relpath rejected %s: %s\n"
                % (run_id, exc))
            continue
        target = os.path.join(asm, relpath)
        target_dir = os.path.dirname(target)
        # Atomic write: tmp file + os.replace. A crash mid-`fh.write`
        # otherwise leaves a truncated file that the immediately-following
        # `git add` would happily stage. Tmp+rename mirrors the spool
        # writer's atomic semantics.
        tmp_target = target + ".phase3.tmp"
        target_replaced = False
        try:
            os.makedirs(target_dir, exist_ok=True)
            with open(tmp_target, "wb") as fh:
                fh.write(raw)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_target, target)
            target_replaced = True
            _fsync_dir(target_dir)
        except OSError as exc:
            sys.stderr.write(
                "[phase3] WARN atomic write failed for %s -> %s: %r\n" % (
                    run_id, relpath, exc))
            # Best-effort cleanup of the tmp file. Target was NOT replaced
            # (os.replace ran last in the try block); nothing to revert.
            try:
                os.unlink(tmp_target)
            except OSError:
                pass
            continue
        try:
            _git(asm, "add", relpath, timeout=15, check=True)
            staged.append((run_id, relpath))
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            sys.stderr.write(
                "[phase3] WARN git add failed for %s -> %s: %r. "
                "Reverting working-tree write to keep the next tick's "
                "clean-tree check passing.\n" % (run_id, relpath, exc))
            if target_replaced:
                _revert_target_to_head(asm, relpath, target)

    # Stage 1b (optional, default-OFF): ALSO materialise the canonical runs/
    # pack for each staged flat manifest so cloud results are immediately
    # scoreable on origin. The pack files are git-added here so they land in
    # the SAME commit as the flat manifest(s) below. Best-effort and gated:
    # with PHASE3_MATERIALIZE_RUNPACK off this is a no-op and the tick is
    # bit-identical to the flat-only writer. Skipped when no manifest staged
    # (a companion-only tick has no flat manifest to pack).
    if PHASE3_MATERIALIZE_RUNPACK and staged:
        n_pack = _materialize_runpacks(asm, staged, "[phase3]")
        if n_pack:
            sys.stdout.write(
                "[phase3] materialised %d run-pack file(s) alongside %d flat "
                "manifest(s)\n" % (n_pack, len(staged)))

    # Stage 1c (optional, default-OFF): materialise each run's COMPANION
    # side-files (e.g. an *_episode_log.json) into evidence/experiments/ and
    # git-add them so they land in the SAME commit as the manifest(s). With
    # PHASE3_SPOOL_SIDEFILES off this is a no-op and the tick is bit-identical
    # to the manifest-only writer.
    n_sidefile = 0
    staged_sidefiles = []
    if PHASE3_SPOOL_SIDEFILES:
        n_sidefile, staged_sidefiles = _materialize_sidefiles(
            asm, sidefile_batch, "[phase3]")
        if n_sidefile:
            sys.stdout.write(
                "[phase3] materialised %d side-file(s) for %d run(s)\n" % (
                    n_sidefile, len(staged_sidefiles)))

    if not staged and not staged_sidefiles:
        sys.stderr.write(
            "[phase3] no manifests or side-files staged this tick; "
            "nothing to commit\n")
        return False

    # Stage 2: single commit + single push for the whole batch.
    today = db.utcnow()[:10]
    # MED-A: build the subject from the same constant the foreign-commit
    # check reads. Drifting one without the other (e.g. dropping the
    # trailing space, or rewording the prefix locally) would make the
    # writer reject its own commits as foreign. The optional side-file
    # clause is appended only when companions were staged, so a manifest-only
    # tick's subject is byte-identical to the pre-side-file writer.
    commit_msg = "%s%d v3 result manifest(s) %s" % (
        _PHASE3_COMMIT_PREFIX, len(staged), today)
    if n_sidefile:
        commit_msg = "%s%d v3 result manifest(s) + %d side-file(s) %s" % (
            _PHASE3_COMMIT_PREFIX, len(staged), n_sidefile, today)
    try:
        # Refresh origin/<branch> once at the top of the push-decision
        # block. Both the ahead-of-origin guard (HIGH-1) and the
        # writer-authored-only push guard (HIGH-2) need an accurate
        # remote-tracking ref:
        #   - HIGH-1: a stale ref would let `ahead==0` lie in case (a),
        #     draining the spool against a stale view of origin.
        #   - HIGH-2: a stale ref over-counts ahead commits, false-
        #     positiving the foreign-commit check (refusing legitimate
        #     work). Fetching first keeps the check on the correct
        #     reference set.
        # Fetch failure refuses the tick rather than proceeding with a
        # possibly-stale ref.
        fetched = _git(
            asm, "fetch", "--quiet", "origin", PHASE3_ASSEMBLY_BRANCH,
            check=False, timeout=30)
        if fetched.returncode != 0:
            sys.stderr.write(
                "[phase3] refusing tick: fetch origin %s failed (%s). "
                "Spool retained for next tick.\n" % (
                    PHASE3_ASSEMBLY_BRANCH,
                    fetched.stderr.strip()[:240]))
            return False

        diff = _git(asm, "diff", "--cached", "--quiet", check=False,
                    timeout=10)
        if diff.returncode == 0:
            # `git add` produced no diff. Two cases are indistinguishable
            # from `git diff --cached` alone:
            #   (a) bytes already on origin (true idempotent re-spool of
            #       a previously-committed-and-pushed run), OR
            #   (b) bytes live in an UNPUSHED local commit -- the tick
            #       after a rejected push (Phase 3 explicitly retires
            #       autostash, so a rejected push leaves the local
            #       commit in HEAD with no operator intervention).
            # Marking committed_at in case (b) without a push is unsafe:
            # the DB says "done" but origin never received the bytes.
            # `git rev-list --count origin/<branch>..HEAD` distinguishes
            # the two (fresh ref guaranteed by the fetch above).
            ahead = _git(
                asm, "rev-list", "--count",
                "origin/" + PHASE3_ASSEMBLY_BRANCH + "..HEAD",
                check=False, timeout=10)
            if ahead.returncode != 0:
                sys.stderr.write(
                    "[phase3] refusing to mark committed: rev-list "
                    "ahead-count failed (%s). Spool retained.\n" % (
                        ahead.stderr.strip()[:240]))
                return False
            ahead_count = ahead.stdout.strip()
            if ahead_count and ahead_count != "0":
                # Case (b): push the existing unpushed commit -- BUT
                # only if every ahead commit is writer-authored. A
                # foreign commit (operator hand-edit, accidental tooling
                # commit) must not be published under sync_daemon's
                # authority; refuse and let the operator investigate.
                ok, foreign = _check_ahead_writer_authored(
                    asm, PHASE3_ASSEMBLY_BRANCH)
                if not ok:
                    sys.stderr.write(
                        "[phase3] refusing tick: %d foreign commit(s) "
                        "in origin/%s..HEAD that the writer did not "
                        "author: %s. NOT marking committed_at; spool "
                        "retained. Operator must investigate (do not "
                        "let the writer publish unrelated commits under "
                        "sync_daemon's authority).\n" % (
                            len(foreign), PHASE3_ASSEMBLY_BRANCH,
                            foreign))
                    return False
                push = _git(
                    asm, "push", "origin",
                    "HEAD:" + PHASE3_ASSEMBLY_BRANCH,
                    timeout=60, check=False)
                if push.returncode != 0:
                    sys.stderr.write(
                        "[phase3] push REJECTED for unpushed local "
                        "commit: %s. NOT marking committed_at; spool "
                        "retained. Operator must investigate (non-"
                        "fast-forward = hub is behind origin; resolve "
                        "by hand).\n" % (push.stderr.strip()[:240]))
                    return False
                sys.stdout.write(
                    "[phase3] pushed unpushed local commit (HEAD was %s "
                    "ahead of origin/%s); %d row(s) committed\n" % (
                        ahead_count, PHASE3_ASSEMBLY_BRANCH, len(staged)))
                _record_writer_commit(
                    "git_writer", repo_path=asm, subject=commit_msg)
            else:
                # Case (a): true idempotent re-spool. ahead == 0 means
                # the bytes really are on origin already.
                sys.stdout.write(
                    "[phase3] batch already on tree and on origin "
                    "(ahead==0); marking %d row(s) committed without "
                    "a push\n" % len(staged))
        else:
            _git(asm, "commit", "-m", commit_msg, timeout=20, check=True)
            # After the writer's own commit lands, origin/<branch>..HEAD
            # includes the writer's commit (matches phase3: prefix) plus
            # any operator commits that were already ahead of origin.
            # Refuse the push if any foreign commit would be carried
            # along. The writer's commit remains in local HEAD; the next
            # tick will re-enter via case (b) and refuse again until the
            # operator resolves the foreign commit.
            ok, foreign = _check_ahead_writer_authored(
                asm, PHASE3_ASSEMBLY_BRANCH)
            if not ok:
                sys.stderr.write(
                    "[phase3] refusing tick: writer's commit landed but "
                    "%d foreign commit(s) are ahead of origin/%s and "
                    "would be carried by the push: %s. NOT marking "
                    "committed_at; spool retained. Operator must "
                    "resolve the foreign commit(s) before next tick.\n"
                    % (len(foreign), PHASE3_ASSEMBLY_BRANCH, foreign))
                return False
            push = _git(
                asm, "push", "origin", "HEAD:" + PHASE3_ASSEMBLY_BRANCH,
                timeout=60, check=False)
            if push.returncode != 0:
                sys.stderr.write(
                    "[phase3] push REJECTED: %s. NOT marking committed_at; "
                    "spool retained for retry on the next tick. Operator "
                    "must investigate (non-fast-forward = hub is behind "
                    "origin; resolve by hand).\n" % (push.stderr.strip()[:240]))
                return False
            _record_writer_commit(
                "git_writer", repo_path=asm, subject=commit_msg)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write("[phase3] git commit/push error: %r\n" % exc)
        return False

    # Stage 3: mark DB committed + delete spool entries. Order matters:
    # update DB first (cheap, atomic) before deleting bytes from disk so a
    # crash between the two leaves the spool entries that the next tick
    # will re-process as if they were uncommitted, which `git add` will
    # detect as no-diff and short-circuit (the idempotent-already-on-tree
    # branch above).
    #
    # MED-1 from the 2026-05-27 review: `UPDATE ... WHERE committed_at
    # IS NULL` returns rowcount 0 both when the row exists but is
    # already marked AND when the row is missing entirely. The second
    # case is an invariant violation -- bytes reached origin via the
    # writer's push but the DB has no record of the run. Surface it
    # loudly with a per-run WARN, but still proceed to drop the spool:
    # bytes ARE on origin, retaining the spool would replay forever
    # against the same missing-row condition.
    now = db.utcnow()
    # Guard the run-id IN (...) query against an empty `staged` -- a
    # companion-only tick (no manifests committed this round, only side-files)
    # has nothing to mark in `results`; `IN ()` is a SQLite syntax error.
    if staged:
        pre_existing = {
            row["run_id"]
            for row in conn.execute(
                "SELECT run_id FROM results WHERE run_id IN (%s)" % (
                    ",".join("?" * len(staged))),
                [run_id for run_id, _ in staged]).fetchall()
        }
        missing = [run_id for run_id, _ in staged
                   if run_id not in pre_existing]
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                "UPDATE results SET committed_at=? WHERE run_id=? "
                "AND committed_at IS NULL",
                [(now, run_id) for run_id, _ in staged],
            )
            conn.execute("COMMIT")
        except Exception as exc:  # noqa: BLE001 -- daemon must not die
            sys.stderr.write(
                "[phase3] WARN committed_at update failed: %r. Spool "
                "retained; next tick will replay idempotently.\n" % exc)
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            return False
        if missing:
            sys.stderr.write(
                "[phase3] WARN invariant violation: %d manifest(s) reached "
                "origin via this commit but have no `results` row in the "
                "coordinator DB: %s. Spool will be drained (bytes ARE on "
                "origin); the DB/origin mismatch needs operator audit. "
                "Likely causes: POST /result wrote spool bytes before "
                "`db.record_result` recorded the row, or the row was "
                "deleted out-of-band.\n" % (
                    len(missing), missing[:5]))

    for run_id, _ in staged:
        manifest_spool.delete_manifest(run_id)

    # Drop the companion spool dirs for every run whose side-files were
    # committed this tick (bytes are now on origin). Manifest-only ticks have
    # no staged_sidefiles -> no-op.
    for run_id in staged_sidefiles:
        manifest_spool.delete_sidefiles(run_id)

    sys.stdout.write(
        "[phase3] committed %d manifest(s)%s (%d remaining in spool)\n" % (
            len(staged),
            (" + %d side-file(s)" % n_sidefile) if n_sidefile else "",
            max(0, len(pending_ids) - len(staged))))
    return True


# Canonical key order for items emitted in experiment_queue.json. Stored
# `item_json` blobs are written with sort_keys=True (db.upsert_experiment),
# so the loaded dict has alphabetical keys; writing the materialised file
# without re-ordering would land an "every item reordered" diff on the
# first Phase 3 cutover commit and lose operator-meaningful field grouping
# (identifier -> executable -> scheduling -> claim -> state). This list is
# the operator-visible shape contract; do NOT reorder casually -- any
# change here produces another full-rewrite commit on every queue file
# already on origin. Unknown keys (forward-compat custom fields) land
# alphabetically AFTER this canonical block.
_QUEUE_ITEM_KEY_ORDER = (
    # 1. Identity
    "queue_id",
    "title",
    "description",
    # 2. Executable
    "script",
    # 3. Scheduling
    "priority",
    "machine_affinity",
    "estimated_minutes",
    # 4. Provenance / lineage
    "supersedes",
    "claim_id",
    "backlog_id",
    # 5. Operator flags
    "force_rerun",
    "note",
    # 6. Coordinator-managed state (overlaid below from live DB columns)
    "status",
    "claimed_by",
)


def _canonicalise_queue_item(item):
    """Return `item` with keys in _QUEUE_ITEM_KEY_ORDER first, then any
    unknown keys alphabetically. Keys absent from `item` are skipped (not
    inserted as None) so the materialised file is shape-faithful to what
    the operator originally wrote."""
    canonical = {}
    for k in _QUEUE_ITEM_KEY_ORDER:
        if k in item:
            canonical[k] = item[k]
    for k in sorted(item.keys()):
        if k not in canonical:
            canonical[k] = item[k]
    return canonical


def _materialise_queue_from_db(conn, current_calibration):
    """Build the canonical experiment_queue.json content from DB rows.

    Returns the queue dict (schema_version + calibration + items). Only
    items with status NOT IN ('completed', 'failed') are emitted -- those
    are terminal states and don't belong in the worklist.

    Each item is reconstructed from the stored `item_json` blob with the
    live DB columns (status, claimed_by_machine, claimed_at) overlaid:
    item_json preserves the operator-supplied fields (script, priority,
    machine_affinity, etc.) verbatim, while the DB columns hold the
    coordinator-managed state. Keys are then reordered into the
    _QUEUE_ITEM_KEY_ORDER canonical shape so per-tick claim/release
    transitions don't produce noisy "claimed_by appeared at end of dict"
    diffs.

    `current_calibration` is preserved verbatim from the existing file --
    the DB doesn't store calibration data so we must round-trip it.
    """
    rows = conn.execute(
        "SELECT queue_id, status, claimed_by_machine, claimed_at, "
        "item_json, priority FROM experiments "
        "WHERE status NOT IN ('completed', 'failed') "
        "ORDER BY priority DESC, queue_id"
    ).fetchall()
    items = []
    for r in rows:
        try:
            item = json.loads(r["item_json"])
        except (ValueError, TypeError):
            # Corrupt blob -- skip, log later. Should not happen in practice.
            sys.stderr.write(
                "[phase3-queue] WARN skipping %s: item_json unparseable\n"
                % r["queue_id"])
            continue
        if not isinstance(item, dict):
            sys.stderr.write(
                "[phase3-queue] WARN skipping %s: item_json not a dict\n"
                % r["queue_id"])
            continue
        # Overlay coordinator-managed state. claim_authority='coordinator'
        # in upsert_experiment preserves claim fields IN THE DB across
        # operator file edits; here we surface that DB state into the
        # written-back file so operators see the canonical view.
        item["status"] = r["status"]
        if r["claimed_by_machine"]:
            item["claimed_by"] = {
                "machine": r["claimed_by_machine"],
                "claimed_at": r["claimed_at"],
            }
        else:
            item.pop("claimed_by", None)
        items.append(_canonicalise_queue_item(item))
    return {
        "schema_version": "v1",
        "calibration": current_calibration or {},
        "items": items,
    }


def phase3_queue_writer(
    conn,
    *,
    ree_v3_path=None,
    queue_relpath=None,
    branch=None,
):
    """PLAN.md step 5: snapshot the canonical queue from the DB into
    `experiment_queue.json` on the hub's ree-v3 checkout, commit, push.

    Pairs with `phase3_git_writer` -- same safety contract:
      - PHASE3_QUEUE_WRITER_READY gates execution (False -> log stub
        message and return False).
      - Refuses on dirty working tree on the ree-v3 checkout.
      - Refuses on fetch failure (a stale `origin/main` ref would fool
        the ahead-of-origin guard).
      - Foreign-commit check before push: only commits whose subject
        starts with `_PHASE3_QUEUE_COMMIT_PREFIX` may be carried by the
        writer's push.
      - Atomic write (tmp + os.replace) before `git add`.
      - Returns True on success (or idle no-op when the materialised
        view matches the current file).
      - Returns False on any refusal; never raises.
    """
    repo = ree_v3_path or PHASE3_REE_V3
    rel = queue_relpath or PHASE3_QUEUE_RELPATH
    br = branch or PHASE3_REE_V3_BRANCH

    if not PHASE3_QUEUE_WRITER_READY:
        sys.stderr.write(
            "[phase3-queue] queue writer stub "
            "(PHASE3_QUEUE_WRITER_READY=False); no git writes performed\n")
        return False

    _record_writer_tick("queue_writer")

    # Claim-guard: pause when an operator session has an active
    # TASK_CLAIMS entry covering ree-v3/experiment_queue.json. Returns
    # True (idle, not a failure) so the main loop schedules another
    # tick when the claim closes. See _active_claim_blocks_writer.
    if _active_claim_blocks_writer(
            "phase3_queue_writer", os.path.dirname(repo)):
        sys.stderr.write(
            "[phase3-queue] claim-guard active on experiment_queue.json; "
            "skipping queue-writer tick\n")
        return True

    target = os.path.join(repo, rel)

    # Sequence must be: clean-tree precondition -> absorb origin moves
    # (rebase writer-authored commits) -> read current file -> materialise
    # from DB -> idempotency check -> write. Reading the file before the
    # rebase risks comparing against pre-rebase content and either
    # over-writing fresh origin content with a stale DB view or
    # false-positiving the idempotent no-op against post-rebase content.
    clean, reason = _hub_working_tree_clean(repo)
    if not clean:
        sys.stderr.write(
            "[phase3-queue] refusing tick: %s at %s is %s. Phase 3 "
            "does NOT autostash -- resolve the dirt by hand, then the "
            "next tick will retry.\n" % (rel, repo, reason))
        return False

    # Additive policy: when PHASE3_QUEUE_CONFLICT_RECOVERY is set, FORCE
    # queue-writer conflict recovery on (pass True). When unset, pass None
    # so _sync_to_origin falls back to the legacy
    # PHASE3_AUTO_RESET_ON_REBASE_CONFLICT env -- the new flag therefore
    # never DISABLES recovery the global env already provides; it only adds
    # a queue-scoped, observable (n_conflict_recoveries) way to turn it on.
    synced, reason = _sync_to_origin(
        repo, br, "[phase3-queue]",
        auto_reset_on_conflict=(True if PHASE3_QUEUE_CONFLICT_RECOVERY
                                else None),
        writer_name="queue_writer")
    if not synced:
        sys.stderr.write(
            "[phase3-queue] refusing tick: %s. Working tree intact.\n"
            % reason)
        return False

    # Read current file (post-sync) to preserve calibration block and to
    # compare for idempotent no-op when DB-materialised view matches.
    current_text = None
    current_calibration = {}
    if os.path.exists(target):
        try:
            with open(target, "r", encoding="utf-8") as fh:
                current_text = fh.read()
            try:
                current_doc = json.loads(current_text)
                if isinstance(current_doc, dict):
                    current_calibration = current_doc.get(
                        "calibration") or {}
            except ValueError:
                # Existing file is unparseable -- the writer should still
                # overwrite it with a fresh DB-materialised view, but we
                # have no calibration to preserve.
                sys.stderr.write(
                    "[phase3-queue] WARN current %s is not valid JSON; "
                    "calibration block will be empty in the rewrite\n"
                    % rel)
        except OSError as exc:
            sys.stderr.write(
                "[phase3-queue] WARN could not read current %s: %r\n"
                % (rel, exc))

    # Materialise the DB view.
    try:
        new_doc = _materialise_queue_from_db(conn, current_calibration)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            "[phase3-queue] DB query failed: %r. Skipping tick.\n" % exc)
        return False

    new_text = json.dumps(new_doc, indent=2, sort_keys=False) + "\n"

    # Idempotent no-op when content matches.
    if current_text is not None and current_text == new_text:
        return True

    # Atomic write to the working tree.
    tmp_target = target + ".phase3.tmp"
    target_dir = os.path.dirname(target)
    try:
        os.makedirs(target_dir, exist_ok=True)
        with open(tmp_target, "w", encoding="utf-8") as fh:
            fh.write(new_text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_target, target)
        _fsync_dir(target_dir)
    except OSError as exc:
        sys.stderr.write(
            "[phase3-queue] WARN write failed: %r. Skipping tick.\n" % exc)
        try:
            os.unlink(tmp_target)
        except OSError:
            pass
        return False

    today = db.utcnow()[:10]
    # MED-A: same constant-sharing rationale as the result writer above.
    commit_msg = "%ssnapshot %s" % (_PHASE3_QUEUE_COMMIT_PREFIX, today)

    # The working-tree edit above is DERIVED from the coordinator DB and is
    # reproduced from scratch on every tick, so any exit that does NOT turn
    # it into a commit must leave the tree clean again. Otherwise the edit
    # (staged or unstaged) trips this writer's own clean-tree precondition
    # on the NEXT tick and every tick after it -- a permanent self-deadlock
    # that needs a human to clear. See _restore_derived_path for the
    # 2026-07-18 incident this guards.
    restore_needed = True
    try:
        # Same fetch + foreign-check + commit/push sequence as the
        # result writer, applied to the ree-v3 repo.
        fetched = _git(
            repo, "fetch", "--quiet", "origin", br,
            check=False, timeout=30)
        if fetched.returncode != 0:
            sys.stderr.write(
                "[phase3-queue] refusing tick: fetch origin %s failed "
                "(%s). Derived snapshot restored; next tick will retry.\n"
                % (br, fetched.stderr.strip()[:240]))
            return False

        # Stage the file. If `git add` produces no diff (the working-tree
        # write was byte-identical to HEAD's blob), fall through to the
        # ahead-of-origin check just like the result writer does.
        _git(repo, "add", rel, timeout=15, check=True)

        diff = _git(repo, "diff", "--cached", "--quiet",
                    check=False, timeout=10)
        if diff.returncode == 0:
            # Byte-identical to HEAD's blob: nothing staged, tree already
            # clean, so there is no derived edit left to restore.
            restore_needed = False
            # No-diff path. ahead-of-origin guard:
            ahead = _git(
                repo, "rev-list", "--count",
                "origin/" + br + "..HEAD",
                check=False, timeout=10)
            if ahead.returncode != 0:
                sys.stderr.write(
                    "[phase3-queue] refusing tick: rev-list ahead-count "
                    "failed (%s).\n" % ahead.stderr.strip()[:240])
                return False
            ahead_count = ahead.stdout.strip()
            if ahead_count and ahead_count != "0":
                ok, foreign = _check_ahead_writer_authored(repo, br)
                if not ok:
                    sys.stderr.write(
                        "[phase3-queue] refusing tick: %d foreign "
                        "commit(s) in origin/%s..HEAD that no phase3 "
                        "writer authored: %s. Operator must "
                        "investigate.\n" % (
                            len(foreign), br, foreign))
                    return False
                push = _git(
                    repo, "push", "origin", "HEAD:" + br,
                    timeout=60, check=False)
                if push.returncode != 0:
                    sys.stderr.write(
                        "[phase3-queue] push REJECTED for unpushed local "
                        "commit: %s. Tree clean; commit retained for "
                        "retry.\n" % push.stderr.strip()[:240])
                    return False
                sys.stdout.write(
                    "[phase3-queue] pushed unpushed local commit "
                    "(HEAD was %s ahead of origin/%s)\n" % (
                        ahead_count, br))
                _record_writer_commit(
                    "queue_writer", repo_path=repo, subject=commit_msg)
                return True
            # ahead == 0 -- view is already on origin. Treat as no-op.
            return True

        # diff.returncode != 0: there's a real change to commit.
        _git(repo, "commit", "-m", commit_msg, timeout=20, check=True)
        # The commit consumed the staged edit; the tree is clean whatever
        # the push does next, and restoring now would revert real work.
        restore_needed = False
        ok, foreign = _check_ahead_writer_authored(repo, br)
        if not ok:
            sys.stderr.write(
                "[phase3-queue] refusing tick: writer's commit landed "
                "but %d foreign commit(s) are ahead of origin/%s and "
                "would be carried by the push: %s. Operator must "
                "resolve the foreign commit(s) before next tick.\n" % (
                    len(foreign), br, foreign))
            return False
        push = _git(
            repo, "push", "origin", "HEAD:" + br,
            timeout=60, check=False)
        if push.returncode != 0:
            sys.stderr.write(
                "[phase3-queue] push REJECTED: %s. Working tree commit "
                "retained for retry.\n" % push.stderr.strip()[:240])
            return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write(
            "[phase3-queue] git commit/push error: %r\n" % exc)
        return False
    finally:
        # Runs on every exit -- the early `return False`s above, the
        # exception handler, and the success path (where restore_needed
        # is already False). A TimeoutExpired on `git commit` is the one
        # ambiguous case: the commit may or may not have landed, so this
        # restores only when the commit was not observed to succeed, and
        # `git checkout HEAD -- <path>` is a no-op against a tree the
        # commit already cleaned.
        if restore_needed:
            _restore_derived_path(repo, rel, "[phase3-queue]")

    _record_writer_commit(
        "queue_writer", repo_path=repo, subject=commit_msg)
    sys.stdout.write(
        "[phase3-queue] snapshot pushed (%d active item(s))\n"
        % len(new_doc["items"]))
    return True


def _atomic_write_text(target_path, text):
    """Atomic working-tree write (tmp + fsync + os.replace). Returns
    True on success, False on OSError. Matches the LOW-2 pattern in
    phase3_git_writer."""
    tmp = target_path + ".phase3.tmp"
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target_path)
        return True
    except OSError as exc:
        sys.stderr.write(
            "[phase3-heartbeats] WARN write failed for %s: %r\n" % (
                target_path, exc))
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def _safe_machine_filename(machine):
    """Defensive normalisation for the per-machine output filename.

    Accept alphanumeric + dot + dash + underscore (matches the existing
    experiment_runner convention for status filenames). Reject anything
    that would let a hostile heartbeat write outside the target
    subdirectory:
      - empty string
      - the bare current/parent indicators "." or ".."
      - ANY string containing the path-traversal sequence ".." (catches
        cases like "../escape" where character-stripping otherwise
        leaves "..escape" still containing a traversal token)
      - leading dot (hidden-file convention; not a real machine name and
        confuses ls/glob)
    """
    if not machine or not isinstance(machine, str):
        return None
    cleaned = "".join(
        c for c in machine if c.isalnum() or c in ("-", "_", "."))
    if not cleaned:
        return None
    if cleaned in (".", ".."):
        return None
    if ".." in cleaned:
        return None
    if cleaned.startswith("."):
        return None
    return cleaned


def phase3_heartbeat_writer(
    conn,
    *,
    ree_assembly_path=None,
    branch=None,
):
    """PLAN.md step 6: materialise runner_heartbeats/<machine>.json and
    runner_status/<machine>.json from the coordinator's heartbeats
    table into the hub's REE_assembly checkout, committing only when
    fleet state actually changes (2026-05-31 state-change redesign).

    Pairs with phase3_git_writer + phase3_queue_writer to give
    sync_daemon sole-writer authority across both coordination repos.

    Commit policy: commit ONLY when one of these is true since the last
    commit, subject to the debounce + liveness floor:
      - queue_id changed for any machine (experiment start / finish /
        switch / release)
      - idle flag flipped (without queue_id change)
      - runner_pid changed (worker restart)
      - a machine appeared or transitioned silent <-> active
        (heartbeat.last_tick_utc stalled past PHASE3_HEARTBEAT_STALE_AFTER)

    Debounce: PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL (default 300s).
    State-change ticks within the window hold; the next post-window
    tick commits whatever has accumulated.

    Liveness floor: PHASE3_HEARTBEAT_LIVENESS_INTERVAL (default 1800s).
    If no state-change commit has fired for this long, force a
    "liveness tick" commit batching current state so external viewers
    see fresh git timestamps proving the fleet is alive.

    Safety contract (preserved verbatim from pre-redesign):
      - PHASE3_HEARTBEAT_WRITER_READY gates execution (False -> stub).
      - Claim-guard: pauses when an operator session holds an active
        TASK_CLAIMS entry covering the writer's paths.
      - Dirty-tree refusal (only when committing -- skip-decision ticks
        do not touch the working tree).
      - Fetch + rebase-on-behind via _sync_to_origin.
      - Atomic per-file write (tmp + fsync + os.replace).
      - Foreign-commit check (any phase3-* writer prefix accepted).
      - Idempotent: skip-decision ticks read DB only; commit-decision
        ticks still re-check actually_changed vs working tree.
      - Never raises.

    Per-machine output is byte-for-byte the JSON payload the runner
    POSTed -- consumers (explorer, scaler workflow, governance) see
    exactly the shape they did under the runner's old git push.
    Rows with both payload columns NULL (legacy clients) are skipped
    silently and produce no file.
    """
    global _PHASE3_HEARTBEAT_LAST_COMMIT_TS, _PHASE3_HEARTBEAT_INITIALIZED

    asm = ree_assembly_path or PHASE3_REE_ASSEMBLY
    br = branch or PHASE3_ASSEMBLY_BRANCH

    if not PHASE3_HEARTBEAT_WRITER_READY:
        sys.stderr.write(
            "[phase3-heartbeats] writer stub "
            "(PHASE3_HEARTBEAT_WRITER_READY=False); no git writes performed\n")
        return False

    _record_writer_tick("heartbeat_writer")

    # Claim-guard: pause when an operator session has an active
    # TASK_CLAIMS entry covering runner_heartbeats/, runner_status/, or
    # runner_status.json. Returns True (idle, not a failure) so the main
    # loop schedules another tick when the claim closes.
    if _active_claim_blocks_writer(
            "phase3_heartbeat_writer", os.path.dirname(asm)):
        sys.stderr.write(
            "[phase3-heartbeats] claim-guard active on runner_heartbeats/"
            " or runner_status/; skipping heartbeat-writer tick\n")
        return True

    rows = conn.execute(
        "SELECT machine, heartbeat_payload_json, status_payload_json "
        "FROM heartbeats "
        "WHERE heartbeat_payload_json IS NOT NULL "
        "   OR status_payload_json IS NOT NULL"
    ).fetchall()
    if not rows:
        return True

    # Parse in one pass: file-content for the eventual commit + per-
    # machine state for change detection. Bad rows (unsafe name /
    # malformed JSON) are skipped without aborting the whole tick.
    pending_writes = []   # list of (relpath, text)
    current_states = {}   # safe_machine -> state dict
    now = _phase3_heartbeat_now()

    for r in rows:
        safe = _safe_machine_filename(r["machine"])
        if not safe:
            sys.stderr.write(
                "[phase3-heartbeats] WARN skipping unsafe machine name %r\n"
                % (r["machine"],))
            continue
        hb_doc = None
        st_doc = None
        for kind, reldir, column in (
            ("heartbeat", PHASE3_HEARTBEATS_RELDIR,
             "heartbeat_payload_json"),
            ("status", PHASE3_STATUS_RELDIR, "status_payload_json"),
        ):
            raw = r[column]
            if not raw:
                continue
            try:
                doc = json.loads(raw)
            except (ValueError, TypeError):
                sys.stderr.write(
                    "[phase3-heartbeats] WARN skipping %s/%s: stored "
                    "%s is not valid JSON\n" % (
                        reldir, safe, column))
                continue
            if kind == "heartbeat":
                hb_doc = doc
            else:
                st_doc = doc
            text = json.dumps(doc, indent=2) + "\n"
            rel = "%s/%s.json" % (reldir, safe)
            pending_writes.append((rel, text))

        if hb_doc is None and st_doc is None:
            continue
        state = _extract_heartbeat_machine_state(st_doc, hb_doc)
        state["silent"] = _phase3_heartbeat_is_silent(
            safe, state.get("last_tick_utc"), now,
            PHASE3_HEARTBEAT_STALE_AFTER)
        current_states[safe] = state

    if not pending_writes:
        return True

    # Commit-decision: empty changes within liveness window -> skip;
    # non-empty changes within debounce window -> hold; otherwise commit.
    # First post-startup tick treats current state as the baseline and
    # commits to seed last_committed_state.
    if not _PHASE3_HEARTBEAT_INITIALIZED:
        changes = []
        commit_reason = "initial"
    else:
        changes = _phase3_heartbeat_compute_changes(
            _PHASE3_HEARTBEAT_LAST_COMMITTED_STATE,
            current_states)
        time_since_commit = now - _PHASE3_HEARTBEAT_LAST_COMMIT_TS
        if changes:
            if time_since_commit < PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL:
                # Hold for debounce; do NOT touch the working tree. The
                # change set is computed against last-committed state, so
                # the next post-window tick re-derives whatever has
                # accumulated and commits it together.
                return True
            commit_reason = "state-change"
        else:
            if time_since_commit < PHASE3_HEARTBEAT_LIVENESS_INTERVAL:
                return True
            commit_reason = "liveness"

    # ---- COMMIT PATH ----
    # Sequence: clean-tree precondition -> absorb origin moves (rebase
    # writer-authored heartbeat commits) -> recompute actually_changed
    # against the post-rebase working tree. Doing the comparison BEFORE
    # the sync would either false-positive (working tree matches origin
    # for some files but our retained-for-retry commit is the only
    # reason) or false-negative on files origin updated in between.
    clean, reason = _hub_working_tree_clean_for_writer(asm, "phase3-heartbeats")
    if not clean:
        sys.stderr.write(
            "[phase3-heartbeats] refusing tick: REE_assembly at %s is "
            "%s. Phase 3 does NOT autostash -- resolve the dirt by "
            "hand.\n" % (asm, reason))
        return False

    synced, reason = _sync_to_origin(asm, br, "[phase3-heartbeats]")
    if not synced:
        sys.stderr.write(
            "[phase3-heartbeats] refusing tick: %s.\n" % reason)
        return False

    actually_changed = []
    for rel, text in pending_writes:
        target = os.path.join(asm, rel)
        if os.path.exists(target):
            try:
                with open(target, "r", encoding="utf-8") as fh:
                    if fh.read() == text:
                        continue
            except OSError:
                pass
        actually_changed.append((rel, text))

    # Atomic writes + git add. Partial-failure tolerant.
    staged = []
    for rel, text in actually_changed:
        target = os.path.join(asm, rel)
        if not _atomic_write_text(target, text):
            continue
        try:
            _git(asm, "add", rel, timeout=15, check=True)
            staged.append(rel)
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            sys.stderr.write(
                "[phase3-heartbeats] WARN git add failed for %s: %r. "
                "Reverting working-tree write to keep the next tick's "
                "clean-tree check passing.\n" % (rel, exc))
            _revert_target_to_head(asm, rel, target)

    commit_msg = _phase3_heartbeat_commit_message(
        changes, len(current_states))
    # Every reach of the commit path is an intentional event (initial /
    # state-change / liveness) and must produce a commit visible in
    # `git log`. Silent <-> active transitions are derived inside the
    # writer (not in the payload bytes) so a state-change fire may
    # legitimately have nothing staged; --allow-empty makes the event
    # visible regardless. Liveness commits use the same path.

    try:
        diff = _git(asm, "diff", "--cached", "--quiet",
                    check=False, timeout=10)
        commit_args = ["commit", "-m", commit_msg]
        if diff.returncode == 0:
            commit_args.append("--allow-empty")
        _git(asm, *commit_args, timeout=20, check=True)
        ok, foreign = _check_ahead_writer_authored(asm, br)
        if not ok:
            sys.stderr.write(
                "[phase3-heartbeats] refusing tick: writer's commit landed "
                "but %d foreign commit(s) are ahead of origin/%s and no "
                "phase3 writer authored them: %s.\n" % (
                    len(foreign), br, foreign))
            return False
        push = _git(
            asm, "push", "origin", "HEAD:" + br,
            timeout=60, check=False)
        if push.returncode != 0:
            sys.stderr.write(
                "[phase3-heartbeats] push REJECTED: %s. Working tree "
                "commit retained for retry.\n" %
                push.stderr.strip()[:240])
            return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write(
            "[phase3-heartbeats] git commit/push error: %r\n" % exc)
        return False

    sys.stdout.write(
        "[phase3-heartbeats] pushed (%s): %s (%d file(s))\n" % (
            commit_reason, commit_msg, len(staged)))
    _record_writer_commit(
        "heartbeat_writer", repo_path=asm, subject=commit_msg)
    _PHASE3_HEARTBEAT_LAST_COMMITTED_STATE.clear()
    _PHASE3_HEARTBEAT_LAST_COMMITTED_STATE.update(current_states)
    _PHASE3_HEARTBEAT_LAST_COMMIT_TS = now
    _PHASE3_HEARTBEAT_INITIALIZED = True
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=os.environ.get(
        "COORDINATOR_QUEUE_FILE", DEFAULT_QUEUE))
    ap.add_argument("--db", default=os.environ.get(
        "COORDINATOR_DB", os.path.join(
            os.path.dirname(__file__), "coordinator.db")))
    ap.add_argument("--interval", type=float,
                    default=_validate_float(
                        os.environ.get("SYNC_INTERVAL", "60"),
                        "SYNC_INTERVAL", 60.0))
    ap.add_argument("--once", action="store_true",
                    help="reconcile once and exit (used by tests)")
    ap.add_argument("--i-understand-phase3", action="store_true")
    args = ap.parse_args()

    sync_mode = os.environ.get("SYNC_MODE", "shadow")
    if sync_mode == "authoritative":
        if not args.i_understand_phase3:
            sys.stderr.write(
                "refusing: SYNC_MODE=authoritative needs "
                "--i-understand-phase3 (Phase 3 not built)\n")
            return 2
        db.init_db(args.db)
        # Seed _WRITER_HEALTH from git history so /writer-health surfaces a
        # meaningful per-writer commit snapshot from process start, before
        # any writer has ticked. Best-effort: git failures log a warning
        # and leave commit fields null (same as the pre-bootstrap path).
        _bootstrap_writer_health_from_git()
        _persist_writer_health()
        while True:
            conn = db.connect(args.db)
            try:
                # Authoritative tick has three jobs:
                # (1) pick up operator-added queue items from the git file
                #     into the DB (upsert_only=True; the DB never deletes
                #     terminal-state rows just because the file lost them,
                #     because the file is now a DERIVED view).
                # (2) write committed result manifests through to
                #     REE_assembly (PLAN.md step 4 = phase3_git_writer).
                # (3) materialise the canonical queue file back from the
                #     DB to ree-v3 (PLAN.md step 5 = phase3_queue_writer).
                try:
                    reconcile_once(conn, args.queue,
                                   claim_authority="coordinator",
                                   upsert_only=True)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(
                        "[sync] authoritative reconcile error "
                        "(non-fatal): %r\n" % exc)
                # phase3_git_writer claims to never raise, but defensive
                # wrap matches the queue + heartbeat writer pattern. On
                # an unexpected raise default ok=False so the `if not ok`
                # below trips and the daemon exits with code 2 (the
                # result writer is the cutover gate -- if it goes off
                # the rails we want loud failure, not silent skip).
                try:
                    ok = phase3_git_writer(conn, args.queue)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(
                        "[sync] phase3_git_writer raised (unexpected): %r. "
                        "Treating as not-ready; daemon will exit.\n" % exc)
                    _record_writer_error("git_writer", exc)
                    ok = False
                # phase3_queue_writer is independent of the result writer;
                # gated by its own PHASE3_QUEUE_WRITER_READY flag. When
                # the flag is False the function logs the stub message
                # and returns False -- treat as a non-fatal no-op in the
                # main loop (the result writer's readiness is the cutover
                # gate; the queue writer is a follow-on step that can be
                # enabled later without disrupting the result path).
                try:
                    phase3_queue_writer(conn)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(
                        "[sync] phase3_queue_writer error "
                        "(non-fatal): %r\n" % exc)
                    _record_writer_error("queue_writer", exc)
                # PLAN.md step 6: materialise runner_heartbeats/*.json
                # and runner_status/*.json from heartbeat_payload_json /
                # status_payload_json columns. Gated by its own
                # PHASE3_HEARTBEAT_WRITER_READY flag; non-fatal.
                try:
                    phase3_heartbeat_writer(conn)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(
                        "[sync] phase3_heartbeat_writer error "
                        "(non-fatal): %r\n" % exc)
                    _record_writer_error("heartbeat_writer", exc)
                # Persist writer-health snapshot for the coordinator's
                # /writer-health endpoint. Reads tick/commit/error stamps
                # recorded inside each writer above. Best-effort; the
                # helper swallows IO errors so a full disk on the hub
                # cannot wedge the writer loop.
                _persist_writer_health()
            finally:
                conn.close()
            if not ok:
                sys.stderr.write(
                    "refusing: phase3 git writer not ready (see "
                    "PHASE3_GIT_WRITER_READY and phase3_preflight.py)\n")
                return 2
            if args.once:
                return 0
            time.sleep(args.interval)
    if sync_mode not in ("shadow", "coordinator"):
        sys.stderr.write(
            "refusing: SYNC_MODE must be shadow, coordinator, or "
            "authoritative (got %r)\n" % sync_mode)
        return 2
    claim_authority = "coordinator" if sync_mode == "coordinator" else "git"

    db.init_db(args.db)
    while True:
        conn = db.connect(args.db)
        try:
            n, div = reconcile_once(conn, args.queue,
                                    claim_authority=claim_authority)
            sys.stdout.write(
                "[sync] reconciled %d items, %d state-divergence(s)\n" % (
                    n, div))
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001 -- daemon must not die
            sys.stderr.write("[sync] reconcile error: %r\n" % exc)
        finally:
            conn.close()
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
