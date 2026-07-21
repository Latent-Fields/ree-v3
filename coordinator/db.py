"""SQLite layer for the experiment coordinator.

Single-process writer. WAL mode + an explicit BEGIN IMMEDIATE around the
conditional claim UPDATE makes the claim atomic without any external mutex:
SQLite serializes the writer and the WHERE clause is the gate.

All stdout/stderr text is ASCII-only (Windows cp1252 safety).
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path):
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    _migrate_heartbeats(conn)
    _migrate_commands(conn)
    _migrate_experiments(conn)
    return conn


def _migrate_experiments(conn):
    """Additive columns recording WHY a queue row went terminal.

    status='completed' is overloaded: it means "no longer claimable", not
    "ran to a scientific outcome". mark_queue_removed() already took a
    `reason` argument and threw it away, so an operator cancellation
    (POST /queue/remove), a runner ERROR and a scientific FAIL all produced
    a byte-identical row. That is what made "phantom completions"
    (status='completed' LEFT JOIN results -> no row) unclassifiable:
    the query cannot separate a crash from a deliberate cancellation.
    Confirmed on V3-EXQ-699a (2026-07-20), which was a /failure-autopsy
    session cancelling an in-flight run to queue its repaired successor
    699b -- correct behaviour recorded as if it were a completion.

    Mirrors _migrate_heartbeats. Purely additive: existing rows keep NULL,
    and no reader is required to consume these columns.
    """
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='experiments'"
    ).fetchone():
        return
    cols = {row[1] for row in conn.execute("PRAGMA table_info(experiments)")}
    if "removal_reason" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN removal_reason TEXT")
    if "removed_at" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN removed_at TEXT")


def _migrate_heartbeats(conn):
    """Additive columns for Phase-2 progress time estimates and the
    Phase-3 lifecycle-state shutdown_notify fields."""
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='heartbeats'"
    ).fetchone():
        return
    cols = {row[1] for row in conn.execute("PRAGMA table_info(heartbeats)")}
    if "seconds_elapsed" not in cols:
        conn.execute(
            "ALTER TABLE heartbeats ADD COLUMN seconds_elapsed INTEGER")
    if "seconds_remaining" not in cols:
        conn.execute(
            "ALTER TABLE heartbeats ADD COLUMN seconds_remaining INTEGER")
    if "last_shutdown_at" not in cols:
        conn.execute(
            "ALTER TABLE heartbeats ADD COLUMN last_shutdown_at TEXT")
    if "shutdown_reason" not in cols:
        conn.execute(
            "ALTER TABLE heartbeats ADD COLUMN shutdown_reason TEXT")
    if "expected_wake_condition" not in cols:
        conn.execute(
            "ALTER TABLE heartbeats ADD COLUMN expected_wake_condition TEXT")
    if "heartbeat_payload_json" not in cols:
        conn.execute(
            "ALTER TABLE heartbeats ADD COLUMN heartbeat_payload_json TEXT")
    if "status_payload_json" not in cols:
        conn.execute(
            "ALTER TABLE heartbeats ADD COLUMN status_payload_json TEXT")


def _migrate_commands(conn):
    """Additive columns for the Phase-3 coordinator command channel.

    The `commands` table predates the command-channel migration (it was
    created by the original schema with id/machine/kind/args/issued_by/
    issued_at/acked_at but never written to). The migration adds the
    ack-time result columns so a live DB picks them up without a rebuild.
    Mirrors _migrate_heartbeats."""
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='commands'"
    ).fetchone():
        return
    cols = {row[1] for row in conn.execute("PRAGMA table_info(commands)")}
    if "result_status" not in cols:
        conn.execute("ALTER TABLE commands ADD COLUMN result_status TEXT")
    if "result_note" not in cols:
        conn.execute("ALTER TABLE commands ADD COLUMN result_note TEXT")


def init_db(db_path):
    conn = connect(db_path)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        conn.executescript(fh.read())
    _migrate_heartbeats(conn)
    _migrate_commands(conn)
    _migrate_experiments(conn)
    conn.close()


# Statuses from which a row never becomes claimable again. Single source of
# truth because three readers must agree on it: sync_daemon.reconcile_once's
# terminal-row guard, sync_daemon._materialise_queue_from_db's worklist
# filter, and POST /queue/add's 409. A disagreement between them is exactly
# how a queue_id gets silently burned (V3-EXQ-728a, 2026-07-20).
TERMINAL_STATUSES = ("completed", "failed")


def _affinity_ok(affinity, machine):
    return affinity in (None, "", "any") or affinity == machine


def upsert_experiment(conn, item, preserve_claim=False):
    """Insert/refresh one queue item into the mirror, preserving the
    authoritative claim state carried in the item itself (git is the
    source of truth in shadow mode).

    In coordinator claim-cutover mode, git remains the worklist but stops
    being the claim authority. For existing rows, preserve the DB claim
    fields while refreshing queue metadata from git.
    """
    existing = None
    if preserve_claim:
        existing = conn.execute(
            "SELECT status, claimed_by_machine, claimed_at FROM experiments "
            "WHERE queue_id=?", (item["queue_id"],)
        ).fetchone()
    cb = item.get("claimed_by") or {}
    status = item.get("status", "pending")
    cb_machine = cb.get("machine")
    cb_at = cb.get("claimed_at")
    if existing is not None:
        status = existing["status"]
        cb_machine = existing["claimed_by_machine"]
        cb_at = existing["claimed_at"]
    conn.execute(
        """
        INSERT INTO experiments
          (queue_id, script, priority, machine_affinity, status,
           estimated_minutes, supersedes, claim_id, backlog_id, title, note,
           force_rerun, claimed_by_machine, claimed_at, item_json, updated_at)
        VALUES
          (:queue_id, :script, :priority, :machine_affinity, :status,
           :estimated_minutes, :supersedes, :claim_id, :backlog_id, :title,
           :note, :force_rerun, :cb_machine, :cb_at, :item_json, :updated_at)
        ON CONFLICT(queue_id) DO UPDATE SET
           script=excluded.script,
           priority=excluded.priority,
           machine_affinity=excluded.machine_affinity,
           status=excluded.status,
           estimated_minutes=excluded.estimated_minutes,
           supersedes=excluded.supersedes,
           claim_id=excluded.claim_id,
           backlog_id=excluded.backlog_id,
           title=excluded.title,
           note=excluded.note,
           force_rerun=excluded.force_rerun,
           claimed_by_machine=excluded.claimed_by_machine,
           claimed_at=excluded.claimed_at,
           item_json=excluded.item_json,
           updated_at=excluded.updated_at
        """,
        {
            "queue_id": item["queue_id"],
            "script": item.get("script", ""),
            "priority": int(item.get("priority", 1)),
            "machine_affinity": item.get("machine_affinity", "any") or "any",
            "status": status,
            "estimated_minutes": item.get("estimated_minutes"),
            "supersedes": item.get("supersedes"),
            "claim_id": item.get("claim_id"),
            "backlog_id": item.get("backlog_id"),
            "title": item.get("title"),
            "note": item.get("note"),
            "force_rerun": 1 if item.get("force_rerun") else 0,
            "cb_machine": cb_machine,
            "cb_at": cb_at,
            "item_json": json.dumps(item, sort_keys=True),
            "updated_at": utcnow(),
        },
    )


def _parse_utc(value):
    if not value:
        return None
    try:
        ts = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.astimezone(timezone.utc)


def _is_stale(claimed_at, stale_hours):
    dt = _parse_utc(claimed_at)
    if dt is None:
        return False
    return datetime.now(timezone.utc) - dt > timedelta(hours=stale_hours)


def _has_fresh_owner_heartbeat(conn, queue_id, claimed_by_machine,
                               heartbeat_fresh_seconds):
    if not claimed_by_machine or not queue_id or heartbeat_fresh_seconds <= 0:
        return False
    row = conn.execute(
        "SELECT last_seen, current_exq FROM heartbeats WHERE machine=?",
        (claimed_by_machine,),
    ).fetchone()
    if row is None or row["current_exq"] != queue_id:
        return False
    last_seen = _parse_utc(row["last_seen"])
    if last_seen is None:
        return False
    age = datetime.now(timezone.utc) - last_seen
    return age.total_seconds() <= heartbeat_fresh_seconds


def _claim_recoverable(conn, row, queue_id, stale_hours,
                       heartbeat_fresh_seconds):
    if row["status"] != "claimed":
        return False
    if not _is_stale(row["claimed_at"], stale_hours):
        return False
    return not _has_fresh_owner_heartbeat(
        conn, queue_id, row["claimed_by_machine"], heartbeat_fresh_seconds)


def try_claim(conn, queue_id, machine, stale_hours=6,
              heartbeat_fresh_seconds=900):
    """Atomic conditional claim. Returns one of: 'ok', 'already_claimed',
    'error'. Mirrors experiment_runner.attempt_claim() semantics exactly so
    the shadow comparison is apples-to-apples.

    Atomicity: BEGIN IMMEDIATE takes the write lock before the SELECT, so no
    other request can interleave between the eligibility check and the UPDATE.
    """
    now = utcnow()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, machine_affinity, claimed_by_machine, claimed_at "
            "FROM experiments WHERE queue_id=?",
            (queue_id,),
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return "error"
        if not _affinity_ok(row["machine_affinity"], machine):
            conn.execute("ROLLBACK")
            return "already_claimed"
        eligible = row["status"] == "pending" or _claim_recoverable(
            conn, row, queue_id, stale_hours, heartbeat_fresh_seconds)
        if not eligible:
            conn.execute("ROLLBACK")
            return "already_claimed"
        conn.execute(
            "UPDATE experiments SET status='claimed', claimed_by_machine=?, "
            "claimed_at=?, updated_at=? WHERE queue_id=?",
            (machine, now, now, queue_id),
        )
        conn.execute("COMMIT")
        return "ok"
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return "error"


def evaluate_claim(conn, queue_id, machine, stale_hours=6,
                   heartbeat_fresh_seconds=900):
    """Read-only: what verdict WOULD the coordinator return, without
    mutating the mirror. Used by the shadow audit so the comparison does
    not perturb state. Same logic as try_claim()."""
    row = conn.execute(
        "SELECT status, machine_affinity, claimed_by_machine, claimed_at "
        "FROM experiments "
        "WHERE queue_id=?",
        (queue_id,),
    ).fetchone()
    if row is None:
        return "error"
    if not _affinity_ok(row["machine_affinity"], machine):
        return "already_claimed"
    eligible = row["status"] == "pending" or _claim_recoverable(
        conn, row, queue_id, stale_hours, heartbeat_fresh_seconds)
    return "ok" if eligible else "already_claimed"


def apply_git_outcome(conn, queue_id, machine, git_verdict):
    """Keep the mirror aligned with git, which is authoritative in shadow
    mode. On a git 'ok' the machine really holds the claim, so reflect it.
    On already_claimed/error leave the mirror as-is; sync_daemon reconciles
    the full truth (including which other machine holds it) from
    experiment_queue.json each tick."""
    if git_verdict == "ok":
        now = utcnow()
        conn.execute(
            "UPDATE experiments SET status='claimed', claimed_by_machine=?, "
            "claimed_at=?, updated_at=? WHERE queue_id=?",
            (machine, now, now, queue_id),
        )


def release_claim(conn, queue_id, machine):
    """Release this machine's live coordinator claim. Returns (ok, note).

    Only the owning machine can release its claim. This mirrors the runner's
    existing git release path and keeps crash/retry cases from waiting for
    the stale-claim TTL during coordinator-authoritative claiming.
    """
    now = utcnow()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, claimed_by_machine FROM experiments "
            "WHERE queue_id=?", (queue_id,)
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return (False, "queue_id not found")
        if row["status"] != "claimed":
            conn.execute("ROLLBACK")
            return (True, "not claimed")
        if row["claimed_by_machine"] != machine:
            conn.execute("ROLLBACK")
            return (False, "claimed by %s" % row["claimed_by_machine"])
        conn.execute(
            "UPDATE experiments SET status='pending', "
            "claimed_by_machine=NULL, claimed_at=NULL, updated_at=? "
            "WHERE queue_id=?",
            (now, queue_id),
        )
        conn.execute("COMMIT")
        return (True, "released")
    except sqlite3.Error as exc:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return (False, "sqlite error: %s" % exc)


def mark_queue_removed(conn, queue_id, reason):
    """Mark a queue item as completed/removed in coordinator mode.

    Phase 2 still writes the canonical queue removal to git, but marking the
    DB row completed immediately prevents another coordinator-mode worker
    from reclaiming the item if the git queue push lags or fails. The shadow
    sync daemon will delete the row once the item disappears from git.

    `reason` is PERSISTED (removal_reason/removed_at). It used to be accepted
    and silently discarded, which is what made status='completed' ambiguous:
    a runner ERROR, a scientific FAIL and an operator cancellation all wrote
    the same row, so `status='completed' LEFT JOIN results -> no row` counted
    deliberate cancellations as suspected crashes. Callers: the runner sends
    "PASS"/"FAIL"/"ERROR" (coordinator_client.report_queue_remove); an
    operator POSTing /queue/remove may send a free-text reason or none at all
    (NULL is recorded, and is itself informative -- it marks the hand path).
    Storing it does NOT change any status transition or claim behaviour.
    """
    now = utcnow()
    cur = conn.execute(
        "UPDATE experiments SET status='completed', claimed_by_machine=NULL, "
        "claimed_at=NULL, updated_at=?, removal_reason=?, removed_at=? "
        "WHERE queue_id=?",
        (now, reason, now, queue_id),
    )
    return cur.rowcount > 0


def get_queue_status(conn, queue_id):
    """Return the current status string for `queue_id`, or None if the row
    is absent. Used by POST /queue/add to refuse re-adding a queue_id whose
    DB row is already terminal (completed/failed) -- mirrors validate_queue's
    "never re-run a completed id" rule at the coordinator ingress."""
    row = conn.execute(
        "SELECT status FROM experiments WHERE queue_id=?", (queue_id,)
    ).fetchone()
    return row["status"] if row is not None else None


def is_explained_divergence(git_verdict, coord_verdict, detail=""):
    """Harness artifacts that must not block Phase 2 (see SOAK_LOG.md)."""
    d = detail or ""
    if d.startswith("state-reconcile"):
        return True
    # E2: sync_daemon mirrored git claim before shadow report; same machine.
    if git_verdict == "ok" and coord_verdict == "already_claimed":
        return True
    return False


def claim_verdicts_diverge(conn, queue_id, machine, git_verdict,
                           coord_verdict):
    """True only when git and coordinator materially disagree (Phase 2 gate)."""
    if not git_verdict or git_verdict == coord_verdict:
        return False
    if git_verdict == "ok" and coord_verdict == "already_claimed":
        row = conn.execute(
            "SELECT status, claimed_by_machine FROM experiments "
            "WHERE queue_id=?", (queue_id,)).fetchone()
        if (row and row["status"] == "claimed" and
                row["claimed_by_machine"] == machine):
            return False
    return True


def divergence_stats(conn):
    """Raw vs explained vs blocking counts from claim_log."""
    rows = conn.execute(
        "SELECT git_verdict, coord_verdict, detail FROM claim_log "
        "WHERE diverged=1").fetchall()
    raw = len(rows)
    explained = sum(
        1 for r in rows
        if is_explained_divergence(
            r["git_verdict"], r["coord_verdict"], r["detail"]))
    return {"raw": raw, "explained": explained,
            "blocking": raw - explained}


def log_claim(conn, queue_id, machine, git_verdict, coord_verdict, detail=""):
    diverged = 1 if claim_verdicts_diverge(
        conn, queue_id, machine, git_verdict, coord_verdict) else 0
    conn.execute(
        "INSERT INTO claim_log (queue_id, machine, git_verdict, "
        "coord_verdict, diverged, detail, logged_at) VALUES (?,?,?,?,?,?,?)",
        (queue_id, machine, git_verdict, coord_verdict, diverged,
         detail, utcnow()),
    )
    return diverged


def record_result(conn, run_id, queue_id, machine, outcome,
                   manifest_sha256, manifest_bytes):
    """Idempotent on run_id. Returns True if newly recorded, False if a
    prior identical run_id already existed (the partition-straggler case)."""
    existing = conn.execute(
        "SELECT 1 FROM results WHERE run_id=?", (run_id,)
    ).fetchone()
    if existing:
        return False
    conn.execute(
        "INSERT INTO results (run_id, queue_id, machine, outcome, "
        "manifest_sha256, manifest_bytes, received_at) VALUES (?,?,?,?,?,?,?)",
        (run_id, queue_id, machine, outcome, manifest_sha256,
         manifest_bytes, utcnow()),
    )
    return True


def upsert_heartbeat(conn, machine, state, current_exq, progress, gpu,
                     seconds_elapsed=None, seconds_remaining=None,
                     payload_json=None):
    """Upsert per-machine heartbeat row. `payload_json` is the full
    runner-side payload (already JSON-stringified by the caller, or None
    to leave the column unchanged on update / NULL on insert).

    PLAN.md step 6 wires this payload through sync_daemon's
    phase3_heartbeat_writer so the runner can stop git-pushing
    runner_heartbeats/*.json directly. None is the legacy path (old
    runners that don't send the rich payload); the coordinator still
    stores the structured fields and lifecycle_state remains derivable.
    """
    # On update, leave heartbeat_payload_json unchanged when caller passes
    # None (so structured-field-only POSTs from legacy clients don't
    # clobber a payload sent by a newer client on a different tick).
    if payload_json is None:
        conn.execute(
            """
            INSERT INTO heartbeats
              (machine, last_seen, state, current_exq, progress_json,
               gpu_json, seconds_elapsed, seconds_remaining)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(machine) DO UPDATE SET
              last_seen=excluded.last_seen, state=excluded.state,
              current_exq=excluded.current_exq,
              progress_json=excluded.progress_json,
              gpu_json=excluded.gpu_json,
              seconds_elapsed=excluded.seconds_elapsed,
              seconds_remaining=excluded.seconds_remaining
            """,
            (machine, utcnow(), state, current_exq,
             json.dumps(progress or {}), json.dumps(gpu or {}),
             seconds_elapsed, seconds_remaining),
        )
    else:
        conn.execute(
            """
            INSERT INTO heartbeats
              (machine, last_seen, state, current_exq, progress_json,
               gpu_json, seconds_elapsed, seconds_remaining,
               heartbeat_payload_json)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(machine) DO UPDATE SET
              last_seen=excluded.last_seen, state=excluded.state,
              current_exq=excluded.current_exq,
              progress_json=excluded.progress_json,
              gpu_json=excluded.gpu_json,
              seconds_elapsed=excluded.seconds_elapsed,
              seconds_remaining=excluded.seconds_remaining,
              heartbeat_payload_json=excluded.heartbeat_payload_json
            """,
            (machine, utcnow(), state, current_exq,
             json.dumps(progress or {}), json.dumps(gpu or {}),
             seconds_elapsed, seconds_remaining, payload_json),
        )


def record_status_payload(conn, machine, payload_json):
    """Store the runner's full status-file payload for `machine`.

    PLAN.md step 6: replaces experiment_runner.git_push_status as the
    transport for runner_status/<machine>.json. Sync_daemon materialises
    the file from this column on each authoritative tick.

    Like record_shutdown_notice, creates a heartbeat row if none exists
    yet (the runner might post status from a machine that hasn't sent
    a heartbeat tick yet on first boot). Uses the
    _NEVER_HEARTBEATED_SENTINEL for last_seen so lifecycle_state stays
    correct -- status payload is not a heartbeat.
    """
    conn.execute(
        """
        INSERT INTO heartbeats
          (machine, last_seen, status_payload_json)
        VALUES (?,?,?)
        ON CONFLICT(machine) DO UPDATE SET
          status_payload_json=excluded.status_payload_json
        """,
        (machine, _NEVER_HEARTBEATED_SENTINEL, payload_json),
    )


_NEVER_HEARTBEATED_SENTINEL = "1970-01-01T00:00:00Z"


def record_shutdown_notice(conn, machine, reason=None,
                           expected_wake_condition=None):
    """Record an intentional shutdown for `machine`.

    Sets last_shutdown_at to now, plus optional reason and wake-condition.
    Creates a heartbeat row if none exists yet (the scaler workflow can
    announce a shutdown for a machine that hasn't checked in yet, e.g.
    on first provisioning). Idempotent on repeated calls -- each call
    overwrites the prior shutdown notice for that machine.

    Contract for `last_seen`: it represents the most recent HEARTBEAT,
    not "most recent write to this row." A shutdown_notify is not a
    heartbeat (the machine is announcing it's about to be unreachable),
    so:
      - On UPDATE: last_seen is preserved (the existing heartbeat
        timestamp keeps its meaning).
      - On INSERT (fresh row, machine never heartbeated): last_seen is
        set to a sentinel epoch value so lifecycle_state computes
        "gracefully_offline" correctly instead of mistaking the row
        creation for liveness. Sentinel is well below any plausible
        live_threshold_seconds. NOT NULL constraint on last_seen is
        preserved (existing production DBs continue to work without
        a table rebuild).
    """
    now = utcnow()
    conn.execute(
        """
        INSERT INTO heartbeats
          (machine, last_seen, last_shutdown_at, shutdown_reason,
           expected_wake_condition)
        VALUES (?,?,?,?,?)
        ON CONFLICT(machine) DO UPDATE SET
          last_shutdown_at=excluded.last_shutdown_at,
          shutdown_reason=excluded.shutdown_reason,
          expected_wake_condition=excluded.expected_wake_condition
        """,
        (machine, _NEVER_HEARTBEATED_SENTINEL, now, reason,
         expected_wake_condition),
    )


def lifecycle_state(last_seen, last_shutdown_at, *,
                    live_threshold_seconds,
                    stale_after_seconds):
    """Derive {live, gracefully_offline, stale} from heartbeat + shutdown
    timestamps. Pure function; called by /shadow/status read path.

      live                -- heartbeat within live_threshold_seconds.
      gracefully_offline  -- a shutdown_notify newer than the last heartbeat
                             AND within the stale_after_seconds watchdog
                             window. The machine intentionally went away.
      stale               -- no fresh heartbeat AND either no shutdown_notify
                             or it predates the heartbeat OR the watchdog
                             window has expired. Operator should look.

    Both timestamps are ISO-8601 strings or None.
    """
    now = datetime.now(timezone.utc)
    seen_dt = _parse_utc(last_seen)
    shutdown_dt = _parse_utc(last_shutdown_at)

    if seen_dt is not None:
        age = (now - seen_dt).total_seconds()
        if age <= live_threshold_seconds:
            return "live"

    if shutdown_dt is not None:
        # Graceful only when the shutdown is the MOST RECENT event for this
        # machine. If the heartbeat is newer, the machine came back online
        # after the shutdown and is now just stale-running (or stale-dead).
        if seen_dt is None or shutdown_dt >= seen_dt:
            since_shutdown = (now - shutdown_dt).total_seconds()
            if since_shutdown <= stale_after_seconds:
                return "gracefully_offline"

    return "stale"


# ---- command channel (Phase 3 git-command-file migration) ------------------

def insert_command(conn, machine, kind, args_json, issued_by):
    """Insert a pending remote-control command for `machine`. Returns the
    full inserted row as a dict (including the autoincrement id).

    `args_json` is a JSON-encoded string or None. Kind validation is the
    caller's responsibility (the HTTP layer holds the allowlist)."""
    now = utcnow()
    cur = conn.execute(
        "INSERT INTO commands (machine, kind, args, issued_by, issued_at) "
        "VALUES (?,?,?,?,?)",
        (machine, kind, args_json, issued_by, now),
    )
    cmd_id = cur.lastrowid
    row = conn.execute(
        "SELECT id, machine, kind, args, issued_by, issued_at, acked_at, "
        "result_status, result_note FROM commands WHERE id=?", (cmd_id,)
    ).fetchone()
    return dict(row)


def fetch_pending_commands(conn, machine):
    """Return the pending (un-acked) commands for `machine`, oldest first.
    Each row is a dict; `args` stays the raw JSON string (the runner
    decodes it)."""
    rows = conn.execute(
        "SELECT id, kind, args, issued_by, issued_at FROM commands "
        "WHERE machine=? AND acked_at IS NULL ORDER BY id", (machine,)
    ).fetchall()
    return [dict(r) for r in rows]


def ack_command(conn, command_id, machine, result_status, result_note):
    """Mark a command acked with its terminal result. Returns (ok, note).

    Idempotent: a second ack on an already-acked row is a no-op that
    returns (True, 'already acked') so a runner that retries after a
    flaky network does not error. Only the owning machine may ack its
    own command (mirrors release_claim's owner guard)."""
    now = utcnow()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT machine, acked_at FROM commands WHERE id=?",
            (command_id,)).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return (False, "command id not found")
        if row["machine"] != machine:
            conn.execute("ROLLBACK")
            return (False, "command belongs to %s" % row["machine"])
        if row["acked_at"] is not None:
            conn.execute("ROLLBACK")
            return (True, "already acked")
        conn.execute(
            "UPDATE commands SET acked_at=?, result_status=?, result_note=? "
            "WHERE id=?",
            (now, result_status, result_note, command_id),
        )
        conn.execute("COMMIT")
        return (True, "acked")
    except sqlite3.Error as exc:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return (False, "sqlite error: %s" % exc)
