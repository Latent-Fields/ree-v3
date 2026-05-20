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
    return conn


def init_db(db_path):
    conn = connect(db_path)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        conn.executescript(fh.read())
    conn.close()


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
    """
    now = utcnow()
    cur = conn.execute(
        "UPDATE experiments SET status='completed', claimed_by_machine=NULL, "
        "claimed_at=NULL, updated_at=? WHERE queue_id=?",
        (now, queue_id),
    )
    return cur.rowcount > 0


def log_claim(conn, queue_id, machine, git_verdict, coord_verdict, detail=""):
    diverged = 1 if (git_verdict and git_verdict != coord_verdict) else 0
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


def upsert_heartbeat(conn, machine, state, current_exq, progress, gpu):
    conn.execute(
        """
        INSERT INTO heartbeats
          (machine, last_seen, state, current_exq, progress_json, gpu_json)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(machine) DO UPDATE SET
          last_seen=excluded.last_seen, state=excluded.state,
          current_exq=excluded.current_exq,
          progress_json=excluded.progress_json, gpu_json=excluded.gpu_json
        """,
        (machine, utcnow(), state, current_exq,
         json.dumps(progress or {}), json.dumps(gpu or {})),
    )
