"""SQLite layer for the experiment coordinator.

Single-process writer. WAL mode + an explicit BEGIN IMMEDIATE around the
conditional claim UPDATE makes the claim atomic without any external mutex:
SQLite serializes the writer and the WHERE clause is the gate.

All stdout/stderr text is ASCII-only (Windows cp1252 safety).
"""

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

# machine_identity.py lives in ree-v3/, one directory up from coordinator/.
# Same import shim phase3_preflight.py and phase3_verify.py already use.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from machine_identity import canonical_machine_name, same_machine  # noqa: E402


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path):
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    # SQLite's foreign-key enforcement is PER-CONNECTION, not persisted in
    # the file: schema.sql's own `PRAGMA foreign_keys=ON` only ever ran on
    # the one-off connection init_db()'s executescript used, so every
    # ordinary connect() silently had FK enforcement OFF -- the
    # task_claim_resources ON DELETE CASCADE (schema.sql) never actually
    # fired outside that one init call. No pre-existing table declares a
    # FOREIGN KEY, so enabling this here changes nothing for them.
    conn.execute("PRAGMA foreign_keys=ON")
    _migrate_heartbeats(conn)
    _migrate_commands(conn)
    _migrate_experiments(conn)
    _migrate_heartbeat_log(conn)
    _migrate_task_claim_chip_tables(conn)
    _migrate_workspace_state_table(conn)
    _migrate_recommendation_log_table(conn)
    _migrate_igw_log_table(conn)
    _migrate_git_intent_log_table(conn)
    _migrate_dispatcher_leases_table(conn)
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
    if "claim_fence_cleared_at" not in cols:
        conn.execute(
            "ALTER TABLE heartbeats ADD COLUMN claim_fence_cleared_at TEXT")
    # Split out of last_shutdown_at/shutdown_reason so a runner process exit
    # can no longer clobber (and thereby disarm) a machine-shutdown notice.
    # See record_shutdown_notice. Purely additive: existing rows keep NULL,
    # which announced_offline_at and the fence both read as "no process-exit
    # recorded" -- so a live DB picks this up without a rebuild, and rows
    # written before the split (which may carry a process-exit reason in
    # shutdown_reason) stay correct via the exclusion in claim_fence_active.
    if "last_process_exit_at" not in cols:
        conn.execute(
            "ALTER TABLE heartbeats ADD COLUMN last_process_exit_at TEXT")
    if "process_exit_reason" not in cols:
        conn.execute(
            "ALTER TABLE heartbeats ADD COLUMN process_exit_reason TEXT")
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


def _migrate_heartbeat_log(conn):
    """Create the append-only heartbeat_log table if missing.

    Unlike _migrate_heartbeats/_migrate_commands/_migrate_experiments (which
    ALTER an existing table with new columns), this is a brand-new table, so
    there is nothing to gate on -- CREATE TABLE/INDEX IF NOT EXISTS is
    idempotent and cheap on every call. Called from connect() (not just
    init_db()) so a live DB picks it up without a rebuild, mirroring the
    "no rebuild needed" contract the other _migrate_* functions document --
    schema.sql's own CREATE TABLE IF NOT EXISTS only runs inside init_db()'s
    executescript, which fires once at process start.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS heartbeat_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            machine       TEXT NOT NULL,
            observed_at   TEXT NOT NULL,
            state         TEXT,
            current_exq   TEXT,
            payload_json  TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_heartbeat_log_machine_time "
        "ON heartbeat_log(machine, observed_at)"
    )


def _migrate_task_claim_chip_tables(conn):
    """Create the TASK_CLAIMS.json/TASK_CHIPS.json shadow-mirror tables if
    missing: task_claims, task_claim_resources, chip_ledger,
    task_claim_chip_drift_log. See schema.sql for the DDL and the plan doc
    (REE_assembly/evidence/planning/task_claim_chip_coordinator_migration_plan.md
    section 5.2) for the design rationale.

    Brand-new tables, nothing to gate on -- mirrors _migrate_heartbeat_log's
    "CREATE TABLE/INDEX IF NOT EXISTS is idempotent and cheap on every call"
    contract, not the ALTER-an-existing-table pattern the other _migrate_*
    functions use. Called from connect() (not just init_db()) so a live DB
    picks up these tables without a rebuild.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS task_claims (
            session_id                   TEXT NOT NULL,
            claimed_at                   TEXT NOT NULL,
            session_label                TEXT NOT NULL DEFAULT '',
            task                         TEXT NOT NULL DEFAULT '',
            status                       TEXT NOT NULL DEFAULT 'active',
            closed_at                    TEXT,
            completion_note              TEXT,
            completion_note_history_json TEXT,
            spawned_by                   TEXT,
            entry_json                   TEXT NOT NULL,
            last_rendered_json           TEXT,
            updated_at                   TEXT NOT NULL,
            PRIMARY KEY (session_id, claimed_at)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_claims_status "
        "ON task_claims(status)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS task_claim_resources (
            session_id  TEXT NOT NULL,
            claimed_at  TEXT NOT NULL,
            resource    TEXT NOT NULL,
            PRIMARY KEY (session_id, claimed_at, resource),
            FOREIGN KEY (session_id, claimed_at)
                REFERENCES task_claims(session_id, claimed_at) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_claim_resources_resource "
        "ON task_claim_resources(resource)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chip_ledger (
            chip_ref                     TEXT PRIMARY KEY,
            task_id                      TEXT,
            session_id                   TEXT NOT NULL DEFAULT '',
            session_label                TEXT NOT NULL DEFAULT '',
            title                        TEXT NOT NULL DEFAULT '',
            tldr                         TEXT NOT NULL DEFAULT '',
            prompt                       TEXT,
            cwd                          TEXT NOT NULL DEFAULT '',
            origin                       TEXT,
            kind                         TEXT,
            urgency                      INTEGER NOT NULL DEFAULT 0,
            spawned_at                   TEXT NOT NULL DEFAULT '',
            origin_host                  TEXT,
            origin_host_raw              TEXT,
            status                       TEXT NOT NULL DEFAULT 'open',
            claimed_by                   TEXT,
            claimed_at                   TEXT,
            claim_note                   TEXT,
            claimed_host                 TEXT,
            claimed_host_raw             TEXT,
            resolved_at                  TEXT,
            resolved_by_session_id       TEXT,
            resolution_note              TEXT,
            resolution_note_auto         INTEGER NOT NULL DEFAULT 0,
            attached_by_session_id       TEXT,
            attached_at                  TEXT,
            archived_json                TEXT,
            prompt_history_json          TEXT,
            urgency_history_json         TEXT,
            resolution_note_history_json TEXT,
            confirmer_verdict_json       TEXT,
            entry_json                   TEXT NOT NULL,
            last_rendered_json           TEXT,
            updated_at                   TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_chip_ledger_task_id "
        "ON chip_ledger(task_id) WHERE task_id IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chip_ledger_status "
        "ON chip_ledger(status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chip_ledger_claimed "
        "ON chip_ledger(status, claimed_by)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chip_ledger_spawned "
        "ON chip_ledger(status, spawned_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS task_claim_chip_drift_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            checked_at       TEXT NOT NULL,
            source_ref       TEXT,
            n_claims_git     INTEGER NOT NULL,
            n_claims_db      INTEGER NOT NULL,
            n_claims_new     INTEGER NOT NULL,
            n_claims_updated INTEGER NOT NULL,
            n_claims_orphan  INTEGER NOT NULL,
            n_chips_git      INTEGER NOT NULL,
            n_chips_db       INTEGER NOT NULL,
            n_chips_new      INTEGER NOT NULL,
            n_chips_updated  INTEGER NOT NULL,
            n_chips_orphan   INTEGER NOT NULL,
            diverged         INTEGER NOT NULL DEFAULT 0,
            detail           TEXT
        )
        """
    )
    # Additive columns (2026-08-28), guarded by the PRAGMA table_info
    # convention this module uses everywhere -- never a table rebuild.
    #
    # last_rendered_json is the 3-way-merge BASE for the PHASE-2b ingest
    # authority fix: the materializer records, per row, the exact entry_json
    # blob it last rendered into git (and verified/pushed there), so
    # upsert_task_claim/upsert_chip can tell "git unchanged since our render,
    # the DB moved" (preserve the DB) apart from "git moved" (adopt git).
    # NULL = pre-migration / never rendered; both upserts then fall back to
    # the terminal-status conflict guard. The coordinator daemon never reads
    # these columns, so deploy is a git pull -- the writer/shadow-sync are
    # fresh processes per timer tick and run this migration via connect().
    for table in ("task_claims", "chip_ledger"):
        tcols = {r[1] for r in conn.execute(
            "PRAGMA table_info(%s)" % table).fetchall()}
        if tcols and "last_rendered_json" not in tcols:
            conn.execute("ALTER TABLE %s ADD COLUMN last_rendered_json TEXT"
                         % table)
    cols = {r[1] for r in conn.execute(
        "PRAGMA table_info(task_claim_chip_drift_log)").fetchall()}
    if "n_claims_retired" not in cols:
        conn.execute("ALTER TABLE task_claim_chip_drift_log "
                     "ADD COLUMN n_claims_retired INTEGER")
    if "n_chips_retired" not in cols:
        conn.execute("ALTER TABLE task_claim_chip_drift_log "
                     "ADD COLUMN n_chips_retired INTEGER")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_claim_chip_drift_log_diverged "
        "ON task_claim_chip_drift_log(diverged)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_claim_chip_drift_log_checked_at "
        "ON task_claim_chip_drift_log(checked_at)"
    )


def upsert_task_claim(conn, claim, now=None):
    """Insert/refresh one TASK_CLAIMS.json claims[] entry into the mirror,
    plus its `resources` child rows.

    PHASE-2b AUTHORITY (2026-08-28): this is no longer an unconditional
    git-wins overwrite. Under PHASE-1 shadow semantics git was the source of
    truth and this function always adopted the git content; after the
    PHASE-2b cutover the DB is authoritative for coordinator-acked mutations,
    and the old overwrite silently REVERTED any close/renew/amend that landed
    in the DB after the row was last rendered into git (the ingest-clobber
    defect, root-caused live 2026-08-28: a suppressed-git-write close was
    acked, persisted, then clobbered back to active by the next materializer
    ingest of the still-active git row). The fix is a 3-way merge against the
    recorded render base (last_rendered_json, written by the materializer
    after a render provably reached git):

      row is None                     -> INSERT (bootstrap + git-fallback
                                         writers -- unchanged).
      row.entry_json == incoming      -> unchanged no-op (as before).
      base set and incoming == base   -> SKIP: git has not moved since our
                                         render, so the difference is a
                                         DB-side mutation. Preserve the DB.
      base set and row.entry_json ==
        base                          -> ADOPT git: only git moved (a client
                                         git-path/fallback write). This is
                                         the proven self-healing direction
                                         and must be kept.
      else (base NULL [pre-migration]
        or both moved)                -> conflict guard: never downgrade a
                                         terminal DB row ('done') to a
                                         non-terminal git one; otherwise
                                         adopt git.

    Returns (created, changed): created=True on a brand-new
    (session_id, claimed_at) key; changed=True when the stored entry_json
    differs from what is already mirrored (created implies changed). A
    merge SKIP returns (False, False) and writes nothing.
    """
    now = now or utcnow()
    session_id = claim["session_id"]
    claimed_at = claim["claimed_at"]
    # VERBATIM key order, deliberately NOT sort_keys=True (changed 2026-08-28
    # for PHASE-2b): entry_json is what the DB->git materializer renders back
    # into TASK_CLAIMS.json, and D15 (plan doc 5.2.7) requires byte-equality
    # with the file the clients write -- whose per-entry key order is the
    # client's construction order, not alphabetical. json.load preserves file
    # order into this dict, so a plain dumps preserves it into the blob. The
    # reconciler overwrites entry_json from git every tick, so pre-change
    # sorted blobs self-heal on the first tick after this lands (a one-tick
    # n_updated blip; `diverged` counts only orphans and is unaffected).
    entry_json = json.dumps(claim)
    existing = conn.execute(
        "SELECT entry_json, last_rendered_json, status FROM task_claims "
        "WHERE session_id=? AND claimed_at=?",
        (session_id, claimed_at),
    ).fetchone()
    created = existing is None
    changed = created or existing["entry_json"] != entry_json
    if not created and existing["entry_json"] != entry_json:
        base = existing["last_rendered_json"]
        if base is not None and entry_json == base:
            # git unchanged since our render; the DB moved. Preserve it.
            return (False, False)
        if base is not None and existing["entry_json"] == base:
            pass  # only git moved -> adopt git below.
        else:
            # base unknown (pre-migration) or both sides moved: never
            # downgrade a terminal DB row to a non-terminal git one.
            db_status = existing["status"] or "active"
            git_status = claim.get("status") or "active"
            if db_status == "done" and git_status != "done":
                return (False, False)
    history = claim.get("completion_note_history")
    conn.execute(
        """
        INSERT INTO task_claims
          (session_id, claimed_at, session_label, task, status, closed_at,
           completion_note, completion_note_history_json, spawned_by,
           entry_json, updated_at)
        VALUES
          (:session_id, :claimed_at, :session_label, :task, :status,
           :closed_at, :completion_note, :completion_note_history_json,
           :spawned_by, :entry_json, :updated_at)
        ON CONFLICT(session_id, claimed_at) DO UPDATE SET
           session_label=excluded.session_label,
           task=excluded.task,
           status=excluded.status,
           closed_at=excluded.closed_at,
           completion_note=excluded.completion_note,
           completion_note_history_json=excluded.completion_note_history_json,
           spawned_by=excluded.spawned_by,
           entry_json=excluded.entry_json,
           updated_at=excluded.updated_at
        """,
        {
            "session_id": session_id,
            "claimed_at": claimed_at,
            "session_label": claim.get("session_label") or "",
            "task": claim.get("task") or "",
            "status": claim.get("status") or "active",
            "closed_at": claim.get("closed_at"),
            "completion_note": claim.get("completion_note"),
            "completion_note_history_json": (
                json.dumps(history) if history is not None else None),
            "spawned_by": claim.get("spawned_by"),
            "entry_json": entry_json,
            "updated_at": now,
        },
    )
    # Delete + reinsert rather than diff: 372 resource rows total across all
    # 154 live claims (2026-08-26), cheap, and correct by construction --
    # no risk of a stale resource row surviving a claim whose resource list
    # shrank between ticks.
    conn.execute(
        "DELETE FROM task_claim_resources WHERE session_id=? AND claimed_at=?",
        (session_id, claimed_at),
    )
    resources = claim.get("resources") or []
    conn.executemany(
        "INSERT OR IGNORE INTO task_claim_resources "
        "(session_id, claimed_at, resource) VALUES (?,?,?)",
        [(session_id, claimed_at, r) for r in resources],
    )
    return created, changed


def upsert_chip(conn, chip, now=None):
    """Insert/refresh one TASK_CHIPS.json chips[] entry into the mirror.
    Same PHASE-2b 3-way-merge semantics as upsert_task_claim (see its
    docstring for the full rule and the 2026-08-28 ingest-clobber incident
    that motivated it): a DB-side resolve/claim on a row git still carries
    in its pre-mutation state is PRESERVED when git matches the recorded
    render base; a git-path write is still adopted. Terminal guard here is
    CHIP_TERMINAL_STATUSES (done/withdrawn). origin_host/claimed_host are
    canonicalised via machine_identity.canonical_machine_name() for logic,
    mirroring D6 / app.py's existing machine_raw-vs-_canon(machine) split;
    the *_raw columns keep exactly what the JSON reported, for audit.

    Returns (created, changed), same contract as upsert_task_claim. A merge
    SKIP returns (False, False) and writes nothing.
    """
    now = now or utcnow()
    chip_ref = chip["chip_ref"]
    # Verbatim key order -- see upsert_task_claim for why (D15 byte-equality).
    entry_json = json.dumps(chip)
    existing = conn.execute(
        "SELECT entry_json, last_rendered_json, status FROM chip_ledger "
        "WHERE chip_ref=?", (chip_ref,)
    ).fetchone()
    created = existing is None
    changed = created or existing["entry_json"] != entry_json
    if not created and existing["entry_json"] != entry_json:
        base = existing["last_rendered_json"]
        if base is not None and entry_json == base:
            # git unchanged since our render; the DB moved. Preserve it.
            return (False, False)
        if base is not None and existing["entry_json"] == base:
            pass  # only git moved -> adopt git below.
        else:
            db_status = existing["status"] or "open"
            git_status = chip.get("status") or "open"
            if (db_status in CHIP_TERMINAL_STATUSES
                    and git_status not in CHIP_TERMINAL_STATUSES):
                return (False, False)

    def _jd(key):
        val = chip.get(key)
        return json.dumps(val) if val is not None else None

    origin_host_raw = chip.get("origin_host")
    claimed_host_raw = chip.get("claimed_host")
    conn.execute(
        """
        INSERT INTO chip_ledger (
          chip_ref, task_id, session_id, session_label, title, tldr, prompt,
          cwd, origin, kind, urgency, spawned_at, origin_host, origin_host_raw,
          status, claimed_by, claimed_at, claim_note, claimed_host, claimed_host_raw,
          resolved_at, resolved_by_session_id, resolution_note, resolution_note_auto,
          attached_by_session_id, attached_at, archived_json, prompt_history_json,
          urgency_history_json, resolution_note_history_json, confirmer_verdict_json,
          entry_json, updated_at
        ) VALUES (
          :chip_ref, :task_id, :session_id, :session_label, :title, :tldr, :prompt,
          :cwd, :origin, :kind, :urgency, :spawned_at, :origin_host, :origin_host_raw,
          :status, :claimed_by, :claimed_at, :claim_note, :claimed_host, :claimed_host_raw,
          :resolved_at, :resolved_by_session_id, :resolution_note, :resolution_note_auto,
          :attached_by_session_id, :attached_at, :archived_json, :prompt_history_json,
          :urgency_history_json, :resolution_note_history_json, :confirmer_verdict_json,
          :entry_json, :updated_at
        )
        ON CONFLICT(chip_ref) DO UPDATE SET
          task_id=excluded.task_id, session_id=excluded.session_id,
          session_label=excluded.session_label, title=excluded.title,
          tldr=excluded.tldr, prompt=excluded.prompt, cwd=excluded.cwd,
          origin=excluded.origin, kind=excluded.kind, urgency=excluded.urgency,
          spawned_at=excluded.spawned_at, origin_host=excluded.origin_host,
          origin_host_raw=excluded.origin_host_raw, status=excluded.status,
          claimed_by=excluded.claimed_by, claimed_at=excluded.claimed_at,
          claim_note=excluded.claim_note, claimed_host=excluded.claimed_host,
          claimed_host_raw=excluded.claimed_host_raw, resolved_at=excluded.resolved_at,
          resolved_by_session_id=excluded.resolved_by_session_id,
          resolution_note=excluded.resolution_note,
          resolution_note_auto=excluded.resolution_note_auto,
          attached_by_session_id=excluded.attached_by_session_id,
          attached_at=excluded.attached_at, archived_json=excluded.archived_json,
          prompt_history_json=excluded.prompt_history_json,
          urgency_history_json=excluded.urgency_history_json,
          resolution_note_history_json=excluded.resolution_note_history_json,
          confirmer_verdict_json=excluded.confirmer_verdict_json,
          entry_json=excluded.entry_json, updated_at=excluded.updated_at
        """,
        {
            "chip_ref": chip_ref,
            "task_id": chip.get("task_id"),
            "session_id": chip.get("session_id") or "",
            "session_label": chip.get("session_label") or "",
            "title": chip.get("title") or "",
            "tldr": chip.get("tldr") or "",
            "prompt": chip.get("prompt"),
            "cwd": chip.get("cwd") or "",
            "origin": chip.get("origin"),
            "kind": chip.get("kind"),
            "urgency": 1 if chip.get("urgency") else 0,
            "spawned_at": chip.get("spawned_at") or "",
            "origin_host": canonical_machine_name(origin_host_raw) or origin_host_raw,
            "origin_host_raw": origin_host_raw,
            "status": chip.get("status") or "open",
            "claimed_by": chip.get("claimed_by"),
            "claimed_at": chip.get("claimed_at"),
            "claim_note": chip.get("claim_note"),
            "claimed_host": canonical_machine_name(claimed_host_raw) or claimed_host_raw,
            "claimed_host_raw": claimed_host_raw,
            "resolved_at": chip.get("resolved_at"),
            "resolved_by_session_id": chip.get("resolved_by_session_id"),
            "resolution_note": chip.get("resolution_note"),
            "resolution_note_auto": 1 if chip.get("resolution_note_auto") else 0,
            "attached_by_session_id": chip.get("attached_by_session_id"),
            "attached_at": chip.get("attached_at"),
            "archived_json": _jd("archived"),
            "prompt_history_json": _jd("prompt_history"),
            "urgency_history_json": _jd("urgency_history"),
            "resolution_note_history_json": _jd("resolution_note_history"),
            "confirmer_verdict_json": _jd("confirmer_verdict"),
            "entry_json": entry_json,
            "updated_at": now,
        },
    )
    return created, changed


def reconcile_task_claims(conn, claims, now=None):
    """Upsert every claims[] entry from a TASK_CLAIMS.json read into the
    mirror. Returns a stats dict; see task_claim_chip_shadow_sync.py's
    module docstring for how this feeds the drift log.

    `orphans`: (session_id, claimed_at) keys present in the DB, absent from
    `claims`, and still ACTIVE in the mirror. Empty on a healthy mirror -- an
    active claim disappearing from git is either a bug in this reconciler or
    an out-of-band mutation, and is the loss direction that actually matters.

    `retired`: the same absence for a `done` entry. EXPECTED, not drift --
    `scripts/prune_task_claims_done.py` removes done entries older than 24h at
    every `/session-land` close. Counted and reported so the mirror's growth
    stays auditable, but it never raises `diverged`. See the long comment at
    the split below for the measured incident that motivated separating these.
    """
    now = now or utcnow()
    seen = set()
    n_new = n_updated = n_unchanged = 0
    for c in claims:
        session_id = c.get("session_id")
        claimed_at = c.get("claimed_at")
        if not session_id or not claimed_at:
            continue
        seen.add((session_id, claimed_at))
        created, changed = upsert_task_claim(conn, c, now=now)
        if created:
            n_new += 1
        elif changed:
            n_updated += 1
        else:
            n_unchanged += 1
    # A DB key absent from git is NOT automatically drift, and treating it as
    # such is the false positive that saturated the whole PHASE-1 soak.
    #
    # WHAT HAPPENED (2026-08-27/28, found while judging the soak for a cutover
    # go/no-go): this function's own docstring asserted "TASK_CLAIMS.json
    # entries are never deleted (root CLAUDE.md), so an orphan here means
    # either a bug in this reconciler or an out-of-band mutation". That premise
    # is FALSE. `scripts/prune_task_claims_done.py` deletes `done` entries
    # older than 24h, and it runs at EVERY `/session-land` close (Phase 2b) --
    # it is routine, documented, and correct. Measured: prune b6907cce removed
    # 127 entries at 19:02:07Z and the very next tick, 78 seconds later, went
    # diverged and STAYED diverged. All 50 reported orphans were `done` in git
    # immediately before that prune and absent after; zero were unexplained.
    #
    # The consequence was worse than a noisy metric: the exit criterion
    # ("N days of diverged_ticks at 0") became UNMEETABLE, because one routine
    # prune arms it permanently -- and a REAL divergence would then hide behind
    # the same already-raised flag.
    #
    # So the split below keys on exactly the predicate the pruner itself uses:
    #   status='done'   absent from git -> RETIRED. Expected. Not drift.
    #   status='active' absent from git -> ORPHAN. Real drift, and the
    #                   dangerous direction: a live claim vanishing from git is
    #                   precisely the read-modify-write loss this whole
    #                   subsystem exists to detect.
    # Retired keys are counted and reported, never silently dropped.
    rows = conn.execute(
        "SELECT session_id, claimed_at, status FROM task_claims").fetchall()
    existing_keys = {(r["session_id"], r["claimed_at"]) for r in rows}
    status_by_key = {(r["session_id"], r["claimed_at"]): r["status"] for r in rows}
    absent = existing_keys - seen
    orphans = sorted("%s|%s" % k for k in absent
                     if status_by_key.get(k) != "done")
    retired = sorted("%s|%s" % k for k in absent
                     if status_by_key.get(k) == "done")
    return {
        "n_git": len(claims),
        "n_db": len(existing_keys),
        "n_new": n_new,
        "n_updated": n_updated,
        "n_unchanged": n_unchanged,
        "orphans": orphans,
        "retired": retired,
    }


def reconcile_chips(conn, chips, now=None):
    """Upsert every chips[] entry from a TASK_CHIPS.json read into the
    mirror. Same contract as reconcile_task_claims. `orphans` here would
    mean a chip_ref present in the DB but no longer in the git file --
    archiving strips fields but keeps the row (D5), so this should also
    always be empty on a healthy mirror.
    """
    now = now or utcnow()
    seen = set()
    n_new = n_updated = n_unchanged = 0
    for c in chips:
        chip_ref = c.get("chip_ref")
        if not chip_ref:
            continue
        seen.add(chip_ref)
        created, changed = upsert_chip(conn, c, now=now)
        if created:
            n_new += 1
        elif changed:
            n_updated += 1
        else:
            n_unchanged += 1
    existing_refs = {
        r["chip_ref"]
        for r in conn.execute("SELECT chip_ref FROM chip_ledger").fetchall()
    }
    orphans = sorted(existing_refs - seen)
    # NO status split here, and the asymmetry with reconcile_task_claims is
    # deliberate rather than an oversight. Claims get pruned; CHIPS ARE NEVER
    # DELETED. Root CLAUDE.md is emphatic about both halves of that: archiving
    # "strips the fields in place and keeps the row" (D5), and
    # merge_origin_into_local "has no deletion path anywhere in it,
    # deliberately" -- teaching it tombstones is explicitly warned against. So
    # a chip present in the mirror and absent from git has no benign
    # explanation, and softening this to match the claims side would discard
    # the one signal that would catch a chip genuinely going missing.
    # `retired` is present-but-always-empty purely so both stats dicts have
    # one shape for log_task_claim_chip_drift to consume.
    return {
        "n_git": len(chips),
        "n_db": len(existing_refs),
        "n_new": n_new,
        "n_updated": n_updated,
        "n_unchanged": n_unchanged,
        "orphans": orphans,
        "retired": [],
    }


def log_task_claim_chip_drift(conn, source_ref, claim_stats, chip_stats, now=None):
    """Append one row to task_claim_chip_drift_log -- the durable soak
    evidence a human reads to judge PHASE-1's exit criterion ("N days of
    the coordinator's mirrored claim/chip state matching git HEAD with zero
    drift"). Returns the `diverged` flag written (1 if any orphan found).
    """
    now = now or utcnow()
    claim_retired = claim_stats.get("retired") or []
    chip_retired = chip_stats.get("retired") or []
    detail = json.dumps({
        "claim_orphans": claim_stats["orphans"][:50],
        "chip_orphans": chip_stats["orphans"][:50],
        # Reported so a pruned entry stays auditable, but see `diverged` below.
        "claim_retired": claim_retired[:50],
        "chip_retired": chip_retired[:50],
    })
    # `retired` is DELIBERATELY not part of this. A pruned `done` claim is
    # expected, routine and correct; counting it as drift is what made the
    # PHASE-1 exit criterion unmeetable (see reconcile_task_claims).
    diverged = 1 if (claim_stats["orphans"] or chip_stats["orphans"]) else 0
    conn.execute(
        """
        INSERT INTO task_claim_chip_drift_log (
          checked_at, source_ref, n_claims_git, n_claims_db, n_claims_new,
          n_claims_updated, n_claims_orphan, n_claims_retired, n_chips_git,
          n_chips_db, n_chips_new, n_chips_updated, n_chips_orphan,
          n_chips_retired, diverged, detail
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            now, source_ref,
            claim_stats["n_git"], claim_stats["n_db"], claim_stats["n_new"],
            claim_stats["n_updated"], len(claim_stats["orphans"]),
            len(claim_retired),
            chip_stats["n_git"], chip_stats["n_db"], chip_stats["n_new"],
            chip_stats["n_updated"], len(chip_stats["orphans"]),
            len(chip_retired),
            diverged, detail,
        ),
    )
    return diverged


def task_claim_chip_drift_summary(conn, limit=50, since=None):
    """Recent drift_log rows plus aggregate counts, for GET /task_claim/drift
    (app.py) and for a human checking soak status by hand. Mirrors
    divergence_stats()/the /shadow/divergence shape used for the existing
    experiment-claim shadow audit.

    `since` (ISO-8601 UTC) restricts the counts AND the rows to ticks at or
    after that instant, and is what makes the soak's exit criterion actually
    checkable. WHY IT WAS NEEDED: the totals are CUMULATIVE over all history,
    so the 64 diverged ticks produced by the pruned-done false positive
    (2026-08-27T19:03Z onward, see reconcile_task_claims) are permanent. With
    cumulative counts alone, "N days of diverged_ticks at 0" can never read as
    satisfied again no matter how clean the mirror becomes -- the historical
    rows are not wrong and must not be deleted, so the QUESTION has to become
    windowed instead.

    Judging a 24h window is therefore one call:
        GET /task_claim/drift?since=<24h-ago>&limit=200
    and the criterion is `window.diverged_ticks == 0` with a `window
    .total_ticks` consistent with the 10-minute tick cadence (~144 per day).
    That second half matters: zero diverged ticks out of two ticks is not
    evidence of anything, and reporting the window's total alongside is what
    stops a stalled timer reading as a clean soak.

    The unwindowed `total_ticks`/`diverged_ticks` keys are retained verbatim
    so every existing reader keeps working.
    """
    total = conn.execute(
        "SELECT COUNT(*) c FROM task_claim_chip_drift_log").fetchone()["c"]
    diverged = conn.execute(
        "SELECT COUNT(*) c FROM task_claim_chip_drift_log WHERE diverged=1"
    ).fetchone()["c"]
    if since:
        rows = conn.execute(
            "SELECT * FROM task_claim_chip_drift_log WHERE checked_at>=? "
            "ORDER BY id DESC LIMIT ?", (since, limit)).fetchall()
        w_total = conn.execute(
            "SELECT COUNT(*) c FROM task_claim_chip_drift_log "
            "WHERE checked_at>=?", (since,)).fetchone()["c"]
        w_div = conn.execute(
            "SELECT COUNT(*) c FROM task_claim_chip_drift_log "
            "WHERE checked_at>=? AND diverged=1", (since,)).fetchone()["c"]
    else:
        rows = conn.execute(
            "SELECT * FROM task_claim_chip_drift_log ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        w_total, w_div = total, diverged
    out = {
        "total_ticks": total,
        "diverged_ticks": diverged,
        "recent": [dict(r) for r in rows],
    }
    if since:
        out["window"] = {
            "since": since,
            "total_ticks": w_total,
            "diverged_ticks": w_div,
            "clean": w_div == 0,
        }
    return out



# ---------------------------------------------------------------------------
# PHASE-2 MUTATING VERBS for TASK_CLAIMS.json / TASK_CHIPS.json
#
# Everything above this line is PHASE-1: a read-only mirror reconciled from
# git. The functions below are the write path the migration plan
# (REE_assembly/evidence/planning/task_claim_chip_coordinator_migration_plan.md
# section 5.2.4/5.2.5) specifies, so that `task_claim.py` / `chip_ledger.py`
# can put their arbitration on a real transaction instead of a best-effort
# read-then-write against a shared git clone.
#
# WHAT THIS BUYS, precisely: today two sessions on two machines can each read
# TASK_CLAIMS.json, each see no rival, and each write -- the confirmed
# 2026-07-28 three-session collision on runner_remote_control.py (three claims
# inside 84 seconds). Every verb below takes BEGIN IMMEDIATE *before* its
# guard SELECT, exactly as try_claim() does for experiment claims, so there is
# no window for a second caller to interleave between the check and the write.
#
# WHAT IT DOES NOT DO, and this is deliberate: nothing here writes git. The
# DB->git materializer (the analogue of sync_daemon's phase3_*_writer) is NOT
# built. Until it is, a caller running in coordinator mode must still perform
# its own git write -- see scripts/coordinator_transport.py's module docstring
# for why the client keeps writing git in this phase and what has to exist
# before that can stop.
#
# `now` is injected on every verb, matching this module's existing convention
# (_is_stale, machine_departed, lifecycle_state all take an explicit now), so
# every test below is time-independent.
#
# ASCII-only in all returned strings, per this module's header.
# ---------------------------------------------------------------------------

TASK_CLAIM_STALE_HOURS_DEFAULT = 6.0
CHIP_CLAIM_STALE_HOURS_DEFAULT = 6.0


def _is_stale_at(claimed_at, stale_hours, now=None):
    """_is_stale() with an injectable clock. The existing _is_stale reads
    datetime.now() directly, which is fine for the experiment-claim path
    (whose tests backdate rows by SQL) but not for arbitration verdicts a
    caller wants to assert deterministically."""
    dt = _parse_utc(claimed_at)
    if dt is None:
        return False
    ref = _parse_utc(now) if now else datetime.now(timezone.utc)
    if ref is None:
        ref = datetime.now(timezone.utc)
    return ref - dt > timedelta(hours=float(stale_hours))


def _is_scope_resource(resource):
    """A directory-shaped resource is a SCOPE claim and is never arbitrated.

    Mirrors task_claim.py's is_scope_resource() and the "Conflict resolution"
    rule in root CLAUDE.md: a verdict fires only on an exact match of a
    FILE-shaped resource. governance.sh holds REE_assembly/evidence/ for the
    length of a regen and fails open on a non-zero `open`, so arbitrating a
    directory would both stop every evidence session and leave every regen
    unprotected. Directory overlaps are reported as a NOTE by the caller.
    """
    return str(resource).endswith("/")


def _claim_entry_json(session_id, claimed_at, session_label, task, resources,
                      status="active", spawned_by=None, closed_at=None,
                      completion_note=None, completion_note_history=None):
    """The lossless entry_json for a task_claims row.

    Field ORDER and field SET both matter: this is what the (not-yet-built)
    materializer will write back into TASK_CLAIMS.json, and it is what the
    PHASE-1 reconciler compares against to decide `diverged`. It therefore
    mirrors task_claim.py cmd_open()'s literal entry dict exactly -- same six
    keys in the same order for an open claim, with the closure fields appended
    only once they exist, which is what `close` does to the JSON today.
    """
    entry = {
        "session_id": session_id,
        "session_label": session_label or "",
        "claimed_at": claimed_at,
        "task": task or "",
        "resources": list(resources or []),
        "status": status,
    }
    if closed_at is not None:
        entry["closed_at"] = closed_at
    if completion_note is not None:
        entry["completion_note"] = completion_note
    if completion_note_history:
        entry["completion_note_history"] = list(completion_note_history)
    if spawned_by is not None:
        entry["spawned_by"] = spawned_by
    return entry


def _carry_unmodelled(stored_json, rebuilt):
    """Carry forward any key the columns do not model.

    Section 5.2.1 of the plan promises entry_json is LOSSLESS -- "a field this
    schema does not model explicitly still round-trips" -- and that promise is
    what makes the migration safe against every existing consumer. A rebuild
    that only emits the modelled columns silently breaks it.

    This is not hypothetical. Measured on the live TASK_CHIPS.json 2026-08-27:
    **198 of 1920 chips carry `handoff_pending`**, which has no column here
    (see chip_ledger.py's "HANDOFF-PENDING" section -- it is set by `resolve
    --handoff-pending` / `declare-handoff`). Before this helper, claiming or
    resolving one of those chips through the coordinator would have rebuilt its
    entry_json WITHOUT that key, and the field would have vanished from the
    materialized JSON with no error anywhere.

    The rebuilt dict WINS on every key it produces -- it is derived from the
    columns, which are what the verbs actually updated. Only keys absent from
    it are taken from the stored blob.

    CALL IT BEFORE ANY DELIBERATE REMOVAL, never after. _chip_entry_from_row
    calls it ahead of its archived-field pop for exactly this reason: run the
    other way round it resurrects a stripped `prompt` out of a stale blob,
    which is the archive-undo D5 forbids -- and that is not a hypothetical
    ordering worry, it is what
    test_archived_fields_are_absent_not_null_in_entry_json caught on the first
    attempt at this fix.
    """
    if not stored_json:
        return rebuilt
    try:
        stored = json.loads(stored_json)
    except (TypeError, ValueError):
        return rebuilt
    if not isinstance(stored, dict):
        return rebuilt
    out = dict(rebuilt)
    for key, val in stored.items():
        if key not in out:
            out[key] = val
    return out


def _reserialise_claim_row(conn, session_id, claimed_at):
    """Rebuild entry_json from the row + its resource child rows, and store it.

    Called at the end of every claim mutation. Keeping entry_json derived
    rather than hand-patched at each call site is what stops the lossless
    column drifting away from the columns -- the failure the PHASE-1
    reconciler would then report forever as drift.
    """
    row = conn.execute(
        "SELECT * FROM task_claims WHERE session_id=? AND claimed_at=?",
        (session_id, claimed_at),
    ).fetchone()
    if row is None:
        return None
    resources = [r["resource"] for r in conn.execute(
        "SELECT resource FROM task_claim_resources "
        "WHERE session_id=? AND claimed_at=? ORDER BY rowid",
        (session_id, claimed_at)).fetchall()]
    history = None
    if row["completion_note_history_json"]:
        try:
            history = json.loads(row["completion_note_history_json"])
        except (TypeError, ValueError):
            history = None
    entry = _claim_entry_json(
        row["session_id"], row["claimed_at"], row["session_label"],
        row["task"], resources, status=row["status"],
        spawned_by=row["spawned_by"], closed_at=row["closed_at"],
        completion_note=row["completion_note"],
        completion_note_history=history)
    entry = _carry_unmodelled(row["entry_json"], entry)
    # Verbatim key order (_claim_entry_json's construction order IS the
    # client's) -- see upsert_task_claim for why (D15 byte-equality).
    conn.execute(
        "UPDATE task_claims SET entry_json=? WHERE session_id=? AND claimed_at=?",
        (json.dumps(entry), session_id, claimed_at))
    return entry


def try_open_task_claim(conn, session_id, session_label, task, resources,
                        allow_overlap=False, spawned_by=None, claimed_at=None,
                        stale_hours=TASK_CLAIM_STALE_HOURS_DEFAULT, now=None):
    """Atomic claim-open with resource arbitration.

    Returns (verdict, payload):
      'ok'             -- claim written; caller owns every named resource
      'idempotent'     -- this session already holds an active claim; nothing
                          written. Preserves task_claim.py's documented
                          per-session_id idempotency (plan doc D8): the git
                          path retries routinely under contention, and a naive
                          INSERT would mint a second row on every retry.
      'owned_by_other' -- a live rival owns at least one named FILE resource;
                          nothing written. payload['rivals'] carries the owner
                          rows so the CLI can render today's exit-3 text.
      'error'          -- sqlite failure; nothing written.

    `claimed_at` is CLIENT-SUPPLIED and optional. The plan's original
    signature had the server stamp it. That is wrong for this phase and the
    reason is concrete: while the client still performs its own git write
    (see the block comment above), a server-stamped claimed_at would give the
    DB row and the JSON entry DIFFERENT halves of the (session_id, claimed_at)
    primary key, and the PHASE-1 reconciler would report that as permanent
    drift on every tick. The client sends the stamp it is about to write to
    git; the server falls back to `now` only when none is given. Recorded as
    D12 in the plan doc.

    Atomicity: BEGIN IMMEDIATE takes the write lock BEFORE the rival SELECT,
    so no other request -- on any machine -- can interleave between the
    arbitration check and the INSERT.
    """
    now = now or utcnow()
    stamp = claimed_at or now
    resources = list(resources or [])
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT claimed_at FROM task_claims "
            "WHERE session_id=? AND status='active' ORDER BY claimed_at",
            (session_id,),
        ).fetchone()
        if existing is not None:
            conn.execute("ROLLBACK")
            return ("idempotent", {"claimed_at": existing["claimed_at"]})

        rivals = []
        notes = []
        files = [r for r in resources if not _is_scope_resource(r)]
        scopes = [r for r in resources if _is_scope_resource(r)]
        if scopes:
            notes.append(
                "%d directory-scope resource(s) not arbitrated: %s"
                % (len(scopes), ", ".join(sorted(scopes))))
        if files:
            q = ",".join("?" * len(files))
            candidates = conn.execute(
                "SELECT c.session_id, c.claimed_at, c.session_label, "
                "       c.task, r.resource "
                "FROM task_claim_resources r "
                "JOIN task_claims c ON c.session_id=r.session_id "
                "                  AND c.claimed_at=r.claimed_at "
                "WHERE r.resource IN (%s) AND c.status='active' "
                "  AND c.session_id<>? "
                "ORDER BY c.claimed_at" % q,
                files + [session_id],
            ).fetchall()
            for r in candidates:
                if _is_stale_at(r["claimed_at"], stale_hours, now):
                    continue
                rivals.append(dict(r))
        if rivals and not allow_overlap:
            conn.execute("ROLLBACK")
            return ("owned_by_other", {"rivals": rivals, "notes": notes})
        if rivals:
            notes.append(
                "allow_overlap: %d live rival claim(s) on a shared file were "
                "downgraded to a note" % len(rivals))

        entry = _claim_entry_json(session_id, stamp, session_label, task,
                                  resources, spawned_by=spawned_by)
        conn.execute(
            "INSERT INTO task_claims (session_id, claimed_at, session_label, "
            " task, status, spawned_by, entry_json, updated_at) "
            "VALUES (?,?,?,?, 'active', ?, ?, ?)",
            (session_id, stamp, session_label or "", task or "", spawned_by,
             json.dumps(entry), now),
        )
        conn.executemany(
            "INSERT OR IGNORE INTO task_claim_resources "
            "(session_id, claimed_at, resource) VALUES (?,?,?)",
            [(session_id, stamp, r) for r in resources],
        )
        conn.execute("COMMIT")
        return ("ok", {"claimed_at": stamp, "rivals": rivals, "notes": notes})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def _select_one_claim(conn, session_id, claimed_at, status):
    """Resolve (session_id, claimed_at?) to exactly one row, or explain why not.

    Returns (row, verdict, payload). verdict is None on success, else
    'not_found' or 'ambiguous'. Mirrors task_claim.py's documented refusal:
    `close`/`amend`/`renew` NEVER guess when a session_id owns more than one
    candidate -- they refuse and print the candidate stamps, because a headless
    worker cannot go and read the JSON. `status` narrows the candidate set the
    same way each CLI verb does (active for renew, done for amend, either for
    close's own two-step).
    """
    params = [session_id]
    sql = "SELECT * FROM task_claims WHERE session_id=?"
    if status is not None:
        sql += " AND status=?"
        params.append(status)
    if claimed_at:
        sql += " AND claimed_at=?"
        params.append(claimed_at)
    sql += " ORDER BY claimed_at"
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return (None, "not_found", {"session_id": session_id,
                                    "claimed_at": claimed_at,
                                    "status": status})
    if len(rows) > 1:
        return (None, "ambiguous", {"candidates": [
            {"claimed_at": r["claimed_at"], "task": r["task"],
             "status": r["status"]} for r in rows]})
    return (rows[0], None, {})


def close_task_claim(conn, session_id, closed_at, completion_note,
                     claimed_at=None, now=None):
    """Flip one active claim to done. Verdicts: 'ok' | 'not_found' |
    'ambiguous' | 'already_closed' | 'error'.

    closed_at and completion_note are BOTH required by the caller (root
    CLAUDE.md: a bare status:done is indistinguishable from an abandoned
    claim). This function does not re-derive closed_at from anything -- the
    CLI resolves it from the landing commit's committer date and sends it.
    """
    now = now or utcnow()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row, verdict, payload = _select_one_claim(conn, session_id, claimed_at,
                                                  "active")
        if verdict == "not_found":
            # Distinguish "no such claim" from "already done" -- the CLI's own
            # message for the second is different, and reporting a completed
            # close as not_found would send a caller re-opening a claim.
            row2, v2, _ = _select_one_claim(conn, session_id, claimed_at, "done")
            conn.execute("ROLLBACK")
            if v2 is None:
                return ("already_closed", {"claimed_at": row2["claimed_at"],
                                           "closed_at": row2["closed_at"]})
            return (verdict, payload)
        if verdict is not None:
            conn.execute("ROLLBACK")
            return (verdict, payload)
        conn.execute(
            "UPDATE task_claims SET status='done', closed_at=?, "
            "completion_note=?, updated_at=? "
            "WHERE session_id=? AND claimed_at=?",
            (closed_at, completion_note, now, session_id, row["claimed_at"]))
        entry = _reserialise_claim_row(conn, session_id, row["claimed_at"])
        conn.execute("COMMIT")
        return ("ok", {"claimed_at": row["claimed_at"], "entry": entry})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def renew_task_claim(conn, session_id, claimed_at=None, new_claimed_at=None,
                     stale_hours=TASK_CLAIM_STALE_HOURS_DEFAULT, now=None):
    """Re-stamp claimed_at to now on this session's own ACTIVE claim so it does
    not silently fall out of staleness protection.

    Verdicts: 'ok' | 'not_found' | 'ambiguous' | 'would_lose_ownership' |
    'error'.

    'would_lose_ownership' mirrors task_claim.py cmd_renew's refusal: renewing
    must not flip arbitration ownership to a live rival. Under a composite PK
    the re-stamp is a key change, so the new stamp is an INSERT and the child
    resource rows are copied to it inside the one transaction.

    THE OLD-STAMP ROW IS TOMBSTONE-CLOSED, NOT DELETED (2026-08-28, part of
    the ingest-authority fix). A DELETE cannot survive the materializer's
    ingest: git still carries the old-stamp row as active until the next
    render lands, and a deleted row has neither an entry to merge against nor
    a base -- ingest would simply re-INSERT it, resurrecting the phantom
    old-stamp row the renew mirror exists to remove. Closing it as done
    ("renewed: ...") instead lets the 3-way merge / terminal guard in
    upsert_task_claim PRESERVE the closure against stale-git ingest, and the
    render's 24h retention then ages it out -- the same lifecycle as any
    other done row. Its child resource rows stay put: arbitration queries
    filter on status='active', so a done row's resources never contend.
    """
    now = now or utcnow()
    stamp = new_claimed_at or now
    try:
        conn.execute("BEGIN IMMEDIATE")
        row, verdict, payload = _select_one_claim(conn, session_id, claimed_at,
                                                  "active")
        if verdict is not None:
            conn.execute("ROLLBACK")
            return (verdict, payload)
        old_stamp = row["claimed_at"]
        resources = [r["resource"] for r in conn.execute(
            "SELECT resource FROM task_claim_resources "
            "WHERE session_id=? AND claimed_at=? ORDER BY rowid",
            (session_id, old_stamp)).fetchall()]
        files = [r for r in resources if not _is_scope_resource(r)]
        if files:
            q = ",".join("?" * len(files))
            rivals = [dict(r) for r in conn.execute(
                "SELECT c.session_id, c.claimed_at, c.session_label, c.task, "
                "       r.resource "
                "FROM task_claim_resources r "
                "JOIN task_claims c ON c.session_id=r.session_id "
                "                  AND c.claimed_at=r.claimed_at "
                "WHERE r.resource IN (%s) AND c.status='active' "
                "  AND c.session_id<>?" % q,
                files + [session_id]).fetchall()]
            live = [r for r in rivals
                    if not _is_stale_at(r["claimed_at"], stale_hours, now)]
            # A rival that is CURRENTLY older than us but would become the
            # owner once we re-stamp forward is exactly what cmd_renew refuses.
            losing = [r for r in live if r["claimed_at"] < stamp]
            if losing:
                conn.execute("ROLLBACK")
                return ("would_lose_ownership", {"rivals": losing})
        conn.execute(
            "INSERT INTO task_claims (session_id, claimed_at, session_label, "
            " task, status, closed_at, completion_note, "
            " completion_note_history_json, spawned_by, entry_json, updated_at) "
            "SELECT session_id, ?, session_label, task, status, closed_at, "
            "       completion_note, completion_note_history_json, spawned_by, "
            "       entry_json, ? FROM task_claims "
            "WHERE session_id=? AND claimed_at=?",
            (stamp, now, session_id, old_stamp))
        conn.executemany(
            "INSERT OR IGNORE INTO task_claim_resources "
            "(session_id, claimed_at, resource) VALUES (?,?,?)",
            [(session_id, stamp, r) for r in resources])
        # Renewal provenance on the NEW row (mirrors cmd_renew's git-path
        # bookkeeping): original_claimed_at + renewal_history ride the
        # lossless entry_json as unmodelled keys.
        try:
            carried = json.loads(row["entry_json"]) if row["entry_json"] else {}
        except (TypeError, ValueError):
            carried = {}
        if isinstance(carried, dict):
            carried.setdefault("original_claimed_at", old_stamp)
            carried.setdefault("renewal_history", []).append(
                {"renewed_at": stamp, "previous_claimed_at": old_stamp})
            conn.execute(
                "UPDATE task_claims SET entry_json=? "
                "WHERE session_id=? AND claimed_at=?",
                (json.dumps(carried), session_id, stamp))
        # Tombstone-close the old stamp (never DELETE -- see docstring).
        conn.execute(
            "UPDATE task_claims SET status='done', closed_at=?, "
            "completion_note=?, updated_at=? "
            "WHERE session_id=? AND claimed_at=?",
            (now, "renewed: claimed_at re-stamped to %s" % stamp, now,
             session_id, old_stamp))
        _reserialise_claim_row(conn, session_id, old_stamp)
        entry = _reserialise_claim_row(conn, session_id, stamp)
        conn.execute("COMMIT")
        return ("ok", {"claimed_at": stamp, "previous_claimed_at": old_stamp,
                       "entry": entry})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def amend_task_claim(conn, session_id, completion_note, claimed_at=None,
                     now=None):
    """Correct completion_note on an already-CLOSED claim; nothing else moves.

    Verdicts: 'ok' | 'not_found' | 'ambiguous' | 'error'.

    The prior text is pushed onto completion_note_history_json, never dropped
    -- same discipline as the CLI, so an amendment cannot silently erase what
    was originally claimed. closed_at/status/task/resources keep the CLI's
    refusal to move: they record a landing and must not drift.
    """
    now = now or utcnow()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row, verdict, payload = _select_one_claim(conn, session_id, claimed_at,
                                                  "done")
        if verdict is not None:
            conn.execute("ROLLBACK")
            return (verdict, payload)
        history = []
        if row["completion_note_history_json"]:
            try:
                history = json.loads(row["completion_note_history_json"]) or []
            except (TypeError, ValueError):
                history = []
        if row["completion_note"] is not None:
            history.append(row["completion_note"])
        conn.execute(
            "UPDATE task_claims SET completion_note=?, "
            "completion_note_history_json=?, updated_at=? "
            "WHERE session_id=? AND claimed_at=?",
            (completion_note, json.dumps(history), now, session_id,
             row["claimed_at"]))
        entry = _reserialise_claim_row(conn, session_id, row["claimed_at"])
        conn.execute("COMMIT")
        return ("ok", {"claimed_at": row["claimed_at"], "entry": entry})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def dedupe_task_claim(conn, session_id, claimed_at=None, now=None):
    """Accepted NO-OP. See plan doc D3.

    `dedupe` removes claim entries byte-identical to another entry under one
    session_id. Under the (session_id, claimed_at) PRIMARY KEY that duplicate
    class CANNOT BE CREATED -- the second INSERT fails -- so the 2026-08-18
    byte-identical-duplicate incident is prevented at the source rather than
    repaired after the fact. The verb is kept so existing call sites and tests
    do not break; its logic is deliberately NOT ported.
    """
    return ("ok", {"removed": 0,
                   "note": "not applicable under the composite primary key"})


# ---- chips ----------------------------------------------------------------


def _chip_entry_from_row(row):
    """Rebuild the TASK_CHIPS.json chips[] entry from a chip_ledger row.

    Same role as _reserialise_claim_row: entry_json is the lossless column the
    materializer and the PHASE-1 reconciler both read, so it must be derived
    from the columns rather than hand-patched per call site.

    ARCHIVED FIELDS ARE PRESERVED AS ABSENT, NOT AS NULL, and that is
    load-bearing (plan doc D5): chip_ledger.archived_field() distinguishes
    "this chip never had a prompt" from "its prompt was archived", and it does
    so by reading the `archived` block, not by probing for a NULL. A row whose
    prompt was stripped keeps archived_json set and simply omits the key.
    """
    def _jl(value, default=None):
        if value is None:
            return default
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default

    archived = _jl(row["archived_json"])
    archived_fields = set((archived or {}).get("fields") or [])
    entry = {
        "task_id": row["task_id"],
        "chip_ref": row["chip_ref"],
        "origin": row["origin"],
        "kind": row["kind"],
        "urgency": bool(row["urgency"]),
        "session_id": row["session_id"],
        "session_label": row["session_label"],
        "title": row["title"],
        "tldr": row["tldr"],
        "prompt": row["prompt"],
        "cwd": row["cwd"],
        "origin_host": row["origin_host_raw"],
        "spawned_at": row["spawned_at"],
        "status": row["status"],
        "resolved_at": row["resolved_at"],
        "resolved_by_session_id": row["resolved_by_session_id"],
        "resolution_note": row["resolution_note"],
        "resolution_note_auto": bool(row["resolution_note_auto"]),
        "claimed_by": row["claimed_by"],
        "claimed_at": row["claimed_at"],
        "claim_note": row["claim_note"],
        "claimed_host": row["claimed_host_raw"],
        "attached_by_session_id": row["attached_by_session_id"],
        "attached_at": row["attached_at"],
    }
    # Carry unmodelled keys BEFORE the archived-field pop, never after: the
    # pop must win. Resurrecting a stripped `prompt` from a stale blob is
    # exactly the archive-undo D5 forbids.
    entry = _carry_unmodelled(row["entry_json"], entry)
    for key in archived_fields:
        entry.pop(key, None)
    if archived is not None:
        entry["archived"] = archived
    for col, key in (("prompt_history_json", "prompt_history"),
                     ("urgency_history_json", "urgency_history"),
                     ("resolution_note_history_json", "resolution_note_history"),
                     ("confirmer_verdict_json", "confirmer_verdict")):
        val = _jl(row[col])
        if val is not None:
            entry[key] = val
    return entry


def _reserialise_chip_row(conn, chip_ref):
    row = conn.execute("SELECT * FROM chip_ledger WHERE chip_ref=?",
                       (chip_ref,)).fetchone()
    if row is None:
        return None
    entry = _chip_entry_from_row(row)
    # Verbatim key order -- see upsert_task_claim for why (D15 byte-equality).
    conn.execute("UPDATE chip_ledger SET entry_json=? WHERE chip_ref=?",
                 (json.dumps(entry), chip_ref))
    return entry


def _find_chip_row(conn, chip_ref=None, task_id=None):
    """chip_ref first, then task_id through the partial unique index.

    task_id CANNOT be an identifier on its own (plan doc D4): it is NULL on
    1043/1692 live rows, because a chip only gets one if spawn_task actually
    minted it. Callers may pass either; a task_id lookup that finds nothing
    404s cleanly rather than matching some NULL-task_id row.
    """
    if chip_ref:
        return conn.execute("SELECT * FROM chip_ledger WHERE chip_ref=?",
                            (chip_ref,)).fetchone()
    if task_id:
        return conn.execute(
            "SELECT * FROM chip_ledger WHERE task_id=? AND task_id IS NOT NULL",
            (task_id,)).fetchone()
    return None


def record_chip(conn, chip, now=None):
    """Append one chip. Verdicts: 'ok' | 'idempotent' | 'ref_collision' |
    'missing_marker' | 'error'.

    'missing_marker': `prompt` must literally contain "[chip_ref: <ref>]".
    This is a REFUSAL, not a warning, and mirrors the CLI's own hard failure
    since 2026-08-03 -- a chip whose stored prompt lacks the marker can never
    self-report at close, and 12 such chips were recorded in one governance
    session while it was only a warning.

    'idempotent' covers the tick shape the CLI already handles: a routine
    re-scanning its backlog calls record with the SAME deterministic chip_ref
    every cycle, and that must be a no-op rather than a failure. Restricted,
    exactly as the CLI restricts it, to the case where BOTH the stored row and
    the incoming chip have no task_id -- a same-ref/different-task_id
    collision is a genuine bug (two distinct spawn_task calls reusing a ref)
    and still refuses.
    """
    now = now or utcnow()
    chip_ref = chip.get("chip_ref")
    if not chip_ref:
        return ("error", {"reason": "chip_ref is required"})
    marker = "[chip_ref: %s]" % chip_ref
    prompt = chip.get("prompt") or ""
    if marker not in prompt:
        return ("missing_marker", {"chip_ref": chip_ref, "marker": marker})
    task_id = chip.get("task_id")
    try:
        conn.execute("BEGIN IMMEDIATE")
        if task_id:
            by_task = conn.execute(
                "SELECT chip_ref FROM chip_ledger "
                "WHERE task_id=? AND task_id IS NOT NULL", (task_id,)).fetchone()
            if by_task is not None:
                conn.execute("ROLLBACK")
                return ("idempotent", {"chip_ref": by_task["chip_ref"],
                                       "reason": "task_id already recorded"})
        clash = conn.execute("SELECT task_id, origin FROM chip_ledger "
                             "WHERE chip_ref=?", (chip_ref,)).fetchone()
        if clash is not None:
            conn.execute("ROLLBACK")
            if task_id is None and clash["task_id"] is None:
                return ("idempotent", {"chip_ref": chip_ref,
                                       "origin": clash["origin"],
                                       "reason": "chip_ref already recorded"})
            return ("ref_collision", {"chip_ref": chip_ref,
                                      "task_id": clash["task_id"]})
        upsert_chip(conn, chip, now=now)
        conn.execute("COMMIT")
        return ("ok", {"chip_ref": chip_ref})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def try_claim_chip(conn, chip_ref=None, task_id=None, claimed_by=None,
                   claimed_host=None, note=None,
                   stale_hours=CHIP_CLAIM_STALE_HOURS_DEFAULT,
                   claimed_at=None, now=None):
    """Atomic chip claim -- the cross-host dispatch MUTEX.

    Verdicts: 'ok' | 'not_found' | 'not_open' | 'already_claimed' | 'error'.
    payload['note_prefix'] carries the CLI's own explanatory prefix
    ("refreshed own claim", "previous claim ... superseded") so the client can
    print the identical line it prints today.

    This is the verb with the most to gain from the migration. Today's claim
    is a git write whose mutual exclusion depends on push ordering across
    hosts, which is why chip_ledger.py has to verify the claim reached origin
    afterwards (verify_claim_on_origin) and why the 2026-08-09 double-dispatch
    happened when a claim stayed on a local branch. Here the SELECT and the
    UPDATE are inside one BEGIN IMMEDIATE, so exactly one of N simultaneous
    claimants wins and the loser is told immediately.
    """
    now = now or utcnow()
    stamp = claimed_at or now
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _find_chip_row(conn, chip_ref=chip_ref, task_id=task_id)
        if row is None:
            conn.execute("ROLLBACK")
            return ("not_found", {"chip_ref": chip_ref, "task_id": task_id})
        if row["status"] != "open":
            conn.execute("ROLLBACK")
            return ("not_open", {"chip_ref": row["chip_ref"],
                                 "status": row["status"]})
        existing = row["claimed_by"]
        note_prefix = ""
        if existing and existing != claimed_by:
            age_stale = _is_stale_at(row["claimed_at"], stale_hours, now)
            if row["claimed_at"] and not age_stale:
                conn.execute("ROLLBACK")
                return ("already_claimed", {
                    "chip_ref": row["chip_ref"],
                    "claimed_by": existing,
                    "claimed_at": row["claimed_at"],
                    "stale_after_hours": stale_hours})
            reason = ("unparseable/missing claimed_at" if not row["claimed_at"]
                      else "past the %.1fh stale threshold" % float(stale_hours))
            note_prefix = ("previous claim by %s (%s) superseded -- "
                           % (existing, reason))
        elif existing == claimed_by:
            note_prefix = "refreshed own claim -- "
        canon_host = canonical_machine_name(claimed_host) or claimed_host
        conn.execute(
            "UPDATE chip_ledger SET claimed_by=?, claimed_at=?, claim_note=?, "
            "claimed_host=?, claimed_host_raw=?, updated_at=? WHERE chip_ref=?",
            (claimed_by, stamp, note or "", canon_host, claimed_host, now,
             row["chip_ref"]))
        entry = _reserialise_chip_row(conn, row["chip_ref"])
        conn.execute("COMMIT")
        return ("ok", {"chip_ref": row["chip_ref"], "claimed_at": stamp,
                       "note_prefix": note_prefix, "entry": entry})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def unclaim_chip(conn, chip_ref=None, task_id=None, note=None, now=None):
    """Release a chip claim. Verdicts: 'ok' | 'not_found' | 'error'.
    payload['was_claimed_by'] is None when the chip was not claimed -- the
    CLI's "nothing to release" case, which is a success, not a failure."""
    now = now or utcnow()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _find_chip_row(conn, chip_ref=chip_ref, task_id=task_id)
        if row is None:
            conn.execute("ROLLBACK")
            return ("not_found", {"chip_ref": chip_ref, "task_id": task_id})
        was = row["claimed_by"]
        conn.execute(
            "UPDATE chip_ledger SET claimed_by=NULL, claimed_at=NULL, "
            "claim_note=?, claimed_host=NULL, claimed_host_raw=NULL, "
            "updated_at=? WHERE chip_ref=?",
            (note or "", now, row["chip_ref"]))
        entry = _reserialise_chip_row(conn, row["chip_ref"])
        conn.execute("COMMIT")
        return ("ok", {"chip_ref": row["chip_ref"], "was_claimed_by": was,
                       "entry": entry})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


CHIP_TERMINAL_STATUSES = ("done", "withdrawn")


def resolve_chip(conn, status, chip_ref=None, task_id=None, note=None,
                 resolved_by_session_id=None, note_auto=False, force=False,
                 resolved_at=None, now=None):
    """Mark a chip done/withdrawn.

    Verdicts: 'ok' | 'not_found' | 'bad_status' | 'terminal_conflict' | 'error'.
    payload['changed'] is False when the chip was ALREADY at that status
    (plan doc D11): today that is a SILENT no-op indistinguishable from a real
    transition, which is exactly the trap a headless worker falls into when
    its report is dropped. Reporting changed=False lets the CLI say so.

    'terminal_conflict' mirrors the CLI's 2026-08-25 guard: a chip already at
    a DIFFERENT terminal status is not overwritten without force, and even
    with force the prior status/resolved_at/resolver/note is preserved in
    resolution_note_history rather than dropped.
    """
    now = now or utcnow()
    stamp = resolved_at or now
    if status not in CHIP_TERMINAL_STATUSES:
        return ("bad_status", {"status": status,
                               "valid": list(CHIP_TERMINAL_STATUSES)})
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _find_chip_row(conn, chip_ref=chip_ref, task_id=task_id)
        if row is None:
            conn.execute("ROLLBACK")
            return ("not_found", {"chip_ref": chip_ref, "task_id": task_id})
        prior = row["status"]
        history = []
        if row["resolution_note_history_json"]:
            try:
                history = json.loads(row["resolution_note_history_json"]) or []
            except (TypeError, ValueError):
                history = []
        if prior == status:
            # Equal-status: leave it alone, with the ONE documented exception
            # -- the stored note is a generic AUTOMATED one and the caller is
            # supplying a real one. That is the recovery path for the
            # 2026-08-14 defect where a routine tick resolved a chip out from
            # under the worker who had done the work and the worker's own
            # resolve then no-op'd and DROPPED its report.
            if row["resolution_note_auto"] and note and not note_auto:
                history.append({
                    "resolution_note": row["resolution_note"],
                    "resolution_note_auto": True,
                    "replaced_at": now,
                    "replaced_by": resolved_by_session_id,
                })
                conn.execute(
                    "UPDATE chip_ledger SET resolution_note=?, "
                    "resolution_note_auto=0, resolution_note_history_json=?, "
                    "updated_at=? WHERE chip_ref=?",
                    (note, json.dumps(history), now, row["chip_ref"]))
                entry = _reserialise_chip_row(conn, row["chip_ref"])
                conn.execute("COMMIT")
                return ("ok", {"chip_ref": row["chip_ref"], "changed": True,
                               "replaced_auto_note": True, "entry": entry})
            conn.execute("ROLLBACK")
            return ("ok", {"chip_ref": row["chip_ref"], "changed": False,
                           "status": prior})
        if prior in CHIP_TERMINAL_STATUSES and not force:
            conn.execute("ROLLBACK")
            return ("terminal_conflict", {
                "chip_ref": row["chip_ref"], "status": prior,
                "resolved_at": row["resolved_at"],
                "resolved_by_session_id": row["resolved_by_session_id"],
                "resolution_note": row["resolution_note"]})
        if prior in CHIP_TERMINAL_STATUSES:
            history.append({
                "status": prior,
                "resolved_at": row["resolved_at"],
                "resolved_by_session_id": row["resolved_by_session_id"],
                "resolution_note": row["resolution_note"],
                "forced_over_at": now,
            })
        conn.execute(
            "UPDATE chip_ledger SET status=?, resolved_at=?, "
            "resolved_by_session_id=?, resolution_note=?, "
            "resolution_note_auto=?, resolution_note_history_json=?, "
            "updated_at=? WHERE chip_ref=?",
            (status, stamp, resolved_by_session_id, note,
             1 if note_auto else 0,
             json.dumps(history) if history else None,
             now, row["chip_ref"]))
        entry = _reserialise_chip_row(conn, row["chip_ref"])
        conn.execute("COMMIT")
        return ("ok", {"chip_ref": row["chip_ref"], "changed": True,
                       "previous_status": prior, "entry": entry})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def attach_chip(conn, chip_ref, task_id, attached_by_session_id=None,
                attached_at=None, now=None,
                stale_hours=CHIP_CLAIM_STALE_HOURS_DEFAULT):
    """Attach a spawn_task task_id to an already-recorded chip.

    Verdicts: 'ok' | 'not_found' | 'not_open' | 'claimed_by_other' |
    'task_id_taken' | 'already_attached' | 'error'.

    Does NOT claim the chip (chip_ledger.py's "RECORD/ATTACH DO NOT AUTO-CLAIM
    ON task_id"): a task_id means only that spawn_task created a clickable UI
    suggestion, not that anyone started it. Auto-claiming on attach is what
    left chips 40-60h old permanently marked claimed with no worktree and no
    process anywhere.

    'not_open' / 'claimed_by_other' close the gap the git path (chip_ledger.py
    cmd_attach's apply_fn) has always enforced locally: refuse to attach a UI
    chip to already-resolved work, and refuse to attach a second task_id onto
    a chip someone else is actively (non-stale-)claimed on. `stale_hours` uses
    the same `claimed_at`-age test as try_claim_chip's own already-claimed
    check above (`_is_stale_at`) -- the established convention in this file --
    rather than reimplementing chip_ledger.py's separate `claim_is_live`
    predicate; callers that want git-path parity (CLAIM_STALE_HOURS=3.0) pass
    it explicitly, same as claim already does.
    """
    now = now or utcnow()
    stamp = attached_at or now
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM chip_ledger WHERE chip_ref=?",
                           (chip_ref,)).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return ("not_found", {"chip_ref": chip_ref})
        if row["status"] != "open":
            conn.execute("ROLLBACK")
            return ("not_open", {"chip_ref": chip_ref,
                                 "status": row["status"]})
        if row["task_id"] and row["task_id"] != task_id:
            conn.execute("ROLLBACK")
            return ("already_attached", {"chip_ref": chip_ref,
                                         "task_id": row["task_id"]})
        other = conn.execute(
            "SELECT chip_ref FROM chip_ledger "
            "WHERE task_id=? AND task_id IS NOT NULL AND chip_ref<>?",
            (task_id, chip_ref)).fetchone()
        if other is not None:
            conn.execute("ROLLBACK")
            return ("task_id_taken", {"task_id": task_id,
                                      "chip_ref": other["chip_ref"]})
        if row["claimed_by"]:
            age_stale = _is_stale_at(row["claimed_at"], stale_hours, now)
            if row["claimed_at"] and not age_stale:
                conn.execute("ROLLBACK")
                return ("claimed_by_other", {
                    "chip_ref": chip_ref, "claimed_by": row["claimed_by"],
                    "claimed_at": row["claimed_at"],
                    "stale_after_hours": stale_hours})
        conn.execute(
            "UPDATE chip_ledger SET task_id=?, attached_by_session_id=?, "
            "attached_at=?, updated_at=? WHERE chip_ref=?",
            (task_id, attached_by_session_id, stamp, now, chip_ref))
        entry = _reserialise_chip_row(conn, chip_ref)
        conn.execute("COMMIT")
        return ("ok", {"chip_ref": chip_ref, "task_id": task_id,
                       "entry": entry})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def amend_chip_prompt(conn, chip_ref, prompt, reason=None, now=None):
    """Replace a chip's stored prompt, preserving the original in
    prompt_history. Verdicts: 'ok' | 'unchanged' | 'not_found' |
    'missing_marker' | 'error'.

    Same marker requirement as record_chip: this verb exists to repair an
    entry recorded with a placeholder instead of the real spawn_task text, so
    accepting another marker-less prompt would defeat it.

    'unchanged': `prompt` already equals the stored value. Mirrors the git
    path's own apply_fn no-op branch ("already matches -- not re-amending") --
    without this check every coordinator-suppressed amend of an unchanged
    prompt would still append a content-free entry to prompt_history.
    """
    now = now or utcnow()
    marker = "[chip_ref: %s]" % chip_ref
    if marker not in (prompt or ""):
        return ("missing_marker", {"chip_ref": chip_ref, "marker": marker})
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM chip_ledger WHERE chip_ref=?",
                           (chip_ref,)).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return ("not_found", {"chip_ref": chip_ref})
        if row["prompt"] == prompt:
            conn.execute("ROLLBACK")
            return ("unchanged", {"chip_ref": chip_ref})
        history = []
        if row["prompt_history_json"]:
            try:
                history = json.loads(row["prompt_history_json"]) or []
            except (TypeError, ValueError):
                history = []
        history.append({"prompt": row["prompt"], "replaced_at": now,
                        "reason": reason or ""})
        conn.execute(
            "UPDATE chip_ledger SET prompt=?, prompt_history_json=?, "
            "updated_at=? WHERE chip_ref=?",
            (prompt, json.dumps(history), now, chip_ref))
        entry = _reserialise_chip_row(conn, chip_ref)
        conn.execute("COMMIT")
        return ("ok", {"chip_ref": chip_ref, "entry": entry})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


# ---------------------------------------------------------------------------
# WORKSPACE_STATE.md append intake (PHASE-4, first slice).
#
# The registry tables above mirror WHOLE documents (every entry, upserted both
# directions). This table is deliberately narrower: it is an APPEND-ONLY
# intake spool. A closing session POSTs one entry; the materializer SPLICES
# pending entries into the live file on origin (byte-preserving -- it never
# re-renders the 2.4MB prose file) and marks them materialized once the text
# provably reached git. There is no ingest of the file's existing entries into
# the DB and no edit/delete verb -- WORKSPACE_STATE.md's git copy stays the
# authority for everything except the not-yet-materialized tail, so the
# ingest-authority clobber class the registries needed a 3-way merge for
# cannot arise here by construction.
#
# client_git_write records the CLIENT's declared intent at submit time:
#   1 -- the client will (also) append+commit the entry itself (dual-write
#        soak, or a box where WS suppression is not yet armed). The
#        materializer must NEVER splice such an entry -- doing so would race
#        the client's own push into a duplicate -- it only WATCHES for the
#        entry to appear in the file and then marks it materialized.
#   0 -- the client suppressed its git write; the materializer owns landing it.
# ---------------------------------------------------------------------------

def _migrate_workspace_state_table(conn):
    """Create the WORKSPACE_STATE.md append-intake table if missing. Same
    idempotent CREATE IF NOT EXISTS contract as _migrate_task_claim_chip_tables
    (called from connect(), so a live DB picks it up without a rebuild)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workspace_state_entries (
            entry_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts               TEXT NOT NULL,
            text             TEXT NOT NULL,
            session_id       TEXT NOT NULL DEFAULT '',
            client_git_write INTEGER NOT NULL DEFAULT 0,
            submitted_at     TEXT NOT NULL,
            submitted_host   TEXT,
            materialized_at  TEXT,
            materialized_ref TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_workspace_state_pending "
        "ON workspace_state_entries(materialized_at) "
        "WHERE materialized_at IS NULL"
    )


def submit_workspace_state_entry(conn, text, ts=None, session_id="",
                                 client_git_write=False, host=None, now=None):
    """Record one WORKSPACE_STATE.md closing entry for materialization.

    Append-only by design: there is no update or delete counterpart, and this
    function never touches an existing row. Verdicts:
      'ok'            -- inserted; payload carries entry_id + ts.
      'idempotent'    -- an identical (ts, text) row already exists (client
                         retry after a lost response); payload carries the
                         existing entry_id. Nothing written.
      'empty_text'    -- text is empty/whitespace. Nothing written.
      'bad_timestamp' -- ts does not parse as %Y-%m-%dT%H:%M:%SZ (the file's
                         heading convention). Nothing written.
      'error'         -- sqlite failure.

    `text` is stored stripped of leading/trailing newlines, exactly what the
    client-side format_entry() folds into '## <ts> -- <text>\\n\\n' -- the
    materializer re-derives the formatted block from these two fields, so
    normalizing here is what makes its presence check byte-exact.
    """
    now = now or utcnow()
    if not isinstance(text, str) or not text.strip():
        return ("empty_text", {})
    text = text.strip("\n")
    ts = ts or now
    try:
        datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return ("bad_timestamp", {"ts": ts})
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT entry_id FROM workspace_state_entries "
            "WHERE ts=? AND text=?", (ts, text)).fetchone()
        if row is not None:
            conn.execute("ROLLBACK")
            return ("idempotent", {"entry_id": row["entry_id"], "ts": ts})
        cur = conn.execute(
            "INSERT INTO workspace_state_entries "
            "(ts, text, session_id, client_git_write, submitted_at, "
            "submitted_host) VALUES (?,?,?,?,?,?)",
            (ts, text, session_id or "", 1 if client_git_write else 0,
             now, host))
        entry_id = cur.lastrowid
        conn.execute("COMMIT")
        return ("ok", {"entry_id": entry_id, "ts": ts})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def pending_workspace_state_entries(conn):
    """All not-yet-materialized entries, oldest submission first (entry_id
    order == submission order; the materializer splices newest-first from
    this so the file stays reverse-chronological)."""
    return conn.execute(
        "SELECT entry_id, ts, text, session_id, client_git_write, "
        "submitted_at FROM workspace_state_entries "
        "WHERE materialized_at IS NULL ORDER BY entry_id").fetchall()


def mark_workspace_state_entries_materialized(conn, entry_ids, ref, now=None):
    """Flip entries to materialized. Call ONLY once the entry's formatted
    text provably IS in git -- the push carrying it succeeded, or the text
    was found already present in origin's file (a dual-write client landed
    it). Same only-after-proof discipline as the registries'
    last_rendered_json writeback. Idempotent; never un-materializes."""
    ids = [int(i) for i in (entry_ids or [])]
    if not ids:
        return 0
    now = now or utcnow()
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.executemany(
            "UPDATE workspace_state_entries "
            "SET materialized_at=?, materialized_ref=? "
            "WHERE entry_id=? AND materialized_at IS NULL",
            [(now, ref, i) for i in ids])
        conn.execute("COMMIT")
        return cur.rowcount if cur.rowcount is not None else len(ids)
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise


def _migrate_recommendation_log_table(conn):
    """Create the RECOMMENDATION_LOG.jsonl append-intake table if missing.
    Same idempotent contract as _migrate_workspace_state_table (called from
    connect(), so a live DB picks it up without a rebuild)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_log_entries (
            entry_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            record           TEXT NOT NULL,
            session_id       TEXT NOT NULL DEFAULT '',
            client_git_write INTEGER NOT NULL DEFAULT 0,
            submitted_at     TEXT NOT NULL,
            submitted_host   TEXT,
            materialized_at  TEXT,
            materialized_ref TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_recommendation_log_pending "
        "ON recommendation_log_entries(materialized_at) "
        "WHERE materialized_at IS NULL"
    )


def submit_recommendation_log_entry(conn, record, session_id="",
                                    client_git_write=False, host=None,
                                    now=None):
    """Record one RECOMMENDATION_LOG.jsonl line for materialization.

    Append-only by design, mirroring submit_workspace_state_entry (the WS
    slice is the reference implementation; a jsonl append is its trivial
    case -- no insertion point, no section structure). Verdicts:
      'ok'          -- inserted; payload carries entry_id.
      'idempotent'  -- an identical record line already exists (client retry
                       after a lost response, or a dual-write client's own
                       earlier submit). Nothing written.
      'empty'       -- record is empty/whitespace. Nothing written.
      'bad_json'    -- record does not parse as a SINGLE-LINE JSON object
                       (the file is jsonl; a multi-line or non-object record
                       would corrupt every line-oriented reader). Nothing
                       written.
      'error'       -- sqlite failure.

    `record` is stored stripped of surrounding whitespace, exactly the line
    the client's own git path would append -- the materializer appends this
    text + newline verbatim, so byte-fidelity here is what makes its
    presence check exact.
    """
    now = now or utcnow()
    if not isinstance(record, str) or not record.strip():
        return ("empty", {})
    record = record.strip()
    if "\n" in record:
        return ("bad_json", {"reason": "record must be a single line"})
    try:
        parsed = json.loads(record)
    except ValueError:
        return ("bad_json", {"reason": "record does not parse as JSON"})
    if not isinstance(parsed, dict):
        return ("bad_json", {"reason": "record must be a JSON object"})
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT entry_id FROM recommendation_log_entries "
            "WHERE record=?", (record,)).fetchone()
        if row is not None:
            conn.execute("ROLLBACK")
            return ("idempotent", {"entry_id": row["entry_id"]})
        cur = conn.execute(
            "INSERT INTO recommendation_log_entries "
            "(record, session_id, client_git_write, submitted_at, "
            "submitted_host) VALUES (?,?,?,?,?)",
            (record, session_id or "", 1 if client_git_write else 0,
             now, host))
        entry_id = cur.lastrowid
        conn.execute("COMMIT")
        return ("ok", {"entry_id": entry_id})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def pending_recommendation_log_entries(conn):
    """All not-yet-materialized records, oldest submission first (the file
    is chronological-append, so the writer appends in this order)."""
    return conn.execute(
        "SELECT entry_id, record, session_id, client_git_write, "
        "submitted_at FROM recommendation_log_entries "
        "WHERE materialized_at IS NULL ORDER BY entry_id").fetchall()


def mark_recommendation_log_entries_materialized(conn, entry_ids, ref,
                                                 now=None):
    """Flip records to materialized. Call ONLY once the line provably IS in
    git -- same only-after-proof discipline as the WS counterpart."""
    ids = [int(i) for i in (entry_ids or [])]
    if not ids:
        return 0
    now = now or utcnow()
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.executemany(
            "UPDATE recommendation_log_entries "
            "SET materialized_at=?, materialized_ref=? "
            "WHERE entry_id=? AND materialized_at IS NULL",
            [(now, ref, i) for i in ids])
        conn.execute("COMMIT")
        return cur.rowcount if cur.rowcount is not None else len(ids)
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise


def _migrate_igw_log_table(conn):
    """Create the igw_routine_log.md append-intake table if missing.

    PHASE-4 slice (2026-09-01, git-traffic-simplification sweep lane 2): the
    IGW routine tick's heartbeat log, in REE_assembly rather than the
    umbrella repo -- see ree_assembly_git_writer.py, the DEDICATED clone
    materializer this table feeds (DP-6: one writer process per repo, never
    piggybacked on the umbrella task_claim_chip_git_writer.py tick, which
    only ever touches REE_Working). Same idempotent contract and same
    append-only shape as _migrate_recommendation_log_table -- a heartbeat
    line is the append-shaped trivial case exactly like a jsonl record."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS igw_log_entries (
            entry_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            line             TEXT NOT NULL,
            session_id       TEXT NOT NULL DEFAULT '',
            client_git_write INTEGER NOT NULL DEFAULT 0,
            submitted_at     TEXT NOT NULL,
            submitted_host   TEXT,
            materialized_at  TEXT,
            materialized_ref TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_igw_log_pending "
        "ON igw_log_entries(materialized_at) "
        "WHERE materialized_at IS NULL"
    )


def submit_igw_log_entry(conn, line, session_id="", client_git_write=False,
                         host=None, now=None):
    """Record one igw_routine_log.md heartbeat line for materialization.

    Append-only, the plain-text sibling of submit_recommendation_log_entry
    (no JSON validation -- the file is one free-text line per hourly tick,
    not jsonl). Verdicts:
      'ok'          -- inserted; payload carries entry_id.
      'idempotent'  -- an identical line already exists (client retry after
                       a lost response, or a dual-write client's own earlier
                       submit).
      'empty'       -- line is empty/whitespace. Nothing written.
      'multiline'   -- line contains a newline (the file is one-line-per-tick;
                       a multi-line entry would corrupt that convention).
      'error'       -- sqlite failure.

    `line` is stored stripped of surrounding whitespace, exactly the text
    scripts/igw_routine_tick.py's append_log() writes -- byte-fidelity is
    what makes the materializer's presence check exact.
    """
    now = now or utcnow()
    if not isinstance(line, str) or not line.strip():
        return ("empty", {})
    line = line.strip()
    if "\n" in line:
        return ("multiline", {"reason": "line must be a single line"})
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT entry_id FROM igw_log_entries "
            "WHERE line=?", (line,)).fetchone()
        if row is not None:
            conn.execute("ROLLBACK")
            return ("idempotent", {"entry_id": row["entry_id"]})
        cur = conn.execute(
            "INSERT INTO igw_log_entries "
            "(line, session_id, client_git_write, submitted_at, "
            "submitted_host) VALUES (?,?,?,?,?)",
            (line, session_id or "", 1 if client_git_write else 0,
             now, host))
        entry_id = cur.lastrowid
        conn.execute("COMMIT")
        return ("ok", {"entry_id": entry_id})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def pending_igw_log_entries(conn):
    """All not-yet-materialized lines, oldest submission first (the file is
    chronological-append, so the writer appends in this order)."""
    return conn.execute(
        "SELECT entry_id, line, session_id, client_git_write, "
        "submitted_at FROM igw_log_entries "
        "WHERE materialized_at IS NULL ORDER BY entry_id").fetchall()


def mark_igw_log_entries_materialized(conn, entry_ids, ref, now=None):
    """Flip lines to materialized. Call ONLY once the line provably IS in
    git -- same only-after-proof discipline as the WS/RECLOG counterparts."""
    ids = [int(i) for i in (entry_ids or [])]
    if not ids:
        return 0
    now = now or utcnow()
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.executemany(
            "UPDATE igw_log_entries "
            "SET materialized_at=?, materialized_ref=? "
            "WHERE entry_id=? AND materialized_at IS NULL",
            [(now, ref, i) for i in ids])
        conn.execute("COMMIT")
        return cur.rowcount if cur.rowcount is not None else len(ids)
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise


def _migrate_git_intent_log_table(conn):
    """Create the /intent/replace audit-log table if missing. Records every
    CAS intent this coordinator has seen applying (verdict included) --
    PHASE-4 whole-file editorial routing (phase4_commit_intake_design.md
    section 3.2). Purely an audit trail (DP-9 actor identity, section 7 soak
    evidence); the intent APPLICATION itself is git_intent.apply_intent's
    job, not this table's -- unlike the append tables above, there is no
    'pending' state here because a CAS intent is applied (or refused)
    synchronously within the one request that submitted it."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS git_intent_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            repo         TEXT NOT NULL,
            path         TEXT NOT NULL,
            base_sha     TEXT,
            verdict      TEXT NOT NULL,
            applied_sha  TEXT,
            session_id   TEXT NOT NULL DEFAULT '',
            machine      TEXT,
            message      TEXT,
            shadow       INTEGER NOT NULL DEFAULT 0,
            size_before  INTEGER,
            size_after   INTEGER,
            created_at   TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_git_intent_log_path "
        "ON git_intent_log(repo, path, created_at)"
    )


def record_git_intent(conn, *, repo, path, base_sha, verdict, applied_sha=None,
                      session_id="", machine=None, message=None,
                      shadow=False, size_before=None, size_after=None,
                      now=None):
    """Append one audit row for an /intent/replace application. Best-effort:
    a logging failure must never fail the intent it is recording, so this
    swallows sqlite errors rather than propagating them (unlike the
    append-intake writers above, whose INSERT is the primary effect)."""
    now = now or utcnow()
    try:
        conn.execute(
            "INSERT INTO git_intent_log "
            "(repo, path, base_sha, verdict, applied_sha, session_id, "
            "machine, message, shadow, size_before, size_after, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (repo, path, base_sha, verdict, applied_sha, session_id or "",
             machine, message, 1 if shadow else 0, size_before, size_after,
             now))
    except sqlite3.Error:
        pass


def _migrate_dispatcher_leases_table(conn):
    """Create the dispatcher run-lease table if missing. Same idempotent
    contract as the other _migrate_* helpers (called from connect())."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dispatcher_leases (
            dispatcher   TEXT PRIMARY KEY,
            requested_at TEXT NOT NULL,
            entry_json   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            updated_via  TEXT
        )
        """
    )


def upsert_dispatcher_lease(conn, dispatcher, entry, via, now=None):
    """Upsert one dispatcher's lease entry, NEWEST requested_at wins.

    `entry` is the full lease dict exactly as dispatcher_control.json's
    dispatchers[<name>] carries it (lossless entry_json doctrine -- the
    renderer emits it verbatim, so unknown future fields survive a round
    trip). Verdicts:
      'ok'          -- inserted or replaced (strictly newer requested_at,
                       or no prior row).
      'idempotent'  -- an identical entry is already stored (client retry).
      'stale'       -- the stored row's requested_at is newer or equal with
                       different content; nothing written. The caller keeps
                       the newer state -- this is what makes an INGEST of an
                       old git file safe after endpoint writes, and an
                       endpoint replay safe after a newer git-side stop.
      'bad_entry'   -- entry is not a dict with a parseable requested_at.
      'error'       -- sqlite failure.
    """
    now = now or utcnow()
    if not isinstance(entry, dict):
        return ("bad_entry", {"reason": "entry must be an object"})
    requested_at = entry.get("requested_at")
    try:
        datetime.strptime(requested_at or "", "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return ("bad_entry", {"reason": "requested_at must be UTC ISO-8601"})
    blob = json.dumps(entry, sort_keys=True)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT requested_at, entry_json FROM dispatcher_leases "
            "WHERE dispatcher=?", (dispatcher,)).fetchone()
        if row is not None:
            if row["entry_json"] == blob:
                conn.execute("ROLLBACK")
                return ("idempotent", {"dispatcher": dispatcher})
            if row["requested_at"] >= requested_at:
                conn.execute("ROLLBACK")
                return ("stale", {"dispatcher": dispatcher,
                                  "stored_requested_at": row["requested_at"]})
        conn.execute(
            "INSERT INTO dispatcher_leases "
            "(dispatcher, requested_at, entry_json, updated_at, updated_via) "
            "VALUES (?,?,?,?,?) ON CONFLICT(dispatcher) DO UPDATE SET "
            "requested_at=excluded.requested_at, "
            "entry_json=excluded.entry_json, "
            "updated_at=excluded.updated_at, "
            "updated_via=excluded.updated_via",
            (dispatcher, requested_at, blob, now, via))
        conn.execute("COMMIT")
        return ("ok", {"dispatcher": dispatcher, "requested_at": requested_at})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def dispatcher_lease_rows(conn):
    """All lease rows, keyed for the renderer."""
    return conn.execute(
        "SELECT dispatcher, requested_at, entry_json, updated_at, "
        "updated_via FROM dispatcher_leases ORDER BY dispatcher").fetchall()


# Bounded-growth cap for a standing chip's episode list (W5a). Old episodes
# are dropped OLDEST-first past this, with episodes_truncated counting the
# drops -- a standing class chip lives for weeks and its per-episode
# forensic detail matters most recently; the count preserves the magnitude
# when the detail ages out.
CHIP_EPISODE_CAP = 500


def record_chip_episode(conn, chip_ref, episode, now=None):
    """Append one observation episode INTO a standing chip (W5a, 2026-08-29
    fleet-wedge campaign; precondition C3 of the redesign's red-team). The
    recurrence-collapse design replaces per-observation `since-<ts>` chip
    minting with ONE standing chip per (class, subject) whose episodes
    accumulate here -- and this verb is what lets that accumulation ride
    the coordinator like resolve/claim do, instead of the mirror-less
    amend-shaped git path that would have re-installed the R3 stranding
    generator inside the workstream meant to end it.

    `episode` is a dict carrying a client-chosen stable `episode_key`
    (e.g. the since-<ts> string the old minting would have used) -- the
    idempotency key, so a tick replaying an observation is a no-op -- plus
    any forensic payload. Verdicts:
      'ok'          -- appended; payload carries episode_count.
      'idempotent'  -- an episode with this episode_key already exists.
      'not_found'   -- no chip with this chip_ref.
      'not_open'    -- the chip is resolved; episodes only accumulate on a
                       standing OPEN chip (the minting client decides
                       whether a re-observation of a resolved class reopens
                       or escalates -- that judgment does not belong in a
                       storage verb).
      'bad_episode' -- episode is not a dict with a non-empty string
                       episode_key.
      'error'       -- sqlite failure.

    Episodes live in entry_json (episodes[], episode_count,
    last_episode_at, episodes_truncated), so the materializer renders them
    with zero changes and every whole-file git reader round-trips them.
    """
    now = now or utcnow()
    if (not isinstance(episode, dict)
            or not isinstance(episode.get("episode_key"), str)
            or not episode["episode_key"].strip()):
        return ("bad_episode", {"reason": "episode must be an object with a "
                                          "non-empty string episode_key"})
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _find_chip_row(conn, chip_ref=chip_ref)
        if row is None:
            conn.execute("ROLLBACK")
            return ("not_found", {"chip_ref": chip_ref})
        if row["status"] != "open":
            conn.execute("ROLLBACK")
            return ("not_open", {"chip_ref": chip_ref,
                                 "status": row["status"]})
        try:
            entry = json.loads(row["entry_json"])
        except (TypeError, ValueError):
            conn.execute("ROLLBACK")
            return ("error", {"reason": "stored entry_json unparseable"})
        episodes = entry.get("episodes") or []
        key = episode["episode_key"]
        if any(isinstance(e, dict) and e.get("episode_key") == key
               for e in episodes):
            conn.execute("ROLLBACK")
            return ("idempotent", {"chip_ref": chip_ref, "episode_key": key})
        ep = dict(episode)
        ep.setdefault("observed_at", now)
        episodes.append(ep)
        truncated = int(entry.get("episodes_truncated") or 0)
        while len(episodes) > CHIP_EPISODE_CAP:
            episodes.pop(0)
            truncated += 1
        entry["episodes"] = episodes
        entry["episode_count"] = len(episodes) + truncated
        entry["last_episode_at"] = ep["observed_at"]
        if truncated:
            entry["episodes_truncated"] = truncated
        conn.execute(
            "UPDATE chip_ledger SET entry_json=?, updated_at=? "
            "WHERE chip_ref=?",
            (json.dumps(entry, sort_keys=True), now, row["chip_ref"]))
        conn.execute("COMMIT")
        return ("ok", {"chip_ref": chip_ref,
                       "episode_count": entry["episode_count"],
                       "last_episode_at": entry["last_episode_at"]})
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        return ("error", {})


def init_db(db_path):
    conn = connect(db_path)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        conn.executescript(fh.read())
    _migrate_heartbeats(conn)
    _migrate_commands(conn)
    _migrate_experiments(conn)
    _migrate_heartbeat_log(conn)
    _migrate_task_claim_chip_tables(conn)
    _migrate_workspace_state_table(conn)
    _migrate_recommendation_log_table(conn)
    _migrate_igw_log_table(conn)
    _migrate_git_intent_log_table(conn)
    _migrate_dispatcher_leases_table(conn)
    conn.close()


# Statuses from which a row never becomes claimable again. Single source of
# truth because three readers must agree on it: sync_daemon.reconcile_once's
# terminal-row guard, sync_daemon._materialise_queue_from_db's worklist
# filter, and POST /queue/add's 409. A disagreement between them is exactly
# how a queue_id gets silently burned (V3-EXQ-728a, 2026-07-20).
TERMINAL_STATUSES = ("completed", "failed")


def _affinity_ok(affinity, machine):
    """True when `machine` is allowed to run a row pinned to `affinity`.

    Compared on CANONICAL identity, not raw string equality, mirroring
    experiment_runner._affinity_matches -- which has done this since the
    machine_identity resolver landed. Without it, a queue entry pinned
    `DLAPTOP-4.local` silently stops being claimable the moment the Mac's
    runner starts reporting its canonical `DLAPTOP`: no error, no log line,
    the experiment just starves.

    `same_machine` is an ALLOWLIST, so `ree-cloud-1` .. `-5` and
    `ree-worker-1` .. `-4` still compare as distinct machines -- collapsing
    those would route every affinity-pinned experiment to the wrong box.
    See machine_identity.SUFFIX_BLIND_BASES, and the distinct-keys contract
    in coordinator/test_machine_identity_ingest.py.
    """
    return affinity in (None, "", "any") or same_machine(affinity, machine)


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


# How fresh an owner's heartbeat must be, against the CURRENT queue_id, to
# block route (a) (the absence-based stale-claim recovery in
# _claim_recoverable below). This is the window that actually decided the
# V3-EXQ-861f incident (2026-08-23): DLAPTOP's claim was already ~25h past
# `stale_hours` (it always will be, for any experiment running longer than a
# few hours -- that floor is about ABANDONMENT, not about how long a single
# run may take), so the whole outcome rode on this one number. Its owner
# heartbeated continuously via POST /heartbeat (confirmed from the hub's
# coordinator access log -- one heartbeat roughly every 5s for the full
# 25-hour run) except for a single ~54-minute gap, almost certainly a
# transient network/sleep blip on the laptop, which is not evidence the
# machine had stopped running the experiment. ree-cloud-4 polled to claim
# exactly 16 minutes into that gap -- past the old 900s (15min) floor -- and
# won, producing a second, duplicate ~9h execution of the same queue_id while
# DLAPTOP's own run continued for another ~15 hours to completion.
#
# WHY NOT JUST RAISE stale_hours instead: see _claim_recoverable's own
# docstring and CLAUDE.md -- that floor is deliberately about long-run
# abandonment, not about heartbeat noise, and this incident's claim was
# already many multiples past it. The fix belongs on the ABSENCE-detection
# granularity, not the eligibility floor.
#
# WHY THIS IS NOT THE SAME NUMBER AS CLAIM_REAP_QUIET_DEFAULT_SECONDS below,
# despite both having read "900" and both answering some version of "how
# long is quiet before we act". They gate different EVIDENCE. Route (b)
# (departure) only engages once the machine has POSITIVELY announced a
# shutdown -- a real signal that costs nothing to wait out quickly (900s),
# because there is no risk of mistaking "still running" for "gone" once a
# shutdown notice already fired. Route (a) has no positive signal at all: it
# is pure absence, on a fleet that includes a laptop subject to sleep, WiFi
# roaming, and VPN reassociation, so it needs slack against exactly the kind
# of transient blip that caused this incident. Widening this constant alone
# (not the departure one) targets that difference without slowing down the
# already-proven-good departure recovery from the V3-EXQ-841 incident.
# 3600s (60min) gives >4x margin over the measured 54-minute gap while
# staying at 16.7% of the 6h stale_hours floor, preserving the two-layer
# defense (an owner must be BOTH long-claimed AND silent for a full hour
# before another machine may take over).
HEARTBEAT_FRESH_DEFAULT_SECONDS = 3600


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
                       heartbeat_fresh_seconds, reap_quiet_seconds=0):
    """Is this claim available to a DIFFERENT machine?

    Two independent routes, ORed:

      (a) TIME -- the claim is older than `stale_hours` and its owner is not
          heartbeating against this very queue_id. The original rule. It is
          evidence of ABSENCE, so it needs the long 6h floor to be safe.
      (b) DEPARTURE -- the owner announced a machine shutdown and has since
          gone quiet. Evidence of PRESENCE (of a departure), so it is safe
          in minutes rather than hours. See claim_orphaned_by_departure.

    reap_quiet_seconds=0 (the default) disables (b) entirely, so every
    existing caller and the shadow comparison stay bit-identical unless the
    reaper is explicitly wired in.
    """
    if row["status"] != "claimed":
        return False
    if claim_orphaned_by_departure(conn, row, reap_quiet_seconds):
        return True
    if not _is_stale(row["claimed_at"], stale_hours):
        return False
    return not _has_fresh_owner_heartbeat(
        conn, queue_id, row["claimed_by_machine"], heartbeat_fresh_seconds)


# ---- drain-window claim fence ---------------------------------------------
#
# A worker the scaler has issued `hcloud server shutdown` against keeps
# running for the whole ACPI drain window (measured 2026-07-30: ~26 min on
# ree-worker-4). Its runner poll loop stays alive throughout, so without a
# fence it happily claims fresh work and then powers off holding it -- the
# claim is then orphaned until the 6h stale threshold in _claim_recoverable
# above. Confirmed incident: V3-EXQ-841, a 600-estimated-minute experiment,
# claimed by ree-cloud-4 at 07:08:39Z -- 8 minutes into a drain announced at
# 07:00:18Z -- and not re-claimable until 17:08Z (~10h).
#
# The scaler cannot retract an in-flight ACPI shutdown, and its three
# protection layers (HUB_NAME skip, HELD_BY_SELF, lease veto) are all
# evaluated at DECISION time -- they can only decline to issue a NEW
# shutdown. All five of the scaler's subsequent "keeping ree-worker-4
# running" votes during that window were undeliverable. So the fence has to
# live on the claim path, not the shutdown path.
#
# Server-side (here) rather than runner-local, deliberately: this also
# covers a runner whose SIGTERM handler never fires. The runner-local drain
# fence does exist and works (experiment_runner breaks out of the claim loop
# on _drain_flag), but in the confirmed incident it never armed -- the
# runner claimed 8 minutes in, so it had not observed a drain signal at all.

# Shutdown reasons that announce a PROCESS exit, not a machine shutdown.
# The runner posts these itself on its way out (experiment_runner
# _announce_intentional_shutdown). The ree-runner unit carries
# Restart=always, so fencing on these would stall the restarted runner for
# the whole fence window on an ordinary `systemctl restart` or remote stop.
# The scaler's own notice (default reason "scaler_idle_after_grace") is the
# one that means "this machine is going away", and anything unrecognised
# fences -- default-deny is the safe direction for the incident above.
PROCESS_EXIT_SHUTDOWN_REASONS = frozenset({
    "runner_drain_complete",
    "runner_signal_exit",
})

# How long a pending shutdown notice fences claims. Must exceed the ACPI
# drain window (~26 min measured) or the fence lapses while the box is
# still up and claiming. Bounded so a machine woken by a MANUAL `hcloud
# server poweron` -- which never calls record_claim_fence_clear -- recovers
# on its own instead of staying fenced forever.
CLAIM_FENCE_DEFAULT_SECONDS = 1800


def claim_fence_active(last_shutdown_at, claim_fence_cleared_at,
                       shutdown_reason, *, fence_seconds, now=None):
    """Is this machine inside a pending shutdown's claim fence?

    Pure function; mirrors lifecycle_state's timestamp-comparison style.
    Fenced when a shutdown notice is (a) present, (b) not a process-exit
    reason, (c) not superseded by a later fence clear, and (d) still inside
    the fence window.

    fence_seconds <= 0 disables the fence entirely (the default on both
    claim entry points, so existing callers and the shadow comparison are
    bit-identical unless the fence is explicitly wired in).
    """
    if not fence_seconds or fence_seconds <= 0:
        return False
    shutdown_dt = _parse_utc(last_shutdown_at)
    if shutdown_dt is None:
        return False
    # Since the 2026-07-30 column split (see record_shutdown_notice) a
    # process-exit reason is written to process_exit_reason and never
    # reaches shutdown_reason, so this is belt-and-braces for NEW rows.
    # KEEP IT: it is still load-bearing for rows written BEFORE the split,
    # which really do carry e.g. 'runner_signal_exit' here -- the live
    # coordinator DB had exactly that for ree-cloud-4 on the day it was
    # found. Without this, adopting the new code would read those legacy
    # rows as an armed machine-shutdown fence and refuse claims from
    # healthy boxes until the stamps aged out of the window.
    if shutdown_reason in PROCESS_EXIT_SHUTDOWN_REASONS:
        return False
    cleared_dt = _parse_utc(claim_fence_cleared_at)
    # STRICTLY newer, unlike lifecycle_state's `shutdown_dt >= seen_dt`.
    # These timestamps have 1-second resolution, so a clear and a shutdown
    # notice landing in the same second are indistinguishable by order --
    # and the two outcomes are not symmetric. Refusing a claim from a
    # machine that is actually fine costs one poll tick; granting one to a
    # machine that is actually draining is the 10h orphan. So a tie leaves
    # the fence ARMED and the clear must be strictly newer to disarm.
    if cleared_dt is not None and cleared_dt > shutdown_dt:
        return False
    now_dt = now or datetime.now(timezone.utc)
    return (now_dt - shutdown_dt).total_seconds() <= fence_seconds


def machine_claim_fenced(conn, machine, fence_seconds, now=None):
    """DB-backed claim_fence_active for `machine`. False when no row."""
    if not fence_seconds or fence_seconds <= 0:
        return False
    row = conn.execute(
        "SELECT last_shutdown_at, claim_fence_cleared_at, shutdown_reason "
        "FROM heartbeats WHERE machine=?",
        (machine,),
    ).fetchone()
    if row is None:
        return False
    return claim_fence_active(
        row["last_shutdown_at"], row["claim_fence_cleared_at"],
        row["shutdown_reason"], fence_seconds=fence_seconds, now=now)


def record_claim_fence_clear(conn, machine):
    """Disarm the claim fence for `machine` (the scaler calls this on
    poweron). Stamps claim_fence_cleared_at=now, which supersedes any
    shutdown notice older than it.

    Deliberately does NOT null last_shutdown_at: lifecycle_state and
    phase3_preflight read that column, and a machine that is powered on but
    has not heartbeated yet should stay `gracefully_offline` rather than
    dropping straight to `stale` for the ~90s it takes to boot.

    Same row-creation contract as record_shutdown_notice -- sentinel
    last_seen so lifecycle_state does not mistake the write for liveness.
    """
    conn.execute(
        """
        INSERT INTO heartbeats (machine, last_seen, claim_fence_cleared_at)
        VALUES (?,?,?)
        ON CONFLICT(machine) DO UPDATE SET
          claim_fence_cleared_at=excluded.claim_fence_cleared_at
        """,
        (machine, _NEVER_HEARTBEATED_SENTINEL, utcnow()),
    )


# ---- stale-claim reaper (departure-evidence recovery) ----------------------
#
# The fence above (claim_fence_active) stops a draining worker TAKING a
# claim it cannot finish. It cannot help a claim that is ALREADY orphaned by
# any other route -- a worker that crashes, is SIGKILLed, loses network, or
# was shut down before the fence existed. For those, _claim_recoverable's
# 6h `stale_hours` floor was until now the only recovery path, and it is a
# floor: the confirmed V3-EXQ-841 incident (2026-07-30) sat orphaned ~10h
# because the owner was provably powered off within minutes and nothing
# could act on that.
#
# WHY NOT SIMPLY LOWER stale_hours. `_is_stale` + `_has_fresh_owner_heartbeat`
# reason from the ABSENCE of telemetry, and absence is not abandonment:
# CLAUDE.md carries an explicit standing warning that a heartbeat-stale
# machine (especially the Mac, DLAPTOP-4) may still be RUNNING the
# experiment. Releasing that claim causes a DUPLICATE RUN, which is strictly
# worse than a slow recovery -- it burns fleet hours and can produce two
# manifests for one queue_id. So the floor stays where it is, and recovery
# is instead widened with a POSITIVE signal.
#
# THE POSITIVE SIGNAL is the shutdown notice the machine (or the scaler on
# its behalf) already posts to /shutdown_notify, plus subsequent silence.
# "Announced it was going away AND then stopped talking" is a different
# claim about the world from "stopped talking", and it is one the Mac never
# satisfies: the runner only ever announces PROCESS_EXIT_SHUTDOWN_REASONS,
# which are excluded below.
#
# WHY NOT hcloud POWER STATE, which is the other authoritative signal (and
# is what cloud-scaler.py reads). It would mean a blocking subprocess to an
# external API inside the /claim request path of a zero-dependency stdlib
# service, on a box whose network path to Hetzner is not part of any other
# claim decision -- a new failure mode on the hot path, for a signal the
# scaler has ALREADY translated into a shutdown notice before it powers a
# box off. The DB is the right place to read it from.
#
# WHY IN THE CLAIM PREDICATE rather than a background sweeper thread. The
# only thing a reaped claim needs to become is CLAIMABLE, and try_claim's
# BEGIN IMMEDIATE already makes exactly one machine win. A sweeper would add
# a scheduler, a second writer, and a window in which the row says pending
# while an owner is still finishing -- for no gain. Cost: the git queue
# snapshot keeps showing the stale `claimed_by` until some worker actually
# asks, which is a visibility lag of one poll (~30s), not a correctness gap.
# `departed` is surfaced on /shadow/status so the orphan is visible anyway.

# How long a machine must be SILENT after announcing a shutdown before its
# claims are reapable. Until 2026-08-23 this was deliberately equal to the
# default heartbeat_fresh_seconds (900) -- "one number for 'we would have
# heard from this machine by now', used by both the absence rule and this
# one." That default has since moved to HEARTBEAT_FRESH_DEFAULT_SECONDS
# (3600, see its own docstring for the V3-EXQ-861f incident that drove it),
# and this constant deliberately did NOT move with it: the two gate different
# EVIDENCE. This route only engages after a POSITIVE signal -- an explicit
# shutdown notice -- so there is no risk of mistaking "still running" for
# "gone" the way a bare heartbeat gap can, and widening it would only slow
# down the V3-EXQ-841 recovery this route exists for, for no safety benefit.
# It must comfortably exceed the runner's ~60s heartbeat cadence; it does NOT
# need to exceed the ~26min ACPI drain window, because a draining box keeps
# heartbeating and so is never quiet while it drains.
CLAIM_REAP_QUIET_DEFAULT_SECONDS = 900


def machine_departed(last_seen, last_shutdown_at, claim_fence_cleared_at,
                     shutdown_reason, *, quiet_seconds, now=None):
    """Has this machine announced a departure and then gone quiet?

    Pure function, same shape as claim_fence_active. True when:
      (a) a shutdown notice is present, and
      (b) it is a MACHINE departure, not a process exit -- the runner's own
          `runner_signal_exit` / `runner_drain_complete` are excluded because
          that runner already released its claim on the way out
          (experiment_runner._do_immediate_exit) and systemd's
          Restart=always brings it straight back, so reaping on those would
          race a live machine for no benefit. As in claim_fence_active, the
          2026-07-30 column split means a process exit no longer lands in
          shutdown_reason at all, so this check now mainly protects rows
          written before the split -- and the split also removes the second
          bug it was masking: a process exit used to overwrite
          last_shutdown_at too, so a real machine departure stopped being
          reapable the moment the exiting runner announced itself, and
      (c) no fence-clear supersedes it -- the scaler stamps one on every
          poweron, so a machine that has been woken is not "departed"
          however long ago its last shutdown was, and
      (d) the machine has been silent for at least `quiet_seconds`.

    (d) is what makes this safe. A worker inside its ACPI drain window is
    still heartbeating, so it is never quiet; the signal only fires once the
    box has actually stopped. quiet_seconds <= 0 disables entirely.
    """
    if not quiet_seconds or quiet_seconds <= 0:
        return False
    if shutdown_reason in PROCESS_EXIT_SHUTDOWN_REASONS:
        return False
    shutdown_dt = _parse_utc(last_shutdown_at)
    if shutdown_dt is None:
        return False
    cleared_dt = _parse_utc(claim_fence_cleared_at)
    # `>=`, which is the OPPOSITE strictness from claim_fence_active's `>`
    # -- deliberately, and the difference is the whole point. Both face the
    # same ambiguity (1-second timestamp resolution cannot order a clear and
    # a notice landing in the same second) and both resolve it toward the
    # cheaper error, but "cheaper" points opposite ways:
    #
    #   fence   -- wrong refusal costs one poll tick; wrong grant is the 10h
    #              orphan. So a tie stays FENCED, i.e. the clear must be
    #              strictly newer to disarm.
    #   reaper  -- wrong deferral costs one poll tick; wrong reap risks a
    #              DUPLICATE RUN on a machine that was just powered back on.
    #              So a tie is NOT a departure, i.e. a clear at the same
    #              second is enough to cancel.
    #
    # This is not hypothetical precision: record_claim_fence_clear and
    # record_shutdown_notice both stamp utcnow(), so a scaler poweron
    # following its own shutdown notice can land in the same second.
    if cleared_dt is not None and cleared_dt >= shutdown_dt:
        return False
    seen_dt = _parse_utc(last_seen)
    if seen_dt is not None:
        now_dt = now or datetime.now(timezone.utc)
        if (now_dt - seen_dt).total_seconds() < quiet_seconds:
            return False
    return True


def machine_has_departed(conn, machine, quiet_seconds, now=None):
    """DB-backed machine_departed for `machine`. False when no row."""
    if not machine or not quiet_seconds or quiet_seconds <= 0:
        return False
    row = conn.execute(
        "SELECT last_seen, last_shutdown_at, claim_fence_cleared_at, "
        "shutdown_reason FROM heartbeats WHERE machine=?",
        (machine,),
    ).fetchone()
    if row is None:
        return False
    return machine_departed(
        row["last_seen"], row["last_shutdown_at"],
        row["claim_fence_cleared_at"], row["shutdown_reason"],
        quiet_seconds=quiet_seconds, now=now)


def claim_orphaned_by_departure(conn, row, quiet_seconds, now=None):
    """Is this claimed row held by a machine that has provably departed?

    Note what this does NOT do: it never marks anything completed, and it
    never touches the results table. The only transition it enables is
    claimed -> claimed-by-someone-else, via try_claim. A run that died
    before writing its manifest stays a run that has no manifest -- the
    reaper cannot manufacture or mask a phantom completion.
    """
    if row["status"] != "claimed":
        return False
    return machine_has_departed(
        conn, row["claimed_by_machine"], quiet_seconds, now=now)


def try_claim(conn, queue_id, machine, stale_hours=6,
              heartbeat_fresh_seconds=HEARTBEAT_FRESH_DEFAULT_SECONDS,
              fence_seconds=0, reap_quiet_seconds=0):
    """Atomic conditional claim. Returns one of: 'ok', 'already_claimed',
    'draining', 'error'. Mirrors experiment_runner.attempt_claim() semantics
    exactly so the shadow comparison is apples-to-apples.

    'draining' means the machine has a pending shutdown notice -- see
    claim_fence_active. Checked first and machine-scoped, so it is
    independent of the item's own eligibility. fence_seconds=0 (the
    default) disables it.

    reap_quiet_seconds enables departure-evidence recovery of a claim held
    by a machine that announced a shutdown and went quiet -- see
    claim_orphaned_by_departure. 0 (the default) disables it.

    The reaper cannot hand a claim to a draining machine: the fence is
    evaluated first and is machine-scoped, so a departed owner's claim
    becoming reapable does not make a departing CALLER eligible to take it.

    Atomicity: BEGIN IMMEDIATE takes the write lock before the SELECT, so no
    other request can interleave between the eligibility check and the UPDATE.
    That is also what makes the reaper safe under a fleet-wide poll: several
    workers can find the same orphan reapable in the same second, and exactly
    one of them wins the UPDATE.
    """
    now = utcnow()
    try:
        if machine_claim_fenced(conn, machine, fence_seconds):
            return "draining"
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
            conn, row, queue_id, stale_hours, heartbeat_fresh_seconds,
            reap_quiet_seconds)
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
                   heartbeat_fresh_seconds=HEARTBEAT_FRESH_DEFAULT_SECONDS,
                   fence_seconds=0, reap_quiet_seconds=0):
    """Read-only: what verdict WOULD the coordinator return, without
    mutating the mirror. Used by the shadow audit so the comparison does
    not perturb state. Same logic as try_claim()."""
    if machine_claim_fenced(conn, machine, fence_seconds):
        return "draining"
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
        conn, row, queue_id, stale_hours, heartbeat_fresh_seconds,
        reap_quiet_seconds)
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


# heartbeat_log retention/write knobs. Env-overridable, same convention as
# app.py's COORDINATOR_* constants. See schema.sql's heartbeat_log comment
# for why this is on-change (not every tick) and why trim runs opportunistic
# rather than needing its own systemd timer.
HEARTBEAT_LOG_RETENTION_DAYS = int(
    os.environ.get("COORDINATOR_HEARTBEAT_LOG_RETENTION_DAYS", "30"))
HEARTBEAT_LOG_TRIM_INTERVAL_SECONDS = float(
    os.environ.get("COORDINATOR_HEARTBEAT_LOG_TRIM_INTERVAL_SECONDS", "3600"))

# Wall-clock (time.monotonic) of the last trim, module-global because the
# coordinator is one process (ThreadingHTTPServer) even though each request
# opens its own sqlite3 connection -- see app.py's per-request db.connect().
# A race between two threads both deciding to trim at once is harmless: the
# DELETE is idempotent and cheap (uses idx_heartbeat_log_machine_time).
_last_heartbeat_log_trim = [0.0]


def trim_heartbeat_log(conn, retention_days=None):
    """Delete heartbeat_log rows older than `retention_days` (default
    HEARTBEAT_LOG_RETENTION_DAYS). Returns the number of rows deleted.
    Safe to call any time; not gated by _maybe_trim_heartbeat_log's interval
    -- that gate is only for the opportunistic call from upsert_heartbeat.
    """
    days = HEARTBEAT_LOG_RETENTION_DAYS if retention_days is None else retention_days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    cur = conn.execute(
        "DELETE FROM heartbeat_log WHERE observed_at < ?", (cutoff,))
    return cur.rowcount


def _maybe_trim_heartbeat_log(conn):
    """Opportunistic trim, at most once per HEARTBEAT_LOG_TRIM_INTERVAL_SECONDS.

    Keeps the retention policy self-enforcing on the write path with no
    separate timer/cron needed -- every POST /heartbeat is a chance to trim,
    throttled so the DELETE only actually runs ~once/hour regardless of
    heartbeat volume.
    """
    now = time.monotonic()
    if now - _last_heartbeat_log_trim[0] < HEARTBEAT_LOG_TRIM_INTERVAL_SECONDS:
        return
    _last_heartbeat_log_trim[0] = now
    trim_heartbeat_log(conn)


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

    Also appends to heartbeat_log (see schema.sql) when (state,
    current_exq) differs from the machine's PRIOR row -- i.e. one history
    row per experiment start/finish/switch/idle transition, read BEFORE
    this upsert overwrites the current-state row.
    """
    prior = conn.execute(
        "SELECT state, current_exq FROM heartbeats WHERE machine=?",
        (machine,),
    ).fetchone()
    transitioned = (
        prior is None
        or (prior["state"], prior["current_exq"]) != (state, current_exq)
    )

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

    if transitioned:
        conn.execute(
            """
            INSERT INTO heartbeat_log
              (machine, observed_at, state, current_exq, payload_json)
            VALUES (?,?,?,?,?)
            """,
            (machine, utcnow(), state, current_exq, payload_json),
        )
    _maybe_trim_heartbeat_log(conn)


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

    TWO KINDS OF EVENT ARRIVE HERE, AND THEY ARE STORED SEPARATELY. Both
    reach this function through the single POST /shutdown_notify endpoint,
    but they are different claims about the world:

      MACHINE SHUTDOWN  -- "this box is going away" (the scaler's
        "scaler_idle_after_grace", a manual poweroff, anything
        unrecognised). Written to last_shutdown_at / shutdown_reason /
        expected_wake_condition. THIS is what arms the claim fence.
      RUNNER PROCESS EXIT -- "the runner process is exiting, the box stays
        up" (PROCESS_EXIT_SHUTDOWN_REASONS). Written to
        last_process_exit_at / process_exit_reason. Deliberately does NOT
        fence, because ree-runner.service carries Restart=always and
        fencing an ordinary `systemctl restart` would stall the restarted
        runner for the whole fence window.

    WHY THE COLUMNS ARE SPLIT (confirmed in production 2026-07-30). They
    used to share one pair, so a process-exit notice OVERWROTE the
    machine-shutdown reason -- and because process-exit reasons are exactly
    the ones claim_fence_active excludes, that overwrite DISARMED the fence
    the scaler had just armed. Measured on ree-cloud-4: the scaler
    announced at 18:10:12Z, and one second later the row read
    shutdown_reason='runner_signal_exit'. The fence survived ~1 second.

    That did not break the incident the fence was built for (there the
    runner never exited -- it kept polling for the whole ~26min ACPI drain,
    so nothing overwrote anything). The residual it DID leave is the
    runner-exits-early path: an idle worker with nothing to drain announces
    its process exit within seconds, the box then takes minutes to actually
    power off, and Restart=always can bring a runner back inside that
    window -- UNFENCED, free to claim into a dying box. Splitting the
    columns removes the interference at the source: neither event can
    clobber the other, and each read path names the one it means.

    Note `expected_wake_condition` is a property of a MACHINE shutdown
    (when will this box come back), so the process-exit path leaves it
    untouched rather than nulling it. The runner never sends one.

    Creates a heartbeat row if none exists yet (the scaler workflow can
    announce a shutdown for a machine that hasn't checked in yet, e.g.
    on first provisioning). Idempotent on repeated calls -- each call
    overwrites the prior notice OF ITS OWN KIND for that machine.

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
    if reason in PROCESS_EXIT_SHUTDOWN_REASONS:
        conn.execute(
            """
            INSERT INTO heartbeats
              (machine, last_seen, last_process_exit_at, process_exit_reason)
            VALUES (?,?,?,?)
            ON CONFLICT(machine) DO UPDATE SET
              last_process_exit_at=excluded.last_process_exit_at,
              process_exit_reason=excluded.process_exit_reason
            """,
            (machine, _NEVER_HEARTBEATED_SENTINEL, now, reason),
        )
        return
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


def announced_offline_at(last_shutdown_at, last_process_exit_at):
    """The most recent "this machine announced it was going away" stamp.

    Exists to keep lifecycle_state's INPUTS equivalent to what they were
    before the columns above were split, without giving lifecycle_state a
    second parameter or a reason-awareness it does not want. It is the
    read-side counterpart of the write-side routing: the fence and the
    reaper ask specifically about a MACHINE shutdown and so read
    last_shutdown_at directly, while lifecycle_state asks the looser
    question "did this machine announce anything on the way out" -- for
    which a runner process exit has always counted, and must keep counting.

    That is load-bearing, not theoretical: experiment_runner announces
    "runner_drain_complete" on a signal-induced drain precisely so
    /shadow/status reports gracefully_offline rather than stale for a
    manual `systemctl stop` (the path the scaler never sees). Feeding
    lifecycle_state only last_shutdown_at would silently regress every such
    machine to `stale`.

    Pure function. Returns the raw ISO string of whichever stamp is later,
    so lifecycle_state's own parsing and malformed-is-missing contract are
    unchanged. Returns last_shutdown_at when neither parses (lifecycle_state
    treats that as missing either way).
    """
    parsed = []
    for raw in (last_shutdown_at, last_process_exit_at):
        dt = _parse_utc(raw)
        if dt is not None:
            parsed.append((dt, raw))
    if not parsed:
        return last_shutdown_at
    return max(parsed)[1]


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
