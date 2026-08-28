-- Experiment coordinator schema. Idempotent. Applied at startup.
-- Mirrors the validate_queue.py item schema so nothing downstream changes.

PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- One row per queue item. Mirror of experiment_queue.json items[].
CREATE TABLE IF NOT EXISTS experiments (
    queue_id            TEXT PRIMARY KEY,
    script              TEXT NOT NULL,
    priority            INTEGER NOT NULL DEFAULT 1,
    machine_affinity    TEXT NOT NULL DEFAULT 'any',
    status              TEXT NOT NULL DEFAULT 'pending',  -- pending|claimed|failed|completed
    estimated_minutes   REAL,
    supersedes          TEXT,
    claim_id            TEXT,
    backlog_id          TEXT,
    title               TEXT,
    note                TEXT,
    force_rerun         INTEGER NOT NULL DEFAULT 0,
    claimed_by_machine  TEXT,
    claimed_at          TEXT,                              -- ISO-8601 UTC
    item_json           TEXT NOT NULL,                     -- full original item, lossless
    updated_at          TEXT NOT NULL,
    -- Why the row went terminal. status='completed' is overloaded: it means
    -- "no longer claimable", NOT "ran to a scientific outcome". An operator
    -- cancellation (POST /queue/remove), a runner ERROR and a scientific FAIL
    -- all land here identically, so a `status='completed' LEFT JOIN results ->
    -- no row` query cannot tell a crash from a deliberate cancellation. These
    -- two columns preserve the `reason` that mark_queue_removed() previously
    -- accepted and discarded. NULL on rows written before this migration and
    -- on rows that never went through mark_queue_removed.
    removal_reason      TEXT,                              -- PASS|FAIL|ERROR|operator string
    removed_at          TEXT                               -- ISO-8601 UTC
);

-- One row per completed run. PK = run_id => result delivery is idempotent.
CREATE TABLE IF NOT EXISTS results (
    run_id          TEXT PRIMARY KEY,
    queue_id        TEXT,
    machine         TEXT,
    outcome         TEXT,                  -- PASS|FAIL|ERROR|UNKNOWN
    manifest_sha256 TEXT,
    manifest_bytes  INTEGER,
    received_at     TEXT NOT NULL,
    committed_at    TEXT                    -- set by sync_daemon in Phase 3
);

-- One row per machine. PK = machine.
-- `state` is the runner-process state (idle/running/draining). The
-- lifecycle_state (live/gracefully_offline/stale) is derived at read time
-- in /shadow/status from (last_seen, last_shutdown_at) -- not stored.
-- last_shutdown_at + shutdown_reason + expected_wake_condition are set by
-- POST /shutdown_notify. They survive across heartbeats so a machine that
-- boots, runs, and shuts down again carries the latest cycle's record.
-- POST /shutdown_notify carries TWO different events and stores them in
-- SEPARATE column pairs: a MACHINE shutdown ("this box is going away",
-- e.g. the scaler's scaler_idle_after_grace) lands in last_shutdown_at +
-- shutdown_reason and is what arms the claim fence; a RUNNER PROCESS EXIT
-- ("the process is exiting, the box stays up" -- runner_signal_exit /
-- runner_drain_complete) lands in last_process_exit_at +
-- process_exit_reason and deliberately does not fence. They were one pair
-- until 2026-07-30, when a process exit was measured overwriting the
-- scaler's reason ~1s after it was written and disarming the fence.
-- See db.record_shutdown_notice.
CREATE TABLE IF NOT EXISTS heartbeats (
    machine                 TEXT PRIMARY KEY,
    last_seen               TEXT NOT NULL,     -- ISO-8601 UTC
    state                   TEXT,
    current_exq             TEXT,
    progress_json           TEXT,
    gpu_json                TEXT,
    seconds_elapsed         INTEGER,
    seconds_remaining       INTEGER,
    last_shutdown_at        TEXT,              -- ISO-8601 UTC, nullable
    shutdown_reason         TEXT,              -- free text, nullable
    expected_wake_condition TEXT,              -- free text, nullable
    last_process_exit_at    TEXT,              -- ISO-8601 UTC, nullable
    process_exit_reason     TEXT,              -- free text, nullable
    -- PLAN.md step 6: full payloads from the runner so sync_daemon can
    -- materialise rich runner_heartbeats/*.json and runner_status/*.json
    -- files in REE_assembly. Populated by POST /heartbeat (payload field)
    -- and POST /status (body). Stored verbatim as JSON text; the writer
    -- writes them out byte-for-byte preserving operator-visible shape.
    heartbeat_payload_json  TEXT,              -- nullable: old clients
    status_payload_json     TEXT               -- nullable: ditto
);

-- Pending + recent remote-control commands.
-- A command is "pending" while acked_at IS NULL (GET /commands returns these).
-- The runner acks via POST /commands/ack, which stamps acked_at + the terminal
-- result_status (done|failed) + a free-text result_note. acked rows are
-- retained as an audit trail (and so a re-fetch never re-delivers them).
CREATE TABLE IF NOT EXISTS commands (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    machine       TEXT NOT NULL,
    kind          TEXT NOT NULL,           -- pause|resume|stop|force_stop|suspend|resume_run|kick|release_claim|reclassify
    args          TEXT,                    -- JSON object, nullable
    issued_by     TEXT,
    issued_at     TEXT NOT NULL,
    acked_at      TEXT,                    -- set by POST /commands/ack
    result_status TEXT,                    -- done|failed (set at ack time)
    result_note   TEXT                     -- free-text outcome (set at ack time)
);
CREATE INDEX IF NOT EXISTS idx_commands_pending
    ON commands(machine, acked_at);

-- Append-only heartbeat HISTORY, sibling to `heartbeats` above. `heartbeats`
-- is upserted on every POST /heartbeat and holds only the LATEST tick per
-- machine (PRIMARY KEY(machine)) -- "what was ree-cloud-3 doing at 04:00
-- last Tuesday" is unanswerable from it, and the only surviving record was
-- git commit history, a poor query interface. This table answers that.
--
-- NOT written on every tick. The runner POSTs /heartbeat every ~5s while an
-- experiment is running (experiment_runner.STATUS_WRITE_INTERVAL), which
-- would be ~720 rows/machine/hour logged unconditionally -- the same
-- growth-without-bound mistake the retired 30-minute git heartbeat liveness
-- tick was pulled for (see CLAUDE.md Coordinator section). A row is
-- appended only when (state, current_exq) differs from the machine's
-- current `heartbeats` row, i.e. one row per experiment start / finish /
-- switch / idle transition, not per tick. See db.upsert_heartbeat.
--
-- Retention: trimmed to the most recent
-- COORDINATOR_HEARTBEAT_LOG_RETENTION_DAYS (default 30) by
-- db.trim_heartbeat_log, invoked opportunistically from upsert_heartbeat at
-- most once per COORDINATOR_HEARTBEAT_LOG_TRIM_INTERVAL_SECONDS (default
-- 3600) -- no separate timer needed on the 3.8 GB hub.
CREATE TABLE IF NOT EXISTS heartbeat_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    machine       TEXT NOT NULL,
    observed_at   TEXT NOT NULL,     -- ISO-8601 UTC, transition time
    state         TEXT,
    current_exq   TEXT,
    payload_json  TEXT               -- verbatim heartbeat payload at the transition, nullable
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_log_machine_time
    ON heartbeat_log(machine, observed_at);

-- Shadow audit. One row per reported claim attempt: what git decided vs
-- what the coordinator's own logic would have decided. diverged=1 rows are
-- the signal the operator watches before advancing past Phase 1.
CREATE TABLE IF NOT EXISTS claim_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    queue_id      TEXT NOT NULL,
    machine       TEXT NOT NULL,
    git_verdict   TEXT,                    -- ok|already_claimed|error|null (unknown)
    coord_verdict TEXT NOT NULL,           -- ok|already_claimed|error
    diverged      INTEGER NOT NULL DEFAULT 0,
    detail        TEXT,
    logged_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claim_log_diverged ON claim_log(diverged);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);

-- ---------------------------------------------------------------------------
-- TASK_CLAIMS.json / TASK_CHIPS.json shadow mirror (PHASE 1, read-only).
-- See REE_assembly/evidence/planning/task_claim_chip_coordinator_migration_plan.md
-- section 5.2 for the reviewed design (D1-D11). The AUTHORITATIVE source for
-- these three tables is git (the REE_Working umbrella repo, NOT this repo) --
-- task_claim_chip_shadow_sync.py reconciles them read-only, on the exact
-- pattern schema.sql already uses for `experiments`/`claim_log` above. No
-- code anywhere in this codebase writes TASK_CLAIMS.json/TASK_CHIPS.json
-- content back to git from these tables; that is PHASE-2, not yet built.
-- ---------------------------------------------------------------------------

-- One row per TASK_CLAIMS.json claims[] entry. PK is COMPOSITE and must stay
-- that way: root CLAUDE.md states the claim key is (session_id, claimed_at),
-- and `task_claim.py close --claimed-at` exists precisely because session_id
-- alone is ambiguous (8/154 live session_ids own more than one claim, 2026-08-26).
CREATE TABLE IF NOT EXISTS task_claims (
    session_id                   TEXT NOT NULL,
    claimed_at                   TEXT NOT NULL,   -- ISO-8601 UTC, 2nd half of the key
    session_label                TEXT NOT NULL DEFAULT '',
    task                         TEXT NOT NULL DEFAULT '',
    status                       TEXT NOT NULL DEFAULT 'active',  -- active|done
    closed_at                    TEXT,            -- committer date of the landing commit
    completion_note              TEXT,
    completion_note_history_json TEXT,            -- JSON array, append-only
    spawned_by                   TEXT,
    entry_json                   TEXT NOT NULL,    -- full original entry, lossless
    updated_at                   TEXT NOT NULL,
    PRIMARY KEY (session_id, claimed_at)
);
CREATE INDEX IF NOT EXISTS idx_task_claims_status ON task_claims(status);

-- `resources` is a LIST and gets a child table, NOT a JSON column -- see
-- design problem D2 in the plan doc. This is what makes a future Phase-2
-- rival lookup a single indexed SELECT inside a BEGIN IMMEDIATE transaction
-- instead of a whole-table JSON scan (i.e. today's git-based best effort,
-- reimplemented server-side with no correctness gain).
CREATE TABLE IF NOT EXISTS task_claim_resources (
    session_id  TEXT NOT NULL,
    claimed_at  TEXT NOT NULL,
    resource    TEXT NOT NULL,           -- repo-relative path, verbatim
    PRIMARY KEY (session_id, claimed_at, resource),
    FOREIGN KEY (session_id, claimed_at)
        REFERENCES task_claims(session_id, claimed_at) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_task_claim_resources_resource
    ON task_claim_resources(resource);

-- One row per TASK_CHIPS.json chips[] entry. PK = chip_ref (unique across
-- every live entry). task_id is NOT the PK: it is null on a chip nobody has
-- ever clicked via spawn_task, so it is nullable with a partial unique index.
CREATE TABLE IF NOT EXISTS chip_ledger (
    chip_ref                     TEXT PRIMARY KEY,
    task_id                      TEXT,
    session_id                   TEXT NOT NULL DEFAULT '',
    session_label                TEXT NOT NULL DEFAULT '',
    title                        TEXT NOT NULL DEFAULT '',
    tldr                         TEXT NOT NULL DEFAULT '',
    prompt                       TEXT,           -- NULL once archived (see D5)
    cwd                          TEXT NOT NULL DEFAULT '',
    origin                       TEXT,           -- spawn_task|headless|hygiene_tick|igw_tick|proposal_tick
    kind                         TEXT,           -- work|decision|report
    urgency                      INTEGER NOT NULL DEFAULT 0,   -- bool
    spawned_at                   TEXT NOT NULL DEFAULT '',
    origin_host                  TEXT,           -- canonical_machine_name()
    origin_host_raw              TEXT,           -- as reported; audit only (see D6)
    status                       TEXT NOT NULL DEFAULT 'open',  -- open|done|withdrawn
    claimed_by                   TEXT,
    claimed_at                   TEXT,
    claim_note                   TEXT,
    claimed_host                 TEXT,           -- canonical
    claimed_host_raw             TEXT,           -- as reported; audit only
    resolved_at                  TEXT,
    resolved_by_session_id       TEXT,
    resolution_note              TEXT,           -- NULL once archived (see D5)
    resolution_note_auto         INTEGER NOT NULL DEFAULT 0,   -- bool
    attached_by_session_id       TEXT,
    attached_at                  TEXT,
    archived_json                TEXT,           -- {file, month, fields[], at}
    prompt_history_json          TEXT,           -- JSON array
    urgency_history_json         TEXT,           -- JSON array
    resolution_note_history_json TEXT,           -- JSON array
    confirmer_verdict_json       TEXT,           -- JSON object
    entry_json                   TEXT NOT NULL,
    updated_at                   TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chip_ledger_task_id
    ON chip_ledger(task_id) WHERE task_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chip_ledger_status  ON chip_ledger(status);
CREATE INDEX IF NOT EXISTS idx_chip_ledger_claimed ON chip_ledger(status, claimed_by);
CREATE INDEX IF NOT EXISTS idx_chip_ledger_spawned ON chip_ledger(status, spawned_at);

-- Shadow soak evidence. One row per reconciliation tick of
-- task_claim_chip_shadow_sync.py -- this is the "N days of zero drift"
-- exit criterion for PHASE-1 (plan doc frontmatter), and the durable record
-- a human reads to judge the soak. Mirrors claim_log's shape (one row per
-- check, a diverged flag, free-text detail) even though the underlying
-- comparison differs: claim_log compares two independently-computed
-- verdicts (git vs coordinator claim logic); this table is a mirror
-- self-consistency check (git content vs what actually landed in the
-- mirror), since nothing yet computes a coordinator-side claim/chip verdict
-- independently of git (that is Phase 2). A drift row here is always
-- anomalous -- it means a mirrored key exists in the DB but the current
-- git-sourced file no longer has it, which should not happen given
-- TASK_CLAIMS.json/TASK_CHIPS.json are append-only registries.
CREATE TABLE IF NOT EXISTS task_claim_chip_drift_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    checked_at       TEXT NOT NULL,     -- ISO-8601 UTC
    source_ref       TEXT,              -- git commit sha the check read
    n_claims_git     INTEGER NOT NULL,
    n_claims_db      INTEGER NOT NULL,
    n_claims_new     INTEGER NOT NULL,
    n_claims_updated INTEGER NOT NULL,
    n_claims_orphan  INTEGER NOT NULL,  -- ACTIVE DB keys absent from git; 0 if healthy
    n_claims_retired INTEGER,           -- `done` keys absent from git: PRUNED, expected, not drift
    n_chips_git      INTEGER NOT NULL,
    n_chips_db       INTEGER NOT NULL,
    n_chips_new      INTEGER NOT NULL,
    n_chips_updated  INTEGER NOT NULL,
    n_chips_orphan   INTEGER NOT NULL,
    n_chips_retired  INTEGER,           -- always 0: chips are never deleted (D5)
    diverged         INTEGER NOT NULL DEFAULT 0,
    detail           TEXT               -- JSON: {"claim_orphans":[...], "chip_orphans":[...]}
);
CREATE INDEX IF NOT EXISTS idx_task_claim_chip_drift_log_diverged
    ON task_claim_chip_drift_log(diverged);
CREATE INDEX IF NOT EXISTS idx_task_claim_chip_drift_log_checked_at
    ON task_claim_chip_drift_log(checked_at);

-- WORKSPACE_STATE.md append intake (PHASE-4 first slice). Append-only spool:
-- clients POST /workspace_state/append, the registry materializer SPLICES
-- pending rows into the live file (byte-preserving) and marks them
-- materialized once the text provably reached git. No ingest of the file's
-- existing entries, no edit/delete verb -- see db._migrate_workspace_state_table.
CREATE TABLE IF NOT EXISTS workspace_state_entries (
    entry_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT NOT NULL,      -- entry heading timestamp (%Y-%m-%dT%H:%M:%SZ)
    text             TEXT NOT NULL,      -- entry body, stripped of leading/trailing newlines
    session_id       TEXT NOT NULL DEFAULT '',
    client_git_write INTEGER NOT NULL DEFAULT 0,  -- 1: client lands it itself; writer only watches
    submitted_at     TEXT NOT NULL,
    submitted_host   TEXT,               -- machine label from the bearer token
    materialized_at  TEXT,               -- NULL until the text provably reached git
    materialized_ref TEXT                -- commit sha that carries (or was seen carrying) it
);
CREATE INDEX IF NOT EXISTS idx_workspace_state_pending
    ON workspace_state_entries(materialized_at)
    WHERE materialized_at IS NULL;
