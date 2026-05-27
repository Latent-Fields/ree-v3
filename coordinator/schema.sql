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
    updated_at          TEXT NOT NULL
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
    expected_wake_condition TEXT               -- free text, nullable
);

-- Pending + recent remote-control commands.
CREATE TABLE IF NOT EXISTS commands (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    machine    TEXT NOT NULL,
    kind       TEXT NOT NULL,              -- pause|resume|stop|force_stop|kick|release_claim
    args       TEXT,
    issued_by  TEXT,
    issued_at  TEXT NOT NULL,
    acked_at   TEXT
);

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
