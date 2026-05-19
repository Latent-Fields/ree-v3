# Experiment Coordinator -- Plan of Record

Cross-session resume primitive. Read this first.

## Why this exists

Git is currently used as three things at once: a version-controlled evidence
store, the experiment-queue mutex, and a live heartbeat bus. The recurring
failure is `git pull --rebase --autostash` on a 1-minute timer fighting
uncommitted local edits and silently stashing/reverting them (incidents:
2026-04-29 EXQ-232, 2026-05-08 substrate_queue.json, 2026-05-10 MECH-314/318,
2026-05-18 e154c853 governance regen). The fix is to move *coordination* off
git entirely and let git only store committed evidence.

## Architecture (target end-state)

- One always-on host (`ree-cloud-1`) runs two systemd services:
  - `coordinator` -- stdlib `http.server` + SQLite (WAL). The single writer
    of queue/claim state. Zero third-party dependencies (deliberate: the
    box is a minimal CX22; a venv/dependency is itself a fragility source).
  - `sync_daemon` -- the ONLY process that runs git against `REE_assembly`
    for coordination. Writes result manifests, commits, pushes; snapshots
    the live queue back to `experiment_queue.json`.
- Every machine joins a WireGuard mesh. The coordinator binds to the
  WireGuard interface IP only (never 0.0.0.0). Per-worker bearer token in
  `Authorization: Bearer <token>` as defense-in-depth.
- Atomic claim = one conditional `UPDATE ... WHERE status='pending' OR
  stale` inside a `BEGIN IMMEDIATE` transaction. SQLite serializes the
  writer; the conditional UPDATE *is* the mutex. No push race, no
  non-fast-forward rollback dance.

## API contract

All endpoints require `Authorization: Bearer <token>`. Body is JSON unless
noted. The runner's existing ordering invariant is preserved: status, then
results, then queue removal.

| Method/Path | Replaces (experiment_runner.py) | Notes |
|---|---|---|
| GET  `/health` | -- | unauthenticated liveness |
| POST `/claim` | `attempt_claim()` ~:1707 | body `{queue_id, machine, git_verdict?}`; returns `{verdict: ok|already_claimed|error, diverged: bool}` |
| POST `/claim/release` | `release_claim()` | Phase 2 only; body `{queue_id, machine}`; owning machine releases a live coordinator claim |
| POST `/heartbeat` | `runner_remote_control.push_heartbeat` | body `{machine, state, current_exq, progress, gpu}` |
| GET  `/commands?machine=X` | `read_commands_file()` | returns pending pause/resume/stop/force_stop/kick/release_claim |
| POST `/status` | `git_push_status()` ~:1920 | body `{machine, status_json}` |
| POST `/result` | `git_push_results()` ~:1926 | gzip JSON body, up to 32MB; idempotent on `run_id` |
| POST `/queue/remove` | `git_push_queue()` ~:1942 | body `{queue_id, reason}` |
| GET  `/shadow/divergence` | -- | shadow audit: rows where coordinator verdict != git verdict |
| GET  `/shadow/status` | -- | one-call soak snapshot: traffic + divergence + per-machine heartbeat freshness (backs `check_shadow.py`) |

Result payload is a single flat JSON document. Verified 2026-05-19: 0 of the
experiment scripts emit sidecar artifacts (png/npz/csv/savefig); manifest
sizes median ~5KB, max ~7MB -> no multipart, gzip single body, 32MB cap.

## SQLite schema

See `schema.sql`. Tables: `experiments` (mirrors validate_queue.py item
schema), `results` (PK run_id -> idempotency), `heartbeats` (PK machine),
`commands`, `claim_log` (shadow divergence audit).

## Phased rollout (each phase independently reversible via COORDINATION_MODE)

- **Phase 0 -- standup.** Coordinator + WireGuard + sync_daemon deployed on
  ree-cloud-1. DB seeded from `experiment_queue.json`. Nothing calls it.
- **Phase 1 -- shadow.** Runners still git-claim (authoritative) AND
  best-effort report to the coordinator. Coordinator runs its own claim
  logic against a mirror that follows git, logs every (git_verdict,
  coord_verdict) pair. Watch `/shadow/divergence` for a few days. Near-zero
  divergence = coordinator logic proven. ZERO risk: git stays source of
  truth; the shadow client swallows all errors and never blocks the runner.
- **Phase 2 -- claim cutover.** Claiming goes through authoritative
  `POST /claim`; git-claim is disabled. Results/status/queue removal still
  use the existing git path, with coordinator result/queue reports as a
  second channel. Required mode pair: workers set
  `COORDINATION_MODE=coordinator`, service sets `COORDINATOR_MODE=coordinator`,
  sync daemon sets `SYNC_MODE=coordinator` so git refreshes queue metadata
  without overwriting DB claim state. Coordinator claim errors are retry-only:
  the runner skips the item and waits instead of running unclaimed work. Do
  not run a mixed claim-authority fleet during cutover; drain/pause git or
  shadow workers before resuming coordinator-mode workers.
- **Phase 3 -- result cutover.** Results to coordinator only; sync_daemon is
  sole git writer. The `runner_remote_control` git heartbeat push (the
  autostash-war bug source) is disabled. Repair cron kept one cycle as
  backstop, then retired.
- **Phase 4 -- cleanup.** Delete dead git-claim code, retire repair cron,
  optionally point serve.py at the coordinator.

`COORDINATION_MODE` env on each runner: `git` (default, byte-identical to
today), `shadow` (Phase 1), `coordinator` (Phase 2+). Default MUST stay
`git` until shadow has proven out under real multi-machine load.

## Failure handling

- Coordinator down -> `POST /claim` fails, worker waits/retries (same UX as
  "nothing claimable"). A finished run is spooled locally and replayed until
  acked. Idempotent on `run_id` -> double-send after partition is a no-op.
- Worker dies mid-run -> heartbeat TTL stale -> reaper flips claim back to
  `pending` and logs (this is the long-stubbed `recover_stale_claims()`).
- Coordinator disk loss -> sync_daemon's periodic queue snapshot in git is
  the disaster-recovery seed; evidence already committed. Worst case: a few
  experiments re-run, deduped in evidence by `run_id`.

Net: every failure mode is "pause and recover," never "silently revert."

## Explorer / governance compatibility

serve.py `/machines` and governance read `runner_heartbeats/<host>.json` and
`experiment_queue.json` from disk. The sync_daemon writes those legacy files
as a *derived view* from the DB so the explorer needs zero changes during
migration. Cutting serve.py over to the coordinator API is optional Phase-4
cleanup, not migration-critical.

## File map

| File | Role |
|---|---|
| `schema.sql` | DB schema (idempotent CREATE IF NOT EXISTS) |
| `db.py` | connection (WAL), atomic-claim transaction, mirror upsert |
| `app.py` | stdlib ThreadingHTTPServer coordinator, bearer auth |
| `seed_from_queue.py` | load experiment_queue.json into the DB |
| `sync_daemon.py` | shadow: git-queue -> DB mirror reconcile + divergence report; Phase-3 git-writer stub |
| `../coordinator_client.py` | env-gated best-effort runner shim (default git = no-op) |
| `test_shadow_e2e.py` | local end-to-end self-test (no WireGuard needed) |
| `deploy/` | WireGuard + systemd + token-gen runbook |

## Status

- 2026-05-19: Phase 0-1 build. Shadow service + client shim + additive
  env-gated runner hooks + local e2e test + deploy runbook. No cutover.
  Coordinator host decided = `ree-cloud-1`. Auth = WireGuard + per-worker
  bearer token. HTTP layer = pure stdlib (no FastAPI; CX22 anti-fragility).
- 2026-05-19: Phase-2 claim-cutover code added but not deployed by default.
  `COORDINATION_MODE=coordinator` now makes `/claim` authoritative in the
  runner; release and queue-remove endpoints prevent stuck or duplicate DB
  claims; `SYNC_MODE=coordinator` preserves DB claim state while git remains
  the worklist/result transport. Default stays `git`.
