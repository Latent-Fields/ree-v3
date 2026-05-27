# Phase 3 -- result cutover (plan of record)

Cross-session resume primitive for **sole git writer** migration. Read with
`PLAN.md`, `OPERATOR_GUIDE.md`, `deploy/FLEET_CHECKLIST.md`, and
`SOAK_LOG.md`.

**Do not enable Phase 3 in production until `phase3_git_writer()` is fully
implemented, `PHASE3_GIT_WRITER_READY` is deliberately set `True`, and both
`phase3_preflight.py` and `phase3_verify.py` exit 0 on the live fleet.**

---

## What Phase 3 changes

| Path | Phase 2 (today) | Phase 3 (target) |
|------|-----------------|------------------|
| Claims | Coordinator authoritative | Unchanged |
| Results / status / queue file on git | Runners `git_push_*` + `--auto-sync` | Coordinator ingest; **sync_daemon** commits/pushes |
| Heartbeats on git | `runner_remote_control.push_heartbeat` every tick | sync_daemon writes derived `runner_heartbeats/*.json`; runner git heartbeat push **off** |
| Hub `SYNC_MODE` | `coordinator` | `authoritative` + `--i-understand-phase3` |
| Autostash incident class | Still possible via heartbeat/status pushes | **Retired** for coordination paths |

Phase 3 is **not** a config flip alone: `sync_daemon.py` must implement the
git writer loop before `SYNC_MODE=authoritative` is safe.

### Required env knobs (Phase 3 only; unset = Phase 2 default = bit-identical)

| Knob | Process | Purpose |
|------|---------|---------|
| `COORDINATOR_SPOOL_DIR` | coordinator (`app.py`) | Directory where `POST /result` persists raw manifest bytes for later commit. Unset -> bytes are dropped after metadata-record (Phase 2 behaviour). |
| `COORDINATOR_SPOOL_DIR` | sync_daemon (same dir) | Source the writer reads pending manifests from. Without it the writer refuses to run even with `PHASE3_GIT_WRITER_READY=True`. |
| `PHASE3_REE_ASSEMBLY` | sync_daemon | Path to the hub's REE_assembly checkout (default `/home/ree/REE_Working/REE_assembly`). |
| `PHASE3_ASSEMBLY_BRANCH` | sync_daemon | Target branch on origin (default `master`). |
| `PHASE3_BATCH_SIZE` | sync_daemon | Max manifests committed per tick (default 32; bounds tick latency and rollback blast radius). |
| `PHASE3_DISABLE_RUNNER_RESULT_PUSH` | runner (`experiment_runner.py`) | When `1`, `git_push_results` is a no-op -- sync_daemon's writer owns result commits. Set on every worker BEFORE flipping `PHASE3_GIT_WRITER_READY=True`; running both is unsafe (autostash race). |
| `PHASE3_DISABLE_RUNNER_QUEUE_PUSH` | runner | When `1`, `git_push_queue` is a no-op -- sync_daemon's queue snapshot writeback (PLAN.md step 5) owns it. Set on every worker BEFORE flipping `PHASE3_QUEUE_WRITER_READY=True`. **Do not set if `PHASE3_QUEUE_WRITER_READY` is still False**, or queue updates stop reaching origin entirely. |
| `PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH` | runner + `runner_remote_control.py` | When `1`, `push_heartbeat`, `push_commands`, AND `git_push_status` are no-ops -- sync_daemon's derived-heartbeats writeback (PLAN.md step 6) owns them. **Do not set until step 6 is wired**, or heartbeats / commands / per-machine status stop reaching origin. |
| `PHASE3_QUEUE_WRITER_READY` | sync_daemon | Implementation flag for the queue snapshot writer. Independent of `PHASE3_GIT_WRITER_READY` so result-cutover and queue-cutover can stage separately. Default `False` -> writer stub-skips with a log line. Flip to `True` only after review + paired with `PHASE3_DISABLE_RUNNER_QUEUE_PUSH=1` on every worker. |
| `PHASE3_REE_V3` | sync_daemon | Path to the hub's ree-v3 checkout (default `/home/ree/REE_Working/ree-v3`). Where the queue writer commits + pushes from. |
| `PHASE3_REE_V3_BRANCH` | sync_daemon | Branch on ree-v3's origin (default `main`). |
| `PHASE3_HEARTBEAT_WRITER_READY` | sync_daemon | Implementation flag for the heartbeat + status writer. Independent flag again. Default `False` -> writer stub-skips. Flip to `True` only after every worker is running a `coordinator_client` that sends the rich payload (PLAN.md step 6) AND `PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH=1` is set on every worker. Old clients (no payload) leave their column NULL and the writer silently skips them; if you flip this flag without rolling out the new client, the affected machines stop producing heartbeat files entirely. |

The `PHASE3_GIT_WRITER_READY` constant in `sync_daemon.py` is the explicit
implementation flag. It stays `False` until the writer body has been
exercised against a test fleet; flipping it to `True` is a deliberate code
change reviewed in a separate commit, not a config tweak.

### Writer behaviour (current sketch, 2026-05-27)

`phase3_git_writer()` already implements steps 1-4 of the planned tick:

1. **Read pending.** `manifest_spool.list_pending_run_ids()` enumerates
   run_ids that have both `<run_id>.json` and `<run_id>.meta.json` on the
   spool (atomicity-preserving; orphaned `.tmp` files are ignored).
2. **Stage.** For each manifest in the batch, derive the
   `evidence/experiments/...` target path (manifest_relpath hint
   preferred, default `evidence/experiments/<run_id>.json` otherwise),
   write the bytes, `git add` the file.
3. **Commit + push.** Single commit per tick (`phase3: N v3 result
   manifest(s) <date>`), single push `HEAD:<branch>`. **Never** runs
   `git pull --rebase --autostash` -- a non-fast-forward push fails
   the tick loudly and leaves the spool intact for the next attempt.
4. **Reconcile.** On successful push, `UPDATE results SET committed_at`
   for the batch and `manifest_spool.delete_manifest()` for each entry.

Refuses to run when any of these is true:

- `PHASE3_GIT_WRITER_READY=False` (the default).
- `COORDINATOR_SPOOL_DIR` is unset (nothing to commit).
- The REE_assembly working tree on the hub is dirty (operator must
  investigate; Phase 3 deliberately retires autostash).
- `git push` is rejected (probably non-fast-forward; hub is behind
  origin; operator must `git pull --ff-only` by hand).

Idle ticks (spool empty) return `True` immediately as a successful no-op.

### PLAN.md steps 4, 5, 6 all landed

Step 5 (**queue snapshot writeback**) LANDED 2026-05-27 alongside the
authoritative-mode reconcile `upsert_only=True` semantics. `phase3_queue_writer`
materialises pending/claimed items from the `experiments` table into
`experiment_queue.json` on the hub's ree-v3 checkout and pushes `origin/main`.
Gated by its own `PHASE3_QUEUE_WRITER_READY` flag. Operator hand-edits
to `experiment_queue.json` under the queue-writer regime MUST be additions
only -- use `POST /queue/remove` via the coordinator API to drop items, since
items missing from the file are NOT deleted from the DB (the DB row's
`status='completed'` is the authoritative "this was done" record and must
survive the writeback round-trip).

Step 6 (**derived heartbeats + runner_status writeback**) LANDED 2026-05-27.
`phase3_heartbeat_writer` materialises `evidence/experiments/runner_heartbeats/
<machine>.json` AND `evidence/experiments/runner_status/<machine>.json` from
`heartbeat_payload_json` and `status_payload_json` columns on the `heartbeats`
table. Gated by `PHASE3_HEARTBEAT_WRITER_READY` (independent flag). Payloads
travel runner -> coordinator via the extended POST `/heartbeat` (added
`payload` field) and POST `/status` (now persists the full body, was
receipt-only). The writer outputs the stored payload byte-for-byte so
consumers (explorer, cloud-scaler workflow, governance) see the same
file shape they did under the runner's old git push -- only the transport
changed. Legacy clients that don't send a payload (older `coordinator_client`
versions) leave their column NULL and are silently skipped by the writer;
their `lifecycle_state` still derives from the structured columns.

---

## Preconditions checklist (machine-readable categories)

Run before any maintenance window:

```bash
/opt/local/bin/python3 ~/REE_Working/ree-v3/coordinator/phase3_preflight.py
```

| Category | ID | Pass criterion |
|----------|-----|----------------|
| **hub** | `hub_health` | `GET /health` -> `ok:true`, `mode:coordinator` |
| **hub** | `hub_sync_mode_safe` | Hub `/etc/ree-coordinator.env` has `SYNC_MODE=coordinator` (not `authoritative`) |
| **hub** | `sync_daemon_active` | `ree-sync-daemon` active on ree-cloud-1 |
| **hub** | `hub_git_clean` | `REE_assembly` checkout on hub: no precious dirty files |
| **implementation** | `phase3_writer_stub` | `PHASE3_GIT_WRITER_READY` is false (writer not half-enabled) |
| **soak** | `phase2_shadow_metrics` | `check_shadow.py` exit 0 (HEALTHY, blocking div 0) when hub reachable |
| **fleet** | `coordination_mode_uniform` | Every experiment host: `COORDINATION_MODE=coordinator` (no `git`/`shadow` mix) |
| **fleet** | `no_mixed_result_writers` | No host still configured to git-push results while hub is authoritative (pre-check: all still on Phase 2 git path is OK) |
| **data** | `db_schema_present` | Local `schema.sql` tables: experiments, results, heartbeats, commands, claim_log |
| **data** | `orphaned_claims` | No `experiments.status=claimed` rows with stale/missing owner heartbeat (hub DB) |
| **reachability** | `coordinator_api` | Authenticated `GET /shadow/status` succeeds |

Categories map 1:1 to `phase3_preflight.py` check IDs. `--dry-run` skips SSH;
`--mock` forces network checks to SKIP (tests only).

### Human gates (not automated)

- [ ] **3+ calendar days** of stable Phase 2 after 2026-05-21 cutover
- [ ] At least one full EXQ: coordinator claim -> run -> git result -> queue remove
- [ ] Operator has read `SOAK_LOG.md` and classified any raw divergence rows
- [ ] Maintenance window scheduled; fleet drain playbook understood
- [ ] `phase3_git_writer()` implementation reviewed and smoke-tested offline

---

## Post-cutover verification checklist

Run **after** Phase 3 config is applied (hub authoritative, workers stopped
git-pushing results/heartbeats):

```bash
/opt/local/bin/python3 ~/REE_Working/ree-v3/coordinator/phase3_verify.py
```

| Category | ID | Pass criterion |
|----------|-----|----------------|
| **hub** | `hub_sync_mode_authoritative` | `SYNC_MODE=authoritative` on hub |
| **hub** | `sync_daemon_phase3_tick` | sync_daemon running; log shows phase3 tick (not stub refusal) |
| **hub** | `hub_git_writer_only` | Recent `REE_assembly` commits attributable to sync_daemon path only |
| **fleet** | `workers_no_result_git_push` | Every worker's runner env has `PHASE3_DISABLE_RUNNER_RESULT_PUSH=1`; recent runner logs show `[runner] phase3 gate: skipping git_push_results` |
| **fleet** | `heartbeat_git_retired` | When step 6 has landed: every worker has `PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH=1`; recent `runner_remote_control` logs show the gate-active line. SKIP until step 6 is wired. |
| **data** | `results_drained` | `results.committed_at` populated for pending rows (no stuck spool) |
| **data** | `queue_snapshot_fresh` | `experiment_queue.json` on origin matches coordinator removals |
| **explorer** | `derived_heartbeats` | `runner_heartbeats/*.json` updating without per-runner git push |
| **soak** | `claims_still_healthy` | `check_shadow.py` exit 0 (claim path unchanged) |

Checks marked **SKIP** in `phase3_verify.py` until the corresponding code
landed (see file header). Re-run verify until all required IDs are PASS.

---

## Cutover procedure (operator)

The cutover window assumes every expected peer is `lifecycle_state=live`
on `/shadow/status` (which lets the post-Phase-3 preflight semantics gate
on positive signal instead of SSH-pinging). Surge-only workers (today
`ree-cloud-4`) are powered off most of the time; the wake-fleet helper
brings them up and pauses the scaler so they stay up through the window.

1. `deploy/phase3_wake_fleet.sh` -> disables the `cloud-scaler.yml`
   workflow, powers on any offline cloud worker via `hcloud`, polls
   `/shadow/status` until every expected peer (ree-cloud-1..4 +
   DLAPTOP-4.local) shows `lifecycle_state=live`. Exits 0 when ready,
   non-zero on timeout. Requires `HCLOUD_TOKEN` env var; reads
   `COORDINATOR_URL` + `COORDINATOR_LOCAL_TOKEN` from
   `REE_assembly/coordinator.env`.
2. `phase3_preflight.py` -> exit 0
3. Drain fleet (same discipline as `deploy/phase2_cutover.sh`)
4. `deploy/phase3_cutover.sh` (calls preflight; refuses if not green)
5. Hub: set `SYNC_MODE=authoritative`, enable writer, restart sync_daemon
6. Workers: set `PHASE3_DISABLE_RUNNER_RESULT_PUSH=1` in each worker's
   runner env (e.g. `/etc/systemd/system/ree-runner.service.d/shadow.conf`
   on the clouds) and restart the runner. This MUST happen alongside the
   `PHASE3_GIT_WRITER_READY=True` flip on the hub -- running both the
   writer AND the runner's `git_push_results` is the autostash-war
   scenario Phase 3 exists to prevent. Do NOT set
   `PHASE3_DISABLE_RUNNER_QUEUE_PUSH` or
   `PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH` -- those gates exist for the
   later steps 5/6 cutovers and would stop their respective artifacts
   from reaching origin until those writers land.
7. `phase3_verify.py` -> exit 0
8. `deploy/phase3_release_fleet.sh` -> re-enables `cloud-scaler.yml`. Run
   this ONLY after verify is all PASS; if verify fails, leave the scaler
   paused while rolling back.
9. Resume runners; watch manifests and queue on origin for one full EXQ

**Script entry:** `ree-v3/coordinator/deploy/phase3_cutover.sh`.
**Wake / release pair:** `deploy/phase3_wake_fleet.sh` +
`deploy/phase3_release_fleet.sh`.

### Why scaler-pause instead of marker-queue-entries

Earlier design considered seeding the queue with `machine_affinity=
ree-cloud-4` markers to keep cloud-4's claimable count > 0 (preventing
the scaler from shutting it down). That works but introduces a window
where a real runner could claim and execute the marker mid-cutover.
Pausing the scaler workflow via `gh workflow disable` is cleaner: one
command, no queue mutation, and the pause covers all scaler decisions
(not just cloud-4's). The 15-minute scheduled cron simply doesn't fire
during the window. Re-enable post-verify and the scaler resumes its
next decision cycle.

---

## Rollback

Fast rollback (claims stay on coordinator; git writers restored):

1. Drain fleet.
2. Hub: `SYNC_MODE=coordinator` (not `authoritative`); restart `ree-sync-daemon`.
3. Workers: re-enable `--auto-sync` git pushes for results/status/queue/heartbeats.
4. Restart runners.
5. Run `phase3_preflight.py` (should pass Phase 2 posture again).

Full rollback to Phase 1 shadow (only if claim cutover must be undone):

- Follow `OPERATOR_GUIDE.md` Phase-2 rollback: hub `COORDINATOR_MODE=shadow`,
  `SYNC_MODE=shadow`, workers `COORDINATION_MODE=shadow` or unset.

Evidence already committed to git is never deleted by rollback.

---

## Explicit do-not-enable list

Until engineering sign-off on `phase3_git_writer`:

- Do **not** set `SYNC_MODE=authoritative` on ree-cloud-1
- Do **not** pass `--i-understand-phase3` to sync_daemon in systemd
- Do **not** set `PHASE3_GIT_WRITER_READY = True` in `sync_daemon.py`
- Do **not** run `deploy/phase3_cutover.sh` except in a dry rehearsal
  (`phase3_preflight.py --dry-run` only)
- Do **not** mix fleet modes: one `COORDINATION_MODE` for all experiment hosts
- Do **not** disable runner git pushes while hub is still `SYNC_MODE=coordinator`
  (would stall results with no writer). The flip from
  `PHASE3_DISABLE_RUNNER_RESULT_PUSH=0` (Phase 2) to `=1` must happen
  in the same maintenance window as the hub's writer-ready flag flip.
- Do **not** set `PHASE3_DISABLE_RUNNER_QUEUE_PUSH=1` or
  `PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH=1` before PLAN.md steps 5 and 6
  are wired in `sync_daemon.phase3_git_writer`. Those gates assume a
  sync_daemon counterpart exists to publish what the runner stops
  pushing; setting them prematurely means the queue / heartbeats /
  commands / per-machine status stop reaching origin entirely.
- Do **not** run `deploy/phase3_wake_fleet.sh` outside a real or rehearsed
  cutover window -- it disables the cloud-scaler workflow, which pauses
  ALL scaling decisions. Leaving the scaler disabled past the cutover
  window means workers won't auto-start when the queue grows.

---

## Tooling map

| Tool | Role |
|------|------|
| `deploy/phase3_wake_fleet.sh` | Pre-cutover prep: pauses scaler, wakes offline workers, waits for lifecycle=live across the fleet |
| `phase3_preflight.py` | Pre-cutover gate (exit 0/1) |
| `phase3_verify.py` | Post-cutover gate (exit 0/1; SKIP until implemented) |
| `deploy/phase3_cutover.sh` | Drains + hub flip; **refuses** if preflight fails |
| `deploy/phase3_release_fleet.sh` | Post-verify: re-enables scaler workflow |
| `sync_daemon.phase3_git_writer()` | Sole writer implementation (stub) |
| `GET /api/coordinator/phase3/preflight` | Read-only summary in Explorer (serve.py) |

---

## Status

- 2026-05-21: Phase 3 **substrate only** (this doc, preflight/verify CLIs,
  cutover script shell, sync_daemon scaffold). Phase 2 remains live.
  **Not safe to cut over.**
- 2026-05-27: writer ahead-of-origin guard landed (2b13f68);
  `POST /shutdown_notify` + lifecycle_state on `/shadow/status` landed
  (0e4a815); cloud-scaler workflow announces shutdowns (dda047b); runner
  SIGTERM handler announces (f0568b4); cutover playbook gains wake/release
  fleet pair (2e5e991); preflight reads lifecycle_state instead of SSH
  (90035e5); writer review HIGH-1 stale-origin fix (f95f9db) +
  HIGH-2 foreign-commit-rejection fix (043e06e); runner-push gating
  env flags wired (this commit).
- `PHASE3_GIT_WRITER_READY` still `False` -- remaining blockers before
  flipping: writer-implementation human sign-off post-fix, plus
  PLAN.md steps 5-6 (queue-snapshot writeback, derived heartbeats).
  The result-push gate's runner counterpart is ready; the queue and
  heartbeat gates exist as scaffolding but their sync_daemon
  counterparts are NOT yet implemented.
