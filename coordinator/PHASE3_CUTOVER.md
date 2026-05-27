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

### Deferred to follow-up PRs

Steps 5 and 6 from PLAN.md are not yet in the sketch:

- **Queue snapshot** -- materialise completed/removed entries from the
  `experiments` table into `experiment_queue.json` and push `ree-v3`.
  Touches the ree-v3 checkout on the hub; will land once the results-side
  path has soaked.
- **Derived heartbeats** -- write `evidence/experiments/runner_heartbeats/*.json`
  and `runner_status/*.json` from the `heartbeats` table, replacing the
  per-runner `runner_remote_control.push_heartbeat` git push (the original
  autostash-war bug source).

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
| **fleet** | `workers_no_result_git_push` | Runners not calling `git_push_results` / `git_push_queue` for coordination |
| **fleet** | `heartbeat_git_retired` | `push_heartbeat` git path disabled or no-op under Phase 3 |
| **data** | `results_drained` | `results.committed_at` populated for pending rows (no stuck spool) |
| **data** | `queue_snapshot_fresh` | `experiment_queue.json` on origin matches coordinator removals |
| **explorer** | `derived_heartbeats` | `runner_heartbeats/*.json` updating without per-runner git push |
| **soak** | `claims_still_healthy` | `check_shadow.py` exit 0 (claim path unchanged) |

Checks marked **SKIP** in `phase3_verify.py` until the corresponding code
landed (see file header). Re-run verify until all required IDs are PASS.

---

## Cutover procedure (operator)

1. `phase3_preflight.py` -> exit 0
2. Drain fleet (same discipline as `deploy/phase2_cutover.sh`)
3. `deploy/phase3_cutover.sh` (calls preflight; refuses if not green)
4. Hub: set `SYNC_MODE=authoritative`, enable writer, restart sync_daemon
5. Workers: disable git result/queue/heartbeat pushes (env flags TBD in code)
6. `phase3_verify.py` -> exit 0
7. Resume runners; watch manifests and queue on origin for one full EXQ

**Script entry:** `ree-v3/coordinator/deploy/phase3_cutover.sh`

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
  (would stall results with no writer)

---

## Tooling map

| Tool | Role |
|------|------|
| `phase3_preflight.py` | Pre-cutover gate (exit 0/1) |
| `phase3_verify.py` | Post-cutover gate (exit 0/1; SKIP until implemented) |
| `deploy/phase3_cutover.sh` | Drains + hub flip; **refuses** if preflight fails |
| `sync_daemon.phase3_git_writer()` | Sole writer implementation (stub) |
| `GET /api/coordinator/phase3/preflight` | Read-only summary in Explorer (serve.py) |

---

## Status

- 2026-05-21: Phase 3 **substrate only** (this doc, preflight/verify CLIs,
  cutover script shell, sync_daemon scaffold). Phase 2 remains live.
  **Not safe to cut over.**
