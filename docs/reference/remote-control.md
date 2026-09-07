# Remote control: telemetry transport, command channel, runner loop placement

Reference detail moved out of [`ree-v3/CLAUDE.md`](../../CLAUDE.md) "Remote Control
(--remote-control flag)" section on 2026-09-07 (context-budget restructure WI-2, plan of record
`REE_assembly/evidence/planning/context_budget_restructure_plan.md`). What an ordinary
`ree-v3` session needs -- that `--remote-control` exists, is default-off and bit-identical when
omitted -- stays inline there.

**Read this before** you change the runner's heartbeat/command handling, add or move a command
kind, touch a `shadow.conf` telemetry gate, or wire anything to the coordinator command
channel. You do NOT need it to run experiments or to start a runner.

Content below is verbatim as it stood inline in `ree-v3/CLAUDE.md`. **Its links and paths are
therefore written from the `ree-v3` repo root**, not from this directory -- left unrewritten on
purpose, so the moved prose stays byte-identical to what it replaced.

---

> **Telemetry is coordinator-only** (Phase 3, 2026-05-29; git render retired 2026-09-06): the runner `POST`s heartbeat/status to `/heartbeat` + `/status` (`heartbeats` table). The runner-side git file/push code is the degraded path behind the worker `shadow.conf` gates (`PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH`, `PHASE3_RUNNER_TELEMETRY_OFF_GIT`, `PHASE3_COMMANDS_OFF_GIT`; template `coordinator/deploy/shadow.conf.worker.example`). Levers, reversal recipe and the retired doctrine verbatim: `REE_Working/docs/skill_archaeology/claude-md-concurrency/a-93-retired-telemetry-git-path-fallback.md` (A-93).
>
> **Live progress:** the `live-status` branch `FLEET_STATUS.md` or coordinator `/shadow/status` (explorer `/machines`). Do not re-enable the hub git writer or its liveness tick to get git-side progress files back (A-93).
>
> **Worker telemetry-off-git gate:** `PHASE3_RUNNER_TELEMETRY_OFF_GIT=1` suppresses ONLY the per-tick in-tree telemetry FILE writes; the coordinator POST stays the transport. Unlike `_HEARTBEAT_WRITE` (hub-only; also gates the command-file writeback and restart-loops a worker, incident 2026-05-30) it is command-channel-safe. Contract: `tests/contracts/test_phase3_telemetry_off_git_gate.py`.
>
> **Command channel is the coordinator:** `POST /commands/issue` (insert), `GET /commands?machine=` (pending = `acked_at IS NULL`), `POST /commands/ack`; db helpers `db.insert_command` / `fetch_pending_commands` / `ack_command`; client helpers `coordinator_client.issue_command` / `fetch_commands` / `ack_command`; `serve.py` issues via `POST /commands/issue`. `PHASE3_COMMANDS_OFF_GIT=1` on every worker makes it the sole channel (it self-guards: with no coordinator URL/token it refuses and falls back to the git command-file, so a worker is never uncontrollable); `PHASE3_COMMANDS_VIA_COORDINATOR=1` is the dual-read canary form. Every command kind is idempotent, so double delivery is harmless. The staged-rollout history and the `_HEARTBEAT_WRITE` interaction are in A-93.

When started with `--remote-control`, the runner emits a per-machine heartbeat each loop tick (coordinator `POST /heartbeat`) and processes pending commands (coordinator `GET /commands`). Default-off; bit-identical when omitted. Helper module: `runner_remote_control.py` (sibling of `experiment_runner.py`).

**`_active_claim_on_evidence_dir()` guard** (contract `tests/contracts/test_active_claim_evidence_guard.py`, C9 covers the `docs/claims/` clause) stays for any runner-side path that calls `_push_telemetry_file`; the three autostash-revert incidents that motivated it are in A-93.

Six command kinds: `stop` (graceful drain), `force_stop` (SIGKILL current proc + exit), `pause` / `resume` (skip new experiments), `kick:<EXQ>` (move to head of queue), `release_claim:<EXQ>` (clear stuck `claimed_by`). `start` is intentionally not in this channel (a stopped runner cannot read its own command file) — use `/api/runner/v3/start` locally or SSH for remote.

When developing the runner: command processing happens at the **top of each pass** in the main `while True:` loop (before the experiment-picking `for item in items:` loop) so `pause` / `stop` / `kick` / `release_claim` take effect before the next claim attempt. Heartbeat write happens at the **bottom**, just before `time.sleep(args.loop_interval)`, with state in `{starting, idle, paused, draining}`.

Multi-machine dashboard: `/machines` in serve.py. POST `/api/machines/<host>/command {kind, args}` to enqueue commands. Trust model: GitHub push access = command-issue access.
