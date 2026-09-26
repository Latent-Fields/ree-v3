# ree-v3

Substrate-feature records used to live inline in this file (1,478,619 chars,
~370,000 tokens, injected whole into every session that touches `ree-v3/`).
Measured over the 25 sessions that loaded it, the median session referenced
**2 of 139 feature IDs**; 9 of 25 referenced none. They now live one per file
under [`docs/substrate/`](docs/substrate/), behind the index in
[`docs/substrate_index.md`](docs/substrate_index.md). Everything still inline below is a general `ree-v3` convention that
any session working in this repo needs.

**When to follow a pointer.** A substrate record tells you what was built, which
config flag turns it on, how the data flows, and which contracts pin it. The
default rule: **you do not need it to work on unrelated code, and you DO need it
before you (a) modify that feature's code, (b) write or queue an experiment that
exercises it or sets one of its flags, or (c) interpret a run that names it.**
Entries whose predicate is wider than that -- standing lints, standing defaults,
roll-up ledgers that bind a session not working on the feature at all -- say so
on their own line. Follow those on sight.

Sizes are approximate token costs of the linked file.

---
**Adding to this file:** most knowledge belongs elsewhere -- a new substrate record goes in `docs/substrate/` plus one line in [`docs/substrate_index.md`](docs/substrate_index.md); everything else routes per the umbrella's `.claude/rules/claude-md-placement.md`. Inline only what every ree-v3 session could break without reading further, justified in the commit message. Budget: umbrella `scripts/claude_md_budget.py`.

## Multi-Session Coordination

See `REE_Working/CLAUDE.md` for session startup protocol.
Check `REE_Working/WORKSPACE_STATE.md` before editing `experiment_queue.json`.

## ASCII-Only in Python Output

All `print()` statements and text reaching stdout/stderr must use ASCII only.
No `→ ← — × …` or other non-ASCII in printed output — these break on Windows cp1252 terminals.
Use `-> <- -- x ...` instead. Comments/docstrings may keep Unicode (read as UTF-8 by Python).

## Python
Use /opt/local/bin/python3 for all execution (has torch 2.10.0).
Use sys.executable for subprocesses within experiment runners.

## Branch Policy
No feature branches. All work to `main` directly.
Push: `git push origin HEAD:main`

## Governance
Run packs go to REE_assembly/evidence/experiments/.
run_id must end _v3. architecture_epoch must be "ree_hybrid_guardrails_v1".
After experiments complete: run sync_v3_results.py then build_experiment_indexes.py.

### Burned queue entries (silently-dropped experiments)

**Before re-queuing an id that "vanished", run** `/opt/local/bin/python3 /Users/dgolden/REE_Working/ree-v3/scripts/audit_burned_queue_entries.py` -- the only thing that can tell a burn (a pre-fix coordinator defect deleted a live entry that never ran) from a normal post-completion removal; `--require-lost` narrows to burns whose science was never recovered. Mechanism and history: [`docs/reference/claude_md_long_form.md`](docs/reference/claude_md_long_form.md).

## Regression Suite

Three layers in `tests/`: **preflight** (`tests/preflight/`, run automatically at runner startup; `--skip-preflight` / `REE_SKIP_PREFLIGHT=1` escape hatches), **contracts** (`tests/contracts/`, interface guarantees), and **changed** (`python3 scripts/run_regression_suite.py --changed <subsystem>` -- the contracts a `ree_core/` subsystem can break; `--list-subsystems`). Before committing a focused `ree_core/<subsystem>/` change run `--changed <subsystem>` (seconds); a cross-cutting change, `pytest tests/contracts -q`. **The FULL suite goes to a cloud worker** via `/Users/dgolden/REE_Working/scripts/remote_pytest.sh`, never the Mac; a worker-green suite is a gate except for a test asserting an exact committed action (umbrella CLAUDE.md "Running the test suite"). **Contracts test contracts, not thresholds** -- a magnitude or sign from an EXQ manifest belongs in an experiment script, never in the regression suite.

## Arm-Reuse Baselines (`experiments/_lib/baselines/`)
Canonical OFF/baseline modules live in `experiments/_lib/baselines/<lineage>.py` (today: `exq610_inv074_crystallization_baseline`, `exq643_modulatory_authority_baseline`), each exposing `build_off_arm(seed)` / `train_off_arm(...)` / `off_path_config_slice()`. **Save a baseline here by default** when a multi-arm family will re-run an *expensive*, *frozen-substrate* OFF arm on the *cloud* class: factor it into a module, then queue a low-priority cloud mint (`experiment_purpose="baseline"`, emit with `include_driver_script_in_hash=False`) so later iterations skip re-training it. The producer recipe + WHEN gate are in the `/queue-experiment` skill ("Saving a baseline for reuse"); design/validity in `REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md` §7b/§9. The module is auto-bound into `substrate_hash` via the `_lib/**` glob, so any edit to it correctly refuses a stale reuse (a false miss is free; a false hit corrupts science).

## Key Architecture Constraints
- E2 trains on motor-sensory error (z_self). NOT harm/goal error.
- E3 is the harm evaluator. harm_eval() belongs on E3Selector.
- ResidueField accumulates world_delta (z_world). NOT z_gamma.
- HippocampalModule navigates action-object space O. NOT raw z_world.
- All replay/simulation content must carry hypothesis_tag=True (MECH-094).
- Precision is E3-derived (E3 prediction error variance). NOT hardcoded.

## Q-020 Decision (2026-03-16)
ARC-007 STRICT: HippocampalModule generates value-flat proposals.
Terrain sensitivity = consequence of navigating residue-shaped z_world, not a separate hippocampal value computation.
MECH-073 reframed as consequence of ARC-013 applied to z_world.
MECH-074 (amygdala write interface) is valid but not a HippocampalModule prerequisite.

## V3 / V4 Scope Boundary

**V3** = the waking mechanisms plus ALL sleep substrates (sleep was rescoped from V4 on 2026-04-07/20). V3 has two gates: the **first-paper gate** (habit-system goal-directed behaviour: SD-012 + EXQ-182a oracle + goal-lift) and the **full-completion gate** (hippocampal multi-step trajectory planning validated, MECH-163), which V4 entry requires. **V4** = social systems (other agents' z_self / z_harm_a) and self-model integration (DR-10..DR-14). **Standing gates:** do NOT implement or experiment on Level-2 MECH-113 (allostatic anticipatory setpoint) until EXQ-075 PASS, EXQ-076 PASS and the Q-022 dissociation result are all in -- it needs ARC-031 z_self navigation, a V4 prerequisite, and earlier experiments are uninterpretable; run the Q-022 dissociation test (EVB-0069) before any MECH-118/119 Hopfield work. Full scope lists: [`docs/reference/claude_md_long_form.md`](docs/reference/claude_md_long_form.md).

## Experiment Queue Rules
- Every queue entry needs `estimated_minutes` (the runner's auto-calibration refines it). Estimate from episodes x steps using the per-machine rates in [`docs/reference/claude_md_long_form.md`](docs/reference/claude_md_long_form.md); add ~20% for stratified replay buffers or event classification.
- `machine_affinity` must be a name in `validate_queue.py`'s `VALID_AFFINITIES` -- `"any"` by default, `"DLAPTOP"` for the Mac; never `"macbook"` or a raw hostname (matching goes through `machine_identity.same_machine()`).
- Queue an experiment in the same session you write its script.

## Experiment IDs and Versioning

V3 ids run `V3-EXQ-001` onward. Letter vs new number, supersession (a `supersedes` field PLUS a `governance_flag.py` evidence_discrepancy flag -- `supersedes` alone reaches no one) and never re-using a run id: the umbrella CLAUDE.md "EXQ Versioning and Supersession Policy" is canonical. `validate_queue.py` runs at runner startup; run it by hand after any manual queue edit.

## Remote Control (--remote-control flag)

When started with `--remote-control`, the runner emits a per-machine heartbeat each loop tick (coordinator `POST /heartbeat`) and processes pending commands (coordinator `GET /commands`). **Default-off; behaviour is bit-identical when the flag is omitted.** Six command kinds: `stop` (graceful drain), `force_stop`, `pause`, `resume`, `kick:<EXQ>`, `release_claim:<EXQ>`. `start` is intentionally NOT in this channel -- a stopped runner cannot read its own commands; use `/api/runner/v3/start` locally or SSH for remote.

**Telemetry is coordinator-only** (Phase 3; the git render was retired 2026-09-06). **Do not re-enable the hub git writer or its liveness tick to get git-side progress files back** (A-93). For live progress read the `live-status` branch `FLEET_STATUS.md`, coordinator `/shadow/status`, or explorer `/machines`.

**The transport details, the `shadow.conf` gates, the command-channel API and the runner-loop placement rules are in [`docs/reference/remote-control.md`](docs/reference/remote-control.md)** (~1,500 tok): which gate suppresses what (`PHASE3_RUNNER_TELEMETRY_OFF_GIT` is worker-safe, `_HEARTBEAT_WRITE` is hub-only and restart-loops a worker), the `POST /commands/issue` / `GET /commands` / `POST /commands/ack` endpoints and their client helpers, the `_active_claim_on_evidence_dir()` guard, and **where in the `while True:` loop command processing and the heartbeat must sit** (top and bottom respectively) so `pause`/`stop`/`kick` take effect before the next claim attempt. Read it before changing the runner's heartbeat or command handling, adding a command kind, or touching a telemetry gate.

## Troubleshooting Runner

**Runner log location**: `REE_assembly/runner.log` (NOT `ree-v3/runner.log`) -- serve.py redirects runner stdout/stderr there. `ree-v3/runner.log` is written only when the runner is started manually with `nohup ... > runner.log`.

**The four canonical runner-stuck diagnoses now live in the `/diagnose-errors` skill** (`.claude/skills/diagnose-errors/SKILL.md`, "Canonical runner-stuck diagnoses", ~1,150 tok): "No new items" from a re-used queue_id already in `runner_status.json` (the 2026-03-23 six-experiment incident) and from a missing `title` field (2026-03-24); `fatal: bad object refs/remotes/origin/main 2`; and the 2026-05-09 manifest-leak in conflict-recovery that cost five runs their manifests. Invoke that skill when a runner is stuck or a run errored -- it is the mandatory path for ERROR-fix re-queues anyway (`REE_Working/CLAUDE.md`, "Experiment Scripts").

**The rule those incidents produced, kept here because it binds anyone touching the queue: never re-queue a failed or completed experiment under the same EXQ ID** -- the runner silently skips any `queue_id` already in `runner_status.json`. Append a letter instead.

---

## Substrate feature index

**The index now lives in [`docs/substrate_index.md`](docs/substrate_index.md)** (~160 feature ids, ~265 records under `docs/substrate/`) -- grep it for an id before touching a feature's code, queueing an experiment that exercises it or sets its flags, or interpreting a run that names it (the rule above). `/implement-substrate` adds new entries there, not here.

**Standing entries -- follow on sight, whatever you are working on:** the `SD-DECISIONS-IMPLEMENTED` roll-up ledger (grep it first when no single file owns an sd_id) and `SD-DECISIONS-VALIDATED` (includes the SUPERSEDED SD-003 counterfactual pipeline -- do not rebuild HarmBridge counterfactuals); `INV-091` (the standing default eval budget -- a new INV-091-family driver must import its constants); `Q-081` (shared recording harness + reach probe -- read before writing a cross-stream-organisation driver); `CEILING-ANCHOR-FLOOR-GUARD` (standing lint + runtime guard -- read before self-routing any diagnostic to a ceiling label); `DV-HEADROOM` (standing lint + precondition kind); `GATE-DV` (shared gate-level DV instrument).
