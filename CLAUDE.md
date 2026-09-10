# ree-v3

Substrate-feature records used to live inline in this file (1,478,619 chars,
~370,000 tokens, injected whole into every session that touches `ree-v3/`).
Measured over the 25 sessions that loaded it, the median session referenced
**2 of 139 feature IDs**; 9 of 25 referenced none. They now live one per file
under [`docs/substrate/`](docs/substrate/), behind the index at the bottom of
this file. Everything still inline below is a general `ree-v3` convention that
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

**Before re-queuing an id that "vanished", run the auditor** -- it is the only
thing that can tell a burn apart from a normal post-completion removal:

```
/opt/local/bin/python3 /Users/dgolden/REE_Working/ree-v3/scripts/audit_burned_queue_entries.py
```

A pre-fix coordinator defect (ree-v3 `d09127bb70`) let `reconcile_once(
upsert_only=True)` upsert a NEW git-queue item onto an already-TERMINAL DB
row; `phase3_queue_writer` materialises only non-terminal rows, so the next
snapshot DELETED the freshly-committed entry. The experiment never ran and
nothing errored. The ingress guard now refuses that upsert, so no NEW burns
occur -- but at the git layer a burned entry's deletion is indistinguishable
from a normal one, which is why sessions re-queued V3-EXQ-683 and V3-EXQ-686
three times each without ever learning they had been dropped.

The auditor walks the full `experiment_queue.json` history (~2 s) and reports
an operator-added entry that the next `phase3-queue: snapshot` deleted with
no manifest for its declared script while it was live. It is deliberately NOT
folded into `validate_queue.py` (the PreToolUse commit hook): a full-history
walk is far too slow to pay per commit, and it answers a question about
history rather than about the queue being committed. `--require-lost` narrows
to burns whose science was never recovered. Contract:
`tests/contracts/test_burned_queue_entry_detector.py`.

## Regression Suite

Three-layer test suite in `tests/`:

- **preflight** (`tests/preflight/`) — cheap wiring checks run before the runner
  starts machine work. Validates imports, queue integrity, and one-tick boot.
  The runner invokes preflight automatically at startup (see
  `experiment_runner.py`). Escape hatches: `--skip-preflight` flag or
  `REE_SKIP_PREFLIGHT=1` env var. If preflight fails, the runner exits non-zero
  and no experiment is started.

- **contracts** (`tests/contracts/`) — interface-level guarantees that should
  hold regardless of tuning. Includes: C1 agent boot, C2 feature-flag boot
  matrix, C3 seed determinism, C4 BG gating (MECH-090 / MECH-091), C5
  imagined/acted isolation (MECH-094), C6/C7/C8 SD-032 cluster wiring
  (dACC / AIC / PCC / pACC). Run: `pytest tests/contracts -q`.

- **changed** — subsystem-targeted contract tests. Resolves a `ree_core/`
  subdirectory name (or a path like `ree_core/residue/field.py`, or a
  substring like `bg`) to the contract tests that could plausibly break.
  `python3 scripts/run_regression_suite.py --changed residue` runs the
  MECH-094 / residue-write contracts only. See `--list-subsystems` for
  the map.

**When to run what:**
- Every experiment run: preflight (automatic via runner).
- Before committing a focused change to `ree_core/<subsystem>/`:
  `python3 scripts/run_regression_suite.py --changed <subsystem>` (~1-4s).
- Before committing a cross-cutting change: `pytest tests/contracts -q` or
  `python3 scripts/run_regression_suite.py --contracts` (~14s).
- **The FULL suite goes to a cloud worker, not the Mac:**
  `/Users/dgolden/REE_Working/scripts/remote_pytest.sh` (it ships your
  uncommitted edits). The targeted runs above are seconds long -- keep those
  local; it is only the full suite, run by several parallel sessions at once,
  that drives the laptop to load 25-30. A worker-green suite is a reasonable
  gate, EXCEPT for any test asserting an exact committed action -- assert
  upstream of the sampled action, never on it. Test counts, timings, routing /
  wake / lease mechanics and the `torch.multinomial` reasoning are canonical in
  REE_Working/CLAUDE.md "Running the test suite" and
  `REE_Working/docs/reference/ree-v3-test-suite-routing.md` -- read them there
  rather than restating any figure here, which is how this bullet went stale.
- Preflight + contracts together:
  `python3 scripts/run_regression_suite.py --preflight && \
   python3 scripts/run_regression_suite.py --contracts`.

**Contracts test contracts, not thresholds.** If a test starts asserting a
specific magnitude or sign from an EXQ manifest, that belongs in an experiment
script, not the regression suite. The regression suite is the thing that has to
keep working when experiments and claim state evolve.

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

## V3 / V4 Scope Boundary (updated 2026-04-02)

**Two-tier V3 completion:**
- V3 FIRST-PAPER GATE: habit-system goal-directed behavior (SD-012 + EXQ-182a oracle +
  goal-lift experiment). Demonstrates goal representations are real and influence behavior.
- V3 FULL COMPLETION GATE: hippocampal multi-step trajectory planning validated (MECH-163
  VTA/planned system). Required before V4 entry because V4 social extension ("sharing
  joys and sorrows") requires planning trajectories that affect another agent's z_harm_a
  and benefit_exposure over time -- structurally inaccessible to 1-step greedy.

**V3 scope (waking mechanisms):**
- Volatility interrupt / LC-NE analog (MECH-104): surprise-spike on running_variance
- BG hysteresis and outcome-valence modulation (MECH-106)
- Hippocampal→BG completion coupling (MECH-105, ARC-028) — IMPLEMENTED 2026-04-04
- Beta gate committed→uncommitted dynamics (MECH-090)
- Trajectory completion signal from HippocampalModule (ARC-028) — IMPLEMENTED 2026-04-04
- Dual goal-directed systems: habit (SNc/model-free) and hippocampally-planned
  (VTA/model-based). Both systems in V3; validation of the planned system is
  V3 full completion gate (MECH-163).

**V3 scope (serotonergic sleep substrate — pulled from V4 2026-04-07):**
- MECH-203: SerotoninModule tonic_5ht state variable + benefit-salience tagging (SR-1/SR-2).
  Without this, ALL SWS replay is harm-biased (depressive consolidation asymmetry is default).
- MECH-204: REM zero-point hook (SR-3). Captures precision_at_rem_entry for recalibration.
- Sleep convenience methods: enter_sws_mode(), enter_rem_mode(), exit_sleep_mode().
- Valence-weighted replay start selection in HippocampalModule.replay(drive_state=...).
- Master switch: tonic_5ht_enabled=False (default). All existing experiments unaffected.
- Location: ree_core/neuromodulation/serotonin.py

(All further sleep substrates — SD-017 sleep passes, SD-032 cingulate cluster,
MECH-261 mode-conditioned write gating — are likewise V3. See the unified
"V3 scope (full sleep mechanisms)" block below.)

**V3 scope (full sleep mechanisms — rescoped from V4 2026-04-20):**
All sleep-related substrates are V3. V4 is reserved for social extensions
(see below). The following items are therefore V3 in-scope, not deferred:
- Full SWR consolidation pipeline (MECH-121 complete implementation)
- Slow-wave sleep prediction error baseline reset
- Sleep-dependent recalibration of commit thresholds (full SR-3/SR-4)
- Theta-gamma coupling during offline replay for memory formation
- Lansink et al. (2009) hippocampus-leads-striatum replay — V3 evidence
- Phase boundary triggers (SR-4: sws_consolidation_complete -> REM transition)
- MECH-261 predicate enrichment on the SD-032a registry (carrier-rhythm
  *function* -> multi-factor admission conjunction; see
  REE_assembly/evidence/literature/targeted_review_mech261_mode_gating/
  synthesis.md for the biology-to-REE mapping)
- Per-mode write-gate weight refinement as new mode-gating literature lands

**V4 scope (social systems — rescoped 2026-04-20):**
V4 is now reserved for social systems ("sharing joys and sorrows"): representing
other agents, their z_self / z_harm_a, and trajectories that affect another
agent's state over time. This remains structurally inaccessible to 1-step greedy
planning, which is why V3 full completion gate (MECH-163 hippocampal multi-step
trajectory planning) is a prerequisite for V4 entry.

**V4 scope (self-model integration — INV-064/MECH-214/MECH-215 audit, 2026-04-07):**

Wiring audit against the maturational sequence claims revealed five architectural gaps.
None are V3 errors — V3's grid-world spatial goals and 4-action motor model are correctly
scoped. All become requirements when the architecture handles richer agents, environments,
or goal types:

- DR-10: z_self in E3 trajectory scoring. Currently score_trajectory() evaluates entirely
  in z_world space. The agent's interoceptive state (energy, fatigue, pain) does not
  influence which trajectory is selected. V4 needs z_self-weighted trajectory costs so that
  bodily state modulates viability (the same path is worse when exhausted vs. fresh).
  Implements: MECH-215 (self-model prerequisite for agentive prediction).

- DR-11: z_self-domain goal representation. Currently z_goal lives purely in z_world space
  (GoalState seeds from z_world_current). Self-state goals ("I want my energy restored",
  "I want to not be in pain") cannot be represented. V4 needs a parallel z_goal_self
  attractor, or GoalState extended to operate on [z_self, z_world] jointly. Without this,
  homeostatic and hedonic goals are structurally inaccessible to the planning system.
  Implements: MECH-214 (goal-referent E1-representability) for the z_self domain.

- DR-12: E2 prediction error -> E3 confidence modulation. Currently E3 trusts E2's
  rollout unconditionally. When E2's capacity model is degraded (producing inflated or
  deflated z_self predictions), E3 inherits the error with no "this rollout might be
  unreliable" signal. V4 needs E2 PE magnitude to modulate E3's confidence in each
  trajectory's self-transition feasibility, so that trajectories generated from
  unreliable E2 predictions are appropriately discounted.
  Implements: MECH-215 pessimistic/optimistic failure modes.

- DR-13: z_self temporal depth. Currently z_self = body_obs -> MLP -> EMA smooth.
  Single hidden layer, no recurrence, no body-state memory. E1's LSTM integrates z_self
  over time but is read-only on z_self (doesn't enrich the representation). V4 needs
  either: (a) recurrent z_self encoder, or (b) E1 feedback into z_self enrichment,
  or (c) dedicated E2-as-self-model that provides capacity trends not just next-step
  predictions. Without temporal self-model, MECH-215 capacity estimates are snapshots
  not trajectories.

- DR-14: Environment must dissociate proxy from hedonic content. CausalGridWorldV2
  conflates location with reward — the z_world at a resource IS the benefit. This
  means the MECH-214 addiction failure mode (wanting system fires on z_goal objects
  that E1 can't ground in genuine hedonic schema) cannot be surfaced. V4 needs an
  environment where goal location and hedonic satisfaction can dissociate, so that
  z_goal tracking a proxy without hedonic grounding produces observable behavioral
  pathology (pursuit without satisfaction, the addiction signature).

**V4 scope (self-navigation — not V3, gated by MECH-113/114 results):**
- ARC-031: Hippocampal z_self trajectory navigation (planning deliberation sequences).
  GATE: Do NOT implement or experiment on Level 2 MECH-113 (allostatic anticipatory
  setpoint) until ALL of the following are met:
  (1) EXQ-075 PASS (Level 1 D_eff reactive homeostasis confirmed)
  (2) EXQ-076 PASS (MECH-114 D_eff commit gating confirmed)
  (3) Q-022 dissociation result available (D_eff vs Hopfield stability)
  Level 2 requires HippocampalModule to navigate z_self space — ARC-031 is a V4
  prerequisite. Premature Level 2 experiments will produce uninterpretable results
  because the anticipatory setpoint mechanism cannot function without z_self navigation.
- MECH-118/119 Hopfield familiarity signal and coherent-unfamiliar pathology detection.
  GATE: Q-022 dissociation test (EVB-0069) must be run first. If D_eff and Hopfield
  stability always co-vary (no dissociation), MECH-118/119 collapse into MECH-113
  and no separate implementation is needed.

## Experiment Queue Rules
- Every queue entry **must** have `estimated_minutes` set (never omit it).
- Estimate from: total episodes × steps_per_episode, calibrated against known runtimes:
  - **Mac (`DLAPTOP-4.local`)** — CPU, CausalGridWorldV2, typical REE agent:
    - ~0.10 min/ep at 200 steps/ep
    - ~0.15 min/ep at 300 steps/ep
  - **Daniel-PC** — CPU preferred (GPU 3x slower at current model scale, batch=1):
    - ~0.50 min/ep at 200 steps/ep  (~5x slower than Mac)
    - ~0.72 min/ep at 300 steps/ep
    - Calibrated from onboarding smoke runs 2026-03-22: 7.0 steps/sec CPU, 2.1 steps/sec GPU
    - GPU NEVER wins at current model scale (world_dim=32): EXQ-070 tested batch 1-512,
      CPU always faster (200k vs 133k samples/s at batch=512). RTX 2060 Super overhead
      dominates for tiny networks. GPU becomes useful ONLY when world_dim >= 128 or
      networks are substantially deeper. Design experiments with larger networks to
      exploit the GPU when the architecture requires it (SD-004, SD-010).
  - **ree-cloud-1** — Hetzner CX22, CPU-only (no GPU), 2 shared vCPU:
    - ~0.23 min/ep at 200 steps/ep  (~2.3x slower than Mac)
    - ~0.35 min/ep at 300 steps/ep
    - Calibrated from onboarding smoke 2026-04-09: 14.2 steps/sec CPU, 1571.9 env steps/sec
    - Suitable for env-heavy and standard experiments. Not for GPU-dependent runs.
  - **ree-cloud-2** — Hetzner CX22, CPU-only (second, nominally identical to cloud-1):
    - Throughput pending -- onboarding smoke V3-ONBOARD-smoke-ree-cloud-2 queued.
    - Estimate as for cloud-1 until its smoke calibrates. Shared-vCPU neighbour noise
      may produce small per-instance divergence; check the smoke result before tight
      runtime estimates.
  - **EWIN-PC** — AMD Ryzen 7 8700F + RTX 5070 12GB (Eoin Golden's machine):
    - Throughput not yet benchmarked (original smoke errored 2026-04-06, -b pending)
    - Use `"EWIN-PC"` affinity string. GPU likely fast at larger world_dim.
  - Add ~20% overhead for scripts with stratified replay buffers or event classification
- Set `machine_affinity` to match compute profile: `"DLAPTOP"` (macbook, online stepping), `"Daniel-PC"` (replay/batch heavy or long overnight runs), `"ree-cloud-1"` / `"ree-cloud-2"` (CPU-only Hetzner CX22, standard/env-heavy), `"EWIN-PC"` (GPU-capable, Eoin's machine), `"any"` (indifferent -- any cloud worker that's already awake will typically claim first)
  - **IMPORTANT:** The runner matches affinity through `machine_identity.same_machine()` (see `machine_identity.py`'s module docstring), NOT raw `socket.gethostname()` equality — this closed a real bug where macOS LocalHostName suffix drift (`DLAPTOP-4.local` <-> `DLAPTOP-5.local`) silently split the Mac's identity in two. `"DLAPTOP"` is the canonical affinity string to use in new queue entries; `"DLAPTOP-4.local"`/`"DLAPTOP-5.local"` still match (they alias forward to `DLAPTOP`), but do NOT use `"macbook"` or any other unlisted string — only names in `validate_queue.py`'s `VALID_AFFINITIES` resolve to a real machine.
- Always queue experiments immediately after writing the script.
- Always include `estimated_minutes` — the runner's auto-calibration refines it over time.

## Experiment IDs and Versioning

V3 experiments: V3-EXQ-001 onward.

**Labeling rule (see also REE_Working/CLAUDE.md "EXQ Versioning and Supersession Policy"):**
- Bug fix / minor implementation tweak to same hypothesis: append next letter (EXQ-047a, 047b, ... 047j).
- New hypothesis / major redesign: new number (EXQ-048).
- NEVER re-use an ID that was previously run. The runner silently skips any queue_id already in `runner_status.json` completed list.

**Supersession:** when a lettered iteration corrects a bug that invalidated the predecessor's evidence, add `"supersedes": "V3-EXQ-047i"` to the new queue entry. After the run completes, set `evidence_direction: "superseded"` on the old manifest and rebuild the index (governance pipeline). This prevents buggy experiments from continuing to weight claim confidence scores.

**Queue validation:** `validate_queue.py` is called automatically at runner startup. Run it manually after any queue edit: `/opt/local/bin/python3 validate_queue.py`

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

255 per-feature records, 155 feature IDs, in `docs/substrate/`. Read the
"when to follow a pointer" rule at the top of this file before deciding to skip one.
- **SD-011** — 2 records, ~780 tok total.
    - [Second Source: Harm History Input (2026-04-08)](docs/substrate/SD-011-second-source-harm-history-input.md) *(~455 tok)*
    - [SD-012 E3 Integration (2026-04-05)](docs/substrate/SD-011-sd-012-e3-integration.md) *(~325 tok)*
- **[SD-013](docs/substrate/SD-013-mech-090-sd-015-sd-019-sd-020-sd-021.md)** — MECH-090, SD-015, SD-019, SD-020, SD-021: Harm Stream + Gate Implementations (2026-04-10) *(~2,513 tok)*
- **SD-016** — 4 records, ~5,839 tok total.
    - [Frontal Cue-Indexed Integration (2026-04-16)](docs/substrate/SD-016-frontal-cue-indexed-integration.md) *(~2,583 tok)*
    - [Path 1: ContextMemory Diversification Loss (2026-04-25)](docs/substrate/SD-016-path-1-contextmemory-diversification.md) *(~394 tok)*
    - [Path 3: Feedforward cue->slot tagger (2026-06-05)](docs/substrate/SD-016-path-3-feedforward-cue-to-slot-tagger.md) *(~1,324 tok)*
    - [H3: Hard/competitive selection operator (2026-08-09)](docs/substrate/SD-016-h3-hard-competitive-selection-operator.md) *(~1,536 tok)*
- **[SD-017](docs/substrate/SD-017-minimal-sleep-phase-infrastructure-sws.md)** — Minimal Sleep-Phase Infrastructure -- SWS/REM Passes (2026-04-09) *(~670 tok)*
- **SD-018** — 2 records, ~1,542 tok total.
    - [Resource Proximity Supervision (2026-04-07)](docs/substrate/SD-018-resource-proximity-supervision.md) *(~258 tok)*
    - [AMEND: encoder.resource_field_supervision (directional resource-field head) -- IMPLEMENTED (2026-09-02)](docs/substrate/SD-018-amend-encoder-resource-field.md) *(~1,284 tok)*
- **[SD-019a](docs/substrate/SD-019a-harm-unpleasantness-channel.md)** — harm_unpleasantness_channel (2026-05-04) *(~873 tok)*
- **[SD-022](docs/substrate/SD-022-scheduled-injection-extension-mech-302.md)** — scheduled-injection extension (MECH-302 unblock, 2026-05-30) *(~1,982 tok)*
- **[SD-023](docs/substrate/SD-023-environmental-gradient-texture.md)** — Environmental Gradient Texture (2026-04-09) *(~425 tok)*
- **SD-024** — 2 records, ~2,186 tok total.
    - [hippocampal_module.da_modulated_rbf_density -- IMPLEMENTED (2026-07-16)](docs/substrate/SD-024-hippocampal-module-da-modulated-rbf-density.md) *(~725 tok)*
    - [LIVE-PATH PRODUCER: residue.benefit_terrain_live_producer -- IMPLEMENTED (2026-07-20)](docs/substrate/SD-024-live-path-producer-residue-benefit-terrain-live.md) *(~1,461 tok)*
- **[SD-025](docs/substrate/SD-025-hippocampal-module-curiosity-drive.md)** — hippocampal_module.curiosity_drive -- IMPLEMENTED (2026-07-16) *(~820 tok)*
- **[SD-029](docs/substrate/SD-029-balanced-hazard-event-curriculum.md)** — Balanced Hazard-Event Curriculum (2026-04-21) *(~813 tok)*
- **[SD-031](docs/substrate/SD-031-e2-world-single-pass-comparator-z-world.md)** — E2_world Single-Pass Comparator (z_world agency) (2026-06-06) *(~1,214 tok)*
- **[SD-032a](docs/substrate/SD-032a-mech-259-mech-261-salience-network.md)** — MECH-259 / MECH-261: Salience-Network Coordinator (2026-04-19) *(~832 tok)*
- **[SD-032b](docs/substrate/SD-032b-mech-258-mech-260-arc-058-dacc-analog.md)** — MECH-258 / MECH-260 / ARC-058: dACC-analog Adaptive Control (2026-04-19) *(~1,394 tok)*
- **[SD-032c](docs/substrate/SD-032c-cingulate-aic-analog-salience-urgency.md)** — cingulate.aic_analog_salience_urgency -- IMPLEMENTED (2026-04-19) *(~1,222 tok)*
- **SD-032d** — 2 records, ~2,243 tok total.
    - [cingulate.pcc_analog_attention_partition -- IMPLEMENTED (2026-04-19)](docs/substrate/SD-032d-cingulate-pcc-analog-attention-partition.md) *(~960 tok)*
    - [AMENDMENT: cingulate.mu_kappa_mode_prior_overlays (MECH-048) -- IMPLEMENTED (2026-07-21)](docs/substrate/SD-032d-amendment-cingulate-mu-kappa-mode-prior.md) *(~1,283 tok)*
- **[SD-032e](docs/substrate/SD-032e-cingulate-pacc-autonomic-coupling.md)** — cingulate.pacc_autonomic_coupling -- IMPLEMENTED (2026-04-19) *(~1,395 tok)*
- **[SD-033a](docs/substrate/SD-033a-lateral-pfc-analog-mech-261-primary.md)** — Lateral-PFC-analog / MECH-261 Primary Consumer (2026-04-20) *(~979 tok)*
- **SD-033b** — 4 records, ~5,896 tok total.
    - [OFC-analog / MECH-261 Second Consumer (2026-04-26)](docs/substrate/SD-033b-ofc-analog-mech-261-second-consumer.md) *(~1,292 tok)*
    - [GAP-8: trainable OFC state_bias_head (mirror of SD-033a GAP-D) (2026-06-09)](docs/substrate/SD-033b-gap-8-trainable-ofc-state-bias-head.md) *(~1,189 tok)*
    - [GAP-8 DECOUPLE: separate OFC devaluation_bias_head (clamp-starved devalued range; failure_autopsy V3-EXQ-485l) (2026-06-22)](docs/substrate/SD-033b-gap-8-decouple-separate-ofc-devaluation.md) *(~1,749 tok)*
    - [GAP-8-affordability: reusable trained-OFC-head cache (chip-20260730-ofc-deval-affordable) (2026-07-31)](docs/substrate/SD-033b-gap-8-affordability-reusable-trained.md) *(~1,665 tok)*
- **[SD-033e](docs/substrate/SD-033e-v3-narrow-frontopolar-analog-de-commit.md)** — (V3-narrow): Frontopolar-analog de-commit lever / MECH-264 (2026-07-09) *(~1,406 tok)*
- **SD-034** — 5 records, ~8,221 tok total.
    - [Governance Closure Operator (2026-04-20)](docs/substrate/SD-034-governance-closure-operator.md) *(~1,224 tok)*
    - [AMEND: commitment-closure-control-plane (env-completion hook + de-commit hold) (2026-06-12)](docs/substrate/SD-034-amend-commitment-closure-control-plane.md) *(~1,591 tok)*
    - [AMEND: commitment-closure-control-plane BETA-ENGAGEMENT (couple closure->beta elevation) (2026-06-17)](docs/substrate/SD-034-amend-commitment-closure-control-plane-2.md) *(~1,598 tok)*
    - [AMEND: commitment-closure-control-plane DE-COMMIT-AUTHORITY MAGNITUDE (committed-run-scaled Leg-B refractory) (2026-06-19)](docs/substrate/SD-034-amend-commitment-closure-control-plane-3.md) *(~2,064 tok)*
    - [AMEND: commitment-closure-control-plane REFRACTORY-INDEPENDENT coupling certifier (decouple the de-commit lever from its non-vacuity metric) (2026-06-19)](docs/substrate/SD-034-amend-commitment-closure-control-plane-4.md) *(~1,743 tok)*
- **[SD-035](docs/substrate/SD-035-amygdala-analogue-bla-cea-peer-modules.md)** — Amygdala Analogue -- BLA + CeA Peer Modules (2026-04-21) *(~2,867 tok)*
- **[SD-036](docs/substrate/SD-036-mech-279-gabaergic-cross-stream-decay.md)** — MECH-279: GABAergic Cross-Stream Decay + PAG Freeze-Gate (2026-04-22) *(~3,516 tok)*
- **SD-037** — 2 records, ~3,587 tok total.
    - [Broadcast Override Regulator (orexin-analog) (2026-04-25)](docs/substrate/SD-037-broadcast-override-regulator-orexin.md) *(~1,937 tok)*
    - [consumer-cascade (MECH-281 motor-coupling axis amend, 2026-05-30)](docs/substrate/SD-037-consumer-cascade-mech-281-motor.md) *(~1,650 tok)*
- **SD-039** — 2 records, ~3,102 tok total.
    - [Dual-Trace Anchor Goal-Snapshot Payload -- Substrate Foundation (2026-04-26)](docs/substrate/SD-039-dual-trace-anchor-goal-snapshot-payload.md) *(~1,791 tok)*
    - [Module-Level Write-Site Population Layer (2026-04-27)](docs/substrate/SD-039-module-level-write-site-population-layer.md) *(~1,310 tok)*
- **[SD-047](docs/substrate/SD-047-multi-source-environmental-dynamics.md)** — Multi-Source Environmental Dynamics (2026-05-03) *(~2,559 tok)*
- **[SD-048](docs/substrate/SD-048-interoceptive-noise-dynamics.md)** — Interoceptive Noise Dynamics (2026-05-03) *(~1,974 tok)*
- **SD-049** — 5 records, ~9,225 tok total.
    - [Multi-Resource Heterogeneity (Phase 1 substrate, 2026-05-03)](docs/substrate/SD-049-multi-resource-heterogeneity-phase-1.md) *(~2,960 tok)*
    - [Phase 2: Hybrid Identity-Aware z_resource Encoder (2026-05-04)](docs/substrate/SD-049-phase-2-hybrid-identity-aware-z.md) *(~2,148 tok)*
    - [drive-coupling amend: kappa-scale + standing differential depletion (V3-EXQ-514r, MECH-436) (2026-06-17)](docs/substrate/SD-049-drive-coupling-amend-kappa-scale.md) *(~1,498 tok)*
    - [drive-coupling amend: BOUNDED kappa raise + deeper standing spread (V3-EXQ-514s, MECH-436) (2026-06-19)](docs/substrate/SD-049-drive-coupling-amend-bounded-kappa.md) *(~1,525 tok)*
    - [density-preserving spawn: per-type resource density held constant across arms (V3-EXQ-693a) (2026-07-20)](docs/substrate/SD-049-density-preserving-spawn-per-type.md) *(~1,093 tok)*
- **[SD-050](docs/substrate/SD-050-suffering-derivative-comparator.md)** — Suffering-Derivative Comparator (2026-05-04) *(~511 tok)*
- **[SD-051](docs/substrate/SD-051-conditioned-safety-store.md)** — Conditioned Safety Store (2026-05-04) *(~621 tok)*
- **[SD-052](docs/substrate/SD-052-contextual-passive-safety-terrain.md)** — Contextual Passive Safety Terrain (2026-05-04) *(~588 tok)*
- **SD-054** — 2 records, ~2,549 tok total.
    - [Reef Enrichment Substrate (2026-05-04)](docs/substrate/SD-054-reef-enrichment-substrate.md) *(~906 tok)*
    - [bipartite layout extension (2026-05-11)](docs/substrate/SD-054-bipartite-layout-extension.md) *(~1,642 tok)*
- **[SD-055](docs/substrate/SD-055-differentiable-cem-selection.md)** — Differentiable CEM Selection Approximation (2026-05-15) *(~433 tok)*
- **SD-056** — 2 records, ~4,054 tok total.
    - [E2 action-conditional divergence preservation (contrastive next-state) (2026-05-29)](docs/substrate/SD-056-e2-action-conditional-divergence.md) *(~1,988 tok)*
    - [multi-step rollout stability amend (2026-05-31)](docs/substrate/SD-056-multi-step-rollout-stability-amend.md) *(~2,066 tok)*
- **SD-057** — 3 records, ~4,008 tok total.
    - [Object-bound incentive-salience layer (GAP-7 L2-L3-L4) (2026-06-04)](docs/substrate/SD-057-object-bound-incentive-salience-layer.md) *(~1,421 tok)*
    - [phase-2: L6 cue-recall + L7 dACC object-discriminative readout (2026-06-04)](docs/substrate/SD-057-phase-2-l6-cue-recall-l7-dacc-object.md) *(~1,268 tok)*
    - [L7 AMEND: dacc_goal_readout calibration fix (arc005_dacc_adapter_goal_proximity_training, IGW-20260801-199) (2026-08-01)](docs/substrate/SD-057-l7-amend-dacc-goal-readout-calibration.md) *(~1,318 tok)*
- **[SD-058](docs/substrate/SD-058-mech-357-instrumental-avoidance.md)** — MECH-357: instrumental-avoidance acquisition (ilPFC-analog freeze-suppression + avoidance action pathway) (2026-06-07) *(~2,224 tok)*
- **SD-059** — 2 records, ~3,613 tok total.
    - [MECH-358: relief/safety escape-affordance bridge (directed escape for the MECH-357 gate) (2026-06-08)](docs/substrate/SD-059-mech-358-relief-safety-escape.md) *(~1,861 tok)*
    - [MECH-358 AMEND: safety-half trained threat-absence wiring (V3-EXQ-603i secondary gap, 2026-06-09)](docs/substrate/SD-059-mech-358-amend-safety-half-trained.md) *(~1,752 tok)*
- **[SD-061](docs/substrate/SD-061-difficulty-gated-proposal-entropy.md)** — difficulty-gated proposal-entropy regulator (stuck-state detector + transient CEM proposal-widening; MECH-343 blocker part 2 / Q-056) (2026-06-19) *(~1,494 tok)*
- **[SD-063](docs/substrate/SD-063-e2-conditional-predictive-uncertainty.md)** — E2 Conditional Predictive-Uncertainty Head (z_world quantile) (2026-07-05) *(~1,161 tok)*
- **[SD-065](docs/substrate/SD-065-environment-conditioned-safety-cue-channel.md)** — environment.conditioned_safety_cue_channel -- IMPLEMENTED (2026-07-14) *(~799 tok)*
- **[SD-066](docs/substrate/SD-066-safety-prediction-common-mode-invariant.md)** — safety_prediction.common_mode_invariant_conditioned_safety_readout -- IMPLEMENTED (2026-07-15) *(~612 tok)*
- **[SD-068](docs/substrate/SD-068-sleep-consolidation-pipeline-lesion-harness.md)** — sleep.consolidation_pipeline_lesion_harness -- IMPLEMENTED (2026-07-17) *(~1,140 tok)*
- **[SD-069](docs/substrate/SD-069-control-plane-phasic-surprise-burst.md)** — control_plane.phasic_surprise_burst -- IMPLEMENTED (2026-07-17) *(~1,323 tok)*
- **SD-070** — 2 records, ~2,565 tok total.
    - [latent.zworld_p0_anticollapse_recipe -- IMPLEMENTED (2026-07-18)](docs/substrate/SD-070-latent-zworld-p0-anticollapse-recipe.md) *(~1,400 tok)*
    - [ADOPTION in the _train_all_on_agent driver family -- IMPLEMENTED (2026-07-20)](docs/substrate/SD-070-adoption-in-the-train-all-on-agent-driver-family.md) *(~1,165 tok)*
- **[SD-074](docs/substrate/SD-074-probe-trained-enough-agent-warmup.md)** — probe.trained_enough_agent_warmup -- IMPLEMENTED (2026-07-18) *(~2,550 tok)*
- **[SD-075](docs/substrate/SD-075-phasic-ema-episode-continuity.md)** — phasic.ema_episode_continuity -- IMPLEMENTED (2026-07-19) *(~1,116 tok)*
- **[SD-076](docs/substrate/SD-076-precision-waking-confidence-inflation.md)** — precision.waking_confidence_inflation -- IMPLEMENTED (2026-07-20) *(~1,396 tok)*
- **[SD-077](docs/substrate/SD-077-goal-common-mode-invariant-super-ordinal-cue-key.md)** — goal.common_mode_invariant_super_ordinal_cue_key -- IMPLEMENTED (2026-07-21) *(~1,097 tok)*
- **[SD-078](docs/substrate/SD-078-policy-common-mode-invariant-candidate-rule.md)** — policy.common_mode_invariant_candidate_rule_field_context_key -- IMPLEMENTED (2026-07-22) *(~550 tok)*
- **[SD-079](docs/substrate/SD-079-hippocampal-common-mode-invariant-goal-anchor.md)** — hippocampal.common_mode_invariant_goal_anchor_match -- IMPLEMENTED (2026-07-22) *(~619 tok)*
- **[SD-081](docs/substrate/SD-081-e3-dualsystem-uncertainty-arbitration.md)** — e3.dualsystem_uncertainty_arbitration -- IMPLEMENTED (2026-07-22) *(~1,073 tok)*
- **SD-082** — 3 records, ~4,106 tok total.
    - [pfc.lateral_pfc.rule_selection_action_consumer -- IMPLEMENTED (2026-07-26)](docs/substrate/SD-082-pfc-lateral-pfc-rule-selection-action-consumer.md) *(~715 tok)*
    - [AMEND: head-internals instrumentation (dead-ReLU / magnitude-ratio diagnostics for the still-zero V3-EXQ-822a propagation) (2026-07-27)](docs/substrate/SD-082-amend-head-internals-instrumentation.md) *(~1,216 tok)*
    - [AMEND: per-candidate summary was a shared constant, not per-candidate (the CORRUPTING defect V3-EXQ-822c confirmed) (2026-08-29)](docs/substrate/SD-082-amend-per-candidate-summary-was-a.md) *(~2,175 tok)*
- **[SD-091](docs/substrate/SD-091-mech-481-coalition-topology-control.md)** — MECH-481: Coalition/Topology Control Substrate -- steps 1-6 of 7 IMPLEMENTED (2026-08-03) *(~1,893 tok)*
- **[SD-092](docs/substrate/SD-092-cross-level-subgoal-credit-implemented.md)** — Cross-Level Subgoal Credit -- IMPLEMENTED (primitive + agent-loop call site, 2026-08-02) *(~1,569 tok)*
- **[SD-093](docs/substrate/SD-093-progress-velocity-effort-persistence.md)** — Progress-Velocity Effort/Persistence Modulation -- IMPLEMENTED (2026-08-02) *(~1,723 tok)*
- **[SD-099](docs/substrate/SD-099-mech-489-defensive-orienting-response.md)** — MECH-489: Defensive-Orienting Response -- IMPLEMENTED (2026-08-09) *(~1,557 tok)*
- **[SD-100](docs/substrate/SD-100-arc-032-mech-089-phase-aware.md)** — ARC-032 / MECH-089: Phase-Aware ThetaBuffer Summary -- IMPLEMENTED (2026-08-10) *(~1,544 tok)*
- **[SD-102](docs/substrate/SD-102-mech-482-policy-epistemic-deficit.md)** — MECH-482: policy.epistemic_deficit_accumulator -- IMPLEMENTED (2026-08-29) *(~1,713 tok)*
- **[SD-104](docs/substrate/SD-104-sd-105-phasic-burst-refractory-duty.md)** — SD-105: phasic burst refractory duty bound + selection-entropy headroom floor (the two coupled regulator defects blocking MECH-063 (ii)) -- IMPLEMENTED (2026-09-04) *(~2,254 tok)*
- **SD-DECISIONS-IMPLEMENTED** — 6 records, ~11,739 tok total. **roll-up ledger of many small SD entries -- grep here first when no single file owns an sd_id.**
    - [SD Design Decisions Implemented](docs/substrate/SD-DECISIONS-IMPLEMENTED-sd-design-decisions-implemented.md) *(~3,435 tok)*
    - [SD Design Decisions Implemented (V3) — continued](docs/substrate/SD-DECISIONS-IMPLEMENTED-sd-design-decisions-implemented-v3.md) *(~220 tok)*
    - [SD Design Decisions Implemented (V3) — continued](docs/substrate/SD-DECISIONS-IMPLEMENTED-sd-design-decisions-implemented-v3-2.md) *(~1,219 tok)*
    - [SD Design Decisions Implemented (V3) — continued](docs/substrate/SD-DECISIONS-IMPLEMENTED-sd-design-decisions-implemented-v3-3.md) *(~92 tok)*
    - [SD Design Decisions Implemented (V3) — continued](docs/substrate/SD-DECISIONS-IMPLEMENTED-sd-design-decisions-implemented-v3-4.md) *(~4,282 tok)*
    - [SD Design Decisions Implemented (V3) — continued](docs/substrate/SD-DECISIONS-IMPLEMENTED-sd-design-decisions-implemented-v3-5.md) *(~2,489 tok)*
- **[SD-DECISIONS-VALIDATED](docs/substrate/SD-DECISIONS-VALIDATED-sd-design-decisions-validated-v3-2026.md)** — SD Design Decisions Validated (V3) — 2026-03-18 *(~423 tok)* **roll-up ledger, incl. the SUPERSEDED SD-003 counterfactual pipeline (do not rebuild HarmBridge counterfactuals).**
- **[SD-E3-CHANNEL-COMMENSURABILITY](docs/substrate/SD-E3-CHANNEL-COMMENSURABILITY-f-dominance-conversion-ceiling-rung-3-sd-e3.md)** — f_dominance_conversion_ceiling rung 3 / SD-E3-CHANNEL-COMMENSURABILITY (MECH-439) -- E3 channel-commensurability operator -- IMPLEMENTED (2026-09-07) *(~1,618 tok)*
- **[SD-MECH267-CEM-SELECTION-FIX](docs/substrate/SD-MECH267-CEM-SELECTION-FIX-mode-content-wash-out-fix-h2-value-term.md)** — Mode-Content Wash-Out Fix (H2 value term + H3 persistent breadth) -- IMPLEMENTED (2026-08-14) *(~1,588 tok)*
- **[SD-MECH267-HORIZON-DEPTH](docs/substrate/SD-MECH267-HORIZON-DEPTH-mode-conditioned-horizon-depth.md)** — Mode-Conditioned Horizon-Depth Modulation -- IMPLEMENTED (2026-08-02) *(~1,174 tok)*
- **[SD-MECH303-THRESHOLD-SOURCING](docs/substrate/SD-MECH303-THRESHOLD-SOURCING-mech303-contextual-safety-gate-dedicated.md)** — mech303.contextual_safety_gate.dedicated_proximity_signal -- IMPLEMENTED (2026-08-14) *(~625 tok)*
- **[SD-MECH457-DISTRIBUTIONAL-CRITIC](docs/substrate/SD-MECH457-DISTRIBUTIONAL-CRITIC-action-learning-distributional-value.md)** — action_learning.distributional_value -- IMPLEMENTED (2026-07-18) *(~1,087 tok)*
- **[SD-MECH457-POLICY-KL-ANCHOR](docs/substrate/SD-MECH457-POLICY-KL-ANCHOR-mech457-policy-trust-region-anchor.md)** — mech457 policy trust-region anchor -- IMPLEMENTED (2026-07-19) *(~1,399 tok)*
- **[SD-MEL-CONSUMER](docs/substrate/SD-MEL-CONSUMER-sleep-adaptive-mel-sleep-cadence.md)** — sleep.adaptive_mel_sleep_cadence -- IMPLEMENTED (2026-07-07) *(~862 tok)*
- **[SD-MEL-PRODUCER](docs/substrate/SD-MEL-PRODUCER-environment-non-converging-world-rule-shift.md)** — environment.non_converging_world_rule_shift -- IMPLEMENTED (2026-07-21) *(~1,402 tok)*
- **[SD-ORIENTING-DECISION-SCALE](docs/substrate/SD-ORIENTING-DECISION-SCALE-mech-489-defensive-orienting-decision.md)** — MECH-489: Defensive-Orienting Decision Normalization -- IMPLEMENTED (2026-08-10) *(~1,182 tok)*
- **[SD-RESIDUE-VALENCE-BOUND](docs/substrate/SD-RESIDUE-VALENCE-BOUND-residue-field-rbflayer-update-valence.md)** — residue.field.RBFLayer.update_valence.accumulator_bound -- IMPLEMENTED (2026-08-11) *(~1,764 tok)*
- **[SD-SLEEP-ENTRY-PRESSURE](docs/substrate/SD-SLEEP-ENTRY-PRESSURE-sleep-entry-pressure-time-integrating-trigger.md)** — (sleep_substrate:GAP-9 follow-up, V3-EXQ-933 fix): sleep.entry_pressure_time_integrating_trigger -- IMPLEMENTED (2026-08-26) *(~1,162 tok)*
- **[SD-WAYPOINT-FIELD](docs/substrate/SD-WAYPOINT-FIELD-environment-waypoint-proximity-field.md)** — environment.waypoint_proximity_field -- IMPLEMENTED (2026-09-04) *(~1,588 tok)*
- **SD-e1-rollout-consistency-training** — 3 records, ~4,351 tok total.
    - [ITEM 1: e1.transition.action_conditioning -- IMPLEMENTED (2026-08-29)](docs/substrate/SD-e1-rollout-consistency-training-item-1-e1-transition-action-conditioning.md) *(~1,419 tok)*
    - [ABSOLUTE-VS-RESIDUAL BRANCH: e1.rollout.output_proj_residual -- IMPLEMENTED (2026-09-01)](docs/substrate/SD-e1-rollout-consistency-training-absolute-vs-residual-branch-e1-rollout-output.md) *(~1,319 tok)*
    - [ITEM 2: e1.transition.rollout_consistency -- IMPLEMENTED (2026-09-01)](docs/substrate/SD-e1-rollout-consistency-training-item-2-e1-transition-rollout-consistency.md) *(~1,613 tok)*
- **[SD-hazard-aware-policy-decomposition](docs/substrate/SD-hazard-aware-policy-decomposition-policy-decomposition-via-event.md)** — policy.decomposition_via_event_segmenter.harm_aware_selection -- IMPLEMENTED 2026-08-01 *(~1,330 tok)*
- **MECH-027** — 2 records, ~2,443 tok total.
    - [precision-scaled commit temperature -- graded consumer for current_precision (2026-09-02)](docs/substrate/MECH-027-precision-scaled-commit-temperature.md) *(~1,159 tok)*
    - [Build 2: force_sleep_cycle_at_eval_boundary -- sleep-cycle interleave reachable inside an eval window (2026-09-02)](docs/substrate/MECH-027-build-2-force-sleep-cycle-at-eval.md) *(~1,283 tok)*
- **MECH-090** — 4 records, ~5,077 tok total.
    - [Layer 1 + MECH-091 Layer 2: Trajectory Stepping + Urgency Interrupt (2026-04-15)](docs/substrate/MECH-090-layer-1-mech-091-layer-2-trajectory.md) *(~610 tok)*
    - [Commit-Entry Predicate: R-c single-gate readiness conjunction (2026-05-28)](docs/substrate/MECH-090-commit-entry-predicate-r-c-single-gate.md) *(~1,483 tok)*
    - [R-c continuation: nav_competence axis (2026-05-29)](docs/substrate/MECH-090-r-c-continuation-nav-competence-axis.md) *(~1,482 tok)*
    - [R-c continuation Phase-2 follow-on: env-emitted readiness-outcome source (2026-06-02)](docs/substrate/MECH-090-r-c-continuation-phase-2-follow-on-env.md) *(~1,501 tok)*
- **[MECH-091](docs/substrate/MECH-091-mech091-salient-event-trigger-wiring.md)** — MECH091-SALIENT-EVENT-TRIGGER-WIRING / MECH-091: Salient-Event Trigger Wiring -- IMPLEMENTED (2026-08-17) *(~843 tok)*
- **[MECH-120](docs/substrate/MECH-120-shy-synaptic-homeostasis-wiring.md)** — SHY Synaptic Homeostasis Wiring (2026-04-08) *(~290 tok)*
- **[MECH-122](docs/substrate/MECH-122-mech122-content-packaging-spindle-selection.md)** — MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (MECH-122 content-packaging half; V3 proxy) -- IMPLEMENTED (2026-08-02) *(~1,345 tok)*
- **[MECH-140](docs/substrate/MECH-140-x-mech-450-disinhibitory-soft.md)** — x MECH-450: disinhibitory soft-competitive settling (parameter-free) (2026-07-02) *(~1,295 tok)*
- **[MECH-189](docs/substrate/MECH-189-super-ordinal-goal-anchor-contextmemory.md)** — Super-ordinal goal-anchor ContextMemory writes substrate (infant_substrate:GAP-11) (2026-06-09) *(~2,080 tok)*
- **[MECH-204](docs/substrate/MECH-204-phase-7-option-b-sleep-accuracy-anchored.md)** — Phase 7 / Option B: sleep.accuracy_anchored_broadcast_recalibration -- IMPLEMENTED (2026-07-20) *(~773 tok)*
- **[MECH-205](docs/substrate/MECH-205-surprise-gated-replay-write-path-fix.md)** — Surprise-Gated Replay Write Path Fix (2026-04-09) *(~350 tok)*
- **[MECH-216](docs/substrate/MECH-216-e1-predictive-wanting-schema-readout.md)** — E1 Predictive Wanting / Schema Readout (2026-04-09) *(~358 tok)*
- **[MECH-219](docs/substrate/MECH-219-sd-019b-affective-harm-hysteretic.md)** — (SD-019b): affective-harm hysteretic integrator (z_harm_suffering) (2026-06-10) *(~1,797 tok)*
- **[MECH-266](docs/substrate/MECH-266-cingulate-asymmetric-per-mode-hysteresis.md)** — cingulate.asymmetric_per_mode_hysteresis -- IMPLEMENTED (2026-04-21) *(~769 tok)*
- **MECH-269** — 3 records, ~4,377 tok total.
    - [Base Substrate -- Phase 1 (2026-04-22)](docs/substrate/MECH-269-base-substrate-phase-1.md) *(~1,095 tok)*
    - [Anchor Sets -- Phase 2 (ii) (2026-04-22)](docs/substrate/MECH-269-anchor-sets-phase-2-ii.md) *(~1,564 tok)*
    - [Per-Region V_s Readout -- Phase 2 (iii, T4) (2026-04-22)](docs/substrate/MECH-269-per-region-v-s-readout-phase-2-iii-t4.md) *(~1,717 tok)*
- **MECH-269b** — 2 records, ~2,078 tok total.
    - [Symmetric V_s Gating on E1/E2 Cortical Rollouts (2026-04-26)](docs/substrate/MECH-269b-symmetric-v-s-gating-on-e1-e2-cortical.md) *(~1,206 tok)*
    - [MECH-284 Staleness-into-Gate Wiring (Q-040b strong reading, 2026-04-29)](docs/substrate/MECH-269b-mech-284-staleness-into-gate-wiring-q.md) *(~871 tok)*
- **[MECH-276](docs/substrate/MECH-276-scientist-agent-counterfactual-backed.md)** — scientist-agent counterfactual-backed attribution feedstock (waking-phase mechanism feeding the MECH-275 sleep aggregator) (2026-06-23) *(~2,341 tok)*
- **[MECH-284](docs/substrate/MECH-284-staleness-accumulator-mech-269-online.md)** — Staleness Accumulator + MECH-269 Online Hysteresis -- Phase 3 (2026-04-24) *(~1,574 tok)*
- **[MECH-286](docs/substrate/MECH-286-override-gated-sleep-onset.md)** — Override-Gated Sleep Onset (2026-05-21) *(~379 tok)*
- **[MECH-287](docs/substrate/MECH-287-invalidation-trigger-phase-2-iv.md)** — Invalidation Trigger -- Phase 2 iv (2026-04-22) *(~1,480 tok)*
- **[MECH-288](docs/substrate/MECH-288-event-segmenter-phase-2.md)** — Event Segmenter -- Phase 2 (2026-04-22) *(~2,202 tok)*
- **[MECH-292](docs/substrate/MECH-292-ranked-ghost-goal-bank.md)** — Ranked Ghost-Goal Bank (2026-04-27) *(~1,444 tok)*
- **[MECH-293](docs/substrate/MECH-293-waking-ghost-goal-probe-search.md)** — Waking Ghost-Goal Probe Search (2026-04-27) *(~1,687 tok)*
- **MECH-294** — 3 records, ~5,709 tok total.
    - [multi-content theta-burst packet (joint {goal,action,risk,state} per-cycle binding) (2026-06-09)](docs/substrate/MECH-294-multi-content-theta-burst-packet-joint.md) *(~2,316 tok)*
    - [AMEND: compose path reads within-cycle co-binding coherence (mode-dependent E3 bias) (2026-06-09)](docs/substrate/MECH-294-amend-compose-path-reads-within-cycle.md) *(~1,496 tok)*
    - [AMEND: per-candidate co-binding coherence (cross-candidate-range rendering so the route-range authority + 569i top-k can carve) (2026-06-19)](docs/substrate/MECH-294-amend-per-candidate-co-binding.md) *(~1,896 tok)*
- **[MECH-295](docs/substrate/MECH-295-drive-to-liking-stream-to-approach-cue.md)** — Drive -> Liking-Stream -> Approach Cue Bridge (2026-04-26) *(~1,451 tok)*
- **MECH-307** — 3 records, ~4,098 tok total.
    - [Anticipatory Affect Conjunction Architecture (2026-05-11)](docs/substrate/MECH-307-anticipatory-affect-conjunction.md) *(~1,823 tok)*
    - [Default-Value Recalibration (2026-05-12)](docs/substrate/MECH-307-default-value-recalibration.md) *(~968 tok)*
    - [from_dims() Reachability Repair (2026-08-07)](docs/substrate/MECH-307-from-dims-reachability-repair.md) *(~1,305 tok)*
- **[MECH-313](docs/substrate/MECH-313-arc-065-child-stochastic-noise-floor-lc.md)** — (ARC-065 child): Stochastic Noise Floor (LC-NE tonic / SAC analog) (2026-05-10) *(~1,439 tok)*
- **[MECH-314](docs/substrate/MECH-314-arc-065-child-structured-curiosity.md)** — (ARC-065 child): Structured Curiosity Bonus + 3 Sub-Flavours (2026-05-10) *(~1,650 tok)*
- **[MECH-314a](docs/substrate/MECH-314a-phase-2-amend-e2-world-forward-novelty.md)** — Phase-2 AMEND: e2.world_forward novelty-candidate-source (V3-EXQ-648 autopsy, 2026-06-07) *(~1,346 tok)*
- **[MECH-319](docs/substrate/MECH-319-arc-062-gap-k-simulation-mode-rule.md)** — (arc_062 GAP-K): Simulation-Mode Rule-Write Gate (Categorical Replay Tag) (2026-05-10) *(~2,006 tok)*
- **[MECH-320](docs/substrate/MECH-320-arc-066-child-tonic-vigor-coupling.md)** — (ARC-066 child): Tonic Vigor Coupling Score Bias (mesolimbic-DA-vigor / avg-reward-rate) (2026-05-10) *(~2,209 tok)*
- **[MECH-339](docs/substrate/MECH-339-c1-composite-cue-outshining-gate.md)** — C1 Composite Cue + Outshining Gate (2026-05-19) *(~892 tok)*
- **[MECH-340](docs/substrate/MECH-340-persistence-efficacy-gate.md)** — Persistence / Efficacy Gate (2026-05-21) *(~413 tok)*
- **MECH-341** — 3 records, ~5,384 tok total.
    - [(ARC-065 Layer-B child): E3 Score Diversity Preservation (2026-05-27)](docs/substrate/MECH-341-arc-065-layer-b-child-e3-score.md) *(~2,131 tok)*
    - [Amend: stratified_within_class_temperature + A-vs-B partial-redundancy probe (2026-06-01)](docs/substrate/MECH-341-amend-stratified-within-class.md) *(~1,878 tok)*
    - [Retune: stratified_select call-site expansion + 6-arm validation (2026-05-28)](docs/substrate/MECH-341-retune-stratified-select-call-site.md) *(~1,375 tok)*
- **[MECH-342](docs/substrate/MECH-342-maintenance-time-readiness-driven.md)** — Maintenance-time readiness-driven commitment-release coupling (B3b) (2026-06-02) *(~1,764 tok)*
- **[MECH-353](docs/substrate/MECH-353-blocked-agency-control-failure-affect.md)** — blocked-agency / control-failure affect stream (z_block) (2026-06-06) *(~1,963 tok)*
- **[MECH-358](docs/substrate/MECH-358-post-603i-successor-scaffold-trainable.md)** — Post-603i successor scaffold: trainable relief/safety escape-affordance learner (2026-06-08) *(~526 tok)*
- **[MECH-423](docs/substrate/MECH-423-readiness-substrate-r2-iterative.md)** — readiness substrate: R2 iterative-inference convergence + R3 interleaved cross-module consolidation + R1 shared-latent grad probe (2026-06-12) *(~2,041 tok)*
- **[MECH-439](docs/substrate/MECH-439-f-dominance-conflict-grade-factor-a.md)** — F-dominance conflict-grade -- Factor A conflict-graded shortlist width + Factor B gap-scaled commit-T (2026-06-18) *(~1,884 tok)*
- **[MECH-440](docs/substrate/MECH-440-mech-441-state-conditioned-exploration.md)** — MECH-441: state-conditioned exploration -- propagating selection-head weight noise (NoisyNet) + model-disagreement directed curiosity (RND/Plan2Explore) (2026-06-27) *(~1,289 tok)*
- **MECH-448** — 2 records, ~3,620 tok total.
    - [ARC-107: rank-preserving F->eligibility demotion (LEAD lever of the basal-ganglia E3-selector constitution) (2026-06-20)](docs/substrate/MECH-448-arc-107-rank-preserving-f-to.md) *(~2,189 tok)*
    - [AMEND: channel-adaptive (mean-relative) eligibility floor (collapse ~5 per-channel hand-floor dances into one knob) (2026-06-21)](docs/substrate/MECH-448-amend-channel-adaptive-mean-relative.md) *(~1,430 tok)*
- **[MECH-449](docs/substrate/MECH-449-arc-107-go-no-go-eligibility.md)** — ARC-107: Go/No-Go eligibility constitution (the OPPONENCY leg of the basal-ganglia E3-selector constitution; generalises MECH-260) (2026-06-21) *(~1,904 tok)*
- **[MECH-450](docs/substrate/MECH-450-arc-108-job-1-step-2-learned-recurrent.md)** — (ARC-108 JOB-1 step-2): learned recurrent-settling step + learned lateral-inhibition W_lat (factor 2 of the learned-gating 2x2; B1 + B3-blend repair) (2026-06-22) *(~2,130 tok)*
- **[MECH-451](docs/substrate/MECH-451-finer-channel-granularity-e3-selection.md)** — finer-channel-granularity E3 selection-gating (the cheap V3 rung BETWEEN ARC-108's single global w_chan and ARC-110's V4 segregated loops; explode the compressed score_bias blend into separately-learnable per-head channels) (2026-06-24) *(~2,860 tok)*
- **MECH-457** — 6 records, ~6,671 tok total.
    - [first-class RPE-driven actor-critic action-learning substrate (2026-07-12)](docs/substrate/MECH-457-first-class-rpe-driven-actor-critic.md) *(~709 tok)*
    - [mech457_competence_bootstrap_explorer: action_learning.competence_bootstrap_explorer -- IMPLEMENTED (2026-07-16)](docs/substrate/MECH-457-mech457-competence-bootstrap-explorer-action.md) *(~1,440 tok)*
    - [mech457_bc_aux_schedule: action_learning.bc_auxiliary_persistence_schedule -- IMPLEMENTED (2026-07-18)](docs/substrate/MECH-457-mech457-bc-aux-schedule-action-learning-bc.md) *(~1,017 tok)*
    - [mech457_retention_trajectory_probe: action_learning.mid_training_competence_probe -- IMPLEMENTED (2026-07-19)](docs/substrate/MECH-457-mech457-retention-trajectory-probe-action.md) *(~1,524 tok)*
    - [mech457_consummatory_act: environment.consummatory_act -- IMPLEMENTED (2026-07-25)](docs/substrate/MECH-457-mech457-consummatory-act-environment.md) *(~1,028 tok)*
    - [mech457_approach_extinction: experiments/_lib approach-drive extinction-on-contact -- IMPLEMENTED (2026-07-25)](docs/substrate/MECH-457-mech457-approach-extinction-experiments-lib.md) *(~953 tok)*
- **[MECH-463](docs/substrate/MECH-463-e3-commit-gate-per-candidate-channel.md)** — E3 commit-gate + per-candidate channel-term diagnostics (arousal-conditioned variance decomposition instrumentation) (2026-07-18) *(~978 tok)*
- **[ARC-006](docs/substrate/ARC-006-mech-045-token-instance-object-file.md)** — MECH-045: token-instance object-file / entity-persistence buffer (2026-06-09) *(~1,446 tok)*
- **[ARC-033](docs/substrate/ARC-033-e2-harm-s-forward-model.md)** — E2_harm_s Forward Model (2026-04-09) *(~508 tok)*
- **[ARC-058](docs/substrate/ARC-058-harm-stream-shared-forward-trunk.md)** — harm_stream.shared_forward_trunk -- REGISTERED (2026-04-19) *(~370 tok)*
- **ARC-062** — 4 records, ~5,010 tok total.
    - [Phase 1 gated-policy (GAP-A, 2026-05-09)](docs/substrate/ARC-062-phase-1-gated-policy-gap-a-2026-05-09.md) *(~323 tok)*
    - [GatedPolicy differential-heads robustness fix (2026-05-18)](docs/substrate/ARC-062-gatedpolicy-differential-heads.md) *(~1,190 tok)*
    - [GAP-B mode-separation floor (2026-05-20)](docs/substrate/ARC-062-gap-b-mode-separation-floor.md) *(~222 tok)*
    - [Phase 1: Gated-Policy Heads + Context Discriminator (2026-05-09)](docs/substrate/ARC-062-phase-1-gated-policy-heads-context.md) *(~3,274 tok)*
- **ARC-063** — 3 records, ~4,943 tok total.
    - [v1: distributed CandidateRule field (GAP-B non-Bayesian rule-creator) (2026-06-04)](docs/substrate/ARC-063-v1-distributed-candidaterule-field-gap.md) *(~1,858 tok)*
    - [AMEND: cross-episode rule-persistence flag (V3-EXQ-654 GAP-B maturity) (2026-06-09)](docs/substrate/ARC-063-amend-cross-episode-rule-persistence.md) *(~1,363 tok)*
    - [AMEND: mature-pool gate/credit/retire dynamics (V3-EXQ-654b GAP-B maturity) (2026-06-11)](docs/substrate/ARC-063-amend-mature-pool-gate-credit-retire.md) *(~1,721 tok)*
- **ARC-065** — 2 records, ~2,444 tok total.
    - [SP-CEM Main-Path Landing (2026-05-17)](docs/substrate/ARC-065-sp-cem-main-path-landing.md) *(~799 tok)*
    - [GAP-A: shared cand_world_summaries e2.world_forward source (V3-EXQ-614e autopsy, 2026-06-07)](docs/substrate/ARC-065-gap-a-shared-cand-world-summaries-e2.md) *(~1,645 tok)*
- **[ARC-070](docs/substrate/ARC-070-mech-321-policy-decomposition-via-event.md)** — MECH-321: policy.decomposition_via_event_segmenter -- IMPLEMENTED (2026-07-24) *(~2,719 tok)*
- **ARC-071** — 5 records, ~10,053 tok total.
    - [policy.composition_via_repeated_grounding -- IMPLEMENTED (2026-07-22)](docs/substrate/ARC-071-policy-composition-via-repeated-grounding.md) *(~1,711 tok)*
    - [MECH-324: chunk dissolution is SUPPRESSION-WITH-RETENTION, not erasure -- IMPLEMENTED (2026-07-27)](docs/substrate/ARC-071-mech-324-chunk-dissolution-is-suppression-with.md) *(~2,303 tok)*
    - [MECH-323: chunk CREDIT RULE is all-position, not trailing-only -- IMPLEMENTED (2026-07-27)](docs/substrate/ARC-071-mech-323-chunk-credit-rule-is-all-position-not.md) *(~1,628 tok)*
    - [MECH-323: chunk SIZE and chunk DEPTH are GROWABLE CEILINGS DERIVED FROM THE DELIBERATION BUDGET, not fiat constants -- IMPLEMENTED (2026-07-27)](docs/substrate/ARC-071-mech-323-chunk-size-and-chunk-depth-are.md) *(~3,140 tok)*
    - [MECH-324: reacquisition-window ISOLATION fix -- corrects V3-EXQ-829's confirmed FALSIFIED result -- IMPLEMENTED (2026-07-31)](docs/substrate/ARC-071-mech-324-reacquisition-window-isolation-fix.md) *(~1,271 tok)*
- **ARC-108** — 4 records, ~7,509 tok total.
    - [JOB-1 step-1: learned dopamine-gated E3 selection (signed-RPE w_chan over the modulatory channels; the next MECH-439 attack, learned-not-arithmetic) (2026-06-22)](docs/substrate/ARC-108-job-1-step-1-learned-dopamine-gated-e3.md) *(~2,133 tok)*
    - [JOB-2: dopaminergic control-plane DRIVER pair -- rho_t maintenance ramp + habenula negative-delta_t de-commit (the driver of the commit/maintain/de-commit machinery REE built but never gave its neuromodulator) (2026-06-22)](docs/substrate/ARC-108-job-2-dopaminergic-control-plane-driver.md) *(~2,396 tok)*
    - [sec-7 C3: learned_channel_rpe_mode signed/unsigned ablation flag (unblocks V3-EXQ-700 C3 arm; the signed-RPE-is-load-bearing falsifier knob) (2026-06-22)](docs/substrate/ARC-108-sec-7-c3-learned-channel-rpe-mode.md) *(~1,336 tok)*
    - [x ARC-110 coupling: LEARNED (dopamine-gated) CROSS-LOOP arbitration -- the named next attack on the F-dominance conversion ceiling (MECH-439) after V3-EXQ-707b (2026-07-01)](docs/substrate/ARC-108-x-arc-110-coupling-learned-dopamine.md) *(~1,642 tok)*
- **ARC-110** — 3 records, ~5,234 tok total.
    - [parallel segregated cortico-BG-thalamic loops (motor / associative / limbic) + S2 in-layer null + ARC-109 D1/D2 split + MECH-452 loop-local traces (2026-06-27)](docs/substrate/ARC-110-parallel-segregated-cortico-bg-thalamic.md) *(~2,216 tok)*
    - [x ARC-108: ascending-spiral gain (V3-EXQ-709/710 loop-effective-weight repair) (2026-07-03)](docs/substrate/ARC-110-x-arc-108-ascending-spiral-gain-v3-exq.md) *(~1,361 tok)*
    - [x ARC-108: BOUNDED ascending-spiral gain -- target-PARITY controller (V3-EXQ-711 runaway repair) (2026-07-04)](docs/substrate/ARC-110-x-arc-108-bounded-ascending-spiral-gain.md) *(~1,657 tok)*
- **[INV-091](docs/substrate/INV-091-inv091-null-validation-run-length.md)** — INV091-NULL-VALIDATION-RUN-LENGTH: standing default eval budget for the INV-091 driver family *(~726 tok)* **standing eval-budget DEFAULT -- a new INV-091-family driver must import these constants.**
- **[Q-080](docs/substrate/Q-080-effort-dissociating-environment-q-080.md)** — Effort-dissociating environment (Q-080) (2026-07-09) *(~758 tok)*
- **Q-081** — 2 records, ~1,973 tok total. **shared recording harness + reach probe -- read before writing a cross-stream-organisation driver.**
    - [per-step multi-stream trace recorder (`experiments/_lib/`, IMPLEMENTED 2026-07-22)](docs/substrate/Q-081-per-step-multi-stream-trace-recorder.md) *(~864 tok)*
    - [Q081-REACH-CHECK-PAIR-SPECIFIC: cheap empirical pre-flight reach probe for Q-081](docs/substrate/Q-081-q081-reach-check-pair-specific-cheap.md) *(~1,109 tok)*
- **[DR-10](docs/substrate/DR-10-self-model-v4-self-3-z-self-enters-e3.md)** — (self_model_v4:SELF-3): z_self enters E3 trajectory viability scoring (2026-07-01) *(~1,187 tok)*
- **[DR-12](docs/substrate/DR-12-self-model-v4-self-4-e2-forward-pe-to.md)** — (self_model_v4:SELF-4): E2 forward-PE -> E3 trajectory-scoring confidence down-weight (FIRST V4 SUBSTRATE BUILD, 2026-06-17) *(~1,438 tok)*
- **[DR-13](docs/substrate/DR-13-self-model-v4-self-1-z-self-temporal.md)** — (self_model_v4:SELF-1): z_self temporal depth -- dedicated self-recurrence anchored by E1 feedback (2026-07-01) *(~1,414 tok)*
- **[ACTION-OBJECT-ROUND-TRIP](docs/substrate/ACTION-OBJECT-ROUND-TRIP-action-object-round-trip-is-not-an.md)** — Action-object round trip is NOT an action source + CEM elite floor (2026-07-22) *(~2,128 tok)*
- **[arm-fingerprint-executed-substrate-identity](docs/substrate/arm-fingerprint-executed-substrate-identity-arm-fingerprint-records-the-executed-substrate.md)** — arm-fingerprint-executed-substrate-identity -- arm_fingerprint records the EXECUTED substrate, not the on-disk substrate -- IMPLEMENTED (2026-07-20) *(~1,559 tok)*
- **[CAUSALGRIDWORLDV2](docs/substrate/CAUSALGRIDWORLDV2-causalgridworldv2-max-episode-steps.md)** — CausalGridWorldV2: max_episode_steps constructor kwarg -- IMPLEMENTED (2026-08-10) *(~1,009 tok)*
- **[CEILING-ANCHOR-FLOOR-GUARD](docs/substrate/CEILING-ANCHOR-FLOOR-GUARD-ceiling-below-random-anchor-guard.md)** — Ceiling-below-random-anchor guard + standing lint (competence-objective autopsy 734/737b/742a) (2026-08-01) *(~765 tok)* **standing LINT + runtime guard -- read before self-routing any diagnostic to a ceiling label.**
- **[consummatory_aware_reference_policies](docs/substrate/consummatory_aware_reference_policies-experiments-lib-capability-eval-py.md)** — consummatory_aware_reference_policies: experiments/_lib/capability_eval.py -- IMPLEMENTED (2026-07-25) *(~565 tok)*
- **contextmemory-write-path-addressing-degeneracy** — 3 records, ~4,635 tok total.
    - [contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_usage_balancing -- IMPLEMENTED (2026-08-19)](docs/substrate/contextmemory-write-path-addressing-degeneracy-e1-context-memory-write-usage-balancing.md) *(~996 tok)*
    - [contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_selection="refractory" -- IMPLEMENTED (2026-08-19)](docs/substrate/contextmemory-write-path-addressing-degeneracy-e1-context-memory-write-selection-refractory.md) *(~1,752 tok)*
    - [contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_selection="gumbel_learned" -- IMPLEMENTED (2026-08-27)](docs/substrate/contextmemory-write-path-addressing-degeneracy-e1-context-memory-write-selection-gumbel-learned.md) *(~1,887 tok)*
- **[CONTROLVECTOR-LOGGING](docs/substrate/CONTROLVECTOR-LOGGING-controlvector-logging-four-signal.md)** — ControlVector logging: four-signal control telemetry (rec-B, 2026-06-07) *(~931 tok)*
- **[crf-availability-maintenance](docs/substrate/crf-availability-maintenance-activity-silent-maintenance-trace.md)** — activity-silent maintenance trace + maintained-pool readout (V3-EXQ-666 successor; ARC-063 amend) (2026-06-11) *(~1,981 tok)*
- **[cross_stream_binding_substrate](docs/substrate/cross_stream_binding_substrate-shared-latent-factor-cross-stream.md)** — Shared-latent-factor cross-stream binding (2026-07-08) *(~2,143 tok)*
- **[dose_saturation](docs/substrate/dose_saturation-lint.md)** — dose_saturation lint -- IMPLEMENTED (2026-07-22) *(~729 tok)*
- **[DV-HEADROOM](docs/substrate/DV-HEADROOM-dv-headroom-class-validate-experiments.md)** — DV-headroom class (`validate_experiments.py` lint + `experiments/_metrics.py` precondition kind, IMPLEMENTED 2026-09-04) *(~1,071 tok)* **standing LINT + precondition kind -- read before adding a DV-headroom precondition to any driver.**
- **F-DOMINANCE-RUNG6** — 5 records, ~9,979 tok total.
    - [Commit/release-DURATION lever: graded natural-commit-occupancy release (rung-6 of f_dominance_conversion_ceiling; PARALLEL to MECH-448) (2026-06-20)](docs/substrate/F-DOMINANCE-RUNG6-commit-release-duration-lever-graded.md) *(~2,000 tok)*
    - [Natural-commit LATCH-HOLD amend: establish the sustained-hold OFF baseline (V3-EXQ-460i gate amend) (2026-06-21)](docs/substrate/F-DOMINANCE-RUNG6-natural-commit-latch-hold-amend.md) *(~1,769 tok)*
    - [Closure-exclusive de-commit eval mode (rung-6 BUILD of f_dominance_conversion_ceiling; the named dissociable substrate from V3-EXQ-460j) (2026-06-22)](docs/substrate/F-DOMINANCE-RUNG6-closure-exclusive-de-commit-eval-mode.md) *(~1,697 tok)*
    - [F-independent closure-plane commit-ENTRY primitive (rung-6 amend; the F-INDEPENDENT arm source the closure-exclusive de-commit eval lacked; closes the 460k/460l ncl_hold_closure_armed_total=0 signature) (2026-06-23)](docs/substrate/F-DOMINANCE-RUNG6-f-independent-closure-plane-commit.md) *(~2,240 tok)*
    - [F-independent closure-plane commit-ENTRY TRAJECTORY primitive (C-STEP extension of the bool latch; the between-tick path now STEPS a closure-formed committed program, not repeats _last_action) (2026-06-23)](docs/substrate/F-DOMINANCE-RUNG6-f-independent-closure-plane-commit-2.md) *(~2,270 tok)*
- **[GATE-DV](docs/substrate/GATE-DV-gate-level-dv-instrument-experiments.md)** — Gate-level DV instrument (`experiments/_lib/gate_dv.py`, IMPLEMENTED 2026-08-19) *(~677 tok)* **shared instrument -- read before adding gate-level DV instrumentation to a driver.**
- **INFANT-CURRICULUM-SCHEDULER** — 2 records, ~3,150 tok total.
    - [InfantCurriculumScheduler Phase 0->1 H_pos Floor Recalibration (2026-05-31)](docs/substrate/INFANT-CURRICULUM-SCHEDULER-infantcurriculumscheduler-phase-0-to-1.md) *(~1,825 tok)*
    - [InfantCurriculumScheduler Phase 0->1 crossing-count criterion (V3-EXQ-591f; GAP-14 c-2) (2026-06-19)](docs/substrate/INFANT-CURRICULUM-SCHEDULER-infantcurriculumscheduler-phase-0-to-1-2.md) *(~1,324 tok)*
- **mode-governance-engagement** — 2 records, ~3,114 tok total.
    - [external_task salience source for SalienceCoordinator (2026-06-13)](docs/substrate/mode-governance-engagement-external-task-salience-source-for.md) *(~1,634 tok)*
    - [regime-occupancy gradedness is a REPRODUCIBILITY test, not an existential one (2026-09-11)](docs/substrate/mode-governance-engagement-regime-occupancy-reproducibility.md) *(~1,480 tok)*
- **modulatory-bias-selection-authority** — 6 records, ~10,455 tok total.
    - [gap-relative E3.select authority (2026-06-03)](docs/substrate/modulatory-bias-selection-authority-gap-relative-e3-select-authority.md) *(~1,169 tok)*
    - [AMEND: float32 catastrophic-cancellation fix (V3-EXQ-643a, 2026-06-06)](docs/substrate/modulatory-bias-selection-authority-amend-float32-catastrophic-cancellation.md) *(~1,333 tok)*
    - [AMEND: route upstream-channel range into the bias the authority rescales (569f/661/654a, 2026-06-10)](docs/substrate/modulatory-bias-selection-authority-amend-route-upstream-channel-range-into.md) *(~2,267 tok)*
    - [AMEND: gain/contrast + shortlist-then-modulate conversion (569g/682, 2026-06-15)](docs/substrate/modulatory-bias-selection-authority-amend-gain-contrast-shortlist-then.md) *(~1,624 tok)*
    - [AMEND: TOP-K shortlist mode (569h conversion-ceiling, 2026-06-16)](docs/substrate/modulatory-bias-selection-authority-amend-top-k-shortlist-mode-569h.md) *(~1,532 tok)*
    - [AMEND: CEM elite-stage authority + behavioural throughput (V3-EXQ-931, 2026-08-19)](docs/substrate/modulatory-bias-selection-authority-amend-cem-elite-stage-authority.md) *(~2,528 tok)*
- **scaffolded_sd054_onboarding** — 15 records, ~21,778 tok total.
    - [AMEND: opt-in STRICT goal isolation (2026-07-27)](docs/substrate/scaffolded_sd054_onboarding-amend-opt-in-strict-goal-isolation.md) *(~1,321 tok)*
    - [Substrate (2026-05-31)](docs/substrate/scaffolded_sd054_onboarding-substrate.md) *(~1,783 tok)*
    - [AMEND: update_z_goal wiring + Stage-0 positive control (2026-06-02)](docs/substrate/scaffolded_sd054_onboarding-amend-update-z-goal-wiring-stage-0.md) *(~1,323 tok)*
    - [AMEND: nursery/feeding scaffold (forced-benefit Stage-0 + survival levers + P2 guard) (2026-06-03)](docs/substrate/scaffolded_sd054_onboarding-amend-nursery-feeding-scaffold-forced.md) *(~1,743 tok)*
    - [AMEND: developmental-window / protected-goal consolidation (2026-06-03b)](docs/substrate/scaffolded_sd054_onboarding-amend-developmental-window-protected.md) *(~1,352 tok)*
    - [AMEND: seeding-calibration + consumption-gated G3 (2026-06-03c)](docs/substrate/scaffolded_sd054_onboarding-amend-seeding-calibration-consumption.md) *(~1,701 tok)*
    - [AMEND: SD-057 cue-recall bridge (wean-to-wild foraging-contact lever) (2026-06-04)](docs/substrate/scaffolded_sd054_onboarding-amend-sd-057-cue-recall-bridge-wean-to.md) *(~1,277 tok)*
    - [AMEND: cue-recall FORMATION fix + diagnostics (2026-06-04b)](docs/substrate/scaffolded_sd054_onboarding-amend-cue-recall-formation-fix.md) *(~1,165 tok)*
    - [AMEND: n_cue_recall_fires aggregation fix (2026-06-04c)](docs/substrate/scaffolded_sd054_onboarding-amend-n-cue-recall-fires-aggregation-fix.md) *(~812 tok)*
    - [AMEND: post-cue action/gradient instrumentation (V3-EXQ-640, 2026-06-05)](docs/substrate/scaffolded_sd054_onboarding-amend-post-cue-action-gradient.md) *(~926 tok)*
    - [AMEND: foraging-competence residual (gating-seeding reconcile + reef-spawn weaning + consumption-gated G3) (2026-06-05)](docs/substrate/scaffolded_sd054_onboarding-amend-foraging-competence-residual.md) *(~1,488 tok)*
    - [AMEND: curriculum decomposition -- isolated hazard-avoidance stage (Stage-H) (2026-06-07)](docs/substrate/scaffolded_sd054_onboarding-amend-curriculum-decomposition-isolated.md) *(~1,742 tok)*
    - [AMEND: Stage-H harm-pathway training (603i nav/survival-competence ceiling) (2026-06-09)](docs/substrate/scaffolded_sd054_onboarding-amend-stage-h-harm-pathway-training.md) *(~1,689 tok)*
    - [AMEND: harm-pathway training STABILIZATION (decoupled encoder LR + LR warmup; 603p seed-fragility) (2026-06-16)](docs/substrate/scaffolded_sd054_onboarding-amend-harm-pathway-training.md) *(~1,558 tok)*
    - [AMEND: Leg C rule_bias_head training (commitment_closure:GAP-4) (2026-06-16)](docs/substrate/scaffolded_sd054_onboarding-amend-leg-c-rule-bias-head-training.md) *(~1,893 tok)*
- **SLEEP-AGGREGATION-CLUSTER** — 5 records, ~2,262 tok total.
    - [Sleep Aggregation Cluster Phase A: Scaffolding (2026-04-25)](docs/substrate/SLEEP-AGGREGATION-CLUSTER-sleep-aggregation-cluster-phase-a.md) *(~390 tok)*
    - [Sleep Aggregation Cluster Phase B: MECH-285 SleepReplaySampler (2026-04-25)](docs/substrate/SLEEP-AGGREGATION-CLUSTER-sleep-aggregation-cluster-phase-b-mech.md) *(~357 tok)*
    - [Sleep Aggregation Cluster Phase C: MECH-272 RoutingGate (2026-04-25)](docs/substrate/SLEEP-AGGREGATION-CLUSTER-sleep-aggregation-cluster-phase-c-mech.md) *(~321 tok)*
    - [Sleep Aggregation Cluster Phase D: MECH-275 BayesianAggregator (2026-04-25)](docs/substrate/SLEEP-AGGREGATION-CLUSTER-sleep-aggregation-cluster-phase-d-mech.md) *(~514 tok)*
    - [Sleep Aggregation Cluster Phase E: MECH-273 SelfModelAggregator (2026-04-25)](docs/substrate/SLEEP-AGGREGATION-CLUSTER-sleep-aggregation-cluster-phase-e-mech.md) *(~678 tok)*
- **[sleep_substrate:GAP-9](docs/substrate/sleep_substrate-GAP-9-sleep-within-life-sleep-trigger.md)** — sleep_substrate:GAP-9: sleep.within_life_sleep_trigger -- IMPLEMENTED (2026-08-14) *(~1,292 tok)*
- **[stdlib_rng_seed](docs/substrate/stdlib_rng_seed-stdlib-random-seeding-in-ree-core-reeconfig.md)** — stdlib-`random` seeding in ree_core (`REEConfig.stdlib_rng_seed`) -- IMPLEMENTED (2026-07-28) *(~2,827 tok)*
