# ree-v3 CLAUDE.md -- long form of compressed sections

Moved verbatim out of [`ree-v3/CLAUDE.md`](../../CLAUDE.md) on 2026-09-26 (session
`reev3-claudemd-trim-20260926`, following the umbrella's `.claude/rules/claude-md-placement.md`).
`CLAUDE.md` keeps every gate, prohibition and must-run command, compressed; this file holds the
detail behind them: the burned-queue-entry auditor's mechanism, the regression-suite layout, the
full V3 / V4 scope lists, the per-machine runtime calibration for `estimated_minutes`, and the
superseded ree-v3 copy of the EXQ versioning rules (the umbrella CLAUDE.md is canonical for those).

**Read this when** estimating `estimated_minutes` for a queue entry, deciding whether a mechanism is
V3 or V4 scope, re-queuing an id that vanished from the queue, or changing the regression layers.
Links are repo-root-relative (verbatim move, left unrewritten).

---

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
- Every queue entry needs `estimated_minutes`; the runner's auto-calibration refines it.
- Estimate from: total episodes × steps_per_episode, calibrated against known runtimes:
  - **Mac (`DLAPTOP`)** — CPU, CausalGridWorldV2, typical REE agent:
    - ~0.10 min/ep at 200 steps/ep
    - ~0.15 min/ep at 300 steps/ep
  - **Daniel-PC** — CPU preferred (GPU 3x slower at current model scale, batch=1):
    - ~0.50 min/ep at 200 steps/ep  (~5x slower than Mac)
    - ~0.72 min/ep at 300 steps/ep
    - Calibrated from onboarding smoke runs 2026-03-22: 7.0 steps/sec CPU, 2.1 steps/sec GPU
    - GPU never wins at current model scale (world_dim=32): EXQ-070 tested batch 1-512,
      CPU always faster (200k vs 133k samples/s at batch=512). RTX 2060 Super overhead
      dominates for tiny networks. GPU becomes useful ONLY when world_dim >= 128 or
      networks are substantially deeper. Design experiments with larger networks to
      exploit the GPU when the architecture requires it (SD-004, SD-010).
  - **ree-cloud-1** — Hetzner CX22, CPU-only (no GPU), 2 shared vCPU:
    - ~0.23 min/ep at 200 steps/ep  (~2.3x slower than Mac)
    - ~0.35 min/ep at 300 steps/ep
    - Calibrated from onboarding smoke 2026-04-09: 14.2 steps/sec CPU, 1571.9 env steps/sec
    - Suitable for env-heavy and standard experiments. Not for GPU-dependent runs.
  - **ree-cloud-2** — Hetzner CX22, CPU-only, 2 shared vCPU (no dedicated onboarding
    smoke landed): re-derived 2026-09-14 from 29 completed manifests since 2026-09-01
    (`REE_assembly/evidence/experiments/*/runs/*/manifest.json`, `machine` field). ~0.03
    min/ep at 200 steps/ep (n=4 runs with a directly comparable config; range
    0.009-0.06 min/ep across those 4 -- real per-experiment compute-cost variance, not
    measurement noise, so treat as a rough basis rather than a clean smoke figure).
  - **EWIN-PC** — zero runs recorded anywhere in the evidence corpus as of 2026-09-14
    (not just since 2026-09-01): estimate as cloud-1 until a smoke lands.
  - Add ~20% overhead for scripts with stratified replay buffers or event classification
- Set `machine_affinity` to match compute profile: `"DLAPTOP"` (macbook, online stepping), `"Daniel-PC"` (replay/batch heavy or long overnight runs), `"ree-cloud-1"` / `"ree-cloud-2"` (CPU-only Hetzner CX22, standard/env-heavy), `"EWIN-PC"` (GPU-capable, Eoin's machine), `"any"` (indifferent -- any cloud worker that's already awake will typically claim first)
  - **IMPORTANT:** The runner matches affinity through `machine_identity.same_machine()` (see `machine_identity.py`'s module docstring), NOT raw `socket.gethostname()` equality — this closed a real bug where macOS LocalHostName suffix drift (`DLAPTOP-4.local` <-> `DLAPTOP-5.local`) silently split the Mac's identity in two. `"DLAPTOP"` is the canonical affinity string to use in new queue entries; `"DLAPTOP-4.local"`/`"DLAPTOP-5.local"` still match (they alias forward to `DLAPTOP`), but do NOT use `"macbook"` or any other unlisted string — only names in `validate_queue.py`'s `VALID_AFFINITIES` resolve to a real machine.
- Always queue experiments immediately after writing the script.

## Experiment IDs and Versioning

V3 experiments: V3-EXQ-001 onward.

**Labeling rule (see also REE_Working/CLAUDE.md "EXQ Versioning and Supersession Policy"):**
- Bug fix / minor implementation tweak to same hypothesis: append next letter (EXQ-047a, 047b, ... 047j).
- New hypothesis / major redesign: new number (EXQ-048).
- Never re-use an ID that was previously run (see "Troubleshooting Runner" below for the mechanism).

**Supersession:** when a lettered iteration corrects a bug that invalidated the predecessor's evidence, add `"supersedes": "V3-EXQ-047i"` to the new queue entry. After the run completes, set `evidence_direction: "superseded"` on the old manifest and rebuild the index (governance pipeline). This prevents buggy experiments from continuing to weight claim confidence scores.

**Queue validation:** `validate_queue.py` is called automatically at runner startup. Run it manually after any queue edit: `/opt/local/bin/python3 validate_queue.py`
