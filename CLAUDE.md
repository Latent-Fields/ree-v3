# ree-v3

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

## SD Design Decisions Implemented
- SD-004: E2 action objects; HippocampalModule navigates action-object space O
- SD-005: z_gamma split into z_self (E2 domain) + z_world (E3/Hippocampal/ResidueField domain)
- SD-006: Asynchronous multi-rate loop execution (phase 1: time-multiplexed)

## SD Design Decisions Implemented (V3) — continued
- SD-007: encoder.perspective_corrected_world_latent — IMPLEMENTED 2026-03-18, FIXED 2026-03-18.
  ReafferencePredictor in ree_core/latent/stack.py. Enabled via reafference_action_dim
  in LatentStackConfig (0=disabled default; set to action_dim to enable). Applied in
  LatentStack.encode(): z_world_corrected = z_world_raw - ReafferencePredictor(z_world_raw_prev, a_prev).
  MECH-101 fix: input is z_world_raw_prev (NOT z_self_prev). EXQ-027 run 1 showed R²=0.027
  with z_self inputs because cell content entering view dominates Δz_world_raw and is
  inaccessible from body state alone. z_world_raw_prev stored in LatentState and used
  as fallback in encode() (falls back to z_world if z_world_raw is None).
  Biological basis: MSTd receives visual optic flow (content-dependent) + efference copy.
  See MECH-098, MECH-101.

## SD Design Decisions Implemented (V3) — continued
- SD-010: harm_stream.nociceptive_separation — IMPLEMENTED. CausalGridWorldV2 emits
  harm_obs; HarmEncoder(harm_obs -> z_harm) trains on proximity labels; E3.harm_eval
  takes z_harm; SD-007 reafference does not apply to z_harm. EXQ-056c/058b PASS.
  SD-010 single-stream is a prerequisite for SD-011 (dual-stream extension).
- SD-011: harm_stream.dual_nociceptive_streams — IMPLEMENTED 2026-03-30.
  AffectiveHarmEncoder added to latent/stack.py; z_harm_a field added to LatentState;
  CausalGridWorldV2 emits harm_obs_a [50] (EMA at tau~20 steps). Validated EXQ-178b PASS.
  (1) z_harm_s: HarmEncoder(harm_obs) -> z_harm -- sensory-discriminative (A-delta analog).
  (2) z_harm_a: AffectiveHarmEncoder(harm_obs_a) -> z_harm_a -- affective-motivational
      (C-fiber analog, EMA-accumulated). NOT counterfactually modeled. Feeds E3 commit
      gating directly as motivational urgency (ARC-016 variance gating).
  E2_harm_s forward model (ARC-033) and SD-003 redesign to use z_harm_s for counterfactual
  attribution remain as next experiments (EXQ-195 queued). See ARC-033, SD-003 note.
- SD-008: encoder.z_world_alpha_correction — IMPLEMENTED in factory presets (alpha_world=0.9).
  LatentStackConfig default is 0.3 for backward compat; REEConfig.from_dims() default is
  0.9 (all experiment configs built via factory get the fix). Set alpha_world=0.9 or 1.0
  explicitly; set 0.3 only for ablation. Evidence: EXQ-013, EXQ-018, EXQ-019 (all failures
  confirmed 0.3 suppresses event responses). See MECH-100.
- SD-012: goal.homeostatic_drive_modulation — IMPLEMENTED 2026-04-02.
  GoalConfig.drive_weight changed from 0.0 to 2.0 (default). drive_weight=2.0 means
  effective_benefit = benefit_exposure * (1.0 + 2.0 * drive_level). With drive_level=1.0
  (fully depleted), a benefit_exposure of 0.04 becomes 0.12 -- above benefit_threshold=0.1.
  drive_weight added to REEConfig.from_dims() parameter list (overridable per experiment).
  Set drive_weight=0.0 explicitly for ablation baselines. EXQ-074e and EXQ-085 successors
  will benefit immediately. See GoalConfig, agent.py update_z_goal().

## SD Design Decisions Pending (V3)
- SD-009: encoder.event_contrastive_supervision — z_world encoder requires event-type
  cross-entropy auxiliary loss during training (MECH-100). Reconstruction + E1-prediction
  losses are invariant to harm-relevance; only supervised event discrimination forces
  z_world to represent hazard-vs-empty distinctions. See EXQ-020.

## SD Design Decisions Validated (V3) — 2026-03-18
- SD-003: self_attribution.counterfactual_e2_pipeline — VALIDATED EXQ-030b PASS
  (on z_world pipeline). REDESIGN IN PROGRESS for z_harm_s pipeline (SD-011/ARC-033).
  EXQ-030b pipeline: z_world_actual = E2.world_forward(z_world, a_actual),
  z_world_cf = E2.world_forward(z_world, a_cf), causal_sig = E3(z_world_actual) - E3(z_world_cf).
  Results: world_forward_r2=0.947, attribution_gap=0.035, correct sign structure.
  NOTE: EXQ-030b validated the counterfactual ARCHITECTURE before SD-010 wired E3 to
  take z_harm. Now that E3 operates on z_harm, the counterfactual must operate on the
  harm stream. EXQ-093/094 confirmed HarmBridge(z_world->z_harm) has bridge_r2=0
  (infeasible: z_world perp z_harm by SD-010 design). Redesigned pipeline (post SD-011):
    z_harm_s_cf = E2_harm_s(z_harm_s, a_cf)
    causal_sig = E3(z_harm_s_actual) - E3(z_harm_s_cf)
  E2_harm_s is a learnable forward model on the sensory-discriminative harm stream (ARC-033).
  DO NOT attempt HarmBridge counterfactuals -- bridge_r2=0 is architectural, not a bug.

## V3 / V4 Scope Boundary (2026-03-21)

**V3 scope (waking mechanisms):**
- Volatility interrupt / LC-NE analog (MECH-104): surprise-spike on running_variance
- BG hysteresis and outcome-valence modulation (MECH-106)
- Hippocampal→BG completion coupling (MECH-105, ARC-028)
- Beta gate committed→uncommitted dynamics (MECH-090)
- Trajectory completion signal from HippocampalModule (ARC-028)

**V4 scope (sleep mechanisms — not V3):**
- Sharp-wave ripple (SWR) consolidation of place-reward associations
- Slow-wave sleep prediction error baseline reset
- Sleep-dependent recalibration of commit thresholds
- Theta-gamma coupling during offline replay for memory formation
- Lansink et al. (2009) hippocampus-leads-striatum replay is V4 evidence
- Biological rationale: sleep mechanisms are evolutionary successors to waking
  decision architecture. V3 must deliver working waking circuit before V4 sleep.

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
  - Add ~20% overhead for scripts with stratified replay buffers or event classification
- Set `machine_affinity` to match compute profile: `"DLAPTOP-4.local"` (macbook, online stepping), `"Daniel-PC"` (replay/batch heavy or long overnight runs), `"any"` (indifferent)
  - **IMPORTANT:** The runner matches affinity against `socket.gethostname()` exactly. The macbook hostname is `DLAPTOP-4.local` — do NOT use `"macbook"` as the affinity string, it will not match.
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

## Troubleshooting Runner

**Runner log location**: `REE_assembly/runner.log` (NOT `ree-v3/runner.log`).
serve.py redirects runner stdout/stderr there. `ree-v3/runner.log` is only written when
the runner is started manually from the command line with `nohup ... > runner.log`.

**Runner says "No new items" despite pending items in queue**:
The runner skips any queue item whose `queue_id` already appears in `runner_status.json`
completed list. If an experiment was previously run (PASS/FAIL/ERROR) and then re-queued
with the same ID, the runner will silently skip it. Fix: rename the queue ID (e.g., append
`b`, `c`, etc.) before re-queueing.

**How this happens in practice (2026-03-23 incident):** Six experiments errored or failed,
were removed from the queue normally, then were re-queued by a subsequent session with the
same IDs to re-run them after script fixes or design tweaks. The runner had no way to
distinguish a re-run intent from a stale entry -- it only checks queue_id against the
completed list. Affected IDs: EXQ-075, EXQ-074b, EXQ-076, EXQ-084 (all ERROR exit 1),
EXQ-085 and EXQ-047g (FAIL). Fix was to rename to 075b, 074c, 076b, 084b, 085b, 047h.
Diagnosis: check `runner_status.json` completed list for the stuck queue IDs.

**Runner says "No new items" due to missing `title` field (2026-03-24 incident)**:
Queue items without a `title` field cause `run_experiment()` to crash with `KeyError: 'title'`
(the runner does a hard dict access at the "Starting:" log line). The UNEXPECTED ERROR handler
adds the item to in-memory `completed_ids` (not persisted to runner_status.json), so the
runner permanently skips it until restarted. Symptom: log shows "UNEXPECTED ERROR in EXQ-XXX:
'title'" once, then "No new items" forever.
Fix: add `"title": "..."` to the queue item, run `validate_queue.py`, then restart the runner.
Note: `title` is optional per schema but the runner required it -- fixed 2026-03-24 to use
`item.get('title', item['queue_id'])`. All new queue entries should still include a title.

**git pull fails with `fatal: bad object refs/remotes/origin/main 2`**:
Run `git remote prune origin` in ree-v3. This cleans up a spurious remote tracking ref.
Verify with `git fetch` (should return silently).
