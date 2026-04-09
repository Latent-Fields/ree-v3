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
- SD-022: body.directional_limb_damage -- IMPLEMENTED 2026-04-09.
  CausalGridWorldV2 (ree_core/environment/causal_grid_world.py): 4-directional limb_damage[4]
  state; accumulates when moving through hazards; heals at heal_rate=0.002/step; movement
  failure P(fail) = damage[d] * failure_prob_scale.
  harm_obs_a re-sourced from body damage state (7 dims: damage[4]+max+mean+residual_pain)
  when limb_damage_enabled=True, replacing 50-dim proximity EMA.
  body_state extended 12->17 dims (+ damage[4] + residual_pain).
  Config: REEConfig.from_dims() params: limb_damage_enabled (False default),
  damage_increment (0.15), failure_prob_scale (0.3), heal_rate (0.002).
  When enabled: body_obs_dim=17, harm_obs_a_dim=7.
  Backward compatible: disabled by default; existing experiments unaffected.
  Biological basis: A-delta/C-fiber distinction. Directional limb damage provides causal
  independence (r2_s_to_a=0.996 ceiling confirmed structural by EXQ-241b).
  MECH-094: not applicable (waking observation stream).
  Validation experiment: V3-EXQ-318 queued.
  See SD-011, SD-022, ARC-030, MECH-112, Q-034, ARC-052.
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

## SD-018: Resource Proximity Supervision (2026-04-07)
- SD-018: encoder.resource_proximity_supervision — IMPLEMENTED 2026-04-07.
  Auxiliary Sigmoid regression head on z_world predicting max(resource_field_view)
  in [0,1]. MSE loss backprops through encoder, forcing z_world to represent resource
  proximity. Without this, benefit_eval_head produces R2=-0.004 (EXQ-085m) and the
  entire benefit/goal pathway (goal_proximity, z_goal seeding, drive modulation,
  dual systems) operates on noise. This is the benefit-side analog of SD-009.
  Config: use_resource_proximity_head (bool, default False),
  resource_proximity_weight (float, default 0.5).
  Agent: compute_resource_proximity_loss(target, latent_state) -> MSE.
  SplitEncoder: resource_proximity_head = Linear(world_dim, 1) + Sigmoid.
  LatentState: resource_prox_pred field. All backward-compatible (disabled default).
  EXQ-257 queued: WITH vs WITHOUT ablation pair, 3 seeds, phased training.
  ALL new benefit/goal experiments MUST set use_resource_proximity_head=True.

## SD-011 Second Source: Harm History Input (2026-04-08)
- SD-011 second source: harm_stream.affective_harm_history_input -- IMPLEMENTED 2026-04-08.
  AffectiveHarmEncoder (latent/stack.py) extended with harm_history input: rolling FIFO
  of past harm_exposure scalars from CausalGridWorldV2. Encoder input grows from
  harm_obs_a_dim to harm_obs_a_dim + harm_history_len when harm_history_len > 0.
  Auxiliary harm_accum_head (Linear+Sigmoid) predicts accumulated harm scalar, forcing
  z_harm_a to integrate temporal information that z_harm_s does not receive. This
  resolves the monotone redundancy confirmed by EXQ-241 (D3 reversal: z_harm_a predicted
  sensory target better than z_harm_s because both received the same spatial signal).
  Config: LatentStackConfig.harm_history_len (int, default 0; set 10 to enable).
  LatentStackConfig.z_harm_a_aux_loss_weight (float, default 0.1).
  CausalGridWorldV2 harm_history_len param (mirrors config; default 0).
  Data flow: env step() -> _harm_history FIFO -> obs_dict["harm_history"] ->
  agent.sense(obs_harm_history=...) -> encode(harm_history=...) ->
  AffectiveHarmEncoder(harm_obs_a, harm_history) -> z_harm_a + harm_accum_pred.
  Agent method: compute_harm_accum_loss(accumulated_harm_target, latent_state) -> loss.
  LatentState: harm_accum_pred field (Optional[Tensor], None when disabled).
  Backward compatible: harm_history_len=0 by default; existing experiments unaffected.
  Encoder hidden dim increased from 32 to 64 (input dim grew from 50 to 60).
  Phased training recommended but not strictly required (aux target is env scalar).
  MECH-094: not applicable (waking observation stream, not replay content).
  Validation experiment: V3-EXQ-241a queued (2-condition ablation, 3 seeds, ~60 min).
  See SD-011, MECH-112, ARC-030, ARC-032, MECH-029, Q-034.

## MECH-120: SHY Synaptic Homeostasis Wiring (2026-04-08)
- MECH-120: sleep.sws_denoising_attractor_flattening -- WIRED 2026-04-08.
  E1DeepPredictor.shy_normalise() (e1_deep.py:283-304) was already implemented but
  not called from enter_sws_mode(). Now wired: enter_sws_mode() calls
  self.e1.shy_normalise(decay=self.config.shy_decay_rate) when shy_enabled=True.
  Config: REEConfig.shy_enabled (bool, default False), REEConfig.shy_decay_rate
  (float, default 0.85). Both wired through from_dims().
  Data flow: enter_sws_mode() -> shy_normalise(decay) -> context_memory.memory.data
  modified in-place (slot weights decayed toward slot-mean).
  Backward compatible: shy_enabled=False by default; existing experiments unaffected.
  No trainable parameters. No gradient flow (.data write). No phased training needed.
  Biological basis: Tononi & Cirelli SHY hypothesis (2006). decay=0.85 = ~15%
  reduction per cycle, consistent with SHY literature.
  MECH-094: not applicable (modifies existing weights, does not generate replay content).
  Validation experiment: EXQ-245a queued.
  See MECH-120, MECH-165 (downstream -- replay diversity requires SHY first).

## MECH-205: Surprise-Gated Replay Write Path Fix (2026-04-09)
- MECH-205: hippocampal.surprise_gated_generative_replay -- WRITE PATH FIXED 2026-04-09.
  Tier 1 implementation (2026-04-07) wired PE EMA tracking + VALENCE_SURPRISE write in
  agent.py update_residue(). EXQ-258 FAIL (P1: surprise_tag_populated=False) had two
  root causes: (1) experiment script checked nonexistent `_rbf_layer` attr (should be
  `rbf_field`); (2) pe_ema_alpha=0.1 tracked PE so fast that surprise stayed near zero.
  Fix: pe_ema_alpha moved from hardcoded 0.1 to config (default 0.02, ~50-step window).
  Added pe_surprise_threshold (default 0.001) gate before update_valence(). Added
  _surprise_write_count diagnostic counter + mech205_write_count metric.
  Config: REEConfig.pe_ema_alpha (float, default 0.02), REEConfig.pe_surprise_threshold
  (float, default 0.001). Both wired through from_dims().
  Data flow: E3.post_action_update() -> e3_metrics["prediction_error"] -> PE EMA ->
  surprise = max(0, pe_mag - pe_ema) -> [gate: > threshold] ->
  ResidueField.update_valence(z_world, VALENCE_SURPRISE, surprise).
  Backward compatible: surprise_gated_replay=False by default; write block never entered.
  No trainable parameters. No phased training needed.
  MECH-094: not applicable (waking observation stream, not replay content).
  Validation experiment: EXQ-258a queued.
  See MECH-205, INV-052 (indirect).

## MECH-216: E1 Predictive Wanting / Schema Readout (2026-04-09)
- MECH-216: e1_predictive_wanting -- IMPLEMENTED 2026-04-09.
  E1DeepPredictor.schema_readout_head (Linear(hidden_dim, 1) + Sigmoid) reads LSTM
  top-layer hidden state -> schema_salience [0,1]. Agent caches in _schema_salience
  via _e1_tick(), seeds VALENCE_WANTING when > threshold via update_schema_wanting().
  Zhang/Berridge: W_m = kappa (drive_level) x V_hat (schema_salience).
  Config: E1Config.schema_wanting_enabled (False default), REEConfig.schema_wanting_threshold
  (0.3), schema_wanting_gain (0.5). Training: compute_schema_readout_loss(resource_proximity_target).
  Data flow: E1.predict_long_horizon() -> hidden[0][-1] -> schema_readout_head -> schema_salience
  -> agent._e1_tick() caches -> agent.update_schema_wanting(drive_level) -> ResidueField.update_valence(
  z_world, VALENCE_WANTING, sal * gain * drive).
  Backward compatible: schema_wanting_enabled=False by default; existing experiments unaffected.
  Literature: Berridge 2012 (incentive salience), Zhang et al 2009 (computational model),
  Gershman 2018 (successor representation), Garvert et al 2023 (spatio-predictive maps).
  Validation experiment: EXQ-263 queued (2-condition ablation, 3 seeds, ~100 min).
  See MECH-216, INV-065 (proxy goal necessity), ARC-051 (multi-level wanting hierarchy).

## SD-011/SD-012 E3 Integration (2026-04-05)
  z_harm_a now flows through the full agent loop into E3:
  - agent.sense(obs_harm_a=...) passes harm_obs_a to LatentStack.encode()
  - agent.select_action() extracts z_harm_a from LatentState, passes to E3.select()
  - E3Config.urgency_weight (default 0.0): z_harm_a.norm() lowers effective commit
    threshold (D2 avoidance escape). Capped by urgency_max (default 0.5).
  - E3Config.affective_harm_scale (default 0.0): amplifies lambda_ethical by
    (1 + affective_harm_scale * z_harm_a_norm). Accumulated threat -> higher M(zeta).
  - E3.compute_harm_forward_cost(): ResidualHarmForward-based trajectory scoring,
    replaces deprecated HarmBridge path. Rolls out z_harm_s step-by-step through
    trajectory actions and evaluates via harm_eval_z_harm_head.
  - Agent.compute_drive_level(obs_body) static method: canonical SD-012 formula
    drive_level = 1.0 - energy (obs_body[3]).
  All new parameters default to 0.0/None for full backward compatibility.
  EXQ-247 queued: full integration validation (4-arm ablation).

## SD Design Decisions Implemented (V3) — continued
- SD-009: encoder.event_contrastive_supervision — IMPLEMENTED (EXQ-020 PASS). z_world
  encoder event-type cross-entropy auxiliary loss. Reconstruction + E1-prediction losses
  are invariant to harm-relevance; supervised event discrimination forces z_world to
  represent hazard-vs-empty distinctions. See MECH-100.

## SD Design Decisions Implemented (V3) — continued
- SD-014: hippocampus.valence_vector_node_recording — IMPLEMENTED 2026-04-04.
  4-component valence vector V=[wanting, liking, harm_discriminative, surprise] added to
  RBFLayer and ResidueField (ree_core/residue/field.py). Each RBF center now stores a
  valence_vecs buffer [num_centers, 4] updated incrementally per visit.
  New methods: RBFLayer.evaluate_valence(z) -> [batch, 4]; ResidueField.update_valence(),
  evaluate_valence(), get_valence_priority(z_world, drive_state). VALENCE_WANTING=0,
  VALENCE_LIKING=1, VALENCE_HARM_DISCRIMINATIVE=2, VALENCE_SURPRISE=3 constants defined
  at module level. ResidueConfig.valence_enabled (default True; set False for ablation).
  MECH-094 gate applies: hypothesis_tag=True blocks valence updates. Prerequisite for
  ARC-036 (multidimensional valence map) and replay prioritisation via drive state.

- MECH-203 + MECH-204: neuromodulation.serotonergic_sleep_substrate — IMPLEMENTED 2026-04-07.
  SerotoninModule (ree_core/neuromodulation/serotonin.py) with SerotoninConfig.
  SR-1: tonic_5ht [0,1] state variable. Waking: rises on benefit, decays to baseline,
  suppressed by z_harm_a. SWS: held at waking level. REM: drops to 0 (dorsal raphe quiescence).
  SR-2: benefit_salience = tonic_5ht * benefit_exposure. Tags SD-014 VALENCE_WANTING for
  balanced replay prioritisation. SR-3: _precision_at_rem_entry captured on enter_rem().
  Dynamic GoalConfig modulation: z_goal_seeding_gain and valence_wanting_floor modulated
  by tonic_5ht each step. Agent methods: serotonin_step(), update_benefit_salience(),
  enter_sws_mode(), enter_rem_mode(), exit_sleep_mode(). HippocampalModule.replay() accepts
  optional drive_state for valence-weighted start selection. Master switch:
  tonic_5ht_enabled=False (default, fully backward compatible).

- ARC-028 + MECH-105: control_plane.hippocampal_betagate_coupling — IMPLEMENTED 2026-04-04.
  HippocampalModule.compute_completion_signal(trajectories) -> float: scores all proposed
  trajectories via _score_trajectory(), maps best score to sigmoid dopamine-analog value
  in [0.5, 1.0). Caches as self._last_completion_signal.
  BetaGate.receive_hippocampal_completion(signal) -> bool: if beta elevated and signal >=
  completion_release_threshold (default 0.75), calls self.release() and returns True.
  Implements Lisman & Grace 2005 subiculum->NAc->VP->VTA loop: high hippocampal completion
  quality -> dopamine signal -> beta drops -> E3 state propagates to action selection.
  get_state() and reset() updated. Return type of propose_trajectories() unchanged.

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

**V4 scope (full sleep mechanisms — not V3):**
- Full SWR consolidation pipeline (MECH-121 complete implementation)
- Slow-wave sleep prediction error baseline reset
- Sleep-dependent recalibration of commit thresholds (full SR-3/SR-4)
- Theta-gamma coupling during offline replay for memory formation
- Lansink et al. (2009) hippocampus-leads-striatum replay is V4 evidence
- Phase boundary triggers (SR-4: sws_consolidation_complete -> REM transition)

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
  - **EWIN-PC** — AMD Ryzen 7 8700F + RTX 5070 12GB (Eoin Golden's machine):
    - Throughput not yet benchmarked (original smoke errored 2026-04-06, -b pending)
    - Use `"EWIN-PC"` affinity string. GPU likely fast at larger world_dim.
  - Add ~20% overhead for scripts with stratified replay buffers or event classification
- Set `machine_affinity` to match compute profile: `"DLAPTOP-4.local"` (macbook, online stepping), `"Daniel-PC"` (replay/batch heavy or long overnight runs), `"ree-cloud-1"` (CPU-only, standard/env-heavy), `"EWIN-PC"` (GPU-capable, Eoin's machine), `"any"` (indifferent)
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

## ARC-033: E2_harm_s Forward Model (2026-04-09)
- ARC-033: harm_stream.sensory_discriminative_forward_model -- IMPLEMENTED 2026-04-09.
  E2HarmSForward in ree_core/predictors/e2_harm_s.py. f(z_harm_s_t, a_t) -> z_harm_s_pred_{t+1}.
  Wraps ResidualHarmForward (ree_core/latent/stack.py) -- residual delta architecture
  avoids identity collapse on autocorrelated z_harm_s signals (r~0.9).
  Config: E2HarmSConfig (standalone dataclass in e2_harm_s.py):
    use_e2_harm_s_forward (bool, default False), z_harm_dim (int, default 32),
    action_dim (int, default 4), hidden_dim (int, default 128),
    action_enc_dim (int, default 16), learning_rate (float, default 5e-4).
  LatentStackConfig.use_e2_harm_s_forward (bool, default False) added to config.py.
  REEConfig.from_dims() param: use_e2_harm_s_forward (default False).
  Data flow: HarmEncoder(harm_obs) -> z_harm_s + action_onehot -> E2HarmSForward -> z_harm_s_pred.
  SD-003 counterfactual pipeline:
    z_harm_s_cf = harm_fwd.counterfactual_forward(z_harm_s_t, a_cf)
    causal_sig  = E3.harm_eval_z_harm_head(z_harm_s_actual) - E3.harm_eval_z_harm_head(z_harm_s_cf)
  Backward compatible: disabled by default; existing experiments unaffected.
  Phased training required: YES (stop-gradient on z_harm_s inputs during P1).
    P0: HarmEncoder warmup (harm proximity supervision).
    P1: E2HarmSForward trains on frozen z_harm_s (z_b.detach(), z1_b.detach()).
    P2: Evaluation (forward_r2, harm_s_cf_gap).
  Biological basis: Keltner et al. (2006, J Neurosci) -- predictability suppresses
    sensory-discriminative (S1/S2) but not affective (ACC) pain responses.
    Forward model cancellation applies to z_harm_s (A-delta analog) not z_harm_a (C-fiber).
  MECH-094: not applicable (waking observation stream, not replay content).
  EXQ-195 evidence: harm_forward_r2=0.914 (forward model component working).
  Validation experiment: V3-EXQ-264 queued.
  Design doc: REE_assembly/docs/architecture/arc_033_e2_harm_s_forward_model.md
  See ARC-033, SD-003, SD-010, SD-011.

## SD-017: Minimal Sleep-Phase Infrastructure -- SWS/REM Passes (2026-04-09)
- SD-017: sleep_phase.minimal_sleep_infrastructure_v3 -- SWS-ANALOG + REM-ANALOG IMPLEMENTED 2026-04-09.
  Two new first-class methods added to REEAgent (ree_core/agent.py):
  (1) run_sws_schema_pass(): SWS-analog schema installation (hippocampus-to-cortex direction).
      Samples diverse z_world prototypes from _world_experience_buffer (stratified across
      buffer history), constructs [z_self, z_world] E1 input, writes to ContextMemory
      bypassing the offline gate (offline gate blocks waking obs; schema writes are
      intentional offline content). Returns: sws_n_writes, sws_slot_diversity (mean pairwise
      cosine distance of ContextMemory slots -- higher = more differentiated), sws_buffer_size.
  (2) run_rem_attribution_pass(): REM-analog attribution replay (slot-filling, MECH-166).
      Replays recent theta_buffer content via hippocampal.replay() (forward) and
      hippocampal.diverse_replay(mode="reverse") (reverse/ARC-045 bidirectional proxy).
      Evaluates residue terrain per trajectory without accumulating new residue
      (hypothesis_tag=True per MECH-094). Returns: rem_n_rollouts, rem_mean_harm_terrain,
      rem_terrain_variance, rem_n_reverse.
  (3) run_sleep_cycle(): Convenience method running SWS then REM in sequence with correct
      mode transitions (enter_sws_mode -> run_sws_schema_pass -> exit_sleep_mode ->
      enter_rem_mode -> run_rem_attribution_pass -> exit_sleep_mode). Returns merged metrics.
  Config (REEConfig, ree_core/utils/config.py):
      sws_enabled (bool, default False), sws_consolidation_steps (int, default 5),
      sws_schema_weight (float, default 0.1), rem_enabled (bool, default False),
      rem_attribution_steps (int, default 10). All wired through REEConfig.from_dims().
  Backward compatible: all switches default False; existing experiments unaffected.
  No trainable parameters. No gradient flow in pass bodies. No phased training needed.
  Prerequisites satisfied: MECH-092 (waking quiescent replay), MECH-120 SHY wiring
  (enter_sws_mode calls shy_normalise), serotonin module (MECH-203/204), enter_offline_mode.
  Distinguishes from EXQ-242: EXQ-242 used proxy hooks (standalone functions, non_contributory).
  This implementation adds first-class REEAgent methods experiments can call directly.
  MECH-094: hypothesis_tag=True in rem_attribution_pass (terrain scoring only; no residue writes).
  Validation experiment: V3-EXQ-265 queued (SD-017 activation + slot differentiation ablation,
  2 conditions x 3 seeds, ~45 min on Mac).
  See SD-017, ARC-045, MECH-166, MECH-120 (SHY gated within enter_sws_mode).
