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
- Preflight + contracts together:
  `python3 scripts/run_regression_suite.py --preflight && \
   python3 scripts/run_regression_suite.py --contracts`.

**Contracts test contracts, not thresholds.** If a test starts asserting a
specific magnitude or sign from an EXQ manifest, that belongs in an experiment
script, not the regression suite. The regression suite is the thing that has to
keep working when experiments and claim state evolve.

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

## MECH-090 Layer 1 + MECH-091 Layer 2: Trajectory Stepping + Urgency Interrupt (2026-04-15)
- MECH-090 Layer 1: control_plane.committed_trajectory_stepping -- IMPLEMENTED 2026-04-15.
  Module: ree_core/agent.py (REEAgent.select_action, REEAgent.reset).
  Config: E3Config.urgency_interrupt_threshold (float, default 0.8).
  Previously: select_action() always used committed_trajectory.actions[:, 0, :] (first
  action repeated every E3 tick during commitment). Now: _committed_step_idx counter steps
  through actions[:, idx, :] on each E3 tick. Counter clamped to (horizon-1) to guard
  against overflow. Reset on beta_gate.release() and agent.reset().
  Data flow: commit -> _committed_step_idx=0; each E3 tick in committed state ->
  action = committed_trajectory.actions[:, _committed_step_idx, :]; idx += 1 (clamped).
  Biological basis: committed motor sequences are unrolled action-by-action, not
  repeated. Striatal-thalamo-cortical propagation advances through the planned motor
  program at each execution step.
  MECH-094: not applicable (waking action selection, not simulation content).
  See MECH-090, ARC-028, MECH-091.

- MECH-091 Layer 2: control_plane.urgency_interrupt -- IMPLEMENTED 2026-04-15.
  Module: ree_core/agent.py (REEAgent.select_action).
  Config: E3Config.urgency_interrupt_threshold (float, default 0.8).
  When beta is elevated (committed state) and z_harm_a.norm() > urgency_interrupt_threshold:
  beta_gate.release() is called and _committed_step_idx is reset to 0, falling through to
  fresh E3 selection on the same tick.
  Data flow: select_action() -> [beta elevated?] -> z_harm_a.norm() -> [> threshold?] ->
  beta_gate.release(); _committed_step_idx=0 -> E3.select() with fresh state.
  Biological basis: unexpected nociceptive escalation (C-fiber burst) triggers STN -> GPe
  urgency signal, interrupting the committed motor program and returning to deliberative
  planning. Links SD-021 descending modulation (z_harm_s attenuated during commitment) with
  the escape mechanism: z_harm_a (affective load, not gated) drives the interrupt.
  Backward compatible: urgency_interrupt_threshold=0.8 (default); only fires when beta is
  elevated AND z_harm_a.norm() exceeds threshold. Existing experiments unaffected (most
  use default E3Config with urgency_weight=0.0 so z_harm_a.norm() stays low).
  MECH-094: not applicable (waking action selection gate).
  See MECH-091, MECH-090, SD-021, SD-011.

## SD-023: Environmental Gradient Texture (2026-04-09)
- SD-023: environment.gradient_texture -- IMPLEMENTED 2026-04-09.
  CausalGridWorldV2 (ree_core/environment/causal_grid_world.py): two new landmark object
  types (Landmark A: navigation anchor, Landmark B: predictive resource cue) each emitting
  a 25-dim gradient field view in world_obs.
  New __init__ params: n_landmarks_a (int, default 0), n_landmarks_b (int, default 0),
  landmark_a_sigma (float, default 1.5), landmark_a_scale (float, default 1.0),
  landmark_b_sigma (float, default 2.5), landmark_b_scale (float, default 0.6),
  landmark_b_resource_bias (float, default 0.7).
  Data flow: reset() -> _place_random_landmarks/_place_biased_near_resources ->
  _compute_landmark_field -> precomputed static Gaussian fields per episode ->
  _get_observation_dict() extracts 5x5 field views -> appended to world_state ->
  world_obs_dim: 250 -> 300 when n_landmarks_a>0 or n_landmarks_b>0.
  obs_dict keys: "landmark_a_field_view" [25], "landmark_b_field_view" [25].
  Backward compatible: n_landmarks_a=0, n_landmarks_b=0 by default; world_obs_dim=250;
  all existing experiments unaffected. Landmarks are gradient-only (no grid entity type).
  Biological basis: all objects in natural environments have a detectable presence
  (olfactory, acoustic, visual texture). Landmark B placed with bias near resources
  provides the predictive co-occurrence structure for MECH-216 (anticipatory wanting).
  No phased training required (env extension only, no new encoder or training target).
  MECH-094: not applicable (waking observation stream).
  Validation experiment: V3-EXQ-263b queued.
  See SD-023, MECH-216, ARC-017, MECH-096, MECH-103.

## SD-013, MECH-090, SD-015, SD-019, SD-020, SD-021: Harm Stream + Gate Implementations (2026-04-10)
- SD-013: self_attribution.e2_harm_s_interventional_training -- IMPLEMENTED 2026-04-10.
  Module: ree_core/predictors/e2_harm_s.py (E2HarmSConfig, E2HarmSForward).
  Config: E2HarmSConfig.use_interventional (bool, default False),
  interventional_fraction (float, default 0.3), interventional_margin (float, default 0.1).
  New method: E2HarmSForward.compute_interventional_loss(z_harm_s, a_actual, a_cf).
  Implementation: samples a_cf != a_actual; runs _residual_fwd for both; applies ReLU margin
  loss forcing ||z_pred_actual - z_pred_cf|| >= interventional_margin. Training applies the
  loss to interventional_fraction of each batch. Gradient backprops through E2_harm_s weights.
  Data flow: (z_harm_s, a_actual, a_cf) -> _residual_fwd x2 -> L2 dist -> margin loss.
  Backward compatible: use_interventional=False (default); existing experiments unaffected.
  Biological basis: Scholkopf et al. (2021) interventional distribution P(z | do(a)) vs
  observational P(z | a). In confounded states, observational training compresses causal_sig.
  Margin loss enforces identifiability: E2_harm_s must produce divergent outputs for different
  actions from the same state, regardless of ambient correlations.
  Phased training: P0 encoder warmup (use_interventional=False) -> P1 frozen-encoder head
  training (use_interventional=True, interventional_fraction=0.3).
  MECH-094: not applicable (waking forward model training, no simulation content).
  Validation experiment: V3-EXQ-320 queued.
  See SD-003, SD-011, ARC-033, SD-013.

- MECH-090: control_plane.commitment_gated_policy_output -- bistable latch IMPLEMENTED 2026-04-10.
  Module: ree_core/agent.py (REEAgent._e3_tick, REEAgent.select_action).
  Config: HeartbeatConfig.beta_gate_bistable (bool, default False).
  Previously: select_action() elevated beta on commit, released on no-commit (per-tick re-eval).
  Now (bistable=True): gate elevates on ENTRY to committed state only. Hippocampal completion
  signal is the primary release trigger. Wiring: _e3_tick() calls
  beta_gate.receive_hippocampal_completion(hippocampal.compute_completion_signal(candidates))
  when beta_gate_bistable=True and beta_gate.is_elevated.
  select_action() guards: if bistable -> elevate only on transition (committed AND NOT elevated);
  else -> legacy per-tick raise/release.
  Backward compatible: beta_gate_bistable=False (default); existing experiments unaffected.
  Biological basis: STN beta power is bistable during committed sequences -- it does not
  re-evaluate commitment each striatal cycle. Release is triggered by hippocampal completion
  signal (sequence end) or surprise interrupt (MECH-091).
  MECH-094: not applicable (gate state is a continuous control variable, not simulation).
  Validation experiment: V3-EXQ-321 queued.
  See MECH-090, ARC-028, MECH-105, HeartbeatConfig.

- SD-015: goal_representation.z_resource_encoder -- IMPLEMENTED 2026-04-10.
  Modules: ree_core/latent/stack.py (ResourceEncoder, LatentState), ree_core/agent.py
  (update_z_goal, compute_resource_encoder_loss).
  Config: LatentStackConfig.use_resource_encoder (bool, default False),
  LatentStackConfig.z_resource_dim (int, default 32 -- matches GoalConfig.goal_dim).
  New class ResourceEncoder: world_obs -> [hidden_dim=64] -> z_resource [32], plus a
  resource_prox_head (Linear -> Sigmoid) predicting resource proximity in [0,1].
  LatentState: added z_resource (Optional[Tensor]), resource_prox_pred_r (Optional[Tensor]).
  LatentState.detach() handles both new fields.
  LatentStack.encode(): when use_resource_encoder=True, runs ResourceEncoder on world_obs;
  z_resource and resource_prox_pred_r set in returned LatentState.
  agent.update_z_goal(): seeds GoalState from z_resource (not z_world) when
  use_resource_encoder=True and z_resource is not None.
  agent.compute_resource_encoder_loss(resource_proximity_target, latent_state): MSE of
  resource_prox_pred_r against proximity label; gradient flows through ResourceEncoder.
  Data flow: world_obs -> ResourceEncoder -> z_resource -> GoalState.update(z_resource) ->
  z_goal attractor. Auxiliary: resource_prox_pred_r -> MSE(target) -> backprop.
  Backward compatible: use_resource_encoder=False (default); z_resource=None in LatentState.
  Biological basis: MECH-112 (structured goal representation) requires a latent that encodes
  object-type features (what a resource IS) separate from spatial position (where z_world
  encodes the full scene). z_resource is location-invariant across resource respawns.
  Phased training required: P0 train ResourceEncoder with proximity labels; P1 freeze encoder,
  seed z_goal from z_resource, train E3 goal evaluation on the seeded representation.
  MECH-094: not applicable (waking encoder, not simulation content).
  Validation experiment: V3-EXQ-322 queued.
  See SD-015, SD-012, MECH-112, INV-065.
  Signal chain provenance: REE_assembly/docs/architecture/goal_wanting_signal_chain.md

- SD-019: harm_stream.affective_nonredundancy_constraint -- IMPLEMENTED 2026-04-10.
  Module: ree_core/agent.py (compute_harm_nonredundancy_loss).
  Config: REEConfig.harm_nonredundancy_weight (float, default 0.0),
  REEConfig.harm_nonredundancy_precision_scale (float, default 0.0).
  New agent method: compute_harm_nonredundancy_loss(latent_state).
  Implementation: projects z_harm_s and z_harm_a to a shared comparison dim via learned
  Linear projections (harm_dim -> harm_dim); computes cosine_similarity(z_s_proj, z_a_proj);
  penalty = cosine_sim^2. When harm_nonredundancy_precision_scale > 0.0, penalty is scaled
  by (1 + scale * e3.current_precision / 500.0), capped at 2x. Sum: weight * penalty.
  ARC-016 wiring: e3.current_precision (1/running_variance) used directly as a property.
  Data flow: (z_harm_s, z_harm_a) -> proj_s, proj_a -> cosine_sim^2 -> precision_scale ->
  loss_term -> add to total loss in training loop.
  Backward compatible: harm_nonredundancy_weight=0.0 (default) -> no penalty applied.
  Biological basis: C-fiber and A-delta projections have distinct laminar terminations and
  functionally non-overlapping representations. A valid dual-stream encoder must not collapse
  z_harm_a into a monotone transform of z_harm_s.
  ARC-016 relevance: higher precision (lower variance) implies a committed, confident state
  where the streams must encode genuinely distinct information.
  Phased training: enable after P0 warmup; apply loss during P1/P2 when encoders are
  discriminative enough for the penalty to be informative.
  MECH-094: not applicable (waking encoder loss, not simulation content).
  Validation experiment: V3-EXQ-323 queued.
  See SD-019, SD-011, ARC-016, MECH-219.

- SD-020: harm_stream.affective_surprise_pe -- IMPLEMENTED 2026-04-10.
  Module: ree_core/agent.py (compute_harm_accum_loss, __init__ adds self._harm_obs_ema).
  Config: REEConfig.harm_surprise_pe_enabled (bool, default False),
  REEConfig.harm_obs_ema_alpha (float, default 0.1).
  Implementation: when harm_surprise_pe_enabled=True, maintain EMA of observed
  accumulated_harm_target; on each step, harm_PE = |accumulated_harm_target - ema|; update
  ema <- (1-alpha)*ema + alpha*target; compute precision_norm = min(e3.current_precision/500, 3);
  training target becomes surprise_target = harm_PE * precision_norm.
  This replaces the raw accumulated_harm scalar as the aux loss target for AffectiveHarmEncoder.
  z_harm_a then encodes how SURPRISING the threat level is (unexpected escalation), not how
  high it is absolutely (Chen 2023, anterior insula as unsigned aversive PE detector).
  Data flow: accumulated_harm_target -> EMA predictor -> harm_PE -> precision_norm ->
  surprise_target -> MSE loss -> z_harm_a encoder.
  Backward compatible: harm_surprise_pe_enabled=False (default); legacy EMA path unchanged.
  Biological basis: anterior insula (AIC) encodes unsigned intensity prediction errors as a
  modality-unspecific aversive surprise signal, NOT raw magnitude (Chen 2023; Hoskin 2023;
  Geuter 2017; Horing 2022). SD-020 aligns z_harm_a with the AIC functional role.
  Phased training: P0 encoder warmup with harm_surprise_pe_enabled=False (raw target);
  P1 switch to harm_surprise_pe_enabled=True once EMA is calibrated (~50 episodes).
  ARC-016 wiring: e3.current_precision scales the surprise target, so high-confidence states
  produce stronger PE-weighted training signal.
  MECH-094: not applicable (waking training loop, not simulation content).
  Validation experiment: V3-EXQ-324 queued.
  See SD-020, SD-011, SD-019, ARC-016, Q-036.

- SD-021: harm_stream.descending_modulation -- IMPLEMENTED 2026-04-10.
  Module: ree_core/agent.py (REEAgent.sense).
  Config: REEConfig.harm_descending_mod_enabled (bool, default False),
  REEConfig.descending_attenuation_factor (float, default 0.5).
  Implementation: after LatentStack.encode(), if harm_descending_mod_enabled=True and
  beta_gate.is_elevated and new_latent.z_harm is not None:
    new_latent.z_harm = new_latent.z_harm * descending_attenuation_factor.
  z_harm_a (new_latent.z_harm_a) is NOT modified -- affective load persists through commitment.
  Data flow: encode() -> z_harm_s [, z_harm_a] -> [beta_gate elevated?] -> z_harm_s * factor
  -> E3 harm_eval, E2_harm_s forward model.
  Backward compatible: harm_descending_mod_enabled=False (default); existing experiments
  unaffected. descending_attenuation_factor default 0.5 is no-op when mod is disabled.
  Biological basis: pgACC -> PAG -> RVM descending inhibitory pathway. During volitional
  action through expected harm, A-delta (z_harm_s) nociceptive input is precision-downweighted.
  C-fiber (z_harm_a) affective load persists -- committed athletes feel motivational urgency
  but sensory discrimination is gated. MECH-090 BetaGate elevation is the committed-state gate.
  MECH-094: not applicable (waking sense path, not simulation content).
  Validation experiment: V3-EXQ-325 queued.
  See SD-021, SD-011, SD-020, MECH-090, ARC-016, MECH-220.

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
  Signal chain provenance: REE_assembly/docs/architecture/goal_wanting_signal_chain.md

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
  Write paths (2026-04-17):
    VALENCE_WANTING (0): update_benefit_salience() [serotonin salience] and
      update_schema_wanting() [E1 schema readout]. Both enabled when tonic_5ht_enabled or
      schema_wanting_enabled respectively.
    VALENCE_LIKING (1): update_liking(benefit_exposure) -- NEW 2026-04-17.
      Call from experiment loop at resource contact (benefit_exposure >= liking_threshold).
      Berridge hedonic impact at consummation (opioid-mediated). Enabled by
      valence_liking_enabled=True in REEConfig.from_dims().
    VALENCE_HARM_DISCRIMINATIVE (2): written automatically in sense() after SD-021
      descending modulation -- NEW 2026-04-17. Post-attenuation z_harm.norm() written at
      current z_world node. Committed-state nodes get stale (attenuated) h, creating the
      analgesia-as-underestimated-h signature for SD-021/SD-014 cross-connection.
      Enabled by valence_harm_enabled=True in REEConfig.from_dims().
    VALENCE_SURPRISE (3): written in update_residue() when MECH-205 surprise_gated_replay
      is active. PE-EMA delta written when magnitude exceeds pe_surprise_threshold.
  All four components now have active write paths. Config flags all default False (backward compat).

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

- MECH-290: hippocampal.backward_trajectory_credit_sweep -- IMPLEMENTED 2026-04-24.
  Module: ree_core/hippocampal/module.py (HippocampalModule.record_committed_trajectory,
  HippocampalModule.backward_credit_sweep, HippocampalModule.reset_committed_trajectory).
  Biological basis: Foster & Wilson 2006 (Nature) -- reverse replay fires at reward
  endpoint during waking, concurrent with dopamine. Credit propagates backward from goal
  to trajectory start.
  Two new methods:
    record_committed_trajectory(trajectory): called at BetaGate elevation (commit entry
      in select_action()), stores a detached copy of the committed trajectory in
      _committed_trajectory_buffer. Distinct from _exploration_buffer (MECH-165
      quiescent replay source): this stores EXECUTED trajectory, not CEM proposals.
    backward_credit_sweep(outcome_quality): called when BetaGate releases via
      receive_hippocampal_completion() in _e3_tick(). Sweeps committed trajectory
      backward; at each z_world state t: credit = outcome_quality * gamma^(T-1-t);
      ResidueField.update_valence(z_world_t, VALENCE_WANTING, credit) called.
      Returns dict: n_steps_swept, mean_credit, outcome_quality.
      No-op when outcome_quality < backward_sweep_min_quality (default 0.6).
    reset_committed_trajectory(): called from agent.reset() on episode boundary.
  Config: HippocampalConfig.use_backward_credit_sweep (bool, default False),
    backward_sweep_gamma (float, default 0.9), backward_sweep_min_quality (float, 0.6).
    All wired through REEConfig.from_dims().
  Agent wiring:
    _e3_tick(): receive_hippocampal_completion() return value captured as `released`;
      when True and flag is on, hippocampal.backward_credit_sweep(
      hippocampal._last_completion_signal) is called.
    select_action(): at bistable commit entry AND legacy non-bistable new-commit:
      hippocampal.record_committed_trajectory(e3._committed_trajectory) called.
    reset(): hippocampal.reset_committed_trajectory() called when flag on.
  No SD-006 dependency: fires synchronously on waking path.
  MECH-094: waking path (hypothesis_tag=False) -- credit from real executed trajectory.
  Requires ResidueConfig.valence_enabled=True to write VALENCE_WANTING; silently skips
  valence write if disabled (backward compat).
  Backward compatible: use_backward_credit_sweep=False by default; all methods are no-ops.
  Smoke: C1-C7 PASS (buffer management, sweep arithmetic, flag OFF no-op, valence write).
  End-to-end: agent boot + direct wiring test PASS 2026-04-24.
  Validation experiment: to be queued post-476a (SD-038 anti-recency sequenced after).
  See MECH-290, ARC-028, MECH-105, SD-014 (VALENCE_WANTING write paths), MECH-165.

## SD Design Decisions Validated (V3) — 2026-03-18
- SD-003: self_attribution.counterfactual_e2_pipeline — **SUPERSEDED 2026-04-18** after
  28 accumulated FAILs across the two-pass counterfactual architecture. Successor layer:
  MECH-256 (general single-pass forward-model comparator, stream-agnostic) + SD-029
  (concrete z_harm_s instantiation; event-conditioned test queued as V3-EXQ-433) + MECH-257
  (dual-function single-substrate E2: comparator vs evaluator, controller-gated). Per-stream
  successors SD-030 (z_self) and SD-031 (z_world) are V4-deferred. Architecture doc:
  `REE_assembly/docs/architecture/self_attribution_per_stream.md`. The EXQ-030b world-pipeline
  PASS (world_forward_r2=0.947, attribution_gap=0.035) is preserved as historical evidence but
  does not transfer to the z_harm_s topology.
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
- Set `machine_affinity` to match compute profile: `"DLAPTOP-4.local"` (macbook, online stepping), `"Daniel-PC"` (replay/batch heavy or long overnight runs), `"ree-cloud-1"` / `"ree-cloud-2"` (CPU-only Hetzner CX22, standard/env-heavy), `"EWIN-PC"` (GPU-capable, Eoin's machine), `"any"` (indifferent -- any cloud worker that's already awake will typically claim first)
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

## Remote Control (--remote-control flag)

When started with `--remote-control`, the runner emits a per-machine heartbeat each loop tick to `REE_assembly/evidence/experiments/runner_heartbeats/<hostname>.json` and processes pending commands from `runner_commands/<hostname>.json`. Default-off; bit-identical when omitted. Helper module: `runner_remote_control.py` (sibling of `experiment_runner.py`).

**Active-claim protection (added 2026-05-01)**: `push_heartbeat()` and `push_commands()` start each tick with `git pull --rebase --autostash` against REE_assembly. To prevent the autostash cycle silently reverting concurrent Claude sessions' uncommitted edits to `evidence/experiments/**`, both functions now bail at the top via `_active_claim_on_evidence_dir()` whenever any TASK_CLAIMS.json entry tagged `"active"` lists a resource path containing `evidence/experiments/`. Heartbeats/commands are best-effort; the next tick after the claim closes resumes pushing. Real-world incident that motivated this: 5 EXQ-232 ARC-026 supersession edits made on 2026-04-29 were silently reverted to original-commit content by 2026-05-01 with no trace in git history (no stash, no orphan commit) -- the autostash mechanism in the per-minute heartbeat was the culprit.

Six command kinds: `stop` (graceful drain), `force_stop` (SIGKILL current proc + exit), `pause` / `resume` (skip new experiments), `kick:<EXQ>` (move to head of queue), `release_claim:<EXQ>` (clear stuck `claimed_by`). `start` is intentionally not in this channel (a stopped runner cannot read its own command file) — use `/api/runner/v3/start` locally or SSH for remote.

When developing the runner: command processing happens at the **top of each pass** in the main `while True:` loop (before the experiment-picking `for item in items:` loop) so `pause` / `stop` / `kick` / `release_claim` take effect before the next claim attempt. Heartbeat write happens at the **bottom**, just before `time.sleep(args.loop_interval)`, with state in `{starting, idle, paused, draining}`.

Multi-machine dashboard: `/machines` in serve.py. POST `/api/machines/<host>/command {kind, args}` to enqueue commands. Trust model: GitHub push access = command-issue access.

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

## SD-016: Frontal Cue-Indexed Integration (2026-04-16)
- SD-016: e1.frontal_cue_indexed_integration -- IMPLEMENTED 2026-04-16.
  Module: ree_core/predictors/e1_deep.py (E1DeepPredictor).
  Three new projections gated by sd016_enabled=True:
    world_query_proj: Linear(world_dim=32, hidden_dim=128) -- z_world-only ContextMemory query
    cue_action_proj:  Linear(latent_dim=64, action_object_dim=16) -- affordance bias for E2
    cue_terrain_proj: Linear(latent_dim=64, 2) -- (w_harm, w_goal) terrain precision weights for E3
  Entry point: E1DeepPredictor.extract_cue_context(z_world) -> (action_bias, terrain_weight).
  Config: E1Config.sd016_enabled (default False; backward compatible).
  Data flow: z_world -> world_query_proj -> ContextMemory attention -> cue_action_proj (affordance)
             and cue_terrain_proj (terrain precision). terrain_weight passed to E3; action_bias to E2.
  Training for cue_terrain_proj: supervised terrain_loss using hazard_field_view proxy (lambda=0.1).
    terrain_loss must be included in experiment E1 training loops to train this projection.
    Pattern: see EXQ-182, EXQ-187a, EXQ-194. Omitting terrain_loss leaves cue_terrain_proj random.
  Training for cue_action_proj: UNRESOLVED (diagnostic open -- see EXP-0155). The original
    claim "implicit via E3 trajectory selection gradient (no new loss)" is DEMONSTRABLY FALSE.
    V3-EXQ-449 (diagnostic probe, 2026-04-20) confirmed cue_action_proj.weight receives
    exactly 0.0 gradient under this path (C1 PASS, 2 seeds, ~1.7k steps) because the CEM
    argmax in HippocampalModule is non-differentiable and agent.py:694 detaches action_bias
    before rollouts. EXQ-449 C2 arm added a supervised MSE loss against
    E2.action_object(z_world, a_executed).detach(): weights trained (grad ~0.013, delta
    ~0.21) but action_bias_divergence stayed at exactly 0.0 in both seeds -- something
    downstream of cue_action_proj zeroes the signal before it reaches E3.select. The simple
    supervision fix is insufficient on its own. EXP-0155 queued to instrument the full
    forward path (extract_cue_context -> cue_action_proj -> ... -> E3.select) and identify
    the specific blocker before any EXQ-418b successor is written. Until EXP-0155
    resolves, cue_action_proj must be treated as CURRENTLY UNGROUNDED: sd016_enabled=True
    experiments should expect action_bias_divergence ~= 0.0 and should not rely on
    cue_action_proj for behavioural effects. (cue_terrain_proj path remains valid --
    trained via terrain_loss.)
  Backward compatible: sd016_enabled=False by default; existing experiments unaffected.
  MECH-094: not applicable (waking encoder query, not replay content).
  Validation experiment: V3-EXQ-418a queued (SD-016+SD-017 combined retest with terrain_loss).
    V3-EXQ-418/418a/418b have all FAILed with action_bias_divergence=0.0; the EXQ-418b
    successor is GATED on EXP-0155 diagnostic resolution.
  See MECH-150, MECH-151, MECH-152, ARC-041, INV-040, EXP-0155 (cue_action_proj diagnostic).
  Design doc: REE_assembly/docs/architecture/sd_016_frontal_cue_integration.md

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

## SD-032b / MECH-258 / MECH-260 / ARC-058: dACC-analog Adaptive Control (2026-04-19)
- SD-032b: cingulate.dacc_analog_adaptive_control -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/dacc.py (DACCAdaptiveControl, DACCConfig,
  DACCtoE3Adapter). First substrate in SD-032 cingulate cluster; resolves
  EXQ-395 MECH-220 z_harm_a wiring gap.
  Config: REEConfig.use_dacc (bool, default False). Sub-weights:
  dacc_weight (overall gain on E3 score_bias), dacc_interaction_weight
  (Croxson 2009 payoff x effort interaction), dacc_foraging_weight
  (Kolling 2015 switch-value), dacc_suppression_weight (MECH-260 recency
  bias suppression), dacc_suppression_memory (FIFO depth, default 8),
  dacc_precision_scale (PE precision normaliser, default 500),
  dacc_effort_cost (Shenhav 2013 EVC cost, default 0.1),
  dacc_drive_coupling (SD-012 hook, default 0).
  Bundle output (per Croxson/Shenhav/Kolling integration, NOT a scalar):
  {mode_ev[K], choice_difficulty, foraging_value, harm_interaction[K],
  suppression[K], pe, drive_gain}.
  DACCtoE3Adapter (stopgap): converts bundle to score_bias[K] passed to
  E3.select() via new score_bias param on E3Selector.select(). MARKED
  FOR REPLACEMENT when SD-032a salience-network coordinator lands --
  coordinator is the architectural consumer of the bundle per SD-032
  design; the adapter is a shim to route the bundle to E3 in its
  absence.
  Data flow: sense() caches z_harm_a_prev -> select_action() builds
  payoffs (from last E3 scores), effort (trajectory horizons),
  action_classes (argmax of first action) -> DACCAdaptiveControl.forward()
  -> bundle -> DACCtoE3Adapter.forward(bundle) -> score_bias -> e3.select()
  -> post-step: E2_harm_a prediction for next tick (no_grad) +
  dacc.record_action(action_class).
  Backward compatible: use_dacc=False by default; existing experiments
  unaffected. All sub-weights default 0.0/0 (no-op). Non-default flags:
  use_e2_harm_a, use_shared_harm_trunk (ARC-058 path selection).
  Phased training required for E2_harm_a (see MECH-258 entry).
  Biological basis: Shackman 2011 (dACC integration hub); Baliki 2010
  (ACC-NAc pathway for affective pain to action value); Shenhav 2013
  EVC (mode_ev = payoff - control_required * effort_cost); Croxson 2009
  (reward x effort interaction); Kolling 2015 (foraging value as dACC
  switch signal); Scholl 2017 (neuromodulator-tunable coupling via
  drive_level).
  MECH-094: not applicable (waking action selection, no simulation).
  Validation experiment: V3-EXQ-445 queued (3-arm ablation:
  dACC-OFF vs dACC-ON-independent-E2_harm_a vs dACC-ON-shared-trunk).
  See SD-032b, MECH-258, MECH-260, ARC-058, ARC-033, SD-032 parent.

- MECH-258: cingulate.precision_weighted_pain_PE -- IMPLEMENTED 2026-04-19.
  Module: ree_core/predictors/e2_harm_a.py (E2HarmAConfig, E2HarmAForward).
  Structurally mirrors E2HarmSForward (ARC-033). Two constructor paths:
    (a) shared_trunk=None (default): owns independent ResidualHarmForward
        -- ARC-033-parallel, independent per-stream forward model.
    (b) shared_trunk=<HarmForwardTrunk>: reuses trunk, owns only
        HarmForwardHead -- ARC-058 shared-substrate path.
  Precision weighting (in DACCAdaptiveControl): pe = ||z_harm_a -
  E2_harm_a(z_harm_a_prev, a_prev)||; pe_weighted = pe * (1 +
  min(precision/dacc_precision_scale, 3)). bundle["pe"] drives the
  dACC adaptive-control magnitude via Shenhav 2013 EVC form.
  Config: REEConfig.use_e2_harm_a (bool, default False), e2_harm_a_lr
  (default 5e-4), use_shared_harm_trunk (bool, default False -- selects
  ARC-033 independent-per-stream vs ARC-058 shared-trunk path).
  Phased training (REQUIRED): P0 AffectiveHarmEncoder warmup (SD-011
  second source: harm_history input + SD-020 surprise-PE target);
  P1 E2_harm_a trains on FROZEN z_harm_a (caller MUST .detach() targets);
  P2 eval harm_a_forward_r2. Joint training with encoder causes head
  collapse (see EXQ-166b/c/d).
  Biological basis: Seymour 2019 pain-as-precision-signal; Chen 2023,
  Hoskin 2023, Geuter 2017 (AIC unsigned aversive PE); Keltner 2006
  (affective pain does not show predictive cancellation at subjective
  report, but PE substrate exists and is used for control demand).
  MECH-094: not applicable (waking forward model, not replay content).
  Consumed by: DACCAdaptiveControl bundle (SD-032b).
  Validation experiment: V3-EXQ-445 queued.
  See MECH-258, ARC-033, ARC-058, SD-020, SD-011.

- MECH-260: cingulate.dacc_bias_suppression -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/dacc.py (DACCAdaptiveControl maintains
  FIFO _action_history of recently-executed action classes).
  Computation: suppression[i] = count(action_class_i in history) /
  len(history). DACCtoE3Adapter adds dacc_suppression_weight *
  suppression to E3 score_bias (positive bias = unfavourable under
  E3's lower-is-better convention -- suppresses re-selection of
  recently-executed action classes).
  Config: REEConfig.dacc_suppression_weight (float, default 0.0),
  dacc_suppression_memory (int, default 8).
  Agent wiring: REEAgent.select_action() calls
  self.dacc.record_action(argmax(action[0])) after action is emitted.
  Backward compatible: suppression_weight=0 (default) -> no suppression.
  Biological basis: Scholl, Kolling et al 2015 (dACC + lateral aPFC
  actively suppress vmPFC/amygdala bias toward recently-rewarded
  choices). Target behavioural signature: fishtank_viz monostrategy
  ablation.
  Validation experiment: V3-EXQ-445 includes suppression_weight=0 vs
  suppression_weight=0.5 comparison.
  See MECH-260, SD-032b.

## SD-032a / MECH-259 / MECH-261: Salience-Network Coordinator (2026-04-19)
- SD-032a: cingulate.salience_network_coordinator -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/salience_coordinator.py
  (SalienceCoordinator, SalienceCoordinatorConfig, DEFAULT_MODE_NAMES,
  DEFAULT_GATE_WEIGHTS). Network-level coordinator that aggregates the
  SD-032b dACC bundle and homeostatic / offline signals into a soft
  operating-mode probability vector and a discrete MECH-259 mode-switch
  trigger. Hosts the MECH-261 dict-keyed write-gate registry.
  Inputs (live in V3): dACC bundle (pe / foraging_value /
  choice_difficulty), drive_level (SD-012; proxy SD-032c), agent
  offline-mode flag (proxy SD-032d). Registered slots aic_salience /
  pcc_stability / pacc_autonomic accept update_signal calls and remain
  no-op until SD-032c/d/e land.
  Outputs: operating_mode dict[str, float] (softmax over per-mode
  affinity logits, default biased to external_task waking baseline);
  current_mode str (Schmitt-trigger hysteresis -- updates only on
  threshold crossing); mode_switch_trigger bool (fires when
  salience_aggregate > switch_threshold * (1 + stability_scaling *
  pcc_stability) AND argmax(operating_mode) != current_mode);
  write_gate(target) float (soft-weighted sum over mode probs).
  MECH-261 default registry covers sd_033a, sd_033b, sd_033c, sd_033d,
  hc_viability, sensory_buffer, autonomic, e3_policy with the per-mode
  weights from the spec table. mode_names is a list, register_target
  accepts arbitrary mode keys -- V4 parallel_goal_deliberation
  (SD-033e) can be added without schema changes.
  Config: REEConfig.use_salience_coordinator (bool, default False).
  Sub-knobs: salience_switch_threshold (1.0), salience_stability_scaling
  (1.0), salience_softmax_temperature (1.0),
  salience_external_task_bias (1.0), salience_dacc_pe_weight (1.0),
  salience_dacc_foraging_weight (0.5), salience_apply_to_dacc_bias
  (False -- when True, scales dACC score_bias by the e3_policy gate so
  internal_replay attenuates dACC influence on action selection).
  Data flow: select_action() builds dACC bundle -> coordinator.tick()
  consumes bundle + drive_level + e1._offline_mode -> caches operating_mode
  + trigger -> optional scale of dacc_score_bias by write_gate("e3_policy")
  -> e3.select() unchanged path.
  Backward compatible: use_salience_coordinator=False by default. Existing
  experiments unaffected. DACCtoE3Adapter is RETAINED as the score_bias
  source until SD-033 substrates consume operating_mode natively (staged
  removal -- adapter shim is now optionally gated rather than fully
  replaced this PR).
  Biological basis: Menon & Uddin 2010 (AIC-dACC salience network);
  Craig 2009 (AIC interoceptive-salience hub); Carr/Jadhav/Frank 2011
  (soft-boundary write subpopulations during awake SWRs); Tambini &
  Davachi 2019 (cross-state persistence, forward propagation bias).
  MECH-094: not authored here -- coordinator emits the gate that MECH-094
  generalises to. Phased training: not applicable (non-trainable
  arithmetic).
  Validation experiment: V3-EXQ-446 queued (coordinator-OFF vs
  coordinator-ON, plus synthetic high-PE injection to confirm trigger
  fires; verifies write_gate values in [0, 1] across 8 default targets).
  See SD-032a, MECH-259, MECH-261, SD-032 parent.

- MECH-266: cingulate.asymmetric_per_mode_hysteresis -- IMPLEMENTED 2026-04-21.
  Module: ree_core/cingulate/salience_coordinator.py.
  Per-mode Schmitt-trigger rails on top of the MECH-259 symmetric
  switch_threshold. Two optional dict overrides on
  SalienceCoordinatorConfig:
    enter_thresholds[target_mode]: salience_aggregate required to enter
      target_mode (falls back to switch_threshold when unset).
    exit_thresholds[current_mode]: operating_mode[current_mode] must be
      strictly less than this value before a switch OUT of the current
      mode is permitted (falls back to 1.0 sentinel = always satisfied
      for any proper softmax, preserving legacy MECH-259 behaviour).
  MECH-266 trigger:
    trigger = (salience_aggregate > enter_threshold * stability_mult)
           AND (operating_mode[current_mode] < exit_threshold)
           AND (soft_argmax != current_mode)
  Over-binding / OCD axis: exit_thresholds[m] near 0 -> current mode
    must collapse to near-zero probability before leaving. Stuck-in-mode
    signature reproducible at exit=0.05.
  Under-binding / depression axis: set lower enter_threshold (e.g. 0.5)
    so salience clears entry rail more readily; exit left at 1.0 no-op.
  Symmetric baseline: empty dicts; trigger reduces to legacy MECH-259.
  Setters:
    set_enter_threshold(mode, value) -- per-mode enter rail.
    set_exit_threshold(mode, value)  -- per-mode exit rail.
    set_hysteresis_ratio(ratio)      -- uniform exit rail across all
      registered modes (EXP-0163 parametric sweep convenience).
  Tick return dict extended with enter_threshold, exit_threshold,
  current_mode_prob; effective_threshold retained as alias for
  enter_threshold (backward-compat diagnostic).
  Backward compatible: default SalienceCoordinatorConfig uses empty
  enter_thresholds / exit_thresholds dicts -- all existing experiments
  unaffected.
  Biological basis: Schmitt-trigger hysteresis is a canonical
  implementation of the per-mode asymmetric switch costs observed
  in task-switching paradigms (over-binding in OCD: hard to leave
  mode; under-binding in depression/ADHD axis: easy to flip). ocd4
  thought file row "competing goals" and "mode stickiness / Hold
  decay" derive from this substrate.
  MECH-094: not applicable (non-trainable arithmetic extension).
  Phased training: none (no parameters).
  Validation experiments: V3-EXQ-464 (EXP-0160 competing-goals, 5
    sub-tests, substrate-landing diagnostic) and V3-EXQ-467 (EXP-0163
    mode stickiness / hold decay, 5-arm parametric sweep r in
    [0.10, 0.50, 1.00, 1.50, 2.00]). Both smoke-PASS all sub-tests.
    Full behavioural competing-goals runs (switch-cost asymmetry,
    goal-completion dose-response) deferred to EXQ-464b / EXQ-467b
    when the CausalGridWorldV2 dual simultaneously active
    resource-cue extension lands.
  See MECH-266, SD-032a, MECH-259, SD-033 parent, REE_assembly
  evidence/planning/sd033_governance_plan.md, docs/thoughts/
  2026-04-20_ocd4.md.

- SD-032c: cingulate.aic_analog_salience_urgency -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/aic_analog.py (AICAnalog, AICConfig).
  Anterior-insula-analog interoceptive-salience / urgency-interrupt module.
  NOT the affective-pain consumer (that is SD-032b); this is the mode-switch
  trigger source AND the descending pain-modulation gate. Subsumes SD-021:
  the raw beta_gate.is_elevated check in agent.sense() is replaced by a
  drive-aware, operating-mode-aware gain function.
  Inputs (per sense() tick):
    z_harm_a_norm  (SD-011 affective stream)
    drive_level    (SD-012 GoalState._last_drive_level)
    beta_gate_elevated (MECH-090 committed-state signal)
    operating_mode (SD-032a coordinator, previous tick; None -> treat
                    p_external_task=1.0, preserves SD-032c function even
                    without coordinator)
    extra_salient  (optional; unexpected z_goal drop, reward-surprise,
                    irreversibility; default no-op via aic_extra_weight=0)
  Outputs (stored on the module, cached in agent._aic_last_tick):
    aic_salience   -- fed to SalienceCoordinator.update_signal("aic_salience",
                      ...) BEFORE coordinator.tick() each select_action cycle
                      (drives MECH-259 urgency-trigger).
    harm_s_gain    -- multiplier on z_harm in sense(), replacing the raw
                      SD-021 beta_gate check when use_aic_analog=True.
                      harm_s_gain < 1.0 only when committed AND the agent is
                      not depleted (drive_protect=1.0 default).
    urgency_signal -- diagnostic threshold crossing on aic_salience.
  Computation:
    baseline <- (1-alpha)*baseline + alpha * z_harm_a_norm  (EMA interoceptive
                                                             baseline)
    urgency  = max(0, (z_harm_a_norm - baseline) / (baseline + eps))
    aic_salience = urgency * (1 + drive_coupling * drive_level)
                 + aic_extra_weight * sum(extra_salient)
    drive_protect = max(0, 1 - drive_protect_weight * drive_level)
    harm_s_gain = clip_[0,1] ( 1 - base_attenuation * p_external *
                               float(beta_gate_elevated) * drive_protect )
  Config: REEConfig.use_aic_analog (bool, default False).
    Sub-knobs: aic_baseline_alpha (0.02, ~50-step window),
    aic_drive_coupling (1.0 -- MUST be non-zero for falsification signature),
    aic_urgency_threshold (1.0, diagnostic only),
    aic_base_attenuation (0.5, matches legacy descending_attenuation_factor),
    aic_drive_protect_weight (1.0; alterable-configuration knob flagged by
                              SD-032c spec: +1 preserve depleted signal,
                              0 drive-independent, -1 opposite-sign),
    aic_extra_weight (0.0, reserved for extra salient-event signals).
  Falsification signature (spec): same z_harm_a -> different mode-switch
    behaviour in depleted vs well-resourced agents. Both aic_salience AND
    harm_s_gain depend on drive_level -- this is the ONLY V3 substrate that
    makes the dependence structural. EXQ-325a FAIL (DESCENDING ==
    CONTROL bit-identical under raw beta_gate check) resolves when the AIC
    path replaces the raw check -- the descending branch becomes a
    genuinely different function of state.
  Data flow: encode() -> z_harm_a, z_harm -> aic.tick(z_harm_a_norm,
    drive_level, beta_gate_elevated, operating_mode_prev) -> aic_salience
    cached + harm_s_gain applied to z_harm if harm_descending_mod_enabled.
    select_action() injects aic_salience into coordinator via
    update_signal("aic_salience", ...) BEFORE coordinator.tick() so MECH-259
    trigger sees it on the current cycle. One-step lag on operating_mode
    read is biologically plausible (AIC->dACC->SAL is a circuit).
  Backward compatible: use_aic_analog=False by default. Legacy SD-021 raw
    beta_gate check retained behind the same harm_descending_mod_enabled
    flag -- selected only when use_aic_analog=False. With both flags off,
    existing experiments unchanged. The old descending_attenuation_factor
    config is still consumed by the legacy path.
  Biological basis: Craig 2009 AIC as interoceptive-salience hub with
    autonomic and motor efferents; Menon & Uddin 2010 salience-network
    coupling; Basbaum 1984 + Keltner 2006 ACC/AIC -> PAG descending
    inhibitory pathway.
  MECH-094: not applicable (waking observation stream, not replay content).
  Phased training: not applicable (non-trainable arithmetic, single EMA).
  Validation experiment: V3-EXQ-325b queued (3-condition x 2-drive-regime
    retest of EXQ-325a; supersedes EXQ-325a; acceptance criteria include
    drive-dependence contrast which the prior metric could not measure).
  See SD-032c, SD-032a, SD-032b, SD-021, MECH-259, MECH-261, SD-032 parent.

- SD-032d: cingulate.pcc_analog_attention_partition -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/pcc_analog.py (PCCAnalog, PCCConfig).
  Posterior-cingulate-analog metastability scalar in [0, 1] that modulates
  the SD-032a MECH-259 effective_threshold. High pcc_stability -> coordinator
  resists mode transitions; low stability (depleted / no recent rest /
  failing task outcomes) -> transitions happen at lower salience. Does NOT
  trigger mode switches directly (that is SD-032c's job). Non-trainable
  arithmetic; no gradient flow.
  Inputs (per select_action tick):
    drive_level (SD-012 fatigue, [0, 1])
    success_ema (EMA over caller-supplied task-outcome scalars, neutral 0.5
                 baseline; experiments opt in via agent.note_task_outcome())
    steps_since_offline (cross-episode counter; reset only by
                         note_offline_entry() called from
                         agent.enter_offline_mode())
  Computation:
    offline_recency = min(1.0, steps_since_offline / window)
    stability = baseline + success_weight * (success_ema - 0.5)
              - fatigue_weight * drive_level
              - offline_weight * offline_recency
    stability = clip_[0,1](stability)
  Config: REEConfig.use_pcc_analog (bool, default False).
    Sub-knobs: pcc_success_alpha (0.02, ~50-step EMA window),
    pcc_success_weight (0.5; centred contribution from success_ema),
    pcc_fatigue_weight (0.5; subtractive from drive_level),
    pcc_offline_recency_window (500 steps; saturation),
    pcc_offline_weight (0.3; subtractive from offline_recency),
    pcc_stability_baseline (0.5; additive baseline before clipping).
  Falsification signature (spec): ablating SD-032d makes the SalienceCoordinator
    effective_threshold insensitive to fatigue / time-since-offline. Agent
    over-commits to external_task without rest-driven relaxation. PCC-ON ->
    drive_level rises -> stability falls -> effective_threshold falls ->
    mode_switch_trigger rate rises under matched salience input.
  Data flow: select_action() -> pcc.tick(drive_level=sal_drive) ->
    salience.update_signal("pcc_stability", pcc.pcc_stability) BEFORE
    coordinator.tick() -> coordinator.effective_threshold modulated.
    enter_offline_mode() -> pcc.note_offline_entry() (single integration
    point shared by MECH-092 within-session quiescence and INV-049
    cross-session sleep). reset() -> pcc.reset() (per-episode; preserves
    cross-episode _steps_since_offline). agent.note_task_outcome(value) ->
    pcc.note_task_outcome(value) feeds success EMA.
  Backward compatible: use_pcc_analog=False by default. Existing experiments
    unaffected. note_task_outcome() is a no-op when pcc is None.
  Biological basis: Leech & Sharp 2013 ("Arousal, Balance, Breadth") --
    PCC tracks the global stability of the current cognitive set vs the
    need to broaden attentional sampling. Treated conservatively: a
    [0, 1] metastability index that biases the threshold for any mode
    change without committing to attention-partition geometry. Frankland
    & Bontempi 2005 systems-consolidation framing: stability falls with
    time-since-last-offline, biasing the system toward requesting offline
    consolidation when held externally too long.
  MECH-094: not applicable (waking arithmetic, no replay content authored).
  Phased training: not applicable (non-trainable arithmetic).
  Validation experiment: V3-EXQ-447 queued (PCC-OFF vs PCC-ON x rest /
    no-rest contrast; acceptance criterion: with PCC-ON and matched dACC
    salience injection, mode-switch trigger rate is monotone in
    drive_level and time-since-offline; PCC-OFF rate is invariant).
  See SD-032d, SD-032a, MECH-259, MECH-261, INV-049, MECH-092, SD-032 parent.

- SD-032e: cingulate.pacc_autonomic_coupling -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/pacc_analog.py (PACCAnalog, PACCConfig).
  Perigenual / subgenual cingulate-analog slow-EMA autonomic write-back.
  Accumulates tanh-normalised z_harm_a magnitude into a bounded drive_bias
  that shifts the effective drive_level passed into GoalState.update(),
  SalienceCoordinator.tick(), SD-032c AICAnalog, SD-032d PCCAnalog, and
  dACC bundle composition. Architectural path for chronic-pain-like
  sensitisation (Baliki 2012) compressed into the V3 drive_level proxy.
  Non-trainable arithmetic; no gradient flow.
  Scoping (see REE_assembly/evidence/literature/
  targeted_review_pacc_autonomic_coupling_write_target/synthesis.md):
    (1) Write target: drive_level as first-pass proxy. Biologically
        tighter targets (valence-signed mood setpoint, fast autonomic
        effectors) do not have V3 substrates; documented simplification.
    (2) Timescale: slow EMA, default alpha=0.002 (pacc_drive_ema=0.998;
        half-life ~347 steps). Scoping synthesis called alpha>=0.005
        "fast end of biological plausibility" -- default is inside the
        envelope; long-horizon sensitisation studies should use
        alpha<=0.0005. Compresses two biological steps (Guo 2018 ACC
        mGluR5 LTP + ACC downstream influence) into one accumulator.
    (3) Offline decay: DEFAULT 0.0 (no decay). Non-zero instantiates a
        DISTINCT sleep-recalibration claim that would need its own
        literature pull -- hook exists so a future claim can wire in
        without another implementation pass.
  Inputs (per select_action tick):
    z_harm_a_norm  (SD-011 affective stream, current latent)
    write_gate     (SalienceCoordinator.write_gate("autonomic") from
                    previous tick; one-step lag, pACC->autonomic->
                    sensitisation is slow. Defaults to 1.0 when
                    salience coordinator is disabled so drift remains
                    observable under ablation.)
    hypothesis_tag (MECH-094 gate; select_action passes False --
                    waking write. Simulation/replay paths that call
                    pacc.tick with True are skipped.)
  Computation:
    if hypothesis_tag: skip
    elif z_harm_a_norm <= z_harm_a_min: target = 0  (Guo 2018 rest relaxation)
    else: target = tanh(z_harm_a_norm) * drive_scale
    drive_bias = (1 - alpha*gate) * drive_bias + alpha*gate*target
    drive_bias = clip(drive_bias, -cap, +cap)
  Read path: effective_drive(base) = clip_[0,1](base + drive_bias).
  Consumers (all in agent.py select_action / sense / update_z_goal):
    - dACC bundle drive_level input (SD-032b)
    - SalienceCoordinator.tick drive_level (SD-032a)
    - AICAnalog.tick drive_level input (SD-032c; one-step lag via next sense)
    - PCCAnalog.tick drive_level input (SD-032d)
    - GoalState.update drive_level (SD-012 wanting-gain scaling)
  Convention: goal_state._last_drive_level stores the BASE drive_level;
  SD-032 consumers apply pacc.effective_drive() themselves to avoid
  double-counting the bias.
  Per-episode reset() clears diagnostics cache only -- drive_bias is
  cross-episode by architectural intent. enter_offline_mode() calls
  note_offline_entry() (default no-op at offline_decay=0.0).
  Config: REEConfig.use_pacc_analog (bool, default False).
    Sub-knobs: pacc_drive_alpha (0.002, ~347-step half-life),
    pacc_drive_scale (1.0), pacc_drive_bias_cap (0.5, absolute cap
    on |drive_bias|), pacc_z_harm_a_min (0.0, threshold below which
    target is 0 -- reversibility under quiescence),
    pacc_offline_decay (0.0, distinct sleep-recalibration claim if
    set non-zero).
  Falsification signature (spec): sustained z_harm_a exposure produces
    drift in drive_level, which modulates SD-032c switch threshold and
    GoalState wanting gain. With SD-032e OFF, same sustained z_harm_a
    leaves drive_level untouched (only obs_body[3] energy depletion
    moves it) -- no chronic-pain-sensitisation signature possible.
  Backward compatible: use_pacc_analog=False by default; agent.pacc is
    None and every integration site is a no-op. Existing experiments
    unaffected.
  Biological basis: Vogt 2005 ACC subdivisions (perigenual/subgenual
    as autonomic/affective-output hub); Mayberg 2005 sgACC
    depression-baseline setpoint (cited for valence-setpoint role the
    current implementation does NOT directly instantiate -- shape
    mismatch documented); Critchley 2003 ACC-autonomic coupling;
    Gianaros 2011 ACC-PAG-medulla fast-effector route (out of V3
    scope; future SD-032f); Guo 2018 ACC mGluR5 LTP days-timescale
    plasticity (primary grounding for slow-EMA default); Baliki 2012
    corticostriatal chronic-pain drift (falsification-signature
    behaviour the substrate targets).
  MECH-094: handled by hypothesis_tag skip in tick(); waking
    select_action writes are valid (tag=False).
  Phased training: not applicable (non-trainable arithmetic).
  Validation experiment: V3-EXQ-448 queued (4-arm ablation:
    pACC-OFF / pACC-ON-normal-z_harm_a / pACC-ON-sustained-z_harm_a /
    pACC-ON-hypothesis-tag-only; acceptance: drive_bias monotone in
    sustained exposure magnitude, MECH-094 skip suppresses accumulation,
    bias bounded by cap, downstream effective_drive shifts AIC
    harm_s_gain and coordinator effective_threshold in expected
    directions).
  See SD-032e, SD-032a, SD-032c, SD-032d, SD-012, SD-011, MECH-261,
  MECH-094, SD-032 parent.

- ARC-058: harm_stream.shared_forward_trunk -- REGISTERED 2026-04-19,
  COMPETES WITH ARC-033.
  Module: ree_core/latent/stack.py (HarmForwardTrunk, HarmForwardHead
  -- pre-existing substrate classes). Selection via shared_trunk
  constructor arg on E2HarmSForward / E2HarmAForward (see MECH-258).
  ARC-033 claim: independent per-stream forward models (separate
  ResidualHarmForward per stream). Biological reading: dorsal posterior
  insula (sensory PE) + anterior insula (affective PE) as separate
  learned substrates.
  ARC-058 claim (competing): shared HarmForwardTrunk (unsigned,
  modality-independent PE substrate) + stream-specific HarmForwardHead
  (signed, per-modality readout). Biological reading: Horing & Buchel
  2022 anterior insula encodes modality-independent unsigned PE shared
  across aversive modalities; dorsal posterior insula encodes
  modality-specific signed PE. Trunk ~ unsigned; head ~ signed.
  Same nn.Module topology, different wiring. Constructor switch arbitrates.
  Falsifiable: V3-EXQ-445 three-arm ablation measures per-stream
  forward_r2 for z_harm_s and z_harm_a + downstream dACC bundle
  usefulness under each path. If shared-trunk matches or beats
  independent with fewer parameters AND produces a useful unsigned
  PE signal, ARC-058 wins and ARC-033 is narrowed. If independence
  wins, ARC-058 is retired.
  See ARC-058, ARC-033, MECH-258, MECH-257, SD-032b.

## SD-029: Balanced Hazard-Event Curriculum (2026-04-21)
- SD-029: self_attribution.comparator_z_harm_s -- CURRICULUM-LEVEL BALANCED HAZARD-EVENT SUPPORT IMPLEMENTED 2026-04-21.
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorldV2).
  The substrate for the z_harm_s comparator (E2_harm_s forward model, ARC-033) and its
  interventional training (SD-013) already pass C1 (forward_r2 >= 0.9) and C2 (partial
  attenuation). The remaining blocker for C3/C4 (event-conditioned SNR on approach-to-harm
  events, with n_self >= 20 AND n_ext >= 20 per seed) is curriculum-level: the default env
  produces highly imbalanced hazard-event densities (some seeds near-zero self-caused,
  others near-zero externally-caused). This substrate change adds a scheduled
  externally-caused hazard injection curriculum to the env.
  Mechanism: when scheduled_external_hazard_enabled=True, every
  scheduled_external_hazard_interval steps, with probability scheduled_external_hazard_prob,
  an existing hazard is moved (or new one spawned) to a cell adjacent to the agent
  (or any empty cell when adjacent_only=False). Purely curriculum-level: the agent did
  not initiate the encounter, so subsequent harm is externally-caused in the
  self-vs-externally-caused taxonomy. Agent and latent code unchanged.
  Config (CausalGridWorldV2 __init__): scheduled_external_hazard_enabled (bool,
  default False -- no-op); scheduled_external_hazard_interval (int, default 50);
  scheduled_external_hazard_prob (float, default 0.5);
  scheduled_external_hazard_adjacent_only (bool, default True -- if no empty neighbour
  and False, falls back to any empty cell; if True, a tick with no adjacency is skipped).
  New env state and info keys:
    self._external_hazard_event_count: per-episode counter (reset in reset()).
    info["external_hazard_injected"]: bool, True on the step the injection fired.
    info["external_hazard_event_count"]: int, cumulative this episode.
  Data flow: step() -> after agent move / env drift checks -> [enabled and steps%interval==0
  and rng<prob] -> _inject_external_hazard() -> hazard relocated/spawned ->
  info tags set -> proximity fields recomputed.
  Backward compatible: scheduled_external_hazard_enabled=False by default; env state
  is unchanged relative to legacy behaviour (_drift_hazards, _respawn_resource unaffected).
  Info dict tags are always present (value 0 / False when disabled), but existing
  experiments that don't read them are unaffected.
  Biological basis: none required (curriculum-level env augmentation). The distinction
  this supports (self-caused vs externally-caused harm events) is a prerequisite of the
  Blakemore / Shergill / Frith comparator literature already grounding SD-029.
  Phased training: not applicable (env only; no new trainable parameters).
  MECH-094: not applicable (env observation stream, not replay content).
  Validation experiment: V3-EXQ-470 queued (diagnostic ablation:
  SCHEDULED vs BASELINE, 4 seeds, confirms per-seed n_ext >= 20 under the curriculum
  and that balanced event counts preserve C1/C2 while enabling C3/C4 measurement).
  See SD-029, MECH-256, ARC-033, SD-013.
  Design-doc reference: REE_assembly/docs/architecture/self_attribution_per_stream.md.

## SD-047: Multi-Source Environmental Dynamics (2026-05-03)
- SD-047: environment.multi_source_dynamics -- IMPLEMENTED 2026-05-03.
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorld /
  CausalGridWorldV2). Three concurrent stochastic event sources at distinct
  spatial / temporal scales, each agent-independent, layered onto the
  existing SD-022 / SD-029 hazard substrate. Substrate-ceiling unblock for
  MECH-095 TPJ agency-detection comparator (V3-EXQ-506 C4-only-PASS pattern,
  2026-05-03). Lit-anchor: 18 PubMed entries (Asai 2016 non-monotonic agency
  S/N; Sawtell 2010 cerebellar cancellation; Pitcher & Ungerleider 2021
  lateral cortex network; Woo/Spelke 2023 falsifier; passivity cluster
  Blakemore/Frith 2000, Synofzik 2008, Stirling 2001, Gallagher 2004,
  Shamanna 2023, Brandt 2017, Ganos 2015, Lyndon 2026, Seth & Friston 2016,
  Jeganathan & Breakspear 2021, Nassar 2021, Jardri & Deneve 2013, Ward 2010).
  SD-047 lit_conf=0.841.
  Three sources:
    Source 1 (weather field): AR(1) coarse-grid additive perturbation on
      hazard_field. Continuous, smooth, autocorrelated, agent-independent.
      Per-cell signature for cerebellar-style cancellation tests (MECH-098).
      Stationary AR(1) form: x_{t+1} = alpha*x_t + sqrt(1-alpha^2)*sigma*N(0,1).
      Variance bounded at sigma^2 across long episodes.
    Source 2 (transient events): Poisson appear / disappear of transient
      hazard cells. Discrete, spatially pointwise, short-lived,
      agent-independent. Tracked separately from self.hazards
      (self._transient_hazards) for bookkeeping; underlying cell still
      registered in self.hazards so proximity field treats it as a hazard.
    Source 3 (background drift): n_drift_sources mobile single-cell hazard-
      analog objects with random_walk / linear_drift / levy_walk dynamics.
      Discrete, mobile, autocorrelated, agent-independent. Tracked in
      self._drift_sources; same dual-list bookkeeping as transients.
  Config (CausalGridWorld __init__ kwargs, env-only -- not surfaced through
  REEConfig.from_dims, matching SD-023 / SD-029 precedent for env-only SDs):
    multi_source_dynamics_enabled (bool, default False) -- master switch.
    multi_source_intensity_scale (float, default 1.0) -- 4-arm noise-sweep
      lever (OFF / 0.25 / 1.0 / 4.0); scales weather sigma, transient
      p_appear, and drift move probability uniformly.
    weather_field_enabled (bool, default False) -- per-source switch.
    weather_super_cells (int, default 4) -- coarse AR(1) grid resolution.
    weather_alpha_ar1 (float, default 0.95) -- temporal autocorrelation.
    weather_sigma (float, default 0.05) -- per-cell perturbation magnitude.
    transient_events_enabled (bool, default False) -- per-source switch.
    transient_p_appear (float, default 1e-3) -- per-tick per-cell appearance.
    transient_p_disappear (float, default 0.1) -- mean lifespan ~10 ticks.
    transient_intensity (float, default 1.0) -- reserved for harm scaling.
    background_drift_enabled (bool, default False) -- per-source switch.
    n_drift_sources (int, default 1) -- count.
    drift_policy (str, default "random_walk") -- one of random_walk /
      linear_drift / levy_walk.
  Data flow (step()): existing logic (move, harm, contamination, SD-022
    limb damage) -> SD-029 _inject_external_hazard -> _drift_hazards
    (legacy env_drift) -> [if multi_source_dynamics_enabled] _step_weather_field
    -> _step_transient_events -> _step_background_drift -> _compute_proximity_fields
    if any source perturbed hazard layout / weather. Existing SD-029 path is
    untouched; SD-047 is purely additive.
  info dict tags (always present, 0 / False when disabled):
    multi_source_dynamics_enabled, multi_source_intensity_scale,
    multi_source_weather_step_delta, multi_source_n_transient_appear,
    multi_source_n_transient_disappear, multi_source_n_transient_active,
    multi_source_n_drift_moved, multi_source_n_drift_active,
    multi_source_n_env_events (cumulative env-caused per tick),
    multi_source_n_agent_events (cumulative agent-caused per tick).
    The ratio multi_source_n_env_events / multi_source_n_agent_events is the
    calibration target signal for the validation experiment (target 1:1-2:1).
  Backward compatible: multi_source_dynamics_enabled=False by default;
    _init_multi_source_state not called; _step_* not called; proximity
    field path unchanged; info tags zero / False. RNG draws guarded inside
    `if multi_source_dynamics_enabled:` so seed sequences for existing
    experiments are bit-identical when disabled. 7/7 preflight + 184/184
    contracts PASS with master OFF (smoke 2026-05-03).
  Activation smoke (2026-05-03, 4-arm sweep, 200 ticks each):
    ARM_0 (OFF): env_events=2 (background drift only), agent_events=193,
      bit-identical to legacy.
    ARM_1 (scale=0.25 ON): env_events=77, agent_events=185.
    ARM_2 (scale=1.0 ON): env_events=330, agent_events=169.
      Calibration ratio 1.95:1 -- matches SD doc target 1:1-2:1.
    ARM_3 (scale=4.0 ON): env_events=354, agent_events=175.
      Saturation at high noise -- transient-pool bound.
    Weather AR(1) firing (super_field abs-mean 0.053 with sigma=0.1);
    drift sources moving 170 / 240 attempts (~0.71 effective move rate);
    transients churning at ~5e-3 per cell per tick.
  Implementation choices (deviations / clarifications from SD doc):
    - Flat kwargs on CausalGridWorld.__init__ rather than nested dataclass
      (MultiSourceDynamicsConfig). Matches SD-022 / SD-023 / SD-029 precedent
      for env-only SDs; nothing else uses dataclass-config for env params.
    - Transient + drift hazards live in self.hazards (with parallel
      bookkeeping lists for movement / disappearance) rather than getting
      a new ENTITY_TYPES entry. Adding a new entity type would change
      NUM_ENTITY_TYPES from 7 to 8 and break local_view shape (5x5x7=175 ->
      5x5x8=200), violating backward compat.
    - Per-source bit-identical OFF preserved: each source's RNG draws are
      gated by its own switch and the master switch, so single-source
      ablation studies are clean.
  No trainable parameters. Pure env-side stochastic substrate enrichment.
  No phased training needed.
  MECH-094: not applicable (env observation stream, not replay / simulation
    content). Validation experiments call sense() / step() in waking mode;
    simulation paths do not invoke env.step.
  Validation experiment: V3-EXQ-509 queued (4-arm noise sweep
    {OFF / 0.25x / 1.0x / 4.0x}, V3-EXQ-506-equivalent metrics). Pre-
    registered prediction per SD doc: ARM_0 replicates V3-EXQ-506 C1-C3
    FAIL; C1, C2, C3 pass-rate forms inverted U across ARM_1 -> ARM_2 ->
    ARM_3 with ARM_2 peak (Asai 2016 non-monotonic). Five-row interpretation
    grid handles validation / calibration miscalibration / Woo & Spelke
    falsifier branch (re-route MECH-095 to substrate_conditional V4) /
    opposite-direction artefact / standard validation.
  Design doc: REE_assembly/docs/architecture/sd_047_multi_source_dynamics.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_sd_047/
    + targeted_review_connectome_mech_095/ (passivity cluster).
  See SD-047, MECH-095 (substrate-ceiling unblock), MECH-098 (reafference
    cancellation, secondary unblock), MECH-099 (downstream agency
    attribution, secondary unblock), SD-022 (body-damage substrate, layered
    on top of), SD-029 (scheduled hazard curriculum, layered on top of),
    ARC-033 (E2_harm_s forward, indirect benefit on cf_gap_ratio), MECH-094
    (call-site scoping; not applicable).

## SD-033a: Lateral-PFC-analog / MECH-261 Primary Consumer (2026-04-20)
- SD-033a: pfc.lateral_pfc_analog -- IMPLEMENTED 2026-04-20.
  Module: ree_core/pfc/lateral_pfc_analog.py (LateralPFCAnalog,
  LateralPFCConfig). First subdivision of SD-033 (PFC subdivision cluster)
  and primary consumer of MECH-261's write-gate registry on SD-032a
  (SalienceCoordinator). Instantiates MECH-262 (rule-selective persistence).
  Config: REEConfig.use_lateral_pfc_analog (bool, default False).
  Sub-knobs: lateral_pfc_rule_dim (16), lateral_pfc_update_eta (0.05),
  lateral_pfc_world_pool_weight (0.5), lateral_pfc_bias_scale (0.1),
  lateral_pfc_hidden_dim (32).
  State: rule_state buffer [1, rule_dim], persistent across ticks within
  episode, reset on agent.reset(). Cross-episode carry-over is NOT
  implemented (V3 simplification; V4 extension if required).
  Update rule: rule_state <- (1 - eff_eta) * rule_state + eff_eta * source
    where eff_eta = update_eta * gate, gate = write_gate("sd_033a"),
    source = delta_proj(z_delta) + world_pool_weight * world_proj(z_world).
    Gate near 0 (internal_replay weight 0.05) -> rule-state near-frozen
    (distractor resistance). Gate near 1 (external_task / internal_planning)
    -> fast update.
  Bias head: frozen-random with last nn.Linear weights and bias ZEROED at
    init so initial bias output is EXACTLY zero. Head takes concat(
    [rule_state, per-candidate z_world summary]) -> scalar per candidate
    -> clamp [-bias_scale, +bias_scale] -> [K] bias vector added to
    dacc_score_bias before E3.select(). Training-dependent emergence
    (SD-033 signature iv) deferred: phased-training protocol not wired.
  Data flow: select_action() -> gate = salience.write_gate("sd_033a") (or
    1.0 if coord disabled) -> lateral_pfc.update(z_delta, z_world, gate)
    -> per-candidate summaries from trajectory.world_states[:, 0, :] ->
    lateral_pfc.compute_bias(summaries) -> add to dacc_score_bias ->
    e3.select(score_bias=...).
  Backward compatible: use_lateral_pfc_analog=False by default. When
    True with the zeroed-last-layer head, initial bias output is exactly
    zero -- agent runs bit-identical to baseline until the head is
    deliberately trained (deferred).
  Biological basis: Miller & Cohen 2001 (rule-as-top-down-bias), Badre
    & Nee 2018 (mid-lateral rule-hierarchy), Mansouri 2020 (rule-selective
    persistence). MECH-261 per-mode weights for sd_033a (from spec):
    external_task=1.0, internal_planning=1.0, internal_replay=0.05,
    offline_consolidation=0.3.
  MECH-094: rule persistence is gated by the MECH-261 registry (not by
    a separate hypothesis_tag check). MECH-261 generalises MECH-094:
    write_gate("sd_033a") = 0.05 in internal_replay mode means replay
    content cannot meaningfully update rule_state. The gate IS the tag.
  DESIGN ALTERNATIVES (documented in design doc, lit-pulls queued in
    task_inbox.md): A1 per-candidate vs uniform bias; A2 frozen-random
    head with zeroed last Linear vs trained head via phased protocol;
    A3 gate-modulated EMA vs recurrent GRU / synaptic-hold persistence.
  Smoke test (2026-04-20): module instantiates; gate=1.0 rule_state delta
    ~0.1 on single tick; gate=0.0 rule_state delta < 1e-6 (freeze); initial
    bias vector exactly zero; reset() zeroes rule_state. E2E five-tick
    loop with SD-033a ON hits the same pre-existing multinomial-on-
    untrained-E3-scores edge case that SD-033a-OFF also hits; confirmed
    not caused by this SD.
  Validation experiment: V3-EXQ-456 queued (diagnostic -- five sub-tests:
    instantiation, gate modulates update rate, bias reaches E3 with
    zero-init contract, backward compat, reset clears rule_state).
  Phased training: deferred until A2 alternative is considered.
  Design doc: REE_assembly/docs/architecture/sd_033a_lateral_pfc_analog.md
  See SD-033, SD-033a, MECH-261, MECH-262, SD-032a, SD-032b, MECH-094.

## SD-033b: OFC-analog / MECH-261 Second Consumer (2026-04-26)
- SD-033b: pfc.ofc_analog -- IMPLEMENTED 2026-04-26.
  Module: ree_core/pfc/ofc_analog.py (OFCAnalog, OFCConfig). Second
  subdivision of SD-033 (PFC subdivision cluster) and second consumer
  of MECH-261's write-gate registry on SD-032a (SalienceCoordinator).
  Substrate for MECH-263 functional signatures (devaluation sensitivity,
  same-sensory / different-task-role discrimination); the behavioural
  signatures themselves are deferred to environment-extension EXQs.
  Config: REEConfig.use_ofc_analog (bool, default False).
  Sub-knobs: ofc_state_dim (16), ofc_update_eta (0.05),
  ofc_outcome_pool_weight (0.5), ofc_bias_scale (0.1),
  ofc_hidden_dim (32), ofc_harm_dim (0). harm_dim=0 (default) builds the
  state_code from z_world only; setting harm_dim to the SD-011 z_harm
  dim turns on outcome_proj so harm-magnitude information enters the
  outcome-state code (the architectural shape MECH-263 devaluation
  sensitivity probes).
  State: state_code buffer [1, state_dim], persistent across ticks
  within episode, reset on agent.reset(). Cross-episode carry-over not
  implemented (V3 simplification, parallel to SD-033a).
  Update rule: state_code <- (1 - eff_eta) * state_code + eff_eta * source
    where eff_eta = update_eta * gate, gate = write_gate("sd_033b"),
    source = world_proj(z_world).mean(0)
           + outcome_pool_weight * outcome_proj(z_harm).mean(0) (if harm_dim>0).
    Gate near 0 (internal_replay weight 0.05) -> state_code near-frozen.
    Gate near 1 (external_task weight 1.0) -> fast update. internal_planning
    weight 0.5 (vs 1.0 for sd_033a) reflects partial replanning during
    planning rollouts.
  Bias head: frozen-random with last nn.Linear weights and bias ZEROED at
    init so initial bias output is EXACTLY zero. Head takes concat(
    [state_code, per-candidate z_world summary]) -> scalar per candidate
    -> clamp [-bias_scale, +bias_scale] -> [K] bias vector added to
    dacc_score_bias before E3.select(). Training-dependent emergence
    deferred along with MECH-263 behavioural signatures (env extension
    required: outcome relabelling, task-role-distinct state pairs).
  Data flow: select_action() -> gate = salience.write_gate("sd_033b") (or
    1.0 if coord disabled) -> ofc.update(z_world, z_harm-if-harm_dim>0,
    gate) -> per-candidate summaries (reused from lateral_pfc tick block
    when SD-033a also on; built fresh otherwise) -> ofc.compute_bias(
    summaries) -> add to dacc_score_bias -> e3.select(score_bias=...).
  Backward compatible: use_ofc_analog=False by default. When True with
    the zeroed-last-layer head, initial bias output is exactly zero --
    agent runs bit-identical to baseline until the head is deliberately
    trained. 143/143 contracts PASS with substrate landed.
  Biological basis: MECH-263 OFC functional signatures (devaluation
    sensitivity, same-sensory / different-task-role discrimination).
    MECH-261 per-mode weights for sd_033b (from spec): external_task=1.0,
    internal_planning=0.5, internal_replay=0.05, offline_consolidation=0.3.
  MECH-094: handled via MECH-261 generalisation. write_gate("sd_033b")=
    0.05 in internal_replay means replay content cannot meaningfully
    update state_code. The gate IS the tag.
  Smoke test (2026-04-26): module instantiates; gate=1.0 state_code delta
    ~0.27 on single tick; gate=0.0 state_code delta exactly 0.0 (freeze);
    initial bias vector max-abs exactly zero; reset() zeroes state_code.
    EXQ-485 5/5 sub-tests PASS.
  Validation experiment: V3-EXQ-485 queued (diagnostic -- five sub-tests:
    instantiation + state_code shape, gate=1 vs gate=0 update modulation,
    bias zero at init, backward compat, reset clears state_code). Smoke
    PASS 2026-04-26. Behavioural validation (MECH-263 devaluation +
    task-role discrimination) deferred to env-extension EXQs.
  Phased training: deferred along with MECH-263 behavioural signatures.
  Design doc: REE_assembly/docs/architecture/sd_033b_ofc_analog.md
  See SD-033, SD-033a (sibling consumer; additive E3 bias composition),
    SD-033b, MECH-261, MECH-263, SD-032a, SD-032b, MECH-094.

## SD-034: Governance Closure Operator (2026-04-20)
- SD-034: governance.closure_operator -- IMPLEMENTED 2026-04-20.
  Module: ree_core/governance/closure_operator.py (ClosureOperator,
  ClosureOperatorConfig, ClosureEvent). First substrate in SD-033
  governance cluster; first consumer of MECH-261 write-gate registry's
  mode-conditioning predicate. Coordinates a five-part "done" token
  emitted at rule-completion events:
    (a) MECH-090 beta_gate.release() -- commitment latch drops.
    (b) MECH-260 dacc.inject_nogo(action_class, count) -- targeted
        No-Go FIFO injection on the just-completed action class
        (semantically distinct from execution record; same mechanism).
    (c) ResidueField.discharge_domain(z_world, factor, radius) --
        rule-domain multiplicative decay on RBF weights; hard 1e-6
        floor preserves the "residue cannot be erased" invariant.
    (d) SalienceCoordinator.update_signal("closure_event", value) --
        re-biases affinity toward internal_planning via registered
        affinity_weights (default internal_planning=0.5).
    (e) dacc.reset_episode_pe() + optional dacc_pe_cap install --
        MECH-268 pe saturation/reset.
  Config: REEConfig.use_closure_operator (bool, default False). Sub-
  knobs: closure_rule_delta_threshold (0.001), closure_stable_ticks
  (3), closure_require_beta_elevated (True), closure_min_sd033a_gate
  (0.5), closure_nogo_injection_count (3), closure_residue_discharge_
  factor (0.5), closure_residue_discharge_radius (1.5),
  closure_signal_value (1.0), closure_reset_pe_ema (True),
  closure_pe_cap_after (None), closure_signal_affinity_internal_
  planning (0.5).
  Completion detector (tick path): rule_state delta < threshold for
  N consecutive ticks AND beta elevated AND current_mode in
  allowed_closure_modes AND sd_033a write_gate >= min. Rule-state
  norm guard prevents firing on unset rule_state. Explicit
  emit_closure() path is the experiment hook (bypass_mode_
  conditioning for controlled ablations).
  Mode conditioning is the falsifiability predicate: if MECH-090 +
  MECH-260 + MECH-094 tuning WITHOUT closure produces the signature
  in follow-up behavioural variants, SD-034 is over-specification.
  ResidueField.discharge_domain API added in same pass: multiplicative
  decay + sign-aware 1e-6 floor + radius-scoped in-domain selection
  via squared distance vs (radius * bandwidth)^2; valence_vecs NOT
  modified (4-component valence preserved so replay prioritisation
  remains faithful). DACCAdaptiveControl extended with dacc_pe_cap,
  inject_nogo(), reset_episode_pe() (distinct from full reset() --
  preserves _action_history where the just-injected No-Go lives).
  Agent wiring: REEAgent.__init__ instantiates closure_operator when
  enabled (requires use_lateral_pfc_analog=True, use_dacc=True;
  salience coordinator optional). select_action() calls tick() after
  action emission with current z_world + argmax action_class +
  operating_mode + sd_033a gate. reset() calls closure_operator.reset().
  register_on_coordinator() wires closure_event into
  salience.config.affinity_weights at init.
  Backward compatible: use_closure_operator=False by default ->
  agent.closure_operator is None; every integration site is a no-op.
  Existing experiments unaffected. Bit-identical with
  closure_signal_affinity_internal_planning=0.0 and the master switch
  off.
  Biological basis: Rich & Shapiro 2009 (OFC sequence-completion
  cells); Collins & Frank 2014 (task-set disengagement); Schuck 2016
  (mPFC task-stage encoding). Five-part signal collocates multiple
  biologically-observed end-of-sequence signatures; EXP-0156 and
  EXP-0162 probe whether the collocation is a single substrate or
  co-occurring independent processes.
  MECH-094: mode-conditioning on operating_mode generalises the
  MECH-094 hypothesis-tag -- internal_replay / offline_consolidation
  modes block closure firing via allowed_closure_modes and via
  sd_033a gate floor (write_gate("sd_033a")=0.05 in internal_replay).
  Validation experiments:
    V3-EXQ-460 (EXP-0156, ocd4 verified-but-not-released) -- landing
      diagnostic (6 sub-tests: backward compat, wiring, beta release,
      No-Go, pe reset, mode conditioning). PASS on smoke.
    V3-EXQ-466 (EXP-0162, ocd4 satisficing / No-Go thresholding) --
      residue-discharge landing diagnostic (5 sub-tests: near
      attenuation, far spared, invariant preserved, closure->discharge
      end-to-end, distant-z spares). PASS on smoke.
  Behavioral variants with full E3 task loop + tolerance-band
  completion env are deferred: they depend on phased rule_state
  training and an env variant not yet on any roadmap item.
  Anchor doc: REE_assembly/evidence/planning/sd033_governance_plan.md
  Source: docs/thoughts/2026-04-20_ocd4.md
  See SD-034, MECH-090, MECH-260, MECH-261, MECH-262, MECH-094,
  MECH-268, SD-032a, SD-033a.

## SD-035: Amygdala Analogue -- BLA + CeA Peer Modules (2026-04-21)
- SD-035: amygdala.analog_bla_cea_peers -- IMPLEMENTED 2026-04-21.
  Modules:
    ree_core/amygdala/bla.py (BLAAnalog, BLAConfig, BLAOutput)
    ree_core/amygdala/cea.py (CeAAnalog, CeAConfig, CeAOutput)
  Two peer non-trainable arithmetic substrates mirroring the biological
  BLA / CeA division. Both read z_harm_a (SD-011 affective stream) and
  write to different downstream consumers. Master switch
  use_amygdala_analog gates both; per-module switches use_bla_analog and
  use_cea_analog give granular control.

  BLAAnalog (basolateral-analog, slow/content) -- MECH-074a/b/d:
    MECH-074a encoding_gain: inverted-U arousal-dependent multiplier on
      HippocampalModule write strength (Roozendaal 2011). Threshold
      on_arousal=0.4, peak=0.7, max gain=2.5, window=18000 steps,
      half-life=3600 steps. Zero below threshold; falls back to 1.0 in
      the tail of the post-encoding window.
    MECH-074b retrieval_bias: content-selective per-trace weight
      vector w_i = 1 + alpha * arousal_tag_i (NOT a scalar; LaBar &
      Cabeza 2006). Requires arousal_tags_in_context from caller
      (hippocampal retrieval side). None-passthrough when no context
      provided.
    MECH-074d remap_signal: Moita 2004 attribution-gated per-code
      remap. Fires when PE z-score > remap_pe_sigma_threshold AND
      candidate_code_contributions attribution dict is supplied.
      Output is {code_idx: 1.0} over the top remap_code_fraction of
      attribution candidates (default 33%).
  BLA outputs are cached on agent._bla_last_output. V3 hippocampal
  consumer wiring (write-gain multiplication, retrieval reweighting,
  remap handoff) is deferred -- the module emits the signals but the
  HippocampalModule does not yet read them. First-pass consumer wiring
  is gated on EXQ-B acceptance.

  CeAAnalog (central-analog, fast/scalar) -- MECH-046/074c:
    MECH-046 mode_prior: pre-softmax additive log-odds bias written
      to SalienceCoordinator.affinity_weights. Fires within 1-2 sim
      steps (~75 ms biological; Mendez-Bertolo 2016) when
      |LowFreq(z_harm_a)| > fast_route_threshold. Distinct from AIC
      urgency (SD-032c): AIC modulates mode-SWITCH threshold; CeA
      mode_prior biases mode SELECTION.
    MECH-074c fast_prime: scalar candidate-prior pulse distinct from
      mode_prior (Pessoa & Adolphs 2010 many-roads framing). Override
      window 5-10 steps; cortical_confirmation signal holds the pulse
      across the window or accelerates decay (tau=4 steps base).
    Q-036 escapability_hint: placeholder input (no-op pass-through)
      so MECH-219 escapability wiring can land without an interface
      refactor.
  CeA outputs are cached on agent._cea_last_output and injected into
  SalienceCoordinator via update_signal calls in select_action() BEFORE
  coordinator.tick() each cycle:
    update_signal("cea_mode_prior", mode_prior_float)
    update_signal("cea_fast_prime", fast_prime_float)
  Signal slots registered at agent __init__:
    affinity_weights["cea_mode_prior"] = {"external_task": 1.0}
    salience_weights["cea_fast_prime"] = 0.5

  Config: REEConfig.use_amygdala_analog (bool, default False).
  Sub-switches: use_bla_analog (bool, default True),
  use_cea_analog (bool, default True) -- only take effect when master
  is True. 14 BLA flat params (bla_encoding_gain_max, bla_encoding_gain_floor,
  bla_arousal_threshold_on, bla_arousal_peak, bla_window_steps,
  bla_window_half_life_steps, bla_retrieval_bias_alpha,
  bla_retrieval_bias_compensation, bla_retrieval_tag_at_encoding,
  bla_remap_pe_sigma_threshold, bla_remap_pe_ema_alpha,
  bla_remap_pe_std_init, bla_remap_code_fraction,
  bla_remap_requires_attribution) and 9 CeA flat params
  (cea_fast_route_threshold, cea_fast_route_input_is_lowfreq,
  cea_mode_prior_log_odds_max, cea_mode_prior_gain,
  cea_pre_softmax_additive, cea_fast_prime_amplitude,
  cea_fast_prime_decay_tau_steps, cea_fast_prime_override_window_steps,
  cea_cortical_confirmation_weight). All wired through
  REEConfig.from_dims() with synthesis-seeded defaults (see
  REE_assembly/evidence/literature/targeted_review_amygdala_analog/
  synthesis.md).

  Data flow:
    sense() -> LatentStack.encode() -> z_harm_a (SD-011, requires
      use_affective_harm_stream=True) -> bla.tick(z_harm_a, z_harm_a_pred)
      AND cea.tick(z_harm_a) -> cache outputs ->
    select_action() -> coordinator update_signal("cea_mode_prior", ...)
      AND update_signal("cea_fast_prime", ...) -> coordinator.tick() ->
      mode affinity and salience aggregate reflect the fast route.
  BLA retrieval_bias / remap_signal hippocampal wiring: DEFERRED.
  Outputs produced and cached; HippocampalModule consumer wiring lands
  when EXQ-B passes and the retrieval-bias-aware replay path is added.

  Backward compatible: use_amygdala_analog=False by default; both
  modules are None; all integration sites are no-ops. 33/33 preflight +
  contract tests PASS with flag OFF (2026-04-21). Bit-identical to
  baseline.

  Activation smoke (2026-04-21, flag ON):
    CeA mode_prior: 0.0 at rest -> 0.3 under synthetic threat
      (L1/dim=0.8; threshold 0.5; cap mode_prior_log_odds_max=0.8 *
      gain=0.5).
    CeA fast_prime: 0.0 at rest -> 0.225 under threat.
    BLA encoding_gain: 1.0 at rest -> 2.5 under synthetic arousal
      (inverted-U cap).
    BLA remap_signal: fires on synthetic PE spike when attribution
      candidates supplied (Moita 2004 gate).
    All three activation signatures confirmed; agent.sense() one-tick
    boot completes without error with amygdala ON.

  Biological basis (see synthesis.md):
    BLA encoding: McGaugh 2004, Roozendaal 2011 (arousal-dependent LTP
      modulation of hippocampal consolidation).
    BLA retrieval: LaBar & Cabeza 2006, Dolcos et al 2005, Phelps 2004
      (content-selective per-trace bias, not global arousal gain).
    BLA remap: Moita 2004, Nader 2000, Schiller 2010 (PE-spike
      remapping on violated expectation; attribution-gated).
    CeA mode_prior: LeDoux 1996 "low road", Pessoa & Adolphs 2010
      (fast subcortical route, mode SELECTION bias distinct from
      cortical mode-SWITCH threshold).
    CeA fast_prime: Mendez-Bertolo 2016 (~75 ms fast visual-amygdalar
      pulvinar route), Pessoa & Adolphs 2010 (cortical confirmation
      window).
  MECH-094: CeAAnalog.tick() accepts simulation_mode argument;
    returns zeroed output without updating state when True.
    BLAAnalog defers to caller for simulation gating (encoding-gain
    writes are gated at the HippocampalModule consumer side via
    MECH-261).
  Phased training: not applicable (non-trainable arithmetic).
  Design doc: REE_assembly/docs/architecture/sd_035_amygdala_analog.md
  Literature synthesis: REE_assembly/evidence/literature/
    targeted_review_amygdala_analog/synthesis.md
  Validation experiments: V3-EXQ-A (CeA mode-prior ablation, MECH-046)
    and V3-EXQ-B (BLA encoding + remap, MECH-074a/d) -- queued in a
    follow-up pass.
  See SD-035, MECH-046, MECH-074, MECH-074a, MECH-074b, MECH-074c,
  MECH-074d, SD-011, SD-032a, SD-032c, Q-036.

## SD-036 + MECH-279: GABAergic Cross-Stream Decay + PAG Freeze-Gate (2026-04-22)
- SD-036: regulators.gabaergic_cross_stream_decay -- IMPLEMENTED 2026-04-22.
  Module: ree_core/regulators/gabaergic_decay.py (GABAergicDecayRegulator,
  GABAergicDecayConfig, StreamRegistration). Regulator-layer substrate
  (NOT per-stream update rule): a single broadly-projecting tonic GABAergic
  decay applied across multiple registered latent streams in parallel.
  Decay formula:
    z_s(t+1) = z_s(t) * exp(-tau_s * gaba_tone(t))
  with per-stream baseline tau and a global gaba_tone multiplier in
  [0, 2] (default 1.0). gaba_tone > 1.0 = benzo-analog (faster decay,
  easier exit from committed states); gaba_tone < 1.0 = withdrawal /
  chronic-stress analog (slower decay); gaba_tone = 0.0 = decay
  suspended.
  Default coverage (tau values from design doc):
    z_harm   tau=0.05  (~20-step half-life)  -- SD-010 sensory harm
    z_harm_a tau=0.02  (~50-step half-life)  -- SD-011 affective harm
    z_beta   tau=0.03  (~30-step half-life)  -- MECH-090 precision/affective
  Drive accumulator (SD-012) intentionally NOT covered -- the homeostatic
  override mechanism (separate, V4-or-late-V3) provides drive dynamics.
  Suspend-on-input gate: per-stream input_threshold; when |z(t)-z(t-1)|
  exceeds threshold, decay is skipped for that tick (the input drives
  the update). Default 0.0 = always decay.
  Decay is OUT-OF-PLACE (detach + scalar multiply + setattr): an in-place
  mul_() on encoder outputs breaks autograd version tracking when those
  outputs are concurrently consumed by SD-018 resource_proximity_head /
  SD-011 harm_accum_head aux losses. Out-of-place is required for the
  EXQ-471 training pipeline.
  Config: REEConfig.use_gabaergic_decay (bool, default False). 14 sub-
  knobs in REEConfig.from_dims: gaba_tone (1.0), gaba_tone_min (0.0),
  gaba_tone_max (2.0), per-stream tau (gaba_tau_z_harm_s/a/beta),
  per-stream coverage flags (gaba_decay_z_harm_s/a/beta), per-stream
  input thresholds (gaba_input_threshold_z_harm_s/a/beta).
  Agent wiring: instantiated in REEAgent.__init__ when master switch is
  on; register_default_streams() called immediately. tick() invoked in
  agent.sense() right after LatentStack.encode() and BEFORE AIC, BLA/CeA,
  salience coordinator, etc. -- so all downstream consumers see the
  decayed latent state on the same tick (no one-step lag). reset() called
  from REEAgent.reset() per-episode.
  Backward compatible: use_gabaergic_decay=False by default; agent.gabaergic_decay
  is None and tick wiring is a no-op. Existing experiments unaffected.
  No trainable parameters. No phased training needed.
  Biological basis: GABAergic system as broadly-projecting tonic
  inhibitory neuromodulator (Vogt 2005, Sohal & Rubenstein 2019).
  Decay-as-regulator-layer (not per-stream update) is the architectural
  commitment: a single GABA tonic value modulates many cortical and
  subcortical sites in parallel. SD-036 implements this commitment.
  MECH-094: simulation_mode=True path returns input unchanged and does
  not advance counters (replay / DMN content not subject to waking decay).
  Validation experiment: V3-EXQ-475 queued (matched re-run of EXQ-471
  with use_gabaergic_decay=True + use_pag_freeze_gate=True; not a
  supersede; EXQ-471 retained as no-decay baseline).
  Design doc: REE_assembly/docs/architecture/sd_036_gabaergic_decay_regulator.md
  See SD-036, MECH-279, MECH-094, SD-010, SD-011, MECH-090, SD-012.

- MECH-279: pag.freeze_gate -- IMPLEMENTED 2026-04-22.
  Module: ree_core/pag/freeze_gate.py (PAGFreezeGate, PAGFreezeGateConfig,
  PAGFreezeGateOutput). Periaqueductal-gray-analog committed-freeze gate.
  Freeze is a *committed* behavioural state -- sustained motor immobility
  plus elevated autonomic arousal -- with its own duration and exit
  criterion. Biologically PAG-gated; freeze-promoting cells are themselves
  GABAergic (so SD-036 gates BOTH entry and exit).
  Logic:
    duration_above_threshold(t) -- ticks since z_harm_a first crossed
      duration_input_threshold (defaults 0.4); resets when z drops below
      that threshold OR on release. Increments only while gate inactive
      (per-cycle "fresh accumulation" semantic; each commit requires a
      new run-up).
    freeze_commit(t) = (z_harm_a(t) * duration_above_threshold(t))
                       > theta_freeze (default 2.0); strict-greater so
      e.g. z=1.0 sustained at duration=2 (product=2.0) does NOT commit.
    exit_threshold(t) = theta_freeze * gaba_tone(t)
    freeze_release    = active AND z < exit_threshold AND
                        ticks_in_freeze >= min_freeze_duration; OR
                        ticks_in_freeze >= max_freeze_duration (cap).
  Action constraint: when freeze_active, REEAgent.select_action()
  replaces the chosen action with a no-op one-hot (action class
  noop_class=0 by convention; matches action shape/dtype/device).
  Tick wired AFTER beta_gate.propagate() and BEFORE _last_action assignment
  so subsequent record_transition / E2_harm_a forward steps see the no-op.
  Config: REEConfig.use_pag_freeze_gate (bool, default False). 4 sub-
  knobs: pag_theta_freeze (2.0), pag_duration_input_threshold (0.4),
  pag_min_freeze_duration (0 -- no minimum), pag_max_freeze_duration
  (0 -- no cap; set positive for forced-release safety in smoke tests).
  Backward compatible: use_pag_freeze_gate=False by default; agent.pag_freeze_gate
  is None. Existing experiments unaffected.
  No trainable parameters. Pure arithmetic over scalars + small counters.
  No phased training needed.
  Biological basis: descending inputs from amygdala / hypothalamus /
  medial PFC converge on PAG freeze-promoting cells; freeze termination
  requires GABAergic inhibition to wane. Same neurotransmitter system
  gates BOTH entry (PAG freeze-cell commitment) and exit (SD-036 decay
  returning z_harm_a below exit_threshold). Architectural prediction:
  GABA agonists treat freeze catatonia (clinical observation as
  architectural consequence, not empirical add-on).
  MECH-094: simulation_mode=True path returns zeroed PAGFreezeGateOutput
  without updating internal state (replay / DMN content must not commit
  the agent into behavioural freeze).
  Validation experiment: V3-EXQ-475 (combined SD-036 + MECH-279
  diagnostic; under default theta_freeze=2.0 the gate is expected to be
  silent on EXQ-471 dynamics, but is wired so the substrate is exercised
  end-to-end).
  See MECH-279, SD-036, SD-011, MECH-090, MECH-094.

## MECH-269 Base Substrate -- Phase 1 (2026-04-22)
- MECH-269 base: hippocampal.per_stream_verisimilitude -- IMPLEMENTED 2026-04-22.
  Module: ree_core/hippocampal/module.py (HippocampalModule.update_per_stream_vs,
  HippocampalModule.reset_per_stream_vs, HippocampalModule._stream_value).
  Phase 1 of the V_s invalidation runtime (architecture doc:
  REE_assembly/docs/architecture/v_s_invalidation_runtime.md). Adds the
  observable per-stream verisimilitude foundation that Phase 2 (MECH-287
  broadcast invalidation trigger) and Phase 3 (MECH-284 staleness
  accumulator + MECH-269 anchor-reset hysteresis) will consume.
  Computation (Phase 1, identity-prediction proxy):
    For each registered stream s in config.per_stream_vs_streams:
      z_curr = LatentState[s] (or GoalState.z_goal for s=='z_goal',
                               LatentState.z_harm for s=='z_harm_s')
      err = ||z_curr - z_prev|| / (||z_curr|| + 1e-6)
      score = clip_[0,1](1 - err)
      V_s[s] = (1-tau)*V_s_prev[s] + tau*score   # EMA
    First observation seeds V_s[s] = 1.0 (perfect verisimilitude assumed)
    and caches z_curr; subsequent ticks compute the proxy.
  Forward-predictor routing (z_world -> ReafferencePredictor SD-007;
  z_harm_s -> HarmForwardModel SD-011) is RESERVED for Phase 2. Phase 1
  uses the identity proxy uniformly to keep HippocampalModule decoupled
  from per-stream predictor wiring; the dict is populated as an
  OBSERVABLE that downstream phases can consume.
  Config: HippocampalConfig.use_per_stream_vs (bool, default False),
  HippocampalConfig.per_stream_vs_tau (float, default 0.1),
  HippocampalConfig.per_stream_vs_streams (tuple, default
  ("z_world", "z_self", "z_harm_s", "z_harm_a", "z_goal", "z_beta")).
  Streams absent from the current LatentState / GoalState are silently
  skipped (no entry written to per_stream_vs).
  Agent wiring:
    REEAgent.sense() -- after new_latent.detach(), before return:
      if hippocampal.config.use_per_stream_vs:
        hippocampal.update_per_stream_vs(new_latent, goal_state=self.goal_state)
    REEAgent.reset() -- after MECH-279 PAG reset:
      if hippocampal.config.use_per_stream_vs:
        hippocampal.reset_per_stream_vs()
  Backward compatible: use_per_stream_vs=False by default; HippocampalModule
  exposes per_stream_vs={} and update_per_stream_vs() returns immediately.
  All 58 contract tests + 7 preflight tests pass with flag OFF
  (bit-identical to legacy).
  Activation smoke (2026-04-22, default agent + flag ON):
    Tick 1 (zero baseline obs): per_stream_vs = {z_world: 1.0,
      z_self: 1.0, z_beta: 1.0}. Streams z_harm_s/z_harm_a/z_goal absent
      because default REEConfig leaves harm streams and goal seeding off.
    Tick 2 (perturbed obs): per_stream_vs = {z_world: 0.958,
      z_self: 0.959, z_beta: 0.959} -- identity proxy responds to the
      change as designed.
    Reset: per_stream_vs = {} (cache cleared).
  No trainable parameters; pure arithmetic over latent norms. No
  phased training needed.
  MECH-094: hypothesis_tag is NOT yet checked in update_per_stream_vs()
  -- Phase 1 is invoked only from REEAgent.sense() (waking observation
  stream, never replay/simulation). Phase 2 will add the gate when
  replay paths begin to consume the V_s signal directly.
  Validation experiment: deferred to Phase 2/3 -- Phase 1 is substrate
  scaffolding for the MECH-287 trigger and MECH-284 staleness layers
  that follow. End-to-end EXQ-476 (re-run of EXQ-475 with full V_s
  invalidation circuit on) is the validation experiment for the
  combined cluster.
  Contract tests: tests/contracts/test_mech_269_per_stream_vs.py
    C1: default config backward-compat.
    C2: master switch OFF -> per_stream_vs stays empty.
    C3: master switch ON -> seeds at 1.0, drops on perturbation.
    C4: per-stream isolation -- a perturbation in one stream does not
        move other streams' V_s.
    C5: EMA correctness under repeated identical observations.
  Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
  See MECH-269, MECH-272 (state-gated routing -- Phase 3), MECH-284
  (staleness accumulator -- Phase 3), MECH-287 (broadcast trigger --
  Phase 2), SD-007/MECH-101 (ReafferencePredictor -- Phase 2 z_world
  routing), SD-011 (HarmForwardModel -- Phase 2 z_harm_s routing),
  MECH-094 (hypothesis_tag gate -- Phase 2 when replay consumes V_s).

## MECH-288 Event Segmenter -- Phase 2 (2026-04-22)
- MECH-288: hippocampal.event_segmenter -- IMPLEMENTED 2026-04-22.
  Module: ree_core/hippocampal/event_segmenter.py (EventSegmenter,
  BoundaryEvent, Scale, _PEThresholdDetector, _BOCPDGaussianDetector).
  Phase 2 of the V_s invalidation runtime (architecture doc:
  REE_assembly/docs/architecture/v_s_invalidation_runtime.md). Emits
  BoundaryEvent objects with nested outer.inner segment IDs at
  event-scale transitions. Downstream consumers are MECH-287
  (broadcast invalidation trigger) and MECH-269 anchor-reset
  hysteresis; the module queues BoundaryEvents on HippocampalModule
  for those consumers to drain.
  Canonical two-scale config (EventSegmenterConfig defaults):
    fast: pe_threshold on (z_world, z_self); pe_window_length=200,
          pe_threshold=0.65, tau=1, min_segment_length=2.
    slow: bocpd_gaussian on (z_goal,); hazard=1/40,
          posterior_threshold=0.5, bocpd_top_k=20, bocpd_prior_var=1.0,
          tau=40, min_segment_length=15.
  BoundaryEvent payload: segment_id_old, segment_id_new (both
  "outer.inner" strings), scale, posterior, sources (list[str]), t.
  Hierarchical rule: slow fire forces outer+=1, inner=0 and suppresses
  a same-tick fast event (slow owns the inner reset). Fast fire
  increments inner only. force_boundary(scale, reason) bypasses
  min_segment_length (supervised / scripted API hook).
  BOCPD implementation: Adams & MacKay 2007 recursion with Welford
  online variance per run. Top-k pruning keeps the posterior O(1).
  Underflow-robust: if every existing run-hypothesis assigns
  negligible log-probability (max(pred_log) < -20) to the observation,
  the regime is treated as a decisive change-point -- fire with
  posterior=1.0 and reseed the posterior. Mirrors the literal
  total<=0 underflow path.
  Config: HippocampalConfig.use_event_segmenter (bool, default False),
  HippocampalConfig.event_segmenter (EventSegmenterConfig; default
  canonical two-scale above). EventSegmenterConfig.scales is a list
  of EventSegmenterScaleConfig entries; emit_to defaults to
  ["mech_287_broadcast", "mech_269_anchor_set"]; scale_id_format
  "{outer}.{inner}"; slow_scale_name "slow".
  HippocampalModule: instantiates event_segmenter when flag is on;
  exposes _boundary_event_queue (List[BoundaryEvent]),
  drain_boundary_events() -> List[BoundaryEvent] (list + clear),
  reset_event_segmenter() (per-episode reset).
  Agent wiring:
    REEAgent.sense() -- after z_harm_a_prev cache, before per-stream
      V_s (MECH-269 Phase 1) update: if hippocampal.config.use_event_segmenter
      and hippocampal.event_segmenter is not None, builds a latent_dict
      over (z_world, z_self, z_harm, z_harm_s, z_harm_a, z_beta,
      z_goal) and calls event_segmenter.step(latent_dict, pe_dict=None,
      t=self._step_count). Emitted events appended to
      hippocampal._boundary_event_queue.
    REEAgent.reset() -- after MECH-269 per_stream_vs reset:
      if use_event_segmenter: hippocampal.reset_event_segmenter().
  Backward compatible: use_event_segmenter=False by default;
  event_segmenter is None; drain_boundary_events() returns []; all
  existing experiments unaffected. 65/65 contracts + 7/7 preflight
  PASS with flag OFF (bit-identical to legacy).
  Activation smoke (2026-04-22): default agent constructed with
  use_event_segmenter=True instantiates both scales; fresh
  current_segment_id() == "0.0"; boundary queue drains to [] on
  empty tick.
  No trainable parameters. Pure arithmetic (sliding z-score + BOCPD
  recursion). No phased training needed.
  MECH-094: hypothesis_tag is NOT checked inside the segmenter; the
  segmenter is invoked only from REEAgent.sense() (waking observation
  stream, never replay/simulation). MECH-094 gating for replay-driven
  segmentation is deferred to the Phase 3 consumer wiring.
  Contract tests: tests/contracts/test_mech_288_event_segmenter.py
    C1: default config backward-compat; event_segmenter is None when
        flag is off; drain queue empty.
    C2: pe_threshold silent on constant baseline, fires on 10x
        sustained spike.
    C3: bocpd_gaussian silent on stationary z_goal, fires on 10x
        regime shift.
    C4: hierarchical outer.inner correctness (slow -> outer+1,
        inner=0; fast -> inner+1).
    C5: force_boundary bypasses min_segment_length, posterior=1.0,
        source tagged "force:<reason>"; unknown scale -> ValueError.
    C6: BoundaryEvent payload invariants (posterior in [0,1], sources
        populated, t within window, segment_id_new != segment_id_old,
        both contain ".").
    C7: min_segment_length suppresses immediate re-fire
        (min_segment_length=5 caps fires at <=2 over 10 ticks).
  Validation experiment: deferred to Phase 3 -- Phase 2 is the
  substrate that emits boundary events. End-to-end validation
  (MECH-287 broadcast consumption + MECH-269 anchor-reset hysteresis)
  is scheduled with the Phase 3 wiring pass; V3-EXQ-476 (re-run of
  EXQ-475 with full V_s invalidation circuit on) remains the
  end-to-end validation experiment for the combined cluster.
  Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
  See MECH-288, MECH-269, MECH-287 (broadcast trigger -- Phase 3
  consumer), MECH-284 (staleness accumulator -- Phase 3 consumer),
  MECH-272 (state-gated routing -- Phase 3), MECH-094 (hypothesis_tag
  -- Phase 3 when replay consumes segments).

## MECH-287 Invalidation Trigger -- Phase 2 iv (2026-04-22)
- MECH-287: regulators.invalidation_trigger -- IMPLEMENTED 2026-04-22.
  Module: ree_core/regulators/invalidation_trigger.py (InvalidationTrigger
  + BroadcastEvent dataclass). Phase 2 iv of the V_s invalidation
  runtime (architecture doc: REE_assembly/docs/architecture/
  v_s_invalidation_runtime.md). Subscribes to MECH-288 BoundaryEvents
  emitted in agent.sense() and re-emits them as graded BroadcastEvent
  objects. Graded output: broadcast_strength = posterior * gain (NO
  binary thresholding of strength). Downstream consumers (MECH-269
  anchor-reset -- T3; MECH-284 staleness accumulator -- Phase 3) drain
  via HippocampalModule.drain_broadcast_events().

  VERDICT-3 ARCHITECTURAL COMMITMENT (option c, V_s foundation lit-pull
  SYNTHESIS verdict 3): the trigger is a BoundaryEvent subscriber, NOT
  an independent comparator. The upstream CA1/CA3 mismatch comparator
  stage (Vinogradova 2001; O'Mara 2009; Lisman & Grace 2005) -- per
  MECH-287's dual-component biological substrate in the claim entry --
  is collapsed HERE to a subscription on the MECH-288 boundary queue.
  The biological-substrate text in claims.yaml remains accurate (biology
  IS a two-stage loop); the implementation collapses ComparatorStage to
  a subscriber. Whether to refactor MECH-287's claim text to make this
  explicit is a downstream governance decision -- NOT resolved in this
  commit.

  Phasic/tonic guardrail (Aston-Jones & Cohen 2005; Clewett 2025 failure
  signature 2): rolling-mean tonic estimate over config.tonic_window
  past-tick aggregated posteriors. If the estimate (measured BEFORE the
  current tick) exceeds config.tonic_threshold, the whole tick's phasic
  broadcast is suppressed (each suppressed BoundaryEvent increments
  n_suppressed). Passive decay via rolling window: once high-frequency
  boundary activity stops, the estimate falls below threshold in
  tonic_window+1 quiet ticks and broadcast resumes.

  Config: InvalidationTriggerConfig (ree_core/utils/config.py) --
  gain=1.0, targets=("mech_269_anchor_set",), tonic_threshold=0.5,
  tonic_window=50. HippocampalConfig.use_invalidation_trigger (default
  False); HippocampalConfig.invalidation_trigger (default factory).

  BroadcastEvent payload: t, strength (posterior * gain), posterior
  (inherited from BoundaryEvent, in [0, 1]), targets (list from config),
  source_scale, source_segment_id_old, source_segment_id_new,
  source_sources (original BoundaryEvent.sources).

  HippocampalModule: instantiates invalidation_trigger when flag is on;
  exposes _broadcast_event_queue (List[BroadcastEvent]),
  drain_broadcast_events() -> List[BroadcastEvent] (list + clear),
  reset_invalidation_trigger() (per-episode reset of tonic history /
  counters / queue).

  Agent wiring:
    REEAgent.sense() -- immediately AFTER the event_segmenter.step()
      call and the _boundary_event_queue extend (so this tick's
      BoundaryEvents are visible). If use_invalidation_trigger is on
      AND the segmenter produced events, the trigger is ticked with
      them and the resulting BroadcastEvents are appended to
      hippocampal._broadcast_event_queue. If use_invalidation_trigger
      is on but use_event_segmenter is OFF, the trigger is ticked with
      an empty boundary list so its tonic history advances in lockstep
      with the clock -- no broadcasts can fire (C5 dissociation).
    REEAgent.reset() -- after reset_event_segmenter:
      if use_invalidation_trigger: hippocampal.reset_invalidation_trigger().

  Backward compatible: use_invalidation_trigger=False by default;
  invalidation_trigger is None; _broadcast_event_queue stays empty;
  drain_broadcast_events() returns []. Regression: 70/70 contracts +
  7/7 preflight PASS with flag OFF (bit-identical to pre-MECH-287 HEAD).
  Activation smoke (2026-04-22): default agent constructed with
  use_invalidation_trigger=True + use_event_segmenter=True instantiates
  InvalidationTrigger; reset clears tonic_estimate to 0.0; broadcast
  queue empty on construction.

  No trainable parameters. Pure arithmetic (rolling mean on boundary
  posteriors). No phased training needed.

  MECH-094: hypothesis_tag is NOT checked inside the trigger. The
  segmenter feeds only from REEAgent.sense() (waking observation
  stream). Forced BoundaryEvents via EventSegmenter.force_boundary()
  would flow through the trigger as real broadcasts -- intentional
  (caller is responsible for the MECH-094 gate at the force-boundary
  call site).

  Contract tests: tests/contracts/test_mech_287_invalidation_trigger.py
    C1: default config backward-compat; invalidation_trigger is None
        when flag is off; drain queue empty.
    C2: BoundaryEvent arrival fires BroadcastEvent with strength =
        posterior * gain; source payload preserved.
    C3: graded posterior -> graded broadcast across [0.01 .. 1.0]
        (NO binary threshold).
    C4: tonic guardrail suppresses next phasic under sustained high-
        activity period; reopens after tonic_window+1 quiet ticks.
    C5: verdict-3 dissociation -- with event_segmenter lesioned (no
        BoundaryEvents queued), trigger never fires a broadcast
        regardless of internal state (including synthetically elevated
        tonic history). This is the falsifiable tertiary prediction
        for MECH-288 and validates the option-c implementation choice.

  Validation experiment: deferred to Phase 3 (T3 wires the MECH-269
  anchor-reset consumer). V3-EXQ-476 (re-run of EXQ-475 with full V_s
  invalidation circuit on) remains the combined-cluster end-to-end
  validation experiment.
  Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
  See MECH-287, MECH-288 (upstream BoundaryEvent emitter), MECH-269
  (Phase 3 anchor-reset consumer), MECH-284 (Phase 3 staleness
  accumulator consumer), MECH-272 (Phase 3 state-gated routing).

## MECH-269 Anchor Sets -- Phase 2 (ii) (2026-04-22)
- MECH-269 Phase 2 (ii): hippocampal.anchor_sets -- IMPLEMENTED 2026-04-22.
  Module: ree_core/hippocampal/anchor_set.py (AnchorSet, Anchor, AnchorKey).
  Phase 2 (ii) of the V_s invalidation runtime. Scale-tagged hippocampal
  anchor store with dual-trace preservation (Bouton 2004) and k-consecutive
  hysteresis on V_s_anchor crossings. Consumes MECH-288 BoundaryEvents
  (via HippocampalModule) to install / remap anchors keyed on
  (scale, segment_id, stream_mixture).
  Key schema:
    AnchorKey = (scale: str, segment_id: str, stream_mixture: tuple[str, ...])
  Phase 2 stand-in for stream_mixture: tuple(sorted(per_stream_vs.keys()))
  at anchor-creation tick. Learned attribution head deferred to Phase 3
  (MECH-284); this gives a deterministic, observable stream-membership
  signature sufficient for the first end-to-end validation.
  Dual-trace routing (Bouton 2004): on remap, the outgoing active anchor
  on (scale, stream_mixture) is marked INACTIVE (not erased) and retained
  in all_anchors() for retrieval / replay consumers; excluded from
  active_anchors(). Erase is never the resolution path.
  Hysteresis: per-anchor below_threshold_streak counter on
  V_s_anchor = avg(V_s over mixture) - staleness (staleness monotonic in
  (tick - last_accessed) * staleness_rate, clipped at staleness_clip).
  Streak increments when V_s_anchor < reset_threshold; resets to 0 on any
  tick at-or-above threshold. At hysteresis_k consecutive below-threshold
  ticks (default 5), the active anchor is marked inactive and returned.
  Config: HippocampalConfig.use_anchor_sets (bool, default False);
  HippocampalConfig.anchor_set (AnchorSetConfig, default factory).
  AnchorSetConfig: scales=("fast","slow"), reset_threshold=0.3,
  hysteresis_k=5, staleness_rate=0.005, staleness_clip=1.0,
  max_anchors_per_scale=128, subscribe_to_boundary_events=True.
  FIFO soft-cap: when active_per_scale exceeds max_anchors_per_scale, the
  oldest (smallest created_at) active anchor in that scale is marked
  inactive. Inactive anchors are preserved.
  HippocampalModule: instantiates anchor_set when flag is on; exposes
  tick_anchor_set(latent_state, events) and reset_anchor_set(). Stream
  mixture is built as tuple(sorted(self.per_stream_vs.keys())) at tick
  time (populated earlier in the same sense() tick by MECH-269 Phase 1).
  Agent wiring:
    REEAgent.sense() -- after per_stream_vs update, with the current
      tick's events list (local var from the event_segmenter branch,
      empty if segmenter is off or fired nothing): if use_anchor_sets
      is on, hippocampal.tick_anchor_set(new_latent, events) is called.
      tick_anchor_set consumes the events (write_anchor per registered
      scale, dual-trace remap internally) then advances
      tick_hysteresis(per_stream_vs).
    REEAgent.reset() -- after reset_invalidation_trigger:
      if use_anchor_sets: hippocampal.reset_anchor_set().
  Public API: write_anchor(scale, segment_id, stream_mixture, z_world)
  -> Anchor; get_anchor(...) -> Optional[Anchor] (refreshes last_accessed);
  mark_inactive(scale, stream_mixture) -> Optional[Anchor];
  reset_region(scale, stream_mixture, new_segment_id, z_world) -> Anchor
  (dual-trace remap; mark_inactive + write_anchor in one call);
  tick_hysteresis(per_stream_vs) -> List[Anchor] (fired this tick);
  consume_boundary_events(events, z_world, stream_mixture) -> List[Anchor]
  (skips scales not in config.scales; skips when z_world is None);
  active_anchors(scale=None) -> List[Anchor]; all_anchors(scale=None)
  -> List[Anchor]; reset() (per-episode: clears active + inactive +
  tick counter).
  Backward compatible: use_anchor_sets=False by default; HippocampalModule.
  anchor_set is None; tick_anchor_set / reset_anchor_set are no-ops.
  85/85 preflight + contracts PASS with flag OFF (bit-identical to
  pre-anchor-set HEAD). Contract tests all pass with flag ON.
  No trainable parameters. Pure arithmetic over latent norms + tick
  counters + detached z_world clones. No phased training needed.
  MECH-094: write_anchor is invoked only from HippocampalModule.tick_anchor_set,
  which is called from REEAgent.sense() (waking observation stream).
  Simulation / replay paths must not route through tick_anchor_set.
  hypothesis_tag gating is therefore achieved by call-site scoping, not
  by an inline tag check (same pattern as MECH-269 Phase 1, MECH-288,
  MECH-287).
  Contract tests: tests/contracts/test_mech_269_anchor_set.py
    C1: default config backward-compat; use_anchor_sets defaults False;
        HippocampalModule.anchor_set is None; tick/reset hooks no-op.
    C2: BoundaryEvent on registered scale installs active anchor with
        correct (scale, segment_id_new, stream_mixture) key; unregistered
        scale ignored.
    C3: second BoundaryEvent on same (scale, stream_mixture) family
        marks prior anchor INACTIVE (not erased); prior retained in
        all_anchors(); exactly one active anchor on the family.
    C4a: k-1 below-threshold ticks then at-or-above resets streak;
         anchor stays active.
    C4b: k consecutive below-threshold ticks fire the reset on the k-th
         tick; inactive anchor retained (dual-trace).
    C5: reset_region marks current active inactive and installs new
        active; both retained in all_anchors().
    C6: per-episode reset() clears active + inactive anchor stores and
        resets the internal tick counter.
    Plus 2 integration smoke tests verifying agent-level flag OFF is
    no-op and flag ON installs anchors via tick_anchor_set with
    stream_mixture drawn from sorted per_stream_vs keys.
  Validation experiment: deferred to V3-EXQ-476 (combined cluster end-
  to-end validation with the full V_s invalidation circuit on). No
  standalone Phase 2 (ii) EXQ is queued -- approved by user 2026-04-22
  in favour of the combined-cluster validation.
  Design doc: REE_assembly/docs/architecture/hippocampal_anchor_selection.md
  See MECH-269, MECH-288 (BoundaryEvent source), MECH-287 (Phase 3
  broadcast consumer for remap), MECH-284 (Phase 3 staleness accumulator
  successor to the local proxy), MECH-272 (Phase 3 state-gated routing),
  MECH-094 (waking-stream call-site scoping).

## MECH-269 Per-Region V_s Readout -- Phase 2 (iii, T4) (2026-04-22)
- MECH-269 Phase 2 (iii, T4): hippocampal.per_region_verisimilitude --
  IMPLEMENTED 2026-04-22. Module: ree_core/hippocampal/module.py
  (HippocampalModule.update_per_region_vs,
  HippocampalModule.apply_invalidation_broadcasts_to_regions,
  HippocampalModule.reset_per_stream_vs extension).
  Promotes the flat per_stream_vs[stream] -> float readout to a
  per-region dict per_region_vs[(scale, segment_id)][stream] -> float
  keyed on AnchorSet (Phase 2 ii) active anchor keys. V_s foundation
  lit-pull verdict 3: per-stream V_s is the projection-readout of the
  integrated mixed-selectivity code; per-region keying provides the
  scale/segment partition so downstream consumers (MECH-284 staleness
  accumulator Phase 3; replay prioritisation; BG / E3 policy
  modulation) can query V_s for a specific region without collapsing
  across all active regions.
  Computation (Phase 1 identity-proxy parity, scoped per region):
    For each active anchor a on (scale, segment_id, stream_mixture):
      region_key = (scale, segment_id)  # stream_mixture dropped for readout
      for stream_name in config.per_stream_vs_streams:
        z_curr = LatentState[stream_name] (or GoalState.z_goal)
        z_prev = self._prev_region_stream_values[region_key][stream_name]
        if z_prev is None: V_s[region_key][stream_name] = 1.0 (seed)
        else:
          err = ||z_curr - z_prev|| / (||z_curr|| + 1e-6)
          score = clip_[0,1](1 - err)
          V_s[region_key][stream_name] = (1-tau)*prev_vs + tau*score
        z_prev <- z_curr
    Regions whose active anchor has disappeared since the previous tick
    (hysteresis mark_inactive from tick_hysteresis, FIFO cap eviction,
    or an earlier apply_invalidation_broadcasts_to_regions call this tick)
    are pruned from per_region_vs and _prev_region_stream_values.
  Invalidation broadcast reset path (MECH-287 consumer):
    apply_invalidation_broadcasts_to_regions(broadcasts) iterates
    BroadcastEvents; for each bcast on (source_scale, source_segment_id_old),
    drops per_region_vs[(scale, segment_id_old)] and mark_inactive's the
    matching active anchor. This is the T3 hysteresis-shortcut reset
    path described in the design doc: k=5 hysteresis is the passive
    path; broadcasts are the explicit-reset path. Idempotent: a
    second broadcast on an already-reset region returns [] and is
    otherwise a no-op.
  Config: HippocampalConfig.use_per_region_vs (bool, default False).
    Orthogonal to use_per_stream_vs -- per-region is a refinement,
    not a replacement; both can be on simultaneously. Requires
    use_anchor_sets=True to do anything (no-op without an anchor set
    to query). Per-stream tau shared with flat path via
    per_stream_vs_tau. Per-stream set shared via per_stream_vs_streams.
  State: per_region_vs: Dict[Tuple[str,str], Dict[str,float]] and
    _prev_region_stream_values: Dict[Tuple[str,str], Dict[str,Tensor]]
    on HippocampalModule. Both cleared by reset_per_stream_vs() on
    episode boundaries (extended in this pass).
  Agent wiring: REEAgent.sense(), immediately after tick_anchor_set
  (which consumes BoundaryEvents and advances hysteresis against
  per_stream_vs):
    if use_per_region_vs:
      broadcasts = list(hippocampal._broadcast_event_queue)  # peek, not drain
      if broadcasts: apply_invalidation_broadcasts_to_regions(broadcasts)
      update_per_region_vs(new_latent, goal_state=self.goal_state)
  Peek-not-drain on the broadcast queue: downstream Phase 3 consumers
  (MECH-284 staleness accumulator) still see the events after this
  tick. The dual consumption (tick_anchor_set's consume_boundary_events
  AND apply_invalidation_broadcasts_to_regions) is intentional: the
  first is the dual-trace remap path keyed on (scale, stream_mixture);
  the second is the explicit safety net keyed on
  (source_scale, source_segment_id_old).
  Backward compatible: use_per_region_vs=False by default. With flag
  OFF, per_region_vs stays empty, update_per_region_vs / apply_invalidation_broadcasts_to_regions
  are no-ops, reset_per_stream_vs extension is inert. 85/85 preflight
  + contracts PASS unchanged (bit-identical to pre-T4 HEAD).
  Activation smoke (2026-04-22, full MECH-269 stack ON + force_boundary):
    per_stream_vs populated as before (Phase 1);
    per_region_vs keys: [('fast', '0.1')] after one forced fast
    boundary; region V_s values non-trivial (0.89 / 0.97 / 0.96 under
    mild latent drift); active anchors reflect the new region.
    Flag OFF: per_region_vs stays empty across multiple sense() ticks.
  No trainable parameters. Pure arithmetic over latent norms + dict
  membership. No phased training needed.
  MECH-094: update_per_region_vs / apply_invalidation_broadcasts_to_regions
    are invoked only from REEAgent.sense() (waking observation stream).
    Simulation / replay paths must not route through sense(), so the
    hypothesis_tag gate is achieved by call-site scoping (same pattern
    as MECH-269 Phase 1 / Phase 2 ii, MECH-288, MECH-287).
  Contract tests: tests/contracts/test_mech_269_per_region_vs.py
    C1: default flag False; with flag OFF update_per_region_vs is a
        no-op even when anchors are present; flat per_stream_vs path
        continues to work.
    C2: per_region_vs populates on BoundaryEvent-installed anchor;
        (scale, segment_id_new) key present; streams seeded at 1.0.
    C3: cross-region isolation -- two active anchors on distinct
        (scale, segment_id) keys; marking one inactive prunes only
        that region's entry; the other region's cached V_s untouched.
    C4: MECH-287 broadcast on (source_scale, source_segment_id_old)
        drops only that region's entry AND mark_inactives the matching
        anchor; other region remains active. Idempotent.
    C5: hysteresis_k=5 honoured -- 5 consecutive below-threshold
        tick_hysteresis calls fire mark_inactive; subsequent
        update_per_region_vs prunes the per_region_vs entry.
    Plus 1 integration smoke test for reset_per_stream_vs clearing
    both flat and per-region state.
  Validation experiment: deferred to V3-EXQ-476 (combined cluster
    end-to-end validation with the full V_s invalidation circuit on;
    tests MECH-288 falsifiable prediction secondary -- z_goal / z_world
    broadcast events should preferentially reset their home-region V_s
    entries rather than peer regions). No standalone T4 EXQ queued in
    this pass (follow-up task per user spec).
  Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
  See MECH-269, MECH-288 (BoundaryEvent source via tick_anchor_set),
    MECH-287 (broadcast reset path consumer), MECH-284 (Phase 3
    staleness accumulator successor; reads per_region_vs), MECH-272
    (Phase 3 state-gated routing), MECH-094 (call-site scoping).

## MECH-284 Staleness Accumulator + MECH-269 Online Hysteresis -- Phase 3 (2026-04-24)
- MECH-284: hippocampal.staleness_accumulator -- IMPLEMENTED 2026-04-24.
  Module: ree_core/hippocampal/staleness_accumulator.py (StalenessAccumulator,
  StalenessAccumulatorConfig, RegionKey). Phase 3 of the V_s invalidation
  runtime (architecture doc: REE_assembly/docs/architecture/
  v_s_invalidation_runtime.md). Region-indexed residual schema-staleness
  accumulator. Integrates MECH-287 BroadcastEvents against the currently
  active MECH-269 anchor set with an attribution weight, decays per tick,
  and exposes a getter consumed by MECH-269 online anchor-reset
  hysteresis (the online arm of the dual-readout; MECH-285 offline
  sleep-priority arm is deferred).
  Region key: (scale, segment_id) -- stream_mixture dropped to match the
  Phase 2 (iii, T4) per_region_vs partition. One (scale, segment_id)
  region reachable by multiple stream_mixture families has its staleness
  merged on the region bucket.
  Operational definition (per claims.yaml refinement 2026-04-22):
    for each schema region r in active_anchor_set(t):
      if MECH-287 trigger(t):
        staleness[r] += attribution_weight(r, source_streams) * magnitude
      staleness[r] *= leak_factor
  Attribution modes (config.attribution_mode):
    "equal"          -- 1/N uniform credit across N active anchors.
    "stream_overlap" -- |source_sources & stream_mixture| /
                        max(|source_sources|, 1) per anchor; cheap
                        cosine-similarity surrogate over stream-name
                        sets. Anchor with zero overlap gets zero credit.
  Staleness is clipped at config.staleness_clip (default 1.0) so
  V_s_anchor = V_s(r) - staleness[r] stays in [-1, 1] whether the
  Phase 2 proxy or Phase 3 lookup drives hysteresis.
  Config: HippocampalConfig.use_staleness_accumulator (bool, default
  False); HippocampalConfig.staleness_accumulator (StalenessAccumulatorConfig,
  default factory). StalenessAccumulatorConfig: leak_factor=0.995,
  attribution_mode="equal", staleness_clip=1.0, drop_epsilon=1e-6.
  MECH-269 online hysteresis swap:
    HippocampalConfig.use_mech284_hysteresis (bool, default False).
    When both use_staleness_accumulator AND use_mech284_hysteresis are
    True, AnchorSet.tick_hysteresis() receives a staleness_lookup
    callable pointing at StalenessAccumulator.lookup_by_anchor_key.
    V_s_anchor = V_s(r) - staleness_lookup(anchor_key). With
    use_staleness_accumulator ON but use_mech284_hysteresis OFF, the
    accumulator is populated as a diagnostic only; hysteresis continues
    to use the Phase 2 internal proxy ((tick - last_accessed) *
    staleness_rate).
  Integration site (HippocampalModule.tick_anchor_set):
    consume_boundary_events (MECH-269 Phase 2 ii) -> integrate broadcasts
    against active anchors (peek, not drain; MECH-287 consumers that
    run after tick_anchor_set still see the queue) -> tick_leak ->
    tick_hysteresis (with staleness_lookup if MECH-284 hysteresis is on).
    This ordering preserves the "this-tick broadcasts affect this-tick
    V_s_anchor check" semantic.
  HippocampalModule public API additions:
    integrate_staleness(broadcasts) -- explicit credit path for code
      that wants to integrate outside the tick_anchor_set cycle; no-op
      when accumulator is disabled. Applies leak after integration.
    reset_staleness_accumulator() -- per-episode reset of region map +
      diagnostic counters.
  Agent wiring (REEAgent):
    reset() -- after reset_anchor_set: if use_staleness_accumulator is
      on, hippocampal.reset_staleness_accumulator().
    sense() -- no additional call-site required: the existing
      tick_anchor_set call handles integration internally via a peek of
      the _broadcast_event_queue populated earlier in the same sense()
      tick by MECH-287.
  Backward compatible: use_staleness_accumulator=False by default;
    staleness_accumulator is None; tick_anchor_set follows the legacy
    Phase 2 path (no integration, no leak, no staleness_lookup). 91/91
    preflight + contracts PASS with flag OFF (bit-identical to pre-
    Phase-3 HEAD, 2026-04-24).
  Activation smoke (2026-04-24, two ARMs):
    ARM1 (use_staleness_accumulator=True, use_mech284_hysteresis=False):
      Two active anchors on (fast, 0.1) and (fast, 0.2) with distinct
      stream_mixtures; one synthetic BroadcastEvent with strength=1.0
      injected; tick_anchor_set called -> snapshot:
        (fast, 0.1): 0.4975, (fast, 0.2): 0.4975
      (0.5 equal credit * leak 0.995); stats: n_integrations=1,
      n_leak_ticks=1, n_regions=2, max_staleness=0.4975. Reset clears
      map + counters. PASS.
    ARM2 (use_staleness_accumulator=True, use_mech284_hysteresis=True):
      staleness_rate=0.0 (passive proxy off), hysteresis_k=3,
      reset_threshold=0.5, per_stream_vs held at 1.0. Inject staleness=0.9
      on region key each tick; tick_anchor_set ticks 3 times -> anchor
      marked inactive on tick 3 (below_threshold_streak=3). Confirms
      staleness_lookup path is exercised under the swap. PASS.
  No trainable parameters. Pure float arithmetic + dict state. No phased
  training needed.
  MECH-094: integrate() is invoked only from HippocampalModule.integrate_staleness
    and HippocampalModule.tick_anchor_set, both of which are called from
    REEAgent.sense() (waking observation stream). Simulation / replay
    paths must not route through these; hypothesis_tag gating is achieved
    by call-site scoping (same pattern as MECH-269 Phase 1 / Phase 2 ii
    / Phase 2 iii, MECH-288, MECH-287).
  Validation experiment: V3-EXQ-478 queued (Phase 3 diagnostic ablation:
    OFF vs ON x 2 seeds; metrics freeze_recommit_count, anchor_reset_count,
    mean_staleness_peak, action_class_entropy). Also unblocks previously
    gated combined-cluster validations (V3-EXQ-445d, V3-EXQ-449c,
    V3-EXQ-455a, V3-EXQ-476/475 re-run).
  Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
  See MECH-284, MECH-269 (Phase 1 + 2 ii + 2 iii; online-arm consumer),
    MECH-287 (broadcast event source), MECH-288 (boundary segmenter),
    MECH-285 (offline sleep-priority readout, deferred), MECH-272
    (Phase 3 state-gated routing), MECH-094 (call-site scoping).

## SD-037: Broadcast Override Regulator (orexin-analog) (2026-04-25)
- SD-037: regulators.broadcast_override -- IMPLEMENTED 2026-04-25.
  Module: ree_core/regulators/broadcast_override.py (BroadcastOverrideRegulator,
  BroadcastOverrideConfig). Third regulatory layer of the V3 control stack
  alongside 5-HT goal-pipeline gain (MECH-186/187/188) and SD-036
  GABAergic cross-stream decay. Orexinergic (hypocretin) hub analog: scalar
  override_signal in [0, 1] driven by SD-012 drive_level + sustained-threat
  rolling-window magnitude over z_harm, EMA-smoothed.
  Computation:
    sustained_threat = clip_[0,1]( rolling_mean(z_harm.norm, window) /
                                   sustained_threat_threshold )
    drive_input      = clip_[0,1]( drive_level )
    raw              = sigmoid( drive_weight*drive_input
                              + harm_weight*sustained_threat
                              - recruitment_threshold )
    override_signal  = clip_[0,1]( (1-decay_rate)*prev + decay_rate*raw )
  Consumed at three sites:
    PAG freeze-gate (MECH-279): exit_threshold scaled by
      (1 + alpha_override * override_signal). Strong override raises
      the bar for entering / staying in committed-freeze (orexin ->
      arousal / escape-from-freeze; Carter et al. 2009 LH -> PAG).
      PAGFreezeGateConfig.alpha_override (default 0.0; agent wires
      override_alpha_pag when both flags on). Override_signal passed
      explicitly per-tick into PAGFreezeGate.tick().
    SalienceCoordinator (SD-032a): update_signal("override_signal", ...)
      injection biases operating-mode aggregate toward external_task
      (registered affinity_weights["override_signal"] =
      {"external_task": override_salience_reweight_alpha}). MECH-261
      generalises MECH-094 here -- registry is the gating point.
    GoalState (SD-012): drive -> z_goal seeding amplified by
      effective_drive *= (1 + (override_goal_seeding_gain - 1) *
      override_signal). Implements "drive becomes action-orienting only
      when override system has recruited" semantic. Default gain 2.0
      means saturated override doubles the seeding multiplier.
  Config: REEConfig.use_broadcast_override (bool, default False).
    Sub-knobs: override_recruitment_threshold (0.5),
    override_alpha_pag (0.5; PAG exit-threshold scaling),
    override_salience_reweight_alpha (0.3; SalienceCoordinator affinity),
    override_drive_weight (1.0), override_harm_weight (1.0),
    override_sustained_threat_window (12 ticks),
    override_sustained_threat_threshold (0.4),
    override_decay_rate (0.05; ~20-tick EMA),
    override_goal_seeding_gain (2.0).
  Defaults are biologically defensible per orexin kinetics lit-pull
  (Mileykovskiy et al. 2005 LH burst firing 5-15 Hz on threat;
  Lee et al. 2005 LHA orexin neuron arousal-correlated activity;
  Karnani et al. 2020 sleep/wake state transitions; Johnson et al. 2012
  PAG-projecting orexin escape behaviours). Two flagged for sweep:
  recruitment_threshold and alpha_pag at low end.
  Agent wiring (REEAgent):
    __init__ -- after PAG instantiation: if use_broadcast_override is on,
      construct BroadcastOverrideConfig from sub-knobs and instantiate
      BroadcastOverrideRegulator. PAG freeze-gate config receives
      alpha_override = override_alpha_pag when both flags are on
      (else 0.0 -- no-op). SalienceCoordinator (if present) gets
      affinity_weights["override_signal"] registered.
    sense() -- after SD-036 GABAergic decay tick: if broadcast_override
      is not None, tick(drive_level=goal_state._last_drive_level,
      z_harm_norm=z_harm.norm, simulation_mode=hypothesis_tag).
      One-step latency on drive_level read is intentional: the
      goal_state value reflects the previous tick's effective_drive,
      which is the post-pACC-bias drive. No double counting.
    select_action() -- before salience.tick(): inject
      update_signal("override_signal", broadcast_override.override_signal).
      PAG.tick() receives override_signal explicitly each cycle so
      exit_threshold scaling responds on the same tick.
    update_z_goal() -- after pacc.effective_drive: amplify effective_drive
      by (1 + (override_goal_seeding_gain - 1) * override_signal),
      clipped to [0, 1].
    reset() -- per episode: broadcast_override.reset() clears threat
      window, EMA state, and diagnostics.
  Backward compatible: use_broadcast_override=False by default;
    agent.broadcast_override is None; PAG receives alpha_override=0.0;
    salience signal slot is no-op; goal seeding multiplier=1.0. 95/95
    contracts PASS with flag OFF (bit-identical to pre-SD-037 HEAD,
    2026-04-25).
  Activation smoke (2026-04-25):
    Flag OFF: agent.broadcast_override is None.
    Flag ON: regulator instantiates with default config; one tick at
      drive=0.9, harm=0.6 produces override_signal=0.040 (sigmoid raw
      ~0.81 EMA-smoothed at decay_rate=0.05). 50 ticks of sustained
      load climb to 0.7431.
    MECH-094: simulation_mode=True returns cached signal unchanged
      (no threat-window advance, no EMA update).
    PAG with both flags on: alpha_override=0.5 wired correctly.
  No trainable parameters. Pure scalar arithmetic. No phased training.
  Biological basis: orexinergic (hypocretin) hub in lateral
    hypothalamus (LH). Persistent depletion (SD-012) plus sustained
    nociceptive signal (z_harm window) recruits LH orexin neurons;
    broad projections (PAG, BLA, LC, VTA, mPFC) gate downstream
    arousal / escape / motivational systems. Lit-pull synthesis:
    REE_assembly/evidence/literature/targeted_review_orexin_kinetics/
    synthesis.md.
  Failure-mode predictions (validation EXQ acceptance criteria):
    PWS-hyperphagia analog: saturated override (chronic high
      drive + harm) -> >=2x approach-commit rate vs balanced arm.
    Narcolepsy/cataplexy analog: lost override (regulator OFF
      under threat) -> <30% approach-commit vs balanced arm.
    Catatonic lock-in (V3-EXQ-471): with SD-036 + SD-037 ON, the
      orexin-analog raises PAG exit_threshold under sustained
      drive+harm so freeze releases instead of persisting.
  MECH-094: simulation_mode argument on tick(); when True, cached
    signal returned unchanged and no state advances. Replay / DMN
    content cannot recruit the override system.
  Validation experiment: V3-EXQ-483 queued (4-arm factorial
    {SD-036, SD-037} x {OFF, ON}, 3 seeds, ~50 min/arm; PWS /
    narcolepsy acceptance criteria from lit-pull synthesis).
  Design doc: REE_assembly/docs/architecture/sd_037_broadcast_override_regulator.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_orexin_kinetics/
  See SD-037, SD-036, MECH-279, SD-012, SD-032a, MECH-261, MECH-094.

## Sleep Aggregation Cluster Phase A: Scaffolding (2026-04-25)
- Sleep cluster Phase A: scaffold ree_core/sleep/ package -- IMPLEMENTED 2026-04-25.
  Module: ree_core/sleep/__init__.py, ree_core/sleep/phase_manager.py.
  New SleepPhase enum (6 phases: WAKING/SLEEP_ENTRY/SWS_ANALOG/PHASE_SWITCH/REM_ANALOG/
  WRITEBACK; only WAKING/SWS_ANALOG/REM_ANALOG visited in Phase A), SleepCycleState
  dataclass, and SleepLoopManager that wraps the existing SD-017 surface
  (REEAgent.run_sleep_cycle / enter_sws_mode / run_sws_schema_pass / enter_rem_mode /
  run_rem_attribution_pass / exit_sleep_mode -- pre-existing per SD-017).
  Master flag use_sleep_loop (default False) + sleep_loop_episodes_K (default 1) +
  sleep_loop_require_passes (default True) wired through REEConfig + REEConfig.from_dims().
  Manager instantiated in REEAgent.__init__ when flag is on; notify_episode_end() called
  at the start of REEAgent.reset() BEFORE per-episode resets so sleep operates on the
  final waking state.
  Validation: 8/8 new contract tests PASS (test_sleep_phase_a_scaffolding.py covering
  import, default backward-compat, master-OFF no instantiation, K=1 cycle drive,
  K=3 fires-on-third, no-substrate refusal, force_cycle, phase returns to WAKING).
  Full suite: 103/103 contracts + 7/7 preflight PASS -- bit-identical OFF guarantee
  holds. Phase A is no-op-consumer scaffolding only; Phases B-E layer additional
  master flags on top.
  See SD-017, MECH-272, MECH-273, MECH-275, MECH-285.
  Design doc: REE_assembly/docs/architecture/sleep_aggregation_cluster.md

## Sleep Aggregation Cluster Phase B: MECH-285 SleepReplaySampler (2026-04-25)
- MECH-285: sleep.replay_sampler -- IMPLEMENTED 2026-04-25.
  Module: ree_core/sleep/replay_sampler.py (SleepReplaySampler).
  At SLEEP_ENTRY freezes StalenessAccumulator.snapshot(), then draws N seeds from
  AnchorSet.all_with_dual_trace() (active + inactive, Bouton 2004 dual-trace
  preserved) with softmax(staleness/temperature) priority. Stateless within cycle;
  uniform-fallback when no accumulator (mech285_allow_uniform_fallback=True default).
  Config: REEConfig.use_mech285_sampler (master, default False),
    mech285_draws_per_cycle (50), mech285_temperature (1.0),
    mech285_allow_uniform_fallback (True). All wired through from_dims.
  Agent wiring: REEAgent constructs sampler when master ON AND hippocampal.anchor_set
  exists (Phase B requires MECH-269 Phase 2 ii); accumulator optional.
  SleepLoopManager extended with replay_sampler + draws_per_cycle ctor args; _run_cycle
  enters SLEEP_ENTRY phase, freezes snapshot, runs draws, merges mech285_* diagnostics
  into SleepCycleState.last_metrics. Phase B is NO-OP CONSUMER -- draws land in metrics
  only (Phases C-E wire routing/aggregator/writeback).
  Added AnchorSet.all_with_dual_trace() alias.
  Validation: 10/10 new contract tests + 113/113 contracts + 7/7 preflight all PASS.
  Bit-identical OFF guarantee holds.
  See MECH-285, MECH-269, MECH-272, MECH-275, MECH-273.

## Sleep Aggregation Cluster Phase C: MECH-272 RoutingGate (2026-04-25)
- MECH-272: sleep.routing_gate -- IMPLEMENTED 2026-04-25.
  Module: ree_core/sleep/routing_gate.py (RoutingGate, RoutedEvent).
  State-conditioned channel weights {anchor_channel, probe_channel} that flip across
  SWS_ANALOG / REM_ANALOG / WAKING rows per the design-doc table.
  Config: REEConfig.use_mech272_routing (master, default False) + 6 sub-knobs:
    sws_anchor_weight, sws_probe_weight, rem_anchor_weight, rem_probe_weight,
    waking_anchor_weight, waking_probe_weight.
  Wired into SleepLoopManager: set weights at SLEEP_ENTRY (SWS row), at PHASE_SWITCH
  (REM row); call route() on each replay draw and surface routed counts as mech272_*
  diagnostics on SleepCycleState.last_metrics.
  Wired flag through REEAgent constructor. No downstream consumer wiring yet
  (HippocampalRouter / E1 ContextMemory consumer / aggregator land in Phases D-E).
  Validation: bit-identical waking with all flags OFF; weights flip across phases when
  ON; backward-compat with use_mech285_sampler ON + use_mech272_routing OFF preserved.
  Result: 10/10 Phase C contracts PASS, 7/7 preflight PASS, 123/123 full contracts PASS.
  See MECH-272, MECH-285, MECH-275, MECH-273, MECH-094 (mode-conditioning generalisation).

## Sleep Aggregation Cluster Phase D: MECH-275 BayesianAggregator (2026-04-25)
- MECH-275: sleep.bayesian_aggregator -- IMPLEMENTED 2026-04-25.
  Module: ree_core/sleep/bayesian_aggregator.py (BayesianAggregator,
  GaussianPosterior, PosteriorUpdate, BayesianAggregatorConfig).
  Per-domain per-region Gaussian posteriors over residuals; conjugate mean-and-variance
  update gated by RoutedEvent.probe_channel * probe_gain (probe<=0 skipped, counted as
  mech275_n_skipped_zero_probe); snapshot+decay contract (snapshot deep-copies live
  posteriors, decay_factor multiplies live variance per cycle); place-domain default
  with (scale, segment_id) region key matching MECH-284.
  Config: REEConfig.use_mech275_aggregator (master, default False) + 6 sub-knobs:
    mech275_domains, mech275_prior_mean, mech275_prior_variance,
    mech275_likelihood_variance, mech275_decay_factor, mech275_probe_gain.
  Wired into SleepLoopManager._run_cycle: SLEEP_ENTRY freezes evidence_snapshot from
  agent.hippocampal.staleness_accumulator.snapshot() (place-domain evidence = staleness
  scalar at routed anchor's region, falls back to 0.0 if absent); each routed draw in
  SWS pass calls bayesian_aggregator.update(routed, evidence, domain=aggregator_domain);
  at PHASE_SWITCH snapshot() fires BEFORE routing_gate.set_phase(REM_ANALOG) so the
  snapshot captures SWS-only posteriors (Phase E reads this); REM re-route loop applies
  same probe-channel-gated update; mech275_* metrics merged into
  SleepCycleState.last_metrics.
  REEAgent.__init__ extended with Phase D conditional construction block;
  SleepLoopManager extended with bayesian_aggregator+aggregator_domain ctor args.
  NO downstream writeback (Phase E / MECH-273 deferred until next pass).
  Validation: 10/10 new contract tests + 38/38 sleep phases A-D + 133/133 contracts +
  7/7 preflight all PASS. Bit-identical OFF guarantee holds. MECH-094 enforced via
  call-site scoping (aggregator only invoked from _run_cycle, never from waking path).
  See MECH-275, MECH-272, MECH-285, MECH-284, MECH-094.

## Sleep Aggregation Cluster Phase E: MECH-273 SelfModelAggregator (2026-04-25)
- MECH-273: sleep.self_model_writeback -- IMPLEMENTED 2026-04-25.
  Module: ree_core/sleep/self_model_aggregator.py (SelfModelAggregator,
  SelfModelAggregatorConfig). Subclass of MECH-275 BayesianAggregator specialised on
  SD-003 causal_sig posterior. offline_gradient_pass(e2_harm_s, replayed_regions,
  n_steps, domain='self', use_snapshot=True) reads posterior means from last_snapshot
  (SWS-only frozen copy at PHASE_SWITCH) when available; constructs synthetic
  (z_harm_s zeros, action one-hot round-robin) batch at E2_harm_s input dims; trains
  via Adam at waking_lr * offline_lr_scale for n_steps bounded MSE steps.
  MECH-094 exception scoped: optimiser constructed locally over e2_harm_s.parameters()
  only -- no other module's params touched. n_steps<=0 short-circuits to no-op; empty
  replayed_regions returns zero-loss diagnostics. Cumulative diagnostics
  (mech273_n_offline_passes/steps/sum_loss/last_offline_loss/n_offline_regions_consumed)
  and per-call (mech273_writeback_regions/n_steps/sum_loss/mean_loss).
  NEW API: StalenessAccumulator.partial_decay(replayed_regions, decay_factor=0.5) ->
  int multiplicatively decays only the supplied region keys (clamped [0,1], drops
  below drop_epsilon, dedupes input via 'seen' set).
  Config: REEConfig.use_mech273_self_model (master, default False) + 3 sub-knobs:
    mech273_offline_lr_scale (0.1), mech273_offline_n_steps (100),
    mech273_partial_decay_factor (0.5). All wired through from_dims.
  REEAgent.__init__: agent-level e2_harm_s construction (parallel to e2_harm_a) when
  config.latent.use_e2_harm_s_forward; sleep_self_model_aggregator instantiated when
  use_mech273_self_model AND e2_harm_s exist; passed to SleepLoopManager via 4 new
  ctor args (self_model_aggregator, self_model_offline_n_steps,
  self_model_partial_decay_factor, self_model_domain).
  SleepLoopManager._run_cycle: replayed_regions set accumulated during SWS+REM update
  loops via _extract_region_key helper (handles RoutedEvent.event.key tuple form and
  direct tuple form); AFTER agent.run_sleep_cycle() set phase WRITEBACK ->
  offline_gradient_pass(use_snapshot=True) -> staleness.partial_decay(replayed_regions,
  decay_factor=self_model_partial_decay_factor); writeback_metrics merged into
  SleepCycleState.last_metrics including mech273_partial_decay_n_regions and
  mech273_partial_decay_factor.
  SHY normalisation (MECH-120) explicitly out of V3 scope.
  Validation: 10/10 Phase E contracts + 150/150 (143 contracts + 7 preflight) all PASS.
  Bit-identical OFF guarantee holds.
  See MECH-273, MECH-275, MECH-272, MECH-285, MECH-284, MECH-094, SD-003, ARC-033.

## SD-016 Path 1: ContextMemory Diversification Loss (2026-04-25)
- SD-016 Path 1: e1.context_memory_diversification_loss -- IMPLEMENTED 2026-04-25.
  Module: ree_core/predictors/e1_deep.py (ContextMemory.compute_diversification_loss),
  ree_core/agent.py (REEAgent.compute_prediction_loss), ree_core/utils/config.py.
  EXQ-418d FAILed across all 4 write-path arms with attn_entropy_mean ~2.76 (uniform
  reference 2.7726) and bimodal seed pattern (seed 42 ~0.46 div, seeds 43/44 collapse
  <1e-4). Diagnosis: no gradient pressure for slot diversification -- read-side
  gradient through cue_terrain_loss + cue_action_loss alone cannot differentiate
  slots, and writes-only path is luck-dependent on init symmetry breaking.
  Path 1 substrate: explicit auxiliary diversification loss on ContextMemory.memory:
  mean squared off-diagonal cosine similarity over normalized slot vectors.
  ContextMemory.compute_diversification_loss() method added; weighted loss term added
  in REEAgent.compute_prediction_loss.
  Config: new sd016_diversification_weight float wired through E1Config + REEConfig
  + REEConfig.from_dims (default 0.0; backward compatible).
  Validation: V3-EXQ-418e 4-arm ablation queued (A0_off baseline, A1_writes_only
  replicates 418d, A2_div_only tests div alone, A3_writes_plus_div tests bootstrap;
  supersedes V3-EXQ-418d). Smoke verified: slot_div climbs 0.2->0.5->1.0 across arms;
  wiring confirmed.
  See SD-016, MECH-150, MECH-151, MECH-152, ARC-041, EXP-0155.
  Design doc: REE_assembly/docs/architecture/sd_016_writepath_v3_diversification_loss.md

## MECH-269b Symmetric V_s Gating on E1/E2 Cortical Rollouts (2026-04-26)
- MECH-269b: cortical_world_model.regional_verisimilitude_rollout_gating -- IMPLEMENTED 2026-04-26.
  Module: ree_core/regulators/vs_rollout_gate.py (VsRolloutGate, VsRolloutGateConfig).
  Read-side consumer of MECH-269 Phase 1 hippocampal.per_stream_vs at the cortical
  forward-prediction call sites. Two integration sites in agent.py:
    (a) _e1_tick: gate latent_state for E1 side BEFORE total_state cat, BEFORE
        e1(...) call AND BEFORE extract_cue_context(). Held streams substitute
        snapshot for current value into z_self / z_world (and z_goal via
        gate_stream when e1_goal_conditioned).
    (b) select_action E2_harm_a forward block: gate _harm_a_prev for E2 side
        BEFORE the per-tick e2_harm_a forward call. Held substitution prevents
        E2_harm_a from rolling forward off a stale-but-confident-looking
        affective stream.
  Snapshot semantics: refresh per-stream snapshot to latent[s].detach().clone()
  in agent.sense() (after update_per_stream_vs / update_per_region_vs) when
  V_s[s] >= vs_gate_snapshot_refresh_threshold (default 0.5). Hold (substitute
  snapshot) at the rollout call sites when V_s[s] < per-side threshold (default
  0.4 on both sides). 0.4-0.5 dead-band gives lightweight Schmitt-trigger
  hysteresis without a streak counter.
  Config: HippocampalConfig.use_vs_rollout_gating (master, default False);
  vs_gate_snapshot_refresh_threshold (0.5), vs_gate_e1_threshold (0.4),
  vs_gate_e2_threshold (0.4), vs_gate_streams (("z_world","z_self","z_harm_s",
  "z_harm_a","z_goal","z_beta")), vs_gate_unknown_stream_passes (True). All
  wired through REEConfig.from_dims. Per-stream override dicts
  (e1_threshold_per_stream / e2_threshold_per_stream) live on
  VsRolloutGateConfig and are not surfaced via from_dims (set on the gate
  config directly when needed for asymmetric per-stream tuning).
  Precondition: agent.__init__ raises ValueError if use_vs_rollout_gating=True
  but use_per_stream_vs=False (the gate has no V_s to read).
  Diagnostics on VsRolloutGate: per-stream held counts (e1, e2), per-stream
  refresh counts, snapshot store, last-tick held flags. Surfaced via
  get_diagnostics() for inclusion in experiment manifests; the V3-EXQ-490
  acceptance criteria read these counters directly (C1).
  Backward compatible: use_vs_rollout_gating=False by default; agent.vs_rollout_gate
  is None and every integration site is no-op. With flag ON but V_s seeded at
  1.0 the gate fires zero times -- bit-identical to flag-OFF in the well-aligned
  regime. Substrate-validation smoke 2026-04-26: 7/7 preflight + 143/143 contracts
  PASS with flag OFF; with flag ON and V_s seeded at 1.0, 5-tick run produced
  zero held substitutions and 5 snapshots. Forced low V_s (per_stream_vs[s]=0.1)
  correctly triggered held substitution on the E1 side.
  No trainable parameters. Pure dataclass-replace + scalar arithmetic. No phased
  training needed.
  Biological basis (lit-pull SYNTHESIS, evidence/literature/
  targeted_review_mech269b_vs_rollout_gating/): Bastos 2012 + Feldman & Friston
  2010 + Kanai 2015 (cortex-side per-stream precision-weighted PE gating);
  Ernst & Banks 2002 (psychophysical foundation for per-stream reliability-
  weighted integration); Adams 2013 + Lawson 2014 (aberrant-precision wired-
  but-inert clinical phenotype). Symmetric application of one V_s vector to
  both proposer and cortical forward predictors is genuinely novel
  architectural ground; no paper in the anchor list demonstrates the symmetric
  claim biologically (see evidence_quality_note in claims.yaml).
  MECH-094: handled by call-site scoping. Gate invoked only from waking paths
  (sense, _e1_tick, select_action). No hypothesis_tag check inside the gate
  primitive; same pattern as MECH-269 Phase 1 / Phase 2 ii / 2 iii, MECH-288,
  MECH-287, MECH-284.
  Validation experiment: V3-EXQ-490 queued. Q-040 factorial: ON_OFF vs ON_ON
  with use_broadcast_override + use_dacc + drive_weight=2.0 + full V_s
  invalidation circuit + use_vs_commit_release ON in both arms; only manipulated
  variable is use_vs_rollout_gating. Acceptance: C1 (gate fires > 0 holds),
  C2 (approach_commit_count > 0 in >=2/3 seeds; OFF reproduces EXQ-483 zero
  baseline), C3 (dacc_score_bias_mean > 0). PASS = C1 AND C2 AND C3. FAIL on
  C2/C3 with C1 PASSing -> Q-040 FAIL branch points evidence at MECH-295
  liking-bridge as dominant blocker. experiment_purpose=diagnostic.
  Design doc: REE_assembly/docs/architecture/mech_269b_vs_rollout_gating.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_mech269b_vs_rollout_gating/
  See MECH-269b, MECH-269 (parent V_s primitive), MECH-284 (online staleness arm),
  MECH-098 (reafference cancellation, one V_s signal source), Q-040 (factorial),
  MECH-295 (complementary candidate cause), SD-032b (dACC adaptive control,
  downstream consumer), SD-037 (broadcast override, fires correctly already),
  ARC-033 (E2_harm_s forward, future gate consumer), MECH-258 (E2_harm_a
  forward, current gate consumer), SD-016 (cue_action_proj reads gated z_world).

## MECH-269b + MECH-284 Staleness-into-Gate Wiring (Q-040b strong reading, 2026-04-29)
- MECH-269b + MECH-284: cortical_world_model.regional_verisimilitude_rollout_gating
  STALENESS-WIRING IMPLEMENTED 2026-04-29.
  Modules: ree_core/regulators/vs_rollout_gate.py (VsRolloutGateConfig.use_staleness_lookup,
  gate / gate_stream / _gate_value extended with per_stream_staleness kwarg);
  ree_core/hippocampal/module.py (HippocampalModule.compute_per_stream_staleness);
  ree_core/agent.py (REEAgent._refresh_vs_gate_staleness, cached per-tick;
  precondition raises on missing use_staleness_accumulator / use_anchor_sets).
  The 2026-04-26 substrate compared raw per_stream_vs[s] to a fixed threshold
  (default 0.4); EXQ-490/490b/490c all had to override that threshold to smoke
  values (0.85/0.85/0.95) to make the gate fire at realistic V_s readings.
  This update wires MECH-284 region staleness into the comparison:
    effective_vs = raw_vs - per_stream_staleness[s]
  with per_stream_staleness aggregated as
    staleness[s] = max over active anchors a where s in a.stream_mixture
                   of staleness_accumulator.lookup_by_anchor_key(a.key)
  (max captures worst-case region staleness the stream is exposed to). Cached
  once per waking tick; reused by all gate / gate_stream call sites in the
  same tick.
  Config: REEConfig.from_dims(use_vs_gate_staleness_lookup=False default).
  When True, agent build raises ValueError unless use_staleness_accumulator
  AND use_anchor_sets are also True.
  Backward compatible: flag-OFF gate behaviour bit-identical to legacy raw-V_s
  path. 191/191 preflight + contracts PASS with the wiring landed.
  Q-040b strong reading: V3-EXQ-490c successor (490d) can drop the smoke
  threshold override and verify the hold path fires only when sustained
  MECH-287 broadcasts have accumulated region staleness. C4 severance arm
  (use_vs_gate_staleness_lookup=False vs True at matched thresholds) becomes
  the falsifiable test of the strong reading.
  MECH-094: aggregator + refresh helper are call-site-scoped to waking
  paths (_e1_tick); replay / simulation paths do not invoke them.
  Contract tests: tests/contracts/test_mech_269b_vs_rollout_gate_staleness.py
    C1: VsRolloutGateConfig.use_staleness_lookup defaults False;
        HippocampalConfig.use_vs_gate_staleness_lookup defaults False.
    C2: flag OFF -- supplied per_stream_staleness ignored (raw-V_s path).
    C3: flag ON without dict -- falls back to raw V_s (staleness=0 default).
    C4: flag ON with dict pushing effective_vs below threshold -- hold fires
        and snapshot is substituted.
    C5: per-stream isolation -- staleness on z_world does not affect z_self.
    C6: HippocampalModule.compute_per_stream_staleness max-over-anchors with
        stream_mixture overlap.
    C7: diagnostics (vs_gate_staleness_lookup_calls,
        vs_gate_max_staleness_<stream>) populated and cleared by reset.
    C8: agent precondition raises on missing accumulator / anchor substrates.
  Validation experiment: V3-EXQ-490d (490c successor) -- queue separately
  per the 490c-successor-tree planning doc once the 490c result lands.
  Design doc: REE_assembly/docs/architecture/mech_269b_vs_rollout_gating.md
  (new section "MECH-284 staleness wiring (Q-040b strong reading,
  2026-04-29)").
  See MECH-269b, MECH-284, MECH-269 (Phase 1 / Phase 2 ii / Phase 2 iii),
  MECH-287 (broadcast trigger -- staleness source), MECH-094 (call-site
  scoping), Q-040, Q-040b.

## MECH-295 Drive -> Liking-Stream -> Approach Cue Bridge (2026-04-26)
- MECH-295 weak-reading bridge: regulators.mech295_liking_bridge -- IMPLEMENTED 2026-04-26.
  Module: ree_core/regulators/mech295_liking_bridge.py (MECH295LikingBridge,
  MECH295LikingBridgeConfig, MECH295LikingBridgeOutput). Wires the missing
  link between SD-012 drive amplification, the SD-014/SD-015 liking-stream
  substrate, and E3 / BG action selection. Without this bridge, drive
  amplification produces a passive z_goal latent without behavioural
  consequence (the EXQ-483 catatonic-lock signature: override_signal climbs
  to mean 0.563, PAG release ratio ON_ON / ON_OFF = 1.69, but
  approach_commit = 0.0 across all four arms).
  Two integration sites:
    (a) update_z_goal() -- after the existing SD-012 / SD-037 effective_drive
        computation and GoalState.update() call: when bridge is active and
        goal_state.is_active(), call
        bridge.compute_anticipatory_liking_write(effective_drive,
        goal_state.goal_norm()) -> non-zero scalar -> ResidueField.update_valence
        at the goal location (z_goal latent, NOT current z_world), component
        VALENCE_LIKING. This is the anticipatory cue-side pulse, distinct
        from the existing consummatory-contact write in update_liking().
    (b) select_action() -- after lateral_pfc + ofc score_bias composition,
        before e3.select(): build per-candidate first-step z_world
        summaries (reuse cand_world_summaries when lateral_pfc / ofc are on),
        compute per-candidate goal_proximity via GoalState.goal_proximity,
        call bridge.compute_approach_cue_score_bias(effective_drive,
        proximities) -> NEGATIVE [K] tensor (E3 lower-is-better, so liking
        favours approach by reducing the score), composed additively with
        existing dacc_score_bias.
  Weak-necessity reading commitment: baseline liking-stream activation is
  sufficient. Cue-side gain is a function of drive * goal_proximity --
  the "is the bridge intact?" surface, not drive * residue.liking which
  would be the level-coupled strong reading. Setting
  mech295_liking_to_approach_cue_gain=0.0 is the SEVERED-BRIDGE arm of
  the falsifiable test: drive elevated AND write side intact AND cue
  side severed -> approach_commit predicted to collapse.
  Config: REEConfig.use_mech295_liking_bridge (bool, default False).
    Sub-knobs: mech295_drive_to_liking_gain (float, 1.0; 0 disables write
    side), mech295_liking_to_approach_cue_gain (float, 0.5; 0 severs cue
    side), mech295_min_drive_to_fire (float, 0.1; drive floor below which
    bridge is silent), mech295_min_z_goal_norm_to_fire (float, 0.05; goal
    norm floor below which bridge does not fire). All wired through
    REEConfig.from_dims().
  Backward compatible: use_mech295_liking_bridge=False by default;
    agent.mech295_bridge is None; both integration sites are no-ops.
    154/154 contracts + 7/7 preflight PASS with flag OFF (bit-identical
    to pre-MECH-295 HEAD, 2026-04-26).
  Activation smoke (2026-04-26): default agent + flag ON + drive_weight=2.0
    + cfg.goal.z_goal_enabled=True + min_z_goal_norm_to_fire=0.001 + 30 ticks
    forced drive=0.8 + benefit=0.4 -> n_write_fires=30, n_cue_fires=4,
    final goal_norm=0.333. Severed-bridge arm (cue gain=0) -> bias_max_abs
    exactly 0.0 with write side still firing.
  No trainable parameters. Pure scalar arithmetic + per-candidate proximity
  read. No phased training needed.
  Biological basis: NAc shell hedonic hotspot (Pecina & Berridge 2005,
    Castro & Berridge 2014), ventral pallidum (Smith Berridge & Aldridge
    2011 -- the strongest direct mechanistic anchor: VP single-unit
    recording shows drive change recodes palatability before cue firing),
    Berridge & Kringelbach 2015 architectural articulation, Dickinson &
    Balleine 1994 foundational behavioural devaluation requires outcome
    re-experience. Strong-vs-weak necessity not arbitrated by literature;
    Pecina 2003 DAT-knockdown finding (more wanting, unchanged liking) is
    compatible only with the weak reading. Bridge commits to the WEAK
    reading provisionally per claims.yaml MECH-295. Lit-pull synthesis:
    REE_assembly/evidence/literature/targeted_review_mech295_liking_approach_bridge/
  MECH-094: simulation_mode argument honoured at both compute methods;
    when True, write returns 0.0 and cue returns zero score_bias and
    counters do not advance.
  Validation experiment: V3-EXQ-493 queued (six-part diagnostic: UC1
    module-importable, UC2 master-OFF no-op, UC3 30-tick env loop write
    fires, UC4 cue side produces monotone-negative bias, UC5 SEVERED-BRIDGE
    COLLAPSE -- cue gain=0 produces zero bias even at elevated drive +
    write intact, UC6 MECH-094 simulation gate). All 6 PASS via --dry-run
    smoke 2026-04-26. Behavioural EXQ-483-style approach_commit recovery
    deferred to a successor (combined-cluster after V3-EXQ-490 lands).
  Design doc: REE_assembly/docs/architecture/mech_295_drive_liking_approach_bridge.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_mech295_liking_approach_bridge/
  See MECH-295, SD-012 (homeostatic drive input), SD-014 (valence vector
    substrate -- VALENCE_LIKING component), SD-015 (z_resource encoder
    upstream of GoalState.update), MECH-117 (existing wanting/liking
    dissociation in REE benefit_eval_head vs z_goal_latent), ARC-036
    (hedonic hotspot anatomical substrate -- prerequisite), MECH-094
    (call-site scoping + simulation_mode argument), SD-037 (broadcast
    override, drives effective_drive that the bridge consumes), MECH-269b
    (complementary candidate cause for EXQ-483 wired-but-inert; Q-040
    factorial points evidence at this bridge if MECH-269b alone fails to
    recover approach_commit).

## SD-039 Dual-Trace Anchor Goal-Snapshot Payload -- Substrate Foundation (2026-04-26)
- SD-039 substrate-side: hippocampal.anchor_goal_snapshot_payload --
  IMPLEMENTED 2026-04-26 (substrate side only; module-level write-site
  population deferred). Module: ree_core/hippocampal/anchor_set.py
  (AnchorGoalPayload dataclass; Anchor.goal_payload field +
  Anchor.goal_match() helper; AnchorSet.write_anchor / mark_inactive /
  reset_region / consume_boundary_events accept optional goal_payload;
  AnchorSet.query_by_goal_match() helper). Resolves the substrate-side
  prerequisite for MECH-292 (ranked ghost-goal bank) and MECH-293
  (waking ghost-goal probes); both downstream consumers can now query
  preserved motivational payloads on dual-trace anchors instead of
  reasoning over staleness-only signatures.
  Architectural design (claims.yaml SD-039): "Current MECH-269 anchors
  preserve z_world and active/inactive status, while MECH-284 preserves
  a region-level staleness scalar. SD-039 adds a compact motivational
  payload to each anchor at write, remap, or invalidation time:
  z_goal_snapshot, wanting strength, arousal tag, and optional last_vs /
  staleness_at_write. The payload is preserved across mark_inactive, so
  inactive anchors remain queryable as blocked or deferred goal traces."
  Refresh-on-invalidate semantic: when a non-None goal_payload is
  supplied to mark_inactive or reset_region, the payload is written
  onto the outgoing anchor BEFORE inactivation. Existing payload is NOT
  cleared on inactivation -- inactive anchors retain motivational
  identity (the entire point of dual-trace preservation). On
  reset_region, the same payload is written to BOTH the outgoing
  inactive trace and the new active anchor (cause-of-blockage payload
  on the outgoing; new motivational state on the new active).
  AnchorGoalPayload fields:
    z_goal_snapshot: Optional[torch.Tensor] -- detached clone of z_goal
      at write time. None when no goal active.
    wanting_strength: float -- VALENCE_WANTING readout at the anchor
      location (or last cached drive*benefit proxy).
    arousal_tag: float -- BLA arousal-tag scalar at write time.
    last_vs: Optional[float] -- last V_s_anchor reading on the parent
      (scale, stream_mixture) family at write/remap/invalidate time.
    staleness_at_write: Optional[float] -- MECH-284 region staleness at
      write time.
    payload_written_step: int -- HippocampalModule tick index at write.
  Anchor.goal_match(current_z_goal) -> float: cosine similarity between
    stored z_goal_snapshot and supplied current_z_goal, clipped to
    non-negative (motivational-relevance is a non-negative signal, not
    a signed correlation). Returns 0.0 when payload is None, snapshot
    is None, current is None, or norms are degenerate.
  AnchorSet.query_by_goal_match(current_z_goal, threshold=0.0,
    scale=None, active_only=False) -> List[Tuple[Anchor, float]]:
    scans the dual-trace pool (active + inactive by default) and
    returns anchors paired with non-zero goal_match scores, sorted by
    score descending. This is the substrate hook MECH-292 will consume.
    SD-039 itself does NOT rank or implement the bank; ranking by
    ghost_priority ~ wanting * goal_match * staleness * recoverability
    lives in MECH-292's module (deferred).
    threshold=0.0 (default) excludes payload-less / norm-zero traces.
    threshold=-1.0 includes every anchor with a payload regardless of
    match (diagnostic path).
    active_only=True restricts to the active half of the dual-trace
    pool (legacy active_anchors() behaviour).
  Config: AnchorSetConfig.use_sd039_anchor_payload (bool, default False).
    Substrate-side flag. Module-level callers populate the payload from
    GoalState / VALENCE_WANTING / amygdala arousal tags when this flag
    is True; with flag OFF callers pass goal_payload=None and behaviour
    is bit-identical to pre-SD-039.
  Backward compatible: anchor.goal_payload defaults to None. Existing
    write_anchor / mark_inactive / reset_region call sites that omit
    the new goal_payload kwarg work unchanged. 164/164 contracts +
    7/7 preflight PASS with flag implicitly off (2026-04-26).
  No trainable parameters. Pure dataclass + cosine arithmetic.
  Out of scope this session (deferred follow-on):
    - Module-level write-site wiring: REEAgent / HippocampalModule
      should populate goal_payload from GoalState (z_goal_snapshot),
      ResidueField VALENCE_WANTING (wanting_strength), and amygdala
      arousal tags (arousal_tag) on anchor write/remap/invalidate.
      The substrate accepts the payload; the population layer is a
      separate session.
    - MECH-292 ranked ghost-goal bank computation.
    - MECH-293 waking ghost-goal probe budget allocation.
    - ARC-060 hybrid field+bank architectural framing.
    - Validation EXQ exercising the falsifiable test (after reward
      relocation or path blockage, inactive anchors on the formerly
      valid approach path retain non-zero goal_match with current
      z_goal while unrelated stale anchors do not).
  Biological basis (lit-pull SYNTHESIS, evidence/literature/
    targeted_review_ghost_goal_search/): Berridge 1998 + Barch 2010
    (persistent wanting / goal representations); Mattar & Daw 2018
    (utility-prioritised replay); Pfeiffer & Foster 2013 (goal-biased
    path search); Gillespie 2021 (broad / non-current trace
    reactivation); Muessig 2019 (one sequence generator across waking
    + offline modes); Berkowitz 1989 (frustration / unresolved goal
    persistence).
  MECH-094: substrate-side scope; the goal_match query and the
    goal_payload dataclass are passive. Population sites (deferred)
    will gate writes via simulation_mode / hypothesis_tag at the
    REEAgent / HippocampalModule call site (same call-site-scoping
    pattern as MECH-269 Phase 1 / 2 ii / 2 iii, MECH-288, MECH-287,
    MECH-284).
  Contract tests: tests/contracts/test_sd_039_anchor_payload.py 10/10
    PASS (S1 imports + symbol presence; S2 default backward-compat;
    S3 ON-payload-attached-on-write; S4 payload-survives-mark_inactive;
    S5 goal_match-zero-on-null-inputs; S6 goal_match-cosine-correctness;
    S7 query-returns-active-and-inactive-sorted; S8 query-empty-for-
    None-current; S9 reset_region-refreshes-payload-on-both-traces;
    S10 per-episode-reset-clears-payloads).
  Validation experiment: deferred until module-level write-site wiring
    lands. The substrate-side foundation is observable through
    contracts; behavioural validation requires the population layer.
  Design doc: REE_assembly/docs/architecture/sd_039_anchor_goal_payload.md
  See SD-039, MECH-269 (Phase 2 ii dual-trace anchor substrate being
    extended), MECH-216 (predictive wanting -- input to wanting_strength
    population), MECH-230 (z_goal latent structure -- z_goal_snapshot
    source), MECH-284 (region staleness -- staleness_at_write source),
    MECH-292 (downstream ghost-goal bank consumer), MECH-293 (downstream
    waking ghost-goal probe consumer), ARC-060 (hybrid field+bank
    architectural framing), MECH-094 (call-site scoping for population
    layer).

## MECH-292 Ranked Ghost-Goal Bank (2026-04-27)
- MECH-292: hippocampal.unresolved_goal_ghost_bank -- IMPLEMENTED 2026-04-27.
  Module: ree_core/hippocampal/ghost_goal_bank.py (GhostGoalBank,
  GhostGoalBankConfig, GhostGoalBankEntry). First downstream consumer of the
  SD-039 dual-trace anchor goal-snapshot payload (substrate landed 2026-04-26;
  population layer landed 2026-04-27 with V3-EXQ-494 6/6 PASS). Pure-
  arithmetic, non-trainable derived view over the existing AnchorSet.all_anchors()
  pool. The bank does NOT own state beyond a small per-call diagnostics cache;
  the anchor pool itself remains the source of truth. Per spec
  (REE_assembly/docs/architecture/mech_292_ghost_goal_bank.md): "Implementation
  is intentionally a derived view, not a persistent store: SD-039 already
  preserves the per-anchor payload; MECH-292 just arranges the existing data."
  Ranking formula (per anchor a clearing goal_match_floor):
    wanting        = a.goal_payload.wanting_strength
    goal_match     = a.goal_match(current_z_goal)             [SD-039 cosine]
    staleness      = staleness_accumulator.snapshot()[(scale, segment_id)]
                     when accumulator present, else
                     clip_[0,1]((current_tick - last_accessed) * staleness_proxy_rate)
    recoverability = clip_[0,1](a.goal_payload.last_vs)
                     when last_vs is not None, else
                     default_recoverability_when_unknown
    ghost_priority = w_w*wanting + w_m*goal_match + w_s*staleness + w_r*recoverability
  goal_match_floor (default 0.05) is the architectural rumination guard:
  anchors with no payload OR with goal_match below the floor are invisible
  to the bank entirely. Pure low-V_s chasing is excluded by construction.
  Default pool: include_inactive=True, include_active=False, scale=None
  (all scales). MECH-293 ghost-goal probes work primarily over inactive
  traces; diagnostic / replay-prioritisation consumers may flip include_active
  on. top_k caps the returned list at 32 by default.
  Config: HippocampalConfig.use_mech292_ghost_bank (bool, default False).
    Nested: HippocampalConfig.ghost_goal_bank_config (GhostGoalBankConfig,
    default factory). REEConfig.from_dims surfaces 6 sub-knobs:
    mech292_wanting_weight (1.0), mech292_goal_match_weight (1.0),
    mech292_staleness_weight (0.5), mech292_recoverability_weight (0.5),
    mech292_goal_match_floor (0.05), mech292_top_k (32). Other GhostGoalBankConfig
    fields (default_recoverability_when_unknown, include_inactive,
    include_active, scale, staleness_proxy_rate) are not surfaced through
    from_dims; set on the nested config directly when needed.
  HippocampalModule wiring (after staleness_accumulator block):
    instantiate GhostGoalBank when use_mech292_ghost_bank is on; raise
    ValueError if anchor_set is None (use_anchor_sets must be True) OR if
    anchor_set.config.use_sd039_anchor_payload is False (otherwise every
    anchor scores goal_match=0.0 and the bank degenerates to empty).
    Public API: rank_ghost_goals(current_z_goal) -> List[GhostGoalBankEntry]
    (returns [] when bank is None or current_z_goal is None);
    reset_ghost_goal_bank() per-episode reset of diagnostics cache (anchor
    pool is reset separately by reset_anchor_set()).
  Agent wiring (REEAgent.reset()): reset_ghost_goal_bank() called on
    episode boundary when use_mech292_ghost_bank is on. No agent.sense /
    select_action wiring -- MECH-293 will be the first behavioural consumer.
  Backward compatible: use_mech292_ghost_bank=False by default;
    hippocampal.ghost_goal_bank is None; rank_ghost_goals returns [].
    164/164 contracts + 7/7 preflight PASS with master OFF (bit-identical
    to pre-MECH-292 HEAD). Smoke (master ON + SD-039 ON + 60-tick episode +
    forced fast-scale boundaries every 8 ticks): 6 inactive anchors with
    populated payload, 6 admitted entries, max_priority 1.609,
    monotone-decreasing.
  No trainable parameters. Pure scalar arithmetic + cosine via
    Anchor.goal_match (SD-039 helper). No phased training needed.
  Falsifiable signature (per spec, behavioural validation deferred): in a
    reward-relocation or blocked-corridor task, anchors from the now-
    obstructed but still-valued path should rank above equally stale but
    goal-irrelevant anchors. Substrate-level dissociation (UC4 of V3-EXQ-496)
    confirmed: Phase A goal-inactive anchors all below floor; Phase B
    goal-active anchors admitted with goal_match component dominant on top
    entry.
  MECH-094: substrate-side scope. Bank reads payloads whose provenance was
    set by the SD-039 population layer (sense() always passes
    simulation_mode=False, so source anchors carry waking-stream provenance).
    The bank itself has no write path -- nothing to gate. Inherits whatever
    provenance the source anchors carry.
  Validation experiment: V3-EXQ-496 queued (5 sub-tests UC1-UC5 covering
    module / config / method exposure, master OFF no-op, ranking_fires,
    goal_irrelevant_excluded, component_breakdown_consistent). Mac
    2026-04-27: 5/5 PASS (39s). Behavioural validation lives in V3-EXQ-495
    (V3 full-completion gate) once MECH-293 wires propose_trajectories()
    to consume the bank.
  Design doc: REE_assembly/docs/architecture/mech_292_ghost_goal_bank.md
  See MECH-292 (parent claim), SD-039 (dual-trace payload substrate),
  MECH-216 (predictive wanting -- wanting_strength source), MECH-230
  (z_goal latent -- cosine query target), MECH-269 Phase 2 (ii) (anchor
  substrate), MECH-269 Phase 1/2 (iii) (per-stream / per-region V_s for
  last_vs), MECH-284 (region staleness accumulator), MECH-293 (downstream
  consumer -- waking ghost-goal probe search), ARC-060 (hybrid field+bank
  architectural framing).

## SD-039 Module-Level Write-Site Population Layer (2026-04-27)
- SD-039 population: hippocampal.anchor_goal_payload_population -- IMPLEMENTED 2026-04-27.
  Modules: ree_core/hippocampal/module.py (HippocampalModule.build_goal_payload,
  tick_anchor_set goal_payload kwarg, apply_invalidation_broadcasts_to_regions
  goal_payload kwarg); ree_core/agent.py (REEAgent.sense() builds payload once
  per tick and threads it through both write/remap and broadcast-invalidate
  call sites); ree_core/utils/config.py (REEConfig.from_dims accepts
  use_sd039_anchor_payload, propagates to AnchorSetConfig).
  Wires the deferred follow-on to the SD-039 substrate (landed 2026-04-26):
  REEAgent / HippocampalModule now populate AnchorGoalPayload from the
  current waking-stream signals at every anchor write / remap / invalidate
  site so that MECH-292 / MECH-293 consumers see live motivational state on
  both halves of the dual trace.
  Sourcing (build_goal_payload):
    z_goal_snapshot     <- goal_state.z_goal.detach().clone() when
                          goal_state.is_active(); None otherwise.
    wanting_strength    <- residue_field.evaluate_valence(z_world)[..., VALENCE_WANTING].mean()
                          when residue + valence_enabled; 0.0 otherwise.
    arousal_tag         <- bla_output.arousal_tag when supplied; 0.0 otherwise.
    last_vs             <- mean(self.per_stream_vs.values()) when non-empty;
                          None otherwise. Phase 2 ii proxy for the parent
                          (scale, stream_mixture) family V_s -- the payload
                          is shared across all anchors written this tick.
    staleness_at_write  <- max(staleness_accumulator.snapshot().values())
                          when MECH-284 accumulator is enabled; None otherwise.
                          Region-keyed; max-across-regions is the most
                          informative scalar for downstream MECH-292 ranking.
    payload_written_step <- agent._step_count (anchor_set._tick fallback).
  build_goal_payload returns None (skipping population entirely) when:
    - the AnchorSet substrate is disabled (anchor_set is None),
    - AnchorSetConfig.use_sd039_anchor_payload is False (master flag OFF),
    - simulation_mode=True (MECH-094 gate; replay/DMN paths must not
      populate payloads from waking signals).
  Wiring sites in agent.sense():
    1. After update_per_stream_vs (Phase 1 V_s populated): build payload.
    2. tick_anchor_set(latent, events, goal_payload=...): boundary-event
       write/remap path; consume_boundary_events forwards the payload to
       each per-event write_anchor (the dual-trace remap path internally
       writes the payload onto BOTH outgoing inactive trace and the new
       active anchor when same family is replaced).
    3. apply_invalidation_broadcasts_to_regions(broadcasts, goal_payload=...):
       MECH-287 broadcast-driven mark_inactive path; payload is refreshed on
       the outgoing anchor at the moment of broadcast invalidation.
    Hysteresis-fired mark_inactive (inside tick_hysteresis) does NOT refresh
    payload -- the prior payload is preserved as the cause-of-blockage trace
    per dual-trace semantics.
  Config: REEConfig.from_dims(use_sd039_anchor_payload=False) propagates to
  config.hippocampal.anchor_set.use_sd039_anchor_payload. Backward
  compatible: master flag default False; agent.sense build_goal_payload
  returns None; tick_anchor_set / apply_invalidation_broadcasts_to_regions
  receive goal_payload=None and behaviour is bit-identical to pre-SD-039.
  170/171 preflight + contracts PASS with population layer landed; the
  remaining failure is unrelated queue-housekeeping (V3-EXQ-418e / 490
  completion-record duplication).
  No trainable parameters. No phased training needed. ASCII-safe (no
  print() output added).
  MECH-094: build_goal_payload accepts simulation_mode argument; sense()
  passes simulation_mode=False (waking observation stream). Hysteresis
  invalidation has no fresh-state context and intentionally leaves the
  prior payload preserved.
  Validation experiment: V3-EXQ-494 6/6 PASS 2026-04-27 (UC1 module
  importable; UC2 master OFF no-op; UC3 population_fires 7/7 anchors with
  populated payloads, max_goal_match 0.9999; UC4 dual-trace preservation
  6 inactive + 1 active all carry payloads; UC5 falsifiable signature
  Phase A mean=0.0 vs Phase B mean=0.998 with 3/3 above 0.3; UC6 MECH-094
  simulation gate -- replay path produces zero anchors with populated
  payload). Validation script extends _step_episode helper to force
  MECH-288 fast-scale boundary events every 8 ticks via
  event_segmenter.force_boundary so the SD-039 contract is exercised
  without depending on stochastic boundary firing within the test window.
  Design doc: REE_assembly/docs/architecture/sd_039_anchor_goal_payload.md
  See SD-039 (parent claim), MECH-269 Phase 2 (ii) anchor substrate,
  MECH-287 broadcast trigger (invalidation site), MECH-284 staleness
  accumulator (staleness_at_write source), MECH-216 predictive wanting
  (wanting_strength source), MECH-230 z_goal structure (z_goal_snapshot
  source), MECH-292 / MECH-293 / ARC-060 (downstream ghost-goal
  consumers), MECH-094 (simulation gate).

## MECH-293 Waking Ghost-Goal Probe Search (2026-04-27)
- MECH-293: hippocampal.awake_ghost_goal_probe_search -- IMPLEMENTED 2026-04-27.
  Modules: ree_core/hippocampal/module.py (HippocampalModule.propose_trajectories
  extended; new private methods _propose_ghost_seeded + _mix_value_flat_with_ghost;
  diagnostic accessor get_last_propose_diagnostics); ree_core/predictors/e2_fast.py
  (Trajectory dataclass extended with hypothesis_tag: bool=False and
  metadata: Optional[Dict[str, Any]]=None fields); ree_core/agent.py
  (REEAgent._e3_tick threads current_z_goal=goal_state.z_goal into
  propose_trajectories when goal is active; record_committed_trajectory
  explicitly sets hypothesis_tag=False / metadata=None on the executed
  committed trajectory). Read-side consumer of MECH-292 ranked ghost-goal
  bank: extends propose_trajectories with a minority budget of CEM probes
  seeded around the highest-priority bank entries' anchor.z_world rather
  than the agent's current z_world. Each ghost trajectory carries
  hypothesis_tag=True and metadata={"source": "mech293_ghost_probe",
  "anchor_key": ..., "ghost_priority": ..., "goal_match": ...} for
  downstream provenance.
  Algorithm:
    1. n_ghost = clamp(round(n_total * mech293_ghost_fraction),
                      [mech293_min_ghost_candidates, mech293_max_ghost_candidates])
       bounded by len(bank.rank()).
    2. For each top entry: seed action-object distribution mean from
       _get_terrain_action_object_mean(anchor.z_world, e1_prior). Single
       noise draw (no inner CEM refit -- ghosts are exploratory probes,
       not optimised plans, so probe cost <= one value-flat sample).
    3. e2.rollout_with_world(z_self, anchor_z, actions, action_bias=...)
       produces the candidate trajectory; tag + metadata stamped.
    4. Mix with value-flat candidates per mech293_replace_lowest_ranked:
         True (default): drop the highest-cost (worst) value-flat
                          candidates, append ghosts at the tail. Preserves
                          downstream E3 selection cost; len(candidates)
                          stays at n_total.
         False: append ghosts on top of the value-flat pool (raises total
                count). Diagnostic-only path.
    5. Diagnostics dict surfaced on _last_propose_diagnostics:
       {mech293_n_ghost_proposed, mech293_n_ghost_admitted,
        mech293_max_ghost_priority, mech293_mean_goal_match_at_seed,
        mech293_reason in {"ok","no_z_goal","empty_bank","n_ghost_zero"}}.
  Config: REEConfig.use_mech293_ghost_probes (bool, default False) +
    sub-knobs: mech293_ghost_fraction (0.2), mech293_min_ghost_candidates
    (1), mech293_max_ghost_candidates (8), mech293_replace_lowest_ranked
    (True). All wired through REEConfig.from_dims.
  Precondition (raised on HippocampalModule.__init__):
    use_mech293_ghost_probes=True requires use_mech292_ghost_bank=True.
    The MECH-292 block transitively guarantees use_anchor_sets=True and
    AnchorSetConfig.use_sd039_anchor_payload=True, so only the bank
    flag needs explicit enforcement here. Loud-not-silent failure mode
    matches the MECH-292 / SD-039 precondition pattern.
  Backward compatible: use_mech293_ghost_probes=False by default;
    propose_trajectories returns value-flat candidates; new current_z_goal
    arg is ignored when MECH-293 is off; new Trajectory.hypothesis_tag
    and .metadata fields default to backward-compat values (False / None).
    record_committed_trajectory now constructs Trajectory with explicit
    hypothesis_tag=False + metadata=None so the executed committed
    trajectory drops any ghost provenance from the source proposal (the
    executed trajectory IS real, regardless of its origin -- spec
    requirement). 12/12 MECH-293 contracts + 183/183 full preflight +
    contracts PASS with flag OFF (bit-identical to pre-MECH-293 HEAD).
  Activation smoke (2026-04-27, V3-EXQ-497 5/5 PASS, 34s on Mac):
    UC1 module surface (config flags + methods + Trajectory fields all
      exposed with correct defaults); UC2 master-OFF no-op (n_ghost=0,
      diagnostics={}, all candidates default-clean); UC3 ghost branch
      fires (n_ghost_admitted=4, max_priority=1.61,
      mean_goal_match_at_seed=0.998, reason='ok'); UC4 hypothesis_tag
      preserved on every ghost + metadata complete + 28 value-flat
      candidates remain default-clean; UC5 budget arithmetic
      (round(0.25*8)=2 in [1,4] arm A; bank-size cap to 1 in arm B;
      min-floor wins over fraction=0.0 in arm C).
  No trainable parameters. Pure routing logic. No phased training needed.
  ASCII-safe (no print() output added).
  MECH-094: ghost trajectories carry hypothesis_tag=True for
    provenance-routing. CEM rollout itself does not write residue or
    anchors during proposal (those are observation-side paths in
    agent.sense()), so no inline gate is needed during proposal. At
    commit boundary, record_committed_trajectory explicitly strips the
    tag (and metadata) so the executed trajectory is treated as real
    for downstream backward-credit-sweep / valence-write paths.
    SD-039's build_goal_payload(simulation_mode=True) path returns None
    already (handled at the SD-039 layer). No new MECH-094 plumbing
    required at the MECH-293 layer.
  ARC-007 strict: ghost probes do NOT add a hippocampal value head.
    Goal-match enters via MECH-292's external ranking over stored
    payloads, which lives outside HippocampalModule. The proposer is
    still proposing trajectories without an internal value computation;
    the ghost-seeded ones are biased BY LOCATION (the anchor's z_world)
    not by an internal value head.
  Validation experiment: V3-EXQ-497 5/5 PASS 2026-04-27 (UC1-UC5 above).
    Behavioural validation = V3-EXQ-495 (V3 full-completion gate,
    MECH-163 dual systems test); already drafted, gated on this
    substrate. queue V3-EXQ-495 as a separate decision after reviewing
    V3-EXQ-497 -- 3 conditions x 2 paradigms x 7 seeds is a several-hour
    behavioural run, not a substrate-readiness diagnostic.
  Design doc: REE_assembly/docs/architecture/mech_293_ghost_goal_probe_search.md
  See MECH-293 (this claim), MECH-292 (upstream ranked-bank source),
    SD-039 (transitive payload substrate), ARC-007 strict (no value head),
    ARC-018 (waking trajectory proposal loop being modified), ARC-032
    (goal-biased sequence generation -- one instantiation), MECH-089
    (theta-packaged waking E3 updates -- architectural context),
    MECH-094 (hypothesis-tag invariant -- preserved at proposal,
    stripped at commit), MECH-269 (anchor / probe substrate -- transitive
    via MECH-292), MECH-291 (mode-sensitive sequence generator framing --
    MECH-293 is the waking arm).
