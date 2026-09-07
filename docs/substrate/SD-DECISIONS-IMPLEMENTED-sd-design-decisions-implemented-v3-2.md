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

- SD-012 sustained-drive amendment (goal_pipeline:GAP-3, Option 1) — IMPLEMENTED 2026-05-17.
  GoalConfig.drive_ema_alpha (default 1.0; goal.py). The SD-012 multiplier now uses an
  EMA trace of drive_level instead of the instantaneous value:
  _drive_trace = (1 - drive_ema_alpha) * _drive_trace + drive_ema_alpha * drive_level;
  effective_benefit = benefit_exposure * z_goal_seeding_gain * (1 + drive_weight * _drive_trace).
  Motivation: instantaneous drive_level collapses to ~0.005 the step a resource is
  consumed (energy resets toward 1.0), cancelling the SD-012 amplification at exactly
  the contact events where seeding must fire (EXQ-536a). Backward compatible:
  drive_ema_alpha=1.0 -> trace == drive_level every step regardless of init ->
  bit-identical to the pre-amendment instantaneous form (contract C1/C2). Surfaced
  through REEConfig.from_dims() mirroring drive_weight. _drive_trace is zero-initialised,
  so alpha < 1.0 carries a deliberate ~1/alpha-step cold-start transient (accepted per
  goal_pipeline Q2). Lit-anchored operating value 0.02 (~35-step half-life;
  wanting_liking synthesis 30-60 step window). Phased training: N/A (no encoder, no
  learning). MECH-094: N/A (no simulation/replay/memory write). Contract:
  tests/contracts/test_sustained_drive_ema_gap3.py (7/7). Validation experiment:
  discriminative drive_ema_alpha sweep {0.01,0.02,0.2,1.0} queued (see goal_pipeline_plan.md
  GAP-3). claims.yaml NOT modified -- MECH-306 sustained_drive_trace registration is the
  governance follow-on gated on the sweep result. See GoalConfig, goal.py GoalState.update().
