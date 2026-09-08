## SD-MEL-PRODUCER: environment.non_converging_world_rule_shift -- IMPLEMENTED (2026-07-21)
- SD-MEL-PRODUCER: environment.non_converging_world_rule_shift -- IMPLEMENTED 2026-07-21.
  The PRODUCER half of the MECH-180 pair -- link (i) novelty -> graded above-reference
  waking MEL. SD-MEL-CONSUMER (above) owns link (ii) and is already PROVEN; link (i) had
  never been demonstrated because the environment could not produce a novelty gradient
  at all: V3-EXQ-677's C1 manipulation check measured a high- vs low-novelty mean E1
  prediction-error difference of 8.8e-07 against a 0.01 threshold, and V3-EXQ-718a
  measured ecological MEL ~1e-5 (noise-level, scrambled vs novelty level) with
  conv_rel_drop ~0.98. Both autopsies classify this measurement_gap (environment /
  test-bed producer gap), NOT a substrate ceiling and NOT a falsification.
  ROOT CAUSE: env_drift_interval fires _drift_hazards(), which only MOVES hazards. The
  optimal prediction of a random walk is its mean, so the world-forward model learns
  that fast and PE floors at the irreducible noise level -- drift adds sampling NOISE,
  not learning LOAD.
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorld._maybe_shift_world_rule;
  inherited by CausalGridWorldV2). No new file, no new module.
  Config (env kwargs, NOT REEConfig -- experiment scripts construct envs directly):
  world_rule_shift_enabled (bool, default False; set True to enable),
  world_rule_shift_interval (int, default 0 = never; WORLD-steps between rule re-draws),
  world_rule_shift_depth (int, default 0; action-pair transpositions per shift),
  world_rule_shift_scope (str, default "action_map"; reserved for a later
  structural-statistics variant, ValueError on anything else).
  Data flow: world_rule_shift_interval -> _maybe_shift_world_rule() re-permutes
  self._action_map from self._rng -> dx,dy at step() -> the actual transition changes
  -> z_world(t+1) diverges from E2.world_forward(z_world(t), a), which takes the action
  as an INPUT -> e3 prediction_error rises -> info["world_rule_shift_occurred" /
  "world_rule_shift_count" / "steps_since_world_rule_shift"].
  WHY NOISE IS NOT A SUBSTITUTE (the central design constraint): grading OBSERVATION
  NOISE would also yield a monotone MEL gradient -- but by construction, on any
  substrate, whether or not MECH-180 is true. That is the DV-symmetry artifact class
  (failure_autopsy_V3-EXQ-604c; the defect that held V3-EXQ-683 on 2026-07-21). Elevated
  PE is only learning LOAD if it is reducible, so the operational discriminator is that
  genuine load DECAYS within a stationary window while noise does not.
  steps_since_world_rule_shift exists to let a consumer bin per-step PE by
  time-since-shift and measure exactly that; the validation carries a matched-PE noise
  arm as the negative control.
  FOUR LOAD-BEARING DECISIONS, all contract-pinned in
  tests/contracts/test_world_rule_shift_producer.py (12 tests) -- do not "simplify":
  (1) the CLASS-level ACTIONS dict is never mutated (it is a class attribute, so an
  in-place permutation would leak into every other env instance in the process); the
  effective map is a per-instance self._action_map.
  (2) the schedule keys off a cumulative self._world_steps_total, NOT episode-local
  self.steps. This was a LIVE DEFECT caught by the implementation probe: episode length
  itself collapses as the world gets less predictable (measured 69.0 -> 13.5 steps), so
  an episode-relative schedule makes the nominal interval stop controlling the actual
  shift RATE -- intervals 60/30/15/8/5 produced 2/2/3/20/21 shifts, non-monotone, taking
  the MEL ladder with it.
  (3) _action_map, the shift counters and _world_steps_total are NOT reset by reset() /
  reset_to() -- the action map is the world's causal structure, not episode state.
  (4) every RNG draw sits inside the enabled guard, so a disabled env consumes NO
  randomness and is bit-identical at the same seed (verified on seeds 42/123/456).
  Backward compatible: disabled by default; existing experiments unaffected, including
  seeded-rollout bit-exactness.
  No trainable parameters / no new encoder head / no new latent field -> NO phased
  training needed (validation still needs a converged recon-only base so PE sits at
  converged scale, per 701c; SD-056 contrastive is a confirmed P0 destabiliser).
  MECH-094: does not apply -- nothing is written to memory during non-waking states.
  MEASUREMENT NOTE for any consumer of this knob: shift rate SHORTENS episodes, so
  measure over a fixed STEP budget, not a fixed episode count, and report mean episode
  length per arm so the confound stays visible.
  Unblocks: MECH-180 link (i), INV-050 ecological end-to-end demonstration (which is a
  SEPARATE, still-gated run -- this test-bed must validate first).
  Validation experiment: V3-EXQ-798 queued 2026-07-21
  (v3_exq_798_sdmelproducer_graded_nonconverging_world; diagnostic; claim_ids=[];
  recon-only converged base + graded rule-shift arms + matched-PE noise negative
  control; DV = mean per-step e3 prediction_error (MEL) and its decay by
  steps_since_world_rule_shift; PROMOTES NOTHING until it scores).
  The consumer is DELIBERATELY ABSENT from that validation: V3-EXQ-718a's
  learning_extracted[1] records that the consumer's DV is a deterministic function of
  MEL, so DV-monotone-in-measured-MEL is near-tautological and cannot validate a producer.
  See REE_assembly/docs/architecture/sd_mel_producer.md; SD-MEL-CONSUMER; SD-017;
  INV-050; MECH-180; plan-of-record
  REE_assembly/evidence/planning/sleep_substrate_plan.md (GAP-5b).
