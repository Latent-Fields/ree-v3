## scaffolded_sd054_onboarding AMEND: post-cue action/gradient instrumentation (V3-EXQ-640, 2026-06-05)
- scaffolded_sd054_onboarding post-cue MEASUREMENT-ONLY instrumentation --
  IMPLEMENTED 2026-06-05. Module: experiments/scaffolded_sd054_onboarding.py
  (harness layer; NO ree_core / goal.py / claims.yaml change). Routed by
  failure_autopsy_V3-EXQ-638a_2026-06-05 (Sections 5-8): 638a settled the cue
  "fires vs lifts contact" question (C3 FAIL, ARM_CUE_ON contact <= ARM_OFF on all
  3 seeds) but logged NO post-cue action trace, so it could not discriminate
  cue-to-action AUTHORITY / displacement / gradient-following / hazard-interrupt.
  This amend adds the missing trace as a PURELY READ-ONLY per-cue-fire diagnostic.
  Changes (all behind scaffold_post_cue_instrumentation, default False ->
  bit-identical OFF -- the accumulator is never built and the _eval_episode block
  is skipped):
    (1) Config: scaffold_post_cue_instrumentation (False) +
        scaffold_post_cue_window_steps (4 -- look-ahead horizon over which each
        cue fire's downstream moves are attributed).
    (2) _new_post_cue_diag() accumulator + helpers _nearest_resource_manhattan
        (non-toroidal Manhattan to nearest resource), _opposite_action (move
        reversal for oscillation), _read_zgoal (||z_goal|| + vector clone),
        _finalize_post_cue_window.
    (3) _eval_episode (P2) gains a post_cue_diag kwarg. When supplied it records,
        windowed around each cue fire: z_goal NORM delta (the displacement test --
        mean<0 = cue pulls toward a weaker token), z_goal PULL magnitude,
        absolute ||z_goal|| at fire (vs ARM_OFF wild attractor norm), SD-016
        _cue_action_bias norm, post-cue selected-action APPROACH rate vs the
        cue-independent background rate, first-gradient-improving-move latency +
        frac-first-move-approach (immediate authority), hazard-salience-interrupt
        count, oscillation count. Cue windows opened at step t are first aged at
        t+1 so the firing step's own move is never miscounted.
    (4) P2OnboardingMetrics.post_cue_diag (empty dict when off); run_p2 builds the
        accumulator only when the flag is set and threads it through every episode.
  READ-ONLY guarantee: the agent senses / selects / steps identically; the
  instrumentation only reads env (agent_x/agent_y/resources/ACTIONS) + goal_state.
  Backward compatible: flag default False -> post_cue_diag={}; bit-identical to
  pre-amend (verified: V3-EXQ-638a --dry-run byte-identical OFF path; 70/70 scaffold
  contracts incl 5 new C10 + 7/7 preflight PASS). MECH-094: N/A (waking P2
  measurement; no simulation/replay write surface). Phased training: N/A (pure
  read-only accounting; no learned parameters).
  Contracts: tests/contracts/test_scaffolded_sd054_onboarding.py C10 group
  (config-defaults-noop / flag-on-threads-accumulator / nearest_resource_manhattan
  / opposite_action / finalize_window accumulation arithmetic).
  Validation experiment: V3-EXQ-640 (measurement-only post-cue action/gradient
  diagnostic; same ARM_OFF/ARM_CUE_ON ablation as 638a, behaviourally unchanged,
  both arms instrumented; experiment_purpose=diagnostic, claim_ids=[]; PASS =
  measurement succeeded, the 638a-autopsy discriminator grid applied at review).
  GATES the planned V3-EXQ-638b interoceptive build -- do NOT build 638b until 640
  routes.
  See scaffolded_sd054_onboarding cue-recall bridge + FORMATION fix entries above,
  V3-EXQ-638a (the FAIL whose un-discriminated branch this measures), SD-057 /
  MECH-347 (L6 cue-recall; ree_core unchanged), MECH-295 (downstream approach
  bridge), goal_pipeline:GAP-2 (foraging-contact ceiling), MECH-094 (N/A).
