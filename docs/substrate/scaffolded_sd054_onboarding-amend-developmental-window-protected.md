## scaffolded_sd054_onboarding AMEND: developmental-window / protected-goal consolidation (2026-06-03b)
- scaffolded_sd054_onboarding developmental-window amend -- IMPLEMENTED 2026-06-03.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change). Routed by the V3-EXQ-634 design-error review.
  ROOT CAUSE (substrate design error, not tuning): GoalState.update()
  (ree_core/goal.py:173) ALWAYS decays the persistent z_goal attractor
  (z_goal *= 1-decay_goal) BEFORE the benefit-gated pull, and REEAgent.reset()
  does NOT reset goal_state (no goal_state.reset() call in agent.py -> z_goal
  persists across episodes/phases). The prior scaffold called update_z_goal
  every step in P1 (seed_goal=True) and P2 (_eval_episode) including UNFED steps,
  so each unfed step is a pure decay-only washout (x0.995 at decay_goal=0.005).
  Stage-0 lights z_goal and P0 preserves it (goal pipeline frozen, update_z_goal
  not called), but P1/P2 then erode the trace before ecological contact -- and
  with the 603e-cluster finding that 2/3 seeds never reach foraging competence,
  P1 is mostly unfed, so the Stage-0 trace is washed out before P2 measurement.
  634 was therefore testing "can the infant stay goal-active while fed-then-
  starved under decay-only updates?" not "form -> consolidate -> learn contact".
  MECH-295/MECH-307 freezes (_set_goal_pipeline_frozen) only short-circuit the
  consumer pathway (liking bridge + conjunction); they do NOT protect the
  GoalState attractor -- a separate developmental window for the attractor was
  needed.
  THE FIX (all behind no-op-default flags; bit-identical when off):
    (1) Stage-0b protected consolidation: ScaffoldedSD054OnboardingScheduler.
        run_stage0b_consolidation(agent, device, stage0_baseline_norm=...) runs a
        short window in the safe nursery env with E1/E2 training open but
        update_z_goal NOT called (seed_goal=False) -> z_goal cannot be washed out
        by decay-only updating. Returns Stage0bConsolidationResult
        (z_goal_norm_start/end, retention_ratio, retention_gate_passed). Acceptance:
        retention_ratio >= scaffold_stage0b_retention_gate (default 0.75 of the
        Stage-0 baseline).
    (2) Contact-gated P1/P2: when scaffold_contact_gated_goal_updates is set
        (under the master scaffold_developmental_window_enabled), _train_episode
        (P1) and _eval_episode (P2) only call update_z_goal on a VALIDATED contact
        step (benefit > scaffold_p2_contact_benefit_threshold); unfed steps are
        skipped -> no decay-only washout. Stage-0 forced-feed is unaffected
        (forced benefit is always supra-threshold). decay_only is thus reserved
        for mature/autonomous tests, NOT the nursery gate.
    (3) Goal-write-mode constants (GOAL_WRITE_FORCED_FEED_OPEN /
        CONSOLIDATE_PROTECTED / ECOLOGICAL_CONTACT_OPEN / DECAY_ONLY_ALLOWED /
        MEASUREMENT_READONLY) + per-phase diagnostics
        (n_contact_refresh_updates / n_decay_only_updates /
        n_skipped_protected_updates on P1OnboardingResult + P2OnboardingMetrics)
        so a manifest distinguishes goal loss due to no-contact vs decay-only
        washout vs failed-formation-despite-contact.
  Config (ScaffoldedSD054OnboardingConfig, all default no-op): master
  scaffold_developmental_window_enabled=False; scaffold_stage0b_enabled=False;
  scaffold_stage0b_episode_budget=10; scaffold_stage0b_retention_gate=0.75;
  scaffold_contact_gated_goal_updates=False. Contact gate reuses
  scaffold_p2_contact_benefit_threshold.
  Backward compatible: with the master flag off, run_stage0b_consolidation aborts
  (stage0b_disabled) and P1/P2 take the legacy every-step decay-only path
  (n_skipped_protected_updates==0, n_decay_only_updates>0) -> bit-identical to the
  pre-amend 634 path. 739/739 contracts + 7/7 preflight PASS; v3_exq_634 --dry-run
  unchanged. Activation smoke: Stage-0b retention=1.000 (z_goal preserved); P1/P2
  contact-gated -> n_decay_only=0, Stage-0 trace (0.328) survives to P2 vs decay
  to ~0 under the legacy path.
  Phased training: N/A (harness scheduling/windowing; no learned parameters; no
  new encoder head). MECH-094: N/A (waking goal-pipeline onboarding; no
  simulation/replay write surface).
  Contracts: tests/contracts/test_scaffolded_sd054_onboarding.py C7 group
  (c7_config_defaults_are_noop; c7_stage0_trace_persists_across_stage0b;
  c7_decay_only_blocked_in_protected_window -- ON-vs-OFF contrast;
  c7_ecological_contact_still_refreshes; c7_flags_off_bit_identical_legacy_path).
  Validation experiment: V3-EXQ-634b (corrected nursery readiness; developmental
  -window flags ON), queued via /queue-experiment. Claim-free (substrate
  diagnostic). Does NOT queue V3-EXQ-603f; 603f stays blocked until the corrected
  nursery passes readiness gates. V3-EXQ-634 left running for diagnostic value; if
  it fails Stage-0-lights / P1-P2-contact-absent / z_goal-collapses, that is
  substrate_not_engaged (developmental-window missing), NOT evidence against the
  goal stream.
  See scaffolded_sd054_onboarding (parent + prior amends), V3-EXQ-634 (the design
  -error review this addresses), V3-EXQ-634b (validation), goal.py GoalState.update
  (the always-decay mechanism), MECH-112/116/117 (GoalState claims; untouched),
  MECH-295/MECH-307 (consumer-pathway freezes; insufficient alone), MECH-094 (N/A).
