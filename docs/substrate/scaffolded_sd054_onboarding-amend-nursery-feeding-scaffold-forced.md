## scaffolded_sd054_onboarding AMEND: nursery/feeding scaffold (forced-benefit Stage-0 + survival levers + P2 guard) (2026-06-03)
- scaffolded_sd054_onboarding amend -- IMPLEMENTED 2026-06-03 (mechanism-
  validated; full-scale runtime readiness still PENDING a substrate-readiness
  run -- see "Validation" below; substrate_queue ready stays FALSE). Routed by
  failure_autopsy_V3-EXQ-603e-626a-622_2026-06-03: the update_z_goal wiring
  amend (deb24cc, 2026-06-02) is necessary-but-insufficient -- 603e showed
  z_goal=0 ecologically (all 15 cells) because 2/3 seeds never reach foraging
  competence and the hard P2 env (hazard_food_attraction=0.7) starves
  benefit_exposure even for survivors; 626a P0 positive control formed z_goal
  on only 1/3 seeds. Core principle: infant REE needs a nursery and feeding
  time before mature autonomous goal formation can be fairly tested.
  Module: experiments/scaffolded_sd054_onboarding.py (ADDITIVE -- no existing
  config default changed; the contract test_c3_config_memo_default_values
  still passes verbatim; the V3-EXQ-603f re-issue sets the strengthened
  curriculum values explicitly, exactly as 603e set restored budget).
  Five additions:
    (1) FORCED-BENEFIT STAGE-0 NURSERY -- new run_stage0_nursery() method +
      Stage0NurseryResult. Runs scaffold_stage0_episode_budget episodes in a
      dense (num_resources=6), hazard-free (num_hazards=0) nursery env with
      the agent spawning in the reef refuge band; goal pipeline UNFROZEN; every
      step feeds a FORCED supra-threshold benefit (scaffold_stage0_forced_benefit
      default 1.0) + forced drive (scaffold_stage0_forced_drive default 0.9)
      into agent.update_z_goal -- the agent is "fed" regardless of actual
      resource contact. DECOUPLES z_goal formation from survival/foraging skill
      and is the positive control "the goal stream lights when fed". Records
      mean_forced_benefit + z_goal_norm_peak + z_goal_formed (peak >
      scaffold_stage0_z_goal_peak_gate default 0.4). Aborts loudly
      (goal_state_none_set_z_goal_enabled_true) if the agent lacks a GoalState
      (z_goal_enabled must be True -- same two-part-fix precondition as 603e).
      Reuses _train_episode via new forced_benefit/forced_drive/goal_peak_sink
      kwargs (forced path active only when seed_goal AND forced_benefit set).
    (2) STRENGTHEN-SURVIVAL LEVER -- scaffold_p1_anneal_hold_fraction (default
      0.0 = current pure-linear anneal; >0 holds full nursery relaxation for
      that fraction of P1 before ramping = staged withdrawal of assistance).
      The other survival levers (more episodes, fewer P0 hazards, gentler hfa
      max) are existing knobs the 603f experiment sets explicitly.
    (3) GUIDED FEEDING-STAGE PLAN made explicit -- module STAGE_PLAN + stage_plan()
      (Stage 0 nursery forced-feed -> Stage 1 guided low-conflict (run_p0) ->
      Stage 2 easy foraging (early P1) -> Stage 3 guarded hazard (late P1) ->
      Stage 4 mature test (run_p2)). For the 603f manifest.
    (4) P2 MEASUREMENT GUARD -- scaffold_p2_hazard_food_attraction_guard
      (default -1.0 = no guard, P2 hfa stays 0.7; >=0 overrides so the
      measurement window admits contact; 603f sets ~0.3) + a foraging-contact-
      rate readout: _eval_episode counts steps where post-step benefit >
      scaffold_p2_contact_benefit_threshold; P2OnboardingMetrics gains
      contact_steps / contact_rate / hazard_food_attraction_used. Distinguishes
      a z_goal=0 read caused by "infant never fed" from a genuine goal-formation
      failure despite contact.
    (5) SUBSTRATE-GATE + INTERPRETATION HELPERS -- evaluate_substrate_gate(...)
      returns {substrate_gate_passed, stage0_positive_control, g1_survival,
      g2_contact, g3_zgoal} (each requires >= min_fraction[=2/3] of seeds);
      classify_interpretation_branch(...) returns the pre-registered five-way
      grid (substrate_not_engaged / fed_but_no_goal / goal_formed_diversity_inert
      / goal_formed_mechanisms_load_bearing / goal_formed_behaviour_random_harmful).
      A same-substrate run with the nursery disabled supplies no Stage-0 z_goal
      peaks -> stage0_positive_control False -> the gate can NEVER pass, so the
      old path cannot masquerade as 603f (z_goal=0 is never interpreted without
      the feeding positive control + contact context).
  Backward compatible: master switch (use_scaffolded_sd054_onboarding_scheduler)
  default False -> scheduler inert; all new knobs default no-op (Stage-0
  disabled, P2 guard off, hold 0.0); no existing config default changed. The
  scheduler's OWN behaviour is unchanged for existing consumers (603d/603e do
  not enable Stage-0); 731 contracts (19 prior scaffolded + 12 new + rest) +
  7/7 preflight PASS; 603e --dry-run runs unchanged (reaches the known
  dry-scale SUBSTRATE_FAILURE branch). Activation smoke 2026-06-03 (real
  REEAgent, z_goal_enabled=True + drive_weight=2.0, Stage-0 2 ep x 25 steps):
  forced feed lights z_goal (z_goal_norm_peak=0.234 > 0; the >0.4 acceptance is
  a full-scale gate, not a dry-scale one), P2 guard hfa=0.3 applied, contact-rate
  readout wired.
  Phased training: N/A new encoder head (Stage-0 warms E1/E2 + seeds z_goal via
  forced input; no new latent-target head -> no new collapse risk). MECH-094:
  N/A -- waking training stream; no simulation/replay write surface (the
  scheduler never sets hypothesis_tag).
  Contracts: tests/contracts/test_scaffolded_sd054_onboarding.py C6 block (12
  new: amend-config-no-op defaults, Stage-0 abort reasons, nursery env shape,
  P2 hfa-guard override, P2 contact-readout fields, substrate-gate full-pass /
  blocks-without-stage0 / blocks-on-starvation, five interpretation branches,
  stage_plan).
  Validation: NOT YET RUN at full scale. The runtime readiness gates
  (Stage-0 z_goal>0.4 on >=2/3 seeds, P1 survival >=2/3, P2 contact >0 on
  >=2/3) require a full-budget run and are what the post-substrate re-issue
  V3-EXQ-603f measures (it evaluates the substrate gate FIRST and self-routes
  to non_contributory/substrate_not_engaged if the gates fail). 603f is queued
  via /queue-experiment ONLY after a full-scale substrate-readiness run
  confirms the gates; substrate_queue.ready stays FALSE until then.
  Design doc: REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md
  (Amend 2026-06-03 section flipped PENDING -> IMPLEMENTED).
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603e-626a-622_2026-06-03.{md,json}.
  See scaffolded_sd054_onboarding (parent + update_z_goal-wiring amend above),
    Q-045 / MECH-313 / MECH-260 (cluster gated on 603f), MECH-295 / MECH-307
    (goal-pipeline consumers), modulatory-bias-selection-authority (the BG-like
    E3.select-authority substrate that becomes the next blocker if 603f hits
    the goal_formed_diversity_inert branch), V3-EXQ-603e (the FAIL this amend
    addresses), V3-EXQ-603f (post-substrate re-issue), MECH-094 (N/A).
