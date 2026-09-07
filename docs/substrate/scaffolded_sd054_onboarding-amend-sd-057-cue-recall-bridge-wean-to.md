## scaffolded_sd054_onboarding AMEND: SD-057 cue-recall bridge (wean-to-wild foraging-contact lever) (2026-06-04)
- scaffolded_sd054_onboarding cue-recall bridge amend -- IMPLEMENTED 2026-06-04.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core
  change -- the SD-057 substrate it uses is already in ree_core). Integrates the
  SD-057 L6 cue-recall + L2 bank-token-binding into the nursery curriculum as a
  candidate lever on the GAP-2 foraging-CONTACT axis (NOT survival).
  HYPOTHESIS (wean-to-wild): the nursery forced-feed already builds z_goal but has
  no path from a nursery-built goal to APPROACHING a resource the agent can SEE
  but has not contacted. SD-057 cue-recall is that path (Pavlovian-instrumental
  transfer / sign-tracking): forced-feed builds per-object tokens; in P1/P2 a
  PERCEIVED resource cue retrieves its token -> pulls z_goal toward it -> MECH-295
  approach bias -> first contact.
  Changes (all behind scaffold_cue_recall_bridge_enabled, default False,
  bit-identical OFF):
    (1) _build_env: new _sd049_kwargs(cfg) spreads multi_resource_heterogeneity_enabled
        + n_resource_types + per_axis_drive_enabled into ALL 4 phase env
        constructors when the bridge is on (the scaffold's envs previously did
        NOT enable SD-049, so they emitted no per-type tags / proximity views /
        per-axis drive -- the central gap this amend closes). Returns {} when off.
    (2) _train_episode (P1) + _eval_episode (P2): pass resource_type
        (_contacted_resource_type(obs_dict): sd049_consumed_type_tag_this_tick ->
        resource_type_at_agent) into agent.update_z_goal so the bank binds
        per-object tokens (L2); and call _maybe_cue_recall(agent, env, obs_dict,
        drive, cfg) each step -- derives the strongest-perceived type from the
        SD-049 per-type proximity field views (argmax over resource_field_view_<name>,
        gated by scaffold_cue_recall_min_proximity), sets agent._per_axis_drive,
        and fires agent.cue_recall_wanting (L6). n_cue_recall_fires surfaced in
        P2OnboardingMetrics.
    (3) Config: scaffold_cue_recall_bridge_enabled (False), scaffold_cue_n_resource_types
        (3), scaffold_cue_recall_min_proximity (0.0).
  REQUIRES the CALLER to build the agent with use_incentive_token_bank=True +
  use_cue_recall=True + use_resource_encoder=True (the SD-057 substrate flags); the
  scheduler accepts the agent, it does not build it. Without those agent flags the
  wiring is a harmless no-op (bank None -> resource_type ignored; cue_recall_wanting
  returns 0).
  Backward compatible: bridge OFF -> _sd049_kwargs={} (legacy envs), resource_type
  None (default), _maybe_cue_recall returns 0 -> bit-identical. 55/55 scaffold
  contracts (42 prior + 13 new C9) + 7/7 preflight. Unit + agent + targeted smoke
  2026-06-04: OFF envs no SD-049; ON envs SD-049 (3 types, per-type views,
  per_axis_drive); bank builds tokens; cue fires for a perceived type WITH a token
  and is silent for a type without (identity-matched); OFF bit-identical.
  Phased training: N/A (harness wiring; no new learned parameters). MECH-094:
  cue_recall_wanting carries simulation_mode (no-op on replay); scheduler is a
  waking training stream.
  Validation experiment: V3-EXQ-638 (cue ON vs OFF contact-rate ablation,
  claim_ids=[], diagnostic). Self-contained: both arms set the landed 634c ARM_3
  seeding regime (drive_floor=0.9 + benefit_threshold=0.02) so wild contact seeds;
  the ONLY difference is the cue-recall bridge, so the P2 contact-rate delta
  isolates its effect. Does NOT depend on the stalled 634c run (only its landed
  code). Acceptance: C1 cue fires ON >=2/3; C2 cue silent OFF (all); C3 P2
  contact_rate ON > OFF per matched seed >=2/3; C4 survival not regressed
  (informational). Dry-run runs end-to-end both arms (ON Stage-0 z_goal 0.43 vs
  OFF 0.23; C2 holds); full-budget needed for C1/C3. Queued via /queue-experiment.
  HONEST SCOPE: targets the CONTACT axis only; does NOT fix survival (2/3 seeds
  die in P1) and may even raise hazard exposure by approaching food in
  hazard_food_attraction>0 envs -- the ablation measures survival too. Complementary
  to the curriculum's survival work + the 634c seeding fix, not a replacement.
  Design routing: user-directed (AskUserQuestion 2026-06-04: "design the
  integration now", scope "L6 cue-recall bridge only"). Sits on the GAP-2 closure
  path; coordinated with the substrate_queue scaffolded_sd054_onboarding track
  (634c seeding-calibration in flight; this is an additional, decoupled lever).
  See SD-057 (ree_core substrate; goal.py IncentiveTokenBank + agent.cue_recall_wanting),
    scaffolded_sd054_onboarding (parent substrate + prior amends), goal_pipeline:GAP-2
    (the foraging-contact ceiling this targets) + GAP-7 (downstream L9 retest),
    V3-EXQ-634c (seeding regime reused), V3-EXQ-636/637 (SD-057 v1 + phase-2
    diagnostics), MECH-295 (approach bridge; cue-recall's downstream), MECH-347
    (L6 cue-recall claim), SD-049 (per-type tags + per-axis drive + proximity
    views), MECH-094 (N/A).
