# ree-v3

## scaffolded_sd054_onboarding Substrate (2026-05-31)
- scaffolded_sd054_onboarding: experiments.scaffolded_sd054_onboarding.ScaffoldedSD054OnboardingScheduler
  -- IMPLEMENTED 2026-05-31. behavioral_diversity_isolation:GAP-C prereq (2) substrate landing.
  Closes IGW-20260531-029. Plan-of-record memo:
  REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md (2026-05-29).
  Triage memo: REE_assembly/evidence/planning/z_goal_collapse_triage_2026-05-31.md (2026-05-31).
  Routed by V3-EXQ-490g cohort autopsy 2026-05-29 (Cluster B / V3-EXQ-603c
  substrate-uniform z_goal-zero family) + V3-EXQ-591 autopsy 2026-05-27 (canonical
  cluster autopsy; prereq enumeration in section 7).
  Modules:
    ree-v3/experiments/scaffolded_sd054_onboarding.py (NEW; ScaffoldedSD054OnboardingConfig
      + ScaffoldedSD054OnboardingScheduler + P0OnboardingResult + P1OnboardingResult
      + P2OnboardingMetrics + clone_trained_agent helper).
    ree-v3/ree_core/environment/causal_grid_world.py (one new kwarg
      reef_bipartite_agent_spawn_in_reef_half default False on CausalGridWorldV2.__init__;
      _build_bipartite_pools agent-pool predicate widened to (agent_band OR reef_half)
      when the kwarg is True).
  ree_core/ otherwise UNTOUCHED per memo "Implementation surface choice" -- the
  curriculum lives in the experiment-harness layer alongside infant_curriculum.py
  and committed_mode_curriculum.py. Master switch
  use_scaffolded_sd054_onboarding_scheduler defaults False on the scheduler config
  dataclass (NOT surfaced through REEConfig.from_dims -- matches the
  committed_mode_curriculum / infant_curriculum precedent for experiment-harness
  helpers). 14 phase-config knobs all default to memo-suggested values.
  Three-phase scheduler:
    run_p0: env config (hazard_food_attraction=0.0, proximity_harm_scale=0.05,
      num_hazards=2, num_resources=3, reef_bipartite_agent_spawn_in_reef_half=True);
      goal pipeline FROZEN (use_mech295_liking_bridge=False + use_mech307_conjunction=False
      runtime mutation on agent.config); E1+E2 training over scaffold_p0_episode_budget
      (default 30) episodes at scaffold_steps_per_episode (default 200).
    run_p1: linear anneal across episode budget --
      hazard_food_attraction 0.0 -> 0.7,
      proximity_harm_scale 0.05 -> 0.1,
      mech295_min_drive_to_fire 1.0 -> 0.01,
      mech307_conjunction_z_beta_threshold 0.6 -> 0.3;
      spawn admissibility narrows back to midline band
      (reef_bipartite_agent_spawn_in_reef_half=False);
      goal pipeline UNFROZEN; end-of-P1 survival gate
      (median episode length over last scaffold_p1_stability_window (10) episodes
      >= scaffold_p1_survival_gate_steps (75)) -- Fix D from V3-EXQ-603c retained.
    run_p2: target env config (hazard_food_attraction=0.7, proximity_harm_scale=0.1,
      num_hazards=4, num_resources=5; matches V3-EXQ-603b GAP-4 Tier-1 measurement env);
      policy frozen (no optimizer steps); measures z_goal_norm_peak,
      approach_commit_rate, bridge_cue_fires, dacc_bias_nonzero_steps per episode.
  Data flow: scheduler mutates the live MECH-295 bridge config dataclass
  (agent.mech295_bridge.config.min_drive_to_fire +
  agent.mech295_bridge.config.mech307_conjunction_z_beta_threshold) and the agent's
  REEConfig flat fields per anneal tick. The bridge reads these per-call (not at
  __init__ time), so mutation takes effect immediately without rebuild.
  Backward compatible: default scheduler config sets master switch to False ->
  agent.config + env defaults match pre-2026-05-31 substrate (V3-EXQ-490g-style
  gap4 config when callers set use_mech295_liking_bridge/use_mech307_conjunction
  themselves; env reef_bipartite_agent_spawn_in_reef_half=False matches legacy
  SD-054 bipartite spawn). 645/645 contracts + 7/7 preflight PASS post-landing
  (was 628 + 17 new MECH-341 retune + SD-056 amend + InfantCurriculumScheduler
  recalibration sessions earlier in 2026-05-31; +17 new scaffolded_sd054_onboarding
  contracts in tests/contracts/test_scaffolded_sd054_onboarding.py). Bit-identical
  OFF guarantee verified across all contract / preflight tests.
  Activation smoke (2026-05-31, full P0/P1/P2 pipeline on a default REEAgent at
  reduced budget 2/2/1 episodes x 20 steps): P0 frozen goal pipeline + reef-half
  spawn confirmed; P1 anneal lands at min_drive_to_fire=0.010 + hfa=0.70 per memo;
  P2 frozen-policy eval runs to completion without crash. Full-scale measurement
  (z_goal_norm_peak > 0 + cascade behavioural metrics) is the V3-EXQ-620
  substrate-readiness validation experiment.
  MECH-094: N/A (scheduler operates on waking-stream env + agent state during the
  agent's act loop; no simulation / replay write surface).
  Phased training: the entire scheduler IS phased training. P0 warms encoder +
  E2 + E3 with goal pipeline frozen; P1 anneals goal pipeline gates ON; P2
  measures frozen-policy. Matches committed_mode_curriculum.py P0/P1/P2
  discipline; no .detach() boundary changes needed in ree_core.
  Substrate scope clarification: this substrate addresses prereq (2) of
  behavioral_diversity_isolation:GAP-C ("goal-pipeline training regime produces
  non-trivial z_goal in default config"). It does NOT address prereq (1) MECH-307
  default-value recalibration (cleared 2026-05-15 by V3-EXQ-540g PASS) or
  prereq (3) InfantCurriculumScheduler Phase 0->1 exit signal recalibration
  (landed 2026-05-31 earlier in the day). It does NOT address the 490 cohort's
  MECH-295 cascade behavioural validation (Phase 4 of goal_pipeline_plan.md) --
  that work continues under the 490 cohort with the MECH-295 narrowing routing
  from the 490j autopsy (2026-05-31).
  Validation experiment: V3-EXQ-620 substrate-readiness diagnostic (4-arm
  design per memo Acceptance section: ARM_0 ALL_OFF_baseline / ARM_1 SCAFFOLD_ONLY
  / ARM_2 SCAFFOLD_AND_ANNEAL / ARM_3 SCAFFOLD_AND_ANNEAL_CONTROL_FROM_SCRATCH;
  3 seeds x 4 arms). Acceptance: C1 (cells complete -- >= total_cells/2 P0+P1+P2
  finish without hitting Fix D survival gate) AND (C2 z_goal_norm_peak >= 0.1
  on >= 2/3 seeds in at least one arm OR C3 cascade behaviourally consequential
  per memo). PASS clears prereq (2); behavioural cluster validation V3-EXQ-603d /
  591b queueable.
  Design doc: REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md
  (status flipped plan-of-record -> IMPLEMENTED 2026-05-31).
  See behavioral_diversity_isolation:GAP-C (closure-plan owner of prereq (2)),
    goal_pipeline:GAP-4 (related but distinct -- 490 cohort owns MECH-295 cascade
    Phase 4, NOT prereq (2)), V3-EXQ-591 / V3-EXQ-540 series / V3-EXQ-590a /
    V3-EXQ-603 series (the substrate-uniform z_goal-zero family this substrate
    addresses), MECH-307 / MECH-295 / SD-014 / SD-015 (downstream consumer claims
    whose v3_pending gate transitively benefits from prereq (2) clearance once
    V3-EXQ-620 PASSes), Q-045 / MECH-313 / MECH-260 (V3-EXQ-603c claim trio
    transitively unblocked once prereq (2) clears), MECH-094 (call-site scoping;
    not applicable -- experiment-harness scheduler with no simulation write surface).

## scaffolded_sd054_onboarding AMEND: update_z_goal wiring + Stage-0 positive control (2026-06-02)
- scaffolded_sd054_onboarding amend -- IMPLEMENTED 2026-06-02. Wires the
  missing goal-pipeline seeding call into the scheduler training/eval loops.
  Routed by failure_autopsy_V3-EXQ-603d_2026-06-01 (Class-1 harness/wiring
  finding) + failure_autopsy_V3-EXQ-625b_2026-06-02 (corroborating
  monostrategy/behavioural-diversity record, plausibly downstream of the same
  inert goal pipeline). Folds both failure records into one implementation pass
  per the substrate_queue amend_hint (status amend_pending -> amend_implemented_pending_validation).
  Module: experiments/scaffolded_sd054_onboarding.py.
    New helper _benefit_and_drive(obs_body) -> (benefit_exposure, drive_level)
      mirroring experiments/goal_stream_stages_sd054.py:_benefit_and_drive
      (benefit=obs_body[11]; drive=clip(1-energy,0,1), energy=obs_body[3];
      reshape(-1) robust to [D] / [1,D]).
    _train_episode gains seed_goal: bool = False kwarg; when True, calls
      agent.update_z_goal(benefit, drive) after each env.step using the
      post-step body-state (mirrors reference runner line 537). run_p1 passes
      seed_goal=True; run_p0 keeps the default False so the P0 warm-up stays
      goal-pipeline-frozen by design (user-confirmed P1+P2-only scope,
      AskUserQuestion 2026-06-02).
    _eval_episode (P2) calls agent.update_z_goal after each env.step and
      re-reads goal_norm into z_goal_norm_peak (mirrors reference runner
      line 590). P2 is frozen-policy measurement; z_goal MUST be driven for
      the C4 z_goal_norm_peak acceptance metric to be non-zero.
  Root cause: before this amend, neither _train_episode nor _eval_episode
  called agent.update_z_goal (zero matches in the module), so GoalState.update
  was never reached and z_goal stayed zero-init across every step of every arm
  -- the V3-EXQ-603d C4 z_goal=0 SUBSTRATE_FAILURE was a 626-class harness/
  wiring artifact living in the substrate module, NOT a substrate ceiling.
  TWO-PART FIX -- the validation config matters: the 603d FAIL config built the
  agent WITHOUT z_goal_enabled=True (from_dims default False), so its
  agent.goal_state was None and update_z_goal would early-return (agent.py:4791)
  even with this wiring in place. The working reference V3-EXQ-622 sets
  z_goal_enabled=True + drive_weight=2.0 explicitly. THE V3-EXQ-603e VALIDATION
  CONFIG MUST SET z_goal_enabled=True + drive_weight=2.0 (and keep
  use_mech295_liking_bridge / use_mech307_conjunction as 603d did). Without
  that, the wiring fix is inert.
  Stage-0 positive control (amend_hint requirement): two new contracts in
  tests/contracts/test_scaffolded_sd054_onboarding.py --
    test_c6_stage0_positive_control_p2_seeds_zgoal: with a z_goal_enabled agent
      and forced supra-threshold benefit+drive (monkeypatched _benefit_and_drive),
      scheduler P2 must produce z_goal_norm_peak_max > 0.0. A scheduler that
      does not call update_z_goal yields exactly 0.0 -> a z_goal=0 scheduler is
      structurally unshippable.
    test_c6_update_z_goal_called_in_p1_not_p0: spy-counts update_z_goal calls;
      asserts 0 in run_p0 (warm-up stays goal-frozen) and >0 in run_p1.
  Backward compatible: the change lives inside the scheduler (only instantiated
  by experiments that explicitly use it: 621/621a/603d). No new config params,
  no REEAgent signature change, no from_dims change. P0 path bit-identical
  (seed_goal defaults False). The scheduler's OWN behaviour changes (z_goal now
  forms) -- the intended fix; it carries no landed governance evidence (621a
  non_contributory, 603d FAIL) so there is no bit-identical-OFF consumer to
  protect. 19/19 scaffolded contracts + 7/7 preflight PASS; full contract suite
  665 passed (1 pre-existing unrelated failure: infant_curriculum_gap9 C6, a
  stale Phase3==Phase2 assertion from the 2026-06-01 IGW-023 enrichment landing,
  flagged separately). End-to-end P0/P1/P2 activation smoke runs without crash.
  Phased training: the scheduler IS phased training; update_z_goal is the
  GoalState seeding call (not an encoder head) -- no collapse risk; P1+P2-only
  gating preserves the P0 goal-frozen warm-up.
  MECH-094: waking-stream only (hypothesis_tag=False); the scheduler has no
  simulation/replay write surface.
  Validation experiment: V3-EXQ-603e queued via /queue-experiment -- re-issue
  of 603d on the hook-fixed scheduler at RESTORED budget (P0/P1=100/50 vs 603d's
  30/30) WITH z_goal_enabled=True. Acceptance per the 603d failure-record
  target: z_goal_norm_peak > 0.4 on >= 2/3 seeds in P2 AND P1 survival gate
  passed on >= 2/3 seeds.
  Design doc: REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md
  Autopsies: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603d_2026-06-01.{md,json},
    failure_autopsy_V3-EXQ-625b_2026-06-02.{md,json}.
  See scaffolded_sd054_onboarding (parent substrate entry above), Q-045 /
    MECH-313 / MECH-260 (the behavioural cluster transitively unblocked once
    603e PASSes), MECH-295 / MECH-307 (goal-pipeline consumers), V3-EXQ-622
    (working reference runner that feeds z_goal), V3-EXQ-603d / V3-EXQ-625b
    (the failure records this amend folds), MECH-094 (call-site scoping).

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

## scaffolded_sd054_onboarding AMEND: seeding-calibration + consumption-gated G3 (2026-06-03c)
- scaffolded_sd054_onboarding seeding-calibration amend -- IMPLEMENTED 2026-06-03.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change). Routed by failure_autopsy_V3-EXQ-634b_2026-06-03.
  ROOT CAUSE (verified in code): the 634b developmental-window amend (Stage-0b +
  contact-gating) decisively fixed the decay-only washout (G0b retention 3/3,
  n_decay_only_updates=0) but exposed a benefit-magnitude / threshold mismatch.
  Contact-gating skipped only benefit <= scaffold_p2_contact_benefit_threshold
  (1e-6), but GoalState.update (goal.py:209-224) seeds z_goal only when
  effective_benefit = benefit * z_goal_seeding_gain(1.0) * (1 + drive_weight(2.0)
  * drive_trace) > benefit_threshold(0.1). Natural wild benefit (obs_body[11]
  ~0.03) stays sub-threshold, so the band (1e-6, ~0.1-effective) was NOT skipped
  yet did NOT seed -- it only applied the 0.5%/step decay (goal.py:173), eroding
  the consolidated trace during real foraging. 634b seed 43 (475 P2 contact-refresh
  calls, contact_rate 0.348) collapsed z_goal to ~4.5e-05 while non-foraging seed
  42 "passed" G3 by carrying the untouched forced-feed nursery trace (0.4398,
  byte-identical to Stage-0b-end) -- G3 anti-correlated with foraging.
  THREE coupled fixes (all no-op-default; bit-identical when off):
    (1) DECOUPLED CONTACT-GATING THRESHOLD. The skip/update decision now keys off
        a SEPARATE gating floor (scaffold_contact_gating_benefit_threshold) so
        sub-seeding whiffs in the band (readout_floor, seeding_floor) are PROTECTED
        (skipped, not decay-only updated) while the contact-RATE readout (g2 "was
        the infant fed at all") keeps using scaffold_p2_contact_benefit_threshold.
        Scheduler._gating_threshold() returns the gating floor; sentinel < 0
        (default -1.0) falls back to the readout threshold -> bit-identical to the
        pre-amend 634b path. Wired through _train_episode (P1) + _eval_episode (P2)
        via a new gating_threshold kwarg (None -> readout fallback).
    (2) GOAL-SEEDING MAGNITUDE PROPAGATION. New scaffold knobs
        scaffold_z_goal_seeding_gain / scaffold_benefit_threshold / scaffold_drive_floor
        (all Optional, default None = no-op) are written onto the agent's live
        GoalConfig (agent.goal_state.config) at the top of each seeding-capable
        run_* stage via Scheduler._apply_goal_seeding_calibration(agent). Lets
        genuine wild contact clear the GoalState firing threshold (e.g. gain 1.5 +
        benefit_threshold 0.02 + drive_floor 0.9 -> wild benefit 0.03 yields
        effective 0.126 > 0.02 -> seeds). None leaves GoalConfig untouched ->
        bit-identical. GoalConfig owns the magnitudes (MECH-186/187/188 / SD-012
        precedent); the scaffold propagates them so the 634c sweep can vary them
        through the scaffold's own config surface.
    (3) CONSUMPTION-EVENT-GATED G3 READOUT. P2OnboardingMetrics gains
        z_goal_norm_at_contact_peak (max goal-norm read AT a genuine seeding event,
        632-style) + num_contact_events. _eval_episode captures the goal-norm only
        on a seeding step; stays 0.0 when wild contact never clears the seeding
        floor -- so a z_goal=0-at-contact read is interpretable rather than masked
        by the carried forced-feed nursery trace (z_goal_norm_peak_max). The 634c
        re-validation feeds this consumption-gated peak as the G3 input instead of
        the frozen peak. evaluate_substrate_gate is unchanged (the experiment
        chooses which peak to pass).
  Division of labor: the substrate amend owns the gating-threshold decoupling +
  GoalConfig propagation surface + consumption-gated readout. The SEEDING-MAGNITUDE
  VALUES (gain / benefit_threshold / drive_floor) + the strengthened P0/P1
  foraging-competence budgets are swept per-arm by the V3-EXQ-634c re-validation
  (autopsy: "one or a combination, pick via a small sweep"), with
  scaffold_contact_gating_benefit_threshold matched to the chosen seeding floor.
  Config (ScaffoldedSD054OnboardingConfig, all default no-op):
  scaffold_contact_gating_benefit_threshold=-1.0 (sentinel -> readout fallback);
  scaffold_z_goal_seeding_gain=None; scaffold_benefit_threshold=None;
  scaffold_drive_floor=None.
  Backward compatible: with the sentinel + None defaults the gating decision falls
  back to the readout threshold and GoalConfig is untouched -> bit-identical to the
  pre-amend 634b path. 744/744 contracts (738 prior unrelated + 42 scaffolded incl
  6 new C8) + 7/7 preflight PASS; v3_exq_634b --dry-run unchanged (decay_only=0,
  contact-gating behaviour identical). (The 7 runner git-conflict-recovery FAILs in
  the full suite are a pre-existing local-git env artifact -- "Not a valid object
  name master" -- with zero overlap with this change.)
  Phased training: N/A (harness scheduling/windowing + GoalConfig assignment; no
  learned parameters; no new encoder head). MECH-094: N/A (waking goal-pipeline
  onboarding; no simulation/replay write surface).
  Contracts: tests/contracts/test_scaffolded_sd054_onboarding.py C8 group (6 new:
  c8_seeding_calibration_config_defaults_are_noop; c8_gating_threshold_falls_back_to
  _readout_then_decouples; c8_apply_seeding_calibration_noop_and_applies;
  c8_calibration_applied_by_run_p1; c8_contact_gating_decoupled_protects_subseeding
  _whiff -- the core decoupling: same 0.05 benefit is PROTECTED under gating 0.1 but
  SEEDS under the sentinel; c8_p2_consumption_gated_peak_distinct_from_frozen_peak --
  G3 redesign: frozen peak > 0 from carried trace while consumption-gated peak == 0
  + num_contact_events == 0 when no genuine seeding).
  Validation experiment: V3-EXQ-634c -- multi-arm sweep over {z_goal_seeding_gain,
  benefit_threshold, drive_floor} x strengthened P0/P1 budgets, contact-gating
  threshold matched to the seeding floor, G3 read at consumption events. Queued via
  /queue-experiment. Claim-free (substrate diagnostic). Does NOT queue V3-EXQ-603f;
  603f + ready=false stay blocked until 634c clears a consumption-event-gated gate.
  See scaffolded_sd054_onboarding (parent + prior amends), V3-EXQ-634b (the autopsy
  this addresses; validated the consolidation half), V3-EXQ-634c (validation),
  goal.py GoalState.update (goal.py:209-224 seeding firing threshold; thresholds set
  via GoalConfig, not changed), MECH-112/116/117 (GoalState claims; untouched),
  MECH-186/187/188 (seeding-gain / floor claims whose GoalConfig knobs the scaffold
  propagates), SD-012 (drive_floor / drive_weight), V3-EXQ-632 seed-42 (existence
  proof that seeding produces a correct z_goal when effective_benefit clears the
  floor), MECH-094 (N/A).

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

## scaffolded_sd054_onboarding AMEND: cue-recall FORMATION fix + diagnostics (2026-06-04b)
- scaffolded_sd054_onboarding cue-recall formation amend -- IMPLEMENTED 2026-06-04.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change). Routed by the V3-EXQ-638 cue-silent autopsy
  (manifest v3_exq_638_scaffold_cue_recall_contact_ablation_20260604T142524Z_v3.json:
  C1_cue_fires_on=false, C3_contact_lift=false, FAIL/non_contributory).
  ROOT CAUSE (code-confirmed): the IncentiveTokenBank is EMPTY entering P1/P2, so
  agent.cue_recall_wanting (agent.py:5134) returns 0 at `k not in bank._base_value`
  -> cue_fires=0. Stage-0 forced feeding DOES pass resource_type into update_z_goal,
  but rt=_contacted_resource_type(obs) is almost always None because forced feeding
  is decoupled from standing on a typed cell, so the L2 bank.update bind (gated
  resource_type>0) is never reached. Tokens otherwise only bind on real P1/P2 typed
  contact -- the very GAP-2 failure-to-thrive the cue was meant to bootstrap
  (chicken-and-egg). Compounded by a bare `except: pass` in _maybe_cue_recall that
  made cue_fires=0 undiagnosable.
  TWO no-op-default changes (both bit-identical OFF):
    (A) INSTRUMENTATION. _maybe_cue_recall gains an optional cue_diag accumulator
      (_new_cue_diag()): every non-fire is attributed to a reason
      (no_token / resource_field_absent / proximity_below_threshold / bank_none /
      amp_zero_or_zobject_none / exception:<Type>) and the substrate quantities are
      recorded (n_external_cues_seen, n_cue_recall_attempts/fires, n_token_matches,
      best_prox_peak, drive_peak, token_bank_size, matched_token_strength_peak;
      n_interoceptive_need_cues + n_joint_cues reserved 0 for the next layer). The
      `except: pass` is replaced with an `exception:<Type>` reason -- thrown errors
      are now VISIBLE (still best-effort; never breaks the episode loop). Surfaced
      on P1OnboardingResult.cue_diag + P2OnboardingMetrics.cue_diag +
      Stage0NurseryResult.token_bank_size_end.
    (B) FORMATION FIX. New flag scaffold_stage0_bind_incentive_token (default False).
      When on (and the bridge is enabled), Stage-0 forced feeding binds the token to
      the STRONGEST-PERCEIVED resource type each step (new shared helper
      _strongest_perceived_type, factored from the cue logic so formation and recall
      use IDENTICAL perception -- user-confirmed binding choice 2026-06-04) instead of
      the near-always-None contacted type. Net: the bank is non-empty entering P1/P2,
      so the wild cue can match a token.
  Backward compatible: scaffold_stage0_bind_incentive_token default False -> Stage-0
  rt = _contacted_resource_type, bit-identical; cue_diag empty when no accumulator
  passed. 62/62 scaffold contracts (55 prior + 7 new C9) + 7/7 preflight PASS.
  Activation smoke 2026-06-04 (bridge + bind ON, real REEAgent, 2-ep Stage-0 -> 2-ep
  P1): Stage-0 token_bank_size_end=2 (was 0); P1 cue fires 34x (n_token_matches=34,
  empty nonfire reasons) -- cue_fires=0 -> 34 purely from populating the bank.
  Notable layer-2 signal: P1 drive_peak=0.037 (agent well-fed) -- the cue fires but
  with modest amplitude, which is where interoceptive need-gating (the NEXT pass,
  NOT this one) becomes the right test.
  Phased training: N/A (harness wiring + perception; no learned parameters). MECH-094:
  cue_recall_wanting / update_z_goal carry simulation_mode; the scheduler is a waking
  training stream.
  Contracts: tests/contracts/test_scaffolded_sd054_onboarding.py C9 cont.
  (stage0_bind flag default-noop; _strongest_perceived_type helper; cue_diag no_token
  reason -- the 638 root cause as a unit; resource_field_absent reason; fire+strength
  recorded; exception path attributed not swallowed; Stage-0 binding populates bank).
  Validation experiment: V3-EXQ-638a re-issue (638 with scaffold_stage0_bind_incentive_token
  =True) queued via /queue-experiment -- confirms C1 cue fires + bank populated +
  whether C3 contact lifts in the full 3-seed run. The interoceptive need-gating layer
  + V3-EXQ-638b (OFF / EXTERNAL_ONLY / INTEROCEPTIVE+EXTERNAL arms) is a SEPARATE later
  pass, pending 638a confirming the bank-empty reason dominated.
  See scaffolded_sd054_onboarding cue-recall bridge entry above (the 638 integration
  this fixes), SD-057 / MECH-347 (L6 cue-recall claim; ree_core substrate unchanged),
  goal.py IncentiveTokenBank (the bank this populates), MECH-295 (downstream approach
  bridge), goal_pipeline:GAP-2 (foraging-contact ceiling), V3-EXQ-638 (the cue-silent
  FAIL), V3-EXQ-638a (validation), MECH-094 (N/A).

## scaffolded_sd054_onboarding AMEND: n_cue_recall_fires aggregation fix (2026-06-04c)
- scaffolded_sd054_onboarding cue-fires aggregation fix -- IMPLEMENTED 2026-06-04.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change). The clean underlying fix for the V3-EXQ-638
  measurement gap surfaced 2026-06-04 while validating V3-EXQ-638a.
  ROOT CAUSE (code-confirmed): _eval_episode RETURNS a per-episode
  n_cue_recall_fires and _train_episode ACCUMULATES cue fires into
  goal_write_diag["n_cue_recall_fires"], but run_p2 never aggregated the
  per-episode value onto the returned P2OnboardingMetrics and run_p1 never
  surfaced a total on P1OnboardingResult -- so P2OnboardingMetrics had NO
  n_cue_recall_fires field at all. Any consumer doing
  getattr(p2, "n_cue_recall_fires", 0) therefore silently read 0 EVEN WHEN THE
  CUE FIRED (directly observed: V3-EXQ-638a smoke fired the cue 30x in P2 --
  visible in P2OnboardingMetrics.cue_diag["n_cue_recall_fires"]==30 -- while
  getattr returned 0). The original V3-EXQ-638 script reads
  getattr(p2, "n_cue_recall_fires", 0) for its C1 gate -> 638's C1 would report
  cue_fires=0 even if the cue fired. 638a already works around it by sourcing
  p2.cue_diag["n_cue_recall_fires"]; this is the clean fix so future consumers
  do not trip on it.
  THE FIX (no-op-default; bit-identical when the cue bridge is off -> 0):
    (1) New aggregated field n_cue_recall_fires: int = 0 on BOTH P2OnboardingMetrics
        and P1OnboardingResult.
    (2) run_p2 sums the per-episode ep_metrics.get("n_cue_recall_fires", 0) across
        episodes (mirroring how total_contact / num_contact_events are already
        aggregated there) and sets it on the constructor.
    (3) run_p1 surfaces goal_write_diag.get("n_cue_recall_fires", 0) (already
        accumulated in _train_episode) onto P1OnboardingResult.
  CONTRACT: the new top-level field EQUALS cue_diag["n_cue_recall_fires"] (the
  per-episode returns / goal_write_diag accumulator and the shared cue_diag both
  count the same fires); 0 when the bridge is off. cue_diag is unchanged (it
  already carries the authoritative count).
  Backward compatible: cue bridge off -> both fields default 0; cue_diag empty;
  bit-identical. 65/65 scaffold contracts (62 prior + 3 new C9) + 7/7 preflight PASS.
  Phased training: N/A (pure readout aggregation; no learned parameters).
  MECH-094: N/A (waking-stream readout; no simulation/replay write surface).
  Contracts: tests/contracts/test_scaffolded_sd054_onboarding.py C9 cont.
  (c9_p2_aggregates_cue_recall_fires_equals_cue_diag -- bridge-on nonzero total ==
  cue_diag + getattr no longer reads a silent 0; c9_p2_cue_recall_fires_zero_when_bridge_off
  -- bridge-off 0 == cue_diag; c9_p1_p2_result_field_defaults_zero_on_master_off --
  field exists, defaults 0, master-off short-circuit carries 0 on both results).
  See scaffolded_sd054_onboarding cue-recall FORMATION fix entry above (the
  diagnostics this completes), SD-057 / MECH-347 (L6 cue-recall claim; ree_core
  unchanged), V3-EXQ-638 (the cue-silent FAIL whose C1 read getattr), V3-EXQ-638a
  (the validation that surfaced the gap and already sources cue_diag), MECH-094 (N/A).

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

## scaffolded_sd054_onboarding AMEND: foraging-competence residual (gating-seeding reconcile + reef-spawn weaning + consumption-gated G3) (2026-06-05)
- scaffolded_sd054_onboarding foraging-competence residual -- IMPLEMENTED 2026-06-05.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change). The residual the substrate_queue
  scaffolded_sd054_onboarding title names after the 634c split: 634c validated the
  z_goal SEEDING half (seeded arms g3_zgoal ~0.44; 634b anti-correlation resolved);
  the remaining substrate-gate failure is purely foraging-competence (reach-contact +
  survival) -- the GAP-2 ceiling (seed-42 zero-contact, P1 survival 1/3). 634c dry-run
  shows the live signature: P1 survival_gate=pass but contact_rate=0.0 / contact_events=0
  (the agent survives yet never reaches food, so z_goal is never ecologically seeded).
  Three coupled no-op-default fixes (all bit-identical OFF):
    (1) AUTO-RECONCILE gating floor to GoalState seeding firing threshold. 634c
      decoupled the gating floor (scaffold_contact_gating_benefit_threshold) from the
      contact-rate readout but still had to HAND-MATCH it to the seeding magnitudes as
      a magic number -- a mismatch IS the 634b anti-correlation (scaffold counts a step
      as "seeded" while GoalState.update only decay-updated it). New flag
      scaffold_auto_reconcile_gating_to_seeding (default False): when on,
      Scheduler._reconciled_gating_threshold(agent) DERIVES the raw-benefit gating floor
      from the agent's LIVE GoalConfig each stage --
      benefit_threshold / (z_goal_seeding_gain * (1 + drive_weight * drive_floor))
      (steady-state lower bound, drive_trace >= drive_floor per SD-012 floor
      goal.py:369). Scheduler._effective_gating_threshold(agent) returns this when the
      flag is on, else the static _gating_threshold() (634c path). So the scaffold's
      `seeds` boolean tracks GoalState.update's actual firing -- genuine wild contact
      seeds, sub-seeding whiffs stay protected -- without the experiment keeping the two
      knobs in sync. Recorded on P1OnboardingResult.reconciled_gating_threshold +
      P2OnboardingMetrics.reconciled_gating_threshold.
    (2) GRADED P1 reef-spawn weaning (survival/foraging lever). P0 spawns the agent in
      the reef refuge band (safe); legacy P1 abruptly moves spawn to midline for EVERY
      P1 episode, so a not-yet-competent agent faces the hazard band before its first
      wild contact (603e survival 1/3). New knob scaffold_p1_reef_spawn_hold_fraction
      (default 0.0): keeps reef_bipartite_agent_spawn_in_reef_half=True for the first
      `fraction` of P1 episodes (then midline), extending the developmental safety
      window. _build_env gains a p1_spawn_in_reef_half param; run_p1 records
      n_reef_spawn_episodes. Complements scaffold_p1_anneal_hold_fraction (holds the
      hazard/food-attraction anneal low) -- this holds the SPAWN safe. Paired with the
      SD-057 cue-recall bridge (contact lever) so the agent both survives long enough
      AND has a path to approach perceived food.
    (3) CONSUMPTION-EVENT-GATED G3 as the canonical mature-test readout. The 634c
      z_goal_norm_at_contact_peak field (z_goal read AT a genuine 632-style seeding
      event) is now the DEFAULT G3 input via new module helper
      substrate_readiness_from_results(stage0_results, p1_results, p2_metrics, *,
      use_consumption_gated_g3=True). A seed carrying an untouched Stage-0 nursery trace
      through a zero-contact P2 (the seed-42 artifact) reads g3=0 -- G3 cannot be passed
      by a non-foraging seed. use_consumption_gated_g3=False falls back to the frozen
      z_goal_norm_peak_max for comparison only; the dict carries g3_source.
  Backward compatible: master switch + every new knob default inert; all defaults
  no-op. 79/79 scaffold contracts (70 prior + 9 new C11) + 7/7 preflight PASS;
  v3_exq_634c --dry-run runs unchanged end-to-end (my new flags off -> bit-identical;
  dry-scale FAIL is the known dry-scale outcome). Contracts:
  tests/contracts/test_scaffolded_sd054_onboarding.py C11 group (config no-op defaults;
  effective-gating falls back to static when off; reconciled floor derives from
  GoalConfig + arithmetic; reconciled None without goal_state; _build_env p1 spawn-param;
  reef-spawn-hold default all-midline + early-P1-spawns-in-reef via run_p1; readiness
  helper uses consumption-gated G3 by default + frozen-peak fallback + full-pass on
  genuine contact).
  Phased training: N/A (harness scheduling + arithmetic; no learned params).
  MECH-094: N/A (waking goal-pipeline onboarding; no simulation/replay write surface).
  Acceptance target (unchanged, 603f gate): z_goal_norm_at_contact_peak > 0.4 on >=2/3
  seeds AND P1 survival >=2/3 AND non-zero contact >=2/3 -- evaluated by
  substrate_readiness_from_results (consumption-gated G3).
  Validation: substrate-readiness EXQ queued via /queue-experiment (combines cue-recall
  bridge + reef-spawn weaning + auto-reconcile + 634c seeding calibration + strengthened
  budgets) -- the full-scale gate. substrate_queue.ready STAYS false until that run
  clears the consumption-gated gate on >=2/3 seeds; V3-EXQ-603f stays blocked until then.
  Design doc: REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md
  (Amend 2026-06-05 section). Substrate_queue: scaffolded_sd054_onboarding.
  See scaffolded_sd054_onboarding (parent + prior amends), SD-057 cue-recall bridge
  (the complementary contact lever), goal.py GoalState.update (goal.py:369/383-388
  seeding firing math the reconciliation derives from; thresholds set via GoalConfig,
  not changed), V3-EXQ-634c (the seeding-half validation this builds on), V3-EXQ-632
  seed-42 (consumption-gated existence proof), goal_pipeline:GAP-2 (the reach-contact
  ceiling), Q-045 / MECH-313 / MECH-260 (cluster gated on 603f), MECH-094 (N/A).

## scaffolded_sd054_onboarding AMEND: curriculum decomposition -- isolated hazard-avoidance stage (Stage-H) (2026-06-07)
- scaffolded_sd054_onboarding curriculum-decomposition amend -- IMPLEMENTED 2026-06-07.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change). Routed by failure_autopsy_V3-EXQ-603f_2026-06-07
  (substrate-readiness FAIL, self-route substrate_not_engaged/foraging_competence_open,
  confirmed). 603f PROVED the goal-formation + ecological-seeding chain is SOUND --
  seed 44 foraged (P2 contact_rate 0.393, 85 events) AND seeded z_goal ecologically
  (z_goal_norm_at_contact_peak 0.450 > 0.4) -- so this is NOT a goal-stream change.
  The single remaining GAP-2 blocker is the P1 SURVIVAL / hazard-avoidance leg
  (G1 0/3; median episode len 12.5/38.0/28.5 vs gate 75; even the foraging seed 44
  died at 28.5). ROOT CAUSE: P1 couples TWO competencies at once (goal-pipeline
  unfreeze + wean into the hazard band) and the agent cannot acquire both
  simultaneously; P0 trains only in the safe reef refuge, so the agent never learns
  hazard navigation before P1 throws it at hazards.
  THE FIX (user-directed, AskUserQuestion 2026-06-07 "Stage-H only"): a SEPARATELY-
  TRAINED isolated hazard-avoidance stage (Stage-H) inserted between P0 (safe,
  goal-frozen warm-up) and P1 (combined wean). The competencies are trained in
  isolation: survival/avoidance alone in Stage-H, then the existing goal-unfreeze +
  final-hazard-ramp in P1, now entered by an already-survival-AND-goal-competent
  policy. Legs (1) safe goal-attainment and (3) combined wean are covered by the
  existing Stage-0 / P0 / early-P1 levers (extendable via budgets + anneal/reef-spawn
  holds in the 603g config); the optional forced-choice micro-env variant is deferred
  (would need a ree_core env mode).
  Changes (all behind scaffold_hazard_stage_enabled, default False, bit-identical OFF):
    (1) New method ScaffoldedSD054OnboardingScheduler.run_hazard_avoidance(agent, device)
        + HazardAvoidanceResult dataclass. Goal pipeline FROZEN
        (_set_goal_pipeline_frozen(frozen=True), seed_goal=False -> update_z_goal
        never called, z_goal untouched -- the isolation); trains E1+E2 (+E3
        running-variance) exactly like run_p0; measures median episode length over
        the last scaffold_hazard_stage_stability_window episodes vs
        scaffold_hazard_stage_survival_gate_steps (G_H survival readout, DIAGNOSTIC
        ONLY -- does NOT abort the curriculum or change the canonical G0/G1/G2/G3
        readiness gate). Aborts (hazard_stage_disabled / master_switch_off) when the
        flag / master switch is off.
    (2) New _build_env phase "hazard": hazards present
        (scaffold_hazard_stage_num_hazards default 4), foraging minimal
        (scaffold_hazard_stage_num_resources default 2),
        hazard_food_attraction=0.0 (hazards drift randomly so foraging does NOT raise
        hazard exposure -- clean avoidance signal),
        proximity_harm_scale=0.1 (target level so avoidance is incentivised), midline
        spawn (scaffold_hazard_stage_spawn_in_reef_half default False so the agent
        must navigate the hazard band; the reef refuge stays available as the
        flee-to-safety attractor). SAME structural kwargs (reef + bipartite + SD-049 +
        limb_damage) as every other phase -> world_obs_dim matches the single shared
        agent (verified by contract).
    (3) Config (ScaffoldedSD054OnboardingConfig, all no-op default):
        scaffold_hazard_stage_enabled (False) + _episode_budget (40) + _num_hazards (4)
        + _num_resources (2) + _hazard_food_attraction (0.0) + _proximity_harm_scale
        (0.1) + _spawn_in_reef_half (False) + _survival_gate_steps (75) +
        _stability_window (10).
  Curriculum becomes: Stage-0 (forced feed) -> Stage-0b (consolidate) -> P0 (encoder
  warm-up, goal frozen) -> Stage-H (isolated hazard avoidance, goal frozen) -> P1
  (combined wean) -> P2 (measure).
  Backward compatible: master switch + every new knob default inert; run_hazard_avoidance
  aborts when disabled; existing scripts never call it; STAGE_PLAN unchanged (the
  5-stage conceptual readout + its contract are untouched). 85/85 scaffold contracts
  (79 prior + 6 new C12) + 7/7 preflight PASS; v3_exq_603f --dry-run runs unchanged
  (same dry-scale FAIL outcome; the hazard stage is not enabled in 603f).
  Activation (C12 contracts, 2026-06-07): enabled Stage-H runs goal-frozen
  (mech295/mech307 short-circuited after), leaves a pre-seeded z_goal byte-unchanged
  (no update_z_goal call), populates the survival readout; "hazard" env emits the
  configured hazards at the same world_obs_dim as p0/p2 and spawns midline by default.
  Phased training: N/A (the scheduler IS phased training; Stage-H is an additional
  E1/E2/E3 warm-up phase with the goal pipeline frozen -- no new encoder head, no new
  latent target, no collapse risk). MECH-094: N/A (waking goal-pipeline onboarding;
  no simulation/replay write surface).
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default curriculum-structure
  amend; every existing experiment uses the default (stage disabled), so no dependent
  claim's measured mechanism changed. KEEP all evidence.
  Contracts: tests/contracts/test_scaffolded_sd054_onboarding.py C12 group (config
  no-op defaults; disabled-aborts; "hazard" env obs-dim parity + hazard count + midline
  spawn; optional reef-half spawn; enabled run goal-frozen + z_goal-untouched + survival
  readout; master-off aborts).
  Validation experiment: V3-EXQ-603g substrate-readiness diagnostic queued via
  /queue-experiment -- copy of 603f with Stage-H inserted (scaffold_hazard_stage_enabled
  ON) and a G_H survival diagnostic, against the SAME G0/G1/G2/G3 gate. GAP-2 stays
  blocked_pending_substrate until 603g clears G1>=2/3 AND G2>=2/3 AND ecological
  G3>=2/3. substrate_queue.ready STAYS false until then.
  Design doc: REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md
  (Amend 2026-06-07 section). Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603f_2026-06-07.{md,json}.
  NOTE: 640b (GAP-7 cue-authority; SD-057/MECH-346/MECH-347) is a SEPARATE thread -- do
  NOT conflate. The 603f script's pre-registered "640b selection-authority is next"
  routing was superseded for GAP-2 by the seed-44 disambiguator (foraged + seeded yet
  died -> cue-authority cannot fix the survival leg).
  See scaffolded_sd054_onboarding (parent + prior amends), goal_pipeline:GAP-2 (the
  P1 survival leg this closes), V3-EXQ-603f (the FAIL this amend addresses), V3-EXQ-603g
  (validation), Q-045 / MECH-313 / MECH-260 / SD-049 / SD-015 / MECH-229 / MECH-230 /
  MECH-117 / MECH-216 / ARC-030 / ARC-032 / Q-030 (the GAP-2 cohort gated on 603g),
  modulatory-bias-selection-authority + V3-EXQ-640b (separate GAP-7 cue-authority
  thread, NOT a GAP-2 fix), MECH-094 (N/A).

## scaffolded_sd054_onboarding AMEND: Stage-H harm-pathway training (603i nav/survival-competence ceiling) (2026-06-09)
- scaffolded_sd054_onboarding harm-pathway-training amend -- IMPLEMENTED 2026-06-09.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change -- the block calls EXISTING ree_core harm heads).
  Routed by failure_autopsy_V3-EXQ-603i_2026-06-08 (PRIMARY: nav/survival-competence
  ceiling; ARM_NAV_CONTROL spawn-in-reef G_H=0.0) + the 603g/624c/651a cluster autopsy
  ("deeper than budget") + lit verdict targeted_review_hazard_avoidance_learning.
  ROOT CAUSE (code trace + empirical probe 2026-06-09, NOT budget): across the ENTIRE
  curriculum (Stage-0/0b/P0/Stage-H/P1/P2), _train_episode optimizes ONLY E1
  (compute_prediction_loss, LSTM MSE) + E2.world_transition/world_action_encoder. The
  hazard-avoidance VALUATION pathway is never in any optimizer:
  E3.harm_eval_head(z_world) -- the harm cost that scores EVERY candidate trajectory
  in E3.select (e3_selector.py:419/564) -- is a near-constant ~0.523 (random init;
  measured range [0.522,0.524] over 300 states), the HarmEncoder z_harm output is
  constant (zero variance vs proximity), and `_harm_signal` from env.step is DISCARDED.
  So the agent navigates a RANDOM harm landscape and dies even handed the reef refuge
  (probe: Stage-H survival slope -0.94 steps/ep == no learning; 24/25 episodes die
  early; median last-window 23 vs gate 75). More budget cannot train a head that is
  not in the loss. This is why SD-058/MECH-357 (IA gate) + SD-059/MECH-358 (bridge)
  were "engaged but insufficient": they bias action selection within whatever harm
  landscape E3 hands them, and that landscape is noise.
  THE FIX (user-confirmed FULL scope + encoder co-train): train the existing-but-
  untrained harm pathway during P0 + Stage-H, supervised by the env hazard-proximity
  label (harm_obs centre, SD-010/SD-018) + accumulated-harm scalar (SD-011). Four
  independently-toggleable terms (Q-044 / MECH-314a-style ablatability):
    (1) harm_eval(z_world): E3.harm_eval_head + the z_world encoder -- the proximity
        MSE backprops INTO latent_stack (SD-018 semantics) so z_world becomes
        hazard-discriminative (head-only fails on the flat z_world the probe measured).
        LOAD-BEARING: makes the trajectory-rollout harm landscape (E2.world_forward ->
        harm_eval_head) predictive.
    (2) z_harm sensory: HarmEncoder + E3.harm_eval_z_harm_head (SD-010) on the same label.
    (3) z_harm_a affective: AffectiveHarmEncoder via compute_harm_accum_loss (SD-011) --
        gives MECH-279 PAG + SD-058 IA gate + SD-059 bridge a TRAINED threat signal.
    (4) E2_harm_s forward: E2HarmSForward (ARC-033) on FROZEN (detached) z_harm_s for
        multi-step harm lookahead.
  Config (ScaffoldedSD054OnboardingConfig, all no-op default): scaffold_train_harm_pathway
  (False, master) + scaffold_train_harm_eval_head / _z_harm_sensory / _z_harm_affective /
  _e2_harm_s_forward (True; consulted only when master on) + scaffold_harm_pathway_lr
  (1e-3) + scaffold_harm_pathway_in_p0 (True) + scaffold_harm_s_buf_max (2000).
  New module helpers: _hazard_proximity_target / _accumulated_harm_target (target
  extraction from obs_dict), _harm_pathway_params (deduped param union per enabled
  term), _harm_pathway_step (one co-train step on the pristine post-sense latent + a
  detached-buffer E2_harm_s step), _measure_harm_discriminativeness (post-training
  NON-VACUITY readout). Scheduler._make_harm_pathway builds the optimizer/buffer/diag;
  run_p0 (when in_p0) + run_hazard_avoidance thread train_harm + harm_opt + harm_s_buf
  + harm_diag through _train_episode; run_hazard_avoidance also runs the
  discriminativeness probe at the end. HazardAvoidanceResult / P0OnboardingResult gain
  harm_pathway_enabled + harm_pathway_diag (+ harm_discriminativeness on Stage-H).
  Terms 2 + 4 require the agent built with use_harm_stream=True (sensory z_harm) +
  use_e2_harm_s_forward=True; inert no-op (correctly skipped) when z_harm / e2_harm_s
  are absent -- the 603k validation config enables both so all four terms engage.
  Backward compatible: master OFF -> train_harm False, harm_opt None, harm block + the
  Stage-H discriminativeness probe skipped -> bit-identical. 97/97 scaffold contracts
  (91 prior + 6 new C15) + 7/7 preflight PASS. Activation smoke 2026-06-09 (ARM_NAV_CONTROL
  spawn-in-reef, harm pathway ON, reduced budget): survival_gate_passed=True,
  median_last_window=80.0 (vs OFF probe 23.0, gate 75); survival slope +0.19 (vs OFF
  -0.94); early deaths 9/25 (vs OFF 24/25); harm_eval loss trains (n_train_steps
  586 P0 + 1511 Stage-H). The headline (G_H 0 -> survival_passed) confirms the fix.
  Phased training: the curriculum IS phased; harm_eval/z_harm proximity heads co-train
  with their encoder (single SD-010/SD-018 regression, no collapse risk); E2_harm_s
  trains on detached z_harm_s (ARC-033 P1 phasing). MECH-094: N/A -- waking training
  stream; no simulation/replay write surface.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (harm pathway off), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  Validation experiment: V3-EXQ-603k substrate-readiness diagnostic (claim_ids=[];
  HARM_OFF vs HARM_ON ablation on the 603i-INTACT base + nav-competence reef spawn;
  use_harm_stream=True + use_e2_harm_s_forward=True so all four terms engage). Acceptance
  per the 603i failure-record target: Stage-H G_H >= 2/3 (median last-window >= 75) with
  nav-to-safety handed AND a non-vacuity gate (harm_eval discriminativeness lifts above
  the flat baseline) AND the HARM_OFF arm reproduces G_H ~ 0. PASS unblocks the
  escape-affordance-bridge retest (the bridge can finally be scored once survival clears)
  + the GAP-2 survival-leg cohort. substrate_queue.ready STAYS false until 603k clears.
  Design doc: REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md
  (Amend 2026-06-09 section). Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603i_2026-06-08.{md,json}.
  See scaffolded_sd054_onboarding (parent + prior amends incl Stage-H curriculum
  decomposition), SD-058/MECH-357 (IA gate -- bias side; this trains the valuation it
  navigates), SD-059/MECH-358 (escape bridge -- blocked on this survival leg),
  SD-010/SD-018 (z_world/z_harm hazard-proximity supervision), SD-011 (z_harm_a
  accumulated-harm), ARC-033 (E2_harm_s forward), MECH-279 (PAG freeze; keys on the
  now-trained z_harm_a), goal_pipeline:GAP-2 (the survival leg this closes),
  V3-EXQ-603i (the FAIL this addresses), V3-EXQ-603k (validation), MECH-094 (N/A).

## scaffolded_sd054_onboarding AMEND: harm-pathway training STABILIZATION (decoupled encoder LR + LR warmup; 603p seed-fragility) (2026-06-16)
- scaffolded_sd054_onboarding harm-pathway-stabilization amend -- IMPLEMENTED 2026-06-16.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change). Routed by the confirmed failure_autopsy_V3-EXQ-603p_2026-06-15
  (Branch B, user-confirmed) + the GAP-C node's BLOCKED-ON-HARM-PATHWAY-STABILIZATION-AMEND
  route. behavioral_diversity_isolation:GAP-C; ISOLATED harm-valuation subsystem (no GAP-A overlap).
  ROOT CAUSE (V3-EXQ-603p, claim-free Stage-H base-harm-landscape diagnostic): the base harm
  landscape (E3.harm_eval_head(z_world)) forms a discriminative head (harm_eval_range >= 0.02)
  on only 1/3 seeds at the EASIEST regime (proximity_harm=0.10), and tripling the global
  harm-pathway LR to 3e-3 COLLAPSES it to ~1e-23 on all three seeds. The 2026-06-09 harm-pathway
  optimizer puts the harm params in a SINGLE Adam group at scaffold_harm_pathway_lr (1e-3): the
  same LR co-trains the latent_stack ENCODER (the SD-018 proximity MSE backprops into it so
  z_world becomes hazard-discriminative) AND the harm HEADS. Raising that one LR drives the
  encoder to the trivial constant-z_world solution (range -> 0, the 3x-LR collapse), while 1e-3
  leaves most seeds under-converged. The mechanism is RIGHT (where the landscape forms, prox_corr
  0.44-0.83); convergence/seed-robustness is the gap. NOT a budget tweak and -- per the autopsy --
  NOT a global-LR raise (it collapses the landscape).
  THE FIX (two no-op-default levers; bit-identical OFF; stabilize WITHOUT raising the global LR):
    (1) scaffold_harm_pathway_encoder_lr (Optional[float], default None): when set, the latent_stack
      ENCODER params get their own Adam param group at this (typically LOWER) LR while the harm
      heads + E2_harm_s keep scaffold_harm_pathway_lr -- the encoder moves gently (escaping the
      collapse-to-constant basin) while the heads still extract the proximity mapping at the base
      rate. None -> single Adam group at scaffold_harm_pathway_lr (bit-identical to the pre-amend
      flat optimizer). New helper _harm_pathway_param_groups builds the two disjoint groups
      (shared encoder-first dedup so no param appears twice).
    (2) scaffold_harm_pathway_warmup_steps (int, default 0): linear LR warmup over the first N
      harm-pathway training steps -- scales every param group's LR from base/N up to base, then
      holds at base, easing the early-training basin where the encoder is most prone to the
      collapse (gradient stabilization). 0 -> no scaling (bit-identical).
  Both levers cover the autopsy's primary prescriptions (lower [encoder] LR with the heads still
  training at base + gradient stabilization); the "more training steps" candidate stays available
  via the existing budget knobs (scaffold_p0_episode_budget / scaffold_hazard_stage_episode_budget).
  Optimizer construction (_make_harm_pathway): enc_lr=None keeps the EXACT legacy
  optim.Adam(params, lr=base); enc_lr set uses optim.Adam(param_groups). Either way the per-group
  base LRs are stashed on harm_opt._harm_base_lrs for the warmup scaling in _harm_pathway_step
  (applied per param group right before the existing grad-clip + step; n_train_steps is the 0-based
  step index).
  Backward compatible: scaffold_harm_pathway_encoder_lr=None + scaffold_harm_pathway_warmup_steps=0
  by default -> single-group Adam at base LR, no LR scaling -> bit-identical to the 2026-06-09
  harm-pathway optimizer. 102/102 scaffolded contracts (97 prior + 5 new C16: stabilization config
  defaults no-op / encoder_lr=None single legacy group / encoder_lr decouples into two disjoint
  groups at distinct LRs / warmup ramps base/N -> base then holds, driven through real
  _harm_pathway_step calls / warmup=0 leaves LR at base) + 7/7 preflight PASS.
  Phased training: N/A (the curriculum IS phased; this changes only the harm-pathway optimizer's
  param-group LRs + a warmup schedule -- no new encoder head, no new latent target, no collapse
  risk introduced). MECH-094: N/A (waking goal-pipeline onboarding; no simulation/replay write
  surface). Evidence-staleness (Step 8.5): NOT triggered -- no-op-default levers; every existing
  experiment uses the defaults (single-group LR, no warmup), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  In-session validation (proof-of-fix probe; scripts/_validate_603q_harm_amend.py replicates the
  603p positive-control cell via 603p's own config builders): the levers run END-TO-END in the
  real 603p pipeline and the harm head learns the correct proximity mapping (prox_corr positive
  even at reduced budget). The full-scale >=2/3-seed seed-robustness confirmation at
  proximity_harm=0.10 is carried as V3-EXQ-603q's FIRST self-routing non-vacuity precondition
  (the cloud establishes it; 603q self-routes substrate_not_ready_requeue if the base does not
  clear >=2/3, never a false bridge verdict) -- per the GAP-C durable 603q spec.
  Validation experiment: V3-EXQ-603q (the corrected SD-059/MECH-358 escape-affordance-bridge
  EVIDENCE re-run, bridge ON vs base) runs on the now-stabilized base with the two levers ON +
  the base-harm-landscape >=2/3 discriminativeness as a self-routing precondition. See Step 8.
  Design doc: REE_assembly/docs/architecture/sd_054_scaffolded_onboarding_substrate_design.md
  (Amend 2026-06-16 section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603p_2026-06-15.md.
  See scaffolded_sd054_onboarding (parent + the 2026-06-09 Stage-H harm-pathway-training amend
  this stabilizes), SD-018 / SD-010 / SD-011 / ARC-033 (the harm-proximity supervision targets),
  SD-059 / MECH-358 (the escape-affordance bridge blocked on a discriminative base harm landscape),
  MECH-313 / MECH-260 / Q-045 (the GAP-C tonic-noise cluster gated downstream of 603q),
  behavioral_diversity_isolation:GAP-C (closure node), V3-EXQ-603p (the FAIL this amend addresses),
  V3-EXQ-603k (the narrow 603p-superseded probe whose "VALIDATED 2026-06-09" status was over-stated),
  V3-EXQ-603q (validation/evidence), MECH-094 (N/A).

## scaffolded_sd054_onboarding AMEND: Leg C rule_bias_head training (commitment_closure:GAP-4) (2026-06-16)
- scaffolded_sd054_onboarding rule-bias-training amend -- IMPLEMENTED 2026-06-16.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change -- the block calls the EXISTING SD-033a/ARC-062 GAP-D
  lateral_pfc.compute_bias / bias_head_parameters substrate, landed 2026-05-17).
  commitment_closure:GAP-4; the third leg of the SD-034 commitment-closure-control-plane
  (Legs A env-completion-hook + B de-commit-refractory landed 2026-06-12; Leg C was
  marked "experiment-side" and never actually built). Routed by the confirmed
  failure_autopsy_SD-034-closure-control-plane-d_2026-06-13.
  ROOT CAUSE (code-confirmed by the autopsy): the V3-EXQ-460d/468d *d retests set
  lateral_pfc_train_rule_bias_head=True (un-zeroing the GAP-D head's last Linear) but
  NEVER added it to any optimizer -- grep for optim|Adam|.backward(|bias_head_parameters
  returned ZERO matches in either script beyond the config mention. So the head stayed
  at random init: the rule_state handed to the SD-034 ClosureOperator carried no
  task-shaped magnitude -> the automatic rule-stability detector stayed inert and the
  closure-coupled de-commit had no net authority over the MECH-090 latch (460d C2_beta_release
  / C4 FAIL: ON latch occupancy >= OFF on seeds 43/44) and the agent committed-without-beta
  on 2/3 seeds (468d total_beta_elevated=0). NOT a falsification of SD-034/MECH-261.
  THE FIX (no-op-default; bit-identical OFF): a scaffold_train_rule_bias_head leg that
  trains agent.lateral_pfc.bias_head_parameters() during P1 (goal-unfrozen, ecological-
  contact, commitment forms) via the V3-EXQ-598b outcome-coupled E3-gradient REINFORCE
  pattern -- mirroring the existing scaffold_train_harm_pathway leg. Episode-level (not
  per-step like the harm pathway): run_p1 builds the optimizer + a persistent runtime
  (outcome buffer + EMA return baseline) via Scheduler._make_rule_bias_pathway; each
  _train_episode records a (candidate_features=world_states[1] of the leading n_probe
  candidates, selected-candidate index) snapshot every N steps and accumulates the
  episode return (-harm); at episode end _rule_bias_episode_update takes one Adam step --
  advantage = ep_return - EMA baseline; bias = lateral_pfc.compute_bias(candidate_features)
  recomputed (gradient flows into the head); loss = mean(-adv * log_softmax(-bias/T)[sel]);
  grad-clip 1.0; step. New module helpers: _rule_bias_params (the trainable-head guard),
  _new_rule_bias_diag, _build_rule_bias_snap, _selected_candidate_index,
  _rule_bias_reinforce_loss (the 598b _lpfc_reinforce_loss, parameterised off cfg),
  _rule_bias_episode_update.
  REQUIRES the agent built with use_lateral_pfc_analog=True AND
  lateral_pfc_train_rule_bias_head=True (the GAP-D un-zero flag); with the head
  zeroed-and-frozen OR no lateral_pfc, _rule_bias_params returns [] -> optimizer None ->
  the leg is a clean inert no-op (a misconfig is surfaced as rule_bias_pathway_enabled=False
  on the P1 manifest, never silently trains the baseline-OFF head).
  Config (ScaffoldedSD054OnboardingConfig, all no-op default -> bit-identical OFF; NOT
  surfaced through REEConfig.from_dims, matching the scaffold_train_harm_pathway / SD-054
  env-only scheduler-config precedent -- the 460e experiment sets them on the scheduler
  config directly): scaffold_train_rule_bias_head (False, master) + scaffold_rule_bias_lr
  (5e-4) + _batch_size (32) + _record_every_n_steps (4) + _outcome_buf_max (512) +
  _n_probe_candidates (8) + _policy_temperature (1.0) + _adv_min_threshold (0.005) +
  _ema_decay (0.9) (the 598b constants). P1OnboardingResult gains rule_bias_pathway_enabled
  + rule_bias_diag (counters + live per-candidate |bias| samples for the non-vacuity gate).
  Backward compatible: scaffold_train_rule_bias_head=False by default -> _make_rule_bias_pathway
  returns (None,None,None); the per-step snap collection + per-episode REINFORCE are skipped;
  the _train_episode early-return-on-done was refactored to for/else + single post-loop return
  (value-identical). 109/109 scaffolded contracts (102 prior + 7 new C17: config no-op /
  pathway off->None / on->Adam over the head params / inert-when-zeroed-or-no-lpfc /
  snap+selection-match / REINFORCE loss gradient reaches the head [the load-bearing inversion
  of the 460d gap] / run_p1 trains ON-vs-bit-identical-OFF) + 7/7 preflight PASS;
  v3_exq_460d --dry-run unchanged (leg off). Activation smoke 2026-06-16 (run_p1, tiny scale,
  leg ON): rule_bias_head last-Linear max|dW|=0.0015 > 0 (head TRAINS -- the 460d bug
  inverted), 3 REINFORCE steps over 57 snaps, mean per-candidate |bias|=0.039 (non-trivial,
  vs the ~0 the untrained 460d head produced); leg OFF -> max|dW|=0.0 exactly (bit-identical).
  Phased training: correctly phased by construction -- P0 warms the encoder (goal frozen),
  the rule_bias_head trains in P1 AFTER warmup (the 598b P0->P1 discipline). No new encoder
  head; the head is the existing GAP-D substrate. MECH-094: N/A -- the REINFORCE training is
  on the waking P1 training loop (no simulation/replay write surface; lateral_pfc.update keeps
  its existing MECH-319 simulation gate). Evidence-staleness (Step 8.5): NOT triggered --
  no-op-default flag; every existing experiment uses the default (leg off), so no dependent
  claim's measured mechanism changed. KEEP all evidence.
  ML/AI engineering notes: outcome-coupled REINFORCE on a policy bias head carries the
  standard high-variance-gradient hazard -- mitigated exactly as 598b: EMA return baseline
  (advantage = return - baseline), an absolute advantage-floor skip (near-baseline episodes
  carry no signal), grad-norm clip 1.0, and an episodic outcome buffer for batched replay.
  The compute_bias +/-bias_scale clamp can saturate the gradient at random init (the SD-033b
  GAP-8 clamp note); the per-candidate variation + small bias_scale keep some candidates
  in-band (598b-proven for this exact head).
  Validation experiment: V3-EXQ-460e (supersedes 460d), queued via /queue-experiment -- the
  closure-control-plane re-run with scaffold_train_rule_bias_head=True +
  lateral_pfc_train_rule_bias_head=True + a non-cap-pinned ON<OFF latch-occupancy-drop DV for
  C2_beta_release + a beta-engagement non-vacuity gate + a rule_bias-magnitude readiness gate
  (rule_bias_diag mean |bias| > floor, else substrate_not_ready_requeue). Acceptance per the
  autopsy failure record: ON<OFF de-commit on a non-cap-pinned statistic on >=2/3 seeds with
  beta-engagement met. substrate_queue commitment-closure-control-plane ready STAYS false
  until 460e scores a contributory PASS.
  Design doc: REE_assembly/docs/architecture/sd_034_governance_closure_operator.md
  (Leg C rule-bias-training amend section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_SD-034-closure-control-plane-d_2026-06-13.{md,json}.
  See SD-034 (parent closure operator + commitment-closure-control-plane Legs A/B),
  SD-033a GAP-D / ARC-062 GAP-D lateral_pfc_train_rule_bias_head (the substrate this trains;
  landed 2026-05-17), V3-EXQ-598b (the REINFORCE pattern mirrored), scaffold_train_harm_pathway
  (the sibling scaffold training leg whose structure this mirrors), MECH-090 (the latch the
  de-commit authority acts on), MECH-260 (No-Go; 460d supports), MECH-261 (mode-conditioning;
  unexercised), MECH-268 (dACC PE; 468d), V3-EXQ-460d/468d (the FAILs this addresses),
  V3-EXQ-460e (validation), MECH-094 (N/A).

## ControlVector logging: four-signal control telemetry (rec-B, 2026-06-07)
- telemetry.control_vector_logging -- IMPLEMENTED 2026-06-07. Read-only,
  default-OFF telemetry from the four-signal control adjudication (2026-06-07):
  makes value / effort / opportunity-cost-of-time / vigor separately inspectable
  each E3 tick and EXPOSES the ARC-068-vs-MECH-320 collapse (opportunity cost and
  vigor are both w*v_t for the SAME MECH-320 v_t scalar -- ARC-068 is registered
  but unbuilt; its lit-pull is pending). Recommendation B (logging only); the
  causal first-class opportunity-cost split (C) + full four-axis controller (D)
  are DEFERRED post-green-board, gated on the ARC-068 lit-pull and MECH-320
  regaining selection authority (V3-EXQ-643a / SD-056 / ARC-065 GAP-A).
  Modules:
    ree_core/utils/config.py -- REEConfig.use_control_vector_logging (bool,
      default False) + from_dims kwarg + assignment. Bit-identical OFF.
    ree_core/cingulate/dacc.py -- bundle gains two additive keys control_required
      (float) + effort_term (control_required * candidate_effort, [K]) so C_effort
      is readable without re-deriving from (payoff - mode_ev). No existing consumer
      reads them.
    ree_core/predictors/e3_selector.py -- self.last_raw_scores (pre-bias
      per-candidate scores = value axis / V_outcome), stored alongside last_scores.
    ree_core/agent.py -- _last_control_vector dict + _cv_vigor cache; the MECH-320
      tv_bias action/no-op split + v_t/w_action/w_passive are cached in the
      tonic_vigor block; REEAgent._assemble_control_vector() writes
      _last_control_vector after e3.select. Read directly by experiment scripts
      (the V3-EXQ-571 _last_score_bias_decomp pattern).
  Schema (_last_control_vector): V_outcome {mean,range,std,value_mean,present} |
    C_effort {mean,range,std,control_required,present} | C_time
    {potential=w_passive*v_t, realised_mean, n_noop_candidates, present} | G_vigor
    {potential=w_action*v_t, realised_mean, noise_floor_temp_lift,
    n_action_candidates, present} | shared {tonic_vigor_v_t, tonic_vigor_v_raw,
    w_action, w_passive, collapse_note} | authority
    {modulatory_authority_active, scale_factor, e3_raw_score_range_mean}.
    shared.tonic_vigor_v_t is logged so C_time.potential and G_vigor.potential are
    both computable as w*v_t for ONE scalar -- the collapse made inspectable.
  Backward compatible: use_control_vector_logging=False by default ->
    _last_control_vector stays {}; bit-identical action stream (contract C4).
    889 contracts + 7 preflight PASS; 4 new contracts in
    tests/contracts/test_control_vector_logging.py (C1 OFF no-op / C2 ON populates
    four signals / C3 collapse C_time,G_vigor == w*v_t same v_t / C4 bit-identical
    OFF-vs-ON action stream). Activation smoke: v_t=0.5 (forced floor; v_raw=-1.75
    -- the documented EXQ-624a sign/scale issue, now visible in telemetry),
    C_time=G_vigor=0.05 (one scalar, two weights).
  Phased training: N/A (read-only telemetry; no learned parameters). MECH-094: N/A
    (waking action-selection readout; no replay/memory write surface).
  Validation experiment: Stage-B C_time<->G_vigor collapse-correlation diagnostic
    queued via /queue-experiment (claim_ids=[]; pre-registered rho ~ 1.0).
  Design doc: REE_assembly/docs/architecture/control_vector_logging.md
  See MECH-320 (the collapse site), ARC-068 (unbuilt opportunity-cost claim;
    post-green-board), MECH-313 (G_vigor temperature lift), SD-032b dACC (C_effort
    source), modulatory-bias-selection-authority + V3-EXQ-643a (authority context),
    V3-EXQ-571 (_last_score_bias_decomp pattern extended), V3-EXQ-624a (MECH-320
    v_t sign/scale issue surfaced), MECH-094 (N/A).

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

## Arm-Reuse Baselines (`experiments/_lib/baselines/`)
Canonical OFF/baseline modules live in `experiments/_lib/baselines/<lineage>.py` (today: `exq610_inv074_crystallization_baseline`, `exq643_modulatory_authority_baseline`), each exposing `build_off_arm(seed)` / `train_off_arm(...)` / `off_path_config_slice()`. **Save a baseline here by default** when a multi-arm family will re-run an *expensive*, *frozen-substrate* OFF arm on the *cloud* class: factor it into a module, then queue a low-priority cloud mint (`experiment_purpose="baseline"`, emit with `include_driver_script_in_hash=False`) so later iterations skip re-training it. The producer recipe + WHEN gate are in the `/queue-experiment` skill ("Saving a baseline for reuse"); design/validity in `REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md` §7b/§9. The module is auto-bound into `substrate_hash` via the `_lib/**` glob, so any edit to it correctly refuses a stale reuse (a false miss is free; a false hit corrupts science).

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

## ARC-065 SP-CEM Main-Path Landing (2026-05-17)
- ARC-065 hippocampal-trajectory-sampling child (support-preserving + stratified
  CEM, "SP-CEM") — LANDED AS MAIN-PATH DEFAULT 2026-05-17. The substrate itself
  was already implemented (HippocampalModule._support_preserving_elite_indices,
  hippocampal/module.py); this change flips its config defaults so the main agent
  action path runs SP-CEM instead of the legacy collapsing CEM.
  Config (6 defaults flipped, in BOTH the HippocampalConfig dataclass AND the
  REEConfig.from_dims() signature, because from_dims assigns its params onto
  config.hippocampal.* unconditionally — flipping only the dataclass would be
  silently reverted for from_dims-built agents):
    use_support_preserving_cem            False -> True
    support_preserving_stratified_elites  False -> True
    support_preserving_ao_std_floor       0.0   -> 0.2
  (support_preserving_min_first_action_classes stays 2; normalize_score_bias_to_e3_range
  untouched — it was NOT in the validated combination.)
  This is an INTENTIONAL non-no-op default change (the one deliberate departure
  from the implement-substrate no-op-default rule): the legacy collapsing CEM
  produced the monostrategy that left SD-029, ARC-062 Rung 2, goal_pipeline
  GAP-2/4 and self_attribution GAP-1/2/3 non_contributory. Every experiment that
  builds config WITHOUT explicitly setting these flags now gets SP-CEM.
  Bit-identical legacy opt-out: explicitly pass use_support_preserving_cem=False,
  support_preserving_stratified_elites=False, support_preserving_ao_std_floor=0.0.
  EXQ-567 ARM_0 and all existing ablation/control arms already pin these
  explicitly, so they are unaffected; only default-reliant scripts change (the
  intended effect).
  Evidence basis: V3-EXQ-567 PASS 2026-05-15 (ARM_1 vs ARM_0:
  selected_action_entropy 0.0124 -> 0.4965, candidate support 1.007 -> 2.810;
  claim ARC-065). Corroborated by V3-EXQ-573 + V3-EXQ-568. Confound-free per the
  V3-EXQ-543e failure autopsy (2026-05-17: H1/H2 FALSE, H3 confirmed; SP-CEM
  delivers ~4.95 unique first-action classes). Phased training: N/A (selection/
  sampling change, no encoder/learning). MECH-094: N/A (no simulation/replay/
  memory write). Smoke: /tmp/smoke_spcem_mainpath.py 4/4 (bare from_dims agent
  runs SP-CEM via StepHarness, gate on every candidate step, no crash, opt-out
  intact); experiments/v3_exq_567_..._sp_cem.py --dry-run still PASS (both
  explicit-off ARM_0 and explicit-on ARM_1 end-to-end). Validation experiment:
  V3-EXQ-583 (experiments/v3_exq_583_spcem_mainpath_default_wiring.py,
  experiment_type v3_exq_583_spcem_mainpath_default_wiring) 3-arm default-wiring
  equivalence queued, DIAGNOSTIC claim_ids=[] (ARM_default == ARM_explicit_on
  within 1e-9 + exact dict match, both >> ARM_explicit_off; dry-run PASS both
  criteria). claims.yaml: ARC-065 carries an
  implementation_note; NOT promoted (promotion is Rung-1 matched-entropy governance,
  gated on V3-EXQ-569). The ARC-062 GatedPolicy head-input one-hot augmentation is
  a SEPARATE follow-on (V3-EXQ-543f), out of scope here. See ARC-065, ARC-062,
  SD-029, MECH-269, MECH-309, behavioral_diversity_acceptance_criteria.md.

## ARC-062 Phase 1 gated-policy (GAP-A, 2026-05-09)
- ARC-062: policy.gated_policy -- IMPLEMENTED 2026-05-09 (arc_062_rule_apprehension
  plan GAP-A). Module: ree_core/policy/gated_policy.py (GatedPolicy +
  GatedPolicyConfig). Two scoring heads (symmetry-broken init) plus 3-stream
  context discriminator (z_world, z_self, z_harm_a) -> sigmoid w in [0, 1];
  gated_score_bias = w*head_0 + (1-w)*head_1 (plus mode_separation_floor when
  enabled). Config: REEConfig.use_gated_policy (default False, bit-identical OFF);
  gated_policy_disc_hidden=24, gated_policy_n_heads=2, gated_policy_disc_init_scale=0.1.
  Data flow: candidate_features [K, world_dim] -> heads -> composed bias ->
  REEAgent.select_action() score_bias (parallel to lateral_pfc / dacc / ofc).
  Phase 1 does NOT wire SD-033a (that is GAP-C/D, commitment_closure GAP-1).
  Contracts: tests/contracts/test_gated_policy.py C1-C5 (7/7 PASS).
  MECH-094: simulation_mode returns neutral (0.5, zeros); no internal state buffer.
  Phased training: required for Phase 2/3 trained-policy falsifiers (P0/P1/P2).
  Validation experiment: V3-EXQ-542a substrate-readiness PASS (UC1-UC6 incl one-hot
  UC6; supersedes V3-EXQ-542). See MECH-309, SD-054, rule_apprehension_layer.md,
  evidence/planning/arc_062_rule_apprehension_plan.md.

## ARC-062 GatedPolicy differential-heads robustness fix (2026-05-18)
- ARC-062: policy.gated_policy two-head reparameterization -- IMPLEMENTED
  2026-05-18. ree_core/policy/gated_policy.py. Motivated by the V3-EXQ-543h
  failure autopsy + cross-machine 543g replication (REE_assembly/evidence/
  planning/failure_autopsy_V3-EXQ-543h_2026-05-18.{md,json}): the same 543g
  config landed gating-ACTIVE on host-A but INERT (n_inert_gating_seeds=3,
  TV<0.05) on cloud-3 AND cloud-4 -- head_0==head_1 collapse is the common
  cross-machine attractor; differentiation was a rare lucky-basin escape.
  Root cause: under outcome-coupled REINFORCE the inert state (head_0==head_1,
  w irrelevant) is a flat equilibrium; head_init_bias_offset is softmax-
  invariant (behaviorally invisible) and the removed head_div term was
  satisfiable by softmax-canceling anti-symmetric offsets.
  Fix: when use_differential_heads=True the two heads are SYNTHESIZED as a
  shared trunk plus a candidate-axis-norm-pinned differential:
    base(x), delta(x) MLPs ; delta_hat = differential_bias_scale *
    delta / (||delta||_K + 1e-8) ; head_0 = base + delta_hat ;
    head_1 = base - delta_hat ; composed = base + (2w-1)*delta_hat.
  Why it makes collapse a non-equilibrium: (route 1) delta==0 is structurally
  unreachable -- delta_hat depends ONLY on delta's DIRECTION (scale-invariant
  normalization), so the loss gradient w.r.t. delta's magnitude is exactly
  zero; gradient descent never drives ||delta||->0 and the nonzero delta
  last-bias init keeps it off zero from step 0. (route 2) at w=0.5,
  d(gated)/dw = head_0-head_1 = 2*delta_hat != 0 by the norm pin, so REINFORCE
  gets a non-vanishing gradient to move w off 0.5 whenever the two pinned
  modes differ in return -- which the pin guarantees. Not the removed head_div
  term: delta_hat is a unit-norm direction over candidates added to a shared
  base, so a nonzero differential necessarily changes the candidate ranking at
  the gating extremes and cannot be softmax-canceled.
  Config: GatedPolicyConfig.use_differential_heads (default False -> two
  independent heads, bit-identical pre-fix path) and .differential_bias_scale
  (default 0.1, mirrors bias_scale; only read on the True path).
  Wiring path (how experiments toggle it): agent.py builds GatedPolicyConfig
  with use_differential_heads=getattr(config,"gated_policy_use_differential_
  heads",False) and differential_bias_scale=getattr(config,"gated_policy_
  differential_bias_scale",0.1) -- same getattr style as
  gated_policy_use_first_action_onehot / crystallize_at_phase3, bit-identical
  when the flat REEConfig attr is absent. An experiment sets
  config.gated_policy_use_differential_heads=True AFTER REEConfig.from_dims()
  and BEFORE REEAgent(config) (single clean build; no rebuild / RNG offset).
  crystallize() freezes (base,delta,discriminator) instead of
  (head_0,head_1,discriminator) when the flag is on -- MECH-334 write-protect
  semantics identical in both configs; expansion device/dtype follows base[0].
  When the flag is on, self.head_0/self.head_1 are None (no external consumer
  touches the modules; downstream reads GatedPolicyOutput.head_*_bias, which
  are the synthesized base +/- delta_hat). get_state() reports
  use_differential_heads for the 543i manifest.
  Backward compatible: default False; every existing experiment runs unchanged
  (verified: v3_exq_543h --dry-run flag-off path). Activation smoke: delta_hat
  L2-over-K == differential_bias_scale exactly; crystallize freezes base+delta
  +disc; heads structurally differ.
  Phased training: N/A (architecture-only; no new encoder/learning signal --
  the P1 loss is deliberately UNCHANGED so V3-EXQ-543i is a clean single-
  variable test of structure-vs-MECH-309). MECH-094: N/A (no simulation/
  replay/memory write; forward simulation_mode path unchanged).
  Validation experiment: V3-EXQ-543i (supersedes V3-EXQ-543g + V3-EXQ-543h),
  queued via /queue-experiment -- same 2x2(x2) design + identical P1 loss,
  only new factor use_differential_heads; acceptance n_inert_gating_seeds==0
  across all seeds AND >=2 machines AND C2/C3 context-discrimination pass.
  Decisive either way: escape -> ARC-062 weak reading viable (MECH-309 holds
  only for unstructured parametric policies); still collapses -> MECH-309
  strong confirmation -> ARC-063/V4 distributed CandidateRule field.
  claims.yaml: ARC-062 + MECH-333 carry an implementation_note only; NO flag/
  confidence/promotion change (governance gated on V3-EXQ-543i). The MECH-309-
  support reading of the 543 cluster is a parked governance follow-on
  (workstream A; not applied this session). See ARC-062, ARC-063, MECH-309,
  MECH-333, MECH-334, INV-074, SD-054, rule_apprehension_layer.md.

## ARC-062 GAP-B mode-separation floor (2026-05-20)
- Follow-on to differential-heads (V3-EXQ-543i autopsy): at discriminator w~0.5
  the composed gated bias is base only -- delta_hat cancels in
  base + (2w-1)*delta_hat, so REINFORCE cannot train differentiation.
- Fix: GatedPolicyConfig.mode_separation_floor (default 0.0, bit-identical OFF).
  Composed bias becomes w*h0 + (1-w)*h1 + floor*(h0-h1). With differential
  heads this injects a non-cancelable mode contrast even when w=0.5.
- Optional P1 aux: p1_w_deviation_aux_weight penalizes w near 0.5 during
  outcome-coupled training (gated_policy.p1_training_auxiliary_loss).
- REEConfig: gated_policy_mode_separation_floor,
  gated_policy_p1_w_deviation_aux_weight (default 0).
- Validation: V3-EXQ-543k (supersedes 543i; same 12-arm design + floor/aux on
  gated arms; K_IDENTICAL_RUNS=3 basin-stability gate; manifest hostname).

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
  MECH-204 GAP-1 consumer (2026-05-08): SleepLoopManager WRITEBACK (phase_manager.py)
  calls SerotoninModule.compute_recalibration_target() and
  E3TrajectorySelector.recalibrate_precision_to(target, step). Config:
  REEConfig.use_rem_precision_recalibration (default False);
  rem_precision_recalibration_step default 0.25 (post V3-EXQ-541c PASS 2026-05-09).
  NOT bundled in use_sleep_aggregation_cluster (separate GAP-1 flag).
  Contracts: tests/contracts/test_mech204_precision_recalibration.py 13/13 PASS.
  Canonical validation: V3-EXQ-541c. Integration closure: V3-EXQ-602 queued.

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

> **Phase 3 (live as of 2026-05-29):** under Phase 3, the **runner still writes** `runner_heartbeats/<hostname>.json` and the `commands` file locally on each tick, but `push_heartbeat()` + `push_commands()` are **no-ops** when `PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH=1` (set in every worker's `shadow.conf`). The runner also `POST`s the same payload to coordinator `/heartbeat` and `/status`, which persist to the `heartbeats` table. The hub's `sync_daemon.phase3_heartbeat_writer` materialises the canonical `runner_heartbeats/<hostname>.json` + `runner_status/<hostname>.json` files from the DB and commits them under `phase3-heartbeats:`. Net effect: the file shape on `REE_assembly/origin/master` is unchanged for downstream consumers (explorer, scaler workflow), only the transport changed.
>
> **Heartbeat git mirror is STATE-CHANGE-ONLY as of 2026-06-23 (liveness tick retired).** The hub `ree-sync-daemon` no longer commits the 30-minute forced "liveness tick" (`PHASE3_HEARTBEAT_LIVENESS_INTERVAL=86400` in its drop-in) -- it was the dominant source of `REE_assembly` git-history bloat. The writer still commits every state-change (start / finish / switch / idle / silent / restart), so *which* experiment is running stays current on `master`, but the **intra-run progress fields (`overall_pct`, `episodes_done`, `recent_lines`) are intentionally stale between state-changes** -- a running experiment's % sits frozen on `master` until it changes state. To read LIVE progress use the **`live-status` branch** (<https://github.com/Latent-Fields/REE_assembly/blob/live-status/FLEET_STATUS.md>, force-updated ~every 3 min by `coordinator/deploy/live-status-writer.py` + `ree-live-status.timer`), the coordinator `/shadow/status`, or the explorer `/machines`. **Do NOT re-enable the liveness tick / lower the interval to "fix" the staleness** -- that reintroduces ~40 redundant commits/day into permanent history. The cloud-scaler already reads telemetry coordinator-primary (`/shadow/status`), with the git mirror only as an unreachable-coordinator fallback. See umbrella `CLAUDE.md` "Reading live fleet / experiment progress".
>
> **Hub-runner co-tenancy (ISOLATED-CHECKOUT model, 2026-06-02):** the hub VM runs the coordinator + `sync_daemon` writers out of `~/REE_Working`, and (optionally) a worker runner out of a SEPARATE checkout `~/REE_Working_runner/{ree-v3,REE_assembly}`. The runner's result-manifest + queue + telemetry writes land in its own tree, so they never dirty the writers' `~/REE_Working` (Phase-3 writers refuse to commit on a dirty tree -- the recurring hub-wedge fleet outage; `reference_hub_writer_wedge.md`). Config: a `WorkingDirectory=/home/ree/REE_Working_runner/ree-v3` drop-in in hub `shadow.conf` + the STANDARD worker gates (`PHASE3_RUNNER_TELEMETRY_OFF_GIT=1` + `PHASE3_COMMANDS_VIA_COORDINATOR=1` + the three `_PUSH` gates). Use `--machine ree-cloud-1` in `ExecStart`. **`PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE` is OBSOLETE** -- it was the shared-checkout stopgap and also gated the command-file writeback (the hub's command spin); do NOT set it under isolation. (Legacy shared-checkout note: the old `_HEARTBEAT_WRITE=1` approach only skipped heartbeat/command FILE writes but left result-manifest + queue writes dirtying the shared tree, so it never fully prevented the wedge -- the isolated checkout is the actual fix.) Template: `coordinator/deploy/shadow.conf.hub.example`.
>
> **Worker telemetry-off-git gate (2026-06-02):** `PHASE3_RUNNER_TELEMETRY_OFF_GIT=1` is set on cloud-2/3/4 `shadow.conf` (template: `coordinator/deploy/shadow.conf.worker.example`). It suppresses ONLY the per-tick `runner_heartbeats/<host>.json` + `runner_status/<host>.json` FILE writes (the coordinator POST stays the transport; the hub writer materialises git). This stops the autostash churn / dormant-stash pileup that the per-tick local writes caused on every `git pull --rebase --autostash` (cloud-3 hit 43 stashes 2026-06-02; 191 by 2026-05-31). Unlike `_HEARTBEAT_WRITE` (hub-only -- also gates the command-file writeback, restart-loops a worker, incident 2026-05-30), this gate does **not** touch the command channel, so it is safe on any worker with no restart-loop risk. Skip-completed + peer-dedup keep reading the in-tree `runner_status/` dir, which is pull-populated from the hub materialisation. Default OFF = bit-identical. Gate logic: `runner_remote_control._phase3_telemetry_file_write_gated()`; contracts: `tests/contracts/test_phase3_telemetry_off_git_gate.py`. **The Mac (`DLAPTOP-4.local`) is now ALSO telemetry-off-git (2026-06-03)** -- set in `~/.local/bin/ree_runner_launchd.sh`. The old "Mac intentionally not gated / serve.py reads local files" caveat is RESOLVED: `serve.py read_machines()` already sources every machine's LIVE card from the coordinator `/shadow/status` (`telemetry_mode=coordinator`; git files only enrich gpu/recent-lines/titles), and the hub `sync_daemon` materialises `runner_heartbeats/`+`runner_status/DLAPTOP-4.local.json` on origin from the Mac's coordinator POST payloads -- so the Mac's local writes were redundant AND caused the autostash-conflict friction on Mac-side pushes. With the gate on, the Mac pulls the writer-materialised versions and never overwrites them, keeping its REE_assembly tree clean. (serve.py's `_default_runner_extra_env` -- the dormant Popen-fallback path, not used while launchd supervises -- still omits this flag; align it if that path is ever revived.)
>
> **Command-channel coordinator migration (2026-06-02, code landed; rollout staged):** the remote-control command channel now has a coordinator transport, replacing the git command-file. The coordinator exposes `POST /commands/issue` (insert) + `GET /commands?machine=` (pending = `acked_at IS NULL`) + `POST /commands/ack` (stamp `acked_at` + `result_status`/`result_note`); db helpers `db.insert_command` / `fetch_pending_commands` / `ack_command`; client helpers `coordinator_client.issue_command` / `fetch_commands` / `ack_command`. **serve.py** dual-writes (git file + coordinator `POST /commands/issue`) when `PHASE3_COMMANDS_DUAL_WRITE` is set (env or `coordinator.env`; default OFF so serve.py is bit-identical until the operator opts in *after* a worker is on the coordinator channel -- otherwise commands accumulate unacked and the first via-coordinator worker drains the backlog). **runner_remote_control.process_pending_commands** gained two env gates (both default OFF = git-only, bit-identical):
> - `PHASE3_COMMANDS_VIA_COORDINATOR=1` -- fetch + ack commands via the coordinator IN ADDITION to the git file (dual-read; the git file stays the proven fallback). Every supported command kind is idempotent, so a command delivered to both channels and executed twice is harmless -- the explicit property that lets the fallback run alongside the new path. This is the **per-worker canary knob** (enable on cloud-2 first).
> - `PHASE3_COMMANDS_OFF_GIT=1` -- the coordinator is the SOLE channel; the worker neither reads nor writes the git command-file. Implies VIA_COORDINATOR; **self-guards** -- if the coordinator channel is unavailable (`COORDINATION_MODE != coordinator` or no URL/token) it REFUSES and falls back to the git file so the worker is never left uncontrollable.
>
> `PHASE3_COMMANDS_OFF_GIT` is what removes the restart-loop hazard that made `_HEARTBEAT_WRITE` hub-only: a stop command's ack now persists in the coordinator `commands` table, not in a per-machine git file the worker may be unable to write. Accordingly `_phase3_heartbeat_write_gated()` now PERMITS `_HEARTBEAT_WRITE` on a non-hub worker **when `PHASE3_COMMANDS_OFF_GIT` is active** (it still refuses otherwise). The migration deliberately did NOT fold command-off-git into `PHASE3_RUNNER_TELEMETRY_OFF_GIT`: that flag is already deployed fleet-wide and documented as command-channel-safe; folding would flip all workers at once (defeating the canary) and reintroduce restart-loop risk on a flag advertised as safe. **Rollout order:** (1) deploy coordinator; (2) deploy runner code (no-op, gates OFF); (3) canary cloud-2 `PHASE3_COMMANDS_VIA_COORDINATOR=1`; (4) enable serve.py `PHASE3_COMMANDS_DUAL_WRITE`; verify stop/pause/resume/release_claim round-trip + no restart loop; (5) cloud-2 `PHASE3_COMMANDS_OFF_GIT=1`, re-verify; (6) fleet (cloud-3/4 then Mac). Gate logic: `runner_remote_control._phase3_commands_via_coordinator_gated()` / `_phase3_commands_off_git_gated()`; contracts: `tests/contracts/test_phase3_command_channel_coordinator.py` (runner gates + dual-read) + `coordinator/test_phase3_command_channel.py` (db + HTTP + client). Rollback at any step: unset the worker's flag (reverts to git command-file) or unset serve.py `PHASE3_COMMANDS_DUAL_WRITE`; the coordinator endpoints are additive and harmless when unused.

When started with `--remote-control`, the runner emits a per-machine heartbeat each loop tick to `REE_assembly/evidence/experiments/runner_heartbeats/<hostname>.json` and processes pending commands from `runner_commands/<hostname>.json`. Default-off; bit-identical when omitted. Helper module: `runner_remote_control.py` (sibling of `experiment_runner.py`).

**Active-claim protection (added 2026-05-01, broadened 2026-05-08 and 2026-06-14; superseded under Phase 3)**: `push_heartbeat()` and `push_commands()` historically started each tick with `git pull --rebase --autostash` against REE_assembly. To prevent the autostash cycle silently reverting concurrent Claude sessions' uncommitted edits, both functions bailed at the top via `_active_claim_on_evidence_dir()` whenever any TASK_CLAIMS.json entry tagged `"active"` lists a resource path containing `evidence/` (originally scoped to `evidence/experiments/`; broadened to the `evidence/` prefix on 2026-05-08 so it also covers `evidence/planning/`, `evidence/literature/`, and any future evidence sibling) **or `docs/claims/`** (added 2026-06-14 -- claims.yaml is the most-contended governance file and lives outside the `evidence/` prefix). Heartbeats/commands are best-effort; the next tick after the claim closes resumes pushing. Three real-world incidents motivated and reinforced this guard: (1) 5 EXQ-232 ARC-026 supersession edits to `evidence/experiments/` made 2026-04-29 silently reverted to original-commit content by 2026-05-01 with no trace in git history (no stash, no orphan commit); (2) an `evidence/planning/substrate_queue.json` MECH-204 design_doc edit made 2026-05-08 silently reverted with the same signature; (3) a 2026-06-14 IGW window where an autostash cycle transiently swept a session's (ABM-1/Q-060) uncommitted `docs/claims/claims.yaml` edits out of the working tree (briefly showing it clean) and restored them a tick later -- no data lost that time, but identical in shape to (1)/(2). In all cases the autostash mechanism in the per-minute heartbeat was the culprit. Contract test: `tests/contracts/test_active_claim_evidence_guard.py` (C9 covers the `docs/claims/` clause). **Under Phase 3 the guard is moot for these two functions** (the PHASE3 gate skips the entire push path before pull-rebase runs), but it still protects any future runner-side path that calls `_push_telemetry_file`.

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

**Manifest-leak in conflict-recovery (fixed 2026-05-09)**:
`experiment_runner._git_push_with_retry` previously lost the manifest when a per-experiment
results push hit a non-fast-forward followed by a `pull --rebase` failure. Root cause: the
recovery path stashed only the WORKING TREE, then `git reset --hard origin/<branch>` destroyed
the manifest-bearing local commit; the followup `git add <manifest_path>` was a silent no-op
because the file no longer existed on disk, and `subprocess.run(..., capture_output=True)`
swallowed the `did not match any files` stderr. The recovery commit captured stashed
sentinel/heartbeat/status only, and the manifest never reached REE_assembly master. Five real
runs lost their manifest this way (V3-EXQ-433f, 537, 537c, 538, 541; V3-EXQ-541 reproduced
with reflog evidence on ree-cloud-1, 2026-05-08T23:43Z). Fix: capture pre-reset HEAD SHA,
restore each `result_files` path via `git checkout <pre_reset_sha> -- <rel>` after the
reset+pop, resolve any stash-pop unmerged paths via `--ours` (taking the post-reset remote
version), and emit a WARN if the post-recovery selective add stages none of the expected
files. Also closes a sibling bug: `git_push_results` was not passing `result_files` through
to `_git_push_with_retry`, so the recovery branch always saw `result_files=None` and fell into
the broad-add fallback. Contract test: `tests/contracts/test_runner_manifest_survives_conflict_recovery.py`
(C1 single manifest, C2 multi manifest, C3 broad-add fallback).

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

## SD-031: E2_world Single-Pass Comparator (z_world agency) (2026-06-06)
- SD-031: self_attribution.comparator_z_world -- IMPLEMENTED 2026-06-06
  (substrate; v3_pending until the discriminative/attribution arm PASSes a
  gated validation experiment). The z_world instantiation of the MECH-256
  single-pass forward-model comparator (sibling to ARC-033 on z_harm_s; the
  z_self instantiation SD-030 stays V4). Rescoped v4 -> v3 the same day:
  SD-031 is a named dependency of the V3 z_world discriminative-granularity
  retest (failure_autopsy_zworld-integration-cluster_2026-06-06, V3-EXQ-177's
  attribution arm), so it is V3-scoped by definition (implementation_phase is
  a prediction, not a permission gate).
  Module: ree_core/predictors/e2_world.py (E2WorldForward + E2WorldConfig +
  MIN_DISCRIMINATIVE_WORLD_DIM=128). Reuses ResidualHarmForward (the generic
  residual-delta module in stack.py -- not harm-specific). Single forward pass,
  NO counterfactual:
    predicted_z_world = E2WorldForward(z_world_prev, a_actual)
    residual_world    = z_world_observed - predicted_z_world   # agency signal
  forward() is the unrestricted evaluator/rollout read (MECH-257);
  comparator_residual() is the MECH-094-gated retrospective comparator read
  (returns zeros under simulation_mode).
  Config: LatentStackConfig.use_e2_world_forward (bool, default False;
  bit-identical OFF) wired through REEConfig.from_dims. z_world_dim is read
  from config.latent.world_dim at agent construction -- NEVER a literal.
  Agent wiring (agent.py): self.e2_world built when use_e2_world_forward is on,
  mirroring the e2_harm_s block (agent-level module, no new LatentState field).
  CARRY-FORWARD GUARD (the load-bearing piece): E2WorldForward.__init__
  HARD-ASSERTS world_dim >= 128 (REEConfig.large). The 2026-06-06 cluster
  autopsy established z_world at world_dim=32 is a competent BULK predictor
  (world_forward_r2 0.72-0.94) but lacks discriminative granularity -> a dim=32
  comparator emits a vacuous zero attribution gap. The agent constructor
  therefore RAISES if use_e2_world_forward=True at the default world_dim (32).
  An explicit allow_subthreshold_dim=True escape hatch (default False) permits
  bulk-only/ablation construction below threshold but reports
  attribution_ready=False and returns a zeroed sentinel residual (never a
  misleadingly-zero gap).
  Phased training (validation experiment, not the substrate): P0 train the
  z_world encoder (SD-009 event-contrastive + SD-018 resource proximity) so
  z_world is discriminative BEFORE the forward model -- a random encoder yields
  a near-position-invariant z_world and a trivially-identity world-forward (the
  MECH-353 / V3-EXQ-642 vacuous-comparator lesson; dimensionality is necessary,
  not sufficient). P1 train E2WorldForward on FROZEN z_world (z_world_next.detach()).
  P2 attribution eval at world_dim >= 128 with ARC-065 diversity active.
  SD-013 analogue: compute_interventional_loss (margin loss pushing predictions
  apart across actions) for the strong ambient world-state correlations.
  Backward compatible: use_e2_world_forward=False by default -> agent.e2_world
  is None; bit-identical OFF (default == explicit-False action stream verified).
  845/845 contracts + 7/7 preflight PASS; 9/9 new contracts in
  tests/contracts/test_e2_world_forward.py (C1 bit-identical OFF / C2 shape +
  action-response / C3 dim-guard raises at 32 + None + subthreshold-opt-in /
  C4 delta-not-identity on autocorrelated z / C5 MECH-094 sim gate / C6
  agent-path guard raises at default dim, constructs at 128).
  Activation smoke 2026-06-06 (world_dim=128): world_forward_r2 0.969;
  single-pass comparator self-caused residual ~2.0 vs externally-caused ~22.6
  (correct attribution gap).
  MECH-094: forward() unrestricted (evaluator mode); comparator_residual()
  returns zeros under simulation_mode (replay/DMN cannot generate a spurious
  agency signal). Phased training: yes for the validation experiment (P0/P1/P2);
  the substrate landing itself adds no trained parameters beyond the forward MLP.
  Validation experiment: NOT queued (out of scope this pass). GATED -- do not
  run until world_dim >= 128 AND ARC-065 balanced-event diversity are both in
  place (ARC-065 currently phase_1 only); else it reproduces the dim=32 +
  monostrategy confound. Hand to /queue-experiment.
  Design doc: REE_assembly/docs/architecture/sd_031_e2_world_forward_model.md
  See SD-031, MECH-256 (parent comparator mechanism), ARC-033 / e2_harm_s
  (the z_harm_s sibling template), SD-029 (z_harm_s instantiation), SD-030
  (z_self instantiation; stays V4), SD-005 (z_world split; dep, implemented),
  SD-010 (harm-stream separation; dep), ARC-065 (behavioral diversity; the
  co-gate on validation), MECH-353 / V3-EXQ-642 (vacuous-untrained-encoder
  lesson), MECH-094 (call-site scoping).

## SD-063: E2 Conditional Predictive-Uncertainty Head (z_world quantile) (2026-07-05)
- SD-063: e2.conditional_predictive_uncertainty_head -- IMPLEMENTED 2026-07-05
  (substrate; v3_pending until a validation experiment shows the head's per-point
  predictive variance improves E3 commitment gating over the running-variance EMA
  AND the SD-031 agency residual is preserved under joint training. PROMOTES
  NOTHING). The concrete realization of the MECH-059 confidence channel: a
  distribution-free quantile/pinball head over (z_world, action) that emits a
  per-input predictive spread tracking realized error, feeding E3's commit gate
  in place of the state-blind, global running-variance EMA. Winner of the
  V3-EXQ-712 diagnostic (quantile CRPS 0.00486 vs point 0.00514;
  precision_error_corr 0.379 vs the EMA null 0.0; Gaussian-family heads -- hetero,
  mixture -- did WORSE than the point baseline, so the distribution-free form is
  load-bearing).
  Module: ree_core/predictors/e2_world_uncertainty.py (E2WorldUncertaintyHead +
  E2WorldUncertaintyConfig; QUANTILE_LEVELS = 9 levels 0.1..0.9, the 712 winner;
  IQR_TO_STD_10_90 = 2.5631). Trunk = 2-layer MLP(ReLU) -> Linear(D*Q) -> [B,D,Q]
  (matches the 712 QuantileHead). compute_loss = pinball; predictive_variance /
  predictive_std = monotone-rearranged (torch.sort, anti-crossing) [q0.1,q0.9] IQR
  -> Gaussian-reference variance, meaned over dims, per batch item, under no_grad.
  Data flow: z_world_t.detach() + a_onehot -> E2WorldUncertaintyHead -> pinball
  (P1) / predictive_variance (read) -> E3.select(conditional_predictive_variance=)
  commit gate.
  Config: LatentStackConfig.use_e2_world_uncertainty (bool, default False;
  bit-identical OFF) + e2_world_uncertainty_hidden_dim (128) +
  e2_world_uncertainty_lr (1e-3); E3Config.use_conditional_precision_gate (bool,
  default False). Both surfaced by REEConfig.from_dims. Like use_e2_world_forward,
  the flag signals intent; the head is instantiated at the experiment/agent level
  -- LatentStack.encode() is UNTOUCHED, so no new LatentState field and OFF is
  byte-identical. z_world_dim is REQUIRED at construction (no literal default);
  unlike E2WorldForward there is NO world_dim>=128 assert (this is a predictive-
  spread readout, not the SD-031 discriminative comparator; the 712 diagnostic ran
  at world_dim=32).
  E3 consumer (e3_selector.py select()): new kwarg conditional_predictive_variance
  (default None). When E3Config.use_conditional_precision_gate is True AND a value
  is supplied, the ARC-016 commit decision compares that per-input variance against
  effective_threshold INSTEAD of self._running_variance; None or flag-off -> EMA
  fallback (byte-identical). Does not touch the use_harm_variance_commit path.
  SD-031 AGENCY-RESIDUAL GUARD (the load-bearing caveat): the head is a SEPARATE
  nn.Module sharing NO parameters with E2WorldForward or the encoder, and its P1
  loss reads DETACHED z_world inputs AND a DETACHED z_world_next target. Its
  gradients therefore never reach the forward model that produces the SD-031
  agency residual -> it cannot explain the residual away by construction. The
  validation experiment must still confirm this empirically under joint training.
  Phased training (validation, not the substrate): P0 z_world encoder warmup
  (SD-009 + SD-018); P1 head on frozen z_world (detach inputs + target); P2 eval
  CRPS + precision_error_corr + agency-residual preservation.
  MECH-094: DOES NOT APPLY -- waking online forward-model read for commitment
  gating; no memory write, no simulation/replay.
  Backward compatible: both switches False by default; agent hot path unchanged;
  1381/1385 suite PASS (the 4 fails are pre-existing, unrelated: E1 SD-016 proj-dim,
  control-vector bit-identical, 2x runner fail-branch -- all fail on the clean base
  tree). 15/15 new contracts in tests/contracts/test_sd063_conditional_uncertainty_head.py
  (config no-op + from_dims surface / head shapes + dim-required + level validation /
  pinball-trains + heteroscedastic conditional variance / SD-031 param-disjoint +
  detach-blocks-encoder-grad / E3 gate OFF-ignores + ON-overrides-both-directions +
  ON-no-value-EMA-fallback).
  Validation experiment: V3-EXQ-716 queued (see below) -- diagnostic falsifier,
  PROMOTES NOTHING.
  Design doc: REE_assembly/docs/architecture/sd_063_e2_conditional_uncertainty_head.md
  See MECH-059 (confidence channel; instantiated), SD-031 / E2WorldForward (the
  agency residual this must not disturb; dep), MECH-256 (comparator family),
  V3-EXQ-712 (motivating diagnostic), ARC-016 (dynamic-precision commit gate this
  feeds).

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
    cue_action_proj for behavioural effects unless use_differentiable_cem=True (SD-055
    substrate-ready 2026-05-15; V3-EXQ-568 PASS grad_max=372 -- gradient barrier only,
    not behavioural divergence). (cue_terrain_proj path remains valid --
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

- SD-MEL-CONSUMER: sleep.adaptive_mel_sleep_cadence -- IMPLEMENTED 2026-07-07.
  Adaptive sleep-cadence MEL consumer (sleep_substrate:GAP-5b). The INV-050 THIRD /
  learning-demand sleep drive: reads accumulated waking Model Error Load (MEL = mean
  per-step e3 prediction error over the wake window, the same signal V3-EXQ-701c
  demonstrated is measurable + monotone in graded novelty) and modulates the offline
  (sleep) phase, REPLACING the K-episode-deterministic scheduler.
  Module: ree_core/sleep/mel_consumer.py (MELConsumer, MELConsumerConfig,
  WakingMELAccumulator).
  Config (REEConfig, ree_core/utils/config.py): use_mel_consumer (bool, default False;
  set True to enable). Sub-knobs: mel_gain (1.0), mel_reference (0.0 = auto to first
  cycle; validation sets ~2e-5), mel_reference_mode ("fixed"|"ema"), mel_ema_alpha (0.1),
  mel_duration_factor_min/max (0.5/3.0), mel_relative_floor (1e-6 -- recalibrated DOWN
  from the 701c-inherited ABS_MEL_FLOOR=1e-4, which was ~5x the converged-base signal),
  mel_scale_sws/mel_scale_rem (True), use_mel_entry (False), mel_entry_threshold (0.0).
  All wired through REEConfig.from_dims().
  Data flow: waking step -> agent.update_residue() -> e3.post_action_update()
  -> e3_prediction_error -> MELConsumer.note_step_pe() (waking-only, hypothesis_tag=False)
  -> [episode end via SleepLoopManager] duration_factor = clamp(1 + mel_gain*(mel/ref - 1),
  min, max) -> scales sws_consolidation_steps / rem_attribution_steps for the cycle
  -> more/fewer sws_n_writes + rem_n_rollouts (the exact V3-EXQ-677 pinned DV, now
  MEL-driven). Secondary entry lever: use_mel_entry fires a cycle when accumulated MEL
  crosses mel_entry_threshold (K-episode counter as safety backstop).
  IMPORTANT driver note: the consumer engages ONLY through the SleepLoopManager path
  (notify_episode_end / force_cycle) and MEL accumulates ONLY when the driver calls
  agent.update_residue() each waking step. A driver that calls agent.run_sleep_cycle()
  DIRECTLY (e.g. V3-EXQ-677) bypasses the consumer -- validation drivers must use the
  manager path + update_residue.
  Backward compatible: use_mel_consumer default False -> agent.mel_consumer is None,
  update_residue skips accumulation, SleepLoopManager reads unmodified config
  -> byte-identical K-episode-deterministic scheduler. Full contracts + preflight PASS.
  No trainable parameters / no new encoder head / no new latent field. No phased training
  needed (validation still needs a converged base so PE is at converged scale, per 701c).
  MECH-094: reads waking PE only; writes nothing to memory during non-waking states.
  DISTINCT from the SD-037 arousal entry gate (use_mech286_sleep_onset_gate /
  sleep_onset_gate.py / sleep_substrate:GAP-5, V4) -- orthogonal signal (arousal vs MEL);
  they compose.
  Unblocks: INV-050 (retest), MECH-180 (v3_pending). Validation experiment: V3-EXQ-718
  queued 2026-07-07 (v3_exq_718_sdmelconsumer_adaptive_cadence_validation; diagnostic;
  recon-only converged base + 4 graded-novelty arms consumer-ON + 1 matched-novelty
  consumer-OFF control; DV = cumulative_sws_writes + cumulative_rem_rollouts +
  mel_duration_factor; PROMOTES NOTHING until it scores).
  See REE_assembly/docs/architecture/sd_mel_consumer.md; SD-017; INV-050; MECH-180;
  plan-of-record REE_assembly/evidence/planning/sleep_substrate_plan.md (GAP-5b).

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

## mode-governance-engagement: external_task salience source for SalienceCoordinator (2026-06-13)
- mode-governance-engagement -- IMPLEMENTED 2026-06-13 (substrate; MECH-266 stays
  provisional / SD-032a stays stable -- PROMOTES NOTHING until the 464d/467d retest
  runs). The external_task salience SOURCE the SD-032a SalienceCoordinator lacked on the
  603n foraging substrate. Routed by the confirmed
  failure_autopsy_SD-034-closure-cluster-ext_2026-06-12 (sub-cluster B: V3-EXQ-464c +
  467c) via the substrate_queue mode-governance-engagement entry minted by the 2026-06-13
  AM governance cycle.
  ROOT CAUSE (code-confirmed in salience_coordinator.py): external_task gets only
  external_task_bias (1.0) + drive_level (affinity weight 1.0), while dacc_pe /
  dacc_foraging / dacc_difficulty all push internal_planning. On the foraging substrate
  drive_level ~ 0.016 (540c probe), so on tick 1 the argmax flips to internal_planning and
  the agent settles there for the episode -> fraction_in_external_task = 0.0 on both arms /
  all seeds, and the 464c/467c eval loops count one episode-initial settle per episode
  (n_switches == n_episodes), so MECH-266's exit-rail had no contested mode to bind and the
  n_switches>=1 non-vacuity gate passed VACUOUSLY.
  THE FIX (no-op-default; bit-identical OFF; mirrors the SD-035 CeA / SD-037 override
  signal-injection pattern exactly -- the SalienceCoordinator class is UNCHANGED, it
  already accepts arbitrary named signals):
    Module: ree_core/agent.py (registration at __init__ + injection at the salience tick
      site in select_action), ree_core/utils/config.py (6 no-op-default flags + from_dims).
    Registration (REEAgent.__init__, gated on use_external_task_drive + salience present):
      affinity_weights["external_task_drive"] = {"external_task": external_task_drive_affinity_weight}
      salience_weights["external_task_drive"] = external_task_drive_salience_weight
      -- registered in BOTH so external_task can win the mode argmax (affinity) AND a switch
      INTO external_task can fire the MECH-259 trigger (salience aggregate).
    Injection (select_action, BEFORE coord.tick(), alongside the aic/cea/override injections):
      engagement = goal_active ? clip(commit_w*float(beta_gate.is_elevated)
                                      + prox_w*float(goal_state.goal_proximity(z_world)), 0, 1) : 0
      coord.update_signal("external_task_drive", engagement)
    The engagement is DYNAMIC by design (gated on an active goal, graded by committed
    pursuit x proximity), so it RELEASES toward internal_planning during deliberation /
    between-goals / just-consumed -- producing GENUINE mode competition, NOT the 464b
    "100% external_task, 0 switches" saturation degeneracy (the opposite failure the
    2026-06-04 MECH-266 evidence_quality_note recorded).
  Config (REEConfig + from_dims, all no-op default -> bit-identical OFF):
    use_external_task_drive (False, master), external_task_drive_affinity_weight (1.0),
    external_task_drive_salience_weight (1.0), external_task_drive_commit_weight (1.0),
    external_task_drive_proximity_weight (1.0), external_task_drive_require_goal_active (True).
  Backward compatible: use_external_task_drive=False by default -> no slot registered, no
    injection, "external_task_drive" never enters the coordinator's _input_signals (tick
    reads 0) -> bit-identical. 7/7 preflight + 1031 contracts (1026 prior + 5 new in
    tests/contracts/test_mech266_external_task_drive.py: C1 OFF no-slot + bit-identical
    action stream / C2 ON registers BOTH affinity+salience slots / C3 coordinator math --
    drive raises external_task probability AND salience_aggregate over an
    internal_planning-pushed baseline / C4 agent injects engagement>0 on a goal-active
    agent + monotone (drive never reduces external_task occupancy) / C5 goal-inactive ->
    injected engagement 0, the release path) PASS. v3_exq_464c --dry-run unchanged
    (drive OFF -> reproduces the prior sym_frac=0.0 / asym_frac=0.0 substrate-ceiling
    signature).
  Phased training: N/A (non-trainable arithmetic signal injection; no learned parameters).
    MECH-094: waking-only by call-site scoping (select_action), as with the neighbouring
    AIC / CeA / override injections. Evidence-staleness (Step 8.5): NOT triggered --
    no-op-default flag; every existing experiment uses the default (drive off), so no
    dependent claim's measured mechanism changed. KEEP all evidence.
  depends_on (unresolved at landing): scaffolded_sd054_onboarding nav-competence (Stage-H).
    The substrate + contract tests land regardless (user-directed); the VALIDATION may
    self-route substrate_not_ready if the agent does not survive/forage long enough, which
    the 603n contact guard + the restated occupancy gate handle cleanly.
  Validation experiments: V3-EXQ-464d (competing-goals) + V3-EXQ-467d (mode-stickiness
    dose-response) -- successors (NEW letter, NOT supersede) of 464c/467c with
    use_external_task_drive=True AND the readiness gate RE-STATED as
    min_across_arms(fraction_in_external_task) > floor (~0.1) replacing the n_switches>=1
    non-vacuity gate, so the asymmetric exit-rail finally has a contested mode to bind.
    claim_ids=[MECH-266, SD-032a]; experiment_purpose=evidence. Queued via /queue-experiment.
    GOVERNANCE: MECH-266 stays provisional / SD-032a stays stable; claims.yaml carries only
    an implementation_note + the pending_retest_after_substrate flag added this cycle (no
    flag/confidence/promotion change). substrate_queue mode-governance-engagement ready
    STAYS false until the retest clears the occupancy gate.
  Design doc: REE_assembly/docs/architecture/sd_032_cingulate_integration_substrate.md
    (mode-governance-engagement section). Substrate_queue:
    REE_assembly/evidence/planning/substrate_queue.json (mode-governance-engagement).
    Autopsy: REE_assembly/evidence/planning/failure_autopsy_SD-034-closure-cluster-ext_2026-06-12.{md,json}.
  See MECH-266 (asymmetric mode hysteresis -- the exit-rail this unblocks), SD-032a
    (SalienceCoordinator -- the mode register the drive feeds), MECH-259 (switch threshold),
    SD-035 CeA / SD-037 override (the affinity+salience injection pattern this mirrors),
    SD-012 drive_level (the inadequate external_task driver this complements), MECH-295
    (goal-pursuit / approach bridge -- adjacent goal machinery), V3-EXQ-464c/467c (the FAILs
    this addresses), V3-EXQ-464d/467d (validation), MECH-094 (call-site scoping).

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

## SD-048: Interoceptive Noise Dynamics (2026-05-03)
- SD-048: body.interoceptive_noise_dynamics -- IMPLEMENTED 2026-05-03.
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorldV2).
  Three concurrent agent-independent stochastic body-state noise sources at
  distinct temporal scales applied to harm_obs_a readout, layered onto the
  existing SD-022 limb-damage / SD-011 EMA substrate. Level 2 counterpart
  to SD-047 at the body-state layer; substrate-ceiling unblock for ARC-058
  (and ARC-033 arbitration) per the V3-EXQ-506-equivalent C4-only-PASS
  signature on z_harm_a comparator testing.
  Three sources:
    Source 1 (autonomic noise): per-element i.i.d. Gaussian additive noise
      on harm_obs_a at every tick. Fast, continuous, low-amplitude
      (HRV / sympathetic-fluctuation analog). default sigma=0.02 normalised
      harm units.
    Source 2 (sensitisation spikes): Poisson onset of transient
      multiplicative amplification on harm_obs_a, exponentially decaying.
      Inflammatory sensitisation analog. default rate=0.008/step,
      magnitude=1.8x (0.8x amplification contribution), halflife=15 steps.
      Cumulative amplification capped at 5.0x to bound long Poisson tails.
    Source 3 (fatigue drift): slow AR(1) latent fatigue state additively
      contributing to harm_obs_a across the episode. Allostatic-load /
      sleep-pressure analog. default ar_coeff=0.995 (very slow decay),
      noise_scale=0.005, contribution_weight=0.15. Resets per episode.
  Order: fatigue (additive baseline shift) -> sensitisation (multiplicative
  gain modulation) -> autonomic (read-out noise floor). Fatigue and
  sensitisation modulate baseline / gain; autonomic noise is true read-out
  noise on top.
  Config (CausalGridWorldV2 __init__ kwargs, env-only -- not surfaced through
  REEConfig.from_dims, matching SD-047 / SD-022 / SD-029 precedent for
  env-only SDs):
    interoceptive_noise_enabled (bool, default False) -- master switch.
    interoceptive_noise_scale (float, default 1.0) -- 4-arm sweep lever
      (OFF / 0.25 / 1.0 / 4.0); scales autonomic_noise_scale,
      sensitisation_rate, and fatigue_noise_scale uniformly.
    autonomic_noise_enabled (bool, default True) -- per-source switch.
    autonomic_noise_scale (float, default 0.02) -- per-step Gaussian SD.
    sensitisation_enabled (bool, default True) -- per-source switch.
    sensitisation_rate (float, default 0.008) -- Poisson onset/step.
    sensitisation_magnitude (float, default 1.8) -- multiplicative amplifier.
    sensitisation_halflife (int, default 15) -- exponential decay half-life.
    fatigue_enabled (bool, default True) -- per-source switch.
    fatigue_ar_coeff (float, default 0.995) -- AR(1) persistence.
    fatigue_noise_scale (float, default 0.005) -- per-step innovation SD.
    fatigue_contribution_weight (float, default 0.15) -- additive weight on
      harm_obs_a.
    interoceptive_change_threshold (float, default 0.01) -- |delta_harm_obs_a|
      floor used to count body-noise vs agent-caused harm-state-change
      events (calibration-target denominator).
  Data flow (_get_observation_dict): existing SD-022 limb-damage build OR
    legacy 50-dim EMA build -> harm_obs_a numpy array ->
    [if interoceptive_noise_enabled] _apply_interoceptive_noise: fatigue
    AR(1) update / additive contribution -> sensitisation Poisson onset and
    multiplicative amplification -> autonomic per-element Gaussian ->
    delta-event classification vs cached previous tick -> torch.from_numpy.
  info dict tags (always present, 0 / False when disabled):
    interoceptive_noise_enabled, interoceptive_noise_scale,
    interoceptive_n_autonomic_events, interoceptive_n_sensitisation_events,
    interoceptive_n_fatigue_events, interoceptive_n_body_noise_events,
    interoceptive_n_agent_caused_harm_events, interoceptive_fatigue_state,
    interoceptive_sensitisation_amplification.
    The ratio interoceptive_n_body_noise_events / interoceptive_n_agent_caused_harm_events
    is the calibration target signal for the validation experiment
    (target 1:1-2:1 at ARM_2). Classification of |delta_harm_obs_a| events
    uses _last_transition_type cached from step() to attribute agent-caused
    transitions; everything else counts as body-noise-caused.
  Backward compatible: interoceptive_noise_enabled=False by default;
    _apply_interoceptive_noise short-circuits; no RNG draws; no state
    advance; harm_obs_a returned unchanged. RNG draws gated inside
    `if interoceptive_noise_enabled:` AND per-source switches so seed
    sequences for existing experiments are bit-identical when disabled.
    Default vs explicit-False bit-identical OFF guarantee verified
    2026-05-03 (50-tick parity check).
  Activation smoke (2026-05-03, 4-arm sweep, 200 ticks each, 8x8 env,
  random policy):
    ARM_0 (OFF, scale=1.0): all SD-048 counters zero.
    ARM_1 (ON, scale=0.25): autonomic 0 (sigma 0.005 below threshold 0.01),
      n_body_noise=7. Sensitisation / fatigue rare at this scale.
    ARM_2 (ON, scale=1.0): n_aut=176, n_sens=1, fatigue state evolves
      ~+/-0.04, max sensitisation amp=0.8 (one event x 0.8x contribution).
    ARM_3 (ON, scale=4.0): n_aut=200 saturated, n_sens=5.
    All three sources fire end-to-end. Body_noise:agent_caused ratio is
    dominated by random-policy contact rate at small env (164/200
    agent-caused events); validation experiment will use a larger env or
    sparser policy to land the calibration band per the SD doc's
    recalibration interpretation row.
  Implementation choices (deviations / clarifications from SD doc):
    - Flat kwargs on CausalGridWorld.__init__ rather than nested dataclass.
      Matches SD-047 / SD-022 / SD-023 / SD-029 precedent for env-only SDs.
    - Output-side perturbation on harm_obs_a readout, NOT modification of
      upstream limb_damage state. Biologically correct (interoceptive
      reporting noise vs change to underlying body) and avoids interaction
      with SD-022's failure_prob_scale movement-failure mechanism.
    - Cumulative sensitisation amplification capped at 5.0x to defend
      against long Poisson tails (numerical stability; not in SD doc).
    - Per-source diagnostic counters distinguish autonomic / sensitisation /
      fatigue source firing (always-on per-source RNG draws cross threshold)
      from the calibration-target body_noise / agent_caused classification
      (steps where readout |delta| > threshold, attributed by
      _last_transition_type).
  No trainable parameters. Pure stochastic readout enrichment. No phased
  training needed (env-only substrate; no new encoder, no new training
  target).
  MECH-094: not applicable (env observation stream, not replay / simulation
    content). Validation experiments call sense() / step() in waking mode;
    simulation paths do not invoke env.step.
  Validation experiment: V3-EXQ-511 queued (4-arm noise sweep
    {OFF / 0.25x / 1.0x / 4.0x}, 3 seeds, larger env + sparser policy to
    land the 1:1-2:1 calibration band; ARC-058-equivalent C1-C4 metrics
    deferred to the comparator-gap behavioural successor V3-EXQ-512).
    Pre-registered prediction per SD doc: ARM_0 reproduces current
    substrate-ceiling pattern (no body-noise events); ARM_2 produces
    calibrated mixed substrate; ARM_1 / ARM_3 quantify under- and
    over-noise regimes per Asai 2016 non-monotonic.
  Design doc: REE_assembly/docs/architecture/sd_048_interoceptive_noise_dynamics.md
  See SD-048, SD-011 (z_harm_a stream prerequisite), SD-022 (agent-caused
    body-state variance prerequisite), SD-047 (Level 1 environmental
    counterpart, parallel architectural logic), ARC-058 (primary Level 2
    comparator unblock), ARC-033 (competing arbitration, same substrate
    enrichment), ARC-061 (reafference comparator family Level 2
    contribution), MECH-094 (call-site scoping; not applicable).

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
  Oracle path (MECH-263, 2026-05-04): OFCConfig.use_outcome_oracle (bool,
    default False). When True, OFCAnalog.query_outcome(z_harm_s, action,
    e2_harm_s) delegates to E2HarmSForward.forward() with no_grad + detach;
    raises AssertionError when oracle is disabled. REEAgent stores
    _ofc_oracle_predictions (per-candidate oracle list, cleared on reset).
    REEConfig.use_ofc_outcome_oracle (default False) wired through from_dims().
    Bit-identical OFF: the oracle block in select_action() is entirely gated
    by oracle_is_ready (False when use_outcome_oracle=False) so all existing
    experiments are unaffected. 7/7 preflight + 184/184 contracts PASS with
    oracle OFF.
  Validation experiment: V3-EXQ-485 queued (diagnostic -- five sub-tests:
    instantiation + state_code shape, gate=1 vs gate=0 update modulation,
    bias zero at init, backward compat, reset clears state_code). Smoke
    PASS 2026-04-26. V3-EXQ-485a queued (6-sub-test oracle round-trip
    extension; UC6 new: A query_outcome() matches e2_harm_s.forward() <1e-6
    diff; B AssertionError when oracle disabled; C reset() clears prediction
    caches; D oracle_is_ready=False by default; E get_state() exposes oracle
    diagnostics). Behavioural validation (MECH-263 devaluation + task-role
    discrimination) deferred to env-extension EXQs.
  Phased training: deferred along with MECH-263 behavioural signatures.
  Design doc: REE_assembly/docs/architecture/sd_033b_ofc_analog.md
  See SD-033, SD-033a (sibling consumer; additive E3 bias composition),
    SD-033b, MECH-261, MECH-263, SD-032a, SD-032b, MECH-094.

## SD-033b GAP-8: trainable OFC state_bias_head (mirror of SD-033a GAP-D) (2026-06-09)
- SD-033b GAP-8 enrichment: pfc.ofc_analog.train_state_bias_head -- IMPLEMENTED
  2026-06-09. The exact OFC-side mirror of the SD-033a GAP-D rule_bias_head
  trainable enrichment (landed 2026-05-17). Unblocks commitment_closure:GAP-8 --
  the deferred trained-OFC-head behavioural arm that takes SD-033b from PARTIAL
  (485b/485c representation-level diagnostics PASS, reviewed 2026-06-04) to the
  FULL candidate->provisional behavioural validation.
  Modules:
    ree_core/pfc/ofc_analog.py -- OFCConfig.train_state_bias_head (bool, default
      False). When False (default): OFCAnalog.state_bias_head's last Linear is
      zeroed at init (bias output exactly 0 -- bit-identical to the original
      SD-033b landing). When True: last Linear keeps random init so the head
      moves under the E3 score-aggregation gradient from the first optimizer
      step. New OFCAnalog.bias_head_parameters() returns state_bias_head.parameters()
      for experiment optimizer inclusion. get_state() now reports
      train_state_bias_head.
    ree_core/utils/config.py -- REEConfig.ofc_train_state_bias_head (bool, default
      False) + from_dims signature param + assignment.
    ree_core/agent.py -- OFC build site forwards
      train_state_bias_head=getattr(config, "ofc_train_state_bias_head", False)
      into the OFCConfig(...) construction (getattr fallback -> bit-identical when
      the flat attr is absent).
  Data flow (when trained): E3 loss -> score_bias -> OFCAnalog.compute_bias() ->
    state_bias_head weights. Identical gradient path to how GAP-D trains the
    SD-033a rule_bias_head (experiments add list(agent.ofc.bias_head_parameters())
    to a P1 optimizer and train via E3-gradient REINFORCE; see v3_exq_598b for the
    SD-033a precedent scaffold).
  Backward compatible: ofc_train_state_bias_head=False by default -> last Linear
    zeroed -> bias output exactly 0 -> every existing experiment bit-identical.
    Smoke 2026-06-09: OFF zeroed/bias==0; ON random-init preserved + 4 head params
    require grad + gradient reaches every head param on the module path + agent
    from_dims wiring + default-OFF agent bit-identical. 80/80 boot+config+ofc
    contracts PASS; v3_exq_485b --dry-run unchanged.
  CLAMP NOTE (experiment-calibration, NOT a wiring defect): compute_bias clamps to
    +/-ofc_bias_scale (default 0.1). At random init the head's own Linear bias
    terms can push the pre-clamp output past the rail for ~all candidates, zeroing
    grad in the saturated region. The behavioural arm must ensure per-candidate
    variation lands some candidates in-band (the SD-033a 598b REINFORCE-over-
    candidates pattern handles this; consider a larger ofc_bias_scale or a
    pre-clamp training signal if saturation stalls C2).
  Phased training REQUIRED for the behavioural arm (not for this substrate
    landing): P0 encoder warmup, P1 trains the head on the frozen-encoder
    state_code path via E3-gradient REINFORCE, P2 eval -- exactly as 598b does for
    SD-033a (defends against the EXQ-166b/c/d joint-encoder-head-collapse mode).
  Substrate-specific design constraint for the behavioural arm: the OFC state_code
    reads only z_world + z_harm (no appetitive/drive input), so SD-049 satiety and
    the GAP-3 counter-evidence primitive are invisible to it. The behavioural
    readout therefore uses AVERSIVE outcome devaluation (set ofc_harm_dim>0 so
    z_harm enters the state_code), parallel to how 485b used aversive devaluation
    and 485c used task-stage structure -- NOT an appetitive devaluation.
  MECH-094: N/A -- OFCAnalog.compute_bias/update run only on the waking
    select_action/sense path; no simulation/replay write surface. The existing
    simulation_mode handling on the OFC oracle path is untouched.
  Validation experiment: V3-EXQ-485d substrate-readiness diagnostic queued via
    /queue-experiment (claim_ids=[]; frozen vs trainable head ablation -- frozen
    bias ~0, trainable head moves under E3 gradient). The FULL SD-033b behavioural
    arm (aversive-devaluation behaviour change, ofc_harm_dim>0) is the separate
    /queue-experiment session that follows and closes commitment_closure:GAP-8.
  Design doc: REE_assembly/docs/architecture/sd_033b_ofc_analog.md (GAP-8 trainable-
    head note).
  See SD-033b (parent), SD-033a GAP-D (the rule_bias_head precedent this mirrors;
    landed 2026-05-17), commitment_closure:GAP-8 (the closure-plan gap this
    unblocks), MECH-263 (the functional signatures the behavioural arm validates),
    MECH-262 (SD-033a sibling behavioural test via 598b C3), V3-EXQ-485b/485c
    (the representation-level diagnostics this complements), MECH-094 (N/A).

## SD-033b GAP-8 DECOUPLE: separate OFC devaluation_bias_head (clamp-starved devalued range; failure_autopsy V3-EXQ-485l) (2026-06-22)
- SD-033b GAP-8 devaluation-head decouple: pfc.ofc_analog.devaluation_bias_head --
  IMPLEMENTED 2026-06-22 (substrate; SD-033b / MECH-263 stay candidate /
  substrate_conditional / pending_retest_after_substrate -- this PROMOTES NOTHING;
  the 485-lineage behavioural retest is GATED behind this build and queued
  separately). Routed by the user-confirmed failure_autopsy_V3-EXQ-485l_2026-06-22
  (re-derive brake FIRED -> implement-substrate; a plain 485m gain-tweak REFUSED).
  ROOT CAUSE (code-confirmed): the single shared OFCAnalog.state_bias_head under the
  +/-ofc_bias_scale clamp has NO feasible gain band. MECH-449 (Go/No-Go constitution)
  is BUILT + validated (689g PASS) but its viability No-Go input is starved: the
  devaluation re-ranking driver must produce a differentiated DEVALUED cross-candidate
  range above the 0.05 readout floor while the SAME head + SAME +/-0.5 clamp also
  carries the C2 high-threat discrimination range. 485k gain 4.0 SATURATED the clamp
  (devalued range 0.0); 485l gain 1.5 UNDERSHOT it (0.031 < 0.05). The bias VECTOR
  inverts cleanly (l2 1.83, cosine -0.716, C1b PASS 2/3) -- direction is RIGHT -- but
  the clamp compresses the MAGNITUDE below floor, so MECH-449 engaged only 1/3 (< 2/3
  gate) and the devaluation behavioural DV could not register. The residual is
  STRUCTURAL SCALE, not a tunable gain on the shared clamped head.
  THE FIX (no-op default; bit-identical OFF): a SECOND output head decoupling the
  devaluation re-ranking from the C2 discrimination readout.
    Module: ree_core/pfc/ofc_analog.py -- OFCConfig.use_devaluation_head (bool,
      default False) + devaluation_bias_scale (float, default 2.0) +
      train_devaluation_head (bool, default False). When use_devaluation_head, OFCAnalog
      builds devaluation_bias_head (same Linear(state_dim+world_dim -> hidden -> 1)
      shape as state_bias_head), sharing the [state_code, per-candidate summary] input
      but clamped to +/-devaluation_bias_scale -- an INDEPENDENT dynamic range. New
      method compute_devaluation_bias() (clamp +/-devaluation_bias_scale; returns zeros
      when the head is off) + devaluation_bias_head_parameters() (empty iter when off) +
      get_state() reports the new fields + _last_devaluation_bias_abs_mean diagnostic.
      Last Linear zeroed unless train_devaluation_head (mirror of the GAP-8
      state_bias_head zeroing) so the head is bit-identical until deliberately trained.
    Config: REEConfig.use_ofc_devaluation_head (False) + ofc_devaluation_bias_scale
      (2.0) + ofc_train_devaluation_head (False) + from_dims signature + assignment.
    Agent: the OFCConfig build site (agent.py) forwards all three via getattr fallback
      (absent flat attr -> bit-identical). NO runtime composition change -- select_action
      does NOT add the devaluation bias; the head is read by the experiment directly
      (compute_devaluation_bias -> _build_viability_nogo), so the action stream is
      bit-identical.
  WHY this is the structural fix (autopsy a+b+c in one): the C2 discrimination head
  keeps its +/-ofc_bias_scale clamp untouched (magnitude no longer traded against C1),
  while the devaluation head's larger independent clamp lets the in-band re-ranking gain
  produce a supra-floor differentiated devalued range WITHOUT saturating (decouple = a;
  independent clamp = rescale = b); the No-Go viability mapping re-derives from the
  decoupled head's range (c). It is a structural decouple, NOT a gain change on the
  shared clamped head -- the lettered-iteration loop the re-derive brake refuses.
  Backward compatible: use_ofc_devaluation_head=False by default -> devaluation_bias_head
  is None, compute_devaluation_bias returns zeros, no params -> bit-identical (every
  existing experiment reads only compute_bias). 8/8 preflight + OFC/boot/config contracts
  PASS; 6 new contracts in tests/contracts/test_sd_033b_gap8_devaluation_head.py (C1
  default-off no-op / C2 trainable decoupled head clears the 0.05 floor at its own clamp
  while C2 head stays clamped to bias_scale untraded / C3 dev-only optimizer trains the
  dev head + freezes state_bias_head / C4 untrained head zeroed + get_state fields / C5
  from_dims wiring / C6 inert when use_ofc_analog off). Module smoke: trainable decoupled
  head range 0.63 >> 0.05 floor at clamp 2.0; C2 head |max| 0.10 <= bias_scale untraded;
  dev-only gradient leaves C2 frozen.
  Phased training: N/A at the substrate level (a second readout head on the existing
  state_code, same E3-gradient path as the GAP-8 state_bias_head -- no new latent
  target, no collapse risk). The 485-lineage retest trains the head in P1 via the
  established REINFORCE-over-candidates pattern (frozen-encoder), exactly as the
  state_bias_head GAP-8 arm does. MECH-094: N/A -- compute_bias / compute_devaluation_bias
  run on the waking select_action / experiment path; no replay/memory write surface.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (no second head), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. SD-033b / MECH-263 stay candidate / substrate_conditional /
  pending_retest_after_substrate; MECH-448 / MECH-449 unweakened (both engaged correctly
  within scope). claims.yaml carries only an implementation_note (no flag/confidence/
  status change).
  Validation experiment: a 485-lineage behavioural retest (NEW letter, supersedes 485l;
  claim_ids=[SD-033b, MECH-263]) GATED behind this build -- arms use_ofc_devaluation_head
  + ofc_train_devaluation_head + the in-band devalued re-ranking driver on the decoupled
  head, reads the devalued range + No-Go viability from compute_devaluation_bias, and
  re-states the readiness gate on the decoupled head's supra-floor differentiated range.
  Queued via /queue-experiment in a separate session per the chip scope; do NOT re-author
  485l.
  Design doc: REE_assembly/docs/architecture/sd_033b_ofc_analog.md (GAP-8 decouple note).
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-485l_2026-06-22.md.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (f_dominance_conversion_ceiling failure record).
  See SD-033b (parent), SD-033b GAP-8 trainable state_bias_head (the head this decouples
  from; landed 2026-06-09), MECH-449 (Go/No-Go constitution -- the No-Go this supra-floor
  range feeds; built + validated 689g, input-starved here), MECH-448 (rank-preserving
  F->eligibility demotion; converts the C2 discrimination signature), MECH-263 (the
  devaluation + task-role signatures the retest validates), commitment_closure:GAP-8 (the
  closure-plan gap), f_dominance_conversion_ceiling (the substrate lineage), V3-EXQ-485k
  (saturate) / V3-EXQ-485l (undershoot -- the FAIL this addresses), MECH-094 (N/A).

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

## SD-034 AMEND: commitment-closure-control-plane (env-completion hook + de-commit hold) (2026-06-12)
- commitment-closure-control-plane -- IMPLEMENTED 2026-06-12. The behavioural-
  authority amend the SD-034 ClosureOperator lacked on the 603n foraging-competent
  substrate. Routed by the confirmed failure_autopsy_SD-034-closure-cluster_2026-06-12
  (V3-EXQ-460c + V3-EXQ-468c: closure wired at unit level but no behavioural
  authority -- 460c n_closures=0 on 3/3 seeds despite env sequence_completions=2/5/6;
  468c closure-coupled beta release fires but committed_frac cap-pins ~39 both arms).
  TWO no-op-default legs (bit-identical OFF):
    Leg A -- explicit env-completion hook seam (closes 460c n_closures=0):
      ree_core/agent.py new REEAgent.notify_env_completion(action_class, z_world=None,
      bypass_mode_conditioning=False, simulation_mode=False) -> Optional[ClosureEvent].
      When use_closure_env_completion_hook=True AND closure_operator is not None AND
      not simulation, routes the env's transition_type=="sequence_complete" signal
      into closure_operator.emit_closure(action_class, z_world or _current_latent.z_world,
      ...). Returns the ClosureEvent so the harness counts fires / No-Go installs;
      None (no-op) when the flag is off / no operator / simulation. The experiment
      harness calls it post-env.step on a completion tick (the *d retest does the
      call). Fixes the *c-cohort gap: the env emitted completions but nothing routed
      them into emit_closure -- the agent relied solely on the automatic
      rule_state-stability detector, whose conjunction (delta<0.001 x3 + meaningful
      magnitude + sd_033a gate>=0.5 + allowed mode) was unmet on the
      untrained/zeroed rule_bias_head + SP-CEM-perturbed agent.
    Leg B -- de-commitment hold / refractory (closes 468c committed_frac cap-pin):
      ree_core/heartbeat/beta_gate.py gains _refractory_remaining + apply_refractory(n)
      + refractory_remaining property; elevate() is a NO-OP while the window is active
      (increments _n_elevation_refractory_blocked); propagate() decrements the window
      once per tick (propagate runs every select_action); reset() + get_state()
      extended (sd034_refractory_remaining / sd034_n_elevation_refractory_blocked).
      ree_core/governance/closure_operator.py: ClosureOperatorConfig.decommit_hold_ticks
      (default 0); _fire() installs beta_gate.apply_refractory(decommit_hold_ticks) on
      any closure fire (recorded as ClosureEvent.decommit_refractory_applied) so the
      closure-driven release survives >1 tick -> measurable latch-occupancy drop
      instead of immediate re-commit.
    Leg C (experiment-side, NOT substrate): the *d retests set the landed GAP-D
      lateral_pfc_train_rule_bias_head so the automatic detector has a
      magnitude-bearing rule_state, gate readiness on n_closures>0 reachable, and
      read de-commitment on a non-cap-pinned statistic (post-completion uncommitted
      fraction / committed-run-length delta).
  Config (REEConfig + from_dims; both no-op default, bit-identical OFF):
    use_closure_env_completion_hook (False), closure_decommit_hold_ticks (0). The
    decommit_hold_ticks is wired into the ClosureOperatorConfig build in
    REEAgent.__init__ via getattr fallback (absent flat attr -> bit-identical).
  Backward compatible: both default no-op -> notify_env_completion returns None,
    _fire never calls apply_refractory, elevate() bit-identical. 1014 contracts
    (1008 prior + 6 new in tests/contracts/test_sd034_decommit_hold_and_env_hook.py:
    C1 BetaGate refractory default-OFF bit-identical / C2 refractory blocks-then-
    expires + max-not-truncate + n<=0 no-op / C3 ClosureOperator installs refractory
    on fire + default-0 no-op / C4 agent hook OFF no-op / C5 hook ON fires emit_closure
    + installs hold + blocks re-commit / C6 MECH-094 simulation gate) + 7/7 preflight
    PASS. v3_exq_460c --dry-run unchanged (hook OFF -> reproduces the prior FAIL
    signature). Agent smoke 2026-06-12: hook ON routes a completion -> n_closures 0->1
    + nogo_pushed=3 (the 460c n_closures=0/nogo=0 defect closed on the positive
    control) + refractory_applied=5 blocks re-commit (the 468c immediate-re-commit
    defect addressed).
  Phased training: N/A (pure wiring + arithmetic; no learned parameters). MECH-094:
    the env hook is waking-only + simulation_mode/hypothesis_tag-gated (a replay/DMN
    completion cannot emit a waking done-token); the refractory is a control-state
    transition, not memory content. Evidence-staleness (Step 8.5): NOT triggered --
    no-op-default; every existing experiment uses the defaults (hook off, hold 0), so
    no dependent claim's measured mechanism changed. KEEP all evidence
    (SD-034/MECH-260/MECH-268/MECH-090).
  Validation experiments: V3-EXQ-460d (supersedes 460c) + V3-EXQ-468d (supersedes
    468c), queued via /queue-experiment. Retest gate (per the autopsy failure_record):
    n_closures>=1 reachable on the positive control AND nogo_installed>=1 on >=2/3
    seeds after the env->emit_closure wiring; and a non-cap-pinned de-commitment DV
    showing ON<OFF on >=2/3 seeds. substrate_queue commitment-closure-control-plane
    ready stays FALSE until both clear.
  GOVERNANCE: SD-034 / MECH-260 / MECH-268 / MECH-090 NEITHER validated NOR weakened;
    SD-034 provisional holds, MECH-261 stable untouched, MECH-260 stays candidate.
    claims.yaml carries only an implementation_note (no flag/confidence change).
  Design doc: REE_assembly/docs/architecture/sd_034_governance_closure_operator.md
    (commitment-closure-control-plane amend section). Autopsy:
    REE_assembly/evidence/planning/failure_autopsy_SD-034-closure-cluster_2026-06-12.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
    (commitment-closure-control-plane).
  See SD-034 (parent), MECH-090 (BetaGate -- the refractory host), MECH-260 (No-Go;
    strictly downstream of a closure fire -- its 460c zero was a positive-negative),
    MECH-268 (dACC PE saturation; coupled in 468c), MECH-261 (mode-conditioning;
    stable, not exercised), SD-033a GAP-D lateral_pfc_train_rule_bias_head (the *d
    rule_state lever), V3-EXQ-460c/468c (the FAILs this amend addresses), V3-EXQ-460d/468d
    (validation), MECH-094 (call-site scoping + simulation gate).

## SD-034 AMEND: commitment-closure-control-plane BETA-ENGAGEMENT (couple closure->beta elevation) (2026-06-17)
- commitment-closure-control-plane beta-engagement amend -- IMPLEMENTED 2026-06-17
  (substrate; SD-034/MECH-260/MECH-261 stay provisional/candidate + non_contributory +
  pending_retest_after_substrate -- PROMOTES NOTHING until the post-amend V3-EXQ-460f
  returns a contributory PASS). Mechanism (a) of the beta-engagement deliverable
  (user-confirmed (a)-only; (b)/(c) deferred). Routed by the confirmed
  failure_autopsy_V3-EXQ-460e_2026-06-17.
  ROOT CAUSE (code-confirmed, agent.py:5855-5868 bistable elevate block): the 460e config
  sets only cfg.heartbeat.beta_gate_bistable=True (both MECH-090 R-c gates OFF), so the
  bistable BetaGate elevates iff result.committed (running_variance < commit_threshold) --
  a decisive NATURAL commit-entry that fires on only 1/3 seeds on the 603n foraging
  substrate. But total_committed_steps counts e3._committed_trajectory is not None, which
  the Leg-A env-completion hook / closure control-plane populates INDEPENDENTLY -> the
  commit-without-beta dissociation (seeds 42/43 committed_steps 2415/2019 but
  total_beta_elevated=0; closures fire 7/6; seed 44 engages beta both arms -> C2 PASS, the
  existence proof). The Leg-C trained rule_bias_head biases per-candidate SCORING, not the
  running_variance that gates commit-ENTRY, so it cannot rescue engagement; the de-commit
  DV (ON<OFF latch occupancy) has nothing to measure when beta never elevates.
  THE FIX (no-op-default; bit-identical OFF):
    Config: REEConfig.use_closure_commit_beta_coupling (bool, default False) + from_dims
      passthrough (ree_core/utils/config.py). Mirrors the use_closure_env_completion_hook /
      closure_decommit_hold_ticks closure-plane flag precedent.
    Agent (ree_core/agent.py, bistable elevate block ~5855): compute
      _closure_commit_active = (flag AND self.e3._committed_trajectory is not None) and
      _commit_for_beta = bool(result.committed) OR _closure_commit_active; the elevate
      condition uses _commit_for_beta instead of result.committed, KEEPING the full
      should_admit_elevation(score_margin, K) AND _readiness_admits conjunction (so the
      coupling composes with MECH-090 when those gates are on; both are permissively True
      on the coupled path when result.committed is False, which is the 460e gates-off case).
      So beta occupancy tracks the closure-plane commitment on every seed where one forms,
      and the Leg-B de-commit refractory then produces a measurable ON<OFF latch-occupancy
      drop (the 460f DV). After elevate(), when _closure_commit_active and not
      result.committed, calls beta_gate.note_closure_coupled_elevation().
    BetaGate (ree_core/heartbeat/beta_gate.py): note_closure_coupled_elevation() +
      _n_closure_coupled_elevations counter (pure diagnostic; does not change gate state) +
      sd034_n_closure_coupled_elevations in get_state() + per-episode reset(). The 460f
      non-vacuity readout (the coupling, not the fragile natural commit-entry, engaged the
      latch).
  Backward compatible: use_closure_commit_beta_coupling=False by default -> _commit_for_beta
    == result.committed -> the bistable block is bit-identical; the diagnostic counter stays
    0. 5/5 new contracts (tests/contracts/test_sd034_closure_beta_coupling.py: C1 BetaGate
    primitive counter/get_state/reset / C2 config default + from_dims / C3 bit-identical-OFF
    action stream over a 12-step run / C4 coupling-ON elevates under a forced closure-plane
    commit WITHOUT a natural commit + counter>=1 / C5 coupling-OFF does NOT elevate under the
    same forced dissociation, counter==0) + 7/7 preflight + full contract suite 1084 passed
    (the 2 failures are the concurrent runner-observability session's experiment_runner.py
    ERROR-branch tests in test_runner_fail_branch_persists_result.py, outside this change set).
  Phased training: N/A (pure control-state wiring + a diagnostic counter; no learned
    parameters). MECH-094: N/A -- waking select_action control-state transition; no
    replay/memory write surface (same scope decision as the Leg-B refractory). Evidence-
    staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment uses
    the default (coupling off), so no dependent claim's measured mechanism changed. KEEP all
    evidence.
  GOVERNANCE: PROMOTES NOTHING. SD-034 stays provisional, MECH-260 candidate, MECH-261
    stable; all stay non_contributory + pending_retest_after_substrate (never fairly tested --
    the 460e readiness gate self-routed before the C2 de-commit DV ran). claims.yaml NOT
    modified (substrate-only amend; the amend record lands in substrate_queue.json
    commitment-closure-control-plane implementation_log).
  Validation experiments: V3-EXQ-460f (de-commit retest on the non-cap-pinned ON<OFF
    latch-occupancy DV) + V3-EXQ-468e (MECH-090 commit-entry conjunction under the trained
    head) -- queued TOGETHER via /queue-experiment on the amended substrate, BOTH arming
    beta_gate_bistable=True + use_closure_commit_beta_coupling=True + the Leg-A env-completion
    hook + Leg-B decommit hold + the Leg-C scaffold_train_rule_bias_head. Do NOT re-author
    460d/468d (superseded). Mechanism (b) commit_threshold adaptation stays available as an
    experiment-config lever if 460f still self-routes on beta engagement. substrate_queue
    commitment-closure-control-plane ready STAYS false until 460f scores a contributory PASS.
  Design doc: REE_assembly/docs/architecture/sd_034_governance_closure_operator.md.
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460e_2026-06-17.md.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
    (commitment-closure-control-plane).
  See SD-034 (parent + Legs A/B 2026-06-12 + Leg C 2026-06-16), MECH-090 (BetaGate -- the
    latch this couples the closure-plane commit to; the should_admit_elevation/_readiness
    conjunction is preserved), MECH-260 (No-Go), MECH-261 (mode-conditioning), the Leg-B
    de-commit refractory (the occupancy-drop actuator the coupling makes measurable), the
    Leg-C scaffold_train_rule_bias_head (per-candidate scoring; orthogonal to commit-entry
    decisiveness), V3-EXQ-460e (the FAIL this amend addresses), V3-EXQ-460f/468e (validation),
    MECH-094 (N/A).

## SD-034 AMEND: commitment-closure-control-plane DE-COMMIT-AUTHORITY MAGNITUDE (committed-run-scaled Leg-B refractory) (2026-06-19)
- commitment-closure-control-plane de-commit-authority magnitude amend -- IMPLEMENTED
  2026-06-19 (substrate; SD-034/MECH-260/MECH-261 stay provisional/candidate/stable +
  non_contributory + pending_retest_after_substrate -- PROMOTES NOTHING until the
  post-amend V3-EXQ-460g returns a contributory PASS). Part (a) of the de-commit-magnitude
  deliverable; part (b) (within-arm C2 DV + sd034_n_closure_coupled_elevations>0 non-vacuity
  gate) is experiment-side in the V3-EXQ-460g re-issue. Routed by the confirmed
  failure_autopsy_V3-EXQ-460f_2026-06-18 (user-adjudicated 2026-06-18T08:04Z governance
  cycle; substrate_queue commitment-closure-control-plane implementation_hint 460f amend).
  ROOT CAUSE (460f, the residual one link past beta-engagement): the 2026-06-17
  beta-engagement amend WORKED -- all 4 readiness gates cleared and the C2 de-commit
  occupancy-drop DV ran for the first time (PASS seed 42: ON 23.73 < OFF 35.67, -33.5%;
  FAIL 2/3) -- but on strong-natural-commit seeds the closure->beta coupling was INERT
  (sd034_n_closure_coupled_elevations 36/52 seed 42 vs 0/0 seeds 43/44), so the DV reduced
  to the bare Leg-B 5-tick refractory whose magnitude (~20-35 tick-blocks) is SWAMPED by the
  ~530-560 natural-commit elevated steps. NOT a falsification (seed 42 + 460e seed 44 are
  existence proofs of the correct de-commit SIGN); the gap is de-commit-authority MAGNITUDE.
  THE FIX (no-op-default; bit-identical OFF): scale the Leg-B de-commit refractory installed
  at a closure fire by the COMMITTED-RUN LENGTH captured from the BetaGate BEFORE the
  closure's own release(), so a long committed run -- the exact source of the swamping latch
  occupancy -- triggers a proportionally long post-closure hold:
    n = closure_decommit_hold_ticks + round(closure_decommit_hold_scale_with_run * run_length),
    clamped to closure_decommit_hold_max_ticks (0 = uncapped).
  Modules:
    ree_core/heartbeat/beta_gate.py -- BetaGate gains a per-run _committed_run_length counter
      + committed_run_length property + sd034_committed_run_length get_state key. Incremented
      once per propagate() tick while elevated; reset to 0 on a FRESH elevate()
      (not-elevated -> elevated transition; a re-elevate while already elevated leaves it
      unchanged) and on release(); cleared in reset(). Pure bookkeeping -- never read unless
      the lever is armed, so bit-identical when off.
    ree_core/governance/closure_operator.py -- ClosureOperatorConfig gains
      decommit_hold_scale_with_run (0.0) + decommit_hold_max_ticks (0). _fire() captures
      run_length_at_fire = beta_gate.committed_run_length BEFORE step (a) release() (which
      resets it), then step (a.2) installs the scaled hold. With scale 0.0 (default),
      hold_ticks == decommit_hold_ticks unchanged -> bit-identical to the fixed-hold path
      (and with decommit_hold_ticks 0 too, no refractory at all).
    ree_core/utils/config.py -- REEConfig.closure_decommit_hold_scale_with_run (0.0) +
      closure_decommit_hold_max_ticks (0) + from_dims signature + assignment.
    ree_core/agent.py -- the ClosureOperatorConfig build site forwards both via getattr
      fallback (absent flat attr -> bit-identical), mirroring the closure_decommit_hold_ticks
      precedent.
  NOTE on the autopsy's "EITHER...OR" (committed-run-scaled refractory OR active
  MECH-342-style release-pressure): the closure _fire() ALREADY calls beta_gate.release()
  (drops the latch at the fire), so option B's "drive the latch DOWN rather than block
  re-entry" distinction is moot for this gap -- the latch is already down at fire; the only
  lever is HOW LONG to keep it down, which is exactly the refractory. So the committed-run-
  scaled refractory is the faithful magnitude lever; an active release-pressure event would
  duplicate MECH-342 with no distinct mechanism here (user-confirmed A-only, AskUserQuestion
  2026-06-19).
  Backward compatible: closure_decommit_hold_scale_with_run=0.0 by default -> the refractory
    uses closure_decommit_hold_ticks exactly as the 2026-06-12 Leg-B landing -> bit-identical;
    the run-length counter increments but is never read. 6 new contracts in
    tests/contracts/test_sd034_decommit_magnitude.py (C1 committed_run_length counter
    increments-while-elevated / resets-on-release / resets-on-fresh-elevate-not-re-elevate /
    reset() clears + get_state key; C2 scale 0.0 bit-identical to the fixed hold independent
    of a 40-tick run; C3 scale>0 -> n = base + round(scale*run_length) captured before
    release + longer run -> longer hold; C4 max_ticks clamp; C5 from_dims surfaces both flags
    + agent forwards them; C6 agent action stream bit-identical default vs explicit scale=0.0
    over an 8-step loop) + 7/7 preflight + full contract suite 1101 passed (the 3 failures --
    control_vector C4 + 2 runner_fail_branch -- are the documented pre-existing flakes,
    CONFIRMED failing identically on a clean stash, outside this change set). Activation smoke
    2026-06-19 (the 460f scenario, 530-step committed run then a closure fire): FIXED
    base5/scale0 -> 5 ticks suppressed (swamped); SCALED base5/scale0.1 -> 58 ticks (capped 60
    -> 58) -- the de-commit authority now scales with the latch occupancy it must overcome.
  Phased training: N/A (control-state counter + scalar arithmetic; no learned parameters).
    MECH-094: N/A -- waking select_action control-state transition; no replay/memory write
    surface (same scope as the Leg-B refractory the lever extends; the run-length counter only
    advances on the waking propagate() path). Evidence-staleness (Step 8.5): NOT triggered --
    no-op-default lever; every existing experiment uses the default (scale 0.0), so no
    dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. SD-034 stays provisional, MECH-260 candidate, MECH-261 stable;
    all stay non_contributory + pending_retest_after_substrate. claims.yaml NOT modified
    (substrate-only amend; the amend record lands in substrate_queue.json
    commitment-closure-control-plane implementation_log).
  Validation experiment: V3-EXQ-460g (supersedes V3-EXQ-460f), queued via /queue-experiment --
    the de-commit retest arming the magnitude lever (closure_decommit_hold_scale_with_run +
    max_ticks) ON TOP of the beta-engagement-amended substrate (beta_gate_bistable +
    use_closure_commit_beta_coupling + Leg-A env-completion hook + Leg-B hold + Leg-C
    scaffold_train_rule_bias_head), with (b) the C2 DV redesigned to a WITHIN-ARM
    around-closure occupancy delta (pre-vs-post-closure window on the ON arm) and the
    non-vacuity gate tightened to sd034_n_closure_coupled_elevations > 0 on scored seeds.
    Acceptance: ON<OFF de-commit on the within-arm non-cap-pinned statistic on >=2/3 seeds
    with the coupling non-vacuity gate met. Do NOT re-author 460d/460e/460f; the parallel
    V3-EXQ-468e (MECH-090 commit-entry conjunction) leg is separately owed. substrate_queue
    commitment-closure-control-plane ready STAYS false until 460g scores a contributory PASS.
  Design doc: REE_assembly/docs/architecture/sd_034_governance_closure_operator.md
    (de-commit-authority magnitude amend section). Autopsy:
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460f_2026-06-18.md.
    Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
    (commitment-closure-control-plane).
  See SD-034 (parent + Legs A/B 2026-06-12 + Leg C 2026-06-16 + beta-engagement 2026-06-17),
    MECH-090 (BetaGate -- the latch the refractory holds down; the run-length counter rides
    its elevate/propagate/release lifecycle), MECH-342 (the active-release-pressure sibling
    the autopsy named as the alternative; redundant here since _fire already releases),
    MECH-260 (No-Go), MECH-261 (mode-conditioning), the Leg-B de-commit refractory (the
    actuator this lever scales), the closure->beta coupling (the engagement the de-commit DV
    rides), V3-EXQ-460f (the FAIL this amend addresses), V3-EXQ-460g (validation),
    V3-EXQ-468e (separate MECH-090 leg), MECH-094 (N/A).

## SD-034 AMEND: commitment-closure-control-plane REFRACTORY-INDEPENDENT coupling certifier (decouple the de-commit lever from its non-vacuity metric) (2026-06-19)
- commitment-closure-control-plane refractory-independent coupling-certifier amend --
  IMPLEMENTED 2026-06-19 (substrate; MECH-445/446 stay candidate/v3_pending/
  pending_retest_after_substrate -- PROMOTES NOTHING; the SD-034 closure cluster was
  decomposed into MECH-445 coupling-engagement + MECH-446 de-commit-magnitude the same
  day). Routed by the confirmed failure_autopsy_V3-EXQ-460g_2026-06-19
  recommended_substrate_queue_entry (SECONDARY action; the PRIMARY /claim-synthesis
  decomposition is its precondition).
  ROOT CAUSE (code-confirmed, the S5 self-defeating entanglement): the 460f-prescribed
  coupling non-vacuity gate keys on sd034_n_closure_coupled_elevations, which counts only
  closure-coupled beta elevations the gate ENTERS -- but note_closure_coupled_elevation
  is called INSIDE the bistable elevate if-block guarded by `not beta_gate.is_elevated`
  (agent.py:6082-6094), so once the closure-coupled commit latches beta elevated for the
  long committed run (~530-560 steps) the per-ENTRY counter is frozen, and the 460g
  committed-run-scaled de-commit-MAGNITUDE lever (apply_refractory cap 60) blocks
  re-elevation so it cannot re-fire as a transition. Net: scaling the de-commit authority
  UP suppresses its own coupling certifier (the counter collapsed 36 -> 0 on seed 42)
  even though the de-commit DID act (seed-42 within-arm occupancy 0.333 -> 0.0, C2 PASS).
  The two 460f amends (magnitude lever + coupling gate) are mutually self-defeating.
  THE FIX (no-op-default; bit-identical OFF; rides use_closure_commit_beta_coupling):
    ree_core/heartbeat/beta_gate.py -- BetaGate gains _n_closure_commit_intent +
      note_closure_commit_intent() + sd034_n_closure_commit_intent in get_state() +
      per-episode reset(). Pure diagnostic; does not change gate state.
    ree_core/agent.py -- the bistable elevate block calls note_closure_commit_intent()
      when `_closure_commit_active and not result.committed` BEFORE the elevate/refractory
      gate (before the `not is_elevated` + should_admit + readiness conjunction), so the
      closure-plane commit INTENT is certified every E3 tick a closure-coupled commitment
      forms without a natural running_variance crossing -- REGARDLESS of whether the latch
      is held elevated OR the de-commit-magnitude refractory then blocks the elevate. The
      existing sd034_n_closure_coupled_elevations stays (it now measures refractory-/latch-
      surviving elevations); the new counter is the refractory-INDEPENDENT MECH-445
      coupling-engagement certifier the magnitude lever (MECH-446) cannot zero.
  Backward compatible: note_closure_commit_intent only increments when
    use_closure_commit_beta_coupling is on (_closure_commit_active stays False otherwise),
    AND it is a pure int readout with no gate-state effect, so the action stream is
    bit-identical both with the coupling flag OFF (every existing experiment) and ON.
    20/20 SD-034 closure contracts (17 prior + 3 new C6/C7/C8 in
    tests/contracts/test_sd034_closure_beta_coupling.py: C6 primitive counter advances
    under an active refractory + get_state + reset; C7 the load-bearing 460h property --
    a held/blocked elevate gate freezes the coupled counter at 0 while the intent counter
    keeps certifying coupling; C8 coupling-OFF intent counter stays 0) + 8/8 preflight +
    full contract suite 1148 passed (the 3 fails -- control_vector C4 + 2 runner_fail_branch
    -- are the documented pre-existing flakes, untouched by this change). Activation smoke
    2026-06-19 (full amended substrate beta_gate_bistable + coupling + de-commit hold +
    magnitude lever, pinned refractory): sd034_n_closure_commit_intent populates and is
    > 0 where the elevate gate is suppressed.
  Phased training: N/A (control-state counter; no learned parameters). MECH-094: N/A --
    waking select_action control-state readout; no replay/memory write surface.
    Evidence-staleness (Step 8.5): NOT triggered -- no-op-default readout; every existing
    experiment uses the default (coupling off) and the counter has no behavioural effect,
    so no dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-445/446 stay candidate / v3_pending /
    pending_retest_after_substrate; SD-034 umbrella narrowed (2026-06-19 decomposition).
    claims.yaml NOT modified (substrate-only amend; the amend record lands in
    substrate_queue.json commitment-closure-control-plane implementation_log).
  Validation experiment: V3-EXQ-460h (supersedes the de-commit lineage; do NOT re-author
    460d/e/f/g), queued via /queue-experiment -- the de-commit retest arming the full
    amended substrate (beta_gate_bistable + use_closure_commit_beta_coupling + Leg-A
    env-completion hook + Leg-B committed-run-scaled refractory magnitude lever + Leg-C
    scaffold_train_rule_bias_head), keeping the 460g within-ARM around-closure
    occupancy-delta C2 DV but gating non-vacuity on the NEW sd034_n_closure_commit_intent
    > 0 (NOT sd034_n_closure_coupled_elevations). Acceptance: closure-coupled commit-intent
    > 0 on >= 2/3 scored seeds (MECH-445 precondition) AND ON within-arm post-closure
    occupancy < pre-closure by >= DECOMMIT_MIN_DROP_FRAC on >= 2/3 seeds (MECH-446 scored);
    the five readiness gates self-route substrate_not_ready_requeue when unmet (never a
    false weakens). MECH-261 stays non_contributory unless the automatic mode-conditioned
    detector fires (n_automatic_fires > 0 -- the Leg-A hook bypasses mode-conditioning).
    claim_ids=[MECH-446] (scored) + MECH-445 (coupling-engagement non-vacuity precondition).
    substrate_queue commitment-closure-control-plane ready STAYS false until 460h scores a
    contributory PASS.
  Design doc: REE_assembly/docs/architecture/sd_034_governance_closure_operator.md
    (refractory-independent commit-intent counter amend section). Autopsy:
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460g_2026-06-19.{md,json}.
    Decomposition: REE_assembly/evidence/planning/claim_synthesis_SD-034-closure_2026-06-19.md.
  See SD-034 (parent + Legs A/B 2026-06-12 + Leg C 2026-06-16 + beta-engagement 2026-06-17
    + de-commit-magnitude 2026-06-19), MECH-445 (closure->beta coupling engagement -- the
    child this counter certifies), MECH-446 (de-commit-authority magnitude -- the scored
    child the lever serves; cannot zero this certifier now), MECH-090 (BetaGate -- the latch
    whose held-elevated / refractory-blocked gate suppressed the old counter),
    sd034_n_closure_coupled_elevations (the 460f counter now measuring refractory-surviving
    elevations only), V3-EXQ-460g (the FAIL this amend addresses), V3-EXQ-460h (validation),
    MECH-261 (mode-conditioning; protect the stable claim), MECH-094 (N/A).

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
  MECH-282 (LPB interoceptive routing) -- IMPLEMENTED 2026-05-21.
    Module: ree_core/regulators/lpb_interoceptive_routing.py (LPBInteroceptiveRouter).
    Config: REEConfig.use_lpb_interoceptive_routing (bool, default False),
      lpb_intero_z_dim (16), lpb_drive_weight, lpb_resource_weight.
    Data flow: harm_obs resource slice zeroed before HarmEncoder -> z_harm (external);
      drive_level + harm_obs_a resource EMA -> z_harm_intero (non-trainable broadcast).
    SD-037 coupling: when both flags on, BroadcastOverrideRegulator.tick uses
      lpb_split_recruitment (intero drives override; external drives PAG freeze proxy).
    Backward compatible: flag OFF -> agent.lpb_router is None; z_harm_intero None.
    Validation experiment: V3-EXQ-600 queued (3-arm substrate diagnostic).
    See MECH-282, SD-037 design doc section MECH-282.

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
  Validation experiment: V3-EXQ-483b PASS substrate-readiness
    (2026-05-08; override_mean>0.30, PAG release ratio 1.875x).
    V3-EXQ-483/483a FAIL behavioural (approach_commit=0 all arms).
    Tier-1 evidence: V3-EXQ-483c queued (GAP-4 fishtank factorial).
    MECH-282 not in SD-037 landing scope; MECH-286 landed separately 2026-05-21.
  Design doc: REE_assembly/docs/architecture/sd_037_broadcast_override_regulator.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_orexin_kinetics/
  See SD-037, SD-036, MECH-279, SD-012, SD-032a, MECH-261, MECH-094.

## SD-037 consumer-cascade (MECH-281 motor-coupling axis amend, 2026-05-30)
- SD-037 / MECH-281 consumer-cascade -- IMPLEMENTED 2026-05-30.
  Amend session (NOT a fresh SD landing) triggered by V3-EXQ-483d FAIL
  (2026-05-29) substrate-ceiling diagnosis: with the GoalState seeding +
  PAG freeze-gate consumers already wired (2026-04-25) but the
  SalienceCoordinator slot dormant in the validation env (no PAG-engaging
  threat) and PFC/BLA/CeA/beta-gate sites unwired, override_signal had
  nowhere to land where it would move goal_norm_peak against the MECH-295
  bridge baseline. Autopsy artifact: evidence/planning/failure_autopsy_V3-EXQ-483d_2026-05-30.{md,json}
  Section 8 (implement-substrate amend SD-037 entry).
  Four additional consumer sites wired:
    (i) LateralPFCAnalog (SD-033a) -- update() gains override_signal +
      override_eta_gain kwargs; eff_eta scaled by (1 + override_eta_gain *
      override_signal). Orexin-recruited state accelerates rule_state EMA.
      Module: ree_core/pfc/lateral_pfc_analog.py.
    (ii) BLAAnalog (SD-035) -- tick() gains override_signal +
      override_encoding_gain kwargs; final encoding_gain (after inverted-U
      + post-event window) scaled by (1 + override_encoding_gain *
      override_signal). Roozendaal 2011 anchor: orexin -> NE / amygdala
      enhanced LTP. Module: ree_core/amygdala/bla.py.
    (iii) CeAAnalog (SD-035) -- tick() gains override_signal +
      override_amplitude_gain kwargs; both mode_prior and fast_prime
      scaled by (1 + override_amplitude_gain * override_signal),
      re-clipped to mode_prior_log_odds_max so CeA still cannot over-rule
      cortex via the amplified path. Module: ree_core/amygdala/cea.py.
    (iv) BetaGate / agent.py urgency_interrupt path (MECH-090) --
      urgency_threshold scaled by max(0.0, 1 - override_beta_interrupt_gain
      * override_signal). Orexin -> escape-from-freeze on the motor side,
      parallel to the existing PAG alpha_override path on the freeze-gate
      side. Lower threshold under recruited state -> committed motor
      program more readily aborted on z_harm_a / z_harm_un urgency.
      Module: ree_core/agent.py select_action() MECH-091 block.
  Already-wired (NOT touched this pass): PAG freeze-gate alpha_override,
  SalienceCoordinator update_signal("override_signal", ...), GoalState
  effective_drive amplification. All landed in 2026-04-25 SD-037 pass.
  Config (REEConfig + REEConfig.from_dims): 4 new scalar gain knobs,
  all defaults 0.0 (bit-identical OFF):
    override_pfc_eta_gain          0.0 (LateralPFCAnalog eff_eta multiplier)
    override_bla_encoding_gain     0.0 (BLAAnalog encoding_gain multiplier)
    override_cea_amplitude_gain    0.0 (CeAAnalog mode_prior + fast_prime multiplier)
    override_beta_interrupt_gain   0.0 (urgency_interrupt_threshold attenuator)
  Each gain requires its parent substrate's master flag to also be True
  (use_lateral_pfc_analog / use_amygdala_analog+use_bla_analog /
  use_amygdala_analog+use_cea_analog) plus use_broadcast_override=True.
  The urgency-interrupt path is always available (no separate master).
  Default 0.0 = scalar (1 + 0 * override_signal) = 1.0 -> existing knob
  untouched -> bit-identical to pre-MECH-281 amend.
  Backward compatible: all four gains default 0.0; broadcast_override
  module + four consumer modules all instantiate to None / no-op blocks
  by default; 556/556 contracts + 7/7 preflight PASS with master OFF
  (regression-clean 2026-05-30; was 543 + 13 new MECH-281 contracts).
  Activation smoke 2026-05-30 (50 ticks drive=0.9 / harm=0.6 ->
  override_signal=0.768): BLA encoding_gain 2.5 -> 5.0 (2x at gain=1);
  CeA mode_prior 0.2 -> 0.4, fast_prime 0.15 -> 0.3 (2x each at gain=1);
  LateralPFC rule_state delta 0.0956 -> 0.2867 (3x at gain=2).
  MECH-094: each consumer site preserves the existing simulation_mode
  argument. BLA / CeA simulation_mode=True short-circuit to zeroed
  output BEFORE the override modulation block runs. LateralPFC has no
  inline simulation_mode argument; replay-driven invocation is gated
  upstream by MECH-319 simulation-mode-rule-gate (covered separately).
  Beta-gate urgency_interrupt fires only on waking action selection so
  MECH-094 is N/A.
  Phased training: N/A. All four modulations are scalar multipliers on
  pure-arithmetic regulators / decision branches. No learned parameters,
  no gradient flow, no encoder heads. The 0.0-default + bit-identical OFF
  pattern matches SD-035 / MECH-279 / MECH-313 / MECH-314 / MECH-320 /
  MECH-341.
  Validation experiment: V3-EXQ-483e queued (4-arm successor under 483
  lineage). claim_ids=[SD-037, MECH-280, MECH-281]. Re-runs 483d ARM
  config with use_salience_coordinator=True + all four consumer-cascade
  gains>0 + PAG-engaging env via SD-036+MECH-279 freeze-engaging
  substrate reuse (V3-EXQ-475-style config: use_gabaergic_decay=True +
  use_pag_freeze_gate=True). Acceptance criteria match 483d but
  inverted: C3_lift_vs_baseline (goal_norm_peak delta vs baseline)
  should lift across 3/3 seeds (vs 1/3 on 483d); action_counts should
  diverge across broadcast_override axis within each seed (vs
  bit-identical on 483d).
  Contract tests: tests/contracts/test_sd_037_consumer_cascade.py
  (13 contracts: C1 + C1b backward-compat / C2-C2b LateralPFC eta
  modulation correctness / C3-C3b BLA encoding_gain modulation /
  C4-C4c CeA mode_prior + fast_prime modulation + cap bound /
  C5-C5b BLA + CeA simulation_mode skip / C6-C6b agent integration +
  override_signal advancement under drive+harm).
  See SD-037 (parent claim), MECH-280 + MECH-281 (sleep-state and
  motor-coupling axes; consumer-cascade lands the motor-coupling axis),
  SD-033a (LateralPFC consumer), SD-035 (BLA + CeA consumers), MECH-090
  (beta-gate urgency_interrupt consumer; orthogonal to the 2026-05-28
  R-c readiness conjunction landing), MECH-279 (PAG freeze-gate sibling
  consumer; already wired), SD-032a (SalienceCoordinator sibling
  consumer; already wired but dormant absent use_salience_coordinator=True),
  SD-036 + MECH-279 (PAG-engaging env substrate reused by V3-EXQ-483e),
  MECH-094 (call-site scoping via existing simulation_mode arguments),
  MECH-295 (liking-bridge baseline that 483d showed dominates effective_drive
  in the absence of consumer cascade), V3-EXQ-483d (substrate-ceiling
  FAIL that triggered this amend pass), V3-EXQ-483e (validation experiment).
  Plan-of-record: REE_assembly/evidence/planning/substrate_queue.json
  SD-037 entry (implementation_log appended with consumer-cascade
  landing record + next_step pointing at V3-EXQ-483e).

## MECH-286: Override-Gated Sleep Onset (2026-05-21)
- MECH-286: sleep.override_gated_state_transition -- IMPLEMENTED 2026-05-21.
  Module: ree_core/sleep/sleep_onset_gate.py (evaluate_sleep_onset_permit,
  SleepOnsetGateConfig). Wake-stability axis of SD-037: the same override_signal
  that gates drive->z_goal seeding also gates wake->offline transition in
  SleepLoopManager._run_cycle (before run_sleep_cycle / cycle_index advance).
  Joint permit (all required):
    override_signal < mech286_theta_sleep_permit
    AND max(MECH-284 region staleness snapshot) > mech286_theta_sleep_recruit
    AND z_harm_a.norm() < mech286_threat_tonic_threshold
  When blocked: episodes_since_sleep counter reset, cycle_index unchanged,
  last_metrics carry mech286_* diagnostics with mech286_sleep_permitted=0.
  Config: REEConfig.use_mech286_sleep_onset_gate (bool, default False).
    mech286_theta_sleep_permit (0.5), mech286_theta_sleep_recruit (0.3),
    mech286_threat_tonic_threshold (0.4).
  Backward compatible: flag OFF preserves deterministic K-episode sleep firing.
  Hyperarousal lesion test requires use_broadcast_override=True;
  staleness leg requires use_staleness_accumulator=True.
  MECH-094: gate evaluated only at episode boundary via notify_episode_end
  (waking state), not during replay/simulation ticks.
  No trainable parameters. No phased training.
  Validation experiment: V3-EXQ-599 queued (3-arm substrate diagnostic).
  See SD-037, MECH-284, MECH-285, MECH-272, MECH-281 (sibling motor axis).

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

## SD-016 Path 3: Feedforward cue->slot tagger (2026-06-05)
- SD-016 Path 3: e1.cue_slot_tagger -- IMPLEMENTED 2026-06-05.
  Module: ree_core/predictors/e1_deep.py (E1DeepPredictor.__init__ +
  extract_cue_context), ree_core/utils/config.py (E1Config + from_dims).
  Root cause it addresses (V3-EXQ-418i, the recommended div-weight sweep at
  1.0/2.0/5.0): Path 1 (auxiliary diversification loss) is "insufficient
  regardless of weight; the attention bottleneck is categorically in query
  selectivity, not slot orthogonality." The z_world-only q.k attention in
  extract_cue_context (e1_deep.py) is pinned at the uniform ln(num_slots)
  saddle because key_proj(memory) with memory init 0.01 yields near-identical
  keys -> softmax stays uniform -> the softmax Jacobian at uniform is a flat
  saddle that the cue_terrain_proj terrain_loss gradient cannot escape.
  THE FIX: a fresh feedforward MLP cue_slot_tagger (Linear(world_dim, hidden)
  -> ReLU -> Linear(hidden, num_slots)) replaces ONLY the slot-SELECTION
  scores; the slot-CONTENT path (value_proj -> output_proj -> cue_context)
  and both downstream projections (cue_action_proj retaining the 449a z_world
  concat band-aid; cue_terrain_proj) are untouched. A random MLP produces
  non-uniform logits from step 0, so it sits OFF the saddle and the existing
  terrain_loss gradient flows back into it and shapes contextual selectivity.
  No new supervised target is invented -- the tagger is a better-conditioned
  replacement for the saddle-stuck attention; the gradient source is the same
  terrain_loss that already trains cue_terrain_proj.
  Config (E1Config + REEConfig.from_dims; all no-op defaults):
    sd016_cue_slot_tagger (bool, default False) -- master switch (requires
      sd016_enabled=True; when False the legacy q.k attention branch runs
      verbatim, bit-identical).
    sd016_cue_slot_tagger_hidden (int, default 32) -- tagger MLP hidden width.
    sd016_cue_slot_tagger_temperature (float, default 1.0) -- softmax temp on
      the tagger logits (selectivity-sharpness knob; sweepable at eval without
      retraining).
  Data flow (Path 3 ON): z_world -> cue_slot_tagger -> slot_logits[num_slots]
    -> softmax(logits/temperature) [REPLACES bmm(q,k)/scale softmax] ->
    weights @ value_proj(memory) -> output_proj -> cue_context (unchanged) ->
    cue_action_proj([cue_context, z_world]) + cue_terrain_proj(cue_context).
  Read-only diagnostic: extract_cue_context caches the last selection
    distribution on E1DeepPredictor._last_cue_slot_weights [batch, num_slots]
    so a validation experiment can measure selection entropy (the V3-EXQ-418i
    bottleneck metric: uniform == ln(16) ~ 2.773).
  Backward compatible: sd016_cue_slot_tagger=False by default ->
    cue_slot_tagger is None and extract_cue_context takes the legacy attention
    branch (verified bit-identical: OFF selection entropy == ln(16) exactly).
    7/7 preflight + predictor-subsystem contracts PASS; 5/5 new contracts in
    tests/contracts/test_sd016_cue_slot_tagger.py PASS.
  Smoke (2026-06-05): OFF entropy 2.7726 (== ln16, on the saddle); ON entropy
    2.7407 at init (off the saddle) with per-context std 0.0098; terrain_loss
    backprops into the tagger (grad-sum > 0); under terrain SGD the tagger
    drives selection entropy toward 0 and fits the terrain target. (A toy with
    cleanly-separable clusters does NOT discriminate Path 3 from legacy; the
    real env -- where V3-EXQ-418i measured the legacy attention stuck at ~2.76
    across training -- is the discriminator. The validation EXQ runs there.)
  Phased training: NOT required -- the tagger trains jointly with cue_terrain_proj
    on the same terrain_loss (compatible objective, same gradient path).
    Experiments enabling it include terrain_loss in their E1 loop (the
    established SD-016 pattern, EXQ-182/187a/194).
  MECH-094: N/A -- extract_cue_context is a waking E1 query (_e1_tick); no
    replay/simulation write surface.
  HONEST SCOPE: Path 3 restores RETRIEVAL (cue_context) selectivity. Full
    behavioural action_bias_div >= 0.05 propagation ALSO depends on
    cue_action_proj, whose gradient path is the separate SD-055 differentiable-
    CEM / ARC-065 concern. The validation measures action_bias_div as a
    secondary/diagnostic, gated on SD-055.
  Validation experiment: V3-EXQ substrate-readiness diagnostic (claim_ids=[];
    OFF vs ON ablation; PRIMARY acceptance = mean selection entropy < 2.5 vs
    the pinned ln(16)=2.773 with the tagger ON; SECONDARY diagnostic =
    cue_context per-channel std + safe-vs-dangerous action_bias_div). Queued
    via /queue-experiment.
  Design doc: REE_assembly/docs/architecture/sd_016_writepath_v3_diversification_loss.md
    (Path 3 section, appended 2026-06-05).
  See SD-016, MECH-150 / MECH-151 / MECH-152, ARC-041, V3-EXQ-418i (the
    div-weight sweep that exhausted Path 1 + named the query-selectivity
    bottleneck), V3-EXQ-449a/449b (the constant-output fix + the residual
    contextual-selectivity gap this addresses), SD-016 Path 1 (exhausted
    auxiliary-diversification approach above), SD-055 (differentiable CEM;
    the cue_action_proj behavioural-propagation half), EXP-0155 (the original
    cue_action_proj forward-path diagnostic), MECH-094 (N/A -- waking query).

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
  Substrate validation: V3-EXQ-601 PASS 2026-05-21 (staleness lookup at default
  0.4/0.5 thresholds; supersedes smoke-only V3-EXQ-490b C1 path). Q-040
  behavioural factorial (V3-EXQ-490/490c cohort) remains deferred to StepHarness
  + MECH-307 substrate; C2/C3 not proven by 601.
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
  Validation experiment: V3-EXQ-601 PASS 2026-05-21 (MECH-269b-followup-A
  severance at default 0.4/0.5 thresholds; evidence_direction=supports).
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

## MECH-339 C1 Composite Cue + Outshining Gate (2026-05-19)
- MECH-339: hippocampal.composite_retrieval_cue_outshining_gate -- IMPLEMENTED 2026-05-19.
  Module: ree_core/hippocampal/ghost_goal_bank.py (GhostGoalBank.rank,
  _outshine_gate, _context_salience_for_anchor).
  Constraint 1 of the retrieval-cue reframe (ghost_goal_search.md
  Section 0.2; claims.yaml MECH-339 / ARC-078). The landed MECH-292 bank
  matched the retrieval cue by z_goal cosine only and ignored the SD-039
  payload fields the match does not use. MECH-339 adds a composite cue: a
  context channel built from the stored-but-match-unused payload, combined
  by an OUTSHINING gate so a strong direct goal_match suppresses the
  context channel (Smith & Vela 2001) rather than it being summed in with
  fixed weight.
  Smallest substrate step: the context channel is sourced from
  payload.arousal_tag only -- the one SD-039 field that is both already
  stored and entirely unused by the bank. last_vs is deferred (already
  consumed by the recoverability channel; reuse double-counts) and a
  `cause` tag is deferred (not present in the implemented
  AnchorGoalPayload -- design-sketch only; an SD-039 payload extension,
  out of scope here).
  Data flow: anchor.goal_payload.arousal_tag
    -> context_salience = 1 - exp(-arousal_tag / arousal_scale)  in [0,1)
    -> gate = clip_[0,1]((outshine_pivot - goal_match) / outshine_pivot)
    -> context_term = context_weight * gate * context_salience
    -> ghost_priority += context_term (overall form stays an additive sum
       of independently gateable channels; Constraint 2 unaffected -- the
       outshining is a within-channel multiplier, not a product across
       wanting/goal_match). New components/component_sums key "context"
       present only when the master switch is on.
  Config: GhostGoalBankConfig.use_composite_cue_outshining (default False;
    set True to enable), context_weight (default 0.0), outshine_pivot
    (default 0.5), arousal_scale (default 1.0). No rank() signature
    change, no anchor_set/agent-loop wiring change, no new payload field.
  Backward compatible: with the master switch off (or context_weight 0.0)
    the bank is bit-identical to pre-MECH-339 -- no "context" key in
    components/diagnostics, priorities unchanged. Verified: inline smoke
    (4-key components + bit-identical priority off; gate 0 at/above pivot,
    monotone decreasing; context term exactly 0.0 for an anchor with a
    strong direct match, > 0 for a weak-match high-arousal anchor;
    components sum == ghost_priority) and v3_exq_496 MECH-292 validation
    --dry-run PASS with the new config fields present.
  Biological basis: encoding specificity is real but moderate and is
    outshone by a strong direct cue (Smith & Vela 2001 meta-analysis;
    Tulving & Thomson 1973); arousal_tag is the LaBar & Cabeza 2006
    per-trace arousal tag.
  Phased training required: no (pure arithmetic; no trainable params).
  MECH-094: not applicable -- read-only over payloads whose provenance was
    set at SD-039 population time (sense() passes simulation_mode=False);
    the bank has no write path.
  Validation experiment: V3-EXQ-594 queued (diagnostic, priority 10;
    experiments/v3_exq_594_mech339_composite_cue_outshining_validation.py;
    4 sub-tests = the 4 MECH-339 falsifiable predictions; smoke PASS 4/4).
  See MECH-339 (this claim), ARC-078 (parent: cue-addressed retrieval
  system), MECH-292 (the bank whose goal_match this composes), SD-039
  (supplies arousal_tag), MECH-230 (z_goal = direct-cue content).

## MECH-340 Persistence / Efficacy Gate (2026-05-21)
- MECH-340: hippocampal.persistence_efficacy_gate -- IMPLEMENTED 2026-05-21.
  Module: ree_core/hippocampal/ghost_goal_bank.py (PersistenceAppraisal,
  _persistence_license, rank exclusion). ARC-079 / Q-053 front-runner:
  persistence of an entry as an active MECH-293 re-probe target is gated;
  disengagement is the default when license < persistence_floor.
  Data flow: optional PersistenceAppraisal passed into
  GhostGoalBank.rank() / HippocampalModule.rank_ghost_goals() /
  _propose_ghost_seeded():
    license = clip_[0,1](control_efficacy * (1 - goal_unattainability))
    exclude anchor when license < persistence_floor (SD-039 trace preserved)
  Config: GhostGoalBankConfig.use_persistence_efficacy_gate (default False),
    persistence_floor (0.05),
    persistence_default_when_appraisal_missing (1.0).
  Q-053 agent wiring (2026-05-21): REEAgent._compute_persistence_appraisal()
    maps prior hippocampal completion + E3 commitment -> control_efficacy;
    1 - goal_proximity -> goal_unattainability (one-shot; not staleness /
    failure). persistence_appraisal_compute.py + HippocampalConfig block;
    threaded through propose_trajectories / MECH-293 ghost branch.
    Reengagement-coupled disengagement STATE still deferred.
  Backward compatible: gate off -> rank() ignores appraisal, bit-identical.
  MECH-094: read-only; no write path.
  Validation: V3-EXQ-607 queued (diagnostic, priority 10;
    experiments/v3_exq_607_mech340_persistence_efficacy_gate_validation.py;
    5 sub-tests; contracts 4/4 + dry-run PASS).
  See MECH-340, ARC-079, ARC-078, MECH-292, MECH-293, Q-053.

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

## SD-049: Multi-Resource Heterogeneity (Phase 1 substrate, 2026-05-03)
- SD-049 Phase 1: environment.multi_resource_heterogeneity -- IMPLEMENTED 2026-05-03
  (env-only Phase 1; encoder rebuild + SD-032 consumer cascade deferred to
  Phase 2, see Phase 2 follow-on note below).
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorld /
  CausalGridWorldV2). Three additions to the env, gated by master switch
  multi_resource_heterogeneity_enabled (default False, bit-identical OFF).
  Substrate-roadmap H-priority #2; lit-anchored at sd_049 lit_conf=0.898
  (Berridge 2018 + Smith & Berridge 2007 + Kidd & Hayden 2015 + Shutts/Spelke
  2009 + Matthews/Tye 2016).
  Three additions:
    Addition 1 (multi-identity resources): num_resources cells split across
      n_resource_types qualitatively distinct types (default 3:
      food / water / novelty) per resource_type_distribution (default uniform).
      Each cell carries an identity tag stored in self._resource_type_grid
      (0 = no resource; type_idx + 1 elsewhere). Per-type resource lists in
      self._resources_by_type[type_idx]. Per-type proximity fields in
      self._resource_field_by_type[type_idx], computed in
      _compute_proximity_fields() when master is on. Per-type 5x5 field
      views appended to world_state in _get_observation_dict(); world_obs_dim
      grows by n_resource_types * 25 (250 -> 325 default 3-type ARM_2;
      250 -> 375 5-type ARM_3 overshoot). Per-type benefit profiles via
      resource_type_benefit_curves: "sigmoidal_saturating" / "sharp_saturation" /
      "novelty_decay" -- the novelty_decay curve attenuates contact benefit
      by per-cell familiarity, producing the structurally-distinct
      non-homeostatic benefit signal the wanting/liking dissociation
      requires (MECH-229 cohort).
    Addition 2 (per-axis homeostatic drive): per_axis_drive[n_axes] vector
      tracked alongside legacy agent_energy. Each per-tick depletion rate
      from per_axis_drive_decay; restoration on contact applied to the
      matching axis only (per the type-axis 1:1 mapping). When
      per_axis_drive_enabled, agent_energy collapses to
      1.0 - combiner(per_axis_drive) (combiner default "max", also
      "mean" / "sum") so all legacy SD-032 consumers (AIC / PCC / pACC /
      dACC / salience / override / MECH-295 bridge) continue to read
      obs_body[3] without modification. New observable:
      obs_dict["per_axis_drive"] for new experiments.
    Addition 3 (curriculum hook): resource_introduction_schedule:
      Dict[str, int] gates per-type spawn availability by self._global_step
      (cross-episode counter that survives reset()). Defaults to None ->
      all types available from step 0 (existing-experiment-equivalent
      behaviour even when master is on). Gate is checked at every reset()
      against current _global_step.
  Config (CausalGridWorld __init__ kwargs, env-only -- not surfaced through
  REEConfig.from_dims, matching SD-022 / SD-023 / SD-029 / SD-047 / SD-048
  precedent for env-only SDs):
    multi_resource_heterogeneity_enabled (bool, default False) -- master switch.
    n_resource_types (int, default 3).
    resource_type_names (tuple, default ("food","water","novelty")).
    resource_type_drive_axes (tuple, default ("hunger","thirst","curiosity")).
    resource_type_benefit_curves (tuple, default ("sigmoidal_saturating",
      "sharp_saturation","novelty_decay")).
    resource_type_distribution (tuple, default uniform via None).
    resource_type_benefit_amplitudes (tuple, default uniform 1.0 via None).
    per_axis_drive_enabled (bool, default False) -- per-axis vector observable
      and legacy-energy collapse.
    per_axis_drive_decay (tuple, default (0.001, 0.0015, 0.0005)).
    per_axis_drive_combiner (str, default "max"; also "mean" / "sum").
    novelty_familiarity_increment (float, default 0.2) -- per-contact
      per-cell familiarity bump.
    novelty_familiarity_recovery (float, default 0.0) -- 0 = permanent
      familiarity within episode; >0 = slow recovery toward 0.
    resource_introduction_schedule (Optional[Dict], default None).
  Defaults (truncated / padded for n_resource_types > 3) ensure the 5-type
  ARM_3 overshoot configuration runs without code changes (extra types get
  generic "type_<i>" / "axis_<i>" / "sigmoidal_saturating" entries).
  Per-resource-type bit-identical OFF: setting an entry to 0.0 in
  resource_type_distribution drops that type without code change (recovers
  ARM_1 from ARM_2 in the validation 4-arm sweep).
  Data flow:
    reset() -> distribute num_resources across active (curriculum-permitted)
      types by weighted draw -> populate _resources_by_type / _resource_type_grid
      -> _compute_proximity_fields rebuilds legacy + per-type fields ->
      world_state cat includes per-type 5x5 patches ->
      obs_dict["resource_field_view_<name>"] for each type +
      obs_dict["per_axis_drive"] + obs_dict["resource_type_at_agent"].
    step() resource consumption -> identify type from
      _resource_type_grid[new_x, new_y] -> apply type-specific benefit_curve
      with per-cell familiarity for novelty_decay -> remove cell from
      per-type list and clear tag -> apply axis restoration on matching
      axis -> increment _novelty_familiarity[cell] -> recompute proximity
      fields.
    step() post-move -> per_axis_drive[i] += per_axis_drive_decay[i]
      (gated on master + per_axis_drive_enabled) -> agent_energy =
      1 - combiner(per_axis_drive) -> _global_step += 1.
  info dict tags (always present, 0 / False when disabled):
    multi_resource_heterogeneity_enabled, sd049_n_resource_types,
    sd049_per_axis_drive_enabled, sd049_per_axis_drive_max,
    sd049_per_axis_drive_mean, sd049_n_resource_contacts_total,
    sd049_n_active_resources_by_type, sd049_global_step,
    sd049_resource_type_at_agent.
  Backward compatible: master switch False by default; all per-type state
  initialized to empty/zero; world_obs_dim returns the legacy value;
  obs_dict surfaces no SD-049 keys; agent_energy follows legacy path.
  RNG draws gated inside `if multi_resource_heterogeneity_enabled:` so seed
  sequences for existing experiments are bit-identical when disabled.
  Bit-identical OFF guarantee verified 2026-05-03 (50-tick parity check
  across explicit-False vs default + V3-EXQ-513 C0 acceptance criterion).
  Activation smoke (2026-05-03):
    50-tick no-resource depletion test: per_axis_drive evolves linearly per
      configured rates (0.5 / 0.25 / 0.0 at 50 ticks * (0.01, 0.005, 0.0)
      decay). agent_energy collapses correctly to 1 - max_drive.
    200-tick 3-type random-policy: per-type spawn counts ~uniform across
      types (5/3/4 with 12 cells, weighted-uniform draw); contacts distribute
      across types; per-axis drive evolves with per-tick decay - per-contact
      restoration; novelty familiarity grows on contacted cells.
    Curriculum hook: water introduced at step 1000 -- spawn count [6, 0, 6]
      at episode 0; spawn count [4, 6, 2] at episode 1 after 1001 ticks
      advance _global_step.
    ARM_3 5-type overshoot: world_obs_dim 250 -> 375 (250 + 5*25); spawn
      distributes across 5 types per uniform distribution.
  Implementation choices (deviations / clarifications from SD doc):
    - Flat kwargs on CausalGridWorld.__init__ rather than nested dataclasses
      (MultiResourceHeterogeneityConfig / ResourceTypeConfig / PerAxisDriveConfig).
      Matches SD-022 / SD-023 / SD-029 / SD-047 / SD-048 precedent for env-only
      SDs. Nested dataclasses are not used elsewhere for env params.
    - Per-axis drive vector is parallel to legacy agent_energy, NOT replacing
      it. agent_energy collapses via configurable combiner so legacy SD-032
      consumers continue to read obs_body[3] unchanged. The per-axis vector
      is observable through obs_dict["per_axis_drive"] for new experiments
      and the deferred Phase 2 encoder upgrade. This is the lighter-cascade
      Phase 1 deviation from the SD doc's "drive_weight scalar must become
      a per-axis vector" -- the cascade through every SD-032 consumer is
      instead deferred to Phase 2 for clean phased validation.
    - Per-resource-type bit-identical OFF preserved: setting an entry to 0.0
      in resource_type_distribution drops the type without code change
      (per the SD doc requirement). All-zero distribution falls back to
      uniform to avoid spawn-zero pathology (use master switch to disable
      SD-049 entirely).
    - Type-axis 1:1 mapping: resource type i restores axis i. Future MECH
      may decouple via a learned mapping (registerable post-validation).
    - Encoder side NOT modified: the existing ResourceEncoder (SD-015)
      consumes the larger world_obs (325 ARM_2 / 375 ARM_3) without code
      changes -- it sees the per-type field views as additional dimensions
      in world_obs. Identity discrimination on z_resource is the Phase 2
      validation target; Phase 1 measures only substrate readiness.
  No trainable parameters. Pure env-side state + arithmetic. No phased
  training needed for Phase 1 (env-only).
  MECH-094: not applicable (env observation stream, not replay / simulation
    content). reset_to() scripted-eval bypass leaves SD-049 OFF (matches
    SD-047 / SD-048 precedent: scripted-eval comparator harnesses
    intentionally bypass enrichment substrates for clean tagging).
  Phase 2 follow-on (REGISTERED, not implemented): the validation
    experiment for SD-049 requires a z_resource encoder upgrade to
    discriminate identities (one-hot identity slot or learned embedding
    on the larger world_obs) and the SD-032 consumer cascade to read
    per_axis_drive directly rather than the collapsed obs_body[3].
    This is registered as a Phase 2 substrate task in
    REE_assembly/evidence/planning/substrate_queue.json (SD-049 entry,
    phase_2 follow-on note); the behavioural validation lands as
    V3-EXQ-514 (goal_resource_r lift + identity-recovery probe + wanting
    != liking trajectory dissociation) post-Phase 2.
  Validation experiment: V3-EXQ-513 queued (substrate readiness
    diagnostic; 4-arm sweep ARM_0/ARM_1/ARM_2/ARM_3 + curriculum check;
    13 acceptance criteria covering bit-identical OFF, per-type
    spawn / contact, per-axis drive evolution, novelty familiarity
    growth, agent_energy divergence from legacy path, ARM_3 overshoot
    obs_dim shape, curriculum gate / release. Dry-run smoke 2026-05-03:
    13/13 PASS at 30 ticks * 1 seed; full run at 200 ticks * 3 seeds
    expected ~5 min on Mac).
  Design doc: REE_assembly/docs/architecture/sd_049_multi_resource_heterogeneity.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_sd_049/
  See SD-049, SD-012 (per-axis drive extension cascade -- triggers
    pending_substrate_reconfirmation on SD-012-emergent invariants per
    invariant-types governance rule when Phase 2 lands), SD-015
    (z_resource encoder upstream substrate this enables in Phase 2),
    SD-005 (z_world routing must hold under multiple resource identities),
    MECH-229 (wanting/liking dissociation -- primary behavioural test
    post-Phase 2), MECH-230 (z_goal latent structure -- non-trivial
    multi-modal structure post-Phase 2), MECH-117 (existing
    wanting/liking trajectory dissociation -- non-degenerate evidence
    post-Phase 2), MECH-216 (schema generalisation across identity-
    distinct cues), Q-030 (6-cell z_resource x z_world routing sweep --
    well-posed post-Phase 2), ARC-030 (approach-avoidance symmetry
    across goal types), ARC-032 (theta-routing across goal identities),
    SD-047 (parallel substrate enrichment, file-coordinated, Phase 1
    landed first), SD-048 (parallel substrate enrichment, file-coordinated).

## SD-049 Phase 2: Hybrid Identity-Aware z_resource Encoder (2026-05-04)
- SD-049 Phase 2: encoder.identity_aware_z_resource (Option C hybrid) --
  IMPLEMENTED 2026-05-04. Encoder-side Phase 2 follow-on to the SD-049
  Phase 1 substrate (env-only) landed 2026-05-03. Lands the architectural
  choice from the 2026-05-04 lit-pull verdict
  (evidence/literature/targeted_review_sd_049_encoder_identity_expansion/
  verdict.md, Option C hybrid at confidence 0.78). Biology-anchored to
  Ballesta-Padoa-Schioppa 2019 OFC labeled-line + Quiroga 2005 sparse
  readouts + Schapiro 2017 hybrid CLS bi-pathway architecture.
  Modules:
    ree_core/latent/stack.py (ResourceEncoder + LatentState)
    ree_core/utils/config.py (LatentStackConfig)
    ree_core/agent.py (compute_resource_identity_loss)
    ree_core/environment/causal_grid_world.py (info dict supervision target)
  Encoder shape (Option C hybrid):
    Shared trunk MLP encoder (Linear -> ReLU -> Linear) producing 32-dim
    z_resource (Schapiro 2017 monosynaptic-analog distributed substrate;
    Schapiro 2016 statistical-learning pathway).
    Identity-classifier head (Linear(z_resource_dim, n_resource_types))
    supervised by cross-entropy on obs_dict["sd049_consumed_type_tag_this_tick"]
    when SD-049 multi_resource_heterogeneity is on (Ballesta-Padoa-Schioppa
    labeled lines; Quiroga sparse readouts; Schapiro 2017 trisynaptic-
    analog episode pattern separation).
    Magnitude head (resource_prox_head) reusing existing SD-018 pattern
    unchanged.
  IMPORTANT design choice: z_resource OUTPUT shape unchanged at
    z_resource_dim (32). Classifier head shapes the trunk via training
    pressure (anti-collapse mitigation per Levi 2021 + identity
    discriminability supervision per the verdict). identity_logits is
    exposed as a SEPARATE LatentState field (Optional[Tensor]) for the
    cross-entropy loss; downstream consumers (GoalState seeding etc.)
    continue to read z_resource unchanged. This avoids the
    GoalState.z_goal seeding dim-mismatch that pure-concat would create
    (z_goal_dim=32; concat would grow z_resource to 32 + n_types).
  Config (LatentStackConfig):
    use_identity_classifier (bool, default False) -- master switch.
    identity_classifier_n_types (int, default 3) -- output dim of head.
    Both surfaced via direct attribute assignment on cfg.latent.* (not
    via REEConfig.from_dims kwargs -- matches existing SD-015
    use_resource_encoder pattern).
  Phased training (per verdict.md):
    P0: enable use_identity_classifier=True. Joint backprop of identity
        cross-entropy + resource_prox MSE + downstream task losses
        through the trunk. Classifier head provides anti-collapse pull
        (Levi 2021 mitigation in spirit) AND identity-discriminability
        supervision per the biology-anchored verdict.
    P1: freeze identity_head.requires_grad_(False). Continue trunk
        training under E1/E3/downstream losses. Trunk embedding develops
        similarity structure beyond what classifier supervision alone
        provides (Schapiro 2016 distributed substrate development).
    P2: evaluate identity-recovery (linear probe on z_resource) AND
        goal_resource_r AND per-axis drive evolution per V3-EXQ-514
        acceptance criteria.
  Data flow:
    ResourceEncoder.forward(world_obs) -> (z_resource [batch, 32],
      resource_prox_pred_r [batch, 1], identity_logits [batch, n_types]
      OR None when classifier disabled).
    LatentStack.encode() -> LatentState with z_resource +
      resource_prox_pred_r + identity_logits all populated when classifier
      enabled.
    agent.compute_resource_identity_loss(target_type, latent_state) ->
      cross-entropy scalar; zero when classifier disabled, target=0
      (no resource at agent), or out-of-range.
  Env supervision target plumbing (SD-049 Phase 2 env fix):
    causal_grid_world.py step() caches consumed-type tag BEFORE clearing
    the cell tag in the resource-consumption branch. Surfaced as
    info["sd049_consumed_type_tag_this_tick"] (1..n_types when consumption
    fired this tick; 0 otherwise). The supervision target for the
    identity classifier in the V3-EXQ-514 training loop. Always present
    (0 when SD-049 OFF or no consumption this tick).
  Backward compatible: use_identity_classifier=False by default;
    ResourceEncoder.identity_head=None; identity_logits=None on every
    LatentState; compute_resource_identity_loss returns 0; all existing
    SD-015 experiments (use_resource_encoder=True with classifier OFF)
    behave bit-identically. 7/7 preflight + 184/184 contracts PASS with
    classifier OFF (regression suite green, 2026-05-04).
  Activation smoke (2026-05-04):
    Default ResourceEncoder + use_identity_classifier=False -> identity_logits
    is None on LatentState; compute_resource_identity_loss returns 0.0.
    use_identity_classifier=True + n_types=3 -> identity_logits shape
    [1, 3]; cross-entropy with target=type_0 returns ~ln(3)~1.10 at random
    init; backward succeeds; target=0 (no-resource) returns 0.0 (skip).
    Env consumed-type plumbing: with multi_resource_heterogeneity_enabled=True,
    info["sd049_consumed_type_tag_this_tick"] correctly reports type_idx+1
    (1/2/3) on resource-contact ticks; 0 on non-contact ticks.
  No phased training needed for the SUBSTRATE itself (encoder change is a
  shape change with backward-compat defaults). Phased training is REQUIRED
  for V3-EXQ-514 (it is the validation methodology, not a substrate
  requirement).
  MECH-094: not applicable (waking observation stream encoder; not replay
    / simulation content). The classifier supervision target is the
    waking-stream env consumption tag.
  ML/AI engineering notes (Layer 7, per implement-substrate skill rule):
    - Class-collapse hazard (Levi et al. 2021 ICCV): mitigated by phased
      training. P0 supervised classifier head provides anti-collapse pull
      on the trunk; P1 freezes the head; P2 evaluates. No additional
      anti-collapse machinery needed at the substrate level.
    - Joint training fragility (EXQ-166b/c/d historical): the phased
      protocol decouples head learning from encoder learning across the
      P0/P1 boundary. Standard mitigation pattern.
    - z_resource shape preservation rather than concat: chosen over the
      verdict.md Option-1 concat instantiation because GoalState.z_goal
      is fixed at goal_dim=32 and would silently break under z_resource =
      concat(trunk, identity_softmax) = 32 + n_types. The "single-output-
      with-supervision" instantiation (verdict.md Option 2) is what
      most ML papers do; the classifier head is training-only in the
      sense that downstream consumers read just z_resource, but the
      identity_logits are still computed at every tick (cheap) and
      exposed for the loss term.
  Validation experiment: V3-EXQ-514 queued -- 4-arm sweep + phased
    training + identity-recovery linear probe + goal_resource_r
    measurement + per-axis drive evolution check. 10 acceptance criteria
    (C0/C1a/C1b/C2a/C2b/C2c/C2d/C2e/C3a/C3b). PASS = SD-049 Phase 2
    validated; SD-015 promotable; SD-049 v3_pending may be cleared. FAIL
    routes to the 6-row interpretation grid in verdict.md, including the
    Woo/Spelke-style substrate-ceiling falsifier branch (joint failure
    across ARM_2 AND ARM_3 routes MECH-229 to substrate_conditional with
    V4-1 multi-agent ecology dependency). estimated_minutes=90.
  Design doc: REE_assembly/docs/architecture/sd_049_multi_resource_heterogeneity.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_sd_049_encoder_identity_expansion/
  See SD-049 (parent claim, Phase 1 substrate landed 2026-05-03), SD-015
    (z_resource encoder upstream substrate this extends), MECH-229
    (wanting/liking dissociation primary test), MECH-230 (z_goal latent
    structure non-trivial), Q-030 (6-cell z_resource x z_world routing),
    SD-012 (per-axis drive extension; SD-012-emergent invariants
    pending_substrate_reconfirmation flag deferred to next governance
    cycle), MECH-094 (call-site scoping; not applicable).
  Deferred Phase 3 follow-on (substrate_queue.json SD-049-PHASE-3 entry):
    SD-032 consumer cascade migrating AIC, PCC, pACC, dACC, salience,
    override, MECH-295 from reading goal_state._last_drive_level
    (collapsed scalar) to reading obs_dict["per_axis_drive"] directly.
    Action-trigger: V3-EXQ-514 failure on SD-032-mediated mode-switching
    pathway (which is not predicted by the current encoder-driven
    failure modes per verdict.md).

## SD-049-PHASE-2 drive-coupling amend: kappa-scale + standing differential depletion (V3-EXQ-514r, MECH-436) (2026-06-17)
- SD-049-PHASE-2 drive-coupling amend -- IMPLEMENTED 2026-06-17 (substrate;
  MECH-436 stays candidate / substrate_ceiling / pending_retest_after_substrate
  until the V3-EXQ-514r-successor retest scores -- this PROMOTES NOTHING). Routed by
  the confirmed failure_autopsy_V3-EXQ-514r_2026-06-17 (overshoot disambiguator:
  drive CAN carve most_wanted at magnitude 5.0 -- flips on 2/5 guard-passing seeds,
  non-zero on all 5 -- so 514q's natural delta=0.0 is a drive-MAGNITUDE / environment
  artifact, NOT a MECH-436 falsification). The autopsy greenlit two coupled levers,
  kappa LOAD-BEARING; the 514q "DO NOT build until 514r resolves" hold is LIFTED.
  ROOT CAUSE (two structural facts the autopsy established): (1) wanting[k] =
  base_value[k] * (1 + kappa * per_axis_drive[k]) with a fixed kappa=2.0 and an in-run
  per-axis spread ~0.006 contributes ~0.012 to the multiplier -- swamped by real object
  base_value gaps that exceed 0.5 on seeds 45/46/47 (so even a magnitude-5.0 overshoot
  cannot flip them at the current kappa); (2) the P2 foraging ecology FULLY restores the
  contacted axis to 0 on consumption (causal_grid_world.py sigmoidal_saturating restore =
  cur_drive), and the WL bank is scored AROUND consumption events -- exactly where full
  restoration zeroes the just-consumed axis -- so the natural spread at scoring time is
  equalised to ~0.006.
  THE FIX (two no-op-default levers; bit-identical OFF):
    Lever (a) KAPPA SCALE (load-bearing) -- ree_core/goal.py
      (GoalConfig.incentive_drive_kappa_scale, default 1.0) + ree_core/utils/config.py
      (from_dims signature + goal_keys + local_goal_vals). IncentiveTokenBank.wanting()
      now uses effective kappa = incentive_drive_kappa_weight * incentive_drive_kappa_scale
      (getattr fallback 1.0). At 1.0 (default) wanting() is byte-identical; >1 lets a
      realistic per-axis drive spread compete with the real object base_value landscape.
    Lever (b) STANDING DIFFERENTIAL DEPLETION -- ree_core/environment/causal_grid_world.py
      (CausalGridWorldV2.per_axis_restoration_fraction, env-only kwarg, default 1.0, NOT
      surfaced through from_dims per the SD-049/SD-047/SD-048 env-only precedent). The
      curve-determined restore on contact is scaled by per_axis_restoration_fraction:
      restore = cur_drive * curve_mult * per_axis_restoration_fraction. At 1.0 (default)
      the contacted axis fully restores to 0 (bit-identical); <1.0 leaves STANDING per-axis
      drive on restored axes so a frequently-but-incompletely-restored axis carries residual
      drive at the WL scoring moment. Paired with the EXISTING divergent per_axis_drive_decay
      tuple (env-only kwarg the retest sets), this produces a persistent argmax-relevant
      per-axis drive SPREAD instead of the equalised ~0.006. (per_axis_restoration_fraction
      clamps to [0, 1].)
  Backward compatible: both default no-op. preflight 7/7 + the changed-subsystem
  contracts (goal + environment) PASS; 7 new contracts in
  tests/contracts/test_sd049_phase2_drive_coupling.py (C1 kappa_scale default bit-identical
  wanting / C2 kappa_scale>1 flips a most_wanted argmax the unscaled kappa cannot --
  the load-bearing behaviour / C3 restoration_fraction default fully restores [legacy] /
  C4 restoration_fraction<1 leaves standing drive on a real consumption step / C5 from_dims
  surfaces incentive_drive_kappa_scale onto cfg.goal / C6 restoration clamp + kappa_scale
  getattr fallback). Activation smoke 2026-06-17: env default==explicit-1.0 bit-identical
  over a scripted walk; partial restoration (0.4) raises the standing tail per-axis spread
  vs full restore.
  Phased training: N/A (no encoder head; scalar arithmetic on already-encoded latents +
  env state). MECH-094: N/A (env observation stream + waking IncentiveTokenBank arithmetic;
  no replay/memory write surface). Evidence-staleness (Step 8.5): NOT triggered -- both
  levers no-op-default; every existing experiment uses the defaults (kappa_scale 1.0,
  full restoration), so no dependent claim's measured mechanism changed. KEEP all evidence.
  GUARDRAILS HELD: MECH-229 leg (a) wanting!=liking object-bound dissociation (V3-EXQ-514o
  PASS 0.80) UNTOUCHED. No claims.yaml status flips (substrate-only); MECH-436 stays
  candidate / substrate_ceiling / pending_retest_after_substrate until the retest scores.
  Validation experiment: V3-EXQ-514s (the 514r-successor MECH-436 retest) -- re-run the
  514r overshoot + OFF/bank-disabled + recalibrated-argmax-relevance-readiness controls
  on the kappa-scaled + partial-restoration substrate. Pre-registered promotion target:
  natural drive-coupled delta mean(WL_drive - WL_nodrive) >= max(k*pstdev(delta), 0.15)
  on >=2/3 seeds converts MECH-436 substrate_ceiling -> supports. Non-vacuity preconditions
  (overshoot still flips a constructed gap, OFF/bank-disabled floor ~0, bank populated,
  recalibrated argmax-relevance) self-route substrate_not_ready_requeue, NEVER a false
  weakens. claim_ids=[MECH-436] (re-evaluated from scratch: 514r tagged MECH-229 broadly,
  but the run exercises the MECH-436 drive-coupling leg only). Queued via /queue-experiment.
  Design doc: REE_assembly/docs/architecture/sd_049_multi_resource_heterogeneity.md
  (drive-coupling amend section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-514r_2026-06-17.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json (SD-049-PHASE-2).
  See SD-049 (parent; Phase 1 2026-05-03 + Phase 2 encoder 2026-05-04), MECH-436
  (drive-state-modulated wanting; the leg this enriches the retest for), MECH-229 leg (a)
  (untouched), SD-057 IncentiveTokenBank (the wanting() host lever (a) scales), V3-EXQ-514q
  (natural delta=0.0), V3-EXQ-514r (the overshoot disambiguator this amend is greenlit by),
  V3-EXQ-514s (validation/retest), MECH-094 (N/A).

## SD-049-PHASE-2 drive-coupling amend: BOUNDED kappa raise + deeper standing spread (V3-EXQ-514s, MECH-436) (2026-06-19)
- SD-049-PHASE-2 drive-coupling BOUNDED-RAISE amend -- IMPLEMENTED 2026-06-19 (substrate
  calibration; MECH-436 stays candidate / substrate_ceiling / pending_retest_after_substrate
  until V3-EXQ-514t scores -- this PROMOTES NOTHING). Routed by the confirmed
  failure_autopsy_V3-EXQ-514s_2026-06-18 (user-adjudicated 2026-06-18: clean self-route,
  substrate-ceiling PARTIALLY LIFTED). NO new flag, NO ree_core default change -- the two
  no-op-default levers landed 2026-06-17 (incentive_drive_kappa_scale, per_axis_restoration_fraction)
  already support the raise; this pass CALIBRATES the bounded operating point + adds the two
  invariant-guard contracts + retests.
  ROOT (514s, all five non-vacuity preconditions MET): lever (b) standing-differential-
  depletion WORKED (enriched_spread_met 1.0, mean_drive_spread_max 0.211 vs 514q's
  equalised ~0.006, per-seed 0.19-0.23) -- the env amend is done. But lever (a) kappa_scale=6.0
  is still SHORT: the WL dissociation metric is argmax-flip-gated, and on 3/5 seeds the real
  object base_value gaps exceed what kappa=6.0 x standing-spread~0.2 flips at natural magnitude
  (natural delta bimodal [-0.067, 0.176, 0.0, 0.211, 0.0]; mean 0.064 < margin 0.15; 2/5 seeds
  clear individually; overshoot mag 5.0 still flips 4/5). Residual shortfall LOCALIZED to the
  kappa lever.
  THE CALIBRATION (BOUNDED so drive does not swamp base_value; uses the autopsy's "and/or"):
    (a) RAISE incentive_drive_kappa_scale 6.0 -> 12.0 (a decisive 2x raise; effective kappa =
        incentive_drive_kappa_weight(2.0) * 12.0 = 24). BOUNDED check: at eff_kappa=24 a MODERATE
        base_value gap (1.0 vs 0.6) flips under a realistic standing spread (~0.25) while a
        CLEARLY-LARGER 10x gap (1.0 vs 0.10) does NOT -- drive carves near-ties without
        overriding a decisive base_value gap (a sated agent still wants the clearly-better
        object). Kappa-alone could not close the gap in a bounded way (the natural spread ~0.2
        is ~25x smaller than the mag-5.0 overshoot, so flipping the hardest seeds by kappa
        alone needs an UNbounded kappa) -- hence pairing with (b).
    (b) DEEPEN the standing spread: per_axis_restoration_fraction 0.3 -> 0.15 on the WL-scoring
        env only (training ecology stays 603n-canonical so survival/foraging competence is
        preserved). 0.15 leaves ~0.43 standing drive on a just-consumed axis (vs ~0.35 at 0.3),
        so the divergent-decay per-axis differences are larger at the WL scoring moment ->
        bigger argmax-relevant spread, less kappa needed per flip.
  INVARIANTS that survive the raise (the two new contracts):
    C7 OFF-floor-hard-zero: kappa multiplies ONLY the per-axis drive term, so at ZERO drive
       wanting()==base_value for ANY kappa_scale -- raising kappa cannot manufacture a
       drive-induced dissociation absent a drive signal (the substrate guarantee behind the
       experiment's bank-disabled wl_off_floor_fraction ~ 0 control). Hard-zero stays hard-zero.
    C8 bounded / wanting!=liking-from-base_value (MECH-229 leg-(a) intact): at kappa_scale=12.0 a
       clearly-larger 10x base_value gap is NOT flipped by a realistic standing spread -> drive
       does not dominate base_value. Brackets C2 (a moderate gap IS flipped).
  Backward compatible: NO ree_core default changed (kappa_scale default 1.0, restoration default
    1.0 both unchanged -> bit-identical OFF for every existing experiment). 9/9
    tests/contracts/test_sd049_phase2_drive_coupling.py (7 prior + new C7/C8) + goal/environment
    subsystem contracts (14/14) + 7/7 preflight PASS. The bounded kappa math was verified against
    the live wanting() formula (kappa_scale=12 flips a 1.0-vs-0.6 gap, holds a 1.0-vs-0.10 gap)
    and the restoration lever against the live env (0.15 -> 0.426 standing vs 0.3 -> 0.351).
  Phased training: N/A (no encoder head; scalar arithmetic on already-encoded latents + env
    state). MECH-094: N/A (env observation stream + waking IncentiveTokenBank arithmetic; no
    replay/memory write surface). Evidence-staleness (Step 8.5): NOT triggered -- no-op-default
    levers unchanged; every existing experiment uses the defaults, so no dependent claim's
    measured mechanism changed. KEEP all evidence.
  GUARDRAILS HELD: MECH-229 leg (a) wanting!=liking object-bound dissociation (V3-EXQ-514o PASS
    0.80) UNTOUCHED. No claims.yaml status flips (MECH-436 stays candidate / substrate_ceiling /
    pending_retest_after_substrate until the retest scores).
  Validation experiment: V3-EXQ-514t (the 514s-successor MECH-436 retest; supersedes V3-EXQ-514s)
    -- re-runs the 514r/514s overshoot + OFF/bank-disabled + recalibrated-argmax-relevance +
    enriched-spread controls on the kappa_scale=12.0 + restoration=0.15 substrate. Pre-registered
    promotion target unchanged: natural drive-coupled delta mean(WL_drive - WL_nodrive) >=
    max(k*pstdev(delta), 0.15) on >= 2/3 seeds -> MECH-436 substrate_ceiling -> supports. The five
    non-vacuity preconditions self-route substrate_not_ready_requeue, NEVER a false weakens; a
    preconditions-met FAIL with overshoot still flipping routes another bounded retune (514u),
    NOT a falsification. claim_ids=[MECH-436]; machine any. Queued via /queue-experiment.
  Design doc: REE_assembly/docs/architecture/sd_049_multi_resource_heterogeneity.md
    (bounded-raise amend section). Autopsy:
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-514s_2026-06-18.{md,json}.
    Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json (SD-049-PHASE-2).
  See SD-049-PHASE-2 drive-coupling amend (the 2026-06-17 lever landing this calibrates),
    MECH-436 (drive-state-modulated wanting; the leg the retest exercises), MECH-229 leg (a)
    (untouched), SD-057 IncentiveTokenBank (the wanting() host lever (a) scales), V3-EXQ-514s
    (the FAIL this amend addresses; lever b worked, kappa short), V3-EXQ-514t (validation/retest),
    MECH-094 (N/A).

## SD-050: Suffering-Derivative Comparator (2026-05-04)
- SD-050 / MECH-302: relief.suffering_derivative_comparator -- IMPLEMENTED 2026-05-04.
  Module: ree_core/comparator/suffering_derivative_comparator.py
    (SufferingDerivativeComparator -- non-trainable, no nn.Module inheritance).
  Config: REEConfig.use_suffering_derivative_comparator (default False; set True to enable).
    Additional params: suffering_window_length (default 5), suffering_drop_threshold
    (default 0.10), suffering_min_initial_norm (default 0.05),
    relief_completion_weight (default 1.0). All defaults are no-op.
  Data flow: sense() -> suffering_comparator.tick(z_harm_a.norm(), sim_mode)
    -> _relief_completion_event: bool (ephemeral flag, not in LatentState)
    -> select_action() -> beta_gate.release() + committed_step_idx reset
    + residue_field.update_valence(z_world, VALENCE_LIKING, relief_weight,
    hypothesis_tag=False) when valence_liking_enabled.
  Pipeline reuse: identical to MECH-057a goal-completion pipeline -- commitment
    release + MECH-094 categorical tag write -- triggered by suffering-stream
    descent rather than goal attainment. Architecturally adjacent to MECH-091
    urgency block in select_action(); opposite polarity.
  MECH-094 compliance: tick(simulation_mode=True) returns False without advancing
    the buffer (waking-path signal only; replay/DMN must not trigger events).
  Backward compatible: disabled by default; existing experiments unaffected;
    bit-identical OFF (no RNG draws; pure arithmetic).
  Phased training: not applicable (non-trainable).
  Validation experiments: V3-EXQ-515 PASS 2026-05-04 (comparator logic: 4-arm
    synthetic-norm unit diagnostic). V3-EXQ-516 queued (agent-loop integration:
    ARM_0 OFF backward-compat, ARM_1 event fires, ARM_2 valence write, ARM_3
    flat signal no false fires).
  See MECH-302, MECH-057a (commitment release pipeline reused), MECH-091
    (urgency block -- adjacent, opposite polarity), MECH-094 (simulation gate),
    SD-011 (z_harm_a source stream).

## SD-051: Conditioned Safety Store (2026-05-04)
- SD-051 / MECH-304: safety_prediction.cue_specific_conditioned_inhibition_substrate -- IMPLEMENTED 2026-05-04.
  Module: ree_core/safety/conditioned_safety_store.py (ConditionedSafetyStore -- non-trainable,
    no nn.Module inheritance; pure arithmetic). New package ree_core/safety/__init__.py.
  Config: REEConfig.use_conditioned_safety_store (default False; set True to enable).
    Additional params: safety_store_ema_alpha (default 0.1), safety_store_decay_rate (default 0.001),
    safety_store_min_norm (default 0.1), safety_store_threshold (default 0.5),
    safety_store_commitment_weight (default 1.0). All defaults are no-op.
  Data flow: sense() -> z_world [world_dim] -> conditioned_safety_store.update(z_world,
    event_fired=_relief_completion_event, sim_mode=hypothesis_tag) -> _conditioned_safety_signal: float
    -> select_action(): when _conditioned_safety_signal > threshold AND beta_gate elevated ->
    beta_gate.release() + optional VALENCE_LIKING write (MECH-094).
  Encoding pathway (dorsal striatum / dlPFC analog): EMA prototype of z_world at MECH-302 event ticks.
    Per-step decay toward zero (safety_store_decay_rate) provides forgetting without reinforcement.
  Expression pathway (IL->CeA analog): cosine similarity between current z_world and prototype ->
    sigmoid -> safety_prediction scalar -> commitment-release gate.
  MECH-094 compliance: update(sim_mode=True) returns 0.0 without advancing the prototype
    (waking-path signal only; replay/DMN ticks are silent).
  Backward compatible: disabled by default; conditioned_safety_store is None when disabled;
    sense() and select_action() skip all SD-051 blocks entirely; bit-identical OFF.
  Phased training: not applicable (non-trainable).
  V4-deferred: (1) Approach attractor toward safety-signaling cues (requires V4 multi-step
    planning infrastructure, MECH-163 V3 completion gate). (2) Contrastive cue-specific
    learning (requires trainable encoder head + phased training; V3 prototype may
    over-generalise in stable-environment settings). See v3_v4_transition_boundary.md.
  Validation experiment: V3-EXQ-519 queued 2026-05-04 (4-arm substrate-readiness diagnostic).
  See MECH-304 (the claim this SD implements), MECH-303 (sister contextual pathway),
    MECH-302 / SD-050 (teaching signal source), MECH-057a (commitment-release pipeline reused),
    MECH-094 (simulation gate), SD-011 (z_harm_a source for MECH-302 event).

## SD-052: Contextual Passive Safety Terrain (2026-05-04)
- SD-052 / MECH-303: safety_prediction.contextual_passive_substrate -- IMPLEMENTED 2026-05-04.
  Module: ree_core/residue/field.py (ResidueField -- extended with safety_terrain_rbf_field,
    accumulate_safety(), evaluate_safety()). No new file; extends existing ResidueField.
  Config: REEConfig.use_contextual_safety_terrain (bool, default False; set True to enable).
    ResidueConfig.safety_terrain_enabled (bool, default False; auto-set by from_dims when
    use_contextual_safety_terrain=True).
    Additional params: contextual_safety_accum_weight (float, default 0.01),
    contextual_safety_harm_threshold (float, default 0.05),
    contextual_safety_release_threshold (float, default 1.0). All defaults are no-op.
  Data flow: sense() -> z_harm_a.norm() < contextual_safety_harm_threshold AND
    hypothesis_tag=False -> residue_field.accumulate_safety(z_world, accum_weight) ->
    safety_terrain_rbf_field.add_residue() [incremental per-step].
    select_action() when beta_gate.is_elevated -> residue_field.evaluate_safety(z_world)
    -> if mean >= contextual_safety_release_threshold -> beta_gate.release() +
    _committed_step_idx=0 + _committed_anchor_keys=None.
  Architecture: follows same RBF pattern as benefit_terrain (ARC-030 / MECH-117).
    Separate safety_terrain_rbf_field -- no sharing with benefit_terrain or residue values.
    Biological analog: vmPFC + hippocampus (vHipp-to-PL). Slow/diffuse (0.01/step);
    contrasts with MECH-304 fast event-driven update.
  MECH-094 compliance: accumulate_safety(hypothesis_tag=True) returns immediately --
    safety accumulation is waking-only; simulation/replay ticks are silent.
  Backward compatible: disabled by default; safety_terrain_rbf_field not instantiated
    when disabled; all agent.py blocks guarded by getattr config check; bit-identical OFF.
  Phased training: not applicable (non-trainable accumulator).
  Validation experiment: V3-EXQ-520 queued 2026-05-04 (4-arm substrate-readiness diagnostic).
  See MECH-303 (the claim this SD implements), MECH-304 / SD-051 (cue-specific sister),
    MECH-302 / SD-050 (harm source monitored by harm_threshold gate), SD-011 (z_harm_a stream),
    ARC-007 (contextual encoding substrate), ARC-030 / MECH-117 (benefit terrain parallel),
    MECH-094 (simulation gate).

## SD-019a: harm_unpleasantness_channel (2026-05-04)
- SD-019a: harm_stream.immediate_affective_valence -- IMPLEMENTED 2026-05-04.
  Module: ree_core/agent.py (REEAgent.__init__, reset(), sense(), select_action()),
    ree_core/latent/stack.py (LatentState.z_harm_un), ree_core/utils/config.py
    (LatentStackConfig.use_harm_un, harm_un_ema_alpha).
  Config: LatentStackConfig.use_harm_un (bool, default False; set True to enable).
    LatentStackConfig.harm_un_ema_alpha (float, default 0.2; ~5-step rise to
    z_harm_s=1.0 at alpha=0.2).
  State: REEAgent._harm_un_ema (Optional[Tensor], same dim as z_harm_s); reset
    per episode; seeded at z_harm_s on first tick; None when feature is OFF.
  Data flow: sense() -> LatentStack.encode() -> new_latent.z_harm (z_harm_s) ->
    [if use_harm_un and not hypothesis_tag] EMA update: _harm_un_ema <-
    (1-alpha)*_harm_un_ema + alpha*z_harm -> new_latent.z_harm_un <-
    _harm_un_ema.clone() -> AIC urgency (aic_z_norm reads z_harm_un.norm() when
    use_harm_un; falls back to z_harm_a.norm() otherwise) + select_action() E3
    short-horizon urgency_weight and MECH-091 interrupt (redirect z_harm_a
    variable to z_harm_un when use_harm_un).
  Three-tier harm hierarchy: z_harm_s (fast, instantaneous) ->
    [EMA alpha=0.2] -> z_harm_un (medium, ~5-step rise) ->
    [MECH-219/SD-019b, not yet implemented] -> z_harm_a (slow, suffering).
  Controllability parity (Loffler 2018 key constraint): SD-021 descending
    modulation only multiplies new_latent.z_harm (z_harm_s). z_harm_un evolves
    exclusively via its own EMA rule and is NOT attenuated during commitment.
    Verified: UC3 in V3-EXQ-518 (norm_un unchanged during commitment, norm_s
    attenuated by 50%).
  AIC redirect (SD-032c): when use_harm_un=True, aic_z_norm reads
    z_harm_un.norm() so the AIC urgency computation tracks the medium-timescale
    unpleasantness channel, not the slow suffering accumulator.
  E3 redirect: z_harm_a variable in select_action() is shadowed by z_harm_un
    when use_harm_un=True and z_harm_un is not None. Affects urgency_weight
    computation and MECH-091 interrupt threshold. dACC (SD-032b), pACC
    (SD-032e) retain independent z_harm_a reads (NOT redirected -- they
    correctly consume the slow suffering accumulator).
  MECH-094 compliance: EMA update gated on hypothesis_tag=False; replay/
    simulation paths do not advance _harm_un_ema.
  Backward compatible: use_harm_un=False by default; z_harm_un=None on
    every LatentState; all integration sites no-op. 184/184 contracts PASS
    with flag OFF.
  Biological basis: Loffler et al. 2018 (Pain Reports) three-way dissociation
    (intensity / unpleasantness / suffering): controllability selectively reduces
    suffering (z_harm_a) without touching unpleasantness (z_harm_un). ACC /
    anterior insula medial-pathway affective-motivational component of pain
    (Price 2000 dual-pathway; Rainville 1997 ACC/S1 double dissociation).
  Phased training: not applicable (non-trainable EMA buffer; no gradient flow).
  Validation experiment: V3-EXQ-518 queued (4-arm diagnostic, 9 acceptance
    criteria UC0a-b/UC1a-d/UC2a-b/UC3; dry-run PASS 2026-05-04).
  See SD-019a, SD-019 (parent nonredundancy), SD-011 (z_harm_s source),
    SD-019b (SD-019a is the input to the MECH-219 hysteretic integrator),
    SD-021 (descending mod; controllability parity), SD-032c (AIC redirect
    consumer), MECH-091 (urgency interrupt; redirect consumer), MECH-094
    (EMA gate).

## SD-054: Reef Enrichment Substrate (2026-05-04)
- SD-054: environment.reef_enrichment_substrate -- IMPLEMENTED 2026-05-04.
  (Note: this entry was originally labelled SD-050 in error; SD-050 is the
  Suffering-Derivative Comparator per claims.yaml. Renamed to SD-054 on
  2026-05-08; SD-053 is informally reserved for a sustained-drive claim.)
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorldV2).
  Monostrategy-breaking behavioral-diversity substrate. Adds reef safe zones and
  food-attracted hazard drift to CausalGridWorldV2. Motivated by the observation
  that monomodal policy prevents balanced agent-vs-env event distributions (SD-029
  C2/C3 blocker). Creates two behavioral attractors -- "flee to reef" vs "forage"
  -- to break the single fixed route the agent otherwise exploits.
  Two mechanisms:
    Reef safe zones: circular patches (Manhattan radius) around corner-adjacent
      centers where hazards cannot enter and no food/hazards spawn. Agent CAN enter
      reef cells (attractor for safety-seeking). Static scent gradient: 5x5
      normalized Manhattan-decay kernel appended to world_state when reef_enabled=True.
      world_obs_dim: 250 -> 275 (+ 25 reef_field_view dimensions).
    Food-attracted hazards: during _drift_hazards(), with probability
      hazard_food_attraction, each hazard biases its random walk step toward the
      nearest food cell rather than sampling uniformly. Default 0.0 (bit-identical OFF).
      Makes foraging actively more dangerous; paired with reef safety creates a
      structurally richer strategy space.
  Config (CausalGridWorldV2 __init__):
    reef_enabled (bool, default False) -- master switch for reef zones.
    n_reef_patches (int, default 3) -- number of circular reef patches.
    reef_patch_radius (int, default 2) -- Manhattan radius of each patch.
    hazard_food_attraction (float, default 0.0) -- probability hazard steps
      toward nearest food. 0.0 = legacy random walk (bit-identical OFF).
  State: self._reef_cells: Set[Tuple[int,int]] (populated at reset());
    world_obs_dim property returns 275 when reef_enabled else 250.
  Observation: obs_dict["reef_field_view"] [25] static Gaussian-decay view of
    reef proximity; appended to world_state in _get_observation_dict().
  Backward compatible: reef_enabled=False and hazard_food_attraction=0.0 by default;
    _reef_cells is empty set; world_obs_dim=250; existing experiments unaffected.
  V3-EXQ-521 substrate readiness diagnostic PASS 7/7 criteria (2026-05-04):
    ARM_0 baseline obs_dim=250 reef_cells=0; ARM_1/ARM_2 obs_dim=275 reef_cells=33;
    0 reef violations in ARM_1/ARM_2 across 30 ep x 200 steps x 3 seeds;
    ARM_2 food_dist=2.057 vs baseline 3.790 (46% reduction); entropy not collapsed
    (ARM_2=4.049 >= 0.7 x baseline 4.557=3.190); agent reef_visits=1987 in ARM_1.
  Biological analog: coral reef refugia as behavioral partitioning substrate --
    distinct safe-zone vs. foraging patch microhabitats force context-dependent
    strategy selection rather than single-template exploitation.
  MECH-094: not applicable (env observation stream, not replay content).
  No trainable parameters. Pure env substrate.
  Validation experiment: V3-EXQ-521 PASS 2026-05-04 (substrate readiness diagnostic).
    V3-EXQ-522 (monostrategy-breaking behavioral diversity test) is the next experiment,
    gated on V3-EXQ-521 PASS.
  See SD-029 (self_attribution.comparator_z_harm_s -- SD-054 substrate unblocks C2/C3
    behavioral diversity measurement), MECH-256 (SD-029 successor), SD-023 (env
    gradient texture -- parallel substrate enrichment pattern).

## SD-054 bipartite layout extension (2026-05-11)
- SD-054 bipartite layout: environment.reef_bipartite_spawn_partition -- IMPLEMENTED 2026-05-11.
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorldV2).
  Extends SD-054 with a geometric bipartite spawn structure so reef-vs-forage
  trajectories require categorically-different first-action argmaxes by
  construction. Resolves the upstream CEM-candidate-distinguishability
  bottleneck surfaced by V3-EXQ-543b diagnose-errors (TASK_CLAIMS session
  diagnose-v3-exq-543c-2026-05-11T0635Z; arc_062_rule_apprehension_plan.md
  decision-log 2026-05-11 option 3a).
  Three new __init__ kwargs (all default to legacy SD-054 behavior; env-only,
  NOT surfaced through REEConfig.from_dims per SD-022 / SD-023 / SD-029 /
  SD-047 / SD-048 / SD-054 precedent):
    reef_bipartite_layout (bool, default False) -- master switch.
    reef_bipartite_axis (str, default "horizontal") -- "horizontal" -> reef
      bottom rows, food top rows. "vertical" -> reef right cols, food left
      cols. Validation: raises ValueError on construction if axis is neither.
    reef_bipartite_agent_band_radius (int, default 1) -- half-width of the
      agent spawn band measured from the midline (inclusive). 0 = midline
      only; 1 = midline +/- 1 (3 rows/cols); 2 = midline +/- 2 (5 rows/cols).
  Geometry (axis="horizontal", radius=1, size=12 default):
    Reef half: rows in (midline + radius .. size - 2] = rows 8, 9, 10
    Agent band: rows in [midline - radius .. midline + radius] = rows 5, 6, 7
    Forage half: rows in [1 .. midline - radius) = rows 1, 2, 3, 4
    Reef patches placed along the bottom edge (row sz-3) with column centres
    evenly distributed across interior columns. Patches that would intersect
    the agent band or forage half are clipped via _is_in_reef_half guard.
  Reset partitioning: when reef_bipartite_layout=True, reset() calls the new
    _place_reef_patches_bipartite() (does NOT consume from `available`) and
    _build_bipartite_pools(available) which returns (agent_pool, forage_pool)
    as two disjoint subsets. Agent pops from agent_pool; hazards / resources /
    waypoints pop from forage_pool. Legacy mode aliases both pools to a single
    `available` list so the pre-existing single-pool pop-from-shared behavior
    is bit-identical.
  Fallback: if agent_pool would be empty (degenerate config, e.g. radius=0 on
    a size where the midline is mostly walls), widens the band by +1 radius
    iteratively until a valid cell exists; records the widen count in
    self._sd054_bipartite_band_widen_count (0 in legacy mode and successful
    bipartite resets).
  No new state in obs_dict, no change to world_obs_dim (still 275 with
    reef_enabled=True), no new training target. Pure env substrate refinement.
  Why the extension was needed: legacy SD-054 places reef patches in fixed
    corners but agent / hazards / food spawn at uniformly-random positions in
    the remaining cells. Per-episode reef / food geometry is randomized; the
    mean optimal policy across episodes converges to a single "head toward
    nearest food" template (a direction-following heuristic that works
    regardless of episode-specific reef-food geometry). The agent-side
    consequence (verified 2026-05-11 by direct numerical probe in
    TASK_CLAIMS session diagnose-v3-exq-543c-2026-05-11T0635Z): CEM proposer
    at init produces 8 candidates all sharing argmax-first-action=3 with
    continuous-action spread ~1e-4 and post-action z_world spread ~1e-5,
    leaving the ARC-062 head reading near-identical inputs and unable to
    discriminate. Bipartite layout forces reef-bound and forage-bound
    trajectories to have categorically opposite first-action argmaxes
    (action 1 = down toward reef, action 0 = up toward food on horizontal
    axis), restoring the structural condition under which a well-trained
    ARC-062 substrate WOULD produce per-candidate first-action argmax
    diversity at probe states sampled from typical training rollouts.
  Diagnostic counter: env.info dict surfacing TBD with the validation EXQ;
    self._sd054_bipartite_band_widen_count is exposed as an attribute for
    direct read.
  Backward compatible: all three new kwargs default to legacy SD-054
    behavior; agent_pool and forage_pool are aliased to a single `available`
    list; pop()-from-shared-pool semantics preserved; bit-identical to
    pre-extension HEAD when reef_bipartite_layout=False.
  Smoke (2026-05-11): backward-compat with legacy reef kwargs reproduces 33
    reef cells spanning rows 1-10, world_obs_dim 275, bipartite_band_widen
    count 0. Activation with reef_bipartite_layout=True, axis=horizontal,
    radius=1 across seeds 0/1/2: agent always spawns in rows [5,6,7]; reef
    cells always in rows [8,9,10] (28 cells with edge-row centres clipped
    by reef-half predicate); hazards + resources always in rows [1,2,3,4];
    band_widen_count 0 in all seeds. Vertical axis with radius=2 verified
    independently (agent col in [4..8], reef cols [9,10], hazards cols
    [1,2]). Bad-axis construction raises ValueError as expected.
  Biological analog: coral-reef refugia in marine systems force categorically
    opposite swim-direction choices (toward reef-shelter vs toward open-water
    foraging grounds) because the two microhabitats are spatially anti-
    correlated, not interleaved. The legacy SD-054 corner-placement is a
    weaker geometric expression of the same claim; the bipartite extension
    is the sharper instantiation.
  No trainable parameters. Pure env substrate. No phased training. No
    MECH-094 interaction (env observation stream, not replay content).
  Validation experiment: V3-EXQ-548 substrate-readiness diagnostic to be
    queued via /queue-experiment immediately following this entry. Will
    measure CEM-candidate first-action argmax entropy at probe states with
    bipartite ON vs OFF across 3 seeds + structural-only (no full P1 falsifier
    rerun in this pass; that decision waits on V3-EXQ-548 PASS).
  See SD-054 (parent claim, unchanged in semantics), MECH-309 (logical-
    necessity claim for which the substrate enables a sharper falsifier),
    ARC-062 (downstream consumer; this extension creates the structural
    conditions for ARC-062 GAP-B falsifier testability), MECH-269 (V_s
    primitive; orthogonal cluster but parallel substrate-readiness pattern),
    SD-023 / SD-047 / SD-048 / SD-049 (parallel env-only substrate-
    enrichment kwargs precedent for not surfacing through REEConfig.from_dims).

## SD-022 scheduled-injection extension (MECH-302 unblock, 2026-05-30)
- SD-022 scheduled-injection: environment.scheduled_limb_damage_curriculum
  -- IMPLEMENTED 2026-05-30. Module: ree_core/environment/causal_grid_world.py
  (CausalGridWorldV2). Env-side curriculum that periodically injects damage
  directly into self.limb_damage independent of agent action or hazard
  contact, supplying detectable damage->heal trajectories so the MECH-302
  SufferingDerivativeComparator (SD-050) has reliable suffering signals
  regardless of a trained avoidance policy.
  Triggered by failure_autopsy_V3-EXQ-517b_2026-05-30 (REE_assembly/evidence/
  planning/): three FAIL discriminative-pair attempts (V3-EXQ-517 / 517a /
  517b, 2026-05-04 / 04 / 06) ruled out parameter tuning; 517a (window=30 /
  thresh=0.005 / min_norm=0.01, 150 steps/ep) got 0.33 events/seed; 517b
  (same params, 300 steps/ep) got 0.00 events/seed. Trained avoidance policy
  filters out hazard-contact -> heal trajectories the comparator needs.
  Architectural distinction (this entry vs SD-029 scheduled_external_hazard):
    SD-029 relocates a hazard adjacent to the agent -- still requires agent
      contact to inflict damage; depends on the agent stepping onto the
      relocated hazard for the SD-022 contact-driven damage_increment path
      to fire.
    SD-022 scheduled-injection bypasses contact entirely: damage is added
      directly to self.limb_damage via the curriculum gate, regardless of
      what action the agent takes or what cell it occupies. Mimics
      allostatic / externally-imposed tissue insult (clinical analog:
      scheduled chemo cycle, sleep deprivation pulse, chronic-disease flare).
    The two curricula are orthogonal axes: both can be enabled
      simultaneously without interaction.
  Five new __init__ kwargs on CausalGridWorldV2 (env-only -- NOT surfaced
  through REEConfig.from_dims; matches SD-022 / SD-023 / SD-029 / SD-047 /
  SD-048 / SD-049 / SD-054 precedent for env-only SDs):
    scheduled_limb_damage_enabled (bool, default False) -- master switch.
    scheduled_limb_damage_interval (int, default 50) -- period in steps
      between injection attempts.
    scheduled_limb_damage_prob (float, default 0.5) -- probability per attempt.
    scheduled_limb_damage_magnitude (float, default 0.4) -- damage added to
      selected limb on fire; clamped to [0, 1] by the same min(1.0, ...) bound
      the existing SD-022 contact-damage path uses.
    scheduled_limb_damage_limb_selection (str, default "random") -- "random"
      (one limb uniformly at random) or "all" (uniform across all 4 limbs).
  Defaults rationale: at magnitude=0.4 and SD-022 default heal_rate=0.002/step,
  one injection takes ~200 steps to heal -- comfortably within a 300-step
  episode. interval=50 x prob=0.5 yields ~3 injections per 300-step episode
  in expectation -> multiple detectable healing trajectories per seed-episode.
  Preconditions (constructor raises ValueError; loud-not-silent matching the
  MECH-269b / MECH-293 / SD-049 precedent):
    scheduled_limb_damage_enabled=True requires limb_damage_enabled=True
      (without the SD-022 substrate active, limb_damage stays at zero and the
      curriculum is a silent no-op rather than the user's intent).
    scheduled_limb_damage_limb_selection must be "random" or "all".
  Data flow (env.step()): agent move + SD-022 damage accumulation + healing
  -> [if scheduled_limb_damage_enabled and steps>0 and steps%interval==0
     and rng<prob] -> _inject_scheduled_limb_damage() -> selects limb(s)
     per scheme and adds magnitude with clamp -> increment counter, set info
     tags -> existing SD-029 scheduled_external_hazard block (UNCHANGED;
     orthogonal axis) -> existing SD-054 / SD-049 / SD-047 / SD-048 blocks
     (UNCHANGED).
  Info dict tags (always present; 0 / False when disabled):
    scheduled_limb_damage_enabled, scheduled_limb_damage_injected_this_step,
    scheduled_limb_damage_event_count, scheduled_limb_damage_last_limb_idx
    (-1 when not injected or all-mode), scheduled_limb_damage_last_magnitude
    (0.0 when not injected).
  Per-episode reset: counter + last_step + last_limb_idx + last_magnitude +
  injected_this_step all cleared in reset() and in the scripted-eval
  reset_to() bypass path (same pattern as SD-029 / SD-047 / SD-048 per-episode
  resets so scripted-eval comparator harnesses start from zero state).
  Backward compatible: all five kwargs default False / inert; RNG draws gated
  inside `if self.scheduled_limb_damage_enabled:` so existing experiment
  seeds remain bit-identical when disabled. 565/565 contracts + 7/7 preflight
  PASS with master OFF (regression-clean 2026-05-30; was 556 + 9 new MECH-302-
  curriculum contracts in tests/contracts/test_scheduled_limb_damage_curriculum.py).
  Activation smoke (2026-05-30): 5 fires across 67 steps at interval=10/
  prob=1.0/magnitude=0.4 (~1 fire per 13 steps, close to expected 1/10);
  bit-identical OFF vs default-constructed control across 30 steps; random-
  mode adds magnitude to exactly one limb; all-mode adds to all four;
  clamp at 1.0 holds across multiple 0.6 injections; per-episode reset clears
  state; info dict tags always present.
  Biological grounding: clinical relief-completion literature (Tanimoto &
    Heisenberg 2004 Drosophila relief-as-reinforcer; Roesch / Calu /
    Schoenbaum 2007 ventral-striatal relief firing at experimenter-imposed
    aversive-state offset; Andreatta 2012 Learn Mem fear/relief double
    dissociation; Navratilova 2012 PNAS pain-relief activates VTA-DA +
    NAc-shell DA + DA-antagonist-blockable place preference). All cited
    studies use experimenter-imposed discrete onset/offset suffering events
    that the animal cannot avoid -- the curriculum reproduces that
    prerequisite at the env layer.
  ML/AI engineering note: scheduled-perturbation curricula (Bengio 2009
    automated curriculum learning) carry a standard failure mode where
    schedule predictability creates a degenerate solution (agent learns to
    time the perturbation rather than handle its consequence). Mitigation
    is built in: stochastic gate (prob<1.0 default) + random limb selection
    ensures the agent cannot predict exact timing or location.
  MECH-094: not applicable (env observation stream, not replay / simulation
    content). Same scope decision as SD-022 / SD-029 / SD-047 / SD-048 /
    SD-049 / SD-054; env-side substrates do not interact with MECH-094.
  No trainable parameters. No phased training needed. ASCII-safe (no
    print() output added).
  Validation experiment: V3-EXQ-517c queued 2026-05-30 (priority 280; supersedes
    V3-EXQ-517b; same 2-arm discriminative-pair structure as 517a/b with
    curriculum knobs identical on both arms; only use_suffering_derivative_comparator
    and valence_liking_enabled differ between arms). Curriculum config:
    interval=50, prob=0.5, magnitude=0.4, limb_selection=random. Acceptance
    criteria unchanged from 517b: C1 ARM_A events>=3/seed; C2 ARM_A writes>=2/seed;
    C3 ARM_B events==0; C4 ARM_B writes==0; PASS_FRACTION_REQUIRED=2/3.
    PASS clears MECH-302 v3_pending gate AND lifts gate (c) for the MECH-304
    conditioned-inhibition experiment.
  See SD-022 (parent claim -- limb damage substrate this curriculum extends),
    SD-050 / MECH-302 (downstream comparator the curriculum unblocks),
    SD-051 / MECH-304 + SD-052 / MECH-303 (sister safety-prediction
    substrates also blocked by MECH-302 v3_pending),
    SD-029 (parallel curriculum on hazard-relocation axis; orthogonal),
    SD-023 / SD-047 / SD-048 / SD-049 / SD-054 (parallel env-only
    substrate-enrichment kwargs precedent for not surfacing through
    REEConfig.from_dims),
    failure_autopsy_V3-EXQ-517b_2026-05-30 (the autopsy that routed this
    implement-substrate amend session),
    MECH-094 (call-site scoping; not applicable -- env observation stream).

## SD-055: Differentiable CEM Selection Approximation (2026-05-15)
- SD-055: hippocampal.differentiable_cem_selection -- IMPLEMENTED 2026-05-15.
  Module: ree_core/hippocampal/module.py (post-elite refit block).
  Config: HippocampalConfig.use_differentiable_cem (bool, default False);
    HippocampalConfig.differentiable_cem_temperature (float, default 1.0);
    REEConfig.from_dims(use_differentiable_cem=..., differentiable_cem_temperature=...).
  Data flow: E2 rollouts score candidates -> softmax(-score/T) weights over ALL
    candidate ao sequences -> differentiable ao_mean (and ao_std) -> downstream
    HippocampalModule consumers. Legacy path (flag False): argsort elite fraction +
    indexed mean unchanged (bit-identical default).
  Motivation: EXP-0155 / EXQ-449 zero gradient through CEM argmax severs SD-016
    cue_action_proj learning (ARC-072 gap 2 diagnostic).
  Backward compatible: use_differentiable_cem=False by default; existing experiments
    unaffected. smoke_sd055_differentiable_cem.py 4/4 PASS (grad_max ~260 smoke,
    EXQ-568 grad_max=372).
  MECH-094: not applicable (waking CEM selection, not replay content).
  Phased training: not an encoder head; no P0/P1/P2 latent-target phasing required
    for the substrate switch itself. Behavioural experiments that train cue_action_proj
    should enable the flag explicitly.
  Validation experiment: V3-EXQ-568 PASS 20260515T204931Z (substrate-readiness,
    evidence_direction=non_contributory). Does NOT validate cue-conditioned behavioural
    divergence on goal-rich env.
  Design doc: REE_assembly/docs/architecture/sd_055_differentiable_cem_selection.md
  See SD-016, ARC-072, MECH-326, EXP-0155, developmental_bootstrapping_hippo_retrieval.md.


## ARC-062 Phase 1: Gated-Policy Heads + Context Discriminator (2026-05-09)
- ARC-062 (Phase 1, weak reading): rule_apprehension.gated_policy_heads --
  IMPLEMENTED 2026-05-09. Phase 1 of arc_062_rule_apprehension_plan.md
  (GAP-A). V3-tractable instantiation of the rule-apprehension architectural
  slot identified by MECH-309 (logical-necessity claim: trainers weight
  rules they do not invent; without a non-Bayesian rule-creator at the
  policy layer, gradient descent on a parametric policy collapses to the
  smoothest single regime good-enough across the whole state space).
  Module: ree_core/policy/gated_policy.py (GatedPolicy + GatedPolicyConfig
  + GatedPolicyOutput). New ree_core/policy/__init__.py package.
  Architecture (Phase 1, weak reading):
    Two scoring heads (head_0, head_1) sharing E3 candidate features [K,
      world_dim]. Each head: Linear(world_dim, head_hidden) -> ReLU ->
      Linear(head_hidden, 1) -> [K] scalar bias. Symmetry-broken init
      on the heads' last-Linear bias term (head_0 +offset, head_1
      -offset; default offset=0.05) so heads can differentiate from
      step 0 under any training pressure -- avoids the all-zero
      degenerate equilibrium where w*head_0 + (1-w)*head_1 collapses to
      a single value.
    Context discriminator: 3-stream input (z_world, z_self, z_harm_a)
      per Pull A SYNTHESIS verdict 1 (Miller & Cohen 2001 + Rigotti 2013
      + Mitchell 2016 macaque MD with insular cluster). Linear(world_dim
      + self_dim + harm_a_dim, disc_hidden=24) -> ReLU -> Linear(24, 1)
      -> sigmoid -> scalar w in [0, 1]. Discriminator weights scaled by
      disc_init_scale=0.1 at init so sigmoid output sits near 0.5 --
      avoids early head over-commitment before either head has
      differentiated.
    Gated bias: gated_score_bias = w * head_0(features) + (1 - w) *
      head_1(features), clamped to [-bias_scale, +bias_scale]
      (bias_scale=0.1; mirrors lateral_pfc_bias_scale so Phase 1
      magnitudes are comparable to existing PFC-side contributions).
  Pull A SYNTHESIS verdicts (resolved defaults; do NOT re-litigate):
    R1 multi-stream input (z_world, z_self, z_harm_a) -- single-stream
      z_world-only is the impoverished case, reserved for Phase 2
      ARM_1a/b/c input-ablation sub-arms.
    R2 N=2 heads at Phase 1 -- substrate-constrained by SD-054 reef-vs-
      forage two-mode partition. GatedPolicyConfig.n_heads != 2 raises
      ValueError on construction; multi-head extension is Phase 4 / GAP-E
      multi-strategy scaling probe.
    R3 score_bias level (option iii) -- engineering reasons dominate
      (SD-033a substrate is wired, gradient path through E3 score-
      aggregation is clean). FAIL chain at score_bias level routes
      discriminator to BG-side first then trajectory-proposal level
      then ARC-063 V4 strong reading.
  Config: REEConfig.use_gated_policy (bool, default False; bit-identical
  OFF). Sub-knobs (REEConfig + REEConfig.from_dims): gated_policy_n_heads
  (2), gated_policy_disc_hidden (24), gated_policy_disc_init_scale (0.1),
  gated_policy_head_hidden (32), gated_policy_bias_scale (0.1),
  gated_policy_head_init_bias_offset (0.05),
  gated_policy_use_first_action_onehot (False; see GAP-B below).
  Agent wiring (REEAgent.__init__): when use_gated_policy=True, instantiate
    GatedPolicy with (world_dim, self_dim, z_harm_a_dim) from
    config.latent. Phase 1 has NO connection to SD-033a LateralPFCAnalog
    -- that wiring is Phase 3 (closes commitment_closure GAP-1) per
    arc_062_rule_apprehension_plan.md. Per-episode reset() clears
    diagnostic counters (no persistent state to clear -- module is
    stateless across ticks).
  Data flow (REEAgent.select_action): immediately before the MECH-295
    block, compose gated_policy_score_bias additively into dacc_score_bias
    (parallel to the dACC / lateral_pfc / ofc / mech295 composition
    pattern). Per-candidate features = first-step z_world summary [K,
    world_dim] (reuses cand_world_summaries from lateral_pfc / ofc when
    they ran earlier this tick; builds fresh otherwise). simulation_mode
    parameter is False at this call site (waking action selection); the
    module's MECH-094 simulation_mode=True path (used by replay/DMN
    consumers if they ever call this module) returns (0.5, zeros[K])
    without advancing diagnostics.
  Backward compatible: use_gated_policy=False by default; agent.gated_policy
    is None and the entire select_action GatedPolicy block is skipped.
    249/249 preflight + contracts PASS (244 prior + 5 new) with master
    OFF. Bit-identical to baseline.
  Biological basis: Pull A SYNTHESIS (lit-pull A: lateral PFC rule-context
    modulation, 8 entries) -- Miller & Cohen 2001 PFC rule-as-bias
    foundational; Rigotti et al. 2013 mixed selectivity; Mitchell et al.
    2016 macaque MD network with insular cluster; Bongard & Nieder 2010
    PFC rule-coding units; Erez & Duncan 2015 MD adaptive coding;
    Capkova/Mansouri 2025 frontal lesion rule-value-learning dissociation.
    Pull A SYNTHESIS verdicts captured in
    REE_assembly/evidence/literature/targeted_review_arc_062_rule_apprehension/
    SYNTHESIS.md.
  MECH-094: simulation_mode argument on forward(); when True, returns
    (gating_weight=0.5, zeros[K], zeros[K], zeros[K]) and increments
    only the simulation-skip counter. Match SD-035 amygdala / MECH-279
    PAG simulation_mode pattern. Module has no internal state buffer
    (no EMA, no rule_state) -- stateless across ticks; reset() only
    clears diagnostic counters.
  Phased training: not required for Phase 1 substrate-readiness. The
    head + discriminator parameters develop training pressure naturally
    via E3 score-aggregation gradient when the validation experiment
    queues an end-to-end task. Phase 2 monomodal-collapse falsifier on
    SD-054 is the first behavioural training environment; Phase 1 only
    validates substrate wiring + architectural prerequisites for the
    Phase 2 falsifier. V3-EXQ-543f (GAP-B falsifier) requires phased
    training (P0 encoder warmup -> P1 frozen-encoder head training ->
    P2 eval).
  Validation experiment: V3-EXQ-542 5/5 PASS 2026-05-09T20:22:11Z (Mac
    runner; v3_exq_542_arc062_gated_policy_substrate_readiness_v3_*.json
    in REE_assembly/evidence/experiments/). Five sub-tests UC1-UC5:
    UC1 forward-pass instantiation; UC2 master-OFF no-op vs baseline E3;
    UC3 discriminator output varies with z_world (input sensitivity,
    threshold 0.001 in the Phase-1 init regime; substantial discriminator
    variation is a Phase-2 training signal); UC4 head differentiation
    under training pressure (>5x output divergence on held-out batch
    after 200 SGD steps); UC5 MECH-094 simulation_mode gate. Phase 2
    monomodal-collapse falsifier on SD-054 reef + hazard_food_attraction
    substrate is the next behavioural validation (queued as a separate
    session per the plan-of-record's six-phase sequencing -- do NOT
    bundle Phase 2 EXQ in the Phase 1 landing session).
  Contract tests: tests/contracts/test_gated_policy.py 6/6 PASS (C1
    default-off no-op; C2 backward-compat flag-on does not raise during
    construction or first sense() tick; C3 discriminator output in
    [0, 1] across 64 diverse latent states with bias_scale clamp
    respected; C4 heads' OUTPUTS diverge >5x on held-out batch after
    200 SGD steps under anti-symmetric loss; C5 simulation_mode=True
    returns zeros + increments skip counter only, subsequent waking
    call does not retroactively re-increment skip counter; C6
    use_first_action_onehot: correct head_in_dim, output shape, differs
    from base, sim-mode still zeros, _last_onehot_was_none diagnostic).
  Plan-of-record: REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md
    GAP-A status open -> done 2026-05-09; owner_exq=V3-EXQ-542; Phase 2
    GAP-B in-progress pending V3-EXQ-543f.
  GAP-B: ARC-062 head-input first-action one-hot augmentation (option 2)
    IMPLEMENTED 2026-05-17. Root cause from EXQ-543e autopsy: SP-CEM
    delivers ~5 distinct first-action classes but E2 world-forward
    compresses them to 0.22% of z_world magnitude before reaching the
    z_world-only GatedPolicy heads -- the heads are under-fed. Fix:
    bypass E2 compression by concatenating the first-action one-hot
    directly onto the head's candidate_features input.
    Config: REEConfig.gated_policy_use_first_action_onehot (bool, default
      False; bit-identical OFF when False). GatedPolicyConfig gains
      use_first_action_onehot (bool, False) and first_action_dim (int, 0;
      set to config.e2.action_dim by REEAgent.__init__ -- single source
      of truth, not a separate REEConfig knob).
    Data flow: c.actions[:, 0, :][0] -> [action_dim] one-hot per
      candidate; stacked to [K, action_dim]; cat with gp_summaries
      [K, world_dim] -> augmented head input [K, world_dim+action_dim].
      Discriminator input (z_world, z_self, z_harm_a) UNCHANGED.
    Backward compatible: gated_policy_use_first_action_onehot=False
      (default); head Linear input stays [K, world_dim]; select_action
      sets first_action_onehots=None; forward() skips the cat. 484/484
      contracts+preflight PASS with defaults unchanged.
    Phased training required for V3-EXQ-543f (P0 -> P1 -> P2).
    Validation: V3-EXQ-543f to be queued via /queue-experiment (supersedes
      V3-EXQ-543e; same 2x2 SP-CEM/dACC factorial; also requires
      dacc_weight>0 + pre-flight non-degeneracy assertion per 543e
      autopsy addendum).
  Cross-plan link: commitment_closure_plan.md GAP-1 (SD-033a bias-head
    training) unblocked at substrate level by GAP-C + GAP-D (below).
    Validation EXQ for GAP-1 deferred until V3-EXQ-543f returns a
    contributory result (GAP-B scientific gate).
  See ARC-062 (this claim, weak reading), MECH-309 (logical-necessity
    diagnostic the substrate addresses), ARC-063 (V4 strong reading,
    distributed CandidateRule field; deferred via Phase 4 / GAP-E
    multi-strategy scaling probe), SD-033a (Phase 3 downstream consumer
    -- not wired in Phase 1), MECH-262 (rule-selective persistence;
    Phase 3 work), SD-029 (monomodal-collapse measurement gate; Phase 2
    falsifier dependent variable), SD-054 (reef + hazard_food_attraction
    substrate; Phase 2 falsifier environment), MECH-094 (simulation_mode
    argument), Pull A SYNTHESIS verdicts R1 / R2 / R3, Pull B SYNTHESIS
    R4 verdict (Phase 2 acceptance criteria).
  GAP-C: ARC-062 discriminator output -> SD-033a rule_state source vector --
    IMPLEMENTED 2026-05-17.
    Changes: LateralPFCConfig gains use_discriminator_source (bool, default
      False), discriminator_pool_weight (float, default 0.3);
      LateralPFCAnalog gains discriminator_proj (nn.Linear(1, rule_dim))
      and accepts optional disc_output arg in update(). REEConfig gains
      lateral_pfc_use_discriminator_source and lateral_pfc_discriminator_pool_weight.
    agent.py reorder: gated_policy block now runs BEFORE lateral_pfc block
      so gp_output.gating_weight is available as disc_output. Score-bias
      composition is additive -- reorder is value-identical to prior ordering.
      gated_policy block sets cand_world_summaries = gp_summaries so
      lateral_pfc / ofc / mech295 blocks reuse rather than rebuild.
    Data flow: GatedPolicy.gating_weight scalar -> tensor([[w]]) [1,1]
      -> discriminator_proj -> [1, rule_dim] added to source with weight
      discriminator_pool_weight. Gated by use_discriminator_source (default
      False = no-op, bit-identical backward compat).
    Backward compatible: use_discriminator_source=False by default; update()
      disc_output=None arg is no-op. 484/484 contracts PASS; 543f dry-run
      exit 0. MECH-094: disc_output is detached (gated_policy under no_grad);
      existing MECH-319 lateral_pfc skip gate unchanged.
  GAP-D: SD-033a rule_bias_head trainable -- IMPLEMENTED 2026-05-17.
    Changes: LateralPFCConfig gains train_rule_bias_head (bool, default False);
      when True, last Linear is NOT zeroed at init (random init preserved).
      LateralPFCAnalog gains bias_head_parameters() method returning
      self.rule_bias_head.parameters() for optimizer inclusion. REEConfig
      gains lateral_pfc_train_rule_bias_head (bool, default False).
    Gradient path: E3 loss -> score_bias -> compute_bias() -> rule_bias_head
      weights. No separate loss term needed. Starting from zero-init (default
      False) OR from random-init (True), gradient flows from tick 1 of P1.
    Experiment use: in P1 optimizer, add:
        list(agent.lateral_pfc.bias_head_parameters())
    Backward compatible: train_rule_bias_head=False by default; last Linear
      remains zeroed (bias output stays 0.0); behavior bit-identical.
    Phased training: bias head can join P1 optimizer alongside gated_policy
      heads (same E3 gradient path). No separate warmup protocol needed.
    Validation EXQ (commitment_closure:GAP-1): 2-arm ablation (head frozen
      vs trainable) deferred until V3-EXQ-543f contributory result.

## MECH-313 (ARC-065 child): Stochastic Noise Floor (LC-NE tonic / SAC analog) (2026-05-10)
- MECH-313: policy.stochastic_noise_floor_lc_ne_tonic_analog -- IMPLEMENTED 2026-05-10.
  Module: ree_core/policy/noise_floor.py (NoiseFloor + NoiseFloorConfig). First of
  four ARC-065 child substrates (sibling MECH-314 / MECH-318 / MECH-319 are
  separate spawned tasks). Pure-arithmetic regulator (no learned parameters; no
  nn.Module inheritance); matches the SD-035 / SD-036 / SD-037 regulator pattern.
  State-independent softmax-temperature lift -- the LC-NE tonic complement to
  MECH-104 phasic spike. Distinct from MECH-260 dACC anti-recency (state-dependent);
  Q-045 falsifies whether they collapse into a single substrate.
  Algorithm at the e3.select() call site in REEAgent.select_action():
    effective_T = max(baseline_T + noise_floor_alpha, noise_floor_min_temperature)
  alpha is the SAC-entropy-bonus analog (Haarnoja 2018) -- additive lift on the
  softmax temperature; min_temperature is a hard floor preventing argmax collapse
  under annealing schedules that drive the baseline below 1.0.
  Config: REEConfig.use_noise_floor (bool, default False; bit-identical OFF) +
    noise_floor_alpha (float, default 0.1; modest +10%% of E3 baseline 1.0;
    Q-043 calibrates) + noise_floor_min_temperature (float, default 1.0;
    matches existing E3 baseline so well-formed callers clear the floor).
    All wired through REEConfig.from_dims().
  Agent wiring (REEAgent.__init__): instantiate NoiseFloor when use_noise_floor=True;
    REEAgent.select_action() reads noise_floor.compute_effective_temperature(
    baseline_temperature=temperature, simulation_mode=False) BEFORE calling
    e3.select(...). Bit-identical when noise_floor is None (the else branch
    passes the unmodified temperature kwarg). reset() clears diagnostic counters.
  MECH-094: simulation_mode=True returns baseline temperature unchanged and
    increments only the simulation-skip counter; replay / DMN consumers (none
    today; reserved for forward-compat) cannot inherit waking-tonic noise floor.
    Match the SD-035 / MECH-279 / gated_policy simulation_mode pattern.
  Phase-1 instantiation choice (NOT a settled architectural commitment):
    a separate NoiseFloor module at the e3.select() call site, rather than
    per-head temperature inside GatedPolicy as the original notes-field hint
    suggested. Phase-1 reasoning: MECH-313 is state-independent and currently
    must fire on baseline E3 selection too (which the per-head approach inside
    GatedPolicy would miss with GatedPolicy disabled). Whether the policy-layer
    regulators ultimately consolidate into one module is OPEN pending
    MECH-314 / MECH-318 / MECH-319 implementations -- those substrates may
    make different placement choices that motivate revisiting MECH-313's
    placement (MECH-314 structured-curiosity in particular may fit naturally
    inside GatedPolicy as a per-head bonus, in which case MECH-313 may want
    to co-locate). Re-evaluate at the point Q-045's 4-arm ablation is queued.
    Phase-1 module surface + config knobs are stable; what could move is the
    file location and call site.
  Backward compatible: use_noise_floor=False by default; agent.noise_floor is None
    and select_action passes the temperature kwarg unchanged. 253/253 contracts +
    7/7 preflight PASS with flag OFF (regression-clean).
  Lit-pull verdicts (resolved defaults; see Pull 1 SYNTHESIS at
    REE_assembly/evidence/literature/targeted_review_arc_065_behavioral_diversity_generation/
    SYNTHESIS.md, 9 entries, lit_conf 0.78-0.82, supports-direction):
    R1 BOTH-CHANNELS-NEEDED (conf 0.85): noise floor (this) AND structured
      curiosity (MECH-314) both required. Wilson 2014 Horizon task + Faisal/
      Selen/Wolpert 2008 noise-substrate irreducibility + Friston 2015
      complementary terms.
    R2 LC-NE tonic LOAD-BEARING (conf 0.84): Aston-Jones & Cohen 2005 adaptive-
      gain model. MECH-104 covers phasic spike (substrate-landed); MECH-313 is
      the tonic complement.
    R4 continuous, every tick (conf 0.80): non-zero softmax temperature on E3
      every waking tick, regardless of context. Aston-Jones & Cohen 2005 +
      Friston 2015. NOT triggered.
    Magnitudes intentionally NOT pinned by the lit-pull (Q-043 calibration
    sweep is the empirical route).
  Phased training: not applicable (pure scalar regulator; no learned parameters;
    no gradient flow).
  Validation experiment: V3-EXQ-544 substrate-readiness diagnostic (5 sub-tests
    UC1-UC5 covering instantiation, master-OFF backward-compat, lift-arithmetic
    sweep across alpha/min_temperature, select_action wiring contract via
    act_with_split_obs, MECH-094 simulation gate). Smoke 5/5 PASS 2026-05-10
    (manifests scrubbed; runner will write the canonical PASS manifest from the
    queued entry). 11 contract tests in tests/contracts/test_mech_313_noise_floor.py
    PASS. Behavioural validation deferred to Q-045 4-arm ablation (MECH-313 OFF /
    313 only / 260 only / both ON) on V3-EXQ-543b/c successors AFTER MECH-314
    also lands.
  Design doc: REE_assembly/docs/architecture/mech_313_stochastic_noise_floor.md.
  See MECH-313 (this claim), ARC-065 (parent architectural commitment),
    MECH-260 (related but distinct mechanism; Q-045 falsifies collapse),
    MECH-104 (LC-NE phasic complement; substrate-landed),
    MECH-314 (sibling structured-curiosity bonus under ARC-065; separate substrate),
    Q-043 (relative weight calibration -- parametric sweep),
    Q-044 (MECH-314a/b/c sub-flavour independence),
    Q-045 (MECH-313 vs MECH-260 collapse falsifier -- 4-arm ablation),
    MECH-094 (simulation_mode argument; call-site scoping for waking-only effects).

## MECH-314 (ARC-065 child): Structured Curiosity Bonus + 3 Sub-Flavours (2026-05-10)
- MECH-314: policy.structured_curiosity_bonus_parent + MECH-314a/b/c sub-flavours
  -- IMPLEMENTED 2026-05-10. Module: ree_core/policy/structured_curiosity.py
  (StructuredCuriosity + StructuredCuriosityConfig). Second of four ARC-065 child
  substrates (MECH-313 noise-floor landed earlier the same day; MECH-318 / MECH-319
  remain separate spawned tasks). Pure-arithmetic, no learned parameters, no
  nn.Module inheritance; sibling to MECH-313 NoiseFloor in the ree_core.policy
  package. Three sub-flavours implemented as a single module with master + 3
  independently-togglable sub-flavour switches per Pull 1 R3 verdict NOT to
  collapse them prematurely; Q-044 holds the empirical resolution path.
  Three sub-flavours:
    MECH-314a striatal novelty (Wittmann 2008): per-candidate min-distance from
      candidate's first-step z_world to nearest ACTIVE ResidueField RBF center,
      normalised by candidate-pool mean norm. Genuinely per-candidate [K].
    MECH-314b frontopolar uncertainty (Daw 2006 / Friston 2010/2015 EFE):
      e3._running_variance scalar. Phase 1 BROADCAST scalar across [K]
      (per-candidate refinement requires E1 forward-variance head, deferred
      to Phase 2 follow-on).
    MECH-314c learning progress (Schmidhuber 1991 / Pathak 2017): EMA of
      |PE_t - PE_{t-K}| where PE feed is e3._running_variance per tick.
      Phase 1 BROADCAST scalar across [K] (per-candidate refinement
      deferred). Pull 1 R3 flagged 314c as least biologically anchored;
      Q-044 outcome may retire it without architectural cost.
  Algorithm at the e3.select() call site in REEAgent.select_action():
    total = zeros[K]
    if 314a ON and residue has active centers:
        total += -w_a * normalised_min_RBF_distance[K]
    if 314b ON: total += -w_b * unc * ones[K]
    if 314c ON and lp_seeded: total += -w_c * lp_ema * ones[K]
    return clamp(total, [-bias_scale, +bias_scale])
  Bonus is non-positive in E3's lower-is-better convention (curiosity makes
  novel/uncertain/LP-rich candidates more attractive). Composed additively into
  dacc_score_bias immediately AFTER the MECH-295 liking-bridge block and
  BEFORE the MECH-313 noise_floor temperature lift (curiosity affects scores;
  noise floor affects temperature; orthogonal). LP feed:
  curiosity.update_prediction_error(e3._running_variance, simulation_mode=False)
  called after each e3.select cycle in select_action (advances 314c LP
  buffer for next tick).
  Config: REEConfig.use_structured_curiosity (default False; bit-identical OFF
  master) + use_curiosity_novelty / _uncertainty / _learning_progress (defaults
  True; consulted only when master ON) + curiosity_novelty_weight /
  curiosity_uncertainty_weight / curiosity_learning_progress_weight (default
  0.05 each; Q-043 / Q-044 calibrate) + curiosity_bias_scale (default 0.1;
  mirrors lateral_pfc_bias_scale) + curiosity_lp_ema_alpha (default 0.1;
  ~10-tick window) + curiosity_lp_window_k (default 5; Schmidhuber 1991
  first-difference). All wired through REEConfig.from_dims().
  MECH-094: compute_score_bias(simulation_mode=True) returns zeros[K] +
  increments only the simulation-skip counter; update_prediction_error(
  simulation_mode=True) no-op on the LP buffer. Match the SD-035 / MECH-279 /
  gated_policy / MECH-313 simulation_mode pattern.
  Phase-1 architectural-placement note (mirrors MECH-313): a SEPARATE
  StructuredCuriosity module at the e3.select() call site, in parallel with
  MECH-313 NoiseFloor and the GatedPolicy bias chain. Whether the policy-
  layer regulators ultimately consolidate into one module is OPEN pending
  MECH-318 / MECH-319 substrates and Q-043 / Q-044 calibration. The
  separate-module choice keeps each sub-flavour independently togglable
  (which is what Q-044 needs).
  Phase 1 honest-scoping caveat: 314a is genuinely per-candidate; 314b and
  314c are state-dependent global scalars broadcast across [K] in Phase 1.
  The architectural shape is correct (bonus magnitude varies with global
  uncertainty / LP; substrate exposes the falsification surface), and
  Q-044's three-arm ablation IS a flag-set decision. What Phase 1 does NOT
  deliver: distinguishable behavioural signatures per sub-flavour at the
  candidate-selection level (broadcast-scalar 314b/c shifts every candidate's
  score by the same amount and does not change selection ordering).
  Per-candidate refinement is a Phase 2 follow-on, deferred until Q-044
  surfaces concrete need.
  Backward compatible: use_structured_curiosity=False by default; agent.curiosity
  is None and select_action skips the entire block + LP feed -> bit-identical
  to baseline. 273/273 contracts + 7/7 preflight PASS with master OFF
  (regression-clean, was 253 + 13 new MECH-314 tests + 7 preflight).
  Phased training: not applicable (pure-arithmetic regulator; no learned
  parameters; no gradient flow).
  Validation experiment: V3-EXQ-545 substrate-readiness diagnostic (5 sub-tests
  UC1-UC5 covering instantiation, master-OFF backward-compat, sub-flavour
  flag-set isolation -- the architectural prerequisite making Q-044 three-arm
  ablation a flag-set decision -- select_action wiring contract via
  act_with_split_obs, MECH-094 simulation gate). Smoke 5/5 PASS 2026-05-10
  (manifest scrubbed; runner will write the canonical PASS manifest from the
  queued entry). 13 contract tests in tests/contracts/test_mech_314_curiosity.py
  PASS. Behavioural validation deferred to Q-044 three-arm ablation (314a-OFF
  / 314b-OFF / 314c-OFF + all-on baseline) on V3-EXQ-543b/c successors AFTER
  MECH-318 / MECH-319 absorption-check sessions complete.
  Design doc: REE_assembly/docs/architecture/mech_314_structured_curiosity_bonus.md.
  See MECH-314 (parent claim), MECH-314a / MECH-314b / MECH-314c (sub-flavours),
    ARC-065 (parent architectural commitment), MECH-313 (sibling noise-floor
    under same parent; substrate-landed earlier the same day),
    MECH-104 (LC-NE phasic complement; substrate-landed),
    MECH-260 (anti-monostrategy related claim; Q-045 collapse falsifier),
    MECH-295 (sibling score-bias contributor composed before MECH-314 in chain),
    Q-043 (relative weight calibration MECH-313 vs MECH-314 -- parametric sweep),
    Q-044 (MECH-314a/b/c sub-flavour independence -- three-arm ablation),
    Q-045 (MECH-313 vs MECH-260 collapse falsifier),
    MECH-318 / MECH-319 (sibling ARC-065 / ARC-064 substrates; separate spawned tasks),
    MECH-094 (simulation_mode argument; call-site scoping for waking-only effects).

## MECH-314a Phase-2 AMEND: e2.world_forward novelty-candidate-source (V3-EXQ-648 autopsy, 2026-06-07)
- MECH-314a Phase-2 amend: policy.structured_curiosity_bonus.e2_world_forward_novelty_source
  -- IMPLEMENTED 2026-06-07. Routed by failure_autopsy_V3-EXQ-648_2026-06-07.json
  (precondition_unmet; routing=implement-substrate; recommended_substrate_queue_entry.action
  =amend on MECH-314a-Phase-2-impl). Design-doc Candidate 1 source on the landed
  Candidate-5A machinery (REE_assembly/docs/architecture/mech_314a_phase2_novelty_source_design.md
  section 3 Candidate 1).
  ROOT CAUSE (V3-EXQ-648 FAIL, run v3_exq_648_..._20260607T025417Z): the MECH-314a
  per-candidate novelty AND the auto-augmentation _candidate_spread key were computed
  in agent.select_action from the hippocampal proposer's first-step z_world
  (trajectory.world_states[:,0,:], cur_summaries), whose cross-candidate spread is
  <0.01 under monostrategy -> curiosity_bias_range=0.0 and curiosity_std_across_K=0.0
  in EVERY arm incl the ARM_1 positive control. The SD-056-trained e2.world_forward(z0,a_i)
  predictions carry spread ~0.1147 (the representation the SD-056 readiness gate already
  validates) but were NOT the consumed signal. The 648 readiness precondition measured
  e2.cand_world_pairwise_dist (0.1147, PASS) while C2 routed on the proposer-derived bias
  (<0.01) -> false READY -> self-route mislabelled a collapsed-input artefact as a wiring
  null (canonical V3-EXQ-642 same-statistic pattern).
  THE FIX (no-op-default; bit-identical OFF):
    Module: ree_core/agent.py (new REEAgent._curiosity_candidate_summaries(candidates)
      helper + curiosity block in select_action consults it first), ree_core/utils/config.py
      (1 new flag + from_dims passthrough). ree_core/policy/structured_curiosity.py UNCHANGED
      -- both _compute_novelty and _candidate_spread already key on the candidate_world_summaries
      argument, so rebuilding that argument fixes both at once.
    Config: REEConfig.curiosity_candidate_source: Literal["proposer","e2_world_forward"]
      = "proposer" (default; wired through REEConfig.from_dims). "proposer" -> helper returns
      None -> legacy proposer-summary reuse-chain runs unchanged (bit-identical).
      "e2_world_forward" -> cur_summaries = e2.world_forward(z0.expand(K,-1), first_actions_K)
      [K, world_dim] (z0 = current observed z_world; first_actions_K = stack(c.actions[:,0,:])),
      the same construction the SD-056 cand_world_pairwise_dist readiness diagnostic uses.
  Data flow (e2_world_forward ON): current latent z_world + per-candidate first action ->
    e2.world_forward -> [K, world_dim] action-divergent predictions -> StructuredCuriosity.
    compute_score_bias(candidate_world_summaries=...) -> BOTH 314a RBF novelty (vs visitation
    buffer / residue centers) AND the auto-augmentation _candidate_spread key now key on the
    action-divergent representation.
  Backward compatible: default "proposer" -> bit-identical (no e2 call; existing reuse-chain).
    877/877 contracts + 7/7 preflight PASS; V3-EXQ-648 --dry-run unchanged (default proposer
    path; dry-scale substrate_not_ready_requeue as before). New C6 contracts (4) in
    tests/contracts/test_mech_314_curiosity.py: proposer returns None (bit-identical);
    e2_world_forward returns [K,world_dim]; with a monkeypatched divergent world_forward the
    curiosity-consumed _candidate_spread exceeds the collapsed proposer spread (the 648
    root-cause made deterministic); proposer-collapsed-baseline sanity.
  Phased training: N/A -- no new learned parameters; reuses the already-trained
    e2.world_forward (SD-056). The validation experiment trains e2 via the SD-056 online
    contrastive warmup in P0 (as V3-EXQ-648 already does). DETECTOR DEPENDS ON A TRAINED
    e2: on an untrained/under-trained e2 the predictions re-collapse -> the V3-EXQ-648a
    readiness precondition (re-targeted to THIS consumed representation) catches it and
    self-routes substrate_not_ready_requeue rather than mislabelling a wiring null.
  MECH-094: preserved. The e2.world_forward read is torch.no_grad() on the waking
    select_action path (no replay / memory write surface); the visitation-buffer write in
    sense() is untouched and stays MECH-094-gated; compute_score_bias's simulation_mode
    gate is unchanged.
  Evidence-staleness: NOT triggered -- no-op-default flag; every existing experiment uses
    the default "proposer" source, so no dependent claim's measured mechanism changed.
  Validation experiment: V3-EXQ-648a (supersedes V3-EXQ-648) -- the corrected
    substrate-readiness diagnostic enabling curiosity_candidate_source="e2_world_forward"
    + re-targeting the readiness precondition to the consumed representation, queued via
    /queue-experiment. PASS gates V3-EXQ-590b + the section-8 MECH-314a/MECH-314/ARC-065
    governance/claims updates (which stay GATED on PASS, not applied here).
  Design doc: REE_assembly/docs/architecture/mech_314a_phase2_novelty_source_design.md
    (Candidate 1 source; V3-EXQ-648a amend section).
  See MECH-314 (parent), MECH-314a (sub-flavour), ARC-065 (parent architectural commitment),
    SD-056 (e2.world_forward action-conditional divergence; the representation now consumed),
    V3-EXQ-648 (the FAIL this amend addresses), V3-EXQ-648a (validation), MECH-094 (call-site
    scoping; preserved).

## ARC-065 GAP-A: shared cand_world_summaries e2.world_forward source (V3-EXQ-614e autopsy, 2026-06-07)
- ARC-065 GAP-A: policy.candidate_pool_per_candidate_signal_preservation.shared_channel
  -- IMPLEMENTED 2026-06-07. Routed by failure_autopsy_V3-EXQ-614e_2026-06-07
  (confirmed; non_contributory / substrate_ceiling / pending_retest_after_substrate
  gated on ARC-065 GAP-A). The shared-channel sibling of the MECH-314a Phase-2
  curiosity amend (landed earlier the same day): that pass fixed ONLY the curiosity
  channel's consumed representation; this pass extends the identical e2.world_forward
  re-sourcing to the SHARED per-candidate cand_world_summaries consumed by ALL the
  other E3-side bias channels (lateral_pfc / ofc / mech295 / gated_policy / tonic_vigor).
  ROOT CAUSE (V3-EXQ-614e): with the modulatory-bias-selection-authority gate now
  PROVEN operative (V3-EXQ-643a PASS), committed-class diversity still showed no lift
  (committed_class_entropy byte-identical across within-class T=0.5/1.0/2.0). The
  autopsy relocated the bottleneck from the authority gate (GAP-B, resolved) to the
  UPSTREAM candidate pool (GAP-A): all K CEM candidates produce identical z_world after
  one E2 world-forward step (cand_world_pairwise_dist=0.0000) despite differing first
  actions, so every E3-side bias channel sees a class-uniform pool. The shared
  cand_world_summaries was built from the collapsed proposer first-step z_world
  (trajectory.world_states[:,0,:]) at five fresh-build sites in select_action; the
  fix re-sources it from the SD-056-trained action-conditional e2.world_forward(z0, a_i)
  predictions (cross-candidate spread ~0.11 when SD-056 is trained), the representation
  the SD-056 readiness gate already validates and the GAP-B first-action-onehot fix
  bypassed only for GatedPolicy.
  THE FIX (no-op-default; bit-identical OFF):
    Module: ree_core/agent.py (new REEAgent._candidate_world_summaries(candidates)
      helper -- shared-channel sibling of _curiosity_candidate_summaries; consulted
      FIRST at all five cand_world_summaries fresh-build sites: gated_policy block,
      lateral_pfc fallback, ofc fallback, mech295 fallback, and via the reuse-chain
      the tonic_vigor anchor). ree_core/utils/config.py (1 new flag + from_dims
      passthrough).
    Config: REEConfig.candidate_summary_source: Literal["proposer","e2_world_forward"]
      = "proposer" (default; wired through REEConfig.from_dims). "proposer" -> helper
      returns None -> legacy collapsed proposer-summary build runs unchanged
      (bit-identical). "e2_world_forward" -> cand_world_summaries =
      e2.world_forward(z0.expand(K,-1), first_actions_K) [K, world_dim], the same
      construction _curiosity_candidate_summaries (648a) + the SD-056
      cand_world_pairwise_dist readiness diagnostic use. Kept SEPARATE from
      curiosity_candidate_source (the in-flight 648a validation) so the two compose
      without perturbing each other; a future pass may unify.
  Data flow (e2_world_forward ON): current z_world + per-candidate first action ->
    e2.world_forward (no_grad) -> [K, world_dim] action-divergent shared summaries ->
    lateral_pfc / ofc / mech295 / gated_policy / tonic_vigor bias channels ->
    dacc_score_bias -> e3.select(score_bias=...) -> modulatory-bias-selection-authority
    (643a) gap-relative authority over the committed argmin -> MECH-341 stratified
    across-class selection -> committed-class diversity expressible.
  Backward compatible: default "proposer" -> bit-identical (no e2 call; existing
    five build paths unchanged). 889/889 contracts (883 prior + 6 new) + 7/7 preflight
    PASS; V3-EXQ-614e --dry-run unchanged (default proposer path; same dry-scale
    outcome). New contracts tests/contracts/test_arc065_gapa_candidate_summary_source.py
    (G1 default-proposer-helper-None bit-identical + default==explicit / G2
    e2_world_forward [K,world_dim] shape / G3 divergent-world_forward shared spread >
    collapsed proposer spread -- the 614e cand_world_pairwise_dist=0.0000 fix made
    deterministic / G4 master-off helper None / G5 select_action end-to-end with source
    ON + lateral_pfc/ofc/mech295 ON).
  Phased training: N/A new params; reuses the already-trained e2.world_forward (SD-056).
    DETECTOR DEPENDS ON A TRAINED e2: on an untrained/under-trained e2 the predictions
    re-collapse -> the validation EXQ trains e2 (SD-056 online) in P0 and a
    cand_world_pairwise_dist readiness precondition guards vacuity (substrate_not_ready_requeue),
    same pattern as 648a.
  MECH-094: the e2.world_forward read is torch.no_grad() on the waking select_action
    path (no replay / memory write surface); not implicated.
  Evidence-staleness: NOT triggered -- no-op-default flag; every existing experiment
    uses the default "proposer" source, so no dependent claim's measured mechanism
    changed. KEEP all evidence.
  Validation experiment: V3-EXQ-649 (queued via /queue-experiment) -- substrate-readiness
    diagnostic (claim_ids=[]) enabling candidate_summary_source="e2_world_forward" + a
    cand_world_pairwise_dist readiness precondition + a shared-bias-channel per-candidate
    range readout (proposer ~0 vs e2_world_forward > floor). PASS unblocks the MECH-341
    committed-class diversity re-test (the within-class-REPRESENTATIVE-diversity readout,
    NOT committed-class entropy -- autopsy Learning #2), which is the governance-weighting
    successor queued separately. MECH-341 stays candidate / v3_pending; not weakened.
  Design doc: REE_assembly/docs/architecture/mech_314a_phase2_novelty_source_design.md
    (Candidate 1 source; GAP-A shared-channel extension section).
  Plan-of-record: REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
    (GAP-A node) + substrate_queue.json (ARC-065 entry, 614e failure_record).
  See ARC-065 (parent architectural commitment), behavioral_diversity_isolation:GAP-A
    (closure node), MECH-314a Phase-2 (curiosity-channel sibling; curiosity_candidate_source),
    SD-056 (e2.world_forward action-conditional divergence; the representation now consumed
    by the shared channel), ARC-062 GAP-B (the GatedPolicy first-action-onehot fix this
    generalises beyond GatedPolicy), modulatory-bias-selection-authority / V3-EXQ-643a
    (the authority gate that lets the now-divergent bias reach the committed argmin),
    MECH-341 (committed-class diversity re-test, pending_retest_after_substrate),
    V3-EXQ-614e (the FAIL this addresses), V3-EXQ-649 (validation), MECH-094 (call-site
    scoping; preserved).

## MECH-319 (arc_062 GAP-K): Simulation-Mode Rule-Write Gate (Categorical Replay Tag) (2026-05-10)
- MECH-319: policy.arbitration.simulation_mode_write_gating_substrate_ree_novel_function
  -- IMPLEMENTED 2026-05-10. Module: ree_core/regulators/simulation_mode_rule_gate.py
  (SimulationModeRuleGate + SimulationModeRuleGateConfig + SimulationModeRuleGateDiagnostics).
  Substrate-level instantiation of MECH-094 at the rule-arbitration layer. Pure-arithmetic
  regulator (no nn.Module inheritance, no learned parameters); sibling to
  GABAergicDecayRegulator (SD-036) and BroadcastOverrideRegulator (SD-037) in the
  regulators package. Resolves arc_062 GAP-K substrate gap registered in the 2026-05-10
  cluster-registration session.
  Single primitive: gate.effective_simulation_mode(simulation_mode, site) -> bool.
  Truth table (master_on, admit_writes, caller_sim) -> output:
    OFF, *,     *     -> caller_sim (identity; bit-identical pre-MECH-319)
    ON,  False, False -> False     (admit waking write)
    ON,  False, True  -> True      (block simulation write -- MECH-319 normal)
    ON,  True,  False -> False     (admit waking; flag has no effect)
    ON,  True,  True  -> False     (admit simulation write -- V3-EXQ-543c falsifier)
  The gate is idempotent for waking calls (always returns False), so wiring it into
  existing waking call sites is bit-identical regardless of admit_writes. The falsifier-
  control asymmetry surfaces only when caller_sim=True (replay paths, ghost-goal probes,
  DMN passes -- not currently exercised by select_action).
  Diagnostic counters (per-call + per-site): n_calls_total, n_waking_admitted,
  n_simulation_blocked, n_simulation_admitted (the falsifier path), plus per_site_*
  dicts keyed on canonical labels SITE_GATED_POLICY, SITE_LATERAL_PFC, SITE_DEFAULT.
  New consumer call sites can pass arbitrary site strings.
  Config: REEConfig.use_simulation_mode_rule_gate (bool, default False; bit-identical OFF
  master) + REEConfig.simulation_mode_rule_gate_admit_writes (bool, default False;
  V3-EXQ-543c artificial-write-channel-routing falsifier flag). All wired through
  REEConfig.from_dims(). Construction raises ValueError when admit_writes=True without
  master ON (loud-not-silent guard against mis-configuration -- admit_writes is meaningless
  without the substrate to gate).
  Agent wiring (REEAgent.__init__): instantiate simulation_mode_rule_gate when master ON;
  None otherwise. Two existing arbitration-write call sites in REEAgent.select_action()
  consult the gate when it is instantiated:
    GatedPolicy block: replace literal simulation_mode=False with
      gate.effective_simulation_mode(False, site=SITE_GATED_POLICY) and pass to
      gated_policy.forward(...). Bit-identical for waking; seam exposed for V3-EXQ-543c.
    LateralPFCAnalog block: consult gate via
      eff_sim = gate.effective_simulation_mode(False, site=SITE_LATERAL_PFC); if eff_sim:
      skip lateral_pfc.update(...) else proceed with the existing MECH-261 mode-
      conditioned EMA. compute_bias still runs (arbitration RECEIVES the bias even
      during simulation; it just does not write back into rule_state on simulation
      ticks). Bit-identical for waking; falsifier-routed simulation writes to
      lateral_pfc would be skipped under default MECH-319 ON, admitted under
      admit_writes=True.
  Per-episode reset() clears diagnostic counters (gate has no persistent state across
  ticks beyond counters).
  MECH-094 INVARIANCE: this substrate does NOT modify MECH-094, GatedPolicy.forward's
  simulation_mode argument semantics, or LateralPFCAnalog.update. Pull 3 SYNTHESIS R1
  GENUINE-NOVELTY-CONFIRMED conf 0.72 + Pull 4 R3 KEEP-AS-IS verdict. The gate is a
  pre-call coordinator that wraps the simulation_mode argument that callers ALREADY
  pass. With MECH-319 disabled, every arbitration-write call site behaves bit-identically
  to its pre-MECH-319 form.
  RELATION TO MECH-261: MECH-261 (mode-conditioned write-gate registry on SD-032a
  SalienceCoordinator) is a complementary, continuous gate. MECH-261 returns a per-mode
  weight in [0, 1] that scales the magnitude of the EMA update on
  LateralPFCAnalog.rule_state. MECH-319 is a categorical (binary admit/block) pre-gate
  keyed to the simulation tag of the caller, not the operating mode of the agent.
  The two gates compose: caller waking + MECH-319 admits -> MECH-261 modulates EMA
  strength based on operating mode. Caller simulation + MECH-319 normal -> MECH-319
  blocks the entire update() call, MECH-261 never consulted. Caller simulation +
  MECH-319 falsifier -> MECH-319 admits -> MECH-261 modulates as if waking.
  Backward compatible: use_simulation_mode_rule_gate=False by default;
  agent.simulation_mode_rule_gate is None and both call sites take the legacy literal
  path. 288/288 contract + preflight tests PASS with master OFF (regression-clean;
  was 273 + 15 new MECH-319 contracts).
  Lit-pull verdicts (resolved defaults; see Pull 3 SYNTHESIS at
  REE_assembly/evidence/literature/targeted_review_mech_312_arbitration_divergences/
  synthesis.md):
    R1 GENUINE-NOVELTY-CONFIRMED (conf 0.72): substrate-availability premise well-
      anchored (Joo & Frank 2018 SWR review + Foster & Wilson 2006 reverse replay
      discriminable signature); the categorical write-gate FUNCTION at the arbitration
      layer is REE-novel. The literature provides only the substrate; downstream
      regions are not yet documented as exploiting the discriminable replay signature
      as a write-gate.
    Pull 4 R3 KEEP-AS-IS: MECH-094 stays as-is (architectural principle); MECH-319
      is registered as the substrate-level instantiation. The two claims are NOT
      redundant.
  Phased training: not applicable (pure boolean / counter arithmetic; no learned
  parameters; no gradient flow).
  Validation experiment: V3-EXQ-546 substrate-readiness diagnostic (6 sub-tests
  UC1-UC5 + UC3b precondition: instantiation + diagnostic keys; master-OFF backward-
  compat; truth-table coverage across the 6 valid (master, admit_writes, caller_sim)
  combinations; precondition raises on admit_writes=True without master ON;
  select_action wiring contract -- gate sees waking calls from both gated_policy and
  lateral_pfc sites after one act_with_split_obs tick, n_simulation_* counters remain
  zero on the waking path; MECH-094 invariance -- master-OFF and master-ON-with-waking-
  caller produce bit-identical wiring outputs, asymmetry surfaces only at
  caller_sim=True). Smoke 6/6 PASS 2026-05-10 (manifest scrubbed; runner will write
  the canonical PASS manifest from the queued entry). 15 contract tests in
  tests/contracts/test_mech_319_simulation_mode_rule_gate.py PASS. Behavioural
  validation -- the V3-EXQ-543c-successor falsifier with the admit_writes=True arm
  and a replay-driven invocation path -- is queued separately AFTER MECH-313 /
  MECH-314 / MECH-318 sibling substrates have landed.
  Design doc: REE_assembly/docs/architecture/mech_319_simulation_mode_rule_gate.md.
  See MECH-319 (this claim), MECH-094 (architectural principle this substrate
    instantiates; KEEP-AS-IS per Pull 3 R1 + Pull 4 R3 verdicts -- not modified),
    MECH-312 (parent arbitration layer whose write-gate this implements),
    MECH-312a/b/c/d (sub-mechanisms gated by the simulation tag),
    MECH-261 (mode-conditioned continuous write gate on SalienceCoordinator;
      complementary to MECH-319's categorical pre-gate),
    ARC-062 (rule apprehension cluster; GAP-K hosted this landing),
    MECH-313 / MECH-314 / MECH-318 (sibling ARC-065 / ARC-064 substrates landed
      in separate spawned tasks the same day),
    MECH-293 (ghost-goal probes; future call site that will pass simulation_mode=
      True through the gate),
    SD-033a (LateralPFCAnalog; one of the two wired arbitration-write call sites),
    GatedPolicy (ARC-062 Phase 1; the other wired arbitration-write call site),
    MECH-309 (logical-necessity claim motivating the broader ARC-062 cluster).

## MECH-320 (ARC-066 child): Tonic Vigor Coupling Score Bias (mesolimbic-DA-vigor / avg-reward-rate) (2026-05-10)
- MECH-320: action.tonic_vigor_coupling_score_bias -- IMPLEMENTED 2026-05-10.
  Module: ree_core/policy/tonic_vigor.py (TonicVigor + TonicVigorConfig +
  TonicVigorOutput). First child mechanism for ARC-066 (the
  non_deficit_action_drives architectural family). Pure-arithmetic regulator
  (no learned parameters; no nn.Module inheritance); sister to MECH-313
  NoiseFloor and MECH-314 StructuredCuriosity in the ree_core.policy package.
  Adds an additive (or multiplicative-gain falsifiable secondary) bias to E3
  trajectory scoring such that action-trajectories receive a NEGATIVE bias
  (REE lower-is-better favours action) and no-op-trajectories receive a
  POSITIVE bias (penalises passivity), proportional to a slow EWMA over the
  realised E3-score-receipt stream gated by secondary internal-state
  modulators (energy / drive / recent PE). TARGET-FREE: bias applies
  regardless of whether any z_goal is currently active -- closes the
  "well-fed-safe-familiar agent has no positive gradient to act" gap that
  ARC-066 registered.
  Algorithm at the e3.select() call site in REEAgent.select_action():
    update_score_receipt: v_raw <- (1-alpha)*v_raw + alpha*(-score_realised)
      (REE-low-is-better internally negated so v_raw climbs in reward-rich
      regimes; alpha = 1 - 0.5**(1/half_life))
    compute_score_bias: v_t = max(0, v_raw) * gate_energy * gate_drive * gate_pe
      bias[i] = -w_action * v_t   if action_classes[i] != noop_class
                +w_passive * v_t  if action_classes[i] == noop_class    (additive)
      bias[i] = (-w_action * v_t * |scores[i]|) on action /
                (+w_passive * v_t * |scores[i]|) on noop                 (multiplicative)
      bias = clamp(bias, [-bias_scale, +bias_scale])
  Composed AFTER MECH-314 curiosity (orthogonal axis: curiosity rewards
  novelty / uncertainty / LP at the candidate level; vigor biases on the
  action-vs-no-op axis) and BEFORE MECH-313 noise_floor (which lifts softmax
  temperature, not scores -- orthogonal to bias).
  Composition order at e3.select call site:
    dacc_score_bias  +=  lateral_pfc_bias       (SD-033a)
                      +  ofc_bias               (SD-033b)
                      +  mech295_liking_bias    (MECH-295)
                      +  curiosity_bias         (MECH-314)
                      +  tonic_vigor_bias       (MECH-320; -action / +noop)
    [then MECH-313 lifts effective_temperature for the softmax]
  Config: REEConfig.use_tonic_vigor (default False; bit-identical OFF) +
    tonic_vigor_half_life (100.0; long-window EWMA per R4 verdict --
    Niv 2007 long-run avg-reward-rate, NOT short-window) + tonic_vigor_w_action
    (0.1) + tonic_vigor_w_passive (0.1) + tonic_vigor_bias_scale (0.1; mirrors
    lateral_pfc / curiosity bias_scale so MECH-320 cannot dominate the
    score-bias chain at extreme reward histories) + tonic_vigor_gate_energy_min
    (0.2) + tonic_vigor_gate_drive_max (0.7) + tonic_vigor_gate_pe_max (1.0) +
    tonic_vigor_form ("additive" | "multiplicative"; validated at construction)
    + tonic_vigor_noop_class (0; matches MECH-279 PAG freeze-gate convention).
    All wired through REEConfig.from_dims().
  Agent wiring (REEAgent.__init__): instantiate TonicVigor when
    use_tonic_vigor=True (parallel to self.noise_floor / self.curiosity).
    REEAgent.select_action() reads tonic_vigor.compute_score_bias(...)
    AFTER MECH-314 curiosity block and composes additively into
    dacc_score_bias; AFTER e3.select() returns, feeds the SELECTED
    candidate's E3 score back into tonic_vigor.update_score_receipt(...) to
    advance the EWMA for the next tick. reset() clears EWMA + diagnostic
    counters.
  Energy proxy: 1 - effective_drive_level (post-pACC). Drive: post-pACC
    effective_drive_level (matches AIC / PCC / SalienceCoordinator reads).
    Recent PE: e3._running_variance (same signal MECH-314c learning-progress
    consumes).
  MECH-094: simulation_mode=True on either compute_score_bias or
    update_score_receipt returns zeros (or skips state advance) and
    increments only the simulation-skip counter. Match the SD-035 / MECH-279 /
    gated_policy / MECH-313 / MECH-314 simulation_mode pattern.
  Lit-pull verdicts (resolved defaults; see ARC-066 SYNTHESIS at
    REE_assembly/evidence/literature/targeted_review_arc_066_tonic_vigor/
    synthesis.md, lit_conf 0.789, supports-direction, 7 entries):
    R1 -- mesolimbic DA-vigor LOAD-BEARING (Niv 2007 formalism + Salamone &
          Correa 2012 substrate identity + Beierholm 2013 human L-DOPA
          causal test). LC-NE-direction REJECTED -- LC-NE tonic mode is
          one mechanism (noise = MECH-313), per Kane et al. 2017 DREADD
          test by the original Aston-Jones / Cohen authorship group.
    R3 -- ADDITIVE form is primary (Niv 2007 opportunity-cost derivation
          is naturally additive). MULTIPLICATIVE GAIN is the falsifiable
          secondary; both implementable via tonic_vigor_form. The
          discriminative-pair behavioural validation (queued separately)
          chooses which lands as primary.
    R4 -- SLOW EWMA over realised E3-score-receipt is the primary scalar
          (Niv 2007 average-reward-rate formalism + Beierholm 2013
          empirical confirmation). Internal-state proxies (energy, drive,
          recent PE) enter as SECONDARY MODULATORS. The slot's
          registration-time pre-guess of "high energy AND low recent PE
          AND low drive" composite is REFINED to the literature-attributed
          history-average primary + internal-state secondaries.
  Phase-1 instantiation choice (mirrors MECH-313 / MECH-314): a SEPARATE
    TonicVigor module at the e3.select() call site, parallel to NoiseFloor
    and StructuredCuriosity. Whether the policy-layer regulators ultimately
    consolidate into one module is OPEN pending future refactor passes.
  Distinct-from contracts: orthogonal to MECH-313 (noise on choice vs
    direction on score axis); target-free vs target-conditioned MECH-216;
    capacity-keyed vs deficit-keyed inverse SD-012; mathematical complement
    of ARC-068 (opportunity-cost no-op penalty -- collapse-vs-separate
    decision deferred to ARC-068 lit-pull); state-INDEPENDENT vs
    state-dependent recency-keyed MECH-260.
  Backward compatible: use_tonic_vigor=False by default; agent.tonic_vigor
    is None and the entire select_action MECH-320 block is skipped.
    309/309 contracts (281 prior + 28 new MECH-320) + 7/7 preflight PASS
    with master OFF (regression-clean, 2026-05-10).
  Phased training: not applicable (pure-arithmetic regulator; no learned
    parameters; no gradient flow).
  Validation experiment: V3-EXQ-547 substrate-readiness diagnostic (6
    sub-tests UC1-UC6 covering instantiation, master-OFF backward-compat,
    EWMA convergence + half-life + sign convention, select_action wiring
    contract via act_with_split_obs (compute_score_bias pre-select +
    update_score_receipt post-select both fire), MECH-094 simulation gate,
    R3 form discriminability additive-vs-multiplicative). Smoke 6/6 PASS
    2026-05-10 (manifest scrubbed; runner will write the canonical PASS
    manifest from the queued entry). 28 contract tests in
    tests/contracts/test_mech_320_tonic_vigor.py PASS. Behavioural
    validation -- the 3-arm discriminative pair (baseline / additive /
    multiplicative on a well-fed-safe-familiar environment substrate) --
    is queued separately AFTER V3-EXQ-547 PASSes via the runner.
  Design doc: REE_assembly/docs/architecture/non_deficit_action_drives.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_arc_066_tonic_vigor/
  See MECH-320 (this claim), ARC-066 (parent architectural commitment),
    MECH-313 (sibling LC-NE noise floor; orthogonal axis -- substrate-landed
    earlier the same day; lit-pull R2 verdict establishes LC-NE tonic mode
    is fully covered by MECH-313 with no remaining ARC-066 LC-NE function),
    MECH-314 (sibling structured curiosity bonus; substrate-landed earlier
    the same day; orthogonal axis at the candidate-feature level),
    MECH-216 (target-conditioned predictive wanting; ARC-066 is target-free),
    SD-012 (deficit-keyed homeostatic drive; ARC-066 is capacity-keyed inverse),
    MECH-260 (state-dependent dACC anti-recency; orthogonal),
    MECH-295 (sibling score-bias contributor composed before MECH-320),
    ARC-068 (opportunity_cost_no_op_penalty; mathematical complement;
      collapse-vs-separate decision deferred to ARC-068 lit-pull),
    SD-037 (broadcast override; deficit-recruited; ARC-066 is surplus-
      recruited; opposite corners of state space),
    MECH-094 (simulation_mode argument; call-site scoping for waking-only
      effects).

## MECH-341 (ARC-065 Layer-B child): E3 Score Diversity Preservation (2026-05-27)
- MECH-341: ethics_engine_3.scoring_trajectory_class_diversity_preservation
  -- IMPLEMENTED 2026-05-27.
  Module: ree_core/predictors/e3_score_diversity.py (E3ScoreDiversity +
  E3ScoreDiversityConfig + E3ScoreDiversityDiagnostics + build_from_ree_config).
  Layer-B (post-CEM scoring) diversity-preservation substrate. Triggered by
  V3-EXQ-608 P2 (2026-05-26T02:58Z) majority R2a_e3_collapse_confirmed_large_gap:
  with SP-CEM main-path delivering frac_pre_ge2=1.0 (>= 2 first-action classes
  in the candidate pool every measured tick), E3 scoring collapsed to a single
  class with mean_top2_class_gap 0.27-0.60 (LARGE-gap; rules out option 3
  jittered tie-breaking; routes to options 1 + 2 per
  behavioral_diversity_isolation_plan.md "Substrate design options" section).
  Two togglable sub-flavours under one master, mirroring MECH-314a/b/c precedent
  so Q-054 falsifier can dissociate which one carries the load.
    Option 1 (entropy_bonus): per-candidate POSITIVE bias proportional to the
      first-action class's frequency in the pool (REE lower-is-better convention,
      so positive bias penalises over-represented classes). Composed into scores
      AFTER the dACC / lateral_pfc / ofc / mech295 / curiosity / tonic_vigor
      score_bias chain and BEFORE last_scores / softmax. Mnih 2016 A3C-style
      entropy regularisation adapted to local candidate-pool first-action
      categorical axis (not global policy entropy).
    Option 2 (stratified_select): partition candidates by first-action class,
      pick argmin within each class as the class representative, softmax-sample
      across class-representatives with temperature stratified_temperature.
      Replaces argmin in the committed-path selection at e3_selector.py:811-820.
      Falls through to legacy argmin when fewer than
      min_classes_for_stratification unique classes are present.
  Pure-arithmetic regulator (no nn.Module inheritance, no learned parameters).
  Sibling pattern to MECH-313 NoiseFloor, MECH-314 StructuredCuriosity, MECH-320
  TonicVigor.
  Config: REEConfig.use_e3_score_diversity (bool, default False; bit-identical
  OFF master). Sub-knobs (all consulted only when master ON; all wired through
  REEConfig.from_dims): use_e3_diversity_entropy_bonus (True),
  use_e3_diversity_stratified_select (True), e3_diversity_entropy_lambda (0.05;
  Q-054 calibrates), e3_diversity_entropy_bias_scale (0.1; mirrors
  lateral_pfc / curiosity / tonic_vigor bias_scale), e3_diversity_stratified_temperature
  (1.0), e3_diversity_min_classes_for_stratification (2).
  Agent wiring (REEAgent.__init__): instantiate via
  build_e3_score_diversity_from_ree_config(config) when master ON; None
  otherwise. reset() clears diagnostic counters. select_action passes
  score_diversity=self.score_diversity kwarg to e3.select(...).
  E3TrajectorySelector.select() gains a score_diversity: Optional[Any] = None
  kwarg; Option 1 composed at the same site as score_bias addition (before
  last_scores); Option 2 consulted in the committed branch before falling
  through to argmin.
  Data flow: scores (per-candidate harm + benefit + goal + residue) -> +
  dacc_score_bias chain -> + MECH-341 Option 1 entropy bonus -> last_scores
  diagnostics -> softmax(-scores / effective_temperature) [MECH-313 lifts
  temperature] -> if committed: MECH-341 Option 2 stratified pick OR legacy
  argmin / else: multinomial(probs).
  Backward compatible: use_e3_score_diversity=False by default;
  agent.score_diversity is None and both call sites are skipped (bit-identical
  to baseline). 506/506 contracts + 7/7 preflight PASS with flag OFF
  (regression-clean 2026-05-27). Single-class candidate pools produce zero
  bonus and fall through to argmin (no-op even when master ON).
  Lit-pull verdicts (defaults per behavioral_diversity_isolation_plan.md):
    Rigotti et al. 2013 -- mixed selectivity in PFC encodes diverse trajectory
      contingencies; preservation across scoring layers required for downstream
      behavioural flexibility.
    Padoa-Schioppa & Conen 2017 -- OFC value comparison preserves option-distinct
      value signals through the comparison stage; collapse to single rank is
      pathological.
  Both options are valid biological renderings; Option 1 is the soft-bias /
  entropy-pressure reading, Option 2 is the OFC categorical-preservation reading.
  Togglable-both architecture lets the Q-054 falsifier dissociate empirically.
  MECH-094: both methods accept simulation_mode argument; when True,
  apply_entropy_bonus returns zeros[K] and stratified_select returns None
  (caller falls through to legacy argmin). Inline gates are defensive (the
  wired call site E3Selector.select is currently invoked only from waking
  REEAgent.select_action paths). Diagnostic counter mech341_n_simulation_skipped
  tracks both paths.
  Phased training: not applicable (pure-arithmetic regulator; no learned
  parameters; no gradient flow). Q-054 calibration of entropy_lambda is a
  parametric sweep, not phased latent-target training.
  Validation experiment: V3-EXQ-611 queued (4-arm substrate-readiness
  diagnostic: ALL_OFF / OPT1_ONLY / OPT2_ONLY / BOTH_ON on the EXQ-608 env
  + metric stack; acceptance criteria reuse EXQ-608's mean_top2_class_gap +
  frac_pre_ge2 + selected_classes_count metrics; PASS = either single-option
  arm produces selected_action_classes_count >= 2 with frac_pre_ge2 >= 0.5).
  V3-EXQ-611 FAILed substrate-readiness 2026-05-27T13:02Z (manifest
  v3_exq_611_mech341_substrate_readiness_4arm_20260527T130213Z_v3.json):
  ARM_1/3 entropy_bonus_max_abs ~0.023-0.044 dwarfed by mean_top2_class_gap
  0.27-1.96; ARM_2 n_stratified_fired=0 across all 3 seeds because the
  committed branch was never entered during measurement and the prior
  implementation gated stratified_select to the committed path only.
  Behavioural validation (Phase P3 B_only / ablate_B / ALL_ON arms per the
  isolation plan + R2.c rule) deferred to a successor queued after the
  retune validation PASS.
  Retune (2026-05-28): two-action substrate update under user-confirmed scope
  via AskUserQuestion (TASK_CLAIMS session
  implement-substrate-mech-341-retune-20260528T165000Z). (a) Module change:
  ree_core/predictors/e3_selector.py applies stratified_select on BOTH
  committed and uncommitted branches (was committed-only); fixes the
  zero-fires issue surfaced by V3-EXQ-611 ARM_2. Bit-identical when
  score_diversity is None or sub-flag is False -- stratified_select returns
  None and caller falls through to legacy argmin (committed) or multinomial
  (uncommitted). MECH-094 preserved via the existing simulation_mode kwarg.
  506/506 contracts PASS post-edit. (b) Parameter sweep: V3-EXQ-611b queued
  (6-arm factorial: 3 option groups x 2 entropy_bias_scale values 1.0/2.0)
  to test scale-commensurability of the entropy bonus against the observed
  score-gap range. NO config default changes per implement-substrate skill
  rule -- scales passed via cfg_overrides per arm. Acceptance criteria:
  C1 stratified fires across all OPT2/BOTH arm seeds (direct test of the
  call-site expansion); C2 entropy_bonus_max_abs >= 0.7 * scale on majority
  of seeds in entropy-ON arms; C3 selected_classes >= 2 with frac_pre_ge2
  >= 0.5 on majority of seeds in at least one arm; R2.c readiness threshold
  cleared by at least one arm. Sentinel routing: PASS -> behavioural
  successor; FAIL with C1=false -> /diagnose-errors on e3_selector wiring;
  FAIL with C1=true and C2/C3=false -> substrate revisit (algorithm-level
  Option-2 redesign or floor adjustment).
  Design doc: REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md
  Plan-of-record: REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
  See MECH-341 (this claim), ARC-065 (parent diversity-generation pathway --
  Layer A), MECH-313 / MECH-314 / MECH-318 / MECH-319 (sibling ARC-065 / ARC-064
  child substrates at proposal / action-selection layers), Q-054 (minimum
  trajectory-class diversity floor for ARC-062 -- entropy_lambda calibration),
  INV-076 (diversity as structural prerequisite for ethical counterfactual
  evaluation -- companion universal invariant), ARC-062 (rule apprehension
  via gated policy; GAP-B downstream beneficiary), SD-003 (superseded;
  counterfactual pipeline retained via MECH-256 / SD-029 / MECH-257),
  MECH-094 (hypothesis_tag invariant; call-site-scoped via simulation_mode
  argument).

## MECH-341 Amend: stratified_within_class_temperature + A-vs-B partial-redundancy probe (2026-06-01)
- MECH-341 amend -- IMPLEMENTED 2026-06-01. Routed by
  failure_autopsy_V3-EXQ-616_2026-05-31.md Sections 7 + 10 (contingent-on-
  614b-FAIL-C1 path: 614b FAILed C1 R2.c with B_only Rung-1 majority=False
  AND C2 necessity_delta 0.087 nats below the pre-amend 0.1 threshold;
  C3 ALL_ON Rung-1 PASSed with mean entropy 0.800 nats -- highest of any
  614-lineage run, positive substrate-readiness for the SD-056 amend at
  the behavioural-runtime horizon). The 616 autopsy's named contingent
  amend is "stratified_temperature default + A-vs-B partial-redundancy
  probe"; this session lands the within-class proportional sampling lever
  + names the A-vs-B factorial via the existing independent flags.
  Two-part amend:
    Part (a) within-class proportional sampling. Extends
      ree_core/predictors/e3_score_diversity.py E3ScoreDiversity.stratified_select
      with a new togglable lever: when stratified_within_class_temperature
      is set (Optional[float], default None = legacy argmin bit-identical),
      sample within each first-action class via
      softmax(-class_scores / T) before the across-class softmax step.
      Decoupled from the existing `stratified_temperature` (which controls
      across-class softmax sampling) so the A-vs-B probe can dissociate
      Layer B within-class sub-axis from across-class sub-axis. Decoupling
      rationale: collapsing both into one knob would make Q-054 sweep
      results uninterpretable -- a single-temperature dial cannot tell
      whether the lift came from within- or across-class sampling.
    Part (b) A-vs-B partial-redundancy probe lever. The autopsy's
      "config flag" requirement is satisfied by the existing independent
      master flags use_support_preserving_cem (Layer A: CEM proposal
      diversity, ARC-065 SP-CEM child) and use_e3_score_diversity
      (Layer B: E3 score-layer preservation, MECH-341). These compose
      to a complete factorial: A_only / B_only / BOTH / NEITHER. No new
      config flag is added (avoids redundancy and dead code); the
      validation experiment V3-EXQ-614c uses these flags directly to
      construct the probe grid. The amend's contribution is naming the
      probe pattern, pinning acceptance criteria, and recording the
      composition convention in substrate_queue.json amend_history.
  Config: E3ScoreDiversityConfig gains stratified_within_class_temperature
    (Optional[float], default None). REEConfig gains
    e3_diversity_stratified_within_class_temperature (Optional[float],
    default None) surfaced through REEConfig.from_dims and propagated to
    config.e3_diversity_stratified_within_class_temperature.
    build_e3_score_diversity_from_ree_config reads the new flat field via
    getattr with default None (bit-identical when the flat REEConfig field
    is absent).
  Diagnostics on E3ScoreDiversity.get_state(): three new keys
    mech341_n_within_class_sampled, mech341_last_within_class_sampled,
    mech341_last_within_class_temperature. V3-EXQ-614c reads these for
    acceptance criteria.
  Backward compatible: with default None, the new branch in
    stratified_select is short-circuited and within-class selection falls
    through to legacy argmin (bit-identical to pre-amend MECH-341).
    With use_e3_score_diversity=False (master OFF), the entire MECH-341
    block at the e3_selector.py call sites is skipped. 655/655 contracts
    (645 prior + 10 new MECH-341 amend contracts in
    tests/contracts/test_mech_341_stratified_temperature_amend.py) + 7/7
    preflight PASS with master OFF and amend OFF.
  Activation smoke (2026-06-01, verified by C2/C3/C4 contracts):
    T=1e-4 -> within-class collapses to argmin (sharpening matches legacy);
    T=1.0 -> stochastic non-deterministic within-class selection
    (n_within_class_sampled advances each call);
    T=1e4 -> uniform-within-class (all members surface across many trials).
  No trainable parameters. No phased training (pure-arithmetic regulator
  extension; same as the original MECH-341 retune). MECH-094 preserved
  by the existing simulation_mode argument on stratified_select -- the
  new branch sits AFTER the simulation_mode short-circuit, so replay
  paths never enter within-class sampling.
  Architectural choice (single-knob vs decoupled): the 616 autopsy's
    "stratified_temperature default" hint could have been read as
    repurposing the existing parameter. That reading was rejected because
    the existing `stratified_temperature` controls across-class softmax
    (line 269 of e3_score_diversity.py) with default 1.0; the V3-EXQ-611c
    retune defaults were calibrated against this exact knob, and any
    semantic shift would break the bit-identical OFF guarantee for callers
    that explicitly set it today. The decoupled-knob design preserves
    backward compat AND lets V3-EXQ-614c sweep within-class temperature
    in isolation while across-class temperature stays at the 611c
    default (1.0).
  Cross-plan impact: same amend transitively unblocks
    arc_062_rule_apprehension:GAP-B (V3-EXQ-543l successor cohort) under
    shared SD-056-amended substrate. Layer B within-class sub-axis
    contributes to ARC-062 head-input candidate-pool diversity if Q-054
    Rung 1 PASS routes via within-class lift. Flagged in V3-EXQ-614c
    queue rationale.
  Constraint preserved: NO flip of use_differentiable_cem default
    (safety note in substrate_queue.json scaffolded_sd054_onboarding /
    SD-055 entry; default OFF remains the baseline 569a/614b
    falsifiers measure against).
  Validation experiment: V3-EXQ-614c queued (4-arm sweep
    stratified_within_class_temperature in {None=legacy, 0.5, 1.0, 2.0}
    on SD-056-amended baseline; all other levers held at the V3-EXQ-614b
    config that produced ALL_ON 0.800 nats; 3 seeds; 30 P0 + 60 P1 ep;
    200 steps/ep; ~3-4 h on Mac). Acceptance: C1 ARM_LEGACY reproduces
    614b ARM_2 ALL_ON within 10% (regression guard); C2 at least one of
    {0.5, 1.0, 2.0} produces mean_selected_class_entropy_nats >= 0.800
    nats on majority of seeds (within-class lift); C3 frac_pre_ge2 > 0.3
    on majority of seeds (substrate-readiness check). PASS = C1 AND
    (C2 OR C3-only-with-no-regression). Cross-link to
    behavioral_diversity_isolation_plan.md GAP-B + arc_062_rule_apprehension:GAP-B.
  Design doc: REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md
    (new "Amend 2026-06-01" section).
  Plan-of-record: REE_assembly/evidence/planning/substrate_queue.json
    MECH-341 entry amend_history (mirroring SD-056 multi-step pattern at
    2026-05-31T11:25Z) and
    REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
    GAP-B node governance_2026_06_01.
  See MECH-341 (parent claim), failure_autopsy_V3-EXQ-616_2026-05-31
    (routing autopsy; Sections 7 + 10), failure_autopsy_V3-EXQ-614b_2026-05-31
    (corroborating cluster-member autopsy on the prior 3-arm behavioural
    falsifier; FAILed C1 structurally), ARC-065 (parent diversity-pathway
    architectural commitment), Q-054 (minimum trajectory-class diversity
    floor; the V3-EXQ-614c sweep is a Q-054 instantiation at the within-
    class sub-axis), arc_062_rule_apprehension:GAP-B (cross-plan
    beneficiary), SD-055 use_differentiable_cem (NOT flipped; safety note
    preserved), MECH-094 (simulation_mode argument; existing gate
    preserved -- new branch sits after the gate).

## MECH-341 Retune: stratified_select call-site expansion + 6-arm validation (2026-05-28)
- MECH-341 retune -- IMPLEMENTED 2026-05-28. Resolves V3-EXQ-611
  substrate-readiness FAIL (2026-05-27T13:02Z) via two parallel actions:
  (a) module-level call-site expansion in ree_core/predictors/e3_selector.py
  lines 848-870, and (b) 6-arm parameter sweep V3-EXQ-611b. The original
  MECH-341 module (e3_score_diversity.py) and its config knobs are
  unchanged -- the retune respects the implement-substrate skill's "Never
  change defaults of existing params" rule.
  Module change: stratified_select previously fired ONLY in the committed
  branch of E3TrajectorySelector.select() (where it replaced argmin); the
  uncommitted branch used multinomial sampling without any class-axis
  preservation. V3-EXQ-611 ARM_2 (OPT2 stratified_select only) measured
  mech341_n_stratified_fired=0 across all 3 seeds: the committed branch
  was never entered in the validation episodes (running_variance never
  fell below effective_threshold), so OPT2 had no opportunity to fire.
  The 2026-05-28 patch invokes stratified_select in BOTH branches: when
  the substrate is enabled AND the pool admits >=2 first-action classes,
  stratified_select replaces the corresponding fallback (argmin in the
  committed branch / multinomial in the uncommitted branch). When
  score_diversity is None or the sub-flag is False, stratified_select
  returns None and both branches fall through to their legacy selection
  rule -- bit-identical to pre-retune behaviour. MECH-094 preserved by
  simulation_mode=False kwarg at both call sites (the gate at
  E3Selector.select is only invoked from waking REEAgent.select_action).
  Architectural rationale: the substrate_queue's failure-record language
  ("substrate-natural pool diversity gets preserved through softmax-sample-
  across-class-representatives") describes a categorical-preservation
  semantic that applies regardless of commit state. The uncommitted-branch
  multinomial path provides stochasticity over ALL candidates by their
  softmax probabilities -- it can concentrate within a single first-action
  class when multiple candidates from that class score well. Stratified
  selection delivers exactly one representative per class with probability
  proportional to softmax(-best_per_class_score / temperature). This is a
  stronger class-diversity guarantee. Restricting it to the committed
  branch mismatched the substrate's intent.
  Config: NO new flags. Existing flags (use_e3_score_diversity master,
  use_e3_diversity_entropy_bonus / use_e3_diversity_stratified_select sub-
  flavours, e3_diversity_entropy_bias_scale, e3_diversity_stratified_temperature,
  e3_diversity_min_classes_for_stratification) all retain 2026-05-27 defaults.
  Backward compatible: 506/506 contracts + 7/7 preflight PASS post-edit;
  bit-identical OFF guarantee verified.
  Validation: V3-EXQ-611b 6-arm factorial (priority 250, machine_affinity
  DLAPTOP-4.local, estimated 200 min). 3 option groups (OPT1_only,
  OPT2_only, BOTH) x 2 entropy_bias_scale values (1.0, 2.0) per the
  substrate_queue retune-target sweep (0.5 dropped as too small per the
  V3-EXQ-611 gap-magnitude analysis). ARMs:
    ARM_1_OPT1_S1   entropy_bonus ON, stratified OFF, scale=1.0
    ARM_2_OPT1_S2   entropy_bonus ON, stratified OFF, scale=2.0
    ARM_3_OPT2_S1   entropy_bonus OFF, stratified ON, scale=1.0 (scale unused on OPT2)
    ARM_4_OPT2_S2   entropy_bonus OFF, stratified ON, scale=2.0 (scale unused on OPT2)
    ARM_5_BOTH_S1   entropy_bonus ON, stratified ON, scale=1.0
    ARM_6_BOTH_S2   entropy_bonus ON, stratified ON, scale=2.0
  ALL_OFF baseline anchored to V3-EXQ-611 ARM_0_ALL_OFF manifest already on
  origin/master. Acceptance: C1 (primary -- n_stratified_fired > 0 across
  all OPT2/BOTH seeds; direct test of the call-site expansion) AND (C2
  bonus scale-commensurate OR C3 selected-class diversity preserved on
  majority of seeds). Phased training matches V3-EXQ-611 budget (P0=30 ep
  warmup, P1=20 ep measurement) so per-arm comparisons are calibrated.
  experiment_purpose=diagnostic; substrate-readiness retunes do NOT weight
  claim confidence per Phase-3 governance rules (Q-054 behavioural
  falsifier is the governance-weighting signal).
  Dry-run smoke 2026-05-28T17:25Z: 6/6 arms run to completion at reduced
  scale (P0=2 ep, P1=2 ep, 30 steps/ep); C1 confirmed firing
  (n_stratified_fired > 0 in OPT2/BOTH arms); C2/C3/R2c not satisfied at
  dry-run scale (requires full P1 budget). Substrate-side validation:
  passed.
  Plan-of-record: REE_assembly/evidence/planning/substrate_queue.json
  MECH-341 entry status pending_retune -> retune_implemented_pending_validation
  with full implementation_log. behavioral_diversity_isolation_plan.md row 2
  GAP-B updated to reflect retune-implemented state.
  Concurrent-session coordination: pathspec-limited commits avoid sweeping
  IGW-008 (plan-doc row 1), IGW-010 (plan-doc row 3 + workset regen), and
  IGW-011 (plan-doc row 4) edits to behavioral_diversity_isolation_plan.md.
  My touch on row 2 (GAP-B) is disjoint from theirs.
  See MECH-341 (this claim), V3-EXQ-611 (substrate-readiness FAIL the
  retune addresses), V3-EXQ-611b (validation experiment),
  REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md
  retune section, REE_assembly/evidence/planning/substrate_queue.json
  implementation_log block, IGW-20260528-025 (workset entry routing the
  retune via /implement-substrate).

## MECH-307 Anticipatory Affect Conjunction Architecture (2026-05-11)
- MECH-307: affect.anticipatory_conjunction_architecture -- SUBSTRATE LANDED 2026-05-11.
  Goal-pipeline GAP-1 / Phase 1 (REE_assembly/evidence/planning/goal_pipeline_plan.md).
  Four-gap substrate amendment to the SD-014 valence vector + MECH-216 schema readout
  + MECH-205 PE write site. Excitement / dread emerge as derived conjunction-states
  from the existing channel set; NOT a new primitive VALENCE channel.
  Gaps 2, 3, 4 substrate-landed 2026-05-08 in the prior MECH-307 session under flags
  use_mech307_schema_multichannel / use_mech307_predicted_location_write. This pass
  resolves Gap 1 via the 2026-05-11 user-override to Option-b (split channels) over
  the design-doc default Option-a (signed single channel). The Option-a path is
  retained behind use_mech307_signed_pe for backward compat; Option-b takes
  precedence when both flags are True.
  Modules:
    ree_core/residue/field.py -- VALENCE_DIM 4 -> 6; new constants
      VALENCE_POSITIVE_SURPRISE=4 and VALENCE_NEGATIVE_SURPRISE=5 added to
      VALENCE_COMPONENTS. valence_vecs buffer auto-resizes via VALENCE_DIM
      (RBFLayer.register_buffer). evaluate_valence return shape grows
      [batch, 4] -> [batch, 6]. Indices 4-5 stay zeroed unless
      use_mech307_split_surprise=True.
    ree_core/utils/config.py -- two new REEConfig fields:
      use_mech307_split_surprise (default False) -- Option-b master.
      use_mech307_conjunction (default False) -- convenience master flag
      that __post_init__ propagates to all three substrate-side sub-flags
      (use_mech307_split_surprise + use_mech307_schema_multichannel +
      use_mech307_predicted_location_write). Path B / consumer-side
      use_mech307_consumer_conjunction_read NOT auto-set (that is a
      downstream wiring decision).
    ree_core/agent.py -- MECH-205 PE write site (line ~3527) dispatches
      between three paths: (1) Option-b split (use_mech307_split_surprise);
      (2) Option-a signed single channel (use_mech307_signed_pe legacy);
      (3) true legacy unsigned magnitude. Option-b routes surprise to
      VALENCE_POSITIVE_SURPRISE or VALENCE_NEGATIVE_SURPRISE based on
      concurrent harm_signal sign; ALSO writes magnitude to legacy
      VALENCE_SURPRISE so MECH-205 / SD-014 consumers reading the
      magnitude slot stay bit-identical to the legacy substrate.
  Data flow (Option-b under master flag ON):
    MECH-205 fires on PE > threshold -> route by harm sign:
      harm_signal <  0 -> update_valence(z_world, VALENCE_NEGATIVE_SURPRISE, ...)
      harm_signal >= 0 -> update_valence(z_world, VALENCE_POSITIVE_SURPRISE, ...)
      Plus magnitude write to VALENCE_SURPRISE (backward compat).
    MECH-216 schema readout (under multichannel flag) -> writes anticipatory
      VALENCE_LIKING + pulses z_beta arousal (Gaps 2 + 3, unchanged from
      2026-05-08 landing).
    MECH-216 write target (under predicted-location flag) -> cached e1_prior
      rather than current z_world (Gap 4, unchanged from 2026-05-08).
  Config: REEConfig.use_mech307_conjunction (bool, default False; bit-identical
    OFF). When True, __post_init__ sets use_mech307_split_surprise =
    use_mech307_schema_multichannel = use_mech307_predicted_location_write =
    True. Sub-flags can still be set independently when finer control is
    needed. Existing gain knobs (mech307_anticipatory_liking_gain,
    mech307_z_beta_schema_gain, conjunction thresholds, conjunction_gain) are
    unchanged from the 2026-05-08 landing.
  Backward compatible: all four flags default False. With defaults:
    - residue field's valence_vecs is shape [num_centers, 6] but indices 4-5
      stay zeroed (extra buffer is ~50% overhead on a small sparse buffer);
    - evaluate_valence returns [batch, 6] but only indices 0-3 carry signal;
    - PE write site falls through to the true-legacy unsigned-magnitude path;
    - all consumers reading VALENCE_SURPRISE (MECH-205 replay-priority, SD-014
      consumers via evaluate_valence) are bit-identical.
  Regression: 309/309 contracts PASS + 7/7 preflight PASS with master OFF
    (verified 2026-05-11). Existing tests/contracts/test_mech307_conjunction_contract.py
    (12 contracts covering Gaps 1-4 under their individual flags) PASSed
    unmodified -- the Option-a Gap-1 path is preserved.
  Direct field-level smoke (2026-05-11): VALENCE_DIM=6 buffer allocation,
    update_valence to VALENCE_POSITIVE_SURPRISE / VALENCE_NEGATIVE_SURPRISE
    accumulates correctly, evaluate_valence returns [1, 6], MECH-094
    hypothesis_tag=True gate respected (write skipped).
  Biological basis: 2026-05-08 lit-pull synthesis (9 entries, lit_conf 0.77)
    on excitement-as-5th-channel; NAcc-anticipation (Knutson 2001a et seq.) +
    dopamine RPE (Bromberg-Martin 2010) + habenula dread (Bromberg-Martin 2010b)
    + Adcock 2006 preplay-priority + Berridge & Robinson 2003 wanting/liking
    dissociation. Conjunction reading is more biologically faithful than a
    new channel: biology has no VALENCE_EXCITEMENT neuron type; excitement
    is the anatomical convergence of DA RPE + hippocampal preplay + ANS
    arousal at NAcc. SD-014 6-channel amendment retained as registered
    fallback if the conjunction reading fails behavioural validation.
  MECH-094: split-channel write inherits the hypothesis_tag=False gate from
    the MECH-205 / SD-014 update_valence call site; replay / simulation
    paths cannot write to either new channel.
  Phased training: not applicable (substrate is buffer-level + write-site
    routing; no learned parameters; no gradient flow).
  Validation experiment: 4-arm discriminative-pair to be queued via
    /queue-experiment in a separate session per the user 2026-05-11 directive
    "substrate only first". Acceptance criteria per anticipatory_affect_conjunction_vs_dual_channel.md
    Validation Experiment section: all-four-gaps-fixed arm produces non-zero
    cue_fires + dacc_bias + approach_commit relative to baseline; any single-
    gap-lesioned arm collapses to baseline (the conjunction-architecture
    falsifier). Fallback (per design doc): SD-014 6-channel amendment if the
    conjunction-fix does not produce the expected derived states.
  Plan-of-record: REE_assembly/evidence/planning/goal_pipeline_plan.md
    GAP-1 status open -> done 2026-05-11; Phase 2 (GAP-2 SD-049 Phase 2
    behavioural validation under MECH-307-fixed substrate) unblocked.
  Design doc: REE_assembly/docs/architecture/anticipatory_affect_conjunction_vs_dual_channel.md
  See MECH-307 (this claim), SD-014 (4-component valence vector substrate
    being extended; 6-channel amendment retained as registered fallback),
    MECH-216 (E1 predictive wanting / schema readout; Gap 2 + 4 consumer),
    MECH-205 (surprise-gated replay write path; Gap 1 write site),
    MECH-093 (z_beta modulates E3 heartbeat rate; Gap 3 downstream consumer),
    MECH-111 (curiosity / novelty drive; likely upstream-blocked by Gap 1),
    MECH-292 (ranked ghost-goal bank; downstream conjunction consumer),
    MECH-094 (hypothesis_tag gate; preserved through call-site scoping),
    MECH-295 (drive-liking-approach bridge; Path B consumer-read target;
      out-of-scope this session per "substrate only first" directive).

## MECH-307 Default-Value Recalibration (2026-05-12)
- MECH-307 default tweaks: TWO bridge-config defaults lowered after V3-EXQ-540c
  read-site probe (10x scale, 1087 bridge calls, 34784 candidate-reads) confirmed
  the V3-EXQ-540a / 540b conj_fire_rate=0 across all arms was caused by two
  config defaults sitting above the achievable substrate ceiling under the
  standard env config (CausalGridWorld with resource_respawn_on_consume=True):
    mech295_min_drive_to_fire        0.1 -> 0.01
    mech307_conjunction_z_beta_threshold  0.6 -> 0.3
  V3-EXQ-540c observations driving the change:
    drive_level: max=0.030 mean=0.016 frac>0.1=0.000 across 1087 bridge calls.
      The legacy 0.1 floor was NEVER crossed -- bridge short-circuited at
      mech295_liking_bridge.py:328-329 on every single call.
    z_beta_arousal: max=0.545 mean=0.518 frac>0.1=1.000. The legacy 0.6 floor
      sat above the achievable ceiling; ARM_default at 0.6 would never fire on
      this gate alone even if the drive gate were cleared.
    Predicate components at half-tier thresholds (0.3/0.15/0.3): all_pass=0.9466
      across 34784 candidates -- substrate writes populate read sites cleanly;
      the predicate WOULD have fired on 94.66% of reads if both gates allowed
      it through. At low-tier (0.1/0.05/0.1) and floor-tier (0.01/0.005/0.01)
      all_pass=1.000.
  Both changes are pure default-value adjustments to substrate-side config:
    ree_core/regulators/mech295_liking_bridge.py: MECH295LikingBridgeConfig
      dataclass defaults.
    ree_core/utils/config.py: REEConfig dataclass field defaults +
      REEConfig.from_dims kwarg default.
    ree_core/agent.py: getattr fallback used by REEAgent.__init__ when
      constructing the bridge config from REEConfig.
    Contract test assertions updated to match new defaults:
      tests/contracts/test_mech_295_liking_bridge.py (min_drive_to_fire == 0.01)
      tests/contracts/test_mech307_consumer_conjunction.py (z_beta default == 0.3
        + the C4 low-z_beta-blocks test uses z_beta_arousal=0.1 instead of 0.4
        to remain below the new default).
  Regression: 314/314 contracts + 7/7 preflight PASS with new defaults
    (verified 2026-05-12). The new defaults preserve the bridge's "some unmet
    need" + "some arousal" semantic while letting the bridge fire under
    realistic substrate output magnitudes.
  Backward compat: callers that explicitly set either flag to a custom value
    are unaffected (the defaults only apply when the caller omits the kwarg).
    Old 540a/540b scripts that explicitly set cfg.mech295_min_drive_to_fire=0.1
    on the cfg object after from_dims would still see the legacy behaviour;
    540e is specifically NOT overriding these to test the new defaults.
  Validation experiment: V3-EXQ-540e (3-arm decomposition under new defaults).
    Dry-run smoke 2026-05-12T06:39Z PASS at 6 ep / 1 seed: ARM_2_full
    conj_fire_rate=0.155 (>= 0.10 floor cleared at dry-run scale; ARM_0_off and
    ARM_1_split_only correctly zero). Full-scale run on DLAPTOP-4.local
    produces the definitive manifest at 3 seeds x 70 ep.
  Deferred follow-on (separate session): Option-b semantic fix at the bridge
    consumer-read site (line 343 of mech295_liking_bridge.py reads v[:, 3] =
    VALENCE_SURPRISE -- the LEGACY unsigned-magnitude channel -- rather than
    v[:, 4] = VALENCE_POSITIVE_SURPRISE under Option-b semantics; (v_s > 0.0)
    therefore loses its sign-filter intent and falsely satisfies on
    harm-paired magnitudes). Not a behavioural blocker (the v_s > 0 gate is
    cleared by Option-b's magnitude write anyway), but is a design-doc
    fidelity bug worth landing once 540e PASS confirms the architecture.
  See MECH-307, MECH-295 (bridge config owner), V3-EXQ-540c (probe diagnosis),
    V3-EXQ-540e (default-fix validation), goal_pipeline:GAP-1 (closure plan).

## SD Design Decisions Implemented (V3) — continued
- infant_substrate:GAP-1 / INF-ENV-001 — harm gradient env feature —
  IMPLEMENTED 2026-05-16. ree_core/environment/causal_grid_world.py.
  Env-only constructor kwargs (NOT REEConfig / from_dims): harm_gradient_enabled
  (default False), harm_gradient_outer_radius (3.0), harm_gradient_inner_radius
  (0.0), harm_gradient_scale (1.0). step(): when transition_type == "none" and
  nearest-hazard distance d in (inner, outer], apply
  -hazard_harm * (1 - d/r_outer)^2 * scale to harm_signal; transition_type
  harm_gradient. Terminal hazard contact unchanged. Info keys: harm_gradient_enabled,
  harm_gradient_reward_this_tick, harm_gradient_dist_to_nearest. Backward compatible:
  disabled by default; bit-identical OFF verified (test_harm_gradient_gap1.py 10/10).
  Not a learning module — no encoder, no phased training, MECH-094 N/A.
  Validation: V3-EXQ-576 PASS 20260516T195014Z (diagnostic, claim_ids=[]).
  Unblocks DEV-NEED-004 gate experiments (tier-1: V3-EXQ-587 GAP-10). See
  infant_substrate_plan.md, infant_substrate_expansion.md Section 5.1, ARC-013.

- commitment_closure:GAP-3 — CausalGridWorldV2 env extensions, primitives 1-3 —
  IMPLEMENTED 2026-05-17. ree_core/environment/causal_grid_world.py.
  Env-only constructor kwargs (NOT REEConfig / from_dims — same precedent as
  harm_gradient_* / microhabitat_* / transient_benefit_*). All master switches
  default no-op; bit-identical OFF verified suite-wide (full contract
  regression 434/434).
  Primitive 1 — adaptive tolerance-band completion: completion_tolerance_enabled
    (default False), _frac (0.0), _cells (-1; >=0 overrides frac), _metric
    ("chebyshev" | "manhattan"), _targets ("waypoint"; "waypoint+resource"
    RESERVED — raises ValueError, ships waypoint-only per Q-1a), _kernel
    ("hard" | "graded_exp"; credit exp(-d/lambda)), _lambda (1.0). Wraps the
    waypoint exact-match; OFF and frac=0.0 both dynamics bit-identical.
  Primitive 2 — counter-evidence injection hook (graded contingency
    degradation, NOT signed perturbation): counter_evidence_enabled (False),
    _interval (50), _prob (0.5), _degrade_step (0.2), _degrade_floor (0.0),
    _requires_persistent_rule (True). _inject_counter_evidence() cloned
    structurally from the SD-029 scheduled-injection pattern; lowers the
    committed target's outcome-validity toward the floor while the rule_state
    is persistent; committed-target reward scaled by validity; context
    (hazards/resources/drift) untouched. transition_type set by the existing
    waypoint path.
  Primitive 3 — dual simultaneously-active resource cue: dual_cue_enabled
    (False), _min_active_ticks (10), _replace_on_early_consume (False =
    invalidate-episode, Q-3b; True is diagnostic-only), _type_tags ((1,2)).
    Rides the SD-049 multi-resource path; RAISES ValueError if SD-049 not
    enabled (Q-3a fail-fast, no silent auto-enable).
  16 always-present info keys (inert sentinels when the relevant primitive is
  disabled). Backward compatible: disabled by default; existing experiments
  unaffected. Not a learning module — no encoder head, no phased training, no
  MECH-094 simulation-write surface.
  Validation: tests/contracts/test_env_extensions_gap3.py 14/14 (C1 bit-
  identical OFF + frac=0.0 dynamics-identical; C2 tolerance/graded_exp/metric;
  C3 counter-evidence persistent-only + monotone validity->floor +
  context-invariant; C4 dual-cue SD-049 fail-fast + accounting; C5 spec
  section-5 integration smoke) + full contract regression 434/434. NO
  claim-validation EXQ queued — spec section 5: Phase 3 is env infrastructure
  with no claim-validation EXQ (a spec-sanctioned deviation from the
  implement-substrate skill Step 8; concurrency also forbade queue edits).
  Spec: REE_assembly/evidence/planning/causalgridworldv2_env_extensions_spec.md
  (Status: IMPLEMENTED 2026-05-17). Closes commitment_closure:GAP-3 (unblocks
  GAP-8). Deliverable 4 (phased rule_state training curriculum) deliberately
  SEPARATE (spec section 6) — the SD-034/MECH-266/MECH-268 behavioural arms
  still need it. See commitment_closure_plan.md, claims SD-034 / MECH-266 /
  MECH-268. claims.yaml NOT modified (env infra unblocks but does not itself
  promote).

- commitment_closure:GAP-11 -- Phased rule_state training curriculum harness helper
  -- IMPLEMENTED 2026-05-17.
  File: experiments/committed_mode_curriculum.py (experiment-harness helper, NOT a
  ree_core substrate scheduler -- O-1 resolved).
  Public API: run_p0_warmup(), run_p1_consolidation(), run_p2_eval(),
  clone_trained_agent(), P0Result, P1Result, CommittedModeMetrics.
  Data flow: P0 trains E1+E2 on easy env (EXQ-321b run_training pattern) until
  running_variance < commit_threshold; P1 consolidates on target env until
  total_committed_steps per episode >= commitment_floor(100); P2 is frozen-policy
  eval measuring committed_steps / hold_rate / rule_state_norm.
  Mid-probe abort gate (default 60% of budget): fires commitment_not_elicited ->
  caller escalates as R1 substrate mis-calibration finding, not a tuning problem.
  O-2 mandatory contrast: every arm must run BOTH emergent (P0->P2) and
  forced-rv clone (clone_trained_agent + set rv=0.001 + run_p2_eval).
  O-3: at most ONE commitment_threshold relaxation step (threshold_relaxation param,
  max meaningful value 0.125); further non-convergence = substrate finding, escalate.
  Backward compatible: no ree_core changes; no experiment script changes.
  Generalises EXQ-321b run_training + clone_for_condition + EXQ-543b P0/P1 scaffolding.
  Blocks cleared: SD-034, MECH-266, MECH-268, MECH-090, SD-021 behavioural arms
  (V3-EXQ-460b/461/463b/464b/466b/467b/468b) -- see commitment_closure_plan.md.
  Pilot experiment: EXP-0157 / V3-EXQ-592 (GAP-11 pilot, 3 arms: EMERGENT/FORCED_RV/STARVED).
  New ID (not V3-EXQ-461b) because V3-EXQ-461 was a synthetic scripted PASS, not
  emergent training. Queued 2026-05-17. Supersedes V3-EXQ-461.

- INV-074 / MECH-333 / MECH-334: Phase-3 plasticity-injection crystallization
  + EWC residue write-protect -- IMPLEMENTED 2026-05-17.
  Files: ree_core/policy/gated_policy.py (GatedPolicy.crystallize() +
  expansion_parameters() + .crystallized; forward gains the post-crystallize
  expansion branch), ree_core/residue/field.py (ResidueField.
  snapshot_ewc_anchor() + ewc_penalty() + .ewc_anchored), experiments/
  infant_curriculum.py (InfantCurriculumScheduler on_phase3_entry fire-once
  hook), ree_core/utils/config.py (REEConfig + ResidueConfig + from_dims),
  ree_core/agent.py (GatedPolicyConfig passthrough).
  Config: REEConfig.crystallize_at_phase3 (default False; set True to enable).
  Subsidiary: gated_policy_crystallize_expansion_hidden (32),
  residue_ewc_lambda (0.0 = anchor captured, penalty inert). When
  crystallize_at_phase3=True, from_dims also arms ResidueConfig.ewc_enabled
  + ewc_lambda.
  Data flow: infant-curriculum Phase 2->3 transition -> scheduler fires the
  experiment's on_phase3_entry closure -> agent.gated_policy.crystallize()
  (requires_grad=False on head_0/head_1/discriminator; fresh plastic
  expansion MLP, last-Linear zero-init so output is bit-identical at the
  transition instant; forward = frozen_gated(x) + expansion(x.detach()),
  the .detach() blocking diversity gradient from the crystallized weights)
  + agent.residue_field.snapshot_ewc_anchor() (centers/weights anchor +
  established-basin Fisher proxy |anchor_w|*active_mask). The experiment's
  post-Phase-3 optimizer targets gated_policy.expansion_parameters() (plus
  dACC / MECH-313 / MECH-314a / MECH-320 diversity params) and adds
  residue_field.ewc_penalty() to its loss.
  Biological basis: Nikishin et al. 2023 NeurIPS plasticity injection
  (MECH-333 option E open-phase channel); Kirkpatrick et al. 2017 EWC
  write-protect (MECH-334 closure, faithful to "high resistance to
  overwriting established basins" -- NOT a hard freeze). Grounds INV-074
  (plasticity crystallization necessity), the V3-tractable subset of
  MECH-333/334, and ARC-075 (infant curriculum plasticity magnitude
  asymmetry, not just temporal scheduling).
  Pre-check (encoded in design doc): MECH-314b (uncertainty, reads
  e3._running_variance) and MECH-314c (learning-progress, EMA of
  |PE_t-PE_{t-K}| fed e3._running_variance) are forward-model-error-
  dependent -- 314c is the canonical Pathak 2017 ICM self-defeat case --
  and decay to ~0 before Phase-3 crystallization fires, so they cannot
  establish competitive weight on the expansion layer. MECH-313 (constant
  temperature), MECH-314a (residue-RBF novelty, Wittmann 2008 RPE-
  independent), MECH-320-primary (avg-reward-rate EWMA, Niv 2007), and
  dACC/MECH-260 (state-dependent recency) are F-robust and are the
  meaningful signals to route.
  Backward compatible: disabled by default; crystallize_at_phase3=False
  is bit-identical (forward never references the expansion; EWC penalty
  returns a 0.0 scalar). Contract regression 484/484 PASS; backward-compat
  543g dry-run reproduces the prior signature exactly (ARM_2=0.444,
  ARM_3=0.243, D2 FAIL). Not an encoder head -> no P0/P1/P2 phased
  training of a latent target; the experiment DOES swap the optimizer
  param-set at Phase 3 (gated_policy.parameters() -> expansion_parameters()
  + diversity params) -- flagged in the queue entry.
  MECH-094: GatedPolicy.forward()'s existing simulation_mode early-return
  precedes the expansion add, so replay/DMN paths never receive the
  expansion bias. Crystallization is a structural weight-state change
  (developmental closure, persists across episodes; reset() does NOT
  un-crystallize), not memory content -> hypothesis_tag N/A.
  Validation experiment: V3-EXQ-543h queued (2x2x2 use_gated_policy x
  use_dacc x crystallize_at_phase3; supersedes V3-EXQ-543g).
  Design doc: REE_assembly/docs/architecture/critical_period_crystallization.md.
  See INV-074, MECH-333, MECH-334, ARC-075, Q-052; arc_062 GAP-B.

## MECH-090 Commit-Entry Predicate: R-c single-gate readiness conjunction (2026-05-28)
- MECH-090 (commit-entry predicate amendment): control_plane.beta_gate.commit_entry_readiness_conjunction
  -- IMPLEMENTED 2026-05-28. Closes commitment_closure_plan.md GAP-4 at the
  substrate-readiness level (behavioural validation pending V3-EXQ-592b PASS).
  Module: ree_core/heartbeat/beta_gate.py (BetaGate.should_admit_elevation +
  __init__ kwargs use_commit_readiness_gate / commit_readiness_floor /
  commit_readiness_strict_single_candidate; get_state / reset extended with
  mech090_n_elevation_admitted / _blocked / _single_candidate /
  _last_readiness_score_margin diagnostics). ree_core/agent.py: BetaGate
  construction at REEAgent.__init__ forwards the three knobs via getattr
  fallback (config.heartbeat.use_commit_readiness_gate etc., default False)
  so the from_dims signature is unchanged. The two beta_gate.elevate() call
  sites in REEAgent.select_action (bistable branch + legacy branch) compute
  _readiness_margin and _n_candidates once from result.scores (REE
  lower-is-better -> margin = sorted(scores)[1] - sorted(scores)[0]) and
  guard the elevate call with should_admit_elevation. Bistable branch:
  gate consulted only on the not-yet-elevated transition tick. Legacy
  branch: gate consulted every committed tick (legacy semantic is per-tick
  re-evaluation); gate block treats the tick as effectively uncommitted
  (releases any prior elevation, single-stage by design).
  Reading: R-c single-gate conjunction (synthesis-strongest) per
  REE_assembly/evidence/literature/targeted_review_connectome_mech_090/
  synthesis.md (commit 9e68c5ca8a, 2026-05-28). Anchored on Cisek &
  Kalaska 2010 (affordance-competition), Hanes & Schall 1996 (FEF
  accumulator-to-threshold), Roesch / Calu / Schoenbaum 2007 (dopaminergic
  readiness signal). R-a (rv-only is correct) not defensible post-pass;
  R-b (rv-only entry + downstream propagation gate) retained as fallback
  if validation fails. Tandetnik 2021 (frontal-lesion dissociation) is
  R-b's anchor and is preserved as the fallback architecture.
  Config (HeartbeatConfig, ree_core/utils/config.py): use_commit_readiness_gate
  (bool, default False; bit-identical OFF master), commit_readiness_floor
  (float, default 0.05 -- small relative to EXQ-608 mean_top2_class_gap range
  0.27-1.96; Q-053-style calibration is a follow-on, not a precondition for
  landing), commit_readiness_strict_single_candidate (bool, default False;
  permissive single-candidate handling, strict-mode is diagnostic-only).
  All knobs NOT surfaced through REEConfig.from_dims (matches the existing
  beta_gate_bistable precedent -- callers set config.heartbeat.use_commit_readiness_gate
  directly; keeps from_dims signature unchanged to avoid concurrent-session
  conflict with the MECH-341 retune signature).
  Data flow: e3.select() -> E3SelectionResult.scores [K] + .committed bool ->
  agent.py: if result.committed and use_commit_readiness_gate:
  _readiness_margin = sorted(scores)[1] - sorted(scores)[0],
  _n_candidates = K -> bistable branch (if not is_elevated) or legacy branch
  (every committed tick) -> beta_gate.should_admit_elevation(margin,
  n_candidates) -> elevate() admitted iff margin >= floor.
  Diagnostics on BetaGate.get_state(): mech090_n_elevation_admitted /
  _blocked / _single_candidate / _last_readiness_score_margin all reset
  per-episode in BetaGate.reset(). V3-EXQ-592b reads these for the
  acceptance criteria.
  Backward compatible: use_commit_readiness_gate=False by default;
  should_admit_elevation returns True unconditionally without incrementing
  counters; agent.py skips the margin computation block entirely when the
  master flag is off (committed=True alone never enters the readiness branch).
  506/506 contracts PASS with master OFF (regression-clean 2026-05-28). 7
  unit tests on the BetaGate primitive (default no-op, gate ON admit / block,
  single-candidate permissive / strict, reset clears, backward-compat
  elevate/propagate/release) all PASS.
  MECH-094: N/A. The gate is a control-state-transition predicate at waking
  action selection; reads E3 scores; writes only the beta-elevation event,
  not memory content. No simulation-write surface. The match the SD-035 /
  MECH-279 / MECH-313 / MECH-314 / MECH-319 / MECH-320 / MECH-341 pattern.
  Phased training: N/A (pure arithmetic regulator; no learned parameters;
  no gradient flow).
  Validation experiment: V3-EXQ-592b queued as 2-arm diagnostic. ARM_0 GATED:
  use_commit_readiness_gate=True, floor=0.05, same env+seed (42) as
  V3-EXQ-592. Acceptance: total_committed_steps == 0 AND
  mech090_n_elevation_blocked >= 1 AND running_variance < commitment_threshold
  at some point during the run (confirming the gate is the load-bearing
  block). ARM_1 GATED_FORCED_READY: same gate config with experiment-side
  score_bias injection forcing margin >= 0.10 by construction. Acceptance:
  total_committed_steps > 0 AND mech090_n_elevation_admitted >= 1
  (confirming the gate does not permanently lock out commitment when
  readiness clears). Joint PASS = commitment_closure:GAP-4 partial -> done.
  Design doc: REE_assembly/docs/architecture/mech_090_commit_entry_predicate.md
  Predecessor synthesis: REE_assembly/evidence/literature/targeted_review_connectome_mech_090/synthesis.md
  See MECH-090 (parent claim), MECH-091 (urgency interrupt; orthogonal release-side
  override; unaffected), ARC-028 + MECH-105 (hippocampal-BetaGate completion
  coupling; release side; unaffected), SD-034 / MECH-266 / MECH-267 / MECH-268
  (downstream behavioural arms; transitively unblocked via GAP-4),
  commitment_closure:GAP-4 (the closure-plan gap this amendment resolves),
  Cisek & Kalaska 2010 + Hanes & Schall 1996 + Roesch / Calu / Schoenbaum 2007
  (literature anchors R1/R2/R3), Tandetnik 2021 (R-b fallback anchor),
  MECH-094 (call-site scoping; not applicable).

## MECH-090 R-c continuation: nav_competence axis (2026-05-29)
- MECH-090 R-c continuation: control_plane.beta_gate.commit_entry_readiness_
  conjunction.nav_competence -- IMPLEMENTED 2026-05-29 (commitment_closure:GAP-4
  substrate landing pass 2 of 2; behavioural validation still pending V3-EXQ-592b
  PASS). The 2026-05-28 landing implemented the WITHIN-TICK DECISIVENESS axis
  (per-candidate score margin -- Hanes & Schall 1996 reading). This pass adds
  the ACROSS-TICK MOTOR-PROGRAM READINESS axis (Cisek & Kalaska 2010 affordance-
  preparation + Roesch / Calu / Schoenbaum 2007 dopaminergic readiness). Both
  axes are R-c readings; both can be enabled/disabled independently; they
  AND-compose at both elevate sites.
  Module: ree_core/policy/commit_readiness.py (CommitReadiness +
  CommitReadinessConfig). Pure-arithmetic regulator (no nn.Module, no learned
  params), sibling pattern to MECH-313 NoiseFloor / MECH-320 TonicVigor.
  Maintains a [0, 1] readiness EMA over per-tick outcome signals plus an
  explicit notify_outcome(value) harness-push seam. Initial value 1.0
  (fail-open). MECH-094 standard simulation_mode pattern.
  Wiring (ree_core/agent.py): REEAgent.__init__ instantiates self.commit_
  readiness when config.use_commit_readiness=True (auto-armed by __post_init__ /
  from_dims OR-only resolver when use_mech090_readiness_conjunction=True).
  REEAgent.select_action computes _readiness_admits =
  commit_readiness.is_above_floor(mech090_readiness_floor) once at the top
  of the beta-gate block and AND-composes with the existing
  should_admit_elevation(score_margin, K) at BOTH call sites (bistable +
  legacy). Block diagnostics advance via commit_readiness.notify_block() at
  the source. REEAgent.reset calls commit_readiness.reset() per-episode.
  Per-tick outcome-signal source (Phase 1): the experiment harness pushes via
  commit_readiness.notify_outcome(value). The substrate-side seam is wired;
  the harness is responsible for the per-tick update. committed_mode_curriculum.py
  pushes its probe-derived nav_competence via this seam. Phase 2 follow-on
  (separate /implement-substrate pass): wire an env-emitted
  "mech090_readiness_outcome" key reading in agent.sense() so the substrate
  advances readiness automatically without harness involvement.
  Config (REEConfig + from_dims, in contrast with the prior session's
  HeartbeatConfig-resident score_margin gate flags):
  use_mech090_readiness_conjunction (bool, default False; bit-identical OFF),
  mech090_readiness_floor (float, default 0.3 -- mid-low floor that V3-EXQ-
  592 seed 42's nav_competence=0.0 clearly fails to clear; calibratable),
  use_commit_readiness (bool, default False; auto-armed True via the
  OR-only resolver when the conjunction flag is on),
  commit_readiness_window (int, default 20; informational, alpha is the
  load-bearing knob), commit_readiness_ema_alpha (float, default 0.1;
  ~10-tick half-life), commit_readiness_initial (float, default 1.0;
  fail-open).
  Backward compatible: 523/523 contracts PASS (506 prior + 17 new MECH-090
  R-c-nav-competence contracts) with both R-c master flags OFF. Master-OFF
  construction produces agent.commit_readiness=None and the agent runs
  bit-identical to pre-amendment. Master-ON with default
  commit_readiness_initial=1.0 produces readiness == 1.0 on first tick, so
  the conjunction admits while the EMA has no real outcome data (fail-open).
  The conjunction begins blocking only once notify_outcome (harness) pushes
  a low value or update drives the EMA below the floor via real outcome
  signals.
  Composition with the score_margin gate (both at both elevate sites):
    _readiness_margin = sorted(scores)[1] - sorted(scores)[0]    (existing)
    _readiness_admits = commit_readiness.is_above_floor(floor)   (NEW)
                        when use_mech090_readiness_conjunction
                        else True (legacy bit-identical)
    elevation admitted iff:
        result.committed
        AND BetaGate.should_admit_elevation(margin, K)   (existing)
        AND _readiness_admits                            (NEW)
  Phased training: N/A (pure-arithmetic regulator; no learned parameters;
  no gradient flow; no encoder head).
  MECH-094: standard simulation_mode pattern. update(simulation_mode=True)
  and notify_outcome(value, simulation_mode=True) return without advancing
  the readiness EMA. Gate decisions at waking action-selection only; the
  substrate is read-only over commit_readiness state at the elevate sites
  and writes only a control-state transition.
  Validation continuation: V3-EXQ-592b grid extended to 4 arms (ARM_2
  GATED_NAV_COMP_ON: nav_competence gate alone; ARM_3 GATED_BOTH_ON: both
  R-c gates active; ARM_4 BOTH_GATES_OFF_HARNESS_FORCES_READY: rv-only
  baseline with harness pushing notify_outcome(1.0) each tick). Falsifier
  grid: see design doc "R-c amendment continued / Falsifiability" section
  for the four orthogonal outcomes (which-axis-carries-the-load discrimination).
  Design doc: REE_assembly/docs/architecture/mech_090_commit_entry_predicate.md
  (R-c continuation section appended 2026-05-29).
  See MECH-090 (this claim's predecessor pass: within-tick decisiveness axis
  landed 2026-05-28 via BetaGate.should_admit_elevation + HeartbeatConfig
  flags), MECH-091 (urgency interrupt; orthogonal release-side override),
  ARC-028 + MECH-105 (hippocampal-BetaGate completion coupling; release side),
  SD-034 / MECH-266 / MECH-267 / MECH-268 (downstream behavioural arms;
  transitively unblocked via GAP-4), commitment_closure:GAP-4 (the closure-
  plan gap this two-pass amendment resolves), Cisek & Kalaska 2010 (across-
  tick affordance-preparation anchor), Roesch / Calu / Schoenbaum 2007
  (dopaminergic readiness anchor), MECH-313 NoiseFloor / MECH-320 TonicVigor
  (sibling pure-arithmetic regulators in ree_core/policy/), MECH-094
  (simulation_mode argument standard pattern).

## MECH-090 R-c continuation Phase-2 follow-on: env-emitted readiness-outcome source (2026-06-02)
- MECH-090 R-c continuation Phase-2: control_plane.beta_gate.commit_entry_readiness_
  conjunction.nav_competence.env_source -- IMPLEMENTED 2026-06-02. The named
  Phase-2 follow-on to the 2026-05-29 landing (the R-c continuation block above:
  "Phase 2 follow-on (separate /implement-substrate pass): wire an env-emitted
  'mech090_readiness_outcome' key reading in agent.sense() so the substrate
  advances readiness automatically without harness involvement"). Closes a real
  gap: the 2026-05-29 pass wired the CONSUMER (commit_readiness.is_above_floor
  AND-composed at both beta_gate.elevate() sites) + the notify_outcome seam, but
  left NO automatic SOURCE -- and no code path anywhere ever called notify_outcome
  (committed_mode_curriculum.py computes nav_competence but does NOT push it via
  the seam; grep-verified zero notify_outcome callers in the repo). So in any
  ecological run the readiness EMA sat pinned at its fail-open initial 1.0 and the
  across-tick nav_competence axis added no signal. This is exactly why V3-EXQ-063a
  (ARC-029 committed-mode revalidation) deliberately left the across-tick axis OFF.
  Two-part wiring (both default-OFF, bit-identical):
    (a) ENV SOURCE -- ree_core/environment/causal_grid_world.py (CausalGridWorldV2).
      New env-only kwarg mech090_readiness_outcome_enabled (default False; NOT
      surfaced through REEConfig.from_dims -- matches SD-022 / SD-029 / SD-047 /
      SD-048 / SD-049 / SD-054 env-only precedent). When True, step() emits
      info["mech090_readiness_outcome"] = clip(1.0 - mean(limb_damage), 0, 1) --
      a [0,1] motor-program-readiness / nav-competence scalar that degrades as
      limb damage accumulates (SD-022 hazard contact + scheduled injection) and
      recovers as damage heals. Requires limb_damage_enabled=True to be
      informative; with limb damage off the outcome is a constant 1.0 (fail-open).
      ABSENT-WHEN-DISABLED (not an always-present sentinel): when the kwarg is
      False the info key is simply not added, so agent.sense reads None and the
      EMA is not advanced (true bit-identical OFF, no RNG draws, no dynamics
      change). Biological reading: Cisek & Kalaska 2010 affordance-preparation
      (can the prepared motor program be executed) + Roesch/Calu/Schoenbaum 2007
      dopaminergic readiness -- the across-tick axis the within-tick score-margin
      gate cannot see.
    (b) AGENT SINK -- ree_core/agent.py REEAgent.sense() gains
      mech090_readiness_outcome: Optional[float] = None. The caller forwards the
      env-emitted info value to the NEXT sense() call (one-tick lag, biologically
      plausible -- readiness reflects the just-experienced motor outcome). When
      self.commit_readiness is not None AND the value is non-None, sense() calls
      commit_readiness.update(outcome_signal=..., simulation_mode=hypothesis_tag)
      near the end of the method, advancing the readiness EMA automatically.
      No-op when commit_readiness is None (master flags off) OR the value is None
      (key absent; CommitReadiness.update's None-sentinel returns readiness
      unchanged).
  No new module, no new config dataclass field on the agent side, no
  REEConfig.from_dims change (the env kwarg is env-only; the agent param is a
  sense() argument). The CommitReadiness module itself is UNCHANGED -- its update()
  already supported the None-sentinel and simulation_mode gate; this pass only
  wires a real source into it.
  MECH-094: sense() passes simulation_mode=hypothesis_tag into update(); a
  simulation/replay latent does not advance the EMA (the same standard pattern as
  every other regulator). Env step() is a waking observation stream.
  Phased training: N/A (pure wiring + arithmetic; no learned parameters, no
  gradient flow, no encoder head).
  Backward compatible: 719/719 contracts (700 prior + ~13 new in
  tests/contracts/test_mech090_readiness_outcome_wiring.py + adjacent) + 7/7
  preflight PASS. Env default-OFF emits no key; agent commit_readiness=None when
  master flags off; existing experiment dry-run (V3-EXQ-063a standard path)
  unchanged.
  Activation smoke (2026-06-02): aggressive all-limb scheduled-injection env
  (magnitude 0.5, heal_rate 0.001) drives mean(limb_damage) -> 1.0 ->
  mech090_readiness_outcome -> 0.0 -> readiness EMA -> 0.001 (well below the
  default floor 0.3); the env signal advances the EMA off the fail-open 1.0 and
  healing recovers it. The across-tick nav_competence axis is now exercisable in
  an ecological run for the first time.
  Contract tests: tests/contracts/test_mech090_readiness_outcome_wiring.py (C1
  env-OFF key-absent / C2 env-ON exact 1-mean(damage) + monotone + heal-recovery +
  limb-off-constant-1 / C3 sense(None) no-op / C4 sustained-low drives below floor
  + sustained-high stays above + recover-after-degrade / C5 master-OFF no-op +
  kwarg-omitted==None bit-identical / C6 MECH-094 simulation no-op).
  Validation experiment: V3-EXQ-630 (queued via /queue-experiment) -- the
  ecological across-tick ARC-029 successor to V3-EXQ-063a that exercises this
  source (592b ARM_2 GATED_NAV_COMP_ON + ARM_3 GATED_BOTH_ON arms ecologically),
  measuring that the nav_competence gate suppresses commitment as readiness
  degrades and admits as it recovers.
  Design doc: REE_assembly/docs/architecture/mech_090_commit_entry_predicate.md
  (R-c amendment continued / "Phase-2 env-source follow-on" note 2026-06-02).
  See MECH-090 (parent), commit_readiness.py (the consumer this feeds; UNCHANGED),
  the 2026-05-29 MECH-090 R-c continuation block above (the pass that wired the
  consumer + seam this completes), MECH-342 (release-side sibling; reads the same
  readiness signal via commit_readiness.get_readiness()), SD-022 (limb-damage
  substrate the env outcome reads), MECH-094 (simulation_mode standard pattern),
  ARC-029 (the committed-mode claim the validation EXQ re-evaluates).

## SD-056: E2 action-conditional divergence preservation (contrastive next-state) (2026-05-29)
- SD-056: e2.action_conditional_divergence_contrastive -- IMPLEMENTED 2026-05-29.
  Module: ree_core/predictors/e2_fast.py (E2FastPredictor.cand_world_pairwise_dist
  + E2FastPredictor.world_forward_contrastive_loss). ree_core/utils/config.py
  (E2Config + REEConfig.from_dims 4 knobs). Substrate-level fix for the
  V3-EXQ-571 root cause finding (2026-05-25): under reconstruction-shaped
  training, E2.world_forward fitted the action contribution to zero
  (cand_world_pairwise_dist = 0.0000 across K=8 candidates differing only in
  first action), collapsing per-candidate signal to every downstream consumer
  of cand_world_summaries. Same root cause as 2026-05-17 ARC-062 GAP-B autopsy;
  the GAP-B fix (gated_policy_use_first_action_onehot) was scoped only to
  GatedPolicy. SD-056 is the architecturally-faithful generalisation: fix the
  predictor's training objective rather than bypass it at each consumer.
  Algorithm: auxiliary InfoNCE-style contrastive loss on world_forward.
    For each anchor i in [K]:
      positive:  (z_world_0, a_i) -> predicted z_world_1[i]
      negatives: (z_world_0, a_j) for j != i (sibling CEM candidates)
      L_contrast_i = -log(exp(-||pred_i - target_i||^2/tau)
                        / sum_j exp(-||pred_j - target_i||^2/tau))
    L_E2 = L_recon + w_contrast * mean_i(L_contrast_i)
  Equivalent to cross-entropy over logits[i,j] = -||pred_j - target_i||^2/tau
  with label i. Asymmetric (anchor-to-prediction); symmetric would double
  cost without architectural gain. Negatives drawn from in-batch sibling CEM
  candidates (z_world_0 shared, first action differs) -- informative by
  construction, no negative-mining sweep needed.
  Scope: applies to world_forward only, not predict_next_self. z_self is not
  the collapse site (V3-EXQ-571 measured cand_world_pairwise_dist on z_world).
  world_forward signature and body unchanged; world_transition and
  world_action_encoder shapes and inits unchanged; predict_next_state,
  action_object, forward, forward_counterfactual unchanged; all downstream
  consumers (E1, E3, hippocampal, residue field) unchanged; existing
  rollout-loss machinery unchanged -- contrastive term is additive.
  Two new helpers on E2FastPredictor:
    cand_world_pairwise_dist(z_world_0, candidate_actions) -> scalar.
      Mean pairwise L2 across K predicted z_world_1 outputs. Headline
      substrate-readiness diagnostic per design memo; named by the
      2026-05-28 lit-pull SYNTHESIS verdict 3 as a methodological gap
      worth publishing as a standalone novel measurement. Accepts
      z_world_0 in [world_dim] / [1, world_dim] / [K, world_dim] forms;
      returns tensor(0.0) on K < 2.
    world_forward_contrastive_loss(z_world_0, actions, z_world_1_targets,
                                   weight=None, temperature=None,
                                   min_batch_classes=None,
                                   simulation_mode=False) -> scalar.
      Returns unweighted CE; caller multiplies by
      config.e2.e2_action_contrastive_weight before adding to L_E2 (matches
      the SD-019 / MECH-258 / MECH-273 auxiliary-helper pattern). Returns
      tensor(0.0) when (a) simulation_mode=True, (b) K < 2, or (c) fewer
      than min_batch_classes distinct first-action classes.
  Config: E2Config gains 4 fields (defaults all no-op):
    e2_action_contrastive_enabled (bool, False) -- master switch.
    e2_action_contrastive_weight (float, 0.01) -- w_contrast in L_E2.
    e2_action_contrastive_temperature (float, 0.1) -- InfoNCE tau.
    e2_action_contrastive_min_batch_classes (int, 2) -- first-action-class
      floor below which the loss returns 0 (no informative negatives).
  REEConfig.from_dims mirrors all four with the same defaults and assigns
  onto config.e2.* after the dataclass build (matches the MECH-313 /
  MECH-314 / MECH-320 wiring pattern, NOT the SD-049 env-only pattern --
  this is a regulator-style training-objective change on a learned
  substrate).
  Backward compatible: e2_action_contrastive_enabled=False by default;
  both helpers exist but are not invoked by any existing call site;
  world_forward signature and body unchanged. 539/539 contracts + 7/7
  preflight PASS with master OFF (regression-clean 2026-05-29). With
  defaults, no existing experiment's behaviour or output changes.
  Biological basis: cerebellar internal model (Tanaka et al. 2020),
  prefrontal counterfactual rollout (Miyamoto / Rushworth / Shea 2023),
  vestibular cerebellum corollary discharge (Cullen 2023) all preserve
  action-specificity at the prediction step via dedicated structural
  mechanisms. The contrastive loss enforces this same property -- actions
  must be discriminable in the predicted z_world. ML/AI engineering
  anchors: Srivastava et al. 2021 contrastive RSSM (lever B technique);
  Saanum / Dayan / Schulz 2024 PLSM (failure-mode diagnosis "lack of
  systematic representation of action effects"); Qiu et al. 2026 SWIRL
  (lever C fallback); InfoNCE temperature 0.1 standard literature value.
  Phased training: NOT required at the substrate level. Unlike encoder-
  head-on-frozen-latent patterns (EXQ-166b/c/d historical), both L_recon
  and L_contrast target the same predictor weights (world_transition +
  world_action_encoder) with compatible objectives. Joint training is the
  designed-for case.
  MECH-094: world_forward_contrastive_loss accepts simulation_mode kwarg
  and returns tensor(0.0) when True. Same defensive pattern as SD-035,
  MECH-279, MECH-313, MECH-314, MECH-319, MECH-320, MECH-341. Called only
  from waking E2 training paths; simulation_mode arg is forward-compat for
  any future replay-driven E2 training.
  Activation smoke (2026-05-29): contract test C7 confirms direction-of-
  change after 200 SGD steps on synthetic K=8 sibling batches:
  cand_world_pairwise_dist rises from random-init baseline by >= 2x or
  clears the 0.01 minimum-observable threshold. UC3 magnitude calibration
  (suggested threshold >= 0.05 in normalised units) lives in V3-EXQ-NEW-1.
  Validation experiment: V3-EXQ-NEW-1 substrate-readiness diagnostic
  (UC1-UC5 covering module surface, master-OFF backward-compat,
  cand_world_pairwise_dist direction-of-change, contrastive-task accuracy
  > 50% on held-out batch -- random baseline 1/K = 12.5% for K=8 --
  MECH-094 simulation gate). Diagnostic claim_ids=[] (substrate-readiness,
  NOT governance evidence yet). Behavioural validation V3-EXQ-569a
  (matched-entropy FP-2 falsifier on the fixed substrate; GAP-A R1.a/R1.b
  decision rule per behavioral_diversity_isolation_plan.md) queued
  separately per plan-of-record sequencing.
  Design doc: REE_assembly/docs/architecture/sd_056_e2_action_conditional_divergence.md
  Plan-of-record memo: REE_assembly/evidence/planning/e2_action_divergence_substrate_design.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_e2_forward_model_action_divergence/SYNTHESIS.md
  See SD-056 (this claim), V3-EXQ-571 (substrate-readiness FAIL the substrate
  addresses), V3-EXQ-NEW-1 (substrate-readiness validation), V3-EXQ-569a
  (behavioural falsifier on fixed substrate), ARC-062 GAP-B (tactical
  first-action one-hot bypass on GatedPolicy that SD-056 generalises),
  MECH-256 (single-pass forward-model comparator family; SD-056 sits at
  the world_forward training-objective layer), MECH-309 (logical-necessity
  diversity claim that SD-056 unblocks at the substrate level), MECH-314a /
  MECH-320 / MECH-295 / SD-033a / SD-033b (downstream bias-channel consumers
  of cand_world_summaries that recover per-candidate signal once SD-056
  lands), MECH-094 (simulation_mode argument standard pattern), SD-005
  (z_world / z_self split; substrate dependency), ARC-033 (E2_harm_s
  forward family; sibling per-stream forward predictor, not subject to this
  SD -- z_world is the collapse site, not z_harm_s).

## SD-056 multi-step rollout stability amend (2026-05-31)
- SD-056 amend: e2.action_conditional_divergence_contrastive multi-step rollout
  stability -- IMPLEMENTED 2026-05-31. Amends -- does not supersede -- the
  SD-056 t=1 substrate landed 2026-05-29.
  Module: ree_core/predictors/e2_fast.py (new helper
  E2FastPredictor.world_forward_contrastive_loss_multistep; per-step output
  norm clamp inside E2FastPredictor.rollout_with_world).
  Triggered by V3-EXQ-569e Pathway A vs B mechanism-probe autopsy 2026-05-31
  (verdict_cell=INSTRUMENTATION_FAILURE): SD-056 contrastive training produced
  numerically explosive E2 rollouts (1e16+ magnitudes) on most ON-arm seeds
  at the behavioural-runtime episode length (P1 50 ep / 200 steps). 569d t=1
  measurements at the SAME contrastive weights {0.01, 0.05, 0.20} clean
  (rollout_skipped_nonfinite=0, top2_class_gap NaN-fraction=0.0). The
  substrate is stable at its t=1 training horizon; the missing piece is
  iterated multi-step rollout stability over the full-horizon
  E2.get_world_state_sequence() consumer surface that the M1 / M3 / M4 / M5
  569e measurement channels depend on.
  Two togglable levers (both default OFF; bit-identical to pre-amend SD-056):
    Lever (a) MULTI-STEP CONTRASTIVE (PRIMARY): extends the t=1 InfoNCE
      objective to an h-step rollout horizon (Dreamer / PlaNet / Srivastava
      2021 contrastive RSSM anchor). Helper
      world_forward_contrastive_loss_multistep(z_world_0, action_sequences,
      z_world_targets, ...) returns the horizon-mean cross-entropy with the
      same MECH-094 / K<2 / min_batch_classes / simulation_mode defensive
      returns as the t=1 helper. Same caller composition pattern: caller
      multiplies by e2_action_contrastive_weight at the loss-summation site.
    Lever (b) PER-STEP OUTPUT NORM CLAMP (DEFENSIVE): inside
      rollout_with_world loop, B2-anchor clamp predicted z_world_{t+1}
      against ratio * ||z_world_0|| (NOT ratio * ||z_t||) so the bound does
      not compound across the rollout horizon. initial_z_world.detach() for
      the threshold so gradient does not flow into the anchor. Default
      ratio=2.0 matches the autopsy acceptance criterion (rollout magnitudes
      within 2x of OFF baseline).
  Config (E2Config + REEConfig.from_dims; 5 new fields):
    e2_action_contrastive_multistep_enabled (default False) -- lever (a) master.
    e2_action_contrastive_horizon (default 5) -- Dreamer-default; calibratable.
    e2_action_contrastive_horizon_weights_decay (default 1.0) -- per-step weight
      decay; 1.0 = uniform across rollout horizon.
    e2_rollout_output_norm_clamp_enabled (default False) -- lever (b) master.
    e2_rollout_output_norm_clamp_ratio (default 2.0) -- B2 anchor: max
      ||z_t|| / ||z_world_0||.
  Architectural choice: implement BOTH levers (autopsy ranks (a) primary,
  (b) tactical; not mutually exclusive per autopsy Section 9). Lever (a)
  trains to bound the substrate at the training horizon scale (architecturally
  correct fix). Lever (b) provides an inference-time hard guarantee on every
  probe tick regardless of training state or OOD probe configurations. The
  acceptance criterion (max-NaN-fraction < 0.05 + rollout magnitudes within
  2x of OFF baseline) is met by (a) on average, by (b) as a hard guarantee.
  Lever (c) consumer-side bounding in M1 metric is cosmetic per autopsy and
  is NOT implemented at the substrate level (the script-side acceptance-
  criteria fixes mentioned in autopsy Section 6 land in the post-amend
  /queue-experiment session, not here).
  Scope (NOT changed): t=1 world_forward_contrastive_loss helper unchanged;
  cand_world_pairwise_dist diagnostic unchanged; world_forward signature +
  body unchanged; world_transition / world_action_encoder shapes + inits
  unchanged; predict_next_state / predict_next_self / action_object / forward /
  forward_counterfactual unchanged; E1, E3, hippocampal module, residue field,
  all downstream consumers unchanged.
  Backward compatible: all five new fields default to no-op; 590/590
  contracts + 7/7 preflight PASS with master OFF (regression-clean 2026-05-31;
  was 580 + 10 new MECH/SD-056-amend contracts in
  tests/contracts/test_sd_056_multistep_amend.py covering A1 config defaults,
  A2 from_dims propagation, A3 helper surface + grad-flow, A4 MECH-094 gate,
  A5 K<2 short-circuit, A6 min_batch_classes floor, A7 horizon clamps
  gracefully, A8 rollout clamp OFF bit-identical, A9 clamp ON enforces B2
  bound, A10 clamp blocks NaN/Inf under stress at 200-step horizon).
  Activation smoke (2026-05-31, both levers ON, h=5, ratio=2.0,
  world_transition weights 10x-amplified, 50-step rollout): multistep loss
  4.75 with grad-norm 1073.7 on world_transition (gradient flows); max
  ||z_world_t|| = 10.91 across all 50 steps equals 2.0 * max(||z_world_0||)
  exactly (B2 bound tight); 0 NaN/Inf anywhere along the rollout. Confirms
  end-to-end wiring with both levers under deliberately-unstable training
  state.
  Phased training: NOT required at substrate level. Multi-step contrastive
  trains the same world_transition + world_action_encoder weights as
  L_recon and the existing t=1 contrastive. Joint training is the
  designed-for case (matches the 2026-05-29 SD-056 landing rationale).
  MECH-094: world_forward_contrastive_loss_multistep accepts simulation_mode
  kwarg returning tensor(0.0); same defensive pattern as the t=1 helper,
  SD-035, MECH-279, MECH-313, MECH-314, MECH-319, MECH-320, MECH-341.
  Rollout clamp is a numerical guard (bounds a forward computation, not
  memory content); not gated by MECH-094.
  Downstream beneficiaries (the load-bearing rationale for the amend):
    ARC-065 GAP-A -- behavioural diversity Pathway A vs B mechanism
      dissociation (the V3-EXQ-569e probe blocked by the instability).
    MECH-309 -- logical-necessity claim for behavioural diversity;
      downstream consumer of action-discriminability at the rollout horizon.
    MECH-341 + ARC-062 GAP-B -- per-candidate signal preservation for the
      ARC-062 gated-policy heads + lateral-PFC consumers. The t=1 path
      already works for these via 569d; multi-step consumers also need
      stability post-amend.
  569c headline reading (~2.4x C3 lift over matched-noise control) remains
  the load-bearing finding on ARC-065 GAP-A pending the amend-and-re-run
  cycle.
  Validation experiment: substrate-readiness diagnostic V3-EXQ-NEW (3-arm
  probe: SD-056-OFF baseline / multi-step ON + clamp OFF / both ON) at the
  569e-equivalent P1 budget (50 ep / 200 steps, 3 seeds). Diagnostic-purpose;
  claim_ids=[]. Acceptance: max-NaN-fraction < 0.05 across both ON arms
  AND rollout magnitudes within 2x of ARM_0 OFF baseline. Queued at the end
  of this implement-substrate session (Step 8 of the skill).
  Behavioural validation (the full 8-arm V3-EXQ-569e-equivalent Pathway A
  vs B falsifier on the amended substrate, bundled with the three script-
  side acceptance-criteria fixes from autopsy Section 6) is the next
  /queue-experiment session per autopsy Section 8 -- NOT bundled in this
  implement-substrate session per the chip's explicit scope.
  Design doc: REE_assembly/docs/architecture/sd_056_e2_action_conditional_divergence.md
  (new section "Multi-step rollout stability amend (2026-05-31)").
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-569e_2026-05-31.md
  See SD-056 (parent claim; t=1 substrate this extends), V3-EXQ-569e
  (autopsy that routed this amend session via INSTRUMENTATION_FAILURE
  verdict + Section 9 amend options ranking), V3-EXQ-569c (the load-bearing
  ARC-065 GAP-A reading the amend preserves; ~2.4x C3 lift), V3-EXQ-569d
  (sister floor-recalibrated falsifier; PASS evidence that the t=1 substrate
  is sound), ARC-065 GAP-A (the behavioural-diversity validation surface the
  amend unblocks), MECH-309 / MECH-341 / ARC-062 GAP-B (downstream multi-step
  consumers), Srivastava et al. 2021 contrastive RSSM (lit-pull anchor for
  lever (a); the t=1 SD-056 already grounds in this paper), Dreamer family
  (Hafner et al. 2019/2020; multi-step latent-dynamics training pattern),
  MECH-094 (simulation_mode argument standard pattern), SD-005 (z_world /
  z_self split; substrate dependency unchanged).

## InfantCurriculumScheduler Phase 0->1 H_pos Floor Recalibration (2026-05-31)
- experiments.infant_curriculum.H_POS_FRAC_OF_MAX default-value recalibration --
  IMPLEMENTED 2026-05-31. behavioral_diversity_isolation:GAP-C prereq (3) per
  failure_autopsy_V3-EXQ-591_2026-05-27 section 7. Module:
  ree-v3/experiments/infant_curriculum.py (experiment-harness helper, NOT a
  ree_core substrate scheduler -- lives alongside StepHarness per the file's
  own docstring; matches the existing precedent of curriculum / scheduler
  helpers being experiments/ harness modules).
  Single module-level-constant change: H_POS_FRAC_OF_MAX 0.70 -> 0.20.
  Phase 0 -> 1 advancement gate at _try_phase_0_to_1 (line 253-264) checks
  h_pos < H_POS_FRAC_OF_MAX * ln(grid_size**2); with size=12 the legacy
  threshold was 0.70 * ln(144) ~= 3.48. V3-EXQ-591 (2026-05-27, ARC-046) ran
  the scheduler with random-policy stepping across 5 seeds * 2000 episodes and
  measured observed rolling-mean H_pos band 0.03-1.08 -- the legacy threshold
  was structurally unreachable in every seed of every arm; the curriculum
  never advanced past Phase 0 across the entire validation run. New 0.20
  calibration yields threshold 0.20 * ln(144) ~= 0.99, sitting inside the
  observed band with ~9% margin at the upper end (autopsy: "probe data implies
  ~0.20 of max ~= 0.99 is reachable"). This is an INTENTIONAL non-no-op
  default change, the same exception class as the MECH-307 default-value
  recalibration (2026-05-12 -- mech295_min_drive_to_fire / mech307_conjunction_z_beta_threshold)
  and the ARC-065 SP-CEM main-path landing (2026-05-17 -- six flags flipped on
  HippocampalConfig + REEConfig.from_dims). Rationale: the legacy default sits
  above the substrate ceiling and prevents the gate from ever firing under
  realistic substrate output magnitudes; preserving it would falsely lock the
  curriculum at Phase 0 in every downstream experiment. Bit-identical opt-out
  for any caller that explicitly imports H_POS_FRAC_OF_MAX and overrides to
  0.70 (rare -- module-level constants are used as defaults, not threaded
  through caller config).
  Path (b) alternative-gate (z_goal-norm / residue-coverage replacement) per
  autopsy section 7 NOT taken in this session: z_goal collapses to ~1e-7
  across all V3-EXQ-591 arms (blocked on goal_pipeline:GAP-4 prereq (2)) and
  residue_coverage saturates to 1.0 trivially per autopsy section 7 ("degenerate
  as a discrimination criterion"). Path (a) is the only data-supported
  substrate-side fix until prereq (2) clears. Once prereq (2) lands, path (b)
  becomes architecturally viable and can be considered as a follow-on
  hardening pass.
  No new flags, no new dataclass fields, no REEConfig changes. Pure module-
  level-constant adjustment; matches the MECH-307 2026-05-12 default-tweaks
  session pattern (same one-line-equivalent change shape).
  Phased training: N/A (curriculum-harness helper; no learned parameters; no
  gradient flow; no encoder head).
  MECH-094: N/A (waking-stream curriculum scheduler driven by per-episode
  measurement-side telemetry; the scheduler has no simulation / replay
  invocation site).
  ML/AI engineering notes: standard curriculum-learning threshold-recalibration
  practice (Bengio et al. 2009 automated curriculum learning) -- thresholds
  set by initial design intent must be revised when measurement reveals the
  substrate ceiling sits below them. No additional engineering hazard.
  Contract tests: tests/contracts/test_infant_curriculum_gap9.py extended with
  3 new C11-prefix contracts -- (a) C11_h_pos_default_within_observed_band
  (regression guard: default H_POS_FRAC_OF_MAX must yield a threshold strictly
  inside the observed band [0.03, 1.08]; pre-recalibration value would fail
  this contract), (b) C11_synthetic_p0_trajectory_advances_phase (substrate-
  level smoke: a synthetic P0 trajectory at observed-band-top H_pos=1.05 must
  advance Phase 0 -> 1 at episode 100 boundary), (c) C11_synthetic_p0_trajectory_marginal_clearance
  (clearance arithmetic: H_pos at 0.99 * threshold blocks; at 1.001 * threshold
  admits; floating-point boundary). 19/19 contracts in the GAP-9 file PASS;
  593/593 full ree-v3 contracts + 7/7 preflight PASS (regression-clean
  2026-05-31).
  Backward compatible behaviour with the previous (broken) default is
  intentionally NOT preserved -- the previous default never produced a
  firing gate, so no caller can have a legitimate dependency on the
  Phase 0 lock-in.
  Downstream beneficiaries:
    GAP-C / V3-EXQ-603d / V3-EXQ-591b -- behavioural cluster validation
      (blocked_pending_substrate until both prereq (2) goal_pipeline:GAP-4
      and prereq (3) -- this -- clear; this session lands prereq (3)).
    ARC-046 / V3-EXQ-591b -- ARC-046 itself is blocked on prereq (2);
      landing (3) is necessary but not sufficient.
    infant_substrate plan (REE_assembly/evidence/planning/infant_substrate_plan.md)
      -- the underlying curriculum's ability to walk Phase 0 -> Phase 3 at
      all depends on this exit-signal being clearable.
    DEV-NEED-004 gate experiments (tier-1: V3-EXQ-587 GAP-10 etc.) --
      transitively dependent on the curriculum being walkable.
  Validation experiment: the behavioural validation gate is V3-EXQ-603d /
  591b (per autopsy section 8), which is also blocked on prereq (2) and
  therefore NOT queued in this session. Contract tests serve as the
  substrate-readiness gate at the unit level. No separate substrate-readiness
  EXQ queued; the unit tests verify the recalibration delivers a clearable
  gate at observed-band signal magnitudes, which is the substrate-level
  acceptance criterion. Behavioural validation lands as V3-EXQ-603d / 591b
  when prereq (2) is cleared by goal_pipeline:GAP-4 / V3-EXQ-490g cohort.
  Plan-of-record: REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
    GAP-C node (substrate_landed_2026_05_31 annotation added; status remains
    blocked_pending_substrate because prereq (2) is still the load-bearing
    blocker; once prereq (2) clears, prereq (3) is no longer the gate).
  Design doc: REE_assembly/docs/architecture/infant_substrate_expansion.md
    Section 6.1 Phase 0 exit condition updated with the recalibration note.
  Cross-link: IGW-20260531-009.
  See ARC-046 (parent claim; behavioural validation owner), GAP-C
    (behavioral_diversity_isolation:GAP-C closure node), MECH-307 (2026-05-12
    default-value recalibration -- same exception class as this default flip;
    sister GAP-1 prereq cleared 2026-05-15 by V3-EXQ-540g PASS), ARC-065
    SP-CEM main-path landing (2026-05-17 -- another intentional non-no-op
    default-flip precedent), SD-017 (sleep / scheduler family the curriculum
    coordinates with via on_phase3_entry hook), goal_pipeline:GAP-4 (prereq
    (2) z_goal-collapse blocker that gates the behavioural retest 591b;
    V3-EXQ-490g cohort), V3-EXQ-603a/b/c (the 603 family the recalibration
    unblocks), V3-EXQ-591 (the autopsy that named the gap), failure_autopsy_V3-EXQ-591_2026-05-27
    section 7 (the routing document for this implement-substrate session),
    behavioral_diversity_isolation_plan.md (closure-plan doc; GAP-C node
    annotated), MECH-094 (simulation gate; not applicable -- waking-stream
    scheduler).

## InfantCurriculumScheduler Phase 0->1 crossing-count criterion (V3-EXQ-591f; GAP-14 c-2) (2026-06-19)
- experiments.infant_curriculum.InfantCurriculumScheduler Phase 0->1 crossing-count
  advancement criterion -- IMPLEMENTED 2026-06-19. infant_substrate:GAP-14 defect
  (c-2) gate-over-permissiveness, resolved at the criterion level by V3-EXQ-591f PASS
  (2026-06-15; recommended_criterion=crossing_count) and wired here via
  /implement-substrate. Module: ree-v3/experiments/infant_curriculum.py
  (experiment-harness helper, NOT a ree_core substrate scheduler -- lives alongside
  StepHarness; same precedent as the 2026-05-31 H_POS_FRAC_OF_MAX recalibration).
  Two new constructor kwargs (NOT REEConfig / from_dims -- the scheduler is
  instantiated directly; matches on_phase3_entry precedent), both no-op default:
    phase_0to1_use_crossing_count (bool, default False) -- master switch.
    phase_0to1_crossing_count_min (int, default 3 = PHASE_01_CROSSING_COUNT_MIN,
      the 591f CROSSING_COUNT_MIN).
  Criterion (when ON): _try_phase_0_to_1 accumulates self._phase01_crossing_count =
  the number of post-PHASE_EP_MIN[1] (>=100) episodes whose h_pos crosses the SPIKE
  bar (H_POS_FRAC_OF_MAX * ln(grid_cells) ~= 0.994 at grid=12) and advances Phase
  0->1 once the count reaches phase_0to1_crossing_count_min. Mirrors the 591f offline
  _advance_crossing_count(seq, spike_threshold, ep_min, min_count) replay EXACTLY
  (contract test asserts online==offline on a genuine-explorer and a seed-45-like
  sequence). The no-telemetry path (h_pos is None) is UNCHANGED under both criteria
  (hard episode-count minimum alone governs -- the crossing-count gate has no signal
  to count). Crossings before ep_min are never counted (the gate returns early below
  PHASE_EP_MIN[1], matching the range(ep_min, len(seq)) replay).
  WHY crossing-count: 591f swept four sustained-level candidates (EMA-of-level,
  window-mean, EMA-with-hold, crossing-count) on the diversity-armed 591c/591d
  reachability traces; only crossing-count>=3 DISCRIMINATED -- it ADMITS genuine
  explorers (seeds 42/43/44 cross the spike bar 7/6/36x post-ep_min) and REJECTS the
  seed-45 false-advancer (one transient h_pos_max=1.453 spike but only 2 crossings).
  591d (single-episode K-of-N) and 591e (EMA-of-level@0.20, spike-vulnerable) failed
  to discriminate. The single-episode SPIKE gate (the legacy default) advanced seed 45
  on its lone spike at ep 142 = the over-permissiveness 591f fixed.
  Backward compatible: phase_0to1_use_crossing_count=False by default -> the legacy
  single-episode SPIKE gate runs (advance on the first post-ep_min crossing); the
  crossing counter is never accumulated; bit-identical to the pre-591f scheduler.
  Every existing caller (V3-EXQ-591/591b-f, 586, 610e, 669, 667 ...) constructs
  InfantCurriculumScheduler(grid_size=...) with no new kwargs -> unaffected. 27/27
  test_infant_curriculum_gap9.py contracts (19 prior + 8 new C12: default-off
  single-episode gate / requires-min-crossings / rejects-2-crossing-false-advancer /
  admits-genuine-explorer / crossings-before-ep_min-ignored / no-telemetry-hard-count
  / custom-min / matches-591f-offline-replay) + 8/8 preflight PASS.
  phase_summary() now exposes phase_0to1_use_crossing_count + phase01_crossing_count.
  Phased training: N/A (curriculum-harness helper; no learned parameters; no gradient
  flow). MECH-094: N/A (waking-stream per-episode scheduler driven by measurement-side
  telemetry; no simulation/replay invocation site). Evidence-staleness (Step 8.5): NOT
  triggered -- no-op-default flag; every existing experiment uses the default (legacy
  single-episode gate), so no dependent claim's measured mechanism changed. KEEP all
  evidence.
  GOVERNANCE: PROMOTES NOTHING. GAP-14 STAYS blocked_pending_substrate: this lands the
  (c-2) crossing-count wiring but GAP-14 also requires (c-1) seed-46 exploration-strength
  collapse to resolve (still blocked on modulatory-bias-selection-authority; Q-043/667
  magnitude sweep exhausted, 667a not yet queued). ARC-046 / DEV-NEED-008 stay
  candidate / blocked; claims.yaml NOT modified. The full curriculum-vs-flat
  EXQ-ISEF-005 (V3-EXQ-591 successor) stays blocked until BOTH legs land.
  Validation: this is the WIRING of an already-validated criterion (591f PASS is the
  evidence), not a new mechanism -- the substrate-readiness gate is the contract suite
  (online==591f-offline replay). No new EXQ queued here; the next experiment is the
  EXQ-ISEF-005 successor, which stays blocked on (c-1).
  Plan node: REE_assembly/evidence/planning/infant_substrate_plan.md
  (infant_substrate:GAP-14). 591f manifest: REE_assembly/evidence/experiments/
  v3_exq_591f_isef005_phase01_gate_criterion_20260615T115131Z_v3.json.
  See infant_substrate:GAP-14 (closure node; c-1 + c-2 legs), V3-EXQ-591f (the PASS
  this wires; recommended_criterion=crossing_count), V3-EXQ-591d/591e (the K-of-N /
  EMA-of-level candidates that failed to discriminate), the 2026-05-31
  H_POS_FRAC_OF_MAX recalibration above (prior GAP-14 prereq (c) work on the same
  gate), ARC-046 / DEV-NEED-008 (gated claims; unchanged), Q-043 / V3-EXQ-667 /
  modulatory-bias-selection-authority (the orthogonal c-1 exploration-strength thread),
  MECH-094 (N/A -- waking-stream scheduler).

## MECH-342: Maintenance-time readiness-driven commitment-release coupling (B3b) (2026-06-02)
- MECH-342: control_plane.commit_maintenance_release -- IMPLEMENTED 2026-06-02.
  Release-side complement to the MECH-090 commit-entry R-c readiness
  conjunction (which the V3-EXQ-592f autopsy + MECH-090 release-path audit +
  motor-cessation lit-pull established is ADMISSION-ONLY by design). The SAME
  two R-c readiness signals MECH-090 AND-composes to ADMIT a commitment here
  drive a graded, bounded-accumulation RELEASE of an already-elevated beta
  latch when they degrade mid-commitment. Closes the V3-EXQ-592f reach gap
  (predicates fire under forced beta-elevated state but produce zero
  state-occupancy suppression + zero decommit transitions). Routed by
  REE_assembly/evidence/planning/mech090_release_path_audit_2026-06-02.md
  (B1 ruled out: none of ARC-028/MECH-105 completion, MECH-091 urgency,
  V_s commit-release, SD-034 closure covers degraded-readiness mid-commitment)
  + targeted_review_mech_090_release_motor_cessation/SYNTHESIS.md (verdict B3b).
  Module: ree-v3/ree_core/policy/commit_maintenance_release.py
    (CommitMaintenanceRelease + CommitMaintenanceReleaseConfig). Pure-arithmetic
    regulator (no nn.Module, no learned params); sibling to commit_readiness.py
    (the admission-side R-c signal) in the policy package.
  Accumulator dynamics (per maintenance tick, only while beta is elevated):
    deficit_d = clip((score_margin_floor - score_margin)/score_margin_floor, 0, 1)
                (within-tick decisiveness axis; 0 when healthy OR no signal)
    deficit_n = clip((nav_floor - nav_competence)/nav_floor, 0, 1)
                (across-tick nav_competence axis; 0 when no signal)
    combined  = max(deficit_d, deficit_n)   (OR-composition; De Morgan dual of
                the MECH-090 AND admission; conflict-graded by the worse axis)
    if combined>0:  pressure += accumulation_rate*combined   (drift-to-bound)
    elif recovered: pressure  = max(0, pressure - leak_rate)  (reengagement)
    else:           pressure  unchanged                       (dead-band hold)
    fire = pressure >= release_bound                          (resets pressure on fire)
  On fire: beta_gate.release() + _committed_step_idx=0 + _committed_anchor_keys=None
    + e3._committed_trajectory=None (so the decommit is observable to the 592f
    probe). Release branch in REEAgent.select_action sits immediately after the
    MECH-091 urgency block (mirrors that template; DO NOT modify MECH-091).
    Reads decisiveness from self.e3.last_scores (available every tick; probe
    sets it directly) and nav from self.commit_readiness.get_readiness() (None
    -> nav axis inert, decisiveness alone drives). Pressure reset to 0 at commit
    ENTRY (both bistable + legacy elevate sites, legacy guarded on the genuine
    not-elevated->elevated transition) so each program accumulates independently.
  Config (REEConfig + from_dims, all default no-op): use_maintenance_release
    (False), maintenance_release_score_margin_floor (0.05),
    maintenance_release_score_margin_reengage (0.10),
    maintenance_release_nav_floor (0.3), maintenance_release_nav_reengage (0.5),
    maintenance_release_accumulation_rate (0.2), maintenance_release_leak_rate
    (0.1), maintenance_release_bound (1.0), maintenance_release_pressure_cap
    (1.5). Floors mirror the MECH-090 admission floors.
  BINDING DESIGN CONSTRAINTS (lit-pull B3b, vs autopsy's naive option-(a) sketch):
    (1) GRADED/ONLINE, not a one-shot Schmitt flag -- drift-to-a-release-bound,
        conflict-scaled by deficit magnitude (Resulaj 2009 bounded accumulation +
        Cavanagh/Frank 2011 conflict-graded STN threshold; Brandstaetter/Klinger
        contested-phase). A single below-floor tick does NOT release.
    (2) TARGETED + HYSTERETIC with reengagement -- releases only the active beta
        latch + committed trajectory (Falasconi/Arber 2025 movement-specific vs
        Wessel 2022 non-selective); separate reengage levels create a hysteresis
        band; recovery leaks pressure back (guards the premature-abort pole).
  Distinct from (falsifiable): MECH-090 admission predicate (entry-only, AND;
    this is maintenance, OR); MECH-091 (z_harm threat -- this fires with z_harm_a
    BELOW threshold); ARC-028 completion (options GOOD -- this fires when
    completion_signal LOW); MECH-269b/V_s (schema staleness -- this fires with a
    STABLE schema); MECH-340 ghost-goal (goal-appraisal timescale on the
    ghost-goal bank -- this is the active beta latch at motor-program timescale).
  Backward compatible: use_maintenance_release=False by default ->
    agent.maintenance_release is None; the select_action release branch is
    skipped entirely. 685/685 contracts + 7/7 preflight PASS with master OFF
    (bit-identical; was 685 before, +15 new MECH-342 contracts in
    tests/contracts/test_mech_342_commit_maintenance_release.py = 700 total).
  Activation smoke (2026-06-02): regulator unit dynamics (graded accumulation,
    fire after ~5 sustained-max-deficit ticks, ~10-11 at half deficit,
    reengagement leak, dead-band hold, OR-composition, MECH-094 sim no-op,
    config validation); agent-loop (default OFF -> None; OFF degraded ->
    reproduces 592f gap, beta stays elevated, no release; ON degraded(nav) ->
    releases at tick 4 + e3 pointer cleared + 1 fire; ON healthy -> no false
    abort).
  MECH-094: tick(simulation_mode=True) returns False without advancing pressure
    (replay must not abort a committed motor program). Match SD-035 / MECH-279 /
    MECH-313 / MECH-320 / commit_readiness pattern.
  Phased training: N/A (pure-arithmetic regulator; no learned parameters).
  Validation experiment: V3-EXQ-592g (queued via /queue-experiment) -- 592f-style
    controlled state-machine probe with use_maintenance_release=True; verifies
    state-occupancy suppression + decommit transition under sustained degraded
    readiness (zero in 592f) + no false abort under healthy readiness. The 592f
    manifest's pending_retest_after_substrate stays TRUE until 592g validates.
  Does NOT reopen the MECH-090 admission axis (V3-EXQ-592d 4-arm validator stays
    live; lit/exp decoupled -- this lit-pull does not weight MECH-090 confidence).
  Design doc: REE_assembly/docs/architecture/mech_342_commit_maintenance_release.md
  Audit + lit-pull: REE_assembly/evidence/planning/mech090_release_path_audit_2026-06-02.md,
    REE_assembly/evidence/literature/targeted_review_mech_090_release_motor_cessation/SYNTHESIS.md,
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-592f_2026-06-02.md.
  See MECH-090 (parent; commit-entry R-c admission), commit_readiness.py
    (admission-side sibling signal), MECH-091 (sibling release pathway; threat
    axis; template mirrored), ARC-028/MECH-105 (completion release; opposite
    regime), MECH-269b/MECH-284 (V_s release; schema axis), SD-034 (closure;
    rule-stability axis), MECH-340/ARC-079/Q-053 (goal-level disengagement),
    MECH-094 (simulation-mode call-site scoping).

## modulatory-bias-selection-authority: gap-relative E3.select authority (2026-06-03)
- modulatory-bias-selection-authority: ethics_engine_3.modulatory_bias_selection_authority
  -- IMPLEMENTED 2026-06-03 (substrate-readiness validation pending V3-EXQ).
  Gives the modulatory / diversity score-bias channels genuine but BOUNDED authority
  over the committed argmin at E3.select. Root cause (604a/624a/614d cluster autopsy):
  fixed small modulatory magnitudes (~0.05-0.1) added to primary scores whose
  raw_score_range is much larger never change the argmin -- 604a curiosity_bias=0.0
  every arm, 624a vigor action_density byte-identical ON==OFF, 614d within-class
  temperature -> committed-class entropy byte-identical across T=0.5/1.0/2.0.
  Approach (b) gap-relative scaling (user-confirmed AskUserQuestion 2026-06-03):
    Modules: ree_core/predictors/e3_selector.py (additive authority),
      ree_core/predictors/e3_score_diversity.py (stratified across-class normalization),
      ree_core/utils/config.py (flags).
    Site 1 (e3_selector.select): after the composed score_bias chain (dACC +
      lateral_pfc + ofc + mech295 + MECH-314 curiosity + MECH-320 vigor) and the
      MECH-341 entropy bonus are added, compute mod = scores - raw_scores and rescale
      so range(mod) == modulatory_authority_gain * raw_score_range, then
      scores = raw_scores + rescaled_mod. Takes precedence over the legacy
      normalize_score_bias_to_e3_range (skipped when this flag is on).
    Site 2 (e3_score_diversity.stratified_select): normalize class-representative
      scores to UNIT range before the stratified_temperature softmax (614d C2 fix --
      absolute class-rep gap no longer collapses committed-class selection).
  SAFETY: primary scores NOT modified -> commit-threshold / running_variance /
    softmax-temperature / urgency-interrupt / MECH-090 admission semantics unchanged.
    gain=0.5 < 1.0 keeps modulatory competitive in near-ties but subdominant when the
    primary harm/goal gap exceeds gain*range (clearly-harmful candidate stays rejected).
  Config (REEConfig + from_dims + E3Config, all default no-op / bit-identical OFF):
    use_modulatory_selection_authority (bool, False; E3Config + REEConfig top-level
      mirror so build_from_ree_config arms the stratified site),
    modulatory_authority_gain (float, 0.5),
    modulatory_authority_min_range_floor (float, 1e-6).
  Diagnostics: e3_selector.last_score_diagnostics gains modulatory_authority_active +
    modulatory_authority_scale_factor; e3_score_diversity.get_state gains
    mech341_n_authority_normalized + mech341_last_authority_normalized +
    mech341_last_rep_score_range.
  Backward compatible: 734/734 contracts + 7/7 preflight PASS with flag OFF, verified
    under two pytest-randomly orderings; bit-identical baseline. Smoke (flag ON, large
    class gap): stratified across-class restores ~24% share to the worse class vs
    OFF collapse 399/1.
  NECESSARY-BUT-NOT-SUFFICIENT for the curiosity lever: 624a/614d are pure drowning
    (fixed directly); 604a had curiosity_bias=0.0 (genuinely zero -- MECH-314a no active
    residue centers + 314b/c broadcast-by-design), so scaling zero is still zero. The
    validation EXQ must guard curiosity_bias_abs_mean > 0 before testing curiosity.
  MECH-094: pure arithmetic on the waking committed-selection path; stratified_select
    carries simulation_mode; no replay write surface. Phased training: N/A.
  Contracts: tests/contracts/test_e3_score_bias_candidate_support.py
    (test_modulatory_authority_ON_rescales_bias + _min_range_floor_prevents_degenerate_scale,
    seeded for order-robustness).
  Validation experiment: substrate-readiness diagnostic queued via /queue-experiment
    (claim_ids=[]; vigor/curiosity/within-class arms OFF vs ON + curiosity non-degeneracy
    guard + no-harm-increase). PASS unblocks the per-claim evidence retests of
    MECH-314/320/341 and the MECH-343 hypothesis.
  Design doc: REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
  Concurrency note: clearing the substrate gate during the 614d review auto-spawned
    IGW-024 for this substrate; the two sessions converged on the identical design and
    the joint working-tree implementation was landed from the interactive session
    (igw-024 stood down, empty worktree).
  See modulatory-bias-selection-authority (substrate_queue), MECH-314 / MECH-320 /
    MECH-341 (unblocked levers, stay v3_pending), Q-044 / ARC-068 (unblocked),
    MECH-343 (difficulty-gated proposal entropy; downstream), failure_autopsy_604a-624a-630_2026-06-03,
    MECH-090 (admission gate; unchanged), MECH-094 (call-site scoping).

## modulatory-bias-selection-authority AMEND: float32 catastrophic-cancellation fix (V3-EXQ-643a, 2026-06-06)
- modulatory-bias-selection-authority amend -- IMPLEMENTED 2026-06-06. Module:
  ree_core/predictors/e3_selector.py (E3TrajectorySelector.select authority block).
  Routed by the V3-EXQ-643 substrate-readiness validation FAIL (2026-06-06,
  modulatory_authority_active_frac=0.0 + scale_factor_mean=0.0 on BOTH ON arms,
  0/3 seeds). The 2026-06-03 substrate computed the combined modulatory
  contribution as `modulatory_total = scores - scores_raw` -- reconstructing it by
  SUBTRACTING two primary-magnitude score tensors. CODE-CONFIRMED ROOT CAUSE
  (corrects failure_autopsy_V3-EXQ-643_2026-06-06, whose stated root cause
  "the modulatory bias is uniform across the K candidates ... range < 1e-6" is
  NUMERICALLY WRONG): the MECH-341 entropy bonus DOES carry a real ~0.17
  cross-candidate range (it keys on first-action class, NOT z_world, so it is NOT
  uniform). The 643 harness trains SD-056 online (e2_action_contrastive) which
  drove the primary E3 scores to ~1e32 (z_world instability; the SD-056 multistep
  rollout-norm clamp is OFF in 643). At ~1e32 the float32 ULP is ~1e25, far above
  0.17, so `scores - scores_raw` collapses to EXACTLY 0.0 -> modulatory_range <
  floor -> the gate never fired. Diagnostic probe confirmed: SD-056-training-OFF
  raw_score_range ~4.5 (gate fires, modrange ~0.25, active=True); SD-056-training-ON
  raw_score_range ~1e32 (modrange 0.0, never fires). The substrate is FUNCTIONALLY
  CORRECT at normal score magnitude; 643 failed from a harness-induced numerical
  degeneracy.
  THE FIX (numerically robust; bit-identical OFF; bit-identical when scores small):
  track the combined modulatory contribution EXPLICITLY as a small accumulated
  tensor `_modulatory_accum` = (score_bias actually added) + (MECH-341 entropy
  bonus actually added), captured at the two add sites. The authority block now
  measures `modulatory_total = _modulatory_accum` (range computed from the small
  ~0.17 bias tensor, independent of primary-score magnitude) instead of
  reconstructing it by subtraction. Mathematically identical to (scores -
  scores_raw) in exact arithmetic; immune to large-score cancellation. The apply
  line `scores = scores_raw + scale_factor * modulatory_total` is unchanged. New
  diagnostic `modulatory_authority_range` on E3Selector.last_score_diagnostics
  exposes the true range the gate keyed on (0.17, not 0) for validation.
  Backward compatible: the whole block is gated on use_modulatory_selection_authority
  (default False) -> bit-identical OFF. When ON with small primary scores the change
  is mathematically identical (differs only in low-order float bits; argmin
  unchanged). 836 contracts (835 prior + 1 new regression) + 7/7 preflight PASS.
  New contract test_modulatory_authority_survives_large_primary_scores
  (tests/contracts/test_e3_score_bias_candidate_support.py): large per-candidate
  world states -> raw_score_range > 1e10, a 0.5-range bias is annihilated by the
  pre-fix subtraction but the post-fix gate reports modulatory_authority_range==0.5
  and active=True (the pre-fix code FAILS this test).
  Proof-of-fix probe (643 config + harness at reduced budget): pre-fix
  authority_active_frac=0.000 / modrange=0; post-fix authority_active_frac=1.000 /
  modrange=0.16 -- the real entropy-bonus range now reaches the gate. NOTE the
  scale_factor is ~1e27 at 1e26-magnitude scores (a VACUOUS fire on degenerate
  scores): the substrate fire is now correct, but a MEANINGFUL 643a REQUIRES the
  harness to keep primary scores bounded -- the V3-EXQ-643a re-validation enables
  the SD-056 rollout-norm clamp (e2_rollout_output_norm_clamp_enabled) and adds a
  raw_score_range readiness precondition (the run is substrate_not_ready_requeue if
  e3_raw_score_range_mean exceeds a sane bound -- a non-vacuity gate, not an
  authority verdict).
  Phased training: N/A (pure arithmetic; no learned parameters). MECH-094: N/A
  (waking committed-selection path; no replay write surface; unchanged).
  Validation experiment: V3-EXQ-643a (supersedes V3-EXQ-643) -- 643 re-run with
  SD-056 online-training numerical stability (rollout clamp) + raw_score_range
  non-vacuity precondition + the 604a curiosity non-degeneracy guard retained.
  claim_ids=[] (substrate-readiness). PASS unblocks the per-claim evidence retests
  of MECH-314/320/341. substrate_queue.ready stays FALSE until 643a clears the
  non-vacuity gate AND the authority changes selection on bounded scores.
  Design doc: REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
  (V3-EXQ-643a fix section). Autopsy correction:
  failure_autopsy_V3-EXQ-643_2026-06-06 Section 2 root cause superseded.
  See modulatory-bias-selection-authority (parent substrate; gap-relative scaling
  landed 2026-06-03), V3-EXQ-643 (the FAIL this amend addresses), V3-EXQ-643a
  (validation), MECH-341 (entropy bonus -- the modulatory channel that genuinely
  carries cross-candidate range), SD-056 (e2_action_contrastive online-training
  instability that exploded the scores; rollout clamp is the harness fix),
  MECH-314 / MECH-320 (z_world-derived channels -- uniform under cand_pairwise=0,
  but NOT the binding cause), MECH-090 (admission gate; unchanged), MECH-094 (N/A).

## modulatory-bias-selection-authority AMEND: route upstream-channel range into the bias the authority rescales (569f/661/654a, 2026-06-10)
- modulatory-bias-selection-authority route-range amend -- IMPLEMENTED 2026-06-10.
  Modules: ree_core/predictors/e3_selector.py (project_channel_range helper +
  channel_route_bias param + accumulator fold + P0 diagnostic), ree_core/agent.py
  (route source selection at the e3.select() site), ree_core/utils/config.py (flags).
  Routed by failure_autopsy_569f-661-654a_2026-06-10 (confirmed; user-adjudicated
  2026-06-10 governance cycle).
  ROOT CAUSE (one structural property, not three bugs): the 2026-06-03/06-06 authority
  rescales _modulatory_accum (the composed score_bias chain + MECH-341 bonus). The
  569f/661/654a cluster showed that a channel whose REPRESENTATION carries genuine
  cross-candidate range (569f consumed world-summary spread 0.196; 654a minted
  rule_state 268/549/220; 661 coherence JOINT 1.0/ALT 0.25/SHUF 0.0) still does NOT
  move committed action -- because that range is flattened by the consuming bias head
  (e.g. the SD-033a/b zeroed-last-layer heads) before it reaches _modulatory_accum, so
  the authority has nothing to amplify (569f selected-action entropy bit-identical
  0.549141 across e2wf / proposer / matched-noise). V3-EXQ-643 established "no range ->
  no authority"; this cluster extends it one link: the channel range must be ROUTED
  into the per-candidate modulatory bias the authority rescales, not merely exist in
  the representation.
  THE FIX (parameter-free; no-op default; bit-identical OFF):
    (1) project_channel_range(features) in e3_selector.py -- a deterministic,
      range-preserving projection of a channel's per-candidate representation into a
      per-candidate scalar bias [K]. For [K, D] (e.g. cand_world_summaries): center
      across the K candidates, project onto the leading right-singular vector of the
      centered matrix -> [K] signed scalar capturing the dominant cross-candidate
      variation (SVD on a detached copy; numerical fallback to the mean-deviation
      axis). For [K] (an already-per-candidate bias): identity. K<2 / flat input ->
      zeroed (below-floor) vector. The singular-vector sign is arbitrary: routing
      makes the channel range REACH and MOVE the committed argmax (the readiness
      property), NOT necessarily move it BENEFICIALLY -- that is the channel's own
      trained head (the separate per-claim evidence retest).
    (2) E3TrajectorySelector.select() gains channel_route_bias: Optional[Tensor] [K].
      When use_modulatory_channel_routing is on and it is supplied, it is normalised
      to unit zero-mean range (so the contribution stays bounded even when the
      authority is OFF; the authority re-normalises the combined accumulator to
      gain*raw_score_range regardless), scaled by modulatory_channel_route_weight, and
      folded into BOTH scores and _modulatory_accum BEFORE the authority's range
      computation -- so the channel's range reaches the bias term the authority
      rescales. New P0 diagnostics on last_score_diagnostics:
      modulatory_channel_route_range (the RAW cross-candidate range of the routed bias,
      pre-normalise/pre-rescale -- the gate signal) + modulatory_channel_route_active.
    (3) REEAgent.select_action builds channel_route_bias from the channel-under-test's
      per-candidate representation (project_channel_range) and passes it. Source
      selectable via modulatory_channel_route_source: "cand_world_summary" (the
      [K, world_dim] world-summary channel -- 569f cluster lead, the genuine
      projection case, sourced from cand_world_summaries / the ARC-065 GAP-A helper) /
      "curiosity" / "gated_policy" / "mech295" / "coherence" (each an already-computed
      per-candidate [K] bias, identity-routed; the MECH-294 compose bias stashed as
      _bdc_coherence). Default "none" -> channel_route_bias=None -> bit-identical.
  P0 readiness gate: modulatory_channel_route_range lets a retest assert the
  modulatory bias ITSELF carries cross-candidate range derived from the channel under
  test BEFORE any behavioural falsifier is scored (so an unrouted channel cannot
  self-route a false negative -- the autopsy's explicit requirement).
  Config (E3Config + REEConfig mirror + from_dims; all no-op default, bit-identical OFF):
    use_modulatory_channel_routing (False, master; E3Config + REEConfig mirror),
    modulatory_channel_route_min_range_floor (1e-6; E3Config, the P0 gate floor),
    modulatory_channel_route_weight (1.0; E3Config, routed-vs-legacy proportion in the
    accumulator), modulatory_channel_route_source (str "none"; REEConfig, agent source
    select). All wired through REEConfig.from_dims.
  HONEST SCOPE: routing makes the channel range reach + move the committed argmax (the
  P0 property). For COHERENCE specifically (661): currency_coherence is a SCALAR
  (uniform across candidates) -- its per-candidate range lives only in the compose
  cosine; routing the compose [K] bias gives it range, but joint-vs-alternation
  MODE-DISCRIMINATION (different per-candidate PATTERN, not just magnitude) is a
  MECH-294-side concern, out of scope here. A scalar gate is rescale-invisible by
  construction; the P0 gate correctly flags a channel that carries no cross-candidate
  range as substrate_not_ready.
  Backward compatible: use_modulatory_channel_routing=False by default -> the routing
  block is skipped, channel_route_bias is None, _modulatory_accum + authority unchanged
  -> bit-identical. 989 contracts (985 prior + 4 new in
  tests/contracts/test_e3_score_bias_candidate_support.py: project_channel_range
  range-preserve/identity/degenerate; routing-OFF bit-identical; routing-ON P0 range +
  scores-reach-rescaled-accumulator; below-floor inactive) + 7/7 preflight PASS.
  Activation smoke 2026-06-10 (StepHarness, world-summary source, authority ON):
  default == explicit-OFF bit-identical; routing-ON modulatory_channel_route_range
  0.04-0.11 (> floor), modulatory_channel_route_active True, committed argmax moves vs
  OFF (the 569f bit-identical-entropy washout broken).
  Phased training: N/A (pure arithmetic on the waking committed-selection path; no
  learned parameters). MECH-094: N/A (no replay/memory write surface; the SVD read is
  on a detached copy). Evidence-staleness (Step 8.5): NOT triggered -- no-op-default
  flag; every existing experiment uses the default (routing off), so no dependent
  claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: this amend resolves NO dependency on its own (validation pending). The
  unblocked claims (ARC-065 / MECH-294 / ARC-062 / MECH-309 / MECH-341 + the existing
  MECH-314/320/Q-044/etc.) stay candidate / v3_pending / pending_retest_after_substrate;
  claims.yaml NOT modified.
  Validation experiment: V3-EXQ-663 substrate-readiness diagnostic (claim_ids=[];
  ARM_0 routing OFF vs ARM_1 routing ON, both authority ON + e2_world_forward + SD-056
  online). Acceptance: READINESS (load-bearing, RANGE) ARM_1 route_range > floor on
  >=2/3 seeds; C1 (load-bearing, same range stat) ARM_1 active + ARM_0 inactive;
  C2 (secondary, behavioural reach) committed-class TV ARM_1-vs-ARM_0 > floor.
  PASS = READINESS AND C1. Below-floor -> substrate_not_ready_requeue. Dry-run runs
  end-to-end (self-routes substrate_not_ready at toy P0, as designed). PASS unblocks
  the SEPARATE per-claim behavioural retests (NOT queued here).
  Design doc: REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
  (route-range amend section). Autopsy: REE_assembly/evidence/planning/failure_autopsy_569f-661-654a_2026-06-10.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (modulatory-bias-selection-authority entry; +3 failure records 569f/661/654a applied
  by the 2026-06-10 governance cycle).
  See modulatory-bias-selection-authority (parent substrate; gap-relative scaling
  2026-06-03 + 643a float32 fix 2026-06-06), V3-EXQ-643 (predecessor: no range -> no
  authority), V3-EXQ-569f/661/654a (the cluster this amend addresses), ARC-065 GAP-A
  (the cand_world_summary e2_world_forward source the world-summary route reads),
  MECH-341 (entropy bonus -- a channel that already carries range in the accumulator),
  ARC-062/MECH-309 GAP-D (trainable rule_bias_head -- the beneficial-selection half for
  the rule_state behavioural retest), MECH-294 compose-coherence (the scalar-gate
  channel the P0 gate correctly flags), V3-EXQ-663 (validation), MECH-090 (admission
  gate; unchanged), MECH-094 (N/A).

## modulatory-bias-selection-authority AMEND: gain/contrast + shortlist-then-modulate conversion (569g/682, 2026-06-15)
- modulatory-bias-selection-authority CONVERSION amend -- IMPLEMENTED 2026-06-15.
  Modules: ree_core/predictors/e3_selector.py (authority block normalize_basis
  branch + shortlist-then-modulate selection block + 3 diagnostics keys),
  ree_core/utils/config.py (3 E3Config fields + from_dims passthrough). Routed by
  the confirmed failure_autopsy_V3-EXQ-569g_2026-06-14 (CORRECTED) + the GAP-A node's
  682-gated route, after V3-EXQ-682 LANDED PASS 2026-06-15 (no_collapse_reproduced).
  GATE CHECK: 682 (the in-arm route-range collapse diagnostic, claim_ids=[]) confirmed
  the 06-10 route-range amend SOLVED REACH -- ARM_1 applies in-arm
  modulatory_channel_route_range ~0.20 at the live select tick, summaries_none_frac 0.0,
  and ALL FOUR collapse causes (cause_i live summary re-collapse / cause_ii
  project_channel_range / cause_iii wiring / cause_iv applied-zero) RULED OUT, including
  on seed 43 (the seed where 569g's CONVERSION failed). So there is NO residual upstream
  re-collapse to fix -> proceed directly to the conversion amend (the clean Branch A).
  ROOT CAUSE (569g, the residual one link past route-range): the gap-relative ADDITIVE
  authority at gain 0.5 (modrange = 0.5*raw_score_range) is subdominant to the
  F-dominated primary (88-89% of E3 variance, V3-EXQ-571), so the routed range flips
  only near-tie OUTLIERS, not near-decisive winners (569g committed entropy 1/3 seeds
  strict-above a temperature-matched control; 662 TV>0 but no entropy lift). Upstream
  magnitude sweeps (667/640a byte-identical) cannot fix it -- the authority
  range-renormalizes its input -- so only authority GAIN and arbitration STRUCTURE are
  live levers.
  TWO no-op-default conversion levers (bit-identical OFF):
    (a) GAIN/CONTRAST NORMALIZATION -- E3Config.modulatory_authority_normalize_basis:
      "range" (default, bit-identical legacy: target = gain*raw_score_range, scaled by
      the modulatory RANGE -- outlier-sensitive, near-tie-only) vs "std" (target =
      gain*raw_score_STD, scaled by the modulatory STD -- robust to outliers, anchoring
      authority to the TYPICAL primary spread so the structured channel competes against
      NEAR-DECISIVE candidates). modulatory_authority_gain stays sweepable. SAFETY NOTE:
      additive gain >= 1.0 breaks the safety bound (modulatory can override a
      clearly-harmful rejection); keep gain < 1.0 on the additive path for the shipped
      config, or use lever (b).
    (b) SHORTLIST-THEN-MODULATE -- use_modulatory_shortlist_then_modulate +
      modulatory_shortlist_margin (0.25): F (raw primary scores) filters to a near-tie
      set (candidates within margin*raw_score_range of the best raw score), then the
      modulatory channel (_modulatory_accum) ARBITRATES the winner WITHIN that set (argmin
      committed / softmax-sampled uncommitted). The structured channel is load-bearing
      WITHOUT out-magnitude-ing F, and SAFETY is preserved at any internal strength
      (clearly-harmful candidates outside the shortlist are never selectable). Takes
      precedence over the additive-authority rescale + argmin/stratified selection at the
      selection site when enabled.
  Diagnostics on last_score_diagnostics: modulatory_authority_normalize_basis,
  modulatory_shortlist_active, modulatory_shortlist_size (the near-tie-set size; the
  readiness EXQ reads these).
  Backward compatible: basis="range" + shortlist OFF by default -> bit-identical to the
  pre-conversion-amend authority path. 1036 contracts + 7/7 preflight PASS; 4 new
  contracts in tests/contracts/test_e3_score_bias_candidate_support.py (conversion-amend
  OFF bit-identical / std-basis distinct scale_factor / shortlist restricts to the F
  near-tie set + safety: worst-primary never selected even with overwhelming pull /
  shortlist arbitrates by the modulatory channel within the set). Bit-identical OFF
  verified (default == explicit-range/shortlist-off action stream, atol 1e-12).
  Phased training: N/A (pure-arithmetic selection rule; no learned parameters). MECH-094:
  N/A (waking committed-selection path; no replay write surface; unchanged).
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flags; every existing
  experiment uses the defaults, so no dependent claim's measured mechanism changed.
  KEEP all evidence.
  GOVERNANCE: ARC-065 / MECH-341 / ARC-062 / MECH-309 / MECH-294 NEITHER promoted NOR
  weakened; ARC-065 stays provisional / non_contributory / pending_retest_after_substrate;
  MECH-341 untouched. claims.yaml NOT modified (substrate-only amend; the amend record
  lands in substrate_queue.json + the GAP-A plan node).
  Validation experiment: V3-EXQ-684 claim-free readiness sweep (gain x {range,std} x
  shortlist vs a VERIFIED-LIFTING matched-noise control; committed-action entropy must
  MOVE with channel range AND beat the control, not merely reach the accumulator -- 569g's
  ARM_2 temperature control under-lifted, so the sweep first verifies the noise control
  actually raises entropy). On PASS, V3-EXQ-569h (the GAP-A falsifier with an IN-ARM
  applied-route-range non-vacuity gate) is queued with the winning conversion config --
  it is GATED on the readiness sweep landing, NOT pre-queued on a guessed config.
  Design doc: REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
  (CONVERSION amend section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-569g_2026-06-14.{md,json}.
  Diagnostic: REE_assembly/evidence/experiments/v3_exq_682_gapa_inarm_routerange_collapse_diagnostic_20260615T032040Z_v3.json.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (modulatory-bias-selection-authority entry, current_pending_amend).
  Closure node: REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
  (behavioral_diversity_isolation:GAP-A).
  See modulatory-bias-selection-authority (parent substrate; gap-relative 2026-06-03 +
  643a float32 fix 2026-06-06 + route-range 2026-06-10 above), V3-EXQ-569g (the FAIL this
  amend addresses), V3-EXQ-682 (the in-arm diagnostic that cleared the gate), V3-EXQ-571
  (F-dominated primary 88-89%), ARC-065 GAP-A (closure node), MECH-341 / ARC-062 /
  MECH-309 / MECH-294 (one shared conversion fix; xref), V3-EXQ-684 (readiness validation)
  / V3-EXQ-569h (GAP-A falsifier, gated), MECH-090 (admission gate; unchanged), MECH-094 (N/A).

## modulatory-bias-selection-authority AMEND: TOP-K shortlist mode (569h conversion-ceiling, 2026-06-16)
- modulatory-bias-selection-authority TOP-K shortlist amend -- IMPLEMENTED 2026-06-16.
  Modules: ree_core/predictors/e3_selector.py (shortlist eligible-set construction +
  1 diagnostic key), ree_core/utils/config.py (2 E3Config fields + from_dims passthrough).
  Routed by the active failure_autopsy_V3-EXQ-569h_2026-06-16 (which hands off
  /implement-substrate amend modulatory-bias-selection-authority) + critical_path_synthesis
  Path 2 (shortlist-then-modulate as the real architectural change, not another gain tweak).
  behavioral_diversity_isolation:GAP-A.
  ROOT CAUSE (the 7th GAP-A amend; both prior conversion levers left ~0 behavioural
  conversion): V3-EXQ-569h FAIL/non_contributory 2026-06-16T10:11Z
  (conversion_ceiling_persists_despite_routing) -- readiness fully MET (route_range 0.31
  3/3, e2-divergence 3/3, non_degenerate, negative control clean) but the additive
  ARM_STD_G2 lever (normalize_basis=std + authority_gain=2.0) cleared C_R1B (selected-action
  entropy strict-above BOTH matched-noise and proposer) on only 1/3 seeds (need 2/3). The
  pre-existing MARGIN shortlist (the 2026-06-15 conversion amend lever (b)) ALSO failed:
  V3-EXQ-684 ARM_SHORTLIST (margin 0.25) converted 0/3, committed entropy 0.337 BELOW the
  collapsed proposer 0.549. Manifest diagnosis: modulatory_shortlist_size_mean 6.25-8.54 --
  the margin cutoff (best_raw + 0.25*raw_score_range) admitted ~7 of ~8 candidates = a
  near-WHOLE, state-STABLE eligible set, so the committed-branch argmin(_modulatory_accum)
  collapsed to the modulatory channel's GLOBAL favourite (monostrategy). The additive arms
  (which blend F) preserved MORE diversity than the margin-shortlist that let modulation
  solely decide over a near-whole set.
  THE FIX (no-op-default; bit-identical OFF): a TOP-K shortlist mode. New E3Config levers
  (mirrored on REEConfig.from_dims): modulatory_shortlist_mode "margin" (default, legacy
  bit-identical) | "top_k", and modulatory_shortlist_k (default 3). In "top_k" the eligible
  set is the k F-best candidates by PRIMARY score (k smallest raw_scores; lower-is-better,
  via torch.topk(raw_scores, k, largest=False); k clamped to [1, K]); the within-set
  selection rule is UNCHANGED (deterministic argmin of the routed _modulatory_accum when
  committed / softmax-multinomial when uncommitted). WHY top-k converts where margin
  collapsed: a SMALL fixed k gives an eligible set whose MEMBERSHIP rotates with state (the
  k F-best change as the agent moves), so argmin-of-the-routed-channel within the rotating
  small set produces committed-action diversity that reflects genuine per-candidate
  structure -- and BEATS unstructured matched-noise (the C_R1B non-vacuity bar; entropy from
  channel STRUCTURE, not sampling noise, because the within-set rule is deterministic).
  SAFETY preserved at any internal strength: only the k F-best are eligible, so a
  clearly-harmful candidate is never selectable (contract-verified). Diagnostic
  modulatory_shortlist_mode added to last_score_diagnostics (pre-seed + active site).
  Backward compatible: modulatory_shortlist_mode="margin" by default AND the entire
  shortlist block is gated on use_modulatory_shortlist_then_modulate (default False), so
  the mode is inert for every existing experiment -> bit-identical. 22/22 conversion +
  shortlist contracts PASS (18 prior + 4 new top-k in
  tests/contracts/test_e3_score_bias_candidate_support.py: default-mode-margin + shortlist-OFF
  bit-identical / restricts-to-exactly-k + safety-worst-primary-never-selected / top_k set
  SMALLER than a loose-margin set on the same pool [the 569h fix made deterministic] /
  within-top-k routed-modulatory argmin decides the winner).
  Phased training: N/A (pure-arithmetic selection rule; no learned parameters; no gradient
  flow). MECH-094: N/A (waking committed-selection path; no replay/memory write surface;
  unchanged). Evidence-staleness (Step 8.5): NOT triggered -- no-op-default levers; every
  existing experiment uses mode='margin' + shortlist OFF, so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  GOVERNANCE: ARC-065 NEITHER promoted NOR weakened; stays provisional / substrate_ceiling /
  pending_retest_after_substrate; MECH-341 / ARC-062 / MECH-309 / MECH-294 untouched.
  claims.yaml NOT modified (substrate-only amend; record lands in substrate_queue.json +
  the GAP-A plan node).
  Validation experiment: V3-EXQ-569i (queued via /queue-experiment) -- a 569-lineage
  successor (3-arm matched-entropy, in-arm route-range gate + e2-divergence gate + C_R1B
  strict-above BOTH matched-noise AND proposer) arming the TOP-K shortlist config as the
  conversion constant across arms (use_modulatory_shortlist_then_modulate + mode=top_k +
  use_modulatory_channel_routing + candidate_summary_source=e2_world_forward). Pre-registers
  the conversion-ceiling off-ramp (readiness met + no C_R1B lift -> conversion_ceiling_persists,
  non_contributory, NOT an ARC-065 falsification). claim_ids=[ARC-065].
  Design doc: REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
  (TOP-K shortlist amend section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-569h_2026-06-16.* (concurrent session).
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (modulatory-bias-selection-authority entry).
  Closure node: REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
  (behavioral_diversity_isolation:GAP-A).
  See modulatory-bias-selection-authority (parent; gap-relative 2026-06-03 + 643a float32 fix
  2026-06-06 + route-range 2026-06-10 + gain/contrast+margin-shortlist 2026-06-15 above),
  V3-EXQ-569h (the FAIL this amend addresses), V3-EXQ-684 (the margin-shortlist 0/3 +
  size 6.25-8.54 diagnosis), V3-EXQ-571 (F-dominated primary 88-89%), ARC-065 GAP-A
  (closure node), MECH-341 / ARC-062 / MECH-309 / MECH-294 (shared conversion substrate; xref),
  V3-EXQ-569i (validation/falsifier), MECH-090 (admission gate; unchanged), MECH-094 (N/A).

## MECH-439: F-dominance conflict-grade -- Factor A conflict-graded shortlist width + Factor B gap-scaled commit-T (2026-06-18)
- MECH-439 conflict-grade levers: ethics_engine_3.f_dominance_conflict_grade -- IMPLEMENTED
  2026-06-18 (substrate; MECH-439 stays candidate -- this PROMOTES NOTHING. A PASS moves it
  toward supports; a preconditions-met FAIL routes to the synthesis-doc V4 directions, NOT a
  dead end). The conversion-ceiling campaign's V3 fix for the live root (F-dominance). Routed
  by REE_assembly/evidence/planning/conversion_ceiling_phase0_synthesis_2026-06-18.md (Phase 0
  four-root + Phase 1 fork-resolution + Phase 1 backfill).
  ROOT (V3-EXQ-571): the primary harm/goal score F carries ~88-89%% of E3 committed-selection
  variance, UNCHANGED by the full diversity stack, so every diversity channel reaches the E3
  accumulator but cannot move the F-dominated committed argmax. The standalone 569i top-k PASS
  (2/3 seeds) only thinly cleared (0.711 vs proposer 0.650). Phase 1 resolved the fork toward
  STRUCTURAL conflict-grading (NOT F-variance rebalancing -- the rebalance lever = std-basis
  additive authority already FAILED 569h 1/3; the structural bound = top-k shortlist PASSED
  569i 2/3). The backfill found the two V3-tractable levers are TWO RENDERINGS OF ONE PRINCIPLE
  -- the BG hyperdirect conflict-grade: grade the committed decision by the normalized top-F
  gap -- and should be tested TOGETHER as a 2-factor experiment.
  Module: ree_core/predictors/e3_selector.py (E3TrajectorySelector.select +
  _gap_scaled_commit_pick helper + 6 diagnostics keys), ree_core/utils/config.py (E3Config 5
  fields + REEConfig.from_dims). One shared quantity gap_norm in [0,1] computed once from
  raw_scores (the F-best to F-second-best gap / raw_score_range; None when < 2 candidates ->
  both factors no-op) drives BOTH levers.
  FACTOR A (conflict-graded shortlist width; the safety-hard primary). The existing top_k
  shortlist block (mode='top_k') gets, under modulatory_shortlist_conflict_graded, a graded k:
  k = clamp(round(k_max - (k_max-1)*gap_norm), 1, K). Near-ties (gap_norm ~ 0) -> k = k_max
  (wider eligible set / slower commit, the STN threshold-raise); a decisive F-gap (~1) -> k = 1
  (fast commit on the F-winner). F gates ELIGIBILITY only; it is ABSENT from the within-set
  arbitration (the routed modulatory channel argmin/sample picks inside the eligible set).
  SAFETY: because the eligible set is the k F-best, a clearly-harmful candidate (large F-gap
  above the best) is never admitted. Flags: modulatory_shortlist_conflict_graded (False) +
  modulatory_shortlist_k_max (6). Default False -> the fixed modulatory_shortlist_k path is
  bit-identical.
  FACTOR B (gap-scaled entropy-regularized commit; the complement). The committed selection --
  otherwise a HARD argmin over the F-dominated scores (or the routed modulatory channel within
  a shortlist) -- becomes, under use_gap_scaled_commit_temperature, multinomial(softmax(-q /
  T_eff)) over the eligible set, with T_eff = base_temperature + gap_scaled_commit_entropy_alpha
  * (1 - gap_norm). Near-ties -> hotter (softer argmax); a decisive gap -> cold (T_eff -> base,
  preserves the decisive F-winner). q is the routed modulatory channel within an active Factor-A
  shortlist (F-bounded eligible set = the safety guarantee), else the F-dominated scores
  restricted to an F-eligibility envelope (candidates within gap_scaled_commit_harm_floor *
  raw_score_range of the best raw score) so a hot commit-T in a near-tie can NEVER softmax-
  promote a clearly-harmful candidate -- the SAFETY GATE the backfill flagged. This softens the
  COMMITTED hard-argmin (where the monostrategy lives); distinct from MECH-313 tonic-noise
  (gap-blind, pre-select) and from the existing uncommitted multinomial. A FLAT (gap-blind)
  commit-T reduces to the 569g temperature control that under-lifted -- the (1 - gap_norm)
  gap-scaling is LOAD-BEARING (asserted by contract). Flags: use_gap_scaled_commit_temperature
  (False) + gap_scaled_commit_entropy_alpha (1.0) + gap_scaled_commit_harm_floor (0.25). Default
  False -> the hard argmin path is bit-identical.
  Backward compatible: both factors default no-op -> the legacy fixed-k top_k argmin path is
  bit-identical (verified). Contracts: tests/contracts/test_e3_conflict_graded_conversion.py
  (12: OFF bit-identical to legacy fixed-k argmin + default flags; Factor-A width clamps to
  k_max at near-tie + to 1 at decisive gap + monotone non-increasing in gap + within [1,K];
  F absent from Factor-A within-set arbitration; Factor-A safety -- clearly-harmful never
  admitted; Factor-B T_eff monotone + gap-scaling load-bearing; Factor-B cold preserves the
  decisive winner + hot softens a near-tie; Factor-B standalone harm-floor safety gate excludes
  a harmful candidate even when its composed score is lowest; Factor-B standalone OFF == hard
  argmin). Preflight 7/7. Full contract suite green.
  Phased training: N/A (pure-arithmetic selection rule; no learned parameters; no gradient
  flow). MECH-094: N/A (waking committed-selection path; no replay/memory write surface).
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default levers; every existing
  experiment uses the defaults, so no dependent claim's measured mechanism changed. KEEP all
  evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-439 stays candidate; ARC-065 / MECH-341 / ARC-062 /
  MECH-309 / MECH-294 untouched. claims.yaml NOT modified (substrate-only amend).
  Validation experiment: V3-EXQ-689 (2-factor 2x2 discriminating falsifier on the GAP-A-ready
  foraging substrate; trained e2.world_forward via SD-056 + ARC-065 GAP-A
  candidate_summary_source=e2_world_forward so the eligible set is genuinely diverse -- the
  non-vacuity precondition; vacuous top_k over a class-uniform pool self-routes
  substrate_not_ready_requeue, NEVER a false weakens). 2x2: Factor A (fixed-k=3 vs conflict-
  graded k) x Factor B (hard argmin vs gap-scaled commit-T), CRF stack constant. PRIMARY:
  committed_action_class_entropy strict-above BOTH collapsed-proposer and matched-noise controls
  on >=2/3 seeds. LOAD-BEARING PRE-REGISTERED FALSIFIER (shared by both factors): bin ticks by
  top-F-gap and regress committed entropy on gap -- the lift MUST correlate with per-tick F-gap;
  uniform lift = the grading adds nothing over a bigger fixed shortlist / hotter flat softmax.
  Non-vacuity self-route: k AND T_eff actually vary across ticks; eligible set diverse.
  claim_ids=[MECH-439] (this is its first falsifier). substrate-side ready stays FALSE until
  the falsifier scores. V3 fallback if both gap-graded margins stay thin =
  rank_preserving_F_to_eligibility_demotion (F removed from the final argmin entirely, used only
  as a graded eligibility envelope) -- next step, NOT built now.
  Synthesis doc: REE_assembly/evidence/planning/conversion_ceiling_phase0_synthesis_2026-06-18.md
  See MECH-439 (this claim; first falsifier), modulatory-bias-selection-authority (parent; the
  top_k shortlist Factor A grades + the authority Factor B composes with), V3-EXQ-571 (F
  monopoly 88-89%), V3-EXQ-569i (top-k PASS 2/3, thin) / 569h (std-basis FAIL 1/3), ARC-065
  GAP-A candidate_summary_source=e2_world_forward + SD-056 (the divergent-pool non-vacuity
  precondition), MECH-341 / ARC-062 / MECH-309 / MECH-294 (the diversity channels that drown at
  the F-dominated argmax), MECH-313 (tonic-noise; distinct -- gap-blind, pre-select), MECH-090
  (admission gate; unchanged), MECH-094 (N/A).

## SD-057: Object-bound incentive-salience layer (GAP-7 L2-L3-L4) (2026-06-04)
- SD-057: drive.object_bound_incentive_salience -- IMPLEMENTED 2026-06-04
  (v1 = L2+L3+L4 core; L6 cue-recall + L7 dACC-wiring deferred to a phase-2
  pass within this SD). Closes the goal_pipeline:GAP-7 middle layer: the goal
  stream wrote a SINGLE z_goal attractor overwritten on every contact
  (wanting target == liking target always; L9 wanting!=liking dissoc stuck at
  0.0, V3-EXQ-514l). SD-057 inserts a per-object incentive layer between the
  benefit pulse and z_goal.
  Modules: ree_core/goal.py (new IncentiveTokenBank class + 5 GoalConfig
  fields + GoalState.incentive_bank member, instantiated when the master flag
  is set, reset() clears it); ree_core/agent.py (update_z_goal gains
  resource_type: Optional[int]=None kwarg + the L2-bind / L3-decay /
  L4-seed-redirect block before goal_state.update); ree_core/utils/config.py
  (from_dims surfaces all 5 knobs); experiments/_harness.py (StepHarness
  forwards obs_dict["resource_type_at_agent"] to update_z_goal).
  Mechanism (L0->L4):
    L2 MECH-344 (bind): on contact, benefit binds to the SD-049 per-type tag k
      (resource_type_at_agent / sd049_consumed_type_tag_this_tick, 1..n) ->
      IncentiveTokenBank.update(k, benefit, z_resource). The associative
      object->benefit node (BLA-analog; lit Cardinal/Everitt 2002).
    L3 MECH-345 (token): per-type bank entry holds base_value[k] (slow-decay,
      revaluable EMA of received benefit) + z_object[k] (stored z_resource
      identity embedding). Wanting at recall = base_value[k] *
      (1 + incentive_drive_kappa_weight * per_axis_drive[k]) -- the Zhang 2009
      V = r*kappa(drive) multiplier RELOCATED from the GoalState seeding gate
      onto the stored per-object value. Per-axis drive (SD-049 hunger/thirst/
      curiosity) makes wanting drive-specific / identity-matched (specific
      PIT; Corbit/Balleine 2005/2011).
    L4 MECH-346 (pointer; MECH-230 amend): z_goal seeded FROM the most-wanted
      object's embedding (k* = argmax_k wanting[k] -> seed_latent =
      z_object[k*]) instead of the raw last-contacted z_resource. GoalState
      firing gate (benefit/drive threshold) UNCHANGED -- only the seed SOURCE
      changes. So liking target (last-contacted) and wanting target (z_goal ->
      most-wanted) can DIFFER (e.g. ate food while thirsty -> z_goal points at
      water), making the L9 dissociation structurally expressible.
  Config (GoalConfig + from_dims; all no-op defaults, bit-identical OFF):
    use_incentive_token_bank (False) -- master switch.
    incentive_decay (0.005) -- per-object base_value slow decay per update().
    incentive_value_alpha (0.1) -- revaluation EMA rate on contact.
    incentive_drive_kappa_weight (2.0) -- relocated drive_weight for value x
      kappa(drive) at recall.
    incentive_use_per_axis_drive (True) -- per-axis (drive-specific) wanting.
  Backward compatible: use_incentive_token_bank=False by default ->
    GoalState.incentive_bank is None; update_z_goal takes the legacy
    single-attractor path; resource_type defaults None (2-arg callers
    unaffected). 747/751 contracts PASS (4 pre-existing local-git-env runner
    failures "Not a valid object name master", unrelated) + 7/7 preflight.
    Unit + agent-level smoke 2026-06-04: bank binds two types, most-wanted is
    drive-specific (food when hungry / water when thirsty) and differs from
    the last-contacted type; bank OFF bit-identical legacy seeding.
  Phased training: NOT required (the bank is a stateful EMA dict -- no trained
    parameters; reuses the already-trained SD-015/SD-049 z_resource encoder).
    A learned-affordance-embedding upgrade WOULD need P0/P1/P2.
  MECH-094: the bank updates only on WAKING contact via update_z_goal; no
    simulation/replay write surface, so hypothesis_tag does not apply
    (guardrail noted in the SD doc if a future revision writes during replay).
  use_resource_encoder (SD-015) must be set on cfg.latent directly (not via
    from_dims) for z_resource to populate -- SD-057's L2 bind requires it.
  Validation: behavioural L9 (wanting!=liking fraction >= 0.6, identity-probe
    > 0.6, per-axis ANOVA p<0.01) is GATED on goal_pipeline:GAP-2 supplying
    foraging contact; SD-057's own validation is a forced-contact MECHANISM
    diagnostic (two types at opposing drives, bank ON vs OFF, wanting_target
    != liking_target > 0 with bank ON / = 0 OFF) decoupled from GAP-2 --
    mirrors how V3-EXQ-626b decoupled the L1 positive control. Queued via
    /queue-experiment (claim_ids=[], diagnostic).
  Contracts: tests/contracts/test_sd_057_incentive_token_bank.py (C1 default-
    off no bank + legacy seeding / C2 L2 bind + tag-0 skip / C3 L3 per-axis
    drive-specific wanting + most_wanted / C4 revaluation + decay / C5 [1,n]
    shape robustness / C6 reset clears bank); test_step_harness_contract.py
    signature + kwargs pins updated in lockstep for the new resource_type arg.
  Design doc: REE_assembly/docs/architecture/sd_057_object_bound_incentive_salience.md
  Plan-of-record: REE_assembly/evidence/planning/goal_pipeline_plan.md GAP-7.
  See SD-057, MECH-344 (L2 bind) / MECH-345 (L3 token) / MECH-346 (L4 pointer;
    MECH-230 amend), MECH-229 / MECH-117 / ARC-030 (unblocked, stay v3_pending
    until validated), SD-049 (per-type tag + per-axis drive) / SD-015
    (z_resource encoder) / SD-012 / MECH-306 (reused substrate), MECH-295
    (approach bridge; downstream consumer), MECH-292/293 (ghost-goal bank;
    inactive-anchor precedent), MECH-347 (L6) / MECH-348 (L7)
    (phase-2, landed 2026-06-04 -- see entry below), MECH-094 (call-site scoping).

## SD-057 phase-2: L6 cue-recall + L7 dACC object-discriminative readout (2026-06-04)
- SD-057 phase-2: drive.object_bound_incentive_salience (L6 MECH-347 + L7
  MECH-348) -- IMPLEMENTED 2026-06-04. Completes the GAP-7 closure map on top of
  the SD-057 v1 (L2-L3-L4) bank landed earlier the same day. Both no-op-default,
  bit-identical OFF, no trained parameters (no phased training).
  Modules:
    ree_core/goal.py: GoalConfig gains use_cue_recall (False) + cue_recall_gain
      (0.05) + cue_recall_min_proximity (0.0); new GoalState.cue_pull(z_object,
      strength) -- a directional z_goal nudge with NO benefit gate and NO token
      revaluation (distinct from update(), which is benefit-gated + EMA-revalues).
    ree_core/agent.py: new REEAgent.cue_recall_wanting(cue_type, drive_level,
      simulation_mode) -- L6 primitive; retrieves the bank token for cue_type,
      computes drive-specific wanting amplitude, and cue_pulls z_goal toward
      z_object[cue_type] by cue_recall_gain*clamp(amp). L7: select_action dACC
      block computes per-candidate goal_proximity (to the object-bound z_goal,
      reusing the MECH-295 first-step z_world summary pattern) under
      use_mech_consume and passes candidate_goal_proximity into self.dacc(...).
      __init__ adds two loud-not-silent preconditions: use_mech_consume requires
      use_dacc; use_cue_recall requires use_incentive_token_bank. DACCConfig
      construction forwards dacc_goal_readout_weight.
    ree_core/cingulate/dacc.py: DACCConfig.dacc_goal_readout_weight (0.0);
      DACCAdaptiveControl.forward gains candidate_goal_proximity kwarg -> bundle
      "goal_readout"; DACCtoE3Adapter restructured so the goal-readout term
      (bias -= dacc_goal_readout_weight * goal_readout; proximity high -> favoured,
      REE lower-is-better) is added INDEPENDENTLY of dacc_weight (so a
      goal-conditioned consumer works even when the legacy dACC bias is off),
      after the dacc_bias clamp; skipped (bit-identical) when weight 0 / readout None.
    ree_core/utils/config.py: from_dims surfaces use_cue_recall / cue_recall_gain
      / cue_recall_min_proximity (GoalConfig) + use_mech_consume +
      dacc_goal_readout_weight (REEConfig); REEConfig dataclass fields added.
    experiments/_harness.py: StepHarness auto cue-perception -- when use_cue_recall,
      derive the strongest-perceived resource type from the SD-049 per-type
      proximity field views (resource_field_view_<name>, argmax over types) and,
      if >= cue_recall_min_proximity, call cue_recall_wanting each step
      (best-effort, try/except; bit-identical no-op when off / bank absent / env
      emits no per-type views). The forced-cue diagnostic calls the primitive
      directly.
  L6 data flow: perceived cue type k (no benefit) -> bank token[k] -> wanting[k]
    = base_value[k]*(1+kappa*per_axis_drive[k]) -> GoalState.cue_pull(z_object[k],
    cue_recall_gain*clamp(wanting)) -> z_goal moves toward k -> E3 goal_proximity
    + MECH-295 approach bias (existing, unchanged) raise pre-consummatory approach
    toward k. Identity-matched + drive-specific (specific PIT).
  L7 data flow: select_action -> per-candidate goal_proximity to object-bound
    z_goal -> dacc(candidate_goal_proximity=...) -> bundle["goal_readout"] ->
    DACCtoE3Adapter bias -= dacc_goal_readout_weight*goal_readout -> dACC->E3
    score_bias becomes object-discriminative.
  Backward compatible: use_cue_recall=False + use_mech_consume=False +
    cue_recall_gain unused + dacc_goal_readout_weight=0.0 by default. Full
    contracts: SD-057 v1 6/6, phase-2 new test_sd_057_phase2_cue_recall_consume.py
    5/5, dACC 5/5, step-harness 7/7, 750/757 full (7 pre-existing local-git-env
    runner-conflict-recovery fails "Not a valid object name master", unrelated) +
    7/7 preflight. Unit + agent smoke 2026-06-04: cue_recall fires + moves z_goal,
    sim_mode no-op, preconditions raise, dACC OFF bit-identical / ON favours
    high-proximity candidate.
  MECH-094: cue_recall_wanting(simulation_mode=True) is a no-op (replay must not
    move z_goal via a cue); dACC readout is waking-only. No replay write surface.
  Phased training: N/A (pure-arithmetic nudge + scalar bias; no learned params).
  Biological basis: Berridge 2009 cue-triggered wanting; Corbit/Balleine 2005/2011
    specific PIT; Schultz 1997/98 DA-transfer-to-cue (L6); Balleine & O'Doherty
    2010 goal-conditioned approach_commit (L7).
  Validation experiment: V3-EXQ-637 forced-cue diagnostic (claim_ids=[],
    decoupled from GAP-2 like V3-EXQ-636) queued via /queue-experiment.
  Design doc: REE_assembly/docs/architecture/sd_057_object_bound_incentive_salience.md
    (Phase-2 section). Plan-of-record: goal_pipeline_plan.md GAP-7.
  See SD-057 (v1 parent entry above), MECH-347 (L6) / MECH-348 (L7), MECH-344/345/346
    (v1 L2-L4), MECH-295 (approach bridge; L6 downstream), SD-032b/dACC (L7 host),
    SD-049 (per-type tag + per-axis drive + proximity views), MECH-229/MECH-117/ARC-030
    (downstream; stay v3_pending), MECH-094 (call-site scoping).

## ARC-063 v1: distributed CandidateRule field (GAP-B non-Bayesian rule-creator) (2026-06-04)
- ARC-063 (v1): policy.rule_apprehension_layer.candidate_rule_field -- IMPLEMENTED
  2026-06-04. The non-Bayesian rule-CREATOR that resolves
  arc_062_rule_apprehension:GAP-B (MECH-309: "trainers weight rules they do not
  invent"). Mint-then-weight over a subspace-partitioned field: CREATION is a
  non-gradient structural mint event; WEIGHTING is eligibility-trace credit on an
  existing rule's availability. The structural fix for the 543/598b rule_state
  collapse (598b C3 trainable_not_monomodal FAIL: two scalar scoring heads sharing
  a return gradient have an inert head_0==head_1 equilibrium -> the rule_state
  handed to SD-033a collapses). ARC-063 makes differentiation STRUCTURAL: distinct
  rules exist because they were minted as distinct pinned subspace directions
  (Weber 2023), so the rule_state is differentiated by construction.
  Brought V4->V3 2026-05-18; V3-tractable design landed 2026-06-04
  (docs/architecture/arc_063_candidate_rule_field.md). Sleep-vs-waking rule-field
  refinement is a V3 follow-on (NOT V4); social in-face (ARC-077/MECH-337) is the
  only genuinely V4-deferred face.
  Module: ree_core/policy/candidate_rule_field.py (CandidateRuleField +
    CandidateRule + CandidateRuleFieldConfig). Pure-arithmetic regulator (no
    nn.Module, no trained parameters, no gradient flow); sibling pattern to
    noise_floor.py (MECH-313) / structured_curiosity.py (MECH-314) /
    tonic_vigor.py (MECH-320). v1 has NO new trained encoder -> NO phased training.
  Five faces (the CandidateRule unit = rule_embedding [rule_dim] + context_tag
    [world_dim] + availability + eligibility + bookkeeping):
    1. CREATE (MECH-349): mint a slot on a recurring (context-bucket ->
       action-object) regularity >= crf_mint_recurrence_threshold times when no
       existing rule's context_tag already covers it; bottom-up (ARC-064) +
       optional top-down ARC-062 discriminator seed (gating_weight nudges initial
       availability when crf_seed_from_arc062).
    2. REPRESENT (MECH-350): pinned-distinct unit-vector slot directions
       (deterministic seeded basis) -> distinct minted rules occupy distinct
       subspace directions (the anti-monomodal geometry; Weber 2023 / Wallis 2001).
    3. GATE (MECH-351): tolerance-gated availability -- a rule is AVAILABLE only
       when availability >= theta = crf_tolerance_floor + crf_tolerance_conflict_gain
       * n_competing_context_matched_rules (Frank 2006 conflict-graded threshold;
       Cavanagh 2011). availability != selection (Frank/O'Reilly 2001).
    4. SELECT (MECH-338): cue-driven context-bound retrieval -- cosine(context,
       context_tag) >= crf_context_match_threshold.
    5. OUTPUT + CREDIT (MECH-352): the available-AND-context-matched rules combine
       (availability-weighted sum of embeddings) into a differentiated [1, rule_dim]
       rule_state vector for SD-033a; an eligibility trace credits the availability
       of rules active when an outcome arrived (raise on success, lower on
       exception; Brzosko 2015 / Kovach 2012), with slow decay + retire-below-floor.
  GAP-B wiring: LateralPFCAnalog (SD-033a) gains config.use_candidate_rule_source
    (set True at REEAgent.__init__ when the field is on) + an optional crf_source
    kwarg on update(). When supplied, crf_source REPLACES the legacy
    delta_proj/world_proj EMA source -- so rule_state tracks the field's
    differentiated active-rule stack (the literal 598b fix). The agent ticks the
    field in select_action at the lateral_pfc block: builds z_world context,
    credits the previous tick's active rules with this tick's outcome proxy
    (lower harm = success), mints on recurrence (keyed on the prev-tick chosen
    action class, stashed after E3 selection), gates+selects, returns crf_source.
  Config (REEConfig + from_dims; all no-op default, bit-identical OFF):
    use_candidate_rule_field (False) master, crf_n_slots (16), crf_rule_dim (16,
    matches lateral_pfc rule_dim), crf_mint_recurrence_threshold (3),
    crf_tolerance_floor (0.3), crf_tolerance_conflict_gain (1.0),
    crf_availability_alpha (0.1), crf_availability_decay (0.005),
    crf_eligibility_window (20), crf_context_match_threshold (0.5),
    crf_seed_from_arc062 (True).
  Precondition (loud ValueError at __init__): use_candidate_rule_field=True
    requires use_lateral_pfc_analog=True (SD-033a is the consumer). Matches the
    use_closure_operator / MECH-269b / MECH-293 precondition pattern.
  MECH-094: every state-advancing method takes simulation_mode and is a no-op when
    True (returns zeros, no mint, no credit); the existing MECH-319 _lpfc_skip gate
    also covers the lateral_pfc write site. Replay/DMN paths never mint or credit.
  Backward compatible: use_candidate_rule_field=False by default ->
    agent.candidate_rule_field is None; lateral_pfc.update sees crf_source=None and
    runs the legacy source; field bit-identical OFF. Regression: 7/7 preflight +
    775/782 contracts PASS (the 7 fails are the pre-existing local-git-env
    runner-conflict-recovery artifact "Not a valid object name master", zero
    overlap with this change). New contracts:
    tests/contracts/test_candidate_rule_field.py 8/8 (C1 default-off no-op / C2
    precondition / C3 CREATE distinct mints / C4 OUTPUT differentiated rule_state /
    C5 GATE conflict-sensitive / C6 CREDIT raises availability / C7 MECH-094 sim
    no-op / C8 agent ON sources + mints).
  Activation smoke (2026-06-04): field mints distinct context rules, rule_state
    differentiates across contexts (the C1 falsifier inversion), conflict gate
    holds under-supported rules out, credit raises availability, sim_mode no-op,
    agent ON populates SD-033a rule_state norm > 0.
  Phased training: NOT required (pure-arithmetic + buffers; no learned params).
    The learned-affordance rule-embedding upgrade WOULD need P0/P1/P2.
  DEFERRED (NOT in v1): sleep-vs-waking rule-field refinement pass (a V3 follow-on
    within ARC-063 reusing the MECH-272/273/285 sleep cluster -- NOT V4); full
    per-action moral-residue evidence records; learned rule-embedding; the social
    in-face (ARC-077/MECH-337, the only V4-by-substrate-necessity face).
  Validation experiment: V3-EXQ-639 substrate-readiness diagnostic (claim_ids=[];
    C1 differentiated rule_state >=2 distinct active rule vectors / C2 minting
    fires >=2 distinct context rules / C3 tolerance gate conflict-sensitive /
    C4 OFF bit-identical). The C4 ARC-062 GAP-B behavioural re-run on the
    field-populated substrate is the governance-weighting successor, queued
    separately.
  Design doc: REE_assembly/docs/architecture/arc_063_candidate_rule_field.md
  See ARC-063 (this), MECH-349 (CREATE) / MECH-350 (REPRESENT) / MECH-351 (GATE) /
    MECH-352 (CREDIT), MECH-338 (SELECT face), MECH-309 (logical necessity -- the
    creator is its answer), ARC-062 (weak-reading top-down source; exhausted
    alone -- GatedPolicy), ARC-064 (bottom-up source), SD-033a (rule_state
    consumer -- GAP-B target), MECH-318 (rule_state abstraction; retire-vs-promote
    downstream), MECH-319 (simulation-mode rule-write gate; covers the lateral_pfc
    site), ARC-077/MECH-337 (social in-face, V4), SD-057/MECH-347 (affective
    cue-recall sibling -- same retrieval-by-context motif), MECH-094 (call-site
    scoping).

## ARC-063 AMEND: cross-episode rule-persistence flag (V3-EXQ-654 GAP-B maturity) (2026-06-09)
- ARC-063 amend: policy.rule_apprehension_layer.candidate_rule_field cross-episode
  rule persistence -- IMPLEMENTED 2026-06-09. Module:
  ree_core/policy/candidate_rule_field.py (CandidateRuleFieldConfig +
  CandidateRuleField.reset()). Routed by the confirmed
  failure_autopsy_V3-EXQ-654_2026-06-09 (applied in the 2026-06-09 governance
  cycle; substrate_queue ARC-062 amend, impl target ARC-063 CandidateRuleField).
  ROOT CAUSE (code-grounded, NOT a falsification): the V3-EXQ-654 arc_062 GAP-B
  behavioural falsifier (MECH-309/ARC-062) FAILed its C1c readiness precondition
  (arm_on_rule_field_differentiated), so the C2 falsifier DV never ran. agent.reset()
  -> candidate_rule_field.reset() (agent.py:1908), called every ~26-tick episode,
  wiped self._rules + self._recurrence, so the live pool cold-started each episode
  and never matured a differentiated rule pool (crf_frac_active 0.116/0.123/0.115
  < 0.30 floor; crf_max_pairwise_rule_dist 0.0 all ARM_ON seeds; seed-42 ARM_ON
  committed-class byte-identical to ARM_OFF). The recurrence-threshold mint (>=3
  within ONE episode) + per-episode wipe starves the pool at behavioural-runtime
  episode lengths despite cumulative crf_n_minted 131-408. Biology accumulates
  task-set rule structure across experiences and is NOT reset per trial (Collins &
  Frank 2014; Mansouri rule-selective persistence) -- the per-episode wipe is a
  translation failure, a discovered prerequisite. V3-EXQ-639 PASS proves the field
  DOES differentiate when its pool matures.
  THE FIX (no-op-default; bit-identical OFF):
    Config: CandidateRuleFieldConfig.persist_rules_across_episode_reset (bool,
      default False). REEConfig.crf_persist_rules_across_episode_reset (default
      False) + REEConfig.from_dims passthrough; agent.py forwards it into the
      CandidateRuleFieldConfig at the field build site (getattr fallback, so an
      absent flat attr is bit-identical).
    Semantics: CandidateRuleField.reset() returns early (a no-op) when the flag
      is set -- the live rule pool (_rules), recurrence counters (_recurrence),
      AND the clock (_step, kept monotonic so minted_step/last_active_step stay
      ordered) PERSIST across the per-episode agent.reset(). The agent.reset()
      call itself is UNCHANGED; only the field's reset() semantics change. With
      the flag set, the field accumulates a live differentiated pool across the
      P1 measurement episodes so crf_frac_active can clear the 0.30 C1c floor at
      ~26-tick episode lengths.
  Default OFF = bit-identical per-episode wipe -- _rules/_recurrence/_step cleared
  exactly as before. The credit/eligibility/decay math uses no step deltas, so a
  continuous clock is safe (last_active_step/minted_step are bookkeeping-only).
  Backward compatible: persist_rules_across_episode_reset=False by default; every
  existing experiment cold-starts each episode unchanged. 11/11 CRF contracts
  (8 prior + 3 new: C9 default-OFF bit-identical wipe, C10 pool survives reset +
  keeps maturing, C11 from_dims + agent config wiring) + 7/7 preflight PASS; full
  contract suite 959 passed (the single control_vector C4 failure is a pre-existing
  baseline flake -- fails 4/5 on a clean tree, unrelated, documented above).
  Phased training: N/A (pure-arithmetic regulator; no learned parameters; the flag
  changes only episode-boundary pool lifecycle). MECH-094: unchanged -- the
  field's per-tick state-advancing methods keep their simulation_mode no-op gate;
  the persistence flag affects only the episode-boundary reset, not replay writes.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every
  existing experiment uses the default (per-episode wipe), so no dependent claim's
  measured mechanism changed. KEEP all evidence.
  GOVERNANCE: MECH-309/ARC-062 NEITHER validated NOR weakened; they stay candidate /
  substrate_ceiling / v3_pending / pending_retest_after_substrate. claims.yaml /
  governance state UNTOUCHED (substrate-only amend).
  Validation experiment: V3-EXQ-654a is the SECONDARY follow-up, queued SEPARATELY
  via /queue-experiment and GATED on this amend -- the same single-variable ARM_OFF
  (use_candidate_rule_field=False) vs ARM_ON (differentiated crf_source) GAP-B
  falsifier with crf_persist_rules_across_episode_reset=True + a TRAINED-bias-head
  P1 arm (GAP-D rule_bias_head trainability, landed 2026-05-17) + a propagation
  non-vacuity precondition (ARM_ON lateral_pfc bias must differ from ARM_OFF on
  firing ticks); committed-class entropy stays the PRIMARY DV. NOT queued in this
  implement-substrate session (user-scoped to the amend only).
  Design doc: REE_assembly/docs/architecture/arc_063_candidate_rule_field.md
  (cross-episode-persistence amend section, 2026-06-09).
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-654_2026-06-09.json.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json ARC-062 entry.
  See ARC-063 (parent v1 entry above), MECH-309 / ARC-062 (the GAP-B claims this
  matures the falsifier for; unweakened), SD-033a (rule_state consumer), GAP-D
  rule_bias_head trainability (the 654a propagation arm), V3-EXQ-654 (the FAIL this
  amend addresses), V3-EXQ-639 (PASS proving the field differentiates when matured),
  V3-EXQ-654a (validation; separate session), MECH-094 (call-site scoping; unchanged).

## ARC-063 AMEND: mature-pool gate/credit/retire dynamics (V3-EXQ-654b GAP-B maturity) (2026-06-11)
- ARC-063 amend: policy.rule_apprehension_layer.candidate_rule_field mature-pool
  gate/credit/retire dynamics -- IMPLEMENTED 2026-06-11. Modules:
  ree_core/policy/candidate_rule_field.py (CandidateRuleFieldConfig + _maybe_mint +
  gate_and_select + credit + __init__ + reset + get_state), ree_core/agent.py
  (CRF config build site + CRF tick-site context routing), ree_core/utils/config.py
  (REEConfig fields + from_dims). Routed by the confirmed
  failure_autopsy_V3-EXQ-654b_2026-06-11 (the ARC-062 substrate_queue
  recommended_substrate_queue_entry hand-off; amend_target_impl = ARC-063
  CandidateRuleField).
  ROOT CAUSE (budget reading EXHAUSTED, not a falsification): the 2026-06-09
  cross-episode persistence amend preserves the pool ACROSS episodes but does
  nothing about WITHIN-run churn. 654 (per-episode wipe) / 654a (crf_persist) /
  654b (crf_persist + 240 ep) all leave crf_frac_active pinned ~0.12-0.14 AND
  crf_max_pairwise_rule_dist EXACTLY 0.0 on every ARM_ON cell -- the field never
  holds >=2 rules present despite 452-1014 cumulative mints (mint -> brief life ->
  retire -> re-mint). Three drivers: (1) retire-churn PRIMARY (availability_decay
  0.005 + negative-outcome credit drive availability below _retire_floor=0.5*
  tolerance_floor=0.15 before a 2nd differentiated rule co-accumulates); (2)
  mint-block SECONDARY (context_match_threshold 0.5 lets one rule cover the
  low-spread z_world space; CRF reads raw z_world so the ARC-065 GAP-A fix does
  not reach it); (3) conflict-gate deadlock LATENT (tolerance_floor 0.3 +
  tolerance_conflict_gain 1.0 -> theta>=1.3 > 1.0 max availability whenever >=2
  rules match -> >=2 matched rules can NEVER both be active).
  THE FIX (no-op-default; bit-identical OFF; two opt-in flags):
    (1) crf_mature_pool_dynamics (CandidateRuleFieldConfig.mature_pool_dynamics,
        default False) routes a bundle of recalibrated knobs consulted ONLY under
        the master flag: mature_tolerance_floor 0.15 / mature_tolerance_conflict_gain
        0.25 -> theta(1)=0.40, theta(2)=0.65, theta(3)=0.90 all < 1.0 (deadlock
        fix -- up to 4 matched rules co-fire); mature_availability_decay 0.001
        (5x slower); mature_retire_floor 0.05 (absolute, decoupled from
        tolerance_floor); mature_availability_alpha_negative 0.02 (asymmetric --
        negative outcomes erode availability gently); mature_mint_protection_ticks
        30 (a freshly minted rule is retirement-protected so a 2nd differentiated
        rule co-accumulates -- the direct "drops below floor before a 2nd
        co-accumulates" fix); mature_mint_block_threshold 0.8 (DECOUPLED from the
        0.5 retrieval threshold so a differentiated mint is blocked only by a very
        similar existing rule). All mirrored as REEConfig.crf_mature_* +
        from_dims; agent build site forwards via getattr fallback (absent flat
        attr -> bit-identical).
    (2) crf_context_from_e2_world_forward (REEConfig, default False) sources the
        CRF mint/match context from e2.world_forward(z_world, prev_action_onehot)
        (action-regime-separated, no_grad, falls back to raw z_world when
        prev-action is unset / e2 absent) instead of raw z_world -- mirrors the
        ARC-065 GAP-A re-sourcing so the mint-block does not collapse under low
        raw-z_world spread.
  CRF-readiness readout: get_state() now emits crf_frac_active (fraction of ticks
  with >=1 active rule) + crf_n_active_steps alongside crf_max_pairwise_rule_dist.
  The Step-8 readiness diagnostic asserts crf_max_pairwise_rule_dist > floor AND
  crf_frac_active >= 0.30 -- the gate that MUST clear before any GAP-B behavioural
  falsifier (654c successor) is scored.
  Backward compatible: both flags default False -> all mature_* knobs inert, CRF
  context = raw z_world -> bit-identical. 16/16 CRF contracts (11 prior + 5 new:
  C12 mature default-OFF bit-identical + frac_active readout / C13 conflict-gate
  admits >=2 matched rules [legacy theta=1.3 deadlocks, mature theta=0.40 admits
  both] / C14 the 654b inversion [legacy max_pairwise_dist=0.0 / 1 rule / READY
  False at cos-0.7 collapsed regime, mature >=2 co-present / READY True] / C15
  mint-youth protection survives a below-floor fresh rule + asymmetric negative
  credit gentler / C16 from_dims + agent wiring of both flags + e2-context path)
  + 7/7 preflight PASS; full suite 1004 passed + 1 pre-existing control_vector C4
  flake (CONFIRMED failing on a clean tree via git stash, unrelated -- CRF is None
  in every existing experiment). Activation smoke (two regimes at cos 0.7,
  frequent hazard negatives): legacy n_present=1 / max_pairwise_dist=0.0 /
  READY=False (the 654b signature) -> mature n_present=2 / max_pairwise_dist=1.49 /
  frac_active~0.99 / READY=True.
  Phased training: N/A (pure-arithmetic regulator; no learned parameters; the
  flags change only gate/credit/retire arithmetic + the context key). MECH-094:
  unchanged -- the field's per-tick state-advancing methods keep their
  simulation_mode no-op gate; the e2.world_forward context read is no_grad on the
  waking select_action path (no replay write surface).
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flags; every
  existing experiment uses the defaults (legacy dynamics + raw z_world context),
  so no dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: MECH-309/ARC-062/ARC-063 NEITHER validated NOR weakened; stay
  candidate / substrate_ceiling / v3_pending / pending_retest_after_substrate.
  claims.yaml / governance state UNTOUCHED (substrate-only amend; the autopsy's
  failure record + amend hand-off land in substrate_queue.json via /governance).
  Validation experiment: V3-EXQ-654c-readiness CRF-readiness substrate diagnostic
  (claim_ids=[]; mature + e2-context arms vs OFF; asserts crf_max_pairwise_rule_dist
  > floor AND crf_frac_active >= 0.30 on >=2/3 seeds) queued via /queue-experiment
  as the Step-8 follow-on. The 654c GAP-B behavioural re-run is a SEPARATE later
  /queue-experiment session GATED on that readiness PASS.
  Design doc: REE_assembly/docs/architecture/arc_063_candidate_rule_field.md
  (mature-pool dynamics amend section, 2026-06-11).
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-654b_2026-06-11.{md,json}.
  See ARC-063 (parent v1 + cross-episode persistence amend above), MECH-309 /
  ARC-062 (the GAP-B claims this matures the falsifier for; unweakened), SD-033a
  (rule_state consumer), ARC-065 GAP-A (the e2.world_forward context-source
  pattern this mirrors), GAP-D rule_bias_head trainability (the 654c propagation
  arm), V3-EXQ-654b (the FAIL this amend addresses), V3-EXQ-639 (PASS proving the
  field differentiates when matured), MECH-094 (call-site scoping; unchanged).

## crf-availability-maintenance: activity-silent maintenance trace + maintained-pool readout (V3-EXQ-666 successor; ARC-063 amend) (2026-06-11)
- crf-availability-maintenance -- IMPLEMENTED 2026-06-11. Module:
  ree_core/policy/candidate_rule_field.py (CandidateRuleFieldConfig + _maybe_mint +
  credit + new maintained_reactivation_threshold / maintained_reactivatable_rules /
  maintained_pairwise_distance + get_state), ree_core/utils/config.py (REEConfig +
  from_dims), ree_core/agent.py (CRF config build site). Routed by the confirmed
  failure_autopsy_V3-EXQ-666_2026-06-11 + the targeted lit-pull
  evidence/literature/targeted_review_arc_063_crf_rule_cell_persistence/ (SYNTHESIS
  verdict B-leaning hybrid). New substrate_queue.json entry crf-availability-maintenance.
  ROOT CAUSE (differentiation<->persistence tension, NOT a falsification): the 654b
  mature-pool amend solves DIFFERENTIATION only via crf_context_from_e2_world_forward
  (V3-EXQ-666 ARM_2: 10-16 distinct rules, crf_max_pairwise_rule_dist 1.71) but WORSENS
  PERSISTENCE -- once each rule matches only a narrow context slice, its match-triggered
  availability EMA never accumulates above theta between sparse matches and decays in the
  gaps (mature_availability_decay 0.001/tick), so crf_frac_active collapses to 0.016
  (worse than the undifferentiated legacy 0.125); 0/3 readiness cells in every arm. The
  next binding constraint is availability MAINTENANCE under sparse matching (a PFC
  sustained-activity / activity-silent analog), NOT the conflict-gate theta (654b fixed
  that). The match-triggered-EMA scheme has the SYMBOL of a rule cell but not its
  maintenance functional role.
  LIT VERDICT (the A-vs-B fork): B (activity-silent synaptic maintenance, Mongillo 2008 /
  Stokes 2015 / Lundqvist 2018) with a bounded role for A (persistent firing reserved for
  the single ENGAGED rule; Funahashi 1989 / Compte-Wang 2000, capacity-bounded to one
  attractor -- cannot scale to a 10-16-rule pool). crf_frac_active (an INSTANTANEOUS active
  fraction) is the WRONG readiness readout for a sparsely-matched pool -- redefine it to a
  maintained-pool metric (survives whichever side of the firing-vs-silent debate prevails;
  Constantinidis 2018 is the live rebuttal).
  THE FIX (no-op default; bit-identical OFF; behind one master flag):
    (1) MAINTENANCE (prescription 1, Mongillo): under crf_availability_maintenance the
        per-tick SILENCE decay in credit() is REMOVED -- a minted differentiated rule HOLDS
        its availability across context-absent ticks. The eligibility-gated negative-outcome
        credit (the exception/interference path) + retirement are UNTOUCHED, so a
        consistently-bad rule still erodes and retires; only the gaps stop being punished. A
        maintenance_floor (default 0.45, above the mature 2-way-match theta(1)=0.40) is
        applied AT MINT so a fresh differentiated rule starts robustly reactivatable; NOT
        re-floored every tick (exceptions can still cross it). maintenance_decay (default 0.0
        = pure hold) is the optional long-horizon leak knob (set the horizon from the
        measured inter-match interval, not the ~1s biological constant).
    (2) READOUT (prescription 2, CONFIRMED "changes the readout" branch): get_state() gains
        crf_n_maintained_reactivatable (minted rules with availability >= the reactivation
        threshold = the single-match gate floor, i.e. would clear theta if their context
        recurred -- independent of whether it is present this tick) + crf_maintained_pairwise_dist
        (differentiation OF the maintained subset) + crf_frac_maintained. The 666-successor
        readiness gate is re-stated on the maintained pool -- crf_maintained_pairwise_dist > 0.1
        AND crf_n_maintained_reactivatable >= 2 -- RETIRING the crf_frac_active >= 0.30 target.
        crf_frac_active is kept as the SECONDARY active-on-match efficiency readout.
    (3) (Optional, default OFF) prescription 3: engaged_sustain adds a short reverberation
        bump to the matched-and-selected rule (fork-A complement). Not the pool fix.
  Config (CandidateRuleFieldConfig + REEConfig.crf_* + from_dims, all no-op default,
  bit-identical OFF): crf_availability_maintenance (False, master), crf_maintenance_floor
  (0.45), crf_maintenance_decay (0.0), crf_engaged_sustain (False), crf_engaged_sustain_rate
  (0.1), crf_maintained_reactivation_threshold (0.0 = derive from the gate floor). Forwarded
  into the field config at the agent.py build site (getattr fallback -> absent flat attr is
  bit-identical). Designed to run WITH crf_mature_pool_dynamics + crf_context_from_e2_world_forward
  (the differentiation source); the three stay independent flags so each is ablatable.
  Backward compatible: crf_availability_maintenance=False by default -> credit() takes the
  legacy decay path, _maybe_mint does not floor init -> bit-identical. The new get_state
  keys are always emitted (cheap, behaviour-neutral). 19/19 CRF contracts (16 prior + C17
  default-OFF bit-identical + readout keys / C18 maintenance HOLDS a differentiated >=2-rule
  reactivatable pool under sparse matching where legacy erodes it out / C19 from_dims + agent
  wiring) + 7/7 preflight PASS; full contract suite 1008 passed, 0 failures (the prior
  control_vector C4 flake did not fire this run). Activation smoke (mint two distinct rules,
  then 3000 context-absent ticks): legacy mature -> both rules RETIRED (n_minted=0,
  n_maintained=0, READY=False -- the 666 churn) vs maintenance -> pool HELD (n_minted=2,
  n_maintained=2, maintained_pairwise_dist=1.49, READY=True); crf_frac_active~0.001 in BOTH
  arms (confirming frac_active is the wrong readout, the maintained-pool metric is the right
  one).
  Phased training: N/A (pure-arithmetic regulator; no learned parameters; the flag changes
  only the credit-loop silence-decay + mint-floor + readout). MECH-094: unchanged -- the
  maintenance logic lives inside the field's already simulation-gated state-advancing methods.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment
  uses the default (legacy decay + active-fraction readout), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  GOVERNANCE: MECH-309/ARC-062/ARC-063 NEITHER validated NOR weakened; stay candidate /
  substrate_ceiling / v3_pending / pending_retest_after_substrate. claims.yaml carries only an
  implementation_note (no flag/confidence change; substrate-only amend).
  Validation experiment: V3-EXQ-666a (claim-free CRF-readiness diagnostic, supersedes
  V3-EXQ-666) -- the 666-successor re-validation enabling crf_mature_pool_dynamics +
  crf_context_from_e2_world_forward + crf_availability_maintenance, gating on the MAINTAINED-POOL
  metric (crf_maintained_pairwise_dist > 0.1 AND crf_n_maintained_reactivatable >= 2 on >=2/3
  seeds) NOT crf_frac_active. Queued via /queue-experiment. PASS unblocks the 654c GAP-B
  behavioural re-run (a SEPARATE later session). substrate_queue.ready stays FALSE until 666a clears.
  Design doc: REE_assembly/docs/architecture/arc_063_candidate_rule_field.md
  (availability-maintenance amend section, 2026-06-11). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-666_2026-06-11.{md,json}. Lit-pull:
  REE_assembly/evidence/literature/targeted_review_arc_063_crf_rule_cell_persistence/SYNTHESIS.md.
  See ARC-063 (parent v1 + cross-episode persistence + mature-pool amends above), MECH-309 /
  ARC-062 (the GAP-B claims this matures the readiness for; unweakened), crf_context_from_e2_world_forward
  / crf_mature_pool_dynamics (the differentiation source this maintenance pairs with), SD-033a
  (rule_state consumer), V3-EXQ-666 (the FAIL this amend addresses), V3-EXQ-666a (validation),
  V3-EXQ-639 (PASS proving the field differentiates when its pool matures), MECH-094 (call-site
  scoping; unchanged).

## MECH-353: blocked-agency / control-failure affect stream (z_block) (2026-06-06)
- MECH-353: affect.blocked_agency_control_failure_stream -- IMPLEMENTED 2026-06-06
  (substrate; v3_pending until the discriminative experiment PASSes). The
  energised "assert / restore" pole REE lacks: REE encodes only the
  capacity-collapsed WITHDRAW pole (SD-019b z_harm_a + Q-036). z_block is the
  capacity-RETAINED ASSERT pole that rises when an intended, predicted-to-succeed
  action is repeatedly BLOCKED while goal + capacity-belief are retained.
  Registered 2026-06-05 from the blocked-agency lit-pull (chip task_64c2e558,
  Davis & Montag 2019 RAGE / Papini 2024 frustrative non-reward / Bertsch 2020
  reactive-aggression assert channel / Carruthers 2012 comparator). Stream A only
  (single-agent); Stream B coercion/injustice is V4-social, NOT built.
  Modules:
    ree_core/affect/blocked_agency.py (NEW package + BlockedAgency +
      BlockedAgencyConfig + BlockedAgencyOutput). Pure-arithmetic regulator (no
      nn.Module, no learned params), mirrors MECH-313 / MECH-320 / MECH-342.
    ree_core/latent/stack.py (LatentState.z_block [batch,1] + detach()).
    ree_core/agent.py (import; self.blocked_agency + _ba_prev_z_world/_z_self
      caches in __init__; reset(); _update_blocked_agency() called from sense()
      after the MECH-095 TPJ update; ASSERT score-bias in select_action after the
      tonic_vigor block; ARC-016-gated DECOMMIT release beside the MECH-342 site).
    ree_core/utils/config.py (REEConfig + from_dims: 17 no-op-default knobs).
    ree_core/environment/causal_grid_world.py (scheduled_action_block_enabled /
      _interval / _prob env-only kwargs; step() cancels the move on a block with
      no damage / no layout change, transition_type="action_blocked"; info tags;
      reset of per-episode counters).
  Detector (in _update_blocked_agency, waking path):
    Action-outcome comparator (SD-029 on z_world; Carruthers 2012):
      zw_pred = E2.world_forward(z_world_prev, last_action);
      pred_mag = ||zw_pred - z_world_prev||;
      outcome_mismatch = ||zw_pred - z_world_now|| / (pred_mag + eps), gated by a
      predicted-effect floor (pred_mag >= blocked_agency_predicted_effect_floor;
      else 0 -> nothing to be blocked from + fails safe untrained). Calibration
      (trained world_forward): blocked -> ~1.0; success -> ~0.0. FNR's
      expected-minus-obtained generalised to action-effect-omission (Papini 2024).
    Motor-agency / external-attribution comparator (z_self channel):
      motor_agency = 1/(1 + ||E2.predict_next_self(z_self_prev, last_action) -
      z_self_now||). High = motor executed as predicted -> external block, not own
      motor error (predict_next_self is E2's trained objective -> reliable gate).
    Expectation (MECH-112): goal_state.is_active() -> there is an intended outcome.
  Integration (BlockedAgency.update): z_block accumulates outcome_mismatch over a
    window ONLY when motor_agency >= attribution_motor_floor AND outcome_mismatch
    >= outcome_mismatch_floor AND goal retained; leaks on success. Capacity split:
    z_block_assert = z_block * capacity_belief; withdraw_handoff = z_block *
    (1 - capacity_belief); capacity_belief = clip(1 - w*||z_harm_a||, 0, 1).
  Consumers (select_action):
    ASSERT: self-emitted per-candidate score-bias (negative on action / positive
      on no-op / extra positive on the blocked action class) composed into
      dacc_score_bias after the tonic_vigor block (the "raise MECH-320 vigor +
      alternative-action search" effect; zero when no asserting block this tick).
    DECOMMIT: update() emits decommit_signal after z_block_assert sustains above
      decommit_bound for decommit_consecutive_ticks; consumed beside the MECH-342
      site -> beta_gate.release(), gated by ARC-016 (e3.current_precision <=
      blocked_agency_decommit_arc016_precision_max; <=0 disables the gate).
    HANDOFF: assert decays as capacity falls; withdraw_handoff surfaced; existing
      SD-019b / z_harm_a / Q-036 withdraw machinery takes over (no forced write).
  Config: REEConfig.use_blocked_agency (default False; bit-identical OFF -> agent.
    blocked_agency is None, LatentState.z_block stays None, no consumer fires) +
    16 sub-knobs (all no-op default). Env: scheduled_action_block_* (env-only,
    not in from_dims; bit-identical OFF, no RNG draws).
  Backward compatible: 803/803 ree-v3 contracts + bit-identical action stream with
    master OFF AND with use_blocked_agency=True but no env block (regulator uses no
    torch RNG; zero bias when z_block~0). 9/9 new contracts in
    tests/contracts/test_mech_353_blocked_agency.py PASS.
  Phased training: N/A (pure-arithmetic regulator; no learned parameters).
  MECH-094: update() no-op under simulation_mode / hypothesis_tag (replay must not
    accumulate blocked-agency on imagined outcomes); z_block left None on tagged
    latents.
  DETECTOR DISCRIMINATION DEPENDS ON A TRAINED SUBSTRATE: at smoke / untrained
    scale z_world deltas do not track single-cell moves (random encoder), so the
    action-outcome comparator is uninformative (expected). The validation EXQ
    trains the encoder + action-conditional world_forward (SD-056) in P0 and
    measures block-vs-control discrimination in P1. A trained-substrate failure to
    discriminate is a substrate-ceiling finding (encoder/world_forward enrichment),
    NOT a falsification of the affective claim.
  Validation experiment: V3-EXQ blocked-action discriminative diagnostic
    (claim_ids=[]; env repeatedly blocks an intended predicted-to-succeed action,
    harm + goal-value held constant; measure z_block rise + assert-vs-withdraw +
    dissociation from z_harm_a under matched controllability). Queued via
    /queue-experiment. PASS clears MECH-353 v3_pending.
  Design doc: REE_assembly/docs/architecture/mech_353_blocked_agency_zblock.md;
    affect-register row: REE_assembly/docs/architecture/affect_primitives.md
    (Extension Register, blocked_agency).
  See MECH-353, SD-029 (comparator detector), MECH-112 (z_goal expectation),
    MECH-320 (assert/vigor pole), MECH-342 (decommit actuator), ARC-016 (decommit
    gate), SD-011 (harm, differentiated-from), SD-019b/Q-036 (suffering withdraw
    pole, opposite controllability), MECH-056 (residue, differentiated-from),
    MECH-090 (commitment-hold, differentiated-from), MECH-095 (TPJ z_self agency
    sibling), SD-056 (action-conditional world_forward; detector prerequisite),
    MECH-094 (simulation gate).

## SD-058 / MECH-357: instrumental-avoidance acquisition (ilPFC-analog freeze-suppression + avoidance action pathway) (2026-06-07)
- SD-058 (architecture) / MECH-357 (mechanism): defensive_action.instrumental_avoidance_acquisition
  -- IMPLEMENTED 2026-06-07 (substrate; v3_pending until the Stage-H validation
  EXQ PASSes). Closes the scaffolded_sd054_onboarding Stage-H / P1 survival-leg
  gap (V3-EXQ-603g G_H 0/3; goal_pipeline GAP-2). Routed by
  failure_autopsy_V3-EXQ-603g-624c-651a_2026-06-07 + the lit verdict
  evidence/literature/targeted_review_hazard_avoidance_learning/SYNTHESIS.md
  (SD-035 x3 + MECH-279 + SD-054): the fix is STRUCTURAL, not budgetary. REE has
  the Pavlovian/defensive REACTION side (SD-035 amygdala salience + MECH-279 PAG
  freeze) but lacked the instrumental-ACQUISITION side. Moscarello & LeDoux 2013:
  active avoidance learning is the resolution of a Pavlovian-instrumental
  conflict -- learning to avoid REQUIRES the infralimbic PFC to SUPPRESS
  CeA-driven freezing (ilPFC lesion -> more freezing, less avoidance). A
  freeze-only substrate freezes instead of learning to avoid -- the 603g G_H 0/3
  signature.
  Module: ree_core/pfc/infralimbic_avoidance_gate.py (InstrumentalAvoidanceGate +
  InstrumentalAvoidanceGateConfig + InstrumentalAvoidanceGateOutput). Pure-
  arithmetic regulator (no nn.Module, no trained params, no gradient flow);
  sibling to the SD-035 CeA/BLA, MECH-279 PAG, MECH-313 NoiseFloor, MECH-320
  TonicVigor pattern. Lives in ree_core/pfc/ alongside lateral_pfc_analog.py
  (dlPFC) + ofc_analog.py -- infralimbic is the third PFC subdivision analogue.
  Three pieces, all behind use_instrumental_avoidance (default False, bit-identical):
    (a) INSTRUMENTAL-AVOIDANCE ACTION pathway: a per-candidate E3 score-bias
        composed last in the dacc_score_bias chain (after MECH-320 vigor +
        MECH-353 assert) that, under retained threat (z_harm_a), PENALISES the
        no-op/freeze class proportional to effective_efficacy * threat_scale --
        releasing the instrumental action. Does NOT compute the escape direction
        (E3's existing harm gradient ranks the directed candidates; ARC-007-
        strict-compatible). compute_action_bias(z_harm_a_norm, action_classes,
        noop_class) -> [K].
    (b) ilPFC FREEZE-SUPPRESSION gate: at the MECH-279 application site in
        select_action, when freeze_active AND should_suppress_freeze(z_harm_a_norm)
        (effective_efficacy * threat_scale >= suppression_threshold), the no-op
        override is SKIPPED so the agent takes its selected instrumental action.
        Inert when use_pag_freeze_gate=False; the action-pathway half (a) still
        operates (freezing is a passive no-op the bias penalises).
    (c) AVOIDANCE-EFFICACY LEARNING (eligibility trace; the acquisition): a scalar
        avoidance_efficacy in [0,1] starting at initial_efficacy (0.0 = freeze-
        default). update(z_harm_a_norm, action_was_directed) in sense() (one-tick
        lag): a directed action under threat that DROPS z_harm_a credits efficacy
        (EMA toward 1); freezing/failed-avoidance under threat decays it. The
        gradual developmental acquisition (Debiec & Sullivan 2017 / Thompson 2008).
        PERSISTS across episodes within a stage -- reset() clears ONLY the
        within-episode threat trace, NOT the learned efficacy.
  PROTECTIVE-SCAFFOLD anneal (the curriculum; SECONDARY): effective_efficacy =
  max(avoidance_efficacy, scaffold_floor). The Stage-H curriculum
  (scaffolded_sd054_onboarding) sets a high scaffold_floor early and anneals it
  down as the learned efficacy takes over (maternal-buffering / Turchetta 2020
  reset-curriculum analogue). Budget escalation is the explicitly-secondary lever.
  Config (REEConfig + from_dims, all no-op default): use_instrumental_avoidance
  (False), avoidance_learn_rate (0.05), avoidance_leak_rate (0.02),
  avoidance_initial_efficacy (0.0), avoidance_scaffold_floor (0.0),
  avoidance_threat_floor (0.1), avoidance_threat_ref (0.5),
  avoidance_efficacy_reward_floor (1e-4), avoidance_action_bias_gain (0.1),
  avoidance_bias_scale (0.1), avoidance_suppression_threshold (0.5),
  avoidance_noop_class (0).
  Agent wiring (ree_core/agent.py): instantiate self.instrumental_avoidance when
  the master flag is on; _ia_last_action_directed cache; update() in sense()
  after _update_blocked_agency (reads new_latent.z_harm_a; MECH-094 no-op under
  hypothesis_tag); action-bias composed after the MECH-353 ba_bias block;
  freeze-suppression at the MECH-279 site; directed-action cache after
  self._last_action = action; reset() clears the within-episode trace + cache
  (preserves learned efficacy).
  Curriculum wiring (experiments/scaffolded_sd054_onboarding.py): run_hazard_avoidance
  (Stage-H) protective-scaffold anneal under scaffold_avoidance_driver_enabled
  (default False) + scaffold_avoidance_scaffold_floor_start (0.8) /
  scaffold_avoidance_scaffold_floor_end (0.0); HazardAvoidanceResult gains
  avoidance_driver_enabled + avoidance_gate_state so the manifest can confirm
  acquisition (efficacy rose, freeze suppressed) rather than survival-by-chance.
  LOAD-BEARING PREREQUISITE (found 2026-06-07): the legacy scaffold called
  sense(body, world) with NO harm args, so z_harm_a was None across the WHOLE
  curriculum -- leaving MECH-279 PAG, SD-035 amygdala AND the SD-058/MECH-357 gate
  all INERT (they key on z_harm_a). New scaffold flag scaffold_feed_harm_stream
  (default False -> bit-identical) + module helper _sense_with_optional_harm feed
  the env harm_obs + harm_obs_a into sense() so z_harm_a is populated (~0.34 in
  Stage-H). The avoidance-driver experiments set it True; without it the gate has
  no threat signal to learn from.
  DISTINCT from the reflexive escape-from-freeze levers (do NOT read as a
  duplicate): SD-037 override_signal (orexin) raises the PAG exit threshold and
  MECH-281 lowers the MECH-091 urgency-interrupt -- BOTH reflexive threat/arousal-
  driven escape. MECH-357's ilPFC suppression is gated by LEARNED avoidance-
  efficacy (eligibility trace), bootstrapped by the protective-scaffold floor --
  the acquisition mechanism (Moscarello & LeDoux), not a reflex.
  Backward compatible: use_instrumental_avoidance=False by default ->
  agent.instrumental_avoidance is None; sense update + select_action bias +
  freeze-suppression all skipped -> bit-identical. Stage-H driver gated by
  scaffold_avoidance_driver_enabled AND requires the gate. 912 contracts + 7/7
  preflight PASS; 7 new contracts in
  tests/contracts/test_mech_357_instrumental_avoidance.py + 4 C13 in
  tests/contracts/test_scaffolded_sd054_onboarding.py. Bit-identical OFF verified
  (default vs use_instrumental_avoidance=True at zero efficacy/floor: identical
  action stream, no RNG, zero bias).
  Phased training: N/A (pure-arithmetic regulator; no learned parameters; the
  "learning" is the eligibility-trace EMA, not an encoder head). MECH-094: both
  compute methods + update() no-op under simulation_mode (replay must not credit
  avoidance or suppress freeze on imagined outcomes). Evidence-staleness: NOT
  triggered (no-op-default flag; no dependent claim's measured mechanism changed).
  Validation experiment: V3-EXQ-603h substrate-readiness diagnostic (claim_ids=[],
  queued via /queue-experiment). LITERAL Moscarello & LeDoux lesion-vs-intact,
  2-arm, 3 seeds: BOTH arms have MECH-279 PAG (tuned to z_harm_a~0.34) + the fed
  harm stream; ARM_LESION (PAG, no gate -> freezes) vs ARM_INTACT (PAG + ilPFC
  gate + driver -> suppresses freeze + acquires avoidance). PRIMARY: G_H_INTACT
  >= 2/3 AND G_H_INTACT_frac > G_H_LESION_frac. Non-vacuity preconditions: PAG
  freezes on LESION (pag_n_commits>0) AND the gate engages+suppresses on INTACT
  (n_credit+n_decay>0 AND n_freeze_suppressed>0), else substrate_not_ready_requeue.
  Dry-run engages the full chain (PAG freezes 14, gate suppresses 8, readiness
  met; primary not met at dry scale as expected). PASS unblocks the goal_pipeline
  GAP-2 retest cohort + the pending_retest_after_substrate claims ARC-060 /
  MECH-320 / ARC-068 / SD-054-readiness. substrate_queue scaffolded_sd054_onboarding
  ready STAYS false until then.
  Design doc: REE_assembly/docs/architecture/sd_058_instrumental_avoidance_acquisition.md
  See SD-035 (amygdala BLA/CeA -- defensive-reaction salience side this builds the
  acquisition side onto), MECH-279 (PAG freeze-gate -- the freeze output the ilPFC
  gate suppresses), SD-011 (z_harm_a -- threat signal), MECH-320 / MECH-314
  (sibling policy-layer score-bias regulators), MECH-353 (blocked-agency assert
  pole -- distinct; SD-058 resolves freeze-vs-avoid under threat), MECH-094 (call-
  site scoping), scaffolded_sd054_onboarding (Stage-H curriculum driver),
  goal_pipeline:GAP-2 (the survival leg this closes), V3-EXQ-603g (the FAIL this
  addresses).

## SD-059 / MECH-358: relief/safety escape-affordance bridge (directed escape for the MECH-357 gate) (2026-06-08)
- SD-059 (architecture) / MECH-358 (mechanism): defensive_action.escape_affordance_bridge
  -- IMPLEMENTED 2026-06-08 (substrate; v3_pending until the 4-arm validation EXQ
  PASSes). Closes the V3-EXQ-603h directed-escape gap. Routed by
  failure_autopsy_V3-EXQ-603h_2026-06-08 + thought_intake_2026-06-07_relief_safety_escape_affordance_bridge.
  SD-058/MECH-357 suppress the MECH-279 freeze but avoidance_efficacy is a GLOBAL
  SCALAR that only penalises the no-op class -- compute_action_bias by design "does
  NOT compute the escape direction". 603h (engaged-but-insufficient, readiness met):
  the gate suppressed freeze on all INTACT seeds but G_H_INTACT=0/3; seed-43 reached
  scalar efficacy 0.633 and survived WORST (11.0). The agent un-froze without
  acquiring a DIRECTED escape. Moscarello & LeDoux 2013: active avoidance needs the
  LA/BA->NAcc relief/safety action-credit half, not only suppression. REE owns relief
  (MECH-302/SD-050) + safety (MECH-303/304/SD-052/SD-051) but they were UNWIRED to
  avoidance -- this is the wiring.
  Module: ree_core/pfc/escape_affordance_bridge.py (EscapeAffordanceBridge +
  EscapeAffordanceBridgeConfig + EscapeAffordanceBridgeOutput). Pure-arithmetic
  regulator (no nn.Module, no trained params, no gradient flow); sibling to the
  SD-058/MECH-357 gate in ree_core/pfc/. Extends MECH-357's scalar avoidance_efficacy
  into a per-FIRST-ACTION-CLASS credit table (the minimal V3 rendering of
  escape_affordance[action] -- the directed escape direction in the discrete action
  space; location/policy indexing deferred, disambiguated by the validation's
  nav-competence control).
  Two independently-toggleable halves (so the 4-arm validation dissociates):
    RELIEF half (MECH-302-consistent): a directed action under threat that DROPS
      z_harm_a (delta = prev - now > relief_reward_floor) credits
      relief_affordance[action_class] (EMA toward 1) -- the d(z_harm_a)/dt<0 signal
      attributed to the specific action.
    SAFETY half (MECH-303/304-consistent): a directed action after which threat is
      absent (threat_scale <= 0) credits safety_affordance[action_class]
      (response-produced safety / conditioned inhibition).
  Approach bonus (the directed escape): under FUTURE threat (threat_scale > 0), E3
  receives a per-candidate NEGATIVE (favoured; REE lower-is-better) score-bias toward
  each candidate whose first-action class carries combined affordance credit:
  -clamp(approach_gain * threat_scale * combined_affordance[class], 0, bias_scale);
  the no-op/freeze class never gets a bonus. THREE guards: bias_scale clamp (cannot
  dominate the chain), threat-context gate (exactly zero when safe -> never swamps
  food/goal approach), per-tick leak (forgetting -> no pathological habit loop).
  DISTINCT from reflexive escape (SD-037 orexin / MECH-281 urgency -- threat/arousal
  reflexes) and from the generic relief/safety rows (MECH-302/303/304 fire on the
  CURRENT state); this is learned-efficacy-gated DIRECTED approach binding an action
  to relief/safety for future-threat use.
  Config (REEConfig + from_dims; all no-op default, bit-identical OFF):
  use_escape_affordance_bridge (False, master) + use_escape_relief_credit (True) +
  use_escape_safety_credit (True) + escape_relief_learn_rate (0.1) +
  escape_safety_learn_rate (0.1) + escape_bridge_leak_rate (0.01) +
  escape_relief_reward_floor (1e-4) + escape_threat_floor (0.1) + escape_threat_ref
  (0.5) + escape_approach_gain (0.1) + escape_bias_scale (0.1) + escape_noop_class
  (0). n_action_classes set from config.e2.action_dim at agent build.
  Agent wiring (ree_core/agent.py): build self.escape_affordance_bridge when the
  master flag is on; _eab_last_action_class cache; update() in sense() after the
  MECH-357 eligibility update (reads new_latent.z_harm_a; MECH-094 no-op under
  hypothesis_tag; bridge directedness = internal non-noop check, independent of
  whether MECH-357 is enabled); approach-bias composed after the MECH-357 action-bias
  block; action-class cache after self._last_action; reset() clears the within-episode
  trace + cache (affordance tables PERSIST across episodes, same as MECH-357 efficacy).
  Curriculum wiring (experiments/scaffolded_sd054_onboarding.py): the bridge runs
  automatically inside the existing Stage-H run_hazard_avoidance training loop once the
  agent is built with use_escape_affordance_bridge=True;
  HazardAvoidanceResult.escape_bridge_state surfaces bridge.get_state() so the
  validation manifest can gate non-vacuity (relief/safety credit actually incremented)
  before scoring G_H.
  Backward compatible: use_escape_affordance_bridge=False by default ->
  agent.escape_affordance_bridge is None; sense update + approach-bias all skipped ->
  bit-identical. Verified: default == explicit-False action stream (8-tick); 603h
  --dry-run runs unchanged (bridge OFF); scaffolded contracts 91/91; 8 new SD-059
  contracts in tests/contracts/test_sd_059_escape_affordance_bridge.py PASS. (Two
  failures in the full contract run are pre-existing/unrelated: control_vector C4
  fails at baseline; scaffolded C12 is test-ordering flakiness -- passes isolated and
  in its own file with these changes.)
  Phased training: N/A at the substrate level (pure-arithmetic regulator). BUT the
  relief detector reads z_harm_a / the world-forward, so the VALIDATION experiment
  keeps the SD-056 e2 contrastive warmup in P0 + a relief-credit non-vacuity readiness
  gate -- without a trained encoder the credit re-starves (603h n_credit 6/0 on 2/3
  seeds).
  MECH-094: both compute methods + update() no-op under simulation_mode (replay/DMN
  must not credit escape affordances or bias action selection on imagined outcomes).
  Call-site: sense() + select_action() are waking-only; no replay/memory write surface.
  Evidence-staleness: NOT triggered (no-op-default flag; no dependent claim's measured
  mechanism changed).
  Validation experiment: V3-EXQ-603i substrate-readiness diagnostic (claim_ids=[],
  queued via /queue-experiment) -- thought-intake Section 5 4-arm
  (ARM_BASE_IA_ONLY / ARM_RELIEF_BRIDGE / ARM_SAFETY_BRIDGE / ARM_RELIEF_SAFETY_BRIDGE)
  + a nav-competence positive control so a flat G_H across all bridge arms is
  attributable to a survival/navigation ceiling rather than the bridge. Non-vacuity
  gate: each enabled bridge half must increment its credit before G_H is scored.
  Acceptance: G_H >= 2/3 AND improves over ARM_BASE_IA_ONLY; secondary P1 survival
  transfer without over-avoidance/starvation. PASS unblocks goal_pipeline GAP-2 +
  ARC-060 / MECH-320 / ARC-068 / SD-054-readiness retests.
  Design doc: REE_assembly/docs/architecture/sd_059_escape_affordance_bridge.md
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603h_2026-06-08.{md,json}.
  See SD-058 / MECH-357 (the gate this extends scalar -> affordance-indexed), MECH-302
  / SD-050 (relief), MECH-303/304 / SD-052/SD-051 (safety), MECH-279 (PAG freeze),
  SD-011 (z_harm_a), SD-037 / MECH-281 (reflexive escape; distinct), SD-056
  (action-conditional divergence; relief-detector prerequisite), ARC-060 / MECH-320 /
  ARC-068 (unblocked on PASS), scaffolded_sd054_onboarding (Stage-H driver),
  goal_pipeline:GAP-2 (the survival leg this closes), V3-EXQ-603h (the FAIL this
  addresses), MECH-094 (call-site scoping).

## Post-603i successor scaffold: trainable relief/safety escape-affordance learner (2026-06-08)
- trainable_escape_affordance_learner -- SCAFFOLDED 2026-06-08, not validated
  substrate. Module: ree_core/pfc/trainable_escape_affordance_learner.py
  (TrainableEscapeAffordanceLearner + Config + Output). This is a successor option
  inspired by REE_assembly/docs/thoughts/2026-06-08_Trainable_Releif_and_Safety.md,
  not a replacement for the active SD-059 / MECH-358 arithmetic bridge and not a
  change to V3-EXQ-603i. Config (REEConfig + from_dims; master default OFF):
  use_trainable_escape_affordance_learner (False),
  use_trainable_relief_critic (True), use_trainable_safety_predictor (True),
  trainable_escape_bias_scale (0.1), trainable_escape_relief_learn_rate (0.1),
  trainable_escape_safety_learn_rate (0.1), trainable_escape_leak_rate (0.01),
  trainable_escape_relief_reward_floor (1e-4),
  trainable_escape_relief_target_scale (0.3), trainable_escape_threat_floor
  (0.1), trainable_escape_noop_class (0), trainable_escape_hidden_dim (32),
  trainable_escape_action_embedding_dim (8), trainable_escape_optimizer_lr
  (0.03), trainable_escape_prediction_floor (0.02). Data flow: sense()
  detached z_world/z_self/z_harm_a + last action class -> local PyTorch
  relief/safety heads update through AdamW (one-tick lag, MECH-094 no-op under
  hypothesis_tag) -> select_action() per-candidate first action classes +
  current z_harm_a -> bounded negative E3 score-bias from model predictions.
  If the arithmetic bridge and trainable learner are both enabled, they compose
  additively, each under its own clamp. Guards: zero when disabled, zero when
  safe, simulation/hypothesis no-op, no-op/freeze receives no credit or approach
  bonus, failed relief and threat recurrence train extinction targets. Episode
  reset clears only one-tick traces and preserves learned head weights. Local note:
  docs/substrate_plans/trainable_escape_affordance_learner.md. No queue entry,
  no claims/governance effect; do not use for promotion until an explicit
  successor experiment is queued and reviewed.

## SD-059 / MECH-358 AMEND: safety-half trained threat-absence wiring (V3-EXQ-603i secondary gap, 2026-06-09)
- SD-059 / MECH-358 safety-half wiring amend -- IMPLEMENTED 2026-06-09. Wires the
  trained MECH-303 (contextual safety terrain) / MECH-304 (conditioned safety
  store) threat-absence prediction into the EscapeAffordanceBridge safety-credit
  path so the SAFETY half can credit non-vacuously. Routed by the SECONDARY gap of
  failure_autopsy_V3-EXQ-603i_2026-06-08 (Section 4 Prerequisites (b) + Section 6
  Learning #2): on V3-EXQ-603i the safety half credited 0/3 in EVERY arm (relief
  half credited 2/3, functional) because the raw safety condition
  (threat_scale(z_now) <= 0, i.e. z_harm_a norm below threat_floor) almost never
  fires under Stage-H -- the threat never goes fully absent after a single directed
  action. The half was wired STRUCTURALLY but STARVED: no trained threat-absence
  predictor fed it (symbol-of-mechanism-without-functional-input). REE already owns
  the trained predictors -- MECH-303 (SD-052, ResidueField.evaluate_safety RBF
  terrain) + MECH-304 (SD-051, ConditionedSafetyStore EMA-prototype cosine->sigmoid)
  -- but they were unwired to the bridge.
  THE FIX (no-op default; bit-identical OFF; bridge stays OFF by default):
    Module: ree_core/pfc/escape_affordance_bridge.py
      (EscapeAffordanceBridgeConfig + EscapeAffordanceBridge.update). Config gains
      use_trained_safety_signal (False) + safety_signal_threshold (0.5). update()
      gains safety_signal: Optional[float]=None. The SAFETY credit now fires when
      raw threat-absence (threat_scale(z_now) <= 0) OR -- when use_trained_safety_signal
      -- the supplied trained safety_signal >= safety_signal_threshold. OR-composed,
      so the original raw mechanism is retained as a fallback; the trained path
      stays INSIDE the existing under-threat (prev > threat_floor) + directed-action
      gate, so it credits genuine response-produced safety (MECH-303/304-consistent),
      not generic safe-context. New diagnostic counter _n_safety_credit_trained
      (surfaced as mech358_n_safety_credit_trained in get_state) attributes credits
      to the trained predictor specifically (the non-vacuity readout for the
      validation manifest).
    Module: ree_core/safety/conditioned_safety_store.py -- new read-only
      ConditionedSafetyStore.predict(z_world) -> float (the cosine->sigmoid query
      WITHOUT decay/EMA mutation). Additive; never called by existing paths -> zero
      behaviour change. Lets the agent read the MECH-304 cue-specific safety
      prediction for the CURRENT post-action state at the bridge-update site (the
      store's own update() runs LATER in the same sense(), so the cached
      _conditioned_safety_signal is one tick stale).
    Module: ree_core/agent.py -- at the bridge.update call site in sense() (after
      the MECH-357 eligibility update), when escape_use_trained_safety_signal is on
      and not simulation, compute _eab_safety_signal = max over enabled trained
      predictors: MECH-304 conditioned_safety_store.predict(z_world) (if the store
      is built) and MECH-303 residue_field.evaluate_safety(z_world).mean() (if
      use_contextual_safety_terrain). Both are pure reads. Passed as
      bridge.update(safety_signal=...). None when the flag is off -> bit-identical.
      Bridge build site forwards the two new config fields into the bridge config.
    Config: REEConfig.escape_use_trained_safety_signal (False) +
      escape_safety_signal_threshold (0.5), both wired through REEConfig.from_dims.
  Backward compatible: escape_use_trained_safety_signal=False by default; the
    safety_signal kwarg defaults None and the OR-branch is never taken, so the
    bridge (and the V3-EXQ-603i path) is byte-identical. ConditionedSafetyStore.predict
    is unused by existing paths. 7/7 preflight + 951 contracts PASS (incl 2 new C9/C10
    in tests/contracts/test_sd_059_escape_affordance_bridge.py + the prior 8). Bridge
    stays bit-identical OFF (default vs explicit-False action stream, C10). Activation
    smoke 2026-06-09 (bridge + MECH-304 + flag ON, affective harm stream on, retained
    threat): the agent feeds a real trained safety signal (~0.375 on an untrained
    encoder), the under-threat gate opens (z_harm_a_prev ~0.79 > floor), and the safety
    half credits 5x via the trained signal (mech358_n_safety_credit_trained=5,
    safety_affordance_max~0.40); flag OFF reproduces the 603i 0/0 starvation.
  Phased training: N/A at the substrate level (pure-arithmetic regulator + read-only
    accessor; no learned parameters). BUT the MECH-303/304 predictors are themselves
    populated from MECH-302 relief events / low-harm contexts, so the validation
    experiment must enable at least one trained predictor AND keep the SD-056 e2
    contrastive warmup (and a non-vacuity readiness gate on mech358_n_safety_credit_trained)
    -- without trained predictors the safety half re-starves.
  MECH-094: preserved. The bridge.update safety-signal path is gated by the existing
    simulation_mode no-op (replay/DMN never credit); ConditionedSafetyStore.predict is
    a pure read; the agent computes the signal only on the waking sense() path (not under
    hypothesis_tag). Evidence-staleness: NOT triggered -- no-op-default flag; no dependent
    claim's measured mechanism changed (the bridge is OFF in every existing experiment).
  GOVERNANCE: SD-059 / MECH-358 NEITHER validated NOR weakened by this amend; they stay
    candidate / v3_pending / pending_retest_after_substrate. V3-EXQ-603i script / queue /
    governance state UNTOUCHED. claims.yaml NOT modified (this amend resolves no
    dependency; the bridge stays v3_pending pending the retest).
  Validation experiment: V3-EXQ-603j claim-free safety-half-credit readiness
    microdiagnostic (ablation: escape_use_trained_safety_signal OFF reproduces the 603i
    safety_credit~0; ON -> mech358_n_safety_credit_trained > 0 on >=2/3 seeds). This is
    the focused SECONDARY-gap gate ONLY. The full 4-arm G_H behavioural bridge retest
    stays gated on the PRIMARY nav/survival-competence ceiling
    (scaffolded_sd054_onboarding Stage-H leg / separate chip) -- a retest can only score
    G_H once nav competence clears too. Does NOT validate/weaken SD-059/MECH-358.
  Design doc: REE_assembly/docs/architecture/sd_059_escape_affordance_bridge.md
    (safety-half trained-signal amend section).
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603i_2026-06-08.{md,json}.
  See SD-059 / MECH-358 (parent; arithmetic bridge landed 2026-06-08), MECH-303 / SD-052
    (contextual safety terrain), MECH-304 / SD-051 (conditioned safety store), MECH-302 /
    SD-050 (relief; the functional half), SD-058 / MECH-357 (parent gate), SD-056
    (e2 action-conditional divergence; predictor prerequisite), V3-EXQ-603i (the FAIL
    this amend's SECONDARY gap addresses), V3-EXQ-603j (validation), MECH-094 (call-site
    scoping; preserved).

## MECH-294: multi-content theta-burst packet (joint {goal,action,risk,state} per-cycle binding) (2026-06-09)
- MECH-294: theta_burst.multi_content_joint_packet -- IMPLEMENTED 2026-06-09
  (substrate; v3_pending until the discriminative validation EXQ PASSes). Closes
  the /queue-experiment substrate-readiness gate that found MECH-294
  blocked_substrate on 2026-06-09: V3 implemented only single-content temporal
  averaging (MECH-089 ThetaBuffer buffers z_world/z_self and returns a
  theta-cycle-AVERAGED z_world), so any MECH-294 joint-binding experiment was
  vacuous. Per design memo REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md.
  SIBLING module, NOT a ThetaBuffer rewrite: MECH-089 is depended on by MECH-092
  replay / MECH-122 consolidation / SD-006 cross-rate E3 read, so the packet
  COMPOSES the existing ThetaBuffer for its state_summary slot and gathers the
  other three streams itself. MECH-089 untouched; MECH-294 is a strict additive
  layer.
  Module: ree_core/latent/multi_content_theta_packet.py (MultiContentThetaPacket
  + MultiContentThetaPacketConfig + ThetaPacket + ThetaPacketVintage). Pure-
  arithmetic (no nn.Module, no learned parameters, no gradient flow); sibling
  pattern to MECH-313 / MECH-320 / MECH-342. Every observed tensor is detached
  (a packet is a read-only snapshot, never a grad path).
  Four streams + concrete REE-v3 sources (memo S3): goal_latent <- GoalState.z_goal
  (when active); risk_sensory <- LatentState.z_harm (SD-010); risk_affective <-
  LatentState.z_harm_a (SD-011, kept as a DISTINCT sub-slot -- not pre-collapsed,
  preserves the SD-011 dissociation into the packet); state_summary <- the
  MECH-089 averaged z_world (theta_buffer.summary()); action_proposal <- the
  hippocampal CEM proposer lead first-action (candidates[0].actions[:,0,:]).
  Per-cycle phase-aligned binding window (memo S4.1): a theta cycle is the
  interval between two E3 heartbeat ticks. The packet opens a window, accumulates
  per-stream values during E1 ticks (observe), takes the proposer first-action at
  the E3 tick (observe_action_proposal), and SEALS one immutable ThetaPacket at
  the E3-heartbeat boundary (seal), exposed as agent.last_theta_packet for the
  proposer/E3 to read as a joint object on the next cycle. Gamma-sub-slot routing:
  type-separated named sub-slots (NOT a flat concat) so the joint-read interface
  can condition action on goal-and-risk.
  Per-stream V_s vintaging (MECH-269 / MECH-269b REUSE, memo S4.3): the packet
  applies the SAME snapshot-or-hold discipline -- refresh the per-stream snapshot
  when V_s >= snapshot_refresh_threshold (0.5), substitute the held last-verified
  snapshot when V_s < hold_threshold (0.4), 0.4-0.5 dead-band = MECH-269b Schmitt
  hysteresis. Each component records ThetaPacketVintage{is_current, age_ticks,
  v_s}; action_proposal has no V_s (its vintage is age in E3 ticks). This makes
  the packet a stream-typed object whose components may carry DIFFERENT temporal
  vintages (the MECH-294 secondary property) rather than a homogeneous current
  latent.
  Three binding regimes (the discriminative core that resolves the Kay-2020
  cross-cycle-alternation falsifier, memo S6) selected by theta_packet_binding_mode:
    "joint"       -- all four streams' current (or V_s-held) values bound
                     SIMULTANEOUSLY within one cycle (the MECH-294 hypothesis).
    "alternation" -- exactly ONE stream live per cycle (round-robin), the other
                     three HELD at their prior snapshots; only the live stream
                     refreshes (Kay-2020 one-stream-per-cycle parsimonious control).
    "shuffled"    -- each slot filled from a DIFFERENT prior cycle's value for
                     that stream (matched marginals, never co-observed;
                     independent-content control).
  All three produce structurally distinct sealed packets from the SAME input
  stream (verified C4 + smoke: last-packet structural dist joint-vs-alt 17.9,
  joint-vs-shuf 44.6) -- the substrate-side discriminability the validation
  depends on.
  Joint-read interface (memo S5): joint_context() (type-tagged concat for a
  single-context consumer), action_conditioned_on(goal, risk) (the literal "which
  action is on the table read against the same-cycle goal + risk" operation),
  risk_vector() (concat over the two SD-011 sub-slots), is_component_stale(name).
  Config (REEConfig + from_dims, all no-op default, bit-identical OFF):
    use_multi_content_theta_packet (False, master). Requires use_per_stream_vs=True
      (the packet consumes the MECH-269 per-stream V_s) -- LOUD ValueError at
      agent __init__ otherwise (same precondition pattern as MECH-269b / MECH-293).
    theta_packet_binding_mode ("joint" / "alternation" / "shuffled").
    theta_packet_snapshot_refresh_threshold (0.5), theta_packet_hold_threshold
      (0.4) -- MECH-269b reuse.
    theta_packet_compose_into_e3_bias (False) -- READ-ONLY-FIRST (memo S5): when
      False the packet is built/sealed/exposed/logged but does NOT touch E3
      selection (the validation measures it read-only). When True, joint_context
      biases E3 via a PARAMETER-FREE clamped arithmetic bias (cosine of candidate
      first-action vs the packet's co-bound action_proposal, clamped to
      +/-bias_scale) -- NO trained head, so NO phased training. Composed LAST in
      the dacc_score_bias chain so it never dominates.
    theta_packet_bias_scale (0.1).
  Agent wiring (ree_core/agent.py): self.multi_content_theta_packet built in
  __init__ when the master flag is on (else None) + self.last_theta_packet; reset()
  clears the window/snapshots/history; _e1_tick observe(...) right after
  theta_buffer.update; _e3_tick observe_action_proposal + seal -> last_theta_packet
  at the theta_buffer.summary() boundary; optional compose hook before the MECH-313
  noise-floor block. All call sites are waking (simulation_mode=False).
  MECH-094: every observe/seal method takes simulation_mode and is a no-op when
  True (a replay/DMN tick must not seal a waking packet). Call sites are waking-only
  (call-site scoping, same as MECH-269 / MECH-288 / MECH-287).
  Phased training: NONE required -- the packet is gather/route/seal arithmetic +
  dataclass plumbing over already-trained latents; no new encoder head, no gradient
  flow. (A future learned type-embedding or trained joint-read head would need
  P0/P1/P2; out of scope here.)
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (packet off), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  Backward compatible: use_multi_content_theta_packet=False by default ->
  agent.multi_content_theta_packet is None; both push sites + the seal are skipped;
  MECH-089 ThetaBuffer byte-identical. 7/7 preflight + 968 contracts PASS (8 new in
  tests/contracts/test_multi_content_theta_packet.py: C1 default-OFF no-op +
  bit-identical / C2 ON 4-slot sealed packet at the E3 boundary / C3 V_s-held
  substitution + stale vintage age / C4 joint vs alternation vs shuffled
  structurally distinct / C5 MECH-094 simulation no-op / C6 action_conditioned_on
  joint read / C7 precondition raises without use_per_stream_vs). Bit-identical OFF
  verified (default vs explicit-False action stream, seed-matched). Activation smoke
  2026-06-09 (full sense path, harm streams on): packet seals at the E3 boundary
  with risk_sensory + risk_affective + state_summary + action_proposal populated.
  GOVERNANCE: MECH-294 NEITHER promoted NOR weakened; stays candidate / v3_pending /
  implementation_phase=v3; the 2026-04-26 governance hold (needs a substrate-side
  joint-vs-alternation falsification test) stands until the C1/C2 discriminative
  validation PASSes. claims.yaml carries only an implementation_note (no flag /
  confidence change).
  Validation experiment: V3-EXQ-657 substrate-readiness diagnostic (claim_ids=[];
  4-arm matched-seed/matched-content ARM_0 OFF / ARM_1 joint / ARM_2 alternation /
  ARM_3 shuffled; G0 packet completeness + G1 vintage heterogeneity +
  NON-VACUITY precondition (shuffled must structurally differ from joint, else
  substrate_not_ready_requeue) + C1 joint!=alternation + C2 joint!=shuffled). Non-
  vacuously tests joint-within-cycle vs Kay-2020 cross-cycle alternation. PASS (per
  the memo S7.3 interpretation grid) clears the substrate-readiness gate that found
  MECH-294 blocked_substrate; only then does the MECH-294 behavioural-evidence
  successor get queued.
  Design doc: REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md
  (Status flipped design-memo -> IMPLEMENTED 2026-06-09).
  See MECH-294 (this claim), MECH-089 (parent ThetaBuffer; composed, untouched),
  MECH-269 / MECH-269b (per-stream V_s vintaging reuse), SD-010 (z_harm_s) / SD-011
  (z_harm_a) (risk sub-slots), SD-012 / SD-015 / MECH-230 (GoalState.z_goal source),
  ARC-018 (hippocampal CEM proposer; action_proposal source), SD-005 (z_world/z_self
  split; dep), MECH-090 (E3-heartbeat commit machinery; cycle boundary), Kay et al.
  2020 (cross-cycle alternation falsifier the S7 experiment settles), V3-EXQ-657
  (validation), MECH-094 (simulation-mode call-site scoping).

## ARC-006 / MECH-045: token-instance object-file / entity-persistence buffer (2026-06-09)
- ARC-006 / MECH-045: entities.object_file_buffer -- IMPLEMENTED 2026-06-09 (v1
  substrate; v3_pending until V3-EXQ-658 clears at full scale). The TOKEN store of
  the ARC-080 type/token/anchor triad -- the missing third per-item store (TYPE =
  SD-057 IncentiveTokenBank goal.py; ANCHOR = SD-039/MECH-292 ghost-goal bank
  hippocampal/). A persistence-vs-ablation probe against the two LIVE stores would be
  vacuous (neither re-identifies a moved entity), which is why proposal EVB-0293 was
  marked blocked_substrate by a 2026-06-09 /queue-experiment Step-2.5 gate. Design
  memo (the spec): REE_assembly/docs/architecture/mech_045_object_file_buffer.md.
  Module: ree_core/entities/object_file_buffer.py (ObjectFileBuffer + ObjectFile +
  EntityObservation + ObjectFileBufferConfig). NON-TRAINABLE stateful regulator (no
  nn.Module, no params, no gradient flow); sibling to IncentiveTokenBank /
  BlockedAgency / EscapeAffordanceBridge. DeepSORT-style data association over
  z_world-local features, one update() per waking tick (memo Section 4.2):
    C1 token KEY    -- label-free per-entity token id by spatiotemporal continuity:
                       hard MOTION gate (obf_continuity_radius) + precision-weighted
                       appearance cost (w_motion*d_pos + w_feat*d_cos); the token
                       SURVIVES the entity moving to a new cell (NOT nearest-cell).
    C2 feature buf  -- per-token precision-weighted z_features EMA (the "file").
    C3 persistence  -- a token survives <= obf_persist_ttl unseen ticks, then dies.
    C4 attention    -- salience-gated births; tokens compete for a bounded buffer
                       (obf_max_tokens, the FINST ~4-5 capacity analogue).
    C5 precision    -- association + feature update precision-weighted (zero-precision
                       obs makes no feature move).
  type_hint records the SD-049 resource_tag as an OPTIONAL wiring hook -- NEVER the
  association key (the key is continuity, not type).
  Config (REEConfig + from_dims, all no-op default; bit-identical OFF):
    use_object_file_buffer (False, master) + obf_max_tokens (5) + obf_continuity_radius
    (2.0) + obf_w_motion (1.0) + obf_w_feat (1.0) + obf_feature_alpha (0.3) +
    obf_persist_ttl (8) + obf_min_birth_salience (0.0) + obf_use_precision_weighting (True).
  Agent wiring (ree_core/agent.py): self.object_file_buffer built when the master flag
  is on; driven on the waking stream via REEAgent.update_object_file_buffer(observations,
  simulation_mode=...) (the caller / experiment supplies the perceived entities -- the v1
  detector dependency, memo Section 4.4: SD-049 per-type resource-field views + grid
  object/hazard cells); reset() clears it per episode.
  v1 lands STANDALONE: NO action-stream consumer -> the action stream is BIT-IDENTICAL
  whether the buffer is on or off (only buffer state changes; nothing reads it yet).
  SD-057 (TYPE->token type_hint) + SD-039 (ANCHOR->token_id cross-ref) wiring is
  explicitly DEFERRED (memo Section 3). OFF the GAP-7 V3-closure path.
  MECH-094: update() is a no-op under simulation_mode (the buffer updates ONLY on the
  waking stream -- a moved-imagined-object must not rewrite the waking object-file);
  agent passes hypothesis_tag through.
  Phased training: N/A (non-trainable; the buffer consumes already-encoded z_world and
  produces token assignments; a learned association metric is the v2 upgrade path).
  Backward compatible: use_object_file_buffer=False by default -> agent.object_file_buffer
  is None and update_object_file_buffer returns {}; bit-identical. 6/6 new contracts
  (tests/contracts/test_mech_045_object_file_buffer.py: C1 default-off + bit-identical
  action stream / C2 cross-motion re-identification with same-type distractor / C3
  persistence-through-absence + eviction / C4 attention capacity cap / C5 precision
  weighting / C6 MECH-094 sim no-op) + 7/7 preflight + 973 contracts PASS (the 1
  control_vector C4 fail is the documented pre-existing baseline flake, unrelated).
  Activation smoke 2026-06-09: at world_dim=128 with two same-type instances (distinct
  appearances) + motion, the moved target re-identifies to the SAME token (reid 1.0)
  while ABLATION-OFF (no token) and ABLATION-SHUFFLE (ids permuted) do not; bit-identical
  action stream OFF==ON verified.
  Validation experiment: V3-EXQ-658 (experiments/v3_exq_658_mech045_object_file_persistence.py)
  -- the EVB-0293/EXP-0117 re-queue: 3 arms INTACT/ABLATION-OFF/ABLATION-SHUFFLE x 3 seeds
  at world_dim=128 with several same-type distractors; readiness gates G0 (token
  maintained) / G1 (feature non-degenerate d_cos>=0.1, the dim=32 guard) / G2 (entity
  moved) before C1 reid>=0.6 (PRIMARY) / C2 INTACT>SHUFFLE+0.3 AND >OFF / C3
  feature-persistence; self-routes substrate_not_ready_requeue if readiness unmet.
  claim_ids=[MECH-045] (ARC-006 bears-on, not co-tagged). substrate_queue.ready stays
  FALSE until it clears at full scale. MECH-045/ARC-006 NOT promoted/weakened (governance
  applies after the run).
  Design memo: REE_assembly/docs/architecture/mech_045_object_file_buffer.md.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json (mech-045-object-file-buffer).
  See ARC-006 / MECH-045 (this claim), ARC-080 (type/token/anchor umbrella + fork),
  SD-057 IncentiveTokenBank (TYPE store; proposed wiring target), SD-039 / MECH-292
  ghost-goal bank (ANCHOR store; proposed wiring target), SD-049 (per-type tags +
  proximity views -- the v1 detector source), MECH-278 (object schema; V4, bypassed),
  E2WorldForward (world_dim>=128 floor the experiment runs at), V3-EXQ-658 (validation),
  MECH-094 (waking-stream-only call-site scoping).

## MECH-294 AMEND: compose path reads within-cycle co-binding coherence (mode-dependent E3 bias) (2026-06-09)
- MECH-294 compose-coherence amend -- IMPLEMENTED 2026-06-09. Module:
  ree_core/latent/multi_content_theta_packet.py (ThetaPacket.currency_coherence +
  MultiContentThetaPacket.compose_e3_bias) + ree_core/agent.py (compose call site) +
  ree_core/utils/config.py (1 new no-op-default flag). Routed by the /queue-experiment
  attempt to queue the MECH-294 behavioural-evidence successor: a code trace found the
  compose path could NOT carry the joint binding into behaviour, so the behavioural
  mode-discrimination experiment was premature (it would self-route a FALSE "joint clause
  not isolated" weakening on an unmet wiring precondition).
  ROOT CAUSE (code-confirmed): compose_e3_bias computed bias = -bias_scale *
  cosine(candidate_first_action, last_packet.action_proposal). seal() sets
  action_proposal = self._win_action IDENTICALLY across joint/alternation/shuffled
  (the binding mode only changes the goal/risk/state slots), and compose_e3_bias read
  ONLY action_proposal -- so with theta_packet_compose_into_e3_bias=True the three modes
  produced BEHAVIOURALLY IDENTICAL action streams (differing only from packet-OFF).
  This is exactly the design-doc S6 C1-FAIL "wiring" case ("the downstream consumer is
  not conditioning on co-binding") and contradicts S5's intent that joint_context be the
  compose input. The 2026-04-26 governance hold demanded a substrate-side test that
  discriminates joint-packet from cross-cycle alternation; an action-only compose makes
  co-binding inert by construction, so that test was not yet buildable.
  THE FIX (parameter-free; no trained head; no phased training; bit-identical OFF):
  the per-candidate action-grounding bias is now GATED by the sealed packet's
  ThetaPacket.currency_coherence() in [0, 1] -- the fraction of the four V_s-gated
  content streams (goal, risk_sensory, risk_affective, state) whose vintage is CURRENT
  this cycle, i.e. the direct operationalisation of "the streams are bound
  co-temporally". bias = clamp(-bias_scale * coherence * cosine(cand_fa, action_proposal),
  +/-bias_scale). So the SAME proposer action yields a STRONG grounding bias under joint
  (coherence ~1.0), a WEAK one under alternation (~0.25, three streams held -- Kay-2020),
  and NONE under shuffled (0.0, every slot drawn from a different cycle). The binding
  regime now reaches E3 behaviour, which the behavioural mode-discrimination experiment
  requires. The per-candidate RANKING stays an in-space action cosine (no
  cross-semantic-space comparison -- avoids the V3-EXQ-657a coherence-metric autopsy
  pitfall); only the GATE is mode-derived.
  Config: REEConfig.theta_packet_compose_use_joint_coherence (bool, default True) +
  from_dims passthrough. True = the spec-faithful mode-dependent gating; False = recovers
  the legacy action-only cosine (gate==1.0) bit-for-bit (the validation ablation arm).
  NO-OP DEFAULT: theta_packet_compose_into_e3_bias still defaults False, so the entire
  compose block never runs for any existing experiment regardless of this flag --
  bit-identical OFF. No compose-ON experiment exists yet (V3-EXQ-657/657a were
  compose-OFF read-only-first), so nothing depended on the old formula.
  Diagnostics (get_diagnostics, for the validation manifest): mech294_last_currency_coherence,
  mech294_n_compose_calls, mech294_last_compose_coherence, mech294_last_compose_bias_absmax.
  Backward compatible: 8/8 multi_content_theta_packet contracts + 7/7 preflight PASS;
  py_compile OK; V3-EXQ-657a --dry-run unchanged (compose-OFF path). Activation smoke
  2026-06-09 (compose ON, forced-benefit regime, 3 ep x 30 steps): JOINT coherence 1.000
  (bias absmax 0.100) / ALTERNATION 0.250 (0.025) / SHUFFLED 0.000 (0.000); JOINT action
  histogram {3:59, 2:31} differs from ALT/SHUF {3:90} (binding mode changed selection);
  coh=OFF recovers gate==1.0; compose-OFF n_compose_calls=0 (bit-identical).
  Phased training: N/A (pure-arithmetic gate; no learned parameters). MECH-094: compose
  is read-only on the waking select_action path (no replay/memory write surface);
  preserved. Evidence-staleness: NOT triggered -- no-op-default flag; every existing
  experiment uses the default (compose OFF), so no dependent claim's measured mechanism
  changed. KEEP all evidence.
  HONEST LIMITATION: the per-candidate term is action-alignment GATED by coherence -- it
  is conjunctive via the gate (co-binding present), not via a learned same-cycle
  goal x risk x action readout. A richer learned joint-context reader is a future
  phased-training upgrade. The validation EXQ must guard candidate first-action diversity
  (non-vacuity: if the proposer emits a single action class, the gate has nothing to rank
  and joint==alternation behaviourally).
  GOVERNANCE: MECH-294 NEITHER promoted NOR weakened; stays candidate / v3_pending /
  implementation_phase=v3; the 2026-04-26 hold stands until the behavioural successor
  PASSes. claims.yaml carries only an implementation_note (no flag/confidence change).
  Validation experiment: V3-EXQ substrate-readiness diagnostic (claim_ids=[]; compose-ON
  mode-discrimination wiring check: does binding mode now change the committed-class
  distribution, with coh-ON vs coh-OFF ablation isolating the gating). PASS gates the
  MECH-294 behavioural-evidence successor (claim_ids=["MECH-294"]) queued as a separate
  /queue-experiment session.
  Design doc: REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md
  (compose-coherence amend section).
  See MECH-294 (parent; substrate landed 2026-06-09), MECH-089 (ThetaBuffer; composed),
  MECH-269 / MECH-269b (per-stream V_s vintaging the coherence reads), ARC-018
  (hippocampal CEM proposer; action_proposal source), Kay et al. 2020 (cross-cycle
  alternation control), V3-EXQ-657a (read-only-first readiness PASS this builds on),
  MECH-094 (call-site scoping; preserved).

## MECH-294 AMEND: per-candidate co-binding coherence (cross-candidate-range rendering so the route-range authority + 569i top-k can carve) (2026-06-19)
- MECH-294 per-candidate-coherence amend -- IMPLEMENTED 2026-06-19 (substrate;
  MECH-294 stays candidate / substrate_ceiling / v3_pending -- this PROMOTES
  NOTHING; the 2026-04-26 governance hold stands, and the behavioural falsifier is
  a SEPARATE later /queue-experiment step). Modules:
  ree_core/latent/multi_content_theta_packet.py (ThetaPacket.action_refs /
  coherence_weights + MultiContentThetaPacket.compose_per_candidate_coherence +
  seal() per-stream action-reference binding + _action_snapshots + 3 diagnostics),
  ree_core/agent.py (compose-block branch + packet-config coherence_hold_weight
  passthrough), ree_core/utils/config.py (2 no-op-default flags + from_dims).
  Routed by the substrate-ceiling-lifted triage 2026-06-19
  (REE_assembly/evidence/planning/substrate_ceiling_lifted_triage_2026-06.md,
  verdict (b) MECH-294 STILL CEILINGED): the named substrate
  (modulatory-bias-selection-authority) is implemented but cannot carve MECH-294.
  ROOT CAUSE (V3-EXQ-661 + the route-range amend's "coherence" source, code +
  manifest confirmed): currency_coherence() is a SCALAR. compose_e3_bias produced
  bias = -scale * coherence * align where align[K] = cosine(candidate_first_action,
  the single bound action_proposal) -- a per-candidate PATTERN identical across
  binding modes (seal sets action_proposal = _win_action regardless of mode);
  coherence only scales magnitude uniformly. The route-range authority normalises
  the routed bias to unit range, so the scalar's only effect (uniform magnitude
  scaling) is ERASED -> JOINT == ALTERNATION; SHUFFLED (coherence 0) floors to 0.
  661 manifest: committed-action histograms BYTE-IDENTICAL across ARM_OFF / JOINT /
  ALT / SHUF / ALT_COH_OFF per seed (committed-dist TV ~0, max 0.0097, incl
  C3 gate-ON-vs-OFF). So the authority + 569i top-k had nothing mode-distinct to
  carve. The route-range readiness gate (V3-EXQ-663) correctly flags coherence as a
  no-range channel.
  THE FIX (no-op-default; bit-identical OFF): a PER-CANDIDATE co-binding coherence
  whose cross-candidate RANGE is mode-distinct. seal() now binds, per V_s-gated
  content stream, the action_proposal co-bound WITH that stream this cycle
  (action_refs) + a currency weight (coherence_weights), via a per-stream action
  snapshot mirror of the existing _snapshots:
    JOINT       -- all four streams current -> all refs = this-cycle action (w 1.0)
                   -> compose_per_candidate_coherence == cosine(cand, a) -> FULL
                   cross-candidate range.
    ALTERNATION -- live stream this-cycle action (w 1.0) + held streams their PRIOR
                   co-bound action at coherence_hold_weight (default 0.5) -> a
                   per-candidate PATTERN distinct from JOINT (smoke: joint-vs-alt
                   bias cosine 0.82, NOT a uniform scaling -> survives
                   range-normalisation), not merely a smaller magnitude.
    SHUFFLED    -- nothing co-bound this cycle (weights 0) -> None / zeros -> the
                   authority reads below-floor (matches currency_coherence 0.0).
  compose_per_candidate_coherence returns clamp(-bias_scale * weighted_mean_s[
  w_s * cosine(cand, ref_s)], +/-bias_scale) -- in-action-space throughout (no
  cross-semantic-space comparison; cf. the V3-EXQ-657a coherence-metric autopsy).
  The agent compose block branches on theta_packet_compose_per_candidate_coherence
  and (when ON) sets _bdc_coherence to the new per-candidate bias, so the EXISTING
  route source "coherence" (modulatory_channel_route_source) now routes a
  mode-distinct per-candidate RANGE into the modulatory accumulator the route-range
  authority rescales + the 569i top-k shortlists. currency_coherence() is NOT
  consulted or modified -- the scalar mode-discrimination (JOINT 1.0 / ALT 0.25 /
  SHUF 0.0) is preserved bit-identically.
  Config (REEConfig + from_dims + MultiContentThetaPacketConfig, all no-op default;
  bit-identical OFF): theta_packet_compose_per_candidate_coherence (False; only
  consulted when theta_packet_compose_into_e3_bias is also True -- OFF recovers the
  legacy scalar-gated action-only compose bit-for-bit) +
  theta_packet_coherence_hold_weight (0.5; held-stream prior-ref weight, 0.0 =
  pure currency-gating).
  Backward compatible: theta_packet_compose_per_candidate_coherence=False by
  default AND the packet master switch use_multi_content_theta_packet defaults
  False, so the seal-time action-ref bookkeeping runs only for packet-ON
  experiments and never changes the bound content slots / currency_coherence ->
  bit-identical. 8/8 prior multi_content_theta_packet contracts + 7/7 preflight +
  the full contract suite PASS; 7 new contracts in
  tests/contracts/test_mech294_per_candidate_coherence.py (C1 JOINT range>floor /
  C2 SHUFFLED ~0 / C3 scalar currency_coherence unregressed / C4 ALT pattern
  distinct from JOINT [the chosen design fork] / C5 monostrategy -> ~0 range
  non-vacuity / C6 default-OFF flag + legacy compose unchanged / C7 agent compose
  fires + routes to the authority via "coherence"). Smoke 2026-06-19: JOINT range
  0.1, SHUF None/0, ALT range 0.02 + joint-vs-alt pattern cosine 0.82.
  Phased training: N/A (pure-arithmetic per-stream action-cosine + weighting; no
  learned parameters). MECH-094: compose is read-only on the waking select_action
  path; the seal-time action-snapshot bookkeeping rides the existing
  simulation_mode-gated seal (no replay/memory write surface). Evidence-staleness
  (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment uses
  the default (per-candidate coherence off), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-294 stays candidate / substrate_ceiling /
  v3_pending; claims.yaml carries only an implementation_note (no flag/confidence/
  status change). The primary lit falsifier (Kay-2020 cross-cycle theta) is
  out-of-substrate for V3, so the V3 path is this per-candidate-range wiring + a
  downstream behavioural readout.
  Validation: the substrate-readiness gate is the contract suite (the carve-able
  channel now EXISTS: JOINT range>floor on a diverse candidate pool, SHUF ~0,
  scalar mode-discrimination preserved). NO behavioural EXQ queued here -- the
  MECH-294 behavioural falsifier on this channel is a SEPARATE later
  /queue-experiment step (per the task scope + the 661 implement-substrate log
  that called this MECH-294-side compose path out of scope).
  Design doc: REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md
  (per-candidate-coherence amend section). Triage:
  REE_assembly/evidence/planning/substrate_ceiling_lifted_triage_2026-06.md.
  See MECH-294 (parent; substrate 2026-06-09 + compose-coherence amend 2026-06-09
  above), modulatory-bias-selection-authority (route-range amend 2026-06-10 -- the
  "coherence" source this gives a mode-distinct per-candidate range; gain/contrast
  + margin-shortlist 2026-06-15 + top-k shortlist 2026-06-16 -- the 569i carve),
  V3-EXQ-661 (the scalar-coherence committed-dist TV ~0 FAIL this addresses),
  V3-EXQ-663 (route-range readiness gate that flagged coherence as no-range),
  MECH-089 (ThetaBuffer; state_summary source, untouched), MECH-269/269b
  (per-stream V_s vintaging that drives currency), Kay et al. 2020 (cross-cycle
  alternation control; out-of-substrate lit falsifier), MECH-094 (call-site
  scoping; preserved).

## MECH-189: Super-ordinal goal-anchor ContextMemory writes substrate (infant_substrate:GAP-11) (2026-06-09)
- MECH-189: development.super_ordinal_goal_formation -- IMPLEMENTED 2026-06-09
  (substrate; v3_pending until the validation EXQ PASSes). Closes the
  infant_substrate:GAP-11 / DEV-NEED-006 substrate gap routed via
  /implement-substrate: the "ContextMemory writes substrate" the V3-EXQ-588 FAIL
  (failure_autopsy_V3-EXQ-588_2026-05-19, non_contributory for MECH-189) was
  deferred to. 588 measured the within-episode GoalState attractor (MECH-112 /
  DEV-NEED-006), NOT the child-phase ContextMemory super-ordinal write path the
  claim describes -- and that path did not exist (developmental_needs_register.md:136:
  "MECH-189 needs cue-indexed persistent goal-anchor writes and an adult z_goal
  seeding readout before it can be gated"). Do NOT re-queue V3-EXQ-588; the
  MECH-189 retest is a 588 successor with a NEW letter.
  ROOT-CAUSE FRAMING: the IncentiveTokenBank (SD-057) is per-object + per-episode
  (GoalState.reset() clears it); the ghost-goal bank (MECH-292) is over
  hippocampal anchors; GoalState resets z_goal each episode. No CROSS-EPISODE
  super-ordinal z_goal store existed -- the persistence that distinguishes a
  childhood-formed goal hierarchy from an episodic z_goal.
  Module: ree_core/goal.py (new SuperOrdinalGoalMemory class + 11 GoalConfig
  fields, sibling to IncentiveTokenBank). AGENT-owned (REEAgent.super_ordinal_goal_memory),
  NOT reset by per-episode agent.reset() -- cross-episode persistence is the
  point. Pure stateful tensor store + cosine arithmetic; no nn.Module, no
  trainable parameters, no gradient flow; no phased training.
  Store: N cue-indexed slots, each (key = z_world context [world_dim], value =
  z_goal anchor [goal_dim], strength). Cue-indexed retrieval by cosine match.
  WRITE (child phase only, wired at agent.update_z_goal AFTER GoalState.update):
  the current z_goal is written keyed on the current z_world context iff the
  MECH-189 conjunction holds -- (a) high salience: salience = benefit_exposure *
  (1 + drive_weight * effective_drive) >= super_ordinal_salience_threshold (the
  "large benefit spike"; routine contacts do not clear it, transient-benefit
  patches do); AND (b) high contextual complexity >= super_ordinal_complexity_threshold.
  The complexity signal is PLUGGABLE (super_ordinal_complexity_mode) because it is
  the DEV-NEED-024 open question ("what contextual-complexity threshold triggers a
  write") -- to be adjudicated by the validation EXQ + a follow-on lit-pull, NOT
  hard-coded: "novelty" (default, self-contained: complexity = 1 - max cosine to
  occupied anchor keys; empty store -> 1.0 bootstraps, covered contexts -> low ->
  no write = selective neoteny/adult stability) or "external" (caller-supplied, so
  an experiment can drive it from E1 cue-context entropy / PE without coupling the
  substrate to those channels). ALLOCATE-vs-REINFORCE (gate (b) governs FORMATION
  only): gate (a) salience is required for any write; a contact within
  super_ordinal_merge_similarity of an existing anchor REINFORCES it (EMA blend
  toward the CURRENT z_goal, raise strength) on salience alone REGARDLESS of
  complexity -- a recurring high-salience context strengthens its meta-goal toward
  the matured z_goal; gate (b) complexity governs only ALLOCATION of a NEW anchor
  (empty slot, else weakest). (Without this split the anchor freezes at the tiny
  z_goal captured at its first contact -- surfaced + fixed by the V3-EXQ-588c
  smoke test: anchor norm 0.019 frozen -> 0.373 matured.) Writes permitted only
  while write_enabled -- the curriculum freezes them at child->adult via
  REEAgent.set_super_ordinal_write_enabled(False) (MECH-334 on_phase3_entry
  precedent).
  READ (adult z_goal seeding readout, wired at the TOP of agent.update_z_goal):
  when the live z_goal norm < super_ordinal_seed_below_norm (default 0.4, matching
  the DEV-NEED-006 gate), retrieve the best-matching anchor for the current
  z_world; if match >= super_ordinal_seed_match_threshold, pull z_goal toward it
  via GoalState.cue_pull (no benefit pulse) by super_ordinal_seed_strength -- the
  "stored anchors bias adult z_goal seeding across novel episodes" readout.
  Config (GoalConfig + REEConfig.from_dims, all no-op default -> bit-identical OFF):
  use_super_ordinal_goal_anchors (False, master), super_ordinal_n_slots (16),
  super_ordinal_salience_threshold (0.5), super_ordinal_complexity_mode ("novelty"),
  super_ordinal_complexity_threshold (0.3), super_ordinal_merge_similarity (0.8),
  super_ordinal_write_alpha (0.3), super_ordinal_seed_below_norm (0.4),
  super_ordinal_seed_match_threshold (0.3), super_ordinal_seed_strength (0.1).
  Agent methods: set_super_ordinal_write_enabled(bool) (curriculum freeze hook),
  reset_super_ordinal_anchors() (developmental-stage clear; NOT per-episode).
  Backward compatible: use_super_ordinal_goal_anchors=False by default ->
  agent.super_ordinal_goal_memory is None; both update_z_goal hooks skipped ->
  bit-identical (verified: default vs explicit-False z_goal identical; store stays
  None after update). 985/985 contracts + 7/7 preflight PASS; 8 new contracts in
  tests/contracts/test_mech_189_super_ordinal_goal_anchors.py (C1 default-OFF +
  agent no-op / C2 write conjunction / C3 reinforce-vs-allocate / C4 retrieve +
  complexity / C5 freeze + MECH-094 sim no-op / C6 cross-episode persistence +
  reset_anchors / C7 agent forms anchor on forced high-salience update_z_goal /
  C8 agent adult-seeding pulls a fresh sub-floor z_goal toward a frozen childhood
  anchor in a matching context).
  MECH-094: waking-only; SuperOrdinalGoalMemory.write(simulation_mode=True) is a
  no-op (replay/DMN must not form super-ordinal anchors). Evidence-staleness
  (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment uses
  the default (store off), so no dependent claim's measured mechanism changed.
  KEEP all evidence.
  GOVERNANCE: MECH-189 NEITHER promoted NOR weakened; stays candidate / confidence
  0.0; claims.yaml carries only an implementation_note. The illusory conflict_ratio
  from the 588 does_not_support stands as already adjudicated by the autopsy.
  Validation experiment: V3-EXQ-588c (NEW letter, supersedes the 588 chain's MECH-189
  framing; claim_ids=["MECH-189"], experiment_purpose=diagnostic, supersedes
  v3_exq_588_isef002...) -- child-phase forced-feed write across episodes -> freeze
  via set_super_ordinal_write_enabled(False) -> adult episodes with fresh sub-floor
  z_goal (goal_state.reset()) seed z_goal from anchors WITHOUT a benefit pulse;
  ARM_ON vs ARM_OFF x 3 seeds. LOAD-BEARING acceptance is DISCRIMINATION (ARM_ON adult
  median z_goal > DISCRIM_FLOOR 0.1 AND > per-seed ARM_OFF + 0.1 on >=2/3 seeds) -- the
  substrate question; the 588 / DEV-NEED-006 0.4 crossing is ADVISORY (reported, not
  gating) because the matured-z_goal anchor norm ceilings at ~0.37 on the
  untrained-encoder readiness harness (forced-feed z_world-EMA asymptote ~0.9*||z_world||),
  so 0.4 is regime-bound -- a near-miss with strong discrimination validates the substrate
  and routes the absolute gate to a trained-encoder evidence successor, NOT a failure.
  Readiness/non-vacuity (else substrate_not_ready_requeue): ARM_ON anchors form
  (n_occupied>0), seeding fires (n_seeds>0), positive-control adult z_goal>0. Dry-run PASS
  (C1 discrimination 1.0; ON 0.28 vs OFF 0.0). Queued via /queue-experiment.
  Design doc: REE_assembly/docs/architecture/mech_189_super_ordinal_goal_anchors.md
  Closure node: REE_assembly/evidence/planning/infant_substrate_plan.md
  (infant_substrate:GAP-11).
  See MECH-189 (this claim), SD-057 IncentiveTokenBank (distinct per-object/
  per-episode store; cue_pull reused for the seed), MECH-292/293 ghost-goal bank
  (distinct hippocampal-anchor store), GoalState (episodic z_goal attractor;
  unchanged), MECH-112/116/117 (GoalState claims), INV-037/038 (stored-vs-active),
  INV-041/055/056 (childhood prerequisite / selective neoteny), SD-016
  (cue-indexed ContextMemory semantics), MECH-329 (wanting-before-liking child of
  MECH-189), MECH-334 (crystallization freeze-hook precedent), V3-EXQ-588 (the FAIL
  this addresses), DEV-NEED-006 / DEV-NEED-024, MECH-094 (call-site scoping).

## MECH-219 (SD-019b): affective-harm hysteretic integrator (z_harm_suffering) (2026-06-10)
- MECH-219: affect.affective_harm_hysteretic_integration -- IMPLEMENTED 2026-06-10
  (substrate; SD-019b stays v3_pending until the validation EXQ PASSes). The
  tier-2 -> tier-3 step of the harm-affect hierarchy: SD-019a builds z_harm_un
  (symmetric EMA of z_harm_s, "make it stop" unpleasantness); MECH-219 turns it
  into a slow, persistent, controllability-gated SUFFERING load state
  (z_harm_suffering) -- the mechanism SD-019b (harm_stream.suffering_accumulator)
  names and is blocked on. Built from the landed design-first plan-of-record
  REE_assembly/evidence/planning/mech_219_hysteretic_integrator_design.md (Section 10
  checklist).
  Module: ree_core/affect/harm_suffering_accumulator.py (HarmSufferingAccumulator +
  HarmSufferingAccumulatorConfig + HarmSufferingAccumulatorOutput). Pure-arithmetic
  regulator (no nn.Module, no learned parameters, no gradient flow); sibling to the
  MECH-353 BlockedAgency / MECH-313 / MECH-320 / MECH-342 regulator pattern. Owns
  only the scalar suffering state s_t; the agent builds the z_harm_suffering vector.
  Dynamics (memo Section 4): u_t = ||z_harm_un|| (+ body_damage_weight*||z_harm_a||);
  g_t = 1 - escapability; drive_t = g_t*u_t (+ pe_gain*unsigned_PE); asymmetric
  accumulation s_t = s_{t-1} + alpha*(drive_t - s_{t-1}) with alpha=alpha_rise when
  building else alpha_fall (alpha_rise >> alpha_fall = the hysteresis); optional
  Schmitt bistable latch (theta_on/theta_off). Output z_harm_suffering = the
  z_harm_un direction scaled to magnitude s_t (same dim as z_harm_un).
  Escapability source PLUGGABLE (harm_suffering_escapability_mode): constant
  (default 1.0 -> g=0 -> inert/bit-identical) / avoidance_efficacy (reads SD-058
  InstrumentalAvoidanceGate.effective_efficacy() -- the literal escapability
  construct; soft dependency on the v3_pending SD-058 substrate) / external
  (REEAgent.set_harm_suffering_escapability() seam for the validation experiment).
  NEVER sourced from MECH-353 capacity_belief (= 1 - w*||z_harm_a||) to avoid the
  z_harm_a -> capacity_belief -> z_harm_a loop; capacity_belief is a validation
  cross-check only (memo Section 3 / R1).
  Module: ree_core/latent/stack.py -- LatentState.z_harm_suffering [batch, harm_dim]
  + detach() handling. ree_core/utils/config.py -- REEConfig fields +
  REEConfig.from_dims passthrough (use_harm_suffering_accumulator + alpha_rise/fall +
  escapability_mode/constant/external + s_cap + body_damage_weight + pe_gain +
  use_bistable_latch + theta_on/off + 5 per-consumer redirect flags).
  Agent wiring (ree_core/agent.py): self.harm_suffering_accumulator built when the
  master flag is on (PRECONDITION: requires use_harm_un=True -- loud ValueError
  otherwise, since z_harm_un is the drive input); ticked in sense() right after the
  SD-019a z_harm_un EMA and BEFORE the SD-032 consumers (AIC in sense; PAG/pACC in
  select_action) so a redirect reads the suffering output same-tick; z_harm_suffering
  vector built from the regulator's s_t; reset() clears it per episode;
  _resolve_harm_suffering_escapability() resolves the escapability scalar per mode;
  set_harm_suffering_escapability() drives the external mode.
  z_harm_a re-source migration (memo Section 6): use_harm_suffering_accumulator is the
  master flag; PER-CONSUMER redirect flags (harm_suffering_redirect_{aic,pag,mech091,
  dacc,pacc}) let the migration be staged + individually ablated. Redirects are
  MAGNITUDE-based (||z_harm_suffering||): v1 WIRES the urgency/PAG/interrupt consumers
  (AIC urgency aic_z_norm; PAG MECH-279 freeze drive pag_z_norm; MECH-091 urgency-
  interrupt _urgency_signal) per memo R3. The dACC/pACC flags are DEFINED but left
  UNWIRED (default off, currently no-ops): their E2_harm_a forward models are keyed on
  the current z_harm_a dim (z_harm_a_dim != harm_dim), so they migrate last after
  measuring R^2 -- v1 keeps them on legacy z_harm_a (memo R3). Body-damage fold-in
  (memo Section 6 fork b): body_damage_weight (default 0.0) folds ||z_harm_a|| into the
  drive so SD-022 / EXQ-319 / EXQ-323a non-redundancy evidence is preserved, not
  orphaned.
  Backward compatible: use_harm_suffering_accumulator=False by default ->
  agent.harm_suffering_accumulator is None, LatentState.z_harm_suffering stays None,
  no consumer redirect fires -> bit-identical (default == explicit-False action stream
  verified). Also inert under the default escapability_mode=constant=1.0 (g=0 -> s->0)
  even when explicitly enabled. 7/7 preflight + full contract suite green (the one
  scaffolded_sd054 C11 reef-spawn failure is a pre-existing env-OS-entropy flake --
  _build_env passes no seed to CausalGridWorldV2(np.random.default_rng(None)) -- ~10%
  failure rate on clean HEAD too; unrelated to MECH-219). 11/11 new contracts in
  tests/contracts/test_mech_219_harm_suffering_accumulator.py.
  Activation smoke (2026-06-10): constant esc=1.0 -> s=0 even at u=1.2 (inert OFF);
  external esc=0.0 (inescapable) -> suffering rises fast (alpha_rise=0.2) to plateau;
  then esc=1.0 (relief) -> SLOW fall (alpha_fall=0.01, after 40 steps still > half-peak
  = hysteresis); body-damage fold-in u=2.0 at body_damage_weight=0.5; bistable latch
  flips at theta_on/off; avoidance_efficacy reads SD-058 effective_efficacy; AIC
  redirect tracks the suffering channel.
  Phased training: N/A (pure-arithmetic regulator; no learned parameters). MECH-094:
  update() no-op under simulation_mode (hypothesis_tag), so replay/DMN ticks do not
  accumulate suffering on imagined outcomes.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (accumulator off), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  Validation experiment: NOT queued in this build session (separate /queue-experiment
  session per the memo Section 7 sketch -- the controllability dissociation falsifier
  C1/C2/C3/C4 under escapable vs inescapable matched-nociception, driving escapability
  from SD-058 effective_efficacy() or an external scripted schedule). PASS clears the
  SD-019b / Q-036 gate.
  Design doc: REE_assembly/docs/architecture/mech_219_hysteretic_integrator.md
  (status flipped plan-of-record -> IMPLEMENTED 2026-06-10). Plan-of-record:
  REE_assembly/evidence/planning/mech_219_hysteretic_integrator_design.md.
  See MECH-219, SD-019b (the suffering_accumulator claim this builds), SD-019a
  (z_harm_un, the tier-2 input + the redirect precedent mirrored), SD-019 (parent
  nonredundancy), SD-011 (z_harm_s / z_harm_a streams), SD-021 (descending modulation;
  controllability parity preserved -- escapability is NOT the attenuation factor),
  SD-022 (body-damage; folded into the drive), SD-058 / MECH-357 (escapability source;
  soft depends_on), MECH-353 (blocked-agency; opposite controllability pole, anti-
  correlation cross-check), SD-032b dACC / SD-032e pACC (z_harm_a forward-model
  consumers; migrate last), MECH-279 PAG / SD-032c AIC / MECH-091 (v1 redirect
  consumers), Q-036 (variable adjudication; resolved-by-design pending validation),
  MECH-094 (simulation gate).

## MECH-423 readiness substrate: R2 iterative-inference convergence + R3 interleaved cross-module consolidation + R1 shared-latent grad probe (2026-06-12)
- MECH-423 super-additivity readiness instrumentation -- IMPLEMENTED 2026-06-12
  (substrate readouts; MECH-423 stays candidate / v3_pending -- this PROMOTES
  NOTHING, it only unblocks EXP-0380 from blocked_substrate). Routed by the
  2026-06-12 /queue-experiment Step-2.5 substrate gate that found the R1/R2/R3
  readiness readouts the EXP-0380 acceptance_checks require ABSENT from the live
  ree-v3 eval path (so the super-additivity ablation would self-route
  substrate_not_ready_requeue every run = vacuous probe). Three capabilities, all
  no-op-default + bit-identical OFF + contract-tested; MECH-094 simulation-gating
  on the replay/consolidation path; pure-arithmetic readouts. Amends the ARC-004
  inference machinery (R2) + MECH-121 consolidation cluster (R3) -- implementation_note
  only, NO new claim minted.
  R2 (iterative-inference convergence; ARC-004): ree_core/latent/stack.py
    (LatentStack.encode) + ree_core/utils/config.py (LatentStackConfig). The legacy
    encode() is a FIXED two-pass amortized recognition (bottom-up init -> ONE
    top-down round): no settling loop, so the per-inference-step ||delta z_shared||
    the EXP-0380 R2 check needs had no source. When use_iterative_inference=True the
    single top-down round is generalised into a predictive-coding settling loop over
    the shared z_beta -> z_theta -> z_delta stack -- iterate the recurrent top-down
    map with the bottom-up data terms (combined_init/z_beta_init/body_obs/world_obs)
    held fixed -- run BEFORE the SD-007 reafference correction + EMA, tracking per-round
    ||delta z_shared|| / (||z_shared|| + eps) with early-stop at rel_tol. Readout:
    LatentState.inference_convergence (plain-float dict per_step_rel_delta[]/converged/
    n_iters/final_rel_delta), cached agent.last_inference_convergence; detach()
    passes it through. Config (LatentStackConfig, no-op): use_iterative_inference
    (False), inference_settle_iters (1), inference_convergence_rel_tol (0.05),
    surfaced via REEConfig.from_dims kwargs.pop (signature-stable). Grounds Gershman
    & Goodman 2014 (amortization gap). EXP-0380 R2 reads final_rel_delta < 0.05.
    Smoke: OFF bit-identical; ON settles 3 rounds 0.062 -> 0.0074 converged=True.
  R3 (module-tagged interleaved cross-module consolidation; MECH-121):
    NEW ree_core/sleep/cross_module_consolidation.py (CrossModuleConsolidator +
    CrossModuleConsolidatorConfig). The legacy MECH-121 offline pass trains
    e2_harm_s ALONE (sleep/phase_manager.py) over region-keyed traces with no module
    identity, so cross_module_replay_share is unmeasurable and the integrated E1<->E2
    representation cannot be acquired under interleaving. The consolidator takes named
    modules + loss closures + param lists, runs a configurable schedule, tags each
    replayed trace by which modules it ACTUALLY updated (a loss returning an
    exactly-zero sentinel == no replay content -> module not touched, so the share
    reflects genuine integration not the schedule label), and returns a flat readout
    {n_updates, n_traces, n_cross_module_traces, cross_module_replay_share, interleaved,
    updates_<name>}. "interleaved" runs one step per module per trace (a trace can
    touch >1 module -> share 1.0); "blocked" trains modules sequentially (each trace
    one module -> share 0.0, the catastrophic-interference control). Wiring: built on
    the agent (agent.cross_module_consolidator) when use_cross_module_consolidation=True
    (standalone-usable by the experiment) AND passed to SleepLoopManager, where a
    flag-gated hook in _run_cycle (AFTER the existing writeback, additive) runs the
    default E1 (compute_prediction_loss over _world_experience_buffer) + E2
    (compute_e2_loss over _e2_transition_buffer) loss set and merges
    cross_module_consolidation_* into the sleep-cycle metrics -- a readout of the LIVE
    MECH-121 pipeline. Config (REEConfig, no-op): use_cross_module_consolidation
    (False), cross_module_consolidation_schedule ("interleaved"), _steps (0 == none),
    _lr (1e-3), _batch (16), surfaced via from_dims kwargs.pop. MECH-094: the SAME
    explicit exception the e2_harm_s writeback uses -- per-module optimisers
    constructed LOCALLY over only the named modules' params; NO residue/anchor/memory
    write; simulation_mode returns the zeroed no-op (the offline call site passes
    False). Grounds McClelland 1995 + Kumaran 2016 CLS (interleaving necessary for
    shared-rep integration; blocked schedule -> catastrophic interference ->
    sub-additive ARTEFACT, so "blocked" is the pre-registered control not a bug).
    EXP-0380 R3 reads n_updates>0 AND cross_module_replay_share>0 AND interleaved==True.
    Smoke: OFF consolidator None; interleaved share 1.0 / 8 updates; blocked share 0.0;
    sim no-op; live force_cycle merges the keys share 1.0.
  R1 (shared-latent gradient probe; reusable utility): NEW
    ree_core/utils/shared_latent_probe.py (shared_latent_gradient_probe). Pure
    function (no substrate state, no hot-path touch): given z_shared + a
    {module: loss_fn(z_shared)} map, computes d(loss)/d(z_shared) per module via
    torch.autograd.grad(retain_graph=True) and returns {per_module_grad_norm,
    min_grad_norm, mean_pairwise_cosine, n_modules, coupled}. EXP-0380's integrated
    arm constructs z_shared (latent fed jointly to E1+E2) and reads the R1 verdict
    (min_grad_norm>0 AND mean_pairwise_cosine>=0). Grounds Yu/PCGrad 2020 (conflicting
    gradients = negative transfer) + Caruana 1997 (shared-rep MTL helps only when
    tasks related). Built per user decision 2026-06-12 (reusable hook in substrate).
  Backward compatible: all flags default no-op; full ree-v3 contract suite green with
    everything OFF (1013 prior PASS + 12 new MECH-423 contracts; the 1 control_vector
    C4 failure is the documented pre-existing baseline flake -- CONFIRMED still failing
    on a clean stash of these changes, PASSES with them in isolation). New contracts:
    tests/contracts/test_mech423_inference_convergence.py (C1 OFF bit-identical / C2 ON
    readout / C3 settling reduces delta / C4 settle_iters=1 == OFF / C5 detach passthrough)
    + tests/contracts/test_mech423_cross_module_consolidation.py (C1 OFF None / C2 config
    validation / C3 interleaved share 1.0 / C4 blocked share 0.0 / C5 single-module-data
    share 0.0 / C6 sim no-op / C7 SleepLoopManager hook merges the readout).
  Phased training: N/A (R2 is a settling-loop readout over the existing encoder, no new
    learned head; R3 reuses the already-trained E1/E2 module losses; R1 is a pure
    autograd probe). Evidence-staleness (Step 8.5): NOT triggered -- no-op-default
    flags; every existing experiment uses the defaults (no iterative inference, no
    cross-module consolidation), so no dependent claim's measured mechanism changed.
    KEEP all evidence.
  Validation experiment: V3-EXQ substrate-readiness diagnostic (claim_ids=[]; asserts
    R2 final_rel_delta < 0.05 + R3 interleaved share>0 vs blocked share=0 + R1 coupling)
    queued via /queue-experiment. PASS confirms the readiness readouts are non-vacuous
    on a trained substrate. EXP-0380 (the super-additivity ablation) is the SEPARATE
    /queue-experiment session this unblocks (flipped blocked_substrate -> proposed).
  Design doc: REE_assembly/docs/architecture/mech_423_superadditivity_readiness_substrate.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_mech_423_integration_prerequisites.
  See MECH-423 (the claim this unblocks; PROMOTES NOTHING), ARC-004 (shared L-space
    latent / inference machinery -- R2 amend), MECH-121 (NREM consolidation cluster --
    R3 amend), MECH-273 / SelfModelAggregator (the single-module e2_harm_s offline pass
    R3 generalises to interleaved E1<->E2), ARC-001/002 (E1/E2 streams), MECH-081/082/033
    (pairwise transfer paths super-additivity generalises), ARC-080 (object spine; triple
    arm), EXP-0380 (the super-additivity ablation), MECH-094 (call-site scoping).

## SD-061: difficulty-gated proposal-entropy regulator (stuck-state detector + transient CEM proposal-widening; MECH-343 blocker part 2 / Q-056) (2026-06-19)
- SD-061: control_plane.difficulty_gated_proposal_entropy -- IMPLEMENTED 2026-06-19
  (substrate; MECH-343 stays candidate / substrate_conditional / v3_pending -- this
  PROMOTES NOTHING, it builds the missing substrate the mechanism is blocked on). Routed
  by the Q-054/Q-055/Q-056 buildability triage (REE_assembly/evidence/planning/
  q054_q055_q056_buildability_triage_2026-06-19.md): MECH-343's evidence_quality_note
  names two blockers -- (1) modulatory-bias-selection-authority (NOW implemented, 569i
  top-k) and (2) "a difficulty-gated proposal-entropy regulator (stuck-state detector +
  transient CEM temperature/candidate-count gain + decay) not yet designed." SD-061 is (2).
  Two coupled no-op-default modules (OFF = bit-identical):
    (1) ree_core/cingulate/stuck_state_detector.py (StuckStateDetector +
      StuckStateDetectorConfig) -- integrates goal-progress stall (GoalState.goal_proximity
      window) + E3 first-action score margin + committed-action-class lock-in (window) +
      dACC choice_difficulty (inverted: small EV spread = hard) into a graded stuck_score
      in [0,1] + binary is_stuck, GUARDED by goal salience (no goal -> 0; the
      stuck-WITH-goal distinction). Present-axis deficits combine by mean|max; asymmetric
      EMA (ema_alpha_rise >> ema_alpha_fall) -> fast rise, slow decay (the MECH-343
      "entropy narrows once a workable candidate is found" hysteresis).
    (2) ree_core/policy/difficulty_gated_proposal_entropy.py (DifficultyGatedProposalEntropy
      + Config) -- maps stuck_score to a transient PROPOSAL-layer gain:
      extra_candidates = round(candidate_widen_max * s); temperature_gain = 1 +
      temperature_gain_max * s. Identity at s=0.
  Pure-arithmetic regulators (no nn.Module, no learned params, no gradient flow); sibling
  to MECH-313 NoiseFloor / MECH-320 TonicVigor / MECH-342 CommitMaintenanceRelease.
  Wiring (ree_core/agent.py): both built in __init__ when
  use_difficulty_gated_proposal_entropy=True (else None); self._last_stuck_score lag seam.
  _e3_tick applies the gain to HippocampalModule.propose_trajectories (num_candidates +=
  extra; differentiable_cem_temperature *= gain, transient, restored in finally).
  select_action updates the detector EVERY tick (after the maintenance_release block, not
  gated on beta elevation) from e3.last_scores margin + _dacc_last_bundle choice_difficulty
  + goal_state proximity/norm + e3._committed_trajectory first-action class ->
  _last_stuck_score (one-tick lag the next _e3_tick reads). reset() clears both + the lag.
  Scoring / commitment (MECH-090/342) / selection authority (569i top-k / MECH-341) are
  UNTOUCHED -- a hard problem widens proposals, not behaviour.
  Config (REEConfig + from_dims, all no-op default): use_difficulty_gated_proposal_entropy
  (False) + stuck_progress_window (8) + stuck_progress_stall_eps (0.01) +
  stuck_score_margin_floor (0.05) + stuck_committed_diversity_window (8) +
  stuck_committed_diversity_floor (0.34) + stuck_choice_difficulty_ref (0.05) +
  stuck_goal_salience_floor (0.05) + stuck_ema_alpha_rise (0.3) + stuck_ema_alpha_fall
  (0.05) + stuck_threshold (0.5) + stuck_combine_mode ("mean") + dgpe_candidate_widen_max
  (8) + dgpe_temperature_gain_max (1.0).
  Backward compatible: use_difficulty_gated_proposal_entropy=False by default -> both
  modules None; the _e3_tick gain + select_action detector-update blocks skipped ->
  bit-identical (verified: default == explicit-False action stream). preflight 8/8 + 8 new
  contracts (tests/contracts/test_sd_061_difficulty_gated_proposal_entropy.py: C1
  default-OFF bit-identical / C2 rises under impasse-with-goal / C3 goal-salience guard /
  C4 hysteretic decay / C5 regulator gain mapping + clamp / C6 MECH-094 sim no-op / C7
  agent build+tick+reset / C8 from_dims surfaces knobs). Activation smoke 2026-06-19:
  detector rises to 0.90 under sustained impasse-with-goal, decays to 0.12 after relief,
  stays 0.0 with no goal; regulator s=1 -> (8 extra, 2.0x temp); agent OFF/ON both run
  end-to-end (ON: detector ticks 30, regulator called 5).
  Phased training: N/A (pure-arithmetic; no learned parameters). MECH-094: both modules'
  state-advancing methods no-op under simulation_mode (replay must not accumulate waking
  impasse or widen an imagined proposal). Evidence-staleness (Step 8.5): NOT triggered --
  no-op-default flag; every existing experiment uses the default (regulator off), so no
  dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-343 stays candidate / substrate_conditional /
  v3_pending; claims.yaml carries only the new SD-061 registration + an implementation_note
  (no MECH-343 status/flag change).
  Validation experiment: a substrate-readiness diagnostic (claim_ids=[]; regulator OFF vs
  ON under an induced stuck-state, confirming candidate_first_action_entropy rises under
  stuck + decays after) queued via /queue-experiment. The Q-056 3-arm governance falsifier
  (off / stuck-gated / always-high, matched easy/hard controls) is a SEPARATE later session
  once this readiness check PASSes.
  Design doc: REE_assembly/docs/architecture/sd_061_difficulty_gated_proposal_entropy.md
  Triage: REE_assembly/evidence/planning/q054_q055_q056_buildability_triage_2026-06-19.md
  See MECH-343 (parent mechanism; the substrate_conditional blocker part 2 this builds),
  modulatory-bias-selection-authority (blocker part 1; implemented 569i top-k), ARC-018
  (HippocampalModule proposal locus the gain widens), MECH-341 / ARC-062 (selection-side
  diversity; untouched), MECH-090 / MECH-342 (commitment predicates; untouched), SD-032b
  dACC choice_difficulty (a detector input), MECH-313 (state-independent action-selection
  noise floor; DISTINCT), Q-056 (the falsifier), MECH-094 (call-site scoping).

## DR-12 (self_model_v4:SELF-4): E2 forward-PE -> E3 trajectory-scoring confidence down-weight (FIRST V4 SUBSTRATE BUILD, 2026-06-17)
- DR-12: ethics_engine_3.pe_conditioned_confidence_weighting -- IMPLEMENTED 2026-06-17.
  THE FIRST-EVER V4 substrate build (generation:v4, off the V3 critical path; PROMOTES
  NOTHING in V3). User-APPROVED via self_model_v4_plan.md SELF-4 graduation_decision_2026_06_16.
  Modules: ree_core/predictors/e3_selector.py (E3TrajectorySelector._pe_confidence_penalty
  helper + score_trajectory penalty term + select() per-candidate threading + 4 diagnostics),
  ree_core/utils/config.py (E3Config 4 fields + REEConfig.from_dims passthrough),
  ree_core/agent.py (_injected_e2_forward_pe attr + set_injected_e2_forward_pe() seam +
  select_action passthrough).
  Problem (v4_spec V4-2 DR-12): E3.score_trajectory scores purely from the E2-rolled-out
  world_states, trusting them as if E2 were reliable everywhere; high E2 forward-PE in a
  trajectory's region does NOT discount that trajectory. E3 already CONSUMES two
  PE-magnitude signals for its own dynamics (_running_variance, ARC-016 world-forward PE;
  _novelty_ema, MECH-111 E1 PE) -- DR-12 adds an E2-forward-PE confidence down-weight
  ALONGSIDE them. A NEW lever on EXISTING machinery; no learned parameters; no stateful
  z_self substrate (keys off PE magnitude present in V3 today).
  THE LEVER (no-op default; bit-identical OFF): in score_trajectory (score is a COST,
  lower-is-better), when use_pe_confidence_weighting AND a per-trajectory e2_forward_pe is
  supplied AND pe_confidence_weight != 0.0:
    score = score + pe_confidence_weight * penalty(e2_forward_pe)
  penalty monotone non-decreasing in PE magnitude (clamped >=0): mode "linear"
  (penalty=pe) | "saturating" (penalty = 1 - exp(-pe/pe_confidence_scale) in [0,1)).
  Threaded PER-CANDIDATE via select(e2_forward_pe_per_candidate=[K]) so a varying PE can
  change the committed argmin -- a UNIFORM scalar is argmin-invariant (the V3-EXQ-571
  deleted-broadcast lesson; C3 contract pins this). Diagnostics on last_score_diagnostics:
  pe_confidence_active, pe_confidence_weight, e2_forward_pe_range (the pilot's non-vacuity
  gate -- a flat PE cannot change selection), pe_confidence_penalty_range.
  Config (E3Config + REEConfig.from_dims, all no-op default): use_pe_confidence_weighting
  (False, master), pe_confidence_weight (0.0), pe_confidence_mode ("linear"),
  pe_confidence_scale (1.0). Default OFF -> the penalty block is skipped entirely ->
  bit-identical.
  Per-candidate PE source (v1 scope, user-confirmed AskUserQuestion 2026-06-17):
  CALLER-SUPPLIED. The lever consumes a per-candidate PE passed into select(); the DR-12
  pilot (V4-EXQ-001) is a CONTROLLED substrate-readiness probe (assigns known high/low
  per-candidate PE, tests ON-vs-OFF selection divergence). REEAgent.select_action plumbs
  an optional injected per-candidate PE (agent._injected_e2_forward_pe via
  set_injected_e2_forward_pe(); default None -> bit-identical) so the lever is reachable
  from the waking loop. DOCUMENTED FOLLOW-ON (NOT v1): an ecological region-PE auto-source
  (extend the existing global _running_variance EMA into a region-keyed E2-forward-PE map
  looked up per-trajectory) -- the only piece that adds new state, deferred to keep v1 a
  "lever on existing machinery."
  Backward compatible: use_pe_confidence_weighting=False by default. preflight 7/7 +
  from_dims activation smoke PASS; 8 new contracts in tests/contracts/test_dr12_pe_confidence.py
  (C1 OFF/weight-0/no-PE all bit-identical / C2 high-PE-on-primary-best flips selection /
  C3 uniform-PE argmin-invariant / C4 linear monotone == weight*pe + saturating bounded
  by weight / C5 negative-PE clamped no-reward). Full contract suite re-run for the OFF
  bit-identical guarantee.
  Phased training: N/A (no learned parameters; pure arithmetic on an existing PE magnitude).
  MECH-094: N/A -- waking action-selection scoring; no replay/memory write surface.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default lever; every existing
  experiment uses the default (lever off), so no dependent claim's measured mechanism
  changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-215 (the claim DR-10+DR-12 unblock) stays candidate /
  implementation_phase=v4 / unchanged -- DR-12 alone does not unblock it (DR-10 + experiments
  remain), so v3_pending/status untouched; claims.yaml NOT modified.
  PRECEDENT (first V4 experiment): the DR-12 pilot V4-EXQ-001 sets a NEW V4 architecture_epoch
  (per v4_spec.md:267) + V4 run_id suffix + assigns owner_exq to the SELF-4 node WHEN queued.
  Verified the generation-aware consumers keep a generation:v4 node with an owner_exq OUT of
  the V3 closure %: check_closure_drift.py:497 skips non-v3 plans; generate_closure_snapshot.py
  + serve.py read_closure segment v4 into a separate roadmap rollup.
  Validation experiment: V4-EXQ-001 (DR-12 pilot) queued via /queue-experiment -- the
  controlled probe; FALSIFIER = if PE-conditioned weighting does NOT change selection in
  high-PE regions vs the unconditional-trust baseline, DR-12 buys nothing and the wiring is
  inert (pre-registered non-vacuity gate + inert-wiring off-ramp). unblocks_claims=MECH-215.
  Design doc: REE_assembly/docs/architecture/dr12_pe_conditioned_e3_confidence.md.
  Plan node: REE_assembly/evidence/planning/self_model_v4_plan.md (self_model_v4:SELF-4).
  See MECH-215 (unblocked; E2 self-transition-accuracy half), ARC-016 (E3 dynamic precision
  -- extended to the E2 stream), MECH-111 (sibling E1-novelty PE), SD-056 (trained E2
  forward divergence -- the source of a meaningful forward-PE), DR-10/SELF-3 (the
  z_self-in-E3 half of the MECH-215 unblock), MECH-094 (N/A).

## DR-13 (self_model_v4:SELF-1): z_self temporal depth -- dedicated self-recurrence anchored by E1 feedback (2026-07-01)
- DR-13: latent_stack.self_recurrence -- IMPLEMENTED 2026-07-01.
  The SUBSTRATE FLOOR of self_model_v4 (generation:v4, off the V3 critical path; PROMOTES
  NOTHING in V3). User-directed build to unblock SELF-3 (DR-10) after the 2026-07-01 IGW-165
  reconcile marked SELF-3 blocked on exactly this substrate. Mechanism resolved on ARC-081
  notes 2026-06-14 (HYBRID -- both motifs committed).
  Modules: ree_core/latent/self_recurrence.py (NEW: SelfRecurrenceCell, a GRUCell over
  z_self), ree_core/latent/stack.py (LatentStack.__init__ conditional instantiation +
  encode() self_e1_anchor param + the recurrence-replaces-EMA lever + LatentState
  .self_recurrence_diag readout), ree_core/utils/config.py (LatentStackConfig 2 fields +
  REEConfig.from_dims passthrough), ree_core/agent.py (_e1_predicted_next_z_self cache in
  _e1_tick + sense() anchor passthrough + reset).
  Problem (v4_spec V4-2 DR-13): z_self = body_obs -> MLP -> fixed-alpha EMA is an
  instantaneous body snapshot; a fixed-decay EMA cannot selectively retain/gate self-state,
  so there is no stable, inspectable, lesionable SUBJECT for DR-10 (z_self-in-E3), DR-11
  (self-state goals) and the INV-064 maturational-stability gate to attach to.
  THE LEVER (no-op default; bit-identical OFF): when use_self_recurrence, encode() REPLACES
  the z_self EMA step ONLY (z_world/z_beta/z_theta/z_delta smoothing untouched) with
    h = SelfRecurrenceCell(z_self_instant, prev.z_self)            # gated recurrence, hidden = prev stateful z_self
    z_self = (1 - c) * h + c * self_e1_anchor  if anchor & c>0     # E1-feedback blend
           = h                                 otherwise           # pure recurrence
  c = self_recurrence_e1_coupling (the recorded residual tunable): 0 = pure recurrence
  (Option A, max stability-isolation) | 1 = pure E1-feedback (Option B) | 0.15 light default
  = HYBRID. The cell is perturbation-ISOLATED (only z_self flows through it -- a +5.0
  perturbation of prev.z_self leaks 0.0 into z_world; contract C5). Anchor SOURCE (v1 scope,
  user-confirmed AskUserQuestion 2026-07-01 "Proceed as planned"): the E1 PREDICTED-NEXT
  z_self, cached at _e1_tick (predictions[:,0,:] -> split_prediction[0], detached) and
  consumed on the next encode() -- side-effect-free (no extra E1 forward, no LSTM
  hidden-state mutation), the volatility_signal plumbing precedent. First tick has no cache
  -> anchor None -> pure recurrence that step. DOCUMENTED FOLLOW-ON (NOT v1): an ecological
  anchor computed inline at encode without the one-tick lag.
  Diagnostics on LatentState.self_recurrence_diag (None when OFF): active, state_departure
  (||stateful z_self - instantaneous z_self||, batch-mean -- the DR-13 non-vacuity readout),
  e1_coupling, anchor_present.
  Config (LatentStackConfig + REEConfig.from_dims, all no-op default): use_self_recurrence
  (False, master), self_recurrence_e1_coupling (0.15). Default OFF -> self_recurrence NOT
  instantiated (mirrors reafference_predictor) + verbatim legacy EMA -> bit-identical.
  Backward compatible: disabled by default; existing experiments unaffected. Full suite
  1336 passed / 4 pre-existing failures (control_vector C4 flake + 2 runner-fail-branch +
  sd016 E1-proj-dim -- all confirmed identical on the clean tree via git stash) + 11 new
  contracts in tests/contracts/test_dr13_self_recurrence.py (C1 OFF bit-identical incl.
  anchor-ignored + deterministic / C2 ON module+diag / C3 state_departure>0 / C4 anchor
  blend exact + shape-mismatch fallback / C5 self perturbation does not leak to z_world /
  C6 coupling=0 ignores anchor / agent-level: OFF never caches anchor, ON caches + runs
  end-to-end).
  Phased training: the GRUCell trains via the EXISTING E1/E2 z_self prediction losses (v1
  adds NO new loss; the anchor is an inference-time detached blend). Standard joint-collapse
  awareness applies; experiments training the recurrence should follow P0 warmup -> P1
  frozen-encoder phasing.
  MECH-094: N/A -- waking perception path (encode/sense/_e1_tick); no replay/memory write
  surface.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default lever; every existing
  experiment uses the default (OFF), so no dependent claim's measured mechanism changed.
  KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. ARC-081 (mechanism resolved on its notes) + MECH-215 stay
  candidate / implementation_phase=v4 -- DR-13 is the substrate half; DR-10 (SELF-3) +
  experiments remain before MECH-215 unblocks. claims.yaml gets an implementation_note only
  (no status/v3_pending change).
  Validation experiment: V4-EXQ-002 (DR-13 self-recurrence substrate-readiness falsifier)
  queued 2026-07-01 (ree-v3 main 4c24214; coordinator /queue/add applied + /queue/active
  PRESENT; smoke PASS hist/iso/anchor 1/1) -- ON vs OFF; FALSIFIER = if the stateful z_self
  does not depart from the instantaneous encode / carries no history / lesioning the
  recurrent hidden state does not change it, DR-13 buys nothing (pre-registered non-vacuity
  gate on state_departure + inert off-ramp). unblocks_claims=ARC-081/MECH-215.
  Design doc: REE_assembly/docs/architecture/dr13_self_recurrence_temporal_depth.md.
  Plan node: REE_assembly/evidence/planning/self_model_v4_plan.md (self_model_v4:SELF-1).
  See ARC-081 (self-as-object pillar, mechanism source), MECH-215 (unblocked half), SD-005
  (the split z_self this upgrades), DR-10/SELF-3 (was blocked on this build), DR-12/SELF-4
  (E2-PE->E3, already built, E-stream-native), INV-064/SELF-7 (maturational-stability gate).

## DR-10 (self_model_v4:SELF-3): z_self enters E3 trajectory viability scoring (2026-07-01)
- DR-10: ethics_engine_3.self_viability_weighting -- IMPLEMENTED 2026-07-01.
  The E3-scoring half of the MECH-215 unblock, built on the same-day DR-13 stateful z_self
  (SELF-1, validated V4-EXQ-002 PASS). generation:v4, off the V3 critical path; PROMOTES
  NOTHING in V3. User-approved graduation (AskUserQuestion 2026-07-01, caller-supplied v1).
  Modules: ree_core/predictors/e3_selector.py (E3TrajectorySelector._self_viability_penalty
  helper + score_trajectory penalty term + select() per-candidate threading + 4 diagnostics),
  ree_core/utils/config.py (E3Config 4 fields + REEConfig.from_dims passthrough),
  ree_core/agent.py (_injected_self_viability attr + set_injected_self_viability() seam +
  select_action version-layering-guarded passthrough).
  Problem (v4_spec V4-2 DR-10): E3.score_trajectory scores purely over z_world (F/M/goal);
  there is NO z_self term in viability, so a trajectory scores identically whether the agent
  is fresh or depleted/damaged. DR-10 makes bodily capacity/affect/damage state (read from
  the DR-13 stateful z_self) gate which trajectories are viable for THIS agent. Sibling lever
  to DR-12 on EXISTING machinery; no learned parameters; needs a STABLE z_self subject
  (hence the DR-13/SELF-1 dependency).
  THE LEVER (no-op default; bit-identical OFF): in score_trajectory (score is a COST,
  lower-is-better), when use_self_viability_weighting AND a per-trajectory self_viability is
  supplied AND self_viability_weight != 0.0:
    score = score + self_viability_weight * penalty(self_viability)
  penalty monotone non-decreasing in the cost (clamped >=0): mode "linear" (penalty=sv) |
  "saturating" (penalty = 1 - exp(-sv/self_viability_scale) in [0,1)). Threaded PER-CANDIDATE
  via select(self_viability_per_candidate=[K]) so a varying cost can change the committed
  argmin -- a UNIFORM scalar is argmin-invariant (V3-EXQ-571 lesson; C3 contract pins this).
  Diagnostics on last_score_diagnostics: self_viability_active, self_viability_weight,
  self_viability_range (the pilot's non-vacuity gate), self_viability_penalty_range.
  Config (E3Config + REEConfig.from_dims, all no-op default): use_self_viability_weighting
  (False, master), self_viability_weight (0.0), self_viability_mode ("linear"),
  self_viability_scale (1.0). Default OFF -> the penalty block is skipped -> bit-identical.
  Per-candidate self-viability source (v1 scope, user-confirmed AskUserQuestion 2026-07-01):
  CALLER/AGENT-SUPPLIED, derived from the DR-13 stateful z_self. REEAgent.select_action plumbs
  an optional injected per-candidate self-viability (agent._injected_self_viability via
  set_injected_self_viability(); default None -> bit-identical; version-layering guard so the
  default V3 path never sends the kwarg). DOCUMENTED FOLLOW-ON (NOT v1): an ecological
  z_self-derived auto-source (allostatic z_self-deviation x per-candidate demand, or a learned
  z_self->viability head needing phased training + SELF-2's per-candidate self-transition).
  Backward compatible: disabled by default; existing experiments unaffected. preflight/full
  suite green + 8 new contracts in tests/contracts/test_dr10_z_self_viability.py (C1 OFF /
  weight-0 / no-signal all bit-identical / C2 differential flips selection / C3 uniform
  argmin-invariant / C4 linear==cost + saturating bounded-monotone + negative clamped).
  Phased training: N/A (no learned parameters; pure arithmetic on a supplied per-candidate cost).
  MECH-094: N/A -- waking action-selection scoring; no replay/memory write surface.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default lever; every existing
  experiment uses the default (OFF). KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-215 (unblocked by DR-10 + DR-12) and ARC-081 stay
  candidate / implementation_phase=v4; DR-10 is the z_self-viability half -- experiments +
  the ecological source remain, so v3_pending/status untouched; claims.yaml gets
  implementation_notes only.
  Validation experiment: V4-EXQ-003 (DR-10 pilot) queued via /queue-experiment -- controlled
  caller-supplied self-viability probe; FALSIFIER = if a decisive per-candidate self-viability
  does NOT change selection vs OFF, DR-10 buys nothing (pre-registered non-vacuity + decisiveness
  gates + inert off-ramp). unblocks_claims=MECH-215/ARC-081.
  Design doc: REE_assembly/docs/architecture/dr10_z_self_in_e3_viability.md.
  Plan node: REE_assembly/evidence/planning/self_model_v4_plan.md (self_model_v4:SELF-3).
  See DR-13/SELF-1 (the stateful z_self subject), DR-12/SELF-4 (sibling lever), MECH-215 +
  ARC-081 (unblocked halves), ARC-016 (E3 dynamic precision precedent), MECH-094 (N/A).

## MECH-448 / ARC-107: rank-preserving F->eligibility demotion (LEAD lever of the basal-ganglia E3-selector constitution) (2026-06-20)
- MECH-448: ethics_engine_3.rank_preserving_f_to_eligibility_demotion -- IMPLEMENTED
  2026-06-20 (substrate; MECH-448 stays candidate -- this PROMOTES NOTHING. The
  689a-successor falsifier is the sequenced next step, NOT queued here). THE FIRST major
  worked application of ARC-106 (brain-like construction: grounding ladder + divergence
  ledger + load-bearing-vs-decorative ablation falsifier + psychiatric failure mode).
  Routed by the user-adjudicated failure_autopsy_V3-EXQ-689a_2026-06-20 Step-8 decision
  ("elevate the constitutional build"; the conflict-grade near-tie parametric family --
  MECH-447 -- is exhausted). Design note:
  REE_assembly/evidence/planning/arc_107_selector_constitution_design_2026-06-20.md s3.1.
  PRECONDITION (build branch): V3-EXQ-689c (Factor-B-alone isolation retest) was PENDING /
  not-landed in the coordinator queue at build time (status pending, pinned ree-cloud-3,
  no results row), so no gap-CONCENTRATED parametric win existed to shrink scope -> PROCEED
  per the design note s5.2/s5.3 branch. The no-op-default lever commits nothing about scope;
  the scope-sensitive falsifier will incorporate 689c when it lands.
  PROBLEM (V3-EXQ-571): F (the primary harm/goal score) monopolises ~88-89%% of E3
  committed-selection variance, unmoved by the full diversity stack -- every diversity
  channel drowns at the F-dominated committed argmin. The required fix is constitutional:
  a signal's STRENGTH must be necessary but not sufficient; it needs lawful ACCESS to
  committed action. MECH-448 = F decides who is ELIGIBLE, not who wins.
  THE LEVER (no-op default; bit-identical OFF):
    Module: ree_core/predictors/e3_selector.py (new helper
    E3TrajectorySelector._f_eligibility_envelope + a "f_demotion" eligibility branch in the
    EXISTING shortlist-then-modulate block + 5 diagnostics), ree_core/utils/config.py
    (E3Config 3 fields + from_dims passthrough).
    Graded eligibility envelope (divisive-normalisation analog, lower-is-better F):
      merit[i] = clamp(raw_scores.max() - raw_scores[i], min=0)   # best=highest
      pooled   = f_eligibility_dn_sigma + merit.sum()
      elig[i]  = merit[i] / pooled                                # share of competing field
      eligible = { i : elig[i] >= f_eligibility_envelope_floor }  # ABSOLUTE share floor
    The absolute share floor is LOAD-BEARING: a fraction-of-max threshold cancels the pooled
    term and degenerates to the margin shortlist. With an absolute floor, a decisive F-winner
    commands most of the merit share so others fall below the floor (NARROW envelope), while a
    near-tie spreads the share (WIDE envelope) -- the BG hyperdirect conflict-grade emerging
    from the field structure, NOT a hard top-k count (which is env-conditional: 569i works only
    on the reef-bipartite guarantee; V3-EXQ-684 margin admits a near-whole state-stable set).
    elig is monotone in merit -> monotone in -F, so the eligible set is an F-RANK PREFIX
    (rank-preserving). The EXISTING _modulatory_accum within-eligible arbitration (reused, NOT
    duplicated -- ARC-106 guardrail 2) then picks the committed action (argmin committed /
    softmax uncommitted) with F REMOVED from the final argmin.
    Fallback: flat F (range ~0) or a genuine N-way tie whose per-candidate share is below the
    floor -> WIDE (all eligible) = correct low-conflict behaviour (and reported excluded_count
    == 0 = the non-degeneracy signal the falsifier checks against a divergent pool).
  Config (E3Config + REEConfig.from_dims, all no-op default; bit-identical OFF):
    use_f_eligibility_demotion (False, master), f_eligibility_envelope_floor (0.30; absolute
    DN-share floor), f_eligibility_dn_sigma (0.0; DN semi-saturation / global tightness).
    Requires a modulatory channel (_modulatory_accum not None, i.e. score_bias / MECH-341
    bonus / route bias) -- with no modulation there is nothing to demote F to, so the block is
    skipped (legacy F argmin, bit-identical); this is the falsifier's non-vacuity precondition.
  Diagnostics (last_score_diagnostics): f_eligibility_demotion_active, f_eligibility_envelope_size,
    f_eligibility_excluded_count (NON-DEGENERACY: >0 = the envelope actually excluded, not all-admit),
    f_eligibility_winner_neq_f_argmin (F demoted at commit), f_eligibility_rank_preserving (eligible
    set is an F-rank prefix; every eligible cost <= every excluded cost -- tie-robust).
  ARC-106 DIVERGENCE LEDGER (LOAD-BEARING): canonical divisive normalisation (Carandini & Heeger
    2012; value DN, Louie/Khaw/Glimcher 2013) is ORDER-PRESERVING + POOLED-SYMMETRIC. REE demotes
    ONLY F and removes it from the commit argmin -- rank-ALTERING at COMMIT -- which EXCEEDS
    canonical DN (the QD/MAP-Elites justification, CDQ-003). Must be lit-anchored (the concurrent
    targeted_review_connectome_mech_439 grounding extension) + falsifier-validated.
  SAFETY: a clearly-harmful candidate has near-zero merit -> near-zero share -> below floor ->
    excluded; no global disinhibition (the envelope is itself the F-bound). Contract-verified
    (an overwhelming modulatory pull toward an excluded harmful candidate never selects it).
  Psychiatric failure mode (ARC-106 mandate): envelope too wide / F removed without bounded No-Go
    -> disinhibition / impulsivity (mania, OCD-spectrum loss of inhibitory braking); envelope too
    tight -> bradykinesia / avolition (the current nothing-but-F-converts failure).
  Backward compatible: use_f_eligibility_demotion=False by default -> the f_demotion branch is
    never entered (the shortlist block guard is OR'd with the master flag; the legacy margin/top_k
    bodies are unchanged); bit-identical OFF. 10/10 new contracts in
    tests/contracts/test_mech_448_f_eligibility_demotion.py PASS; 8/8 preflight + 172/172
    E3-related contracts (e3/selector/mech_439/mech_341/score_bias/candidate_support/modulatory/
    arc065/dr12/mech090) PASS unchanged. Pure-envelope unit check: near-tie wider (2) than
    decisive (1), harmful outlier excluded (safety), exact 4-tie wide fallback (excluded 0).
    Full-agent activation smoke (MECH-341 modulatory channel, 20-tick loop): OFF 0/20 ticks
    demotion-active, ON 20/20 ticks demotion-active with winner!=F-argmin on every tick (F removed
    at commit); excluded=0 on the toy near-flat-F substrate (the divergent-pool exclusion the
    falsifier requires is contract-proven, not a smoke artifact).
  Phased training: N/A (pure-arithmetic selection rule; no learned parameters; no gradient flow).
    MECH-094: N/A (waking committed-selection path; no replay/memory write surface).
    Evidence-staleness (Step 8.5): NOT triggered -- no-op-default lever; every existing experiment
    uses the default (lever off), so no dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-448 stays candidate; ARC-107 / MECH-447 / MECH-449 /
    MECH-439 / Q-078 untouched. claims.yaml NOT modified (substrate-only build; co-claimed by a
    concurrent governance-689a-apply session at build time). substrate_queue f_dominance_conversion_ceiling
    build-rung amend DEFERRED -- live conflict with governance-689a-apply (re-checked at write time).
  Validation experiment: NOT queued here. The 689a-successor falsifier on the demotion lever
    (acceptance criteria = design note section 4; committed-class entropy reaches the proposer
    ceiling on >=2/3 seeds AND order preserved on the numerators AND no harmful class disinhibited;
    NON-DEGENERACY excluded_count>0 on a divergent pool) is the sequenced next /queue-experiment
    step, run on the GAP-A-ready foraging substrate (SD-056-trained e2.world_forward + ARC-065
    GAP-A candidate_summary_source=e2_world_forward = the divergent-pool non-vacuity precondition).
  Design doc: REE_assembly/docs/architecture/mech_448_f_eligibility_demotion.md
  Design note: REE_assembly/evidence/planning/arc_107_selector_constitution_design_2026-06-20.md
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-689a_2026-06-20.md
  See MECH-448 (this claim), ARC-107 (architecture), ARC-106 (grounding framework -- first worked
    application), modulatory-bias-selection-authority (the shortlist-then-modulate _modulatory_accum
    arbitration reused), MECH-439 (F-dominance root), MECH-447 (conflict-grade near-tie family;
    exhausted), MECH-449 (Go/No-Go constitution; follow-on, double-gated), Q-078 (umbrella),
    V3-EXQ-571 (F monopoly 88-89%%), V3-EXQ-689a (the autopsy that routed this), V3-EXQ-684 / 569i
    (margin-vs-top_k shortlist evidence), MECH-094 (N/A).

## MECH-448 AMEND: channel-adaptive (mean-relative) eligibility floor (collapse ~5 per-channel hand-floor dances into one knob) (2026-06-21)
- MECH-448 channel-adaptive envelope amend -- IMPLEMENTED 2026-06-21 (substrate;
  MECH-448 stays candidate -- PROMOTES NOTHING; the readiness EXQ + the 689a-successor
  falsifier are the sequenced next steps). Makes the rank-preserving F->eligibility
  demotion envelope CHANNEL-ADAPTIVE so it auto-calibrates per channel instead of needing
  a hand-tuned global floor. Routed by the confirmed-twice no-op signature: the absolute
  share floor (f_eligibility_envelope_floor=0.30) was tuned to PASS on the GAP-A foraging
  bank (V3-EXQ-689d), but each downstream channel has a different F-merit distribution, so
  the same fixed floor mis-fires -- V3-EXQ-654h (arc_062 rule-apprehension) admitted ALL
  candidates (f_eligibility_excluded_count==0, the lever never engaged; "485i twin",
  failure_autopsy_V3-EXQ-654h REE_assembly f41fe845fd); V3-EXQ-485i->485j (OFC) needed a
  bespoke per-seed envelope-floor recalibration to engage (485j then CONFIRMED OFC
  discrimination CONVERTS under demotion -- the lever generalises off GAP-A; residual was a
  separate devaluation test-design gap re-queued as 485k, NOT the envelope; 32c2518c1d).
  Direction confirmed (MECH-448 generalises); the only gap was the per-channel floor sweep.
  THE FIX (no-op default; bit-identical OFF):
    Module: ree_core/predictors/e3_selector.py (_f_eligibility_envelope branch),
      ree_core/utils/config.py (E3Config 2 fields + from_dims passthrough).
    use_f_eligibility_adaptive_floor (E3Config, default False): replaces the fixed
      absolute share floor with a MEAN-RELATIVE one --
      floor = f_eligibility_adaptive_mean_factor * elig.mean() -- so a candidate is
      eligible iff its share of the competing merit exceeds mean_factor times the field's
      OWN mean share, rather than an absolute constant. Mean-relative is SCALE-INVARIANT
      (auto-calibrates to each channel's F-merit distribution, no per-channel hand-tuning)
      AND retains the MECH-448 CONFLICT-GRADE (a decisive F-winner pulls the mean up so
      others fall below -> narrow envelope; a near-tie sits near the mean -> wide). It is
      still a threshold on elig (monotone in merit), so the eligible set stays an F-RANK
      PREFIX (rank-preserving). For mean_factor >= 1.0 on any NON-uniform field at least
      one candidate is below the mean share, so the envelope EXCLUDES (excluded_count > 0)
      by construction -- the 654h all-admit no-op cannot recur, and the 485i/485j bespoke
      floor recalibration is no longer needed. Collapses the ~5 per-channel hand-floor
      dances (654h/485i/485j + the pending 625/445/687 successors) into ONE global knob.
    f_eligibility_adaptive_mean_factor (E3Config, default 1.0): "above-average share"
      threshold multiple. Single GLOBAL knob (NOT per-channel), sweepable.
  Preserved: the existing exact-tie / flat-F early returns (merit_sum<=1e-8 -> wide) + the
    empty-eligible all-admit fallback are unchanged; the demotion still requires a
    modulatory channel (_modulatory_accum not None). Still emits the
    f_eligibility_excluded_count>0 non-degeneracy + winner_neq_f_argmin + envelope_size +
    rank_preserving diagnostics (e3_selector.py ~1391-1423).
  Backward compatible: use_f_eligibility_adaptive_floor=False by default -> the floor reads
    the legacy fixed config value -> bit-identical; AND use_f_eligibility_demotion itself
    stays default OFF so the whole block is skipped for every existing run (double guard).
    16/16 MECH-448 contracts (8 prior + 8 new amend: adaptive config defaults no-op /
    from_dims surfaces flags / adaptive-OFF bit-identical to fixed floor / EXCLUDES across
    2 differing-scale synthetic dists where the fixed 0.30 floor no-ops on the 654h-like D1
    and collapses on D2 / rank-preserving holds / conflict-grade preserved near-tie>decisive)
    + 8/8 preflight + 48/48 E3-cluster contracts (e3 score-bias / conflict-grade / DR-12 /
    ARC-065 GAP-A / stratified-temperature) PASS.
  Phased training: N/A (pure-arithmetic selection rule; no learned parameters; no gradient
    flow). MECH-094: N/A (waking committed-selection path; no replay/memory write surface).
    Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
    experiment uses the default (adaptive off + demotion off), so no dependent claim's
    measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-448 stays candidate; ARC-107 untouched; claims.yaml
    NOT modified. substrate_queue f_dominance_conversion_ceiling rung-2 implementation_log
    amended.
  Validation experiment: V3-EXQ-689e channel-adaptive envelope readiness diagnostic
    (claim_ids=[]) queued via /queue-experiment -- shows f_eligibility_excluded_count > 0
    lands in a productive range on >= 2 real channel substrates (the arc_062 rule-apprehension
    bank that no-opped in 654h + the OFC/foraging bank) with the SAME global adaptive config
    (no per-channel hand-tuning); bit-identical OFF as the negative control;
    substrate_not_ready_requeue if the adaptive floor still no-ops on any channel.
  Design doc: REE_assembly/docs/architecture/mech_448_f_eligibility_demotion.md
    (channel-adaptive amend section).
  See MECH-448 (parent; rank-preserving demotion lever landed 2026-06-20 above), ARC-107
    (architecture), MECH-439 (F-dominance root), V3-EXQ-654h (the arc_062 all-admit no-op
    this fixes), V3-EXQ-485i/485j (the OFC bespoke-recalibration this removes), V3-EXQ-689d
    (the GAP-A bank the fixed 0.30 floor was tuned to), V3-EXQ-689e (validation), MECH-094 (N/A).

## MECH-449 / ARC-107: Go/No-Go eligibility constitution (the OPPONENCY leg of the basal-ganglia E3-selector constitution; generalises MECH-260) (2026-06-21)
- MECH-449: selection.go_nogo_eligibility_constitution -- IMPLEMENTED 2026-06-21
  (substrate; MECH-449 stays candidate / substrate_conditional -- PROMOTES NOTHING;
  the ablation falsifier gates promotion, NOT the build). Component 1 (the core
  Go/No-Go opponency) of the ARC-107 BG constitution; the missing core complementing
  the landed MECH-448 (component 3, pallidal permission gate). Build gate cleared on
  BOTH faces (gate INVERTED 2026-06-21, anti-partial-instantiation): selection-face
  V3-EXQ-689f (No-Go-necessity falsifier: the live _f_eligibility_envelope admits
  high-F undesirable candidates only an active No-Go can suppress) + behavioural-face
  the 485e->485k six-autopsy lineage (demotion-alone converts the passive OFC
  discrimination signature but cannot express the active No-Go WITHDRAWAL devaluation
  requires). Second major worked application of ARC-106 (grounding synthesis 2.1:
  Kravitz 2010 D1/D2 opponency; Mink 1996 focal-go + surround-no-go; Maia & Frank 2011).
  PROBLEM: MECH-448's rank-preserving F->eligibility demotion is order-preserving over
  F, so it structurally CANNOT exclude an F-eligible-but-undesirable candidate on a
  non-F axis (safety/staleness/perseveration/low-viability) -- only an active No-Go can
  (exactly what 689f demonstrates).
  THE FIX (no-op default; bit-identical OFF): a BOUNDED Go/No-Go pressure SET housed as
  METHODS on E3TrajectorySelector (_go_nogo_eligibility_gate), the same pattern as
  _f_eligibility_envelope / _gap_scaled_commit_pick -- NO parallel module (ARC-106 G2).
  The gate runs inside the shortlist-then-modulate block AFTER the F-built eligible set
  (eligible_idx) is computed and BEFORE the within-eligible _modulatory_accum
  arbitration -- governing WHICH candidates may compete for the pallidal-like
  permission-to-commit gate:
    No-Go (suppress): drop a candidate from the eligible set when ANY bounded axis
      crosses its floor -- safety (>= gng_safety_floor), staleness (>= gng_staleness_floor),
      perseveration (recency-share >= gng_perseveration_floor), viability
      (< gng_viability_floor). ORTHOGONAL to F-rank; a No-Go'd candidate is removed
      from the eligible set so the within-eligible argmin can NEVER select it regardless
      of its modulatory pull (the SAFETY guarantee).
    Go (promote): re-admit (bounded by gng_go_max_promote) a candidate F demoted OUT of
      the envelope whose go-evidence >= gng_go_threshold (and not itself No-Go'd).
    FAIL-OPEN (gng_protect_min_eligible): No-Go never drops the eligible set below this
      many survivors UNLESS they are SAFETY-No-Go'd (safety never overridden) -- guards
      the No-Go-over-pressure -> catatonia/avolition pole from deadlocking the gate.
  REUSE-BEFORE-DUPLICATE (ARC-106 G2): the PERSEVERATION No-Go axis CONSUMES MECH-260's
  existing dACC anti-recency suppression vector (agent routes
  _dacc_last_bundle["suppression"] in as the perseveration signal) -- generalising
  MECH-260 from a drowned score-bias into an eligibility-access gate; NO duplicate
  recency buffer. The other axes are genuinely new functions MECH-260 lacks.
  Modules: ree_core/predictors/e3_selector.py (_go_nogo_eligibility_gate + invocation
  in the shortlist block + go_nogo_signals kwarg on select() + 6 diagnostics), and
  ree_core/utils/config.py (E3Config 8 fields + from_dims passthrough),
  ree_core/agent.py (build go_nogo_signals with the MECH-260 perseveration reuse +
  set_injected_go_nogo_signals() falsifier seam; passed ONLY when the constitution is
  engaged -- version-layering guard mirroring DR-12).
  Per-candidate signals via select(go_nogo_signals=...) (optional [K] tensors keyed
  safety/staleness/perseveration/viability/go; missing axis inert). Default loop wires
  only the MECH-260 perseveration reuse; the falsifier supplies the constructed-bank
  axes via REEAgent.set_injected_go_nogo_signals().
  Config (E3Config + from_dims, all no-op default -> bit-identical OFF):
  use_go_nogo_constitution (False) + gng_safety_floor (0.5) + gng_staleness_floor (0.5)
  + gng_perseveration_floor (0.5) + gng_viability_floor (0.1) + gng_go_threshold (0.5)
  + gng_go_max_promote (2) + gng_protect_min_eligible (1).
  PRECONDITION (same as MECH-448): the gate runs only when a modulatory channel is
  present (_modulatory_accum not None) -- with no modulatory arbitration there is
  nothing to govern, legacy F-argmin runs (gate inert). This is the non-vacuity
  precondition the falsifier enforces (SD-056-trained e2.world_forward + ARC-065 GAP-A
  candidate_summary_source=e2_world_forward divergent pool + a modulatory channel).
  Backward compatible: use_go_nogo_constitution=False by default -> the gate block is
  skipped, eligible_idx passes through unchanged -> bit-identical to the MECH-448
  selector. 6/6 new contracts in tests/contracts/test_mech_449_go_nogo_constitution.py
  (C1 bit-identical OFF even with signals passed / C2 No-Go suppresses within the
  eligible set / C3 SAFETY holds under overwhelming modulatory pull / C4 bounded Go
  re-admits a demoted candidate / C5 composes with the MECH-448 f_demotion envelope /
  C6 fail-open never empties the eligible set) + 8/8 preflight + 148 E3-cluster
  contracts (e3/selector/score-bias/mech_448/modulatory/mech_341/conflict-grade/dr12)
  PASS. Full-agent boot smoke: default == explicit-OFF action stream bit-identical;
  gate-ON path runs without error.
  Phased training: N/A (pure-arithmetic gate; no learned parameters; no gradient flow).
  MECH-094: waking committed-selection path only; no replay/memory write surface;
  call-site-scoped to select_action like the sibling levers.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (constitution off), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-449 stays candidate / substrate_conditional;
  ARC-107 / MECH-448 / MECH-260 / MECH-439 untouched. claims.yaml carries only an
  implementation_note.
  Validation experiment: V3-EXQ-689g (the MECH-449 ablation falsifier; claim_ids reflect
  MECH-449 + ARC-107) queued via /queue-experiment -- on the GAP-A-ready foraging
  substrate, the built Go/No-Go constitution must CONVERT >=1 previously-gated downstream
  channel beyond what MECH-448 achieves (over-specification if not). Pre-registered: a
  built-but-no-conversion result is non_contributory / does-not-promote, NOT an ARC-107
  falsification; self-route substrate_not_ready_requeue if the candidate pool is not
  divergent or Go/No-Go variables do not vary.
  Design doc: REE_assembly/docs/architecture/mech_449_go_nogo_constitution.md
  Design note: REE_assembly/evidence/planning/arc_107_selector_constitution_design_2026-06-20.md
  (s3.2 + s6b completeness ledger). Grounding: REE_assembly/evidence/literature/
  targeted_review_connectome_mech_439/ARC107_GROUNDING_SYNTHESIS.md (s2.1).
  See MECH-449 (this claim), ARC-107 (architecture umbrella), MECH-448 (the eligibility
  envelope this governs; pallidal gate), MECH-260 (dACC No-Go generalised; ree_core/
  cingulate/dacc.py), MECH-439 (F-dominance root), ARC-106 (grounding framework; second
  worked application), modulatory-bias-selection-authority (the shortlist-then-modulate
  arbitration the gate composes with), V3-EXQ-689f (No-Go-necessity falsifier; positive
  build trigger), V3-EXQ-689g (validation), Q-078 (umbrella), MECH-094 (N/A).

## Commit/release-DURATION lever: graded natural-commit-occupancy release (rung-6 of f_dominance_conversion_ceiling; PARALLEL to MECH-448) (2026-06-20)
- control_plane.natural_commit_occupancy_release -- IMPLEMENTED 2026-06-20
  (substrate; PROMOTES NOTHING. claims.yaml NOT modified; the 460i-successor
  de-commit falsifier is the SEQUENCED NEXT step, NOT queued here). The
  commit/release-DURATION face of the F-dominance front -- PARALLEL to, not an
  escalation of, the selection-face levers (MECH-439 conflict-grade; MECH-448
  rank-preserving F->eligibility demotion). NOT blocked on GAP-I / V3-EXQ-689c
  (a dead-end selection-face parametric retest); the duration face is its own
  lever per the 460h governance note (behavioral_diversity_isolation:GAP-I
  governance_2026_06_20).
  PROBLEM (V3-EXQ-460h): on strong (F-decisive) seeds the bistable beta latch
  elevates once and HOLDS ~2400-2600 steps because nothing releases it (decisive
  F-gap = good options -> MECH-342 maintenance-release silent; no closure ->
  SD-034 silent). That monolithic natural-commit occupancy SWAMPS the SD-034
  closure de-commit -> MECH-445 commit-intent (375 on weak seed 44) and MECH-446
  de-commit occupancy-drop (only where committed_steps weak) never co-occur on
  the same seed (the 460h disjoint-certifier problem). This lever makes the
  F-driven natural commit LESS MONOLITHIC so weak-natural-commit is the norm
  across seeds, dissolving the 460h problem.
  BG-3 SYNTHESIS divergence D1 (load-bearing): biology does NOT set commitment
  DURATION with a fixed refractory clock -- it times the hold with a GRADED
  BG/pallidal urgency (Thura/Cisek 2022) and/or makes maintenance co-extensive
  with the executing action (Jin 2014). REE's committed-run-scaled beta-gate
  refractory is the "tuned, not bio-sourced" divergence. THEREFORE the lever is
  a GRADED release, never another fixed refractory constant.
  Module: ree_core/policy/natural_commit_urgency.py (NaturalCommitUrgencyRelease
  + NaturalCommitUrgencyReleaseConfig). Pure-arithmetic regulator (no nn.Module,
  no learned params, no gradient flow); sibling to commit_maintenance_release.py
  (MECH-342). REUSES BetaGate.committed_run_length (the MECH-090 commit-gate
  machinery) -- NO parallel latch module (ARC-106 guardrail G2). Two D1-faithful
  release modes, both togglable under one master flag (the sequenced 460i-
  successor falsifier discriminates which lifts):
    (1) URGENCY (Thura/Cisek): per natural-commit tick,
        decisiveness_scale = 1 + gap_entry_sensitivity * gap_norm_at_entry;
        urgency += urgency_rate * decisiveness_scale; fire at >= release_bound.
        gap_norm_at_entry in [0,1] = normalised top-F decisiveness at entry
        (1 = decisive F-gap = the commit that monopolises). The gap-scaling is
        LOAD-BEARING: an F-decisive commit accrues urgency FASTER -> the
        strongest-F (most monopolising) holds are shortened most -> attacks
        F-dominance in the duration domain AND folds in the "gap-scaled
        commit-entry threshold" impl_hint candidate. gap_entry_sensitivity=0
        reduces to a flat fixed-rate timeout (the contrasted "fixed refractory"
        control the D1 falsifier compares against).
    (2) ACTION-EXTENT (Jin): release when the committed trajectory's executed
        action sequence completes (_committed_step_idx >= horizon) rather than
        repeating the last action indefinitely. Renders "maintenance
        co-extensive with the executing action" + the "natural-commit run-length
        cap" candidate as a BEHAVIOURALLY-grounded cap (the trajectory horizon),
        NOT a tuned constant. Fires regardless of urgency on sequence completion.
  DISTINCT from siblings: MECH-342 fires on DEGRADED readiness -> silent on the
  healthy-but-prolonged decisive commit that actually monopolises (why strong
  seeds hold ~2400 steps); SD-034 Leg-B refractory (MECH-446) holds the latch
  DOWN post-closure (this shortens the natural commit's occupancy UP, it does
  NOT install a refractory); MECH-091 is z_harm urgency (this is duration
  urgency, no harm input); ARC-028/MECH-105 releases on a HIGH completion signal
  (this releases on held-duration urgency / sequence completion regardless of
  plan quality).
  Config (REEConfig + from_dims, all no-op default -> bit-identical OFF):
  use_natural_commit_urgency_release (False, master) +
  natural_commit_release_urgency_mode (True) +
  natural_commit_release_action_extent_mode (True) + natural_commit_urgency_rate
  (0.01) + natural_commit_urgency_release_bound (1.0) + natural_commit_urgency_cap
  (1.5) + natural_commit_gap_entry_sensitivity (1.0, the load-bearing gap-scale;
  0.0 = flat control) + natural_commit_urgency_onset_ticks (0).
  Agent wiring (ree_core/agent.py): instantiate self.natural_commit_urgency when
  the master flag is on; ARM at the bistable elevate site on a fresh NATURAL
  commit (result.committed) via note_commit_entry(gap_norm) (gap_norm computed
  from result.scores; a purely closure-coupled elevation is NOT armed -> its
  occupancy stays governed by SD-034); TICK at the MECH-342 release region
  (committed_run_length=beta_gate.committed_run_length, action_sequence_complete
  =_committed_step_idx>=horizon) -> on fire release the latch (beta_gate.release()
  + _committed_step_idx=0 + _committed_anchor_keys=None + e3._committed_trajectory
  =None, mirroring the MECH-342 block); reset() per episode. NOT touched:
  e3_selector.py (clean separation from MECH-448), beta_gate.py (reuses the
  existing committed_run_length property), claims.yaml (PROMOTES NOTHING).
  Diagnostics (get_state): ncur_last_occupancy_at_release (latch-occupancy length
  at release) / ncur_n_urgency_releases + ncur_n_action_extent_releases +
  ncur_n_releases_total (release-event counts) / urgency + last_decisiveness_scale
  (graded-release magnitude) / gap_norm_at_entry / natural_commit_armed /
  ncur_n_simulation_skips.
  MECH-094: tick(simulation_mode=True) is a no-op (a replay/DMN tick must not
  abort a committed motor program); matches the SD-035/MECH-279/MECH-313/
  MECH-320/MECH-342 pattern. Phased training: N/A (pure-arithmetic regulator).
  Backward compatible: use_natural_commit_urgency_release=False by default ->
  agent.natural_commit_urgency is None; arm site + release block skipped ->
  bit-identical. 10/10 new contracts (tests/contracts/test_natural_commit_urgency.py:
  C1 config defaults + master-off None / C2 gap-scaled rate LOAD-BEARING
  [decisive releases sooner than near-tie; sensitivity=0 gap-independent] /
  C3 action-extent fires on sequence completion / C4 unarmed no-op /
  C5 MECH-094 sim no-op / C6 config validation / C7 agent release wiring +
  bounded occupancy [OFF holds ~12-tick run; ON releases, occupancy << OFF] /
  C8 agent bit-identical OFF [ON-inert == OFF action stream] / C9 agent arm-site
  [is_armed + gap_norm captured] / C10 release-only safety) + 7/7 preflight +
  full contract suite 1169 passed (the 2 test_runner_fail_branch failures are
  the documented pre-existing local-git-env runner-conflict flakes, CONFIRMED
  failing identically on a clean stash). Activation smoke = the C7/C8 agent
  contracts (occupancy bounded ON vs held OFF; ON-inert bit-identical).
  Validation experiment: NOT queued -- the 460i-successor de-commit falsifier
  (MECH-445/446 on this lever) is the SEQUENCED NEXT step.
  Design doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (f_dominance_conversion_ceiling rung 6 implementation_log).
  See SD-034 / MECH-090 / MECH-342 / MECH-445 / MECH-446 (the commit/release-
  duration cluster this serves; all candidate, unweakened), MECH-448 (selection-
  face sibling lever; PARALLEL), MECH-439 (F-dominance root), BG-3 SYNTHESIS
  (targeted_review_commit_release_duration_latch; divergence D1), ARC-106
  (grounding framework), V3-EXQ-460h (the FAIL this addresses), MECH-094 (N/A).

## Natural-commit LATCH-HOLD amend: establish the sustained-hold OFF baseline (V3-EXQ-460i gate amend) (2026-06-21)
- control_plane.natural_commit_latch_hold -- IMPLEMENTED 2026-06-21 (substrate;
  PROMOTES NOTHING; claims.yaml NOT modified; MECH-446/445 stay candidate /
  v3_pending / pending_retest_after_substrate). The HOLD that establishes a sustained
  natural-commit beta-latch occupancy for the rung-6 RELEASE
  (natural_commit_occupancy_release, landed 2026-06-20) to act on. Routed by the
  user-adjudicated failure_autopsy_V3-EXQ-460i_2026-06-21 (Option B "make the OFF
  baseline actually sustain"). Pairs with the V3-EXQ-460j gate-3 sustained-hold
  redesign (the experiment side).
  ROOT CAUSE (V3-EXQ-460i, confirmed): the rung-6 release self-routed
  substrate_not_ready_requeue at readiness gate 3 (lever_did_not_shorten_occupancy).
  The lever was correctly ARMED (lever_present=true on all 3 armed arms; _clone_arm
  set use_natural_commit_urgency_release + modes + gap_sensitivity) and its arm-site
  note_commit_entry was reached on NATURAL result.committed commits, but it fired
  ZERO releases because the 460h sustained ~2400-step monolithic natural-commit hold
  DID NOT REPRODUCE -- the active SD-034 de-commit control-plane (closure->beta
  coupling re-toggle + the Leg-B committed-run-scaled refractory + closure releases)
  fragmented the beta latch to ~1-tick blips EVEN WITH THE RELEASE OFF (ARM_LEVER_OFF
  total_beta_elevated ~= beta_release_events, 415/405 seed 43; ~35 re-commits/episode),
  so there was no sustained occupancy to shorten and the urgency accumulator (reset
  per fresh entry, ~0.01-0.02/tick) could not reach release_bound over ~1 tick.
  Readiness gate 3's mean_beta_elevated_steps proxy is BLIND to sustained-vs-fragmented
  and green-lit the fragmented regime. The release lever is sound; the REGIME was missing.
  THE FIX (no-op default; bit-identical OFF): a latch-HOLD SEPARATE from (and
  independent of) the release lever -- so it arms in the ARM_LEVER_OFF baseline too
  (where natural_commit_urgency is None). A fresh NATURAL commit (result.committed)
  ARMS the hold; while armed AND the committed trajectory persists, the beta latch is
  RE-ASSERTED each tick (kept elevated against the de-commit churn) so the
  natural-commit occupancy sustains BY CONSTRUCTION -- the sustained reference the
  rung-6 release shortens and the gate-3 sustained-hold proxy certifies. With the hold
  keeping beta elevated, note_commit_entry fires ONCE (the bistable elevate block is
  skipped while already elevated), so the rung-6 urgency accumulates monotonically over
  the held duration and FIRES -- exactly what 460i lacked. The hold YIELDS to (disarms
  on) the three PRINCIPLED releases so it never papers over them: (a) an SD-034 closure
  de-commit (beta_gate.refractory_remaining > 0 -> the latch is held DOWN by the
  closure; the hold does not fight it, preserving the MECH-446 within-arm occupancy-drop
  DV), (b) the MECH-091 genuine-threat urgency interrupt (safety -- NEVER overridden),
  (c) the rung-6 NaturalCommitUrgencyRelease's own duration release (the lever
  shortening the held commit -- the whole point; the hold disarms so the occupancy
  stays shortened). Also disarms when the committed trajectory ends or the optional
  max-ticks safety cap is reached.
  Modules: ree_core/utils/config.py (REEConfig.use_natural_commit_latch_hold [False,
  master, INDEPENDENT of use_natural_commit_urgency_release] + natural_commit_latch_hold_max_ticks
  [0 = unbounded safety cap] + from_dims passthrough); ree_core/agent.py (hold state
  _ncl_hold_active/_ncl_hold_ticks/_ncl_hold_reassert_count + per-tick principled-release
  flags _ncl_mech091_fired/_ncl_lever_fired in __init__ + reset; ARM at the bistable
  natural-commit elevate site on result.committed; per-tick RE-ASSERTION after all
  release sites + before the between-tick branch; flag-set at the MECH-091 + rung-6
  release sites). natural_commit_urgency.py UNCHANGED (the hold is agent-level, since it
  must work without the release lever instantiated). Data flow: arm-site -> _ncl_hold_active
  True; end-of-tick re-assertion -> if armed AND committed trajectory persists AND
  refractory_remaining==0 AND no MECH-091/rung-6 release this tick AND under max-ticks ->
  beta_gate.elevate() (keep elevated); else disarm.
  Backward compatible: use_natural_commit_latch_hold=False by default -> _ncl_hold_active
  stays False, no arm, no re-assert; the per-tick flags are no-op bool writes ->
  bit-identical OFF. 7/7 new contracts (tests/contracts/test_natural_commit_latch_hold.py:
  C1 defaults + master-off no-op / C2 arm-site / C3 re-assert-against-churn LOAD-BEARING
  [hold ON sustains where hold OFF drops a non-re-committing latch] / C4 yield to closure
  refractory / C5 yield when the commit ends / C6 max-ticks cap / C7 bit-identical OFF) +
  8/8 preflight + full contract suite 1176 passed + 39 subtests (the 2
  test_runner_fail_branch failures are the documented pre-existing local-git-env runner
  flakes, zero overlap with this change). Activation = the C3 contract (the OFF latch
  drops under churn-without-recommit; the ON hold re-asserts it sustained).
  Phased training: N/A (control-state wiring + bool flags; no learned parameters).
  MECH-094: the hold re-assertion runs on the waking select_action path; it never
  overrides the MECH-091 safety interrupt and writes no memory content.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (hold off), so no dependent claim's measured mechanism
  changed. KEEP all evidence.
  Validation experiment: V3-EXQ-460j (supersedes V3-EXQ-460i; queued via /queue-experiment)
  -- arms the hold in ALL arms + redesigns readiness gate 3 to a SUSTAINED-HOLD proxy
  (longest consecutive beta-elevated run + mean per-commit hold length
  total_beta_elevated/max(1,beta_release_events)) above a floor on >=2/3 OFF-arm guard
  seeds AND ncur_n_releases_total>0 with a >= LEVER_OCC_DROP_FRAC occupancy drop vs OFF on
  ARM_GAP_SCALED, BEFORE the CO_OCCURRENCE DV. Do NOT re-author 460d/e/f/g/h/i.
  MECH-446/445 stay candidate/v3_pending/pending_retest_after_substrate until 460j scores
  a contributory result.
  Design doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md
  (natural-commit LATCH-HOLD amend section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460i_2026-06-21.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (f_dominance_conversion_ceiling rung-6 implementation_log).
  See natural_commit_occupancy_release (the rung-6 RELEASE this HOLD establishes the
  baseline for), SD-034 / MECH-090 / MECH-342 / MECH-445 / MECH-446 (the commitment-
  closure-control-plane cluster whose de-commit machinery the hold yields to; all
  candidate, unweakened), MECH-091 (genuine-threat interrupt the hold never overrides),
  MECH-439 / MECH-448 (F-dominance front), V3-EXQ-460i (the FAIL this addresses),
  V3-EXQ-460j (validation), MECH-094 (N/A).

## Closure-exclusive de-commit eval mode (rung-6 BUILD of f_dominance_conversion_ceiling; the named dissociable substrate from V3-EXQ-460j) (2026-06-22)
- control_plane.closure_exclusive_decommit_eval -- IMPLEMENTED 2026-06-22 (substrate;
  PROMOTES NOTHING; claims.yaml NOT modified; MECH-445/446 stay candidate / standard /
  v3_pending / pending_retest_after_substrate). The named awaited substrate the rung-6
  natural-commit-occupancy-release lever was PARKED on 2026-06-22 (failure_autopsy_V3-EXQ-460j,
  user-adjudicated "Park + amend, name the substrate"). Routed via /implement-substrate
  commitment_closure:GAP-4; the mandatory pre-build rethink re-confirmed buildable
  (user-confirmed the closure-exclusive design fork: suppress the natural path).
  ROOT CAUSE (re-confirmed in code): the natural-commit latch-hold (2026-06-21 amend)
  arms ONLY on a decisive natural commit (result.committed), which does not form on the
  full closure-coupling substrate (460j ARM_LEVER_OFF 3/3: ncl_hold_reassert_total=0,
  max_consecutive_beta_run=1, sd034_n_closure_commit_intent=0). The SD-034 closure
  de-commit control-plane fragments the latch to ~1-tick blips even with the rung-6
  release OFF, so natural-commit and closure-de-commit are NON-DISSOCIABLE: no sustained
  natural-commit occupancy exists for the de-commit to act on, and no fair test of
  MECH-445 (commit-intent) / MECH-446 (occupancy-drop) is reachable. A plain yield-clause
  patch (a "460k") was REFUSED -- it targets the release/yield logic, the WRONG cause; the
  actual cause is the ARM SOURCE of the occupancy.
  THE FIX (no-op default; bit-identical OFF): a closure-exclusive de-commit eval mode that
  re-points the occupancy's arm source onto the closure plane.
    Module: ree_core/agent.py (bistable elevate block + __init__ precondition + a
      _ncl_hold_closure_armed_count readout), ree_core/utils/config.py (1 no-op-default
      flag + from_dims). NO new module; reuses the existing latch-hold + re-assertion +
      yield + BetaGate.committed_run_length (ARC-106 G2 reuse-before-duplicate).
    (a) Closure-EXCLUSIVE elevation: when closure_exclusive_decommit_eval is set,
      _commit_for_beta is driven ONLY by _closure_commit_active (the closure->beta
      coupling) -- the fragile F-driven result.committed path is SUPPRESSED from beta
      elevation. So the beta occupancy is provably closure-formed (the literal "beta
      elevates ONLY via _closure_commit_active during the eval").
    (b) Closure-coupled hold-arm: the natural-commit latch-hold ARMS on
      _closure_commit_active (a closure-plane commitment forming) in addition to
      result.committed, GUARDED by beta_gate.refractory_remaining == 0 so it does NOT
      re-arm while an SD-034 closure de-commit is actively holding beta down (the hold
      yields to the de-commit, preserving the MECH-446 occupancy-drop DV). The closure
      commitment persists across ticks (unlike the fragile result.committed), so the
      hold's existing re-assertion sustains the occupancy reliably on all seeds.
  DISSOCIATION (the whole point): a closure commitment forms -> the hold arms + sustains a
  beta occupancy -> the SD-034 closure FIRES -> release() + refractory -> the hold yields
  -> the occupancy drops -> the refractory expires -> the next closure commit re-arms.
  Occupancy FORMATION (closure-coupled latch-hold) is now dissociable from closure
  DE-COMMIT (the SD-034 refractory), making MECH-445 commit-intent + MECH-446
  occupancy-drop co-measurable on the same seed -- dissolving the 460h disjoint-certifier
  problem. The fragile, seed-dependent F-driven natural commit is removed from the
  occupancy entirely.
  PRECONDITIONS (loud ValueError at REEAgent.__init__): closure_exclusive_decommit_eval=True
  requires use_closure_commit_beta_coupling=True AND use_natural_commit_latch_hold=True
  (the eval has no path to form / sustain the occupancy otherwise) -- the MECH-269b/293
  precondition pattern.
  Backward compatible: closure_exclusive_decommit_eval=False by default ->
  _commit_for_beta is the legacy result.committed OR _closure_commit_active, the hold arms
  only on result.committed -> bit-identical (verified: default == explicit-OFF action
  stream). 7/7 new contracts in tests/contracts/test_closure_exclusive_decommit_eval.py
  (C1 config defaults / C2 preconditions raise / C3 closure-coupled commit arms the hold
  under eval [LOAD-BEARING] + legacy does not / C4 natural-commit suppressed under eval /
  C5 yield to the closure refractory preserved / C6 bit-identical OFF) + 8/8 preflight +
  31/31 sibling closure/latch/beta-gate contracts PASS. V3-EXQ-460j --dry-run reproduces
  its eval-OFF baseline (off_occ~4, reassert=0, sustained_hold=False); activation: eval ON
  arms+re-asserts (closure_armed=1, reassert=8) where eval OFF stays 0 (the 460j signature).
  Phased training: N/A (control-state wiring; no learned parameters). MECH-094: N/A (waking
  select_action control-state transition; no replay/memory write surface). Evidence-staleness
  (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment uses the default
  (eval off), so no dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-445/446 stay candidate / standard / v3_pending /
  pending_retest_after_substrate; SD-034 / MECH-090 / MECH-342 / MECH-448 / MECH-439
  untouched. claims.yaml NOT modified (substrate-only build).
  Validation experiment: NOT queued here -- a 460-lineage successor (NEW letter, supersedes
  V3-EXQ-460j; queued SEPARATELY via /queue-experiment AFTER this build lands) runs in the
  closure-exclusive de-commit eval mode and gates on (a) the ARM_LEVER_OFF baseline
  sustains a natural-commit occupancy on >=2/3 seeds, THEN (b) the rung-6 release shortens
  it AND MECH-445 commit-intent + MECH-446 occupancy-drop co-occur on the same seeds.
  Design doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md
  (closure-exclusive de-commit eval mode section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460j_2026-06-21.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (f_dominance_conversion_ceiling rung-6 implementation_log + status PARKED -> BUILT).
  See natural_commit_occupancy_release (the rung-6 RELEASE this eval mode supports) +
  the natural-commit LATCH-HOLD amend above (the hold whose arm source this re-points),
  SD-034 / MECH-090 / MECH-445 / MECH-446 (the commitment-closure-control-plane cluster
  this dissociates; all candidate, unweakened), MECH-091 (genuine-threat interrupt the
  hold never overrides), MECH-439 / MECH-448 (F-dominance front; the duration face is
  PARALLEL), V3-EXQ-460j (the FAIL this addresses), MECH-094 (N/A).

## F-independent closure-plane commit-ENTRY primitive (rung-6 amend; the F-INDEPENDENT arm source the closure-exclusive de-commit eval lacked; closes the 460k/460l ncl_hold_closure_armed_total=0 signature) (2026-06-23)
- control_plane.closure_commit_entry -- IMPLEMENTED 2026-06-23 (substrate;
  PROMOTES NOTHING; claims.yaml carries an implementation_note at most; MECH-445/446 stay
  candidate / standard / v3_pending / pending_retest_after_substrate). The F-INDEPENDENT
  commit-ENTRY the closure plane lacked. Routed by the confirmed
  failure_autopsy_V3-EXQ-460k_2026-06-22 + failure_autopsy_V3-EXQ-460l_2026-06-23 via
  /implement-substrate (commitment_closure:GAP-4; rung-6 of f_dominance_conversion_ceiling).
  ROOT CAUSE (code-confirmed; the autopsies already name the circularity): the 2026-06-22
  closure-exclusive de-commit eval arms the latch-hold ONLY via _closure_commit_active
  (agent.py:6365), which gates on e3._committed_trajectory is not None -- whose ONLY
  non-None writer in all of ree_core/ is e3_selector.py:1926 under `if committed:`
  (committed = self._running_variance < effective_threshold, pure variance/F). On the 460j
  substrate the F-driven natural commit never sustains (off_baseline_not_sustained, 0/3
  seeds), so _committed_trajectory is rarely non-None -> the eval rarely elevates -> the
  latch-hold rarely arms = the 460k/460l signature closure_exclusive_eval_did_not_arm_hold /
  ncl_hold_closure_armed_total=0. The design semantics ("a closure-plane commitment forming,
  F-independent") and the implementation (F-commit trajectory presence) CONTRADICT. The
  closure plane today has only a completion/RELEASE event (ClosureOperator.tick / emit_closure
  / habenula_tick); it had NO commit-ENTRY event -- a 460k-successor on the old code re-fails
  identically.
  THE FIX (Option A, user-confirmed via AskUserQuestion 2026-06-23; no-op-default;
  bit-identical OFF):
    (1) New F-INDEPENDENT latch e3._closure_committed_active (bool, init False;
      ree_core/predictors/e3_selector.py __init__). Unlike _committed_trajectory (set ONLY
      under `if committed:` and torn down every tick by post_action_update), this latch is
      STICKY across ticks until a principled closure teardown.
    (2) SET site (ree_core/agent.py REEAgent.select_action, right after e3.select returns,
      BEFORE the bistable _closure_commit_active computation), guarded by
      getattr(config, "use_closure_commit_entry", False): SET when a goal-active,
      rule-directed commitment forms -- goal_state.is_active() AND a trajectory was selected
      toward it (the hippocampal proposer is goal-seeded under an active goal) AND a rule is
      being followed (lateral_pfc.rule_state.norm() >= closure_commit_entry_rule_norm_floor,
      mirroring the SD-034 ClosureOperator's rule-stability precursor). Faithful to SD-034:
      closure governs rule-directed commitments. MECH-094: select_action is the waking path
      (simulation_mode=False, as the neighbouring tonic_vigor/closure-coupling sites assume);
      no replay/memory write surface (a waking control-state transition).
    (3) Redefinition (agent.py:6365): _closure_commit_active = use_closure_commit_beta_coupling
      AND (e3._committed_trajectory is not None OR e3._closure_committed_active) -- the UNION.
      The legacy F-commit path is preserved exactly (latch never set when the flag is off ->
      _closure_commit_active reduces to `_committed_trajectory is not None` -> bit-identical),
      and the eval can now arm WITHOUT a sustained F-commit.
    (4) CLEAR sites (ree_core/agent.py): the four existing `_committed_trajectory = None`
      de-commit sites (MECH-342 maintenance release / NaturalCommitUrgencyRelease duration
      release / MECH-353 blocked-agency abort / habenula de-commit), the SD-034 closure fire
      (ClosureOperator.tick auto-detector + the notify_env_completion emit_closure hook, when
      _fire installs the de-commit refractory + releases beta), and agent.reset() (episode
      boundary).
    (5) LATCH-HOLD persistence UNION (the one necessary yield-clause extension): the rung-6
      latch-hold re-assertion (agent.py) yields when e3._committed_trajectory is None. Since
      the F-independent latch deliberately leaves _committed_trajectory None (the closure
      plane installs NO trajectory -- the user's explicit NOTE), the persistence sub-condition
      is made UNION-aware (_ncl_commit_present = _committed_trajectory is not None OR
      _closure_committed_active) so the closure-formed occupancy sustains instead of arming
      then immediately yielding. Bit-identical OFF (latch never set -> reduces to the legacy
      `_committed_trajectory is None`). The other principled yields (closure refractory /
      MECH-091 / rung-6 lever / max-ticks) are UNCHANGED, so the SD-034 de-commit still tears
      the hold down (the MECH-446 occupancy-drop DV intact). LEFT UNCHANGED: the elevate
      conjunction, the closure-arm, the rest of the yield logic.
  Config (REEConfig + from_dims, all no-op default -> bit-identical OFF):
  use_closure_commit_entry (False, master) + closure_commit_entry_rule_norm_floor (0.01,
  the "rule is being followed" floor). PRECONDITIONS (loud ValueError at REEAgent.__init__,
  the MECH-269b/SD-058 pattern): use_closure_commit_entry=True requires
  use_closure_commit_beta_coupling=True (the coupling is the latch's consumer) AND
  use_natural_commit_latch_hold=True (the hold sustains the closure-formed occupancy).
  Backward compatible: use_closure_commit_entry=False by default -> e3._closure_committed_active
  is never set -> _closure_commit_active reduces to the legacy `_committed_trajectory is not
  None` and the persistence union reduces to the legacy `_committed_trajectory is None` ->
  bit-identical for every existing run. preflight 8/8 + the closure/latch/beta-gate/decommit
  cluster (58, incl 6 new) + the full contract suite PASS. New contracts:
  tests/contracts/test_closure_commit_entry.py (C1 config defaults + latch attr init False +
  default agent never arms / C2 preconditions raise / C3 C-KEY LOAD-BEARING: entry ON + eval
  ON + committed never True [_committed_trajectory stays None] + goal-active rule-directed ->
  the latch arms [ncl_hold_closure_armed_count > 0] AND the hold SUSTAINS beta [multi-tick
  committed_run_length + hold stays armed], while the SAME scenario with entry OFF == the
  pre-fix legacy eval arms EXACTLY 0 and beta never sustains / C4 default-OFF bit-identical
  action stream / C5 episode-reset clears the latch).
  Phased training: N/A (control-state wiring + bool flags; no learned parameters). MECH-094:
  the SET is a waking control-state transition (select_action waking-only by call-site
  scoping); no replay/memory write surface (the latch is not memory content). Evidence-
  staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment uses
  the default (entry off), so no dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-445/446 stay candidate / standard / v3_pending /
  pending_retest_after_substrate; SD-034 / MECH-090 / MECH-342 / MECH-439 / MECH-448 untouched.
  claims.yaml carries an implementation_note at most (no flag/confidence/status change).
  Validation experiment: V3-EXQ-460m claim-free substrate-readiness diagnostic (queued via
  /queue-experiment) -- on the closure-exclusive de-commit eval substrate, gates in order
  (a) ARM_LEVER_OFF baseline SUSTAINS a closure-formed occupancy on >=2/3 seeds with ZERO
  F-commits (the precondition every prior 460 run failed); (b) the rung-6 release shortens it;
  (c) MECH-445 commit-intent + MECH-446 occupancy-drop co-occur on the same seeds. A 460-lineage
  successor (NEW letter, supersedes the parked 460k/460l line) runs the full de-commit falsifier
  once (a) clears. Self-routes substrate_not_ready_requeue if (a) still fails. Do NOT re-author
  460d/e/f/g/h/i/j/k/l.
  Design doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md
  (closure-plane commit-ENTRY primitive section). Autopsies:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460k_2026-06-22.{md,json} +
  failure_autopsy_V3-EXQ-460l_2026-06-23.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (f_dominance_conversion_ceiling rung-6 implementation_log).
  See the closure-exclusive de-commit eval mode + natural-commit LATCH-HOLD amend +
  natural_commit_occupancy_release (the rung-6 cluster this completes), SD-034 / MECH-090 /
  MECH-445 / MECH-446 (the commitment-closure-control-plane cluster; all candidate, unweakened),
  SD-033a lateral_pfc rule_state (the "rule is being followed" source), MECH-091 (genuine-threat
  interrupt the hold never overrides), MECH-439 / MECH-448 (F-dominance front; this is the
  duration/de-commit face), V3-EXQ-460k/460l (the FAILs this addresses), V3-EXQ-460m (validation),
  MECH-094 (N/A).

## ARC-108 JOB-1 step-1: learned dopamine-gated E3 selection (signed-RPE w_chan over the modulatory channels; the next MECH-439 attack, learned-not-arithmetic) (2026-06-22)
- ARC-108 (JOB 1 step 1): selection.unified_dopamine_substrate_learned_gating --
  IMPLEMENTED 2026-06-22 (substrate; ARC-108 stays candidate / substrate_conditional /
  implementation_phase=v3 -- this PROMOTES NOTHING. The sec-7 selection falsifier + the
  MECH-450 settling step + the JOB-2 control-plane pair are SEPARATE downstream chips,
  NOT queued here). THE FIRST learned object in the ARC-107 arbitration layer (which was
  pure arithmetic with no learned parameters -- the A.4 learning gap and the predicted
  root of F-dominance, MECH-439). User-ratified V3 pull-forward (claims.yaml master
  045c4b73df). Design-of-record: dopamine_into_gating_design_2026-06-22.md secs 2-4 +
  unified_dopamine_substrate_design_2026-06-22.md sec 1/10. Second worked instance of
  ARC-106 (the learning rule is grounded at FUNCTION -- three-factor Hebbian x signed
  RPE with D1-LTP/D2-LTD asymmetry -- and EARNS the dopaminergic psychiatric axis).
  WHAT LANDS (all in ree_core/predictors/e3_selector.py -- extend the selector, NO
  parallel module, ARC-106 G2):
    (1) A single LEARNED per-channel selection-weight vector w_chan over the modulatory
      channels that feed the select() _modulatory_accum composition site. At that site
      the genuinely-separable constituents are the three add-terms already tracked there
      -- score_bias (the composed dACC+lPFC+OFC+MECH-295+MECH-314+MECH-320 chain, summed
      UPSTREAM in agent.py), the MECH-341 entropy bonus, and the route bias -- so the
      minimal channel registry is _LCG_CHANNEL_NAMES=("score_bias","mech341","route")
      (C=3), indexed by name (a channel absent on a tick simply does not contribute, so
      w_chan stays a stable learned object). A finer per-head channel split (dACC vs
      lPFC vs ...) is a documented follow-on -- those are summed upstream before reaching
      select(), so splitting them needs either un-summed biases threaded into select() or
      w_chan moved upstream; out of step-1 scope per "compose at the _modulatory_accum
      site / extend e3_selector".
      Composition: _modulatory_accum = sum_c softplus(w_chan[c]) * channel_bias_c (was
      the unweighted sum). w_chan is a register_buffer (NOT nn.Parameter -- the
      three-factor plasticity is a LOCAL update, never touched by an optimizer/autograd),
      init w_chan[c]=ln(e-1) so softplus(w_chan[c])==1.0 EXACTLY in float32 -> the
      recompose reproduces the unweighted accumulator bit-for-bit (1.0*x==x, matching add
      order). Only _modulatory_accum is re-weighted -- raw scores / F (the MECH-448
      envelope + the commit decision) are UNTOUCHED, so learning composes strictly INSIDE
      the F-bounded MECH-448/449 eligible set the within-eligible shortlist-argmin + the
      authority rescale consume, and a learned weight can NEVER re-admit a No-Go-suppressed
      candidate (safety inherited from the envelope; contract C6).
    (2) Signed RPE delta_t = R_t - V-hat_t, formed in post_action_update(actual_z_world)
      (the waking post-step hook). R_t = (benefit_eval_head - harm_eval_head)(actual_z_world)
      -- realised outcome valence from the ALREADY-TRAINED valuation heads (reuse, detached;
      NO new encoder, NO phased training). V-hat_t = a slow EMA leaky-integrator baseline
      (scalar). delta_t is SIGNED by construction and is explicitly NOT the unsigned ARC-016
      prediction-error VARIANCE (e3._running_variance) -- divergence B5; the two are kept
      separate (an unsigned magnitude cannot supply the directional Go-up/No-Go-down credit
      a learned gate needs). Contract C5 makes the signedness load-bearing (a positive
      delta_t POTENTIATES the voting channel; a negative delta_t DEPRESSES it -- opposite-sign
      w_chan changes an unsigned substitution could not produce).
    (3) Three-factor update Delta w_chan[c] = eta * delta_t * eligibility_c * asym(delta_t),
      applied in-place under no_grad in post_action_update. eligibility_c =
      |channel_bias_c[selected]| recorded after selected_idx in select() into a decayed
      last-K-ticks Hebbian co-activation trace. asym renders the D1-LTP/D2-LTD asymmetry as
      a single asymmetric gain (potentiation on delta_t>=0 faster than depression on
      delta_t<0) -- the V3 single-vector rendering; the D1/D2 opponent-population split is
      ARC-109 (deferred V4).
    (4) MECH-450 recurrent-settling step (learned W_lat) is the SECOND factor and is OFF in
      this build (W_lat==0): the integration point is reserved (a marked comment at the
      _modulatory_accum recompose) but NOT enabled -- it is the next chip.
  Waking-only gate (MECH-094 / divergence B5 contract): select() gains a keyword-only
  simulation_mode (default False, backward-compatible); eligibility is recorded ONLY when
  not simulation_mode, and post_action_update updates w_chan ONLY when a fresh waking
  eligibility trace is pending. A replay/DMN tick forms no delta_t and writes no w_chan
  (contract C4). Per-episode: agent.reset() calls e3.clear_learned_channel_eligibility()
  to clear the within-episode credit window (the trace + pending flag); w_chan and V-hat_t
  PERSIST across episodes as the learned state.
  Config (E3Config + REEConfig.from_dims, all no-op default -> bit-identical OFF):
  use_learned_channel_gating (False, master) + learned_channel_gating_eta (0.01) +
  learned_channel_gating_elig_decay (0.9) + learned_channel_value_baseline_beta (0.05) +
  learned_channel_asym_potentiation (1.0) + learned_channel_asym_depression (0.5).
  Backward compatible: use_learned_channel_gating=False by default -> the recompose +
  eligibility-recording + post-action update blocks are skipped -> bit-identical. 9/9 new
  contracts in tests/contracts/test_arc108_learned_channel_gating.py (C1 config defaults +
  from_dims + softplus-unity init / C2 OFF == ON-at-init EXACT scores+selection + OFF writes
  no w_chan / C3 w_chan MOVES under a non-flat delta_t when ON, stays at init when OFF /
  C4 simulation tick writes no w_chan / C5 signed-RPE potentiate-vs-depress load-bearing /
  C6 envelope intact -- learned weight cannot re-admit an F-excluded candidate) + 8/8
  preflight + 168 e3-cluster contracts PASS. Agent-level activation smoke (real env, MECH-341
  channel ON): w_chan[mech341] moved 0.54132->0.54632 over 13 updates with non-flat delta_t;
  OFF stayed exactly at init; first action ON==OFF bit-identical at init.
  Phased training: N/A -- the three-factor rule is a local (non-backprop) update; R_t reuses
  the already-trained valuation heads. MECH-094: waking-only by the simulation_mode gate +
  the pending-eligibility coupling. Evidence-staleness (Step 8.5): NOT triggered -- no-op-default
  flag; every existing experiment uses the default (gating off), so no dependent claim's
  measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. ARC-108 stays candidate / substrate_conditional /
  implementation_phase=v3; MECH-450 / ARC-107 / MECH-448 / MECH-449 / MECH-439 / ARC-016
  untouched. claims.yaml carries only an implementation_note on ARC-108 + MECH-450.
  Validation experiment: NOT queued here -- the sec-7 selection 2x2 falsifier (learned-w_chan
  x learned-W_lat on the GAP-A divergent pool: committed-class entropy strict-above the
  envelope-only arm AND a matched-noise control, GROWING with training, with the
  signed-vs-unsigned-RPE ablation arm) is a SEPARATE /queue-experiment chip, sequenced after
  the MECH-450 settling step lands (factor 2 of the 2x2).
  Design doc: REE_assembly/docs/architecture/dopamine_into_gating.md. Design-of-record:
  REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md (secs 2-4) +
  unified_dopamine_substrate_design_2026-06-22.md (sec 1/10).
  See ARC-108 (this claim; JOB-1 step-1 slice), ARC-107 (the BG-constitution umbrella whose
  arithmetic envelope this adds the missing LEARNING afferent to), MECH-448 / MECH-449 (the
  F-bounded eligible set learning composes INSIDE; safety inherited), MECH-450 (the coupled
  settling step -- factor 2, OFF here), MECH-439 (the F-dominance conversion ceiling this
  attacks by learned re-weighting), ARC-016 (the unsigned variance kept SEPARATE -- divergence
  B5), ARC-106 (grounding framework; second worked application), ARC-109 (the D1/D2 population
  split -- deferred V4), MECH-094 (waking-only call-site scoping).

## MECH-450 (ARC-108 JOB-1 step-2): learned recurrent-settling step + learned lateral-inhibition W_lat (factor 2 of the learned-gating 2x2; B1 + B3-blend repair) (2026-06-22)
- MECH-450 (ARC-108 JOB-1 step-2): selection.minimal_recurrent_settling_step --
  IMPLEMENTED 2026-06-22 (substrate; MECH-450 stays candidate / substrate_conditional /
  implementation_phase=v3 -- this PROMOTES NOTHING. The sec-7 selection 2x2 falsifier is a
  separate /queue-experiment chip). The SECOND factor of the learned-gating 2x2 (JOB-1
  step-1 w_chan landed earlier 2026-06-22, ree-v3 ae907b5), coupled to it and sharing the
  SAME signed-RPE delta_t / V-hat_t / D1-D2 asym. Design-of-record:
  REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md sec 4 +
  REE_assembly/docs/architecture/dopamine_into_gating.md.
  WHAT LANDS (all in ree-v3/ree_core/predictors/e3_selector.py -- extend the selector, NO
  parallel module, ARC-106 G2):
    (1) A bounded recurrent LATERAL-INHIBITION SETTLING step over the F-bounded eligible set,
      run at the within-eligible arbitration site (on mod_eligible =
      _modulatory_accum[eligible_idx]) BEFORE the commit, in new helper
      E3TrajectorySelector._lateral_settle:
        accum = mod_eligible
        for r in range(R):                       # R = learned_settling_rounds, default 3
            a       = softmax(-accum / T)         # support over eligible (low cost -> high)
            a_class = onehot.T @ a                # [C] per-action-class aggregated support
            accum   = accum + onehot @ (W_lat @ a_class)   # learned lateral inhibition
        commit = argmin(settled accum) (committed) / sample(softmax(-settled accum/T)) (uncommitted)
      Fixes divergence B1 (one-shot argmin -> recurrent settling) AND B3-blend (additive
      _modulatory_accum blend -> competitive winner-take-most) together. The settling
      transforms ONLY the eligible subset and the commit reads the SETTLED accum (argmin /
      gap-scaled / softmax-sample all read it).
    (2) W_lat = the LEARNED lateral-inhibition matrix over candidate first-action CLASSES
      (a stable [C,C] object: the per-candidate set is variable-size with no stable identity,
      so the inhibition is parametrised by action class -- the BG surround-inhibition between
      competing motor programs, Mink 1996, the opponency MECH-449 grounds). register_buffer
      (NOT nn.Parameter -- the three-factor plasticity is a LOCAL update, never an
      optimizer/autograd target; rides device + state_dict), init 0 -> the settling step is a
      no-op -> BIT-IDENTICAL OFF and bit-identical at init. C = learned_settling_n_action_classes
      (default 8; first-action class clamped into range).
    (3) W_lat learned by the SAME three-factor rule as w_chan, off ONE shared signed RPE:
      post_action_update computes delta_t = R_t - V-hat_t ONCE (R_t = benefit_eval - harm_eval
      at the realised state from the already-trained heads, detached; V-hat_t slow EMA) and
      applies Delta W_lat = eta_w * delta_t * asym(delta_t) * coact_trace, where coact_trace is
      the decayed Hebbian co-activation (outer product) of the per-round settling-step class
      activations recorded in _lateral_settle. One dopaminergic RPE drives both w_chan and W_lat
      (the post_action_update block was restructured to compute delta_t / V-hat_t / asym once and
      branch to each lever -- the w_chan-only path stays byte-identical).
  Composes INSIDE the F-bounded MECH-448/449 eligible set (raw scores / F untouched), so a
  learned W_lat can never re-admit a No-Go-excluded candidate -- safety inherited from the
  envelope, exactly as for w_chan. Waking-only (MECH-094): the settling is gated on
  not simulation_mode (no settling, no co-activation trace on a replay/DMN tick), and the
  three-factor W_lat update fires only on a pending waking trace; agent.reset() clears the
  within-episode settling trace via the extended clear_learned_channel_eligibility (W_lat and
  V-hat_t persist across episodes). Diagnostics on last_score_diagnostics: learned_settling_active
  + learned_settling_round_delta (the L2 cross-round movement of the eligible accumulator -- the
  NON-DEGENERACY signal the falsifier checks); post_action_update metrics gain wlat_delta_t /
  wlat_range.
  Config (E3Config + REEConfig.from_dims, all no-op default -> bit-identical OFF):
  use_learned_settling_step (False, master) + learned_settling_rounds (3, R) +
  learned_settling_temperature (1.0, T) + learned_settling_eta (0.01, W_lat learning rate) +
  learned_settling_elig_decay (0.9, cross-tick co-activation decay) +
  learned_settling_n_action_classes (8, the W_lat dimension, clamped). Reuses the step-1
  delta_t / V-hat_t / learned_channel_asym_potentiation/_depression (one shared signed RPE).
  Backward compatible: use_learned_settling_step=False by default -> the settling block is
  skipped, the within-eligible arbitration runs the legacy argmin/sample on the unsettled accum,
  W_lat stays zero -> bit-identical. 10/10 new contracts in
  tests/contracts/test_mech450_learned_settling_step.py (C1 config defaults + from_dims +
  W_lat zero-init / C2 OFF == ON-at-init EXACT scores+selection + settling no-op at init [round_delta
  0] + OFF writes no W_lat / C3 W_lat MOVES under a non-flat delta_t when ON [w_chan OFF -- the
  settling learns independently] / stays at init when OFF / C4 simulation tick writes no W_lat /
  C5 non-degeneracy -- a non-zero W_lat MOVES the field across rounds [round_delta > 0], no-op at
  init / C6 envelope intact -- a strong W_lat cannot re-admit an F-excluded candidate / C7 shared
  delta_t -- both w_chan and W_lat move off init under one RPE) + 9/9 JOB-1 + 8/8 preflight + 74
  e3-cluster + 1232/1235 full contract suite (the 3 fails are the documented pre-existing
  runner-fail-branch / control_vector flakes -- CONFIRMED failing identically with the e3_selector +
  config edits stashed, zero e3 overlap). Selector activation smoke: W_lat range 0.0 -> 4.23 over 13
  updates with a non-flat delta_t (last 0.7623); settling round_delta 6.16 (live cross-round
  movement); OFF == ON-at-init bit-identical (exact scores + selection). Agent env-loop smoke (real
  CausalGridWorldV2, top-k shortlist eligible set + MECH-341 channel): the settling engaged 60/60
  ticks ON, 0 OFF (wiring engages end-to-end).
  Phased training: N/A (pure-arithmetic settling + a local three-factor plasticity rule; no learned
  parameters in the optimizer/autograd sense, no new encoder head). MECH-094: waking-only by the
  simulation_mode gate + the pending-trace coupling. Evidence-staleness (Step 8.5): NOT triggered --
  no-op-default flag; every existing experiment uses the default (settling off), so no dependent
  claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-450 stays candidate / substrate_conditional /
  implementation_phase=v3; ARC-108 / ARC-107 / MECH-448 / MECH-449 / MECH-439 untouched. claims.yaml
  carries only the MECH-450 implementation_note (NOT-built -> BUILT); claims.json byte-identical
  (implementation_note not projected); validate_claims --strict exit 0.
  Validation experiment: NOT queued here -- the sec-7 selection 2x2 falsifier (learned-w_chan x
  learned-W_lat on the GAP-A divergent pool: committed-class entropy strict-above the envelope-only
  arm AND a matched-noise control, growing with training, with the signed-vs-unsigned-RPE ablation)
  is now fully runnable (both factors built) and is a separate /queue-experiment chip.
  Design doc: REE_assembly/docs/architecture/dopamine_into_gating.md (JOB-1 steps 1+2). Design-of-record:
  REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md (sec 4) +
  unified_dopamine_substrate_design_2026-06-22.md.
  See MECH-450 (this claim; factor 2 of the 2x2), ARC-108 JOB-1 step-1 (the coupled w_chan + signed-RPE
  delta_t this shares; landed earlier 2026-06-22), ARC-107 (the BG-constitution umbrella whose one-shot
  pallidal readout this replaces with a bounded recurrent settling competition), MECH-448 / MECH-449
  (the F-bounded eligible set the settling composes inside; safety inherited), MECH-439 (the F-dominance
  conversion ceiling -- a hard argmin returns the F-winner; a settling competition can flip the
  attractor), ARC-106 (grounding framework; G2 reuse-before-duplicate -- extend e3_selector, no parallel
  module), ARC-109 (D1/D2 population split -- deferred V4), MECH-094 (waking-only call-site scoping).

## ARC-108 JOB-2: dopaminergic control-plane DRIVER pair -- rho_t maintenance ramp + habenula negative-delta_t de-commit (the driver of the commit/maintain/de-commit machinery REE built but never gave its neuromodulator) (2026-06-22)
- ARC-108 (JOB 2): control_plane.dopaminergic_commit_maintain_decommit_driver -- IMPLEMENTED
  2026-06-22 (substrate; ARC-108 stays candidate / substrate_conditional / implementation_phase=v3 --
  this PROMOTES NOTHING. The sec-7.2 L0/L1/L2 control-plane falsifier is a SEPARATE /queue-experiment
  chip, sequenced after this build). The control-plane half of the unified dopamine substrate
  (JOB-1 = learned selection-gating; JOB-2 = the driver of the commit/maintain/de-commit control
  plane). Built on the MECH-450 base (ree-v3 9839492) after serializing behind that concurrent
  ARC-108 JOB-1 step-2 session (shared post_action_update RPE block). User-ratified V3 pull-forward
  (claims.yaml 045c4b73df). Design-of-record:
  REE_assembly/evidence/planning/unified_dopamine_substrate_design_2026-06-22.md secs 3-6/10;
  doc REE_assembly/docs/architecture/arc_108_job2_control_plane.md.
  PROBLEM (assembly-map A.6): REE built the commit/maintain/de-commit MACHINERY (MECH-090 beta-gate
  latch, the natural-commit latch-hold, the SD-034 closure operator, the de-commit refractory) and
  ran it off arithmetic readiness signals with NO dopaminergic driver. Two deviations: (B6 / 460h)
  maintenance is a FLAT hold -- the latch-hold re-asserts beta UNCONDITIONALLY each tick, no
  intrinsic decay, so it monopolises ~2400 steps; (rung-6) the de-commit fires on a refractory
  CLOCK, not on outcome content, so it is non-dissociable from the commit it releases. Biology: a DA
  ramp scaled to goal-proximity x value peaks-then-declines so it CANNOT monopolise (Howe 2013,
  Mohebi 2019); de-commit is lateral-habenula -> negative RPE, a content-driven "worse than expected"
  trigger (Matsumoto 2007, Hong 2011, Sosa 2021).
  THE PAIR (COMPOSE with MECH-090/342/SD-034 -- keep gate + operator + refractory as safety plumbing;
  REPLACE only the flat maintenance DRIVER, ADD the de-commit DRIVER; no parallel module, ARC-106 G2;
  both no-op-default -> bit-identical OFF; waking-only MECH-094):
    (c) rho_t MAINTENANCE RAMP -- ree_core/policy/rho_maintenance_ramp.py (RhoMaintenanceRamp,
      pure-arithmetic, no params). rho_t = goal_proximity(z_world) x value, formed from quantities
      REE already has (GoalState.goal_proximity in [0,1] x the benefit valuation feeding F,
      E3.benefit_eval_head clamped >= 0; built by REEAgent._compute_rho_t). The ramp tracks a running
      proximity PEAK and SELF-LIMITS (returns release) once rho_t has declined from the peak by
      >= release_margin*peak (or below hold_floor), after an onset grace. WIRING: at the
      natural-commit latch-hold RE-ASSERTION site (REEAgent.select_action), when
      use_rho_maintenance_ramp is on the UNCONDITIONAL re-assert is REPLACED by a ramp-gated one --
      the ramp self-limit is ADDED as a yield condition; ALL existing latch-hold yields are kept
      (refractory / MECH-091 threat / rung-6 release / max-ticks = the safety plumbing). The ramp
      only decides WHEN the hold ends. This is the STRUCTURAL B6 fix: a flat hold never crosses the
      decline test (no decline term) so it never self-limits (the 460h monopoly); a proximity-scaled
      rho_t peaks-then-declines by construction. PRECONDITION (loud ValueError): requires
      use_natural_commit_latch_hold=True (the hold this ramp drives).
    (d) HABENULA negative-delta_t DE-COMMIT -- ree_core/governance/closure_operator.py
      (ClosureOperator.habenula_tick + ClosureOperatorConfig.habenula_abort_enabled /
      habenula_delta_threshold + _n_habenula_aborts). The SAME signed delta_t = R_t - V-hat_t the
      ARC-108 JOB-1 slice computes in e3_selector.post_action_update (REUSED, not recomputed) is a
      new INTERNAL-scalar abort input: when delta_t < habenula_delta_threshold ("worse than
      expected") AND beta is elevated, habenula_tick fires the SAME 5-part _fire the operator
      already runs (beta release + No-Go + residue discharge + salience + PE cap) + the de-commit
      refractory. ADDED ALONGSIDE the rule-stability detector (tick) and the refractory-timer release
      -- the operator/refractory/No-Go machinery is NOT replaced. Content-driven (fires on outcome
      valence, not the clock) and DISSOCIABLE from the latch's refractory state -- the D3 property the
      rung-6 timer lineage could not produce. Internal scalar only (the routed GPi->habenula efferent
      drain stays V4). WIRING: REEAgent.update_residue reads e3_metrics["habenula_delta_t"] after
      post_action_update and routes it into closure_operator.habenula_tick; on a fire it tears down
      the committed program (beta released, e3._committed_trajectory=None, hold disarmed). Requires
      use_closure_operator=True (forwarded onto ClosureOperatorConfig via the closure_decommit_hold_ticks
      getattr-fallback pattern).
  delta_t REUSE plumbing: e3_selector.post_action_update computes delta_t + advances the shared
  V-hat_t whenever JOB-1 learned gating (or MECH-450 W_lat) OR use_habenula_decommit is on (broadened
  from the JOB-1/W_lat-only gate; the JOB-1 path is bit-identical), and emits habenula_delta_t.
  use_habenula_decommit is mirrored onto E3Config.use_habenula_decommit (so post_action_update, which
  reads the E3Config, computes the RPE) AND REEConfig (agent wiring + operator), both from one
  from_dims param.
  Config (REEConfig + from_dims, all no-op default -> bit-identical OFF): use_rho_maintenance_ramp
  (False, master; requires the latch-hold) + rho_hold_floor (0.05) + rho_release_margin (0.5) +
  rho_onset_grace_ticks (3); use_habenula_decommit (False, master; requires use_closure_operator) +
  habenula_decommit_delta_threshold (0.0 = fire on any negative RPE). E3Config gains the mirror
  use_habenula_decommit (False).
  Backward compatible: both masters default False -> agent.rho_maintenance_ramp is None (the
  latch-hold's flat re-assert is unchanged) and ClosureOperatorConfig.habenula_abort_enabled False
  (habenula_tick no-ops); post_action_update emits no habenula_delta_t; bit-identical OFF (verified:
  default == explicit-False). 8/8 new contracts in tests/contracts/test_arc108_job2_control_plane.py
  (C1 config defaults + bit-identical OFF / C2 rho ramp peaks-then-declines self-limit where a FLAT
  rho never self-limits + floor release / C3 rho MECH-094 sim no-op / C4 habenula_tick fires the
  SD-034 closure on a negative delta_t below threshold while beta elevated + no-ops disabled/beta-
  down/delta-above-threshold/hypothesis_tag + threshold respected / C5 agent preconditions + operator
  forwarding / C6 LOAD-BEARING agent-level: a DECLINING rho_t self-limits the hold where the flat
  hold ramp-OFF keeps re-asserting against churn / C7 e3 delta_t reuse: JOB-2-ON emits habenula_delta_t
  + advances shared V-hat, JOB-2-OFF emits none + JOB-1 w_chan untouched / C8 agent habenula de-commit
  end-to-end: a committed+elevated agent whose realised outcome is worse-than-expected fires the abort,
  releases beta, tears down the program) + 8/8 preflight + full contract suite 1240 passed (the 3
  failures -- control_vector C4 + 2 runner_fail_branch -- are the documented pre-existing flakes,
  CONFIRMED failing identically on a clean stash of these changes; control_vector C4 passes in
  isolation = the known order-dependent flake).
  Phased training: N/A (pure-arithmetic regulators + scalar reuse of the already-formed RPE; no
  learned parameters). MECH-094: the rho ramp's tick(simulation_mode=True) never self-limits and does
  not advance the peak; habenula_tick(hypothesis_tag=True) is a no-op; delta_t is computed only on the
  waking update_residue path. Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flags;
  every existing experiment uses the defaults, so no dependent claim's measured mechanism changed.
  KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. ARC-108 stays candidate / substrate_conditional /
  implementation_phase=v3; MECH-450 / ARC-107 / MECH-448 / MECH-449 / MECH-090 / MECH-342 / SD-034 /
  MECH-439 untouched. claims.yaml carries only an implementation_note on ARC-108 (DEFERRED behind the
  live governance-cycle session that holds claims.yaml; PROMOTES NOTHING, no flag/status change).
  Validation experiment: NOT queued here -- the sec-7.2 control-plane L0/L1/L2 falsifier
  (ramp-releases-where-flat-latch-monopolises D1, release content-driven not re-parameterised-timer D2,
  habenula de-commit dissociable D3, on the closure_exclusive_decommit_eval substrate, the 460k
  successor) is a SEPARATE /queue-experiment chip.
  Design doc: REE_assembly/docs/architecture/arc_108_job2_control_plane.md.
  See ARC-108 JOB-1 (docs/architecture/dopamine_into_gating.md; the shared delta_t / V-hat_t this
  reuses), MECH-450 (ARC-108 JOB-1 step-2 -- the maintenance ramp's selection-side twin, design sec 5;
  the base this built on), MECH-090 (beta-gate latch the ramp re-asserts), MECH-342 (maintenance-
  release sibling; degraded-readiness face), SD-034 (closure operator the habenula aborts),
  natural_commit_occupancy_release.md (the latch-hold the ramp drives + the closure_exclusive_decommit_eval
  substrate the falsifier runs on), MECH-439 (F-dominance front -- JOB-2 attacks the duration/de-commit
  face while JOB-1 / MECH-448 attack selection), ARC-109 (D1/D2 split -- V4), ARC-106 (grounding
  framework), MECH-094 (waking-only call-site scoping).
## ARC-108 sec-7 C3: learned_channel_rpe_mode signed/unsigned ablation flag (unblocks V3-EXQ-700 C3 arm; the signed-RPE-is-load-bearing falsifier knob) (2026-06-22)
- ARC-108 sec-7 C3: selection.learned_channel_rpe_mode -- IMPLEMENTED 2026-06-22
  (substrate; ARC-108 stays candidate / substrate_conditional / implementation_phase=v3 --
  this PROMOTES NOTHING. The C3 ablation arm itself is a SEPARATE /queue-experiment session
  gated on this flag landing). The runtime knob the V3-EXQ-700 ARC-108 sec-7 learned-gating
  2x2 DEFERRED: C3 ("swap the signed delta_t for the unsigned ARC-016 variance -- it must
  FAIL to convert; if unsigned converts just as well, route back to ARC-016; falsifies
  divergence B5") had NO runtime knob because delta_t = R_t - V-hat_t was hard-wired in the
  JOB-1/W_lat three-factor update block. Design-of-record:
  REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md sec 5.2 C3.
  THE FLAG (no-op default; bit-identical OFF): E3Config.learned_channel_rpe_mode
  (Literal["signed","unsigned"], default "signed"), surfaced through REEConfig.from_dims onto
  config.e3. In ree_core/predictors/e3_selector.py post_action_update (the shared JOB-1
  w_chan + MECH-450 W_lat three-factor block), a single learn_signal is formed:
    "signed" (default) -> learn_signal = delta_t (the signed R_t - V-hat_t) -- BYTE-IDENTICAL
      to the original substrate (learn_signal == delta_t, learn_asym == the old asym, the
      add_() and _lcg_last_delta/_wlat_last_delta unchanged).
    "unsigned" -> learn_signal = abs(self._running_variance), the UNSIGNED ARC-016
      prediction-error magnitude (always >= 0; divergence B5's precision signal), substituted
      for delta_t in BOTH the w_chan AND W_lat updates -- removing the directional
      potentiate-vs-depress credit. learn_asym is computed from learn_signal, so under unsigned
      it is fixed at potentiation (learn_signal >= 0 always); the structural reason an unsigned
      signal CANNOT produce opposite-sign w_chan moves.
  SCOPE: only the LEARNING updates (w_chan + W_lat) change. The signed delta_t itself is kept
  intact for the JOB-2 habenula negative-delta_t de-commit (it reads metrics["habenula_delta_t"]
  = delta_t) and the V-hat_t baseline EMA (_lcg_value_baseline tracks R_t signed). So the C3
  ablation isolates the SELECTION teaching signal without touching the control-plane driver.
  Config: REEConfig.from_dims(learned_channel_rpe_mode="signed"|"unsigned") -> config.e3.
  Backward compatible: default "signed" -> learn_signal == delta_t exactly -> bit-identical to
  the pre-flag substrate for every existing run. 10/10 ARC-108 contracts (9 prior + new C7) +
  the MECH-450 / ARC-108 JOB-2 / e3-cluster (mech_448 / mech_449 / conflict-grade / score-bias /
  modulatory / DR-12 / ARC-065 GAP-A) 108-test cluster + 8/8 preflight PASS. C7
  (test_arc108_learned_channel_gating.py, mirror of the C5 signed potentiate-vs-depress
  contract): under learned_channel_rpe_mode="unsigned" a good outcome (R_t=+0.8) and a bad
  outcome (R_t=-0.8) move the voting channel's w_chan in the SAME direction
  (dw_pos * dw_neg > 0) -- the opposite-sign credit C5 shows for the signed delta_t is
  structurally impossible. (A non-zero realised state drives a positive ARC-016 running
  variance; the constant valuation-head stubs make R_t independent of it, so only the unsigned
  magnitude varies.) from_dims wiring smoke: explicit "unsigned" reaches config.e3; default
  "signed".
  Phased training: N/A (no learned parameters added; reuses the JOB-1 local three-factor rule
  and the already-trained valuation heads). MECH-094: unchanged -- the substitution lives
  inside the existing waking-only post_action_update gate (a replay/DMN tick forms no
  learn_signal and writes no w_chan/W_lat). Evidence-staleness (Step 8.5): NOT triggered --
  no-op-default flag; every existing experiment uses the default ("signed"), so no dependent
  claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. ARC-108 stays candidate / substrate_conditional /
  implementation_phase=v3; MECH-450 / MECH-439 / ARC-016 / ARC-107 untouched. claims.yaml
  carries only an implementation_note on ARC-108.
  Validation experiment: NOT queued here -- the C3 ablation arm (an A1-config arm +
  learned_channel_rpe_mode="unsigned") is added to a V3-EXQ-700 sibling/successor in a SEPARATE
  /queue-experiment session, with the pre-registered acceptance that the unsigned arm must FAIL
  to convert while A1 (signed) converts; if unsigned converts just as well, the signed-RPE claim
  is refuted and the mechanism collapses to a precision re-weighting (route back to ARC-016, do
  NOT mint a learning claim).
  Design-of-record: REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md
  (sec 5.2 C3). Doc: REE_assembly/docs/architecture/dopamine_into_gating.md.
  See ARC-108 JOB-1 step-1 (the w_chan + signed-RPE delta_t this ablates; landed 2026-06-22),
  MECH-450 (ARC-108 JOB-1 step-2 -- the W_lat update the flag also covers), ARC-016 (the
  unsigned variance the ablation substitutes -- divergence B5; the route-back target if unsigned
  converts), MECH-439 (the F-dominance conversion ceiling the 2x2 attacks), V3-EXQ-700 (the
  selection 2x2 that deferred C3), MECH-094 (waking-only call-site scoping).

## F-independent closure-plane commit-ENTRY TRAJECTORY primitive (C-STEP extension of the bool latch; the between-tick path now STEPS a closure-formed committed program, not repeats _last_action) (2026-06-23)
- control_plane.closure_commit_entry_trajectory -- IMPLEMENTED 2026-06-23 (substrate;
  PROMOTES NOTHING; claims.yaml carries an implementation_note at most; MECH-445/446 stay
  candidate / standard / v3_pending / pending_retest_after_substrate). The C-STEP extension of
  the same-day bool-latch commit-ENTRY primitive (use_closure_commit_entry, ree-v3 84c1e7c).
  Routed via /implement-substrate (commitment_closure:GAP-4; rung-6 of
  f_dominance_conversion_ceiling) on top of the bool latch, after the bool-vs-trajectory
  buildability rethink (user-confirmed "build both": queue a readiness test for the bool latch +
  build the corrected trajectory latch the evidence can feed into).
  ROOT CAUSE (the residual one link past the bool latch): the bool latch e3._closure_committed_active
  arms + SUSTAINS the closure-formed beta occupancy (C-KEY -- the latch-hold yield/persistence
  check is union-aware, so the hold re-asserts beta), but a bare BOOL cannot be STEPPED. The
  between-E3-tick path (agent.py select_action) reads e3._committed_trajectory to advance a
  committed PROGRAM; on the closure-exclusive de-commit eval _committed_trajectory stays None, so
  a closure-armed hold falls through to `action = self._last_action` -- it holds beta but executes
  NO closure-formed committed program (the C-STEP gap). A bool fundamentally cannot fill the
  stepping site no matter how it is widened -- closing C-STEP REQUIRES a parallel trajectory the
  de-commit machinery consults.
  THE FIX (no-op default; bit-identical OFF; rides the bool flag):
    Module: ree_core/predictors/e3_selector.py (new field e3._closure_committed_trajectory +
      union in get_commitment_state is_committed telemetry), ree_core/agent.py (SET site +
      _closure_commit_active arm-gate union + _ncl_commit_present latch-hold-persistence union +
      the between-tick stepping union + the de-commit / closure-fire / reset clears +
      get_state is_committed telemetry + the precondition), ree_core/utils/config.py (1 new
      no-op-default flag + from_dims).
    A new sub-flag use_closure_commit_entry_trajectory (default False; PRECONDITION: requires
      use_closure_commit_entry, the bool latch it extends -- loud ValueError otherwise, the
      MECH-269b/SD-058 pattern). When on, REEAgent.select_action (at the SAME Option-A SET
      predicate where the bool is set: goal_state.is_active() AND a trajectory selected toward it
      AND lateral_pfc rule_state norm >= floor) ALSO installs the goal/rule-directed
      result.selected_trajectory into a PARALLEL STICKY latch e3._closure_committed_trajectory
      (resetting _committed_step_idx on a FRESH arm so stepping starts at the program head;
      subsequent E3 ticks refresh the trajectory but the counter keeps advancing across the held
      occupancy, mirroring the F-commit stepping which resets the counter only on beta release /
      reset). Unlike _committed_trajectory it is NOT torn down by post_action_update -- it
      persists across ticks until a principled closure teardown.
    Three UNION sites now read (_committed_trajectory OR _closure_committed_trajectory) -- all
      bit-identical when the trajectory latch is None (flag off):
      (1) agent.py _closure_commit_active arm gate -- a closure-FORMED trajectory also arms the
          coupling (alongside the legacy F-commit + the bool latch);
      (2) agent.py _ncl_commit_present latch-hold persistence -- a closure trajectory keeps the
          hold present so it re-asserts beta instead of yielding;
      (3) agent.py between-tick stepping (`_step_traj = _committed_trajectory or
          _closure_committed_trajectory`) -- the closure-armed hold ADVANCES the closure-formed
          committed PROGRAM (C-STEP) instead of repeating _last_action.
    CLEARED at the SAME sites as the bool latch (mirrored exactly): the three agent de-commit
      sites, the SD-034 auto-closure-fire teardown, and agent.reset() (episode boundary). NOT
      cleared at the e3_selector post_action_update teardown -- the closure latch must survive
      post_action_update (stickiness), same as the bool.
    is_committed telemetry (e3_selector.get_commitment_state + agent.get_state) widened with
      `or _closure_committed_trajectory is not None` so the closure-formed commit reads honestly
      on the closure-exclusive eval where _committed_trajectory stays None. The residue-write /
      hippocampal-record sites are LEFT UNCHANGED (the design wants occupancy + stepping, not
      memory recording of closure-formed commits -- matching the bool latch's scope).
  Config (REEConfig + from_dims, no-op default -> bit-identical OFF):
    use_closure_commit_entry_trajectory (False). PRECONDITION: requires use_closure_commit_entry
    (which itself requires use_closure_commit_beta_coupling + use_natural_commit_latch_hold).
  Backward compatible: use_closure_commit_entry_trajectory=False by default ->
    e3._closure_committed_trajectory is never installed -> every union reduces to the bool-latch
    behaviour (use_closure_commit_entry-only) -> bit-identical (the trajectory flag OFF reproduces
    A's 84c1e7c path byte-for-byte; verified by contract). preflight 8/8 + the closure/latch/beta-
    gate cluster (44, incl A's 6 bool-latch contracts + my 6 trajectory contracts) PASS; full
    contract suite 1253 passed (the 3 failures -- control_vector C4 + 2 runner_fail_branch -- are
    the documented pre-existing flakes, CONFIRMED failing identically with the 3 core edits
    stashed; control_vector C4 is the order-dependent flake that passes in isolation). New
    contracts: tests/contracts/test_closure_commit_entry_trajectory.py (C-KEY the F-independent
    TRAJECTORY latch installs + the hold sustains beta with ZERO F-commits / C-STEP LOAD-BEARING:
    the between-tick path STEPS the closure trajectory [proven with a trajectory whose per-step
    actions differ -- t=1 class 1, not the repeated _last_action t=0 class 0] / C-YIELD the hold
    still yields to the SD-034 de-commit refractory / C-OFF default-OFF bit-identical to the bool
    latch + reset clears the trajectory latch + the precondition raises).
  Phased training: N/A (control-state wiring; no learned parameters). MECH-094: the SET is a
    waking control-state transition (select_action waking-only by call-site scoping); no
    replay/memory write surface. Evidence-staleness (Step 8.5): NOT triggered -- no-op-default
    flag; every existing experiment uses the default (trajectory off), so no dependent claim's
    measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-445/446 stay candidate / standard / v3_pending /
    pending_retest_after_substrate; SD-034 / MECH-090 / MECH-342 / MECH-439 / MECH-448 untouched.
    claims.yaml carries an implementation_note at most (DEFERRED -- the governance-cycle session
    holds claims.yaml + substrate_queue at build time).
  Validation experiment: V3-EXQ-460m claim-free substrate-readiness diagnostic queued via
    /queue-experiment -- on the closure-exclusive de-commit eval substrate, an ARM_BOOL_LATCH
    (use_closure_commit_entry, no trajectory) vs an ARM_TRAJECTORY (+ use_closure_commit_entry_-
    trajectory) vs an ARM_ENTRY_OFF control, measuring (a) the closure-formed occupancy sustains
    with ZERO F-commits (ncl_hold_closure_armed_count>0, max consecutive beta run >> 1) on >=2/3
    seeds for BOTH latch arms where ARM_ENTRY_OFF shows 0 (the 460k/460l signature), and (b) the
    bool-vs-trajectory comparison (does the trajectory arm execute a stepped committed program the
    bool arm cannot) so the evidence informs which latch the de-commit falsifier uses. The
    460-lineage de-commit SUCCESSOR (NEW letter, supersedes the parked 460k/460l line) is NOT
    queued here -- blocked on this landing + the readiness result.
  Design doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md
    (closure-plane commit-ENTRY TRAJECTORY / C-STEP extension section). Autopsies:
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460k_2026-06-22.{md,json} +
    failure_autopsy_V3-EXQ-460l_2026-06-23.{md,json}.
  See the F-independent closure-plane commit-ENTRY primitive (the bool latch this extends; landed
    84c1e7c earlier 2026-06-23) + the closure-exclusive de-commit eval mode + natural-commit
    LATCH-HOLD amend + natural_commit_occupancy_release (the rung-6 cluster this completes),
    SD-034 / MECH-090 / MECH-445 / MECH-446 (the commitment-closure-control-plane cluster; all
    candidate, unweakened), SD-033a lateral_pfc rule_state (the "rule is being followed" source),
    MECH-091 (genuine-threat interrupt the hold never overrides), MECH-439 / MECH-448 (F-dominance
    front; this is the duration/de-commit face), V3-EXQ-460k/460l (the FAILs the cluster
    addresses), V3-EXQ-460m (validation), MECH-094 (N/A).

## MECH-276: scientist-agent counterfactual-backed attribution feedstock (waking-phase mechanism feeding the MECH-275 sleep aggregator) (2026-06-23)
- MECH-276: agent.scientist_intervention -- IMPLEMENTED 2026-06-23 (substrate;
  MECH-276 stays candidate / v3_pending, and MECH-275 stays candidate /
  substrate_conditional / v3_pending -- this PROMOTES NOTHING. The substrate-readiness
  diagnostic is queued; the MECH-275 sleep-aggregation promotion run is a SEPARATE
  later session gated on the readiness PASS). The waking-phase feedstock the MECH-275
  BayesianAggregator (ree_core/sleep/bayesian_aggregator.py) was structurally blocked
  on: MECH-275 is "only coherent given a waking-phase feedstock of counterfactual-backed
  attributions produced by MECH-276 (aggregating arbitrary correlations would produce
  noise-fit)" (claims.yaml MECH-275 functional_restatement + governance_2026_06_10).
  Because MECH-276 was unbuilt, the sleep_substrate:GAP-3b promotion run V3-EXQ-702
  (2026-06-23) had to DROP MECH-275 from its tag set; this build is the owed upstream.
  Module: ree_core/attribution/scientist_attribution_buffer.py (NEW package
  ree_core/attribution/; ScientistAttributionBuffer + ScientistAttributionConfig).
  Pure-arithmetic buffer (no nn.Module, no learned parameters, no gradient flow);
  sibling to the SD-057 IncentiveTokenBank / MECH-353 BlockedAgency regulator pattern.
  COMPOSES the existing single-pass comparators (NO new comparator, ARC-106-style
  reuse-before-duplicate): SD-031 E2WorldForward (z_world causal-footprint, domain
  "place") and ARC-033 E2HarmSForward (z_harm_s / SD-003, domain "self").
  Mechanism (the counterfactual-backed attribution, the falsifiable core of the
  MECH-275 claim): on each waking tick the agent runs the comparator on
  (z_prev, observed, a_actual) with a deterministic discriminating counterfactual
  a_cf (argmax-shifted action class):
    attribution           = ||z_observed - E2(z_prev, a_actual)||   (agency residual)
    counterfactual_contrast = ||E2(z_prev, a_actual) - E2(z_prev, a_cf)||
    is_counterfactual_backed = counterfactual_contrast >= cf_margin
  A discriminating action (the agent's choice mattered, contrast >= margin) BACKS the
  attribution with a counterfactual; a near-zero-contrast action did not discriminate
  outcomes, so its attribution is mere correlation. Under only_counterfactual_backed
  (default True) correlational records are SKIPPED -> the feedstock the aggregator reads
  is structured. Setting it False is the correlational-control arm (feed everything ->
  the predicted noise-fit, no schema revision -- the MECH-275 falsifiable secondary).
  Per-(domain, region) EMA keyed on (scale, segment_id) = the MECH-284/MECH-269 anchor
  RegionKey the sleep loop routes on (read from hippocampal.anchor_set.active_anchors()
  most-recently-created anchor; falls back to a GLOBAL_REGION sentinel when no anchor
  substrate is active). evidence_snapshot() merges across domains into a region ->
  attribution map (the v1 single-snapshot integration; per-domain snapshots are a
  documented follow-on) + carries a GLOBAL_REGION sentinel holding the global mean for
  sleep-loop fallback lookups of regions not visited that waking phase.
  Agent wiring (ree_core/agent.py): self.scientist_attribution_buffer built when
  use_scientist_attribution is on (PRECONDITION, loud ValueError: requires
  use_e2_world_forward OR use_e2_harm_s_forward -- no feedstock source otherwise).
  _update_scientist_attribution(new_latent) called from sense() after
  _update_blocked_agency, using prev-latent caches (_sci_prev_z_world / _sci_prev_z_harm_s,
  one-tick lag, the MECH-353 comparator-cache pattern). The buffer PERSISTS across
  episodes (the MECH-275 aggregator runs cross-episode); agent.reset() clears only the
  prev-latent caches, not the buffer.
  Sleep-loop hook (ree_core/sleep/phase_manager.py): _build_evidence_snapshot sources
  the MECH-276 feedstock REPLACING the MECH-284 staleness scalar when the buffer is
  present (user-confirmed: the MECH-275 claim aggregates counterfactual-backed
  attribution, not staleness); _lookup_evidence honors the GLOBAL_REGION sentinel
  fallback (no-op for staleness snapshots -> legacy path bit-identical); decay_cycle()
  + mech276_* metrics merge at cycle end.
  Config (REEConfig + from_dims, all no-op default -> bit-identical OFF):
  use_scientist_attribution (False, master) + scientist_attribution_cf_margin (0.05) +
  scientist_attribution_only_counterfactual_backed (True, the correlational-control
  lever) + scientist_attribution_ema_alpha (0.3) + scientist_attribution_decay (1.0).
  E2_world "place" recording requires world_dim>=128 (E2WorldForward.attribution_ready,
  the SD-031 discriminative-granularity guard from failure_autopsy_zworld-integration-
  cluster_2026-06-06); below threshold the comparator returns zeros and nothing is
  buffered (fails safe, NOT a vacuous zero-gap).
  Backward compatible: use_scientist_attribution=False by default ->
  agent.scientist_attribution_buffer is None and the sleep loop sources the MECH-284
  staleness scalar exactly as before -> bit-identical. 9/9 new contracts in
  tests/contracts/test_mech276_scientist_attribution.py (C1 default-OFF no-op +
  bit-identical action stream [ON-with-comparator-but-not-attribution-ready emits no
  bias] / C2 precondition raises without a comparator / C3 records cf-backed + skips
  correlational + MECH-094 sim no-op / C4 region-merged evidence_snapshot + GLOBAL
  sentinel + lookup fallback / C5 correlational-control arm buffers everything / C6
  config validation / C7 agent activation @ dim128 records + buffer persists across
  reset / C8 sleep-loop source-select + sentinel fallback + legacy staleness
  bit-identical / C9 decay_cycle) + 8/8 preflight + 179 sleep/config/boot/aggregator
  contracts PASS unchanged. Activation smoke 2026-06-23 (e2_world @ world_dim=128,
  25 steps): 24 counterfactual-backed attributions recorded (mean cf_contrast 0.35 >
  margin 0.05), buffer feeds the evidence snapshot; default OFF buffer None.
  Phased training: N/A at the substrate level (the buffer is pure arithmetic over the
  already-built comparators; no new learned parameters). BUT the comparator
  discrimination DEPENDS ON A TRAINED, action-conditional forward model (SD-056) + a
  trained encoder -- the readiness diagnostic trains them in P0 (the MECH-353 / SD-031
  lesson: an untrained world_forward floors the comparator to a vacuous zero). A
  trained-substrate failure to discriminate counterfactual-backed-vs-correlational is a
  substrate-ceiling finding, NOT a falsification of MECH-276.
  MECH-094: record() is a no-op under simulation_mode (a replay/DMN tick must not write
  attribution feedstock -- the attribution must come from real consequential action);
  doubly enforced (the comparators' comparator_residual/forward reads the agent feeds
  are themselves MECH-094-gated upstream). Evidence-staleness (Step 8.5): NOT triggered
  -- no-op-default flag; every existing experiment uses the default (buffer off), so no
  dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-276 stays candidate / v3_pending; MECH-275 stays
  candidate / substrate_conditional / v3_pending (this build lands its upstream feedstock
  but the promotion run + readiness PASS are still owed). claims.yaml carries only
  implementation_notes (no flag/confidence/status change).
  Validation experiment: V3-EXQ-703 substrate-readiness diagnostic (claim_ids=[];
  counterfactual-backed feedstock vs correlational-control arm
  [only_counterfactual_backed True vs False], posterior-movement / discrimination at
  world_dim=128 with a trained encoder + SD-056-trained world_forward in P0). PASS is the
  gate that unblocks the SEPARATE MECH-275 sleep-aggregation promotion run (a sibling of
  v3_exq_702_gap3b_sleep_cluster_promotion.py extended with a MECH-275 cross-episode
  posterior-aggregation discriminative metric over the now-real feedstock) + re-adds
  MECH-275 to sleep_substrate:GAP-3b unblocks_claims.
  Design doc: REE_assembly/docs/architecture/scientist_agent_developmental_ordering.md
  (MECH-276 section, V3 substrate status IMPLEMENTED). Plan node:
  REE_assembly/evidence/planning/sleep_substrate_plan.md (sleep_substrate:GAP-3b).
  See MECH-276 (this claim), MECH-275 (the consumer; sleep-phase Bayesian aggregation
  this feeds -- stays substrate_conditional pending the readiness PASS + promotion run),
  MECH-273 (self-model aggregation specialisation; its offline_gradient_pass is the
  z_harm_s sibling), SD-031 E2WorldForward (place comparator) / ARC-033 E2HarmSForward
  (self comparator) / MECH-256 (the general single-pass comparator mechanism these
  instantiate), MECH-269 (probe channel hypothesis generator; anchor RegionKey source),
  MECH-272 (state-gated routing), MECH-284/MECH-285 (staleness + replay sampler; the
  evidence/routing this composes with), SD-056 (action-conditional world_forward; the
  comparator-discrimination prerequisite trained in P0), MECH-277/MECH-278 (stage-1/2
  specialisations of MECH-276; V4-leaning developmental tests), SD-003 (stage-1 closure;
  superseded by MECH-256/SD-029), V3-EXQ-702 (the GAP-3b run that DROPPED MECH-275
  pending this build), V3-EXQ-703 (validation), MECH-094 (simulation gate).

## MECH-451: finer-channel-granularity E3 selection-gating (the cheap V3 rung BETWEEN ARC-108's single global w_chan and ARC-110's V4 segregated loops; explode the compressed score_bias blend into separately-learnable per-head channels) (2026-06-24)
- MECH-451: selection.intermediate_channel_granularity -- IMPLEMENTED 2026-06-24 (substrate;
  MECH-451 stays candidate / substrate_conditional / implementation_phase=v3 -- this PROMOTES
  NOTHING; EXP-0391 is the validation falsifier, NOT queued in this build session). The cheap
  V3-tractable rung the sd_v4_loop_segregation "Sequencing" section + ARC-110's notes say must be
  EXHAUSTED FIRST: if finer channels convert non-motor influence to committed action, the
  F-dominance conversion ceiling (MECH-439) is REPRESENTATIONAL COMPRESSION and the expensive V4
  ARC-110 loop-segregation build is PRE-EMPTED. Routed by failure_autopsy_V3-EXQ-700b_2026-06-24
  (the V3-EXQ-700 lineage could not validly test learned-gating conversion on the single arena;
  the V4 escalation + this V3 rung were opened concurrently). Second worked application of ARC-106
  (reuse-the-mechanism / parallel buffer / no-op-default + the cargo-cult ablation guard).
  PROBLEM: ARC-108 added the first LEARNING afferent to the ARC-107 arbitration layer -- a SINGLE
  global learned w_chan over the modulatory channels feeding the E3 _modulatory_accum site. But at
  that site "score_bias" is ALREADY the COMPRESSED dACC+lPFC+OFC+MECH-295+MECH-320+gated_policy
  blend, summed UPSTREAM in agent.py before reaching select() (the ARC-108 comment names "a finer
  per-head channel split" as the documented follow-on, out of step-1 scope). A learner that can
  only re-weight a pre-compressed blend cannot dissociate the control functions compression fused.
  MECH-451 IS that follow-on.
  THE BUILD (strict ADDITIVE extension behind a no-op-default master flag; the ARC-108 w_chan path
  is BYTE-IDENTICAL -- ARC-106 G2 reuse-the-mechanism, parallel buffer, zero risk to the V3-frozen
  substrate):
    (1) FINER REGISTRY (ree_core/predictors/e3_selector.py): the single ARC-108 "score_bias" slot
      exploded into _FCG_CHANNEL_NAMES = ("ofc","dacc","lpfc","vigour","liking","gated_policy",
      "residual","mech341","route"). The six FINER_NAMED_CHANNELS map onto the existing per-head
      biases (OFC<-SD-033b, dACC<-SD-032b adapter, lateral-PFC<-SD-033a, vigour<-MECH-320,
      liking<-MECH-295, gated_policy<-ARC-062); mech341/route preserved unchanged. "residual" =
      score_bias - sum(named present) (computed by SUBTRACTION in select()) captures everything
      ELSE summed into score_bias (MECH-314 curiosity / MECH-353 blocked-agency / SD-058 avoidance
      / SD-059 escape / any future term) -> the decomposition is EXHAUSTIVE so sum(finer) ==
      score_bias EXACTLY (the bit-identical-at-init guarantee in the authority/shortlist path).
    (2) PARALLEL LEARNED BUFFER w_chan_finer (register_buffer, NOT nn.Parameter) + _fcg_elig_trace
      + _fcg_pending/_fcg_last_delta/_fcg_n_updates, sized to the finer registry. Init at
      _LCG_W_INIT = ln(e-1) so softplus(w_chan_finer[c]) == 1.0 -> reproduces the compressed blend
      EXACTLY at init (bit-identical even when ON, until the weights train apart). V-hat_t SHARED
      with the ARC-108 _lcg_value_baseline (the two gating modes are mutually exclusive -- A1 vs A2
      arms). The ARC-108 w_chan / _lcg_elig_trace are UNTOUCHED.
    (3) REGISTRY-AGNOSTIC machinery: select() picks the active (_FCG vs _LCG) registry + buffer via
      _fcg = use_finer_channel_gating. The score_bias add-site registers one _lcg_term per present
      finer channel + the residual (finer) instead of the single "score_bias" term (legacy); the
      mech341/route add-sites use the active index. scores = raw + summed score_bias in BOTH modes
      (the authority-OFF selection path + the scores tensor are unchanged); only the recomposed
      _modulatory_accum the authority/shortlist consumes differs. The recompose
      (sum_c softplus(w_buf[c])*term), the eligibility (_fcg_elig_trace[c] += |term[selected]|),
      and the three-factor update (Delta w_chan_finer[c] = eta*learn_signal*elig_c*asym, ONE shared
      signed RPE delta_t with ARC-108) all ride the active buffer. As the finer weights diverge
      under per-channel credit, _modulatory_accum becomes a per-candidate vector != the uniform sum
      -- the conversion MECH-451 tests.
    (4) AGENT WIRING (ree_core/agent.py select_action): _fcg_channels dict (None unless the flag is
      on) captures each finer constituent's un-summed [K] bias at its existing add-site (the SAME
      tensors already summed into dacc_score_bias: dacc post-e3_gate / gp_bias / lpfc_bias /
      ofc_bias / m295_bias / tv_bias); passed as score_bias_channels=... into e3.select() ONLY when
      the flag is on (version-layering guard -- the default V3 path never sends the kwarg, so an
      older e3.select cannot raise; same doctrine as the DR-12 / Go-No-Go guards).
    (4-DEFECT, FIXED 2026-06-28 -- ARC-110 V3-EXQ-707 autopsy) The _fcg_channels builder gate read
      the TOP-LEVEL self.config.use_finer_channel_gating, which is NEVER set anywhere in ree_core
      (always False), while the consumer gate at the e3.select() call site reads
      config.e3.use_finer_channel_gating. Net: _fcg_channels was ALWAYS None -> the named channels
      (ofc/dacc/lpfc/vigour/liking/gated_policy) NEVER reached the selector; only the lumped
      residual/mech341/route did. So EVERY finer-channel experiment (704/704b/706*/707/708) ran with
      MECH-451's named decomposition DEAD -- A2_FINER_CHANNELS was only a residual/mech341/route
      3-way split, NOT the named decomposition the claim asserts. Fixed agent.py:5238 to read
      config.e3.use_finer_channel_gating (matching the consumer). Regression guard
      tests/test_arc110_loop_segregation.py::TestFinerChannelsReachSelector (fails pre-fix, passes
      post-fix). MECH-451 is effectively UNTESTED (pending_retest_after_substrate). NB a genuine
      retest is itself gated on the named bias heads carrying per-candidate range -- they currently
      emit per-candidate-FLAT output (OFC input range 0.028 -> output 0.0; MECH-191 phasic gap),
      see REE_assembly sd_v4_loop_segregation.md "VALIDATION + DEFECT 2026-06-28".
  Config (REEConfig + from_dims + E3Config, no-op default -> bit-identical OFF):
  use_finer_channel_gating (False). REUSES the ARC-108 learning knobs (learned_channel_gating_eta /
  _elig_decay / _value_baseline_beta / _asym_potentiation / _asym_depression / _rpe_mode) so
  A1_GLOBAL_WCHAN vs A2_FINER differ ONLY in channel granularity (EXP-0391's single-variable design).
  Backward compatible: use_finer_channel_gating=False by default -> score_bias_channels=None ->
  legacy single "score_bias" term -> bit-identical; the A1 use_learned_channel_gating path is
  unchanged. preflight 8/8 + 1274 contracts PASS (the only 3 failures are the documented pre-existing
  flakes -- control_vector C4 [order-dependent] + 2 runner_fail_branch [local-git-env]; the runner
  fail CONFIRMED failing identically with the e3/agent/config edits stashed). 12 new contracts in
  tests/contracts/test_mech451_finer_channel_gating.py (C1 config no-op + from_dims + softplus-unity
  init at the finer registry size / C2 finer-ON-at-init EXACT-equal scores+selection to legacy +
  ARC-108 w_chan untouched / C3 w_chan_finer MOVES under a non-flat delta_t while w_chan stays at
  init / C4 MECH-094 simulation no-op / C5 residual exhaustiveness -- only some named channels
  supplied, residual absorbs, recompose still reproduces score_bias / C6 dissociation -- different
  per-channel eligibility drives the weights APART [range>0] while identical-eligibility channels
  move together [the degenerate re-labelled-blend the noise guard catches] / C7 envelope intact --
  a finer weight cannot re-admit an F-excluded candidate). Agent-level activation smoke (real
  REEAgent via from_dims, CausalGridWorldV2, dacc+lpfc+ofc+vigour+authority on): bit-identical
  action stream finer-OFF vs finer-ON-at-init; w_chan_finer (9,) present; the three-factor update
  fired on waking ticks (_fcg_n_updates=5); ARC-108 w_chan untouched.
  Phased training: N/A (local non-backprop three-factor rule; reuses the already-trained valuation
  heads R_t = benefit_eval - harm_eval; no encoder head, no collapse risk). MECH-094: waking-only --
  eligibility recorded only on a non-simulation select(); the finer three-factor update gated on a
  pending waking finer trace (a replay/DMN tick leaves _fcg_pending False -> no write). Inherited
  from the ARC-108 path. Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every
  existing experiment uses the default (finer off), so no dependent claim's measured mechanism
  changed. KEEP all evidence.
  ARC-106 cargo-cult guard (built into EXP-0391, NOT the substrate): the degeneracy hazard is "the
  finer channels move IDENTICALLY = the compressed blend re-labelled." (a) the non-degeneracy
  readiness gate (dissociable cross-channel w_chan_finer variance via fcg_w_chan_finer_range/_std +
  a divergent GAP-A candidate pool) self-routes substrate_not_ready_requeue if unmet; (b) the
  load-bearing ablation = A1_GLOBAL_WCHAN (collapse-to-blend = one global w_chan over the sum) -- if
  A1 reproduces A2's lift the decomposition is NOT load-bearing; A1 IS an EXP-0391 arm.
  GOVERNANCE: PROMOTES NOTHING. MECH-451 stays candidate / substrate_conditional /
  implementation_phase=v3; ARC-108 / MECH-450 / MECH-439 / ARC-110 untouched. claims.yaml carries
  only an implementation_note.
  Validation experiment: EXP-0391 (manual_proposals.v1.json) -> a V3-EXQ-700-sibling on the
  GAP-A-ready foraging substrate (arms A0_ENVELOPE_ONLY / A1_GLOBAL_WCHAN / A2_FINER_CHANNELS /
  ARM_NOISE, settling W_lat OFF, landed arithmetic envelope a matched constant, SD-056-trained
  e2.world_forward + ARC-065 GAP-A candidate_summary_source=e2_world_forward divergent-pool
  precondition, PRIMARY DV = committed-action-class entropy). Queued via /queue-experiment (separate
  session). A2 lift beyond A1 -> representational compression -> pre-empts the V4 ARC-110 build; A2
  finer-weights-move-but-no-lift -> positive evidence FOR ARC-110.
  Design doc: REE_assembly/docs/architecture/mech_451_finer_channel_granularity.md.
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-700b_2026-06-24.{md,json}.
  See MECH-451 (this claim), ARC-108 (the learned-gating machinery this reuses; the single global
  w_chan over the compressed blend = the A1 reference), MECH-450 (the coupled settling step; W_lat
  OFF in this slice), ARC-110 / sd_v4_loop_segregation (the V4 per-loop build this is the cheap rung
  BEFORE; pre-empted on PASS, routed-to on no-lift), MECH-439 (the F-dominance conversion ceiling
  under test), ARC-107 (BG-constitution arbitration layer the learning composes inside),
  modulatory-bias-selection-authority / MECH-448 / MECH-449 (the F-bounded eligible set the
  re-weighted _modulatory_accum is arbitrated within; safety inherited), SD-033a/b / SD-032b /
  MECH-320 / MECH-295 / ARC-062 (the per-head biases the finer channels decompose), ARC-106
  (grounding framework; second worked application), V3-EXQ-700/700a/700b (the lineage that could not
  test conversion on the single arena), EXP-0391 (validation), MECH-094 (waking-only call-site scoping).

## ARC-110: parallel segregated cortico-BG-thalamic loops (motor / associative / limbic) + S2 in-layer null + ARC-109 D1/D2 split + MECH-452 loop-local traces (2026-06-27)
- ARC-110: selection.parallel_segregated_loops -- IMPLEMENTED 2026-06-27 (substrate; ARC-110 +
  ARC-109 + MECH-452 stay candidate / substrate_conditional / implementation_phase=v3 -- this
  PROMOTES NOTHING; the validation falsifier is queued, not yet scored). The full BG-thalamo-cortical
  loop substrate the conversion-ceiling lineage converged on from FOUR angles (V3-EXQ-700b/700c
  learned-gating+settling; V3-EXQ-704b finer-channel; V3-EXQ-706b double-gated curiosity), all of
  which showed the V3 single E3 foraging arena structurally denies (a) committed-action-class
  conversion from non-motor channels and (b) a valid same-layer null. Gate CLEARED by V3-EXQ-704b
  FAIL-to-convert (the pre-registered positive-evidence-FOR-ARC-110 outcome: compression is NOT the
  binding constraint). Cluster autopsy: REE_assembly/evidence/planning/
  failure_autopsy_704b-706b-conversion-ceiling_2026-06-27.{md,json}. Design-of-record:
  REE_assembly/docs/architecture/sd_v4_loop_segregation.md. Third worked application of ARC-106
  (reuse-the-mechanism / parallel buffers / no-op-default + load-bearing-vs-decorative ablation).
  THE BUILD (strict additive extension behind no-op-default flags; every existing path -- ARC-108
  w_chan, MECH-451 finer, MECH-450 settling, single-arena argmin -- is BYTE-IDENTICAL OFF):
    (1) SEGREGATED LOOPS (ree_core/predictors/e3_selector.py _segregated_loop_arbitrate, gated on
      config.e3.use_loop_segregation): REPLACES the single-arena within-eligible argmin. The MOTOR
      loop is F (raw_scores); the modulatory channels partition into the ASSOCIATIVE (dACC + lPFC,
      _LOOP_DEFAULT_CHANNEL_MAP) and LIMBIC (OFC + liking + vigour) loops via the active channel
      registry (finer or base). Each non-motor loop accumulates its OWN channel subset, optionally
      settles (MECH-450 W_lat) and splits into D1/D2 (ARC-109); cross-loop arbitration runs AFTER via
      Haber's ascending dopamine spiral (limbic -> assoc -> motor: weighted sum of per-loop
      preferences with loop_segregation_spiral_gain_assoc / _limbic / _motor_authority). Each loop's
      preference is NORMALISED (loop_segregation_normalize, zscore default) BEFORE arbitration -- this
      strips F's raw-magnitude advantage and is the conversion mechanism (F dominates only the motor
      loop). Runs STRICTLY within the F + MECH-449 Go/No-Go eligible set, so a non-motor loop can FLIP
      the within-eligible winner but can NEVER re-admit a suppressed candidate (safety inherited from
      the envelope, orthogonal-to-F guarantee). Meaningful only WITH finer-channel gating on (the base
      registry has one score_bias lump); the validation runs the full stack.
    (2) S2 IN-LAYER SAME-LAYER NULL (_loop_inlayer_null, gated on loop_segregation_noise_on /
      loop_segregation_noise_alpha): replaces each NON-motor loop accumulator with a magnitude-matched
      random-structure (gaussian) perturbation -- range == alpha * the real loop range -- injected at
      the SAME eligibility/settling field the loops settle on (NOT policy softmax temperature, the
      decoupled 700-lineage null). So noise_verified_lifting becomes a MEANINGFUL non-vacuity
      precondition -- the substrate-level fix for the measurement ROOT the 704b/706b autopsies named.
      Motor (F) is never nulled (the thing conversion is tested against). Selection-only (no memory
      write; MECH-094 not engaged).
    (3) ARC-109 D1/D2 POPULATION SPLIT (_d1_d2_split, gated on use_d1_d2_population_split /
      d1_da_gain / d2_da_gain): each loop's accumulator (COST) decomposes into a Go/D1 population
      (relu(-accum)) potentiated by DA (LTP) and a No-Go/D2 population (relu(+accum)) depressed by DA
      (LTD); net cost = D2 - D1; da = the shared ARC-108 value baseline V-hat_t (tanh-squashed). At
      da==0 net == accum EXACTLY (bit-identical; the dissociation is earned only once da != 0). The
      two populations make approach-avoidance CONFLICT (both high) dissociable from indifference (both
      low) -- the representational distinction the additive scalar destroys (the OCD/Parkinson/
      dyskinesia CSTC axis substrate, ARC-106 EARNS). Diagnostic loop_d1_d2_conflict_signal.
    (4) MECH-452 LOOP-LOCAL ELIGIBILITY TRACES (eligibility-recording site, gated on
      use_loop_local_eligibility_traces): credit a channel for the shared signed-RPE delta_t ONLY if
      its loop's within-loop winner matched the committed action (the loop "voted for" the outcome),
      so credit stays loop-local under one broadcast dopamine signal. Diagnostic
      loop_local_credited_channels.
  Config: E3Config.use_loop_segregation + loop_segregation_channel_map / _default_loop /
  _spiral_gain_assoc / _spiral_gain_limbic / _motor_authority / _normalize / _noise_on / _noise_alpha
  + use_d1_d2_population_split / d1_da_gain / d2_da_gain + use_loop_local_eligibility_traces (all
  no-op default; threaded through REEConfig.from_dims). Default False -> the legacy single-arena
  within-eligible path runs UNCHANGED (verified: V3-EXQ-704b finer-channel dry-run ran clean through
  the modified selector, 4/4 arms, exit 0).
  Backward compatible: disabled by default; existing experiments unaffected.
  Biological basis: functional translation of the Alexander/DeLong/Strick parallel cortico-BG-thalamic
  loop organisation integrated by Haber's ascending striato-nigro-striatal dopamine spiral (ARC-106
  L1; NOT anatomical mimicry -- each loop carries a divergence-ledger row + per-loop ablation
  falsifier; the non-degeneracy guard = live cross-loop variance, a loop pinned to the motor winner is
  a vacuous split -> substrate_not_ready_requeue).
  Phased training required: no (reuses already-trained valuation heads; all learned objects ride the
  existing ARC-108 LOCAL three-factor update, not autograd). MECH-094: all learning writes inherit the
  existing simulation_mode=False waking gate.
  C2 RELEASE 2026-06-28 (per-named-channel range-preserving routing -- unblocks C2 limbic load-bearing):
    new no-op-default flag E3Config.use_named_channel_routing. The named cortical bias HEADS emit
    per-candidate-FLAT output (MECH-191 phasic gap), so under per-loop zscore the limbic channels were
    inert and ARM_DROP_LIMBIC was byte-identical to A1_LOOPS (V3-EXQ-707, C2 untestable). When the flag
    is on (with loop seg + finer gating), each named channel's loop-arbitration term is sourced from its
    per-candidate REPRESENTATION -- agent.select_action captures ofc/lpfc -> cand_world_summaries [K,D],
    liking -> goal-proximity [K], vigour -> first-action one-hots [K,A], dacc -> payoff/effort [K,2],
    gated_policy -> summaries [K,D] -> project_channel_range (range-preserving, the SAME GAP-A path that
    keeps `route` phasic) -> score_bias_channel_routed kwarg -> e3_selector.select builds a
    loop_term_override -> _segregated_loop_arbitrate substitutes the routed term for the flat scalar in
    the LOOP ACCUMULATION ONLY. The _lcg_terms eligibility traces, the authority/shortlist
    _modulatory_accum recompose, and the F/score commit path are UNCHANGED -> bit-identical OFF. New
    diagnostics loop_named_channel_routed_ranges / loop_limbic_routed_max_range expose the per-named-channel
    routed per-candidate range for the C2 non-degeneracy gate. Selection-only (MECH-094 not engaged).
    Regression guard tests/test_arc110_loop_segregation.py::TestNamedChannelRoutingC2Release (limbic range
    > 0 + DROP != A1 once routed terms carry range) + ::TestRoutedRepsReachSelectorThroughAgent (plumbing).
  Validation experiment: V3-EXQ-707 -> superseded by V3-EXQ-707a (in-layer-null gate fix) ->
    V3-EXQ-707b (v3_exq_707b_arc110_loop_segregation_c2_release, the C2 release validation: enables
    use_named_channel_routing + limbic input modules, adds the named_channel_routing_live precondition
    before scoring C2). See /queue-experiment.
  Non-degeneracy diagnostics: loop_committed_neq_motor_winner, loop_cross_loop_winner_disagreement,
  loop_assoc_pref_range / loop_limbic_pref_range, loop_d1_d2_conflict_signal, loop_local_credited_channels,
  loop_named_channel_routed_ranges / loop_limbic_routed_max_range (C2 release).
  See ARC-110, ARC-109, MECH-452 (built here); MECH-439 (the F-dominance conversion ceiling under
  test), ARC-108 / MECH-450 / MECH-451 (the within-loop machinery reused), ARC-107 (BG constitution),
  MECH-448 / MECH-449 (the F-bounded eligible set arbitration runs within; safety inherited), ARC-106
  (grounding framework; third worked application), V3-EXQ-700b/704b/706b (the lineage that could not
  test conversion / a valid null on the single arena), MECH-094 (waking-only call-site scoping).

## ARC-108 x ARC-110 coupling: LEARNED (dopamine-gated) CROSS-LOOP arbitration -- the named next attack on the F-dominance conversion ceiling (MECH-439) after V3-EXQ-707b (2026-07-01)
- selection.cross_loop_arbitration_plasticity -- IMPLEMENTED 2026-07-01 (substrate; ARC-108 + ARC-110
  stay candidate / substrate_conditional / implementation_phase=v3 -- PROMOTES NOTHING; the validation
  falsifier is queued SEPARATELY, new EXQ / different claim_ids, not yet scored). This is the ARC-108 x
  ARC-110 INTERSECTION the 2026-06-29 MECH-439 /claim-synthesis named (borderline child MECH-453,
  recommended DROP -- ARC-108 already depends_on ARC-110), so NO new claim is minted; the coupling is
  annotated onto ARC-108 + ARC-110. Escalation source: REE_assembly/evidence/planning/
  failure_autopsy_V3-EXQ-707b_2026-06-29.{md,json} (707b built ARC-110 loop segregation FULLY LIVE -- all
  6 non-degeneracy gates passed, limbic range 1.414 -- yet the limbic loop NEVER won: A1_LOOPS 0.838 ~
  A0_SINGLE_ARENA 0.914; the autopsy traced this to the STATIC ARITHMETIC cross-loop combine, which
  inherits F's dominance because it cannot LEARN to down-weight F). Design-of-record:
  REE_assembly/docs/architecture/learned_cross_loop_arbitration.md.
  THE BUILD (strict additive extension behind a no-op-default flag; every existing path -- the static
  spiral combine, ARC-108 w_chan, MECH-450 W_lat, MECH-451 finer -- is BYTE-IDENTICAL OFF):
    LEARNED [3,3] CROSS-LOOP MATRIX (ree_core/predictors/e3_selector.py _segregated_loop_arbitrate,
    gated on config.e3.use_learned_cross_loop_arbitration). The fixed cross-loop combine
    `final = m_a*motor_z + g_a*assoc_z + g_l*limbic_z` is replaced by a learned matrix
    W_cross = I + M_cross (M_cross a [3,3] register_buffer, loop order motor/associative/limbic =
    _LOOP_NAMES, init 0): eff = W_cross @ [motor_z; assoc_z; limbic_z]; final = m_a*eff_motor +
    g_a*eff_assoc + g_l*eff_limbic. At init M_cross==0 -> W_cross==I -> eff==the per-loop zscores -> final
    is BIT-IDENTICAL to the static combine (and OFF -> the static combine runs untouched). M_cross[i,j] is
    the learned directed influence of loop j on loop i's effective preference; M_cross[motor,limbic] is the
    learnable ascending-spiral path by which the limbic value loop learns to drive the motor commit.
    LEARNING: the SAME ARC-108 signed-RPE three-factor rule as w_chan/W_lat (one shared dopaminergic
    delta_t = R_t - V-hat_t + D1-D2 asym -- Haber's single ascending spiral), via an outer-product Hebbian
    co-activation trace coact[i,j] = post_i * pre_j where pre_j = -loop_z_j[committed] (SIGNED: >0 when
    loop j preferred the committed candidate -- directional cross-loop credit) and post_i =
    -eff_i[committed]; Delta M_cross = eta_c * delta_t * asym * coact_trace (learned_cross_loop_eta 0.01).
    Waking-only (a replay/DMN tick forms no delta_t and writes no M_cross; MECH-094). M_cross is SIGNED
    (mirrors the signed W_lat). NO autograd (register_buffer + no_grad LOCAL update). Per-episode clear of
    the coact trace + pending flag (clear_learned_channel_eligibility); learned M_cross persists.
  Config: E3Config.use_learned_cross_loop_arbitration (False, master) + learned_cross_loop_eta (0.01);
    elig_decay / asym / rpe_mode / value_baseline_beta are SHARED with the ARC-108 learned_channel_* knobs
    (one dopamine system). Threaded through REEConfig.from_dims. Requires use_loop_segregation on to act.
  Coupling with MECH-450: the per-loop settling (W_lat) shapes each loop's WITHIN-loop competition BEFORE
    normalise/arbitrate; the learned cross-loop weights arbitrate ACROSS the settled loops; both ride the
    same shared delta_t in one post_action_update (the "dopamine-into-gating + recurrent-settling, coupled"
    the BG-assembly map names).
  Safety: arbitration runs STRICTLY within the F + MECH-448/449 Go/No-Go eligible set, so a learned weight
    can reorder within-eligible candidates but can NEVER re-admit a suppressed one (inherited, weights
    irrelevant to admission).
  Backward compatible: disabled by default; existing experiments unaffected (verified: full suite green;
    byte-identical-at-init contract test asserts ON-at-init == static combine across 12 seeds).
  Biological basis: functional translation of dopamine-gated striatal plasticity operating on the
    Alexander/DeLong/Strick segregated loops via Haber's ascending dopamine spiral (ARC-106 L1-L2; NOT
    anatomical mimicry -- 4 divergence-ledger rows incl. the honest forward-linearity note; the [3,3]'s
    value is the DIRECTED credit structure it learns). Psychiatric-failure-mode column (ARC-106 EARNS):
    motor-dominant-unadaptable = avolition; runaway limbic->motor = OCD-like over-valuation; dead delta_t =
    apathy/inflexibility (loop-specific CSTC axis with a PLASTIC arbitration knob).
  Phased training required: no (reuses the already-trained valuation heads; the matrix rides the LOCAL
    three-factor update, not autograd). MECH-094: waking-only; selection-only (writes nothing to memory).
  New diagnostics: loop_learned_cross_loop_active, loop_cross_loop_w_motor_eff / _assoc_eff / _limbic_eff
    (effective column weights w_eff[j] = sum_i gain_i*W_cross[i,j] -- what the linear forward commit
    depends on; limbic learning to win == w_eff[limbic] rising toward/above w_eff[motor]),
    loop_cross_loop_limbic_ge_motor, loop_cross_loop_m_range (non-vacuity: >0 == weights moved off init),
    loop_cross_loop_limbic_to_motor (M_cross[motor,limbic]); post_action_update metrics clg_delta_t /
    clg_m_cross_range / clg_limbic_to_motor / clg_n_updates.
  Regression guard: tests/contracts/test_learned_cross_loop_arbitration.py (byte-identical-OFF/at-init;
    non-vacuity M_cross moves; MECH-094 waking gate; limbic-can-win mechanism; from_dims plumbing;
    within-eligible-set safety).
  Validation experiment: SEPARATE new-EXQ falsifier (A1_LOOPS + learned DA-gated cross-loop arbitration
    STRICT-ABOVE A1_LOOPS + static-arithmetic arbitration; different claim_ids), queued via
    /queue-experiment AFTER this build lands + tests pass (per 707b routing step 3). EXQ TBD.
  See ARC-108 / ARC-110 (annotated -- the coupling), MECH-439 (conversion-ceiling umbrella under test),
    MECH-448 / MECH-449 / ARC-107 (F-bounded eligible set; safety), MECH-450 (settling, coupled), MECH-452
    (loop-local traces), ARC-109 (D1/D2), ARC-106 (grounding), V3-EXQ-707b (escalation source),
    REE_assembly/evidence/planning/basal_ganglia_assembly_map_2026-06-22.md (the named next attack).

## MECH-440 / MECH-441: state-conditioned exploration -- propagating selection-head weight noise (NoisyNet) + model-disagreement directed curiosity (RND/Plan2Explore) (2026-06-27)
- MECH-440: selection.state_conditioned_self_annealing_noise_floor -- IMPLEMENTED 2026-06-27
  (substrate; stays candidate / substrate_ceiling / v3_pending -- PROMOTES NOTHING; falsifier queued,
  not scored). Module ree_core/policy/noisy_selection_head.py (NoisySelectionHead /
  NoisySelectionHeadConfig). The mechanistic refinement of MECH-313's NON-PROPAGATING temperature
  floor (V3-EXQ-687 r1a_entropy_only_artefact): factorised-Gaussian WEIGHT NOISE built per-candidate
  from each candidate's first-action vector and added into _modulatory_accum BEFORE the within-eligible
  argmin (and into the ARC-110 segregated-loop `final`), so it PROPAGATES into the committed action.
  mu FROZEN at 0 (pure noise injector -- isolates the falsifier). State-conditioned via the action
  activation; SELF-ANNEALS via a LOCAL confidence EMA on (1 - F-gap_norm). Lazy-built in e3_selector
  (action_dim from the candidate width). Config (E3Config): use_noisy_selection_head (default False;
  bit-identical OFF), noisy_selection_sigma_init (default 0.0 -> exactly-zero output -> bit-identical
  even ON), noisy_selection_weight, noisy_selection_anneal{,_floor,_ema_alpha}. Data flow: candidate
  first-actions -> NoisySelectionHead -> per-candidate bias -> _modulatory_accum + scores -> committed
  argmin. MECH-094: zero perturbation on simulation ticks. Phased training: NOT required (sigma is a
  buffer annealed by a local rule, not gradient-trained -- ARC-106 divergence #2). Diagnostics:
  noisy_selection_active / noisy_selection_bias_range (+ get_state()). Smoke-tested: sigma=0
  bit-identical; sigma>0 flips the committed within-eligible argmin across resamples (propagation
  confirmed); finiteness + EMA guards prevent any nan injection.
  Validation experiment: <EXQ pending Step 8 -- the corrected 4-arm tonic-noise falsifier on the 569i
  top-k + MECH-448 demotion stack, with loop-seg OFF vs ON arms to disambiguate injection-site from the
  single-arena/ARC-110 axis>.
- MECH-441: selection.model_disagreement_directed_curiosity -- IMPLEMENTED 2026-06-27 (substrate;
  stays candidate / substrate_ceiling / v3_pending -- PROMOTES NOTHING; FALSIFIER HELD, see below).
  Module ree_core/policy/model_disagreement.py (ModelDisagreementEnsemble / ModelDisagreementConfig):
  a standalone SMALL K-head ensemble of ResidualHarmForward delta-predictors over (z_world, action);
  per-candidate cross-head VARIANCE feeds E3 selection as a propagating per-candidate curiosity BONUS
  (a cost reduction, lower=preferred) added into _modulatory_accum -- unlike the V3-EXQ-590a broadcast
  EMA, this is per-candidate so it can change the committed argmin (WITHIN the F-eligible set; safety
  inherited). Self-annealing is INTRINSIC (variance -> 0 as the heads train). Built at the agent level
  (self.disagreement_ensemble) only when use_model_disagreement_curiosity AND n_disagreement_heads>=2;
  per-candidate disagreement computed in select_action and passed via the version-layering-guarded
  model_disagreement_per_candidate kwarg (default V3 path never sends it). Config: E3Config.
  use_model_disagreement_curiosity (default False), model_disagreement_weight (default 0.0),
  model_disagreement_mode/scale; LatentStackConfig.n_disagreement_heads (default 0 -> not built),
  disagreement_bootstrap_mask_prob, disagreement_learning_rate. Data flow: z_world + candidate
  first-actions -> K-head ensemble cross-head variance -> per-candidate bonus -> _modulatory_accum ->
  committed argmin. Phased training REQUIRED: train each head on the FROZEN z_world target
  (disagreement_ensemble.train_step(); P1 stop-gradient). MECH-094: waking-only no_grad read.
  Smoke-tested: ensemble disagreement supra-floor; per-candidate bonus steers the committed
  within-eligible argmin; sim-gated; train_step runs. FALSIFIER HELD (blocked_substrate) gated on
  ARC-110 validation V3-EXQ-707 returning contributory -- failure_autopsy_704b-706b-conversion-ceiling_
  2026-06-27 found the single-arena collapse (not the curiosity channel; 706b proved the channel works)
  is the binding constraint, so a MECH-441 run before ARC-110 is validated would re-derive the arena
  ceiling (vacuous FAIL). User decision 2026-06-27.
- Both: ARC-106 divergences logged in the grounding ledger -- (1) per-parameter sigma is one level
  below biology's systems-level tonic/phasic LC-NE mode gate; (2) sigma self-anneals via REE's LOCAL
  confidence EMA, not NoisyNet's RL gradient (REE does not backprop through E3 selection). Decision of
  record: REE_assembly/evidence/decisions/cpkt_tonic_exploration_noise_build_decision_2026-06-27.md.
  Design-of-record: REE_assembly/docs/architecture/state_conditioned_exploration_noise_floor.md
  (#mech-440 / #mech-441). See ARC-065 (parents), MECH-313 (440 refines), MECH-314 (441 refines),
  ARC-110 (the multi-arena substrate both build on top of), MECH-439 (the F-dominance ceiling),
  MECH-448 / 569i top-k (the SOTA conversion stack the falsifiers run on).

## MECH-140 x MECH-450: disinhibitory soft-competitive settling (parameter-free) (2026-07-02)
- MECH-140 x MECH-450: selection.disinhibitory_soft_competitive_settling -- IMPLEMENTED
  2026-07-02 (substrate; MECH-140 + MECH-450 stay candidate -- this PROMOTES NOTHING, they
  stay candidate until a falsifier converts them). ree_core/predictors/e3_selector.py
  (_soft_competitive_settle) + ree_core/utils/config.py. The PARAMETER-FREE, always-graded
  complement to the LEARNED W_lat settling (use_learned_settling_step, a no-op at init): a
  few rounds of soft-competitive lateral inhibition over the F + MECH-448/449 within-eligible
  field BEFORE the commit, so the committed action emerges from a bounded recurrent SETTLING
  competition rather than a one-shot argmin (MECH-450), with losing options down-weighted
  GRADED-ly but never silenced (MECH-140 -- soft-competitive disinhibition, not
  winner-take-all). Unlike W_lat it bites IMMEDIATELY (no dopaminergic learning) so it can
  flip the committed attractor at init -- the "recurrence gives the readout an attractor-flip
  the argmin lacks" property MECH-450 asserts.
  Dynamics (COST units, lower=better): x=-field (activation); for r in R: support=softmax(x/T)
  (graded, all > 0 -> never silenced); inhib_i = gain * sum_j K_ij*support_j; x -= inhib_i;
  return -x. K is the PARAMETER-FREE class-surround kernel (1.0 within a first-action class,
  cross_class < 1 across classes, 0 diagonal -- surround inhibition between competing motor
  programs, Mink 1996; the SAME structure W_lat learns, here FIXED). The STRUCTURED kernel is
  what lets settling REORDER (a candidate crowded by same-class rivals loses to an isolated
  slightly-worse one) -- the behavioural non-vacuity the MECH-439 ceiling needs, not a
  rank-preserving sharpen.
  Config: E3Config.use_soft_competitive_settling (default False -> bit-identical OFF) +
  soft_competitive_settling_gain (default 0.0 -> EXACT no-op even when the flag is on;
  byte-identical at flag-off / at-default, mirroring noisy_selection_sigma_init=0.0 -- set a
  positive gain to activate) + soft_competitive_settling_rounds (3) / _temperature (1.0) /
  _cross_class (0.25). Threaded through REEConfig.from_dims.
  Data flow: _modulatory_accum[eligible_idx] (single arena) OR the arbitrated cross-loop
  `final` (ARC-110 segregation) -> _soft_competitive_settle -> settled field -> committed
  argmin. COMPOSES with the learned W_lat within-loop settling and the learned cross-loop
  arbitration (M_cross / use_learned_cross_loop_arbitration, V3-EXQ-709) -- it is a
  within-eligible / within-loop transform at a DIFFERENT level, orthogonal flag, default-OFF
  -> no collision with 709.
  Safety: transforms ONLY the eligible subset -> a No-Go/F-excluded candidate is never touched
  and never selectable (no global disinhibition); >= 1 survivor always; needs >= 2 eligible.
  MECH-094: waking-only (no settling on a simulation/replay tick). No learned parameters, no
  autograd (detached), so phased training NOT required. ARC-106 divergence (logged): the kernel
  is a FIXED class-surround, biology's is learned + graded-by-similarity (that is W_lat's job) --
  this is the always-on floor beneath the learned inhibition.
  Contract tests: tests/contracts/test_soft_competitive_settling.py (byte-identical OFF/gain-0
  across 12 seeds; non-vacuity flips the readout + round_delta>0; graded-not-WTA loser stays
  nonzero; safety harmful-outlier never selected; MECH-094 sim-gate; segregated-path compose).
  Full ree-v3 suite green (1312 contracts pass + 11 new; the 3 residual failures --
  control_vector C4 flake + 2 runner-fail-branch -- are pre-existing, clean-tree stash-confirmed).
  Validation experiment: <EXQ pending Step 8 -- a validation falsifier (settling ON vs OFF on the
  569i top-k + MECH-448 demotion conversion-ceiling stack) AND the PLOS-Biology ablation falsifier
  (lateral-inhibition edge intact vs ablated; pre-registered: ablation collapses task-context/
  switching while single-task performance survives -- the mouse-V1-silencing signature)>.
  Biology + ARC-106 grounding ladder, load-bearing-vs-decorative ablation test, divergence ledger,
  and the required psychiatric_failure_mode column:
  REE_assembly/docs/architecture/soft_competitive_disinhibition_settling.md. Lit basis (2026-07-02
  /lit-pull): Gallo Aquino/Rungratsameetaweemana PLOS Biology 2026 (inhibition-on-inhibition = the
  top-down context channel; ablating it collapses task-switching) + Keller et al Neuron 2020
  (VIP->SOM disinhibition necessary+sufficient for contextual modulation); Lee & Sabatini Nature
  2021 (indirect-pathway competitive disinhibition, not hard suppression) + Morita 2016 (striatal
  WTA vs cortical soft-max) for MECH-140; Wang 2002 / Rolls 2021 recurrent-attractor for MECH-450.
  See MECH-140 (soft-competitive disinhibition), MECH-450 (minimal recurrent settling),
  use_learned_settling_step (W_lat, the learned counterpart), ARC-110 / use_learned_cross_loop_
  arbitration (the loops it composes with), MECH-448 / MECH-449 (the eligible set it runs within),
  MECH-439 (the F-dominance conversion ceiling), ARC-106 (grounding framework).

## ARC-110 x ARC-108: ascending-spiral gain (V3-EXQ-709/710 loop-effective-weight repair) (2026-07-03)
- ARC-110 x ARC-108: selection.cross_loop_ascending_spiral_gain -- IMPLEMENTED 2026-07-03
  (substrate; PROMOTES NOTHING -- MECH-439/ARC-108/ARC-110/MECH-450/MECH-140 stay candidate until the
  validation falsifier converts them). ree_core/predictors/e3_selector.py (_ascending_gain_matrix +
  the W_cross forward assembly + the M_cross post_action_update) + ree_core/utils/config.py.
  Repairs the loop-effective-weight ceiling BOTH the V3-EXQ-709 AND V3-EXQ-710 confirmed autopsies
  routed to as the load-bearing V3-closure build: the learned cross-loop matrix M_cross ENGAGES
  (709: 6/7 readiness gates, M_cross range 0.116, limbic routing 1.414) yet the ascending path
  M_cross[motor,limbic] peaks ~0.03 -- functionally too weak to lift a non-motor (limbic) loop to the
  motor loop's effective column weight w_eff[j]=sum_i gain_i*W_cross[i,j], so limbic_loop_can_win was
  met on only 1/4 divergent seeds (thr 2). Biology (Haber 2000): the striato-nigro-striatal spiral is
  anatomically ASYMMETRIC -- ascending (limbic -> associative -> motor) influence is the
  developmentally-strengthened, load-bearing direction. In the motor(0)/assoc(1)/limbic(2) ordering
  the forward map eff_i=sum_j W_cross[i,j]*z_j makes the ascending entries exactly the strict upper
  triangle (row<col): W_cross[0,2] (limbic->motor), W_cross[0,1] (assoc->motor), W_cross[1,2]
  (limbic->assoc). Two knobs scale ONLY those entries:
    1. forward gain (_ascending_gain_matrix in W_cross): W_cross = I + (G_fwd .* M_cross), G_fwd
       upper-tri = spiral_gain else 1.0 -- the ANATOMICAL ascending-projection strength. Raises
       w_eff[limbic]/w_eff[assoc] WITHOUT touching w_eff[motor] (motor column is diagonal + descending,
       un-amplified) -> strengthens the ascending coupling AND implicitly DE-PINS the motor(F) default.
       Keeps the map LINEAR (constant elementwise scaling of M_cross) -> w_eff-collapsibility (CLA-3)
       and bit-identical-at-init both hold (at init M_cross==0 -> gain*0==0 -> W_cross==I for any gain).
    2. plasticity maturation gain (post_action_update): the ascending entries of the M_cross
       three-factor UPDATE are scaled by plasticity_gain -- the ascending SPIRAL-MATURATION RATE
       (ascending credit accrues faster than descending). eta stays the base rate.
  Config: E3Config.use_ascending_spiral_gain (default False, master) +
  loop_segregation_ascending_spiral_gain (default 1.0, forward) +
  loop_segregation_ascending_plasticity_gain (default 1.0, maturation). Threaded through
  REEConfig.from_dims. Requires use_learned_cross_loop_arbitration (hence use_loop_segregation) on to
  act. Default False / gains 1.0 -> BIT-IDENTICAL OFF (G matrices become all-ones).
  Data flow: M_cross -> (forward) G_fwd .* M_cross -> W_cross @ [motor_z; assoc_z; limbic_z] -> final
  -> commit; (learning) three-factor delta -> G_plast .* delta (ascending entries) -> M_cross.add_.
  The w_eff / limbic_ge_motor diagnostics use the SAME gained W_cross (so limbic_loop_can_win reads
  true effective weights); clg_limbic_to_motor stays the RAW M_cross[0,2] (measures learning). New
  diagnostics: loop_ascending_spiral_gain_active / _forward. eta + P2 length remain independently
  sweepable complementary levers (already exposed).
  Safety: unchanged -- arbitration stays STRICTLY within the F+MECH-448/449 eligible set; the gain
  reorders within-eligible candidates and can NEVER re-admit a No-Go-suppressed one. F still fully owns
  the MOTOR loop; the gain only stops F from drowning the limbic value. MECH-094: the maturation gain
  rides the existing waking-only M_cross update (a simulation tick forms no delta_t, writes no M_cross);
  no new encoder / no autograd -> phased training NOT required.
  Contract tests: tests/contracts/test_ascending_spiral_gain.py (8 contracts: byte-identical OFF across
  12 seeds; byte-identical at gain==1.0; at-init identity under a large gain; asymmetry -- motor
  effective weight invariant while limbic/assoc rise; mechanism -- a small learned ascending M_cross
  flips the commit off the motor F-winner under gain + limbic_ge_motor crosses True; plasticity
  maturation scales ascending entries ~Nx while descending/diagonal stay bit-identical; from_dims
  plumbing; safety within the eligible set). Full ree-v3 suite green (1320 contracts + 8 new; the 3
  residual failures -- control_vector C4 + 2 runner-fail-branch -- are pre-existing, clean-tree
  stash-confirmed). Backward-compat: V3-EXQ-709 --dry-run reproduces the 709 FAIL signature
  bit-for-bit (preconditions_met=False, limbic_can_win=False, m_range_peak=0.0497, C1 static==learned).
  Validation experiment: V3-EXQ-711 queued (Step 8) -- a NEW-EXQ falsifier re-running the 709/710-style
  STATIC/OFF vs ascending-gain-ON arms on the GAP-A reef-bipartite foraging substrate, matched seeds,
  same non-vacuity self-route (substrate_not_ready_requeue when limbic_loop_can_win still unmet; never
  a false weakens).
  Biology + ARC-106 divergence ledger: REE_assembly/docs/architecture/learned_cross_loop_arbitration.md
  (Addendum: ascending-spiral gain). See ARC-110 / ARC-108 (owning coupling), MECH-439 (the
  F-dominance conversion ceiling), MECH-450 (settling), MECH-140 (disinhibition -- also unblocked),
  use_learned_cross_loop_arbitration (the matrix this gains), ARC-106 (grounding framework).

## ARC-110 x ARC-108: BOUNDED ascending-spiral gain -- target-PARITY controller (V3-EXQ-711 runaway repair) (2026-07-04)
- ARC-110 x ARC-108: selection.cross_loop_ascending_parity_controller -- IMPLEMENTED 2026-07-04
  (substrate; PROMOTES NOTHING -- MECH-439/ARC-108/ARC-110 stay candidate + pending_retest_after_substrate
  until the successor falsifier converts them). ree_core/predictors/e3_selector.py (_parity_forward_gain +
  the W_cross forward assembly + the diagnostic w_eff block + the M_cross post_action_update clamp) +
  ree_core/utils/config.py. Repairs the RUNAWAY the confirmed failure_autopsy_V3-EXQ-711_2026-07-04 found
  in the RAW-scalar ascending gain above: at raw scalar 20x-forward x 5x-plasticity the plastic ascending
  M_cross entries compounded through the positive-feedback plastic loop and RAN AWAY (M_cross range peak
  4897.8 vs the un-gained ~0.02-0.12; w_eff[limbic] peak 10-2274x w_eff[motor] across the 3 divergent
  seeds) -- a limbic-loop MONOPOLY that merely replaces the F/motor-pinning, not a fair parity win, and
  committed-class entropy FELL below baseline on 2/3 divergent seeds. The 709->711 pattern showed the raw
  scalar has NO stable parity regime (sub-threshold 709 never wins; runaway 711 monopolizes): the mechanism
  was MISSING A CONTROLLER. This replaces the unbounded multiply with actuator-saturated setpoint control:
    1. FORWARD parity-ceiling (_parity_forward_gain in the W_cross assembly): a per-step ascending gain in
       [0, parity_forward_gain] SOLVED so the limbic effective column weight w_eff[limbic] is LIFTED toward
       but HARD-CAPPED at parity_ceiling_ratio * w_eff[motor]. The motor column carries no strict-upper-tri
       entry -> w_eff[motor] is gain-invariant -> the fixed parity reference. Bounds the
       w_eff[limbic]/w_eff[motor] RATIO -> a FAIR within-eligible reorder, never a monopoly. Applied via the
       same _ascending_gain_matrix so the map stays LINEAR and bit-identical-at-init (M_cross==0 ->
       W_cross==I for any gain).
    2. MATURATION bounded loop (post_action_update): the ascending three-factor update is scaled by the
       BOUNDED parity_plasticity_gain, then the ascending (upper-tri) M_cross entries are clamped to
       [-m_cross_clamp, m_cross_clamp] -- an anti-windup clamp that stops the plastic positive-feedback loop
       from running away (the second 711 runaway source).
  Config: E3Config.use_ascending_parity_controller (default False, master switch) +
  loop_segregation_parity_forward_gain (default 1.0, lift strength) + loop_segregation_parity_ceiling_ratio
  (default 0.0 = disabled, the w_eff[limbic]<=ratio*w_eff[motor] cap) + loop_segregation_parity_plasticity_gain
  (default 1.0, bounded maturation rate) + loop_segregation_m_cross_clamp (default 0.0 = disabled, the
  ascending |M_cross| bound). Threaded through REEConfig.from_dims. Requires
  use_learned_cross_loop_arbitration (hence use_loop_segregation) on to act. TAKES PRECEDENCE over
  use_ascending_spiral_gain when both are on (the raw path is retained ONLY for 709/711 reproducibility).
  Master switch False -> BIT-IDENTICAL OFF; sub-params default inert (forward_gain 1.0 = no lift,
  ceiling_ratio 0.0 / m_cross_clamp 0.0 = disabled) so the successor's ON arm configures them explicitly.
  Data flow: M_cross -> (forward) g=_parity_forward_gain(M) -> G(g) .* M_cross -> W_cross @
  [motor_z;assoc_z;limbic_z] -> final -> commit; (learning) three-factor delta -> G_plast .* delta ->
  M_cross.add_ -> clamp ascending entries to [-clamp, clamp]. The w_eff / limbic_ge_motor diagnostics use
  the SAME parity-gained W_cross. New diagnostics: loop_ascending_parity_controller_active /
  _parity_forward_gain_applied / _parity_ceiling_ratio (and loop_ascending_spiral_gain_active now reports
  False whenever the controller is active, so a saturation guard can read which path drove selection).
  Safety: unchanged -- arbitration stays STRICTLY within the F+MECH-448/449 eligible set; the controller
  reorders within-eligible candidates and can NEVER re-admit a No-Go-suppressed one. F still fully owns the
  MOTOR loop. MECH-094: the maturation clamp rides the existing waking-only M_cross update (a simulation
  tick forms no delta_t, writes no M_cross); no new encoder / no autograd -> phased training NOT required
  by the substrate (the successor still phases P0/P1/P2 because the cross-loop M_cross LEARNS, same as 711).
  ML/AI note (Layer 7): actuator-saturated setpoint control -- output saturation = the parity ceiling,
  integrator clamp = the M_cross anti-windup clamp; the standard fix for an unbounded gain on a
  positive-feedback plastic loop that diverges. Biologically compatible: Haber's spiral is a graded, BOUNDED
  modulation held in parity by tonic-DA homeostasis + striatal lateral inhibition -- the raw scalar had the
  symbol without that bounding dependency.
  Contract tests: tests/contracts/test_ascending_parity_controller.py (9 contracts: byte-identical OFF
  across 12 seeds; at-init identity under large params; inert ON (fwd 1.0 + ceil 0.0); parity ceiling
  bounds w_eff[limbic]/w_eff[motor] where the raw scalar runs to 144x while motor stays invariant;
  maturation clamp bounds ascending M_cross where it otherwise runs to ~7.7e5; parity win not monopoly --
  limbic_ge_motor True while staying under the ceiling; from_dims plumbing; safety within the eligible set;
  precedence over the raw scalar). Both ascending suites green (18 contracts). Backward-compat: the raw
  test_ascending_spiral_gain.py 8 contracts still pass bit-for-bit (raw path untouched).
  Validation experiment: V3-EXQ-713 queued (Step 8) -- a NEW-EXQ successor falsifier (OFF vs bounded-ON on
  the GAP-A reef-bipartite substrate; a saturation-guarded limbic_loop_can_win gate requiring a parity BAND
  win + a w_eff/M_cross ceiling; C1 committed-class entropy strict-above the un-gained baseline on >=2/3
  divergent seeds; non-vacuity self-route substrate_not_ready_requeue, never a false weakens). This is a
  redesign of a DIFFERENT mechanism (the controller), NOT a raw-gain-magnitude re-letter -- the
  re-derive brake in the 711 autopsy explicitly REFUSES a same-claim raw-gain re-queue.
  Biology + ARC-106 divergence ledger: REE_assembly/docs/architecture/learned_cross_loop_arbitration.md
  (Addendum: ascending-spiral gain -> Sub-addendum: bounded parity controller). See ARC-110 / ARC-108
  (owning coupling), MECH-439 (the F-dominance conversion ceiling), use_ascending_spiral_gain (the raw path
  this bounds), use_learned_cross_loop_arbitration (the matrix this gains), ARC-106 (grounding framework).
