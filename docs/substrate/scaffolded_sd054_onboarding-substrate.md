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
