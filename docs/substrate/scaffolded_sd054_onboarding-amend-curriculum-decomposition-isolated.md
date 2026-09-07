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
