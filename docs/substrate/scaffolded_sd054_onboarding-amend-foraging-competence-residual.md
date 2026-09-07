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
