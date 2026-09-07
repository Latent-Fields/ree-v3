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
