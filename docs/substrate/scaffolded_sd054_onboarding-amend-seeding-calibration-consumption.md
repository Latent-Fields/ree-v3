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
