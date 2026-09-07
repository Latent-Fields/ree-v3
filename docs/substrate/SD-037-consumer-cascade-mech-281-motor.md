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
