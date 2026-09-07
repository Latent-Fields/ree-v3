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
