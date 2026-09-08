## MECH-204 Phase 7 / Option B: sleep.accuracy_anchored_broadcast_recalibration -- IMPLEMENTED (2026-07-20)
- MECH-204 Phase 7 / Option B: sleep.accuracy_anchored_broadcast_recalibration --
  IMPLEMENTED 2026-07-20. Ungated from V4 deferral by the confirmed
  failure_autopsy_V3-EXQ-774_2026-07-17 (adjudicated substrate_ceiling,
  "F1 alone insufficient": precision saturates during waking before the
  per-cycle WRITEBACK lever gains headroom; effect on 1/3 seeds only).
  Method: E3TrajectorySelector.broadcast_precision_pull(target_precision, gain)
  in ree_core/predictors/e3_selector.py; called at the TOP of
  REEAgent.select_action (ree_core/agent.py) so the anchored rv is what this
  tick's commit gate and current_precision consumers see.
  Config: REEConfig.use_rem_precision_broadcast (default False) +
  REEConfig.rem_precision_broadcast_gain (default 0.0). Both also on the
  from_dims factory path. Per-STEP gain -- keep well below
  rem_precision_recalibration_step, which fires once per sleep CYCLE.
  Data flow: serotonin._persistent_zero_point -> compute_recalibration_target()
  -> select_action broadcast read -> broadcast_precision_pull ->
  E3._running_variance. Reads the F1 cumulative reference per lit choice (a)
  (targeted_review_rem_precision_recalibration_timing SYNTHESIS: Hobson-Hong-
  Friston 2014 + Walker-Stickgold 2006). Runs ALONGSIDE F1, not replacing it
  (Q-042 dual-arm pattern).
  WRITE-SITE CORRECTION -- read this before "restoring" the spec: the
  2026-05-09 spec said "additive bias on E3 score". That site is PROVABLY
  SELECTION-INERT for a broadcast. A broadcast is ONE scalar for all K
  candidates; e3_selector applies score_bias as `scores = scores + bias_tensor`
  (a uniform shift, invariant under argmax AND softmax), and every downstream
  consumer is relative (raw_scores.max() - raw_scores[i], raw_score_range,
  topk, cutoff/envelope) with several reading raw_scores, which score_bias
  never touches. It would register a nonzero modulatory channel while changing
  no behaviour -- the exact shape the inert_arm_knob lint exists to catch. The
  lit-pull adjudicated WHAT TO READ, never WHERE TO WRITE. Precision space is
  non-inert: rv feeds the ABSOLUTE commit threshold and 1/(rv + 1e-6), and is
  where V3-EXQ-774's own DV lives. Decision-log entry 2026-07-20 in
  evidence/planning/sleep_substrate_plan.md.
  Backward compatible: disabled by default; existing experiments unaffected.
  Phased training required: no (no new head, no learned parameters).
  MECH-094: not applicable (no simulation/replay content written to memory).
  PAIRED WITH SD-076 -- the broadcast corrects drift, SD-076 creates it. Phase 7
  alone cannot lift the 774 ceiling, because without a drift source the DV is a
  tautology. Do not run the Phase-7 arm without considering SD-076.
  Smoke: 6/6 PASS (UC1 bit-identical OFF by explicit float equality, UC4 pull
  arithmetic, UC4b gain=0 no-op, UC5 no-REM sentinel no-op, UC6 defaults).
  Validation experiment: see sleep_substrate_plan.md Phase 7 status row.
  See MECH-204, MECH-173, Q-042, SD-076.
