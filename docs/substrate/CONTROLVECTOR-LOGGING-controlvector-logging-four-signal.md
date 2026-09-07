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
