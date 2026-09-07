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
