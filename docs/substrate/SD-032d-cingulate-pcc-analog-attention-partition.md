## SD-032d: cingulate.pcc_analog_attention_partition -- IMPLEMENTED (2026-04-19)
- SD-032d: cingulate.pcc_analog_attention_partition -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/pcc_analog.py (PCCAnalog, PCCConfig).
  Posterior-cingulate-analog metastability scalar in [0, 1] that modulates
  the SD-032a MECH-259 effective_threshold. High pcc_stability -> coordinator
  resists mode transitions; low stability (depleted / no recent rest /
  failing task outcomes) -> transitions happen at lower salience. Does NOT
  trigger mode switches directly (that is SD-032c's job). Non-trainable
  arithmetic; no gradient flow.
  Inputs (per select_action tick):
    drive_level (SD-012 fatigue, [0, 1])
    success_ema (EMA over caller-supplied task-outcome scalars, neutral 0.5
                 baseline; experiments opt in via agent.note_task_outcome())
    steps_since_offline (cross-episode counter; reset only by
                         note_offline_entry() called from
                         agent.enter_offline_mode())
  Computation:
    offline_recency = min(1.0, steps_since_offline / window)
    stability = baseline + success_weight * (success_ema - 0.5)
              - fatigue_weight * drive_level
              - offline_weight * offline_recency
    stability = clip_[0,1](stability)
  Config: REEConfig.use_pcc_analog (bool, default False).
    Sub-knobs: pcc_success_alpha (0.02, ~50-step EMA window),
    pcc_success_weight (0.5; centred contribution from success_ema),
    pcc_fatigue_weight (0.5; subtractive from drive_level),
    pcc_offline_recency_window (500 steps; saturation),
    pcc_offline_weight (0.3; subtractive from offline_recency),
    pcc_stability_baseline (0.5; additive baseline before clipping).
  Falsification signature (spec): ablating SD-032d makes the SalienceCoordinator
    effective_threshold insensitive to fatigue / time-since-offline. Agent
    over-commits to external_task without rest-driven relaxation. PCC-ON ->
    drive_level rises -> stability falls -> effective_threshold falls ->
    mode_switch_trigger rate rises under matched salience input.
  Data flow: select_action() -> pcc.tick(drive_level=sal_drive) ->
    salience.update_signal("pcc_stability", pcc.pcc_stability) BEFORE
    coordinator.tick() -> coordinator.effective_threshold modulated.
    enter_offline_mode() -> pcc.note_offline_entry() (single integration
    point shared by MECH-092 within-session quiescence and INV-049
    cross-session sleep). reset() -> pcc.reset() (per-episode; preserves
    cross-episode _steps_since_offline). agent.note_task_outcome(value) ->
    pcc.note_task_outcome(value) feeds success EMA.
  Backward compatible: use_pcc_analog=False by default. Existing experiments
    unaffected. note_task_outcome() is a no-op when pcc is None.
  Biological basis: Leech & Sharp 2013 ("Arousal, Balance, Breadth") --
    PCC tracks the global stability of the current cognitive set vs the
    need to broaden attentional sampling. Treated conservatively: a
    [0, 1] metastability index that biases the threshold for any mode
    change without committing to attention-partition geometry. Frankland
    & Bontempi 2005 systems-consolidation framing: stability falls with
    time-since-last-offline, biasing the system toward requesting offline
    consolidation when held externally too long.
  MECH-094: not applicable (waking arithmetic, no replay content authored).
  Phased training: not applicable (non-trainable arithmetic).
  Validation experiment: V3-EXQ-447 queued (PCC-OFF vs PCC-ON x rest /
    no-rest contrast; acceptance criterion: with PCC-ON and matched dACC
    salience injection, mode-switch trigger rate is monotone in
    drive_level and time-since-offline; PCC-OFF rate is invariant).
  See SD-032d, SD-032a, MECH-259, MECH-261, INV-049, MECH-092, SD-032 parent.
