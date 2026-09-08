## SD-032a / MECH-259 / MECH-261: Salience-Network Coordinator (2026-04-19)
- SD-032a: cingulate.salience_network_coordinator -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/salience_coordinator.py
  (SalienceCoordinator, SalienceCoordinatorConfig, DEFAULT_MODE_NAMES,
  DEFAULT_GATE_WEIGHTS). Network-level coordinator that aggregates the
  SD-032b dACC bundle and homeostatic / offline signals into a soft
  operating-mode probability vector and a discrete MECH-259 mode-switch
  trigger. Hosts the MECH-261 dict-keyed write-gate registry.
  Inputs (live in V3): dACC bundle (pe / foraging_value /
  choice_difficulty), drive_level (SD-012; proxy SD-032c), agent
  offline-mode flag (proxy SD-032d). Registered slots aic_salience /
  pcc_stability / pacc_autonomic accept update_signal calls and remain
  no-op until SD-032c/d/e land.
  Outputs: operating_mode dict[str, float] (softmax over per-mode
  affinity logits, default biased to external_task waking baseline);
  current_mode str (Schmitt-trigger hysteresis -- updates only on
  threshold crossing); mode_switch_trigger bool (fires when
  salience_aggregate > switch_threshold * (1 + stability_scaling *
  pcc_stability) AND argmax(operating_mode) != current_mode);
  write_gate(target) float (soft-weighted sum over mode probs).
  MECH-261 default registry covers sd_033a, sd_033b, sd_033c, sd_033d,
  hc_viability, sensory_buffer, autonomic, e3_policy with the per-mode
  weights from the spec table. mode_names is a list, register_target
  accepts arbitrary mode keys -- V4 parallel_goal_deliberation
  (SD-033e) can be added without schema changes.
  Config: REEConfig.use_salience_coordinator (bool, default False).
  Sub-knobs: salience_switch_threshold (1.0), salience_stability_scaling
  (1.0), salience_softmax_temperature (1.0),
  salience_external_task_bias (1.0), salience_dacc_pe_weight (1.0),
  salience_dacc_foraging_weight (0.5), salience_apply_to_dacc_bias
  (False -- when True, scales dACC score_bias by the e3_policy gate so
  internal_replay attenuates dACC influence on action selection).
  Data flow: select_action() builds dACC bundle -> coordinator.tick()
  consumes bundle + drive_level + e1._offline_mode -> caches operating_mode
  + trigger -> optional scale of dacc_score_bias by write_gate("e3_policy")
  -> e3.select() unchanged path.
  Backward compatible: use_salience_coordinator=False by default. Existing
  experiments unaffected. DACCtoE3Adapter is RETAINED as the score_bias
  source until SD-033 substrates consume operating_mode natively (staged
  removal -- adapter shim is now optionally gated rather than fully
  replaced this PR).
  Biological basis: Menon & Uddin 2010 (AIC-dACC salience network);
  Craig 2009 (AIC interoceptive-salience hub); Carr/Jadhav/Frank 2011
  (soft-boundary write subpopulations during awake SWRs); Tambini &
  Davachi 2019 (cross-state persistence, forward propagation bias).
  MECH-094: not authored here -- coordinator emits the gate that MECH-094
  generalises to. Phased training: not applicable (non-trainable
  arithmetic).
  Validation experiment: V3-EXQ-446 queued (coordinator-OFF vs
  coordinator-ON, plus synthetic high-PE injection to confirm trigger
  fires; verifies write_gate values in [0, 1] across 8 default targets).
  See SD-032a, MECH-259, MECH-261, SD-032 parent.
