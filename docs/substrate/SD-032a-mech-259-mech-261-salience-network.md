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

- MECH-266: cingulate.asymmetric_per_mode_hysteresis -- IMPLEMENTED 2026-04-21.
  Module: ree_core/cingulate/salience_coordinator.py.
  Per-mode Schmitt-trigger rails on top of the MECH-259 symmetric
  switch_threshold. Two optional dict overrides on
  SalienceCoordinatorConfig:
    enter_thresholds[target_mode]: salience_aggregate required to enter
      target_mode (falls back to switch_threshold when unset).
    exit_thresholds[current_mode]: operating_mode[current_mode] must be
      strictly less than this value before a switch OUT of the current
      mode is permitted (falls back to 1.0 sentinel = always satisfied
      for any proper softmax, preserving legacy MECH-259 behaviour).
  MECH-266 trigger:
    trigger = (salience_aggregate > enter_threshold * stability_mult)
           AND (operating_mode[current_mode] < exit_threshold)
           AND (soft_argmax != current_mode)
  Over-binding / OCD axis: exit_thresholds[m] near 0 -> current mode
    must collapse to near-zero probability before leaving. Stuck-in-mode
    signature reproducible at exit=0.05.
  Under-binding / depression axis: set lower enter_threshold (e.g. 0.5)
    so salience clears entry rail more readily; exit left at 1.0 no-op.
  Symmetric baseline: empty dicts; trigger reduces to legacy MECH-259.
  Setters:
    set_enter_threshold(mode, value) -- per-mode enter rail.
    set_exit_threshold(mode, value)  -- per-mode exit rail.
    set_hysteresis_ratio(ratio)      -- uniform exit rail across all
      registered modes (EXP-0163 parametric sweep convenience).
  Tick return dict extended with enter_threshold, exit_threshold,
  current_mode_prob; effective_threshold retained as alias for
  enter_threshold (backward-compat diagnostic).
  Backward compatible: default SalienceCoordinatorConfig uses empty
  enter_thresholds / exit_thresholds dicts -- all existing experiments
  unaffected.
  Biological basis: Schmitt-trigger hysteresis is a canonical
  implementation of the per-mode asymmetric switch costs observed
  in task-switching paradigms (over-binding in OCD: hard to leave
  mode; under-binding in depression/ADHD axis: easy to flip). ocd4
  thought file row "competing goals" and "mode stickiness / Hold
  decay" derive from this substrate.
  MECH-094: not applicable (non-trainable arithmetic extension).
  Phased training: none (no parameters).
  Validation experiments: V3-EXQ-464 (EXP-0160 competing-goals, 5
    sub-tests, substrate-landing diagnostic) and V3-EXQ-467 (EXP-0163
    mode stickiness / hold decay, 5-arm parametric sweep r in
    [0.10, 0.50, 1.00, 1.50, 2.00]). Both smoke-PASS all sub-tests.
    Full behavioural competing-goals runs (switch-cost asymmetry,
    goal-completion dose-response) deferred to EXQ-464b / EXQ-467b
    when the CausalGridWorldV2 dual simultaneously active
    resource-cue extension lands.
  See MECH-266, SD-032a, MECH-259, SD-033 parent, REE_assembly
  evidence/planning/sd033_governance_plan.md, docs/thoughts/
  2026-04-20_ocd4.md.
