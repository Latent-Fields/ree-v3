## MECH-266: cingulate.asymmetric_per_mode_hysteresis -- IMPLEMENTED (2026-04-21)
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
