## Sleep Aggregation Cluster Phase C: MECH-272 RoutingGate (2026-04-25)
- MECH-272: sleep.routing_gate -- IMPLEMENTED 2026-04-25.
  Module: ree_core/sleep/routing_gate.py (RoutingGate, RoutedEvent).
  State-conditioned channel weights {anchor_channel, probe_channel} that flip across
  SWS_ANALOG / REM_ANALOG / WAKING rows per the design-doc table.
  Config: REEConfig.use_mech272_routing (master, default False) + 6 sub-knobs:
    sws_anchor_weight, sws_probe_weight, rem_anchor_weight, rem_probe_weight,
    waking_anchor_weight, waking_probe_weight.
  Wired into SleepLoopManager: set weights at SLEEP_ENTRY (SWS row), at PHASE_SWITCH
  (REM row); call route() on each replay draw and surface routed counts as mech272_*
  diagnostics on SleepCycleState.last_metrics.
  Wired flag through REEAgent constructor. No downstream consumer wiring yet
  (HippocampalRouter / E1 ContextMemory consumer / aggregator land in Phases D-E).
  Validation: bit-identical waking with all flags OFF; weights flip across phases when
  ON; backward-compat with use_mech285_sampler ON + use_mech272_routing OFF preserved.
  Result: 10/10 Phase C contracts PASS, 7/7 preflight PASS, 123/123 full contracts PASS.
  See MECH-272, MECH-285, MECH-275, MECH-273, MECH-094 (mode-conditioning generalisation).
