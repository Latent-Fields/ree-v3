## SD-SLEEP-ENTRY-PRESSURE (sleep_substrate:GAP-9 follow-up, V3-EXQ-933 fix): sleep.entry_pressure_time_integrating_trigger -- IMPLEMENTED (2026-08-26)
- SD-SLEEP-ENTRY-PRESSURE (sleep_substrate:GAP-9 follow-up, V3-EXQ-933 fix): sleep.
  entry_pressure_time_integrating_trigger -- IMPLEMENTED 2026-08-26. Fixes the two failure modes
  V3-EXQ-933 found in the GAP-9 need arm above: MELConsumer.need_crossed() thresholds
  current_mel(), a time-invariant MEAN built for GAP-5b's scale-free DURATION lever -- reused for
  ENTRY TIMING it never crossed under constant sub-threshold demand (NEED_SUB: 0/120 fires) and
  fired every step under supra-threshold demand with no refractory (NEED_HIGH: 120/120 fires).
  Module: ree_core/sleep/mel_consumer.py (new EntryPressureAccumulator class -- a running SUM,
  current_mel()'s MEAN left untouched; MELConsumer.entry_pressure_crossed() /
  current_entry_pressure(); note_step_pe() also accumulates into it when use_entry_pressure is
  on; on_cycle_complete()/reset() discharge it). ree_core/sleep/phase_manager.py
  (SleepLoopManager constructor gains within_life_entry_pressure_refractory_steps, validated >=1;
  notify_waking_step() gains a THIRD arm, pressure_crossed = entry_pressure_crossed() AND
  steps_since_sleep >= the refractory floor, OR'd with need_crossed/at_ceiling; arm-attribution
  priority need > pressure > ceiling).
  Config (REEConfig, ree_core/utils/config.py): use_entry_pressure (bool, default False; set True
  to enable), entry_pressure_gain (1.0), entry_pressure_threshold (0.0),
  within_life_entry_pressure_refractory_steps (int, default 2 -- 1 is degenerate and does NOT
  bound the fire rate, since steps_since_sleep already reads 1 on the very next step after a
  reset). All wired through REEConfig.from_dims() (silent-kwargs guard: dataclass field +
  from_dims signature param + from_dims body assignment + agent.py MELConsumerConfig/
  SleepLoopManager construction, 4 sites).
  Data flow: waking step -> note_step_pe(demand) [gated on use_entry_pressure] -> running SUM
  accumulates -> notify_waking_step() checks SUM*gain >= threshold AND steps_since_sleep >=
  refractory_floor -> fires -> on_cycle_complete() discharges the SUM to 0 (Process-S homeostatic
  reset), matching current_mel()'s accumulator reset. Emits within_life_trigger_arm_pressure /
  within_life_pressure_at_fire / within_life_pressure_threshold into the fired cycle's metrics +
  cycle_history, alongside the existing need/ceiling keys.
  Backward compatible: use_entry_pressure default False -> note_step_pe never accumulates into the
  new term (stays exactly 0.0, not merely unread) -> entry_pressure_crossed() always False ->
  byte-identical to the pre-SD-SLEEP-ENTRY-PRESSURE need+ceiling-arm-only trigger. need_crossed()
  / current_mel() / the GAP-5b duration lever are completely untouched. No trainable parameters /
  no new encoder head / no new latent field. No phased training needed.
  MECH-094: fires only on waking steps (not hypothesis_tag), reuses the existing
  _run_cycle/run_sleep_cycle path -> no new memory writes.
  Contracts: tests/contracts/test_sleep_within_life_trigger_gap9.py G15-G19 (19 tests total, all
  pass): G15 sub-threshold demand crosses in bounded time (Process-S fix), G16 supra-threshold
  demand is bounded by the refractory floor (fire-rate fix), G17 OFF is byte-identical even under
  huge injected demand, G18 the SUM discharges on cycle completion (a second sub-threshold window
  needs the same step count to re-cross), G19 arm-attribution priority (need > pressure > ceiling)
  + REEConfig default/from_dims round-trip. use_entry_pressure registered in
  tests/test_flag_inertness.py KNOWN_UNPROBED (same rationale as its sibling
  use_within_life_sleep_trigger -- behavioural inertness/effect already pinned by the contracts
  above).
  Validation experiment: V3-EXQ-933a (diagnostic; reproduces V3-EXQ-933's exact NEED_SUB/NEED_HIGH
  injected-demand levels and threshold against the new mechanism; run directly 2026-08-26 rather
  than through the runner queue since the manifest is the decisive readout and re-queuing it would
  duplicate compute per GOV-REUSE-1 -- PASS, 3 seeds, all of C1 sub-threshold-crosses-in-bounded-
  time / C2 supra-threshold-rate-bounded / C3 off-arm-inert satisfied; PROMOTES NOTHING).
  See REE_assembly/evidence/planning/substrate_queue.json SD-SLEEP-ENTRY-PRESSURE;
  REE_assembly/evidence/planning/sleep_substrate_plan.md (GAP-9); this file's own
  sleep_substrate:GAP-9 entry above (whose need arm's failure this fixes); SD-MEL-CONSUMER
  (GAP-5b, whose current_mel() duration statistic is left untouched).
