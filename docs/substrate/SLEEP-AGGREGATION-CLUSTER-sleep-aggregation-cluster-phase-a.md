## Sleep Aggregation Cluster Phase A: Scaffolding (2026-04-25)
- Sleep cluster Phase A: scaffold ree_core/sleep/ package -- IMPLEMENTED 2026-04-25.
  Module: ree_core/sleep/__init__.py, ree_core/sleep/phase_manager.py.
  New SleepPhase enum (6 phases: WAKING/SLEEP_ENTRY/SWS_ANALOG/PHASE_SWITCH/REM_ANALOG/
  WRITEBACK; only WAKING/SWS_ANALOG/REM_ANALOG visited in Phase A), SleepCycleState
  dataclass, and SleepLoopManager that wraps the existing SD-017 surface
  (REEAgent.run_sleep_cycle / enter_sws_mode / run_sws_schema_pass / enter_rem_mode /
  run_rem_attribution_pass / exit_sleep_mode -- pre-existing per SD-017).
  Master flag use_sleep_loop (default False) + sleep_loop_episodes_K (default 1) +
  sleep_loop_require_passes (default True) wired through REEConfig + REEConfig.from_dims().
  Manager instantiated in REEAgent.__init__ when flag is on; notify_episode_end() called
  at the start of REEAgent.reset() BEFORE per-episode resets so sleep operates on the
  final waking state.
  Validation: 8/8 new contract tests PASS (test_sleep_phase_a_scaffolding.py covering
  import, default backward-compat, master-OFF no instantiation, K=1 cycle drive,
  K=3 fires-on-third, no-substrate refusal, force_cycle, phase returns to WAKING).
  Full suite: 103/103 contracts + 7/7 preflight PASS -- bit-identical OFF guarantee
  holds. Phase A is no-op-consumer scaffolding only; Phases B-E layer additional
  master flags on top.
  See SD-017, MECH-272, MECH-273, MECH-275, MECH-285.
  Design doc: REE_assembly/docs/architecture/sleep_aggregation_cluster.md
