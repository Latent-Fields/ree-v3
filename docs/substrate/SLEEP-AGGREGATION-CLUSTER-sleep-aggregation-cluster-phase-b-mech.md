## Sleep Aggregation Cluster Phase B: MECH-285 SleepReplaySampler (2026-04-25)
- MECH-285: sleep.replay_sampler -- IMPLEMENTED 2026-04-25.
  Module: ree_core/sleep/replay_sampler.py (SleepReplaySampler).
  At SLEEP_ENTRY freezes StalenessAccumulator.snapshot(), then draws N seeds from
  AnchorSet.all_with_dual_trace() (active + inactive, Bouton 2004 dual-trace
  preserved) with softmax(staleness/temperature) priority. Stateless within cycle;
  uniform-fallback when no accumulator (mech285_allow_uniform_fallback=True default).
  Config: REEConfig.use_mech285_sampler (master, default False),
    mech285_draws_per_cycle (50), mech285_temperature (1.0),
    mech285_allow_uniform_fallback (True). All wired through from_dims.
  Agent wiring: REEAgent constructs sampler when master ON AND hippocampal.anchor_set
  exists (Phase B requires MECH-269 Phase 2 ii); accumulator optional.
  SleepLoopManager extended with replay_sampler + draws_per_cycle ctor args; _run_cycle
  enters SLEEP_ENTRY phase, freezes snapshot, runs draws, merges mech285_* diagnostics
  into SleepCycleState.last_metrics. Phase B is NO-OP CONSUMER -- draws land in metrics
  only (Phases C-E wire routing/aggregator/writeback).
  Added AnchorSet.all_with_dual_trace() alias.
  Validation: 10/10 new contract tests + 113/113 contracts + 7/7 preflight all PASS.
  Bit-identical OFF guarantee holds.
  See MECH-285, MECH-269, MECH-272, MECH-275, MECH-273.
