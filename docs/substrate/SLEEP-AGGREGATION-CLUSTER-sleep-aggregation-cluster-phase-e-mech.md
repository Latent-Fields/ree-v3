## Sleep Aggregation Cluster Phase E: MECH-273 SelfModelAggregator (2026-04-25)
- MECH-273: sleep.self_model_writeback -- IMPLEMENTED 2026-04-25.
  Module: ree_core/sleep/self_model_aggregator.py (SelfModelAggregator,
  SelfModelAggregatorConfig). Subclass of MECH-275 BayesianAggregator specialised on
  SD-003 causal_sig posterior. offline_gradient_pass(e2_harm_s, replayed_regions,
  n_steps, domain='self', use_snapshot=True) reads posterior means from last_snapshot
  (SWS-only frozen copy at PHASE_SWITCH) when available; constructs synthetic
  (z_harm_s zeros, action one-hot round-robin) batch at E2_harm_s input dims; trains
  via Adam at waking_lr * offline_lr_scale for n_steps bounded MSE steps.
  MECH-094 exception scoped: optimiser constructed locally over e2_harm_s.parameters()
  only -- no other module's params touched. n_steps<=0 short-circuits to no-op; empty
  replayed_regions returns zero-loss diagnostics. Cumulative diagnostics
  (mech273_n_offline_passes/steps/sum_loss/last_offline_loss/n_offline_regions_consumed)
  and per-call (mech273_writeback_regions/n_steps/sum_loss/mean_loss).
  NEW API: StalenessAccumulator.partial_decay(replayed_regions, decay_factor=0.5) ->
  int multiplicatively decays only the supplied region keys (clamped [0,1], drops
  below drop_epsilon, dedupes input via 'seen' set).
  Config: REEConfig.use_mech273_self_model (master, default False) + 3 sub-knobs:
    mech273_offline_lr_scale (0.1), mech273_offline_n_steps (100),
    mech273_partial_decay_factor (0.5). All wired through from_dims.
  REEAgent.__init__: agent-level e2_harm_s construction (parallel to e2_harm_a) when
  config.latent.use_e2_harm_s_forward; sleep_self_model_aggregator instantiated when
  use_mech273_self_model AND e2_harm_s exist; passed to SleepLoopManager via 4 new
  ctor args (self_model_aggregator, self_model_offline_n_steps,
  self_model_partial_decay_factor, self_model_domain).
  SleepLoopManager._run_cycle: replayed_regions set accumulated during SWS+REM update
  loops via _extract_region_key helper (handles RoutedEvent.event.key tuple form and
  direct tuple form); AFTER agent.run_sleep_cycle() set phase WRITEBACK ->
  offline_gradient_pass(use_snapshot=True) -> staleness.partial_decay(replayed_regions,
  decay_factor=self_model_partial_decay_factor); writeback_metrics merged into
  SleepCycleState.last_metrics including mech273_partial_decay_n_regions and
  mech273_partial_decay_factor.
  SHY normalisation (MECH-120) explicitly out of V3 scope.
  Validation: 10/10 Phase E contracts + 150/150 (143 contracts + 7 preflight) all PASS.
  Bit-identical OFF guarantee holds.
  See MECH-273, MECH-275, MECH-272, MECH-285, MECH-284, MECH-094, SD-003, ARC-033.
