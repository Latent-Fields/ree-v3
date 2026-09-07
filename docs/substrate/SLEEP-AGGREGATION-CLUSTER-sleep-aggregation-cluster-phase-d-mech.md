## Sleep Aggregation Cluster Phase D: MECH-275 BayesianAggregator (2026-04-25)
- MECH-275: sleep.bayesian_aggregator -- IMPLEMENTED 2026-04-25.
  Module: ree_core/sleep/bayesian_aggregator.py (BayesianAggregator,
  GaussianPosterior, PosteriorUpdate, BayesianAggregatorConfig).
  Per-domain per-region Gaussian posteriors over residuals; conjugate mean-and-variance
  update gated by RoutedEvent.probe_channel * probe_gain (probe<=0 skipped, counted as
  mech275_n_skipped_zero_probe); snapshot+decay contract (snapshot deep-copies live
  posteriors, decay_factor multiplies live variance per cycle); place-domain default
  with (scale, segment_id) region key matching MECH-284.
  Config: REEConfig.use_mech275_aggregator (master, default False) + 6 sub-knobs:
    mech275_domains, mech275_prior_mean, mech275_prior_variance,
    mech275_likelihood_variance, mech275_decay_factor, mech275_probe_gain.
  Wired into SleepLoopManager._run_cycle: SLEEP_ENTRY freezes evidence_snapshot from
  agent.hippocampal.staleness_accumulator.snapshot() (place-domain evidence = staleness
  scalar at routed anchor's region, falls back to 0.0 if absent); each routed draw in
  SWS pass calls bayesian_aggregator.update(routed, evidence, domain=aggregator_domain);
  at PHASE_SWITCH snapshot() fires BEFORE routing_gate.set_phase(REM_ANALOG) so the
  snapshot captures SWS-only posteriors (Phase E reads this); REM re-route loop applies
  same probe-channel-gated update; mech275_* metrics merged into
  SleepCycleState.last_metrics.
  REEAgent.__init__ extended with Phase D conditional construction block;
  SleepLoopManager extended with bayesian_aggregator+aggregator_domain ctor args.
  NO downstream writeback (Phase E / MECH-273 deferred until next pass).
  Validation: 10/10 new contract tests + 38/38 sleep phases A-D + 133/133 contracts +
  7/7 preflight all PASS. Bit-identical OFF guarantee holds. MECH-094 enforced via
  call-site scoping (aggregator only invoked from _run_cycle, never from waking path).
  See MECH-275, MECH-272, MECH-285, MECH-284, MECH-094.
