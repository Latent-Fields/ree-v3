## SD-PP-1 Observation Reliability Estimator (2026-09-22)
- SD-PP-1: precision.observation_reliability -- IMPLEMENTED 2026-09-22.
  Evidence (sensory) precision producer: an organism-side estimate of how
  reliable the exteroceptive observation channel is right now, using ONLY the
  observation stream (`obs_world` handed to `agent.sense()`) and the encoder's
  own `z_world` output -- no environment internals, no labels, no future
  frames. Module: `ree_core/precision/observation_reliability.py`
  (`ObservationReliabilityConfig`, `ObservationReliabilityEstimator`).
  Statistic (consecutive exteroceptive frames o_{t-1}, o_t): d_t = |o_t -
  o_{t-1}| elementwise; sigma_hat_t = median_e(d_t) / mad_scale (mad_scale =
  0.954, since median|N(0,2s^2)| = 0.954 s); sigma_sq_ema <- (1-a_o)
  sigma_sq_ema + a_o sigma_hat_t^2 (a_o = obs_ema_alpha, first differenced
  frame initialises); sigma_obs_sq = max(sigma_sq_ema, sigma_floor^2);
  precision_obs = 1 / sigma_obs_sq. Encoder gain (expresses evidence
  precision in z units): kappa_t = mean_d((z_t - z_{t-1})^2) / mean_e((o_t -
  o_{t-1})^2), skipped when the denominator < 1e-12; kappa <- (1-a_k) kappa +
  a_k kappa_t (a_k = kappa_ema_alpha, initialised to the first kappa_t;
  ready=False until then, kappa reads 1.0 while not ready);
  evidence_variance_z = kappa * sigma_obs_sq; evidence_precision_z = 1 /
  (evidence_variance_z + eps). Measured 2026-09-22 (probe, seed 42, 300
  random-action steps, CausalGridWorldV2 world_state, 250 elements): clean
  median|d| = 0.0000 (p90 0.0000); sigma 0.03 -> recovered 0.0333; sigma 0.12
  -> recovered 0.1271 (clean frac_changed = 0.127, so the median sits at 0
  without noise and at ~sigma with it).
  Config: `use_observation_reliability` (bool, default False),
  `obs_ema_alpha` (float, default 0.2), `kappa_ema_alpha` (float, default
  0.05), `sigma_floor` (float, default 0.005), `mad_scale` (float, default
  0.954), `eps` (float, default 1e-9). Set `use_observation_reliability=True`
  to enable (integration wires `REEConfig.use_observation_reliability`, being
  done by the session).
  Data flow: `agent.sense()` obs_world -> `observe_obs` (before encode) ->
  z_world -> `observe_latent` (after encode, same tick) -> feeds
  `ReplayProvenancePacket` (SD-PP-3) via `evidence_variance_z` /
  `evidence_precision_z` / `sigma_obs` / `kappa` / `ready`.
  Backward compatible: disabled by default -- with the flag off no estimator
  is constructed, no call is made, no RNG is drawn.
  Biological basis: sensory precision estimation / precision-weighting of
  evidence (Friston active inference; Meyniel & Dehaene 2017 confidence-
  weighted updating).
  Phased training: no. `ObservationReliabilityEstimator` trains nothing; it is
  pure float/tensor arithmetic (median, mean, EMA) with no learned parameters
  and no RNG draws anywhere.
  Validation experiment: V3-EXQ-1073 (reserved).
  See MECH-572, MECH-016, ARC-055, MECH-043.
  LIMITATION: kappa is measured on real motion (the encoder's actual response
  to actual state change) and then applied uniformly to noise. An encoder
  that suppresses high-frequency jitter has a smaller true gain on noise than
  on motion, so this proxy OVER-estimates z-space noise and the downstream
  evidence_precision_z / consolidation gain is therefore conservative under
  noisy observation conditions. This is a named proxy, not a learned
  per-state sensory-precision head; that upgrade is registered separately.
