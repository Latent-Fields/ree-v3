## SD-MEL-CONSUMER: sleep.adaptive_mel_sleep_cadence -- IMPLEMENTED (2026-07-07)
- SD-MEL-CONSUMER: sleep.adaptive_mel_sleep_cadence -- IMPLEMENTED 2026-07-07.
  Adaptive sleep-cadence MEL consumer (sleep_substrate:GAP-5b). The INV-050 THIRD /
  learning-demand sleep drive: reads accumulated waking Model Error Load (MEL = mean
  per-step e3 prediction error over the wake window, the same signal V3-EXQ-701c
  demonstrated is measurable + monotone in graded novelty) and modulates the offline
  (sleep) phase, REPLACING the K-episode-deterministic scheduler.
  Module: ree_core/sleep/mel_consumer.py (MELConsumer, MELConsumerConfig,
  WakingMELAccumulator).
  Config (REEConfig, ree_core/utils/config.py): use_mel_consumer (bool, default False;
  set True to enable). Sub-knobs: mel_gain (1.0), mel_reference (0.0 = auto to first
  cycle; validation sets ~2e-5), mel_reference_mode ("fixed"|"ema"), mel_ema_alpha (0.1),
  mel_duration_factor_min/max (0.5/3.0), mel_relative_floor (1e-6 -- recalibrated DOWN
  from the 701c-inherited ABS_MEL_FLOOR=1e-4, which was ~5x the converged-base signal),
  mel_scale_sws/mel_scale_rem (True), use_mel_entry (False), mel_entry_threshold (0.0).
  All wired through REEConfig.from_dims().
  Data flow: waking step -> agent.update_residue() -> e3.post_action_update()
  -> e3_prediction_error -> MELConsumer.note_step_pe() (waking-only, hypothesis_tag=False)
  -> [episode end via SleepLoopManager] duration_factor = clamp(1 + mel_gain*(mel/ref - 1),
  min, max) -> scales sws_consolidation_steps / rem_attribution_steps for the cycle
  -> more/fewer sws_n_writes + rem_n_rollouts (the exact V3-EXQ-677 pinned DV, now
  MEL-driven). Secondary entry lever: use_mel_entry fires a cycle when accumulated MEL
  crosses mel_entry_threshold (K-episode counter as safety backstop).
  IMPORTANT driver note: the consumer engages ONLY through the SleepLoopManager path
  (notify_episode_end / force_cycle) and MEL accumulates ONLY when the driver calls
  agent.update_residue() each waking step. A driver that calls agent.run_sleep_cycle()
  DIRECTLY (e.g. V3-EXQ-677) bypasses the consumer -- validation drivers must use the
  manager path + update_residue.
  Backward compatible: use_mel_consumer default False -> agent.mel_consumer is None,
  update_residue skips accumulation, SleepLoopManager reads unmodified config
  -> byte-identical K-episode-deterministic scheduler. Full contracts + preflight PASS.
  No trainable parameters / no new encoder head / no new latent field. No phased training
  needed (validation still needs a converged base so PE is at converged scale, per 701c).
  MECH-094: reads waking PE only; writes nothing to memory during non-waking states.
  DISTINCT from the SD-037 arousal entry gate (use_mech286_sleep_onset_gate /
  sleep_onset_gate.py / sleep_substrate:GAP-5, V4) -- orthogonal signal (arousal vs MEL);
  they compose.
  Unblocks: INV-050 (retest), MECH-180 (v3_pending). Validation experiment: V3-EXQ-718
  queued 2026-07-07 (v3_exq_718_sdmelconsumer_adaptive_cadence_validation; diagnostic;
  recon-only converged base + 4 graded-novelty arms consumer-ON + 1 matched-novelty
  consumer-OFF control; DV = cumulative_sws_writes + cumulative_rem_rollouts +
  mel_duration_factor; PROMOTES NOTHING until it scores).
  See REE_assembly/docs/architecture/sd_mel_consumer.md; SD-017; INV-050; MECH-180;
  plan-of-record REE_assembly/evidence/planning/sleep_substrate_plan.md (GAP-5b).
