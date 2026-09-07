## MECH-286: Override-Gated Sleep Onset (2026-05-21)
- MECH-286: sleep.override_gated_state_transition -- IMPLEMENTED 2026-05-21.
  Module: ree_core/sleep/sleep_onset_gate.py (evaluate_sleep_onset_permit,
  SleepOnsetGateConfig). Wake-stability axis of SD-037: the same override_signal
  that gates drive->z_goal seeding also gates wake->offline transition in
  SleepLoopManager._run_cycle (before run_sleep_cycle / cycle_index advance).
  Joint permit (all required):
    override_signal < mech286_theta_sleep_permit
    AND max(MECH-284 region staleness snapshot) > mech286_theta_sleep_recruit
    AND z_harm_a.norm() < mech286_threat_tonic_threshold
  When blocked: episodes_since_sleep counter reset, cycle_index unchanged,
  last_metrics carry mech286_* diagnostics with mech286_sleep_permitted=0.
  Config: REEConfig.use_mech286_sleep_onset_gate (bool, default False).
    mech286_theta_sleep_permit (0.5), mech286_theta_sleep_recruit (0.3),
    mech286_threat_tonic_threshold (0.4).
  Backward compatible: flag OFF preserves deterministic K-episode sleep firing.
  Hyperarousal lesion test requires use_broadcast_override=True;
  staleness leg requires use_staleness_accumulator=True.
  MECH-094: gate evaluated only at episode boundary via notify_episode_end
  (waking state), not during replay/simulation ticks.
  No trainable parameters. No phased training.
  Validation experiment: V3-EXQ-599 queued (3-arm substrate diagnostic).
  See SD-037, MECH-284, MECH-285, MECH-272, MECH-281 (sibling motor axis).
