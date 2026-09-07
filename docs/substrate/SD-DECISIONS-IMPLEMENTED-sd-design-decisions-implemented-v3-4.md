## SD Design Decisions Implemented (V3) — continued
- SD-014: hippocampus.valence_vector_node_recording — IMPLEMENTED 2026-04-04.
  4-component valence vector V=[wanting, liking, harm_discriminative, surprise] added to
  RBFLayer and ResidueField (ree_core/residue/field.py). Each RBF center now stores a
  valence_vecs buffer [num_centers, 4] updated incrementally per visit.
  New methods: RBFLayer.evaluate_valence(z) -> [batch, 4]; ResidueField.update_valence(),
  evaluate_valence(), get_valence_priority(z_world, drive_state). VALENCE_WANTING=0,
  VALENCE_LIKING=1, VALENCE_HARM_DISCRIMINATIVE=2, VALENCE_SURPRISE=3 constants defined
  at module level. ResidueConfig.valence_enabled (default True; set False for ablation).
  MECH-094 gate applies: hypothesis_tag=True blocks valence updates. Prerequisite for
  ARC-036 (multidimensional valence map) and replay prioritisation via drive state.
  Write paths (2026-04-17):
    VALENCE_WANTING (0): update_benefit_salience() [serotonin salience] and
      update_schema_wanting() [E1 schema readout]. Both enabled when tonic_5ht_enabled or
      schema_wanting_enabled respectively.
    VALENCE_LIKING (1): update_liking(benefit_exposure) -- NEW 2026-04-17.
      Call from experiment loop at resource contact (benefit_exposure >= liking_threshold).
      Berridge hedonic impact at consummation (opioid-mediated). Enabled by
      valence_liking_enabled=True in REEConfig.from_dims().
    VALENCE_HARM_DISCRIMINATIVE (2): TWO independent write paths, neither is a
      replacement for the other:
      (a) written automatically in sense() after SD-021 descending modulation --
          NEW 2026-04-17. Post-attenuation z_harm.norm() written at current
          z_world node, every step. Committed-state nodes get stale (attenuated)
          h, creating the analgesia-as-underestimated-h signature for
          SD-021/SD-014 cross-connection. Enabled by valence_harm_enabled=True
          in REEConfig.from_dims(). NOT tonic-5HT-modulated, not symmetric with
          benefit_salience's calibration.
      (b) update_harm_salience(harm_exposure) -- NEW 2026-08-02 (MECH-203 SR-2
          harm-symmetric gap fix). Mirrors update_benefit_salience() exactly:
          writes SerotoninModule.harm_salience() = (1 - tonic_5ht) * harm_exposure
          into VALENCE_HARM_DISCRIMINATIVE via the same
          ResidueField.update_valence() nearest-active-center path
          update_benefit_salience() uses. Fixes the gap where no calibrated,
          tonic-5HT-modulated harm-salience write existed at all --
          update_residue()'s accumulate() path only ever wrote the legacy
          scalar `weights` (residue density), never valence_vecs. Gated by the
          SAME tonic_5ht_enabled master switch as update_benefit_salience()
          (no new config flag). `harm_exposure` must be the EMA'd nociceptive
          exposure convention (CausalGridWorldV2.harm_exposure / body_obs[10],
          symmetric to benefit_exposure / body_obs[11]) -- NOT raw
          harm_signal. Feeding raw |harm_signal| (~0.05-0.13, a ~13-20x larger
          scale than benefit_salience's own ~0.0037-0.0066 output) into the
          shared-capacity RBF field via a naive direct update_valence() call
          was tried and reverted: it swamped the field to a 100% harm / 0%
          benefit degenerate split, because both channels write to the SAME
          nearest-ACTIVE-center pool (see ResidueField.update_valence()).
          The calibrated EMA-scale convention keeps both channels coexisting.
          Contracts: tests/contracts/test_mech203_harm_salience_writepath.py
          (6/6 PASS, including a swamping-regression fixture that pins the
          naive-fix failure mode and a fixture proving the calibrated fix
          avoids it).
    VALENCE_SURPRISE (3): written in update_residue() when MECH-205 surprise_gated_replay
      is active. PE-EMA delta written when magnitude exceeds pe_surprise_threshold.
  All four components now have active write paths. Config flags all default False (backward compat).
  VALENCE_WANTING incentive-sensitization DECOUPLE -- IMPLEMENTED 2026-08-07 (V3-EXQ-887 fix).
    V3-EXQ-887 (2026-08-04) gave SD-014 its first genuine FAIL: |Spearman(wanting, liking)|
    = 0.93-0.97 (C2 requires <= 0.90). Root cause (read from source, autopsied
    failure_autopsy_2026-08-05 #3): VALENCE_WANTING was written as
    serotonin.benefit_salience(benefit_exposure) = tonic_5ht * benefit_exposure while
    VALENCE_LIKING is written as raw benefit_exposure -- both monotone transforms of ONE
    shared input, so rank collinearity is near-guaranteed. That is not the independent
    dopamine(wanting)/opioid(liking) architecture the claim is grounded in.
    Fix: a per-node, drive-coupled, saturating incentive-sensitization gain g_i amplifies
    ONLY the wanting write (Smith/Berridge/Aldridge 2011: DA sensitization raises wanting,
    not liking). On each qualifying write at nearest active center i:
      g_i  <- min(sensitization_max, g_i + sensitization_rate * drive_level)
      w_i  += benefit_salience * (1 + sensitization_coupling * g_i)
    drive_level = REEAgent.compute_drive_level(body_obs) (SD-012, = 1 - energy), a signal
    ORTHOGONAL to the benefit magnitude VALENCE_LIKING reads -- so wanting diverges from
    raw hedonic magnitude over repeated exposure (the incentive-trap w >> l signature).
    New RBFLayer buffer sensitization_gain [num_centers] + RBFLayer.update_sensitization_gain();
    ResidueField._nearest_active_center() (shared helper) + ResidueField.update_wanting_sensitized();
    REEAgent.update_benefit_salience() gains an optional drive_level=0.0 arg and routes
    through the sensitized path when incentive_sensitization_enabled.
    Config (REEConfig / from_dims): incentive_sensitization_enabled (default False -> no-op),
      sensitization_rate (0.05), sensitization_max (4.0), sensitization_coupling (1.0).
    Backward compatible: disabled by default -- update_benefit_salience() is bit-identical,
      the gain buffer stays zero and is never consulted, and drive_level is ignored (smoke
      test: OFF-path wanting == benefit_salience exactly, gain sum 0.0). Also inert if
      enabled but drive_level never supplied (legacy drivers).
    Phased training: NOT required (gain is a no_grad buffer accumulator, not a trained head).
    MECH-094: honours hypothesis_tag exactly as update_valence (waking-only write).
    Validation experiment: V3-EXQ-887a queued (SD-014 representational-separability retest
      with the feature enabled; reuses V3-EXQ-887's validated wall-independent instrument).
    See REE_assembly/docs/architecture/sd_014_wanting_liking_decouple.md, claims.yaml SD-014.

- MECH-203 + MECH-204: neuromodulation.serotonergic_sleep_substrate — IMPLEMENTED 2026-04-07.
  SerotoninModule (ree_core/neuromodulation/serotonin.py) with SerotoninConfig.
  SR-1: tonic_5ht [0,1] state variable. Waking: rises on benefit, decays to baseline,
  suppressed by z_harm_a. SWS: held at waking level. REM: drops to 0 (dorsal raphe quiescence).
  SR-2: benefit_salience = tonic_5ht * benefit_exposure. Tags SD-014 VALENCE_WANTING for
  balanced replay prioritisation. SR-3: _precision_at_rem_entry captured on enter_rem().
  Dynamic GoalConfig modulation: z_goal_seeding_gain and valence_wanting_floor modulated
  by tonic_5ht each step. Agent methods: serotonin_step(), update_benefit_salience(),
  enter_sws_mode(), enter_rem_mode(), exit_sleep_mode(). HippocampalModule.replay() accepts
  optional drive_state for valence-weighted start selection. Master switch:
  tonic_5ht_enabled=False (default, fully backward compatible).
  SR-2 harm-symmetric AMEND (2026-08-02, MECH-203 gap fix): harm_salience(harm_exposure)
  = (1 - tonic_5ht) * harm_exposure and agent method update_harm_salience() added --
  see SD-014 VALENCE_HARM_DISCRIMINATIVE write-path (b) above for the full rationale
  (no calibrated harm-salience write existed prior to this; a naive raw-harm_signal
  write was tried and swamped the shared RBF field to 100% harm / 0% benefit).
  Gated by the same tonic_5ht_enabled switch, no new flag. Contracts:
  tests/contracts/test_mech203_harm_salience_writepath.py 6/6 PASS.
  MECH-204 GAP-1 consumer (2026-05-08): SleepLoopManager WRITEBACK (phase_manager.py)
  calls SerotoninModule.compute_recalibration_target() and
  E3TrajectorySelector.recalibrate_precision_to(target, step). Config:
  REEConfig.use_rem_precision_recalibration (default False);
  rem_precision_recalibration_step default 0.25 (post V3-EXQ-541c PASS 2026-05-09).
  NOT bundled in use_sleep_aggregation_cluster (separate GAP-1 flag).
  Contracts: tests/contracts/test_mech204_precision_recalibration.py 13/13 PASS.
  Canonical validation: V3-EXQ-541c. Integration closure: V3-EXQ-602 queued.

- ARC-028 + MECH-105: control_plane.hippocampal_betagate_coupling — IMPLEMENTED 2026-04-04.
  HippocampalModule.compute_completion_signal(trajectories) -> float: scores all proposed
  trajectories via _score_trajectory(), maps best score to sigmoid dopamine-analog value
  in [0.5, 1.0). Caches as self._last_completion_signal.
  BetaGate.receive_hippocampal_completion(signal) -> bool: if beta elevated and signal >=
  completion_release_threshold (default 0.75), calls self.release() and returns True.
  Implements Lisman & Grace 2005 subiculum->NAc->VP->VTA loop: high hippocampal completion
  quality -> dopamine signal -> beta drops -> E3 state propagates to action selection.
  get_state() and reset() updated. Return type of propose_trajectories() unchanged.

- MECH-290: hippocampal.backward_trajectory_credit_sweep -- IMPLEMENTED 2026-04-24.
  Module: ree_core/hippocampal/module.py (HippocampalModule.record_committed_trajectory,
  HippocampalModule.backward_credit_sweep, HippocampalModule.reset_committed_trajectory).
  Biological basis: Foster & Wilson 2006 (Nature) -- reverse replay fires at reward
  endpoint during waking, concurrent with dopamine. Credit propagates backward from goal
  to trajectory start.
  Two new methods:
    record_committed_trajectory(trajectory): called at BetaGate elevation (commit entry
      in select_action()), stores a detached copy of the committed trajectory in
      _committed_trajectory_buffer. Distinct from _exploration_buffer (MECH-165
      quiescent replay source): this stores EXECUTED trajectory, not CEM proposals.
    backward_credit_sweep(outcome_quality): called when BetaGate releases via
      receive_hippocampal_completion() in _e3_tick(). Sweeps committed trajectory
      backward; at each z_world state t: credit = outcome_quality * gamma^(T-1-t);
      ResidueField.update_valence(z_world_t, VALENCE_WANTING, credit) called.
      Returns dict: n_steps_swept, mean_credit, outcome_quality.
      No-op when outcome_quality < backward_sweep_min_quality (default 0.6).
    reset_committed_trajectory(): called from agent.reset() on episode boundary.
  Config: HippocampalConfig.use_backward_credit_sweep (bool, default False),
    backward_sweep_gamma (float, default 0.9), backward_sweep_min_quality (float, 0.6).
    All wired through REEConfig.from_dims().
  Agent wiring:
    _e3_tick(): receive_hippocampal_completion() return value captured as `released`;
      when True and flag is on, hippocampal.backward_credit_sweep(
      hippocampal._last_completion_signal) is called.
    select_action(): at bistable commit entry AND legacy non-bistable new-commit:
      hippocampal.record_committed_trajectory(e3._committed_trajectory) called.
    reset(): hippocampal.reset_committed_trajectory() called when flag on.
  No SD-006 dependency: fires synchronously on waking path.
  MECH-094: waking path (hypothesis_tag=False) -- credit from real executed trajectory.
  Requires ResidueConfig.valence_enabled=True to write VALENCE_WANTING; silently skips
  valence write if disabled (backward compat).
  Backward compatible: use_backward_credit_sweep=False by default; all methods are no-ops.
  Smoke: C1-C7 PASS (buffer management, sweep arithmetic, flag OFF no-op, valence write).
  End-to-end: agent boot + direct wiring test PASS 2026-04-24.
  Validation experiment: to be queued post-476a (SD-038 anti-recency sequenced after).
  See MECH-290, ARC-028, MECH-105, SD-014 (VALENCE_WANTING write paths), MECH-165.

- MECH-217: goal.replay_wanting_spread -- IMPLEMENTED 2026-07-30.
  Offline (SWS/REM) complement to MECH-290 (waking): spreads VALENCE_WANTING
  backward along a previously-traversed approach path during the SD-017 REM
  pass, instead of at waking BetaGate release.
  Module: ree_core/hippocampal/module.py
    HippocampalModule.spread_reverse_replay_wanting(trajectory) -> dict.
    Called from REEAgent.run_rem_attribution_pass() (agent.py) for each
    MECH-165 reverse-replayed trajectory in the reverse branch (only when
    the exploration buffer has stored trajectories -- the branch that
    already exists at agent.py's reverse-replay pass, ~10222-10251).
  Biological basis: Foster & Wilson 2006 (Nature) -- reverse replay during
  sleep/quiet rest reactivates the reverse of a real traversed path.
  Data flow: HippocampalModule._exploration_buffer (real recorded waking
    trajectory, MECH-165) -> reverse_replay() (pure temporal reversal,
    no new E2 rollout) -> spread_reverse_replay_wanting() reads the CURRENT
    VALENCE_WANTING at world_states[0] (the terminus -- the reversed
    sequence's first element is the original episode's most recent state)
    via ResidueField.evaluate_valence(), then writes at each EARLIER
    waypoint (steps_from_terminus >= 1):
      spread_t = gain * wanting_at_terminus * gamma^steps_from_terminus
    via ResidueField.update_valence(z_w, VALENCE_WANTING, spread_t,
    hypothesis_tag=trajectory.hypothesis_tag).
  Deliberately does NOT write at the terminus itself (steps_from_terminus=0):
    it is the read source; self-writing it would create a
    v <- v*(1+gain) feedback loop compounding across repeated REM cycles.
    Verified by smoke test: terminus value bit-for-bit unchanged after a
    pass that wrote 25 waypoint updates derived from it.
  No-op (returns {}) when wanting_at_terminus <= 0 -- nothing yet learned at
    that endpoint to spread; naturally gates on whether the trajectory
    actually ended somewhere valuable, no separate "was this a resource
    contact" bookkeeping needed.
  Scope decision: the write is NOT placed inside the shared
    HippocampalModule.reverse_replay()/diverse_replay() methods (as the
    claims.yaml note originally suggested), because those methods are also
    called from REEAgent._do_replay() (MECH-092 quiescent WAKING replay,
    agent.py ~8580) whenever mode="auto" rolls "reverse" -- that path is
    explicitly documented as "All replay content carries hypothesis_tag=True
    -- cannot produce residue." Doing the write inside the shared method
    would leak the offline-only effect into ordinary waking quiescent
    cycles. The call is made only from run_rem_attribution_pass(), so it
    fires strictly on the SD-017 REM pass.
  MECH-094: trajectory.hypothesis_tag is False here (reverse_replay() does
    not set it True) -- this is real recorded waking experience, reverse-
    ordered, not simulated/imagined content, so the write does not violate
    the MECH-094 gate (ResidueField.update_valence enforces the gate itself
    via the hypothesis_tag argument passed through).
  Config (HippocampalConfig, ree_core/utils/config.py):
    use_offline_wanting_spread (bool, default False) -- master switch.
    offline_wanting_spread_gamma (float, default 0.9) -- per-waypoint decay,
      matches the MECH-217 claim text.
    offline_wanting_spread_gain (float, default 0.1) -- NOT part of the
      MECH-217 claim text; an added numerical-stability guard (bounded
      EMA/TD-style step size) so the write is a bounded fraction of the
      terminus value rather than a 1:1 copy, further damping compounding
      across overlapping paths visited in repeated sleep cycles.
    All wired through REEConfig.from_dims().
  Telemetry: run_rem_attribution_pass() returns two new metrics keys,
    rem_wanting_spread_n_steps and rem_wanting_spread_mean (both 0.0 when
    the flag is off).
  Backward compatible: use_offline_wanting_spread=False by default; all
    methods are no-ops; existing experiments unaffected.
  No trainable parameters. No phased training needed.
  Smoke: flag-off no-op verified (rem_wanting_spread_n_steps=0); flag-on
    verified an earlier waypoint received nonzero VALENCE_WANTING while the
    terminus itself stayed exactly unchanged (no feedback loop). Full
    tests/test_flag_inertness.py + sleep-phase + MECH-293 contract suite
    (94 tests) PASS on ree-cloud-1 with this change applied.
  Validation experiment: EXP-0301/EVB-0349 queued (see /queue-experiment).
  See MECH-290 (waking analog), MECH-165 (reverse_replay substrate),
  SD-014 (VALENCE_WANTING write paths), SD-017 (REM pass host), MECH-216
  (waking schema-conditioned wanting, the sibling mechanism), ARC-051
  (multi-level wanting hierarchy consumer), MECH-094 (hypothesis_tag gate).
