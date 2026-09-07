## SD-017: Minimal Sleep-Phase Infrastructure -- SWS/REM Passes (2026-04-09)
- SD-017: sleep_phase.minimal_sleep_infrastructure_v3 -- SWS-ANALOG + REM-ANALOG IMPLEMENTED 2026-04-09.
  Two new first-class methods added to REEAgent (ree_core/agent.py):
  (1) run_sws_schema_pass(): SWS-analog schema installation (hippocampus-to-cortex direction).
      Samples diverse z_world prototypes from _world_experience_buffer (stratified across
      buffer history), constructs [z_self, z_world] E1 input, writes to ContextMemory
      bypassing the offline gate (offline gate blocks waking obs; schema writes are
      intentional offline content). Returns: sws_n_writes, sws_slot_diversity (mean pairwise
      cosine distance of ContextMemory slots -- higher = more differentiated), sws_buffer_size.
  (2) run_rem_attribution_pass(): REM-analog attribution replay (slot-filling, MECH-166).
      Replays recent theta_buffer content via hippocampal.replay() (forward) and
      hippocampal.diverse_replay(mode="reverse") (reverse/ARC-045 bidirectional proxy).
      Evaluates residue terrain per trajectory without accumulating new residue
      (hypothesis_tag=True per MECH-094). Returns: rem_n_rollouts, rem_mean_harm_terrain,
      rem_terrain_variance, rem_n_reverse.
  (3) run_sleep_cycle(): Convenience method running SWS then REM in sequence with correct
      mode transitions (enter_sws_mode -> run_sws_schema_pass -> exit_sleep_mode ->
      enter_rem_mode -> run_rem_attribution_pass -> exit_sleep_mode). Returns merged metrics.
  Config (REEConfig, ree_core/utils/config.py):
      sws_enabled (bool, default False), sws_consolidation_steps (int, default 5),
      sws_schema_weight (float, default 0.1), rem_enabled (bool, default False),
      rem_attribution_steps (int, default 10). All wired through REEConfig.from_dims().
  Backward compatible: all switches default False; existing experiments unaffected.
  No trainable parameters. No gradient flow in pass bodies. No phased training needed.
  Prerequisites satisfied: MECH-092 (waking quiescent replay), MECH-120 SHY wiring
  (enter_sws_mode calls shy_normalise), serotonin module (MECH-203/204), enter_offline_mode.
  Distinguishes from EXQ-242: EXQ-242 used proxy hooks (standalone functions, non_contributory).
  This implementation adds first-class REEAgent methods experiments can call directly.
  MECH-094: hypothesis_tag=True in rem_attribution_pass (terrain scoring only; no residue writes).
  Validation experiment: V3-EXQ-265 queued (SD-017 activation + slot differentiation ablation,
  2 conditions x 3 seeds, ~45 min on Mac).
  See SD-017, ARC-045, MECH-166, MECH-120 (SHY gated within enter_sws_mode).

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

- sleep_substrate:GAP-9: sleep.within_life_sleep_trigger -- IMPLEMENTED 2026-08-14 ((a)+(b)
  composed: v1 ceiling arm + the MEL/need-crossing PRIMARY arm, both landed 2026-08-14).
  Adds a WITHIN-LIFE sleep trigger so a TRUE single-continuous life
  (num_episodes=1) can sleep. Before this, the sleep trigger was BOUNDARY-only:
  SleepLoopManager.notify_episode_end() (the sole K-episode-cadence entry) is reachable only
  across an inter-episode boundary (REEAgent.reset()), so a continuous-life driver never
  slept -- no MECH-204 recalibration, no Phase B-E aggregation, no GAP-5b duration scaling
  could ever fire within such a life, independent of cadence config.
  Module: ree_core/sleep/phase_manager.py -- new SleepLoopManager.notify_waking_step(agent)
  (constructor args within_life_trigger / within_life_step_ceiling; SleepCycleState gains
  steps_since_sleep; _run_cycle takes an optional within_life_meta for arm attribution +
  resets the step counter at every reset point). Call site: ree_core/agent.py
  REEAgent.update_residue() (waking path, after the MEL note_step_pe), gated on
  use_within_life_sleep_trigger and not hypothesis_tag.
  Config (REEConfig, ree_core/utils/config.py): use_within_life_sleep_trigger (bool, default
  False; set True to enable), within_life_sleep_step_ceiling (int, default 1000). Both wired
  through REEConfig.from_dims() (3 sites).
  Data flow: waking step -> agent.update_residue(hypothesis_tag=False) [MEL note_step_pe runs
  FIRST, so the accumulator reflects this step] -> sleep_loop.notify_waking_step() ->
  increments state.steps_since_sleep -> fires _run_cycle when EITHER
  need_crossed (mel_consumer.need_crossed(): accumulated waking MEL >= mel_entry_threshold,
  the PRIMARY arm) OR steps_since_sleep >= within_life_sleep_step_ceiling (the BACKSTOP
  ceiling arm) -> reuses the existing SD-017 run_sleep_cycle path, then resets the counter.
  Emits within_life_trigger_fired / within_life_trigger_arm_ceiling / within_life_trigger_arm_need
  / within_life_steps_at_fire / within_life_mel_at_fire / within_life_need_threshold into the
  fired cycle's metrics + cycle_history.
  DESIGN: (a)+(b) composed per the 2026-08-14 lit synthesis
  (targeted_review_sleep_onset_multiinput_gap9): MEL/need-crossing (design (b)) as PRIMARY
  with a step-count ceiling (design (a)) as anti-starvation backstop, i.e.
  `need_crossed or at_ceiling`. The need arm REUSES GAP-5b's SD-MEL-CONSUMER accumulator via
  the new MELConsumer.need_crossed() (the demand-side term of entry_permitted()'s
  `crossed or at_ceiling`, factored out; entry_permitted() now delegates to it, bit-identical).
  The need arm requires a mel_consumer with use_mel_entry on + some accumulated waking PE;
  absent that it degrades gracefully to the ceiling arm alone (the intended CausalGridWorldV2
  path -- measured MEL there is noise-level per GAP-5b, so the ceiling carries firing). Design
  (c) (experimenter virtual boundary) is instrumentation (force_cycle()), NOT counted as
  closing GAP-9.
  Backward compatible: use_within_life_sleep_trigger default False -> update_residue makes NO
  new call (short-circuits) -> byte-identical; notify_episode_end() is untouched, so
  multi-episode drivers are bit-identical. No trainable parameters / no new encoder head / no
  new latent field. No phased training needed.
  MECH-094: fires ONLY on waking steps (not hypothesis_tag) and reuses the existing
  _run_cycle/run_sleep_cycle path -> adds NO new memory writes; all offline/replay content
  keeps its existing hypothesis_tag=True tagging. Re-entrancy guard (_within_life_cycle_active)
  as belt-and-braces (sleep passes never call the waking update_residue path).
  use_mech286_sleep_onset_gate deliberately NOT part of this build (its threat term reads a
  signal V3-EXQ-917 measured at chance-level place-safety discrimination).
  Contracts: tests/contracts/test_sleep_within_life_trigger_gap9.py (14 tests: OFF
  bit-identical, ceiling-arm fires on the ceiling step, periodic re-fire, boundary path
  untouched/no within_life keys, reset clears counter+guard, no-substrate declines,
  end-to-end continuous life fires ON / never OFF, re-entrancy guard, ceiling validation,
  and the need arm: G11 fires before the ceiling, G12 through update_residue's real call
  site, G13 entry-lever-off is ceiling-only even with a consumer, G14 need_crossed /
  entry_permitted delegation). use_within_life_sleep_trigger registered in
  tests/test_flag_inertness.py KNOWN_UNPROBED (v1 landed the flag without its entry, which
  had left test_flag_registry_is_current red on main).
  Validation experiments: V3-EXQ-929 (v1 ceiling arm; OFF vs ON x 3 seeds, single continuous
  life; C1 OFF fires 0, C2 ON fires >=1, C3 ON all ceiling-arm) + V3-EXQ-933 (need arm;
  diagnostic consumer-validation with a controlled MEL stimulus -- demand-sensitivity,
  threshold gating, graceful ceiling-degradation; PROMOTES NOTHING).
  See REE_assembly/evidence/planning/sleep_substrate_plan.md (GAP-9); SD-017; SD-MEL-CONSUMER
  (GAP-5b, whose accumulator the need arm reuses).

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

- SD-MEL-PRODUCER: environment.non_converging_world_rule_shift -- IMPLEMENTED 2026-07-21.
  The PRODUCER half of the MECH-180 pair -- link (i) novelty -> graded above-reference
  waking MEL. SD-MEL-CONSUMER (above) owns link (ii) and is already PROVEN; link (i) had
  never been demonstrated because the environment could not produce a novelty gradient
  at all: V3-EXQ-677's C1 manipulation check measured a high- vs low-novelty mean E1
  prediction-error difference of 8.8e-07 against a 0.01 threshold, and V3-EXQ-718a
  measured ecological MEL ~1e-5 (noise-level, scrambled vs novelty level) with
  conv_rel_drop ~0.98. Both autopsies classify this measurement_gap (environment /
  test-bed producer gap), NOT a substrate ceiling and NOT a falsification.
  ROOT CAUSE: env_drift_interval fires _drift_hazards(), which only MOVES hazards. The
  optimal prediction of a random walk is its mean, so the world-forward model learns
  that fast and PE floors at the irreducible noise level -- drift adds sampling NOISE,
  not learning LOAD.
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorld._maybe_shift_world_rule;
  inherited by CausalGridWorldV2). No new file, no new module.
  Config (env kwargs, NOT REEConfig -- experiment scripts construct envs directly):
  world_rule_shift_enabled (bool, default False; set True to enable),
  world_rule_shift_interval (int, default 0 = never; WORLD-steps between rule re-draws),
  world_rule_shift_depth (int, default 0; action-pair transpositions per shift),
  world_rule_shift_scope (str, default "action_map"; reserved for a later
  structural-statistics variant, ValueError on anything else).
  Data flow: world_rule_shift_interval -> _maybe_shift_world_rule() re-permutes
  self._action_map from self._rng -> dx,dy at step() -> the actual transition changes
  -> z_world(t+1) diverges from E2.world_forward(z_world(t), a), which takes the action
  as an INPUT -> e3 prediction_error rises -> info["world_rule_shift_occurred" /
  "world_rule_shift_count" / "steps_since_world_rule_shift"].
  WHY NOISE IS NOT A SUBSTITUTE (the central design constraint): grading OBSERVATION
  NOISE would also yield a monotone MEL gradient -- but by construction, on any
  substrate, whether or not MECH-180 is true. That is the DV-symmetry artifact class
  (failure_autopsy_V3-EXQ-604c; the defect that held V3-EXQ-683 on 2026-07-21). Elevated
  PE is only learning LOAD if it is reducible, so the operational discriminator is that
  genuine load DECAYS within a stationary window while noise does not.
  steps_since_world_rule_shift exists to let a consumer bin per-step PE by
  time-since-shift and measure exactly that; the validation carries a matched-PE noise
  arm as the negative control.
  FOUR LOAD-BEARING DECISIONS, all contract-pinned in
  tests/contracts/test_world_rule_shift_producer.py (12 tests) -- do not "simplify":
  (1) the CLASS-level ACTIONS dict is never mutated (it is a class attribute, so an
  in-place permutation would leak into every other env instance in the process); the
  effective map is a per-instance self._action_map.
  (2) the schedule keys off a cumulative self._world_steps_total, NOT episode-local
  self.steps. This was a LIVE DEFECT caught by the implementation probe: episode length
  itself collapses as the world gets less predictable (measured 69.0 -> 13.5 steps), so
  an episode-relative schedule makes the nominal interval stop controlling the actual
  shift RATE -- intervals 60/30/15/8/5 produced 2/2/3/20/21 shifts, non-monotone, taking
  the MEL ladder with it.
  (3) _action_map, the shift counters and _world_steps_total are NOT reset by reset() /
  reset_to() -- the action map is the world's causal structure, not episode state.
  (4) every RNG draw sits inside the enabled guard, so a disabled env consumes NO
  randomness and is bit-identical at the same seed (verified on seeds 42/123/456).
  Backward compatible: disabled by default; existing experiments unaffected, including
  seeded-rollout bit-exactness.
  No trainable parameters / no new encoder head / no new latent field -> NO phased
  training needed (validation still needs a converged recon-only base so PE sits at
  converged scale, per 701c; SD-056 contrastive is a confirmed P0 destabiliser).
  MECH-094: does not apply -- nothing is written to memory during non-waking states.
  MEASUREMENT NOTE for any consumer of this knob: shift rate SHORTENS episodes, so
  measure over a fixed STEP budget, not a fixed episode count, and report mean episode
  length per arm so the confound stays visible.
  Unblocks: MECH-180 link (i), INV-050 ecological end-to-end demonstration (which is a
  SEPARATE, still-gated run -- this test-bed must validate first).
  Validation experiment: V3-EXQ-798 queued 2026-07-21
  (v3_exq_798_sdmelproducer_graded_nonconverging_world; diagnostic; claim_ids=[];
  recon-only converged base + graded rule-shift arms + matched-PE noise negative
  control; DV = mean per-step e3 prediction_error (MEL) and its decay by
  steps_since_world_rule_shift; PROMOTES NOTHING until it scores).
  The consumer is DELIBERATELY ABSENT from that validation: V3-EXQ-718a's
  learning_extracted[1] records that the consumer's DV is a deterministic function of
  MEL, so DV-monotone-in-measured-MEL is near-tautological and cannot validate a producer.
  See REE_assembly/docs/architecture/sd_mel_producer.md; SD-MEL-CONSUMER; SD-017;
  INV-050; MECH-180; plan-of-record
  REE_assembly/evidence/planning/sleep_substrate_plan.md (GAP-5b).
