## sleep_substrate:GAP-9: sleep.within_life_sleep_trigger -- IMPLEMENTED (2026-08-14)
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
