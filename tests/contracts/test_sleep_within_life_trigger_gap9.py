"""Contract tests for the sleep_substrate:GAP-9 within-life sleep trigger.

GAP-9 (registered 2026-08-12): the sleep trigger is BOUNDARY-only --
SleepLoopManager.notify_episode_end() is the sole K-episode-cadence entry and
is reachable only across an inter-episode boundary (REEAgent.reset()), so a TRUE
single-continuous life (num_episodes=1) can never sleep, independent of any
cadence config. This is the wall these tests fix.

Design (per REE_assembly/evidence/planning/sleep_substrate_plan.md GAP-9 + the
2026-08-14 lit synthesis targeted_review_sleep_onset_multiinput_gap9): the
(a)+(b) composed trigger -- fire iff `need_crossed or at_ceiling`. Design (b),
the MEL/learning-demand need-crossing arm, is the PRIMARY trigger (reuses
GAP-5b's SD-MEL-CONSUMER accumulator via MELConsumer.need_crossed()); design (a),
the STEP-COUNT CEILING arm, is the anti-starvation BACKSTOP. v1 (2026-08-14,
ree-v3 5f14036) wired the ceiling arm only; the need arm landed as the follow-up
(this file's G11-G14). The arm-attribution diagnostics are emitted so a
ceiling-carried run is never mistaken for a demand-sensitive one (the V3-EXQ-718a
failure mode one level up). Expected in CausalGridWorldV2: the ceiling carries
firing (measured MEL there is noise-level per GAP-5b) -- graceful degradation to
design (a), not a bug.

Guarantees enforced:
  G1. Default REEConfig: use_within_life_sleep_trigger=False,
      within_life_sleep_step_ceiling==1000; from_dims accepts the knobs.
  G2. OFF (bit-identical): flag off -> notify_waking_step returns None every
      call, cycle_history stays empty over > ceiling steps, and the manager's
      within_life_trigger is False.
  G3. ON, ceiling arm: flag on + small ceiling + SWS/REM enabled -> a cycle
      fires exactly on the ceiling-th waking step, with within_life_trigger_fired
      / _arm_ceiling==1.0, _arm_need==0.0, _steps_at_fire==ceiling, and those
      keys are recorded in cycle_history (authoritative).
  G4. Periodicity: a second ceiling window fires a second cycle (counter resets).
  G5. Boundary path untouched: notify_episode_end() still fires per K and its
      metrics carry NO within_life_* keys (multi-episode drivers bit-identical).
  G6. reset() clears steps_since_sleep and the re-entrancy flag.
  G7. No substrate: flag on but neither sws_enabled nor rem_enabled -> the
      trigger evaluates but _run_cycle declines; steps reset, history empty.
  G8. End-to-end: in a TRUE single continuous life (sense + update_residue,
      NEVER reset()), sleep fires after `ceiling` waking steps with the flag on,
      and NEVER fires with the flag off. This is the GAP-9 acceptance criterion.
  G9. Re-entrancy guard: a fired cycle cannot recursively re-trigger.
  G10. Ceiling validation: within_life_step_ceiling < 1 raises ValueError.
  G11. Need arm (design (b)): consumer + entry lever on + crossable threshold ->
       accumulated MEL fires the cycle BEFORE the (high) step ceiling, with
       _arm_need==1.0 / _arm_ceiling==0.0 and the demand diagnostics
       (_mel_at_fire / _need_threshold) captured at fire time.
  G12. Need arm at the real call site: update_residue() invokes notify_waking_step
       internally; with a high injected waking PE the need signal flows
       note_step_pe -> need_crossed() and carries the firing on an early step,
       below the ceiling (e3's PE production is env-dependent, so it is injected).
  G13. Entry lever OFF (consumer present): a high injected MEL does NOT fire
       early -> the ceiling arm alone carries (v1 behaviour preserved).
  G14. Predicate unit: MELConsumer.need_crossed() gates on use_mel_entry + count
       + threshold; entry_permitted() == `need_crossed() or at_ceiling`
       (bit-identical delegation in both lever states).
"""

from __future__ import annotations

import pytest
import torch


def _build_agent(
    *,
    trigger: bool,
    ceiling: int = 3,
    K: int = 1000,
    sws: bool = True,
    rem: bool = True,
    use_mel_consumer: bool = False,
    use_mel_entry: bool = False,
    mel_entry_threshold: float = 0.0,
):
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_loop=True,
        # K high so the boundary path effectively never fires -- isolates the
        # within-life trigger under test.
        sleep_loop_episodes_K=K,
        use_within_life_sleep_trigger=trigger,
        within_life_sleep_step_ceiling=ceiling,
        # design (b) need arm: the MEL consumer (GAP-5b) supplies need_crossed().
        use_mel_consumer=use_mel_consumer,
        use_mel_entry=use_mel_entry,
        mel_entry_threshold=mel_entry_threshold,
    )
    cfg.sws_enabled = sws
    cfg.rem_enabled = rem
    return REEAgent(cfg)


def _drive_continuous_life(agent, n_steps: int) -> None:
    """One TRUE continuous life: sense + update_residue per step, NEVER reset()."""
    for _ in range(n_steps):
        agent.sense(torch.zeros(12), torch.zeros(250))
        agent.update_residue(harm_signal=0.0)


def test_g1_default_config_backward_compatible():
    from ree_core.utils.config import REEConfig

    cfg = REEConfig()
    assert getattr(cfg, "use_within_life_sleep_trigger", None) is False
    assert getattr(cfg, "within_life_sleep_step_ceiling", None) == 1000
    # from_dims accepts + forwards the knobs (silent-kwargs guard).
    cfg2 = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_within_life_sleep_trigger=True,
        within_life_sleep_step_ceiling=42,
    )
    assert cfg2.use_within_life_sleep_trigger is True
    assert cfg2.within_life_sleep_step_ceiling == 42


def test_g2_off_is_bit_identical_no_within_life_sleep():
    agent = _build_agent(trigger=False, ceiling=3)
    assert agent.sleep_loop is not None
    assert agent.sleep_loop.within_life_trigger is False
    # notify_waking_step is a no-op returning None even past the ceiling.
    for _ in range(10):
        assert agent.sleep_loop.notify_waking_step(agent) is None
    assert agent.sleep_loop.state.steps_since_sleep == 0
    assert agent.sleep_loop.cycle_history == []


def test_g3_ceiling_arm_fires_on_the_ceiling_step():
    agent = _build_agent(trigger=True, ceiling=3, sws=True, rem=True)
    assert agent.sleep_loop.within_life_trigger is True
    assert agent.sleep_loop.within_life_step_ceiling == 3
    # First two steps: below ceiling -> no fire.
    assert agent.sleep_loop.notify_waking_step(agent) is None
    assert agent.sleep_loop.state.steps_since_sleep == 1
    assert agent.sleep_loop.notify_waking_step(agent) is None
    assert agent.sleep_loop.state.steps_since_sleep == 2
    # Third step: ceiling reached -> fire.
    metrics = agent.sleep_loop.notify_waking_step(agent)
    assert metrics is not None
    assert metrics["within_life_trigger_fired"] == 1.0
    assert metrics["within_life_trigger_arm_ceiling"] == 1.0
    assert metrics["within_life_trigger_arm_need"] == 0.0
    assert metrics["within_life_steps_at_fire"] == 3.0
    assert agent.sleep_loop.state.cycle_index == 1
    assert agent.sleep_loop.state.steps_since_sleep == 0
    # cycle_history is the authoritative record and carries the arm attribution.
    assert len(agent.sleep_loop.cycle_history) == 1
    hist = agent.sleep_loop.cycle_history[-1]
    assert hist["within_life_trigger_fired"] == 1.0
    assert hist["within_life_trigger_arm_ceiling"] == 1.0


def test_g4_periodic_second_window_fires_second_cycle():
    agent = _build_agent(trigger=True, ceiling=2, sws=True, rem=True)
    for _ in range(4):  # two full ceiling windows
        agent.sleep_loop.notify_waking_step(agent)
    assert agent.sleep_loop.state.cycle_index == 2
    assert len(agent.sleep_loop.cycle_history) == 2


def test_g5_boundary_path_untouched_no_within_life_keys():
    # K=1 so notify_episode_end fires immediately; within-life trigger also on.
    agent = _build_agent(trigger=True, ceiling=1000, K=1, sws=True, rem=True)
    metrics = agent.sleep_loop.notify_episode_end(agent)
    assert metrics is not None
    # The boundary path must NOT stamp within-life arm attribution.
    assert "within_life_trigger_fired" not in metrics
    assert "within_life_trigger_arm_ceiling" not in metrics
    assert agent.sleep_loop.state.cycle_index == 1


def test_g6_reset_clears_step_counter_and_guard():
    agent = _build_agent(trigger=True, ceiling=5)
    agent.sleep_loop.notify_waking_step(agent)
    agent.sleep_loop.notify_waking_step(agent)
    assert agent.sleep_loop.state.steps_since_sleep == 2
    agent.sleep_loop.reset()
    assert agent.sleep_loop.state.steps_since_sleep == 0
    assert agent.sleep_loop._within_life_cycle_active is False


def test_g7_no_substrate_trigger_evaluates_but_run_cycle_declines():
    agent = _build_agent(trigger=True, ceiling=1, sws=False, rem=False)
    # Ceiling=1 so the very first step attempts a cycle; _run_cycle declines
    # (no SD-017 passes) and returns None, but the step counter is still reset.
    assert agent.sleep_loop.notify_waking_step(agent) is None
    assert agent.sleep_loop.state.steps_since_sleep == 0
    assert agent.sleep_loop.state.cycle_index == 0
    assert agent.sleep_loop.cycle_history == []


def test_g8_end_to_end_continuous_life_fires_on_and_not_off():
    # ON: a true single continuous life (no reset()) sleeps after `ceiling` steps.
    on = _build_agent(trigger=True, ceiling=3, sws=True, rem=True)
    _drive_continuous_life(on, 7)  # fires at step 3 and step 6
    assert len(on.sleep_loop.cycle_history) == 2
    assert all(
        h["within_life_trigger_arm_ceiling"] == 1.0
        for h in on.sleep_loop.cycle_history
    )
    # OFF: the identical continuous life never sleeps (the GAP-9 wall).
    off = _build_agent(trigger=False, ceiling=3, sws=True, rem=True)
    _drive_continuous_life(off, 7)
    assert len(off.sleep_loop.cycle_history) == 0


def test_g9_reentrancy_guard_blocks_recursive_trigger():
    agent = _build_agent(trigger=True, ceiling=1, sws=True, rem=True)
    # Simulate being mid-cycle: the guard must make notify_waking_step a no-op.
    agent.sleep_loop._within_life_cycle_active = True
    assert agent.sleep_loop.notify_waking_step(agent) is None
    # Counter is not advanced while guarded.
    assert agent.sleep_loop.state.steps_since_sleep == 0
    agent.sleep_loop._within_life_cycle_active = False


def test_g10_ceiling_validation():
    from ree_core.sleep import SleepLoopManager

    with pytest.raises(ValueError):
        SleepLoopManager(within_life_trigger=True, within_life_step_ceiling=0)


def test_g11_need_arm_fires_before_ceiling():
    # PRIMARY arm (design (b)): consumer + entry lever on + a crossable threshold.
    # ceiling=100 (high) so ONLY the need arm can be the cause of an early fire.
    agent = _build_agent(
        trigger=True, ceiling=100, sws=True, rem=True,
        use_mel_consumer=True, use_mel_entry=True, mel_entry_threshold=1e-3,
    )
    assert agent.mel_consumer is not None
    # Inject a high waking PE (as update_residue.note_step_pe would on a real
    # high-demand step); mean MEL = 1.0 >> threshold -> need_crossed True.
    agent.mel_consumer.note_step_pe(1.0)
    metrics = agent.sleep_loop.notify_waking_step(agent)
    assert metrics is not None
    assert metrics["within_life_trigger_fired"] == 1.0
    assert metrics["within_life_trigger_arm_need"] == 1.0
    assert metrics["within_life_trigger_arm_ceiling"] == 0.0
    # Fired on step 1, far below the ceiling -> the need arm carried it.
    assert metrics["within_life_steps_at_fire"] == 1.0
    assert metrics["within_life_steps_at_fire"] < 100
    # Demand-side diagnostics captured at fire time (before the accumulator reset).
    assert metrics["within_life_mel_at_fire"] == pytest.approx(1.0)
    assert metrics["within_life_need_threshold"] == pytest.approx(1e-3)
    # cycle_history is authoritative and carries the need-arm attribution.
    hist = agent.sleep_loop.cycle_history[-1]
    assert hist["within_life_trigger_arm_need"] == 1.0


def test_g12_need_arm_carries_through_update_residue():
    # Integration at the REAL production call site: REEAgent.update_residue()
    # invokes notify_waking_step() internally (agent.py). e3's prediction_error
    # production is environment-dependent -- it needs a selected trajectory and
    # is noise-level in CausalGridWorldV2 (GAP-5b) -- so we inject the waking PE
    # via note_step_pe (exactly the call update_residue makes when e3 emits one),
    # isolating the NEED-ARM plumbing: note_step_pe -> need_crossed() ->
    # notify_waking_step fires the need arm on an early step, below the ceiling.
    agent = _build_agent(
        trigger=True, ceiling=100, sws=True, rem=True,
        use_mel_consumer=True, use_mel_entry=True, mel_entry_threshold=1e-3,
    )
    agent.sense(torch.zeros(12), torch.zeros(250))
    agent.mel_consumer.note_step_pe(1.0)     # a high-demand waking step
    agent.update_residue(harm_signal=0.0)    # real path -> notify_waking_step
    assert len(agent.sleep_loop.cycle_history) == 1
    first = agent.sleep_loop.cycle_history[0]
    assert first["within_life_trigger_arm_need"] == 1.0
    assert first["within_life_trigger_arm_ceiling"] == 0.0
    assert first["within_life_steps_at_fire"] < 100


def test_g13_entry_lever_off_is_ceiling_only_even_with_consumer():
    # use_mel_entry OFF but a consumer present: need_crossed() is always False, so
    # a high injected MEL must NOT fire early -- the ceiling arm alone carries
    # (v1 behaviour preserved even when a MEL consumer is attached).
    agent = _build_agent(
        trigger=True, ceiling=3, sws=True, rem=True,
        use_mel_consumer=True, use_mel_entry=False, mel_entry_threshold=1e-6,
    )
    agent.mel_consumer.note_step_pe(1.0)  # high demand, but the lever is off
    assert agent.sleep_loop.notify_waking_step(agent) is None  # step 1: no fire
    assert agent.sleep_loop.notify_waking_step(agent) is None  # step 2: no fire
    metrics = agent.sleep_loop.notify_waking_step(agent)       # step 3: ceiling
    assert metrics is not None
    assert metrics["within_life_trigger_arm_ceiling"] == 1.0
    assert metrics["within_life_trigger_arm_need"] == 0.0
    assert metrics["within_life_steps_at_fire"] == 3.0


def test_g14_need_crossed_and_entry_permitted_delegation():
    from ree_core.sleep.mel_consumer import MELConsumer, MELConsumerConfig

    # Lever OFF: need_crossed always False; entry_permitted is pure ceiling.
    off = MELConsumer(MELConsumerConfig(use_mel_entry=False, mel_entry_threshold=0.5))
    off.note_step_pe(10.0)  # huge MEL, but the lever is off
    assert off.need_crossed() is False
    assert off.entry_permitted(0, 3) is False  # 0 < 3
    assert off.entry_permitted(3, 3) is True   # ceiling
    assert off.entry_permitted(5, 3) is True

    # Lever ON: need_crossed reflects crossing; entry_permitted == crossed OR ceiling.
    on = MELConsumer(MELConsumerConfig(use_mel_entry=True, mel_entry_threshold=0.5))
    assert on.need_crossed() is False          # no PE accumulated yet (count 0)
    assert on.entry_permitted(0, 3) is False
    on.note_step_pe(1.0)                        # mean 1.0 >= 0.5 -> crossed
    assert on.need_crossed() is True
    assert on.entry_permitted(0, 3) is True     # crossed carries below ceiling
    # Below threshold: not crossed, but the ceiling still backstops.
    on.reset()
    on.note_step_pe(0.1)                        # mean 0.1 < 0.5
    assert on.need_crossed() is False
    assert on.entry_permitted(0, 3) is False
    assert on.entry_permitted(3, 3) is True
