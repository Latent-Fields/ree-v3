"""Contract tests for the sleep_substrate:GAP-9 within-life sleep trigger.

GAP-9 (registered 2026-08-12): the sleep trigger is BOUNDARY-only --
SleepLoopManager.notify_episode_end() is the sole K-episode-cadence entry and
is reachable only across an inter-episode boundary (REEAgent.reset()), so a TRUE
single-continuous life (num_episodes=1) can never sleep, independent of any
cadence config. This is the wall these tests fix.

Design (v1, per REE_assembly/evidence/planning/sleep_substrate_plan.md GAP-9 +
the 2026-08-14 lit synthesis targeted_review_sleep_onset_multiinput_gap9): the
(a)+(b) composed trigger, wiring the STEP-COUNT CEILING arm (design (a), the
anti-starvation backstop) ONLY. The MEL/learning-demand need-crossing arm
(design (b), the primary trigger) is a planned follow-up; the arm-attribution
diagnostics are already emitted so a ceiling-only run is never mistaken for a
demand-sensitive one (the V3-EXQ-718a failure mode one level up).

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
