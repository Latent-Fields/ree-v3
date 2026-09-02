"""Contracts for REEAgent.force_sleep_cycle_at_eval_boundary (MECH-027 Build 2).

Before this method existed, the only documented convention for
SleepLoopManager.force_cycle() was WARMUP-only (called before an eval loop
starts). The only place a driver actually called it AT an eval segment
boundary -- without a full agent.reset() -- was V3-EXQ-909's driver, which
hand-rolled a two-step sequence (`agent._flush_exploration_episode()` then
`agent.sleep_loop.force_cycle(agent)`) with no documented contract that the
ordering was required or that the two-step sequence was safe.
force_sleep_cycle_at_eval_boundary() formalizes that sequence as a single,
tested, documented REEAgent method.

Guarantees enforced:
  C1. Fires a sleep cycle immediately, bypassing the K-episode cadence
      (episodes_since_sleep never incremented via notify_episode_end).
  C2. Flushes the just-completed segment's exploration trajectory into the
      hippocampal exploration buffer BEFORE the cycle fires (matches
      agent.reset()'s own ordering).
  C3. Does NOT reset agent state that a full agent.reset() would reset
      (step counter, episode harm accumulator, current latent) -- the
      defining difference from calling agent.reset() at a boundary.
  C4. Returns None (no-op downstream) when use_sleep_loop is False, without
      raising -- mirrors notify_episode_end's None-safe consumption.
  C5. Resets the K-episode / within-life step counters on completion, same
      as force_cycle() itself, so the cadence stays consistent afterward.
"""

from __future__ import annotations

import torch


def _build_agent(*, sleep_loop: bool = True, sws: bool = True, rem: bool = True):
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_loop=sleep_loop,
        sleep_loop_episodes_K=100,  # never reached via notify_episode_end
        replay_diversity_enabled=True,
    )
    cfg.sws_enabled = sws
    cfg.rem_enabled = rem
    return REEAgent(cfg)


def _populate_episode_buffers(agent, n_steps: int = 6):
    """Hand-populate the private per-episode exploration buffers with
    synthetic tensors, mirroring what REEAgent._record_exploration_state() /
    the action-append call would have produced during real waking steps
    (test_stdlib_rng_seed_determinism.py uses the equivalent direct-injection
    pattern one level up, via record_exploration_trajectory). Shapes are
    arbitrary but internally consistent -- _flush_exploration_episode /
    record_exploration_trajectory do not validate against the agent's real
    latent dims, only .detach() and stack the tensors as given."""
    world_dim = 16
    self_dim = 16
    action_dim = 4
    for _ in range(n_steps):
        agent._episode_world_states.append(torch.randn(1, world_dim))
        agent._episode_self_states.append(torch.randn(1, self_dim))
    for _ in range(n_steps - 1):
        agent._episode_actions.append(torch.zeros(1, action_dim))


def test_c1_bypasses_k_episode_cadence():
    agent = _build_agent()
    _populate_episode_buffers(agent)
    assert agent.sleep_loop.state.episodes_since_sleep == 0
    assert agent.sleep_loop.state.cycle_index == 0

    metrics = agent.force_sleep_cycle_at_eval_boundary()

    assert metrics is not None
    assert agent.sleep_loop.state.cycle_index == 1  # fired, despite K=100


def test_c2_flushes_exploration_buffer_before_firing():
    agent = _build_agent()
    n_before = len(agent.hippocampal._exploration_buffer)
    _populate_episode_buffers(agent, n_steps=6)
    assert len(agent._episode_world_states) == 6  # not yet flushed

    agent.force_sleep_cycle_at_eval_boundary()

    assert len(agent.hippocampal._exploration_buffer) == n_before + 1
    assert len(agent._episode_world_states) == 0  # cleared by the flush


def test_c3_does_not_reset_agent_state():
    agent = _build_agent()
    _populate_episode_buffers(agent)
    agent._step_count = 17
    agent._harm_this_episode = 3.5
    latent_before = agent._current_latent

    agent.force_sleep_cycle_at_eval_boundary()

    assert agent._step_count == 17
    assert agent._harm_this_episode == 3.5
    assert agent._current_latent is latent_before  # untouched object identity


def test_c4_none_when_sleep_loop_off_no_raise():
    agent = _build_agent(sleep_loop=False)
    assert agent.sleep_loop is None
    _populate_episode_buffers(agent)

    result = agent.force_sleep_cycle_at_eval_boundary()

    assert result is None
    # The flush still runs unconditionally (it does not depend on sleep_loop).
    assert len(agent._episode_world_states) == 0


def test_c5_resets_cadence_counters_after_firing():
    agent = _build_agent()
    _populate_episode_buffers(agent)
    # Simulate some accumulated (but not-yet-fired) cadence state.
    agent.sleep_loop.state.episodes_since_sleep = 4
    agent.sleep_loop.state.steps_since_sleep = 9

    agent.force_sleep_cycle_at_eval_boundary()

    assert agent.sleep_loop.state.episodes_since_sleep == 0
    assert agent.sleep_loop.state.steps_since_sleep == 0
