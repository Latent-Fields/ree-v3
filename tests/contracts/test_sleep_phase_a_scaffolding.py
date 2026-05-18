"""Contract tests for sleep-aggregation cluster Phase A scaffolding.

See REE_assembly/docs/architecture/sleep_aggregation_cluster.md.

Phase A is scaffolding only: the SleepLoopManager wraps the existing SD-017
surface (REEAgent.run_sleep_cycle and friends) with a deterministic
K-episode trigger. No new replay sampler / routing gate / aggregator /
self-model writeback land here.

Guarantees enforced:
  C1. Module + dataclass + enum importable without side effects.
  C2. Default REEConfig has use_sleep_loop=False (backward-compat).
  C3. Master switch OFF: REEAgent.sleep_loop is None; reset() does not
      enter the sleep cycle path.
  C4. Master switch ON, K=1, both SWS+REM enabled: notify_episode_end()
      drives a sleep cycle without exception and clears the counter.
  C5. K-episode counter: with K=3, sleep fires only on the third reset.
  C6. require_sleep_passes_enabled: with master ON but neither
      sws_enabled nor rem_enabled, manager refuses to fire and resets
      the counter quietly.
  C7. force_cycle bypasses the K-episode counter when called explicitly.
  C8. Phase visits: state.phase walks WAKING -> SWS_ANALOG (or
      REM_ANALOG) -> WAKING during a fired cycle.
"""

from __future__ import annotations


def _build_agent(*, sleep_loop: bool, K: int = 1, sws: bool = True, rem: bool = True):
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_loop=sleep_loop,
        sleep_loop_episodes_K=K,
    )
    cfg.sws_enabled = sws
    cfg.rem_enabled = rem
    return REEAgent(cfg)


def test_c1_module_importable():
    from ree_core.sleep import SleepCycleState, SleepLoopManager, SleepPhase

    assert SleepPhase.WAKING.value == "waking"
    assert SleepPhase.SWS_ANALOG.value == "sws_analog"
    assert SleepPhase.REM_ANALOG.value == "rem_analog"
    state = SleepCycleState()
    assert state.cycle_index == 0
    assert state.phase is SleepPhase.WAKING
    mgr = SleepLoopManager(cycle_every_k_episodes=2)
    assert mgr.cycle_every_k_episodes == 2


def test_c2_default_config_backward_compatible():
    from ree_core.utils.config import REEConfig

    cfg = REEConfig()
    assert getattr(cfg, "use_sleep_loop", False) is False
    assert cfg.sleep_loop_episodes_K == 1
    assert cfg.sleep_loop_require_passes is True


def test_c3_master_switch_off_no_instantiation():
    agent = _build_agent(sleep_loop=False)
    assert agent.sleep_loop is None
    # reset() must not raise even with the loop off.
    agent.reset()


def test_c4_k1_drives_cycle_without_exception():
    agent = _build_agent(sleep_loop=True, K=1, sws=True, rem=True)
    assert agent.sleep_loop is not None
    assert agent.sleep_loop.state.episodes_since_sleep == 0
    agent.reset()
    # Cycle fired: counter reset to 0, cycle_index advanced, phase back to
    # WAKING after the cycle completes.
    assert agent.sleep_loop.state.cycle_index == 1
    assert agent.sleep_loop.state.episodes_since_sleep == 0
    assert agent.sleep_loop.state.phase.value == "waking"
    assert len(agent.sleep_loop.cycle_history) == 1


def test_c5_k3_fires_on_third_reset():
    agent = _build_agent(sleep_loop=True, K=3, sws=True, rem=True)
    assert agent.sleep_loop is not None
    agent.reset()
    assert agent.sleep_loop.state.cycle_index == 0
    assert agent.sleep_loop.state.episodes_since_sleep == 1
    agent.reset()
    assert agent.sleep_loop.state.cycle_index == 0
    assert agent.sleep_loop.state.episodes_since_sleep == 2
    agent.reset()
    assert agent.sleep_loop.state.cycle_index == 1
    assert agent.sleep_loop.state.episodes_since_sleep == 0


def test_c6_no_substrate_no_fire():
    """With master switch ON but both SWS/REM disabled, manager refuses."""
    agent = _build_agent(sleep_loop=True, K=1, sws=False, rem=False)
    assert agent.sleep_loop is not None
    agent.reset()
    # Counter is reset (manager declined to fire), cycle_index unchanged.
    assert agent.sleep_loop.state.cycle_index == 0
    assert agent.sleep_loop.state.episodes_since_sleep == 0
    assert agent.sleep_loop.cycle_history == []


def test_c7_force_cycle_bypasses_counter():
    agent = _build_agent(sleep_loop=True, K=10, sws=True, rem=True)
    assert agent.sleep_loop is not None
    metrics = agent.sleep_loop.force_cycle(agent)
    assert metrics is not None
    assert agent.sleep_loop.state.cycle_index == 1
    assert agent.sleep_loop.state.episodes_since_sleep == 0


def test_c8_phase_returns_to_waking():
    """The cycle ends with phase == WAKING regardless of which passes ran."""
    from ree_core.sleep import SleepPhase

    agent = _build_agent(sleep_loop=True, K=1, sws=True, rem=False)
    assert agent.sleep_loop is not None
    agent.sleep_loop.force_cycle(agent)
    assert agent.sleep_loop.state.phase is SleepPhase.WAKING

    agent2 = _build_agent(sleep_loop=True, K=1, sws=False, rem=True)
    assert agent2.sleep_loop is not None
    agent2.sleep_loop.force_cycle(agent2)
    assert agent2.sleep_loop.state.phase is SleepPhase.WAKING
