"""Contract tests for MECH-286 override-gated sleep-mode entry."""

from __future__ import annotations


def _build_agent(**kwargs):
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1,
        sws_enabled=True,
        rem_enabled=False,
        **kwargs,
    )
    return REEAgent(cfg)


def test_c1_default_off_bit_identical_sleep_fires():
    agent = _build_agent(use_mech286_sleep_onset_gate=False)
    assert agent.sleep_loop is not None
    agent.reset()
    assert agent.sleep_loop.state.cycle_index == 1
    assert "mech286_sleep_permitted" not in (agent.sleep_loop.state.last_metrics or {})


def test_c2_gate_on_high_override_blocks_sleep():
    agent = _build_agent(
        use_mech286_sleep_onset_gate=True,
        use_broadcast_override=True,
        use_staleness_accumulator=True,
        use_e2_harm_a=True,
    )
    assert agent.broadcast_override is not None
    for _ in range(60):
        agent.broadcast_override.tick(drive_level=0.95, z_harm_norm=0.8)
    acc = agent.hippocampal.staleness_accumulator
    assert acc is not None
    acc._staleness[("fast", "0.0")] = 0.9
    agent.reset()
    assert agent.sleep_loop.state.cycle_index == 0
    m = agent.sleep_loop.state.last_metrics or {}
    assert m.get("mech286_sleep_permitted", -1.0) == 0.0
    assert m.get("mech286_override_ok", 1.0) == 0.0


def test_c3_gate_on_favorable_conditions_permit_sleep():
    agent = _build_agent(
        use_mech286_sleep_onset_gate=True,
        use_staleness_accumulator=True,
        use_e2_harm_a=True,
    )
    acc = agent.hippocampal.staleness_accumulator
    assert acc is not None
    acc._staleness[("fast", "0.0")] = 0.8
    agent.reset()
    assert agent.sleep_loop.state.cycle_index == 1
    m = agent.sleep_loop.state.last_metrics or {}
    assert m.get("mech286_sleep_permitted", 0.0) == 1.0


def test_c4_evaluate_permit_api_matches_manager():
    from ree_core.sleep.sleep_onset_gate import evaluate_sleep_onset_permit

    agent = _build_agent(
        use_mech286_sleep_onset_gate=True,
        use_staleness_accumulator=True,
    )
    agent.hippocampal.staleness_accumulator._staleness[("slow", "1.0")] = 0.5
    permitted, diag = evaluate_sleep_onset_permit(agent)
    assert permitted is True
    assert diag["mech286_staleness_ok"] == 1.0
