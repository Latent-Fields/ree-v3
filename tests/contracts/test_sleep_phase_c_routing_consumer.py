"""Contract tests for sleep-aggregation cluster GAP-8: MECH-272 routing-gate
downstream consumer (anchor_channel -> ContextMemory write scale).

See REE_assembly/evidence/planning/sleep_substrate_plan.md GAP-8.

Guarantees enforced:
  C1. use_mech272_routing_consumer defaults False in REEConfig (backward-compat).
  C2. Consumer OFF, routing ON: SleepLoopManager.use_mech272_routing_consumer
      is False; mean_anchor stays 1.0 -> run_sws_schema_pass writes at full
      strength (sws_anchor_weight_applied == 1.0).
  C3. Consumer ON, routing ON: force_cycle produces sws_anchor_weight_applied
      matching mech272_sws_anchor_weight config (0.6 default) when draws exist.
  C4. Consumer ON, no routing gate (no draws): mean_anchor falls back to 1.0
      -> sws_anchor_weight_applied == 1.0.
  C5. SleepLoopManager.__init__ propagates use_mech272_routing_consumer
      correctly from the constructor argument.
  C6. REEAgent instantiation passes use_mech272_routing_consumer from config
      to SleepLoopManager (end-to-end constructor wiring).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #

def _build_agent(
    *,
    sleep_loop: bool = True,
    sampler: bool = True,
    routing: bool = True,
    routing_consumer: bool = False,
    sws: bool = True,
    rem: bool = False,
    anchor_sets: bool = True,
    staleness: bool = True,
    draws: int = 8,
    sws_anchor: float = 0.6,
    sws_probe: float = 0.4,
):
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_loop=sleep_loop,
        sleep_loop_episodes_K=1,
        use_mech285_sampler=sampler,
        mech285_draws_per_cycle=draws,
        use_anchor_sets=anchor_sets,
        use_staleness_accumulator=staleness,
        use_mech272_routing=routing,
        mech272_sws_anchor_weight=sws_anchor,
        mech272_sws_probe_weight=sws_probe,
        use_mech272_routing_consumer=routing_consumer,
    )
    cfg.sws_enabled = sws
    cfg.rem_enabled = rem
    return REEAgent(cfg)


def _install_anchors(agent, *, n: int = 4):
    import torch

    anchor_set = agent.hippocampal.anchor_set
    assert anchor_set is not None
    for i in range(n):
        z = torch.randn(1, 32)
        anchor_set.write_anchor(
            scale="fast",
            segment_id=str(i),
            stream_mixture=(f"s{i}",),
            z_world=z,
        )


def _run_cycle_with_buffer(agent):
    """Push observations into buffers then run force_cycle."""
    import torch

    obs_body = torch.zeros(12)
    obs_world = torch.zeros(250)
    for _ in range(6):
        agent.sense(obs_body=obs_body, obs_world=obs_world)
    _install_anchors(agent)
    return agent.sleep_loop.force_cycle(agent)


# ---------------------------------------------------------------------------- #
# Contracts                                                                    #
# ---------------------------------------------------------------------------- #

def test_c1_default_config_backward_compatible():
    """use_mech272_routing_consumer defaults False."""
    from ree_core.utils.config import REEConfig

    cfg = REEConfig()
    assert cfg.use_mech272_routing_consumer is False

    cfg2 = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    assert cfg2.use_mech272_routing_consumer is False


def test_c2_consumer_off_routing_on_writes_at_full_strength():
    """Routing ON + consumer OFF: schema writes at anchor_weight=1.0 (bit-identical)."""
    agent = _build_agent(routing=True, routing_consumer=False, sws_anchor=0.6)
    assert agent.sleep_loop.use_mech272_routing_consumer is False
    metrics = _run_cycle_with_buffer(agent)
    # When consumer is off, sws_anchor_weight_applied must be 1.0
    applied = metrics.get("sws_anchor_weight_applied", 1.0)
    assert applied == 1.0, f"Expected 1.0 but got {applied}"


def test_c3_consumer_on_routing_on_writes_at_sws_anchor_weight():
    """Routing ON + consumer ON: schema writes at mech272_sws_anchor_weight."""
    sws_anchor = 0.6
    agent = _build_agent(routing=True, routing_consumer=True, sws_anchor=sws_anchor)
    assert agent.sleep_loop.use_mech272_routing_consumer is True
    metrics = _run_cycle_with_buffer(agent)
    applied = metrics.get("sws_anchor_weight_applied", 1.0)
    assert abs(applied - sws_anchor) < 1e-6, (
        f"Expected sws_anchor_weight_applied~={sws_anchor} but got {applied}"
    )


def test_c4_consumer_on_no_routing_gate_falls_back_to_1():
    """Consumer ON but no routing gate -> mean_anchor=1.0 fallback."""
    agent = _build_agent(routing=False, routing_consumer=True, sws_anchor=0.6)
    # No routing gate even when consumer flag is on
    assert agent.sleep_loop.routing_gate is None
    metrics = _run_cycle_with_buffer(agent)
    applied = metrics.get("sws_anchor_weight_applied", 1.0)
    assert applied == 1.0, f"Expected 1.0 fallback but got {applied}"


def test_c5_sleep_loop_manager_constructor_propagates_flag():
    """SleepLoopManager.__init__ stores use_mech272_routing_consumer correctly."""
    from ree_core.sleep.phase_manager import SleepLoopManager

    mgr_off = SleepLoopManager(use_mech272_routing_consumer=False)
    assert mgr_off.use_mech272_routing_consumer is False

    mgr_on = SleepLoopManager(use_mech272_routing_consumer=True)
    assert mgr_on.use_mech272_routing_consumer is True


def test_c6_agent_init_passes_flag_to_sleep_loop():
    """REEAgent passes use_mech272_routing_consumer from config to SleepLoopManager."""
    agent_off = _build_agent(routing=True, routing_consumer=False)
    assert agent_off.sleep_loop.use_mech272_routing_consumer is False

    agent_on = _build_agent(routing=True, routing_consumer=True)
    assert agent_on.sleep_loop.use_mech272_routing_consumer is True
