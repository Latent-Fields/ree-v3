"""Contract tests for sleep-aggregation cluster Phase C (MECH-272).

See REE_assembly/docs/architecture/sleep_aggregation_cluster.md.

Phase C lands the state-conditioned RoutingGate. The gate is a NO-OP
CONSUMER: it tags replay events with (anchor_channel, probe_channel)
weights that flip across phases, but no downstream consumer (E1
ContextMemory anchor consumer / Phase D Bayesian aggregator probe
consumer) is wired yet, so routed events land as diagnostics on
SleepCycleState.last_metrics.

Guarantees enforced:
  C1. Module + class importable without side effects.
  C2. Default REEConfig has use_mech272_routing=False (backward-compat).
  C3. Master switch OFF: REEAgent.sleep_routing_gate is None.
  C4. Master switch ON: REEAgent.sleep_routing_gate exists at the
      WAKING row by default.
  C5. set_phase(SWS_ANALOG / SLEEP_ENTRY) -> SWS row weights.
  C6. set_phase(REM_ANALOG / PHASE_SWITCH) -> REM row weights.
  C7. route() emits a RoutedEvent carrying the current channel weights;
      the underlying event is preserved.
  C8. SleepLoopManager.force_cycle drives the gate through SWS_ANALOG
      then PHASE_SWITCH (when rem_enabled), then parks at WAKING.
      mech272_n_routed reflects the draws.
  C9. Bit-identical OFF: agent with use_sleep_loop=True but
      use_mech272_routing=False has sleep_routing_gate=None and the
      sleep cycle metrics carry no mech272_* keys.
  C10. Anchor-only ablation (probe_weight=0 across phases) and
       probe-only ablation (anchor_weight=0 across phases) round-trip
       through the gate without raising.
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
    sws: bool = True,
    rem: bool = True,
    anchor_sets: bool = True,
    staleness: bool = True,
    draws: int = 8,
    sws_anchor: float = 0.6,
    sws_probe: float = 0.4,
    rem_anchor: float = 0.2,
    rem_probe: float = 0.8,
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
        mech272_rem_anchor_weight=rem_anchor,
        mech272_rem_probe_weight=rem_probe,
    )
    cfg.sws_enabled = sws
    cfg.rem_enabled = rem
    return REEAgent(cfg)


def _install_anchors(agent, *, scale: str, segment_ids):
    import torch

    anchor_set = agent.hippocampal.anchor_set
    assert anchor_set is not None
    anchors = []
    for i, seg in enumerate(segment_ids):
        z = torch.randn(1, 32) * (i + 1)
        a = anchor_set.write_anchor(
            scale=scale,
            segment_id=str(seg),
            stream_mixture=(f"stream_{seg}",),
            z_world=z,
        )
        anchors.append(a)
    return anchors


# ---------------------------------------------------------------------------- #
# Contracts                                                                    #
# ---------------------------------------------------------------------------- #

def test_c1_module_importable():
    from ree_core.sleep import RoutedEvent, RoutingGate, RoutingGateConfig  # noqa: F401
    from ree_core.sleep.routing_gate import RoutingGate as _direct  # noqa: F401


def test_c2_default_config_backward_compatible():
    from ree_core.utils.config import REEConfig

    cfg = REEConfig()
    assert getattr(cfg, "use_mech272_routing", False) is False
    # Defaults present even when master switch is off; match design-doc table.
    assert cfg.mech272_waking_anchor_weight == 1.0
    assert cfg.mech272_waking_probe_weight == 0.0
    assert cfg.mech272_sws_anchor_weight == 0.6
    assert cfg.mech272_sws_probe_weight == 0.4
    assert cfg.mech272_rem_anchor_weight == 0.2
    assert cfg.mech272_rem_probe_weight == 0.8


def test_c3_master_switch_off_no_instantiation():
    agent = _build_agent(routing=False)
    assert agent.sleep_routing_gate is None


def test_c4_master_switch_on_parks_at_waking_row():
    agent = _build_agent(routing=True)
    gate = agent.sleep_routing_gate
    assert gate is not None
    from ree_core.sleep import SleepPhase

    assert gate.phase == SleepPhase.WAKING
    assert gate.anchor_weight == 1.0
    assert gate.probe_weight == 0.0


def test_c5_set_phase_sws_row():
    from ree_core.sleep import RoutingGate, RoutingGateConfig, SleepPhase

    gate = RoutingGate(RoutingGateConfig())
    gate.set_phase(SleepPhase.SWS_ANALOG)
    assert gate.anchor_weight == 0.6
    assert gate.probe_weight == 0.4
    # SLEEP_ENTRY is the transition token preceding SWS; collapses to SWS row.
    gate.set_phase(SleepPhase.WAKING)
    gate.set_phase(SleepPhase.SLEEP_ENTRY)
    assert gate.anchor_weight == 0.6
    assert gate.probe_weight == 0.4


def test_c6_set_phase_rem_row():
    from ree_core.sleep import RoutingGate, RoutingGateConfig, SleepPhase

    gate = RoutingGate(RoutingGateConfig())
    gate.set_phase(SleepPhase.REM_ANALOG)
    assert gate.anchor_weight == 0.2
    assert gate.probe_weight == 0.8
    # PHASE_SWITCH is the transition token preceding REM; collapses to REM row.
    gate.set_phase(SleepPhase.WAKING)
    gate.set_phase(SleepPhase.PHASE_SWITCH)
    assert gate.anchor_weight == 0.2
    assert gate.probe_weight == 0.8


def test_c7_route_emits_routed_event():
    from ree_core.sleep import RoutedEvent, RoutingGate, RoutingGateConfig, SleepPhase

    gate = RoutingGate(RoutingGateConfig())
    gate.set_phase(SleepPhase.SWS_ANALOG)
    sentinel = ("anchor-stub", 42)
    routed = gate.route(sentinel)
    assert isinstance(routed, RoutedEvent)
    assert routed.event is sentinel
    assert routed.anchor_channel == 0.6
    assert routed.probe_channel == 0.4
    assert routed.phase == SleepPhase.SWS_ANALOG


def test_c8_force_cycle_drives_sws_then_rem_then_waking():
    from ree_core.sleep import SleepPhase

    agent = _build_agent(routing=True, draws=5, sws=True, rem=True)
    _install_anchors(agent, scale="fast", segment_ids=("0.1", "0.2", "0.3"))
    metrics = agent.sleep_loop.force_cycle(agent)
    assert metrics is not None
    # 5 SWS draws + 5 REM re-routes (same draws, REM destination)
    assert metrics.get("mech272_n_routed") == 10.0
    assert metrics.get("mech272_n_routed_sws") == 5.0
    assert metrics.get("mech272_n_routed_rem") == 5.0
    # After cycle the gate parks at the WAKING row.
    gate = agent.sleep_routing_gate
    assert gate.phase == SleepPhase.WAKING
    assert gate.anchor_weight == 1.0
    assert gate.probe_weight == 0.0


def test_c9_bit_identical_off_no_mech272_keys_in_metrics():
    agent = _build_agent(routing=False, draws=4)
    _install_anchors(agent, scale="fast", segment_ids=("0.1",))
    assert agent.sleep_routing_gate is None
    metrics = agent.sleep_loop.force_cycle(agent)
    for key in metrics:
        assert not key.startswith("mech272_"), (
            f"Phase C OFF leaked metric key: {key}"
        )


def test_c10_ablation_arms_round_trip():
    """Anchor-only and probe-only ablations exercise the same plumbing."""
    from ree_core.sleep import SleepPhase

    # Anchor-only: probe weights zeroed across SWS/REM.
    agent_a = _build_agent(
        routing=True,
        draws=3,
        sws_anchor=1.0,
        sws_probe=0.0,
        rem_anchor=1.0,
        rem_probe=0.0,
    )
    _install_anchors(agent_a, scale="fast", segment_ids=("0.1",))
    metrics_a = agent_a.sleep_loop.force_cycle(agent_a)
    assert metrics_a["mech272_mean_probe_channel"] == 0.0
    assert metrics_a["mech272_mean_anchor_channel"] == 1.0

    # Probe-only: anchor weights zeroed across SWS/REM.
    agent_p = _build_agent(
        routing=True,
        draws=3,
        sws_anchor=0.0,
        sws_probe=1.0,
        rem_anchor=0.0,
        rem_probe=1.0,
    )
    _install_anchors(agent_p, scale="fast", segment_ids=("0.1",))
    metrics_p = agent_p.sleep_loop.force_cycle(agent_p)
    assert metrics_p["mech272_mean_anchor_channel"] == 0.0
    assert metrics_p["mech272_mean_probe_channel"] == 1.0
