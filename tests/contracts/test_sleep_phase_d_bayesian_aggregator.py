"""Contract tests for sleep-aggregation cluster Phase D (MECH-275).

See REE_assembly/docs/architecture/sleep_aggregation_cluster.md.

Phase D lands the general Bayesian aggregator. The aggregator is a
NO-OP CONSUMER for V3: posterior updates are emitted as diagnostic
metrics on SleepCycleState.last_metrics. The Phase E self-model
writeback (MECH-273) is the first downstream consumer; for the "place"
domain there is no V3 consumer beyond metrics.

Guarantees enforced:
  D1. Module + classes importable without side effects.
  D2. Default REEConfig has use_mech275_aggregator=False (backward-compat).
  D3. Master switch OFF: REEAgent.sleep_bayesian_aggregator is None.
  D4. Master switch ON: REEAgent.sleep_bayesian_aggregator exists and
      starts empty (no posteriors, no snapshots).
  D5. Conjugate update: posterior precision strictly increases on a
      single update; posterior mean moves toward the observation.
  D6. probe_channel * probe_gain <= 0 -> no update (counted as skipped).
  D7. snapshot() captures a deep copy and applies decay to live posterior;
      the captured snapshot is invariant under subsequent updates.
  D8. SleepLoopManager.force_cycle drives the aggregator through SWS
      (probe_channel=0.4) and REM (probe_channel=0.8) re-routes; metrics
      reflect both passes and exactly one snapshot fires per cycle.
  D9. Bit-identical OFF: agent with use_sleep_loop=True but
      use_mech275_aggregator=False has sleep_bayesian_aggregator=None
      and the sleep cycle metrics carry no mech275_* keys.
  D10. Anchor-only ablation (probe_weight=0 across phases) drives N
       routed events through the aggregator without firing any update;
       probe-only ablation (anchor_weight=0) updates on every routed
       event in both passes.
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
    aggregator: bool = True,
    sws: bool = True,
    rem: bool = True,
    anchor_sets: bool = True,
    staleness: bool = True,
    draws: int = 8,
    sws_anchor: float = 0.6,
    sws_probe: float = 0.4,
    rem_anchor: float = 0.2,
    rem_probe: float = 0.8,
    prior_mean: float = 0.0,
    prior_variance: float = 1.0,
    likelihood_variance: float = 1.0,
    decay_factor: float = 1.0,
    probe_gain: float = 1.0,
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
        use_mech275_aggregator=aggregator,
        mech275_prior_mean=prior_mean,
        mech275_prior_variance=prior_variance,
        mech275_likelihood_variance=likelihood_variance,
        mech275_decay_factor=decay_factor,
        mech275_probe_gain=probe_gain,
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

def test_d1_module_importable():
    from ree_core.sleep import (  # noqa: F401
        BayesianAggregator,
        BayesianAggregatorConfig,
        GaussianPosterior,
        PosteriorUpdate,
    )
    from ree_core.sleep.bayesian_aggregator import (  # noqa: F401
        BayesianAggregator as _direct,
    )


def test_d2_default_config_backward_compatible():
    from ree_core.utils.config import REEConfig

    cfg = REEConfig()
    assert getattr(cfg, "use_mech275_aggregator", False) is False
    # Sub-knobs present with the defaults from the design-doc table.
    assert cfg.mech275_prior_mean == 0.0
    assert cfg.mech275_prior_variance == 1.0
    assert cfg.mech275_likelihood_variance == 1.0
    assert cfg.mech275_decay_factor == 1.0
    assert cfg.mech275_probe_gain == 1.0
    assert tuple(cfg.mech275_domains) == ("place",)


def test_d3_master_switch_off_no_instantiation():
    agent = _build_agent(aggregator=False)
    assert agent.sleep_bayesian_aggregator is None


def test_d4_master_switch_on_starts_empty():
    agent = _build_agent(aggregator=True)
    agg = agent.sleep_bayesian_aggregator
    assert agg is not None
    assert agg.n_updates == 0
    assert agg.n_posteriors == 0
    assert agg.last_snapshot is None


def test_d5_conjugate_update_moves_mean_and_increases_precision():
    from ree_core.sleep import (
        BayesianAggregator,
        BayesianAggregatorConfig,
        RoutedEvent,
        SleepPhase,
    )

    agg = BayesianAggregator(
        BayesianAggregatorConfig(
            prior_mean=0.0,
            prior_variance=1.0,
            likelihood_variance=1.0,
            probe_gain=1.0,
        )
    )
    routed = RoutedEvent(
        event=("fast", "0.1"),
        anchor_channel=0.6,
        probe_channel=0.4,
        phase=SleepPhase.SWS_ANALOG,
    )
    update = agg.update(routed, evidence=2.0, domain="place")
    assert update is not None
    assert update.delta_mean > 0.0  # pulled toward 2.0
    assert update.delta_variance < 0.0  # variance shrinks
    posterior = agg.get_posterior("place", ("fast", "0.1"))
    assert posterior is not None
    assert posterior.n == 1
    assert 0.0 < posterior.mean < 2.0
    assert posterior.variance < 1.0


def test_d6_zero_probe_weight_skips_update():
    from ree_core.sleep import (
        BayesianAggregator,
        BayesianAggregatorConfig,
        RoutedEvent,
        SleepPhase,
    )

    agg = BayesianAggregator(BayesianAggregatorConfig(probe_gain=1.0))
    routed = RoutedEvent(
        event=("fast", "0.1"),
        anchor_channel=1.0,
        probe_channel=0.0,
        phase=SleepPhase.WAKING,
    )
    update = agg.update(routed, evidence=5.0, domain="place")
    assert update is None
    assert agg.n_updates == 0
    assert agg.n_posteriors == 0
    metrics = agg.get_metrics()
    assert metrics["mech275_n_skipped_zero_probe"] == 1.0

    # probe_gain=0 also skips even with non-zero probe_channel.
    agg2 = BayesianAggregator(BayesianAggregatorConfig(probe_gain=0.0))
    routed2 = RoutedEvent(
        event=("fast", "0.1"),
        anchor_channel=0.6,
        probe_channel=0.4,
        phase=SleepPhase.SWS_ANALOG,
    )
    assert agg2.update(routed2, evidence=5.0, domain="place") is None
    assert agg2.n_updates == 0


def test_d7_snapshot_captures_deep_copy_and_decays_live():
    from ree_core.sleep import (
        BayesianAggregator,
        BayesianAggregatorConfig,
        RoutedEvent,
        SleepPhase,
    )

    agg = BayesianAggregator(
        BayesianAggregatorConfig(
            prior_variance=1.0,
            likelihood_variance=1.0,
            probe_gain=1.0,
            decay_factor=0.5,  # variance multiplied by 1/0.5 = 2 on snapshot
        )
    )
    routed = RoutedEvent(
        event=("fast", "0.1"),
        anchor_channel=0.6,
        probe_channel=0.4,
        phase=SleepPhase.SWS_ANALOG,
    )
    agg.update(routed, evidence=2.0, domain="place")
    live_before = agg.get_posterior("place", ("fast", "0.1"))
    assert live_before is not None
    snap = agg.snapshot()
    snap_post = snap[("place", ("fast", "0.1"))]
    assert snap_post.mean == live_before.mean
    assert snap_post.variance == live_before.variance

    # Decay applied to LIVE posterior (variance grows because precision
    # multiplied by decay < 1.0).
    live_after = agg.get_posterior("place", ("fast", "0.1"))
    assert live_after is not None
    assert live_after.variance > live_before.variance

    # Subsequent updates do NOT change the captured snapshot.
    agg.update(routed, evidence=10.0, domain="place")
    snap_again = agg.last_snapshot
    assert snap_again is not None
    assert snap_again[("place", ("fast", "0.1"))].mean == snap_post.mean
    assert snap_again[("place", ("fast", "0.1"))].variance == snap_post.variance

    metrics = agg.get_metrics()
    assert metrics["mech275_n_snapshots"] == 1.0


def test_d8_force_cycle_runs_sws_then_snapshot_then_rem():
    agent = _build_agent(aggregator=True, draws=5, sws=True, rem=True)
    _install_anchors(agent, scale="fast", segment_ids=("0.1", "0.2", "0.3"))
    metrics = agent.sleep_loop.force_cycle(agent)
    assert metrics is not None
    # 5 SWS draws + 5 REM re-routes -- both fire the aggregator since
    # sws_probe=0.4 and rem_probe=0.8 are both > 0.
    assert metrics["mech275_n_updates"] == 10.0
    assert metrics["mech275_n_snapshots"] == 1.0
    assert metrics["mech275_n_skipped_zero_probe"] == 0.0
    assert metrics["mech275_n_posteriors"] >= 1.0
    # Sum of weights = 5*0.4 + 5*0.8 = 6.0 with probe_gain=1.0.
    assert abs(metrics["mech275_sum_weight"] - 6.0) < 1e-6


def test_d9_bit_identical_off_no_mech275_keys_in_metrics():
    agent = _build_agent(aggregator=False, draws=4)
    _install_anchors(agent, scale="fast", segment_ids=("0.1",))
    assert agent.sleep_bayesian_aggregator is None
    metrics = agent.sleep_loop.force_cycle(agent)
    for key in metrics:
        assert not key.startswith("mech275_"), (
            f"Phase D OFF leaked metric key: {key}"
        )


def test_d10_ablation_arms_round_trip():
    """Anchor-only and probe-only ablations both run the aggregator path."""

    # Anchor-only: probe weights zero across SWS/REM -> no updates fire,
    # but every routed event is observed by the aggregator's diagnostics.
    agent_a = _build_agent(
        aggregator=True,
        draws=4,
        sws_anchor=1.0,
        sws_probe=0.0,
        rem_anchor=1.0,
        rem_probe=0.0,
    )
    _install_anchors(agent_a, scale="fast", segment_ids=("0.1",))
    metrics_a = agent_a.sleep_loop.force_cycle(agent_a)
    assert metrics_a["mech275_n_updates"] == 0.0
    # 4 SWS routed + 4 REM re-routed = 8 zero-probe events skipped.
    assert metrics_a["mech275_n_skipped_zero_probe"] == 8.0
    assert metrics_a["mech275_n_posteriors"] == 0.0
    # Snapshot still fires at PHASE_SWITCH.
    assert metrics_a["mech275_n_snapshots"] == 1.0

    # Probe-only: anchor weights zero, probe=1 -> every routed event
    # updates the posterior.
    agent_p = _build_agent(
        aggregator=True,
        draws=3,
        sws_anchor=0.0,
        sws_probe=1.0,
        rem_anchor=0.0,
        rem_probe=1.0,
    )
    _install_anchors(agent_p, scale="fast", segment_ids=("0.1",))
    metrics_p = agent_p.sleep_loop.force_cycle(agent_p)
    assert metrics_p["mech275_n_updates"] == 6.0  # 3 SWS + 3 REM
    assert metrics_p["mech275_n_skipped_zero_probe"] == 0.0
    # Sum of weights = 3*1 + 3*1 = 6.
    assert abs(metrics_p["mech275_sum_weight"] - 6.0) < 1e-6
