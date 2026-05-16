"""Contract tests for sleep-aggregation cluster Phase E (MECH-273).

See REE_assembly/docs/architecture/sleep_aggregation_cluster.md.

Phase E lands the self-model writeback. The SelfModelAggregator is a
SUBCLASS of MECH-275 BayesianAggregator specialised on the SD-003
causal_sig posterior in the "self" domain. It exposes
offline_gradient_pass(e2_harm_s, replayed_regions, n_steps) -- the
SINGLE EXPLICIT EXCEPTION to MECH-094 simulation_mode "no parameter
writes" (gated to E2_harm_s parameters only).

Phase E also drives StalenessAccumulator.partial_decay on the regions
replayed during the cycle (decay_factor default 0.5). SHY normalisation
is explicitly out of V3 scope.

Guarantees enforced:
  E1. Module + classes importable without side effects.
  E2. Default REEConfig has use_mech273_self_model=False with sub-knob
      defaults (offline_lr_scale=0.1, offline_n_steps=100,
      partial_decay_factor=0.5).
  E3. Master switch OFF: agent.sleep_self_model_aggregator is None.
  E4. Master switch ON (with use_e2_harm_s_forward=True): aggregator
      exists, starts empty, inherits parent BayesianAggregator state.
  E5. SelfModelAggregator inherits parent update/snapshot semantics on
      the "self" domain (subclass relationship; conjugate update fires).
  E6. offline_gradient_pass with n_steps<=0 short-circuits to a no-op;
      with no replayed regions returns zero-loss diagnostics.
  E7. offline_gradient_pass with replayed regions reduces MSE loss
      across steps and updates E2_harm_s parameters (MECH-094 exception
      scoped to e2_harm_s).
  E8. SleepLoopManager.force_cycle drives WRITEBACK; metrics include
      mech273_writeback_* keys and partial_decay diagnostics fire when
      a staleness accumulator is present and replayed regions are
      non-empty.
  E9. Bit-identical OFF: agent with use_sleep_loop=True but
      use_mech273_self_model=False has no mech273_* keys in cycle metrics.
  E10. partial_decay called only on replayed regions (StalenessAccumulator
       targeted-decay contract: untouched regions are not decayed).
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
    self_model: bool = True,
    e2_harm_s: bool = True,
    sws: bool = True,
    rem: bool = True,
    anchor_sets: bool = True,
    staleness: bool = True,
    draws: int = 6,
    sws_anchor: float = 0.6,
    sws_probe: float = 0.4,
    rem_anchor: float = 0.2,
    rem_probe: float = 0.8,
    self_model_n_steps: int = 5,
    self_model_decay_factor: float = 0.5,
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
        use_e2_harm_s_forward=e2_harm_s,
        use_mech273_self_model=self_model,
        mech273_offline_n_steps=self_model_n_steps,
        mech273_partial_decay_factor=self_model_decay_factor,
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

def test_e1_module_importable():
    from ree_core.sleep import (  # noqa: F401
        SelfModelAggregator,
        SelfModelAggregatorConfig,
    )
    from ree_core.sleep.self_model_aggregator import (  # noqa: F401
        SelfModelAggregator as _direct,
    )
    from ree_core.sleep.bayesian_aggregator import BayesianAggregator
    # SelfModelAggregator is a subclass of BayesianAggregator.
    assert issubclass(SelfModelAggregator, BayesianAggregator)


def test_e2_default_config_backward_compatible():
    from ree_core.utils.config import REEConfig

    cfg = REEConfig()
    assert getattr(cfg, "use_mech273_self_model", False) is False
    # Sub-knob defaults from the C6 design-doc commitment.
    assert cfg.mech273_offline_lr_scale == 0.1
    assert cfg.mech273_offline_n_steps == 100
    assert cfg.mech273_partial_decay_factor == 0.5


def test_e3_master_switch_off_no_instantiation():
    agent = _build_agent(self_model=False)
    assert agent.sleep_self_model_aggregator is None


def test_e4_master_switch_on_starts_empty():
    agent = _build_agent(self_model=True, e2_harm_s=True)
    agg = agent.sleep_self_model_aggregator
    assert agg is not None
    assert agent.e2_harm_s is not None
    # Inherits BayesianAggregator empty state.
    assert agg.n_updates == 0
    assert agg.n_posteriors == 0
    assert agg.last_snapshot is None
    # Phase E diagnostics start at zero.
    assert agg.n_offline_passes == 0
    assert agg.last_offline_loss == 0.0


def test_e5_inherits_parent_update_and_snapshot_on_self_domain():
    from ree_core.sleep import (
        RoutedEvent,
        SelfModelAggregator,
        SelfModelAggregatorConfig,
        SleepPhase,
    )

    agg = SelfModelAggregator(
        SelfModelAggregatorConfig(
            prior_mean=0.0,
            prior_variance=1.0,
            likelihood_variance=1.0,
            probe_gain=1.0,
        )
    )
    # Default domain is "self".
    assert tuple(agg.config.domains) == ("self",)
    routed = RoutedEvent(
        event=("fast", "0.1"),
        anchor_channel=0.6,
        probe_channel=0.4,
        phase=SleepPhase.SWS_ANALOG,
    )
    update = agg.update(routed, evidence=2.0, domain="self")
    assert update is not None
    assert update.delta_mean > 0.0
    posterior = agg.get_posterior("self", ("fast", "0.1"))
    assert posterior is not None
    assert 0.0 < posterior.mean < 2.0
    # Snapshot captures the SWS-only state.
    snap = agg.snapshot()
    assert ("self", ("fast", "0.1")) in snap


def test_e6_offline_gradient_pass_short_circuits():
    from ree_core.sleep import SelfModelAggregator, SelfModelAggregatorConfig
    from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward

    e2 = E2HarmSForward(E2HarmSConfig(use_e2_harm_s_forward=True))
    agg = SelfModelAggregator(SelfModelAggregatorConfig())

    # n_steps <= 0 -> no-op metrics.
    metrics = agg.offline_gradient_pass(
        e2_harm_s=e2, replayed_regions=[("fast", "0.1")], n_steps=0
    )
    assert metrics["mech273_writeback_n_steps"] == 0.0
    assert metrics["mech273_writeback_regions"] == 0.0
    assert metrics["mech273_writeback_sum_loss"] == 0.0
    assert agg.n_offline_passes == 1

    # No replayed regions -> zero-region metrics, n_steps=0.
    metrics = agg.offline_gradient_pass(
        e2_harm_s=e2, replayed_regions=[], n_steps=10
    )
    assert metrics["mech273_writeback_regions"] == 0.0
    assert metrics["mech273_writeback_n_steps"] == 0.0
    assert agg.n_offline_passes == 2


def test_e7_offline_gradient_pass_reduces_loss_and_updates_params():
    import torch

    from ree_core.sleep import (
        RoutedEvent,
        SelfModelAggregator,
        SelfModelAggregatorConfig,
        SleepPhase,
    )
    from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward

    torch.manual_seed(0)
    e2 = E2HarmSForward(
        E2HarmSConfig(use_e2_harm_s_forward=True, learning_rate=1e-2)
    )
    # Snapshot params for delta check.
    params_before = [p.detach().clone() for p in e2.parameters()]

    agg = SelfModelAggregator(
        SelfModelAggregatorConfig(
            prior_mean=0.0,
            prior_variance=1.0,
            likelihood_variance=1.0,
            probe_gain=1.0,
        )
    )
    # Seed a posterior with a non-zero target mean on the "self" domain.
    routed = RoutedEvent(
        event=("fast", "0.1"),
        anchor_channel=0.6,
        probe_channel=1.0,
        phase=SleepPhase.SWS_ANALOG,
    )
    for _ in range(5):
        agg.update(routed, evidence=3.0, domain="self")

    # Take a snapshot so use_snapshot=True path is exercised.
    agg.snapshot()
    metrics = agg.offline_gradient_pass(
        e2_harm_s=e2,
        replayed_regions=[("fast", "0.1")],
        n_steps=20,
        domain="self",
        use_snapshot=True,
    )
    assert metrics["mech273_writeback_regions"] == 1.0
    assert metrics["mech273_writeback_n_steps"] == 20.0
    assert metrics["mech273_writeback_sum_loss"] > 0.0
    assert agg.n_offline_passes == 1
    # Loss should be a finite positive number (mean-loss not NaN).
    assert metrics["mech273_writeback_mean_loss"] > 0.0

    # At least one parameter has moved (MECH-094 exception is scoped to
    # e2_harm_s, and it IS being updated here).
    moved = False
    for p_before, p_after in zip(params_before, e2.parameters()):
        if not torch.allclose(p_before, p_after.detach()):
            moved = True
            break
    assert moved, "E2_harm_s parameters were not updated by offline_gradient_pass"


def test_e8_force_cycle_drives_writeback_and_emits_metrics():
    agent = _build_agent(
        self_model=True,
        e2_harm_s=True,
        draws=4,
        self_model_n_steps=3,
    )
    _install_anchors(agent, scale="fast", segment_ids=("0.1", "0.2", "0.3"))
    metrics = agent.sleep_loop.force_cycle(agent)
    assert metrics is not None
    # Writeback ran -- at least the cycle-summary keys are present.
    assert "mech273_writeback_regions" in metrics
    assert "mech273_writeback_n_steps" in metrics
    assert "mech273_writeback_sum_loss" in metrics
    assert "mech273_writeback_mean_loss" in metrics
    # Cumulative diagnostics from get_metrics() also present.
    assert "mech273_n_offline_passes" in metrics
    assert metrics["mech273_n_offline_passes"] >= 1.0
    # Replayed regions touched the staleness accumulator -- partial decay
    # diagnostics fire when both flags are on.
    assert "mech273_partial_decay_n_regions" in metrics
    assert metrics["mech273_partial_decay_factor"] == 0.5


def test_e9_bit_identical_off_no_mech273_keys_in_metrics():
    agent = _build_agent(self_model=False, e2_harm_s=False, draws=4)
    _install_anchors(agent, scale="fast", segment_ids=("0.1",))
    assert agent.sleep_self_model_aggregator is None
    metrics = agent.sleep_loop.force_cycle(agent)
    for key in metrics:
        assert not key.startswith("mech273_"), (
            f"Phase E OFF leaked metric key: {key}"
        )


def test_e10_partial_decay_targets_only_replayed_regions():
    """StalenessAccumulator.partial_decay only decays the supplied regions."""
    from ree_core.hippocampal.staleness_accumulator import StalenessAccumulator
    from ree_core.utils.config import StalenessAccumulatorConfig

    acc = StalenessAccumulator(
        StalenessAccumulatorConfig(
            leak_factor=1.0,
            attribution_mode="equal",
            staleness_clip=1.0,
            drop_epsilon=1e-12,
        )
    )
    # Seed the accumulator with two regions at known staleness.
    acc._staleness[("fast", "0.1")] = 0.8
    acc._staleness[("fast", "0.2")] = 0.4

    # Decay only the replayed region.
    n_decayed = acc.partial_decay(
        [("fast", "0.1")], decay_factor=0.5
    )
    assert n_decayed == 1
    # Replayed region halved; untouched region unchanged.
    assert abs(acc.get(("fast", "0.1")) - 0.4) < 1e-9
    assert abs(acc.get(("fast", "0.2")) - 0.4) < 1e-9

    # Region absent from the accumulator is silently skipped.
    n_decayed = acc.partial_decay(
        [("slow", "absent")], decay_factor=0.5
    )
    assert n_decayed == 0

    # Duplicates in the input list are deduped.
    n_decayed = acc.partial_decay(
        [("fast", "0.2"), ("fast", "0.2")], decay_factor=0.5
    )
    assert n_decayed == 1
    assert abs(acc.get(("fast", "0.2")) - 0.2) < 1e-9


# ---------------------------------------------------------------------------- #
# GAP-4 / MECH-273 real-replay-buffer contracts                               #
# ---------------------------------------------------------------------------- #

def test_e11_harm_replay_buffer_real_tuple_path_updates_params():
    """Real (z_harm_s, action) tuples [1, z_dim] from the buffer are sampled and
    used in offline_gradient_pass, resulting in E2_harm_s parameter updates."""
    import torch

    from ree_core.sleep import (
        RoutedEvent,
        SelfModelAggregator,
        SelfModelAggregatorConfig,
        SleepPhase,
    )
    from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward

    torch.manual_seed(42)
    e2 = E2HarmSForward(
        E2HarmSConfig(use_e2_harm_s_forward=True, learning_rate=1e-2)
    )
    params_before = [p.detach().clone() for p in e2.parameters()]

    agg = SelfModelAggregator(
        SelfModelAggregatorConfig(
            prior_mean=0.0, prior_variance=1.0, likelihood_variance=1.0, probe_gain=1.0
        )
    )
    routed = RoutedEvent(
        event=("fast", "0.1"),
        anchor_channel=0.6,
        probe_channel=1.0,
        phase=SleepPhase.SWS_ANALOG,
    )
    for _ in range(5):
        agg.update(routed, evidence=3.0, domain="self")
    agg.snapshot()

    z_dim = int(e2.config.z_harm_dim)
    a_dim = int(e2.config.action_dim)
    # Simulate sense()-produced tensors: shape [1, z_dim] and [1, a_dim].
    buf = [
        (
            torch.randn(1, z_dim),
            torch.zeros(1, a_dim).scatter_(1, torch.tensor([[i % a_dim]]), 1.0),
        )
        for i in range(3)
    ]

    metrics = agg.offline_gradient_pass(
        e2_harm_s=e2,
        replayed_regions=[("fast", "0.1")],
        n_steps=10,
        domain="self",
        use_snapshot=True,
        harm_replay_buffer=buf,
    )
    assert metrics["mech273_writeback_regions"] == 1.0
    assert metrics["mech273_writeback_n_steps"] == 10.0
    assert metrics["mech273_writeback_sum_loss"] > 0.0

    moved = any(
        not torch.allclose(pb, pa.detach())
        for pb, pa in zip(params_before, e2.parameters())
    )
    assert moved, "E2_harm_s parameters must move with real replay buffer"


def test_e12_harm_replay_buffer_empty_uses_synthetic_fallback():
    """An empty harm_replay_buffer triggers the synthetic zeros/round-robin
    one-hot fallback path and completes without error."""
    import torch

    from ree_core.sleep import (
        RoutedEvent,
        SelfModelAggregator,
        SelfModelAggregatorConfig,
        SleepPhase,
    )
    from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward

    torch.manual_seed(0)
    e2 = E2HarmSForward(E2HarmSConfig(use_e2_harm_s_forward=True))
    agg = SelfModelAggregator(
        SelfModelAggregatorConfig(
            prior_mean=0.0, prior_variance=1.0, likelihood_variance=1.0, probe_gain=1.0
        )
    )
    routed = RoutedEvent(
        event=("fast", "0.1"),
        anchor_channel=0.6,
        probe_channel=1.0,
        phase=SleepPhase.SWS_ANALOG,
    )
    for _ in range(5):
        agg.update(routed, evidence=2.0, domain="self")
    agg.snapshot()

    metrics = agg.offline_gradient_pass(
        e2_harm_s=e2,
        replayed_regions=[("fast", "0.1")],
        n_steps=5,
        harm_replay_buffer=[],
    )
    assert metrics["mech273_writeback_regions"] == 1.0
    assert metrics["mech273_writeback_n_steps"] == 5.0
    assert agg.n_offline_passes == 1


def test_e13_harm_replay_buffer_none_uses_synthetic_fallback():
    """None harm_replay_buffer (backward-compat default) triggers the synthetic
    zeros/round-robin one-hot fallback path."""
    import torch

    from ree_core.sleep import (
        RoutedEvent,
        SelfModelAggregator,
        SelfModelAggregatorConfig,
        SleepPhase,
    )
    from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward

    torch.manual_seed(1)
    e2 = E2HarmSForward(E2HarmSConfig(use_e2_harm_s_forward=True))
    agg = SelfModelAggregator(
        SelfModelAggregatorConfig(
            prior_mean=0.0, prior_variance=1.0, likelihood_variance=1.0, probe_gain=1.0
        )
    )
    routed = RoutedEvent(
        event=("fast", "0.1"),
        anchor_channel=0.6,
        probe_channel=1.0,
        phase=SleepPhase.SWS_ANALOG,
    )
    for _ in range(5):
        agg.update(routed, evidence=2.0, domain="self")
    agg.snapshot()

    # Omit harm_replay_buffer entirely (defaults to None).
    metrics = agg.offline_gradient_pass(
        e2_harm_s=e2,
        replayed_regions=[("fast", "0.1")],
        n_steps=5,
    )
    assert metrics["mech273_writeback_regions"] == 1.0
    assert metrics["mech273_writeback_n_steps"] == 5.0
    assert agg.n_offline_passes == 1


def test_e14_harm_replay_buffer_smaller_than_n_regions_samples_with_replacement():
    """When len(harm_replay_buffer) < n_regions, random.choices with replacement
    satisfies the k=n_regions demand without error."""
    import torch

    from ree_core.sleep import (
        RoutedEvent,
        SelfModelAggregator,
        SelfModelAggregatorConfig,
        SleepPhase,
    )
    from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward

    torch.manual_seed(7)
    e2 = E2HarmSForward(E2HarmSConfig(use_e2_harm_s_forward=True))
    agg = SelfModelAggregator(
        SelfModelAggregatorConfig(
            prior_mean=0.0, prior_variance=1.0, likelihood_variance=1.0, probe_gain=1.0
        )
    )
    # Seed 4 distinct regions so n_regions = 4 at writeback time.
    for seg in ("0.1", "0.2", "0.3", "0.4"):
        routed = RoutedEvent(
            event=("fast", seg),
            anchor_channel=0.6,
            probe_channel=1.0,
            phase=SleepPhase.SWS_ANALOG,
        )
        for _ in range(3):
            agg.update(routed, evidence=2.0, domain="self")
    agg.snapshot()

    z_dim = int(e2.config.z_harm_dim)
    a_dim = int(e2.config.action_dim)
    # Only 2 entries in the buffer; n_regions will be 4.
    # random.choices(buf, k=4) samples with replacement -- must not raise.
    buf = [
        (
            torch.randn(1, z_dim),
            torch.zeros(1, a_dim).scatter_(1, torch.tensor([[i]]), 1.0),
        )
        for i in range(2)
    ]

    metrics = agg.offline_gradient_pass(
        e2_harm_s=e2,
        replayed_regions=[("fast", "0.1"), ("fast", "0.2"), ("fast", "0.3"), ("fast", "0.4")],
        n_steps=5,
        harm_replay_buffer=buf,
    )
    assert metrics["mech273_writeback_regions"] == 4.0
    assert metrics["mech273_writeback_n_steps"] == 5.0
    assert metrics["mech273_writeback_sum_loss"] > 0.0
