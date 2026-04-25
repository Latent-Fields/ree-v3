"""Contract tests for sleep-aggregation cluster Phase B (MECH-285).

See REE_assembly/docs/architecture/sleep_aggregation_cluster.md.

Phase B is the offline sleep-priority arm of the V_s invalidation runtime
dual-readout (MECH-284 is the online arm). At sleep entry, the
SleepReplaySampler freezes a snapshot of the StalenessAccumulator and
draws N seeds from the AnchorSet broad pool (active + inactive, dual-trace
preserved per Bouton 2004) with probability proportional to softmax over
frozen staleness.

Phase B is a NO-OP CONSUMER. Draws land as diagnostics on
SleepCycleState.last_metrics; no downstream consumer (routing / aggregator
/ writeback) sees them yet.

Guarantees enforced:
  C1. Module + class importable without side effects.
  C2. Default REEConfig has use_mech285_sampler=False (backward-compat).
  C3. Master switch OFF: REEAgent.sleep_replay_sampler is None.
  C4. Master switch ON without anchor_set: sampler is None silently
      (Phase B requires MECH-269 Phase 2 ii).
  C5. Master switch ON with anchor_set + accumulator: sampler exists
      and runs draws when sleep_loop fires.
  C6. Broad pool: draw() returns from active + inactive anchors
      (dual-trace preserved).
  C7. Staleness skews softmax: regions with higher staleness are drawn
      more frequently than regions with lower staleness (statistical).
  C8. Uniform fallback when no accumulator (allow_uniform_fallback=True).
  C9. Snapshot frozen at SLEEP_ENTRY: subsequent staleness writes do not
      affect in-cycle draw distribution.
  C10. Sleep cycle metrics include mech285_* diagnostics when sampler on.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #

def _build_agent(
    *,
    sleep_loop: bool = True,
    sampler: bool = True,
    sws: bool = True,
    rem: bool = True,
    anchor_sets: bool = True,
    staleness: bool = True,
    draws: int = 8,
    temperature: float = 1.0,
    allow_uniform_fallback: bool = True,
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
        mech285_temperature=temperature,
        mech285_allow_uniform_fallback=allow_uniform_fallback,
        use_anchor_sets=anchor_sets,
        use_staleness_accumulator=staleness,
    )
    cfg.sws_enabled = sws
    cfg.rem_enabled = rem
    return REEAgent(cfg)


def _install_anchors(agent, *, scale: str, segment_ids, mark_inactive_ids=()):
    """Install several anchors with distinct stream_mixtures so each becomes
    a separate active entry. Optionally mark some inactive afterward.
    Returns the list of installed Anchors in install order.
    """
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
    # Mark some inactive AFTER all installs so the dual-trace is preserved.
    for seg in mark_inactive_ids:
        anchor_set.mark_inactive(scale, (f"stream_{seg}",))
    return anchors


# ---------------------------------------------------------------------------- #
# Contracts                                                                    #
# ---------------------------------------------------------------------------- #

def test_c1_module_importable():
    from ree_core.sleep import SleepReplaySampler  # noqa: F401
    from ree_core.sleep.replay_sampler import SleepReplaySampler as _direct  # noqa: F401


def test_c2_default_config_backward_compatible():
    from ree_core.utils.config import REEConfig

    cfg = REEConfig()
    assert getattr(cfg, "use_mech285_sampler", False) is False
    # Defaults present even when master switch is off.
    assert getattr(cfg, "mech285_draws_per_cycle", None) == 50
    assert getattr(cfg, "mech285_temperature", None) == 1.0
    assert getattr(cfg, "mech285_allow_uniform_fallback", None) is True


def test_c3_master_switch_off_no_instantiation():
    agent = _build_agent(sampler=False)
    assert agent.sleep_replay_sampler is None


def test_c4_no_anchor_set_no_sampler():
    """Phase B requires MECH-269 Phase 2 ii. Without anchor_set, no sampler."""
    agent = _build_agent(sampler=True, anchor_sets=False, staleness=False)
    assert agent.sleep_replay_sampler is None


def test_c5_sampler_runs_in_cycle():
    agent = _build_agent(sampler=True, draws=5)
    assert agent.sleep_replay_sampler is not None
    _install_anchors(agent, scale="fast", segment_ids=("0.1", "0.2", "0.3"))
    metrics = agent.sleep_loop.force_cycle(agent)
    assert metrics is not None
    assert metrics.get("mech285_n_draws") == 5.0
    assert metrics.get("mech285_snapshot_size") is not None


def test_c6_broad_pool_includes_inactive():
    """Dual-trace preserved: inactive anchors are valid draw seeds."""
    agent = _build_agent(sampler=True, draws=200)
    install = _install_anchors(
        agent,
        scale="fast",
        segment_ids=("0.1", "0.2", "0.3"),
        mark_inactive_ids=("0.1",),
    )
    # Confirm broad-pool size includes active + inactive.
    pool = agent.hippocampal.anchor_set.all_with_dual_trace()
    assert len(pool) == len(install)
    # Confirm at least one is inactive.
    inactive_keys = {a.key for a in pool if not a.active}
    assert len(inactive_keys) >= 1
    # Run cycle; sampler should be able to draw from the broad pool with
    # the inactive seed represented at least once (uniform fallback ->
    # large draw count makes a hit overwhelmingly likely).
    agent.sleep_loop.force_cycle(agent)
    counts = agent.sleep_replay_sampler.draw_region_counts
    drawn_keys = set(counts.keys())
    inactive_region_keys = {(k[0], k[1]) for k in inactive_keys}
    assert drawn_keys & inactive_region_keys, (
        f"No inactive region drawn in 200 attempts: drawn={drawn_keys}, "
        f"inactive={inactive_region_keys}"
    )


def test_c7_staleness_skews_softmax():
    """Region with higher staleness is drawn more often than low-staleness one."""
    agent = _build_agent(sampler=True, draws=0, temperature=0.3)
    _install_anchors(agent, scale="fast", segment_ids=("hot", "cold"))
    # Inject staleness manually via the accumulator's internal map.
    acc = agent.hippocampal.staleness_accumulator
    assert acc is not None
    acc._staleness[("fast", "hot")] = 0.9
    acc._staleness[("fast", "cold")] = 0.0

    sampler = agent.sleep_replay_sampler
    sampler.freeze_snapshot()
    rng = np.random.default_rng(20260425)
    counts = Counter()
    N = 1000
    for _ in range(N):
        a = sampler.draw(rng)
        counts[(a.key[0], a.key[1])] += 1
    hot = counts[("fast", "hot")]
    cold = counts[("fast", "cold")]
    # T=0.3, staleness gap 0.9 -> exp(3) skew. Hot should dominate cold.
    assert hot > cold * 5, (
        f"Expected hot >> cold; got hot={hot}, cold={cold}, total={N}"
    )


def test_c8_uniform_fallback_no_accumulator():
    """When no StalenessAccumulator, sampler falls back to uniform draws."""
    from ree_core.hippocampal.anchor_set import AnchorSet
    from ree_core.sleep.replay_sampler import SleepReplaySampler
    from ree_core.utils.config import AnchorSetConfig
    import torch

    anchor_set = AnchorSet(AnchorSetConfig())
    for seg in ("a", "b", "c"):
        anchor_set.write_anchor(
            scale="fast",
            segment_id=seg,
            stream_mixture=(seg,),
            z_world=torch.randn(1, 32),
        )

    sampler = SleepReplaySampler(
        anchor_set=anchor_set,
        staleness_accumulator=None,
        temperature=1.0,
        allow_uniform_fallback=True,
    )
    sampler.freeze_snapshot()
    assert sampler.snapshot_is_uniform is True
    assert sampler.snapshot_size == 0

    rng = np.random.default_rng(20260425)
    counts = Counter()
    N = 600
    for _ in range(N):
        a = sampler.draw(rng)
        counts[(a.key[0], a.key[1])] += 1
    # Each region should be drawn roughly N/3 times. Allow generous slack.
    assert all(80 < counts[(k[0], k[1])] for k in [
        ("fast", "a"), ("fast", "b"), ("fast", "c")
    ]), f"Uniform draw bias: {counts}"


def test_c9_snapshot_is_frozen_at_entry():
    """In-cycle staleness writes do NOT affect the current draw distribution."""
    agent = _build_agent(sampler=True, draws=0, temperature=0.3)
    _install_anchors(agent, scale="fast", segment_ids=("a", "b"))
    acc = agent.hippocampal.staleness_accumulator
    acc._staleness[("fast", "a")] = 0.9
    acc._staleness[("fast", "b")] = 0.0

    sampler = agent.sleep_replay_sampler
    sampler.freeze_snapshot()
    snap_size_before = sampler.snapshot_size

    # Mutate accumulator AFTER freeze. Snapshot must be unaffected.
    acc._staleness[("fast", "b")] = 5.0
    acc._staleness[("fast", "new_region")] = 9.0

    assert sampler.snapshot_size == snap_size_before
    # Distribution should still favour 'a' (the snapshot value), not 'b'
    # (post-freeze mutation) or 'new_region' (post-freeze insertion;
    # additionally 'new_region' has no anchor so cannot be drawn).
    rng = np.random.default_rng(20260425)
    counts = Counter()
    for _ in range(500):
        a = sampler.draw(rng)
        counts[(a.key[0], a.key[1])] += 1
    assert counts[("fast", "a")] > counts[("fast", "b")] * 3


def test_c10_metrics_in_cycle_output():
    agent = _build_agent(sampler=True, draws=12)
    _install_anchors(agent, scale="fast", segment_ids=("0.1", "0.2"))
    metrics = agent.sleep_loop.force_cycle(agent)
    for key in (
        "mech285_n_draws",
        "mech285_n_distinct_regions_drawn",
        "mech285_snapshot_size",
        "mech285_snapshot_is_uniform",
        "mech285_temperature",
    ):
        assert key in metrics, f"Missing diagnostic key: {key}"
    assert metrics["mech285_n_draws"] == 12.0
