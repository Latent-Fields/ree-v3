"""Contract tests for MECH-269 Phase 2 (iii, T4): per-region per-stream V_s.

Promotes flat per_stream_vs[stream] -> float to per_region_vs[(scale,
segment_id)][stream] -> float keyed on AnchorSet active-anchor keys.

Guarantees enforced here:
  C1. Backward compat: default HippocampalConfig has
      use_per_region_vs=False; with the flag off the flat per-stream path
      continues to work and per_region_vs stays empty.
  C2. Per-region population: with use_per_region_vs=True and an active
      anchor installed via a BoundaryEvent, update_per_region_vs populates
      per_region_vs[(scale, segment_id_new)][stream] for every registered
      stream present on the LatentState (seeded at 1.0 on first tick).
  C3. Cross-region isolation: two regions on distinct (scale, segment_id)
      keys maintain independent prev-value caches. Marking one region's
      anchor inactive (e.g. via mark_inactive) drops only that region's
      entry; the other region's per_region_vs entry is untouched.
  C4. MECH-287 invalidation-broadcast reset: a BroadcastEvent on
      (source_scale, source_segment_id_old) drops only that region's
      per_region_vs entry AND mark_inactives the matching active anchor.
      Other regions remain active and present.
  C5. k=5 hysteresis honoured: tick_anchor_set advances hysteresis; when
      an anchor is marked inactive after hysteresis_k consecutive below-
      threshold ticks, update_per_region_vs prunes its per_region_vs
      entry on the next tick.
"""

from __future__ import annotations

import pytest
import torch


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #

def _make_module(
    *,
    use_per_region_vs: bool = False,
    use_per_stream_vs: bool = False,
    use_anchor_sets: bool = True,
    tau: float = 1.0,
    hysteresis_k: int = 5,
    reset_threshold: float = 0.3,
):
    from ree_core.utils.config import (
        HippocampalConfig,
        E2Config,
        ResidueConfig,
        AnchorSetConfig,
    )
    from ree_core.predictors.e2_fast import E2FastPredictor
    from ree_core.residue.field import ResidueField
    from ree_core.hippocampal.module import HippocampalModule

    anchor_cfg = AnchorSetConfig(
        hysteresis_k=hysteresis_k,
        reset_threshold=reset_threshold,
        staleness_rate=0.0,  # disable staleness for deterministic tests
    )
    hcfg = HippocampalConfig(
        world_dim=8, action_dim=4, action_object_dim=4,
        hidden_dim=16, horizon=3, num_candidates=4,
        num_cem_iterations=1, elite_fraction=0.5,
        use_per_stream_vs=use_per_stream_vs,
        use_per_region_vs=use_per_region_vs,
        use_anchor_sets=use_anchor_sets,
        anchor_set=anchor_cfg,
        per_stream_vs_tau=tau,
    )
    e2cfg = E2Config(self_dim=8, world_dim=8, action_dim=4, action_object_dim=4)
    rcfg = ResidueConfig(world_dim=8, num_basis_functions=4)
    e2 = E2FastPredictor(e2cfg)
    residue = ResidueField(rcfg)
    return HippocampalModule(hcfg, e2, residue)


def _make_latent(z_world=None, z_self=None, z_beta=None):
    from ree_core.latent.stack import LatentState
    return LatentState(
        z_self=z_self if z_self is not None else torch.zeros(1, 8),
        z_world=z_world if z_world is not None else torch.zeros(1, 8),
        z_beta=z_beta if z_beta is not None else torch.zeros(1, 8),
        z_theta=torch.zeros(1, 8),
        z_delta=torch.zeros(1, 8),
        precision={},
        z_harm=None,
        z_harm_a=None,
    )


def _install_anchor(mod, scale, segment_id, z_world, mixture=("z_world", "z_self", "z_beta")):
    """Install an active anchor for a given (scale, segment_id) via consume_boundary_events."""
    from ree_core.hippocampal.event_segmenter import BoundaryEvent
    ev = BoundaryEvent(
        segment_id_old="__seed__",
        segment_id_new=segment_id,
        scale=scale,
        posterior=1.0,
        sources=["z_world"],
        t=0,
    )
    mod.anchor_set.consume_boundary_events(
        events=[ev], z_world=z_world, stream_mixture=mixture,
    )


def _make_broadcast(scale, segment_id_old, segment_id_new="new"):
    from ree_core.regulators.invalidation_trigger import BroadcastEvent
    return BroadcastEvent(
        t=0,
        strength=1.0,
        posterior=1.0,
        targets=["mech_269_anchor_set"],
        source_scale=scale,
        source_segment_id_old=segment_id_old,
        source_segment_id_new=segment_id_new,
        source_sources=["z_world"],
    )


# ------------------------------------------------------------------ #
# C1                                                                 #
# ------------------------------------------------------------------ #

def test_c1_default_config_and_flag_off_backward_compat():
    """C1: default flag False; flat per_stream_vs path continues unchanged."""
    from ree_core.utils.config import HippocampalConfig
    cfg = HippocampalConfig()
    assert getattr(cfg, "use_per_region_vs", False) is False

    mod = _make_module(use_per_region_vs=False, use_per_stream_vs=True)
    assert mod.per_region_vs == {}
    # Flat per-stream path still works when per_region is off.
    z0 = torch.zeros(1, 8); z0[0, 0] = 1.0
    mod.update_per_stream_vs(_make_latent(
        z_world=z0.clone(), z_self=z0.clone(), z_beta=z0.clone(),
    ))
    assert "z_world" in mod.per_stream_vs
    # update_per_region_vs is a no-op when flag is off (even with anchors).
    _install_anchor(mod, "fast", "1.0", z_world=z0)
    mod.update_per_region_vs(_make_latent(
        z_world=z0.clone(), z_self=z0.clone(), z_beta=z0.clone(),
    ))
    assert mod.per_region_vs == {}, (
        "update_per_region_vs must be a no-op when use_per_region_vs=False"
    )


# ------------------------------------------------------------------ #
# C2                                                                 #
# ------------------------------------------------------------------ #

def test_c2_per_region_populates_on_boundary_event():
    """C2: active anchor -> update_per_region_vs seeds V_s=1.0 for each stream."""
    mod = _make_module(
        use_per_region_vs=True, use_per_stream_vs=True,
        use_anchor_sets=True, tau=1.0,
    )
    z0 = torch.zeros(1, 8); z0[0, 0] = 1.0
    latent = _make_latent(
        z_world=z0.clone(), z_self=z0.clone(), z_beta=z0.clone(),
    )
    # per_stream_vs must be populated first so stream_mixture for tick_anchor_set
    # matches the readout streams; Phase 1 update does that.
    mod.update_per_stream_vs(latent)
    _install_anchor(mod, "fast", "1.0", z_world=z0)

    mod.update_per_region_vs(latent)

    assert ("fast", "1.0") in mod.per_region_vs, (
        f"expected region key present, got {list(mod.per_region_vs.keys())}"
    )
    region_vs = mod.per_region_vs[("fast", "1.0")]
    for s in ("z_world", "z_self", "z_beta"):
        assert region_vs[s] == 1.0, f"first tick should seed {s}=1.0"


# ------------------------------------------------------------------ #
# C3                                                                 #
# ------------------------------------------------------------------ #

def test_c3_cross_region_isolation():
    """C3: invalidation of one region does not affect a different region's V_s."""
    mod = _make_module(
        use_per_region_vs=True, use_per_stream_vs=True,
        use_anchor_sets=True, tau=1.0,
    )
    z0 = torch.zeros(1, 8); z0[0, 0] = 1.0
    latent = _make_latent(
        z_world=z0.clone(), z_self=z0.clone(), z_beta=z0.clone(),
    )
    mod.update_per_stream_vs(latent)
    _install_anchor(mod, "fast", "3.7", z_world=z0)
    _install_anchor(mod, "slow", "3", z_world=z0)

    # Tick update to seed both regions.
    mod.update_per_region_vs(latent)
    assert ("fast", "3.7") in mod.per_region_vs
    assert ("slow", "3") in mod.per_region_vs
    slow_before = dict(mod.per_region_vs[("slow", "3")])

    # Perturb latent; both regions see the same delta, but their cached
    # prev values are independent -- that independence is the structural
    # property the test guards.
    z1 = z0.clone(); z1[0, 1] = 0.5
    latent_b = _make_latent(
        z_world=z1, z_self=z0.clone(), z_beta=z0.clone(),
    )
    mod.update_per_stream_vs(latent_b)
    mod.update_per_region_vs(latent_b)

    # Mark the fast region's anchor inactive (simulate targeted removal
    # that does not touch the slow region's anchor/mixture family).
    # Find the active fast-region stream_mixture.
    fast_anchors = [a for a in mod.anchor_set.active_anchors(scale="fast")
                    if a.key[1] == "3.7"]
    assert len(fast_anchors) == 1
    mod.anchor_set.mark_inactive(
        scale="fast", stream_mixture=fast_anchors[0].key[2],
    )

    # Next update prunes the fast region but leaves the slow region intact.
    mod.update_per_region_vs(latent_b)
    assert ("fast", "3.7") not in mod.per_region_vs, (
        "fast region must be pruned after its anchor is marked inactive"
    )
    assert ("slow", "3") in mod.per_region_vs, (
        "slow region must remain after fast region invalidation"
    )
    # Slow region's prev cache was unaffected by the fast invalidation.
    slow_after = mod.per_region_vs[("slow", "3")]
    for s in ("z_world", "z_self", "z_beta"):
        assert s in slow_after
    assert slow_after["z_self"] == pytest.approx(slow_before["z_self"])
    assert slow_after["z_beta"] == pytest.approx(slow_before["z_beta"])


# ------------------------------------------------------------------ #
# C4                                                                 #
# ------------------------------------------------------------------ #

def test_c4_invalidation_broadcast_resets_target_region_only():
    """C4: broadcast on (scale, segment_id_old) drops only that region's entry."""
    mod = _make_module(
        use_per_region_vs=True, use_per_stream_vs=True,
        use_anchor_sets=True, tau=1.0,
    )
    z0 = torch.zeros(1, 8); z0[0, 0] = 1.0
    latent = _make_latent(
        z_world=z0.clone(), z_self=z0.clone(), z_beta=z0.clone(),
    )
    mod.update_per_stream_vs(latent)
    _install_anchor(mod, "fast", "1.2", z_world=z0)
    _install_anchor(mod, "slow", "0", z_world=z0)
    mod.update_per_region_vs(latent)

    assert ("fast", "1.2") in mod.per_region_vs
    assert ("slow", "0") in mod.per_region_vs

    # Broadcast a reset on the fast region.
    bcast = _make_broadcast(scale="fast", segment_id_old="1.2")
    reset_keys = mod.apply_invalidation_broadcasts_to_regions([bcast])

    assert reset_keys == [("fast", "1.2")]
    assert ("fast", "1.2") not in mod.per_region_vs
    assert ("slow", "0") in mod.per_region_vs, (
        "unrelated region must not be reset by a broadcast on a different scale"
    )
    # Anchor side: fast (1.2) anchor should now be inactive; slow still active.
    active_fast = [a for a in mod.anchor_set.active_anchors(scale="fast")
                   if a.key[1] == "1.2"]
    assert active_fast == [], (
        "broadcast reset must mark the matching fast anchor inactive"
    )
    active_slow = [a for a in mod.anchor_set.active_anchors(scale="slow")
                   if a.key[1] == "0"]
    assert len(active_slow) == 1

    # Idempotency: apply the same broadcast again -- must be no-op on
    # already-reset region and return no new reset keys.
    reset_keys2 = mod.apply_invalidation_broadcasts_to_regions([bcast])
    assert reset_keys2 == []


# ------------------------------------------------------------------ #
# C5                                                                 #
# ------------------------------------------------------------------ #

def test_c5_k5_hysteresis_honoured_by_reset_path():
    """C5: hysteresis_k=5 ticks below threshold -> anchor inactive -> per_region entry pruned on next update."""
    mod = _make_module(
        use_per_region_vs=True, use_per_stream_vs=True,
        use_anchor_sets=True, tau=1.0,
        hysteresis_k=5, reset_threshold=0.3,
    )
    z0 = torch.zeros(1, 8); z0[0, 0] = 1.0
    latent = _make_latent(
        z_world=z0.clone(), z_self=z0.clone(), z_beta=z0.clone(),
    )
    mod.update_per_stream_vs(latent)
    _install_anchor(mod, "fast", "9.9", z_world=z0)
    mod.update_per_region_vs(latent)
    assert ("fast", "9.9") in mod.per_region_vs

    # Feed per_stream_vs scores BELOW the reset_threshold to tick_hysteresis.
    # below_threshold_streak increments on each call; after hysteresis_k=5
    # consecutive below-threshold ticks the anchor fires mark_inactive.
    low_vs = {s: 0.1 for s in ("z_world", "z_self", "z_beta")}
    fired = []
    for _ in range(5):
        fired.extend(mod.anchor_set.tick_hysteresis(low_vs))
    assert len(fired) >= 1, "hysteresis must fire after k consecutive below-threshold ticks"
    # Anchor now inactive.
    active_fast = [a for a in mod.anchor_set.active_anchors(scale="fast")
                   if a.key[1] == "9.9"]
    assert active_fast == []

    # Next update_per_region_vs sees no active anchor for (fast, 9.9) and
    # prunes the per_region_vs entry.
    mod.update_per_region_vs(latent)
    assert ("fast", "9.9") not in mod.per_region_vs, (
        "per_region_vs entry must be pruned after k-hysteresis mark_inactive"
    )


# ------------------------------------------------------------------ #
# Reset integration smoke                                            #
# ------------------------------------------------------------------ #

def test_reset_per_stream_vs_clears_per_region_state():
    """Per-episode reset clears both per_stream_vs and per_region_vs state."""
    mod = _make_module(
        use_per_region_vs=True, use_per_stream_vs=True,
        use_anchor_sets=True, tau=1.0,
    )
    z0 = torch.zeros(1, 8); z0[0, 0] = 1.0
    latent = _make_latent(
        z_world=z0.clone(), z_self=z0.clone(), z_beta=z0.clone(),
    )
    mod.update_per_stream_vs(latent)
    _install_anchor(mod, "fast", "1.0", z_world=z0)
    mod.update_per_region_vs(latent)
    assert mod.per_region_vs != {}

    mod.reset_per_stream_vs()
    assert mod.per_stream_vs == {}
    assert mod.per_region_vs == {}
    assert mod._prev_region_stream_values == {}
