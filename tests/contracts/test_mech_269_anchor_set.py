"""Contract tests for MECH-269 Phase 2 (ii) anchor sets.

Scale-tagged hippocampal anchor store with dual-trace preservation
(Bouton 2004) and k-consecutive hysteresis on V_s_anchor crossings.
Consumes MECH-288 BoundaryEvents to install / remap anchors.

Guarantees enforced here:
  C1. Backward compat: default HippocampalConfig.use_anchor_sets is
      False; HippocampalModule.anchor_set is None; all hooks are no-op.
  C2. BoundaryEvent -> new active anchor install: a BoundaryEvent on
      a registered scale triggers write_anchor; the anchor carries the
      scale, new segment_id, and the stream_mixture tuple supplied by
      the caller.
  C3. Dual-trace on remap: a second BoundaryEvent on the same
      (scale, stream_mixture) family marks the prior anchor inactive
      (retained in all_anchors) and installs a new active anchor.
      Erase is never the resolution path.
  C4. Hysteresis (k=5 default): an active anchor whose V_s_anchor
      stays below reset_threshold for EXACTLY hysteresis_k consecutive
      tick_hysteresis calls is marked inactive. Fewer consecutive
      ticks below threshold do NOT fire the reset (streak resets on
      any tick at-or-above threshold).
  C5. reset_region: dual-trace remap API preserves the outgoing
      anchor inactive and installs a new active one.
  C6. Per-episode reset: AnchorSet.reset() clears both active and
      inactive anchor stores and resets the internal tick counter.
"""

from __future__ import annotations

import pytest
import torch


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #

def _make_boundary_event(
    *,
    posterior: float = 1.0,
    scale: str = "fast",
    segment_id_old: str = "0.0",
    segment_id_new: str = "0.1",
    sources=None,
    t: int = 0,
):
    from ree_core.hippocampal.event_segmenter import BoundaryEvent
    return BoundaryEvent(
        segment_id_old=segment_id_old,
        segment_id_new=segment_id_new,
        scale=scale,
        posterior=float(posterior),
        sources=list(sources or ["z_world"]),
        t=int(t),
    )


def _make_anchor_set(**overrides):
    from ree_core.hippocampal.anchor_set import AnchorSet
    from ree_core.utils.config import AnchorSetConfig
    cfg = AnchorSetConfig(**overrides) if overrides else AnchorSetConfig()
    return AnchorSet(cfg)


def _make_module(use_anchor_sets: bool = False, **anchor_overrides):
    from ree_core.utils.config import (
        HippocampalConfig,
        E2Config,
        ResidueConfig,
        AnchorSetConfig,
    )
    from ree_core.predictors.e2_fast import E2FastPredictor
    from ree_core.residue.field import ResidueField
    from ree_core.hippocampal.module import HippocampalModule

    anchor_cfg = (
        AnchorSetConfig(**anchor_overrides)
        if anchor_overrides
        else AnchorSetConfig()
    )
    hcfg = HippocampalConfig(
        world_dim=8, action_dim=4, action_object_dim=4,
        hidden_dim=16, horizon=3, num_candidates=4,
        num_cem_iterations=1, elite_fraction=0.5,
        use_anchor_sets=use_anchor_sets,
        anchor_set=anchor_cfg,
    )
    e2cfg = E2Config(self_dim=8, world_dim=8, action_dim=4, action_object_dim=4)
    rcfg = ResidueConfig(world_dim=8, num_basis_functions=4)
    e2 = E2FastPredictor(e2cfg)
    residue = ResidueField(rcfg)
    return HippocampalModule(hcfg, e2, residue)


# ------------------------------------------------------------------ #
# C1                                                                 #
# ------------------------------------------------------------------ #

def test_c1_default_config_backward_compatible():
    """C1: flag off by default; module surface is inert."""
    from ree_core.utils.config import HippocampalConfig
    cfg = HippocampalConfig()
    assert getattr(cfg, "use_anchor_sets", False) is False

    mod = _make_module(use_anchor_sets=False)
    assert mod.anchor_set is None
    # tick_anchor_set and reset_anchor_set are safe no-ops when off.
    class _L:
        z_world = None
    mod.tick_anchor_set(_L(), [])
    mod.reset_anchor_set()


# ------------------------------------------------------------------ #
# C2                                                                 #
# ------------------------------------------------------------------ #

def test_c2_boundary_event_installs_active_anchor():
    """C2: a BoundaryEvent on a registered scale installs an active anchor."""
    aset = _make_anchor_set()
    z = torch.randn(1, 8)
    mixture = ("z_world", "z_self")
    ev = _make_boundary_event(scale="fast", segment_id_new="0.3")

    installed = aset.consume_boundary_events([ev], z_world=z, stream_mixture=mixture)

    assert len(installed) == 1
    a = installed[0]
    assert a.active is True
    assert a.key == ("fast", "0.3", mixture)
    assert a.z_world.shape == z.shape
    # Present in active + full indexes.
    active = aset.active_anchors()
    assert len(active) == 1 and active[0] is a
    allset = aset.all_anchors()
    assert len(allset) == 1 and allset[0] is a

    # Unregistered scale is ignored.
    ev_unknown = _make_boundary_event(scale="ultra", segment_id_new="9.9")
    out = aset.consume_boundary_events([ev_unknown], z_world=z, stream_mixture=mixture)
    assert out == []
    assert len(aset.active_anchors()) == 1


# ------------------------------------------------------------------ #
# C3                                                                 #
# ------------------------------------------------------------------ #

def test_c3_dual_trace_on_remap_preserves_inactive():
    """C3: second event on same family marks prior anchor inactive, not erased."""
    aset = _make_anchor_set()
    mixture = ("z_world", "z_self")
    z1 = torch.randn(1, 8)
    z2 = torch.randn(1, 8)

    ev1 = _make_boundary_event(scale="fast", segment_id_new="0.1")
    ev2 = _make_boundary_event(scale="fast", segment_id_new="0.2")
    aset.consume_boundary_events([ev1], z_world=z1, stream_mixture=mixture)
    aset.consume_boundary_events([ev2], z_world=z2, stream_mixture=mixture)

    # Exactly one active anchor on that family; it is the second one.
    actives = aset.active_anchors(scale="fast")
    assert len(actives) == 1
    assert actives[0].key[1] == "0.2"

    # Previous anchor is preserved (inactive) in all_anchors().
    allset = aset.all_anchors(scale="fast")
    keys = {a.key[1]: a for a in allset}
    assert "0.1" in keys and "0.2" in keys
    assert keys["0.1"].active is False
    assert keys["0.2"].active is True


# ------------------------------------------------------------------ #
# C4                                                                 #
# ------------------------------------------------------------------ #

def test_c4_hysteresis_k_consecutive_below_threshold():
    """C4: reset fires only after hysteresis_k consecutive below-threshold ticks.

    Sub-test 4a: k-1 below then one at-threshold resets streak. No reset.
    Sub-test 4b: k below-threshold ticks in a row fire reset_region marker
    (mark_inactive on the currently-active anchor).
    """
    aset = _make_anchor_set(reset_threshold=0.3, hysteresis_k=5,
                            staleness_rate=0.0, staleness_clip=0.0)
    mixture = ("z_world",)
    z = torch.randn(1, 8)
    aset.consume_boundary_events(
        [_make_boundary_event(scale="fast", segment_id_new="0.1")],
        z_world=z, stream_mixture=mixture,
    )
    anchor = aset.active_anchors()[0]

    # 4a: 4 low ticks then one high. Streak must reset; anchor stays active.
    for _ in range(4):
        fired = aset.tick_hysteresis({"z_world": 0.1})
        assert fired == []
    assert anchor.below_threshold_streak == 4
    fired = aset.tick_hysteresis({"z_world": 0.9})
    assert fired == []
    assert anchor.below_threshold_streak == 0
    assert anchor.active is True

    # 4b: 5 consecutive low ticks -> reset fires on the 5th.
    for i in range(4):
        fired = aset.tick_hysteresis({"z_world": 0.1})
        assert fired == []
        assert anchor.active is True
    fired = aset.tick_hysteresis({"z_world": 0.1})
    assert len(fired) == 1 and fired[0] is anchor
    assert anchor.active is False
    # Dual-trace: inactive anchor is retained.
    assert any(not a.active for a in aset.all_anchors())


# ------------------------------------------------------------------ #
# C5                                                                 #
# ------------------------------------------------------------------ #

def test_c5_reset_region_dual_trace():
    """C5: reset_region marks current active inactive and installs new active."""
    aset = _make_anchor_set()
    mixture = ("z_world",)
    z_old = torch.randn(1, 8)
    z_new = torch.randn(1, 8)

    aset.consume_boundary_events(
        [_make_boundary_event(scale="fast", segment_id_new="1.0")],
        z_world=z_old, stream_mixture=mixture,
    )
    prior = aset.active_anchors()[0]

    new_anchor = aset.reset_region(
        scale="fast", stream_mixture=mixture,
        new_segment_id="2.0", z_world=z_new,
    )

    assert prior.active is False
    assert new_anchor.active is True
    assert new_anchor.key == ("fast", "2.0", mixture)
    # Both retained in all_anchors.
    ids = {a.key[1] for a in aset.all_anchors()}
    assert ids == {"1.0", "2.0"}
    # Exactly one active anchor on this family.
    assert len(aset.active_anchors()) == 1


# ------------------------------------------------------------------ #
# C6                                                                 #
# ------------------------------------------------------------------ #

def test_c6_reset_clears_state():
    """C6: AnchorSet.reset() clears active, inactive, and tick counter."""
    aset = _make_anchor_set()
    mixture = ("z_world",)
    z = torch.randn(1, 8)
    aset.consume_boundary_events(
        [_make_boundary_event(scale="fast", segment_id_new="0.1")],
        z_world=z, stream_mixture=mixture,
    )
    # Advance a tick of hysteresis.
    aset.tick_hysteresis({"z_world": 0.9})
    assert aset._tick == 1
    assert len(aset.all_anchors()) == 1

    aset.reset()
    assert aset.all_anchors() == []
    assert aset.active_anchors() == []
    assert aset._tick == 0


# ------------------------------------------------------------------ #
# Integration smoke: agent-level wiring                              #
# ------------------------------------------------------------------ #

def test_agent_level_flag_off_is_noop():
    """Flag OFF at the HippocampalModule level: no tick/reset side effects."""
    mod = _make_module(use_anchor_sets=False)
    assert mod.anchor_set is None
    # Reset + tick with None latent: pure no-ops.
    class _L:
        z_world = None
    mod.tick_anchor_set(_L(), [])
    mod.reset_anchor_set()


def test_agent_level_flag_on_ticks_on_boundary_event():
    """Flag ON: tick_anchor_set installs an active anchor for a queued event."""
    mod = _make_module(use_anchor_sets=True)
    assert mod.anchor_set is not None
    # Simulate the per_stream_vs having been populated by Phase 1 this tick.
    mod.per_stream_vs = {"z_world": 0.9, "z_self": 0.9}
    ev = _make_boundary_event(scale="fast", segment_id_new="0.1")

    class _L:
        z_world = torch.randn(1, 8)
    mod.tick_anchor_set(_L(), [ev])

    actives = mod.anchor_set.active_anchors()
    assert len(actives) == 1
    assert actives[0].key[0] == "fast"
    assert actives[0].key[1] == "0.1"
    # Stream mixture drawn from sorted per_stream_vs keys.
    assert actives[0].key[2] == ("z_self", "z_world")
