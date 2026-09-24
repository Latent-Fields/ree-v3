"""Contract tests for MECH-468 A/C/D/E per-anchor relational-edge recording.

Four independent read-side (A/C/D) plus in-loop (E) recording additions,
proposed by REE_assembly/evidence/planning/mech468_edge_type_inventory_spike.md
Section 4 and registered as substrate_queue.json entry
"mech468-anchor-relational-dump". All four are pure recording: no change
to any selection, ranking, or accumulation math. Relation type B (action
transition) is explicitly OUT OF SCOPE for this entry.

Guarantees enforced here:
  C1. Bit-identical OFF: with every new flag at its default (False), the
      new accessors return empty, and the underlying computation (rank()
      priorities/components/sums, StalenessAccumulator staleness values)
      is identical to a config with the flags never having existed.
  C2. Liveness (ON vs OFF differ): on a realistic, non-hand-picked pool,
      turning each flag on populates its accessor; turning it off does
      not, on the SAME inputs.
  C3. Type A non-degeneracy: a >=15-anchor pool with genuinely distinct
      z_world vectors produces a proximity-edge spread that is neither
      collapsed (~all 1.0) nor empty (~all 0.0) -- spike Section 3 row A.
  C4. Type C: events sharing the same tick `t` across >=2 distinct
      (scale, stream_mixture) families are recoverable as a shared-event
      group by joining the logged records on `t` -- spike Section 3 row C.
  C5. Type D non-degeneracy: per-anchor goal_match spread (IQR) clears a
      floor on a pool of genuinely distinct z_goal_snapshot vectors --
      spike Section 3 row D.
  C6. Type E: attribution_mode="equal" (the default) is fully-connected
      by construction (one edge per active anchor per broadcast) exactly
      as the spike documents as the degenerate case; "stream_overlap"
      is NOT fully-connected on a partially-overlapping pool -- spike
      Section 3 row E, and its explicit recommendation that any future
      E-type probe use stream_overlap, not equal.
  C7. Ring-buffer bound: the type-C and type-E logs are capped at their
      configured max length (oldest dropped first), so recording stays
      bounded across a long rollout.
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


def _make_broadcast(
    *,
    t: int = 0,
    strength: float = 1.0,
    posterior: float = 1.0,
    targets=None,
    source_scale: str = "fast",
    source_segment_id_old: str = "0.0",
    source_segment_id_new: str = "0.1",
    source_sources=None,
):
    from ree_core.regulators.invalidation_trigger import BroadcastEvent
    return BroadcastEvent(
        t=int(t),
        strength=float(strength),
        posterior=float(posterior),
        targets=list(targets or []),
        source_scale=source_scale,
        source_segment_id_old=source_segment_id_old,
        source_segment_id_new=source_segment_id_new,
        source_sources=list(source_sources or []),
    )


def _make_bank(anchors, cfg):
    from ree_core.hippocampal.anchor_set import AnchorSet
    from ree_core.hippocampal.ghost_goal_bank import GhostGoalBank
    from ree_core.utils.config import AnchorSetConfig
    s = AnchorSet(AnchorSetConfig())
    s._all = {a.key: a for a in anchors}
    return GhostGoalBank(cfg, s)


def _ghost_anchor(seg: str, zsnap, wanting: float = 0.3, arousal: float = 0.1):
    from ree_core.hippocampal.anchor_set import Anchor, AnchorGoalPayload
    a = Anchor(key=("fast", seg, ("s",)), z_world=torch.zeros(4), active=False)
    a.goal_payload = AnchorGoalPayload(
        z_goal_snapshot=torch.tensor(zsnap, dtype=torch.float32).unsqueeze(0),
        wanting_strength=wanting,
        arousal_tag=arousal,
    )
    return a


# ------------------------------------------------------------------ #
# C1: bit-identical OFF                                              #
# ------------------------------------------------------------------ #

def test_c1_defaults_are_off():
    from ree_core.utils.config import (
        AnchorSetConfig,
        GhostGoalBankConfig,
        StalenessAccumulatorConfig,
    )
    assert AnchorSetConfig().record_relational_snapshot is False
    assert GhostGoalBankConfig().record_relational_components is False
    assert StalenessAccumulatorConfig().record_edge_log is False


def test_c1_anchor_set_dump_is_noop_when_off():
    s = _make_anchor_set(subscribe_to_boundary_events=True)
    for i in range(20):
        s.consume_boundary_events(
            events=[_make_boundary_event(segment_id_new=f"0.{i}", t=i)],
            z_world=torch.randn(4),
            stream_mixture=("s",),
        )
    assert s.dump_relational_snapshot() == {}
    assert list(s._relational_event_log) == []


def test_c1_ghost_bank_rank_bit_identical_off_vs_on():
    from ree_core.utils.config import GhostGoalBankConfig
    anchors_off = [_ghost_anchor(f"S{i}", [float(i % 3), 1.0, 0.0, 0.0]) for i in range(6)]
    anchors_on = [_ghost_anchor(f"S{i}", [float(i % 3), 1.0, 0.0, 0.0]) for i in range(6)]
    z_goal = torch.tensor([1.0, 1.0, 0.0, 0.0])

    bank_off = _make_bank(anchors_off, GhostGoalBankConfig(goal_match_floor=0.0))
    bank_on = _make_bank(
        anchors_on,
        GhostGoalBankConfig(goal_match_floor=0.0, record_relational_components=True),
    )
    off = bank_off.rank(z_goal)
    on = bank_on.rank(z_goal)

    assert len(off) == len(on) == 6
    for e_off, e_on in zip(off, on):
        assert e_off.ghost_priority == e_on.ghost_priority
        assert e_off.components == e_on.components
    assert bank_off.get_diagnostics()["component_sums"] == bank_on.get_diagnostics()["component_sums"]
    # Only the new accessor differs.
    assert bank_off.get_relational_components() == []
    assert len(bank_on.get_relational_components()) == 6


def test_c1_staleness_accumulator_bit_identical_off_vs_on():
    from ree_core.hippocampal.anchor_set import Anchor
    from ree_core.hippocampal.staleness_accumulator import StalenessAccumulator
    from ree_core.utils.config import StalenessAccumulatorConfig

    anchors = [Anchor(key=("fast", "0.0", ("a", "b")), z_world=torch.zeros(2)),
               Anchor(key=("fast", "0.1", ("b", "c")), z_world=torch.zeros(2))]
    broadcasts = [_make_broadcast(strength=0.7, source_sources=["a", "b"])]

    acc_off = StalenessAccumulator(StalenessAccumulatorConfig())
    acc_on = StalenessAccumulator(StalenessAccumulatorConfig(record_edge_log=True))
    acc_off.integrate(broadcasts, anchors)
    acc_on.integrate(broadcasts, anchors)

    assert acc_off.snapshot() == acc_on.snapshot()
    assert acc_off.get_stats()["n_integrations"] == acc_on.get_stats()["n_integrations"]
    assert acc_off.edge_log() == []
    assert len(acc_on.edge_log()) == 2  # one per active anchor, "equal" mode


# ------------------------------------------------------------------ #
# C2/C3: type A -- latent proximity, liveness + non-degeneracy        #
# ------------------------------------------------------------------ #

def test_c2_c3_type_a_proximity_liveness_and_nondegeneracy():
    torch.manual_seed(101)
    s_off = _make_anchor_set(record_relational_snapshot=False)
    s_on = _make_anchor_set(record_relational_snapshot=True)

    # A realistic-sized pool (spike's measured range 6-24; use 18) with
    # genuinely distinct z_world vectors -- not hand-picked to collapse
    # or spread by construction, just independent random draws.
    n = 18
    for i in range(n):
        z = torch.randn(8)
        for s in (s_off, s_on):
            s.write_anchor("fast", f"0.{i}", ("stream",), z_world=z.clone())

    assert s_off.dump_relational_snapshot() == {}  # C2: off stays empty

    snap = s_on.dump_relational_snapshot()  # C2: on is populated
    assert len(snap["anchors"]) == n
    n_pairs = n * (n - 1) // 2
    assert len(snap["proximity_edges"]) == n_pairs

    # C3: non-degeneracy -- neither fully-collapsed (~all near 1.0) nor
    # fully-disjoint (~all near 0.0) on random independent draws.
    proximities = [e["proximity"] for e in snap["proximity_edges"]]
    assert all(0.0 <= p <= 1.0 for p in proximities)
    frac_high = sum(1 for p in proximities if p > 0.9) / len(proximities)
    frac_zero = sum(1 for p in proximities if p == 0.0) / len(proximities)
    assert frac_high < 0.7, "collapsed: proximity matrix looks fully connected"
    assert frac_zero < 0.7, "degenerate: proximity matrix looks fully disjoint"


# ------------------------------------------------------------------ #
# C2/C4: type C -- shared event                                       #
# ------------------------------------------------------------------ #

def test_c2_c4_type_c_shared_event_liveness_and_joinability():
    s_off = _make_anchor_set(record_relational_snapshot=False)
    s_on = _make_anchor_set(record_relational_snapshot=True)

    # Two concurrently-active families (distinct stream_mixture), each
    # gets a BoundaryEvent tagged with the SAME originating tick t=7 --
    # the shape a real shared-cause boundary would produce.
    for s in (s_off, s_on):
        s.consume_boundary_events(
            events=[_make_boundary_event(segment_id_new="0.1", t=7, sources=["x"])],
            z_world=torch.randn(4),
            stream_mixture=("fam_a",),
        )
        s.consume_boundary_events(
            events=[_make_boundary_event(segment_id_new="0.1", t=7, sources=["x"])],
            z_world=torch.randn(4),
            stream_mixture=("fam_b",),
        )
        # A third, unrelated tick with no sharing.
        s.consume_boundary_events(
            events=[_make_boundary_event(segment_id_new="0.2", t=9, sources=["y"])],
            z_world=torch.randn(4),
            stream_mixture=("fam_a",),
        )

    assert list(s_off._relational_event_log) == []  # C2: off stays empty

    log = s_on.dump_relational_snapshot()["shared_event_edges"]
    assert len(log) == 3  # C2: on is populated, one row per installed anchor

    # C4: joinable on t -- two distinct anchor_keys share t=7 (from
    # different families), demonstrating a genuine shared-event edge;
    # t=9 has exactly one (no sharing).
    by_t = {}
    for row in log:
        by_t.setdefault(row["t"], []).append(row["anchor_key"])
    assert len(by_t[7]) == 2
    assert by_t[7][0] != by_t[7][1]
    assert len(by_t[9]) == 1


def test_c7_type_c_ring_buffer_bounded():
    s = _make_anchor_set(record_relational_snapshot=True, relational_event_log_max_len=5)
    for i in range(20):
        s.consume_boundary_events(
            events=[_make_boundary_event(segment_id_new=f"0.{i}", t=i)],
            z_world=torch.randn(4),
            stream_mixture=("s",),
        )
    log = s.dump_relational_snapshot()["shared_event_edges"]
    assert len(log) == 5
    # Oldest dropped first -- the surviving records are the most recent.
    assert [row["t"] for row in log] == [15, 16, 17, 18, 19]


def test_type_a_c_reset_clears_relational_state():
    s = _make_anchor_set(record_relational_snapshot=True)
    s.write_anchor("fast", "0.0", ("s",), z_world=torch.randn(4))
    s.consume_boundary_events(
        events=[_make_boundary_event(segment_id_new="0.1", t=1)],
        z_world=torch.randn(4), stream_mixture=("s",),
    )
    assert len(s._relational_event_log) > 0
    s.reset()
    assert list(s._relational_event_log) == []
    assert s.dump_relational_snapshot()["anchors"] == []


# ------------------------------------------------------------------ #
# C2/C5: type D -- shared goal/valence                                #
# ------------------------------------------------------------------ #

def test_c2_c5_type_d_liveness_and_nondegeneracy():
    from ree_core.utils.config import GhostGoalBankConfig

    torch.manual_seed(202)
    # Genuinely distinct z_goal_snapshot directions across a realistic
    # pool (spike's measured range), each with distinct wanting/arousal.
    specs = [
        (f"S{i}", torch.randn(6).tolist(), 0.1 * (i % 5), 0.05 * (i % 4))
        for i in range(20)
    ]
    z_goal = torch.tensor([1.0, 0.3, -0.2, 0.5, 0.1, -0.4])

    anchors_off = [_ghost_anchor(seg, zs, w, a) for seg, zs, w, a in specs]
    anchors_on = [_ghost_anchor(seg, zs, w, a) for seg, zs, w, a in specs]
    bank_off = _make_bank(anchors_off, GhostGoalBankConfig(goal_match_floor=0.0))
    bank_on = _make_bank(
        anchors_on,
        GhostGoalBankConfig(goal_match_floor=0.0, record_relational_components=True),
    )
    bank_off.rank(z_goal)
    bank_on.rank(z_goal)

    assert bank_off.get_relational_components() == []  # C2: off stays empty
    comps = bank_on.get_relational_components()         # C2: on is populated
    assert len(comps) == 20
    for row in comps:
        assert set(row.keys()) == {"anchor_key", "goal_match", "wanting_strength", "arousal_tag"}

    # C5: non-degeneracy -- goal_match spread (IQR) clears a floor on a
    # genuinely distinct pool (not the SD-079 pre-centering collapse case,
    # which test_sd_079_centered_goal_anchor_match.py covers separately).
    matches = sorted(row["goal_match"] for row in comps)
    q1 = matches[len(matches) // 4]
    q3 = matches[(3 * len(matches)) // 4]
    iqr = q3 - q1
    assert iqr > 0.05, f"collapsed: goal_match IQR={iqr} too narrow to be a non-degenerate edge type"

    # The raw per-anchor value is consistent with, but a strictly richer
    # record than, the WEIGHTED composite term already retained in
    # components: at a non-1.0 weight the two diverge, confirming
    # get_relational_components() carries the pre-weighting raw value
    # rather than merely re-deriving what components already exposed.
    bank_weighted = _make_bank(
        [_ghost_anchor(seg, zs, w, a) for seg, zs, w, a in specs],
        GhostGoalBankConfig(
            goal_match_floor=0.0,
            goal_match_weight=0.4,
            record_relational_components=True,
        ),
    )
    ranked = bank_weighted.rank(z_goal)
    by_key = {row["anchor_key"]: row for row in bank_weighted.get_relational_components()}
    saw_divergence = False
    for entry in ranked:
        raw = by_key[entry.anchor.key]["goal_match"]
        weighted = entry.components["goal_match"]
        assert abs(weighted - 0.4 * raw) < 1e-6
        if raw > 1e-6:
            saw_divergence = saw_divergence or (weighted != raw)
    assert saw_divergence, "raw and weighted goal_match never diverged -- fixture too degenerate"


# ------------------------------------------------------------------ #
# C2/C6: type E -- causal/outcome (staleness attribution)             #
# ------------------------------------------------------------------ #

def test_c2_c6_type_e_liveness_and_degenerate_equal_mode():
    from ree_core.hippocampal.anchor_set import Anchor
    from ree_core.hippocampal.staleness_accumulator import StalenessAccumulator
    from ree_core.utils.config import StalenessAccumulatorConfig

    anchors = [
        Anchor(key=("fast", f"0.{i}", (f"stream_{i}",)), z_world=torch.zeros(2))
        for i in range(6)
    ]
    broadcasts = [_make_broadcast(strength=0.9, source_sources=["stream_0"])]

    acc_off = StalenessAccumulator(StalenessAccumulatorConfig(attribution_mode="equal"))
    acc_on = StalenessAccumulator(
        StalenessAccumulatorConfig(attribution_mode="equal", record_edge_log=True)
    )
    acc_off.integrate(broadcasts, anchors)
    acc_on.integrate(broadcasts, anchors)

    assert acc_off.edge_log() == []  # C2: off stays empty
    log_equal = acc_on.edge_log()     # C2: on is populated
    # C6: "equal" mode is fully-connected by construction -- one edge per
    # active anchor, regardless of stream overlap (spike's documented
    # degenerate case).
    assert len(log_equal) == len(anchors) == 6
    assert all(row["attribution_weight"] == pytest.approx(1.0 / 6) for row in log_equal)


def test_c6_type_e_stream_overlap_is_not_fully_connected():
    from ree_core.hippocampal.anchor_set import Anchor
    from ree_core.hippocampal.staleness_accumulator import StalenessAccumulator
    from ree_core.utils.config import StalenessAccumulatorConfig

    # Only 2 of 6 anchors' stream_mixture overlaps the broadcast's source.
    anchors = [Anchor(key=("fast", "0.0", ("stream_0",)), z_world=torch.zeros(2)),
               Anchor(key=("fast", "0.1", ("stream_0", "other")), z_world=torch.zeros(2))]
    anchors += [
        Anchor(key=("fast", f"0.{i}", (f"unrelated_{i}",)), z_world=torch.zeros(2))
        for i in range(2, 6)
    ]
    broadcasts = [_make_broadcast(strength=0.9, source_sources=["stream_0"])]

    acc = StalenessAccumulator(
        StalenessAccumulatorConfig(attribution_mode="stream_overlap", record_edge_log=True)
    )
    acc.integrate(broadcasts, anchors)
    log = acc.edge_log()

    # C6: NOT fully connected -- only the 2 overlapping anchors are logged
    # (zero-overlap anchors get incr<=0 and are skipped by construction),
    # in contrast to equal mode's guaranteed len(anchors) above.
    assert 0 < len(log) < len(anchors)
    logged_keys = {row["target_anchor_key"] for row in log}
    assert ("fast", "0.0", ("stream_0",)) in logged_keys
    assert ("fast", "0.1", ("stream_0", "other")) in logged_keys
    assert ("fast", "0.2", ("unrelated_2",)) not in logged_keys


def test_c7_type_e_ring_buffer_bounded():
    from ree_core.hippocampal.anchor_set import Anchor
    from ree_core.hippocampal.staleness_accumulator import StalenessAccumulator
    from ree_core.utils.config import StalenessAccumulatorConfig

    anchors = [Anchor(key=("fast", "0.0", ("s",)), z_world=torch.zeros(2))]
    acc = StalenessAccumulator(
        StalenessAccumulatorConfig(record_edge_log=True, edge_log_max_len=3)
    )
    for i in range(10):
        acc.integrate([_make_broadcast(t=i, strength=0.5, source_sources=[])], anchors)
    log = acc.edge_log()
    assert len(log) == 3
    assert [row["t"] for row in log] == [7, 8, 9]


def test_type_e_reset_clears_edge_log():
    from ree_core.hippocampal.anchor_set import Anchor
    from ree_core.hippocampal.staleness_accumulator import StalenessAccumulator
    from ree_core.utils.config import StalenessAccumulatorConfig

    anchors = [Anchor(key=("fast", "0.0", ("s",)), z_world=torch.zeros(2))]
    acc = StalenessAccumulator(StalenessAccumulatorConfig(record_edge_log=True))
    acc.integrate([_make_broadcast(strength=0.5, source_sources=[])], anchors)
    assert acc.edge_log() != []
    acc.reset()
    assert acc.edge_log() == []
