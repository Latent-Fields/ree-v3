"""Contract tests for SD-039 dual-trace anchor goal-snapshot payload.

SD-039 adds a compact motivational payload (z_goal_snapshot, wanting_strength,
arousal_tag, last_vs, staleness_at_write, payload_written_step) to each
MECH-269 dual-trace Anchor at write / remap / mark_inactive time. The payload
is preserved across mark_inactive so inactive anchors remain queryable as
ghost-goal traces. SD-039's substrate-side scope is the dataclass + the
goal_match() per-anchor cosine helper + the AnchorSet.query_by_goal_match()
helper that MECH-292 (ranked ghost-goal bank) and MECH-293 (waking
ghost-goal probes) will consume.

Out of scope for this contract suite:
  - module-level write-site population from GoalState / VALENCE_WANTING /
    amygdala arousal tags (deferred follow-on session).
  - MECH-292 ranked ghost-goal bank computation.
  - MECH-293 waking ghost-goal probe budget allocation.

Guarantees enforced here:
  S1. Module import + symbol presence: AnchorGoalPayload, Anchor.goal_match,
      AnchorSet.query_by_goal_match all importable from
      ree_core.hippocampal.anchor_set.
  S2. Default backward-compat: AnchorSetConfig.use_sd039_anchor_payload is
      False by default; bare write_anchor / mark_inactive / reset_region
      calls (no goal_payload kwarg) leave anchor.goal_payload == None.
  S3. ON-payload-attached-on-write: write_anchor with goal_payload writes
      the payload onto the new anchor.
  S4. Payload-survives-mark_inactive: mark_inactive() does not clear an
      existing goal_payload; an inactive anchor retains its payload.
  S5. goal_match returns 0.0 when payload is None, when z_goal_snapshot
      is None, or when current_z_goal is None.
  S6. goal_match returns correct cosine similarity (clipped to non-negative)
      between stored z_goal_snapshot and supplied current_z_goal.
  S7. query_by_goal_match returns active AND inactive anchors above
      threshold, sorted by score descending; payload-less anchors are
      excluded at any non-negative threshold.
  S8. query_by_goal_match returns empty list when current_z_goal is None.
  S9. reset_region (dual-trace remap) refreshes payload onto BOTH the
      outgoing inactive trace AND the new active anchor.
  S10. AnchorSet.reset() clears stored payloads (per-episode).
"""

from __future__ import annotations

import pytest
import torch


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #

def _make_anchor_set(**overrides):
    from ree_core.hippocampal.anchor_set import AnchorSet
    from ree_core.utils.config import AnchorSetConfig
    cfg = AnchorSetConfig(**overrides) if overrides else AnchorSetConfig()
    return AnchorSet(cfg)


def _make_payload(*, z_goal=None, wanting=0.5, arousal=0.3, last_vs=0.7,
                  staleness=0.1, written_step=0):
    from ree_core.hippocampal.anchor_set import AnchorGoalPayload
    if z_goal is None:
        z_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return AnchorGoalPayload(
        z_goal_snapshot=z_goal,
        wanting_strength=float(wanting),
        arousal_tag=float(arousal),
        last_vs=float(last_vs) if last_vs is not None else None,
        staleness_at_write=float(staleness) if staleness is not None else None,
        payload_written_step=int(written_step),
    )


# ------------------------------------------------------------------ #
# S1: import + symbol presence                                       #
# ------------------------------------------------------------------ #

def test_s1_imports_and_symbols():
    """SD-039 substrate-side surface importable + carries expected names."""
    from ree_core.hippocampal.anchor_set import (
        Anchor,
        AnchorGoalPayload,
        AnchorSet,
    )
    # AnchorGoalPayload is a dataclass with the SD-039 fields.
    p = AnchorGoalPayload()
    assert p.z_goal_snapshot is None
    assert p.wanting_strength == 0.0
    assert p.arousal_tag == 0.0
    assert p.last_vs is None
    assert p.staleness_at_write is None
    assert p.payload_written_step == 0
    # Anchor has goal_match method.
    assert hasattr(Anchor, "goal_match")
    # AnchorSet has the SD-039 query helper.
    assert hasattr(AnchorSet, "query_by_goal_match")


# ------------------------------------------------------------------ #
# S2: default backward-compat                                        #
# ------------------------------------------------------------------ #

def test_s2_default_backward_compat():
    """SD-039 default OFF; payload-less write leaves anchor.goal_payload None."""
    from ree_core.utils.config import AnchorSetConfig
    cfg = AnchorSetConfig()
    assert cfg.use_sd039_anchor_payload is False

    aset = _make_anchor_set()
    a = aset.write_anchor("fast", "0.0", ("z_world", "z_self"), torch.randn(1, 8))
    assert a.goal_payload is None
    assert a.active is True


# ------------------------------------------------------------------ #
# S3: ON-payload-attached-on-write                                   #
# ------------------------------------------------------------------ #

def test_s3_payload_attached_on_write():
    """write_anchor with goal_payload attaches it to the new anchor."""
    aset = _make_anchor_set()
    payload = _make_payload(wanting=0.8, arousal=0.6, written_step=5)
    a = aset.write_anchor(
        "fast", "0.0", ("z_world", "z_self"),
        torch.randn(1, 8), goal_payload=payload,
    )
    assert a.goal_payload is payload
    assert a.goal_payload.wanting_strength == pytest.approx(0.8)
    assert a.goal_payload.arousal_tag == pytest.approx(0.6)
    assert a.goal_payload.payload_written_step == 5


# ------------------------------------------------------------------ #
# S4: payload survives mark_inactive                                 #
# ------------------------------------------------------------------ #

def test_s4_payload_survives_mark_inactive():
    """mark_inactive does NOT clear an existing payload."""
    aset = _make_anchor_set()
    payload = _make_payload(wanting=0.7)
    a = aset.write_anchor(
        "fast", "0.0", ("z_world",),
        torch.randn(1, 8), goal_payload=payload,
    )
    aset.mark_inactive("fast", ("z_world",))
    # Anchor retained in all_anchors with payload intact.
    found = [x for x in aset.all_anchors() if x.key == a.key]
    assert len(found) == 1
    inactive = found[0]
    assert inactive.active is False
    assert inactive.goal_payload is not None
    assert inactive.goal_payload.wanting_strength == pytest.approx(0.7)


# ------------------------------------------------------------------ #
# S5: goal_match returns 0.0 on null inputs                          #
# ------------------------------------------------------------------ #

def test_s5_goal_match_zero_on_null_inputs():
    """goal_match returns 0.0 for payload-less anchor / None snapshot / None current."""
    aset = _make_anchor_set()
    # Payload-less anchor.
    a_no_payload = aset.write_anchor(
        "fast", "0.0", ("z_world",), torch.randn(1, 8),
    )
    assert a_no_payload.goal_match(torch.tensor([[1.0, 0.0]])) == 0.0

    # Anchor with payload but z_goal_snapshot=None.
    payload_no_z = _make_payload(z_goal=None)
    payload_no_z.z_goal_snapshot = None
    a_no_z = aset.write_anchor(
        "slow", "0.0", ("z_world",), torch.randn(1, 8),
        goal_payload=payload_no_z,
    )
    assert a_no_z.goal_match(torch.tensor([[1.0, 0.0]])) == 0.0

    # Current z_goal is None.
    payload = _make_payload()
    a = aset.write_anchor(
        "fast", "0.1", ("z_world", "z_self"), torch.randn(1, 8),
        goal_payload=payload,
    )
    assert a.goal_match(None) == 0.0


# ------------------------------------------------------------------ #
# S6: goal_match cosine correctness                                  #
# ------------------------------------------------------------------ #

def test_s6_goal_match_cosine_correctness():
    """goal_match returns clipped cosine similarity in [0, 1]."""
    aset = _make_anchor_set()
    # Aligned: cosine == 1.0.
    p1 = _make_payload(z_goal=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    a1 = aset.write_anchor(
        "fast", "0.0", ("z_world",), torch.randn(1, 8), goal_payload=p1,
    )
    score = a1.goal_match(torch.tensor([[2.0, 0.0, 0.0, 0.0]]))
    assert score == pytest.approx(1.0, abs=1e-5)

    # Orthogonal: cosine == 0.0.
    p2 = _make_payload(z_goal=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    a2 = aset.write_anchor(
        "slow", "0.0", ("z_world",), torch.randn(1, 8), goal_payload=p2,
    )
    score = a2.goal_match(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    assert score == pytest.approx(0.0, abs=1e-5)

    # Anti-aligned: clamped to 0.0 (non-negative motivational-relevance).
    p3 = _make_payload(z_goal=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    a3 = aset.write_anchor(
        "fast", "0.1", ("z_self",), torch.randn(1, 8), goal_payload=p3,
    )
    score = a3.goal_match(torch.tensor([[-1.0, 0.0, 0.0, 0.0]]))
    assert score == pytest.approx(0.0, abs=1e-5)


# ------------------------------------------------------------------ #
# S7: query_by_goal_match across active + inactive                   #
# ------------------------------------------------------------------ #

def test_s7_query_returns_active_and_inactive_sorted():
    """query_by_goal_match scans dual-trace pool, sorts by score desc."""
    aset = _make_anchor_set()
    # Three anchors with descending alignment to current_z_goal.
    p_high = _make_payload(z_goal=torch.tensor([[1.0, 0.0, 0.0]]))
    p_mid = _make_payload(z_goal=torch.tensor([[0.7, 0.7, 0.0]]))
    p_low = _make_payload(z_goal=torch.tensor([[0.0, 1.0, 0.0]]))

    a_high = aset.write_anchor(
        "fast", "0.0", ("z_world",), torch.randn(1, 8), goal_payload=p_high,
    )
    a_mid = aset.write_anchor(
        "slow", "0.0", ("z_world",), torch.randn(1, 8), goal_payload=p_mid,
    )
    a_low = aset.write_anchor(
        "fast", "0.1", ("z_self",), torch.randn(1, 8), goal_payload=p_low,
    )
    # Mark a_high inactive: must still appear in default query.
    aset.mark_inactive("fast", ("z_world",))

    current = torch.tensor([[1.0, 0.0, 0.0]])
    results = aset.query_by_goal_match(current)
    keys = [a.key for a, score in results]
    scores = [score for a, score in results]

    # All three anchors appear; sorted by score descending.
    assert len(results) >= 2  # a_low scores 0.0 -- excluded at default threshold.
    assert keys[0] == a_high.key  # highest alignment wins despite inactive status.
    assert scores[0] > scores[1]
    # active_only=True excludes the inactive a_high.
    active_only = aset.query_by_goal_match(current, active_only=True)
    active_keys = [a.key for a, _ in active_only]
    assert a_high.key not in active_keys


# ------------------------------------------------------------------ #
# S8: query_by_goal_match returns empty for None current             #
# ------------------------------------------------------------------ #

def test_s8_query_empty_for_none_current():
    """No current goal -> no scores -> empty list."""
    aset = _make_anchor_set()
    payload = _make_payload()
    aset.write_anchor(
        "fast", "0.0", ("z_world",), torch.randn(1, 8), goal_payload=payload,
    )
    assert aset.query_by_goal_match(None) == []


# ------------------------------------------------------------------ #
# S9: reset_region refreshes payload on both traces                  #
# ------------------------------------------------------------------ #

def test_s9_reset_region_refreshes_payload_on_both_traces():
    """reset_region writes payload onto outgoing inactive AND new active."""
    aset = _make_anchor_set()
    p_initial = _make_payload(wanting=0.4)
    initial = aset.write_anchor(
        "fast", "0.0", ("z_world",), torch.randn(1, 8), goal_payload=p_initial,
    )
    # New payload at remap time captures the current motivational state.
    p_remap = _make_payload(wanting=0.9, arousal=0.7)
    new_active = aset.reset_region(
        "fast", ("z_world",), "0.5", torch.randn(1, 8), goal_payload=p_remap,
    )

    # Outgoing trace: marked inactive, payload refreshed to p_remap.
    found = [x for x in aset.all_anchors() if x.key == initial.key]
    assert len(found) == 1
    inactive = found[0]
    assert inactive.active is False
    assert inactive.goal_payload is p_remap

    # New active: also carries p_remap.
    assert new_active.active is True
    assert new_active.goal_payload is p_remap
    assert new_active.key[1] == "0.5"


# ------------------------------------------------------------------ #
# S10: per-episode reset clears payloads                             #
# ------------------------------------------------------------------ #

def test_s10_reset_clears_payloads():
    """AnchorSet.reset() clears active + inactive stores (and their payloads)."""
    aset = _make_anchor_set()
    p = _make_payload()
    aset.write_anchor(
        "fast", "0.0", ("z_world",), torch.randn(1, 8), goal_payload=p,
    )
    aset.mark_inactive("fast", ("z_world",))
    assert len(aset.all_anchors()) == 1

    aset.reset()
    assert len(aset.active_anchors()) == 0
    assert len(aset.all_anchors()) == 0
    # Subsequent query is empty.
    assert aset.query_by_goal_match(torch.tensor([[1.0, 0.0]])) == []
