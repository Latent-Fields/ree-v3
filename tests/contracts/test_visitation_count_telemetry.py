"""
Contract tests for the unconditional visitation-count telemetry
(chip-20260810-fishtank-visitation-count-telemetry).

REE_assembly/evidence/planning/developmental_ecology_curiosity_foraging_correction_2026-08-10.md
Section 6 names a per-state visitation count among the untelemetered signals
needed to discriminate curiosity-driven discovery from diffuse-gradient
exploitation. ree_core/hippocampal/visitation.py's VisitationCounter closes that
gap: unlike SD-025's FamiliarityTracker (curiosity_weight-gated) or
REEAgent._zworld_visitation_buffer (MECH-314 use_structured_curiosity-gated),
it is instantiated UNCONDITIONALLY by HippocampalModule.__init__ -- no flag
gates it -- so a per-step visit count is available in every fishtank-family
experiment regardless of which curiosity substrate is enabled.

Contracts:
  C1  VisitationCounter mechanics -- empty region -> 0; revisiting the same
      region raises the count; a distant region stays at 0.
  C2  update() returns the PRE-increment count (0 on a first visit, the
      PRIOR count on a revisit) -- mirrors compute_novelty_score's
      pre-visit-read convention so a caller does not need a separate
      query() call.
  C3  query() is read-only -- repeated calls do not mutate state or advance
      the count; matches update()'s own count afterward.
  C4  Always-on invariant -- HippocampalModule.visitation_counter is a real
      VisitationCounter (not None) and record_visitation() advances it with
      curiosity_weight=0.0 (SD-025 off) and no MECH-314 flag set, in direct
      contrast to familiarity_tracker, which IS None in that same config.
  C5  MECH-094 gate -- record_visitation(is_waking=False) does not advance
      the counter and returns None (replay / DMN ticks are not genuine
      per-tick observations).
  C6  last_visitation_count cache + experiments._lib.hippocampal_telemetry.
      read_state_visitation_count -- populated after a waking
      record_visitation() call, read via the shared one-import helper,
      unaffected by a non-waking call.
"""

import torch

from ree_core.utils.config import HippocampalConfig, ResidueConfig, E2Config
from ree_core.residue.field import ResidueField
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.hippocampal.module import HippocampalModule
from ree_core.hippocampal.visitation import VisitationCounter
from experiments._lib.hippocampal_telemetry import read_state_visitation_count

WORLD_DIM = 8
ACTION_DIM = 4
ACTION_OBJECT_DIM = 16


def _residue():
    cfg = ResidueConfig(world_dim=WORLD_DIM)
    return ResidueField(cfg)


def _hip(residue, curiosity_weight=0.0):
    cfg = HippocampalConfig(
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM,
        hidden_dim=64,
        horizon=5,
        num_candidates=4,
        num_cem_iterations=1,
        curiosity_weight=curiosity_weight,
    )
    e2 = E2FastPredictor(E2Config(
        self_dim=6, world_dim=WORLD_DIM, action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM, rollout_horizon=5, num_candidates=4,
    ))
    return HippocampalModule(cfg, e2, residue)


def _loc(coord, dim0=0):
    z = torch.zeros(1, WORLD_DIM)
    z[0, dim0] = coord
    return z


# ---------------------------------------------------------------------------
# C1: VisitationCounter mechanics
# ---------------------------------------------------------------------------

def test_c1_mechanics_empty_revisit_distant():
    vc = VisitationCounter(world_dim=WORLD_DIM, num_anchors=32, bandwidth=0.20)
    z = _loc(0.0)
    assert vc.query(z) == 0                        # empty -> never visited
    vc.update(z)
    assert vc.query(z) == 1                         # first visit -> count 1
    for _ in range(4):
        vc.update(z)
    assert vc.query(z) == 5                          # revisits raise the count
    assert vc.query(_loc(6.0)) == 0                  # distant region untouched


# ---------------------------------------------------------------------------
# C2: update() returns the PRE-increment count
# ---------------------------------------------------------------------------

def test_c2_update_returns_pre_increment_count():
    vc = VisitationCounter(world_dim=WORLD_DIM, num_anchors=32, bandwidth=0.20)
    z = _loc(1.0)
    assert vc.update(z) == 0                          # first visit -> pre-count 0
    assert vc.update(z) == 1                          # second visit -> pre-count 1
    assert vc.update(z) == 2                          # third visit -> pre-count 2
    assert vc.query(z) == 3                            # post-state: 3 total visits


# ---------------------------------------------------------------------------
# C3: query() is read-only
# ---------------------------------------------------------------------------

def test_c3_query_is_read_only():
    vc = VisitationCounter(world_dim=WORLD_DIM, num_anchors=32, bandwidth=0.20)
    z = _loc(2.0)
    vc.update(z)
    before = vc.query(z)
    for _ in range(10):
        vc.query(z)
    after = vc.query(z)
    assert before == after == 1
    assert int(vc.active_mask.sum()) == 1


# ---------------------------------------------------------------------------
# C4: always-on invariant
# ---------------------------------------------------------------------------

def test_c4_always_on_with_curiosity_off():
    residue = _residue()
    hip = _hip(residue, curiosity_weight=0.0)
    # SD-025 is off in this config -> no FamiliarityTracker.
    assert hip.familiarity_tracker is None
    # visitation_counter is NEVER gated -> always a real tracker.
    assert isinstance(hip.visitation_counter, VisitationCounter)

    z = _loc(3.0)
    assert hip.record_visitation(z, is_waking=True) == 0
    assert hip.record_visitation(z, is_waking=True) == 1
    assert hip.visitation_counter.query(z) == 2


def test_c4_also_always_on_with_curiosity_enabled():
    """Confirms the two substrates are independent -- turning SD-025 ON does
    not change visitation_counter's own count semantics."""
    residue = _residue()
    hip = _hip(residue, curiosity_weight=0.5)
    assert hip.familiarity_tracker is not None
    assert isinstance(hip.visitation_counter, VisitationCounter)

    z = _loc(3.0)
    assert hip.record_visitation(z, is_waking=True) == 0
    assert hip.record_visitation(z, is_waking=True) == 1


# ---------------------------------------------------------------------------
# C5: MECH-094 gate
# ---------------------------------------------------------------------------

def test_c5_mech094_replay_does_not_advance_count():
    residue = _residue()
    hip = _hip(residue, curiosity_weight=0.0)
    z = _loc(4.0)

    assert hip.record_visitation(z, is_waking=False) is None
    assert hip.visitation_counter.query(z) == 0        # replay wrote no memory

    assert hip.record_visitation(z, is_waking=True) == 0
    assert hip.record_visitation(z, is_waking=False) is None
    assert hip.visitation_counter.query(z) == 1         # unchanged by the replay call


# ---------------------------------------------------------------------------
# C6: last_visitation_count cache + shared _lib read
# ---------------------------------------------------------------------------

def test_c6_cache_and_shared_lib_read():
    residue = _residue()
    hip = _hip(residue, curiosity_weight=0.0)

    class _FakeAgent:
        pass

    agent = _FakeAgent()
    agent.hippocampal = hip

    assert read_state_visitation_count(agent) is None   # before any waking tick

    z = _loc(5.0)
    # Emulate agent.py's sense() call site: record, then cache when waking.
    pre = hip.record_visitation(z, is_waking=True)
    hip.last_visitation_count = pre
    assert read_state_visitation_count(agent) == 0

    pre2 = hip.record_visitation(z, is_waking=True)
    hip.last_visitation_count = pre2
    assert read_state_visitation_count(agent) == 1

    # A non-waking (replay) call must not overwrite the cache -- mirrors
    # agent.py's `if is_waking:` guard around the cache assignment.
    hip.record_visitation(z, is_waking=False)
    assert read_state_visitation_count(agent) == 1


def test_c6_read_helper_none_when_no_hippocampal():
    class _FakeAgent:
        pass

    agent = _FakeAgent()
    agent.hippocampal = None
    assert read_state_visitation_count(agent) is None
