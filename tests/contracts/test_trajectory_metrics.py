"""Contracts for experiments/_lib/trajectory_metrics.py.

Two things this pins:
  (a) `spatial_trajectory_stats` reproduces `_trajectory_organization_stats`
      (v3_exq_913_developmental_ecology_fishtank.py) bit-for-bit on a range
      of fixtures, including its degenerate branches -- the "verbatim port"
      promise from
      behavioural_trajectory_metrics_library_scoping_2026-08-11.md Section 5
      item 1 is a claim this test enforces, not just states.
  (b) the new `action_sequence_stats` / `degeneracy_flags` functions behave
      per their own docstrings on hand-worked fixtures, including the
      world-rule-shift `spans_rule_shift` True/False/None distinction that
      is the whole point of that function existing.
"""

from __future__ import annotations

import pytest

from experiments._lib import trajectory_metrics as tm
from experiments.v3_exq_913_developmental_ecology_fishtank import (
    HAZARD_NEAR_RADIUS as SOURCE_HAZARD_NEAR_RADIUS,
    _trajectory_organization_stats,
)
from ree_core.environment.causal_grid_world import CausalGridWorld


def _steps(positions):
    return [{"pos": list(p)} for p in positions]


STRAIGHT_LINE = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
L_TURN = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
ZIGZAG = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]
RETURN_TO_ORIGIN = [(0, 0), (1, 0), (2, 0), (1, 0), (0, 0)]  # net_displacement == 0
SINGLE_STEP = [(0, 0), (1, 0)]  # n == 2: one delta, no turning-angle sample possible
DEGENERATE_ONE_STEP = [(0, 0)]  # n == 1
STATIC = [(3, 3), (3, 3), (3, 3), (3, 3)]  # every delta is (0, 0)

FIXTURES = [STRAIGHT_LINE, L_TURN, ZIGZAG, RETURN_TO_ORIGIN, SINGLE_STEP,
            DEGENERATE_ONE_STEP, STATIC]

HAZARD_POSITIONS = [(2, 1), (5, 5)]


# --- spatial_trajectory_stats: bit-identical regression against the source ---

@pytest.mark.parametrize("hazard_positions", [None, HAZARD_POSITIONS])
@pytest.mark.parametrize("positions", FIXTURES)
def test_spatial_trajectory_stats_matches_source_bit_for_bit(positions, hazard_positions):
    steps = _steps(positions)
    expected = _trajectory_organization_stats(steps, hazard_positions)
    actual = tm.spatial_trajectory_stats(steps, hazard_positions=hazard_positions)
    assert actual == expected


@pytest.mark.parametrize("positions", FIXTURES)
def test_spatial_trajectory_stats_matches_source_with_window(positions):
    steps = _steps(positions)
    window = max(1, len(positions) - 1)
    expected = _trajectory_organization_stats(steps, HAZARD_POSITIONS, window=window)
    actual = tm.spatial_trajectory_stats(steps, hazard_positions=HAZARD_POSITIONS, window=window)
    assert actual == expected


def test_spatial_trajectory_stats_default_hazard_near_radius_matches_source_constant():
    assert tm.HAZARD_NEAR_RADIUS == SOURCE_HAZARD_NEAR_RADIUS


def test_spatial_trajectory_stats_hazard_near_radius_is_overridable():
    # a path far from both hazards, so the default radius=3 puts every
    # turning-angle sample in "far" -- verified by direct computation
    # (nearest-hazard Manhattan distances 31, 32), not assumed.
    far_positions = [(20, 20), (21, 20), (21, 21), (22, 21)]
    steps = _steps(far_positions)
    default = tm.spatial_trajectory_stats(steps, hazard_positions=HAZARD_POSITIONS)
    assert default["n_turning_near_hazard"] == 0
    assert default["n_turning_far_hazard"] == 2
    widened = tm.spatial_trajectory_stats(
        steps, hazard_positions=HAZARD_POSITIONS, hazard_near_radius=100,
    )
    # a radius wide enough to cover the whole board reclassifies every
    # turning-angle sample as "near" -- confirms the parameter is live, not
    # a decorative default that the function ignores.
    assert widened["n_turning_far_hazard"] == 0
    assert widened["n_turning_near_hazard"] == 2


# --- action_sequence_stats ---------------------------------------------------

def test_action_sequence_stats_reversal_rate_worked_example():
    # actions: 0(left),1(right),0(left),4(stay),3(down),2(up)
    # transitions: (0,1) reversal, (1,0) reversal, (0,4) no, (4,3) no,
    # (3,2) reversal -- 3 of 5.
    actions = [0, 1, 0, 4, 3, 2]
    out = tm.action_sequence_stats(actions)
    assert out["n_actions"] == 6
    assert out["n_transitions"] == 5
    assert out["reversal_count"] == 3
    assert out["reversal_rate"] == pytest.approx(3 / 5)


def test_action_sequence_stats_repeat_rate_and_run_length_worked_example():
    actions = [1, 1, 1, 2, 2, 4]  # runs: [3, 2, 1]
    out = tm.action_sequence_stats(actions)
    assert out["repeat_count"] == 3  # (1,1), (1,1), (2,2)
    assert out["repeat_rate"] == pytest.approx(3 / 5)
    assert out["mean_run_length"] == pytest.approx((3 + 2 + 1) / 3)
    assert out["max_run_length"] == 3


def test_action_sequence_stats_default_inverse_pairs_match_causal_grid_world():
    for a, b in tm.DEFAULT_ACTION_INVERSE_PAIRS.items():
        da = CausalGridWorld.ACTIONS[a]
        db = CausalGridWorld.ACTIONS[b]
        assert (da[0] + db[0], da[1] + db[1]) == (0, 0)
    # action 4 (stay) and 5 (consume, when enabled) are correctly absent --
    # neither has a spatial inverse.
    assert 4 not in tm.DEFAULT_ACTION_INVERSE_PAIRS
    assert 5 not in tm.DEFAULT_ACTION_INVERSE_PAIRS


def test_action_sequence_stats_custom_inverse_pairs_override():
    # a toy action space where 0<->2 are declared inverse instead of the default
    actions = [0, 2, 0]
    out = tm.action_sequence_stats(actions, inverse_pairs={0: 2, 2: 0})
    assert out["reversal_count"] == 2


def test_action_sequence_stats_spans_rule_shift_true_false_and_none():
    actions = [0, 1, 0, 1, 0]  # window covers global indices [0, 5)
    inside = tm.action_sequence_stats(actions, rule_shift_boundaries=[3])
    assert inside["spans_rule_shift"] is True
    outside = tm.action_sequence_stats(actions, rule_shift_boundaries=[99])
    assert outside["spans_rule_shift"] is False
    # no boundary information supplied at all -- must be None, not False,
    # per the module docstring's fail-informative-not-fail-silent design.
    unknown = tm.action_sequence_stats(actions)
    assert unknown["spans_rule_shift"] is None


def test_action_sequence_stats_window_start_index_offsets_boundary_check():
    actions = [0, 1, 0, 1, 0]
    # this window starts at global index 100 and runs to 105; a boundary at
    # global index 3 must NOT be reported as inside this window.
    out = tm.action_sequence_stats(actions, rule_shift_boundaries=[3], window_start_index=100)
    assert out["spans_rule_shift"] is False
    out2 = tm.action_sequence_stats(actions, rule_shift_boundaries=[102], window_start_index=100)
    assert out2["spans_rule_shift"] is True


def test_action_sequence_stats_degenerate_single_action():
    out = tm.action_sequence_stats([2])
    assert out["n_actions"] == 1
    assert out["reversal_rate"] is None
    assert out["mean_run_length"] is None
    assert out["spans_rule_shift"] is None


# --- degeneracy_flags ---------------------------------------------------------

def test_degeneracy_flags_flags_near_static_window():
    # the exact shape of the V3-EXQ-913 seed1/no_sleep/seg9 artefact this
    # function was added to catch: a few real ticks, then motionless.
    positions = [(0, 0), (1, 0), (2, 0), (3, 0)] + [(3, 0)] * 96
    steps = _steps(positions)
    out = tm.degeneracy_flags(steps)
    assert out["static_frac"] > 0.9
    assert out["static_frac_exceeds_ceiling"] is True


def test_degeneracy_flags_clean_window_not_flagged():
    positions = [(i, 0) for i in range(20)]
    steps = _steps(positions)
    out = tm.degeneracy_flags(steps)
    assert out["static_frac_exceeds_ceiling"] is False
    assert out["net_displacement_zero"] is False


def test_degeneracy_flags_net_displacement_zero():
    out = tm.degeneracy_flags(_steps(RETURN_TO_ORIGIN))
    assert out["net_displacement_zero"] is True


def test_degeneracy_flags_below_turning_floor():
    # two points -> only one heading -> zero turning-angle SAMPLES are even
    # possible (a sample needs two consecutive non-null headings) -- 0 is
    # below any positive floor, including the default.
    out = tm.degeneracy_flags(_steps(SINGLE_STEP))
    assert out["n_turning_samples"] == 0
    assert out["turning_samples_below_floor"] is True


def test_degeneracy_flags_degenerate_single_step():
    out = tm.degeneracy_flags(_steps(DEGENERATE_ONE_STEP))
    assert out["n_steps"] == 1
    assert out["static_frac"] is None
    assert out["turning_samples_below_floor"] is True
