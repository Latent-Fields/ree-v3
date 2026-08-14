"""Contracts for experiments/_lib/trajectory_metrics.py.

Three things this pins:
  (a) `spatial_trajectory_stats` reproduces the ORIGINAL
      `_trajectory_organization_stats` body bit-for-bit on a range of
      fixtures, including its degenerate branches -- the "verbatim port"
      promise from
      behavioural_trajectory_metrics_library_scoping_2026-08-11.md Section 5
      item 1 is a claim this test enforces, not just states. It is pinned
      against GOLDEN literals rather than against the driver, because the
      driver no longer holds a second copy to compare to (see below).
  (b) `v3_exq_913_developmental_ecology_fishtank._trajectory_organization_stats`
      DELEGATES to this module faithfully -- forwarding `hazard_positions`,
      `window`, and its own `HAZARD_NEAR_RADIUS`.
  (c) the new `action_sequence_stats` / `degeneracy_flags` functions behave
      per their own docstrings on hand-worked fixtures, including the
      world-rule-shift `spans_rule_shift` True/False/None distinction that
      is the whole point of that function existing.

WHY (a) IS A GOLDEN PIN AND NOT A TWO-IMPLEMENTATION COMPARISON. When this
file landed (ree-v3 3200863ba5) the driver still carried the original body,
so asserting `spatial_trajectory_stats(...) == _trajectory_organization_stats(...)`
compared two independent implementations and was a real regression test. The
driver was then refactored to delegate to this module, which would have
silently reduced that assertion to a self-comparison -- structurally unable
to fail, and leaving NOTHING pinning either side's absolute behaviour. The
GOLDEN values below were captured from the pre-delegation driver body
(v3_exq_913 at ree-v3 b2454447, lines 387-472) and verified bit-identical
across those fixtures plus 400 randomized walks, so the port promise stays
enforced against real prior-art numbers after the second copy is gone.
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

# Captured from the PRE-delegation `_trajectory_organization_stats` body
# (v3_exq_913_developmental_ecology_fishtank.py:387-472 at ree-v3 b2454447),
# evaluated with `hazard_positions=HAZARD_POSITIONS` and no window. Full
# repr precision on purpose: `turning_angle_entropy_bits` is a
# -1.44e-12 near-zero that a rounded literal would silently mask, and it is
# exactly the kind of arithmetic detail a "verbatim port" claim is about.
# Every branch of the function is represented -- degenerate n<2 early return
# (DEGENERATE_ONE_STEP), all-null headings (STATIC), zero net displacement so
# tortuosity is None (RETURN_TO_ORIGIN), n==2 so no turning-angle sample is
# possible (SINGLE_STEP), and three ordinary paths.
GOLDEN_WITH_HAZARDS = {
    "STRAIGHT_LINE": {
        'n_steps': 5,
        'turning_angle_mean': 0.0,
        'turning_angle_entropy_bits': -1.4428232973175175e-12,
        'mean_straight_run_length': 4.0,
        'max_straight_run_length': 4,
        'tortuosity': 1.0,
        'path_length': 4,
        'net_displacement': 4,
        'turning_near_hazard_mean': 0.0,
        'turning_far_hazard_mean': None,
        'n_turning_near_hazard': 3,
        'n_turning_far_hazard': 0,
    },
    "L_TURN": {
        'n_steps': 5,
        'turning_angle_mean': 0.5235987755982988,
        'turning_angle_entropy_bits': 0.9182958340516041,
        'mean_straight_run_length': 2.0,
        'max_straight_run_length': 2,
        'tortuosity': 1.0,
        'path_length': 4,
        'net_displacement': 4,
        'turning_near_hazard_mean': 0.5235987755982988,
        'turning_far_hazard_mean': None,
        'n_turning_near_hazard': 3,
        'n_turning_far_hazard': 0,
    },
    "ZIGZAG": {
        'n_steps': 6,
        'turning_angle_mean': 1.5707963267948966,
        'turning_angle_entropy_bits': -1.4428232973175175e-12,
        'mean_straight_run_length': 1.0,
        'max_straight_run_length': 1,
        'tortuosity': 1.0,
        'path_length': 5,
        'net_displacement': 5,
        'turning_near_hazard_mean': 1.5707963267948966,
        'turning_far_hazard_mean': None,
        'n_turning_near_hazard': 4,
        'n_turning_far_hazard': 0,
    },
    "RETURN_TO_ORIGIN": {
        'n_steps': 5,
        'turning_angle_mean': 1.0471975511965976,
        'turning_angle_entropy_bits': 0.9182958340516041,
        'mean_straight_run_length': 2.0,
        'max_straight_run_length': 2,
        'tortuosity': None,
        'path_length': 4,
        'net_displacement': 0,
        'turning_near_hazard_mean': 1.0471975511965976,
        'turning_far_hazard_mean': None,
        'n_turning_near_hazard': 3,
        'n_turning_far_hazard': 0,
    },
    "SINGLE_STEP": {
        'n_steps': 2,
        'turning_angle_mean': None,
        'turning_angle_entropy_bits': None,
        'mean_straight_run_length': 1.0,
        'max_straight_run_length': 1,
        'tortuosity': 1.0,
        'path_length': 1,
        'net_displacement': 1,
        'turning_near_hazard_mean': None,
        'turning_far_hazard_mean': None,
        'n_turning_near_hazard': 0,
        'n_turning_far_hazard': 0,
    },
    "DEGENERATE_ONE_STEP": {
        'n_steps': 1,
    },
    "STATIC": {
        'n_steps': 4,
        'turning_angle_mean': None,
        'turning_angle_entropy_bits': None,
        'mean_straight_run_length': None,
        'max_straight_run_length': None,
        'tortuosity': None,
        'path_length': 0,
        'net_displacement': 0,
        'turning_near_hazard_mean': None,
        'turning_far_hazard_mean': None,
        'n_turning_near_hazard': 0,
        'n_turning_far_hazard': 0,
    },
}

NAMED_FIXTURES = {
    "STRAIGHT_LINE": STRAIGHT_LINE,
    "L_TURN": L_TURN,
    "ZIGZAG": ZIGZAG,
    "RETURN_TO_ORIGIN": RETURN_TO_ORIGIN,
    "SINGLE_STEP": SINGLE_STEP,
    "DEGENERATE_ONE_STEP": DEGENERATE_ONE_STEP,
    "STATIC": STATIC,
}


# --- spatial_trajectory_stats: bit-identical regression against the source ---

@pytest.mark.parametrize("name", sorted(GOLDEN_WITH_HAZARDS))
def test_spatial_trajectory_stats_matches_source_golden_values(name):
    """The port-verbatim promise, pinned against the ORIGINAL body's numbers.

    Exact `==` on purpose, not `pytest.approx`: "verbatim port" is a claim
    about bit-identity, and an approx comparison would accept a
    reimplementation that merely agrees to a tolerance.
    """
    actual = tm.spatial_trajectory_stats(
        _steps(NAMED_FIXTURES[name]), hazard_positions=HAZARD_POSITIONS,
    )
    assert actual == GOLDEN_WITH_HAZARDS[name]


def test_golden_fixture_set_covers_every_fixture():
    # Guards against a fixture being added to FIXTURES for the delegation
    # tests below while quietly acquiring no golden pin of its own.
    assert [list(p) for p in NAMED_FIXTURES.values()] == [list(p) for p in FIXTURES]
    assert set(GOLDEN_WITH_HAZARDS) == set(NAMED_FIXTURES)


# --- the driver's wrapper delegates faithfully --------------------------------
#
# These used to compare two independent implementations. Post-refactor the
# driver delegates to this module, so what they now pin is the FORWARDING:
# that the wrapper passes `hazard_positions` and `window` through, and that
# its own HAZARD_NEAR_RADIUS reaches the library's `hazard_near_radius`
# parameter rather than being dropped in favour of the library default. A
# wrapper that silently swallowed `window`, or hardcoded a different radius,
# fails here.

@pytest.mark.parametrize("hazard_positions", [None, HAZARD_POSITIONS])
@pytest.mark.parametrize("positions", FIXTURES)
def test_driver_wrapper_delegates_bit_for_bit(positions, hazard_positions):
    steps = _steps(positions)
    expected = tm.spatial_trajectory_stats(
        steps, hazard_positions=hazard_positions,
        hazard_near_radius=SOURCE_HAZARD_NEAR_RADIUS,
    )
    actual = _trajectory_organization_stats(steps, hazard_positions)
    assert actual == expected


@pytest.mark.parametrize("positions", FIXTURES)
def test_driver_wrapper_forwards_window(positions):
    steps = _steps(positions)
    window = max(1, len(positions) - 1)
    expected = tm.spatial_trajectory_stats(
        steps, hazard_positions=HAZARD_POSITIONS, window=window,
        hazard_near_radius=SOURCE_HAZARD_NEAR_RADIUS,
    )
    actual = _trajectory_organization_stats(steps, HAZARD_POSITIONS, window=window)
    assert actual == expected


def test_driver_wrapper_forwards_its_own_hazard_near_radius():
    # A wrapper that dropped `hazard_near_radius` would be indistinguishable
    # from a correct one while the two constants agree. Compare against a
    # DELIBERATELY WRONG radius to prove the driver is not merely inheriting
    # the library default by coincidence.
    steps = _steps([(20, 20), (21, 20), (21, 21), (22, 21)])
    out = _trajectory_organization_stats(steps, HAZARD_POSITIONS)
    wide = tm.spatial_trajectory_stats(
        steps, hazard_positions=HAZARD_POSITIONS, hazard_near_radius=100,
    )
    assert out["n_turning_near_hazard"] == 0
    assert wide["n_turning_near_hazard"] == 2
    assert out != wide


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
