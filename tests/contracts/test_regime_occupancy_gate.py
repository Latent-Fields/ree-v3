"""Contracts for experiments/_lib/regime_occupancy_gate.py.

TWO governing requirements, from two different confirmed defects. Both are
live -- the second does not replace the first.

REQUIREMENT 1, from `failure_autopsy_mech266-464e-467e-cluster_2026-08-13.md`
(sections 1-2):

    A `min(fractions) > floor` non-vacuity gate cannot distinguish "the mode is
    never occupied" from "the mode is occupied but saturated (0/1 step
    function, no mixed regime in the sweep)". Both read `min == 0` (or below
    floor), and the old gate emits the SAME route_reason
    ("external_task_mode_not_occupied") for both -- factually false in the
    saturated case, which routes substrate work in the wrong direction.

REQUIREMENT 2, from `mode-governance-engagement`'s `severity_note` (severity
`corrupting`) and its two OPEN failure_record targets:

    The FIRST fix's own "graded" predicate was a flat per-cell EXISTENTIAL
    (`any(floor < f < ceiling for f in fracs)`). One mixed cell out of N set
    `route_reason = None`, i.e. PASS. The bar those targets actually state is
    reproducibility: mixed on >= 2/3 SEEDS at >= 2 ADJACENT swept values.
    V3-EXQ-934 routed `cap_recalibration_admits_mixed_regime` from data in
    which two seeds were mixed at DISJOINT ends of the cap sweep -- evidence
    that LOOKS valid and is not.

These tests replay the CONFIRMED numbers from all three targets (464e, 467e,
934) and assert the gate reports the true regime shape in each case.
"""

import pytest

from experiments._lib.regime_occupancy_gate import (
    DEFAULT_MIN_ADJACENT,
    DEFAULT_MIN_SEED_FRACTION,
    DEFAULT_MIN_SEEDS,
    OccupancyCell,
    classify_regime_shape,
    evaluate_regime_occupancy_gate,
)

OCCUPANCY_FLOOR = 0.10

# --- V3-EXQ-464e replay: seed 42, two-arm contrast (autopsy Section 1 table) - #
# ARM_SYMMETRIC = 0.0, ARM_ASYM_STICKY_TASK = 1.0. Old gate:
# occupancy_min = min(0.0, 1.0) = 0.0 -> "not occupied". FALSE: the sticky arm
# is occupied at 100%.
V3_EXQ_464E_SEED42 = [
    OccupancyCell(label="ARM_SYMMETRIC", fraction=0.0),
    OccupancyCell(label="ARM_ASYM_STICKY_TASK", fraction=1.0),
]

# seed 44: ARM_SYMMETRIC = 0.4331 (genuinely mixed), ARM_ASYM = 1.0.
V3_EXQ_464E_SEED44 = [
    OccupancyCell(label="ARM_SYMMETRIC", fraction=0.4331),
    OccupancyCell(label="ARM_ASYM_STICKY_TASK", fraction=1.0),
]

# --- V3-EXQ-467e replay: seed 42, 5-point hysteresis-ratio sweep -------------- #
# r=0.10 -> 1.0, r>=0.50 -> exactly 0.0 at every ratio (autopsy Section 1 table).
# Old gate: occupancy_min = min(1.0, 0.0, 0.0, 0.0, 0.0) = 0.0 -> "not occupied".
# FALSE: r=0.10 is occupied at 100%.
V3_EXQ_467E_SEED42 = [
    OccupancyCell(label="r=0.10", fraction=1.0),
    OccupancyCell(label="r=0.50", fraction=0.0),
    OccupancyCell(label="r=1.00", fraction=0.0),
    OccupancyCell(label="r=1.50", fraction=0.0),
    OccupancyCell(label="r=2.00", fraction=0.0),
]

# A genuinely unreachable sweep (the PRE-2026-08-12 substrate signature: 0.0 on
# every seed/arm, e.g. v3_exq_464c) -- the gate must still call this correctly.
GENUINELY_UNREACHABLE = [
    OccupancyCell(label="ARM_SYMMETRIC", fraction=0.0),
    OccupancyCell(label="ARM_ASYM_STICKY_TASK", fraction=0.0),
]

# --- V3-EXQ-934 replay: the CORRUPTING case (requirement 2) ------------------- #
# Confirmed from the run's failure_record entry
# (`v3_exq_934_mech266_cap_sweep_mode_occupancy_20260815T015216Z_v3`):
#
#   "no single cap yields occupancy in (0.1,0.9) on more than 1 of 3 seeds
#    (0.75 -> 1/3; 1.0/1.25/1.5 -> 0/3; 1.75 -> 1/3). Per-seed mixed bands are
#    disjoint singletons at opposite ends of the sweep (seed 42 {0.75},
#    seed 43 {1.75}, seed 44 {})"
#
# The MIXED/NOT-MIXED PATTERN below is that confirmed record exactly. The
# individual NON-mixed fractions are stand-ins (the record states the pattern
# and the per-cap seed counts, not every raw fraction); every one is placed
# outside (0.1, 0.9) so it cannot affect any assertion here.
V3_EXQ_934_ARM_SYMMETRIC = [
    OccupancyCell(label="cap=0.75", fraction=0.42, seed=42, sweep_value=0.75),
    OccupancyCell(label="cap=1.0", fraction=1.0, seed=42, sweep_value=1.0),
    OccupancyCell(label="cap=1.25", fraction=1.0, seed=42, sweep_value=1.25),
    OccupancyCell(label="cap=1.5", fraction=1.0, seed=42, sweep_value=1.5),
    OccupancyCell(label="cap=1.75", fraction=1.0, seed=42, sweep_value=1.75),

    OccupancyCell(label="cap=0.75", fraction=0.0, seed=43, sweep_value=0.75),
    OccupancyCell(label="cap=1.0", fraction=0.0, seed=43, sweep_value=1.0),
    OccupancyCell(label="cap=1.25", fraction=0.0, seed=43, sweep_value=1.25),
    OccupancyCell(label="cap=1.5", fraction=0.0, seed=43, sweep_value=1.5),
    OccupancyCell(label="cap=1.75", fraction=0.58, seed=43, sweep_value=1.75),

    OccupancyCell(label="cap=0.75", fraction=0.0, seed=44, sweep_value=0.75),
    OccupancyCell(label="cap=1.0", fraction=0.0, seed=44, sweep_value=1.0),
    OccupancyCell(label="cap=1.25", fraction=1.0, seed=44, sweep_value=1.25),
    OccupancyCell(label="cap=1.5", fraction=1.0, seed=44, sweep_value=1.5),
    OccupancyCell(label="cap=1.75", fraction=1.0, seed=44, sweep_value=1.75),
]

# What a WORKING dose-response instrument looks like: a reproducible mixed band
# -- >= 2 ADJACENT r values, each mixed on >= 2/3 seeds.
GENUINELY_GRADED_SWEEP = [
    # r=0.10 saturated on every seed (by design -- the tight rail)
    OccupancyCell(label="r=0.10", fraction=1.0, seed=s, sweep_value=0.10)
    for s in (42, 43, 44)
] + [
    # r=0.50 and r=1.00: mixed on 3/3 and 2/3 -- the reproducible band
    OccupancyCell(label="r=0.50", fraction=0.71, seed=42, sweep_value=0.50),
    OccupancyCell(label="r=0.50", fraction=0.64, seed=43, sweep_value=0.50),
    OccupancyCell(label="r=0.50", fraction=0.58, seed=44, sweep_value=0.50),
    OccupancyCell(label="r=1.00", fraction=0.44, seed=42, sweep_value=1.00),
    OccupancyCell(label="r=1.00", fraction=0.39, seed=43, sweep_value=1.00),
    OccupancyCell(label="r=1.00", fraction=0.0, seed=44, sweep_value=1.00),
] + [
    # r=2.00 released on every seed (by design -- the loose rail)
    OccupancyCell(label="r=2.00", fraction=0.0, seed=s, sweep_value=2.00)
    for s in (42, 43, 44)
]


# ---------------------------------------------------------------------------- #
# REQUIREMENT 1 -- the min()-vs-any defect. Unchanged by the 2026-09-11 fix.
# ---------------------------------------------------------------------------- #

def test_464e_seed42_min_would_say_not_occupied_but_it_is_saturated():
    """The confirmed regression: min()==0.0 while one arm is 100% occupied."""
    old_min = min(c.fraction for c in V3_EXQ_464E_SEED42)
    assert old_min == 0.0  # the old gate's exact false-negative trigger

    gate = evaluate_regime_occupancy_gate(
        V3_EXQ_464E_SEED42, mode_label="external_task", floor=OCCUPANCY_FLOOR)

    assert gate["reachable"] is True
    assert gate["regime_shape"] == "saturated_bimodal"
    assert gate["route_reason"] == "external_task_mode_saturated_no_mixed_regime"
    assert gate["route_reason"] != "external_task_mode_not_occupied"
    assert gate["max_fraction"] == 1.0


def test_467e_seed42_step_function_reads_saturated_not_unreachable():
    old_min = min(c.fraction for c in V3_EXQ_467E_SEED42)
    assert old_min == 0.0

    gate = evaluate_regime_occupancy_gate(
        V3_EXQ_467E_SEED42, mode_label="external_task", floor=OCCUPANCY_FLOOR)

    assert gate["reachable"] is True
    assert gate["regime_shape"] == "saturated_bimodal"
    assert gate["route_reason"] == "external_task_mode_saturated_no_mixed_regime"
    # the anti-correlation defect (M2): a sweep DESIGNED to drive occupancy to
    # 0 at high r must not fail reachability just because the high-r cells are
    # (correctly, by design) at 0.
    assert gate["max_fraction"] == 1.0
    reachable_cells = [c for c in gate["cells"] if c["reachable"]]
    assert len(reachable_cells) == 1
    assert reachable_cells[0]["label"] == "r=0.10"


def test_genuinely_unreachable_sweep_still_reads_unreachable():
    """Pathology (a) must not be relabelled by the fix -- only (b) changes."""
    gate = evaluate_regime_occupancy_gate(
        GENUINELY_UNREACHABLE, mode_label="external_task", floor=OCCUPANCY_FLOOR)
    assert gate["reachable"] is False
    assert gate["regime_shape"] == "unreachable"
    assert gate["route_reason"] == "external_task_mode_unreachable"


def test_classify_regime_shape_empty_cells_is_unreachable():
    assert classify_regime_shape([], floor=OCCUPANCY_FLOOR) == "unreachable"


def test_min_across_sweep_would_have_blocked_scoring_new_gate_does_not():
    """End-to-end: replay what the OLD driver logic would have done vs the new
    gate, on the 464e seed-42 cells, to pin the exact behavioural difference.
    """
    fractions = [c.fraction for c in V3_EXQ_464E_SEED42]
    old_occupancy_non_vacuity = min(fractions) > OCCUPANCY_FLOOR
    assert old_occupancy_non_vacuity is False  # old: run marked non-scorable

    gate = evaluate_regime_occupancy_gate(
        V3_EXQ_464E_SEED42, mode_label="external_task", floor=OCCUPANCY_FLOOR)
    assert gate["reachable"] is True  # new: the mode IS reachable
    # but the shape flag says it is not a usable dose-response instrument
    # EITHER -- for the TRUE reason (saturation), not a false "not occupied".
    assert gate["regime_shape"] == "saturated_bimodal"


# ---------------------------------------------------------------------------- #
# REQUIREMENT 2 -- the existential-"graded" defect (severity `corrupting`).
# ---------------------------------------------------------------------------- #

def test_934_disjoint_per_seed_mixed_caps_are_not_graded():
    """THE corrupting case, replayed from the confirmed V3-EXQ-934 record.

    Two of three seeds have a mixed cap, so the OLD existential says "graded"
    and passes -- but the two caps are 0.75 and 1.75, disjoint singletons at
    opposite ends of the sweep. No COMMON cap works, which is what the entry's
    own target demands.
    """
    gate = evaluate_regime_occupancy_gate(
        V3_EXQ_934_ARM_SYMMETRIC, mode_label="external_task",
        floor=OCCUPANCY_FLOOR)

    assert gate["regime_shape"] == "mixed_not_reproducible"
    assert gate["graded"] is False
    assert gate["route_reason"] == "external_task_mixed_regime_not_reproducible"
    # reachable is still TRUE -- requirement 1 is not undone by requirement 2.
    assert gate["reachable"] is True
    # no cap met the >= 2/3-seed bar, so there is no reproducible band at all.
    assert gate["reproducible_conditions"] == []
    assert gate["reproducible_band"] is None
    assert gate["seeds"] == [42, 43, 44]


def test_934_old_existential_predicate_would_have_said_graded():
    """Pin the exact behavioural difference the 2026-09-11 fix makes.

    This is the assertion that would have failed before the fix, on the real
    data that produced the false `cap_recalibration_admits_mixed_regime`
    routing.
    """
    fracs = [c.fraction for c in V3_EXQ_934_ARM_SYMMETRIC]
    old_graded = any(OCCUPANCY_FLOOR < f < 0.9 for f in fracs)
    assert old_graded is True  # old gate: PASS, route_reason None

    gate = evaluate_regime_occupancy_gate(
        V3_EXQ_934_ARM_SYMMETRIC, mode_label="external_task",
        floor=OCCUPANCY_FLOOR)
    assert gate["graded"] is False
    assert gate["route_reason"] is not None


def test_934_per_seed_call_shape_is_underdetermined_not_graded():
    """The 934 CALL SHAPE (one call per seed) must not re-create the defect.

    Seed 42's own cells contain a mixed cap, so a per-seed call is 1-of-1
    seeds mixed == 100% -- which would satisfy any seed FRACTION. The
    min_seeds floor is what stops that reading.
    """
    seed42 = [c for c in V3_EXQ_934_ARM_SYMMETRIC if c.seed == 42]
    gate = evaluate_regime_occupancy_gate(
        seed42, mode_label="external_task", floor=OCCUPANCY_FLOOR)
    assert gate["regime_shape"] == "underdetermined"
    assert gate["graded"] is False
    assert gate["route_reason"] == "external_task_gradedness_underdetermined"


def test_cells_without_seed_identity_fail_closed():
    """No seed metadata at all -> underdetermined, never a silent pass.

    This is the 464e seed-44 shape, which the PRE-fix contract asserted read
    "graded" off a single mixed arm. Two cells, one mixed, no seeds: the bar
    is not evaluable, so the gate must refuse rather than fall back.
    """
    gate = evaluate_regime_occupancy_gate(
        V3_EXQ_464E_SEED44, mode_label="external_task", floor=OCCUPANCY_FLOOR)
    assert gate["regime_shape"] == "underdetermined"
    assert gate["graded"] is False
    assert gate["route_reason"] is not None
    assert gate["seeds"] == []
    # still reachable, and the mixed cell is still REPORTED -- failing closed
    # is not the same as denying what was measured.
    assert gate["reachable"] is True
    assert any(c["mixed"] for c in gate["cells"])


def test_genuinely_graded_sweep_reads_graded_with_no_route_reason():
    """A reproducible mixed band -- the only shape that passes."""
    gate = evaluate_regime_occupancy_gate(
        GENUINELY_GRADED_SWEEP, mode_label="external_task",
        floor=OCCUPANCY_FLOOR)
    assert gate["regime_shape"] == "graded"
    assert gate["graded"] is True
    assert gate["route_reason"] is None
    assert gate["adjacency_evaluated"] is True
    assert gate["reproducible_conditions"] == [0.50, 1.00]
    assert gate["reproducible_band"] == [0.50, 1.00]


def test_adjacency_is_required_not_just_two_qualifying_points():
    """Two reproducible values that are NOT adjacent must not read graded.

    This is the sweep-position half of the bar. Without it, a gate could pass
    on two isolated islands with a dead zone between them -- not a
    dose-response.
    """
    cells = []
    for s in (42, 43, 44):
        cells.append(OccupancyCell(label="r=0.10", fraction=0.5, seed=s,
                                   sweep_value=0.10))   # qualifies
        cells.append(OccupancyCell(label="r=0.50", fraction=1.0, seed=s,
                                   sweep_value=0.50))   # saturated, breaks run
        cells.append(OccupancyCell(label="r=1.00", fraction=0.5, seed=s,
                                   sweep_value=1.00))   # qualifies
    gate = evaluate_regime_occupancy_gate(
        cells, mode_label="external_task", floor=OCCUPANCY_FLOOR)
    assert gate["reproducible_conditions"] == [0.10, 1.00]  # two of them
    assert gate["regime_shape"] == "mixed_not_reproducible"  # but not adjacent
    # a one-condition run is NOT reported as a band -- see the module's
    # reproducible_band note. The raw run length stays visible.
    assert gate["reproducible_band"] is None
    assert gate["longest_adjacent_run"] == 1


def test_one_of_three_seeds_at_a_common_value_is_not_enough():
    """The seed half of the bar, isolated from the adjacency half.

    Two ADJACENT values, both mixed, but only ever on 1 of 3 seeds.
    """
    cells = []
    for value in (0.75, 1.00):
        cells.append(OccupancyCell(label=f"cap={value}", fraction=0.5, seed=42,
                                   sweep_value=value))
        cells.append(OccupancyCell(label=f"cap={value}", fraction=1.0, seed=43,
                                   sweep_value=value))
        cells.append(OccupancyCell(label=f"cap={value}", fraction=0.0, seed=44,
                                   sweep_value=value))
    gate = evaluate_regime_occupancy_gate(
        cells, mode_label="external_task", floor=OCCUPANCY_FLOOR)
    assert gate["per_value"][0]["seed_fraction"] == pytest.approx(1.0 / 3.0,
                                                                 abs=1e-4)
    assert gate["regime_shape"] == "mixed_not_reproducible"


def test_arm_contrast_with_seeds_grades_without_adjacency():
    """464e's own target has NO adjacency clause -- arms are unordered.

    With `sweep_value` unset and `min_adjacent=1`, the seed-reproducibility
    bar is the whole bar, and `adjacency_evaluated` says so.
    """
    cells = [
        OccupancyCell(label="ARM_SYMMETRIC", fraction=f, seed=s)
        for s, f in ((42, 0.43), (43, 0.38), (44, 1.0))
    ]
    gate = evaluate_regime_occupancy_gate(
        cells, mode_label="external_task", floor=OCCUPANCY_FLOOR,
        min_adjacent=1)
    assert gate["adjacency_evaluated"] is False
    assert gate["regime_shape"] == "graded"
    assert gate["route_reason"] is None


def test_arm_contrast_asking_for_adjacency_it_cannot_express_fails_closed():
    """Same unordered cells at the DEFAULT min_adjacent=2 -> underdetermined.

    The caller asked for an adjacency the data has no axis for. Refusing is
    the fail-closed behaviour; silently ignoring the request is not.
    """
    cells = [
        OccupancyCell(label="ARM_SYMMETRIC", fraction=f, seed=s)
        for s, f in ((42, 0.43), (43, 0.38), (44, 1.0))
    ]
    gate = evaluate_regime_occupancy_gate(
        cells, mode_label="external_task", floor=OCCUPANCY_FLOOR)
    assert gate["regime_shape"] == "underdetermined"
    assert gate["route_reason"] is not None


def test_min_fraction_is_reported_but_never_gates():
    """The ORIGINAL defect must not creep back via the diagnostics field.

    A sweep whose min_fraction is 0.0 -- the exact old-gate failure trigger --
    still reads graded when the bar is met.
    """
    gate = evaluate_regime_occupancy_gate(
        GENUINELY_GRADED_SWEEP, mode_label="external_task",
        floor=OCCUPANCY_FLOOR)
    assert gate["min_fraction"] == 0.0
    assert gate["regime_shape"] == "graded"
    assert gate["route_reason"] is None


def test_route_reason_overrides_are_honoured():
    gate = evaluate_regime_occupancy_gate(
        GENUINELY_UNREACHABLE, mode_label="external_task", floor=OCCUPANCY_FLOOR,
        not_ready_reason="custom_not_ready")
    assert gate["route_reason"] == "custom_not_ready"

    gate2 = evaluate_regime_occupancy_gate(
        V3_EXQ_464E_SEED42, mode_label="external_task", floor=OCCUPANCY_FLOOR,
        saturated_reason="custom_saturated")
    assert gate2["route_reason"] == "custom_saturated"

    gate3 = evaluate_regime_occupancy_gate(
        V3_EXQ_934_ARM_SYMMETRIC, mode_label="external_task",
        floor=OCCUPANCY_FLOOR, not_reproducible_reason="custom_not_reproducible")
    assert gate3["route_reason"] == "custom_not_reproducible"

    gate4 = evaluate_regime_occupancy_gate(
        V3_EXQ_464E_SEED44, mode_label="external_task", floor=OCCUPANCY_FLOOR,
        underdetermined_reason="custom_underdetermined")
    assert gate4["route_reason"] == "custom_underdetermined"


def test_bar_defaults_match_the_failure_record_targets():
    """The thresholds are transcribed from the entry, not chosen here.

    467e: ">= 2/3 seeds at >= 2 adjacent hysteresis ratios".
    """
    assert DEFAULT_MIN_SEED_FRACTION == pytest.approx(2.0 / 3.0)
    assert DEFAULT_MIN_ADJACENT == 2
    assert DEFAULT_MIN_SEEDS == 2
