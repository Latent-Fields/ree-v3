"""Contracts for experiments/_lib/regime_occupancy_gate.py.

The governing requirement, from
`failure_autopsy_mech266-464e-467e-cluster_2026-08-13.md` (sections 1-2):

    A `min(fractions) > floor` non-vacuity gate cannot distinguish "the mode is
    never occupied" from "the mode is occupied but saturated (0/1 step
    function, no mixed regime in the sweep)". Both read `min == 0` (or below
    floor), and the old gate emits the SAME route_reason
    ("external_task_mode_not_occupied") for both -- factually false in the
    saturated case, which routes substrate work in the wrong direction.

These tests replay the CONFIRMED numbers from both targets in that cluster
autopsy and assert the new gate reports the true regime shape, not "not
occupied".
"""

from experiments._lib.regime_occupancy_gate import (
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

# A genuinely graded sweep (what a working dose-response instrument should
# look like -- no cell at the extremes, real intermediate values).
GENUINELY_GRADED = [
    OccupancyCell(label="r=0.10", fraction=0.82),
    OccupancyCell(label="r=0.50", fraction=0.61),
    OccupancyCell(label="r=1.00", fraction=0.44),
    OccupancyCell(label="r=1.50", fraction=0.29),
    OccupancyCell(label="r=2.00", fraction=0.15),
]


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


def test_464e_seed44_mixed_symmetric_arm_reads_graded():
    gate = evaluate_regime_occupancy_gate(
        V3_EXQ_464E_SEED44, mode_label="external_task", floor=OCCUPANCY_FLOOR)
    assert gate["reachable"] is True
    assert gate["regime_shape"] == "graded"
    assert gate["route_reason"] is None


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


def test_genuinely_graded_sweep_reads_graded_with_no_route_reason():
    gate = evaluate_regime_occupancy_gate(
        GENUINELY_GRADED, mode_label="external_task", floor=OCCUPANCY_FLOOR)
    assert gate["reachable"] is True
    assert gate["regime_shape"] == "graded"
    assert gate["route_reason"] is None


def test_classify_regime_shape_empty_cells_is_unreachable():
    assert classify_regime_shape([], floor=OCCUPANCY_FLOOR) == "unreachable"


def test_route_reason_overrides_are_honoured():
    gate = evaluate_regime_occupancy_gate(
        GENUINELY_UNREACHABLE, mode_label="external_task", floor=OCCUPANCY_FLOOR,
        not_ready_reason="custom_not_ready")
    assert gate["route_reason"] == "custom_not_ready"

    gate2 = evaluate_regime_occupancy_gate(
        V3_EXQ_464E_SEED42, mode_label="external_task", floor=OCCUPANCY_FLOOR,
        saturated_reason="custom_saturated")
    assert gate2["route_reason"] == "custom_saturated"


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
