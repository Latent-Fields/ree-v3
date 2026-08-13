"""Regime-conditioned occupancy non-vacuity gates for swept-condition probes.

THE FAILURE MODE THIS CLOSES
----------------------------
A multi-condition probe (a dose-response sweep over ratio/threshold arms, or a
multi-arm contrast) builds a single non-vacuity floor check by taking the MIN of
a per-condition occupancy fraction across the whole swept range:

    occupancy_min = min(c["fraction_in_<mode>"] for c in condition_results)
    occupancy_non_vacuity = bool(occupancy_min > OCCUPANCY_FLOOR)
    route_reason = "<mode>_not_occupied" if not occupancy_non_vacuity else None

That MIN cannot distinguish two different pathologies, both of which read
`occupancy_min == 0`:

  (a) UNREACHABLE  -- the mode is occupied 0.0 everywhere in the sweep. The
      substrate genuinely never produces the contested mode.
  (b) SATURATED    -- the mode IS occupied, strongly, at SOME point(s) in the
      sweep (often 1.0), but collapses to 0.0 at other points, with no
      intermediate value anywhere. The substrate reaches the mode fine; there
      is simply no MIXED regime in the swept range.

Confirmed instance: `failure_autopsy_mech266-464e-467e-cluster_2026-08-13.md`
(V3-EXQ-464e / V3-EXQ-467e, MECH-266 / SD-032a). 19 of 21 arm/ratio cells across
both runs sat at EXACTLY 0.0 or 1.0 occupancy (dose-response step function; a
sticky arm at 1.0 occupancy with 0 switches). Both runs' `min()`-based gate read
`occupancy_min == 0.0` (or 0.333 for the two-arm contrast, still below the 0.1
floor on 1/3 seeds) and emitted `route_reason = "external_task_mode_not_occupied"`
-- a claim that is FACTUALLY FALSE: the mode was occupied at 100% on the sticky
arm of every 464e seed and at r=0.10 on every 467e seed. The gate was written for
pathology (a) and silently applied to pathology (b), routing a fixable
calibration/instrumentation problem as if it were "supply more pressure" -- the
exact direction the substrate work had ALREADY completed
(`REE_assembly/evidence/planning/substrate_queue.json` `mode-governance-engagement`
`implementation_hint_update_2026_08_13`).

THE SECOND, RELATED DEFECT THIS CLOSES (467e specifically)
------------------------------------------------------------
467e's gate additionally took the min OVER THE ENTIRE SWEPT RANGE, including the
condition the sweep was DESIGNED to drive the mode away from. When the swept
parameter (a hysteresis/exit-rail ratio) genuinely produces the predicted effect
-- occupancy falling as the rail loosens -- the min-over-range statistic is
ANTI-CORRELATED with that effect: the stronger the mechanism, the more certainly
the gate fails. A non-vacuity check must ask "is the mode reachable AT ALL in
this sweep" (satisfied by ANY cell clearing the floor), not "does every cell,
including the one the manipulation is designed to suppress, clear the floor."

THE FIX
-------
`evaluate_regime_occupancy_gate` replaces MIN-across-the-sweep with:

  1. Per-cell reachability: cell.fraction > floor.
  2. Aggregate reachability: ANY cell reachable (not ALL / not MIN) -- mirrors
     `precondition_gate.aggregate_arm_gates`'s any-green-not-all-green fix for
     the analogous V3-EXQ-785 pattern (min-vs-any is the occupancy-statistic
     form of that same any-not-all correction).
  3. `regime_shape` classification that separates the two pathologies MIN
     conflates: "unreachable" (every cell at/below floor), "saturated_bimodal"
     (reachable, but every cell sits outside the mixed band -- 464e/467e's
     observed signature), or "graded" (at least one cell genuinely mixed).
  4. A `route_reason` that names the ACTUAL pathology, so a downstream reader
     (a human, or `build_experiment_indexes.py`'s adjudication) is not told
     "not occupied" when the true finding is "occupied but saturated, no mixed
     regime in this sweep."

USAGE
-----
    from experiments._lib.regime_occupancy_gate import (
        OccupancyCell, evaluate_regime_occupancy_gate,
    )

    cells = [OccupancyCell(label=f"r={r}", fraction=frac) for r, frac in ...]
    gate = evaluate_regime_occupancy_gate(cells, mode_label="external_task",
                                           floor=OCCUPANCY_FLOOR)
    route_reason = gate["route_reason"]          # None when reachable
    manifest["interpretation"]["occupancy_gate"] = gate

Deliberately narrow scope: this module fixes the NON-VACUITY / route_reason
statistic only (M1/M2 in the cluster autopsy). It does not address M3 (a
mode-agnostic dwell statistic silently changing which mode it summarizes
across a sweep) -- that is a per-driver DV design choice, not a reusable
primitive; see the autopsy Section 3 for the fix pattern (condition the dwell
statistic on mode identity, not on "whichever mode changed").

ASCII-only in printed output (Windows cp1252 terminals).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "OccupancyCell",
    "classify_regime_shape",
    "evaluate_regime_occupancy_gate",
]


@dataclass
class OccupancyCell:
    """One measurement point in a swept-condition or multi-arm occupancy probe.

    `label` names the condition (an arm id, or a swept-parameter value such as
    "r=0.50") so a verdict is attributable back to the specific cell.
    `fraction` is the measured occupancy fraction in [0, 1] for the contested
    mode at this cell (e.g. `fraction_in_external_task`).
    """

    label: str
    fraction: float


def classify_regime_shape(cells: Sequence[OccupancyCell],
                          floor: float,
                          ceiling: float = 0.9) -> str:
    """Classify the shape of the occupancy distribution across `cells`.

    Returns one of:
      "unreachable"       -- every cell's fraction <= floor. The mode never
                             becomes occupied anywhere in the sweep (pathology
                             (a) in the module docstring).
      "saturated_bimodal" -- at least one cell > floor, but no cell falls in
                             the open band (floor, ceiling) -- the mode is
                             reachable but every reachable cell is (near-)
                             maximal, with no intermediate value anywhere
                             (pathology (b); the confirmed 464e/467e signature).
      "graded"             -- at least one cell falls strictly inside
                             (floor, ceiling) -- a genuine mixed regime exists
                             in this sweep.

    Empty `cells` classifies as "unreachable" (vacuously -- there is nothing to
    be reachable).
    """
    fracs = [float(c.fraction) for c in cells]
    if not fracs:
        return "unreachable"
    if all(f <= floor for f in fracs):
        return "unreachable"
    if any(floor < f < ceiling for f in fracs):
        return "graded"
    return "saturated_bimodal"


def evaluate_regime_occupancy_gate(cells: Sequence[OccupancyCell],
                                   mode_label: str,
                                   floor: float,
                                   ceiling: float = 0.9,
                                   not_ready_reason: Optional[str] = None,
                                   saturated_reason: Optional[str] = None
                                   ) -> Dict[str, Any]:
    """Regime-conditioned replacement for a `min(fractions) > floor` gate.

    `mode_label` names the contested mode (e.g. "external_task") and is used
    only to build readable default `route_reason` strings; pass
    `not_ready_reason` / `saturated_reason` to override them.

    Returns a dict:
      reachable       bool   -- ANY cell's fraction > floor (fixes the MIN ->
                                ANY defect; M1/M2 in the module docstring).
      regime_shape    str    -- "unreachable" | "saturated_bimodal" | "graded"
      route_reason    str|None -- non-None only when `reachable` is False, or
                                when reachable but saturated (no mixed regime
                                -- still not scorable as a genuine dose-response,
                                but for a DIFFERENT, truthful reason). None
                                when the regime is genuinely graded.
      cells           list   -- per-cell {"label", "fraction", "reachable"}.
      max_fraction    float  -- the best (most-occupied) cell's fraction.
      min_fraction    float  -- the worst (least-occupied) cell's fraction,
                                retained for diagnostics ONLY -- not used to
                                gate (that is precisely the defect this module
                                fixes).
    """
    fracs = [float(c.fraction) for c in cells]
    per_cell = [
        {"label": c.label, "fraction": float(c.fraction), "reachable": float(c.fraction) > floor}
        for c in cells
    ]
    shape = classify_regime_shape(cells, floor=floor, ceiling=ceiling)
    reachable = shape != "unreachable"
    max_fraction = max(fracs) if fracs else 0.0
    min_fraction = min(fracs) if fracs else 0.0

    if shape == "unreachable":
        route_reason = not_ready_reason or f"{mode_label}_mode_unreachable"
    elif shape == "saturated_bimodal":
        route_reason = saturated_reason or f"{mode_label}_mode_saturated_no_mixed_regime"
    else:
        route_reason = None

    return {
        "reachable": reachable,
        "regime_shape": shape,
        "route_reason": route_reason,
        "cells": per_cell,
        "max_fraction": max_fraction,
        "min_fraction": min_fraction,
        "floor": float(floor),
        "ceiling": float(ceiling),
    }
