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

THE THIRD DEFECT -- IN THIS FILE'S OWN FIRST VERSION (2026-09-11)
-----------------------------------------------------------------
The `bd59f7e` version of this module decided "graded" with a flat per-cell
EXISTENTIAL over every fraction it was handed:

    if any(floor < f < ceiling for f in fracs):
        return "graded"

That predicate carries neither SEED IDENTITY nor SWEEP POSITION, so ONE mixed
cell out of N declared the whole regime genuinely graded and set
`route_reason = None` (gate PASSES). `mode-governance-engagement`'s
`severity_note` classified this `corrupting` -- evidence that LOOKS valid and is
not.

The subtlety worth stating, because it is why fixing the predicate alone would
NOT have been enough: V3-EXQ-934's driver did NOT simply trust one cell. It
called this gate once per (seed, arm) and then applied its own `>= 2/3 seeds`
rule on top of the per-seed booleans. It still routed
`cap_recalibration_admits_mixed_regime` from this (confirmed, run
`v3_exq_934_..._20260815T015216Z_v3`):

    seed 42  mixed caps {0.75}          -> per-seed "graded" = True
    seed 43  mixed caps {1.75}          -> per-seed "graded" = True
    seed 44  mixed caps {}              -> per-seed "graded" = False
                                           => 2/3 seeds "graded" => PASS

Two seeds were mixed at DISJOINT, OPPOSITE ends of the cap sweep. There is no
COMMON cap. The per-seed existential had already discarded WHICH cap was mixed,
so no amount of counting seeds afterwards could recover it -- and the
`[min_fraction, max_fraction]` band summary hid it as well. The reproducibility
question is therefore only answerable in ONE place, over ALL (seed, sweep_value)
cells at once. That is why `OccupancyCell` now carries `seed` and `sweep_value`.

THE BAR, read off `mode-governance-engagement`'s own two OPEN failure_record
targets (this module does not invent a threshold; it implements those):

  V3-EXQ-467e: "per-arm fraction_in_external_task strictly between
      OCCUPANCY_FLOOR (0.1) and 0.9 on >= 2/3 seeds at >= 2 ADJACENT hysteresis
      ratios"
  V3-EXQ-934:  "A COMMON cap value yielding per-arm occupancy in (0.1, 0.9) on
      >= 2/3 seeds ... with a mixed band at least 2 grid steps wide"

Both have the same shape, and it is the shape `classify_regime_shape` now
implements:

    GRADED iff there exists a run of >= `min_adjacent` (2) CONSECUTIVE swept
    values, each of which is mixed on >= `min_seed_fraction` (2/3) of the seeds
    measured at that value.

("2 grid steps wide" is read as 2 measured grid POINTS, matching 467e's
unambiguous ">= 2 adjacent hysteresis ratios".)

THE FIX
-------
`evaluate_regime_occupancy_gate` replaces MIN-across-the-sweep with:

  1. Per-cell reachability: cell.fraction > floor.
  2. Aggregate reachability: ANY cell reachable (not ALL / not MIN) -- mirrors
     `precondition_gate.aggregate_arm_gates`'s any-green-not-all-green fix for
     the analogous V3-EXQ-785 pattern (min-vs-any is the occupancy-statistic
     form of that same any-not-all correction).
  3. A `regime_shape` classification that separates the pathologies MIN and the
     bare existential conflate:
       "unreachable"           -- every cell at/below floor.
       "saturated_bimodal"     -- reachable, but NO cell anywhere falls in the
                                  mixed band (464e/467e's observed signature).
       "mixed_not_reproducible"-- mixed cells DO exist, but they do not survive
                                  the seed-reproducibility / adjacency bar
                                  above (934's observed signature).
       "graded"                -- the bar is met: a genuinely reproducible
                                  mixed regime exists in this sweep.
       "underdetermined"       -- mixed cells exist but the cells carry too
                                  little metadata to decide (see FAILS CLOSED).
  4. A `route_reason` that names the ACTUAL pathology, so a downstream reader
     (a human, or `build_experiment_indexes.py`'s adjudication) is not told
     "not occupied" when the true finding is "occupied but saturated", nor
     "graded" when the true finding is "mixed on two seeds at two different,
     non-overlapping sweep points".

FAILS CLOSED, NEVER OPEN
------------------------
`seed` and `sweep_value` are optional on the dataclass (so existing callers
still CONSTRUCT), but a call that cannot support the bar does NOT silently fall
back to the old existential -- that would reinstate the exact defect. It
classifies "underdetermined" with a non-None `route_reason`, i.e. the gate does
not pass. Three ways to land there:

  * no cell carries a `seed`;
  * fewer than `min_seeds` (2) distinct seeds are present -- this is the guard
    that stops the 934 call shape from re-creating the defect: a PER-SEED call
    has n_seeds == 1, and 1-of-1 would otherwise satisfy any seed FRACTION;
  * cells carry seeds but the caller asked for adjacency it cannot evaluate.

CALL SHAPE (this changed -- read it before porting a driver)
------------------------------------------------------------
Call ONCE over ALL (seed, sweep_value) cells. Do NOT call per-seed and count
booleans afterwards; that is precisely the 934 shape whose collapse of "which
cap" produced the false routing.

    from experiments._lib.regime_occupancy_gate import (
        OccupancyCell, evaluate_regime_occupancy_gate,
    )

    cells = [
        OccupancyCell(label=f"cap={cap}", fraction=frac, seed=seed,
                      sweep_value=cap)
        for seed, cap, frac in all_measured_cells
    ]
    gate = evaluate_regime_occupancy_gate(cells, mode_label="external_task",
                                          floor=OCCUPANCY_FLOOR)
    route_reason = gate["route_reason"]          # None only when GRADED
    manifest["interpretation"]["occupancy_gate"] = gate

For an ARM CONTRAST (unordered conditions -- 464e's ARM_SYMMETRIC vs
ARM_ASYM_STICKY_TASK) leave `sweep_value` unset and pass `seed`: conditions are
then grouped by `label`, the seed-reproducibility bar still applies, and
adjacency is skipped and reported as `adjacency_evaluated: False` (adjacency is
undefined without an ordered axis; 464e's own target says "on >= 2/3 seeds" with
no adjacency clause).

`min_fraction` is retained for DIAGNOSTICS ONLY and must never be used to gate
-- re-gating on it is the original defect this module was written to remove.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "OccupancyCell",
    "DEFAULT_MIN_SEED_FRACTION",
    "DEFAULT_MIN_ADJACENT",
    "DEFAULT_MIN_SEEDS",
    "classify_regime_shape",
    "evaluate_regime_occupancy_gate",
]

# The bar from `mode-governance-engagement`'s two OPEN failure_record targets
# (V3-EXQ-467e and V3-EXQ-934). See the module docstring "THE BAR" section --
# these are not tuning knobs chosen here, they are those targets transcribed.
DEFAULT_MIN_SEED_FRACTION = 2.0 / 3.0
DEFAULT_MIN_ADJACENT = 2

# Structural floor on the number of distinct seeds required before a seed
# FRACTION means anything. Without it a per-seed call (n_seeds == 1) reads
# 1-of-1 == 100% of seeds and re-creates the V3-EXQ-934 false positive.
DEFAULT_MIN_SEEDS = 2


@dataclass
class OccupancyCell:
    """One measurement point in a swept-condition or multi-arm occupancy probe.

    `label` names the condition (an arm id, or a swept-parameter value such as
    "r=0.50") so a verdict is attributable back to the specific cell.
    `fraction` is the measured occupancy fraction in [0, 1] for the contested
    mode at this cell (e.g. `fraction_in_external_task`).
    `seed` is the seed this cell was measured on. REQUIRED to decide
    gradedness -- without it the reproducibility bar cannot be evaluated and
    the gate classifies "underdetermined" (see the module docstring).
    `sweep_value` is the cell's position on the ORDERED swept axis (the cap
    value, the hysteresis ratio). Required only for the adjacency half of the
    bar; leave it unset for an unordered arm contrast.
    """

    label: str
    fraction: float
    seed: Optional[int] = None
    sweep_value: Optional[float] = None


def _condition_groups(
    cells: Sequence[OccupancyCell],
) -> Tuple[List[Any], Dict[Any, List[OccupancyCell]], bool]:
    """Group cells by swept condition.

    Returns (ordered_keys, groups, adjacency_evaluable).

    Groups by `sweep_value` -- numerically ordered, so "adjacent" means
    adjacent on the measured grid -- when EVERY cell carries one. Otherwise
    falls back to grouping by `label` in first-seen order, and reports
    adjacency as not evaluable (an unordered arm contrast has no adjacency).
    """
    all_have_sweep = bool(cells) and all(c.sweep_value is not None for c in cells)
    groups: Dict[Any, List[OccupancyCell]] = {}
    order: List[Any] = []
    for c in cells:
        key = float(c.sweep_value) if all_have_sweep else c.label
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(c)
    if all_have_sweep:
        order = sorted(order)
    return order, groups, all_have_sweep


def _mixed(fraction: float, floor: float, ceiling: float) -> bool:
    return bool(floor < float(fraction) < ceiling)


def _per_value_reproducibility(
    cells: Sequence[OccupancyCell],
    floor: float,
    ceiling: float,
    min_seed_fraction: float,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Per swept condition, how many of its seeds land in the mixed band.

    Returns (per_value_records, adjacency_evaluable). Each record carries
    `qualifies`: the seed-reproducibility half of the bar, at that one
    condition.
    """
    order, groups, adjacency_evaluable = _condition_groups(cells)
    per_value: List[Dict[str, Any]] = []
    for key in order:
        group = groups[key]
        seeds_here = sorted({int(c.seed) for c in group if c.seed is not None})
        mixed_seeds = sorted({
            int(c.seed) for c in group
            if c.seed is not None and _mixed(c.fraction, floor, ceiling)
        })
        n_seeds = len(seeds_here)
        seed_fraction = (len(mixed_seeds) / n_seeds) if n_seeds else 0.0
        per_value.append({
            "condition": key if not adjacency_evaluable else float(key),
            "label": group[0].label,
            "n_seeds": n_seeds,
            "n_mixed_seeds": len(mixed_seeds),
            "mixed_seeds": mixed_seeds,
            "seed_fraction": round(seed_fraction, 4),
            "qualifies": bool(n_seeds and seed_fraction >= min_seed_fraction),
        })
    return per_value, adjacency_evaluable


def _longest_adjacent_run(per_value: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Longest run of CONSECUTIVE qualifying conditions, in grid order."""
    best: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    for rec in per_value:
        if rec["qualifies"]:
            current.append(rec)
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    return best


def classify_regime_shape(cells: Sequence[OccupancyCell],
                          floor: float,
                          ceiling: float = 0.9,
                          min_seed_fraction: float = DEFAULT_MIN_SEED_FRACTION,
                          min_adjacent: int = DEFAULT_MIN_ADJACENT,
                          min_seeds: int = DEFAULT_MIN_SEEDS) -> str:
    """Classify the shape of the occupancy distribution across `cells`.

    Returns one of:
      "unreachable"            -- every cell's fraction <= floor. The mode never
                                  becomes occupied anywhere in the sweep
                                  (pathology (a) in the module docstring).
      "saturated_bimodal"      -- at least one cell > floor, but NO cell falls
                                  in the open band (floor, ceiling) -- the mode
                                  is reachable but every reachable cell is
                                  (near-)maximal, with no intermediate value
                                  anywhere (pathology (b); the confirmed
                                  464e/467e signature).
      "mixed_not_reproducible" -- mixed cells exist, but they fail the bar:
                                  no run of `min_adjacent` consecutive
                                  conditions is mixed on `min_seed_fraction` of
                                  its seeds (the confirmed 934 signature --
                                  seeds mixed at disjoint ends of the sweep).
      "graded"                 -- the bar is met: a reproducible mixed regime
                                  exists in this sweep.
      "underdetermined"        -- mixed cells exist but the cells cannot
                                  support the bar (no seeds, or fewer than
                                  `min_seeds` distinct seeds). This FAILS
                                  CLOSED: it is never a pass.

    Empty `cells` classifies as "unreachable" (vacuously -- there is nothing to
    be reachable).

    NOTE the deliberate absence of any per-cell existential shortcut. A single
    mixed cell is NOT sufficient for "graded" and never was the bar; treating
    it as sufficient is the `corrupting` defect this signature exists to close.
    """
    fracs = [float(c.fraction) for c in cells]
    if not fracs:
        return "unreachable"
    if all(f <= floor for f in fracs):
        return "unreachable"
    if not any(_mixed(f, floor, ceiling) for f in fracs):
        return "saturated_bimodal"

    # Mixed cells exist. Whether that amounts to a GRADED regime is a
    # reproducibility question, answerable only with seed identity.
    seeds = {int(c.seed) for c in cells if c.seed is not None}
    if not seeds or len(seeds) < int(min_seeds):
        return "underdetermined"

    per_value, adjacency_evaluable = _per_value_reproducibility(
        cells, floor=floor, ceiling=ceiling, min_seed_fraction=min_seed_fraction)
    qualifying = [rec for rec in per_value if rec["qualifies"]]
    if not qualifying:
        return "mixed_not_reproducible"

    if not adjacency_evaluable:
        # Unordered conditions (an arm contrast). The seed-reproducibility half
        # of the bar is the whole bar here -- see the module docstring.
        if int(min_adjacent) > 1:
            # The caller asked for an adjacency the cells cannot express.
            return "underdetermined"
        return "graded"

    run = _longest_adjacent_run(per_value)
    if len(run) >= int(min_adjacent):
        return "graded"
    return "mixed_not_reproducible"


def evaluate_regime_occupancy_gate(cells: Sequence[OccupancyCell],
                                   mode_label: str,
                                   floor: float,
                                   ceiling: float = 0.9,
                                   not_ready_reason: Optional[str] = None,
                                   saturated_reason: Optional[str] = None,
                                   not_reproducible_reason: Optional[str] = None,
                                   underdetermined_reason: Optional[str] = None,
                                   min_seed_fraction: float = DEFAULT_MIN_SEED_FRACTION,
                                   min_adjacent: int = DEFAULT_MIN_ADJACENT,
                                   min_seeds: int = DEFAULT_MIN_SEEDS
                                   ) -> Dict[str, Any]:
    """Regime-conditioned replacement for a `min(fractions) > floor` gate.

    `mode_label` names the contested mode (e.g. "external_task") and is used
    only to build readable default `route_reason` strings; pass the four
    `*_reason` overrides to replace them.

    Call ONCE over ALL (seed, sweep_value) cells -- see the module docstring's
    CALL SHAPE section for why a per-seed call is the shape that produced the
    V3-EXQ-934 false routing.

    Returns a dict:
      reachable          bool   -- ANY cell's fraction > floor (fixes the MIN ->
                                   ANY defect; M1/M2 in the module docstring).
      regime_shape       str    -- "unreachable" | "saturated_bimodal" |
                                   "mixed_not_reproducible" | "graded" |
                                   "underdetermined"
      route_reason       str|None -- None ONLY when the regime is genuinely
                                   graded. Every other shape names its own
                                   actual pathology.
      graded             bool   -- convenience: regime_shape == "graded".
      cells              list   -- per-cell {"label", "fraction", "seed",
                                   "sweep_value", "reachable", "mixed"}.
      per_value          list   -- per swept condition, the seed-reproducibility
                                   record: n_seeds, n_mixed_seeds, mixed_seeds,
                                   seed_fraction, qualifies.
      reproducible_conditions list -- the conditions that met the seed bar.
      reproducible_band  list|None -- [first, last] of the longest ADJACENT run
                                   of those conditions, but ONLY when that run
                                   met `min_adjacent`; None otherwise. This is
                                   the truthful replacement for the old
                                   [min, max] "band" summary, which reported the
                                   spread of raw fractions and so could look
                                   wide while no single condition was
                                   reproducible at all.
      longest_adjacent_run int  -- length of that run, reported even when it
                                   fell short of `min_adjacent` (0 for an
                                   unordered arm contrast).
      seeds              list   -- distinct seeds present.
      adjacency_evaluated bool  -- False for an unordered arm contrast.
      max_fraction       float  -- the best (most-occupied) cell's fraction.
      min_fraction       float  -- the worst cell's fraction, retained for
                                   DIAGNOSTICS ONLY -- not used to gate (that is
                                   precisely the defect this module fixes).
    """
    fracs = [float(c.fraction) for c in cells]
    per_cell = [
        {
            "label": c.label,
            "fraction": float(c.fraction),
            "seed": None if c.seed is None else int(c.seed),
            "sweep_value": None if c.sweep_value is None else float(c.sweep_value),
            "reachable": float(c.fraction) > floor,
            "mixed": _mixed(c.fraction, floor, ceiling),
        }
        for c in cells
    ]
    shape = classify_regime_shape(
        cells, floor=floor, ceiling=ceiling,
        min_seed_fraction=min_seed_fraction, min_adjacent=min_adjacent,
        min_seeds=min_seeds)
    reachable = shape != "unreachable"
    max_fraction = max(fracs) if fracs else 0.0
    min_fraction = min(fracs) if fracs else 0.0

    per_value, adjacency_evaluated = _per_value_reproducibility(
        cells, floor=floor, ceiling=ceiling, min_seed_fraction=min_seed_fraction)
    reproducible = [rec["condition"] for rec in per_value if rec["qualifies"]]
    run = _longest_adjacent_run(per_value) if adjacency_evaluated else []
    # A band is reported ONLY when it actually MET the adjacency bar. A run of
    # one qualifying condition is not a band, and reporting it as [x, x] reads
    # as success to anyone skimming the manifest -- the precise misreading this
    # module exists to prevent. The raw length stays available below.
    band = ([run[0]["condition"], run[-1]["condition"]]
            if len(run) >= int(min_adjacent) else None)

    if shape == "unreachable":
        route_reason = not_ready_reason or f"{mode_label}_mode_unreachable"
    elif shape == "saturated_bimodal":
        route_reason = saturated_reason or f"{mode_label}_mode_saturated_no_mixed_regime"
    elif shape == "mixed_not_reproducible":
        route_reason = (not_reproducible_reason
                        or f"{mode_label}_mixed_regime_not_reproducible")
    elif shape == "underdetermined":
        route_reason = (underdetermined_reason
                        or f"{mode_label}_gradedness_underdetermined")
    else:
        route_reason = None

    return {
        "reachable": reachable,
        "regime_shape": shape,
        "graded": bool(shape == "graded"),
        "route_reason": route_reason,
        "cells": per_cell,
        "per_value": per_value,
        "reproducible_conditions": reproducible,
        "reproducible_band": band,
        "longest_adjacent_run": len(run),
        "seeds": sorted({int(c.seed) for c in cells if c.seed is not None}),
        "adjacency_evaluated": bool(adjacency_evaluated),
        "max_fraction": max_fraction,
        "min_fraction": min_fraction,
        "floor": float(floor),
        "ceiling": float(ceiling),
        "min_seed_fraction": float(min_seed_fraction),
        "min_adjacent": int(min_adjacent),
        "min_seeds": int(min_seeds),
    }
