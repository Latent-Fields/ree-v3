"""Per-arm readout-headroom diagnostic (V3-EXQ-779b autopsy section 7, Learning 1).

THE GAP THIS CLOSES
-------------------
A saturation precondition of the form

    r5_headroom = bool(baseline_rows) and all(
        E_SAT_LOW < r["S_sustained_entropy"] < E_SAT_HIGH for r in baseline_rows
    )

guarantees that the BASELINE has room to move. It says nothing whatsoever about the
TREATMENT arms. The manipulation is precisely the thing that pushes the readout toward
a bound, so scoping the band to the baseline partition checks the one arm least likely
to saturate and leaves the arms that carry the effect entirely unguarded.

Worked case -- V3-EXQ-779b, MECH-063 sub-claim (ii). Seed 23's T0P0 baseline sat at
0.6093, comfortably inside the 0.02 < S < 0.98 band, so `baseline_entropy_headroom`
reported met=True. That same seed's tonic-ON arms reached 0.8489 (T1P0) and 0.8587
(T1P1) -- within 0.12 of E_SAT_HIGH. The tonic effect for seed 23 was measured against
a near-ceiling readout with no guard of any kind, and nothing in the manifest recorded
that fact. It surfaced only because a human read the per-arm numbers during the autopsy.

WHY THIS IS A DIAGNOSTIC AND NOT A PRECONDITION
-----------------------------------------------
This module deliberately does NOT gate. That is a decision (user gate, 2026-07-19),
not an oversight, and the reasoning is worth keeping because the opposite choice is the
intuitive one:

  * Ceiling compression biases a difference-of-arms readout TOWARD ZERO. For 779b that
    is CONSERVATIVE -- it can only understate dS_tonic, so it could not have manufactured
    779b's null. A gate would have thrown away a valid result to protect against a bias
    running the safe way.
  * But the same compression is ANTI-conservative for a run whose adjudication *rests*
    on a null ("no effect" is exactly what a saturated readout reports). Whether
    saturation threatens a conclusion therefore depends on which way that run's
    adjudication cuts -- which a precondition, evaluated before adjudication, cannot know.
  * Folding it into the existing R5 (option (a)) would additionally mislabel: a
    saturating treatment arm would self-route `substrate_not_ready`, when the substrate
    was ready and the MANIPULATION was too strong for the readout's dynamic range. That
    is the same label-imprecision V3-EXQ-779 was criticised for.

So: always measure, always emit, never gate. The autopsy reader gets the number without
the adjudicator getting a veto it cannot exercise correctly.

USAGE
-----
    from experiments._lib.entropy_headroom import per_arm_headroom

    rows = [...]  # one dict per (arm, seed) cell
    headroom = per_arm_headroom(
        rows,
        value_key="S_sustained_entropy",
        low=E_SAT_LOW,
        high=E_SAT_HIGH,
    )
    manifest["diagnostics"]["entropy_headroom_per_arm"] = headroom

Emit it unconditionally -- on PASS runs too. A diagnostic that only appears when
something already looks wrong cannot establish that anything was ever right, and the
779b exposure was invisible precisely because nothing was emitting it.

The returned `margin` is the DISTANCE TO THE NEARER BOUND in readout units: large is
healthy, 0.0 means the cell is sitting on a bound, negative means it is outside the band
entirely. `saturating_arms` applies `warn_margin` purely as a reporting convenience --
it changes no verdict anywhere in this module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = ["per_arm_headroom", "headroom_margin"]

# Reporting-only threshold. An arm whose worst cell sits within this distance of a bound
# is listed in `saturating_arms` so an autopsy reader is pointed at it. 0.15 is set from
# the 779b observation (worst margin 0.121 on T1P1/seed23) -- close enough to the ceiling
# to matter, loose enough that a mid-band arm never appears. It gates nothing.
DEFAULT_WARN_MARGIN = 0.15


def headroom_margin(value: float, low: float, high: float) -> float:
    """Distance from `value` to the NEARER of the two bounds.

    Positive = inside the band with that much room; 0.0 = exactly on a bound; negative =
    outside the band by that much. Deliberately signed so that a violation is not
    silently clipped to zero and thereby made to look merely tight.
    """
    return min(value - low, high - value)


def per_arm_headroom(
    rows: Sequence[Dict[str, Any]],
    *,
    value_key: str,
    low: float,
    high: float,
    arm_key: str = "arm",
    seed_key: str = "seed",
    warn_margin: float = DEFAULT_WARN_MARGIN,
) -> Dict[str, Any]:
    """Per-arm readout headroom against a two-sided band. Pure reporting, never gates.

    Every arm present in `rows` is reported -- including the baseline. The point of the
    diagnostic is the COMPARISON between the baseline's headroom and the treatment arms',
    so dropping the baseline would remove the reference the number is read against.

    Returns a JSON-serialisable dict:

        {
          "value_key": ..., "band_low": ..., "band_high": ...,
          "warn_margin": ...,                      # reporting threshold, gates nothing
          "per_arm": {
             "<arm>": {"n_cells": int, "min": float, "max": float,
                       "worst_margin": float, "worst_seed": <seed>,
                       "worst_value": float, "within_band": bool},
             ...
          },
          "worst_arm": <arm or None>,              # smallest worst_margin across arms
          "worst_margin": float or None,
          "saturating_arms": [<arm>, ...],         # worst_margin < warn_margin
          "n_rows_scored": int,
        }

    Rows missing `value_key`, or carrying a non-numeric / non-finite value there, are
    skipped rather than raising: this is a diagnostic and must never be the thing that
    crashes a multi-hour run. `n_rows_scored` vs `len(rows)` is what exposes any such
    skipping, so a silently-empty diagnostic is still visible in the manifest.
    """
    per_arm: Dict[str, Dict[str, Any]] = {}
    n_scored = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get(value_key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        val = float(raw)
        if val != val or val in (float("inf"), float("-inf")):  # NaN / inf
            continue
        arm = str(row.get(arm_key, "unknown"))
        margin = headroom_margin(val, low, high)
        n_scored += 1

        entry = per_arm.get(arm)
        if entry is None:
            per_arm[arm] = {
                "n_cells": 1,
                "min": val,
                "max": val,
                "worst_margin": margin,
                "worst_seed": row.get(seed_key),
                "worst_value": val,
                "within_band": margin > 0.0,
            }
            continue
        entry["n_cells"] += 1
        entry["min"] = min(entry["min"], val)
        entry["max"] = max(entry["max"], val)
        if margin < entry["worst_margin"]:
            entry["worst_margin"] = margin
            entry["worst_seed"] = row.get(seed_key)
            entry["worst_value"] = val
        entry["within_band"] = entry["within_band"] and margin > 0.0

    worst_arm: Optional[str] = None
    worst_margin: Optional[float] = None
    saturating: List[str] = []
    for arm, entry in per_arm.items():
        m = entry["worst_margin"]
        if worst_margin is None or m < worst_margin:
            worst_margin, worst_arm = m, arm
        if m < warn_margin:
            saturating.append(arm)

    return {
        "value_key": value_key,
        "band_low": low,
        "band_high": high,
        "warn_margin": warn_margin,
        "per_arm": per_arm,
        "worst_arm": worst_arm,
        "worst_margin": worst_margin,
        "saturating_arms": sorted(saturating),
        "n_rows_scored": n_scored,
    }
