"""Contracts for the per-arm readout-headroom diagnostic (779b autopsy section 7).

The diagnostic exists because V3-EXQ-779b's `baseline_entropy_headroom` precondition
certified the BASELINE had room to move and never looked at the treatment arms, which
on seed 23 sat within 0.12 of the ceiling. Invariants asserted here:

  (1) FIDELITY: replaying 779b's real seed-23 numbers reproduces the autopsy's finding
      -- baseline healthy, tonic-ON arms flagged as saturating.
  (2) NON-GATING: the module exposes no verdict, no exception, and no boolean anyone
      could mistake for a precondition result. This is the whole design decision
      (user gate 2026-07-19) and a later "just make it fail the run" edit must break
      a test, not sail through review.
  (3) WORST-CASE, not mean: an arm is summarised by its tightest cell, and the
      offending seed is named -- the same recomputability discipline branch (d) of
      the precondition lint steers preconditions toward.
  (4) ROBUSTNESS: a malformed row is skipped, never raised on -- a diagnostic must
      not be the thing that kills a multi-hour run -- and the skipping stays visible
      via n_rows_scored.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._lib.entropy_headroom import (  # noqa: E402
    DEFAULT_WARN_MARGIN,
    headroom_margin,
    per_arm_headroom,
)

E_SAT_LOW, E_SAT_HIGH = 0.02, 0.98

# The real V3-EXQ-779b seed-23 readouts (autopsy section 7).
_779B_SEED23 = [
    {"arm": "T0P0", "seed": 23, "S_sustained_entropy": 0.6093},
    {"arm": "T1P0", "seed": 23, "S_sustained_entropy": 0.8489},
    {"arm": "T1P1", "seed": 23, "S_sustained_entropy": 0.8587},
]


def _hr(rows, **kw):
    return per_arm_headroom(rows, value_key="S_sustained_entropy",
                            low=E_SAT_LOW, high=E_SAT_HIGH, **kw)


# ---- (1) fidelity to the incident -----------------------------------------

def test_reproduces_779b_seed23_finding():
    out = _hr(_779B_SEED23)
    # The baseline is the arm R5 checked, and it really was healthy -- R5 was not
    # wrong, it was NARROW. The diagnostic must agree with it, or it is measuring
    # something else.
    assert out["per_arm"]["T0P0"]["within_band"] is True
    assert out["per_arm"]["T0P0"]["worst_margin"] > DEFAULT_WARN_MARGIN
    # Both tonic-ON arms are the unguarded exposure the autopsy identified.
    assert out["saturating_arms"] == ["T1P0", "T1P1"]
    assert out["worst_arm"] == "T1P1"
    assert math.isclose(out["worst_margin"], 0.98 - 0.8587, rel_tol=1e-9)
    # Still INSIDE the band -- this was tightness, not a violation. If the
    # diagnostic reported it as out-of-band it would be overstating the incident.
    assert out["per_arm"]["T1P1"]["within_band"] is True


# ---- (2) non-gating ---------------------------------------------------------

def test_exposes_no_verdict_key():
    """A saturating arm must not produce anything shaped like a precondition result.
    Gating here would have FAILed 779b for a bias that ran CONSERVATIVE for its own
    null -- see the module docstring."""
    out = _hr(_779B_SEED23)
    assert out["saturating_arms"], "precondition for this test"
    forbidden = {"met", "passed", "ok", "readiness_met", "verdict", "outcome",
                 "self_route", "label", "degeneracy_reason"}
    assert not (forbidden & set(out)), sorted(forbidden & set(out))


def test_fully_saturated_arm_still_returns_normally():
    """Even a genuine out-of-band arm only reports; it does not raise."""
    rows = _779B_SEED23 + [{"arm": "T9", "seed": 23, "S_sustained_entropy": 0.995}]
    out = _hr(rows)
    assert out["per_arm"]["T9"]["within_band"] is False
    assert out["per_arm"]["T9"]["worst_margin"] < 0.0, "violation must stay SIGNED"
    assert out["worst_arm"] == "T9"


# ---- (3) worst-case summarisation ------------------------------------------

def test_arm_summarised_by_tightest_cell_and_names_the_seed():
    rows = [
        {"arm": "T1P1", "seed": 11, "S_sustained_entropy": 0.50},
        {"arm": "T1P1", "seed": 23, "S_sustained_entropy": 0.9587},
        {"arm": "T1P1", "seed": 29, "S_sustained_entropy": 0.55},
    ]
    out = _hr(rows)
    e = out["per_arm"]["T1P1"]
    assert e["n_cells"] == 3
    # A mean over these three (0.670) sits comfortably mid-band and would hide seed 23
    # entirely -- exactly the mean-vs-worst-case masking branch (d) of the lint exists
    # to catch. The worst cell, and its seed, must survive the summary.
    assert e["worst_seed"] == 23
    assert math.isclose(e["worst_value"], 0.9587, rel_tol=1e-9)
    assert out["saturating_arms"] == ["T1P1"]


def test_margin_is_distance_to_nearer_bound():
    assert math.isclose(headroom_margin(0.50, 0.02, 0.98), 0.48)
    assert math.isclose(headroom_margin(0.95, 0.02, 0.98), 0.03)   # near ceiling
    assert math.isclose(headroom_margin(0.05, 0.02, 0.98), 0.03)   # near floor
    assert headroom_margin(0.99, 0.02, 0.98) < 0.0                 # above ceiling


def test_baseline_arm_is_reported_not_dropped():
    """The number is only interpretable against the baseline's own headroom, so the
    reference arm must stay in the report."""
    assert "T0P0" in _hr(_779B_SEED23)["per_arm"]


# ---- (4) robustness ---------------------------------------------------------

def test_malformed_rows_are_skipped_and_the_skipping_is_visible():
    rows = _779B_SEED23 + [
        {"arm": "T2", "seed": 1},                                    # key missing
        {"arm": "T2", "seed": 2, "S_sustained_entropy": None},       # non-numeric
        {"arm": "T2", "seed": 3, "S_sustained_entropy": float("nan")},
        {"arm": "T2", "seed": 4, "S_sustained_entropy": True},       # bool is not a value
        "not a row",
    ]
    out = _hr(rows)
    assert out["n_rows_scored"] == 3, "only the three real cells score"
    assert "T2" not in out["per_arm"]


def test_empty_rows_returns_empty_report_not_error():
    out = _hr([])
    assert out["per_arm"] == {}
    assert out["worst_arm"] is None and out["worst_margin"] is None
    assert out["saturating_arms"] == []
    assert out["n_rows_scored"] == 0


def test_report_is_json_serialisable():
    """It goes straight into a manifest, so anything unserialisable is a live bug."""
    import json
    json.dumps(_hr(_779B_SEED23))
