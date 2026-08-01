"""
Contract: `compute_dynamic_eval_budget` (experiments/_lib/inv091_driver_defaults.py,
AMENDMENT 2, 2026-08-01) -- the per-run PILOT-based sizing that replaces a single
static run-length guess for the INV-091 driver family.

Companion to `test_inv091_null_validation_run_length.py`, which pins the STATIC
defaults (AMENDMENT 1). This file pins the DYNAMIC path added because AMENDMENT 1
was itself confirmed insufficient by V3-EXQ-828a
(failure_autopsy_V3-EXQ-828a_2026-08-01): a static margin computed from one prior
run's worst-case deficit does not reliably bound a DIFFERENT run's (or even a
different seed's) worst case -- tau_max varies substantially across seeds, and the
BINDING term in `q081_surrogate.plan_blocks`'s bound is frequently a rare-firing
stream's estimated recording PERIOD, not `safety_factor * tau_max`.

Three things are pinned here:

  1. POSITIVE (a) -- `compute_dynamic_eval_budget`, given a synthetic pilot with a
     KNOWN (measured, not mocked) autocorrelation structure, returns a step count
     that a REAL eval phase of that length genuinely clears
     `q081_surrogate.plan_blocks` at -- end to end, not just an arithmetic check.
  2. MARGIN (b) -- the returned size is not a bare pass: it exceeds the bare
     `plan_blocks`-clearing minimum by at least `margin_multiplier`.
  3. REGRESSION / NEGATIVE CONTROL (c) -- against 828a's OWN recorded worst-case
     profile (18 real (arm, seed) cells; see
     `inv091_driver_defaults.py`'s AMENDMENT 2 docstring table), the OLD static
     budget (AMENDMENT 1's 3600 eval-phase steps) falls far short of the recorded
     "Need at least 14648 steps" floor, while a pilot reproducing the SAME
     qualitative hazard (a rare-firing stream whose measured period dominates,
     alongside a moderate-tau stream) drives `compute_dynamic_eval_budget` to a
     budget that clears it. 828a's manifest stores only the `SurrogateDesignError`
     message's scalars (`lower`/`tau_max`/`n_steps`/`needed`) for each of the 18
     cells -- the check raises BEFORE any array-level detail is captured, so the
     raw per-step arrays that produced those scalars are not recoverable from the
     manifest. This test therefore reconstructs a synthetic pilot whose measured
     tau_steps_max/period_max land close to the worst recorded cell (seed 2:
     tau_max=56.0 steps, lower/period_max=1831.0, needed=14648) rather than
     replaying the exact recorded arrays -- the real recorded SCALARS anchor the
     construction and the regression assertion, per this file's own approach (c).

ASCII-only output (repo rule).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments._lib import inv091_driver_defaults as defaults
from experiments._lib import q081_surrogate as surro

# 828a's REAL recorded worst-case cell (intact, seed index 2), read verbatim from
# REE_assembly/evidence/experiments/
#   v3_exq_828a_inv091_cross_stream_similarity_band_null_validated_20260801T073417Z_v3.json
# `surrogate_p_values["intact"][2]["error"]`:
#   "... the slowest stream needs blocks of at least 1831.0 steps (safety 2.0 x
#   tau_max 56.0 steps) but 1831 steps only supports 228.9-step blocks at 8 blocks
#   minimum. Need at least 14648 steps. ..."
V3_EXQ_828A_WORST_CELL_TAU_STEPS_MAX = 56.0
V3_EXQ_828A_WORST_CELL_LOWER = 1831.0   # == period_max here: 2*56=112 << 1831
V3_EXQ_828A_WORST_CELL_SAFETY_FACTOR = 2.0
V3_EXQ_828A_WORST_CELL_MIN_BLOCKS = 8
V3_EXQ_828A_WORST_CELL_NEEDED = 14648   # ceil(lower * min_blocks)

# AMENDMENT 1's static budget (experiments/_lib/inv091_driver_defaults.py,
# INV091_EVAL_EPISODES=24 * INV091_STEPS_PER_EPISODE=150) -- confirmed
# insufficient by 828a; used below as the OLD-approach negative control.
_OLD_STATIC_BUDGET = defaults.INV091_EVAL_STEPS_TOTAL


def _sanity_check_recorded_arithmetic():
    """The recorded scalars above are internally consistent with
    q081_surrogate.plan_blocks's own formula -- confirms this test's anchors are
    read correctly before anything is built on top of them."""
    lower = max(
        V3_EXQ_828A_WORST_CELL_SAFETY_FACTOR * V3_EXQ_828A_WORST_CELL_TAU_STEPS_MAX,
        V3_EXQ_828A_WORST_CELL_LOWER,
    )
    assert lower == V3_EXQ_828A_WORST_CELL_LOWER
    needed = int(math.ceil(lower * V3_EXQ_828A_WORST_CELL_MIN_BLOCKS))
    assert needed == V3_EXQ_828A_WORST_CELL_NEEDED


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) with unit stationary variance -- a stream with KNOWN, MEASURED (not
    hand-set) autocorrelation structure. Mirrors test_q081_surrogate_null.py's
    generator; duplicated locally so this file's synthetic construction is
    self-contained and does not couple to that file's fixtures."""
    x = np.zeros(n)
    e = rng.standard_normal(n) * np.sqrt(1.0 - phi ** 2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _fresh_every_step_arrays(name: str, n: int, phi: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vals = _ar1(n, phi, rng).reshape(-1, 1).astype(np.float32)
    fresh = np.ones(n, dtype=bool)
    return {name: vals, f"{name}__fresh": fresh}


# --------------------------------------------------------------------------
# (a) + (b): a reliable pilot's estimate clears plan_blocks with margin
# --------------------------------------------------------------------------

# phi chosen so the theoretical AR(1) 1/e-crossing tau (~ -1/ln(phi)) is ~300
# steps. The MEASURED tau (via the same _acf_first_crossing estimator
# compute_dynamic_eval_budget uses) comes in well below that theoretical value
# for a near-unit-root process at finite sample size -- confirmed empirically at
# seed=0, n=1800: tau_ticks=129.0, comfortably below the lag_cap (450) so it is
# NOT saturated, and large enough (>89) that the dynamic estimate genuinely
# exceeds the static floor (INV091_MIN_EVAL_STEPS_REQUIRED=2848). This exercises
# the DYNAMIC sizing arithmetic rather than vacuously returning the floor.
_RELIABLE_PHI = math.exp(-1.0 / 300.0)
_RELIABLE_PILOT_N = 1800  # lag_cap = 450


def test_dynamic_budget_from_a_reliable_pilot_clears_plan_blocks_end_to_end():
    """(a) Pilot -> compute -> a REAL eval phase of the returned size clears
    plan_blocks, not just an arithmetic re-check."""
    pilot_arrays = _fresh_every_step_arrays("A", _RELIABLE_PILOT_N, _RELIABLE_PHI, seed=0)

    required_steps = defaults.compute_dynamic_eval_budget(pilot_arrays, ["A"])
    assert isinstance(required_steps, int)
    assert required_steps > _RELIABLE_PILOT_N, (
        "the computed budget should be a real eval-phase target, larger than the "
        "short pilot that measured it"
    )

    full_arrays = _fresh_every_step_arrays("A", required_steps, _RELIABLE_PHI, seed=1)
    plan = surro.plan_blocks(full_arrays, ["A"])  # must NOT raise
    assert plan.n_steps == required_steps


def test_dynamic_budget_exceeds_bare_plan_blocks_minimum_by_the_margin():
    """(b) Not a bare pass: the returned size clears the bare `plan_blocks`
    minimum by at least `margin_multiplier`, per this module's AMENDMENT 2
    rationale (a third zero-margin sizing attempt was explicitly rejected)."""
    pilot_arrays = _fresh_every_step_arrays("A", _RELIABLE_PILOT_N, _RELIABLE_PHI, seed=0)

    detail = defaults.compute_dynamic_eval_budget(pilot_arrays, ["A"], return_detail=True)
    assert detail["unreliable_pilot_extra_multiplier_applied"] is False, (
        "this pilot is deliberately sized to be reliable (not saturated, plenty of "
        "fresh samples) -- if this flips, the phi/pilot-length constants above need "
        "re-tuning, not the assertion below"
    )
    assert detail["floor_applied"] is False, (
        "this test wants the DYNAMIC estimate exercised, not the static floor -- "
        "see the _RELIABLE_PHI comment"
    )
    assert detail["required_eval_steps"] == pytest.approx(
        detail["bare_plan_blocks_minimum_steps"] * defaults.INV091_DYNAMIC_MARGIN_MULTIPLIER,
        rel=0.01,
    )
    assert detail["required_eval_steps"] >= (
        detail["bare_plan_blocks_minimum_steps"] * defaults.INV091_DYNAMIC_MARGIN_MULTIPLIER
    )


def test_dynamic_budget_falls_back_to_the_static_floor_when_the_dynamic_estimate_is_small():
    """A pilot with a genuinely short tau still respects INV091_MIN_EVAL_STEPS_REQUIRED
    -- the static default remains a sanity floor under the dynamic path, per the
    module docstring's "keep the existing static constants as a documented
    FALLBACK/floor" instruction."""
    short_tau_phi = math.exp(-1.0 / 5.0)  # tau ~ 5 steps
    pilot_arrays = _fresh_every_step_arrays("A", 200, short_tau_phi, seed=2)

    required_steps = defaults.compute_dynamic_eval_budget(pilot_arrays, ["A"])
    assert required_steps == defaults.INV091_MIN_EVAL_STEPS_REQUIRED


# --------------------------------------------------------------------------
# Reliability: a rare-firing / saturated stream is not trusted at face value
# --------------------------------------------------------------------------


def test_unreliable_pilot_stream_triggers_the_extra_margin():
    """A stream that fires only twice in the pilot window (simulating a rare
    commitment-style stream) is right-censored on its period reading -- exactly
    AMENDMENT 2's diagnosed failure mode, now guarded against on the pilot side.
    The extra multiplier must engage rather than silently trusting the reading."""
    n = 500
    rare_fresh = np.zeros(n, dtype=bool)
    rare_fresh[[10, 480]] = True  # 2 fresh samples: below INV091_PILOT_MIN_FRESH_SAMPLES
    rare_vals = np.zeros((n, 1), dtype=np.float32)

    pilot_arrays = {"RARE": rare_vals, "RARE__fresh": rare_fresh}
    detail = defaults.compute_dynamic_eval_budget(pilot_arrays, ["RARE"], return_detail=True)

    assert detail["per_stream"]["RARE"]["n_fresh_in_pilot"] == 2
    assert detail["per_stream"]["RARE"]["reliable"] is False
    assert detail["unreliable_pilot_extra_multiplier_applied"] is True
    assert detail["effective_multiplier"] == pytest.approx(
        defaults.INV091_DYNAMIC_MARGIN_MULTIPLIER * defaults.INV091_UNRELIABLE_PILOT_EXTRA_MULTIPLIER
    )


def test_saturated_acf_is_treated_as_unreliable_not_taken_at_face_value():
    """A stream whose ACF never crosses 1/e within the pilot's evaluated lag
    window is right-censored on TAU (the true tau could be longer) -- the
    saturated reading must not be trusted at face value.

    Forced deterministically via an explicit small `max_lag` (5) against a
    strongly-autocorrelated AR(1) stream, rather than tuned empirically against
    the default `max(4, pos.size // 4)` cap: `_acf_first_crossing`'s specific
    (unnormalized-by-overlap-count) correlation estimator crosses 1/e at a
    roughly fixed FRACTION of whatever lag window it is given regardless of the
    underlying process's true persistence (confirmed empirically while building
    this test -- linear ramps, step functions and long-period sinusoids all
    crossed around ~0.21-0.22 of the auto lag cap), so relying on the auto cap
    to reproduce a genuine under-resolution is fragile. Passing `max_lag`
    directly is also the realistic case: a driver author who knows a stream
    fires rarely enough to warrant a tighter cap would pass one explicitly."""
    long_tau_phi = math.exp(-1.0 / 400.0)
    n = 200
    pilot_arrays = _fresh_every_step_arrays("SLOW", n, long_tau_phi, seed=3)

    detail = defaults.compute_dynamic_eval_budget(
        pilot_arrays, ["SLOW"], max_lag=5, return_detail=True
    )
    assert detail["per_stream"]["SLOW"]["tau_ticks"] == 5.0
    assert detail["per_stream"]["SLOW"]["saturated"] is True
    assert detail["per_stream"]["SLOW"]["reliable"] is False
    assert detail["unreliable_pilot_extra_multiplier_applied"] is True


def test_pilot_that_measures_nothing_raises_rather_than_guessing():
    """Every named stream absent from the pilot entirely (never fresh) -- the
    pilot informed nothing, so PilotTooShortError is raised rather than silently
    returning the static floor as if it meant something. Mirrors
    q081_surrogate.SurrogateDesignError's refuse-rather-than-guess philosophy,
    explicitly called out in failure_autopsy_V3-EXQ-828a_2026-08-01 as worth
    preserving."""
    n = 100
    never_fresh = np.zeros(n, dtype=bool)
    pilot_arrays = {"GHOST": np.zeros((n, 1)), "GHOST__fresh": never_fresh}

    with pytest.raises(defaults.PilotTooShortError):
        defaults.compute_dynamic_eval_budget(pilot_arrays, ["GHOST"])


# --------------------------------------------------------------------------
# (c) Regression / negative control against 828a's real recorded worst case
# --------------------------------------------------------------------------


def test_old_static_budget_undershoots_828a_recorded_worst_case():
    """NEGATIVE CONTROL, part 1: confirms the OLD (AMENDMENT 1) static budget
    genuinely falls short of the REAL recorded 828a worst-case requirement --
    the regression this whole module exists to fix, stated as a plain fact
    check on recorded numbers (no reconstruction needed for this half)."""
    _sanity_check_recorded_arithmetic()
    assert _OLD_STATIC_BUDGET < V3_EXQ_828A_WORST_CELL_NEEDED, (
        "if this ever flips, AMENDMENT 1's static budget alone would have cleared "
        "828a's worst recorded cell -- re-examine whether the regression this test "
        "guards is still real"
    )


def test_dynamic_budget_clears_828a_recorded_worst_case_from_a_realistic_pilot():
    """NEGATIVE CONTROL, part 2: a pilot reproducing 828a's worst-cell HAZARD
    (a rare-firing stream whose period dominates the bound, per AMENDMENT 2's
    "the binding term is often period_max, not safety_factor*tau_max" finding,
    alongside a moderate-tau stream near the recorded tau_max=56.0) drives
    `compute_dynamic_eval_budget` to a size that clears the REAL recorded
    14648-step floor -- where the old static 3600-step budget did not."""
    n = 2000
    # FAST: fresh every step, tau close to the recorded worst-cell tau_max=56.0.
    fast_phi = math.exp(-1.0 / V3_EXQ_828A_WORST_CELL_TAU_STEPS_MAX)
    fast = _fresh_every_step_arrays("FAST", n, fast_phi, seed=4)

    # RARE: fires only twice, near the ends of the pilot window -- its measured
    # period (~1800) lands close to the recorded worst-cell period_max (1831.0),
    # reproducing the SAME structural hazard (a rare stream's right-censored
    # period estimate dominating the bound) rather than the exact recorded value.
    rare_fresh = np.zeros(n, dtype=bool)
    rare_fresh[[100, 1900]] = True
    rare = {"RARE": np.zeros((n, 1), dtype=np.float32), "RARE__fresh": rare_fresh}

    pilot_arrays = {**fast, **rare}
    detail = defaults.compute_dynamic_eval_budget(
        pilot_arrays, ["FAST", "RARE"], return_detail=True
    )

    # The construction reproduced the hazard: RARE dominates period_max and is
    # flagged unreliable (2 fresh samples), same as 828a's real worst cell.
    assert detail["period_max"] > 10 * V3_EXQ_828A_WORST_CELL_TAU_STEPS_MAX
    assert detail["unreliable_pilot_extra_multiplier_applied"] is True

    required_steps = detail["required_eval_steps"]
    assert required_steps > _OLD_STATIC_BUDGET, (
        "dynamic sizing must clear where the old static default fell short"
    )
    assert required_steps > V3_EXQ_828A_WORST_CELL_NEEDED, (
        "dynamic sizing from a pilot reproducing 828a's worst-cell hazard must "
        "clear the REAL recorded worst-case requirement, not just beat the old "
        "(also-insufficient) static default"
    )
