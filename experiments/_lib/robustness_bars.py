"""Correctly-denominated robustness bars (autopsy followup 4, MECH-063 777a/779a cluster).

THE FAILURE MODE THIS CLOSES
----------------------------
A load-bearing "the effect exceeds its own noise" criterion written as

    c1_robust = (mean(vals) - pstdev(vals)) > MARGIN

subtracts a POPULATION DISPERSION from a mean and calls the result a noise test. It
is not one, and it fails in two independent ways:

1. `pstdev` DOES NOT SHRINK WITH n. Adding seeds cannot meet the bar. A criterion
   that no achievable sample size can satisfy is unreachable by construction, not
   demanding -- and nothing in the run reports it as such.
2. IT CONFLATES DISPERSION WITH NOISE. Seed-to-seed spread is initialisation and
   biological variation -- a property of the population being measured. Measurement
   noise is a property of the ESTIMATE of its mean. Only the second belongs in the
   denominator of "the effect exceeds its own noise".

Confirmed instance: `v3_exq_777a_mech063_orthogonal_control_axes_dissociation.py`
(c1_robust at :697, `_pooled_std` at :541). `mean_sin_angle` 0.5454 EXCEEDED its own
`SIN_MARGIN` 0.500 and `c1_seed_count` PASSED at 3 of 4 informative seeds, yet
`c1_robust` was the ONLY failing bar -- and the run self-routed a collinearity verdict
its own numbers contradict. See
`REE_assembly/evidence/planning/failure_autopsy_MECH-063-777a-779a-cluster_2026-07-18.md`
target 1.

MARGIN IS WHAT DECIDES HOW BADLY THIS BITES -- read the whole carrier list
-------------------------------------------------------------------------
The idiom propagated by copy across the 2x2 telemetry-probe family, but its
consequence is NOT uniform. It depends entirely on MARGIN:

  MARGIN > 0  -- COMPOUNDS into unreachability. The non-shrinking dispersion is
                 subtracted AND a positive bar must still be cleared. This is 777a
                 (MARGIN 0.5), the confirmed defect.
  MARGIN == 0 -- reduces to `mean > dispersion`, which is a CONSERVATIVE bar: it is
                 STRICTER than the SEM form it was mistaken for (pstdev >= pstdev/
                 sqrt(n) for all n >= 1). A criterion that PASSED in this form
                 cleared a HARDER test than intended. Such a PASS is not undermined
                 by this defect and must not be "corrected" as though it were.

That second row is why the fix is not a blanket rewrite. Re-denominating a bar that
a run already CLEARED moves the goalposts and loosens it retroactively.

CHOOSING A BAR
--------------
Decide what the criterion actually claims, then name it accordingly:

  "the effect exceeds its own MEASUREMENT NOISE"  -> `robust_by_sem`
      Denominator is the standard error, pstdev/sqrt(n). Tightens with n, so the
      bar is sample-size-improvable and a null is informative about the effect
      rather than about the instrument.

  "the effect exceeds CROSS-SEED DISPERSION"      -> `exceeds_cross_seed_dispersion`
      A different and defensible claim -- roughly "the typical seed shows it, not
      just the mean". Keep it if that is what you mean, but NAME it so no reader
      mistakes it for a noise test, and record that it is NOT sample-size-improvable.
      This function forces both: the name says dispersion, and it returns the fact.

PRE-REGISTER k. `robust_by_sem` takes an explicit multiplier so the strictness of the
bar is a recorded design decision rather than an accident of which form got copied.
k=1.0 is "mean exceeds margin by one SEM"; k=2.0 approximates a 95% one-sided bound.

BEFORE YOU QUEUE: ASK WHAT THE BAR COSTS
----------------------------------------
Repairing the denominator is NECESSARY BUT NOT SUFFICIENT, and this is the autopsy's
sharpest finding. Re-run 777a's own numbers through the corrected bar and it STILL
fails at n=4 (SE 0.1614, mean-SE 0.384 vs margin 0.500) and still fails at n=14. It
needs ~51 informative seeds -- at the observed 28.6% informative yield, ~177 raw seeds
and ~31 h of wall clock. The binding constraint was the informative-seed YIELD, not
the bar.

So call `seeds_required_for_sem_bar` at DESIGN time, with the effect size and
dispersion you expect. If the answer is not affordable at your informative yield, the
experiment is not fixed by fixing the criterion -- fix the yield, widen the margin on
stated grounds, or do not queue it.

ASCII-only output (CLAUDE.md).
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Sequence


def _clean(vals: Sequence[float]) -> List[float]:
    """Finite float values only. Non-finite entries are dropped, not raised on."""
    return [float(v) for v in vals if v is not None and math.isfinite(float(v))]


def sem(vals: Sequence[float]) -> float:
    """Standard error of the mean: pstdev / sqrt(n).

    This is the quantity that shrinks with n, and therefore the only one that
    belongs in a "the effect exceeds its own measurement noise" bar. Returns 0.0
    for fewer than 2 values (a single observation carries no spread estimate --
    callers must gate on n separately; see `robust_by_sem`'s `min_n`).
    """
    xs = _clean(vals)
    if len(xs) < 2:
        return 0.0
    return float(statistics.pstdev(xs)) / math.sqrt(len(xs))


def robust_by_sem(
    vals: Sequence[float],
    margin: float,
    k: float = 1.0,
    min_n: int = 3,
) -> Dict[str, object]:
    """Sample-size-improvable robustness bar: mean - k*SEM > margin.

    THE REPLACEMENT for `(mean - pstdev) > margin` wherever the stated intent is
    "the effect exceeds its own measurement noise". Because SEM shrinks as
    1/sqrt(n), this bar CAN be met by adding informative seeds -- check first that
    the number required is affordable (`seeds_required_for_sem_bar`).

    `k` is the pre-registered strictness multiplier; record it in the manifest
    alongside `margin` so the bar is reconstructable from the recorded run.

    `min_n` guards the degenerate case: with too few values the SEM estimate is
    itself noise, and a bar built on it is not meaningful. Below `min_n` the result
    is `passes: False` with `reason: "insufficient_n"` -- deliberately NOT a silent
    pass, and deliberately distinguishable from a real failure so a self-route can
    tell "not enough data" from "effect too small".
    """
    xs = _clean(vals)
    n = len(xs)
    if n < min_n:
        return {
            "passes": False,
            "reason": "insufficient_n",
            "n": n,
            "min_n": int(min_n),
            "mean": float(statistics.fmean(xs)) if xs else 0.0,
            "sem": sem(xs),
            "k": float(k),
            "margin": float(margin),
            "sample_size_improvable": True,
        }
    mean = float(statistics.fmean(xs))
    se = sem(xs)
    lower = mean - float(k) * se
    return {
        "passes": bool(lower > float(margin)),
        "reason": "ok",
        "n": n,
        "min_n": int(min_n),
        "mean": mean,
        "sem": se,
        "k": float(k),
        "margin": float(margin),
        "lower_bound": lower,
        "bar": "mean - k*SEM > margin",
        "sample_size_improvable": True,
    }


def exceeds_cross_seed_dispersion(
    vals: Sequence[float],
    margin: float = 0.0,
    min_n: int = 3,
) -> Dict[str, object]:
    """Dispersion bar: mean - pstdev(vals) > margin. NOT sample-size-improvable.

    Use this ONLY when the claim really is "the effect exceeds cross-seed
    dispersion" -- roughly, "the typical seed shows the effect, not merely the
    mean". That is a defensible and sometimes preferable claim, and at margin 0.0
    it is STRICTER than `robust_by_sem`.

    It is exposed under this name, rather than left to be open-coded, so that the
    777a confusion cannot recur silently: the name states the denominator, and the
    returned `sample_size_improvable: False` states the consequence. Propagate that
    flag into the manifest criterion block. A reader who sees this bar fail must not
    conclude "add seeds" -- adding seeds does not move it.

    If `margin` > 0 this compounds a non-shrinking denominator with a positive
    threshold, which is exactly the 777a defect. `margin_compounds_unreachability`
    flags that case; prefer `robust_by_sem` there.
    """
    xs = _clean(vals)
    n = len(xs)
    if n < min_n:
        return {
            "passes": False,
            "reason": "insufficient_n",
            "n": n,
            "min_n": int(min_n),
            "sample_size_improvable": False,
        }
    mean = float(statistics.fmean(xs))
    sd = float(statistics.pstdev(xs))
    return {
        "passes": bool(mean - sd > float(margin)),
        "reason": "ok",
        "n": n,
        "min_n": int(min_n),
        "mean": mean,
        "pstdev": sd,
        "margin": float(margin),
        "lower_bound": mean - sd,
        "bar": "mean - pstdev > margin (cross-seed dispersion, NOT measurement noise)",
        "sample_size_improvable": False,
        "margin_compounds_unreachability": bool(float(margin) > 0.0),
    }


def seeds_required_for_sem_bar(
    mean: float,
    pstdev_est: float,
    margin: float,
    k: float = 1.0,
    informative_yield: float = 1.0,
    max_n: int = 10000,
) -> Dict[str, object]:
    """Design-time cost of a `robust_by_sem` bar: how many seeds would it take?

    Solves `mean - k * pstdev_est / sqrt(n) > margin` for the smallest integer n,
    then divides by `informative_yield` to get the RAW seeds implied. Call this
    BEFORE queueing, with the effect size and dispersion you expect (or that a
    predecessor run recorded).

    This is the check that would have caught 777a at design time. Feeding it that
    run's own numbers (mean 0.5454, pstdev 0.3228, margin 0.500, yield 0.286)
    returns ~51 informative and ~177 raw seeds -- an answer that reframes the
    experiment as yield-limited rather than criterion-limited, before 31 h of
    wall clock is spent discovering it.

    `feasible: False` means no n up to `max_n` satisfies the bar, i.e. the effect
    does not clear the margin even with the noise term driven to zero. That is a
    margin or design problem and no amount of sampling fixes it.
    """
    gap = float(mean) - float(margin)
    if gap <= 0.0:
        return {
            "feasible": False,
            "reason": "mean_does_not_exceed_margin",
            "gap": gap,
            "note": "SEM -> 0 as n -> inf, so the bar is unreachable at any n.",
        }
    if float(pstdev_est) <= 0.0:
        n_informative = 2
    else:
        n_exact = (float(k) * float(pstdev_est) / gap) ** 2
        n_informative = max(2, int(math.ceil(n_exact)))
    if n_informative > int(max_n):
        return {
            "feasible": False,
            "reason": "exceeds_max_n",
            "n_informative_required": n_informative,
            "max_n": int(max_n),
        }
    y = float(informative_yield)
    raw = int(math.ceil(n_informative / y)) if 0.0 < y <= 1.0 else n_informative
    return {
        "feasible": True,
        "reason": "ok",
        "n_informative_required": n_informative,
        "informative_yield": y,
        "raw_seeds_implied": raw,
        "gap": gap,
        "k": float(k),
    }
