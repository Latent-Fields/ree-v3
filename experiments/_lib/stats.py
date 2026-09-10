"""Canonical statistics helpers for experiment DVs (no scipy dependency).

This module exists to end a copy-paste defect. Prior to SD-081 there was NO
shared rank-correlation helper: 18 experiment scripts each carried their own
`_spearman*` copy, and the copies shared a latent degeneracy bug -- they
guarded on the variance of the RANK vector after a double ``argsort`` instead
of on the input vector:

    ra = np.argsort(np.argsort(a))
    if np.std(ra) == 0.0:   # NEVER True -- see below
        return None

Double-``argsort`` of a *constant* input returns a permutation of ``0..K-1``
whose std is large (9.23 at K=32), not 0. So a constant input sails past the
guard and Spearman is computed against an arbitrary stable-sort tie-break
ordering -- deterministic noise, not a measurement (confirmed magnitudes up to
|0.74| on genuinely constant vectors; failure autopsy
``failure_autopsy_sd081-spearman-degenerate-dv_2026-07-27``).

The correct guard is on the INPUT vector. Average-ranking of ties fixes the
same class of bug structurally (a constant input -> all-equal ranks ->
genuine 0 rank-variance), which is why the corpus's tie-averaged helpers were
already safe. This canonical helper does both: it guards on the input vector
AND average-ranks ties.

Keep this module scipy-free (matches the corpus convention) and ASCII-only in
any printed output (project rule; there is no printed output here today).
"""

import math
from typing import NamedTuple, Optional, Sequence

import numpy as np


def _average_ranks(arr: np.ndarray) -> np.ndarray:
    """Rank ``arr`` ascending with ties assigned the AVERAGE of their ranks.

    Ranks are 1-based. Tied values receive the mean of the ordinal ranks they
    would otherwise occupy (the standard "fractional"/"midrank" convention used
    by Spearman's rho). This makes a constant input map to all-equal ranks,
    whose variance is genuinely 0 -- the structural half of the degeneracy fix.
    """
    n = arr.size
    order = np.argsort(arr, kind="mergesort")  # stable, deterministic ties
    sorted_arr = arr[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        # 0-based positions i..j -> average of 1-based ranks (i+1)..(j+1)
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation, computed as Pearson over average-ranks.

    Returns ``None`` -- meaning "undefined / degenerate, exclude this sample" --
    when either input vector is constant (``np.std == 0`` or fewer than two
    distinct values) or too short (< 2), or when the two vectors differ in
    length. It NEVER returns a spurious correlation on a constant input.

    The guard is deliberately on the INPUT vectors, not their ranks: the
    variance of the ranks of a constant vector is maximal, not zero (see module
    docstring). Callers that need a float DV should map ``None`` to their own
    degenerate sentinel explicitly, rather than silently treating a constant
    input as "zero correlation".
    """
    n = len(a)
    if n < 2 or len(b) != n:
        return None
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    # Input-vector degeneracy guard (the fix). Both forms are kept: len(set)<2
    # is the exact "constant" test; std==0.0 also fails closed on all-equal
    # floating input. A constant vector past this point is impossible.
    if (
        float(np.std(aa)) == 0.0
        or float(np.std(bb)) == 0.0
        or len(set(aa.tolist())) < 2
        or len(set(bb.tolist())) < 2
    ):
        return None
    ra = _average_ranks(aa)
    rb = _average_ranks(bb)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# tost_equivalence -- two one-sided tests (TOST), no scipy dependency.
# ---------------------------------------------------------------------------
#
# Added for hippocampal_campaign_assay_specifications_20260910.md section 1.6:
# "Equivalence, where the design requires sameness rather than difference ...
# is asserted by a TWO ONE-SIDED TESTS procedure at a pre-declared band, never
# by a non-significant difference." No equivalence test existed anywhere in
# `experiments/` before this (verified by search, section 0.2 of that spec).
#
# `tost_equivalence(a, b, band)` treats `a` and `b` as a PAIRED sample (the
# per-seed delta convention this codebase already uses throughout -- see e.g.
# the spec's own "mean(Delta) >= 2 x SD(Delta) across seeds" language). The
# paired difference d_i = a_i - b_i is tested against the two one-sided null
# hypotheses H01: mean(d) <= -band and H02: mean(d) >= +band; equivalence is
# declared only when BOTH nulls are rejected at the caller's alpha, i.e. the
# usual TOST decision rule (Schuirmann 1987). This is NOT a non-significant
# two-sided difference test -- see the module docstring above and the SD-081
# `spearman` degeneracy fix for the same "guard on the right thing" lesson.
#
# No scipy: the one-sided p-values need the Student-t CDF, computed here via
# the regularised incomplete beta function (Numerical Recipes' continued-
# fraction form). Kept private (leading underscore) -- callers only need
# `tost_equivalence`.


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-12) -> float:
    """Continued-fraction evaluation of the incomplete beta function, used only
    inside its convergence region (x < (a+1)/(a+b+2)); see `_betainc`."""
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b), for a, b > 0 and x in [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: int) -> float:
    """CDF of the Student-t distribution with `df` degrees of freedom (df >= 1)."""
    if df < 1:
        raise ValueError("df must be >= 1, got %r" % (df,))
    x = df / (df + t * t)
    ib = _betainc(df / 2.0, 0.5, x)
    if t >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


class TostResult(NamedTuple):
    """Result of a paired two-one-sided-tests equivalence check.

    `equivalent` is True iff BOTH one-sided nulls are rejected at `alpha`
    (the standard TOST decision). `p_lower`/`p_upper` are the two one-sided
    p-values (testing mean(delta) > -band and mean(delta) < +band
    respectively) -- report both, never only the decision, per the spec's
    "returning the decision and both one-sided p-values" contract.
    """
    equivalent: bool
    p_lower: float
    p_upper: float
    mean_diff: float
    band: float
    alpha: float
    n: int
    df: int


def tost_equivalence(a: Sequence[float], b: Sequence[float], band: float,
                      alpha: float = 0.05) -> Optional[TostResult]:
    """Paired TOST equivalence test: is mean(a - b) inside (-band, +band)?

    `a` and `b` must be the same length (a paired per-seed/per-unit sample,
    matching every other paired-delta convention in this codebase). `band`
    must be > 0. Returns `None` -- meaning "undefined / degenerate, exclude
    this sample" -- when there are fewer than 2 pairs or the lengths differ,
    mirroring `spearman`'s degenerate-input convention above rather than
    raising.

    Degenerate-variance handling: when the paired differences have exactly
    zero sample standard deviation (every pair agrees exactly), the usual
    t-statistic is 0/0. In that case the mean difference alone decides:
    strictly inside the band -> both one-sided p-values are 0.0 (maximal
    evidence for equivalence); on or outside the band -> 1.0 (no evidence).
    This is the honest limit of the t-test as SD -> 0, not a special case
    invented for convenience.
    """
    n = len(a)
    if n < 2 or len(b) != n:
        return None
    if band <= 0.0:
        raise ValueError("band must be > 0, got %r" % (band,))
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    mean_d = float(diff.mean())
    sd_d = float(diff.std(ddof=1))
    df = n - 1

    if sd_d == 0.0:
        inside = (-band < mean_d < band)
        p_lower = 0.0 if inside else 1.0
        p_upper = 0.0 if inside else 1.0
    else:
        se = sd_d / math.sqrt(n)
        # H01: mean(d) <= -band  vs  H11: mean(d) > -band  (upper-tail p-value)
        t_lower = (mean_d + band) / se
        p_lower = 1.0 - _t_cdf(t_lower, df)
        # H02: mean(d) >= +band  vs  H12: mean(d) < +band  (lower-tail p-value)
        t_upper = (mean_d - band) / se
        p_upper = _t_cdf(t_upper, df)

    equivalent = (p_lower < alpha) and (p_upper < alpha)
    return TostResult(equivalent=equivalent, p_lower=p_lower, p_upper=p_upper,
                       mean_diff=mean_d, band=float(band), alpha=float(alpha),
                       n=n, df=df)
