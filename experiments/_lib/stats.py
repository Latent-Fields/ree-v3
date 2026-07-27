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

from typing import List, Optional, Sequence

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
