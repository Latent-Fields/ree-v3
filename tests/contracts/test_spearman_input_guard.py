"""Contract for the canonical Spearman helper's INPUT-vector degeneracy guard.

Pins the fix for the SD-081 degenerate-DV defect (failure autopsy
``failure_autopsy_sd081-spearman-degenerate-dv_2026-07-27``): a constant input
vector must yield ``None`` ("undefined, exclude this sample"), NEVER a spurious
correlation produced by ranking an all-tied vector against a stable-sort
tie-break ordering.

The regression this locks down is specific: the historical helpers guarded on
``np.std(np.argsort(np.argsort(a))) == 0``, which is NEVER true for a constant
input (double-argsort of a constant returns a permutation of 0..K-1, std ~9.23
at K=32). This test feeds exactly that adversarial case and asserts ``None``.

ASCII-only. Run: pytest tests/contracts/test_spearman_input_guard.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

# conftest puts ree-v3 root on sys.path -> `experiments._lib.*` importable.
from experiments._lib.stats import spearman, _average_ranks


# ---------------------------------------------------------------------------
# The load-bearing guarantee: a constant input returns None, not a phantom rho.
# ---------------------------------------------------------------------------

def test_constant_first_arg_returns_none():
    # The exact defect shape: a constant vector at K=32, where the OLD
    # rank-variance guard (std of double-argsort) measured ~9.23 and never fired.
    b = list(np.random.RandomState(0).randn(32))
    assert spearman([5.0] * 32, b) is None


def test_constant_second_arg_returns_none():
    a = list(np.random.RandomState(1).randn(32))
    assert spearman(a, [3.0] * 32) is None


def test_both_constant_returns_none():
    # 071d's latent false-PASS path: both-constant previously drove the d^2
    # formula to exactly 1.0 (a passing rank_corr). Must now be None.
    assert spearman([2.0] * 10, [9.0] * 10) is None


def test_all_zero_indicator_returns_none():
    # 543 family: a zero-reef arm has an all-zero occupancy indicator. This is
    # the concrete constant vector that emitted spurious |rho| up to 0.74.
    drives = list(np.sin(np.linspace(0, 6, 40)))  # temporally autocorrelated
    in_reef = [0.0] * 40
    assert spearman(drives, in_reef) is None


# ---------------------------------------------------------------------------
# Normal-correlation sanity: non-degenerate input behaves like Spearman.
# ---------------------------------------------------------------------------

def test_perfect_monotonic_increasing_is_one():
    x = list(np.arange(20.0))
    y = [v * 3.0 + 2.0 for v in x]  # strictly increasing, non-linear scale ok
    assert spearman(x, y) == pytest.approx(1.0)


def test_perfect_monotonic_decreasing_is_minus_one():
    x = list(np.arange(20.0))
    y = list(reversed([v * 3.0 + 2.0 for v in x]))
    assert spearman(x, y) == pytest.approx(-1.0)


def test_matches_d2_formula_on_distinct_valued_input():
    # With no ties, average-ranking == ordinal-ranking, so the canonical helper
    # must reproduce the classic 1 - 6*sum(d^2)/(n(n^2-1)) formula exactly. This
    # is why migrating a past run with distinct-valued inputs (e.g. 071d) does
    # NOT change its computed DV.
    rng = np.random.RandomState(7)
    xs = list(rng.randn(50))
    ys = list(rng.randn(50))
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    n = 50.0
    d2 = float(((rx - ry) ** 2).sum())
    d2_formula = 1.0 - 6.0 * d2 / (n * (n * n - 1.0))
    assert spearman(xs, ys) == pytest.approx(d2_formula, abs=1e-9)


# ---------------------------------------------------------------------------
# Tie-heavy but non-degenerate: average-ranking is applied and finite.
# ---------------------------------------------------------------------------

def test_tie_heavy_non_degenerate_returns_finite_value():
    a = [1, 1, 2, 2, 3, 3, 4, 4]
    b = [1, 2, 2, 3, 3, 4, 4, 5]
    rho = spearman(a, b)
    assert rho is not None
    assert -1.0 <= rho <= 1.0
    # Positive, near-1 monotonic association through the ties.
    assert rho > 0.8


def test_ties_use_average_ranks_not_ordinal():
    # A block of identical values must all receive the SAME (averaged) rank, not
    # be split by stable-sort order. Ordinal (argsort-argsort) ranking would give
    # [0,1,2,3,4,5]; average-ranking gives [2,2,2,5,5,5] (1-based midranks). This
    # is the structural half of the degeneracy fix: a constant block -> equal
    # ranks -> genuine zero rank-variance.
    tied = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    ranks = _average_ranks(tied)
    assert list(ranks) == [2.0, 2.0, 2.0, 5.0, 5.0, 5.0]
    # And a fully-constant vector average-ranks to all-equal (variance exactly 0),
    # which is precisely what the old post-argsort guard failed to detect.
    assert float(np.std(_average_ranks(np.asarray([7.0] * 8)))) == 0.0


# ---------------------------------------------------------------------------
# Length / shape guards.
# ---------------------------------------------------------------------------

def test_too_short_returns_none():
    assert spearman([1.0], [2.0]) is None
    assert spearman([], []) is None


def test_mismatched_lengths_return_none():
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0]) is None
