"""Contract tests for experiment_runner._estimate_seconds_remaining.

Background -- heterogeneous-arm ETA pathology (2026-06-06):
  A multi-arm experiment runs (seed x arm) cells of WILDLY different cost: an
  OFF/baseline arm can finish in ~0s (reused/skipped under arm-reuse, or a cheap
  dry/early-exit cell) while treatment arms take minutes-to-hours. The legacy
  estimator used cumulative_elapsed / runs_done (mean per run), so when the first
  cells were instantaneous the rate collapsed and the ETA reported ~2s remaining
  for an experiment that still had expensive arms to run.

Fix:
  Calibrate the remaining-run rate from REAL (>= floor) arm durations only
  (median of the most-recent real runs), and fall back to the static
  estimated_minutes estimate while NO real arm has been timed yet -- never
  extrapolate ~0 from zero-cost cells.

Contracts:
  C1. Instant early arms + one real arm -> ETA reflects the real arm rate, not ~0.
  C2. Only instant arms completed so far -> ETA falls back to the static estimate.
  C3. Homogeneous arms -> ETA ~= legacy mean-per-run behaviour (no regression).
  C4. pct <= 0 -> static estimate (unchanged legacy contract).
  C5. Completions present but no run-end deltas / blend path stays finite > 0.
  C6. A single slow outlier among real arms does not blow up the median estimate.

Run: pytest tests/contracts/test_runner_eta_heterogeneous_arms.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402

_est = experiment_runner._estimate_seconds_remaining


def test_c1_instant_early_arms_use_real_rate_not_zero():
    # 3 instant cells (0.1s each) then 1 real cell (~600s); 8 total runs, 4 done.
    cum = [0.1, 0.2, 0.3, 600.3]
    eta = _est(
        run_end_times=cum, runs_done=4, total_runs=8,
        episodes_in_run=0, episodes_per_run=10,
        elapsed=600.3, static_secs=99999.0, pct=50.0,
    )
    # remaining = 4 runs, per-run rate from the real arm ~600s -> ~2400s.
    # The legacy mean estimator would have given ~600.3/4 * 4 = ~600s, but with
    # only the instants timed before the real arm it collapsed toward 0.
    assert eta > 1000.0, f"ETA collapsed to {eta}; should reflect the real arm rate"


def test_c2_all_instant_falls_back_to_static():
    cum = [0.1, 0.2, 0.3]
    eta = _est(
        run_end_times=cum, runs_done=3, total_runs=8,
        episodes_in_run=0, episodes_per_run=10,
        elapsed=0.3, static_secs=3600.0, pct=37.5,
    )
    # No real arm timed yet -> static estimate minus elapsed, NOT ~0.
    assert eta > 3000.0, f"ETA {eta} should fall back near the static estimate"


def test_c3_homogeneous_matches_legacy_mean():
    # 4 runs, 100s each (cumulative 100,200,300,400); 8 total -> 4 remaining.
    cum = [100.0, 200.0, 300.0, 400.0]
    eta = _est(
        run_end_times=cum, runs_done=4, total_runs=8,
        episodes_in_run=0, episodes_per_run=10,
        elapsed=400.0, static_secs=99999.0, pct=50.0,
    )
    # per-run ~100s x 4 remaining ~= 400s (legacy mean estimator gave the same).
    assert 300.0 <= eta <= 500.0, f"homogeneous ETA {eta} drifted from legacy ~400s"


def test_c4_pct_zero_returns_static():
    eta = _est(
        run_end_times=[], runs_done=0, total_runs=8,
        episodes_in_run=0, episodes_per_run=10,
        elapsed=0.0, static_secs=1234.0, pct=0.0,
    )
    assert eta == 1234.0


def test_c5_no_run_deltas_blend_path_finite_positive():
    # pct > 0 but no completed runs -> legacy static/live blend path, > 0.
    eta = _est(
        run_end_times=[], runs_done=0, total_runs=8,
        episodes_in_run=5, episodes_per_run=10,
        elapsed=60.0, static_secs=600.0, pct=10.0,
    )
    assert eta > 0.0


def test_c6_slow_outlier_does_not_blow_up_estimate():
    # 4 real arms ~100s and one 10000s outlier; median is robust.
    cum = [100.0, 200.0, 300.0, 400.0, 10400.0]
    eta = _est(
        run_end_times=cum, runs_done=5, total_runs=8,
        episodes_in_run=0, episodes_per_run=10,
        elapsed=10400.0, static_secs=99999.0, pct=62.5,
    )
    # median of recent real durations ~100-300s, 3 remaining -> well under the
    # 10000s outlier x 3 a mean estimator would have produced.
    assert eta < 3000.0, f"median-robust ETA {eta} should ignore the slow outlier"


def test_c7_instant_cells_excluded_from_rate_with_real_present():
    # Interleaved: real(120), instant(0.05), real(110), instant(0.05); 6 total.
    cum = [120.0, 120.05, 230.05, 230.10]
    eta = _est(
        run_end_times=cum, runs_done=4, total_runs=6,
        episodes_in_run=0, episodes_per_run=10,
        elapsed=230.10, static_secs=99999.0, pct=66.6,
    )
    # 2 remaining real arms at ~110-120s each -> ~220-240s; the instants must
    # not be averaged in (which would roughly halve the per-run rate).
    assert eta > 150.0, f"ETA {eta} suggests instant cells contaminated the rate"
