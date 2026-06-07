"""Contract tests for experiment_runner cell-completion progress counting.

Background -- multi-stage / multi-cell ETA pathology (V3-EXQ-640b, 2026-06-07):
  Multi-cell experiments delimit their (seed x condition) cells with a
  "Seed N Condition X" label and print cumulative per-stage progress as
  "ep <done>/<total>" lines. Several such experiments (the scaffold cue-recall
  lineage, e.g. v3_exq_640b) print a per-cell completion line that carries NO
  PASS/FAIL token ("verdict: seed=... arm=..."), and print no "Status: PASS"
  line per cell. The runner detected cell completion ONLY via PASS/FAIL run-done
  patterns, so for those experiments run_end_times stayed EMPTY and runs_done
  stayed 0 for the entire run. The ETA estimator therefore never used its stable
  median-per-cell path and fell back to live-extrapolation driven by a percent
  stuck near 0 -- the displayed time-left was "all over the place".

Fix (this contract):
  A new "Seed N Condition X" label finalises the PREVIOUS cell for progress
  purposes (records a real per-cell duration in run_end_times and increments
  runs_done). PASS/FAIL run-done counting is suppressed once in this mode to
  avoid double-counting cells that DO emit a per-cell verdict word.

These tests mirror the runner's stdout-parsing state machine (the inline loop in
experiment_runner.run_experiment) using the REAL regexes and the REAL estimator,
so they guard both the regex set and the estimator integration. Keep this mirror
in sync with the loop blocks around the RE_SEED_CONDITION / RE_RUN_DONE_PATTERNS
handling in experiment_runner.py.

Run: pytest tests/contracts/test_runner_progress_cell_counting.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner as er  # noqa: E402


def _replay(lines, *, total_runs, episodes_per_run_floor,
            count_cell_boundary=True):
    """Faithful mirror of the runner's per-line progress state machine.

    Returns the final (runs_done, episodes_in_run, episodes_per_run,
    run_end_times, seen_seed_condition). `count_cell_boundary=False` reproduces
    the legacy (pre-fix) behaviour for the regression contrast.

    A monotonic clock is simulated: each line advances time by 1.0s.
    """
    runs_done = 0
    episodes_in_run = 0
    episodes_per_run = episodes_per_run_floor or 130
    run_end_times: list[float] = []
    seen_seed_condition = False
    clock = 0.0

    for line in lines:
        clock += 1.0

        m = er.RE_SEED_CONDITION.search(line)
        if m:
            if count_cell_boundary:
                if seen_seed_condition:
                    run_end_times.append(clock)
                    runs_done += 1
                seen_seed_condition = True
            episodes_in_run = 0

        m = er.RE_TRAIN_PROGRESS.search(line)
        if m:
            episodes_in_run = int(m.group(1))

        m = er.RE_EP_PROGRESS.search(line)
        if m:
            episodes_in_run = int(m.group(1))
            new_denom = int(m.group(2))
            if episodes_per_run_floor == 0 or new_denom >= episodes_per_run_floor:
                episodes_per_run = new_denom

        if not (count_cell_boundary and seen_seed_condition):
            done = False
            for pat in er.RE_RUN_DONE_PATTERNS:
                if pat.search(line):
                    run_end_times.append(clock)
                    runs_done += 1
                    episodes_in_run = episodes_per_run
                    done = True
                    break
            if not done and er.RE_STATUS_LINE.match(line):
                run_end_times.append(clock)
                runs_done = max(runs_done, 1)

    return runs_done, episodes_in_run, episodes_per_run, run_end_times, seen_seed_condition


def _exq640b_transcript(n_cells):
    """Synthetic 640b-style transcript: per-cell Seed/Condition label, cumulative
    per-stage `ep done/235` lines, and a non-PASS/FAIL `verdict: seed=...` line."""
    lines = []
    arms = ["ARM_OFF", "ARM_CUE_g0p2_k2", "ARM_CUE_g1_k2", "ARM_CUE_g5_k10"]
    for i in range(n_cells):
        seed = 42 + (i // len(arms))
        arm = arms[i % len(arms)]
        lines.append(f"Seed {seed} Condition {arm}")
        # cumulative per-stage prints (stage0 20, +0b 10, +p0 120, +p1 70, +p2 15)
        for done in (20, 30, 150, 220, 235):
            lines.append(f"  [train] stage seed={seed} arm={arm} ep {done}/235 foo=1")
        lines.append(f"verdict: seed={seed} arm={arm} pc_fire_steps=3 pc_windows=2")
    return lines


def test_c1_legacy_no_completion_detected_for_nonpassfail_verdict():
    # Pre-fix contrast: with cell-boundary counting OFF, the 640b transcript
    # produces ZERO recorded completions -> the estimator's stable path is dead.
    transcript = _exq640b_transcript(n_cells=8)
    runs_done, _, eppr, run_end_times, seen = _replay(
        transcript, total_runs=21, episodes_per_run_floor=235,
        count_cell_boundary=False,
    )
    assert run_end_times == [], "legacy path should record no cell completions"
    assert runs_done == 0
    assert eppr == 235  # floor protects the denominator


def test_c2_boundary_counting_records_each_completed_cell():
    # New behaviour: 8 cell labels -> 7 completed cells recorded (the 8th is
    # still in flight), runs_done climbs, run_end_times populated.
    transcript = _exq640b_transcript(n_cells=8)
    runs_done, _, eppr, run_end_times, seen = _replay(
        transcript, total_runs=21, episodes_per_run_floor=235,
    )
    assert seen is True
    assert runs_done == 7, f"expected 7 finalised cells, got {runs_done}"
    assert len(run_end_times) == 7
    assert eppr == 235


def test_c3_eta_uses_stable_median_path_after_first_cell():
    # After cells complete, run_end_times is non-empty -> estimator takes the
    # median-per-cell path and returns a finite, bounded ETA (NOT the inflating
    # live-extrapolation fallback).
    transcript = _exq640b_transcript(n_cells=8)
    runs_done, eir, eppr, run_end_times, _ = _replay(
        transcript, total_runs=21, episodes_per_run_floor=235,
    )
    # Each cell is 6 lines apart in the mirror clock (label + 5 stage prints),
    # so per-cell delta ~6s; 21 total, 7 done -> ~14 remaining.
    eta = er._estimate_seconds_remaining(
        run_end_times=run_end_times, runs_done=runs_done, total_runs=21,
        episodes_in_run=eir, episodes_per_run=eppr,
        elapsed=run_end_times[-1], static_secs=600 * 60, pct=50.0,
    )
    assert run_end_times, "stable path requires recorded completions"
    remaining_cells = 21 - runs_done
    # median delta ~6s -> ETA should be on the order of remaining_cells * 6s,
    # far below the 600-min static estimate the broken fallback inflated past.
    assert 0 < eta < remaining_cells * 60, f"ETA {eta}s not on the per-cell scale"


def test_c4_single_run_passfail_experiment_still_counts_via_verdict():
    # Experiments that emit NO Seed/Condition label must keep using the
    # PASS/FAIL run-done detection (no regression for single-run experiments).
    transcript = [
        "  [train] ep 1/100",
        "  [train] ep 50/100",
        "  [train] ep 100/100",
        "verdict: PASS some-experiment done",
    ]
    runs_done, _, _, run_end_times, seen = _replay(
        transcript, total_runs=1, episodes_per_run_floor=100,
    )
    assert seen is False
    assert runs_done == 1
    assert len(run_end_times) == 1


def test_c5_no_double_count_when_cell_emits_passfail_verdict():
    # A multi-cell experiment that ALSO prints a per-cell "verdict: PASS" must
    # count each cell exactly once (boundary), not twice (boundary + verdict).
    lines = []
    for i in range(4):
        lines.append(f"Seed {42 + i} Condition ARM_X")
        lines.append("  [train] ep 130/130")
        lines.append("verdict: PASS cell done")
    runs_done, _, _, run_end_times, _ = _replay(
        lines, total_runs=4, episodes_per_run_floor=130,
    )
    # 4 labels -> 3 finalised cells (last in flight); verdict lines must not add.
    assert runs_done == 3, f"double-counting detected: runs_done={runs_done}"
    assert len(run_end_times) == 3
