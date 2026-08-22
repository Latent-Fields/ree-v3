"""Contracts for the precondition-recomputability lint.

Surfaces under test:
  (1) validate_experiments.precondition_recomputability_lint -- flags a precondition
      whose `met` is NOT recomputable from the reported measured/threshold/direction
      triple, in either of two ways: a missing `direction` key, or a `met` computed
      from a demonstrably different statistic than `measured`.
  (2) validate_experiments.py --checks precondition_recomputability -- the selector,
      and the invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under
      --paths, never affects the exit code even under --strict).

WHY THIS GATE EXISTS. A precondition's whole job is to let a manifest reader re-derive
the self-route's premise, and `build_experiment_indexes._compute_adjudication` does
exactly that -- it RECOMPUTES `met` from the numeric measured/threshold pair and does
NOT trust the author's `met`. Two ways that breaks:

  (a) NO `direction` -> the indexer defaults to a FLOOR recompute, which silently
      inverts a ceiling-shaped check (the 2026-06-07 V3-EXQ-648a/649 directionality
      bug: a "stayed below a ceiling" precondition with measured << threshold gets
      false-flagged `precondition_unmet`).
  (b) `met` COMPUTED FROM A DIFFERENT STATISTIC. Confirmed: V3-EXQ-726 shipped
      `strong_f_monolithic_latch_present` with `measured` a median-across-seeds of
      per-seed medians against a `met` that was a seed COUNT (`>= 2 seeds`). Those
      coincide at exactly n=3 seeds and diverge in dry-run and at every other seed
      count -- and it carried no `direction` either. Fixed 2026-07-18 in ree-v3
      fd7ca8c7cb by re-expressing both sides as a seed FRACTION (numerically identical
      to the count gate at n=3, so the pre-registered gate was unchanged).

Sibling gate: test_anchor_reachability_lint.py -- that one checks an anchor-kind
precondition ships a reachability GUARD; it does NOT check that measured and met are
the same statistic, nor that `direction` is declared. This file covers that gap.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# The V3-EXQ-726 defect shape, reduced: `measured` is a median-across-seeds while
# `met` resolves to a seed COUNT, and no `direction` is declared.
_MISMATCHED = '''
import statistics
FLOOR = 2000.0
MIN_SEEDS = 2

def _median(xs):
    return statistics.median(xs) if xs else 0.0

def main():
    seeds = [42, 43, 44]
    contrast = {s: {"occupancy_median": 3000.0} for s in seeds}
    contrast_occ = [contrast[s]["occupancy_median"] for s in seeds]
    contrast_seeds_strongf = [s for s in seeds if contrast[s]["occupancy_median"] > FLOOR]
    strong_f_ok = len(contrast_seeds_strongf) >= MIN_SEEDS
    interpretation = {
        "preconditions": [
            {
                "name": "strong_f_monolithic_latch_present",
                "description": "the sustained natural-commit hold forms in GAP-A",
                "measured": round(_median(contrast_occ), 3),
                "threshold": FLOOR,
                "met": bool(strong_f_ok),
            },
        ],
    }
    return interpretation

if __name__ == "__main__":
    main()
'''

# The fix: ONE statistic on both sides (a seed fraction), plus an explicit direction.
_RECOMPUTABLE = '''
FLOOR = 2000.0
MIN_SEEDS = 2
ANCHOR_MIN_FRAC = 2.0 / 3.0

def main():
    seeds = [42, 43, 44]
    contrast = {s: {"occupancy_median": 3000.0} for s in seeds}
    contrast_seeds_strongf = [s for s in seeds if contrast[s]["occupancy_median"] > FLOOR]
    latch_seeds_frac = (float(len(contrast_seeds_strongf)) / float(len(seeds))) if seeds else 0.0
    strong_f_ok = bool(seeds) and latch_seeds_frac >= ANCHOR_MIN_FRAC
    interpretation = {
        "preconditions": [
            {
                "name": "strong_f_monolithic_latch_present",
                "description": "measured/threshold/met are one statistic",
                "measured": round(latch_seeds_frac, 4),
                "threshold": round(ANCHOR_MIN_FRAC, 4),
                "direction": "lower",
                "met": bool(strong_f_ok),
            },
        ],
    }
    return interpretation

if __name__ == "__main__":
    main()
'''


_BAND_SINGLE_BOUND = '''
E_SAT_LOW = 0.02
E_SAT_HIGH = 0.98

def main():
    rows = [{"S": 0.53}]
    headroom = bool(rows) and all(E_SAT_LOW < r["S"] < E_SAT_HIGH for r in rows)
    interpretation = {
        "preconditions": [
            {
                "name": "baseline_entropy_headroom",
                "measured": rows[0]["S"],
                "threshold": E_SAT_HIGH,
                "direction": "upper",
                "met": bool(headroom),
            },
        ],
    }
    return interpretation

if __name__ == "__main__":
    main()
'''


_BAND_CONJOINED = _BAND_SINGLE_BOUND.replace(
    'all(E_SAT_LOW < r["S"] < E_SAT_HIGH for r in rows)',
    'all(r["S"] > E_SAT_LOW and r["S"] < E_SAT_HIGH for r in rows)')


_BAND_INTERVAL = '''
E_SAT_LOW = 0.02
E_SAT_HIGH = 0.98

def main():
    rows = [{"S": 0.53}]
    headroom = bool(rows) and all(E_SAT_LOW < r["S"] < E_SAT_HIGH for r in rows)
    interpretation = {
        "preconditions": [
            {
                "name": "baseline_entropy_headroom",
                "measured": rows[0]["S"],
                "threshold_low": E_SAT_LOW,
                "threshold_high": E_SAT_HIGH,
                "comparator_low": ">",
                "comparator_high": "<",
                "direction": "interval",
                "met": bool(headroom),
            },
        ],
    }
    return interpretation

if __name__ == "__main__":
    main()
'''


def _lint(src: str):
    """Write src to a temp .py under experiments/ and return the lint result."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        return V.precondition_recomputability_lint(Path(name))
    finally:
        os.unlink(name)


# ---- (1c) two-sided band declared with a single bound (2026-07-19) ---------
# V3-EXQ-779b baseline_entropy_headroom: a strict 0.02 < S < 0.98 band shipped as
# direction:"upper" + threshold 0.98. It passed the (a) branch (it HAS a direction)
# while its floor leg was absent from the manifest entirely, so the indexer
# recomputed `met` from the ceiling alone and a saturated-to-zero baseline -- the
# exact degeneracy the check exists to catch -- recomputed as MET.

def test_r0a_chained_band_with_single_bound_flagged():
    out = _lint(_BAND_SINGLE_BOUND)
    assert out is not None
    assert "TWO-SIDED band" in out, out
    assert "baseline_entropy_headroom" in out, out


def test_r0b_conjoined_band_with_single_bound_flagged():
    """`x > LOW and x < HIGH` is the same defect spelled differently."""
    out = _lint(_BAND_CONJOINED)
    assert out is not None
    assert "TWO-SIDED band" in out, out


def test_r0c_interval_declaration_silences_it():
    """The fix: threshold_low + threshold_high. Nothing else may fire either --
    notably the (a) no-direction branch must not trip on an interval entry that
    carries no single `threshold`."""
    assert _lint(_BAND_INTERVAL) is None


def test_r0d_single_bound_check_is_not_flagged_as_a_band():
    """Conservatism guard: an ordinary one-sided floor must NOT trip the band
    branch. Regression target is over-firing, which would bury the real hits."""
    out = _lint(_RECOMPUTABLE)
    assert out is None or "TWO-SIDED band" not in out, out


def test_r0e_opposed_chain_is_not_a_band():
    """`a < b > c` bounds nothing -- the ops must point the SAME way to squeeze
    the middle operand."""
    src = _BAND_SINGLE_BOUND.replace(
        'all(E_SAT_LOW < r["S"] < E_SAT_HIGH for r in rows)',
        'all(E_SAT_LOW < r["S"] > E_SAT_HIGH for r in rows)')
    out = _lint(src)
    assert out is None or "TWO-SIDED band" not in out, out


def test_r0f_conjunction_on_different_subjects_is_not_a_band():
    """`a > LOW and b < HIGH` bounds two different things, not one interval."""
    src = _BAND_SINGLE_BOUND.replace(
        'all(E_SAT_LOW < r["S"] < E_SAT_HIGH for r in rows)',
        'all(r["S"] > E_SAT_LOW and r["T"] < E_SAT_HIGH for r in rows)')
    out = _lint(src)
    assert out is None or "TWO-SIDED band" not in out, out


def test_r0g_band_branch_is_warn_only():
    """INVARIANT: like every branch of this gate, the band branch never blocks."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_BAND_SINGLE_BOUND)
        name = f.name
    try:
        r = _run("--checks", "precondition_recomputability", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "TWO-SIDED band" in r.stdout
    finally:
        os.unlink(name)


# ---- (1d) central-tendency `measured` vs worst-case `met` (2026-07-19) -----
# V3-EXQ-779b tonic_axis_live: `measured` is an fmean over the TONIC-ON cells while
# `met` is `all(cell >= FLOOR)`. Mean and worst-case are different statistics, so a
# single out-of-band row masked by an in-band mean recomputes MET while the script's
# own `met` is False. Same CLASS as the 726 branch (b), but (b) only fires on
# central-tendency-vs-CARDINALITY, so this shape slipped through. Note the sibling
# SAMPLE-kind preconditions in the same file get it right via a `_worst_cell(...)`
# helper -- `measured` IS the worst case there, and recomputes exactly.

_MEAN_VS_ALL = '''
import statistics
FLOOR = 0.05

def main():
    t1_rows = [{"lift": 0.20}, {"lift": 0.01}]
    live = bool(t1_rows) and all(r["lift"] >= FLOOR for r in t1_rows)
    interpretation = {
        "preconditions": [
            {
                "name": "tonic_axis_live",
                "kind": "capability",
                "measured": (statistics.fmean([r["lift"] for r in t1_rows])
                             if t1_rows else 0.0),
                "threshold": FLOOR,
                "direction": "lower",
                "met": bool(live),
            },
        ],
    }
    return interpretation

if __name__ == "__main__":
    main()
'''

# The fix: report the WORST cell, so `measured` and `met` are one statistic.
_WORST_CELL = _MEAN_VS_ALL.replace(
    '"measured": (statistics.fmean([r["lift"] for r in t1_rows])\n'
    '                             if t1_rows else 0.0),',
    '"measured": (min(r["lift"] for r in t1_rows) if t1_rows else 0.0),')


def test_r0h_mean_measured_vs_all_met_flagged():
    out = _lint(_MEAN_VS_ALL)
    assert out is not None
    assert "WORST-CASE claim over the SAME collection" in out, out
    assert "tonic_axis_live" in out, out


def test_r0i_any_and_extremum_met_are_the_same_defect():
    """`any(...)` and a `min()/max()` reduction are the other spellings of a
    worst-case `met`; all three are claims about a ROW, not about the centre."""
    for repl in ('any(r["lift"] >= FLOOR for r in t1_rows)',
                 'min(r["lift"] for r in t1_rows) >= FLOOR'):
        src = _MEAN_VS_ALL.replace('all(r["lift"] >= FLOOR for r in t1_rows)', repl)
        out = _lint(src)
        assert out is not None and "WORST-CASE claim" in out, (repl, out)


def test_r0j_worst_cell_measured_silences_it():
    """The fix, and the shape this branch steers toward: `measured = min(...)`
    recomputes exactly against an `all(... >= FLOOR)` met. Nothing may fire."""
    assert _lint(_WORST_CELL) is None


def test_r0k_extremum_over_group_means_is_not_flagged():
    """Conservatism guard for conjunct 3. A central-tendency call INSIDE a
    worst-case reduction (`max` of per-group means) is still a worst-case
    `measured`, so the presence of `fmean` must not on its own trip the branch."""
    src = _MEAN_VS_ALL.replace(
        '(statistics.fmean([r["lift"] for r in t1_rows])\n'
        '                             if t1_rows else 0.0)',
        '(min(statistics.fmean(r["lift"]) for r in t1_rows) if t1_rows else 0.0)')
    out = _lint(src)
    assert out is None or "WORST-CASE claim" not in out, out


def test_r0l_different_collections_are_not_flagged():
    """Conservatism guard for conjunct 4. The shared variable is the ANCHOR for
    this branch (it proves both sides read the same rows) -- inverted relative to
    the (b) branch, where a shared variable is the let-off. With no collection in
    common there is no evidence the two statistics describe the same thing.

    Note the empty-guard subtlety this test had to be written around: the idiomatic
    `fmean(...) if rows else 0.0` mentions the collection in its GUARD as well as in
    the reduction, so swapping only the reduction leaves the name shared and the
    branch still (correctly, on its own terms) fires. Both had to move for the
    collections to be genuinely disjoint."""
    src = _MEAN_VS_ALL.replace(
        '(statistics.fmean([r["lift"] for r in t1_rows])\n'
        '                             if t1_rows else 0.0)',
        '(statistics.fmean([q["lift"] for q in other_rows])\n'
        '                             if other_rows else 0.0)').replace(
        '    t1_rows = [{"lift": 0.20}, {"lift": 0.01}]',
        '    t1_rows = [{"lift": 0.20}, {"lift": 0.01}]\n'
        '    other_rows = [{"lift": 0.20}]')
    out = _lint(src)
    assert out is None or "WORST-CASE claim" not in out, out


def test_r0m_central_tendency_on_both_sides_is_not_flagged():
    """A mean `measured` against a mean `met` is ONE statistic -- recomputable,
    and outside this branch entirely (there is no quantifier in `met`)."""
    src = _MEAN_VS_ALL.replace(
        'all(r["lift"] >= FLOOR for r in t1_rows)',
        'statistics.fmean([r["lift"] for r in t1_rows]) >= FLOOR')
    out = _lint(src)
    assert out is None or "WORST-CASE claim" not in out, out


def test_r0n_central_vs_worst_branch_is_warn_only():
    """INVARIANT: like every branch of this gate, this one never blocks -- see
    test_r13/test_r0g. `measured` is computed from live run data, so a static scan
    can only ever flag a SUSPECTED mismatch, never prove one."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_MEAN_VS_ALL)
        name = f.name
    try:
        r = _run("--checks", "precondition_recomputability", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "WORST-CASE claim" in r.stdout
    finally:
        os.unlink(name)


# ---- (1) lint detection branches -------------------------------------------

def test_r1_median_measured_vs_count_met_flagged():
    """The 726 shape: central-tendency `measured` against a cardinality `met`."""
    issue = _lint(_MISMATCHED)
    assert issue is not None
    assert "strong_f_monolithic_latch_present" in issue
    assert "CENTRAL-TENDENCY" in issue and "CARDINALITY" in issue


def test_r2_one_statistic_with_direction_is_clean():
    """Both sides routing through one statistic, direction declared -> silent."""
    assert _lint(_RECOMPUTABLE) is None


def test_r3_missing_direction_alone_flagged():
    """(a) fires independently of the statistic-mismatch branch."""
    src = _RECOMPUTABLE.replace('                "direction": "lower",\n', "")
    assert '"direction"' not in src
    issue = _lint(src)
    assert issue is not None
    assert "NO `direction` key" in issue
    # ...and ONLY that branch -- the two sides still share latch_seeds_frac.
    assert "CENTRAL-TENDENCY" not in issue


def test_r3b_comparator_key_also_satisfies_the_direction_requirement():
    """`comparator` is an EQUIVALENT declaration, not a missing one.

    build_experiment_indexes._precondition_direction consults `comparator` FIRST
    (">="/">" -> lower, "<="/"<" -> upper) and only falls back to `direction`. A
    precondition authored the comparator way is fully recomputable, so keying this
    branch on `direction` alone would false-fire on correct code. Verified against
    the indexer 2026-07-18; no corpus script uses `comparator` yet, so this test is
    the only thing holding the behaviour.
    """
    src = _RECOMPUTABLE.replace('                "direction": "lower",\n',
                                '                "comparator": ">=",\n')
    assert '"direction"' not in src
    assert _lint(src) is None


def test_r4_shared_variable_suppresses_the_mismatch_branch():
    """THE CONSERVATISM GUARD.

    `latch_seeds_frac` is itself defined from a `len(...)`. If the resolution chased
    variables transitively, the post-fix 726 shape would look like a median-vs-count
    mismatch and the check would fire on the FIX. The shallow one-level resolution plus
    the shared-variable test is what prevents that; this test fails if someone deepens
    the resolution.
    """
    assert "len(" in _RECOMPUTABLE, "fixture must have a len() upstream of measured"
    assert _lint(_RECOMPUTABLE) is None


def test_r5_criterion_dict_is_not_a_precondition():
    """A dict carrying load_bearing/passed is a CRITERION -- out of scope."""
    src = _MISMATCHED.replace('                "met": bool(strong_f_ok),\n',
                              '                "met": bool(strong_f_ok),\n'
                              '                "load_bearing": True,\n')
    assert _lint(src) is None


def test_r6_non_numeric_measured_out_of_scope():
    """A string `measured` makes no numeric floor/ceiling claim to recompute."""
    src = _MISMATCHED.replace('                "measured": round(_median(contrast_occ), 3),\n',
                              '                "measured": "monolithic_latch_regime",\n')
    assert _lint(src) is None


def test_r7_exempt_marker_suppresses():
    """PRECONDITION_RECOMPUTABILITY_EXEMPT opts out."""
    src = ('PRECONDITION_RECOMPUTABILITY_EXEMPT = "categorical admissibility check"\n'
           + _MISMATCHED)
    assert _lint(src) is None


def test_r8_library_file_without_main_exempt():
    """No __main__ entry point -> library helper, exempt."""
    src = _MISMATCHED.replace('if __name__ == "__main__":\n    main()\n', "")
    assert _lint(src) is None


def test_r9_gate_is_wider_than_the_anchor_gate():
    """REGRESSION GUARD ON THE GATE'S OWN SCOPE.

    Recomputability is owed by EVERY precondition the indexer reads, not only the
    anchor-kind ones (those carrying a `control` key). The 726 defect is a
    recomputability failure whether or not the entry anchors to a known-positive
    control -- and indeed the pre-fix 726 precondition DID carry a `control` key while
    the anchor gate, which keys on exactly that, still could not see this defect. This
    test fails if someone narrows the scan to `control`-bearing dicts.
    """
    assert '"control"' not in _MISMATCHED, "fixture must carry no control key"
    assert _lint(_MISMATCHED) is not None, (
        "preconditions without a `control` key must stay in scope; narrowing to "
        "anchor-kind would exempt most of the defect class")


# ---- (2) real-corpus regression pair (V3-EXQ-726) ---------------------------

_FIX_COMMIT = "fd7ca8c7cb"
_S726 = "experiments/v3_exq_726_sd033e_frontopolar_decommit_gapa.py"


def _at_revision(rev: str):
    """Materialise `_S726` at `rev` into a temp file under experiments/, or None."""
    r = subprocess.run(["git", "show", f"{rev}:{_S726}"],
                       capture_output=True, text=True, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        return None  # shallow clone / history unavailable -- synthetic fixtures cover it
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(r.stdout)
        return f.name


def test_r10_fires_on_the_pre_fix_726():
    """The check must catch the defect that motivated it."""
    name = _at_revision(f"{_FIX_COMMIT}~1")
    if name is None:
        return
    try:
        issue = V.precondition_recomputability_lint(Path(name))
        assert issue is not None
        assert "strong_f_monolithic_latch_present" in issue
        # BOTH defects were present pre-fix: the statistic mismatch AND no direction.
        assert "CENTRAL-TENDENCY" in issue
        assert "NO `direction` key" in issue
    finally:
        os.unlink(name)


def test_r11_silent_on_the_fixed_726():
    """And must go quiet on the fix at fd7ca8c7cb."""
    name = _at_revision(_FIX_COMMIT)
    if name is None:
        return
    try:
        assert V.precondition_recomputability_lint(Path(name)) is None
    finally:
        os.unlink(name)


# ---- (3) CLI selector + never-blocking invariant ---------------------------

def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def test_r12_checks_selector_accepts_precondition_recomputability():
    assert "precondition_recomputability" in V.CHECK_NAMES
    target = EXPERIMENTS_DIR / Path(_S726).name
    if not target.exists():
        return
    r = _run("--checks", "precondition_recomputability", "--quiet", "--paths", str(target))
    assert r.returncode == 0
    assert "precondition-recomputability-warning(s)" in r.stdout


def test_r13_warn_only_under_paths_and_strict():
    """INVARIANT: this gate never blocks -- not under --paths, not under --strict.

    Like the anchor-reachability gate and unlike the arm-fingerprint / degeneracy /
    manifest-writer gates, it stays advisory in BOTH modes. `measured` is computed from
    live run data, so the lint can only ever flag a SUSPECTED mismatch between two
    expressions -- never prove the reported triple fails to recompute. It must not fail
    a commit. This test exists so nobody later "tidies" it into a hard gate.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_MISMATCHED)
        name = f.name
    try:
        r = _run("--checks", "precondition_recomputability", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, (
            "precondition-recomputability must be WARN-only even under --strict "
            f"--paths; stdout={r.stdout[-2000:]}")
        assert "RECOMPUTABILITY WARNINGS" in r.stdout
    finally:
        os.unlink(name)


def test_r14_selector_is_surgical():
    """--checks precondition_recomputability runs ONLY this gate."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_MISMATCHED)
        name = f.name
    try:
        r = _run("--checks", "precondition_recomputability", "--quiet", "--paths", name)
        assert r.returncode == 0
        assert "0 non-conforming" in r.stdout
    finally:
        os.unlink(name)


# ---- (1e) saturation band scoped to ONE partition (2026-07-19) ------------
# V3-EXQ-779b autopsy section 7. `baseline_entropy_headroom` ranged over
# `baseline_rows` (arm == T0P0) while `t1_rows` / `p1_rows` -- sibling partitions
# of the same `rows` -- were never band-checked. Seed 23 reported met=True at
# baseline 0.6093 while its tonic-ON arms sat at 0.8489 / 0.8587 against
# E_SAT_HIGH = 0.98: an unguarded near-ceiling exposure on the arms carrying the
# manipulation, which surfaced only when a human read the per-arm numbers.

_PARTITION_SCOPED = '''
E_SAT_LOW = 0.02
E_SAT_HIGH = 0.98
FLOOR = 0.05


def main():
    rows = [
        {"arm": "T0P0", "S": 0.61, "lift": 0.2},
        {"arm": "T1P0", "S": 0.85, "lift": 0.2},
        {"arm": "T1P1", "S": 0.86, "lift": 0.2},
    ]
    baseline_rows = [r for r in rows if r["arm"] == "T0P0"]
    t1_rows = [r for r in rows if r["arm"] != "T0P0"]
    headroom = all(E_SAT_LOW < r["S"] < E_SAT_HIGH for r in baseline_rows)
    tonic_live = all(r["lift"] >= FLOOR for r in t1_rows)
    interpretation = {
        "preconditions": [
            {
                "name": "baseline_entropy_headroom",
                "measured": min(r["S"] for r in baseline_rows),
                "threshold_low": E_SAT_LOW,
                "threshold_high": E_SAT_HIGH,
                "comparator_low": ">",
                "comparator_high": "<",
                "direction": "interval",
                "met": headroom,
            },
        ],
        "tonic_live": tonic_live,
    }
    return interpretation

if __name__ == "__main__":
    main()
'''


def test_r0o_partition_scoped_band_flagged():
    out = _lint(_PARTITION_SCOPED)
    assert out is not None
    assert "SATURATION GUARD" in out, out
    assert "baseline_entropy_headroom" in out, out


def test_r0o2_fix_advice_is_diagnostic_not_wider_precondition():
    """The remedy must NOT be 'check all arms'. A saturating TREATMENT arm is not a
    substrate-readiness failure, and self-routing it as one mislabels the cause -- the
    substrate was ready, the manipulation exceeded the readout's dynamic range. The
    message has to steer to the non-gating diagnostic, and say so explicitly."""
    out = _lint(_PARTITION_SCOPED)
    assert "NON-GATING diagnostic" in out, out
    assert "per_arm_headroom" in out, out
    assert "do NOT" in out and "widen the precondition" in out, out


def test_r0p_band_over_unfiltered_collection_not_flagged():
    """CONSERVATISM: a band that already ranges over every row has nothing unguarded,
    even though sibling partitions exist elsewhere in the file for other purposes."""
    src = _PARTITION_SCOPED.replace(
        'all(E_SAT_LOW < r["S"] < E_SAT_HIGH for r in baseline_rows)',
        'all(E_SAT_LOW < r["S"] < E_SAT_HIGH for r in rows)')
    out = _lint(src)
    assert out is None or "SATURATION GUARD" not in out, out


def test_r0q_no_sibling_partition_not_flagged():
    """CONSERVATISM: with only ONE partition there are no unchecked sibling arms, so
    the band's scope is not evidence of anything."""
    src = _PARTITION_SCOPED.replace(
        '    t1_rows = [r for r in rows if r["arm"] != "T0P0"]\n', '').replace(
        'all(r["lift"] >= FLOOR for r in t1_rows)',
        'all(r["lift"] >= FLOOR for r in rows)')
    out = _lint(src)
    assert out is None or "SATURATION GUARD" not in out, out


def test_r0r_one_sided_floor_not_flagged():
    """CONSERVATISM, and the load-bearing half of the ceiling/floor asymmetry.

    A one-sided FLOOR is not a saturation guard: `S > LOW` asserts the readout is
    above a minimum, which says nothing about whether it has room left to MOVE. A
    one-sided CEILING is a saturation guard (test_r0t), and conflating the two is
    exactly the over-generalisation that made this branch miss V3-EXQ-777. This test
    pins the floor half -- it must keep NOT firing after the ceiling widening."""
    src = _PARTITION_SCOPED.replace(
        'all(E_SAT_LOW < r["S"] < E_SAT_HIGH for r in baseline_rows)',
        'all(r["S"] > E_SAT_LOW for r in baseline_rows)').replace(
        '"threshold_low": E_SAT_LOW,\n                "threshold_high": E_SAT_HIGH,',
        '"threshold": E_SAT_LOW,')
    out = _lint(src)
    assert out is None or "SATURATION GUARD" not in out, out


# ---------------------------------------------------------------------------
# r0t-r0w: branch (e) widened to ONE-SIDED CEILINGS (2026-07-19).
#
# Branch (e) originally required _is_two_sided. The stated rationale -- "a one-sided
# floor is not a saturation guard" -- is true of a FLOOR but was over-generalised to
# CEILINGS, which are exactly saturation guards. Confirmed miss, same claim family:
# v3_exq_777_mech063_orthogonal_control_axes_dissociation.py:441-444 scoped
# `r["E_norm_entropy_mean"] < E_SAT_CEIL` to `baseline_rows` (arm == A0B0) with
# `a1_rows` / `b1_rows` sibling partitions unchecked -- structurally identical to
# 779b, and it did not fire. Note _is_two_sided is CORRECT to decline it: the
# conjunction's two Compares have different subjects (E_norm_entropy_mean vs
# D_action_mass_std), so it is not a band on one subject. The defect was branch (e)'s
# two-sided requirement, not that predicate.
#
# Fire rate re-measured over all 1142 scripts in experiments/ after widening: 5 hits,
# all `baseline_entropy_headroom` (779/779a/779b two-sided + 777/777a ceiling), zero
# false positives -- so no name-substring or constant-name narrowing was needed.
#
# 777/777a have RUN. They are the detection witness, not files to repair: the lint is
# WARN-only and gates NEW scripts, and retro-editing a completed run's pre-registered
# emission is off-limits (user decision 2026-07-19, mean-vs-all lint).

_PARTITION_SCOPED_CEILING = '''
E_SAT_CEIL = 0.98
FLOOR = 0.05


def main():
    rows = [
        {"arm": "A0B0", "S": 0.61, "spread": 0.3, "lift": 0.2},
        {"arm": "A1B0", "S": 0.97, "spread": 0.3, "lift": 0.2},
        {"arm": "A1B1", "S": 0.97, "spread": 0.3, "lift": 0.2},
    ]
    a1_rows = [r for r in rows if r["arm"] != "A0B0"]
    baseline_rows = [r for r in rows if r["arm"] == "A0B0"]
    headroom = bool(baseline_rows) and all(
        r["S"] < E_SAT_CEIL and r["spread"] > 0.0 for r in baseline_rows
    )
    axis_live = all(r["lift"] >= FLOOR for r in a1_rows)
    interpretation = {
        "preconditions": [
            {
                "name": "baseline_entropy_headroom",
                "measured": min(r["S"] for r in baseline_rows),
                "threshold": E_SAT_CEIL,
                "direction": "upper",
                "met": bool(headroom),
            },
        ],
        "axis_live": axis_live,
    }
    return interpretation

if __name__ == "__main__":
    main()
'''


def test_r0t_partition_scoped_one_sided_ceiling_flagged():
    """The V3-EXQ-777 shape: a baseline-scoped CEILING with unchecked sibling arms.

    This is the whole point of the widening -- before it, this returned None."""
    out = _lint(_PARTITION_SCOPED_CEILING)
    assert out is not None
    assert "SATURATION GUARD" in out, out
    assert "baseline_entropy_headroom" in out, out


def test_r0t2_ceiling_hit_gets_the_same_diagnostic_advice():
    """The remedy for a ceiling hit is the same as for a band hit: the NON-GATING
    per-arm diagnostic, never widening the precondition to all arms."""
    out = _lint(_PARTITION_SCOPED_CEILING)
    assert "NON-GATING diagnostic" in out, out
    assert "per_arm_headroom" in out, out
    assert "do NOT" in out and "widen the precondition" in out, out


def test_r0u_ceiling_over_unfiltered_collection_not_flagged():
    """CONSERVATISM carries over: a ceiling already covering every row guards nothing
    less than the full collection, sibling partitions notwithstanding."""
    src = _PARTITION_SCOPED_CEILING.replace(
        'for r in baseline_rows\n    )', 'for r in rows\n    )')
    out = _lint(src)
    assert out is None or "SATURATION GUARD" not in out, out


def test_r0v_ceiling_with_no_sibling_partition_not_flagged():
    """CONSERVATISM carries over: one partition means no unchecked sibling arms."""
    src = _PARTITION_SCOPED_CEILING.replace(
        '    a1_rows = [r for r in rows if r["arm"] != "A0B0"]\n', '').replace(
        'all(r["lift"] >= FLOOR for r in a1_rows)',
        'all(r["lift"] >= FLOOR for r in rows)')
    out = _lint(src)
    assert out is None or "SATURATION GUARD" not in out, out


def test_r0w_ceiling_branch_is_warn_only():
    """INVARIANT: the widened branch never blocks either, in either mode."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_PARTITION_SCOPED_CEILING)
        name = f.name
    try:
        r = _run("--checks", "precondition_recomputability", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "SATURATION GUARD" in r.stdout
    finally:
        os.unlink(name)


def test_r0x_real_777_is_the_detection_witness():
    """Pins the confirmed miss against the real file, not just a synthetic shape."""
    real = EXPERIMENTS_DIR / "v3_exq_777_mech063_orthogonal_control_axes_dissociation.py"
    if not real.exists():
        return
    out = V.precondition_recomputability_lint(real)
    assert out is not None and "SATURATION GUARD" in out, out
    assert "baseline_entropy_headroom" in out, out


def test_r0s_partition_branch_is_warn_only():
    """INVARIANT: never blocks, like every other branch of this gate."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_PARTITION_SCOPED)
        name = f.name
    try:
        r = _run("--checks", "precondition_recomputability", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "SATURATION GUARD" in r.stdout
    finally:
        os.unlink(name)


# ---- (1f) zero-or-negative FLOOR on a manifestly non-negative `measured` ---
# (2026-08-21/22). V3-EXQ-914 mech293_ghost_branch_live: `measured` is a worst-cell
# MINIMUM of `mech293_n_ghost_admitted_mean` (a per-seed admitted-ghost COUNT, so
# >= 0 by construction) reported against `threshold: 0.0, direction: "lower"`.
# build_experiment_indexes recomputes that as `measured >= 0.0`, which is trivially
# true for a non-negative measured -- so the floor CANNOT FAIL, regardless of what
# `met` (p1_ghost_live -- an unrelated SEED FRACTION against SEED_PASS_FRACTION)
# actually says. 1 of 5 seeds admitted zero ghosts (bit-identical to the
# closed-channel control arm) and the precondition still reported met=True.
#
# Different from (b)/(d): those key on `measured` and `met` sharing (or visibly
# NOT sharing) a collection/statistic. This branch does not look at `met` at all --
# it flags the threshold/measured PAIR alone, because the floor is unfalsifiable
# independent of whatever `met` happens to compute.

_ZERO_FLOOR_NONNEG = '''
SEED_PASS_FRACTION = 2.0 / 3.0

def _worst_cell(rows, key, mode="min"):
    fn = min if mode == "min" else max
    r = fn(rows, key=lambda x: x[key])
    return float(r[key]), f"seed{r['seed']}"

def main():
    rows = [
        {"seed": 42, "n_ghost_admitted_mean": 3.1},
        {"seed": 43, "n_ghost_admitted_mean": 0.0},
        {"seed": 44, "n_ghost_admitted_mean": 1.2},
    ]
    ghost_fire_frac = sum(1 for r in rows if r["n_ghost_admitted_mean"] > 0) / len(rows)
    p1_ghost_live = ghost_fire_frac >= SEED_PASS_FRACTION
    ghost_worst, ghost_worst_cell = _worst_cell(rows, "n_ghost_admitted_mean", mode="min")
    interpretation = {
        "preconditions": [
            {
                "name": "ghost_branch_live",
                "measured": ghost_worst,
                "threshold": 0.0,
                "direction": "lower",
                "met": p1_ghost_live,
            },
        ],
    }
    return interpretation

if __name__ == "__main__":
    main()
'''


def test_r0y_zero_floor_on_nonneg_worst_cell_flagged():
    """The V3-EXQ-914 shape itself: worst-cell min of an `n_`-prefixed count field
    against a non-strict zero floor, `met` a disjoint seed fraction."""
    out = _lint(_ZERO_FLOOR_NONNEG)
    assert out is not None
    assert "non-strict FLOOR" in out, out
    assert "ghost_branch_live" in out, out


def test_r0y2_strict_comparator_is_not_flagged():
    """CONSERVATISM, and the load-bearing discriminator for this branch. A STRICT
    ">" floor recomputes as `measured > threshold`, which a non-negative measured
    CAN fail at exactly zero -- genuinely falsifiable, not the defect shape. This
    is the real V3-EXQ-819 shape (`comparator: ">"` on an abs-delta worst cell) and
    it must stay quiet."""
    src = _ZERO_FLOOR_NONNEG.replace(
        '                "direction": "lower",\n',
        '                "direction": "lower",\n'
        '                "comparator": ">",\n')
    out = _lint(src)
    assert out is None or "non-strict FLOOR" not in out, out


def test_r0y3_nonstrict_comparator_still_flags():
    """An explicit `comparator: ">="` is the same non-strict recompute as the
    absent-comparator default, so it must fire identically."""
    src = _ZERO_FLOOR_NONNEG.replace(
        '                "direction": "lower",\n',
        '                "direction": "lower",\n'
        '                "comparator": ">=",\n')
    out = _lint(src)
    assert out is not None and "non-strict FLOOR" in out, out


def test_r0y4_unhinted_field_name_is_not_flagged():
    """CONSERVATISM: without a recognised non-negative marker token (abs/n/count)
    anywhere in the reduction's string arguments, non-negativity is UNPROVEN, not
    disproven -- this branch must stay silent rather than guess. Mirrors the real
    108b real_zworld_nondegenerate case, which this branch also does not catch."""
    src = _ZERO_FLOOR_NONNEG.replace("n_ghost_admitted_mean", "vt_mean")
    out = _lint(src)
    assert out is None or "non-strict FLOOR" not in out, out


def test_r0y5_positive_threshold_is_not_flagged():
    """CONSERVATISM: a real, non-zero floor on the same non-negative quantity is a
    genuinely falsifiable check (measured can fall short of a positive threshold),
    so it is out of this branch's scope entirely -- unlike threshold<=0, it is not
    definitionally unfalsifiable."""
    src = _ZERO_FLOOR_NONNEG.replace('"threshold": 0.0,', '"threshold": 1.5,')
    out = _lint(src)
    assert out is None or "non-strict FLOOR" not in out, out


def test_r0y6_negative_threshold_also_flags():
    """threshold < 0 is an even more clearly unfalsifiable floor than threshold ==
    0 on a non-negative measured (measured >= 0 > any negative threshold, always)
    -- covers the "or <= 0" half of the brief, not just the literal zero case."""
    src = _ZERO_FLOOR_NONNEG.replace('"threshold": 0.0,', '"threshold": -1.0,')
    out = _lint(src)
    assert out is not None and "non-strict FLOOR" in out, out


def test_r0y7_ceiling_direction_is_not_a_floor():
    """CONSERVATISM: `direction: "upper"` is a ceiling, not a floor -- threshold<=0
    there means "measured must stay at or below zero", which is not the
    unfalsifiable-floor shape at all and must never fire."""
    src = _ZERO_FLOOR_NONNEG.replace('"direction": "lower",', '"direction": "upper",')
    out = _lint(src)
    assert out is None or "non-strict FLOOR" not in out, out


def test_r0y8_interval_precondition_is_not_flagged():
    """CONSERVATISM: the interval (threshold_low/threshold_high) shape has no
    single `threshold` for this branch to read at all, and must stay outside its
    scope exactly like it stays outside (b)'s undeclared-band scope."""
    out = _lint(_BAND_INTERVAL)
    assert out is None or "non-strict FLOOR" not in out, out


def test_r0y9_ternary_config_constant_idiom_flagged():
    """The V3-EXQ-720/725/725a shape: `measured` is a ternary choosing between a
    positive module-level constant and a literal 0.0 fallback -- both provably
    non-negative, so `_is_manifestly_nonneg` must see through the conditional."""
    src = '''
BINDING_STRENGTH = 0.5

def main():
    binding_active = False
    interpretation = {
        "preconditions": [
            {
                "name": "binding_active",
                "measured": float(BINDING_STRENGTH if binding_active else 0.0),
                "threshold": 0.0,
                "met": bool(binding_active),
            },
        ],
    }
    return interpretation

if __name__ == "__main__":
    main()
'''
    out = _lint(src)
    assert out is not None and "non-strict FLOOR" in out, out
    assert "binding_active" in out, out


def test_r0y10_abs_call_measured_flagged():
    """`abs(...)` is non-negative regardless of its argument -- the direct
    safe-call path, not the reduction-over-a-hinted-field path."""
    src = '''
def main():
    delta = -0.0001
    interpretation = {
        "preconditions": [
            {
                "name": "moved_at_all",
                "measured": abs(delta),
                "threshold": 0.0,
                "direction": "lower",
                "met": bool(delta != 0.0),
            },
        ],
    }
    return interpretation

if __name__ == "__main__":
    main()
'''
    out = _lint(src)
    assert out is not None and "non-strict FLOOR" in out, out


def test_r0y11_branch_f_is_warn_only():
    """INVARIANT: like every other branch of this gate, never blocks."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_ZERO_FLOOR_NONNEG)
        name = f.name
    try:
        r = _run("--checks", "precondition_recomputability", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "non-strict FLOOR" in r.stdout
    finally:
        os.unlink(name)


def test_r0y12_real_914_is_the_detection_witness():
    """Pins the confirmed miss against the real file, not just a synthetic shape."""
    real = EXPERIMENTS_DIR / "v3_exq_914_mech236_hippocampal_zgoal_channel_ablation.py"
    if not real.exists():
        return
    out = V.precondition_recomputability_lint(real)
    assert out is not None and "non-strict FLOOR" in out, out
    assert "mech293_ghost_branch_live" in out, out


# ---- (4) whole-corpus fire-count pin (2026-08-22) --------------------------
# Every branch of this gate combined, via the SHARED corpus_scan fixture in
# tests/contracts/conftest.py -- per that module's own "a new corpus-wide lint
# belongs in path_lints ... its corpus test takes corpus_scan" convention, not a
# second full-corpus enumeration.
#
# Baseline before branch (f) (2026-08-21, ree-v3 ca856563): 148. Branch (f) adds
# exactly 2 NEW files to the total --
# v3_exq_108b_mech135_inv088_zworld_disambiguation.py (encoder_trained: worst-cell
# min of a "max_abs_delta" field against a threshold-0.0 floor, `met` an unrelated
# per-seed boolean) and v3_exq_914_mech236_hippocampal_zgoal_channel_ablation.py
# (mech293_ghost_branch_live, the motivating defect). Three files with the SAME
# threshold==0 floor shape were correctly EXCLUDED -- their comparator is strict
# ">", genuinely falsifiable at measured==0, not "cannot fail":
# v3_exq_819_mech457_..., v3_exq_819a_mech457_..., v3_exq_874_mech467_...,
# v3_exq_895_mech074c_.... Four more zero-floor hits (v3_exq_689g_...,
# v3_exq_720_..., v3_exq_725_..., v3_exq_725a_...) were ALREADY in the 148
# baseline via other branches (no_direction / central_vs_worst) -- branch (f) adds
# a second reason to an existing warning there, not a new file to the total.
_PINNED_CORPUS_FIRE_COUNT = 150

# The branch-(f)-specific subset, isolated from the other five branches sharing
# the aggregate pin above. Pinned as an exact FILE SET (not just a count) since
# corpus_scan only tracks whether the whole lint fired, not which branch.
_PINNED_BRANCH_F_FILES = frozenset({
    "v3_exq_108b_mech135_inv088_zworld_disambiguation.py",
    "v3_exq_689g_mech449_go_nogo_conversion_falsifier.py",
    "v3_exq_720_coherence_nonreducibility_bound_substrate.py",
    "v3_exq_725_coherence_nonreducibility_learned_binder.py",
    "v3_exq_725a_coherence_nonreducibility_converged_binder.py",
    "v3_exq_914_mech236_hippocampal_zgoal_channel_ablation.py",
})


def test_r0z_corpus_fire_rate_is_pinned(corpus_scan):
    fired = corpus_scan["precondition_recomputability_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"precondition-recomputability fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW script is in this list, that is "
        f"very likely a genuine defect in a NEW precondition -- fix the script "
        f"rather than re-pinning. If you deliberately widened or narrowed a "
        f"branch, re-pin and say so in the commit message. "
        f"Fired: {sorted(p.name for p in fired)}")


def test_r0z2_branch_f_corpus_fire_set_is_pinned(corpus_scan):
    """Re-lints only the (small) already-fired subset from corpus_scan to isolate
    branch (f)'s own contribution -- cheap, and avoids a second full-corpus walk."""
    hits = {
        p.name for p in corpus_scan["precondition_recomputability_lint"]
        if "non-strict FLOOR" in (V.precondition_recomputability_lint(p) or "")
    }
    assert hits == _PINNED_BRANCH_F_FILES, (
        f"branch (f) corpus hit SET moved: {sorted(hits)} vs pinned "
        f"{sorted(_PINNED_BRANCH_F_FILES)}")
