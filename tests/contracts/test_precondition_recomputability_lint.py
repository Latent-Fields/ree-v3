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
