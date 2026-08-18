"""Contracts for the swept-axis-dry-run-slice-excludes-keyed-point lint.

Surfaces under test:
  (1) validate_experiments.dry_run_sweep_excludes_keyed_point_lint -- flags a driver
      whose `--dry-run` slices a swept axis down to a subset that drops a
      pre-registered point a load-bearing criterion is keyed on.
  (2) validate_experiments.py --checks dry_run_sweep_excludes_keyed_point -- the
      selector, and the invariant that this gate is WARN-ONLY IN BOTH MODES (never
      hardens under --paths, never affects the exit code even under --strict).
  (3) The corpus fire count, pinned, with the non-vacuity floor.

SIBLING OF dry_run_unreachable_criterion_lint, DIFFERENT SHAPE. That gate is about an
episode-INDEX loop bound shrinking past an absolute threshold -- "the smoke never
reaches episode 30". This one is about a swept-AXIS LIST being sliced down so a
specific pre-registered element falls out of the subset -- "the smoke never includes
r=2.25". Both are the same FAMILY of defect (a --dry-run smoke silently starves a
criterion, which then reports a structural absence as if it were a measurement), and
both were found by the same 2026-07-28 dry-run audit discipline, but they share no
AST pattern and are kept as separate lint functions for that reason.

CONFIRMED, NOT HYPOTHETICAL. The first smoke of
experiments/v3_exq_935_mech266_margin_normalised_cap_rule.py (2026-08-16) took
`r_values = R_SWEEP[:2] if dry_run else R_SWEEP` with
`R_SWEEP: List[float] = [1.85, 2.05, 2.25, 2.45, 2.65]` and `R_STAR = 2.25` -- the
evaluation point of the load-bearing criterion C1, keyed via
`is_r_star = abs(r - R_STAR) < 1e-9`. `R_SWEEP[:2]` = `[1.85, 2.05]` excludes R_STAR,
so `occ_at_r_star` came back None and the smoke still routed the
`cap_recalibration_is_seed_idiosyncratic` VERDICT label. Fixed the same day
(ree-v3 `cbe407e54a`) by factoring the subset into a single-source-of-truth helper
(`_r_values(dry_run)`) that always includes R_STAR, and by adding an explicit
`r_star_measured` instrument-condition flag/route. The landed carrier is therefore
already fixed and does not appear in the live corpus fire count -- the pin here is
0, and it is a forward-looking regression guard rather than a backlog list.
"""
import ast as _ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def _lint_src(src: str):
    """Lint a synthetic script written into experiments/ (so relative scoping holds)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        return V.dry_run_sweep_excludes_keyed_point_lint(Path(name))
    finally:
        os.unlink(name)


# The target shape, reduced to its skeleton from the 935 specimen: a module-level
# swept axis, a module-level pre-registered point that is a member of it, a dry-run
# slice that drops the point, and a tolerance-comparison key that selects the cell and
# reports it.
_DEFECTIVE = '''
"""A driver whose --dry-run subset drops the swept axis's pre-registered point."""
import argparse
from typing import List

SWEEP: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
POINT = 3.0


def _run_seed(dry_run):
    values = SWEEP[:2] if dry_run else SWEEP
    cell = None
    for v in values:
        is_point = abs(v - POINT) < 1e-9
        if is_point:
            cell = {"occ": 0.5}
    occ_at_point = cell["occ"] if cell is not None else None
    return {"occ_at_point": occ_at_point, "graded_at_point": bool(cell is not None)}


def run_experiment(dry_run: bool = False):
    return _run_seed(dry_run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(run_experiment(dry_run=args.dry_run))
'''


# ---- (1) the defect fires ---------------------------------------------------------------

def test_dsp_fires_on_the_canonical_shape():
    assert _lint_src(_DEFECTIVE) is not None


def test_dsp_fires_on_the_annotated_sweep_declaration():
    """The real specimen declares its axis as `R_SWEEP: List[float] = [...]`
    (an AnnAssign), which is exactly what _DEFECTIVE already uses -- this test pins
    that the annotated form is not silently missed by asserting the PLAIN-assignment
    rewrite still fires too, so a future edit cannot quietly narrow to one spelling
    without a test noticing which one broke."""
    src = _DEFECTIVE.replace("SWEEP: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0]",
                             "SWEEP = [1.0, 2.0, 3.0, 4.0, 5.0]")
    assert _lint_src(src) is not None


def test_dsp_fires_on_the_block_form_reduction():
    """`if dry_run: values = SWEEP[:2]` is the other reduction shape and must be
    seen too, not just the ternary."""
    src = _DEFECTIVE.replace(
        "    values = SWEEP[:2] if dry_run else SWEEP",
        "    values = SWEEP\n"
        "    if dry_run:\n"
        "        values = SWEEP[:2]")
    assert _lint_src(src) is not None


def test_dsp_fires_on_a_middle_slice():
    """The point can be excluded by ANY resolvable slice, not just a leading `[:n]`."""
    src = _DEFECTIVE.replace("SWEEP[:2] if dry_run else SWEEP", "SWEEP[3:5] if dry_run else SWEEP")
    assert _lint_src(src) is not None


def test_dsp_fires_on_an_equality_keyed_point():
    """The keying idiom can be a plain `==`, not only the float-tolerance `abs(...) <`
    workaround -- both are "this point selects a cell", and int sweeps commonly use
    plain equality since there is no float-precision reason to avoid it."""
    src = _DEFECTIVE.replace("SWEEP: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0]",
                             "SWEEP: List[int] = [1, 2, 3, 4, 5]")
    src = src.replace("POINT = 3.0", "POINT = 3")
    src = src.replace("is_point = abs(v - POINT) < 1e-9", "is_point = (v == POINT)")
    assert _lint_src(src) is not None


def test_dsp_names_the_line_the_expr_and_the_point():
    """The message must let a reader act without re-deriving which axis/point/line."""
    msg = _lint_src(_DEFECTIVE)
    assert msg is not None
    assert "SWEEP[:2]" in msg
    assert "POINT" in msg
    assert "SWEEP" in msg
    # and it must point at the fix, not merely at the defect
    assert "single source of truth" in msg


def test_dsp_message_is_ascii_only():
    """CLAUDE.md: anything reaching stdout is cp1252-safe."""
    msg = _lint_src(_DEFECTIVE)
    assert msg is not None
    assert all(ord(c) < 128 for c in msg), "non-ASCII in a printed lint message"


# ---- (2) negative controls -----------------------------------------------------------

def test_dsp_a_subset_that_contains_the_point_is_silent():
    """THE LINE THIS GATE HOLDS. `SWEEP[:3]` = `[1.0, 2.0, 3.0]` still contains
    POINT = 3.0 -- nothing is excluded, so the criterion is still evaluable in the
    smoke and there is nothing to warn about."""
    src = _DEFECTIVE.replace("SWEEP[:2] if dry_run else SWEEP", "SWEEP[:3] if dry_run else SWEEP")
    assert _lint_src(src) is None


def test_dsp_a_sweep_with_no_keyed_scalar_is_silent():
    """A module scalar that numerically coincides with a sweep element but is never
    used to SELECT a cell (no `==`/tolerance-`abs` comparison anywhere) is not a
    pre-registered evaluation point -- it is an unrelated constant, and firing on
    it would be a false positive with no fix a reader could act on."""
    src = _DEFECTIVE.replace(
        "    for v in values:\n"
        "        is_point = abs(v - POINT) < 1e-9\n"
        "        if is_point:\n"
        "            cell = {\"occ\": 0.5}\n",
        "    for v in values:\n"
        "        cell = {\"occ\": 0.5}\n")
    assert _lint_src(src) is None


def test_dsp_a_script_with_no_dry_run_branch_is_silent():
    """No --dry-run gate at all means no smoke-specific subset, and therefore
    nothing this lint's shape can apply to."""
    src = _DEFECTIVE.replace("    values = SWEEP[:2] if dry_run else SWEEP", "    values = SWEEP")
    src = src.replace('    ap.add_argument("--dry-run", action="store_true")\n', "")
    src = src.replace("def _run_seed(dry_run):", "def _run_seed(dry_run=False):")
    src = src.replace("def run_experiment(dry_run: bool = False):", "def run_experiment():")
    src = src.replace("    return _run_seed(dry_run)", "    return _run_seed()")
    src = src.replace("    args = ap.parse_args()\n    print(run_experiment(dry_run=args.dry_run))",
                      "    args = ap.parse_args()\n    print(run_experiment())")
    assert _lint_src(src) is None


def test_dsp_a_point_not_a_member_of_the_axis_is_silent():
    """A scalar that is NOT an element of the sweep at all cannot be "excluded by a
    slice" -- it was never present to begin with, so this is a different (unscannable)
    shape, not this gate's failure mode."""
    src = _DEFECTIVE.replace("POINT = 3.0", "POINT = 3.5")
    assert _lint_src(src) is None


def test_dsp_a_single_index_subscript_is_not_a_slice():
    """`SWEEP[0]` picks ONE element -- it is not a subset that can exclude a point by
    construction, so it must never match. This is also exactly the FIXED shape's
    building block (`[SWEEP[0], POINT]`), so a false positive here would make the
    fix itself keep re-firing."""
    src = _DEFECTIVE.replace("    values = SWEEP[:2] if dry_run else SWEEP",
                             "    values = [SWEEP[0], POINT] if dry_run else SWEEP")
    assert _lint_src(src) is None


def test_dsp_an_unresolvable_slice_bound_is_silent():
    """A slice bound that is a name/expression rather than a literal int cannot be
    evaluated statically -- silence rather than a guess, same convention as the
    sibling gate's unresolvable-bound handling."""
    src = _DEFECTIVE.replace(
        "def _run_seed(dry_run):\n"
        "    values = SWEEP[:2] if dry_run else SWEEP",
        "def _run_seed(dry_run, n=2):\n"
        "    values = SWEEP[:n] if dry_run else SWEEP")
    assert _lint_src(src) is None


def test_dsp_a_short_axis_is_silent():
    """A module list of a single numeric literal is not a swept axis (a sweep of one
    point has no subset to exclude anything from)."""
    src = _DEFECTIVE.replace("SWEEP: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0]",
                             "SWEEP: List[float] = [3.0]")
    src = src.replace("SWEEP[:2] if dry_run else SWEEP", "SWEEP[:1] if dry_run else SWEEP")
    assert _lint_src(src) is None


def test_dsp_explicit_opt_out_is_honoured():
    src = _DEFECTIVE.replace(
        '"""A driver whose --dry-run subset drops the swept axis\'s pre-registered point."""',
        'DRY_RUN_SWEPT_POINT_EXEMPT = "criterion is full-run-only by design"')
    assert _lint_src(src) is None


def test_dsp_syntax_error_is_silent_not_fatal():
    assert _lint_src("def broken(:\n    pass\n") is None


def test_dsp_no_dry_run_flag_or_parameter_reaches_the_precondition():
    """NON-VACUITY for the no-dry-run-branch test above: the flag/param precondition
    must be what silences it, not an earlier short-circuit that happens to agree.
    Directly exercises the helper the lint's first gate uses."""
    src = _DEFECTIVE.replace('    ap.add_argument("--dry-run", action="store_true")\n', "")
    tree = _ast.parse(src)
    has_flag = any(
        isinstance(n, _ast.Call) and V._call_name(n) == "add_argument"
        and any(isinstance(a, _ast.Constant) and a.value in ("--dry-run", "--dry_run")
                for a in n.args)
        for n in _ast.walk(tree))
    assert not has_flag, (
        "the no-flag fixture still declares --dry-run, so it does not exercise the "
        "flag/parameter precondition it is meant to")


# ---- (3) helper-level unit coverage (the arithmetic, isolated from the AST plumbing) -----

def test_dsp_static_slice_matches_python_semantics():
    src = "SWEEP[:2]"
    sub = _ast.parse(src, mode="eval").body
    assert V._static_slice([1.0, 2.0, 3.0, 4.0, 5.0], sub.slice) == [1.0, 2.0]


def test_dsp_static_slice_handles_negative_bounds():
    src = "SWEEP[1:-1]"
    sub = _ast.parse(src, mode="eval").body
    assert V._static_slice([1.0, 2.0, 3.0, 4.0, 5.0], sub.slice) == [2.0, 3.0, 4.0]


def test_dsp_static_slice_is_none_for_an_unresolvable_bound():
    src = "SWEEP[:n]"
    sub = _ast.parse(src, mode="eval").body
    assert V._static_slice([1.0, 2.0, 3.0], sub.slice) is None


def test_dsp_module_list_constants_reads_both_spellings():
    tree = _ast.parse(
        "PLAIN = [1.0, 2.0]\n"
        "ANNOTATED: list = [3.0, 4.0]\n"
        "NOT_A_LIST = 5.0\n"
        "TOO_SHORT = [1.0]\n")
    out = V._module_list_constants(tree)
    assert out == {"PLAIN": [1.0, 2.0], "ANNOTATED": [3.0, 4.0]}


def test_dsp_module_numeric_constants_reads_both_spellings():
    tree = _ast.parse("PLAIN = 3.0\nANNOTATED: float = 4.0\nNOT_NUMERIC = 'x'\n")
    out = V._module_numeric_constants(tree)
    assert out == {"PLAIN": 3.0, "ANNOTATED": 4.0}


# ---- (4) invariants: WARN-only, selectable -----------------------------------------------

def test_dsp_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, like every other branch of the dry-run gate family."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_DEFECTIVE)
        name = f.name
    try:
        r = _run("--checks", "dry_run_sweep_excludes_keyed_point", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "DRY_RUN-SWEEP-EXCLUDES-KEYED-POINT" in r.stdout
    finally:
        os.unlink(name)


def test_dsp_is_selectable_and_does_not_drag_in_other_checks():
    r = _run("--checks", "dry_run_sweep_excludes_keyed_point", "--quiet")
    assert r.returncode == 0
    assert "dry_run-sweep-excludes-keyed-point-warning(s)" in r.stdout
    assert "0 dry_run-unreachable-criterion-warning(s)" in r.stdout
    assert "0 write_pack-dry_run-warning(s)" in r.stdout


# ---- (5) the corpus pin --------------------------------------------------------------

# Pinned 2026-08-18, at the commit that introduced this gate.
#
# 0 -- the only known carrier of this shape (V3-EXQ-935's first smoke) was fixed the
# same day it was found (ree-v3 cbe407e54a), before this gate existed, so the live
# corpus is clean. This pin is a forward-looking regression guard, not a backlog list:
# a rise means a NEW driver introduced the shape and should be fixed the same way 935
# was (factor the dry-run subset into a single source of truth that always includes
# the keyed point), not re-pinned.
_PINNED_CORPUS_FIRE_COUNT = 0

# Same floor the sibling dry-run gates use -- loose enough never to fire on ordinary
# corpus churn, tight enough to catch a broken walk.
_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN = 500


def test_dsp_corpus_fire_rate_is_pinned(corpus_scan):
    """Consumes the SHARED corpus walk (tests/contracts/conftest.py) rather than
    enumerating experiments/ itself -- the standing pattern conftest's module
    docstring lays down for a new corpus-wide lint.
    """
    # NON-VACUITY, and it must come FIRST -- see test_dry_run_unreachable_criterion_lint.py
    # for why this repo insists on it (three uncollected-tests incidents on 2026-07-27).
    assert corpus_scan.n_glob_files > _MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN, (
        f"corpus walk covered only {corpus_scan.n_glob_files} v3_exq_* drivers, below the "
        f"{_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN} floor -- the fire-count pin would be "
        f"measuring a truncated corpus. Fix the walk (tests/contracts/conftest.py) rather "
        f"than lowering this floor.")
    fired = corpus_scan["dry_run_sweep_excludes_keyed_point_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"dry_run-sweep-excludes-keyed-point fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW driver is in this list, fix the driver "
        f"(factor the dry-run subset into a single source of truth that always includes "
        f"the keyed point) rather than re-pinning. If you deliberately widened or "
        f"narrowed the rule, re-pin and say so in the commit message. "
        f"Fired: {sorted(p.name for p in fired)}")


def test_dsp_the_pin_is_not_vacuous_the_synthetic_specimen_actually_fires():
    """SECOND non-vacuity guard, at the function level rather than the corpus level --
    the corpus pin above is 0, so it cannot itself prove the rule still detects
    anything. This re-derives the exact 935 shape (reduced to its skeleton) and
    asserts it still fires, which is what would catch a rule that quietly stopped
    detecting anything while the corpus pin kept passing at 0."""
    assert _lint_src(_DEFECTIVE) is not None
