"""Contracts for the hardcoded-dry_run lint.

Surfaces under test:
  (1) validate_experiments.hardcoded_dry_run_lint -- flags a driver that accepts
      `--dry-run`, reduces its work under it, and STILL hands
      `pack_writer.write_flat_manifest` a literal `dry_run=False` at a call site a smoke
      run actually reaches.
  (2) validate_experiments.py --checks hardcoded_dry_run -- the selector, and the
      invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under --paths,
      never affects the exit code even under --strict).
  (3) The WRITER FACTS the gate rests on, asserted against experiments/pack_writer.py
      rather than trusted: `dry_run` is what selects the `_dry_` filename prefix, and it
      is what gates the `[smoke] z_goal_stream:` report.

WHY THIS GATE EXISTS. Hardcoding the flag silently disables two independent things:

  (1) The V3-EXQ-696 relocation. `dry_run=True` makes the writer emit
      `_dry_<run_id>.json` instead of `<run_id>.json`, which is what keeps a smoke
      manifest out of `build_experiment_indexes.py`'s scoring set and off
      `pending_review.md` as an action-required FAIL. Hardcoded, a `--dry-run` smoke
      writes a real-looking 1-seed / toy-episode manifest straight into
      REE_assembly/evidence/experiments/. `emit_outcome(dry_run=True)` relocates it
      afterwards and `generate_pending_review.py` excludes dry_run-flagged manifests, so
      this is a defence-in-depth gap rather than a live fire -- the first and cheapest
      layer is simply off.
  (2) The `[smoke] z_goal_stream:` report added in ree-v3 2f6ece412f. It prints the
      z_goal liveness block ONLY under `dry_run`, deliberately: gating it there is what
      keeps it from scrolling past unread over a multi-hour run and from firing across
      the contract suite. So hardcoding is the one thing that turns the V3-EXQ-830
      early-warning off entirely. Demonstrated live while this gate was being written:
      smoking the repaired V3-EXQ-798 immediately printed
      `active_frac=0.000 writer_calls=0 -- WRITER DEFECT`, a real finding that its
      hardcoded predecessor could not have surfaced.

REACHABILITY IS THE WHOLE DISCRIMINATOR. 617 corpus drivers pass a literal `False` and
MOST ARE CORRECT: the overwhelmingly common shape returns before ever reaching the
writer (`if dry_run: print("manifest not written"); return`), which makes the literal
honest. The guard is also frequently in the CALLER, with the writer inside a helper, so
an intraprocedural check still over-fires -- hence the call-graph fixpoint. Measured:
617 literal-False sites -> 453 correctly quiet -> 164 genuine at the commit that
introduced this gate. A grep over-counts by ~4x, which is exactly why this is an AST
gate and not a grep.

SCOPE. This gates NEW scripts, and the gate stays WARN-only: a landed carrier's run is
complete and its pre-registered emission is not rewritten to chase a lint.

The fire-rate pin was introduced as a BACKLOG SIZE rather than a target of zero. As of
2026-07-27 the backlog HAS been drawn to zero (see the drawdown note at the pin), so the
number now happens to be 0 -- but its MEANING is unchanged. It is still a movement
detector, not a conformance target: a rise means a new script carries the defect, and
the correct response is to fix that script rather than re-pin. The gate deliberately
remains advisory in both modes (asserted by test_hdr_is_warn_only_in_both_modes) --
drawing the backlog down is not a licence to harden it, because the honest-literal
population (~450 drivers that guard the writer, plus ~45 with no smoke path at all) is
what makes an AST fixpoint necessary in the first place and a false positive there would
block a commit for no reason.
"""
import ast
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
        return V.hardcoded_dry_run_lint(Path(name))
    finally:
        os.unlink(name)


_DEFECTIVE = '''
"""A driver whose smoke path reaches the writer with the flag hardcoded."""
import argparse
from pathlib import Path
from experiments.pack_writer import write_flat_manifest


def run_experiment(dry_run: bool = False):
    seeds = [42] if dry_run else [1, 2, 3, 4, 5]
    return {"run_id": "v3_exq_999_thing_v3", "seeds": seeds, "outcome": "PASS"}


def main(dry_run: bool = False):
    manifest = run_experiment(dry_run=dry_run)
    return write_flat_manifest(
        manifest,
        None,
        dry_run=False,
        script_path=Path(__file__),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
'''


# ---- (1) the defect fires --------------------------------------------------------------

def test_hdr_fires_on_the_reachable_hardcoded_shape():
    assert _lint_src(_DEFECTIVE) is not None


def test_hdr_fires_when_the_flag_lives_only_on_argparse():
    """No `dry_run` parameter anywhere -- the argparse flag alone is a smoke path."""
    src = _DEFECTIVE.replace("def main(dry_run: bool = False):",
                             "def main():").replace(
        "    manifest = run_experiment(dry_run=dry_run)",
        "    manifest = run_experiment(dry_run=args.dry_run)").replace(
        "    main(dry_run=args.dry_run)", "    main()")
    assert _lint_src(src) is not None


def test_hdr_names_the_line_and_the_enclosing_function():
    msg = _lint_src(_DEFECTIVE)
    assert "in main" in msg, msg
    assert "line " in msg, msg


# ---- (2) reachability: the correct shapes stay silent -----------------------------------

def test_hdr_early_return_before_the_writer_is_silent():
    """The corpus's dominant shape -- `if dry_run: return` -- makes the literal honest."""
    src = _DEFECTIVE.replace(
        "    manifest = run_experiment(dry_run=dry_run)",
        "    manifest = run_experiment(dry_run=dry_run)\n"
        "    if dry_run:\n"
        "        print('dry-run complete; manifest not written.')\n"
        "        return None")
    assert _lint_src(src) is None


def test_hdr_not_dry_run_branch_is_silent():
    src = _DEFECTIVE.replace(
        "    return write_flat_manifest(",
        "    if not dry_run:\n"
        "        return write_flat_manifest(").replace(
        "        manifest,\n        None,\n        dry_run=False,\n"
        "        script_path=Path(__file__),\n    )",
        "            manifest,\n            None,\n            dry_run=False,\n"
        "            script_path=Path(__file__),\n        )")
    assert _lint_src(src) is None


def test_hdr_else_of_if_dry_run_is_silent():
    src = _DEFECTIVE.replace(
        "    return write_flat_manifest(",
        "    if dry_run:\n"
        "        return None\n"
        "    return write_flat_manifest(")
    assert _lint_src(src) is None


def test_hdr_caller_side_guard_is_silent():
    """INTERPROCEDURAL: the guard is on the CALLER, the writer sits inside the helper.

    This is the shape an intraprocedural check gets wrong -- it is why the lint runs a
    call-graph fixpoint. V3-EXQ-514g/h/i and V3-EXQ-812 are real instances.
    """
    src = _DEFECTIVE.replace(
        "def main(dry_run: bool = False):\n"
        "    manifest = run_experiment(dry_run=dry_run)\n"
        "    return write_flat_manifest(",
        "def write_manifest(manifest):\n"
        "    return write_flat_manifest(")
    src = src.replace(
        "    args = ap.parse_args()\n    main(dry_run=args.dry_run)",
        "    args = ap.parse_args()\n"
        "    _m = run_experiment(dry_run=args.dry_run)\n"
        "    if not args.dry_run:\n"
        "        write_manifest(_m)")
    assert _lint_src(src) is None


def test_hdr_a_smoke_named_guard_is_honoured():
    """V3-EXQ-729 names its parameter `smoke`, not `dry_run`; the guard still counts."""
    src = _DEFECTIVE.replace("def main(dry_run: bool = False):",
                             "def main(smoke: bool = False):").replace(
        "    manifest = run_experiment(dry_run=dry_run)",
        "    manifest = run_experiment(dry_run=smoke)\n"
        "    if smoke:\n"
        "        return None").replace(
        "    main(dry_run=args.dry_run)", "    main(smoke=args.dry_run)")
    assert _lint_src(src) is None


# ---- (3) the other two preconditions ----------------------------------------------------

def test_hdr_no_dry_run_flag_at_all_is_silent():
    """`dry_run=False` is HONEST in a script with no smoke path -- never invent one."""
    src = _DEFECTIVE.replace('    ap.add_argument("--dry-run", action="store_true")\n', "")
    src = src.replace("def run_experiment(dry_run: bool = False):",
                      "def run_experiment():").replace(
        "    seeds = [42] if dry_run else [1, 2, 3, 4, 5]", "    seeds = [1, 2, 3, 4, 5]")
    src = src.replace("def main(dry_run: bool = False):", "def main():").replace(
        "    manifest = run_experiment(dry_run=dry_run)", "    manifest = run_experiment()")
    src = src.replace("    main(dry_run=args.dry_run)", "    main()")
    assert _lint_src(src) is None


def test_hdr_a_flag_that_gates_nothing_is_silent():
    """A `--dry-run` only forwarded to emit_outcome is not a smoke path."""
    src = _DEFECTIVE.replace("    seeds = [42] if dry_run else [1, 2, 3, 4, 5]",
                             "    seeds = [1, 2, 3, 4, 5]")
    src = src.replace("def run_experiment(dry_run: bool = False):", "def run_experiment():")
    src = src.replace("    manifest = run_experiment(dry_run=dry_run)",
                      "    manifest = run_experiment()")
    src = src.replace("def main(dry_run: bool = False):", "def main():")
    src = src.replace("    main(dry_run=args.dry_run)", "    main()")
    assert _lint_src(src) is None


def test_hdr_threaded_flag_is_silent():
    """The fix itself must not fire -- the canonical V3-EXQ-722 shape."""
    for expr in ("dry_run", "bool(args.dry_run)", "args.dry_run"):
        src = _DEFECTIVE.replace("dry_run=False,", f"dry_run={expr},")
        assert _lint_src(src) is None, expr


def test_hdr_omitted_kwarg_is_silent():
    """No `dry_run=` at all is a different defect (the writer defaults False); not ours."""
    src = _DEFECTIVE.replace("        dry_run=False,\n", "")
    assert _lint_src(src) is None


def test_hdr_a_non_writer_call_is_not_matched():
    src = _DEFECTIVE.replace("write_flat_manifest(", "some_other_writer(")
    assert _lint_src(src) is None


def test_hdr_explicit_opt_out_is_honoured():
    src = _DEFECTIVE.replace(
        '"""A driver whose smoke path reaches the writer with the flag hardcoded."""',
        'HARDCODED_DRY_RUN_EXEMPT = "caller already relocates the output"')
    assert _lint_src(src) is None


def test_hdr_syntax_error_is_silent_not_fatal():
    assert _lint_src("def broken(:\n    pass\n") is None


# ---- (4) the writer facts the gate rests on ---------------------------------------------

def _pack_writer_tree():
    return ast.parse((EXPERIMENTS_DIR / "pack_writer.py").read_text(encoding="utf-8"))


def test_hdr_dry_run_is_what_selects_the_dry_filename_prefix():
    """Asserted against pack_writer, not trusted: consequence (1) is real."""
    src = (EXPERIMENTS_DIR / "pack_writer.py").read_text(encoding="utf-8")
    assert "_dry_" in src, "pack_writer no longer emits a _dry_ prefix"
    fn = next(n for n in ast.walk(_pack_writer_tree())
              if isinstance(n, ast.FunctionDef) and n.name == "write_flat_manifest")
    assert any(a.arg == "dry_run" for a in fn.args.kwonlyargs + fn.args.args), \
        "write_flat_manifest no longer takes dry_run -- this whole gate is moot"
    # the prefix must be chosen under a dry_run test, not unconditionally
    assert any(
        isinstance(n, (ast.If, ast.IfExp)) and "dry_run" in ast.unparse(n.test)
        and "_dry_" in ast.unparse(n)
        for n in ast.walk(fn)), "the _dry_ prefix is no longer gated on dry_run"


def test_hdr_the_smoke_z_goal_report_is_gated_on_dry_run():
    """Asserted against pack_writer, not trusted: consequence (2) is real."""
    src = (EXPERIMENTS_DIR / "pack_writer.py").read_text(encoding="utf-8")
    assert "[smoke] z_goal_stream" in src, \
        "the smoke z_goal report is gone -- re-justify or retire this gate"
    tree = _pack_writer_tree()
    printers = [n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
                and "[smoke] z_goal_stream" in ast.unparse(n)]
    assert printers, "no function carries the smoke z_goal print"
    # every call to the printer must sit under a dry_run test
    names = {f.name for f in printers}
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) in names]
    assert calls, "the smoke z_goal printer is never called"
    parent = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent[id(c)] = n
    for call in calls:
        cur, gated = parent.get(id(call)), False
        while cur is not None:
            if isinstance(cur, ast.If) and "dry_run" in ast.unparse(cur.test):
                gated = True
                break
            cur = parent.get(id(cur))
        assert gated, f"smoke z_goal print at line {call.lineno} is not dry_run-gated"


# ---- (5) real corpus witnesses ----------------------------------------------------------

def test_hdr_the_722_canonical_shape_is_clean():
    """V3-EXQ-722 is the driver the fix message points authors at."""
    p = EXPERIMENTS_DIR / "v3_exq_722_crossbank_conversion_ceiling_selection_falsifier.py"
    if p.exists():
        assert V.hardcoded_dry_run_lint(p) is None


def test_hdr_the_repaired_drivers_are_clean():
    """The 23 live-subset drivers repaired alongside this gate must stay repaired."""
    repaired = [
        "v3_exq_701_inv050_mel_measurability_diagnostic.py",
        "v3_exq_712_distributional_world_forward_heads.py",
        "v3_exq_716_sd063_conditional_uncertainty_validation.py",
        "v3_exq_718_sdmelconsumer_adaptive_cadence_validation.py",
        "v3_exq_726_sd033e_frontopolar_decommit_gapa.py",
        "v3_exq_730_q080a_effort_harm_coupling.py",
        "v3_exq_798_sdmelproducer_graded_nonconverging_world.py",
    ]
    for name in repaired:
        p = EXPERIMENTS_DIR / name
        if p.exists():
            assert V.hardcoded_dry_run_lint(p) is None, name


def test_hdr_the_early_return_family_is_clean():
    """Real corpus instances of the honest shape -- these must never be flagged."""
    for name in ("v3_exq_812_mech295_cue_authority_sd054.py",
                 "v3_exq_514g_sd049_bg_gating_wider_seeds_stepharness.py",
                 "v3_exq_729_mech268_dacc_saturation_liveloop.py",
                 "v3_exq_740_inv064_maturational_sequence_e3_bounded.py"):
        p = EXPERIMENTS_DIR / name
        if p.exists():
            assert V.hardcoded_dry_run_lint(p) is None, name


# ---- (6) invariants: WARN-only, and the backlog does not silently grow -------------------

def test_hdr_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, like every other branch of this gate family."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_DEFECTIVE)
        name = f.name
    try:
        r = _run("--checks", "hardcoded_dry_run", "--quiet", "--strict", "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "HARDCODED-DRY_RUN" in r.stdout
    finally:
        os.unlink(name)


def test_hdr_is_selectable_and_does_not_drag_in_other_checks():
    r = _run("--checks", "hardcoded_dry_run", "--quiet")
    assert r.returncode == 0
    assert "hardcoded-dry_run-warning(s)" in r.stdout
    assert "0 dead-z_goal-stream-warning(s)" in r.stdout


# Pinned 2026-07-27 against the v3_exq_*.py corpus, at the commit that introduced this
# gate -- AFTER repairing the 23-driver live subset (the firing set was 164 before those
# repairs). This is a BACKLOG SIZE, not a target: the remaining 141 are historical
# drivers whose runs are complete, and a completed run's pre-registered emission is not
# rewritten to chase a lint. The pin exists so a NEW script carrying the defect shows up
# as a rise, and so a later widening or narrowing of the rule announces its own blast
# radius instead of drifting silently.
#
# RECONCILIATION with the raw grep that commissioned this gate. `grep -l write_flat_manifest
# | grep -l dry_run=False` over experiments/v3_exq_*.py returned 617 BEFORE the repairs and
# returned 594 at the commit that added this gate (the 23 repaired drivers no longer carry a
# literal at all); the batch drawdown below takes it to 465. That residue is the HONEST
# population -- 45 drivers with no smoke path at all plus ~420 that guard the writer so a
# smoke never reaches it. It is not a remaining backlog and must not be swept.
# Splitting the 594 as it stood then: 45 have no smoke path whatsoever, so the literal is
# HONEST -- do not invent a flag for them; 408 guard the writer so a smoke run never reaches
# it, overwhelmingly `if dry_run: return` in the same function or in the CALLER; 141 fire. The ~4x
# gap between the grep and the true population is the entire reason this is an AST gate with
# a call-graph fixpoint rather than a grep, and the reason a blind sed would have been wrong.
#
# PRIORITISATION of the repaired 23: every firing driver with an EXQ number >= 700, i.e.
# the drivers most likely to be re-run or cloned as templates. It includes V3-EXQ-798, the
# only firing driver already wired for the z_goal liveness block -- smoking it after the
# repair immediately printed `active_frac=0.000 writer_calls=0 -- WRITER DEFECT`, which is
# the V3-EXQ-830 failure mode caught for free by the print the hardcoding had suppressed.
#
# BACKLOG DRAWDOWN (2026-07-27). The residual 141 are being repaired in batches, newest EXQ
# first. This is NOT "rewriting a completed run's emission to chase a lint": threading the
# flag is PROVABLY INERT on the evidence path, because `dry_run=bool(args.dry_run)` is
# identical to `dry_run=False` whenever `--dry-run` is absent, which is every real run. It
# only changes behaviour under `--dry-run`, which by definition produced no evidence. What
# made the drawdown worth doing rather than deferring to "fix it when re-run" is that the
# defect propagates by CLONING, not by re-running: a clone lands under a NEW filename, so
# nothing at clone time ever consults the source file's backlog entry. Measured at the time
# of the drawdown: 100 of the 141 firing drivers sit in an EXQ family that has already
# spawned lettered iterations, and 19 of the 23 drivers repaired above were themselves
# lettered clones (701/a/b/c, 733/a/a/b/c, ...) -- i.e. the observed repairs are mostly the
# defect having already propagated.
#   batch 1 = 34 drivers, EXQ 598..697
#   batch 2 = 34 drivers, EXQ 517..597
#   batch 3 = 34 drivers, EXQ 328..514
#   batch 4 = 31 drivers, EXQ 115..327
#   batch 5 =  8 drivers, the shape-C residue -- writer in a helper, so a real signature
#              change rather than a substitution. 6 of them (610c/d/e/f, 655, 656) turned
#              out to be a THIRD mechanical shape, `dry = "--dry-run" in sys.argv` at
#              module level -> dry_run=dry. Only 635 and 650 needed a threaded parameter.
#
# ONE JUDGEMENT CALL worth recording, because it is the case this comment tells you to
# watch for. V3-EXQ-635's caller already handled its dry-run output -- it os.remove()s the
# manifest and prints "(dry-run manifest scrubbed)" -- which is the shape that normally
# earns HARDCODED_DRY_RUN_EXEMPT. It was THREADED anyway, on two grounds: (1) the scrub
# runs AFTER the write, so a kill between the two, or the OSError the caller silently
# swallows, leaves a real-looking <run_id>.json sitting in the scoring set -- the prefix
# prevents the file ever existing there, which the scrub cannot; (2) a scrub does nothing
# for the second consequence, the suppressed [smoke] z_goal_stream: report. Verified by
# smoking it: the manifest is now _dry_-prefixed AND the scrub still fires AND the z_goal
# report prints. Exempt when threading would be WRONG, not merely when the author built
# an alternative.
#
# THE PIN IS 0, SO IT NEEDS A NON-VACUITY GUARD (added 2026-07-28, mirroring the sibling
# gate `test_emit_outcome_dry_run_lint.py` at 715189f246). `len([]) == 0` is true of a walk
# that scanned NOTHING, so without the floor below this movement detector silently becomes a
# no-op the moment the shared walk breaks or is re-scoped. A NONZERO pin cannot fail this
# way -- an empty walk yields 0 fires, which is != the pin, so it fails loudly -- which is
# why only the 0-pinned gates carry this guard and the others deliberately do not.
# `test_corpus_scan_sharing.py` already asserts `n_glob_files > 500` globally, but that
# backstop lives in a DIFFERENT FILE: it does not run when this test is run selectively, and
# a re-scoping that shrank only THIS lint's file set would not trip it. The 2026-07-27
# uncollected-tests family was exactly the lesson that a check must be reachable from the
# thing it protects.
_PINNED_CORPUS_FIRE_COUNT = 0

# ~1166 v3_exq_* drivers at the time of writing; the same `> 500` floor
# `test_corpus_scan_sharing.py` uses -- loose enough never to fire on corpus churn, tight
# enough to catch a walk that broke.
_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN = 500


def test_hdr_corpus_fire_rate_is_pinned(corpus_scan):
    """Consumes the SHARED corpus walk (`tests/contracts/conftest.py`) rather than
    enumerating `experiments/` itself.

    The file set is unchanged -- `corpus_scan` applies this lint to exactly
    `sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))`, in the same order, via the same
    `V.hardcoded_dry_run_lint(p) is not None` test, with the lint function itself
    untouched. What changes is only that the parse is shared with the five other
    corpus-wide lints instead of this test paying for a sixth full walk and parse
    of ~1160 drivers. Verified full-corpus differentially against an independent
    uncached scan when this was wired in (identical fire sets, identical file
    counts); the pin below is the standing evidence -- it would move if the
    sharing altered what is scanned.
    """
    # NON-VACUITY, and it must come FIRST. The pin below is 0, and `len([]) == 0` is true
    # of a walk that scanned nothing at all, so without this the assertion would keep
    # passing while silently measuring an empty corpus.
    assert corpus_scan.n_glob_files > _MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN, (
        f"corpus walk covered only {corpus_scan.n_glob_files} v3_exq_* drivers, below the "
        f"{_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN} floor -- the fire-count pin below is 0 "
        f"and would pass VACUOUSLY on a walk this small. Fix the walk "
        f"(tests/contracts/conftest.py) rather than lowering this floor.")
    fired = corpus_scan["hardcoded_dry_run_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"hardcoded-dry_run fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW script is in this list, fix the script "
        f"(thread its own flag -- `dry_run=bool(args.dry_run)`) rather than re-pinning. "
        f"If you deliberately widened or narrowed the rule, re-pin and say so in the "
        f"commit message. Fired: {[p.name for p in fired]}")
