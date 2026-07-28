"""Contracts for the criterion-unreachable-under---dry-run lint.

Surfaces under test:
  (1) validate_experiments.dry_run_unreachable_criterion_lint -- flags a driver whose
      `--dry-run` reduces its episode count below the absolute episode index that a
      REPORTED detector is gated on, so the detector's latch is structurally unsettable in
      any smoke and its `false` is emitted as if it were a measurement.
  (2) validate_experiments.py --checks dry_run_unreachable_criterion -- the selector, and
      the invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under
      --paths, never affects the exit code even under --strict).
  (3) The corpus fire count, pinned, with a non-vacuity guard that names the canonical
      specimen rather than trusting the number alone.

FOURTH SIBLING, AND THE ODD ONE OUT. `test_hardcoded_dry_run_lint.py`,
`test_emit_outcome_dry_run_lint.py` and `test_write_pack_dry_run_lint.py` all ask the same
question about a smoke -- "does its OUTPUT get marked as a smoke?" -- at three points of
the write path (filename, relocation, pack flag). This one asks whether the smoke's
CRITERIA are evaluable at all. A driver can thread every flag correctly, so that all three
siblings stay silent, and still report a criterion it never had the episodes to test.

THE DEFECT, and why the vacuous negative is worse than a missing one. A smoke reduces
episodes drastically (`p1_eps = 4 if dry_run else P1_TRAIN_EPISODES`). A detector gated on
an absolute mid-training index (`(ep + 1) >= MID_TRAINING_EP`, with MID_TRAINING_EP = 30)
is then arithmetically unreachable: the latch stays False not because the policy behaved
well but because the comparison cannot be satisfied. The driver reports that False. In the
manifest and on stdout it is indistinguishable from a measured negative.

CONFIRMED, NOT HYPOTHETICAL -- and the sign was INVERTED. In
`v3_exq_543i_arc062_differential_heads_falsifier.py` (gate at line 1286, MID_TRAINING_EP at
line 255, reduction at line ~1495) the smoke's gated arms had `mean_tv_distance` about 100x
BELOW the inert threshold, so EVERY arm would have flagged INERT had the gate not blocked
it. `p1_inert_gating_detected` came back false for 0/36 cells, was read as evidence of
escape, and a bistability finding built on it blocked a claim disposition for two months.
Full write-up, section 1.1:
REE_assembly/evidence/planning/dry_run_smoke_in_autopsy_audit_2026-07-28.md

THIS IS A MANUAL STEP MECHANISED. The 2026-07-28 `/failure-autopsy` Step 2a guard tells an
adjudicating session to read the dry-run reduction block against every criterion by hand.
This gate does that over the corpus instead of relying on a reader remembering to.

THE DISCRIMINATOR IS THE REPORTED LATCH, NOT THE DEAD BRANCH -- this is the part to
preserve if the rule is ever touched. Of the 409 corpus drivers whose dry loop bound the
scan can resolve, 13 contain an unsatisfiable episode-index conjunct. Two are CORRECT and
must stay silent: `v3_exq_430` line 229 gates a SLEEP CYCLE and `v3_exq_165` line 449 gates
a VALUE SHUFFLE. Both gate a scheduled ACTION, and an action a smoke skips is just a
smaller smoke -- nothing false is reported. The other 11 all gate a latch that is
initialised False, set True only in the unreachable branch, and whose value escapes into a
dict entry or an f-string. `test_duc_a_scheduled_action_without_a_latch_is_silent` and
`test_duc_an_unreported_latch_is_silent` are the two tests that hold that line.

SCOPE, and the pin. This gates NEW scripts and stays WARN-only, like all three siblings: a
landed carrier's run is complete and its pre-registered emission is not rewritten to chase
a lint. The right remedy for a landed carrier is to adjudicate the affected RESULT, which
is what the autopsy guard is for.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SPECIMEN = "v3_exq_543i_arc062_differential_heads_falsifier.py"


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
        return V.dry_run_unreachable_criterion_lint(Path(name))
    finally:
        os.unlink(name)


# The target shape, reduced to its skeleton from the 543 lineage: a module-level absolute
# threshold, a dry-run episode reduction in the CALLER, and the gate inside a helper that
# receives the reduced count as a parameter. The interprocedural hop is not incidental --
# it is how the real specimen is written, and an intraprocedural check misses it entirely.
_DEFECTIVE = '''
"""A driver whose mid-training detector cannot fire under its own smoke."""
import argparse

MID_TRAINING_EP = 30
N_TRAIN_EPISODES = 200


def _train(agent, num_episodes, dry_run=False):
    inert_gating_detected = False
    for ep in range(num_episodes):
        tv = agent.probe(ep)
        if (ep + 1) >= MID_TRAINING_EP and tv < 0.05 and not inert_gating_detected:
            inert_gating_detected = True
    return {"inert_gating_detected": bool(inert_gating_detected)}


def run_experiment(dry_run: bool = False):
    n_eps = 4 if dry_run else N_TRAIN_EPISODES
    return _train(None, n_eps, dry_run=dry_run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(run_experiment(dry_run=args.dry_run))
'''


# ---- (1) the defect fires ---------------------------------------------------------------

def test_duc_fires_on_the_canonical_shape():
    assert _lint_src(_DEFECTIVE) is not None


def test_duc_fires_on_a_direct_loop_bound():
    """No interprocedural hop: the loop reads the reduced name in the same scope."""
    src = _DEFECTIVE.replace(
        "def _train(agent, num_episodes, dry_run=False):\n"
        "    inert_gating_detected = False\n"
        "    for ep in range(num_episodes):",
        "def _train(agent, dry_run=False):\n"
        "    n_eps = 4 if dry_run else N_TRAIN_EPISODES\n"
        "    inert_gating_detected = False\n"
        "    for ep in range(n_eps):")
    src = src.replace("    return _train(None, n_eps, dry_run=dry_run)",
                      "    return _train(None, dry_run=dry_run)")
    src = src.replace("    n_eps = 4 if dry_run else N_TRAIN_EPISODES\n"
                      "    return _train", "    return _train")
    assert _lint_src(src) is not None


def test_duc_fires_on_the_block_form_reduction():
    """`if dry_run: n = 4` is the other reduction shape and must be seen too."""
    src = _DEFECTIVE.replace(
        "    n_eps = 4 if dry_run else N_TRAIN_EPISODES",
        "    n_eps = N_TRAIN_EPISODES\n"
        "    if dry_run:\n"
        "        n_eps = 4")
    assert _lint_src(src) is not None


def test_duc_fires_on_the_mirrored_comparison():
    """`MID_TRAINING_EP <= ep + 1` is the same gate written the other way round."""
    src = _DEFECTIVE.replace("if (ep + 1) >= MID_TRAINING_EP and",
                             "if MID_TRAINING_EP <= (ep + 1) and")
    assert _lint_src(src) is not None


def test_duc_fires_on_a_bare_loop_variable():
    """`ep >= THRESHOLD`, with no `+ 1` offset."""
    src = _DEFECTIVE.replace("if (ep + 1) >= MID_TRAINING_EP and", "if ep >= MID_TRAINING_EP and")
    assert _lint_src(src) is not None


def test_duc_fires_on_a_strict_comparison_at_the_boundary():
    """`ep + 1 > 4` at a bound of 4: max is 4, so `> 4` is unsatisfiable."""
    src = _DEFECTIVE.replace("MID_TRAINING_EP = 30", "MID_TRAINING_EP = 4")
    src = src.replace("if (ep + 1) >= MID_TRAINING_EP and", "if (ep + 1) > MID_TRAINING_EP and")
    assert _lint_src(src) is not None


def test_duc_names_the_line_the_bound_the_threshold_and_the_latch():
    """The message must let a reader act without re-deriving the arithmetic."""
    msg = _lint_src(_DEFECTIVE)
    assert msg is not None
    assert "tops out at 4 < 30" in msg
    assert "inert_gating_detected" in msg
    assert "ep + 1 >= MID_TRAINING_EP" in msg
    # and it must point at the fix, not merely at the defect
    assert "derive the threshold from the actual episode count" in msg


def test_duc_message_is_ascii_only():
    """CLAUDE.md: anything reaching stdout is cp1252-safe."""
    msg = _lint_src(_DEFECTIVE)
    assert msg is not None
    assert all(ord(c) < 128 for c in msg), "non-ASCII in a printed lint message"


# ---- (2) THE DISCRIMINATOR: a reported latch, not merely a dead branch -------------------

def test_duc_a_scheduled_action_without_a_latch_is_silent():
    """THE LINE THIS GATE HOLDS. A skipped scheduled ACTION is a smaller smoke, not a lie.

    This is the shape of the two corpus drivers the gate deliberately spares:
    `v3_exq_430` line 229 (`ep >= WARMUP_EPISODES` gating a sleep cycle) and `v3_exq_165`
    line 449 (`ep > 0` gating a value shuffle). Neither reports anything false; without
    this exclusion the gate would fire on reduced training schedules across the corpus,
    get suppressed, and stop working.
    """
    src = _DEFECTIVE.replace(
        "        if (ep + 1) >= MID_TRAINING_EP and tv < 0.05 and not inert_gating_detected:\n"
        "            inert_gating_detected = True",
        "        if (ep + 1) >= MID_TRAINING_EP and tv < 0.05:\n"
        "            agent.run_sleep_cycle()")
    assert _lint_src(src) is None


def test_duc_an_unreported_latch_is_silent():
    """A latch that never escapes into a manifest or a log cannot be MISREAD as a result.

    Escaping into output is the property that makes a vacuous false harmful, so it is the
    only property tested -- an earlier draft also accepted detector-ish NAMES
    (`*_detected`, `*_flag`, ...) and was dropped because on this corpus it selects exactly
    the same 11 files while adding a naming convention to game.
    """
    src = _DEFECTIVE.replace(
        '    return {"inert_gating_detected": bool(inert_gating_detected)}',
        "    return {}")
    assert _lint_src(src) is None


def test_duc_a_latch_reported_only_through_an_f_string_still_fires():
    """The run log counts as output: a reader takes a printed `false` for a measurement."""
    src = _DEFECTIVE.replace(
        '    return {"inert_gating_detected": bool(inert_gating_detected)}',
        '    print(f"  inert_gating={inert_gating_detected}")\n'
        "    return {}")
    assert _lint_src(src) is not None


def test_duc_a_latch_wrapped_in_a_call_still_counts_as_reported():
    """The specimen emits `bool(inert_gating_detected)`; a bare-Name test would miss it
    and make this gate vacuous against the very file it was written for."""
    src = _DEFECTIVE.replace('bool(inert_gating_detected)}', 'int(inert_gating_detected)}')
    assert _lint_src(src) is not None


# ---- (3) reachable gates, and the arithmetic ---------------------------------------------

def test_duc_a_satisfiable_threshold_is_silent():
    """The gate IS reachable at the reduced count -- nothing vacuous about it."""
    src = _DEFECTIVE.replace("MID_TRAINING_EP = 30", "MID_TRAINING_EP = 3")
    assert _lint_src(src) is None


def test_duc_a_strict_comparison_just_inside_the_bound_is_silent():
    """`ep + 1 > 3` at a bound of 4 is satisfiable at the last episode."""
    src = _DEFECTIVE.replace("MID_TRAINING_EP = 30", "MID_TRAINING_EP = 3")
    src = src.replace("if (ep + 1) >= MID_TRAINING_EP and", "if (ep + 1) > MID_TRAINING_EP and")
    assert _lint_src(src) is None


def test_duc_a_scaled_threshold_is_silent():
    """THE CANONICAL FIX: derive the threshold from the run's own episode count.

    The threshold is then unresolvable as an int, which is correct -- it moves with the
    reduction, so the criterion stays evaluable in a smoke.
    """
    src = _DEFECTIVE.replace("if (ep + 1) >= MID_TRAINING_EP and",
                             "if (ep + 1) >= max(1, num_episodes // 2) and")
    assert _lint_src(src) is None


def test_duc_an_or_disjunct_is_silent():
    """An unsatisfiable OR-disjunct does not make the branch unreachable."""
    src = _DEFECTIVE.replace(
        "if (ep + 1) >= MID_TRAINING_EP and tv < 0.05 and not inert_gating_detected:",
        "if (ep + 1) >= MID_TRAINING_EP or tv < 0.05:")
    assert _lint_src(src) is None


def test_duc_a_full_size_second_call_site_is_silent():
    """CONSERVATISM. Bound resolution takes the MAX over call sites, so one caller passing
    the full count means the gate IS reachable somewhere and the file stays quiet."""
    src = _DEFECTIVE.replace(
        "    return _train(None, n_eps, dry_run=dry_run)",
        "    _train(None, N_TRAIN_EPISODES, dry_run=False)\n"
        "    return _train(None, n_eps, dry_run=dry_run)")
    assert _lint_src(src) is None


def test_duc_an_unresolvable_call_site_is_silent():
    """A single call site the scan cannot resolve silences the file rather than guessing."""
    src = _DEFECTIVE.replace(
        "    return _train(None, n_eps, dry_run=dry_run)",
        "    _train(None, compute_budget(), dry_run=False)\n"
        "    return _train(None, n_eps, dry_run=dry_run)")
    assert _lint_src(src) is None


def test_duc_a_dry_guarded_gate_is_silent():
    """The message claims the gate IS evaluated in a smoke; a phase the smoke skips
    entirely is a different (and quieter) shape, so it must not be reported here."""
    src = _DEFECTIVE.replace(
        "    return _train(None, n_eps, dry_run=dry_run)",
        "    if dry_run:\n"
        "        return {}\n"
        "    return _train(None, n_eps, dry_run=dry_run)")
    assert _lint_src(src) is None


# ---- (4) the remaining preconditions -----------------------------------------------------

def test_duc_no_dry_reduction_is_silent():
    """No reduced episode count means no smoke-specific unreachability.

    This ALSO subsumes the siblings' separate `gates_work` precondition, and saying so is
    worth more than a test that pretends to check it independently. Those gates accept a
    `--dry-run` that is merely forwarded and then separately require that something gates
    on it. Here a dry-run INT REDUCTION can only be written as `... if dry_run else ...`
    or `if dry_run: n = 4`, so any file with a reduction necessarily gates work: the two
    conditions cannot be varied apart, and a flag that reduces nothing falls out here
    rather than at a distinct check.
    """
    src = _DEFECTIVE.replace("    n_eps = 4 if dry_run else N_TRAIN_EPISODES",
                             "    n_eps = N_TRAIN_EPISODES")
    assert _lint_src(src) is None


def test_duc_no_dry_run_flag_or_parameter_is_silent():
    """Never invent a smoke path for a driver that has none.

    The reduction is DELIBERATELY KEPT (renamed to a local `smoke`, which
    `_mentions_dry` still recognises) so that this reaches the flag/parameter check
    instead of short-circuiting on the reduction check that the test above covers. An
    earlier draft dropped the reduction too, and so passed without ever exercising the
    precondition it names.
    """
    src = _DEFECTIVE.replace('    ap.add_argument("--dry-run", action="store_true")\n', "")
    src = src.replace("def _train(agent, num_episodes, dry_run=False):",
                      "def _train(agent, num_episodes):")
    src = src.replace("def run_experiment(dry_run: bool = False):", "def run_experiment(smoke):")
    src = src.replace("    n_eps = 4 if dry_run else N_TRAIN_EPISODES",
                      "    n_eps = 4 if smoke else N_TRAIN_EPISODES")
    src = src.replace("    return _train(None, n_eps, dry_run=dry_run)",
                      "    return _train(None, n_eps)")
    src = src.replace("    print(run_experiment(dry_run=args.dry_run))",
                      "    print(run_experiment(False))")
    assert _lint_src(src) is None


def test_duc_the_no_flag_case_reaches_the_flag_check():
    """NON-VACUITY for the test above: it must fail at the flag/parameter precondition,
    not earlier. Without this, a future edit could quietly restore the short-circuit and
    the test would keep passing while checking nothing."""
    import ast as _ast
    src = _DEFECTIVE.replace('    ap.add_argument("--dry-run", action="store_true")\n', "")
    src = src.replace("def _train(agent, num_episodes, dry_run=False):",
                      "def _train(agent, num_episodes):")
    src = src.replace("def run_experiment(dry_run: bool = False):", "def run_experiment(smoke):")
    src = src.replace("    n_eps = 4 if dry_run else N_TRAIN_EPISODES",
                      "    n_eps = 4 if smoke else N_TRAIN_EPISODES")
    tree = _ast.parse(src)
    assert V._dry_reduced_int_bindings(tree), (
        "the no-flag fixture lost its dry-run reduction, so it now short-circuits before "
        "the flag/parameter check it is meant to exercise")


def test_duc_a_non_range_loop_is_not_matched():
    """Only `range()` loops carry a resolvable integer bound; enumerate/zip do not."""
    src = _DEFECTIVE.replace("    for ep in range(num_episodes):",
                             "    for ep in enumerate(num_episodes):")
    assert _lint_src(src) is None


def test_duc_explicit_opt_out_is_honoured():
    src = _DEFECTIVE.replace(
        '"""A driver whose mid-training detector cannot fire under its own smoke."""',
        'DRY_RUN_UNREACHABLE_CRITERION_EXEMPT = "criterion is full-run-only by design"')
    assert _lint_src(src) is None


def test_duc_syntax_error_is_silent_not_fatal():
    assert _lint_src("def broken(:\n    pass\n") is None


# ---- (5) invariants: WARN-only, selectable -----------------------------------------------

def test_duc_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, like every other branch of this gate family."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_DEFECTIVE)
        name = f.name
    try:
        r = _run("--checks", "dry_run_unreachable_criterion", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "DRY_RUN-UNREACHABLE-CRITERION" in r.stdout
    finally:
        os.unlink(name)


def test_duc_is_selectable_and_does_not_drag_in_other_checks():
    r = _run("--checks", "dry_run_unreachable_criterion", "--quiet")
    assert r.returncode == 0
    assert "dry_run-unreachable-criterion-warning(s)" in r.stdout
    assert "0 write_pack-dry_run-warning(s)" in r.stdout
    assert "0 hardcoded-dry_run-warning(s)" in r.stdout
    assert "0 emit_outcome-dry_run-warning(s)" in r.stdout


# ---- (6) the corpus pin --------------------------------------------------------------

# Pinned 2026-07-28, at the commit that introduced this gate.
#
# 11 = the whole 543 lineage, b through l, and NOTHING else. Every carrier has the identical
# `(ep + 1) >= MID_TRAINING_EP` gate with MID_TRAINING_EP = 30 against a dry bound of 4,
# latching `inert_gating_detected` into its emitted metrics. They inherited the shape from
# one another -- one authoring mistake, copied eleven times, which is the ordinary way a
# defect reaches this scale in this corpus.
#
# THE PIN'S JOB IS MOVEMENT DETECTION. A RISE means a NEW driver carries the defect: fix
# that driver (scale the gate with the run's own episode count) rather than re-pinning.
# A FALL is expected only if a carrier is deliberately repaired -- and note that repairing
# a LANDED carrier is NOT the remedy this gate asks for, since its run is complete and its
# pre-registered emission is not rewritten to chase a lint; the remedy is to adjudicate the
# affected RESULT. So a fall should be accompanied by a reason, not just a smaller number.
# A deliberate widening or narrowing of the rule re-pins and says so in the commit message.
_PINNED_CORPUS_FIRE_COUNT = 11

# Below this, the pin could pass while measuring a corpus far smaller than the real one.
# The corpus is ~1167 drivers; the same `> 500` floor the sibling gates and
# `test_corpus_scan_sharing.py` use -- loose enough never to fire on ordinary corpus churn,
# tight enough to catch a broken walk.
_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN = 500


def test_duc_corpus_fire_rate_is_pinned(corpus_scan):
    """Consumes the SHARED corpus walk (`tests/contracts/conftest.py`) rather than
    enumerating `experiments/` itself -- the standing pattern conftest's module docstring
    lays down for a new corpus-wide lint.
    """
    # NON-VACUITY, and it must come FIRST. A count assertion is true of an empty walk in
    # exactly the way that matters least; this repo hit that defect class three times on
    # 2026-07-27 (the uncollected-tests family).
    assert corpus_scan.n_glob_files > _MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN, (
        f"corpus walk covered only {corpus_scan.n_glob_files} v3_exq_* drivers, below the "
        f"{_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN} floor -- the fire-count pin would be "
        f"measuring a truncated corpus. Fix the walk (tests/contracts/conftest.py) rather "
        f"than lowering this floor.")
    fired = corpus_scan["dry_run_unreachable_criterion_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"dry_run-unreachable-criterion fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW driver is in this list, fix the driver "
        f"(scale the gate with the run's own episode count) rather than re-pinning. If you "
        f"deliberately widened or narrowed the rule, re-pin and say so in the commit "
        f"message. Fired: {sorted(p.name for p in fired)}")


def test_duc_the_pin_is_not_vacuous_the_specimen_actually_fires(corpus_scan):
    """SECOND non-vacuity guard, and the one that would survive a rule that quietly stopped
    detecting anything. A count of 11 could in principle be reached by a different set of
    files; the gate exists because of ONE driver, so that driver is named here.

    If this fails while the count above passes, the rule has drifted off its own specimen --
    do not re-pin, fix the rule.
    """
    fired = {p.name for p in corpus_scan["dry_run_unreachable_criterion_lint"]}
    assert SPECIMEN in fired, (
        f"{SPECIMEN} no longer fires -- it is the canonical carrier this gate was written "
        f"for (gate at line 1286, MID_TRAINING_EP = 30 at line 255, `p1_eps = 4 if dry_run` "
        f"at line ~1495). Fired instead: {sorted(fired)}")
    assert all(n.startswith("v3_exq_543") for n in fired), (
        f"the pinned population is the 543 lineage; a non-543 carrier appeared: "
        f"{sorted(n for n in fired if not n.startswith('v3_exq_543'))}. That is a NEW "
        f"driver with the defect -- fix the driver, do not re-pin.")
