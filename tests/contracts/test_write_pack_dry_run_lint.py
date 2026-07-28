"""Contracts for the unthreaded-write_pack(dry_run=) lint.

Surfaces under test:
  (1) validate_experiments.write_pack_dry_run_lint -- flags a driver that already threads
      `dry_run` into `write_flat_manifest`/`emit_outcome`, reduces its work under
      `--dry-run`, and STILL calls `pack_writer.write_pack` with no threaded `dry_run=` at
      a call site a smoke run actually reaches.
  (2) validate_experiments.py --checks write_pack_dry_run -- the selector, and the
      invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under --paths,
      never affects the exit code even under --strict).
  (3) The WRITER FACT the gate rests on, asserted against experiments/pack_writer.py rather
      than trusted: `write_pack` takes `dry_run`, and it is what MARKS the pack.

THIRD SIBLING. `test_hardcoded_dry_run_lint.py` watches the WRITER
(`write_flat_manifest`, whose `dry_run` picks the `_dry_<run_id>.json` filename);
`test_emit_outcome_dry_run_lint.py` watches the SENTINEL EMITTER (`emit_outcome`, whose
`dry_run=True` RELOCATES the flat manifest out of evidence/experiments/). This one watches
the PACK WRITER. Read all three together: the flags are threaded from the same
`args.dry_run` and a driver can get any subset right.

WHY THE PACK IS THE ONE THAT SCORES. The flat manifest is not on the indexer's scoring
path at all -- `build_experiment_indexes._scan_runs` scores the RUN PACK at
`<experiment_type>/runs/<run_id>/manifest.json`. That is the whole mechanism of the
MECH-245 incident the sibling documents: the flat smoke kept its `dry_run` flag and was
inert, while the pack minted from it carried NO dry_run field and was by construction
indistinguishable from a real run.

WHAT THIS GATE IS FOR, i.e. what 2026-07-28 left open. Two repairs landed that day
(REE_assembly b84252a0ba; ree-v3 0f3bedbed4 / d5b1613615 / b281854a04) so that a smoke's
RUN PACK can finally self-identify: `sync_v3_results.build_runpack_docs` copies a truthy
top-level `dry_run` into the pack, and `ExperimentPackWriter.write_pack` gained a
`dry_run=False` keyword. But `write_pack`'s argument is OPT-IN. A driver that threads
`dry_run` into `write_flat_manifest` (so the flat file is correctly `_dry_`-prefixed) but
NOT into `write_pack` still emits an unflagged pack, which is then excluded only via
`_load_dry_run_run_ids`' cross-file carry BY RUN_ID -- the exact coupling that work set out
to dissolve, and the reason that run_id arm had to be KEPT rather than simplified away. The
coupling therefore survives PRECISELY for drivers of this shape, and this gate is what
makes that population visible instead of implicit.

THAT THE FALLBACK IS LOAD-BEARING IS NOT THEORETICAL. The same session found
`_load_dry_run_run_ids` had never globbed `*/*.json` (per-experiment-subdirectory flat
manifests), so 13 dry packs were still scored as real evidence, contributing phantom runs
to ARC-042, MECH-070, MECH-075, MECH-104, MECH-153, MECH-155, MECH-156 and MECH-231. A
silent fallback is exactly where that class of hole hides.

WHY THE "ALREADY THREADS ELSEWHERE" PRECONDITION. It is what makes this gate about the
PACK rather than about dry-run awareness in general. A driver that threads nothing is
already reported by one or both siblings; firing here too would triple-report a single
undifferentiated "the author did not think about --dry-run". Requiring a demonstrated
threading elsewhere isolates the genuinely half-threaded shape. A literal
`dry_run=False` does NOT satisfy it -- that is the `hardcoded_dry_run` gate's subject and
is evidence of the opposite of intent (`test_wpd_a_literal_false_witness_does_not_count`).

SCOPE, and the pin. This gates NEW scripts and stays WARN-only, like both siblings: a
landed carrier's run is complete and its pre-registered emission is not rewritten to chase
a lint. The corpus pin below is 0 -- see the long comment above
`_PINNED_CORPUS_FIRE_COUNT` for why that is EMPTY rather than VACUOUS, and for exactly
what would make this gate fire.
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
PACK_WRITER_PY = EXPERIMENTS_DIR / "pack_writer.py"


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
        return V.write_pack_dry_run_lint(Path(name))
    finally:
        os.unlink(name)


# The target shape: flat writer threaded (so the author demonstrably knows), pack writer
# not. This is the driver whose `_dry_` flat manifest is correct while its scored pack is
# unflagged -- the one remaining case that depends on the run_id cross-file carry.
_DEFECTIVE = '''
"""A driver whose smoke path reaches write_pack with the flag unthreaded."""
import argparse
from pathlib import Path
from experiments.pack_writer import ExperimentPackWriter, write_flat_manifest


def run_experiment(dry_run: bool = False):
    seeds = [42] if dry_run else [1, 2, 3, 4, 5]
    return {"run_id": "v3_exq_999_thing_v3", "seeds": seeds, "outcome": "PASS"}


def _write_manifest(manifest, *, dry_run: bool = False):
    return write_flat_manifest(manifest, None, dry_run=dry_run,
                               script_path=Path(__file__))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    out_path = _write_manifest(result, dry_run=bool(args.dry_run))
    writer = ExperimentPackWriter(output_root=None, repo_root=Path(__file__).parent)
    writer.write_pack(
        experiment_type="thing",
        run_id=result["run_id"],
        timestamp_utc="20260728T000000Z",
        status=result["outcome"],
        metrics_values={},
        summary_markdown="# thing",
    )
'''


# ---- (1) the defect fires --------------------------------------------------------------

def test_wpd_fires_on_the_reachable_unthreaded_shape():
    assert _lint_src(_DEFECTIVE) is not None


def test_wpd_fires_on_a_literal_false():
    """A literal `False` leaves the pack unmarked exactly as an omission does."""
    src = _DEFECTIVE.replace('        summary_markdown="# thing",\n',
                             '        summary_markdown="# thing",\n        dry_run=False,\n')
    assert _lint_src(src) is not None


def test_wpd_fires_when_the_threading_witness_is_emit_outcome():
    """The witness is EITHER sibling call, not just `write_flat_manifest`.

    A driver can reach the pack while its only demonstrated threading is on the sentinel
    emitter; that is the same half-threaded shape and must still be seen.
    """
    src = _DEFECTIVE.replace(
        "from experiments.pack_writer import ExperimentPackWriter, write_flat_manifest",
        "from experiment_protocol import emit_outcome\n"
        "from experiments.pack_writer import ExperimentPackWriter, write_flat_manifest")
    src = src.replace("    return write_flat_manifest(manifest, None, dry_run=dry_run,",
                      "    return write_flat_manifest(manifest, None,")
    src = src.replace(
        "    writer = ExperimentPackWriter(",
        '    emit_outcome(outcome=result["outcome"], manifest_path=out_path,\n'
        "                 dry_run=bool(args.dry_run))\n"
        "    writer = ExperimentPackWriter(")
    assert _lint_src(src) is not None


def test_wpd_fires_when_the_flag_lives_only_on_argparse():
    """No `dry_run` parameter anywhere -- the smoke path is carried by argparse alone."""
    src = _DEFECTIVE.replace("def run_experiment(dry_run: bool = False):",
                             "def run_experiment(args):")
    src = src.replace("    seeds = [42] if dry_run else [1, 2, 3, 4, 5]",
                      "    seeds = [42] if args.dry_run else [1, 2, 3, 4, 5]")
    src = src.replace("def _write_manifest(manifest, *, dry_run: bool = False):",
                      "def _write_manifest(manifest, args):")
    src = src.replace("    return write_flat_manifest(manifest, None, dry_run=dry_run,",
                      "    return write_flat_manifest(manifest, None, dry_run=bool(args.dry_run),")
    src = src.replace("    result = run_experiment(dry_run=args.dry_run)",
                      "    result = run_experiment(args)")
    src = src.replace("    out_path = _write_manifest(result, dry_run=bool(args.dry_run))",
                      "    out_path = _write_manifest(result, args)")
    assert _lint_src(src) is not None


def test_wpd_names_the_line_and_the_enclosing_scope():
    msg = _lint_src(_DEFECTIVE)
    assert msg is not None
    assert "line " in msg and "<module>" in msg
    # The message must point at the fix, not merely at the defect.
    assert "dry_run=bool(args.dry_run)" in msg


# ---- (2) the threading is silent, and so is a guarded call ------------------------------

def test_wpd_threaded_flag_is_silent():
    """The canonical fix: thread the script's own flag into write_pack."""
    src = _DEFECTIVE.replace(
        '        summary_markdown="# thing",\n',
        '        summary_markdown="# thing",\n        dry_run=bool(args.dry_run),\n')
    assert _lint_src(src) is None


def test_wpd_else_branch_of_if_dry_run_is_silent():
    src = _DEFECTIVE.replace(
        "    writer = ExperimentPackWriter(output_root=None, repo_root=Path(__file__).parent)",
        "    writer = ExperimentPackWriter(output_root=None, repo_root=Path(__file__).parent)\n"
        "    if args.dry_run:\n"
        '        print("dry-run complete; pack not written.")\n'
        "    else:\n"
        "        pass")
    src = src.replace("    writer.write_pack(\n", "        writer.write_pack(\n")
    src = src.replace('        experiment_type="thing",', '            experiment_type="thing",')
    src = src.replace("        run_id=result[\"run_id\"],", "            run_id=result[\"run_id\"],")
    src = src.replace('        timestamp_utc="20260728T000000Z",',
                      '            timestamp_utc="20260728T000000Z",')
    src = src.replace('        status=result["outcome"],', '            status=result["outcome"],')
    src = src.replace("        metrics_values={},", "            metrics_values={},")
    src = src.replace('        summary_markdown="# thing",', '            summary_markdown="# thing",')
    src = src.replace("    )\n", "        )\n")
    assert _lint_src(src) is None


def test_wpd_not_dry_run_guard_is_silent():
    src = _DEFECTIVE.replace("    writer.write_pack(",
                             "    if not args.dry_run:\n        writer.write_pack(")
    src = src.replace('        experiment_type="thing",', '            experiment_type="thing",')
    src = src.replace('        run_id=result["run_id"],', '            run_id=result["run_id"],')
    src = src.replace('        timestamp_utc="20260728T000000Z",',
                      '            timestamp_utc="20260728T000000Z",')
    src = src.replace('        status=result["outcome"],', '            status=result["outcome"],')
    src = src.replace("        metrics_values={},", "            metrics_values={},")
    src = src.replace('        summary_markdown="# thing",', '            summary_markdown="# thing",')
    src = src.replace("    )\n", "        )\n")
    assert _lint_src(src) is None


def test_wpd_caller_side_guard_is_silent():
    """INTERPROCEDURAL: the guard is on the CALLER, write_pack sits in the helper.

    This is the shape an intraprocedural check gets wrong -- it is why the lint runs the
    same call-graph fixpoint both sibling gates do.
    """
    src = _DEFECTIVE.replace(
        "    writer = ExperimentPackWriter(output_root=None, repo_root=Path(__file__).parent)\n"
        "    writer.write_pack(\n"
        '        experiment_type="thing",\n'
        '        run_id=result["run_id"],\n'
        '        timestamp_utc="20260728T000000Z",\n'
        '        status=result["outcome"],\n'
        "        metrics_values={},\n"
        '        summary_markdown="# thing",\n'
        "    )\n",
        "    if not args.dry_run:\n        _emit_pack(result)\n")
    src = src.replace(
        "def _write_manifest(manifest, *, dry_run: bool = False):",
        "def _emit_pack(result):\n"
        "    writer = ExperimentPackWriter(output_root=None, repo_root=Path(__file__).parent)\n"
        '    writer.write_pack(experiment_type="thing", run_id=result["run_id"],\n'
        '                      timestamp_utc="20260728T000000Z", status=result["outcome"],\n'
        '                      metrics_values={}, summary_markdown="# thing")\n'
        "\n"
        "\ndef _write_manifest(manifest, *, dry_run: bool = False):")
    assert _lint_src(src) is None


def test_wpd_early_return_before_the_pack_is_silent():
    src = _DEFECTIVE.replace(
        "    out_path = _write_manifest(result, dry_run=bool(args.dry_run))",
        "    out_path = _write_manifest(result, dry_run=bool(args.dry_run))\n"
        "    if args.dry_run:\n"
        "        raise SystemExit(0)")
    assert _lint_src(src) is None


def test_wpd_a_smoke_named_guard_is_honoured():
    """A local named `smoke` rather than `dry_run` still counts as a guard."""
    src = _DEFECTIVE.replace("    writer.write_pack(",
                             "    smoke = bool(args.dry_run)\n"
                             "    if not smoke:\n        writer.write_pack(")
    src = src.replace('        experiment_type="thing",', '            experiment_type="thing",')
    src = src.replace('        run_id=result["run_id"],', '            run_id=result["run_id"],')
    src = src.replace('        timestamp_utc="20260728T000000Z",',
                      '            timestamp_utc="20260728T000000Z",')
    src = src.replace('        status=result["outcome"],', '            status=result["outcome"],')
    src = src.replace("        metrics_values={},", "            metrics_values={},")
    src = src.replace('        summary_markdown="# thing",', '            summary_markdown="# thing",')
    src = src.replace("    )\n", "        )\n")
    assert _lint_src(src) is None


# ---- (3) the "already threads elsewhere" discriminator ----------------------------------

def test_wpd_no_threading_witness_anywhere_is_silent():
    """THE DISCRIMINATOR. A driver that threads NOTHING belongs to the sibling gates.

    Without this precondition the three gates would triple-report one undifferentiated
    "the author did not think about --dry-run" on the same file.
    """
    src = _DEFECTIVE.replace("    return write_flat_manifest(manifest, None, dry_run=dry_run,",
                             "    return write_flat_manifest(manifest, None,")
    assert _lint_src(src) is None


def test_wpd_a_literal_false_witness_does_not_count():
    """`write_flat_manifest(dry_run=False)` is evidence of the OPPOSITE of intent.

    That literal is the `hardcoded_dry_run` gate's own subject, so it must not be allowed
    to satisfy this gate's "the author already threads it" precondition.
    """
    src = _DEFECTIVE.replace("    return write_flat_manifest(manifest, None, dry_run=dry_run,",
                             "    return write_flat_manifest(manifest, None, dry_run=False,")
    assert _lint_src(src) is None


def test_wpd_a_driver_that_never_calls_write_pack_is_silent():
    """The overwhelmingly common corpus shape: flat manifest plus the converter."""
    src = _DEFECTIVE.replace(
        "    writer = ExperimentPackWriter(output_root=None, repo_root=Path(__file__).parent)\n"
        "    writer.write_pack(\n"
        '        experiment_type="thing",\n'
        '        run_id=result["run_id"],\n'
        '        timestamp_utc="20260728T000000Z",\n'
        '        status=result["outcome"],\n'
        "        metrics_values={},\n"
        '        summary_markdown="# thing",\n'
        "    )\n",
        "    print(out_path)\n")
    assert _lint_src(src) is None


# ---- (4) the remaining preconditions ----------------------------------------------------

def test_wpd_no_dry_run_flag_at_all_is_silent():
    """Never invent a smoke path: a driver with no `--dry-run` cannot leak a smoke.

    This is also the shape of `experiments/run.py`, the corpus' only real `write_pack`
    caller -- see the pin comment.
    """
    src = _DEFECTIVE.replace('    ap.add_argument("--dry-run", action="store_true")\n', "")
    src = src.replace("def run_experiment(dry_run: bool = False):", "def run_experiment():")
    src = src.replace("    seeds = [42] if dry_run else [1, 2, 3, 4, 5]",
                      "    seeds = [1, 2, 3, 4, 5]")
    src = src.replace("    result = run_experiment(dry_run=args.dry_run)",
                      "    result = run_experiment()")
    src = src.replace("    out_path = _write_manifest(result, dry_run=bool(args.dry_run))",
                      "    out_path = _write_manifest(result, dry_run=False)")
    assert _lint_src(src) is None


def test_wpd_a_flag_that_gates_nothing_is_silent():
    """A `--dry-run` that reduces no work is not a smoke path."""
    src = _DEFECTIVE.replace("    seeds = [42] if dry_run else [1, 2, 3, 4, 5]",
                             "    seeds = [1, 2, 3, 4, 5]")
    assert _lint_src(src) is None


def test_wpd_a_non_pack_call_is_not_matched():
    src = _DEFECTIVE.replace("    writer.write_pack(\n", "    writer.write_something_else(\n")
    assert _lint_src(src) is None


def test_wpd_explicit_opt_out_is_honoured():
    src = _DEFECTIVE.replace(
        '"""A driver whose smoke path reaches write_pack with the flag unthreaded."""',
        'WRITE_PACK_DRY_RUN_EXEMPT = "caller relocates the pack itself"')
    assert _lint_src(src) is None


def test_wpd_syntax_error_is_silent_not_fatal():
    assert _lint_src("def broken(:\n    pass\n") is None


# ---- (5) the writer fact the gate rests on ----------------------------------------------

def test_wpd_write_pack_takes_dry_run():
    """Asserted against pack_writer.py rather than trusted.

    If this keyword is ever removed the gate's recommended fix becomes a TypeError, so the
    fact has to be pinned where the recommendation is made.
    """
    tree = ast.parse(PACK_WRITER_PY.read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "write_pack"), None)
    assert fn is not None, "pack_writer.write_pack has moved or been renamed"
    names = [a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)]
    assert "dry_run" in names, (
        "write_pack no longer takes dry_run -- this gate recommends threading a keyword "
        "that does not exist. Landed 2026-07-28 (ree-v3 b281854a04).")


def test_wpd_dry_run_is_what_marks_the_pack():
    """`dry_run` must reach the pack MANIFEST -- a parameter that is accepted and ignored
    would make the gate's whole premise false.

    Deliberately a source-level check rather than a write: writing a pack needs a run_id
    ending `_v3`, an output root and a repo root, and the sibling gate already exercises
    the round trip end-to-end (`test_eod_pack_writer_can_mark_a_pack_dry`). What is pinned
    here is the narrower thing this gate depends on -- that the flag is CONSUMED.
    """
    src = PACK_WRITER_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "write_pack"), None)
    assert fn is not None
    uses = [n for n in ast.walk(fn)
            if isinstance(n, ast.Name) and n.id == "dry_run" and isinstance(n.ctx, ast.Load)]
    assert uses, (
        "write_pack accepts dry_run but never reads it -- the flag would be inert and this "
        "gate would be recommending a no-op.")


# ---- (6) invariants: WARN-only, selectable ----------------------------------------------

def test_wpd_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, like every other branch of this gate family."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_DEFECTIVE)
        name = f.name
    try:
        r = _run("--checks", "write_pack_dry_run", "--quiet", "--strict", "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "WRITE_PACK-DRY_RUN" in r.stdout
    finally:
        os.unlink(name)


def test_wpd_is_selectable_and_does_not_drag_in_other_checks():
    r = _run("--checks", "write_pack_dry_run", "--quiet")
    assert r.returncode == 0
    assert "write_pack-dry_run-warning(s)" in r.stdout
    assert "0 emit_outcome-dry_run-warning(s)" in r.stdout
    assert "0 hardcoded-dry_run-warning(s)" in r.stdout


# ---- (7) the corpus pin -----------------------------------------------------------------

# Pinned 2026-07-28, at the commit that introduced this gate.
#
# THE PIN IS 0 BECAUSE THE POPULATION IS EMPTY TODAY, AND THAT IS THE HONEST STATE.
# Measured over `experiments/v3_exq_*.py` at pin time: 1167 drivers, of which **ZERO
# mention `write_pack` at all**. The only non-test caller anywhere in ree-v3 is
# `experiments/run.py`, and it has no `dry_run` and therefore no smoke path, so it fails
# precondition (2) as well. Drivers reach the pack INDIRECTLY -- `write_flat_manifest` plus
# the `sync_v3_results` converter -- which is the path the converter-side repair
# (REE_assembly b84252a0ba) already gates. So unlike its two siblings, this gate was never
# a backlog: it did not start at 273 and get drawn down; it starts empty.
#
# EMPTY IS NOT VACUOUS, and the distinction is the reason this gate ships rather than being
# reported as unnecessary. The shape it detects is real, currently unguarded by anything
# else in this repo, and REACHABLE: `write_pack` is a public method of a public class that
# any driver may import, and its `dry_run=False` keyword -- added 2026-07-28 -- is opt-in,
# so the very repair that made packs self-identifying is also what creates a keyword an
# author can forget. `test_wpd_fires_on_the_reachable_unthreaded_shape` demonstrates the
# gate firing on that shape, so "0" here is a measurement of the corpus, not a property of
# the lint.
#
# WHAT WOULD MAKE IT FIRE, concretely: any driver that calls `ExperimentPackWriter(...)
# .write_pack(...)` directly under a smoke path while threading its flag into
# `write_flat_manifest`/`emit_outcome`. That is a plausible next step precisely because the
# pack is the artifact the indexer scores -- a driver wanting per-seed/latent/config
# sections in `metrics.json` has a standing reason to write the pack itself rather than
# route through the flat manifest and the converter.
#
# WHY THE GATE IS WORTH HAVING AT ZERO. Without it, such a driver's pack is excluded ONLY
# by `_load_dry_run_run_ids` carrying the flag over from the flat sibling BY RUN_ID -- the
# silent cross-file coupling the 2026-07-28 work set out to dissolve and had to KEEP for
# exactly this shape. That fallback is demonstrably where holes hide: it had never globbed
# `*/*.json` (per-experiment-subdirectory flat manifests), so 13 dry packs were scored as
# real evidence against ARC-042, MECH-070, MECH-075, MECH-104, MECH-153, MECH-155,
# MECH-156 and MECH-231. This gate makes the dependency on that fallback announce itself at
# authoring time, in this repo, before a manifest is on disk.
#
# THE PIN'S JOB IS MOVEMENT DETECTION, unchanged by being 0. A rise means a NEW driver
# carries the defect; fix that driver (thread the flag) rather than re-pinning. A
# deliberate widening or narrowing of the rule re-pins and says so in the commit message.
_PINNED_CORPUS_FIRE_COUNT = 0

# Below this, a `== 0` pin would pass vacuously on an empty walk. The corpus is ~1167
# drivers; the same `> 500` floor the sibling gate and `test_corpus_scan_sharing.py` use --
# loose enough never to fire on ordinary corpus churn, tight enough to catch a broken walk.
_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN = 500


def test_wpd_corpus_fire_rate_is_pinned(corpus_scan):
    """Consumes the SHARED corpus walk (`tests/contracts/conftest.py`) rather than
    enumerating `experiments/` itself -- the standing pattern that conftest's module
    docstring lays down for a new corpus-wide lint.
    """
    # NON-VACUITY, and it must come FIRST. The pin below is 0, and `len([]) == 0` is true
    # of a walk that scanned nothing at all, so without this the assertion would keep
    # passing while silently measuring an empty corpus. This repo hit that defect class
    # three times on 2026-07-27 (the uncollected-tests family).
    assert corpus_scan.n_glob_files > _MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN, (
        f"corpus walk covered only {corpus_scan.n_glob_files} v3_exq_* drivers, below the "
        f"{_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN} floor -- the fire-count pin below is 0 "
        f"and would pass VACUOUSLY on a walk this small. Fix the walk "
        f"(tests/contracts/conftest.py) rather than lowering this floor.")
    fired = corpus_scan["write_pack_dry_run_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"write_pack-dry_run fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW driver is in this list, fix the driver "
        f"(thread its own flag -- `write_pack(..., dry_run=bool(args.dry_run))`) rather "
        f"than re-pinning. If you deliberately widened or narrowed the rule, re-pin and "
        f"say so in the commit message. Fired: {[p.name for p in fired][:20]}")
