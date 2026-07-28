"""Contracts for the unthreaded-emit_outcome(dry_run=) lint.

Surfaces under test:
  (1) validate_experiments.emit_outcome_dry_run_lint -- flags a driver that accepts
      `--dry-run`, reduces its work under it, and STILL calls
      `experiment_protocol.emit_outcome` with a manifest_path and no threaded `dry_run=`
      at a call site a smoke run actually reaches.
  (2) validate_experiments.py --checks emit_outcome_dry_run -- the selector, and the
      invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under --paths,
      never affects the exit code even under --strict).
  (3) The WRITER FACTS the gate rests on, asserted against experiment_protocol.py rather
      than trusted: `dry_run` RELOCATES the manifest and does NOT skip or move the
      sentinel.
  (4) The DOWNSTREAM facts, asserted against REE_assembly when it is present: the indexer
      AND the flat->pack converter both refuse dry-run manifests, and pending_review does.

SIBLING GATE. This is the second half of `hardcoded_dry_run` (test_hardcoded_dry_run_lint.py).
That one watches the WRITER -- `write_flat_manifest`, whose `dry_run` picks the
`_dry_<run_id>.json` filename. This one watches the SENTINEL EMITTER -- `emit_outcome`,
whose `dry_run=True` MOVES the just-written manifest out of
REE_assembly/evidence/experiments/ into a scratch dir under the system tempdir. Read them
together; the two flags are threaded from the same `args.dry_run` and a driver can get
one right and the other wrong. V3-EXQ-650 was exactly that: it threaded the writer and not
the emitter.

DEFENCE-IN-DEPTH, LIKE ITS SIBLING -- DOWNGRADED 2026-07-28. This gate was authored as a
LIVE FIRE on the reasoning that both downstream backstops failed here: an unthreaded
`emit_outcome` is precisely the case where no relocation happens, and
`build_experiment_indexes.py` contained no `dry_run` handling at all, so only
`generate_pending_review.py` filtered smokes -- pending_review.md, the surface humans
watch, staying clean while claim_evidence.v1.json did not.

REE_assembly `cb7298c1c4` closed the downstream half, so the consequence claim is now
defence-in-depth. THE GATE ITSELF STAYS: it is the first and cheapest layer, it fires at
authoring time rather than after a manifest is already on disk, and it is the only layer
that fires in this repo at all.

THE ACTUAL MECHANISM WAS NOT THE FLAT `_dry_` MANIFEST -- worth reading before trusting
this docstring's original account, which had it wrong. The flat smoke keeps its `dry_run`
flag and is not on the scoring path at all; `_scan_runs` scores the RUN PACK at
`<experiment_type>/runs/<run_id>/manifest.json`. `sync_v3_results._is_flat_v3` minted that
pack from the smoke -- it converted canonical `..._v3` run_ids UNCONDITIONALLY, consulting
`_is_dry_run` only via `_is_evidence_grade` on the mid-string casualty branch -- and
`build_runpack_docs` emits an `experiment_pack/v1` manifest with NO dry_run field. So the
scored artifact was, by construction, indistinguishable from a real run. The repair gates
the converter on both branches and has the indexer carry the flag across from the flat
sibling by run_id.

CONFIRMED INSTANCE, MECH-245 (2026-07-28). Two 1-seed (`seeds: [0]`) `--dry-run` smokes of
V3-EXQ-825 dated 2026-07-26T15:12:07Z / 15:14:39Z were tracked in git as BOTH
`_dry_v3_exq_825_..._v3.json` (flagged, harmless) and two `status: FAIL` /
`evidence_direction: weakens` run packs (unflagged, scored). They were that claim's ENTIRE
negative evidence base -- `fail_runs: 2, pass_runs: 1, experimental_confidence: 0.571,
evidence_quadrant: plausible_unproven` -- while the one genuine run PASSED. After the
repair: `fail_runs: 0`, `experimental_confidence: 0.771`, quadrant `confirmed_established`.
The exposure was corpus-wide, not one claim: 25 dry-run manifests on disk had 25 matching
scored packs, and the rebuild dropped 22 entries (the rest already carried inactive
directions). 825's own source threads the flag today, so its fix landed after the damage --
which is why the corpus witness below asserts it CLEAN rather than firing.

The relocation demonstrably works when threaded (48 relocated `_dry_` manifests sat in the
scratch dir when this was written); the 825 pair is simply absent from it.

REACHABILITY IS AGAIN THE DISCRIMINATOR, but the grep error runs the OPPOSITE way to the
sibling's. A naive `grep -L` for the threading idioms over drivers that call emit_outcome
and take `--dry-run` returns 82 -- roughly a third of the true population -- because a
driver that threads `dry_run` into `write_flat_manifest` matches the grep while its
`emit_outcome` still does not. Measured on this corpus: 1164 drivers -> 928 call
emit_outcome -> 734 also have a REAL smoke path -> 273 reach an unthreaded call (272 after
V3-EXQ-650 was repaired alongside this gate). So a grep UNDER-counts by ~3.3x where the
sibling's over-counted by ~4x. Either way the AST fixpoint is what produces a usable
number.

SCOPE. This gates NEW scripts, and the gate stays WARN-only: a landed carrier's run is
complete and its pre-registered emission is not rewritten to chase a lint. The pin below
is a BACKLOG SIZE, not a conformance target -- a movement detector. A rise means a new
script carries the defect and the correct response is to fix that script rather than
re-pin. Threading the flag is PROVABLY INERT on the evidence path
(`dry_run=bool(args.dry_run)` equals `dry_run=False` whenever `--dry-run` is absent, which
is every real run), so a drawdown is safe -- but it is a drawdown of ~272 historical
drivers and belongs in its own batches, not here. That drawdown is now UNDER WAY; the
running record, batch by batch, is the comment above `_PINNED_CORPUS_FIRE_COUNT`.
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
PROTOCOL_PY = REPO_ROOT / "experiment_protocol.py"
# Sibling repo; absent on a bare worker checkout, so every use is existence-guarded.
ASSEMBLY = REPO_ROOT.parent / "REE_assembly"


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
        return V.emit_outcome_dry_run_lint(Path(name))
    finally:
        os.unlink(name)


# The V3-EXQ-650 shape: writer threaded, emitter not, both at module level.
_DEFECTIVE = '''
"""A driver whose smoke path reaches emit_outcome with the flag unthreaded."""
import argparse
from pathlib import Path
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest


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
    emit_outcome(
        outcome=result["outcome"],
        manifest_path=out_path,
    )
'''


# ---- (1) the defect fires --------------------------------------------------------------

def test_eod_fires_on_the_reachable_unthreaded_shape():
    assert _lint_src(_DEFECTIVE) is not None


def test_eod_fires_on_a_literal_false():
    """A literal `False` leaves the relocation off exactly as an omission does."""
    src = _DEFECTIVE.replace("        manifest_path=out_path,\n",
                             "        manifest_path=out_path,\n        dry_run=False,\n")
    assert _lint_src(src) is not None


def test_eod_fires_on_the_positional_manifest_spelling():
    """`emit_outcome(outcome, manifest_path)` -- position 1 is manifest_path."""
    src = _DEFECTIVE.replace(
        '    emit_outcome(\n        outcome=result["outcome"],\n'
        "        manifest_path=out_path,\n    )",
        '    emit_outcome(result["outcome"], out_path)')
    assert _lint_src(src) is not None


_ARGPARSE_ONLY = '''
"""Smoke path carried entirely by argparse -- no `dry_run` parameter anywhere."""
import argparse
from pathlib import Path
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [42] if args.dry_run else [1, 2, 3, 4, 5]
    manifest = {"run_id": "v3_exq_999_thing_v3", "seeds": seeds, "outcome": "PASS"}
    out_path = write_flat_manifest(manifest, None, dry_run=bool(args.dry_run),
                                   script_path=Path(__file__))
    emit_outcome(outcome=manifest["outcome"], manifest_path=out_path)
'''


def test_eod_fires_when_the_flag_lives_only_on_argparse():
    """No `dry_run` PARAMETER anywhere -- the argparse flag alone is the smoke path.

    Written as its own source rather than as a mutation of `_DEFECTIVE`, because the
    obvious mutation (rename every `dry_run` parameter to something else) also renames
    it out of the GATING expression -- `seeds = [42] if d else [...]` mentions no dry
    token, so `gates_work` goes false and the lint is correctly silent for a reason that
    has nothing to do with the argparse-only shape. Caught by this test failing on the
    first run of this file; the mutation form would have passed vacuously if the
    assertion had been `is None`.
    """
    assert _lint_src(_ARGPARSE_ONLY) is not None


def test_eod_argparse_only_shape_is_silent_when_threaded():
    """Control for the above: the same script with the flag threaded must go quiet."""
    src = _ARGPARSE_ONLY.replace(
        'emit_outcome(outcome=manifest["outcome"], manifest_path=out_path)',
        'emit_outcome(outcome=manifest["outcome"], manifest_path=out_path,\n'
        "                 dry_run=bool(args.dry_run))")
    assert _lint_src(src) is None


def test_eod_names_the_line_and_the_enclosing_scope():
    msg = _lint_src(_DEFECTIVE)
    assert "line " in msg, msg
    assert "<module>" in msg, msg


# ---- (2) reachability: the correct shapes stay silent -----------------------------------

def test_eod_threaded_flag_is_silent():
    """The fix itself must not fire -- every spelling seen in the corpus."""
    for expr in ("bool(args.dry_run)", "args.dry_run", 'result["dry_run"]', "dry"):
        src = _DEFECTIVE.replace("        manifest_path=out_path,\n",
                                 f"        manifest_path=out_path,\n"
                                 f"        dry_run={expr},\n")
        assert _lint_src(src) is None, expr


def test_eod_else_branch_of_if_dry_run_is_silent():
    """V3-EXQ-635's shape: the emitter sits in the not-dry branch, so it is honest."""
    src = _DEFECTIVE.replace(
        '    emit_outcome(\n        outcome=result["outcome"],\n'
        "        manifest_path=out_path,\n    )",
        "    if args.dry_run:\n"
        "        print('dry-run complete; no sentinel')\n"
        "    else:\n"
        '        emit_outcome(\n            outcome=result["outcome"],\n'
        "            manifest_path=out_path,\n        )")
    assert _lint_src(src) is None


def test_eod_not_dry_run_guard_is_silent():
    src = _DEFECTIVE.replace(
        '    emit_outcome(\n        outcome=result["outcome"],\n'
        "        manifest_path=out_path,\n    )",
        "    if not args.dry_run:\n"
        '        emit_outcome(\n            outcome=result["outcome"],\n'
        "            manifest_path=out_path,\n        )")
    assert _lint_src(src) is None


def test_eod_caller_side_guard_is_silent():
    """INTERPROCEDURAL: the guard is on the CALLER, emit_outcome sits in the helper.

    This is the shape an intraprocedural check gets wrong -- it is why the lint runs a
    call-graph fixpoint, exactly as the sibling gate does.
    """
    src = _DEFECTIVE.replace(
        '    emit_outcome(\n        outcome=result["outcome"],\n'
        "        manifest_path=out_path,\n    )",
        "    if not args.dry_run:\n"
        "        _finish(result, out_path)")
    src = src.replace(
        "def _write_manifest(manifest, *, dry_run: bool = False):",
        "def _finish(result, out_path):\n"
        '    emit_outcome(outcome=result["outcome"], manifest_path=out_path)\n'
        "\n"
        "\ndef _write_manifest(manifest, *, dry_run: bool = False):")
    assert _lint_src(src) is None


def test_eod_early_return_before_the_emitter_is_silent():
    src = _DEFECTIVE.replace(
        "    out_path = _write_manifest(result, dry_run=bool(args.dry_run))",
        "    out_path = _write_manifest(result, dry_run=bool(args.dry_run))\n"
        "    if args.dry_run:\n"
        "        raise SystemExit(0)")
    assert _lint_src(src) is None


def test_eod_a_smoke_named_guard_is_honoured():
    """A parameter named `smoke` rather than `dry_run` still counts as a guard."""
    src = _DEFECTIVE.replace(
        '    emit_outcome(\n        outcome=result["outcome"],\n'
        "        manifest_path=out_path,\n    )",
        "    smoke = bool(args.dry_run)\n"
        "    if not smoke:\n"
        '        emit_outcome(\n            outcome=result["outcome"],\n'
        "            manifest_path=out_path,\n        )")
    assert _lint_src(src) is None


# ---- (3) the manifest_path precondition -------------------------------------------------

def test_eod_no_manifest_path_is_silent():
    """Nothing to relocate. The sentinel alone is gitignored -- see the writer facts."""
    src = _DEFECTIVE.replace("        manifest_path=out_path,\n", "")
    assert _lint_src(src) is None


def test_eod_literal_none_manifest_path_is_silent():
    src = _DEFECTIVE.replace("        manifest_path=out_path,",
                             "        manifest_path=None,")
    assert _lint_src(src) is None


# ---- (4) the other two preconditions ----------------------------------------------------

def test_eod_no_dry_run_flag_at_all_is_silent():
    """Never invent a smoke path: a driver with no `--dry-run` cannot leak a smoke."""
    src = _DEFECTIVE.replace('    ap.add_argument("--dry-run", action="store_true")\n', "")
    src = src.replace("def run_experiment(dry_run: bool = False):", "def run_experiment():")
    src = src.replace("    seeds = [42] if dry_run else [1, 2, 3, 4, 5]",
                      "    seeds = [1, 2, 3, 4, 5]")
    src = src.replace("def _write_manifest(manifest, *, dry_run: bool = False):",
                      "def _write_manifest(manifest):")
    src = src.replace("    return write_flat_manifest(manifest, None, dry_run=dry_run,",
                      "    return write_flat_manifest(manifest, None,")
    src = src.replace("    result = run_experiment(dry_run=args.dry_run)",
                      "    result = run_experiment()")
    src = src.replace("    out_path = _write_manifest(result, dry_run=bool(args.dry_run))",
                      "    out_path = _write_manifest(result)")
    assert _lint_src(src) is None


def test_eod_a_flag_that_gates_nothing_is_silent():
    """A `--dry-run` that reduces no work is not a smoke path."""
    src = _DEFECTIVE.replace("    seeds = [42] if dry_run else [1, 2, 3, 4, 5]",
                             "    seeds = [1, 2, 3, 4, 5]")
    src = src.replace("def run_experiment(dry_run: bool = False):", "def run_experiment():")
    src = src.replace("def _write_manifest(manifest, *, dry_run: bool = False):",
                      "def _write_manifest(manifest):")
    src = src.replace("    return write_flat_manifest(manifest, None, dry_run=dry_run,",
                      "    return write_flat_manifest(manifest, None,")
    src = src.replace("    result = run_experiment(dry_run=args.dry_run)",
                      "    result = run_experiment()")
    src = src.replace("    out_path = _write_manifest(result, dry_run=bool(args.dry_run))",
                      "    out_path = _write_manifest(result)")
    assert _lint_src(src) is None


def test_eod_a_non_emitter_call_is_not_matched():
    src = _DEFECTIVE.replace("    emit_outcome(\n", "    some_other_emitter(\n")
    assert _lint_src(src) is None


def test_eod_explicit_opt_out_is_honoured():
    src = _DEFECTIVE.replace(
        '"""A driver whose smoke path reaches emit_outcome with the flag unthreaded."""',
        'EMIT_OUTCOME_DRY_RUN_EXEMPT = "caller deletes the manifest itself"')
    assert _lint_src(src) is None


def test_eod_syntax_error_is_silent_not_fatal():
    assert _lint_src("def broken(:\n    pass\n") is None


# ---- (5) the writer facts the gate rests on ---------------------------------------------

def _protocol_tree():
    return ast.parse(PROTOCOL_PY.read_text(encoding="utf-8"))


def test_eod_emit_outcome_takes_dry_run():
    fn = next(n for n in ast.walk(_protocol_tree())
              if isinstance(n, ast.FunctionDef) and n.name == "emit_outcome")
    assert any(a.arg == "dry_run" for a in fn.args.kwonlyargs + fn.args.args), \
        "emit_outcome no longer takes dry_run -- this whole gate is moot"


def test_eod_dry_run_is_what_relocates_the_manifest():
    """The ONLY consequence, asserted rather than trusted."""
    src = PROTOCOL_PY.read_text(encoding="utf-8")
    assert "_relocate_dry_run_manifest" in src, \
        "the dry-run relocation is gone -- re-justify or retire this gate"
    fn = next(n for n in ast.walk(_protocol_tree())
              if isinstance(n, ast.FunctionDef) and n.name == "emit_outcome")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None))
             == "_relocate_dry_run_manifest"]
    assert calls, "emit_outcome no longer relocates anything"
    parent = {}
    for n in ast.walk(fn):
        for c in ast.iter_child_nodes(n):
            parent[id(c)] = n
    for call in calls:
        cur, gated = parent.get(id(call)), False
        while cur is not None:
            if isinstance(cur, ast.If) and "dry_run" in ast.unparse(cur.test):
                gated = True
                break
            cur = parent.get(id(cur))
        assert gated, f"the relocation at line {call.lineno} is not dry_run-gated"


def test_eod_the_relocation_target_is_outside_the_evidence_tree():
    """It must move the manifest to a scratch dir, not merely rename it in place."""
    fn = next(n for n in ast.walk(_protocol_tree())
              if isinstance(n, ast.FunctionDef) and n.name == "_relocate_dry_run_manifest")
    body = ast.unparse(fn)
    assert "gettempdir" in body, \
        "the dry-run manifest is no longer moved to the system tempdir"
    assert "move" in body, "the relocation no longer moves the file"


def test_eod_dry_run_does_not_skip_or_move_the_sentinel():
    """The NEGATIVE writer fact, and the reason this gate needs a manifest_path.

    If `dry_run` ever started diverting the sentinel too, an unthreaded call would leak a
    sentinel as well as a manifest and the manifest_path precondition would be wrong.
    """
    fn = next(n for n in ast.walk(_protocol_tree())
              if isinstance(n, ast.FunctionDef) and n.name == "emit_outcome")
    # the sentinel path is chosen from queue_id, never from dry_run
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "out_path" for t in n.targets)]
    assert assigns, "emit_outcome no longer builds an out_path"
    for a in assigns:
        assert "dry_run" not in ast.unparse(a), \
            "the sentinel path now depends on dry_run -- revisit the manifest_path gate"
    writes = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
              and getattr(n.func, "id", getattr(n.func, "attr", None))
              == "_atomic_write_json"]
    assert writes, "emit_outcome no longer writes a sentinel"
    parent = {}
    for n in ast.walk(fn):
        for c in ast.iter_child_nodes(n):
            parent[id(c)] = n
    for w in writes:
        cur = parent.get(id(w))
        while cur is not None:
            assert not (isinstance(cur, ast.If) and "dry_run" in ast.unparse(cur.test)), \
                "the sentinel write is now dry_run-gated -- revisit this gate's rationale"
            cur = parent.get(id(cur))


# ---- (6) the downstream facts (REE_assembly, existence-guarded) -------------------------

def test_eod_the_indexer_excludes_dry_run_manifests():
    """The downstream backstop this gate used to be the sole defence for.

    Inverted 2026-07-28 (REE_assembly cb7298c1c4). It previously asserted the ABSENCE of
    any exclusion, which is what made this gate a live fire; the fix landed, so it now
    pins the exclusion's PRESENCE. If it fails, the smoke manifests reach
    claim_evidence.v1.json again and MECH-245-shaped contamination is live -- restore the
    filter rather than relaxing the test.

    `_load_dry_run_run_ids` specifically: the run pack that `_scan_runs` scores carries no
    dry_run field of its own, so a per-manifest flag check is NOT sufficient. The flag has
    to be carried across from the flat sibling by run_id, and that function is where.
    """
    indexer = ASSEMBLY / "evidence" / "experiments" / "scripts" / "build_experiment_indexes.py"
    if not indexer.exists():
        return  # sibling repo not checked out (bare worker) -- nothing to assert
    src = indexer.read_text(encoding="utf-8", errors="ignore")
    for token in ("_is_dry_run", "_load_dry_run_run_ids", "dry_run_ids"):
        assert token in src, (
            f"build_experiment_indexes.py no longer defines/uses {token} -- the dry-run "
            "exclusion has been dropped or refactored away. A --dry-run smoke's run pack "
            "is indistinguishable from a real run on its own, so without this the smokes "
            "score again (confirmed on MECH-245). See the module docstring.")


def test_eod_the_pack_converter_refuses_dry_run_manifests():
    """The UPSTREAM half -- and the one that actually minted the scored artifact.

    `sync_v3_results._is_flat_v3` converted canonical `..._v3` run_ids unconditionally,
    so a smoke became a run pack on the next governance run. Gating the indexer alone is
    not enough: the converter would keep re-minting a pack for every dry flat manifest on
    disk, including ones a cleanup had just removed.
    """
    conv = ASSEMBLY / "evidence" / "experiments" / "scripts" / "sync_v3_results.py"
    if not conv.exists():
        return
    src = conv.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_is_flat_v3"), None)
    assert fn is not None, "sync_v3_results._is_flat_v3 is gone -- re-derive this gate"
    body = ast.unparse(fn)
    assert "_is_dry_run" in body, (
        "sync_v3_results._is_flat_v3 no longer consults _is_dry_run. Dry-run smokes are "
        "being converted into scored run packs again -- this is the mechanism behind the "
        "MECH-245 contamination, not the flat _dry_ manifest.")


def test_eod_pending_review_does_exclude_dry_run_manifests():
    """The other half of the asymmetry: this is why the defect stays invisible."""
    pending = ASSEMBLY / "scripts" / "generate_pending_review.py"
    if not pending.exists():
        return
    src = pending.read_text(encoding="utf-8", errors="ignore")
    assert "_is_dry_run" in src, (
        "generate_pending_review.py no longer filters dry_run manifests -- the smoke "
        "leak is now visible on pending_review.md too, which CHANGES the triage story "
        "in this gate's docstring.")


# ---- (7) real corpus witnesses ----------------------------------------------------------

def test_eod_the_825_canonical_shape_is_clean():
    """V3-EXQ-825 threads `dry_run=result["dry_run"]` -- the shape the message points at.

    It is also the driver whose two pre-fix smokes contaminated MECH-245, so a regression
    here is the exact defect recurring on its own witness.
    """
    p = EXPERIMENTS_DIR / "v3_exq_825_mech245_generative_dominance_deafferentation.py"
    if p.exists():
        assert V.emit_outcome_dry_run_lint(p) is None


def test_eod_the_650_repair_holds():
    """V3-EXQ-650 is the driver this gate was commissioned from; it was repaired here."""
    p = EXPERIMENTS_DIR / "v3_exq_650_control_vector_collapse_diagnostic.py"
    if p.exists():
        assert V.emit_outcome_dry_run_lint(p) is None


def test_eod_the_else_gated_family_is_clean():
    """Real corpus instances of the honest `else:`-guarded shape -- never flag these."""
    for name in ("v3_exq_635_modulatory_bias_selection_authority_readiness.py",
                 "v3_exq_722_crossbank_conversion_ceiling_selection_falsifier.py"):
        p = EXPERIMENTS_DIR / name
        if p.exists():
            assert V.emit_outcome_dry_run_lint(p) is None, name


# ---- (8) invariants: WARN-only, and the backlog does not silently grow -------------------

def test_eod_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, like every other branch of this gate family."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_DEFECTIVE)
        name = f.name
    try:
        r = _run("--checks", "emit_outcome_dry_run", "--quiet", "--strict", "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "EMIT_OUTCOME-DRY_RUN" in r.stdout
    finally:
        os.unlink(name)


def test_eod_is_selectable_and_does_not_drag_in_other_checks():
    r = _run("--checks", "emit_outcome_dry_run", "--quiet")
    assert r.returncode == 0
    assert "emit_outcome-dry_run-warning(s)" in r.stdout
    assert "0 hardcoded-dry_run-warning(s)" in r.stdout
    assert "0 dead-z_goal-stream-warning(s)" in r.stdout


# Pinned 2026-07-28 against the v3_exq_*.py corpus, at the commit that introduced this
# gate -- AFTER repairing V3-EXQ-650, the driver the gate was commissioned from (the
# firing set was 273 before that one repair).
#
# THIS IS A BACKLOG SIZE, NOT A TARGET. The remaining 272 are historical drivers whose
# runs are complete; a completed run's pre-registered emission is not rewritten to chase
# a lint. The pin exists so a NEW script carrying the defect shows up as a rise, and so a
# later widening or narrowing of the rule announces its own blast radius instead of
# drifting silently. Contrast the sibling gate, whose pin now reads 0 because its backlog
# WAS drawn down -- that is a legitimate end state here too, but it is a separate piece of
# work in batches, not a reason to re-pin this number on the way past.
#
# RECONCILIATION with the grep, which errs the OPPOSITE way to the sibling's. Over
# `experiments/v3_exq_*.py`: 1164 drivers; 928 call emit_outcome; 734 of those also have a
# REAL smoke path (a --dry-run flag or dry_run parameter that some if/IfExp actually
# tests); 273 of those reach an emit_outcome call, with a non-None manifest_path, that is
# neither threaded nor dry-guarded. A `grep -l emit_outcome | grep -l -- --dry-run |
# grep -L <threading idioms>` returns 82 -- it UNDER-counts by ~3.3x, because a driver
# that threads `dry_run` into `write_flat_manifest` (or anywhere else) matches the grep
# while its `emit_outcome` still does not. The sibling's grep OVER-counted by ~4x for the
# mirror-image reason. Both numbers are unusable; the AST fixpoint is what makes the
# population knowable either way.
#
# DISTRIBUTION of the 273 at pin time, which is what makes the drawdown lower-priority
# than the sibling's was: 7 at EXQ >= 700, 187 in 500-699, 47 in 300-499, 32 below 300.
# The newest cohort -- the drivers most likely to be re-run or cloned as templates -- is
# almost clean already, because `emit_outcome(dry_run=...)` became the house style around
# the V3-EXQ-696 incident. The mass is historical.
#
# DRAWDOWN, batch 1 (2026-07-28): 272 -> 265. The whole EXQ >= 700 cohort -- 720, 725,
# 725a, 814, 824, 824a, 831 -- taken first for the reason the distribution note gives:
# the defect propagates by CLONING, and a clone lands under a new filename where nothing
# consults the source's backlog entry, so the newest drivers are worth more than their
# count. Batches continue downward through 500-699 (186 at this pin), 300-499 (47) and
# below 300 (32). Re-pin per batch; this comment is the running record.
#
# WHAT BATCH 1 ACTUALLY FOUND, because it changes how later batches should be read: NONE
# of the 7 was the plain unthreaded-and-leaking shape the gate was commissioned from.
# Two distinct benign shapes, both repaired by threading rather than exempted, per the
# V3-EXQ-635 judgement recorded in the sibling gate ("exempt when threading would be
# WRONG, not merely when the author built an alternative"):
#
#   (a) 814/824/824a/831 -- `main()` returns 0 BEFORE writing any manifest under
#       --dry-run, so the flagged call is unreachable in a smoke run and 824/824a/831's
#       literal `dry_run=False` was correct. The gate cannot see this: the guard is
#       `if result == 0`, a sentinel-value test naming no dry token, so the caller-side
#       guard heuristic does not match. Threaded anyway -- it costs nothing and makes the
#       guarantee LOCAL, surviving a future edit that lets the dry-run path fall through.
#
#   (b) 720/725/725a -- the 635 shape: the writer IS threaded (so the file is already
#       `_dry_`-prefixed), a post-hoc `out_path.unlink()` scrubs it, and
#       `manifest_for_sentinel` is already None under --dry-run. The gate cannot see the
#       None because it flows through a tuple return. Threaded via a 3-tuple
#       (`return outcome, manifest, bool(args.dry_run)`); the unlink was DELIBERATELY
#       KEPT rather than replaced by the relocation, because `_relocate_dry_run_manifest`
#       swallows every exception and returns the path unchanged, so on a move failure it
#       LEAVES the file in evidence/ where the unlink removes it. Relocation is the
#       weaker guarantee in the failure case here, not the stronger one.
#
# So batch 1 lowered the count without any of the 7 having leaked a manifest. Expect more
# of this below 700: the count is an upper bound on real exposure, and the per-driver
# judgement is the work. Verified by smoking 720 and 814 --dry-run (both exit 0, no
# manifest left in evidence/experiments/, and 720's sentinel now records
# `"dry_run": true` where it recorded false before).
_PINNED_CORPUS_FIRE_COUNT = 265


def test_eod_corpus_fire_rate_is_pinned(corpus_scan):
    """Consumes the SHARED corpus walk (`tests/contracts/conftest.py`) rather than
    enumerating `experiments/` itself -- the standing pattern that conftest's module
    docstring lays down for a new corpus-wide lint.
    """
    fired = corpus_scan["emit_outcome_dry_run_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"emit_outcome-dry_run fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW script is in this list, fix the script "
        f"(thread its own flag -- `dry_run=bool(args.dry_run)`) rather than re-pinning. "
        f"If you deliberately widened or narrowed the rule, or drew the backlog down, "
        f"re-pin and say so in the commit message. Fired: {[p.name for p in fired][:20]}")
