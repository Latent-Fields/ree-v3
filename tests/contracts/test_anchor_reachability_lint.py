"""Contracts for the readiness-anchor reachability gate.

Surfaces under test:
  (1) validate_experiments.anchor_reachability_lint -- flags a diagnostic/baseline
      script that declares an ANCHOR-KIND readiness precondition (a readiness dict
      carrying a `control` key naming the known-positive control it must reproduce)
      and self-routes on it to a consequential label, but never calls
      experiments/_lib/readiness_anchor.assert_anchor_reachable at setup.
  (2) validate_experiments.py --checks anchor_reachability -- the selector, and the
      invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under
      --paths, never affects the exit code even under --strict).

WHY THIS GATE EXISTS. A readiness anchor's gate is scored by a hand-written
predicate. A predicate NARROWER than the state it anchors to is unmeetable by
construction: a bit-perfect replication of the control cannot clear the gate, so the
precondition reports met=false on every run forever and mislabels an
instrument-specification gap as a substrate verdict. Confirmed: V3-EXQ-778d's
`null_zero_anchor_reproduces_778c_railed_signature` scored only the SATURATION rail of
a TWO-rail degeneracy -> max 5/8 = 0.625 against a 0.75 gate, and because
`criteria_non_degenerate.C1_unpaired_null_derails = (readiness_ok and anchor_ok)` that
one statistic accounted for the entire degeneracy flag on the load-bearing criterion.

Source: REE_assembly/evidence/planning/failure_autopsy_SD-068-rem-fanout-cluster_2026-07-18.md
sec 2 (Learning 1); guard landed ree-v3 2d5f40b9b9.
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

# A diagnostic declaring an anchor-kind precondition (name+measured+threshold+control)
# that self-routes to substrate_not_ready_requeue -- the 778d shape, minus the guard.
_ANCHOR_UNGUARDED = '''
EXPERIMENT_PURPOSE = "diagnostic"
ANCHOR_GATE = 0.75

def _railed(cell):
    return cell["slope"] == 0.0

def main():
    anchor_frac = 0.5
    interpretation = {
        "preconditions": [
            {
                "name": "null_zero_anchor_reproduces_control_signature",
                "description": "the known-degenerate control must reproduce its rail",
                "measured": float(anchor_frac),
                "threshold": ANCHOR_GATE,
                "direction": "lower",
                "control": "ARM_NULL_ZERO == the prior null (known-degenerate positive control)",
                "met": bool(anchor_frac >= ANCHOR_GATE),
            },
        ],
        "self_route": "substrate_not_ready_requeue",
    }
    return interpretation

if __name__ == "__main__":
    main()
'''

# Same script, with the setup-time reachability guard in place.
_ANCHOR_GUARDED = _ANCHOR_UNGUARDED.replace(
    "def main():",
    "from experiments._lib.readiness_anchor import assert_anchor_reachable\n"
    "_REFERENCE = [{\"slope\": 0.0}] * 8\n"
    "\n"
    "def main():\n"
    "    assert_anchor_reachable(\n"
    "        anchor_name=\"null_zero_anchor_reproduces_control_signature\",\n"
    "        reference_cells=_REFERENCE, score_fn=_railed, threshold=ANCHOR_GATE)",
    1,
)


def _lint(src: str):
    """Write src to a temp .py under experiments/ and return the lint result."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        return V.anchor_reachability_lint(Path(name))
    finally:
        os.unlink(name)


# ---- (1) lint detection branches -------------------------------------------

def test_a1_anchor_without_guard_flagged():
    """Anchor-kind precondition + consequential route, no guard -> flagged."""
    issue = _lint(_ANCHOR_UNGUARDED)
    assert issue is not None
    assert "null_zero_anchor_reproduces_control_signature" in issue
    assert "assert_anchor_reachable" in issue


def test_a2_anchor_with_guard_clean():
    """The same script with assert_anchor_reachable at setup -> no issue."""
    assert _lint(_ANCHOR_GUARDED) is None


def test_a3_generic_readiness_without_control_key_not_flagged():
    """A readiness gate with NO `control` key anchors nothing reproducible.

    This is the discriminator that keeps the gate off the ~10x larger population of
    ordinary 'is the substrate trained enough' readiness preconditions: only a
    precondition naming a known-positive control can be unmeetable in the 778d way.
    """
    src = _ANCHOR_UNGUARDED.replace(
        '                "control": "ARM_NULL_ZERO == the prior null '
        '(known-degenerate positive control)",\n', "")
    # The `control` KEY is gone (the word still occurs in the name/description --
    # which is the point: the gate keys on the dict key, not on prose).
    assert '"control":' not in src
    assert _lint(src) is None


def test_a4_criterion_dict_is_not_an_anchor():
    """A dict carrying load_bearing/passed is a CRITERION, never a precondition."""
    src = _ANCHOR_UNGUARDED.replace(
        '                "met": bool(anchor_frac >= ANCHOR_GATE),\n',
        '                "met": bool(anchor_frac >= ANCHOR_GATE),\n'
        '                "load_bearing": True,\n')
    assert _lint(src) is None


def test_a5_no_consequential_route_not_flagged():
    """An anchor gating no substrate-verdict / requeue self-route is not gated."""
    src = _ANCHOR_UNGUARDED.replace('"substrate_not_ready_requeue"', '"informational"')
    assert _lint(src) is None


def test_a6_substrate_verdict_label_also_triggers():
    """The pre-existing SUBSTRATE_VERDICT_LABELS route triggers the gate too."""
    src = _ANCHOR_UNGUARDED.replace('"substrate_not_ready_requeue"', '"substrate_ceiling"')
    assert _lint(src) is not None


def test_a7_requeue_route_triggers_though_not_substrate_verdict_class():
    """REGRESSION GUARD ON THE GATE'S OWN SCOPE.

    The motivating defect (778d) does NOT route to any SUBSTRATE_VERDICT_LABELS label
    -- it routes to `substrate_not_ready_requeue`. Scoping this gate to
    SUBSTRATE_VERDICT_LABELS alone would have exempted the very run that motivated it
    (corpus check 2026-07-18: 106 of 112 anchor-kind scripts are requeue-route and NOT
    substrate-verdict-class). This test fails if someone 'tidies' the trigger back to
    the narrower label set.
    """
    src = _ANCHOR_UNGUARDED
    strings_routing = [s for s in V.SUBSTRATE_VERDICT_LABELS if s in src]
    assert not strings_routing, "fixture must not carry a SUBSTRATE_VERDICT_LABELS label"
    assert not any(src.rstrip().endswith(sfx) for sfx in V.SUBSTRATE_VERDICT_SUFFIXES)
    assert _lint(src) is not None, (
        "requeue-route anchors must be in scope; narrowing the trigger to "
        "SUBSTRATE_VERDICT_LABELS would exempt the 778d defect class")


def test_a8_exempt_marker_suppresses():
    """ANCHOR_REACHABILITY_EXEMPT opts out (predicate IS the degeneracy definition)."""
    src = 'ANCHOR_REACHABILITY_EXEMPT = "exact-equality reproduction check"\n' + _ANCHOR_UNGUARDED
    assert _lint(src) is None


def test_a9_non_diagnostic_purpose_not_gated():
    """A non-diagnostic/baseline script presses no verdict through the anchor."""
    src = _ANCHOR_UNGUARDED.replace('EXPERIMENT_PURPOSE = "diagnostic"',
                                    'EXPERIMENT_PURPOSE = "mechanism"')
    assert _lint(src) is None


def test_a10_library_file_without_main_exempt():
    """No __main__ entry point -> library helper, exempt."""
    src = _ANCHOR_UNGUARDED.replace('if __name__ == "__main__":\n    main()\n', "")
    assert _lint(src) is None


# ---- (2) real-corpus regression pair ---------------------------------------

_D778 = EXPERIMENTS_DIR / "v3_exq_sd068_rem_unpaired_null_diagnostic.py"
_H778 = EXPERIMENTS_DIR / "v3_exq_sd068_rem_unpaired_null_anchorfix_diagnostic.py"


def test_a11_fires_on_the_778d_defect():
    """The gate must catch the run that motivated it."""
    if not _D778.exists():
        return  # script retired; the synthetic fixtures still cover the shape
    issue = V.anchor_reachability_lint(_D778)
    assert issue is not None
    assert "null_zero_anchor_reproduces_778c_railed_signature" in issue


def test_a12_silent_on_the_778h_fix():
    """And must go quiet once the guard is installed."""
    if not _H778.exists():
        return
    assert V.anchor_reachability_lint(_H778) is None


# ---- (3) CLI selector + never-blocking invariant ---------------------------

def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def test_a13_checks_selector_accepts_anchor_reachability():
    r = _run("--checks", "anchor_reachability", "--quiet", "--paths", str(_H778))
    assert r.returncode == 0
    assert "anchor-reachability-warning(s)" in r.stdout


def test_a14_warn_only_under_paths_and_strict():
    """INVARIANT: this gate never blocks -- not under --paths, not under --strict.

    Unlike the arm-fingerprint / degeneracy / manifest-writer gates it stays advisory
    in BOTH modes, because whether a gate is actually reachable is not statically
    decidable (`measured` comes from live run data). The lint can only ever flag a
    missing GUARD, never a proven-unreachable gate -- so it must not fail a commit.
    """
    if not _D778.exists():
        return
    r = _run("--checks", "anchor_reachability", "--quiet", "--strict", "--paths", str(_D778))
    assert r.returncode == 0, (
        "anchor-reachability must be WARN-only even under --strict --paths; "
        f"stdout={r.stdout[-2000:]}")
    assert "REACHABILITY WARNINGS" in r.stdout


def test_a15_selector_is_surgical():
    """--checks anchor_reachability runs ONLY this gate (no conformance failures)."""
    r = _run("--checks", "anchor_reachability", "--quiet", "--paths", str(_D778))
    assert r.returncode == 0
    assert "0 non-conforming" in r.stdout


# ---- (4) the SECOND exemption category: already-ran-and-superseded ----------
#
# WHY THIS EXISTS. Before 2026-07-19 the only opt-out was ANCHOR_REACHABILITY_EXEMPT,
# framed narrowly ("the predicate IS the degeneracy definition") but implemented as a
# free-text marker that silences the lint regardless of what the reason says. That
# single category does not cover a script which HAS the defect, has ALREADY RUN, and
# whose repair belongs in a successor EXQ letter -- an in-place guard would force a
# threshold change that retroactively alters what its recorded evidence means.
#
# Reaching for EXEMPT there is the documented error: it makes an unrepaired defect
# indistinguishable from a fixed one. ANCHOR_REACHABILITY_SUPERSEDED records the status
# machine-readably and pointedly does NOT suppress -- the tests below pin BOTH halves of
# that (status readable; warning still fires).

_SUPERSEDED_SRC = (
    'ANCHOR_REACHABILITY_SUPERSEDED = "repaired in V3-EXQ-778h; already ran"\n'
    + _ANCHOR_UNGUARDED
)


def test_a16_superseded_marker_does_NOT_suppress_the_warning():
    """THE LOAD-BEARING CONTRACT. SUPERSEDED annotates; only EXEMPT silences.

    If this inverts, an already-ran defective anchor goes quiet and becomes
    indistinguishable from a repaired one -- which is exactly the 2026-07-19 mistake
    this whole category exists to prevent.
    """
    issue = _lint(_SUPERSEDED_SRC)
    assert issue is not None, (
        "ANCHOR_REACHABILITY_SUPERSEDED must NOT silence the lint -- the defect is "
        "real, it is merely not repairable in place")
    assert "null_zero_anchor_reproduces_control_signature" in issue


def test_a17_superseded_status_is_machine_readable():
    """The status is a parsed payload, not a free-text string a human must read."""
    src = _SUPERSEDED_SRC
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        sup = V.anchor_supersession_lint(Path(name))
    finally:
        os.unlink(name)
    assert sup is not None
    assert "778h" in sup["reason"]
    assert sup["lineage_ok"] is True, "reason names V3-EXQ-778h, so lineage is checkable"
    assert sup["note"] is None


def test_a18_no_marker_means_no_supersession_payload():
    """SILENT DIRECTION: an ordinary unguarded anchor declares no supersession."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_ANCHOR_UNGUARDED)
        name = f.name
    try:
        assert V.anchor_supersession_lint(Path(name)) is None
    finally:
        os.unlink(name)


def test_a19_superseded_without_a_named_successor_is_flagged():
    """A supersession claim naming no successor is unfalsifiable prose -> note set.

    Advisory only: 778d itself carries no SUPERSEDES constant despite genuinely being
    superseded, so absence is a smell, not a proof. The lint still fires either way.
    """
    src = ('ANCHOR_REACHABILITY_SUPERSEDED = "fixed elsewhere, trust me"\n'
           + _ANCHOR_UNGUARDED)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        sup = V.anchor_supersession_lint(Path(name))
    finally:
        os.unlink(name)
    assert sup is not None
    assert sup["lineage_ok"] is False
    assert sup["note"] is not None and "no successor" in sup["note"]


def test_a20_supersedes_constant_satisfies_the_cross_check():
    """The corpus's existing SUPERSEDES constant is accepted as the successor id."""
    src = ('ANCHOR_REACHABILITY_SUPERSEDED = "fixed elsewhere"\n'
           'SUPERSEDES = "V3-EXQ-778h"\n' + _ANCHOR_UNGUARDED)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        sup = V.anchor_supersession_lint(Path(name))
    finally:
        os.unlink(name)
    assert sup is not None
    assert sup["lineage"]["SUPERSEDES"] == "V3-EXQ-778h"
    assert sup["lineage_ok"] is True
    assert sup["note"] is None


def test_a21_exempt_still_silences_and_the_two_markers_are_distinct():
    """REGRESSION GUARD ON THE DISTINCTION ITSELF.

    Fails if someone 'unifies' the two markers -- in either direction. EXEMPT must keep
    silencing (it means 'no defect'); SUPERSEDED must keep warning (it means 'real
    defect, repaired elsewhere').
    """
    exempt_src = 'ANCHOR_REACHABILITY_EXEMPT = "predicate IS the definition"\n' + _ANCHOR_UNGUARDED
    assert _lint(exempt_src) is None, "EXEMPT must still silence"
    assert _lint(_SUPERSEDED_SRC) is not None, "SUPERSEDED must still warn"


# ---- (5) the lint-specimen guard -------------------------------------------

def test_a22_specimen_registry_matches_the_files_these_tests_depend_on():
    """The registry must name exactly the corpus files asserted on above.

    This is the discoverability contract: a file whose lint status these tests pin MUST
    be findable from the SCRIPT side. If someone adds a new real-corpus assertion here
    without registering the file, the guard would silently not cover it.
    """
    assert _D778.name in V._LINT_SPECIMEN_FILES
    assert _H778.name in V._LINT_SPECIMEN_FILES
    for reason in V._LINT_SPECIMEN_FILES.values():
        assert "test_anchor_reachability_lint.py" in reason, (
            "each specimen entry must name the test file that depends on it")


def test_a23_the_778d_specimen_is_unmarked_and_still_warns():
    """LIVE STATE: 778d carries neither marker, so the canary is audible.

    Pairs with a11/a14. Those assert the lint fires; this asserts WHY it still can --
    no marker has crept onto the specimen.
    """
    if not _D778.exists():
        return
    src = _D778.read_text(encoding="utf-8")
    assert "ANCHOR_REACHABILITY_EXEMPT" not in src, (
        "778d must never be exempted -- it is the gate's own regression specimen "
        "(confirmed 2026-07-19: an exemption here broke a11 + a14)")
    assert V.anchor_specimen_lint(_D778) is None, "unmarked specimen -> no guard warning"
    assert V.anchor_reachability_lint(_D778) is not None


def test_a24_marking_a_specimen_fires_the_guard():
    """FIRES DIRECTION: a marker on a registered specimen produces a loud warning."""
    if not _D778.exists():
        return
    marked = 'ANCHOR_REACHABILITY_EXEMPT = "already ran, superseded by 778h"\n' + \
        _D778.read_text(encoding="utf-8")
    # Write into a TEMP DIR under the specimen's own NAME -- the registry keys on the
    # basename, so this exercises the guard without ever touching the real corpus file
    # (these are shared multi-session checkouts; mutating a tracked file in a test that
    # might crash mid-run is how you hand the next session a dirty tree).
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / _D778.name
        tmp.write_text(marked, encoding="utf-8")
        warn = V.anchor_specimen_lint(tmp)
        assert warn is not None
        assert "LINT SPECIMEN" in warn
        assert "test_anchor_reachability_lint" in warn
        # and the exemption really would have silenced the gate -- which is the harm
        assert V.anchor_reachability_lint(tmp) is None


def test_a25_specimen_guard_silent_on_unregistered_files():
    """SILENT DIRECTION: the guard is scoped to the registry, not to all markers."""
    src = 'ANCHOR_REACHABILITY_EXEMPT = "predicate IS the definition"\n' + _ANCHOR_UNGUARDED
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        assert V.anchor_specimen_lint(Path(name)) is None
    finally:
        os.unlink(name)


def test_a26_superseded_marker_on_a_specimen_also_warns_but_stays_audible():
    """Both halves at once, on the real specimen: annotated AND still warning."""
    if not _D778.exists():
        return
    marked = 'ANCHOR_REACHABILITY_SUPERSEDED = "V3-EXQ-778h"\n' + \
        _D778.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:  # never mutate the real corpus file
        tmp = Path(td) / _D778.name
        tmp.write_text(marked, encoding="utf-8")
        assert V.anchor_specimen_lint(tmp) is not None, "annotating a specimen is worth flagging"
        assert V.anchor_reachability_lint(tmp) is not None, (
            "SUPERSEDED must leave the specimen's warning audible -- this is what makes "
            "it a safe marker to apply to 778d, unlike EXEMPT")


def test_a27_module_level_only_a_marker_inside_a_function_is_not_a_declaration():
    """A string mentioning the marker in a docstring/body is not a declaration."""
    src = _ANCHOR_UNGUARDED.replace(
        "def main():",
        "def main():\n    _note = 'see ANCHOR_REACHABILITY_SUPERSEDED for the policy'", 1)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        assert V.anchor_supersession_lint(Path(name)) is None
    finally:
        os.unlink(name)
