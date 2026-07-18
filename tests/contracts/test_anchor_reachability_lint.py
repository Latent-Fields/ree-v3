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
