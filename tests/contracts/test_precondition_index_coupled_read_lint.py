"""Contract for `precondition_index_read` -- a precondition list read BY POSITION.

THE INCIDENT. V3-EXQ-993a recorded `metrics.worst_harm_action_sensitivity` from
`preconditions[0]["measured"]`. That index was CORRECT when written; a red-team
fix applied to the same driver BEFORE the run inserted a new
`control_arm_coverage_complete` check at index 0, and nothing re-pointed the
read. The landed manifest therefore carries 8.0 -- the coverage COUNT -- under
the sensitivity key, while the true value 0.11382 sat one slot along. There was
no exception, no failing test and no smoke-test signal, because a positional
read stays silently correct until someone reorders the list, and reordering a
precondition list is a ROUTINE red-team repair. Found by the fable red-team of
`failure_autopsy_V3-EXQ-993a_2026-09-05` (hygiene H1); recorded there under
`recording_defects`. The manifest is NOT edited; the driver was fixed forward.

WARN-ONLY, and that is a design decision rather than caution: a positional read
is FRAGILE, not wrong at the moment it is written, so a fire asks a human to
re-read. The standing corpus carrier is a landed driver whose run is complete,
and this corpus does not rewrite history to silence a warning.

WHAT THIS FILE PINS: the positive fixture (the exact 993a shape), the negative
fixture (the by-name lookup that replaced it), the four container spellings the
detector accepts, the shapes it deliberately does NOT accept, the exemption
marker, and a corpus non-vacuity anchor. See tests/contracts/LINT_INDEX.md.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# The corpus's standing carrier, used below as the non-vacuity anchor: a bare
# integer pin would go green if the detector stopped firing entirely.
CARRIER = "v3_exq_865_q081_zgoal_reach_preflight_scan.py"


def _lint_src(src: str, name_hint: str = "probe"):
    """Lint a synthetic script. Written to a temp dir, not experiments/.

    Deliberately NOT experiments/: this detector's whole subject is a corpus
    lint contaminating other sessions, and a temp file dropped into the shared
    checkout's experiments/ is exactly that hazard. Nothing in this lint is
    path-relative, so a temp dir is equivalent.
    """
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"v3_exq_000_{name_hint}.py"
        p.write_text(src, encoding="utf-8")
        return V.precondition_index_read_lint(p)


# --------------------------------------------------------------------------- #
# (1) the incident, reduced to its skeleton -- positive and negative
# --------------------------------------------------------------------------- #

_POSITIVE = '''
"""The 993a shape: a metric recorded from a precondition read by position."""


def build_preconditions(rows):
    return [
        {"name": "control_arm_coverage_complete", "measured": float(len(rows))},
        {"name": "harm_head_action_sensitivity_present", "measured": 0.11382},
    ]


def run_experiment():
    preconditions = build_preconditions([1, 2, 3])
    return {"metrics": {"worst_harm_action_sensitivity": preconditions[0]["measured"]}}
'''

_NEGATIVE = '''
"""The repair: the same driver looking the entry up BY NAME."""


def build_preconditions(rows):
    return [
        {"name": "control_arm_coverage_complete", "measured": float(len(rows))},
        {"name": "harm_head_action_sensitivity_present", "measured": 0.11382},
    ]


def precondition_measured(preconditions, name):
    for p in preconditions:
        if p.get("name") == name:
            return float(p["measured"])
    raise KeyError(name)


def run_experiment():
    preconditions = build_preconditions([1, 2, 3])
    return {"metrics": {"worst_harm_action_sensitivity": precondition_measured(
        preconditions, "harm_head_action_sensitivity_present")}}
'''


def test_fires_on_the_incident_shape():
    out = _lint_src(_POSITIVE)
    assert out is not None
    assert "positional read" in out
    assert 'preconditions[0]' in out


def test_silent_on_the_by_name_repair():
    """The negative control. Without this, a detector that fires on everything
    would pass the positive test and the pair would prove nothing."""
    assert _lint_src(_NEGATIVE) is None


# --------------------------------------------------------------------------- #
# (2) the four container spellings it accepts
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("expr", [
    'preconditions[0]["measured"]',                      # Name
    'self.preconditions[1]["met"]',                      # Attribute
    'interpretation["preconditions"][0]["measured"]',    # Subscript, string key
    'build_preconditions(rows)[2]["measured"]',          # Call
    'preconditions[-1]["met"]',                          # negative literal (a UnaryOp)
])
def test_container_spellings_that_fire(expr):
    assert _lint_src(f"x = {expr}\n") is not None, expr


@pytest.mark.parametrize("expr", [
    'preconditions[name]["measured"]',      # variable index -- already not positional
    'preconditions["harm"]["measured"]',    # string key
    'rows[0]["measured"]',                  # not a precondition list
    'checks[0]["measured"]',               # documented miss: name carries no signal
    'preconditions[0:2]',                   # a slice is not an element read
])
def test_shapes_that_do_not_fire(expr):
    """Including the DOCUMENTED MISS (`checks[0]`), pinned so a later widening of
    the container test is a deliberate change to this list rather than a silent
    one. Widening to every list-shaped name fires on ordinary sequence indexing
    across the corpus and makes the check worthless."""
    assert _lint_src(f"preconditions = None\nx = {expr}\n") is None, expr


def test_exemption_marker_silences_it():
    src = _POSITIVE + '\nPRECONDITION_INDEX_READ_EXEMPT = "single-element list"\n'
    assert _lint_src(src) is None


def test_a_file_never_mentioning_preconditions_is_skipped_cheaply():
    assert _lint_src('x = rows[0]["measured"]\n') is None


def test_a_syntax_error_is_not_a_finding():
    """Never fail closed on an unparseable file -- that turns a lint into a
    false blocker on someone else's in-flight edit."""
    assert _lint_src("def f(:\n    pass\n") is None


# --------------------------------------------------------------------------- #
# (3) the selector, and the WARN-ONLY invariant
# --------------------------------------------------------------------------- #

def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def test_check_is_selectable_and_reports_its_own_counter():
    p = EXPERIMENTS_DIR / CARRIER
    if not p.exists():
        pytest.skip("carrier not present")
    r = _run("--checks", "precondition_index_read", "--quiet",
             "--paths", f"experiments/{CARRIER}")
    assert r.returncode == 0
    assert "precondition-index-read-warning(s)" in r.stdout


def test_the_gate_is_warn_only_in_both_modes():
    """The family invariant: never hardens, never changes the exit code. The
    standing carrier is a landed driver whose run is complete."""
    p = EXPERIMENTS_DIR / CARRIER
    if not p.exists():
        pytest.skip("carrier not present")
    for extra in ([], ["--strict"]):
        r = _run("--checks", "precondition_index_read", "--quiet", *extra,
                 "--paths", f"experiments/{CARRIER}")
        assert r.returncode == 0, r.stdout[-2000:]


# --------------------------------------------------------------------------- #
# (4) corpus non-vacuity
# --------------------------------------------------------------------------- #

def test_the_named_carrier_still_fires():
    """A count alone goes vacuously green if the detector stops firing. Name the
    carrier instead. Measured 2026-09-07 over 1465 experiments/*.py: 2 files
    fired before V3-EXQ-993a was fixed forward, 1 after.
    """
    p = EXPERIMENTS_DIR / CARRIER
    if not p.exists():
        pytest.skip("carrier not present")
    out = V.precondition_index_read_lint(p)
    assert out is not None, f"{CARRIER} stopped firing -- the detector's shape changed"
    assert "preconditions[-1]" in out


def test_the_incident_driver_no_longer_fires():
    """V3-EXQ-993a is the repaired case; a regression there would re-open the
    exact recording defect this whole file exists for."""
    p = EXPERIMENTS_DIR / "v3_exq_993a_arc021_merged_channel_action_conditioned_harm.py"
    if not p.exists():
        pytest.skip("993a not present")
    assert V.precondition_index_read_lint(p) is None
