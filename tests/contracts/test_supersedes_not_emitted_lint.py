"""Contracts for the supersedes-not-emitted gate
(chip-20260922-supersedes-not-emitted-validator).

Surfaces under test:
  (1) validate_experiments.supersedes_not_emitted_lint -- flags a manifest-
      writing driver that declares a module-level `SUPERSEDES = "V3-EXQ-..."`
      constant but never writes `"supersedes": SUPERSEDES` into the manifest.
  (2) validate_experiments.py --checks supersedes_not_emitted -- the selector,
      and the invariant that this gate is WARN-ONLY IN BOTH MODES (never
      hardens under --paths, never affects the exit code even under --strict).

WHY THIS GATE EXISTS. CLAUDE.md "EXQ Versioning and Supersession Policy"
requires `supersedes` on BOTH the queue entry and the manifest, so a
superseded predecessor's evidence stops weighting claim confidence. A driver
that sets the module-level `SUPERSEDES` constant (the corpus's own lineage
convention, `_ANCHOR_LINEAGE_NAMES`) but never threads it into the manifest
dict lands with `supersedes: null` -- confirmed on a real landed manifest,
v3_exq_1043a's, which reads null despite its driver declaring
`SUPERSEDES = "V3-EXQ-1043"`. This is a recording-completeness gap, not a
live scoring bug (`supersedes` alone routes nothing to governance).

THE TEST HALF (positive/negative controls pulled from real git history, not
hand-built): V3-EXQ-1043a (REE_Working chip-era corpus) is the exact
incident this gate exists for; V3-EXQ-1043b is the reference implementation
that emits correctly. Both are asserted against their REAL on-disk source in
experiments/, not a synthetic reconstruction, so a regression in either
file's actual content is what this test would catch.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

POSITIVE = "v3_exq_1043a_mech537_communication_subspace_permutation_null.py"
NEGATIVE = "v3_exq_1043b_mech537_communication_subspace_randrank.py"


# A manifest-writing driver that declares SUPERSEDES but never emits it.
DECLARES_BUT_DROPS = '''
import json

SUPERSEDES = "V3-EXQ-047i"

def run():
    manifest = {
        "run_id": "v3_exq_047j_x_20260909T000000Z_v3",
        "evidence_direction": "non_contributory",
        "queue_id": "V3-EXQ-047j",
        "outcome": "FAIL",
    }
    with open("out.json", "w") as f:
        json.dump(manifest, f)

if __name__ == "__main__":
    run()
'''

# Same driver, fixed: threads SUPERSEDES into the manifest.
EMITS_CORRECTLY = DECLARES_BUT_DROPS.replace(
    '"queue_id": "V3-EXQ-047j",',
    '"queue_id": "V3-EXQ-047j",\n        "supersedes": SUPERSEDES,')

# No SUPERSEDES declared at all -- the ordinary, non-lineage case. Must never
# fire; this is the vast majority of the corpus.
NO_LINEAGE_DECLARED = '''
import json

def run():
    manifest = {
        "run_id": "v3_exq_100_x_20260909T000000Z_v3",
        "evidence_direction": "supports",
        "queue_id": "V3-EXQ-100",
    }
    with open("out.json", "w") as f:
        json.dump(manifest, f)

if __name__ == "__main__":
    run()
'''

# SUPERSEDES declared but explicitly None -- must not fire (nothing to emit).
SUPERSEDES_IS_NONE = DECLARES_BUT_DROPS.replace(
    'SUPERSEDES = "V3-EXQ-047i"', "SUPERSEDES = None")

EXEMPTED = DECLARES_BUT_DROPS.replace(
    "import json",
    'import json\n\nSUPERSEDES_EMISSION_EXEMPT = "predecessor evidence already superseded via flag"')

# No manifest identity -- SUPERSEDES declared but this is not a result-manifest
# writer (e.g. a shared _lib helper importing another driver's constant).
NOT_A_MANIFEST_WRITER = '''
SUPERSEDES = "V3-EXQ-047i"

def helper():
    return {"note": SUPERSEDES}
'''

LIBRARY_HELPER = '''
SUPERSEDES = "V3-EXQ-047i"

def build():
    return {"run_id": "x", "evidence_direction": "supports"}
'''


def _write(tmpdir, name, src):
    p = Path(tmpdir) / name
    p.write_text(src, encoding="utf-8")
    return p


def test_declares_but_drops_warns():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_047j.py", DECLARES_BUT_DROPS)
        msg = V.supersedes_not_emitted_lint(p)
        assert msg is not None
        assert "SUPERSEDES" in msg
        assert "V3-EXQ-047i" in msg


def test_emits_correctly_discharges():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_047j_fixed.py", EMITS_CORRECTLY)
        assert V.supersedes_not_emitted_lint(p) is None


def test_no_lineage_declared_never_fires():
    """The ordinary case -- no SUPERSEDES constant at all. Must not fire;
    this is nearly the entire corpus and firing here would be the exact
    "gate that fires on ordinary work" failure CLAUDE.md warns against."""
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_100.py", NO_LINEAGE_DECLARED)
        assert V.supersedes_not_emitted_lint(p) is None


def test_supersedes_explicitly_none_does_not_fire():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_047j_none.py", SUPERSEDES_IS_NONE)
        assert V.supersedes_not_emitted_lint(p) is None


def test_exempt_marker_discharges():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_exempt.py", EXEMPTED)
        assert V.supersedes_not_emitted_lint(p) is None


def test_non_manifest_writer_not_gated():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "lineage_helper.py", NOT_A_MANIFEST_WRITER)
        assert V.supersedes_not_emitted_lint(p) is None


def test_library_helper_not_gated():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "_lib_helper.py", LIBRARY_HELPER)
        assert V.supersedes_not_emitted_lint(p) is None


def test_unparseable_file_is_silent():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "broken.py", "def f(:\n")
        assert V.supersedes_not_emitted_lint(p) is None


def test_selector_is_registered():
    assert "supersedes_not_emitted" in V.CHECK_NAMES


def test_warn_only_under_paths():
    """WARN-ONLY IN BOTH MODES -- must NEVER harden under --paths or --strict,
    matching flat_scalar_readout / criteria_threshold's posture. A finding
    here is a recording-completeness gap, not a conformance failure the
    author should be blocked from committing over."""
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_047j.py", DECLARES_BUT_DROPS)
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "validate_experiments.py"),
             "--checks", "supersedes_not_emitted", "--strict", "--paths", str(p)],
            capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "SUPERSEDES-NOT-EMITTED WARNINGS" in proc.stdout


def test_lint_output_is_ascii():
    """CLAUDE.md: anything reaching stdout must be ASCII (cp1252 terminals)."""
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_047j.py", DECLARES_BUT_DROPS)
        msg = V.supersedes_not_emitted_lint(p)
        msg.encode("ascii")


# --------------------------------------------------------------------------------
# REAL-CORPUS POSITIVE/NEGATIVE CONTROLS -- the actual incident, not a
# reconstruction. Skips gracefully if either driver is ever renamed/retired,
# rather than failing on an unrelated corpus change.
# --------------------------------------------------------------------------------

def test_the_1043a_incident_is_flagged_on_its_real_source():
    path = EXPERIMENTS_DIR / POSITIVE
    if not path.is_file():
        return  # renamed/retired -- not this test's business to fail on
    msg = V.supersedes_not_emitted_lint(path)
    assert msg is not None, (
        "%s declares SUPERSEDES and writes a manifest -- if this now passes, "
        "either the incident was fixed (good -- update POSITIVE to a live "
        "backlog member) or the lint regressed (bad)" % POSITIVE)


def test_the_1043b_reference_implementation_is_compliant():
    path = EXPERIMENTS_DIR / NEGATIVE
    if not path.is_file():
        return
    assert V.supersedes_not_emitted_lint(path) is None, (
        "%s is cited as the reference implementation that emits "
        "\"supersedes\": SUPERSEDES correctly -- it must satisfy the lint, "
        "or the documentation points at a non-example" % NEGATIVE)


def test_corpus_backlog_is_a_known_bounded_set():
    """Not a fixed-count assertion (CLAUDE.md 'The test half': a stale
    literal that silently drifts is worse than no check) -- a CEILING. The
    backlog was 19 on 2026-09-22 and 3 when this landed on 2026-09-23 as
    other authoring work closed most of it; this only catches a runaway
    regression (a systemic new source of undischarged SUPERSEDES), not
    ordinary day-to-day fluctuation in a handful of stragglers."""
    backlog = []
    for path in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py")):
        msg = V.supersedes_not_emitted_lint(path)
        if msg:
            backlog.append(path.name)
    assert len(backlog) < 50, (
        "supersedes-not-emitted backlog jumped to %d (%r) -- this is either "
        "a real regression in how drivers are authored, or the lint "
        "over-firing; investigate before raising this ceiling" % (len(backlog), backlog))
