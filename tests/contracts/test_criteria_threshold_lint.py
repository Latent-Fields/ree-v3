"""Contracts for the criteria-re-derivability gate.

Surfaces under test:
  (1) validate_experiments.criteria_threshold_lint -- flags a manifest-writing
      driver whose LOAD-BEARING criterion literals carry no threshold-family
      field, so the bar the verdict turns on never reaches the artifact.
  (2) validate_experiments.py --checks criteria_threshold -- the selector, and
      the invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens
      under --paths, never affects the exit code even under --strict).

WHY THIS GATE EXISTS. A criterion recorded `passed: true` with no measured value
and no bar is an ASSERTION, not a record: nothing in the manifest says what was
measured or what it was compared against. `/governance` Step 2b's mandatory
driver skim carries a threshold-arithmetic clause -- "given the magnitudes this
run actually measured, is the bar attainable in both directions, or does one
branch fire BY CONSTRUCTION" -- and that clause cannot be discharged from such
an artifact at all; it forces a driver read every cycle.

V3-EXQ-936a's absolute bar sat ~7,900x above the maximum attainable effect and
was logged clean for three consecutive governance cycles on exactly this shape.
In the 2026-09-09 cycle alone the gap forced three separate driver reads, each
finding something the manifest could not have shown: V3-EXQ-900's PASS label
asserting a functional half no criterion tests (GFLAG-0246); V3-EXQ-642b's C1/C2
reading a DV clamped at 1.5 in both arms, so separation is 0.0 by construction
(GFLAG-0141); V3-EXQ-231a's C2 being an arithmetic consequence of C1 under the
driver's own linear map (GFLAG-0163, MECH-106 demoted on it).

Measured 2026-09-09: this lint warns on 461 of 1474 drivers in experiments/
(31.3%), while the artifact-side check finds 184 of 216 flat manifests carrying
load-bearing criteria record NONE that is re-derivable (85.2%) -- which is why
it is WARN-only in both modes.

THE TWO RATES DIFFER BECAUSE THE CHECKS CATCH DIFFERENT HALVES, and that gap is
itself the finding: a driver can compute both numbers and drop them on the way
to the manifest (v3_exq_936a interpolates its bars into a `description` string),
or build its criteria dynamically so no literal is statically visible at all
(v3_exq_642b -- zero criterion literals, manifest flagged). Neither is reachable
from source alone. Both checks are needed.

Source: GFLAG-0249; standard section 3b "Re-derivable criteria" in
REE_assembly/evidence/planning/experimental_recording_standard_2026-07-12.md.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import validate_experiments as V  # noqa: E402


# A manifest-writing driver whose load-bearing criteria carry only a verdict --
# the V3-EXQ-936a / 642b shape, and the one this lint exists to catch.
BARE_CRITERIA = '''
import json

def run():
    manifest = {
        "run_id": "v3_exq_999_x_20260909T000000Z_v3",
        "evidence_direction": "non_contributory",
        "outcome": "PASS",
        "criteria": [
            {"name": "C1_converts", "load_bearing": True, "passed": True,
             "description": "lifts entropy above ARM_OFF on >= 3 seeds"},
            {"name": "C2_reduces", "load_bearing": True, "passed": False,
             "description": "reduces F variance share by >= 0.05"},
        ],
    }
    with open("out.json", "w") as f:
        json.dump(manifest, f)

if __name__ == "__main__":
    run()
'''

WITH_THRESHOLD = BARE_CRITERIA.replace(
    '"passed": True,\n             "description": "lifts entropy above ARM_OFF on >= 3 seeds"},',
    '"passed": True, "measured": 0.7, "threshold": 0.5},').replace(
    '"passed": False,\n             "description": "reduces F variance share by >= 0.05"},',
    '"passed": False, "measured": 0.01, "threshold": 0.05},')

EXEMPTED = BARE_CRITERIA.replace(
    "import json",
    'import json\n\nCRITERIA_THRESHOLD_EXEMPT = "count-based negative existentials"')

# Criteria present but explicitly NOT load-bearing -- no bar required.
NOT_LOAD_BEARING = BARE_CRITERIA.replace('"load_bearing": True', '"load_bearing": False')

# No criteria block at all -- makes no criterion claim to re-derive.
NO_CRITERIA = '''
import json

def run():
    manifest = {
        "run_id": "v3_exq_998_x_20260909T000000Z_v3",
        "evidence_direction": "supports",
        "readout": {"dv": 0.2},
    }
    with open("out.json", "w") as f:
        json.dump(manifest, f)

if __name__ == "__main__":
    run()
'''

# No manifest identity -- a telemetry helper, not a result-manifest writer.
NOT_A_MANIFEST_WRITER = '''
import json

def main():
    criteria = [{"name": "C1", "load_bearing": True, "passed": True}]
    with open("t.json", "w") as f:
        json.dump({"criteria": criteria}, f)

if __name__ == "__main__":
    main()
'''

LIBRARY_HELPER = '''
def build():
    return {"run_id": "x", "evidence_direction": "supports",
            "criteria": [{"name": "C1", "load_bearing": True, "passed": True}]}
'''


def _write(tmpdir, name, src):
    p = Path(tmpdir) / name
    p.write_text(src, encoding="utf-8")
    return p


def test_bare_criteria_driver_warns():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_bare.py", BARE_CRITERIA)
        msg = V.criteria_threshold_lint(p)
        assert msg is not None
        assert "carry no threshold-family" in msg


def test_threshold_fields_discharge():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_ok.py", WITH_THRESHOLD)
        assert V.criteria_threshold_lint(p) is None


def test_threshold_family_spellings_discharge():
    """The corpus does not use one spelling. A literal spelling list would flag
    26 criteria across 14 genuinely-compliant manifests -- these pairs are real
    ones from those manifests."""
    for thr in ("threshold", "requirement", "required", "bar", "rho_floor",
                "threshold_rho", "seeds_required", "tol"):
        src = BARE_CRITERIA.replace(
            '{"name": "C1_converts", "load_bearing": True, "passed": True,',
            '{"name": "C1_converts", "load_bearing": True, "passed": True,'
            f' "measured": 0.7, "{thr}": 0.5,').replace(
            '{"name": "C2_reduces", "load_bearing": True, "passed": False,',
            '{"name": "C2_reduces", "load_bearing": True, "passed": False,'
            f' "measured": 0.1, "{thr}": 0.5,')
        with tempfile.TemporaryDirectory() as td:
            p = _write(td, f"v3_exq_{thr}.py", src)
            assert V.criteria_threshold_lint(p) is None, thr


def test_one_bare_load_bearing_criterion_still_warns():
    """Each load-bearing criterion needs its OWN bar. Discharging the file when
    ANY criterion carries one was an earlier cut and it missed v3_exq_936a,
    which builds a separate verdict-grid dict carrying measured/threshold while
    its two actual load-bearing criteria carry only `passed` + prose."""
    src = BARE_CRITERIA.replace(
        '{"name": "C1_converts", "load_bearing": True, "passed": True,',
        '{"name": "C1_converts", "load_bearing": True, "passed": True,'
        ' "measured": 0.7, "threshold": 0.5,')
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_partial.py", src)
        assert V.criteria_threshold_lint(p) is not None


def test_non_load_bearing_criteria_not_gated():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_nlb.py", NOT_LOAD_BEARING)
        assert V.criteria_threshold_lint(p) is None


def test_unmarked_criteria_are_all_load_bearing():
    """A block that never says which criteria matter has not narrowed, so all of
    them count. This is what reaches the bare name->bool map shape (101 flat
    manifests), the worst-recorded in the corpus."""
    src = BARE_CRITERIA.replace('"load_bearing": True, ', '')
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_unmarked.py", src)
        assert V.criteria_threshold_lint(p) is not None


def test_exempt_marker_discharges():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_exempt.py", EXEMPTED)
        assert V.criteria_threshold_lint(p) is None


def test_no_criteria_block_not_gated():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_nocrit.py", NO_CRITERIA)
        assert V.criteria_threshold_lint(p) is None


def test_non_manifest_writer_not_gated():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "telemetry_helper.py", NOT_A_MANIFEST_WRITER)
        assert V.criteria_threshold_lint(p) is None


def test_library_helper_not_gated():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "_lib_helper.py", LIBRARY_HELPER)
        assert V.criteria_threshold_lint(p) is None


def test_unparseable_file_is_silent():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "broken.py", "def f(:\n")
        assert V.criteria_threshold_lint(p) is None


def test_selector_is_registered():
    assert "criteria_threshold" in V.CHECK_NAMES


def test_warn_only_under_paths():
    """WARN-ONLY IN BOTH MODES. Unlike arm_fingerprint / degeneracy /
    manifest_writer, a finding here must NEVER harden under --paths or --strict:
    85.2% of flat manifests carrying load-bearing criteria record none that is
    re-derivable, and a gate firing on that share of ordinary work gets disabled
    (CLAUDE.md)."""
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_bare.py", BARE_CRITERIA)
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "validate_experiments.py"),
             "--checks", "criteria_threshold", "--strict", "--paths", str(p)],
            capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "CRITERIA-THRESHOLD WARNINGS" in proc.stdout


def test_tokens_match_validate_recording():
    """The two checks must agree on the threshold-family token set. A drift here
    silently un-gates a whole recording shape at one of the two moments."""
    sys.path.insert(0, str(REPO_ROOT))
    import validate_recording as VR
    assert set(V._CRITERION_THRESHOLD_TOKENS) == set(VR._THRESHOLD_TOKENS)
