"""Contracts for the flat-scalar-readout gate.

Surfaces under test:
  (1) validate_experiments.flat_scalar_readout_lint -- flags a manifest-writing
      driver that emits NO flat scalar readout block under any of the four
      spellings the runpack converter harvests (`readout` / `metrics` /
      `aggregates` / `summary_metrics`).
  (2) validate_experiments.py --checks flat_scalar_readout -- the selector, and
      the invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens
      under --paths, never affects the exit code even under --strict).

WHY THIS GATE EXISTS. The runpack converter
(REE_assembly evidence/experiments/scripts/sync_v3_results.build_runpack_docs)
builds a pack's metrics.json `values` by harvesting a FLAT scalar dict under one
of exactly those four spellings, and build_experiment_indexes reads only the
NUMERIC entries of that block (`_is_number`, l.315, which excludes bool as an int
subclass). A readout recorded only as a dict keyed by arm or by seed -- the shape
nearly every multi-arm driver emits -- matches none of the four, so the pack
scores with `values == {}`. Consequences, all verified against the indexer
source: no `fail_if` stop threshold can fire (the lookup returns None, the check
is skipped, and final_status falls back to the manifest's OWN self-declared
status, which is what claim_evidence.v1.json records); the duplicate-emission
supersession fingerprint is skipped entirely (l.2252) so a byte-identical
re-emission is never auto-superseded and both copies score; and the index carries
no deltas or key-metrics columns.

Measured 2026-09-09: 1094 of 2931 packs score with no numeric metrics.values, and
the rate for packs emitted in 2026-09 is 65% -- the current default, not a legacy
backlog. 703 of 1358 manifest-writing drivers trip this lint, which is exactly
why it is WARN-only: a gate firing on the majority of ordinary work gets
disabled.

Source: REE_assembly/evidence/planning/flat_scalar_readout_recording_gap_20260909.md;
standard section 3b "Machine-readable verdict readout" in
experimental_recording_standard_2026-07-12.md. Reference implementation:
experiments/v3_exq_1015_mech465_zworld_warmup_budget_dispersion_sweep.py.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import validate_experiments as V  # noqa: E402


# A manifest-writing driver with rich blocks that are ALL keyed by arm or seed --
# the V3-EXQ-1015 shape, and the one this lint exists to catch.
NESTED_ONLY = '''
import json

def run():
    manifest = {
        "run_id": "v3_exq_999_x_20260909T000000Z_v3",
        "evidence_direction": "non_contributory",
        "outcome": "FAIL",
        "arm_results": [{"arm_id": "A", "dv": 0.2}],
        "cell_summary": {"A": {"dv": 0.2}},
        "per_arm_gate": {"green_arms": []},
    }
    with open("out.json", "w") as f:
        json.dump(manifest, f)

if __name__ == "__main__":
    run()
'''

# The same driver with the flat scalar projection added.
WITH_READOUT = NESTED_ONLY.replace(
    '"per_arm_gate": {"green_arms": []},',
    '"per_arm_gate": {"green_arms": []},\n        "readout": {"max_dv": 0.2, "bar": 0.25},')

EXEMPTED = NESTED_ONLY.replace(
    "import json",
    'import json\n\nFLAT_SCALAR_READOUT_EXEMPT = "verdict turns on nothing scalar"')

# No manifest identity -- a telemetry helper, not a result-manifest writer.
NOT_A_MANIFEST_WRITER = '''
import json

def main():
    with open("t.json", "w") as f:
        json.dump({"ticks": 3}, f)

if __name__ == "__main__":
    main()
'''

# No __main__ entry point -- library-style helper.
LIBRARY_HELPER = '''
import json

def build():
    return {"run_id": "x", "evidence_direction": "supports"}
'''


def _write(tmpdir, name, src):
    p = Path(tmpdir) / name
    p.write_text(src, encoding="utf-8")
    return p


def test_nested_only_driver_warns():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_nested.py", NESTED_ONLY)
        msg = V.flat_scalar_readout_lint(p)
        assert msg is not None
        assert "NO flat scalar readout block" in msg


def test_readout_block_discharges():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_ok.py", WITH_READOUT)
        assert V.flat_scalar_readout_lint(p) is None


def test_each_recognised_spelling_discharges():
    for spelling in ("readout", "metrics", "aggregates", "summary_metrics"):
        src = NESTED_ONLY.replace(
            '"per_arm_gate": {"green_arms": []},',
            f'"per_arm_gate": {{"green_arms": []}},\n        "{spelling}": {{"n": 1}},')
        with tempfile.TemporaryDirectory() as td:
            p = _write(td, f"v3_exq_{spelling}.py", src)
            assert V.flat_scalar_readout_lint(p) is None, spelling


def test_exempt_marker_discharges():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_exempt.py", EXEMPTED)
        assert V.flat_scalar_readout_lint(p) is None


def test_non_manifest_writer_not_gated():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "telemetry_helper.py", NOT_A_MANIFEST_WRITER)
        assert V.flat_scalar_readout_lint(p) is None


def test_library_helper_not_gated():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "_lib_helper.py", LIBRARY_HELPER)
        assert V.flat_scalar_readout_lint(p) is None


def test_unparseable_file_is_silent():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "broken.py", "def f(:\n")
        assert V.flat_scalar_readout_lint(p) is None


def test_selector_is_registered():
    assert "flat_scalar_readout" in V.CHECK_NAMES


def test_reference_driver_is_compliant():
    """The driver the standard cites as the reference implementation must
    actually satisfy the lint -- otherwise the documentation points at a
    non-example."""
    ref = (REPO_ROOT / "experiments"
           / "v3_exq_1015_mech465_zworld_warmup_budget_dispersion_sweep.py")
    if not ref.is_file():
        return  # renamed/retired -- not this test's business to fail on
    assert V.flat_scalar_readout_lint(ref) is None


def test_warn_only_under_paths():
    """WARN-ONLY IN BOTH MODES. Unlike arm_fingerprint / degeneracy /
    manifest_writer, a finding here must NEVER harden under --paths or --strict:
    703 of 1358 manifest-writing drivers trip it, and a gate that fires on the
    majority of ordinary work gets disabled (CLAUDE.md)."""
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "v3_exq_nested.py", NESTED_ONLY)
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "validate_experiments.py"),
             "--checks", "flat_scalar_readout", "--strict", "--paths", str(p)],
            capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "FLAT-SCALAR-READOUT WARNINGS" in proc.stdout


def test_spellings_match_validate_recording():
    """The two checks must agree on the four spellings. A drift here silently
    un-gates a whole recording shape at one of the two enforcement moments."""
    sys.path.insert(0, str(REPO_ROOT))
    import validate_recording as VR
    assert set(V._READOUT_SPELLINGS) == set(VR._READOUT_SPELLINGS)
