"""Contracts for the manifest-writer chokepoint gate.

Two surfaces:
  (1) validate_experiments.manifest_writer_lint -- flags a script that carries the
      manifest-identity tokens (run_id + evidence_direction) AND does a raw
      json.dump/json.dumps without routing through the single sanctioned writer
      experiments/pack_writer.write_flat_manifest (or write_pack /
      ExperimentPackWriter), with a MANIFEST_WRITER_EXEMPT opt-out.
  (2) validate_experiments.py --checks manifest_writer -- the surgical selector the
      commit-time gate (scripts/precommit_contracts.sh Block 1b) uses so it runs ONLY
      the manifest-writer lint (HARD under --paths), without expanding the
      emit_outcome/degeneracy/arm-fingerprint contracts onto the broader v3_*.py set
      it also scopes.

Design plan: REE_assembly/evidence/planning/pack_writer_single_writer_migration_plan.md
sec 7 item 3 (harden the lint to a commit gate); Experimental Recording Standard sec 4.
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

# A script that carries the manifest-identity tokens and hand-rolls a raw json.dump.
_RAW_DUMP = (
    'if __name__ == "__main__":\n'
    "    import json\n"
    '    manifest = {"run_id": "exq_000zz_v3",\n'
    '                "architecture_epoch": "ree_hybrid_guardrails_v1",\n'
    '                "evidence_direction": "supports", "status": "PASS"}\n'
    '    with open("out.json", "w") as f:\n'
    "        json.dump(manifest, f, indent=2)\n"
)


def _lint(src: str):
    """Write src to a temp .py under experiments/ and return the lint result."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        return V.manifest_writer_lint(Path(name))
    finally:
        os.unlink(name)


# ---- (1) lint detection branches -------------------------------------------

def test_m1_raw_dump_with_identity_flagged():
    """run_id + evidence_direction + raw json.dump, no route -> flagged."""
    issue = _lint(_RAW_DUMP)
    assert issue is not None
    assert "write_flat_manifest" in issue


def test_m2_routed_write_flat_manifest_ok():
    """Routing through write_flat_manifest discharges the check even with a dump."""
    src = (
        "from experiments.pack_writer import write_flat_manifest\n"
        + _RAW_DUMP
        + "    write_flat_manifest(manifest, 'out')\n"
    )
    assert _lint(src) is None


def test_m3_routed_write_pack_ok():
    """The pack path (write_pack) also discharges the check."""
    src = (
        "from experiments.pack_writer import write_pack\n"
        + _RAW_DUMP
        + "    write_pack(manifest)\n"
    )
    assert _lint(src) is None


def test_m4_exempt_marker_suppresses():
    """MANIFEST_WRITER_EXEMPT opt-out suppresses the check."""
    src = 'MANIFEST_WRITER_EXEMPT = "crash-report smoke"\n' + _RAW_DUMP
    assert _lint(src) is None


def test_m5_no_identity_tokens_not_gated():
    """A dump without both manifest-identity tokens is not a manifest write."""
    src = (
        'if __name__ == "__main__":\n'
        "    import json\n"
        '    telemetry = {"note": "hello"}\n'
        '    with open("t.json", "w") as f:\n'
        "        json.dump(telemetry, f)\n"
    )
    assert _lint(src) is None


def test_m6_identity_without_dump_not_gated():
    """Identity tokens present but no raw dump (e.g. emits via a helper) -> not gated."""
    src = (
        'if __name__ == "__main__":\n'
        '    manifest = {"run_id": "exq_000zz_v3",\n'
        '                "evidence_direction": "supports"}\n'
        "    emit_outcome(manifest)\n"
    )
    assert _lint(src) is None


def test_m7_library_no_main_block_exempt():
    """A library-style file with no __main__ entry point is exempt."""
    src = (
        "import json\n"
        "def helper(m):\n"
        '    with open("x.json", "w") as f:\n'
        "        json.dump(m, f)\n"
    )
    assert _lint(src) is None


# ---- (2) --checks manifest_writer selector (commit-gate surface) ------------

def _run_cli(*args):
    """Run validate_experiments.py as the gate invokes it; return (rc, stdout)."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _write_probe(src: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     prefix="v3_zz_probe_", dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        return f.name


def test_m8_checks_manifest_writer_blocks_under_paths():
    """--strict --checks manifest_writer --paths blocks a raw-dump script (HARD)."""
    name = _write_probe(_RAW_DUMP)
    try:
        rc, out = _run_cli("--strict", "--quiet", "--checks", "manifest_writer",
                           "--paths", name)
        assert rc == 1, out
        assert "write_flat_manifest" in out
    finally:
        os.unlink(name)


def test_m9_checks_manifest_writer_is_surgical():
    """--checks manifest_writer runs ONLY that lint: a script that is fine for the
    manifest gate but would trip the emit_outcome conformance check passes here."""
    # No route needed: routes through write_flat_manifest -> manifest gate clean.
    # It also lacks emit_outcome in __main__, which the FULL run would flag -- proving
    # the selector does not run the conformance check.
    src = (
        "from experiments.pack_writer import write_flat_manifest\n"
        + _RAW_DUMP
        + "    write_flat_manifest(manifest, 'out')\n"
    )
    name = _write_probe(src)
    try:
        rc_only, _ = _run_cli("--strict", "--quiet", "--checks", "manifest_writer",
                              "--paths", name)
        rc_full, out_full = _run_cli("--strict", "--quiet", "--paths", name)
        assert rc_only == 0            # manifest gate alone: clean
        assert rc_full == 1            # full run trips emit_outcome conformance
        assert "emit_outcome" in out_full
    finally:
        os.unlink(name)


def test_m10_checks_manifest_writer_respects_exempt_under_paths():
    """The exempt marker suppresses the block even under the HARD --paths gate."""
    name = _write_probe('MANIFEST_WRITER_EXEMPT = "smoke"\n' + _RAW_DUMP)
    try:
        rc, _ = _run_cli("--strict", "--quiet", "--checks", "manifest_writer",
                         "--paths", name)
        assert rc == 0
    finally:
        os.unlink(name)
