"""Smoke tests for phase3_preflight.py (no live hub required)."""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFLIGHT = HERE / "phase3_preflight.py"


def test_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--help"],
        capture_output=True, text=True, timeout=10, check=False)
    assert proc.returncode == 0
    assert "Phase 3 pre-cutover" in proc.stdout


def test_mock_json_all_pass_structure():
    proc = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--mock", "--json"],
        capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert "checks" in data
    assert data.get("ok") is True
    ids = {c["id"] for c in data["checks"]}
    assert "phase3_writer_stub" in ids
    assert "db_schema_present" in ids


def test_dry_run_local_checks():
    proc = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--dry-run", "--json"],
        capture_output=True, text=True, timeout=60, check=False)
    # May FAIL without coordinator.env URL; must emit valid JSON.
    data = json.loads(proc.stdout)
    assert "checks" in data
    stub = next(c for c in data["checks"]
                if c["id"] == "phase3_writer_stub")
    assert stub["status"] == "PASS"


if __name__ == "__main__":
    test_help_exits_zero()
    print("PASS test_help_exits_zero")
    test_mock_json_all_pass_structure()
    print("PASS test_mock_json_all_pass_structure")
    test_dry_run_local_checks()
    print("PASS test_dry_run_local_checks")
    print("RESULT: PASS (phase3 preflight smoke)")
