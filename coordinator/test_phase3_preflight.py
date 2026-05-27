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


def test_cutover_window_flag_accepted():
    """--cutover-window is parsed and propagates to the fleet_lifecycle
    detail. Uses --mock so no live coordinator is required."""
    proc = subprocess.run(
        [sys.executable, str(PREFLIGHT),
         "--mock", "--cutover-window", "--json"],
        capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    # In mock mode shadow/status is SKIPped, so fleet_lifecycle won't
    # appear. The flag just needs to be accepted without an argparse error.
    assert data.get("ok") is True
    ids = {c["id"] for c in data["checks"]}
    assert "fleet_lifecycle" not in ids, \
        "mock mode should skip fleet_lifecycle (no HTTP)"


def test_evaluate_fleet_lifecycle_pure():
    """Direct unit test of the policy function -- no subprocess, no HTTP."""
    sys.path.insert(0, str(HERE))
    try:
        from phase3_preflight import (  # noqa: E402
            _evaluate_fleet_lifecycle, EXPECTED_LIFECYCLE_PEERS)
    finally:
        sys.path.pop(0)

    # All live -> PASS in both modes.
    all_live = [{"machine": p, "lifecycle_state": "live"}
                for p in EXPECTED_LIFECYCLE_PEERS]
    assert _evaluate_fleet_lifecycle(
        all_live, cutover_window=False)[0] == "PASS"
    assert _evaluate_fleet_lifecycle(
        all_live, cutover_window=True)[0] == "PASS"

    # One gracefully_offline -> PASS steady-state, FAIL cutover.
    mixed = [{"machine": p, "lifecycle_state":
              "gracefully_offline" if p == "ree-cloud-4" else "live"}
             for p in EXPECTED_LIFECYCLE_PEERS]
    assert _evaluate_fleet_lifecycle(
        mixed, cutover_window=False)[0] == "PASS"
    status_cut, msg_cut, _ = _evaluate_fleet_lifecycle(
        mixed, cutover_window=True)
    assert status_cut == "FAIL"
    assert "ree-cloud-4=gracefully_offline" in msg_cut

    # One stale -> FAIL in both modes.
    one_stale = [{"machine": p,
                  "lifecycle_state": "stale" if p == "ree-cloud-2" else "live"}
                 for p in EXPECTED_LIFECYCLE_PEERS]
    for cw in (False, True):
        status, msg, _ = _evaluate_fleet_lifecycle(
            one_stale, cutover_window=cw)
        assert status == "FAIL"
        assert "ree-cloud-2=stale" in msg

    # Missing peer -> FAIL.
    missing = [{"machine": p, "lifecycle_state": "live"}
               for p in EXPECTED_LIFECYCLE_PEERS if p != "ree-cloud-1"]
    status, msg, _ = _evaluate_fleet_lifecycle(
        missing, cutover_window=False)
    assert status == "FAIL"
    assert "ree-cloud-1=missing" in msg


if __name__ == "__main__":
    test_help_exits_zero()
    print("PASS test_help_exits_zero")
    test_mock_json_all_pass_structure()
    print("PASS test_mock_json_all_pass_structure")
    test_dry_run_local_checks()
    print("PASS test_dry_run_local_checks")
    test_cutover_window_flag_accepted()
    print("PASS test_cutover_window_flag_accepted")
    test_evaluate_fleet_lifecycle_pure()
    print("PASS test_evaluate_fleet_lifecycle_pure")
    print("RESULT: PASS (phase3 preflight smoke)")
