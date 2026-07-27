"""Smoke tests for phase3_preflight.py (no live hub required)."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFLIGHT = HERE / "phase3_preflight.py"


def _hermetic_env() -> dict:
    """Environment with every COORDINATOR_URL source stripped.

    run_preflight() falls back to `os.environ` for both COORDINATOR_URL and
    COORDINATOR_ENV_FILE when the env file supplies neither, so pinning
    --env-file alone is not sufficient to keep a test off the live hub.
    """
    env = dict(os.environ)
    for key in ("COORDINATOR_URL", "COORDINATOR_TOKEN",
                "COORDINATOR_ENV_FILE"):
        env.pop(key, None)
    return env


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
    assert "phase3_writer_ready" in ids
    assert "db_schema_present" in ids


def test_dry_run_local_checks():
    """--dry-run reaches the local checks via an env file with no URL.

    HERMETICITY: --env-file is pinned to a temp file that EXISTS but declares
    no COORDINATOR_URL. Until 2026-07-27 this test used the DEFAULT --env-file,
    and --dry-run does NOT skip the HTTP block (only --mock does; --dry-run
    skips the SSH probes, `if dry_run or mock`). On the operator's Mac the
    default resolves to a real $HOME/REE_Working/REE_assembly/coordinator.env
    supplying a live URL, so every run of this unit suite fired /health and
    /shadow/status at the PRODUCTION coordinator -- making the test's runtime
    and outcome depend on whether the hub was up.

    Fixed in the TEST, not in run_preflight(): live HTTP under --dry-run is the
    documented operator contract ("--dry-run skips SSH; --mock forces network
    checks to SKIP (tests only)" -- PHASE3_CUTOVER.md; and _print_report
    distinguishes "(dry-run: SSH checks skipped)" from "(mock: network checks
    skipped)"). --dry-run is the mode for a host with coordinator.env but only
    "optional SSH to hub" (module header). Making it skip HTTP would gut it.

    Distinct from test_local_modes_do_not_need_coordinator_env below, which
    pins a NONEXISTENT path and so also exercises main()'s missing-env-file
    branch. Here the file exists and simply lacks the key.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / "coordinator.env"
        env_file.write_text(
            "# no COORDINATOR_URL: forces the local-checks-only path\n"
            "COORDINATOR_SSH_USER=ree\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(PREFLIGHT),
             "--env-file", str(env_file), "--dry-run", "--json"],
            capture_output=True, text=True, timeout=60, check=False,
            env=_hermetic_env())
    # The overall verdict may be FAIL (no URL); --json must still emit a body.
    data = json.loads(proc.stdout)
    assert "checks" in data
    ids = {c["id"] for c in data["checks"]}
    # No URL -> the HTTP block must SKIP rather than dial out. A PASS/FAIL on
    # either of these means the run made a live request to a real coordinator.
    assert {"hub_health", "coordinator_api"} <= ids, sorted(ids)
    for cid in ("hub_health", "coordinator_api"):
        check = next(c for c in data["checks"] if c["id"] == cid)
        assert check["status"] == "SKIP", check
    assert "fleet_lifecycle" not in ids, sorted(ids)
    writer = next(c for c in data["checks"]
                  if c["id"] == "phase3_writer_ready")
    # Post-cutover polarity: sync_daemon IS the git writer, so the flag must
    # be True. This asserted the pre-cutover "still a stub" invariant until
    # 2026-07-27; see phase3_preflight._import_sync_daemon_ready.
    assert writer["status"] == "PASS", writer


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


def test_local_modes_do_not_need_coordinator_env():
    """--mock and --dry-run must work on a host with no coordinator.env.

    THE POINT OF THIS TEST IS TO BE HOST-INDEPENDENT. The four subprocess tests
    above pass on the operator's Mac whether or not this invariant holds,
    because they use the DEFAULT --env-file, and on the Mac that resolves to a
    real $HOME/REE_Working/REE_assembly/coordinator.env. That file is
    GITIGNORED and outside this repo, so it exists nowhere else -- and until
    2026-07-27 run_preflight() short-circuited on the resulting empty URL,
    skipping every local check including the phase3_writer_ready gate. Result:
    three tests that were green on one laptop and red on the whole fleet, which
    nobody noticed because remote_pytest.sh's default args did not collect
    coordinator/ at all.

    Forcing a nonexistent --env-file reproduces the non-Mac condition
    deterministically, so a regression fails everywhere rather than only where
    it is least likely to be run.
    """
    for extra in (["--mock"], ["--dry-run"], ["--mock", "--cutover-window"]):
        proc = subprocess.run(
            [sys.executable, str(PREFLIGHT),
             "--env-file", "/nonexistent/coordinator.env", "--json"] + extra,
            capture_output=True, text=True, timeout=60, check=False,
            env=_hermetic_env())
        # --json is a machine contract: JSON on stdout regardless of exit code.
        data = json.loads(proc.stdout)
        ids = {c["id"] for c in data["checks"]}
        # The local checks are the whole reason these modes exist.
        assert "phase3_writer_ready" in ids, (extra, sorted(ids))
        assert "db_schema_present" in ids, (extra, sorted(ids))
        writer = next(c for c in data["checks"]
                      if c["id"] == "phase3_writer_ready")
        assert writer["status"] == "PASS", (extra, writer)
        # A missing URL must degrade to SKIP here, not FAIL: there is nothing
        # to reach and nothing was asked of it.
        config = next(c for c in data["checks"] if c["id"] == "config")
        assert config["status"] == "SKIP", (extra, config)


def test_live_mode_still_fails_without_url():
    """Negative control for the test above -- the relaxation is mode-scoped.

    Without --mock/--dry-run, a missing COORDINATOR_URL is still a blocking
    config FAIL. If this ever passes-by-accident, the check above proves
    nothing.
    """
    proc = subprocess.run(
        [sys.executable, str(PREFLIGHT),
         "--env-file", "/nonexistent/coordinator.env", "--json"],
        capture_output=True, text=True, timeout=60, check=False,
        env=_hermetic_env())
    assert proc.returncode == 2, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    config = next(c for c in data["checks"] if c["id"] == "config")
    assert config["status"] == "FAIL", config
    assert data.get("ok") is False


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
    test_local_modes_do_not_need_coordinator_env()
    print("PASS test_local_modes_do_not_need_coordinator_env")
    test_live_mode_still_fails_without_url()
    print("PASS test_live_mode_still_fails_without_url")
    print("RESULT: PASS (phase3 preflight smoke)")
