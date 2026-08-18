"""Smoke tests for phase3_preflight.py (no live hub required)."""

import io
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


# ---------------------------------------------------------------------------
# Deployment stage (2026-08-18, chip-20260816-phase3-preflight-postcutover-verdict)
#
# The defect these pin: this tool was a PRE-cutover gate, Phase 3 cut over
# 2026-05-29, and `hub_sync_mode_safe` therefore FAILed precisely BECAUSE the
# cutover succeeded -- making the top-line verdict permanently
# "FAIL -- N blocking check(s) / Do NOT run phase3_cutover.sh". Callers were
# reduced to diffing per-check lines against a hand-captured baseline, so a
# genuinely NEW failure could not be seen. Roughly half the tests below are
# negative controls: the pre-cutover gate must still gate, and fleet_lifecycle
# must still block, or the fix is just a way to make red look green.
# ---------------------------------------------------------------------------


def _import_preflight():
    sys.path.insert(0, str(HERE))
    try:
        import phase3_preflight  # noqa: E402
        return phase3_preflight
    finally:
        sys.path.pop(0)


def test_observed_sync_mode_classification():
    pf = _import_preflight()
    assert pf._observed_sync_mode("SYNC_MODE=authoritative") == "authoritative"
    assert pf._observed_sync_mode("SYNC_MODE=coordinator") == "coordinator"
    assert pf._observed_sync_mode("SYNC_MODE=shadow") == "other"
    assert pf._observed_sync_mode("") == "unreadable"
    assert pf._observed_sync_mode(None) == "unreadable"


def test_resolve_stage_signal_precedence():
    """A flag always wins; the hub beats the local flag; nothing -> unknown."""
    pf = _import_preflight()
    # Flag wins even against a contradicting hub reading. This is what makes
    # deploy/phase3_cutover.sh safe: it pins pre-cutover and the hub's actual
    # (authoritative) mode must not silently re-label the run.
    assert pf._resolve_stage(pf.STAGE_PRE, "authoritative", True) == (
        pf.STAGE_PRE, "flag")
    assert pf._resolve_stage(pf.STAGE_POST, "coordinator", False) == (
        pf.STAGE_POST, "flag")
    # Hub reading beats the weak local writer-ready flag.
    assert pf._resolve_stage(None, "authoritative", True) == (
        pf.STAGE_POST, "hub_sync_mode")
    assert pf._resolve_stage(None, "coordinator", True) == (
        pf.STAGE_PRE, "hub_sync_mode")
    # Hub not probed -> fall back to the local flag, LABELLED as weak.
    assert pf._resolve_stage(None, None, True) == (
        pf.STAGE_POST, "writer_ready_flag")
    assert pf._resolve_stage(None, "unreadable", True) == (
        pf.STAGE_POST, "writer_ready_flag")
    # No signal at all.
    assert pf._resolve_stage(None, "unreadable", False) == (
        pf.STAGE_UNKNOWN, "undetermined")


def test_evaluate_hub_sync_mode_three_polarities():
    """The whole point: same observation, three different verdicts."""
    pf = _import_preflight()

    def ev(observed, stage, pinned, raw="SYNC_MODE=x", err="boom"):
        return pf._evaluate_hub_sync_mode(
            observed, raw, err, stage=stage, pinned=pinned)

    # -- auto (stage inferred FROM this probe) -> always advisory ------------
    # Blocking here would be circular: the check would supply its own
    # expectation and could never fail.
    for observed in ("authoritative", "coordinator", "other", "unreadable"):
        st, msg, blocking = ev(observed, pf.STAGE_POST, False)
        assert blocking is False, (observed, st, msg)
        assert st != "FAIL", (observed, st, msg)

    # -- pinned pre-cutover: the ORIGINAL gate, unchanged -------------------
    st, msg, blocking = ev("authoritative", pf.STAGE_PRE, True)
    assert (st, blocking) == ("FAIL", True)
    assert msg == "hub already SYNC_MODE=authoritative"
    st, msg, blocking = ev("coordinator", pf.STAGE_PRE, True)
    assert (st, blocking) == ("PASS", True)
    st, msg, blocking = ev("unreadable", pf.STAGE_PRE, True)
    assert (st, blocking) == ("FAIL", True)
    assert "cannot read hub env" in msg

    # -- pinned post-cutover: OPPOSITE polarity ----------------------------
    st, msg, blocking = ev("authoritative", pf.STAGE_POST, True)
    assert (st, blocking) == ("PASS", True), msg
    st, msg, blocking = ev("coordinator", pf.STAGE_POST, True)
    assert (st, blocking) == ("FAIL", True)
    assert "REGRESSED" in msg, msg
    st, msg, blocking = ev("unreadable", pf.STAGE_POST, True)
    assert (st, blocking) == ("FAIL", True)


def _fake_probes(pf, sync_mode="SYNC_MODE=authoritative",
                 lifecycle="live", writer_ready=True):
    """Patch the live-hub surface. Returns (restore_callable).

    HERMETICITY, and a real hazard worth knowing about: these are the first
    IN-PROCESS tests in this file -- every pre-existing one runs the tool in a
    subprocess, so none of them was ever exposed to interpreter state left by
    sibling modules. `_import_sync_daemon_ready()` does `import sync_daemon`
    and reads its module global PHASE3_GIT_WRITER_READY. Several other modules
    in coordinator/ (test_manifest_spool.py, test_phase3_writer_smoke.py,
    test_phase3_sidefile_sync.py) toggle that global and "restore" it to
    False -- the PRE-cutover default, which stopped being the real value at
    d98f9a5. So under a whole-directory `pytest coordinator/` run the flag is
    left False, and phase3_writer_ready FAILs for reasons that have nothing to
    do with the code under test. Confirmed 2026-08-18 on ree-worker-2: green
    file-alone, two failures collected with the directory.

    Pinning it here is scoping, not papering over: these tests are about the
    STAGE machinery, and phase3_writer_ready has its own coverage. The
    `writer_ready` knob is also load-bearing in its own right -- it is the
    weak fallback signal _resolve_stage uses when the hub is not probed.
    """
    orig = (pf._ssh_run, pf._http_get, pf._run_check_shadow,
            pf._import_sync_daemon_ready)
    pf._import_sync_daemon_ready = lambda: (
        (True, "PHASE3_GIT_WRITER_READY is True (writer live)")
        if writer_ready else
        (False, "PHASE3_GIT_WRITER_READY is False (pinned by test)"))

    def fake_ssh(host, user, remote_cmd, *, dry_run, timeout=20):
        if "SYNC_MODE" in remote_cmd:
            return True, sync_mode, ""
        if "systemctl is-active" in remote_cmd:
            return True, "active", ""
        return True, "", ""

    def fake_http(url, token, path, timeout=8.0):
        if path == "/health":
            return 200, {"ok": True, "mode": "coordinator"}, ""
        if path == "/shadow/status":
            return 200, {"machines": [
                {"machine": p, "lifecycle_state": lifecycle}
                for p in pf.EXPECTED_LIFECYCLE_PEERS]}, ""
        return 200, {}, ""

    pf._ssh_run = fake_ssh
    pf._http_get = fake_http
    pf._run_check_shadow = lambda url, token: (0, "HEALTHY")

    def restore():
        (pf._ssh_run, pf._http_get, pf._run_check_shadow,
         pf._import_sync_daemon_ready) = orig
    return restore


def _run_fake(pf, tmp, **kwargs):
    env_file = Path(tmp) / "coordinator.env"
    env_file.write_text(
        "COORDINATOR_URL=http://127.0.0.1:1/\n"
        "COORDINATOR_LOCAL_TOKEN=tok\n", encoding="utf-8")
    return pf.run_preflight(env_file=env_file, **kwargs)


def test_authoritative_hub_no_longer_forces_a_fail_verdict():
    """THE REGRESSION PIN. A healthy post-cutover fleet must verdict PASS.

    Before the fix this exact fleet -- everything live, hub authoritative,
    sync_daemon active, no orphaned claims -- produced
    "VERDICT: FAIL -- Do NOT run phase3_cutover.sh", because the one check
    asserting the hub was NOT yet flipped is satisfied only before cutover.
    """
    pf = _import_preflight()
    restore = _fake_probes(pf)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            summary = _run_fake(pf, tmp)
    finally:
        restore()
    assert summary["stage"] == pf.STAGE_POST, summary["stage"]
    assert summary["stage_source"] == "hub_sync_mode", summary
    assert summary["ok"] is True, [
        c for c in summary["checks"] if c["status"] == "FAIL"]
    assert summary["fail_count"] == 0
    hub = next(c for c in summary["checks"] if c["id"] == "hub_sync_mode_safe")
    assert hub["blocking"] is False, hub
    lines = pf._verdict_lines(summary)
    assert "steady state healthy" in lines[0], lines
    # The cutover advice must NOT print post-cutover -- it is unactionable
    # there, and printing it unconditionally is what trained callers to
    # ignore the verdict.
    assert not any("phase3_cutover.sh" in ln for ln in lines), lines


def test_pre_cutover_gate_still_gates():
    """NEGATIVE CONTROL for the test above.

    If the fix worked by making hub_sync_mode_safe toothless everywhere, the
    original gate would be gone. Pinned pre-cutover it must still FAIL on an
    already-authoritative hub, and still print the cutover advice.
    """
    pf = _import_preflight()
    restore = _fake_probes(pf)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            summary = _run_fake(pf, tmp, stage=pf.STAGE_PRE)
    finally:
        restore()
    assert summary["stage"] == pf.STAGE_PRE
    assert summary["stage_source"] == "flag"
    assert summary["ok"] is False
    hub = next(c for c in summary["checks"] if c["id"] == "hub_sync_mode_safe")
    assert (hub["status"], hub["blocking"]) == ("FAIL", True), hub
    lines = pf._verdict_lines(summary)
    assert any("Do NOT run phase3_cutover.sh" in ln for ln in lines), lines


def test_cutover_window_implies_pre_cutover():
    """deploy/phase3_cutover.sh passes ONLY --cutover-window and nothing else.

    That script is out of scope for this change, so --cutover-window must
    keep selecting the pre-cutover polarity on its own or the cutover gate
    would silently become a health check.
    """
    pf = _import_preflight()
    restore = _fake_probes(pf)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            summary = _run_fake(pf, tmp, cutover_window=True)
    finally:
        restore()
    assert summary["stage"] == pf.STAGE_PRE, summary["stage"]
    assert summary["stage_source"] == "flag", summary
    assert summary["ok"] is False, "cutover gate must still refuse"
    hub = next(c for c in summary["checks"] if c["id"] == "hub_sync_mode_safe")
    assert hub["status"] == "FAIL" and hub["blocking"] is True, hub


def test_post_cutover_pinned_catches_a_regressed_hub():
    """The new blocking direction: a hub back in coordinator mode is a fault."""
    pf = _import_preflight()
    restore = _fake_probes(pf, sync_mode="SYNC_MODE=coordinator")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            summary = _run_fake(pf, tmp, stage=pf.STAGE_POST)
    finally:
        restore()
    assert summary["ok"] is False
    hub = next(c for c in summary["checks"] if c["id"] == "hub_sync_mode_safe")
    assert (hub["status"], hub["blocking"]) == ("FAIL", True), hub
    assert "REGRESSED" in hub["message"], hub
    # ...and under auto the same hub reads as an ordinary pre-cutover posture,
    # not as a fault -- which is exactly why pinning exists.
    restore = _fake_probes(pf, sync_mode="SYNC_MODE=coordinator")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            auto = _run_fake(pf, tmp)
    finally:
        restore()
    assert auto["stage"] == pf.STAGE_PRE, auto["stage"]
    assert auto["ok"] is True, auto


def test_fleet_lifecycle_stays_blocking_in_every_stage():
    """NEGATIVE CONTROL: the check that carries the real signal must block.

    The chip that motivated this change said so explicitly. A stale peer is
    the finding an operator needs; if the stage machinery ever demoted it,
    the verdict would be legible and useless.
    """
    pf = _import_preflight()
    for stage in (None, pf.STAGE_PRE, pf.STAGE_POST):
        restore = _fake_probes(pf, lifecycle="stale")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                summary = _run_fake(pf, tmp, stage=stage)
        finally:
            restore()
        fleet = next(c for c in summary["checks"]
                     if c["id"] == "fleet_lifecycle")
        assert fleet["status"] == "FAIL", (stage, fleet)
        assert fleet["blocking"] is True, (stage, fleet)
        assert summary["ok"] is False, (stage, summary["fail_count"])
        assert summary["fail_count"] >= 1, (stage, summary)


def test_stage_falls_back_to_the_local_writer_flag_when_hub_unprobed():
    """--dry-run/--mock never reads the hub, so the stage comes from the weak
    local signal -- and must be LABELLED as weak rather than passed off as a
    hub reading. With the flag False there is no signal at all and the tool
    must say so instead of guessing.
    """
    pf = _import_preflight()
    with tempfile.TemporaryDirectory() as tmp:
        restore = _fake_probes(pf, writer_ready=True)
        try:
            ready = _run_fake(pf, tmp, dry_run=True)
        finally:
            restore()
        restore = _fake_probes(pf, writer_ready=False)
        try:
            unready = _run_fake(pf, tmp, dry_run=True)
        finally:
            restore()
    assert (ready["stage"], ready["stage_source"]) == (
        pf.STAGE_POST, "writer_ready_flag"), ready
    assert (unready["stage"], unready["stage_source"]) == (
        pf.STAGE_UNKNOWN, "undetermined"), unready
    # No stage -> no stage-specific advice, and in particular no unactionable
    # cutover warning.
    lines = pf._verdict_lines(unready)
    assert any("undetermined" in ln for ln in lines), lines


def test_json_carries_stage_and_blocking_fields():
    """--json is a machine contract; the new fields are part of it now."""
    proc = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--mock", "--json"],
        capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    assert data["stage"] in ("pre-cutover", "post-cutover", "undetermined")
    assert "stage_source" in data
    assert "advisory_fail_count" in data
    for c in data["checks"]:
        assert "blocking" in c, c
    # --mock never probes the hub, so the stage can only come from the weak
    # local signal. Labelling that honestly is the point of stage_source.
    assert data["stage_source"] == "writer_ready_flag", data["stage_source"]


def test_cutover_window_and_post_cutover_are_rejected():
    """Contradictory flags must be an argparse error, not a silent winner."""
    proc = subprocess.run(
        [sys.executable, str(PREFLIGHT),
         "--mock", "--cutover-window", "--post-cutover"],
        capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "cannot be combined" in proc.stderr, proc.stderr


def test_pre_and_post_cutover_are_mutually_exclusive():
    proc = subprocess.run(
        [sys.executable, str(PREFLIGHT),
         "--mock", "--pre-cutover", "--post-cutover"],
        capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_report_marks_advisory_checks():
    """A non-blocking FAIL/WARN must not read like one that blocks."""
    pf = _import_preflight()
    restore = _fake_probes(pf)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            summary = _run_fake(pf, tmp)
    finally:
        restore()
    buf = io.StringIO()
    orig_stdout = sys.stdout
    sys.stdout = buf
    try:
        pf._print_report(summary)
    finally:
        sys.stdout = orig_stdout
    out = buf.getvalue()
    assert "stage: post-cutover [detected from hub SYNC_MODE]" in out, out
    assert "hub_sync_mode_safe (advisory):" in out, out
    # ASCII-only contract (CLAUDE.md).
    out.encode("ascii")


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
    for _fn in (
            test_observed_sync_mode_classification,
            test_resolve_stage_signal_precedence,
            test_evaluate_hub_sync_mode_three_polarities,
            test_authoritative_hub_no_longer_forces_a_fail_verdict,
            test_pre_cutover_gate_still_gates,
            test_cutover_window_implies_pre_cutover,
            test_post_cutover_pinned_catches_a_regressed_hub,
            test_fleet_lifecycle_stays_blocking_in_every_stage,
            test_stage_falls_back_to_the_local_writer_flag_when_hub_unprobed,
            test_json_carries_stage_and_blocking_fields,
            test_cutover_window_and_post_cutover_are_rejected,
            test_pre_and_post_cutover_are_mutually_exclusive,
            test_report_marks_advisory_checks):
        _fn()
        print("PASS %s" % _fn.__name__)
    print("RESULT: PASS (phase3 preflight smoke)")
