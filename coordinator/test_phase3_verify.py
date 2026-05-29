"""Smoke + unit tests for phase3_verify.py.

Coverage:
  * Pure helpers (_classify_commits, _find_phase3_tick_evidence,
    _parse_systemd_environment, _diff_queue_against_db, _is_fresh,
    _journal_has_gate_line).
  * Each of the 7 post-cutover live checks exercised with a fake `ssh`
    that hands back canned outputs -- no subprocess, no live hub.
  * The run_verify entry point under --mock and --expect-cutover modes.

All tests are pure-Python pytest-style functions; can also be invoked
directly via `python3 test_phase3_verify.py`.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase3_verify as pv  # noqa: E402


# --------------------------------------------------------------------------
# Pure helper unit tests
# --------------------------------------------------------------------------

def test_classify_commits_writer_only():
    lines = [
        "phase3: 2 v3 result manifest(s) 2026-05-28",
        "phase3-queue: queue snapshot 2026-05-28",
        "phase3-heartbeats: 4 derived heartbeats 2026-05-28",
        "",
        "   ",
    ]
    writer, foreign = pv._classify_commits(lines)
    assert len(writer) == 3
    assert foreign == []


def test_classify_commits_finds_foreign():
    lines = [
        "phase3: 1 v3 result manifest 2026-05-28",
        "operator: hand fix queue file",
        "Merge pull request #99",
    ]
    writer, foreign = pv._classify_commits(lines)
    assert len(writer) == 1
    assert len(foreign) == 2
    assert "operator: hand fix queue file" in foreign


def test_find_phase3_tick_evidence_positive():
    j = (
        "Jan 01 idle\n"
        "[phase3] committed 3 manifest(s) (0 remaining in spool)\n"
        "Jan 01 idle\n"
    )
    ok, evidence, stub = pv._find_phase3_tick_evidence(j)
    assert ok is True
    assert "committed 3 manifest" in evidence
    assert stub is False


def test_find_phase3_tick_evidence_stub_only():
    j = (
        "Jan 01 boot\n"
        "[phase3] git writer stub (PHASE3_GIT_WRITER_READY=False); "
        "no git writes performed\n"
    )
    ok, evidence, stub = pv._find_phase3_tick_evidence(j)
    assert ok is False
    assert evidence is None
    assert stub is True


def test_find_phase3_tick_evidence_empty():
    ok, evidence, stub = pv._find_phase3_tick_evidence("")
    assert ok is False
    assert evidence is None
    assert stub is False


def test_parse_systemd_environment():
    out = (
        "Environment=PHASE3_DISABLE_RUNNER_RESULT_PUSH=1 "
        "PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH=1 OTHER=foo\n"
    )
    env = pv._parse_systemd_environment(out)
    assert env["PHASE3_DISABLE_RUNNER_RESULT_PUSH"] == "1"
    assert env["PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH"] == "1"
    assert env["OTHER"] == "foo"


def test_parse_systemd_environment_empty():
    assert pv._parse_systemd_environment("") == {}
    # malformed entries silently skipped
    env = pv._parse_systemd_environment("Environment=BAD-TOKEN\n")
    assert env == {}


def test_diff_queue_against_db_in_sync():
    items = [{"queue_id": "V3-EXQ-001"}, {"queue_id": "V3-EXQ-002"}]
    db = ["V3-EXQ-001", "V3-EXQ-002"]
    file_only, db_only = pv._diff_queue_against_db(items, db)
    assert file_only == []
    assert db_only == []


def test_diff_queue_against_db_drifted():
    items = [{"queue_id": "V3-EXQ-001"}, {"queue_id": "V3-EXQ-OLD"}]
    db = ["V3-EXQ-001", "V3-EXQ-NEW"]
    file_only, db_only = pv._diff_queue_against_db(items, db)
    assert file_only == ["V3-EXQ-OLD"]
    assert db_only == ["V3-EXQ-NEW"]


def test_diff_queue_against_db_ignores_garbage():
    items = [{"queue_id": "ok"}, {"not_queue_id": "x"}, 42, None]
    file_only, db_only = pv._diff_queue_against_db(items, ["ok"])
    assert file_only == []
    assert db_only == []


def test_is_fresh():
    assert pv._is_fresh(1000, 1200, 300) is True
    assert pv._is_fresh(1000, 2000, 300) is False
    assert pv._is_fresh("notanint", 2000, 300) is False
    # tolerate small clock skew (now slightly behind file time)
    assert pv._is_fresh(1003, 1000, 300) is True


def test_journal_has_gate_line():
    j = (
        "noise\n"
        "[runner] phase3 gate: skipping git_push_results (gate=1)\n"
        "more noise\n"
    )
    assert pv._journal_has_gate_line(j, "phase3 gate: skipping git_push_results")
    assert not pv._journal_has_gate_line(j, "no such marker")


# --------------------------------------------------------------------------
# Fake-ssh harness for live-check tests
# --------------------------------------------------------------------------

class FakeSSH:
    """Records calls and returns canned (ok, out, err) tuples by command
    substring match. Default fallback (ok, "", "no script") if nothing matches.
    """

    def __init__(self, script):
        # script: list of (substring, (ok, out, err))
        self.script = list(script)
        self.calls = []

    def __call__(self, host, user, cmd, *, dry_run=False, timeout=20):
        self.calls.append((host, user, cmd))
        for sub, resp in self.script:
            if sub in cmd:
                return resp
        return (False, "", "no canned response for: " + cmd[:80])


# --------------------------------------------------------------------------
# Per-check live tests
# --------------------------------------------------------------------------

def test_sync_daemon_phase3_tick_pass():
    j = (
        "[phase3] committed 2 manifest(s) (0 remaining in spool)\n"
    )
    ssh = FakeSSH([("journalctl -u ree-sync-daemon", (True, j, ""))])
    status, msg = pv.check_sync_daemon_phase3_tick(
        "10.8.0.1", "ree", ssh=ssh)
    assert status == "PASS", msg
    assert "committed 2" in msg


def test_sync_daemon_phase3_tick_fail_stub_only():
    j = (
        "[phase3] git writer stub (PHASE3_GIT_WRITER_READY=False); "
        "no git writes performed\n"
    )
    ssh = FakeSSH([("journalctl -u ree-sync-daemon", (True, j, ""))])
    status, msg = pv.check_sync_daemon_phase3_tick(
        "10.8.0.1", "ree", ssh=ssh)
    assert status == "FAIL"
    assert "stub" in msg


def test_sync_daemon_phase3_tick_fail_empty_journal():
    ssh = FakeSSH([("journalctl -u ree-sync-daemon", (True, "", ""))])
    status, msg = pv.check_sync_daemon_phase3_tick(
        "10.8.0.1", "ree", ssh=ssh)
    assert status == "FAIL"


def test_hub_git_writer_only_pass():
    log = "\n".join([
        "phase3-heartbeats: 4 derived heartbeats 2026-05-28",
        "phase3-queue: queue snapshot 2026-05-28",
        "phase3: 1 v3 result manifest(s) 2026-05-28",
    ])
    ssh = FakeSSH([("git -C", (True, log, ""))])
    status, msg = pv.check_hub_git_writer_only(
        "10.8.0.1", "ree", ssh=ssh, recent_n=3)
    assert status == "PASS", msg


def test_hub_git_writer_only_fail_foreign():
    log = "\n".join([
        "phase3: 1 v3 result manifest 2026-05-28",
        "operator: emergency hand fix 2026-05-28",
        "phase3-queue: snapshot 2026-05-28",
    ])
    ssh = FakeSSH([("git -C", (True, log, ""))])
    status, msg = pv.check_hub_git_writer_only(
        "10.8.0.1", "ree", ssh=ssh, recent_n=3)
    assert status == "FAIL"
    assert "operator" in msg


def test_workers_no_result_git_push_pass():
    show_out = (
        "Environment=PHASE3_DISABLE_RUNNER_RESULT_PUSH=1 "
        "PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH=1\n"
    )
    j = "[runner] phase3 gate: skipping git_push_results (gate=1)\n"
    ssh = FakeSSH([
        ("systemctl show ree-runner", (True, show_out, "")),
        ("journalctl -u ree-runner", (True, j, "")),
    ])
    # Use only cloud workers to avoid local-Mac SKIP path
    status, msg, detail = pv.check_workers_no_result_git_push(
        ["ree-cloud-1", "ree-cloud-2"], "ree", ssh=ssh)
    assert status == "PASS", msg
    assert "ree-cloud-1" in detail and "ree-cloud-2" in detail


def test_workers_no_result_git_push_fail_missing_env():
    show_out = "Environment=OTHER=foo\n"
    ssh = FakeSSH([("systemctl show ree-runner", (True, show_out, ""))])
    status, msg, _ = pv.check_workers_no_result_git_push(
        ["ree-cloud-1"], "ree", ssh=ssh)
    assert status == "FAIL"
    assert "PHASE3_DISABLE_RUNNER_RESULT_PUSH" in msg


def test_workers_no_result_git_push_warn_no_log_then_fail():
    """env OK but no recent gate-log line on ANY worker -> overall FAIL
    (no PASS evidence anywhere)."""
    show_out = "Environment=PHASE3_DISABLE_RUNNER_RESULT_PUSH=1\n"
    ssh = FakeSSH([
        ("systemctl show ree-runner", (True, show_out, "")),
        ("journalctl -u ree-runner", (True, "no relevant logs\n", "")),
    ])
    status, msg, _ = pv.check_workers_no_result_git_push(
        ["ree-cloud-1"], "ree", ssh=ssh)
    assert status == "FAIL", "no PASS anywhere should be promoted to FAIL"


def test_heartbeat_git_retired_pass():
    show_out = (
        "Environment=PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH=1\n"
    )
    j = (
        "[remote-control] phase3 gate active: heartbeat + commands "
        "git push suppressed\n"
    )
    ssh = FakeSSH([
        ("systemctl show ree-runner", (True, show_out, "")),
        ("journalctl -u ree-runner", (True, j, "")),
    ])
    status, msg, detail = pv.check_heartbeat_git_retired(
        ["ree-cloud-1"], "ree", ssh=ssh)
    assert status == "PASS", msg
    assert detail["ree-cloud-1"].startswith("PASS")


def test_heartbeat_git_retired_fail_env_missing():
    ssh = FakeSSH([("systemctl show ree-runner",
                    (True, "Environment=\n", ""))])
    status, msg, _ = pv.check_heartbeat_git_retired(
        ["ree-cloud-1"], "ree", ssh=ssh)
    assert status == "FAIL"
    assert "PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH" in msg


def test_results_drained_pass():
    ssh = FakeSSH([("sqlite3 -batch", (True, "0\n", ""))])
    status, msg = pv.check_results_drained("10.8.0.1", "ree", ssh=ssh)
    assert status == "PASS"


def test_results_drained_fail_pending():
    ssh = FakeSSH([("sqlite3 -batch", (True, "7\n", ""))])
    status, msg = pv.check_results_drained("10.8.0.1", "ree", ssh=ssh)
    assert status == "FAIL"
    assert "7" in msg


def test_results_drained_fail_unparseable():
    ssh = FakeSSH([("sqlite3 -batch", (True, "weird\n", ""))])
    status, msg = pv.check_results_drained("10.8.0.1", "ree", ssh=ssh)
    assert status == "FAIL"


def test_queue_snapshot_fresh_pass():
    qjson = json.dumps({"items": [
        {"queue_id": "V3-EXQ-001"}, {"queue_id": "V3-EXQ-002"}]})
    db = "V3-EXQ-001\nV3-EXQ-002\n"
    ssh = FakeSSH([
        ("git -C", (True, qjson, "")),
        ("sqlite3 -batch", (True, db, "")),
    ])
    status, msg = pv.check_queue_snapshot_fresh(
        "10.8.0.1", "ree", ssh=ssh)
    assert status == "PASS", msg


def test_queue_snapshot_fresh_fail_drift():
    qjson = json.dumps({"items": [{"queue_id": "V3-EXQ-OLD"}]})
    db = "V3-EXQ-NEW\n"
    ssh = FakeSSH([
        ("git -C", (True, qjson, "")),
        ("sqlite3 -batch", (True, db, "")),
    ])
    status, msg = pv.check_queue_snapshot_fresh(
        "10.8.0.1", "ree", ssh=ssh)
    assert status == "FAIL"
    assert "V3-EXQ-OLD" in msg or "V3-EXQ-NEW" in msg


def test_queue_snapshot_fresh_fail_bad_json():
    ssh = FakeSSH([("git -C", (True, "not json", ""))])
    status, msg = pv.check_queue_snapshot_fresh(
        "10.8.0.1", "ree", ssh=ssh)
    assert status == "FAIL"


def test_derived_heartbeats_pass():
    now = 2_000_000
    ssh = FakeSSH([
        ("--format=%ct origin/master --",
         (True, str(now - 60) + "\n", "")),
        ("--format=%s origin/master --",
         (True, "phase3-heartbeats: 4 files 2026-05-28\n", "")),
    ])
    status, msg = pv.check_derived_heartbeats(
        "10.8.0.1", "ree", ssh=ssh, lookback_sec=600, now=now)
    assert status == "PASS", msg


def test_derived_heartbeats_fail_stale():
    now = 2_000_000
    ssh = FakeSSH([
        ("--format=%ct origin/master --",
         (True, str(now - 99999) + "\n", "")),
        ("--format=%s origin/master --",
         (True, "phase3-heartbeats: stale\n", "")),
    ])
    status, msg = pv.check_derived_heartbeats(
        "10.8.0.1", "ree", ssh=ssh, lookback_sec=600, now=now)
    assert status == "FAIL"
    assert "old" in msg


def test_derived_heartbeats_fail_foreign_commit():
    now = 2_000_000
    log_subjects = (
        "phase3-heartbeats: 4 files 2026-05-28\n"
        "operator: hand-edited runner_status\n"
    )
    ssh = FakeSSH([
        ("--format=%ct origin/master --",
         (True, str(now - 60) + "\n", "")),
        ("--format=%s origin/master --",
         (True, log_subjects, "")),
    ])
    status, msg = pv.check_derived_heartbeats(
        "10.8.0.1", "ree", ssh=ssh, lookback_sec=600, now=now)
    assert status == "FAIL"
    assert "operator" in msg


# --------------------------------------------------------------------------
# CLI + run_verify wiring tests
# --------------------------------------------------------------------------

def test_help_exits_zero():
    script = HERE / "phase3_verify.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=10, check=False)
    assert proc.returncode == 0
    assert "Phase 3 post-cutover" in proc.stdout


def test_mock_json_structure_and_skip():
    script = HERE / "phase3_verify.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--mock", "--json"],
        capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout)
    ids = {c["id"] for c in data["checks"]}
    # All 7 stub-replaced checks must appear...
    for cid in ("sync_daemon_phase3_tick", "hub_git_writer_only",
                "workers_no_result_git_push", "heartbeat_git_retired",
                "results_drained", "queue_snapshot_fresh",
                "derived_heartbeats"):
        assert cid in ids, cid
    # ...and all SKIP under --mock.
    by_id = {c["id"]: c["status"] for c in data["checks"]}
    for cid in ("sync_daemon_phase3_tick", "hub_git_writer_only",
                "workers_no_result_git_push", "heartbeat_git_retired",
                "results_drained", "queue_snapshot_fresh",
                "derived_heartbeats"):
        assert by_id[cid] == "SKIP", "%s: %s" % (cid, by_id[cid])
    assert data["ok"] is True


def test_run_verify_expect_cutover_uses_ssh():
    """With expect_cutover=True and a stubbed ssh_runner, the 7 checks
    fire and the canned PASS data flows through to PASS verdicts."""
    # Build a fake_ssh that satisfies every check.
    qjson = json.dumps({"items": [{"queue_id": "V3-EXQ-001"}]})
    show_out = (
        "Environment=PHASE3_DISABLE_RUNNER_RESULT_PUSH=1 "
        "PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH=1\n"
    )
    runner_journal = (
        "[runner] phase3 gate: skipping git_push_results (gate=1)\n"
        "[remote-control] phase3 gate active: heartbeat + commands "
        "git push suppressed\n"
    )
    daemon_journal = (
        "[phase3] committed 1 manifest(s) (0 remaining in spool)\n"
    )
    asm_log = "phase3: 1 v3 result manifest 2026-05-28\n"
    hb_log = "phase3-heartbeats: 4 files 2026-05-28\n"

    def fake_ssh(host, user, cmd, *, dry_run=False, timeout=20):
        # Hub SYNC_MODE check
        if "SYNC_MODE=" in cmd:
            return True, "SYNC_MODE=authoritative\n", ""
        if "journalctl -u ree-sync-daemon" in cmd:
            return True, daemon_journal, ""
        if "git -C %s log" % pv.HUB_REE_ASSEMBLY in cmd \
                and "evidence/experiments" in cmd \
                and "--format=%ct" in cmd:
            # derived_heartbeats freshness
            import time
            return True, str(int(time.time()) - 30) + "\n", ""
        if "git -C %s log" % pv.HUB_REE_ASSEMBLY in cmd \
                and "evidence/experiments" in cmd \
                and "--format=%s" in cmd:
            return True, hb_log, ""
        if "git -C %s" % pv.HUB_REE_ASSEMBLY in cmd \
                and "--format=%s" in cmd:
            # hub_git_writer_only
            return True, asm_log, ""
        if "git -C %s" % pv.HUB_REE_V3 in cmd and "show" in cmd:
            return True, qjson, ""
        if "sqlite3 -batch" in cmd and "results" in cmd:
            return True, "0\n", ""
        if "sqlite3 -batch" in cmd and "experiments" in cmd:
            return True, "V3-EXQ-001\n", ""
        if "systemctl show ree-runner" in cmd:
            return True, show_out, ""
        if "journalctl -u ree-runner" in cmd:
            return True, runner_journal, ""
        return True, "", ""

    summary = pv.run_verify(
        env_file=Path("/dev/null"),
        dry_run=False,
        expect_cutover=True,
        mock=False,
        ssh_runner=fake_ssh,
    )
    by_id = {c["id"]: c["status"] for c in summary["checks"]}
    # All 7 substantive checks PASS (Mac SKIP within worker checks is OK).
    for cid in ("sync_daemon_phase3_tick", "hub_git_writer_only",
                "workers_no_result_git_push", "heartbeat_git_retired",
                "results_drained", "queue_snapshot_fresh",
                "derived_heartbeats", "hub_sync_mode_authoritative"):
        assert by_id[cid] == "PASS", "%s: %s" % (cid, by_id[cid])


if __name__ == "__main__":
    # Direct invocation: run every test_* in this module and report.
    import inspect
    names = [n for n, fn in list(globals().items())
             if n.startswith("test_") and inspect.isfunction(fn)]
    failed = 0
    for n in names:
        try:
            globals()[n]()
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (n, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %s: %r" % (n, exc))
        else:
            print("PASS %s" % n)
    if failed:
        print("RESULT: FAIL (%d/%d)" % (failed, len(names)))
        sys.exit(1)
    print("RESULT: PASS (%d tests)" % len(names))
