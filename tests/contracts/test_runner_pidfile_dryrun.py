"""Contract: `experiment_runner.py --dry-run` must never touch runner.pid.

Incident class (found 2026-09-26, REE_assembly
evidence/planning/runner_multislot_design_spike_20260926.md sec 8): main()
wrote runner.pid unconditionally before the --dry-run early-return, and the
dry-run branch then unlinked it. A dry run on a box whose runner was live
therefore DELETED the live runner's pid file -- and REE_assembly/serve.py
(_runner_pid) reads that file to decide whether a runner is up.

  C1. A --dry-run invocation leaves a pre-existing runner.pid byte-identical.
  C2. A --dry-run invocation with no runner.pid present does not create one.
  C3. _release_pid_file() removes the pid file only when it records THIS
      process's pid -- a foreign pid (a live runner, or a newer instance) is
      left alone. This is what makes the signal-exit path safe too: a SIGINT
      during a dry run goes through _do_immediate_exit().

Every test points PID_FILE / status / queue at tmp_path; the real
ree-v3/runner.pid is never read or written.
"""
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

REE_V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REE_V3))

import experiment_runner as er  # noqa: E402

FOREIGN_PID = b"424242"  # stands in for a live runner's pid


@pytest.fixture
def isolated_runner(tmp_path, monkeypatch):
    pid_file = tmp_path / "runner.pid"
    monkeypatch.setattr(er, "PID_FILE", pid_file)
    monkeypatch.setattr(er, "SCRIPT_TIMING_FILE", tmp_path / "script_timing.json")
    monkeypatch.setattr(er, "load_queue",
                        lambda: {"items": [], "calibration": {}})
    monkeypatch.setattr(er, "find_ree_assembly_path", lambda: None)
    monkeypatch.setattr(
        er, "merge_peer_status",
        lambda status_path: er.PeerStatusResult(set(), True, "none"))
    monkeypatch.setattr(sys, "argv", [
        "experiment_runner.py", "--dry-run", "--skip-preflight",
        "--status-file", str(tmp_path / "status" / "test-box.json"),
        "--machine", "test-box", "--no-laptop-yield-to-cloud",
    ])
    (tmp_path / "status").mkdir()
    # main() installs SIGINT/SIGTERM handlers; restore the originals.
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    yield pid_file
    for s, h in saved.items():
        signal.signal(s, h)


def test_dry_run_leaves_live_runner_pid_byte_identical(isolated_runner):
    pid_file = isolated_runner
    pid_file.write_bytes(FOREIGN_PID)
    er.main()
    assert pid_file.exists(), "--dry-run deleted a live runner's runner.pid"
    assert pid_file.read_bytes() == FOREIGN_PID, (
        "--dry-run overwrote a live runner's runner.pid")


def test_dry_run_does_not_create_pid_file(isolated_runner):
    pid_file = isolated_runner
    er.main()
    assert not pid_file.exists()


def test_release_pid_file_only_removes_own_pid(tmp_path, monkeypatch):
    pid_file = tmp_path / "runner.pid"
    monkeypatch.setattr(er, "PID_FILE", pid_file)

    pid_file.write_bytes(FOREIGN_PID)
    er._release_pid_file()
    assert pid_file.read_bytes() == FOREIGN_PID

    pid_file.write_text(str(os.getpid()))
    er._release_pid_file()
    assert not pid_file.exists()

    er._release_pid_file()  # absent file: no error
