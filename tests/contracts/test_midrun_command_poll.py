"""Contract tests for the mid-run remote-command poll thread
(chip-20260801-runner-mid-run-command-poll).

Background -- confirmed incident, V3-EXQ-858 (2026-08-01): a remote
`suspend` command sat unacknowledged for 6+ hours against a live 1200-minute
experiment. Root cause: `runner_remote_control.process_pending_commands()`
was only ever called from the TOP of experiment_runner.py's outer pass loop,
before claiming the next queue item -- never while `run_experiment()`'s
subprocess-monitor loop (`for line in proc.stdout:`) is blocking on the
CURRENTLY running experiment. Neither the git-file channel nor the
coordinator channel could reach an already-running experiment; remote
`force_stop` could not actually interrupt one either, despite the runner's
own inline comment at experiment_runner.py claiming otherwise.

Fix: a 4th daemon thread, `_command_poll`, nested inside `run_experiment()`
exactly like the existing `_heartbeat` / `_background_sync` threads (see
tests/contracts/test_daemon_thread_loop_guards.py, which pins the set of
four). It polls `process_pending_commands()` every `COMMAND_POLL_INTERVAL`
seconds WHILE the subprocess is running, restricted via a new `only_kinds`
parameter (runner_remote_control.py) to exactly {"suspend", "force_stop"} --
the only two command kinds whose entire purpose is to interrupt something
happening right now. Every other kind (stop/pause/resume/resume_run/kick/
release_claim/reclassify) is left pending for the existing unfiltered
top-of-pass call, completely unchanged -- deliberately, because e.g. a
`release_claim` for the CURRENTLY RUNNING item executing early would open a
real duplicate-run hazard (another machine claiming the same item), not
merely shift when a command is acknowledged.

Gated off by default via RUNNER_MIDRUN_COMMAND_POLL_ENABLED (unset/false):
run_experiment() does not create the thread at all in that case, so a normal
run is bit-identical to before this chip -- same thread count, same timing.
The flag is landed OFF everywhere in this chip; enabling it on any live
worker is a deliberate, separate human rollout step.

Layers tested, from safest/most isolated to most integrated:

  A. runner_remote_control.process_pending_commands()'s new `only_kinds`
     filter, directly -- no threading, no experiment_runner involved. This
     is the layer that actually prevents the duplicate-run hazard above: a
     "stop" command mixed into the same pending batch as a "suspend" must
     come out of a filtered call completely untouched (still "pending"),
     while the "suspend" is executed. Both git-file and coordinator
     channels.
  B. `_midrun_command_poll_gated()` -- the env-var no-op-by-default gate.
  C. `_command_poll`'s loop body, compiled and executed against stand-ins
     (mirrors the existing C18 harness for `_delayed_pulls`): the closure is
     nested inside run_experiment() and not reachable for monkeypatching, so
     this is the established idiom in this file for testing it directly.
  D. Full `run_experiment()` end-to-end, with `subprocess.Popen` monkeypatched
     to a fake process and `threading.Thread` wrapped to record which
     targets get constructed. This is the only layer that actually proves
     "no observable difference when disabled" and "suspend mid-run
     terminates the subprocess promptly" at the level a live worker
     experiences it.
"""

from __future__ import annotations

import json
import subprocess as _subprocess_module
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import experiment_runner  # noqa: E402
import runner_remote_control as rrc  # noqa: E402


# ---------------------------------------------------------------------------
# Layer A: only_kinds filtering in process_pending_commands (pure, no threads)
# ---------------------------------------------------------------------------


def _write_pending(tmp_path: Path, machine: str, cmds: list[dict]) -> None:
    data = {
        "schema_version": "v1",
        "machine": machine,
        "commands": [
            {
                "id": f"cmd-{i}",
                "kind": kind,
                "args": {},
                "issued_at_utc": "2026-08-02T00:00:00Z",
                "issued_by": "test",
                "status": "pending",
                "ack_at_utc": None,
                "completed_at_utc": None,
                "error": None,
                "result_note": None,
            }
            for i, kind in enumerate(cmds)
        ],
    }
    path = rrc.commands_path(tmp_path, machine)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _read_statuses(tmp_path: Path, machine: str) -> dict[str, str]:
    data = rrc.read_commands_file(tmp_path, machine)
    return {c["kind"]: c["status"] for c in data["commands"]}


def _base_kwargs(queue_file: Path) -> dict:
    return dict(
        drain_flag=[],
        pause_flag=[],
        force_stop_flag=[],
        suspend_flag=[],
        resume_run_target=[],
        current_proc=[],
        auto_sync=False,
    )


def test_a1_git_channel_only_kinds_leaves_other_commands_untouched(tmp_path):
    """A 'stop' command mixed with a 'suspend' must survive a filtered call
    completely unprocessed -- still status=pending, drain_flag untouched.
    This is the property that makes the mid-run poll thread safe to call
    process_pending_commands() from: nothing it doesn't understand gets
    silently marked done-but-ineffective.
    """
    machine = "test-machine-a1"
    _write_pending(tmp_path, machine, ["stop", "suspend"])
    kwargs = _base_kwargs(tmp_path)
    kwargs["current_proc"] = [mock.Mock()]  # so suspend has an active proc

    processed = rrc.process_pending_commands(
        tmp_path, machine, tmp_path / "experiment_queue.json",
        only_kinds=frozenset({"suspend", "force_stop"}),
        **kwargs,
    )

    assert [p["kind"] for p in processed] == ["suspend"]
    statuses = _read_statuses(tmp_path, machine)
    assert statuses["suspend"] == "done"
    assert statuses["stop"] == "pending", (
        "a command outside only_kinds must be left exactly as found, not "
        "acked or executed -- acking-without-executing would silently drop "
        "it forever (git channel only considers status=='pending')"
    )
    assert kwargs["drain_flag"] == [], "the un-executed stop must not set drain_flag"
    assert kwargs["suspend_flag"] == [True]


def test_a1b_stop_command_is_processed_by_a_later_unfiltered_call(tmp_path):
    """The 'stop' left pending by A1 must still be reachable -- proving it
    was deferred, not lost. Mirrors the real flow: the poll thread's
    filtered call now, the outer loop's unfiltered call once run_experiment
    returns."""
    machine = "test-machine-a1b"
    _write_pending(tmp_path, machine, ["stop", "suspend"])
    kwargs = _base_kwargs(tmp_path)
    kwargs["current_proc"] = [mock.Mock()]
    rrc.process_pending_commands(
        tmp_path, machine, tmp_path / "experiment_queue.json",
        only_kinds=frozenset({"suspend", "force_stop"}), **kwargs)

    # Second, unfiltered call (the real top-of-pass call shape).
    kwargs2 = _base_kwargs(tmp_path)
    processed2 = rrc.process_pending_commands(
        tmp_path, machine, tmp_path / "experiment_queue.json",
        only_kinds=None, **kwargs2)

    assert [p["kind"] for p in processed2] == ["stop"]
    assert kwargs2["drain_flag"] == [True]
    assert _read_statuses(tmp_path, machine)["stop"] == "done"


def test_a2_only_kinds_none_is_bit_identical_to_pre_filter_default(tmp_path):
    """only_kinds=None (the default for every EXISTING call site) processes
    every pending command, exactly as before this chip."""
    machine = "test-machine-a2"
    _write_pending(tmp_path, machine, ["stop", "pause", "suspend"])
    kwargs = _base_kwargs(tmp_path)
    kwargs["current_proc"] = [mock.Mock()]

    processed = rrc.process_pending_commands(
        tmp_path, machine, tmp_path / "experiment_queue.json",
        **kwargs)  # only_kinds omitted -> None

    assert {p["kind"] for p in processed} == {"stop", "pause", "suspend"}
    statuses = _read_statuses(tmp_path, machine)
    assert all(s == "done" for s in statuses.values())


def test_a3_coordinator_channel_only_kinds_leaves_unmatched_unacked(monkeypatch, tmp_path):
    """Coordinator channel: a command outside only_kinds must never be
    fetched-and-acked -- it simply never appears in `processed`, so the
    coordinator (which the test simulates) still shows it pending and will
    redeliver it on a later unfiltered fetch."""
    fetched = {
        "commands": [
            {"id": "c-stop", "kind": "stop", "args": "{}"},
            {"id": "c-force", "kind": "force_stop", "args": "{}"},
        ]
    }
    acked_ids = []

    monkeypatch.setattr(rrc, "_phase3_commands_via_coordinator_gated", lambda: True)
    monkeypatch.setattr(rrc, "_phase3_commands_off_git_gated", lambda: False)
    monkeypatch.setattr(
        rrc.coordinator_client, "fetch_commands", lambda machine: fetched)
    monkeypatch.setattr(
        rrc.coordinator_client, "ack_command",
        lambda cmd_id, machine, status, note: acked_ids.append(cmd_id) or True)

    machine = "test-machine-a3"
    kwargs = _base_kwargs(tmp_path)
    kwargs["current_proc"] = [mock.Mock()]

    processed = rrc.process_pending_commands(
        tmp_path, machine, tmp_path / "experiment_queue.json",
        only_kinds=frozenset({"suspend", "force_stop"}), **kwargs)

    assert [p["kind"] for p in processed] == ["force_stop"]
    assert acked_ids == ["c-force"], "the filtered-out 'stop' must never be acked"


def test_a4_force_stop_kills_current_proc_directly(tmp_path):
    """Pin the existing (pre-chip) behaviour this fix depends on: force_stop
    already kills the live subprocess synchronously inside _execute_command.
    The chip's whole contribution for force_stop is making sure this
    function gets CALLED while a run is in progress -- it does not need to
    duplicate the kill."""
    machine = "test-machine-a4"
    _write_pending(tmp_path, machine, ["force_stop"])
    fake_proc = mock.Mock()
    kwargs = _base_kwargs(tmp_path)
    kwargs["current_proc"] = [fake_proc]

    rrc.process_pending_commands(
        tmp_path, machine, tmp_path / "experiment_queue.json",
        only_kinds=frozenset({"suspend", "force_stop"}), **kwargs)

    fake_proc.kill.assert_called_once()
    assert kwargs["force_stop_flag"] == [True]
    assert kwargs["drain_flag"] == [True], "force_stop also requests drain"


def test_a5_suspend_sets_flag_but_does_not_terminate_itself(tmp_path):
    """Pin the existing split this fix relies on: suspend only SETS
    suspend_flag; _execute_command does NOT call terminate()/kill() for it.
    (The stdout-reading loop -- or, with this chip, the poll thread itself
    -- is responsible for the actual proc.terminate() call.) If this ever
    changes, _command_poll's own terminate() call becomes redundant rather
    than load-bearing, which is safe, but the loop-body test in layer C
    below assumes this split holds."""
    machine = "test-machine-a5"
    _write_pending(tmp_path, machine, ["suspend"])
    fake_proc = mock.Mock()
    kwargs = _base_kwargs(tmp_path)
    kwargs["current_proc"] = [fake_proc]

    rrc.process_pending_commands(
        tmp_path, machine, tmp_path / "experiment_queue.json",
        only_kinds=frozenset({"suspend", "force_stop"}), **kwargs)

    fake_proc.terminate.assert_not_called()
    fake_proc.kill.assert_not_called()
    assert kwargs["suspend_flag"] == [True]


# ---------------------------------------------------------------------------
# Layer B: the no-op-by-default gate
# ---------------------------------------------------------------------------


def test_b1_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("RUNNER_MIDRUN_COMMAND_POLL_ENABLED", raising=False)
    assert experiment_runner._midrun_command_poll_gated() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
def test_b2_gate_on_when_enabled(monkeypatch, value):
    monkeypatch.setenv("RUNNER_MIDRUN_COMMAND_POLL_ENABLED", value)
    assert experiment_runner._midrun_command_poll_gated() is True


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_b3_gate_off_for_falsy_values(monkeypatch, value):
    monkeypatch.setenv("RUNNER_MIDRUN_COMMAND_POLL_ENABLED", value)
    assert experiment_runner._midrun_command_poll_gated() is False


# ---------------------------------------------------------------------------
# Layer C: _command_poll's loop body, compiled+executed against stand-ins
# (mirrors test_daemon_thread_loop_guards.py's C18 harness for _delayed_pulls)
# ---------------------------------------------------------------------------


def _find_command_poll_fn():
    import ast
    src = Path(experiment_runner.__file__).read_text(encoding="utf-8")
    found = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.FunctionDef) and n.name == "_command_poll"]
    assert len(found) == 1
    return ast.Module(body=[found[0]], type_ignores=[])


def _exec_command_poll(*, process_pending_commands, suspend_flag, proc_ref,
                        n_ticks=1):
    """Compile+exec the real `_command_poll` source against stand-ins for
    every free variable its closure captures. `_hb_stop` is a real Event
    pre-loaded to fire after `n_ticks` iterations (via a side-effecting wait
    stand-in), so the loop runs a bounded, deterministic number of times
    without a real timer."""
    import ast
    module = _find_command_poll_fn()
    ast.fix_missing_locations(module)

    calls = {"n": 0}

    class _Event:
        def wait(self, timeout=None):
            calls["n"] += 1
            return calls["n"] > n_ticks  # True (stop) once ticks exhausted

    logged: list[str] = []
    ns = {
        "_hb_stop": _Event(),
        "COMMAND_POLL_INTERVAL": 0,
        "_rrc": type("R", (), {"process_pending_commands": staticmethod(
            process_pending_commands)}),
        "ree_assembly_path": Path("/nonexistent/REE_assembly"),
        "machine": "test-machine",
        "QUEUE_FILE": Path("/nonexistent/experiment_queue.json"),
        "auto_sync": False,
        "status": {},
        "status_path": Path("/nonexistent/status.json"),
        "write_status": lambda *a, **k: None,
        "_MIDRUN_COMMAND_POLL_KINDS": experiment_runner._MIDRUN_COMMAND_POLL_KINDS,
        "_cmdpoll_drain_flag": [],
        "_cmdpoll_pause_flag": [],
        "_cmdpoll_force_stop_flag": [],
        "_cmdpoll_resume_run_target": [],
        "_cmdpoll_suspend_flag": suspend_flag,
        "_cmdpoll_proc_ref": proc_ref,
        "print": lambda *a, **k: logged.append(" ".join(str(x) for x in a)),
    }
    exec(compile(module, "<_command_poll>", "exec"), ns)
    ns["_command_poll"]()
    return logged, calls["n"]


def test_c1_happy_path_no_pending_command_no_terminate_call():
    """Nothing pending: process_pending_commands is called (that's the
    thread's whole job), but with no command it is a pure no-op -- no
    terminate(), no exception, no log line."""
    seen_calls = []

    def fake_ppc(*a, **kw):
        seen_calls.append(kw)

    proc_ref = [mock.Mock()]
    logged, n = _exec_command_poll(
        process_pending_commands=fake_ppc, suspend_flag=[], proc_ref=proc_ref,
        n_ticks=1)

    assert n == 2  # one real tick, then the stop-check tick
    assert len(seen_calls) == 1
    assert seen_calls[0]["only_kinds"] == experiment_runner._MIDRUN_COMMAND_POLL_KINDS
    proc_ref[0].terminate.assert_not_called()
    assert logged == []


def test_c2_suspend_flag_set_after_call_triggers_terminate():
    """The core mid-run fix: when process_pending_commands leaves
    suspend_flag populated (mimicking a real suspend command having just
    been executed), _command_poll must itself call proc.terminate() --
    it cannot wait for the (possibly blocked) stdout-reading loop."""
    suspend_flag = []

    def fake_ppc(*a, **kw):
        suspend_flag.append(True)  # mimic _execute_command's suspend effect

    proc_ref = [mock.Mock()]
    logged, n = _exec_command_poll(
        process_pending_commands=fake_ppc, suspend_flag=suspend_flag,
        proc_ref=proc_ref, n_ticks=1)

    proc_ref[0].terminate.assert_called_once()
    proc_ref[0].kill.assert_not_called()
    assert logged == []


def test_c3_no_current_proc_suspend_flag_set_does_not_raise():
    """Defensive: if suspend_flag is set but proc_ref is empty (already
    cleared, e.g. by the finally-block race at the very end of a run), the
    guard `if _cmdpoll_suspend_flag and _cmdpoll_proc_ref:` must skip the
    terminate() call rather than raising IndexError."""
    logged, n = _exec_command_poll(
        process_pending_commands=lambda *a, **kw: None,
        suspend_flag=[True], proc_ref=[], n_ticks=1)
    assert logged == []  # no exception surfaced as a warning either


def test_c4_process_pending_commands_raising_does_not_kill_the_thread():
    """Per-iteration guard, same shape as C17 pins structurally for all four
    daemon threads: one escaped exception must cost exactly one tick, not
    the rest of the poll loop's life."""
    call_count = {"n": 0}

    def flaky_ppc(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError(107, "Transport endpoint is not connected")

    proc_ref = [mock.Mock()]
    logged, n = _exec_command_poll(
        process_pending_commands=flaky_ppc, suspend_flag=[], proc_ref=proc_ref,
        n_ticks=2)

    assert call_count["n"] == 2, "the second tick must still run after the first raises"
    assert len(logged) == 1 and "command poll tick warn" in logged[0]


# ---------------------------------------------------------------------------
# Layer D: full run_experiment() end-to-end, fake subprocess + fake threads
# ---------------------------------------------------------------------------


class _FakeStdout:
    """Stands in for `proc.stdout`. Blocks in small increments (checking
    terminated/killed flags each time) to simulate a real subprocess whose
    pipe only closes once it is actually signalled -- exactly the "silent
    for a long time" shape that made the original bug invisible to the
    inline `for line in proc.stdout:` suspend-check. Gives up after a
    bounded number of increments so a test can never hang even if a bug
    means the signal never arrives."""

    def __init__(self, proc, lines=None, max_wait_s=3.0, step_s=0.01):
        self._proc = proc
        self._lines = list(lines or [])
        self._max_ticks = int(max_wait_s / step_s)
        self._step_s = step_s

    def __iter__(self):
        return self

    def __next__(self):
        if self._lines:
            return self._lines.pop(0)
        for _ in range(self._max_ticks):
            if self._proc.terminated.is_set() or self._proc.killed.is_set():
                raise StopIteration
            time.sleep(self._step_s)
        raise StopIteration


class FakeProc:
    def __init__(self, lines=None):
        self.pid = 4242
        self.returncode = 0
        self.terminated = threading.Event()
        self.killed = threading.Event()
        self.stdout = _FakeStdout(self, lines=lines)

    def terminate(self):
        self.terminated.set()
        self.returncode = -15

    def kill(self):
        self.killed.set()
        self.returncode = -9

    def wait(self):
        return self.returncode


class _RecordingThread(threading.Thread):
    """Real threading.Thread subclass (so start/join/daemon all behave
    normally); just also records the target function's __name__ so tests
    can assert which daemon threads a run actually constructed, without
    relying on default thread naming."""

    created_targets: list[str] = []

    def __init__(self, *a, **kw):
        target = kw.get("target")
        _RecordingThread.created_targets.append(
            getattr(target, "__name__", repr(target)))
        super().__init__(*a, **kw)


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    """A minimal, isolated environment for a real run_experiment() call:
    an ree_assembly_path tmp tree, a status dict/path, and a queue item
    pointing at a script path that is never actually executed (Popen is
    monkeypatched)."""
    ree_assembly_path = tmp_path / "REE_assembly"
    ree_assembly_path.mkdir()
    status_path = tmp_path / "runner_status.json"
    status = {"completed": [], "queue": [{"queue_id": "TEST-EXQ-1", "status": "pending"}]}
    item = {
        "queue_id": "TEST-EXQ-1",
        "script": "experiments/does_not_run.py",
        "title": "test",
        "args": [],
        "estimated_minutes": 1,
    }
    _RecordingThread.created_targets = []
    monkeypatch.setattr(experiment_runner.threading, "Thread", _RecordingThread)
    monkeypatch.setattr(experiment_runner, "COMMAND_POLL_INTERVAL", 0.02)
    return dict(item=item, status=status, status_path=status_path,
                ree_assembly_path=ree_assembly_path)


def test_d1_disabled_by_default_no_command_poll_thread(monkeypatch, runner_env):
    """The headline safety requirement: with the flag unset, run_experiment()
    must not create a _command_poll thread at all -- not a gated no-op body,
    an absent thread."""
    monkeypatch.delenv("RUNNER_MIDRUN_COMMAND_POLL_ENABLED", raising=False)
    fake_proc = FakeProc(lines=["hello\n", "verdict: PASS\n"])
    monkeypatch.setattr(experiment_runner.subprocess, "Popen", lambda *a, **k: fake_proc)

    experiment_runner.run_experiment(
        runner_env["item"], runner_env["status"], runner_env["status_path"], {},
        remote_control=True, proc_ref=[],
        ree_assembly_path=runner_env["ree_assembly_path"],
        machine="test-machine-d1",
    )

    assert "_command_poll" not in _RecordingThread.created_targets
    assert "_heartbeat" in _RecordingThread.created_targets, (
        "sanity check: the harness itself is wired correctly and the "
        "existing heartbeat thread is unaffected"
    )


def test_d2_enabled_but_remote_control_off_no_command_poll_thread(monkeypatch, runner_env):
    """The flag alone is not enough -- matches the existing preconditions
    _push_remote_heartbeat/_background_sync already require."""
    monkeypatch.setenv("RUNNER_MIDRUN_COMMAND_POLL_ENABLED", "1")
    fake_proc = FakeProc(lines=["verdict: PASS\n"])
    monkeypatch.setattr(experiment_runner.subprocess, "Popen", lambda *a, **k: fake_proc)

    experiment_runner.run_experiment(
        runner_env["item"], runner_env["status"], runner_env["status_path"], {},
        remote_control=False, proc_ref=[],
        ree_assembly_path=runner_env["ree_assembly_path"],
        machine="test-machine-d2",
    )

    assert "_command_poll" not in _RecordingThread.created_targets


def test_d3_enabled_and_wired_creates_command_poll_thread_and_joins_cleanly(
        monkeypatch, runner_env):
    monkeypatch.setenv("RUNNER_MIDRUN_COMMAND_POLL_ENABLED", "1")
    fake_proc = FakeProc(lines=["verdict: PASS\n"])
    monkeypatch.setattr(experiment_runner.subprocess, "Popen", lambda *a, **k: fake_proc)
    # No commands file exists -> process_pending_commands (real function) is
    # a fast, harmless no-op each tick.
    experiment_runner.run_experiment(
        runner_env["item"], runner_env["status"], runner_env["status_path"], {},
        remote_control=True, proc_ref=[],
        ree_assembly_path=runner_env["ree_assembly_path"],
        machine="test-machine-d3",
    )
    assert "_command_poll" in _RecordingThread.created_targets


def test_d4_suspend_command_mid_run_terminates_promptly(monkeypatch, runner_env):
    """End-to-end reproduction of the fix: a subprocess that emits nothing
    for a long time (the exact shape of the confirmed incident) is
    interrupted within about one poll interval of a suspend command
    appearing, not left to run indefinitely."""
    monkeypatch.setenv("RUNNER_MIDRUN_COMMAND_POLL_ENABLED", "1")
    fake_proc = FakeProc(lines=[])  # silent from the start
    monkeypatch.setattr(experiment_runner.subprocess, "Popen", lambda *a, **k: fake_proc)

    # Real git-file channel: append a genuine pending suspend command.
    rrc.append_command(runner_env["ree_assembly_path"], "test-machine-d4",
                        "suspend", issued_by="test")

    started = time.monotonic()
    experiment_runner.run_experiment(
        runner_env["item"], runner_env["status"], runner_env["status_path"], {},
        remote_control=True, proc_ref=[],
        ree_assembly_path=runner_env["ree_assembly_path"],
        machine="test-machine-d4",
    )
    elapsed = time.monotonic() - started

    assert fake_proc.terminated.is_set(), (
        "suspend must reach the live subprocess without waiting for it to "
        "produce output or finish on its own"
    )
    assert elapsed < 2.0, (
        f"suspend took {elapsed:.2f}s to take effect against a 0.02s poll "
        "interval -- should be near-immediate, not anywhere close to the "
        "confirmed 6+ hour incident shape"
    )


def test_d5_force_stop_command_mid_run_kills_promptly(monkeypatch, runner_env):
    monkeypatch.setenv("RUNNER_MIDRUN_COMMAND_POLL_ENABLED", "1")
    fake_proc = FakeProc(lines=[])
    monkeypatch.setattr(experiment_runner.subprocess, "Popen", lambda *a, **k: fake_proc)
    rrc.append_command(runner_env["ree_assembly_path"], "test-machine-d5",
                        "force_stop", issued_by="test")

    started = time.monotonic()
    experiment_runner.run_experiment(
        runner_env["item"], runner_env["status"], runner_env["status_path"], {},
        remote_control=True, proc_ref=[],
        ree_assembly_path=runner_env["ree_assembly_path"],
        machine="test-machine-d5",
    )
    elapsed = time.monotonic() - started

    assert fake_proc.killed.is_set()
    assert elapsed < 2.0


def test_d6_poll_thread_crash_does_not_break_the_run(monkeypatch, runner_env):
    """If process_pending_commands itself raises on every call (e.g. a
    transient git/coordinator error), the experiment must still complete
    normally -- the poll thread's failure must stay isolated."""
    monkeypatch.setenv("RUNNER_MIDRUN_COMMAND_POLL_ENABLED", "1")
    fake_proc = FakeProc(lines=["verdict: PASS\n"])
    monkeypatch.setattr(experiment_runner.subprocess, "Popen", lambda *a, **k: fake_proc)
    monkeypatch.setattr(
        experiment_runner._rrc, "process_pending_commands",
        mock.Mock(side_effect=OSError("boom")))

    result = experiment_runner.run_experiment(
        runner_env["item"], runner_env["status"], runner_env["status_path"], {},
        remote_control=True, proc_ref=[],
        ree_assembly_path=runner_env["ree_assembly_path"],
        machine="test-machine-d6",
    )

    assert result["result"] in ("PASS", "FAIL", "ERROR", "UNKNOWN")
    assert not fake_proc.terminated.is_set()
    assert not fake_proc.killed.is_set()


def test_d7_no_command_pending_is_observably_identical_result(monkeypatch, runner_env):
    """With the poll thread enabled but nothing ever issued, the experiment
    result must be identical in shape to a normal run -- the requirement
    that enabling the mechanism costs nothing when it never fires."""
    monkeypatch.setenv("RUNNER_MIDRUN_COMMAND_POLL_ENABLED", "1")
    fake_proc = FakeProc(lines=["verdict: PASS\n"])
    monkeypatch.setattr(experiment_runner.subprocess, "Popen", lambda *a, **k: fake_proc)

    result = experiment_runner.run_experiment(
        runner_env["item"], runner_env["status"], runner_env["status_path"], {},
        remote_control=True, proc_ref=[],
        ree_assembly_path=runner_env["ree_assembly_path"],
        machine="test-machine-d7",
    )

    assert not fake_proc.terminated.is_set()
    assert not fake_proc.killed.is_set()
    assert result["exit_code"] == 0
