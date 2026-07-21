"""Contract: a git `TimeoutExpired` must not kill the experiment runner.

The runner's git helpers are all written to DEGRADE -- `_list_unmerged_paths`
documents "Returns None if `git status --porcelain` failed" and its callers
handle None; `git_pull` wraps its pull in `except Exception`. But every one of
those paths keys on `returncode != 0`, and a TIMEOUT RAISED instead of
returning, so the degrade path never ran: the exception propagated out of
`git_pull` and out of `main()`.

Two structural details made that unavoidable rather than unlucky:
  * `git_pull` calls `_list_unmerged_paths` and `_prepull_stash_blocking_untracked`
    BEFORE its `for attempt` loop, i.e. outside the `except Exception`;
  * the recovery calls inside that handler (`git rebase --abort` / `--quit`)
    carry their own timeouts, so a timeout could escape on the way OUT of it.

Evidence: `REE_assembly/runner.log` held 274 `TimeoutExpired` tracebacks against
270 `[runner] Runner version` startup lines -- essentially every runner restart
was a git-timeout crash, respawned by launchd (`com.ree.runner`, KeepAlive=true,
ThrottleInterval 60).

The fix is `graceful_timeout.run_soft_timeout` behind `_git_run`: a timeout
becomes a failed `CompletedProcess` (rc 124) on the branch each call site
already has. NOT wider timeouts (a stalled git stays stalled), and NOT silence
(every timeout prints, with a running count).

Sibling contract: `test_graceful_timeout_lockfile.py` pins the SIGNAL half of
this (SIGTERM before SIGKILL, so a timed-out git unlinks its own index.lock).
That one deliberately left the exception fatal; this is the other half.
"""

import ast
import subprocess as stdlib_subprocess
from pathlib import Path

import pytest

import experiment_runner
import graceful_timeout
import runner_remote_control

REE_V3 = Path(__file__).resolve().parents[2]


def _always_timeout(*popenargs, **kwargs):
    """Stand-in for graceful_timeout.run that always times out."""
    cmd = popenargs[0] if popenargs else kwargs.get("args", ["git"])
    raise stdlib_subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 10))


# --- the primitive ---------------------------------------------------------

def test_soft_timeout_returns_failed_result_instead_of_raising():
    r = graceful_timeout.run_soft_timeout(
        ["sleep", "5"], capture_output=True, text=True, timeout=1,
    )
    assert r.returncode == graceful_timeout.TIMEOUT_RETURNCODE
    assert r.returncode != 0, "must land on every caller's failure branch"
    assert "timed out" in r.stderr


def test_soft_timeout_output_matches_the_callers_stream_mode():
    """`TimeoutExpired.output`/`.stderr` are None (or bytes) even in text mode.

    Callers do `r.stdout.splitlines()`, so an unnormalised None would raise
    AttributeError on the very path that is supposed to be the safe one.
    """
    txt = graceful_timeout.run_soft_timeout(
        ["sleep", "5"], capture_output=True, text=True, timeout=1,
    )
    assert isinstance(txt.stdout, str) and isinstance(txt.stderr, str)
    txt.stdout.splitlines()

    raw = graceful_timeout.run_soft_timeout(
        ["sleep", "5"], capture_output=True, timeout=1,
    )
    assert isinstance(raw.stdout, bytes) and isinstance(raw.stderr, bytes)


def test_soft_timeout_invokes_on_timeout_callback():
    seen = []
    graceful_timeout.run_soft_timeout(
        ["sleep", "5"], capture_output=True, text=True, timeout=1,
        on_timeout=seen.append,
    )
    assert len(seen) == 1
    assert isinstance(seen[0], stdlib_subprocess.TimeoutExpired)


def test_soft_timeout_still_raises_under_check_true():
    """A caller asking for failures-as-exceptions must still get one."""
    with pytest.raises(stdlib_subprocess.TimeoutExpired):
        graceful_timeout.run_soft_timeout(
            ["sleep", "5"], capture_output=True, text=True, timeout=1,
            check=True,
        )


def test_soft_timeout_delegates_when_no_timeout_is_set():
    r = graceful_timeout.run_soft_timeout(
        ["echo", "hi"], capture_output=True, text=True,
    )
    assert r.returncode == 0 and r.stdout.strip() == "hi"


# --- the runner's entry point ---------------------------------------------

def test_git_run_logs_every_timeout_with_a_running_count(capsys, monkeypatch):
    """Timeouts must stay VISIBLE -- the rate is the operational signal."""
    monkeypatch.setattr(experiment_runner, "_GIT_TIMEOUT_COUNT", 0)
    monkeypatch.setattr(graceful_timeout, "run", _always_timeout)

    for _ in range(2):
        r = experiment_runner._git_run(
            ["git", "status", "--porcelain"], capture_output=True,
            text=True, timeout=10,
        )
        assert r.returncode == graceful_timeout.TIMEOUT_RETURNCODE

    out = capsys.readouterr().out
    assert "git TIMEOUT #1" in out and "git TIMEOUT #2" in out
    assert "git status --porcelain" in out


def test_list_unmerged_paths_returns_none_on_timeout(monkeypatch, tmp_path):
    """Its docstring promises None on failure; a timeout is a failure."""
    monkeypatch.setattr(graceful_timeout, "run", _always_timeout)
    assert experiment_runner._list_unmerged_paths(tmp_path) is None


def test_git_pull_survives_a_timeout_on_every_git_call(monkeypatch, tmp_path):
    """THE regression. Every git call times out; git_pull must still return.

    This covers the pre-loop calls (outside `except Exception`) and the
    `rebase --abort` / `--quit` calls inside the handler alike.
    """
    monkeypatch.setattr(graceful_timeout, "run", _always_timeout)
    experiment_runner.git_pull(tmp_path, "ree-v3")          # must not raise
    experiment_runner.git_pull(tmp_path, "REE_assembly")    # prepull-stash path


def test_git_push_with_retry_survives_a_timeout(monkeypatch, tmp_path):
    """`_git_push_with_retry` had no exception guard at all."""
    monkeypatch.setattr(graceful_timeout, "run", _always_timeout)
    assert experiment_runner._git_push_with_retry(
        str(tmp_path), "main", "test-label",
    ) is False


def test_remote_control_git_degrades_and_logs(monkeypatch, capsys, tmp_path):
    """rrc already caught broadly -- but SILENTLY. Now it logs."""
    monkeypatch.setattr(runner_remote_control, "_GIT_TIMEOUT_COUNT", 0)
    monkeypatch.setattr(graceful_timeout, "run", _always_timeout)

    assert runner_remote_control._hard_sync_is_safe(str(tmp_path), "master") is False
    assert "git TIMEOUT #1" in capsys.readouterr().out


# --- regression lint -------------------------------------------------------

@pytest.mark.parametrize(
    "module", ["experiment_runner.py", "runner_remote_control.py"],
)
def test_no_bare_subprocess_run_call_sites_remain(module):
    """A new `subprocess.run(...)` here would silently reintroduce the crash.

    The module-local `subprocess` is graceful_timeout's shim, which still
    RAISES on timeout by design -- so every git call must go through
    `_git_run` / `run_soft_timeout` instead.
    """
    tree = ast.parse((REE_V3 / module).read_text(encoding="utf-8"))
    bare = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in ("subprocess.run", "_subprocess.run")
    ]
    assert not bare, (
        f"{module}: bare subprocess.run at line(s) {bare} -- route git calls "
        f"through _git_run so a timeout degrades instead of killing the runner"
    )
