"""Contract: a timed-out git must not orphan `.git/index.lock`.

`subprocess.run(..., timeout=N)` SIGKILLs the child, and SIGKILL bypasses the
signal handler git uses to unlink its lockfile. The runner calls index-taking
git commands (`status`, `add`, `commit`, `pull --rebase --autostash`,
`rebase --abort`) under exactly that timeout, on both `ree-v3` and
`REE_assembly`, every loop tick -- so before `graceful_timeout` each timeout
left an orphan lock. An orphan lock blocks `scripts/ree_commit.py`'s
shared-index refresh, which leaves the committed paths staged as a REVERT of
their own commit for the next session's plain `git commit` to land.

The negative control matters as much as the positive one: it pins that the
hazard is real on this platform's git, so a future git that cleaned up after
SIGKILL would make the control fail loudly rather than let the positive test
pass vacuously.
"""

import os
import subprocess as stdlib_subprocess
import time

import pytest

import graceful_timeout

SLOW_FILTER_SECONDS = 20
KILL_AFTER_SECONDS = 3


def _repo_with_slow_add(tmp_path):
    """A git repo where `git add slow.dat` blocks in a clean filter.

    The filter holds the process inside `git add`, which holds `.git/index.lock`
    for its whole run -- the window the runner's timeouts fire in.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return stdlib_subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True)

    git("init", "-q", ".")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    git("config", "filter.slow.clean", "sleep %d; cat" % SLOW_FILTER_SECONDS)
    (repo / ".gitattributes").write_text("*.dat filter=slow\n")
    (repo / "seed.txt").write_text("seed\n")
    git("add", "seed.txt", ".gitattributes")
    git("commit", "-qm", "init")
    (repo / "slow.dat").write_text("payload\n")
    return repo


def _add_with_timeout(runner, repo):
    """Run the blocking `git add` under `runner`; return True if a lock is left."""
    with pytest.raises(stdlib_subprocess.TimeoutExpired):
        runner(["git", "add", "slow.dat"], cwd=repo,
               capture_output=True, timeout=KILL_AFTER_SECONDS)
    lock = repo / ".git" / "index.lock"
    # git unlinks from its signal handler; give the exiting process a moment.
    for _ in range(20):
        if not lock.exists():
            return False
        time.sleep(0.1)
    return lock.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
def test_graceful_run_does_not_orphan_the_index_lock(tmp_path):
    repo = _repo_with_slow_add(tmp_path)
    assert _add_with_timeout(graceful_timeout.run, repo) is False, (
        "graceful_timeout.run left an orphan .git/index.lock -- the SIGTERM "
        "escalation is not reaching git")


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
def test_negative_control_stdlib_run_does_orphan_the_index_lock(tmp_path):
    repo = _repo_with_slow_add(tmp_path)
    assert _add_with_timeout(stdlib_subprocess.run, repo) is True, (
        "stdlib subprocess.run no longer orphans the lock on this platform -- "
        "re-derive whether graceful_timeout is still load-bearing before "
        "relaxing anything that depends on it")


def test_wrap_preserves_module_identity_and_leaves_stdlib_unmutated():
    shim = graceful_timeout.wrap(stdlib_subprocess)
    assert shim.run is graceful_timeout.run
    assert stdlib_subprocess.run is not graceful_timeout.run
    for name in ("Popen", "PIPE", "DEVNULL", "TimeoutExpired",
                 "CalledProcessError", "CompletedProcess"):
        assert getattr(shim, name) is getattr(stdlib_subprocess, name), name


def test_runner_modules_use_the_graceful_run():
    """The wiring, not just the helper -- an unwired module is the silent failure."""
    import experiment_runner
    import runner_remote_control

    assert experiment_runner.subprocess.run is graceful_timeout.run
    assert runner_remote_control.subprocess.run is graceful_timeout.run


@pytest.mark.parametrize("kwargs, expected", [
    (dict(capture_output=True, text=True), "hi\n"),
    (dict(stdout=stdlib_subprocess.PIPE, text=True), "hi\n"),
])
def test_run_matches_stdlib_on_the_non_timeout_paths(kwargs, expected):
    """capture_output/input/check are run()-level kwargs we translate by hand."""
    r = graceful_timeout.run(["echo", "hi"], timeout=30, **kwargs)
    assert r.returncode == 0 and r.stdout == expected

    r = graceful_timeout.run(["cat"], input="piped\n", capture_output=True,
                             text=True, timeout=30)
    assert r.stdout == "piped\n"

    with pytest.raises(stdlib_subprocess.CalledProcessError):
        graceful_timeout.run(["false"], timeout=30, check=True)

    # timeout=None delegates to the stdlib untouched.
    r = graceful_timeout.run(["echo", "hi"], capture_output=True, text=True)
    assert r.stdout == "hi\n"
