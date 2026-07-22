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

The second contract here is that `timeout=` is a WALL-CLOCK ceiling. A killed
child does not close the pipes its own children inherited, so a surviving
GRANDCHILD keeps the read blocked long after the child is dead: git's clean
filter holds git's stderr (verified with `lsof` -- the filter's fd 2 is the
same pipe as git's fd 2). Measured 2026-07-21: a `run(..., timeout=30)`
against a `git add` whose clean filter was `sleep 120` returned after ~120s.
Same reproducer below, with the elapsed time asserted.

The negative control for THAT one is not `subprocess.run`: on POSIX the stdlib
does `process.wait()` after the kill rather than re-draining, so it is already
bounded (measured 3.0s for `timeout=3` against the same repo). The unbounded
drain was a deviation `graceful_timeout` introduced when it grew the SIGTERM
escalation, so the control re-enacts THAT body verbatim.

None of this ever touches a shared checkout: every repo is built fresh under
`tmp_path`. An orphan `.git/index.lock` in a live checkout would block every
other session in this workspace.
"""

import os
import subprocess as stdlib_subprocess
import time

import pytest

import graceful_timeout

SLOW_FILTER_SECONDS = 20
KILL_AFTER_SECONDS = 3

# Sleeps for the wall-clock tests. Distinct, and distinctive enough that the
# best-effort `pkill -f` cleanup below cannot plausibly match anything else on
# the machine -- the filter process by construction OUTLIVES the git it was
# spawned from, so nothing else will reap it.
STALL_SLEEP_SECONDS = 137      # >> any bound the test asserts
CONTROL_SLEEP_SECONDS = 23     # short enough to pay for as a negative control


def _repo_with_slow_add(tmp_path, filter_seconds=SLOW_FILTER_SECONDS):
    """A git repo where `git add slow.dat` blocks in a clean filter.

    The filter holds the process inside `git add`, which holds `.git/index.lock`
    for its whole run -- the window the runner's timeouts fire in. It is also a
    GRANDCHILD of the caller holding git's inherited stdout/stderr, which is the
    unbounded-drain reproducer.

    `tmp_path` only -- never point this at a live checkout (see module docstring).
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return stdlib_subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True)

    git("init", "-q", ".")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    git("config", "filter.slow.clean", "sleep %d; cat" % filter_seconds)
    (repo / ".gitattributes").write_text("*.dat filter=slow\n")
    (repo / "seed.txt").write_text("seed\n")
    git("add", "seed.txt", ".gitattributes")
    git("commit", "-qm", "init")
    (repo / "slow.dat").write_text("payload\n")
    return repo


def _reap_orphan_filter(seconds):
    """Best-effort kill of the filter the test deliberately orphaned.

    Matches both the `sh -c` wrapper and the `sleep` under it. Left
    unanchored because macOS `pgrep -f` does not honour `^...$` against the
    full command line; the durations above are distinctive enough that this
    cannot plausibly match anything else the user owns. Failure is harmless --
    the filter exits on its own -- so the return code is ignored.
    """
    stdlib_subprocess.run(["pkill", "-f", "sleep %d" % seconds],
                          capture_output=True)


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


# `timeout + DEFAULT_GRACE_SECONDS + DEFAULT_REAP_SECONDS` is ~10s here. The
# assertion is deliberately loose: it is discriminating against the ~137s the
# unbounded drain takes, not measuring scheduler jitter on a loaded worker.
WALL_CLOCK_CEILING_SECONDS = 40


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
def test_timeout_is_a_wall_clock_ceiling_when_a_grandchild_holds_the_pipes(tmp_path):
    """The child dies on time; a process IT spawned keeps the pipes open.

    git's clean filter inherits git's stdout/stderr, so killing git does not
    bring the read side to EOF -- the stdlib's unbounded post-kill
    `communicate()` then waits out the FILTER, and the caller's `timeout=`
    silently stops being a ceiling.
    """
    repo = _repo_with_slow_add(tmp_path, filter_seconds=STALL_SLEEP_SECONDS)
    try:
        started = time.monotonic()
        with pytest.raises(stdlib_subprocess.TimeoutExpired) as excinfo:
            graceful_timeout.run(["git", "add", "slow.dat"], cwd=repo,
                                 capture_output=True, timeout=KILL_AFTER_SECONDS)
        elapsed = time.monotonic() - started
    finally:
        _reap_orphan_filter(STALL_SLEEP_SECONDS)

    assert elapsed < WALL_CLOCK_CEILING_SECONDS, (
        "run() took %.1fs for a timeout=%ss -- the post-kill drain is unbounded "
        "again, so an unattended caller (serve.py's _auto_pull thread, the "
        "hourly igw tick) can stall for as long as a grandchild lives"
        % (elapsed, KILL_AFTER_SECONDS))
    # Still the stdlib's exception, still carrying whatever was captured:
    # giving up on the STREAMS must not change what the caller catches.
    assert excinfo.value.timeout == KILL_AFTER_SECONDS
    assert excinfo.value.output in (b"", None) or isinstance(
        excinfo.value.output, bytes)


def _prefix_drain(argv, cwd, timeout, grace):
    """The pre-fix timeout path, verbatim, as the negative control.

    Identical to `graceful_timeout.run`'s except-branch as it stood before the
    drain was bounded: SIGTERM, bounded grace wait, SIGKILL, then an UNBOUNDED
    `communicate()`. Re-enacted here rather than referenced so the control
    cannot silently start testing the fixed code.
    """
    with stdlib_subprocess.Popen(argv, cwd=cwd, stdout=stdlib_subprocess.PIPE,
                                 stderr=stdlib_subprocess.PIPE) as process:
        try:
            process.communicate(timeout=timeout)
        except stdlib_subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.communicate(timeout=grace)
            except stdlib_subprocess.TimeoutExpired:
                process.kill()
                process.communicate()          # <-- the defect under test
            raise


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
def test_negative_control_the_unbounded_drain_overruns_its_own_timeout(tmp_path):
    """Pins that the grandchild really does hold the pipes on this platform.

    Without this, the test above would pass vacuously on any machine where the
    filter did not inherit one of git's captured streams -- there would be
    nothing to bound, and the bound would be untested rather than satisfied.
    """
    repo = _repo_with_slow_add(tmp_path, filter_seconds=CONTROL_SLEEP_SECONDS)
    try:
        started = time.monotonic()
        with pytest.raises(stdlib_subprocess.TimeoutExpired):
            _prefix_drain(["git", "add", "slow.dat"], repo,
                          KILL_AFTER_SECONDS, grace=1)
        elapsed = time.monotonic() - started
    finally:
        _reap_orphan_filter(CONTROL_SLEEP_SECONDS)

    assert elapsed > CONTROL_SLEEP_SECONDS * 0.6, (
        "the pre-fix drain returned in %.1fs for a timeout=%ss against a %ss "
        "filter -- the grandchild is no longer holding the pipes on this "
        "platform, so re-derive whether the bounded drain in "
        "graceful_timeout.run is still load-bearing"
        % (elapsed, KILL_AFTER_SECONDS, CONTROL_SLEEP_SECONDS))


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
