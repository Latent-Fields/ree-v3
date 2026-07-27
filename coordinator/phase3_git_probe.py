"""Test helper: intercept EVERY git invocation a runner module can make.

WHY THIS EXISTS
---------------
The Phase 3 runner-push gate tests assert that a gate is a true no-op by
counting git subprocess invocations. They did that with::

    with patch.object(er.subprocess, "run") as mock_run:
        er.git_push_results(...)
    self.assertEqual(mock_run.call_count, 0)

That intercept stopped working at ree-v3 `b0a6dd8` ("runner: a git
TimeoutExpired is a skipped tick, not a process death"), which routed all
64 git call sites in `experiment_runner` through `_git_run` ->
`graceful_timeout.run_soft_timeout`, and both git call sites in
`runner_remote_control` through `_git` -> the same. Neither reaches
`<module>.subprocess.run`:

  * `experiment_runner.subprocess` / `runner_remote_control.subprocess` are
    `graceful_timeout.wrap(subprocess)` shims, so patching `.run` on them
    only rebinds the shim's own attribute, and
  * `run_soft_timeout` calls `graceful_timeout.run`, which goes straight to
    `subprocess.Popen` in the stdlib module.

So the patch intercepted nothing. That is visible in two ways, and only the
first is loud:

  * the three "gate OFF -> subprocess IS invoked" sanity tests FAILED, with
    real git running against a temp dir ("fatal: not a git repository");
  * the ten "gate ON -> call_count == 0" assertions passed VACUOUSLY. A
    count of zero was guaranteed whether or not the gate worked, so the
    tests that actually guard the Phase 3 writer-vs-runner index race had
    been blind since 2026-07-21.

WHAT IT DOES
------------
Patches the module's git helper (`_git_run` / `_git`) AND its
`subprocess.run` with a SINGLE shared mock, so `call_count` is the total
number of git invocations by any route. Patching both means a call site
that goes back to a bare `subprocess.run` is still counted -- the probe
measures "did any git run", not "was one particular helper called".

The mock returns a `returncode == 0` / empty-output shape by default, which
is the "git succeeded, nothing to do" branch every caller here expects.
Override on the yielded mock for a test that needs a different branch.

ASCII-only output.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

__all__ = ["patch_git_calls", "GIT_HELPERS"]

# Per-module name of the soft-timeout git helper introduced by b0a6dd8.
# A module is probed through whichever of these it defines.
GIT_HELPERS = ("_git_run", "_git")


@contextmanager
def patch_git_calls(mod, returncode: int = 0, stdout: str = "",
                    stderr: str = ""):
    """Patch every git entry point on `mod`; yield the shared call mock.

    `mod` is an imported `experiment_runner` / `runner_remote_control`.
    Raises AssertionError if the module exposes neither a git helper nor a
    `subprocess` attribute -- a silent no-op patch is exactly the failure
    mode this helper exists to prevent, so it must never fail open.
    """
    mock = MagicMock(name="git_call")
    mock.return_value.returncode = returncode
    mock.return_value.stdout = stdout
    mock.return_value.stderr = stderr

    patches = []
    for helper in GIT_HELPERS:
        if hasattr(mod, helper):
            patches.append(patch.object(mod, helper, mock))
    subproc = getattr(mod, "subprocess", None)
    if subproc is not None and hasattr(subproc, "run"):
        patches.append(patch.object(subproc, "run", mock))

    assert patches, (
        "no git entry point found on %s: expected one of %s or a "
        "subprocess.run attribute. Patching nothing would make every "
        "call-count assertion vacuous." % (getattr(mod, "__name__", mod),
                                           ", ".join(GIT_HELPERS)))

    for p in patches:
        p.start()
    try:
        yield mock
    finally:
        for p in reversed(patches):
            p.stop()
