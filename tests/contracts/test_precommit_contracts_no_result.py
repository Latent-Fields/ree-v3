"""Contract tests for precommit_contracts.sh ROUTER NO-RESULT handling (2026-09-18).

chip-20260917-precommit-gate-shared-tree, defect 2.

THE DEFECT. remote_pytest.sh documents its exit codes in two OVERLAPPING bands:
"0-5 pytest's own result" and "2-8 this wrapper's PRE-RUN failures". pytest
itself returns 2 (interrupted), 3 (internal error), 4 (usage error) and 5 (no
tests collected), so rc=4 alone cannot distinguish "pytest usage error" from
"no box could be acquired". The gate treated any non-zero as a contract failure
and blocked the commit. Measured 2026-09-17: session substrate-build-...-sd097
lost two rejected commit attempts and then a 42-minute starved LOCAL run
(aborted at 73%, zero failures) to that misreading.

THE FIX IS NOT "IGNORE RC 4". The signal is remote_pytest.sh's own sentinel line
"remote-pytest: NO RESULT (infra exit=<rc>)", printed on every path where no
suite verdict exists and on no other path. The gate matches THAT, never the
number -- so N3 and N4 below (a red suite at rc=1, and rc=4 WITHOUT the
sentinel) must both still block. N4 is the fail-closed case that matters most:
an older remote_pytest.sh, or a lost log capture, must never be read as "the
gate did not run, carry on".

WHAT A NO-RESULT DOES is the policy the user already set on 2026-09-08 for the
structurally identical missing-router case: re-check the memory floor, run
LOCALLY if it clears (N1), and BLOCK with "the gate DID NOT RUN" if it does not
(N2). Blocking stays the safe direction; what changes is that the gate no longer
claims the tests failed, and no longer takes ~40 minutes to say so.
"""

import os
import subprocess
import sys
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "scripts" / "precommit_contracts.sh"
SENTINEL = "remote-pytest: NO RESULT (infra exit="

_GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


def _git(repo, *args):
    env = {**os.environ, **_GIT_ENV}
    return subprocess.run(["git", *args], cwd=repo, check=True, env=env,
                          capture_output=True, text=True)


def _repo(tmp_path):
    repo = tmp_path / "ree-v3"
    (repo / "ree_core").mkdir(parents=True)
    (repo / "tests" / "contracts").mkdir(parents=True)
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("x\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "base")
    p = repo / "ree_core" / "x.py"
    p.write_text("X = 1\n")
    _git(repo, "add", "ree_core/x.py")
    return repo


def _router(tmp_path, exit_code, sentinel=True):
    """A stand-in remote_pytest.sh. `sentinel` controls whether it announces
    that no suite verdict exists."""
    stub = tmp_path / ("router_%s_%s.sh" % (exit_code, sentinel))
    body = "#!/usr/bin/env bash\n"
    if sentinel:
        body += 'echo "%s%s)  -- no box could be acquired" >&2\n' % (SENTINEL, exit_code)
    else:
        body += 'echo "1 failed, 5000 passed" >&2\n'
    body += "exit %d\n" % exit_code
    stub.write_text(body)
    stub.chmod(0o755)
    return stub


def _local(tmp_path, exit_code, name="local"):
    stub = tmp_path / ("local_%s_%s.sh" % (name, exit_code))
    stub.write_text("#!/usr/bin/env bash\necho 'LOCAL STUB RAN' >&2\nexit %d\n" % exit_code)
    stub.chmod(0o755)
    return stub


def _run(repo, extra_env, args=()):
    env = {**os.environ}
    env.pop("GIT_INDEX_FILE", None)
    env.pop("GIT_DIR", None)
    for key in list(env):
        if key.startswith("REE_PRECOMMIT_"):
            del env[key]
    env.update(_GIT_ENV)
    env["REE_PRECOMMIT_VALIDATION_CACHE_DISABLE"] = "1"
    # These cases stub both the router and the local run, so they should never
    # reach a real pytest -- but pin the interpreter anyway. If a future edit
    # drops a stub, the gate would otherwise fall back to a PATH python3 that on
    # the fleet has no pytest, and exit 2 for that reason instead of the one the
    # test means to pin. The assertion below is what makes that visible.
    env["REE_PRECOMMIT_CONTRACTS_PYTHON"] = sys.executable
    # The race loop checks for the remote result BEFORE sleeping, so an
    # instantly-returning stub still costs one full poll interval (15s by
    # default). These stubs never need a real interval; without this the six
    # tests add ~75s to a suite already brushing its CI timeout.
    env["REE_PRECOMMIT_CONTRACTS_RACE_POLL_SEC"] = "1"
    env.update(extra_env)
    p = subprocess.run(["bash", str(GATE), *args], cwd=repo,
                       capture_output=True, text=True, env=env)
    blob = (p.stdout or "") + (p.stderr or "")
    assert "No module named pytest" not in blob, (
        "the gate's interpreter has no pytest -- its exit code then says nothing "
        "about routing vs a red suite, and these tests would pin nothing. "
        "Output was:\n" + blob)
    return p


# --------------------------------------------------------------------------- N1
def test_n1_no_result_above_floor_reroutes_local(tmp_path):
    """THE MEASURED DEFECT. A routing refusal with memory available must run the
    suite here instead of blocking -- coverage unchanged, no verdict invented."""
    repo = _repo(tmp_path)
    p = _run(repo, {
        "REE_PRECOMMIT_CONTRACTS_TARGET": "remote",
        "REE_PRECOMMIT_REMOTE_PYTEST": str(_router(tmp_path, 4)),
        "REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST": str(_local(tmp_path, 0)),
        "REE_PRECOMMIT_CONTRACTS_FREE_MB": "8000",     # >= 3000 floor
    })
    assert p.returncode == 0, (
        "a routing refusal blocked the commit -- the 2026-09-17 defect.\n"
        + p.stdout + p.stderr)
    assert "NOT A TEST FAILURE" in p.stderr, p.stderr
    assert "LOCAL STUB RAN" in p.stderr, "the gate did not actually re-run:\n" + p.stderr


# --------------------------------------------------------------------------- N2
def test_n2_no_result_below_floor_blocks_saying_the_gate_did_not_run(tmp_path):
    """Below the floor, blocking is still right -- but it must say the gate did
    not run rather than that the tests failed."""
    repo = _repo(tmp_path)
    p = _run(repo, {
        "REE_PRECOMMIT_CONTRACTS_TARGET": "remote",
        "REE_PRECOMMIT_REMOTE_PYTEST": str(_router(tmp_path, 4)),
        "REE_PRECOMMIT_CONTRACTS_FREE_MB": "100",      # << 3000 floor
    })
    assert p.returncode == 2, p.stdout + p.stderr
    assert "DID NOT RUN" in p.stderr, p.stderr
    assert "contract tests failed" not in p.stderr, (
        "a routing condition is still being reported as a red suite:\n" + p.stderr)


# --------------------------------------------------------------------------- N3
def test_n3_negative_control_real_red_suite_still_blocks(tmp_path):
    """NEGATIVE CONTROL. A genuine red suite carries no sentinel and must block
    as a test failure -- and must NOT be re-routed to a local run, which would
    hand a second chance to an already-failed suite."""
    repo = _repo(tmp_path)
    p = _run(repo, {
        "REE_PRECOMMIT_CONTRACTS_TARGET": "remote",
        "REE_PRECOMMIT_REMOTE_PYTEST": str(_router(tmp_path, 1, sentinel=False)),
        "REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST": str(_local(tmp_path, 0)),
        "REE_PRECOMMIT_CONTRACTS_FREE_MB": "8000",
    })
    assert p.returncode == 2, "a red suite no longer blocks:\n" + p.stdout + p.stderr
    assert "LOCAL STUB RAN" not in p.stderr, (
        "a red remote suite was re-routed to a local run and could have been "
        "overturned by it:\n" + p.stderr)


# --------------------------------------------------------------------------- N4
def test_n4_fail_closed_rc4_without_the_sentinel_still_blocks(tmp_path):
    """FAIL-CLOSED, the case that matters most. rc=4 with NO sentinel -- an older
    remote_pytest.sh, a truncated log, a lost capture, or a genuine pytest usage
    error. The gate must block, not assume it merely failed to run."""
    repo = _repo(tmp_path)
    p = _run(repo, {
        "REE_PRECOMMIT_CONTRACTS_TARGET": "remote",
        "REE_PRECOMMIT_REMOTE_PYTEST": str(_router(tmp_path, 4, sentinel=False)),
        "REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST": str(_local(tmp_path, 0)),
        "REE_PRECOMMIT_CONTRACTS_FREE_MB": "8000",
    })
    assert p.returncode == 2, (
        "an UNLABELLED rc=4 was treated as 'the gate did not run' -- the gate must "
        "fail closed on an ambiguous code.\n" + p.stdout + p.stderr)
    assert "LOCAL STUB RAN" not in p.stderr, p.stderr


# --------------------------------------------------------------------------- N5
def test_n5_no_block_downgrades_the_did_not_run_block(tmp_path):
    """--no-block is advisory mode (CI): it must not block even here."""
    repo = _repo(tmp_path)
    p = _run(repo, {
        "REE_PRECOMMIT_CONTRACTS_TARGET": "remote",
        "REE_PRECOMMIT_REMOTE_PYTEST": str(_router(tmp_path, 4)),
        "REE_PRECOMMIT_CONTRACTS_FREE_MB": "100",
    }, args=("--no-block",))
    assert p.returncode == 0, p.stdout + p.stderr


# --------------------------------------------------------------------------- N6
def test_n6_router_emits_the_sentinel_the_gate_matches(tmp_path):
    """The two halves must not drift. This pins the REAL remote_pytest.sh
    emitting the REAL string the gate greps for -- a stub-only test would pass
    vacuously if the wrapper's wording changed."""
    router = Path(__file__).resolve().parents[3] / "scripts" / "remote_pytest.sh"
    if not router.exists():                      # staged tree without the sibling
        import pytest
        pytest.skip("remote_pytest.sh not present in this tree")
    text = router.read_text()
    assert 'NO_RESULT_SENTINEL="%s' % SENTINEL in text, (
        "remote_pytest.sh no longer defines the sentinel the gate matches on")
    gate = GATE.read_text()
    assert SENTINEL in gate, "precommit_contracts.sh no longer matches the sentinel"
