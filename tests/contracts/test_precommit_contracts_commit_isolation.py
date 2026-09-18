"""Contract tests for precommit_contracts.sh COMMIT-CONTENT ISOLATION (2026-09-18).

chip-20260917-precommit-gate-shared-tree, defect 1.

THE DEFECT. Every gate block used to run against the ambient WORKING TREE. In
the shared checkout that tree is not the commit: it is HEAD plus this session's
staged paths plus its unstaged edits plus EVERY concurrent session's uncommitted
edits plus every untracked file anyone has dropped in it. ree_commit.py mean-
while commits exactly the declared paths from a private index seeded read-tree
HEAD. So the tested tree and the committed tree were different trees, and the
gap was everyone else's work in progress.

Measured 2026-09-17, three sessions in one afternoon: substrate-build-...-triad
saw 16 failed / 5081 passed with ALL 16 in one foreign file belonging to another
session's unlanded build; substrate-build-...-allon lost four attempts over ~100
minutes to two foreign untracked test files, zero failures in its own paths.

THE HALF THAT MATTERS MOST IS THE FALSE PASS, and I6 is its guard. A foreign (or
your own unstaged) file can SATISFY a dependency your commit introduces, so the
gate goes green on a combination that will never exist on trunk. Demonstrated
live on 2026-09-18: ree_core/hippocampal/ghost_goal_bank.py carried a
module-scope import of an UNTRACKED possibility_topology.py; committing
ghost_goal_bank.py alone gave "import OK" against the ambient tree and
ModuleNotFoundError against the commit's own tree.

WHY THE NEGATIVE CONTROLS ARE THE POINT. "Stop failing on other people's files"
is also achieved by not running the gate at all, so every test that asserts
something no longer fails is paired with one asserting that a genuine break in
the COMMITTED content still blocks (I2, I6), and that a failure to isolate falls
back to running rather than skipping (I4).
"""

import os
import subprocess
import sys
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "scripts" / "precommit_contracts.sh"

_GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}

PASSING_LINT = "def test_ok():\n    assert True\n"
FAILING_LINT = "def test_bad():\n    assert False, 'a genuine break in the committed content'\n"


def _git(repo, *args):
    env = {**os.environ, **_GIT_ENV}
    return subprocess.run(["git", *args], cwd=repo, check=True, env=env,
                          capture_output=True, text=True)


def _repo(tmp_path, committed=None):
    """A minimal ree-v3-shaped repo with a REAL HEAD commit.

    A HEAD is load-bearing here and not incidental: stage_commit_tree builds the
    stage with `commit-tree <tree> -p HEAD`, so a repo with no commits exercises
    the fall-back path instead of the isolation path.
    """
    repo = tmp_path / "ree-v3"
    (repo / "ree_core").mkdir(parents=True)
    (repo / "tests" / "contracts").mkdir(parents=True)
    (repo / "experiments").mkdir(parents=True)
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    for rel, body in (committed or {}).items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        _git(repo, "add", rel)
    (repo / "README.md").write_text("x\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _stage(repo, rel, body="# staged\n"):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    _git(repo, "add", rel)


def _untracked(repo, rel, body):
    """A file present on disk and in NO index -- another session's work."""
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


def _run(repo, extra_env=None, args=()):
    env = {**os.environ}
    env.pop("GIT_INDEX_FILE", None)
    env.pop("GIT_DIR", None)
    for key in list(env):
        if key.startswith("REE_PRECOMMIT_"):
            del env[key]
    env.update(_GIT_ENV)
    # Pin the gate's interpreter to the one running these tests. Without this the
    # gate falls back to whatever `python3` is first on PATH, which on the cloud
    # fleet is /usr/bin/python3 with NO pytest -- every block that shells out to
    # pytest then dies with "No module named pytest" and exits 2. Measured on the
    # hub 2026-09-18: that made I1/I5/I6 fail outright AND made I2/I4 pass
    # VACUOUSLY, since they assert exit 2 and got it for entirely the wrong
    # reason. _assert_gate_ran below is the guard against that recurring.
    env["REE_PRECOMMIT_CONTRACTS_PYTHON"] = sys.executable
    env.update(extra_env or {})
    p = subprocess.run(["bash", str(GATE), *args], cwd=repo,
                       capture_output=True, text=True, env=env)
    _assert_gate_ran(p)
    return p


def _assert_gate_ran(p):
    """Fail loudly when the gate could not run pytest at all.

    Every test here reads the gate's EXIT CODE. An interpreter without pytest
    produces the same exit 2 as a real contract break, so without this check the
    tests that expect a block would go green on an environment fault and pin
    nothing. Asserted on every invocation, including the ones expecting exit 0,
    so it cannot rot.
    """
    blob = (p.stdout or "") + (p.stderr or "")
    assert "No module named pytest" not in blob, (
        "the gate's interpreter has no pytest, so its exit code says nothing "
        "about contract content -- these tests would pass or fail for the wrong "
        "reason. Output was:\n" + blob)


# --------------------------------------------------------------------------- I1
def test_i1_foreign_untracked_failing_test_does_not_fail_this_commit():
    """THE MEASURED DEFECT. Another session's untracked, failing contract test
    sits in the shared tree; it must not block a commit that does not contain
    it."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        repo = _repo(Path(td), committed={"tests/contracts/test_ok_lint.py": PASSING_LINT})
        _stage(repo, "experiments/v3_probe.py")          # triggers Block 1c
        _untracked(repo, "tests/contracts/test_foreign_lint.py", FAILING_LINT)
        p = _run(repo)
        assert p.returncode == 0, (
            "a foreign untracked failing test blocked this commit -- the 2026-09-17 "
            "defect is back.\n" + p.stderr)
        assert "isolated from the shared checkout" in p.stderr, p.stderr


# --------------------------------------------------------------------------- I2
def test_i2_negative_control_genuine_break_in_committed_content_still_blocks():
    """NEGATIVE CONTROL for I1. If isolation were achieved by weakening or
    skipping the gate, this would pass too. It must not."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        repo = _repo(Path(td), committed={"tests/contracts/test_bad_lint.py": FAILING_LINT})
        _stage(repo, "experiments/v3_probe.py")
        p = _run(repo)
        assert p.returncode == 2, (
            "a genuine break in the COMMITTED content did not block the commit -- "
            "isolation must never become a bypass.\n" + p.stdout + p.stderr)


# --------------------------------------------------------------------------- I3
def test_i3_gate_runs_against_the_commits_tree_not_the_working_tree():
    """Structural assertion, independent of any test outcome: the directory the
    suite runs in carries the STAGED content and NOT the foreign untracked
    file."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        repo = _repo(Path(td))
        _stage(repo, "ree_core/mine.py", "MINE = 1\n")   # triggers Block 2
        _untracked(repo, "ree_core/foreign.py", "FOREIGN = 1\n")
        report = Path(td) / "report.txt"
        probe = Path(td) / "probe.sh"
        probe.write_text(
            "#!/usr/bin/env bash\n"
            f'{{ echo "cwd=$PWD"; '
            'echo "mine=$([ -f ree_core/mine.py ] && echo yes || echo no)"; '
            'echo "foreign=$([ -f ree_core/foreign.py ] && echo yes || echo no)"; '
            f'}} > "{report}"\nexit 0\n')
        probe.chmod(0o755)
        p = _run(repo, {
            "REE_PRECOMMIT_CONTRACTS_TARGET": "local",
            "REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST": str(probe),
            "REE_PRECOMMIT_VALIDATION_CACHE_DISABLE": "1",
        })
        assert p.returncode == 0, p.stderr
        text = report.read_text()
        assert "mine=yes" in text, "the commit's own staged content is missing:\n" + text
        assert "foreign=no" in text, (
            "a foreign UNTRACKED file was present in the tested tree -- the gate is "
            "still reading the shared working tree.\n" + text)
        assert "cwd=%s" % repo not in text, (
            "the suite ran in the shared checkout itself:\n" + text)


# --------------------------------------------------------------------------- I4
def test_i4_failsafe_isolation_off_still_runs_and_still_blocks():
    """FAIL-SAFE DIRECTION. With isolation disabled the gate reverts to the
    ambient tree -- it does NOT skip. A gate that does not run is worse than one
    reading a contaminated tree, so the fallback must still block a real break."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        repo = _repo(Path(td))
        _stage(repo, "experiments/v3_probe.py")
        _untracked(repo, "tests/contracts/test_foreign_lint.py", FAILING_LINT)
        p = _run(repo, {"REE_PRECOMMIT_CONTRACTS_ISOLATE": "0"})
        assert p.returncode == 2, (
            "with isolation off the gate must still run against the ambient tree "
            "and block on its failures.\n" + p.stdout + p.stderr)
        assert "isolation OFF" in p.stderr, p.stderr


# --------------------------------------------------------------------------- I5
def test_i5_staging_worktree_is_not_leaked():
    """The stage is registered with git. If it is not removed, every gated
    commit leaks an entry into `git worktree list` and a tree into TMPDIR."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        repo = _repo(Path(td), committed={"tests/contracts/test_ok_lint.py": PASSING_LINT})
        _stage(repo, "experiments/v3_probe.py")
        p = _run(repo)
        assert p.returncode == 0, p.stderr
        listing = _git(repo, "worktree", "list").stdout
        assert listing.count("\n") == 1, (
            "the staging worktree leaked:\n" + listing)


# --------------------------------------------------------------------------- I6
def test_i6_false_pass_guard_staged_dependency_on_an_untracked_file_is_caught():
    """THE FALSE-PASS DIRECTION -- the reason this change STRENGTHENS the gate.

    The committed content depends on a file that exists only in the shared
    working tree. Against the ambient tree the dependency resolves and the gate
    goes green on a combination that will never exist on trunk; against the
    commit's own tree it is correctly RED.
    """
    import tempfile
    dep_test = (
        "from pathlib import Path\n"
        "def test_dependency_present():\n"
        "    root = Path(__file__).resolve().parents[2]\n"
        "    assert (root / 'ree_core' / 'helper.py').exists(), 'helper.py missing'\n"
    )
    with tempfile.TemporaryDirectory() as td:
        repo = _repo(Path(td), committed={"tests/contracts/test_dep_lint.py": dep_test})
        _stage(repo, "experiments/v3_probe.py")
        _untracked(repo, "ree_core/helper.py", "HELPER = 1\n")   # never committed

        # Ambient tree: the dependency resolves -> the OLD gate said green.
        ambient = _run(repo, {"REE_PRECOMMIT_CONTRACTS_ISOLATE": "0"})
        assert ambient.returncode == 0, (
            "precondition failed: the ambient tree should satisfy the dependency\n"
            + ambient.stdout + ambient.stderr)

        # Commit's own tree: the dependency is absent -> BLOCK.
        isolated = _run(repo)
        assert isolated.returncode == 2, (
            "the gate passed a commit whose dependency exists only in another "
            "session's untracked file -- this is the false-PASS the isolation "
            "exists to close.\n" + isolated.stdout + isolated.stderr)
