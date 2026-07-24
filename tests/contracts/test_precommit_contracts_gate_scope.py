"""Contracts for scripts/precommit_contracts.sh Block 2 trigger scope.

Block 2 runs the tests/contracts suite when the staged set touches the shared
substrate. It keyed on ^ree_core/ only until 2026-07-19, which left
experiments/_lib/** -- the shared training substrate (train_a2c,
mech457_bootstrap_explorer, mech457_fanout, capability_eval, arm_fingerprint),
consumed by every mech457-family experiment and bound into substrate_hash --
matching NO block at all: Block 1 globs experiments/v3_exq_*.py and Block 1b
globs experiments/v3_*.py. A change to the actual A2C training loop therefore
committed with no contract run and no warning (instance: the
mech457_retention_trajectory_probe build, ree-v3 7e4f6e932b).

These tests pin the trigger so it cannot silently regress to ree_core-only.
They run the real script against a synthetic throwaway git repo shaped like
ree-v3 (ree_core/ + tests/contracts/), so they exercise the actual grep and the
actual exit codes without touching the live repo or running the real ~1873-test
suite.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
SCRIPT = REPO_ROOT / "scripts" / "precommit_contracts.sh"

FIRED = "running contracts"  # Block 2's announcement on stderr


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


@pytest.fixture
def py_shim(tmp_path_factory):
    """A PATH-injectable `python3` that is guaranteed to have pytest.

    precommit_contracts.sh resolves its interpreter as /opt/local/bin/python3
    (the Mac/MacPorts project default) and falls back to `command -v python3`.
    On a cloud worker the MacPorts path does not exist and /usr/bin/python3 has
    no pytest, so the fallback yields an interpreter that CANNOT run the inner
    suite -- the script then exits 2 for an environment reason having nothing
    to do with the trigger under test.

    That is not hypothetical: it failed exactly this way on ree-cloud-4
    (2026-07-19), and it silently turned test_lib_change_blocks_on_failing_
    contracts into a FALSE PASS -- that test asserts exit 2, and got exit 2
    because pytest was missing rather than because the suite failed.

    Shimming `python3` to the interpreter running these tests makes the inner
    run real on every machine class. On the Mac the script still prefers the
    absolute /opt/local/bin/python3, which has pytest, so the shim is inert
    there.
    """
    bindir = tmp_path_factory.mktemp("shimbin")
    shim = bindir / "python3"
    shim.write_text('#!/bin/sh\nexec "%s" "$@"\n' % sys.executable)
    shim.chmod(0o755)
    return bindir


STUB_VALIDATE_EXPERIMENTS = (
    "import sys\n"
    "argv = sys.argv[1:]\n"
    "paths = argv[argv.index('--paths') + 1:] if '--paths' in argv else []\n"
    "for p in paths:\n"
    "    text = open(p).read()  # FileNotFoundError if REPO resolved to the wrong tree\n"
    "    if 'CONFORMS' not in text:\n"
    "        print('non-conforming:', p)\n"
    "        sys.exit(1)\n"
    "print('checked', len(paths), 'scripts: all conform')\n"
)


@pytest.fixture
def fake_repo(tmp_path):
    """A throwaway repo with the layout precommit_contracts.sh expects.

    Its tests/contracts holds ONE trivial passing test, so a Block 2 firing is
    fast and its exit code reflects the suite result rather than collection
    noise.
    """
    repo = tmp_path / "ree-v3"
    (repo / "ree_core").mkdir(parents=True)
    (repo / "tests" / "contracts").mkdir(parents=True)
    (repo / "experiments" / "_lib").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "docs").mkdir()

    (repo / "ree_core" / "thing.py").write_text("VALUE = 1\n")
    (repo / "experiments" / "_lib" / "shared.py").write_text("def train_a2c():\n    pass\n")
    (repo / "docs" / "notes.md").write_text("notes\n")
    (repo / "tests" / "contracts" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    # Stub validate_experiments.py: reads each staged path's CONTENT off disk,
    # exactly like the real tool, so a wrong-REPO resolution surfaces as
    # FileNotFoundError instead of being masked by an in-memory fixture.
    (repo / "validate_experiments.py").write_text(STUB_VALIDATE_EXPERIMENTS)

    # Copy the script under test in at its real relative location -- the script
    # falls back to resolving REPO from its own dirname/.. when
    # CLAUDE_PROJECT_DIR does not point at a tree containing ree-v3/.
    shutil.copy2(SCRIPT, repo / "scripts" / "precommit_contracts.sh")

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", "-A")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")
    return repo


@pytest.fixture
def fake_worktree(fake_repo):
    """A `git worktree add --detach` off fake_repo.

    This is the throwaway-worktree pattern the dazzling-taussig-f58f4c session
    used to isolate work from a busy, multi-session-dirty shared checkout, and
    the exact shape of the worktree-blindness bug: hooks are shared across all
    worktrees via the common .git dir, but a hardcoded/env-derived REPO cannot
    tell which tree a given commit is actually happening in.
    """
    wt = fake_repo.parent / "fake-repo-worktree"
    _git(fake_repo, "worktree", "add", "--detach", "-q", str(wt))
    return wt


def _run(repo, shim, *args, cwd=None):
    """Run the gate against `repo`'s current staged set.

    cwd defaults to `repo` itself, matching real git hook invocation: git
    invokes hooks with cwd already set to the root of the working tree the
    commit is happening in (githooks(5)), which is what
    precommit_contracts.sh's REPO resolution now relies on (`git rev-parse
    --show-toplevel`, tried before the CLAUDE_PROJECT_DIR/SCRIPT_DIR
    fallbacks). Without an explicit cwd here, the script would resolve
    against whatever directory the TEST RUNNER happens to be in -- not the
    synthetic `repo` fixture -- which would silently validate the wrong tree.
    """
    env = dict(os.environ)
    env.pop("CLAUDE_PROJECT_DIR", None)  # force script-location resolution
    env["PATH"] = "%s:%s" % (shim, env.get("PATH", ""))
    return subprocess.run(
        ["bash", str(repo / "scripts" / "precommit_contracts.sh"), *args],
        capture_output=True, text=True, env=env, cwd=str(cwd or repo),
    )


def _assert_inner_suite_really_ran(r):
    """Guard against the missing-pytest false pass.

    If the inner interpreter has no pytest the script still exits 2, which
    would satisfy a bare 'blocks the commit' assertion for the wrong reason.
    """
    assert "No module named pytest" not in r.stderr, (
        "the inner pytest could not start -- this result says nothing about "
        "the gate. stderr:\n" + r.stderr
    )


def _stage(repo, relpath, text):
    p = repo / relpath
    p.write_text(text)
    _git(repo, "add", str(relpath))


# ---- Block 2 trigger scope --------------------------------------------------

def test_lib_change_triggers_contracts(fake_repo, py_shim):
    """THE REGRESSION GUARD: a staged experiments/_lib/ path must fire Block 2."""
    _stage(fake_repo, "experiments/_lib/shared.py",
           "def train_a2c():\n    return 'hooked'\n")
    r = _run(fake_repo, py_shim)
    assert FIRED in r.stderr, (
        "experiments/_lib/ change did not trigger the contracts suite -- Block 2 "
        "has regressed to ree_core-only and shared-substrate changes commit "
        "ungated.\nstderr:\n" + r.stderr
    )
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 0, r.stderr


def test_ree_core_change_still_triggers_contracts(fake_repo, py_shim):
    """The pre-existing ree_core/ trigger must survive the widening."""
    _stage(fake_repo, "ree_core/thing.py", "VALUE = 2\n")
    r = _run(fake_repo, py_shim)
    assert FIRED in r.stderr, r.stderr
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 0, r.stderr


def test_unrelated_change_is_a_silent_noop(fake_repo, py_shim):
    """Self-gating shape preserved: a docs-only commit must not run anything.

    This is what keeps REE_assembly / other-repo commits unpenalised.
    """
    _stage(fake_repo, "docs/notes.md", "more notes\n")
    r = _run(fake_repo, py_shim)
    assert FIRED not in r.stderr
    assert r.returncode == 0
    assert r.stderr.strip() == "", "expected no output for an unrelated commit"


def test_lib_change_blocks_on_failing_contracts(fake_repo, py_shim):
    """A _lib change with a failing suite must BLOCK (exit 2), not warn."""
    (fake_repo / "tests" / "contracts" / "test_ok.py").write_text(
        "def test_ok():\n    assert False\n"
    )
    _git(fake_repo, "add", "tests/contracts/test_ok.py")
    _stage(fake_repo, "experiments/_lib/shared.py",
           "def train_a2c():\n    return 'broken'\n")
    r = _run(fake_repo, py_shim)
    assert FIRED in r.stderr, r.stderr
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 2, (
        "a failing suite under a _lib change must block the commit "
        f"(exit 2), got {r.returncode}\n" + r.stderr
    )


def test_no_block_flag_downgrades_lib_failure(fake_repo, py_shim):
    """--no-block keeps the advisory escape hatch working for _lib too."""
    (fake_repo / "tests" / "contracts" / "test_ok.py").write_text(
        "def test_ok():\n    assert False\n"
    )
    _git(fake_repo, "add", "tests/contracts/test_ok.py")
    _stage(fake_repo, "experiments/_lib/shared.py", "X = 1\n")
    r = _run(fake_repo, py_shim, "--no-block")
    _assert_inner_suite_really_ran(r)
    assert FIRED in r.stderr, r.stderr
    assert r.returncode == 0, r.stderr


# ---- worktree-awareness (dazzling-taussig-f58f4c, 2026-07-24) --------------

def test_worktree_commit_validates_its_own_new_script(fake_worktree, py_shim):
    """THE REGRESSION GUARD (failure mode 1): a NEW experiment script that
    exists ONLY in a `git worktree add` checkout must be found and validated
    THERE, not looked up on the primary/shared checkout's disk (where it does
    not exist).

    Before the fix, REPO was always resolved to the primary checkout
    regardless of which tree was actually being committed, so
    validate_experiments.py (invoked via `cd "$REPO" && ... --paths <relative
    path>`) raised FileNotFoundError for a script that only exists in the
    worktree.
    """
    rel = "experiments/v3_exq_999_worktree_only.py"
    script = fake_worktree / rel
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# CONFORMS\n")
    _git(fake_worktree, "add", rel)

    # Sanity: the file genuinely does not exist in the primary checkout.
    assert not (fake_worktree.parent / "ree-v3" / rel).exists()

    r = _run(fake_worktree, py_shim)
    assert "FileNotFoundError" not in r.stderr, (
        "Block 1 read the staged script off the wrong tree (the "
        "worktree-blindness bug) instead of the worktree actually being "
        "committed.\nstderr:\n" + r.stderr
    )
    assert r.returncode == 0, r.stderr


def test_worktree_commit_uses_its_own_contracts_suite(fake_worktree, py_shim):
    """THE REGRESSION GUARD (failure mode 2) for Block 2: a worktree commit
    must run the WORKTREE's own tests/contracts/, not the primary checkout's.

    Break the PRIMARY checkout's suite -- left deliberately UNCOMMITTED,
    simulating another session's unrelated in-flight edit sitting in the
    shared, multi-session-dirty checkout -- while the worktree's own copy
    stays healthy, then stage a _lib change only in the worktree. Before the
    fix, Block 2 always ran against the primary checkout and would have
    blocked on the primary's irrelevant failure (or, per the reported
    incident, collected garbage from whatever else was sitting uncommitted
    there).
    """
    primary = fake_worktree.parent / "ree-v3"
    (primary / "tests" / "contracts" / "test_ok.py").write_text(
        "def test_ok():\n    assert False  # unrelated: lives only in the shared checkout\n"
    )

    _stage(fake_worktree, "experiments/_lib/shared.py",
           "def train_a2c():\n    return 'worktree-local change'\n")
    r = _run(fake_worktree, py_shim)
    assert FIRED in r.stderr, r.stderr
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 0, (
        "the worktree's own (healthy) contracts suite should have run and "
        "passed; a non-zero exit means Block 2 ran against the shared "
        "checkout's unrelated broken tree instead.\nstderr:\n" + r.stderr
    )


def test_cwd_outside_ree_v3_falls_back_to_claude_project_dir(fake_repo, py_shim, tmp_path_factory):
    """Tier-2 fallback preserved: when cwd is NOT inside any ree-v3 tree at
    all -- e.g. a Claude Code umbrella-repo worktree session, which has no
    ree-v3 of its own, invoking this via the settings.json PreToolUse hook
    rather than the git-level one -- CLAUDE_PROJECT_DIR/ree-v3 must still be
    picked up. Tier 1 (git rev-parse --show-toplevel) resolving to an
    unrelated tree must not make the gate silently no-op.
    """
    _stage(fake_repo, "experiments/_lib/shared.py", "def train_a2c():\n    return 'x'\n")

    elsewhere = tmp_path_factory.mktemp("not-ree-v3")
    init = subprocess.run(["git", "init", "-q"], cwd=str(elsewhere), capture_output=True, text=True)
    assert init.returncode == 0, init.stderr

    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = str(fake_repo.parent)
    env["PATH"] = "%s:%s" % (py_shim, env.get("PATH", ""))
    r = subprocess.run(
        ["bash", str(fake_repo / "scripts" / "precommit_contracts.sh")],
        capture_output=True, text=True, env=env, cwd=str(elsewhere),
    )
    assert FIRED in r.stderr, (
        "cwd outside ree-v3 with CLAUDE_PROJECT_DIR set should still fall "
        "through to the CLAUDE_PROJECT_DIR/ree-v3 tier, not silently "
        "no-op.\nstderr:\n" + r.stderr
    )
    assert r.returncode == 0, r.stderr


# ---- the live script, not just the copy ------------------------------------

def test_live_script_greps_both_prefixes():
    """Pin the trigger in the committed script itself.

    The tests above run a copy in a synthetic tree; this asserts the real file
    carries both prefixes, so the guard cannot be satisfied by a fixture drift.
    """
    src = SCRIPT.read_text()
    assert "experiments/_lib/" in src, (
        "precommit_contracts.sh no longer mentions experiments/_lib/ -- the "
        "shared training substrate is ungated again"
    )
    assert "^(ree_core/|experiments/_lib/)" in src
