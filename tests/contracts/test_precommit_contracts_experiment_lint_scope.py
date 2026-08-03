"""Contracts for scripts/precommit_contracts.sh Block 1c (experiment-lint trigger).

Block 2 runs the full contracts suite -- all 47 tests/contracts/test_*_lint.py
corpus lints included -- but only when ree_core/ or experiments/_lib/ is
staged. A brand-new experiments/v3_exq_*.py, the single most common artifact
/queue-experiment produces, touches neither: it got only Block 1 (conformance)
and Block 1b (manifest-writer only), so it could introduce a fresh instance of
an already-known bad pattern with nothing catching it until some unrelated
later commit happened to touch ree_core/ or _lib/.

Block 1c closes that gap: a staged experiments/*.py outside _lib/, when Block
2 will NOT already run, triggers just the test_*_lint.py subset (not the full
suite -- cheap, dominated by the shared corpus-scan fixture rather than the
slow non-lint contracts).

These tests run the real script against a synthetic throwaway git repo, same
pattern as test_precommit_contracts_gate_scope.py, so they exercise the actual
grep and exit codes without touching the live repo or its real corpus.

See REE_assembly/evidence/planning/experiment_verification_harness_plan.md
(Gap 1).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
SCRIPT = REPO_ROOT / "scripts" / "precommit_contracts.sh"

LINT_FIRED = "running corpus-lint subset"  # Block 1c's announcement on stderr
BLOCK2_FIRED = "running contracts"  # Block 2's announcement on stderr


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


@pytest.fixture
def py_shim(tmp_path_factory):
    """A PATH-injectable `python3` guaranteed to have pytest -- see the
    identically-named fixture in test_precommit_contracts_gate_scope.py for
    why this is required rather than trusting /opt/local/bin/python3 or the
    bare `python3` fallback.
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
    "    text = open(p).read()\n"
    "    if 'CONFORMS' not in text:\n"
    "        print('non-conforming:', p)\n"
    "        sys.exit(1)\n"
    "print('checked', len(paths), 'scripts: all conform')\n"
)


@pytest.fixture
def fake_repo(tmp_path):
    """A throwaway repo shaped like ree-v3, with ONE passing lint test and ONE
    passing non-lint contract test in tests/contracts/ -- the split lets a
    test below prove Block 1c ran only the lint subset, not the full suite,
    by breaking the non-lint test and asserting it does NOT block.
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
    (repo / "tests" / "contracts" / "test_something_lint.py").write_text(
        "def test_something_lint():\n    assert True\n"
    )
    (repo / "tests" / "contracts" / "test_unrelated_contract.py").write_text(
        "def test_unrelated_contract():\n    assert True\n"
    )
    (repo / "validate_experiments.py").write_text(STUB_VALIDATE_EXPERIMENTS)

    shutil.copy2(SCRIPT, repo / "scripts" / "precommit_contracts.sh")

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", "-A")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")
    return repo


def _run(repo, shim, *args):
    env = dict(os.environ)
    env.pop("CLAUDE_PROJECT_DIR", None)
    env["PATH"] = "%s:%s" % (shim, env.get("PATH", ""))
    return subprocess.run(
        ["bash", str(repo / "scripts" / "precommit_contracts.sh"), *args],
        capture_output=True, text=True, env=env, cwd=str(repo),
    )


def _assert_inner_suite_really_ran(r):
    assert "No module named pytest" not in r.stderr, (
        "the inner pytest could not start -- this result says nothing about "
        "the gate. stderr:\n" + r.stderr
    )


def _stage(repo, relpath, text):
    p = repo / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    _git(repo, "add", str(relpath))


# ---- Block 1c trigger scope --------------------------------------------------

def test_new_experiment_script_triggers_lint_subset(fake_repo, py_shim):
    """THE REGRESSION GUARD: a new experiments/v3_exq_*.py outside _lib/, with
    neither ree_core/ nor _lib/ also staged, must fire Block 1c."""
    _stage(fake_repo, "experiments/v3_exq_900_new_thing.py", "# CONFORMS\n")
    r = _run(fake_repo, py_shim)
    assert LINT_FIRED in r.stderr, (
        "a new experiment script did not trigger the corpus-lint subset -- "
        "Block 1c is not firing.\nstderr:\n" + r.stderr
    )
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 0, r.stderr


def test_lint_subset_runs_only_lints_not_full_suite(fake_repo, py_shim):
    """Block 1c must run ONLY the test_*_lint.py subset -- a failure in a
    non-lint contract test must NOT block the commit."""
    (fake_repo / "tests" / "contracts" / "test_unrelated_contract.py").write_text(
        "def test_unrelated_contract():\n    assert False  # not a lint\n"
    )
    _git(fake_repo, "add", "tests/contracts/test_unrelated_contract.py")
    _stage(fake_repo, "experiments/v3_exq_901_new_thing.py", "# CONFORMS\n")
    r = _run(fake_repo, py_shim)
    assert LINT_FIRED in r.stderr, r.stderr
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 0, (
        "a failure in a NON-lint contract test blocked the commit -- Block "
        "1c is running more than the test_*_lint.py subset.\nstderr:\n" + r.stderr
    )


def test_failing_lint_blocks_commit(fake_repo, py_shim):
    """A failing corpus lint under Block 1c must BLOCK (exit 2)."""
    (fake_repo / "tests" / "contracts" / "test_something_lint.py").write_text(
        "def test_something_lint():\n    assert False\n"
    )
    _git(fake_repo, "add", "tests/contracts/test_something_lint.py")
    _stage(fake_repo, "experiments/v3_exq_902_new_thing.py", "# CONFORMS\n")
    r = _run(fake_repo, py_shim)
    assert LINT_FIRED in r.stderr, r.stderr
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 2, (
        "a failing corpus lint under a staged experiment script must block "
        f"the commit (exit 2), got {r.returncode}\n" + r.stderr
    )


def test_no_block_flag_downgrades_lint_failure(fake_repo, py_shim):
    """--no-block keeps the advisory escape hatch working for Block 1c too."""
    (fake_repo / "tests" / "contracts" / "test_something_lint.py").write_text(
        "def test_something_lint():\n    assert False\n"
    )
    _git(fake_repo, "add", "tests/contracts/test_something_lint.py")
    _stage(fake_repo, "experiments/v3_exq_903_new_thing.py", "# CONFORMS\n")
    r = _run(fake_repo, py_shim, "--no-block")
    _assert_inner_suite_really_ran(r)
    assert LINT_FIRED in r.stderr, r.stderr
    assert r.returncode == 0, r.stderr


def test_lib_staged_alongside_skips_lint_block_redundancy(fake_repo, py_shim):
    """When _lib/ is ALSO staged, Block 2 already covers every lint -- Block
    1c must not redundantly re-run the subset a second time."""
    _stage(fake_repo, "experiments/_lib/shared.py",
           "def train_a2c():\n    return 'x'\n")
    _stage(fake_repo, "experiments/v3_exq_904_new_thing.py", "# CONFORMS\n")
    r = _run(fake_repo, py_shim)
    assert BLOCK2_FIRED in r.stderr, (
        "expected Block 2 (full suite) to fire when _lib/ is staged\nstderr:\n" + r.stderr
    )
    assert LINT_FIRED not in r.stderr, (
        "Block 1c redundantly re-ran the lint subset even though Block 2 "
        "already covers it.\nstderr:\n" + r.stderr
    )
    assert r.returncode == 0, r.stderr


def test_lib_only_change_does_not_trigger_block_1c(fake_repo, py_shim):
    """A staged experiments/_lib/ path alone (no other experiments/*.py) must
    not fire Block 1c's own announcement -- it is excluded by the _lib/ grep
    -v and is already Block-2's job."""
    _stage(fake_repo, "experiments/_lib/shared.py",
           "def train_a2c():\n    return 'y'\n")
    r = _run(fake_repo, py_shim)
    assert LINT_FIRED not in r.stderr, r.stderr
    assert BLOCK2_FIRED in r.stderr, r.stderr
    assert r.returncode == 0, r.stderr


def test_unrelated_change_is_a_silent_noop(fake_repo, py_shim):
    """Self-gating shape preserved: a docs-only commit must not run anything."""
    _stage(fake_repo, "docs/notes.md", "more notes\n")
    r = _run(fake_repo, py_shim)
    assert LINT_FIRED not in r.stderr
    assert BLOCK2_FIRED not in r.stderr
    assert r.returncode == 0
    assert r.stderr.strip() == "", "expected no output for an unrelated commit"


# ---- the live script, not just the copy ------------------------------------

def test_live_script_carries_block_1c():
    """Pin the trigger in the committed script itself, so the guard cannot be
    satisfied by a fixture drift (mirrors test_live_script_greps_both_prefixes
    in test_precommit_contracts_gate_scope.py)."""
    src = SCRIPT.read_text()
    assert "test_*_lint.py" in src, (
        "precommit_contracts.sh no longer mentions the test_*_lint.py subset "
        "-- Block 1c has regressed or been removed"
    )
    assert "STAGED_EXPERIMENT_PY" in src
