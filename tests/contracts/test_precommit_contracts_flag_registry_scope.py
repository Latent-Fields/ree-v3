"""Contracts for scripts/precommit_contracts.sh Block 1d (flag-registry gate).

chip-20260907-flag-registry-commit-time-gate: a new/renamed `use_*`/
`*_enabled` config flag that is not categorized into PROBED / KNOWN_INERT /
KNOWN_UNPROBED / KNOWN_UNPROBED_NESTED leaves
tests/test_flag_inertness.py::test_flag_registry_is_current red on trunk
until an unrelated later session stumbles on it -- confirmed three times
(GFLAG-0051/MECH-151 84e211a, SD-e1 ITEM 2 6447b45, SD-105 ba95c43, red for
three days). Block 2 does not already cover this: it runs `pytest
tests/contracts` only, and test_flag_inertness.py lives directly under
tests/, not tests/contracts/.

Block 1d closes that gap: a staged ree_core/utils/config.py triggers just
`tests/test_flag_inertness.py::test_flag_registry_is_current` -- a single
fast introspection test, run locally like Block 1c, not routed through Block
2's OOM-avoidance machinery.

These tests run the real script against a synthetic throwaway git repo, same
pattern as test_precommit_contracts_experiment_lint_scope.py and
test_precommit_contracts_gate_scope.py, so they exercise the actual grep and
exit codes without touching the live repo, its real config.py, or the real
flag registry.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
SCRIPT = REPO_ROOT / "scripts" / "precommit_contracts.sh"

FLAG_GATE_FIRED = "checking flag registry currency"  # Block 1d's announcement
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


PASSING_FLAG_TEST = (
    "def test_flag_registry_is_current():\n    assert True\n"
)
FAILING_FLAG_TEST = (
    "def test_flag_registry_is_current():\n"
    "    assert False, 'new/uncategorized flag(s): fake_new_flag_enabled'\n"
)


@pytest.fixture
def fake_repo(tmp_path):
    """A throwaway repo shaped like ree-v3, with ONE passing
    tests/test_flag_inertness.py::test_flag_registry_is_current and ONE
    unrelated tests/contracts/ test -- the split proves Block 1d targets only
    the single named test node, never the broader tests/contracts/ suite."""
    repo = tmp_path / "ree-v3"
    (repo / "ree_core" / "utils").mkdir(parents=True)
    (repo / "tests" / "contracts").mkdir(parents=True)
    (repo / "experiments" / "_lib").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "docs").mkdir()

    (repo / "ree_core" / "utils" / "config.py").write_text(
        "USE_SOMETHING = True\n"
    )
    (repo / "ree_core" / "thing.py").write_text("VALUE = 1\n")
    (repo / "experiments" / "_lib" / "shared.py").write_text("def train_a2c():\n    pass\n")
    (repo / "docs" / "notes.md").write_text("notes\n")
    (repo / "tests" / "test_flag_inertness.py").write_text(PASSING_FLAG_TEST)
    (repo / "tests" / "contracts" / "test_unrelated_contract.py").write_text(
        "def test_unrelated_contract():\n    assert True\n"
    )

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
    # Same reasoning as test_precommit_contracts_experiment_lint_scope.py's
    # _run(): pin TARGET=local so Block 2 (if it also fires) is deterministic
    # regardless of the ambient test-runner Mac's free memory.
    env.setdefault("REE_PRECOMMIT_CONTRACTS_TARGET", "local")
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


# ---- Block 1d trigger scope --------------------------------------------------

def test_staged_config_py_triggers_flag_registry_check(fake_repo, py_shim):
    """THE REGRESSION GUARD: a staged ree_core/utils/config.py, with neither
    ree_core/ (other than config.py itself is under ree_core/, so this also
    exercises the ree_core/ Block-2 case) nor experiments/_lib/ ALSO staged in
    a way that changes the outcome, must fire Block 1d."""
    _stage(fake_repo, "ree_core/utils/config.py", "USE_SOMETHING = True\nUSE_NEW_THING = True\n")
    r = _run(fake_repo, py_shim)
    assert FLAG_GATE_FIRED in r.stderr, (
        "a staged ree_core/utils/config.py did not trigger the flag-registry "
        "check -- Block 1d is not firing.\nstderr:\n" + r.stderr
    )
    _assert_inner_suite_really_ran(r)


def test_passing_flag_registry_does_not_block(fake_repo, py_shim):
    """A current (passing) flag registry must not block the commit."""
    # Must actually DIFFER from the base commit's content -- `git add` on an
    # unchanged file stages nothing, and the gate would never see it staged.
    _stage(fake_repo, "ree_core/utils/config.py",
           "USE_SOMETHING = True\nUSE_ANOTHER_THING = True\n")
    r = _run(fake_repo, py_shim)
    assert FLAG_GATE_FIRED in r.stderr, r.stderr
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 0, (
        "a passing flag registry blocked the commit.\nstderr:\n" + r.stderr
    )


def test_stale_flag_registry_blocks_commit(fake_repo, py_shim):
    """A stale (failing) flag registry under Block 1d must BLOCK (exit 2) --
    this is the whole point of the gate: catching the uncategorized-flag miss
    at commit time instead of days later on trunk."""
    (fake_repo / "tests" / "test_flag_inertness.py").write_text(FAILING_FLAG_TEST)
    _git(fake_repo, "add", "tests/test_flag_inertness.py")
    _stage(fake_repo, "ree_core/utils/config.py", "USE_SOMETHING = True\nUSE_NEW_THING = True\n")
    r = _run(fake_repo, py_shim)
    assert FLAG_GATE_FIRED in r.stderr, r.stderr
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 2, (
        "a stale flag registry did not block the commit (exit 2), got "
        f"{r.returncode}\n" + r.stderr
    )


def test_no_block_flag_downgrades_flag_registry_failure(fake_repo, py_shim):
    """--no-block keeps the advisory escape hatch working for Block 1d too."""
    (fake_repo / "tests" / "test_flag_inertness.py").write_text(FAILING_FLAG_TEST)
    _git(fake_repo, "add", "tests/test_flag_inertness.py")
    _stage(fake_repo, "ree_core/utils/config.py", "USE_SOMETHING = True\nUSE_NEW_THING = True\n")
    r = _run(fake_repo, py_shim, "--no-block")
    assert FLAG_GATE_FIRED in r.stderr, r.stderr
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 0, r.stderr


def test_other_ree_core_file_does_not_trigger_block_1d(fake_repo, py_shim):
    """A staged ree_core/ file OTHER than utils/config.py must not fire Block
    1d's own announcement -- Block 2 (the full suite) is that path's gate,
    not the narrow single-test one."""
    _stage(fake_repo, "ree_core/thing.py", "VALUE = 2\n")
    r = _run(fake_repo, py_shim)
    assert FLAG_GATE_FIRED not in r.stderr, r.stderr
    assert BLOCK2_FIRED in r.stderr, (
        "expected Block 2 to fire for a plain ree_core/ change\nstderr:\n" + r.stderr
    )
    assert r.returncode == 0, r.stderr


def test_unrelated_change_is_a_silent_noop(fake_repo, py_shim):
    """Self-gating shape preserved: a docs-only commit must not run anything."""
    _stage(fake_repo, "docs/notes.md", "more notes\n")
    r = _run(fake_repo, py_shim)
    assert FLAG_GATE_FIRED not in r.stderr
    assert BLOCK2_FIRED not in r.stderr
    assert r.returncode == 0
    assert r.stderr.strip() == "", "expected no output for an unrelated commit"


# ---- the live script, not just the copy ------------------------------------

def test_live_script_carries_block_1d():
    """Pin the trigger in the committed script itself, so the guard cannot be
    satisfied by a fixture drift (mirrors test_live_script_carries_block_1c
    in test_precommit_contracts_experiment_lint_scope.py)."""
    src = SCRIPT.read_text()
    assert "tests/test_flag_inertness.py::test_flag_registry_is_current" in src, (
        "precommit_contracts.sh no longer mentions the flag-registry test "
        "node id -- Block 1d has regressed or been removed"
    )
    assert "STAGED_FLAG_CONFIG" in src, (
        "precommit_contracts.sh no longer defines STAGED_FLAG_CONFIG -- "
        "Block 1d's trigger variable has been renamed or removed without "
        "updating this pin"
    )
