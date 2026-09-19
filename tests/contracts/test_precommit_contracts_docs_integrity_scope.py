"""Contracts for scripts/precommit_contracts.sh Block 1e (substrate-docs gate).

chip-20260919-wi1-index-contract-prose-false-positive:
tests/docs_integrity/test_wi1_substrate_split_index_integrity.py is a pure
text lint over docs/substrate/*.md and CLAUDE.md's index. While it lived in
tests/contracts/ it was run only by Block 2, which keys on ree_core/** and
experiments/_lib/** -- so the docs commit that broke it (ree-v3 65c1f72, one
prose bullet opening with a foreign id) was checked by nothing, and the red
blocked every UNRELATED ree_core commit fleet-wide for ~6.5h.

Two halves, and these tests pin both:

  * Block 1e fires the lint when docs/substrate/*.md or CLAUDE.md is staged,
    so the block lands on the author.
  * The lint is OUT of tests/contracts/, so Block 2 cannot run it and a docs
    defect cannot block a code commit. Moving it back re-arms the wedge.

Same synthetic-throwaway-repo pattern as
test_precommit_contracts_flag_registry_scope.py: the real script, a fake
ree-v3-shaped repo, no contact with the live docs or the live index.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
SCRIPT = REPO_ROOT / "scripts" / "precommit_contracts.sh"
LINT_RELPATH = "tests/docs_integrity/test_wi1_substrate_split_index_integrity.py"

DOCS_GATE_FIRED = "checking substrate index integrity"  # Block 1e's announcement
BLOCK2_FIRED = "running contracts"  # Block 2's announcement on stderr


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


@pytest.fixture
def py_shim(tmp_path_factory):
    """A PATH-injectable `python3` guaranteed to have pytest -- see the
    identically-named fixture in test_precommit_contracts_gate_scope.py."""
    bindir = tmp_path_factory.mktemp("shimbin")
    shim = bindir / "python3"
    shim.write_text('#!/bin/sh\nexec "%s" "$@"\n' % sys.executable)
    shim.chmod(0o755)
    return bindir


PASSING_LINT = "def test_index_integrity():\n    assert True\n"
FAILING_LINT = (
    "def test_index_integrity():\n"
    "    assert False, 'bullet names an id foreign to its heading: MECH-094'\n"
)


@pytest.fixture
def fake_repo(tmp_path):
    """A throwaway repo shaped like ree-v3 with ONE docs-integrity lint and ONE
    unrelated tests/contracts/ test. The contract test FAILS on purpose: if
    Block 1e ever widened to the contracts suite, a docs-only commit would go
    red here instead of passing."""
    repo = tmp_path / "ree-v3"
    (repo / "ree_core").mkdir(parents=True)
    (repo / "tests" / "contracts").mkdir(parents=True)
    (repo / "tests" / "docs_integrity").mkdir(parents=True)
    (repo / "experiments" / "_lib").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "docs" / "substrate").mkdir(parents=True)

    (repo / "ree_core" / "thing.py").write_text("VALUE = 1\n")
    (repo / "experiments" / "_lib" / "shared.py").write_text("def train_a2c():\n    pass\n")
    (repo / "docs" / "notes.md").write_text("notes\n")
    (repo / "docs" / "substrate" / "SD-001-thing.md").write_text("## SD-001 thing\n- SD-001: a.b\n")
    (repo / "CLAUDE.md").write_text("## Substrate feature index\n")
    (repo / LINT_RELPATH).write_text(PASSING_LINT)
    (repo / "tests" / "contracts" / "test_unrelated_contract.py").write_text(
        "def test_unrelated_contract():\n    assert False, 'contracts suite must not run for docs'\n"
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


# ---- Block 1e trigger scope --------------------------------------------------

@pytest.mark.parametrize("relpath", [
    "docs/substrate/SD-001-thing.md",
    "docs/substrate/MECH-018-brand-new-record.md",
    "CLAUDE.md",
])
def test_staged_substrate_docs_trigger_the_lint(fake_repo, py_shim, relpath):
    """THE REGRESSION GUARD: the exact shape of 65c1f72 (a docs/substrate file
    and/or CLAUDE.md, nothing else) must fire Block 1e -- and ONLY Block 1e:
    the deliberately-red contracts test in the fixture must not run."""
    _stage(fake_repo, relpath, "## SD-001 thing\n- SD-001: a.b\n- more\n")
    r = _run(fake_repo, py_shim)
    assert DOCS_GATE_FIRED in r.stderr, (
        "a staged %s did not trigger the substrate-index lint -- Block 1e is "
        "not firing.\nstderr:\n%s" % (relpath, r.stderr)
    )
    _assert_inner_suite_really_ran(r)
    assert BLOCK2_FIRED not in r.stderr, r.stderr
    assert r.returncode == 0, (
        "a passing docs lint blocked a docs commit (did Block 1e widen to "
        "tests/contracts?).\nstderr:\n" + r.stderr
    )


def test_failing_lint_blocks_the_docs_commit(fake_repo, py_shim):
    """The point of the block: the AUTHOR is stopped (exit 2), at commit time,
    with the remedy in the message."""
    (fake_repo / LINT_RELPATH).write_text(FAILING_LINT)
    _git(fake_repo, "add", LINT_RELPATH)
    _stage(fake_repo, "docs/substrate/SD-001-thing.md",
           "## SD-001 thing\n- MECH-094 does not apply here\n")
    r = _run(fake_repo, py_shim)
    assert DOCS_GATE_FIRED in r.stderr, r.stderr
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 2, (
        "a failing substrate-index lint did not block the commit (exit 2), "
        f"got {r.returncode}\n" + r.stderr
    )
    assert "Note: " in r.stderr, (
        "the block message no longer names the one-word remedy\n" + r.stderr
    )


def test_no_block_flag_downgrades_docs_lint_failure(fake_repo, py_shim):
    (fake_repo / LINT_RELPATH).write_text(FAILING_LINT)
    _git(fake_repo, "add", LINT_RELPATH)
    _stage(fake_repo, "CLAUDE.md", "## Substrate feature index\n- x\n")
    r = _run(fake_repo, py_shim, "--no-block")
    assert DOCS_GATE_FIRED in r.stderr, r.stderr
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 0, r.stderr


def test_a_red_docs_lint_cannot_block_a_code_commit(fake_repo, py_shim):
    """THE OTHER HALF, and the actual 2026-09-19 wedge: with the docs lint RED
    on trunk, a staged ree_core/ change must still get through Block 2. It can
    only do so because the lint is not under tests/contracts/."""
    (fake_repo / LINT_RELPATH).write_text(FAILING_LINT)
    (fake_repo / "tests" / "contracts" / "test_unrelated_contract.py").write_text(
        "def test_unrelated_contract():\n    assert True\n"
    )
    _git(fake_repo, "add", "-A")
    _git(fake_repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "red docs lint on trunk")
    _stage(fake_repo, "ree_core/thing.py", "VALUE = 2\n")
    r = _run(fake_repo, py_shim)
    assert DOCS_GATE_FIRED not in r.stderr, r.stderr
    assert BLOCK2_FIRED in r.stderr, r.stderr
    _assert_inner_suite_really_ran(r)
    assert r.returncode == 0, (
        "a red DOCS lint blocked a CODE commit -- the fleet-wide wedge is "
        "re-armed.\nstderr:\n" + r.stderr
    )


@pytest.mark.parametrize("relpath", [
    "docs/notes.md",
    "docs/substrate/nested/deeper.md",
    "docs/substrate/figure.png",
    "docs/CLAUDE.md",
])
def test_other_docs_are_a_silent_noop(fake_repo, py_shim, relpath):
    """Self-gating shape preserved: only top-level docs/substrate/*.md and the
    ROOT CLAUDE.md are what the lint reads, so only they trigger it."""
    _stage(fake_repo, relpath, "anything\n")
    r = _run(fake_repo, py_shim)
    assert DOCS_GATE_FIRED not in r.stderr
    assert BLOCK2_FIRED not in r.stderr
    assert r.returncode == 0
    assert r.stderr.strip() == "", "expected no output for an unrelated commit"


# ---- the live tree, not just the copy ---------------------------------------

def test_live_script_carries_block_1e():
    src = SCRIPT.read_text()
    assert LINT_RELPATH in src, (
        "precommit_contracts.sh no longer names the docs-integrity lint -- "
        "Block 1e has regressed or been removed"
    )
    assert "STAGED_SUBSTRATE_DOCS" in src, (
        "precommit_contracts.sh no longer defines STAGED_SUBSTRATE_DOCS -- "
        "Block 1e's trigger variable was renamed or removed without updating "
        "this pin"
    )


def test_live_lint_exists_where_block_1e_looks_for_it():
    """Block 1e is not `[ -f ]`-guarded on purpose (a missing file should block
    loudly, not fail open). This pin makes a rename fail HERE, with the cause
    named, rather than as a confusing pytest 'file not found' at commit time."""
    assert (REPO_ROOT / LINT_RELPATH).is_file(), (
        f"{LINT_RELPATH} is missing -- Block 1e would block every docs commit"
    )


def test_live_lint_is_not_under_tests_contracts():
    """Block 2 runs `pytest tests/contracts`. Any copy of the docs lint in there
    lets a docs defect block every ree_core commit again (2026-09-19, ~6.5h)."""
    strays = sorted(
        p.name for p in (REPO_ROOT / "tests" / "contracts").glob("test_wi1_substrate*")
    )
    assert strays == [], (
        f"docs-integrity lint found under tests/contracts/: {strays} -- move it "
        "back to tests/docs_integrity/ (see that file's module docstring)"
    )


def test_ci_runs_the_lint_on_docs_only_pushes():
    """Boxes with no commit guards (the cloud workers, where 65c1f72 was
    authored) never see Block 1e, and contract-tests.yml's `paths:` filter
    excludes docs/** and CLAUDE.md. docs-integrity.yml is what goes red on the
    pushed commit itself for them."""
    wf = REPO_ROOT / ".github" / "workflows" / "docs-integrity.yml"
    assert wf.is_file(), "docs-integrity.yml is missing"
    src = wf.read_text()
    for needle in ("docs/substrate/**", "CLAUDE.md", LINT_RELPATH):
        assert needle in src, f"docs-integrity.yml no longer mentions {needle!r}"
