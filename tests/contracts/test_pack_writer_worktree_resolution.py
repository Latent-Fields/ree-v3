"""Contract tests for pack_writer.resolve_evidence_experiments_dir worktree-awareness.

Regression guard for the dazzling-taussig-f58f4c worktree-blindness fix
(2026-07-24): write_flat_manifest's output-directory resolution used to be
computed inline by every experiment script as
``Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"``,
which silently assumes the script's ree-v3 checkout sits directly under
REE_Working/ with REE_assembly as a sibling. That assumption breaks for a script
run from inside a `git worktree add` checkout nested under the repo's own tree
(confirmed real shape: ree-v3/.claude/worktrees/<slug>/experiments/<script>.py,
where parents[2] lands on ree-v3/.claude/worktrees/ instead of REE_Working/) --
``mkdir -p`` then happily creates a wrong tree there and the manifest is
silently dropped where the indexer never scans it. Confirmed 2026-07-24 during
a --dry-run smoke test for V3-EXQ-815 (session gracious-villani-1d1ac4), the
same class of bug already fixed the same day in scripts/git-hooks/pre-commit.local
(ree-v3 906466a) and scripts/remote_pytest.sh (REE_Working 2d572a1).

These tests exercise the real resolve_evidence_experiments_dir /
write_flat_manifest functions against synthetic throwaway git fixtures shaped
like REE_Working (a `ree-v3` checkout with a sibling `REE_assembly`), same
style as tests/contracts/test_precommit_contracts_gate_scope.py's
fake_repo/fake_worktree fixtures -- but calling the (pure-Python,
subprocess-free) functions directly rather than shelling out to a script.

ASCII-only. Run: pytest tests/contracts/test_pack_writer_worktree_resolution.py -q
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# conftest puts ree-v3 root on sys.path -> `experiments.*` importable.
from experiments import pack_writer as pw

_REE_V3_ROOT = Path(__file__).resolve().parents[2]


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


@pytest.fixture
def fake_working(tmp_path):
    """A throwaway REE_Working/ shape: ree-v3/ (a real git checkout with one
    committed experiment script) alongside REE_assembly/evidence/experiments/.

    Returns (fake_working, ree_v3_dir, script_path, expected_evidence_dir).
    """
    working = tmp_path / "REE_Working"
    ree_v3 = working / "ree-v3"
    (ree_v3 / "experiments").mkdir(parents=True)
    (ree_v3 / "experiments" / "some_script.py").write_text("# script\n")

    _git(ree_v3, "init", "-q")
    _git(ree_v3, "config", "user.email", "test@example.invalid")
    _git(ree_v3, "config", "user.name", "test")
    _git(ree_v3, "add", "-A")
    _git(ree_v3, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")

    evidence_dir = working / "REE_assembly" / "evidence" / "experiments"
    evidence_dir.mkdir(parents=True)

    return working, ree_v3, ree_v3 / "experiments" / "some_script.py", evidence_dir


@pytest.fixture
def fake_worktree(fake_working):
    """A `git worktree add --detach` nested INSIDE the ree-v3 checkout's own
    tree, at the exact real-world shape confirmed on disk:
    ree-v3/.claude/worktrees/<slug>/ -- the shape that broke the naive
    parents[2] arithmetic (it lands on ree-v3/.claude/worktrees/ instead of
    REE_Working/).

    Returns (script_path_inside_worktree, expected_evidence_dir).
    """
    working, ree_v3, _script, evidence_dir = fake_working
    wt = ree_v3 / ".claude" / "worktrees" / "fake-slug"
    _git(ree_v3, "worktree", "add", "--detach", "-q", str(wt))
    return wt / "experiments" / "some_script.py", evidence_dir


# ---- resolve_evidence_experiments_dir --------------------------------------

def test_primary_checkout_resolves_correctly(fake_working):
    """Sanity: the non-worktree (primary checkout) case must keep working."""
    _working, _ree_v3, script_path, expected = fake_working
    result = pw.resolve_evidence_experiments_dir(script_path)
    assert result == expected.resolve()


def test_worktree_commit_resolves_to_primary_assembly(fake_worktree):
    """THE REGRESSION GUARD: a script run from inside a nested ree-v3 worktree
    checkout must still resolve to the PRIMARY checkout's REE_assembly, not to
    some wrongly-nested path under the worktree's own directory tree.
    """
    script_path, expected = fake_worktree
    result = pw.resolve_evidence_experiments_dir(script_path)
    assert result == expected.resolve(), (
        "worktree-blindness regression: resolved to the wrong directory "
        f"({result}) instead of the primary REE_assembly ({expected})"
    )


def test_ignores_inherited_git_dir_env_vars(fake_worktree, tmp_path, monkeypatch):
    """A parent `git` process (e.g. a pre-commit hook shelling out to a python
    script) may leave GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE set in the
    environment. Left unstripped, `git rev-parse --git-common-dir` would
    resolve against whatever repo the PARENT pointed at instead of the
    script's own checkout. Point those vars at a wholly unrelated repo and
    confirm the resolver still finds the correct REE_assembly.
    """
    script_path, expected = fake_worktree

    unrelated = tmp_path / "unrelated_repo"
    unrelated.mkdir()
    _git(unrelated, "init", "-q")
    _git(unrelated, "config", "user.email", "test@example.invalid")
    _git(unrelated, "config", "user.name", "test")
    (unrelated / "f.txt").write_text("x\n")
    _git(unrelated, "add", "-A")
    _git(unrelated, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "unrelated")

    monkeypatch.setenv("GIT_DIR", str(unrelated / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(unrelated))

    result = pw.resolve_evidence_experiments_dir(script_path)
    assert result == expected.resolve(), (
        "inherited GIT_DIR/GIT_WORK_TREE misdirected resolution to the wrong "
        f"repo: got {result}, expected {expected}"
    )


def test_fixture_reproduces_the_naive_bug_shape(fake_worktree):
    """Guard the guard: confirm the fixture actually exercises the bug -- i.e.
    the historical naive parents[2] arithmetic, run against this exact
    worktree script path, really does land somewhere else. If this assertion
    ever fails, the fixture stopped reproducing the real-world shape and the
    regression guard above is not testing anything.
    """
    script_path, expected = fake_worktree
    naive = script_path.resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    assert naive != expected.resolve()
    assert "worktrees" in naive.parts


def test_falls_back_to_historical_arithmetic_when_nothing_findable(tmp_path):
    """No git repo, no discoverable REE_assembly anywhere nearby: must fall
    back to the historical hardcoded parents[2] arithmetic rather than raise.
    """
    orphan = tmp_path / "somewhere" / "unrelated" / "experiments" / "script.py"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("# orphan\n")

    result = pw.resolve_evidence_experiments_dir(orphan)
    expected_fallback = orphan.resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    assert result == expected_fallback


# ---- write_flat_manifest(out_dir=None) integration -------------------------

def test_write_flat_manifest_out_dir_none_uses_resolver(fake_worktree):
    """write_flat_manifest(out_dir=None, script_path=...) must write under the
    PRIMARY REE_assembly, not under the worktree-nested wrong path -- this is
    the actual call-site fix a retrofitted experiment script would adopt.
    """
    script_path, expected_dir = fake_worktree
    manifest = {
        "run_id": "some_experiment_20260724T000000_v3",
        "outcome": "PASS",
    }
    out_path = pw.write_flat_manifest(
        manifest, None, script_path=script_path, stamp=False,
    )
    assert out_path.parent == expected_dir.resolve()
    assert out_path.name == "some_experiment_20260724T000000_v3.json"
    assert out_path.is_file()


def test_write_flat_manifest_out_dir_none_requires_script_path():
    with pytest.raises(ValueError, match="out_dir or script_path"):
        pw.write_flat_manifest({"run_id": "x_v3", "outcome": "PASS"}, None)


# ---- pin the live module ----------------------------------------------------

def test_live_module_exports_resolver():
    """Pin that the fix is present in the committed file, not just in-memory."""
    src = (_REE_V3_ROOT / "experiments" / "pack_writer.py").read_text()
    assert "def resolve_evidence_experiments_dir(" in src
    assert "rev-parse" in src and "--git-common-dir" in src
