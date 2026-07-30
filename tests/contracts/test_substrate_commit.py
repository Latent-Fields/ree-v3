"""Contracts for manifest_core.substrate_commit -- the WHICH-commit provenance field.

Motivating incident (V3-EXQ-614 vs V3-EXQ-614a, resolved 2026-07-30): two runs with
bit-identical driver bodies, identical seeds and a field-for-field identical
`config_summary` disagreed and flipped a verdict FAIL -> PASS, because ree-v3
`a45ca7f` changed `e3_diversity_entropy_lambda` 0.05 -> 0.5 between them. The corpus
already carried `substrate_hash`, a content hash that DETECTS such a difference but
is opaque, so it cannot say what changed. `substrate_commit` is the diagnosis half.

These tests build REAL git repos in a tmpdir rather than mocking subprocess, because
every bug found while writing this field was in the git-interface details -- the
pathlib-vs-git glob dialect mismatch, porcelain's leading-space column, and
empty-stdout-means-clean. A mock would have reproduced none of them.

TIME-INDEPENDENT and network-free. Runs anywhere git exists; skips cleanly if not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments._lib import manifest_core as mc  # noqa: E402


def _git(*args: str, cwd: Path) -> str:
    env = dict(os.environ)
    for k in mc._GIT_LOCATION_ENV_VARS:
        env.pop(k, None)
    env.setdefault("GIT_AUTHOR_NAME", "t")
    env.setdefault("GIT_AUTHOR_EMAIL", "t@example.com")
    env.setdefault("GIT_COMMITTER_NAME", "t")
    env.setdefault("GIT_COMMITTER_EMAIL", "t@example.com")
    return subprocess.run(
        ["git", *args], cwd=str(cwd), check=True,
        capture_output=True, text=True, env=env,
    ).stdout


def _have_git() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_git(), reason="git not available")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal repo shaped like ree-v3's substrate trees."""
    r = tmp_path / "ree-v3"
    (r / "ree_core" / "utils").mkdir(parents=True)
    (r / "experiments" / "_lib").mkdir(parents=True)
    # Top-level files in BOTH substrate trees -- these are the ones the git-glob
    # dialect silently missed, so they must exist for the regression test to bite.
    (r / "ree_core" / "agent.py").write_text("A = 1\n")
    (r / "ree_core" / "utils" / "config.py").write_text("LAMBDA = 0.05\n")
    (r / "experiments" / "_lib" / "manifest_core.py").write_text("B = 1\n")
    (r / "experiments" / "_harness.py").write_text("C = 1\n")
    (r / "experiments" / "_metrics.py").write_text("D = 1\n")
    # A NON-substrate file: coordination data many sessions edit concurrently.
    (r / "experiment_queue.json").write_text(json.dumps({"items": []}) + "\n")
    _git("init", "-q", "-b", "main", cwd=r)
    _git("add", "-A", cwd=r)
    _git("commit", "-q", "-m", "base", cwd=r)
    return r


def test_reports_commit_branch_and_clean_on_a_pristine_checkout(repo: Path):
    out = mc.substrate_commit(repo_root=repo)
    assert out is not None
    assert out["commit"] == _git("rev-parse", "HEAD", cwd=repo).strip()
    assert len(out["commit"]) == 40
    assert out["branch"] == "main"
    # The load-bearing assertion: CLEAN must be False, never None. `git status`
    # emits empty stdout when clean, and an `or None` idiom would turn the common
    # case into "unmeasured" on virtually every run.
    assert out["dirty"] is False
    assert "dirty_paths" not in out


def test_dirty_fires_on_a_top_level_substrate_file(repo: Path):
    """Regression: git's `**/` needs an intervening directory, pathlib's does not.

    `ree_core/**/*.py` selects `ree_core/agent.py` under Path.glob (so it IS in
    substrate_hash) but does NOT match it as a git pathspec. Passing the raw globs
    to `git status` therefore reported a clean tree while the hashed substrate had
    changed -- the exact silent under-report this field exists to prevent.
    """
    (repo / "ree_core" / "agent.py").write_text("A = 2\n")
    out = mc.substrate_commit(repo_root=repo)
    assert out["dirty"] is True
    assert out["dirty_paths"] == ["ree_core/agent.py"]


def test_dirty_fires_on_a_top_level_lib_file(repo: Path):
    (repo / "experiments" / "_lib" / "manifest_core.py").write_text("B = 2\n")
    out = mc.substrate_commit(repo_root=repo)
    assert out["dirty"] is True
    assert out["dirty_paths"] == ["experiments/_lib/manifest_core.py"]


def test_dirty_path_is_not_truncated(repo: Path):
    """Regression: porcelain v1 column 0 is a SPACE for an unstaged modification.

    A full .strip() of stdout eats it on the FIRST line only, shifting the fixed
    path offset by one and yielding 'xperiments/...' -- corrupted but plausible.
    """
    (repo / "experiments" / "_lib" / "manifest_core.py").write_text("B = 3\n")
    out = mc.substrate_commit(repo_root=repo)
    for p in out["dirty_paths"]:
        assert (repo / p).exists(), f"recorded dirty path does not resolve: {p!r}"
        assert not p.startswith("xperiments"), "leading character eaten"


def test_staged_and_untracked_substrate_changes_both_count(repo: Path):
    """Column 0 (staged) and '??' (untracked) must parse as well as column 1."""
    (repo / "ree_core" / "utils" / "config.py").write_text("LAMBDA = 0.5\n")
    _git("add", "ree_core/utils/config.py", cwd=repo)          # staged -> 'M '
    (repo / "ree_core" / "newmod.py").write_text("E = 1\n")     # untracked -> '??'
    out = mc.substrate_commit(repo_root=repo)
    assert out["dirty"] is True
    assert set(out["dirty_paths"]) == {"ree_core/utils/config.py", "ree_core/newmod.py"}
    for p in out["dirty_paths"]:
        assert (repo / p).exists()


def test_non_substrate_dirt_does_NOT_mark_the_substrate_dirty(repo: Path):
    """The scoping is the whole point of the flag.

    These are shared multi-session checkouts: an unrelated session with an open
    `experiment_queue.json` edit is the normal state, and an unscoped
    `git status --porcelain` would report dirty on nearly every run, making the
    field carry no information.
    """
    (repo / "experiment_queue.json").write_text(json.dumps({"items": [1]}) + "\n")
    (repo / "NOTES.md").write_text("scratch\n")
    out = mc.substrate_commit(repo_root=repo)
    assert out["dirty"] is False, "non-substrate edits must not flip the flag"


def test_returns_none_outside_a_git_repo(tmp_path: Path):
    """Fails open: absent beats wrong, and never raises into the experiment."""
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    assert mc.substrate_commit(repo_root=plain) is None


def test_ignores_inherited_git_location_env_vars(repo: Path, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
    """A parent git process (pre-commit hook, runner) exports GIT_DIR et al.

    Unstripped, `git rev-parse` resolves against the PARENT's repo and the manifest
    records a confidently wrong SHA -- worse than recording none.
    """
    other = tmp_path / "other"
    other.mkdir()
    (other / "f.txt").write_text("x\n")
    _git("init", "-q", "-b", "main", cwd=other)
    _git("add", "-A", cwd=other)
    _git("commit", "-q", "-m", "other", cwd=other)
    other_sha = _git("rev-parse", "HEAD", cwd=other).strip()
    repo_sha = _git("rev-parse", "HEAD", cwd=repo).strip()
    assert other_sha != repo_sha

    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    out = mc.substrate_commit(repo_root=repo)
    assert out["commit"] == repo_sha, "resolved against the inherited repo, not our own"


def test_pathspecs_cover_every_substrate_glob(repo: Path):
    """The derived pathspecs must be a SUPERSET of the hashed globs.

    Over-approximation is the safe direction (a spurious dirty costs one git diff);
    under-approximation is the silent miss. Asserted against arm_fingerprint's live
    globs so the two cannot drift apart unnoticed.
    """
    from experiments._lib import arm_fingerprint as afp
    globs = getattr(afp, "_SUBSTRATE_GLOBS", None)
    assert globs, "arm_fingerprint no longer exposes _SUBSTRATE_GLOBS"
    assert tuple(globs) == tuple(mc._SUBSTRATE_GLOBS_FALLBACK), (
        "manifest_core._SUBSTRATE_GLOBS_FALLBACK has drifted from arm_fingerprint"
    )
    specs = mc._git_pathspecs_from_globs(globs)
    for g in globs:
        head = g.split("*", 1)[0].rstrip("/")
        assert any(head == s or head.startswith(s + "/") or s == "." for s in specs), (
            f"glob {g!r} is not covered by pathspecs {specs!r}"
        )


def test_stamp_recording_core_fills_substrate_commit(repo: Path):
    m: dict = {}
    mc.stamp_recording_core(m, config={"k": 1}, seeds=[42], repo_root=repo)
    assert isinstance(m["substrate_commit"], dict)
    assert m["substrate_commit"]["commit"] == _git("rev-parse", "HEAD", cwd=repo).strip()


def test_stamp_does_not_clobber_an_author_set_value(repo: Path):
    m = {"substrate_commit": {"commit": "PRESET"}}
    mc.stamp_recording_core(m, repo_root=repo)
    assert m["substrate_commit"] == {"commit": "PRESET"}
    mc.stamp_recording_core(m, repo_root=repo, overwrite=True)
    assert m["substrate_commit"]["commit"] != "PRESET"


def test_stamp_survives_a_non_repo_without_raising(tmp_path: Path):
    """The staged tree used by remote_pytest.sh excludes .git/ -- stamping there
    must degrade to an absent field, never a crash or a phantom failure."""
    plain = tmp_path / "nope"
    plain.mkdir()
    m: dict = {}
    mc.stamp_recording_core(m, config={"k": 1}, seeds=[1], repo_root=plain)
    assert "substrate_commit" not in m
    assert m["recording_schema"] == mc.RECORDING_SCHEMA  # the rest still stamped


def test_substrate_commit_is_in_always_core_keys():
    assert "substrate_commit" in mc.ALWAYS_CORE_KEYS
    assert "substrate_commit" in mc.missing_core_fields({})


def test_validate_recording_standalone_fallback_matches_canonical_list():
    """validate_recording.py carries a hardcoded ALWAYS_CORE_KEYS copy for the
    standalone case where the package import fails. It went stale the moment
    substrate_commit was added -- pin it so the next added key cannot repeat that
    silently, which would make the linter under-report exactly the field the
    import path reports correctly.
    """
    src = (REPO_ROOT / "validate_recording.py").read_text(encoding="utf-8")
    start = src.index("ALWAYS_CORE_KEYS = (")
    literal = src[start + len("ALWAYS_CORE_KEYS = ("):]
    literal = literal[: literal.index(")")]
    fallback = tuple(
        tok.strip().strip('"').strip("'")
        for tok in literal.replace("\n", " ").split(",")
        if tok.strip()
    )
    assert fallback == tuple(mc.ALWAYS_CORE_KEYS), (
        "validate_recording.py's standalone fallback has drifted from "
        f"manifest_core.ALWAYS_CORE_KEYS: {fallback} != {tuple(mc.ALWAYS_CORE_KEYS)}"
    )


def test_dirty_paths_are_capped_but_the_cap_is_not_silent(repo: Path):
    """A truncated list must not read as the complete set.

    `dirty_count` records the true total, so `len(dirty_paths) < dirty_count` is
    the truncation signal. Understating the blast radius of a dirty substrate is
    exactly wrong at the moment someone is judging whether a run is trustworthy.
    """
    extra = 10
    for i in range(mc._MAX_DIRTY_PATHS + extra):
        (repo / "ree_core" / f"gen_{i:03d}.py").write_text("x = 1\n")
    out = mc.substrate_commit(repo_root=repo)
    assert out["dirty"] is True
    assert len(out["dirty_paths"]) == mc._MAX_DIRTY_PATHS
    assert out["dirty_count"] == mc._MAX_DIRTY_PATHS + extra
    assert len(out["dirty_paths"]) < out["dirty_count"]


def test_dirty_count_equals_list_length_when_under_the_cap(repo: Path):
    (repo / "ree_core" / "agent.py").write_text("A = 2\n")
    out = mc.substrate_commit(repo_root=repo)
    assert out["dirty_count"] == len(out["dirty_paths"]) == 1
