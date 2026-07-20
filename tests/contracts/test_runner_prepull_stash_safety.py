"""Contract tests for prepull-stash safety and untracked-blocked pull recovery
in experiment_runner.git_pull.

Background -- 2026-07-20 ree-cloud-3 wedge:
  REE_assembly's `git pull` aborted every ~60s for an extended period with
  "untracked working tree files would be overwritten by merge". The blocking
  file was NESTED one level inside a run pack
  (`evidence/experiments/v3_exq_664_.../..._episode_log.json`), and the
  prepull stash only ever matched FLAT manifests -- so nothing in the runner
  could clear it and the worker's checkout stayed behind origin indefinitely.

  Two further defects made the same code path unsafe rather than merely
  ineffective:

  - _postpull_restore_prepull_stash inspected only `git stash list -1` and
    returned early unless the prepull stash was on TOP. Any intervening entry
    (typically the heartbeat's own autostash) stranded it. cloud-3 was still
    holding one from ~2026-06-12.
  - On pop failure it ran `git stash drop`, logging "paths likely on origin
    now". That inference is unsound: the stranded cloud-3 stash contained a
    V3-EXQ-673 manifest (FAIL / does_not_support) present at NO path on
    origin/master. A drop there destroys the only copy of a completed run.

Contracts:
  C1. _find_prepull_stash_ref locates the prepull stash by MESSAGE even when
      it is not the top entry.
  C2. A pop failure KEEPS the stash (never drops it) -- the evidence-loss
      guard.
  C3. A successful restore pops the prepull stash even with a foreign entry
      stacked on top of it.
  C4. _untracked_path_is_redundant is TRUE for byte-identical content and for
      an origin JSON superset, and FALSE when the path is absent upstream or
      any shared key's value differs.
  C5. git_pull recovers from an untracked-blocked pull when the blocking file
      is a NESTED run-pack file provably already on origin (the wedge), and
      does NOT delete a blocking file it cannot verify.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True,
        timeout=30,
    )


@pytest.fixture()
def origin_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """A bare origin plus a clone, both with an initial commit on master."""
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "-b", "master")
    _git(seed, "config", "user.email", "t@t.t")
    _git(seed, "config", "user.name", "t")
    (seed / "README.md").write_text("seed\n")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "-q", "-m", "seed")
    _git(seed, "clone", "-q", "--bare", str(seed), str(origin))

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)],
                   capture_output=True, timeout=60)
    _git(clone, "config", "user.email", "t@t.t")
    _git(clone, "config", "user.name", "t")
    return origin, clone


def _push_from_origin_side(origin: Path, tmp_path: Path, rel: str,
                           payload: dict) -> None:
    """Commit `rel` onto origin/master via a throwaway working clone."""
    work = tmp_path / f"work_{abs(hash(rel)) % 10000}"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)],
                   capture_output=True, timeout=60)
    _git(work, "config", "user.email", "t@t.t")
    _git(work, "config", "user.name", "t")
    p = work / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    _git(work, "add", rel)
    _git(work, "commit", "-q", "-m", f"add {rel}")
    _git(work, "push", "-q", "origin", "master")


# --- C1 / C2 / C3: prepull stash handling ---------------------------------

def _make_prepull_stash(clone: Path) -> None:
    rel = "evidence/experiments/v3_exq_999_probe.json"
    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"run_id": "v3_exq_999_probe"}) + "\n")
    _git(clone, "stash", "push", "--include-untracked", "-m",
         experiment_runner._PREPULL_STASH_MESSAGE, "--", rel)


def _make_foreign_stash(clone: Path) -> None:
    (clone / "README.md").write_text("locally modified\n")
    _git(clone, "stash", "push", "-m", "autostash")


def test_c1_find_prepull_stash_by_message_not_position(origin_and_clone):
    _, clone = origin_and_clone
    _make_prepull_stash(clone)
    _make_foreign_stash(clone)   # now on top

    top = _git(clone, "stash", "list", "-1", "--format=%s").stdout
    assert experiment_runner._PREPULL_STASH_MESSAGE not in top, (
        "fixture invalid: prepull stash must NOT be the top entry"
    )
    ref = experiment_runner._find_prepull_stash_ref(clone)
    assert ref == "stash@{1}"


def test_c2_pop_failure_keeps_stash(origin_and_clone, capsys):
    _, clone = origin_and_clone
    _make_prepull_stash(clone)
    # Recreate the stashed path so the pop collides and fails.
    rel = "evidence/experiments/v3_exq_999_probe.json"
    (clone / rel).parent.mkdir(parents=True, exist_ok=True)
    (clone / rel).write_text('{"run_id": "collision"}\n')

    before = _git(clone, "stash", "list").stdout
    experiment_runner._postpull_restore_prepull_stash(clone, "REE_assembly")
    after = _git(clone, "stash", "list").stdout

    assert experiment_runner._PREPULL_STASH_MESSAGE in after, (
        "prepull stash was DROPPED on pop failure -- this is the "
        "evidence-loss defect"
    )
    assert before == after
    assert "KEPT (not dropped)" in capsys.readouterr().out


def test_c3_restore_pops_prepull_stash_under_foreign_entry(origin_and_clone):
    _, clone = origin_and_clone
    rel = "evidence/experiments/v3_exq_999_probe.json"
    _make_prepull_stash(clone)
    _make_foreign_stash(clone)

    experiment_runner._postpull_restore_prepull_stash(clone, "REE_assembly")

    assert (clone / rel).exists(), "prepull stash content not restored"
    assert experiment_runner._PREPULL_STASH_MESSAGE not in \
        _git(clone, "stash", "list").stdout


# --- C4: redundancy proof --------------------------------------------------

def test_c4_redundancy_predicate(origin_and_clone, tmp_path):
    origin, clone = origin_and_clone
    rel = "evidence/experiments/v3_exq_664_pack/episode_log.json"
    worker_payload = {"run_id": "v3_exq_664", "episodes": [1, 2, 3]}
    origin_payload = dict(worker_payload)
    origin_payload.update({"machine": "ree-cloud-3", "queue_id": "V3-EXQ-664",
                           "evidence_direction": "supports"})
    _push_from_origin_side(origin, tmp_path, rel, origin_payload)
    _git(clone, "fetch", "-q", "origin")

    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)

    # superset -> redundant
    p.write_text(json.dumps(worker_payload, indent=2) + "\n")
    assert experiment_runner._untracked_path_is_redundant(
        clone, rel, "origin/master") is True

    # byte-identical -> redundant
    p.write_text(json.dumps(origin_payload, indent=2) + "\n")
    assert experiment_runner._untracked_path_is_redundant(
        clone, rel, "origin/master") is True

    # a shared key disagrees -> NOT redundant
    diverged = dict(worker_payload)
    diverged["episodes"] = [9, 9, 9]
    p.write_text(json.dumps(diverged, indent=2) + "\n")
    assert experiment_runner._untracked_path_is_redundant(
        clone, rel, "origin/master") is False

    # absent upstream -> NOT redundant
    other = clone / "evidence/experiments/v3_exq_673_missing.json"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text(json.dumps({"run_id": "v3_exq_673"}) + "\n")
    assert experiment_runner._untracked_path_is_redundant(
        clone, "evidence/experiments/v3_exq_673_missing.json",
        "origin/master") is False


# --- C5: end-to-end wedge recovery ----------------------------------------

def test_c5_nested_untracked_wedge_recovers(origin_and_clone, tmp_path,
                                            capsys):
    origin, clone = origin_and_clone
    rel = "evidence/experiments/v3_exq_664_pack/episode_log.json"
    payload = {"run_id": "v3_exq_664", "episodes": [1, 2, 3]}
    _push_from_origin_side(origin, tmp_path, rel,
                           {**payload, "machine": "ree-cloud-3"})

    # Worker has the same run pack untracked -- the pull cannot proceed.
    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")

    blocked = _git(clone, "pull", "--rebase", "--autostash")
    assert blocked.returncode != 0, "fixture invalid: pull should be blocked"
    assert "untracked working tree files would be overwritten" in \
        (blocked.stderr + blocked.stdout).lower()

    experiment_runner.git_pull(clone, "REE_assembly")

    behind = _git(clone, "rev-list", "--count", "HEAD..origin/master")
    assert behind.stdout.strip() == "0", (
        f"pull still wedged: {capsys.readouterr().out}"
    )
    assert json.loads((clone / rel).read_text())["machine"] == "ree-cloud-3"


def test_c5_unverifiable_blocking_file_is_not_deleted(origin_and_clone,
                                                      tmp_path, capsys):
    origin, clone = origin_and_clone
    rel = "evidence/experiments/v3_exq_664_pack/episode_log.json"
    _push_from_origin_side(origin, tmp_path, rel,
                           {"run_id": "v3_exq_664", "episodes": [1, 2, 3]})

    # Local content DISAGREES with origin on a shared key -- unprovable.
    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    local_text = json.dumps({"run_id": "v3_exq_664",
                             "episodes": [7, 7, 7]}, indent=2) + "\n"
    p.write_text(local_text)

    experiment_runner.git_pull(clone, "REE_assembly")

    assert p.exists(), "unverifiable blocking file was DELETED"
    assert p.read_text() == local_text
    assert "LEFT IN PLACE" in capsys.readouterr().out
