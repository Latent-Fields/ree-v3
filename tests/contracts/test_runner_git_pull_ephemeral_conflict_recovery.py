"""Contract tests for ephemeral-conflict recovery in
experiment_runner.git_pull.

Background -- 2026-05-31 cloud-3 wedge (17:48-18:16Z):
  cloud-3 claimed V3-EXQ-618 (worker mutated experiment_queue.json with
  claimed_by=ree-cloud-3 and pushed), ran the experiment to PASS, and the
  hub's phase3_queue_writer dropped the completed entry from origin on its
  next tick. cloud-3's next runner loop ran `git pull --rebase --autostash`:
  the rebase applied origin's snapshot (no V3-EXQ-618) cleanly but the
  autostash pop produced UU markers in experiment_queue.json (autostash
  held the claimed entry; origin had it absent). The wedge:

    - Preflight test_queue_schema_valid failed on the conflict markers.
    - Every subsequent pull failed with "Pulling is not possible because
      you have unmerged files."
    - After 5 failed-loop retries within 15 min, systemd's
      Restart=on-failure + StartLimitBurst=5 marked ree-runner failed.
    - cloud-3 sat dead for 10 min while V3-EXQ-621 went unclaimed.

Fix:
  The worker's local mutations to the queue file claim flag and to
  heartbeat/status/commands files are NOT canonical -- the hub writer is
  the sole git writer for these paths under Phase 3, and any worker-side
  state is either already pushed (queue claim) or about to be re-emitted
  next tick (heartbeat). git_pull therefore detects UU markers on these
  ephemeral paths and auto-resolves by taking origin's version, then
  retries the pull. Non-ephemeral UU markers are left in place with a
  warning -- those would indicate something the operator must inspect.

Contracts:
  C1. Stash-pop conflict on experiment_queue.json (the canonical cloud-3
      wedge) -- git_pull resolves it and leaves the working tree clean.
  C2. Pre-existing UU state from a prior wedge -- git_pull heals it
      before attempting the new pull (so the wedge cannot persist across
      ticks even if the recovery on the conflicting tick somehow misses).
  C3. UU on a non-ephemeral path -- git_pull does NOT auto-resolve it
      (the file is preserved untouched) so the operator can inspect.
  C4. autostash entry dropped after recovery so the stash stack doesn't
      accumulate dormant entries (cloud-3 had 191; that's the historical
      scarred timeline of many of these races).
  C5. _path_is_ephemeral_worker_owned predicate covers the queue file +
      the three heartbeat / status / commands dir prefixes and rejects
      other evidence/experiments/ paths.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _make_repos(root: Path) -> tuple[Path, Path]:
    """Bare remote + two clones with a single shared baseline commit."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", "remote.git"],
                   cwd=str(root), check=True, capture_output=True)
    local = root / "local"
    other = root / "other"
    for d in (local, other):
        subprocess.run(["git", "clone", "remote.git", d.name],
                       cwd=str(root), check=True, capture_output=True)
    queue_seed = {"items": [
        {"queue_id": "V3-EXQ-001", "experiment_type": "fake",
         "title": "seed entry", "machine_affinity": "any"},
    ]}
    (local / "experiment_queue.json").write_text(
        json.dumps(queue_seed, indent=2) + "\n")
    (local / "README.md").write_text("seed\n")
    _run(["git", "config", "user.email", "t@t"], local)
    _run(["git", "config", "user.name", "T"], local)
    _run(["git", "add", "-A"], local)
    _run(["git", "commit", "-m", "init"], local)
    _run(["git", "push", "origin", "HEAD:master"], local)
    _run(["git", "config", "user.email", "t@t"], other)
    _run(["git", "config", "user.name", "T"], other)
    _run(["git", "fetch", "origin"], other)
    _run(["git", "reset", "--hard", "origin/master"], other)
    return local, other


def _porcelain_uu(repo: Path) -> list[str]:
    """Return list of UU paths from `git status --porcelain`."""
    r = _run(["git", "status", "--porcelain"], repo)
    out = []
    for line in r.stdout.splitlines():
        if len(line) >= 4 and ("U" in line[:2] or line[:2] in ("AA", "DD")):
            out.append(line[3:].strip())
    return out


def _file_text(p: Path) -> str:
    try:
        return p.read_text()
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# C5: ephemeral-path predicate
# ---------------------------------------------------------------------------
def test_c5_ephemeral_path_predicate():
    f = experiment_runner._path_is_ephemeral_worker_owned
    # Queue file (ree-v3).
    assert f("experiment_queue.json")
    # Heartbeat / status / commands dirs (REE_assembly).
    assert f("evidence/experiments/runner_heartbeats/ree-cloud-3.json")
    assert f("evidence/experiments/runner_status/DLAPTOP-4.local.json")
    assert f("evidence/experiments/runner_commands/Daniel-PC.json")
    # Other evidence files MUST NOT match -- manifests are canonical
    # worker output, not ephemeral.
    assert not f("evidence/experiments/v3_exq_618_fake_v3.json")
    assert not f("evidence/experiments/_runner_signals/V3-EXQ-618.json")
    assert not f("evidence/planning/substrate_queue.json")
    # Empty / quoted / leading whitespace handled.
    assert not f("")
    assert not f("   ")
    assert f('"experiment_queue.json"')


# ---------------------------------------------------------------------------
# Build the cloud-3 wedge signature
# ---------------------------------------------------------------------------
def _build_cloud_3_wedge(local: Path, other: Path) -> None:
    """Reproduce the 2026-05-31 cloud-3 wedge signature:

    - local has experiment_queue.json mutated with claimed_by (worker
      mutation; not yet committed).
    - origin has the entry DROPPED (hub writer completed it).
    - git pull --rebase --autostash leaves UU markers because the
      autostash pop conflicts with the rebased state.
    """
    # Worker mutates queue locally with the claim flag.
    local_queue = local / "experiment_queue.json"
    data = json.loads(local_queue.read_text())
    data["items"][0]["claimed_by"] = {
        "machine": "ree-cloud-3",
        "claimed_at": "2026-05-31T17:48:28Z",
    }
    local_queue.write_text(json.dumps(data, indent=2) + "\n")

    # Hub writer drops the completed entry on origin via `other`.
    other_queue = other / "experiment_queue.json"
    other_queue.write_text(json.dumps({"items": []}, indent=2) + "\n")
    _run(["git", "add", "-A"], other)
    _run(["git", "commit", "-m", "phase3-queue: dropped V3-EXQ-001"], other)
    push = _run(["git", "push", "origin", "HEAD:master"], other)
    assert push.returncode == 0, f"other push failed: {push.stderr}"

    # Trigger the wedge: pull --rebase --autostash leaves the working
    # tree with UU markers (rebase fast-forwarded; autostash pop
    # conflicted because both sides modified queue.json). Modern git
    # returns exit 0 even though the pop failed -- which is the trap
    # that wedged cloud-3. Verify the wedge signature.
    pr = _run(["git", "pull", "--rebase", "--autostash",
               "origin", "master"], local)
    assert "autostash resulted in conflicts" in (pr.stdout + pr.stderr), (
        f"expected autostash pop conflict line; got:\n"
        f"stdout: {pr.stdout}\nstderr: {pr.stderr}"
    )
    uu = _porcelain_uu(local)
    assert "experiment_queue.json" in uu, (
        f"expected UU on experiment_queue.json; got {uu}"
    )
    # Confirm the conflict markers actually landed on disk.
    contents = (local / "experiment_queue.json").read_text()
    assert "<<<<<<<" in contents and "=======" in contents, (
        "expected conflict markers in queue file"
    )


# ---------------------------------------------------------------------------
# C1: canonical cloud-3 wedge recovers
# ---------------------------------------------------------------------------
def test_c1_cloud3_wedge_recovers(tmp_path, capsys):
    local, other = _make_repos(tmp_path)
    _build_cloud_3_wedge(local, other)

    # Confirm we are in the wedge state.
    assert _porcelain_uu(local) == ["experiment_queue.json"]

    # Heal via the runner's recovery helper.
    ok = experiment_runner._recover_ephemeral_pull_conflict(local, "ree-v3")
    assert ok, "recovery returned False"
    assert _porcelain_uu(local) == [], (
        f"working tree still has UU after recovery: {_porcelain_uu(local)}"
    )

    # Origin's version is what landed on disk -- the dropped entry stays
    # dropped, the worker's claim flag is gone (it was meaningless on a
    # completed item).
    queue = json.loads((local / "experiment_queue.json").read_text())
    assert queue == {"items": []}, (
        f"queue should match origin (no entries); got {queue}"
    )


# ---------------------------------------------------------------------------
# C2: end-to-end git_pull heals the wedge and the post-recovery pull is clean
# ---------------------------------------------------------------------------
def test_c2_git_pull_end_to_end_recovery(tmp_path, capsys):
    local, other = _make_repos(tmp_path)
    _build_cloud_3_wedge(local, other)
    assert _porcelain_uu(local) == ["experiment_queue.json"]

    experiment_runner.git_pull(local, "ree-v3")

    # Working tree must be clean of UU markers; queue must match origin.
    assert _porcelain_uu(local) == [], (
        f"git_pull left UU markers: {_porcelain_uu(local)}"
    )
    queue = json.loads((local / "experiment_queue.json").read_text())
    assert queue == {"items": []}, (
        f"expected empty items after pull; got {queue}"
    )

    # Subsequent pulls must continue to succeed (no wedge across ticks).
    experiment_runner.git_pull(local, "ree-v3")
    assert _porcelain_uu(local) == []


# ---------------------------------------------------------------------------
# C3: non-ephemeral UU is left in place (operator must inspect)
# ---------------------------------------------------------------------------
def test_c3_non_ephemeral_conflict_preserved(tmp_path, capsys):
    local, other = _make_repos(tmp_path)

    # Worker dirties README locally (NOT ephemeral).
    (local / "README.md").write_text("worker-side change\n")
    # Origin changes README too -- guarantees pop conflict.
    (other / "README.md").write_text("hub-side change\n")
    _run(["git", "add", "-A"], other)
    _run(["git", "commit", "-m", "hub readme"], other)
    _run(["git", "push", "origin", "HEAD:master"], other)

    pr = _run(["git", "pull", "--rebase", "--autostash",
               "origin", "master"], local)
    assert "autostash resulted in conflicts" in (pr.stdout + pr.stderr), (
        pr.stdout + pr.stderr)
    assert "README.md" in _porcelain_uu(local)

    # Recovery must refuse to touch the non-ephemeral conflict.
    ok = experiment_runner._recover_ephemeral_pull_conflict(local, "ree-v3")
    assert not ok, (
        "recovery should refuse to touch non-ephemeral conflicts"
    )
    assert "README.md" in _porcelain_uu(local), (
        "README.md UU must be preserved for manual inspection"
    )
    # The README's worker-side content must remain (no destructive reset).
    assert "worker-side change" in _file_text(local / "README.md") or \
           "<<<<<<<" in _file_text(local / "README.md"), (
        "worker's edits to non-ephemeral file should not be destroyed"
    )


# ---------------------------------------------------------------------------
# C4: autostash dropped after recovery
# ---------------------------------------------------------------------------
def test_c4_autostash_dropped_after_recovery(tmp_path):
    local, other = _make_repos(tmp_path)
    _build_cloud_3_wedge(local, other)

    # Confirm an autostash entry is parked on the stack pre-recovery.
    sl_pre = _run(["git", "stash", "list"], local).stdout
    assert "autostash" in sl_pre, (
        f"expected autostash entry pre-recovery; got: {sl_pre}"
    )

    ok = experiment_runner._recover_ephemeral_pull_conflict(local, "ree-v3")
    assert ok

    sl_post = _run(["git", "stash", "list"], local).stdout
    assert "autostash" not in sl_post, (
        f"autostash entry should be dropped post-recovery; "
        f"got: {sl_post}"
    )


# ---------------------------------------------------------------------------
# C6: mixed conflict (ephemeral + non-ephemeral) refused to avoid partial fix
# ---------------------------------------------------------------------------
def test_c6_mixed_conflict_refused(tmp_path):
    local, other = _make_repos(tmp_path)

    # Worker dirties BOTH the queue file (ephemeral) AND README (not).
    data = json.loads((local / "experiment_queue.json").read_text())
    data["items"][0]["claimed_by"] = {"machine": "ree-cloud-3",
                                      "claimed_at": "x"}
    (local / "experiment_queue.json").write_text(
        json.dumps(data, indent=2) + "\n")
    (local / "README.md").write_text("worker readme\n")

    # Origin diverges on both.
    (other / "experiment_queue.json").write_text(
        json.dumps({"items": []}, indent=2) + "\n")
    (other / "README.md").write_text("hub readme\n")
    _run(["git", "add", "-A"], other)
    _run(["git", "commit", "-m", "hub divergence"], other)
    _run(["git", "push", "origin", "HEAD:master"], other)

    pr = _run(["git", "pull", "--rebase", "--autostash",
               "origin", "master"], local)
    assert "autostash resulted in conflicts" in (pr.stdout + pr.stderr), (
        pr.stdout + pr.stderr)
    uu = set(_porcelain_uu(local))
    assert "experiment_queue.json" in uu
    assert "README.md" in uu

    ok = experiment_runner._recover_ephemeral_pull_conflict(local, "ree-v3")
    assert not ok, (
        "mixed conflict must be refused so non-ephemeral path is not "
        "silently auto-resolved"
    )
    # Neither path should have been touched -- the queue file still has
    # UU markers, and the README still has its conflict.
    uu_after = set(_porcelain_uu(local))
    assert "README.md" in uu_after
    assert "experiment_queue.json" in uu_after


# ---------------------------------------------------------------------------
# C7: REE_assembly-side heartbeat dirty-tree stall recovers via git_pull
# ---------------------------------------------------------------------------
def test_c7_ree_assembly_heartbeat_stall_recovers(tmp_path):
    """REE_assembly serve.py auto-pull historically used `pull --ff-only`
    which refuses on any local modification to a tracked file, including
    runner_heartbeats/<host>.json files the runner subprocess writes
    every minute (2026-05-31 cloud-1 stall signature: "Your local changes
    to the following files would be overwritten by merge. Aborting.").

    serve.py now imports experiment_runner.git_pull, which does
    `pull --rebase --autostash` plus ephemeral-path UU recovery. With a
    divergent-content heartbeat file on both sides (worker mid-write
    vs. hub writer publishing the canonical snapshot), git_pull must
    leave the working tree clean of UU markers and land origin's bytes
    for the heartbeat path (the hub writer is authoritative under
    Phase 3).
    """
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", "remote.git"],
                   cwd=str(root), check=True, capture_output=True)
    local = root / "local"
    other = root / "other"
    for d in (local, other):
        subprocess.run(["git", "clone", "remote.git", d.name],
                       cwd=str(root), check=True, capture_output=True)
    # Seed: a heartbeat file under the canonical REE_assembly path.
    heartbeat_rel = "evidence/experiments/runner_heartbeats/cloud-1.json"
    seed_dir = local / "evidence" / "experiments" / "runner_heartbeats"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (local / heartbeat_rel).write_text(
        json.dumps({"state": "starting", "ts": "2026-05-31T00:00:00Z"}) + "\n")
    _run(["git", "config", "user.email", "t@t"], local)
    _run(["git", "config", "user.name", "T"], local)
    _run(["git", "add", "-A"], local)
    _run(["git", "commit", "-m", "init"], local)
    _run(["git", "push", "origin", "HEAD:master"], local)
    _run(["git", "config", "user.email", "t@t"], other)
    _run(["git", "config", "user.name", "T"], other)
    _run(["git", "fetch", "origin"], other)
    _run(["git", "reset", "--hard", "origin/master"], other)

    # Worker dirties the heartbeat (uncommitted modification of a
    # tracked file -- the cloud-1 stall surface).
    (local / heartbeat_rel).write_text(
        json.dumps({"state": "idle", "ts": "2026-05-31T01:00:00Z"}) + "\n")

    # Hub writer publishes a divergent version to origin via `other`.
    (other / heartbeat_rel).write_text(
        json.dumps({"state": "draining",
                    "ts": "2026-05-31T01:00:30Z"}) + "\n")
    _run(["git", "add", "-A"], other)
    _run(["git", "commit", "-m", "phase3-heartbeats: cloud-1 publish"], other)
    push = _run(["git", "push", "origin", "HEAD:master"], other)
    assert push.returncode == 0, push.stderr

    # Confirm the stall signature: plain ff-only pull refuses to proceed.
    ff = _run(["git", "pull", "--ff-only"], local)
    assert ff.returncode != 0, (
        "ff-only pull should refuse on dirty tracked file"
    )
    assert ("would be overwritten by merge" in ff.stderr or
            "would be overwritten" in ff.stderr), (
        f"expected dirty-tree refusal; got: {ff.stderr!r}"
    )

    # The serve.py path -- via experiment_runner.git_pull -- must heal it.
    experiment_runner.git_pull(local, "REE_assembly")

    # Working tree must be free of UU markers.
    assert _porcelain_uu(local) == [], (
        f"git_pull left UU markers: {_porcelain_uu(local)}"
    )
    # Origin's bytes must have landed (hub writer authoritative).
    landed = json.loads((local / heartbeat_rel).read_text())
    assert landed.get("state") == "draining", (
        f"expected origin's heartbeat content; got {landed}"
    )

    # Subsequent pulls must continue to succeed (no wedge across ticks).
    experiment_runner.git_pull(local, "REE_assembly")
    assert _porcelain_uu(local) == []
