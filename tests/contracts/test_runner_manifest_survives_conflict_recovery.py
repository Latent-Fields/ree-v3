"""Contract tests for the manifest-leak fix in
experiment_runner._git_push_with_retry.

Background -- V3-EXQ-541 leak incident (2026-05-08, ree-cloud-1):
  When the per-experiment results push hits a non-fast-forward and the
  followup pull --rebase fails (e.g. dirty working tree from a concurrent
  heartbeat / signal write), the original recovery path stashed only the
  WORKING TREE, then `git reset --hard origin/<branch>` destroyed the
  manifest-bearing local commit, then re-staged the manifest path -- but
  the file no longer existed on disk, so the `git add` was a silent
  no-op.  The recovery commit captured the stashed signal/heartbeat/status
  files only, and the manifest never reached REE_assembly master.

Fix (Option A from /tmp/cloud_manifest_leak_diagnosis_prompt.md):
  - capture HEAD SHA before reset --hard;
  - after reset, restore each result_files path via
    `git checkout <pre_reset_sha> -- <rel>` so the file is back on disk
    and staged before any `git add` is attempted;
  - resolve unmerged paths from a stash-pop conflict by taking the
    remote version (--ours after reset) and unstaging, so the manifest
    can still commit + push;
  - emit a WARN if the post-recovery selective add stages none of the
    expected files (the silent-no-op is what masked this bug for weeks).

Contracts:
  C1. Single manifest, conflict on overlapping heartbeat -> manifest
      lands on remote master after recovery.
  C2. Two manifests, both committed, conflict same as C1 -> both land
      on remote master.
  C3. result_files=None broad-fallback path -> the legacy
      `git add evidence/experiments/` recovery still produces a remote
      master that includes the manifest (existing fallback semantics
      preserved).
  C4. Host has NO git identity configured (no user.name/user.email, and
      auto-detection disabled) -> git_push_results still lands the
      manifest, via _git_commit's fallback identity.

      Regression for a real bug found via a GitHub Actions ubuntu-latest
      CI run (2026-07-27, workflow run 30245773773): git_push_results'
      initial `git commit` was never checked for success. On a host with
      no git identity and no GECOS full name to auto-derive one from,
      that commit fails outright (exit 128, "fatal: empty ident name ...
      not allowed"), so the manifest never enters a commit at all --
      `pre_reset_sha` (captured later in _git_push_with_retry) then
      points at the PRIOR, manifest-less HEAD, and the conflict-recovery
      restore-from-pre_reset_sha step fails with "pathspec ... did not
      match any file(s)" while the function still logs "pushed results".
      This is the same silent-loss shape the V3-EXQ-541 recovery path
      (C1-C3 above) exists to prevent, just triggered by missing identity
      instead of a destructive reset ordering bug. Every currently-live
      production host was checked directly and has identity configured,
      so this had not fired in production -- but nothing enforces that
      for a future host, hence the fallback in _git_commit and this test.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _make_repos(root: Path) -> tuple[Path, Path]:
    """Create a bare remote + two clones (local, other) with one shared
    baseline commit that both clones can then diverge from."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", "remote.git"],
                   cwd=str(root), check=True, capture_output=True)
    local = root / "local"
    other = root / "other"
    for d in (local, other):
        subprocess.run(["git", "clone", "remote.git", d.name],
                       cwd=str(root), check=True, capture_output=True)
    # Seed a baseline heartbeat + readme on local and push.
    for sub in ("evidence/experiments/_runner_signals",
                "evidence/experiments/runner_status",
                "evidence/experiments/runner_heartbeats"):
        (local / sub).mkdir(parents=True, exist_ok=True)
    (local / "evidence/experiments/runner_heartbeats/ree-cloud-1.json").write_text(
        '{"machine":"ree-cloud-1","tick":0}\n')
    (local / "README.md").write_text("seed\n")
    _run(["git", "add", "-A"], local)
    _run(["git", "-c", "user.email=t@t", "-c", "user.name=T",
          "commit", "-m", "init"], local)
    _run(["git", "push", "origin", "HEAD:master"], local)
    _run(["git", "fetch", "origin"], other)
    _run(["git", "reset", "--hard", "origin/master"], other)
    return local, other


def _other_pushes_heartbeat_update(other: Path) -> None:
    """`other` modifies the same heartbeat path local will dirty, then
    pushes. This produces the conflict signature seen in production."""
    f = other / "evidence/experiments/runner_heartbeats/ree-cloud-1.json"
    f.write_text('{"machine":"ree-cloud-1","tick":42,"from":"other"}\n')
    _run(["git", "add", "-A"], other)
    _run(["git", "-c", "user.email=t@t", "-c", "user.name=T",
          "commit", "-m", "other heartbeat"], other)
    r = _run(["git", "push", "origin", "HEAD:master"], other)
    assert r.returncode == 0, f"other push failed: {r.stderr}"


def _dirty_local_concurrents(local: Path) -> None:
    """Mimic concurrent runner-side writes (heartbeat / signal / status)
    that are dirty when git_push_results runs."""
    (local / "evidence/experiments/runner_heartbeats/ree-cloud-1.json").write_text(
        '{"machine":"ree-cloud-1","tick":1234,"local":"dirty"}\n')
    (local / "evidence/experiments/_runner_signals/V3-EXQ-541.json").write_text(
        json.dumps({"queue_id": "V3-EXQ-541", "outcome": "FAIL"}))
    (local / "evidence/experiments/runner_status/ree-cloud-1.json").write_text(
        '{"completed":["V3-EXQ-541"]}\n')


def _write_manifest(local: Path, name: str) -> Path:
    p = local / f"evidence/experiments/{name}"
    p.write_text(json.dumps({
        "run_id": name.removesuffix(".json"),
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "evidence_direction": "supports",
        "claim_ids": ["MECH-204"],
    }, indent=2))
    return p


def _files_on_remote(local: Path) -> set[str]:
    r = _run(["git", "ls-tree", "-r", "master", "--name-only"], local)
    assert r.returncode == 0, f"ls-tree failed: {r.stderr}"
    return set(r.stdout.splitlines())


def _no_active_evidence_claim(monkeypatch):
    """Bypass the TASK_CLAIMS guard so the recovery path actually runs.
    Without this the guard could short-circuit if any session in the
    real TASK_CLAIMS.json holds a claim covering evidence/experiments/."""
    monkeypatch.setattr(experiment_runner, "_check_active_claim_on_file",
                        lambda _path: False)


def _mask_git_identity(monkeypatch, tmp_path):
    """Make every git subprocess this test spawns behave like a host with
    NO configured identity and no GECOS full name to auto-derive one from
    -- deterministically, regardless of what identity is actually
    configured on the machine running the test (so this reproduces the
    GitHub Actions ubuntu-latest failure on any host, including a
    developer's Mac that already has user.name/user.email set globally).

    GIT_CONFIG_GLOBAL / GIT_CONFIG_SYSTEM redirect config lookup away from
    the real global/system config entirely; user.useConfigOnly=true (set
    via the GIT_CONFIG_COUNT/KEY/VALUE env-config mechanism, so it applies
    process-wide without threading a `-c` flag through every call site)
    disables git's GECOS/hostname auto-guess fallback. `-c user.email=...
    -c user.name=...` on this test's OWN setup commits (_make_repos etc.)
    still wins over both, since an explicit `-c` is a real config source,
    not auto-detection -- so only the SUT's un-decorated `git commit`
    calls are affected.
    """
    empty_config = tmp_path / "empty_gitconfig"
    empty_config.write_text("")
    for var in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL",
                "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty_config))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.useConfigOnly")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")


# ---------------------------------------------------------------------------
# C1: single manifest survives conflict recovery
# ---------------------------------------------------------------------------
def test_c1_single_manifest_survives_conflict(tmp_path, monkeypatch):
    _no_active_evidence_claim(monkeypatch)
    local, other = _make_repos(tmp_path)
    _other_pushes_heartbeat_update(other)
    manifest = _write_manifest(
        local, "v3_exq_541_fake_20260509T1149Z_v3.json")
    _dirty_local_concurrents(local)

    experiment_runner.git_push_results(local, [str(manifest)])

    rel = str(manifest.relative_to(local))
    files = _files_on_remote(local)
    assert rel in files, (
        f"C1 FAIL: manifest {rel} missing from remote master.\n"
        f"files on master: {sorted(files)}"
    )


# ---------------------------------------------------------------------------
# C2: multiple manifests all survive conflict recovery
# ---------------------------------------------------------------------------
def test_c2_multiple_manifests_survive_conflict(tmp_path, monkeypatch):
    _no_active_evidence_claim(monkeypatch)
    local, other = _make_repos(tmp_path)
    _other_pushes_heartbeat_update(other)
    m1 = _write_manifest(local, "v3_exq_aaa_fake_v3.json")
    m2 = _write_manifest(local, "v3_exq_bbb_fake_v3.json")
    _dirty_local_concurrents(local)

    experiment_runner.git_push_results(local, [str(m1), str(m2)])

    rel1 = str(m1.relative_to(local))
    rel2 = str(m2.relative_to(local))
    files = _files_on_remote(local)
    assert rel1 in files and rel2 in files, (
        f"C2 FAIL: multi-manifest leak.\n"
        f"  m1={rel1} present={rel1 in files}\n"
        f"  m2={rel2} present={rel2 in files}\n"
        f"  files on master: {sorted(files)}"
    )


# ---------------------------------------------------------------------------
# C3: result_files=None broad-fallback recovery still ships the manifest
# ---------------------------------------------------------------------------
def test_c3_result_files_none_fallback_preserves_manifest(tmp_path, monkeypatch):
    _no_active_evidence_claim(monkeypatch)
    local, other = _make_repos(tmp_path)
    _other_pushes_heartbeat_update(other)
    manifest = _write_manifest(local, "v3_exq_ccc_fallback_v3.json")
    _dirty_local_concurrents(local)

    # result_files=None -> git_push_results uses the broad
    # `git add evidence/experiments/` path, which sweeps in everything
    # dirty including the manifest.  After recovery it must still arrive
    # on remote master.
    experiment_runner.git_push_results(local, None)

    rel = str(manifest.relative_to(local))
    files = _files_on_remote(local)
    assert rel in files, (
        f"C3 FAIL: fallback path lost the manifest {rel}.\n"
        f"files on master: {sorted(files)}"
    )


# ---------------------------------------------------------------------------
# C4: no git identity configured on the host -> manifest still lands
# ---------------------------------------------------------------------------
def test_c4_no_git_identity_still_lands_manifest(tmp_path, monkeypatch):
    _no_active_evidence_claim(monkeypatch)
    _mask_git_identity(monkeypatch, tmp_path)
    local, other = _make_repos(tmp_path)
    _other_pushes_heartbeat_update(other)
    manifest = _write_manifest(
        local, "v3_exq_541_noident_20260727T2234Z_v3.json")
    _dirty_local_concurrents(local)

    experiment_runner.git_push_results(local, [str(manifest)])

    rel = str(manifest.relative_to(local))
    files = _files_on_remote(local)
    assert rel in files, (
        f"C4 FAIL: manifest {rel} missing from remote master when the host "
        f"has no git identity configured.\n"
        f"files on master: {sorted(files)}"
    )
