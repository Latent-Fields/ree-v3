"""Contract tests for the active-claim guard on the --auto-sync background
pull (experiment_runner._sync_pull_tick).

Background -- daemon autostash revert incident class (EXQ-232 / 2026-04-29
ARC-026 supersession, recurring 2026-05-08 on substrate_queue.json):
  experiment_runner.run_experiment() spawns a daemon `_background_sync`
  thread that, every 60s while a run is in progress, did:
      git_pull(REPO_ROOT, "ree-v3")
      git_pull(ree_assembly_path, "REE_assembly")   # <-- unguarded
  git_pull uses `git pull --rebase --autostash`. Unlike
  runner_remote_control.push_heartbeat / push_commands -- which early-return
  when a Claude session holds an active TASK_CLAIMS claim covering any
  evidence/ path -- this daemon pull had no such guard. While a
  governance/evidence session held uncommitted edits to REE_assembly
  evidence/claims files, the daemon autostashed them every 60s and stacked
  failing autostash-pop stashes, silently reverting claims.yaml / manifest /
  review_tracker edits. The thread is daemon=True and not _pause_flag-aware,
  so `pause` did not stop it -- only full runner shutdown did.

Fix: the per-tick pull body was extracted to module-level
`_sync_pull_tick(ree_assembly_path)`, which skips ONLY the REE_assembly
pull for the tick when `_rrc._active_claim_on_evidence_dir(...)` is True --
the exact guard push_heartbeat / push_commands already use. The ree-v3
pull (REPO_ROOT) is intentionally unguarded; only REE_assembly carries the
high-contention evidence files.

Contracts:
  C1. No active evidence claim -> both ree-v3 AND REE_assembly are pulled
      (bit-identical to the pre-guard default path).
  C2. Active evidence claim -> ree-v3 is pulled, REE_assembly is SKIPPED.
  C3. ree_assembly_path is None -> only ree-v3 is pulled, no crash.
  C4. runner_remote_control unimportable (_rrc is None) -> both pulled
      (default path preserved even when the guard module is unavailable;
      mirrors push_heartbeat / push_commands call-site, which are also
      gated on _rrc is not None).
  C5. The guard consulted is exactly _rrc._active_claim_on_evidence_dir --
      the same function push_heartbeat / push_commands use -- so the two
      cannot silently diverge.
  C6. _sync_pull_tick never raises even if git_pull itself raises
      (best-effort; matches the try/except the daemon previously inlined).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402
import runner_remote_control  # noqa: E402


@pytest.fixture
def record_pulls(monkeypatch):
    """Replace git_pull with a recorder of the labels it was called with."""
    calls: list[str] = []
    monkeypatch.setattr(
        experiment_runner, "git_pull",
        lambda repo, label: calls.append(label),
    )
    return calls


def _set_claim(monkeypatch, active: bool) -> None:
    monkeypatch.setattr(
        runner_remote_control, "_active_claim_on_evidence_dir",
        lambda _p: active,
    )
    monkeypatch.setattr(experiment_runner, "_rrc", runner_remote_control)


def test_c1_no_active_claim_pulls_both(record_pulls, monkeypatch, tmp_path):
    _set_claim(monkeypatch, active=False)
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")
    assert record_pulls == ["ree-v3", "REE_assembly"], (
        "C1 FAIL: default path must pull both repos bit-identically."
    )


def test_c2_active_claim_skips_assembly(record_pulls, monkeypatch, tmp_path):
    _set_claim(monkeypatch, active=True)
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")
    assert record_pulls == ["ree-v3"], (
        "C2 FAIL: an active evidence claim must skip the REE_assembly pull "
        "(the EXQ-232 autostash-revert guard) while leaving ree-v3 alone."
    )


def test_c3_none_assembly_path_pulls_only_reev3(
    record_pulls, monkeypatch
):
    _set_claim(monkeypatch, active=False)
    experiment_runner._sync_pull_tick(None)
    assert record_pulls == ["ree-v3"], (
        "C3 FAIL: ree_assembly_path=None must pull ree-v3 only, no crash."
    )


def test_c4_rrc_none_preserves_default_path(record_pulls, monkeypatch, tmp_path):
    # Guard module unavailable: cannot consult claims -> must not change the
    # pre-guard behaviour (both pulls run), matching push_heartbeat /
    # push_commands which are themselves gated on `_rrc is not None`.
    monkeypatch.setattr(experiment_runner, "_rrc", None)
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")
    assert record_pulls == ["ree-v3", "REE_assembly"], (
        "C4 FAIL: with _rrc None the default both-pull path must be "
        "bit-identical to pre-guard behaviour."
    )


def test_c5_uses_the_same_guard_as_push_heartbeat(
    record_pulls, monkeypatch, tmp_path
):
    # Sentinel: if _sync_pull_tick consulted any guard OTHER than
    # _rrc._active_claim_on_evidence_dir, patching that exact symbol would
    # not flip the behaviour and this assertion would fail -- catching guard
    # drift between the daemon and push_heartbeat / push_commands.
    seen = {"called": False}

    def _spy(_p):
        seen["called"] = True
        return True

    monkeypatch.setattr(
        runner_remote_control, "_active_claim_on_evidence_dir", _spy
    )
    monkeypatch.setattr(experiment_runner, "_rrc", runner_remote_control)
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")
    assert seen["called"], (
        "C5 FAIL: _sync_pull_tick did not consult "
        "_rrc._active_claim_on_evidence_dir -- guard drift risk."
    )
    assert record_pulls == ["ree-v3"], (
        "C5 FAIL: the consulted guard did not gate the REE_assembly pull."
    )


def test_c6_never_raises_when_git_pull_raises(monkeypatch, tmp_path):
    def _boom(repo, label):
        raise RuntimeError(f"git exploded for {label}")

    monkeypatch.setattr(experiment_runner, "git_pull", _boom)
    _set_claim(monkeypatch, active=False)
    # Must swallow the exception for BOTH pulls -- best-effort, exactly as
    # the daemon's inlined try/except did before the extraction.
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")
