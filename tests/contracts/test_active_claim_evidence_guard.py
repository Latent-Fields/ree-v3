"""Contract tests for the active-claim guard in runner_remote_control.

The guard (`_active_claim_on_evidence_dir`) is consulted by `push_heartbeat`
and `push_commands` once per minute. When True, both helpers skip the
`git pull --rebase --autostash` cycle that would otherwise risk silently
reverting uncommitted edits made by a concurrent Claude session.

History:
- 2026-05-01: guard added, scoped to 'evidence/experiments/'. Motivated by
  the EXQ-232 ARC-026 supersession revert incident (2026-04-29).
- 2026-05-08: scope broadened to the 'evidence/' prefix after the same
  autostash-revert signature reappeared on an evidence/planning/
  substrate_queue.json edit. The autostash mechanism is not specific to
  experiments/, so the guard should not be either.

Contracts:
  C1. No TASK_CLAIMS.json file -> guard returns False (best-effort default).
  C2. Empty / no active claims -> False.
  C3. Active claim with a resource path under evidence/experiments/ -> True.
  C4. Active claim with a resource path under evidence/planning/ -> True
      (this is the 2026-05-08 broadening; was False under the old guard).
  C5. Active claim with a resource path under evidence/literature/ -> True
      (forward-compat: any evidence/ subdir).
  C6. Done / completed claims do not fire the guard, even when the resource
      list contains an evidence/ path.
  C7. Resource paths outside REE_assembly/evidence/ (e.g. claims.yaml,
      WORKSPACE_STATE.md, ree-v3 source) do not fire the guard.
  C8. Malformed JSON / missing keys -> False (no exception leaks).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Build a tmp REE_Working/REE_assembly layout the helper expects.

    The helper resolves TASK_CLAIMS.json as ree_assembly_path.parent /
    "TASK_CLAIMS.json", so we pass the assembly subdir as the arg.
    """
    assembly = tmp_path / "REE_assembly"
    assembly.mkdir()
    return assembly


def _write_claims(repo: Path, claims: list[dict]) -> None:
    payload = {
        "schema_version": "v1",
        "stale_after_hours": 6,
        "claims": claims,
    }
    (repo.parent / "TASK_CLAIMS.json").write_text(json.dumps(payload))


def test_c1_missing_claims_file_returns_false(fake_repo: Path) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    assert _active_claim_on_evidence_dir(fake_repo) is False


def test_c2_no_active_claims_returns_false(fake_repo: Path) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(fake_repo, [])
    assert _active_claim_on_evidence_dir(fake_repo) is False


def test_c3_active_claim_on_evidence_experiments_fires(fake_repo: Path) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(
        fake_repo,
        [
            {
                "session_id": "s",
                "status": "active",
                "resources": [
                    "REE_assembly/evidence/experiments/some_run/manifest.json"
                ],
            }
        ],
    )
    assert _active_claim_on_evidence_dir(fake_repo) is True


def test_c4_active_claim_on_evidence_planning_fires(fake_repo: Path) -> None:
    """2026-05-08 broadening: planning/ must be covered.

    Real-world incident: an evidence/planning/substrate_queue.json edit
    made under an active claim was silently reverted by the autostash
    cycle because the guard was scoped only to 'evidence/experiments/'.
    """
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(
        fake_repo,
        [
            {
                "session_id": "s",
                "status": "active",
                "resources": [
                    "REE_assembly/evidence/planning/substrate_queue.json"
                ],
            }
        ],
    )
    assert _active_claim_on_evidence_dir(fake_repo) is True


def test_c5_active_claim_on_evidence_literature_fires(fake_repo: Path) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(
        fake_repo,
        [
            {
                "session_id": "s",
                "status": "active",
                "resources": [
                    "REE_assembly/evidence/literature/targeted_review_x/synthesis.md"
                ],
            }
        ],
    )
    assert _active_claim_on_evidence_dir(fake_repo) is True


def test_c6_done_claim_does_not_fire(fake_repo: Path) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(
        fake_repo,
        [
            {
                "session_id": "s",
                "status": "done",
                "resources": [
                    "REE_assembly/evidence/planning/substrate_queue.json"
                ],
            }
        ],
    )
    assert _active_claim_on_evidence_dir(fake_repo) is False


def test_c7_non_evidence_resources_do_not_fire(fake_repo: Path) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(
        fake_repo,
        [
            {
                "session_id": "s",
                "status": "active",
                "resources": [
                    "REE_assembly/docs/claims/claims.yaml",
                    "WORKSPACE_STATE.md",
                    "ree-v3/experiments/foo.py",
                ],
            }
        ],
    )
    assert _active_claim_on_evidence_dir(fake_repo) is False


def test_c8_malformed_json_returns_false(fake_repo: Path) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    (fake_repo.parent / "TASK_CLAIMS.json").write_text("{not valid json")
    assert _active_claim_on_evidence_dir(fake_repo) is False
