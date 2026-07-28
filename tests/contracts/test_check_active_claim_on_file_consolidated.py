"""Contract tests for experiment_runner._check_active_claim_on_file.

This was the LAST standalone TASK_CLAIMS matcher in the runner. Until
2026-07-28 it read /Users/dgolden/REE_Working/TASK_CLAIMS.json with its own
hand-rolled rule (`relative_path in res or res.endswith(relative_path)`) and
its own claims-root resolution -- a THIRD matching rule beside
runner_remote_control._active_claim_on_evidence_dir and
_active_claim_on_ree_v3_code, both of which had already been consolidated onto
_active_claim_on_paths in ree-v3 4a22888bec.

That is drift debt rather than a live bug: the rules happened to agree. It
matters because test_background_sync_claim_guard.py C5 exists specifically to
stop the runner's claim guards diverging from one another, and this function
sat outside that protection -- so a future broadening of
_active_claim_on_paths would have been applied to two guards and silently
missed the third.

The sole call site is git_push_with_retry's rebase-conflict recovery: True
skips the push rather than proceeding into `git reset --hard`, which would
discard a live session's uncommitted REE_assembly edits. So the guard is
PROTECTIVE (True == do the safe thing) while every failure direction is
fail-OPEN (False == proceed), and both halves are pinned below.

Contracts:
  D1. Delegation (the C5 analogue): the guard consulted is exactly
      _rrc._active_claim_on_paths, called with the umbrella claims root and
      the caller's prefix as a single-element tuple, unanchored and unbounded.
      A re-hand-rolled body fails this test.
  D2. It still FIRES for an active claim naming an evidence/experiments/
      resource -- end to end, against a real file on disk.
  D3. Reach is unchanged: substring semantics, so a resource carrying the
      prefix anywhere matches and an unrelated resource does not. Pinned
      differentially against the exact pre-consolidation rule.
  D4. status != "active" does not fire.
  D5. Missing claims file -> False (fail-open).
  D6. Unparseable claims file -> False (fail-open, no raise).
  D7. _rrc is None -> False, matching every other _rrc call site in the
      runner.
  D8. A non-string resources entry is skipped rather than aborting the scan,
      so it cannot mask a real claim later in the same file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402
import runner_remote_control  # noqa: E402


PREFIX = "evidence/experiments/"


def _claims_root(monkeypatch, tmp_path: Path) -> Path:
    """Point the guard at an isolated umbrella dir and return it.

    The guard resolves TASK_CLAIMS.json as REPO_ROOT.parent, so REPO_ROOT is
    redirected to a fake ree-v3 checkout under tmp_path. Without this the test
    would read the REAL TASK_CLAIMS.json and depend on whatever other sessions
    happen to be holding open.
    """
    root = tmp_path / "REE_Working"
    (root / "ree-v3").mkdir(parents=True)
    monkeypatch.setattr(experiment_runner, "REPO_ROOT", root / "ree-v3")
    monkeypatch.setattr(experiment_runner, "_rrc", runner_remote_control)
    return root


def _write_claims(root: Path, claims: list) -> None:
    (root / "TASK_CLAIMS.json").write_text(
        json.dumps({"claims": claims}), encoding="utf-8"
    )


def _active(resources: list) -> dict:
    return {
        "session_id": "test-session",
        "status": "active",
        "resources": resources,
    }


def test_d1_delegates_to_the_shared_helper(monkeypatch, tmp_path):
    """The consolidated guard must call _active_claim_on_paths, not re-roll it."""
    root = _claims_root(monkeypatch, tmp_path)
    _write_claims(root, [])

    seen: list[tuple] = []

    def _spy(claims_root, prefixes, **kwargs):
        seen.append((claims_root, prefixes, kwargs))
        return True

    monkeypatch.setattr(runner_remote_control, "_active_claim_on_paths", _spy)

    assert experiment_runner._check_active_claim_on_file(PREFIX) is True, (
        "D1 FAIL: the guard did not return the shared helper's verdict."
    )
    assert len(seen) == 1, (
        "D1 FAIL: _check_active_claim_on_file did not consult "
        "_rrc._active_claim_on_paths exactly once -- it appears to have its "
        "own TASK_CLAIMS matcher again. That is the divergence this "
        "consolidation removed."
    )
    claims_root, prefixes, kwargs = seen[0]
    assert Path(claims_root) == root, (
        f"D1 FAIL: claims root was {claims_root!r}, expected the umbrella dir "
        f"{root!r} (REPO_ROOT.parent)."
    )
    assert prefixes == (PREFIX,), (
        f"D1 FAIL: prefixes were {prefixes!r}, expected a single-element tuple "
        f"of the caller's path. Reusing _EVIDENCE_CLAIM_PREFIXES here would "
        f"WIDEN this call site's reach to all of evidence/ and docs/claims/."
    )
    assert not kwargs.get("anchored", False), (
        "D1 FAIL: anchored mode changes the matching rule; the pre-"
        "consolidation behaviour is unanchored (substring)."
    )
    assert kwargs.get("max_age_hours") is None, (
        "D1 FAIL: an age bound would stop honouring live session claims that "
        "the pre-consolidation guard honoured indefinitely."
    )


def test_d2_fires_for_an_active_evidence_experiments_claim(monkeypatch, tmp_path):
    root = _claims_root(monkeypatch, tmp_path)
    _write_claims(root, [_active(["REE_assembly/evidence/experiments/foo.json"])])

    assert experiment_runner._check_active_claim_on_file(PREFIX) is True, (
        "D2 FAIL: the guard no longer fires for evidence/experiments/ -- a "
        "live session's uncommitted edits would be reset --hard away."
    )


@pytest.mark.parametrize(
    "resource",
    [
        "evidence/experiments/",
        "REE_assembly/evidence/experiments/runs/abc/manifest.json",
        "./evidence/experiments/foo.json",
        "evidence/",
        "evidence/experiments",
        "docs/claims/claims.yaml",
        "ree-v3/experiments/foo.py",
        "",
    ],
)
def test_d3_reach_matches_the_preconsolidation_rule(monkeypatch, tmp_path, resource):
    """Differential pin against the exact rule the old body used.

    `res.endswith(prefix)` was dead code -- endswith implies substring
    containment -- so the old rule reduces to `prefix in res`, which is
    precisely unanchored mode. Any future edit that changes the reach breaks
    this test rather than silently re-gating (or un-gating) the runner.
    """
    root = _claims_root(monkeypatch, tmp_path)
    _write_claims(root, [_active([resource])])

    expected = PREFIX in resource or resource.endswith(PREFIX)
    actual = experiment_runner._check_active_claim_on_file(PREFIX)
    assert actual is expected, (
        f"D3 FAIL: resource {resource!r} -- pre-consolidation rule said "
        f"{expected}, consolidated guard said {actual}. Matching reach must "
        f"stay identical."
    )


def test_d4_inactive_claim_does_not_fire(monkeypatch, tmp_path):
    root = _claims_root(monkeypatch, tmp_path)
    entry = _active(["REE_assembly/evidence/experiments/foo.json"])
    entry["status"] = "done"
    _write_claims(root, [entry])

    assert experiment_runner._check_active_claim_on_file(PREFIX) is False, (
        "D4 FAIL: a closed claim must not hold the push."
    )


def test_d5_missing_claims_file_is_false(monkeypatch, tmp_path):
    _claims_root(monkeypatch, tmp_path)  # no TASK_CLAIMS.json written

    assert experiment_runner._check_active_claim_on_file(PREFIX) is False, (
        "D5 FAIL: a missing claims file must fail OPEN. The cloud workers "
        "have no TASK_CLAIMS.json at all, so every worker depends on this."
    )


def test_d6_unparseable_claims_file_is_false(monkeypatch, tmp_path):
    root = _claims_root(monkeypatch, tmp_path)
    (root / "TASK_CLAIMS.json").write_text("{not json", encoding="utf-8")

    assert experiment_runner._check_active_claim_on_file(PREFIX) is False, (
        "D6 FAIL: an unparseable claims file must fail OPEN, not raise. This "
        "runs on the fleet's critical path."
    )


def test_d7_rrc_none_is_false(monkeypatch, tmp_path):
    root = _claims_root(monkeypatch, tmp_path)
    _write_claims(root, [_active(["REE_assembly/evidence/experiments/foo.json"])])
    monkeypatch.setattr(experiment_runner, "_rrc", None)

    assert experiment_runner._check_active_claim_on_file(PREFIX) is False, (
        "D7 FAIL: with runner_remote_control unimportable the guard must "
        "return False, matching every other _rrc call site in the runner."
    )


def test_d8_non_string_resource_does_not_mask_a_later_claim(monkeypatch, tmp_path):
    """A malformed entry must not abort the scan.

    The pre-consolidation body raised TypeError on `in` and the outer except
    swallowed it into a blanket False, so one malformed resource could hide a
    real claim further down the file. The shared helper skips it instead.
    """
    root = _claims_root(monkeypatch, tmp_path)
    _write_claims(root, [
        _active([None, 42]),
        _active(["REE_assembly/evidence/experiments/foo.json"]),
    ])

    assert experiment_runner._check_active_claim_on_file(PREFIX) is True, (
        "D8 FAIL: a non-string resources entry masked a real claim later in "
        "the file -- the push would proceed into reset --hard over live work."
    )
