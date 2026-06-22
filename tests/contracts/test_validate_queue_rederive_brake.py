"""
Contract tests for the MOVE-3 re-derive-brake warn-only backstop in
validate_queue.py (landed 2026-06-22, ree-v3 main 521106a).

validate_queue.py had NO pytest coverage; the brake logic
(_scan_substrate_ceiling_autopsies, _upstream_substrate_from_target,
_substrate_is_built, and the per-item brake block in validate()) was verified
only by an in-session throwaway script (WORKSPACE_STATE.md 2026-06-22T01:07Z).
This suite locks it in.

The brake re-applies the /queue-experiment Step 2.5b + /failure-autopsy Step 7
counting logic: for a queued item tagging a claim with
>= RE_DERIVE_BRAKE_THRESHOLD (default 2) substrate_ceiling/non_contributory
autopsies (one hit per failure_autopsy_*.json per claim, matched on
'substrate_ceiling' in recommended_epistemic_category OR 'non_contributory' in
recommended_evidence_direction) whose named upstream substrate
(recommended_substrate_queue_entry.target_sd_id / sd_id_suggested /
re_derive_brake.upstream_substrate) is NOT shown IMPLEMENTED/VALIDATED on a
single ree-v3/CLAUDE.md line, it appends a non-blocking _LAST_WARNINGS advisory.
It must NOT enter the returned errors list and must not change the exit code.

Branches pinned (mirroring the in-session test recorded in WORKSPACE_STATE.md):
  (1) WARN when count>=2 and the upstream substrate is unbuilt.
  (2) SUPPRESSED when the upstream substrate appears IMPLEMENTED/VALIDATED on
      the same CLAUDE.md line.
  (3) below-threshold (count==1) -> no warn.
  (4) note-clearance ('re-derive brake' substring in item.note) -> skip.
  (5) claimless (claim_ids: []) -> skip.
  (6) _substrate_is_built token-boundary: 'SD-BBB0 -- IMPLEMENTED' must NOT
      satisfy 'SD-BBB', and a nearby unrelated 'IMPLEMENTED' line must NOT
      release the target.
  (7) the brake warnings never appear in validate()'s returned errors list
      (exit code unaffected).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import validate_queue  # noqa: E402


BRAKE_MARK = "re-derive brake"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _write_autopsy(
    planning_dir: Path,
    slug: str,
    date: str,
    claims: list[str],
    *,
    category: str = "substrate_ceiling",
    direction: str = "",
    upstream: str = "SD-AAA",
    upstream_key: str = "target_sd_id",
) -> None:
    """Write a failure_autopsy_<slug>_<date>.json with one counted target.

    The scanner counts a target when 'substrate_ceiling' is in
    recommended_epistemic_category OR 'non_contributory' is in
    recommended_evidence_direction.
    """
    target: dict = {
        "recommended_epistemic_category": category,
        "recommended_evidence_direction": direction,
        "claim_ids": claims,
    }
    if upstream_key in ("target_sd_id", "sd_id_suggested"):
        target["recommended_substrate_queue_entry"] = {upstream_key: upstream}
    elif upstream_key == "re_derive_brake":
        target["re_derive_brake"] = {"upstream_substrate": upstream}
    fname = f"failure_autopsy_{slug}_{date}.json"
    (planning_dir / fname).write_text(json.dumps({"targets": [target]}), encoding="utf-8")


def _make_item(claim_ids=None, note=None, queue_id="V3-EXQ-901") -> dict:
    item = {
        "queue_id": queue_id,
        "script": "experiments/__nonexistent_brake_test__.py",
        "priority": 1,
        "machine_affinity": "any",
        "status": "pending",
        "estimated_minutes": 10,
    }
    if claim_ids is not None:
        item["claim_ids"] = claim_ids
    if note is not None:
        item["note"] = note
    return item


def _run(tmp_path, monkeypatch, items, claude_md_text="") -> tuple[list[str], list[str]]:
    """Write a queue + CLAUDE.md, point the brake at the tmp planning dir, and
    run validate(). Returns (errors, brake_warnings)."""
    planning_dir = tmp_path / "planning"
    planning_dir.mkdir(exist_ok=True)
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(claude_md_text, encoding="utf-8")

    monkeypatch.setattr(
        validate_queue, "_REE_ASSEMBLY_PLANNING_DIR_CANDIDATES", [planning_dir]
    )
    monkeypatch.setattr(
        validate_queue, "_REE_V3_CLAUDE_MD_CANDIDATES", [claude_md]
    )

    queue = {"schema_version": "v1", "calibration": {}, "items": items}
    queue_path = tmp_path / "experiment_queue.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    errors = validate_queue.validate(queue_path)
    brake_warnings = [w for w in validate_queue._LAST_WARNINGS if BRAKE_MARK in w]
    return errors, brake_warnings


def _seed_two_ceiling_autopsies(tmp_path, claim, upstream="SD-AAA", **kw):
    planning_dir = tmp_path / "planning"
    planning_dir.mkdir(exist_ok=True)
    _write_autopsy(planning_dir, "first", "2026-06-01", [claim], upstream=upstream, **kw)
    _write_autopsy(planning_dir, "second", "2026-06-02", [claim], upstream=upstream, **kw)


# ------------------------------------------------------------------
# (1) WARN when count>=2 and upstream substrate unbuilt
# ------------------------------------------------------------------
def test_warn_when_count_ge_threshold_and_upstream_unbuilt(tmp_path, monkeypatch):
    _seed_two_ceiling_autopsies(tmp_path, "MECH-999", upstream="SD-AAA")
    errors, brake_warnings = _run(
        tmp_path, monkeypatch,
        items=[_make_item(claim_ids=["MECH-999"])],
        claude_md_text="# ree-v3\nNo substrate built here.\n",
    )
    assert len(brake_warnings) == 1
    w = brake_warnings[0]
    assert "MECH-999" in w
    assert "SD-AAA" in w
    # threshold default is 2 -- assert the test matches the default it is locking in.
    assert validate_queue.RE_DERIVE_BRAKE_THRESHOLD == 2


def test_warn_counts_via_non_contributory_direction(tmp_path, monkeypatch):
    # The OTHER counting key: non_contributory in recommended_evidence_direction.
    _seed_two_ceiling_autopsies(
        tmp_path, "MECH-998", upstream="SD-AAA",
        category="standard", direction="non_contributory",
    )
    _, brake_warnings = _run(
        tmp_path, monkeypatch,
        items=[_make_item(claim_ids=["MECH-998"])],
        claude_md_text="",
    )
    assert len(brake_warnings) == 1
    assert "MECH-998" in brake_warnings[0]


def test_warn_resolves_upstream_via_re_derive_brake_key(tmp_path, monkeypatch):
    # Upstream named via re_derive_brake.upstream_substrate (the third lookup).
    _seed_two_ceiling_autopsies(
        tmp_path, "MECH-997", upstream="SD-ZZZ", upstream_key="re_derive_brake",
    )
    _, brake_warnings = _run(
        tmp_path, monkeypatch,
        items=[_make_item(claim_ids=["MECH-997"])],
        claude_md_text="",
    )
    assert len(brake_warnings) == 1
    assert "SD-ZZZ" in brake_warnings[0]


# ------------------------------------------------------------------
# (2) SUPPRESSED when upstream appears IMPLEMENTED/VALIDATED on same line
# ------------------------------------------------------------------
def test_suppressed_when_upstream_implemented_same_line(tmp_path, monkeypatch):
    _seed_two_ceiling_autopsies(tmp_path, "MECH-999", upstream="SD-AAA")
    _, brake_warnings = _run(
        tmp_path, monkeypatch,
        items=[_make_item(claim_ids=["MECH-999"])],
        claude_md_text="- SD-AAA the upstream substrate -- IMPLEMENTED 2026-01-01.\n",
    )
    assert brake_warnings == []


def test_suppressed_when_upstream_validated_same_line(tmp_path, monkeypatch):
    _seed_two_ceiling_autopsies(tmp_path, "MECH-999", upstream="SD-AAA")
    _, brake_warnings = _run(
        tmp_path, monkeypatch,
        items=[_make_item(claim_ids=["MECH-999"])],
        claude_md_text="- SD-AAA substrate -- VALIDATED 2026-02-02 by V3-EXQ-123.\n",
    )
    assert brake_warnings == []


# ------------------------------------------------------------------
# (3) below-threshold (count==1) -> no warn
# ------------------------------------------------------------------
def test_below_threshold_count_one_no_warn(tmp_path, monkeypatch):
    planning_dir = tmp_path / "planning"
    planning_dir.mkdir(exist_ok=True)
    # only ONE counted autopsy for the claim
    _write_autopsy(planning_dir, "only", "2026-06-01", ["MECH-999"], upstream="SD-AAA")
    _, brake_warnings = _run(
        tmp_path, monkeypatch,
        items=[_make_item(claim_ids=["MECH-999"])],
        claude_md_text="",
    )
    assert brake_warnings == []


# ------------------------------------------------------------------
# (4) note-clearance ('re-derive brake' substring in item.note) -> skip
# ------------------------------------------------------------------
def test_note_clearance_skips_brake(tmp_path, monkeypatch):
    _seed_two_ceiling_autopsies(tmp_path, "MECH-999", upstream="SD-AAA")
    _, brake_warnings = _run(
        tmp_path, monkeypatch,
        items=[_make_item(
            claim_ids=["MECH-999"],
            note="Cleared: re-derive brake released, redesign of a different mechanism.",
        )],
        claude_md_text="",
    )
    assert brake_warnings == []


# ------------------------------------------------------------------
# (5) claimless (claim_ids: []) -> skip
# ------------------------------------------------------------------
def test_claimless_item_skips_brake(tmp_path, monkeypatch):
    # Autopsies exist for MECH-999, but the item declares claim_ids: [] (an
    # intentional claimless diagnostic). No brake warning may fire.
    _seed_two_ceiling_autopsies(tmp_path, "MECH-999", upstream="SD-AAA")
    _, brake_warnings = _run(
        tmp_path, monkeypatch,
        items=[_make_item(claim_ids=[])],
        claude_md_text="",
    )
    assert brake_warnings == []


def test_claimless_item_skipped_even_when_a_sibling_is_tagged(tmp_path, monkeypatch):
    # Force the autopsy scan to run (a sibling item carries a claim tag with no
    # autopsies, so _any_claim_tagged is True) -- the claimless item must still
    # get no brake warning.
    _seed_two_ceiling_autopsies(tmp_path, "MECH-999", upstream="SD-AAA")
    _, brake_warnings = _run(
        tmp_path, monkeypatch,
        items=[
            _make_item(claim_ids=["MECH-NOAUTOPSY"], queue_id="V3-EXQ-902"),
            _make_item(claim_ids=[], queue_id="V3-EXQ-903"),
        ],
        claude_md_text="",
    )
    assert brake_warnings == []


# ------------------------------------------------------------------
# (6) _substrate_is_built token-boundary
# ------------------------------------------------------------------
def test_substrate_is_built_token_boundary():
    # 'SD-BBB0 -- IMPLEMENTED' must NOT satisfy 'SD-BBB' (suffix-extended id).
    assert validate_queue._substrate_is_built(
        "SD-BBB", "- SD-BBB0 some other substrate -- IMPLEMENTED 2026-01-01.\n"
    ) is False
    # A nearby unrelated IMPLEMENTED line must NOT release SD-BBB (SD-BBB is
    # mentioned, but not on the same line as IMPLEMENTED/VALIDATED).
    nearby = (
        "Discussion of SD-BBB and its blockers below.\n"
        "- SD-CCC unrelated -- IMPLEMENTED 2026-01-01.\n"
    )
    assert validate_queue._substrate_is_built("SD-BBB", nearby) is False
    # Positive control: SD-BBB on the same line as IMPLEMENTED -> True.
    assert validate_queue._substrate_is_built(
        "SD-BBB", "- SD-BBB the substrate -- IMPLEMENTED 2026-01-01.\n"
    ) is True
    # Empty / missing inputs fail-soft to False.
    assert validate_queue._substrate_is_built("", "anything IMPLEMENTED") is False
    assert validate_queue._substrate_is_built("SD-BBB", "") is False


def test_substrate_is_built_token_boundary_end_to_end(tmp_path, monkeypatch):
    # The token-boundary discrimination must hold through the full validate()
    # path: a CLAUDE.md whose only IMPLEMENTED line names SD-AAA0 (not SD-AAA)
    # must NOT release the brake for an SD-AAA upstream.
    _seed_two_ceiling_autopsies(tmp_path, "MECH-999", upstream="SD-AAA")
    _, brake_warnings = _run(
        tmp_path, monkeypatch,
        items=[_make_item(claim_ids=["MECH-999"])],
        claude_md_text="- SD-AAA0 a different substrate -- IMPLEMENTED 2026-01-01.\n",
    )
    assert len(brake_warnings) == 1  # NOT released by SD-AAA0


# ------------------------------------------------------------------
# (7) brake warnings never appear in the returned errors list (exit unaffected)
# ------------------------------------------------------------------
def test_brake_warning_is_warn_only_not_an_error(tmp_path, monkeypatch):
    _seed_two_ceiling_autopsies(tmp_path, "MECH-999", upstream="SD-AAA")
    errors, brake_warnings = _run(
        tmp_path, monkeypatch,
        items=[_make_item(claim_ids=["MECH-999"])],
        claude_md_text="",
    )
    # The brake fired ...
    assert len(brake_warnings) == 1
    # ... but it stayed out of the errors list entirely.
    assert all(BRAKE_MARK not in e for e in errors)


def test_brake_does_not_change_exit_code_when_queue_otherwise_valid(tmp_path, monkeypatch):
    # Point the item's script at a real tracked script so the only thing that
    # could fail validation is the brake -- and confirm it does not. A valid
    # queue must return [] (exit 0) even when the brake warning fires.
    repo_root = Path(validate_queue.__file__).resolve().parent
    tracked = sorted((repo_root / "experiments").glob("v3_exq_*.py"))
    if not tracked:
        pytest.skip("no tracked experiment script available to point the item at")
    script_rel = tracked[0].relative_to(repo_root).as_posix()

    planning_dir = tmp_path / "planning"
    planning_dir.mkdir(exist_ok=True)
    _write_autopsy(planning_dir, "first", "2026-06-01", ["MECH-999"], upstream="SD-AAA")
    _write_autopsy(planning_dir, "second", "2026-06-02", ["MECH-999"], upstream="SD-AAA")
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        validate_queue, "_REE_ASSEMBLY_PLANNING_DIR_CANDIDATES", [planning_dir]
    )
    monkeypatch.setattr(validate_queue, "_REE_V3_CLAUDE_MD_CANDIDATES", [claude_md])

    # Build the queue file inside the real repo dir so the script-on-disk +
    # git-tracked checks resolve against the real (tracked) script.
    item = {
        "queue_id": "V3-EXQ-904",
        "script": script_rel,
        "priority": 1,
        "machine_affinity": "any",
        "status": "pending",
        "estimated_minutes": 10,
        "claim_ids": ["MECH-999"],
        "force_rerun": True,  # bypass any silent-re-queue completion record
    }
    queue = {"schema_version": "v1", "calibration": {}, "items": [item]}
    queue_path = repo_root / "experiment_queue.__brake_test__.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    try:
        errors = validate_queue.validate(queue_path)
    finally:
        queue_path.unlink(missing_ok=True)

    brake_warnings = [w for w in validate_queue._LAST_WARNINGS if BRAKE_MARK in w]
    assert len(brake_warnings) == 1  # brake fired
    assert errors == []  # but the queue is still valid -> exit 0
