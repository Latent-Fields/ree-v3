"""
Contract tests for _substrate_is_built() against the THIN substrate feature index
(ree-v3/CLAUDE.md "## Substrate feature index", since 89907e3e1c 2026-09-07).

Background (chip-20260914-substrate-is-built-grouped-entry-miss). The re-derive
brake's release condition, _substrate_is_built(), originally required the id and
an IMPLEMENTED/VALIDATED token on ONE physical CLAUDE.md line. The 2026-09-07
reflow moved every per-feature record out of CLAUDE.md into docs/substrate/ and
left a compact index whose lines carry the id but -- for grouped multi-record
entries, and for single entries whose title lacks the token -- not the status.
Measured 2026-09-15: 83 of 156 index ids flipped from built to unbuilt across
the reflow (SD-011 among them), so the brake WARNED on substrates that were
built. The fix scopes the check to the id's OWN index entry (header + indented
sub-bullets) and follows that entry's docs/substrate/ links, applying the
same-line rule to each record's logical lines (a status bullet may wrap onto an
indented continuation line, e.g. SD-022's record). Nothing outside the entry is
consulted, which also closes the same-line co-mention false positive (SD-018
read as built off SD-106's "successor shape to SD-018) -- IMPLEMENTED" line).

Pinned:
  (1) grouped multi-record entry: built iff SOME linked record declares it
      (tokens live in the records, not on the index lines).
  (2) grouped entry whose records carry no token stays unbuilt (MECH-027 shape).
  (3) single-record entry: token on the index line still releases (no
      regression); token only inside the linked record releases; a status
      bullet wrapped across an indented continuation line releases.
  (4) same-line co-mention on ANOTHER entry's line does NOT release an id whose
      own entry declares nothing (SD-018 / SD-106 shape).
  (5) flat pre-reflow corpus (no index entry) keeps the original whole-file
      same-line rule, including its token-boundary discipline.
  (6) only docs/substrate/ links are followed; a path outside it or one that
      climbs out is ignored.
  (7) end to end through validate(): the brake is suppressed for a grouped
      entry whose record declares the upstream substrate built, and fires when
      the record does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import validate_queue  # noqa: E402


BRAKE_MARK = "re-derive brake"

# The generated index uses an em-dash after the id; keep it verbatim so the
# fixtures are the real shape, not an ASCII approximation of it.
EM = "—"


def _group_header(sid: str, n: int) -> str:
    return f"- **{sid}** {EM} {n} records, ~1,000 tok total.\n"


def _group_record(title: str, rel: str) -> str:
    return f"    - [{title}]({rel}) *(~500 tok)*\n"


def _single(sid: str, rel: str, title: str) -> str:
    return f"- **[{sid}]({rel})** {EM} {title} *(~500 tok)*\n"


def _reader(docs: dict[str, str]):
    """A doc_reader over an in-memory docs/substrate/ tree."""
    return lambda rel: docs.get(rel, "")


# ------------------------------------------------------------------
# (1) grouped multi-record entry: built iff some linked record declares it
# ------------------------------------------------------------------
def test_grouped_entry_built_via_linked_record():
    md = (
        "## Substrate feature index\n\n"
        + _group_header("SD-011", 2)
        + _group_record("Second Source: Harm History Input (2026-04-08)",
                        "docs/substrate/SD-011-second-source.md")
        + _group_record("SD-012 E3 Integration (2026-04-05)",
                        "docs/substrate/SD-011-sd-012-e3-integration.md")
    )
    docs = {
        "docs/substrate/SD-011-second-source.md":
            "## SD-011 Second Source: Harm History Input (2026-04-08)\n"
            "- SD-011 second source: harm_stream.affective_harm_history_input"
            " -- IMPLEMENTED 2026-04-08.\n",
        "docs/substrate/SD-011-sd-012-e3-integration.md":
            "## SD-011 SD-012 E3 Integration (2026-04-05)\n- wiring notes only.\n",
    }
    # Neither index line carries the token; the record does.
    assert validate_queue._substrate_is_built("SD-011", md, _reader(docs)) is True


def test_grouped_entry_built_via_second_record_only():
    md = (
        _group_header("SD-018", 2)
        + _group_record("Resource Proximity Supervision (2026-04-07)",
                        "docs/substrate/SD-018-resource-proximity.md")
        + _group_record("AMEND: encoder.resource_field_supervision -- IMPLEMENTED (2026-09-02)",
                        "docs/substrate/SD-018-amend-encoder-resource-field.md")
    )
    docs = {
        "docs/substrate/SD-018-resource-proximity.md": "## SD-018 (2026-04-07)\n- notes.\n",
        "docs/substrate/SD-018-amend-encoder-resource-field.md":
            "## SD-018 AMEND: encoder.resource_field_supervision -- IMPLEMENTED (2026-09-02)\n",
    }
    # The sub-bullet's own line has the token but the id only inside the link
    # path ('SD-018-amend-...', a boundary violation on purpose) -- the release
    # comes from following the link, not from relaxing the id token.
    assert validate_queue._substrate_is_built("SD-018", md, _reader(docs)) is True


# ------------------------------------------------------------------
# (2) grouped entry whose records carry no token stays unbuilt
# ------------------------------------------------------------------
def test_grouped_entry_without_token_in_any_record_is_unbuilt():
    md = (
        _group_header("MECH-027", 2)
        + _group_record("precision-scaled commit temperature (2026-09-02)",
                        "docs/substrate/MECH-027-precision.md")
        + _group_record("Build 2: force_sleep_cycle_at_eval_boundary (2026-09-02)",
                        "docs/substrate/MECH-027-build-2.md")
    )
    docs = {
        "docs/substrate/MECH-027-precision.md":
            "## MECH-027 precision-scaled commit temperature (2026-09-02)\n"
            "- MECH-027 needed a graded consumer. FIX: use_precision_scaled_commit_temperature.\n",
        "docs/substrate/MECH-027-build-2.md":
            "## MECH-027 Build 2 (2026-09-02)\n- INVESTIGATION FINDING: force_cycle() already works.\n",
    }
    assert validate_queue._substrate_is_built("MECH-027", md, _reader(docs)) is False


# ------------------------------------------------------------------
# (3) single-record entry shapes
# ------------------------------------------------------------------
def test_single_entry_token_on_index_line_still_releases():
    md = _single("SD-078", "docs/substrate/SD-078-policy.md",
                 "policy.common_mode_invariant -- IMPLEMENTED (2026-07-22)")
    # No reader needed: the index line itself declares it. A reader that
    # raises proves the link was never followed.
    def _boom(rel):
        raise AssertionError(f"link followed unnecessarily: {rel}")
    assert validate_queue._substrate_is_built("SD-078", md, _boom) is True


def test_single_entry_token_only_in_linked_record_releases():
    md = _single("SD-013", "docs/substrate/SD-013-harm-stream.md",
                 "MECH-090, SD-015: Harm Stream + Gate Implementations (2026-04-10)")
    docs = {"docs/substrate/SD-013-harm-stream.md":
            "## SD-013, MECH-090, SD-015: Harm Stream + Gate Implementations (2026-04-10)\n"
            "- SD-013: self_attribution.e2_harm_s_interventional_training -- IMPLEMENTED 2026-04-10.\n"
            "- MECH-090: control_plane.commitment_gated_policy_output -- bistable latch IMPLEMENTED 2026-04-10.\n"}
    # 'Implementations' on the index line is not the token (case-sensitive).
    assert validate_queue._substrate_is_built("SD-013", md, _reader(docs)) is True
    # A multi-id record releases only the ids it declares on their own lines.
    assert validate_queue._substrate_is_built("SD-015", md, _reader(docs)) is False


def test_single_entry_wrapped_status_bullet_in_record_releases():
    md = _single("SD-022", "docs/substrate/SD-022-scheduled-injection.md",
                 "scheduled-injection extension (MECH-302 unblock, 2026-05-30)")
    docs = {"docs/substrate/SD-022-scheduled-injection.md":
            "## SD-022 scheduled-injection extension (MECH-302 unblock, 2026-05-30)\n"
            "- SD-022 scheduled-injection: environment.scheduled_limb_damage_curriculum\n"
            "  -- IMPLEMENTED 2026-05-30. Module: ree_core/environment/causal_grid_world.py\n"
            "  (CausalGridWorldV2).\n"}
    assert validate_queue._substrate_is_built("SD-022", md, _reader(docs)) is True


def test_record_join_is_bounded_by_continuation_not_proximity():
    # Two SEPARATE bullets in one record: the id on one, the token on the next.
    # That is the nearby-unrelated-line shape the original same-line rule
    # refused, and the continuation join must not widen into it.
    md = _single("SD-XYZ", "docs/substrate/SD-XYZ-thing.md", "a thing (2026-01-01)")
    docs = {"docs/substrate/SD-XYZ-thing.md":
            "## a thing (2026-01-01)\n"
            "- SD-XYZ: the substrate under discussion.\n"
            "- SD-OTHER: an unrelated substrate -- IMPLEMENTED 2026-01-01.\n"}
    assert validate_queue._substrate_is_built("SD-XYZ", md, _reader(docs)) is False


# ------------------------------------------------------------------
# (4) same-line co-mention on ANOTHER entry's line does not release
# ------------------------------------------------------------------
def test_comention_on_another_entry_line_does_not_release():
    md = (
        _group_header("SD-018", 1)
        + _group_record("Resource Proximity Supervision (2026-04-07)",
                        "docs/substrate/SD-018-resource-proximity.md")
        + _single("SD-106", "docs/substrate/SD-106-bottleneck.md",
                  "encoder.generic_bottleneck_variance_preservation (successor shape to SD-018)"
                  " -- IMPLEMENTED (2026-09-11)")
    )
    docs = {"docs/substrate/SD-018-resource-proximity.md": "## SD-018 (2026-04-07)\n- notes.\n"}
    # SD-106's line names SD-018 next to IMPLEMENTED: that is SD-106's status.
    assert validate_queue._substrate_is_built("SD-018", md, _reader(docs)) is False
    assert validate_queue._substrate_is_built("SD-106", md, _reader(docs)) is True


# ------------------------------------------------------------------
# (5) flat pre-reflow corpus keeps the original whole-file same-line rule
# ------------------------------------------------------------------
def test_flat_corpus_without_index_entry_uses_legacy_same_line_rule():
    flat = (
        "## SD-018: Resource Proximity Supervision (2026-04-07)\n"
        "- SD-018: encoder.resource_proximity_supervision -- IMPLEMENTED 2026-04-07.\n"
        "- SD-BBB mentioned here.\n"
        "- SD-CCC unrelated -- IMPLEMENTED 2026-01-01.\n"
        "- SD-BBB0 some other substrate -- IMPLEMENTED 2026-01-01.\n"
    )
    assert validate_queue._substrate_is_built("SD-018", flat) is True
    assert validate_queue._substrate_is_built("SD-BBB", flat) is False  # nearby / suffix-extended
    assert validate_queue._substrate_is_built("SD-CCC", flat) is True
    assert validate_queue._substrate_is_built("", flat) is False
    assert validate_queue._substrate_is_built("SD-018", "") is False


def test_index_entry_present_means_legacy_scan_is_not_consulted():
    # An id WITH an entry is decided by that entry alone, even if a body line
    # elsewhere would have satisfied the legacy rule.
    md = (
        "- Hippocampal completion coupling (MECH-105, ARC-028) -- IMPLEMENTED 2026-04-04\n"
        "## Substrate feature index\n"
        + _single("ARC-028", "docs/substrate/ARC-028-thing.md", "trajectory completion (2026-04-04)")
    )
    assert validate_queue._substrate_is_built("ARC-028", md, _reader({})) is False
    # ... while an id WITHOUT an entry still uses the body line.
    assert validate_queue._substrate_is_built("MECH-105", md, _reader({})) is True


# ------------------------------------------------------------------
# (6) only docs/substrate/ links are followed
# ------------------------------------------------------------------
def test_only_docs_substrate_links_are_followed(tmp_path, monkeypatch):
    claude_md = tmp_path / "CLAUDE.md"
    (tmp_path / "docs" / "substrate").mkdir(parents=True)
    (tmp_path / "docs" / "substrate" / "SD-OK-x.md").write_text(
        "- SD-OK: x -- IMPLEMENTED 2026-01-01.\n", encoding="utf-8")
    (tmp_path / "docs" / "elsewhere.md").write_text(
        "- SD-ELSE: y -- IMPLEMENTED 2026-01-01.\n", encoding="utf-8")
    (tmp_path / "secret.md").write_text(
        "- SD-UP: z -- IMPLEMENTED 2026-01-01.\n", encoding="utf-8")
    md = (
        _single("SD-OK", "docs/substrate/SD-OK-x.md", "x (2026-01-01)")
        + _single("SD-ELSE", "docs/elsewhere.md", "y (2026-01-01)")
        + _single("SD-UP", "docs/substrate/../../secret.md", "z (2026-01-01)")
    )
    claude_md.write_text(md, encoding="utf-8")
    monkeypatch.setattr(validate_queue, "_REE_V3_CLAUDE_MD_CANDIDATES", [claude_md])
    # Default reader resolves against the CLAUDE.md that was read.
    assert validate_queue._substrate_is_built("SD-OK", md) is True
    assert validate_queue._substrate_is_built("SD-ELSE", md) is False
    assert validate_queue._substrate_is_built("SD-UP", md) is False
    assert validate_queue._read_substrate_doc("docs/substrate/missing.md") == ""


# ------------------------------------------------------------------
# (7) end to end through validate()
# ------------------------------------------------------------------
def _write_autopsy(planning_dir: Path, slug: str, date: str, claim: str, upstream: str) -> None:
    # Same shape as test_validate_queue_rederive_brake._write_autopsy: the
    # scanner counts a target on recommended_epistemic_category and reads the
    # upstream from recommended_substrate_queue_entry.target_sd_id.
    target = {
        "recommended_epistemic_category": "substrate_ceiling",
        "recommended_evidence_direction": "",
        "claim_ids": [claim],
        "recommended_substrate_queue_entry": {"target_sd_id": upstream},
    }
    (planning_dir / f"failure_autopsy_{slug}_{date}.json").write_text(
        json.dumps({"targets": [target]}), encoding="utf-8")


def _run(tmp_path, monkeypatch, claude_md_text: str, docs: dict[str, str]):
    planning_dir = tmp_path / "planning"
    planning_dir.mkdir(exist_ok=True)
    _write_autopsy(planning_dir, "first", "2026-06-01", "MECH-999", "SD-GRP")
    _write_autopsy(planning_dir, "second", "2026-06-02", "MECH-999", "SD-GRP")
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(claude_md_text, encoding="utf-8")
    for rel, body in docs.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    monkeypatch.setattr(validate_queue, "_REE_ASSEMBLY_PLANNING_DIR_CANDIDATES", [planning_dir])
    monkeypatch.setattr(validate_queue, "_REE_V3_CLAUDE_MD_CANDIDATES", [claude_md])
    item = {
        "queue_id": "V3-EXQ-901",
        "script": "experiments/v3_exq_901_nonexistent.py",
        "priority": 1,
        "machine_affinity": "any",
        "status": "pending",
        "estimated_minutes": 10,
        "claim_ids": ["MECH-999"],
    }
    queue_path = tmp_path / "experiment_queue.json"
    queue_path.write_text(
        json.dumps({"schema_version": "v1", "calibration": {}, "items": [item]}),
        encoding="utf-8")
    errors = validate_queue.validate(queue_path)
    return errors, [w for w in validate_queue._LAST_WARNINGS if BRAKE_MARK in w]


_GRP_MD = (
    "## Substrate feature index\n"
    + _group_header("SD-GRP", 2)
    + _group_record("first record (2026-01-01)", "docs/substrate/SD-GRP-first.md")
    + _group_record("second record (2026-02-02)", "docs/substrate/SD-GRP-second.md")
)


def test_brake_suppressed_when_grouped_entry_record_declares_built(tmp_path, monkeypatch):
    _, brake = _run(tmp_path, monkeypatch, _GRP_MD, {
        "docs/substrate/SD-GRP-first.md": "## SD-GRP first (2026-01-01)\n- notes.\n",
        "docs/substrate/SD-GRP-second.md":
            "## SD-GRP second (2026-02-02)\n- SD-GRP: the thing\n  -- VALIDATED 2026-02-02.\n",
    })
    assert brake == []


def test_brake_fires_when_grouped_entry_records_declare_nothing(tmp_path, monkeypatch):
    errors, brake = _run(tmp_path, monkeypatch, _GRP_MD, {
        "docs/substrate/SD-GRP-first.md": "## SD-GRP first (2026-01-01)\n- notes.\n",
        "docs/substrate/SD-GRP-second.md": "## SD-GRP second (2026-02-02)\n- more notes.\n",
    })
    assert len(brake) == 1
    assert all(BRAKE_MARK not in e for e in errors)  # still warn-only
