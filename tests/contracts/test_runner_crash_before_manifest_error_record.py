"""Contract tests for the crash-before-manifest observability fix in
experiment_runner._write_synthetic_error_manifest.

Background -- V3-EXQ-654e phantom-completion incident (2026-06-17, ree-cloud-1):
  A script crashed 3s in with a TypeError (non-zero exit, no runner sentinel,
  no manifest written). The runner's ERROR branch saw result["result"]=="ERROR"
  with an EMPTY output_file and proceeded straight to
  coordinator_client.report_queue_remove(queue_id, "ERROR") -- which flips the
  coordinator DB row to completed with NO results row and NO manifest. The
  phantom completion was invisible to pending_review.md and never routed to
  /diagnose-errors; its only trace was the worker's journalctl. This is the
  FAIL/ERROR-class twin of the already-fixed UNKNOWN silent-drop (f36461d).

Fix:
  _write_synthetic_error_manifest() mints a scoring-neutral, REVIEWABLE ERROR
  manifest at evidence/experiments/<run_id>.json which the ERROR branch then
  ships via _report_result_and_align (creating a coordinator results row +
  materialising the manifest on origin) before queue removal.

Contracts:
  C1. A non-zero-exit / no-sentinel / empty-output_file crash produces a
      manifest file on disk with outcome/result == ERROR.
  C2. The synthetic record is scoring-neutral: claim_ids == [],
      experiment_purpose == diagnostic, evidence_direction == non_contributory
      -- so a crash never weights any claim's confidence.
  C3. The stdout-derived verdict is NOT trusted: result_summary carries the
      classify summary, not a PASS/FAIL; outcome stays ERROR even if a partial
      stdout PASS was present on the result dict.
  C4. The run_id obeys the flat-manifest conventions (starts v3_, ends _v3,
      matches _UNTRACKED_FLAT_MANIFEST_RE) so the spool writer materialises it
      at the flat evidence/experiments/<run_id>.json path that
      generate_pending_review.py scans, and the manifest meets that script's
      ERROR-detection criterion.
  C5. No REE_assembly checkout available -> (None, None) (caller falls back to
      the legacy queue-removal path); no crash.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402


def _crash_result() -> dict:
    """A result dict shaped exactly like a crash-before-manifest:
    non-zero exit, no sentinel, empty output_file."""
    return {
        "result": "ERROR",
        "result_summary": (
            "Non-zero exit code 1; no runner sentinel "
            "(stdout-derived 'PASS' not trusted on crash)"
        ),
        "started_at": "2026-06-17T08:53:02Z",
        "completed_at": "2026-06-17T08:53:05Z",
        "output_file": "",
        "actual_secs": 3.0,
        "exit_code": 1,
        "has_sentinel": False,
    }


def _item() -> dict:
    return {
        "queue_id": "V3-EXQ-654e",
        "script": "experiments/v3_exq_654e_arc062_gapb_rule_apprehension_behavioural_falsifier.py",
        "title": "arc_062 GAP-B behavioural falsifier",
    }


def _make_evidence_dir(tmp_path: Path) -> Path:
    asm = tmp_path / "REE_assembly"
    (asm / "evidence" / "experiments").mkdir(parents=True)
    return asm


def test_c1_crash_before_manifest_writes_error_record(tmp_path):
    asm = _make_evidence_dir(tmp_path)
    run_id, manifest_path = experiment_runner._write_synthetic_error_manifest(
        asm, "V3-EXQ-654e", _item(), _crash_result(), "ree-cloud-1")
    assert run_id is not None and manifest_path is not None
    p = Path(manifest_path)
    assert p.is_file()
    doc = json.loads(p.read_text())
    assert doc["outcome"] == "ERROR"
    assert doc["result"] == "ERROR"
    assert doc["queue_id"] == "V3-EXQ-654e"
    assert doc["machine"] == "ree-cloud-1"
    assert doc.get("crash_before_manifest") is True
    assert doc.get("exit_code") == 1


def test_c2_record_is_scoring_neutral(tmp_path):
    asm = _make_evidence_dir(tmp_path)
    _run_id, manifest_path = experiment_runner._write_synthetic_error_manifest(
        asm, "V3-EXQ-654e", _item(), _crash_result(), "ree-cloud-1")
    doc = json.loads(Path(manifest_path).read_text())
    # A crash is not evidence for or against any hypothesis: it must never
    # weight claim confidence (build_experiment_indexes scores on claim_ids).
    assert doc["claim_ids"] == []
    assert doc["experiment_purpose"] == "diagnostic"
    assert doc["evidence_direction"] == "non_contributory"


def test_c3_stdout_verdict_not_trusted(tmp_path):
    asm = _make_evidence_dir(tmp_path)
    # Even with a partial stdout 'PASS' visible in the summary (the V3-EXQ-624
    # crash-magnet shape), the synthesized record stays ERROR.
    crash = _crash_result()
    crash["result_summary"] = "verdict: PASS (ARM_0) | Non-zero exit code 1; no runner sentinel"
    _run_id, manifest_path = experiment_runner._write_synthetic_error_manifest(
        asm, "V3-EXQ-624", _item(), crash, "ree-cloud-3")
    doc = json.loads(Path(manifest_path).read_text())
    assert doc["outcome"] == "ERROR"
    assert doc["result"] == "ERROR"
    # The classify summary is recorded verbatim, but it never becomes the verdict.
    assert "PASS" not in (doc["result"], doc["outcome"])


@pytest.mark.parametrize("queue_id,gen", [
    ("V3-EXQ-654e", "v3"),
    ("V4-EXQ-001", "v4"),
    ("V5-EXQ-001", "v5"),
    # Unparseable queue_id falls back to v3 (previous behaviour).
    ("EXQ-999", "v3"),
])
def test_c4_run_id_matches_flat_manifest_conventions(tmp_path, queue_id, gen):
    asm = _make_evidence_dir(tmp_path)
    run_id, manifest_path = experiment_runner._write_synthetic_error_manifest(
        asm, queue_id, _item(), _crash_result(), "ree-cloud-1")
    # Generation follows the queue_id: a V4 crash must NOT be recorded under a
    # V3 run_id (it would mislabel the evidence).
    assert run_id.startswith(f"{gen}_")
    assert run_id.endswith(f"_{gen}")
    rel = "evidence/experiments/" + Path(manifest_path).name
    assert experiment_runner._UNTRACKED_FLAT_MANIFEST_RE.match(rel), rel
    # generate_pending_review.py surfaces ERROR-class manifests by reading the
    # manifest result/outcome -- mirror that detection here so the record is
    # provably reviewable.
    doc = json.loads(Path(manifest_path).read_text())
    detected = doc.get("outcome") or doc.get("result")
    assert detected == "ERROR"


def test_c5_no_assembly_checkout_returns_none(tmp_path):
    # No ree_assembly_path at all.
    run_id, manifest_path = experiment_runner._write_synthetic_error_manifest(
        None, "V3-EXQ-654e", _item(), _crash_result(), "ree-cloud-1")
    assert run_id is None and manifest_path is None
    # ree_assembly_path set but no evidence/experiments dir -> also (None, None).
    bare = tmp_path / "empty_assembly"
    bare.mkdir()
    run_id2, manifest_path2 = experiment_runner._write_synthetic_error_manifest(
        bare, "V3-EXQ-654e", _item(), _crash_result(), "ree-cloud-1")
    assert run_id2 is None and manifest_path2 is None
