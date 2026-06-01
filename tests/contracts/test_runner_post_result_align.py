"""Contracts for post-result checkout alignment (Phase 3 read path)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402


def test_untracked_flat_manifest_paths_selected():
  paths = [
      "evidence/experiments/v3_exq_614c_foo_20260601T124509Z_v3.json",
      "evidence/experiments/_runner_signals/V3-EXQ-614c.json",
      "evidence/experiments/v3_exq_614c_foo/INDEX.md",
      "evidence/planning/foo.md",
  ]
  assert experiment_runner._UNTRACKED_FLAT_MANIFEST_RE.match(paths[0])
  assert experiment_runner._UNTRACKED_RUNNER_SIGNAL_RE.match(paths[1])
  assert not experiment_runner._UNTRACKED_FLAT_MANIFEST_RE.match(paths[2])
  assert not experiment_runner._UNTRACKED_FLAT_MANIFEST_RE.match(paths[3])


def test_fail_branch_reports_and_aligns():
    src = Path(experiment_runner.__file__).read_text(encoding="utf-8")
    start = src.find('if result["result"] == "FAIL":')
    end = src.find('if result["result"] == "UNKNOWN":', start)
    assert start != -1 and end != -1
    block = src[start:end]
    assert "_report_result_and_align(" in block
    assert block.find("_report_result_and_align(") < block.find(
        "report_queue_remove(queue_id, \"FAIL\")"
    )


def test_pass_branch_reports_and_aligns():
    src = Path(experiment_runner.__file__).read_text(encoding="utf-8")
    assert "_report_result_and_align(" in src
    assert "align_after_coordinator_result(" in src
    assert "def align_ree_assembly_checkout(" in src
