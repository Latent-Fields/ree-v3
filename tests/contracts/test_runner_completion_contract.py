"""Contract tests for runner completion acceptance.

The runner may see PASS/FAIL from legacy stdout scraping before a script has
written an evidence manifest. Such a result is not evidence-grade and must not
remove the queue item.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_runner import (  # noqa: E402
    _result_manifest_exists,
    _classify_no_sentinel_result,
)


def test_empty_output_file_is_not_a_valid_manifest():
    assert _result_manifest_exists({"output_file": ""}) is False


def test_whitespace_output_file_is_not_a_valid_manifest():
    assert _result_manifest_exists({"output_file": "   "}) is False


def test_missing_output_file_is_not_a_valid_manifest():
    assert _result_manifest_exists({}) is False


def test_nonexistent_output_file_is_not_a_valid_manifest(tmp_path):
    assert _result_manifest_exists({
        "output_file": str(tmp_path / "missing_manifest.json"),
    }) is False


def test_existing_output_file_is_a_valid_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"queue_id": "V3-EXQ-590"}\n')
    assert _result_manifest_exists({"output_file": str(manifest)}) is True


# --- No-sentinel classification (V3-EXQ-624 phantom-PASS fix, 2026-06-02) ---
# A crash exits non-zero and writes no sentinel; a process that dies mid-run
# may have already printed `verdict: PASS` lines, so a non-zero exit must
# ALWAYS be ERROR regardless of any stdout-derived verdict.


def test_nonzero_exit_with_stdout_pass_is_error():
    # The V3-EXQ-624 incident: crashed in ARM_2 after ARM_0/ARM_1 printed
    # `verdict: PASS`; exit code 1; no sentinel. Must be ERROR, not PASS.
    result, summary = _classify_no_sentinel_result("PASS", exit_code=1)
    assert result == "ERROR"
    assert "not trusted on crash" in summary


def test_nonzero_exit_with_stdout_fail_is_error():
    result, summary = _classify_no_sentinel_result("FAIL", exit_code=1)
    assert result == "ERROR"
    assert summary


def test_nonzero_exit_with_unknown_is_error():
    result, summary = _classify_no_sentinel_result("UNKNOWN", exit_code=137)
    assert result == "ERROR"
    assert summary


def test_clean_exit_with_stdout_pass_is_trusted_legacy_path():
    # Clean exit + stdout PASS + no sentinel = legacy un-retrofitted script.
    # Trusted: result is PASS and summary is None (caller prints retrofit NOTE).
    result, summary = _classify_no_sentinel_result("PASS", exit_code=0)
    assert result == "PASS"
    assert summary is None


def test_clean_exit_with_stdout_fail_is_trusted_legacy_path():
    result, summary = _classify_no_sentinel_result("FAIL", exit_code=0)
    assert result == "FAIL"
    assert summary is None


def test_clean_exit_with_no_stdout_verdict_is_error():
    result, summary = _classify_no_sentinel_result("UNKNOWN", exit_code=0)
    assert result == "ERROR"
    assert summary
