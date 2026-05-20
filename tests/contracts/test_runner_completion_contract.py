"""Contract tests for runner completion acceptance.

The runner may see PASS/FAIL from legacy stdout scraping before a script has
written an evidence manifest. Such a result is not evidence-grade and must not
remove the queue item.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_runner import _result_manifest_exists  # noqa: E402


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
