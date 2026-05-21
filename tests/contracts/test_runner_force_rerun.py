"""Contract: force_rerun queue items stay eligible after prior completion."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_runner import (  # noqa: E402
    item_has_force_rerun,
    should_skip_as_completed,
)


def test_item_has_force_rerun_requires_literal_true():
    assert item_has_force_rerun({"force_rerun": True}) is True
    assert item_has_force_rerun({"force_rerun": "true"}) is False
    assert item_has_force_rerun({}) is False


def test_should_skip_as_completed_when_in_set():
    completed = {"V3-EXQ-543k"}
    item = {"queue_id": "V3-EXQ-543k"}
    assert should_skip_as_completed(item, completed) is True


def test_force_rerun_not_skipped_when_in_completed_set():
    completed = {"V3-EXQ-543k"}
    item = {"queue_id": "V3-EXQ-543k", "force_rerun": True}
    assert should_skip_as_completed(item, completed) is False


def test_force_rerun_not_skipped_for_peer_completion():
    completed = {"V3-EXQ-599a"}  # e.g. merged from ree-cloud-2.json
    item = {"queue_id": "V3-EXQ-599a", "force_rerun": True}
    assert should_skip_as_completed(item, completed) is False
