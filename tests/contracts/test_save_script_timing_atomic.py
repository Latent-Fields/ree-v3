"""Contract: save_script_timing() writes script_timing.json atomically.

script_timing.json is the runner's ETA calibration source. save_script_timing
used to read-modify-write it with Path.write_text, which truncates the file
before writing: a crash (or disk-full, or kill) mid-write left it empty or
partial, and load_script_timing() then silently returned {} -- losing every
script's calibration. (Found 2026-09-26, REE_assembly
evidence/planning/runner_multislot_design_spike_20260926.md sec 8.)

  C1. If the write is interrupted, the previous file is left byte-identical
      and no temp file is left behind.
  C2. A normal write lands the new entry and preserves existing ones.

All paths are redirected to tmp_path; the real script_timing.json is never
touched.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REE_V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REE_V3))

import experiment_runner as er  # noqa: E402

ORIGINAL = {"experiments/existing.py": 12.5}


@pytest.fixture
def timing_file(tmp_path, monkeypatch):
    path = tmp_path / "script_timing.json"
    path.write_text(json.dumps(ORIGINAL, indent=2))
    monkeypatch.setattr(er, "SCRIPT_TIMING_FILE", path)
    return path


def _crash(*_a, **_k):
    raise OSError("simulated crash mid-write")


@pytest.mark.parametrize("target", ["replace", "write"])
def test_interrupted_write_leaves_previous_file_intact(
        timing_file, monkeypatch, target):
    before = timing_file.read_bytes()
    if target == "replace":
        # Crash at the commit step (after the new bytes exist somewhere).
        monkeypatch.setattr(os, "replace", _crash)
        monkeypatch.setattr(Path, "replace", _crash)
    else:
        # Crash while the bytes are being written: truncate-then-fail, which
        # is exactly what a mid-write kill does to a write_text target.
        def _truncate_then_crash(self, *a, **k):
            with open(self, "w") as fh:
                fh.write('{"partial"')
            raise OSError("simulated crash mid-write")
        monkeypatch.setattr(Path, "write_text", _truncate_then_crash)

    with pytest.raises(OSError):
        er.save_script_timing("experiments/new.py", 60.0, 1, 1, 10)

    assert timing_file.read_bytes() == before, (
        "interrupted save_script_timing damaged script_timing.json")
    leftovers = sorted(p.name for p in timing_file.parent.iterdir()
                       if p != timing_file)
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_normal_write_merges_entry(timing_file):
    er.save_script_timing("experiments/new.py", 60.0, 2, 1, 10)
    data = json.loads(timing_file.read_text())
    assert data["experiments/existing.py"] == 12.5
    assert data["experiments/new.py"] == 3000.0  # 60000 ms / (2*1*10)
