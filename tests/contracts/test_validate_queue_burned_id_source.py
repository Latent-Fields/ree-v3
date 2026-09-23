"""
Contract tests for the burned-ID guard's DENOMINATOR in validate_queue.py
(_scan_completed_queue_ids / _completed_queue_ids_available), repaired
2026-09-23 (chip-20260922-validate-queue-burned-id-guard-inert).

THE DEFECT THIS CLOSES. The guard's original source was
REE_assembly/evidence/experiments/runner_status/ (a per-machine directory).
That directory was deliberately DELETED on 2026-09-06 (REE_assembly
6320b7f3fad, "retire the frozen telemetry dirs"; CLAUDE.md A-93). From that
date `_find_status_dir()` returned None and `_scan_completed_queue_ids()`
returned `{}` -- silently, with no distinguishable signal from "nothing is
burned". The guard passed every commit for 16 days without checking anything,
and V3-EXQ-1055 (re-landed 2026-09-20 onto an id that had already PASSED two
days earlier) is the confirmed real-world instance this would have caught.

THIS IS A NEGATIVE-INSTRUMENT DEFECT (CLAUDE.md "General Rules" -- negative
instruments): "nothing found" and "the search broke" produced the identical
`{}`. Every OTHER test that touches `_scan_completed_queue_ids` monkeypatches
it wholesale with `lambda: {}` to get it out of the way (see
test_validate_queue_seed_enforcement.py, test_validate_queue_prereg_share_
feasibility.py) -- which tests the comparison logic downstream of the scan,
never whether the scan itself can still find anything. This file is the
missing other half: it asserts the SOURCE resolves and yields a non-empty
set under a REAL fixture (C3), and that losing the source is structurally
distinguishable from finding it empty (C1/C2), rather than stubbing the
question away.

Branches pinned:
  C1  the source-unavailable case is a distinct signal (`available=False`),
      not merely an empty `_scan_completed_queue_ids()` result -- and
      `validate()` surfaces it as a WARNING (fail-OPEN: this is a per-commit
      hook, and a hook that blocks every commit on a transient FS/pull gap
      is worse than one that misses a pass this time).
  C2  the SAME condition (source unavailable) must NOT block: a queue item
      whose id would otherwise be flagged as burned produces no ERROR when
      the source cannot be reached -- fail-open is a property of the
      commit-hook caller, verified here at the validate() level since that
      is the shared entry point every caller (hook, runner preflight) goes
      through.
  C3  THE REGRESSION PIN, against the real corpus (skipped, not failed, when
      the real REE_assembly evidence tree is absent -- e.g. a bare worktree
      or a box that has not pulled REE_assembly, mirroring
      test_burned_queue_entry_detector.py's own skipUnless): the source
      resolves (`available=True`) and yields a NON-EMPTY set, with a
      specific known-complete id (V3-EXQ-1025, which PASSED 2026-09-11 and
      is permanent history) present as a canary. This is the test that
      would have caught the 16-day inert window: it fails the moment the
      denominator collapses back to empty, rather than passing on a stub.
  C4  end-to-end: with the REAL source live, re-queuing a REAL completed id
      (V3-EXQ-1025) without force_rerun is a blocking ERROR from validate().
  C5  the same id WITH force_rerun: true is not blocked (existing escape
      hatch preserved).
  C6  a fabricated, never-completed id is not blocked (no false positive
      from the new source).
  C7  the "resolved but zero matches despite a non-trivial candidate pool"
      tier-3 defence fires its own distinct warning (a parsing/regex drift
      degrading silently back to "empty means clean" a second way).

THE BLIND-SPOT MEASUREMENT this module exists to report (see also the
module-level comment in validate_queue.py's guard section): running C3/C4
against the OLD (pre-repair) `_scan_completed_queue_ids`/`_find_status_dir`
implementation -- i.e. a source pinned at the now-deleted
runner_status/ directory -- FAILS (C3: `available` is False and the id set
is empty; C4: no error is raised for a real, known-completed id). The OLD
guard code does not raise or crash on that input -- it returns `{}` and
`validate()` reports zero errors, i.e. it silently PASSES a queue this repair
now correctly blocks. That asymmetry (new test fails on the old code; the
old code's own callers see no failure) is the 16-day blind spot, reproduced
here rather than only asserted in prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

import validate_queue  # noqa: E402

EVIDENCE_DIR = Path(
    "/Users/dgolden/REE_Working/REE_assembly/evidence/experiments"
)
REAL_EVIDENCE_PRESENT = EVIDENCE_DIR.is_dir()

# A permanent, already-landed real completion: V3-EXQ-1025 PASSED
# 2026-09-11T18:11:00Z (v3_exq_1025_mech349_crf_churn_retirement_...).
# History cannot un-happen, so this id is a safe canary -- it can only ever
# gain sibling stints (already-adjudicated by
# tests/contracts/test_burned_queue_entry_detector.py), never lose this one.
KNOWN_COMPLETED_ID = "V3-EXQ-1025"


def _minimal_queue_item(queue_id, **overrides):
    item = {
        "queue_id": queue_id,
        "script": "experiments/__validate_queue_burn_source_test__.py",
        "priority": 1,
        "machine_affinity": "any",
        "status": "pending",
        "estimated_minutes": 1,
        "claim_ids": [],
    }
    item.update(overrides)
    return item


def _write_queue(tmp_path, items):
    import json

    p = tmp_path / "queue.json"
    p.write_text(json.dumps({
        "schema_version": "v1",
        "calibration": {},
        "items": items,
    }))
    return p


@pytest.fixture
def missing_source(monkeypatch):
    """Point the guard at candidate dirs that do not exist -- the exact
    shape of the 2026-09-06 retirement (source present in code, absent on
    disk)."""
    monkeypatch.setattr(
        validate_queue,
        "_REE_ASSEMBLY_EVIDENCE_DIR_CANDIDATES",
        [Path("/nonexistent/burned-id-guard-test/evidence/experiments")],
    )


# ---- C1 -- source-unavailable is a DISTINCT signal -------------------------

def test_c1_unavailable_source_is_not_an_empty_result(missing_source):
    assert validate_queue._completed_queue_ids_available() is False
    # The dict-shaped scan still degrades to {} (existing monkeypatch
    # contract downstream tests rely on), but that must not be read alone.
    assert validate_queue._scan_completed_queue_ids() == {}


def test_c1b_validate_warns_when_source_unavailable(missing_source, tmp_path):
    queue_path = _write_queue(tmp_path, [_minimal_queue_item(KNOWN_COMPLETED_ID)])
    validate_queue.validate(queue_path)
    warnings = validate_queue._LAST_WARNINGS
    assert any(
        "COULD NOT DETERMINE" in w and "burned-ID guard" in w for w in warnings
    ), f"expected a cannot-determine warning, got: {warnings}"


# ---- C2 -- fail-OPEN: unavailable source must not block --------------------

def test_c2_unavailable_source_does_not_block_a_would_be_burn(
    missing_source, tmp_path
):
    """A per-commit hook must not wedge every commit on a transient source
    outage -- see the module docstring and validate_queue.py's own comment
    on the fail-open/fail-closed boundary."""
    queue_path = _write_queue(tmp_path, [_minimal_queue_item(KNOWN_COMPLETED_ID)])
    errors = validate_queue.validate(queue_path)
    assert not any("completion manifest" in e for e in errors), (
        f"burned-ID guard must not fire when its source is unreachable: {errors}"
    )


# ---- C3 -- THE REGRESSION PIN: real corpus, real fixture -------------------

@pytest.mark.skipif(
    not REAL_EVIDENCE_PRESENT,
    reason="needs the real REE_assembly evidence tree",
)
def test_c3_source_resolves_and_is_non_empty_on_the_real_corpus():
    """THE test that would have caught the 16-day inert window. Not a stub:
    this calls the guard's real resolution against the real, live
    REE_Working checkout's evidence tree, exactly as a commit-time hook run
    on this machine would."""
    assert validate_queue._completed_queue_ids_available() is True, (
        "the evidence dir exists on this machine but the guard could not "
        "find it -- the guard's candidate paths have drifted from reality"
    )
    scan = validate_queue._scan_completed_queue_ids()
    assert scan, (
        "the evidence dir resolved but the guard extracted ZERO completed "
        "queue ids from a non-empty corpus -- this is the exact silent-"
        "empty failure mode this repair exists to close, now via a parsing "
        "miss instead of a missing directory"
    )
    assert KNOWN_COMPLETED_ID in scan, (
        f"{KNOWN_COMPLETED_ID} is a permanent, already-landed PASS "
        "(2026-09-11) and must always be visible to the guard; its absence "
        "means the stem/id regex has drifted from the real naming "
        "convention"
    )


# ---- C4/C5/C6 -- end-to-end against the real, live source -----------------

@pytest.mark.skipif(
    not REAL_EVIDENCE_PRESENT,
    reason="needs the real REE_assembly evidence tree",
)
def test_c4_real_completed_id_blocks_without_force_rerun(tmp_path):
    queue_path = _write_queue(
        tmp_path, [_minimal_queue_item(KNOWN_COMPLETED_ID)]
    )
    errors = validate_queue.validate(queue_path)
    matching = [e for e in errors if "completion manifest" in e]
    assert len(matching) == 1, f"expected exactly one burned-id error, got: {errors}"
    assert KNOWN_COMPLETED_ID in matching[0]


@pytest.mark.skipif(
    not REAL_EVIDENCE_PRESENT,
    reason="needs the real REE_assembly evidence tree",
)
def test_c5_force_rerun_still_suppresses_the_guard(tmp_path):
    queue_path = _write_queue(
        tmp_path,
        [_minimal_queue_item(KNOWN_COMPLETED_ID, force_rerun=True)],
    )
    errors = validate_queue.validate(queue_path)
    assert not any("completion manifest" in e for e in errors)


@pytest.mark.skipif(
    not REAL_EVIDENCE_PRESENT,
    reason="needs the real REE_assembly evidence tree",
)
def test_c6_never_completed_id_is_not_flagged(tmp_path):
    queue_path = _write_queue(
        tmp_path, [_minimal_queue_item("V3-EXQ-999999z")]
    )
    errors = validate_queue.validate(queue_path)
    assert not any("completion manifest" in e for e in errors)


# ---- C7 -- resolved-but-empty tier-3 defence -------------------------------

def test_c7_resolved_but_zero_matches_warns_distinctly(tmp_path, monkeypatch):
    """A source dir that resolves but contains no conforming filenames at
    all (a timestamp-shaped name with no v[34]_exq_ prefix) must still warn
    -- the guard's denominator being legitimately 0 despite candidates
    existing is itself suspicious on what is meant to be a live corpus."""
    fake_dir = tmp_path / "evidence_experiments"
    fake_dir.mkdir()
    (fake_dir / "not_a_queue_id_20260101T000000Z_v3.json").write_text("{}")
    monkeypatch.setattr(
        validate_queue, "_REE_ASSEMBLY_EVIDENCE_DIR_CANDIDATES", [fake_dir]
    )
    queue_path = _write_queue(tmp_path, [_minimal_queue_item("V3-EXQ-1")])
    validate_queue.validate(queue_path)
    warnings = validate_queue._LAST_WARNINGS
    assert any("ZERO matched" in w for w in warnings), (
        f"expected the resolved-but-empty tier-3 warning, got: {warnings}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
