"""
Contract tests for the cross-machine completed-queue-id DENOMINATOR in
experiment_runner.py's `merge_peer_status()` (repaired 2026-09-23,
chip-20260923-runner-merge-peer-status-dead).

THE DEFECT THIS CLOSES. `merge_peer_status()`'s ONLY cross-machine source
used to be REE_assembly/evidence/experiments/runner_status/*.json (one file
per machine, synced via git). That channel was retired 2026-09-06
(REE_assembly 6320b7f3fad, "retire the frozen telemetry dirs"; CLAUDE.md
A-93): sync_daemon no longer materialises any OTHER machine's file into that
directory. The directory itself is NOT missing on a live machine --
find_default_status_path() recreates it locally every runner startup (its
parent, evidence/experiments/, always exists) and this machine writes its
OWN file there -- so `status_dir.is_dir()` stays True and the old code never
took the early-return branch. What silently died is the CROSS-MACHINE half:
the glob only ever finds this machine's own file, which completed_ids
already has from `existing_completed`. A duplicate run dispatched by another
machine has been invisible to this guard since 2026-09-06, with nothing
distinguishing that state from a genuinely clean fleet -- the same
negative-instrument shape as the sibling defect in validate_queue.py
(chip-20260922-validate-queue-burned-id-guard-inert, ree-v3 08f981336b), but
MORE dangerous here: validate_queue.py is a commit-time advisory,
merge_peer_status() is the runner's own dispatch-time last line of defence
against running the same multi-hour cloud experiment twice.

THE FIX reuses validate_queue.py's REE_assembly/evidence/experiments
manifest-filename scan (`_scan_completed_queue_ids()` /
`_completed_queue_ids_available()`) as the LIVE cross-machine source, rather
than reimplementing it -- both files live in this ree-v3/ checkout and
experiment_runner.py already imports validate_queue.validate elsewhere.

FAIL-OPEN, DELIBERATELY, WITH A RECORDED TRACE -- this is the DELICATE part
and is NOT simply copied from the sibling fix. validate_queue.py stays
fail-open because it is a per-commit hook and a wedged commit hook blocks
the whole fleet. merge_peer_status() sits in the runner's DISPATCH path, so
naively going fail-closed here ("refuse to dispatch anything when the
cross-machine source can't be read") sounds more defensible -- a duplicate
run wastes hours of cloud compute and corrupts the evidence record. This
repair does NOT do that: a hard dispatch-halt on every transient source
outage (evidence dir not yet mounted on a freshly-booted cloud worker, a
momentary rsync/NFS hiccup) would convert a soft, rare degradation into a
FLEET-WIDE experiment-dispatch stop -- worse than the failure mode it
guards against, and the same class of mistake CLAUDE.md's stale-claim
section warns against for a structurally identical reason ("absence of
telemetry is not abandonment"). The chosen middle path: merge_peer_status()
never blocks dispatch, but its return value now STRUCTURALLY records
whether the cross-machine source was actually live this pass
(PeerStatusResult.available), and both call sites persist that into the
per-machine `status` dict -- which write_status() serialises to disk and
coordinator_client.report_status() forwards to the coordinator every status
write. A degraded pass is therefore ATTRIBUTABLE after the fact (which
machine, which pass, whether the denominator it dispatched against was
real), rather than silently indistinguishable from a clean one.

Branches pinned:
  C1  the source-unavailable case is a distinct signal
      (PeerStatusResult.available=False, source="none"), not merely an
      empty queue_ids -- and it is a print, but ALSO structural: the return
      value itself carries it, independent of whatever gets printed.
  C1b merge_peer_status() prints a COULD-NOT-DETERMINE warning distinct from
      "nothing found" when the source is unavailable.
  C2  fail-open: an unavailable source does not raise, and does not prevent
      merge_peer_status() from returning a usable (if degraded) result --
      the caller's dispatch loop is never blocked by this function.
  C3  THE REGRESSION PIN, against the real corpus (skipped, not failed, when
      the real REE_assembly evidence tree is absent): the source resolves
      (available=True) and PeerStatusResult.queue_ids is non-empty, with a
      specific known-complete id (V3-EXQ-1025, PASSED 2026-09-11, permanent
      history) present as a canary. This is the test that would have caught
      the inert window: it fails the moment the denominator collapses back
      to "this machine's own history only".
  C4  end-to-end via should_skip_as_completed(): with the REAL source live,
      a queue item re-using a real completed id is skipped (without
      force_rerun) and not skipped (with force_rerun=True) -- the actual
      consumer of merge_peer_status()'s return value in the runner loop.
  C5  a fabricated, never-completed id is not in queue_ids (no false
      positive from the new source).
  C6  the legacy per-machine runner_status/*.json merge (ERROR/non-ERROR
      dedup, monolithic-file write) still works when that directory DOES
      have real peer files -- this repair adds a source, it does not remove
      the old one, and the refactor (wrapping the glob in
      `if status_dir.is_dir():`) must not have broken it.

THE BLIND-SPOT MEASUREMENT (see also the CLOSE note / task report): running
C3/C4 against the OLD (pre-repair) `merge_peer_status`, which returned a
bare `set` sourced only from status_path.parent's glob, FAILS on a scratch
status_path (no peer files there) -- it silently returns `set()` with no
way to distinguish "confirmed no peer completions" from "the cross-machine
source could not be read", and should_skip_as_completed() then does not
flag a real, already-completed id as skippable. The existing guard test
(tests/contracts/test_runner_force_rerun.py) continues to pass unchanged
against both old and new code, since it exercises should_skip_as_completed()
directly against a hand-built set and never calls merge_peer_status() at
all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

import validate_queue  # noqa: E402
from experiment_runner import (  # noqa: E402
    PeerStatusResult,
    merge_peer_status,
    should_skip_as_completed,
)

EVIDENCE_DIR = Path(
    "/Users/dgolden/REE_Working/REE_assembly/evidence/experiments"
)
REAL_EVIDENCE_PRESENT = EVIDENCE_DIR.is_dir()

# Same permanent canary validate_queue.py's sibling test uses: V3-EXQ-1025
# PASSED 2026-09-11T18:11:00Z. History cannot un-happen.
KNOWN_COMPLETED_ID = "V3-EXQ-1025"


def _require_corpus_under_test():
    """Skip -- as an explicit CANNOT-DETERMINE, never a pass -- when the
    evidence dir the MODULE UNDER TEST resolves is not the real corpus and
    holds no files at all. Added 2026-09-25
    (chip-20260925-remote-staging-evidence-corpus).

    WHY. `REAL_EVIDENCE_PRESENT` above asks about a hardcoded Mac path, but
    validate_queue resolves its OWN candidate list, sibling-of-the-queue
    first. On a remote_pytest.sh staged tree that sibling
    (<STAGE_ROOT>/REE_assembly/evidence/experiments) exists but ships only
    its scripts/ subdir, so the module resolves an EMPTY dir. On ree-worker-4
    -- the one fleet box that also mirrors /Users/dgolden/REE_Working --
    the hardcoded guard therefore did not skip, and C3/C4 went red on every
    remotely-gated commit (measured 2026-09-25: resolved dir 0 files, 0 ids,
    guard path present), pushing sessions to --no-verify; C5/C6 passed there
    only vacuously.

    WHY THIS CANNOT HIDE A REAL BREAK (the test half). The skip is decided by
    a RAW listing (any regular file at all), never by the parser under test,
    and ONLY when the resolved dir is not EVIDENCE_DIR itself:
      * resolved == the real corpus (the Mac, main checkout or a worktree)
        -> strict, whatever it contains: a parse/regex regression still
        FAILS C3, and a corpus that vanished still fails.
      * resolved is None -> strict: the test's own "candidate paths have
        drifted" assertion fires.
      * resolved elsewhere but holding >=1 file -> strict: zero parsed ids
        from a non-empty listing is exactly the silent-empty mode C3 pins.
    """
    resolved = validate_queue._find_evidence_dir()
    if resolved is None:
        return
    try:
        if resolved.resolve() == EVIDENCE_DIR.resolve():
            return
        has_any_file = any(p.is_file() for p in resolved.iterdir())
    except OSError:
        has_any_file = False
    if not has_any_file:
        pytest.skip(
            "CANNOT DETERMINE (not a pass): validate_queue resolves its "
            "evidence dir to %s, which holds no files -- a remote_pytest.sh "
            "staged tree ships only that dir's scripts/ subdir. The "
            "real-corpus verdict for this test is the Mac run against %s."
            % (resolved, EVIDENCE_DIR))


@pytest.fixture
def missing_evidence_source(monkeypatch):
    """Point validate_queue's evidence-dir candidates at a path that does
    not exist -- the exact shape of the 2026-09-06 retirement as it now
    manifests for the runner's cross-machine dedup source."""
    monkeypatch.setattr(
        validate_queue,
        "_REE_ASSEMBLY_EVIDENCE_DIR_CANDIDATES",
        [Path("/nonexistent/merge-peer-status-test/evidence/experiments")],
    )


# ---- C1 / C1b -- source-unavailable is a DISTINCT, STRUCTURAL signal ------

def test_c1_unavailable_source_yields_structural_cannot_determine(
    missing_evidence_source, tmp_path
):
    status_path = tmp_path / "runner_status" / "TESTMACHINE.json"
    result = merge_peer_status(status_path)
    assert isinstance(result, PeerStatusResult)
    assert result.available is False
    assert result.source == "none"
    assert result.queue_ids == set()


def test_c1b_warns_distinctly_when_unavailable(
    missing_evidence_source, tmp_path, capsys
):
    status_path = tmp_path / "runner_status" / "TESTMACHINE.json"
    merge_peer_status(status_path)
    out = capsys.readouterr().out
    assert "COULD NOT DETERMINE" in out
    assert "peer-status guard" in out


# ---- C2 -- fail-OPEN: unavailable source must not block/raise -------------

def test_c2_unavailable_source_does_not_raise_and_stays_usable(
    missing_evidence_source, tmp_path
):
    status_path = tmp_path / "runner_status" / "TESTMACHINE.json"
    result = merge_peer_status(status_path)  # must not raise
    # The dispatch loop's consumer must keep working against a degraded
    # (empty, available=False) result rather than crashing on it.
    item = {"queue_id": "V3-EXQ-000000"}
    assert should_skip_as_completed(item, result.queue_ids) is False


# ---- C3 -- THE REGRESSION PIN: real corpus, real fixture -------------------

@pytest.mark.skipif(
    not REAL_EVIDENCE_PRESENT,
    reason="needs the real REE_assembly evidence tree",
)
def test_c3_source_resolves_and_is_non_empty_on_the_real_corpus(tmp_path):
    """THE test that would have caught the inert cross-machine window. Not a
    stub: calls the guard's real resolution against the real, live
    REE_Working checkout's evidence tree, exactly as the runner would at
    dispatch time on this machine."""
    _require_corpus_under_test()
    status_path = tmp_path / "runner_status" / "TESTMACHINE.json"
    result = merge_peer_status(status_path)
    assert result.available is True, (
        "the evidence dir exists on this machine but the guard could not "
        "find it -- the guard's candidate paths have drifted from reality"
    )
    assert result.queue_ids, (
        "the evidence dir resolved but ZERO completed queue ids were "
        "extracted from a non-empty corpus -- this is the exact "
        "silent-empty failure mode this repair exists to close"
    )
    assert KNOWN_COMPLETED_ID in result.queue_ids, (
        f"{KNOWN_COMPLETED_ID} is a permanent, already-landed PASS "
        "(2026-09-11) and must always be visible to this guard; its "
        "absence means the source has drifted"
    )
    assert result.source in ("evidence_manifest", "both")


# ---- C4/C5 -- end-to-end against the real, live source, via the actual ----
# ---- consumer (should_skip_as_completed), not just the raw set ------------

@pytest.mark.skipif(
    not REAL_EVIDENCE_PRESENT,
    reason="needs the real REE_assembly evidence tree",
)
def test_c4_real_completed_id_is_skipped_without_force_rerun(tmp_path):
    _require_corpus_under_test()
    status_path = tmp_path / "runner_status" / "TESTMACHINE.json"
    result = merge_peer_status(status_path)
    item = {"queue_id": KNOWN_COMPLETED_ID}
    assert should_skip_as_completed(item, result.queue_ids) is True


@pytest.mark.skipif(
    not REAL_EVIDENCE_PRESENT,
    reason="needs the real REE_assembly evidence tree",
)
def test_c4b_force_rerun_still_overrides(tmp_path):
    _require_corpus_under_test()
    status_path = tmp_path / "runner_status" / "TESTMACHINE.json"
    result = merge_peer_status(status_path)
    item = {"queue_id": KNOWN_COMPLETED_ID, "force_rerun": True}
    assert should_skip_as_completed(item, result.queue_ids) is False


@pytest.mark.skipif(
    not REAL_EVIDENCE_PRESENT,
    reason="needs the real REE_assembly evidence tree",
)
def test_c5_never_completed_id_is_not_in_the_set(tmp_path):
    _require_corpus_under_test()
    status_path = tmp_path / "runner_status" / "TESTMACHINE.json"
    result = merge_peer_status(status_path)
    assert "V3-EXQ-999999z" not in result.queue_ids


# ---- C6 -- legacy per-machine peer-dir merge still works -------------------

def test_c6_peer_status_dir_merge_still_dedups_and_writes_monolithic(
    missing_evidence_source, tmp_path
):
    """With the evidence-manifest source unavailable (isolating this test
    from the real corpus), the ORIGINAL per-machine runner_status/*.json
    merge -- ERROR/non-ERROR dedup and the monolithic runner_status.json
    write -- must still work exactly as before this repair; the refactor
    only ADDED a source, it must not have broken the existing one."""
    status_dir = tmp_path / "evidence" / "experiments" / "runner_status"
    status_dir.mkdir(parents=True)

    (status_dir / "machineA.json").write_text(json.dumps({
        "completed": [
            {"queue_id": "V3-EXQ-700a", "result": "ERROR", "completed_at": "t1"},
        ]
    }))
    (status_dir / "machineB.json").write_text(json.dumps({
        "completed": [
            # Same id, non-ERROR result from a different machine -- must win.
            {"queue_id": "V3-EXQ-700a", "result": "PASS", "completed_at": "t2"},
            {"queue_id": "V3-EXQ-701b", "result": "FAIL", "completed_at": "t3"},
        ]
    }))

    status_path = status_dir / "machineC.json"
    result = merge_peer_status(status_path)

    assert result.queue_ids >= {"V3-EXQ-700a", "V3-EXQ-701b"}
    assert result.source in ("peer_status_dir", "both")

    monolithic = status_dir.parent / "runner_status.json"
    assert monolithic.exists()
    merged = json.loads(monolithic.read_text())
    completed_by_id = {c["queue_id"]: c for c in merged["completed"]}
    assert completed_by_id["V3-EXQ-700a"]["result"] == "PASS", (
        "non-ERROR must win over ERROR for the same queue_id across "
        "machine files"
    )
