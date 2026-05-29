"""Contract tests for the V3-EXQ-592b FAIL/ERROR silent-drop fix.

Background -- V3-EXQ-592b silent-drop (2026-05-29, DLAPTOP-4):
  The FAIL branch and ERROR branch in experiment_runner.py main loop
  skipped the three calls the PASS branch makes after a manifest is
  delivered:
    1. _result_manifest_exists(result)             -- verify manifest exists
    2. git_push_results(ree_assembly_path, [...])  -- ship to REE_assembly
    3. coordinator_client.report_result(...)       -- ship bytes to coord
  V3-EXQ-592b FAILed on the runner with the sentinel-named manifest
  absent on disk. The FAIL branch went straight from completed-list
  append to report_queue_remove. The coordinator `results` table row
  stayed empty (queue_id committed_at NULL with no manifest_bytes), the
  spool was empty, and no manifest commit landed on REE_assembly
  origin/master. Same root-cause class as the line-1394 UNKNOWN-drop
  bug that was retrofitted to the PASS branch on 2026-05-08.

Fix:
  - FAIL: enforce the manifest-existence check; on missing-manifest,
          WARN + release_active_claim + _pass_skip + continue (matches
          PASS pattern at line ~2331); on present-manifest, run
          git_push_results + coordinator_client.report_result BEFORE
          report_queue_remove.
  - ERROR: enforce the same contract WHEN output_file is non-empty
           (the script claimed a manifest but did not deliver). ERROR
           with empty output_file is a normal script crash and the
           pre-fix flow proceeds to queue removal.

Contracts (source-inspection over experiment_runner.py main loop):
  C1. FAIL branch invokes the _result_manifest_exists guard with a
      missing-manifest skip path (release_active_claim + _pass_skip.add
      + continue, matching the PASS pattern).
  C2. FAIL branch invokes git_push_results + coordinator_client.report_result
      BEFORE coordinator_client.report_queue_remove on the present-
      manifest path.
  C3. ERROR branch invokes _result_manifest_exists guard, gated on
      output_file non-empty (preserves legitimacy of crash-with-no-
      manifest path which does not exercise the manifest contract).
  C4. ERROR branch invokes git_push_results + coordinator_client.report_result
      BEFORE coordinator_client.report_queue_remove on the present-
      manifest path, gated on output_file non-empty.

These are source-text contracts rather than behavioural tests because
the FAIL/ERROR branches live inline in main()'s while-True loop and are
not exposed via an extracted helper. The contract surface that V3-EXQ-
592b lost -- "FAIL/ERROR with a missing manifest must NOT silently
remove the queue entry, and FAIL/ERROR with a present manifest must
ship it before queue removal" -- maps directly to the presence and
ordering of these calls inside the right branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402


_RUNNER_SRC = Path(experiment_runner.__file__).read_text(encoding="utf-8")


def _slice(src: str, start_marker: str, end_marker: str) -> str:
    """Return the substring from start_marker (inclusive) to end_marker
    (exclusive). Raises AssertionError if markers are not found in order."""
    s = src.find(start_marker)
    assert s != -1, f"start_marker not found in source: {start_marker!r}"
    e = src.find(end_marker, s + len(start_marker))
    assert e != -1, f"end_marker not found after start in source: {end_marker!r}"
    return src[s:e]


# Branch slices. Order: ERROR runs first (at ~line 2229), then FAIL (at
# ~line 2267), then the UNKNOWN guard (at ~line 2313). End each branch
# at the next outer guard's opening line so the slice contains exactly
# the branch body (and trailing `continue`).
_ERROR_BRANCH = _slice(
    _RUNNER_SRC,
    'if result["result"] == "ERROR":',
    'if result["result"] == "FAIL":',
)
_FAIL_BRANCH = _slice(
    _RUNNER_SRC,
    'if result["result"] == "FAIL":',
    'if result["result"] == "UNKNOWN":',
)


def _idx(branch: str, needle: str, label: str) -> int:
    pos = branch.find(needle)
    assert pos != -1, (
        f"{label}: expected substring not found in branch.\n"
        f"needle: {needle!r}\n"
        f"branch (first 400 chars):\n{branch[:400]}"
    )
    return pos


# ---------------------------------------------------------------------------
# C1. FAIL branch -- manifest guard + missing-manifest skip semantic.
# ---------------------------------------------------------------------------
def test_c1_fail_branch_has_manifest_existence_guard_with_skip_semantic():
    """FAIL branch must call _result_manifest_exists, and the missing-
    manifest path must release the claim, _pass_skip.add(queue_id), and
    continue -- WITHOUT removing the queue entry. Matches the PASS-branch
    pattern at experiment_runner.py:2331-2341."""
    guard_idx = _idx(
        _FAIL_BRANCH,
        "if not _result_manifest_exists(result):",
        "C1 FAIL branch manifest guard",
    )
    # All three skip-path calls must follow the guard and precede the
    # next major action (git_push_results or queue file edit), and must
    # all sit before report_queue_remove which lives later in the branch.
    release_idx = _idx(
        _FAIL_BRANCH,
        "release_active_claim(QUEUE_FILE, queue_id, machine)",
        "C1 FAIL branch missing-manifest release_active_claim",
    )
    pass_skip_idx = _idx(
        _FAIL_BRANCH,
        "_pass_skip.add(queue_id)",
        "C1 FAIL branch missing-manifest _pass_skip.add",
    )
    # The guard must come before its skip-path calls.
    assert guard_idx < release_idx < pass_skip_idx, (
        "C1: FAIL branch missing-manifest skip path must call "
        "release_active_claim then _pass_skip.add after the "
        "_result_manifest_exists guard."
    )
    # And a `continue` must exist between _pass_skip.add and the start
    # of the present-manifest ship-results block (git_push_results) so
    # the missing-manifest path leaves the queue file untouched.
    git_push_idx = _idx(
        _FAIL_BRANCH,
        'git_push_results(ree_assembly_path, [result["output_file"]])',
        "C1 FAIL branch git_push_results",
    )
    between = _FAIL_BRANCH[pass_skip_idx:git_push_idx]
    assert "continue" in between, (
        "C1: FAIL branch missing-manifest path must `continue` before "
        "reaching git_push_results / queue removal. Without the "
        "continue, the present-manifest ship-results block runs even "
        "when the manifest is absent."
    )


# ---------------------------------------------------------------------------
# C2. FAIL branch ships the manifest BEFORE report_queue_remove.
# ---------------------------------------------------------------------------
def test_c2_fail_branch_ships_manifest_before_queue_remove():
    """FAIL with a valid manifest must call git_push_results AND
    coordinator_client.report_result, and BOTH must precede
    coordinator_client.report_queue_remove (so the manifest reaches
    origin/master before the queue commit marks the experiment 'done')."""
    gp_idx = _idx(
        _FAIL_BRANCH,
        'git_push_results(ree_assembly_path, [result["output_file"]])',
        "C2 FAIL git_push_results",
    )
    rr_idx = _idx(
        _FAIL_BRANCH,
        "coordinator_client.report_result(",
        "C2 FAIL coordinator_client.report_result",
    )
    rqr_idx = _idx(
        _FAIL_BRANCH,
        'coordinator_client.report_queue_remove(queue_id, "FAIL")',
        "C2 FAIL coordinator_client.report_queue_remove",
    )
    assert gp_idx < rqr_idx, (
        "C2: FAIL branch must call git_push_results BEFORE "
        "report_queue_remove (the V3-EXQ-592b invariant: manifest "
        "ships before queue commits 'done')."
    )
    assert rr_idx < rqr_idx, (
        "C2: FAIL branch must call coordinator_client.report_result "
        "BEFORE report_queue_remove."
    )


# ---------------------------------------------------------------------------
# C3. ERROR branch -- manifest guard gated on output_file non-empty.
# ---------------------------------------------------------------------------
def test_c3_error_branch_manifest_guard_gated_on_output_file():
    """ERROR with output_file non-empty (script *claimed* a manifest)
    must enforce the manifest-existence contract. ERROR with no
    output_file is a legitimate script crash and must NOT fall into
    the WARN + leave-in-queue path -- it proceeds straight to queue
    removal as before the fix."""
    # The guard must be gated on output_file non-empty so a normal
    # crash-without-manifest ERROR still removes from queue. The fix
    # uses the local _err_manifest_str variable for this.
    gate_idx = _idx(
        _ERROR_BRANCH,
        "_err_manifest_str and not _result_manifest_exists(result)",
        "C3 ERROR manifest guard gated on _err_manifest_str non-empty",
    )
    release_idx = _idx(
        _ERROR_BRANCH,
        "release_active_claim(QUEUE_FILE, queue_id, machine)",
        "C3 ERROR missing-manifest release_active_claim",
    )
    pass_skip_idx = _idx(
        _ERROR_BRANCH,
        "_pass_skip.add(queue_id)",
        "C3 ERROR missing-manifest _pass_skip.add",
    )
    assert gate_idx < release_idx < pass_skip_idx, (
        "C3: ERROR branch missing-manifest skip path must call "
        "release_active_claim then _pass_skip.add after the gated "
        "manifest-existence guard."
    )
    # The `continue` separating the missing-manifest path from the
    # ship-results block.
    git_push_idx = _idx(
        _ERROR_BRANCH,
        'git_push_results(ree_assembly_path, [result["output_file"]])',
        "C3 ERROR git_push_results",
    )
    between = _ERROR_BRANCH[pass_skip_idx:git_push_idx]
    assert "continue" in between, (
        "C3: ERROR branch missing-manifest path must `continue` before "
        "reaching git_push_results / queue removal."
    )


# ---------------------------------------------------------------------------
# C4. ERROR branch ships the manifest BEFORE queue removal when present.
# ---------------------------------------------------------------------------
def test_c4_error_branch_ships_manifest_before_queue_remove_when_present():
    """ERROR with a present manifest must call git_push_results AND
    coordinator_client.report_result before report_queue_remove. The
    ship block is gated on output_file non-empty so an ERROR with no
    claimed manifest (legitimate crash) skips these calls and proceeds
    to queue removal directly."""
    # Gating clause for the ship block.
    assert "args.auto_sync and ree_assembly_path and _err_manifest_str" in _ERROR_BRANCH, (
        "C4: ERROR branch ship-results block must be gated on "
        "_err_manifest_str non-empty so an ERROR with no claimed "
        "manifest (script crash) still removes from queue without "
        "firing the manifest-ship calls."
    )
    gp_idx = _idx(
        _ERROR_BRANCH,
        'git_push_results(ree_assembly_path, [result["output_file"]])',
        "C4 ERROR git_push_results",
    )
    rr_idx = _idx(
        _ERROR_BRANCH,
        "coordinator_client.report_result(",
        "C4 ERROR coordinator_client.report_result",
    )
    rqr_idx = _idx(
        _ERROR_BRANCH,
        'coordinator_client.report_queue_remove(queue_id, "ERROR")',
        "C4 ERROR coordinator_client.report_queue_remove",
    )
    assert gp_idx < rqr_idx, (
        "C4: ERROR branch must call git_push_results BEFORE "
        "report_queue_remove (the V3-EXQ-592b invariant for ERROR with "
        "a claimed manifest)."
    )
    assert rr_idx < rqr_idx, (
        "C4: ERROR branch must call coordinator_client.report_result "
        "BEFORE report_queue_remove."
    )
