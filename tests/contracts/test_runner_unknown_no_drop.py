"""Contract tests for the UNKNOWN-result no-silent-drop guard.

Background -- the "line 1394" UNKNOWN silent-drop bug (pre-2026-05-08):
  `result_info["result"]` is initialised to "UNKNOWN"
  (experiment_runner.py ~:1889). Before commit f36461d (2026-05-08
  "runner: sentinel-file conformance contract replacing stdout regex
  scraping"), outcome was scraped from subprocess stdout via a set of
  regexes. When a script printed a verdict in a format none of the
  regexes matched (cloud scripts emitting `**Status:** FAIL` markdown, or
  `[EXQ-056] PASS (5/5 criteria)`), the result stayed "UNKNOWN" and the
  fall-through let UNKNOWN reach the queue-removal block -- the item was
  committed as "done" and dropped from the queue with no manifest. This
  produced 193 UNKNOWN rows in runner_status.json (latest 2026-05-08),
  all carrying their missed verdict in `result_summary`.

Fix (f36461d, 2026-05-08): the sentinel file is authoritative; a missing
  sentinel with no stdout verdict is classified ERROR; and an explicit
  guard ensures a residual UNKNOWN can NEVER reach the success / queue-
  removal block. The UNKNOWN guard logs loudly, releases the claim,
  leaves the item in the queue (_pass_skip), writes NO completed row, and
  continues.

The sibling tests (test_runner_fail_branch_persists_result.py,
test_runner_sigterm_no_phantom_completion.py) reference the UNKNOWN guard
only as a slice boundary; none asserts the UNKNOWN branch's own behavior.
This file closes that coverage gap.

Contracts (source-inspection over experiment_runner.py main loop):
  C1. The UNKNOWN guard appears BEFORE the PASS/manifest-verify path
      (so UNKNOWN can never fall through to queue removal).
  C2. The UNKNOWN branch releases the claim (release_active_claim) and
      marks _pass_skip.add(queue_id) -- leaving the item claimable on the
      next pass / by another machine.
  C3. The UNKNOWN branch writes NO completed row: it does NOT call
      status["completed"].append(...) on its path.
  C4. The UNKNOWN branch ends in `continue` (no queue-file edit, no
      git_push_queue, no report_queue_remove on its path).

These are source-text contracts rather than behavioural tests because the
UNKNOWN branch lives inline in main()'s while-True loop and is not exposed
via an extracted helper -- matching the established pattern in
test_runner_fail_branch_persists_result.py.
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


# The UNKNOWN branch runs from its guard to the PASS-path manifest-verify
# comment that immediately follows it.
_UNKNOWN_BRANCH = _slice(
    _RUNNER_SRC,
    'if result["result"] == "UNKNOWN":',
    "# Verify the manifest exists before declaring done.",
)


def _idx(branch: str, needle: str, label: str) -> int:
    pos = branch.find(needle)
    assert pos != -1, (
        f"{label}: expected substring not found in UNKNOWN branch.\n"
        f"needle: {needle!r}\n"
        f"branch (first 600 chars):\n{branch[:600]}"
    )
    return pos


# ---------------------------------------------------------------------------
# C1. UNKNOWN guard precedes the PASS / manifest-verify path.
# ---------------------------------------------------------------------------
def test_c1_unknown_guard_precedes_success_path():
    """The UNKNOWN guard must appear before the PASS-path
    `_result_manifest_exists` verify block. If the guard were after it,
    an UNKNOWN result could reach queue removal -- the original line-1394
    leak. The slice itself (guard -> manifest-verify comment) is the
    proof of ordering; this test makes the boundary explicit and fails
    loudly if the branch is ever reordered."""
    assert _UNKNOWN_BRANCH.startswith('if result["result"] == "UNKNOWN":'), (
        "C1: UNKNOWN branch slice must begin at the UNKNOWN guard."
    )
    # The PASS-path manifest verify (the slice's end marker) must exist
    # after the UNKNOWN branch in the full source.
    guard_pos = _RUNNER_SRC.find('if result["result"] == "UNKNOWN":')
    verify_pos = _RUNNER_SRC.find(
        "# Verify the manifest exists before declaring done.", guard_pos
    )
    assert guard_pos != -1 and verify_pos != -1 and guard_pos < verify_pos, (
        "C1: UNKNOWN guard must precede the PASS-path manifest-verify "
        "block so UNKNOWN can never fall through to queue removal."
    )


# ---------------------------------------------------------------------------
# C2. UNKNOWN branch releases the claim and skips the item this pass.
# ---------------------------------------------------------------------------
def test_c2_unknown_branch_releases_claim_and_skips():
    """UNKNOWN must release_active_claim and _pass_skip.add(queue_id) so
    the item stays claimable (by the next pass or another machine) rather
    than being committed as done."""
    release_idx = _idx(
        _UNKNOWN_BRANCH,
        "release_active_claim(QUEUE_FILE, queue_id, machine)",
        "C2 UNKNOWN release_active_claim",
    )
    pass_skip_idx = _idx(
        _UNKNOWN_BRANCH,
        "_pass_skip.add(queue_id)",
        "C2 UNKNOWN _pass_skip.add",
    )
    assert release_idx < pass_skip_idx, (
        "C2: UNKNOWN branch must release the claim before marking "
        "_pass_skip.add(queue_id)."
    )


# ---------------------------------------------------------------------------
# C3. UNKNOWN branch writes NO completed row.
# ---------------------------------------------------------------------------
def test_c3_unknown_branch_writes_no_completion():
    """The UNKNOWN branch must NOT append to status["completed"]. Writing
    a completed row is exactly what the silent-drop did -- it committed an
    UNKNOWN as a finished experiment. Leaving the item in the queue with
    no completion row is the fix."""
    assert 'status["completed"].append' not in _UNKNOWN_BRANCH, (
        "C3: UNKNOWN branch must NOT call status['completed'].append -- a "
        "completed row commits the experiment as 'done'. The guard must "
        "leave the item in the queue with no completion written."
    )


# ---------------------------------------------------------------------------
# C4. UNKNOWN branch ends in `continue` with no queue mutation on its path.
# ---------------------------------------------------------------------------
def test_c4_unknown_branch_continues_without_queue_mutation():
    """The UNKNOWN branch must `continue` and must not remove the item
    from the queue file or report a queue removal on its path."""
    assert "continue" in _UNKNOWN_BRANCH, (
        "C4: UNKNOWN branch must `continue` so it does not fall through to "
        "the success / queue-removal path."
    )
    # No queue-file mutation or queue-remove report on the UNKNOWN path.
    assert "_atomic_write_queue(QUEUE_FILE" not in _UNKNOWN_BRANCH, (
        "C4: UNKNOWN branch must not rewrite the queue file (no item "
        "removal)."
    )
    assert "report_queue_remove" not in _UNKNOWN_BRANCH, (
        "C4: UNKNOWN branch must not report a queue removal -- the item "
        "is being left in the queue, not committed as done."
    )
