"""Contract tests for the SIGTERM-mid-experiment phantom-completion fix.

Background -- 2026-05-30 fleet incident:
  The cloud-scaler workflow killed ree-cloud-2 and ree-cloud-3 mid-experiment
  via systemd SIGTERM while they held active claims (V3-EXQ-483e and
  V3-EXQ-490i). Each worker's subprocess returned exit_code=-15 (SIGTERM
  delivered directly to the subprocess). The runner main loop's infra-crash
  early-exit block (lines ~2197-2227) recognised exit codes {137, -9, -11}
  (SIGKILL/OOM, SIGKILL direct, SIGSEGV) but NOT -15 / 143 (SIGTERM). The
  SIGTERM cases fell through to the ERROR branch and wrote a phantom
  completion row to runner_status/<host>.json with no manifest. On the
  next runner boot, validate_queue.py preflight detected the phantom
  completion record for V3-EXQ-483e / V3-EXQ-490i and blocked startup with
  the silent-skip-guard error -- the workers wedged in restart-rate-limit
  and required manual SSH cleanup (edit out the phantom rows from each
  worker's status file + systemctl reset-failed).

  Sibling fix: cloud-scaler.yml is being patched (separate session) so the
  scaler doesn't power off workers holding active claims. This test covers
  the runner-side cleanup contract: even when something does send SIGTERM
  mid-experiment (cloud-scaler bug, manual kill, systemd shutdown), the
  runner must not leave a phantom completion behind.

Fix (Option A in the spawning prompt):
  Add -15 (SIGTERM as Python subprocess.returncode) and 143 (shell-wrapper
  positive form, 128 + 15) to the existing _transient_exit_codes set at
  the top of the result-processing block in experiment_runner.main(). The
  existing infra-crash early-exit path already does exactly what Option A
  requires:
    - release_active_claim(QUEUE_FILE, queue_id, machine)   [coordinator + git path]
    - _pass_skip.add(queue_id)                               [don't re-pick this pass]
    - status["current"] = None + write_status               [status without completion entry]
    - continue                                                [skip the ERROR/FAIL/PASS branches]
  No completion row is appended to status["completed"], so preflight will
  not block the next runner boot.

Contracts (source-inspection over experiment_runner.py main loop, matching
the pattern used by test_runner_fail_branch_persists_result.py):

  C1. _transient_exit_codes set contains -15 (SIGTERM as Python negative
      returncode -- the form ree-cloud-2 / ree-cloud-3 hit on 2026-05-30).

  C2. _transient_exit_codes set contains 143 (SIGTERM as shell-wrapper
      positive form 128 + 15; appears when subprocess wrapped by sh -c or
      certain systemd-managed paths).

  C3. The infra-crash block's release_active_claim call is gated on
      args.auto_sync (Phase 3 coordinator path) -- this is the existing
      semantic and must not regress, otherwise SIGTERM in a no-auto-sync
      run would leave the claim coordinator-held with no cleanup.

  C4. The infra-crash block does NOT call status["completed"].append on
      its path. Source-text contract: between the `if _is_infra_crash:`
      line and the `continue` that closes the branch, there is no
      `status["completed"].append(`. (This is what makes "no phantom
      completion written" structurally true regardless of result["result"].)

  C5. The ERROR branch's completion-append (line ~2247) is NOT the next
      branch reached by SIGTERM. Source-text ordering contract: the
      _is_infra_crash early-exit appears BEFORE the `if result["result"] ==
      "ERROR":` branch in the source. Verifies the intercept order.

These are source-text contracts rather than behavioural tests because the
result-processing block lives inline in main()'s while-True loop. The
contract surface that the 2026-05-30 incident lost -- "SIGTERM mid-
experiment must not leave a completion record behind" -- maps directly to
membership in _transient_exit_codes plus the structural absence of a
completion append on the intercept path.

See also:
  - test_runner_fail_branch_persists_result.py: the 2026-05-29 fix that
    added the manifest-exists gate to FAIL/ERROR branches. That fix didn't
    cover SIGTERM because the ERROR branch's manifest check is gated on
    `_err_manifest_str` non-empty -- a SIGTERM-killed subprocess has
    output_file="" (script died before emit_outcome wrote the sentinel),
    so the gate is skipped and the phantom write happens anyway. The
    SIGTERM fix intercepts EARLIER, before the ERROR branch is reached.
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


# Slice the infra-crash block: from the _transient_exit_codes assignment
# to the `if result["result"] == "ERROR":` line that opens the next branch.
_INFRA_CRASH_BLOCK = _slice(
    _RUNNER_SRC,
    "_transient_exit_codes = {",
    'if result["result"] == "ERROR":',
)


def test_c1_sigterm_negative_in_transient_set():
    """C1: -15 (Python subprocess SIGTERM returncode) is in _transient_exit_codes."""
    set_line = _INFRA_CRASH_BLOCK.split("\n", 1)[0]
    assert "-15" in set_line, (
        f"_transient_exit_codes must include -15 (SIGTERM as Python negative "
        f"returncode -- the form ree-cloud-2/3 hit on 2026-05-30); got: {set_line!r}"
    )


def test_c2_sigterm_shell_form_in_transient_set():
    """C2: 143 (shell-wrapper SIGTERM, 128 + 15) is in _transient_exit_codes."""
    set_line = _INFRA_CRASH_BLOCK.split("\n", 1)[0]
    # Word-boundary check: "143" must not be a substring of "1430" etc.
    # Cheap approach: look for "143" surrounded by non-digit chars.
    import re
    assert re.search(r"(?<!\d)143(?!\d)", set_line), (
        f"_transient_exit_codes must include 143 (SIGTERM shell-wrapper "
        f"positive form, 128 + 15); got: {set_line!r}"
    )


def test_c3_release_claim_gated_on_auto_sync():
    """C3: release_active_claim is gated on args.auto_sync on the infra-crash path."""
    # The release_active_claim call must be inside an `if args.auto_sync:` guard.
    # Allow whitespace and other lines between them; check the call doesn't appear
    # as a top-of-block unconditional.
    assert "release_active_claim(QUEUE_FILE, queue_id, machine)" in _INFRA_CRASH_BLOCK, (
        "infra-crash block must call release_active_claim to release the claim "
        "back to the coordinator/queue"
    )
    # Find the line index of release_active_claim and walk back to nearest
    # non-whitespace control structure; assert it's an auto_sync guard.
    lines = _INFRA_CRASH_BLOCK.split("\n")
    release_idx = next(
        i for i, ln in enumerate(lines)
        if "release_active_claim(QUEUE_FILE" in ln
    )
    # Walk back to find the controlling `if`.
    for j in range(release_idx - 1, -1, -1):
        stripped = lines[j].strip()
        if stripped.startswith("if "):
            assert "args.auto_sync" in stripped, (
                f"release_active_claim on infra-crash path must be gated on "
                f"args.auto_sync; controlling guard is: {stripped!r}"
            )
            break
    else:
        raise AssertionError(
            "could not find controlling `if` for release_active_claim on "
            "infra-crash path"
        )


def test_c4_no_completion_append_on_infra_crash_path():
    """C4: the infra-crash early-exit block does NOT append to status['completed'].

    This is the structural guarantee that SIGTERM does not write a phantom
    completion. Whatever the runner does on the SIGTERM intercept path, it
    must NOT add a row to status['completed'] -- otherwise preflight on the
    next boot will block startup with the silent-skip-guard error.
    """
    # The branch body is `if _is_infra_crash: ... continue`. Take that whole body.
    body = _slice(
        _RUNNER_SRC,
        "if _is_infra_crash:",
        "if result[\"result\"] == \"ERROR\":",
    )
    assert 'status["completed"].append(' not in body, (
        "infra-crash early-exit block must NOT append to status['completed'] -- "
        "doing so would leave a phantom completion row that trips preflight on "
        "next boot (the 2026-05-30 ree-cloud-2/3 wedge symptom). Body was:\n"
        + body
    )


def test_c5_infra_crash_intercept_precedes_error_branch():
    """C5: the infra-crash intercept appears in source BEFORE the ERROR branch.

    Source ordering matters because the result-processing block is a series
    of `if result['result'] == ...:` checks followed by branch bodies that
    each end in `continue`. The infra-crash intercept must run FIRST so
    SIGTERM never reaches the ERROR branch's `status['completed'].append`
    at line ~2247.
    """
    infra_idx = _RUNNER_SRC.find("if _is_infra_crash:")
    error_idx = _RUNNER_SRC.find('if result["result"] == "ERROR":')
    assert infra_idx != -1, "if _is_infra_crash: marker not found in runner source"
    assert error_idx != -1, 'if result["result"] == "ERROR": marker not found in runner source'
    assert infra_idx < error_idx, (
        f"infra-crash intercept (idx {infra_idx}) must appear BEFORE the ERROR "
        f"branch (idx {error_idx}) in source order -- otherwise SIGTERM falls "
        f"through to the ERROR completion-append at line ~2247"
    )
