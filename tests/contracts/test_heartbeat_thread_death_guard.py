"""Contract tests for the silent-death guard on the runner's `_heartbeat`
daemon thread and on `update_status_current` (experiment_runner).

Background -- the sibling of the defect fixed by ree-v3 3dcb3ccd2d.

`run_experiment()` spawns TWO daemon threads off the same `_hb_stop` event:
`_background_sync` (--auto-sync pulls) and `_heartbeat` (status + telemetry).
3dcb3ccd2d found `_background_sync`'s loop body unguarded, so one escaped
exception ended --auto-sync for the remainder of the experiment -- silently,
and permanently, since nothing restarts the thread. It fixed that one and
recorded the sibling as unexamined. It was the same defect:

    def _heartbeat():
        while not _hb_stop.wait(timeout=STATUS_WRITE_INTERVAL):
            update_status_current()      # <-- NOT internally wrapped
            _push_remote_heartbeat()     # <-- internally wrapped

`_push_remote_heartbeat` carried its own blanket `except Exception`, so it
was total. `update_status_current` was not: it builds the status dict and
ends in `write_status(status, status_path)`, a disk write. A transient IO
error there -- full disk, permission change, a locked or truncated status
file -- propagated out of `update_status_current`, out of the loop (which had
no try/except of its own), and terminated the heartbeat thread for the rest
of the run.

Blast radius, and why this is not cosmetic: `runner_status/*.json` stops
updating and the per-machine remote-control heartbeat stops being written and
pushed, so the explorer /machines dashboard and the cloud-scaler's telemetry
both go stale for that machine. The scaler's HEARTBEAT_FRESH_MIN git-fallback
path reads a stale heartbeat as IDLE.

Fix (2026-07-29), in two layers, mirroring 3dcb3ccd2d:
  (a) `update_status_current` is made total at its own definition -- its body
      moved to `_update_status_current_inner` and the wrapper swallows
      Exception, logging via the module-level `_on_status_write_error`.
      Chosen over guarding only the thread because the function has three
      callers and the OTHER two failure directions were worse (C13/C14).
  (b) `_heartbeat`'s loop body is wrapped per-iteration, as defence in depth,
      so a future addition to the body degrades to one skipped tick instead
      of terminating the thread.

Contracts:
  C11.  `_heartbeat`'s loop body is wrapped in try/except Exception, with
        BOTH calls inside the try. Structural, because the closure is nested
        inside run_experiment() and is not reachable for monkeypatching --
        the same constraint C9 works around in
        test_background_sync_claim_guard.py.
  C11b. The extracted loop source, COMPILED AND EXECUTED against a raising
        stand-in, actually survives -- an execution-level proof that the AST
        shape C11 pins does what it claims.
  C12.  `update_status_current` is total at its own definition, and its
        handler catches Exception (not bare), so SystemExit /
        KeyboardInterrupt can still stop the thread. Structural + executed.
  C13.  Regression guard for the pre-launch caller, which sits OUTSIDE
        run_experiment's big try: its failure direction was a BURNED queue
        entry, not a stale status file.
  C14.  Regression guard for the stdout-pump caller: its failure direction
        was a healthy run recorded ERROR with an orphaned subprocess.
  C15.  `_on_status_write_error` logs the first occurrence then at most
        hourly, and is rate-limited on the CLOCK rather than a tick count.

NEGATIVE CONTROLS. Every structural assertion here is re-run against the
PRE-FIX source and must FAIL there. Without that, an AST assertion phrased
slightly wrong passes vacuously on both revisions and pins nothing -- that
check is what validated the C9/C10 contracts this file is modelled on. The
controls share the predicate FUNCTION with the live-source assertions rather
than re-stating it, since a control that re-words the predicate proves
nothing about the predicate actually used.

The pre-fix source is EMBEDDED below (PRE_FIX_HEARTBEAT /
PRE_FIX_UPDATE_STATUS_CURRENT) rather than read from git, because
`scripts/remote_pytest.sh` rsyncs the tree WITHOUT `.git/` -- a git-backed
control would silently skip on precisely the machine the suite is supposed
to run on, leaving the vacuity check unexecuted while the file still
reported green. The snapshots are frozen at a specific commit and can never
drift; `test_negative_control_snapshots_match_git` re-derives them from git
when `.git/` happens to be present, so a transcription error is still
caught, and skips when it is not.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import experiment_runner  # noqa: E402


# The commit that fixed `_background_sync` and left `_heartbeat` unexamined.
# Its `experiment_runner.py` is the last revision carrying the pre-fix
# `_heartbeat`, and is therefore this file's negative control.
PRE_FIX_REF = "3dcb3ccd2d"

# Verbatim from `git show 3dcb3ccd2d:experiment_runner.py`, dedented (both are
# nested inside run_experiment there). Frozen history -- these cannot drift.
PRE_FIX_HEARTBEAT = '''\
def _heartbeat():
    while not _hb_stop.wait(timeout=STATUS_WRITE_INTERVAL):
        update_status_current()
        _push_remote_heartbeat()
'''

PRE_FIX_UPDATE_STATUS_CURRENT = '''\
def update_status_current():
    status["current"] = {
        "queue_id": item["queue_id"],
        "backlog_id": item.get("backlog_id", ""),
        "claim_id": item.get("claim_id", ""),
        "title": item.get("title", ""),
        "description": item.get("description", ""),
        "script": item["script"],
        "started_at": started_at_utc,
        "progress": {
            "run_label": current_run_label,
            "runs_done": runs_done,
            "runs_total": total_runs,
            "episodes_done": episodes_in_run,
            "episodes_total": episodes_per_run,
            "overall_pct": overall_pct(),
        },
        "seconds_elapsed": round(time.monotonic() - started_at),
        "seconds_remaining": round(seconds_remaining()),
        "recent_lines": recent_lines[-RECENT_LINES_DISPLAY:],
        "ree_version": "v3",
    }
    for qi in status["queue"]:
        if qi["queue_id"] == item["queue_id"]:
            qi["status"] = "running"
    write_status(status, status_path)
'''


# --------------------------------------------------------------------------
# Source access helpers.
# --------------------------------------------------------------------------


def _live_source() -> str:
    return Path(experiment_runner.__file__).read_text(encoding="utf-8")


def _find_fn(src: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(
        f"{name} not found in experiment_runner.py -- if it was renamed, "
        "re-point this contract rather than deleting it."
    )


# --------------------------------------------------------------------------
# The structural predicates, factored so the SAME code runs against the live
# source and against the pre-fix source. A negative control that re-states
# the predicate in its own words proves nothing about the predicate actually
# used; sharing the function is what makes the control binding.
# --------------------------------------------------------------------------


def _heartbeat_loop_is_guarded(src: str) -> tuple[bool, str]:
    """(guarded, why-not). Both loop-body calls inside one try/except Exception."""
    fn = _find_fn(src, "_heartbeat")
    whiles = [n for n in fn.body if isinstance(n, ast.While)]
    if not whiles:
        return False, "_heartbeat has no while loop"
    tries = [n for n in whiles[0].body if isinstance(n, ast.Try)]
    if not tries:
        return False, "loop body contains no try/except"
    try_node = tries[0]

    called = {
        n.func.id for n in ast.walk(try_node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    for required in ("update_status_current", "_push_remote_heartbeat"):
        if required not in called:
            return False, f"{required}() is not INSIDE the try"

    if not any(
        h.type is not None and isinstance(h.type, ast.Name) and h.type.id == "Exception"
        for h in try_node.handlers
    ):
        return False, "handler does not catch Exception specifically"
    return True, ""


def _update_status_current_is_total(src: str) -> tuple[bool, str]:
    """(total, why-not). Whole body wrapped, handler catches Exception."""
    fn = _find_fn(src, "update_status_current")
    body = [n for n in fn.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str)
    )]  # drop the docstring
    if len(body) != 1 or not isinstance(body[0], ast.Try):
        kinds = [type(n).__name__ for n in body]
        return False, (
            "body is not a single Try -- statements outside it can still "
            f"escape (found {kinds})"
        )
    if not any(
        h.type is not None and isinstance(h.type, ast.Name) and h.type.id == "Exception"
        for h in body[0].handlers
    ):
        return False, "handler does not catch Exception specifically"
    return True, ""


# --------------------------------------------------------------------------
# C11 -- _heartbeat's loop body is guarded.
# --------------------------------------------------------------------------


def test_c11_heartbeat_loop_body_is_exception_guarded():
    """One escaped exception must cost one tick, not every remaining tick.

    The thread is never restarted and its death is silent, so an unguarded
    loop turns a transient IO error into permanently stale telemetry for the
    rest of the experiment -- which the cloud-scaler's git-fallback path
    reads as an idle machine.
    """
    guarded, why = _heartbeat_loop_is_guarded(_live_source())
    assert guarded, f"C11 FAIL: {why}."


def test_c11_negative_control_pre_fix_source_is_unguarded():
    """The predicate must FAIL on pre-fix source, or it pins nothing."""
    guarded, why = _heartbeat_loop_is_guarded(PRE_FIX_HEARTBEAT)
    assert not guarded, (
        "C11 NEGATIVE CONTROL FAIL: the pre-fix _heartbeat at "
        f"{PRE_FIX_REF} passes the same predicate the live source is "
        "asserted against, so C11 proves nothing. Fix the predicate."
    )
    assert why == "loop body contains no try/except", (
        f"C11 NEGATIVE CONTROL: pre-fix source failed for an UNEXPECTED "
        f"reason ({why!r}). Expected the absent try/except -- if the reason "
        "changed, the control is no longer testing the documented defect."
    )


def test_c11b_extracted_loop_survives_a_raising_call():
    """Execute the real loop source: the AST shape must actually swallow.

    C11 pins a shape; this runs it. The closure's free variables become
    globals in the exec namespace, so the loop body executes verbatim without
    needing run_experiment() to be entered.
    """
    fn = _find_fn(_live_source(), "_heartbeat")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)

    ticks = {"n": 0}

    class _StopAfter:
        """Stand-in for _hb_stop: False (keep looping) then True (stop)."""
        def wait(self, timeout=None):
            ticks["n"] += 1
            return ticks["n"] > 3

    def _boom():
        raise OSError(28, "No space left on device")

    logged: list[str] = []
    ns = {
        "_hb_stop": _StopAfter(),
        "STATUS_WRITE_INTERVAL": 0,
        "update_status_current": _boom,
        "_push_remote_heartbeat": lambda: None,
        "print": lambda *a, **k: logged.append(" ".join(str(x) for x in a)),
    }
    exec(compile(module, "<_heartbeat>", "exec"), ns)

    ns["_heartbeat"]()  # must return normally, not raise

    assert ticks["n"] == 4, (
        f"C11b FAIL: the loop ran {ticks['n']} time(s) -- it must survive the "
        "raising call and keep ticking until _hb_stop is set."
    )
    assert logged, (
        "C11b FAIL: the failure must be LOGGED, not silently swallowed -- "
        "the _on_git_timeout lesson."
    )


def test_c11b_negative_control_pre_fix_loop_dies():
    """The same execution harness against pre-fix source must let it escape."""
    fn = _find_fn(PRE_FIX_HEARTBEAT, "_heartbeat")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)

    class _NeverStop:
        def wait(self, timeout=None):
            return False

    def _boom():
        raise OSError(28, "No space left on device")

    ns = {
        "_hb_stop": _NeverStop(),
        "STATUS_WRITE_INTERVAL": 0,
        "update_status_current": _boom,
        "_push_remote_heartbeat": lambda: None,
        "print": lambda *a, **k: None,
    }
    exec(compile(module, "<_heartbeat_prefix>", "exec"), ns)

    with pytest.raises(OSError):
        ns["_heartbeat"]()  # pre-fix: the exception ends the thread


# --------------------------------------------------------------------------
# C12 -- update_status_current is total at its own definition.
# --------------------------------------------------------------------------


def test_c12_update_status_current_is_total():
    """Made total at the DEFINITION, not only at the thread's call site.

    It has three callers and the thread is only one of them; see C13/C14 for
    why the other two mattered more than the thread did.
    """
    total, why = _update_status_current_is_total(_live_source())
    assert total, f"C12 FAIL: {why}."


def test_c12_negative_control_pre_fix_is_not_total():
    total, _why = _update_status_current_is_total(PRE_FIX_UPDATE_STATUS_CURRENT)
    assert not total, (
        "C12 NEGATIVE CONTROL FAIL: pre-fix update_status_current passes the "
        "totality predicate, so C12 proves nothing."
    )


@pytest.mark.parametrize(
    "name, snapshot",
    [
        ("_heartbeat", PRE_FIX_HEARTBEAT),
        ("update_status_current", PRE_FIX_UPDATE_STATUS_CURRENT),
    ],
)
def test_negative_control_snapshots_match_git(name, snapshot):
    """The embedded pre-fix snapshots must be faithful transcriptions.

    Skips when `.git/` is unavailable, which is the normal case under
    `scripts/remote_pytest.sh` (its rsync excludes .git/). That is exactly
    why the controls above are embedded rather than git-backed: this check
    is a transcription audit that runs opportunistically, NOT the negative
    control itself. A negative control that can only run on the developer's
    laptop is not a control.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show",
             f"{PRE_FIX_REF}:experiment_runner.py"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git unavailable for snapshot audit: {exc!r}")
    if out.returncode != 0:
        pytest.skip(
            f"git show {PRE_FIX_REF} unavailable "
            f"({out.stderr.strip()[:200]}) -- staged tree excludes .git/"
        )

    lines = out.stdout.splitlines()
    fn = _find_fn(out.stdout, name)
    real = textwrap.dedent("\n".join(lines[fn.lineno - 1:fn.end_lineno])).rstrip()
    assert real == snapshot.rstrip(), (
        f"SNAPSHOT DRIFT: the embedded pre-fix {name} does not match "
        f"{PRE_FIX_REF}. The negative controls are testing text that never "
        "existed. Re-transcribe from git rather than editing the assertion."
    )


def test_c12b_wrapper_swallows_and_reports():
    """Execute the wrapper: a raising inner must be swallowed AND reported."""
    fn = _find_fn(_live_source(), "update_status_current")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)

    seen: list[BaseException] = []

    def _boom():
        raise PermissionError(13, "Permission denied")

    ns = {
        "_update_status_current_inner": _boom,
        "_on_status_write_error": seen.append,
    }
    exec(compile(module, "<update_status_current>", "exec"), ns)

    ns["update_status_current"]()  # must not raise

    assert len(seen) == 1 and isinstance(seen[0], PermissionError), (
        "C12b FAIL: the wrapper must route the exception to "
        f"_on_status_write_error, not absorb it silently (saw {seen!r})."
    )


def test_c12c_write_status_is_still_reached_on_the_happy_path():
    """Totality must not have severed the write.

    A wrapper that swallows everything trivially satisfies C12 while writing
    nothing at all, so pin that the inner function still ends in a
    write_status call. Otherwise the fix could silently disable the very
    telemetry it exists to keep alive.
    """
    inner = _find_fn(_live_source(), "_update_status_current_inner")
    calls = [
        n.func.id for n in ast.walk(inner)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "write_status" in calls, (
        "C12c FAIL: _update_status_current_inner no longer calls "
        "write_status. The guard is supposed to make the write survivable, "
        "not optional."
    )


# --------------------------------------------------------------------------
# C13/C14 -- the two NON-thread callers, whose failure directions were worse
# than the thread's and are the reason the fix went at the definition.
# --------------------------------------------------------------------------


def _call_lines(src: str, name: str) -> list[int]:
    fn = _find_fn(src, "run_experiment")
    return sorted(
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == name
    )


def _enclosing_tries(src: str, lineno: int) -> list[ast.Try]:
    fn = _find_fn(src, "run_experiment")
    return [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Try) and n.lineno <= lineno <= n.end_lineno
    ]


def test_c13_prelaunch_caller_is_still_outside_the_big_try():
    """Documents WHY totality had to go at the definition, caller 1.

    The pre-launch `update_status_current()` sits OUTSIDE run_experiment's
    big try, so pre-fix an IO error escaped the function entirely into
    main()'s `except Exception as _run_exc` -- which adds the queue_id to
    `completed_ids` and drops it from the queue. A status-file hiccup BURNED
    the queue entry without the experiment ever being launched.

    This asserts the structural fact the reasoning rests on. If a later
    refactor moves that call inside the try, this fails and the rationale
    above should be re-read rather than the assertion relaxed.
    """
    src = _live_source()
    lines = _call_lines(src, "update_status_current")
    assert lines, "C13 FAIL: no update_status_current() calls found."
    unguarded = [ln for ln in lines if not _enclosing_tries(src, ln)]
    assert unguarded, (
        "C13 FAIL: expected at least one update_status_current() call "
        "outside run_experiment's try -- the pre-launch one. If it moved "
        "inside, the C13 rationale needs re-reading, not this line deleting."
    )


def test_c14_all_three_callers_still_exist():
    """Documents WHY totality had to go at the definition, caller 2 (+ count).

    Caller 2 is the stdout pump. Pre-fix, an IO error there aborted the
    `for line in proc.stdout` loop: the run was recorded ERROR with an IO
    message as its result_summary, `_hb_stop.set()` and `proc.wait()` were
    both skipped, and the `finally` cleared `proc_ref` -- orphaning a
    still-live subprocess the SIGTERM handler could no longer kill.

    Guarding only the heartbeat thread would have left both that and C13
    unfixed, which is the whole argument for fixing the definition. Pinning
    the caller count is what makes a NEW, unconsidered caller visible.
    """
    n = len(_call_lines(_live_source(), "update_status_current"))
    assert n == 3, (
        f"C14 FAIL: expected 3 update_status_current() call sites "
        f"(pre-launch, stdout pump, heartbeat thread), found {n}. A new "
        "caller inherits fail-open-and-log; confirm that is right for it, "
        "then update this count."
    )


# --------------------------------------------------------------------------
# C15 -- the log is rate-limited, on the clock.
# --------------------------------------------------------------------------


def test_c15_status_write_error_log_is_rate_limited_by_time(monkeypatch, capsys):
    """First occurrence, then at most hourly.

    Not once (a single line scrolls out of a multi-hour run's output) and not
    every occurrence: the write fires every STATUS_WRITE_INTERVAL (5s) from
    the heartbeat thread AND again from the stdout pump, so a persistent
    fault would emit on the order of 1400 identical lines/hour.
    """
    clock = {"t": 1000.0}
    monkeypatch.setattr(
        experiment_runner.time, "monotonic", lambda: clock["t"]
    )
    monkeypatch.setattr(experiment_runner, "_STATUS_WRITE_ERROR_COUNT", 0)
    monkeypatch.setattr(
        experiment_runner, "_STATUS_WRITE_ERROR_LAST_LOG", [0.0]
    )

    exc = OSError(28, "No space left on device")

    # 720 ticks at 5s == one hour: 1 logged (the first), none after.
    for _ in range(720):
        experiment_runner._on_status_write_error(exc)
        clock["t"] += 5.0

    first_hour = capsys.readouterr().out.count("status write FAILED")
    assert first_hour == 1, (
        f"C15 FAIL: expected exactly 1 line in the first hour, got "
        f"{first_hour}."
    )

    # Cross the hour boundary -> exactly one more.
    experiment_runner._on_status_write_error(exc)
    second = capsys.readouterr().out.count("status write FAILED")
    assert second == 1, (
        f"C15 FAIL: expected 1 line just past the hour boundary, got {second}."
    )

    assert experiment_runner._STATUS_WRITE_ERROR_COUNT == 721, (
        "C15 FAIL: suppressed occurrences must still be COUNTED, so the "
        "line that does print can say how many it stands for."
    )


def test_c15b_rate_limit_is_on_the_clock_not_a_tick_count(monkeypatch, capsys):
    """A modulus would be wrong HERE, and this pins the difference.

    The sibling `_on_evidence_guard_error` counts ticks, which is exact for
    its single 60s loop. This helper serves two callers on different
    cadences, so no modulus corresponds to a fixed interval. Under a
    count-based limiter these 5 calls in the same instant would be silent
    until the modulus hit; under a clock-based one, advancing the clock an
    hour between them logs every time regardless of count.
    """
    clock = {"t": 500.0}
    monkeypatch.setattr(
        experiment_runner.time, "monotonic", lambda: clock["t"]
    )
    monkeypatch.setattr(experiment_runner, "_STATUS_WRITE_ERROR_COUNT", 0)
    monkeypatch.setattr(
        experiment_runner, "_STATUS_WRITE_ERROR_LAST_LOG", [0.0]
    )

    for _ in range(5):
        experiment_runner._on_status_write_error(RuntimeError("x"))
        clock["t"] += 3601.0

    n = capsys.readouterr().out.count("status write FAILED")
    assert n == 5, (
        f"C15b FAIL: 5 occurrences each an hour apart must all log "
        f"(got {n}) -- the limiter is on elapsed time, not on a count."
    )
