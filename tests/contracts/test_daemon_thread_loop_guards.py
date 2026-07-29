"""Contract tests for the silent-death guard on EVERY daemon-thread loop in
experiment_runner.py.

This is the tail of a three-commit sweep, and it deliberately replaces the
per-thread argument with a per-FILE invariant.

  * ree-v3 3dcb3ccd2d guarded `_background_sync` (contracts C9/C10 in
    tests/contracts/test_background_sync_claim_guard.py) and recorded the
    sibling `_heartbeat` as unexamined.
  * ree-v3 2b95400d15 guarded `_heartbeat` and made `update_status_current`
    total (contracts C11-C15 in test_heartbeat_thread_death_guard.py), and
    recorded `_delayed_pulls` as the remaining one.
  * This file guards `_delayed_pulls` and stops the pattern of discovering
    the next instance by hand.

The defect class: `run_experiment()` and `align_after_coordinator_result()`
spawn daemon threads whose target is a loop. Nothing restarts those threads
and nothing surfaces their death, so ONE exception escaping a loop body ends
that thread's work for the rest of the experiment -- silently. Guarding the
body per-iteration converts that into one skipped tick.

Why a contract for `_delayed_pulls` specifically, given it could not raise
today. Three facts made its severity LOW, and none of them is a reason to
leave it unguarded:

  (a) Its only call, `_sync_pull_tick`, is provably total as of 3dcb3ccd2d,
      and that totality is itself pinned by contract C10 ("no unguarded call
      other than the provably-total `_pull_ree_v3`").
  (b) The loop is BOUNDED at two iterations, so even a raise cost at most the
      second delayed pull -- not a permanently dead thread, which is what
      made the other two serious.
  (c) `time.sleep` does not raise in normal operation.

So this was hardening against a future regression, not a live defect. The
argument for doing it anyway: C10 protects the CALLEE and nothing protected
this CALL SITE, so a second statement added to the loop body would have
re-armed the class silently -- and consistency across all three threads is
what makes "every daemon loop body in this file is guarded" checkable at all.

WHY THIS FILE ENUMERATES RATHER THAN NAMES. The two predecessor contracts are
bespoke per thread, which is why the second and third instances were each
found by hand, after the fix that should have surfaced them. C16 below
discovers the daemon threads from the AST and pins the SET, so a fourth
thread added later fails this file on the day it lands rather than waiting
for someone to notice the pattern a fourth time. The per-thread contracts are
kept, not folded in: they assert thread-specific facts (which calls must be
inside the try, the blast radius of each) that a generic enumerator cannot.

Contracts:
  C16.  Every `threading.Thread(target=..., daemon=True)` in the module
        resolves to a function defined in the module, and the SET of such
        targets is exactly the three known ones. A new daemon thread fails
        here by construction.
  C17.  Each enumerated target's loop body is exactly one `try/except
        Exception` -- handler not bare, not BaseException, and not a silent
        `pass`. Structural, because all three are closures nested inside
        their enclosing function and are not reachable for monkeypatching --
        the same constraint C9 and C11 work around.
  C18.  The extracted `_delayed_pulls` source, COMPILED AND EXECUTED against
        a raising `_sync_pull_tick`, actually survives and still performs its
        SECOND pull -- an execution-level proof that the AST shape C17 pins
        does what it claims, mirroring C11b.

NEGATIVE CONTROLS. Every structural assertion is re-run against PRE-FIX
source and must FAIL there. Without that, an AST assertion phrased slightly
wrong passes vacuously on both revisions and pins nothing. The controls share
the predicate FUNCTION with the live-source assertions rather than re-stating
it, since a control that re-words the predicate proves nothing about the
predicate actually used. Note the controls here are per-THREAD (each of the
three pre-fix bodies), so C17 is negative-controlled at every instance it
asserts over, not just at the newest one.

The pre-fix sources are EMBEDDED below (PRE_FIX_BODIES) rather than read from
git, because `scripts/remote_pytest.sh` rsyncs the tree WITHOUT `.git/` -- a
git-backed control would silently SKIP on precisely the machine the suite is
supposed to run on, leaving the vacuity check unexecuted while this file
still reported green. The snapshots are frozen at specific commits and can
never drift; `test_negative_control_snapshots_match_git` re-derives them from
git when `.git/` happens to be present, so a transcription error is still
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


# The daemon threads this file knows about. Pinned as a SET by C16 so a
# fourth one cannot be added without landing here first. `where` is the
# enclosing function, `why` the blast radius of a silent death -- both are
# for whoever reads a failure, not for the assertions.
KNOWN_DAEMON_TARGETS = {
    "_heartbeat": (
        "run_experiment",
        "runner_status/*.json and the remote-control heartbeat stop, so the "
        "explorer /machines dashboard and the cloud-scaler's telemetry go "
        "stale; the scaler's git-fallback path reads that as an idle machine",
    ),
    "_background_sync": (
        "run_experiment",
        "--auto-sync stops pulling for the rest of the experiment",
    ),
    "_delayed_pulls": (
        "align_after_coordinator_result",
        "the second post-result alignment pull is skipped, so a phase3: "
        "commit appears locally later than intended (BOUNDED at 2 iterations "
        "-- the least severe of the three, see the module docstring)",
    ),
}

# Pre-fix source per target: (ref, dedented function source). Each ref is the
# last revision carrying that thread's UNGUARDED body, i.e. the commit that
# fixed the PREVIOUS thread in the sweep. Verbatim from
# `git show <ref>:experiment_runner.py`, dedented (all three are nested
# inside their enclosing function there). Frozen history -- cannot drift.
PRE_FIX_BODIES = {
    "_background_sync": (
        "92b9b99d13",
        '''\
def _background_sync():
    while not _hb_stop.wait(timeout=60):
        _sync_pull_tick(ree_assembly_path)
''',
    ),
    "_heartbeat": (
        "3dcb3ccd2d",
        '''\
def _heartbeat():
    while not _hb_stop.wait(timeout=STATUS_WRITE_INTERVAL):
        update_status_current()
        _push_remote_heartbeat()
''',
    ),
    "_delayed_pulls": (
        "2b95400d15",
        '''\
def _delayed_pulls() -> None:
    for delay in (45, 120):
        time.sleep(delay)
        _sync_pull_tick(ree_assembly_path)
''',
    ),
}


# --------------------------------------------------------------------------
# Source access helpers.
# --------------------------------------------------------------------------


def _live_source() -> str:
    return Path(experiment_runner.__file__).read_text(encoding="utf-8")


def _find_fn(src: str, name: str) -> ast.FunctionDef:
    found = [
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert found, (
        f"{name} not found in experiment_runner.py -- if it was renamed, "
        "re-point this contract rather than deleting it."
    )
    assert len(found) == 1, (
        f"{name} is defined {len(found)} times in experiment_runner.py. This "
        "file resolves a thread target by NAME, which is only sound while "
        "names are unique. Disambiguate by scope before relaxing this."
    )
    return found[0]


def _daemon_thread_targets(src: str) -> list[str]:
    """Every `Thread(target=<Name>, daemon=True)` target name in the module.

    Deliberately structural and deliberately narrow: a target that is not a
    bare Name (a lambda, a bound method, a functools.partial) is NOT silently
    ignored -- it is returned as a sentinel so C16's set comparison fails and
    a human decides whether the guard argument applies to it.
    """
    out: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_thread = (
            (isinstance(func, ast.Attribute) and func.attr == "Thread")
            or (isinstance(func, ast.Name) and func.id == "Thread")
        )
        if not is_thread:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        daemon = kw.get("daemon")
        if not (isinstance(daemon, ast.Constant) and daemon.value is True):
            continue
        target = kw.get("target")
        if isinstance(target, ast.Name):
            out.append(target.id)
        else:
            rendered = ast.unparse(target) if target is not None else "<none>"
            out.append(f"<non-name target at line {node.lineno}: {rendered}>")
    return out


# --------------------------------------------------------------------------
# The structural predicate, factored so the SAME code runs against the live
# source and against every pre-fix snapshot. A negative control that
# re-states the predicate in its own words proves nothing about the predicate
# actually used; sharing the function is what makes the control binding.
# --------------------------------------------------------------------------


def _loop_body_is_guarded(src: str, name: str) -> tuple[bool, str, str]:
    """(guarded, code, detail) for a daemon-thread target's loop body.

    The rule is the strict one: the body of the target's single top-level
    loop must be EXACTLY one `try`, whose handlers include a bare `Exception`
    name, with no bare/BaseException handler and no silent handler body.

    "Exactly one statement" is stricter than "every call sits inside a try",
    on purpose. The weaker form is satisfied by a body that guards its calls
    and then leaves a bare attribute lookup or subscript outside -- which is
    exactly how `_background_sync`'s guard escaped notice in 3dcb3ccd2d (two
    sub-expressions sat outside an inner try that looked total). If a future
    thread legitimately needs a statement outside the try, this fires and the
    rationale gets re-read rather than the assertion quietly relaxed.
    """
    fn = _find_fn(src, name)
    loops = [n for n in fn.body if isinstance(n, (ast.While, ast.For))]
    if len(loops) != 1:
        return False, "loop-shape", (
            f"expected exactly one top-level loop in {name}, found "
            f"{len(loops)}"
        )
    body = [n for n in loops[0].body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str)
    )]  # drop a docstring-shaped string expression, if any
    if len(body) != 1 or not isinstance(body[0], ast.Try):
        kinds = [type(n).__name__ for n in body]
        return False, "no-try", (
            "loop body is not a single try/except -- an exception escaping "
            f"here kills the thread for the rest of the run (found {kinds})"
        )
    try_node = body[0]

    for h in try_node.handlers:
        if h.type is None:
            return False, "bare-handler", (
                "bare `except:` also swallows SystemExit/KeyboardInterrupt, "
                "so the thread could no longer be stopped"
            )
        if isinstance(h.type, ast.Name) and h.type.id == "BaseException":
            return False, "bare-handler", "handler catches BaseException"
    if not any(
        isinstance(h.type, ast.Name) and h.type.id == "Exception"
        for h in try_node.handlers
    ):
        return False, "no-exception-handler", (
            "no handler catches Exception specifically"
        )

    for h in try_node.handlers:
        if all(isinstance(s, ast.Pass) for s in h.body):
            return False, "silent-handler", (
                "handler body is a bare `pass` -- the failure must be "
                "LOGGED, not silently absorbed (the _on_git_timeout lesson)"
            )
    return True, "", ""


# --------------------------------------------------------------------------
# C16 -- the enumeration itself.
# --------------------------------------------------------------------------


def test_c16_daemon_thread_target_set_is_pinned():
    """A fourth daemon thread must land HERE before it lands unguarded.

    The two predecessor contracts named their thread, so each next instance
    was found by hand after the fix that should have surfaced it. Pinning the
    discovered SET is what converts that into an automatic failure.
    """
    found = sorted(_daemon_thread_targets(_live_source()))
    expected = sorted(KNOWN_DAEMON_TARGETS)
    assert found == expected, (
        f"C16 FAIL: daemon-thread targets changed.\n"
        f"  found:    {found}\n"
        f"  expected: {expected}\n"
        "If a thread was ADDED: give its loop body a per-iteration "
        "`try/except Exception` that LOGS, then add it to "
        "KNOWN_DAEMON_TARGETS with its blast radius. Nothing restarts these "
        "threads and their death is silent, so an unguarded loop turns one "
        "transient exception into permanent, invisible loss of that thread's "
        "work. If a target is reported as `<non-name target ...>`, decide "
        "whether the guard argument applies to it rather than widening the "
        "enumerator to skip it."
    )


def test_c16b_every_target_resolves_to_a_module_function():
    """The names C17 asserts over must actually be resolvable, uniquely."""
    for name in _daemon_thread_targets(_live_source()):
        _find_fn(_live_source(), name)  # raises with its own message


# --------------------------------------------------------------------------
# C17 -- every enumerated loop body is guarded.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(KNOWN_DAEMON_TARGETS))
def test_c17_daemon_loop_body_is_exception_guarded(name):
    """One escaped exception must cost one tick, not every remaining tick."""
    guarded, code, detail = _loop_body_is_guarded(_live_source(), name)
    where, why = KNOWN_DAEMON_TARGETS[name]
    assert guarded, (
        f"C17 FAIL for {name} (in {where}): {detail} [{code}].\n"
        f"Blast radius of a silent death: {why}."
    )


@pytest.mark.parametrize("name", sorted(PRE_FIX_BODIES))
def test_c17_negative_control_pre_fix_bodies_are_unguarded(name):
    """The predicate must FAIL on each pre-fix body, or it pins nothing.

    Parametrized over all three rather than only the newest, so C17 is
    negative-controlled at every instance it asserts over.
    """
    ref, snapshot = PRE_FIX_BODIES[name]
    guarded, code, detail = _loop_body_is_guarded(snapshot, name)
    assert not guarded, (
        f"C17 NEGATIVE CONTROL FAIL: the pre-fix {name} at {ref} passes the "
        "same predicate the live source is asserted against, so C17 proves "
        "nothing for it. Fix the predicate."
    )
    assert code == "no-try", (
        f"C17 NEGATIVE CONTROL: pre-fix {name} failed for an UNEXPECTED "
        f"reason ({code}: {detail}). Expected the absent try/except -- if the "
        "reason changed, the control is no longer testing the documented "
        "defect."
    )


# --------------------------------------------------------------------------
# C18 -- execution-level proof for `_delayed_pulls`, the thread this commit
# fixes. C17 pins a shape; this runs it.
# --------------------------------------------------------------------------


def _exec_delayed_pulls(src: str, tick):
    """Compile+exec the real `_delayed_pulls` source against stand-ins.

    The closure's free variables become globals in the exec namespace, so the
    loop body executes verbatim without entering
    `align_after_coordinator_result()`. `time` is stubbed so the contract does
    not actually sleep for 165 seconds.
    """
    fn = _find_fn(src, "_delayed_pulls")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)

    slept: list[float] = []

    class _Clock:
        @staticmethod
        def sleep(seconds):
            slept.append(seconds)

    logged: list[str] = []
    ns = {
        "time": _Clock,
        "_sync_pull_tick": tick,
        "ree_assembly_path": Path("/nonexistent/REE_assembly"),
        "print": lambda *a, **k: logged.append(" ".join(str(x) for x in a)),
    }
    exec(compile(module, "<_delayed_pulls>", "exec"), ns)
    ns["_delayed_pulls"]()
    return slept, logged


def test_c18_delayed_pulls_survives_a_raising_tick():
    """A raise on the FIRST pull must not cost the second one.

    This is the whole value of guarding a bounded loop: per-iteration, not
    once around the outside. A single try wrapping the entire `for` would
    also stop the thread from dying yet would still lose the second pull, and
    would pass a weaker "is there a try somewhere" assertion.
    """
    pulls = {"n": 0}

    def _boom(_path):
        pulls["n"] += 1
        raise OSError(107, "Transport endpoint is not connected")

    slept, logged = _exec_delayed_pulls(_live_source(), _boom)

    assert pulls["n"] == 2, (
        f"C18 FAIL: `_sync_pull_tick` was called {pulls['n']} time(s). The "
        "guard must be INSIDE the loop, so a failed first pull still leaves "
        "the second."
    )
    assert slept == [45, 120], (
        f"C18 FAIL: expected the documented 45s/120s schedule, saw {slept}. "
        "The delays cover hub writer tick latency; changing them changes "
        "how long a phase3: commit takes to appear locally."
    )
    assert len(logged) == 2, (
        f"C18 FAIL: expected both failures LOGGED, saw {len(logged)} line(s) "
        "-- silent swallowing is the _on_git_timeout lesson."
    )


def test_c18_negative_control_pre_fix_delayed_pulls_dies():
    """The same execution harness against pre-fix source must let it escape."""
    _ref, snapshot = PRE_FIX_BODIES["_delayed_pulls"]
    pulls = {"n": 0}

    def _boom(_path):
        pulls["n"] += 1
        raise OSError(107, "Transport endpoint is not connected")

    with pytest.raises(OSError):
        _exec_delayed_pulls(snapshot, _boom)

    assert pulls["n"] == 1, (
        "C18 NEGATIVE CONTROL: pre-fix, the first raise must end the thread "
        f"before the second pull (saw {pulls['n']} pull(s))."
    )


def test_c18b_happy_path_still_performs_both_pulls():
    """Totality must not have severed the work.

    A guard that swallowed everything would satisfy C17 and C18 while doing
    nothing useful, so pin that both pulls still happen when nothing raises.
    """
    pulls: list[object] = []
    slept, logged = _exec_delayed_pulls(_live_source(), pulls.append)

    assert len(pulls) == 2, (
        f"C18b FAIL: expected 2 alignment pulls, saw {len(pulls)}. The guard "
        "is supposed to make the pull survivable, not optional."
    )
    assert slept == [45, 120]
    assert logged == [], (
        f"C18b FAIL: the happy path must be silent, saw {logged}."
    )


# --------------------------------------------------------------------------
# Transcription audit for the embedded snapshots -- NOT the negative control.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PRE_FIX_BODIES))
def test_negative_control_snapshots_match_git(name):
    """The embedded pre-fix snapshots must be faithful transcriptions.

    Skips when `.git/` is unavailable, which is the normal case under
    `scripts/remote_pytest.sh` (its rsync excludes .git/). That is exactly
    why the controls above are embedded rather than git-backed: this check is
    an opportunistic transcription audit, NOT the negative control itself. A
    negative control that can only run on the developer's laptop is not a
    control.
    """
    ref, snapshot = PRE_FIX_BODIES[name]
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show",
             f"{ref}:experiment_runner.py"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git unavailable for snapshot audit: {exc!r}")
    if out.returncode != 0:
        pytest.skip(
            f"git show {ref} unavailable ({out.stderr.strip()[:200]}) -- "
            "staged tree excludes .git/"
        )

    lines = out.stdout.splitlines()
    fn = _find_fn(out.stdout, name)
    real = textwrap.dedent("\n".join(lines[fn.lineno - 1:fn.end_lineno])).rstrip()
    assert real == snapshot.rstrip(), (
        f"SNAPSHOT DRIFT: the embedded pre-fix {name} does not match {ref}. "
        "The negative controls are testing text that never existed. "
        "Re-transcribe from git rather than editing the assertion."
    )
