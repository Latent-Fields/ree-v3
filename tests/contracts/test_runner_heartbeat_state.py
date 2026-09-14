"""Contract for `experiment_runner._heartbeat_state` -- the mid-run remote-control
heartbeat's drain/pause state computation.

THE DEFECT (chip-20260909-runner-midrun-heartbeat-drain-state, fixed 2026-09-14).
`_push_remote_heartbeat()` (the nested heartbeat helper inside `run_experiment()`,
the ONLY heartbeat writer that runs WHILE an experiment is executing) used to call
`_rrc.write_heartbeat(..., state="running", ...)` with `state` a hardcoded literal.
`handle_signal()` sets the runner's drain flag on SIGTERM and the runner keeps
running until the current experiment finishes -- minutes to hours -- so throughout
that entire window the mid-run heartbeat kept reporting "running" regardless of an
in-flight drain or pause. The coordinator DB, `/shadow/status`, and the explorer
`/machines` dashboard all inherited the stale value; only the end-of-pass tick
(between experiments, i.e. after the window of interest) ever computed the real
state, via `hb_state = "paused" if _pause_flag else ("draining" if _drain_flag else
"idle")`.

THE FIX. `_heartbeat_state(pause_flag, drain_flag, default)` is now the SINGLE
source of that precedence (paused > draining > default), consumed by BOTH the
mid-run call site (`default="running"` -- an experiment IS executing whenever
neither flag is set there) and the between-pass tick (`default="idle"`). See the
helper's own docstring for the full precedence and why the two call sites need
different defaults.

Note on scope: `pause`/`drain` here are THIS RUNNER PROCESS's own state, computed
from the SAME mutable flag lists main() holds (_pause_flag / _drain_flag).
Deliberately unrelated to the coordinator's own separate "draining" verdict
(coordinator/db.py, a pending MACHINE shutdown notice) -- see
REE_assembly/serve.py's `_runner_draining()` docstring for that distinction; this
test does not touch the coordinator side.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiment_runner as R  # noqa: E402


# ---- precedence -------------------------------------------------------------------

def test_heartbeat_state_paused_beats_draining():
    assert R._heartbeat_state(pause_flag=[True], drain_flag=[True],
                               default="running") == "paused"


def test_heartbeat_state_draining_when_only_drain_set():
    assert R._heartbeat_state(pause_flag=[], drain_flag=[True],
                               default="running") == "draining"


def test_heartbeat_state_paused_when_only_pause_set():
    assert R._heartbeat_state(pause_flag=[True], drain_flag=[],
                               default="idle") == "paused"


def test_heartbeat_state_falls_through_to_default_when_neither_set():
    assert R._heartbeat_state(pause_flag=[], drain_flag=[],
                               default="running") == "running"
    assert R._heartbeat_state(pause_flag=[], drain_flag=[],
                               default="idle") == "idle"


def test_heartbeat_state_none_flags_treated_as_unset():
    # drain_flag / pause_flag are Optional[list] on run_experiment()'s own
    # signature (not threaded through when the caller omits them) -- None
    # must behave exactly like an empty list, never raise.
    assert R._heartbeat_state(pause_flag=None, drain_flag=None,
                               default="running") == "running"
    assert R._heartbeat_state(pause_flag=None, drain_flag=[True],
                               default="running") == "draining"
    assert R._heartbeat_state(pause_flag=[True], drain_flag=None,
                               default="idle") == "paused"


# ---- the mid-run call site's DEFAULT is "running", not "idle" ----------------------
# This is the substance of the defect: an experiment is actively executing at the
# mid-run call site, so "neither flag set" must read "running" there, never "idle"
# (the between-pass tick's own default, where nothing is executing).

def test_default_running_differs_from_default_idle_with_no_flags_set():
    assert R._heartbeat_state([], [], default="running") != \
        R._heartbeat_state([], [], default="idle")


# ---- regression: the mid-run heartbeat must not go back to a literal --------------

def test_push_remote_heartbeat_source_computes_state_not_literal():
    """`_push_remote_heartbeat`'s write_heartbeat call must not pass a hardcoded
    state="running" literal -- the exact regression this chip fixed. Static-source
    check (not an execution test) because `_push_remote_heartbeat` is a closure
    nested inside `run_experiment()` and not independently callable; the precedence
    unit is covered directly by the tests above, and the two call sites both routing
    through `_heartbeat_state` is covered by the AST check below."""
    src = Path(R.__file__).read_text(encoding="utf-8")
    assert 'state="running"' not in src, (
        "experiment_runner.py has a hardcoded state=\"running\" literal again -- "
        "this is the chip-20260909-runner-midrun-heartbeat-drain-state regression. "
        "Compute the state via _heartbeat_state(pause_flag, drain_flag, "
        "default=\"running\") instead.")


def test_both_drain_pause_sensitive_heartbeat_call_sites_use_the_shared_helper():
    """Every `write_heartbeat(...)` call site whose `state=` is anything other than
    a plain string constant (i.e. every site that is actually trying to report a
    computed runner state, as opposed to the one-shot "starting" / "offline"
    lifecycle markers written at process start/exit) must route through
    `_heartbeat_state(...)` -- covers the mid-run site (a `_heartbeat_state(...)`
    call expression) and the between-pass tick (an `hb_state` variable, itself
    assigned from `_heartbeat_state(...)`), so the two computations cannot drift
    apart again the way the mid-run site drifted from the between-pass site's own
    correct precedence before this fix. `state="starting"` / `state="offline"` are
    deliberately out of scope: those are unconditional lifecycle markers, not a
    drain/pause snapshot, and were never part of this defect."""
    import ast

    tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"), filename=R.__file__)
    write_heartbeat_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "write_heartbeat"
    ]
    computed_state_calls = [
        call for call in write_heartbeat_calls
        if not any(
            kw.arg == "state" and isinstance(kw.value, ast.Constant)
            for kw in call.keywords
        )
    ]
    assert len(computed_state_calls) >= 2, (
        f"expected at least 2 write_heartbeat call sites with a COMPUTED (non-literal) "
        f"state= (mid-run + between-pass), found {len(computed_state_calls)} -- "
        f"re-derive this test if the runner's heartbeat call sites changed shape.")
    for call in computed_state_calls:
        state_kw = next((kw for kw in call.keywords if kw.arg == "state"), None)
        assert state_kw is not None, (
            f"write_heartbeat call at line {call.lineno} has no state= kwarg")
        is_helper_call = (
            isinstance(state_kw.value, ast.Call)
            and isinstance(state_kw.value.func, ast.Name)
            and state_kw.value.func.id == "_heartbeat_state"
        )
        is_helper_var = (
            isinstance(state_kw.value, ast.Name)
            and state_kw.value.id == "hb_state"
        )
        assert is_helper_call or is_helper_var, (
            f"write_heartbeat call at line {call.lineno} does not route state= "
            f"through _heartbeat_state(...) (found: {ast.dump(state_kw.value)})")
