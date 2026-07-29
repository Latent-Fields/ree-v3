"""Contract tests for the active-claim guard on the --auto-sync background
pull (experiment_runner._sync_pull_tick).

Background -- daemon autostash revert incident class (EXQ-232 / 2026-04-29
ARC-026 supersession, recurring 2026-05-08 on substrate_queue.json):
  experiment_runner.run_experiment() spawns a daemon `_background_sync`
  thread that, every 60s while a run is in progress, did:
      git_pull(REPO_ROOT, "ree-v3")
      git_pull(ree_assembly_path, "REE_assembly")   # <-- unguarded
  git_pull uses `git pull --rebase --autostash`. Unlike
  runner_remote_control.push_heartbeat / push_commands -- which early-return
  when a Claude session holds an active TASK_CLAIMS claim covering any
  evidence/ path -- this daemon pull had no such guard. While a
  governance/evidence session held uncommitted edits to REE_assembly
  evidence/claims files, the daemon autostashed them every 60s and stacked
  failing autostash-pop stashes, silently reverting claims.yaml / manifest /
  review_tracker edits. The thread is daemon=True and not _pause_flag-aware,
  so `pause` did not stop it -- only full runner shutdown did.

Fix: the per-tick pull body was extracted to module-level
`_sync_pull_tick(ree_assembly_path)`, which skips ONLY the REE_assembly
pull for the tick when `_rrc._active_claim_on_evidence_dir(...)` is True --
the exact guard push_heartbeat / push_commands already use.

2026-07-28 UPDATE: the ree-v3 pull is no longer unguarded either. It now
routes through `experiment_runner._pull_ree_v3`, which applies an analogous
(but deliberately NOT symmetric) guard for ree-v3 substrate paths -- see
tests/contracts/test_ree_v3_pull_claim_guard.py for that guard's own
contracts and for the confirmed 2026-07-27 orphaned-autostash incident that
motivated it. The tests below therefore pin BOTH guards explicitly rather
than letting the ree-v3 side read the real TASK_CLAIMS.json.

Contracts:
  C1. No active evidence claim -> both ree-v3 AND REE_assembly are pulled
      (bit-identical to the pre-guard default path).
  C2. Active evidence claim -> ree-v3 is pulled, REE_assembly is SKIPPED.
  C3. ree_assembly_path is None -> only ree-v3 is pulled, no crash.
  C4. runner_remote_control unimportable (_rrc is None) -> both pulled
      (default path preserved even when the guard module is unavailable;
      mirrors push_heartbeat / push_commands call-site, which are also
      gated on _rrc is not None).
  C5. The guard consulted is exactly _rrc._active_claim_on_evidence_dir --
      the same function push_heartbeat / push_commands use -- so the two
      cannot silently diverge.
  C6. _sync_pull_tick never raises even if git_pull itself raises
      (best-effort; matches the try/except the daemon previously inlined).
  C7. The two guards are INDEPENDENT: an active ree-v3 substrate claim
      skips only the ree-v3 pull and leaves REE_assembly alone, and vice
      versa (C2). A single claim must not gate both repos.
  C8. A guard that cannot be EVALUATED (stale _rrc lacking the function;
      any exception from it) fails OPEN and is logged, rather than
      propagating and killing the sync thread. C8c pins the log's rate
      limit.
  C9. _background_sync's loop body is try/except-guarded, so no future
      addition to the tick can terminate the thread. Structural, because
      the closure is nested inside run_experiment().
  C10. _sync_pull_tick contains no unguarded call other than the
      provably-total _pull_ree_v3 -- the property whose absence made the
      docstring's "Never raises" false from the 2026-07-28 extraction until
      2026-07-29.

2026-07-29 UPDATE (C8/C9/C10): `_sync_pull_tick` said "Never raises" but its
guard call was unwrapped, and `_background_sync`'s loop had no try/except of
its own -- so an exception there ended --auto-sync for the rest of the
experiment, silently and unrecoverably. `_active_claim_on_paths`' blanket
`except Exception` made the guard's BODY total, but not the attribute lookup
`_rrc._active_claim_on_evidence_dir` nor the `ree_assembly_path.parent`
evaluated ahead of it. Fail-open was chosen over fail-closed because the
failure mode is deterministic: a fail-closed guard would suppress every
REE_assembly pull for the whole run. Context: WORKSPACE_STATE.md
2026-07-29T17:0xZ, session beautiful-tereshkova-263da4, side-finding (2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402
import runner_remote_control  # noqa: E402


@pytest.fixture(autouse=True)
def no_ree_v3_claim(monkeypatch):
    """Default the ree-v3 substrate guard OFF for this file's contracts.

    Without this the ree-v3 half would consult the REAL
    /Users/dgolden/REE_Working/TASK_CLAIMS.json and the real working tree,
    making every assertion below depend on whatever other sessions happen to
    be holding open. The ree-v3 guard has its own contract file.
    """
    monkeypatch.setattr(
        runner_remote_control, "_active_claim_on_ree_v3_code", lambda _p: False
    )


@pytest.fixture
def record_pulls(monkeypatch):
    """Replace git_pull with a recorder of the labels it was called with."""
    calls: list[str] = []
    monkeypatch.setattr(
        experiment_runner, "git_pull",
        lambda repo, label: calls.append(label),
    )
    return calls


def _set_claim(monkeypatch, active: bool) -> None:
    monkeypatch.setattr(
        runner_remote_control, "_active_claim_on_evidence_dir",
        lambda _p: active,
    )
    monkeypatch.setattr(experiment_runner, "_rrc", runner_remote_control)


def test_c1_no_active_claim_pulls_both(record_pulls, monkeypatch, tmp_path):
    _set_claim(monkeypatch, active=False)
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")
    assert record_pulls == ["ree-v3", "REE_assembly"], (
        "C1 FAIL: default path must pull both repos bit-identically."
    )


def test_c2_active_claim_skips_assembly(record_pulls, monkeypatch, tmp_path):
    _set_claim(monkeypatch, active=True)
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")
    assert record_pulls == ["ree-v3"], (
        "C2 FAIL: an active evidence claim must skip the REE_assembly pull "
        "(the EXQ-232 autostash-revert guard) while leaving ree-v3 alone."
    )


def test_c3_none_assembly_path_pulls_only_reev3(
    record_pulls, monkeypatch
):
    _set_claim(monkeypatch, active=False)
    experiment_runner._sync_pull_tick(None)
    assert record_pulls == ["ree-v3"], (
        "C3 FAIL: ree_assembly_path=None must pull ree-v3 only, no crash."
    )


def test_c4_rrc_none_preserves_default_path(record_pulls, monkeypatch, tmp_path):
    # Guard module unavailable: cannot consult claims -> must not change the
    # pre-guard behaviour (both pulls run), matching push_heartbeat /
    # push_commands which are themselves gated on `_rrc is not None`.
    monkeypatch.setattr(experiment_runner, "_rrc", None)
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")
    assert record_pulls == ["ree-v3", "REE_assembly"], (
        "C4 FAIL: with _rrc None the default both-pull path must be "
        "bit-identical to pre-guard behaviour."
    )


def test_c5_uses_the_same_guard_as_push_heartbeat(
    record_pulls, monkeypatch, tmp_path
):
    # Sentinel: if _sync_pull_tick consulted any guard OTHER than
    # _rrc._active_claim_on_evidence_dir, patching that exact symbol would
    # not flip the behaviour and this assertion would fail -- catching guard
    # drift between the daemon and push_heartbeat / push_commands.
    seen = {"called": False}

    def _spy(_p):
        seen["called"] = True
        return True

    monkeypatch.setattr(
        runner_remote_control, "_active_claim_on_evidence_dir", _spy
    )
    monkeypatch.setattr(experiment_runner, "_rrc", runner_remote_control)
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")
    assert seen["called"], (
        "C5 FAIL: _sync_pull_tick did not consult "
        "_rrc._active_claim_on_evidence_dir -- guard drift risk."
    )
    assert record_pulls == ["ree-v3"], (
        "C5 FAIL: the consulted guard did not gate the REE_assembly pull."
    )


def test_c7_ree_v3_claim_skips_only_ree_v3(record_pulls, monkeypatch, tmp_path):
    """The two guards are independent: a ree-v3 substrate claim must not gate
    the REE_assembly pull, just as an evidence claim does not gate ree-v3."""
    monkeypatch.setattr(
        runner_remote_control, "_active_claim_on_evidence_dir", lambda _p: False
    )
    monkeypatch.setattr(
        runner_remote_control, "_active_claim_on_ree_v3_code", lambda _p: True
    )
    monkeypatch.setattr(experiment_runner, "_ree_v3_code_dirty", lambda: True)
    monkeypatch.setattr(experiment_runner, "_rrc", runner_remote_control)
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")
    assert record_pulls == ["REE_assembly"], (
        "C7 FAIL: an active ree-v3 substrate claim must skip ONLY the ree-v3 "
        "pull; REE_assembly has its own, separate guard."
    )


def test_c6_never_raises_when_git_pull_raises(monkeypatch, tmp_path):
    def _boom(repo, label):
        raise RuntimeError(f"git exploded for {label}")

    monkeypatch.setattr(experiment_runner, "git_pull", _boom)
    _set_claim(monkeypatch, active=False)
    # Must swallow the exception for BOTH pulls -- best-effort, exactly as
    # the daemon's inlined try/except did before the extraction.
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")


# --------------------------------------------------------------------------
# C8/C9/C10 -- "Never raises" made TRUE, not merely intended (2026-07-29).
#
# The guard call was _sync_pull_tick's ONLY unwrapped statement and
# _background_sync had no try/except, so one exception from the guard killed
# the sync thread for the remainder of the experiment -- silently, and
# permanently, since nothing restarts it.
#
# `_active_claim_on_paths` already carries a blanket `except Exception:
# return False`, so the guard BODY was total. What was not covered is the
# part of the call expression evaluated BEFORE that try runs: the attribute
# lookup on the best-effort-imported `_rrc` module, and the
# `ree_assembly_path.parent` the wrapper computes. C8 pins the first of
# those, which is the realistic one -- a worker whose runner_remote_control.py
# predates _active_claim_on_evidence_dir.
# --------------------------------------------------------------------------


def test_c8_stale_rrc_without_the_guard_fn_fails_open(
    record_pulls, monkeypatch, tmp_path, capsys
):
    """A `_rrc` module lacking _active_claim_on_evidence_dir must not raise.

    This is the concrete escape route the blanket except inside
    `_active_claim_on_paths` cannot close: AttributeError is raised by the
    attribute LOOKUP, before any code inside that function runs.
    """
    class _StaleRRC:  # no _active_claim_on_evidence_dir at all
        pass

    monkeypatch.setattr(experiment_runner, "_rrc", _StaleRRC())
    monkeypatch.setattr(experiment_runner, "_EVIDENCE_GUARD_ERROR_COUNT", 0)
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")

    assert record_pulls == ["ree-v3", "REE_assembly"], (
        "C8 FAIL: an unevaluable guard must fail OPEN (pull proceeds), "
        "matching _pull_ree_v3 and _active_claim_on_paths' documented "
        "failure directions. Failing closed would stop REE_assembly pulls "
        "for the whole run, since this failure is deterministic."
    )
    assert "evidence-claim guard FAILED" in capsys.readouterr().out, (
        "C8 FAIL: the failure must be LOGGED, not silently absorbed -- the "
        "_on_git_timeout lesson."
    )


def test_c8b_guard_raising_arbitrary_exception_does_not_escape(
    record_pulls, monkeypatch, tmp_path
):
    """Same contract for any exception from the guard, not just AttributeError."""
    def _boom(_p):
        raise RuntimeError("claims file on fire")

    monkeypatch.setattr(
        runner_remote_control, "_active_claim_on_evidence_dir", _boom
    )
    monkeypatch.setattr(experiment_runner, "_rrc", runner_remote_control)
    experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")
    assert record_pulls == ["ree-v3", "REE_assembly"], (
        "C8b FAIL: _sync_pull_tick's docstring promises it never raises."
    )


def test_c8c_guard_error_log_is_rate_limited(monkeypatch, tmp_path, capsys):
    """First occurrence then hourly -- not once (invisible), not every tick
    (1440 identical lines/day, the loop being 60s)."""
    monkeypatch.setattr(experiment_runner, "git_pull", lambda repo, label: None)
    monkeypatch.setattr(
        runner_remote_control, "_active_claim_on_evidence_dir",
        lambda _p: (_ for _ in ()).throw(RuntimeError("x")),
    )
    monkeypatch.setattr(experiment_runner, "_rrc", runner_remote_control)
    monkeypatch.setattr(experiment_runner, "_EVIDENCE_GUARD_ERROR_COUNT", 0)

    for _ in range(120):
        experiment_runner._sync_pull_tick(tmp_path / "REE_assembly")

    n_logged = capsys.readouterr().out.count("evidence-claim guard FAILED")
    assert n_logged == 3, (
        f"C8c FAIL: expected 3 log lines over 120 ticks (#1, #60, #120), "
        f"got {n_logged}."
    )


def _background_sync_ast():
    """Return the ast.FunctionDef for the nested `_background_sync` closure.

    Source-level because the closure is defined inside run_experiment() and
    is not reachable as an attribute for monkeypatching.
    """
    import ast

    src = Path(experiment_runner.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_background_sync":
            return node
    raise AssertionError(
        "_background_sync not found in experiment_runner.py -- if it was "
        "renamed, re-point this contract rather than deleting it."
    )


def test_c9_background_sync_loop_body_is_exception_guarded():
    """Defence in depth: no future addition to the tick body can kill the thread.

    The loop is the only thing between one escaped exception and a sync thread
    that is dead for the rest of the experiment (it is never restarted, and the
    death is silent). This pins the try/except INSIDE the while, so a failing
    tick costs one skipped pull rather than all remaining pulls.
    """
    import ast

    fn = _background_sync_ast()
    whiles = [n for n in fn.body if isinstance(n, ast.While)]
    assert whiles, "C9 FAIL: _background_sync no longer has a while loop."

    tries = [n for n in whiles[0].body if isinstance(n, ast.Try)]
    assert tries, (
        "C9 FAIL: _background_sync's loop body must wrap its work in "
        "try/except. Without it, one raise ends --auto-sync for the run."
    )

    calls = [
        n for n in ast.walk(tries[0])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_sync_pull_tick"
    ]
    assert calls, (
        "C9 FAIL: the _sync_pull_tick call must be INSIDE the try, not "
        "beside it."
    )

    handlers = tries[0].handlers
    assert any(
        h.type is not None and isinstance(h.type, ast.Name)
        and h.type.id == "Exception"
        for h in handlers
    ), (
        "C9 FAIL: the handler must catch Exception specifically. A bare "
        "`except:` would also swallow SystemExit/KeyboardInterrupt, which "
        "SHOULD be able to stop this thread."
    )


def test_c10_sync_pull_tick_has_no_unguarded_statements():
    """Every statement in _sync_pull_tick's body is a guard, a return, or
    wrapped -- the property C8 restores, asserted structurally so it cannot
    silently regress the way it did between the 2026-07-28 extraction and
    2026-07-29.
    """
    import ast

    src = Path(experiment_runner.__file__).read_text(encoding="utf-8")
    fn = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_sync_pull_tick"
    )
    # Calls that are NOT inside a Try within this function.
    guarded = {
        id(c) for t in ast.walk(fn) if isinstance(t, ast.Try)
        for c in ast.walk(t) if isinstance(c, ast.Call)
    }
    unguarded = [
        c for c in ast.walk(fn)
        if isinstance(c, ast.Call) and id(c) not in guarded
    ]
    names = sorted(
        c.func.id if isinstance(c.func, ast.Name)
        else getattr(c.func, "attr", "<expr>")
        for c in unguarded
    )
    # _pull_ree_v3 is itself documented "Never raises" and is total by
    # construction (both halves internally wrapped); it is the one sanctioned
    # unguarded call.
    assert names == ["_pull_ree_v3"], (
        "C10 FAIL: unguarded call(s) in _sync_pull_tick: "
        f"{names}. Every call here must either be inside a try/except or be "
        "itself provably total -- the function's docstring promises it never "
        "raises, and _background_sync's thread dies permanently if it does."
    )
