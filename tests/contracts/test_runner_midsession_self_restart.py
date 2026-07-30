"""Contract tests for the runner's MID-SESSION code refresh (self re-exec).

Background -- the other half of the 2026-07-30 stale-code defect:
  The COLD-BOOT half was fixed by 310a80e (ExecStartPre runs
  coordinator/deploy/runner-prestart-pull.sh, refreshing ree-v3 before the
  Python process loads it) and is pinned by test_runner_prestart_code_refresh.py.
  That fix acts at process START only. A LONG-LIVED runner keeps executing
  whatever it loaded at launch -- forever -- because Python does not hot-reload,
  while its own in-loop `git pull` advances the code ON DISK underneath it.

  Measured 2026-07-30 (session friendly-antonelli-b0f414):
    - The hub (ree-cloud-1) ran `r4072 35c103c 2026-07-20` for TEN DAYS. Its
      last "Runner version:" journal line was Jul 20 05:22:05; the process only
      exited 2026-07-30T06:07:04Z.
    - ree-cloud-3 ran `r4515 e68d52b` (launched 2026-07-29T21:27Z) through the
      2026-07-30 morning while its disk sat at origin/main, logging
      "git pull ree-v3: Already up to date." every 60s against code the process
      was not running.
    - _refresh_runner_version() ALREADY rendered the divergence as
      "r4530 9077eb7 2026-07-30 (running r4515 e68d52b)". Nothing consumed it.

Contracts:
  C1.  The between-pass loop calls _maybe_self_restart, and that is the ONLY
       call site (the fix exists, and it cannot fire from anywhere unvetted).
  C2.  _self_restart_busy_reason refuses whenever the runner is not idle --
       above all while a claim is held or an experiment subprocess is alive.
       This is what keeps a restart from orphaning a coordinator claim or
       manufacturing a phantom completion.
  C3.  A restart requires the on-disk build to be STRICTLY NEWER by commit
       count. execve does NOT go through systemd, so the unit's
       StartLimitBurst/StartLimitIntervalSec do not bound it; a sideways or
       backwards HEAD move must not ping-pong the runner.
  C4.  A minimum interval is enforced between restarts, and the bookkeeping
       that enforces it travels through the exec in the ENVIRONMENT. In-memory
       state would reset on every restart -- i.e. the brake would be released
       by the very event it exists to limit.
  C5.  A newer build that touches no module this process loaded does NOT
       restart. experiments/ drivers run as subprocesses and are already fresh
       on every run; they dominate ree-v3 traffic by volume.
  C6.  When the diff cannot be computed the runner restarts CONSERVATIVELY
       (it cannot prove it is current, and C3/C4 still bound it).
  C7.  The exec preserves argv -- every runner flag survives -- and rebuilds
       the script path from REPO_ROOT rather than reusing sys.argv[0], which
       may be relative.
  C8.  The new build is vetted BEFORE the exec: it must compile, and it must
       pass the SAME startup preflight suite the new process will run. main()
       does sys.exit(rc) on a failing preflight, and the Mac runner has no
       systemd Restart=, so an unvetted exec there is a permanently dead runner.
  C9.  _maybe_self_restart is a no-op outside long-lived fleet mode
       (--loop --auto-sync, not --dry-run) and never raises.
  C10. The off switch (REE_RUNNER_NO_SELF_RESTART=1) is honoured.
  C11. Every string this feature prints is ASCII-only (journals are read on
       cp1252 terminals).
"""

import ast
import os
import re
import sys
from pathlib import Path

import pytest

REE_V3 = Path(__file__).resolve().parents[2]
RUNNER_SRC = REE_V3 / "experiment_runner.py"

sys.path.insert(0, str(REE_V3))

import experiment_runner as er  # noqa: E402


# --- helpers ---------------------------------------------------------------

def _runner_text() -> str:
    return RUNNER_SRC.read_text(encoding="utf-8")


def _call_names(tree: ast.AST, func_name: str) -> list[ast.Call]:
    """Every ast.Call to a bare name `func_name` in the module."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == func_name:
            out.append(node)
    return out


def _ok(**overrides):
    """Kwargs for _should_self_restart that DO produce a restart, so each test
    can flip exactly one field and attribute the refusal to that field."""
    base = dict(
        disk_version="r4530 9077eb7 2026-07-30",
        process_version="r4515 e68d52b 2026-07-29",
        changed_paths=["ree_core/predictors/e3_selector.py"],
        loaded_paths={"ree_core/predictors/e3_selector.py",
                      "experiment_runner.py"},
        now=10_000.0,
        last_restart=None,
        min_interval_sec=600,
        disabled=False,
    )
    base.update(overrides)
    return base


# --- C1 --------------------------------------------------------------------

def test_c1_between_pass_loop_calls_maybe_self_restart_exactly_once():
    tree = ast.parse(_runner_text())
    calls = _call_names(tree, "_maybe_self_restart")
    assert calls, (
        "experiment_runner.py never calls _maybe_self_restart. That is the "
        "2026-07-30 mid-session stale-code defect: a long-lived runner executes "
        "its launch-time build forever because Python does not hot-reload, even "
        "while its own loop pulls newer code to disk. The hub ran a ten-day-old "
        "build this way."
    )
    assert len(calls) == 1, (
        f"_maybe_self_restart must have exactly ONE call site -- the between-pass "
        f"point, where no claim is held and no experiment subprocess is alive. "
        f"Found {len(calls)} calls; any other site risks re-exec'ing mid-run."
    )


def test_c1_call_site_is_after_the_between_pass_pull():
    text = _runner_text()
    pull = text.index('_pull_ree_v3("between-pass")')
    # Search from `pull` so this finds the CALL, not the `def` above it.
    call = text.index("_maybe_self_restart(args", pull)
    assert call > pull, (
        "the self-restart check must run AFTER the between-pass pull -- the "
        "pull is what can make the running process stale, so checking before it "
        "always reads a one-pass-old disk state."
    )


# --- C2 --------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("current_claim", ["V3-EXQ-999"]),
    ("current_proc", ["<popen>"]),
    ("drain_flag", [True]),
    ("pause_flag", [True]),
    ("force_stop_flag", [True]),
    ("suspend_flag", [True]),
])
def test_c2_busy_reason_refuses_when_runner_is_not_idle(field, value):
    kwargs = dict(
        current_claim=[], current_proc=[], drain_flag=[],
        pause_flag=[], force_stop_flag=[], suspend_flag=[],
    )
    kwargs[field] = value
    reason = er._self_restart_busy_reason(**kwargs)
    assert reason is not None, (
        f"_self_restart_busy_reason returned None (=idle, safe to restart) "
        f"while {field}={value!r}. Re-exec'ing then would abandon in-flight "
        f"work: with a live claim it orphans the coordinator claim and can "
        f"leave a phantom completion (see "
        f"reference_phantom_completion_crash_before_manifest)."
    )
    assert reason.strip(), "the refusal reason must be a non-empty message"


def test_c2_busy_reason_is_none_when_fully_idle():
    assert er._self_restart_busy_reason(
        current_claim=[], current_proc=[], drain_flag=[],
        pause_flag=[], force_stop_flag=[], suspend_flag=[],
    ) is None


def test_c2_claim_is_checked_before_everything_else():
    """A held claim is the most dangerous state, and its message must name the
    queue_id so the journal shows WHICH run was protected."""
    reason = er._self_restart_busy_reason(
        current_claim=["V3-EXQ-843"], current_proc=["<popen>"],
        drain_flag=[True], pause_flag=[True],
        force_stop_flag=[True], suspend_flag=[True],
    )
    assert "V3-EXQ-843" in reason, (
        f"with a claim held, the refusal must name the claim; got {reason!r}"
    )


def test_c2_call_site_passes_the_live_state_containers():
    """The guard is only real if the call site feeds it main()'s ACTUAL flags.
    A call passing literals or a subset would type-check and pin nothing."""
    text = _runner_text()
    # Search from the between-pass pull, or `.index` lands on the `def` line.
    call_start = text.index("_maybe_self_restart(args",
                            text.index('_pull_ree_v3("between-pass")'))
    call_blob = text[call_start:call_start + 700]
    for name in ("_current_claim", "_current_proc", "_drain_flag",
                 "_pause_flag", "_force_stop_flag", "_suspend_flag"):
        assert name in call_blob, (
            f"the self-restart call site does not pass {name} to "
            f"_self_restart_busy_reason, so that state cannot block a restart"
        )


# --- C3 --------------------------------------------------------------------

def test_c3_equal_builds_do_not_restart():
    should, reason = er._should_self_restart(**_ok(
        disk_version="r4515 e68d52b 2026-07-29",
        process_version="r4515 e68d52b 2026-07-29",
    ))
    assert not should, reason


@pytest.mark.parametrize("disk,proc", [
    ("r4515 aaaaaaa 2026-07-30", "r4515 e68d52b 2026-07-29"),   # sideways
    ("r4500 aaaaaaa 2026-07-30", "r4515 e68d52b 2026-07-29"),   # backwards
])
def test_c3_requires_strictly_newer_commit_count(disk, proc):
    should, reason = er._should_self_restart(**_ok(
        disk_version=disk, process_version=proc))
    assert not should, (
        f"restarted onto a build that is NOT strictly newer ({disk} vs {proc}). "
        f"os.execve does NOT go through systemd, so StartLimitBurst=5 / "
        f"StartLimitIntervalSec=900 do NOT bound this -- a HEAD that moves "
        f"sideways (force-push, reset) would ping-pong the runner forever."
    )
    assert "strictly newer" in reason


def test_c3_strictly_newer_does_restart():
    should, _ = er._should_self_restart(**_ok())
    assert should


@pytest.mark.parametrize("bad", ["", "no-count 9077eb7", "x4530 9077eb7"])
def test_c3_unparseable_version_refuses_rather_than_guessing(bad):
    should, reason = er._should_self_restart(**_ok(disk_version=bad))
    assert not should, (
        f"restarted on an unreadable version string {bad!r}. An unreadable "
        f"version must fail CLOSED: without a commit count the strictly-newer "
        f"brake (C3) cannot be applied at all."
    )
    assert reason


def test_c3_missing_process_version_refuses():
    should, _ = er._should_self_restart(**_ok(process_version=None))
    assert not should
    should, _ = er._should_self_restart(**_ok(disk_version=None))
    assert not should


# --- C4 --------------------------------------------------------------------

def test_c4_rate_limit_blocks_a_recent_restart():
    should, reason = er._should_self_restart(**_ok(
        now=10_000.0, last_restart=9_800.0, min_interval_sec=600))
    assert not should, "restarted only 200s after the previous restart"
    assert "rate-limited" in reason


def test_c4_rate_limit_releases_after_the_interval():
    should, _ = er._should_self_restart(**_ok(
        now=10_000.0, last_restart=9_000.0, min_interval_sec=600))
    assert should


def test_c4_restart_bookkeeping_is_carried_in_the_environment(monkeypatch):
    """execve destroys all in-memory state. If the last-restart timestamp lived
    in a module global it would reset on every restart -- releasing the brake
    exactly when it is needed. It must be written into the exec'd env."""
    captured = {}

    def fake_execve(path, argv, env):
        captured["env"] = env
        raise SystemExit("execve reached")

    monkeypatch.setattr(os, "execve", fake_execve)
    monkeypatch.setenv(er._REEXEC_COUNT_ENV, "3")
    monkeypatch.delenv(er._REEXEC_LAST_ENV, raising=False)

    with pytest.raises(SystemExit):
        er._self_restart_now("test", now=12_345.0)

    env = captured["env"]
    assert env.get(er._REEXEC_LAST_ENV) == "12345", (
        f"the restart timestamp must ride in ${er._REEXEC_LAST_ENV}; got "
        f"{env.get(er._REEXEC_LAST_ENV)!r}"
    )
    assert env.get(er._REEXEC_COUNT_ENV) == "4", (
        f"the generation counter must increment across the exec; got "
        f"{env.get(er._REEXEC_COUNT_ENV)!r}"
    )


def test_c4_last_restart_is_read_from_that_same_env_var(monkeypatch):
    """The write side (above) and the read side must agree on the var name --
    a mismatch would silently disable the rate limit."""
    monkeypatch.setenv(er._REEXEC_LAST_ENV, "9800")
    assert er._env_epoch(er._REEXEC_LAST_ENV) == 9800.0
    monkeypatch.setenv(er._REEXEC_LAST_ENV, "not-a-number")
    assert er._env_epoch(er._REEXEC_LAST_ENV) is None


def test_c4_min_interval_has_a_nonzero_default(monkeypatch):
    monkeypatch.delenv(er._REEXEC_MIN_INTERVAL_ENV, raising=False)
    assert er._reexec_min_interval_sec() > 0, (
        "the default minimum interval must be positive, or the only remaining "
        "loop brake is the strictly-newer check"
    )


def test_c4_min_interval_env_override_and_garbage_fallback(monkeypatch):
    monkeypatch.setenv(er._REEXEC_MIN_INTERVAL_ENV, "1800")
    assert er._reexec_min_interval_sec() == 1800
    monkeypatch.setenv(er._REEXEC_MIN_INTERVAL_ENV, "banana")
    assert er._reexec_min_interval_sec() == er._REEXEC_DEFAULT_MIN_INTERVAL_SEC
    monkeypatch.setenv(er._REEXEC_MIN_INTERVAL_ENV, "-5")
    assert er._reexec_min_interval_sec() >= 0, (
        "a negative interval must clamp, not invert the comparison"
    )


# --- C5 --------------------------------------------------------------------

def test_c5_new_experiment_drivers_do_not_restart_the_runner():
    """The dominant commit class in ree-v3. Drivers execute as SUBPROCESSES
    from the current disk on every run, so they are never stale."""
    should, reason = er._should_self_restart(**_ok(
        changed_paths=[
            "experiments/v3_exq_842_mech217_offline_wanting_spread.py",
            "experiments/_lib/baselines/mech217.py",
            "experiment_queue.json",
        ],
        loaded_paths={"experiment_runner.py", "ree_core/config.py"},
    ))
    assert not should, (
        "restarted for a commit that only added experiment drivers. Those run "
        "as subprocesses and are already fresh; restarting for them would burn "
        "a full startup (preflight included) for no change in behaviour."
    )
    assert "none of them loaded" in reason


def test_c5_a_loaded_module_change_does_restart():
    should, reason = er._should_self_restart(**_ok(
        changed_paths=["experiment_runner.py", "docs/whatever.md"],
        loaded_paths={"experiment_runner.py"},
    ))
    assert should
    assert "experiment_runner.py" in reason, (
        "the reason must name the changed module, so the journal says WHY the "
        "runner restarted"
    )


def test_c5_loaded_module_set_is_derived_from_sys_modules_not_a_hardcoded_list():
    loaded = er._loaded_repo_modules()
    assert "experiment_runner.py" in loaded, (
        "the runner's own file must always count as loaded -- it is the module "
        "the whole defect is about"
    )
    assert all(not p.startswith("/") for p in loaded), (
        "paths must be repo-relative to compare against `git diff --name-only`"
    )
    assert all(not p.startswith("..") for p in loaded), (
        "paths outside REPO_ROOT (stdlib, site-packages) must be filtered out"
    )
    # An allowlist would rot toward "never restart" -- i.e. back into the bug.
    src = _runner_text()
    fn = src[src.index("def _loaded_repo_modules"):]
    fn = fn[:fn.index("\ndef ")]
    assert "sys.modules" in fn, (
        "the loaded-module set must be derived from sys.modules. A "
        "hand-maintained allowlist silently rots when the runner starts "
        "importing something new, and it rots toward NOT restarting -- "
        "reproducing the very defect this fixes."
    )


# --- C6 --------------------------------------------------------------------

def test_c6_unavailable_diff_restarts_conservatively():
    should, reason = er._should_self_restart(**_ok(changed_paths=None))
    assert should, (
        "when the diff cannot be computed (e.g. the running build's sha is "
        "unreachable after a force-push) the runner cannot prove it is current. "
        "Staying put is how the ten-day staleness happened; C3 and C4 still "
        "bound the restart."
    )
    assert "conservatively" in reason


def test_c6_unavailable_diff_still_obeys_the_other_brakes():
    should, _ = er._should_self_restart(**_ok(
        changed_paths=None, disk_version="r4500 aaaaaaa 2026-07-30"))
    assert not should, "the conservative path must not bypass strictly-newer"
    should, _ = er._should_self_restart(**_ok(
        changed_paths=None, now=10_000.0, last_restart=9_990.0))
    assert not should, "the conservative path must not bypass the rate limit"


# --- C7 --------------------------------------------------------------------

def test_c7_exec_preserves_argv_and_uses_the_resolved_runner_path(monkeypatch):
    captured = {}

    def fake_execve(path, argv, env):
        captured["path"] = path
        captured["argv"] = argv
        raise SystemExit("execve reached")

    monkeypatch.setattr(os, "execve", fake_execve)
    monkeypatch.setattr(sys, "argv", [
        "experiment_runner.py", "--machine", "ree-cloud-1",
        "--auto-sync", "--loop", "--remote-control", "--loop-interval", "60",
    ])
    with pytest.raises(SystemExit):
        er._self_restart_now("test", now=1.0)

    argv = captured["argv"]
    assert argv[0] == sys.executable, "argv[0] must be the interpreter"
    assert captured["path"] == sys.executable
    assert Path(argv[1]).is_absolute(), (
        f"the script path must be absolute -- sys.argv[0] can be relative, and "
        f"resolving it after the exec would depend on the CWD. Got {argv[1]!r}"
    )
    assert Path(argv[1]).name == "experiment_runner.py"
    for flag in ("--machine", "ree-cloud-1", "--auto-sync", "--loop",
                 "--remote-control", "--loop-interval", "60"):
        assert flag in argv[2:], (
            f"{flag!r} did not survive the exec. A runner that loses "
            f"--machine misidentifies itself to the coordinator; one that "
            f"loses --loop exits after one pass."
        )
    assert argv[2:] == sys.argv[1:], (
        "the exec'd arguments must be this process's arguments verbatim"
    )


def test_c7_failed_exec_does_not_kill_the_runner(monkeypatch, capsys):
    """execve only returns by failing. Stale code beats a dead runner."""
    def boom(path, argv, env):
        raise OSError("ENOENT")

    monkeypatch.setattr(os, "execve", boom)
    er._self_restart_now("test", now=1.0)   # must NOT raise
    out = capsys.readouterr().out
    assert "FAILED" in out and "continuing" in out


# --- C8 --------------------------------------------------------------------

def test_c8_noncompiling_on_disk_build_is_rejected(monkeypatch, tmp_path):
    (tmp_path / "experiment_runner.py").write_text(
        "def broken(:\n", encoding="utf-8")
    monkeypatch.setattr(er, "REPO_ROOT", tmp_path)
    ok, detail = er._new_build_is_loadable(skip_preflight=True)
    assert not ok, (
        "exec'd into an experiment_runner.py that does not even compile. Once "
        "execve succeeds there is no way back, and a truncated or half-written "
        "checkout is exactly what a mid-pull exec would find."
    )
    assert "compile" in detail


def test_c8_failing_preflight_blocks_the_restart(monkeypatch, tmp_path):
    (tmp_path / "experiment_runner.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "preflight").mkdir(parents=True)
    monkeypatch.setattr(er, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("REE_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(er.subprocess, "call", lambda *a, **k: 1)
    ok, detail = er._new_build_is_loadable(skip_preflight=False)
    assert not ok, (
        "exec'd into a build that FAILS the startup preflight. main() does "
        "sys.exit(rc) on that failure, and the Mac runner is launched by hand "
        "from the explorer with no systemd Restart= -- so this would leave the "
        "laptop with NO runner at all."
    )
    assert "preflight" in detail


def test_c8_passing_preflight_allows_the_restart(monkeypatch, tmp_path):
    (tmp_path / "experiment_runner.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "preflight").mkdir(parents=True)
    monkeypatch.setattr(er, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("REE_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(er.subprocess, "call", lambda *a, **k: 0)
    ok, _ = er._new_build_is_loadable(skip_preflight=False)
    assert ok


def test_c8_preflight_argv_is_shared_with_the_startup_gate():
    """A drifted copy would let the runner exec into a build that then
    sys.exit()s -- the pre-exec check is only meaningful if it runs the SAME
    suite the new process runs at startup."""
    tree = ast.parse(_runner_text())
    calls = _call_names(tree, "_preflight_pytest_argv")
    assert len(calls) >= 2, (
        f"_preflight_pytest_argv must be used by BOTH the startup preflight "
        f"and the pre-exec gate; found {len(calls)} call site(s). If either "
        f"builds its own argv the two can drift apart silently."
    )
    argv = er._preflight_pytest_argv(Path("/tmp/preflight"))
    assert argv[0] == sys.executable and argv[1:3] == ["-m", "pytest"]
    assert str(Path("/tmp/preflight")) in argv


def test_c8_new_build_check_runs_before_the_exec():
    src = _runner_text()
    fn = src[src.index("def _maybe_self_restart"):]
    assert fn.index("_new_build_is_loadable") < fn.index("_self_restart_now"), (
        "the on-disk build must be vetted BEFORE execve is called; after the "
        "exec there is no way back"
    )


# --- C9 --------------------------------------------------------------------

class _Args:
    def __init__(self, **kw):
        self.loop = kw.get("loop", True)
        self.auto_sync = kw.get("auto_sync", True)
        self.dry_run = kw.get("dry_run", False)
        self.skip_preflight = kw.get("skip_preflight", False)


@pytest.mark.parametrize("kw", [
    {"loop": False},
    {"auto_sync": False},
    {"dry_run": True},
])
def test_c9_no_restart_outside_long_lived_fleet_mode(kw, monkeypatch):
    def boom(*a, **k):
        raise AssertionError(f"attempted a self-restart with {kw}")

    monkeypatch.setattr(er, "_self_restart_now", boom)
    er._maybe_self_restart(_Args(**kw), None)


def test_c9_busy_runner_never_reaches_the_restart(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("attempted a self-restart while busy")

    monkeypatch.setattr(er, "_self_restart_now", boom)
    er._maybe_self_restart(_Args(), busy_reason="holding claim V3-EXQ-1")


def test_c9_never_raises(monkeypatch):
    """A crash here would take down a runner that was otherwise healthy."""
    monkeypatch.setattr(er, "_git_code_version",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("git gone")))
    er._maybe_self_restart(_Args(), None)   # must not raise


# --- C10 -------------------------------------------------------------------

def test_c10_off_switch_is_honoured():
    should, reason = er._should_self_restart(**_ok(disabled=True))
    assert not should
    assert er._REEXEC_DISABLE_ENV in reason, (
        "the refusal must name the env var, so an operator reading the journal "
        "knows why the runner is not refreshing itself"
    )


def test_c10_off_switch_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(er._REEXEC_DISABLE_ENV, "1")
    calls = []
    monkeypatch.setattr(er, "_self_restart_now",
                        lambda *a, **k: calls.append(a))
    monkeypatch.setattr(er, "_git_code_version",
                        lambda *a, **k: "r9999 abcdef0 2026-07-30")
    er._maybe_self_restart(_Args(), None)
    assert not calls, (
        f"{er._REEXEC_DISABLE_ENV}=1 did not prevent a restart"
    )


def test_c10_refusal_is_logged_once_per_reason_not_once_per_pass(monkeypatch, capsys):
    """The builds diverge for most of a worker's life (new experiments/ drivers
    land constantly), so an unconditional line would be ~1440 journal lines per
    worker per day -- and a message everyone skips is as good as no message.
    But it must still be said at least once: the ABSENCE of a signal is what
    let the ten-day hub staleness go unnoticed."""
    monkeypatch.setattr(er, "_RUNNER_PROCESS_VERSION", "r4515 e68d52b 2026-07-29")
    monkeypatch.setattr(er, "_LAST_SELF_RESTART_NOTE", None)
    monkeypatch.setattr(er, "_git_code_version", lambda *a, **k: "r4530 9077eb7 2026-07-30")
    # Newer build, but it touches nothing this process loaded -> steady refusal.
    monkeypatch.setattr(er, "_changed_paths_since",
                        lambda *a, **k: ["experiments/v3_exq_900_thing.py"])
    monkeypatch.setattr(er, "_loaded_repo_modules", lambda: {"experiment_runner.py"})

    for _ in range(5):
        er._maybe_self_restart(_Args(), None)
    out = capsys.readouterr().out
    n = out.count("Not self-restarting")
    assert n == 1, (
        f"the refusal printed {n} times across 5 identical passes; it must be "
        f"logged once per REASON, not once per pass"
    )
    assert "none of them loaded" in out, (
        "the one line that is printed must still explain the refusal"
    )


def test_c10_busy_refusal_is_also_logged_once_not_per_pass(monkeypatch, capsys):
    """Same flood risk on the busy path: a paused runner stays paused, so this
    branch would otherwise print every loop_interval indefinitely."""
    monkeypatch.setattr(er, "_LAST_SELF_RESTART_NOTE", None)
    for _ in range(5):
        er._maybe_self_restart(_Args(), busy_reason="paused")
    out = capsys.readouterr().out
    assert out.count("Self-restart check skipped") == 1, (
        f"the busy refusal printed {out.count('Self-restart check skipped')} "
        f"times across 5 identical passes; it must be logged once per reason"
    )


def test_c10_a_changed_reason_is_logged_again(monkeypatch, capsys):
    """Dedup must be per-REASON, not a one-shot latch -- otherwise the first
    refusal permanently silences every later, different one."""
    monkeypatch.setattr(er, "_LAST_SELF_RESTART_NOTE", None)
    er._maybe_self_restart(_Args(), busy_reason="paused")
    er._maybe_self_restart(_Args(), busy_reason="holding claim V3-EXQ-7")
    out = capsys.readouterr().out
    assert "paused" in out and "V3-EXQ-7" in out, (
        f"a second, DIFFERENT refusal was swallowed by the dedup: {out!r}"
    )


# --- C11 -------------------------------------------------------------------

def test_c11_feature_output_is_ascii_only():
    src = _runner_text()
    start = src.index("# Mid-session code refresh")
    end = src.index("# Guard B (Version-Layering Doctrine)")
    block = src[start:end]
    for i, ln in enumerate(block.splitlines(), start=1):
        try:
            ln.encode("ascii")
        except UnicodeEncodeError as exc:
            raise AssertionError(
                f"self-restart block line {i} contains non-ASCII, which renders "
                f"as mojibake on cp1252 terminals reading the journal: "
                f"{ln!r} ({exc})"
            ) from None


def test_c11_no_unicode_arrows_or_dashes_in_printed_reasons():
    """Belt and braces for the strings that actually reach stdout."""
    src = _runner_text()
    start = src.index("# Mid-session code refresh")
    end = src.index("# Guard B (Version-Layering Doctrine)")
    block = src[start:end]
    for bad in ("—", "→", "←", "×", "…", "≈"):
        assert bad not in block, f"non-ASCII {bad!r} in the self-restart block"
    assert not re.search(r"print\([^)]*[^\x00-\x7f]", block)
