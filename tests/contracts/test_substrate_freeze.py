"""Contract tests for the pause-the-puller substrate mutex (option A2).

Design of record: REE_assembly/evidence/planning/substrate_stability_and_drift_detection_plan.md
section 3, node `substrate_stability:ISO-design`.

Background -- V3-EXQ-875 (MECH-471): a ~20.5h run recorded
`substrate_stable_across_run: false` because six `ree_core`-touching commits
landed on `origin/main` inside its window while the runner executed the driver
IN PLACE in the same shared checkout a co-resident loop was pulling into. Benign
that time; nothing structural prevented it. A2 makes a running experiment hold a
local freeze that the puller defers around.

The contracts below are grouped by what they protect. Roughly half are NEGATIVE
CONTROLS, and they are the load-bearing ones: this guard sits in front of the
ONLY path that pulls `ree-v3`, and `experiment_queue.json` lives in `ree-v3`, so
a guard that defers when it should not blinds a worker to new work. Widening
this predicate by one case is a fleet-wide sync outage.

  A. Identity     -- the guarded prefix set does not drift from what
                     arm_fingerprint actually hashes.
  B. Default off  -- an unflagged runner acquires nothing and behaves as before.
  C. Lifecycle    -- acquire/release/hold, including release on exception.
  D. Self-healing -- a crashed, expired, or corrupt holder never wedges the
                     puller.
  E. Precision    -- a freeze defers ONLY pulls that actually move substrate,
                     so the queue stays fresh under a 20h run.
  F. Fail-open    -- every undeterminable state resolves to "pull anyway".
  G. Wiring       -- both halves of the mutex are really reachable from
                     experiment_runner, asserted against the real source.

All tests are time-independent (TTLs are set explicitly or backdated by writing
the record) and use real git repositories in a tempdir.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_REE_V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REE_V3))

import substrate_freeze as sf  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", str(repo)] + list(args),
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"git {args} failed: {r.stderr}")
    return r


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "T")
    (path / "ree_core").mkdir(exist_ok=True)
    (path / "ree_core" / "agent.py").write_text("X = 1\n")
    (path / "experiment_queue.json").write_text('{"items": []}\n')
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")
    return path


@pytest.fixture
def repo(tmp_path):
    return _init_repo(tmp_path / "work")


@pytest.fixture
def cloned(tmp_path):
    """An `origin` plus a clone of it -- the real runner topology."""
    origin = _init_repo(tmp_path / "origin")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    _git(clone, "config", "user.email", "t@example.com")
    _git(clone, "config", "user.name", "T")
    return origin, clone


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("REE_SUBSTRATE_FREEZE", "REE_SUBSTRATE_FREEZE_IGNORE",
              "REE_SUBSTRATE_FREEZE_TTL_SECONDS"):
        monkeypatch.delenv(k, raising=False)


# --------------------------------------------------------------------------
# A. Identity
# --------------------------------------------------------------------------

def test_prefixes_cover_arm_fingerprint_globs():
    """SUBSTRATE_PREFIXES must guard every path arm_fingerprint HASHES.

    These are two independent literals in two files. If a glob is added to
    _SUBSTRATE_GLOBS and not here, that path flips
    `substrate_stable_across_run` while the freeze silently ignores it -- an
    inert mutex, the worst failure mode available. This test is the drift
    guard that makes keeping them in lockstep mechanical.
    """
    from experiments._lib import arm_fingerprint
    for glob in arm_fingerprint._SUBSTRATE_GLOBS:
        # Reduce each glob to the literal path prefix before any wildcard.
        head = glob.split("*", 1)[0]
        assert any(head.startswith(p) or p.startswith(head)
                   for p in sf.SUBSTRATE_PREFIXES), (
            f"arm_fingerprint hashes {glob!r} but SUBSTRATE_PREFIXES does not "
            f"guard it -- the freeze would be inert for that path")


def test_prefixes_do_not_guard_the_queue_file():
    """NEGATIVE CONTROL. experiment_queue.json must NOT be guarded.

    It lives in this same repo and changes constantly. Guarding it would make
    every queue snapshot defer the pull for the life of a 20h run, which is
    precisely the stale-queue wedge this design avoids.
    """
    assert not "experiment_queue.json".startswith(sf.SUBSTRATE_PREFIXES)


# --------------------------------------------------------------------------
# B. Default off
# --------------------------------------------------------------------------

def test_freeze_disabled_by_default():
    assert sf.freeze_enabled() is False


@pytest.mark.parametrize("val,expected",
                         [("1", True), ("true", True), ("YES", True),
                          ("on", True), ("0", False), ("", False),
                          ("maybe", False)])
def test_freeze_flag_parsing(monkeypatch, val, expected):
    monkeypatch.setenv("REE_SUBSTRATE_FREEZE", val)
    assert sf.freeze_enabled() is expected


def test_hold_disabled_acquires_nothing(repo):
    """NEGATIVE CONTROL: the unflagged path must leave no trace at all."""
    with sf.hold(repo, "V3-EXQ-001") as token:
        assert token is None
        assert sf.held(repo) == []
    assert not sf.freeze_dir(repo).exists()


# --------------------------------------------------------------------------
# C. Lifecycle
# --------------------------------------------------------------------------

def test_acquire_then_release_round_trip(repo):
    token = sf.acquire(repo, "V3-EXQ-002")
    assert token is not None
    holders = sf.held(repo)
    assert len(holders) == 1
    assert holders[0]["run_label"] == "V3-EXQ-002"
    assert holders[0]["pid"] == os.getpid()
    sf.release(token)
    assert sf.held(repo) == []


def test_hold_releases_on_exception(repo, monkeypatch):
    """A run that raises must not leak its freeze -- that would wedge the
    puller until the TTL, for a run that is already over."""
    monkeypatch.setenv("REE_SUBSTRATE_FREEZE", "1")
    with pytest.raises(RuntimeError):
        with sf.hold(repo, "V3-EXQ-003"):
            assert len(sf.held(repo)) == 1
            raise RuntimeError("boom")
    assert sf.held(repo) == []


def test_multiple_holders_are_independent(repo):
    a = sf.acquire(repo, "A")
    b = sf.acquire(repo, "B")
    assert len(sf.held(repo)) == 2
    sf.release(a)
    labels = [h["run_label"] for h in sf.held(repo)]
    assert labels == ["B"]
    sf.release(b)


def test_release_is_safe_on_none_and_missing(repo):
    sf.release(None)
    token = sf.acquire(repo, "X")
    sf.release(token)
    sf.release(token)  # already gone -- must not raise


def test_freeze_dir_with_git_directory(repo):
    assert sf.freeze_dir(repo) == repo / ".git" / "ree_substrate_freeze"


def test_freeze_dir_with_git_FILE_worktree(tmp_path, repo):
    """The documented .git-file-vs-directory trap (CLAUDE.md, remote_pytest.sh
    2026-07-31). In a `git worktree` .git is a FILE. A freeze written under a
    mis-resolved git dir is invisible to the puller reading the right one.
    """
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", "--detach", str(wt))
    assert (wt / ".git").is_file(), "fixture precondition: worktree .git is a file"
    d = sf.freeze_dir(wt)
    assert d.name == "ree_substrate_freeze"
    assert ".git" in str(d) and "worktrees" in str(d)
    # And it round-trips through that resolution.
    token = sf.acquire(wt, "WT")
    assert token is not None and sf.held(wt)[0]["run_label"] == "WT"
    sf.release(token)


def test_freeze_dir_without_git_falls_back_outside_the_tree(tmp_path):
    """The rsync-staged tree remote_pytest.sh builds has no .git at all. The
    fallback must still work and must NOT write inside the tree (an untracked
    file there would collide with the runner's untracked-path recovery)."""
    bare = tmp_path / "staged"
    bare.mkdir()
    d = sf.freeze_dir(bare)
    assert not str(d).startswith(str(bare))
    token = sf.acquire(bare, "S")
    assert token is not None
    assert sf.held(bare)[0]["run_label"] == "S"
    sf.release(token)


# --------------------------------------------------------------------------
# D. Self-healing
# --------------------------------------------------------------------------

def _write_record(repo, **over):
    d = sf.freeze_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    import socket
    import time as _t
    rec = {"run_label": "R", "pid": os.getpid(), "host": socket.gethostname(),
           "acquired_at": "2026-01-01T00:00:00Z",
           "acquired_monotonic_wall": _t.time(),
           "ttl_seconds": sf.DEFAULT_TTL_SECONDS, "repo_root": str(repo)}
    rec.update(over)
    p = d / "rec.json"
    p.write_text(json.dumps(rec))
    return p


def test_expired_ttl_is_reaped(repo):
    import time as _t
    p = _write_record(repo, acquired_monotonic_wall=_t.time() - 100,
                      ttl_seconds=10)
    assert sf.held(repo) == []
    assert not p.exists(), "an expired freeze must be removed, not just ignored"


def test_dead_pid_on_this_host_is_reaped(repo):
    """The primary reaper: a crashed experiment must not hold the puller for
    the whole TTL. pid liveness is meaningful precisely because this lock is
    LOCAL to the checkout, so the holder is always on this machine."""
    dead = _find_dead_pid()
    p = _write_record(repo, pid=dead)
    assert sf.held(repo) == []
    assert not p.exists()


def _find_dead_pid():
    for candidate in range(300000, 300200):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except OSError:
            continue
    pytest.skip("could not find a provably dead pid")


def test_foreign_host_record_is_not_pid_reaped(repo):
    """NEGATIVE CONTROL for the reaper. A pid from another host means nothing
    locally; checking it would reap a legitimate freeze whenever that number
    happens not to exist here. TTL alone bounds a foreign holder."""
    p = _write_record(repo, host="some-other-box", pid=_find_dead_pid())
    assert len(sf.held(repo)) == 1
    assert p.exists()


def test_corrupt_record_is_reaped(repo):
    """A record that parses as neither aged nor pid-checkable can never be
    adjudicated, so honouring it would wedge the puller permanently."""
    d = sf.freeze_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    bad = d / "bad.json"
    bad.write_text("{not json")
    assert sf.held(repo) == []
    assert not bad.exists()


def test_ttl_env_override(monkeypatch):
    monkeypatch.setenv("REE_SUBSTRATE_FREEZE_TTL_SECONDS", "42")
    assert sf.ttl_seconds() == 42
    monkeypatch.setenv("REE_SUBSTRATE_FREEZE_TTL_SECONDS", "garbage")
    assert sf.ttl_seconds() == sf.DEFAULT_TTL_SECONDS
    monkeypatch.setenv("REE_SUBSTRATE_FREEZE_TTL_SECONDS", "-5")
    assert sf.ttl_seconds() == sf.DEFAULT_TTL_SECONDS


def test_ignore_env_disarms_every_holder(repo, monkeypatch):
    sf.acquire(repo, "R")
    assert len(sf.held(repo)) == 1
    monkeypatch.setenv("REE_SUBSTRATE_FREEZE_IGNORE", "1")
    assert sf.held(repo) == []
    assert sf.pull_deferred_reason(repo) is None


# --------------------------------------------------------------------------
# E. Precision -- defer substrate movement, nothing else
# --------------------------------------------------------------------------

def test_incoming_substrate_change_is_detected(cloned):
    origin, clone = cloned
    (origin / "ree_core" / "agent.py").write_text("X = 2\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "substrate change")
    assert sf.substrate_paths_incoming(clone) is True


def test_incoming_queue_only_change_is_NOT_substrate(cloned):
    """NEGATIVE CONTROL, and the property that makes A2 usable.

    Deferring every pull for a freeze's whole life would also stall
    experiment_queue.json, leaving a worker blind to new work for the length of
    a 20h run. Only genuine substrate movement may defer.
    """
    origin, clone = cloned
    (origin / "experiment_queue.json").write_text('{"items": [1]}\n')
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "queue snapshot")
    assert sf.substrate_paths_incoming(clone) is False


def test_pull_deferred_when_frozen_and_substrate_incoming(cloned):
    origin, clone = cloned
    (origin / "ree_core" / "agent.py").write_text("X = 3\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "substrate change")
    token = sf.acquire(clone, "V3-EXQ-875")
    why = sf.pull_deferred_reason(clone)
    assert why is not None
    assert "V3-EXQ-875" in why
    assert why.isascii(), "runner output must be ASCII (Windows cp1252)"
    sf.release(token)


def test_pull_NOT_deferred_when_frozen_but_only_queue_incoming(cloned):
    """NEGATIVE CONTROL: frozen, but nothing substrate-y inbound -> pull runs,
    so the queue stays fresh under a long run."""
    origin, clone = cloned
    (origin / "experiment_queue.json").write_text('{"items": [2]}\n')
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "queue snapshot")
    token = sf.acquire(clone, "V3-EXQ-875")
    assert sf.pull_deferred_reason(clone) is None
    sf.release(token)


def test_pull_NOT_deferred_when_nothing_held(cloned):
    """NEGATIVE CONTROL: substrate IS inbound, but no run is in flight."""
    origin, clone = cloned
    (origin / "ree_core" / "agent.py").write_text("X = 4\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "substrate change")
    assert sf.pull_deferred_reason(clone) is None


def test_unheld_check_makes_no_git_calls(repo, monkeypatch):
    """The unflagged/unfrozen path must stay cheap: no fetch, no subprocess.
    A per-tick fetch on every worker for a feature nobody enabled would be a
    real cost regression."""
    called = []
    monkeypatch.setattr(sf, "_run_git",
                        lambda *a, **k: called.append(a) or None)
    assert sf.pull_deferred_reason(repo) is None
    assert called == [], "no git subprocess may run when nothing is frozen"


# --------------------------------------------------------------------------
# F. Fail-open
# --------------------------------------------------------------------------

def test_no_upstream_does_not_defer(repo):
    """A checkout with no upstream cannot be reasoned about -> pull anyway."""
    token = sf.acquire(repo, "R")
    assert sf.substrate_paths_incoming(repo) is None
    assert sf.pull_deferred_reason(repo) is None
    sf.release(token)


def test_git_failure_does_not_defer(cloned, monkeypatch):
    origin, clone = cloned
    (origin / "ree_core" / "agent.py").write_text("X = 5\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "substrate change")
    token = sf.acquire(clone, "R")
    monkeypatch.setattr(sf, "_run_git", lambda *a, **k: None)
    assert sf.substrate_paths_incoming(clone) is None
    assert sf.pull_deferred_reason(clone) is None
    sf.release(token)


def test_unreadable_freeze_state_does_not_defer(repo, monkeypatch):
    def boom(*a, **k):
        raise OSError("nope")
    monkeypatch.setattr(sf, "freeze_dir", boom)
    assert sf.held(repo) == []
    assert sf.pull_deferred_reason(repo) is None


def test_acquire_on_unwritable_root_returns_none(repo, monkeypatch):
    """Acquiring must never be able to stop an experiment from running."""
    def boom(*a, **k):
        raise OSError("read-only")
    monkeypatch.setattr(Path, "mkdir", boom)
    assert sf.acquire(repo, "R") is None


# --------------------------------------------------------------------------
# G. Wiring -- both halves really reachable from experiment_runner
# --------------------------------------------------------------------------

_RUNNER_SRC = (_REE_V3 / "experiment_runner.py").read_text()
_RUNNER_AST = ast.parse(_RUNNER_SRC)


def test_runner_imports_substrate_freeze():
    import experiment_runner
    assert experiment_runner._sfreeze is not None, (
        "substrate_freeze failed to import into the runner")


def test_runner_freeze_guard_is_inert_without_the_module(monkeypatch):
    """NEGATIVE CONTROL: an older checkout without substrate_freeze.py must
    behave exactly as before, not fail."""
    import experiment_runner
    monkeypatch.setattr(experiment_runner, "_sfreeze", None)
    assert experiment_runner._substrate_freeze_blocked("t") is False
    with experiment_runner._substrate_freeze("V3-EXQ-000") as token:
        assert token is None


def test_pull_ree_v3_consults_the_freeze_guard():
    """Without this call the puller half is dead code and the mutex is
    one-sided."""
    fn = next(n for n in ast.walk(_RUNNER_AST)
              if isinstance(n, ast.FunctionDef) and n.name == "_pull_ree_v3")
    names = {c.func.id for c in ast.walk(fn)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "_substrate_freeze_blocked" in names


def test_run_experiment_call_is_wrapped_in_the_freeze():
    """The load-bearing wiring assertion. Without this the ACQUIRING half never
    runs, every freeze set is empty, and the whole feature is silently inert
    while every unit test above still passes.
    """
    wrapped = False
    for node in ast.walk(_RUNNER_AST):
        if not isinstance(node, ast.With):
            continue
        ctxs = {i.context_expr.func.id for i in node.items
                if isinstance(i.context_expr, ast.Call)
                and isinstance(i.context_expr.func, ast.Name)}
        if "_substrate_freeze" not in ctxs:
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "run_experiment"):
                wrapped = True
    assert wrapped, (
        "the run_experiment() call must sit inside `with _substrate_freeze(...)`"
        " -- otherwise no run ever acquires a freeze")


def test_runner_freeze_output_is_ascii():
    """ASCII-only printed output (CLAUDE.md: cp1252 mojibake on Windows)."""
    for node in ast.walk(_RUNNER_AST):
        if (isinstance(node, ast.FunctionDef)
                and node.name in ("_substrate_freeze", "_substrate_freeze_blocked")):
            seg = ast.get_source_segment(_RUNNER_SRC, node) or ""
            assert seg.isascii(), f"{node.name} contains non-ASCII output"


# --------------------------------------------------------------------------
# H. End-to-end -- the substrate really does not move under a held freeze
# --------------------------------------------------------------------------

def test_end_to_end_frozen_pull_leaves_substrate_bytes_unchanged(cloned, monkeypatch):
    """The whole point of A2, asserted on actual file bytes rather than on the
    guard's return value.

    Replays V3-EXQ-875's shape: a run is in flight, a `ree_core` commit lands on
    origin, the puller ticks. The working-tree file the running subprocess would
    be importing must be byte-identical afterwards.
    """
    import experiment_runner as er
    origin, clone = cloned
    monkeypatch.setattr(er, "REPO_ROOT", clone)
    before = (clone / "ree_core" / "agent.py").read_bytes()

    (origin / "ree_core" / "agent.py").write_text("X = 999\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "substrate lands mid-run")

    token = sf.acquire(clone, "V3-EXQ-875")
    try:
        assert er._pull_ree_v3("test tick") is False, "pull should be deferred"
        assert (clone / "ree_core" / "agent.py").read_bytes() == before
    finally:
        sf.release(token)

    # And once the run finishes, the very next tick picks the change up --
    # the freeze DEFERS sync, it must never permanently strand a worker.
    assert er._pull_ree_v3("post-run tick") is True
    assert (clone / "ree_core" / "agent.py").read_bytes() != before


def test_end_to_end_unfrozen_pull_still_applies_substrate(cloned, monkeypatch):
    """NEGATIVE CONTROL for the test above: with no freeze held, the same pull
    must apply the substrate change exactly as it does today. This is what
    proves the guard is not simply breaking ree-v3 sync outright."""
    import experiment_runner as er
    origin, clone = cloned
    monkeypatch.setattr(er, "REPO_ROOT", clone)
    before = (clone / "ree_core" / "agent.py").read_bytes()
    (origin / "ree_core" / "agent.py").write_text("X = 1234\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "substrate change")
    assert er._pull_ree_v3("test tick") is True
    assert (clone / "ree_core" / "agent.py").read_bytes() != before
