"""Contract tests for precommit_contracts.sh Block 2 ROUTING (2026-07-29).

The full contract suite must not run on the shared 8GB Mac (OOM hazard). Block 2 now routes:
memory-gated local (Mac only with a real margin) vs delegate to the fleet router
remote_pytest.sh. Coverage is unchanged (the full suite runs either way) -- these tests pin the
DECISION and the fail-safe, driving the gate as a subprocess with a stub router and a fake
ree-v3 repo, so nothing real (pytest / a worker) runs and the tests are deterministic on any OS.

  R1 self-gate: no ree_core/ or experiments/_lib/ path staged -> exit 0, no routing at all.
  R2 auto + a loaded Mac (low available MB) -> target=remote.
  R3 auto + a flush Mac (high available MB) -> target=local.
  R4 remote path invokes the router and PASSES its exit through on green (exit 0).
  R5 remote path BLOCKS the commit on a red router run (exit 2).
  R6 FAIL-SAFE: remote chosen but the router is missing -> fall back to local, never skip.
  R7 the header documents routing (guards against a silent revert to unconditional local pytest).

STAGGERED LOCAL-RACE FALLBACK (2026-08-01): when remote is chosen but is still running after
RACE_AFTER seconds, the gate re-checks the SAME memory floor and, if it clears and no other
session holds the cross-session lock, ALSO starts a local run -- whichever finishes first decides
the commit. Confirmed live 2026-08-01: a hub run nice'd behind a real experiment took 2h+ against
a ~13min clean baseline (not the ~2x remote_pytest.sh's own advisory assumes), while a same-tree
local run finished in 13m53s.

  R8  remote finishes before the race deadline -> local is never started, even with a
      near-zero race window (the REMOTE_RC check comes before the deadline check).
  R9  remote outlives the race window, Mac clears the floor -> local starts, wins, and its
      (green) exit code decides the commit; "race winner: local" is logged.
  R10 remote outlives the race window, Mac is BELOW the floor -> local is never started;
      the commit waits on (and is decided by) remote alone.
  R11 remote outlives the race window, Mac clears the floor, but another session already
      holds the race lock -> local is never started, and the pre-existing lock is left
      untouched (this session must not release a lock it did not acquire).
  R12 a RED local win still blocks the commit (exit 2), same as a red remote run.
  R13 REE_PRECOMMIT_CONTRACTS_DISABLE_RACE=1 -> local is never started even with an
      immediately-expired race window; old block-on-remote-only behaviour.
  R14 a race lock older than RACE_LOCK_MAX_MIN is STOLEN, not wedged forever -- the
      recovery a session that dies without its EXIT trap running (SIGKILL / OOM) needs.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[2] / "scripts" / "precommit_contracts.sh"


def _fake_repo(tmp_path, staged_rel):
    """A minimal repo that passes is_ree_v3_repo, with `staged_rel` staged."""
    repo = tmp_path / "ree-v3"
    (repo / "ree_core").mkdir(parents=True)
    (repo / "tests" / "contracts").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True, env=env)
    target = repo / staged_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# staged\n")
    subprocess.run(["git", "add", staged_rel], cwd=repo, check=True, env=env)
    return repo


def _run(repo, extra_env, decide_only=True):
    env = {**os.environ}
    # Isolate git so the fake repo's toplevel is what the gate resolves.
    env.pop("GIT_INDEX_FILE", None)
    env.pop("GIT_DIR", None)
    # Isolate the gate's own knobs: an ambient REE_PRECOMMIT_* in the outer
    # (test-runner) shell must not leak into the subprocess under test -- each
    # test's `extra_env` is the sole source of truth for what the gate sees.
    # Without this, e.g. a developer/CI shell exporting
    # REE_PRECOMMIT_CONTRACTS_TARGET=local to force a local run corrupts every
    # test asserting "TARGET unset" auto-routing (R2/R3), since os.environ's
    # ambient value silently substitutes for the unset precondition.
    for key in list(env):
        if key.startswith("REE_PRECOMMIT_"):
            del env[key]
    if decide_only:
        env["REE_PRECOMMIT_CONTRACTS_DECIDE_ONLY"] = "1"
    env.update(extra_env)
    p = subprocess.run(["bash", str(GATE)], cwd=repo, capture_output=True, text=True, env=env)
    return p


def _stub_router(tmp_path, exit_code):
    """A stand-in remote_pytest.sh: records that it ran, then exits `exit_code`."""
    marker = tmp_path / "router_ran"
    stub = tmp_path / "remote_pytest_stub.sh"
    stub.write_text(f'#!/usr/bin/env bash\necho "STUB ROUTER args: $*" >&2\ntouch "{marker}"\nexit {exit_code}\n')
    stub.chmod(0o755)
    return stub, marker


def _stub_sleeper(tmp_path, name, exit_code, sleep_sec=0.0):
    """A stand-in for either the remote router or `run_local_pytest` that
    records it ran, sleeps `sleep_sec` (to simulate "still running" past the
    race deadline), then exits `exit_code`."""
    marker = tmp_path / f"{name}_ran"
    stub = tmp_path / f"{name}_stub.sh"
    stub.write_text(
        f'#!/usr/bin/env bash\ntouch "{marker}"\nsleep {sleep_sec}\nexit {exit_code}\n'
    )
    stub.chmod(0o755)
    return stub, marker


# --------------------------------------------------------------------------- R1
def test_r1_self_gate_no_relevant_path(tmp_path):
    repo = _fake_repo(tmp_path, "README.md")     # not ree_core/ or _lib/
    p = _run(repo, {}, decide_only=False)
    assert p.returncode == 0, p.stderr
    assert "target=" not in p.stderr, "self-gate should not reach the routing decision"


# --------------------------------------------------------------------------- R2
def test_r2_auto_loaded_mac_routes_remote(tmp_path):
    repo = _fake_repo(tmp_path, "experiments/_lib/x.py")
    p = _run(repo, {"REE_PRECOMMIT_CONTRACTS_FREE_MB": "100"})   # 100MB << 3000 floor
    assert p.returncode == 0, p.stderr
    assert "target=remote" in p.stderr, p.stderr


# --------------------------------------------------------------------------- R3
def test_r3_auto_flush_mac_routes_local(tmp_path):
    repo = _fake_repo(tmp_path, "ree_core/y.py")
    p = _run(repo, {"REE_PRECOMMIT_CONTRACTS_FREE_MB": "99999"})   # well above floor
    assert p.returncode == 0, p.stderr
    assert "target=local" in p.stderr, p.stderr


# --------------------------------------------------------------------------- R4
def test_r4_remote_passes_through_green(tmp_path):
    repo = _fake_repo(tmp_path, "experiments/_lib/x.py")
    stub, marker = _stub_router(tmp_path, 0)
    p = _run(repo, {"REE_PRECOMMIT_CONTRACTS_TARGET": "remote",
                    "REE_PRECOMMIT_REMOTE_PYTEST": str(stub)}, decide_only=False)
    assert marker.exists(), "router was not invoked"
    assert p.returncode == 0, p.stderr


# ------------------------------------------------------------------------ R4b
def test_r4b_direct_local_path_invokes_run_local_pytest_override(tmp_path):
    """TARGET=local chosen from the start (no race involved) must also honour
    the REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST override, and pass its exit
    through -- the non-race counterpart to R4."""
    repo = _fake_repo(tmp_path, "ree_core/y.py")
    stub, marker = _stub_sleeper(tmp_path, "local", 0, sleep_sec=0.0)
    p = _run(repo, {"REE_PRECOMMIT_CONTRACTS_TARGET": "local",
                    "REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST": str(stub)}, decide_only=False)
    assert marker.exists(), "local override was not invoked"
    assert p.returncode == 0, p.stderr


# --------------------------------------------------------------------------- R5
def test_r5_remote_blocks_on_red(tmp_path):
    repo = _fake_repo(tmp_path, "experiments/_lib/x.py")
    stub, marker = _stub_router(tmp_path, 2)
    p = _run(repo, {"REE_PRECOMMIT_CONTRACTS_TARGET": "remote",
                    "REE_PRECOMMIT_REMOTE_PYTEST": str(stub)}, decide_only=False)
    assert marker.exists(), "router was not invoked"
    assert p.returncode == 2, f"a red router run must BLOCK the commit (got {p.returncode})\n{p.stderr}"


# --------------------------------------------------------------------------- R6
def test_r6_failsafe_missing_router_falls_back_to_local(tmp_path):
    repo = _fake_repo(tmp_path, "experiments/_lib/x.py")
    p = _run(repo, {"REE_PRECOMMIT_CONTRACTS_TARGET": "remote",
                    "REE_PRECOMMIT_REMOTE_PYTEST": str(tmp_path / "does_not_exist.sh")})
    assert p.returncode == 0, p.stderr
    assert "FALLING BACK to local" in p.stderr, p.stderr
    assert "target=local" in p.stderr, "must not skip the gate when the router is missing"


# --------------------------------------------------------------------------- R7
def test_r7_header_documents_routing():
    src = GATE.read_text()
    assert "ROUTING" in src and "remote_pytest" in src, "Block 2 routing header missing"
    assert "OUT-OF-MEMORY" in src or "OOM" in src, "the OOM rationale must stay documented"


def _race_env(tmp_path, remote_stub, local_stub, race_after_sec, free_mb, lock_dir=None):
    return {
        "REE_PRECOMMIT_CONTRACTS_TARGET": "remote",
        "REE_PRECOMMIT_REMOTE_PYTEST": str(remote_stub),
        "REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST": str(local_stub),
        "REE_PRECOMMIT_CONTRACTS_REMOTE_RACE_AFTER_SEC": str(race_after_sec),
        "REE_PRECOMMIT_CONTRACTS_RACE_POLL_SEC": "0.1",
        "REE_PRECOMMIT_CONTRACTS_FREE_MB": str(free_mb),
        "REE_PRECOMMIT_CONTRACTS_RACE_LOCK_DIR": str(lock_dir or (tmp_path / "race.lock")),
    }


# --------------------------------------------------------------------------- R8
def test_r8_fast_remote_never_races_when_it_finishes_within_the_window(tmp_path):
    """REMOTE_RC is checked before the deadline on every poll, so a remote
    that finishes comfortably inside the race window must win outright. Uses
    a non-zero window (2s) with an instant remote (0s) so the test itself
    isn't racing the same clock it's trying to pin -- a literal
    race_after_sec=0 has no buffer for remote's own process-spawn latency
    and is a test-harness flake, not a gate behaviour to assert on."""
    repo = _fake_repo(tmp_path, "experiments/_lib/x.py")
    remote, remote_marker = _stub_sleeper(tmp_path, "remote", 0, sleep_sec=0.0)
    local, local_marker = _stub_sleeper(tmp_path, "local", 0, sleep_sec=0.0)
    p = _run(repo, _race_env(tmp_path, remote, local, race_after_sec=2, free_mb=99999),
              decide_only=False)
    assert p.returncode == 0, p.stderr
    assert remote_marker.exists()
    assert not local_marker.exists(), "local must not run when remote already finished"
    assert "race winner: remote" in p.stderr, p.stderr


# --------------------------------------------------------------------------- R9
def test_r9_slow_remote_races_and_local_wins(tmp_path):
    repo = _fake_repo(tmp_path, "experiments/_lib/x.py")
    remote, remote_marker = _stub_sleeper(tmp_path, "remote", 0, sleep_sec=2.0)
    local, local_marker = _stub_sleeper(tmp_path, "local", 0, sleep_sec=0.1)
    p = _run(repo, _race_env(tmp_path, remote, local, race_after_sec=0, free_mb=99999),
              decide_only=False)
    assert p.returncode == 0, p.stderr
    assert remote_marker.exists()
    assert local_marker.exists(), "local should have been started once the window expired"
    assert "starting a local race" in p.stderr, p.stderr
    assert "race winner: local" in p.stderr, p.stderr


# --------------------------------------------------------------------------- R10
def test_r10_slow_remote_below_floor_never_races(tmp_path):
    repo = _fake_repo(tmp_path, "experiments/_lib/x.py")
    remote, remote_marker = _stub_sleeper(tmp_path, "remote", 0, sleep_sec=0.5)
    local, local_marker = _stub_sleeper(tmp_path, "local", 0, sleep_sec=0.0)
    p = _run(repo, _race_env(tmp_path, remote, local, race_after_sec=0, free_mb=100),
              decide_only=False)
    assert p.returncode == 0, p.stderr
    assert remote_marker.exists()
    assert not local_marker.exists(), "local must not run below the memory floor"
    assert "NOT racing locally" in p.stderr and "mac_available=100MB" in p.stderr, p.stderr
    assert "race winner: remote" in p.stderr, p.stderr


# --------------------------------------------------------------------------- R11
def test_r11_slow_remote_lock_held_by_another_session_never_races(tmp_path):
    repo = _fake_repo(tmp_path, "experiments/_lib/x.py")
    remote, remote_marker = _stub_sleeper(tmp_path, "remote", 0, sleep_sec=0.5)
    local, local_marker = _stub_sleeper(tmp_path, "local", 0, sleep_sec=0.0)
    lock_dir = tmp_path / "race.lock"
    lock_dir.mkdir()  # simulate another session already racing
    p = _run(repo, _race_env(tmp_path, remote, local, race_after_sec=0, free_mb=99999,
                              lock_dir=lock_dir), decide_only=False)
    assert p.returncode == 0, p.stderr
    assert not local_marker.exists(), "local must not run while another session holds the lock"
    assert "another session already racing" in p.stderr, p.stderr
    assert lock_dir.exists(), "must not release a lock this session did not acquire"


# --------------------------------------------------------------------------- R12
def test_r12_red_local_win_blocks_commit(tmp_path):
    repo = _fake_repo(tmp_path, "experiments/_lib/x.py")
    remote, remote_marker = _stub_sleeper(tmp_path, "remote", 0, sleep_sec=2.0)
    local, local_marker = _stub_sleeper(tmp_path, "local", 2, sleep_sec=0.1)
    p = _run(repo, _race_env(tmp_path, remote, local, race_after_sec=0, free_mb=99999),
              decide_only=False)
    assert local_marker.exists()
    assert p.returncode == 2, f"a red local race win must BLOCK the commit (got {p.returncode})\n{p.stderr}"
    assert "race winner: local (rc=2)" in p.stderr, p.stderr


# --------------------------------------------------------------------------- R13
def test_r13_disable_race_never_starts_local(tmp_path):
    repo = _fake_repo(tmp_path, "experiments/_lib/x.py")
    remote, remote_marker = _stub_sleeper(tmp_path, "remote", 0, sleep_sec=0.5)
    local, local_marker = _stub_sleeper(tmp_path, "local", 0, sleep_sec=0.0)
    env = _race_env(tmp_path, remote, local, race_after_sec=0, free_mb=99999)
    env["REE_PRECOMMIT_CONTRACTS_DISABLE_RACE"] = "1"
    p = _run(repo, env, decide_only=False)
    assert p.returncode == 0, p.stderr
    assert not local_marker.exists(), "REE_PRECOMMIT_CONTRACTS_DISABLE_RACE=1 must suppress the race"
    assert "race winner: remote" in p.stderr, p.stderr


# --------------------------------------------------------------------------- R14
def test_r14_stale_lock_is_stolen_not_wedged_forever(tmp_path):
    """A lock left behind by a session that died without its EXIT trap
    running (SIGKILL / OOM) must not permanently disable racing for every
    future session."""
    repo = _fake_repo(tmp_path, "experiments/_lib/x.py")
    remote, remote_marker = _stub_sleeper(tmp_path, "remote", 0, sleep_sec=0.5)
    local, local_marker = _stub_sleeper(tmp_path, "local", 0, sleep_sec=0.0)
    lock_dir = tmp_path / "race.lock"
    lock_dir.mkdir()
    old = time.time() - 120  # 2 minutes old
    os.utime(lock_dir, (old, old))
    env = _race_env(tmp_path, remote, local, race_after_sec=0, free_mb=99999, lock_dir=lock_dir)
    env["REE_PRECOMMIT_CONTRACTS_RACE_LOCK_MAX_MIN"] = "1"   # 2min old >= 1min max -> stale
    p = _run(repo, env, decide_only=False)
    assert p.returncode == 0, p.stderr
    assert "stealing stale race lock" in p.stderr, p.stderr
    assert local_marker.exists(), "the stale lock must have been stolen so local could race"
    assert "race winner: local" in p.stderr, p.stderr
