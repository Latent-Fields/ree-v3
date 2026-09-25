"""Contract tests: the commit gate stops its remote run when it stops waiting for it.

REE_Working docs/reference/commit_latency_diagnosis_20260925.md P5 (second half) and
remote_pytest.sh's "A DEAD CALLER STOPS ITS REMOTE RUN" block (REE_Working 2568d2b4b).

The gate (scripts/precommit_contracts.sh Block 2) runs the fleet router in a background
SUBSHELL -- `( cd ... && remote_pytest.sh ...; echo $? > rc ) &` -- and its EXIT trap
(cleanup_race) used to `kill` only that subshell. The wrapper is the subshell's CHILD, so it
was orphaned and its detached suite ran on with nobody reading the result: after a local race
win, after a gate killed mid-run, and on 2026-09-25 12:08-12:41 on the hub (the orphaned nested
gate, sec 2 of the diagnosis). remote_pytest.sh now stops its remote run BY RUN DIR on TERM and
when a registered caller dies; the gate has to (a) signal the wrapper itself and (b) register
its own pid, and it must never let the router credit the validation cache for the gate's own
staged-tree run (the gate records for itself).

  L1 local race wins -> the gate's exit TERMs the router process itself (not just the subshell).
  L2 the router runs with REMOTE_PYTEST_CALLER_PID = the gate's own pid (alive while it waits).
  L3 the router runs with REMOTE_PYTEST_CACHE_CREDIT=0 (the gate's own record is the only one).
  L4 NEGATIVE CONTROL: a router that finished normally is not signalled at all.

The router here is a stub that records its env and whether it was TERMed; nothing real runs.
"""

import os
import subprocess
import time
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "scripts" / "precommit_contracts.sh"


def _fake_repo(tmp_path):
    repo = tmp_path / "ree-v3"
    (repo / "ree_core").mkdir(parents=True)
    (repo / "tests" / "contracts").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    for k in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"):
        env.pop(k, None)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True, env=env)
    (repo / "experiments" / "_lib").mkdir(parents=True)
    (repo / "experiments" / "_lib" / "x.py").write_text("# staged\n")
    subprocess.run(["git", "add", "experiments/_lib/x.py"], cwd=repo, check=True, env=env)
    return repo


def _router_stub(tmp_path, sleep_sec):
    envf = tmp_path / "router_env.txt"
    termf = tmp_path / "router_termed.txt"
    pidf = tmp_path / "router_pid.txt"
    stub = tmp_path / "router_stub.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$$" > "{pidf}"\n'
        f'echo "credit=${{REMOTE_PYTEST_CACHE_CREDIT:-<unset>}} '
        f'caller=${{REMOTE_PYTEST_CALLER_PID:-<unset>}}" > "{envf}"\n'
        f"trap 'echo TERM > \"{termf}\"; kill $! 2>/dev/null; exit 143' TERM\n"
        f"sleep {sleep_sec} & wait $!\n"
        "exit 0\n"
    )
    stub.chmod(0o755)
    return stub, envf, termf, pidf


def _local_stub(tmp_path, sleep_sec):
    stub = tmp_path / "local_stub.sh"
    stub.write_text(f"#!/usr/bin/env bash\nsleep {sleep_sec}\nexit 0\n")
    stub.chmod(0o755)
    return stub


def _gate_env(tmp_path, router, local, race_after_sec):
    env = {**os.environ}
    for k in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"):
        env.pop(k, None)
    for k in list(env):
        if k.startswith("REE_PRECOMMIT_") or k.startswith("REMOTE_PYTEST_"):
            del env[k]
    env.update({
        "REE_PRECOMMIT_CONTRACTS_TARGET": "remote",
        "REE_PRECOMMIT_REMOTE_PYTEST": str(router),
        "REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST": str(local),
        "REE_PRECOMMIT_CONTRACTS_REMOTE_RACE_AFTER_SEC": str(race_after_sec),
        "REE_PRECOMMIT_CONTRACTS_RACE_POLL_SEC": "0.1",
        "REE_PRECOMMIT_CONTRACTS_FREE_MB": "99999",
        "REE_PRECOMMIT_CONTRACTS_RACE_LOCK_DIR": str(tmp_path / "race.lock"),
        "REE_PRECOMMIT_VALIDATION_CACHE_DISABLE": "1",
    })
    return env


def _wait_for(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return pred()


def _kill_leftover(pidf):
    try:
        pid = int(pidf.read_text().strip())
        os.kill(pid, 9)
    except (OSError, ValueError):
        pass


def test_l1_local_win_terms_the_router_itself(tmp_path):
    repo = _fake_repo(tmp_path)
    router, envf, termf, pidf = _router_stub(tmp_path, sleep_sec=30)
    local = _local_stub(tmp_path, sleep_sec=0.2)
    try:
        p = subprocess.run(["bash", str(GATE)], cwd=repo, capture_output=True, text=True,
                           env=_gate_env(tmp_path, router, local, race_after_sec=0), timeout=60)
        assert p.returncode == 0, p.stderr
        assert "race winner: local" in p.stderr, p.stderr
        assert _wait_for(termf.exists, 5), (
            "the gate exited without signalling the router process -- only its subshell was "
            "killed, so the wrapper (and its detached suite on the box) runs on unread")
    finally:
        _kill_leftover(pidf)


def test_l2_l3_router_env_registers_caller_and_disables_cache_credit(tmp_path):
    repo = _fake_repo(tmp_path)
    router, envf, termf, pidf = _router_stub(tmp_path, sleep_sec=30)
    local = _local_stub(tmp_path, sleep_sec=2)
    gate = subprocess.Popen(["bash", str(GATE)], cwd=repo, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            env=_gate_env(tmp_path, router, local, race_after_sec=0))
    try:
        assert _wait_for(envf.exists, 15), "router stub never started"
        line = envf.read_text().strip()
        assert "credit=0" in line, line
        caller = line.split("caller=", 1)[1]
        assert caller == str(gate.pid), (
            "REMOTE_PYTEST_CALLER_PID must be the gate's own pid, got %r (gate %d)"
            % (caller, gate.pid))
        os.kill(int(caller), 0)   # the registered caller is alive while it waits
    finally:
        try:
            gate.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            gate.kill()
        _kill_leftover(pidf)


def test_l4_finished_router_is_not_signalled(tmp_path):
    repo = _fake_repo(tmp_path)
    router, envf, termf, pidf = _router_stub(tmp_path, sleep_sec=0)
    local = _local_stub(tmp_path, sleep_sec=0)
    p = subprocess.run(["bash", str(GATE)], cwd=repo, capture_output=True, text=True,
                       env=_gate_env(tmp_path, router, local, race_after_sec=30), timeout=60)
    assert p.returncode == 0, p.stderr
    assert "race winner: remote" in p.stderr, p.stderr
    time.sleep(0.5)
    assert not termf.exists(), "a router that already finished must not be signalled"


def test_l5_source_pins():
    src = GATE.read_text()
    assert "REMOTE_PYTEST_CALLER_PID=$$" in src
    assert "REMOTE_PYTEST_CACHE_CREDIT=0" in src
    assert 'pkill -TERM -P "$REMOTE_PID"' in src
