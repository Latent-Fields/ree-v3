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
"""

import os
import subprocess
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
