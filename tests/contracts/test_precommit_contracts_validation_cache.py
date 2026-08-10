"""Smoke/integration tests for precommit_contracts.sh Block 2's VALIDATION CACHE
wiring (2026-08-10).

Design record: REE_assembly/docs/architecture/landing_integration_worker_investigation.md
section 6. Unit coverage of the cache's own hashing/TTL/machine-class/fail-open
logic lives in ree-v3/scripts/check_validation_cache.py; THIS file exercises the
real hook script end to end -- the actual bash CHECK/RECORD call sites wired
into Block 2's routing, driven as a subprocess with a fake ree-v3 repo, mirroring
test_precommit_contracts_routing.py's approach (nothing real -- no worker, no
real pytest suite -- runs; only the gate script and validation_cache.py are real).

  V1 a cache MISS (nothing recorded yet) falls straight through to routing,
     completely unchanged from pre-cache behaviour (regression guard).
  V2 a cache HIT skips the suite entirely -- neither the remote router nor
     local pytest is ever invoked -- and exits 0 with the HIT logged.
  V3 a PASSING local run records a cache entry (a later check() call is then a
     HIT) -- the record call site actually fires on green.
  V4 a FAILING local run records NOTHING (a later check() call is still a MISS)
     -- record() no-ops on red, exactly as validation_cache.py's own contract
     requires, and the gate still blocks the commit (exit 2) as before.
  V5 REE_PRECOMMIT_VALIDATION_CACHE_DISABLE=1 skips the whole mechanism: no
     check is attempted, and a passing run does not record anything either.
  V6 the tier string this file seeds with matches the one the gate actually
     uses -- guards the seeding helper against silently drifting out of sync
     with precommit_contracts.sh's own VALIDATION_CACHE_TIER.
"""
import os
import subprocess
import sys
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "scripts" / "precommit_contracts.sh"
REAL_VALIDATION_CACHE = GATE.parent / "validation_cache.py"
# Matches precommit_contracts.sh's own VALIDATION_CACHE_TIER -- pinned by
# test_v6_tier_constant_matches_the_gate below so this cannot silently drift.
TIER = "ree-v3-contracts-full-suite"
# The gate resolves $PY as /opt/local/bin/python3 first (falling back to `command
# -v python3` only when that is absent) -- it does not consult PATH when the
# absolute path exists, so a PATH shim cannot redirect it on this machine. Use
# the SAME interpreter to seed the cache, so compute_machine_class() agrees
# between seed time and check time; see py_shim in
# test_precommit_contracts_gate_scope.py for the general portability caveat
# this sidesteps by hardcoding rather than shimming.
GATE_PY = "/opt/local/bin/python3" if os.path.exists("/opt/local/bin/python3") else sys.executable


def _fake_repo_with_base_commit(tmp_path, staged_rel):
    """Like test_precommit_contracts_routing._fake_repo, but with a real base
    commit. validation_cache.py's record path shells out to ree_commit.py,
    which needs a resolvable HEAD -- the routing tests' unborn-branch fixture
    never provides one because routing never reaches that code path."""
    repo = tmp_path / "ree-v3"
    (repo / "ree_core").mkdir(parents=True)
    (repo / "tests" / "contracts").mkdir(parents=True)
    (repo / "experiments" / "_lib").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True, env=env)
    (repo / "README.md").write_text("base\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True, env=env)
    target = repo / staged_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# staged\n")
    subprocess.run(["git", "add", staged_rel], cwd=repo, check=True, env=env)
    return repo


def _run(repo, extra_env, decide_only=False):
    env = {**os.environ}
    env.pop("GIT_INDEX_FILE", None)
    env.pop("GIT_DIR", None)
    for key in list(env):
        if key.startswith("REE_PRECOMMIT_"):
            del env[key]
    if decide_only:
        env["REE_PRECOMMIT_CONTRACTS_DECIDE_ONLY"] = "1"
    env.update(extra_env)
    return subprocess.run(["bash", str(GATE)], cwd=repo, capture_output=True, text=True, env=env)


def _stub_sleeper(tmp_path, name, exit_code):
    marker = tmp_path / ("%s_ran" % name)
    stub = tmp_path / ("%s_stub.sh" % name)
    stub.write_text('#!/usr/bin/env bash\ntouch "%s"\nexit %d\n' % (marker, exit_code))
    stub.chmod(0o755)
    return stub, marker


def _seed_cache_hit(repo, tier=TIER):
    """Populate a real cache entry for `repo`'s CURRENT on-disk ree_core/ +
    experiments/_lib/ content, via the real validation_cache.py -- exercises
    the actual record path rather than hand-crafting the cache JSON, so the
    seeded entry's key genuinely matches what the gate itself will compute."""
    p = subprocess.run(
        [GATE_PY, str(REAL_VALIDATION_CACHE), "record", "--repo-root", str(repo),
         "--tier", tier, "--result", "pass", "--session-id", "test-seed"],
        cwd=repo, capture_output=True, text=True)
    assert p.returncode == 0, "seed record failed:\n%s\n%s" % (p.stdout, p.stderr)
    assert "written and committed" in p.stdout, p.stdout
    return p


def _base_env(tmp_path):
    """A fast, deterministic env: no race, both stubs green by default,
    generous free memory so routing (when reached) goes local without a
    real worker/wake. Individual tests override what they need.

    REE_PRECOMMIT_VALIDATION_CACHE_PY points at the REAL validation_cache.py
    -- the fake repo's own scripts/ dir is deliberately empty (no copy of it
    inside the fixture), so without this override `[ -f "$VALIDATION_CACHE_PY" ]`
    is false and the whole cache block silently no-ops (correct fail-open
    behaviour for a repo that genuinely lacks the script, but not what these
    tests are exercising)."""
    remote, remote_marker = _stub_sleeper(tmp_path, "remote", 0)
    local, local_marker = _stub_sleeper(tmp_path, "local", 0)
    env = {
        "REE_PRECOMMIT_CONTRACTS_TARGET": "local",
        "REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST": str(local),
        "REE_PRECOMMIT_REMOTE_PYTEST": str(remote),
        "REE_PRECOMMIT_VALIDATION_CACHE_PY": str(REAL_VALIDATION_CACHE),
    }
    return env, remote_marker, local_marker


# --------------------------------------------------------------------------- V1
def test_v1_miss_falls_through_to_routing_unchanged(tmp_path):
    repo = _fake_repo_with_base_commit(tmp_path, "experiments/_lib/x.py")
    env, remote_marker, local_marker = _base_env(tmp_path)
    p = _run(repo, env)
    assert p.returncode == 0, p.stderr
    assert "MISS" in p.stderr, p.stderr
    assert local_marker.exists(), "a miss must still run the suite (local stub invoked)"
    assert not remote_marker.exists(), "TARGET=local must never invoke the remote stub"


# --------------------------------------------------------------------------- V2
def test_v2_hit_skips_the_suite_entirely(tmp_path):
    repo = _fake_repo_with_base_commit(tmp_path, "experiments/_lib/x.py")
    _seed_cache_hit(repo)
    env, remote_marker, local_marker = _base_env(tmp_path)
    p = _run(repo, env)
    assert p.returncode == 0, p.stderr
    assert "validation cache HIT" in p.stderr, p.stderr
    assert not local_marker.exists(), "a HIT must never invoke the local stub"
    assert not remote_marker.exists(), "a HIT must never invoke the remote stub"


# --------------------------------------------------------------------------- V3
def test_v3_a_passing_run_records_a_hit_for_next_time(tmp_path):
    repo = _fake_repo_with_base_commit(tmp_path, "ree_core/y.py")
    env, remote_marker, local_marker = _base_env(tmp_path)  # local stub exits 0
    p = _run(repo, env)
    assert p.returncode == 0, p.stderr
    assert local_marker.exists()

    check = subprocess.run(
        [GATE_PY, str(REAL_VALIDATION_CACHE), "check", "--repo-root", str(repo), "--tier", TIER],
        cwd=repo, capture_output=True, text=True)
    assert check.returncode == 0, "the just-passed run must have recorded a HIT:\n%s\n%s" \
        % (check.stdout, check.stderr)


# --------------------------------------------------------------------------- V4
def test_v4_a_failing_run_records_nothing_and_still_blocks(tmp_path):
    repo = _fake_repo_with_base_commit(tmp_path, "ree_core/y.py")
    remote, remote_marker = _stub_sleeper(tmp_path, "remote", 2)
    local, local_marker = _stub_sleeper(tmp_path, "local", 2)  # local stub exits 2 (red)
    env = {
        "REE_PRECOMMIT_CONTRACTS_TARGET": "local",
        "REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST": str(local),
        "REE_PRECOMMIT_REMOTE_PYTEST": str(remote),
        "REE_PRECOMMIT_VALIDATION_CACHE_PY": str(REAL_VALIDATION_CACHE),
    }
    p = _run(repo, env)
    assert p.returncode == 2, "a red run must still block the commit:\n%s" % p.stderr
    assert local_marker.exists()

    check = subprocess.run(
        [GATE_PY, str(REAL_VALIDATION_CACHE), "check", "--repo-root", str(repo), "--tier", TIER],
        cwd=repo, capture_output=True, text=True)
    assert check.returncode == 1, "a FAILED run must never have recorded a hit:\n%s\n%s" \
        % (check.stdout, check.stderr)


# --------------------------------------------------------------------------- V5
def test_v5_disable_env_skips_the_mechanism_entirely(tmp_path):
    repo = _fake_repo_with_base_commit(tmp_path, "experiments/_lib/x.py")
    _seed_cache_hit(repo)  # would be a HIT if the mechanism ran at all
    env, remote_marker, local_marker = _base_env(tmp_path)
    env["REE_PRECOMMIT_VALIDATION_CACHE_DISABLE"] = "1"
    p = _run(repo, env)
    assert p.returncode == 0, p.stderr
    assert "validation cache HIT" not in p.stderr, p.stderr
    assert "MISS" not in p.stderr, "the cache block must not even attempt a check when disabled"
    assert local_marker.exists(), "disabling the cache must fall through to running the suite"


# --------------------------------------------------------------------------- V6
def test_v6_tier_constant_matches_the_gate():
    src = GATE.read_text()
    assert ('VALIDATION_CACHE_TIER="%s"' % TIER) in src, \
        "this file's TIER must match precommit_contracts.sh's VALIDATION_CACHE_TIER"
