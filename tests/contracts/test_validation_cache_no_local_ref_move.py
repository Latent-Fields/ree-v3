"""The validation-cache record must never move the LOCAL branch ref.

WHY THIS IS A CONTRACT AND NOT A PREFERENCE
-------------------------------------------
`validation_cache.py record` only ever runs from inside a pre-commit hook
(precommit_contracts.sh Block 2's record_validation_cache_result, reached via
ree-v3/.git/hooks/pre-commit -> pre-commit.local). ree_commit.py's flow is:

    old_head = rev-parse HEAD
    build private index
    run_pre_commit_hook(...)              <-- the gate, and the record, run HERE
    commit-tree <tree> -p old_head
    update-ref <ref> <new> <old_head>     <-- compare-and-swap

so a record that lands an ordinary LOCAL commit advances the branch, makes
`old_head` stale, and the CAS fails with "branch moved under us while building
the commit ... Nothing was committed." The gate rejects the very commit it was
invoked to gate, after paying the full ~13min suite.

It is deterministic, not a race, and specific to ree_commit.py:

  * A cache HIT `exit 0`s before recording, so only the COLD path is affected --
    which is why the defect survived undetected from 2026-08-10.
  * scripts/bash_command_gate.py tags a `ree_commit.py` invocation REE_COMMIT,
    NOT GIT_COMMIT, and .claude/settings.json only runs precommit_contracts.sh
    under GIT_COMMIT. So for a ree_commit.py commit the PreToolUse copy of the
    gate never runs at all, and the ONLY invocation is the in-process one
    between old_head and the CAS. (A plain `git commit` gets the opposite
    treatment: the PreToolUse gate runs first, warms the cache, and the git-hook
    copy then HITs -- which is why this was never seen on that path.)
  * ree_commit.py stages into a PRIVATE index, so Block 2's
    `git diff --cached` trigger only sees the declared paths there. The nested
    ree_commit.py that writes the cache file stages ONLY the cache file, so
    Block 2 does not re-trigger -- that is why production does not recurse
    infinitely, and the fixtures below reproduce that self-gating faithfully.

Confirmed on five historical instances, all with the identical signature (the
cache commit lands and the gated commit lands as its CHILD, after the failure
and a retry that then hit the cache): 55dbe77 -> 031377d (2026-08-27),
0481c1c -> 88287f1, 547f053 -> 492e51f, 12d6839 -> 1a4b6be, 6702d5a -> 775eb55.
Chip: chip-20260827-precommit-cache-self-collision.

This also regresses an invariant the design doc explicitly claims is preserved
("Path-scoped/intended-file commits, CAS ... untouched" --
REE_assembly/docs/architecture/landing_integration_worker_investigation.md sec 7).

THE FIX: `record` routes through ree_commit.py --to-remote-tip, which lands the
commit on origin/<branch> via the already-hardened throwaway-worktree path and
never moves the local ref.

WHY BOTH A SOURCE PIN AND A BEHAVIOURAL TEST. The behavioural tests need
REE_Working/scripts (ree_commit.py, and validation_cache.py's cross-repo
`from task_claim import ...`), which scripts/remote_pytest.sh does not stage --
so they SKIP on the fleet, exactly like
test_precommit_contracts_validation_cache.py. The source pins carry no such
dependency and therefore run everywhere, including the cloud worker that
actually gates a merge. Neither alone is enough: the pins cannot prove the flag
works, and the behavioural tests do not run where the gate runs.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REE_V3 = Path(__file__).resolve().parents[2]
VALIDATION_CACHE = REE_V3 / "scripts" / "validation_cache.py"
GATE = REE_V3 / "scripts" / "precommit_contracts.sh"
UMBRELLA_SCRIPTS = REE_V3.parent / "scripts"
REE_COMMIT = UMBRELLA_SCRIPTS / "ree_commit.py"

needs_umbrella_scripts = pytest.mark.skipif(
    not (UMBRELLA_SCRIPTS / "task_claim.py").exists() or not REE_COMMIT.exists(),
    reason="REE_Working/scripts (ree_commit.py, task_claim.py) not present -- "
           "validation_cache.py's cross-repo import cannot resolve here (e.g. a "
           "remote pytest worker, which stages only ree-v3/). The source pins in "
           "this module still run.")

GATE_PY = "/opt/local/bin/python3" if os.path.exists("/opt/local/bin/python3") else sys.executable


# --------------------------------------------------------------------------- pins
# These read source text only -- no umbrella import -- so they run on the fleet.

def test_s1_record_routes_through_to_remote_tip():
    """The ree_commit_once call must pass to_remote_tip. Without it the record
    lands a local commit and defeats the outer CAS (see module docstring)."""
    src = VALIDATION_CACHE.read_text(encoding="utf-8")
    call = re.search(r"sha = ree_commit_once\((.*?)\n                \)", src, re.S)
    assert call, "could not locate the ree_commit_once( ... ) call in validation_cache.py"
    assert "to_remote_tip=" in call.group(1), (
        "validation_cache.py's ree_commit_once call no longer passes to_remote_tip. "
        "That re-arms the pre-commit self-collision: the record's commit moves "
        "refs/heads/<branch> while ree_commit.py is mid-build, and the outer "
        "compare-and-swap then fails. Do not simplify this flag away.")


def test_s2_remote_tip_defaults_on():
    """Default True, not opt-in. Every production caller goes through
    precommit_contracts.sh, which passes no such flag."""
    src = VALIDATION_CACHE.read_text(encoding="utf-8")
    assert re.search(r"def _write_and_commit_pass\(.*?remote_tip: bool = True", src, re.S), (
        "_write_and_commit_pass's remote_tip parameter must default to True -- "
        "precommit_contracts.sh passes no flag, so an opt-in default would leave "
        "the only production caller unprotected.")


def test_s3_the_gate_still_passes_push_to_record():
    """--to-remote-tip REQUIRES --push (ree_commit.py dies otherwise, and
    ree_commit_once silently downgrades to a plain local commit). So dropping
    --push from the gate would silently restore the collision."""
    gate = GATE.read_text(encoding="utf-8")
    record = re.search(r"record_validation_cache_result\(\) \{(.*?)\n\}", gate, re.S)
    assert record, "could not locate record_validation_cache_result() in the gate"
    # The INVOCATION line specifically, not the function body: the body also
    # carries a comment explaining why --push matters, and matching that instead
    # made an earlier version of this pin vacuous (caught by mutation-testing it).
    invocations = [ln for ln in record.group(1).splitlines()
                   if '"$VALIDATION_CACHE_PY" record' in ln and not ln.lstrip().startswith("#")]
    assert len(invocations) == 1, (
        "expected exactly one validation_cache.py record invocation, found %d:\n%s"
        % (len(invocations), record.group(1)))
    assert "--push" in invocations[0], (
        "precommit_contracts.sh's record call dropped --push. --to-remote-tip "
        "requires it, so without --push the record silently falls back to a LOCAL "
        "commit -- which is exactly the ref move that defeats the outer CAS.\n"
        "  invocation: %s" % invocations[0])


# --------------------------------------------------------------------- behavioural

def _repo_with_origin(tmp_path, name="ree-v3"):
    """A real ree-v3-shaped repo on `main` with a real bare origin. A real remote
    is required: --to-remote-tip pushes via a throwaway worktree, so a fixture
    with no origin could not exercise the path under test at all."""
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    origin = tmp_path / "origin.git"
    repo = tmp_path / name
    subprocess.run(["git", "init", "-q", "--bare", str(origin), "-b", "main"], check=True, env=env)
    subprocess.run(["git", "init", "-q", str(repo), "-b", "main"], check=True, env=env)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "config", k, v], cwd=repo, check=True, env=env)
    (repo / "ree_core").mkdir(parents=True)
    (repo / "README.md").write_text("base\n")
    (repo / "ree_core" / "y.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=repo, check=True, env=env)
    subprocess.run(["git", "push", "-q", "origin", "main"], cwd=repo, check=True, env=env)
    return repo, origin, env


def _install_recording_hook(repo, extra_flags=""):
    """A pre-commit hook that records a cache pass, self-gated on a staged
    ree_core/ path exactly as precommit_contracts.sh Block 2 is. The self-gate is
    load-bearing, not decoration: without it the NESTED ree_commit.py that writes
    the cache file re-enters this hook and recurses forever (measured)."""
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "git diff --cached --name-only | grep -q '^ree_core/' || exit 0\n"
        '"%s" "%s" record --repo-root "%s" --tier t --result pass --push '
        '--session-id contract-test %s >/dev/null 2>&1\n'
        "exit 0\n" % (GATE_PY, VALIDATION_CACHE, repo, extra_flags))
    hook.chmod(0o755)


def _head(repo, env):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, env=env).stdout.strip()


def _outer_ree_commit(repo, env):
    """The commit the gate exists to gate."""
    (repo / "ree_core" / "y.py").write_text("x = 2  # the gated change\n")
    return subprocess.run(
        [GATE_PY, str(REE_COMMIT), "--repo", str(repo), "-m", "gated ree_core change",
         "--bot", "--", "ree_core/y.py"],
        cwd=repo, capture_output=True, text=True, env=env, timeout=300)


@needs_umbrella_scripts
def test_b1_record_leaves_the_local_branch_ref_untouched(tmp_path):
    repo, origin, env = _repo_with_origin(tmp_path)
    before = _head(repo, env)
    p = subprocess.run(
        [GATE_PY, str(VALIDATION_CACHE), "record", "--repo-root", str(repo),
         "--tier", "t", "--result", "pass", "--push", "--session-id", "contract-test"],
        cwd=repo, capture_output=True, text=True, env=env, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "written and committed" in p.stdout, p.stdout + p.stderr
    assert _head(repo, env) == before, (
        "the record moved refs/heads/main. That is the pre-commit self-collision: "
        "run inside ree_commit.py's hook it invalidates the outer CAS.\n" + p.stdout + p.stderr)
    landed = subprocess.run(["git", "ls-remote", "origin", "refs/heads/main"],
                            cwd=repo, capture_output=True, text=True, env=env).stdout.split()[0]
    assert landed != before, "the record must still LAND -- this is reuse, not a no-op"


@needs_umbrella_scripts
def test_b2_a_recording_hook_does_not_defeat_the_outer_ree_commit_cas(tmp_path):
    """The regression itself: the gated commit must survive its own gate."""
    repo, origin, env = _repo_with_origin(tmp_path)
    _install_recording_hook(repo)
    p = _outer_ree_commit(repo, env)
    assert p.returncode == 0, (
        "the outer ree_commit.py failed -- the cache record moved the branch under "
        "it (CAS failed) and the gate rejected the commit it was gating:\n"
        + p.stdout + p.stderr)
    committed = subprocess.run(["git", "show", "HEAD:ree_core/y.py"], cwd=repo,
                               capture_output=True, text=True, env=env).stdout
    assert "the gated change" in committed, (
        "outer ree_commit.py reported success but HEAD does not carry the change:\n"
        + p.stdout + p.stderr)


@needs_umbrella_scripts
def test_b3_negative_control_no_remote_tip_still_reproduces_the_collision(tmp_path):
    """Proves b2 has teeth. --no-remote-tip is the documented escape hatch and
    restores the pre-fix behaviour exactly, so it must still fail here -- if this
    ever passes, b2 is passing for some reason other than the fix and the pair
    has quietly become vacuous."""
    repo, origin, env = _repo_with_origin(tmp_path)
    _install_recording_hook(repo, extra_flags="--no-remote-tip")
    p = _outer_ree_commit(repo, env)
    assert p.returncode != 0, (
        "--no-remote-tip no longer reproduces the collision, so b2 is no longer "
        "evidence the fix works. Re-derive the mechanism before deleting this "
        "test.\n" + p.stdout + p.stderr)
    assert "CAS failed" in (p.stdout + p.stderr), p.stdout + p.stderr
    unchanged = subprocess.run(["git", "show", "HEAD:ree_core/y.py"], cwd=repo,
                               capture_output=True, text=True, env=env).stdout
    assert "x = 1" in unchanged, "the collision should leave the gated change uncommitted"
