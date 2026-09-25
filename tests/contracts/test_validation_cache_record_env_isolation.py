"""The validation-cache record must not inherit the OUTER commit's git env.

THE INCIDENT (2026-09-25, chip-20260925-precommit-cache-index-leak-fix;
REE_Working docs/reference/commit_latency_diagnosis_20260925.md R4/P2).
precommit_contracts.sh's record_validation_cache_result runs INSIDE the
pre-commit hook, and git exports the outer commit's GIT_INDEX_FILE -- plus, from
a linked worktree, GIT_DIR -- to that hook. `validation_cache.py record` and the
`ree_commit.py --to-remote-tip` it calls inherited both, so the throwaway-worktree
git calls acted on the LANDER's index and gitdir. Reproduced in a sandbox
(git 2.51.2) before this fix: the lander's staged set was reset out of its
index, its HEAD was moved onto the cache commit, its own `git commit` failed
(rc 128), and the structural re-apply's commit fired the hook again -- the
nested full contract gate that ran on the hub 12:08-12:41 that day with nobody
reading it. The inner commit's message is "contract validation cache: record
pass"; had it carried the lander's index it would have put the lander's whole
build on origin/main under that label.

THE FIX, three layers (each pinned here or in REE_Working's
scripts/test_ree_commit_inherited_git_env.py):
  1. precommit_contracts.sh runs record under `env -u GIT_INDEX_FILE -u GIT_DIR
     -u GIT_WORK_TREE` (the scrub the router call already had);
  2. validation_cache.py `record` scrubs the same variables itself and commits
     the single cache file with --no-verify (no nested hook at all);
  3. ree_commit.py's throwaway-worktree paths REFUSE an inherited env, and its
     structural re-apply commits --no-verify.

WHAT IS ASSERTED.
  Source pins (run everywhere, including the fleet -- no umbrella import):
    s1  the gate's record invocation is prefixed by the env -u scrub;
    s2  validation_cache.py's record scrubs and passes the no-verify kwarg.
  Behavioural (Mac only -- they need REE_Working/scripts, which
  remote_pytest.sh does not stage, so they SKIP on the fleet exactly like
  test_validation_cache_no_local_ref_move.py): a real `git commit` of a
  ree_core build, from the main checkout and from a linked worktree, with and
  without an origin that moved the cache file (the conflict that sends
  ree_commit.py down the structural re-apply path), through a hook that runs
  the gate's OWN record_validation_cache_result function extracted from
  precommit_contracts.sh -- not a restatement of it. Each case asserts:
    * the outer commit succeeds and carries exactly the lander's files;
    * the record still LANDS on origin (non-vacuity), touching exactly the
      cache file;
    * the hook's Block-2 trigger fired exactly once (no nested gate).

FAIL-BEFORE. REE_TEST_R4_GATE / REE_TEST_R4_VALIDATION_CACHE /
REE_TEST_R4_REE_COMMIT point the behavioural cases at other copies (test-only;
defaults are the real files). Pointed at the pre-fix gate, validation_cache.py
and ree_commit.py, every behavioural case FAILED when this file was written --
see the diagnosis doc's "P2/P6 landed" section.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REE_V3 = Path(__file__).resolve().parents[2]
GATE_REAL = REE_V3 / "scripts" / "precommit_contracts.sh"
VALIDATION_CACHE_REAL = REE_V3 / "scripts" / "validation_cache.py"


def _umbrella_scripts():
    """Un-worktreed like validation_cache.py's own resolver, so this also runs
    from a ree-v3 worktree outside REE_Working."""
    try:
        p = subprocess.run(["git", "-C", str(REE_V3), "rev-parse", "--path-format=absolute",
                            "--git-common-dir"], capture_output=True, text=True, timeout=10)
        cand = Path(p.stdout.strip()).parent.parent / "scripts"
        if p.returncode == 0 and (cand / "task_claim.py").is_file():
            return cand
    except Exception:
        pass
    return REE_V3.parent / "scripts"


UMBRELLA_SCRIPTS = _umbrella_scripts()
GATE = Path(os.environ.get("REE_TEST_R4_GATE") or GATE_REAL)
VALIDATION_CACHE = Path(os.environ.get("REE_TEST_R4_VALIDATION_CACHE") or VALIDATION_CACHE_REAL)
REE_COMMIT = Path(os.environ.get("REE_TEST_R4_REE_COMMIT") or UMBRELLA_SCRIPTS / "ree_commit.py")
GATE_PY = "/opt/local/bin/python3" if os.path.exists("/opt/local/bin/python3") else sys.executable
CACHE = ".contract_validation_cache.json"

needs_umbrella_scripts = pytest.mark.skipif(
    not (UMBRELLA_SCRIPTS / "task_claim.py").exists() or not REE_COMMIT.exists(),
    reason="REE_Working/scripts (ree_commit.py, task_claim.py) not present -- "
           "validation_cache.py's cross-repo import cannot resolve here (e.g. a "
           "remote pytest worker, which stages only ree-v3/). The source pins in "
           "this module still run.")


def _record_function_text(gate_path):
    src = Path(gate_path).read_text(encoding="utf-8")
    m = re.search(r"^record_validation_cache_result\(\) \{\n.*?^\}\n", src, re.S | re.M)
    assert m, "could not locate record_validation_cache_result() in %s" % gate_path
    return m.group(0)


# --------------------------------------------------------------------------- pins

def test_s1_gate_scrubs_git_env_on_the_record_call():
    body = _record_function_text(GATE_REAL)
    # Join backslash continuations so the invocation reads as one command.
    logical = re.sub(r"\\\n\s*", " ", body)
    invocations = [ln for ln in logical.splitlines()
                   if '"$VALIDATION_CACHE_PY" record' in ln and not ln.lstrip().startswith("#")]
    assert len(invocations) == 1, invocations
    inv = invocations[0]
    for var in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"):
        assert re.search(r"env\b[^\"]*-u %s\b" % var, inv), (
            "precommit_contracts.sh runs `validation_cache.py record` without "
            "unsetting %s. Inside the pre-commit hook that variable describes the "
            "OUTER commit; the record's nested ree_commit.py then acts on the "
            "lander's index/gitdir (diagnosis R4).\n  invocation: %s" % (var, inv))


def test_s2_validation_cache_record_scrubs_and_skips_the_hook():
    src = VALIDATION_CACHE_REAL.read_text(encoding="utf-8")
    rec = re.search(r"def cmd_record\(args\) -> int:\n(.*?)\n(?=def |\Z)", src, re.S)
    assert rec and "scrub_inherited_git_env()" in rec.group(1), (
        "validation_cache.py cmd_record no longer scrubs the inherited git env "
        "before calling ree_commit.py.")
    scrub = re.search(r"SCRUBBED_GIT_ENV_VARS = \((.*?)\)", src, re.S)
    assert scrub, "SCRUBBED_GIT_ENV_VARS missing"
    for var in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"):
        assert '"%s"' % var in scrub.group(1), var
    call = re.search(r"sha = ree_commit_once\((.*?)\n                \)", src, re.S)
    assert call and "**_NO_VERIFY_KW" in call.group(1), (
        "the record's ree_commit_once call no longer passes the no-verify kwarg: "
        "committing the cache file then fires the pre-commit hook again from "
        "inside the hook -- a nested gate.")
    assert re.search(r'_NO_VERIFY_KW = \(\{"no_verify": True\}', src), (
        "_NO_VERIFY_KW no longer resolves to no_verify=True when supported")


# --------------------------------------------------------------------- behavioural

def _git_env():
    env = {k: v for k, v in os.environ.items()
           if k not in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE")}
    env.update({"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"})
    return env


def _g(cwd, *args):
    p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                       env=_git_env())
    assert p.returncode == 0, "git %s: %s" % (" ".join(args), p.stderr)
    return p.stdout.strip()


def _sandbox(tmp_path, conflict):
    origin = tmp_path / "origin.git"
    repo = tmp_path / "ree-v3"
    _g(tmp_path, "init", "-q", "--bare", str(origin), "-b", "main")
    _g(tmp_path, "init", "-q", str(repo), "-b", "main")
    _g(repo, "config", "user.email", "t@t")
    _g(repo, "config", "user.name", "t")
    (repo / "ree_core").mkdir()
    (repo / "ree_core" / "y.py").write_text("x = 1\n")
    (repo / "README.md").write_text("base\n")
    (repo / CACHE).write_text(json.dumps(
        {"schema_version": "contract_validation_cache/v1", "records": {}},
        indent=2, sort_keys=True) + "\n")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-qm", "base")
    _g(repo, "remote", "add", "origin", str(origin))
    _g(repo, "push", "-q", "origin", "main")
    _g(repo, "fetch", "-q", "origin")
    _g(repo, "remote", "set-head", "origin", "main")
    if conflict:
        other = tmp_path / "other"
        _g(tmp_path, "clone", "-q", str(origin), str(other))
        _g(other, "config", "user.email", "o@o")
        _g(other, "config", "user.name", "o")
        (other / CACHE).write_text(json.dumps(
            {"records": {"zzz": {"result": "pass"}},
             "schema_version": "contract_validation_cache/v1"},
            indent=2, sort_keys=True) + "\n")
        _g(other, "commit", "-qam", "origin moved the cache file")
        _g(other, "push", "-q", "origin", "main")
    return repo, origin


def _install_hook(repo, log):
    """Block 2's trigger + the gate's OWN record function, extracted verbatim."""
    func = _record_function_text(GATE)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"hook cwd=$(pwd) staged=$(git diff --cached --name-only | tr '\\n' ' ')\" >> '%(log)s'\n"
        "git diff --cached --name-only | grep -qE '^(ree_core/|experiments/_lib/)' || exit 0\n"
        "echo BLOCK2 >> '%(log)s'\n"
        "if [ -n \"${R4_TEST_IN_GATE:-}\" ]; then echo NESTED >> '%(log)s'; exit 0; fi\n"
        "export R4_TEST_IN_GATE=1\n"
        "PY='%(py)s'\n"
        "VALIDATION_CACHE_PY='%(vc)s'\n"
        "TOP=\"$(git rev-parse --show-toplevel)\"\n"
        "VALIDATION_CACHE_ARGS=(--repo-root \"$TOP\" --hash-root \"$TOP\" --tier t "
        "--ttl-minutes 45 --session-id r4-contract-test --ree-commit-path '%(rc)s')\n"
        "%(func)s"
        "record_validation_cache_result pass 2>> '%(log)s'\n"
        "exit 0\n" % {"log": log, "py": GATE_PY, "vc": VALIDATION_CACHE,
                      "rc": REE_COMMIT, "func": func})
    hook.chmod(0o755)


@needs_umbrella_scripts
@pytest.mark.parametrize("conflict", [False, True], ids=["origin-quiet", "origin-moved-cache"])
@pytest.mark.parametrize("mode", ["main-checkout", "linked-worktree"])
def test_b1_record_inside_a_real_commit_does_not_touch_the_outer_commit(tmp_path, mode, conflict):
    repo, origin = _sandbox(tmp_path, conflict)
    log = tmp_path / "hook.log"
    _install_hook(repo, log)
    where = repo
    if mode == "linked-worktree":
        where = tmp_path / "lander_wt"
        _g(repo, "worktree", "add", "-q", "--detach", str(where), "main")
    (where / "ree_core" / "y.py").write_text("x = 2  # the lander build\n")
    (where / "ree_core" / "z_new.py").write_text("z = 1\n")
    (where / "README.md").write_text("lander readme\n")
    _g(where, "add", "ree_core/y.py", "ree_core/z_new.py", "README.md")
    base_origin = _g(repo, "rev-parse", "origin/main")

    p = subprocess.run(["git", "commit", "-q", "-m", "LANDER build"], cwd=str(where),
                       capture_output=True, text=True, env=_git_env(), timeout=300)
    hook_log = log.read_text() if log.exists() else ""
    diag = "\n--- commit stdout/stderr:\n%s%s\n--- hook log:\n%s" % (p.stdout, p.stderr, hook_log)

    assert p.returncode == 0, "the lander's own commit failed -- the record moved its HEAD " \
                              "or reset its index under it" + diag
    assert _g(where, "log", "-1", "--format=%s") == "LANDER build", diag
    files = sorted(_g(where, "show", "--name-only", "--format=", "HEAD").split())
    assert files == ["README.md", "ree_core/y.py", "ree_core/z_new.py"], diag
    assert _g(where, "status", "--porcelain", "--untracked-files=no") in ("", "M " + CACHE,
                                                                           " M " + CACHE), diag

    assert hook_log.count("BLOCK2") == 1 and "NESTED" not in hook_log, (
        "the contract gate's trigger fired more than once -- a nested gate" + diag)

    _g(repo, "fetch", "-q", "origin")
    new = _g(repo, "log", "--format=%H %s", "%s..origin/main" % base_origin).splitlines()
    records = [ln.split(" ", 1)[0] for ln in new
               if ln.split(" ", 1)[1].startswith("contract validation cache: record pass")]
    assert records, "the record did not land on origin at all -- this test would " \
                    "then pass for the wrong reason" + diag
    for sha in records:
        touched = _g(repo, "show", "--name-only", "--format=", sha).split()
        assert touched == [CACHE], (
            "a cache-record commit on origin touches %r -- the outer commit's staged "
            "set leaked into it" % touched + diag)
