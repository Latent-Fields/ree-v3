#!/usr/bin/env python3
"""REE phone-dispatch executor (runs on the Mac).

Polls the hub dispatch service for `pending` jobs, claims one atomically, runs
`claude -p "<prompt>"` headless in a FRESH git worktree of the job's repo, and
reports the outcome back (which the service turns into a phone push via ntfy).

This runs on the machine that HAS the repo checkout + an authenticated Claude
Code CLI (the Mac). It does ONE job at a time. It never merges or deletes the
worktree -- like a chip, the work is left in a branch for the user to review.

Why a worktree: it mirrors the spawn_task chip model (isolated copy), so a
dispatched session can't disturb the user's working tree.

Env:
  DISPATCH_URL            hub base URL (e.g. http://10.8.0.1:8799)   [required]
  DISPATCH_TOKEN          bearer token for the service                [required]
  DISPATCH_MACHINE        this machine's label (default: hostname)
  DISPATCH_POLL_SECONDS   poll interval (default 20)
  DISPATCH_DEFAULT_CWD    repo to use when a job has no cwd. DEFAULT EMPTY:
                          a job with no cwd FAILS LOUDLY rather than running in
                          a guessed repo. Set it explicitly only if you want the
                          old silent-fallback behaviour (see routing notes below).
  DISPATCH_STRICT_REPO_MATCH
                          "1" (default) pre-flight check: fail a job whose prompt
                          references absolute paths in git repos OTHER than the
                          one its cwd resolves to. "0" disables the check.
  DISPATCH_PATH_ROOT      only paths under this root are considered by the
                          strict check (default /Users/dgolden/REE_Working)
  DISPATCH_WORKTREE_BASE  where worktrees go
                          (default <repo>/../.dispatch-worktrees)
  DISPATCH_LOG_DIR        where run logs go (default ./dispatch-logs)
  DISPATCH_CLAUDE_BIN     claude binary (default "claude")
  DISPATCH_CLAUDE_FLAGS   extra flags, space-split, passed before -p
                          (e.g. "--permission-mode acceptEdits"). Default none.
                          NOTE: a headless `claude -p` may stall on a permission
                          prompt; configure this to match how autonomous you
                          want dispatched sessions to be. See deploy/README.md.
  DISPATCH_JOB_TIMEOUT    seconds before a job is killed (default 3600)
  DISPATCH_KEEP_WORKTREE  "1" keep (default), "0" remove on success
  DISPATCH_ONESHOT        "1" process at most one job then exit (for testing)

Routing (why a job can be REFUSED instead of run)
-------------------------------------------------
A dispatched job runs in a worktree of ONE repo. REE_Working is a *container* of
several independent repos (REE_assembly, ree-v3, ree-v2, ...) -- it is not a
superrepo and tracks none of their files -- so the repo a job runs in has to be
chosen correctly or the work lands nowhere:

  * no cwd            -> we do NOT guess. Silently defaulting sent every
                         cwd-less chip into a REE_assembly worktree, including
                         chips whose prompts only touch ree-v3/ paths that do
                         not exist there (audited 2026-07-18).
  * cwd inside an existing worktree (.claude/worktrees/, .dispatch-worktrees/)
                      -> refused: a worktree holds only its own repo's files,
                         so work in the sibling repos would land nowhere.
  * prompt/cwd repo mismatch
                      -> refused when the prompt's absolute paths live in other
                         repos than cwd's (DISPATCH_STRICT_REPO_MATCH).
  * cwd is the container itself, prompt names a sibling repo by relative path
                      -> refused for the same reason (a container worktree has
                         none of the sibling repos' files).

Refusals fail the job with an actionable summary. Repair a staged job's cwd via
the phone page's "cwd" button or POST /api/set-cwd, then Launch it.
"""
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

URL = os.environ.get("DISPATCH_URL", "").rstrip("/")
TOKEN = os.environ.get("DISPATCH_TOKEN", "")
MACHINE = os.environ.get("DISPATCH_MACHINE", socket.gethostname())
POLL_SECONDS = float(os.environ.get("DISPATCH_POLL_SECONDS", "20"))
# Empty by default ON PURPOSE: a job with no cwd is refused, not guessed.
DEFAULT_CWD = os.environ.get("DISPATCH_DEFAULT_CWD", "").strip()
STRICT_REPO_MATCH = os.environ.get("DISPATCH_STRICT_REPO_MATCH", "1") == "1"
PATH_ROOT = os.environ.get("DISPATCH_PATH_ROOT",
                           "/Users/dgolden/REE_Working").rstrip("/")
# A cwd inside one of these is a worktree, not a repo checkout (see docstring).
WORKTREE_MARKERS = (".claude/worktrees/", ".dispatch-worktrees/")
WORKTREE_BASE = os.environ.get("DISPATCH_WORKTREE_BASE", "")
LOG_DIR = os.environ.get(
    "DISPATCH_LOG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dispatch-logs"))
CLAUDE_BIN = os.environ.get("DISPATCH_CLAUDE_BIN", "claude")
CLAUDE_FLAGS = shlex.split(os.environ.get("DISPATCH_CLAUDE_FLAGS", ""))
JOB_TIMEOUT = float(os.environ.get("DISPATCH_JOB_TIMEOUT", "3600"))
KEEP_WORKTREE = os.environ.get("DISPATCH_KEEP_WORKTREE", "1") == "1"
ONESHOT = os.environ.get("DISPATCH_ONESHOT", "0") == "1"

# Unattended-automation commit identity for the headless `claude -p` session.
# This runs on the Mac, OUTSIDE HSE employment hours-tracking; the clinical-hours
# guard (REE_Working/scripts/clinical_hours_guard.py + the pre-commit hook) blocks
# a PERSONAL-identity commit during clinical hours and tells the caller to set
# REE_OFFDUTY=1 or re-author as the bot. A headless session cannot answer that, so
# its commits author as the bot up front -> the guard never blocks dispatched work
# and personal-authorship provenance stays clean. Name/email MUST match BOT_NAME /
# BOT_EMAIL in clinical_hours_guard.py + scripts/ree_bot_identity.sh. The noreply
# email is the same address the cloud "REE Cloud Worker" phase3-* writers use.
# Scoped to the `claude -p` subprocess env only (run_claude) -- NOT exported
# globally, so it never touches the operator's interactive shell.
BOT_GIT_NAME = os.environ.get("DISPATCH_BOT_GIT_NAME", "REE Automation (Mac)")
BOT_GIT_EMAIL = os.environ.get(
    "DISPATCH_BOT_GIT_EMAIL", "nooarche@users.noreply.github.com")


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    sys.stdout.write("[executor] %s %s\n" % (now_iso(), msg))
    sys.stdout.flush()


# --------------------------------------------------------------------------
# hub HTTP client
# --------------------------------------------------------------------------
def _req(path, method="GET", body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method)
    req.add_header("Authorization", "Bearer " + TOKEN)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return exc.code, {}
    except (urllib.error.URLError, socket.timeout, ConnectionError) as exc:
        return None, {"error": str(exc)}


def fetch_pending():
    code, data = _req("/api/pending")
    if code != 200:
        return []
    return data.get("jobs", [])


def claim(job_id):
    code, data = _req("/api/claim", "POST", {"id": job_id, "machine": MACHINE})
    return code == 200, data


def update(job_id, status, **fields):
    payload = {"id": job_id, "status": status}
    payload.update(fields)
    return _req("/api/update", "POST", payload)


# --------------------------------------------------------------------------
# git worktree + claude run
# --------------------------------------------------------------------------
def _run(cmd, cwd=None, timeout=None, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def _bot_git_env():
    """Process env with GIT_{AUTHOR,COMMITTER}_{NAME,EMAIL} pinned to the bot.

    Inherits the current environment and overrides only the four git-identity
    vars, so the dispatched `claude -p` session's commits are bot-authored while
    everything else (PATH, auth, etc.) is preserved.
    """
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = BOT_GIT_NAME
    env["GIT_AUTHOR_EMAIL"] = BOT_GIT_EMAIL
    env["GIT_COMMITTER_NAME"] = BOT_GIT_NAME
    env["GIT_COMMITTER_EMAIL"] = BOT_GIT_EMAIL
    return env


def repo_root_for(cwd):
    """git repo root containing `cwd`, or None. No fallback -- see resolve_repo."""
    if not cwd:
        return None
    res = _run(["git", "-C", cwd, "rev-parse", "--show-toplevel"])
    if res.returncode != 0:
        return None
    return res.stdout.strip()


PATH_RE = None


def prompt_repo_paths(prompt):
    """Absolute paths under PATH_ROOT that a prompt references, deepest-first.

    Only used to VALIDATE a cwd the human chose -- never to pick one.
    """
    global PATH_RE  # noqa: PLW0603  compiled once, cheap
    if PATH_RE is None:
        PATH_RE = re.compile(re.escape(PATH_ROOT) + r"/[A-Za-z0-9._/\-]+")
    seen = []
    for m in PATH_RE.finditer(prompt or ""):
        p = m.group(0).rstrip(".,;:)`'\"")
        if p not in seen:
            seen.append(p)
    return seen


def _repo_of_path(path):
    """Repo root for `path`, walking up to the nearest EXISTING ancestor.

    A prompt may name a file that does not exist yet (something to create), so
    resolve the deepest existing directory on its way up.
    """
    cur = path
    while cur.startswith(PATH_ROOT) and len(cur) > len(PATH_ROOT):
        if os.path.isdir(cur):
            root = repo_root_for(cur)
            return root
        cur = os.path.dirname(cur)
    return None


def prompt_repo_mismatch(prompt, repo):
    """Error string if the prompt's paths all live in OTHER repos than `repo`.

    Returns None when the check passes, is disabled, or is inconclusive (no
    resolvable paths) -- it only refuses on positive evidence of a mismatch.
    """
    if not STRICT_REPO_MATCH:
        return None
    repos = []
    for p in prompt_repo_paths(prompt):
        r = _repo_of_path(p)
        if r and r not in repos:
            repos.append(r)
    if not repos or repo in repos:
        return None
    return ("prompt targets %s but cwd resolves to %s -- set the job's cwd to "
            "the repo the work belongs in (POST /api/set-cwd), then Launch"
            % (", ".join(repos), repo))


SIBLING_REPOS = None


def sibling_repos():
    """Names of git repos nested directly under PATH_ROOT (REE_assembly, ree-v3...).

    PATH_ROOT is a *container* directory, not a superrepo: it tracks none of
    their files. So a worktree of the container has none of them either.
    """
    global SIBLING_REPOS  # noqa: PLW0603  discovered once per process
    if SIBLING_REPOS is None:
        names = []
        try:
            for name in sorted(os.listdir(PATH_ROOT)):
                d = os.path.join(PATH_ROOT, name)
                if os.path.isdir(os.path.join(d, ".git")):
                    names.append(name)
        except OSError:
            pass
        SIBLING_REPOS = names
    return SIBLING_REPOS


def container_repo_mismatch(prompt, repo):
    """Error if cwd is the container repo but the prompt targets a sibling repo.

    Catches prompts written with repo-RELATIVE paths ("REE_assembly/docs/x.md"),
    which the absolute-path check cannot see. Running those in a container
    worktree silently lands the work nowhere.
    """
    if not STRICT_REPO_MATCH or os.path.realpath(repo) != os.path.realpath(PATH_ROOT):
        return None
    hits = [n for n in sibling_repos() if (n + "/") in (prompt or "")]
    if not hits:
        return None
    return ("cwd is the container %s, but the prompt targets %s -- a worktree "
            "of the container holds none of their files. Set the job's cwd to "
            "the repo the work belongs in (POST /api/set-cwd), then Launch"
            % (PATH_ROOT, ", ".join(hits)))


def resolve_repo(job):
    """(repo_root, error). Refuses rather than guessing -- see module docstring."""
    cwd = (job.get("cwd") or "").strip() or DEFAULT_CWD
    if not cwd:
        return None, ("job has no cwd and DISPATCH_DEFAULT_CWD is unset -- "
                      "refusing to guess a repo. Set the job's cwd (POST "
                      "/api/set-cwd) to the repo this work belongs in, then "
                      "Launch it.")
    if not os.path.isdir(cwd):
        return None, "cwd does not exist: %s" % cwd
    norm = os.path.abspath(cwd) + "/"
    for marker in WORKTREE_MARKERS:
        if marker in norm:
            return None, ("cwd %s is inside a git worktree (%s) -- a worktree "
                          "carries only its own repo's files, so work in the "
                          "sibling repos would land nowhere. Use the real repo "
                          "checkout." % (cwd, marker.rstrip("/")))
    repo = repo_root_for(cwd)
    if not repo:
        return None, "no git repo at cwd=%s" % cwd
    prompt = job.get("prompt") or ""
    for check in (prompt_repo_mismatch, container_repo_mismatch):
        err = check(prompt, repo)
        if err:
            return None, err
    return repo, None


def make_worktree(repo, job_id):
    base = WORKTREE_BASE or os.path.join(os.path.dirname(repo),
                                         ".dispatch-worktrees")
    os.makedirs(base, exist_ok=True)
    wt = os.path.join(base, "dispatch-" + job_id)
    branch = "dispatch/" + job_id
    res = _run(["git", "-C", repo, "worktree", "add", "-b", branch, wt, "HEAD"])
    if res.returncode != 0:
        return None, None, res.stderr.strip()
    return wt, branch, None


def remove_worktree(repo, wt):
    _run(["git", "-C", repo, "worktree", "remove", "--force", wt])


def run_claude(prompt, cwd, log_path):
    """Run `claude -p` headless. Returns (exit_code, summary, log_tail)."""
    cmd = [CLAUDE_BIN] + CLAUDE_FLAGS + ["-p", prompt,
                                         "--output-format", "json"]
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("CMD: %s\nCWD: %s\nSTART: %s\n\n" % (
            " ".join(shlex.quote(c) for c in cmd[:2] + ["-p", "<prompt>"]),
            cwd, now_iso()))
        lf.flush()
        try:
            res = _run(cmd, cwd=cwd, timeout=JOB_TIMEOUT, env=_bot_git_env())
        except subprocess.TimeoutExpired:
            lf.write("\nTIMEOUT after %ss\n" % JOB_TIMEOUT)
            return 124, "timed out after %ss" % int(JOB_TIMEOUT), "timeout"
        lf.write(res.stdout or "")
        if res.stderr:
            lf.write("\n--- stderr ---\n" + res.stderr)
    summary = _summarize(res.stdout, res.stderr, res.returncode)
    tail = (res.stdout or res.stderr or "")[-500:]
    return res.returncode, summary, tail


def _summarize(stdout, stderr, code):
    # claude -p --output-format json emits a single JSON object with a
    # "result" field (the final assistant text) and "is_error"/"subtype".
    out = (stdout or "").strip()
    if out:
        try:
            obj = json.loads(out)
            if isinstance(obj, dict):
                txt = obj.get("result") or obj.get("subtype") or ""
                if obj.get("is_error"):
                    return ("error: " + str(txt))[:280]
                return (str(txt) or "completed")[:280]
        except ValueError:
            return out[:280]
    if code != 0 and stderr:
        return ("error: " + stderr.strip())[:280]
    return "completed (exit %d)" % code


# --------------------------------------------------------------------------
# one job
# --------------------------------------------------------------------------
def process(job):
    job_id = job["id"]
    ok, _ = claim(job_id)
    if not ok:
        return False  # someone else got it / race
    log("claimed %s (%s)" % (job_id, job.get("title") or ""))
    repo, err = resolve_repo(job)
    if repo is None:
        update(job_id, "failed", exit_code=3,
               summary=("routing refused: " + err)[:280])
        log("REFUSED %s: %s" % (job_id, err))
        return True
    wt, branch, err = make_worktree(repo, job_id)
    if wt is None:
        update(job_id, "failed", exit_code=2,
               summary="worktree create failed: %s" % (err or "")[:200])
        log("FAILED %s: worktree" % job_id)
        return True
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "dispatch-%s.log" % job_id)
    update(job_id, "running", summary="worktree %s (branch %s)" % (wt, branch))
    log("running %s in %s" % (job_id, wt))
    code, summary, tail = run_claude(job["prompt"], wt, log_path)
    status = "done" if code == 0 else "failed"
    full_summary = "%s | branch %s | log %s" % (summary, branch, log_path)
    update(job_id, status, exit_code=code, summary=full_summary[:280],
           log_tail=tail)
    if status == "done" and not KEEP_WORKTREE:
        remove_worktree(repo, wt)
    log("%s %s (exit %d)" % (status.upper(), job_id, code))
    return True


def main():
    if not URL or not TOKEN:
        sys.stderr.write(
            "[executor] DISPATCH_URL and DISPATCH_TOKEN are required.\n")
        sys.exit(2)
    log("executor up: url=%s machine=%s poll=%ss claude=%s" % (
        URL, MACHINE, POLL_SECONDS, CLAUDE_BIN))
    while True:
        try:
            pending = fetch_pending()
        except Exception as exc:  # noqa: BLE001
            log("poll error: %s" % exc)
            pending = []
        did = False
        for job in pending:
            if process(job):
                did = True
                break  # one at a time; re-poll fresh after a job
        if ONESHOT:
            log("oneshot done (processed=%s)" % did)
            return
        time.sleep(POLL_SECONDS if not did else 1.0)


if __name__ == "__main__":
    main()
