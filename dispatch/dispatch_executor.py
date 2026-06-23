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
  DISPATCH_DEFAULT_CWD    repo to use when a job has no cwd
                          (default /Users/dgolden/REE_Working/REE_assembly)
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
"""
import json
import os
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
DEFAULT_CWD = os.environ.get(
    "DISPATCH_DEFAULT_CWD", "/Users/dgolden/REE_Working/REE_assembly")
WORKTREE_BASE = os.environ.get("DISPATCH_WORKTREE_BASE", "")
LOG_DIR = os.environ.get(
    "DISPATCH_LOG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dispatch-logs"))
CLAUDE_BIN = os.environ.get("DISPATCH_CLAUDE_BIN", "claude")
CLAUDE_FLAGS = shlex.split(os.environ.get("DISPATCH_CLAUDE_FLAGS", ""))
JOB_TIMEOUT = float(os.environ.get("DISPATCH_JOB_TIMEOUT", "3600"))
KEEP_WORKTREE = os.environ.get("DISPATCH_KEEP_WORKTREE", "1") == "1"
ONESHOT = os.environ.get("DISPATCH_ONESHOT", "0") == "1"


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
def _run(cmd, cwd=None, timeout=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def repo_root_for(cwd):
    cwd = cwd or DEFAULT_CWD
    res = _run(["git", "-C", cwd, "rev-parse", "--show-toplevel"])
    if res.returncode != 0:
        return None
    return res.stdout.strip()


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
            res = _run(cmd, cwd=cwd, timeout=JOB_TIMEOUT)
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
    repo = repo_root_for(job.get("cwd"))
    if not repo:
        update(job_id, "failed", exit_code=2,
               summary="no git repo at cwd=%s" % (job.get("cwd") or DEFAULT_CWD))
        log("FAILED %s: no repo" % job_id)
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
