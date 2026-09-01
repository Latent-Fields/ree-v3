"""git_intent.py -- POST /intent/replace, the generic whole-file CAS verb.

Design: REE_assembly/evidence/planning/phase4_commit_intake_design.md
section 3.2 (wire contract), section 5 (routing table), section 9 (DP-1..
DP-11). This module is the "hub applies intents ONE AT A TIME per repo"
half of that design; app.py's `_intent_replace` handler is a thin wrapper
around `apply_intent()` below.

WHY SYNCHRONOUS, UNLIKE THE APPEND-SHAPED INTAKES
--------------------------------------------------
`/workspace_state/append` and `/recommendation_log/append` (task_claim_chip_
git_writer.py) SPOOL to a DB table and wait for a periodic tick to splice
them into the file -- correct for an append, where there is nothing to
validate against a prior state. A CAS replacement is different: its whole
point is a same-request verdict (`applied` / `base_moved`, section 3.2's
pseudocode), so the caller can immediately rebase-and-retry on a conflict.
That means the fetch/compare/write/validate/commit/push cycle happens
in-process, inside the HTTP request handler's thread -- there is no writer
tick for this verb.

WHY A SEPARATE DEDICATED CLONE PER REPO (DP-6)
------------------------------------------------
Every repo routed here gets its OWN writable clone, configured via
`COORDINATOR_INTENT_REPO_<REPO>` (e.g. COORDINATOR_INTENT_REPO_REE_ASSEMBLY).
Never the phase3 sync_daemon writers' clone (they run on their own tick and
would race a `git reset --hard` against this module's), and never the
umbrella registry writer's clone (task_claim_chip_git_writer.py operates on
a different repo entirely -- REE_Working, not REE_assembly). Per DP-6:
"correctness is per-file CAS, not per-repo locking" -- the base_sha compare
plus git's own non-fast-forward push rejection is what makes two concurrent
appliers safe, even across processes; the in-process `threading.Lock` below
exists only to stop two THREADS in this one process stepping on the same
local working directory, not as a cross-process correctness mechanism.

WHAT APPLY_INTENT DOES, ONE ATTEMPT
------------------------------------
1. Look up `path` in ROUTING. Not present, or present under a different
   repo -- 'not_routed', always (section 3.2: never a generic byte-intake;
   an unrouted path is refused unconditionally, not merely unvalidated).
2. Run the path's validator over `content`. Failure -- 'validation_failed'.
3. Resolve the dedicated clone for `repo`. Unconfigured (DP-11: new code,
   pre-activation) -- 'repo_not_configured'.
4. Fetch origin, resolve `origin/<branch>` to a sha. Unreachable --
   'origin_unreadable' (the caller degrades to git, same as any other
   coordinator-transport failure).
5. Compare the file's content AT `base_sha` against its content AT the
   resolved origin tip -- file-content compare, not commit-equality (an
   unrelated commit to another file must not bounce the intent). A mismatch
   -- 'base_moved', with the current sha/content for the caller to rebase
   from (DP-1: rebase from ORIGIN's content, never a working-tree read).
6. Identical content already at origin tip -- 'applied' as a no-op (no
   commit; commit-on-state-change only, matching every writer tick in this
   codebase -- an empty `git commit` is also simply an error).
7. Size guard: `suspicious_shrink` unless `allow_shrink` is set (mirrors the
   append-intake writers' truncation guards, scaled to an editorial
   replacement rather than a splice).
8. `shadow=True` -- checked and logged, never written (section 7's soak
   mode). Returns 'applied' with `shadow: True`.
9. Otherwise: reset the dedicated clone hard to origin tip, write `content`,
   commit (message + DP-9 session_id/machine trailer), push. A rejected
   push re-runs the WHOLE cycle from step 4 (up to `max_push_retries`) --
   ingesting the new origin tip makes the retry lossless by construction,
   the same argument task_claim_chip_git_writer.py's push-retry makes.

Every outcome is logged to `git_intent_log` (db.record_git_intent) for audit
and section-7 soak evidence. Logging failures never propagate -- an audit
row is best-effort, never the primary effect.

All printed text is ASCII-only (Windows cp1252 safety, matching the rest of
this package).
"""

import json
import os
import subprocess
import sys
import threading

import db

# Refuse a replacement that drops below this fraction of the current file's
# size unless the caller explicitly set allow_shrink -- same spirit as the
# WS/RECLOG materializer's grow-only/append-only guards, scaled to an
# editorial whole-file replacement (which legitimately shrinks sometimes,
# hence a ratio guard rather than "grow-only").
SHRINK_GUARD_RATIO = 0.5

DEFAULT_BRANCH = {
    "REE_assembly": "master",
}


def _validate_json(content):
    try:
        json.loads(content)
    except ValueError as exc:
        return False, "invalid JSON: %s" % (exc,)
    return True, None


def _validate_nonempty_text(content):
    if not isinstance(content, str) or not content.strip():
        return False, "content is empty"
    return True, None


# Server-side routing allowlist (phase4_commit_intake_design.md section 5).
# Deliberately narrow: only the paths THIS slice (the IGW tick intake
# client) needs are routed here. claims.yaml, hypothesis_space_registry.v1
# .json, substrate_queue.json, review_tracker.json are all listed in the
# design doc's full table but are NOT added -- each needs its own soak plan
# and go-live decision, and this slice is scoped to the igw tick's files.
ROUTING = {
    "evidence/planning/igw_routine_ledger.json": {
        "repo": "REE_assembly", "validate": _validate_json,
    },
    "evidence/planning/igw_assignments.json": {
        "repo": "REE_assembly", "validate": _validate_json,
    },
    "evidence/planning/inter_governance_workset.v1.json": {
        "repo": "REE_assembly", "validate": _validate_json,
    },
    "evidence/planning/inter_governance_workset.md": {
        "repo": "REE_assembly", "validate": _validate_nonempty_text,
    },
    "evidence/planning/experiment_proposals.v1.json": {
        "repo": "REE_assembly", "validate": _validate_json,
    },
}


def _repo_env_var(repo):
    return "COORDINATOR_INTENT_REPO_" + "".join(
        c if (c.isalnum() and c.isascii()) else "_" for c in repo.upper())


def _clone_path(repo):
    return os.environ.get(_repo_env_var(repo), "")


_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def _repo_lock(repo):
    with _LOCKS_GUARD:
        lock = _LOCKS.get(repo)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[repo] = lock
        return lock


def _git(repo_path, *args, check=True, timeout=60):
    return subprocess.run(
        ["git", "-C", repo_path] + list(args),
        check=check, capture_output=True, timeout=timeout)


def _fetch_and_resolve(repo_path, branch):
    """fetch origin, resolve origin/<branch> to a sha. None on failure --
    callers must treat that as 'cannot establish origin state', never as
    'origin is empty'."""
    try:
        _git(repo_path, "fetch", "--quiet", "origin", branch)
        out = _git(repo_path, "rev-parse", "origin/%s" % (branch,), timeout=15)
        return out.stdout.decode("utf-8").strip()
    except Exception as exc:  # noqa: BLE001 -- degrade, never crash
        sys.stderr.write("[git-intent] WARN fetch/resolve failed for %r "
                         "origin/%s: %r\n" % (repo_path, branch, exc))
        return None


def _show_text(repo_path, rev, rel_path):
    if not rev:
        return None
    try:
        out = _git(repo_path, "show", "%s:%s" % (rev, rel_path), timeout=30)
        return out.stdout.decode("utf-8")
    except Exception:  # noqa: BLE001 -- missing path/rev is a normal case
        return None


def _format_commit_message(message, session_id, machine):
    """DP-9: intents record session_id (body) AND the token's machine,
    exactly the D6 attribution split every other intake verb uses."""
    msg = (message or "").strip() or "intent/replace"
    trailer = []
    if session_id:
        trailer.append("session_id: %s" % (session_id,))
    if machine:
        trailer.append("machine: %s" % (machine,))
    if trailer:
        return msg + "\n\n" + "\n".join(trailer)
    return msg


def _log(conn, *, repo, path, base_sha, verdict, session_id="", machine=None,
         message=None, shadow=False, applied_sha=None, size_before=None,
         size_after=None, now=None):
    if conn is None:
        return
    try:
        db.record_git_intent(
            conn, repo=repo, path=path, base_sha=base_sha, verdict=verdict,
            applied_sha=applied_sha, session_id=session_id, machine=machine,
            message=message, shadow=shadow, size_before=size_before,
            size_after=size_after, now=now)
    except Exception:  # noqa: BLE001 -- audit logging must never break the intent
        pass


def apply_intent(conn, *, repo, path, base_sha, content, message,
                 session_id="", machine=None, shadow=False,
                 allow_shrink=False, branch=None, max_push_retries=3,
                 now=None):
    """Apply one whole-file CAS intent. Never raises -- every outcome is a
    typed (verdict, payload) pair; see the module docstring for the full
    verdict list and what each one means for the caller."""
    session_id = session_id or ""
    route = ROUTING.get(path)
    if route is None or route.get("repo") != repo:
        _log(conn, repo=repo, path=path, base_sha=base_sha,
             verdict="not_routed", session_id=session_id, machine=machine,
             message=message, shadow=shadow, now=now)
        return "not_routed", {"path": path, "repo": repo}

    if not isinstance(content, str):
        _log(conn, repo=repo, path=path, base_sha=base_sha,
             verdict="validation_failed", session_id=session_id,
             machine=machine, message=message, shadow=shadow, now=now)
        return "validation_failed", {"reason": "content must be a string"}
    ok, reason = route["validate"](content)
    if not ok:
        _log(conn, repo=repo, path=path, base_sha=base_sha,
             verdict="validation_failed", session_id=session_id,
             machine=machine, message=message, shadow=shadow, now=now)
        return "validation_failed", {"reason": reason}

    clone_path = _clone_path(repo)
    if not clone_path:
        _log(conn, repo=repo, path=path, base_sha=base_sha,
             verdict="repo_not_configured", session_id=session_id,
             machine=machine, message=message, shadow=shadow, now=now)
        return "repo_not_configured", {"repo": repo}

    branch = branch or DEFAULT_BRANCH.get(repo, "master")
    lock = _repo_lock(repo)
    with lock:
        last_result = ("error", {"reason": "no attempt made"})
        for attempt in range(max_push_retries + 1):
            sha = _fetch_and_resolve(clone_path, branch)
            if sha is None:
                result = ("origin_unreadable", {"repo": repo})
                _log(conn, repo=repo, path=path, base_sha=base_sha,
                     verdict=result[0], session_id=session_id,
                     machine=machine, message=message, shadow=shadow,
                     now=now)
                return result

            current_content = _show_text(clone_path, sha, path)
            base_content = (_show_text(clone_path, base_sha, path)
                            if base_sha else None)
            if base_content != current_content:
                result = ("base_moved",
                          {"current_sha": sha,
                           "current_content": current_content})
                _log(conn, repo=repo, path=path, base_sha=base_sha,
                     verdict=result[0], session_id=session_id,
                     machine=machine, message=message, shadow=shadow,
                     now=now)
                return result

            if content == current_content:
                # No-op: the desired end state already holds at origin tip.
                # Commit-on-state-change only -- an empty commit is refused
                # by git anyway, and this codebase never reintroduces one.
                result = ("applied", {"commit": sha, "noop": True})
                _log(conn, repo=repo, path=path, base_sha=base_sha,
                     verdict=result[0], applied_sha=sha,
                     session_id=session_id, machine=machine, message=message,
                     shadow=shadow, now=now)
                return result

            size_before = len(current_content) if current_content else 0
            size_after = len(content)
            if (not allow_shrink and size_before > 0
                    and size_after < size_before * SHRINK_GUARD_RATIO):
                result = ("suspicious_shrink",
                          {"size_before": size_before, "size_after": size_after})
                _log(conn, repo=repo, path=path, base_sha=base_sha,
                     verdict=result[0], session_id=session_id,
                     machine=machine, message=message, shadow=shadow,
                     size_before=size_before, size_after=size_after, now=now)
                return result

            if shadow:
                result = ("applied", {"shadow": True, "checked_against": sha})
                _log(conn, repo=repo, path=path, base_sha=base_sha,
                     verdict=result[0], session_id=session_id,
                     machine=machine, message=message, shadow=True,
                     size_before=size_before, size_after=size_after, now=now)
                return result

            try:
                _git(clone_path, "checkout", "--quiet", branch)
                _git(clone_path, "reset", "--hard", "--quiet", sha)
                full_path = os.path.join(clone_path, path)
                parent = os.path.dirname(full_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as fh:
                    fh.write(content)
                _git(clone_path, "add", "--", path)
                commit_msg = _format_commit_message(message, session_id,
                                                     machine)
                _git(clone_path, "commit", "--quiet", "-m", commit_msg)
                push = _git(clone_path, "push", "--quiet", "origin",
                           "HEAD:%s" % (branch,), check=False, timeout=120)
            except Exception as exc:  # noqa: BLE001
                result = ("error", {"reason": repr(exc)})
                _log(conn, repo=repo, path=path, base_sha=base_sha,
                     verdict=result[0], session_id=session_id,
                     machine=machine, message=message, shadow=shadow,
                     now=now)
                return result

            if push.returncode == 0:
                try:
                    head = _git(clone_path, "rev-parse",
                                "HEAD").stdout.decode("utf-8").strip()
                except Exception:  # noqa: BLE001
                    head = None
                result = ("applied", {"commit": head})
                _log(conn, repo=repo, path=path, base_sha=base_sha,
                     verdict=result[0], applied_sha=head,
                     session_id=session_id, machine=machine, message=message,
                     shadow=shadow, size_before=size_before,
                     size_after=size_after, now=now)
                return result

            sys.stderr.write(
                "[git-intent] push rejected for %s (attempt %d/%d); "
                "re-verifying base against refreshed origin\n"
                % (path, attempt + 1, max_push_retries + 1))
            last_result = ("push_failed", {"repo": repo, "path": path})

        _log(conn, repo=repo, path=path, base_sha=base_sha,
             verdict=last_result[0], session_id=session_id, machine=machine,
             message=message, shadow=shadow, now=now)
        return last_result
