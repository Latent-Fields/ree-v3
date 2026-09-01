"""ree_assembly_git_writer.py -- DB->git materializer for REE_assembly's
append-shaped PHASE-4 intake: igw_routine_log.md.

WHY A SEPARATE WRITER FROM task_claim_chip_git_writer.py
-----------------------------------------------------------
That module's dedicated clone is of `REE_Working` (the umbrella repo) --
WORKSPACE_STATE.md and RECOMMENDATION_LOG.jsonl both live there. IGW's
files live in `REE_assembly`, a different repo entirely, so per DP-6
(phase4_commit_intake_design.md section 9) they need their OWN dedicated
writable clone rather than piggybacking on that tick -- two timers pushing
the same repo would race each other, and "correctness is per-file CAS, not
per-repo locking" is the argument for git_intent.py's synchronous CAS
apply, not for sharing a materializer clone across repos.

SCOPE: append-shaped only
--------------------------
REE_assembly's OTHER PHASE-4-routed files (igw_routine_ledger.json,
igw_assignments.json, inter_governance_workset.v1.json/.md,
experiment_proposals.v1.json) are editorial/whole-file and go through
POST /intent/replace (git_intent.py), applied SYNCHRONOUSLY within the
request that submits them -- they have no DB spool and this tick does not
touch them. This tick covers exactly one file: igw_routine_log.md, the
append-shaped sibling of WORKSPACE_STATE.md/RECOMMENDATION_LOG.jsonl (a
one-line-per-tick heartbeat log, no section structure, no jsonl parsing --
the EOF-append trivial case, same as render_recommendation_log).

One tick per invocation (pace with a systemd timer, same convention as
task_claim_chip_git_writer.py). Each tick:
  1. INGEST -- none. There is no ingest for an append-only file (same
     argument as the WS/RECLOG slices: the file's git copy stays
     authoritative for everything already in it).
  2. RENDER -- render_igw_log() appends pending DB entries the file does
     not yet carry, byte-preserving everything already there. Same
     carried/awaiting/append trichotomy and same only-after-proof
     materialization contract as render_recommendation_log.
  3. COMPARE -- mode=check: report, write nothing (the deployment soak).
     mode=write: commit + push on a state change, retrying on a rejected
     push up to --max-push-retries times (ingesting the refreshed origin
     tip each retry, so the retry is lossless by construction).

Deploy shape: deploy/ree-assembly-git-writer.{service,timer}, Type=oneshot +
timer, a WRITABLE clone of REE_assembly DEDICATED to this writer (e.g.
/home/ree/REE_Working_assembly_writer -- nothing else may edit it, this
writer resets it hard to origin every write tick), env
ASSEMBLY_WRITER_MODE=check until the deployment soak (section 7) completes.

All printed text is ASCII-only.
"""

import argparse
import os
import subprocess
import sys

import db

DEFAULT_REPO_PATH = os.environ.get("ASSEMBLY_WRITER_REPO_PATH", "")
DEFAULT_BRANCH = os.environ.get("ASSEMBLY_WRITER_BRANCH", "master")
DEFAULT_MODE = os.environ.get("ASSEMBLY_WRITER_MODE", "check")
IGW_LOG_REL_PATH = "evidence/planning/igw_routine_log.md"
COMMIT_PREFIX = "phase4-igw-log:"


def _git(repo_path, *args, check=True, timeout=60):
    return subprocess.run(
        ["git", "-C", repo_path] + list(args),
        check=check, capture_output=True, timeout=timeout)


def _fetch_and_resolve(repo_path, branch):
    """fetch origin, resolve origin/<branch> to a sha. None on failure --
    callers must abort the tick rather than render against unknown origin
    state."""
    try:
        _git(repo_path, "fetch", "--quiet", "origin", branch)
        out = _git(repo_path, "rev-parse", "origin/%s" % (branch,), timeout=15)
        return out.stdout.decode("utf-8").strip()
    except Exception as exc:  # noqa: BLE001 -- degrade, never crash
        sys.stderr.write("[assembly-writer] WARN fetch/resolve failed for "
                         "%r origin/%s: %r\n" % (repo_path, branch, exc))
        return None


def _show_text(repo_path, rev, rel_path, warn=True):
    try:
        out = _git(repo_path, "show", "%s:%s" % (rev, rel_path), timeout=30)
        return out.stdout.decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        if warn:
            sys.stderr.write("[assembly-writer] WARN could not read %s at "
                             "%s: %r\n" % (rel_path, rev, exc))
        return None


def render_igw_log(conn, source_text):
    """Compute the append-only igw_routine_log.md render: append pending DB
    lines the file does not yet carry, byte-preserving everything already
    there. The plain-text sibling of task_claim_chip_git_writer.
    render_recommendation_log (same carried/awaiting/append trichotomy, same
    only-after-proof marking contract for the caller) -- no per-line JSON
    parsing, since this file is one free-text heartbeat line per tick, not
    jsonl.

    Returns (new_text, stats, carried_ids, appended_ids):
      * carried_ids -- pending lines already a substring of the file (this
        writer's own earlier push, or a dual-write client's own push,
        landed them). Mark materialized regardless of mode.
      * appended_ids -- lines this render appends (client_git_write=0 only).
        Mark materialized ONLY after the push carrying them succeeds.
      * client_git_write=1 lines not yet carried are counted as
        n_awaiting_client and never appended (appending would race the
        client's own push into a duplicate line).
    """
    pending = db.pending_igw_log_entries(conn)
    carried_ids = []
    append_rows = []
    awaiting = 0
    for row in pending:
        line = row["line"].strip()
        present = source_text is not None and any(
            ln.strip() == line for ln in source_text.splitlines())
        if present:
            carried_ids.append(row["entry_id"])
        elif row["client_git_write"]:
            awaiting += 1
        else:
            append_rows.append(row)
    stats = {
        "n_pending": len(pending),
        "n_carried": len(carried_ids),
        "n_awaiting_client": awaiting,
        "n_appended": len(append_rows),
        "guard_error": None,
    }
    if source_text is None or not append_rows:
        stats["n_appended"] = 0
        if source_text is None and append_rows:
            sys.stderr.write("[assembly-writer] WARN %s unreadable at "
                             "origin with %d line(s) pending append -- "
                             "holding them\n"
                             % (IGW_LOG_REL_PATH, len(append_rows)))
        return source_text, stats, carried_ids, []
    base = source_text
    if base and not base.endswith("\n"):
        base = base + "\n"
    block = "".join(r["line"].strip() + "\n" for r in append_rows)
    new_text = base + block
    # Append-only guard, same shape as render_recommendation_log's: the
    # render must START with the (newline-normalized) source byte-for-byte,
    # and must add exactly the appended block.
    if not new_text.startswith(base) or new_text[len(base):] != block:
        stats["guard_error"] = "append_only"
        stats["n_appended"] = 0
        sys.stderr.write("[assembly-writer] ERROR %s render failed the "
                         "append_only guard -- refusing to write it this "
                         "tick\n" % (IGW_LOG_REL_PATH,))
        return source_text, stats, carried_ids, []
    return new_text, stats, carried_ids, [r["entry_id"] for r in append_rows]


def materialize_once(conn, repo_path, branch=None, mode=None,
                     max_push_retries=3):
    """One materializer tick. Returns a result dict (ASCII-safe values), or
    None when origin state could not be established."""
    branch = branch or DEFAULT_BRANCH
    mode = mode or DEFAULT_MODE
    for attempt in range(max_push_retries + 1):
        sha = _fetch_and_resolve(repo_path, branch)
        if sha is None:
            return None
        log_text = _show_text(repo_path, sha, IGW_LOG_REL_PATH, warn=False)
        log_render, log_stats, log_carried, log_append_ids = render_igw_log(
            conn, log_text)
        if log_carried:
            db.mark_igw_log_entries_materialized(conn, log_carried, sha)

        result = {
            "source_ref": sha,
            "mode": mode,
            "igw_log": log_stats,
            "committed": False,
        }
        needs_write = bool(log_append_ids)
        if mode != "write" or not needs_write:
            return result

        try:
            _git(repo_path, "checkout", "--quiet", branch)
            _git(repo_path, "reset", "--hard", "--quiet", sha)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write("[assembly-writer] ERROR could not reset "
                             "writer checkout: %r\n" % (exc,))
            return result
        with open(os.path.join(repo_path, IGW_LOG_REL_PATH), "w",
                  encoding="utf-8") as fh:
            fh.write(log_render)
        msg = "%s append %s (+%d line%s)" % (
            COMMIT_PREFIX, IGW_LOG_REL_PATH, len(log_append_ids),
            "" if len(log_append_ids) == 1 else "s")
        try:
            _git(repo_path, "add", "--", IGW_LOG_REL_PATH)
            _git(repo_path, "commit", "--quiet", "-m", msg)
            push = _git(repo_path, "push", "--quiet", "origin",
                       "HEAD:%s" % (branch,), check=False, timeout=120)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write("[assembly-writer] ERROR commit failed: %r\n"
                             % (exc,))
            return result
        if push.returncode == 0:
            result["committed"] = True
            try:
                head = _git(repo_path, "rev-parse",
                           "HEAD").stdout.decode("utf-8").strip()
            except Exception:  # noqa: BLE001
                head = sha
            db.mark_igw_log_entries_materialized(conn, log_append_ids, head)
            return result
        sys.stderr.write("[assembly-writer] push rejected (attempt %d/%d); "
                         "re-running tick\n"
                         % (attempt + 1, max_push_retries + 1))
    sys.stderr.write("[assembly-writer] ERROR push retries exhausted\n")
    return result


def main():
    ap = argparse.ArgumentParser(
        description="DB->git materializer for REE_assembly's append-shaped "
                    "PHASE-4 intake (igw_routine_log.md). One tick per "
                    "invocation; pace with the paired systemd timer.")
    ap.add_argument("--repo-path", default=DEFAULT_REPO_PATH,
                    help="WRITABLE clone of REE_assembly dedicated to this "
                         "writer.")
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--mode", choices=("check", "write"), default=DEFAULT_MODE,
                    help="check: render + compare + report, never write "
                         "(the deployment soak). write: commit + push on a "
                         "state change.")
    ap.add_argument("--db", default=os.environ.get(
        "COORDINATOR_DB", os.path.join(
            os.path.dirname(__file__), "coordinator.db")))
    ap.add_argument("--max-push-retries", type=int, default=3)
    args = ap.parse_args()

    if not args.repo_path:
        sys.stderr.write("[assembly-writer] ERROR --repo-path (or "
                         "ASSEMBLY_WRITER_REPO_PATH) is required\n")
        return 2
    db.init_db(args.db)
    conn = db.connect(args.db)
    try:
        result = materialize_once(conn, args.repo_path, branch=args.branch,
                                  mode=args.mode,
                                  max_push_retries=args.max_push_retries)
    finally:
        conn.close()
    if result is None:
        sys.stderr.write("[assembly-writer] nothing materialized this tick "
                         "(origin unreadable)\n")
        return 1
    log = result.get("igw_log") or {}
    sys.stdout.write(
        "[assembly-writer] ref=%s mode=%s igw_log: pending=%d carried=%d "
        "appended=%d awaiting_client=%d%s | committed=%s\n" % (
            result["source_ref"][:12], result["mode"],
            log.get("n_pending", 0), log.get("n_carried", 0),
            log.get("n_appended", 0), log.get("n_awaiting_client", 0),
            (" GUARD=%s" % log["guard_error"]) if log.get("guard_error") else "",
            result["committed"]))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
