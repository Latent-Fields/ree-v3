"""Sync daemon.

PHASE 1 (shadow) behaviour -- the default behaviour:
  * Periodically read experiment_queue.json (git is authoritative in
    shadow) and reconcile the coordinator DB mirror to match it, so the
    coordinator's claim logic always evaluates against fresh state.
  * Claim-level shadow checks happen via runner POST /claim (git_verdict
    vs coordinator evaluate_claim). State-level pre-upsert reconcile was
    removed 2026-05-20 (false positives; see SOAK_LOG.md E1/E2).
  * It does NOT write git. Read-only on the queue file. No autostash, no
    rebase -- this daemon is structurally incapable of the failure class
    the whole project exists to remove.

PHASE 3 (authoritative) behaviour is present but guarded OFF: becoming the
sole git writer (commit result manifests, push, snapshot queue) only
activates when SYNC_MODE=authoritative AND --i-understand-phase3 is passed.
Stubbed deliberately; do not enable until Phase 1 has proven out.

PHASE 2 (claim cutover) behaviour is selected by SYNC_MODE=coordinator:
git remains the queue worklist and result/status transport, but the DB is
the claim authority. Reconciliation refreshes metadata and removals from
git without overwriting coordinator claim state.

All printed text is ASCII-only.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import db
import manifest_spool

DEFAULT_QUEUE = os.path.join(
    os.path.dirname(__file__), "..", "experiment_queue.json")


def _load_queue_json(queue_path):
    """Return the parsed queue dict from the AUTHORITATIVE git ref
    (SYNC_QUEUE_REF, default origin/main), fetched read-only.

    Why not just read queue_path: the local working-tree copy is only as
    fresh as this box's last `git pull`. When this box's runner is drained
    nothing pulls, so the file goes stale and every other machine's
    git-claim looks like a state-divergence (mirror=claimed vs stale
    file=pending) -- a false positive, not a coordinator-logic fault.
    `git fetch` + `git show <ref>:file` never touches the working tree
    (no autostash risk, consistent with sync_daemon being git-read-only).
    Degrades to the local file if git is unavailable, logging that it is
    running on a possibly-stale source."""
    repo = os.path.dirname(os.path.abspath(queue_path))
    rel = os.path.basename(queue_path)
    ref = os.environ.get("SYNC_QUEUE_REF", "origin/main")
    try:
        subprocess.run(["git", "-C", repo, "fetch", "--quiet", "origin"],
                        check=True, capture_output=True, timeout=30)
        out = subprocess.run(
            ["git", "-C", repo, "show", "%s:%s" % (ref, rel)],
            check=True, capture_output=True, timeout=15)
        return json.loads(out.stdout.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 -- degrade, never crash
        if not os.path.exists(queue_path):
            sys.stderr.write(
                "[sync] no git queue and no local file: %r\n" % exc)
            return None
        sys.stderr.write(
            "[sync] WARN authoritative git queue unavailable (%r); "
            "falling back to STALE local file\n" % exc)
        with open(queue_path, "r", encoding="utf-8") as fh:
            return json.load(fh)


def reconcile_once(conn, queue_path, *, claim_authority="git"):
    """Make the mirror match the AUTHORITATIVE git queue (not this box's
    possibly-stale local file). Returns (n_items, n_state_divergences).

    claim_authority='git' is Phase 1 shadow: git claims are truth and
    state-level mismatches are logged. claim_authority='coordinator' is
    Phase 2: git is only the worklist, so existing DB claim state is
    preserved and git-vs-DB claim mismatches are not divergence rows.
    """
    qdata = _load_queue_json(queue_path)
    if qdata is None:
        return (0, 0)
    items = {it["queue_id"]: it for it in qdata.get("items", [])
             if it.get("queue_id")}

    divergences = 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        mirror = {r["queue_id"]: r for r in conn.execute(
            "SELECT queue_id, status, claimed_by_machine FROM experiments"
        ).fetchall()}

        for qid, item in items.items():
            # Phase 1 (git authority): upsert mirror from authoritative git
            # queue each tick. Pre-upsert state-reconcile logged false
            # divergences when the mirror was briefly ahead of origin/main
            # (harness E1 in SOAK_LOG.md) -- claim-level shadow /claim
            # compares git_verdict vs evaluate_claim instead.
            db.upsert_experiment(
                conn, item, preserve_claim=(claim_authority == "coordinator"))

        # Items no longer in the queue file have been completed/removed by
        # the authoritative path; drop them from the mirror so the
        # coordinator does not hand them out.
        stale = set(mirror) - set(items)
        for qid in stale:
            conn.execute("DELETE FROM experiments WHERE queue_id=?", (qid,))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return (len(items), divergences)


# Set True only after phase3_git_writer steps 1-6 are implemented and
# phase3_preflight.py + phase3_verify.py pass on the live fleet.
PHASE3_GIT_WRITER_READY = False

# Hub paths (override via env when deploying).
PHASE3_REE_ASSEMBLY = os.environ.get(
    "PHASE3_REE_ASSEMBLY",
    "/home/ree/REE_Working/REE_assembly")
PHASE3_QUEUE_FILE = os.environ.get(
    "COORDINATOR_QUEUE_FILE", DEFAULT_QUEUE)


# Max manifests to write+commit per writer tick. Bounds tick latency and
# limits worst-case rollback if a push fails (we ROLLBACK committed_at for
# the unpushed batch, then retry).
PHASE3_BATCH_SIZE = int(os.environ.get("PHASE3_BATCH_SIZE", "32"))

# Default branch on the hub's REE_assembly checkout. Override via env if the
# hub is ever moved to a non-master deploy layout.
PHASE3_ASSEMBLY_BRANCH = os.environ.get("PHASE3_ASSEMBLY_BRANCH", "master")


def _git(repo, *args, timeout=30, check=True):
    """Run git in repo. Returns CompletedProcess. capture_output=True so
    nothing leaks to stdout/stderr unless we choose to log it."""
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, timeout=timeout, check=check,
    )


def _hub_working_tree_clean(repo):
    """Phase 3 explicitly retires autostash, so the writer refuses to
    operate on a dirty tree -- any uncommitted edit on the hub checkout
    must be resolved by a human, not silently stashed. Returns (clean,
    reason). reason is a one-line string when clean=False."""
    try:
        out = _git(repo, "status", "--porcelain", check=True).stdout
    except subprocess.CalledProcessError as exc:
        return (False, "git status failed: %r" % exc)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (False, "git status error: %r" % exc)
    if out.strip():
        first = out.strip().splitlines()[0]
        return (False, "dirty working tree: %s" % first[:120])
    return (True, "")


def phase3_git_writer(
    conn,
    queue_path,
    *,
    ree_assembly_path=None,
    dry_run=False,
):
    """Sole git writer tick (Phase 3).

    Reads pending manifests from the filesystem spool, writes them under
    REE_assembly/evidence/experiments/, commits, and pushes. Marks
    `results.committed_at` only after a successful push so a crash midway
    leaves the manifest available for the next tick (idempotent retry).

    SAFETY:
      - PHASE3_GIT_WRITER_READY is checked at every entry. While False the
        writer logs intent and returns False; main()'s authoritative-mode
        loop then refuses to advance, so no git writes can happen even if
        the operator flips SYNC_MODE prematurely.
      - Refuses to operate on a dirty REE_assembly working tree (the whole
        point of Phase 3 is to retire the autostash war; a human must
        clean up unexpected dirt).
      - Never calls `git pull --rebase --autostash`. A non-fast-forward
        push fails the tick loudly and leaves the spool intact for retry.
      - Before marking results.committed_at on a "no new diff" tick, checks
        `git rev-list --count origin/<branch>..HEAD`. The diff-cached
        short-circuit only fires when ahead==0 (bytes truly on origin);
        ahead>0 forces a push of the unpushed local commit first, or
        refuses the tick if that push is still rejected. Without this
        guard a rejected-push tick followed by a no-operator-action tick
        would silently drain the spool without origin ever receiving the
        bytes.
      - Batched to PHASE3_BATCH_SIZE manifests per tick; the rest land in
        subsequent ticks.

    Returns True only when a full tick completed (or dry_run simulated).
    Returns False when the writer stub guard is active, the tree is dirty,
    or the spool is empty (nothing to do -- the daemon is idle).

    Out-of-scope for this sketch (deferred TODO):
      - Step 5: snapshot completed queue items from `experiments` table
        into the ree-v3 checkout's experiment_queue.json and push.
      - Step 6: write derived runner_heartbeats/*.json + runner_status/
        from the heartbeats table (replaces the per-runner git heartbeat
        push that runner_remote_control.push_heartbeat does today).
      Both extensions live in this same function once the results path is
      validated under a test fleet.
    """
    asm = ree_assembly_path or PHASE3_REE_ASSEMBLY
    if not PHASE3_GIT_WRITER_READY:
        sys.stderr.write(
            "[phase3] git writer stub (PHASE3_GIT_WRITER_READY=False); "
            "no git writes performed\n")
        return False

    # Spool is the prerequisite. Without it /result has no bytes to
    # commit; refusing is louder than producing empty ticks forever.
    if manifest_spool.spool_root() is None:
        sys.stderr.write(
            "[phase3] COORDINATOR_SPOOL_DIR unset; refusing -- /result "
            "is not persisting manifest bytes, so nothing to commit\n")
        return False

    pending_ids = list(manifest_spool.list_pending_run_ids())
    if not pending_ids:
        return True  # idle tick is a successful no-op

    batch = pending_ids[:PHASE3_BATCH_SIZE]

    if dry_run:
        sys.stdout.write(
            "[phase3] dry_run tick: %d pending, would commit %d\n" % (
                len(pending_ids), len(batch)))
        return True

    clean, reason = _hub_working_tree_clean(asm)
    if not clean:
        sys.stderr.write(
            "[phase3] refusing tick: REE_assembly at %s is %s. Phase 3 "
            "does NOT autostash -- resolve the dirt by hand, then the "
            "next tick will retry.\n" % (asm, reason))
        return False

    # Stage 1: write manifests onto the working tree and stage them.
    staged = []  # list of (run_id, relpath) successfully written
    for run_id in batch:
        raw = manifest_spool.read_manifest(run_id)
        meta = manifest_spool.read_meta(run_id) or {}
        if raw is None:
            sys.stderr.write(
                "[phase3] WARN missing manifest bytes for %s; skipping\n"
                % run_id)
            continue
        try:
            manifest_doc = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            manifest_doc = {}
        # Prefer the meta sidecar's hint (runner-supplied at /result time);
        # fall back to the manifest body; finally to the run_id default.
        hint = meta.get("manifest_relpath") or manifest_doc.get(
            "manifest_relpath")
        try:
            relpath = manifest_spool.derive_evidence_relpath(
                run_id, {"manifest_relpath": hint} if hint else manifest_doc)
        except ValueError as exc:
            sys.stderr.write(
                "[phase3] WARN derive_evidence_relpath rejected %s: %s\n"
                % (run_id, exc))
            continue
        target = os.path.join(asm, relpath)
        target_dir = os.path.dirname(target)
        try:
            os.makedirs(target_dir, exist_ok=True)
            with open(target, "wb") as fh:
                fh.write(raw)
            _git(asm, "add", relpath, timeout=15, check=True)
            staged.append((run_id, relpath))
        except (OSError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            sys.stderr.write(
                "[phase3] WARN stage failed for %s -> %s: %r\n" % (
                    run_id, relpath, exc))

    if not staged:
        sys.stderr.write(
            "[phase3] no manifests staged this tick; nothing to commit\n")
        return False

    # Stage 2: single commit + single push for the whole batch.
    today = db.utcnow()[:10]
    commit_msg = "phase3: %d v3 result manifest(s) %s" % (len(staged), today)
    try:
        diff = _git(asm, "diff", "--cached", "--quiet", check=False,
                    timeout=10)
        if diff.returncode == 0:
            # `git add` produced no diff. Two cases are indistinguishable
            # from `git diff --cached` alone:
            #   (a) bytes already on origin (true idempotent re-spool of
            #       a previously-committed-and-pushed run), OR
            #   (b) bytes live in an UNPUSHED local commit -- the tick
            #       after a rejected push (Phase 3 explicitly retires
            #       autostash, so a rejected push leaves the local
            #       commit in HEAD with no operator intervention).
            # Marking committed_at in case (b) without a push is unsafe:
            # the DB says "done" but origin never received the bytes.
            # `git rev-list --count origin/<branch>..HEAD` distinguishes:
            # 0 -> case (a); >0 -> case (b), must push before marking.
            ahead = _git(
                asm, "rev-list", "--count",
                "origin/" + PHASE3_ASSEMBLY_BRANCH + "..HEAD",
                check=False, timeout=10)
            if ahead.returncode != 0:
                sys.stderr.write(
                    "[phase3] refusing to mark committed: rev-list "
                    "ahead-count failed (%s). Spool retained.\n" % (
                        ahead.stderr.strip()[:240]))
                return False
            ahead_count = ahead.stdout.strip()
            if ahead_count and ahead_count != "0":
                # Case (b): push the existing unpushed commit. If push
                # fails (still non-FF), refuse to mark -- the spool
                # entry survives for the next tick.
                push = _git(
                    asm, "push", "origin",
                    "HEAD:" + PHASE3_ASSEMBLY_BRANCH,
                    timeout=60, check=False)
                if push.returncode != 0:
                    sys.stderr.write(
                        "[phase3] push REJECTED for unpushed local "
                        "commit: %s. NOT marking committed_at; spool "
                        "retained. Operator must investigate (non-"
                        "fast-forward = hub is behind origin; resolve "
                        "by hand).\n" % (push.stderr.strip()[:240]))
                    return False
                sys.stdout.write(
                    "[phase3] pushed unpushed local commit (HEAD was %s "
                    "ahead of origin/%s); %d row(s) committed\n" % (
                        ahead_count, PHASE3_ASSEMBLY_BRANCH, len(staged)))
            else:
                # Case (a): true idempotent re-spool. ahead == 0 means
                # the bytes really are on origin already.
                sys.stdout.write(
                    "[phase3] batch already on tree and on origin "
                    "(ahead==0); marking %d row(s) committed without "
                    "a push\n" % len(staged))
        else:
            _git(asm, "commit", "-m", commit_msg, timeout=20, check=True)
            push = _git(
                asm, "push", "origin", "HEAD:" + PHASE3_ASSEMBLY_BRANCH,
                timeout=60, check=False)
            if push.returncode != 0:
                sys.stderr.write(
                    "[phase3] push REJECTED: %s. NOT marking committed_at; "
                    "spool retained for retry on the next tick. Operator "
                    "must investigate (non-fast-forward = hub is behind "
                    "origin; resolve by hand).\n" % (push.stderr.strip()[:240]))
                return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write("[phase3] git commit/push error: %r\n" % exc)
        return False

    # Stage 3: mark DB committed + delete spool entries. Order matters:
    # update DB first (cheap, atomic) before deleting bytes from disk so a
    # crash between the two leaves the spool entries that the next tick
    # will re-process as if they were uncommitted, which `git add` will
    # detect as no-diff and short-circuit (the idempotent-already-on-tree
    # branch above).
    now = db.utcnow()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "UPDATE results SET committed_at=? WHERE run_id=? "
            "AND committed_at IS NULL",
            [(now, run_id) for run_id, _ in staged],
        )
        conn.execute("COMMIT")
    except Exception as exc:  # noqa: BLE001 -- daemon must not die
        sys.stderr.write(
            "[phase3] WARN committed_at update failed: %r. Spool retained; "
            "next tick will replay idempotently.\n" % exc)
        try:
            conn.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            pass
        return False

    for run_id, _ in staged:
        manifest_spool.delete_manifest(run_id)

    sys.stdout.write(
        "[phase3] committed %d manifest(s) (%d remaining in spool)\n" % (
            len(staged), max(0, len(pending_ids) - len(staged))))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=os.environ.get(
        "COORDINATOR_QUEUE_FILE", DEFAULT_QUEUE))
    ap.add_argument("--db", default=os.environ.get(
        "COORDINATOR_DB", os.path.join(
            os.path.dirname(__file__), "coordinator.db")))
    ap.add_argument("--interval", type=float,
                    default=float(os.environ.get(
                        "SYNC_INTERVAL", "60")))
    ap.add_argument("--once", action="store_true",
                    help="reconcile once and exit (used by tests)")
    ap.add_argument("--i-understand-phase3", action="store_true")
    args = ap.parse_args()

    sync_mode = os.environ.get("SYNC_MODE", "shadow")
    if sync_mode == "authoritative":
        if not args.i_understand_phase3:
            sys.stderr.write(
                "refusing: SYNC_MODE=authoritative needs "
                "--i-understand-phase3 (Phase 3 not built)\n")
            return 2
        db.init_db(args.db)
        while True:
            conn = db.connect(args.db)
            try:
                ok = phase3_git_writer(conn, args.queue)
            finally:
                conn.close()
            if not ok:
                sys.stderr.write(
                    "refusing: phase3 git writer not ready (see "
                    "PHASE3_GIT_WRITER_READY and phase3_preflight.py)\n")
                return 2
            if args.once:
                return 0
            time.sleep(args.interval)
    if sync_mode not in ("shadow", "coordinator"):
        sys.stderr.write(
            "refusing: SYNC_MODE must be shadow, coordinator, or "
            "authoritative (got %r)\n" % sync_mode)
        return 2
    claim_authority = "coordinator" if sync_mode == "coordinator" else "git"

    db.init_db(args.db)
    while True:
        conn = db.connect(args.db)
        try:
            n, div = reconcile_once(conn, args.queue,
                                    claim_authority=claim_authority)
            sys.stdout.write(
                "[sync] reconciled %d items, %d state-divergence(s)\n" % (
                    n, div))
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001 -- daemon must not die
            sys.stderr.write("[sync] reconcile error: %r\n" % exc)
        finally:
            conn.close()
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
