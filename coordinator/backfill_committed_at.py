"""One-shot backfill for coordinator.db results.committed_at NULL rows.

Background (2026-05-29):
    8 of 11 rows in the coordinator's results table had committed_at IS NULL
    even though 7 of the 8 corresponding manifest files were already on
    REE_assembly origin/master. The runs predated the Phase 3 writer's
    committed_at instrumentation -- they committed via the legacy runner-side
    push path, the manifest reached origin, but the writer never marked the
    DB row "committed" because it never wrote those manifests itself.

What this script does:
    For each row in `results` where committed_at IS NULL:
      1. Look for a matching manifest in REE_assembly/evidence/experiments/
         (and one fallback subdir _partial/) by queue_id substring match.
      2. If a manifest is found, set committed_at to the manifest file's
         last-commit time on origin/master (so the timestamp reflects when
         the content actually landed, not when this script ran).
      3. If no manifest is found, leave the row alone and report it so the
         operator can decide whether to requeue.

This script is idempotent and safe to re-run; rows whose committed_at is
already set are never touched.

Usage:
    /opt/local/bin/python3 backfill_committed_at.py \
        --db /home/ree/REE_Working/ree-v3/coordinator/coordinator.db \
        --assembly /home/ree/REE_Working/REE_assembly \
        [--dry-run]

Dry-run lists the proposed updates without writing. Default run prints the
audit, asks no questions, applies all writes in a single transaction, and
reports the per-row outcome.

All printed text is ASCII-only (matches the sync_daemon house style).
"""

import argparse
import os
import sqlite3
import subprocess
import sys


def _run_git(repo, *args, timeout=10):
    """Helper that returns CompletedProcess and never raises."""
    return subprocess.run(
        ["git", "-C", repo] + list(args),
        check=False, capture_output=True, text=True, timeout=timeout)


def _manifest_candidates(assembly_root, queue_id):
    """Return repo-relative paths under evidence/experiments/ whose name
    contains the lowercased queue_id sans the V3- prefix.

    Looks both at flat manifests under evidence/experiments/ and at
    evidence/experiments/_partial/<dir>/<file>.json (annotated procedural
    PASSes live there -- see V3-EXQ-590a 2026-05-25 annotation pass).

    Returns [] when no candidate exists.
    """
    needle = queue_id.lower().replace("v3-exq-", "v3_exq_").replace(
        "exq-", "exq_")
    flat = os.path.join(assembly_root, "evidence", "experiments")
    partial = os.path.join(flat, "_partial")
    candidates = []
    if os.path.isdir(flat):
        for name in os.listdir(flat):
            full = os.path.join(flat, name)
            if not os.path.isfile(full):
                continue
            if not name.endswith(".json"):
                continue
            if needle in name.lower():
                candidates.append(os.path.relpath(full, assembly_root))
    if os.path.isdir(partial):
        for sub in os.listdir(partial):
            subdir = os.path.join(partial, sub)
            if not os.path.isdir(subdir):
                continue
            for name in os.listdir(subdir):
                full = os.path.join(subdir, name)
                if not os.path.isfile(full):
                    continue
                if not name.endswith(".json"):
                    continue
                if needle in name.lower() or needle in sub.lower():
                    candidates.append(
                        os.path.relpath(full, assembly_root))
    return candidates


def _manifest_commit_time(assembly_root, rel_path):
    """Last-commit ISO timestamp for a path on origin/master, or None.

    Uses `git log -1 --format=%cI origin/master -- <path>`. Returns None
    if the path is not tracked or has no commits.
    """
    out = _run_git(
        assembly_root, "log", "-1", "--format=%cI",
        "origin/master", "--", rel_path)
    if out.returncode != 0:
        return None
    ts = out.stdout.strip()
    return ts or None


def _audit(conn, assembly_root):
    """Yield (queue_id, outcome, received_at, picked_manifest, picked_ts,
    all_candidates) tuples for every result row with committed_at NULL.
    picked_manifest / picked_ts are None when no candidate matched."""
    rows = conn.execute(
        "SELECT queue_id, outcome, received_at "
        "FROM results WHERE committed_at IS NULL "
        "ORDER BY received_at DESC").fetchall()
    for queue_id, outcome, received_at in rows:
        candidates = _manifest_candidates(assembly_root, queue_id)
        picked = None
        picked_ts = None
        # Prefer flat manifests over _partial subdirs (governance index reads
        # flat by default). Within flat, prefer the most recently-committed.
        flat_candidates = [
            c for c in candidates if "/_partial/" not in c]
        partial_candidates = [
            c for c in candidates if "/_partial/" in c]
        ordered = flat_candidates + partial_candidates
        for cand in ordered:
            ts = _manifest_commit_time(assembly_root, cand)
            if ts is not None:
                picked = cand
                picked_ts = ts
                break
        yield (
            queue_id, outcome, received_at, picked, picked_ts, candidates)


def main():
    p = argparse.ArgumentParser(
        description=("Backfill coordinator.db results.committed_at NULL "
                     "rows from REE_assembly manifest history."))
    p.add_argument(
        "--db", required=True,
        help="Path to coordinator.db (hub: /home/ree/REE_Working/"
             "ree-v3/coordinator/coordinator.db)")
    p.add_argument(
        "--assembly", required=True,
        help="Path to REE_assembly checkout (hub: /home/ree/REE_Working/"
             "REE_assembly)")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print proposed updates without writing.")
    args = p.parse_args()

    if not os.path.isfile(args.db):
        sys.stderr.write("ERROR: db not found: %s\n" % args.db)
        return 2
    if not os.path.isdir(args.assembly):
        sys.stderr.write(
            "ERROR: assembly checkout not found: %s\n" % args.assembly)
        return 2

    # Refresh origin/master so manifest_commit_time sees the canonical
    # history (the writer's commits land on origin asynchronously; a stale
    # local ref would silently miss recent landings).
    fetched = _run_git(args.assembly, "fetch", "--quiet", "origin", "master",
                       timeout=60)
    if fetched.returncode != 0:
        sys.stderr.write(
            "WARN: `git fetch origin master` failed (%s); proceeding "
            "against current local ref.\n"
            % fetched.stderr.strip()[:240])

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    print("Auditing rows with committed_at IS NULL...")
    audit_rows = list(_audit(conn, args.assembly))
    if not audit_rows:
        print("No NULL committed_at rows. Nothing to do.")
        conn.close()
        return 0

    updates = []
    missing = []
    for q, outcome, rx, picked, picked_ts, cands in audit_rows:
        if picked and picked_ts:
            print("  %s [%s] rx=%s -> %s (%s)"
                  % (q, outcome, rx, picked, picked_ts))
            updates.append((picked_ts, q, picked))
        else:
            print("  %s [%s] rx=%s -> NO MANIFEST FOUND (candidates=%d)"
                  % (q, outcome, rx, len(cands)))
            if cands:
                for c in cands[:3]:
                    print("      candidate (untracked / not on master): %s"
                          % c)
            missing.append(q)

    print()
    print("Summary: %d to update, %d unresolved (no manifest on master)"
          % (len(updates), len(missing)))

    if args.dry_run:
        print("Dry run; no writes.")
        conn.close()
        return 0

    if not updates:
        print("Nothing to write.")
        conn.close()
        return 0

    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        for ts, q, _picked in updates:
            cur.execute(
                "UPDATE results SET committed_at=? "
                "WHERE queue_id=? AND committed_at IS NULL",
                (ts, q))
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    print("Wrote %d update(s)." % len(updates))
    if missing:
        print()
        print("Operator: the following rows have no manifest on "
              "origin/master and were NOT updated. Either find the "
              "manifest (it may live on another remote / in a worker's "
              "outbox / never landed) or decide whether to requeue:")
        for q in missing:
            print("  %s" % q)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
