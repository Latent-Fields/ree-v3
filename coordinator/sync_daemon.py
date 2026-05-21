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


def phase3_git_writer(
    conn,
    queue_path,
    *,
    ree_assembly_path=None,
    dry_run=False,
):
    """Sole git writer tick (Phase 3).

    Safe no-op while PHASE3_GIT_WRITER_READY is False: logs intent and
    returns False so main() refuses authoritative mode.

    Planned steps (TODO -- implement in order):
      1. SELECT results WHERE committed_at IS NULL ORDER BY received_at
      2. Write manifest bytes to evidence/experiments/... (idempotent paths)
      3. git add/commit/push REE_assembly (no pull --rebase --autostash)
      4. UPDATE results SET committed_at for shipped run_ids
      5. Snapshot queue removals/completions -> experiment_queue.json + push
         ree-v3 main (or hub queue checkout per deploy layout)
      6. Write derived runner_heartbeats/*.json + runner_status from DB
         (replaces per-runner git heartbeat push)

    Returns True only when a full tick completed (or dry_run simulated).
    """
    asm = ree_assembly_path or PHASE3_REE_ASSEMBLY
    if not PHASE3_GIT_WRITER_READY:
        sys.stderr.write(
            "[phase3] git writer stub (PHASE3_GIT_WRITER_READY=False); "
            "no git writes performed\n")
        return False
    if dry_run:
        sys.stdout.write(
            "[phase3] dry_run tick (writer ready but no filesystem/git IO)\n")
        return True
    # TODO step 1: pending = conn.execute(
    #     "SELECT run_id, queue_id, manifest ... FROM results "
    #     "WHERE committed_at IS NULL ...").fetchall()
    # TODO step 2-3: write manifests under asm / commit / push master
    # TODO step 4: mark committed_at
    # TODO step 5: queue snapshot from experiments table -> queue_path
    # TODO step 6: derived telemetry files under asm/evidence/experiments/
    sys.stderr.write(
        "[phase3] PHASE3_GIT_WRITER_READY=True but body not implemented\n")
    return False


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
