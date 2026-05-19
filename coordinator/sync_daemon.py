"""Sync daemon.

PHASE 1 (shadow) behaviour -- the only behaviour enabled by default:
  * Periodically read experiment_queue.json (git is authoritative in
    shadow) and reconcile the coordinator DB mirror to match it, so the
    coordinator's claim logic always evaluates against fresh state.
  * Detect STATE-level divergence: cases where the mirror's claim state
    disagrees with the authoritative queue file. These are logged so the
    operator can confirm the coordinator would have made the same
    decisions before any cutover.
  * It does NOT write git. Read-only on the queue file. No autostash, no
    rebase -- this daemon is structurally incapable of the failure class
    the whole project exists to remove.

PHASE 3 (authoritative) behaviour is present but guarded OFF: becoming the
sole git writer (commit result manifests, push, snapshot queue) only
activates when SYNC_MODE=authoritative AND --i-understand-phase3 is passed.
Stubbed deliberately; do not enable until Phase 1 has proven out.

All printed text is ASCII-only.
"""

import argparse
import json
import os
import sys
import time

import db

DEFAULT_QUEUE = os.path.join(
    os.path.dirname(__file__), "..", "experiment_queue.json")


def reconcile_once(conn, queue_path):
    """Make the mirror match the authoritative queue file. Returns
    (n_items, n_state_divergences)."""
    if not os.path.exists(queue_path):
        return (0, 0)
    with open(queue_path, "r", encoding="utf-8") as fh:
        qdata = json.load(fh)
    items = {it["queue_id"]: it for it in qdata.get("items", [])
             if it.get("queue_id")}

    divergences = 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        mirror = {r["queue_id"]: r for r in conn.execute(
            "SELECT queue_id, status, claimed_by_machine FROM experiments"
        ).fetchall()}

        for qid, item in items.items():
            cb = (item.get("claimed_by") or {}).get("machine")
            git_status = item.get("status", "pending")
            m = mirror.get(qid)
            if m is not None:
                # State-level shadow check: did the coordinator's mirror
                # (driven only by reported claim attempts) end up agreeing
                # with what git actually recorded?
                mirror_claimed = m["status"] == "claimed"
                git_claimed = git_status == "claimed"
                if mirror_claimed != git_claimed or (
                        git_claimed and cb and
                        m["claimed_by_machine"] not in (None, cb)):
                    divergences += 1
                    db.log_claim(
                        conn, qid, cb or "?", git_status,
                        m["status"],
                        detail="state-reconcile mirror=%s/%s git=%s/%s" % (
                            m["status"], m["claimed_by_machine"],
                            git_status, cb))
            db.upsert_experiment(conn, item)

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


def phase3_git_writer(*_a, **_k):
    # PHASE 3 STUB. Sole-git-writer: drain results table -> write manifest
    # files into the REE_assembly evidence checkout -> commit -> push;
    # snapshot the live queue back to experiment_queue.json. Intentionally
    # not implemented for the shadow build. Do not wire this up until
    # /shadow/divergence has been ~zero under real multi-machine load.
    raise NotImplementedError("Phase 3 git writer is not enabled in shadow")


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
        phase3_git_writer()  # raises NotImplementedError by design
        return 2

    db.init_db(args.db)
    while True:
        conn = db.connect(args.db)
        try:
            n, div = reconcile_once(conn, args.queue)
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
