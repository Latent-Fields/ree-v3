"""Seed the coordinator DB from experiment_queue.json.

Phase 0 standup step, and also the disaster-recovery reseed path. Safe to
re-run: upsert_experiment is idempotent on queue_id.

Usage:
  python3 seed_from_queue.py [--queue PATH] [--db PATH]

All printed text is ASCII-only.
"""

import argparse
import json
import os
import sys

import db

DEFAULT_QUEUE = os.path.join(
    os.path.dirname(__file__), "..", "experiment_queue.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=os.environ.get(
        "COORDINATOR_QUEUE_FILE", DEFAULT_QUEUE))
    ap.add_argument("--db", default=os.environ.get(
        "COORDINATOR_DB", os.path.join(
            os.path.dirname(__file__), "coordinator.db")))
    args = ap.parse_args()

    if not os.path.exists(args.queue):
        sys.stderr.write("queue file not found: %s\n" % args.queue)
        return 2

    with open(args.queue, "r", encoding="utf-8") as fh:
        qdata = json.load(fh)
    items = qdata.get("items", [])

    db.init_db(args.db)
    conn = db.connect(args.db)
    n = 0
    try:
        conn.execute("BEGIN")
        for item in items:
            if not item.get("queue_id"):
                continue
            db.upsert_experiment(conn, item)
            n += 1
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    sys.stdout.write("seeded %d items into %s\n" % (n, args.db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
