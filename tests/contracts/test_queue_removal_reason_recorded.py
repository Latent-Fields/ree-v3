"""Contract: a terminal queue row records WHY it went terminal.

Background -- the "phantom completion" taxonomy defect (diagnosed 2026-07-20,
session cranky-pascal-46cd9a, live case V3-EXQ-699a):

  `experiments.status='completed'` is overloaded. It means "no longer
  claimable", NOT "ran to a scientific outcome". Three very different events
  all wrote a byte-identical row:

    - a runner ERROR      -- coordinator_client.report_queue_remove(qid, "ERROR")
    - a scientific FAIL   -- report_queue_remove(qid, "FAIL")
    - an operator cancel  -- POST /queue/remove (reason free-text, or absent)

  db.mark_queue_removed() ALREADY took a `reason` argument and threw it away,
  so the standard detector query

      status='completed' LEFT JOIN results -> no results row

  could not separate a crash-before-manifest from a deliberate cancellation.
  All 23 all-time "phantom completions" were reported under that one label.

  V3-EXQ-699a is the worked example, and it was NOT a crash: a
  /failure-autopsy session POSTed /queue/remove at 2026-07-20T08:45:21Z to
  cancel an 11.7-hour in-flight run and queue its repaired successor 699b
  (`supersedes: V3-EXQ-699a`). Correct operator behaviour, recorded as if it
  were a completion. The runner's own SIGTERM interceptor behaved correctly
  and is NOT implicated -- see test_runner_sigterm_no_phantom_completion.py,
  and note V3-EXQ-708a hit the identical interceptor the same morning and was
  re-run normally.

Fix under test: persist the reason (removal_reason + removed_at). Purely
additive -- no status transition, claim behaviour, or reader is changed.

Contracts:
  C1. The migration is additive and safe on a PRE-migration DB: it adds the
      columns in place, leaves existing rows untouched with NULL reason, and
      is idempotent across repeated connect() calls.
  C2. mark_queue_removed persists the reason it is given, so ERROR / FAIL /
      operator-cancel become distinguishable rather than identical.
  C3. A reason-less removal (the hand `POST /queue/remove` path) records NULL
      reason but still stamps removed_at -- NULL is itself the signal that the
      removal came through the operator path, not the runner.
  C4. THE REGRESSION THAT MATTERS: a reconcile tick must not clobber the
      recorded reason. reconcile_once() re-upserts every item still present in
      the derived queue file, and the file lags the DB (that lag is exactly why
      699a's updated_at read 08:50:17Z when the real flip was 08:45:21Z). If
      upsert_experiment overwrote these columns, the reason would be erased
      minutes after it was written and the fix would silently do nothing.
  C5. mark_queue_removed still returns False for an absent row (unchanged
      contract -- app.py reports it as `applied`).
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT / "coordinator"))

import db  # noqa: E402


# The experiments table exactly as it stood BEFORE this migration, so C1
# exercises a real upgrade rather than a freshly-created schema.
PRE_MIGRATION_SCHEMA = """
CREATE TABLE experiments (
    queue_id            TEXT PRIMARY KEY,
    script              TEXT NOT NULL,
    priority            INTEGER NOT NULL DEFAULT 1,
    machine_affinity    TEXT NOT NULL DEFAULT 'any',
    status              TEXT NOT NULL DEFAULT 'pending',
    estimated_minutes   REAL,
    supersedes          TEXT,
    claim_id            TEXT,
    backlog_id          TEXT,
    title               TEXT,
    note                TEXT,
    force_rerun         INTEGER NOT NULL DEFAULT 0,
    claimed_by_machine  TEXT,
    claimed_at          TEXT,
    item_json           TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
"""


def _item(queue_id):
    return {"queue_id": queue_id, "script": "experiments/x.py",
            "item_json": "{}"}


class QueueRemovalReasonTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "coordinator.db")

    def _pre_migration_db(self):
        """Materialise a DB on the OLD schema with one legacy terminal row."""
        raw = sqlite3.connect(self.db_path)
        raw.executescript(PRE_MIGRATION_SCHEMA)
        raw.execute(
            "INSERT INTO experiments (queue_id, script, item_json, "
            "updated_at, status) VALUES (?,?,?,?,?)",
            ("V3-EXQ-LEGACY", "experiments/x.py", "{}",
             "2026-01-01T00:00:00Z", "completed"))
        raw.commit()
        raw.close()

    def _cols(self, conn):
        return {row[1] for row in conn.execute(
            "PRAGMA table_info(experiments)")}

    def _row(self, conn, queue_id):
        return conn.execute(
            "SELECT status, removal_reason, removed_at FROM experiments "
            "WHERE queue_id=?", (queue_id,)).fetchone()

    # C1 -----------------------------------------------------------------
    def test_c1_migration_is_additive_and_idempotent(self):
        self._pre_migration_db()

        raw_cols = {row[1] for row in sqlite3.connect(self.db_path).execute(
            "PRAGMA table_info(experiments)")}
        self.assertNotIn("removal_reason", raw_cols,
                         "fixture must start PRE-migration")

        conn = db.connect(self.db_path)
        self.assertIn("removal_reason", self._cols(conn))
        self.assertIn("removed_at", self._cols(conn))

        # the pre-existing row survives untouched, with a NULL reason
        legacy = self._row(conn, "V3-EXQ-LEGACY")
        self.assertEqual(legacy["status"], "completed")
        self.assertIsNone(legacy["removal_reason"])
        self.assertIsNone(legacy["removed_at"])
        conn.close()

        # idempotent: connecting again must not raise (duplicate column)
        for _ in range(2):
            db.connect(self.db_path).close()

    # C2 -----------------------------------------------------------------
    def test_c2_removal_reason_is_persisted(self):
        db.init_db(self.db_path)
        conn = db.connect(self.db_path)
        for qid, reason in (("V3-EXQ-E", "ERROR"), ("V3-EXQ-F", "FAIL")):
            db.upsert_experiment(conn, _item(qid))
            self.assertTrue(db.mark_queue_removed(conn, qid, reason))
            row = self._row(conn, qid)
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["removal_reason"], reason)
            self.assertTrue(row["removed_at"])

        # the whole point: the two are no longer indistinguishable
        self.assertNotEqual(self._row(conn, "V3-EXQ-E")["removal_reason"],
                            self._row(conn, "V3-EXQ-F")["removal_reason"])
        conn.close()

    # C3 -----------------------------------------------------------------
    def test_c3_reasonless_operator_removal_records_null_but_stamps_time(self):
        db.init_db(self.db_path)
        conn = db.connect(self.db_path)
        db.upsert_experiment(conn, _item("V3-EXQ-OP"))
        self.assertTrue(db.mark_queue_removed(conn, "V3-EXQ-OP", None))
        row = self._row(conn, "V3-EXQ-OP")
        self.assertEqual(row["status"], "completed")
        self.assertIsNone(row["removal_reason"])
        self.assertTrue(row["removed_at"],
                        "removed_at must be stamped even without a reason")
        conn.close()

    # C4 -----------------------------------------------------------------
    def test_c4_reconcile_upsert_does_not_clobber_reason(self):
        db.init_db(self.db_path)
        conn = db.connect(self.db_path)
        db.upsert_experiment(conn, _item("V3-EXQ-699A"))
        db.mark_queue_removed(conn, "V3-EXQ-699A", "ERROR")
        before = self._row(conn, "V3-EXQ-699A")

        # The Phase-3 reconcile path (claim_authority='coordinator' ->
        # preserve_claim=True). It must keep BOTH the removal record and the
        # terminal status. Repeated, because the derived queue file can lag
        # for many ticks -- that lag is why 699a's updated_at read 08:50:17Z
        # when the real flip was 08:45:21Z.
        for _ in range(3):
            db.upsert_experiment(conn, _item("V3-EXQ-699A"),
                                 preserve_claim=True)
            after = self._row(conn, "V3-EXQ-699A")
            self.assertEqual(after["removal_reason"], before["removal_reason"],
                             "Phase-3 reconcile tick erased removal_reason")
            self.assertEqual(after["removed_at"], before["removed_at"],
                             "Phase-3 reconcile tick erased removed_at")
            self.assertEqual(after["status"], "completed",
                             "Phase-3 reconcile tick resurrected a "
                             "terminal row")

        # The legacy shadow path (claim_authority='git' -> preserve_claim=
        # False) DOES reset status from the item, i.e. back to 'pending'.
        # That is pre-existing, deliberate shadow-mode behaviour and is not
        # changed here -- asserted so the difference is on the record rather
        # than discovered by surprise. What matters for this fix is that the
        # removal record SURVIVES even then, so the audit trail is not lost
        # if a row is ever resurrected.
        db.upsert_experiment(conn, _item("V3-EXQ-699A"), preserve_claim=False)
        after = self._row(conn, "V3-EXQ-699A")
        self.assertEqual(after["status"], "pending")
        self.assertEqual(after["removal_reason"], before["removal_reason"])
        self.assertEqual(after["removed_at"], before["removed_at"])
        conn.close()

    # C5 -----------------------------------------------------------------
    def test_c5_absent_row_still_returns_false(self):
        db.init_db(self.db_path)
        conn = db.connect(self.db_path)
        self.assertFalse(db.mark_queue_removed(conn, "V3-EXQ-NOPE", "ERROR"))
        conn.close()


if __name__ == "__main__":
    unittest.main()
