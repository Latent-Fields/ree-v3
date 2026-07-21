"""Contract: a NEW queue item must never be absorbed into a TERMINAL row.

The defect (V3-EXQ-728a, confirmed 2026-07-20/21 on the hub coordinator DB)
------------------------------------------------------------------------
Under Phase 3 the coordinator DB owns the queue and `experiment_queue.json`
is a DERIVED view. Two properties combine badly:

  1. `reconcile_once(upsert_only=True, claim_authority='coordinator')`
     upserted EVERY item in the git queue file with `preserve_claim=True`.
     For a queue_id whose DB row was already terminal, that kept
     status='completed' while OVERWRITING note / item_json / priority /
     script with the new experiment's content.
  2. `_materialise_queue_from_db` emits only non-terminal rows, so the very
     next phase3_queue_writer tick DELETED the freshly committed entry from
     experiment_queue.json.

Net effect: the id was burned. The row was simultaneously note-updated and
unservable, the git entry vanished, and the experiment never ran -- with no
error at any layer. The operator's commit succeeded, reconcile succeeded,
the writer succeeded. Live case: V3-EXQ-728a was a real ZWORLD-GUARD entry
removed FAIL at 2026-07-20T15:54:17Z; ree-v3 b523b9c reused the id at
19:01:58Z for unrelated SD-070 adoption-validation content; the entry was
gone from the next snapshot (dbafb66) and had to be re-queued as 728b
(0e7d33497f). No results row for 728a ever existed.

POST /queue/add already 409s on exactly this case. It did not fire because
the ingress was a git commit reconciled in, not the HTTP endpoint -- so the
guard existed on one of the two ingresses only.

Contracts
---------
  C1. THE REGRESSION THAT MATTERS: after reconcile ingests a NEW item under
      a terminal queue_id, the DB must not hold a row that is both
      note-updated and unservable. The terminal row is left byte-untouched.
  C2. The refusal is LOUD -- the burned id is named on stderr and counted
      as a state divergence, not swallowed.
  C3. `force_rerun: true` is the deliberate opt-in: it resurrects the row to
      pending so the item is served AND materialised, rather than leaving a
      note-updated husk. The removal audit trail survives (pinned by
      test_queue_removal_reason_recorded C4).
  C4. Non-terminal rows are unaffected -- claim state is still preserved and
      metadata still refreshed, so the guard cannot wedge a live run.
  C5. Shadow/Phase-1 semantics (upsert_only=False) are unchanged; the guard
      is Phase-3-only, where the DB (not git) is the status authority.
  C6. phase3_queue_writer's terminal-drop check WARNs when the snapshot
      deletes an entry whose DB row is terminal with NO results row -- the
      phantom / burned-id signature -- and stays silent for an ordinary
      completion sweep, so the warning does not become noise.

All printed text is ASCII-only.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "coordinator"))

import db  # noqa: E402
import sync_daemon  # noqa: E402


def _item(queue_id, note, **extra):
    item = {
        "queue_id": queue_id,
        "script": "experiments/%s.py" % queue_id.lower().replace("-", "_"),
        "priority": 5,
        "note": note,
    }
    item.update(extra)
    return item


class TerminalRowBurnGuardTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "coordinator.db")
        self.queue_path = os.path.join(self.tmp, "experiment_queue.json")
        db.init_db(self.db_path)
        self.conn = db.connect(self.db_path)
        self.addCleanup(self.conn.close)

    def _write_queue(self, items):
        with open(self.queue_path, "w", encoding="utf-8") as fh:
            json.dump({"schema_version": "v1", "calibration": {},
                       "items": items}, fh)

    def _reconcile(self, upsert_only=True, claim_authority="coordinator"):
        """Run one authoritative reconcile tick, capturing stderr."""
        buf = io.StringIO()
        with redirect_stderr(buf):
            n, div = sync_daemon.reconcile_once(
                self.conn, self.queue_path,
                claim_authority=claim_authority,
                upsert_only=upsert_only)
        return n, div, buf.getvalue()

    def _row(self, queue_id):
        return self.conn.execute(
            "SELECT queue_id, status, note, item_json, priority, script, "
            "removal_reason, removed_at, updated_at FROM experiments "
            "WHERE queue_id=?", (queue_id,)).fetchone()

    def _materialised_ids(self):
        doc = sync_daemon._materialise_queue_from_db(self.conn, {})
        return [it["queue_id"] for it in doc["items"]]

    def _burn_the_id(self, qid="V3-EXQ-728a"):
        """Reproduce the live precondition: a real entry that ran and went
        terminal, then a NEW experiment committed under the same id."""
        self._write_queue([_item(qid, "ZWORLD-GUARD RE-RUN")])
        self._reconcile()
        db.mark_queue_removed(self.conn, qid, "FAIL")
        self._write_queue([_item(qid, "SD-070 ADOPTION VALIDATION")])
        return qid

    # C1 -----------------------------------------------------------------
    def test_c1_new_item_never_lands_in_a_terminal_row(self):
        qid = self._burn_the_id()
        before = self._row(qid)

        self._reconcile()

        after = self._row(qid)
        # The precise failure shape: note-updated AND unservable.
        self.assertFalse(
            after["status"] in db.TERMINAL_STATUSES
            and "SD-070" in (after["note"] or ""),
            "row is both terminal (unservable) and carries the NEW "
            "experiment's note -- the queue_id is burned")
        # Nothing about the terminal row moved at all.
        for field in ("status", "note", "item_json", "priority", "script",
                      "removal_reason", "removed_at", "updated_at"):
            self.assertEqual(after[field], before[field],
                             "terminal row field %r was mutated" % field)
        # And the writer would still drop it from the file, so the item must
        # not be silently "accepted".
        self.assertNotIn(qid, self._materialised_ids())

    # C2 -----------------------------------------------------------------
    def test_c2_refusal_is_loud(self):
        qid = self._burn_the_id()
        _, divergences, err = self._reconcile()

        self.assertIn(qid, err, "burned id must be named on stderr")
        self.assertIn("REFUSED", err)
        self.assertIn("force_rerun", err,
                      "the operator needs to be told the way out")
        self.assertGreaterEqual(
            divergences, 1,
            "a refused item is a git-vs-DB state divergence and must be "
            "counted, not swallowed")
        self.assertTrue(err.isascii(), "stderr output must be ASCII-only")

    # C3 -----------------------------------------------------------------
    def test_c3_force_rerun_resurrects_the_row_properly(self):
        qid = self._burn_the_id()
        self._write_queue([
            _item(qid, "SD-070 ADOPTION VALIDATION", force_rerun=True)])

        _, _, err = self._reconcile()

        row = self._row(qid)
        self.assertEqual(row["status"], "pending",
                         "force_rerun must make the row servable again")
        self.assertIn("SD-070", row["note"])
        self.assertIn(qid, self._materialised_ids(),
                      "a force_rerun row must survive re-materialisation")
        self.assertIn(qid, err)
        # The audit trail of the earlier terminal event is NOT erased --
        # test_queue_removal_reason_recorded C4 depends on this.
        self.assertEqual(row["removal_reason"], "FAIL")
        self.assertTrue(row["removed_at"])

    # C4 -----------------------------------------------------------------
    def test_c4_non_terminal_rows_are_unaffected(self):
        qid = "V3-EXQ-900"
        self._write_queue([_item(qid, "original")])
        self._reconcile()
        self.conn.execute(
            "UPDATE experiments SET status='claimed', "
            "claimed_by_machine='ree-cloud-2' WHERE queue_id=?", (qid,))

        self._write_queue([_item(qid, "edited note")])
        _, _, err = self._reconcile()

        row = self._row(qid)
        self.assertEqual(row["status"], "claimed",
                         "an in-flight claim must still be preserved")
        self.assertEqual(row["note"], "edited note",
                         "metadata refresh on live rows must still happen")
        self.assertNotIn("REFUSED", err)

    # C5 -----------------------------------------------------------------
    def test_c5_shadow_mode_semantics_unchanged(self):
        qid = self._burn_the_id()
        # Phase 1 shadow: git is the status authority, so a git item with no
        # status resets the row to pending. Deliberately NOT guarded.
        _, _, err = self._reconcile(upsert_only=False, claim_authority="git")
        row = self._row(qid)
        self.assertEqual(row["status"], "pending")
        self.assertNotIn("REFUSED", err,
                         "the guard is Phase-3-only; shadow must not warn")

    # C6 -----------------------------------------------------------------
    def test_c6_writer_warns_when_dropping_a_terminal_row_with_no_result(self):
        qid = self._burn_the_id()
        self._reconcile()  # refused; row still terminal, no results row

        buf = io.StringIO()
        with redirect_stderr(buf):
            flagged = sync_daemon._warn_on_terminal_drops_without_results(
                self.conn, {qid})
        err = buf.getvalue()

        self.assertEqual(flagged, [qid])
        self.assertIn(qid, err)
        self.assertIn("NO results row", err)
        self.assertTrue(err.isascii())

    def test_c6b_ordinary_completion_sweep_is_silent(self):
        qid = "V3-EXQ-901"
        self._write_queue([_item(qid, "ran fine")])
        self._reconcile()
        db.mark_queue_removed(self.conn, qid, "PASS")
        self.conn.execute(
            "INSERT INTO results (run_id, queue_id, machine, outcome, "
            "received_at) VALUES (?,?,?,?,?)",
            ("v3_exq_901_x_v3", qid, "ree-cloud-2", "PASS",
             "2026-07-20T10:00:00Z"))

        buf = io.StringIO()
        with redirect_stderr(buf):
            flagged = sync_daemon._warn_on_terminal_drops_without_results(
                self.conn, {qid})

        self.assertEqual(flagged, [],
                         "a completed run WITH a results row is the normal "
                         "sweep and must not warn -- otherwise the signal "
                         "drowns in noise")
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
