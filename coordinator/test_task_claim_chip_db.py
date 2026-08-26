"""Contracts for the TASK_CLAIMS.json/TASK_CHIPS.json shadow-mirror tables
and db.py helpers (schema.sql + db.py PHASE-1 additions).

See REE_assembly/evidence/planning/
task_claim_chip_coordinator_migration_plan.md section 5.2 for the reviewed
design (D1-D11). These tests pin, at the db.py layer (no git, no HTTP):

  1. table/index provisioning via a bare connect() (mirrors
     test_heartbeat_log.py's TestTableProvisioning contract for brand-new
     tables).
  2. upsert_task_claim / upsert_chip: created vs updated vs unchanged,
     idempotent re-application, resources child-table replace semantics.
  3. reconcile_task_claims / reconcile_chips: stats accounting and orphan
     detection (a DB key absent from the current source is always anomalous
     for these append-only registries -- see schema.sql's comment on
     task_claim_chip_drift_log).
  4. log_task_claim_chip_drift / task_claim_chip_drift_summary: the durable
     soak-evidence trail a human reads to judge PHASE-1's exit criterion.

ASCII-only.
"""

import json
import os
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db  # noqa: E402


def _claim(session_id="s1", claimed_at="2026-08-26T10:00:00Z", **kw):
    base = {
        "session_id": session_id,
        "claimed_at": claimed_at,
        "session_label": "label",
        "task": "do the thing",
        "resources": ["a/b.json", "c/d.py"],
        "status": "active",
    }
    base.update(kw)
    return base


def _chip(chip_ref="chip-1", **kw):
    base = {
        "chip_ref": chip_ref,
        "task_id": None,
        "origin": "headless",
        "kind": "work",
        "urgency": False,
        "session_id": "unknown",
        "session_label": "",
        "title": "a title",
        "tldr": "a tldr",
        "prompt": "[chip_ref: %s]\n\nbody" % chip_ref,
        "cwd": "/Users/dgolden/REE_Working",
        "spawned_at": "2026-08-26T09:00:00Z",
        "origin_host": "DLAPTOP",
        "status": "open",
        "claimed_by": None,
        "claimed_at": None,
        "claim_note": None,
        "claimed_host": None,
        "resolved_at": None,
        "resolved_by_session_id": None,
        "resolution_note": None,
        "resolution_note_auto": False,
        "attached_by_session_id": None,
        "attached_at": None,
    }
    base.update(kw)
    return base


class _Fixture(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="task_claim_chip_db_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)

    def tearDown(self):
        self._conn.close()


class TestTableProvisioning(_Fixture):

    def test_bare_connect_creates_all_four_tables_on_a_fresh_db(self):
        fresh_path = os.path.join(self._tmp, "fresh.db")
        conn = db.connect(fresh_path)
        try:
            for name in ("task_claims", "task_claim_resources",
                         "chip_ledger", "task_claim_chip_drift_log"):
                present = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name=?", (name,)).fetchone()
                self.assertIsNotNone(present, "%s not provisioned" % name)
            # Sibling brand-new-table contract: heartbeat_log yes,
            # heartbeats no (only init_db()'s executescript creates that).
            absent = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='heartbeats'").fetchone()
            self.assertIsNone(absent)
        finally:
            conn.close()

    def test_init_db_is_idempotent(self):
        db.init_db(self._dbpath)
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='chip_ledger'").fetchall()
        self.assertEqual(len(rows), 1)

    def test_chip_ledger_task_id_partial_unique_index_allows_many_nulls(self):
        db.upsert_chip(self._conn, _chip("c1", task_id=None))
        db.upsert_chip(self._conn, _chip("c2", task_id=None))
        rows = self._conn.execute(
            "SELECT chip_ref FROM chip_ledger WHERE task_id IS NULL"
        ).fetchall()
        self.assertEqual(len(rows), 2)

    def test_chip_ledger_task_id_unique_when_present(self):
        db.upsert_chip(self._conn, _chip("c1", task_id="tid-1"))
        with self.assertRaises(Exception):
            db.upsert_chip(self._conn, _chip("c2", task_id="tid-1"))


class TestUpsertTaskClaim(_Fixture):

    def test_first_upsert_is_created(self):
        created, changed = db.upsert_task_claim(self._conn, _claim())
        self.assertTrue(created)
        self.assertTrue(changed)

    def test_reapplying_identical_claim_is_unchanged(self):
        c = _claim()
        db.upsert_task_claim(self._conn, c)
        created, changed = db.upsert_task_claim(self._conn, c)
        self.assertFalse(created)
        self.assertFalse(changed)

    def test_status_change_is_reported_as_changed_not_created(self):
        c = _claim(status="active")
        db.upsert_task_claim(self._conn, c)
        c2 = _claim(status="done", closed_at="2026-08-26T12:00:00Z")
        created, changed = db.upsert_task_claim(self._conn, c2)
        self.assertFalse(created)
        self.assertTrue(changed)
        row = self._conn.execute(
            "SELECT status, closed_at FROM task_claims WHERE session_id=? "
            "AND claimed_at=?", (c["session_id"], c["claimed_at"])
        ).fetchone()
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["closed_at"], "2026-08-26T12:00:00Z")

    def test_composite_primary_key_distinguishes_same_session_two_claims(self):
        """8/154 live session_ids own more than one claim (plan doc 5.2.2) --
        the PK must be (session_id, claimed_at), not session_id alone."""
        db.upsert_task_claim(self._conn, _claim(
            "s1", "2026-08-26T10:00:00Z", task="first"))
        db.upsert_task_claim(self._conn, _claim(
            "s1", "2026-08-26T11:00:00Z", task="second"))
        rows = self._conn.execute(
            "SELECT claimed_at, task FROM task_claims WHERE session_id='s1' "
            "ORDER BY claimed_at").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["task"], "first")
        self.assertEqual(rows[1]["task"], "second")

    def test_resources_land_in_child_table(self):
        db.upsert_task_claim(self._conn, _claim(
            resources=["x/y.json", "z/w.py"]))
        rows = self._conn.execute(
            "SELECT resource FROM task_claim_resources WHERE session_id=? "
            "AND claimed_at=? ORDER BY resource",
            ("s1", "2026-08-26T10:00:00Z")).fetchall()
        self.assertEqual([r["resource"] for r in rows],
                         ["x/y.json", "z/w.py"])

    def test_resources_shrinking_removes_the_dropped_row(self):
        c = _claim(resources=["a.json", "b.json"])
        db.upsert_task_claim(self._conn, c)
        c2 = _claim(resources=["a.json"])
        db.upsert_task_claim(self._conn, c2)
        rows = self._conn.execute(
            "SELECT resource FROM task_claim_resources WHERE session_id=? "
            "AND claimed_at=?", ("s1", "2026-08-26T10:00:00Z")).fetchall()
        self.assertEqual([r["resource"] for r in rows], ["a.json"])

    def test_resources_cascade_delete_with_parent(self):
        db.upsert_task_claim(self._conn, _claim(resources=["a.json"]))
        self._conn.execute(
            "DELETE FROM task_claims WHERE session_id=? AND claimed_at=?",
            ("s1", "2026-08-26T10:00:00Z"))
        rows = self._conn.execute(
            "SELECT * FROM task_claim_resources").fetchall()
        self.assertEqual(len(rows), 0)

    def test_entry_json_round_trips_losslessly(self):
        c = _claim(spawned_by="igw_routine_tick.py",
                   completion_note_history=[{"amended_at": "x",
                                             "previous_note": "y"}])
        db.upsert_task_claim(self._conn, c)
        row = self._conn.execute(
            "SELECT entry_json, completion_note_history_json FROM "
            "task_claims WHERE session_id=? AND claimed_at=?",
            ("s1", "2026-08-26T10:00:00Z")).fetchone()
        self.assertEqual(json.loads(row["entry_json"]), c)
        self.assertEqual(
            json.loads(row["completion_note_history_json"]),
            c["completion_note_history"])


class TestUpsertChip(_Fixture):

    def test_first_upsert_is_created(self):
        created, changed = db.upsert_chip(self._conn, _chip())
        self.assertTrue(created)
        self.assertTrue(changed)

    def test_reapplying_identical_chip_is_unchanged(self):
        c = _chip()
        db.upsert_chip(self._conn, c)
        created, changed = db.upsert_chip(self._conn, c)
        self.assertFalse(created)
        self.assertFalse(changed)

    def test_status_transition_is_changed_not_created(self):
        db.upsert_chip(self._conn, _chip(status="open"))
        created, changed = db.upsert_chip(self._conn, _chip(
            status="done", resolved_at="2026-08-26T12:00:00Z",
            resolution_note="landed"))
        self.assertFalse(created)
        self.assertTrue(changed)

    def test_origin_host_canonicalised_raw_preserved(self):
        db.upsert_chip(self._conn, _chip(origin_host="DLAPTOP-4.local"))
        row = self._conn.execute(
            "SELECT origin_host, origin_host_raw FROM chip_ledger "
            "WHERE chip_ref='chip-1'").fetchone()
        self.assertEqual(row["origin_host"], "DLAPTOP")
        self.assertEqual(row["origin_host_raw"], "DLAPTOP-4.local")

    def test_cloud_fleet_host_not_collapsed(self):
        """canonical_machine_name is an ALLOWLIST -- ree-cloud-1..5 must not
        be treated as suffix-drift variants of one machine."""
        db.upsert_chip(self._conn, _chip("c1", claimed_host="ree-cloud-4"))
        db.upsert_chip(self._conn, _chip("c2", claimed_host="ree-cloud-5"))
        rows = {r["chip_ref"]: r["claimed_host"] for r in self._conn.execute(
            "SELECT chip_ref, claimed_host FROM chip_ledger").fetchall()}
        self.assertEqual(rows["c1"], "ree-cloud-4")
        self.assertEqual(rows["c2"], "ree-cloud-5")

    def test_urgency_bool_stored_as_integer(self):
        db.upsert_chip(self._conn, _chip(urgency=True))
        row = self._conn.execute(
            "SELECT urgency FROM chip_ledger WHERE chip_ref='chip-1'"
        ).fetchone()
        self.assertEqual(row["urgency"], 1)

    def test_archived_chip_keeps_row_prompt_nulled(self):
        """D5: archiving strips FIELDS and keeps the ROW -- never DELETE."""
        db.upsert_chip(self._conn, _chip(status="done"))
        archived = _chip(status="done", prompt=None, resolution_note=None,
                         archived={"file": "chip_archive/2026-08.json",
                                   "month": "2026-08",
                                   "fields": ["prompt", "resolution_note"],
                                   "at": "2026-09-10T00:00:00Z"})
        created, changed = db.upsert_chip(self._conn, archived)
        self.assertFalse(created)
        row = self._conn.execute(
            "SELECT prompt, resolution_note, archived_json FROM chip_ledger "
            "WHERE chip_ref='chip-1'").fetchone()
        self.assertIsNone(row["prompt"])
        self.assertIsNone(row["resolution_note"])
        self.assertIsNotNone(row["archived_json"])
        count = self._conn.execute(
            "SELECT COUNT(*) c FROM chip_ledger").fetchone()["c"]
        self.assertEqual(count, 1, "archiving must not delete the row")


class TestReconcileTaskClaims(_Fixture):

    def test_first_reconcile_reports_all_new(self):
        stats = db.reconcile_task_claims(
            self._conn, [_claim("s1"), _claim("s2", "2026-08-26T11:00:00Z")])
        self.assertEqual(stats["n_git"], 2)
        self.assertEqual(stats["n_new"], 2)
        self.assertEqual(stats["n_updated"], 0)
        self.assertEqual(stats["n_db"], 2)
        self.assertEqual(stats["orphans"], [])

    def test_second_reconcile_with_no_changes_reports_unchanged(self):
        claims = [_claim("s1"), _claim("s2", "2026-08-26T11:00:00Z")]
        db.reconcile_task_claims(self._conn, claims)
        stats = db.reconcile_task_claims(self._conn, claims)
        self.assertEqual(stats["n_new"], 0)
        self.assertEqual(stats["n_updated"], 0)
        self.assertEqual(stats["n_unchanged"], 2)
        self.assertEqual(stats["orphans"], [])

    def test_key_missing_from_new_source_is_reported_as_orphan(self):
        """TASK_CLAIMS.json entries are never deleted in practice -- an
        orphan here is always anomalous, never expected steady state."""
        db.reconcile_task_claims(
            self._conn, [_claim("s1"), _claim("s2", "2026-08-26T11:00:00Z")])
        stats = db.reconcile_task_claims(self._conn, [_claim("s1")])
        self.assertEqual(stats["orphans"], ["s2|2026-08-26T11:00:00Z"])

    def test_malformed_entry_missing_keys_is_skipped_not_raised(self):
        stats = db.reconcile_task_claims(
            self._conn, [_claim("s1"), {"task": "no session_id or claimed_at"}])
        self.assertEqual(stats["n_git"], 2)
        self.assertEqual(stats["n_new"], 1)


class TestReconcileChips(_Fixture):

    def test_first_reconcile_reports_all_new(self):
        stats = db.reconcile_chips(self._conn, [_chip("c1"), _chip("c2")])
        self.assertEqual(stats["n_new"], 2)
        self.assertEqual(stats["orphans"], [])

    def test_orphan_detection(self):
        db.reconcile_chips(self._conn, [_chip("c1"), _chip("c2")])
        stats = db.reconcile_chips(self._conn, [_chip("c1")])
        self.assertEqual(stats["orphans"], ["c2"])

    def test_updated_vs_unchanged_accounting(self):
        db.reconcile_chips(self._conn, [_chip("c1"), _chip("c2")])
        stats = db.reconcile_chips(self._conn, [
            _chip("c1"), _chip("c2", status="withdrawn",
                              resolved_at="2026-08-26T12:00:00Z")])
        self.assertEqual(stats["n_unchanged"], 1)
        self.assertEqual(stats["n_updated"], 1)


class TestDriftLog(_Fixture):

    def test_healthy_reconcile_logs_zero_diverged(self):
        claim_stats = db.reconcile_task_claims(self._conn, [_claim("s1")])
        chip_stats = db.reconcile_chips(self._conn, [_chip("c1")])
        diverged = db.log_task_claim_chip_drift(
            self._conn, "deadbeef", claim_stats, chip_stats)
        self.assertEqual(diverged, 0)
        summary = db.task_claim_chip_drift_summary(self._conn)
        self.assertEqual(summary["total_ticks"], 1)
        self.assertEqual(summary["diverged_ticks"], 0)
        self.assertEqual(summary["recent"][0]["source_ref"], "deadbeef")

    def test_orphan_marks_the_tick_diverged(self):
        db.reconcile_task_claims(self._conn, [_claim("s1")])
        claim_stats = db.reconcile_task_claims(self._conn, [])
        chip_stats = db.reconcile_chips(self._conn, [])
        diverged = db.log_task_claim_chip_drift(
            self._conn, "sha2", claim_stats, chip_stats)
        self.assertEqual(diverged, 1)
        summary = db.task_claim_chip_drift_summary(self._conn)
        self.assertEqual(summary["diverged_ticks"], 1)
        detail = json.loads(summary["recent"][0]["detail"])
        self.assertIn("s1|2026-08-26T10:00:00Z", detail["claim_orphans"])

    def test_summary_limit_is_respected(self):
        for i in range(5):
            claim_stats = db.reconcile_task_claims(self._conn, [])
            chip_stats = db.reconcile_chips(self._conn, [])
            db.log_task_claim_chip_drift(
                self._conn, "sha-%d" % i, claim_stats, chip_stats)
        summary = db.task_claim_chip_drift_summary(self._conn, limit=2)
        self.assertEqual(summary["total_ticks"], 5)
        self.assertEqual(len(summary["recent"]), 2)
        # ORDER BY id DESC -- most recent first.
        self.assertEqual(summary["recent"][0]["source_ref"], "sha-4")


if __name__ == "__main__":
    unittest.main()
