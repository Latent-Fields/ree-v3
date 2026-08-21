"""Contracts for the append-only heartbeat_log history table (schema.sql).

`heartbeats` (PRIMARY KEY(machine)) is upserted on every POST /heartbeat and
holds only the LATEST tick per machine -- there was no queryable history of
fleet telemetry beyond git commit history. heartbeat_log fixes that, but
ONLY if it stays cheap: the runner POSTs /heartbeat every ~5s while an
experiment is running (experiment_runner.STATUS_WRITE_INTERVAL), so logging
every tick unconditionally would be ~720 rows/machine/hour -- the same
growth-without-bound mistake the retired 30-minute git heartbeat liveness
tick was pulled for. These contracts pin: (1) a row is appended only on a
(state, current_exq) TRANSITION, never on an unchanged repeat heartbeat;
(2) the retention trim actually deletes old rows and is throttled so it
does not run on every single write; (3) the table is available to write
into whether it was provisioned via init_db()'s executescript or via a
bare connect() (mirrors the "live DB picks it up without a rebuild"
contract the other _migrate_* functions in db.py document).

ASCII-only.
"""

import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db  # noqa: E402


class _Fixture(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="heartbeat_log_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        # Isolate the module-global trim throttle between tests.
        self._orig_last_trim = db._last_heartbeat_log_trim[0]
        db._last_heartbeat_log_trim[0] = 0.0

    def tearDown(self):
        db._last_heartbeat_log_trim[0] = self._orig_last_trim
        self._conn.close()

    def _log_rows(self, machine=None):
        if machine is None:
            return self._conn.execute(
                "SELECT * FROM heartbeat_log ORDER BY id").fetchall()
        return self._conn.execute(
            "SELECT * FROM heartbeat_log WHERE machine=? ORDER BY id",
            (machine,)).fetchall()


class TestTransitionLogging(_Fixture):

    def test_first_heartbeat_logs_one_transition(self):
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {}, {})
        rows = self._log_rows("ree-cloud-2")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "running")
        self.assertEqual(rows[0]["current_exq"], "V3-EXQ-001")

    def test_repeated_identical_heartbeat_does_not_log(self):
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {}, {})
        for _ in range(5):
            db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                                "V3-EXQ-001", {"episodes_done": 3}, {})
        rows = self._log_rows("ree-cloud-2")
        self.assertEqual(len(rows), 1,
                         "progress-only changes must not append a row")

    def test_current_exq_switch_logs_a_new_row(self):
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {}, {})
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-002", {}, {})
        rows = self._log_rows("ree-cloud-2")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["current_exq"], "V3-EXQ-002")

    def test_state_flip_logs_a_new_row(self):
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {}, {})
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "idle", None, {}, {})
        rows = self._log_rows("ree-cloud-2")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["state"], "idle")
        self.assertIsNone(rows[1]["current_exq"])

    def test_machines_are_tracked_independently(self):
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {}, {})
        db.upsert_heartbeat(self._conn, "ree-cloud-3", "running",
                            "V3-EXQ-001", {}, {})
        self.assertEqual(len(self._log_rows("ree-cloud-2")), 1)
        self.assertEqual(len(self._log_rows("ree-cloud-3")), 1)

    def test_payload_json_stored_verbatim_on_transition(self):
        payload = '{"machine": "ree-cloud-2", "recent_lines": ["a", "b"]}'
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {}, {}, payload_json=payload)
        rows = self._log_rows("ree-cloud-2")
        self.assertEqual(rows[0]["payload_json"], payload)

    def test_heartbeats_table_unaffected(self):
        """Regression: the read-before-write for transition detection must
        not change what lands in the current-state `heartbeats` row."""
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {"episodes_done": 1}, {},
                            seconds_elapsed=10, seconds_remaining=90)
        row = self._conn.execute(
            "SELECT * FROM heartbeats WHERE machine=?",
            ("ree-cloud-2",)).fetchone()
        self.assertEqual(row["state"], "running")
        self.assertEqual(row["current_exq"], "V3-EXQ-001")
        self.assertEqual(row["seconds_elapsed"], 10)
        self.assertEqual(row["seconds_remaining"], 90)


class TestRetentionTrim(_Fixture):

    def test_trim_deletes_rows_older_than_retention(self):
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {}, {})
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "idle", None, {}, {})
        self._conn.execute(
            "UPDATE heartbeat_log SET observed_at='2020-01-01T00:00:00Z' "
            "WHERE state='running'")
        deleted = db.trim_heartbeat_log(self._conn)
        self.assertEqual(deleted, 1)
        rows = self._log_rows("ree-cloud-2")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "idle")

    def test_trim_respects_custom_retention_days(self):
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {}, {})
        eight_days_ago = (
            db.datetime.now(db.timezone.utc) - db.timedelta(days=8)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._conn.execute(
            "UPDATE heartbeat_log SET observed_at=?", (eight_days_ago,))
        # 30-day default retention: not yet due for deletion.
        self.assertEqual(db.trim_heartbeat_log(self._conn), 0)
        # A 7-day retention window: now past due.
        self.assertEqual(
            db.trim_heartbeat_log(self._conn, retention_days=7), 1)

    def test_trim_is_a_noop_on_an_empty_table(self):
        self.assertEqual(db.trim_heartbeat_log(self._conn), 0)

    def test_maybe_trim_is_throttled_by_interval(self):
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {}, {})
        self._conn.execute(
            "UPDATE heartbeat_log SET observed_at='2020-01-01T00:00:00Z'")
        clock = [1_000_000.0]
        with mock.patch.object(db.time, "monotonic", lambda: clock[0]):
            db._last_heartbeat_log_trim[0] = clock[0]
            # Within the interval: opportunistic trim must not fire.
            clock[0] += 10.0
            db._maybe_trim_heartbeat_log(self._conn)
            self.assertEqual(len(self._log_rows("ree-cloud-2")), 1)
            # Past the interval: it must fire exactly now.
            clock[0] += db.HEARTBEAT_LOG_TRIM_INTERVAL_SECONDS
            db._maybe_trim_heartbeat_log(self._conn)
            self.assertEqual(len(self._log_rows("ree-cloud-2")), 0)

    def test_maybe_trim_runs_from_upsert_heartbeat_when_due(self):
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {}, {})
        self._conn.execute(
            "UPDATE heartbeat_log SET observed_at='2020-01-01T00:00:00Z'")
        db._last_heartbeat_log_trim[0] = 0.0
        # Any subsequent heartbeat (even a no-op repeat) is a trim chance.
        db.upsert_heartbeat(self._conn, "ree-cloud-2", "running",
                            "V3-EXQ-001", {}, {})
        rows = self._conn.execute(
            "SELECT * FROM heartbeat_log WHERE observed_at='2020-01-01T00:00:00Z'"
        ).fetchall()
        self.assertEqual(len(rows), 0,
                         "the stale row must be gone once trim is due")


class TestTableProvisioning(_Fixture):

    def test_bare_connect_creates_heartbeat_log_on_a_fresh_db(self):
        """A live DB "picks it up without a rebuild" -- the same contract
        _migrate_heartbeats/_migrate_commands/_migrate_experiments document
        for column additions, extended here to a whole new table. Unlike
        those functions (which gate on their parent table already
        existing), _migrate_heartbeat_log has no such dependency, so a bare
        connect() on a totally fresh file provisions the TABLE even though
        every other table (created only by init_db()'s executescript) is
        absent.

        Writing through upsert_heartbeat still requires init_db() to have
        run at least once, same as it always has for `heartbeats` itself
        (connect()'s migrations only ALTER an existing table; they never
        CREATE the ones init_db()'s executescript is responsible for) --
        this test is about heartbeat_log's own provisioning, not a new
        writability guarantee this change does not make."""
        fresh_path = os.path.join(self._tmp, "fresh.db")
        conn = db.connect(fresh_path)
        try:
            present = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='heartbeat_log'").fetchone()
            self.assertIsNotNone(present)
            absent = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='heartbeats'").fetchone()
            self.assertIsNone(
                absent,
                "heartbeats is only created by init_db()'s executescript; "
                "if this now exists, the two provisioning paths have "
                "silently diverged from the documented contract")
        finally:
            conn.close()

    def test_index_exists(self):
        idx = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' "
            "AND name='idx_heartbeat_log_machine_time'").fetchone()
        self.assertIsNotNone(idx)

    def test_init_db_is_idempotent(self):
        # Re-running init_db against an already-provisioned DB (the normal
        # coordinator-restart-on-deploy path) must not raise or duplicate
        # the table/index.
        db.init_db(self._dbpath)
        conn = db.connect(self._dbpath)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='heartbeat_log'").fetchall()
            self.assertEqual(len(rows), 1)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
