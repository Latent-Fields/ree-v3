"""Contracts for task_claims.claude_session_id -- the claim -> session bridge.

WHY THE COLUMN EXISTS. A claim's `session_id` is free text the opening agent
picks. It shares NO id space with the session ids Claude Code's SessionStart
hooks record: measured 2026-09-18, of 114 claim ids and 166 ledger ids, ZERO
matched. So a session that finds uncommitted work in a shared checkout could
name the owning CLAIM (by resource) and still have no way to reach the SESSION
holding it. That decision -- "is this safe to disturb, or is someone mid-edit?"
-- is exactly what the concurrency rules in CLAUDE.md turn on.

WHY A COORDINATOR COLUMN RATHER THAN A PER-MACHINE FILE. 60% of sessions run on
ree-cloud-4/5, not the Mac (1284 host-stamped ledger entries, 2026-09-18), so a
per-machine index would miss the majority -- including the headless igw and
metaworker sessions whose owner is hardest to find by any other means.

THE THREE PROPERTIES PINNED HERE, each of which failed silently if got wrong:

  1. ADDITIVE MIGRATION. CREATE TABLE IF NOT EXISTS is a no-op against a live
     DB, so without an explicit ALTER every existing deployment keeps the old
     shape and the writers' new field is dropped at the SQL layer -- no error,
     no warning, just a column that is always ''.

  2. ENTRY_JSON SHAPE IS UNCHANGED WHEN THE ID IS ABSENT. entry_json is
     rendered VERBATIM into TASK_CLAIMS.json and compared field-set-for-field-
     set by the PHASE-1 reconciler to decide `diverged`. Emitting
     claude_session_id="" unconditionally would change the shape of every
     pre-2026-09-18 claim at once and read as fleet-wide divergence.

  3. WRITE-ONCE. The column records WHO OPENED the claim. close/amend
     legitimately arrive from a DIFFERENT session (a /session-land sweep, an
     audit reaper, the materializer re-upserting from git with no session id at
     all); plain excluded.* would let any of those overwrite the opener with
     themselves or with ''. An empty value is still backfillable.

ASCII-only.
"""

import json
import pathlib
import sqlite3
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db  # noqa: E402

UUID_A = "1f2c5e79-9288-4a1f-b57c-2c3f678381c9"
UUID_B = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

# The exact key set/order an open claim's entry_json had before this column
# existed. Hard-coded rather than derived: deriving it from the code under test
# would make this contract agree with whatever that code does.
PRE_CHANGE_KEYS = ["session_id", "session_label", "claimed_at", "task",
                   "resources", "status"]

OLD_TABLE_SQL = """
CREATE TABLE task_claims (
    session_id                   TEXT NOT NULL,
    claimed_at                   TEXT NOT NULL,
    session_label                TEXT NOT NULL DEFAULT '',
    task                         TEXT NOT NULL DEFAULT '',
    status                       TEXT NOT NULL DEFAULT 'active',
    closed_at                    TEXT,
    completion_note              TEXT,
    completion_note_history_json TEXT,
    spawned_by                   TEXT,
    entry_json                   TEXT NOT NULL,
    last_rendered_json           TEXT,
    updated_at                   TEXT NOT NULL,
    PRIMARY KEY (session_id, claimed_at)
)
"""


def _fresh_db():
    path = pathlib.Path(tempfile.mkdtemp()) / "coordinator.sqlite"
    return db.connect(str(path)), path


def _legacy_db():
    """A DB carrying the PRE-migration table shape and one legacy row."""
    path = pathlib.Path(tempfile.mkdtemp()) / "coordinator.sqlite"
    raw = sqlite3.connect(str(path))
    raw.execute(OLD_TABLE_SQL)
    raw.execute(
        "INSERT INTO task_claims (session_id, claimed_at, session_label, task, "
        "status, entry_json, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("legacy-sess", "2026-09-01T00:00:00Z", "L", "T", "active",
         json.dumps({"session_id": "legacy-sess"}), "2026-09-01T00:00:00Z"))
    raw.commit()
    raw.close()
    return path


def _columns(conn):
    return {row[1] for row in conn.execute("PRAGMA table_info(task_claims)")}


def _entry(conn, session_id):
    row = conn.execute("SELECT entry_json FROM task_claims WHERE session_id=?",
                       (session_id,)).fetchone()
    return json.loads(row["entry_json"] if hasattr(row, "keys") else row[0])


def _stored(conn, session_id):
    row = conn.execute(
        "SELECT claude_session_id FROM task_claims WHERE session_id=?",
        (session_id,)).fetchone()
    return row["claude_session_id"] if hasattr(row, "keys") else row[0]


class TestMigration(unittest.TestCase):

    def test_fresh_db_has_the_column(self):
        conn, _ = _fresh_db()
        self.assertIn("claude_session_id", _columns(conn))

    def test_live_db_gains_the_column_on_connect(self):
        """The property CREATE TABLE IF NOT EXISTS cannot deliver."""
        path = _legacy_db()
        pre = sqlite3.connect(str(path))
        self.assertNotIn(
            "claude_session_id",
            {r[1] for r in pre.execute("PRAGMA table_info(task_claims)")},
            "fixture is not actually pre-migration; this test would be vacuous")
        pre.close()

        conn = db.connect(str(path))
        self.assertIn("claude_session_id", _columns(conn))

    def test_legacy_rows_default_to_empty_string_not_null(self):
        """A NULL would make every reader that does .strip()/!= '' a crash or a
        wrong answer; the column is declared NOT NULL DEFAULT ''."""
        conn = db.connect(str(_legacy_db()))
        self.assertEqual(_stored(conn, "legacy-sess"), "")

    def test_migration_is_idempotent(self):
        path = _legacy_db()
        db.connect(str(path)).close()
        conn = db.connect(str(path))          # second connect must not raise
        self.assertIn("claude_session_id", _columns(conn))


class TestOpenRecordsIt(unittest.TestCase):

    def test_open_with_id_stores_column_and_entry_json(self):
        conn, _ = _fresh_db()
        verdict, _ = db.try_open_task_claim(
            conn, session_id="s-with", session_label="L", task="T",
            resources=["ree-v3/foo.py"], claude_session_id=UUID_A)
        self.assertEqual(verdict, "ok")
        self.assertEqual(_stored(conn, "s-with"), UUID_A)
        self.assertEqual(_entry(conn, "s-with").get("claude_session_id"), UUID_A)

    def test_open_without_id_leaves_entry_json_shape_untouched(self):
        """Property 2: no fleet-wide `diverged` for pre-existing claims."""
        conn, _ = _fresh_db()
        db.try_open_task_claim(conn, session_id="s-without", session_label="L",
                               task="T", resources=["ree-v3/bar.py"])
        entry = _entry(conn, "s-without")
        self.assertNotIn("claude_session_id", entry)
        self.assertEqual(list(entry.keys()), PRE_CHANGE_KEYS)

    def test_empty_string_is_treated_as_absent(self):
        """'' must behave exactly like None -- a caller that reads a missing
        env var into '' must not change the entry shape."""
        conn, _ = _fresh_db()
        db.try_open_task_claim(conn, session_id="s-empty", session_label="L",
                               task="T", resources=["ree-v3/baz.py"],
                               claude_session_id="")
        self.assertNotIn("claude_session_id", _entry(conn, "s-empty"))
        self.assertEqual(_stored(conn, "s-empty"), "")


class TestWriteOnce(unittest.TestCase):

    def _opened(self):
        conn, _ = _fresh_db()
        _, payload = db.try_open_task_claim(
            conn, session_id="s1", session_label="L", task="T",
            resources=["ree-v3/foo.py"], claude_session_id=UUID_A)
        return conn, payload["claimed_at"]

    def test_close_from_another_session_cannot_clobber_the_opener(self):
        conn, claimed_at = self._opened()
        db.upsert_task_claim(conn, {
            "session_id": "s1", "claimed_at": claimed_at, "session_label": "L",
            "task": "T", "status": "done", "closed_at": "2026-09-18T13:00:00Z",
            "completion_note": "closed by a sweep",
            "claude_session_id": UUID_B,
        })
        self.assertEqual(_stored(conn, "s1"), UUID_A,
                         "a later writer overwrote the opening session")

    def test_upsert_with_no_id_cannot_blank_the_opener(self):
        """The materializer re-upserting from git sends no session id at all."""
        conn, claimed_at = self._opened()
        db.upsert_task_claim(conn, {
            "session_id": "s1", "claimed_at": claimed_at, "session_label": "L",
            "task": "T", "status": "done", "closed_at": "2026-09-18T13:00:00Z",
        })
        self.assertEqual(_stored(conn, "s1"), UUID_A)

    def test_write_once_does_not_freeze_the_other_columns(self):
        """Write-once must be scoped to THIS column: a close still lands."""
        conn, claimed_at = self._opened()
        db.upsert_task_claim(conn, {
            "session_id": "s1", "claimed_at": claimed_at, "session_label": "L",
            "task": "T", "status": "done", "closed_at": "2026-09-18T13:00:00Z",
            "completion_note": "landed abc123",
        })
        row = conn.execute(
            "SELECT status, closed_at, completion_note FROM task_claims "
            "WHERE session_id='s1'").fetchone()
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["closed_at"], "2026-09-18T13:00:00Z")
        self.assertEqual(row["completion_note"], "landed abc123")

    def test_empty_value_is_still_backfillable(self):
        """Write-once protects a REAL id, not the absence of one."""
        conn, _ = _fresh_db()
        _, payload = db.try_open_task_claim(
            conn, session_id="s2", session_label="L", task="T",
            resources=["ree-v3/foo.py"])
        self.assertEqual(_stored(conn, "s2"), "")
        db.upsert_task_claim(conn, {
            "session_id": "s2", "claimed_at": payload["claimed_at"],
            "session_label": "L", "task": "T", "status": "active",
            "claude_session_id": UUID_B,
        })
        self.assertEqual(_stored(conn, "s2"), UUID_B)


class TestSharedSessionIdGuard(unittest.TestCase):
    """The id_collision verdict: two DISTINCT Claude sessions, one session_id.

    MEASURED 2026-09-18 against the live hub, before this guard existed: two
    `open` calls under one session_id carrying DIFFERENT --task and
    --resources produced ONE row with caller A's fields intact and caller B's
    resource absent from the DB entirely. B was told "claim already active --
    nothing to do" and proceeded, believing the row was its own.

    Two mechanisms conspire and BOTH are pinned below: the early-return
    idempotent branch fires before the rival scan, and the rival scan itself
    excludes `c.session_id<>?`. So B's resources are arbitrated against
    NOBODY -- not merely against A. test_colliding_resources_are_not_recorded
    is the one that would have caught the incident.
    """

    def _held_by_a(self, conn):
        verdict, _ = db.try_open_task_claim(
            conn, session_id="shared-id", session_label="A",
            task="A's task", resources=["ree-v3/a.py"],
            claude_session_id=UUID_A)
        self.assertEqual(verdict, "ok")

    def test_same_session_rerunning_is_still_idempotent(self):
        """THE REGRESSION GUARD. The git path retries `open` routinely under
        contention; turning that into a refusal would break the documented
        per-session_id idempotency (plan doc D8)."""
        conn, _ = _fresh_db()
        self._held_by_a(conn)
        verdict, payload = db.try_open_task_claim(
            conn, session_id="shared-id", session_label="A",
            task="A's task", resources=["ree-v3/a.py"],
            claude_session_id=UUID_A)
        self.assertEqual(verdict, "idempotent")
        self.assertTrue(payload["claimed_at"])

    def test_different_session_same_id_is_refused(self):
        conn, _ = _fresh_db()
        self._held_by_a(conn)
        verdict, payload = db.try_open_task_claim(
            conn, session_id="shared-id", session_label="B",
            task="B's DIFFERENT task", resources=["ree-v3/b.py"],
            claude_session_id=UUID_B)
        self.assertEqual(verdict, "id_collision")
        self.assertEqual(payload["holder_claude_session_id"], UUID_A)
        self.assertEqual(payload["holder_session_label"], "A")
        self.assertEqual(payload["holder_task"], "A's task")

    def test_colliding_resources_are_not_recorded(self):
        """The harm itself: B's declared scope must not silently vanish into
        a success. Pinned as a property of the REFUSAL -- B is told to re-run
        under its own id, which is what gets its resources arbitrated."""
        conn, _ = _fresh_db()
        self._held_by_a(conn)
        db.try_open_task_claim(
            conn, session_id="shared-id", session_label="B",
            task="B's task", resources=["ree-v3/b.py"],
            claude_session_id=UUID_B)
        rows = conn.execute(
            "SELECT resource FROM task_claim_resources").fetchall()
        self.assertEqual(sorted(r["resource"] for r in rows), ["ree-v3/a.py"])
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM task_claims").fetchone()["c"],
            1)

    def test_holder_fields_are_never_overwritten(self):
        """Negative control for the mechanism the chip originally alleged: an
        ON CONFLICT field overwrite. There is none on this path -- the branch
        ROLLBACKs -- and this pins that it stays that way."""
        conn, _ = _fresh_db()
        self._held_by_a(conn)
        db.try_open_task_claim(
            conn, session_id="shared-id", session_label="B OVERWRITE?",
            task="B's task", resources=["ree-v3/b.py"],
            claude_session_id=UUID_B)
        row = conn.execute(
            "SELECT session_label, task FROM task_claims "
            "WHERE session_id='shared-id'").fetchone()
        self.assertEqual(row["session_label"], "A")
        self.assertEqual(row["task"], "A's task")

    def test_unknown_caller_id_falls_back_to_idempotent(self):
        """FAIL-SAFE. A degraded client or non-Claude caller sends nothing;
        refusing on missing data would invent a stop out of ignorance."""
        conn, _ = _fresh_db()
        self._held_by_a(conn)
        for mine in (None, ""):
            verdict, _ = db.try_open_task_claim(
                conn, session_id="shared-id", session_label="B",
                task="T", resources=["ree-v3/b.py"], claude_session_id=mine)
            self.assertEqual(verdict, "idempotent")

    def test_unknown_holder_id_falls_back_to_idempotent(self):
        """The pre-migration shape: the row predates the column, so '' is
        'unrecorded', not 'a different session'."""
        conn, _ = _fresh_db()
        db.try_open_task_claim(
            conn, session_id="legacy-id", session_label="A", task="T",
            resources=["ree-v3/a.py"])
        verdict, _ = db.try_open_task_claim(
            conn, session_id="legacy-id", session_label="B", task="T2",
            resources=["ree-v3/b.py"], claude_session_id=UUID_B)
        self.assertEqual(verdict, "idempotent")

    def test_a_closed_claim_does_not_collide(self):
        """Only an ACTIVE row holds the id. Worktree slugs are reused across
        sessions for days (measured: one slug, 12 claims, distinct sessions);
        that must stay free once the previous claim is closed."""
        conn, _ = _fresh_db()
        self._held_by_a(conn)
        conn.execute("UPDATE task_claims SET status='done' "
                     "WHERE session_id='shared-id'")
        # Explicit claimed_at: the reopen would otherwise collide with the
        # closed row on the (session_id, claimed_at) PRIMARY KEY whenever both
        # opens land in the same second, and surface as verdict 'error'.
        # Pre-existing behaviour of the INSERT, not of this guard -- pinning
        # it here would test the clock.
        verdict, _ = db.try_open_task_claim(
            conn, session_id="shared-id", session_label="B", task="T",
            resources=["ree-v3/b.py"], claude_session_id=UUID_B,
            claimed_at="2030-01-01T00:00:00Z")
        self.assertEqual(verdict, "ok")

    def test_a_real_rival_still_outranks_the_id_check(self):
        """Ordering: a DIFFERENT session_id contending for the same file is
        still 'owned_by_other'. The two refusals must not be confused --
        their remedies differ (stop, vs re-run under your own id)."""
        conn, _ = _fresh_db()
        self._held_by_a(conn)
        verdict, payload = db.try_open_task_claim(
            conn, session_id="other-id", session_label="C", task="T",
            resources=["ree-v3/a.py"], claude_session_id=UUID_B)
        self.assertEqual(verdict, "owned_by_other")
        self.assertTrue(payload["rivals"])


class TestLiveFleetPatternsStillWork(unittest.TestCase):
    """Negative controls drawn from claims LIVE on the fleet 2026-09-18.

    Raised by the orchestrator session while this was being built: a guard
    that fires on correct current usage gets disabled, which is worse than no
    guard. These are the real shapes it must stay silent on.
    """

    def test_suffixed_sibling_ids_from_one_session_all_open(self):
        """The SANCTIONED pattern -- and the one this guard's own CLI message
        recommends as the remedy. One session legitimately holds several
        claims under DISTINCT ids (live: eloquent-jepsen-5f6242 alongside
        -coord and -tests; metaworker-science-...-p1p3 alongside
        ...-p1p3-exq-1056). Same csid, different session_id, different
        resources: every one must be 'ok'. The guard keys on an EXACT
        session_id match, so it is never even reached here."""
        conn, _ = _fresh_db()
        for sid, res in (("sess", "ree-v3/one.py"),
                         ("sess-coord", "ree-v3/two.py"),
                         ("sess-tests", "ree-v3/three.py")):
            verdict, _ = db.try_open_task_claim(
                conn, session_id=sid, session_label="L", task="T",
                resources=[res], claude_session_id=UUID_A)
            self.assertEqual(verdict, "ok", "%s must open" % sid)

    def test_suffixed_sibling_ids_from_DIFFERENT_sessions_also_open(self):
        """Two sessions cooperating on one campaign under distinct ids is
        also correct usage -- the ids differ, so there is no collision to
        find. Only a genuine resource overlap may refuse them, and that is
        the pre-existing owned_by_other path, untouched here."""
        conn, _ = _fresh_db()
        db.try_open_task_claim(
            conn, session_id="camp-hub", session_label="A", task="T",
            resources=["ree-v3/a.py"], claude_session_id=UUID_A)
        verdict, _ = db.try_open_task_claim(
            conn, session_id="camp-transport", session_label="B", task="T",
            resources=["ree-v3/b.py"], claude_session_id=UUID_B)
        self.assertEqual(verdict, "ok")


if __name__ == "__main__":
    unittest.main()
