"""Contracts for the PHASE-2b ingest AUTHORITY fix (2026-08-28).

The defect these pin: the materializer's ingest-before-render step called
db.upsert_task_claim / db.upsert_chip, which under PHASE-1 shadow semantics
ALWAYS adopted the git content -- so any coordinator-acked close/resolve on
a row already rendered into git as active/open was clobbered back within one
materializer (or shadow-sync) tick. Root-caused live 2026-08-28: a
suppressed-git-write close was acked at 15:06:48Z, its DB row went done
seconds later, and the next ingest of the still-active git row reverted it.

The fix is a 3-way merge with a recorded base (last_rendered_json, written
by the materializer once a render provably reached git):

  T1  the exact bug -- ingest a git active row, db.close_task_claim it,
      re-ingest the same git content -> the closure SURVIVES.
  T2  git-path close adoption -- DB unchanged since render (entry == base),
      git carries the close -> the close is ADOPTED (the self-healing
      fallback direction is kept).
  T3  pre-migration conflict guard -- base NULL, DB done, git active ->
      stays done (terminal is never downgraded even without a base).
  T4  render-writeback round trip -- render, write back the base, mutate
      the DB, re-ingest that render's own text -> the mutation survives.
  T5  chip mirror of T1 -- a DB-side resolve survives re-ingest of the
      still-open git row once the base is recorded.
  T6  chip claim, both directions -- a suppressed (DB-side) chip claim is
      preserved when git==base; a git-path claim is adopted when DB==base.

Time-independent (all stamps injected); pure-DB fixtures, no git repos --
render/writeback are exercised through the writer's own functions against
an in-memory DB, which is exactly the seam the live tick uses.

ASCII-only.
"""

import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db  # noqa: E402
import task_claim_chip_git_writer as writer  # noqa: E402

NOW = "2026-08-28T12:00:00Z"
T_OPEN = "2026-08-28T10:00:00Z"
T_CLOSE = "2026-08-28T11:00:00Z"


def _claim(session_id, claimed_at=T_OPEN, status="active", **extra):
    entry = {
        "session_id": session_id,
        "claimed_at": claimed_at,
        "session_label": "label-%s" % session_id,
        "task": "task for %s" % session_id,
        "resources": ["some/file.txt"],
        "status": status,
    }
    entry.update(extra)
    return entry


def _chip(chip_ref, status="open", **extra):
    entry = {
        "chip_ref": chip_ref,
        "task_id": "task_%s" % chip_ref,
        "session_id": "spawner",
        "session_label": "spawner label",
        "title": "Title %s" % chip_ref,
        "tldr": "tldr",
        "prompt": "[chip_ref: %s] do the thing" % chip_ref,
        "cwd": "/Users/dgolden/REE_Working",
        "spawned_at": T_OPEN,
        "status": status,
    }
    entry.update(extra)
    return entry


def _claims_doc(claims):
    return {"claims": claims, "schema_version": "task_claims/v1",
            "stale_after_hours": 6}


def _chips_doc(chips):
    return {"schema_version": "task_chips/v1", "chips": chips}


class IngestAuthorityBase(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")

    def tearDown(self):
        self.conn.close()

    # -- helpers ---------------------------------------------------------
    def _claim_row(self, session_id, claimed_at=T_OPEN):
        return self.conn.execute(
            "SELECT * FROM task_claims WHERE session_id=? AND claimed_at=?",
            (session_id, claimed_at)).fetchone()

    def _chip_row(self, chip_ref):
        return self.conn.execute(
            "SELECT * FROM chip_ledger WHERE chip_ref=?",
            (chip_ref,)).fetchone()

    def _render_and_writeback(self, claims_doc=None, chips_doc=None):
        """Simulate the materializer's render + confirmed-in-git base
        writeback for whatever is currently in the DB."""
        claim_snaps, chip_snaps = [], []
        if claims_doc is not None:
            _text, _stats, claim_snaps = writer.render_task_claims(
                self.conn, source_doc=claims_doc, now_iso=NOW)
        if chips_doc is not None:
            _text, _stats, chip_snaps = writer.render_chips(
                self.conn, source_doc=chips_doc)
        writer._writeback_rendered_base(self.conn, claim_snaps, chip_snaps)


class TestClaimIngestAuthority(IngestAuthorityBase):
    def test_t1_suppressed_close_survives_reingest(self):
        """The exact live bug: open ingested, rendered into git, closed
        DB-side, then the still-active git row is re-ingested."""
        entry = _claim("s1")
        db.reconcile_task_claims(self.conn, [entry], now=NOW)
        # The row has been rendered into git (base recorded).
        self._render_and_writeback(claims_doc=_claims_doc([entry]))
        verdict, _payload = db.close_task_claim(
            self.conn, "s1", T_CLOSE, "landed as abc123", claimed_at=T_OPEN)
        self.assertEqual(verdict, "ok")
        # Materializer tick ingests origin, which still carries the row
        # as ACTIVE (the close was git-write-suppressed).
        stats = db.reconcile_task_claims(self.conn, [entry], now=NOW)
        row = self._claim_row("s1")
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["closed_at"], T_CLOSE)
        self.assertEqual(row["completion_note"], "landed as abc123")
        self.assertEqual(stats["n_updated"], 0)

    def test_t2_git_path_close_is_adopted(self):
        """Self-healing direction preserved: only git moved -> adopt."""
        entry = _claim("s2")
        db.reconcile_task_claims(self.conn, [entry], now=NOW)
        self._render_and_writeback(claims_doc=_claims_doc([entry]))
        # DB untouched since the render; git now carries the close
        # (a client fell back to the git path and its commit landed).
        closed = dict(entry)
        closed["status"] = "done"
        closed["closed_at"] = T_CLOSE
        closed["completion_note"] = "git-forced close"
        db.reconcile_task_claims(self.conn, [closed], now=NOW)
        row = self._claim_row("s2")
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["completion_note"], "git-forced close")

    def test_t3_pre_migration_terminal_guard(self):
        """base NULL (pre-migration row): a done DB row is never
        downgraded to an active git one."""
        entry = _claim("s3")
        db.reconcile_task_claims(self.conn, [entry], now=NOW)
        verdict, _payload = db.close_task_claim(
            self.conn, "s3", T_CLOSE, "closed pre-migration",
            claimed_at=T_OPEN)
        self.assertEqual(verdict, "ok")
        self.assertIsNone(self._claim_row("s3")["last_rendered_json"])
        created, changed = db.upsert_task_claim(self.conn, entry, now=NOW)
        self.assertEqual((created, changed), (False, False))
        row = self._claim_row("s3")
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["completion_note"], "closed pre-migration")

    def test_t3b_pre_migration_nonterminal_still_adopts_git(self):
        """The guard is NARROW: without a base, a non-terminal difference
        still adopts git (renew/amend drift heals from git as before)."""
        entry = _claim("s3b")
        db.reconcile_task_claims(self.conn, [entry], now=NOW)
        amended = dict(entry)
        amended["task"] = "amended task text from git"
        db.reconcile_task_claims(self.conn, [amended], now=NOW)
        row = self._claim_row("s3b")
        self.assertEqual(json.loads(row["entry_json"])["task"],
                         "amended task text from git")

    def test_t4_render_writeback_round_trip(self):
        """Full loop: render -> writeback -> DB mutation -> re-ingest the
        render's own text -> the mutation is preserved."""
        entry = _claim("s4")
        doc = _claims_doc([entry])
        db.reconcile_task_claims(self.conn, [entry], now=NOW)
        text, _stats, snaps = writer.render_task_claims(
            self.conn, source_doc=doc, now_iso=NOW)
        writer._writeback_rendered_base(self.conn, snaps, [])
        verdict, _payload = db.close_task_claim(
            self.conn, "s4", T_CLOSE, "closed after render",
            claimed_at=T_OPEN)
        self.assertEqual(verdict, "ok")
        rendered_doc = json.loads(text)
        db.reconcile_task_claims(self.conn, rendered_doc["claims"], now=NOW)
        row = self._claim_row("s4")
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["completion_note"], "closed after render")


class TestChipIngestAuthority(IngestAuthorityBase):
    def test_t5_suppressed_resolve_survives_reingest(self):
        entry = _chip("chip-t5")
        db.reconcile_chips(self.conn, [entry], now=NOW)
        self._render_and_writeback(chips_doc=_chips_doc([entry]))
        verdict, payload = db.resolve_chip(
            self.conn, "done", chip_ref="chip-t5", note="work landed",
            resolved_by_session_id="worker-1")
        self.assertEqual(verdict, "ok")
        self.assertTrue(payload.get("changed", True))
        db.reconcile_chips(self.conn, [entry], now=NOW)
        row = self._chip_row("chip-t5")
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["resolution_note"], "work landed")

    def test_t6a_suppressed_claim_preserved_when_git_matches_base(self):
        entry = _chip("chip-t6a")
        db.reconcile_chips(self.conn, [entry], now=NOW)
        self._render_and_writeback(chips_doc=_chips_doc([entry]))
        verdict, _payload = db.try_claim_chip(
            self.conn, chip_ref="chip-t6a", claimed_by="sess-x",
            note="started work", now=NOW)
        self.assertEqual(verdict, "ok")
        db.reconcile_chips(self.conn, [entry], now=NOW)
        row = self._chip_row("chip-t6a")
        self.assertEqual(row["claimed_by"], "sess-x")

    def test_t6b_git_path_claim_adopted_when_db_matches_base(self):
        entry = _chip("chip-t6b")
        db.reconcile_chips(self.conn, [entry], now=NOW)
        self._render_and_writeback(chips_doc=_chips_doc([entry]))
        claimed = dict(entry)
        claimed["claimed_by"] = "sess-git"
        claimed["claimed_at"] = T_CLOSE
        claimed["claim_note"] = "claimed via git path"
        db.reconcile_chips(self.conn, [claimed], now=NOW)
        row = self._chip_row("chip-t6b")
        self.assertEqual(row["claimed_by"], "sess-git")

    def test_t6c_pre_migration_terminal_guard_for_chips(self):
        """base NULL: a done/withdrawn DB chip is never reopened by an
        open git row."""
        entry = _chip("chip-t6c")
        db.reconcile_chips(self.conn, [entry], now=NOW)
        verdict, _payload = db.resolve_chip(
            self.conn, "withdrawn", chip_ref="chip-t6c", note="superseded")
        self.assertEqual(verdict, "ok")
        self.assertIsNone(self._chip_row("chip-t6c")["last_rendered_json"])
        created, changed = db.upsert_chip(self.conn, entry, now=NOW)
        self.assertEqual((created, changed), (False, False))
        row = self._chip_row("chip-t6c")
        self.assertEqual(row["status"], "withdrawn")


class TestMigrationAndWriteback(IngestAuthorityBase):
    def test_migration_adds_column_to_existing_tables(self):
        """A pre-migration DB (tables without the column) gains it on the
        next connect() -- the deploy-auto-migrates contract."""
        import sqlite3
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            raw = sqlite3.connect(path)
            raw.execute(
                "CREATE TABLE task_claims ("
                " session_id TEXT NOT NULL, claimed_at TEXT NOT NULL,"
                " session_label TEXT NOT NULL DEFAULT '',"
                " task TEXT NOT NULL DEFAULT '',"
                " status TEXT NOT NULL DEFAULT 'active',"
                " closed_at TEXT, completion_note TEXT,"
                " completion_note_history_json TEXT, spawned_by TEXT,"
                " entry_json TEXT NOT NULL, updated_at TEXT NOT NULL,"
                " PRIMARY KEY (session_id, claimed_at))")
            raw.execute(
                "CREATE TABLE chip_ledger ("
                " chip_ref TEXT PRIMARY KEY, task_id TEXT,"
                " status TEXT NOT NULL DEFAULT 'open', claimed_by TEXT,"
                " spawned_at TEXT NOT NULL DEFAULT '',"
                " entry_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
            raw.commit()
            raw.close()
            conn = db.connect(path)
            for table in ("task_claims", "chip_ledger"):
                cols = {r[1] for r in conn.execute(
                    "PRAGMA table_info(%s)" % table).fetchall()}
                self.assertIn("last_rendered_json", cols,
                              "migration missed %s" % table)
            conn.close()
        finally:
            os.unlink(path)

    def test_writeback_only_touches_rendered_rows(self):
        kept = _claim("s-kept")
        db.reconcile_task_claims(self.conn, [kept], now=NOW)
        # An aged-out done row is NOT rendered and must keep base NULL.
        old = _claim("s-old", claimed_at="2026-08-20T00:00:00Z",
                     status="done", closed_at="2026-08-20T01:00:00Z",
                     completion_note="ancient")
        db.reconcile_task_claims(self.conn, [old], now=NOW)
        text, stats, snaps = writer.render_task_claims(
            self.conn, source_doc=_claims_doc([kept, old]), now_iso=NOW)
        self.assertEqual(stats["n_retention_dropped"], 1)
        writer._writeback_rendered_base(self.conn, snaps, [])
        self.assertIsNotNone(self._claim_row("s-kept")["last_rendered_json"])
        self.assertIsNone(self._claim_row(
            "s-old", "2026-08-20T00:00:00Z")["last_rendered_json"])
        # The recorded base is the verbatim entry_json blob.
        row = self._claim_row("s-kept")
        self.assertEqual(row["last_rendered_json"], row["entry_json"])


if __name__ == "__main__":
    unittest.main()
