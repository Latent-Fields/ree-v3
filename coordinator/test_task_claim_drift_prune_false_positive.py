"""Regression contracts: a PRUNED `done` claim is not drift.

Run: /opt/local/bin/python3 -m pytest coordinator/test_task_claim_drift_prune_false_positive.py

THE INCIDENT THIS PINS (2026-08-27/28, found while judging the PHASE-1 soak
for a cutover go/no-go, not by a test).

`reconcile_task_claims` treated ANY DB key absent from git as an orphan, and
`log_task_claim_chip_drift` raised `diverged` on any orphan. Its docstring
justified that with "TASK_CLAIMS.json entries are never deleted (root
CLAUDE.md)". That premise is false: `scripts/prune_task_claims_done.py`
deletes `done` entries older than 24h and runs at EVERY `/session-land` close.

Measured on the live hub: prune `b6907cce` removed 127 entries at
2026-08-27T19:02:07Z; the next shadow-sync tick, 78 seconds later, went
`diverged=1` and stayed there. 64 of the next 200 ticks were diverged, and all
50 reported orphans were `done` in git immediately before that prune and
absent after -- zero unexplained.

Why that was worse than a noisy metric, and the reason these tests exist:
  1. the exit criterion ("N days of diverged_ticks at 0") became UNMEETABLE --
     one routine prune arms it permanently; and
  2. a REAL divergence would then hide behind an already-raised flag.

So the split keys on the pruner's own predicate. `done` + absent -> RETIRED
(expected, counted, never `diverged`). `active` + absent -> ORPHAN (real
drift, and the direction that actually loses work).

Time-independent; real sqlite in a tempdir. ASCII-only.
"""

import json
import os
import pathlib
import shutil
import sqlite3
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db  # noqa: E402

T0 = "2026-08-26T10:00:00Z"
T1 = "2026-08-27T10:00:00Z"


def claim(sid, at, status="active", **kw):
    c = {"session_id": sid, "claimed_at": at, "session_label": "l",
         "task": "t", "resources": ["a.py"], "status": status}
    if status == "done":
        c.setdefault("closed_at", at)
        c.setdefault("completion_note", "landed")
    c.update(kw)
    return c


def chip(ref, **kw):
    c = {"chip_ref": ref, "task_id": None, "origin": "spawn_task",
         "kind": "work", "urgency": False, "session_id": "s",
         "session_label": "", "title": "t", "tldr": "d",
         "prompt": "p [chip_ref: %s]" % ref, "cwd": "/x",
         "origin_host": "DLAPTOP", "spawned_at": T0, "status": "open"}
    c.update(kw)
    return c


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="drift_prune_")
        self.dbpath = os.path.join(self.tmp, "coord.db")
        db.init_db(self.dbpath)
        self.conn = db.connect(self.dbpath)

    def tearDown(self):
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def tick(self, claims, chips=()):
        cs = db.reconcile_task_claims(self.conn, list(claims), now=T1)
        hs = db.reconcile_chips(self.conn, list(chips), now=T1)
        diverged = db.log_task_claim_chip_drift(self.conn, "abc123", cs, hs,
                                                now=T1)
        return cs, hs, diverged

    def last_row(self):
        return self.conn.execute(
            "SELECT * FROM task_claim_chip_drift_log ORDER BY id DESC LIMIT 1"
        ).fetchone()


class TestPrunedDoneIsNotDrift(Base):

    def test_the_incident_replayed_end_to_end(self):
        """A tick, then a prune, then the next tick. Before the fix the second
        tick went diverged=1 and never came back."""
        git = [claim("s-active", T0), claim("s-done", T0, status="done")]
        _, _, d1 = self.tick(git)
        self.assertEqual(d1, 0, "baseline tick must be clean")

        # prune_task_claims_done.py drops the done entry from git.
        pruned = [claim("s-active", T0)]
        cs, _, d2 = self.tick(pruned)

        self.assertEqual(d2, 0,
                         "a PRUNED done claim must not raise diverged -- this "
                         "is the exact 2026-08-27T19:03Z regression")
        self.assertEqual(cs["orphans"], [])
        self.assertEqual(cs["retired"], ["s-done|%s" % T0])

    def test_the_retired_entry_is_reported_not_silently_dropped(self):
        """Not-drift must not mean invisible: the mirror keeps growing past
        git, and that has to stay auditable."""
        self.tick([claim("s-done", T0, status="done")])
        self.tick([])
        row = self.last_row()
        self.assertEqual(row["n_claims_retired"], 1)
        self.assertEqual(row["n_claims_orphan"], 0)
        self.assertEqual(json.loads(row["detail"])["claim_retired"],
                         ["s-done|%s" % T0])

    def test_the_row_survives_in_the_mirror(self):
        """Retiring is a CLASSIFICATION, not a deletion. The reconciler has no
        delete path and must not grow one."""
        self.tick([claim("s-done", T0, status="done")])
        self.tick([])
        n = self.conn.execute(
            "SELECT COUNT(*) c FROM task_claims WHERE session_id='s-done'"
        ).fetchone()["c"]
        self.assertEqual(n, 1)

    def test_many_pruned_entries_still_clean(self):
        """The live prune removed 127 at once."""
        git = [claim("s%d" % i, T0, status="done") for i in range(127)]
        git.append(claim("live", T0))
        self.tick(git)
        cs, _, d = self.tick([claim("live", T0)])
        self.assertEqual(d, 0)
        self.assertEqual(len(cs["retired"]), 127)
        self.assertEqual(cs["orphans"], [])


class TestRealDriftStillDetected(Base):
    """The negative controls. A fix that made everything clean would be worse
    than the bug -- these pin what must STILL raise diverged."""

    def test_an_ACTIVE_claim_vanishing_from_git_is_drift(self):
        """The loss direction that matters: a live claim disappearing is
        exactly the read-modify-write loss this subsystem exists to catch."""
        self.tick([claim("s-active", T0), claim("other", T0)])
        cs, _, d = self.tick([claim("other", T0)])
        self.assertEqual(d, 1, "an ACTIVE claim absent from git IS drift")
        self.assertEqual(cs["orphans"], ["s-active|%s" % T0])
        self.assertEqual(cs["retired"], [])

    def test_a_missing_CHIP_is_still_drift(self):
        """Chips are never deleted -- archiving strips FIELDS and keeps the
        row (D5), and merge_origin_into_local has no deletion path at all. So
        the claims-side softening must NOT be mirrored here."""
        self.tick([], [chip("c1"), chip("c2")])
        _, hs, d = self.tick([], [chip("c2")])
        self.assertEqual(d, 1, "a missing chip has no benign explanation")
        self.assertEqual(hs["orphans"], ["c1"])
        self.assertEqual(hs["retired"], [])

    def test_a_resolved_chip_vanishing_is_STILL_drift(self):
        """The tempting symmetry, explicitly refused: `done`/`withdrawn` is
        not a licence to disappear for a chip the way it is for a claim."""
        self.tick([], [chip("c1", status="done")])
        _, hs, d = self.tick([], [])
        self.assertEqual(d, 1)
        self.assertEqual(hs["orphans"], ["c1"])

    def test_mixed_prune_and_real_loss_is_drift(self):
        """The dangerous shape: a genuine loss arriving in the same tick as an
        ordinary prune must not be masked by it."""
        self.tick([claim("gone-live", T0),
                   claim("pruned", T0, status="done"),
                   claim("kept", T0)])
        cs, _, d = self.tick([claim("kept", T0)])
        self.assertEqual(d, 1, "the real loss must survive the prune noise")
        self.assertEqual(cs["orphans"], ["gone-live|%s" % T0])
        self.assertEqual(cs["retired"], ["pruned|%s" % T0])


class TestWindowedSummary(Base):
    """The soak criterion has to remain CHECKABLE after the false positive.

    The 64 historically-diverged rows are not wrong and must not be deleted,
    so cumulative `diverged_ticks` stays non-zero forever. Without a window,
    "N days at 0" could never read as satisfied again."""

    def _row(self, at, diverged):
        self.conn.execute(
            "INSERT INTO task_claim_chip_drift_log (checked_at, source_ref, "
            "n_claims_git, n_claims_db, n_claims_new, n_claims_updated, "
            "n_claims_orphan, n_chips_git, n_chips_db, n_chips_new, "
            "n_chips_updated, n_chips_orphan, diverged) "
            "VALUES (?,'x',0,0,0,0,0,0,0,0,0,0,?)", (at, 1 if diverged else 0))

    def test_a_window_excludes_the_historical_false_positives(self):
        self._row("2026-08-27T19:03:00Z", True)     # the false-positive era
        self._row("2026-08-27T19:13:00Z", True)
        self._row("2026-08-28T06:00:00Z", False)    # after the fix
        self._row("2026-08-28T06:10:00Z", False)
        s = db.task_claim_chip_drift_summary(self.conn,
                                             since="2026-08-28T00:00:00Z")
        self.assertEqual(s["diverged_ticks"], 2, "cumulative count is retained")
        self.assertEqual(s["window"]["diverged_ticks"], 0)
        self.assertEqual(s["window"]["total_ticks"], 2)
        self.assertTrue(s["window"]["clean"])

    def test_the_window_still_sees_real_drift_inside_it(self):
        self._row("2026-08-28T06:00:00Z", False)
        self._row("2026-08-28T06:10:00Z", True)
        s = db.task_claim_chip_drift_summary(self.conn,
                                             since="2026-08-28T00:00:00Z")
        self.assertEqual(s["window"]["diverged_ticks"], 1)
        self.assertFalse(s["window"]["clean"])

    def test_the_window_reports_its_own_total_so_a_stalled_timer_is_visible(self):
        """Zero diverged out of two ticks is not evidence of anything. The
        window's total is what stops a dead timer reading as a clean soak."""
        self._row("2026-08-28T06:00:00Z", False)
        s = db.task_claim_chip_drift_summary(self.conn,
                                             since="2026-08-28T00:00:00Z")
        self.assertTrue(s["window"]["clean"])
        self.assertEqual(s["window"]["total_ticks"], 1,
                         "a caller must be able to see the sample was tiny")

    def test_no_since_keeps_the_original_shape_exactly(self):
        """Every existing reader must keep working unchanged."""
        self._row("2026-08-28T06:00:00Z", True)
        s = db.task_claim_chip_drift_summary(self.conn)
        self.assertNotIn("window", s)
        self.assertEqual(s["total_ticks"], 1)
        self.assertEqual(s["diverged_ticks"], 1)


class TestMigration(Base):
    def test_the_new_columns_are_additive_and_old_rows_survive(self):
        """PRAGMA table_info guard, never a rebuild -- so the soak history
        accumulated before this fix is preserved rather than reset."""
        self.conn.execute(
            "INSERT INTO task_claim_chip_drift_log (checked_at, source_ref, "
            "n_claims_git, n_claims_db, n_claims_new, n_claims_updated, "
            "n_claims_orphan, n_chips_git, n_chips_db, n_chips_new, "
            "n_chips_updated, n_chips_orphan, diverged) "
            "VALUES ('2026-01-01T00:00:00Z','old',0,0,0,0,0,0,0,0,0,0,0)")
        db._migrate_task_claim_chip_tables(self.conn)
        row = self.conn.execute(
            "SELECT * FROM task_claim_chip_drift_log WHERE source_ref='old'"
        ).fetchone()
        self.assertIsNone(row["n_claims_retired"], "pre-fix rows carry NULL")
        self.assertEqual(row["diverged"], 0)

    def test_migration_is_idempotent(self):
        for _ in range(3):
            db._migrate_task_claim_chip_tables(self.conn)
        cols = {r[1] for r in self.conn.execute(
            "PRAGMA table_info(task_claim_chip_drift_log)").fetchall()}
        self.assertIn("n_claims_retired", cols)
        self.assertIn("n_chips_retired", cols)


if __name__ == "__main__":
    unittest.main()
