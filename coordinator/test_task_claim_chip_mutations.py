"""Contracts for the PHASE-2 mutating verbs in db.py (plan doc section 5.2.4).

These are the write path that makes the migration a CORRECTNESS change rather
than a contention reduction: every verb takes BEGIN IMMEDIATE before its guard
SELECT, so the read-then-write window a git-based mutex structurally cannot
close is gone. The headline test here is
test_two_concurrent_opens_on_one_resource_produce_exactly_one_owner, which
replays the 2026-07-28 three-session collision shape against real threads.

Time-independent: every verb takes an injected `now`, and no test reads the
wall clock to decide an assertion. Real sqlite files in a tempdir, no mocks --
the whole point is the transaction semantics, which a mock would not have.

ASCII-only.
"""

import json
import os
import pathlib
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db  # noqa: E402

T0 = "2026-08-27T10:00:00Z"
T1 = "2026-08-27T11:00:00Z"
# 7h after T0 -- past the 6h default stale threshold.
T_STALE = "2026-08-27T17:30:00Z"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tc_chip_mut_")
        self.dbpath = os.path.join(self.tmp, "coord.db")
        db.init_db(self.dbpath)
        self.conn = db.connect(self.dbpath)

    def tearDown(self):
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def open_claim(self, sid, resources, at=T0, now=T0, **kw):
        return db.try_open_task_claim(
            self.conn, session_id=sid, session_label="lbl", task="t",
            resources=resources, claimed_at=at, now=now, **kw)

    def entry(self, sid, at):
        row = self.conn.execute(
            "SELECT entry_json FROM task_claims WHERE session_id=? AND claimed_at=?",
            (sid, at)).fetchone()
        return None if row is None else json.loads(row["entry_json"])


# --------------------------------------------------------------------------
# task_claim: open / arbitration
# --------------------------------------------------------------------------

class TestOpen(Base):

    def test_open_writes_claim_and_resource_rows(self):
        verdict, payload = self.open_claim("s1", ["a.py", "b.py"])
        self.assertEqual(verdict, "ok")
        self.assertEqual(payload["claimed_at"], T0)
        rows = self.conn.execute(
            "SELECT resource FROM task_claim_resources WHERE session_id='s1'"
        ).fetchall()
        self.assertEqual(sorted(r["resource"] for r in rows), ["a.py", "b.py"])

    def test_entry_json_mirrors_the_cli_entry_shape(self):
        """The materializer and the PHASE-1 reconciler both read entry_json, so
        it must match what task_claim.py cmd_open() writes into the JSON --
        same keys, or the reconciler reports permanent drift."""
        self.open_claim("s1", ["a.py"])
        e = self.entry("s1", T0)
        self.assertEqual(sorted(e), sorted([
            "session_id", "session_label", "claimed_at", "task", "resources",
            "status"]))
        self.assertEqual(e["status"], "active")
        self.assertEqual(e["resources"], ["a.py"])

    def test_client_supplied_claimed_at_is_used_verbatim(self):
        """D12: a server-stamped claimed_at would give the DB row and the git
        entry different halves of the primary key while the client still writes
        git, and the reconciler would call that drift forever."""
        verdict, payload = self.open_claim("s1", ["a.py"], at="2026-01-02T03:04:05Z")
        self.assertEqual(verdict, "ok")
        self.assertEqual(payload["claimed_at"], "2026-01-02T03:04:05Z")
        self.assertIsNotNone(self.entry("s1", "2026-01-02T03:04:05Z"))

    def test_server_stamps_claimed_at_when_client_omits_it(self):
        verdict, payload = db.try_open_task_claim(
            self.conn, session_id="s1", session_label="l", task="t",
            resources=["a.py"], now=T0)
        self.assertEqual(verdict, "ok")
        self.assertEqual(payload["claimed_at"], T0)

    def test_rival_on_the_same_file_refuses_and_writes_nothing(self):
        self.open_claim("s1", ["shared.py"])
        verdict, payload = self.open_claim("s2", ["shared.py"], at=T1, now=T1)
        self.assertEqual(verdict, "owned_by_other")
        self.assertEqual(payload["rivals"][0]["session_id"], "s1")
        self.assertEqual(payload["rivals"][0]["resource"], "shared.py")
        self.assertIsNone(self.entry("s2", T1))

    def test_disjoint_resources_do_not_contend(self):
        self.open_claim("s1", ["a.py"])
        verdict, _ = self.open_claim("s2", ["b.py"], at=T1, now=T1)
        self.assertEqual(verdict, "ok")

    def test_a_stale_rival_does_not_block(self):
        """Matches task_claim.py's arbitration: rivals older than
        stale_after_hours are excluded."""
        self.open_claim("s1", ["shared.py"], at=T0, now=T0)
        verdict, _ = self.open_claim("s2", ["shared.py"], at=T_STALE, now=T_STALE)
        self.assertEqual(verdict, "ok")

    def test_a_done_rival_does_not_block(self):
        self.open_claim("s1", ["shared.py"])
        db.close_task_claim(self.conn, "s1", closed_at=T1,
                            completion_note="landed", now=T1)
        verdict, _ = self.open_claim("s2", ["shared.py"], at=T1, now=T1)
        self.assertEqual(verdict, "ok")

    def test_directory_scope_resources_are_never_arbitrated(self):
        """governance.sh holds REE_assembly/evidence/ for a whole regen and
        fails open on a non-zero open. Arbitrating a directory would stop every
        evidence session AND leave every regen unprotected."""
        self.open_claim("gov", ["REE_assembly/evidence/"])
        verdict, payload = self.open_claim(
            "s2", ["REE_assembly/evidence/"], at=T1, now=T1)
        self.assertEqual(verdict, "ok")
        self.assertTrue(any("not arbitrated" in n for n in payload["notes"]))

    def test_allow_overlap_downgrades_the_verdict_to_a_note(self):
        self.open_claim("s1", ["shared.py"])
        verdict, payload = self.open_claim(
            "s2", ["shared.py"], at=T1, now=T1, allow_overlap=True)
        self.assertEqual(verdict, "ok")
        self.assertEqual(len(payload["rivals"]), 1)
        self.assertTrue(any("allow_overlap" in n for n in payload["notes"]),
                        "the overlap must still be REPORTED, never silenced")

    def test_open_is_idempotent_per_session_id(self):
        """D8. The git path retries routinely under contention; a naive INSERT
        would mint a second row every time."""
        self.open_claim("s1", ["a.py"])
        verdict, payload = self.open_claim("s1", ["a.py"], at=T1, now=T1)
        self.assertEqual(verdict, "idempotent")
        self.assertEqual(payload["claimed_at"], T0)
        n = self.conn.execute(
            "SELECT COUNT(*) c FROM task_claims WHERE session_id='s1'"
        ).fetchone()["c"]
        self.assertEqual(n, 1)

    def test_a_session_does_not_contend_with_its_own_closed_claim(self):
        self.open_claim("s1", ["a.py"])
        db.close_task_claim(self.conn, "s1", closed_at=T1,
                            completion_note="done", now=T1)
        verdict, _ = self.open_claim("s1", ["a.py"], at=T1, now=T1)
        self.assertEqual(verdict, "ok")

    def test_two_concurrent_opens_on_one_resource_produce_exactly_one_owner(self):
        """THE reason this migration exists. Replays the 2026-07-28 shape:
        several sessions released by one event, all claiming the same file
        within seconds. Under git each read TASK_CLAIMS.json clean and each
        wrote. Here BEGIN IMMEDIATE serialises them and exactly one wins.

        Real threads on real connections -- a mocked conn would not exercise
        the transaction at all, which is the only thing under test."""
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(3)

        def worker(sid, stamp):
            conn = db.connect(self.dbpath)
            try:
                barrier.wait(timeout=10)
                v, _ = db.try_open_task_claim(
                    conn, session_id=sid, session_label="l", task="t",
                    resources=["ree-v3/runner_remote_control.py"],
                    claimed_at=stamp, now=stamp)
                with lock:
                    results.append((sid, v))
            finally:
                conn.close()

        threads = [threading.Thread(target=worker, args=("s%d" % i,
                                                         "2026-08-27T12:00:0%dZ" % i))
                   for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        oks = [sid for sid, v in results if v == "ok"]
        losers = [sid for sid, v in results if v == "owned_by_other"]
        self.assertEqual(len(results), 3, results)
        self.assertEqual(len(oks), 1, "exactly one owner: %r" % (results,))
        self.assertEqual(len(losers), 2, "the other two must be told: %r" % (results,))
        n = self.conn.execute(
            "SELECT COUNT(*) c FROM task_claims WHERE status='active'"
        ).fetchone()["c"]
        self.assertEqual(n, 1)


# --------------------------------------------------------------------------
# task_claim: close / renew / amend / dedupe
# --------------------------------------------------------------------------

class TestCloseRenewAmend(Base):

    def test_close_sets_status_closed_at_and_note(self):
        self.open_claim("s1", ["a.py"])
        verdict, payload = db.close_task_claim(
            self.conn, "s1", closed_at="2026-08-27T09:00:00Z",
            completion_note="REE_assembly abc123", now=T1)
        self.assertEqual(verdict, "ok")
        e = self.entry("s1", T0)
        self.assertEqual(e["status"], "done")
        self.assertEqual(e["closed_at"], "2026-08-27T09:00:00Z")
        self.assertEqual(e["completion_note"], "REE_assembly abc123")

    def test_close_refuses_rather_than_guessing_when_ambiguous(self):
        """The claim key is (session_id, claimed_at). close NEVER guesses --
        it refuses and prints the candidate stamps, because a headless worker
        cannot go and read the JSON."""
        self.open_claim("s1", ["a.py"], at=T0)
        # Second active claim under one session_id, as a twice-dispatched chip
        # produces. Inserted directly: `open` is idempotent by design.
        db.upsert_task_claim(self.conn, {
            "session_id": "s1", "claimed_at": T1, "session_label": "l",
            "task": "t2", "resources": ["b.py"], "status": "active"}, now=T1)
        verdict, payload = db.close_task_claim(
            self.conn, "s1", closed_at=T1, completion_note="x", now=T1)
        self.assertEqual(verdict, "ambiguous")
        self.assertEqual(sorted(c["claimed_at"] for c in payload["candidates"]),
                         [T0, T1])
        self.assertEqual(self.entry("s1", T0)["status"], "active")

    def test_close_with_claimed_at_disambiguates(self):
        self.open_claim("s1", ["a.py"], at=T0)
        db.upsert_task_claim(self.conn, {
            "session_id": "s1", "claimed_at": T1, "session_label": "l",
            "task": "t2", "resources": [], "status": "active"}, now=T1)
        verdict, _ = db.close_task_claim(
            self.conn, "s1", closed_at=T1, completion_note="x",
            claimed_at=T1, now=T1)
        self.assertEqual(verdict, "ok")
        self.assertEqual(self.entry("s1", T0)["status"], "active")
        self.assertEqual(self.entry("s1", T1)["status"], "done")

    def test_close_on_an_already_closed_claim_says_so(self):
        self.open_claim("s1", ["a.py"])
        db.close_task_claim(self.conn, "s1", closed_at=T1,
                            completion_note="x", now=T1)
        verdict, payload = db.close_task_claim(
            self.conn, "s1", closed_at=T1, completion_note="y", now=T1)
        self.assertEqual(verdict, "already_closed")
        self.assertEqual(self.entry("s1", T0)["completion_note"], "x")

    def test_close_on_an_unknown_session_is_not_found(self):
        verdict, _ = db.close_task_claim(self.conn, "nope", closed_at=T1,
                                         completion_note="x", now=T1)
        self.assertEqual(verdict, "not_found")

    def test_renew_moves_the_stamp_and_carries_the_resources(self):
        self.open_claim("s1", ["a.py", "b.py"])
        verdict, payload = db.renew_task_claim(
            self.conn, "s1", new_claimed_at=T1, now=T1)
        self.assertEqual(verdict, "ok")
        self.assertEqual(payload["previous_claimed_at"], T0)
        self.assertIsNone(self.entry("s1", T0))
        e = self.entry("s1", T1)
        self.assertEqual(sorted(e["resources"]), ["a.py", "b.py"])
        rows = self.conn.execute(
            "SELECT claimed_at FROM task_claim_resources WHERE session_id='s1'"
        ).fetchall()
        self.assertEqual({r["claimed_at"] for r in rows}, {T1},
                         "child rows must move with the key, not be orphaned")

    def test_renew_refuses_when_it_would_hand_ownership_to_a_live_rival(self):
        self.open_claim("s1", ["shared.py"], at=T0)
        db.upsert_task_claim(self.conn, {
            "session_id": "s2", "claimed_at": "2026-08-27T10:30:00Z",
            "session_label": "l", "task": "t", "resources": ["shared.py"],
            "status": "active"}, now=T1)
        verdict, payload = db.renew_task_claim(
            self.conn, "s1", new_claimed_at=T1, now=T1)
        self.assertEqual(verdict, "would_lose_ownership")
        self.assertEqual(payload["rivals"][0]["session_id"], "s2")
        self.assertIsNotNone(self.entry("s1", T0))

    def test_amend_replaces_the_note_and_keeps_the_original(self):
        self.open_claim("s1", ["a.py"])
        db.close_task_claim(self.conn, "s1", closed_at=T1,
                            completion_note="first", now=T1)
        verdict, _ = db.amend_task_claim(self.conn, "s1",
                                         completion_note="second", now=T1)
        self.assertEqual(verdict, "ok")
        e = self.entry("s1", T0)
        self.assertEqual(e["completion_note"], "second")
        self.assertEqual(e["completion_note_history"], ["first"])

    def test_amend_does_not_move_closed_at_or_status(self):
        self.open_claim("s1", ["a.py"])
        db.close_task_claim(self.conn, "s1", closed_at="2026-08-01T00:00:00Z",
                            completion_note="first", now=T1)
        db.amend_task_claim(self.conn, "s1", completion_note="second", now=T1)
        e = self.entry("s1", T0)
        self.assertEqual(e["closed_at"], "2026-08-01T00:00:00Z")
        self.assertEqual(e["status"], "done")

    def test_amend_refuses_on_an_active_claim(self):
        """amend corrects a note on an already-CLOSED claim. An active one has
        no landing to describe."""
        self.open_claim("s1", ["a.py"])
        verdict, _ = db.amend_task_claim(self.conn, "s1",
                                         completion_note="x", now=T1)
        self.assertEqual(verdict, "not_found")

    def test_dedupe_is_an_accepted_no_op(self):
        """D3: under the composite primary key the duplicate class dedupe
        cleans up cannot be created, so the verb is kept for compatibility and
        its logic is deliberately NOT ported."""
        verdict, payload = db.dedupe_task_claim(self.conn, "s1")
        self.assertEqual(verdict, "ok")
        self.assertEqual(payload["removed"], 0)

    def test_the_composite_pk_makes_a_byte_identical_duplicate_impossible(self):
        """The other half of D3: the 2026-08-18 incident is prevented at the
        source. A second INSERT at the same key fails rather than appending."""
        self.open_claim("s1", ["a.py"], at=T0)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO task_claims (session_id, claimed_at, "
                "session_label, task, status, entry_json, updated_at) "
                "VALUES ('s1', ?, 'l', 't', 'active', '{}', ?)", (T0, T0))


# --------------------------------------------------------------------------
# chips
# --------------------------------------------------------------------------

def _chip(ref, **kw):
    c = {
        "chip_ref": ref,
        "task_id": None,
        "origin": "spawn_task",
        "kind": "work",
        "urgency": False,
        "session_id": "spawner",
        "session_label": "lbl",
        "title": "t",
        "tldr": "d",
        "prompt": "do the thing [chip_ref: %s] now" % ref,
        "cwd": "/Users/dgolden/REE_Working",
        "origin_host": "DLAPTOP",
        "spawned_at": T0,
        "status": "open",
        "resolved_at": None,
        "resolved_by_session_id": None,
        "resolution_note": None,
        "resolution_note_auto": False,
        "claimed_by": None,
        "claimed_at": None,
        "claim_note": None,
        "claimed_host": None,
        "attached_by_session_id": None,
        "attached_at": None,
    }
    c.update(kw)
    return c


class TestChips(Base):

    def chip_entry(self, ref):
        row = self.conn.execute(
            "SELECT entry_json FROM chip_ledger WHERE chip_ref=?", (ref,)
        ).fetchone()
        return None if row is None else json.loads(row["entry_json"])

    def test_record_writes_the_chip(self):
        verdict, _ = db.record_chip(self.conn, _chip("c1"), now=T0)
        self.assertEqual(verdict, "ok")
        self.assertEqual(self.chip_entry("c1")["status"], "open")

    def test_record_refuses_a_prompt_without_the_chip_ref_marker(self):
        """Hard refusal, matching the CLI since 2026-08-03. A chip whose
        stored prompt lacks the marker can never self-report at close -- 12
        such chips were recorded in one governance session while this was
        only a warning."""
        verdict, payload = db.record_chip(
            self.conn, _chip("c1", prompt="see the session history"), now=T0)
        self.assertEqual(verdict, "missing_marker")
        self.assertIn("[chip_ref: c1]", payload["marker"])
        self.assertIsNone(self.chip_entry("c1"))

    def test_record_is_idempotent_for_a_task_id_less_tick_repeat(self):
        db.record_chip(self.conn, _chip("c1", origin="hygiene_tick"), now=T0)
        verdict, _ = db.record_chip(
            self.conn, _chip("c1", origin="hygiene_tick"), now=T1)
        self.assertEqual(verdict, "idempotent")

    def test_record_refuses_a_ref_reused_by_a_different_task_id(self):
        db.record_chip(self.conn, _chip("c1", task_id="task_a"), now=T0)
        verdict, _ = db.record_chip(
            self.conn, _chip("c1", task_id="task_b"), now=T1)
        self.assertEqual(verdict, "ref_collision")

    def test_record_does_not_auto_claim_on_a_task_id(self):
        """chip_ledger.py's RECORD/ATTACH DO NOT AUTO-CLAIM ON task_id: a
        task_id means only that spawn_task made a clickable suggestion. The
        2026-08-22..25 auto-claim window left 40-60h-old chips marked claimed
        with no worktree and no process anywhere."""
        db.record_chip(self.conn, _chip("c1", task_id="task_a"), now=T0)
        self.assertIsNone(self.chip_entry("c1")["claimed_by"])

    def test_claim_marks_the_chip_and_canonicalises_the_host(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        verdict, payload = db.try_claim_chip(
            self.conn, chip_ref="c1", claimed_by="sess",
            claimed_host="DLAPTOP-5.local", claimed_at=T1, now=T1)
        self.assertEqual(verdict, "ok")
        row = self.conn.execute(
            "SELECT claimed_host, claimed_host_raw FROM chip_ledger "
            "WHERE chip_ref='c1'").fetchone()
        self.assertEqual(row["claimed_host"], "DLAPTOP")
        self.assertEqual(row["claimed_host_raw"], "DLAPTOP-5.local",
                         "the raw report is the audit trail and must survive")

    def test_a_live_claim_by_another_session_is_refused(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.try_claim_chip(self.conn, chip_ref="c1", claimed_by="a",
                          claimed_host="DLAPTOP", claimed_at=T0, now=T0)
        verdict, payload = db.try_claim_chip(
            self.conn, chip_ref="c1", claimed_by="b",
            claimed_host="ree-cloud-5", claimed_at=T1, now=T1)
        self.assertEqual(verdict, "already_claimed")
        self.assertEqual(payload["claimed_by"], "a")

    def test_a_stale_claim_is_superseded_with_an_explanatory_prefix(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.try_claim_chip(self.conn, chip_ref="c1", claimed_by="a",
                          claimed_host="DLAPTOP", claimed_at=T0, now=T0)
        verdict, payload = db.try_claim_chip(
            self.conn, chip_ref="c1", claimed_by="b", claimed_host="DLAPTOP",
            claimed_at=T_STALE, now=T_STALE)
        self.assertEqual(verdict, "ok")
        self.assertIn("superseded", payload["note_prefix"])

    def test_reclaiming_your_own_chip_is_a_refresh(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.try_claim_chip(self.conn, chip_ref="c1", claimed_by="a",
                          claimed_host="DLAPTOP", claimed_at=T0, now=T0)
        verdict, payload = db.try_claim_chip(
            self.conn, chip_ref="c1", claimed_by="a", claimed_host="DLAPTOP",
            claimed_at=T1, now=T1)
        self.assertEqual(verdict, "ok")
        self.assertIn("refreshed", payload["note_prefix"])

    def test_claiming_a_resolved_chip_is_refused(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.resolve_chip(self.conn, "done", chip_ref="c1", note="n", now=T0)
        verdict, _ = db.try_claim_chip(
            self.conn, chip_ref="c1", claimed_by="a", claimed_host="DLAPTOP",
            now=T1)
        self.assertEqual(verdict, "not_open")

    def test_two_concurrent_chip_claims_produce_exactly_one_winner(self):
        """The cross-host dispatch mutex. Under git this depended on push
        ordering, which is why the 2026-08-09 double-dispatch happened when a
        claim stayed on a local branch."""
        db.record_chip(self.conn, _chip("c1"), now=T0)
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(4)

        def worker(i):
            conn = db.connect(self.dbpath)
            try:
                barrier.wait(timeout=10)
                v, _ = db.try_claim_chip(
                    conn, chip_ref="c1", claimed_by="sess%d" % i,
                    claimed_host="DLAPTOP", claimed_at=T1, now=T1)
                with lock:
                    results.append(v)
            finally:
                conn.close()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(len(results), 4, results)
        self.assertEqual(results.count("ok"), 1, results)
        self.assertEqual(results.count("already_claimed"), 3, results)

    def test_unclaim_releases_and_reports_the_previous_holder(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.try_claim_chip(self.conn, chip_ref="c1", claimed_by="a",
                          claimed_host="DLAPTOP", claimed_at=T0, now=T0)
        verdict, payload = db.unclaim_chip(self.conn, chip_ref="c1", now=T1)
        self.assertEqual(verdict, "ok")
        self.assertEqual(payload["was_claimed_by"], "a")
        self.assertIsNone(self.chip_entry("c1")["claimed_by"])

    def test_unclaim_on_an_unclaimed_chip_is_a_success(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        verdict, payload = db.unclaim_chip(self.conn, chip_ref="c1", now=T1)
        self.assertEqual(verdict, "ok")
        self.assertIsNone(payload["was_claimed_by"])

    def test_resolve_marks_done(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        verdict, payload = db.resolve_chip(
            self.conn, "done", chip_ref="c1", note="landed",
            resolved_by_session_id="w", resolved_at=T1, now=T1)
        self.assertEqual(verdict, "ok")
        self.assertTrue(payload["changed"])
        e = self.chip_entry("c1")
        self.assertEqual(e["status"], "done")
        self.assertEqual(e["resolution_note"], "landed")

    def test_resolve_on_an_already_resolved_chip_reports_changed_false(self):
        """D11. Today this is a SILENT no-op indistinguishable from a real
        transition -- exactly the trap that drops a headless worker's report."""
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.resolve_chip(self.conn, "done", chip_ref="c1", note="a", now=T0)
        verdict, payload = db.resolve_chip(
            self.conn, "done", chip_ref="c1", note="b", now=T1)
        self.assertEqual(verdict, "ok")
        self.assertFalse(payload["changed"])
        self.assertEqual(self.chip_entry("c1")["resolution_note"], "a")

    def test_a_real_note_replaces_an_automated_one_at_equal_status(self):
        """The 2026-08-14 recovery path: a routine tick resolved a chip out
        from under the worker who had done the work, and the worker's own
        resolve then no-op'd and DROPPED its report."""
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.resolve_chip(self.conn, "done", chip_ref="c1", note="auto: cleared",
                        note_auto=True, now=T0)
        verdict, payload = db.resolve_chip(
            self.conn, "done", chip_ref="c1", note="the real report", now=T1)
        self.assertEqual(verdict, "ok")
        self.assertTrue(payload["changed"])
        e = self.chip_entry("c1")
        self.assertEqual(e["resolution_note"], "the real report")
        self.assertFalse(e["resolution_note_auto"])
        self.assertEqual(e["resolution_note_history"][0]["resolution_note"],
                         "auto: cleared")

    def test_a_human_note_is_never_overwritten_at_equal_status(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.resolve_chip(self.conn, "done", chip_ref="c1", note="worker report",
                        now=T0)
        verdict, payload = db.resolve_chip(
            self.conn, "done", chip_ref="c1", note="later note", now=T1)
        self.assertFalse(payload["changed"])
        self.assertEqual(self.chip_entry("c1")["resolution_note"],
                         "worker report")

    def test_a_different_terminal_status_refuses_without_force(self):
        """The 2026-08-25 guard: a session called resolve --status withdrawn on
        a chip that was already legitimately done, with a long note describing
        real landed work."""
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.resolve_chip(self.conn, "done", chip_ref="c1", note="real work",
                        now=T0)
        verdict, payload = db.resolve_chip(
            self.conn, "withdrawn", chip_ref="c1", note="oops", now=T1)
        self.assertEqual(verdict, "terminal_conflict")
        self.assertEqual(payload["resolution_note"], "real work")
        self.assertEqual(self.chip_entry("c1")["status"], "done")

    def test_force_preserves_the_prior_terminal_resolution_in_history(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.resolve_chip(self.conn, "done", chip_ref="c1", note="real work",
                        resolved_by_session_id="w1", now=T0)
        verdict, _ = db.resolve_chip(
            self.conn, "withdrawn", chip_ref="c1", note="superseded",
            force=True, now=T1)
        self.assertEqual(verdict, "ok")
        e = self.chip_entry("c1")
        self.assertEqual(e["status"], "withdrawn")
        self.assertEqual(e["resolution_note_history"][0]["resolution_note"],
                         "real work")
        self.assertEqual(e["resolution_note_history"][0]["status"], "done")

    def test_resolve_rejects_a_status_outside_the_terminal_set(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        verdict, _ = db.resolve_chip(self.conn, "open", chip_ref="c1", now=T1)
        self.assertEqual(verdict, "bad_status")

    def test_resolve_by_task_id_resolves_through_the_partial_index(self):
        db.record_chip(self.conn, _chip("c1", task_id="task_x"), now=T0)
        db.record_chip(self.conn, _chip("c2"), now=T0)
        verdict, payload = db.resolve_chip(
            self.conn, "done", task_id="task_x", note="n", now=T1)
        self.assertEqual(verdict, "ok")
        self.assertEqual(payload["chip_ref"], "c1")

    def test_a_task_id_lookup_never_matches_a_null_task_id_row(self):
        """D4: task_id is NULL on 1043/1692 live rows."""
        db.record_chip(self.conn, _chip("c2"), now=T0)
        verdict, _ = db.resolve_chip(self.conn, "done", task_id="task_x",
                                     note="n", now=T1)
        self.assertEqual(verdict, "not_found")

    def test_attach_sets_the_task_id_without_claiming(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        verdict, _ = db.attach_chip(self.conn, "c1", "task_x",
                                    attached_by_session_id="s", now=T1)
        self.assertEqual(verdict, "ok")
        e = self.chip_entry("c1")
        self.assertEqual(e["task_id"], "task_x")
        self.assertIsNone(e["claimed_by"])

    def test_attach_refuses_a_task_id_already_on_another_chip(self):
        db.record_chip(self.conn, _chip("c1", task_id="task_x"), now=T0)
        db.record_chip(self.conn, _chip("c2"), now=T0)
        verdict, payload = db.attach_chip(self.conn, "c2", "task_x", now=T1)
        self.assertEqual(verdict, "task_id_taken")
        self.assertEqual(payload["chip_ref"], "c1")

    def test_attach_is_idempotent_for_the_same_task_id(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.attach_chip(self.conn, "c1", "task_x", now=T0)
        verdict, _ = db.attach_chip(self.conn, "c1", "task_x", now=T1)
        self.assertEqual(verdict, "ok")

    def test_amend_prompt_keeps_the_broken_original(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        verdict, _ = db.amend_chip_prompt(
            self.conn, "c1", "the real text [chip_ref: c1] here",
            reason="was a placeholder", now=T1)
        self.assertEqual(verdict, "ok")
        e = self.chip_entry("c1")
        self.assertIn("the real text", e["prompt"])
        self.assertIn("do the thing", e["prompt_history"][0]["prompt"])

    def test_amend_prompt_still_requires_the_marker(self):
        db.record_chip(self.conn, _chip("c1"), now=T0)
        verdict, _ = db.amend_chip_prompt(self.conn, "c1", "no marker here",
                                          now=T1)
        self.assertEqual(verdict, "missing_marker")

    def test_archived_fields_are_absent_not_null_in_entry_json(self):
        """D5: chip_ledger.archived_field() distinguishes 'never had a prompt'
        from 'its prompt was archived' by reading the `archived` block. A
        stripped field must be OMITTED, and the ROW must survive -- a DELETE
        here would be the tombstone CLAUDE.md forbids."""
        db.record_chip(self.conn, _chip("c1"), now=T0)
        db.resolve_chip(self.conn, "done", chip_ref="c1", note="n", now=T0)
        self.conn.execute(
            "UPDATE chip_ledger SET prompt=NULL, resolution_note=NULL, "
            "archived_json=? WHERE chip_ref='c1'",
            (json.dumps({"file": "chip_archive/2026-08.json", "month": "2026-08",
                         "fields": ["prompt", "resolution_note"], "at": T1}),))
        entry = db._reserialise_chip_row(self.conn, "c1")
        self.assertNotIn("prompt", entry)
        self.assertNotIn("resolution_note", entry)
        self.assertEqual(entry["chip_ref"], "c1")
        self.assertEqual(entry["status"], "done")
        self.assertEqual(entry["archived"]["month"], "2026-08")


# --------------------------------------------------------------------------
# Negative controls -- what this phase must NOT have grown
# --------------------------------------------------------------------------

class TestScopeGuards(Base):

    def test_no_archive_verb_exists_in_db(self):
        """D7: chip archive stays git-side. Its correctness gate is that the
        archive file reached ORIGIN, which is inherently a git fact with no DB
        equivalent. If this ever fails, someone ported it here -- which must be
        a deliberate Phase-3 decision, not a side effect."""
        for name in ("archive_chip", "archive_chips", "cmd_archive"):
            self.assertFalse(hasattr(db, name),
                             "db.%s must not exist in Phase 2 (D7)" % name)

    def test_no_verb_writes_git(self):
        """Every verb here is pure sqlite. The DB->git materializer is NOT part
        of this phase; if one of these ever shells out to git, the whole
        'structurally incapable of dirtying the source repo' property the
        PHASE-1 reconciler established is gone."""
        import inspect
        for name in ("try_open_task_claim", "close_task_claim",
                     "renew_task_claim", "amend_task_claim",
                     "dedupe_task_claim", "record_chip", "try_claim_chip",
                     "unclaim_chip", "resolve_chip", "attach_chip",
                     "amend_chip_prompt"):
            src = inspect.getsource(getattr(db, name))
            for banned in ("subprocess", "os.system", '"git"', "'git'"):
                self.assertNotIn(banned, src,
                                 "%s must not touch git: found %s"
                                 % (name, banned))


if __name__ == "__main__":
    unittest.main()
