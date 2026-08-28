"""Contracts for task_claim_chip_git_writer.py -- the PHASE-2b DB->git
materializer tick.

Same real-git fixture shape as test_task_claim_chip_shadow_sync.py (bare
remote + working clone), so fetch/show/reset/commit/push exercise the real
code paths. Time-independent: every retention assertion injects now_iso.

Pins:
  1. ROUND-TRIP BYTE-EQUALITY (D15): ingest a realistic file -> render ==
     the exact file text, both registries, including an entry carrying an
     unmodelled key in a non-alphabetical key order (the verbatim
     entry_json contract from the 2026-08-28 db.py change).
  2. RETENTION (D14): a done entry older than RETAIN_HOURS is dropped from
     the render; active and recent-done entries are kept; chips are NEVER
     dropped.
  3. FALLBACK-COMMIT RACE: an entry committed to origin between ticks is
     ingested before rendering and therefore survives materialization --
     the render can never revert a fallback commit.
  4. COORDINATOR-MUTATED ROWS render in the client's canonical key order
     (_claim_entry_json's construction order == task_claim.py cmd_open's).
  5. WRITE MODE commits + pushes only on a state change; a matching tick
     commits nothing (no liveness-tick commits); CHECK MODE never writes,
     never commits, never moves origin.
  6. SOURCE-ORDER FIDELITY: a client-side file reorder (merge machinery)
     does not produce a spurious rewrite -- the render follows the FILE's
     row order for rows the file carries.

ASCII-only.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db  # noqa: E402
import task_claim_chip_git_writer as writer  # noqa: E402

NOW = "2026-08-28T12:00:00Z"


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=check,
    )


def _bare_remote(parent, name="REE_Working.git"):
    remote = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _claims_doc(claims):
    # Real top-level key order: claims, schema_version, stale_after_hours.
    return {"claims": claims, "schema_version": "v1", "stale_after_hours": 6}


def _chips_doc(chips):
    # Real top-level key order: schema_version, chips.
    return {"schema_version": "task_chips/v1", "chips": chips}


def _claim(session_id, claimed_at, status="active", **extra):
    # Client construction order (task_claim.py cmd_open).
    entry = {
        "session_id": session_id,
        "session_label": "label-%s" % session_id,
        "claimed_at": claimed_at,
        "task": "task for %s" % session_id,
        "resources": ["r/%s.json" % session_id],
        "status": status,
    }
    entry.update(extra)
    return entry


def _chip(chip_ref, **extra):
    entry = {
        "task_id": None,
        "chip_ref": chip_ref,
        "origin": "headless",
        "kind": "work",
        "urgency": False,
        "session_id": "unknown",
        "session_label": "",
        "title": "t %s" % chip_ref,
        "tldr": "td",
        "prompt": "[chip_ref: %s]\n\nbody" % chip_ref,
        "cwd": "/x",
        "spawned_at": "2026-08-27T09:00:00Z",
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
    entry.update(extra)
    return entry


def _serialise(doc):
    return json.dumps(doc, indent=2) + "\n"


def _seed_repo(parent, remote, claims_doc, chips_doc, name="REE_Working"):
    repo = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)],
                   check=True)
    _git(repo, "config", "user.email", "writer@test")
    _git(repo, "config", "user.name", "writer-test")
    (repo / writer.CLAIMS_REL_PATH).write_text(_serialise(claims_doc))
    (repo / writer.CHIPS_REL_PATH).write_text(_serialise(chips_doc))
    _git(repo, "add", writer.CLAIMS_REL_PATH, writer.CHIPS_REL_PATH)
    _git(repo, "commit", "-q", "-m", "seed")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "master")
    return repo


class _Fixture(unittest.TestCase):

    CLAIMS = None  # set per test class
    CHIPS = None

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="registry_writer_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._remote = _bare_remote(self._tmp)
        claims = self.CLAIMS if self.CLAIMS is not None else _claims_doc(
            [_claim("s-active", "2026-08-28T10:00:00Z"),
             _claim("s-done-recent", "2026-08-28T09:00:00Z", status="done",
                    closed_at="2026-08-28T11:00:00Z",
                    completion_note="landed")])
        chips = self.CHIPS if self.CHIPS is not None else _chips_doc(
            [_chip("chip-a"), _chip("chip-b")])
        self._repo = _seed_repo(self._tmp, self._remote, claims, chips)

    def tearDown(self):
        self._conn.close()

    def _tick(self, mode="check", now_iso=NOW):
        return writer.materialize_once(
            self._conn, str(self._repo), branch="master", mode=mode,
            now_iso=now_iso)

    def _origin_text(self, rel_path):
        out = _git(self._repo, "show", "origin/master:%s" % rel_path)
        return out.stdout


class TestRoundTripByteEquality(_Fixture):

    def test_fresh_ingest_renders_byte_identical(self):
        result = self._tick()
        self.assertIsNotNone(result)
        self.assertTrue(result["claims_match"])
        self.assertTrue(result["chips_match"])
        self.assertFalse(result["committed"])

    def test_unmodelled_keys_and_odd_order_round_trip(self):
        claims = _claims_doc([
            _claim("s1", "2026-08-28T10:00:00Z"),
            # closed fields + an unmodelled key, deliberately interleaved in
            # a non-alphabetical, non-canonical order.
            {"session_id": "s2", "handoff_marker": True,
             "session_label": "L2", "claimed_at": "2026-08-28T09:30:00Z",
             "status": "active", "task": "t2", "resources": [],
             "custom_field": {"nested": [1, 2]}},
        ])
        chips = _chips_doc([
            _chip("chip-h", handoff_pending=True),
        ])
        repo2 = _seed_repo(self._tmp, _bare_remote(self._tmp, "r2.git"),
                           claims, chips, name="REE_Working_2")
        result = writer.materialize_once(self._conn, str(repo2),
                                         branch="master", mode="check",
                                         now_iso=NOW)
        self.assertTrue(result["claims_match"])
        self.assertTrue(result["chips_match"])

    def test_source_missing_trailing_newline_still_matches(self):
        # The live TASK_CHIPS.json on origin ends without a final newline
        # (historical writer quirk). Same content must read as a match --
        # rewriting 10 MB to add one byte is the churn this writer exists
        # to avoid.
        remote = _bare_remote(self._tmp, "r_nl.git")
        repo2 = _seed_repo(self._tmp, remote,
                           _claims_doc([_claim("s1", "2026-08-28T10:00:00Z")]),
                           _chips_doc([_chip("chip-nl")]),
                           name="REE_Working_nl")
        text = (repo2 / writer.CHIPS_REL_PATH).read_text()
        (repo2 / writer.CHIPS_REL_PATH).write_text(text.rstrip("\n"))
        _git(repo2, "add", writer.CHIPS_REL_PATH)
        _git(repo2, "commit", "-q", "-m", "strip trailing newline")
        _git(repo2, "push", "-q", "origin", "master")
        result = writer.materialize_once(self._conn, str(repo2),
                                         branch="master", mode="write",
                                         now_iso=NOW)
        self.assertTrue(result["chips_match"])
        self.assertFalse(result["committed"])


class TestRetention(_Fixture):

    CLAIMS = _claims_doc([
        _claim("s-active", "2026-08-28T10:00:00Z"),
        _claim("s-done-old", "2026-08-26T09:00:00Z", status="done",
               closed_at="2026-08-26T10:00:00Z"),
        _claim("s-done-recent", "2026-08-28T08:00:00Z", status="done",
               closed_at="2026-08-28T11:00:00Z"),
    ])

    def test_aged_done_entry_is_dropped_and_rest_byte_exact(self):
        result = self._tick()
        self.assertFalse(result["claims_match"])
        self.assertEqual(result["claims"]["n_retention_dropped"], 1)
        self.assertEqual(result["claims"]["n_rendered"], 2)
        self.assertEqual(result["claims_delta"], {"added": 0, "dropped": 1})
        # And the render equals the source doc minus exactly that entry.
        claims_render, _, _snaps = writer.render_task_claims(
            self._conn, source_doc=self.CLAIMS, now_iso=NOW)
        expect = dict(self.CLAIMS)
        expect["claims"] = [e for e in self.CLAIMS["claims"]
                            if e["session_id"] != "s-done-old"]
        self.assertEqual(claims_render, _serialise(expect))

    def test_done_with_no_closed_at_falls_back_to_claimed_at(self):
        claims = _claims_doc([
            _claim("s-old-no-close", "2026-08-25T09:00:00Z", status="done"),
        ])
        repo2 = _seed_repo(self._tmp, _bare_remote(self._tmp, "r3.git"),
                           claims, _chips_doc([]), name="REE_Working_3")
        result = writer.materialize_once(self._conn, str(repo2),
                                         branch="master", mode="check",
                                         now_iso=NOW)
        self.assertEqual(result["claims"]["n_retention_dropped"], 1)
        self.assertEqual(result["claims"]["n_rendered"], 0)

    def test_chips_are_never_dropped(self):
        result = self._tick()
        self.assertTrue(result["chips_match"])
        self.assertEqual(result["chips"]["n_rendered"], 2)


class TestFallbackCommitRace(_Fixture):

    def test_entry_committed_to_origin_survives_materialization(self):
        # Tick 1 populates the DB.
        self._tick()
        # A fallback session commits a NEW claim straight to origin.
        doc = json.loads((self._repo / writer.CLAIMS_REL_PATH).read_text())
        doc["claims"].append(_claim("s-fallback", "2026-08-28T11:30:00Z"))
        (self._repo / writer.CLAIMS_REL_PATH).write_text(_serialise(doc))
        _git(self._repo, "add", writer.CLAIMS_REL_PATH)
        _git(self._repo, "commit", "-q", "-m", "claim: fallback open")
        _git(self._repo, "push", "-q", "origin", "master")
        # Tick 2 must ingest-before-render: the fallback entry is kept and
        # the render matches origin exactly (no spurious rewrite).
        result = self._tick(mode="write")
        self.assertTrue(result["claims_match"])
        self.assertFalse(result["committed"])
        self.assertIn("s-fallback", self._origin_text(writer.CLAIMS_REL_PATH))


class TestCoordinatorMutatedRows(_Fixture):

    def test_open_via_coordinator_renders_in_client_key_order(self):
        self._tick()
        verdict, payload = db.try_open_task_claim(
            self._conn, "s-coord", "coord label", "coord task",
            ["x.json"], claimed_at="2026-08-28T11:45:00Z")
        self.assertEqual(verdict, "ok")
        claims_render, stats, _snaps = writer.render_task_claims(
            self._conn,
            source_doc=json.loads(self._origin_text(writer.CLAIMS_REL_PATH)),
            now_iso=NOW)
        rendered = json.loads(claims_render)["claims"]
        new = [e for e in rendered if e["session_id"] == "s-coord"]
        self.assertEqual(len(new), 1)
        # DB-only row appends at the END, in the client's key order.
        self.assertEqual(rendered[-1]["session_id"], "s-coord")
        self.assertEqual(
            list(new[0].keys()),
            ["session_id", "session_label", "claimed_at", "task",
             "resources", "status"])

    def test_write_mode_materializes_coordinator_row_to_origin(self):
        self._tick()
        db.try_open_task_claim(
            self._conn, "s-coord2", "l", "t", [],
            claimed_at="2026-08-28T11:50:00Z")
        result = self._tick(mode="write")
        self.assertTrue(result["committed"])
        origin = json.loads(self._origin_text(writer.CLAIMS_REL_PATH))
        self.assertIn("s-coord2", [e["session_id"] for e in origin["claims"]])
        # The commit message carries the writer's prefix.
        log = _git(self._repo, "log", "-1", "--format=%s",
                   "origin/master").stdout.strip()
        self.assertTrue(log.startswith(writer.COMMIT_PREFIX))


class TestWriteModeStateChangeOnly(_Fixture):

    def test_matching_tick_commits_nothing(self):
        before = _git(self._repo, "rev-parse", "origin/master").stdout.strip()
        result = self._tick(mode="write")
        self.assertTrue(result["claims_match"] and result["chips_match"])
        self.assertFalse(result["committed"])
        _git(self._repo, "fetch", "-q", "origin")
        after = _git(self._repo, "rev-parse", "origin/master").stdout.strip()
        self.assertEqual(before, after)

    def test_second_write_tick_after_a_write_is_a_no_op(self):
        db.try_open_task_claim(self._conn, "s-w", "l", "t", [],
                               claimed_at="2026-08-28T11:55:00Z")
        # tick 1 has nothing in the DB until it ingests; open the claim
        # first, then tick twice.
        r1 = self._tick(mode="write")
        self.assertTrue(r1["committed"])
        r2 = self._tick(mode="write")
        self.assertFalse(r2["committed"])
        self.assertTrue(r2["claims_match"] and r2["chips_match"])


class TestCheckModeNeverWrites(_Fixture):

    CLAIMS = _claims_doc([
        _claim("s-done-old", "2026-08-25T09:00:00Z", status="done",
               closed_at="2026-08-25T10:00:00Z"),
    ])

    def test_mismatch_in_check_mode_moves_nothing(self):
        before = _git(self._repo, "rev-parse", "origin/master").stdout.strip()
        head_before = _git(self._repo, "rev-parse", "master").stdout.strip()
        result = self._tick(mode="check")
        self.assertFalse(result["claims_match"])  # retention would drop one
        self.assertFalse(result["committed"])
        _git(self._repo, "fetch", "-q", "origin")
        self.assertEqual(
            before,
            _git(self._repo, "rev-parse", "origin/master").stdout.strip())
        self.assertEqual(
            head_before,
            _git(self._repo, "rev-parse", "master").stdout.strip())
        # Working tree untouched (file still holds the aged entry).
        self.assertIn("s-done-old",
                      (self._repo / writer.CLAIMS_REL_PATH).read_text())


class TestSourceOrderFidelity(_Fixture):

    def test_client_side_reorder_is_followed_not_rewritten(self):
        self._tick()
        # A client-side rewrite reorders the chips file (merge machinery).
        doc = json.loads((self._repo / writer.CHIPS_REL_PATH).read_text())
        doc["chips"].reverse()
        (self._repo / writer.CHIPS_REL_PATH).write_text(_serialise(doc))
        _git(self._repo, "add", writer.CHIPS_REL_PATH)
        _git(self._repo, "commit", "-q", "-m", "chips: client reorder")
        _git(self._repo, "push", "-q", "origin", "master")
        result = self._tick(mode="write")
        # Order-faithful render: content identical, order follows the file,
        # so nothing to commit.
        self.assertTrue(result["chips_match"])
        self.assertFalse(result["committed"])


if __name__ == "__main__":
    unittest.main()
