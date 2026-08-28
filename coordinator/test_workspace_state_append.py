"""Contracts for the PHASE-4 WORKSPACE_STATE.md append intake: the
db.submit_workspace_state_entry family, the /workspace_state/* endpoints,
and task_claim_chip_git_writer.render_workspace_state + its
materialize_once integration.

Pins, in descending order of damage-if-wrong:

  1. BYTE PRESERVATION: the materializer only ever SPLICES pending entries
     in at the structural insertion point -- removing the spliced block
     reproduces origin's previous file byte-for-byte, and the preamble
     (pinned block + ROTATE-INDEX) is untouched. This file has THREE
     confirmed silent-truncation incidents; the guards exist because of
     them, and the sabotage tests here prove the guards CAN fire (the real
     pipeline structurally cannot produce a failing case).
  2. DUAL-WRITE SAFETY: an entry submitted with client_git_write=1 is NEVER
     spliced (splicing would race the client's own push into a duplicate);
     it is watched for and marked materialized once its text is seen in
     origin's file.
  3. NO RESURRECTION: a materialized entry later rotated out of the file is
     never re-spliced -- materialized_at is a one-way flip, so rotation
     cannot be undone by the writer.
  4. DURABILITY ORDERING: spliced entries flip to materialized ONLY after
     the push carrying them succeeds; carried entries flip in check mode
     too (origin's text is the proof either way).
  5. WIRING: POST /workspace_state/append status codes + verdict
     passthrough; GET /workspace_state/pending shape; append-only (no other
     /workspace_state/* verb resolves).

Same real-git fixture shape as test_task_claim_chip_git_writer.py (bare
remote + working clone). Time-independent: retention never applies to WS
entries, and every timestamp below is an arbitrary fixed value with no
staleness semantics. ASCII-only.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import app  # noqa: E402
import db  # noqa: E402
import task_claim_chip_git_writer as writer  # noqa: E402

NOW = "2026-08-28T12:00:00Z"
TS1 = "2026-08-28T11:00:00Z"
TS2 = "2026-08-28T11:30:00Z"

PREAMBLE = (
    "<!-- PINNED (SHP-6): keep this block at the very top; prepend new "
    "session log entries BELOW it. -->\n"
    "> pinned pointer line\n"
    "<!-- ROTATE-INDEX:BEGIN -->\n"
    "> Months: [2026-08](docs/workspace_state_archive/2026-08.md)\n"
    "<!-- ROTATE-INDEX:END -->\n"
    "\n"
)
EXISTING_ENTRIES = (
    "## 2026-08-27T10:00:00Z -- existing entry one\n\n"
    "body paragraph of entry one\n\n"
    "## 2026-08-26T09:00:00Z -- existing entry two\n\n"
)
WS_SEED = PREAMBLE + EXISTING_ENTRIES


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=check,
    )


def _bare_remote(parent, name="REE_Working.git"):
    remote = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _claims_doc():
    return {"claims": [{
        "session_id": "s-active",
        "session_label": "L",
        "claimed_at": "2026-08-28T10:00:00Z",
        "task": "t",
        "resources": ["r.json"],
        "status": "active",
    }], "schema_version": "v1", "stale_after_hours": 6}


def _chips_doc():
    return {"schema_version": "task_chips/v1", "chips": []}


def _seed_repo(parent, remote, ws_text=WS_SEED, name="REE_Working"):
    repo = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)],
                   check=True)
    _git(repo, "config", "user.email", "writer@test")
    _git(repo, "config", "user.name", "writer-test")
    (repo / writer.CLAIMS_REL_PATH).write_text(
        json.dumps(_claims_doc(), indent=2) + "\n")
    (repo / writer.CHIPS_REL_PATH).write_text(
        json.dumps(_chips_doc(), indent=2) + "\n")
    paths = [writer.CLAIMS_REL_PATH, writer.CHIPS_REL_PATH]
    if ws_text is not None:
        (repo / writer.WORKSPACE_STATE_REL_PATH).write_text(
            ws_text, encoding="utf-8")
        paths.append(writer.WORKSPACE_STATE_REL_PATH)
    _git(repo, "add", *paths)
    _git(repo, "commit", "-q", "-m", "seed")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "master")
    return repo


class _Fixture(unittest.TestCase):

    WS_TEXT = WS_SEED

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="ws_append_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._remote = _bare_remote(self._tmp)
        self._repo = _seed_repo(self._tmp, self._remote, ws_text=self.WS_TEXT)

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _tick(self, mode="check"):
        return writer.materialize_once(
            self._conn, str(self._repo), branch="master", mode=mode,
            now_iso=NOW)

    def _origin_ws(self):
        return _git(self._repo, "show",
                    "origin/master:%s" % writer.WORKSPACE_STATE_REL_PATH
                    ).stdout

    def _submit(self, text, ts, client_git_write=False):
        verdict, payload = db.submit_workspace_state_entry(
            self._conn, text, ts=ts, session_id="sess-x",
            client_git_write=client_git_write, now=NOW)
        self.assertEqual(verdict, "ok")
        return payload["entry_id"]

    def _pending_ids(self):
        return [r["entry_id"]
                for r in db.pending_workspace_state_entries(self._conn)]


class TestSubmitDb(_Fixture):

    def test_verdicts(self):
        eid = self._submit("first entry", TS1)
        v, p = db.submit_workspace_state_entry(
            self._conn, "first entry", ts=TS1, now=NOW)
        self.assertEqual(v, "idempotent")
        self.assertEqual(p["entry_id"], eid)
        self.assertEqual(db.submit_workspace_state_entry(
            self._conn, "   \n", ts=TS1, now=NOW)[0], "empty_text")
        self.assertEqual(db.submit_workspace_state_entry(
            self._conn, "x", ts="28/08/2026 11:00", now=NOW)[0],
            "bad_timestamp")
        # ts defaults to `now` and text is newline-normalized.
        v, p = db.submit_workspace_state_entry(
            self._conn, "\n\nsecond entry\n", now=NOW)
        self.assertEqual(v, "ok")
        self.assertEqual(p["ts"], NOW)
        rows = db.pending_workspace_state_entries(self._conn)
        self.assertEqual([r["text"] for r in rows],
                         ["first entry", "second entry"])

    def test_pending_order_and_mark(self):
        e1 = self._submit("one", TS1)
        e2 = self._submit("two", TS2)
        self.assertEqual(self._pending_ids(), [e1, e2])
        db.mark_workspace_state_entries_materialized(
            self._conn, [e1], "sha-1", now=NOW)
        self.assertEqual(self._pending_ids(), [e2])
        # One-way flip: marking again with a different ref changes nothing.
        db.mark_workspace_state_entries_materialized(
            self._conn, [e1], "sha-2", now=NOW)
        row = self._conn.execute(
            "SELECT materialized_ref FROM workspace_state_entries "
            "WHERE entry_id=?", (e1,)).fetchone()
        self.assertEqual(row["materialized_ref"], "sha-1")


class TestGuards(_Fixture):
    """Direct sabotage of _ws_render_guards -- the real splice cannot
    produce these, which is exactly why they are tested synthetically
    (same convention as the client tool's own guard tests)."""

    def test_correct_splice_passes(self):
        block = writer._ws_format_entry(TS1, "new entry")
        pos = writer._ws_insertion_point(WS_SEED)
        new = WS_SEED[:pos] + block + WS_SEED[pos:]
        ok, reason = writer._ws_render_guards(WS_SEED, new, block, pos)
        self.assertTrue(ok, reason)

    def test_truncation_fails_conservation(self):
        block = writer._ws_format_entry(TS1, "new entry")
        pos = writer._ws_insertion_point(WS_SEED)
        # A stale-read style bug: the tail of the file is gone.
        new = WS_SEED[:pos] + block
        ok, reason = writer._ws_render_guards(WS_SEED, new, block, pos)
        self.assertFalse(ok)
        self.assertEqual(reason, "conservation")

    def test_wrong_size_fails(self):
        block = writer._ws_format_entry(TS1, "new entry")
        pos = writer._ws_insertion_point(WS_SEED)
        new = WS_SEED[:pos] + block + WS_SEED[pos:]
        ok, reason = writer._ws_render_guards(
            WS_SEED, new, block + "extra", pos)
        self.assertFalse(ok)
        # conservation trips first with a mis-declared block; both refuse.
        self.assertIn(reason, ("conservation", "size"))

    def test_heading_count_guard_fires_on_swallowed_heading(self):
        # Simulate a mangled splice that overwrote an existing heading's
        # '## ' prefix: conservation of the declared span still holds only
        # if we also declare the mangled text as the block, so entry_count
        # is the guard that catches it.
        block = writer._ws_format_entry(TS1, "new entry")
        mangled = WS_SEED.replace("## 2026-08-26T09:00:00Z", "-- swallowed",
                                  1)
        pos = writer._ws_insertion_point(mangled)
        new = mangled[:pos] + block + mangled[pos:]
        ok, reason = writer._ws_render_guards(mangled, new, block, pos)
        self.assertTrue(ok)  # against the mangled base it is a clean splice
        ok, reason = writer._ws_render_guards(WS_SEED, new, block, pos)
        self.assertFalse(ok)  # against the TRUE base it is refused

    def test_embedded_heading_in_entry_body_is_not_a_false_trip(self):
        block = writer._ws_format_entry(
            TS1, "entry whose body contains\n## a heading-looking line")
        pos = writer._ws_insertion_point(WS_SEED)
        new = WS_SEED[:pos] + block + WS_SEED[pos:]
        ok, reason = writer._ws_render_guards(WS_SEED, new, block, pos)
        self.assertTrue(ok, reason)

    def test_unreadable_file_holds_entries(self):
        self._submit("held entry", TS1)
        new_text, stats, carried, spliced = writer.render_workspace_state(
            self._conn, None)
        self.assertIsNone(new_text)
        self.assertEqual(spliced, [])
        self.assertEqual(carried, [])
        self.assertEqual(stats["n_spliced"], 0)
        self.assertEqual(self._pending_ids(), [self._pending_ids()[0]])


class TestCheckMode(_Fixture):

    def test_no_pending_is_a_clean_tick(self):
        result = self._tick()
        self.assertTrue(result["claims_match"])
        self.assertEqual(result["workspace_state"]["n_pending"], 0)
        self.assertFalse(result["committed"])

    def test_would_splice_reported_nothing_written(self):
        eid = self._submit("new closing entry", TS1)
        before = self._origin_ws()
        result = self._tick(mode="check")
        self.assertEqual(result["workspace_state"]["n_spliced"], 1)
        self.assertFalse(result["committed"])
        self.assertEqual(self._origin_ws(), before)
        self.assertEqual(self._pending_ids(), [eid])

    def test_carried_entry_marked_in_check_mode(self):
        # The formatted text is already in the seeded file: proof enough.
        self._submit("existing entry one\n\nbody paragraph of entry one",
                     "2026-08-27T10:00:00Z")
        result = self._tick(mode="check")
        self.assertEqual(result["workspace_state"]["n_carried"], 1)
        self.assertEqual(result["workspace_state"]["n_spliced"], 0)
        self.assertEqual(self._pending_ids(), [])
        self.assertFalse(result["committed"])

    def test_awaiting_client_never_spliced_then_marked_when_seen(self):
        eid = self._submit("dual-write entry", TS1, client_git_write=True)
        result = self._tick(mode="write")
        self.assertEqual(result["workspace_state"]["n_awaiting_client"], 1)
        self.assertEqual(result["workspace_state"]["n_spliced"], 0)
        self.assertFalse(result["committed"])
        self.assertEqual(self._pending_ids(), [eid])
        # Now the client's own append lands on origin...
        block = writer._ws_format_entry(TS1, "dual-write entry")
        ws_path = self._repo / writer.WORKSPACE_STATE_REL_PATH
        text = ws_path.read_text()
        pos = writer._ws_insertion_point(text)
        ws_path.write_text(text[:pos] + block + text[pos:])
        _git(self._repo, "add", writer.WORKSPACE_STATE_REL_PATH)
        _git(self._repo, "commit", "-q", "-m", "client append")
        _git(self._repo, "push", "-q", "origin", "master")
        # ...and the next tick marks it materialized without writing.
        result = self._tick(mode="check")
        self.assertEqual(result["workspace_state"]["n_carried"], 1)
        self.assertEqual(self._pending_ids(), [])


class TestWriteMode(_Fixture):

    def test_splice_lands_byte_preserving_and_marks_after_push(self):
        before = self._origin_ws()
        e1 = self._submit("older new entry", TS1)
        e2 = self._submit("newer new entry", TS2)
        result = self._tick(mode="write")
        self.assertTrue(result["committed"])
        self.assertEqual(result["workspace_state"]["n_spliced"], 2)
        after = self._origin_ws()
        # Byte preservation: removing the spliced block reproduces `before`.
        block = (writer._ws_format_entry(TS2, "newer new entry")
                 + writer._ws_format_entry(TS1, "older new entry"))
        pos = writer._ws_insertion_point(before)
        self.assertEqual(after, before[:pos] + block + before[pos:])
        # Newest-first: the newer entry is the file's first section.
        self.assertLess(after.index("newer new entry"),
                        after.index("older new entry"))
        # Preamble bytes untouched.
        self.assertTrue(after.startswith(PREAMBLE))
        # Marked materialized with the pushed commit.
        self.assertEqual(self._pending_ids(), [])
        head = _git(self._repo, "rev-parse",
                    "origin/master").stdout.strip()
        for eid in (e1, e2):
            row = self._conn.execute(
                "SELECT materialized_ref FROM workspace_state_entries "
                "WHERE entry_id=?", (eid,)).fetchone()
            self.assertEqual(row["materialized_ref"], head)
        # Second tick: nothing to do, no liveness-tick commit.
        result = self._tick(mode="write")
        self.assertFalse(result["committed"])
        self.assertEqual(result["workspace_state"]["n_pending"], 0)

    def test_commit_message_names_ws(self):
        self._submit("message check entry", TS1)
        result = self._tick(mode="write")
        self.assertTrue(result["committed"])
        msg = _git(self._repo, "log", "-1", "--format=%s",
                   "origin/master").stdout.strip()
        self.assertIn("ws +1", msg)
        self.assertIn(writer.WORKSPACE_STATE_REL_PATH, msg)

    def test_no_resurrection_after_rotation(self):
        self._submit("rotates away", TS1)
        self._tick(mode="write")
        self.assertEqual(self._pending_ids(), [])
        # Simulate rotation: the entry leaves the head file.
        ws_path = self._repo / writer.WORKSPACE_STATE_REL_PATH
        text = ws_path.read_text()
        block = writer._ws_format_entry(TS1, "rotates away")
        self.assertIn(block, text)
        ws_path.write_text(text.replace(block, "", 1))
        _git(self._repo, "add", writer.WORKSPACE_STATE_REL_PATH)
        _git(self._repo, "commit", "-q", "-m", "rotate")
        _git(self._repo, "push", "-q", "origin", "master")
        result = self._tick(mode="write")
        self.assertFalse(result["committed"])
        self.assertNotIn("rotates away", self._origin_ws())

    def test_ws_only_change_still_commits(self):
        # Registries match; only WS differs -- the tick must still write.
        self._submit("lone ws entry", TS1)
        result = self._tick(mode="write")
        self.assertTrue(result["claims_match"])
        self.assertTrue(result["chips_match"])
        self.assertTrue(result["committed"])
        self.assertIn("lone ws entry", self._origin_ws())


class TestMissingFile(_Fixture):

    WS_TEXT = None  # repo seeded WITHOUT WORKSPACE_STATE.md

    def test_absent_file_holds_entries_and_does_not_crash(self):
        eid = self._submit("stranded entry", TS1)
        result = self._tick(mode="write")
        self.assertFalse(result["committed"])
        self.assertEqual(result["workspace_state"]["n_spliced"], 0)
        self.assertEqual(self._pending_ids(), [eid])


TOKEN = "ws-test-token"
MACHINE = "ree-cloud-9"


def _http(method, url, token=None, body=None):
    headers = {}
    data = None
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except (ValueError, OSError):
            return e.code, None


class TestEndpointWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="ws_append_ep_")
        cls._dbpath = os.path.join(cls._tmp, "coord.db")
        db.init_db(cls._dbpath)
        cls._prev_db_path = app.DB_PATH
        cls._prev_tokens = app._tokens
        app.DB_PATH = cls._dbpath
        app._tokens = {TOKEN: MACHINE}
        cls._srv = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls._port = cls._srv.server_address[1]
        cls._thread = threading.Thread(target=cls._srv.serve_forever,
                                       daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls):
        cls._srv.shutdown()
        cls._srv.server_close()
        cls._thread.join(timeout=5)
        app.DB_PATH = cls._prev_db_path
        app._tokens = cls._prev_tokens
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _url(self, path):
        return "http://127.0.0.1:%d%s" % (self._port, path)

    def test_append_and_pending_round_trip(self):
        code, out = _http("POST", self._url("/workspace_state/append"),
                          token=TOKEN,
                          body={"text": "endpoint entry", "timestamp": TS1,
                                "session_id": "sess-ep",
                                "client_git_write": True})
        self.assertEqual(code, 200)
        self.assertEqual(out["verdict"], "ok")
        entry_id = out["entry_id"]
        # Idempotent retry.
        code, out = _http("POST", self._url("/workspace_state/append"),
                          token=TOKEN,
                          body={"text": "endpoint entry", "timestamp": TS1})
        self.assertEqual(code, 200)
        self.assertEqual(out["verdict"], "idempotent")
        self.assertEqual(out["entry_id"], entry_id)
        # Pending readout carries the submitted host from the bearer token.
        code, out = _http("GET", self._url("/workspace_state/pending"),
                          token=TOKEN)
        self.assertEqual(code, 200)
        self.assertEqual(out["n_pending"], 1)
        self.assertEqual(out["n_awaiting_client"], 1)
        self.assertEqual(out["pending"][0]["entry_id"], entry_id)
        self.assertEqual(out["pending"][0]["session_id"], "sess-ep")
        self.assertTrue(out["pending"][0]["client_git_write"])

    def test_bad_requests(self):
        code, out = _http("POST", self._url("/workspace_state/append"),
                          token=TOKEN, body={"text": "   "})
        self.assertEqual(code, 400)
        code, out = _http("POST", self._url("/workspace_state/append"),
                          token=TOKEN,
                          body={"text": "x", "timestamp": "yesterday"})
        self.assertEqual(code, 400)
        self.assertEqual(out["verdict"], "bad_timestamp")

    def test_auth_required_both_verbs(self):
        code, _ = _http("POST", self._url("/workspace_state/append"),
                        body={"text": "x"})
        self.assertEqual(code, 401)
        code, _ = _http("GET", self._url("/workspace_state/pending"))
        self.assertEqual(code, 401)

    def test_append_only_no_other_verb_resolves(self):
        for path in ("/workspace_state/edit", "/workspace_state/delete",
                     "/workspace_state/rotate"):
            code, _ = _http("POST", self._url(path), token=TOKEN,
                            body={"entry_id": 1})
            self.assertEqual(code, 404, path)


if __name__ == "__main__":
    unittest.main()
