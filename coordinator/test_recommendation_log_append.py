"""Contracts for the PHASE-4 RECOMMENDATION_LOG.jsonl append intake
(2026-08-29 fleet-wedge campaign endpoint batch): the
db.submit_recommendation_log_entry family, the /recommendation_log/append
endpoint handler, and task_claim_chip_git_writer.render_recommendation_log
+ its materialize_once integration.

The jsonl-trivial sibling of test_workspace_state_append.py -- same fixture
shape (bare remote + working clone), same pin ordering:

  1. BYTE PRESERVATION / APPEND-ONLY: the materializer only ever APPENDS
     pending records at end-of-file; origin's previous content is a strict
     prefix of the render (modulo a normalizing trailing newline).
  2. DUAL-WRITE SAFETY: client_git_write=1 records are never appended, only
     watched, and mark materialized once their line is seen in origin.
  3. DURABILITY ORDERING: appended records flip to materialized ONLY after
     the push carrying them succeeds; carried records flip in check mode too.
  4. WIRING: POST handler status codes + verdict passthrough; dispatch-table
     membership.

Time-independent. ASCII-only.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import app  # noqa: E402
import db  # noqa: E402
import task_claim_chip_git_writer as writer  # noqa: E402

NOW = "2026-08-29T12:00:00Z"

REC1 = json.dumps({"ts": "2026-08-29T10:00:00Z", "header": "Existing rec",
                   "selected": "opt-a"}, sort_keys=True)
REC2 = json.dumps({"ts": "2026-08-29T11:00:00Z", "header": "New rec",
                   "selected": "opt-b"}, sort_keys=True)
REC3 = json.dumps({"ts": "2026-08-29T11:30:00Z", "header": "Another new rec",
                   "selected": "opt-c"}, sort_keys=True)

SEED = REC1 + "\n"


def _git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=check)


def _bare_remote(parent, name="REE_Working.git"):
    remote = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _seed_repo(parent, remote, reclog_text=SEED, name="REE_Working"):
    repo = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)],
                   check=True)
    _git(repo, "config", "user.email", "writer@test")
    _git(repo, "config", "user.name", "writer-test")
    (repo / writer.CLAIMS_REL_PATH).write_text(json.dumps(
        {"claims": [], "schema_version": "v1", "stale_after_hours": 6},
        indent=2) + "\n")
    (repo / writer.CHIPS_REL_PATH).write_text(json.dumps(
        {"schema_version": "task_chips/v1", "chips": []}, indent=2) + "\n")
    paths = [writer.CLAIMS_REL_PATH, writer.CHIPS_REL_PATH]
    if reclog_text is not None:
        (repo / writer.RECLOG_REL_PATH).write_text(reclog_text,
                                                   encoding="utf-8")
        paths.append(writer.RECLOG_REL_PATH)
    _git(repo, "add", *paths)
    _git(repo, "commit", "-q", "-m", "seed")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "master")
    return repo


class _Fixture(unittest.TestCase):

    RECLOG_TEXT = SEED

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="reclog_append_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._remote = _bare_remote(self._tmp)
        self._repo = _seed_repo(self._tmp, self._remote,
                                reclog_text=self.RECLOG_TEXT)

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _tick(self, mode="check"):
        return writer.materialize_once(self._conn, str(self._repo),
                                       branch="master", mode=mode,
                                       now_iso=NOW)

    def _origin_reclog(self):
        return _git(self._repo, "show",
                    "origin/master:%s" % writer.RECLOG_REL_PATH).stdout

    def _submit(self, record, client_git_write=False):
        verdict, payload = db.submit_recommendation_log_entry(
            self._conn, record, session_id="sess-x",
            client_git_write=client_git_write, now=NOW)
        self.assertEqual(verdict, "ok")
        return payload["entry_id"]

    def _pending_ids(self):
        return [r["entry_id"]
                for r in db.pending_recommendation_log_entries(self._conn)]


class TestSubmitDb(_Fixture):

    def test_ok_then_idempotent(self):
        eid = self._submit(REC2)
        verdict, payload = db.submit_recommendation_log_entry(
            self._conn, REC2, now=NOW)
        self.assertEqual(verdict, "idempotent")
        self.assertEqual(payload["entry_id"], eid)
        self.assertEqual(self._pending_ids(), [eid])

    def test_empty_refused(self):
        for bad in ("", "   ", None):
            verdict, _ = db.submit_recommendation_log_entry(
                self._conn, bad, now=NOW)
            self.assertEqual(verdict, "empty")

    def test_bad_json_refused(self):
        for bad, why in (("not json at all", "unparseable"),
                         ('["a", "b"]', "non-object"),
                         ('{"a": 1}\n{"b": 2}', "multi-line")):
            verdict, _ = db.submit_recommendation_log_entry(
                self._conn, bad, now=NOW)
            self.assertEqual(verdict, "bad_json", why)
        self.assertEqual(self._pending_ids(), [])

    def test_whitespace_stripped_for_idempotency(self):
        eid = self._submit(REC2)
        verdict, payload = db.submit_recommendation_log_entry(
            self._conn, "  %s  " % REC2, now=NOW)
        self.assertEqual(verdict, "idempotent")
        self.assertEqual(payload["entry_id"], eid)


class TestCheckMode(_Fixture):

    def test_pending_append_does_not_commit_in_check_mode(self):
        self._submit(REC2)
        before = self._origin_reclog()
        result = self._tick(mode="check")
        self.assertEqual(result["recommendation_log"]["n_appended"], 1)
        self.assertFalse(result["committed"])
        self.assertEqual(self._origin_reclog(), before)
        self.assertEqual(len(self._pending_ids()), 1)

    def test_carried_record_marks_materialized_even_in_check_mode(self):
        eid = self._submit(REC1)  # already in the seed file
        result = self._tick(mode="check")
        self.assertEqual(result["recommendation_log"]["n_carried"], 1)
        self.assertNotIn(eid, self._pending_ids())


class TestWriteMode(_Fixture):

    def test_append_lands_on_origin_byte_preserving(self):
        e2 = self._submit(REC2)
        e3 = self._submit(REC3)
        before = self._origin_reclog()
        result = self._tick(mode="write")
        self.assertTrue(result["committed"])
        after = self._origin_reclog()
        self.assertTrue(after.startswith(before),
                        "append-only: origin's previous content must be a "
                        "strict prefix of the render")
        self.assertEqual(after, before + REC2 + "\n" + REC3 + "\n")
        self.assertEqual(self._pending_ids(), [],
                         "appended records flip to materialized after the "
                         "push succeeds")
        rows = self._conn.execute(
            "SELECT materialized_ref FROM recommendation_log_entries "
            "WHERE entry_id IN (?,?)", (e2, e3)).fetchall()
        for r in rows:
            self.assertTrue(r["materialized_ref"])

    def test_dual_write_record_is_watched_never_appended(self):
        eid = self._submit(REC2, client_git_write=True)
        result = self._tick(mode="write")
        self.assertEqual(result["recommendation_log"]["n_awaiting_client"], 1)
        self.assertNotIn(REC2, self._origin_reclog())
        self.assertIn(eid, self._pending_ids())
        # the client now lands its own line; the next tick marks it carried
        path = self._repo / writer.RECLOG_REL_PATH
        path.write_text(self._origin_reclog() + REC2 + "\n", encoding="utf-8")
        _git(self._repo, "add", writer.RECLOG_REL_PATH)
        _git(self._repo, "commit", "-q", "-m", "client lands its own record")
        _git(self._repo, "push", "-q", "origin", "master")
        result = self._tick(mode="check")
        self.assertEqual(result["recommendation_log"]["n_carried"], 1)
        self.assertNotIn(eid, self._pending_ids())


class TestNoTrailingNewline(_Fixture):

    RECLOG_TEXT = REC1  # seed WITHOUT trailing newline

    def test_append_never_fuses_records(self):
        self._submit(REC2)
        self._tick(mode="write")
        after = self._origin_reclog()
        lines = [ln for ln in after.splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        for ln in lines:
            json.loads(ln)  # every line must stay independently parseable


class TestMissingFile(_Fixture):

    RECLOG_TEXT = None  # repo predates the file

    def test_pending_rows_are_held_not_crashed(self):
        eid = self._submit(REC2)
        result = self._tick(mode="write")
        self.assertEqual(result["recommendation_log"]["n_appended"], 0)
        self.assertIn(eid, self._pending_ids())


class TestEndpointWiring(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="reclog_wire_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_dispatch_table_membership(self):
        self.assertIn("/recommendation_log/append",
                      app._TASK_CLAIM_CHIP_POST)
        self.assertIs(app._TASK_CLAIM_CHIP_POST["/recommendation_log/append"],
                      app._reclog_append)

    def test_handler_ok_and_verdict_passthrough(self):
        code, out = app._reclog_append(
            self._conn, {"record": REC2, "session_id": "s"}, "test-host")
        self.assertEqual(code, 200)
        self.assertEqual(out["verdict"], "ok")
        code, out = app._reclog_append(
            self._conn, {"record": REC2}, "test-host")
        self.assertEqual(code, 200)
        self.assertEqual(out["verdict"], "idempotent")

    def test_handler_rejects_bad_input(self):
        code, out = app._reclog_append(self._conn, {"record": ""}, "h")
        self.assertEqual(code, 400)
        self.assertEqual(out["verdict"], "empty")
        code, out = app._reclog_append(self._conn, {"record": "not json"}, "h")
        self.assertEqual(code, 400)
        self.assertEqual(out["verdict"], "bad_json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
