"""Contracts for the PHASE-4 igw_routine_log.md append intake (2026-09-01,
git-traffic-simplification sweep lane 2): the db.submit_igw_log_entry
family, the /igw_log/append endpoint handler, and
ree_assembly_git_writer.render_igw_log + its materialize_once integration.

The plain-text sibling of test_recommendation_log_append.py -- same fixture
shape (bare remote + working clone, but of REE_assembly, not REE_Working),
same pin ordering:

  1. BYTE PRESERVATION / APPEND-ONLY: the writer only ever APPENDS pending
     lines at end-of-file; origin's previous content is a strict prefix of
     the render (modulo a normalizing trailing newline).
  2. DUAL-WRITE SAFETY: client_git_write=1 lines are never appended, only
     watched, and mark materialized once their text is seen in origin.
  3. DURABILITY ORDERING: appended lines flip to materialized ONLY after the
     push carrying them succeeds; carried lines flip in check mode too.
  4. WIRING: POST handler status codes + verdict passthrough; dispatch-table
     membership.

Time-independent. ASCII-only.
"""

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
import ree_assembly_git_writer as writer  # noqa: E402

NOW = "2026-09-01T12:00:00Z"

LINE1 = "2026-09-01T10:00:00Z tick: existing heartbeat line"
LINE2 = "2026-09-01T11:00:00Z tick: new heartbeat line"
LINE3 = "2026-09-01T11:30:00Z tick: another new heartbeat line"

SEED = "# IGW Auto-Spawn Routine Log\n\nOne line per hourly tick. ASCII only.\n\n" + LINE1 + "\n"


def _git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=check)


def _bare_remote(parent, name="REE_assembly.git"):
    remote = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _seed_repo(parent, remote, log_text=SEED, name="REE_assembly"):
    repo = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)],
                   check=True)
    _git(repo, "config", "user.email", "writer@test")
    _git(repo, "config", "user.name", "writer-test")
    paths = []
    if log_text is not None:
        log_path = repo / writer.IGW_LOG_REL_PATH
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(log_text, encoding="utf-8")
        paths.append(writer.IGW_LOG_REL_PATH)
    if paths:
        _git(repo, "add", *paths)
    else:
        # Need at least one commit for origin/master to resolve.
        (repo / ".keep").write_text("")
        _git(repo, "add", ".keep")
    _git(repo, "commit", "-q", "-m", "seed")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "master")
    return repo


class _Fixture(unittest.TestCase):

    LOG_TEXT = SEED

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="igw_log_append_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._remote = _bare_remote(self._tmp)
        self._repo = _seed_repo(self._tmp, self._remote, log_text=self.LOG_TEXT)

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _tick(self, mode="check"):
        return writer.materialize_once(self._conn, str(self._repo),
                                       branch="master", mode=mode)

    def _origin_log(self):
        return _git(self._repo, "show",
                    "origin/master:%s" % writer.IGW_LOG_REL_PATH).stdout

    def _submit(self, line, client_git_write=False):
        verdict, payload = db.submit_igw_log_entry(
            self._conn, line, session_id="igw_routine_tick",
            client_git_write=client_git_write, now=NOW)
        self.assertEqual(verdict, "ok")
        return payload["entry_id"]

    def _pending_ids(self):
        return [r["entry_id"] for r in db.pending_igw_log_entries(self._conn)]


class TestSubmitDb(_Fixture):

    def test_ok_then_idempotent(self):
        eid = self._submit(LINE2)
        verdict, payload = db.submit_igw_log_entry(self._conn, LINE2, now=NOW)
        self.assertEqual(verdict, "idempotent")
        self.assertEqual(payload["entry_id"], eid)
        self.assertEqual(self._pending_ids(), [eid])

    def test_empty_refused(self):
        for bad in ("", "   ", None):
            verdict, _ = db.submit_igw_log_entry(self._conn, bad, now=NOW)
            self.assertEqual(verdict, "empty")

    def test_multiline_refused(self):
        verdict, _ = db.submit_igw_log_entry(
            self._conn, "line one\nline two", now=NOW)
        self.assertEqual(verdict, "multiline")
        self.assertEqual(self._pending_ids(), [])

    def test_whitespace_stripped_for_idempotency(self):
        eid = self._submit(LINE2)
        verdict, payload = db.submit_igw_log_entry(
            self._conn, "  %s  " % LINE2, now=NOW)
        self.assertEqual(verdict, "idempotent")
        self.assertEqual(payload["entry_id"], eid)


class TestCheckMode(_Fixture):

    def test_pending_append_does_not_commit_in_check_mode(self):
        self._submit(LINE2)
        before = self._origin_log()
        result = self._tick(mode="check")
        self.assertEqual(result["igw_log"]["n_appended"], 1)
        self.assertFalse(result["committed"])
        self.assertEqual(self._origin_log(), before)
        self.assertEqual(len(self._pending_ids()), 1)

    def test_carried_line_marks_materialized_even_in_check_mode(self):
        eid = self._submit(LINE1)  # already in the seed file
        result = self._tick(mode="check")
        self.assertEqual(result["igw_log"]["n_carried"], 1)
        self.assertNotIn(eid, self._pending_ids())


class TestWriteMode(_Fixture):

    def test_append_lands_on_origin_byte_preserving(self):
        e2 = self._submit(LINE2)
        e3 = self._submit(LINE3)
        before = self._origin_log()
        result = self._tick(mode="write")
        self.assertTrue(result["committed"])
        after = self._origin_log()
        self.assertTrue(after.startswith(before),
                        "append-only: origin's previous content must be a "
                        "strict prefix of the render")
        self.assertEqual(after, before + LINE2 + "\n" + LINE3 + "\n")
        self.assertEqual(self._pending_ids(), [],
                         "appended lines flip to materialized after the "
                         "push succeeds")
        rows = self._conn.execute(
            "SELECT materialized_ref FROM igw_log_entries "
            "WHERE entry_id IN (?,?)", (e2, e3)).fetchall()
        for r in rows:
            self.assertTrue(r["materialized_ref"])

    def test_dual_write_line_is_watched_never_appended(self):
        eid = self._submit(LINE2, client_git_write=True)
        result = self._tick(mode="write")
        self.assertEqual(result["igw_log"]["n_awaiting_client"], 1)
        self.assertNotIn(LINE2, self._origin_log())
        self.assertIn(eid, self._pending_ids())
        # the client now lands its own line; the next tick marks it carried
        path = self._repo / writer.IGW_LOG_REL_PATH
        path.write_text(self._origin_log() + LINE2 + "\n", encoding="utf-8")
        _git(self._repo, "add", writer.IGW_LOG_REL_PATH)
        _git(self._repo, "commit", "-q", "-m", "client lands its own line")
        _git(self._repo, "push", "-q", "origin", "master")
        result = self._tick(mode="check")
        self.assertEqual(result["igw_log"]["n_carried"], 1)
        self.assertNotIn(eid, self._pending_ids())

    def test_commit_message_carries_prefix(self):
        self._submit(LINE2)
        self._tick(mode="write")
        log = _git(self._repo, "log", "-1", "--format=%s")
        self.assertTrue(log.stdout.startswith(writer.COMMIT_PREFIX),
                        log.stdout)


class TestNoTrailingNewline(_Fixture):

    LOG_TEXT = LINE1  # seed WITHOUT trailing newline

    def test_append_never_fuses_lines(self):
        self._submit(LINE2)
        self._tick(mode="write")
        after = self._origin_log()
        lines = [ln for ln in after.splitlines() if ln.strip()]
        self.assertEqual(lines, [LINE1, LINE2])


class TestMissingFile(_Fixture):

    LOG_TEXT = None  # repo predates the file

    def test_pending_rows_are_held_not_crashed(self):
        eid = self._submit(LINE2)
        result = self._tick(mode="write")
        self.assertEqual(result["igw_log"]["n_appended"], 0)
        self.assertIn(eid, self._pending_ids())


class TestEndpointWiring(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="igw_log_wire_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_dispatch_table_membership(self):
        self.assertIn("/igw_log/append", app._TASK_CLAIM_CHIP_POST)
        self.assertIs(app._TASK_CLAIM_CHIP_POST["/igw_log/append"],
                      app._igw_log_append)

    def test_handler_ok_and_verdict_passthrough(self):
        code, out = app._igw_log_append(
            self._conn, {"line": LINE2, "session_id": "s"}, "test-host")
        self.assertEqual(code, 200)
        self.assertEqual(out["verdict"], "ok")
        code, out = app._igw_log_append(
            self._conn, {"line": LINE2}, "test-host")
        self.assertEqual(code, 200)
        self.assertEqual(out["verdict"], "idempotent")

    def test_handler_rejects_bad_input(self):
        code, out = app._igw_log_append(self._conn, {"line": ""}, "h")
        self.assertEqual(code, 400)
        self.assertEqual(out["verdict"], "empty")
        code, out = app._igw_log_append(
            self._conn, {"line": "a\nb"}, "h")
        self.assertEqual(code, 400)
        self.assertEqual(out["verdict"], "multiline")


if __name__ == "__main__":
    unittest.main(verbosity=2)
