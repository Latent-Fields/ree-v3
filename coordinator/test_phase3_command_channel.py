"""Tests for the Phase 3 coordinator command channel.

Three test classes:

  CommandDBTest               -- db.insert_command / fetch_pending_commands /
                                 ack_command round-trip + idempotency + guards.
  CommandChannelHTTPTest      -- end-to-end via the real app.py entrypoint:
                                 POST /commands/issue, GET /commands,
                                 POST /commands/ack, validation.
  CoordinatorClientCommandTest -- the runner's coordinator_client.issue_command
                                 / fetch_commands / ack_command against live
                                 app.py (mirrors the runner import path).

All printed text is ASCII-only.
"""

import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db  # noqa: E402


# ---------------------------------------------------------------------------
# 1. DB layer
# ---------------------------------------------------------------------------

class CommandDBTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="command_db_")
        self._dbpath = os.path.join(self._tmp, "c.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)

    def tearDown(self):
        self._conn.close()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_insert_returns_row_with_id(self):
        row = db.insert_command(
            self._conn, "ree-cloud-2", "stop", json.dumps({}), "explorer")
        self.assertIsInstance(row["id"], int)
        self.assertEqual(row["machine"], "ree-cloud-2")
        self.assertEqual(row["kind"], "stop")
        self.assertIsNone(row["acked_at"])

    def test_fetch_pending_returns_unacked_in_order(self):
        db.insert_command(self._conn, "ree-cloud-2", "pause", None, "x")
        db.insert_command(self._conn, "ree-cloud-2", "resume", None, "x")
        db.insert_command(self._conn, "ree-cloud-3", "stop", None, "x")
        pend = db.fetch_pending_commands(self._conn, "ree-cloud-2")
        self.assertEqual([c["kind"] for c in pend], ["pause", "resume"])

    def test_ack_marks_acked_and_drops_from_pending(self):
        row = db.insert_command(self._conn, "ree-cloud-2", "stop", None, "x")
        ok, note = db.ack_command(
            self._conn, row["id"], "ree-cloud-2", "done", "drain requested")
        self.assertTrue(ok)
        self.assertEqual(note, "acked")
        self.assertEqual(db.fetch_pending_commands(self._conn, "ree-cloud-2"), [])
        stored = self._conn.execute(
            "SELECT acked_at, result_status, result_note FROM commands "
            "WHERE id=?", (row["id"],)).fetchone()
        self.assertIsNotNone(stored["acked_at"])
        self.assertEqual(stored["result_status"], "done")
        self.assertEqual(stored["result_note"], "drain requested")

    def test_ack_idempotent(self):
        row = db.insert_command(self._conn, "ree-cloud-2", "stop", None, "x")
        db.ack_command(self._conn, row["id"], "ree-cloud-2", "done", None)
        ok, note = db.ack_command(
            self._conn, row["id"], "ree-cloud-2", "done", None)
        self.assertTrue(ok)
        self.assertEqual(note, "already acked")

    def test_ack_owner_guard(self):
        row = db.insert_command(self._conn, "ree-cloud-2", "stop", None, "x")
        ok, note = db.ack_command(
            self._conn, row["id"], "ree-cloud-3", "done", None)
        self.assertFalse(ok)
        self.assertIn("belongs to", note)
        # Still pending for the rightful owner.
        self.assertEqual(
            len(db.fetch_pending_commands(self._conn, "ree-cloud-2")), 1)

    def test_ack_unknown_id(self):
        ok, note = db.ack_command(self._conn, 9999, "ree-cloud-2", "done", None)
        self.assertFalse(ok)
        self.assertIn("not found", note)

    def test_migrate_commands_adds_columns_on_legacy_table(self):
        # Simulate a pre-migration commands table (no result_* columns).
        tmp2 = tempfile.mkdtemp(prefix="legacy_cmd_")
        try:
            p = os.path.join(tmp2, "legacy.db")
            import sqlite3
            c = sqlite3.connect(p)
            c.execute(
                "CREATE TABLE commands (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "machine TEXT NOT NULL, kind TEXT NOT NULL, args TEXT, "
                "issued_by TEXT, issued_at TEXT NOT NULL, acked_at TEXT)")
            c.commit()
            c.close()
            conn = db.connect(p)  # runs _migrate_commands
            cols = {r[1] for r in conn.execute("PRAGMA table_info(commands)")}
            self.assertIn("result_status", cols)
            self.assertIn("result_note", cols)
            # And the new helpers work against the migrated table.
            row = db.insert_command(conn, "m", "stop", None, "x")
            ok, _ = db.ack_command(conn, row["id"], "m", "done", "n")
            self.assertTrue(ok)
            conn.close()
        finally:
            import shutil
            shutil.rmtree(tmp2, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. HTTP integration via the real app.py
# ---------------------------------------------------------------------------

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


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
        except (ValueError, UnicodeDecodeError, OSError):
            return e.code, None


class _LiveCoordinatorMixin:
    """Spin the real app.py entrypoint (shared by the HTTP + client tests).
    Mix in before unittest.TestCase; each subclass gets its own isolated
    coordinator process + DB."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="command_http_")
        cls._dbpath = os.path.join(cls._tmp, "c.db")
        cls._tokens = os.path.join(cls._tmp, "tokens.json")
        with open(cls._tokens, "w", encoding="utf-8") as fh:
            json.dump({"tok-op": "operator", "tok-c2": "ree-cloud-2"}, fh)
        cls._port = _free_port()
        env = dict(os.environ)
        env.update({
            "COORDINATOR_DB": cls._dbpath,
            "COORDINATOR_TOKENS_FILE": cls._tokens,
            "COORDINATOR_BIND_HOST": "127.0.0.1",
            "COORDINATOR_BIND_PORT": str(cls._port),
            "COORDINATOR_MODE": "coordinator",
        })
        cls._proc = subprocess.Popen(
            [sys.executable, str(HERE / "app.py")],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        cls._base = "http://127.0.0.1:%d" % cls._port
        import time
        for _ in range(50):
            try:
                st, _ = _http("GET", cls._base + "/health")
                if st == 200:
                    return
            except urllib.error.URLError:
                time.sleep(0.1)
        cls._proc.terminate()
        raise RuntimeError("coordinator did not come up")

    @classmethod
    def tearDownClass(cls):
        cls._proc.terminate()
        try:
            cls._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls._proc.kill()
        import shutil
        shutil.rmtree(cls._tmp, ignore_errors=True)


class CommandChannelHTTPTest(_LiveCoordinatorMixin, unittest.TestCase):

    def test_issue_requires_auth(self):
        st, _ = _http("POST", self._base + "/commands/issue",
                      body={"machine": "ree-cloud-2", "kind": "stop"})
        self.assertEqual(st, 401)

    def test_issue_fetch_ack_round_trip(self):
        st, jb = _http("POST", self._base + "/commands/issue", token="tok-op",
                       body={"machine": "ree-cloud-2", "kind": "stop",
                             "args": {}, "issued_by": "explorer"})
        self.assertEqual(st, 200)
        self.assertTrue(jb["ok"])
        cmd_id = jb["command"]["id"]

        st, jb = _http("GET",
                       self._base + "/commands?machine=ree-cloud-2",
                       token="tok-c2")
        self.assertEqual(st, 200)
        ids = [c["id"] for c in jb["commands"]]
        self.assertIn(cmd_id, ids)

        st, jb = _http("POST", self._base + "/commands/ack", token="tok-c2",
                       body={"id": cmd_id, "machine": "ree-cloud-2",
                             "result_status": "done",
                             "result_note": "drain requested"})
        self.assertEqual(st, 200)
        self.assertTrue(jb["ok"])

        # No longer pending.
        st, jb = _http("GET",
                       self._base + "/commands?machine=ree-cloud-2",
                       token="tok-c2")
        self.assertNotIn(cmd_id, [c["id"] for c in jb["commands"]])

    def test_issue_rejects_unknown_kind(self):
        st, jb = _http("POST", self._base + "/commands/issue", token="tok-op",
                       body={"machine": "ree-cloud-2", "kind": "explode"})
        self.assertEqual(st, 400)
        self.assertEqual(jb["error"], "unknown command kind")

    def test_issue_requires_machine(self):
        st, jb = _http("POST", self._base + "/commands/issue", token="tok-op",
                       body={"kind": "stop"})
        self.assertEqual(st, 400)
        self.assertEqual(jb["error"], "machine required")

    def test_issue_kick_requires_queue_id(self):
        st, jb = _http("POST", self._base + "/commands/issue", token="tok-op",
                       body={"machine": "ree-cloud-2", "kind": "kick"})
        self.assertEqual(st, 400)
        self.assertIn("queue_id", jb["error"])
        # With queue_id it succeeds.
        st, jb = _http("POST", self._base + "/commands/issue", token="tok-op",
                       body={"machine": "ree-cloud-2", "kind": "kick",
                             "args": {"queue_id": "V3-EXQ-999"}})
        self.assertEqual(st, 200)
        self.assertTrue(jb["ok"])

    def test_ack_requires_integer_id(self):
        st, jb = _http("POST", self._base + "/commands/ack", token="tok-c2",
                       body={"id": "not-an-int", "machine": "ree-cloud-2"})
        self.assertEqual(st, 400)

    def test_ack_owner_guard_409(self):
        st, jb = _http("POST", self._base + "/commands/issue", token="tok-op",
                       body={"machine": "ree-cloud-3", "kind": "pause"})
        cmd_id = jb["command"]["id"]
        # Wrong machine acking -> 409.
        st, jb = _http("POST", self._base + "/commands/ack", token="tok-c2",
                       body={"id": cmd_id, "machine": "ree-cloud-2"})
        self.assertEqual(st, 409)
        self.assertFalse(jb["ok"])


# ---------------------------------------------------------------------------
# 3. coordinator_client (runner import path)
# ---------------------------------------------------------------------------

class CoordinatorClientCommandTest(_LiveCoordinatorMixin, unittest.TestCase):

    def test_client_issue_fetch_ack_round_trip(self):
        import importlib
        root = HERE.parent
        sys.path.insert(0, str(root))
        try:
            os.environ["COORDINATION_MODE"] = "coordinator"
            os.environ["COORDINATOR_URL"] = self._base
            os.environ["COORDINATOR_TOKEN"] = "tok-c2"
            if "coordinator_client" in sys.modules:
                cc = importlib.reload(sys.modules["coordinator_client"])
            else:
                import coordinator_client as cc
            self.assertTrue(cc.enabled())

            r = cc.issue_command("ree-cloud-2", "pause", {}, "test")
            self.assertIsNotNone(r)
            self.assertTrue(r["ok"])
            cmd_id = r["command"]["id"]

            fetched = cc.fetch_commands("ree-cloud-2")
            self.assertIn(cmd_id, [c["id"] for c in fetched["commands"]])

            ack = cc.ack_command(cmd_id, "ree-cloud-2", "done", "paused")
            self.assertIsNotNone(ack)
            self.assertTrue(ack["ok"])

            fetched = cc.fetch_commands("ree-cloud-2")
            self.assertNotIn(cmd_id, [c["id"] for c in fetched["commands"]])
        finally:
            for k in ("COORDINATION_MODE", "COORDINATOR_URL",
                      "COORDINATOR_TOKEN"):
                os.environ.pop(k, None)
            if str(root) in sys.path:
                sys.path.remove(str(root))

    def test_client_disabled_in_git_mode(self):
        import importlib
        root = HERE.parent
        sys.path.insert(0, str(root))
        try:
            os.environ["COORDINATION_MODE"] = "git"
            os.environ.pop("COORDINATOR_URL", None)
            os.environ.pop("COORDINATOR_TOKEN", None)
            if "coordinator_client" in sys.modules:
                cc = importlib.reload(sys.modules["coordinator_client"])
            else:
                import coordinator_client as cc
            self.assertFalse(cc.enabled())
            self.assertIsNone(cc.issue_command("m", "stop"))
            self.assertIsNone(cc.ack_command(1, "m"))
            self.assertIsNone(cc.fetch_commands("m"))
        finally:
            os.environ.pop("COORDINATION_MODE", None)
            if str(root) in sys.path:
                sys.path.remove(str(root))


if __name__ == "__main__":
    unittest.main(verbosity=2)
