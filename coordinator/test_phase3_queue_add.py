"""Tests for the Phase 3 POST /queue/add coordinator ingress (audit rec R7).

Under Phase 3 the DB owns the queue and experiment_queue.json is a derived
view materialised by phase3_queue_writer. A producer (e.g. a headless IGW
/queue-experiment session) that only `git commit`s the queue file can lose
its addition to the snapshot-writer race. POST /queue/add upserts the item
straight into the experiments table so it survives re-materialisation.

Four test classes:

  QueueAddDBTest          -- db.get_queue_status + db.upsert_experiment land a
                             new row that _materialise_queue_from_db emits
                             (the survives-re-materialisation property).
  QueueAddHTTPTest        -- end-to-end via the real app.py entrypoint:
                             POST /queue/add validation + add + terminal-id
                             refusal + force_rerun + shadow-mode no-op.
  QueueAddClientTest      -- coordinator_client.add_queue_item against live
                             app.py (mirrors the producer import path).

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
import sync_daemon  # noqa: E402


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


def _item(qid="V3-EXQ-999", **over):
    it = {
        "queue_id": qid,
        "title": "queue-add test item",
        "experiment_type": "v3_exq_999_queue_add_test",
        "script": "experiments/v3_exq_999_queue_add_test.py",
        "priority": 50,
        "machine_affinity": "any",
        "status": "pending",
        "estimated_minutes": 5,
    }
    it.update(over)
    return it


# ---------------------------------------------------------------------------
# 1. DB layer + survives-re-materialisation
# ---------------------------------------------------------------------------

class QueueAddDBTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="queue_add_db_")
        self._dbpath = os.path.join(self._tmp, "c.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)

    def tearDown(self):
        self._conn.close()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_get_queue_status_absent_then_present(self):
        self.assertIsNone(db.get_queue_status(self._conn, "V3-EXQ-999"))
        db.upsert_experiment(self._conn, _item(), preserve_claim=True)
        self.assertEqual(db.get_queue_status(self._conn, "V3-EXQ-999"),
                         "pending")

    def test_added_row_is_materialised_into_queue_view(self):
        # The core property: an item upserted into the experiments table is
        # emitted by _materialise_queue_from_db (what phase3_queue_writer
        # writes back to experiment_queue.json). A git-only add that the
        # writer race erased would NOT appear here; a DB add does.
        db.upsert_experiment(self._conn, _item(), preserve_claim=True)
        view = sync_daemon._materialise_queue_from_db(self._conn, {})
        qids = [it["queue_id"] for it in view["items"]]
        self.assertIn("V3-EXQ-999", qids)
        row = next(it for it in view["items"]
                   if it["queue_id"] == "V3-EXQ-999")
        self.assertEqual(row["script"],
                         "experiments/v3_exq_999_queue_add_test.py")
        self.assertEqual(row["status"], "pending")

    def test_terminal_row_excluded_from_view(self):
        db.upsert_experiment(
            self._conn, _item(status="completed"), preserve_claim=False)
        self.assertEqual(db.get_queue_status(self._conn, "V3-EXQ-999"),
                         "completed")
        view = sync_daemon._materialise_queue_from_db(self._conn, {})
        self.assertNotIn("V3-EXQ-999",
                         [it["queue_id"] for it in view["items"]])


# ---------------------------------------------------------------------------
# shared live-coordinator harness (coordinator mode)
# ---------------------------------------------------------------------------

class _LiveCoordinatorMixin:

    MODE = "coordinator"

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="queue_add_http_")
        cls._dbpath = os.path.join(cls._tmp, "c.db")
        cls._tokens = os.path.join(cls._tmp, "tokens.json")
        with open(cls._tokens, "w", encoding="utf-8") as fh:
            json.dump({"tok-op": "operator"}, fh)
        cls._port = _free_port()
        env = dict(os.environ)
        env.update({
            "COORDINATOR_DB": cls._dbpath,
            "COORDINATOR_TOKENS_FILE": cls._tokens,
            "COORDINATOR_BIND_HOST": "127.0.0.1",
            "COORDINATOR_BIND_PORT": str(cls._port),
            "COORDINATOR_MODE": cls.MODE,
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


# ---------------------------------------------------------------------------
# 2. HTTP endpoint (coordinator mode)
# ---------------------------------------------------------------------------

class QueueAddHTTPTest(_LiveCoordinatorMixin, unittest.TestCase):

    def test_add_requires_auth(self):
        st, _ = _http("POST", self._base + "/queue/add",
                      body={"item": _item("V3-EXQ-AUTH")})
        self.assertEqual(st, 401)

    def test_add_requires_item_with_queue_id(self):
        st, jb = _http("POST", self._base + "/queue/add", token="tok-op",
                       body={"item": {"script": "x.py"}})
        self.assertEqual(st, 400)

    def test_add_requires_script(self):
        st, jb = _http("POST", self._base + "/queue/add", token="tok-op",
                       body={"item": {"queue_id": "V3-EXQ-NOSCRIPT"}})
        self.assertEqual(st, 400)

    def test_add_then_present_in_queue_active(self):
        st, jb = _http("POST", self._base + "/queue/add", token="tok-op",
                       body={"item": _item("V3-EXQ-ADD1")})
        self.assertEqual(st, 200)
        self.assertTrue(jb["ok"])
        self.assertTrue(jb["applied"])
        self.assertFalse(jb["existed"])

        st, jb = _http("GET", self._base + "/queue/active", token="tok-op")
        self.assertEqual(st, 200)
        items = jb if isinstance(jb, list) else jb.get("items", [])
        qids = [(it.get("queue_id") if isinstance(it, dict) else it)
                for it in items]
        self.assertIn("V3-EXQ-ADD1", qids)

    def test_bare_item_body_accepted(self):
        st, jb = _http("POST", self._base + "/queue/add", token="tok-op",
                       body=_item("V3-EXQ-BARE"))
        self.assertEqual(st, 200)
        self.assertTrue(jb["applied"])

    def test_status_forced_pending_not_caller_injected(self):
        # A caller cannot inject a claimed/completed state via add.
        st, jb = _http("POST", self._base + "/queue/add", token="tok-op",
                       body={"item": _item("V3-EXQ-INJECT",
                                           status="completed",
                                           claimed_by={"machine": "evil"})})
        self.assertEqual(st, 200)
        self.assertTrue(jb["applied"])
        conn = db.connect(self._dbpath)
        try:
            self.assertEqual(
                db.get_queue_status(conn, "V3-EXQ-INJECT"), "pending")
        finally:
            conn.close()

    def test_terminal_id_refused_without_force(self):
        conn = db.connect(self._dbpath)
        try:
            db.upsert_experiment(
                conn, _item("V3-EXQ-DONE", status="completed"),
                preserve_claim=False)
        finally:
            conn.close()
        st, jb = _http("POST", self._base + "/queue/add", token="tok-op",
                       body={"item": _item("V3-EXQ-DONE")})
        self.assertEqual(st, 409)
        self.assertFalse(jb["applied"])
        self.assertEqual(jb["existing_status"], "completed")

    def test_terminal_id_allowed_with_force_rerun(self):
        conn = db.connect(self._dbpath)
        try:
            db.upsert_experiment(
                conn, _item("V3-EXQ-FORCE", status="failed"),
                preserve_claim=False)
        finally:
            conn.close()
        st, jb = _http("POST", self._base + "/queue/add", token="tok-op",
                       body={"item": _item("V3-EXQ-FORCE", force_rerun=True)})
        self.assertEqual(st, 200)
        self.assertTrue(jb["applied"])
        conn = db.connect(self._dbpath)
        try:
            self.assertEqual(
                db.get_queue_status(conn, "V3-EXQ-FORCE"), "pending")
        finally:
            conn.close()


class QueueAddShadowModeTest(_LiveCoordinatorMixin, unittest.TestCase):

    MODE = "shadow"

    def test_shadow_mode_does_not_mutate_db(self):
        st, jb = _http("POST", self._base + "/queue/add", token="tok-op",
                       body={"item": _item("V3-EXQ-SHADOW")})
        self.assertEqual(st, 200)
        self.assertFalse(jb["applied"])
        conn = db.connect(self._dbpath)
        try:
            self.assertIsNone(
                db.get_queue_status(conn, "V3-EXQ-SHADOW"))
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# 3. coordinator_client (producer import path)
# ---------------------------------------------------------------------------

class QueueAddClientTest(_LiveCoordinatorMixin, unittest.TestCase):

    def _client(self):
        # Fresh import of coordinator_client with env pointed at the live
        # coordinator (it reads URL/TOKEN/MODE at import time).
        os.environ["COORDINATOR_URL"] = self._base
        os.environ["COORDINATOR_TOKEN"] = "tok-op"
        os.environ["COORDINATION_MODE"] = "coordinator"
        sys.path.insert(0, str(HERE.parent))
        import importlib
        import coordinator_client
        importlib.reload(coordinator_client)
        return coordinator_client

    def test_client_add_queue_item(self):
        cc = self._client()
        self.assertTrue(cc.enabled())
        r = cc.add_queue_item(_item("V3-EXQ-CLIENT"))
        self.assertIsNotNone(r)
        self.assertTrue(r["ok"])
        self.assertTrue(r["applied"])

        st, jb = _http("GET", self._base + "/queue/active", token="tok-op")
        items = jb if isinstance(jb, list) else jb.get("items", [])
        qids = [(it.get("queue_id") if isinstance(it, dict) else it)
                for it in items]
        self.assertIn("V3-EXQ-CLIENT", qids)


if __name__ == "__main__":
    unittest.main()
