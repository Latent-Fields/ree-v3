"""Contracts for the PHASE-1 read-only observability endpoints added to
app.py: GET /task_claim/list, GET /task_claim/check, GET /chip/list,
GET /task_claim/drift.

These are READ-ONLY by design (plan doc D10, and the PHASE-1 scope note in
task_claim_chip_shadow_sync.py's module docstring) -- there is deliberately
no mutating /task_claim/* or /chip/* endpoint yet. These tests pin that the
new GET routes work AND that every mutating verb keeps 404ing, so a future
PHASE-2 session cannot silently assume a write path exists here already.

Runs the real app.Handler in-process via ThreadingHTTPServer on an
ephemeral port (no subprocess -- faster and avoids the token-file/env-var
bootstrap subprocess dance test_shadow_e2e.py uses for the full-process
smoke test). ASCII-only.
"""

import json
import os
import pathlib
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

TOKEN = "test-token"
MACHINE = "ree-cloud-4"


def _http(method, url, token=None, body=None):
    headers = {}
    data = None
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except (ValueError, OSError):
            return e.code, None


class TestEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="task_claim_chip_endpoints_")
        cls._dbpath = os.path.join(cls._tmp, "coord.db")
        db.init_db(cls._dbpath)

        # Monkeypatch app.py's module-level config instead of a subprocess +
        # env-var bootstrap: these globals are read once at import time, so
        # reassigning them here (before starting the server) configures the
        # real Handler class against a throwaway DB and an in-memory token.
        app.DB_PATH = cls._dbpath
        app._tokens = {TOKEN: MACHINE}

        conn = db.connect(cls._dbpath)
        try:
            db.reconcile_task_claims(conn, [
                {"session_id": "s1", "claimed_at": "2026-08-26T10:00:00Z",
                 "session_label": "l", "task": "t1",
                 "resources": ["a/b.json"], "status": "active"},
                {"session_id": "s2", "claimed_at": "2026-08-26T11:00:00Z",
                 "session_label": "l2", "task": "t2", "resources": [],
                 "status": "done", "closed_at": "2026-08-26T12:00:00Z",
                 "completion_note": "done"},
            ])
            db.reconcile_chips(conn, [
                {"chip_ref": "chip-open-1", "task_id": None,
                 "origin": "headless", "kind": "work", "urgency": False,
                 "session_id": "unknown", "session_label": "", "title": "t",
                 "tldr": "td", "prompt": "[chip_ref: chip-open-1]\n\nbody",
                 "cwd": "/x", "spawned_at": "2026-08-26T09:00:00Z",
                 "origin_host": "DLAPTOP", "status": "open",
                 "claimed_by": None, "claimed_at": None, "claim_note": None,
                 "claimed_host": None, "resolved_at": None,
                 "resolved_by_session_id": None, "resolution_note": None,
                 "resolution_note_auto": False,
                 "attached_by_session_id": None, "attached_at": None},
                {"chip_ref": "chip-done-1", "task_id": None,
                 "origin": "hygiene_tick", "kind": "work", "urgency": False,
                 "session_id": "s3", "session_label": "", "title": "t2",
                 "tldr": "td2", "prompt": "[chip_ref: chip-done-1]\n\nbody",
                 "cwd": "/x", "spawned_at": "2026-08-26T08:00:00Z",
                 "origin_host": "DLAPTOP", "status": "done",
                 "claimed_by": "s3", "claimed_at": "2026-08-26T08:30:00Z",
                 "claim_note": None, "claimed_host": "DLAPTOP",
                 "resolved_at": "2026-08-26T09:00:00Z",
                 "resolved_by_session_id": "s3", "resolution_note": "ok",
                 "resolution_note_auto": False,
                 "attached_by_session_id": None, "attached_at": None},
            ])
            claim_stats = db.reconcile_task_claims(conn, [])
            chip_stats = db.reconcile_chips(conn, [])
            db.log_task_claim_chip_drift(conn, "deadbeef",
                                         {"orphans": [], "n_git": 0,
                                          "n_db": 0, "n_new": 0,
                                          "n_updated": 0},
                                         {"orphans": [], "n_git": 0,
                                          "n_db": 0, "n_new": 0,
                                          "n_updated": 0})
        finally:
            conn.close()

        cls._srv = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls._port = cls._srv.server_address[1]
        cls._thread = threading.Thread(
            target=cls._srv.serve_forever, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls):
        cls._srv.shutdown()
        cls._srv.server_close()
        cls._thread.join(timeout=5)

    def _url(self, path):
        return "http://127.0.0.1:%d%s" % (self._port, path)

    # ---- auth ------------------------------------------------------

    def test_task_claim_list_requires_auth(self):
        status, _ = _http("GET", self._url("/task_claim/list"))
        self.assertEqual(status, 401)

    def test_chip_list_rejects_bad_token(self):
        status, _ = _http("GET", self._url("/chip/list"), token="wrong")
        self.assertEqual(status, 401)

    # ---- /task_claim/list -------------------------------------------

    def test_task_claim_list_returns_both_claims(self):
        status, body = _http("GET", self._url("/task_claim/list"), token=TOKEN)
        self.assertEqual(status, 200)
        session_ids = {c["session_id"] for c in body["claims"]}
        self.assertEqual(session_ids, {"s1", "s2"})

    def test_task_claim_list_filters_by_status(self):
        status, body = _http(
            "GET", self._url("/task_claim/list?status=active"), token=TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(len(body["claims"]), 1)
        self.assertEqual(body["claims"][0]["session_id"], "s1")

    def test_task_claim_list_entries_are_lossless(self):
        status, body = _http(
            "GET", self._url("/task_claim/list?status=done"), token=TOKEN)
        entry = body["claims"][0]
        self.assertEqual(entry["completion_note"], "done")
        self.assertEqual(entry["closed_at"], "2026-08-26T12:00:00Z")

    # ---- /task_claim/check -------------------------------------------

    def test_task_claim_check_owned_resource(self):
        status, body = _http(
            "GET", self._url("/task_claim/check?resource=a/b.json"),
            token=TOKEN)
        self.assertEqual(status, 200)
        self.assertTrue(body["owned"])
        self.assertEqual(body["rivals"][0]["session_id"], "s1")

    def test_task_claim_check_unowned_resource(self):
        status, body = _http(
            "GET", self._url("/task_claim/check?resource=never/claimed.json"),
            token=TOKEN)
        self.assertEqual(status, 200)
        self.assertFalse(body["owned"])
        self.assertEqual(body["rivals"], [])

    def test_task_claim_check_ignores_done_claims(self):
        """s2's claim is status=done and had no resources anyway -- this
        pins that check() only ever considers ACTIVE claims, mirroring
        task_claim.py's own arbitration."""
        status, body = _http(
            "GET", self._url("/task_claim/check?resource=a/b.json"),
            token=TOKEN)
        rivals_sessions = {r["session_id"] for r in body["rivals"]}
        self.assertNotIn("s2", rivals_sessions)

    def test_task_claim_check_is_a_get_and_writes_nothing(self):
        before = db.connect(self._dbpath).execute(
            "SELECT COUNT(*) c FROM task_claims").fetchone()["c"]
        _http("GET", self._url("/task_claim/check?resource=a/b.json"),
             token=TOKEN)
        after = db.connect(self._dbpath).execute(
            "SELECT COUNT(*) c FROM task_claims").fetchone()["c"]
        self.assertEqual(before, after)

    # ---- /chip/list -------------------------------------------------

    def test_chip_list_returns_both(self):
        status, body = _http("GET", self._url("/chip/list"), token=TOKEN)
        self.assertEqual(status, 200)
        refs = {c["chip_ref"] for c in body["chips"]}
        self.assertEqual(refs, {"chip-open-1", "chip-done-1"})

    def test_chip_list_filters_by_status(self):
        status, body = _http(
            "GET", self._url("/chip/list?status=open"), token=TOKEN)
        self.assertEqual(len(body["chips"]), 1)
        self.assertEqual(body["chips"][0]["chip_ref"], "chip-open-1")

    def test_chip_list_filters_by_origin(self):
        status, body = _http(
            "GET", self._url("/chip/list?origin=hygiene_tick"), token=TOKEN)
        self.assertEqual(len(body["chips"]), 1)
        self.assertEqual(body["chips"][0]["chip_ref"], "chip-done-1")

    def test_chip_list_respects_limit(self):
        status, body = _http(
            "GET", self._url("/chip/list?limit=1"), token=TOKEN)
        self.assertEqual(len(body["chips"]), 1)

    # ---- /task_claim/drift -------------------------------------------

    def test_drift_summary(self):
        status, body = _http(
            "GET", self._url("/task_claim/drift"), token=TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(body["total_ticks"], 1)
        self.assertEqual(body["diverged_ticks"], 0)
        self.assertEqual(body["recent"][0]["source_ref"], "deadbeef")

    # ---- no mutating verb exists yet (PHASE-2 scope guard) ------------

    def test_no_mutating_task_claim_or_chip_post_route_exists(self):
        """PHASE-1 is read-only by design (see the plan doc's HARD STOP
        note). If any of these ever starts returning something other than
        404, PHASE-2 write plumbing has been added -- which must be a
        deliberate, reviewed, user-ratified change, not a silent side
        effect of an unrelated edit."""
        for path in ("/task_claim/open", "/task_claim/close",
                     "/task_claim/renew", "/task_claim/amend",
                     "/task_claim/dedupe", "/chip/record", "/chip/claim",
                     "/chip/unclaim", "/chip/resolve", "/chip/attach",
                     "/chip/amend-prompt"):
            status, _ = _http("POST", self._url(path), token=TOKEN, body={})
            self.assertEqual(
                status, 404,
                "%s must not exist yet -- PHASE-1 is read-only" % path)


if __name__ == "__main__":
    unittest.main()
