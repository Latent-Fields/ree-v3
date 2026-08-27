"""End-to-end contracts for the PHASE-2 mutating endpoints (plan doc 5.2.5).

Separate module from test_task_claim_chip_endpoints.py on purpose: that file's
PHASE-1 read-only assertions are written against a fixed, class-scoped fixture
DB, and a write test sharing it would pollute them (confirmed while writing
this -- two of its list assertions failed the moment round-trip claims landed
in the same table). These tests need a mutable DB of their own, so they get
their own server on their own ephemeral port.

Scope: the WIRING -- route table, body parse, status-code mapping, verdict
passthrough. The verb SEMANTICS are pinned at the db.py layer in
test_task_claim_chip_mutations.py, and are not re-asserted here.

ASCII-only.
"""

import json
import os
import pathlib
import shutil
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

TOKEN = "test-token-mut"
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


class TestMutatingEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="task_claim_chip_mut_ep_")
        cls._dbpath = os.path.join(cls._tmp, "coord.db")
        db.init_db(cls._dbpath)
        app.DB_PATH = cls._dbpath
        app._tokens = {TOKEN: MACHINE}
        cls._srv = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls._port = cls._srv.server_address[1]
        cls._thread = threading.Thread(target=cls._srv.serve_forever, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls):
        cls._srv.shutdown()
        cls._srv.server_close()
        cls._thread.join(timeout=5)
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _url(self, path):
        return "http://127.0.0.1:%d%s" % (self._port, path)

    def test_open_close_roundtrip_over_http(self):
        """One end-to-end pass through the real Handler, so the wiring between
        the route table, the body parse and db.py is exercised -- not just the
        db layer in isolation."""
        status, payload = _http("POST", self._url("/task_claim/open"),
                                token=TOKEN,
                                body={"session_id": "http-s1",
                                      "session_label": "l", "task": "t",
                                      "resources": ["http/a.py"],
                                      "claimed_at": "2026-08-27T20:00:00Z"})
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["verdict"], "ok")

        # A rival on the same file is a 409 VERDICT, not an error to retry.
        status, payload = _http("POST", self._url("/task_claim/open"),
                                token=TOKEN,
                                body={"session_id": "http-s2",
                                      "session_label": "l", "task": "t",
                                      "resources": ["http/a.py"],
                                      "claimed_at": "2026-08-27T20:00:01Z"})
        self.assertEqual(status, 409, payload)
        self.assertEqual(payload["verdict"], "owned_by_other")
        self.assertEqual(payload["rivals"][0]["session_id"], "http-s1")

        status, payload = _http("POST", self._url("/task_claim/close"),
                                token=TOKEN,
                                body={"session_id": "http-s1",
                                      "closed_at": "2026-08-27T20:05:00Z",
                                      "completion_note": "REE_Working abc123"})
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["verdict"], "ok")

        # And the closed claim no longer blocks the rival.
        status, payload = _http("POST", self._url("/task_claim/open"),
                                token=TOKEN,
                                body={"session_id": "http-s2",
                                      "session_label": "l", "task": "t",
                                      "resources": ["http/a.py"],
                                      "claimed_at": "2026-08-27T20:06:00Z"})
        self.assertEqual(status, 200, payload)

    def test_chip_record_claim_resolve_roundtrip_over_http(self):
        ref = "chip-http-1"
        chip = {"chip_ref": ref, "origin": "spawn_task", "kind": "work",
                "session_id": "spawner", "title": "t", "tldr": "d",
                "prompt": "work [chip_ref: %s] here" % ref,
                "cwd": "/Users/dgolden/REE_Working", "origin_host": "DLAPTOP",
                "spawned_at": "2026-08-27T20:00:00Z", "status": "open"}
        status, payload = _http("POST", self._url("/chip/record"), token=TOKEN,
                                body={"chip": chip})
        self.assertEqual(status, 200, payload)

        status, payload = _http("POST", self._url("/chip/claim"), token=TOKEN,
                                body={"chip_ref": ref, "claimed_by": "w1",
                                      "claimed_host": "DLAPTOP",
                                      "claimed_at": "2026-08-27T20:01:00Z"})
        self.assertEqual(status, 200, payload)

        status, payload = _http("POST", self._url("/chip/claim"), token=TOKEN,
                                body={"chip_ref": ref, "claimed_by": "w2",
                                      "claimed_host": "ree-cloud-5",
                                      "claimed_at": "2026-08-27T20:02:00Z"})
        self.assertEqual(status, 409, payload)
        self.assertEqual(payload["claimed_by"], "w1")

        status, payload = _http("POST", self._url("/chip/resolve"), token=TOKEN,
                                body={"chip_ref": ref, "status": "done",
                                      "note": "landed",
                                      "resolved_by_session_id": "w1"})
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["changed"])

        # D11: the second resolve is reported, not silently swallowed.
        status, payload = _http("POST", self._url("/chip/resolve"), token=TOKEN,
                                body={"chip_ref": ref, "status": "done",
                                      "note": "again"})
        self.assertEqual(status, 200, payload)
        self.assertFalse(payload["changed"])

    def test_a_marker_less_chip_prompt_is_a_400(self):
        status, payload = _http("POST", self._url("/chip/record"), token=TOKEN,
                                body={"chip": {"chip_ref": "chip-http-nm",
                                               "prompt": "no marker",
                                               "spawned_at": "2026-08-27T20:00:00Z",
                                               "session_id": "s"}})
        self.assertEqual(status, 400, payload)
        self.assertEqual(payload["verdict"], "missing_marker")

    def test_an_unknown_chip_is_a_404(self):
        status, payload = _http("POST", self._url("/chip/claim"), token=TOKEN,
                                body={"chip_ref": "chip-does-not-exist",
                                      "claimed_by": "w"})
        self.assertEqual(status, 404, payload)



if __name__ == "__main__":
    unittest.main()
