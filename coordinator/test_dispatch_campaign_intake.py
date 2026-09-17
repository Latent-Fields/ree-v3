"""Contracts for the dispatch CAMPAIGN LEDGER intake (2026-09-16,
chip-20260916-campaign-ledger-coordinator): db.add_campaign /
record_campaign_launch / set_campaign_status / upsert_campaign_ingest, the
POST /campaign/* + GET /campaign/list endpoints, and the registry
materializer's ingest_dispatch_campaigns + render_dispatch_campaigns.

Pins, in descending order of damage-if-wrong:

  1. RECORD-LAUNCH IS A MUTEX: of N dispatchers racing for one campaign,
     exactly one gets 'ok' (BEGIN IMMEDIATE), the same session_uuid retried
     is 'idempotent', every other loser is 'already_launched'.
  2. NEWEST-WINS BOTH WAYS: an ingest of an OLDER git entry never clobbers a
     newer endpoint write, and a NEWER git-side write (a fallback launch or
     status change during a hub outage) is adopted -- the degraded-fallback
     doctrine.
  3. BYTE STABILITY: every server mutation produces exactly the entry the
     client's git path writes (shape pinned here; the cross-repo parity
     against scripts/dispatch_campaigns.py itself lives in the umbrella's
     test_dispatch_campaigns_coordinator_branch.py), and an in-sync render is
     byte-identical to save_doc's serialization, so the writer never churns.
  4. ENVELOPE PRESERVATION + PRE-FLIP QUIESCENCE: the render carries the
     _comment/schema_version envelope verbatim; with no DB rows it emits
     nothing, so landing this changes no git behaviour until something POSTs.
  5. ROUTE WIRING over real HTTP: the /campaign/ prefix reaches the handler
     table (a handler in the table but outside the prefix gate 404s silently).
  6. THE SCIENCE LANE IS ACCEPTED, AND AN UNKNOWN LANE IS NOT DELETED
     (2026-09-17). Until this date the id regex alone answered every science
     entry `bad_entry`, campaign_is_live called the lane dead so
     record-launch -- the cross-box mutex -- refused it, and the render then
     DROPPED the git-side entries ingest had refused (REE_Working fd7730f70
     deleted two curated science entries two minutes after they landed). The
     client leading the hub by a deploy is the normal order, so both halves
     are pinned: the lane is admitted with its invariants, and an entry this
     hub cannot ingest survives the render regardless.

Time-independent (every verdict takes the client's `now`). ASCII-only.
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

NOW = "2026-09-16T12:00:00Z"
LATER = "2026-09-16T13:00:00Z"
PAST_EXPIRY = "2026-09-18T00:00:00Z"

ENVELOPE = {
    "_comment": ["Dispatch CAMPAIGN LEDGER -- doctrine line one", "line two"],
    "schema_version": 1,
}


def campaign(cid="campaign-20260916-gc", members=("chip-a", "chip-b"),
             status="open", expires_at="2026-09-17T12:00:00Z", **extra):
    c = {
        "campaign_id": cid,
        "lane": "campaign-bundle",
        "title": "GC stranded worktrees",
        "members": list(members),
        "bundle_basis": {"kind": "class", "value": "worktree-gc"},
        "target_box": "cloud",
        "model": "claude-sonnet-5",
        "brief": "Clear the stranded worktrees.\n",
        "created_by": "orch-1",
        "created_at": NOW,
        "expires_at": expires_at,
        "status": status,
        "checks": {},
        "launches": [],
        "status_history": [{"status": "open", "at": NOW, "by": "orch-1",
                            "note": "curated"}],
    }
    c.update(extra)
    return c


def science(cid="science-20260917-arc029-p1p3", members=("chip-sci-a",),
            preflight_verdict="GREEN", **extra):
    """A science-lane entry exactly as scripts/dispatch_campaigns.build_campaign
    writes one: ONE member, bundle_basis kind `single` naming that member, and
    the recorded read-only pre-flight riding ON the entry."""
    c = campaign(cid=cid, members=members)
    c["lane"] = "science"
    c["title"] = "ARC-029 P1/P3 gradedness"
    c["model"] = "claude-opus-5"
    c["bundle_basis"] = {"kind": "single",
                         "value": members[0] if members else ""}
    if preflight_verdict is not None:
        c["preflight"] = {"verdict": preflight_verdict, "at": NOW,
                          "by": "preflight-sonnet", "source": "/tmp/pf.md",
                          "findings": "premises hold; 2 of 2 reachable"}
    c.update(extra)  # last, so a test can override `lane` itself
    return c


def launch(uuid="sess-1", box="ree-cloud-5"):
    return {"box": box, "session_uuid": uuid, "worktree": "/wt/%s" % uuid,
            "at": LATER, "launched_by": "dispatcher@%s" % box}


def doc_text(doc):
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def ledger_doc(*campaigns):
    d = dict(ENVELOPE)
    d["campaigns"] = list(campaigns)
    return d


def _git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=check)


def _bare_remote(parent):
    remote = pathlib.Path(parent) / "REE_Working.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _seed_repo(parent, remote, camp_doc):
    repo = pathlib.Path(parent) / "REE_Working"
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
    if camp_doc is not None:
        path = repo / writer.CAMPAIGNS_REL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc_text(camp_doc), encoding="utf-8")
        paths.append(writer.CAMPAIGNS_REL_PATH)
    _git(repo, "add", *paths)
    _git(repo, "commit", "-q", "-m", "seed")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "master")
    return repo


class _Db(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="dispatch_campaign_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)


class TestAdd(_Db):

    def test_ok_idempotent_exists(self):
        c = campaign()
        v, p = db.add_campaign(self._conn, c, via="t", now=NOW)
        self.assertEqual(v, "ok")
        self.assertEqual(p["entry"], c, "the ack echoes the lossless entry")
        v, _ = db.add_campaign(self._conn, c, via="t", now=NOW)
        self.assertEqual(v, "idempotent")
        c2 = campaign(title="different")
        v, p = db.add_campaign(self._conn, c2, via="t", now=NOW)
        self.assertEqual(v, "exists")
        self.assertEqual(p["status"], "open")
        self.assertEqual(db.find_campaign(self._conn, c["campaign_id"]), c)

    def test_member_overlap_against_live_only(self):
        db.add_campaign(self._conn, campaign(), via="t", now=NOW)
        v, p = db.add_campaign(
            self._conn, campaign(cid="campaign-20260916-two",
                                 members=("chip-b", "chip-c")), via="t", now=NOW)
        self.assertEqual(v, "member_overlap")
        self.assertEqual(p["overlap"], {"campaign-20260916-gc": ["chip-b"]})
        # once the first campaign is no longer live, the member is free again
        db.set_campaign_status(self._conn, "campaign-20260916-gc", "withdrawn",
                               by="orch-1", note="dead", at=LATER, now=LATER)
        v, _ = db.add_campaign(
            self._conn, campaign(cid="campaign-20260916-two",
                                 members=("chip-b", "chip-c")), via="t", now=NOW)
        self.assertEqual(v, "ok")

    def test_bad_entry(self):
        for bad in (None, "x", campaign(cid="chip-20260916-x"),
                    campaign(members=()), campaign(expires_at="tomorrow"),
                    campaign(status="sprinting")):
            v, _ = db.add_campaign(self._conn, bad, via="t", now=NOW)
            self.assertEqual(v, "bad_entry", repr(bad)[:80])


class TestRecordLaunch(_Db):

    def setUp(self):
        super().setUp()
        db.add_campaign(self._conn, campaign(), via="t", now=NOW)

    def test_ok_shape_matches_the_client_mutation(self):
        v, p = db.record_campaign_launch(self._conn, "campaign-20260916-gc",
                                         launch(), now=LATER)
        self.assertEqual(v, "ok")
        e = p["entry"]
        self.assertEqual(e["status"], "launched")
        self.assertEqual(e["launches"], [launch()])
        self.assertEqual(e["status_history"][-1], {
            "status": "launched", "at": LATER, "by": "dispatcher@ree-cloud-5",
            "note": "launched on ree-cloud-5"})
        self.assertNotIn("launched_at", e, "record-launch adds no <status>_at")

    def test_idempotent_same_session_refuses_other(self):
        db.record_campaign_launch(self._conn, "campaign-20260916-gc",
                                  launch("sess-1"), now=LATER)
        v, _ = db.record_campaign_launch(self._conn, "campaign-20260916-gc",
                                         launch("sess-1"), now=LATER)
        self.assertEqual(v, "idempotent")
        v, p = db.record_campaign_launch(self._conn, "campaign-20260916-gc",
                                         launch("sess-2", box="ree-cloud-4"),
                                         now=LATER)
        self.assertEqual(v, "already_launched")
        self.assertEqual(p["session_uuid"], "sess-1")
        self.assertEqual(p["box"], "ree-cloud-5")
        e = db.find_campaign(self._conn, "campaign-20260916-gc")
        self.assertEqual(len(e["launches"]), 1, "the loser wrote nothing")

    def test_not_found_not_live_bad_launch(self):
        v, _ = db.record_campaign_launch(self._conn, "campaign-20260916-nope",
                                         launch(), now=LATER)
        self.assertEqual(v, "not_found")
        v, p = db.record_campaign_launch(self._conn, "campaign-20260916-gc",
                                         launch(), now=PAST_EXPIRY)
        self.assertEqual(v, "not_live")
        self.assertEqual(p["status"], "open")
        db.set_campaign_status(self._conn, "campaign-20260916-gc", "withdrawn",
                               by="o", at=LATER, now=LATER)
        v, _ = db.record_campaign_launch(self._conn, "campaign-20260916-gc",
                                         launch(), now=LATER)
        self.assertEqual(v, "not_live")
        for bad in (None, {}, {"box": "b"}):
            v, _ = db.record_campaign_launch(self._conn, "campaign-20260916-gc",
                                             bad, now=LATER)
            self.assertEqual(v, "bad_launch", repr(bad))

    def test_race_exactly_one_winner(self):
        results = []
        lock = threading.Lock()

        def run(i):
            conn = db.connect(self._dbpath)
            try:
                v, _ = db.record_campaign_launch(
                    conn, "campaign-20260916-gc", launch("sess-%d" % i),
                    now=LATER)
            finally:
                conn.close()
            with lock:
                results.append(v)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(results.count("ok"), 1, results)
        self.assertEqual(results.count("already_launched"), 7, results)


class TestSetStatus(_Db):

    def setUp(self):
        super().setUp()
        db.add_campaign(self._conn, campaign(), via="t", now=NOW)

    def test_ok_shape_idempotent_closed(self):
        v, p = db.set_campaign_status(self._conn, "campaign-20260916-gc",
                                      "expired", by="gc",
                                      note="2 of 2 member(s) still open at expiry",
                                      at=LATER, now=LATER)
        self.assertEqual(v, "ok")
        e = p["entry"]
        self.assertEqual(e["status"], "expired")
        self.assertEqual(e["expired_at"], LATER)
        self.assertEqual(e["status_history"][-1], {
            "status": "expired", "at": LATER, "by": "gc",
            "note": "2 of 2 member(s) still open at expiry"})
        v, _ = db.set_campaign_status(self._conn, "campaign-20260916-gc",
                                      "expired", by="gc", at=LATER, now=LATER)
        self.assertEqual(v, "idempotent")
        v, _ = db.set_campaign_status(self._conn, "campaign-20260916-gc",
                                      "withdrawn", by="o", at=LATER, now=LATER)
        self.assertEqual(v, "ok", "expired is not closed -- withdraw still allowed")
        v, p = db.set_campaign_status(self._conn, "campaign-20260916-gc",
                                      "expired", by="o", at=LATER, now=LATER)
        self.assertEqual(v, "closed")
        self.assertEqual(p["status"], "withdrawn")

    def test_note_defaults_to_empty_string_like_the_client(self):
        v, p = db.set_campaign_status(self._conn, "campaign-20260916-gc",
                                      "withdrawn", by="o", at=LATER, now=LATER)
        self.assertEqual(p["entry"]["status_history"][-1]["note"], "")

    def test_bad_status_not_found(self):
        for bad in ("open", "launched", "sprinting", None):
            v, _ = db.set_campaign_status(self._conn, "campaign-20260916-gc",
                                          bad, by="o", now=LATER)
            self.assertEqual(v, "bad_status", repr(bad))
        v, _ = db.set_campaign_status(self._conn, "campaign-20260916-gc",
                                      "expired", by="", now=LATER)
        self.assertEqual(v, "bad_status")
        v, _ = db.set_campaign_status(self._conn, "campaign-20260916-x",
                                      "expired", by="o", now=LATER)
        self.assertEqual(v, "not_found")


class TestIngestNewestWins(_Db):

    def test_insert_idempotent_stale_newer(self):
        c = campaign()
        v, _ = db.upsert_campaign_ingest(self._conn, c)
        self.assertEqual(v, "ok")
        v, _ = db.upsert_campaign_ingest(self._conn, c)
        self.assertEqual(v, "idempotent")
        # endpoint moves the row forward
        db.record_campaign_launch(self._conn, c["campaign_id"], launch(),
                                  now=LATER)
        # an ingest of the OLD git copy is stale, never clobbers
        v, _ = db.upsert_campaign_ingest(self._conn, c)
        self.assertEqual(v, "stale")
        self.assertEqual(len(db.find_campaign(self._conn, c["campaign_id"])
                             ["launches"]), 1)
        # a NEWER git-side write (fallback withdraw during an outage) is adopted
        newer = db.find_campaign(self._conn, c["campaign_id"])
        newer["status"] = "withdrawn"
        newer["withdrawn_at"] = "2026-09-16T14:00:00Z"
        newer["status_history"].append({"status": "withdrawn",
                                        "at": "2026-09-16T14:00:00Z",
                                        "by": "operator", "note": "outage"})
        v, _ = db.upsert_campaign_ingest(self._conn, newer)
        self.assertEqual(v, "ok")
        self.assertEqual(db.find_campaign(self._conn, c["campaign_id"])
                         ["status"], "withdrawn")

    def test_list_filters(self):
        db.upsert_campaign_ingest(self._conn, campaign())
        db.upsert_campaign_ingest(self._conn, campaign(
            cid="campaign-20260916-old", members=("chip-z",),
            expires_at="2026-09-16T11:00:00Z"))
        live = db.list_campaigns(self._conn, live=True, now=NOW)
        self.assertEqual([c["campaign_id"] for c in live],
                         ["campaign-20260916-gc"])
        self.assertEqual(len(db.list_campaigns(self._conn)), 2)
        self.assertEqual(len(db.list_campaigns(
            self._conn, campaign_id="campaign-20260916-old")), 1)
        self.assertEqual(len(db.list_campaigns(self._conn, status="launched")), 0)


class TestEndpointHandlers(_Db):

    def test_add_codes(self):
        code, out = app._campaign_add(self._conn, {"entry": campaign(),
                                                   "now": NOW}, "mac-tok")
        self.assertEqual((code, out["verdict"]), (200, "ok"))
        code, out = app._campaign_add(self._conn, {"entry": campaign(
            title="x"), "now": NOW}, "mac-tok")
        self.assertEqual((code, out["verdict"]), (409, "exists"))
        code, out = app._campaign_add(self._conn, {"entry": campaign(
            cid="campaign-20260916-two"), "now": NOW}, "mac-tok")
        self.assertEqual((code, out["verdict"]), (409, "member_overlap"))
        code, out = app._campaign_add(self._conn, {"entry": "x"}, "mac-tok")
        self.assertEqual((code, out["verdict"]), (400, "bad_entry"))

    def test_record_launch_codes(self):
        app._campaign_add(self._conn, {"entry": campaign(), "now": NOW}, "t")
        code, out = app._campaign_record_launch(self._conn, {
            "campaign_id": "campaign-20260916-gc", "launch": launch(),
            "now": LATER}, "cloud-tok")
        self.assertEqual((code, out["verdict"]), (200, "ok"))
        code, out = app._campaign_record_launch(self._conn, {
            "campaign_id": "campaign-20260916-gc", "box": "ree-cloud-4",
            "session_uuid": "sess-2", "worktree": "/wt/2",
            "launched_by": "d", "at": LATER, "now": LATER}, "cloud-tok")
        self.assertEqual((code, out["verdict"]), (409, "already_launched"))
        code, out = app._campaign_record_launch(self._conn, {
            "campaign_id": "campaign-20260916-x", "launch": launch()}, "t")
        self.assertEqual((code, out["verdict"]), (404, "not_found"))
        code, out = app._campaign_record_launch(self._conn, {
            "launch": launch()}, "t")
        self.assertEqual((code, out["verdict"]), (400, "bad_launch"))

    def test_status_codes(self):
        app._campaign_add(self._conn, {"entry": campaign(), "now": NOW}, "t")
        code, out = app._campaign_status(self._conn, {
            "campaign_id": "campaign-20260916-gc", "status": "withdrawn",
            "by": "o", "note": "n", "at": LATER, "now": LATER}, "t")
        self.assertEqual((code, out["verdict"]), (200, "ok"))
        code, out = app._campaign_status(self._conn, {
            "campaign_id": "campaign-20260916-gc", "status": "expired",
            "by": "gc", "at": LATER, "now": LATER}, "t")
        self.assertEqual((code, out["verdict"]), (409, "closed"))
        code, out = app._campaign_status(self._conn, {
            "campaign_id": "campaign-20260916-gc", "status": "open",
            "by": "o"}, "t")
        self.assertEqual((code, out["verdict"]), (400, "bad_status"))
        code, out = app._campaign_status(self._conn, {
            "campaign_id": "campaign-20260916-x", "status": "expired",
            "by": "o"}, "t")
        self.assertEqual((code, out["verdict"]), (404, "not_found"))

    def test_list_payload(self):
        app._campaign_add(self._conn, {"entry": campaign(), "now": NOW}, "t")
        app._campaign_add(self._conn, {"entry": campaign(
            cid="campaign-20260916-old", members=("chip-z",),
            expires_at="2026-09-16T11:00:00Z"), "now": NOW}, "t")
        payload = app._campaign_list_payload(self._conn, {})
        self.assertEqual(len(payload["campaigns"]), 2)
        payload = app._campaign_list_payload(
            self._conn, {"live": ["1"], "now": [NOW]})
        self.assertEqual([c["campaign_id"] for c in payload["campaigns"]],
                         ["campaign-20260916-gc"])

    def test_dispatch_table_membership(self):
        for path in ("/campaign/add", "/campaign/record-launch",
                     "/campaign/status"):
            self.assertIn(path, app._TASK_CLAIM_CHIP_POST)


TOKEN = "test-token-campaign"


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


class TestRouteWiringOverHttp(unittest.TestCase):
    """The /campaign/ prefix gate in do_POST and the GET route -- driven over
    a real socket, because a handler present in the table but missing from
    the prefix tuple answers 404 {"error": "not found"} exactly like an
    unrestarted hub does, and the client degrades to git on both."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="campaign_http_")
        cls._dbpath = os.path.join(cls._tmp, "coord.db")
        db.init_db(cls._dbpath)
        cls._saved = (app.DB_PATH, dict(app._tokens))
        app.DB_PATH = cls._dbpath
        app._tokens = {TOKEN: "ree-cloud-5"}
        cls._srv = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls._port = cls._srv.server_address[1]
        cls._thread = threading.Thread(target=cls._srv.serve_forever,
                                       daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls):
        cls._srv.shutdown()
        cls._srv.server_close()
        app.DB_PATH, app._tokens = cls._saved
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _url(self, path):
        return "http://127.0.0.1:%d%s" % (self._port, path)

    def test_round_trip(self):
        code, out = _http("POST", self._url("/campaign/add"), TOKEN,
                          {"entry": campaign(), "now": NOW})
        self.assertEqual((code, out["verdict"]), (200, "ok"))
        code, out = _http("POST", self._url("/campaign/record-launch"), TOKEN,
                          {"campaign_id": "campaign-20260916-gc",
                           "launch": launch(), "now": LATER})
        self.assertEqual((code, out["verdict"]), (200, "ok"))
        self.assertEqual(out["entry"]["status"], "launched")
        code, out = _http("POST", self._url("/campaign/record-launch"), TOKEN,
                          {"campaign_id": "campaign-20260916-gc",
                           "launch": launch("sess-2"), "now": LATER})
        self.assertEqual((code, out["verdict"]), (409, "already_launched"))
        code, out = _http("GET", self._url("/campaign/list?live=1&now=%s"
                                           % NOW), TOKEN)
        self.assertEqual(code, 200)
        self.assertEqual([c["campaign_id"] for c in out["campaigns"]],
                         ["campaign-20260916-gc"])
        code, out = _http("POST", self._url("/campaign/status"), TOKEN,
                          {"campaign_id": "campaign-20260916-gc",
                           "status": "withdrawn", "by": "o", "at": LATER})
        self.assertEqual((code, out["verdict"]), (200, "ok"))

    def test_auth_required(self):
        code, _ = _http("POST", self._url("/campaign/add"), None,
                        {"entry": campaign()})
        self.assertEqual(code, 401)


class _Writer(unittest.TestCase):

    CAMP_DOC = ledger_doc(campaign())

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="campaign_writer_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._remote = _bare_remote(self._tmp)
        self._repo = _seed_repo(self._tmp, self._remote, self.CAMP_DOC)

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _tick(self, mode="check"):
        return writer.materialize_once(self._conn, str(self._repo),
                                       branch="master", mode=mode,
                                       now_iso=NOW)

    def _origin_text(self):
        return _git(self._repo, "show",
                    "origin/master:%s" % writer.CAMPAIGNS_REL_PATH).stdout


class TestIngestAndRender(_Writer):

    def test_preflip_quiescence_no_rows_no_render(self):
        render, stats = writer.render_dispatch_campaigns(self._conn,
                                                         self.CAMP_DOC)
        self.assertIsNone(render)
        self.assertEqual(stats["n_rows"], 0)

    def test_tick_ingests_git_state_and_stays_byte_stable(self):
        result = self._tick(mode="write")
        st = result["dispatch_campaigns"]
        self.assertEqual((st["n_seen"], st["n_adopted"]), (1, 1))
        self.assertFalse(st["differs"], "after ingest the render must be "
                         "byte-identical to origin -- no churn commit")
        self.assertFalse(result["committed"])
        render, _ = writer.render_dispatch_campaigns(self._conn, self.CAMP_DOC)
        self.assertEqual(render, doc_text(self.CAMP_DOC))

    def test_endpoint_write_renders_to_origin_with_envelope(self):
        self._tick(mode="check")  # ingest the seed
        code, out = app._campaign_record_launch(self._conn, {
            "campaign_id": "campaign-20260916-gc", "launch": launch(),
            "now": LATER}, "cloud-tok")
        self.assertEqual(out["verdict"], "ok")
        result = self._tick(mode="write")
        self.assertTrue(result["committed"])
        log = _git(self._repo, "log", "-1", "--format=%s",
                   "origin/master").stdout.strip()
        self.assertTrue(log.startswith(writer.COMMIT_PREFIX))
        self.assertIn("campaigns", log)
        doc = json.loads(self._origin_text())
        self.assertEqual(doc["_comment"], ENVELOPE["_comment"],
                         "the doctrine envelope is preserved verbatim")
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["campaigns"][0]["status"], "launched")
        self.assertEqual(doc["campaigns"][0]["launches"], [launch()])
        # byte-stability against the client's own serializer
        self.assertEqual(self._origin_text(), doc_text(doc))
        # and the next tick is quiet
        self.assertFalse(self._tick(mode="write")["committed"])

    def test_endpoint_add_appends_in_ledger_order(self):
        self._tick(mode="check")
        app._campaign_add(self._conn, {"entry": campaign(
            cid="campaign-20260916-two", members=("chip-c",)), "now": NOW},
            "t")
        self._tick(mode="write")
        doc = json.loads(self._origin_text())
        self.assertEqual([c["campaign_id"] for c in doc["campaigns"]],
                         ["campaign-20260916-gc", "campaign-20260916-two"])

    def test_git_fallback_newer_write_beats_older_endpoint_state(self):
        self._tick(mode="check")
        # hub-side: a launch at 13:00
        app._campaign_record_launch(self._conn, {
            "campaign_id": "campaign-20260916-gc", "launch": launch(),
            "now": LATER}, "t")
        # hub outage: the orchestrator withdraws GIT-SIDE at 14:00 on top of
        # the launched state it could see in its materialized file
        fallback = db.find_campaign(self._conn, "campaign-20260916-gc")
        fallback["status"] = "withdrawn"
        fallback["withdrawn_at"] = "2026-09-16T14:00:00Z"
        fallback["status_history"].append({
            "status": "withdrawn", "at": "2026-09-16T14:00:00Z",
            "by": "orch-1", "note": "worker dead, git fallback"})
        path = self._repo / writer.CAMPAIGNS_REL_PATH
        path.write_text(doc_text(ledger_doc(fallback)), encoding="utf-8")
        _git(self._repo, "add", writer.CAMPAIGNS_REL_PATH)
        _git(self._repo, "commit", "-q", "-m", "fallback withdraw")
        _git(self._repo, "push", "-q", "origin", "master")
        result = self._tick(mode="write")
        self.assertEqual(result["dispatch_campaigns"]["n_adopted"], 1,
                         "the newer git-side write is ADOPTED, not fought")
        self.assertFalse(result["committed"], "and nothing to re-render")
        self.assertEqual(db.find_campaign(self._conn, "campaign-20260916-gc")
                         ["status"], "withdrawn")

    def test_git_fallback_older_state_does_not_clobber(self):
        self._tick(mode="check")
        app._campaign_record_launch(self._conn, {
            "campaign_id": "campaign-20260916-gc", "launch": launch(),
            "now": LATER}, "t")
        # origin still carries the pre-launch entry; a write tick must push
        # the launched state, not revert to origin's older copy
        result = self._tick(mode="write")
        self.assertEqual(result["dispatch_campaigns"]["n_adopted"], 0)
        self.assertTrue(result["committed"])
        self.assertEqual(json.loads(self._origin_text())["campaigns"][0]
                         ["status"], "launched")


class TestScienceLane(_Db):
    """Lane 2 (2026-09-17). The gate here is STRUCTURAL -- it pins what every
    later reader in this module depends on. `budget_category()` ("the member
    must classify as science") is deliberately NOT mirrored server-side; see
    db._campaign_validate for why a second copy of that heuristic would be
    the same failure again."""

    def test_add_accepts_a_science_entry(self):
        v, p = db.add_campaign(self._conn, science(), via="t", now=NOW)
        self.assertEqual(v, "ok")
        self.assertEqual(p["entry"], science(),
                         "lossless passthrough -- preflight and all")
        self.assertEqual(db.find_campaign(self._conn,
                                          "science-20260917-arc029-p1p3"),
                         science())

    def test_live_and_listed_alongside_bundles(self):
        self.assertTrue(db.campaign_is_live(science(), NOW))
        db.add_campaign(self._conn, science(), via="t", now=NOW)
        db.add_campaign(self._conn, campaign(), via="t", now=NOW)
        live = db.list_campaigns(self._conn, live=True, now=NOW)
        self.assertEqual([c["campaign_id"] for c in live],
                         ["science-20260917-arc029-p1p3",
                          "campaign-20260916-gc"])

    def test_record_launch_is_the_same_mutex_on_this_lane(self):
        """The bug that mattered most: campaign_is_live refusing the lane made
        record-launch -- the campaign-side claim-first -- answer `not_live`,
        so a science launch fell back to git and lost the one-worker-per-entry
        guarantee."""
        db.add_campaign(self._conn, science(), via="t", now=NOW)
        cid = "science-20260917-arc029-p1p3"
        v, p = db.record_campaign_launch(self._conn, cid, launch(), now=LATER)
        self.assertEqual(v, "ok")
        self.assertEqual(p["entry"]["status"], "launched")
        v, _ = db.record_campaign_launch(self._conn, cid, launch(), now=LATER)
        self.assertEqual(v, "idempotent")
        v, p = db.record_campaign_launch(self._conn, cid, launch("sess-2"),
                                         now=LATER)
        self.assertEqual(v, "already_launched")
        self.assertEqual(p["session_uuid"], "sess-1")

    def test_member_overlap_sees_a_live_science_entry(self):
        db.add_campaign(self._conn, science(), via="t", now=NOW)
        v, p = db.add_campaign(self._conn, campaign(
            cid="campaign-20260917-bundle", members=("chip-sci-a", "chip-q")),
            via="t", now=NOW)
        self.assertEqual(v, "member_overlap")
        self.assertEqual(p["overlap"],
                         {"science-20260917-arc029-p1p3": ["chip-sci-a"]})

    def test_status_transitions_work_on_this_lane(self):
        db.add_campaign(self._conn, science(), via="t", now=NOW)
        v, p = db.set_campaign_status(self._conn,
                                      "science-20260917-arc029-p1p3",
                                      "expired", by="gc", at=LATER, now=LATER)
        self.assertEqual(v, "ok")
        self.assertEqual(p["entry"]["expired_at"], LATER)

    def test_refusals_keep_the_lane_invariants(self):
        cases = {
            "two members": science(members=("chip-a", "chip-b")),
            "no preflight": science(preflight_verdict=None),
            "RED preflight": science(preflight_verdict="RED"),
            "unknown verdict": science(preflight_verdict="PURPLE"),
            "bundle-shaped id": science(cid="campaign-20260917-sci"),
            "unknown lane": campaign(cid="campaign-20260917-x", lane="quantum"),
        }
        for label, entry in cases.items():
            v, _ = db.add_campaign(self._conn, entry, via="t", now=NOW)
            self.assertEqual(v, "bad_entry", label)
            self.assertIsNone(db.find_campaign(self._conn,
                                               entry["campaign_id"]), label)

    def test_a_science_id_on_the_bundle_lane_is_refused(self):
        """The prefix is how every reader without the ledger beside it (a
        worktree, a branch, a listing) tells the lanes apart, so it may not
        disagree with `lane` in either direction."""
        entry = science()
        entry["lane"] = "campaign-bundle"
        v, p = db.add_campaign(self._conn, entry, via="t", now=NOW)
        self.assertEqual(v, "bad_entry")
        self.assertIn("science-lane id", p["reason"])

    def test_bundle_lane_rules_are_unchanged(self):
        for bad in (campaign(cid="chip-20260916-x"), campaign(members=()),
                    campaign(expires_at="tomorrow"),
                    campaign(status="sprinting")):
            v, _ = db.add_campaign(self._conn, bad, via="t", now=NOW)
            self.assertEqual(v, "bad_entry", repr(bad)[:80])
        # the bundle cap HAS an --allow-oversize escape on the client, so the
        # server must not enforce one
        v, _ = db.add_campaign(self._conn, campaign(
            cid="campaign-20260916-big",
            members=tuple("chip-%d" % i for i in range(9))), via="t", now=NOW)
        self.assertEqual(v, "ok")

    def test_lane_defaults_to_bundle_for_pre_2026_09_17_entries(self):
        entry = campaign()
        entry.pop("lane")
        v, _ = db.add_campaign(self._conn, entry, via="t", now=NOW)
        self.assertEqual(v, "ok")
        self.assertTrue(db.campaign_is_live(entry, NOW))

    def test_endpoint_and_ingest_accept_the_lane(self):
        code, out = app._campaign_add(self._conn, {"entry": science(),
                                                   "now": NOW}, "mac-tok")
        self.assertEqual((code, out["verdict"]), (200, "ok"),
                         "the HTTP 400 bad_entry that forced every science "
                         "curation onto the git path")
        code, out = app._campaign_add(self._conn, {
            "entry": science(cid="science-20260917-red",
                             members=("chip-sci-b",),
                             preflight_verdict="RED"), "now": NOW}, "mac-tok")
        self.assertEqual((code, out["verdict"]), (400, "bad_entry"))
        v, _ = db.upsert_campaign_ingest(self._conn, science(
            cid="science-20260917-other", members=("chip-sci-c",)))
        self.assertEqual(v, "ok")


class TestUnknownLaneSurvivesTheRender(_Writer):
    """The second half of the 2026-09-17 fix. render_dispatch_campaigns
    rebuilds `campaigns` from the DB rows, and the DB never deletes a row --
    so an entry present in git and absent here can only be one ingest
    REFUSED, and rendering without it is deletion of a curated record. That
    is what fd7730f70 did to two science entries."""

    CAMP_DOC = ledger_doc(campaign(), science(lane="science-fiction"))

    def test_an_unrepresentable_entry_is_preserved_not_dropped(self):
        result = self._tick(mode="write")
        st = result["dispatch_campaigns"]
        self.assertEqual((st["n_seen"], st["n_adopted"], st["n_rejected"]),
                         (2, 1, 1))
        self.assertEqual(st["rejected"][0]["campaign_id"],
                         "science-20260917-arc029-p1p3")
        self.assertEqual(st["n_preserved"], 1)
        self.assertFalse(st["differs"], "preserving it is byte-stable -- the "
                         "writer must not churn on an entry it cannot hold")
        self.assertFalse(result["committed"])
        self.assertEqual([c["campaign_id"]
                          for c in json.loads(self._origin_text())["campaigns"]],
                         ["campaign-20260916-gc",
                          "science-20260917-arc029-p1p3"])

    def test_it_survives_a_render_the_db_does_drive(self):
        self._tick(mode="check")
        app._campaign_record_launch(self._conn, {
            "campaign_id": "campaign-20260916-gc", "launch": launch(),
            "now": LATER}, "t")
        result = self._tick(mode="write")
        self.assertTrue(result["committed"])
        doc = json.loads(self._origin_text())
        self.assertEqual([c["campaign_id"] for c in doc["campaigns"]],
                         ["campaign-20260916-gc",
                          "science-20260917-arc029-p1p3"])
        self.assertEqual(doc["campaigns"][0]["status"], "launched")


class TestMissingFile(_Writer):

    CAMP_DOC = None

    def test_missing_file_never_invented(self):
        db.upsert_campaign_ingest(self._conn, campaign())
        result = self._tick(mode="write")
        self.assertFalse(result["committed"])
        ls = _git(self._repo, "show", "origin/master:%s"
                  % writer.CAMPAIGNS_REL_PATH, check=False)
        self.assertNotEqual(ls.returncode, 0,
                            "the renderer must never invent the envelope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
