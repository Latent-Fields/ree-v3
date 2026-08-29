"""Contracts for the PHASE-4 dispatcher run-lease intake (2026-08-29
fleet-wedge campaign, W3 fold-in): db.upsert_dispatcher_lease,
POST /dispatcher/lease, and the materializer's
ingest_dispatcher_control + render_dispatcher_control integration.

Pins, in descending order of damage-if-wrong:

  1. NEWEST-WINS BOTH WAYS: an ingest of an OLD git file never clobbers a
     newer endpoint write, and an endpoint replay never clobbers a newer
     git-side stop -- the degraded-fallback doctrine (RT3: the emergency
     STOP must survive a hub outage) depends on both directions.
  2. ENVELOPE PRESERVATION: the render carries the _comment doctrine block
     (and any future top-level key) verbatim from origin; only the
     dispatchers map is DB-owned.
  3. BYTE STABILITY: an in-sync render is byte-identical to
     scripts/dispatcher_control.py._save's serialization (indent=2,
     sort_keys, trailing newline), so the writer never churns commits.
  4. PRE-FLIP QUIESCENCE: with no DB rows the renderer emits nothing --
     landing this slice changes no git behaviour until something POSTs.

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

ENTRY_STOP_OLD = {"requested_state": "stop", "requested_at":
                  "2026-08-29T08:00:00Z", "requested_by": "orch-1",
                  "note": "seed stop"}
ENTRY_RUN_NEW = {"requested_state": "run", "requested_at":
                 "2026-08-29T10:00:00Z", "requested_by": "orch-2",
                 "note": "granted", "lease_hours": 4.0,
                 "expires_at": "2026-08-29T14:00:00Z"}
ENTRY_STOP_NEWER = {"requested_state": "stop", "requested_at":
                    "2026-08-29T11:00:00Z", "requested_by": "operator-ssh",
                    "note": "emergency stop, git fallback"}

SEED_DOC = {
    "_comment": ["doctrine block line one", "line two"],
    "dispatchers": {"ree-cloud-4": ENTRY_STOP_OLD},
}


def _doc_text(doc):
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def _git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=check)


def _bare_remote(parent):
    remote = pathlib.Path(parent) / "REE_Working.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _seed_repo(parent, remote, dc_doc=SEED_DOC):
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
    if dc_doc is not None:
        (repo / writer.DISPATCHER_CONTROL_REL_PATH).write_text(
            _doc_text(dc_doc), encoding="utf-8")
        paths.append(writer.DISPATCHER_CONTROL_REL_PATH)
    _git(repo, "add", *paths)
    _git(repo, "commit", "-q", "-m", "seed")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "master")
    return repo


class _Fixture(unittest.TestCase):

    DC_DOC = SEED_DOC

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="dispatcher_lease_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._remote = _bare_remote(self._tmp)
        self._repo = _seed_repo(self._tmp, self._remote, dc_doc=self.DC_DOC)

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _tick(self, mode="check"):
        return writer.materialize_once(self._conn, str(self._repo),
                                       branch="master", mode=mode,
                                       now_iso=NOW)

    def _origin_dc(self):
        return json.loads(_git(
            self._repo, "show",
            "origin/master:%s" % writer.DISPATCHER_CONTROL_REL_PATH).stdout)


class TestUpsertDb(_Fixture):

    def test_ok_idempotent_stale(self):
        v, _ = db.upsert_dispatcher_lease(self._conn, "d", ENTRY_STOP_OLD,
                                          via="t", now=NOW)
        self.assertEqual(v, "ok")
        v, _ = db.upsert_dispatcher_lease(self._conn, "d", ENTRY_STOP_OLD,
                                          via="t", now=NOW)
        self.assertEqual(v, "idempotent")
        v, _ = db.upsert_dispatcher_lease(self._conn, "d", ENTRY_RUN_NEW,
                                          via="t", now=NOW)
        self.assertEqual(v, "ok", "strictly newer requested_at replaces")
        v, _ = db.upsert_dispatcher_lease(self._conn, "d", ENTRY_STOP_OLD,
                                          via="t", now=NOW)
        self.assertEqual(v, "stale", "an older entry never clobbers newer")
        rows = db.dispatcher_lease_rows(self._conn)
        self.assertEqual(json.loads(rows[0]["entry_json"]), ENTRY_RUN_NEW)

    def test_bad_entry(self):
        for bad in (None, "x", {"requested_state": "run"},
                    {"requested_at": "yesterday"}):
            v, _ = db.upsert_dispatcher_lease(self._conn, "d", bad, via="t")
            self.assertEqual(v, "bad_entry", repr(bad))


class TestEndpoint(_Fixture):

    def test_typed_fields_ok(self):
        code, out = app._dispatcher_lease(self._conn, {
            "dispatcher": "ree-cloud-4", "requested_state": "stop",
            "requested_by": "orch-x", "note": "n",
            "requested_at": "2026-08-29T09:00:00Z"}, "mac-tok")
        self.assertEqual(code, 200)
        self.assertEqual(out["verdict"], "ok")

    def test_entry_passthrough_ok(self):
        code, out = app._dispatcher_lease(self._conn, {
            "dispatcher": "local", "entry": ENTRY_RUN_NEW}, "mac-tok")
        self.assertEqual(code, 200)
        self.assertEqual(out["verdict"], "ok")
        rows = db.dispatcher_lease_rows(self._conn)
        self.assertEqual(json.loads(rows[0]["entry_json"]), ENTRY_RUN_NEW)

    def test_validation(self):
        code, out = app._dispatcher_lease(self._conn, {
            "requested_state": "run"}, "t")
        self.assertEqual(code, 400)
        code, out = app._dispatcher_lease(self._conn, {
            "dispatcher": "d", "requested_state": "sprint"}, "t")
        self.assertEqual(code, 400)

    def test_dispatch_table_membership(self):
        self.assertIn("/dispatcher/lease", app._TASK_CLAIM_CHIP_POST)


class TestIngestAndRender(_Fixture):

    def test_preflip_quiescence_no_rows_no_render(self):
        render, stats = writer.render_dispatcher_control(self._conn, SEED_DOC)
        self.assertIsNone(render)
        self.assertEqual(stats["n_rows"], 0)

    def test_tick_ingests_git_state_and_stays_byte_stable(self):
        result = self._tick(mode="write")
        dc = result["dispatcher_control"]
        self.assertEqual(dc["n_seen"], 1)
        self.assertEqual(dc["n_adopted"], 1)
        self.assertFalse(dc["differs"],
                         "after ingest the render must be byte-identical "
                         "to origin -- no churn commit")
        self.assertFalse(result["committed"])

    def test_endpoint_newer_write_renders_to_origin_with_envelope(self):
        self._tick(mode="check")  # ingest the seed
        code, out = app._dispatcher_lease(self._conn, {
            "dispatcher": "ree-cloud-4", "entry": ENTRY_RUN_NEW}, "mac-tok")
        self.assertEqual(out["verdict"], "ok")
        result = self._tick(mode="write")
        self.assertTrue(result["committed"])
        doc = self._origin_dc()
        self.assertEqual(doc["_comment"], SEED_DOC["_comment"],
                         "the doctrine envelope is preserved verbatim")
        self.assertEqual(doc["dispatchers"]["ree-cloud-4"], ENTRY_RUN_NEW)
        # byte-stability against the client's own serializer
        raw = _git(self._repo, "show", "origin/master:%s"
                   % writer.DISPATCHER_CONTROL_REL_PATH).stdout
        self.assertEqual(raw, _doc_text(doc))
        # and the next tick is quiet
        result = self._tick(mode="write")
        self.assertFalse(result["committed"])

    def test_git_fallback_stop_beats_older_endpoint_state(self):
        self._tick(mode="check")  # DB now holds the seed stop (08:00)
        code, out = app._dispatcher_lease(self._conn, {
            "dispatcher": "ree-cloud-4", "entry": ENTRY_RUN_NEW}, "mac-tok")
        self.assertEqual(out["verdict"], "ok")  # run granted at 10:00
        # hub outage: an operator writes an emergency stop GIT-SIDE at 11:00
        doc = dict(SEED_DOC)
        doc["dispatchers"] = {"ree-cloud-4": ENTRY_STOP_NEWER}
        path = self._repo / writer.DISPATCHER_CONTROL_REL_PATH
        path.write_text(_doc_text(doc), encoding="utf-8")
        _git(self._repo, "add", writer.DISPATCHER_CONTROL_REL_PATH)
        _git(self._repo, "commit", "-q", "-m", "operator emergency stop")
        _git(self._repo, "push", "-q", "origin", "master")
        result = self._tick(mode="write")
        self.assertEqual(result["dispatcher_control"]["n_adopted"], 1,
                         "the newer git-side stop is ADOPTED, not fought")
        self.assertEqual(
            self._origin_dc()["dispatchers"]["ree-cloud-4"],
            ENTRY_STOP_NEWER)
        rows = db.dispatcher_lease_rows(self._conn)
        self.assertEqual(json.loads(rows[0]["entry_json"]), ENTRY_STOP_NEWER)


class TestMissingFile(_Fixture):

    DC_DOC = None

    def test_missing_file_never_invented(self):
        db.upsert_dispatcher_lease(self._conn, "d", ENTRY_RUN_NEW, via="t")
        result = self._tick(mode="write")
        self.assertFalse(result["committed"])
        ls = _git(self._repo, "show", "origin/master:%s"
                  % writer.DISPATCHER_CONTROL_REL_PATH, check=False)
        self.assertNotEqual(ls.returncode, 0,
                            "the renderer must never invent the envelope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
