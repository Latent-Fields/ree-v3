"""Contract: the coordinator canonicalises machine identity AT INGEST, and
the fleet's genuinely-distinct boxes still get genuinely-distinct DB keys.

WHY THIS FILE EXISTS SEPARATELY FROM tests/contracts/test_machine_identity.py
----------------------------------------------------------------------------
That file pins the RESOLVER. This file pins the COORDINATOR'S USE of it, and
the two are not the same assertion: until 2026-08-16 the resolver was correct
and `coordinator/db.py` + `coordinator/app.py` did not call it at all (a grep
for `machine_identity` in those two files returned nothing, while the
diagnostic layer -- phase3_preflight, phase3_verify -- had been canonicalising
for weeks). So the tooling that reports identity agreement was resolving while
the code that keys real coordination state on identity was not. A resolver
test cannot catch that; only a test of the ingest path can.

THE CATASTROPHIC FAILURE THIS GUARDS
------------------------------------
Collapsing two genuinely distinct boxes. `ree-cloud-1` .. `-5` and
`ree-worker-1` .. `-4` differ ONLY in a trailing digit that IS their whole
identity. If canonicalisation ever collapsed them, the fleet would share one
`heartbeats` row (PK is `machine`), every claim would collide with every
other, and every affinity-pinned experiment would route to the wrong box --
a fleet-wide outage, not a cosmetic bug. `DistinctFleetKeysTest` below is an
exhaustive all-pairs assertion over the real ingest path, deliberately not a
spot check, because widening the predicate by one entry is what would cause
that outage.

THE FAILURE THIS FIXES
----------------------
The same physical box presenting two spellings. macOS re-suffixes
LocalHostName on a Bonjour collision, so this Mac has reported both
`DLAPTOP-4.local` and `DLAPTOP-5.local`, and `experiment_runner
._get_machine_name` now resolves every spelling to `DLAPTOP`. Against a
string-matching coordinator that rename splits the box: a second heartbeat
row, a claim its owner cannot release, and -- worst -- a claim that
`_has_fresh_owner_heartbeat` reads as ownerless and re-issues to a second
worker while the first is still running the experiment.

Design record: REE_assembly/evidence/planning/
coordinator_canonical_machine_identity_staged_20260816.md

Test classes:
  DistinctFleetKeysTest   -- all-pairs: distinct boxes stay distinct, the
                             Mac's three spellings collapse to one. Both at
                             the resolver and through real DB writes.
  AffinityDriftTest       -- db._affinity_ok across drift and across the fleet.
  HeartbeatDriftHTTPTest  -- POST /heartbeat under three spellings -> ONE row.
  ClaimDriftHTTPTest      -- claim under one spelling, release under another.
  CommandChannelDriftTest -- issue / fetch / ack round-trip across drift.
                             The issue and fetch sides must move TOGETHER;
                             canonicalising one alone silently converts a
                             deliverable command into a permanently pending
                             one. That is what this class exists to pin.
  ProvenanceStaysRawTest  -- NEGATIVE CONTROL. results.machine and
                             claim_log.machine must NOT be canonicalised.

All printed text is ASCII-only.
"""

import itertools
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import db  # noqa: E402
import machine_identity as mi  # noqa: E402

# The Mac. One physical laptop; three spellings it has actually reported.
# `DLAPTOP` is `scutil --get ComputerName`, the name that has never drifted.
MAC_SPELLINGS = ("DLAPTOP", "DLAPTOP-4.local", "DLAPTOP-5.local")
MAC_CANONICAL = "DLAPTOP"

# Genuinely distinct machines. Their trailing digit IS their identity, so
# every one of these must survive ingest as its own key. Both spellings are
# listed because the fleet really does answer to both: `ree-cloud-N` is the
# affinity name the runners are launched with, `ree-worker-N` is the hcloud
# name and what `socket.gethostname()` reports on cloud-1 and cloud-3.
DISTINCT_FLEET = (
    "ree-cloud-1", "ree-cloud-2", "ree-cloud-3", "ree-cloud-4", "ree-cloud-5",
    "ree-worker-1", "ree-worker-2", "ree-worker-3", "ree-worker-4",
)


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
    """Spin the real app.py entrypoint. Mix in before unittest.TestCase.

    Every token maps to a machine label so the `or machine_tok` fallback
    paths are exercised with a REAL label rather than a placeholder -- the
    fallback is the one route a body-only test would never reach.
    """

    MODE = "coordinator"
    TOKENS = {"tok-op": "operator", "tok-mac": "DLAPTOP-4.local",
              "tok-c2": "ree-cloud-2"}
    EXTRA_ENV = {}
    SEED_QUEUE = ()

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="mid_ingest_")
        cls._dbpath = os.path.join(cls._tmp, "c.db")
        cls._tokens = os.path.join(cls._tmp, "tokens.json")
        with open(cls._tokens, "w", encoding="utf-8") as fh:
            json.dump(cls.TOKENS, fh)
        db.init_db(cls._dbpath)
        if cls.SEED_QUEUE:
            conn = db.connect(cls._dbpath)
            for qid, affinity in cls.SEED_QUEUE:
                db.upsert_experiment(conn, {
                    "queue_id": qid,
                    "script": "experiments/%s.py" % qid.lower(),
                    "priority": 50,
                    "machine_affinity": affinity,
                    "status": "pending",
                    "estimated_minutes": 10,
                })
            conn.close()
        cls._port = _free_port()
        env = dict(os.environ)
        env.update({
            "COORDINATOR_DB": cls._dbpath,
            "COORDINATOR_TOKENS_FILE": cls._tokens,
            "COORDINATOR_BIND_HOST": "127.0.0.1",
            "COORDINATOR_BIND_PORT": str(cls._port),
            "COORDINATOR_MODE": cls.MODE,
        })
        env.update(cls.EXTRA_ENV)
        cls._proc = subprocess.Popen(
            [sys.executable, str(HERE / "app.py")],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        cls._base = "http://127.0.0.1:%d" % cls._port
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

    def _query(self, sql, args=()):
        conn = db.connect(self._dbpath)
        try:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# 1. Distinct boxes stay distinct; one box's spellings collapse.
# ---------------------------------------------------------------------------

class DistinctFleetKeysTest(unittest.TestCase):
    """The property named in the staged doc's section 4, pinned
    coordinator-side rather than inherited from the resolver's own tests."""

    def test_every_distinct_fleet_pair_stays_distinct_at_the_resolver(self):
        for a, b in itertools.combinations(DISTINCT_FLEET, 2):
            with self.subTest(a=a, b=b):
                self.assertNotEqual(
                    mi.canonical_machine_name(a),
                    mi.canonical_machine_name(b),
                    "%s and %s collapsed to one identity -- this is the "
                    "fleet-wide-outage failure, not a cosmetic one" % (a, b))

    def test_mac_never_collides_with_any_fleet_machine(self):
        for spelling in MAC_SPELLINGS:
            for fleet in DISTINCT_FLEET:
                with self.subTest(mac=spelling, fleet=fleet):
                    self.assertFalse(mi.same_machine(spelling, fleet))

    def test_mac_spellings_all_collapse_to_one_canonical_identity(self):
        for spelling in MAC_SPELLINGS:
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    mi.canonical_machine_name(spelling), MAC_CANONICAL)
        for a, b in itertools.combinations(MAC_SPELLINGS, 2):
            with self.subTest(a=a, b=b):
                self.assertTrue(mi.same_machine(a, b))

    def test_distinct_fleet_names_produce_distinct_heartbeat_rows(self):
        """Same property, but through REAL DB writes rather than the pure
        resolver -- this is the assertion that would fail if a future change
        canonicalised somewhere the resolver is not consulted."""
        tmp = tempfile.mkdtemp(prefix="mid_keys_")
        try:
            path = os.path.join(tmp, "c.db")
            db.init_db(path)
            conn = db.connect(path)
            for name in DISTINCT_FLEET:
                db.upsert_heartbeat(
                    conn, mi.canonical_machine_name(name), "idle", None,
                    None, None, None, None)
            rows = [r["machine"] for r in conn.execute(
                "SELECT machine FROM heartbeats").fetchall()]
            conn.close()
            self.assertEqual(len(rows), len(DISTINCT_FLEET))
            self.assertEqual(sorted(rows), sorted(DISTINCT_FLEET))
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. Affinity matching (db._affinity_ok)
# ---------------------------------------------------------------------------

class AffinityDriftTest(unittest.TestCase):

    def test_affinity_matches_across_mac_drift(self):
        for affinity in MAC_SPELLINGS:
            for reported in MAC_SPELLINGS:
                with self.subTest(affinity=affinity, reported=reported):
                    self.assertTrue(db._affinity_ok(affinity, reported))

    def test_affinity_does_not_match_a_different_fleet_box(self):
        for affinity, reported in itertools.permutations(DISTINCT_FLEET, 2):
            with self.subTest(affinity=affinity, reported=reported):
                self.assertFalse(db._affinity_ok(affinity, reported))

    def test_affinity_still_matches_itself_across_the_fleet(self):
        for name in DISTINCT_FLEET:
            with self.subTest(name=name):
                self.assertTrue(db._affinity_ok(name, name))

    def test_any_and_blank_affinity_still_match_anything(self):
        for affinity in (None, "", "any"):
            with self.subTest(affinity=affinity):
                self.assertTrue(db._affinity_ok(affinity, "ree-cloud-2"))

    def test_unknown_machine_does_not_satisfy_a_pinned_affinity(self):
        # An unknown identity must never satisfy a host-gated branch by
        # accident -- same rule same_machine() states for two empty names.
        for reported in (None, ""):
            with self.subTest(reported=reported):
                self.assertFalse(db._affinity_ok("ree-cloud-2", reported))


# ---------------------------------------------------------------------------
# 3. Heartbeats: three spellings, one row.
# ---------------------------------------------------------------------------

class HeartbeatDriftHTTPTest(_LiveCoordinatorMixin, unittest.TestCase):

    def test_three_spellings_produce_one_heartbeat_row(self):
        for i, spelling in enumerate(MAC_SPELLINGS):
            st, _ = _http("POST", self._base + "/heartbeat", token="tok-op",
                          body={"machine": spelling, "state": "running",
                                "current_exq": "V3-EXQ-90%d" % i})
            self.assertEqual(st, 200)
        # Scoped to the DLAPTOP family, not the whole table: sibling tests in
        # this class share one coordinator process and DB, and one of them
        # deliberately inserts cloud-fleet rows.
        rows = self._query("SELECT machine, current_exq FROM heartbeats "
                           "WHERE machine LIKE 'DLAPTOP%'")
        self.assertEqual([r["machine"] for r in rows], [MAC_CANONICAL])
        # Last write wins on the single row, which is what proves they were
        # UPDATEs of one row and not three inserts.
        self.assertEqual(rows[0]["current_exq"], "V3-EXQ-902")

    def test_token_label_fallback_is_canonicalised_too(self):
        # No `machine` in the body -> the token's label is substituted. That
        # label is a config string a human typed, so it drifts exactly like a
        # reported hostname does.
        st, _ = _http("POST", self._base + "/heartbeat", token="tok-mac",
                      body={"state": "idle"})
        self.assertEqual(st, 200)
        machines = [r["machine"] for r in
                    self._query("SELECT machine FROM heartbeats")]
        self.assertIn(MAC_CANONICAL, machines)
        self.assertNotIn("DLAPTOP-4.local", machines)

    def test_status_payload_lands_on_the_same_row(self):
        _http("POST", self._base + "/heartbeat", token="tok-op",
              body={"machine": "DLAPTOP-4.local", "state": "idle"})
        st, _ = _http("POST", self._base + "/status", token="tok-op",
                      body={"machine": "DLAPTOP-5.local",
                            "status": {"queue_depth": 3}})
        self.assertEqual(st, 200)
        rows = self._query(
            "SELECT machine, status_payload_json FROM heartbeats "
            "WHERE status_payload_json IS NOT NULL")
        self.assertEqual([r["machine"] for r in rows], [MAC_CANONICAL])

    def test_shutdown_notice_and_fence_clear_hit_the_same_row(self):
        _http("POST", self._base + "/shutdown_notify", token="tok-op",
              body={"machine": "DLAPTOP-4.local", "reason": "scaler_idle"})
        st, _ = _http("POST", self._base + "/claim_fence/clear",
                      token="tok-op", body={"machine": "DLAPTOP"})
        self.assertEqual(st, 200)
        rows = self._query(
            "SELECT machine, last_shutdown_at, claim_fence_cleared_at "
            "FROM heartbeats WHERE last_shutdown_at IS NOT NULL")
        self.assertEqual([r["machine"] for r in rows], [MAC_CANONICAL])
        # The fence really was disarmed -- clearing under a different
        # spelling must not leave the armed row armed.
        self.assertIsNotNone(rows[0]["claim_fence_cleared_at"])

    def test_fleet_spellings_are_not_merged_over_http(self):
        # Negative control at the live boundary, not only at the resolver.
        for name in ("ree-cloud-2", "ree-worker-2"):
            _http("POST", self._base + "/heartbeat", token="tok-op",
                  body={"machine": name, "state": "idle"})
        machines = {r["machine"] for r in
                    self._query("SELECT machine FROM heartbeats")}
        self.assertIn("ree-cloud-2", machines)
        self.assertIn("ree-worker-2", machines)


# ---------------------------------------------------------------------------
# 4. Claims: claim under one spelling, release under another.
# ---------------------------------------------------------------------------

class ClaimDriftHTTPTest(_LiveCoordinatorMixin, unittest.TestCase):

    SEED_QUEUE = (
        ("V3-EXQ-9001", "any"),
        ("V3-EXQ-9002", "DLAPTOP-4.local"),
        ("V3-EXQ-9003", "ree-cloud-2"),
    )

    def test_claim_then_release_across_drifted_spellings(self):
        st, jb = _http("POST", self._base + "/claim", token="tok-op",
                       body={"queue_id": "V3-EXQ-9001",
                             "machine": "DLAPTOP-4.local"})
        self.assertEqual(st, 200)
        self.assertEqual(jb["verdict"], "ok")
        row = self._query("SELECT claimed_by_machine FROM experiments "
                          "WHERE queue_id=?", ("V3-EXQ-9001",))[0]
        self.assertEqual(row["claimed_by_machine"], MAC_CANONICAL)

        # The rename the whole change exists for: the same box comes back
        # under its canonical name and must be able to release its own claim.
        st, jb = _http("POST", self._base + "/claim/release", token="tok-op",
                       body={"queue_id": "V3-EXQ-9001", "machine": "DLAPTOP"})
        self.assertEqual(st, 200, jb)
        self.assertTrue(jb["ok"])

    def test_affinity_pinned_to_a_drifted_spelling_is_still_claimable(self):
        st, jb = _http("POST", self._base + "/claim", token="tok-op",
                       body={"queue_id": "V3-EXQ-9002", "machine": "DLAPTOP"})
        self.assertEqual(st, 200)
        self.assertEqual(jb["verdict"], "ok",
                         "a queue entry pinned DLAPTOP-4.local must not "
                         "starve once the Mac reports DLAPTOP")

    def test_a_different_box_still_cannot_take_a_pinned_row(self):
        st, jb = _http("POST", self._base + "/claim", token="tok-op",
                       body={"queue_id": "V3-EXQ-9003",
                             "machine": "ree-cloud-3"})
        self.assertEqual(st, 200)
        self.assertEqual(jb["verdict"], "already_claimed")

    def test_another_box_cannot_release_a_claim_it_does_not_own(self):
        _http("POST", self._base + "/claim", token="tok-op",
              body={"queue_id": "V3-EXQ-9003", "machine": "ree-cloud-2"})
        st, jb = _http("POST", self._base + "/claim/release", token="tok-op",
                       body={"queue_id": "V3-EXQ-9003",
                             "machine": "ree-cloud-3"})
        self.assertEqual(st, 409)
        self.assertFalse(jb["ok"])


# ---------------------------------------------------------------------------
# 5. Command channel: issue and fetch must move together.
# ---------------------------------------------------------------------------

class CommandChannelDriftTest(_LiveCoordinatorMixin, unittest.TestCase):
    """The staged doc's second named risk. Canonicalising the ISSUE side
    without the FETCH side (or the reverse) turns a working delivery path
    into a silent miss -- a command that sits pending forever with no error
    anywhere. Every ordering of drifted spellings must round-trip."""

    def test_issue_drifted_fetch_canonical(self):
        st, jb = _http("POST", self._base + "/commands/issue", token="tok-op",
                       body={"machine": "DLAPTOP-4.local", "kind": "pause",
                             "issued_by": "explorer"})
        self.assertEqual(st, 200, jb)
        cmd_id = jb["command"]["id"]
        st, jb = _http("GET", self._base + "/commands?machine=DLAPTOP",
                       token="tok-op")
        self.assertEqual(st, 200)
        self.assertIn(cmd_id, [c["id"] for c in jb["commands"]],
                      "command issued under a drifted spelling was not "
                      "deliverable to the canonical one -- issue and fetch "
                      "sides have diverged")

    def test_issue_canonical_fetch_drifted(self):
        st, jb = _http("POST", self._base + "/commands/issue", token="tok-op",
                       body={"machine": "DLAPTOP", "kind": "resume",
                             "issued_by": "explorer"})
        cmd_id = jb["command"]["id"]
        st, jb = _http("GET",
                       self._base + "/commands?machine=DLAPTOP-5.local",
                       token="tok-op")
        self.assertEqual(st, 200)
        self.assertIn(cmd_id, [c["id"] for c in jb["commands"]])

    def test_ack_across_drift_clears_the_command(self):
        st, jb = _http("POST", self._base + "/commands/issue", token="tok-op",
                       body={"machine": "DLAPTOP-5.local", "kind": "stop",
                             "issued_by": "explorer"})
        cmd_id = jb["command"]["id"]
        st, jb = _http("POST", self._base + "/commands/ack", token="tok-op",
                       body={"id": cmd_id, "machine": "DLAPTOP-4.local",
                             "result_status": "done"})
        self.assertEqual(st, 200, jb)
        self.assertTrue(jb["ok"])
        st, jb = _http("GET", self._base + "/commands?machine=DLAPTOP",
                       token="tok-op")
        self.assertNotIn(cmd_id, [c["id"] for c in jb["commands"]])

    def test_ack_from_a_genuinely_different_box_is_still_refused(self):
        # Negative control: the owner guard must survive canonicalisation.
        st, jb = _http("POST", self._base + "/commands/issue", token="tok-op",
                       body={"machine": "ree-cloud-2", "kind": "pause",
                             "issued_by": "explorer"})
        cmd_id = jb["command"]["id"]
        st, jb = _http("POST", self._base + "/commands/ack", token="tok-op",
                       body={"id": cmd_id, "machine": "ree-cloud-3",
                             "result_status": "done"})
        self.assertEqual(st, 409)
        self.assertFalse(jb["ok"])

    def test_command_for_one_fleet_box_is_not_delivered_to_another(self):
        st, jb = _http("POST", self._base + "/commands/issue", token="tok-op",
                       body={"machine": "ree-cloud-4", "kind": "pause",
                             "issued_by": "explorer"})
        cmd_id = jb["command"]["id"]
        st, jb = _http("GET", self._base + "/commands?machine=ree-cloud-5",
                       token="tok-op")
        self.assertNotIn(cmd_id, [c["id"] for c in jb["commands"]])

    def test_stored_command_row_holds_the_canonical_key(self):
        _http("POST", self._base + "/commands/issue", token="tok-op",
              body={"machine": "DLAPTOP-4.local", "kind": "pause",
                    "issued_by": "explorer"})
        machines = {r["machine"] for r in
                    self._query("SELECT machine FROM commands")}
        self.assertIn(MAC_CANONICAL, machines)
        self.assertNotIn("DLAPTOP-4.local", machines)


# ---------------------------------------------------------------------------
# 6. NEGATIVE CONTROL: append-only provenance is NOT canonicalised.
# ---------------------------------------------------------------------------

class ProvenanceStaysRawTest(_LiveCoordinatorMixin, unittest.TestCase):
    """`results.machine` and `claim_log.machine` record WHAT WAS REPORTED,
    not what the coordinator decided. Rewriting them to a name the box was
    not using at the time falsifies provenance -- and would erase the only
    evidence that an identity split ever happened, which is exactly the
    evidence machine_identity.py's "10533 vs 0" argument rests on.

    These assertions are inverted on purpose. If a later change makes them
    fail by canonicalising more, read the staged doc's section 3 before
    'fixing' them."""

    SEED_QUEUE = (("V3-EXQ-9101", "any"),)

    def test_results_machine_is_stored_exactly_as_reported(self):
        manifest = {"run_id": "v3_exq_9101_drift_20260816T000000Z_v3",
                    "queue_id": "V3-EXQ-9101",
                    "machine": "DLAPTOP-5.local", "outcome": "PASS"}
        req = urllib.request.Request(
            self._base + "/result",
            data=json.dumps(manifest).encode("utf-8"),
            headers={"Authorization": "Bearer tok-op",
                     "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
        rows = self._query("SELECT machine FROM results WHERE run_id=?",
                           (manifest["run_id"],))
        self.assertEqual(rows[0]["machine"], "DLAPTOP-5.local")

    def test_claim_log_machine_is_stored_exactly_as_reported(self):
        _http("POST", self._base + "/claim", token="tok-op",
              body={"queue_id": "V3-EXQ-9101", "machine": "DLAPTOP-4.local"})
        rows = self._query("SELECT machine FROM claim_log WHERE queue_id=?",
                           ("V3-EXQ-9101",))
        self.assertTrue(rows, "no claim_log row written")
        self.assertEqual(rows[-1]["machine"], "DLAPTOP-4.local")

    def test_the_claim_itself_still_recorded_the_canonical_owner(self):
        # The point of the split: the AUDIT row keeps the raw spelling while
        # the OWNERSHIP field -- which decides behaviour -- is canonical.
        _http("POST", self._base + "/claim", token="tok-op",
              body={"queue_id": "V3-EXQ-9101", "machine": "DLAPTOP-4.local"})
        row = self._query("SELECT claimed_by_machine FROM experiments "
                          "WHERE queue_id=?", ("V3-EXQ-9101",))[0]
        self.assertEqual(row["claimed_by_machine"], MAC_CANONICAL)


if __name__ == "__main__":
    unittest.main()
