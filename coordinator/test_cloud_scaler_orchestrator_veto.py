"""Contracts for cloud-scaler.py's orchestrator veto (2026-08-18).

Motivating incident, live and reproducible: ree-worker-4 was powered on to
trial a co-resident metaworker and the scaler shut it down 4 minutes later --
`claimable=0 held_by_self=0 status=running idle_ok=1 reason=clean_idle
lease=none`. Dispatch work creates no queue claim and no runner heartbeat, so
every pre-existing signal reads clean_idle while live `claude -p` workers are
mid-flight.

Time-independent: every case injects `now`. Roughly half are NEGATIVE
CONTROLS, and they are the load-bearing half -- this veto keeps a billable box
alive, so each way it can FAIL OPEN is pinned explicitly. Widening the
predicate by one case is a box that can never be shut down.
"""
import importlib.util
import re
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "cloud_scaler", os.path.join(_HERE, "deploy", "cloud-scaler.py"))
cs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cs)

NOW = datetime(2026, 8, 18, 19, 0, 0, tzinfo=timezone.utc)
FRESH = 20


def _hb(dirpath, affinity, **over):
    hb = {
        "schema_version": "v1",
        "role": "orchestrator",
        "machine": "%s-metaworker" % affinity,
        "last_tick_utc": (NOW - timedelta(minutes=2)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "state": "dispatching",
        "in_flight_dispatches": 2,
    }
    hb.update(over)
    path = os.path.join(dirpath, "%s-metaworker.json" % affinity)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(hb, fh)
    return path


class OrchestratorVetoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _read(self, affinity="ree-cloud-4"):
        return cs.read_orchestrator(self.tmp, affinity, FRESH, now=NOW)

    # ---------- positive ----------
    def test_fresh_orchestrator_heartbeat_vetoes(self):
        _hb(self.tmp, "ree-cloud-4")
        active, reason = self._read()
        self.assertTrue(active)
        self.assertIn("orchestrator_active", reason)

    def test_boundary_age_equal_to_window_still_vetoes(self):
        _hb(self.tmp, "ree-cloud-4",
            last_tick_utc=(NOW - timedelta(minutes=FRESH)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"))
        self.assertTrue(self._read()[0])

    def test_idle_but_fresh_still_vetoes(self):
        # A dispatcher between cycles is still a live metaworker; killing the
        # box in that window loses the timer, not just one cycle.
        _hb(self.tmp, "ree-cloud-4", state="idle", in_flight_dispatches=0)
        self.assertTrue(self._read()[0])

    # ---------- negative controls: every fail-open path ----------
    def test_no_file_does_not_veto(self):
        active, reason = self._read()
        self.assertFalse(active)
        self.assertEqual(reason, "no_orchestrator")

    def test_missing_directory_does_not_veto(self):
        active, _ = cs.read_orchestrator(
            os.path.join(self.tmp, "nope"), "ree-cloud-4", FRESH, now=NOW)
        self.assertFalse(active)

    def test_stale_heartbeat_does_not_veto(self):
        _hb(self.tmp, "ree-cloud-4",
            last_tick_utc=(NOW - timedelta(minutes=FRESH + 1)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"))
        active, reason = self._read()
        self.assertFalse(active)
        self.assertIn("orchestrator_stale", reason)

    def test_runner_role_does_not_veto(self):
        # A runner heartbeat that happens to land at this name must not keep a
        # box alive -- that would defeat auto-shutdown entirely.
        _hb(self.tmp, "ree-cloud-4", role="runner")
        self.assertFalse(self._read()[0])

    def test_absent_role_does_not_veto(self):
        hb = {"machine": "x", "last_tick_utc": NOW.strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(os.path.join(self.tmp, "ree-cloud-4-metaworker.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(hb, fh)
        self.assertFalse(self._read()[0])

    def test_malformed_json_does_not_veto(self):
        with open(os.path.join(self.tmp, "ree-cloud-4-metaworker.json"),
                  "w", encoding="utf-8") as fh:
            fh.write("{not json")
        active, reason = self._read()
        self.assertFalse(active)
        self.assertIn("unreadable", reason)

    def test_non_object_json_does_not_veto(self):
        with open(os.path.join(self.tmp, "ree-cloud-4-metaworker.json"),
                  "w", encoding="utf-8") as fh:
            json.dump([1, 2], fh)
        self.assertFalse(self._read()[0])

    def test_missing_tick_does_not_veto(self):
        _hb(self.tmp, "ree-cloud-4", last_tick_utc=None)
        self.assertFalse(self._read()[0])

    def test_unparseable_tick_does_not_veto(self):
        _hb(self.tmp, "ree-cloud-4", last_tick_utc="not-a-timestamp")
        self.assertFalse(self._read()[0])

    def test_other_affinity_file_is_not_read(self):
        # cloud-4's metaworker must never keep cloud-2 alive.
        _hb(self.tmp, "ree-cloud-4")
        self.assertFalse(self._read("ree-cloud-2")[0])

    def test_plain_runner_heartbeat_filename_is_not_consulted(self):
        # <affinity>.json is the RUNNER heartbeat and must not be mistaken for
        # an orchestrator one -- the suffix is what separates the two roles.
        with open(os.path.join(self.tmp, "ree-cloud-4.json"),
                  "w", encoding="utf-8") as fh:
            json.dump({"role": "orchestrator",
                       "last_tick_utc": NOW.strftime("%Y-%m-%dT%H:%M:%SZ")}, fh)
        self.assertFalse(self._read()[0])


class VetoIsWiredIntoTheDecisionLoopTest(unittest.TestCase):
    """The predicate being right is worthless if run_once() never consults it."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.queue = os.path.join(self.tmp, "queue.json")
        with open(self.queue, "w", encoding="utf-8") as fh:
            json.dump({"items": []}, fh)          # claimable=0, held=0
        self.hb = os.path.join(self.tmp, "hb")
        os.makedirs(self.hb)
        self.shutdowns = []
        self._orig_status = cs.hcloud_describe_status
        self._orig_shutdown = cs.hcloud_shutdown
        self._orig_announce = cs.announce_shutdown
        cs.hcloud_describe_status = lambda name, dry_run=False: "running"
        cs.hcloud_shutdown = lambda name, dry_run=False: self.shutdowns.append(name)
        cs.announce_shutdown = lambda *a, **k: None

    def tearDown(self):
        cs.hcloud_describe_status = self._orig_status
        cs.hcloud_shutdown = self._orig_shutdown
        cs.announce_shutdown = self._orig_announce

    def _run(self):
        return cs.run_once(
            queue_path=self.queue, heartbeats_dir=self.hb,
            announce_script="/bin/true", idle_grace_min=0,
            heartbeat_fresh_min=35, surge_queue_threshold=2,
            hub_name="ree-worker-1",
            workers=[("ree-worker-4", "ree-cloud-4", "surge")],
            dry_run=True, lease_dir=os.path.join(self.tmp, "noleases"),
            clear_fence_script="/bin/true")

    def test_idle_box_with_no_orchestrator_is_shut_down(self):
        # NEGATIVE CONTROL -- proves the test can observe a shutdown at all,
        # so the veto assertion below is not vacuously passing.
        self._run()
        self.assertEqual(self.shutdowns, ["ree-worker-4"])

    def test_idle_box_running_metaworker_is_NOT_shut_down(self):
        _hb(self.hb, "ree-cloud-4",
            last_tick_utc=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"))
        self._run()
        self.assertEqual(self.shutdowns, [])




class FreshnessWindowCoversTheCommitCadenceTest(unittest.TestCase):
    """The veto window MUST exceed the heartbeat writer's liveness floor.

    These two constants live in different repos and neither file mentions the
    other's number, so nothing but this test couples them. Confirmed live
    2026-08-18: with the window at 20 and the floor at 30, ree-worker-4 was
    shut down while healthy and mid-cycle -- a heartbeat whose fields have not
    changed legitimately goes uncommitted for the floor's duration, so the
    hub's copy ages past a shorter window every single time.
    """

    FLOOR_PATH = "/Users/dgolden/REE_Working/scripts/ree_metaworker_heartbeat.py"

    def _floor(self):
        try:
            with open(self.FLOOR_PATH, encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            self.skipTest("umbrella scripts/ not present in this checkout")
        m = re.search(r"^LIVENESS_FLOOR_MINUTES\s*=\s*(\d+)", src, re.M)
        self.assertIsNotNone(m, "LIVENESS_FLOOR_MINUTES not found -- renamed?")
        return int(m.group(1))

    def test_window_exceeds_the_writers_liveness_floor(self):
        floor = self._floor()
        window = cs.DEFAULTS["ORCHESTRATOR_FRESH_MIN"]
        self.assertGreater(
            window, floor,
            "ORCHESTRATOR_FRESH_MIN (%d) must exceed LIVENESS_FLOOR_MINUTES "
            "(%d): a healthy box with unchanged heartbeat fields does not "
            "commit for the floor's duration, so a shorter window vetoes "
            "nothing and the box is shut down mid-cycle." % (window, floor))

    def test_window_leaves_headroom_for_git_propagation(self):
        # The hub reads its own checkout, which lags origin by minutes.
        floor = self._floor()
        window = cs.DEFAULTS["ORCHESTRATOR_FRESH_MIN"]
        self.assertGreaterEqual(
            window - floor, 15,
            "only %dmin of headroom over the %dmin liveness floor; the hub's "
            "checkout was measured lagging origin by up to ~13min."
            % (window - floor, floor))

    def test_window_still_bounded_so_a_dead_metaworker_releases_the_box(self):
        # NEGATIVE CONTROL on the direction of the fix: widening the window is
        # not free -- it is how long a DEAD metaworker keeps a billable box up.
        self.assertLessEqual(cs.DEFAULTS["ORCHESTRATOR_FRESH_MIN"], 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
