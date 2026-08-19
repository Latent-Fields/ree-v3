"""Contracts for the COORDINATOR transport of cloud-scaler.py's orchestrator
veto, and for the bootstrap wake hold (2026-08-19,
chip-20260819-cloud4-orchestrator-veto-bootstrap-deadlock).

WHAT WENT WRONG. The orchestrator veto shipped 2026-08-18 reading ONE thing:
the git-materialised runner_heartbeats/<affinity>-metaworker.json. That is the
transport this same file had already DEMOTED for runner telemetry on
2026-06-23 (fetch_coordinator_status / evaluate_heartbeat's coord_row), and
which CLAUDE.md documents as deliberately stale -- the phase3 writer commits
on state-changes only, the 30-minute liveness tick having been retired to stop
git-history bloat. So an OPERATIONAL decision (power a billable box off,
killing live `claude -p` workers) was being made from a channel designed to
lag. Live scaler lines read `hb_src=coord ... orch=none`: runner telemetry
fresh from the DB, orchestrator telemetry from git, in the same log line.

Measured 2026-08-19 on ree-worker-4:
  * ree-cloud-4-metaworker.json last_tick_utc = 01:32:44Z, box off since
    ~01:39Z, in_flight_dispatches=2 chips_open_work=80 frozen in the file.
  * 16:45:04Z the hub timer logged `status=running idle_ok=1
    reason=clean_idle lease=none orch=none` and shut the box down ONE SECOND
    after an operator's `hcloud server poweron`.
So the box could never survive long enough to refresh the very signal that
would have kept it alive -- the bootstrap deadlock.

TWO DISTINCT FIXES ARE PINNED HERE, and they are not substitutes:
  1. TRANSPORT -- read_orchestrator is coordinator-primary, git as fallback.
     This removes the commit+push lag and the state-change gating, which is
     what let ORCHESTRATOR_COORD_FRESH_MIN (12) be a quarter of the git
     figure (50).
  2. BOOTSTRAP -- orchestrator_wake_hold. No transport, however fast, can be
     published by a box that is still BOOTING. The 16:45:04Z kill happened
     one second after power-on. This is why the hold is structural rather
     than a band-aid on the transport, and why it is bounded (a box whose
     metaworker never starts is still shut down).

Time-independent: every case injects `now` or a real tempdir. Roughly half are
NEGATIVE CONTROLS -- the fail-open paths and the bounds. This veto keeps a
BILLABLE box alive, so every way it must yield is pinned explicitly; widening
either predicate by one case is a box that can never be powered off.
"""
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "cloud_scaler_transport", os.path.join(_HERE, "deploy", "cloud-scaler.py"))
cs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cs)

NOW = datetime(2026, 8, 19, 17, 0, 0, tzinfo=timezone.utc)
AFF = "ree-cloud-4"
GIT_FRESH = 50
COORD_FRESH = 12


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _coord(age_min=1, in_flight=2, open_work=5, state="dispatching",
           role="orchestrator", machine=None, drop_progress=False):
    """A /shadow/status-shaped snapshot carrying one orchestrator row."""
    progress = {
        "role": role,
        "state": state,
        "in_flight_dispatches": in_flight,
        "chips_open_work": open_work,
    }
    row = {
        "machine": machine or ("%s-metaworker" % AFF),
        "last_seen": _iso(NOW - timedelta(minutes=age_min)),
        "state": state,
    }
    if not drop_progress:
        row["progress"] = progress
    return {row["machine"]: row}


def _git_hb(dirpath, age_min=2, in_flight=2, open_work=5,
            role="orchestrator", affinity=AFF, **over):
    hb = {
        "schema_version": "v1",
        "role": role,
        "machine": "%s-metaworker" % affinity,
        "last_tick_utc": _iso(NOW - timedelta(minutes=age_min)),
        "state": "dispatching",
        "in_flight_dispatches": in_flight,
        "chips_open_work": open_work,
    }
    hb.update(over)
    with open(os.path.join(dirpath, "%s-metaworker.json" % affinity),
              "w", encoding="utf-8") as fh:
        json.dump(hb, fh)


class CoordinatorTransportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _read(self, coord_status, **kw):
        return cs.read_orchestrator(
            self.tmp, AFF, GIT_FRESH, now=NOW, coord_status=coord_status,
            coord_fresh_min=COORD_FRESH, **kw)

    # ---------- the transport is actually used ----------
    def test_coordinator_row_vetoes_with_no_git_file_at_all(self):
        """THE HEADLINE. A box with live dispatches keeps itself alive over
        the coordinator alone -- no commit, no push, no checkout pull. Under
        the 2026-08-18 code this returned no veto and the box was killed."""
        active, reason, src = self._read(_coord(in_flight=2, open_work=0))
        self.assertTrue(active)
        self.assertEqual(src, "coord")
        self.assertIn("orchestrator_busy", reason)

    def test_coordinator_backlog_alone_vetoes(self):
        active, reason, src = self._read(_coord(in_flight=0, open_work=80))
        self.assertTrue(active)
        self.assertEqual(src, "coord")
        self.assertIn("orchestrator_backlog", reason)

    def test_coordinator_is_preferred_over_a_disagreeing_git_file(self):
        """PRIMARY means primary: a FRESH coordinator row saying idle-and-empty
        wins over a stale-but-still-in-window git file that says busy. The git
        file is where a dead metaworker's last numbers sit frozen -- exactly
        the 80-chips-and-2-dispatches snapshot that outlived the box."""
        _git_hb(self.tmp, age_min=40, in_flight=2, open_work=80)
        active, reason, src = self._read(_coord(in_flight=0, open_work=0))
        self.assertFalse(active)
        self.assertEqual(src, "coord")
        self.assertIn("orchestrator_idle_and_empty", reason)

    def test_fresh_coordinator_row_that_says_idle_does_not_consult_git(self):
        _git_hb(self.tmp, age_min=1, in_flight=9, open_work=9)
        active, reason, _ = self._read(_coord(in_flight=0, open_work=0))
        self.assertFalse(active)
        self.assertNotIn("git:", reason)

    # ---------- fallback ----------
    def test_no_coordinator_snapshot_falls_back_to_git(self):
        """Coordinator unreachable this tick -> the pre-2026-08-19 behaviour,
        bit-identical. fetch_coordinator_status returns {} on any failure."""
        _git_hb(self.tmp, age_min=2, in_flight=1)
        active, reason, src = self._read({})
        self.assertTrue(active)
        self.assertEqual(src, "git")
        self.assertIn("git:orchestrator_busy", reason)

    def test_stale_coordinator_row_falls_back_to_git(self):
        """A row older than COORD_FRESH means the box stopped POSTing; the git
        mirror may still hold a newer-looking committed tick, so consult it
        rather than concluding from silence."""
        _git_hb(self.tmp, age_min=2, in_flight=1)
        active, reason, src = self._read(_coord(age_min=30))
        self.assertTrue(active)
        self.assertEqual(src, "git")
        self.assertIn("orchestrator_stale", reason)
        self.assertIn("git:orchestrator_busy", reason)

    def test_row_without_progress_falls_back_to_git(self):
        """A runner-shaped row under the metaworker name carries no demand
        fields. Falling back is right; concluding 'idle' from it is not."""
        _git_hb(self.tmp, age_min=2, in_flight=3)
        active, _, src = self._read(_coord(drop_progress=True))
        self.assertTrue(active)
        self.assertEqual(src, "git")

    def test_both_transports_silent_does_not_veto(self):
        active, reason, src = self._read({})
        self.assertFalse(active)
        self.assertEqual(src, "none")
        self.assertIn("no_orchestrator", reason)

    # ---------- negative controls: every fail-open path ----------
    def test_coordinator_role_mismatch_does_not_veto(self):
        active, reason, _ = self._read(_coord(role="runner"))
        self.assertFalse(active)
        self.assertIn("role_mismatch", reason)

    def test_coordinator_missing_demand_fields_does_not_veto(self):
        active, reason, _ = self._read(_coord(in_flight=None, open_work=None))
        self.assertFalse(active)
        self.assertIn("no_demand_fields", reason)

    def test_coordinator_idle_and_empty_does_not_veto(self):
        active, _, _ = self._read(_coord(in_flight=0, open_work=0))
        self.assertFalse(active)

    def test_another_boxes_orchestrator_row_is_not_read(self):
        """Row keyed ree-cloud-3-metaworker must not keep ree-cloud-4 alive."""
        snap = _coord(machine="ree-cloud-3-metaworker")
        active, _, src = self._read(snap)
        self.assertFalse(active)
        self.assertEqual(src, "none")

    def test_the_runners_own_row_is_not_read_as_an_orchestrator(self):
        """A bare `ree-cloud-4` row is the RUNNER. Reading it here would let
        ordinary runner telemetry veto shutdown forever."""
        snap = {AFF: {"machine": AFF, "last_seen": _iso(NOW),
                      "state": "idle", "progress": {}}}
        active, _, src = self._read(snap)
        self.assertFalse(active)
        self.assertEqual(src, "none")

    def test_coordinator_freshness_boundary_is_inclusive(self):
        """Exactly at the window still vetoes; one minute past does not."""
        at, _, _ = self._read(_coord(age_min=COORD_FRESH))
        self.assertTrue(at)
        past, reason, _ = self._read(_coord(age_min=COORD_FRESH + 1))
        self.assertFalse(past)
        self.assertIn("orchestrator_stale", reason)

    def test_coord_window_is_tighter_than_the_git_window(self):
        """The whole point of the transport change: a dead metaworker stops
        vetoing in ~12 minutes rather than ~50. If someone widens the coord
        window to the git one, this is the assertion that fails."""
        self.assertLess(cs.DEFAULTS["ORCHESTRATOR_COORD_FRESH_MIN"],
                        cs.DEFAULTS["ORCHESTRATOR_FRESH_MIN"])

    def test_git_window_still_clears_the_liveness_floor(self):
        """Unchanged invariant -- the git path must still sit above
        ree_metaworker_heartbeat.LIVENESS_FLOOR_MINUTES (30), or a healthy but
        unchanging box ages out of its own veto on schedule."""
        self.assertGreater(cs.DEFAULTS["ORCHESTRATOR_FRESH_MIN"], 30)


class WakeHoldTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _hold(self, status, now=NOW, aff=AFF, grace=12,
              is_orchestrator=True, state_dir=None):
        return cs.orchestrator_wake_hold(
            state_dir if state_dir is not None else self.tmp,
            aff, status, grace, now=now, is_orchestrator=is_orchestrator)

    def test_first_running_sighting_holds(self):
        """THE DEADLOCK BREAKER. First tick that sees the box running after it
        was off: it has published nothing because it was booting."""
        self._hold("off")
        hold, reason = self._hold("running")
        self.assertTrue(hold)
        self.assertIn("orchestrator_wake_hold", reason)

    def test_hold_expires_and_the_box_becomes_shutdownable(self):
        """BOUNDED. A box whose metaworker never comes up must not be pinned:
        this is the billing guard, same shape as PYTEST_LEASE_MAX_MIN."""
        self._hold("off")
        self.assertTrue(self._hold("running")[0])
        later = NOW + timedelta(minutes=13)
        hold, reason = self._hold("running", now=later)
        self.assertFalse(hold)
        self.assertIn("wake_grace_expired", reason)

    def test_hold_survives_intermediate_ticks_within_the_window(self):
        self._hold("off")
        self._hold("running")
        hold, _ = self._hold("running", now=NOW + timedelta(minutes=5))
        self.assertTrue(hold)

    def test_boundary_exactly_at_grace_still_holds(self):
        self._hold("off")
        self._hold("running")
        hold, _ = self._hold("running", now=NOW + timedelta(minutes=12))
        self.assertTrue(hold)

    def test_a_second_power_cycle_reopens_the_window(self):
        """off -> running -> off -> running must hold again, not stay expired
        from the first boot."""
        self._hold("off")
        self._hold("running")
        late = NOW + timedelta(minutes=60)
        self.assertFalse(self._hold("running", now=late)[0])
        self._hold("off", now=late)
        hold, _ = self._hold("running", now=late + timedelta(minutes=1))
        self.assertTrue(hold)

    # ---------- negative controls ----------
    def test_undeclared_affinity_never_holds(self):
        """The hold is gated on CONFIG membership. An ordinary worker must be
        shut down exactly as before -- this is what stops the guard becoming a
        fleet-wide 12-minute shutdown delay."""
        self._hold("off", aff="ree-cloud-2", is_orchestrator=False)
        hold, reason = self._hold("running", aff="ree-cloud-2",
                                  is_orchestrator=False)
        self.assertFalse(hold)
        self.assertEqual(reason, "not_declared_orchestrator")

    def test_off_box_never_holds(self):
        hold, reason = self._hold("off")
        self.assertFalse(hold)
        self.assertEqual(reason, "not_running")

    def test_unknown_status_never_holds(self):
        self.assertFalse(self._hold("unknown")[0])

    def test_lost_state_is_self_limiting_not_permanent(self):
        """A missing state file reads as 'first sight of it running', which
        starts the window NOW and therefore expires. It must not be read as an
        indefinite hold."""
        hold, _ = self._hold("running")          # no prior file at all
        self.assertTrue(hold)
        hold2, reason = self._hold("running", now=NOW + timedelta(minutes=30))
        self.assertFalse(hold2)
        self.assertIn("wake_grace_expired", reason)

    def test_unwritable_state_dir_fails_open(self):
        """A guard that cannot persist must degrade to NO hold, never to an
        unbounded one -- same direction as every other fail-open here."""
        blocked = os.path.join(self.tmp, "nope")
        with open(blocked, "w") as fh:      # a FILE where a dir is wanted
            fh.write("x")
        hold, reason = self._hold("running", state_dir=blocked)
        self.assertFalse(hold)
        self.assertEqual(reason, "wake_state_unwritable")

    def test_declared_affinity_matches_the_real_config(self):
        """ree-cloud-4 is the resident dispatcher (cloud_workers.md, and the
        wrapper's LEASE_AFFINITY derivation). If this list is edited, the GHA
        backstop's ORCHESTRATOR_AFFINITIES must move with it -- pinned in
        test_cloud_scaler_transport_parity.py."""
        self.assertIn("ree-cloud-4", cs.ORCHESTRATOR_AFFINITIES)
        self.assertNotIn("ree-cloud-1", cs.ORCHESTRATOR_AFFINITIES)


class IncidentReplayTest(unittest.TestCase):
    """The 2026-08-19T16:45:04Z tick, replayed against old and new logic."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.state, True)
        # The real frozen file: last_tick 01:32:44Z, ~15h before the tick.
        _git_hb(self.tmp, age_min=15 * 60, in_flight=2, open_work=80)

    def test_old_logic_shape_would_still_not_veto(self):
        """Both transports silent/stale -> no veto. This is CORRECT fail-open
        behaviour and is exactly why the wake hold, not the transport, is what
        saves the box at power-on."""
        active, reason, _ = cs.read_orchestrator(
            self.tmp, AFF, GIT_FRESH, now=NOW, coord_status={},
            coord_fresh_min=COORD_FRESH)
        self.assertFalse(active)
        self.assertIn("orchestrator_stale", reason)

    def test_wake_hold_is_what_saves_the_box_one_second_after_poweron(self):
        prev_off = cs.orchestrator_wake_hold(
            self.state, AFF, "off", 12, now=NOW - timedelta(minutes=5),
            is_orchestrator=True)
        self.assertFalse(prev_off[0])
        hold, reason = cs.orchestrator_wake_hold(
            self.state, AFF, "running", 12, now=NOW, is_orchestrator=True)
        self.assertTrue(hold, "the box must survive its own boot: %s" % reason)

    def test_and_then_the_coordinator_takes_over_before_the_hold_expires(self):
        """The handover that makes the pair work: the dispatcher's first POST
        lands inside the wake window (ree-metaworker.timer OnBootSec=20s), so
        the veto is live before the hold expires. If OnBootSec is ever raised
        above ORCHESTRATOR_WAKE_GRACE_MIN this gap reopens."""
        first_post = NOW + timedelta(minutes=1)
        active, _, src = cs.read_orchestrator(
            self.tmp, AFF, GIT_FRESH, now=first_post,
            coord_status=_coord(age_min=-1, in_flight=0, open_work=80),
            coord_fresh_min=COORD_FRESH)
        self.assertTrue(active)
        self.assertEqual(src, "coord")


if __name__ == "__main__":
    unittest.main()
