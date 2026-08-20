"""UNKNOWN is not IDLE: the cloud-scaler must never read a failed
orchestrator read as evidence that there is no work.

THE INCIDENT (2026-08-20T03:45:11Z, ree-worker-4). Two consecutive hub-timer
ticks, five minutes apart, nothing about the box changed in between:

  03:40:04  [ree-worker-4 affinity=ree-cloud-4] ... orch=active orch_src=coord
            -> ree-cloud-4 is running metaworker-dispatch, keeping
               ree-worker-4 running (orchestrator_busy in_flight=3
               state=dispatching age=2min)

  03:45:11  [ree-worker-4 affinity=ree-cloud-4] ... orch=none orch_src=coord
            -> no matching work AND runner idle past grace window,
               shutting down ree-worker-4
  03:45:13  Sent shutdown signal to server 131490371

ree-cloud-4 is the RESIDENT metaworker-dispatch box. Its own orchestrator
heartbeat, written 03:42:48Z -- 2min23s before that tick, i.e. FRESHER than
the age=2min the scaler had just accepted -- recorded
in_flight_dispatches=2, "3 dispatched worker(s) still running" and
chips_open_work=49. Both the strong veto (live dispatches) and the weak one
(chip backlog) should have fired. Neither was ever evaluated.

WHY. read_orchestrator returned a BOOLEAN, and False meant two different
things: "measured, and there is no work" and "could not read it". The
coordinator row came back degraded (orch_src=coord proves a row WAS returned
and judged), judge_orchestrator hit one of its fail-open non-answers, and
because that non-answer was not specifically `orchestrator_stale` the git
mirror -- which held the correct numbers -- was never opened. A 04:00:18Z
tick later logged `orch=active orch_src=git`, so the fallback was present,
reachable and correct throughout.

Absence of a signal was read as evidence of absence of work, and it was wired
to an IRREVERSIBLE action: power off a billable box, killing live `claude -p`
workers mid-flight and stranding their claimed chips until CLAIM_STALE_HOURS.

WHAT IS ASSERTED HERE, in two layers:

  * verdict layer -- orchestrator_verdict() distinguishes ORCH_ACTIVE /
    ORCH_IDLE / ORCH_UNKNOWN, falls back to the git mirror on ANY unknown
    (not only on `orchestrator_stale`), and reaches ORCH_IDLE by exactly one
    path: a parsed, role-correct, fresh record whose two demand counters are
    both integers and both zero.
  * decision layer -- run_once() actually HOLDS on unknown, for a DECLARED
    orchestrator affinity only, and still shuts everything else down.

Both layers are needed. The verdict function can be perfect and the decision
matrix still fall through to the shutdown branch, which is precisely the
class of gap test_cloud_scaler_transport_parity.py was written for.

ROUGHLY HALF OF THESE ARE NEGATIVE CONTROLS, and they are the load-bearing
half. "Do not shut a box down when you are unsure" degenerates into "never
shut anything down" -- a 24/7 bill on four VMs -- unless something pins the
paths that must still collect: a plain experiment runner on clean_idle, a
measured-idle orchestrator with an empty ledger, an undecidable record on a
box that is NOT declared an orchestrator, the hub, and every pre-existing
veto. Those are asserted below and are the first thing to check if a later
session widens this predicate.

TIME-INDEPENDENT: every fixture timestamp is generated relative to real now,
so nothing here depends on the wall-clock date. run_once() takes no `now`
injection, so the end-to-end cases feed it fixtures aged relative to now
rather than pinning a clock.
"""

import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCALER_PY = os.path.join(_HERE, "deploy", "cloud-scaler.py")


def _load_scaler_module():
    """Import cloud-scaler.py despite the hyphen in its filename."""
    spec = importlib.util.spec_from_file_location("cloud_scaler_unk", _SCALER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load_scaler_module()

AFF = "ree-cloud-4"
NOW = datetime(2027, 3, 4, 12, 0, 0, tzinfo=timezone.utc)
GIT_FRESH = cs.DEFAULTS["ORCHESTRATOR_FRESH_MIN"]
COORD_FRESH = cs.DEFAULTS["ORCHESTRATOR_COORD_FRESH_MIN"]


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ago(minutes, base=None):
    return iso((base or NOW) - timedelta(minutes=minutes))


def coord_snapshot(affinity=AFF, age_min=2, in_flight=0, open_work=0,
                   role="orchestrator", state="dispatching",
                   drop_progress=False, drop_keys=(), base=None):
    """A /shadow/status snapshot as fetch_coordinator_status returns it.

    `in_flight=None` is NOT a synthetic shape -- it is what the real
    transport produces. ree_metaworker_heartbeat.coordinator_progress()
    projects the tick with `heartbeat.get(k)` over a fixed field list, so a
    field the emitting caller omitted arrives as an explicit JSON null inside
    an otherwise-populated progress blob: the row is present and non-empty
    (hence orch_src=coord), and undecidable.
    """
    progress = {
        "role": role,
        "state": state,
        "in_flight_dispatches": in_flight,
        "chips_open_work": open_work,
        "chips_open_decision": 0,
        "coordination_plane_paused": False,
        "last_tick_utc": _ago(age_min, base),
    }
    for k in drop_keys:
        progress.pop(k, None)
    row = {
        "machine": "%s-metaworker" % affinity,
        "last_seen": _ago(age_min, base),
        "state": state,
        "progress": {} if drop_progress else progress,
    }
    return {row["machine"]: row}


class VerdictMixin:
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write_git_orch(self, age_min=2, in_flight=0, open_work=0,
                       role="orchestrator", state="dispatching",
                       affinity=AFF, base=None, extra=None):
        hb = {
            "schema_version": "v1",
            "role": role,
            "machine": "%s-metaworker" % affinity,
            "last_tick_utc": _ago(age_min, base),
            "state": state,
            "in_flight_dispatches": in_flight,
            "chips_open_work": open_work,
        }
        hb.update(extra or {})
        with open(os.path.join(self.tmp, "%s-metaworker.json" % affinity),
                  "w", encoding="utf-8") as fh:
            json.dump(hb, fh)

    def verdict(self, coord_status, affinity=AFF, now=NOW):
        return cs.orchestrator_verdict(
            self.tmp, affinity, GIT_FRESH, now=now,
            coord_status=coord_status, coord_fresh_min=COORD_FRESH)


class UnknownIsNotIdleTest(VerdictMixin, unittest.TestCase):
    """The verdict layer. UNKNOWN must survive as its own answer, and must
    cross-check the other transport before it is believed."""

    # ---- the incident, and the class of read failure it belongs to ----

    def test_incident_replay_2026_08_20(self):
        """THE REGRESSION TEST. The coordinator row is undecidable (a demand
        counter arrived as null); the git mirror, 2min23s old, says
        in_flight=2 with a 49-chip backlog. The old code returned
        (False, ..., "coord") without ever opening the mirror, and the box was
        powered off with three live workers."""
        self.write_git_orch(age_min=2, in_flight=2, open_work=49)
        verdict, reason, src = self.verdict(
            coord_snapshot(in_flight=None, open_work=None))
        self.assertEqual(verdict, cs.ORCH_ACTIVE)
        self.assertEqual(src, "git")
        self.assertIn("orchestrator_no_demand_fields", reason)
        self.assertIn("git:orchestrator_busy", reason)

    def test_incident_replay_backlog_alone_is_also_recovered(self):
        """Same read failure, weak veto only: nothing in flight but 49 chips
        of backlog this box would pick up on its next tick."""
        self.write_git_orch(age_min=2, in_flight=0, open_work=49)
        verdict, reason, _ = self.verdict(coord_snapshot(in_flight=None))
        self.assertEqual(verdict, cs.ORCH_ACTIVE)
        self.assertIn("git:orchestrator_backlog", reason)

    def test_coordinator_role_mismatch_now_consults_git(self):
        self.write_git_orch(age_min=2, in_flight=3)
        verdict, _, src = self.verdict(coord_snapshot(role="runner"))
        self.assertEqual(verdict, cs.ORCH_ACTIVE)
        self.assertEqual(src, "git")

    def test_coordinator_missing_tick_now_consults_git(self):
        self.write_git_orch(age_min=2, in_flight=3)
        snap = coord_snapshot(drop_keys=("last_tick_utc",))
        for row in snap.values():
            row["last_seen"] = None
        verdict, _, src = self.verdict(snap)
        self.assertEqual(verdict, cs.ORCH_ACTIVE)
        self.assertEqual(src, "git")

    def test_coordinator_unparseable_tick_now_consults_git(self):
        self.write_git_orch(age_min=2, in_flight=3)
        snap = coord_snapshot()
        for row in snap.values():
            row["last_seen"] = "not-a-timestamp"
            row["progress"]["last_tick_utc"] = "not-a-timestamp"
        verdict, _, src = self.verdict(snap)
        self.assertEqual(verdict, cs.ORCH_ACTIVE)
        self.assertEqual(src, "git")

    def test_stale_coordinator_row_still_consults_git(self):
        """Pre-existing behaviour -- the one unknown that ALREADY fell
        through. It must keep doing so; the change widens the door, it does
        not move it."""
        self.write_git_orch(age_min=2, in_flight=3)
        verdict, _, src = self.verdict(
            coord_snapshot(age_min=COORD_FRESH + 20, in_flight=0))
        self.assertEqual(verdict, cs.ORCH_ACTIVE)
        self.assertEqual(src, "git")

    # ---- the harder case: nothing is readable anywhere ----

    def test_undecidable_on_both_transports_is_unknown(self):
        """Coordinator row undecidable AND no git mirror file. The old code
        returned False, which authorised a shutdown. It must be UNKNOWN."""
        verdict, reason, src = self.verdict(coord_snapshot(in_flight=None))
        self.assertEqual(verdict, cs.ORCH_UNKNOWN)
        self.assertEqual(src, "none")
        self.assertIn("no_demand_fields", reason)
        self.assertIn("no_orchestrator", reason)

    def test_silence_on_both_transports_is_unknown(self):
        """Coordinator unreachable this tick ({}), no git file either."""
        verdict, _, src = self.verdict({})
        self.assertEqual(verdict, cs.ORCH_UNKNOWN)
        self.assertEqual(src, "none")

    def test_stale_on_both_transports_is_unknown(self):
        """A doubly-stale record says nothing about current demand. Note this
        is the case that CHANGES behaviour for a dead metaworker: it used to
        age out into a shutdown, and now holds. That cost is accepted
        deliberately -- see the ORCH_* block in cloud-scaler.py."""
        self.write_git_orch(age_min=GIT_FRESH + 40, in_flight=2, open_work=80)
        verdict, _, src = self.verdict(
            coord_snapshot(age_min=COORD_FRESH + 40, in_flight=2))
        self.assertEqual(verdict, cs.ORCH_UNKNOWN)
        self.assertEqual(src, "git")

    def test_malformed_git_mirror_is_unknown_not_idle(self):
        with open(os.path.join(self.tmp, "%s-metaworker.json" % AFF),
                  "w", encoding="utf-8") as fh:
            fh.write("{not json at all")
        verdict, _, _ = self.verdict(coord_snapshot(in_flight=None))
        self.assertEqual(verdict, cs.ORCH_UNKNOWN)

    # ---- NEGATIVE CONTROLS: a MEASUREMENT must stay a measurement ----

    def test_measured_idle_from_coordinator_is_idle_not_unknown(self):
        """THE load-bearing negative control at this layer. A fresh,
        role-correct, well-formed row with both counters at integer zero is a
        measurement of 'no work' and must remain distinguishable from a
        failed read -- otherwise the box is simply always on."""
        verdict, reason, src = self.verdict(
            coord_snapshot(in_flight=0, open_work=0, state="idle"))
        self.assertEqual(verdict, cs.ORCH_IDLE)
        self.assertEqual(src, "coord")
        self.assertIn("orchestrator_idle_and_empty", reason)

    def test_measured_idle_from_coordinator_does_not_consult_git(self):
        """Invariant (5) is untouched: COORDINATOR-PRIMARY means a
        measurement from the primary transport ends the read, even when the
        deliberately-lagging git mirror disagrees. Widening the fall-through
        to unknowns must not quietly demote the primary."""
        self.write_git_orch(age_min=1, in_flight=9, open_work=9)
        verdict, reason, src = self.verdict(
            coord_snapshot(in_flight=0, open_work=0, state="idle"))
        self.assertEqual(verdict, cs.ORCH_IDLE)
        self.assertEqual(src, "coord")
        self.assertNotIn("git:", reason)

    def test_measured_idle_from_git_is_idle_not_unknown(self):
        """Coordinator unreachable, git mirror fresh and well-formed and
        empty -- still a measurement, so the box stays collectable on the
        fallback path too."""
        self.write_git_orch(age_min=2, in_flight=0, open_work=0, state="idle")
        verdict, reason, src = self.verdict({})
        self.assertEqual(verdict, cs.ORCH_IDLE)
        self.assertEqual(src, "git")
        self.assertIn("idle_and_empty", reason)

    def test_active_from_coordinator_is_unchanged(self):
        verdict, reason, src = self.verdict(coord_snapshot(in_flight=2))
        self.assertEqual(verdict, cs.ORCH_ACTIVE)
        self.assertEqual(src, "coord")
        self.assertIn("orchestrator_busy", reason)

    def test_another_boxes_record_does_not_make_this_one_unknown_active(self):
        """cloud-3's orchestrator record must not be read for cloud-4 -- and
        the resulting verdict is UNKNOWN (nothing was read), not ACTIVE."""
        self.write_git_orch(age_min=1, in_flight=5, affinity="ree-cloud-3")
        verdict, _, src = self.verdict(
            coord_snapshot(affinity="ree-cloud-3", in_flight=5))
        self.assertEqual(verdict, cs.ORCH_UNKNOWN)
        self.assertEqual(src, "none")

    def test_read_orchestrator_boolean_view_is_active_only(self):
        """The back-compat wrapper the GHA parity suite compares against.
        True for ACTIVE only; both IDLE and UNKNOWN read as False there,
        which is why nothing may decide a shutdown from it."""
        self.assertTrue(cs.read_orchestrator(
            self.tmp, AFF, GIT_FRESH, now=NOW,
            coord_status=coord_snapshot(in_flight=2),
            coord_fresh_min=COORD_FRESH)[0])
        self.assertFalse(cs.read_orchestrator(
            self.tmp, AFF, GIT_FRESH, now=NOW,
            coord_status=coord_snapshot(in_flight=0, open_work=0),
            coord_fresh_min=COORD_FRESH)[0])
        self.assertFalse(cs.read_orchestrator(
            self.tmp, AFF, GIT_FRESH, now=NOW,
            coord_status={}, coord_fresh_min=COORD_FRESH)[0])

    def test_exactly_one_path_reaches_idle(self):
        """Differential guard on judge_orchestrator itself. Every defect
        shape must be UNKNOWN; only the fully-determined empty record is
        IDLE. A later session relaxing one of these back to IDLE re-arms the
        incident for that shape alone, which is how this bug survived the
        first two rounds of hardening."""
        good = {"role": "orchestrator", "last_tick_utc": _ago(1),
                "state": "idle", "in_flight_dispatches": 0,
                "chips_open_work": 0}
        self.assertEqual(cs.judge_orchestrator(good, GIT_FRESH, NOW)[0],
                         cs.ORCH_IDLE)
        for label, mutate in (
            ("role", lambda h: h.update(role="runner")),
            ("no_tick", lambda h: h.pop("last_tick_utc")),
            ("bad_tick", lambda h: h.update(last_tick_utc="nope")),
            ("stale", lambda h: h.update(
                last_tick_utc=_ago(GIT_FRESH + 10))),
            ("no_in_flight", lambda h: h.pop("in_flight_dispatches")),
            ("no_open_work", lambda h: h.pop("chips_open_work")),
            ("null_in_flight", lambda h: h.update(in_flight_dispatches=None)),
            ("str_open_work", lambda h: h.update(chips_open_work="49")),
        ):
            with self.subTest(defect=label):
                hb = dict(good)
                mutate(hb)
                self.assertEqual(
                    cs.judge_orchestrator(hb, GIT_FRESH, NOW)[0],
                    cs.ORCH_UNKNOWN,
                    "%s must be UNKNOWN, not a measurement" % label)


# ---------------------------------------------------------------------------
# Decision layer. The verdict can be right and the box still die, if the
# if/elif chain in run_once falls through to the shutdown branch.
# ---------------------------------------------------------------------------

class RunOnceMixin:
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.hb_dir = os.path.join(self.root, "heartbeats")
        self.lease_dir = os.path.join(self.root, "leases")
        self.state_dir = os.path.join(self.root, "state")
        for d in (self.hb_dir, self.lease_dir, self.state_dir):
            os.makedirs(d)
        self.queue_path = os.path.join(self.root, "experiment_queue.json")
        self.write_queue([])
        self.status = {}          # server_name -> hcloud status
        self.shutdowns = []
        self.poweron = []
        self.coord = {}

        real_shutdown = cs.hcloud_shutdown
        real_poweron = cs.hcloud_poweron
        real_describe = cs.hcloud_describe_status
        real_announce = cs.announce_shutdown
        real_fetch = cs.fetch_coordinator_status

        def _describe(server_name, dry_run=False):
            return self.status.get(server_name, "running")

        cs.hcloud_describe_status = _describe
        cs.hcloud_shutdown = lambda name, dry_run=False: self.shutdowns.append(name)
        cs.hcloud_poweron = lambda name, **kw: self.poweron.append(name)
        cs.announce_shutdown = lambda *a, **kw: None
        cs.fetch_coordinator_status = lambda *a, **kw: self.coord

        def _restore():
            cs.hcloud_shutdown = real_shutdown
            cs.hcloud_poweron = real_poweron
            cs.hcloud_describe_status = real_describe
            cs.announce_shutdown = real_announce
            cs.fetch_coordinator_status = real_fetch

        self.addCleanup(_restore)

    def write_queue(self, items):
        with open(self.queue_path, "w", encoding="utf-8") as fh:
            json.dump({"items": items}, fh)

    def runner_row(self, affinity, state="idle", age_min=1, current_exq=None):
        """A /shadow/status row for the experiment RUNNER on a box. state
        idle + fresh + no current_exq is what produces idle_ok=1
        reason=clean_idle, i.e. the shutdown precondition."""
        return {"machine": affinity, "last_seen": _ago(age_min, _real_now()),
                "state": state, "current_exq": current_exq}

    def seed_wake_state(self, affinity, up_min=120):
        """Pre-age the bootstrap wake hold so it is expired. Without this a
        declared orchestrator box holds for ORCHESTRATOR_WAKE_GRACE_MIN on
        its first running observation and would mask what is under test."""
        with open(cs.wake_state_path(self.state_dir, affinity),
                  "w", encoding="utf-8") as fh:
            json.dump({"status": "running",
                       "running_since": _ago(up_min, _real_now()),
                       "observed_at": _ago(0, _real_now())}, fh)

    def write_git_orch(self, affinity=AFF, age_min=2, in_flight=0,
                       open_work=0, role="orchestrator", state="dispatching"):
        with open(os.path.join(self.hb_dir, "%s-metaworker.json" % affinity),
                  "w", encoding="utf-8") as fh:
            json.dump({"schema_version": "v1", "role": role,
                       "machine": "%s-metaworker" % affinity,
                       "last_tick_utc": _ago(age_min, _real_now()),
                       "state": state,
                       "in_flight_dispatches": in_flight,
                       "chips_open_work": open_work}, fh)

    def run_tick(self, workers=None, orchestrator_affinities=None):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cs.run_once(
                self.queue_path, self.hb_dir,
                os.path.join(self.root, "no-such-announce.sh"),
                cs.DEFAULTS["IDLE_GRACE_MIN"],
                cs.DEFAULTS["HEARTBEAT_FRESH_MIN"],
                cs.DEFAULTS["SURGE_QUEUE_THRESHOLD"],
                cs.DEFAULTS["HUB_NAME"],
                workers if workers is not None else list(cs.WORKERS),
                coordinator_url="http://test", coordinator_token="t",
                lease_dir=self.lease_dir, state_dir=self.state_dir,
                orchestrator_affinities=(
                    cs.ORCHESTRATOR_AFFINITIES
                    if orchestrator_affinities is None
                    else orchestrator_affinities),
            )
        self.assertEqual(rc, 0)
        return buf.getvalue()


def _real_now():
    return datetime.now(timezone.utc)


class UnreadableOrchestratorHoldTest(RunOnceMixin, unittest.TestCase):
    """run_once() must actually reach a hold branch, not just compute a
    verdict nobody acts on."""

    W4 = [("ree-worker-4", AFF, "surge")]

    def test_incident_replay_2026_08_20_does_not_shut_down(self):
        """END TO END. Everything as it was at 03:45:11Z: queue empty, runner
        clean_idle, no pytest lease, wake grace long expired, coordinator
        orchestrator row present but undecidable -- and the git mirror
        holding the real numbers. Must not power the box off."""
        self.seed_wake_state(AFF)
        self.coord = dict(self.runner_coord(in_flight=None))
        self.write_git_orch(in_flight=2, open_work=49)
        out = self.run_tick(workers=self.W4)
        self.assertEqual(self.shutdowns, [],
                         "ree-worker-4 was powered off with live dispatches "
                         "readable on the git mirror -- this is the "
                         "2026-08-20 incident, reproduced")
        self.assertIn("keeping ree-worker-4 running", out)

    def test_unreadable_from_both_transports_holds(self):
        """The harder case: nothing readable anywhere. Silence is not idle."""
        self.seed_wake_state(AFF)
        self.coord = dict(self.runner_coord(in_flight=None))
        out = self.run_tick(workers=self.W4)
        self.assertEqual(self.shutdowns, [])
        self.assertIn("could not be read from EITHER transport", out)
        self.assertIn("orch=unknown", out)

    def test_no_signal_at_all_holds(self):
        """Coordinator carries no orchestrator row and there is no mirror
        file -- the shape a wedged metaworker leaves behind."""
        self.seed_wake_state(AFF)
        self.coord = {AFF: self.runner_row(AFF)}
        out = self.run_tick(workers=self.W4)
        self.assertEqual(self.shutdowns, [])
        self.assertIn("orch=unknown", out)

    def test_log_line_distinguishes_unknown_from_measured_idle(self):
        """The observability half of the fix. On 2026-08-20 both a measured
        idle and a failed read printed `orch=none`, and orch_reason was
        logged only on the branch where the veto fired -- so the tick that
        killed the box recorded nothing about why. An operator must be able
        to tell the two apart from the journal alone."""
        self.seed_wake_state(AFF)
        self.coord = dict(self.runner_coord(in_flight=None))
        unknown_out = self.run_tick(workers=self.W4)

        self.shutdowns = []
        self.coord = dict(self.runner_coord(in_flight=0, open_work=0))
        idle_out = self.run_tick(workers=self.W4)

        self.assertIn("orch=unknown", unknown_out)
        self.assertIn("orch=idle", idle_out)
        self.assertNotIn("orch=none", unknown_out)
        for out in (unknown_out, idle_out):
            self.assertIn("orch_why=", out)
        self.assertIn("orch_why=orchestrator_idle_and_empty", idle_out)

    # ---- NEGATIVE CONTROLS: what must STILL be collected ----------------
    # If a later session widens the hold until nothing is ever shut down,
    # these are what fail. Do not relax them to make a change pass.

    def test_measured_idle_orchestrator_with_empty_ledger_still_shuts_down(self):
        """THE load-bearing negative control. This is what keeps ree-cloud-4
        demand-sensitive rather than unconditionally always-on: a metaworker
        that is ticking, fresh, well-formed and reports nothing in flight and
        an empty ledger falls through to the ordinary shutdown test."""
        self.seed_wake_state(AFF)
        self.coord = dict(self.runner_coord(in_flight=0, open_work=0))
        self.run_tick(workers=self.W4)
        self.assertEqual(self.shutdowns, ["ree-worker-4"])

    def test_plain_worker_on_clean_idle_still_shuts_down(self):
        """A box with no orchestrator role has no orchestrator record BY
        DESIGN, so its absence is a determinate fact, not a failed read.
        This is the cost-control path -- if it stops working the fleet bills
        24/7."""
        self.coord = {"ree-cloud-2": self.runner_row("ree-cloud-2")}
        self.run_tick(workers=[("ree-worker-2", "ree-cloud-2", "full")])
        self.assertEqual(self.shutdowns, ["ree-worker-2"])

    def test_undecidable_record_on_a_NON_declared_box_still_shuts_down(self):
        """Scope control. Even a genuinely unreadable orchestrator-shaped
        record must not hold a box that ORCHESTRATOR_AFFINITIES does not
        declare -- config is the authority on which boxes have the role
        (invariant 6), so an unknown there is not a reason to keep paying."""
        self.write_git_orch(affinity="ree-cloud-3", age_min=2,
                            in_flight=None, open_work=None)
        self.coord = {"ree-cloud-3": self.runner_row("ree-cloud-3")}
        out = self.run_tick(workers=[("ree-worker-3", "ree-cloud-3", "full")])
        self.assertEqual(self.shutdowns, ["ree-worker-3"])
        self.assertNotIn("EITHER transport", out)

    def test_hub_is_never_touched(self):
        """Invariant (1), unchanged. The hub is skipped before any read or
        decision, so no verdict of any kind can act on it."""
        self.coord = {"ree-cloud-1": self.runner_row("ree-cloud-1")}
        out = self.run_tick(
            workers=[("ree-worker-1", "ree-cloud-1", "full")])
        self.assertEqual(self.shutdowns, [])
        self.assertEqual(self.poweron, [])
        self.assertIn("HUB_NAME match", out)
        self.assertNotIn("orch=", out)

    def test_held_by_self_still_vetoes(self):
        """Invariant (2), unchanged and still ahead of everything here."""
        self.seed_wake_state(AFF)
        self.write_queue([{"queue_id": "V3-EXQ-999", "status": "claimed",
                           "claimed_by": {"machine": AFF}}])
        self.coord = dict(self.runner_coord(in_flight=0, open_work=0))
        out = self.run_tick(workers=self.W4)
        self.assertEqual(self.shutdowns, [])
        self.assertIn("active claim", out)

    def test_pytest_lease_still_vetoes(self):
        """The lease veto is out of scope and must be unaffected -- it sits
        after the new hold in the chain, so a hold that swallowed it would be
        invisible except on a box with a lease and no orchestrator role."""
        with open(os.path.join(self.lease_dir, "ree-cloud-2.lease"),
                  "w", encoding="utf-8") as fh:
            json.dump({"expires_at": iso(_real_now() + timedelta(minutes=10)),
                       "owner": "test", "purpose": "remote_pytest"}, fh)
        self.coord = {"ree-cloud-2": self.runner_row("ree-cloud-2")}
        out = self.run_tick(workers=[("ree-worker-2", "ree-cloud-2", "full")])
        self.assertEqual(self.shutdowns, [])
        self.assertIn("work lease", out)

    def test_an_off_box_is_never_powered_ON_by_the_hold(self):
        """The hold withholds a shutdown; it must not manufacture a start.
        Surge power-on is out of scope and its conditions are unrelated."""
        self.seed_wake_state(AFF)
        self.status["ree-worker-4"] = "off"
        self.coord = dict(self.runner_coord(in_flight=None))
        self.run_tick(workers=self.W4)
        self.assertEqual(self.poweron, [])
        self.assertEqual(self.shutdowns, [])

    def test_active_veto_still_fires_and_is_reported_as_such(self):
        """The pre-existing strong veto must keep its own branch and its own
        message -- a hold reported where a veto belongs would hide the
        difference between 'known busy' and 'unknown'."""
        self.seed_wake_state(AFF)
        self.coord = dict(self.runner_coord(in_flight=3, open_work=49))
        out = self.run_tick(workers=self.W4)
        self.assertEqual(self.shutdowns, [])
        self.assertIn("is running metaworker-dispatch", out)
        self.assertIn("orch=active", out)
        self.assertNotIn("EITHER transport", out)

    # -- helper --------------------------------------------------------

    def runner_coord(self, in_flight=0, open_work=0, age_min=2):
        """Both rows a live ree-cloud-4 publishes: the RUNNER row (which
        drives idle_ok/clean_idle) and the ORCHESTRATOR row."""
        snap = coord_snapshot(age_min=age_min, in_flight=in_flight,
                              open_work=open_work, base=_real_now())
        snap[AFF] = self.runner_row(AFF)
        return snap


if __name__ == "__main__":
    unittest.main(verbosity=2)
