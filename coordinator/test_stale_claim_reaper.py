"""Tests for the stale-claim reaper (departure-evidence claim recovery).

The drain-window claim fence (test_shutdown_notify.py class 4) stops a
DRAINING worker taking a claim it cannot finish. It cannot help a claim that
is ALREADY orphaned by any other route -- a worker that crashes, is
SIGKILLed, loses network, or was shut down before the fence existed. For
those, _claim_recoverable's 6h `stale_hours` floor was the only recovery
path, and the confirmed V3-EXQ-841 orphan (2026-07-30) sat unclaimable for
~10h even though its owner was provably powered off within ~26 minutes.

The reaper widens recovery WITHOUT lowering that floor, because the floor is
protecting something real: CLAUDE.md's standing warning that a
heartbeat-stale machine (especially the Mac, DLAPTOP-4) may still be RUNNING
the experiment, and releasing that claim causes a DUPLICATE RUN. So the
absence-based rule keeps its 6h, and a second, PRESENCE-based rule is added
alongside it: the machine announced a shutdown AND has since gone quiet.

Four test classes:

  MachineDepartedTest       -- pure function, no HTTP, no DB write.
  ReaperDBTest              -- try_claim / evaluate_claim against a real DB.
  ReaperSafetyTest          -- the cases where it must NOT fire.
  ReaperHTTPTest            -- end-to-end via the real app.py entrypoint.

Time-independence: nothing sleeps out a threshold. The pure function takes
`now`; the DB tests backdate `heartbeats.last_seen` /
`experiments.claimed_at` with direct SQL, which is exactly the state a real
elapsed hour would produce.

All printed text is ASCII-only.
"""

import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db  # noqa: E402


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now():
    return datetime.now(timezone.utc)


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


# ---------------------------------------------------------------------------
# 1. Pure function: db.machine_departed
# ---------------------------------------------------------------------------

class MachineDepartedTest(unittest.TestCase):

    QUIET = 900

    def _departed(self, last_seen, shutdown, cleared=None,
                  reason="scaler_idle_after_grace", quiet=None, now=None):
        return db.machine_departed(
            last_seen, shutdown, cleared, reason,
            quiet_seconds=self.QUIET if quiet is None else quiet, now=now)

    def test_no_shutdown_notice_is_not_departure(self):
        # THE central safety property. A machine that has simply stopped
        # talking is not departed, however long the silence -- that is the
        # Mac case, and reaping it duplicates a live run.
        silent = _iso(_now() - timedelta(days=3))
        self.assertFalse(self._departed(silent, None))

    def test_shutdown_plus_silence_is_departure(self):
        old = _iso(_now() - timedelta(hours=1))
        self.assertTrue(self._departed(old, old))

    def test_fresh_heartbeat_is_not_departure(self):
        # A worker inside its ACPI drain window keeps heartbeating for the
        # whole ~26 minutes. It has announced a shutdown but has NOT gone,
        # and it is still running whatever it holds.
        self.assertFalse(self._departed(
            _iso(_now() - timedelta(seconds=30)),
            _iso(_now() - timedelta(minutes=5))))

    def test_incident_replay_and_why_lifecycle_state_cannot_be_reused(self):
        # V3-EXQ-841: announce 07:00:18Z, heartbeats continue through the
        # drain to ~07:26Z, box off. Asserted at 07:45Z.
        #
        # Note what lifecycle_state says here: because the last heartbeat
        # (07:26) is NEWER than the shutdown notice (07:00), its
        # `shutdown_dt >= seen_dt` test fails and it returns "stale", NOT
        # "gracefully_offline". So a reaper keyed on gracefully_offline
        # would not fire on the very incident it was built for. That is why
        # machine_departed is its own predicate: it asks "was a departure
        # announced, and has the machine since gone quiet", which does not
        # require the notice to be the most recent event.
        now = _now()
        announce = now - timedelta(minutes=45)
        last_hb = now - timedelta(minutes=19)
        self.assertEqual(
            db.lifecycle_state(_iso(last_hb), _iso(announce),
                               live_threshold_seconds=300,
                               stale_after_seconds=7 * 86400),
            "stale")
        self.assertTrue(self._departed(_iso(last_hb), _iso(announce),
                                       now=now))

    def test_process_exit_reasons_are_not_departure(self):
        # The runner posts these itself on the way out -- and by then it has
        # already released its claim (experiment_runner._do_immediate_exit),
        # while ree-runner.service's Restart=always brings it right back. So
        # there is nothing to reap and a live machine to race.
        old = _iso(_now() - timedelta(hours=1))
        for reason in ("runner_drain_complete", "runner_signal_exit"):
            self.assertFalse(self._departed(old, old, reason=reason), reason)

    def test_unrecognised_reason_is_departure(self):
        # Symmetric with the fence, which treats an unknown reason as a
        # machine shutdown. One reason-set, one meaning: whatever the fence
        # calls "this machine is going away" is what the reaper acts on
        # once it goes quiet. A new scaler reason string therefore does not
        # silently switch recovery back off.
        old = _iso(_now() - timedelta(hours=1))
        self.assertTrue(self._departed(old, old, reason="some_future_reason"))
        self.assertTrue(self._departed(old, old, reason=None))

    def test_poweron_fence_clear_cancels_departure(self):
        # The scaler stamps a fence clear on every poweron. A machine that
        # has been woken is not departed, however old its last shutdown --
        # and this is what closes the window between power-on and the first
        # heartbeat, during which last_seen is still ancient.
        shutdown = _iso(_now() - timedelta(hours=2))
        cleared = _iso(_now() - timedelta(minutes=1))
        self.assertFalse(self._departed(shutdown, shutdown, cleared))

    def test_clear_from_a_previous_wake_does_not_cancel(self):
        cleared = _iso(_now() - timedelta(days=1))
        shutdown = _iso(_now() - timedelta(hours=2))
        self.assertTrue(self._departed(shutdown, shutdown, cleared))

    def test_same_second_tie_is_not_departure(self):
        # 1-second timestamp resolution cannot order a clear and a notice
        # landing in the same second. The fence resolves that tie by staying
        # ARMED; the reaper must resolve it the other way, by NOT reaping --
        # a wrong deferral costs one poll tick, a wrong reap risks a
        # duplicate run on a box that was just powered back on. Hence `>=`
        # in machine_departed against the fence's `>`. Asserted together so
        # the asymmetry cannot be "tidied" into agreement.
        t = _iso(_now() - timedelta(hours=1))
        self.assertFalse(self._departed(t, t, t))
        self.assertTrue(db.claim_fence_active(
            t, t, "scaler_idle_after_grace", fence_seconds=1800,
            now=db._parse_utc(t) + timedelta(minutes=1)))

    def test_never_heartbeated_machine_is_departure(self):
        # record_shutdown_notice on a fresh row writes the epoch sentinel
        # for last_seen (a shutdown notice is not a heartbeat). That is
        # maximally quiet, so it must not read as "still talking".
        self.assertTrue(self._departed(
            db._NEVER_HEARTBEATED_SENTINEL,
            _iso(_now() - timedelta(minutes=30))))

    def test_null_last_seen_is_departure(self):
        self.assertTrue(self._departed(None, _iso(_now() - timedelta(hours=1))))

    def test_zero_quiet_seconds_disables(self):
        old = _iso(_now() - timedelta(hours=1))
        self.assertFalse(self._departed(old, old, quiet=0))

    def test_quiet_window_is_a_threshold_not_a_formality(self):
        # Just inside the window -> not yet departed; just outside -> yes.
        now = _now()
        shutdown = _iso(now - timedelta(hours=2))
        self.assertFalse(self._departed(
            _iso(now - timedelta(seconds=self.QUIET - 60)), shutdown, now=now))
        self.assertTrue(self._departed(
            _iso(now - timedelta(seconds=self.QUIET + 60)), shutdown, now=now))

    def test_default_quiet_seconds_matches_heartbeat_fresh_default(self):
        # One number for "we would have heard from this machine by now".
        # It must also comfortably exceed the runner's ~60s heartbeat
        # cadence, or an ordinary missed tick would read as departure.
        self.assertEqual(db.CLAIM_REAP_QUIET_DEFAULT_SECONDS, 900)
        self.assertGreater(db.CLAIM_REAP_QUIET_DEFAULT_SECONDS, 5 * 60)


# ---------------------------------------------------------------------------
# 2. DB level: try_claim / evaluate_claim
# ---------------------------------------------------------------------------

class _ReaperDBBase(unittest.TestCase):

    QUIET = 900
    FENCE = 1800
    QID = "V3-EXQ-841"

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="reaper_db_")
        self._dbpath = os.path.join(self._tmp, "c.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        db.upsert_experiment(self._conn, {
            "queue_id": self.QID,
            "script": "experiments/v3_exq_841.py",
            "priority": 50,
            "machine_affinity": "any",
            "status": "pending",
            "estimated_minutes": 600,
        })

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    # -- helpers ----------------------------------------------------------

    def _claim(self, machine, reap=None, fence=0, qid=None):
        return db.try_claim(
            self._conn, qid or self.QID, machine,
            fence_seconds=fence,
            reap_quiet_seconds=self.QUIET if reap is None else reap)

    def _backdate_heartbeat(self, machine, minutes):
        self._conn.execute(
            "UPDATE heartbeats SET last_seen=? WHERE machine=?",
            (_iso(_now() - timedelta(minutes=minutes)), machine))

    def _backdate_claim(self, minutes, qid=None):
        self._conn.execute(
            "UPDATE experiments SET claimed_at=? WHERE queue_id=?",
            (_iso(_now() - timedelta(minutes=minutes)), qid or self.QID))

    def _row(self, qid=None):
        return self._conn.execute(
            "SELECT status, claimed_by_machine, claimed_at FROM experiments "
            "WHERE queue_id=?", (qid or self.QID,)).fetchone()

    def _orphan(self, owner="ree-cloud-4", claim_age_min=37,
                silence_min=19, reason="scaler_idle_after_grace"):
        """Reproduce the V3-EXQ-841 state: `owner` claimed the item, kept
        heartbeating through its drain, then powered off."""
        self.assertEqual(self._claim(owner), "ok")
        db.upsert_heartbeat(self._conn, owner, state="running",
                            current_exq=self.QID, progress=None, gpu=None)
        db.record_shutdown_notice(self._conn, owner, reason=reason)
        self._backdate_heartbeat(owner, silence_min)
        self._backdate_claim(claim_age_min)


class ReaperDBTest(_ReaperDBBase):

    def test_orphaned_claim_is_recovered_by_a_sibling(self):
        self._orphan()
        # 37 minutes in -- nowhere near the 6h floor, which is the point.
        self.assertEqual(self._claim("ree-cloud-3"), "ok")
        row = self._row()
        self.assertEqual(row["status"], "claimed")
        self.assertEqual(row["claimed_by_machine"], "ree-cloud-3")

    def test_recovery_refreshes_claimed_at(self):
        # Otherwise the new owner inherits the dead owner's clock and is
        # itself reapable by the stale rule far too early.
        self._orphan(claim_age_min=300)
        self.assertEqual(self._claim("ree-cloud-3"), "ok")
        claimed_at = db._parse_utc(self._row()["claimed_at"])
        self.assertLess((_now() - claimed_at).total_seconds(), 120)

    def test_evaluate_claim_agrees_with_try_claim(self):
        self._orphan()
        self.assertEqual(
            db.evaluate_claim(self._conn, self.QID, "ree-cloud-3",
                              reap_quiet_seconds=self.QUIET),
            "ok")
        # ... and read-only really is read-only.
        self.assertEqual(self._row()["claimed_by_machine"], "ree-cloud-4")

    def test_exactly_one_of_several_claimants_wins(self):
        # The stub's own TODO called this out ("needs care to avoid two
        # machines both recovering the same claim") and proposed a
        # push-and-back-off protocol. try_claim's BEGIN IMMEDIATE gives it
        # for free -- every worker can find the same orphan reapable in the
        # same second.
        self._orphan()
        verdicts = [self._claim(m) for m in
                    ("ree-cloud-2", "ree-cloud-3", "DLAPTOP-4")]
        self.assertEqual(verdicts.count("ok"), 1)
        self.assertEqual(verdicts.count("already_claimed"), 2)

    def test_reaper_never_marks_anything_completed(self):
        # Guard against manufacturing a phantom completion (DB completed
        # with no manifest). The only transition is claimed -> claimed.
        self._orphan()
        self._claim("ree-cloud-3")
        self.assertEqual(self._row()["status"], "claimed")
        self.assertEqual(
            self._conn.execute("SELECT COUNT(*) c FROM results").fetchone()["c"],
            0)

    def test_terminal_row_is_never_reaped(self):
        self._orphan()
        self._conn.execute(
            "UPDATE experiments SET status='completed' WHERE queue_id=?",
            (self.QID,))
        self.assertEqual(self._claim("ree-cloud-3"), "already_claimed")

    def test_machine_with_no_heartbeat_row_has_not_departed(self):
        self.assertFalse(
            db.machine_has_departed(self._conn, "never-seen", self.QUIET))

    def test_affinity_still_applies_to_a_reaped_claim(self):
        self._conn.execute(
            "UPDATE experiments SET machine_affinity='ree-cloud-4' "
            "WHERE queue_id=?", (self.QID,))
        self._orphan()
        self.assertEqual(self._claim("ree-cloud-3"), "already_claimed")


class ReaperSafetyTest(_ReaperDBBase):
    """The cases where the reaper must NOT fire. These are the ones that
    would cost a duplicate run, so they carry the weight."""

    def test_heartbeat_stale_without_a_shutdown_notice_is_not_reaped(self):
        # THE load-bearing negative. This is the Mac: telemetry has gone
        # quiet for hours, the machine never announced anything, and it may
        # well still be running the experiment. It must wait out the full
        # stale_hours floor exactly as before.
        self.assertEqual(self._claim("DLAPTOP-4"), "ok")
        db.upsert_heartbeat(self._conn, "DLAPTOP-4", state="running",
                            current_exq=self.QID, progress=None, gpu=None)
        self._backdate_heartbeat("DLAPTOP-4", 5 * 60)   # 5h silent
        self._backdate_claim(5 * 60)                    # 5h claimed
        self.assertEqual(self._claim("ree-cloud-3"), "already_claimed")

    def test_the_six_hour_floor_still_recovers_that_case(self):
        # ... and the original rule is untouched underneath: past the
        # floor, the same claim is recoverable as it always was.
        self.assertEqual(self._claim("DLAPTOP-4"), "ok")
        db.upsert_heartbeat(self._conn, "DLAPTOP-4", state="running",
                            current_exq=self.QID, progress=None, gpu=None)
        self._backdate_heartbeat("DLAPTOP-4", 7 * 60)
        self._backdate_claim(7 * 60)
        self.assertEqual(self._claim("ree-cloud-3"), "ok")

    def test_draining_owner_still_heartbeating_is_not_reaped(self):
        # Mid-drain: shutdown announced, box still up and running the
        # experiment. Reaping here duplicates a run that is in progress.
        self.assertEqual(self._claim("ree-cloud-4"), "ok")
        db.upsert_heartbeat(self._conn, "ree-cloud-4", state="running",
                            current_exq=self.QID, progress=None, gpu=None)
        db.record_shutdown_notice(self._conn, "ree-cloud-4",
                                  reason="scaler_idle_after_grace")
        self.assertEqual(self._claim("ree-cloud-3"), "already_claimed")

    def test_runner_restart_is_not_reaped(self):
        # `systemctl restart` -> runner_signal_exit. Restart=always brings
        # it back; the exiting runner already released its own claim. If it
        # did NOT release (claim still held), reaping would race the
        # restarted runner for a run it is about to resume.
        self._orphan(reason="runner_signal_exit")
        self.assertEqual(self._claim("ree-cloud-3"), "already_claimed")

    def test_powered_back_on_owner_is_not_reaped(self):
        # The scaler clears the fence on poweron, which also cancels
        # departure -- closing the window between power-on and the first
        # heartbeat, when last_seen is still ancient.
        self._orphan()
        db.record_claim_fence_clear(self._conn, "ree-cloud-4")
        self.assertEqual(self._claim("ree-cloud-3"), "already_claimed")

    def test_default_reap_quiet_seconds_is_off(self):
        # Both claim entry points default to 0, so every existing caller --
        # and the shadow comparison, which must stay apples-to-apples with
        # the git path -- is unchanged unless the reaper is wired in.
        self._orphan()
        self.assertEqual(
            db.try_claim(self._conn, self.QID, "ree-cloud-3"),
            "already_claimed")
        self.assertEqual(
            db.evaluate_claim(self._conn, self.QID, "ree-cloud-3"),
            "already_claimed")

    def test_reaped_claim_is_not_handed_to_a_draining_machine(self):
        # The fence is evaluated first and is machine-scoped. Recovering an
        # orphan must not become a way to hand it straight to the next box
        # on its way out -- that is the original incident again, one hop
        # along.
        self._orphan()
        db.record_shutdown_notice(self._conn, "ree-cloud-3",
                                  reason="scaler_idle_after_grace")
        self.assertEqual(
            self._claim("ree-cloud-3", fence=self.FENCE), "draining")
        self.assertEqual(self._row()["claimed_by_machine"], "ree-cloud-4")
        # A healthy box still gets it.
        self.assertEqual(
            self._claim("ree-cloud-2", fence=self.FENCE), "ok")

    def test_owner_may_still_reclaim_its_own_departed_looking_row(self):
        # A machine that reads as departed but comes back and asks for the
        # item it already holds gets 'ok' rather than being locked out of
        # its own claim.
        self._orphan()
        self.assertEqual(self._claim("ree-cloud-4"), "ok")
        self.assertEqual(self._row()["claimed_by_machine"], "ree-cloud-4")


# ---------------------------------------------------------------------------
# 3. End-to-end via the real app.py, MODE=coordinator
# ---------------------------------------------------------------------------

class ReaperHTTPTest(unittest.TestCase):

    QUIET = 900

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="reaper_http_")
        cls._dbpath = os.path.join(cls._tmp, "c.db")
        cls._tokens = os.path.join(cls._tmp, "tokens.json")
        with open(cls._tokens, "w", encoding="utf-8") as fh:
            json.dump({
                "tok-cloud-4": "ree-cloud-4",
                "tok-cloud-3": "ree-cloud-3",
                "tok-scaler": "scaler",
            }, fh)
        db.init_db(cls._dbpath)
        conn = db.connect(cls._dbpath)
        for qid in ("V3-EXQ-841", "V3-EXQ-842"):
            db.upsert_experiment(conn, {
                "queue_id": qid,
                "script": "experiments/%s.py" % qid.lower(),
                "priority": 50,
                "machine_affinity": "any",
                "status": "pending",
                "estimated_minutes": 600,
            })
        conn.close()
        cls._port = _free_port()
        env = dict(os.environ)
        env.update({
            "COORDINATOR_DB": cls._dbpath,
            "COORDINATOR_TOKENS_FILE": cls._tokens,
            "COORDINATOR_BIND_HOST": "127.0.0.1",
            "COORDINATOR_BIND_PORT": str(cls._port),
            "COORDINATOR_MODE": "coordinator",
            "COORDINATOR_CLAIM_REAP_QUIET_SECONDS": str(cls.QUIET),
        })
        cls._proc = subprocess.Popen(
            [sys.executable, str(HERE / "app.py")],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)
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
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _backdate_heartbeat(self, machine, minutes):
        """Age the row out of band -- the server holds no cached state, it
        re-reads the heartbeats table on every claim."""
        conn = db.connect(self._dbpath)
        try:
            conn.execute(
                "UPDATE heartbeats SET last_seen=? WHERE machine=?",
                (_iso(_now() - timedelta(minutes=minutes)), machine))
        finally:
            conn.close()

    def test_orphaned_claim_incident_end_to_end(self):
        # 07:08:39Z -- ree-cloud-4 claims a 600-minute experiment.
        st, jb = _http("POST", self._base + "/claim", token="tok-cloud-4",
                       body={"queue_id": "V3-EXQ-841",
                             "machine": "ree-cloud-4"})
        self.assertEqual(st, 200)
        self.assertEqual(jb["verdict"], "ok")

        # It heartbeats through its drain, the scaler announces, box off.
        st, _ = _http("POST", self._base + "/heartbeat", token="tok-cloud-4",
                      body={"machine": "ree-cloud-4", "state": "running",
                            "current_exq": "V3-EXQ-841"})
        self.assertEqual(st, 200)
        st, _ = _http("POST", self._base + "/shutdown_notify",
                      token="tok-scaler",
                      body={"machine": "ree-cloud-4",
                            "reason": "scaler_idle_after_grace"})
        self.assertEqual(st, 200)

        # Before the quiet window elapses the claim is still ree-cloud-4's.
        st, jb = _http("POST", self._base + "/claim", token="tok-cloud-3",
                       body={"queue_id": "V3-EXQ-841",
                             "machine": "ree-cloud-3"})
        self.assertEqual(jb["verdict"], "already_claimed")

        # 19 minutes of silence later it is reapable -- against the ~10h
        # the stale_hours floor alone would have imposed.
        self._backdate_heartbeat("ree-cloud-4", 19)
        st, jb = _http("POST", self._base + "/claim", token="tok-cloud-3",
                       body={"queue_id": "V3-EXQ-841",
                             "machine": "ree-cloud-3"})
        self.assertEqual(st, 200)
        self.assertTrue(jb["authoritative"])
        self.assertEqual(jb["verdict"], "ok")

    def test_status_surfaces_departed(self):
        # Own machine label: these tests share one server process.
        machine = "ree-reaper-status"
        _http("POST", self._base + "/heartbeat", token="tok-scaler",
              body={"machine": machine, "state": "idle"})
        _http("POST", self._base + "/shutdown_notify", token="tok-scaler",
              body={"machine": machine, "reason": "scaler_idle_after_grace"})
        self._backdate_heartbeat(machine, 30)
        st, jb = _http("GET", self._base + "/shadow/status",
                       token="tok-scaler")
        self.assertEqual(st, 200)
        machines = {m["machine"]: m for m in jb["machines"]}
        self.assertTrue(machines[machine]["departed"])

    def test_status_does_not_report_a_live_machine_as_departed(self):
        machine = "ree-reaper-live"
        _http("POST", self._base + "/heartbeat", token="tok-scaler",
              body={"machine": machine, "state": "running",
                    "current_exq": "V3-EXQ-842"})
        st, jb = _http("GET", self._base + "/shadow/status",
                       token="tok-scaler")
        machines = {m["machine"]: m for m in jb["machines"]}
        self.assertFalse(machines[machine]["departed"])


if __name__ == "__main__":
    unittest.main()
