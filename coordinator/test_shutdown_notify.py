"""Tests for the shutdown_notify endpoint + lifecycle_state derivation.

Three test classes:

  LifecycleStateTest    -- pure function, no HTTP, no DB write.
  ShutdownNoticeDBTest  -- record_shutdown_notice round-trip + idempotency.
  ShutdownNotifyHTTPTest -- end-to-end via the real app.py entrypoint.

All printed text is ASCII-only.
"""

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
from datetime import datetime, timedelta, timezone

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db  # noqa: E402


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 1. Pure function: db.lifecycle_state
# ---------------------------------------------------------------------------

class LifecycleStateTest(unittest.TestCase):

    LIVE = 300         # 5 min
    STALE_AFTER = 7 * 86400  # 7 days

    def _state(self, last_seen, last_shutdown_at):
        return db.lifecycle_state(
            last_seen, last_shutdown_at,
            live_threshold_seconds=self.LIVE,
            stale_after_seconds=self.STALE_AFTER)

    def test_live_when_heartbeat_fresh(self):
        seen = _iso(_now() - timedelta(seconds=30))
        self.assertEqual(self._state(seen, None), "live")

    def test_live_takes_precedence_over_old_shutdown(self):
        # Machine shut down, then came back up and heartbeated again.
        # The freshest signal is the heartbeat; lifecycle is live.
        seen = _iso(_now() - timedelta(seconds=30))
        shutdown = _iso(_now() - timedelta(hours=1))
        self.assertEqual(self._state(seen, shutdown), "live")

    def test_gracefully_offline_when_shutdown_after_heartbeat(self):
        # Shutdown_notify arrived after the last heartbeat -> intentional
        # offline.
        seen = _iso(_now() - timedelta(hours=2))
        shutdown = _iso(_now() - timedelta(hours=1))
        self.assertEqual(
            self._state(seen, shutdown), "gracefully_offline")

    def test_gracefully_offline_when_no_heartbeat_ever(self):
        # First contact for a machine is a shutdown announcement (e.g. the
        # scaler posting on its behalf before the box ever booted). We
        # treat that as graceful too -- the operator's affirmative signal
        # is present, just from a different source.
        shutdown = _iso(_now() - timedelta(hours=3))
        self.assertEqual(
            self._state(None, shutdown), "gracefully_offline")

    def test_stale_when_no_heartbeat_no_shutdown(self):
        self.assertEqual(self._state(None, None), "stale")

    def test_stale_when_heartbeat_old_and_no_shutdown(self):
        # Silent disappearance -- exactly the case the operator should
        # care about.
        seen = _iso(_now() - timedelta(hours=2))
        self.assertEqual(self._state(seen, None), "stale")

    def test_stale_after_watchdog_window_expires(self):
        # Shutdown WAS graceful, but it's been > 7 days. The machine never
        # came back; escalate to stale regardless of prior intent.
        seen = _iso(_now() - timedelta(days=10))
        shutdown = _iso(_now() - timedelta(days=8))
        self.assertEqual(self._state(seen, shutdown), "stale")

    def test_malformed_timestamps_treated_as_missing(self):
        self.assertEqual(self._state("garbage", None), "stale")
        self.assertEqual(self._state(None, "garbage"), "stale")
        self.assertEqual(self._state("garbage", "garbage"), "stale")

    def test_just_inside_live_threshold(self):
        # 5s inside the live window -- avoids sub-second clock skew at the
        # exact boundary (the boundary itself is implementation-defined and
        # not load-bearing).
        seen = _iso(_now() - timedelta(seconds=self.LIVE - 5))
        self.assertEqual(self._state(seen, None), "live")

    def test_boundary_just_past_live_threshold_with_shutdown(self):
        # Just past live, but a fresh shutdown -> gracefully_offline.
        seen = _iso(_now() - timedelta(seconds=self.LIVE + 60))
        shutdown = _iso(_now() - timedelta(seconds=self.LIVE + 30))
        self.assertEqual(
            self._state(seen, shutdown), "gracefully_offline")


# ---------------------------------------------------------------------------
# 2. DB write path: record_shutdown_notice
# ---------------------------------------------------------------------------

class ShutdownNoticeDBTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="shutdown_notice_db_")
        self._dbpath = os.path.join(self._tmp, "c.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)

    def tearDown(self):
        self._conn.close()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _row(self, machine):
        return self._conn.execute(
            "SELECT machine, last_seen, last_shutdown_at, shutdown_reason, "
            "expected_wake_condition FROM heartbeats WHERE machine=?",
            (machine,)).fetchone()

    def test_creates_row_if_machine_has_no_heartbeat(self):
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4",
            reason="scaler_idle_after_grace",
            expected_wake_condition="claimable>0")
        row = self._row("ree-cloud-4")
        self.assertIsNotNone(row)
        self.assertEqual(row["shutdown_reason"], "scaler_idle_after_grace")
        self.assertEqual(row["expected_wake_condition"], "claimable>0")
        self.assertIsNotNone(row["last_shutdown_at"])

    def test_idempotent_overwrites_prior_notice(self):
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4", reason="first")
        time.sleep(1.1)  # ensure timestamp differs at 1-second resolution
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4", reason="second")
        row = self._row("ree-cloud-4")
        self.assertEqual(row["shutdown_reason"], "second")

    def test_preserves_last_seen_when_updating_existing_heartbeat(self):
        # Pre-existing heartbeat -> shutdown_notify must not clobber it.
        db.upsert_heartbeat(
            self._conn, "ree-cloud-4", state="idle",
            current_exq=None, progress=None, gpu=None)
        prior = self._row("ree-cloud-4")["last_seen"]
        time.sleep(1.1)
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4", reason="systemd_sigterm")
        after = self._row("ree-cloud-4")
        # last_seen unchanged; last_shutdown_at populated and >= last_seen
        self.assertEqual(after["last_seen"], prior)
        self.assertEqual(after["shutdown_reason"], "systemd_sigterm")
        self.assertIsNotNone(after["last_shutdown_at"])
        self.assertGreater(after["last_shutdown_at"], prior)

    def test_reason_and_wake_condition_optional(self):
        db.record_shutdown_notice(self._conn, "ree-cloud-4")
        row = self._row("ree-cloud-4")
        self.assertIsNone(row["shutdown_reason"])
        self.assertIsNone(row["expected_wake_condition"])
        self.assertIsNotNone(row["last_shutdown_at"])


# ---------------------------------------------------------------------------
# 3. HTTP integration: POST /shutdown_notify + GET /shadow/status
# ---------------------------------------------------------------------------

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


class ShutdownNotifyHTTPTest(unittest.TestCase):
    """Spins the real app.py entrypoint; mirrors test_shadow_e2e.py."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="shutdown_http_")
        cls._dbpath = os.path.join(cls._tmp, "c.db")
        cls._tokens = os.path.join(cls._tmp, "tokens.json")
        with open(cls._tokens, "w", encoding="utf-8") as fh:
            json.dump({
                "tok-cloud-4": "ree-cloud-4",
                "tok-scaler": "scaler",
            }, fh)
        cls._port = _free_port()
        env = dict(os.environ)
        env.update({
            "COORDINATOR_DB": cls._dbpath,
            "COORDINATOR_TOKENS_FILE": cls._tokens,
            "COORDINATOR_BIND_HOST": "127.0.0.1",
            "COORDINATOR_BIND_PORT": str(cls._port),
            "COORDINATOR_MODE": "shadow",
        })
        cls._proc = subprocess.Popen(
            [sys.executable, str(HERE / "app.py")],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)
        cls._base = "http://127.0.0.1:%d" % cls._port
        # Wait for liveness.
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

    def test_post_requires_auth(self):
        st, _ = _http(
            "POST", self._base + "/shutdown_notify",
            body={"machine": "ree-cloud-4"})
        self.assertEqual(st, 401)

    def test_post_writes_notice_and_status_reflects_it(self):
        st, jb = _http(
            "POST", self._base + "/shutdown_notify",
            token="tok-scaler",
            body={"machine": "ree-cloud-4",
                  "reason": "scaler_idle_after_grace",
                  "expected_wake_condition": "claimable>0"})
        self.assertEqual(st, 200)
        self.assertTrue(jb["ok"])
        self.assertEqual(jb["machine"], "ree-cloud-4")
        self.assertEqual(jb["reason"], "scaler_idle_after_grace")

        st, jb = _http("GET", self._base + "/shadow/status",
                       token="tok-scaler")
        self.assertEqual(st, 200)
        machines = {m["machine"]: m for m in jb["machines"]}
        self.assertIn("ree-cloud-4", machines)
        m = machines["ree-cloud-4"]
        self.assertEqual(m["lifecycle_state"], "gracefully_offline")
        self.assertEqual(m["shutdown_reason"], "scaler_idle_after_grace")
        self.assertEqual(
            m["expected_wake_condition"], "claimable>0")

    def test_post_requires_explicit_machine_field(self):
        # The token's machine label is NEVER substituted. A probe with no
        # machine field used to write a stray heartbeat row for the token's
        # label (e.g. the scaler token created a "scaler" row); the
        # endpoint now demands an explicit machine.
        st, jb = _http(
            "POST", self._base + "/shutdown_notify",
            token="tok-cloud-4",
            body={"reason": "systemd_sigterm"})
        self.assertEqual(st, 400)
        self.assertEqual(jb["error"], "machine required")

    def test_post_empty_body_rejected(self):
        # Empty body (zero bytes) -> 400. Previously this returned 200
        # because _json_body() coerces empty to {} and the token-fallback
        # fired, writing a stray row for the token's machine label.
        req = urllib.request.Request(
            self._base + "/shutdown_notify",
            data=b"",
            headers={"Authorization": "Bearer tok-scaler",
                     "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                self.fail("expected HTTPError, got %d" % r.status)
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_post_empty_object_body_rejected(self):
        st, jb = _http(
            "POST", self._base + "/shutdown_notify",
            token="tok-scaler", body={})
        self.assertEqual(st, 400)
        self.assertEqual(jb["error"], "machine required")

    def test_post_body_missing_machine_rejected(self):
        st, jb = _http(
            "POST", self._base + "/shutdown_notify",
            token="tok-scaler", body={"reason": "x"})
        self.assertEqual(st, 400)
        self.assertEqual(jb["error"], "machine required")

    def test_heartbeat_after_shutdown_returns_live(self):
        # Announce shutdown, then heartbeat -> machine is back; lifecycle
        # state must flip back to live.
        st, _ = _http(
            "POST", self._base + "/shutdown_notify",
            token="tok-scaler",
            body={"machine": "ree-cloud-2", "reason": "test"})
        self.assertEqual(st, 200)
        # Confirm gracefully_offline first.
        st, jb = _http("GET", self._base + "/shadow/status",
                       token="tok-scaler")
        m = {x["machine"]: x for x in jb["machines"]}["ree-cloud-2"]
        self.assertEqual(m["lifecycle_state"], "gracefully_offline")

        # Now heartbeat (must use timestamps newer than the shutdown for
        # the lifecycle math to flip; 1-second resolution suffices).
        time.sleep(1.1)
        st, _ = _http(
            "POST", self._base + "/heartbeat",
            token="tok-scaler",
            body={"machine": "ree-cloud-2", "state": "idle",
                  "current_exq": None})
        self.assertEqual(st, 200)

        st, jb = _http("GET", self._base + "/shadow/status",
                       token="tok-scaler")
        m = {x["machine"]: x for x in jb["machines"]}["ree-cloud-2"]
        self.assertEqual(m["lifecycle_state"], "live")

    def test_coordinator_client_report_shutdown_e2e(self):
        # Mirrors the runner's import path: drive coordinator_client at
        # the same level the runner does, against this live app.py.
        import importlib
        # Import from the ree-v3 root, not coordinator/. The runner imports
        # coordinator_client from its own working directory.
        root = HERE.parent
        sys.path.insert(0, str(root))
        try:
            os.environ["COORDINATION_MODE"] = "shadow"
            os.environ["COORDINATOR_URL"] = self._base
            os.environ["COORDINATOR_TOKEN"] = "tok-cloud-4"
            if "coordinator_client" in sys.modules:
                cc = importlib.reload(sys.modules["coordinator_client"])
            else:
                import coordinator_client as cc
            self.assertTrue(cc.enabled())
            r = cc.report_shutdown(
                machine="ree-cloud-runner-test",
                reason="runner_drain_complete")
            self.assertIsNotNone(r)
            self.assertTrue(r["ok"])
            self.assertEqual(r["machine"], "ree-cloud-runner-test")
            self.assertEqual(r["reason"], "runner_drain_complete")
            # And confirm /shadow/status reflects the announcement.
            st, jb = _http(
                "GET", self._base + "/shadow/status", token="tok-cloud-4")
            machines = {m["machine"]: m for m in jb["machines"]}
            self.assertIn("ree-cloud-runner-test", machines)
            self.assertEqual(
                machines["ree-cloud-runner-test"]["lifecycle_state"],
                "gracefully_offline")
        finally:
            for k in ("COORDINATION_MODE", "COORDINATOR_URL",
                      "COORDINATOR_TOKEN"):
                os.environ.pop(k, None)
            if str(root) in sys.path:
                sys.path.remove(str(root))

    def test_coordinator_client_report_shutdown_disabled_in_git_mode(self):
        # COORDINATION_MODE=git (the default for workers not yet on shadow)
        # must make report_shutdown a no-op returning None, never raising.
        import importlib
        root = HERE.parent
        sys.path.insert(0, str(root))
        try:
            os.environ["COORDINATION_MODE"] = "git"
            os.environ.pop("COORDINATOR_URL", None)
            os.environ.pop("COORDINATOR_TOKEN", None)
            if "coordinator_client" in sys.modules:
                cc = importlib.reload(sys.modules["coordinator_client"])
            else:
                import coordinator_client as cc
            self.assertFalse(cc.enabled())
            r = cc.report_shutdown(machine="x", reason="y")
            self.assertIsNone(r)
        finally:
            os.environ.pop("COORDINATION_MODE", None)
            if str(root) in sys.path:
                sys.path.remove(str(root))

    def test_bad_body_returns_400(self):
        # Bearer ok but JSON missing -> 400.
        req = urllib.request.Request(
            self._base + "/shutdown_notify",
            data=b"not json",
            headers={"Authorization": "Bearer tok-scaler",
                     "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                self.fail("expected HTTPError, got %d" % r.status)
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)


# ---------------------------------------------------------------------------
# 4. Drain-window claim fence
#
# Incident, 2026-07-30 (hub `journalctl -u cloud-scaler`): the scaler decided
# correctly at 07:00:18Z to shut ree-worker-4 down (claimable=0
# held_by_self=0 lease=none idle_ok=1 reason=clean_idle), announced the
# shutdown, and issued `hcloud server shutdown` -- an ACPI soft-off, so the
# box kept running while it drained. Its runner then CLAIMED V3-EXQ-841, a
# 600-estimated-minute experiment, at 07:08:39Z, eight minutes in, and the
# box powered off ~07:26-07:30Z holding it. The claim was not re-claimable
# until 17:08:58Z (~10h): _claim_recoverable needs BOTH stale_hours (6) and
# a stale owner heartbeat.
#
# The scaler was not at fault and could not have fixed it: HELD_BY_SELF
# worked throughout (its 07:15/07:20/07:25 votes all said "keeping
# ree-worker-4 running") and was simply undeliverable -- an in-flight ACPI
# shutdown cannot be retracted, and all three protection layers are
# decision-time-only. Hence a fence on the CLAIM path.
# ---------------------------------------------------------------------------

class ClaimFenceStateTest(unittest.TestCase):
    """Pure function: db.claim_fence_active."""

    FENCE = 1800

    def _fenced(self, shutdown, cleared=None, reason="scaler_idle_after_grace",
                fence_seconds=None):
        return db.claim_fence_active(
            shutdown, cleared, reason,
            fence_seconds=(self.FENCE if fence_seconds is None
                           else fence_seconds))

    def test_no_shutdown_notice_is_not_fenced(self):
        self.assertFalse(self._fenced(None))

    def test_fresh_scaler_shutdown_fences(self):
        self.assertTrue(self._fenced(_iso(_now() - timedelta(minutes=1))))

    def test_incident_replay_eight_minutes_into_the_drain(self):
        # The exact moment V3-EXQ-841 was claimed: 8m21s after the
        # 07:00:18Z announce. This is the case the fence exists for.
        self.assertTrue(
            self._fenced(_iso(_now() - timedelta(minutes=8, seconds=21))))

    def test_fence_window_outlasts_the_measured_drain(self):
        # The drain ran ~26 minutes. A fence that lapsed before the box
        # actually powered off would leave the tail of the window open --
        # the whole defect. Pinned against the shipped default, not FENCE.
        self.assertGreater(db.CLAIM_FENCE_DEFAULT_SECONDS, 26 * 60)
        self.assertTrue(db.claim_fence_active(
            _iso(_now() - timedelta(minutes=26)), None,
            "scaler_idle_after_grace",
            fence_seconds=db.CLAIM_FENCE_DEFAULT_SECONDS))

    def test_expired_shutdown_no_longer_fences(self):
        # Bounded so a machine woken by a MANUAL `hcloud server poweron`
        # (which never calls the clear) recovers on its own.
        self.assertFalse(self._fenced(_iso(_now() - timedelta(minutes=31))))

    def test_clear_newer_than_shutdown_disarms(self):
        shutdown = _iso(_now() - timedelta(minutes=5))
        cleared = _iso(_now() - timedelta(minutes=1))
        self.assertFalse(self._fenced(shutdown, cleared))

    def test_clear_older_than_shutdown_does_not_disarm(self):
        # A clear from a PREVIOUS wake must not disarm a NEW shutdown.
        cleared = _iso(_now() - timedelta(hours=3))
        shutdown = _iso(_now() - timedelta(minutes=2))
        self.assertTrue(self._fenced(shutdown, cleared))

    def test_same_second_tie_leaves_the_fence_armed(self):
        # These timestamps are 1-second resolution, so a collision is
        # order-ambiguous. The two errors are not symmetric: a wrong
        # refusal costs one poll tick, a wrong grant is the 10h orphan.
        # Ties must therefore stay fenced -- hence a STRICT `>` in
        # claim_fence_active, unlike lifecycle_state's `>=`.
        t = _iso(_now() - timedelta(minutes=1))
        self.assertTrue(self._fenced(t, t))

    def test_process_exit_reasons_do_not_fence(self):
        # The runner posts these itself on the way out. ree-runner.service
        # carries Restart=always, so fencing on them would stall the
        # restarted runner for the whole window on an ordinary
        # `systemctl restart` or a remote stop -- a regression, not a fix.
        fresh = _iso(_now() - timedelta(minutes=1))
        for reason in ("runner_drain_complete", "runner_signal_exit"):
            self.assertFalse(self._fenced(fresh, reason=reason), reason)

    def test_unrecognised_reason_fences(self):
        # Default-deny: an unknown reason is treated as a machine shutdown.
        fresh = _iso(_now() - timedelta(minutes=1))
        self.assertTrue(self._fenced(fresh, reason="some_future_reason"))
        self.assertTrue(self._fenced(fresh, reason=None))

    def test_zero_fence_seconds_disables(self):
        fresh = _iso(_now() - timedelta(minutes=1))
        self.assertFalse(self._fenced(fresh, fence_seconds=0))


class ClaimFenceDBTest(unittest.TestCase):
    """try_claim / evaluate_claim / record_claim_fence_clear against a
    real DB."""

    FENCE = 1800

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="claim_fence_db_")
        self._dbpath = os.path.join(self._tmp, "c.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        db.upsert_experiment(self._conn, {
            "queue_id": "V3-EXQ-841",
            "script": "experiments/v3_exq_841.py",
            "priority": 50,
            "machine_affinity": "any",
            "status": "pending",
            "estimated_minutes": 600,
        })

    def tearDown(self):
        self._conn.close()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _claim(self, machine="ree-cloud-4", fence=None):
        return db.try_claim(
            self._conn, "V3-EXQ-841", machine,
            fence_seconds=self.FENCE if fence is None else fence)

    def test_unfenced_machine_claims_normally(self):
        self.assertEqual(self._claim(), "ok")

    def test_fenced_machine_is_refused(self):
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4", reason="scaler_idle_after_grace")
        self.assertEqual(self._claim(), "draining")

    def test_refusal_does_not_mutate_the_item(self):
        # The point of the fence is that the item stays claimable by
        # someone else. A refusal that still stamped claimed_by would
        # reproduce the orphan it exists to prevent.
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4", reason="scaler_idle_after_grace")
        self.assertEqual(self._claim(), "draining")
        row = self._conn.execute(
            "SELECT status, claimed_by_machine FROM experiments "
            "WHERE queue_id=?", ("V3-EXQ-841",)).fetchone()
        self.assertEqual(row["status"], "pending")
        self.assertIsNone(row["claimed_by_machine"])

    def test_fence_is_machine_scoped(self):
        # A healthy sibling must still be able to take the work. This is
        # what turns a 10h orphan into a re-route.
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4", reason="scaler_idle_after_grace")
        self.assertEqual(self._claim(), "draining")
        self.assertEqual(self._claim(machine="ree-cloud-3"), "ok")

    def test_fence_survives_a_fresh_heartbeat(self):
        # THE load-bearing case. The runner keeps heartbeating for the
        # whole drain window, so lifecycle_state reads "live" throughout
        # and a gracefully_offline-based fence would never fire. The fence
        # must key off the pending shutdown notice, not liveness.
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4", reason="scaler_idle_after_grace")
        db.upsert_heartbeat(
            self._conn, "ree-cloud-4", state="idle",
            current_exq=None, progress=None, gpu=None)
        state = db.lifecycle_state(
            self._conn.execute(
                "SELECT last_seen FROM heartbeats WHERE machine=?",
                ("ree-cloud-4",)).fetchone()["last_seen"],
            self._conn.execute(
                "SELECT last_shutdown_at FROM heartbeats WHERE machine=?",
                ("ree-cloud-4",)).fetchone()["last_shutdown_at"],
            live_threshold_seconds=300, stale_after_seconds=7 * 86400)
        self.assertEqual(state, "live")
        self.assertEqual(self._claim(), "draining")

    def test_clear_disarms_and_claiming_resumes(self):
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4", reason="scaler_idle_after_grace")
        self.assertEqual(self._claim(), "draining")
        time.sleep(1.1)  # 1-second timestamp resolution
        db.record_claim_fence_clear(self._conn, "ree-cloud-4")
        self.assertEqual(self._claim(), "ok")

    def test_clear_preserves_last_shutdown_at(self):
        # lifecycle_state and phase3_preflight read that column: a machine
        # powered on but not yet heartbeating should stay
        # gracefully_offline for the ~90s it takes to boot, not drop
        # straight to stale.
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4", reason="scaler_idle_after_grace")
        db.record_claim_fence_clear(self._conn, "ree-cloud-4")
        row = self._conn.execute(
            "SELECT last_shutdown_at, shutdown_reason FROM heartbeats "
            "WHERE machine=?", ("ree-cloud-4",)).fetchone()
        self.assertIsNotNone(row["last_shutdown_at"])
        self.assertEqual(row["shutdown_reason"], "scaler_idle_after_grace")

    def test_clear_is_idempotent(self):
        # The scaler calls it unconditionally on every poweron.
        db.record_claim_fence_clear(self._conn, "ree-cloud-4")
        db.record_claim_fence_clear(self._conn, "ree-cloud-4")
        self.assertEqual(self._claim(), "ok")

    def test_default_fence_seconds_is_off(self):
        # Both claim entry points default to fence_seconds=0 so existing
        # callers -- and the shadow comparison, which must stay
        # apples-to-apples with the git path -- are unchanged unless the
        # fence is explicitly wired in.
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4", reason="scaler_idle_after_grace")
        self.assertEqual(
            db.try_claim(self._conn, "V3-EXQ-841", "ree-cloud-4"), "ok")

    def test_evaluate_claim_agrees_with_try_claim(self):
        db.record_shutdown_notice(
            self._conn, "ree-cloud-4", reason="scaler_idle_after_grace")
        self.assertEqual(
            db.evaluate_claim(self._conn, "V3-EXQ-841", "ree-cloud-4",
                              fence_seconds=self.FENCE),
            "draining")

    def test_machine_with_no_row_is_not_fenced(self):
        self.assertFalse(
            db.machine_claim_fenced(self._conn, "never-seen", self.FENCE))


class ClaimFenceHTTPTest(unittest.TestCase):
    """End-to-end via the real app.py, in MODE=coordinator.

    Separate from ShutdownNotifyHTTPTest because that one runs
    MODE=shadow, where the fence is deliberately off (git is the claim
    authority there, so a "draining" verdict would register as a spurious
    divergence).
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="claim_fence_http_")
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
        import shutil
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_drain_window_incident_end_to_end(self):
        # 07:00:18Z -- scaler announces, then issues hcloud shutdown.
        st, _ = _http("POST", self._base + "/shutdown_notify",
                      token="tok-scaler",
                      body={"machine": "ree-cloud-4",
                            "reason": "scaler_idle_after_grace",
                            "expected_wake_condition": "claimable>0"})
        self.assertEqual(st, 200)

        # 07:08:39Z -- the draining runner tries to take a 600-minute
        # experiment it cannot possibly finish. Refused.
        st, jb = _http("POST", self._base + "/claim", token="tok-cloud-4",
                       body={"queue_id": "V3-EXQ-841",
                             "machine": "ree-cloud-4"})
        self.assertEqual(st, 200)
        self.assertTrue(jb["authoritative"])
        self.assertEqual(jb["verdict"], "draining")

        # The item is still there for a healthy worker -- a re-route, not
        # a 10h orphan.
        st, jb = _http("POST", self._base + "/claim", token="tok-cloud-3",
                       body={"queue_id": "V3-EXQ-841",
                             "machine": "ree-cloud-3"})
        self.assertEqual(st, 200)
        self.assertEqual(jb["verdict"], "ok")

    def test_status_surfaces_claim_fenced(self):
        # Own machine label: these tests share one server process, and a
        # fence clear left by a sibling test would otherwise decide this
        # one at 1-second timestamp resolution.
        _http("POST", self._base + "/shutdown_notify", token="tok-scaler",
              body={"machine": "ree-fence-status",
                    "reason": "scaler_idle_after_grace"})
        st, jb = _http("GET", self._base + "/shadow/status",
                       token="tok-scaler")
        self.assertEqual(st, 200)
        machines = {m["machine"]: m for m in jb["machines"]}
        self.assertTrue(machines["ree-fence-status"]["claim_fenced"])

    def test_clear_endpoint_disarms(self):
        machine = "ree-fence-clear"
        _http("POST", self._base + "/shutdown_notify", token="tok-scaler",
              body={"machine": machine,
                    "reason": "scaler_idle_after_grace"})
        st, jb = _http("POST", self._base + "/claim", token="tok-scaler",
                       body={"queue_id": "V3-EXQ-842",
                             "machine": machine})
        self.assertEqual(jb["verdict"], "draining")

        time.sleep(1.1)  # 1-second timestamp resolution
        st, jb = _http("POST", self._base + "/claim_fence/clear",
                       token="tok-scaler", body={"machine": machine})
        self.assertEqual(st, 200)
        self.assertTrue(jb["ok"])

        st, jb = _http("POST", self._base + "/claim", token="tok-scaler",
                       body={"queue_id": "V3-EXQ-842",
                             "machine": machine})
        self.assertEqual(jb["verdict"], "ok")

    def test_clear_requires_auth(self):
        st, _ = _http("POST", self._base + "/claim_fence/clear",
                      body={"machine": "ree-cloud-4"})
        self.assertEqual(st, 401)

    def test_clear_requires_explicit_machine_field(self):
        # Same contract as /shutdown_notify: the token's own label is
        # never substituted, so an empty body cannot create a stray
        # heartbeat row named after the token.
        st, _ = _http("POST", self._base + "/claim_fence/clear",
                      token="tok-scaler", body={})
        self.assertEqual(st, 400)


class ScalerClearFenceWiringTest(unittest.TestCase):
    """The scaler must actually CALL the clear on poweron.

    Loads the real cloud-scaler.py by path (it lives under deploy/ and is
    not importable as a package module) and drives hcloud_poweron with
    subprocess.run stubbed, so a poweron branch that forgets the clear
    fails here rather than in production.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cloud_scaler_under_test",
            str(HERE / "deploy" / "cloud-scaler.py"))
        cls.scaler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.scaler)

    def _run_poweron(self, script_exists):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            return None

        real_run = self.scaler.subprocess.run
        real_exists = self.scaler.os.path.exists
        self.scaler.subprocess.run = fake_run
        self.scaler.os.path.exists = lambda p: script_exists
        try:
            self.scaler.hcloud_poweron(
                "ree-worker-4", affinity="ree-cloud-4",
                clear_fence_script="/usr/local/bin/clear.sh")
        finally:
            self.scaler.subprocess.run = real_run
            self.scaler.os.path.exists = real_exists
        return calls

    def test_poweron_clears_the_fence(self):
        calls = self._run_poweron(script_exists=True)
        self.assertEqual(calls[0][:3], ["hcloud", "server", "poweron"])
        self.assertEqual(calls[-1], ["/usr/local/bin/clear.sh",
                                     "ree-cloud-4"])

    def test_missing_clear_script_does_not_break_poweron(self):
        # Degraded, not broken: the fence expires on its own timer.
        calls = self._run_poweron(script_exists=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:3], ["hcloud", "server", "poweron"])

    def test_every_poweron_call_site_passes_the_clear_script(self):
        # hcloud_poweron takes affinity/clear_fence_script rather than the
        # call sites doing the clear themselves, precisely so a branch
        # cannot forget. Pin that no call site regresses to the bare form.
        src = (HERE / "deploy" / "cloud-scaler.py").read_text()
        sites = [ln for ln in src.splitlines()
                 if "hcloud_poweron(" in ln and not ln.strip().startswith(
                     ("def ", "#"))]
        self.assertGreaterEqual(len(sites), 2)
        for ln in sites:
            self.assertIn("affinity=affinity", ln)


if __name__ == "__main__":
    unittest.main(verbosity=2)
