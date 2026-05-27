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


if __name__ == "__main__":
    unittest.main(verbosity=2)
