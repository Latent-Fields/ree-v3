"""Contract: PHASE3_RUNNER_TELEMETRY_OFF_GIT flag (worker-safe telemetry gate).

Phase 3 moved telemetry TRANSPORT off git (workers POST /heartbeat + /status
to the coordinator; the hub sync_daemon is the sole git materialiser). But
workers still wrote their own runner_heartbeats/<host>.json +
runner_status/<host>.json into the shared REE_assembly checkout every tick,
which conflicts with the hub-materialised version on
`git pull --rebase --autostash` and accumulates dormant autostashes
(cloud-3 hit 43 on 2026-06-02; 191 by 2026-05-31).

PHASE3_RUNNER_TELEMETRY_OFF_GIT suppresses ONLY those in-tree telemetry FILE
writes. Crucially, unlike the hub-only PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE
(which ALSO gates the command-file writeback and therefore restart-loops a
worker -- incident 2026-05-30), this gate does NOT touch the command channel,
so it is safe on any worker (cloud-2/3/4) and carries no restart-loop risk.

Contracts verified:

  C1: gate default OFF -- legacy behaviour unchanged (file written).
  C2: gate ON on a NON-HUB worker hostname -- heartbeat file write suppressed
      (the property the hub-only gate cannot provide; _HEARTBEAT_WRITE refuses
      on non-hub hosts).
  C3: gate ON -- command-file write is NOT suppressed (the no-restart-loop
      guarantee). This is the load-bearing distinction from _HEARTBEAT_WRITE.
  C4: gate ON does NOT imply the heartbeat PUSH gate (commands/heartbeat can
      still be pushed by any path that isn't separately push-gated).
  C5: gate is case-insensitive and accepts 1 / true / yes; falsy ignored.
  C6: status-write gate (_phase3_hub_local_ree_assembly_writes_gated) fires
      under the new env on a non-hub worker.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

import runner_remote_control as rrc  # noqa: E402

TELEMETRY_KEY = "PHASE3_RUNNER_TELEMETRY_OFF_GIT"
WRITE_KEY = "PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE"
PUSH_KEY = "PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH"
_KEYS = (TELEMETRY_KEY, WRITE_KEY, PUSH_KEY)


def _reset_log_flags():
    rrc._HEARTBEAT_GATE_LOGGED[0] = False
    rrc._HEARTBEAT_WRITE_GATE_LOGGED[0] = False
    rrc._HEARTBEAT_WRITE_GATE_NON_HUB_REFUSED_LOGGED[0] = False
    rrc._TELEMETRY_OFF_GIT_GATE_LOGGED[0] = False


class TestTelemetryOffGitGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ree_assembly = Path(self.tmp.name) / "REE_assembly"
        (self.ree_assembly / "evidence" / "experiments"
         / "runner_heartbeats").mkdir(parents=True)
        (self.ree_assembly / "evidence" / "experiments"
         / "runner_commands").mkdir(parents=True)
        self._orig_env = {k: os.environ.get(k) for k in _KEYS}
        for k in _KEYS:
            os.environ.pop(k, None)
        # Pretend we are a NON-HUB worker -- the whole point of this gate is
        # that it works where PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE refuses.
        self._orig_gethostname = rrc.socket.gethostname
        rrc.socket.gethostname = lambda: "ree-cloud-2"
        _reset_log_flags()

    def tearDown(self):
        for k, v in self._orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        rrc.socket.gethostname = self._orig_gethostname
        _reset_log_flags()
        self.tmp.cleanup()

    # ------ C1 default OFF preserves legacy behaviour -----------------
    def test_c1_default_off_writes_file(self):
        self.assertFalse(rrc._phase3_telemetry_file_write_gated())
        path = rrc.write_heartbeat(self.ree_assembly,
                                   machine="ree-cloud-2", state="idle")
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

    # ------ C2 gate ON on a NON-HUB worker suppresses heartbeat write --
    def test_c2_on_worker_suppresses_heartbeat_write(self):
        os.environ[TELEMETRY_KEY] = "1"
        # Sanity: the hub-only gate would REFUSE on this non-hub hostname.
        self.assertFalse(rrc._phase3_heartbeat_write_gated())
        # The worker-safe gate fires.
        self.assertTrue(rrc._phase3_telemetry_file_write_gated())
        path = rrc.write_heartbeat(self.ree_assembly,
                                   machine="ree-cloud-2", state="idle")
        self.assertIsNone(path)
        hb_dir = (self.ree_assembly / "evidence" / "experiments"
                  / "runner_heartbeats")
        self.assertEqual(list(hb_dir.glob("*.json")), [])

    # ------ C3 gate ON does NOT suppress command-file write -----------
    def test_c3_command_file_still_written_no_restart_loop(self):
        os.environ[TELEMETRY_KEY] = "1"
        result = rrc.write_commands_file(
            self.ree_assembly, "ree-cloud-2",
            {"commands": [{"id": "x", "kind": "stop", "status": "done"}]},
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.exists())
        data = json.loads(result.read_text())
        self.assertEqual(data["commands"][0]["kind"], "stop")

    # ------ C4 gate ON does not imply the heartbeat PUSH gate ----------
    def test_c4_does_not_imply_push_gate(self):
        os.environ[TELEMETRY_KEY] = "1"
        # The telemetry gate is independent of the push gate; with only the
        # telemetry gate set, _phase3_heartbeat_gated (push) stays OFF.
        self.assertFalse(rrc._phase3_heartbeat_gated())

    # ------ C5 truthy / falsy parsing ---------------------------------
    def test_c5_truthy_values_enable(self):
        for val in ("1", "true", "TRUE", "yes", "Yes"):
            with self.subTest(val=val):
                _reset_log_flags()
                os.environ[TELEMETRY_KEY] = val
                self.assertTrue(rrc._phase3_telemetry_file_write_gated())

    def test_c5_falsy_values_disable(self):
        for val in ("0", "false", "no", "", "False"):
            with self.subTest(val=val):
                _reset_log_flags()
                os.environ[TELEMETRY_KEY] = val
                self.assertFalse(rrc._phase3_telemetry_file_write_gated())

    # ------ C6 status-write gate honours the telemetry env ------------
    def test_c6_status_write_gate_fires_on_worker(self):
        import experiment_runner as er
        os.environ[TELEMETRY_KEY] = "1"
        er._PHASE3_HUB_FILE_WRITE_GATE_LOGGED = False
        self.assertTrue(er._phase3_hub_local_ree_assembly_writes_gated())
        os.environ.pop(TELEMETRY_KEY, None)
        er._PHASE3_HUB_FILE_WRITE_GATE_LOGGED = False
        self.assertFalse(er._phase3_hub_local_ree_assembly_writes_gated())


if __name__ == "__main__":
    unittest.main()
