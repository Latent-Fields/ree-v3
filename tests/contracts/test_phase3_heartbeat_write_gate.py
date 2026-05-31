"""Contract: PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE flag.

The 2026-05-28 cutover surfaced a hub-co-location bug: on ree-cloud-1
(which runs BOTH the writer hub services AND a worker runner), the
runner's local heartbeat file write dirties the same REE_assembly
checkout the writer is trying to publish from. The workaround at
cutover time was to `systemctl disable ree-runner` on cloud-1,
costing ~20% of fleet capacity.

The fix is PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE -- a third env knob
that, parallel to the existing _PUSH gates, ALSO suppresses local
file writes for heartbeats + commands so the runner stops dirtying
the writer's checkout. Setting this flag on the hub only is what
re-enables cloud-1 as a worker.

Contracts verified:

  C1: WRITE flag default OFF -- legacy behaviour unchanged.
  C2: WRITE flag ON -- write_heartbeat returns None and writes
      no file.
  C3: WRITE flag ON -- write_commands_file returns None and writes
      no file.
  C4: WRITE flag ON implies PUSH flag effective -- push_heartbeat
      short-circuits the same as if _PUSH were set.
  C5: Gate is case-insensitive and accepts 1 / true / yes.
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


PHASE3_KEY = "PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE"
PHASE3_PUSH_KEY = "PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH"


def _reset_log_flags():
    """Reset module-level once-per-process log latches between tests."""
    rrc._HEARTBEAT_GATE_LOGGED[0] = False
    rrc._HEARTBEAT_WRITE_GATE_LOGGED[0] = False
    rrc._HEARTBEAT_WRITE_GATE_NON_HUB_REFUSED_LOGGED[0] = False


class TestHeartbeatWriteGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ree_assembly = Path(self.tmp.name) / "REE_assembly"
        (self.ree_assembly / "evidence" / "experiments"
         / "runner_heartbeats").mkdir(parents=True)
        (self.ree_assembly / "evidence" / "experiments"
         / "runner_commands").mkdir(parents=True)
        self._orig_env = {
            k: os.environ.get(k)
            for k in (PHASE3_KEY, PHASE3_PUSH_KEY)
        }
        for k in (PHASE3_KEY, PHASE3_PUSH_KEY):
            os.environ.pop(k, None)
        # 2026-05-31 hub-only self-guard: gate refuses on non-hub hostnames.
        # This test class exercises the gate-ON behaviour, so pretend we are
        # the hub. Restored in tearDown.
        self._orig_gethostname = rrc.socket.gethostname
        rrc.socket.gethostname = lambda: "ree-cloud-1"
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

    # ------ C1 default OFF preserves legacy behaviour ------------------

    def test_c1_write_gate_default_off_writes_file(self):
        path = rrc.write_heartbeat(
            self.ree_assembly,
            machine="test-machine",
            state="idle",
        )
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text())
        self.assertEqual(payload["machine"], "test-machine")
        self.assertEqual(payload["state"], "idle")

    # ------ C2 WRITE flag ON suppresses heartbeat file write -----------

    def test_c2_write_gate_on_suppresses_heartbeat_file_write(self):
        os.environ[PHASE3_KEY] = "1"
        path = rrc.write_heartbeat(
            self.ree_assembly,
            machine="test-machine",
            state="idle",
        )
        self.assertIsNone(path)
        hb_dir = (self.ree_assembly / "evidence" / "experiments"
                  / "runner_heartbeats")
        self.assertEqual(list(hb_dir.glob("*.json")), [])

    # ------ C3 WRITE flag ON suppresses commands file write -----------

    def test_c3_write_gate_on_suppresses_commands_file_write(self):
        os.environ[PHASE3_KEY] = "1"
        result = rrc.write_commands_file(
            self.ree_assembly, "test-machine",
            {"commands": [{"id": "x", "kind": "stop", "status": "pending"}]},
        )
        self.assertIsNone(result)
        cmd_dir = (self.ree_assembly / "evidence" / "experiments"
                   / "runner_commands")
        self.assertEqual(list(cmd_dir.glob("*.json")), [])

    # ------ C4 WRITE implies PUSH effectively -------------------------

    def test_c4_write_implies_push_short_circuit(self):
        # With WRITE on but PUSH explicitly unset, the push gate must
        # still report active so push_heartbeat short-circuits.
        os.environ[PHASE3_KEY] = "1"
        self.assertTrue(rrc._phase3_heartbeat_gated())

    def test_c4_pure_push_only_default(self):
        # With PUSH on but WRITE unset, write_heartbeat should still
        # write the local file (the PUSH gate doesn't suppress writes).
        os.environ[PHASE3_PUSH_KEY] = "1"
        path = rrc.write_heartbeat(
            self.ree_assembly,
            machine="test-machine",
            state="idle",
        )
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

    # ------ C5 truthy values accepted, falsy ignored ------------------

    def test_c5_truthy_values_enable_gate(self):
        for val in ("1", "true", "TRUE", "yes", "Yes"):
            with self.subTest(val=val):
                _reset_log_flags()
                os.environ[PHASE3_KEY] = val
                self.assertTrue(rrc._phase3_heartbeat_write_gated())

    def test_c5_falsy_values_leave_gate_off(self):
        for val in ("0", "false", "no", "", "False"):
            with self.subTest(val=val):
                _reset_log_flags()
                os.environ[PHASE3_KEY] = val
                self.assertFalse(rrc._phase3_heartbeat_write_gated())


if __name__ == "__main__":
    unittest.main()
