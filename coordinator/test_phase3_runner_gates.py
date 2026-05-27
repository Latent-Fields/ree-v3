"""Tests for the Phase 3 runner-push gating env flags.

The runner has four git push paths that fight sync_daemon's phase3_git_writer
for the index when both are active:

  experiment_runner.git_push_queue       (-> ree-v3 main)
  experiment_runner.git_push_results     (-> REE_assembly master)
  experiment_runner.git_push_status      (-> REE_assembly master)
  runner_remote_control.push_heartbeat   (-> REE_assembly master)
  runner_remote_control.push_commands    (-> REE_assembly master)

Three env flags gate them:

  PHASE3_DISABLE_RUNNER_RESULT_PUSH      -> git_push_results
  PHASE3_DISABLE_RUNNER_QUEUE_PUSH       -> git_push_queue
  PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH   -> push_heartbeat + push_commands +
                                            git_push_status (status and
                                            heartbeats land via the same
                                            sync_daemon step 6)

Each test verifies the gate is a true no-op: the underlying `subprocess.run`
is never invoked. We assert by intercepting subprocess.run at the module
level. ASCII-only output.
"""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def _reimport(name):
    """Re-import a runner module fresh so module-level state (e.g. the
    one-shot heartbeat-gate log) doesn't leak across tests."""
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


class GatesEnvOff(unittest.TestCase):
    """Sanity: when no gate env vars are set, the gate predicate returns
    False. This protects against accidental default-on regressions."""

    def setUp(self):
        for k in ("PHASE3_DISABLE_RUNNER_RESULT_PUSH",
                  "PHASE3_DISABLE_RUNNER_QUEUE_PUSH",
                  "PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH"):
            os.environ.pop(k, None)

    def test_predicate_false_when_unset(self):
        er = _reimport("experiment_runner")
        self.assertFalse(er._phase3_gate("PHASE3_DISABLE_RUNNER_RESULT_PUSH"))
        self.assertFalse(er._phase3_gate("PHASE3_DISABLE_RUNNER_QUEUE_PUSH"))

    def test_predicate_accepts_1_true_yes(self):
        er = _reimport("experiment_runner")
        for val in ("1", "true", "TRUE", "yes", "Yes"):
            os.environ["PHASE3_DISABLE_RUNNER_RESULT_PUSH"] = val
            self.assertTrue(
                er._phase3_gate("PHASE3_DISABLE_RUNNER_RESULT_PUSH"),
                "%r should be truthy" % val)
        for val in ("0", "false", "no", ""):
            os.environ["PHASE3_DISABLE_RUNNER_RESULT_PUSH"] = val
            self.assertFalse(
                er._phase3_gate("PHASE3_DISABLE_RUNNER_RESULT_PUSH"),
                "%r should be falsy" % val)
        os.environ.pop("PHASE3_DISABLE_RUNNER_RESULT_PUSH", None)


class ResultPushGate(unittest.TestCase):
    """PHASE3_DISABLE_RUNNER_RESULT_PUSH=1 must make git_push_results a
    no-op: no subprocess.run calls fire (no git invocations at all)."""

    def test_gate_off_invokes_subprocess(self):
        os.environ.pop("PHASE3_DISABLE_RUNNER_RESULT_PUSH", None)
        er = _reimport("experiment_runner")
        with patch.object(er.subprocess, "run") as mock_run:
            # Need a CompletedProcess shape for the diff check; default
            # MagicMock has .returncode = MagicMock(), so set it explicit.
            mock_run.return_value.returncode = 0  # "no diff" branch -> return
            er.git_push_results(Path("/tmp/fake"), ["fake.json"])
        # At least the git add + diff calls happened.
        self.assertGreater(mock_run.call_count, 0,
                           "subprocess.run must be invoked when gate is off")

    def test_gate_on_skips_all_subprocess(self):
        os.environ["PHASE3_DISABLE_RUNNER_RESULT_PUSH"] = "1"
        try:
            er = _reimport("experiment_runner")
            with patch.object(er.subprocess, "run") as mock_run:
                er.git_push_results(Path("/tmp/fake"), ["fake.json"])
            self.assertEqual(
                mock_run.call_count, 0,
                "subprocess.run must NOT be invoked when result-push gate is on")
        finally:
            os.environ.pop("PHASE3_DISABLE_RUNNER_RESULT_PUSH", None)


class QueuePushGate(unittest.TestCase):
    def test_gate_on_skips_all_subprocess(self):
        os.environ["PHASE3_DISABLE_RUNNER_QUEUE_PUSH"] = "1"
        try:
            er = _reimport("experiment_runner")
            with patch.object(er.subprocess, "run") as mock_run:
                er.git_push_queue()
            self.assertEqual(mock_run.call_count, 0)
        finally:
            os.environ.pop("PHASE3_DISABLE_RUNNER_QUEUE_PUSH", None)


class StatusPushGate(unittest.TestCase):
    """git_push_status is grouped with the heartbeat gate -- both
    publish via the same sync_daemon step 6 once wired."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_status_")
        self._asm = Path(self._tmp)
        self._status = self._asm / "runner_status" / "test.json"
        self._status.parent.mkdir(parents=True, exist_ok=True)
        self._status.write_text("{}")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_gate_on_skips_all_subprocess(self):
        os.environ["PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH"] = "1"
        try:
            er = _reimport("experiment_runner")
            with patch.object(er.subprocess, "run") as mock_run:
                er.git_push_status(self._asm, self._status, "V3-EXQ-TEST")
            self.assertEqual(mock_run.call_count, 0)
        finally:
            os.environ.pop("PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH", None)


class HeartbeatAndCommandsGate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_hb_")
        self._asm = Path(self._tmp)
        self._hb = self._asm / "evidence" / "experiments" / \
            "runner_heartbeats" / "test.json"
        self._hb.parent.mkdir(parents=True, exist_ok=True)
        self._hb.write_text("{}")
        # Stop the one-shot log being a stale-state contaminant.
        rrc = _reimport("runner_remote_control")
        rrc._HEARTBEAT_GATE_LOGGED[0] = False

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH", None)

    def test_heartbeat_gate_on_skips_all_subprocess(self):
        os.environ["PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH"] = "1"
        rrc = _reimport("runner_remote_control")
        with patch.object(rrc.subprocess, "run") as mock_run:
            rrc.push_heartbeat(self._asm, self._hb)
        self.assertEqual(mock_run.call_count, 0)

    def test_commands_gate_on_skips_all_subprocess(self):
        os.environ["PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH"] = "1"
        rrc = _reimport("runner_remote_control")
        cmds = self._asm / "evidence" / "experiments" / \
            "runner_commands" / "test.json"
        cmds.parent.mkdir(parents=True, exist_ok=True)
        cmds.write_text("{}")
        with patch.object(rrc.subprocess, "run") as mock_run:
            rrc.push_commands(self._asm, cmds)
        self.assertEqual(mock_run.call_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
