"""Contract: Phase 3 command-channel migration (commands via coordinator).

The legacy remote-control command channel is the per-machine git file
REE_assembly/evidence/experiments/runner_commands/<host>.json. serve.py writes
it; the runner reads/acks it via runner_remote_control.process_pending_commands.
That git-file dependency is why PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE is hub-only
(it ALSO gates the command-file writeback; a worker that cannot persist a stop
command's status->done restart-loops -- incident 2026-05-30).

This migration moves the command channel onto the coordinator:
  - PHASE3_COMMANDS_VIA_COORDINATOR=1 -- fetch + ack commands via the coordinator
    IN ADDITION to the git file (dual-read; the git file stays the proven
    fallback during transition).
  - PHASE3_COMMANDS_OFF_GIT=1 -- the coordinator is the SOLE channel; the worker
    neither reads nor writes the git command-file. Implies VIA_COORDINATOR.
    Self-guards: refuses (falls back to git) when the coordinator command
    channel is unavailable, so the worker is never left uncontrollable.

Both default OFF -> bit-identical to the pre-migration git-only behaviour.

Contracts verified:
  C1: both gates default OFF -- git command-file is the sole channel; the
      coordinator is never consulted.
  C2: VIA_COORDINATOR dual-read -- a coordinator command is executed AND acked
      via the coordinator, while the git file is still processed.
  C3: OFF_GIT -- the git command-file is NOT read or written; a coordinator
      command is executed + acked; a pending git-file command is left untouched.
  C4: OFF_GIT self-guard -- with the coordinator channel unavailable, the gate
      refuses and the git command-file remains the channel (worker stays
      controllable).
  C5: OFF_GIT implies VIA_COORDINATOR and requires channel availability.
  C6: _HEARTBEAT_WRITE relaxation -- refused on a non-hub worker by default, but
      ALLOWED when PHASE3_COMMANDS_OFF_GIT is active (the restart-loop hazard is
      gone because acks persist in the coordinator).
  C7: truthy/falsy parsing of the two new env flags.
"""

import os
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

import runner_remote_control as rrc  # noqa: E402

VIA_KEY = "PHASE3_COMMANDS_VIA_COORDINATOR"
OFF_GIT_KEY = "PHASE3_COMMANDS_OFF_GIT"
MODE_KEY = "COORDINATION_MODE"
WRITE_KEY = "PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE"
_KEYS = (VIA_KEY, OFF_GIT_KEY, MODE_KEY, WRITE_KEY)


class _FakeCoordClient:
    """Stand-in for coordinator_client with a controllable command queue."""

    def __init__(self, enabled=True, pending=None):
        self._enabled = enabled
        self._pending = list(pending or [])
        self.acked = []  # (id, machine, status, note)

    def enabled(self):
        return self._enabled

    def fetch_commands(self, machine):
        if not self._enabled:
            return None
        return {"machine": machine, "commands": list(self._pending)}

    def ack_command(self, command_id, machine, result_status="done",
                    result_note=None):
        self.acked.append((command_id, machine, result_status, result_note))
        # Mirror server behaviour: acked commands no longer fetched.
        self._pending = [c for c in self._pending if c.get("id") != command_id]
        return {"ok": True, "applied": True, "note": "acked"}


def _reset_log_flags():
    for name in (
        "_HEARTBEAT_GATE_LOGGED", "_HEARTBEAT_WRITE_GATE_LOGGED",
        "_HEARTBEAT_WRITE_GATE_NON_HUB_REFUSED_LOGGED",
        "_HEARTBEAT_WRITE_GATE_OFF_GIT_ALLOWED_LOGGED",
        "_TELEMETRY_OFF_GIT_GATE_LOGGED",
        "_COMMANDS_VIA_COORD_GATE_LOGGED",
        "_COMMANDS_OFF_GIT_GATE_LOGGED",
        "_COMMANDS_OFF_GIT_REFUSED_LOGGED",
    ):
        getattr(rrc, name)[0] = False


class TestCommandChannelCoordinator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ree_assembly = Path(self.tmp.name) / "REE_assembly"
        (self.ree_assembly / "evidence" / "experiments"
         / "runner_commands").mkdir(parents=True)
        self.queue_file = Path(self.tmp.name) / "experiment_queue.json"
        self.queue_file.write_text('{"items": []}\n')
        self._orig_env = {k: os.environ.get(k) for k in _KEYS}
        for k in _KEYS:
            os.environ.pop(k, None)
        self._orig_gethostname = rrc.socket.gethostname
        rrc.socket.gethostname = lambda: "ree-cloud-2"
        self._orig_cc = rrc.coordinator_client
        _reset_log_flags()
        self.machine = "ree-cloud-2"

    def tearDown(self):
        for k, v in self._orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        rrc.socket.gethostname = self._orig_gethostname
        rrc.coordinator_client = self._orig_cc
        _reset_log_flags()
        self.tmp.cleanup()

    # ---- helpers ------------------------------------------------------
    def _flags(self):
        return {
            "drain_flag": [], "pause_flag": [], "force_stop_flag": [],
            "suspend_flag": [], "resume_run_target": [], "current_proc": [],
        }

    def _process(self, **flags):
        return rrc.process_pending_commands(
            self.ree_assembly, self.machine, self.queue_file, **flags)

    def _enable_coordinator(self, pending=None, enabled=True):
        os.environ[MODE_KEY] = "coordinator"
        fake = _FakeCoordClient(enabled=enabled, pending=pending)
        rrc.coordinator_client = fake
        return fake

    # ---- C1 default OFF: git-only -------------------------------------
    def test_c1_default_off_git_only(self):
        fake = self._enable_coordinator(
            pending=[{"id": 7, "kind": "stop", "args": "{}"}])
        # but DO NOT set the via/off-git flags -> coordinator must be ignored.
        self.assertFalse(rrc._phase3_commands_via_coordinator_gated())
        self.assertFalse(rrc._phase3_commands_off_git_gated())
        rrc.append_command(self.ree_assembly, self.machine, "pause")
        flags = self._flags()
        self._process(**flags)
        # git command executed; coordinator never acked.
        self.assertEqual(flags["pause_flag"], [True])
        self.assertEqual(fake.acked, [])
        data = rrc.read_commands_file(self.ree_assembly, self.machine)
        self.assertEqual(data["commands"][0]["status"], "done")

    # ---- C2 VIA_COORDINATOR dual-read ---------------------------------
    def test_c2_via_coordinator_dual_read(self):
        fake = self._enable_coordinator(
            pending=[{"id": 11, "kind": "stop", "args": "{}"}])
        os.environ[VIA_KEY] = "1"
        self.assertTrue(rrc._phase3_commands_via_coordinator_gated())
        # git-file command present too.
        rrc.append_command(self.ree_assembly, self.machine, "pause")
        flags = self._flags()
        self._process(**flags)
        # coordinator stop -> drain; git pause -> pause. Both executed.
        self.assertEqual(flags["drain_flag"], [True])
        self.assertEqual(flags["pause_flag"], [True])
        self.assertEqual(fake.acked, [(11, self.machine, "done", "drain requested")])
        # git file command marked done.
        data = rrc.read_commands_file(self.ree_assembly, self.machine)
        self.assertEqual(data["commands"][0]["kind"], "pause")
        self.assertEqual(data["commands"][0]["status"], "done")

    # ---- C3 OFF_GIT: coordinator sole channel -------------------------
    def test_c3_off_git_coordinator_sole_channel(self):
        fake = self._enable_coordinator(
            pending=[{"id": 21, "kind": "stop", "args": "{}"}])
        os.environ[OFF_GIT_KEY] = "1"
        self.assertTrue(rrc._phase3_commands_off_git_gated())
        # A pending git-file command must NOT be processed under off-git.
        rrc.append_command(self.ree_assembly, self.machine, "pause")
        flags = self._flags()
        self._process(**flags)
        # coordinator stop executed + acked; git pause left pending.
        self.assertEqual(flags["drain_flag"], [True])
        self.assertEqual(flags["pause_flag"], [])  # git file NOT read
        self.assertEqual(fake.acked, [(21, self.machine, "done", "drain requested")])
        data = rrc.read_commands_file(self.ree_assembly, self.machine)
        self.assertEqual(data["commands"][0]["status"], "pending")

    # ---- C4 OFF_GIT self-guard ----------------------------------------
    def test_c4_off_git_refuses_without_coordinator(self):
        # Flag set but coordinator unavailable -> refuse, keep git channel.
        os.environ[OFF_GIT_KEY] = "1"
        os.environ.pop(MODE_KEY, None)  # COORDINATION_MODE=git default
        rrc.coordinator_client = _FakeCoordClient(enabled=False)
        self.assertFalse(rrc._phase3_commands_off_git_gated())
        rrc.append_command(self.ree_assembly, self.machine, "pause")
        flags = self._flags()
        self._process(**flags)
        # git command executed (worker stays controllable).
        self.assertEqual(flags["pause_flag"], [True])
        data = rrc.read_commands_file(self.ree_assembly, self.machine)
        self.assertEqual(data["commands"][0]["status"], "done")

    # ---- C5 OFF_GIT requires availability; VIA requires availability --
    def test_c5_gates_require_channel_availability(self):
        os.environ[VIA_KEY] = "1"
        os.environ[OFF_GIT_KEY] = "1"
        # No coordinator configured -> both gates OFF.
        rrc.coordinator_client = _FakeCoordClient(enabled=False)
        self.assertFalse(rrc._phase3_commands_via_coordinator_gated())
        self.assertFalse(rrc._phase3_commands_off_git_gated())
        # Configure it -> both gates ON.
        _reset_log_flags()
        self._enable_coordinator()
        self.assertTrue(rrc._phase3_commands_via_coordinator_gated())
        self.assertTrue(rrc._phase3_commands_off_git_gated())

    # ---- C6 _HEARTBEAT_WRITE relaxation on non-hub under OFF_GIT -------
    def test_c6_heartbeat_write_relaxed_under_off_git(self):
        # Non-hub worker (ree-cloud-2 per setUp). Default: refused.
        os.environ[WRITE_KEY] = "1"
        self.assertFalse(rrc._phase3_heartbeat_write_gated())
        # With OFF_GIT active + coordinator available: ALLOWED.
        _reset_log_flags()
        self._enable_coordinator()
        os.environ[OFF_GIT_KEY] = "1"
        self.assertTrue(rrc._phase3_commands_off_git_gated())
        self.assertTrue(rrc._phase3_heartbeat_write_gated())

    def test_c6b_hub_unaffected(self):
        # Hub hostname always allowed regardless of off-git.
        rrc.socket.gethostname = lambda: "ree-worker-1"
        os.environ[WRITE_KEY] = "1"
        self.assertTrue(rrc._phase3_heartbeat_write_gated())

    # ---- C7 truthy / falsy parsing ------------------------------------
    def test_c7_truthy_falsy(self):
        self._enable_coordinator()
        for val in ("1", "true", "TRUE", "yes", "Yes"):
            with self.subTest(val=val):
                _reset_log_flags()
                os.environ[VIA_KEY] = val
                self.assertTrue(rrc._phase3_commands_via_coordinator_gated())
        for val in ("0", "false", "no", "", "False"):
            with self.subTest(val=val):
                _reset_log_flags()
                os.environ[VIA_KEY] = val
                self.assertFalse(rrc._phase3_commands_via_coordinator_gated())


if __name__ == "__main__":
    unittest.main()
