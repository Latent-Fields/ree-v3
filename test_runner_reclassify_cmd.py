"""Unit tests for the runner_remote_control 'reclassify' command kind.

Covers:
  - reclassify mutates status['completed'] in-place and rewrites the
    local runner_status file.
  - missing args.queue_id -> failed cmd, no mutation.
  - missing args.result -> failed cmd, no mutation.
  - invalid args.result (e.g. "ok") -> failed cmd, no mutation.
  - queue_id not in status['completed'] -> failed cmd, no mutation.
  - status_ref omitted (older caller) -> failed cmd with clear error.
  - idempotent: result already matches -> ok=True, no rewrite.
  - end-to-end via process_pending_commands: an inbound reclassify cmd
    flips the entry's result on the in-memory status AND on the
    on-disk status file.
  - write_status_fn raising -> failed cmd, in-memory state rolled back.

ASCII-only.
"""

import json
import pathlib
import shutil
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import runner_remote_control as rrc  # noqa: E402


def _seed_status(queue_id="V3-EXQ-517c", result="ERROR"):
    return {
        "schema_version": "v1",
        "machine": "DLAPTOP-4.local",
        "last_updated": "2026-05-30T19:00:00Z",
        "idle": True,
        "current": None,
        "queue": [],
        "completed": [
            {
                "queue_id": queue_id,
                "backlog_id": "",
                "claim_id": "MECH-302",
                "title": "MECH-302 relief-completion discriminative pair",
                "description": "",
                "result": result,
                "result_summary": "No runner sentinel emitted ...",
                "started_at": "2026-05-30T12:45:55Z",
                "completed_at": "2026-05-30T14:31:01Z",
                "output_file": "",
                "completed_by": "DLAPTOP-4.local",
                "actual_secs": 6301.7,
            },
        ],
    }


class _CaptureWriter:
    """Stand-in for experiment_runner.write_status. Records calls and
    persists to a file so we can assert atomic-rewrite semantics."""

    def __init__(self):
        self.calls = 0
        self.last_path = None

    def __call__(self, status, path):
        self.calls += 1
        self.last_path = path
        path.write_text(json.dumps(status, indent=2))


# ---------------------------------------------------------------------------
# _apply_reclassify unit tests
# ---------------------------------------------------------------------------

class ApplyReclassify(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="rrc_reclassify_unit_")
        self._status_path = pathlib.Path(self._tmp) / "status.json"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_happy_path_flips_error_to_pass(self):
        status = _seed_status(result="ERROR")
        writer = _CaptureWriter()
        ok, note = rrc._apply_reclassify(
            status, self._status_path, writer,
            {"queue_id": "V3-EXQ-517c", "result": "PASS",
             "output_file": "evidence/experiments/v3_exq_517c_...json",
             "note": "manifest PASS; sentinel-detection miss"},
            "DLAPTOP-4.local")
        self.assertTrue(ok)
        self.assertIn("V3-EXQ-517c", note)
        # In-memory mutation
        entry = status["completed"][0]
        self.assertEqual(entry["result"], "PASS")
        self.assertIn("Reclassified ERROR->PASS", entry["result_summary"])
        self.assertEqual(entry["output_file"],
                         "evidence/experiments/v3_exq_517c_...json")
        # Disk rewrite
        self.assertEqual(writer.calls, 1)
        on_disk = json.loads(self._status_path.read_text())
        self.assertEqual(on_disk["completed"][0]["result"], "PASS")

    def test_missing_queue_id(self):
        status = _seed_status()
        writer = _CaptureWriter()
        ok, note = rrc._apply_reclassify(
            status, self._status_path, writer,
            {"result": "PASS"}, "host")
        self.assertFalse(ok)
        self.assertIn("queue_id", note)
        self.assertEqual(writer.calls, 0)
        self.assertEqual(status["completed"][0]["result"], "ERROR")

    def test_missing_result(self):
        status = _seed_status()
        writer = _CaptureWriter()
        ok, note = rrc._apply_reclassify(
            status, self._status_path, writer,
            {"queue_id": "V3-EXQ-517c"}, "host")
        self.assertFalse(ok)
        self.assertIn("result", note)
        self.assertEqual(writer.calls, 0)

    def test_invalid_result_value(self):
        status = _seed_status()
        writer = _CaptureWriter()
        ok, note = rrc._apply_reclassify(
            status, self._status_path, writer,
            {"queue_id": "V3-EXQ-517c", "result": "MAYBE"}, "host")
        self.assertFalse(ok)
        self.assertIn("invalid args.result", note)
        self.assertEqual(writer.calls, 0)
        self.assertEqual(status["completed"][0]["result"], "ERROR")

    def test_queue_id_not_in_completed(self):
        status = _seed_status()
        writer = _CaptureWriter()
        ok, note = rrc._apply_reclassify(
            status, self._status_path, writer,
            {"queue_id": "V3-EXQ-NONESUCH", "result": "PASS"}, "host")
        self.assertFalse(ok)
        self.assertIn("no completed entry", note)
        self.assertEqual(writer.calls, 0)

    def test_missing_status_context(self):
        # Caller is on an older API that didn't pass status_ref / path /
        # writer. Should fail loudly, not silently no-op.
        ok, note = rrc._apply_reclassify(
            None, None, None,
            {"queue_id": "V3-EXQ-517c", "result": "PASS"}, "host")
        self.assertFalse(ok)
        self.assertIn("did not pass status context", note)

    def test_idempotent_when_result_already_matches(self):
        status = _seed_status(result="PASS")
        writer = _CaptureWriter()
        ok, note = rrc._apply_reclassify(
            status, self._status_path, writer,
            {"queue_id": "V3-EXQ-517c", "result": "PASS"}, "host")
        self.assertTrue(ok)
        self.assertIn("no-op", note)
        self.assertEqual(writer.calls, 0)
        # Summary unchanged (no fresh "Reclassified ..." line).
        self.assertEqual(status["completed"][0]["result_summary"],
                         "No runner sentinel emitted ...")

    def test_write_failure_rolls_back(self):
        status = _seed_status()

        def boom(status, path):  # noqa: ARG001
            raise OSError("disk full")

        ok, note = rrc._apply_reclassify(
            status, self._status_path, boom,
            {"queue_id": "V3-EXQ-517c", "result": "PASS"}, "host")
        self.assertFalse(ok)
        self.assertIn("write_status failed", note)
        # Rolled back to original.
        self.assertEqual(status["completed"][0]["result"], "ERROR")


# ---------------------------------------------------------------------------
# process_pending_commands end-to-end
# ---------------------------------------------------------------------------

class ProcessPendingReclassify(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="rrc_reclassify_e2e_")
        # Mock REE_assembly layout: <tmp>/asm/evidence/experiments/
        # runner_commands/<machine>.json
        self._asm = pathlib.Path(self._tmp) / "asm"
        (self._asm / "evidence" / "experiments"
         / "runner_commands").mkdir(parents=True)
        (self._asm / "evidence" / "experiments"
         / "runner_heartbeats").mkdir(parents=True)
        self._machine = "DLAPTOP-4.local"
        self._status_path = (
            self._asm / "evidence" / "experiments"
            / "runner_status" / f"{self._machine}.json")
        self._status_path.parent.mkdir(parents=True)
        self._status = _seed_status()
        self._status_path.write_text(json.dumps(self._status, indent=2))
        self._queue_file = pathlib.Path(self._tmp) / "experiment_queue.json"
        self._queue_file.write_text(
            '{"schema_version": "v1", "items": []}\n')

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _seed_cmd(self, kind, args):
        cmds_path = (
            self._asm / "evidence" / "experiments"
            / "runner_commands" / f"{self._machine}.json")
        cmds_path.write_text(json.dumps({
            "schema_version": "v1",
            "machine": self._machine,
            "commands": [{
                "id": "cmd-1",
                "kind": kind,
                "args": args,
                "status": "pending",
                "issued_at_utc": "2026-05-30T23:00:00Z",
            }],
        }, indent=2))
        return cmds_path

    def _writer(self):
        captured = {"calls": 0}

        def w(status, path):
            captured["calls"] += 1
            path.write_text(json.dumps(status, indent=2))

        return w, captured

    def test_reclassify_command_flips_entry_and_persists(self):
        self._seed_cmd("reclassify", {
            "queue_id": "V3-EXQ-517c",
            "result": "PASS",
            "note": "manifest PASS",
        })
        write_status_fn, captured = self._writer()
        processed = rrc.process_pending_commands(
            self._asm, self._machine, self._queue_file,
            drain_flag=[], pause_flag=[], force_stop_flag=[],
            suspend_flag=[], resume_run_target=[],
            current_proc=[], auto_sync=False,
            status_ref=self._status,
            status_path=self._status_path,
            write_status_fn=write_status_fn,
        )
        # One command processed, ok=True.
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0]["status"], "done")
        # In-memory mutated.
        self.assertEqual(self._status["completed"][0]["result"], "PASS")
        # write_status_fn invoked.
        self.assertEqual(captured["calls"], 1)
        # Status file on disk reflects the change.
        on_disk = json.loads(self._status_path.read_text())
        self.assertEqual(on_disk["completed"][0]["result"], "PASS")

    def test_reclassify_command_without_status_context_fails_gracefully(self):
        # Caller forgot to pass status_ref / path / writer. The command
        # should fail with a clear note, no crash.
        self._seed_cmd("reclassify", {
            "queue_id": "V3-EXQ-517c", "result": "PASS",
        })
        processed = rrc.process_pending_commands(
            self._asm, self._machine, self._queue_file,
            drain_flag=[], pause_flag=[], force_stop_flag=[],
            suspend_flag=[], resume_run_target=[],
            current_proc=[], auto_sync=False,
        )
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0]["status"], "failed")
        self.assertIn("status context", processed[0]["result_note"])

    def test_reclassify_command_with_bad_args(self):
        self._seed_cmd("reclassify", {
            "queue_id": "V3-EXQ-517c",
            "result": "garbage",
        })
        write_status_fn, captured = self._writer()
        processed = rrc.process_pending_commands(
            self._asm, self._machine, self._queue_file,
            drain_flag=[], pause_flag=[], force_stop_flag=[],
            suspend_flag=[], resume_run_target=[],
            current_proc=[], auto_sync=False,
            status_ref=self._status,
            status_path=self._status_path,
            write_status_fn=write_status_fn,
        )
        self.assertEqual(processed[0]["status"], "failed")
        self.assertEqual(captured["calls"], 0)
        # No mutation
        self.assertEqual(self._status["completed"][0]["result"], "ERROR")


class CommandKindRegistered(unittest.TestCase):

    def test_reclassify_in_valid_kinds(self):
        self.assertIn("reclassify", rrc.VALID_COMMAND_KINDS)


if __name__ == "__main__":
    unittest.main()
