"""Tests for the 2026-05-28 cutover-fix companion bugs in experiment_runner.

Bug 1: legacy claim push (attempt_claim / release_claim) was not gated
under PHASE3_DISABLE_RUNNER_QUEUE_PUSH, so even with the flag set the
runner pushed `claim: V3-EXQ-XXX -> <machine>` commits directly to
ree-v3 origin/main from every worker (observed during the cutover --
e.g. 0743b04, a09836c). Coordinator /claim already owns the race in
Phase 3; the git push is legacy.

Bug 2: queue_file.write_text() is a non-atomic open+truncate+write, so
a concurrent `git pull` reader could see a partial file. During the
2026-05-28 cutover, cloud-2's experiment_queue.json ended up with
literal git conflict markers embedded mid-file, preflight failed with
json.decoder.JSONDecodeError, and systemd Restart=on-failure +
StartLimitBurst=5 took the unit dead. Recovery required reset --hard.

Both fixes verified here:
  Bug 1 -- attempt_claim and release_claim with the gate set:
    * subprocess.run is NEVER invoked (no git pull / add / commit / push)
    * the local queue file IS updated atomically (claimed_by populated /
      cleared) so the runner's next tick reads the right state
  Bug 2 -- _atomic_write_queue:
    * intermediate tmp file is used and replaced atomically
    * readers only ever see the old file or the new file, never partial
"""

import importlib
import json
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
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def _make_queue(tmpdir: Path, queue_id: str = "V3-EXQ-TEST",
                claimed_by=None) -> Path:
    """Write a minimal valid queue file and return its path."""
    qf = tmpdir / "experiment_queue.json"
    item = {
        "queue_id": queue_id,
        "script": "experiments/v3_exq_test.py",
        "title": "test",
        "machine_affinity": "any",
        "estimated_minutes": 1,
        "status": "pending",
    }
    if claimed_by is not None:
        item["claimed_by"] = claimed_by
    qf.write_text(json.dumps({"items": [item]}, indent=2) + "\n")
    return qf


class Phase3GateClaimPush(unittest.TestCase):
    """With PHASE3_DISABLE_RUNNER_QUEUE_PUSH=1, attempt_claim must NOT
    invoke any git subprocess but MUST update the local queue file."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_claim_")
        self._dir = Path(self._tmp)
        self._qf = _make_queue(self._dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("PHASE3_DISABLE_RUNNER_QUEUE_PUSH", None)

    def test_gate_on_attempt_claim_no_subprocess_but_writes_local(self):
        os.environ["PHASE3_DISABLE_RUNNER_QUEUE_PUSH"] = "1"
        er = _reimport("experiment_runner")
        with patch.object(er.subprocess, "run") as mock_run:
            result = er.attempt_claim(self._qf, "V3-EXQ-TEST", "host-A")
        self.assertEqual(result, "ok")
        self.assertEqual(
            mock_run.call_count, 0,
            "attempt_claim must NOT invoke git subprocess under "
            "PHASE3_DISABLE_RUNNER_QUEUE_PUSH (legacy claim-push bug)")
        # Local file IS updated atomically with the claim.
        data = json.loads(self._qf.read_text())
        item = data["items"][0]
        self.assertEqual(item["status"], "claimed")
        self.assertIsNotNone(item.get("claimed_by"))
        self.assertEqual(item["claimed_by"]["machine"], "host-A")

    def test_gate_off_attempt_claim_does_invoke_subprocess(self):
        # Sanity: outside Phase 3 the legacy git-as-mutex path still fires.
        os.environ.pop("PHASE3_DISABLE_RUNNER_QUEUE_PUSH", None)
        er = _reimport("experiment_runner")
        with patch.object(er.subprocess, "run") as mock_run:
            # Mock returncode=0 so the function thinks pulls/commits succeeded.
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""
            er.attempt_claim(self._qf, "V3-EXQ-TEST", "host-A")
        self.assertGreater(
            mock_run.call_count, 0,
            "attempt_claim must invoke git subprocess when gate is off "
            "(legacy git-as-mutex path)")

    def test_gate_on_release_claim_no_subprocess_but_clears_local(self):
        # Seed with a claim owned by host-A; release should clear it locally
        # without invoking git.
        self._qf = _make_queue(
            self._dir, claimed_by={"machine": "host-A",
                                   "claimed_at": "2026-05-28T00:00:00Z"})
        os.environ["PHASE3_DISABLE_RUNNER_QUEUE_PUSH"] = "1"
        er = _reimport("experiment_runner")
        with patch.object(er.subprocess, "run") as mock_run:
            er.release_claim(self._qf, "V3-EXQ-TEST", "host-A")
        self.assertEqual(
            mock_run.call_count, 0,
            "release_claim must NOT invoke git subprocess under "
            "PHASE3_DISABLE_RUNNER_QUEUE_PUSH (legacy release-push bug)")
        data = json.loads(self._qf.read_text())
        item = data["items"][0]
        self.assertIsNone(item.get("claimed_by"))
        self.assertEqual(item["status"], "pending")


class AtomicQueueWrite(unittest.TestCase):
    """Bug 2: _atomic_write_queue uses tmp + os.replace so partial reads
    cannot expose a half-written or git-conflict-marker-bearing file."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="atomic_write_")
        self._dir = Path(self._tmp)
        self._qf = self._dir / "experiment_queue.json"
        self._qf.write_text(json.dumps({"items": []}, indent=2) + "\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_uses_tmp_file_and_replaces(self):
        er = _reimport("experiment_runner")
        # Patch Path.replace on the *tmp* path object to capture the swap.
        captured = {}
        real_replace = Path.replace

        def fake_replace(self_path, target):
            captured["src"] = str(self_path)
            captured["dst"] = str(target)
            return real_replace(self_path, target)

        with patch.object(Path, "replace", fake_replace):
            er._atomic_write_queue(self._qf, {"items": [{"queue_id": "X"}]})
        self.assertTrue(captured["src"].endswith(".tmp"),
                        f"expected tmp source, got {captured['src']!r}")
        self.assertEqual(captured["dst"], str(self._qf))
        # Final file contains the new content.
        data = json.loads(self._qf.read_text())
        self.assertEqual(data["items"][0]["queue_id"], "X")
        # Tmp file does not linger after replace.
        self.assertFalse((self._dir / "experiment_queue.json.tmp").exists())

    def test_concurrent_reader_never_sees_partial_file(self):
        """Concurrent reads during many atomic writes must always parse
        cleanly as JSON. This is the structural guarantee that the
        2026-05-28 cloud-2 crashloop needed: even if the writer is
        interrupted between truncate and write completion (the
        non-atomic write_text failure mode), a reader must never
        observe a malformed file.

        We simulate the race by interleaving writer ticks (50 distinct
        contents) with reader ticks and asserting every read is either
        the previous content or a new content -- never a JSONDecodeError.
        """
        er = _reimport("experiment_runner")
        import threading

        stop = threading.Event()
        errors = []

        def reader():
            while not stop.is_set():
                try:
                    data = json.loads(self._qf.read_text())
                    # Sanity invariant: items is always a list.
                    if not isinstance(data.get("items"), list):
                        errors.append(f"bad shape: {data!r}")
                except (json.JSONDecodeError, OSError) as e:
                    errors.append(repr(e))

        t = threading.Thread(target=reader)
        t.start()
        try:
            for i in range(200):
                er._atomic_write_queue(
                    self._qf,
                    {"items": [{"queue_id": f"V3-EXQ-{i:04d}",
                                "status": "pending"}]})
        finally:
            stop.set()
            t.join(timeout=5)
        self.assertEqual(
            errors, [],
            "atomic writes must not expose partial state to concurrent "
            "readers (Bug 2 -- 2026-05-28 cloud-2 crashloop signature)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
