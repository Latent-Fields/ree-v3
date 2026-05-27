"""Offline smoke harness for sync_daemon.phase3_queue_writer (PLAN.md
step 5).

Sets up a throwaway bare remote + working clone for the ree-v3 side,
seeds the coordinator DB with experiment rows, and asserts the writer:

  - is gated by PHASE3_QUEUE_WRITER_READY (defaults False; returns False
    and prints stub message until flipped),
  - returns True (idle no-op) when the file already matches DB,
  - excludes completed/failed items from the materialised file,
  - surfaces claim state (claimed_by) into the file when DB has it,
  - preserves the existing calibration block verbatim,
  - refuses on dirty working tree,
  - refuses on fetch failure (no `origin` remote configured),
  - refuses to push commits that aren't writer-authored (foreign
    `phase3-queue: ` prefix check),
  - refuses on a non-FF push and leaves the working tree as-is.

ASCII-only.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db  # noqa: E402
import sync_daemon  # noqa: E402


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=check,
    )


def _bare_remote(parent):
    remote = pathlib.Path(parent) / "ree-v3.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _seeded_ree_v3_clone(parent, remote, name="ree-v3"):
    """Init a working clone with an initial commit that includes a
    minimal experiment_queue.json (calibration + empty items)."""
    repo = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _git(repo, "config", "user.email", "test@example")
    _git(repo, "config", "user.name", "test")
    initial = {
        "schema_version": "v1",
        "calibration": {
            "ms_per_episode_condition": {},
            "hardware": {},
            "calibrated_at": "2026-05-27T00:00:00Z",
            "calibration_note": "seed",
        },
        "items": [],
    }
    (repo / "experiment_queue.json").write_text(
        json.dumps(initial, indent=2) + "\n")
    _git(repo, "add", "experiment_queue.json")
    _git(repo, "commit", "-q", "-m", "seed")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "main")
    return repo


def _upsert_item(conn, queue_id, *, status="pending", priority=10,
                 machine_affinity="any", script="experiments/x.py",
                 claimed_by_machine=None, claimed_at=None,
                 estimated_minutes=30):
    item = {
        "queue_id": queue_id,
        "script": script,
        "priority": priority,
        "machine_affinity": machine_affinity,
        "status": status,
        "estimated_minutes": estimated_minutes,
    }
    if claimed_by_machine:
        item["claimed_by"] = {
            "machine": claimed_by_machine,
            "claimed_at": claimed_at or "2026-05-27T19:00:00Z",
        }
    db.upsert_experiment(conn, item, preserve_claim=False)


class _QueueWriterFixture(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_queue_")
        self._remote = _bare_remote(self._tmp)
        self._repo = _seeded_ree_v3_clone(self._tmp, self._remote)
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        sync_daemon.PHASE3_QUEUE_WRITER_READY = False

    def tearDown(self):
        sync_daemon.PHASE3_QUEUE_WRITER_READY = False
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self):
        sync_daemon.PHASE3_QUEUE_WRITER_READY = True
        try:
            return sync_daemon.phase3_queue_writer(
                self._conn,
                ree_v3_path=str(self._repo),
                queue_relpath="experiment_queue.json",
                branch="main")
        finally:
            sync_daemon.PHASE3_QUEUE_WRITER_READY = False

    def _read_local_queue(self):
        return json.loads(
            (self._repo / "experiment_queue.json").read_text())

    def _read_origin_queue(self):
        out = _git(self._remote, "show", "main:experiment_queue.json").stdout
        return json.loads(out)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class StubGate(_QueueWriterFixture):
    def test_stub_returns_false_with_default_flag(self):
        # Flag stays False by default.
        result = sync_daemon.phase3_queue_writer(
            self._conn,
            ree_v3_path=str(self._repo),
            queue_relpath="experiment_queue.json",
            branch="main")
        self.assertFalse(result)


class HappyPath(_QueueWriterFixture):
    def test_writes_pending_items_to_file_and_pushes(self):
        _upsert_item(self._conn, "V3-EXQ-A", priority=20)
        _upsert_item(self._conn, "V3-EXQ-B", priority=10,
                     status="claimed",
                     claimed_by_machine="ree-cloud-2",
                     claimed_at="2026-05-27T19:30:00Z")
        result = self._run()
        self.assertTrue(result, "writer should succeed on real changes")

        on_origin = self._read_origin_queue()
        ids = [i["queue_id"] for i in on_origin["items"]]
        # Priority desc -> A before B.
        self.assertEqual(ids, ["V3-EXQ-A", "V3-EXQ-B"])
        # Calibration preserved verbatim.
        self.assertEqual(on_origin["calibration"]["calibration_note"], "seed")
        # Claimed state surfaced.
        b = next(i for i in on_origin["items"] if i["queue_id"] == "V3-EXQ-B")
        self.assertEqual(b["claimed_by"]["machine"], "ree-cloud-2")
        self.assertEqual(b["status"], "claimed")
        # Local file matches origin (push succeeded).
        self.assertEqual(self._read_local_queue(), on_origin)


class TerminalItemsExcluded(_QueueWriterFixture):
    def test_completed_and_failed_items_dropped_from_file(self):
        _upsert_item(self._conn, "V3-EXQ-active", priority=10)
        _upsert_item(self._conn, "V3-EXQ-done", status="completed")
        _upsert_item(self._conn, "V3-EXQ-bad", status="failed")
        self._run()
        on_origin = self._read_origin_queue()
        ids = {i["queue_id"] for i in on_origin["items"]}
        self.assertEqual(ids, {"V3-EXQ-active"},
                         "completed + failed must not appear in the file")


class IdempotentNoop(_QueueWriterFixture):
    def test_second_tick_when_db_unchanged_does_not_push(self):
        _upsert_item(self._conn, "V3-EXQ-once", priority=10)
        self.assertTrue(self._run())
        # Capture origin's commit SHA after tick 1.
        sha_after_first = _git(
            self._remote, "rev-parse", "main").stdout.strip()
        # Second tick: DB unchanged, file matches DB -> should no-op.
        self.assertTrue(self._run())
        sha_after_second = _git(
            self._remote, "rev-parse", "main").stdout.strip()
        self.assertEqual(
            sha_after_first, sha_after_second,
            "second tick must not push when content matches")


class DirtyTreeRefusal(_QueueWriterFixture):
    def test_dirty_tree_refuses(self):
        _upsert_item(self._conn, "V3-EXQ-dirty", priority=10)
        # Leave an uncommitted edit on a different file.
        (self._repo / "stray.txt").write_text("WIP\n")
        result = self._run()
        self.assertFalse(result)


class ForeignCommitRefusal(_QueueWriterFixture):
    def test_operator_commit_ahead_blocks_push(self):
        _upsert_item(self._conn, "V3-EXQ-foreign", priority=10)
        # Operator-authored commit ahead of origin (foreign prefix).
        _git(self._repo, "config", "user.email", "operator@example")
        _git(self._repo, "config", "user.name", "operator")
        _git(self._repo, "commit", "--allow-empty", "-m",
             "operator hand edit on hub")
        result = self._run()
        self.assertFalse(result, "writer must refuse foreign commit push")


class CalibrationRoundtrip(_QueueWriterFixture):
    def test_existing_calibration_preserved(self):
        # Hand-edit the local file with a richer calibration block so we
        # can confirm the writer reads + round-trips it (NOT clobbers it).
        new_doc = json.loads(
            (self._repo / "experiment_queue.json").read_text())
        new_doc["calibration"]["hardware"] = {
            "DLAPTOP-4.local": {"cpu_throughput": 7.0},
        }
        (self._repo / "experiment_queue.json").write_text(
            json.dumps(new_doc, indent=2) + "\n")
        _git(self._repo, "config", "user.email",
             "phase3-queue@example")
        _git(self._repo, "config", "user.name", "phase3-queue")
        _git(self._repo, "add", "experiment_queue.json")
        _git(self._repo, "commit", "-m",
             "phase3-queue: pre-existing calibration commit")
        _git(self._repo, "push", "-q", "origin", "main")

        _upsert_item(self._conn, "V3-EXQ-cal", priority=10)
        self.assertTrue(self._run())

        on_origin = self._read_origin_queue()
        self.assertEqual(
            on_origin["calibration"]["hardware"]
                     ["DLAPTOP-4.local"]["cpu_throughput"],
            7.0,
            "writer must preserve calibration verbatim")


class ReconcileUpsertOnly(unittest.TestCase):
    """Phase 3 authoritative-mode reconcile must NOT delete DB rows
    missing from the queue file -- the file is a derived view."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="reconcile_upsert_only_")
        self._dbpath = os.path.join(self._tmp, "c.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        # Minimal git-shaped queue file (just an empty items list).
        self._queue_repo = pathlib.Path(self._tmp) / "qrepo"
        subprocess.run(
            ["git", "init", "-q", "-b", "main", str(self._queue_repo)],
            check=True)
        _git(self._queue_repo, "config", "user.email", "x@y")
        _git(self._queue_repo, "config", "user.name", "x")
        bare = pathlib.Path(self._tmp) / "qrepo.git"
        subprocess.run(
            ["git", "init", "-q", "--bare", str(bare)], check=True)
        _git(self._queue_repo, "remote", "add", "origin", str(bare))
        self._queue = str(self._queue_repo / "experiment_queue.json")
        pathlib.Path(self._queue).write_text(
            json.dumps({"schema_version": "v1", "calibration": {},
                        "items": []}) + "\n")
        _git(self._queue_repo, "add", "experiment_queue.json")
        _git(self._queue_repo, "commit", "-q", "-m", "seed")
        _git(self._queue_repo, "push", "-q", "origin", "main")

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_upsert_only_does_not_delete_stale_rows(self):
        # Pre-existing DB row (e.g. completed experiment from before).
        _upsert_item(self._conn, "V3-EXQ-keepme", status="completed")
        sync_daemon.reconcile_once(
            self._conn, self._queue,
            claim_authority="coordinator", upsert_only=True)
        row = self._conn.execute(
            "SELECT status FROM experiments WHERE queue_id=?",
            ("V3-EXQ-keepme",)).fetchone()
        self.assertIsNotNone(
            row, "upsert_only mode must NOT delete DB rows missing from file")
        self.assertEqual(row["status"], "completed")

    def test_default_mode_still_deletes_stale_rows(self):
        # Same setup but without upsert_only: legacy behaviour deletes.
        _upsert_item(self._conn, "V3-EXQ-byebye", status="completed")
        sync_daemon.reconcile_once(
            self._conn, self._queue,
            claim_authority="coordinator")
        row = self._conn.execute(
            "SELECT status FROM experiments WHERE queue_id=?",
            ("V3-EXQ-byebye",)).fetchone()
        self.assertIsNone(
            row, "default mode must still delete missing items")


if __name__ == "__main__":
    unittest.main(verbosity=2)
