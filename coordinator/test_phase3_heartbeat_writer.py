"""Offline smoke harness for sync_daemon.phase3_heartbeat_writer
(PLAN.md step 6).

Covers:
  - Stub gate (PHASE3_HEARTBEAT_WRITER_READY=False -> no-op return False).
  - Happy path: DB row with both heartbeat_payload_json and
    status_payload_json populated -> writer materialises both files,
    commits, pushes to the bare remote.
  - Idempotent no-op: second tick with unchanged payloads emits no
    additional commit.
  - NULL payloads skipped: row with both columns NULL doesn't generate
    a file (legacy clients haven't sent rich payload yet).
  - Unsafe machine name skipped.
  - Malformed JSON in stored column skipped + warned.
  - Dirty tree refusal.
  - Foreign-commit refusal (prefix `phase3-heartbeats: `).
  - DB helper integration: upsert_heartbeat(payload_json=...) and
    record_status_payload write the right column.

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
    remote = pathlib.Path(parent) / "asm.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _seeded_clone(parent, remote, *, name="asm",
                  seed_msg="phase3-heartbeats: seed"):
    """Seed clone with a writer-authored seed commit so the foreign-
    commit check doesn't trip on the seed in tests that don't
    intentionally probe that branch."""
    repo = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)],
                   check=True)
    _git(repo, "config", "user.email", "test@example")
    _git(repo, "config", "user.name", "test")
    (repo / "README.md").write_text("seed\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", seed_msg)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "master")
    return repo


def _sample_heartbeat_payload(machine, *, state="running", current_exq="V3-EXQ-X"):
    return {
        "machine": machine,
        "hostname": "host-" + machine,
        "last_tick_utc": "2026-05-27T22:00:00Z",
        "state": state,
        "current_exq": current_exq,
        "progress": {"runs_done": 1, "runs_total": 3},
        "gpu": {},
        "recent_completed": [
            {"queue_id": "V3-EXQ-prev", "result": "PASS",
             "completed_at": "2026-05-27T21:00:00Z"},
        ],
    }


def _sample_status_payload(machine, *, last_completed="V3-EXQ-X"):
    return {
        "schema_version": "v1",
        "machine": machine,
        "runner_pid": 12345,
        "last_updated": "2026-05-27T22:00:00Z",
        "idle": False,
        "current": {"queue_id": last_completed},
        "queue": [],
        "completed": [{"queue_id": last_completed, "result": "PASS"}],
    }


class _Fixture(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_hb_writer_")
        self._remote = _bare_remote(self._tmp)
        self._repo = _seeded_clone(self._tmp, self._remote)
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        sync_daemon.PHASE3_HEARTBEAT_WRITER_READY = False

    def tearDown(self):
        sync_daemon.PHASE3_HEARTBEAT_WRITER_READY = False
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self):
        sync_daemon.PHASE3_HEARTBEAT_WRITER_READY = True
        try:
            return sync_daemon.phase3_heartbeat_writer(
                self._conn,
                ree_assembly_path=str(self._repo),
                branch="master")
        finally:
            sync_daemon.PHASE3_HEARTBEAT_WRITER_READY = False

    def _seed_heartbeat(self, machine, **payload_kwargs):
        payload = _sample_heartbeat_payload(machine, **payload_kwargs)
        db.upsert_heartbeat(
            self._conn, machine,
            state=payload["state"],
            current_exq=payload["current_exq"],
            progress=payload["progress"], gpu=payload["gpu"],
            payload_json=json.dumps(payload))
        return payload

    def _seed_status(self, machine, **payload_kwargs):
        payload = _sample_status_payload(machine, **payload_kwargs)
        db.record_status_payload(
            self._conn, machine, json.dumps(payload))
        return payload

    def _origin_blob(self, relpath):
        out = _git(self._remote, "show",
                   "master:" + relpath, check=False)
        if out.returncode != 0:
            return None
        return out.stdout


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class StubGate(_Fixture):
    def test_default_flag_returns_false(self):
        # Even with seeded payloads, the gate must hold.
        self._seed_heartbeat("ree-cloud-1")
        result = sync_daemon.phase3_heartbeat_writer(
            self._conn,
            ree_assembly_path=str(self._repo),
            branch="master")
        self.assertFalse(result)


class HappyPath(_Fixture):
    def test_writes_both_files_and_pushes(self):
        hb_a = self._seed_heartbeat("ree-cloud-1", current_exq="V3-EXQ-A")
        st_a = self._seed_status("ree-cloud-1", last_completed="V3-EXQ-A")
        self.assertTrue(self._run())

        hb_on_origin = self._origin_blob(
            "evidence/experiments/runner_heartbeats/ree-cloud-1.json")
        self.assertIsNotNone(hb_on_origin)
        self.assertEqual(json.loads(hb_on_origin), hb_a)

        st_on_origin = self._origin_blob(
            "evidence/experiments/runner_status/ree-cloud-1.json")
        self.assertIsNotNone(st_on_origin)
        self.assertEqual(json.loads(st_on_origin), st_a)


class IdempotentNoop(_Fixture):
    def test_second_tick_no_new_commit(self):
        self._seed_heartbeat("ree-cloud-1")
        self.assertTrue(self._run())
        sha1 = _git(self._remote, "rev-parse", "master").stdout.strip()
        self.assertTrue(self._run())  # nothing changed in DB
        sha2 = _git(self._remote, "rev-parse", "master").stdout.strip()
        self.assertEqual(sha1, sha2,
                         "unchanged payloads must not emit a new commit")


class LegacyClientsSkipped(_Fixture):
    """A row with both payload columns NULL (old client that doesn't send
    rich payload yet) must not generate any file."""

    def test_row_with_no_payloads_no_files(self):
        # Use upsert_heartbeat with payload_json=None (legacy default).
        db.upsert_heartbeat(
            self._conn, "ree-cloud-legacy",
            state="running", current_exq="V3-EXQ-Z",
            progress={}, gpu={})
        result = self._run()
        # Returns True (idle no-op) -- no rows to write.
        self.assertTrue(result)
        # No file produced.
        self.assertIsNone(self._origin_blob(
            "evidence/experiments/runner_heartbeats/"
            "ree-cloud-legacy.json"))


class UnsafeMachineNameSkipped(_Fixture):
    def test_unsafe_name_does_not_write(self):
        # Directly inject a row with an unsafe machine name.
        self._conn.execute(
            "INSERT INTO heartbeats "
            "(machine, last_seen, heartbeat_payload_json) "
            "VALUES (?, ?, ?)",
            ("../escape",
             "2026-05-27T22:00:00Z",
             json.dumps({"machine": "../escape"})))
        # Also a clean row so the writer has SOMETHING to ship.
        self._seed_heartbeat("ree-cloud-2")
        self.assertTrue(self._run())
        # Good machine got its file.
        self.assertIsNotNone(self._origin_blob(
            "evidence/experiments/runner_heartbeats/ree-cloud-2.json"))
        # Unsafe one did NOT, anywhere.
        files_on_origin = _git(
            self._remote, "ls-tree", "-r", "--name-only", "master",
        ).stdout
        self.assertNotIn("..", files_on_origin)
        self.assertNotIn("escape", files_on_origin)


class MalformedJsonSkipped(_Fixture):
    def test_invalid_json_does_not_block_other_rows(self):
        # Inject garbage into one row's payload column.
        self._conn.execute(
            "INSERT INTO heartbeats "
            "(machine, last_seen, heartbeat_payload_json) "
            "VALUES (?, ?, ?)",
            ("ree-cloud-bad", "2026-05-27T22:00:00Z", "{not json"))
        self._seed_heartbeat("ree-cloud-good")
        self.assertTrue(self._run())
        # Good machine landed; bad machine skipped.
        self.assertIsNotNone(self._origin_blob(
            "evidence/experiments/runner_heartbeats/"
            "ree-cloud-good.json"))
        self.assertIsNone(self._origin_blob(
            "evidence/experiments/runner_heartbeats/"
            "ree-cloud-bad.json"))


class DirtyTreeRefusal(_Fixture):
    def test_refuses_on_dirty_tree(self):
        self._seed_heartbeat("ree-cloud-1")
        (self._repo / "stray.txt").write_text("WIP\n")
        self.assertFalse(self._run())


class ForeignCommitRefusal(_Fixture):
    def test_operator_commit_blocks_push(self):
        self._seed_heartbeat("ree-cloud-1")
        _git(self._repo, "config", "user.email", "operator@example")
        _git(self._repo, "config", "user.name", "operator")
        _git(self._repo, "commit", "--allow-empty", "-m",
             "operator hand edit")
        result = self._run()
        self.assertFalse(result, "must refuse to push foreign commit")


class SiblingWriterCommitTolerated(_Fixture):
    """Regression: heartbeat writer and result writer share REE_assembly.
    A leftover commit from the OTHER phase3 writer (e.g. a `phase3: ...`
    commit retained after a transient push failure on the result writer)
    must NOT be treated as foreign by the heartbeat writer's check, or
    the two writers permanently deadlock each other."""

    def test_result_writer_commit_does_not_block_heartbeat_push(self):
        # Simulate a stuck result-writer commit ahead of origin (push
        # failed on a prior tick; commit retained in local HEAD).
        _git(self._repo, "config", "user.email", "phase3@example")
        _git(self._repo, "config", "user.name", "phase3")
        _git(self._repo, "commit", "--allow-empty", "-m",
             "phase3: 1 v3 result manifest(s) 2026-05-28")
        # Now run the heartbeat writer. It must tolerate the sibling
        # writer's leftover commit AND push its own new telemetry.
        self._seed_heartbeat("ree-cloud-1")
        result = self._run()
        self.assertTrue(
            result,
            "heartbeat writer must accept sibling result-writer's commit "
            "as writer-authored (shared REE_assembly repo)")
        # And the telemetry file landed.
        self.assertIsNotNone(self._origin_blob(
            "evidence/experiments/runner_heartbeats/ree-cloud-1.json"))


# ---------------------------------------------------------------------------
# DB helper sanity (no writer involvement)
# ---------------------------------------------------------------------------

class DBHelpers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_hb_db_")
        self._dbpath = os.path.join(self._tmp, "c.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_upsert_heartbeat_stores_payload(self):
        payload = {"x": 1}
        db.upsert_heartbeat(
            self._conn, "ree-cloud-1",
            state="running", current_exq="V3-EXQ-1",
            progress={}, gpu={},
            payload_json=json.dumps(payload))
        row = self._conn.execute(
            "SELECT heartbeat_payload_json FROM heartbeats "
            "WHERE machine=?", ("ree-cloud-1",)).fetchone()
        self.assertEqual(json.loads(row["heartbeat_payload_json"]), payload)

    def test_upsert_heartbeat_with_none_payload_preserves_existing(self):
        # First call seeds the payload.
        db.upsert_heartbeat(
            self._conn, "ree-cloud-1",
            state="running", current_exq="V3-EXQ-1",
            progress={}, gpu={},
            payload_json=json.dumps({"first": True}))
        # Second call WITHOUT payload (legacy client) should NOT
        # clobber the stored payload.
        db.upsert_heartbeat(
            self._conn, "ree-cloud-1",
            state="idle", current_exq=None,
            progress={}, gpu={},
            payload_json=None)
        row = self._conn.execute(
            "SELECT heartbeat_payload_json, state FROM heartbeats "
            "WHERE machine=?", ("ree-cloud-1",)).fetchone()
        self.assertEqual(row["state"], "idle")  # structured field updated
        self.assertEqual(
            json.loads(row["heartbeat_payload_json"]),
            {"first": True},  # payload preserved
            "legacy client calling upsert_heartbeat with payload_json=None "
            "must NOT overwrite a previously-stored rich payload")

    def test_record_status_payload_creates_row(self):
        db.record_status_payload(
            self._conn, "ree-cloud-fresh",
            json.dumps({"completed": ["V3-EXQ-X"]}))
        row = self._conn.execute(
            "SELECT status_payload_json FROM heartbeats "
            "WHERE machine=?", ("ree-cloud-fresh",)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(
            json.loads(row["status_payload_json"]),
            {"completed": ["V3-EXQ-X"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
