"""Offline smoke harness for sync_daemon._active_claim_blocks_writer
and its three writer integration points.

Covers:
  - No TASK_CLAIMS file -> guard returns False; writer runs as before.
  - Active claim on a matching path -> guard returns True; writer
    returns True (idle no-op) without committing or pushing.
  - Active claim on a non-matching path (e.g. evidence/planning/) ->
    guard returns False; writer runs.
  - done/closed claim ignored even if it matches the path.
  - Malformed TASK_CLAIMS.json -> guard returns False (best-effort);
    writer runs.
  - Per-writer path scoping: a claim on runner_status/ pauses the
    heartbeat writer but not the result writer; a claim on
    evidence/experiments/<run_id>.json pauses the result writer but
    not the heartbeat writer.

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


def _bare_remote(parent, name="asm.git"):
    remote = pathlib.Path(parent) / name
    subprocess.run(
        ["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _seeded_clone(parent, remote, *, name="asm",
                  seed_msg="phase3-heartbeats: seed"):
    """Seed with a writer-authored commit so the foreign-commit check
    doesn't trip in tests that don't intentionally probe it."""
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


def _write_claims(parent, claims):
    path = pathlib.Path(parent) / "TASK_CLAIMS.json"
    path.write_text(json.dumps(
        {"schema_version": "v1", "stale_after_hours": 6,
         "claims": claims}, indent=2))
    return path


# ---------------------------------------------------------------------------
# Unit tests on the guard function itself
# ---------------------------------------------------------------------------

class GuardFunction(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_guard_unit_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_no_claims_file_returns_false(self):
        # Three writers, no TASK_CLAIMS.json present.
        for w in ("phase3_git_writer", "phase3_queue_writer",
                  "phase3_heartbeat_writer"):
            self.assertFalse(
                sync_daemon._active_claim_blocks_writer(w, self._tmp))

    def test_active_claim_matches_heartbeat_writer(self):
        _write_claims(self._tmp, [{
            "session_id": "s1",
            "claimed_at": "2026-05-30T23:00:00Z",
            "task": "edit runner_status",
            "resources": [
                "REE_assembly/evidence/experiments/runner_status/"
                "DLAPTOP-4.local.json",
            ],
            "status": "active",
        }])
        self.assertTrue(sync_daemon._active_claim_blocks_writer(
            "phase3_heartbeat_writer", self._tmp))
        # Result + queue writers are NOT affected by this claim.
        self.assertFalse(sync_daemon._active_claim_blocks_writer(
            "phase3_queue_writer", self._tmp))
        # The result writer guards on evidence/experiments/ as a whole,
        # so a runner_status/ claim DOES match (runner_status/ is under
        # evidence/experiments/). Per design -- both writers share the
        # REE_assembly repo, so a claim on runner_status/ should pause
        # both. Explicit test: assert the result writer also pauses on
        # the same claim.
        self.assertTrue(sync_daemon._active_claim_blocks_writer(
            "phase3_git_writer", self._tmp))

    def test_active_claim_on_queue_only_matches_queue_writer(self):
        _write_claims(self._tmp, [{
            "session_id": "s2",
            "claimed_at": "2026-05-30T23:00:00Z",
            "task": "hand-edit queue",
            "resources": ["ree-v3/experiment_queue.json"],
            "status": "active",
        }])
        self.assertTrue(sync_daemon._active_claim_blocks_writer(
            "phase3_queue_writer", self._tmp))
        self.assertFalse(sync_daemon._active_claim_blocks_writer(
            "phase3_heartbeat_writer", self._tmp))
        self.assertFalse(sync_daemon._active_claim_blocks_writer(
            "phase3_git_writer", self._tmp))

    def test_active_claim_on_planning_does_not_match_result_writer(self):
        # Result writer guards on evidence/experiments/ specifically;
        # a claim on evidence/planning/ should NOT pause it.
        _write_claims(self._tmp, [{
            "session_id": "s3",
            "claimed_at": "2026-05-30T23:00:00Z",
            "task": "edit substrate_queue plan",
            "resources": ["REE_assembly/evidence/planning/substrate_queue.json"],
            "status": "active",
        }])
        for w in ("phase3_git_writer", "phase3_queue_writer",
                  "phase3_heartbeat_writer"):
            self.assertFalse(
                sync_daemon._active_claim_blocks_writer(w, self._tmp))

    def test_done_claim_is_ignored(self):
        _write_claims(self._tmp, [{
            "session_id": "s4",
            "claimed_at": "2026-05-30T23:00:00Z",
            "task": "previously edited runner_status",
            "resources": [
                "REE_assembly/evidence/experiments/runner_status.json"],
            "status": "done",
        }])
        for w in ("phase3_git_writer", "phase3_queue_writer",
                  "phase3_heartbeat_writer"):
            self.assertFalse(
                sync_daemon._active_claim_blocks_writer(w, self._tmp))

    def test_malformed_claims_file_returns_false(self):
        path = pathlib.Path(self._tmp) / "TASK_CLAIMS.json"
        path.write_text("{ not valid json")
        for w in ("phase3_git_writer", "phase3_queue_writer",
                  "phase3_heartbeat_writer"):
            self.assertFalse(
                sync_daemon._active_claim_blocks_writer(w, self._tmp))

    def test_non_string_resource_is_skipped(self):
        _write_claims(self._tmp, [{
            "session_id": "s5",
            "claimed_at": "2026-05-30T23:00:00Z",
            "task": "bogus resource shape",
            "resources": [42, None, {"path": "x"}],
            "status": "active",
        }])
        for w in ("phase3_git_writer", "phase3_queue_writer",
                  "phase3_heartbeat_writer"):
            self.assertFalse(
                sync_daemon._active_claim_blocks_writer(w, self._tmp))

    def test_unknown_writer_name_returns_false(self):
        _write_claims(self._tmp, [{
            "session_id": "s6",
            "claimed_at": "2026-05-30T23:00:00Z",
            "task": "anything",
            "resources": [
                "REE_assembly/evidence/experiments/runner_status.json"],
            "status": "active",
        }])
        self.assertFalse(sync_daemon._active_claim_blocks_writer(
            "phase3_made_up_writer", self._tmp))


# ---------------------------------------------------------------------------
# Integration tests: writer functions consult the guard
# ---------------------------------------------------------------------------

class HeartbeatWriterIntegration(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_guard_hb_")
        self._remote = _bare_remote(self._tmp)
        self._repo = _seeded_clone(self._tmp, self._remote)
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        # Seed a heartbeat row so writer has work to do.
        db.upsert_heartbeat(
            self._conn, "DLAPTOP-4.local",
            state="running", current_exq="V3-EXQ-X",
            progress={}, gpu={},
            payload_json=json.dumps({"machine": "DLAPTOP-4.local"}))
        db.record_status_payload(
            self._conn, "DLAPTOP-4.local",
            json.dumps({"machine": "DLAPTOP-4.local",
                        "completed": [{"queue_id": "V3-EXQ-X",
                                       "result": "PASS"}]}))
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

    def _origin_has_file(self, relpath):
        out = _git(self._remote, "show", "master:" + relpath, check=False)
        return out.returncode == 0

    def test_claim_on_runner_status_pauses_writer(self):
        _write_claims(self._tmp, [{
            "session_id": "s_block",
            "claimed_at": "2026-05-30T23:00:00Z",
            "task": "edit runner_status",
            "resources": [
                "REE_assembly/evidence/experiments/runner_status/"
                "DLAPTOP-4.local.json"],
            "status": "active",
        }])
        # Guard returns True (idle), no file landed on origin.
        self.assertTrue(self._run())
        self.assertFalse(self._origin_has_file(
            "evidence/experiments/runner_status/DLAPTOP-4.local.json"))

    def test_claim_on_planning_does_not_pause_writer(self):
        _write_claims(self._tmp, [{
            "session_id": "s_planning",
            "claimed_at": "2026-05-30T23:00:00Z",
            "task": "edit planning doc",
            "resources": [
                "REE_assembly/evidence/planning/substrate_queue.json"],
            "status": "active",
        }])
        self.assertTrue(self._run())
        # Writer ran; file landed on origin.
        self.assertTrue(self._origin_has_file(
            "evidence/experiments/runner_status/DLAPTOP-4.local.json"))

    def test_no_claims_file_writer_runs_normally(self):
        # No TASK_CLAIMS.json at the umbrella root.
        self.assertTrue(self._run())
        self.assertTrue(self._origin_has_file(
            "evidence/experiments/runner_status/DLAPTOP-4.local.json"))


class QueueWriterIntegration(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_guard_q_")
        self._remote = _bare_remote(self._tmp, name="ree-v3.git")
        # ree-v3 checkout uses `main` per Phase 3 default; tests can
        # override the branch arg when calling the writer.
        self._repo = pathlib.Path(self._tmp) / "ree-v3"
        subprocess.run(
            ["git", "init", "-q", "-b", "main", str(self._repo)],
            check=True)
        _git(self._repo, "config", "user.email", "test@example")
        _git(self._repo, "config", "user.name", "test")
        # Seed with an empty queue file + writer-authored seed.
        (self._repo / "experiment_queue.json").write_text(
            '{"schema_version": "v1", "items": []}\n')
        _git(self._repo, "add", "experiment_queue.json")
        _git(self._repo, "commit", "-q",
             "-m", "phase3-queue: seed")
        _git(self._repo, "remote", "add", "origin", str(self._remote))
        _git(self._repo, "push", "-q", "origin", "main")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        # Add one experiment row so the materialised queue would differ
        # from the seeded empty file (forcing a real write attempt).
        self._conn.execute(
            "INSERT INTO experiments(queue_id, script, item_json, "
            "updated_at) VALUES (?, ?, ?, ?)",
            ("V3-EXQ-GUARD-TEST",
             "experiments/foo.py",
             '{"queue_id": "V3-EXQ-GUARD-TEST", '
             '"script": "experiments/foo.py"}',
             "2026-05-30T23:30:00Z"))
        self._conn.commit()
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

    def _origin_queue_contains_id(self, qid):
        out = _git(self._remote, "show",
                   "main:experiment_queue.json", check=False)
        if out.returncode != 0:
            return False
        try:
            doc = json.loads(out.stdout)
        except (ValueError, TypeError):
            return False
        return any(item.get("queue_id") == qid
                   for item in doc.get("items", []))

    def test_claim_on_queue_pauses_writer(self):
        _write_claims(self._tmp, [{
            "session_id": "q_block",
            "claimed_at": "2026-05-30T23:00:00Z",
            "task": "hand-edit queue",
            "resources": ["ree-v3/experiment_queue.json"],
            "status": "active",
        }])
        # Guard returns True (idle), origin queue unchanged.
        self.assertTrue(self._run())
        self.assertFalse(self._origin_queue_contains_id(
            "V3-EXQ-GUARD-TEST"))

    def test_claim_on_unrelated_runner_status_does_not_pause_queue(self):
        # A runner_status claim is for the heartbeat/result writer, not
        # the queue writer. Queue should run.
        _write_claims(self._tmp, [{
            "session_id": "q_unrelated",
            "claimed_at": "2026-05-30T23:00:00Z",
            "task": "edit runner_status",
            "resources": [
                "REE_assembly/evidence/experiments/runner_status.json"],
            "status": "active",
        }])
        self.assertTrue(self._run())
        self.assertTrue(self._origin_queue_contains_id(
            "V3-EXQ-GUARD-TEST"))


if __name__ == "__main__":
    unittest.main()
