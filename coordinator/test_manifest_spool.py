"""Tests for the manifest spool + Phase 3 writer guards.

Covers:
  - Spool round-trip (write -> list -> read -> delete).
  - No-op when COORDINATOR_SPOOL_DIR is unset (the Phase 2 default).
  - Atomic write semantics (no partial files visible to list_pending).
  - Unsafe run_id rejection (filesystem traversal defence).
  - Evidence-path derivation including manifest_relpath hint sanitisation.
  - sync_daemon.phase3_git_writer guards:
      * Returns False when PHASE3_GIT_WRITER_READY is False (the default).
      * Returns False when the spool root is unset.
      * Returns True on an empty spool (idle-tick no-op).
      * Refuses to operate on a dirty REE_assembly working tree.
      * Dry-run does not touch the filesystem outside the spool/test repo.

All printed text is ASCII-only.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manifest_spool  # noqa: E402
import sync_daemon  # noqa: E402
import db  # noqa: E402


def _scratch_repo(tmpdir):
    repo = pathlib.Path(tmpdir) / "asm"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)],
                   check=True)
    subprocess.run(["git", "-C", str(repo), "config",
                    "user.email", "test@example"], check=True)
    subprocess.run(["git", "-C", str(repo), "config",
                    "user.name", "test"], check=True)
    (repo / "README.md").write_text("seed\n")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "seed"],
                   check=True)
    return repo


class SpoolUnsetTest(unittest.TestCase):
    """Phase 2 default: COORDINATOR_SPOOL_DIR unset -> every operation
    is a no-op. This is the bit-identical-to-today guarantee."""

    def setUp(self):
        self._saved = os.environ.pop("COORDINATOR_SPOOL_DIR", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["COORDINATOR_SPOOL_DIR"] = self._saved

    def test_root_none(self):
        self.assertIsNone(manifest_spool.spool_root())

    def test_write_returns_none(self):
        self.assertIsNone(
            manifest_spool.write_manifest("run_abc", b'{"ok":true}'))

    def test_list_empty(self):
        self.assertEqual(list(manifest_spool.list_pending_run_ids()), [])

    def test_count_pending_disabled(self):
        self.assertIsNone(manifest_spool.count_pending())

    def test_read_none(self):
        self.assertIsNone(manifest_spool.read_manifest("run_abc"))
        self.assertIsNone(manifest_spool.read_meta("run_abc"))


class SpoolRoundtripTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="spool_rt_")
        os.environ["COORDINATOR_SPOOL_DIR"] = self._tmp

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("COORDINATOR_SPOOL_DIR", None)

    def test_write_then_read_then_delete(self):
        raw = b'{"run_id":"v3_test_001","outcome":"PASS"}'
        path = manifest_spool.write_manifest(
            "v3_test_001", raw,
            manifest_relpath="evidence/experiments/v3_test_001.json",
            received_at="2026-05-27T06:30:00Z",
            sha256_hex="deadbeef")
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())
        ids = list(manifest_spool.list_pending_run_ids())
        self.assertEqual(ids, ["v3_test_001"])
        self.assertEqual(manifest_spool.count_pending(), 1)
        self.assertEqual(manifest_spool.read_manifest("v3_test_001"), raw)
        meta = manifest_spool.read_meta("v3_test_001")
        self.assertEqual(meta["sha256"], "deadbeef")
        self.assertEqual(
            meta["manifest_relpath"],
            "evidence/experiments/v3_test_001.json")
        self.assertTrue(manifest_spool.delete_manifest("v3_test_001"))
        self.assertEqual(list(manifest_spool.list_pending_run_ids()), [])
        self.assertEqual(manifest_spool.count_pending(), 0)

    def test_partial_meta_only_not_yielded(self):
        # Drop a stray .meta.json without a .json sibling -- list must skip it
        # (the atomicity-preserving filter).
        pending = pathlib.Path(self._tmp) / "pending"
        pending.mkdir(parents=True, exist_ok=True)
        (pending / "orphan.meta.json").write_text("{}")
        self.assertEqual(list(manifest_spool.list_pending_run_ids()), [])

    def test_unsafe_run_id_refused(self):
        for bad in ("..", "../escape", "a/b", "x" * 1000):
            self.assertIsNone(
                manifest_spool.write_manifest(bad, b"{}"),
                "should refuse %r" % bad)


class DeriveEvidencePathTest(unittest.TestCase):
    def test_default_path_when_no_hint(self):
        rel = manifest_spool.derive_evidence_relpath(
            "v3_exq_001_v3", {"outcome": "PASS"})
        self.assertEqual(rel, "evidence/experiments/v3_exq_001_v3.json")

    def test_accepts_in_tree_hint(self):
        rel = manifest_spool.derive_evidence_relpath(
            "v3_exq_001_v3",
            {"manifest_relpath": "evidence/experiments/subdir/x.json"})
        self.assertEqual(rel, "evidence/experiments/subdir/x.json")

    def test_rejects_traversal_hint(self):
        rel = manifest_spool.derive_evidence_relpath(
            "v3_exq_001_v3",
            {"manifest_relpath": "../../etc/passwd"})
        self.assertEqual(rel, "evidence/experiments/v3_exq_001_v3.json")

    def test_rejects_outside_evidence_prefix(self):
        rel = manifest_spool.derive_evidence_relpath(
            "v3_exq_001_v3",
            {"manifest_relpath": "scripts/governance.sh"})
        self.assertEqual(rel, "evidence/experiments/v3_exq_001_v3.json")


class WriterGuardsTest(unittest.TestCase):
    """phase3_git_writer must refuse to run unless every safety knob is
    deliberately enabled. Defaults are the safe state."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="writer_guard_")
        self._saved_spool = os.environ.pop("COORDINATOR_SPOOL_DIR", None)
        # Fresh in-memory-ish DB
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._queue = os.path.join(self._tmp, "queue.json")
        with open(self._queue, "w", encoding="utf-8") as fh:
            json.dump({"items": []}, fh)

    def tearDown(self):
        self._conn.close()
        if self._saved_spool is not None:
            os.environ["COORDINATOR_SPOOL_DIR"] = self._saved_spool
        else:
            os.environ.pop("COORDINATOR_SPOOL_DIR", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_refuses_when_writer_ready_flag_false(self):
        self.assertFalse(sync_daemon.PHASE3_GIT_WRITER_READY)
        self.assertFalse(sync_daemon.phase3_git_writer(
            self._conn, self._queue, ree_assembly_path=self._tmp))

    def test_refuses_when_spool_unset(self):
        sync_daemon.PHASE3_GIT_WRITER_READY = True
        try:
            self.assertFalse(sync_daemon.phase3_git_writer(
                self._conn, self._queue, ree_assembly_path=self._tmp))
        finally:
            sync_daemon.PHASE3_GIT_WRITER_READY = False

    def test_idle_tick_returns_true(self):
        os.environ["COORDINATOR_SPOOL_DIR"] = self._tmp
        sync_daemon.PHASE3_GIT_WRITER_READY = True
        try:
            self.assertTrue(sync_daemon.phase3_git_writer(
                self._conn, self._queue, ree_assembly_path=self._tmp))
        finally:
            sync_daemon.PHASE3_GIT_WRITER_READY = False

    def test_dry_run_with_pending_does_not_touch_repo(self):
        os.environ["COORDINATOR_SPOOL_DIR"] = self._tmp
        repo = _scratch_repo(self._tmp)
        manifest_spool.write_manifest(
            "v3_test_002", b'{"run_id":"v3_test_002"}',
            received_at="2026-05-27T06:30:00Z", sha256_hex="abc")
        sync_daemon.PHASE3_GIT_WRITER_READY = True
        try:
            self.assertTrue(sync_daemon.phase3_git_writer(
                self._conn, self._queue,
                ree_assembly_path=str(repo), dry_run=True))
        finally:
            sync_daemon.PHASE3_GIT_WRITER_READY = False
        # Spool intact + repo untouched
        self.assertEqual(
            list(manifest_spool.list_pending_run_ids()), ["v3_test_002"])
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout
        self.assertEqual(status.strip(), "")

    def test_refuses_dirty_tree(self):
        os.environ["COORDINATOR_SPOOL_DIR"] = self._tmp
        repo = _scratch_repo(self._tmp)
        (repo / "uncommitted.txt").write_text("WIP\n")
        manifest_spool.write_manifest(
            "v3_test_003", b'{"run_id":"v3_test_003"}',
            received_at="2026-05-27T06:30:00Z", sha256_hex="abc")
        sync_daemon.PHASE3_GIT_WRITER_READY = True
        try:
            self.assertFalse(sync_daemon.phase3_git_writer(
                self._conn, self._queue, ree_assembly_path=str(repo)))
        finally:
            sync_daemon.PHASE3_GIT_WRITER_READY = False
        # Spool still has the manifest; nothing was committed.
        self.assertIn("v3_test_003",
                      list(manifest_spool.list_pending_run_ids()))


class WriterE2ELocalTest(unittest.TestCase):
    """End-to-end against a local bare-remote pair. Verifies the happy
    path: spool -> commit -> push -> committed_at set -> spool drained."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="writer_e2e_")
        self._saved_spool = os.environ.get("COORDINATOR_SPOOL_DIR")
        os.environ["COORDINATOR_SPOOL_DIR"] = self._tmp
        # Bare remote so `git push origin HEAD:master` succeeds.
        self._remote = pathlib.Path(self._tmp) / "remote.git"
        subprocess.run(["git", "init", "-q", "--bare", str(self._remote)],
                       check=True)
        self._repo = _scratch_repo(self._tmp)
        subprocess.run(
            ["git", "-C", str(self._repo), "remote", "add", "origin",
             str(self._remote)], check=True)
        subprocess.run(
            ["git", "-C", str(self._repo), "push", "-q", "origin", "master"],
            check=True)
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._queue = os.path.join(self._tmp, "queue.json")
        with open(self._queue, "w", encoding="utf-8") as fh:
            json.dump({"items": []}, fh)

    def tearDown(self):
        self._conn.close()
        if self._saved_spool is not None:
            os.environ["COORDINATOR_SPOOL_DIR"] = self._saved_spool
        else:
            os.environ.pop("COORDINATOR_SPOOL_DIR", None)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_happy_path(self):
        # Pre-load a results row + matching spool entry.
        run_id = "v3_e2e_001"
        raw = json.dumps({
            "run_id": run_id, "queue_id": "V3-EXQ-001",
            "outcome": "PASS", "machine": "test"}).encode("utf-8")
        db.record_result(self._conn, run_id, "V3-EXQ-001", "test",
                         "PASS", "sha", len(raw))
        manifest_spool.write_manifest(
            run_id, raw,
            received_at="2026-05-27T06:30:00Z", sha256_hex="sha")

        sync_daemon.PHASE3_GIT_WRITER_READY = True
        try:
            ok = sync_daemon.phase3_git_writer(
                self._conn, self._queue, ree_assembly_path=str(self._repo))
        finally:
            sync_daemon.PHASE3_GIT_WRITER_READY = False
        self.assertTrue(ok)

        # The commit landed on origin/master with the manifest in the
        # expected default path.
        log = subprocess.run(
            ["git", "-C", str(self._remote), "log", "--name-only",
             "--pretty=format:"],
            capture_output=True, text=True, check=True).stdout
        self.assertIn("evidence/experiments/%s.json" % run_id, log)

        # results.committed_at set; spool empty.
        row = self._conn.execute(
            "SELECT committed_at FROM results WHERE run_id=?",
            (run_id,)).fetchone()
        self.assertIsNotNone(row["committed_at"])
        self.assertEqual(list(manifest_spool.list_pending_run_ids()), [])


if __name__ == "__main__":
    unittest.main()
