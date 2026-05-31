"""Contract tests for sync_daemon._bootstrap_writer_health_from_git.

Covers the four contracts spelled out in the 2026-05-31 writer-health
bootstrap chip:

  C1 -- clean state seed: bootstrap populates last_commit_sha /
        _at / _subject from each writer's most recent matching commit on
        origin/<branch>.
  C2 -- failure tolerance: a git subprocess raising does not crash; the
        affected writer's commit fields stay null and a warning is
        logged.
  C3 -- no override of fresher data: bootstrap run a second time after
        a live tick recorded a newer commit does NOT replace the
        newer-stamp data with the older bootstrap stamp.
  C4 -- last_tick_at is untouched: bootstrap MUST NOT touch the
        last_tick_at field (that signal is owned by the live tick path).

Run: /opt/local/bin/python3 test_phase3_writer_health_bootstrap.py
or:  /opt/local/bin/python3 -m unittest test_phase3_writer_health_bootstrap

All printed text is ASCII-only (Windows cp1252 safety).
"""

import io
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sync_daemon  # noqa: E402


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=check,
    )


def _init_repo_with_commit(path, branch, commit_subject):
    """Create a git repo at `path` on `branch` with a single commit whose
    subject is `commit_subject`. Wire `origin/<branch>` to point at HEAD
    via a synthetic refs/remotes/origin/<branch> file -- avoids needing a
    real remote in the test harness while preserving the
    `git log origin/<branch>` lookup shape the bootstrap uses."""
    os.makedirs(path, exist_ok=True)
    _git(path, "init", "-b", branch, ".")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    fname = os.path.join(path, "seed.txt")
    with open(fname, "w") as fh:
        fh.write("seed\n")
    _git(path, "add", "seed.txt")
    _git(path, "commit", "-m", commit_subject)
    head_sha = _git(path, "rev-parse", "HEAD").stdout.strip()
    refs_dir = os.path.join(path, ".git", "refs", "remotes", "origin")
    os.makedirs(refs_dir, exist_ok=True)
    with open(os.path.join(refs_dir, branch), "w") as fh:
        fh.write(head_sha + "\n")
    return head_sha


class BootstrapWriterHealthTests(unittest.TestCase):

    def setUp(self):
        # Build two scratch repos: REE_assembly (master) for git_writer +
        # heartbeat_writer; ree-v3 (main) for queue_writer. Each carries
        # one matching commit so the bootstrap has something to seed.
        self.tmp = tempfile.mkdtemp(prefix="bootstrap-writer-health-")
        self.asm_repo = os.path.join(self.tmp, "REE_assembly")
        self.v3_repo = os.path.join(self.tmp, "ree-v3")

        # Two distinct phase3 prefixes land in REE_assembly. The bootstrap
        # routes git_writer -> phase3:, heartbeat_writer -> phase3-heartbeats:.
        # Land the heartbeats commit first, then a results commit on top --
        # `git log -1 --grep=^phase3-heartbeats:` should still find the
        # underlying commit because it walks history, not just HEAD.
        _init_repo_with_commit(
            self.asm_repo, "master",
            "phase3-heartbeats: cloud-2 -> V3-EXQ-123 started")
        # Add a second commit so git_writer's prefix has a distinct hit.
        fname = os.path.join(self.asm_repo, "later.txt")
        with open(fname, "w") as fh:
            fh.write("later\n")
        _git(self.asm_repo, "add", "later.txt")
        _git(self.asm_repo, "commit", "-m",
             "phase3: result V3-EXQ-456 PASS")
        # Refresh origin/master to point at the new HEAD.
        head_sha = _git(self.asm_repo, "rev-parse", "HEAD").stdout.strip()
        with open(os.path.join(self.asm_repo, ".git", "refs", "remotes",
                               "origin", "master"), "w") as fh:
            fh.write(head_sha + "\n")
        self.asm_git_writer_sha = head_sha
        self.asm_heartbeat_writer_sha = _git(
            self.asm_repo, "rev-parse", "HEAD^").stdout.strip()

        _init_repo_with_commit(
            self.v3_repo, "main",
            "phase3-queue: snapshot 2026-05-31")
        self.v3_queue_writer_sha = _git(
            self.v3_repo, "rev-parse", "HEAD").stdout.strip()

        # Monkey-patch sync_daemon's hub-path constants to point at the
        # scratch repos. Restored in tearDown.
        self._orig_asm = sync_daemon.PHASE3_REE_ASSEMBLY
        self._orig_v3 = sync_daemon.PHASE3_REE_V3
        self._orig_asm_br = sync_daemon.PHASE3_ASSEMBLY_BRANCH
        self._orig_v3_br = sync_daemon.PHASE3_REE_V3_BRANCH
        sync_daemon.PHASE3_REE_ASSEMBLY = self.asm_repo
        sync_daemon.PHASE3_REE_V3 = self.v3_repo
        sync_daemon.PHASE3_ASSEMBLY_BRANCH = "master"
        sync_daemon.PHASE3_REE_V3_BRANCH = "main"

        sync_daemon._reset_writer_health_state()

    def tearDown(self):
        sync_daemon.PHASE3_REE_ASSEMBLY = self._orig_asm
        sync_daemon.PHASE3_REE_V3 = self._orig_v3
        sync_daemon.PHASE3_ASSEMBLY_BRANCH = self._orig_asm_br
        sync_daemon.PHASE3_REE_V3_BRANCH = self._orig_v3_br
        sync_daemon._reset_writer_health_state()
        try:
            import shutil
            shutil.rmtree(self.tmp, ignore_errors=True)
        except Exception:
            pass

    # -----------------------------------------------------------------
    # C1: clean-state seed
    # -----------------------------------------------------------------
    def test_c1_clean_state_seeds_all_three_writers(self):
        sync_daemon._bootstrap_writer_health_from_git()
        with sync_daemon._WRITER_HEALTH_LOCK:
            gw = dict(sync_daemon._WRITER_HEALTH["git_writer"])
            qw = dict(sync_daemon._WRITER_HEALTH["queue_writer"])
            hw = dict(sync_daemon._WRITER_HEALTH["heartbeat_writer"])

        # git_writer
        self.assertEqual(gw["last_commit_sha"], self.asm_git_writer_sha)
        self.assertIn("phase3: result", gw["last_commit_subject"])
        self.assertIsNotNone(gw["last_commit_at"])

        # queue_writer
        self.assertEqual(qw["last_commit_sha"], self.v3_queue_writer_sha)
        self.assertIn("phase3-queue:", qw["last_commit_subject"])
        self.assertIsNotNone(qw["last_commit_at"])

        # heartbeat_writer (walked back past the result commit on master)
        self.assertEqual(
            hw["last_commit_sha"], self.asm_heartbeat_writer_sha)
        self.assertIn("phase3-heartbeats:", hw["last_commit_subject"])
        self.assertIsNotNone(hw["last_commit_at"])

        # last_tick_at remains null on every writer.
        for rec in (gw, qw, hw):
            self.assertIsNone(rec["last_tick_at"])

    # -----------------------------------------------------------------
    # C2: failure tolerance
    # -----------------------------------------------------------------
    def test_c2_failure_tolerance_leaves_fields_null_and_warns(self):
        orig_run = sync_daemon.subprocess.run

        def boom(*args, **kwargs):
            raise OSError("simulated git failure")

        sync_daemon.subprocess.run = boom
        captured = io.StringIO()
        orig_stderr = sys.stderr
        sys.stderr = captured
        try:
            sync_daemon._bootstrap_writer_health_from_git()
        finally:
            sys.stderr = orig_stderr
            sync_daemon.subprocess.run = orig_run

        with sync_daemon._WRITER_HEALTH_LOCK:
            for name in ("git_writer", "queue_writer", "heartbeat_writer"):
                rec = sync_daemon._WRITER_HEALTH[name]
                self.assertIsNone(rec["last_commit_sha"])
                self.assertIsNone(rec["last_commit_at"])
                self.assertIsNone(rec["last_commit_subject"])

        msg = captured.getvalue()
        self.assertIn("[writer-health] bootstrap failed", msg)
        # All three writers attempt the call -> three warning lines.
        self.assertEqual(msg.count("bootstrap failed"), 3)

    # -----------------------------------------------------------------
    # C3: no override of fresher live-tick data
    # -----------------------------------------------------------------
    def test_c3_does_not_overwrite_fresher_commit_data(self):
        # Pre-seed git_writer with a "live tick" that is strictly newer
        # than any plausible bootstrap stamp.
        future_at = "2099-01-01T00:00:00Z"
        with sync_daemon._WRITER_HEALTH_LOCK:
            sync_daemon._WRITER_HEALTH["git_writer"].update({
                "last_commit_sha":     "deadbeef" * 5,
                "last_commit_at":      future_at,
                "last_commit_subject": "live tick stamped this",
            })

        sync_daemon._bootstrap_writer_health_from_git()

        with sync_daemon._WRITER_HEALTH_LOCK:
            gw = dict(sync_daemon._WRITER_HEALTH["git_writer"])
            qw = dict(sync_daemon._WRITER_HEALTH["queue_writer"])

        # git_writer untouched -- bootstrap stamp is older than future_at.
        self.assertEqual(gw["last_commit_at"], future_at)
        self.assertEqual(gw["last_commit_sha"], "deadbeef" * 5)
        self.assertEqual(gw["last_commit_subject"], "live tick stamped this")

        # queue_writer was null -> seeded normally.
        self.assertEqual(qw["last_commit_sha"], self.v3_queue_writer_sha)
        self.assertIsNotNone(qw["last_commit_at"])

    # -----------------------------------------------------------------
    # C4: last_tick_at unchanged
    # -----------------------------------------------------------------
    def test_c4_last_tick_at_unchanged_by_bootstrap(self):
        # Record a live tick on git_writer; capture the timestamp.
        sync_daemon._record_writer_tick("git_writer")
        with sync_daemon._WRITER_HEALTH_LOCK:
            tick_before = sync_daemon._WRITER_HEALTH["git_writer"]["last_tick_at"]

        # Sleep one whole second so any inadvertent overwrite would
        # produce a strictly different ISO stamp.
        time.sleep(1.1)
        sync_daemon._bootstrap_writer_health_from_git()

        with sync_daemon._WRITER_HEALTH_LOCK:
            tick_after = sync_daemon._WRITER_HEALTH["git_writer"]["last_tick_at"]
            qw_tick = sync_daemon._WRITER_HEALTH["queue_writer"]["last_tick_at"]
            hw_tick = sync_daemon._WRITER_HEALTH["heartbeat_writer"]["last_tick_at"]

        self.assertEqual(tick_after, tick_before)
        # Other writers, which never ticked, stay null.
        self.assertIsNone(qw_tick)
        self.assertIsNone(hw_tick)


if __name__ == "__main__":
    unittest.main(verbosity=2)
