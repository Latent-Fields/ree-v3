"""Tests for sync_daemon telemetry-only dirty-tree auto-recovery.

When the hub checkout has dirt only under runner_heartbeats/ and
runner_status/, phase3_git_writer and phase3_heartbeat_writer may
revert those paths so result spool commits are not stalled.
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

import sync_daemon  # noqa: E402


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=check,
    )


class TelemetryDirtRecoveryTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_telemetry_dirt_")
        self._repo = pathlib.Path(self._tmp) / "asm"
        subprocess.run(
            ["git", "init", "-q", "-b", "master", str(self._repo)],
            check=True)
        _git(self._repo, "config", "user.email", "test@example")
        _git(self._repo, "config", "user.name", "test")
        hb_dir = (
            self._repo / "evidence" / "experiments" / "runner_heartbeats")
        hb_dir.mkdir(parents=True)
        (hb_dir / "ree-cloud-1.json").write_text("{}\n")
        _git(self._repo, "add", ".")
        _git(self._repo, "commit", "-q", "-m", "seed")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_reverts_modified_heartbeat_only(self):
        hb = (self._repo / "evidence" / "experiments"
              / "runner_heartbeats" / "ree-cloud-1.json")
        hb.write_text(json.dumps({"stale": True}) + "\n")
        self.assertTrue(
            sync_daemon._maybe_revert_exclusive_telemetry_dirt(
                str(self._repo), "test"))
        clean, _ = sync_daemon._hub_working_tree_clean(str(self._repo))
        self.assertTrue(clean)
        self.assertNotIn("stale", hb.read_text())

    def test_refuses_when_manifest_also_dirty(self):
        hb = (self._repo / "evidence" / "experiments"
              / "runner_heartbeats" / "ree-cloud-1.json")
        hb.write_text("{}\n")
        manifest = (self._repo / "evidence" / "experiments"
                      / "v3_exq_001_foo_20260101T000000Z_v3.json")
        manifest.write_text("{}\n")
        self.assertFalse(
            sync_daemon._maybe_revert_exclusive_telemetry_dirt(
                str(self._repo), "test"))
        clean, _ = sync_daemon._hub_working_tree_clean(str(self._repo))
        self.assertFalse(clean)

    def test_clean_tree_is_noop(self):
        self.assertTrue(
            sync_daemon._maybe_revert_exclusive_telemetry_dirt(
                str(self._repo), "test"))


if __name__ == "__main__":
    unittest.main()
