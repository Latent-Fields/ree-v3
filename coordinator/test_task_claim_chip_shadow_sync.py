"""Contracts for task_claim_chip_shadow_sync.py -- the PHASE-1 read-only
reconciliation tick.

Uses a real throwaway git repo (bare remote + working clone with `origin`
configured), the same fixture shape test_phase3_queue_writer.py already
uses for sync_daemon, so `git fetch origin` / `git show <ref>:<path>`
exercise the real code path rather than a mock.

Pins:
  1. reconcile_once reads TASK_CLAIMS.json/TASK_CHIPS.json from the git ref
     (never the working tree) and reconciles both tables in one tick.
  2. A no-op second tick is all-unchanged, zero diverged.
  3. A source-side edit is picked up on the next tick.
  4. Structural incapability: no git-mutating command is ever invoked by
     this module -- the tick must not touch the working tree, stage, or
     commit anything in the source repo, regardless of outcome.
  5. When the ref cannot be fetched and resolved, the tick is SKIPPED -- it
     never falls back to repo_path's working-tree copy, and never raises
     (statusregress 70f6849fab: that fallback ingested a three-week-old
     working tree into the authoritative tables).
  6. Neither source file present anywhere returns None (nothing to log),
     not a misleading all-zero drift row.

ASCII-only.
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

import db  # noqa: E402
import task_claim_chip_shadow_sync as sync  # noqa: E402


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=check,
    )


def _bare_remote(parent, name="REE_Working.git"):
    remote = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


CLAIMS_DOC_V1 = {
    "schema_version": "v1",
    "stale_after_hours": 6,
    "claims": [
        {"session_id": "s1", "claimed_at": "2026-08-26T10:00:00Z",
         "session_label": "l", "task": "t", "resources": ["a.json"],
         "status": "active"},
    ],
}
CHIPS_DOC_V1 = {
    "schema_version": "task_chips/v1",
    "chips": [
        {"chip_ref": "chip-1", "task_id": None, "origin": "headless",
         "kind": "work", "urgency": False, "session_id": "unknown",
         "session_label": "", "title": "t", "tldr": "td",
         "prompt": "[chip_ref: chip-1]\n\nbody", "cwd": "/x",
         "spawned_at": "2026-08-26T09:00:00Z", "origin_host": "DLAPTOP",
         "status": "open", "claimed_by": None, "claimed_at": None,
         "claim_note": None, "claimed_host": None, "resolved_at": None,
         "resolved_by_session_id": None, "resolution_note": None,
         "resolution_note_auto": False, "attached_by_session_id": None,
         "attached_at": None},
    ],
}


def _seeded_umbrella_clone(parent, remote, name="REE_Working",
                           claims=None, chips=None):
    repo = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)],
                    check=True)
    _git(repo, "config", "user.email", "test@example")
    _git(repo, "config", "user.name", "test")
    (repo / "TASK_CLAIMS.json").write_text(
        json.dumps(claims if claims is not None else CLAIMS_DOC_V1,
                   indent=2) + "\n")
    (repo / "TASK_CHIPS.json").write_text(
        json.dumps(chips if chips is not None else CHIPS_DOC_V1,
                   indent=2) + "\n")
    _git(repo, "add", "TASK_CLAIMS.json", "TASK_CHIPS.json")
    _git(repo, "commit", "-q", "-m", "seed")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "master")
    return repo


class _Fixture(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="task_claim_chip_sync_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._remote = _bare_remote(self._tmp)
        self._repo = _seeded_umbrella_clone(self._tmp, self._remote)

    def tearDown(self):
        self._conn.close()


class TestReconcileOnce(_Fixture):

    def test_first_tick_upserts_both_tables(self):
        result = sync.reconcile_once(self._conn, str(self._repo),
                                     ref="origin/master")
        self.assertIsNotNone(result)
        claim_stats, chip_stats, source_ref, diverged = result
        self.assertEqual(claim_stats["n_new"], 1)
        self.assertEqual(chip_stats["n_new"], 1)
        self.assertEqual(diverged, 0)
        self.assertTrue(source_ref)
        self.assertNotIn("stale-local", source_ref)
        row = self._conn.execute(
            "SELECT task FROM task_claims WHERE session_id='s1'").fetchone()
        self.assertEqual(row["task"], "t")

    def test_second_tick_with_no_source_change_is_unchanged(self):
        sync.reconcile_once(self._conn, str(self._repo), ref="origin/master")
        result = sync.reconcile_once(self._conn, str(self._repo),
                                     ref="origin/master")
        claim_stats, chip_stats, _, diverged = result
        self.assertEqual(claim_stats["n_new"], 0)
        self.assertEqual(claim_stats["n_updated"], 0)
        self.assertEqual(claim_stats["n_unchanged"], 1)
        self.assertEqual(diverged, 0)

    def test_source_side_status_change_is_picked_up(self):
        sync.reconcile_once(self._conn, str(self._repo), ref="origin/master")
        updated = json.loads(json.dumps(CLAIMS_DOC_V1))
        updated["claims"][0]["status"] = "done"
        updated["claims"][0]["closed_at"] = "2026-08-26T12:00:00Z"
        (self._repo / "TASK_CLAIMS.json").write_text(
            json.dumps(updated, indent=2) + "\n")
        _git(self._repo, "add", "TASK_CLAIMS.json")
        _git(self._repo, "commit", "-q", "-m", "close s1")
        _git(self._repo, "push", "-q", "origin", "master")

        result = sync.reconcile_once(self._conn, str(self._repo),
                                     ref="origin/master")
        claim_stats = result[0]
        self.assertEqual(claim_stats["n_updated"], 1)
        row = self._conn.execute(
            "SELECT status FROM task_claims WHERE session_id='s1'"
        ).fetchone()
        self.assertEqual(row["status"], "done")

    def test_source_ref_is_the_resolved_commit_sha(self):
        expected = _git(self._repo, "rev-parse",
                        "origin/master").stdout.strip()
        result = sync.reconcile_once(self._conn, str(self._repo),
                                     ref="origin/master")
        self.assertEqual(result[2], expected)


class TestStructuralIncapability(_Fixture):

    def test_reconcile_never_dirties_the_source_working_tree(self):
        before = _git(self._repo, "status", "--porcelain").stdout
        sync.reconcile_once(self._conn, str(self._repo), ref="origin/master")
        after = _git(self._repo, "status", "--porcelain").stdout
        self.assertEqual(before, "")
        self.assertEqual(after, "",
                         "shadow sync must never touch the source working "
                         "tree -- read-only by design")

    def test_reconcile_never_advances_local_master(self):
        before = _git(self._repo, "rev-parse", "master").stdout.strip()
        # Move origin ahead without touching the local branch, mirroring
        # the real deploy shape (a fetch-only mirror clone).
        other_clone = pathlib.Path(self._tmp) / "other_writer"
        _git(pathlib.Path(self._tmp), "clone", "-q", str(self._remote),
             str(other_clone), check=True)
        _git(other_clone, "config", "user.email", "test@example")
        _git(other_clone, "config", "user.name", "test")
        (other_clone / "TASK_CLAIMS.json").write_text(
            json.dumps(CLAIMS_DOC_V1, indent=2) + "\n")
        _git(other_clone, "commit", "-q", "--allow-empty", "-m", "noop")
        _git(other_clone, "push", "-q", "origin", "master")

        sync.reconcile_once(self._conn, str(self._repo), ref="origin/master")
        after = _git(self._repo, "rev-parse", "master").stdout.strip()
        self.assertEqual(
            before, after,
            "reconcile_once must never move the local branch ref -- only "
            "fetch/show against origin")


class TestDegradePath(_Fixture):

    def test_no_git_repo_with_local_files_present_skips_the_tick(self):
        """Working-tree copies exist but there is no resolvable git ref: the
        tick is skipped, nothing is ingested (was: fell back to the files
        and labelled the source '<ref>-stale-local')."""
        plain_dir = pathlib.Path(self._tmp) / "plain"
        plain_dir.mkdir()
        (plain_dir / "TASK_CLAIMS.json").write_text(
            json.dumps(CLAIMS_DOC_V1) + "\n")
        (plain_dir / "TASK_CHIPS.json").write_text(
            json.dumps(CHIPS_DOC_V1) + "\n")
        result = sync.reconcile_once(self._conn, str(plain_dir),
                                     ref="origin/master")
        self.assertIsNone(result)
        self.assertEqual(
            self._conn.execute("SELECT COUNT(*) FROM chip_ledger").fetchone()[0],
            0, "a working-tree file must never be ingested")
        self.assertEqual(
            db.task_claim_chip_drift_summary(self._conn)["total_ticks"], 0)

    def test_fetch_failure_never_ingests_the_frozen_working_tree(self):
        """The 70f6849fab shape end to end. This module never checks
        anything out, so the mirror clone's working tree stays frozen at
        its provisioning commit while origin and the DB move on. A fetch
        failure must skip the tick, NOT read that frozen file: the old
        fallback reverted a since-changed OPEN chip's claim fields -- damage
        the monotone terminal/archived guards in upsert_chip cannot see."""
        # Tick 1 (healthy): DB mirrors origin; record the render base the
        # way the registry writer does, so the row sits in the steady state
        # (entry_json == last_rendered_json) where "adopt git" applies.
        self.assertIsNotNone(sync.reconcile_once(self._conn, str(self._repo)))
        # Origin and the DB move on: the chip gets claimed. The mirror's
        # WORKING TREE is left at the seed content, as on the hub.
        newer = json.loads(json.dumps(CHIPS_DOC_V1))
        newer["chips"][0].update(
            claimed_by="worker-9", claimed_at="2026-09-18T12:00:00Z",
            claim_note="started work", claimed_host="ree-cloud-5")
        other = pathlib.Path(self._tmp) / "other_clone"
        subprocess.run(["git", "clone", "-q", str(self._remote), str(other)],
                        check=True)
        _git(other, "config", "user.email", "test@example")
        _git(other, "config", "user.name", "test")
        (other / "TASK_CHIPS.json").write_text(
            json.dumps(newer, indent=2) + "\n")
        _git(other, "commit", "-q", "-am", "claim")
        _git(other, "push", "-q", "origin", "master")
        self.assertIsNotNone(sync.reconcile_once(self._conn, str(self._repo)))
        self._conn.execute(
            "UPDATE chip_ledger SET last_rendered_json = entry_json")
        self._conn.commit()
        before = self._conn.execute(
            "SELECT entry_json FROM chip_ledger WHERE chip_ref='chip-1'"
        ).fetchone()[0]
        self.assertEqual(json.loads(before)["claimed_by"], "worker-9")
        on_disk = json.loads((self._repo / "TASK_CHIPS.json").read_text())
        self.assertIsNone(on_disk["chips"][0]["claimed_by"],
                          "fixture: the working tree must still be the "
                          "frozen seed for this test to mean anything")
        # The fetch now fails (origin unreachable).
        _git(self._repo, "remote", "set-url", "origin",
             str(pathlib.Path(self._tmp) / "no_such_remote.git"))
        result = sync.reconcile_once(self._conn, str(self._repo))
        self.assertIsNone(result, "a failed fetch must skip the tick")
        after = self._conn.execute(
            "SELECT entry_json FROM chip_ledger WHERE chip_ref='chip-1'"
        ).fetchone()[0]
        self.assertEqual(after, before,
                         "the frozen working tree was ingested")

    def test_neither_source_readable_returns_none(self):
        empty_dir = pathlib.Path(self._tmp) / "empty"
        empty_dir.mkdir()
        result = sync.reconcile_once(self._conn, str(empty_dir),
                                     ref="origin/master")
        self.assertIsNone(result)
        summary = db.task_claim_chip_drift_summary(self._conn)
        self.assertEqual(summary["total_ticks"], 0,
                         "an unreadable source must not log a misleading "
                         "all-zero drift row")


class TestDriftLogIntegration(_Fixture):

    def test_healthy_tick_writes_one_drift_row(self):
        sync.reconcile_once(self._conn, str(self._repo), ref="origin/master")
        summary = db.task_claim_chip_drift_summary(self._conn)
        self.assertEqual(summary["total_ticks"], 1)
        self.assertEqual(summary["diverged_ticks"], 0)


if __name__ == "__main__":
    unittest.main()
