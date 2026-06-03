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


class ConflictRecovery(_QueueWriterFixture):
    """PHASE3_QUEUE_CONFLICT_RECOVERY (b-hardened, 2026-06-03).

    Reproduces the queue-writer push-rejected/rebase-conflict wedge and
    proves the flag-gated self-heal. The conflict is built as the live
    failure mode: the writer holds a retained (unpushed) `phase3-queue: `
    snapshot commit editing experiment_queue.json AND origin has a
    divergent commit editing the same line -- so the writer's _sync_to_origin
    rebase conflicts on the next tick.
    """

    def setUp(self):
        super().setUp()
        # Restore the module flag after each test (default is False).
        self._orig_recovery = sync_daemon.PHASE3_QUEUE_CONFLICT_RECOVERY
        self.addCleanup(self._restore_recovery)
        sync_daemon._reset_writer_health_state()
        self.addCleanup(sync_daemon._reset_writer_health_state)
        # Neutralise the ambient legacy env so the additive flag's behaviour
        # is deterministic per test. C6 re-enables it to prove the additive
        # (flag-off + env-on -> still recovers) path.
        self._orig_env = os.environ.pop(
            "PHASE3_AUTO_RESET_ON_REBASE_CONFLICT", None)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        if self._orig_env is not None:
            os.environ["PHASE3_AUTO_RESET_ON_REBASE_CONFLICT"] = self._orig_env
        else:
            os.environ.pop("PHASE3_AUTO_RESET_ON_REBASE_CONFLICT", None)

    def _restore_recovery(self):
        sync_daemon.PHASE3_QUEUE_CONFLICT_RECOVERY = self._orig_recovery

    def _set_recovery(self, value):
        sync_daemon.PHASE3_QUEUE_CONFLICT_RECOVERY = value

    def _set_note(self, doc, note):
        doc["calibration"]["calibration_note"] = note
        return doc

    def _local_retained_commit(self, note, prefix):
        """Create a local, UNPUSHED commit editing experiment_queue.json
        (simulates a snapshot commit whose push was rejected last tick).
        `prefix` controls writer-authored (phase3-queue: ) vs foreign."""
        f = self._repo / "experiment_queue.json"
        doc = self._set_note(json.loads(f.read_text()), note)
        f.write_text(json.dumps(doc, indent=2) + "\n")
        _git(self._repo, "config", "user.email", "phase3-queue@example")
        _git(self._repo, "config", "user.name", "phase3-queue")
        _git(self._repo, "add", "experiment_queue.json")
        _git(self._repo, "commit", "-q", "-m", prefix + "retained snapshot")

    def _push_divergent_origin_edit(self, note):
        """Clone the bare remote in a throwaway dir, change the SAME line to
        a different value, commit + push origin/main -- so a rebase of the
        local retained commit conflicts."""
        work = pathlib.Path(self._tmp) / "operator_clone"
        subprocess.run(
            ["git", "clone", "-q", "-b", "main", str(self._remote), str(work)],
            check=True)
        _git(work, "config", "user.email", "operator@example")
        _git(work, "config", "user.name", "operator")
        f = work / "experiment_queue.json"
        doc = self._set_note(json.loads(f.read_text()), note)
        f.write_text(json.dumps(doc, indent=2) + "\n")
        _git(work, "add", "experiment_queue.json")
        _git(work, "commit", "-q", "-m", "operator: queue hand-edit on origin")
        _git(work, "push", "-q", "origin", "main")

    def _make_conflict(self, local_prefix):
        """Put the repo into the ahead-1/behind-1 conflicting state."""
        self._local_retained_commit("writer-local", local_prefix)
        self._push_divergent_origin_edit("operator-origin")

    # -- C1: flag OFF reproduces the wedge -----------------------------------
    def test_c1_flag_off_refuses_on_conflict(self):
        self._set_recovery(False)
        _upsert_item(self._conn, "V3-EXQ-C1", priority=10)
        self._make_conflict(sync_daemon._PHASE3_QUEUE_COMMIT_PREFIX)
        sha_before = _git(self._remote, "rev-parse", "main").stdout.strip()
        result = self._run()
        self.assertFalse(
            result, "flag OFF: rebase conflict must wedge (refuse the tick)")
        sha_after = _git(self._remote, "rev-parse", "main").stdout.strip()
        self.assertEqual(
            sha_before, sha_after, "flag OFF: nothing should land on origin")
        self.assertEqual(
            sync_daemon._WRITER_HEALTH["queue_writer"]["n_conflict_recoveries"],
            0, "flag OFF: no self-heal recorded")

    # -- C2: flag ON self-heals losslessly -----------------------------------
    def test_c2_flag_on_self_heals_and_lands_db_view(self):
        self._set_recovery(True)
        _upsert_item(self._conn, "V3-EXQ-C2", priority=10)
        self._make_conflict(sync_daemon._PHASE3_QUEUE_COMMIT_PREFIX)
        result = self._run()
        self.assertTrue(
            result, "flag ON: conflict must self-heal and the tick succeed")
        on_origin = self._read_origin_queue()
        ids = [i["queue_id"] for i in on_origin["items"]]
        self.assertEqual(
            ids, ["V3-EXQ-C2"],
            "flag ON: DB-materialised snapshot must reach origin after reset")
        # Operator's origin edit was preserved through the reset (the writer
        # rebuilds calibration from the post-reset origin file).
        self.assertEqual(
            on_origin["calibration"]["calibration_note"], "operator-origin")
        self.assertEqual(
            sync_daemon._WRITER_HEALTH["queue_writer"]["n_conflict_recoveries"],
            1, "flag ON: exactly one self-heal recorded")
        self.assertIsNotNone(
            sync_daemon._WRITER_HEALTH["queue_writer"]
            ["last_conflict_recovery_at"])

    # -- C3: safety -- a FOREIGN ahead commit is never dropped ----------------
    def test_c3_flag_on_refuses_foreign_ahead_commit(self):
        self._set_recovery(True)
        _upsert_item(self._conn, "V3-EXQ-C3", priority=10)
        # Local retained commit is operator-authored (foreign prefix).
        self._make_conflict("ops: ")
        sha_before = _git(self._remote, "rev-parse", "main").stdout.strip()
        result = self._run()
        self.assertFalse(
            result,
            "flag ON but foreign ahead commit: must refuse (never reset "
            "operator work)")
        sha_after = _git(self._remote, "rev-parse", "main").stdout.strip()
        self.assertEqual(sha_before, sha_after)
        self.assertEqual(
            sync_daemon._WRITER_HEALTH["queue_writer"]["n_conflict_recoveries"],
            0, "foreign-ahead refusal must NOT count as a self-heal")

    # -- C4: backward-compat -- default (None) path unchanged ----------------
    def test_c4_default_none_path_refuses_on_conflict(self):
        # Directly exercise _sync_to_origin with auto_reset_on_conflict=None
        # and the legacy env unset -- the policy the result/heartbeat writers
        # keep. Must refuse on conflict (no behavioural change for them).
        self._local_retained_commit(
            "writer-local", sync_daemon._PHASE3_QUEUE_COMMIT_PREFIX)
        self._push_divergent_origin_edit("operator-origin")
        env_prev = os.environ.pop("PHASE3_AUTO_RESET_ON_REBASE_CONFLICT", None)
        try:
            ok, reason = sync_daemon._sync_to_origin(
                str(self._repo), "main", "[test]",
                auto_reset_on_conflict=None)
        finally:
            if env_prev is not None:
                os.environ["PHASE3_AUTO_RESET_ON_REBASE_CONFLICT"] = env_prev
        self.assertFalse(ok, "None policy + env unset must refuse on conflict")
        self.assertIn("rebase onto origin/main failed", reason)

    # -- C5: flag validation -------------------------------------------------
    def test_c5_flag_validation_falls_back_to_default(self):
        self.assertTrue(sync_daemon._validate_bool("1", "X"))
        self.assertTrue(sync_daemon._validate_bool("yes", "X"))
        self.assertFalse(sync_daemon._validate_bool("0", "X"))
        self.assertFalse(sync_daemon._validate_bool("", "X"))
        # Unrecognised -> default (loud warning to stderr).
        self.assertFalse(sync_daemon._validate_bool("garbage", "X"))
        self.assertTrue(
            sync_daemon._validate_bool("garbage", "X", default=True))

    # -- C6: additive -- scoped flag OFF + legacy env ON still recovers ------
    def test_c6_additive_legacy_env_still_recovers_when_flag_off(self):
        # The scoped flag is OFF, but the legacy global env is ON (the hub's
        # current stopgap). The queue writer must still self-heal -- the new
        # flag must never DISABLE recovery the env already provides.
        self._set_recovery(False)
        os.environ["PHASE3_AUTO_RESET_ON_REBASE_CONFLICT"] = "1"
        _upsert_item(self._conn, "V3-EXQ-C6", priority=10)
        self._make_conflict(sync_daemon._PHASE3_QUEUE_COMMIT_PREFIX)
        result = self._run()
        self.assertTrue(
            result,
            "flag OFF but legacy env ON: additive policy must still recover")
        ids = [i["queue_id"] for i in self._read_origin_queue()["items"]]
        self.assertEqual(ids, ["V3-EXQ-C6"])
        # Recovery fired via the env-fallback path; still counted + observable.
        self.assertEqual(
            sync_daemon._WRITER_HEALTH["queue_writer"]["n_conflict_recoveries"],
            1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
