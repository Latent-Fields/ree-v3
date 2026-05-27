"""Offline smoke harness for sync_daemon.phase3_git_writer.

Covers the cutover-gate cases PHASE3_CUTOVER.md asks for that
test_manifest_spool.py does not yet exercise:

  - Non-fast-forward push rejection: writer returns False, spool retained,
    committed_at not set, no autostash performed.
  - Non-FF then retry: the second tick must NOT silently mark committed
    against an unpushed local commit. Surfaces the "byte-identical to
    local tree" short-circuit (sync_daemon.py:292-300) which currently
    cannot distinguish "already on origin" from "in unpushed local
    commit". A failing assertion here is a real Phase 3 blocker.
  - Batch boundary at PHASE3_BATCH_SIZE: 33 pending -> 32 committed +
    1 retained.
  - Multi-manifest single commit: one commit, one push, N files touched.
  - Meta sidecar's manifest_relpath hint is consulted end-to-end (the
    derive_evidence_relpath tests cover the helper in isolation, not the
    writer's wiring through meta).

Run: /opt/local/bin/python3 test_phase3_writer_smoke.py
or:  /opt/local/bin/python3 -m unittest test_phase3_writer_smoke

All printed text is ASCII-only (Windows cp1252 safety).
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
import manifest_spool  # noqa: E402
import sync_daemon  # noqa: E402


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------

def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=check,
    )


def _bare_remote(parent):
    remote = pathlib.Path(parent) / "remote.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _seeded_clone(parent, remote, name="asm"):
    """Init a working clone, seed with a README, push so origin/master
    points to a real commit."""
    repo = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)], check=True)
    _git(repo, "config", "user.email", "test@example")
    _git(repo, "config", "user.name", "test")
    (repo / "README.md").write_text("seed\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "seed")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "master")
    return repo


def _spool_manifest(run_id, queue_id, conn, *, manifest_relpath=None,
                    extra=None):
    """Write a results row + spool entry mirroring what POST /result does
    on the live coordinator. Returns the raw bytes for later assertions."""
    body = {
        "run_id": run_id,
        "queue_id": queue_id,
        "outcome": "PASS",
        "machine": "test",
    }
    if extra:
        body.update(extra)
    raw = json.dumps(body, sort_keys=True).encode("utf-8")
    db.record_result(conn, run_id, queue_id, "test", "PASS", "sha", len(raw))
    manifest_spool.write_manifest(
        run_id, raw,
        manifest_relpath=manifest_relpath,
        received_at="2026-05-27T18:00:00Z",
        sha256_hex="sha")
    return raw


class _WriterFixture(unittest.TestCase):
    """Common per-test scaffolding: bare-remote + working clone + DB +
    spool root. tearDown restores env to whatever it was before."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_smoke_")
        self._saved_spool = os.environ.get("COORDINATOR_SPOOL_DIR")
        os.environ["COORDINATOR_SPOOL_DIR"] = self._tmp
        self._remote = _bare_remote(self._tmp)
        self._repo = _seeded_clone(self._tmp, self._remote, name="asm")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._queue = os.path.join(self._tmp, "queue.json")
        with open(self._queue, "w", encoding="utf-8") as fh:
            json.dump({"items": []}, fh)
        # Default-False; each test flips and restores explicitly.
        sync_daemon.PHASE3_GIT_WRITER_READY = False

    def tearDown(self):
        sync_daemon.PHASE3_GIT_WRITER_READY = False
        self._conn.close()
        if self._saved_spool is not None:
            os.environ["COORDINATOR_SPOOL_DIR"] = self._saved_spool
        else:
            os.environ.pop("COORDINATOR_SPOOL_DIR", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run_writer(self):
        sync_daemon.PHASE3_GIT_WRITER_READY = True
        try:
            return sync_daemon.phase3_git_writer(
                self._conn, self._queue,
                ree_assembly_path=str(self._repo))
        finally:
            sync_daemon.PHASE3_GIT_WRITER_READY = False

    def _committed_at(self, run_id):
        row = self._conn.execute(
            "SELECT committed_at FROM results WHERE run_id=?",
            (run_id,)).fetchone()
        return row["committed_at"] if row is not None else None

    def _spool_ids(self):
        return list(manifest_spool.list_pending_run_ids())

    def _origin_log(self):
        return _git(self._remote, "log", "--name-only", "--pretty=format:%H"
                    ).stdout


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class NonFastForwardRejectionTest(_WriterFixture):
    """The Phase 3 promise: the writer never autostashes. A non-FF push
    must fail the tick loudly, leave the spool intact, and NOT mark
    committed_at."""

    def _advance_remote(self, message="external commit"):
        """Push an unrelated commit to the bare remote from a sibling
        clone so the writer's HEAD becomes non-FF."""
        sibling = pathlib.Path(self._tmp) / "sibling"
        _git(self._tmp, "clone", "-q", str(self._remote), str(sibling))
        _git(sibling, "config", "user.email", "ext@example")
        _git(sibling, "config", "user.name", "ext")
        (sibling / "ext.txt").write_text("from another writer\n")
        _git(sibling, "add", "ext.txt")
        _git(sibling, "commit", "-q", "-m", message)
        _git(sibling, "push", "-q", "origin", "master")

    def test_non_ff_push_returns_false_and_retains_spool(self):
        run_id = "v3_smoke_nonff_001"
        _spool_manifest(run_id, "V3-EXQ-001", self._conn)
        self._advance_remote()

        result = self._run_writer()

        self.assertFalse(result, "writer should return False on non-FF")
        self.assertIsNone(
            self._committed_at(run_id),
            "committed_at must NOT be set when push rejected")
        self.assertEqual(
            self._spool_ids(), [run_id],
            "spool entry must survive a rejected push for the next tick")
        # Manifest must not have made it to origin (only the sibling's
        # ext.txt did).
        self.assertNotIn(
            "evidence/experiments/%s.json" % run_id,
            self._origin_log())

    def test_non_ff_retry_without_operator_action_does_not_silently_commit(
            self):
        """Ahead-of-origin guard: after a failed push, the writer's local
        history holds an unpushed commit with the manifest bytes. If the
        operator does NOTHING between ticks, the writer must NOT mark
        committed_at against bytes that never reached origin.

        With the guard at sync_daemon.py the diff-cached short-circuit
        only fires when `git rev-list --count origin/<branch>..HEAD`
        returns 0. When ahead>0 the writer either:
          (i) re-pushes the existing local commit (succeeds if origin
              has caught up to or is reachable from that commit), OR
          (ii) refuses the tick (push still rejected; spool retained).

        In this scenario origin has DIVERGED (sibling pushed `ext.txt`),
        so retry-without-reset stays on path (ii): writer returns False,
        spool retained, committed_at unset, no bytes on origin.
        """
        run_id = "v3_smoke_nonff_silent"
        _spool_manifest(run_id, "V3-EXQ-SILENT", self._conn)
        self._advance_remote()

        # Tick 1: push rejected. Local HEAD now has an unpushed phase3
        # commit; spool retained.
        self.assertFalse(self._run_writer())
        ahead = _git(self._repo, "rev-list", "--count",
                     "origin/master..HEAD").stdout.strip()
        self.assertEqual(
            ahead, "1",
            "local HEAD should be 1 commit ahead after rejected push")

        # Tick 2 WITHOUT operator action. Guard must refuse to mark.
        self.assertFalse(self._run_writer())

        on_origin = (
            "evidence/experiments/%s.json" % run_id in self._origin_log())
        self.assertFalse(
            on_origin,
            "manifest must not be on origin without operator action")
        self.assertIsNone(
            self._committed_at(run_id),
            "committed_at must NOT be set when origin lacks the bytes")
        self.assertIn(
            run_id, self._spool_ids(),
            "spool entry must survive an unsafe tick")

    def test_non_ff_retry_with_operator_reset_succeeds(self):
        """After a failed push, the writer's local history holds an
        unpushed commit. On retry, `git add` produces no diff because the
        bytes are already on HEAD's tree, and the diff-cached-quiet
        short-circuit (sync_daemon.py:292-300) marks committed_at +
        drains the spool WITHOUT a push. Bytes never reach origin.

        Asserts the WANTED behaviour: either the writer detects the
        unpushed-local-commit and re-pushes, or it refuses to short-
        circuit. If this test fails, the short-circuit needs an
        ahead-of-origin guard before Phase 3 is safe.
        """
        run_id = "v3_smoke_nonff_retry"
        _spool_manifest(run_id, "V3-EXQ-002", self._conn)
        self._advance_remote()

        # Tick 1: push rejected.
        self.assertFalse(self._run_writer())

        # Operator action: resolve the divergence by hand (Phase 3
        # rollback discipline). Here we simulate the manual `git
        # pull --ff-only` documented in PHASE3_CUTOVER.md by hard-
        # resetting the writer's branch back to origin/master so the
        # unpushed phase3 commit is dropped. The next tick should
        # then re-stage and push the manifest cleanly.
        _git(self._repo, "fetch", "-q", "origin")
        _git(self._repo, "reset", "--hard", "-q", "origin/master")

        # Tick 2: must push the manifest to origin and mark committed.
        result = self._run_writer()
        self.assertTrue(result, "post-reset retry should succeed")
        self.assertIsNotNone(
            self._committed_at(run_id),
            "committed_at must be set after a successful retry")
        self.assertIn(
            "evidence/experiments/%s.json" % run_id,
            self._origin_log())
        self.assertEqual(self._spool_ids(), [])


class BatchBoundaryTest(_WriterFixture):
    """PHASE3_BATCH_SIZE bounds tick latency. 33 pending -> 32 committed
    on tick 1 + 1 retained for tick 2."""

    def test_batch_size_honoured(self):
        # Spool one more than the batch.
        n = sync_daemon.PHASE3_BATCH_SIZE + 1
        run_ids = ["v3_smoke_batch_%03d" % i for i in range(n)]
        for i, rid in enumerate(run_ids):
            _spool_manifest(rid, "V3-EXQ-BATCH-%03d" % i, self._conn)
        self.assertEqual(len(self._spool_ids()), n)

        self.assertTrue(self._run_writer())

        remaining = self._spool_ids()
        self.assertEqual(
            len(remaining), 1,
            "exactly one manifest should remain after a bounded tick")
        # Committed count matches the batch limit.
        committed = [
            rid for rid in run_ids
            if self._committed_at(rid) is not None]
        self.assertEqual(len(committed), sync_daemon.PHASE3_BATCH_SIZE)
        # The retained one is the last (sorted-order spool yields
        # alphabetical; the writer takes the first BATCH_SIZE).
        self.assertEqual(remaining, [run_ids[-1]])

        # Tick 2 drains the remainder.
        self.assertTrue(self._run_writer())
        self.assertEqual(self._spool_ids(), [])
        self.assertIsNotNone(self._committed_at(run_ids[-1]))


class MultiManifestSingleCommitTest(_WriterFixture):
    """One commit per tick, regardless of batch size > 1. Verifies the
    operator-visible log msg shape AND that all manifests reach origin
    in a single push."""

    def test_three_manifests_one_commit(self):
        run_ids = [
            "v3_smoke_multi_a", "v3_smoke_multi_b", "v3_smoke_multi_c",
        ]
        for i, rid in enumerate(run_ids):
            _spool_manifest(rid, "V3-EXQ-MULTI-%d" % i, self._conn)

        self.assertTrue(self._run_writer())

        # All three manifests on origin.
        log_files = self._origin_log()
        for rid in run_ids:
            self.assertIn("evidence/experiments/%s.json" % rid, log_files)

        # Exactly one phase3 commit added on top of the seed.
        commits = _git(
            self._remote, "log", "--pretty=format:%s", "master",
        ).stdout.splitlines()
        phase3_commits = [c for c in commits if c.startswith("phase3:")]
        self.assertEqual(
            len(phase3_commits), 1,
            "writer must batch into a single commit per tick")
        self.assertIn(
            "%d v3 result manifest(s)" % len(run_ids), phase3_commits[0])

        # committed_at populated for every run.
        for rid in run_ids:
            self.assertIsNotNone(self._committed_at(rid))


class MetaHintHonouredTest(_WriterFixture):
    """The runner is expected to pass manifest_relpath in the POST /result
    payload (PHASE3_CUTOVER.md "deferred to follow-up PRs" notwithstanding,
    the writer already wires it through meta). End-to-end: a hint that
    routes to a subdirectory under evidence/experiments/ must land at
    that exact path on origin."""

    def test_meta_relpath_lands_at_subdir(self):
        run_id = "v3_smoke_meta_hint"
        rel = "evidence/experiments/runs/v3_smoke_meta_hint/manifest.json"
        _spool_manifest(
            run_id, "V3-EXQ-META", self._conn, manifest_relpath=rel)

        self.assertTrue(self._run_writer())
        self.assertIn(rel, self._origin_log())
        # Default-path file should NOT exist.
        self.assertNotIn(
            "evidence/experiments/%s.json" % run_id, self._origin_log())
        self.assertIsNotNone(self._committed_at(run_id))


if __name__ == "__main__":
    unittest.main(verbosity=2)
