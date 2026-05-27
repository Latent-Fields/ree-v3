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


def _seeded_clone(parent, remote, name="asm", seed_msg="seed"):
    """Init a working clone, seed with a README, push so origin/master
    points to a real commit.

    `seed_msg` defaults to a non-`phase3: `-prefixed subject so the
    foreign-commit check sees the seed as foreign in tests that probe
    that path. StaleOriginRefTest overrides this to use a writer-
    authored subject so the seed clears _check_ahead_writer_authored
    and the non-FF rejection path is the one actually exercised."""
    repo = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)], check=True)
    _git(repo, "config", "user.email", "test@example")
    _git(repo, "config", "user.name", "test")
    (repo / "README.md").write_text("seed\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", seed_msg)
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

    # Subclasses override to control the seed commit's subject (see
    # LOW-A from the 2026-05-27 review: StaleOriginRefTest needs the
    # seed to be writer-authored so it doesn't get caught by the
    # foreign-commit check before reaching the non-FF push path it
    # claims to exercise).
    SEED_MSG = "seed"

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_smoke_")
        self._saved_spool = os.environ.get("COORDINATOR_SPOOL_DIR")
        os.environ["COORDINATOR_SPOOL_DIR"] = self._tmp
        self._remote = _bare_remote(self._tmp)
        self._repo = _seeded_clone(
            self._tmp, self._remote, name="asm",
            seed_msg=self.SEED_MSG)
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


class ForeignCommitRejectionTest(_WriterFixture):
    """HIGH-2 from the 2026-05-27 review: the writer's push must publish
    ONLY writer-authored commits. An operator hand-commit on the hub's
    master (between two writer ticks) would otherwise get carried along
    with the writer's next push, attributing it to sync_daemon and
    breaking the post-cutover invariant "all REE_assembly commits
    attributable to the sync_daemon path."

    Two scenarios -- the foreign commit can be ahead of origin via
    either path:
      - new-commit branch (`else` arm): writer commits its own work on
        top of an existing operator commit; push would carry both.
      - diff-cached short-circuit case (b): a re-spool tick sees no
        diff, ahead>0 (the operator commit alone), would push the
        operator commit.

    Both scenarios are tested here.
    """

    def _operator_commits_on_hub(self, message="operator hand edit"):
        """Add an empty commit on the hub's working clone (not pushed).
        Simulates an operator hand-edit between writer ticks."""
        _git(self._repo, "config", "user.email", "operator@example")
        _git(self._repo, "config", "user.name", "operator")
        _git(self._repo, "commit", "--allow-empty", "-m", message)

    def test_else_branch_refuses_when_operator_commit_is_ahead(self):
        # Tick 1: clean push.
        run_id_1 = "v3_smoke_h2_else_1"
        _spool_manifest(run_id_1, "V3-EXQ-H2A", self._conn)
        self.assertTrue(self._run_writer())
        self.assertIn(
            "evidence/experiments/%s.json" % run_id_1, self._origin_log())

        # Operator-authored commit lands on hub master, NOT pushed.
        self._operator_commits_on_hub("operator hand edit while writer idle")

        # Tick 2: new manifest -> else branch -> writer commits + tries
        # to push. Foreign-check must catch the operator commit between
        # origin and the writer's new commit.
        run_id_2 = "v3_smoke_h2_else_2"
        _spool_manifest(run_id_2, "V3-EXQ-H2B", self._conn)
        result = self._run_writer()

        self.assertFalse(
            result,
            "writer must refuse to push when operator commit is ahead")
        # manifest #2 still in spool; its committed_at unset.
        self.assertIn(run_id_2, self._spool_ids())
        self.assertIsNone(self._committed_at(run_id_2))
        # manifest #1 still committed_at-marked from tick 1.
        self.assertIsNotNone(self._committed_at(run_id_1))
        # Origin master must NOT have manifest #2 OR the operator commit.
        log = self._origin_log()
        self.assertNotIn(
            "evidence/experiments/%s.json" % run_id_2, log,
            "writer's manifest #2 must not be on origin")
        operator_on_origin = _git(
            self._remote, "log", "--format=%s", "master",
        ).stdout
        self.assertNotIn(
            "operator hand edit", operator_on_origin,
            "operator commit must not be on origin")

    def test_case_b_refuses_when_foreign_commit_is_the_unpushed_one(self):
        # Tick 1: clean push.
        run_id = "v3_smoke_h2_caseb"
        raw = _spool_manifest(run_id, "V3-EXQ-H2C", self._conn)
        self.assertTrue(self._run_writer())
        self.assertEqual(self._spool_ids(), [])

        # Re-spool the same manifest (duplicate POST /result simulation),
        # plus operator commit lands ahead of origin.
        manifest_spool.write_manifest(
            run_id, raw,
            received_at="2026-05-27T21:00:00Z", sha256_hex="sha")
        self._conn.execute(
            "UPDATE results SET committed_at=NULL WHERE run_id=?",
            (run_id,))
        self._operator_commits_on_hub("operator hand edit pre case-b")

        # Tick 2: writes file (already on HEAD via writer's tick-1
        # commit), git add no-diff, fetch refreshes ref, rev-list ahead
        # = 1 (the operator commit), case (b) entered. Foreign-check
        # must catch it.
        result = self._run_writer()

        self.assertFalse(
            result,
            "writer must refuse case (b) push when ahead commit is foreign")
        self.assertIn(
            run_id, self._spool_ids(),
            "spool entry must survive the refused tick")
        self.assertIsNone(
            self._committed_at(run_id),
            "writer must not mark committed when push refused")
        # Operator commit must NOT be on origin.
        origin_log = _git(
            self._remote, "log", "--format=%s", "master",
        ).stdout
        self.assertNotIn(
            "operator hand edit pre case-b", origin_log)


class StaleOriginRefTest(_WriterFixture):
    """HIGH-1 from the 2026-05-27 review: phase3_git_writer's
    ahead-of-origin guard reads the LOCAL remote-tracking ref
    `origin/<branch>` without fetching first. If origin has been advanced
    or force-pushed externally since the hub's last fetch, the local ref
    is stale and `rev-list origin/<branch>..HEAD` lies (returns 0 when
    the writer's commit actually isn't on origin anymore).

    Setup: writer pushes successfully (tick 1), then an external party
    force-pushes the bare remote to a different history that no longer
    contains the writer's commit. The same manifest re-spools
    (simulating a duplicate /result POST or a crash-before-delete).

    Without the pre-fetch fix:
      - rev-list returns 0 (local origin/master still points at the
        writer's commit), writer marks committed without push, drains
        spool. Bytes are no longer on origin -- silent failure.

    With the fix:
      - fetch refreshes the ref, rev-list returns 1, writer attempts to
        push HEAD, push fails (non-FF vs the force-pushed remote), writer
        refuses the tick. Spool retained, committed_at unset by THIS
        tick (the prior tick 1 already set it, so we check spool +
        absence of bytes on origin instead).
    """

    # LOW-A from the 2026-05-27 review: author the seed commit with the
    # writer's prefix so the foreign-commit check (HIGH-2) does NOT
    # catch the seed as foreign in tick 2. Without this override the
    # test still passes -- but via the wrong protection: the foreign-
    # check refuses before the non-FF push is ever attempted, so the
    # stale-ref ahead-of-origin guard this test docstring describes is
    # not the one actually exercised.
    SEED_MSG = "phase3: seed"

    def _force_remote_to_independent_history(self):
        """Force-push the bare remote to a sibling-built history that
        does NOT contain the writer's commit. After this, the writer's
        local origin/master ref still points at the writer's commit
        (the successful push from tick 1 set it), but origin's actual
        master is at a different SHA."""
        sibling = pathlib.Path(self._tmp) / "force_sibling"
        sibling.mkdir()
        _git(self._tmp, "init", "-q", "-b", "master", str(sibling))
        _git(sibling, "config", "user.email", "force@example")
        _git(sibling, "config", "user.name", "force")
        (sibling / "external.txt").write_text("force-push payload\n")
        _git(sibling, "add", "external.txt")
        _git(sibling, "commit", "-q", "-m", "external force push")
        _git(sibling, "remote", "add", "origin", str(self._remote))
        # Force-push: replace bare remote's master with this independent
        # history. The writer's commit is no longer reachable from
        # origin/master on the remote.
        _git(sibling, "push", "-q", "--force", "origin", "master")

    def test_stale_origin_ref_does_not_drain_spool(self):
        run_id = "v3_smoke_stale_ref"
        raw = _spool_manifest(run_id, "V3-EXQ-STALE", self._conn)

        # Tick 1: clean push to bare remote. After this, the writer's
        # local origin/master is freshly updated by the successful push.
        self.assertTrue(self._run_writer())
        self.assertIn(
            "evidence/experiments/%s.json" % run_id,
            self._origin_log())
        self.assertIsNotNone(self._committed_at(run_id))
        self.assertEqual(self._spool_ids(), [])

        # External actor force-pushes the bare remote. The writer's
        # local origin/master ref is now stale -- it still points at
        # the writer's commit, but the bare remote does not.
        self._force_remote_to_independent_history()

        # Re-spool the same manifest. Simulates a duplicate POST /result,
        # or a prior tick that committed_at-marked but crashed before
        # delete_manifest. Clear committed_at so the writer would try to
        # mark it again (tick 2 has work to do).
        manifest_spool.write_manifest(
            run_id, raw,
            received_at="2026-05-27T20:00:00Z", sha256_hex="sha")
        self._conn.execute(
            "UPDATE results SET committed_at=NULL WHERE run_id=?",
            (run_id,))
        self.assertEqual(self._spool_ids(), [run_id])
        self.assertIsNone(self._committed_at(run_id))

        # Tick 2 with the fix in place: fetch refreshes the ref, ahead>0
        # detected, push attempted, push fails non-FF, writer refuses.
        # The spool entry survives; the writer's commit is NOT on origin.
        result = self._run_writer()
        self.assertFalse(
            result,
            "writer must refuse tick when stale ref reveals ahead>0 "
            "and push fails")
        self.assertNotIn(
            "evidence/experiments/%s.json" % run_id,
            self._origin_log(),
            "manifest must not be on the (force-pushed) origin")
        self.assertIsNone(
            self._committed_at(run_id),
            "writer must not mark committed when push fails")
        self.assertIn(
            run_id, self._spool_ids(),
            "spool entry must survive the refused tick")


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


class MissingResultsRowTest(_WriterFixture):
    """MED-1 from the 2026-05-27 review: a spool entry without a
    matching `results` row produced a silent rowcount-0 UPDATE and the
    spool was dropped anyway. Bytes reached origin, DB had no record.

    Fix surfaces a per-tick WARN naming the missing run_ids while still
    proceeding (bytes ARE on origin; retaining the spool would loop
    indefinitely). This test verifies both: warning surfaces, spool
    drains, push happens.
    """

    def test_spool_without_results_row_warns_and_drains(self):
        # Write a manifest into the spool WITHOUT calling db.record_result.
        run_id = "v3_smoke_med1_no_row"
        raw = json.dumps({"run_id": run_id, "outcome": "PASS",
                          "machine": "test"}).encode("utf-8")
        manifest_spool.write_manifest(
            run_id, raw,
            received_at="2026-05-27T22:00:00Z", sha256_hex="sha")
        self.assertEqual(self._spool_ids(), [run_id])
        # No row in results yet.
        self.assertIsNone(self._committed_at(run_id))
        row = self._conn.execute(
            "SELECT 1 FROM results WHERE run_id=?", (run_id,)).fetchone()
        self.assertIsNone(row, "precondition: results row absent")

        # Capture stderr so we can verify the WARN surfaces.
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            result = self._run_writer()
        stderr = buf.getvalue()

        self.assertTrue(result, "tick should succeed (bytes reach origin)")
        self.assertIn("invariant violation", stderr)
        self.assertIn(run_id, stderr)
        # Bytes ARE on origin.
        self.assertIn(
            "evidence/experiments/%s.json" % run_id, self._origin_log())
        # Spool drained.
        self.assertEqual(self._spool_ids(), [])
        # Row still absent (writer doesn't backfill); UPDATE no-oped.
        row = self._conn.execute(
            "SELECT 1 FROM results WHERE run_id=?", (run_id,)).fetchone()
        self.assertIsNone(row)


class BatchSizeValidationTest(unittest.TestCase):
    """MED-2: PHASE3_BATCH_SIZE=0 or negative used to slice an empty
    batch from a non-empty spool, then refuse with the misleading
    "no manifests staged" message. _validate_batch_size now catches
    invalid values at module load and falls back to default 32 with a
    loud warning.
    """

    def test_default_when_unset(self):
        self.assertEqual(sync_daemon._validate_batch_size("32"), 32)

    def test_explicit_positive(self):
        self.assertEqual(sync_daemon._validate_batch_size("64"), 64)
        self.assertEqual(sync_daemon._validate_batch_size("1"), 1)

    def test_zero_falls_back(self):
        self.assertEqual(sync_daemon._validate_batch_size("0"), 32)

    def test_negative_falls_back(self):
        self.assertEqual(sync_daemon._validate_batch_size("-5"), 32)

    def test_non_integer_falls_back(self):
        self.assertEqual(sync_daemon._validate_batch_size("abc"), 32)
        self.assertEqual(sync_daemon._validate_batch_size(""), 32)
        self.assertEqual(sync_daemon._validate_batch_size(None), 32)

    def test_custom_default(self):
        self.assertEqual(
            sync_daemon._validate_batch_size("bogus", default=100), 100)


class AtomicWorkingTreeWriteTest(_WriterFixture):
    """LOW-2: the writer's working-tree manifest write was not atomic
    (plain `open(target, "wb").write(raw)`), so a mid-write crash could
    leave a truncated file that the immediately-following `git add`
    would happily stage. Fix uses tmp + os.replace.

    Hard to inject a real crash mid-write in a unit test, so this
    test verifies the observable side effects of the new path:
      - no `.phase3.tmp` file is left behind after a successful tick
      - the written file's bytes equal the spool's raw payload
    """

    def test_no_tmp_file_left_after_successful_tick(self):
        run_id = "v3_smoke_low2_atomic"
        raw = _spool_manifest(run_id, "V3-EXQ-LOW2", self._conn)
        self.assertTrue(self._run_writer())
        # Working tree should have the canonical file but no .phase3.tmp.
        evidence_dir = self._repo / "evidence" / "experiments"
        files = list(evidence_dir.iterdir())
        names = [f.name for f in files]
        self.assertIn("%s.json" % run_id, names)
        self.assertNotIn("%s.json.phase3.tmp" % run_id, names)
        # Bytes round-trip cleanly.
        self.assertEqual(
            (evidence_dir / ("%s.json" % run_id)).read_bytes(), raw)


class EnvKnobValidationTest(unittest.TestCase):
    """LOW-C: extend _validate_batch_size's pattern to the other env
    knobs that can silently misdirect the writer when malformed."""

    def test_branch_name_default_when_blank(self):
        self.assertEqual(
            sync_daemon._validate_branch_name(
                "", "PHASE3_ASSEMBLY_BRANCH", "master"),
            "master")
        self.assertEqual(
            sync_daemon._validate_branch_name(
                "   ", "PHASE3_ASSEMBLY_BRANCH", "master"),
            "master")

    def test_branch_name_rejects_whitespace_and_separators(self):
        self.assertEqual(
            sync_daemon._validate_branch_name(
                "feature branch", "PHASE3_ASSEMBLY_BRANCH", "master"),
            "master")
        self.assertEqual(
            sync_daemon._validate_branch_name(
                "refs/heads/main", "PHASE3_ASSEMBLY_BRANCH", "master"),
            "master")
        self.assertEqual(
            sync_daemon._validate_branch_name(
                "back\\slash", "PHASE3_ASSEMBLY_BRANCH", "master"),
            "master")

    def test_branch_name_accepts_valid(self):
        self.assertEqual(
            sync_daemon._validate_branch_name(
                "main", "PHASE3_REE_V3_BRANCH", "main"),
            "main")
        self.assertEqual(
            sync_daemon._validate_branch_name(
                "  master  ", "PHASE3_ASSEMBLY_BRANCH", "main"),
            "master")
        self.assertEqual(
            sync_daemon._validate_branch_name(
                "feature-x.1", "PHASE3_REE_V3_BRANCH", "main"),
            "feature-x.1")

    def test_repo_relpath_default_when_blank(self):
        self.assertEqual(
            sync_daemon._validate_repo_relpath(
                "", "PHASE3_QUEUE_RELPATH", "experiment_queue.json"),
            "experiment_queue.json")

    def test_repo_relpath_rejects_absolute(self):
        self.assertEqual(
            sync_daemon._validate_repo_relpath(
                "/etc/passwd", "PHASE3_QUEUE_RELPATH", "experiment_queue.json"),
            "experiment_queue.json")

    def test_repo_relpath_rejects_parent_escape(self):
        self.assertEqual(
            sync_daemon._validate_repo_relpath(
                "../outside.json", "PHASE3_QUEUE_RELPATH",
                "experiment_queue.json"),
            "experiment_queue.json")
        self.assertEqual(
            sync_daemon._validate_repo_relpath(
                "evidence/../../escape.json", "PHASE3_QUEUE_RELPATH",
                "experiment_queue.json"),
            "experiment_queue.json")

    def test_repo_relpath_accepts_nested_subdir(self):
        self.assertEqual(
            sync_daemon._validate_repo_relpath(
                "evidence/experiments/queue.json", "PHASE3_QUEUE_RELPATH",
                "experiment_queue.json"),
            "evidence/experiments/queue.json")

    def test_float_default_when_unparseable(self):
        self.assertEqual(
            sync_daemon._validate_float(
                "abc", "SYNC_INTERVAL", 60.0),
            60.0)
        self.assertEqual(
            sync_daemon._validate_float(
                None, "SYNC_INTERVAL", 60.0),
            60.0)

    def test_float_rejects_non_positive(self):
        self.assertEqual(
            sync_daemon._validate_float("0", "SYNC_INTERVAL", 60.0),
            60.0)
        self.assertEqual(
            sync_daemon._validate_float("-5", "SYNC_INTERVAL", 60.0),
            60.0)

    def test_float_accepts_valid(self):
        self.assertEqual(
            sync_daemon._validate_float("30", "SYNC_INTERVAL", 60.0),
            30.0)
        self.assertEqual(
            sync_daemon._validate_float("0.5", "SYNC_INTERVAL", 60.0),
            0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
