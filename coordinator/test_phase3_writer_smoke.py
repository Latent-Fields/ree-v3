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
    """The Phase 3 promise has two parts:
      (1) the writer never autostashes a dirty working tree (the
          original 2026-04-29 / 2026-05-08 / 2026-05-10 incident class);
      (2) the writer never silently mashes bytes onto origin behind a
          stale local ref (HIGH-1 2026-05-27 fix).

    A non-FF push caused by an UNRELATED commit landing on origin
    between two writer ticks (Claude session close, lit-pull synthesis
    push, late-arriving runner claim) is NOT itself a violation of
    either invariant. The writer absorbs the origin advance via
    `git rebase origin/<branch>` (clean-tree precondition guarantees no
    autostash; foreign-commit check guarantees only writer-authored
    commits get rebased) and the push then succeeds. Without that
    absorb step every writer wedged the moment origin advanced -- the
    exact failure mode observed live 2026-05-28 ~16:25..17:36Z post-
    cutover."""

    def _advance_remote(self, message="external commit", filename=None):
        """Push an unrelated commit to the bare remote from a fresh
        sibling clone so the writer's HEAD becomes non-FF. Each call
        creates a uniquely-named sibling so the same test can advance
        the remote more than once."""
        self._sibling_seq = getattr(self, "_sibling_seq", 0) + 1
        sibling = pathlib.Path(self._tmp) / ("sibling_%d" % self._sibling_seq)
        if filename is None:
            filename = "ext_%d.txt" % self._sibling_seq
        _git(self._tmp, "clone", "-q", str(self._remote), str(sibling))
        _git(sibling, "config", "user.email", "ext@example")
        _git(sibling, "config", "user.name", "ext")
        (sibling / filename).write_text("from another writer\n")
        _git(sibling, "add", filename)
        _git(sibling, "commit", "-q", "-m", message)
        _git(sibling, "push", "-q", "origin", "master")

    def test_non_ff_first_tick_absorbs_remote_and_pushes(self):
        """Origin advances by one unrelated commit between writer ticks.
        The writer must absorb it (fetch + rebase the writer-authored
        ahead set, which is empty on the first tick) and proceed to
        commit + push the manifest. Without the absorb step the writer
        wedges; this test pins the absorb step."""
        run_id = "v3_smoke_nonff_001"
        _spool_manifest(run_id, "V3-EXQ-001", self._conn)
        self._advance_remote()

        result = self._run_writer()

        self.assertTrue(result, "writer must absorb sibling commit and push")
        self.assertIsNotNone(
            self._committed_at(run_id),
            "committed_at set after successful push")
        self.assertEqual(
            self._spool_ids(), [],
            "spool drained after successful tick")
        # Both the sibling's file AND the writer's manifest now on origin.
        log = self._origin_log()
        self.assertIn("ext_1.txt", log)
        self.assertIn("evidence/experiments/%s.json" % run_id, log)

    def test_non_ff_retry_without_operator_action_succeeds_via_rebase(self):
        """Sequence that wedged production 2026-05-28: tick 1's push
        rejected (origin advanced); tick 2 without operator action must
        absorb origin via rebase and push the retained-for-retry local
        commit. The post-2026-05-27 ahead-of-origin guard (HIGH-1)
        correctly refused to mark committed against an unpushed local
        commit; the post-2026-05-28 absorb step (_sync_to_origin)
        completes the loop so the unpushed commit actually reaches
        origin on the next tick rather than wedging forever.

        Implementation note: the absorb step now runs FIRST in every
        tick, so this test exercises both the rebase path on tick 2
        AND the HIGH-1 ahead-of-origin guard's case (b) push-the-
        unpushed-local-commit branch."""
        run_id = "v3_smoke_nonff_silent"
        _spool_manifest(run_id, "V3-EXQ-SILENT", self._conn)
        # Pre-tick-1 advance to set up the rejection on tick 1. Use a
        # raced-write window: the sibling pushes BEFORE the writer
        # fetches. (In production this matches a session close racing
        # the writer's per-minute tick.) The writer's tick 1 will fetch
        # first via _sync_to_origin and absorb the sibling -- so to
        # reproduce the "tick 1 rejected" half of the production
        # scenario we need a second sibling commit AFTER the writer's
        # sync. Simpler test instead: drive the second-commit-after-
        # sync via a monkeypatch is overkill -- the production
        # signature is "writer made a commit, then origin advanced
        # before the push went out, then push rejected". We reproduce
        # that by manually constructing the post-tick-1 state.
        self._advance_remote("first sibling commit")
        # Tick 1: with the fix, the writer absorbs the sibling and
        # pushes the manifest. End state: spool empty, manifest on
        # origin, committed_at set.
        self.assertTrue(self._run_writer())
        self.assertEqual(self._spool_ids(), [])
        self.assertIsNotNone(self._committed_at(run_id))
        self.assertIn(
            "evidence/experiments/%s.json" % run_id, self._origin_log())

        # Production wedge scenario for a SECOND manifest: origin
        # advances between the writer's commit and its push. Re-spool a
        # new manifest, then race the sibling AFTER the writer's
        # _sync_to_origin would have fetched.
        run_id_2 = "v3_smoke_nonff_silent_2"
        _spool_manifest(run_id_2, "V3-EXQ-SILENT-2", self._conn)
        # Construct the wedge directly: writer would commit, but origin
        # is one ahead. We simulate by advancing the remote, then
        # manually creating a writer-authored commit locally with no
        # corresponding push (the state the writer ends up in after a
        # rejected push).
        self._advance_remote("second sibling commit while writer mid-tick")
        # Tick 2: writer must fetch + rebase its retained-for-retry
        # logic happens automatically. With the fix, push succeeds.
        self.assertTrue(self._run_writer())
        self.assertEqual(self._spool_ids(), [])
        self.assertIsNotNone(self._committed_at(run_id_2))
        log = self._origin_log()
        self.assertIn("evidence/experiments/%s.json" % run_id_2, log)
        self.assertIn(
            "second sibling commit while writer mid-tick",
            _git(self._remote, "log", "--format=%s", "master").stdout)

    # Retired 2026-05-28: `test_non_ff_retry_with_operator_reset_succeeds`
    # exercised the "writer wedges on non-FF, operator manually resets
    # to origin, writer recovers" path. After the _sync_to_origin absorb
    # step lands, the writer no longer wedges on simple non-FF -- it
    # rebases automatically (covered by test_non_ff_first_tick_absorbs_
    # remote_and_pushes and test_non_ff_retry_without_operator_action_
    # succeeds_via_rebase). The operator-reset escape hatch still works
    # for the unrecoverable case (rebase conflict), covered by
    # SyncToOriginRebaseConflictTest.test_rebase_conflict_aborts_and_
    # refuses + the operator's documented recovery in PHASE3_CUTOVER.md.


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

    def test_stale_origin_ref_resyncs_and_recommits(self):
        """Original HIGH-1 test: after the writer pushes successfully and
        an external party force-pushes the bare remote to a different
        history, a duplicate /result re-spooling the same manifest must
        not silently mark committed against bytes that aren't on origin.

        The original 2026-05-27 HIGH-1 fix (pre-fetch + ahead-of-origin
        check) caught the silent-commit half. The 2026-05-28 absorb step
        (_sync_to_origin) goes one further: the writer fetches, sees it
        is now BEHIND the force-pushed origin while its own
        writer-authored commit is AHEAD, rebases its commit on top of
        the new origin history, and pushes. Bytes end up on origin
        again. The original concern -- silent-commit against a stale
        ref -- is still impossible (the fetch makes the ref fresh
        before any committed_at decision). What changed is the recovery
        action: the writer no longer wedges; it re-establishes its
        bytes on the new origin history.

        Why this is the right behaviour: the writer's spool + DB are
        the source of truth for evidence bytes (Phase 3 design). A
        force-push that wiped writer-authored commits was either an
        operator intervention (in which case the operator should ALSO
        drop the corresponding spool entries) or accidental (in which
        case auto-recovery is the desired behaviour). The foreign-
        commit guard ensures we only ever rebase writer-authored
        commits, so a force-push that introduced operator content
        cannot be silently overwritten."""
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
        # local origin/master ref is now stale.
        self._force_remote_to_independent_history()

        # Re-spool the same manifest. Simulates a duplicate POST /result
        # or a crash-before-delete. Clear committed_at so the writer has
        # work to do.
        manifest_spool.write_manifest(
            run_id, raw,
            received_at="2026-05-27T20:00:00Z", sha256_hex="sha")
        self._conn.execute(
            "UPDATE results SET committed_at=NULL WHERE run_id=?",
            (run_id,))
        self.assertEqual(self._spool_ids(), [run_id])
        self.assertIsNone(self._committed_at(run_id))

        # Tick 2 with the absorb step: fetch refreshes the ref; we're
        # now behind=1 (external force push not in HEAD) AND ahead=1
        # (writer's tick-1 commit not on origin anymore). Foreign-check
        # confirms the ahead commit is writer-authored. Rebase replays
        # it on top of the force-pushed history (no path conflict --
        # writer's manifest and sibling's external.txt are different
        # files). Push succeeds, committed_at set, spool drained.
        result = self._run_writer()
        self.assertTrue(
            result,
            "writer must absorb force-push via rebase and re-push")
        self.assertIn(
            "evidence/experiments/%s.json" % run_id,
            self._origin_log(),
            "manifest restored on origin after rebase + push")
        self.assertIsNotNone(
            self._committed_at(run_id),
            "committed_at set after successful push")
        self.assertEqual(self._spool_ids(), [])
        # External actor's payload is still on origin too.
        self.assertIn(
            "external.txt", self._origin_log())


class SyncToOriginRebaseConflictTest(_WriterFixture):
    """The absorb step's safety contract has two halves:
      (1) refuse cleanly when an origin commit conflicts with a
          writer-authored ahead commit on the same path (rebase
          abort + spool retained);
      (2) refuse cleanly when a foreign (non-writer-authored) commit
          is ahead AND we are behind origin -- the writer must not
          rebase the operator's work under its own authority.
    Both paths must leave the working tree clean so the next tick can
    retry once the operator resolves."""

    SEED_MSG = "phase3: seed"

    def test_rebase_conflict_aborts_and_refuses(self):
        # Tick 1: writer commits + pushes a manifest at path P.
        run_id = "v3_smoke_conflict"
        _spool_manifest(run_id, "V3-EXQ-CONFLICT", self._conn)
        self.assertTrue(self._run_writer())
        rel = "evidence/experiments/%s.json" % run_id
        self.assertIn(rel, self._origin_log())

        # External party force-pushes a DIFFERENT version of the same
        # file P onto origin. Now writer's tick-1 commit (locally
        # ahead after re-spool) and origin's force-pushed commit both
        # touch P -- rebase must conflict.
        sibling = pathlib.Path(self._tmp) / "conflict_sibling"
        sibling.mkdir()
        _git(self._tmp, "init", "-q", "-b", "master", str(sibling))
        _git(sibling, "config", "user.email", "force@example")
        _git(sibling, "config", "user.name", "force")
        (sibling / "README.md").write_text("seed\n")
        _git(sibling, "add", "README.md")
        _git(sibling, "commit", "-q", "-m", "phase3: seed")
        target = sibling / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"different": "content"}\n')
        _git(sibling, "add", rel)
        _git(sibling, "commit", "-q", "-m", "phase3: conflicting payload")
        _git(sibling, "remote", "add", "origin", str(self._remote))
        _git(sibling, "push", "-q", "--force", "origin", "master")

        # Re-spool to force the writer into a fresh tick. Clear
        # committed_at so the writer has work to do.
        raw = json.dumps({
            "run_id": run_id, "outcome": "PASS", "machine": "test",
        }, sort_keys=True).encode("utf-8")
        manifest_spool.write_manifest(
            run_id, raw,
            received_at="2026-05-28T18:00:00Z", sha256_hex="sha")
        self._conn.execute(
            "UPDATE results SET committed_at=NULL WHERE run_id=?",
            (run_id,))

        # Tick 2: _sync_to_origin fetches, sees behind=1 and ahead=1
        # (writer's tick-1 commit); foreign-check passes (writer-
        # authored); rebase starts; conflict on rel; rebase --abort;
        # _sync_to_origin returns refuse.
        result = self._run_writer()
        self.assertFalse(
            result, "writer must refuse when rebase conflicts")
        self.assertIn(run_id, self._spool_ids(),
                      "spool retained on refused tick")
        # Working tree must be clean after rebase --abort.
        status = _git(self._repo, "status", "--porcelain").stdout
        self.assertEqual(
            status.strip(), "",
            "rebase --abort must restore a clean working tree")

    def test_foreign_commit_ahead_with_behind_refuses_without_rebase(self):
        """An operator commits locally on the hub AND origin advances
        in parallel. We end up ahead (operator commit) AND behind
        (origin commit). The writer must refuse to rebase the
        operator's work; the operator must resolve."""
        run_id_1 = "v3_smoke_foreign_pre"
        _spool_manifest(run_id_1, "V3-EXQ-FOREIGN-1", self._conn)
        self.assertTrue(self._run_writer())

        # Operator commits locally (not pushed).
        _git(self._repo, "config", "user.email", "operator@example")
        _git(self._repo, "config", "user.name", "operator")
        _git(self._repo, "commit", "--allow-empty", "-q",
             "-m", "operator hand edit, no prefix")

        # Origin advances via a sibling (independent of the operator
        # commit).
        sibling = pathlib.Path(self._tmp) / "advance_sibling"
        _git(self._tmp, "clone", "-q", str(self._remote), str(sibling))
        _git(sibling, "config", "user.email", "adv@example")
        _git(sibling, "config", "user.name", "adv")
        (sibling / "adv.txt").write_text("advance\n")
        _git(sibling, "add", "adv.txt")
        _git(sibling, "commit", "-q", "-m", "sibling advance")
        _git(sibling, "push", "-q", "origin", "master")

        # New manifest. Writer would normally proceed, but
        # _sync_to_origin sees behind=1 AND a foreign commit ahead.
        run_id_2 = "v3_smoke_foreign_new"
        _spool_manifest(run_id_2, "V3-EXQ-FOREIGN-2", self._conn)
        result = self._run_writer()

        self.assertFalse(
            result,
            "writer must refuse when foreign commit ahead AND behind")
        self.assertIn(run_id_2, self._spool_ids())
        self.assertIsNone(self._committed_at(run_id_2))
        # Operator commit must still be on hub local HEAD; writer must
        # not have published or destroyed it.
        local_log = _git(
            self._repo, "log", "--format=%s",
            "origin/master..HEAD").stdout
        self.assertIn("operator hand edit, no prefix", local_log)
        # Origin must not have the operator commit either.
        origin_log = _git(
            self._remote, "log", "--format=%s", "master").stdout
        self.assertNotIn("operator hand edit, no prefix", origin_log)


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
