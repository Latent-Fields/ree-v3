"""Contracts for the 2026-05-31 state-change-triggered commit redesign
of sync_daemon.phase3_heartbeat_writer.

The legacy writer committed every SYNC_INTERVAL seconds whenever the
materialised bytes differed from HEAD, producing a `phase3-heartbeats: 1
telemetry file(s)` commit storm dominated by stale `last_tick_utc`
timestamp updates. The redesign commits only on meaningful fleet state
transitions, with a debounce floor (no more than one commit per N
minutes) and a liveness floor (force a commit every M minutes so
external viewers see git timestamps proving the fleet is alive).

Each contract here is a property of the redesign that must hold; if it
breaks, the noise-reduction promise is gone.

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


def _hb_payload(machine, *, last_tick_utc="2026-05-31T07:00:00Z"):
    return {
        "machine": machine,
        "hostname": "host-" + machine,
        "last_tick_utc": last_tick_utc,
        "state": "running",
        "current_exq": None,
        "progress": {},
        "gpu": {},
        "recent_completed": [],
    }


def _status_payload(machine, *, queue_id=None, idle=False,
                    runner_pid=12345, completed=None):
    return {
        "schema_version": "v1",
        "machine": machine,
        "runner_pid": runner_pid,
        "last_updated": "2026-05-31T07:00:00Z",
        "idle": idle,
        "current": {"queue_id": queue_id} if queue_id else {},
        "queue": [],
        "completed": completed or [],
    }


class _Fixture(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="phase3_hb_state_")
        self._remote = _bare_remote(self._tmp)
        self._repo = _seeded_clone(self._tmp, self._remote)
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        sync_daemon.PHASE3_HEARTBEAT_WRITER_READY = False
        sync_daemon._reset_phase3_heartbeat_state()
        # Virtual clock: tests advance _now to test debounce + liveness
        # without sleeping. Reset between tests.
        self._now = [1_000_000.0]  # seconds, monotonic-ish
        self._orig_clock = sync_daemon._phase3_heartbeat_now
        sync_daemon._phase3_heartbeat_now = lambda: self._now[0]

    def tearDown(self):
        sync_daemon._phase3_heartbeat_now = self._orig_clock
        sync_daemon.PHASE3_HEARTBEAT_WRITER_READY = False
        sync_daemon._reset_phase3_heartbeat_state()
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _advance(self, seconds):
        self._now[0] += seconds

    def _run(self):
        sync_daemon.PHASE3_HEARTBEAT_WRITER_READY = True
        try:
            return sync_daemon.phase3_heartbeat_writer(
                self._conn,
                ree_assembly_path=str(self._repo),
                branch="master")
        finally:
            sync_daemon.PHASE3_HEARTBEAT_WRITER_READY = False

    def _seed(self, machine, *, queue_id=None, idle=False,
              runner_pid=12345, completed=None,
              last_tick_utc="2026-05-31T07:00:00Z"):
        hb = _hb_payload(machine, last_tick_utc=last_tick_utc)
        st = _status_payload(
            machine, queue_id=queue_id, idle=idle,
            runner_pid=runner_pid, completed=completed)
        db.upsert_heartbeat(
            self._conn, machine,
            state="running",
            current_exq=queue_id,
            progress={}, gpu={},
            payload_json=json.dumps(hb))
        db.record_status_payload(
            self._conn, machine, json.dumps(st))

    def _head_sha(self):
        return _git(self._remote, "rev-parse", "master").stdout.strip()

    def _head_subject(self):
        return _git(self._remote, "log", "-1", "--format=%s",
                    "master").stdout.strip()


# ---------------------------------------------------------------------------
# (a) No commit when nothing changes across ticks
# ---------------------------------------------------------------------------

class NoCommitOnUnchangedState(_Fixture):
    """After the initial seed-commit, repeated ticks with identical DB
    state must NOT add commits. This is the core noise-reduction
    property: the commit storm was caused by per-tick commits even when
    only last_tick_utc moved."""

    def test_initial_commit_then_quiet(self):
        self._seed("ree-cloud-1", queue_id="V3-EXQ-100")
        self.assertTrue(self._run())  # initial commit
        sha_init = self._head_sha()

        # Same DB state, advance 60s -- well inside the liveness window.
        for _ in range(5):
            self._advance(60)
            self.assertTrue(self._run())
        self.assertEqual(
            self._head_sha(), sha_init,
            "no commit should fire while state and bytes are unchanged")

    def test_last_tick_utc_alone_does_not_commit(self):
        """The whole point of the redesign: a payload whose only diff is
        last_tick_utc must NOT commit. The runner pushes a fresh
        last_tick_utc every ~30s; pre-redesign those drove the storm."""
        self._seed("ree-cloud-1", queue_id="V3-EXQ-100",
                   last_tick_utc="2026-05-31T07:00:00Z")
        self.assertTrue(self._run())  # initial commit
        sha_init = self._head_sha()

        # Update last_tick_utc only (10 ticks, each 60s of virtual time).
        for i in range(10):
            self._advance(60)
            self._seed(
                "ree-cloud-1", queue_id="V3-EXQ-100",
                last_tick_utc="2026-05-31T07:%02d:00Z" % (i + 1))
            self.assertTrue(self._run())
        self.assertEqual(
            self._head_sha(), sha_init,
            "last_tick_utc churn alone must not fire any commit")


# ---------------------------------------------------------------------------
# (b) Commit on the first material state change
# ---------------------------------------------------------------------------

class CommitOnMaterialChange(_Fixture):

    def test_queue_id_start_commits(self):
        # Initial seed: machine idle.
        self._seed("ree-cloud-1", queue_id=None)
        self.assertTrue(self._run())
        sha_init = self._head_sha()

        # Past the debounce window so a real change fires immediately.
        self._advance(sync_daemon.PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL + 1)
        self._seed("ree-cloud-1", queue_id="V3-EXQ-201")
        self.assertTrue(self._run())

        new_sha = self._head_sha()
        self.assertNotEqual(new_sha, sha_init,
                            "queue_id start must commit")
        subject = self._head_subject()
        self.assertIn("ree-cloud-1", subject)
        self.assertIn("V3-EXQ-201", subject)
        self.assertIn("started", subject)

    def test_queue_id_finish_names_result(self):
        self._seed("ree-cloud-2", queue_id="V3-EXQ-202")
        self.assertTrue(self._run())

        # Past debounce window: experiment finishes with FAIL.
        self._advance(sync_daemon.PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL + 1)
        self._seed("ree-cloud-2", queue_id=None,
                   completed=[{"queue_id": "V3-EXQ-202", "result": "FAIL"}])
        self.assertTrue(self._run())

        subject = self._head_subject()
        self.assertIn("ree-cloud-2", subject)
        self.assertIn("V3-EXQ-202", subject)
        self.assertIn("FAIL", subject)
        self.assertIn("ran", subject)

    def test_runner_pid_change_is_a_restart_event(self):
        self._seed("ree-cloud-3", queue_id=None, runner_pid=1111)
        self.assertTrue(self._run())

        self._advance(sync_daemon.PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL + 1)
        self._seed("ree-cloud-3", queue_id=None, runner_pid=2222)
        self.assertTrue(self._run())

        subject = self._head_subject()
        self.assertIn("ree-cloud-3", subject)
        self.assertIn("restart", subject)
        self.assertIn("2222", subject)

    def test_new_machine_appearance_is_a_state_change(self):
        self._seed("ree-cloud-1", queue_id=None)
        self.assertTrue(self._run())

        self._advance(sync_daemon.PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL + 1)
        self._seed("ree-cloud-4", queue_id=None)
        self.assertTrue(self._run())

        subject = self._head_subject()
        self.assertIn("ree-cloud-4", subject)
        self.assertIn("added", subject)


# ---------------------------------------------------------------------------
# (c) Debounce holds further commits for the debounce interval
# ---------------------------------------------------------------------------

class DebounceHoldsFurtherCommits(_Fixture):

    def test_change_inside_window_holds(self):
        """A second state change inside the debounce window must NOT
        produce a second commit. The post-window tick commits whatever
        has accumulated."""
        # Initial commit.
        self._seed("ree-cloud-1", queue_id=None)
        self.assertTrue(self._run())
        # First real change (past debounce window): commit fires.
        self._advance(sync_daemon.PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL + 1)
        self._seed("ree-cloud-1", queue_id="V3-EXQ-300")
        self.assertTrue(self._run())
        sha_after_first = self._head_sha()

        # Inside debounce window: another change. Must NOT commit.
        debounce = sync_daemon.PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL
        self._advance(debounce / 2.0)
        self._seed(
            "ree-cloud-1", queue_id=None,
            completed=[{"queue_id": "V3-EXQ-300", "result": "PASS"}])
        self.assertTrue(self._run())
        self.assertEqual(
            self._head_sha(), sha_after_first,
            "debounce must hold the second commit inside the window")

        # Cross the debounce boundary: the pending change commits.
        self._advance(debounce)
        self.assertTrue(self._run())
        self.assertNotEqual(
            self._head_sha(), sha_after_first,
            "post-window tick must commit the held change")
        subject = self._head_subject()
        self.assertIn("V3-EXQ-300", subject)
        self.assertIn("PASS", subject)


# ---------------------------------------------------------------------------
# (d) Liveness commit fires after the liveness interval of zero changes
# ---------------------------------------------------------------------------

class LivenessFloor(_Fixture):

    def _reseed_with_fresh_tick(self, tag):
        """Re-write the payload with a fresh last_tick_utc so the
        going-silent detector does not interfere with this test (we
        want to isolate the liveness floor here)."""
        self._seed("ree-cloud-1", queue_id="V3-EXQ-400",
                   last_tick_utc="2026-05-31T07:%02d:00Z" % tag)

    def test_idle_fleet_commits_at_liveness_interval(self):
        self._reseed_with_fresh_tick(0)
        self.assertTrue(self._run())  # initial
        sha_init = self._head_sha()

        # Stay just under liveness floor: no commit.
        liveness = sync_daemon.PHASE3_HEARTBEAT_LIVENESS_INTERVAL
        self._advance(liveness * 0.9)
        self._reseed_with_fresh_tick(1)
        self.assertTrue(self._run())
        self.assertEqual(self._head_sha(), sha_init,
                         "pre-liveness tick must not commit")

        # Cross the liveness floor: forced liveness commit.
        self._advance(liveness * 0.2)
        self._reseed_with_fresh_tick(2)
        self.assertTrue(self._run())
        self.assertNotEqual(
            self._head_sha(), sha_init,
            "tick past liveness floor must force a liveness commit")
        subject = self._head_subject()
        self.assertIn("liveness", subject)
        self.assertIn("machine", subject)


# ---------------------------------------------------------------------------
# (e) Commit-message naming matches the state-change pattern
# ---------------------------------------------------------------------------

class CommitMessageNaming(_Fixture):
    """Commit messages must name what actually changed, not just the
    file count -- this is what the user sees in `git log` and is the
    main reason for the redesign (`1 telemetry file(s)` is uninformative
    noise)."""

    def test_start_message_format(self):
        self._seed("cloud-a", queue_id=None)
        self.assertTrue(self._run())
        self._advance(sync_daemon.PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL + 1)
        self._seed("cloud-a", queue_id="V3-EXQ-500")
        self.assertTrue(self._run())
        subj = self._head_subject()
        self.assertTrue(subj.startswith("phase3-heartbeats: "), subj)
        self.assertIn("cloud-a -> V3-EXQ-500 started", subj)

    def test_finish_message_format(self):
        self._seed("cloud-b", queue_id="V3-EXQ-501")
        self.assertTrue(self._run())
        self._advance(sync_daemon.PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL + 1)
        self._seed(
            "cloud-b", queue_id=None,
            completed=[{"queue_id": "V3-EXQ-501", "result": "PASS"}])
        self.assertTrue(self._run())
        subj = self._head_subject()
        self.assertIn("cloud-b ran V3-EXQ-501 -> PASS", subj)

    def test_liveness_message_format(self):
        self._seed("cloud-c", queue_id="V3-EXQ-502",
                   last_tick_utc="2026-05-31T07:00:00Z")
        self._seed("cloud-d", queue_id="V3-EXQ-503",
                   last_tick_utc="2026-05-31T07:00:00Z")
        self.assertTrue(self._run())
        # Re-seed with fresh last_tick_utc so the going-silent transition
        # does not fire and steal the commit -- we want a clean liveness
        # subject here.
        self._advance(sync_daemon.PHASE3_HEARTBEAT_LIVENESS_INTERVAL + 1)
        self._seed("cloud-c", queue_id="V3-EXQ-502",
                   last_tick_utc="2026-05-31T08:00:00Z")
        self._seed("cloud-d", queue_id="V3-EXQ-503",
                   last_tick_utc="2026-05-31T08:00:00Z")
        self.assertTrue(self._run())
        subj = self._head_subject()
        self.assertIn("liveness tick", subj)
        self.assertIn("2 machine(s) active", subj)

    def test_multi_change_message_lists_events(self):
        self._seed("cloud-e", queue_id=None)
        self._seed("cloud-f", queue_id=None)
        self.assertTrue(self._run())
        self._advance(sync_daemon.PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL + 1)
        self._seed("cloud-e", queue_id="V3-EXQ-600")
        self._seed("cloud-f", queue_id="V3-EXQ-601")
        self.assertTrue(self._run())
        subj = self._head_subject()
        # Both machines named in the same commit (batched within tick).
        self.assertIn("cloud-e", subj)
        self.assertIn("V3-EXQ-600", subj)
        self.assertIn("cloud-f", subj)
        self.assertIn("V3-EXQ-601", subj)


# ---------------------------------------------------------------------------
# (f) Going-silent transition is detected and tracked once
# ---------------------------------------------------------------------------

class GoingSilentTransition(_Fixture):
    """A machine whose heartbeat.last_tick_utc stops advancing past the
    staleness threshold must be flagged silent exactly once (transition
    edge, not every tick), and must flag came-back if it resumes."""

    def test_went_silent_then_came_back(self):
        self._seed("cloud-z", queue_id=None,
                   last_tick_utc="2026-05-31T08:00:00Z")
        self.assertTrue(self._run())

        # Hold the same last_tick_utc and advance past the stale-after
        # threshold + debounce so the silent transition fires.
        stale = sync_daemon.PHASE3_HEARTBEAT_STALE_AFTER
        debounce = sync_daemon.PHASE3_HEARTBEAT_DEBOUNCE_INTERVAL
        self._advance(max(stale, debounce) + 1)
        self.assertTrue(self._run())  # detect stale; tick advances cache
        # Cross past debounce so the commit can fire on the silent event.
        self._advance(debounce + 1)
        self.assertTrue(self._run())
        subj_silent = self._head_subject()
        self.assertIn("cloud-z", subj_silent)
        self.assertIn("went silent", subj_silent)

        # Same payload again: no second went-silent commit.
        sha_at_silent = self._head_sha()
        self._advance(debounce + 1)
        self.assertTrue(self._run())
        self.assertEqual(
            self._head_sha(), sha_at_silent,
            "went-silent must not re-fire while the machine stays silent")

        # Resume: last_tick_utc advances, came-back fires past debounce.
        self._advance(debounce + 1)
        self._seed("cloud-z", queue_id=None,
                   last_tick_utc="2026-05-31T09:00:00Z")
        self.assertTrue(self._run())
        subj_back = self._head_subject()
        self.assertIn("cloud-z", subj_back)
        self.assertIn("came back", subj_back)


# ---------------------------------------------------------------------------
# (g) Skip-decision ticks never touch the working tree
# ---------------------------------------------------------------------------

class SkipDecisionDoesNotDirtyTree(_Fixture):
    """When the writer decides to skip a tick (no state change inside
    liveness window, or debounce holding), it must NOT materialise any
    file in the working tree. Touching the working tree would block the
    NEXT commit-decision tick on the clean-tree check, deadlocking the
    writer."""

    def test_skip_tick_leaves_working_tree_clean(self):
        self._seed("cloud-x", queue_id="V3-EXQ-700",
                   last_tick_utc="2026-05-31T07:00:00Z")
        self.assertTrue(self._run())  # initial commit

        # Update payload bytes (new last_tick_utc) but NO state change.
        self._advance(60)
        self._seed("cloud-x", queue_id="V3-EXQ-700",
                   last_tick_utc="2026-05-31T07:01:00Z")
        self.assertTrue(self._run())  # skip-decision tick

        status = _git(self._repo, "status", "--porcelain").stdout
        self.assertEqual(
            status.strip(), "",
            "skip-decision ticks must leave the working tree clean "
            "(found dirt: %r)" % status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
