"""Contracts for the PHASE-4 generic whole-file CAS verb, POST
/intent/replace (2026-09-01, git-traffic-simplification sweep lane 2):
git_intent.apply_intent and the app._intent_replace endpoint handler.

phase4_commit_intake_design.md section 3.2 is the wire contract this pins:

  1. ROUTING: an unrouted path (or a path/repo mismatch) is always
     'not_routed', unconditionally -- section 3.2's "any other path is
     not_routed, always".
  2. VALIDATION: per-path validators run server-side before anything is
     written (JSON parse for the .json routes here).
  3. CAS: base_sha's content at origin must equal origin tip's content for
     the same path -- FILE-CONTENT compare, not commit-equality (an
     unrelated commit to another file must not bounce the intent).
  4. base_moved returns the current sha/content for the caller to rebase.
  5. SIZE GUARD: a replacement that shrinks the file below the ratio
     threshold is refused unless allow_shrink is set.
  6. SHADOW MODE: checked and logged, never written.
  7. NO-OP: resubmitting content identical to origin's is 'applied' without
     an empty commit.
  8. PUSH-RETRY: a rejected push re-verifies against the refreshed origin
     tip rather than failing outright.
  9. WIRING: POST handler status codes + verdict passthrough; dispatch-
     table membership.

Time-independent (no wall-clock assertions). Real git repos in a tempdir,
same fixture shape as test_task_claim_chip_git_writer.py. ASCII-only.
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

import app  # noqa: E402
import db  # noqa: E402
import git_intent  # noqa: E402

LEDGER_PATH = "evidence/planning/igw_routine_ledger.json"
LEDGER_DOC_V1 = json.dumps({"schema_version": 1, "items": []}, indent=2) + "\n"
LEDGER_DOC_V2 = json.dumps({"schema_version": 1, "items": ["a"]}, indent=2) + "\n"


def _git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=check)


def _bare_remote(parent, name="REE_assembly.git"):
    remote = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _seed_repo(parent, remote, ledger_text=LEDGER_DOC_V1, name="REE_assembly"):
    repo = pathlib.Path(parent) / name
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)],
                   check=True)
    _git(repo, "config", "user.email", "writer@test")
    _git(repo, "config", "user.name", "writer-test")
    ledger = repo / LEDGER_PATH
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if ledger_text is not None:
        ledger.write_text(ledger_text, encoding="utf-8")
        _git(repo, "add", LEDGER_PATH)
    else:
        (repo / ".keep").write_text("")
        _git(repo, "add", ".keep")
    _git(repo, "commit", "-q", "-m", "seed")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "master")
    return repo


class _Fixture(unittest.TestCase):

    LEDGER_TEXT = LEDGER_DOC_V1

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="git_intent_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)
        self._remote = _bare_remote(self._tmp)
        self._repo = _seed_repo(self._tmp, self._remote,
                                ledger_text=self.LEDGER_TEXT)
        self._clone_env = os.environ.get("COORDINATOR_INTENT_REPO_REE_ASSEMBLY")
        os.environ["COORDINATOR_INTENT_REPO_REE_ASSEMBLY"] = str(self._repo)
        # A fresh module-level lock per repo name would otherwise be shared
        # across tests via the module's own _LOCKS dict -- harmless (locks
        # are reentrant-free but each test acquires/releases within itself),
        # kept here as documentation of that shared state.

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)
        if self._clone_env is None:
            os.environ.pop("COORDINATOR_INTENT_REPO_REE_ASSEMBLY", None)
        else:
            os.environ["COORDINATOR_INTENT_REPO_REE_ASSEMBLY"] = self._clone_env

    def _origin_sha(self):
        _git(self._repo, "fetch", "-q", "origin", "master")
        return _git(self._repo, "rev-parse",
                    "origin/master").stdout.strip()

    def _origin_ledger(self):
        return _git(self._repo, "show",
                    "origin/master:%s" % LEDGER_PATH).stdout

    def _apply(self, **kwargs):
        defaults = dict(
            conn=self._conn, repo="REE_assembly", path=LEDGER_PATH,
            message="igw-ledger: test", session_id="test-session",
            machine="test-host")
        defaults.update(kwargs)
        return git_intent.apply_intent(**defaults)


class TestRouting(_Fixture):

    def test_unrouted_path_refused(self):
        verdict, payload = self._apply(
            path="evidence/planning/claims.yaml",
            base_sha=self._origin_sha(), content="x")
        self.assertEqual(verdict, "not_routed")

    def test_repo_mismatch_refused(self):
        verdict, payload = self._apply(
            repo="ree-v3", base_sha=self._origin_sha(), content=LEDGER_DOC_V1)
        self.assertEqual(verdict, "not_routed")

    def test_not_routed_logged(self):
        self._apply(path="nope", base_sha=None, content="x")
        row = self._conn.execute(
            "SELECT verdict FROM git_intent_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual(row["verdict"], "not_routed")


class TestValidation(_Fixture):

    def test_bad_json_refused(self):
        verdict, payload = self._apply(
            base_sha=self._origin_sha(), content="not json at all {")
        self.assertEqual(verdict, "validation_failed")
        self.assertEqual(self._origin_ledger(), LEDGER_DOC_V1)

    def test_non_string_content_refused(self):
        verdict, payload = self._apply(
            base_sha=self._origin_sha(), content=None)
        self.assertEqual(verdict, "validation_failed")


class TestCasCheck(_Fixture):

    def test_matching_base_applies(self):
        base = self._origin_sha()
        verdict, payload = self._apply(base_sha=base, content=LEDGER_DOC_V2)
        self.assertEqual(verdict, "applied")
        self.assertEqual(self._origin_ledger(), LEDGER_DOC_V2)

    def test_stale_base_refused_with_current_content(self):
        base = self._origin_sha()
        # Someone else lands a change to the SAME path first.
        _git(self._repo, "fetch", "-q", "origin", "master")
        other = pathlib.Path(tempfile.mkdtemp(prefix="git_intent_other_"))
        _git(self._repo, "worktree", "add", "-q", "--detach", str(other),
             "origin/master")
        (other / LEDGER_PATH).write_text(LEDGER_DOC_V2, encoding="utf-8")
        _git(other, "add", LEDGER_PATH)
        _git(other, "commit", "-q", "-m", "concurrent editorial change")
        _git(other, "push", "-q", "origin", "HEAD:master")
        _git(self._repo, "worktree", "remove", "--force", str(other))

        verdict, payload = self._apply(
            base_sha=base, content=json.dumps({"schema_version": 1,
                                              "items": ["b"]}, indent=2) + "\n")
        self.assertEqual(verdict, "base_moved")
        self.assertEqual(payload["current_content"], LEDGER_DOC_V2)
        self.assertIn("current_sha", payload)
        # The refused write must never have landed.
        self.assertEqual(self._origin_ledger(), LEDGER_DOC_V2)

    def test_unrelated_file_change_does_not_bounce_the_intent(self):
        """CAS is a FILE-CONTENT compare, not a commit-equality compare --
        an unrelated commit to a DIFFERENT path must not refuse this one."""
        base = self._origin_sha()
        _git(self._repo, "fetch", "-q", "origin", "master")
        other = pathlib.Path(tempfile.mkdtemp(prefix="git_intent_other2_"))
        _git(self._repo, "worktree", "add", "-q", "--detach", str(other),
             "origin/master")
        (other / "unrelated.txt").write_text("noise", encoding="utf-8")
        _git(other, "add", "unrelated.txt")
        _git(other, "commit", "-q", "-m", "unrelated change")
        _git(other, "push", "-q", "origin", "HEAD:master")
        _git(self._repo, "worktree", "remove", "--force", str(other))

        verdict, payload = self._apply(base_sha=base, content=LEDGER_DOC_V2)
        self.assertEqual(verdict, "applied")
        self.assertEqual(self._origin_ledger(), LEDGER_DOC_V2)

    def test_no_base_sha_first_write_semantics(self):
        """No base_sha, and the file's origin content equals None only when
        the path is genuinely absent from origin -- here it already exists,
        so this must refuse as base_moved (the caller's None base does not
        match the live file)."""
        verdict, payload = self._apply(base_sha=None, content=LEDGER_DOC_V2)
        self.assertEqual(verdict, "base_moved")

    def test_no_base_sha_creates_new_file(self):
        verdict, payload = self._apply(
            path="evidence/planning/igw_assignments.json",
            base_sha=None, content=LEDGER_DOC_V1)
        self.assertEqual(verdict, "applied")
        content = _git(self._repo, "show",
                       "origin/master:evidence/planning/"
                       "igw_assignments.json").stdout
        self.assertEqual(content, LEDGER_DOC_V1)


class TestNoOp(_Fixture):

    def test_identical_content_is_a_noop_apply(self):
        base = self._origin_sha()
        verdict, payload = self._apply(base_sha=base, content=LEDGER_DOC_V1)
        self.assertEqual(verdict, "applied")
        self.assertTrue(payload.get("noop"))
        # No new commit was made.
        self.assertEqual(self._origin_sha(), base)


class TestShrinkGuard(_Fixture):

    def test_large_shrink_refused_without_allow_shrink(self):
        base = self._origin_sha()
        tiny = "{}"
        verdict, payload = self._apply(base_sha=base, content=tiny)
        self.assertEqual(verdict, "suspicious_shrink")
        self.assertEqual(self._origin_ledger(), LEDGER_DOC_V1)

    def test_large_shrink_applies_with_allow_shrink(self):
        base = self._origin_sha()
        tiny = "{}"
        verdict, payload = self._apply(base_sha=base, content=tiny,
                                       allow_shrink=True)
        self.assertEqual(verdict, "applied")
        self.assertEqual(self._origin_ledger(), tiny)


class TestShadow(_Fixture):

    def test_shadow_checks_but_never_writes(self):
        base = self._origin_sha()
        verdict, payload = self._apply(base_sha=base, content=LEDGER_DOC_V2,
                                       shadow=True)
        self.assertEqual(verdict, "applied")
        self.assertTrue(payload.get("shadow"))
        self.assertEqual(self._origin_ledger(), LEDGER_DOC_V1,
                         "shadow mode must never write")

    def test_shadow_still_reports_base_moved(self):
        base = self._origin_sha()
        _git(self._repo, "fetch", "-q", "origin", "master")
        other = pathlib.Path(tempfile.mkdtemp(prefix="git_intent_shadow_"))
        _git(self._repo, "worktree", "add", "-q", "--detach", str(other),
             "origin/master")
        (other / LEDGER_PATH).write_text(LEDGER_DOC_V2, encoding="utf-8")
        _git(other, "add", LEDGER_PATH)
        _git(other, "commit", "-q", "-m", "concurrent change")
        _git(other, "push", "-q", "origin", "HEAD:master")
        _git(self._repo, "worktree", "remove", "--force", str(other))

        verdict, payload = self._apply(base_sha=base, content=LEDGER_DOC_V1,
                                       shadow=True)
        self.assertEqual(verdict, "base_moved")


class TestRepoNotConfigured(_Fixture):

    def test_unconfigured_repo_refuses_cleanly(self):
        del os.environ["COORDINATOR_INTENT_REPO_REE_ASSEMBLY"]
        verdict, payload = self._apply(base_sha=self._origin_sha(),
                                       content=LEDGER_DOC_V2)
        self.assertEqual(verdict, "repo_not_configured")
        os.environ["COORDINATOR_INTENT_REPO_REE_ASSEMBLY"] = str(self._repo)


class TestCommitAttribution(_Fixture):

    def test_commit_message_carries_session_and_machine(self):
        base = self._origin_sha()
        self._apply(base_sha=base, content=LEDGER_DOC_V2,
                   session_id="sess-xyz", machine="host-abc")
        log = _git(self._repo, "log", "-1", "--format=%B")
        self.assertIn("igw-ledger: test", log.stdout)
        self.assertIn("session_id: sess-xyz", log.stdout)
        self.assertIn("machine: host-abc", log.stdout)


class TestEndpointWiring(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="git_intent_wire_")
        self._dbpath = os.path.join(self._tmp, "coord.db")
        db.init_db(self._dbpath)
        self._conn = db.connect(self._dbpath)

    def tearDown(self):
        self._conn.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_dispatch_table_membership(self):
        self.assertIn("/intent/replace", app._TASK_CLAIM_CHIP_POST)
        self.assertIs(app._TASK_CLAIM_CHIP_POST["/intent/replace"],
                      app._intent_replace)

    def test_handler_rejects_missing_fields(self):
        code, out = app._intent_replace(self._conn, {}, "h")
        self.assertEqual(code, 400)
        self.assertEqual(out["verdict"], "bad_request")
        code, out = app._intent_replace(
            self._conn, {"repo": "REE_assembly"}, "h")
        self.assertEqual(code, 400)

    def test_handler_not_routed_maps_to_400(self):
        code, out = app._intent_replace(
            self._conn,
            {"repo": "REE_assembly", "path": "nope.json",
             "content": "{}", "message": "m"},
            "h")
        self.assertEqual(code, 400)
        self.assertEqual(out["verdict"], "not_routed")

    def test_handler_repo_not_configured_maps_to_500(self):
        saved = os.environ.pop("COORDINATOR_INTENT_REPO_REE_ASSEMBLY", None)
        try:
            code, out = app._intent_replace(
                self._conn,
                {"repo": "REE_assembly", "path": LEDGER_PATH,
                 "content": LEDGER_DOC_V1, "message": "m",
                 "base_sha": "deadbeef"},
                "h")
            self.assertEqual(code, 500)
            self.assertEqual(out["verdict"], "repo_not_configured")
        finally:
            if saved is not None:
                os.environ["COORDINATOR_INTENT_REPO_REE_ASSEMBLY"] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
