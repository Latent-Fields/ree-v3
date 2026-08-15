#!/usr/bin/env python3
"""Integration + unit tests for the REE phone-dispatch service + executor.

Runs the service as a subprocess on an ephemeral port with a temp DB + token,
drives the HTTP API, and runs the executor (ONESHOT) against a temp git repo
with a FAKE `claude` on PATH -- a true end-to-end smoke with no real Claude call.

Run:  python3 test_dispatch.py
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.join(HERE, "dispatch_service.py")
EXECUTOR = os.path.join(HERE, "dispatch_executor.py")
TOKEN = "test-token-abc123"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def http(method, url, body=None, token=TOKEN):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


class DispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dispatch-test-")
        cls.port = free_port()
        cls.base = "http://127.0.0.1:%d" % cls.port
        env = dict(os.environ)
        env.update({
            "DISPATCH_BIND_HOST": "127.0.0.1",
            "DISPATCH_BIND_PORT": str(cls.port),
            "DISPATCH_DB": os.path.join(cls.tmp, "dispatch.db"),
            "DISPATCH_TOKEN": TOKEN,
            "DISPATCH_NTFY_TOPIC": "",  # notifications off in tests
        })
        cls.proc = subprocess.Popen([sys.executable, SERVICE], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # wait for /health
        for _ in range(50):
            try:
                code, _ = http("GET", cls.base + "/health", token=None)
                if code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("service did not start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()

    def test_01_health_no_auth(self):
        code, data = http("GET", self.base + "/health", token=None)
        self.assertEqual(code, 200)
        self.assertTrue(data["ok"])

    def test_02_page_no_auth(self):
        req = urllib.request.Request(self.base + "/")
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode()
        self.assertEqual(r.status, 200)
        self.assertIn("REE Dispatch", html)

    def test_03_auth_required(self):
        code, data = http("GET", self.base + "/api/jobs", token=None)
        self.assertEqual(code, 401)
        code, _ = http("GET", self.base + "/api/jobs", token="wrong")
        self.assertEqual(code, 401)

    def test_04_lifecycle(self):
        # enqueue staged
        code, d = http("POST", self.base + "/api/enqueue",
                       {"title": "t1", "prompt": "do a thing", "status": "staged"})
        self.assertEqual(code, 200)
        jid = d["id"]
        self.assertEqual(d["status"], "staged")
        # not yet pending
        code, d = http("GET", self.base + "/api/pending")
        self.assertNotIn(jid, [j["id"] for j in d["jobs"]])
        # launch -> pending
        code, d = http("POST", self.base + "/api/launch", {"id": jid})
        self.assertEqual(code, 200)
        self.assertEqual(d["job"]["status"], "pending")
        # appears in pending
        code, d = http("GET", self.base + "/api/pending")
        self.assertIn(jid, [j["id"] for j in d["jobs"]])
        # claim
        code, d = http("POST", self.base + "/api/claim",
                       {"id": jid, "machine": "macbook"})
        self.assertEqual(code, 200)
        self.assertEqual(d["job"]["status"], "claimed")
        self.assertEqual(d["job"]["claimed_by"], "macbook")
        # second claim -> 409
        code, d = http("POST", self.base + "/api/claim", {"id": jid})
        self.assertEqual(code, 409)
        # running then done
        code, d = http("POST", self.base + "/api/update",
                       {"id": jid, "status": "running"})
        self.assertEqual(code, 200)
        code, d = http("POST", self.base + "/api/update",
                       {"id": jid, "status": "done", "exit_code": 0,
                        "summary": "ok"})
        self.assertEqual(code, 200)
        self.assertEqual(d["job"]["status"], "done")

    def test_05_guards(self):
        code, d = http("POST", self.base + "/api/enqueue",
                       {"prompt": "p", "status": "pending"})
        jid = d["id"]
        # launch a non-staged -> 409
        code, _ = http("POST", self.base + "/api/launch", {"id": jid})
        self.assertEqual(code, 409)
        # claim + running, then cancel -> 409 (not cancellable while running)
        http("POST", self.base + "/api/claim", {"id": jid})
        http("POST", self.base + "/api/update", {"id": jid, "status": "running"})
        code, _ = http("POST", self.base + "/api/cancel", {"id": jid})
        self.assertEqual(code, 409)
        # update with bad status
        code, _ = http("POST", self.base + "/api/update",
                       {"id": jid, "status": "bogus"})
        self.assertEqual(code, 400)

    def test_06_enqueue_requires_prompt(self):
        code, _ = http("POST", self.base + "/api/enqueue", {"title": "x"})
        self.assertEqual(code, 400)

    def test_07_executor_end_to_end(self):
        # temp git repo
        repo = os.path.join(self.tmp, "repo")
        os.makedirs(repo)
        subprocess.run(["git", "init", "-q", repo], check=True)
        subprocess.run(["git", "-C", repo, "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", repo, "config", "user.name", "t"], check=True)
        with open(os.path.join(repo, "seed.txt"), "w") as f:
            f.write("seed\n")
        subprocess.run(["git", "-C", repo, "add", "-A"], check=True)
        subprocess.run(["git", "-C", repo, "commit", "-qm", "init"], check=True)

        # fake claude on PATH: writes a JSON result + a file in cwd (worktree)
        bindir = os.path.join(self.tmp, "bin")
        os.makedirs(bindir)
        fake = os.path.join(bindir, "claude")
        with open(fake, "w") as f:
            f.write("#!/usr/bin/env bash\n"
                    'echo "ran" > dispatched_output.txt\n'
                    'echo \'{"result":"did the thing","is_error":false}\'\n')
        os.chmod(fake, 0o755)

        # enqueue a pending job pointed at the repo
        code, d = http("POST", self.base + "/api/enqueue",
                       {"title": "e2e", "prompt": "make a file",
                        "cwd": repo, "status": "pending"})
        jid = d["id"]

        env = dict(os.environ)
        env.update({
            "DISPATCH_URL": self.base,
            "DISPATCH_TOKEN": TOKEN,
            "DISPATCH_MACHINE": "macbook",
            "DISPATCH_ONESHOT": "1",
            "DISPATCH_DEFAULT_CWD": repo,
            "DISPATCH_WORKTREE_BASE": os.path.join(self.tmp, "wt"),
            "DISPATCH_LOG_DIR": os.path.join(self.tmp, "logs"),
            "PATH": bindir + os.pathsep + env.get("PATH", ""),
        })
        res = subprocess.run([sys.executable, EXECUTOR], env=env,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(res.returncode, 0, res.stderr)

        # job should be done
        code, d = http("GET", self.base + "/api/jobs")
        job = next(j for j in d["jobs"] if j["id"] == jid)
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["exit_code"], 0)
        self.assertIn("did the thing", job["summary"])
        # worktree branch created + file written there
        wt = os.path.join(self.tmp, "wt", "dispatch-" + jid)
        self.assertTrue(os.path.exists(os.path.join(wt, "dispatched_output.txt")))
        branches = subprocess.run(
            ["git", "-C", repo, "branch", "--list", "dispatch/" + jid],
            capture_output=True, text=True).stdout
        self.assertIn("dispatch/" + jid, branches)


    def test_08_mirror_hook(self):
        # Simulate a PostToolUse spawn_task hook firing: pipe the hook JSON to
        # the mirror script with DISPATCH_URL/TOKEN env -> a staged chip job.
        mirror = os.path.join(HERE, "hooks", "mirror_chip_to_dispatch.py")
        hook_json = json.dumps({
            "tool_name": "mcp__ccd_session__spawn_task",
            "tool_input": {"title": "chip via hook", "prompt": "do the chip work",
                           "tldr": "x", "cwd": "/some/repo"},
            "tool_response": {"task_id": "task_x"},
        })
        env = dict(os.environ)
        env.update({"DISPATCH_URL": self.base, "DISPATCH_TOKEN": TOKEN})
        res = subprocess.run([sys.executable, mirror], input=hook_json, env=env,
                             capture_output=True, text=True, timeout=15)
        self.assertEqual(res.returncode, 0)  # fail-open: always 0
        code, d = http("GET", self.base + "/api/jobs")
        chip = next((j for j in d["jobs"]
                     if j["title"] == "chip via hook"), None)
        self.assertIsNotNone(chip)
        self.assertEqual(chip["status"], "staged")
        self.assertEqual(chip["source"], "chip")
        self.assertEqual(chip["cwd"], "/some/repo")

    def test_09_mirror_hook_failopen_when_unconfigured(self):
        # No DISPATCH_URL/TOKEN, no client config -> silent no-op, exit 0.
        mirror = os.path.join(HERE, "hooks", "mirror_chip_to_dispatch.py")
        env = {k: v for k, v in os.environ.items()
               if k not in ("DISPATCH_URL", "DISPATCH_TOKEN")}
        # MUST point CLIENT_CONFIG at a nonexistent path: it resolves relative
        # to the hook FILE, so clearing the env vars alone leaves the real
        # on-disk config in play and this test posts to the LIVE queue.
        env["DISPATCH_CLIENT_CONFIG"] = os.path.join(self.tmp, "no-such.json")
        res = subprocess.run([sys.executable, mirror],
                             input='{"tool_input":{"prompt":"x"}}',
                             env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(res.returncode, 0)


class SummarizeUnitTest(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("DISPATCH_URL", "http://x")
        os.environ.setdefault("DISPATCH_TOKEN", "x")
        sys.path.insert(0, HERE)
        import dispatch_executor  # noqa: E402
        self.ex = dispatch_executor

    def test_json_result(self):
        s = self.ex._summarize('{"result":"hello world","is_error":false}', "", 0)
        self.assertEqual(s, "hello world")

    def test_json_error(self):
        s = self.ex._summarize('{"result":"boom","is_error":true}', "", 1)
        self.assertTrue(s.startswith("error: boom"))

    def test_plain_text(self):
        s = self.ex._summarize("not json output", "", 0)
        self.assertEqual(s, "not json output")

    def test_stderr_on_failure(self):
        s = self.ex._summarize("", "traceback boom", 1)
        self.assertTrue(s.startswith("error: traceback boom"))


class RoutingUnitTest(unittest.TestCase):
    """Chip misrouting guards (audited 2026-07-18).

    Before this, a job with cwd='' silently fell back to REE_assembly and got a
    worktree of the WRONG repo -- e.g. a chip whose prompt only touches ree-v3/
    paths that do not exist there. Routing now refuses rather than guesses.
    """

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("DISPATCH_URL", "http://x")
        os.environ.setdefault("DISPATCH_TOKEN", "x")
        sys.path.insert(0, HERE)
        import dispatch_executor  # noqa: E402
        cls.ex = dispatch_executor
        cls.tmp = tempfile.mkdtemp(prefix="dispatch-routing-")
        # Two sibling repos under a shared root, mirroring REE_Working.
        cls.repo_a = os.path.join(cls.tmp, "repo_a")
        cls.repo_b = os.path.join(cls.tmp, "repo_b")
        for repo in (cls.repo_a, cls.repo_b):
            os.makedirs(os.path.join(repo, "sub"))
            subprocess.run(["git", "init", "-q", repo], check=True)
        cls.ex.PATH_ROOT = cls.tmp
        cls.ex.PATH_RE = None          # recompile against the temp root
        cls.ex.DEFAULT_CWD = ""
        cls.ex.STRICT_REPO_MATCH = True

    def test_empty_cwd_refused_not_guessed(self):
        repo, err = self.ex.resolve_repo({"cwd": "", "prompt": "do a thing"})
        self.assertIsNone(repo)
        self.assertIn("no cwd", err)

    def test_explicit_default_cwd_still_honoured(self):
        self.ex.DEFAULT_CWD = self.repo_a
        try:
            repo, err = self.ex.resolve_repo({"cwd": "", "prompt": "do a thing"})
            self.assertIsNone(err)
            self.assertEqual(os.path.realpath(repo),
                             os.path.realpath(self.repo_a))
        finally:
            self.ex.DEFAULT_CWD = ""

    def test_good_cwd_resolves(self):
        job = {"cwd": os.path.join(self.repo_a, "sub"),
               "prompt": "edit %s/sub/x.py" % self.repo_a}
        repo, err = self.ex.resolve_repo(job)
        self.assertIsNone(err)
        self.assertEqual(os.path.realpath(repo), os.path.realpath(self.repo_a))

    def test_prompt_targets_other_repo_refused(self):
        # The audited case: cwd says repo_a, every path in the prompt is repo_b.
        job = {"cwd": self.repo_a,
               "prompt": "update %s/sub/queue.json please" % self.repo_b}
        repo, err = self.ex.resolve_repo(job)
        self.assertIsNone(repo)
        self.assertIn("prompt targets", err)
        self.assertIn(os.path.realpath(self.repo_b), os.path.realpath(err))

    def test_mismatch_check_can_be_disabled(self):
        self.ex.STRICT_REPO_MATCH = False
        try:
            job = {"cwd": self.repo_a, "prompt": "%s/sub/x" % self.repo_b}
            repo, err = self.ex.resolve_repo(job)
            self.assertIsNone(err)
            self.assertIsNotNone(repo)
        finally:
            self.ex.STRICT_REPO_MATCH = True

    def test_cwd_inside_worktree_refused(self):
        wt = os.path.join(self.repo_a, ".claude", "worktrees", "slug")
        os.makedirs(wt, exist_ok=True)
        repo, err = self.ex.resolve_repo({"cwd": wt, "prompt": "x"})
        self.assertIsNone(repo)
        self.assertIn("worktree", err)

    def test_missing_cwd_dir_refused(self):
        repo, err = self.ex.resolve_repo(
            {"cwd": os.path.join(self.tmp, "nope"), "prompt": "x"})
        self.assertIsNone(repo)
        self.assertIn("does not exist", err)

    def test_nonexistent_prompt_path_resolves_via_ancestor(self):
        # A prompt may name a file to CREATE; resolve the deepest real ancestor.
        job = {"cwd": self.repo_a,
               "prompt": "create %s/sub/brand_new.py" % self.repo_a}
        repo, err = self.ex.resolve_repo(job)
        self.assertIsNone(err)
        self.assertEqual(os.path.realpath(repo), os.path.realpath(self.repo_a))

    def test_container_cwd_with_relative_sibling_path_refused(self):
        # Prompts written with repo-RELATIVE paths ("repo_b/x.md") are invisible
        # to the absolute-path check; running them in a worktree of the
        # container lands the work nowhere.
        subprocess.run(["git", "init", "-q", self.tmp], check=True)
        self.ex.SIBLING_REPOS = None
        try:
            job = {"cwd": self.tmp, "prompt": "update repo_b/notes.md please"}
            repo, err = self.ex.resolve_repo(job)
            self.assertIsNone(repo)
            self.assertIn("container", err)
            self.assertIn("repo_b", err)
        finally:
            subprocess.run(["rm", "-rf", os.path.join(self.tmp, ".git")],
                           check=True)
            self.ex.SIBLING_REPOS = None

    def test_container_cwd_without_sibling_reference_allowed(self):
        # Work genuinely ON the container repo itself must still run.
        subprocess.run(["git", "init", "-q", self.tmp], check=True)
        self.ex.SIBLING_REPOS = None
        try:
            job = {"cwd": self.tmp, "prompt": "update the top-level README"}
            repo, err = self.ex.resolve_repo(job)
            self.assertIsNone(err)
            self.assertEqual(os.path.realpath(repo), os.path.realpath(self.tmp))
        finally:
            subprocess.run(["rm", "-rf", os.path.join(self.tmp, ".git")],
                           check=True)
            self.ex.SIBLING_REPOS = None

    def test_prompt_with_no_paths_is_inconclusive_not_refused(self):
        repo, err = self.ex.resolve_repo(
            {"cwd": self.repo_a, "prompt": "no absolute paths here at all"})
        self.assertIsNone(err)
        self.assertIsNotNone(repo)


class WorktreeRemovalTest(unittest.TestCase):
    """`DISPATCH_KEEP_WORKTREE=0` must not be able to destroy agent output.

    remove_worktree used to be a bare `git worktree remove --force` whose return
    code was discarded. --force bypasses git's own dirty check, which is the
    last line of defence against an untracked DURABLE ARTIFACT a headless agent
    staged in the worktree -- content with no commit, no stash, no reflog and no
    recovery once the directory is gone (the confirmed loss in
    chip-20260807-thoughtdigestion-trial-5). Same defect class as the fixes in
    hygiene_routine_tick.py (817f2524) and igw_routine_tick.py (39f62c88).

    Two directions matter here and both are asserted, because a guard that only
    ever refuses is as broken as one that never does: the artifact cases must be
    KEPT, and the ordinary success case must still be COLLECTED.
    """

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("DISPATCH_URL", "http://x")
        os.environ.setdefault("DISPATCH_TOKEN", "x")
        sys.path.insert(0, HERE)
        import dispatch_executor  # noqa: E402
        cls.ex = dispatch_executor
        cls.tmp = tempfile.mkdtemp(prefix="dispatch-wt-")
        cls.repo = os.path.join(cls.tmp, "repo")
        os.makedirs(cls.repo)
        subprocess.run(["git", "init", "-q", cls.repo], check=True)
        for k, v in (("user.email", "t@t"), ("user.name", "t")):
            subprocess.run(["git", "-C", cls.repo, "config", k, v], check=True)
        with open(os.path.join(cls.repo, "seed.txt"), "w") as f:
            f.write("seed\n")
        with open(os.path.join(cls.repo, ".gitignore"), "w") as f:
            f.write("*.scratch\n")
        subprocess.run(["git", "-C", cls.repo, "add", "-A"], check=True)
        subprocess.run(["git", "-C", cls.repo, "commit", "-qm", "init"],
                       check=True)

    def _worktree(self, job_id):
        """A real dispatch worktree, made the way the executor makes one."""
        self.ex.WORKTREE_BASE = os.path.join(self.tmp, "wt")
        wt, branch, err = self.ex.make_worktree(self.repo, job_id)
        self.assertIsNone(err, err)
        return wt, branch

    def _branch_exists(self, branch):
        out = subprocess.run(["git", "-C", self.repo, "branch", "--list", branch],
                             capture_output=True, text=True).stdout
        return branch in out

    # -- the negative control: removal must still actually happen -------------

    def test_clean_worktree_is_removed_and_branch_survives(self):
        """The documented success path: the agent COMMITS, so the tree is clean.

        This is the control that matters most -- the failure mode of the fix is
        removal silently never happening again. The branch must outlive the
        worktree, because the branch is what the user reviews.
        """
        wt, branch = self._worktree("clean1")
        with open(os.path.join(wt, "agent_work.py"), "w") as f:
            f.write("print('done')\n")
        subprocess.run(["git", "-C", wt, "add", "-A"], check=True)
        subprocess.run(["git", "-C", wt, "commit", "-qm", "agent work"],
                       check=True)
        removed, detail = self.ex.remove_worktree(self.repo, wt)
        self.assertTrue(removed, detail)
        self.assertFalse(os.path.exists(wt))
        self.assertTrue(self._branch_exists(branch))
        # ...and the committed work is still reachable through that branch.
        show = subprocess.run(["git", "-C", self.repo, "show",
                               "%s:agent_work.py" % branch],
                              capture_output=True, text=True)
        self.assertEqual(show.returncode, 0, show.stderr)
        self.assertIn("done", show.stdout)

    def test_untouched_worktree_is_removed(self):
        """An agent that wrote nothing leaves nothing to protect."""
        wt, _ = self._worktree("clean2")
        removed, detail = self.ex.remove_worktree(self.repo, wt)
        self.assertTrue(removed, detail)
        self.assertFalse(os.path.exists(wt))

    def test_ignored_only_worktree_is_removed(self):
        """A .gitignore'd file must not pin the worktree forever.

        git's own check does not count ignored files, so the plain remove
        collects here. This is what stops the guard degrading into "nothing is
        ever collectable" for an operator with genuine scratch to declare.
        """
        wt, _ = self._worktree("ignored1")
        with open(os.path.join(wt, "junk.scratch"), "w") as f:
            f.write("throwaway\n")
        removed, detail = self.ex.remove_worktree(self.repo, wt)
        self.assertTrue(removed, detail)
        self.assertFalse(os.path.exists(wt))

    # -- the guard: unrecoverable content must survive ------------------------

    def test_untracked_artifact_is_kept_and_reported(self):
        """One extra untracked file => KEEP, and name it.

        This is the chip-20260807 shape: a staged durable artifact the agent
        never committed. --force would have deleted it with no trace and no
        error.
        """
        wt, _ = self._worktree("artifact1")
        artifact = os.path.join(wt, "design_review_staged.md")
        with open(artifact, "w") as f:
            f.write("# real work nobody else has a copy of\n")
        removed, detail = self.ex.remove_worktree(self.repo, wt)
        self.assertFalse(removed)
        self.assertTrue(os.path.exists(wt))
        self.assertTrue(os.path.exists(artifact))
        with open(artifact) as f:
            self.assertIn("real work", f.read())
        self.assertIn("design_review_staged.md", detail)

    def test_untracked_artifact_inside_a_new_directory_is_reported_by_file(self):
        """`-uall`: an untracked DIRECTORY must not hide its contents.

        Plain porcelain collapses this to `?? out/`, which tells an operator
        nothing about what is actually at risk.
        """
        wt, _ = self._worktree("artifact2")
        os.makedirs(os.path.join(wt, "out"))
        with open(os.path.join(wt, "out", "analysis.md"), "w") as f:
            f.write("findings\n")
        removed, detail = self.ex.remove_worktree(self.repo, wt)
        self.assertFalse(removed)
        self.assertIn("out/analysis.md", detail)

    def test_modified_tracked_file_is_kept(self):
        """Uncommitted edits to a tracked file are unrecoverable too."""
        wt, _ = self._worktree("dirty1")
        with open(os.path.join(wt, "seed.txt"), "w") as f:
            f.write("edited but never committed\n")
        removed, detail = self.ex.remove_worktree(self.repo, wt)
        self.assertFalse(removed)
        self.assertTrue(os.path.exists(wt))
        with open(os.path.join(wt, "seed.txt")) as f:
            self.assertIn("never committed", f.read())
        self.assertIn("seed.txt", detail)

    # -- the premise the empty-scratch-set claim rests on ---------------------

    def test_run_log_lands_outside_the_worktree(self):
        """This lane writes NOTHING untracked into the job's worktree.

        That is why remove_worktree needs no known-scratch exclusion, unlike the
        IGW and metaworker lanes (which write IGW_START_HERE.md / .dispatch_pid /
        DISPATCH_BRIEF.md / claude.log into theirs and must clear a bounded
        disposable set before a plain remove can collect anything). LOG_DIR is
        derived from __file__, so the log lands beside the executor. If that
        ever changes, the plain remove starts refusing on every job and this
        test is where it gets caught.
        """
        wt, _ = self._worktree("logsite1")
        log_path = os.path.join(self.tmp, "logs", "dispatch-logsite1.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fake_bin = os.path.join(self.tmp, "fake-claude")
        with open(fake_bin, "w") as f:
            f.write("#!/usr/bin/env bash\n"
                    'echo \'{"result":"ok","is_error":false}\'\n')
        os.chmod(fake_bin, 0o755)
        old_bin = self.ex.CLAUDE_BIN
        self.ex.CLAUDE_BIN = fake_bin
        try:
            code, _summary, _tail = self.ex.run_claude("do it", wt, log_path)
        finally:
            self.ex.CLAUDE_BIN = old_bin
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(log_path))
        # The log is NOT in the worktree, so the worktree is still pristine...
        status = subprocess.run(["git", "-C", wt, "status", "--porcelain", "-uall"],
                                capture_output=True, text=True)
        self.assertEqual(status.stdout.strip(), "",
                         "run_claude left untracked files in the worktree: %r"
                         % status.stdout)
        # ...and therefore still collectable by the un-forced remove.
        removed, detail = self.ex.remove_worktree(self.repo, wt)
        self.assertTrue(removed, detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
