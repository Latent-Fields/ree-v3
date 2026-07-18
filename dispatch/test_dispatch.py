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


if __name__ == "__main__":
    unittest.main(verbosity=2)
