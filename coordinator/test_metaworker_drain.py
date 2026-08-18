"""Contracts for ree-metaworker-drain.sh -- the shutdown drain for chip workers.

The scaler's shutdown is graceful for EXPERIMENTS (the runner has a SIGTERM
handler) but nothing drained chip workers: a dispatched `claude -p` process has
no signal handling, so a box shutdown killed it mid-edit and left its chip
claimed until CLAIM_STALE_HOURS (3h), after which the work is redone from
scratch. Confirmed live 2026-08-18 on ree-worker-4.

Time-independent in the sense that matters: no test depends on wall-clock date,
and every wait is bounded by an injected REE_DRAIN_MAX_SEC of a few seconds.

The load-bearing assertions are the two failure paths, not the happy one. A
drain that blocks a shutdown forever, or that reports success while workers are
still alive, is worse than no drain at all.
"""
import os
import signal
import subprocess
import tempfile
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
DRAIN = os.path.join(_HERE, "deploy", "ree-metaworker-drain.sh")


class DrainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.wt = os.path.join(self.tmp, ".claude", "worktrees")
        os.makedirs(self.wt)
        self.procs = []

    def tearDown(self):
        for pid in self.procs:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass

    def _worker(self, chip, seconds):
        d = os.path.join(self.wt, "metaworker-%s" % chip)
        os.makedirs(d, exist_ok=True)
        # Double-fork via sh so the sleep is reparented to init and reaped when
        # it exits. A plain Popen child becomes a ZOMBIE until Python reaps it,
        # and a zombie still passes `kill -0` -- which is exactly the bug this
        # suite found in the drain script, so the fixture must not reintroduce it.
        # The redirect on the BACKGROUNDED sleep is load-bearing, not tidiness:
        # it inherits sh's stdout, which is the capture pipe, so without it
        # subprocess.run blocks until the sleep exits (waiting for EOF) and the
        # "worker" is already gone before the drain is ever invoked. That made
        # every timeout test silently assert against an empty worktree dir.
        launcher = subprocess.run(
            ["/bin/sh", "-c", "sleep %d >/dev/null 2>&1 & echo $!" % seconds],
            capture_output=True, text=True)
        pid = int(launcher.stdout.strip())
        self.procs.append(pid)
        with open(os.path.join(d, ".dispatch_pid"), "w") as fh:
            fh.write(str(pid))
        return pid

    def _run(self, max_sec=5):
        env = dict(os.environ)
        env.update(REE_REPO=self.tmp,
                   REE_DRAIN_LOG=os.path.join(self.tmp, "log"),
                   REE_DRAIN_POLL_SEC="1",
                   REE_DRAIN_MAX_SEC=str(max_sec))
        r = subprocess.run(["bash", DRAIN], env=env, capture_output=True,
                           text=True, timeout=max_sec + 60)
        return r.returncode, r.stdout + r.stderr

    def test_no_workers_exits_immediately(self):
        rc, out = self._run()
        self.assertEqual(rc, 0)
        self.assertIn("nothing to drain", out)

    def test_waits_for_a_live_worker_then_reports_clean(self):
        self._worker("alpha", 3)
        rc, out = self._run(max_sec=40)
        self.assertEqual(rc, 0)
        self.assertIn("drained cleanly", out)

    def test_dead_pid_is_not_counted_as_in_flight(self):
        pid = self._worker("alpha", 30)
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        rc, out = self._run()
        self.assertIn("nothing to drain", out)

    # ---------- the two failure paths that actually matter ----------
    def test_timeout_never_blocks_the_shutdown(self):
        # Exit 0 even with survivors. A non-zero exit here would make systemd
        # treat the shutdown transaction as failed.
        self._worker("alpha", 120)
        rc, out = self._run(max_sec=3)
        self.assertEqual(rc, 0)
        self.assertIn("TIMEOUT", out)

    def test_timeout_names_the_survivors_and_the_consequence(self):
        pid = self._worker("alpha", 120)
        _, out = self._run(max_sec=3)
        self.assertIn("metaworker-alpha", out)
        self.assertIn(str(pid), out)
        # The operator has to know the chip will be redone, not silently lost.
        self.assertIn("CLAIM_STALE_HOURS", out)

    def test_never_kills_a_worker(self):
        # NEGATIVE CONTROL, and the most important one: this script must only
        # ever WAIT. If it started killing workers it would be causing exactly
        # the data loss it was written to prevent.
        pid = self._worker("alpha", 120)
        self._run(max_sec=3)
        time.sleep(0.5)
        os.kill(pid, 0)   # raises if the drain killed it

    def test_bound_is_respected_rather_than_waiting_forever(self):
        self._worker("alpha", 120)
        started = time.monotonic()
        self._run(max_sec=3)
        self.assertLess(time.monotonic() - started, 45,
                        "drain did not honour REE_DRAIN_MAX_SEC")


if __name__ == "__main__":
    unittest.main(verbosity=2)
