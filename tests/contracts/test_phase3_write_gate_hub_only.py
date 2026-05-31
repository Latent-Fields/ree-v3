"""Contract: PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE is HUB-ONLY.

Background
----------
The Phase 3 `PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1` env flag suppresses
the per-tick local writes of `runner_heartbeats/<host>.json` AND the
per-machine `runner_commands/<host>.json`. The flag is intended ONLY for
the hub VM (`ree-cloud-1`, hostname `ree-worker-1`) where the runner
co-tenants the same REE_assembly checkout the writer publishes from.

On any OTHER worker, setting the flag is a fleet-killing footgun: the
file-channel command writeback (`write_commands_file`) is the only place
the runner can flip a stop command's status to `done`. Without that
persistence, every systemd restart picks the same stop command up again
-> graceful drain -> exit -> systemd restart -> ... until
`start-limit-hit` kills the unit and the worker disappears from the fleet.

Incident 2026-05-30: cloud-2, cloud-3, and cloud-4 all wedged in
`failed/start-limit-hit` for 6-12 hours after a fleet-wide
`shadow.conf` template push mis-included the flag on non-hub VMs.

This contracts test enforces two complementary protections:

Layer 1 (template scan)
  C1: `ree-v3/ree-runner.service` (the in-tree template) must NOT
      mention `PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE` -- the flag is
      configured per-VM in `/etc/systemd/system/ree-runner.service.d/
      shadow.conf` which is intentionally not in the tree.
  C2: `ree-v3/coordinator/deploy/*.service` likewise must not mention it.

Layer 2 (runtime self-guard)
  C3: With `PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1` AND a non-hub
      hostname, `_phase3_heartbeat_write_gated()` returns False (refuses
      the gate) so the worker does not lock itself into the restart loop.
  C4: With the same flag and a HUB hostname (`ree-cloud-1` or
      `ree-worker-1`), the gate correctly returns True.
  C5: The non-hub refusal is one-shot logged (re-arms across fresh
      module state but does not spam the journal every tick).

Layer-2 is the load-bearing fix in production; Layer-1 prevents the bad
config from re-appearing in tree.
"""

import io
import os
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import runner_remote_control as rrc  # noqa: E402


PHASE3_KEY = "PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE"


def _reset_log_flags() -> None:
    rrc._HEARTBEAT_GATE_LOGGED[0] = False
    rrc._HEARTBEAT_WRITE_GATE_LOGGED[0] = False
    rrc._HEARTBEAT_WRITE_GATE_NON_HUB_REFUSED_LOGGED[0] = False


def _read_text_or_skip(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text()


def _strip_comments(text: str) -> str:
    """Drop systemd-unit comment lines (starting with '#' optionally
    after whitespace). Allows the in-tree template to mention the flag
    inside an explicit `# HUB-ONLY:` documentation block without
    failing the scan -- the scan is for ACTIVE configuration, not
    inline docs."""
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


class TestPhase3WriteGateHubOnlyTemplateScan(unittest.TestCase):
    # ------ Layer 1: template scan -------------------------------------

    def test_c1_runner_service_template_does_not_mention_write_gate(self):
        path = REPO_ROOT / "ree-runner.service"
        text = _read_text_or_skip(path)
        if text is None:
            self.skipTest(f"{path} not present in tree")
        active = _strip_comments(text)
        self.assertNotIn(
            PHASE3_KEY, active,
            f"In-tree systemd template {path} sets {PHASE3_KEY} in an "
            f"active (non-comment) line. This flag is HUB-ONLY and is "
            f"configured per-VM in shadow.conf, NOT in the shared "
            f"template. See 2026-05-30 cloud-2/3/4 fleet wedge incident.",
        )

    def test_c2_coordinator_deploy_templates_do_not_mention_write_gate(self):
        deploy_dir = REPO_ROOT / "coordinator" / "deploy"
        if not deploy_dir.exists():
            self.skipTest(f"{deploy_dir} not present in tree")
        offenders = []
        # Scan .service and .conf templates (sync_daemon / coordinator)
        for pattern in ("*.service", "*.conf"):
            for path in deploy_dir.glob(pattern):
                text = path.read_text()
                active = _strip_comments(text)
                if PHASE3_KEY in active:
                    offenders.append(str(path))
        self.assertEqual(
            offenders, [],
            f"Deploy template(s) {offenders} set {PHASE3_KEY} in an "
            f"active (non-comment) line. This flag is HUB-ONLY; deploy "
            f"templates ship to ALL workers and must not carry it.",
        )


class TestPhase3WriteGateHubOnlyRuntimeGuard(unittest.TestCase):
    # ------ Layer 2: runtime self-guard --------------------------------

    def setUp(self):
        self._orig_env = os.environ.get(PHASE3_KEY)
        os.environ.pop(PHASE3_KEY, None)
        self._orig_gethostname = rrc.socket.gethostname
        _reset_log_flags()

    def tearDown(self):
        if self._orig_env is None:
            os.environ.pop(PHASE3_KEY, None)
        else:
            os.environ[PHASE3_KEY] = self._orig_env
        rrc.socket.gethostname = self._orig_gethostname
        _reset_log_flags()

    def test_c3_flag_set_on_non_hub_hostname_refuses_gate(self):
        os.environ[PHASE3_KEY] = "1"
        for non_hub in (
            "DLAPTOP-4.local",
            "Daniel-PC",
            "ree-cloud-2",
            "ree-cloud-3",
            "ree-cloud-4",
            "EWIN-PC",
            "",
        ):
            with self.subTest(hostname=non_hub):
                _reset_log_flags()
                rrc.socket.gethostname = lambda h=non_hub: h
                self.assertFalse(
                    rrc._phase3_heartbeat_write_gated(),
                    f"Gate fired on non-hub hostname {non_hub!r} -- "
                    f"the self-guard MUST refuse here to avoid the "
                    f"2026-05-30 cloud-2/3/4 restart-loop incident.",
                )

    def test_c4_flag_set_on_hub_hostname_admits_gate(self):
        os.environ[PHASE3_KEY] = "1"
        for hub in ("ree-cloud-1", "ree-worker-1"):
            with self.subTest(hostname=hub):
                _reset_log_flags()
                rrc.socket.gethostname = lambda h=hub: h
                self.assertTrue(
                    rrc._phase3_heartbeat_write_gated(),
                    f"Gate refused on hub hostname {hub!r} -- the "
                    f"self-guard MUST admit here (hub co-tenancy is "
                    f"the exact scenario the flag was designed for).",
                )

    def test_c5_non_hub_refusal_logs_loud_warning_once(self):
        os.environ[PHASE3_KEY] = "1"
        rrc.socket.gethostname = lambda: "ree-cloud-2"
        buf = io.StringIO()
        with redirect_stdout(buf):
            # First call: prints the warning.
            self.assertFalse(rrc._phase3_heartbeat_write_gated())
            # Subsequent calls: do not spam.
            for _ in range(5):
                self.assertFalse(rrc._phase3_heartbeat_write_gated())
        out = buf.getvalue()
        self.assertIn("WARNING", out)
        self.assertIn("PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE", out)
        self.assertIn("ree-cloud-2", out)
        # Exactly one WARNING line emitted across the 6 calls.
        warning_lines = [
            line for line in out.splitlines() if "WARNING" in line
        ]
        self.assertEqual(
            len(warning_lines), 1,
            f"Expected exactly 1 WARNING log line, got "
            f"{len(warning_lines)}: {warning_lines!r}",
        )


if __name__ == "__main__":
    unittest.main()
