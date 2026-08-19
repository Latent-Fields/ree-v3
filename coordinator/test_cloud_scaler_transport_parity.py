"""Parity between the TWO cloud-scaler transports.

There are two scalers running the same decision loop:

  * coordinator/deploy/cloud-scaler.py -- the AUTHORITATIVE one, a systemd
    timer on the hub firing every 5 minutes.
  * .github/workflows/cloud-scaler.yml -- a 6-hourly GitHub Actions BACKSTOP
    that only exists to catch a wedged hub.

The workflow header asserted for months that "logic in both transports is
identical 1:1". That assertion was untested, and it went false: the hub scaler
grew a pytest lease veto (2026-07-19) and an orchestrator veto (2026-08-18)
that the workflow never received.

The drift then powered off a live box. On 2026-08-19T01:39:26Z the GHA backstop
shut down ree-worker-4 with reason=heartbeat_stale while that box was serving as
the RESIDENT metaworker-dispatch dispatcher, with in_flight_dispatches=2 and
chips_open_work=80 in its published orchestrator heartbeat. The hub timer had
correctly vetoed the same box minutes earlier -- the two transports disagreed,
and the one with the stale logic won because it acts on the same hcloud API.
Nothing detected the outage for ~15 hours.

These tests exist so the NEXT veto added to cloud-scaler.py fails a test here
rather than silently killing a box six hours later.

METHOD -- the snippet is EXECUTED, not pattern-matched. The workflow's veto is
an embedded python program inside a YAML string inside a bash command
substitution. Grepping it for keywords would pass against a program that
crashes, or one that inverted its verdict. So each case below extracts that
program from the YAML, runs it for real against a real fixture tree, and
compares its verdict to read_orchestrator() from cloud-scaler.py on the same
fixture. Differential, not descriptive.

ONE DIVERGENCE IS DELIBERATE AND IS PINNED AS SUCH: the pytest lease veto reads
/home/ree/pytest_leases on the hub's own disk, which a GitHub-hosted runner
cannot see. It is structural, not drift, and test_lease_veto_absence_is_
deliberate documents it so a later session does not "fix" parity by inventing a
lease path that can never exist.

Time-independent: every fixture timestamp is generated relative to real now,
so these tests do not depend on the wall-clock date.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_REE_V3 = os.path.dirname(_HERE)
_WORKFLOW = os.path.join(_REE_V3, ".github", "workflows", "cloud-scaler.yml")
_SCALER_PY = os.path.join(_HERE, "deploy", "cloud-scaler.py")

_HB_REL = "ree-assembly/evidence/experiments/runner_heartbeats"
_AFFINITY = "ree-cloud-4"


def _load_scaler_module():
    """Import cloud-scaler.py despite the hyphen in its filename."""
    spec = importlib.util.spec_from_file_location("cloud_scaler_mod", _SCALER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCALER = _load_scaler_module()
WORKFLOW_SRC = open(_WORKFLOW, encoding="utf-8").read()


def extract_orchestrator_snippet():
    """Pull the embedded orchestrator-veto python out of the workflow YAML.

    Returns the dedented program text. Raises if the shape has changed --
    a rename or restructure should fail loudly here rather than silently
    stop testing anything.
    """
    m = re.search(
        r'ORCH_INFO=\$\((?:.|\n)*?python3 -c "\n((?:.|\n)*?)\n\s*"\)',
        WORKFLOW_SRC)
    if not m:
        raise AssertionError(
            "orchestrator veto snippet not found in %s -- renamed or "
            "restructured? This test cannot verify what it cannot find."
            % _WORKFLOW)
    raw = m.group(1)
    lines = [ln[10:] if ln.startswith(" " * 10) else ln
             for ln in raw.split("\n")]
    return "\n".join(lines)


ORCH_SNIPPET = extract_orchestrator_snippet()


def run_snippet(root, affinity=_AFFINITY, fresh_min=50):
    """Execute the workflow's embedded veto program against a fixture tree.

    Returns (veto: int, reason: str). Non-zero exit or unparseable output is
    a hard failure -- the workflow parses this with `sed -n 's/^orch_veto=
    \\([01]\\) .*/\\1/p'`, so a program that prints anything else yields an
    EMPTY ORCH_VETO and the veto silently never fires.
    """
    proc = subprocess.run(
        [sys.executable, "-c", ORCH_SNIPPET],
        cwd=root, capture_output=True, text=True,
        env={**os.environ, "AFFINITY": affinity,
             "ORCHESTRATOR_FRESH_MIN": str(fresh_min)})
    if proc.returncode != 0:
        raise AssertionError("snippet exited %d: %s"
                             % (proc.returncode, proc.stderr[:2000]))
    out = proc.stdout.strip()
    m = re.match(r"^orch_veto=([01]) orch_reason=(\S+)$", out)
    if not m:
        raise AssertionError(
            "snippet printed %r, which the workflow's sed cannot parse -- "
            "ORCH_VETO would be empty and the veto would never fire." % out)
    return int(m.group(1)), m.group(2)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class FixtureMixin:
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.hb_dir = os.path.join(self.root, _HB_REL)
        os.makedirs(self.hb_dir)
        self.addCleanup(shutil.rmtree, self.root, True)

    def write_hb(self, payload, affinity=_AFFINITY, raw=None):
        path = os.path.join(self.hb_dir, "%s-metaworker.json" % affinity)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(raw if raw is not None else json.dumps(payload))
        return path

    def hb(self, age_min=1, in_flight=0, open_work=0,
           role="orchestrator", state="dispatching"):
        return {
            "schema_version": "v1",
            "role": role,
            "machine": "%s-metaworker" % _AFFINITY,
            "last_tick_utc": iso(datetime.now(timezone.utc)
                                 - timedelta(minutes=age_min)),
            "state": state,
            "in_flight_dispatches": in_flight,
            "chips_open_work": open_work,
        }

    def assert_both(self, expected_veto, fresh_min=50):
        """Both transports must agree, and agree with `expected_veto`."""
        yaml_veto, yaml_reason = run_snippet(self.root, fresh_min=fresh_min)
        # Empty coord_status forces the GIT path, which is the transport the
        # GHA backstop has -- a GitHub-hosted runner cannot reach the
        # WireGuard-only coordinator. Parity is asserted on the judgement of
        # the same committed heartbeat, which is the thing that can drift.
        py_veto, py_reason = SCALER.read_orchestrator(
            self.hb_dir, _AFFINITY, fresh_min, coord_status={})[:2]
        self.assertEqual(
            bool(yaml_veto), bool(py_veto),
            "TRANSPORT DISAGREEMENT: workflow says veto=%s (%s), "
            "cloud-scaler.py says veto=%s (%s). This is the 2026-08-19 "
            "ree-worker-4 shape -- one transport keeps a box alive while "
            "the other powers it off."
            % (bool(yaml_veto), yaml_reason, bool(py_veto), py_reason))
        self.assertEqual(bool(yaml_veto), expected_veto,
                         "expected veto=%s, got %s (%s)"
                         % (expected_veto, bool(yaml_veto), yaml_reason))
        return yaml_reason, py_reason


class OrchestratorVetoParityTest(FixtureMixin, unittest.TestCase):
    """Both transports, same fixture, same verdict."""

    # ---- the veto FIRES -------------------------------------------------

    def test_live_dispatches_veto_in_both(self):
        self.write_hb(self.hb(in_flight=2, open_work=80))
        y, p = self.assert_both(True)
        self.assertIn("busy", y)
        self.assertIn("busy", p)

    def test_backlog_alone_vetoes_in_both(self):
        """chips_open_work > 0 with nothing in flight is the WEAK veto:
        nothing running this instant, but work this box would pick up."""
        self.write_hb(self.hb(in_flight=0, open_work=5))
        y, p = self.assert_both(True)
        self.assertIn("backlog", y)
        self.assertIn("backlog", p)

    def test_incident_replay_2026_08_19(self):
        """The exact heartbeat ree-worker-4 was publishing when the GHA
        backstop powered it off. Must now veto in BOTH transports."""
        self.write_hb({
            "schema_version": "v1", "role": "orchestrator",
            "machine": "ree-cloud-4-metaworker",
            "hostname": "ree-cloud-4-metaworker",
            "last_tick_utc": iso(datetime.now(timezone.utc)
                                 - timedelta(minutes=7)),
            "state": "dispatching", "cycles_completed": 65,
            "chips_dispatched_total": 35, "chips_open_work": 80,
            "chips_open_decision": 6, "in_flight_dispatches": 2,
            "coordination_plane_paused": False,
        })
        self.assert_both(True)

    def test_boundary_exactly_at_fresh_min_still_vetoes(self):
        self.write_hb(self.hb(age_min=49, in_flight=1))
        self.assert_both(True, fresh_min=50)

    # ---- NEGATIVE CONTROLS: the veto must NOT fire ----------------------
    # These are what stop the veto being widened into "cloud-4 is simply
    # always on", which would bill a 24/7 VM whether or not there is work.

    def test_idle_and_empty_does_not_veto(self):
        """THE load-bearing negative control. An orchestrator that is
        ticking but has nothing in flight AND an empty ledger must fall
        through to the ordinary shutdown test. This is what makes the box
        demand-sensitive rather than unconditionally always-on."""
        self.write_hb(self.hb(in_flight=0, open_work=0, state="idle"))
        y, p = self.assert_both(False)
        self.assertIn("idle_and_empty", y)
        self.assertIn("idle_and_empty", p)

    def test_stale_heartbeat_does_not_veto(self):
        """A dead metaworker must stop vetoing -- otherwise a crashed
        dispatcher pins a billable box forever."""
        self.write_hb(self.hb(age_min=90, in_flight=2, open_work=80))
        y, p = self.assert_both(False)
        self.assertIn("stale", y)
        self.assertIn("stale", p)

    def test_missing_heartbeat_does_not_veto(self):
        self.assert_both(False)  # nothing written

    def test_role_mismatch_does_not_veto(self):
        self.write_hb(self.hb(in_flight=9, open_work=9, role="runner"))
        self.assert_both(False)

    def test_missing_demand_fields_does_not_veto(self):
        """A legacy/malformed heartbeat cannot judge demand, so it must
        fail open rather than become what keeps a billable box alive."""
        hb = self.hb()
        del hb["in_flight_dispatches"]
        del hb["chips_open_work"]
        self.write_hb(hb)
        self.assert_both(False)

    def test_non_integer_demand_fields_do_not_veto(self):
        self.write_hb(self.hb(in_flight="2", open_work="80"))
        self.assert_both(False)

    def test_malformed_json_does_not_veto(self):
        self.write_hb(None, raw="{not json at all")
        self.assert_both(False)

    def test_non_object_heartbeat_does_not_veto(self):
        self.write_hb(None, raw="[1, 2, 3]")
        self.assert_both(False)

    def test_missing_tick_does_not_veto(self):
        hb = self.hb(in_flight=2)
        del hb["last_tick_utc"]
        self.write_hb(hb)
        self.assert_both(False)

    def test_unparseable_tick_does_not_veto(self):
        hb = self.hb(in_flight=2)
        hb["last_tick_utc"] = "not-a-timestamp"
        self.write_hb(hb)
        self.assert_both(False)

    def test_other_affinity_heartbeat_is_not_read(self):
        """cloud-3's orchestrator heartbeat must not keep cloud-4 alive."""
        self.write_hb(self.hb(in_flight=5), affinity="ree-cloud-3")
        self.assert_both(False)


class WiringTest(unittest.TestCase):
    """The snippet can be perfect and still never run. These pin the wiring."""

    def test_snippet_contains_no_dollar_sign(self):
        """The program lives inside a DOUBLE-QUOTED bash string, so any `$`
        would be expanded by bash before python ever sees it -- silently
        corrupting the program. The sibling snippets in this workflow follow
        the same constraint."""
        self.assertNotIn(
            "$", ORCH_SNIPPET,
            "`$` inside the double-quoted python -c string is expanded by "
            "bash. Use a format that avoids it, or switch to single quotes "
            "(which would then break the existing `$AFFINITY` style).")

    def test_veto_branch_precedes_the_shutdown_branch(self):
        """Ordering IS the logic in an if/elif chain. A veto placed after the
        shutdown test is unreachable and tests nothing."""
        veto = WORKFLOW_SRC.index('"$ORCH_VETO" = "1"')
        shutdown = WORKFLOW_SRC.index(
            '"$CLAIMABLE" -eq 0 ] && [ "$STATUS" = "running" ]')
        self.assertLess(
            veto, shutdown,
            "the orchestrator veto branch must come BEFORE the shutdown "
            "branch, or it can never fire")

    def test_orch_veto_is_parsed_from_snippet_output(self):
        self.assertIn("ORCH_VETO=$(echo \"$ORCH_INFO\"", WORKFLOW_SRC)

    def test_orch_info_is_logged(self):
        """Without this the journal cannot explain why a box stayed up."""
        self.assertIn("$IDLE_INFO $ORCH_INFO", WORKFLOW_SRC)

    def test_workflow_yaml_parses(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("pyyaml not available")
        d = yaml.safe_load(WORKFLOW_SRC)
        self.assertIn("scale", d["jobs"])

    def test_workflow_bash_body_is_syntactically_valid(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("pyyaml not available")
        d = yaml.safe_load(WORKFLOW_SRC)
        body = [s for s in d["jobs"]["scale"]["steps"]
                if "Check queue" in (s.get("name") or "")][0]["run"]
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
            fh.write(body)
            path = fh.name
        self.addCleanup(os.unlink, path)
        proc = subprocess.run(["bash", "-n", path],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class ConstantParityTest(unittest.TestCase):
    """Shared tunables must hold the same value in both transports."""

    def test_orchestrator_fresh_min_matches(self):
        m = re.search(r"^\s*ORCHESTRATOR_FRESH_MIN:\s*(\d+)",
                      WORKFLOW_SRC, re.M)
        self.assertIsNotNone(m, "ORCHESTRATOR_FRESH_MIN missing from workflow")
        self.assertEqual(
            int(m.group(1)), SCALER.DEFAULTS["ORCHESTRATOR_FRESH_MIN"],
            "ORCHESTRATOR_FRESH_MIN drifted between transports; the two "
            "scalers would age a metaworker out of its veto at different "
            "times, so which one ran last would decide whether a live "
            "dispatcher box survives.")

    def test_idle_grace_and_heartbeat_fresh_match(self):
        for name in ("IDLE_GRACE_MIN", "HEARTBEAT_FRESH_MIN",
                     "SURGE_QUEUE_THRESHOLD"):
            m = re.search(r"^\s*%s:\s*(\d+)" % name, WORKFLOW_SRC, re.M)
            self.assertIsNotNone(m, "%s missing from workflow" % name)
            self.assertEqual(int(m.group(1)), SCALER.DEFAULTS[name],
                             "%s drifted between transports" % name)

    def test_hub_name_matches(self):
        m = re.search(r"^\s*HUB_NAME:\s*(\S+)", WORKFLOW_SRC, re.M)
        self.assertEqual(m.group(1), SCALER.DEFAULTS["HUB_NAME"])

    def test_worker_list_matches(self):
        """A worker present in one transport and absent from the other is
        managed by only one of them -- exactly the asymmetry that let the
        backstop act on a box the hub was protecting."""
        yaml_workers = re.findall(r'"(ree-worker-\d+):(ree-cloud-\d+)(?::(\w+))?"',
                                  WORKFLOW_SRC)
        yaml_set = {(a, b, c or "full") for a, b, c in yaml_workers}
        self.assertEqual(yaml_set, set(SCALER.WORKERS),
                         "WORKERS list drifted between transports")


class OrchestratorAffinityParityTest(unittest.TestCase):
    """ORCHESTRATOR_AFFINITIES is a CONFIG declaration that both transports
    read. It must not drift -- a box declared on one side and not the other is
    the same asymmetry that let the backstop power off a box the hub was
    protecting (2026-08-19T01:39:26Z)."""

    def _yaml_affinities(self):
        m = re.search(r"^\s*ORCHESTRATOR_AFFINITIES:\s*(.+)$",
                      WORKFLOW_SRC, re.M)
        self.assertIsNotNone(
            m, "the workflow must declare ORCHESTRATOR_AFFINITIES")
        return set(m.group(1).split())

    def test_declared_orchestrator_set_matches(self):
        self.assertEqual(self._yaml_affinities(),
                         set(SCALER.ORCHESTRATOR_AFFINITIES),
                         "ORCHESTRATOR_AFFINITIES drifted between transports")

    def test_the_set_is_not_empty(self):
        """An empty set silently disables both the wake hold and the backstop
        exclusion, restoring the exact 2026-08-19 failure with no error."""
        self.assertTrue(SCALER.ORCHESTRATOR_AFFINITIES)

    def test_declared_boxes_are_real_managed_workers(self):
        """A typo'd affinity would match nothing and protect nothing."""
        managed = {aff for _, aff, _ in SCALER.WORKERS}
        for aff in SCALER.ORCHESTRATOR_AFFINITIES:
            self.assertIn(aff, managed,
                          "%s is declared an orchestrator but is not a "
                          "scaler-managed worker" % aff)

    def test_the_hub_is_never_declared_an_orchestrator(self):
        """Negative control. The hub is skipped before any decision anyway,
        but declaring it here would be a second, contradictory statement
        about the one box the 2026-05-28 incident is about."""
        self.assertNotIn("ree-cloud-1", SCALER.ORCHESTRATOR_AFFINITIES)


class BackstopShutdownExclusionTest(unittest.TestCase):
    """The GHA backstop holds NO shutdown authority over a declared
    orchestrator box. This is the second deliberate divergence."""

    def test_exclusion_branch_exists(self):
        self.assertIn('"$IS_ORCHESTRATOR" = "1"', WORKFLOW_SRC)

    def test_is_orchestrator_is_actually_computed(self):
        """A branch on an always-empty variable is a no-op that reads as a
        guard -- the vacuous-assertion shape this file exists to catch."""
        self.assertIn("IS_ORCHESTRATOR=0", WORKFLOW_SRC)
        self.assertIn("IS_ORCHESTRATOR=1", WORKFLOW_SRC)
        self.assertIn("${ORCHESTRATOR_AFFINITIES}", WORKFLOW_SRC)

    def test_exclusion_precedes_both_the_veto_and_the_shutdown_branch(self):
        """Ordering IS the logic. Placed after ORCH_VETO it would only fire
        when the git heartbeat already said 'idle and empty' -- i.e. never
        when it matters, since that is precisely the state a stale git
        heartbeat produces for a box that is actually busy."""
        excl = WORKFLOW_SRC.index('"$IS_ORCHESTRATOR" = "1"')
        veto = WORKFLOW_SRC.index('"$ORCH_VETO" = "1"')
        shutdown = WORKFLOW_SRC.index(
            '"$CLAIMABLE" -eq 0 ] && [ "$STATUS" = "running" ]')
        self.assertLess(excl, veto)
        self.assertLess(excl, shutdown)

    def test_power_ON_authority_is_retained(self):
        """Only the SHUTDOWN branch is withheld. The backstop must still be
        able to start a box -- withholding that would make the backstop
        useless for the surge case it exists to cover."""
        excl = WORKFLOW_SRC.index('"$IS_ORCHESTRATOR" = "1"')
        poweron = WORKFLOW_SRC.index('hcloud server poweron')
        self.assertLess(poweron, excl,
                        "the power-on branches must come before the "
                        "orchestrator exclusion, or they become unreachable")

    def test_header_documents_the_second_divergence(self):
        self.assertIn("TWO divergences", WORKFLOW_SRC)
        self.assertIn("COORDINATOR-PRIMARY", WORKFLOW_SRC)


class DeliberateDivergenceTest(unittest.TestCase):
    """The divergences that are structural rather than drift."""

    def test_lease_veto_absence_is_deliberate(self):
        """cloud-scaler.py's pytest lease veto reads a HUB-LOCAL directory
        (/home/ree/pytest_leases). A GitHub-hosted runner has no access to
        it, so this cannot be ported -- and must not be "fixed" by inventing
        a path that will always be missing (which would fail open forever
        while looking like parity). The header must say so."""
        self.assertTrue(hasattr(SCALER, "read_lease"))
        self.assertNotIn("PYTEST_LEASE_DIR", WORKFLOW_SRC,
                         "a lease dir in the workflow can never resolve on a "
                         "GitHub-hosted runner; see this test's docstring")
        self.assertIn("pytest lease veto", WORKFLOW_SRC,
                      "the header must document why this divergence exists, "
                      "so the next reader does not re-assert 1:1 parity")

    def test_header_no_longer_ASSERTS_identical_parity(self):
        """The header must not re-assert parity as a fact.

        Deliberately matches the ASSERTION, not the phrase: the current
        header quotes "identical 1:1" precisely in order to say it was
        false, and a naive substring ban would forbid the correction along
        with the error.
        """
        self.assertNotIn(
            "Logic in both transports is identical 1:1", WORKFLOW_SRC,
            "that claim was false and cost a 15h outage; describe the "
            "divergences instead of asserting there are none")
        self.assertIn("TRANSPORT PARITY", WORKFLOW_SRC,
                      "the header must carry the divergence note")


if __name__ == "__main__":
    unittest.main(verbosity=2)
