"""Contracts for systemd_dropin_drift.py.

TIME-INDEPENDENT and FLEET-INDEPENDENT by construction: every test injects
fixture `systemctl show` text via `_show_text` (or a stub `_run_ssh`), so
nothing here sshes anywhere, powers anything on, or depends on what the fleet
happens to be doing. A test that needed a live worker would be skipped exactly
when the fleet was broken -- i.e. when it mattered.

The load-bearing test is test_negative_control_matching_machine_has_no_finding.
An audit whose all-clear is vacuous is worse than no audit: it reports a
confident pass while grading nothing. So that test asserts BOTH "zero findings"
AND "a nonzero number of directives were actually compared" -- the second half
is what makes the first half mean something. (Same trap that produced ten
vacuously-passing gate assertions in precommit_contracts on 2026-07-27.)
"""

from __future__ import annotations

import os

import pytest

import systemd_dropin_drift as sdd


DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))


# A self-contained template/live pair that MATCHES. Deliberately exercises
# every parser branch: a plain directive, a quoted Environment, an
# angle-bracket placeholder, and a commented staged-rollout line.
MATCHING_TEMPLATE = """\
# Header prose that mentions Restart and Environment in passing.
# Staged rollout, enable canary first:
# Environment=STAGED_FLAG=1

[Service]
WorkingDirectory=/home/ree/REE_Working_runner/ree-v3
Environment=COORDINATION_MODE=coordinator
Environment=COORDINATOR_TOKEN=<paste from coordinator.env>
Environment="PHASE3_DISABLE_RUNNER_RESULT_PUSH=1"
Restart=always
"""

MATCHING_SHOW = """\
ExecStart={ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 \
experiment_runner.py --auto-sync --machine ree-cloud-9 ; ignore_errors=no }
ActiveState=active
DropInPaths=/etc/systemd/system/ree-runner.service.d/shadow.conf
FragmentPath=/etc/systemd/system/ree-runner.service
WorkingDirectory=/home/ree/REE_Working_runner/ree-v3
Restart=always
Environment=COORDINATION_MODE=coordinator COORDINATOR_TOKEN=s3cr3t-value \
PHASE3_DISABLE_RUNNER_RESULT_PUSH=1
"""


@pytest.fixture()
def fake_machine(tmp_path):
    """A machine spec pointing at a throwaway template directory."""
    (tmp_path / "shadow.conf.fake.example").write_text(
        MATCHING_TEMPLATE, encoding="utf-8")
    return {
        "hcloud": "ree-worker-9",
        "affinity": "ree-cloud-9",
        "host": "203.0.113.9",
        "template": "shadow.conf.fake.example",
        "note": "",
    }, str(tmp_path)


def _verdicts(finding):
    return {r["directive"]: r["verdict"] for r in finding["results"]}


# --------------------------------------------------------------------------
# THE NEGATIVE CONTROL
# --------------------------------------------------------------------------

def test_negative_control_matching_machine_has_no_finding(fake_machine):
    """A machine that matches its template produces NO finding and NO note.

    Anti-vacuous: also asserts real directives were compared and graded OK.
    Without that, this test would still pass if compare() returned [].
    """
    spec, deploy_dir = fake_machine
    finding = sdd.check_machine(
        spec, deploy_dir=deploy_dir, _show_text=MATCHING_SHOW)

    assert finding["status"] == sdd.OK
    findings = [r for r in finding["results"]
                if r["verdict"] in sdd.FINDING_VERDICTS]
    notes = [r for r in finding["results"]
             if r["verdict"] in sdd.NOTE_VERDICTS]
    assert findings == [], "clean machine must yield no finding"
    assert notes == [], "clean machine must yield no note"

    # Anti-vacuous half: the pass must come from comparing things.
    ok_rows = [r for r in finding["results"] if r["verdict"] == sdd.OK]
    assert len(ok_rows) >= 5, (
        "all-clear is vacuous -- only %d directive(s) were graded" % len(ok_rows))
    graded = _verdicts(finding)
    for key in ("WorkingDirectory", "Restart", "COORDINATION_MODE",
                "COORDINATOR_TOKEN", "PHASE3_DISABLE_RUNNER_RESULT_PUSH"):
        assert graded[key] == sdd.OK

    assert sdd.count_findings([finding]) == 0
    assert sdd.count_notes([finding]) == 0


def test_negative_control_is_sensitive(fake_machine):
    """The clean fixture is one edit away from failing -- proof the OK verdict
    is responsive to the input rather than structurally guaranteed."""
    spec, deploy_dir = fake_machine
    broken = MATCHING_SHOW.replace("Restart=always", "Restart=on-failure")
    finding = sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=broken)
    assert _verdicts(finding)["Restart"] == sdd.DIFFERENT
    assert sdd.count_findings([finding]) == 1


# --------------------------------------------------------------------------
# The two confirmed incidents
# --------------------------------------------------------------------------

def test_incident_1_restart_always_shadowed_by_main_unit(fake_machine):
    """Incident 1: the hub's drop-in lacked Restart=always, so the main unit's
    Restart=on-failure governed and a clean exit-0 drain never respawned.

    The merged value is the only place this is visible, which is why the audit
    reads `systemctl show` rather than the drop-in file.
    """
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace("Restart=always", "Restart=on-failure")
    finding = sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=show)

    row = [r for r in finding["results"] if r["directive"] == "Restart"][0]
    assert row["verdict"] == sdd.DIFFERENT
    assert row["template"] == "always"
    assert row["live"] == "on-failure"
    assert finding["status"] in sdd.FINDING_VERDICTS


def test_incident_2_missing_env_directive_is_a_finding(fake_machine):
    """Incident 2: a directive present in the reviewed template and on zero
    machines. `systemctl show -p Environment` is exhaustive, so absence is
    definitive -- this must be MISSING, never UNKNOWN."""
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace(" PHASE3_DISABLE_RUNNER_RESULT_PUSH=1", "")
    finding = sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=show)

    row = [r for r in finding["results"]
           if r["directive"] == "PHASE3_DISABLE_RUNNER_RESULT_PUSH"][0]
    assert row["verdict"] == sdd.MISSING
    assert row["live"] is None
    assert sdd.count_findings([finding]) == 1


def test_real_templates_still_declare_the_incident_directives():
    """Integration guard on the TRACKED templates.

    If a template is renamed, reformatted, or has its [Service] block
    restructured such that the parser stops seeing directives, every machine
    would silently grade clean. This asserts the real files still yield the
    directives whose absence caused the two incidents.
    """
    for name in ("shadow.conf.hub.example", "shadow.conf.worker.example"):
        path = os.path.join(DEPLOY_DIR, name)
        with open(path, "r", encoding="utf-8") as fh:
            declared, _mentioned = sdd.parse_template(fh.read())
        assert declared, "%s parsed to zero directives" % name
        assert "PHASE3_DISABLE_RUNNER_CLAIM_PUSH" in declared, name
        assert declared["PHASE3_DISABLE_RUNNER_CLAIM_PUSH"] == ("1", "env")
        assert "Restart" in declared, name
        assert declared["Restart"] == ("always", "prop")


def test_hub_template_does_not_declare_heartbeat_write():
    """PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE is obsolete under the isolated
    checkout and re-arms the hub command spin. If it ever reappears as an
    ACTIVE hub directive, the audit would start demanding it on the hub."""
    path = os.path.join(DEPLOY_DIR, "shadow.conf.hub.example")
    with open(path, "r", encoding="utf-8") as fh:
        declared, _ = sdd.parse_template(fh.read())
    assert "PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE" not in declared


# --------------------------------------------------------------------------
# Template parsing
# --------------------------------------------------------------------------

def test_parse_template_separates_active_from_commented():
    declared, mentioned = sdd.parse_template(MATCHING_TEMPLATE)
    assert declared["COORDINATION_MODE"] == ("coordinator", "env")
    assert declared["Restart"] == ("always", "prop")
    assert "STAGED_FLAG" in mentioned
    assert "STAGED_FLAG" not in declared


def test_parse_template_handles_quoted_environment():
    """The live fleet writes Environment="KEY=1"; the templates use both forms."""
    declared, _ = sdd.parse_template(
        '[Service]\nEnvironment="PHASE3_X=1"\nEnvironment=PHASE3_Y=2\n')
    assert declared["PHASE3_X"] == ("1", "env")
    assert declared["PHASE3_Y"] == ("2", "env")


def test_parse_template_ignores_directives_outside_service():
    declared, _ = sdd.parse_template(
        "[Unit]\nDescription=x\n[Service]\nRestart=always\n")
    assert "Description" not in declared
    assert "Restart" in declared


def test_declared_key_is_not_also_mentioned():
    """A key both active and discussed in comments is DECLARED, so it can
    never be graded UNDECLARED."""
    declared, mentioned = sdd.parse_template(
        "# Environment=DUP=1 explained here\n[Service]\nEnvironment=DUP=1\n")
    assert "DUP" in declared
    assert "DUP" not in mentioned


# --------------------------------------------------------------------------
# systemctl show parsing
# --------------------------------------------------------------------------

def test_parse_show_explodes_environment():
    live = sdd.parse_show(
        "Restart=always\nEnvironment=A=1 B=two C=http://h:8787\n")
    assert live["_env"] == {"A": "1", "B": "two", "C": "http://h:8787"}
    assert live["_props"]["Restart"] == "always"


def test_parse_show_survives_unbalanced_quotes():
    """Never raise on odd input -- a parser crash would take the whole fleet
    report down over one malformed value."""
    live = sdd.parse_show('Environment=A=1 B="unterminated\n')
    assert "A" in live["_env"]


# --------------------------------------------------------------------------
# EXTRA vs UNDECLARED
# --------------------------------------------------------------------------

def test_undeclared_when_template_mentions_it_commented(fake_machine):
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace(
        "Environment=COORDINATION_MODE=coordinator",
        "Environment=STAGED_FLAG=1 COORDINATION_MODE=coordinator")
    finding = sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=show)
    assert _verdicts(finding)["STAGED_FLAG"] == sdd.UNDECLARED
    # A note, not a finding: the machine works, the docs lag.
    assert sdd.count_findings([finding]) == 0
    assert sdd.count_notes([finding]) == 1


def test_extra_when_template_never_mentions_it(fake_machine):
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace(
        "Environment=COORDINATION_MODE=coordinator",
        "Environment=WHOLLY_UNKNOWN=1 COORDINATION_MODE=coordinator")
    finding = sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=show)
    assert _verdicts(finding)["WHOLLY_UNKNOWN"] == sdd.EXTRA
    assert sdd.count_findings([finding]) == 0


def test_extra_scoped_to_environment_not_systemd_defaults(fake_machine):
    """systemd exposes hundreds of default properties the template was never
    meant to restate. Grading those EXTRA would bury the real signal."""
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW + "LimitNOFILE=524288\nCPUWeight=100\n"
    finding = sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=show)
    graded = _verdicts(finding)
    assert "LimitNOFILE" not in graded
    assert "CPUWeight" not in graded
    assert sdd.count_notes([finding]) == 0


# --------------------------------------------------------------------------
# Per-machine values and secrets
# --------------------------------------------------------------------------

def test_placeholder_value_is_presence_only(fake_machine):
    """COORDINATOR_TOKEN legitimately differs per worker; the template carries
    `<paste from coordinator.env>`. Presence is checked, value never is."""
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace("COORDINATOR_TOKEN=s3cr3t-value",
                                 "COORDINATOR_TOKEN=a-completely-different-one")
    finding = sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=show)
    assert _verdicts(finding)["COORDINATOR_TOKEN"] == sdd.OK
    assert sdd.count_findings([finding]) == 0


def test_absent_secret_is_still_a_finding(fake_machine):
    """Presence-only must not degrade to not-checked-at-all: a runner with no
    COORDINATOR_TOKEN cannot talk to the coordinator."""
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace(" COORDINATOR_TOKEN=s3cr3t-value", "")
    finding = sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=show)
    assert _verdicts(finding)["COORDINATOR_TOKEN"] == sdd.MISSING


def test_secret_values_never_appear_in_output(fake_machine):
    """The report is pasted into notes and commit messages. A live token must
    not survive into any rendered line or JSON field."""
    spec, deploy_dir = fake_machine
    finding = sdd.check_machine(
        spec, deploy_dir=deploy_dir, _show_text=MATCHING_SHOW)
    rendered = "\n".join(sdd.render([finding], show_ok=True))
    assert "s3cr3t-value" not in rendered
    assert "<redacted:" in rendered
    for row in finding["results"]:
        assert row["live"] != "s3cr3t-value"
        assert row["template"] != "s3cr3t-value"


def test_redact_only_masks_secret_shaped_keys():
    assert sdd.redact("COORDINATOR_TOKEN", "abc") == "<redacted:3 chars>"
    assert sdd.redact("COORDINATOR_URL", "http://h") == "http://h"
    assert sdd.redact("COORDINATOR_TOKEN", None) is None


# --------------------------------------------------------------------------
# Machine affinity (documented in template prose, not as a directive)
# --------------------------------------------------------------------------

def test_affinity_pinned_is_ok(fake_machine):
    spec, deploy_dir = fake_machine
    finding = sdd.check_machine(
        spec, deploy_dir=deploy_dir, _show_text=MATCHING_SHOW)
    assert _verdicts(finding)["ExecStart:--machine"] == sdd.OK


def test_affinity_unpinned_is_a_finding(fake_machine):
    """socket.gethostname() on the hub is `ree-worker-1`, so an unpinned hub
    runner reports under a machine identity the queue does not know."""
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace(" --machine ree-cloud-9", "")
    finding = sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=show)
    assert _verdicts(finding)["ExecStart:--machine"] == sdd.DIFFERENT
    assert sdd.count_findings([finding]) == 1


def test_affinity_wrong_machine_is_a_finding(fake_machine):
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace("--machine ree-cloud-9",
                                 "--machine ree-cloud-2")
    finding = sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=show)
    assert _verdicts(finding)["ExecStart:--machine"] == sdd.DIFFERENT


# --------------------------------------------------------------------------
# Unreachable / unknown handling
# --------------------------------------------------------------------------

def test_unreachable_machine_is_not_a_finding(fake_machine):
    """A powered-off worker is NORMAL. It must never be graded as drift, and
    must never be woken (billable)."""
    spec, deploy_dir = fake_machine
    finding = sdd.check_machine(
        spec, deploy_dir=deploy_dir, _run_ssh=lambda argv: None)
    assert finding["status"] == sdd.UNREACHABLE
    assert sdd.count_findings([finding]) == 0
    assert sdd.count_notes([finding]) == 0
    assert finding["results"] == []


def test_unreachable_is_visible_in_the_summary(fake_machine):
    spec, deploy_dir = fake_machine
    finding = sdd.check_machine(
        spec, deploy_dir=deploy_dir, _run_ssh=lambda argv: None)
    rendered = "\n".join(sdd.render([finding]))
    assert "1 unreachable" in rendered
    assert "ree-worker-9" in rendered


def test_unexposed_property_is_unknown_not_missing(tmp_path):
    """A property `systemctl show` does not expose may simply not be a
    queryable name. A false MISSING against a reviewed template erodes trust
    in the check, so this degrades to UNKNOWN."""
    (tmp_path / "t.example").write_text(
        "[Service]\nNotARealProperty=x\n", encoding="utf-8")
    spec = {"hcloud": "h", "affinity": "a", "host": "203.0.113.9",
            "template": "t.example", "note": ""}
    finding = sdd.check_machine(
        spec, deploy_dir=str(tmp_path),
        _show_text="ActiveState=active\nEnvironment=\n")
    assert _verdicts(finding)["NotARealProperty"] == sdd.UNKNOWN
    assert sdd.count_findings([finding]) == 0


def test_missing_template_file_does_not_raise(tmp_path):
    spec = {"hcloud": "h", "affinity": "a", "host": "203.0.113.9",
            "template": "nope.example", "note": ""}
    finding = sdd.check_machine(spec, deploy_dir=str(tmp_path))
    assert finding["status"] == sdd.UNKNOWN
    assert "cannot read template" in finding["detail"]


# --------------------------------------------------------------------------
# Queried-property derivation, exit codes, safety
# --------------------------------------------------------------------------

def test_queried_properties_are_derived_from_the_template(tmp_path):
    """Adding a directive to a template must automatically extend coverage.
    A hardcoded property list would silently stop covering the next
    Restart=always."""
    (tmp_path / "t.example").write_text(
        "[Service]\nRestart=always\nSomeNewDirective=7\n", encoding="utf-8")
    spec = {"hcloud": "h", "affinity": "a", "host": "203.0.113.9",
            "template": "t.example", "note": ""}
    seen = {}

    def fake_ssh(argv):
        seen["cmd"] = argv[-1]
        return "Restart=always\nSomeNewDirective=7\nEnvironment=\n"

    sdd.check_machine(spec, deploy_dir=str(tmp_path), _run_ssh=fake_ssh)
    assert "-p SomeNewDirective" in seen["cmd"]
    assert "-p Restart" in seen["cmd"]
    assert "-p Environment" in seen["cmd"]


def test_remote_command_is_read_only(tmp_path):
    """The tool must never write to a live unit, and must never touch
    /home/ree/REE_Working on the hub (dirtying it wedges every Phase-3
    writer). `systemctl show` is a property query."""
    (tmp_path / "t.example").write_text(
        "[Service]\nRestart=always\n", encoding="utf-8")
    spec = {"hcloud": "h", "affinity": "a", "host": "203.0.113.9",
            "template": "t.example", "note": ""}
    seen = {}

    def fake_ssh(argv):
        seen["argv"] = argv
        return "Restart=always\nEnvironment=\n"

    sdd.check_machine(spec, deploy_dir=str(tmp_path), _run_ssh=fake_ssh)
    cmd = seen["argv"][-1]

    # Assert at TOKEN level, not by substring. A substring blocklist is both
    # too weak (misses anything not enumerated) and too strong -- "start" is a
    # substring of the property names ExecStart and Restart, so a naive check
    # fails on a command that is perfectly read-only. Whitelisting the shape
    # is the honest form of this contract.
    tokens = cmd.split()
    assert tokens[0] == "systemctl"
    assert tokens[1] == "show", "subcommand must be the read-only property query"
    assert tokens[2] == sdd.UNIT
    for token in tokens[3:]:
        assert token == "-p" or token.replace("_", "").isalnum(), (
            "unexpected token %r in remote command" % token)

    # No shell metacharacter can reach the remote shell, so the command cannot
    # be extended into a write.
    for meta in (";", "|", "&", ">", "<", "$", "`", "(", ")", "\n"):
        assert meta not in cmd, "shell metacharacter %r in remote command" % meta

    assert "-o BatchMode=yes" in " ".join(seen["argv"])


def test_default_exit_is_zero_even_with_findings(fake_machine, monkeypatch):
    """Exit 0 by default so it chains safely in /session-land."""
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace("Restart=always", "Restart=on-failure")
    monkeypatch.setattr(sdd, "MACHINES", [spec])
    monkeypatch.setattr(sdd, "fetch_show", lambda *a, **k: show)
    assert sdd.main(["--deploy-dir", deploy_dir]) == 0


def test_exit_nonzero_gates_on_findings(fake_machine, monkeypatch):
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace("Restart=always", "Restart=on-failure")
    monkeypatch.setattr(sdd, "MACHINES", [spec])
    monkeypatch.setattr(sdd, "fetch_show", lambda *a, **k: show)
    assert sdd.main(["--deploy-dir", deploy_dir, "--exit-nonzero"]) == 1


def test_exit_nonzero_is_clean_when_only_notes(fake_machine, monkeypatch):
    """Notes alone do not fail the gate; --strict opts into that."""
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace(
        "Environment=COORDINATION_MODE=coordinator",
        "Environment=WHOLLY_UNKNOWN=1 COORDINATION_MODE=coordinator")
    monkeypatch.setattr(sdd, "MACHINES", [spec])
    monkeypatch.setattr(sdd, "fetch_show", lambda *a, **k: show)
    assert sdd.main(["--deploy-dir", deploy_dir, "--exit-nonzero"]) == 0
    assert sdd.main(
        ["--deploy-dir", deploy_dir, "--exit-nonzero", "--strict"]) == 1


def test_exit_zero_on_clean_fleet(fake_machine, monkeypatch):
    spec, deploy_dir = fake_machine
    monkeypatch.setattr(sdd, "MACHINES", [spec])
    monkeypatch.setattr(sdd, "fetch_show", lambda *a, **k: MATCHING_SHOW)
    assert sdd.main(
        ["--deploy-dir", deploy_dir, "--exit-nonzero", "--strict"]) == 0


def test_machine_filter_selects_by_affinity_or_hcloud_name(
        fake_machine, monkeypatch):
    spec, deploy_dir = fake_machine
    monkeypatch.setattr(sdd, "MACHINES", [spec])
    monkeypatch.setattr(sdd, "fetch_show", lambda *a, **k: MATCHING_SHOW)
    assert sdd.main(["--deploy-dir", deploy_dir,
                     "--machine", "ree-cloud-9"]) == 0
    assert sdd.main(["--deploy-dir", deploy_dir,
                     "--machine", "ree-worker-9"]) == 0
    assert sdd.main(["--deploy-dir", deploy_dir,
                     "--machine", "no-such-box"]) == 2


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_report_renders_every_machine_including_all_clear(fake_machine):
    """A section that appears only on failure cannot be distinguished from one
    that silently stopped running."""
    spec, deploy_dir = fake_machine
    finding = sdd.check_machine(
        spec, deploy_dir=deploy_dir, _show_text=MATCHING_SHOW)
    rendered = "\n".join(sdd.render([finding]))
    assert "ree-worker-9" in rendered
    assert "0 finding(s)" in rendered


def test_report_names_remediation_but_never_performs_it(fake_machine):
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace("Restart=always", "Restart=on-failure")
    finding = sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=show)
    rendered = "\n".join(sdd.render([finding]))
    assert "never edits a live unit" in rendered
    assert "daemon-reload" in rendered
    assert "never touch /home/ree/REE_Working" in rendered


def test_all_output_is_ascii(fake_machine):
    """ASCII-only: non-ASCII renders as mojibake on Windows terminals."""
    spec, deploy_dir = fake_machine
    show = MATCHING_SHOW.replace("Restart=always", "Restart=on-failure")
    findings = [
        sdd.check_machine(spec, deploy_dir=deploy_dir, _show_text=show),
        sdd.check_machine(spec, deploy_dir=deploy_dir,
                          _run_ssh=lambda argv: None),
    ]
    rendered = "\n".join(sdd.render(findings, show_ok=True))
    rendered.encode("ascii")


def test_module_source_prints_are_ascii():
    """The docstring and every literal in the module must be ASCII-safe."""
    path = os.path.join(DEPLOY_DIR, "systemd_dropin_drift.py")
    with open(path, "rb") as fh:
        fh.read().decode("ascii")


def test_machines_table_covers_the_documented_fleet():
    """Each machine must be graded against ITS OWN template -- the hub
    legitimately differs (isolated checkout, no _HEARTBEAT_WRITE)."""
    by_affinity = {m["affinity"]: m for m in sdd.MACHINES}
    assert set(by_affinity) == {
        "ree-cloud-1", "ree-cloud-2", "ree-cloud-3", "ree-cloud-4"}
    assert by_affinity["ree-cloud-1"]["template"] == "shadow.conf.hub.example"
    for affinity in ("ree-cloud-2", "ree-cloud-3", "ree-cloud-4"):
        assert by_affinity[affinity]["template"] == (
            "shadow.conf.worker.example")
    assert by_affinity["ree-cloud-1"]["hcloud"] == "ree-worker-1"
