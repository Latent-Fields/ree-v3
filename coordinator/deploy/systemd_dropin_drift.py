"""Fleet systemd drop-in drift check -- does the LIVE unit config match the
tracked template?

WHY THIS EXISTS
    `coordinator/deploy/shadow.conf.hub.example` and
    `shadow.conf.worker.example` are the documented source of truth for each
    machine's /etc/systemd/system/ree-runner.service.d/ drop-in. They are
    tracked, reviewed and reasoned about in commit messages -- and until this
    check existed, NOTHING verified that any machine actually matched them.
    The templates were a description of intent that no one compared to
    reality, so every divergence was found by a human noticing an outage.

    Two confirmed instances of that, both found by accident:

    1. Restart=always missing on the hub for ~2 months. cloud-2 and cloud-3
       carried it in their drop-in since 2026-05-30; the hub never got it.
       The main unit's Restart=on-failure does not respawn on a CLEAN exit, so
       when the hub runner drained cleanly (exit 0) at 2026-07-30T06:07:04Z it
       stayed dead for hours. Worse: ExecStartPre=runner-prestart-pull.sh only
       pulls at START, so a runner that never restarts never reloads code --
       the hub had been executing r4072 35c103c (2026-07-20) for TEN DAYS.
       Fixed 2026-07-30 (ree-v3 3544757 + a live drop-in edit), but only
       because a human noticed the runner was down.

    2. PHASE3_DISABLE_RUNNER_CLAIM_PUSH=1 is in BOTH tracked templates and on
       NO live machine. Verified 2026-07-30 across all four boxes. Impact is
       currently ~zero (0 legacy `claim:` commits in the last 200 on
       origin/main, because PHASE3_DISABLE_RUNNER_QUEUE_PUSH=1 already
       suppresses that push via _claim_push_gated). It is reported not because
       it is urgent but because it is EVIDENCE THE DRIFT IS SYSTEMIC rather
       than a one-off: a directive can sit in a reviewed template, on zero
       machines, for two months, with nothing noticing.

WHAT IT COMPARES
    Per machine: every [Service] directive in that machine's template against
    the EFFECTIVE merged value from `systemctl show`. Four verdicts, because
    "the live unit does not match the template" has four distinct causes with
    different remedies:

      MISSING     -- template declares it (uncommented), live does not have
                     it. The incident-1 shape. Fix: add it to the drop-in.
      DIFFERENT   -- both have it, values disagree. Also the incident-1 shape
                     when a main-unit default (Restart=on-failure) shadows an
                     un-applied template directive (Restart=always).
      EXTRA       -- live has a directive the template never mentions at all.
                     The template is missing a machine's real config.
      UNDECLARED  -- live has it, but the template carries it only as a
                     COMMENTED staged-rollout line ("enable canary first").
                     Means a rollout completed and was never recorded. Real
                     drift, but of the docs, so it is a NOTE not a finding.

    Effective merged values are read via `systemctl show`, NOT by parsing
    drop-in files. That is deliberate and load-bearing:
      - systemd merges EVERY *.conf in the .d directory. The fleet has a
        second live drop-in (sidefiles.conf) beside shadow.conf, so a
        shadow.conf-only reader grades against a partial config.
      - systemd IGNORES files not ending in .conf. Every box carries 4-5
        shadow.conf.bak.* backups; a naive `cat *.conf*` would read retired
        config as live. (A backup named `<something>.conf` WOULD be read --
        which is why DropInPaths is reported, so a stray one is visible.)
      - A main-unit default that shadows an unapplied drop-in directive is
        only visible in the merged value. That IS incident 1.

WHY IT LIVES HERE, AS A SIBLING OF daemon_code_drift.py
    daemon_code_drift.py is the closest precedent and this mirrors its shape,
    but it is a separate module rather than an extension of it, for two
    reasons that are not style preferences:
      - Different subject. That module asks "is the running PROCESS bound to
        stale bytecode"; this asks "is the unit CONFIG that launches the
        process the one we think it is". A machine can be perfectly current on
        one and drifted on the other -- incident 1 was both at once, and the
        code-drift check could not have caught it, because a runner that never
        restarts is trivially "current" against whatever it started with.
      - Different execution locus. daemon_code_drift runs HUB-LOCAL, imported
        by the ree-live-status oneshot on a 3-minute timer. This one fans out
        over ssh to four boxes. Putting network fan-out to the whole fleet on
        a 3-minute timer would be a different and worse thing than either
        check is today.

DETECTION ONLY
    This never writes to /etc/systemd/system/**, never runs daemon-reload, and
    never restarts a unit. It prints the exact remediation command instead.
    Repair direction is the documented failure mode in the analogous
    audit_vendored_copies.py work (REE_Working/CLAUDE.md, Session Startup step
    7a): a repair that runs the wrong way destroys the thing it was meant to
    protect. Here the asymmetry is sharper still -- a wrong automatic edit to
    a live unit takes a fleet machine down.

    It also never powers a machine on. Workers are powered off routinely by
    the cloud-scaler and that is NORMAL, not a fault: an unreachable box is
    reported UNREACHABLE (an unknown), never as a finding. Waking one is
    billable, and auditing is not a good enough reason.

    Secret-shaped values (TOKEN/SECRET/KEY/PASSWORD) are REDACTED in all
    output. COORDINATOR_TOKEN legitimately differs per worker and the template
    carries a `<paste from coordinator.env>` placeholder, so any template value
    in <angle brackets> is compared for PRESENCE ONLY, never by value.

EXIT STATUS
    Exit 0 even with findings, so it chains safely in /session-land. Same
    convention as audit_stashes.py / audit_vendored_copies.py /
    audit_worktree_skills.py. --exit-nonzero opts into a gate.

All printed output (stdout/stderr) is ASCII-only.

  python3 systemd_dropin_drift.py                 # operator check
  python3 systemd_dropin_drift.py --json
  python3 systemd_dropin_drift.py --machine ree-cloud-1
  python3 systemd_dropin_drift.py --exit-nonzero  # gate
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

UNIT = "ree-runner"
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))

SSH_CONNECT_TIMEOUT_SECONDS = 10
SSH_TIMEOUT_SECONDS = 30.0

# Verdicts.
OK = "OK"
MISSING = "MISSING"
DIFFERENT = "DIFFERENT"
EXTRA = "EXTRA"
UNDECLARED = "UNDECLARED"
UNREACHABLE = "UNREACHABLE"
UNKNOWN = "UNKNOWN"

# MISSING/DIFFERENT are findings (the live unit does not do what the reviewed
# template says it does). EXTRA/UNDECLARED are notes: the machine is doing
# something real that the template fails to record, which is a docs defect
# rather than an operational one.
FINDING_VERDICTS = (MISSING, DIFFERENT)
NOTE_VERDICTS = (EXTRA, UNDECLARED)

# Values that legitimately differ per machine are never compared BY VALUE.
# Presence is still checked -- a missing COORDINATOR_TOKEN is a real fault.
SECRET_HINTS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "PASSWD")

# Directives whose value is per-machine by construction even without an
# angle-bracket placeholder in the template.
PER_MACHINE_KEYS = ("COORDINATOR_TOKEN",)

# The hub legitimately diverges from the workers -- isolated checkout, and it
# must NOT carry PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE (obsolete under the
# isolated-checkout model; it re-arms the command spin). So each machine is
# graded against ITS OWN template, never against another machine.
MACHINES = [
    {
        "hcloud": "ree-worker-1",
        "affinity": "ree-cloud-1",
        "host": "91.98.130.117",
        "template": "shadow.conf.hub.example",
        "note": "hub -- coordinator + sync_daemon; isolated runner checkout",
    },
    {
        "hcloud": "ree-worker-2",
        "affinity": "ree-cloud-2",
        "host": "116.203.216.181",
        "template": "shadow.conf.worker.example",
        "note": "",
    },
    {
        "hcloud": "ree-worker-3",
        "affinity": "ree-cloud-3",
        "host": "46.62.170.133",
        "template": "shadow.conf.worker.example",
        "note": "",
    },
    {
        "hcloud": "ree-worker-4",
        "affinity": "ree-cloud-4",
        "host": "91.99.68.94",
        "template": "shadow.conf.worker.example",
        "note": "surge -- manual start only; powered off is NORMAL",
    },
]

# systemd properties always queried, independent of what the templates declare.
BASE_PROPERTIES = ("ExecStart", "ActiveState", "DropInPaths", "FragmentPath")


def _is_secret(key):
    upper = key.upper()
    return any(hint in upper for hint in SECRET_HINTS)


def redact(key, value):
    """Never print a live secret. Presence is the auditable property."""
    if value is None:
        return None
    if _is_secret(key):
        return "<redacted:%d chars>" % len(value)
    return value


def _is_placeholder(value):
    """A template value like `<paste from coordinator.env>` declares that the
    directive must be PRESENT while saying nothing about what it should be."""
    v = (value or "").strip()
    return v.startswith("<") and v.endswith(">")


def parse_template(text):
    """Parse a tracked shadow.conf.*.example.

    Returns (declared, mentioned):
      declared  -- {key: (value, kind)} for ACTIVE (uncommented) [Service]
                   directives, where kind is "env" or "prop".
                   Environment=K=V is keyed by K with kind "env"; anything
                   else by its directive name (WorkingDirectory, Restart, ...)
                   with kind "prop".
      mentioned -- {key} for directives that appear ONLY inside comments.

    The kind is not bookkeeping -- it decides what ABSENCE means, and getting
    it wrong silences the check. `systemctl show -p Environment` enumerates the
    environment exhaustively, so an absent env key is DEFINITIVELY missing.
    A `-p SomeProperty` that comes back unset, by contrast, may just not be a
    queryable property name, so it can only be reported UNKNOWN. Collapsing
    the two turns finding 2 (PHASE3_DISABLE_RUNNER_CLAIM_PUSH absent
    fleet-wide) into a shrug.

    `mentioned` is what separates UNDECLARED from EXTRA. Both templates carry
    commented staged-rollout lines ("Step 1 (canary): ...
    # Environment=PHASE3_COMMANDS_VIA_COORDINATOR=1"). A live machine running
    such a flag is not an unexplained EXTRA -- it is a rollout that finished
    and was never written down. Collapsing the two would either bury a real
    unknown-config finding in rollout noise, or silently bless config the
    template never contemplated.
    """
    declared = {}
    mentioned = set()
    in_service = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("["):
            in_service = line.lower().startswith("[service]")
            continue
        commented = line.startswith("#")
        if commented:
            body = line.lstrip("#").strip()
        else:
            body = line
        if "=" not in body:
            continue
        key, value, kind = _split_directive(body)
        if key is None:
            continue
        if commented:
            # A commented directive can appear anywhere, including the trailing
            # reference block after the [Service] body -- do not gate on
            # in_service, or the hub template's documented sync-daemon env
            # would be classified as never-mentioned.
            mentioned.add(key)
        elif in_service:
            declared[key] = (value, kind)
    # A key that is both declared and mentioned is declared.
    mentioned -= set(declared)
    return declared, mentioned


def _split_directive(body):
    """`Environment=K=V` -> (K, V, "env"); `Restart=x` -> (Restart, x, "prop").

    Handles the quoted form systemd allows in drop-ins
    (Environment="PHASE3_X=1"), which the live fleet uses and the templates
    partly use. Returns (None, None, None) for anything unparseable.
    """
    name, _, rest = body.partition("=")
    name = name.strip()
    rest = rest.strip()
    if not name or not name[0].isalpha():
        return None, None, None
    if name != "Environment":
        return name, rest, "prop"
    try:
        tokens = shlex.split(rest)
    except ValueError:
        tokens = [rest.strip('"')]
    if not tokens:
        return None, None, None
    env_key, _, env_value = tokens[0].partition("=")
    env_key = env_key.strip()
    if not env_key:
        return None, None, None
    return env_key, env_value.strip(), "env"


def parse_show(text):
    """Parse `systemctl show` output into {property: value}.

    Environment is exploded into its individual KEY=VALUE pairs, so the
    returned mapping is keyed exactly like parse_template's `declared`.
    """
    props = {}
    env = {}
    for line in (text or "").splitlines():
        if "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip()
        if name == "Environment":
            try:
                tokens = shlex.split(value)
            except ValueError:
                tokens = value.split()
            for token in tokens:
                k, _, v = token.partition("=")
                if k:
                    env[k.strip()] = v.strip()
        else:
            props[name] = value
    merged = dict(props)
    merged.update(env)
    # Keep the raw property view for base properties whose names could in
    # principle collide with an env key.
    merged["_props"] = props
    merged["_env"] = env
    return merged


def _run(argv, timeout):
    """Run a command, returning stdout or None on any failure. Never raises."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def fetch_show(host, properties, _run_ssh=None):
    """Read the EFFECTIVE merged unit config from one machine over ssh.

    Read-only: `systemctl show` is a property query. It never touches
    /home/ree/REE_Working on the hub -- dirtying that tree wedges every
    Phase-3 writer and takes the whole coordination-data plane down.
    """
    argv = ["ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=%d" % SSH_CONNECT_TIMEOUT_SECONDS,
            "ree@%s" % host,
            "systemctl show %s %s" % (
                UNIT, " ".join("-p %s" % p for p in properties))]
    if _run_ssh is not None:
        return _run_ssh(argv)
    return _run(argv, SSH_TIMEOUT_SECONDS)


def compare(declared, mentioned, live, expected_affinity=None):
    """Compare one machine's template against its effective live config.

    Returns a list of per-directive result dicts (including OK rows, so the
    caller can show what WAS verified -- a check that prints only failures is
    indistinguishable from a check that silently stopped running).
    """
    results = []
    props = live.get("_props", {})
    env = live.get("_env", {})

    for key, (want, kind) in sorted(declared.items()):
        if kind == "env":
            # `systemctl show -p Environment` enumerates the environment
            # exhaustively, so absence here is definitive.
            if key not in env:
                results.append(_row(
                    key, MISSING, want, None,
                    "declared in template, absent from live unit environment"))
                continue
            have = env[key]
        else:
            if key not in props:
                # Not exposed by `systemctl show` at all. Do NOT call that
                # MISSING: it may simply not be a queryable property name, and
                # a false MISSING against a reviewed template erodes the check.
                results.append(_row(key, UNKNOWN, want, None,
                                    "not exposed by systemctl show"))
                continue
            have = props[key]
            if have == "" and (want or "") != "":
                results.append(_row(
                    key, MISSING, want, None,
                    "declared in template, unset on the live unit"))
                continue

        if _is_placeholder(want) or key in PER_MACHINE_KEYS:
            results.append(_row(key, OK, want, have,
                                "present (per-machine value, not compared)"))
            continue

        if (have or "") == (want or ""):
            results.append(_row(key, OK, want, have, "matches template"))
        else:
            results.append(_row(key, DIFFERENT, want, have,
                                "live value differs from template"))

    # Live directives the template does not declare. Scoped to environment
    # variables: the merged property set is full of systemd defaults the
    # template was never meant to restate, so grading those would drown the
    # real signal.
    for key in sorted(env):
        if key in declared:
            continue
        have = env[key]
        if key in mentioned:
            results.append(_row(
                key, UNDECLARED, None, have,
                "live, but template carries it only as a commented/staged line"))
        else:
            results.append(_row(
                key, EXTRA, None, have,
                "live, but template does not mention it at all"))

    if expected_affinity:
        results.append(_check_affinity(props.get("ExecStart", ""),
                                       expected_affinity))
    return [r for r in results if r is not None]


def _check_affinity(exec_start, expected):
    """ExecStart must pin --machine <affinity>.

    Both templates state this in PROSE ("ExecStart must include: --machine
    ree-cloud-1") rather than as a directive, so the directive comparison above
    cannot see it -- but it is load-bearing: socket.gethostname() on the hub is
    `ree-worker-1`, so an unpinned hub runner reports under the wrong machine
    name and its claims, heartbeats and affinity routing all go to a machine
    identity that does not exist in the queue.
    """
    flag = "--machine %s" % expected
    if not exec_start:
        return _row("ExecStart:--machine", UNKNOWN, flag, None,
                    "ExecStart not readable")
    if flag in exec_start:
        return _row("ExecStart:--machine", OK, flag, expected,
                    "affinity pinned correctly")
    return _row("ExecStart:--machine", DIFFERENT, flag, "<not pinned>",
                "ExecStart does not pin the documented machine affinity")


def _row(key, verdict, want, have, detail):
    return {
        "directive": key,
        "verdict": verdict,
        "template": redact(key, want),
        "live": redact(key, have),
        "detail": detail,
    }


def check_machine(spec, deploy_dir=None, _run_ssh=None, _show_text=None):
    """Evaluate one machine. Never raises."""
    deploy_dir = deploy_dir or DEPLOY_DIR
    finding = {
        "hcloud": spec["hcloud"],
        "affinity": spec["affinity"],
        "template": spec["template"],
        "note": spec.get("note", ""),
        "status": UNKNOWN,
        "detail": "",
        "active_state": None,
        "drop_in_paths": [],
        "results": [],
    }

    template_path = os.path.join(deploy_dir, spec["template"])
    try:
        with open(template_path, "r", encoding="utf-8") as fh:
            declared, mentioned = parse_template(fh.read())
    except OSError as exc:
        finding["detail"] = "cannot read template %s (%s)" % (
            spec["template"], exc.__class__.__name__)
        return finding

    if not declared:
        finding["detail"] = "template %s declares no [Service] directives" % (
            spec["template"],)
        return finding

    # The queried property set is DERIVED FROM THE TEMPLATES, not hardcoded:
    # adding a directive to a template automatically extends what is checked.
    # A hardcoded list would silently stop covering the next Restart=always.
    non_env = sorted(k for k, (_v, kind) in declared.items() if kind == "prop")
    properties = tuple(BASE_PROPERTIES) + tuple(
        k for k in non_env if k not in BASE_PROPERTIES) + ("Environment",)

    show_text = _show_text
    if show_text is None:
        show_text = fetch_show(spec["host"], properties, _run_ssh=_run_ssh)
    if not show_text:
        finding["status"] = UNREACHABLE
        finding["detail"] = (
            "ssh/systemctl unavailable -- powered off is NORMAL for a worker; "
            "not powered on to audit (billable). `hcloud server list` is the "
            "authority on power state.")
        return finding

    live = parse_show(show_text)
    finding["active_state"] = live.get("_props", {}).get("ActiveState")
    finding["drop_in_paths"] = [
        p for p in (live.get("_props", {}).get("DropInPaths") or "").split()
        if p]
    finding["results"] = compare(
        declared, mentioned, live, expected_affinity=spec["affinity"])

    findings = [r for r in finding["results"]
                if r["verdict"] in FINDING_VERDICTS]
    notes = [r for r in finding["results"] if r["verdict"] in NOTE_VERDICTS]
    if findings:
        finding["status"] = findings[0]["verdict"] if len(
            set(r["verdict"] for r in findings)) == 1 else DIFFERENT
        finding["detail"] = "%d directive(s) diverge from %s" % (
            len(findings), spec["template"])
    elif notes:
        finding["status"] = OK
        finding["detail"] = (
            "matches %s; %d live directive(s) the template does not record"
            % (spec["template"], len(notes)))
    else:
        finding["status"] = OK
        finding["detail"] = "matches %s" % spec["template"]
    return finding


def check_all(machines=None, deploy_dir=None, _run_ssh=None):
    return [check_machine(spec, deploy_dir=deploy_dir, _run_ssh=_run_ssh)
            for spec in (machines if machines is not None else MACHINES)]


def count_findings(all_findings):
    n = 0
    for f in all_findings:
        n += len([r for r in f["results"] if r["verdict"] in FINDING_VERDICTS])
    return n


def count_notes(all_findings):
    n = 0
    for f in all_findings:
        n += len([r for r in f["results"] if r["verdict"] in NOTE_VERDICTS])
    return n


def render(all_findings, show_ok=False):
    """ASCII-only operator report. Always renders every machine, including
    the all-clear -- a report that appears only on failure cannot be told
    apart from one that silently stopped running."""
    lines = []
    lines.append("=" * 72)
    lines.append("FLEET systemd drop-in drift -- %s" % UNIT)
    lines.append("=" * 72)

    for f in all_findings:
        lines.append("")
        head = "%s (%s)  [%s]" % (f["hcloud"], f["affinity"], f["status"])
        lines.append(head)
        lines.append("-" * len(head))
        if f["note"]:
            lines.append("  note:     %s" % f["note"])
        lines.append("  template: %s" % f["template"])
        if f["active_state"]:
            lines.append("  unit:     ActiveState=%s" % f["active_state"])
        if f["drop_in_paths"]:
            lines.append("  drop-ins: %s" % " ".join(
                os.path.basename(p) for p in f["drop_in_paths"]))
        if f["detail"]:
            lines.append("  %s" % f["detail"])

        rows = [r for r in f["results"]
                if show_ok or r["verdict"] != OK]
        if not rows and f["results"]:
            lines.append("  (all %d declared directive(s) match)"
                         % len(f["results"]))
        for r in rows:
            lines.append("    %-11s %s" % (r["verdict"], r["directive"]))
            if r["verdict"] in (MISSING,):
                lines.append("                template: %s" % r["template"])
            elif r["verdict"] == DIFFERENT:
                lines.append("                template: %s" % r["template"])
                lines.append("                live:     %s" % r["live"])
            else:
                lines.append("                live:     %s" % r["live"])
            lines.append("                %s" % r["detail"])

    n_find = count_findings(all_findings)
    n_note = count_notes(all_findings)
    unreachable = [f for f in all_findings if f["status"] == UNREACHABLE]

    lines.append("")
    lines.append("-" * 72)
    lines.append("%d finding(s) [MISSING/DIFFERENT], %d note(s) "
                 "[EXTRA/UNDECLARED], %d unreachable"
                 % (n_find, n_note, len(unreachable)))
    if unreachable:
        lines.append("  unreachable is an UNKNOWN, not a finding: %s"
                     % ", ".join(f["hcloud"] for f in unreachable))
        lines.append("  (powered off is normal; `hcloud server list` is the "
                     "authority. Not woken -- that is billable.)")
    if n_find or n_note:
        lines.append("")
        lines.append("REMEDIATION -- this tool never edits a live unit.")
        lines.append("  Decide the direction FIRST: a divergence can mean the")
        lines.append("  machine is wrong OR the template is stale. EXTRA and")
        lines.append("  UNDECLARED usually mean the TEMPLATE needs updating.")
        lines.append("  To change a machine, on that box:")
        lines.append("    sudoedit /etc/systemd/system/%s.service.d/shadow.conf"
                     % UNIT)
        lines.append("    sudo systemctl daemon-reload")
        lines.append("    sudo systemctl restart %s   # drains first" % UNIT)
        lines.append("  On the HUB, never touch /home/ree/REE_Working -- "
                     "dirtying it")
        lines.append("  wedges every Phase-3 writer.")
    lines.append("")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare live ree-runner systemd drop-ins against the "
                    "tracked templates (detection only).")
    parser.add_argument("--machine", action="append", default=None,
                        help="limit to an affinity or hcloud name "
                             "(repeatable)")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    parser.add_argument("--show-ok", action="store_true",
                        help="also list directives that match")
    parser.add_argument("--exit-nonzero", action="store_true",
                        help="exit 1 if any finding (default exits 0 so it "
                             "chains safely)")
    parser.add_argument("--strict", action="store_true",
                        help="with --exit-nonzero, also gate on EXTRA/"
                             "UNDECLARED notes")
    parser.add_argument("--deploy-dir", default=None,
                        help="directory holding the shadow.conf.*.example "
                             "templates (default: this script's directory)")
    args = parser.parse_args(argv)

    machines = MACHINES
    if args.machine:
        wanted = set(args.machine)
        machines = [m for m in MACHINES
                    if m["affinity"] in wanted or m["hcloud"] in wanted]
        if not machines:
            sys.stderr.write("no machine matches %s\n"
                             % ", ".join(sorted(wanted)))
            return 2

    all_findings = check_all(machines=machines, deploy_dir=args.deploy_dir)

    if args.json:
        print(json.dumps({
            "unit": UNIT,
            "n_findings": count_findings(all_findings),
            "n_notes": count_notes(all_findings),
            "machines": all_findings,
        }, indent=2, sort_keys=True))
    else:
        for line in render(all_findings, show_ok=args.show_ok):
            print(line)

    if args.exit_nonzero:
        n = count_findings(all_findings)
        if args.strict:
            n += count_notes(all_findings)
        return 1 if n else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
