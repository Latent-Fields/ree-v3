"""Daemon code-drift check -- has a landed fix actually reached the process?

WHY THIS EXISTS
    Python binds a module at import. A long-running systemd daemon therefore
    executes the bytecode that existed when it started, forever, no matter
    what lands in the checkout underneath it. Nothing on the hub restarts
    these daemons when their source changes, and trunk-first means coordinator
    code lands continuously. So a fix can be committed, pushed, tested and
    present on disk while the live process keeps running the broken code --
    silently, with no error anywhere.

    Confirmed 2026-07-18: commit b2d2ef1 fixed a permanent self-deadlock in
    phase3_queue_writer (a staged experiment_queue.json wedged the queue
    snapshot for 5h31m, last_error=None, tick loop still running, while
    workers re-claimed already-completed work off the frozen git file). The
    fix landed origin/main at 16:31Z. ree-sync-daemon had been running since
    2026-07-09 18:28:38Z -- ~8.9 days BEFORE the fix existed. The live process
    ran pre-fix bytecode for ~3.4h and would have done so indefinitely; the
    original wedge was cleared by hand, not by the new guard.

    Audited at the same time: ree-coordinator (up since 2026-06-24) happened
    to be running current code, because nothing had touched its source since
    2026-06-18. Fine BY LUCK, not by design. This check is the "by design"
    part.

WHAT IT COMPARES
    For each daemon: the unit's ExecMainStartTimestamp against the committer
    date of the last commit touching that daemon's IMPORT SURFACE -- the
    modules bound into the long-running process, not everything the service
    can touch. The distinction matters: the runner launches experiment
    scripts and ree_core as fresh subprocesses on every run, so those pick up
    new code automatically and are NOT stale-bound. Only imported modules are.

    Two distinct gaps are reported, because "the fix did not reach the
    process" has two causes:
      DRIFT         -- the commit is in the local checkout, but the daemon
                       started before it. Fix: restart the unit.
      BEHIND-ORIGIN -- the commit is on origin but not in the local checkout.
                       Fix: pull, THEN restart. A daemon can be CURRENT
                       against a stale checkout and still be running old code,
                       so checking only DRIFT would give a false all-clear.

WHY IT LIVES HERE (and not in sync_daemon)
    ree-live-status is Type=oneshot on a 3-minute timer: a FRESH process every
    tick. It re-imports this module every run, so the drift checker can never
    itself go stale. A checker inside sync_daemon -- a long-running daemon --
    would be subject to the exact bug it is meant to detect, and would go
    silent precisely when sync_daemon is the thing running old code.

    It is READ-ONLY and never restarts anything. sync_daemon is the sole git
    writer for coordination data; an ill-timed automatic restart is its own
    hazard (in-flight tick, non-empty spool), so remediation stays a
    deliberate operator action with the pre-flight in OPERATOR_GUIDE.md.

    Reads only: `systemctl show` (read-only property query) and `git log` in
    the daemons' checkouts. It never writes to those trees -- the Phase-3
    writers refuse to commit on a dirty tree, which is the recurring hub-wedge
    outage.

All printed output (stdout/stderr) is ASCII-only.

  python3 daemon_code_drift.py            # operator check; exit 1 if drifted
  python3 daemon_code_drift.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
REE_V3 = os.path.join(HOME, "REE_Working", "ree-v3")
REE_ASSEMBLY = os.path.join(HOME, "REE_Working", "REE_assembly")

# A remote-tracking ref older than this cannot support a BEHIND-ORIGIN
# verdict, so the check says so instead of implying an all-clear.
STALE_REMOTE_REF_DAYS = 7
GIT_TIMEOUT_SECONDS = 20.0
SYSTEMCTL_TIMEOUT_SECONDS = 10.0

# Status constants.
CURRENT = "CURRENT"
DRIFT = "DRIFT"
BEHIND_ORIGIN = "BEHIND-ORIGIN"
UNKNOWN = "UNKNOWN"
INACTIVE = "INACTIVE"

# Each daemon's IMPORT SURFACE -- the modules bound into the long-running
# process. NOT everything the service can execute: the runner shells out to
# experiment scripts and ree_core per run, so those are never stale-bound and
# must stay out of this list or every experiment commit would raise a false
# DRIFT on the runner.
UNITS = [
    {
        "unit": "ree-sync-daemon",
        "repo": REE_V3,
        "branch": "main",
        "paths": [
            "coordinator/sync_daemon.py",
            "coordinator/db.py",
            "coordinator/manifest_spool.py",
        ],
        "note": "sole git writer for coordination data",
    },
    {
        "unit": "ree-coordinator",
        "repo": REE_V3,
        "branch": "main",
        "paths": [
            "coordinator/app.py",
            "coordinator/db.py",
            "coordinator/schema.sql",
            "coordinator/manifest_spool.py",
        ],
        "note": "claim arbitration + result spool",
    },
    {
        "unit": "ree-runner",
        "repo": REE_V3,
        "branch": "main",
        "paths": [
            "experiment_runner.py",
            "coordinator_client.py",
            "runner_remote_control.py",
            "runner_checkpoint.py",
        ],
        "note": "experiment scripts run as subprocesses (not stale-bound)",
    },
    {
        "unit": "ree-explorer",
        "repo": REE_ASSEMBLY,
        "branch": "master",
        "paths": ["serve.py"],
        "note": "claims explorer",
    },
]


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


def resolve_repo(unit, fallback, _wd=None, _run_git=None):
    """Resolve the repo a unit ACTUALLY runs from, via its systemd
    WorkingDirectory -> `git rev-parse --show-toplevel`.

    Do not replace this with a hardcoded path. The hub runs the same code out
    of THREE different checkouts, and one of them is not where you would
    guess (verified 2026-07-18):

        ree-sync-daemon / ree-coordinator  ~/REE_Working/ree-v3
        ree-runner                         ~/REE_Working_runner/ree-v3
        ree-explorer                       ~/REE_Working/REE_assembly

    ree-explorer ran from ~/Documents/GitHub/REE_Working/REE_assembly (the path
    REE_Working/CLAUDE.md marks stale) until it was repointed at the canonical
    checkout on 2026-07-18. That it could drift there at all is the reason this
    resolves dynamically rather than by table: the wrong-tree case is not
    hypothetical, and a hardcoded map would have hidden it for six weeks.

    The first cut of this checker hardcoded ~/REE_Working/* and so evaluated
    two of the four daemons against trees their processes never read -- it
    reported ree-runner CURRENT against the wrong repo entirely. A drift check
    that silently grades the wrong tree is worse than no check, because it
    reports a confident all-clear. Returns (repo_path, how_resolved).
    """
    wd = _wd
    if wd is None:
        out = _run(["systemctl", "show", unit, "-p", "WorkingDirectory",
                    "--value"], SYSTEMCTL_TIMEOUT_SECONDS)
        wd = (out or "").strip()
    if wd:
        runner = _run_git or (lambda argv: _run(argv, GIT_TIMEOUT_SECONDS))
        top = runner(["git", "-C", wd, "rev-parse", "--show-toplevel"])
        if top and top.strip():
            return top.strip(), "systemd WorkingDirectory"
    return fallback, "fallback default"


def _boot_epoch(proc_stat_path="/proc/stat"):
    """Epoch seconds at boot, from /proc/stat btime. None if unavailable."""
    try:
        with open(proc_stat_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("btime "):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def unit_start_epoch(unit, boot_epoch=None, _show=None):
    """Return (epoch_seconds|None, active_state|None, display|None).

    Derived from ExecMainStartTimestampMonotonic (microseconds since boot)
    plus /proc/stat btime -- both plain integers, so this is immune to the
    locale- and timezone-abbreviation parsing that makes the human-readable
    ExecMainStartTimestamp unsafe to parse ("Wed 2026-06-24 05:49:51 UTC"
    becomes BST/CEST/... on a non-UTC host, and %Z round-trips unreliably).
    The human string is carried through for display only.
    """
    show = _show
    if show is None:
        show = _run(
            ["systemctl", "show", unit,
             "-p", "ExecMainStartTimestampMonotonic",
             "-p", "ExecMainStartTimestamp",
             "-p", "ActiveState"],
            SYSTEMCTL_TIMEOUT_SECONDS)
    if not show:
        return None, None, None

    props = {}
    for line in show.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            props[k.strip()] = v.strip()

    active_state = props.get("ActiveState") or None
    display = props.get("ExecMainStartTimestamp") or None

    if boot_epoch is None:
        boot_epoch = _boot_epoch()
    mono = props.get("ExecMainStartTimestampMonotonic")
    try:
        mono_us = int(mono)
    except (TypeError, ValueError):
        mono_us = 0
    if boot_epoch is None or mono_us <= 0:
        return None, active_state, display
    return boot_epoch + int(mono_us // 1000000), active_state, display


def last_source_commit(repo, paths, rev="HEAD", _run_git=None):
    """Return (sha, epoch, subject) for the last commit on `rev` touching any
    of `paths`, or (None, None, None). Read-only; never touches the tree."""
    runner = _run_git or (lambda argv: _run(argv, GIT_TIMEOUT_SECONDS))
    out = runner(
        ["git", "-C", repo, "log", "-1", "--format=%H%x00%ct%x00%s",
         rev, "--", *paths])
    if not out or not out.strip():
        return None, None, None
    parts = out.strip().split("\x00")
    if len(parts) < 3:
        return None, None, None
    try:
        return parts[0], int(parts[1]), parts[2]
    except ValueError:
        return None, None, None


def check_unit(spec, boot_epoch=None, _show=None, _run_git=None):
    """Evaluate one daemon. Returns a finding dict; never raises."""
    unit = spec["unit"]
    repo, repo_source = resolve_repo(
        unit, spec["repo"], _wd=spec.get("_wd"), _run_git=_run_git)
    finding = {
        "unit": unit,
        "repo": os.path.basename(repo),
        "repo_path": repo,
        "repo_resolved_by": repo_source,
        "note": spec.get("note", ""),
        "status": UNKNOWN,
        "detail": "",
        "started_utc": None,
        "commit": None,
        "commit_utc": None,
        "commit_subject": None,
        "behind_origin": None,
    }

    started, active_state, _display = unit_start_epoch(
        unit, boot_epoch=boot_epoch, _show=_show)
    if active_state and active_state != "active":
        finding["status"] = INACTIVE
        finding["detail"] = "unit ActiveState=%s" % active_state
        return finding
    if started is None:
        finding["status"] = UNKNOWN
        finding["detail"] = "no start timestamp (unit absent or not running?)"
        return finding
    finding["started_utc"] = _iso(started)

    sha, commit_epoch, subject = last_source_commit(
        repo, spec["paths"], _run_git=_run_git)
    if sha is None:
        finding["status"] = UNKNOWN
        finding["detail"] = "no commit found for source paths in %s" % (
            finding["repo"],)
        return finding
    finding["commit"] = sha[:10]
    finding["commit_utc"] = _iso(commit_epoch)
    finding["commit_subject"] = subject

    # BEHIND-ORIGIN is checked FIRST and reported in preference to CURRENT:
    # a daemon can be newer than everything in a stale checkout and still be
    # running old code. Reads the existing remote-tracking ref only -- no
    # fetch, no network (sync_daemon refreshes it constantly anyway).
    origin_ref = "refs/remotes/origin/%s" % spec.get("branch", "main")
    o_sha, o_epoch, o_subject = last_source_commit(
        repo, spec["paths"], rev=origin_ref, _run_git=_run_git)
    if o_sha and o_sha != sha and o_epoch and commit_epoch and (
            o_epoch > commit_epoch):
        finding["behind_origin"] = {
            "commit": o_sha[:10],
            "commit_utc": _iso(o_epoch),
            "commit_subject": o_subject,
        }
        finding["status"] = BEHIND_ORIGIN
        finding["detail"] = (
            "origin has %s (%s) touching this daemon's source, not in the "
            "local checkout -- pull, THEN restart" % (o_sha[:10], _iso(o_epoch)))
        return finding

    age = remote_ref_age_days(
        repo, origin_ref, int(datetime.now(timezone.utc).timestamp()),
        _run_git=_run_git)
    if age is not None and age > STALE_REMOTE_REF_DAYS:
        finding["remote_ref_stale_days"] = round(age, 1)

    if commit_epoch > started:
        finding["status"] = DRIFT
        finding["detail"] = (
            "started %s, but %s (%s) touched its source %s later -- process "
            "is running pre-commit bytecode" % (
                _iso(started), sha[:10], _iso(commit_epoch),
                _age_str(commit_epoch - started)))
        return finding

    finding["status"] = CURRENT
    finding["detail"] = "started %s, after last source commit %s (%s)" % (
        _iso(started), sha[:10], _iso(commit_epoch))
    return finding


def remote_ref_age_days(repo, origin_ref, now_epoch, _run_git=None):
    """Days since the remote-tracking ref's tip was committed, or None.

    Why this matters: BEHIND-ORIGIN is computed from the LOCAL remote-tracking
    ref, deliberately without fetching (a fetch is network I/O on every tick
    and races the writers). If nothing has fetched that repo in weeks, the ref
    is frozen and BEHIND-ORIGIN silently reports "0 behind" -- a false
    all-clear at the pull layer rather than the process layer.

    Confirmed 2026-07-18: ree-explorer ran from a separate clone at
    ~/Documents/GitHub/REE_Working/REE_assembly whose origin/master ref was
    last updated 2026-06-07. Its serve.py sat at 241835246a while the live
    REE_assembly had moved to 756a4d0a31 -- so the daemon graded CURRENT while
    running ~6-week-old code. The verdict was RIGHT about the process and
    useless about the deployment. This annotation named that gap instead of
    hiding it, and is what prompted the repoint to the canonical checkout
    later the same day; the annotation now clears for ree-explorer.

    Keep this check. The root cause was that nothing fetched that clone -- its
    own auto-pull had been failing for weeks against a dirty tree -- and the
    staleness was invisible everywhere except this annotation.
    """
    runner = _run_git or (lambda argv: _run(argv, GIT_TIMEOUT_SECONDS))
    out = runner(["git", "-C", repo, "log", "-1", "--format=%ct", origin_ref])
    if not out or not out.strip():
        return None
    try:
        return (now_epoch - int(out.strip().splitlines()[0])) / 86400.0
    except (ValueError, IndexError):
        return None


def _iso(epoch):
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _short_repo(path):
    """~-relative repo path. The basename alone is NOT enough: two units run
    from different checkouts that share the basename `ree-v3`
    (REE_Working/ree-v3 vs REE_Working_runner/ree-v3)."""
    if not path:
        return "--"
    if path.startswith(HOME):
        return "~" + path[len(HOME):]
    return path


def _age_str(seconds):
    seconds = max(0, int(seconds))
    if seconds < 5400:
        return "%dm" % (seconds // 60)
    if seconds < 172800:
        return "%.1fh" % (seconds / 3600.0)
    return "%.1fd" % (seconds / 86400.0)


def check_all(units=None, boot_epoch=None):
    """Evaluate every daemon.

    Returns [] on a non-systemd host (no /proc/stat btime -- e.g. the Mac,
    where the repos are present but the units are not). That is an honest
    "not applicable here" rather than a wall of UNKNOWN that an operator
    would have to squint at to distinguish from a real fault.
    """
    findings = []
    if boot_epoch is None:
        boot_epoch = _boot_epoch()
    if boot_epoch is None:
        return findings
    for spec in (units if units is not None else UNITS):
        f = check_unit(spec, boot_epoch=boot_epoch)
        if not os.path.isdir(os.path.join(f["repo_path"], ".git")):
            continue
        findings.append(f)
    return findings


def worst_status(findings):
    for status in (BEHIND_ORIGIN, DRIFT, UNKNOWN, INACTIVE):
        if any(f["status"] == status for f in findings):
            return status
    return CURRENT


def render_markdown(findings):
    """Compact ASCII markdown block for FLEET_STATUS.md.

    Always renders, including the all-clear -- a section that appears only on
    failure is indistinguishable from a section that silently stopped running,
    which is the same class of bug this check exists to catch.
    """
    lines = []
    if not findings:
        lines.append("## Daemon code freshness")
        lines.append("")
        lines.append("_(not checked -- not a systemd host)_")
        lines.append("")
        return lines

    stale = [f for f in findings
             if f["status"] in (DRIFT, BEHIND_ORIGIN)]
    if stale:
        lines.append("## Daemon code freshness -- %d STALE" % len(stale))
    else:
        lines.append("## Daemon code freshness -- all current")
    lines.append("")
    # The repo column is not decoration: a CURRENT verdict computed against a
    # tree the process does not read is the silent failure this check exists
    # to prevent, so the tree that was graded is always shown.
    lines.append("| Daemon | Status | Repo (graded) | Started | "
                 "Last source commit |")
    lines.append("|---|---|---|---|---|")
    for f in sorted(findings, key=lambda x: x["unit"]):
        lines.append("| %s | %s | %s | %s | %s |" % (
            f["unit"],
            f["status"],
            _short_repo(f.get("repo_path")),
            f["started_utc"] or "--",
            ("%s %s" % (f["commit"] or "--", f["commit_utc"] or ""))
            .strip(),
        ))
    lines.append("")
    unfetched = [f for f in findings if f.get("remote_ref_stale_days")]
    if unfetched:
        lines.append("**Cannot vouch for these against origin.** Their "
                     "remote-tracking ref has not been fetched recently, so "
                     "BEHIND-ORIGIN is computed from a frozen ref and a "
                     "`CURRENT` verdict here means \"current with a stale "
                     "checkout\", not \"running the latest code\":")
        lines.append("")
        for f in unfetched:
            lines.append("- `%s` (%s) -- origin ref last updated %s days ago"
                         % (f["unit"], _short_repo(f.get("repo_path")),
                            f["remote_ref_stale_days"]))
        lines.append("")

    if stale:
        lines.append("**A landed fix is not reaching a running process.** "
                     "Python binds modules at import, so these daemons keep "
                     "executing the code that existed when they started.")
        lines.append("")
        for f in stale:
            lines.append("- `%s` (%s) -- %s" % (
                f["unit"], _short_repo(f.get("repo_path")), f["detail"]))
        lines.append("")
        lines.append("Restart is a deliberate operator action with a "
                     "pre-flight (clean trees + empty spool) -- see "
                     "`ree-v3/coordinator/OPERATOR_GUIDE.md`, "
                     "\"Daemon code drift\".")
        lines.append("")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check whether each REE daemon is running current code.")
    parser.add_argument("--json", action="store_true",
                        help="emit findings as JSON")
    args = parser.parse_args(argv)

    findings = check_all()
    if args.json:
        sys.stdout.write(json.dumps(findings, indent=2) + "\n")
    else:
        for line in render_markdown(findings):
            sys.stdout.write(line + "\n")
    return 1 if worst_status(findings) in (DRIFT, BEHIND_ORIGIN) else 0


if __name__ == "__main__":
    sys.exit(main())
