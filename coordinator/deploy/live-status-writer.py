"""Live-status writer -- force-update the REE_assembly `live-status` branch
with a fresh fleet-progress snapshot every tick.

WHY THIS EXISTS
    Experiment progress (current EXQ, episodes-done, overall_pct, ETA) used
    to reach github.com only via the sync_daemon's 30-minute forced heartbeat
    "liveness tick" commit -- which was the dominant source of REE_assembly
    git-history bloat (every tick is a permanent commit). That liveness tick
    was retired 2026-06-23. This writer restores the github.com-readable
    progress view WITHOUT the permanent-history cost: it publishes the
    snapshot to a dedicated `live-status` branch that is FORCE-RESET to a
    fresh ROOT (parentless) commit on every tick. Each push abandons the
    previous commit, which then dangles and is reclaimed by gc on both the
    local mirror and GitHub -- so the branch always shows the current state
    and the permanent history never grows.

    Read it at:
        https://github.com/Latent-Fields/REE_assembly/blob/live-status/FLEET_STATUS.md

ISOLATION
    Operates ENTIRELY inside a dedicated local repo dir (REPO_DIR), never the
    sync_daemon's ~/REE_Working/REE_assembly working tree. The Phase-3 writers
    refuse to commit on a dirty tree (the recurring hub-wedge outage), so this
    writer must never touch their tree. It only reads the queue file
    read-only and the coordinator HTTP plane.

TRANSPORT
    Freshness/state/progress come from the coordinator /shadow/status (the
    DB-authoritative source, sub-second fresh) over WireGuard, using the same
    COORDINATOR_URL + COORDINATOR_SCALER_TOKEN env the cloud-scaler already
    has. The queue file is read read-only for the pending/claimed summary.

All printed output (stdout/stderr) is ASCII-only.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import daemon_code_drift
except ImportError:  # pragma: no cover -- checker absent is not fatal
    daemon_code_drift = None


DEFAULT_COORDINATOR_URL = "http://10.8.0.1:8787"
DEFAULT_QUEUE_PATH = "/home/ree/REE_Working/ree-v3/experiment_queue.json"
# Dedicated, isolated repo dir -- NOT the sync_daemon's REE_assembly tree.
DEFAULT_REPO_DIR = "/home/ree/live-status-repo"
DEFAULT_REMOTE_URL = "https://github.com/Latent-Fields/REE_assembly.git"
DEFAULT_BRANCH = "live-status"
FETCH_TIMEOUT_SECONDS = 6.0


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sys.stdout.write("[%s] %s\n" % (ts, msg))
    sys.stdout.flush()


def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def fetch_coordinator_status(url, token, timeout=FETCH_TIMEOUT_SECONDS):
    """Return the parsed /shadow/status doc, or None on any failure. Never
    raises -- a coordinator blip should publish a 'coordinator unreachable'
    snapshot, not crash the timer."""
    url = (url or "").rstrip("/")
    if not url or not token:
        return None
    try:
        req = urllib.request.Request(
            url + "/shadow/status",
            headers={"Authorization": "Bearer " + token},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError,
            json.JSONDecodeError, ValueError) as exc:
        log("WARN coordinator /shadow/status fetch failed (%r)" % exc)
        return None


def load_queue(path):
    """Read-only queue load. Returns the items list or [] on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("items", []) or []
    except Exception as exc:  # noqa: BLE001
        log("WARN queue read failed at %s (%r)" % (path, exc))
        return []


def _age_str(last_seen_iso, now):
    t = parse_utc(last_seen_iso)
    if t is None:
        return "?"
    secs = (now - t).total_seconds()
    if secs < 0:
        return "0s"
    if secs < 90:
        return "%ds" % int(secs)
    if secs < 5400:
        return "%dm" % int(secs // 60)
    return "%.1fh" % (secs / 3600.0)


def _progress_cell(m):
    """Render the per-machine progress block into one compact ASCII cell."""
    state = (m.get("state") or "?")
    exq = m.get("current_exq")
    if not exq:
        return "--"
    p = m.get("progress") or {}
    bits = []
    label = p.get("run_label")
    if label:
        bits.append(str(label))
    ed, et = p.get("episodes_done"), p.get("episodes_total")
    if ed is not None and et:
        bits.append("ep %s/%s" % (ed, et))
    pct = p.get("overall_pct")
    if pct is not None:
        try:
            bits.append("%.1f%%" % float(pct))
        except (TypeError, ValueError):
            pass
    return " - ".join(bits) if bits else "(running, no progress field)"


def _eta_cell(m):
    sr = m.get("seconds_remaining")
    try:
        sr = float(sr)
    except (TypeError, ValueError):
        return "--"
    if sr <= 0:
        return "--"
    if sr < 5400:
        return "~%dm" % int(sr // 60)
    return "~%.1fh" % (sr / 3600.0)


def collect_drift():
    """Return (markdown_lines, findings). Never raises -- a fault in the
    drift checker must degrade to a visible 'unavailable' note, never take
    down the fleet snapshot the operator relies on."""
    if daemon_code_drift is None:
        return (["## Daemon code freshness", "",
                 "_(checker module not importable on this host)_", ""], [])
    try:
        findings = daemon_code_drift.check_all()
        return daemon_code_drift.render_markdown(findings), findings
    except Exception as exc:  # noqa: BLE001
        log("WARN daemon code-drift check failed (%r)" % exc)
        return (["## Daemon code freshness", "",
                 "_(check failed this tick: %s)_" % type(exc).__name__,
                 ""], [])


def build_markdown(status_doc, queue_items, now, drift_lines=None):
    """Build the phone-readable FLEET_STATUS.md content (ASCII)."""
    lines = []
    lines.append("# REE Fleet -- Live Status")
    lines.append("")
    lines.append("**Updated:** %s &middot; refreshes every few minutes "
                 "&middot; source: coordinator DB (live)." % utcnow_iso())
    lines.append("")
    lines.append("> This is the force-updated `live-status` branch -- a "
                 "current snapshot, **not** part of `master` history. The "
                 "branch is reset every tick so it never bloats the repo.")
    lines.append("")

    if status_doc is None:
        lines.append("## Coordinator unreachable this tick")
        lines.append("")
        lines.append("The coordinator /shadow/status could not be reached "
                     "when this snapshot was written (%s). The fleet may "
                     "still be running; this branch will refresh on the next "
                     "successful tick." % utcnow_iso())
        # Drift is independent of coordinator reachability -- and a wedged
        # coordinator is exactly when "is it running current code?" matters.
        lines.append("")
        lines.extend(drift_lines or [])
        return "\n".join(lines) + "\n"

    machines = status_doc.get("machines") or []
    running = [m for m in machines if m.get("current_exq")]
    lines.append("## Workers (%d total, %d running)"
                 % (len(machines), len(running)))
    lines.append("")
    lines.append("| Machine | State | Experiment | Progress | ETA | "
                 "Last seen |")
    lines.append("|---|---|---|---|---|---|")
    for m in sorted(machines, key=lambda x: x.get("machine") or ""):
        lines.append("| %s | %s | %s | %s | %s | %s ago |" % (
            m.get("machine") or "?",
            m.get("state") or "?",
            m.get("current_exq") or "--",
            _progress_cell(m),
            _eta_cell(m),
            _age_str(m.get("last_seen"), now),
        ))
    lines.append("")

    pending = [i for i in queue_items if i.get("status") == "pending"]
    claimed = [i for i in queue_items if i.get("status") == "claimed"]
    lines.append("## Queue -- %d pending, %d claimed"
                 % (len(pending), len(claimed)))
    lines.append("")
    if pending or claimed:
        lines.append("| Queue ID | Status | Claimed by | Priority |")
        lines.append("|---|---|---|---|")
        for i in claimed + pending:
            cb = i.get("claimed_by") or {}
            machine = cb.get("machine") if isinstance(cb, dict) else cb
            lines.append("| %s | %s | %s | %s |" % (
                i.get("queue_id") or "?",
                i.get("status") or "?",
                machine or "--",
                i.get("priority", "--"),
            ))
    else:
        lines.append("_(queue empty)_")
    lines.append("")
    lines.extend(drift_lines or [])
    return "\n".join(lines) + "\n"


def _git(repo_dir, *args, check=True, capture=False):
    return subprocess.run(
        ["git", "-C", repo_dir, *args],
        check=check,
        capture_output=capture,
        text=True,
        timeout=90,
    )


def ensure_repo(repo_dir, remote_url):
    """Initialise the dedicated, isolated repo dir once (idempotent)."""
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        os.makedirs(repo_dir, exist_ok=True)
        _git(repo_dir, "init", "-q")
        _git(repo_dir, "config", "user.email", "coordinator@ree.local")
        _git(repo_dir, "config", "user.name", "ree-live-status")
        # Aggressive prune so abandoned root commits do not accumulate.
        _git(repo_dir, "config", "gc.pruneExpire", "now")
    # Remote (idempotent set).
    try:
        _git(repo_dir, "remote", "set-url", "origin", remote_url,
             capture=True)
    except subprocess.CalledProcessError:
        _git(repo_dir, "remote", "add", "origin", remote_url)


def publish(repo_dir, branch, files, dry_run=False):
    """Stage `files` (name -> content), build a PARENTLESS commit, point the
    branch at it, and force-push. No parent chain -> the previous commit
    dangles and is reclaimed by gc; permanent history never grows."""
    for name, content in files.items():
        with open(os.path.join(repo_dir, name), "w", encoding="utf-8") as fh:
            fh.write(content)
    _git(repo_dir, "add", "--", *files.keys())
    tree = _git(repo_dir, "write-tree", capture=True).stdout.strip()
    # commit-tree with NO -p => a fresh root (orphan) commit every tick.
    msg = "live-status snapshot %s" % utcnow_iso()
    commit = _git(repo_dir, "commit-tree", tree, "-m", msg,
                  capture=True).stdout.strip()
    _git(repo_dir, "update-ref", "refs/heads/%s" % branch, commit)
    if dry_run:
        log("[DRY] would force-push %s -> origin/%s (commit %s)"
            % (branch, branch, commit[:10]))
        return
    _git(repo_dir, "push", "-q", "-f", "origin",
         "refs/heads/%s:refs/heads/%s" % (branch, branch))
    # Keep the dedicated dir tidy; abandoned roots are unreachable.
    _git(repo_dir, "gc", "--auto", "-q", check=False)
    log("published %s snapshot to origin/%s (commit %s)"
        % (branch, branch, commit[:10]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Force-update the REE_assembly live-status branch.")
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    parser.add_argument("--remote-url", default=DEFAULT_REMOTE_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--queue", default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="build the snapshot and log, but do not push")
    args = parser.parse_args(argv)

    coordinator_url = (os.environ.get("COORDINATOR_URL")
                       or DEFAULT_COORDINATOR_URL)
    coordinator_token = (os.environ.get("COORDINATOR_SCALER_TOKEN")
                         or os.environ.get("COORDINATOR_LOCAL_TOKEN") or "")

    now = datetime.now(timezone.utc)
    status_doc = fetch_coordinator_status(coordinator_url, coordinator_token)
    queue_items = load_queue(args.queue)
    drift_lines, drift_findings = collect_drift()
    md = build_markdown(status_doc, queue_items, now, drift_lines=drift_lines)
    stale = [f for f in drift_findings
             if f.get("status") in ("DRIFT", "BEHIND-ORIGIN")]
    if stale:
        log("WARN %d daemon(s) running stale code: %s"
            % (len(stale), ", ".join(f["unit"] for f in stale)))
    snapshot = {
        "updated_utc": utcnow_iso(),
        "coordinator_reachable": status_doc is not None,
        "daemon_code_drift": drift_findings,
        "machines": (status_doc or {}).get("machines", []),
        "queue_pending": sum(
            1 for i in queue_items if i.get("status") == "pending"),
        "queue_claimed": sum(
            1 for i in queue_items if i.get("status") == "claimed"),
    }
    files = {
        "FLEET_STATUS.md": md,
        "fleet_status.json": json.dumps(snapshot, indent=2) + "\n",
        "README.md": (
            "# REE_assembly `live-status` branch\n\n"
            "Force-updated fleet snapshot. See "
            "[FLEET_STATUS.md](FLEET_STATUS.md). This branch is reset every "
            "tick and is not part of `master` history.\n"),
    }

    ensure_repo(args.repo_dir, args.remote_url)
    publish(args.repo_dir, args.branch, files, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
