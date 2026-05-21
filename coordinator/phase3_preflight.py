"""Phase 3 pre-cutover precondition checks.

Run from the Mac (or any host with coordinator.env + optional SSH to hub).

  /opt/local/bin/python3 phase3_preflight.py
  /opt/local/bin/python3 phase3_preflight.py --dry-run
  /opt/local/bin/python3 phase3_preflight.py --json

Exit codes:
  0  all required checks PASS (or SKIP where noted)
  1  one or more FAIL
  2  configuration error (missing coordinator.env / URL)

All stdout/stderr text is ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_BASE = Path.home() / "REE_Working"
DEFAULT_ENV = DEFAULT_BASE / "REE_assembly" / "coordinator.env"
SCHEMA_PATH = HERE / "schema.sql"

CLOUD_HOSTS = ("ree-cloud-1", "ree-cloud-2", "ree-cloud-3", "ree-cloud-4")
# Mac runner is often serve-managed; cloud systemd drop-ins are the gate.
FLEET_SSH_HOSTS = CLOUD_HOSTS
HUB_WG = "10.8.0.1"
DEFAULT_SSH_HOSTS = {
    "ree-cloud-1": "91.98.130.117",
    "ree-cloud-2": "116.203.216.181",
    "ree-cloud-3": "46.62.170.133",
    "ree-cloud-4": "91.99.68.94",
}
HUB_REE_ASSEMBLY = "/home/ree/REE_Working/REE_assembly"
HUB_COORDINATOR_DB = (
    "/home/ree/REE_Working/ree-v3/coordinator/coordinator.db"
)
REGENERABLE_PREFIXES = (
    "evidence/experiments/runner_heartbeats/",
    "evidence/experiments/runner_status/",
    "evidence/experiments/runner_commands/",
)


@dataclass
class CheckResult:
    check_id: str
    category: str
    status: str  # PASS | FAIL | SKIP | WARN
    message: str
    detail: dict = field(default_factory=dict)


def _load_env_file(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        cfg[k.strip()] = v.strip()
    return cfg


def _ssh_targets(cfg: dict[str, str]) -> dict[str, str]:
    out = dict(DEFAULT_SSH_HOSTS)
    for h in CLOUD_HOSTS:
        key = "SHADOW_SSH_HOST_" + h
        if cfg.get(key):
            out[h] = cfg[key]
    return out


def _ssh_run(host: str, user: str, remote_cmd: str, *, dry_run: bool,
             timeout: int = 20) -> tuple[bool, str, str]:
    if dry_run:
        return True, "", "dry-run skip"
    target = "%s@%s" % (user, host)
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=accept-new",
        target,
        remote_cmd,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, "", "ssh timed out"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, "", err or "ssh exit %d" % proc.returncode
    return True, (proc.stdout or "").strip(), ""


def _http_get(url: str, token: str | None, path: str,
              timeout: float = 8.0) -> tuple[int | None, dict | None, str]:
    req = urllib.request.Request(
        url.rstrip("/") + path,
        headers={"Authorization": "Bearer " + token} if token else {},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), ""
    except urllib.error.HTTPError as exc:
        return exc.code, None, "HTTP %s" % exc.code
    except (urllib.error.URLError, OSError) as exc:
        return None, None, repr(exc)


def _run_check_shadow(url: str, token: str) -> tuple[int, str]:
    script = HERE / "check_shadow.py"
    if not script.exists():
        return 2, "check_shadow.py missing"
    cmd = [sys.executable, str(script), "--url", url, "--token", token]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        return 2, "check_shadow timeout"
    tail = (proc.stdout or "").strip().splitlines()
    summary = tail[-1] if tail else "no output"
    return proc.returncode, summary


def _local_schema_ok() -> tuple[bool, str]:
    if not SCHEMA_PATH.exists():
        return False, "schema.sql missing"
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    required = ("experiments", "results", "heartbeats", "commands", "claim_log")
    missing = [t for t in required if t not in text]
    if missing:
        return False, "schema missing tables: " + ", ".join(missing)
    return True, "schema tables present"


def _import_sync_daemon_ready() -> tuple[bool, str]:
    try:
        import sync_daemon  # noqa: WPS433 -- intentional local import
        ready = bool(getattr(sync_daemon, "PHASE3_GIT_WRITER_READY", False))
        if ready:
            return False, "PHASE3_GIT_WRITER_READY is True (unsafe)"
        return True, "PHASE3_GIT_WRITER_READY is False (stub safe)"
    except Exception as exc:  # noqa: BLE001
        return False, "sync_daemon import failed: %r" % exc


def run_preflight(
    *,
    env_file: Path | None = None,
    dry_run: bool = False,
    mock: bool = False,
    quiet: bool = False,
) -> dict:
    """Run all pre-cutover checks. Returns a serialisable summary dict."""
    env_path = env_file or Path(
        os.environ.get("COORDINATOR_ENV_FILE", str(DEFAULT_ENV)))
    cfg = _load_env_file(env_path)
    url = cfg.get("COORDINATOR_URL") or os.environ.get("COORDINATOR_URL", "")
    token = (cfg.get("COORDINATOR_LOCAL_TOKEN")
             or os.environ.get("COORDINATOR_TOKEN", ""))
    ssh_user = cfg.get("COORDINATOR_SSH_USER", "ree")
    ssh_hosts = _ssh_targets(cfg)
    hub_ssh = ssh_hosts.get("ree-cloud-1", DEFAULT_SSH_HOSTS["ree-cloud-1"])

    checks: list[CheckResult] = []

    def add(cid: str, cat: str, status: str, msg: str, **detail):
        checks.append(CheckResult(cid, cat, status, msg, detail=detail))

    if not url:
        add("config", "reachability", "FAIL",
            "COORDINATOR_URL not set (coordinator.env)")
        return _summary(checks, quiet=quiet)

    # --- reachability / hub ---
    if mock:
        add("coordinator_api", "reachability", "SKIP",
            "mock: skip HTTP checks")
        add("hub_health", "hub", "SKIP", "mock: skip hub health")
    else:
        st, body, err = _http_get(url, None, "/health")
        if st == 200 and body and body.get("ok"):
            mode = body.get("mode", "?")
            if mode == "coordinator":
                add("hub_health", "hub", "PASS",
                    "health ok mode=coordinator")
            else:
                add("hub_health", "hub", "FAIL",
                    "health ok but mode=%r (expect coordinator)" % mode)
        else:
            add("hub_health", "hub", "FAIL",
                "health unreachable: %s" % (err or st))

        st, body, err = _http_get(url, token, "/shadow/status")
        if st == 200 and body:
            add("coordinator_api", "reachability", "PASS",
                "shadow/status ok (%d machines)" % len(
                    body.get("machines") or []))
        else:
            add("coordinator_api", "reachability", "FAIL",
                "shadow/status failed: %s" % (err or st))

    # --- soak (Phase 2 metrics) ---
    if mock:
        add("phase2_shadow_metrics", "soak", "SKIP",
            "mock: skip check_shadow")
    elif not token:
        add("phase2_shadow_metrics", "soak", "SKIP",
            "no COORDINATOR_LOCAL_TOKEN for check_shadow")
    else:
        code, summary = _run_check_shadow(url, token)
        if code == 0:
            add("phase2_shadow_metrics", "soak", "PASS",
                "check_shadow exit 0: %s" % summary)
        elif code == 2:
            add("phase2_shadow_metrics", "soak", "WARN",
                "check_shadow NO_SIGNAL: %s" % summary)
        else:
            add("phase2_shadow_metrics", "soak", "FAIL",
                "check_shadow exit %d: %s" % (code, summary))

    # --- implementation gate ---
    ok, note = _import_sync_daemon_ready()
    add("phase3_writer_stub", "implementation",
        "PASS" if ok else "FAIL", note)

    ok, note = _local_schema_ok()
    add("db_schema_present", "data", "PASS" if ok else "FAIL", note)

    # --- hub env / services (SSH) ---
    if dry_run or mock:
        add("hub_sync_mode_safe", "hub", "SKIP",
            "dry-run/mock: skip hub env SSH")
        add("sync_daemon_active", "hub", "SKIP",
            "dry-run/mock: skip systemctl")
        add("hub_git_clean", "hub", "SKIP",
            "dry-run/mock: skip hub git status")
        add("orphaned_claims", "data", "SKIP",
            "dry-run/mock: skip hub DB query")
        add("coordination_mode_uniform", "fleet", "SKIP",
            "dry-run/mock: skip fleet SSH")
    else:
        ok, out, err = _ssh_run(
            hub_ssh, ssh_user,
            "grep -E '^SYNC_MODE=' /etc/ree-coordinator.env 2>/dev/null || true",
            dry_run=False)
        if ok and out:
            if "authoritative" in out:
                add("hub_sync_mode_safe", "hub", "FAIL",
                    "hub already SYNC_MODE=authoritative")
            elif "coordinator" in out:
                add("hub_sync_mode_safe", "hub", "PASS",
                    "hub SYNC_MODE=coordinator")
            else:
                add("hub_sync_mode_safe", "hub", "WARN",
                    "hub SYNC_MODE not coordinator: %s" % out)
        else:
            add("hub_sync_mode_safe", "hub", "FAIL",
                "cannot read hub env: %s" % err)

        ok, out, err = _ssh_run(
            hub_ssh, ssh_user,
            "systemctl is-active ree-sync-daemon 2>/dev/null",
            dry_run=False)
        if ok and out.strip() == "active":
            add("sync_daemon_active", "hub", "PASS",
                "ree-sync-daemon active")
        else:
            add("sync_daemon_active", "hub", "FAIL",
                "ree-sync-daemon not active: %s" % (err or out))

        ok, out, err = _ssh_run(
            hub_ssh, ssh_user,
            "git -C %s status --porcelain --untracked-files=no 2>/dev/null"
            % HUB_REE_ASSEMBLY,
            dry_run=False)
        if not ok:
            add("hub_git_clean", "hub", "WARN",
                "hub git status failed: %s" % err)
        else:
            bad = []
            for line in out.splitlines():
                entry = line[3:].strip()
                if entry and not any(
                        entry.startswith(p) for p in REGENERABLE_PREFIXES):
                    bad.append(entry)
            if bad:
                add("hub_git_clean", "hub", "FAIL",
                    "hub REE_assembly dirty non-telemetry: %s" % bad[:5])
            else:
                add("hub_git_clean", "hub", "PASS",
                    "hub REE_assembly clean or telemetry-only")

        sql = (
            "SELECT e.queue_id, e.claimed_by_machine "
            "FROM experiments e "
            "LEFT JOIN heartbeats h ON h.machine = e.claimed_by_machine "
            "WHERE e.status='claimed' AND ("
            "  h.last_seen IS NULL OR "
            "  h.last_seen < datetime('now', '-900 seconds') OR "
            "  (h.current_exq IS NOT NULL AND h.current_exq != e.queue_id)"
            ");"
        )
        remote = (
            "sqlite3 -batch %s %s 2>/dev/null | head -20"
            % (HUB_COORDINATOR_DB, repr(sql))
        )
        ok, out, err = _ssh_run(hub_ssh, ssh_user, remote, dry_run=False)
        if not ok:
            add("orphaned_claims", "data", "WARN",
                "hub DB query failed: %s" % err)
        elif out.strip():
            rows = out.strip().splitlines()[:10]
            add("orphaned_claims", "data", "FAIL",
                "possible orphaned claims: %s" % rows)
        else:
            add("orphaned_claims", "data", "PASS",
                "no stale claimed rows in hub DB")

        modes: dict[str, str] = {}
        for host in FLEET_SSH_HOSTS:
            h = ssh_hosts.get(host, DEFAULT_SSH_HOSTS.get(host, host))
            dropin = (
                "/etc/systemd/system/ree-runner.service.d/shadow.conf"
            )
            ok, out, err = _ssh_run(
                h, ssh_user,
                "grep -E 'COORDINATION_MODE=' %s 2>/dev/null || echo MISSING"
                % dropin,
                dry_run=False)
            if ok:
                m = re.search(r"COORDINATION_MODE=(\w+)", out)
                modes[host] = m.group(1) if m else "MISSING"
            else:
                modes[host] = "SSH_FAIL"

        bad = {h: m for h, m in modes.items()
               if m not in ("coordinator",)}
        if bad:
            add("coordination_mode_uniform", "fleet", "FAIL",
                "mixed or non-coordinator modes: %s" % bad)
        else:
            add("coordination_mode_uniform", "fleet", "PASS",
                "all fleet hosts COORDINATION_MODE=coordinator")

    required_fail = [c for c in checks if c.status == "FAIL"]
    required_warn = [c for c in checks if c.status == "WARN"]
    ok = len(required_fail) == 0

    return {
        "ok": ok,
        "checked_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": dry_run,
        "mock": mock,
        "coordinator_url": url,
        "fail_count": len(required_fail),
        "warn_count": len(required_warn),
        "checks": [
            {
                "id": c.check_id,
                "category": c.category,
                "status": c.status,
                "message": c.message,
                "detail": c.detail,
            }
            for c in checks
        ],
    }


def _summary(checks: list[CheckResult], *, quiet: bool) -> dict:
    fail = [c for c in checks if c.status == "FAIL"]
    return {
        "ok": len(fail) == 0,
        "checked_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": False,
        "mock": False,
        "coordinator_url": "",
        "fail_count": len(fail),
        "warn_count": 0,
        "checks": [
            {
                "id": c.check_id,
                "category": c.category,
                "status": c.status,
                "message": c.message,
                "detail": c.detail,
            }
            for c in checks
        ],
    }


def _print_report(summary: dict) -> None:
    print("Phase 3 preflight @ %s" % summary.get("checked_at", "?"))
    if summary.get("dry_run"):
        print("  (dry-run: SSH checks skipped)")
    if summary.get("mock"):
        print("  (mock: network checks skipped)")
    url = summary.get("coordinator_url") or "?"
    print("  coordinator: %s" % url)
    for c in summary.get("checks", []):
        print("  [%s] %s/%s: %s" % (
            c["status"], c["category"], c["id"], c["message"]))
    print("")
    if summary.get("ok"):
        print("VERDICT: PASS -- safe to schedule Phase 3 cutover prep")
        print("  (implementation still required before SYNC_MODE=authoritative)")
    else:
        print("VERDICT: FAIL -- %d blocking check(s)" % summary.get(
            "fail_count", 0))
        print("  Do NOT run phase3_cutover.sh or enable authoritative mode.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 3 pre-cutover precondition checks")
    ap.add_argument("--env-file", default=str(DEFAULT_ENV),
                    help="path to REE_assembly/coordinator.env")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip SSH probes; local checks only")
    ap.add_argument("--mock", action="store_true",
                    help="skip live HTTP/SSH (tests)")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON summary on stdout")
    args = ap.parse_args()

    env_path = Path(args.env_file).expanduser()
    if not env_path.exists() and not args.mock:
        sys.stderr.write("ERROR: missing env file: %s\n" % env_path)
        return 2

    summary = run_preflight(
        env_file=env_path,
        dry_run=args.dry_run,
        mock=args.mock,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_report(summary)

    if not summary.get("coordinator_url") and not args.mock:
        return 2
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
