"""Periodic fleet integrity check (standalone deploy).

Run from the Mac (or any host with SSH to the Hetzner workers). Read-only
remote probes: authorized_keys fingerprint, enabled systemd units, top CPU
processes, optional GPU compute apps, WireGuard peer count on the hub, and
coordinator heartbeat machine names.

Does NOT change shadow/coordinator behaviour. Complements check_shadow.py
(scientific mutex agreement) with host-level "is this still our box?" checks.

  /opt/local/bin/python3 fleet_integrity_check.py
  /opt/local/bin/python3 fleet_integrity_check.py --write-baseline
  /opt/local/bin/python3 fleet_integrity_check.py --baseline ~/fleet_baseline.json

Loads SSH targets from REE_assembly/coordinator.env (SHADOW_SSH_HOST_*)
when present, else public IPv4 defaults from FLEET_CHECKLIST.md.

Exit codes:
  0  OK      -- no alerts (warnings may still print)
  1  WARN    -- drift or soft signals (review)
  2  ALERT   -- likely compromise / foreign compute (act)
  3  ERROR   -- SSH or config failure on one or more required hosts

ASCII-only stdout/stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COORD_DIR = SCRIPT_DIR.parent
DEFAULT_BASELINE = SCRIPT_DIR / "fleet_integrity_baseline.json"
DEFAULT_ENV = Path.home() / "REE_Working" / "REE_assembly" / "coordinator.env"

# Public IPv4 fallbacks (Mac cannot SSH via worker WG IPs). See FLEET_CHECKLIST.md
DEFAULT_SSH_HOSTS = {
    "ree-cloud-1": "91.98.130.117",
    "ree-cloud-2": "116.203.216.181",
    "ree-cloud-3": "46.62.170.133",
    "ree-cloud-4": "91.99.68.94",
}

CLOUD_HOSTS = ("ree-cloud-1", "ree-cloud-2", "ree-cloud-3", "ree-cloud-4")
HUB_HOST = "ree-cloud-1"

ALLOWED_MACHINES = frozenset({
    "Mac", "ree-cloud-1", "ree-cloud-2", "ree-cloud-3", "ree-cloud-4",
    "Daniel-PC", "EWIN-PC", "DLAPTOP-4.local",
})

# New enabled units matching these are ALERT even if baseline missed them.
SUSPICIOUS_UNIT_RE = re.compile(
    r"(miner|xmrig|kinsing|docker\.service$|k3s|kubelet|nomad)",
    re.IGNORECASE,
)

SUSPICIOUS_CMD_RE = re.compile(
    r"(xmrig|minerd|kinsing|kdevtmpfsi|\.kinsing|stratum\+|cryptonight|"
    r"nbminer|t-rex|lolminer|phoenixminer|teamredminer|cgminer|bminer|"
    r"ethminer|nanominer|dockerd\s|/tmp/[a-z0-9]{8,})",
    re.IGNORECASE,
)

GPU_OK_RE = re.compile(r"python|ree|experiment_runner|torch", re.IGNORECASE)


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


def _ssh_targets(cfg: dict) -> dict[str, str]:
    out = dict(DEFAULT_SSH_HOSTS)
    for h in CLOUD_HOSTS:
        key = "SHADOW_SSH_HOST_" + h
        if cfg.get(key):
            out[h] = cfg[key]
    return out


def _ssh_run(host: str, user: str, remote_cmd: str, timeout: int = 25) -> tuple:
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
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "", "ssh timed out"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        msg = detail[-1] if detail else "ssh exit %d" % proc.returncode
        return False, "", msg
    return True, proc.stdout, ""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_authorized_keys(host: str, user: str, baseline: dict,
                           write_baseline: bool) -> list[str]:
    issues: list[str] = []
    ok, out, err = _ssh_run(
        host, user,
        "if [ -f ~/.ssh/authorized_keys ]; then "
        "sha256sum ~/.ssh/authorized_keys | awk '{print $1}'; "
        "else echo MISSING; fi")
    if not ok:
        return ["SSH failed: %s" % err]

    fp = out.strip().splitlines()[-1] if out.strip() else "MISSING"
    entry = baseline.setdefault("hosts", {}).setdefault(host, {})
    prev = entry.get("authorized_keys_sha256")
    if write_baseline:
        entry["authorized_keys_sha256"] = fp
        entry["recorded_at"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        return issues

    if prev is None:
        issues.append(
            "WARN %s: no baseline for authorized_keys (run --write-baseline)"
            % host)
    elif prev != fp:
        issues.append(
            "ALERT %s: authorized_keys changed (was %s.. now %s..)"
            % (host, prev[:12], fp[:12]))
    return issues


def _enabled_units(host: str, user: str) -> tuple[bool, set[str], str]:
    ok, out, err = _ssh_run(
        host, user,
        "systemctl list-unit-files --state=enabled --no-pager --no-legend "
        "2>/dev/null | awk '{print $1}'")
    if not ok:
        return False, set(), err
    units = set()
    for line in out.splitlines():
        u = line.strip()
        if u.endswith(".service") or u.endswith(".socket"):
            units.add(u)
    return True, units, ""


def _check_systemd(host: str, user: str, baseline: dict,
                   write_baseline: bool) -> list[str]:
    issues: list[str] = []
    ok, units, err = _enabled_units(host, user)
    if not ok:
        return ["SSH failed (systemd): %s" % err]

    entry = baseline.setdefault("hosts", {}).setdefault(host, {})
    if write_baseline:
        entry["enabled_units"] = sorted(units)
        return issues

    prev = set(entry.get("enabled_units") or [])
    if not prev:
        issues.append(
            "WARN %s: no systemd baseline (re-run --write-baseline)" % host)
        return issues

    new_units = units - prev
    for unit in sorted(new_units):
        if SUSPICIOUS_UNIT_RE.search(unit):
            issues.append(
                "ALERT %s: newly enabled unit %s" % (host, unit))
        else:
            issues.append(
                "WARN %s: newly enabled unit %s" % (host, unit))
    return issues


def _check_processes(host: str, user: str) -> list[str]:
    issues: list[str] = []
    ok, out, err = _ssh_run(
        host, user,
        "ps aux --sort=-%cpu 2>/dev/null | head -n 20")
    if not ok:
        return ["SSH failed (ps): %s" % err]

    for line in out.splitlines()[1:]:
        if SUSPICIOUS_CMD_RE.search(line):
            issues.append(
                "ALERT %s: suspicious process line: %s"
                % (host, line[:120]))
    return issues


def _check_gpu(host: str, user: str) -> list[str]:
    issues: list[str] = []
    ok, out, err = _ssh_run(
        host, user,
        "nvidia-smi --query-compute-apps=pid,process_name "
        "--format=csv,noheader 2>/dev/null || true")
    if not ok:
        return []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.lower() == "pid, process_name":
            continue
        # csv: pid, name
        parts = [p.strip() for p in line.split(",", 1)]
        pname = parts[1] if len(parts) > 1 else line
        if pname and not GPU_OK_RE.search(pname):
            issues.append(
                "ALERT %s: unexpected GPU compute process: %s"
                % (host, pname))
    return issues


def _check_wg_hub(host: str, user: str, min_peers: int) -> list[str]:
    issues: list[str] = []
    ok, out, err = _ssh_run(host, user, "sudo wg show wg0 peers 2>/dev/null")
    if not ok:
        return ["WARN hub: cannot read wg peers (%s)" % err]
    n = len([ln for ln in out.splitlines() if ln.strip()])
    if n < min_peers:
        issues.append(
            "WARN hub: wg0 has %d peer(s), expected >= %d" % (n, min_peers))
    return issues


def _check_coordinator_machines(url: str, token: str) -> list[str]:
    issues: list[str] = []
    if not url:
        return []
    req = urllib.request.Request(
        url.rstrip("/") + "/shadow/status",
        headers={"Authorization": "Bearer " + token} if token else {},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            st = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return ["WARN coordinator: bad token (skipped machine check)"]
        return ["WARN coordinator: HTTP %s" % exc.code]
    except (urllib.error.URLError, OSError) as exc:
        return ["WARN coordinator: unreachable (%r)" % exc]

    for m in st.get("machines", []):
        name = m.get("machine") or ""
        if name and name not in ALLOWED_MACHINES:
            issues.append(
                "ALERT coordinator: unknown machine name %r" % name)
    return issues


def _classify_issues(issues: list[str]) -> int:
    code = 0
    for it in issues:
        if it.startswith("ALERT"):
            code = max(code, 2)
        elif it.startswith("WARN"):
            code = max(code, 1)
        elif it.startswith("SSH failed"):
            code = max(code, 3)
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--env-file",
        default=os.environ.get("REE_COORDINATOR_ENV", str(DEFAULT_ENV)),
        help="coordinator.env path (SSH hosts + optional token)",
    )
    ap.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="JSON baseline for authorized_keys hashes (local, gitignored)",
    )
    ap.add_argument(
        "--write-baseline",
        action="store_true",
        help="record current authorized_keys hashes, then exit 0",
    )
    ap.add_argument("--ssh-user", default="",
                    help="default: COORDINATOR_SSH_USER or ree")
    ap.add_argument(
        "--min-wg-peers",
        type=int,
        default=4,
        help="minimum wg0 peers on hub (Mac + clouds on mesh)",
    )
    ap.add_argument(
        "--skip-coordinator",
        action="store_true",
        help="skip HTTP machine-name check",
    )
    ap.add_argument(
        "--hosts",
        default="",
        help="comma-separated subset of ree-cloud-1..4 (default: all)",
    )
    args = ap.parse_args()

    cfg = _load_env_file(Path(args.env_file))
    ssh_user = args.ssh_user or cfg.get("COORDINATOR_SSH_USER", "ree")
    targets = _ssh_targets(cfg)
    coord_url = cfg.get("COORDINATOR_URL", os.environ.get("COORDINATOR_URL", ""))
    coord_tok = cfg.get(
        "COORDINATOR_LOCAL_TOKEN",
        os.environ.get("COORDINATOR_TOKEN",
                        os.environ.get("COORDINATOR_LOCAL_TOKEN", "")),
    )

    if args.hosts.strip():
        want = {h.strip() for h in args.hosts.split(",") if h.strip()}
        targets = {k: v for k, v in targets.items() if k in want}

    baseline_path = Path(args.baseline)
    baseline: dict = {"version": 1, "hosts": {}}
    if baseline_path.exists() and not args.write_baseline:
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            sys.stderr.write("baseline read error: %r\n" % exc)
            return 3

    all_issues: list[str] = []
    print("fleet integrity check  %s"
          % datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    print("ssh user=%s  targets=%s" % (ssh_user, ", ".join(sorted(targets))))

    for logical, addr in sorted(targets.items()):
        print("-- %s (%s) --" % (logical, addr))
        all_issues.extend(
            _check_authorized_keys(addr, ssh_user, baseline,
                                   args.write_baseline))
        all_issues.extend(
            _check_systemd(addr, ssh_user, baseline, args.write_baseline))
        if not args.write_baseline:
            all_issues.extend(_check_processes(addr, ssh_user))
            all_issues.extend(_check_gpu(addr, ssh_user))
            if logical == HUB_HOST:
                all_issues.extend(
                    _check_wg_hub(addr, ssh_user, args.min_wg_peers))

    if args.write_baseline:
        baseline["updated"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = baseline_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, baseline_path)
        try:
            os.chmod(baseline_path, 0o600)
        except OSError:
            pass
        print("wrote baseline: %s" % baseline_path)
        return 0

    if not args.skip_coordinator:
        all_issues.extend(_check_coordinator_machines(coord_url, coord_tok))

    print("")
    if not all_issues:
        print("VERDICT: OK -- no integrity alerts")
        return 0

    for it in sorted(all_issues):
        print(it)
    code = _classify_issues(all_issues)
    labels = {0: "OK", 1: "WARN", 2: "ALERT", 3: "ERROR"}
    print("")
    print("VERDICT: %s (%d issue line(s))" % (labels.get(code, "?"), len(all_issues)))
    return code


if __name__ == "__main__":
    sys.exit(main())
