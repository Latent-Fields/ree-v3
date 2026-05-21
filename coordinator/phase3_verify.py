"""Phase 3 post-cutover verification checks.

Run after hub SYNC_MODE=authoritative and workers stop git-pushing
coordination artifacts. Many checks SKIP until the corresponding code
lands; re-run after each implementation milestone.

  /opt/local/bin/python3 phase3_verify.py
  /opt/local/bin/python3 phase3_verify.py --expect-cutover
  /opt/local/bin/python3 phase3_verify.py --dry-run --json

Exit codes:
  0  all required (non-SKIP) checks PASS
  1  FAIL present
  2  configuration error

ASCII-only output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ENV = Path.home() / "REE_Working" / "REE_assembly" / "coordinator.env"

# Checks that require phase3_git_writer + runner changes before they can PASS.
_POST_CUTOVER_STUBS = frozenset({
    "sync_daemon_phase3_tick",
    "hub_git_writer_only",
    "workers_no_result_git_push",
    "heartbeat_git_retired",
    "results_drained",
    "queue_snapshot_fresh",
    "derived_heartbeats",
})


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


def _import_writer_state() -> tuple[bool, str]:
    try:
        import sync_daemon  # noqa: WPS433
        ready = bool(getattr(sync_daemon, "PHASE3_GIT_WRITER_READY", False))
        if ready:
            return True, "PHASE3_GIT_WRITER_READY True"
        return False, "PHASE3_GIT_WRITER_READY False (writer still stub)"
    except Exception as exc:  # noqa: BLE001
        return False, "import failed: %r" % exc


def run_verify(
    *,
    env_file: Path | None = None,
    dry_run: bool = False,
    expect_cutover: bool = False,
) -> dict:
    from phase3_preflight import (  # noqa: WPS433
        DEFAULT_SSH_HOSTS,
        _http_get,
        _load_env_file,
        _run_check_shadow,
        _ssh_run,
    )

    env_path = env_file or Path(
        os.environ.get("COORDINATOR_ENV_FILE", str(DEFAULT_ENV)))
    cfg = _load_env_file(env_path)
    url = cfg.get("COORDINATOR_URL") or os.environ.get("COORDINATOR_URL", "")
    token = (cfg.get("COORDINATOR_LOCAL_TOKEN")
             or os.environ.get("COORDINATOR_TOKEN", ""))
    ssh_user = cfg.get("COORDINATOR_SSH_USER", "ree")
    hub_ssh = cfg.get("SHADOW_SSH_HOST_ree-cloud-1") or DEFAULT_SSH_HOSTS[
        "ree-cloud-1"]

    checks = []

    def add(cid: str, cat: str, status: str, msg: str):
        checks.append({
            "id": cid, "category": cat, "status": status, "message": msg,
        })

    writer_ready, writer_note = _import_writer_state()

    if not expect_cutover:
        add("cutover_expected", "meta", "SKIP",
            "pass --expect-cutover after maintenance window")

    # Hub SYNC_MODE should be authoritative only after cutover.
    if dry_run:
        add("hub_sync_mode_authoritative", "hub", "SKIP",
            "dry-run: skip hub env SSH")
    else:
        ok, out, err = _ssh_run(
            hub_ssh, ssh_user,
            "grep -E '^SYNC_MODE=' /etc/ree-coordinator.env 2>/dev/null || true",
            dry_run=False)
        if not ok:
            add("hub_sync_mode_authoritative", "hub", "FAIL",
                "cannot read hub env: %s" % err)
        elif "authoritative" in (out or ""):
            add("hub_sync_mode_authoritative", "hub", "PASS",
                "hub SYNC_MODE=authoritative")
        elif expect_cutover:
            add("hub_sync_mode_authoritative", "hub", "FAIL",
                "expected authoritative; got: %s" % (out or "?"))
        else:
            add("hub_sync_mode_authoritative", "hub", "SKIP",
                "pre-cutover (SYNC_MODE not authoritative yet)")

    # Stub checks: SKIP until writer + runner paths land.
    for cid in sorted(_POST_CUTOVER_STUBS):
        if not expect_cutover or not writer_ready:
            add(cid, _stub_category(cid), "SKIP",
                "awaits phase3_git_writer + runner git-push retirement")
        else:
            add(cid, _stub_category(cid), "FAIL",
                "check not implemented yet (%s)" % writer_note)

    # Claims path should stay healthy (same as Phase 2).
    if url and token:
        code, summary = _run_check_shadow(url, token)
        if code == 0:
            add("claims_still_healthy", "soak", "PASS",
                "check_shadow exit 0: %s" % summary)
        else:
            add("claims_still_healthy", "soak", "WARN",
                "check_shadow exit %d: %s" % (code, summary))
    elif url:
        st, body, err = _http_get(url, None, "/health")
        if st == 200 and body and body.get("ok"):
            add("claims_still_healthy", "soak", "WARN",
                "health ok; no token for check_shadow")
        else:
            add("claims_still_healthy", "soak", "FAIL",
                "hub unreachable: %s" % err)
    else:
        add("claims_still_healthy", "soak", "SKIP", "no COORDINATOR_URL")

    fail = [c for c in checks if c["status"] == "FAIL"]
    skip = [c for c in checks if c["status"] == "SKIP"]
    ok = len(fail) == 0

    return {
        "ok": ok,
        "checked_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "expect_cutover": expect_cutover,
        "dry_run": dry_run,
        "writer_ready": writer_ready,
        "fail_count": len(fail),
        "skip_count": len(skip),
        "checks": checks,
    }


def _stub_category(cid: str) -> str:
    if cid.startswith("hub_"):
        return "hub"
    if cid.startswith("workers_") or cid.startswith("heartbeat_"):
        return "fleet"
    if cid.startswith("results_") or cid.startswith("queue_"):
        return "data"
    if cid == "derived_heartbeats":
        return "explorer"
    return "hub"


def _print_report(summary: dict) -> None:
    print("Phase 3 verify @ %s" % summary.get("checked_at", "?"))
    if summary.get("expect_cutover"):
        print("  (--expect-cutover: post-cutover mode)")
    for c in summary.get("checks", []):
        print("  [%s] %s/%s: %s" % (
            c["status"], c["category"], c["id"], c["message"]))
    print("")
    if summary.get("ok"):
        print("VERDICT: PASS -- no blocking FAIL checks")
    else:
        print("VERDICT: FAIL -- %d blocking check(s)" % summary.get(
            "fail_count", 0))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 3 post-cutover verification")
    ap.add_argument("--env-file", default=str(DEFAULT_ENV))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--expect-cutover", action="store_true",
                    help="enable post-cutover required checks (not SKIP)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    summary = run_verify(
        env_file=Path(args.env_file).expanduser(),
        dry_run=args.dry_run,
        expect_cutover=args.expect_cutover,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_report(summary)

    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
