"""Phase 3 post-cutover verification checks.

Run after hub SYNC_MODE=authoritative and workers stop git-pushing
coordination artifacts.

  /opt/local/bin/python3 phase3_verify.py
  /opt/local/bin/python3 phase3_verify.py --expect-cutover
  /opt/local/bin/python3 phase3_verify.py --dry-run --json
  /opt/local/bin/python3 phase3_verify.py --mock --json   # tests

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
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ENV = Path.home() / "REE_Working" / "REE_assembly" / "coordinator.env"

# Required prefixes on every commit attributable to the sync_daemon writer.
# Mirrors the constants declared in sync_daemon.py; if those drift, both the
# writer and this guard need updating together (same coupling rationale as
# the writer's own foreign-commit guard).
WRITER_COMMIT_PREFIXES = (
    "phase3: ",
    "phase3-queue: ",
    "phase3-heartbeats: ",
)

# Workers under systemd control. DLAPTOP-4.local is the developer Mac;
# treated specially (local env probe rather than SSH).
CLOUD_WORKERS = ("ree-cloud-1", "ree-cloud-2", "ree-cloud-3", "ree-cloud-4")
LOCAL_WORKER = "DLAPTOP-4.local"
ALL_WORKERS = CLOUD_WORKERS + (LOCAL_WORKER,)

# Hub paths (mirror phase3_preflight).
HUB_REE_ASSEMBLY = "/home/ree/REE_Working/REE_assembly"
HUB_REE_V3 = "/home/ree/REE_Working/ree-v3"
HUB_COORDINATOR_DB = (
    "/home/ree/REE_Working/ree-v3/coordinator/coordinator.db"
)
HUB_QUEUE_REF = "origin/main"

# How fresh "recent" must be (seconds) for tick / heartbeat-file checks.
DEFAULT_TICK_LOOKBACK_SEC = 300
DEFAULT_HEARTBEAT_LOOKBACK_SEC = 600

# How many recent REE_assembly commits the hub_git_writer_only check
# inspects. Small enough that operator-fingered foreign commits show up
# loudly; large enough that the writer batch cadence does not bury them.
DEFAULT_RECENT_COMMITS = 30


# --------------------------------------------------------------------------
# Pure helpers (no SSH, no clock dependency unless an explicit "now" is
# passed in). Cover everything we want a unit test to exercise.
# --------------------------------------------------------------------------

def _classify_commits(subject_lines, allowed_prefixes=WRITER_COMMIT_PREFIXES):
    """Split commit subject lines into (writer_authored, foreign).

    Empty / whitespace-only lines are ignored. A "writer-authored" line
    starts with any string in allowed_prefixes. Anything else is foreign.
    """
    writer = []
    foreign = []
    for line in subject_lines:
        s = line.strip()
        if not s:
            continue
        if any(s.startswith(p) for p in allowed_prefixes):
            writer.append(s)
        else:
            foreign.append(s)
    return writer, foreign


def _find_phase3_tick_evidence(journal_text):
    """Search recent sync_daemon journal text for evidence of a live tick.

    Returns (ok, evidence_line, stub_seen). ok=True if any "[phase3]",
    "[phase3-queue]" or "[phase3-heartbeats]" line is present that is NOT
    the stub-refusal line. stub_seen flags the stub line for diagnostics.
    """
    stub_marker = "git writer stub (PHASE3_GIT_WRITER_READY=False)"
    stub_seen = False
    evidence = None
    # Prefer the most-recent positive line. Walk in reverse.
    for line in reversed(journal_text.splitlines()):
        s = line.strip()
        if not s:
            continue
        if stub_marker in s:
            stub_seen = True
            continue
        if ("[phase3]" in s
                or "[phase3-queue]" in s
                or "[phase3-heartbeats]" in s):
            evidence = s
            break
    return (evidence is not None), evidence, stub_seen


def _parse_systemd_environment(systemctl_show_output):
    """Parse `systemctl show <unit> -p Environment` output to a dict.

    Output shape (single line): `Environment=KEY=VAL KEY2=VAL2 ...` or
    empty / missing when no Environment= directive is set. Values
    containing spaces would need quoting from systemd; we accept the
    common KEY=VAL whitespace-separated form used by our drop-in confs.
    """
    out = {}
    for line in systemctl_show_output.splitlines():
        s = line.strip()
        if not s.startswith("Environment="):
            continue
        rest = s[len("Environment="):]
        for tok in rest.split():
            if "=" not in tok:
                continue
            k, _, v = tok.partition("=")
            out[k.strip()] = v.strip()
    return out


def _journal_has_gate_line(journal_text, needle):
    """True iff any non-empty line in journal_text contains `needle`.

    Used for the two "recent runner log shows phase3 gate ..." checks.
    """
    return any(needle in line for line in journal_text.splitlines())


def _diff_queue_against_db(file_items, db_queue_ids):
    """Compare experiment_queue.json items against the DB's
    pending/claimed set.

    Returns (file_only, db_only). file_items is the parsed JSON's
    `items` list (a list of dicts with a `queue_id` field).
    db_queue_ids is an iterable of strings.
    """
    file_ids = set()
    for it in file_items or []:
        if isinstance(it, dict) and isinstance(it.get("queue_id"), str):
            file_ids.add(it["queue_id"])
    db_ids = set(db_queue_ids or [])
    return sorted(file_ids - db_ids), sorted(db_ids - file_ids)


def _is_fresh(epoch_seconds, now_seconds, lookback_seconds):
    """True iff epoch_seconds is within lookback_seconds of now_seconds."""
    try:
        e = int(epoch_seconds)
    except (TypeError, ValueError):
        return False
    return (now_seconds - e) <= lookback_seconds and e <= now_seconds + 5


# --------------------------------------------------------------------------
# Live checks. Each takes an injected `ssh` callable so tests can supply
# canned outputs without spawning subprocess.
# --------------------------------------------------------------------------

def check_sync_daemon_phase3_tick(hub_ssh, ssh_user, *, ssh,
                                  lookback_sec=DEFAULT_TICK_LOOKBACK_SEC):
    """Hub journal shows a real phase3 tick within the lookback window."""
    since = "%d seconds ago" % lookback_sec
    cmd = ("journalctl -u ree-sync-daemon --since '%s' --no-pager "
           "--output=cat 2>/dev/null | tail -500" % since)
    ok, out, err = ssh(hub_ssh, ssh_user, cmd, dry_run=False)
    if not ok:
        return "FAIL", "ssh/journalctl failed: %s" % (err or "?")
    found, evidence, stub = _find_phase3_tick_evidence(out or "")
    if found:
        return "PASS", "phase3 tick: %s" % evidence[:160]
    if stub:
        return "FAIL", (
            "no phase3 tick in last %ds; writer is still stub-refusing"
            % lookback_sec)
    return "FAIL", (
        "no phase3 tick evidence in last %ds (journal had %d lines)"
        % (lookback_sec, len((out or "").splitlines())))


def check_hub_git_writer_only(hub_ssh, ssh_user, *, ssh,
                              recent_n=DEFAULT_RECENT_COMMITS):
    """The N most recent REE_assembly commits on origin/master are all
    writer-authored."""
    cmd = ("git -C %s fetch --quiet origin master 2>/dev/null && "
           "git -C %s log -%d --format=%%s origin/master"
           % (HUB_REE_ASSEMBLY, HUB_REE_ASSEMBLY, recent_n))
    ok, out, err = ssh(hub_ssh, ssh_user, cmd, dry_run=False)
    if not ok:
        return "FAIL", "ssh/git failed: %s" % (err or "?")
    writer, foreign = _classify_commits((out or "").splitlines())
    if not writer and not foreign:
        return "FAIL", "git log returned no commits (unexpected)"
    if foreign:
        sample = foreign[:3]
        return "FAIL", (
            "%d foreign commit(s) in last %d on origin/master: %r"
            % (len(foreign), recent_n, sample))
    return "PASS", (
        "%d/%d recent commits are writer-authored"
        % (len(writer), recent_n))


def _check_one_worker_env(worker, ssh_user, *, ssh, env_keys, journal_needle):
    """Inspect a single worker. Returns (status, message, env_dict).

    env_keys: list of env-var names that must all be set to "1" on the
    runner unit. journal_needle: substring required in recent runner
    journal output.
    """
    if worker == LOCAL_WORKER:
        return _check_local_worker_env(env_keys, journal_needle)
    show_cmd = ("systemctl show ree-runner.service -p Environment "
                "--no-pager 2>/dev/null")
    ok, out, err = ssh(worker, ssh_user, show_cmd, dry_run=False)
    if not ok:
        return ("FAIL", "ssh/systemctl failed: %s" % (err or "?"), {})
    env = _parse_systemd_environment(out or "")
    missing = [k for k in env_keys if env.get(k) != "1"]
    if missing:
        return ("FAIL",
                "missing env keys (need =1): %s" % ", ".join(missing),
                env)
    j_cmd = ("journalctl -u ree-runner.service --since '15 min ago' "
             "--no-pager --output=cat 2>/dev/null | tail -500")
    ok, j_out, j_err = ssh(worker, ssh_user, j_cmd, dry_run=False)
    if not ok:
        return ("FAIL",
                "env OK but runner journal unreadable: %s" % (j_err or "?"),
                env)
    if not _journal_has_gate_line(j_out or "", journal_needle):
        return ("WARN",
                "env OK but no recent log line matching %r" % journal_needle,
                env)
    return ("PASS", "env + journal OK", env)


def _check_local_worker_env(env_keys, journal_needle):
    """Best-effort local check on the Mac (DLAPTOP-4.local).

    The Mac runner is not under systemd; the operator sets env vars in
    the runner-launch shell. We can only inspect this process's
    environment, which is NOT the runner's. Report SKIP with a clear
    note so the operator knows to verify it by hand.
    """
    host = socket.gethostname()
    if host != LOCAL_WORKER:
        return ("SKIP",
                "local worker %s not reachable (verifier on %s)"
                % (LOCAL_WORKER, host),
                {})
    return ("SKIP",
            "Mac runner is not under systemd; verify %s by hand "
            "(operator-managed shell env)" % ",".join(env_keys),
            {})


def check_workers_no_result_git_push(workers, ssh_user, *, ssh):
    """Every worker's runner env has PHASE3_DISABLE_RUNNER_RESULT_PUSH=1
    AND recent runner journal shows the gate-active line."""
    per_worker = {}
    fail = []
    skip = []
    warn = []
    for w in workers:
        status, msg, _ = _check_one_worker_env(
            w, ssh_user, ssh=ssh,
            env_keys=["PHASE3_DISABLE_RUNNER_RESULT_PUSH"],
            journal_needle="phase3 gate: skipping git_push_results")
        per_worker[w] = "%s: %s" % (status, msg)
        if status == "FAIL":
            fail.append("%s: %s" % (w, msg))
        elif status == "SKIP":
            skip.append("%s: %s" % (w, msg))
        elif status == "WARN":
            warn.append("%s: %s" % (w, msg))
    if fail:
        return "FAIL", "workers failing: %s" % "; ".join(fail), per_worker
    if warn and not [k for k in per_worker if per_worker[k].startswith("PASS")]:
        # All workers warn -- promote to FAIL: no positive evidence.
        return "FAIL", "no PASS evidence: %s" % "; ".join(warn), per_worker
    if warn:
        return "WARN", "some workers missing gate log: %s" % "; ".join(warn), per_worker
    return ("PASS",
            "result_push gate active on %d worker(s); %d skipped"
            % (len(workers) - len(skip), len(skip)),
            per_worker)


def check_heartbeat_git_retired(workers, ssh_user, *, ssh):
    """Every worker has PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH=1 AND
    runner_remote_control logged the gate-active line."""
    per_worker = {}
    fail = []
    skip = []
    warn = []
    for w in workers:
        status, msg, _ = _check_one_worker_env(
            w, ssh_user, ssh=ssh,
            env_keys=["PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH"],
            journal_needle=(
                "phase3 gate active: heartbeat + commands"))
        per_worker[w] = "%s: %s" % (status, msg)
        if status == "FAIL":
            fail.append("%s: %s" % (w, msg))
        elif status == "SKIP":
            skip.append("%s: %s" % (w, msg))
        elif status == "WARN":
            warn.append("%s: %s" % (w, msg))
    if fail:
        return "FAIL", "workers failing: %s" % "; ".join(fail), per_worker
    if warn and not [k for k in per_worker if per_worker[k].startswith("PASS")]:
        return "FAIL", "no PASS evidence: %s" % "; ".join(warn), per_worker
    if warn:
        return "WARN", "some workers missing gate log: %s" % "; ".join(warn), per_worker
    return ("PASS",
            "heartbeat gate active on %d worker(s); %d skipped"
            % (len(workers) - len(skip), len(skip)),
            per_worker)


def check_results_drained(hub_ssh, ssh_user, *, ssh):
    """Hub DB: count(*) of results rows missing committed_at == 0."""
    sql = "SELECT COUNT(*) FROM results WHERE committed_at IS NULL;"
    remote = ("sqlite3 -batch %s %s 2>/dev/null"
              % (HUB_COORDINATOR_DB, repr(sql)))
    ok, out, err = ssh(hub_ssh, ssh_user, remote, dry_run=False)
    if not ok:
        return "FAIL", "ssh/sqlite3 failed: %s" % (err or "?")
    try:
        n = int((out or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return "FAIL", "could not parse count: %r" % (out or "")
    if n == 0:
        return "PASS", "all results committed (0 pending)"
    return "FAIL", "%d result(s) still uncommitted on hub" % n


def check_queue_snapshot_fresh(hub_ssh, ssh_user, *, ssh):
    """`experiment_queue.json` on origin/main matches DB pending/claimed."""
    qcmd = ("git -C %s fetch --quiet origin main 2>/dev/null && "
            "git -C %s show %s:experiment_queue.json"
            % (HUB_REE_V3, HUB_REE_V3, HUB_QUEUE_REF))
    ok, out, err = ssh(hub_ssh, ssh_user, qcmd, dry_run=False)
    if not ok:
        return "FAIL", "ssh/git show failed: %s" % (err or "?")
    try:
        qdata = json.loads(out or "")
    except ValueError as exc:
        return "FAIL", "queue JSON parse failed: %r" % exc
    file_items = qdata.get("items") or []
    sql = ("SELECT queue_id FROM experiments "
           "WHERE status IN ('pending','claimed');")
    remote = ("sqlite3 -batch %s %s 2>/dev/null"
              % (HUB_COORDINATOR_DB, repr(sql)))
    ok, out2, err = ssh(hub_ssh, ssh_user, remote, dry_run=False)
    if not ok:
        return "FAIL", "ssh/sqlite3 failed: %s" % (err or "?")
    db_ids = [r.strip() for r in (out2 or "").splitlines() if r.strip()]
    file_only, db_only = _diff_queue_against_db(file_items, db_ids)
    if not file_only and not db_only:
        return ("PASS",
                "queue snapshot in sync (%d item(s) on both sides)"
                % len(db_ids))
    parts = []
    if file_only:
        parts.append("on file but not in DB: %s" % file_only[:5])
    if db_only:
        parts.append("in DB but not on file: %s" % db_only[:5])
    return "FAIL", "snapshot stale -- " + "; ".join(parts)


def check_derived_heartbeats(hub_ssh, ssh_user, *, ssh,
                             lookback_sec=DEFAULT_HEARTBEAT_LOOKBACK_SEC,
                             now=None):
    """Heartbeat + status files on origin/master updated recently AND
    only by writer-authored commits."""
    if now is None:
        now = int(datetime.now(timezone.utc).timestamp())
    # 1) Most recent commit time touching either subtree.
    paths = (
        "evidence/experiments/runner_heartbeats/",
        "evidence/experiments/runner_status/",
    )
    cmd = (
        "git -C %s fetch --quiet origin master 2>/dev/null && "
        "git -C %s log -1 --format=%%ct origin/master -- %s %s"
    ) % (HUB_REE_ASSEMBLY, HUB_REE_ASSEMBLY, paths[0], paths[1])
    ok, out, err = ssh(hub_ssh, ssh_user, cmd, dry_run=False)
    if not ok:
        return "FAIL", "ssh/git log failed: %s" % (err or "?")
    epoch_str = (out or "").strip().splitlines()[:1]
    if not epoch_str:
        return "FAIL", (
            "no commits ever touched runner_heartbeats/ or runner_status/")
    if not _is_fresh(epoch_str[0], now, lookback_sec):
        try:
            age = now - int(epoch_str[0])
        except ValueError:
            age = "?"
        return "FAIL", (
            "latest heartbeat/status commit is %ss old (> %ds)"
            % (age, lookback_sec))
    # 2) The N most recent commits touching these subtrees must all be
    #    writer-authored. A runner-pushed commit there means the gate
    #    is leaking.
    cmd2 = ("git -C %s log -10 --format=%%s origin/master -- %s %s"
            % (HUB_REE_ASSEMBLY, paths[0], paths[1]))
    ok, out, err = ssh(hub_ssh, ssh_user, cmd2, dry_run=False)
    if not ok:
        return "FAIL", "ssh/git log subjects failed: %s" % (err or "?")
    writer, foreign = _classify_commits((out or "").splitlines())
    if foreign:
        return "FAIL", (
            "%d non-writer commit(s) touched heartbeat/status files: %r"
            % (len(foreign), foreign[:3]))
    return "PASS", (
        "heartbeat/status fresh (latest commit within %ds, "
        "%d writer-authored)" % (lookback_sec, len(writer)))


# --------------------------------------------------------------------------
# Runner: orchestrates the above into a structured summary.
# --------------------------------------------------------------------------

def _import_writer_state():
    try:
        import sync_daemon  # noqa: WPS433
        ready = bool(getattr(sync_daemon, "PHASE3_GIT_WRITER_READY", False))
        if ready:
            return True, "PHASE3_GIT_WRITER_READY True"
        return False, "PHASE3_GIT_WRITER_READY False (writer still stub)"
    except Exception as exc:  # noqa: BLE001
        return False, "import failed: %r" % exc


def _stub_category(cid):
    if cid.startswith("hub_") or cid == "sync_daemon_phase3_tick":
        return "hub"
    if cid.startswith("workers_") or cid.startswith("heartbeat_"):
        return "fleet"
    if cid in ("results_drained", "queue_snapshot_fresh"):
        return "data"
    if cid == "derived_heartbeats":
        return "explorer"
    return "hub"


def run_verify(
    *,
    env_file=None,
    dry_run=False,
    expect_cutover=False,
    mock=False,
    ssh_runner=None,
):
    from phase3_preflight import (  # noqa: WPS433
        DEFAULT_SSH_HOSTS,
        _http_get,
        _load_env_file,
        _run_check_shadow,
        _ssh_run,
    )
    ssh = ssh_runner or _ssh_run

    env_path = env_file or Path(
        os.environ.get("COORDINATOR_ENV_FILE", str(DEFAULT_ENV)))
    cfg = _load_env_file(env_path) if env_path.exists() else {}
    url = cfg.get("COORDINATOR_URL") or os.environ.get("COORDINATOR_URL", "")
    token = (cfg.get("COORDINATOR_LOCAL_TOKEN")
             or os.environ.get("COORDINATOR_TOKEN", ""))
    ssh_user = cfg.get("COORDINATOR_SSH_USER", "ree")
    hub_ssh = (cfg.get("SHADOW_SSH_HOST_ree-cloud-1")
               or DEFAULT_SSH_HOSTS["ree-cloud-1"])

    # Worker SSH targets: cloud workers come from coordinator.env overrides
    # or the DEFAULT_SSH_HOSTS map; the Mac is treated specially.
    worker_targets = []
    for w in CLOUD_WORKERS:
        host = (cfg.get("SHADOW_SSH_HOST_" + w) or DEFAULT_SSH_HOSTS.get(w))
        if host:
            worker_targets.append(host)
    worker_targets.append(LOCAL_WORKER)

    checks = []

    def add(cid, cat, status, msg, **detail):
        rec = {"id": cid, "category": cat, "status": status, "message": msg}
        if detail:
            rec["detail"] = detail
        checks.append(rec)

    writer_ready, writer_note = _import_writer_state()

    if not expect_cutover:
        add("cutover_expected", "meta", "SKIP",
            "pass --expect-cutover after maintenance window")

    # --- hub_sync_mode_authoritative ---
    if dry_run or mock:
        add("hub_sync_mode_authoritative", "hub", "SKIP",
            "dry-run/mock: skip hub env SSH")
    else:
        ok, out, err = ssh(
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

    # If we are not in expect-cutover mode, all 7 post-cutover checks SKIP.
    # In mock or dry-run mode they also SKIP (no live signal to read).
    suppress = (not expect_cutover) or dry_run or mock
    if suppress:
        for cid in (
            "sync_daemon_phase3_tick",
            "hub_git_writer_only",
            "workers_no_result_git_push",
            "heartbeat_git_retired",
            "results_drained",
            "queue_snapshot_fresh",
            "derived_heartbeats",
        ):
            if mock:
                msg = "mock: skip live signal"
            elif dry_run:
                msg = "dry-run: skip live signal"
            else:
                msg = "awaits --expect-cutover after maintenance window"
            add(cid, _stub_category(cid), "SKIP", msg)
    else:
        # 1. sync_daemon_phase3_tick
        st, msg = check_sync_daemon_phase3_tick(hub_ssh, ssh_user, ssh=ssh)
        add("sync_daemon_phase3_tick", "hub", st, msg)
        # 2. hub_git_writer_only
        st, msg = check_hub_git_writer_only(hub_ssh, ssh_user, ssh=ssh)
        add("hub_git_writer_only", "hub", st, msg)
        # 3. workers_no_result_git_push
        st, msg, detail = check_workers_no_result_git_push(
            worker_targets, ssh_user, ssh=ssh)
        add("workers_no_result_git_push", "fleet", st, msg,
            per_worker=detail)
        # 4. heartbeat_git_retired
        st, msg, detail = check_heartbeat_git_retired(
            worker_targets, ssh_user, ssh=ssh)
        add("heartbeat_git_retired", "fleet", st, msg, per_worker=detail)
        # 5. results_drained
        st, msg = check_results_drained(hub_ssh, ssh_user, ssh=ssh)
        add("results_drained", "data", st, msg)
        # 6. queue_snapshot_fresh
        st, msg = check_queue_snapshot_fresh(hub_ssh, ssh_user, ssh=ssh)
        add("queue_snapshot_fresh", "data", st, msg)
        # 7. derived_heartbeats
        st, msg = check_derived_heartbeats(hub_ssh, ssh_user, ssh=ssh)
        add("derived_heartbeats", "explorer", st, msg)

    # Claims path should stay healthy (same as Phase 2).
    if mock:
        add("claims_still_healthy", "soak", "SKIP",
            "mock: skip check_shadow")
    elif url and token:
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
        "mock": mock,
        "writer_ready": writer_ready,
        "writer_note": writer_note,
        "fail_count": len(fail),
        "skip_count": len(skip),
        "checks": checks,
    }


def _print_report(summary):
    print("Phase 3 verify @ %s" % summary.get("checked_at", "?"))
    if summary.get("expect_cutover"):
        print("  (--expect-cutover: post-cutover mode)")
    if summary.get("dry_run"):
        print("  (--dry-run: SSH skipped)")
    if summary.get("mock"):
        print("  (--mock: live checks skipped)")
    for c in summary.get("checks", []):
        print("  [%s] %s/%s: %s" % (
            c["status"], c["category"], c["id"], c["message"]))
    print("")
    if summary.get("ok"):
        print("VERDICT: PASS -- no blocking FAIL checks")
    else:
        print("VERDICT: FAIL -- %d blocking check(s)"
              % summary.get("fail_count", 0))


def main():
    ap = argparse.ArgumentParser(
        description="Phase 3 post-cutover verification")
    ap.add_argument("--env-file", default=str(DEFAULT_ENV))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true",
                    help="skip live SSH/HTTP (tests only)")
    ap.add_argument("--expect-cutover", action="store_true",
                    help="enable post-cutover required checks (not SKIP)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    summary = run_verify(
        env_file=Path(args.env_file).expanduser(),
        dry_run=args.dry_run,
        expect_cutover=args.expect_cutover,
        mock=args.mock,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_report(summary)

    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
