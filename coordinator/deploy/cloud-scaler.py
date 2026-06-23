"""Cloud Worker Scaler -- systemd-timer-driven, hub-resident.

Auto-starts each Hetzner cloud worker when experiments matching its
machine_affinity are queued, stops it when no matching work remains.

This is the 1:1 Python rewrite of ree-v3/.github/workflows/cloud-scaler.yml.
It runs on the coordinator hub VM (ree-worker-1, 10.8.0.1) as a systemd
oneshot driven by cloud-scaler.timer (OnCalendar=*:0/5), reading the same
queue + heartbeat files that the GHA workflow used to sparse-checkout.
Behaviour is intentionally identical to the bash version -- the migration
is a transport change (GHA -> hub timer), not a logic change.

Routing rules (per worker, mirroring cloud-scaler.yml):
  - "claimable" items = status==pending AND machine_affinity in {any, <server_affinity>}
  - mode "full" (default): server off + claimable > 0 -> poweron
  - mode "shutdown-only": never auto-starts; must be powered on manually
  - mode "surge": auto-start only when claimable >= SURGE_QUEUE_THRESHOLD
      AND cloud-2 + cloud-3 are both running with held claims.
  - server running + claimable == 0 + heartbeat says idle + last completed
      > IDLE_GRACE_MIN ago -> shutdown (graceful, drains current run).

Authoritative invariants preserved from the bash version (DO NOT REMOVE):
  (1) HUB_NAME guard -- ree-worker-1 is skipped before any read or decision.
      The 2026-05-28 incident this guards against is in the YAML header.
  (2) HELD_BY_SELF veto -- a worker holding any status==claimed item where
      claimed_by.machine == its affinity is NEVER shut down, even when the
      heartbeat is stale and idle_ok=1. This is the 2026-05-30 fleet
      incident guard.
  (3) Surge sister-worker pre-check -- cloud-4 is considered AFTER cloud-2
      and cloud-3 in the WORKERS list so their power state and held-count
      are available when cloud-4's surge decision is made.
  (4) HEARTBEAT_FRESH_MIN >= 35 -- the FALLBACK floor for the git heartbeat
      mirror. Since 2026-06-23 freshness/state primarily comes from the
      coordinator /shadow/status (DB-authoritative, sub-second fresh), so
      heartbeat_aging effectively never trips for a live worker. The floor
      still guards the git-fallback path (coordinator unreachable), where
      the file is refreshed on state-changes only. Tighter values regress
      to the heartbeat-aging bug fixed 2026-05-31 on that fallback path.

All printed output is ASCII-only.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone


# Hub-resident paths. The sync_daemon keeps these refreshed against
# origin/main (queue file) and origin/master (heartbeats); both refer to
# the local working tree which the writers update at every commit cycle.
DEFAULT_QUEUE_PATH = "/home/ree/REE_Working/ree-v3/experiment_queue.json"
DEFAULT_HEARTBEATS_DIR = (
    "/home/ree/REE_Working/REE_assembly/evidence/experiments/runner_heartbeats"
)
DEFAULT_ANNOUNCE_SCRIPT = "/usr/local/bin/coordinator_announce_shutdown.sh"

# Phase-3 telemetry transport (2026-06-23). Freshness/state/current_exq now
# come from the coordinator DB over WireGuard (COORDINATOR_URL/shadow/status,
# DB-authoritative, no git-commit lag) instead of the git-materialised
# heartbeat file. This is what let the sync_daemon retire its 30-minute
# forced "liveness tick" commit -- the single biggest source of REE_assembly
# git-history churn. The git heartbeat mirror is kept as a fallback when the
# coordinator is unreachable (bit-identical to the pre-Phase-3 transport) and
# is still the source for the recent_completed grace window. Both values are
# read from the scaler's existing EnvironmentFile (/etc/ree-coordinator.env).
DEFAULT_COORDINATOR_URL = "http://10.8.0.1:8787"
COORDINATOR_FETCH_TIMEOUT_SECONDS = 6.0


# Defaults match cloud-scaler.yml env block exactly.
DEFAULTS = {
    "IDLE_GRACE_MIN": 20,
    # HEARTBEAT_FRESH_MIN was 5 (assumed every-tick file refresh). Under
    # the phase3 sync_daemon heartbeat-writer redesign (2026-05-31), the
    # heartbeat file on REE_assembly is refreshed on state-changes only
    # with a 5-min debounce and a 30-min liveness floor; an idle-but-
    # alive worker's file can legitimately lag 5-30 minutes behind
    # reality. 35 sits just above the liveness floor. Do NOT drop below
    # 35 -- regresses to the heartbeat_aging idle-stall bug.
    "HEARTBEAT_FRESH_MIN": 35,
    # 2026-06-01: lowered 3->2 so a 2-deep "any"-affinity backlog wakes
    # cloud-4 as overflow before the laptop (DLAPTOP-4.local) is pulled
    # in. The sister-worker pre-check (cloud-2 + cloud-3 both running with
    # held claims) still gates the power-on, so a 2-deep queue only surges
    # cloud-4 when cloud-2/3 are already saturated.
    "SURGE_QUEUE_THRESHOLD": 2,
    "HUB_NAME": "ree-worker-1",
}


# server_name:affinity[:mode]. Mode "shutdown-only" = auto-shutdown when
# idle but never auto-start. Mode "surge" = auto-start only when cloud-2
# AND cloud-3 are saturated AND claimable >= SURGE_QUEUE_THRESHOLD.
# IMPORTANT: ordering matters -- cloud-2 and cloud-3 must iterate before
# cloud-4 so cloud-4's surge branch can read sister-worker state.
WORKERS = [
    ("ree-worker-1", "ree-cloud-1", "full"),
    ("ree-worker-2", "ree-cloud-2", "full"),
    ("ree-worker-3", "ree-cloud-3", "full"),
    ("ree-worker-4", "ree-cloud-4", "surge"),
]


def log(msg):
    """Single-line ASCII output to stdout; systemd journal captures it."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sys.stdout.write("[%s] %s\n" % (ts, msg))
    sys.stdout.flush()


def parse_utc(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def load_queue(path):
    """Return parsed queue dict; empty dict on missing file (treat as
    no work). Matches the GHA path of bailing if checkout missed."""
    if not os.path.exists(path):
        log("ERROR queue file not found at %s -- skipping tick" % path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        log("ERROR queue file unreadable at %s: %r -- skipping tick"
            % (path, exc))
        return None


def count_claimable(queue, affinity):
    """status==pending AND machine_affinity in {any, <affinity>}.

    Mirrors the bash inline python:
        n = sum(
            1 for i in q.get('items', [])
            if i.get('status') == 'pending'
            and i.get('machine_affinity') in ('any', aff)
        )
    """
    n = 0
    for item in queue.get("items", []):
        if item.get("status") != "pending":
            continue
        if item.get("machine_affinity") not in ("any", affinity):
            continue
        n += 1
    return n


def count_held_by_self(queue, affinity):
    """status==claimed AND claimed_by.machine == affinity. Vetos shutdown
    even when the pending count is zero and heartbeat is stale -- a
    worker mid-experiment that has already moved its item from pending
    to claimed is invisible to the pending-only CLAIMABLE count.

    Accepts both nested form ({machine, claimed_at}) and a bare string
    form for legacy entries, matching the bash version.
    """
    n = 0
    for item in queue.get("items", []):
        if item.get("status") != "claimed":
            continue
        cb = item.get("claimed_by") or {}
        if isinstance(cb, dict):
            machine = cb.get("machine")
        else:
            machine = cb
        if machine == affinity:
            n += 1
    return n


def fetch_coordinator_status(coordinator_url, coordinator_token,
                             timeout=COORDINATOR_FETCH_TIMEOUT_SECONDS):
    """Return {machine: row} from the coordinator /shadow/status, or {} when
    unconfigured / unreachable / unauthorised. NEVER raises -- a coordinator
    blip must degrade the scaler to the git-mirror fallback, not crash the
    tick. Each row carries the DB-authoritative last_seen / state /
    current_exq the scaler needs (the same columns the git mirror is
    materialised from, but without the commit lag the liveness tick used to
    paper over).
    """
    url = (coordinator_url or "").rstrip("/")
    if not url or not coordinator_token:
        return {}
    try:
        req = urllib.request.Request(
            url + "/shadow/status",
            headers={"Authorization": "Bearer " + coordinator_token},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            doc = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError,
            json.JSONDecodeError, ValueError) as exc:
        log("WARN coordinator /shadow/status fetch failed (%r) -- falling "
            "back to git heartbeat mirror for this tick" % exc)
        return {}
    out = {}
    for m in doc.get("machines") or []:
        name = m.get("machine")
        if name:
            out[name] = m
    return out


def _read_git_heartbeat(heartbeats_dir, affinity):
    """Read the git-materialised heartbeat file for one affinity.

    Returns the parsed dict, None when the file is absent, or a sentinel
    {"_unreadable": "<repr>"} when it exists but cannot be parsed.
    """
    path = os.path.join(heartbeats_dir, "%s.json" % affinity)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return {"_unreadable": repr(exc)}


def evaluate_heartbeat(heartbeats_dir, affinity, idle_grace_min,
                       heartbeat_fresh_min, now=None, coord_row=None):
    """Return (idle_ok: 0|1, reason: str) describing whether the runner
    is genuinely idle and the shutdown grace window has expired.

    Decision tree (unchanged from the bash original):
      - heartbeat missing -> idle_ok=1 reason=no_heartbeat
      - tick > 1h stale   -> idle_ok=1 reason=heartbeat_stale (dead)
      - tick > fresh_min  -> idle_ok=0 reason=heartbeat_aging (uncertain)
      - state != idle or current_exq set -> idle_ok=0 reason=state_...
      - last completed within grace -> idle_ok=0 reason=grace_window
      - else -> idle_ok=1 reason=clean_idle

    Transport (2026-06-23): when coord_row (a /shadow/status machine row) is
    supplied, freshness (last_seen), state, and current_exq are read from the
    coordinator DB -- always current, no git-commit lag -- which is why the
    sync_daemon no longer forces a 30-minute liveness commit. The
    recent_completed grace window still reads the git mirror (result and
    state-change commits keep it fresh; the coordinator snapshot carries no
    completion history). When coord_row is None (coordinator unreachable / not
    configured), the whole decision falls back to the git heartbeat file --
    bit-identical to the pre-Phase-3 transport.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    git_hb = _read_git_heartbeat(heartbeats_dir, affinity)

    if coord_row is not None:
        last_tick = parse_utc(coord_row.get("last_seen"))
        state = coord_row.get("state")
        current_exq = coord_row.get("current_exq")
    elif git_hb is None:
        return 1, "no_heartbeat"
    elif "_unreadable" in git_hb:
        # Treat as no heartbeat rather than crashing -- best-effort tick.
        return 1, "no_heartbeat_unreadable(%s)" % git_hb["_unreadable"]
    else:
        last_tick = parse_utc(git_hb.get("last_tick_utc"))
        state = git_hb.get("state")
        current_exq = git_hb.get("current_exq")

    # recent_completed grace window -- always from the git mirror. It is
    # fed by result + state-change commits (which continue), and a
    # completion timestamp does not go stale, so it stays correct even
    # though the file's own last_tick no longer advances every 30 minutes.
    completed = []
    if git_hb and "_unreadable" not in git_hb:
        completed = git_hb.get("recent_completed") or []
    last_completed = None
    for c in completed:
        t = parse_utc(c.get("completed_at"))
        if t and (last_completed is None or t > last_completed):
            last_completed = t

    if last_tick is None or (now - last_tick) > timedelta(hours=1):
        return 1, "heartbeat_stale"
    if (now - last_tick) > timedelta(minutes=heartbeat_fresh_min):
        return 0, "heartbeat_aging"
    if state != "idle" or current_exq is not None:
        return 0, "state_%s_current_%s" % (state, current_exq)
    if last_completed is not None and (now - last_completed) < timedelta(
        minutes=idle_grace_min
    ):
        age_min = int((now - last_completed).total_seconds() // 60)
        return 0, "grace_window_age=%dmin" % age_min
    return 1, "clean_idle"


def hcloud_describe_status(server_name, dry_run=False):
    """Return one of {"running", "off", "starting", "stopping", "unknown"}.

    Bash uses:
        hcloud server describe "$SERVER_NAME" -o format='{{.Status}}'
    """
    try:
        out = subprocess.run(
            ["hcloud", "server", "describe", server_name,
             "-o", "format={{.Status}}"],
            check=True, capture_output=True, timeout=30,
        )
        status = out.stdout.decode("utf-8").strip()
        return status or "unknown"
    except subprocess.CalledProcessError as exc:
        # The bash version swallowed errors as "unknown"; do the same so
        # a single unprovisioned server can't take down the whole tick.
        err = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        log("WARN hcloud describe %s failed (%s)" % (server_name, err))
        return "unknown"
    except Exception as exc:  # noqa: BLE001
        log("WARN hcloud describe %s exception: %r" % (server_name, exc))
        return "unknown"


def hcloud_poweron(server_name, dry_run=False):
    if dry_run:
        log("[DRY] would poweron %s" % server_name)
        return
    try:
        subprocess.run(["hcloud", "server", "poweron", server_name],
                       check=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        log("ERROR hcloud poweron %s failed: %r" % (server_name, exc))


def hcloud_shutdown(server_name, dry_run=False):
    """Graceful shutdown -- systemd sends SIGTERM, runner can drain."""
    if dry_run:
        log("[DRY] would shutdown %s" % server_name)
        return
    try:
        subprocess.run(["hcloud", "server", "shutdown", server_name],
                       check=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        log("ERROR hcloud shutdown %s failed: %r" % (server_name, exc))


def announce_shutdown(affinity, announce_script, dry_run=False):
    """Best-effort call to coordinator_announce_shutdown.sh -- flips
    /shadow/status lifecycle_state to gracefully_offline before the VM
    powers off. Failure does NOT block the underlying hcloud shutdown.
    """
    if not os.path.exists(announce_script):
        log("[scaler] announce script %s missing -- skipping for %s"
            % (announce_script, affinity))
        return
    if dry_run:
        log("[DRY] would announce shutdown_notify for %s via %s"
            % (affinity, announce_script))
        return
    try:
        subprocess.run([announce_script, affinity],
                       check=True, timeout=15)
        log("    [scaler] announced shutdown_notify for %s" % affinity)
    except Exception as exc:  # noqa: BLE001
        log("    [scaler] WARN: shutdown_notify announce failed for %s "
            "(%r) (proceeding)" % (affinity, exc))


def run_once(queue_path, heartbeats_dir, announce_script,
             idle_grace_min, heartbeat_fresh_min, surge_queue_threshold,
             hub_name, workers, dry_run=False,
             coordinator_url=None, coordinator_token=None):
    """One pass over the WORKERS list. Mirrors the bash for-loop body
    one-to-one. Returns 0 on success."""
    queue = load_queue(queue_path)
    if queue is None:
        return 1

    # Coordinator telemetry snapshot (one fetch per tick, shared across all
    # workers). {} when unconfigured/unreachable -> every worker degrades to
    # the git heartbeat mirror, exactly the pre-Phase-3 behaviour.
    coord_status = fetch_coordinator_status(coordinator_url, coordinator_token)
    if coord_status:
        log("telemetry source=coordinator (%d machine(s) in /shadow/status)"
            % len(coord_status))
    else:
        log("telemetry source=git_mirror_fallback (coordinator "
            "unconfigured/unreachable this tick)")

    # Per-affinity state for the surge-mode sister-worker pre-check.
    # The WORKERS list is ordered so cloud-2 and cloud-3 are populated
    # before cloud-4 reads from this dict.
    worker_status = {}  # affinity -> hcloud status
    worker_held = {}    # affinity -> held_by_self count

    for server_name, affinity, mode in workers:
        # HUB GUARD -- skip the coordinator VM before any read or decision.
        # The 2026-05-28 incident hardcoded this: cloud-1's runner was
        # systemctl disabled (separate hub/runner co-tenancy fix), the
        # scaler saw the VM as an idle worker, shut it down, and took
        # the coordinator + sync_daemon offline. Recovery required manual
        # `hcloud server poweron ree-worker-1`.
        if server_name == hub_name:
            log("[%s affinity=%s] HUB_NAME match -- scaler skips hub VM "
                "(no power-on, no shutdown)" % (server_name, affinity))
            continue

        claimable = count_claimable(queue, affinity)
        held_by_self = count_held_by_self(queue, affinity)
        coord_row = coord_status.get(affinity)
        idle_ok, idle_reason = evaluate_heartbeat(
            heartbeats_dir, affinity, idle_grace_min, heartbeat_fresh_min,
            coord_row=coord_row,
        )
        hb_src = "coord" if coord_row is not None else "git"
        status = hcloud_describe_status(server_name, dry_run=dry_run)

        # Stash for the surge branch on a later worker (cloud-4).
        worker_status[affinity] = status
        worker_held[affinity] = held_by_self

        log("[%s affinity=%s] claimable=%d held_by_self=%d status=%s "
            "idle_ok=%d reason=%s hb_src=%s"
            % (server_name, affinity, claimable, held_by_self, status,
               idle_ok, idle_reason, hb_src))

        if status == "unknown":
            log("  -> server not provisioned yet, skipping")
            continue

        # Decision matrix -- ORDERED, identical to the bash if/elif chain.
        if claimable > 0 and status == "off" and mode == "full":
            log("  -> queue has work, starting %s" % server_name)
            hcloud_poweron(server_name, dry_run=dry_run)

        elif status == "off" and mode == "surge":
            # Surge mode: cloud-4 powers on only when cloud-2 + cloud-3
            # are both running with claims held AND queue depth >=
            # threshold. Designed to make cloud-4 the overflow before
            # the laptop is ever pulled in. Sister-worker state read
            # from worker_status / worker_held populated above (cloud-2
            # and cloud-3 iterate first by WORKERS ordering).
            cloud_2_status = worker_status.get("ree-cloud-2", "unknown")
            cloud_2_held = worker_held.get("ree-cloud-2", 0)
            cloud_3_status = worker_status.get("ree-cloud-3", "unknown")
            cloud_3_held = worker_held.get("ree-cloud-3", 0)
            if (claimable >= surge_queue_threshold
                    and cloud_2_status == "running" and cloud_2_held > 0
                    and cloud_3_status == "running" and cloud_3_held > 0):
                log("  -> surge conditions met (claimable=%d>=%d, "
                    "cloud-2 running+held=%d, cloud-3 running+held=%d), "
                    "starting %s"
                    % (claimable, surge_queue_threshold,
                       cloud_2_held, cloud_3_held, server_name))
                hcloud_poweron(server_name, dry_run=dry_run)
            else:
                log("  -> surge mode but conditions not met "
                    "(claimable=%d/threshold=%d, cloud-2=%s/held=%d, "
                    "cloud-3=%s/held=%d)"
                    % (claimable, surge_queue_threshold,
                       cloud_2_status, cloud_2_held,
                       cloud_3_status, cloud_3_held))

        elif (claimable > 0 and status == "off"
                and mode == "shutdown-only"):
            log("  -> queue has work but %s is shutdown-only "
                "(manual start required)" % server_name)

        elif held_by_self > 0 and status == "running":
            # HELD_BY_SELF VETO. Authoritative invariant: a worker
            # holding a claimed-and-not-completed item is NEVER shut
            # down, even when CLAIMABLE=0 and IDLE_OK=1 (e.g. heartbeat
            # stale because a long experiment hasn't refreshed it).
            # 2026-05-30 fleet incident guard.
            log("  -> worker holds %d active claim(s), keeping %s running"
                % (held_by_self, server_name))

        elif (claimable == 0 and status == "running" and idle_ok == 1):
            log("  -> no matching work AND runner idle past grace "
                "window, shutting down %s" % server_name)
            # Announce BEFORE hcloud shutdown so lifecycle_state flips
            # to gracefully_offline immediately rather than waiting on
            # the runner SIGTERM handler (which would miss this signal
            # if the runner is already crashed). Best-effort: failure
            # does not block shutdown.
            announce_shutdown(affinity, announce_script, dry_run=dry_run)
            # "shutdown" (not poweroff) -> SIGTERM for graceful drain.
            hcloud_shutdown(server_name, dry_run=dry_run)

        else:
            log("  -> no action needed")

    return 0


def env_int(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:  # noqa: BLE001
        log("WARN %s=%r not parseable as int, using default %d"
            % (name, v, default))
        return default


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cloud worker scaler (hub-resident).",
    )
    parser.add_argument("--queue", default=DEFAULT_QUEUE_PATH,
                        help="experiment_queue.json path")
    parser.add_argument("--heartbeats-dir", default=DEFAULT_HEARTBEATS_DIR,
                        help="runner_heartbeats directory path")
    parser.add_argument("--announce-script", default=DEFAULT_ANNOUNCE_SCRIPT,
                        help="coordinator_announce_shutdown.sh path")
    parser.add_argument("--dry-run", action="store_true",
                        help=("read state and log decisions, but do NOT "
                              "issue hcloud poweron/shutdown or announce"))
    parser.add_argument("--once", action="store_true",
                        help=("run a single pass and exit (default; "
                              "preserved for clarity in systemd ExecStart)"))
    args = parser.parse_args(argv)

    # Env overrides match the GHA env block. systemd EnvironmentFile or
    # operator-set values take precedence over the in-file DEFAULTS.
    idle_grace_min = env_int("IDLE_GRACE_MIN", DEFAULTS["IDLE_GRACE_MIN"])
    heartbeat_fresh_min = env_int(
        "HEARTBEAT_FRESH_MIN", DEFAULTS["HEARTBEAT_FRESH_MIN"])
    surge_queue_threshold = env_int(
        "SURGE_QUEUE_THRESHOLD", DEFAULTS["SURGE_QUEUE_THRESHOLD"])
    hub_name = os.environ.get("HUB_NAME") or DEFAULTS["HUB_NAME"]

    # Coordinator telemetry transport (read from the same EnvironmentFile,
    # /etc/ree-coordinator.env, that already supplies COORDINATOR_URL +
    # COORDINATOR_SCALER_TOKEN to this service). When either is absent the
    # scaler silently uses the git heartbeat mirror.
    coordinator_url = (os.environ.get("COORDINATOR_URL")
                       or DEFAULT_COORDINATOR_URL)
    coordinator_token = os.environ.get("COORDINATOR_SCALER_TOKEN") or ""

    # Floor on HEARTBEAT_FRESH_MIN: tighter values regress to the
    # heartbeat-aging idle-stall bug fixed 2026-05-31. If an operator
    # sets a value below 35, clamp and warn.
    if heartbeat_fresh_min < 35:
        log("WARN HEARTBEAT_FRESH_MIN=%d below 35-minute floor; clamping "
            "to 35 (phase3 heartbeat-writer 30-minute liveness floor + "
            "5-minute headroom)" % heartbeat_fresh_min)
        heartbeat_fresh_min = 35

    if args.dry_run:
        log("DRY-RUN MODE: state-read and decisions only; no hcloud "
            "or announce calls will fire")

    if not os.environ.get("HCLOUD_TOKEN") and not args.dry_run:
        log("WARN HCLOUD_TOKEN not set in environment -- hcloud calls "
            "will likely fail; check /etc/ree-coordinator.env")

    log("cloud-scaler tick: queue=%s heartbeats=%s coordinator=%s(token:%s) "
        "idle_grace=%d heartbeat_fresh=%d surge_threshold=%d hub=%s"
        % (args.queue, args.heartbeats_dir, coordinator_url,
           "set" if coordinator_token else "MISSING",
           idle_grace_min, heartbeat_fresh_min, surge_queue_threshold,
           hub_name))

    rc = run_once(
        queue_path=args.queue,
        heartbeats_dir=args.heartbeats_dir,
        announce_script=args.announce_script,
        idle_grace_min=idle_grace_min,
        heartbeat_fresh_min=heartbeat_fresh_min,
        surge_queue_threshold=surge_queue_threshold,
        hub_name=hub_name,
        workers=WORKERS,
        dry_run=args.dry_run,
        coordinator_url=coordinator_url,
        coordinator_token=coordinator_token,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
