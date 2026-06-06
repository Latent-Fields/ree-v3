"""Cross-machine remote-control surface for the V3 experiment runner.

Phase 1: heartbeats.
  - write_heartbeat() emits a JSON snapshot of this runner's state to
    REE_assembly/evidence/experiments/runner_heartbeats/<hostname>.json
    on every loop tick when --remote-control is active.
  - The file is auto-synced to GitHub by the existing --auto-sync push
    path (callers pass ree_assembly_path so we can do a focused commit).
  - Other machines / serve.py read these files via /api/machines.

Phase 2: command channel.
  - read_commands_file() / write_commands_file() manage per-machine
    REE_assembly/evidence/experiments/runner_commands/<hostname>.json.
  - process_pending_commands() is called from the runner main loop. It
    drains pending commands and mutates the runner's drain / pause /
    queue state in response. Acked commands are pushed back through the
    same auto-sync git path so the explorer and other sessions see the
    completion record.
  - append_command() is used by serve.py POST /api/machines/<host>/command
    (and equivalents from other authorised callers) to enqueue work.

Supported command kinds (phase 2):
  stop          -- graceful drain after current experiment.
  force_stop    -- kill current experiment process and exit.
  pause         -- runner stays alive but skips picking up new experiments.
  resume        -- clear pause state.
  suspend       -- terminate current experiment; keep partial checkpoint if any.
  resume_run    -- clear pause; on next pick, resume checkpointed EXQ (--resume).
  kick          -- args.queue_id moved to head of queue (front-of-line).
  release_claim -- args.queue_id has its claimed_by cleared (recovery).

`start` is NOT supported by this channel because a stopped runner cannot
read its own command file. For the local machine, use serve.py's existing
/api/runner/v3/start. Remote machines need their own serve.py or a
supervisor process (out of scope for phase 2).

ASCII-only in print() output (Windows cp1252 compatibility, per CLAUDE.md).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

# Coordinator shadow shim. Env-gated no-op unless COORDINATION_MODE=shadow;
# guarded import so it can never break heartbeat writing.
try:
    import coordinator_client
except Exception:  # pragma: no cover -- shim must never break the runner
    class _NoCoordClient:
        def __getattr__(self, _name):
            return lambda *a, **k: None
    coordinator_client = _NoCoordClient()


HEARTBEAT_SUBPATH = Path("evidence") / "experiments" / "runner_heartbeats"
COMMANDS_SUBPATH = Path("evidence") / "experiments" / "runner_commands"

VALID_COMMAND_KINDS = (
    "stop", "force_stop", "pause", "resume", "suspend", "resume_run",
    "kick", "release_claim",
    # reclassify: fix a runner_status entry that was misclassified at
    # ingest time (canonical example: V3-EXQ-517c, runner declared
    # ERROR despite the script emitting "outcome: PASS" on stdout). Args:
    #   queue_id  (str, required) -- entry to mutate in status["completed"]
    #   result    (str, required) -- new value: "PASS" | "FAIL" | "ERROR"
    #   output_file (str, optional) -- repo-relative manifest path to record
    #   note      (str, optional) -- free-text added to result_summary
    # Effect: in-memory status["completed"] entry is mutated, local
    # runner_status.json is rewritten atomically, the next heartbeat tick
    # POSTs the corrected status_payload_json to the coordinator, the hub's
    # phase3_heartbeat_writer materialises the corrected per-machine file
    # on origin/master.
    "reclassify",
)

# Keep the last N done/failed commands in the file; older ones are pruned to
# bound file growth across long-running machines.
MAX_HISTORY_PER_FILE = 50

# Tail of live stdout lines carried in each per-machine heartbeat (the explorer's
# scrollable card readout). Kept as its own constant -- separate from the runner's
# RECENT_LINES_DISPLAY -- because this payload is git-committed every minute by the
# Phase-3 heartbeat writer, so its depth is the git/DB-growth knob for remote cards.
HEARTBEAT_RECENT_LINES = 100


def _now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def get_machine_id(override: str | None = None) -> str:
    return override or socket.gethostname()


def heartbeat_path(ree_assembly_path: Path, machine: str) -> Path:
    return ree_assembly_path / HEARTBEAT_SUBPATH / f"{machine}.json"


def _safe_filename(machine: str) -> str:
    keep = "-_."
    return "".join(c if (c.isalnum() or c in keep) else "_" for c in machine)


def _gpu_info() -> dict[str, Any]:
    """Best-effort GPU summary. Returns empty dict if torch unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return {
            "available": True,
            "device_name": props.name,
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "device_index": idx,
        }
    except Exception:
        return {"available": False}


def write_heartbeat(
    ree_assembly_path: Path,
    machine: str,
    state: str,
    *,
    current_exq: str | None = None,
    current_exq_started_utc: str | None = None,
    queue_depth: int | None = None,
    queue_id_at_head: str | None = None,
    recent_completed: list[dict] | None = None,
    runner_pid: int | None = None,
    runner_version: str | None = None,
    extra: dict | None = None,
    current_title: str | None = None,
    current_claim_id: str | None = None,
    current_description: str | None = None,
    progress: dict | None = None,
    seconds_elapsed: int | None = None,
    seconds_remaining: int | None = None,
    recent_lines: list[str] | None = None,
) -> Path | None:
    """Write a heartbeat snapshot for this runner. Never raises.

    Returns the path written, or None on failure.

    The progress / recent_lines / *_seconds / current_* fields let the
    explorer render a per-machine "Now Running" card with a progress bar
    for cloud workers (parity with the local card sourced from the
    per-machine runner_status.json file). They are all optional; idle-tick
    heartbeats omit them and the explorer falls back to the bare state.

    Phase 3 _WRITE gate (hub co-location + worker pull-conflict): when
    PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1, skip the local file write
    (the conflict path with the hub's sync_daemon.phase3_heartbeat_writer-
    materialised version). The coordinator POST is NEVER gated here --
    the writer materialises the canonical runner_heartbeats/<host>.json
    from the heartbeats table, which is only populated by POST /heartbeat.
    Suppressing the POST would leave the writer with nothing to publish.

    History: this gate previously short-circuited the whole function
    (including the coordinator POST). That caused fleet-wide stale
    heartbeats once serve.py started setting WRITE=1 for the Mac runner
    on 2026-05-29 (commit e82b2a823f). The gate is now scoped to the
    local file write only.
    """
    if not ree_assembly_path or not ree_assembly_path.is_dir():
        return None

    # Phase 3 _WRITE gate (hub co-location + worker pull-conflict) --
    # determines whether the LOCAL file write happens. The coordinator
    # POST below fires regardless: the writer materialises the canonical
    # file from the heartbeats table. The hub-only _HEARTBEAT_WRITE gate and
    # the worker-safe telemetry-off-git gate both suppress the local file
    # write; either being set is sufficient.
    skip_local_write = (_phase3_heartbeat_write_gated()
                        or _phase3_telemetry_file_write_gated())

    hb_dir = ree_assembly_path / HEARTBEAT_SUBPATH
    if not skip_local_write:
        try:
            hb_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[remote-control] mkdir heartbeat dir failed: {exc}", flush=True)
            return None

    safe = _safe_filename(machine)
    path = hb_dir / f"{safe}.json"

    payload: dict[str, Any] = {
        "machine": machine,
        "hostname": socket.gethostname(),
        "last_tick_utc": _now_utc(),
        "state": state,
        "current_exq": current_exq,
        "current_exq_started_utc": current_exq_started_utc,
        "current_title": current_title,
        "current_claim_id": current_claim_id,
        "current_description": current_description,
        "progress": progress,
        "seconds_elapsed": seconds_elapsed,
        "seconds_remaining": seconds_remaining,
        "recent_lines": (recent_lines or [])[-HEARTBEAT_RECENT_LINES:] if recent_lines is not None else None,
        "queue_depth": queue_depth,
        "queue_id_at_head": queue_id_at_head,
        "recent_completed": (recent_completed or [])[-5:],
        "runner_pid": runner_pid if runner_pid is not None else os.getpid(),
        "runner_version": runner_version,
        "gpu": _gpu_info(),
        # Surface the runner's coordinator state so cross-machine audits
        # (and the /queue-experiment skill's post-push check) can detect
        # workers stuck in git mode without an SSH round-trip. Absent on
        # pre-2026-05-28 runners; readers should treat the missing field
        # as "git" + "unknown runner version, please restart".
        "coordination_mode": os.environ.get("COORDINATION_MODE", "git"),
        "schema_version": "v1",
    }
    if extra:
        payload["extra"] = extra

    if not skip_local_write:
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(tmp, path)
        except Exception as exc:
            print(f"[remote-control] heartbeat write failed: {exc}", flush=True)
            # Fall through to the coordinator POST anyway: the writer is
            # the canonical materialiser; a transient local-write failure
            # must not cost us the DB update.
    # SHADOW / COORDINATOR: mirror the heartbeat to the coordinator. Best-
    # effort. `payload=payload` is the PLAN.md step 6 wiring: the same dict
    # that (would have been) written to runner_heartbeats/<machine>.json
    # travels to the coordinator so sync_daemon.phase3_heartbeat_writer can
    # materialise the file in REE_assembly from the heartbeats table. This
    # POST is NEVER gated by PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE -- under
    # that flag, the local write is skipped but the coordinator POST is
    # the sole transport, so suppressing it would leave the writer with
    # nothing to publish.
    coordinator_client.report_heartbeat(
        machine, state, current_exq, payload.get("progress"),
        payload.get("gpu"),
        seconds_elapsed=payload.get("seconds_elapsed"),
        seconds_remaining=payload.get("seconds_remaining"),
        payload=payload)
    return path if not skip_local_write else None


def _active_claim_on_evidence_dir(ree_assembly_path: Path) -> bool:
    """Return True if TASK_CLAIMS.json has an active claim covering any
    REE_assembly/evidence/ subdirectory (experiments/, planning/, literature/,
    or any future evidence sibling).

    Used by push_heartbeat / push_commands to skip the per-minute pull-rebase-
    autostash dance whenever a Claude session is mid-edit on evidence files.
    The autostash mechanism is mostly safe but can lose uncommitted edits in
    rare interleavings (e.g. autostash-pop conflict that the loop then talks
    over with subsequent commits). Best-effort -- returns False on any error.

    Scope was originally just 'evidence/experiments/' (added 2026-05-01 after
    the EXQ-232 ARC-026 supersession revert incident); broadened 2026-05-08
    to the 'evidence/' prefix after the same signature reappeared on an
    evidence/planning/substrate_queue.json edit. The autostash mechanism is
    not specific to experiments/, so the guard should not be either.
    """
    try:
        claims_path = ree_assembly_path.parent / "TASK_CLAIMS.json"
        if not claims_path.exists():
            return False
        data = json.loads(claims_path.read_text(encoding="utf-8"))
        for entry in data.get("claims", []):
            if entry.get("status") != "active":
                continue
            for res in entry.get("resources", []):
                if "evidence/" in res:
                    return True
        return False
    except Exception:
        return False


# Runner telemetry dirs whose working-tree contents are fully regenerable on
# the next tick. `git checkout -f -B <branch> origin/<branch>` may freely
# discard local changes (uncommitted or committed-but-unpushed) confined to
# these paths; anything else is treated as precious and the push is skipped.
_REGENERABLE_PREFIXES = (
    "evidence/experiments/runner_heartbeats/",
    "evidence/experiments/runner_status/",
    "evidence/experiments/runner_commands/",
)

_PUSH_BRANCH = "master"
_MAX_PUSH_ATTEMPTS = 3


def _git(repo: str, *args: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo,
        capture_output=True, text=True, timeout=timeout,
    )


def _is_regenerable(rel_path: str) -> bool:
    rel_path = rel_path.strip().strip('"')
    return bool(rel_path) and any(
        rel_path.startswith(p) for p in _REGENERABLE_PREFIXES
    )


def _hard_sync_is_safe(repo: str, branch: str) -> bool:
    """True iff resetting the worktree to origin/<branch> cannot destroy
    precious work -- i.e. every uncommitted *tracked* change AND every
    commit ahead of origin/<branch> is confined to the regenerable
    telemetry dirs. Mirrors the launchd repair script's safety envelope so
    this per-tick path is never less safe than the 3-hourly net. Untracked
    files are irrelevant (a hard checkout leaves them in place).
    Conservative: any git error -> False (caller skips the push this tick).
    """
    try:
        st = _git(repo, "status", "--porcelain", "--untracked-files=no",
                  timeout=10)
        if st.returncode != 0:
            return False
        for line in st.stdout.splitlines():
            entry = line[3:]
            if " -> " in entry:  # rename/copy: the new path is what lands
                entry = entry.split(" -> ", 1)[1]
            if entry.strip() and not _is_regenerable(entry):
                return False
        ahead = _git(repo, "rev-list", f"origin/{branch}..HEAD", timeout=10)
        if ahead.returncode != 0:
            return False
        for sha in ahead.stdout.split():
            files = _git(repo, "diff-tree", "--no-commit-id",
                         "--name-only", "-r", sha, timeout=10)
            if files.returncode != 0:
                return False
            for f in files.stdout.splitlines():
                if f.strip() and not _is_regenerable(f):
                    return False
        return True
    except Exception:
        return False


def _push_telemetry_file(
    ree_assembly_path: Path, path: Path, rel: Path, label: str,
) -> None:
    """Rebase-free commit+push of one regenerable telemetry file.

    The old path ran `git pull --rebase --autostash` every tick on every
    machine against a single shared branch; losing the push race left a
    stuck .git/rebase-merge that wedged the repo for hours. This path
    never rebases, so it cannot wedge: capture the file's intended bytes,
    fetch origin, force the local branch onto origin/<branch> (discarding
    only regenerable telemetry -- verified safe by _hard_sync_is_safe),
    restore our bytes, commit just this file, push. A non-fast-forward
    rejection (another machine pushed first) just re-fetches and replays,
    bounded to _MAX_PUSH_ATTEMPTS; the next tick retries regardless.

    Best-effort, never raises. If precious (non-telemetry) uncommitted or
    unpushed work exists, the push is skipped this tick rather than risk
    destroying it -- heartbeats resume once the working tree is clean.
    """
    repo = str(ree_assembly_path)
    try:
        content = path.read_bytes()
    except Exception:
        return
    rel_s = str(rel)
    try:
        for attempt in range(_MAX_PUSH_ATTEMPTS):
            fetch = _git(repo, "fetch", "origin", _PUSH_BRANCH, timeout=20)
            if fetch.returncode != 0:
                return  # offline / transient -- next tick retries
            if not _hard_sync_is_safe(repo, _PUSH_BRANCH):
                return  # precious work present -- do not hard-reset
            # Clean slate vs origin: handles detached HEAD or divergence,
            # discards regenerable telemetry, never rebases.
            sync = _git(repo, "checkout", "-f", "-B", _PUSH_BRANCH,
                        f"origin/{_PUSH_BRANCH}", timeout=15)
            if sync.returncode != 0:
                return
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            except Exception:
                return
            add = _git(repo, "add", rel_s, timeout=10)
            if add.returncode != 0:
                return
            # Nothing staged -> unchanged vs origin within this tick.
            diff = _git(repo, "diff", "--cached", "--quiet", timeout=10)
            if diff.returncode == 0:
                return
            commit = _git(repo, "commit", "-m", f"{label}: {path.stem}",
                          timeout=15)
            if commit.returncode != 0:
                return
            push = _git(repo, "push", "origin",
                        f"HEAD:{_PUSH_BRANCH}", timeout=30)
            if push.returncode == 0:
                return
            stderr = push.stderr or ""
            lost_race = any(
                tok in stderr for tok in
                ("fast-forward", "non-fast-forward", "rejected",
                 "fetch first", "[remote rejected]")
            )
            if not lost_race or attempt == _MAX_PUSH_ATTEMPTS - 1:
                lines = stderr.strip().splitlines()
                tail = lines[-1] if lines else "?"
                print(f"[remote-control] {label} push warn: {tail}",
                      flush=True)
                return
            # Lost the race; loop to re-fetch the new tip and replay.
    except Exception as exc:
        print(f"[remote-control] {label} push error: {exc}", flush=True)


# Phase 3 gate for heartbeat AND commands push paths. When sync_daemon's
# PLAN.md step 6 (derived heartbeats + runner_status writeback) is wired,
# operators set this env var to 1 on each worker to stop the per-tick
# pull --rebase --autostash that has been the source of the autostash-war
# incidents. Default OFF preserves Phase 2 behaviour. Wired symmetrically
# in experiment_runner.git_push_status (per-completion status push).
#
# Phase 3 hub co-location: the _PUSH gate disables only the git push,
# not the local file write. On the hub VM (ree-cloud-1) -- which is
# both the writer host AND a worker -- the runner's local heartbeat
# file write dirties the same REE_assembly checkout the writer is
# trying to publish from, which breaks the writer's clean-tree refusal.
# The _WRITE gate (added 2026-05-29) suppresses the file write too,
# letting the cloud-1 runner come back online without breaking the
# writer. Set on the hub only -- other workers don't need it because
# their REE_assembly checkout is separate from the hub's.
_HEARTBEAT_GATE_LOGGED = [False]  # one-element mutable for module-level state
_HEARTBEAT_WRITE_GATE_LOGGED = [False]
_HEARTBEAT_WRITE_GATE_NON_HUB_REFUSED_LOGGED = [False]
_HEARTBEAT_WRITE_GATE_OFF_GIT_ALLOWED_LOGGED = [False]
_TELEMETRY_OFF_GIT_GATE_LOGGED = [False]
_COMMANDS_VIA_COORD_GATE_LOGGED = [False]
_COMMANDS_OFF_GIT_GATE_LOGGED = [False]
_COMMANDS_OFF_GIT_REFUSED_LOGGED = [False]

# Canonical hostname(s) of the hub VM where the WRITE gate is safe to set.
# Documented as `ree-cloud-1` in cloud_workers.md, but `socket.gethostname()`
# on that VM returns `ree-worker-1` (per journalctl evidence from the
# 2026-05-30 fleet-rescue session). Accept both so the guard fires
# correctly on whichever string the underlying VM reports.
_PHASE3_HUB_HOSTNAMES = frozenset({"ree-cloud-1", "ree-worker-1"})


def _phase3_heartbeat_gated() -> bool:
    # WRITE implies PUSH: if the local file write is gated, there is
    # nothing on disk to push and the push path must also short-circuit.
    if _phase3_heartbeat_write_gated():
        return True
    enabled = os.environ.get(
        "PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH", "").strip().lower() in (
            "1", "true", "yes")
    if enabled and not _HEARTBEAT_GATE_LOGGED[0]:
        print("[remote-control] phase3 gate active: heartbeat + commands "
              "git pushes will be skipped (sync_daemon step 6 owns "
              "derived heartbeats once wired)", flush=True)
        _HEARTBEAT_GATE_LOGGED[0] = True
    return enabled


def _phase3_heartbeat_write_gated() -> bool:
    """Phase 3 _WRITE gate: suppresses the local heartbeat file write too.

    For the hub VM co-location case (ree-cloud-1 = hub + worker). On
    other workers leave this OFF; their local writes don't affect the
    writer's REE_assembly checkout.

    Implies _PUSH gating: writing to a tmp file we never rename and
    then trying to push is incoherent, so callers should treat WRITE
    as a strict-superset of PUSH.

    Hub-only self-guard (2026-05-31): if the env var is set on a worker
    whose hostname is NOT in _PHASE3_HUB_HOSTNAMES, REFUSE the gate
    (return False as if not set) and print a loud warning. Rationale:
    the file-channel command writeback (write_commands_file) short-
    circuits when the WRITE gate is on, and the per-machine commands
    file is the only place a non-hub runner can persist a stop command's
    status -> done. Without that persistence, systemd restarts the unit,
    the same stop command is picked up again, the runner drains and
    exits, systemd restarts, ... until start-limit-hit kills the unit.
    Incident 2026-05-30: cloud-2/3/4 all wedged in failed/start-limit-hit
    for 6-12 hours after a fleet-wide shadow.conf template push
    mis-included this flag.
    """
    enabled = os.environ.get(
        "PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE", "").strip().lower() in (
            "1", "true", "yes")
    if not enabled:
        return False
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = ""
    if hostname not in _PHASE3_HUB_HOSTNAMES:
        # The hub-only restriction exists ONLY because _HEARTBEAT_WRITE also
        # short-circuits the git command-file writeback (write_commands_file),
        # and a non-hub worker that cannot persist a stop command's
        # status->done restart-loops -> start-limit-hit (incident 2026-05-30).
        # When PHASE3_COMMANDS_OFF_GIT is active the command channel is the
        # coordinator (the ack persists server-side in the commands table,
        # not in a per-machine git file), so that hazard is gone and a
        # non-hub worker may safely take the WRITE gate. This is the
        # command-file dependency that previously forced _HEARTBEAT_WRITE
        # hub-only; the migration removes it.
        if _phase3_commands_off_git_gated():
            if not _HEARTBEAT_WRITE_GATE_OFF_GIT_ALLOWED_LOGGED[0]:
                print("[remote-control] phase3 gate active: "
                      "PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE permitted on "
                      "non-hub worker (hostname=" + repr(hostname) + ") "
                      "because PHASE3_COMMANDS_OFF_GIT routes the command "
                      "channel through the coordinator -- stop-command acks "
                      "persist server-side, so the restart-loop hazard "
                      "(incident 2026-05-30) does not apply.", flush=True)
                _HEARTBEAT_WRITE_GATE_OFF_GIT_ALLOWED_LOGGED[0] = True
            return True
        if not _HEARTBEAT_WRITE_GATE_NON_HUB_REFUSED_LOGGED[0]:
            print("[remote-control] WARNING: "
                  "PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1 set on "
                  "non-hub worker (hostname=" + repr(hostname) +
                  "). This flag is HUB-ONLY (ree-cloud-1 / "
                  "ree-worker-1) unless PHASE3_COMMANDS_OFF_GIT is also "
                  "set. On non-hub workers it suppresses "
                  "the per-machine commands file write, which breaks "
                  "stop-command persistence and produces a systemd "
                  "restart loop -> start-limit-hit (incident 2026-05-30 "
                  "cloud-2/3/4). REFUSING the gate; treating as if "
                  "unset. Remove the env var from this worker's "
                  "shadow.conf to silence this warning.", flush=True)
            _HEARTBEAT_WRITE_GATE_NON_HUB_REFUSED_LOGGED[0] = True
        return False
    if not _HEARTBEAT_WRITE_GATE_LOGGED[0]:
        print("[remote-control] phase3 gate active: heartbeat + commands "
              "FILE WRITES will be skipped (hub co-location -- "
              "sync_daemon owns derived heartbeats; coordinator owns "
              "command-channel writeback)", flush=True)
        _HEARTBEAT_WRITE_GATE_LOGGED[0] = True
    return True


def _phase3_telemetry_file_write_gated() -> bool:
    """Worker-safe gate: suppress ONLY the in-tree heartbeat + status FILE writes.

    Phase 3 moved telemetry TRANSPORT off git -- workers POST /heartbeat and
    /status to the coordinator and the hub's sync_daemon.phase3_heartbeat_writer
    is the sole git materialiser. But workers still wrote their own
    runner_heartbeats/<host>.json + runner_status/<host>.json into the shared
    REE_assembly checkout every tick, which conflicts with the hub-materialised
    version on `git pull --rebase --autostash` and produces dormant-autostash
    accumulation (cloud-3 hit 43 entries 2026-06-02; 191 by 2026-05-31).

    This gate stops those local telemetry-FILE writes on workers. Unlike
    PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE (hub-only -- ALSO gates the
    command-file writeback, which workers need for stop-command persistence;
    enabling it on a worker restart-loops the unit, incident 2026-05-30), this
    gate does NOT touch the command channel: write_commands_file /
    read_commands_file / process_pending_commands are unaffected, so stop-command
    persistence is preserved and there is no restart loop. Safe on any worker.

    The coordinator POST (/heartbeat, /status) is NEVER gated here -- it is the
    canonical transport. Skip-completed and peer-dedup keep reading the in-tree
    runner_status/ dir, which is populated by pull (hub materialisation), so they
    are unaffected; the worker's own completions arrive via coordinator -> hub ->
    pull (lag tolerated; the coordinator /claim is authoritative against dupes).

    Env: PHASE3_RUNNER_TELEMETRY_OFF_GIT=1|true|yes. No hostname restriction.
    """
    enabled = os.environ.get(
        "PHASE3_RUNNER_TELEMETRY_OFF_GIT", "").strip().lower() in (
            "1", "true", "yes")
    if enabled and not _TELEMETRY_OFF_GIT_GATE_LOGGED[0]:
        print("[remote-control] phase3 telemetry-off-git gate active: "
              "in-tree heartbeat + status FILE writes skipped (coordinator "
              "POST is the transport; hub sync_daemon materialises git). "
              "Command channel unaffected.", flush=True)
        _TELEMETRY_OFF_GIT_GATE_LOGGED[0] = True
    return enabled


def _coordinator_command_channel_available() -> bool:
    """True when the coordinator command channel can carry commands:
    COORDINATION_MODE=coordinator AND the coordinator_client shim is enabled
    (COORDINATOR_URL + COORDINATOR_TOKEN set). Never raises."""
    try:
        mode = os.environ.get("COORDINATION_MODE", "git").strip().lower()
        return mode == "coordinator" and bool(coordinator_client.enabled())
    except Exception:  # pragma: no cover -- shim must never break the runner
        return False


def _phase3_commands_via_coordinator_gated() -> bool:
    """Read + ack remote-control commands via the coordinator (in ADDITION to
    the git command-file, which stays the proven fallback during transition).

    Env: PHASE3_COMMANDS_VIA_COORDINATOR=1|true|yes. Requires the coordinator
    command channel to be available; if the flag is set but the coordinator is
    not configured, this is a no-op (the git file remains the channel). Default
    OFF = git command-file is the sole channel (bit-identical to pre-migration).

    This is the per-worker activation knob the canary uses (cloud-2 first) so
    the coordinator command path can be proven on ONE idle worker before the
    fleet, rather than flipping on for every worker the moment the code lands.
    """
    enabled = os.environ.get(
        "PHASE3_COMMANDS_VIA_COORDINATOR", "").strip().lower() in (
            "1", "true", "yes")
    if not enabled:
        return False
    if not _coordinator_command_channel_available():
        return False
    if not _COMMANDS_VIA_COORD_GATE_LOGGED[0]:
        print("[remote-control] phase3 commands-via-coordinator gate active: "
              "remote-control commands are also fetched + acked via the "
              "coordinator (git command-file retained as fallback).",
              flush=True)
        _COMMANDS_VIA_COORD_GATE_LOGGED[0] = True
    return True


def _phase3_commands_off_git_gated() -> bool:
    """The coordinator is the SOLE command channel: the worker neither reads
    nor writes the git command-file. Implies PHASE3_COMMANDS_VIA_COORDINATOR.

    This is what removes the restart-loop hazard that made
    PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE hub-only: a stop command's ack now
    persists in the coordinator `commands` table, not in a per-machine git
    file the worker may be unable to write. Once a worker is off-git for
    commands, it can also take the heartbeat WRITE gate without restart-looping
    (see _phase3_heartbeat_write_gated).

    Self-guard (mirrors the _HEARTBEAT_WRITE non-hub refusal): if the flag is
    set but the coordinator command channel is NOT available (COORDINATION_MODE
    != coordinator, or no URL/token), REFUSE the gate and fall back to the git
    command-file. Dropping the only channel would leave the worker
    uncontrollable (no way to stop/pause it).

    Env: PHASE3_COMMANDS_OFF_GIT=1|true|yes. Default OFF.
    """
    enabled = os.environ.get(
        "PHASE3_COMMANDS_OFF_GIT", "").strip().lower() in (
            "1", "true", "yes")
    if not enabled:
        return False
    if not _coordinator_command_channel_available():
        if not _COMMANDS_OFF_GIT_REFUSED_LOGGED[0]:
            print("[remote-control] WARNING: PHASE3_COMMANDS_OFF_GIT=1 set "
                  "but the coordinator command channel is unavailable "
                  "(COORDINATION_MODE != coordinator or COORDINATOR_URL/"
                  "TOKEN unset). REFUSING the gate and keeping the git "
                  "command-file as the channel -- dropping it would leave "
                  "this worker uncontrollable (no stop/pause path).",
                  flush=True)
            _COMMANDS_OFF_GIT_REFUSED_LOGGED[0] = True
        return False
    if not _COMMANDS_OFF_GIT_GATE_LOGGED[0]:
        print("[remote-control] phase3 commands-off-git gate active: "
              "git command-file is NOT read or written; the coordinator is "
              "the sole remote-control command channel.", flush=True)
        _COMMANDS_OFF_GIT_GATE_LOGGED[0] = True
    return True


def push_heartbeat(ree_assembly_path: Path, path: Path) -> None:
    """Commit + push a single heartbeat file. Best-effort, never raises.

    Used when --auto-sync is on. Failure (e.g. losing the push race) is
    logged but does not interrupt the runner -- the next tick rewrites the
    heartbeat and retries.

    If an active TASK_CLAIMS entry covers any REE_assembly/evidence/ subdir,
    the entire push is skipped for this tick so a concurrent Claude session's
    uncommitted edits are left untouched. Heartbeats resume on the next tick
    once the claim closes. See _push_telemetry_file for the (rebase-free)
    git strategy.

    Phase 3: gated by PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH. When that env
    var is 1, the entire push path is skipped -- sync_daemon's step-6
    writeback (when wired) takes over derived heartbeat publication.
    """
    if _phase3_heartbeat_gated():
        return
    if not path or not path.exists():
        return
    if _active_claim_on_evidence_dir(ree_assembly_path):
        return
    try:
        rel = path.relative_to(ree_assembly_path)
    except ValueError:
        return
    _push_telemetry_file(ree_assembly_path, path, rel, "heartbeat")


# ────────────────────────────────────────────────────────────────────────────
# Phase 2: command channel
# ────────────────────────────────────────────────────────────────────────────


def commands_path(ree_assembly_path: Path, machine: str) -> Path:
    return ree_assembly_path / COMMANDS_SUBPATH / f"{_safe_filename(machine)}.json"


def read_commands_file(ree_assembly_path: Path, machine: str) -> dict:
    """Read the per-machine commands file, or return a fresh empty container."""
    path = commands_path(ree_assembly_path, machine)
    if not path.exists():
        return {
            "schema_version": "v1",
            "machine": machine,
            "commands": [],
        }
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "commands" not in data:
            return {"schema_version": "v1", "machine": machine, "commands": []}
        if not isinstance(data["commands"], list):
            data["commands"] = []
        return data
    except Exception as exc:
        print(f"[remote-control] commands read error ({path.name}): {exc}", flush=True)
        return {"schema_version": "v1", "machine": machine, "commands": []}


def write_commands_file(ree_assembly_path: Path, machine: str, data: dict) -> Path | None:
    """Atomic write of the commands file. Prunes done/failed history to the
    most recent MAX_HISTORY_PER_FILE entries.

    Phase 3 _WRITE gate (hub co-location): when
    PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1, skip the local file write.
    On the hub the coordinator's commands API is the source of truth
    (process_pending_commands fetches + acks via coordinator_client). On
    workers the git command-file remains the channel UNLESS
    PHASE3_COMMANDS_OFF_GIT is set -- under that gate process_pending_commands
    never calls this function (the coordinator is the sole channel), and a
    worker may then also take the WRITE gate without restart-looping a stop
    command (its ack persists in the coordinator, not in this file).
    """
    cmds = data.get("commands", [])
    pending = [c for c in cmds if c.get("status") in ("pending", "ack")]
    history = [c for c in cmds if c.get("status") in ("done", "failed")]
    if len(history) > MAX_HISTORY_PER_FILE:
        history = history[-MAX_HISTORY_PER_FILE:]
    data["commands"] = pending + history

    # Phase 3 _WRITE gate -- hub co-location safeguard.
    if _phase3_heartbeat_write_gated():
        return None

    path = commands_path(ree_assembly_path, machine)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, path)
        return path
    except Exception as exc:
        print(f"[remote-control] commands write error: {exc}", flush=True)
        return None


def push_commands(ree_assembly_path: Path, path: Path, label: str = "commands") -> None:
    """Stage + commit + push a commands file (best-effort, never raises).

    Same active-TASK_CLAIMS protection as push_heartbeat -- skip the entire
    push if a Claude session is mid-edit on any REE_assembly/evidence/
    subdir, to avoid autostash interactions destabilising uncommitted edits.

    Phase 3: gated by the same PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH env
    var as push_heartbeat (commands and heartbeats live in the same
    runner_commands / runner_heartbeats layer, both retired by step 6).
    """
    if _phase3_heartbeat_gated():
        return
    if not path or not path.exists():
        return
    if _active_claim_on_evidence_dir(ree_assembly_path):
        return
    try:
        rel = path.relative_to(ree_assembly_path)
    except ValueError:
        return
    _push_telemetry_file(ree_assembly_path, path, rel, label)


def append_command(
    ree_assembly_path: Path,
    machine: str,
    kind: str,
    args: dict | None = None,
    issued_by: str = "unknown",
) -> dict | None:
    """Append a pending command to the per-machine queue. Used by serve.py
    POST handlers and ad-hoc callers (e.g. Claude editing the file directly).

    Returns the newly-appended command dict, or None on failure.
    """
    if kind not in VALID_COMMAND_KINDS:
        raise ValueError(f"unknown command kind: {kind!r}; "
                         f"valid: {VALID_COMMAND_KINDS}")
    data = read_commands_file(ree_assembly_path, machine)
    now = _now_utc()
    cmd_id = f"cmd-{now}-{os.urandom(3).hex()}"
    cmd = {
        "id": cmd_id,
        "kind": kind,
        "args": args or {},
        "issued_at_utc": now,
        "issued_by": issued_by,
        "status": "pending",
        "ack_at_utc": None,
        "completed_at_utc": None,
        "error": None,
        "result_note": None,
    }
    data.setdefault("commands", []).append(cmd)
    data["machine"] = machine
    data["schema_version"] = "v1"
    written = write_commands_file(ree_assembly_path, machine, data)
    if written is None:
        return None
    return cmd


def _kick_queue(queue_file: Path, queue_id: str) -> tuple[bool, str]:
    """Move queue_id to the head of experiment_queue.json. Returns (ok, note)."""
    if not queue_file.exists():
        return False, f"queue file not found: {queue_file}"
    try:
        data = json.loads(queue_file.read_text())
        items = data.get("items", [])
        idx = next((i for i, qi in enumerate(items)
                    if qi.get("queue_id") == queue_id), None)
        if idx is None:
            return False, f"{queue_id} not in queue"
        if idx == 0:
            return True, f"{queue_id} already at head"
        items.insert(0, items.pop(idx))
        data["items"] = items
        queue_file.write_text(json.dumps(data, indent=2) + "\n")
        return True, f"{queue_id} moved to head (was idx {idx})"
    except Exception as exc:
        return False, f"kick error: {exc}"


def _release_claim_in_queue(queue_file: Path, queue_id: str) -> tuple[bool, str]:
    """Clear claimed_by on a queue item. Returns (ok, note)."""
    if not queue_file.exists():
        return False, f"queue file not found: {queue_file}"
    try:
        data = json.loads(queue_file.read_text())
        items = data.get("items", [])
        for qi in items:
            if qi.get("queue_id") == queue_id:
                prev = qi.get("claimed_by")
                qi["claimed_by"] = None
                queue_file.write_text(json.dumps(data, indent=2) + "\n")
                return True, f"{queue_id} claim cleared (was {prev})"
        return False, f"{queue_id} not in queue"
    except Exception as exc:
        return False, f"release_claim error: {exc}"


def _apply_reclassify(
    status_ref: dict,
    status_path: Path,
    write_status_fn,
    args: dict,
    machine: str,
) -> tuple[bool, str]:
    """Mutate status['completed'] to fix a misclassified entry, then
    atomically rewrite the local runner_status.json.

    Returns (ok, note). On ok=True the next heartbeat POST will carry the
    corrected status_payload_json -- no further action required.

    Refuses when the entry is missing or the requested result is not in
    {PASS, FAIL, ERROR}. Idempotent: if the entry already has the
    requested result, no write is performed and ok=True with a "no-op"
    note.
    """
    qid = args.get("queue_id")
    new_result = (args.get("result") or "").upper()
    if not qid:
        return (False, "missing args.queue_id")
    if new_result not in ("PASS", "FAIL", "ERROR"):
        return (False,
                f"invalid args.result {args.get('result')!r}; "
                "expected PASS / FAIL / ERROR")
    if status_ref is None or status_path is None or write_status_fn is None:
        return (False,
                "runner did not pass status context; cannot reclassify "
                "(caller may be on an older runner_remote_control API)")

    completed = status_ref.get("completed")
    if not isinstance(completed, list):
        return (False, "status['completed'] missing or wrong shape")

    entry = next((e for e in completed
                  if isinstance(e, dict) and e.get("queue_id") == qid),
                 None)
    if entry is None:
        return (False, f"no completed entry for {qid}")

    prior = entry.get("result", "")
    if prior == new_result:
        return (True, f"already {new_result}; no-op")

    note_extra = args.get("note") or ""
    new_summary = (
        f"Reclassified {prior or '?'}->{new_result} via remote-control "
        f"at {_now_utc()} on {machine}"
        + (f": {note_extra}" if note_extra else "")
    )

    entry["result"] = new_result
    entry["result_summary"] = new_summary
    if args.get("output_file"):
        entry["output_file"] = str(args["output_file"])

    try:
        write_status_fn(status_ref, status_path)
    except Exception as exc:  # noqa: BLE001
        # Roll back the in-memory mutation so the runner's internal
        # state stays consistent with what's on disk.
        entry["result"] = prior
        return (False, f"write_status failed: {exc}")

    return (True,
            f"reclassified {qid}: {prior or '?'}->{new_result} "
            "(next heartbeat will propagate)")


def _execute_command(
    kind: str,
    args: dict,
    *,
    machine: str,
    queue_file: Path,
    drain_flag: list,
    pause_flag: list,
    force_stop_flag: list,
    suspend_flag: list,
    resume_run_target: list,
    current_proc: list,
    status_ref: dict | None,
    status_path: Path | None,
    write_status_fn,
) -> tuple[bool, str]:
    """Execute one command kind against the runner's state. Returns
    (ok, note). Channel-agnostic: the git command-file path and the
    coordinator path both dispatch here so behaviour is identical
    regardless of how the command arrived. Never raises (exceptions are
    captured into a failed result)."""
    args = args or {}
    try:
        if kind == "stop":
            if not drain_flag:
                drain_flag.append(True)
            return True, "drain requested"
        if kind == "force_stop":
            force_stop_flag.append(True)
            note = "no active proc; runner will exit on next tick"
            ok = True
            if current_proc:
                proc = current_proc[0]
                try:
                    proc.kill()
                    note = f"killed pid {proc.pid}"
                except Exception as exc:
                    ok = False
                    note = f"kill failed: {exc}"
            if not drain_flag:
                drain_flag.append(True)
            return ok, note
        if kind == "pause":
            if not pause_flag:
                pause_flag.append(True)
            return True, "paused"
        if kind == "resume":
            pause_flag.clear()
            return True, "resumed"
        if kind == "suspend":
            if current_proc:
                if not suspend_flag:
                    suspend_flag.append(True)
                return True, "suspend requested (terminate current run)"
            return True, "no active proc; suspend is a no-op until a run starts"
        if kind == "resume_run":
            pause_flag.clear()
            resume_run_target.clear()
            qid = args.get("queue_id")
            if qid:
                resume_run_target.append(str(qid))
                return True, f"resume_run queued for {qid}"
            return True, "resume_run queued (next checkpointed item)"
        if kind == "kick":
            qid = args.get("queue_id")
            if not qid:
                return False, "missing args.queue_id"
            return _kick_queue(queue_file, qid)
        if kind == "release_claim":
            qid = args.get("queue_id")
            if not qid:
                return False, "missing args.queue_id"
            return _release_claim_in_queue(queue_file, qid)
        if kind == "reclassify":
            return _apply_reclassify(
                status_ref, status_path, write_status_fn, args, machine)
        return False, f"unsupported kind: {kind!r}"
    except Exception as exc:
        return False, f"exception: {exc}"


def _process_git_command_file(
    ree_assembly_path: Path,
    machine: str,
    queue_file: Path,
    *,
    auto_sync: bool,
    exec_kwargs: dict,
) -> list[dict]:
    """Legacy git command-file channel: read pending commands, mark them
    acked, execute each, write the terminal state back, and (when auto_sync)
    push. Bit-identical to the pre-migration process_pending_commands body."""
    data = read_commands_file(ree_assembly_path, machine)
    cmds = data.get("commands", [])
    pending = [c for c in cmds if c.get("status") == "pending"]
    if not pending:
        return []

    for cmd in pending:
        cmd["status"] = "ack"
        cmd["ack_at_utc"] = _now_utc()
    # Persist ack state immediately so observers see prompt acknowledgment.
    write_commands_file(ree_assembly_path, machine, data)

    processed: list[dict] = []
    for cmd in pending:
        kind = cmd.get("kind")
        ok, note = _execute_command(
            kind, cmd.get("args") or {}, machine=machine,
            queue_file=queue_file, **exec_kwargs)
        cmd["status"] = "done" if ok else "failed"
        cmd["completed_at_utc"] = _now_utc()
        cmd["result_note"] = note
        if not ok:
            cmd["error"] = note
        processed.append(cmd)
        print(f"[remote-control] cmd {cmd['id']} {kind} -> "
              f"{cmd['status']} ({note}) [git]", flush=True)

    written = write_commands_file(ree_assembly_path, machine, data)
    if auto_sync and written is not None:
        push_commands(ree_assembly_path, written, label="commands-ack")
    return processed


def _process_coordinator_commands(
    machine: str,
    queue_file: Path,
    *,
    exec_kwargs: dict,
) -> list[dict]:
    """Coordinator command channel: fetch pending commands via
    coordinator_client.fetch_commands, execute each, ack via
    coordinator_client.ack_command. A None fetch (coordinator unreachable)
    is a silent no-op -- the git fallback covers it during transition, and
    under commands-off-git the command simply re-delivers next tick (the
    supported kinds are idempotent). Never raises into the runner loop."""
    try:
        resp = coordinator_client.fetch_commands(machine)
    except Exception as exc:  # pragma: no cover -- shim swallows already
        print(f"[remote-control] coordinator fetch_commands error: {exc}",
              flush=True)
        return []
    if not resp or not isinstance(resp, dict):
        return []
    cmds = resp.get("commands") or []
    processed: list[dict] = []
    for cmd in cmds:
        cmd_id = cmd.get("id")
        kind = cmd.get("kind")
        # The coordinator stores args as a JSON string (commands.args TEXT).
        raw_args = cmd.get("args")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args else {}
            except ValueError:
                args = {}
        else:
            args = raw_args or {}
        ok, note = _execute_command(
            kind, args, machine=machine, queue_file=queue_file, **exec_kwargs)
        status = "done" if ok else "failed"
        ack = None
        try:
            ack = coordinator_client.ack_command(cmd_id, machine, status, note)
        except Exception as exc:  # pragma: no cover
            print(f"[remote-control] coordinator ack error: {exc}", flush=True)
        if ack is None:
            print(f"[remote-control] WARNING: coordinator ack for cmd "
                  f"{cmd_id} ({kind}) did not confirm; it will re-deliver "
                  f"next tick (supported kinds are idempotent).", flush=True)
        processed.append({
            "id": cmd_id, "kind": kind, "status": status,
            "result_note": note, "channel": "coordinator",
            "acked": ack is not None,
        })
        print(f"[remote-control] cmd {cmd_id} {kind} -> {status} ({note}) "
              f"[coordinator]", flush=True)
    return processed


def process_pending_commands(
    ree_assembly_path: Path,
    machine: str,
    queue_file: Path,
    *,
    drain_flag: list,
    pause_flag: list,
    force_stop_flag: list,
    suspend_flag: list,
    resume_run_target: list,
    current_proc: list,
    auto_sync: bool = False,
    status_ref: dict | None = None,
    status_path: Path | None = None,
    write_status_fn=None,
) -> list[dict]:
    """Drain pending remote-control commands and execute them, returning the
    processed commands for logging. Mutates the runner's flag lists (used as
    mutable references), the same single-element-or-empty container convention
    experiment_runner.py uses for _drain_flag / _current_proc / etc.

    Channels (Phase 3 command-channel migration):
      - git command-file (legacy): read/write
        REE_assembly/evidence/experiments/runner_commands/<machine>.json.
        Used unless PHASE3_COMMANDS_OFF_GIT is active.
      - coordinator: fetch + ack via coordinator_client. Enabled by
        PHASE3_COMMANDS_VIA_COORDINATOR (dual-read alongside git) or implied
        by PHASE3_COMMANDS_OFF_GIT (coordinator sole channel).
    Both default OFF -> bit-identical to the pre-migration git-only behaviour.
    During the dual-read transition a command issued to BOTH channels is
    executed once per channel; every supported kind is idempotent so the
    double-apply is harmless (the explicit safety property that lets the
    fallback run alongside the new path).

    `status_ref`, `status_path`, `write_status_fn` are optional context used
    only by the `reclassify` command kind; when omitted reclassify fails with
    a clear error and other kinds are unaffected.
    """
    off_git = _phase3_commands_off_git_gated()
    via_coord = _phase3_commands_via_coordinator_gated()

    exec_kwargs = {
        "drain_flag": drain_flag,
        "pause_flag": pause_flag,
        "force_stop_flag": force_stop_flag,
        "suspend_flag": suspend_flag,
        "resume_run_target": resume_run_target,
        "current_proc": current_proc,
        "status_ref": status_ref,
        "status_path": status_path,
        "write_status_fn": write_status_fn,
    }

    processed: list[dict] = []

    # Git command-file channel -- the proven path, retained as fallback until
    # commands-off-git is set. Skipped entirely when off-git so the worker
    # neither reads nor writes the per-machine file.
    if not off_git:
        processed.extend(_process_git_command_file(
            ree_assembly_path, machine, queue_file,
            auto_sync=auto_sync, exec_kwargs=exec_kwargs))

    # Coordinator channel -- active when explicitly enabled or when off-git
    # makes it the sole channel (off_git already guarantees the channel is
    # available via its self-guard).
    if via_coord or off_git:
        processed.extend(_process_coordinator_commands(
            machine, queue_file, exec_kwargs=exec_kwargs))

    return processed
