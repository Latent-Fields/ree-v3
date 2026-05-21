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
)

# Keep the last N done/failed commands in the file; older ones are pruned to
# bound file growth across long-running machines.
MAX_HISTORY_PER_FILE = 50


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
    """
    if not ree_assembly_path or not ree_assembly_path.is_dir():
        return None

    hb_dir = ree_assembly_path / HEARTBEAT_SUBPATH
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
        "recent_lines": (recent_lines or [])[-5:] if recent_lines is not None else None,
        "queue_depth": queue_depth,
        "queue_id_at_head": queue_id_at_head,
        "recent_completed": (recent_completed or [])[-5:],
        "runner_pid": runner_pid if runner_pid is not None else os.getpid(),
        "runner_version": runner_version,
        "gpu": _gpu_info(),
        "schema_version": "v1",
    }
    if extra:
        payload["extra"] = extra

    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, path)
    except Exception as exc:
        print(f"[remote-control] heartbeat write failed: {exc}", flush=True)
        return None
    # SHADOW (no-op unless COORDINATION_MODE=shadow): mirror the heartbeat
    # to the coordinator alongside the existing file write. Best-effort.
    coordinator_client.report_heartbeat(
        machine, state, current_exq, payload.get("progress"),
        payload.get("gpu"),
        seconds_elapsed=payload.get("seconds_elapsed"),
        seconds_remaining=payload.get("seconds_remaining"))
    return path


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
    """
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
    """
    cmds = data.get("commands", [])
    pending = [c for c in cmds if c.get("status") in ("pending", "ack")]
    history = [c for c in cmds if c.get("status") in ("done", "failed")]
    if len(history) > MAX_HISTORY_PER_FILE:
        history = history[-MAX_HISTORY_PER_FILE:]
    data["commands"] = pending + history

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
    """
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
) -> list[dict]:
    """Drain pending commands and execute them. Mutates the runner's flag
    lists (used as mutable references) and returns the processed commands
    for logging.

    Each list parameter is a single-element-or-empty mutable container, the
    same convention experiment_runner.py already uses for _drain_flag,
    _current_proc, etc. Setting drain_flag.append(True) requests a graceful
    drain on the next loop iteration.
    """
    data = read_commands_file(ree_assembly_path, machine)
    cmds = data.get("commands", [])
    pending = [c for c in cmds if c.get("status") == "pending"]
    if not pending:
        return []

    processed: list[dict] = []
    for cmd in pending:
        cmd["status"] = "ack"
        cmd["ack_at_utc"] = _now_utc()
    # Persist ack state immediately so observers see prompt acknowledgment.
    write_commands_file(ree_assembly_path, machine, data)

    for cmd in pending:
        kind = cmd.get("kind")
        args = cmd.get("args") or {}
        ok = True
        note = ""
        try:
            if kind == "stop":
                if not drain_flag:
                    drain_flag.append(True)
                note = "drain requested"
            elif kind == "force_stop":
                force_stop_flag.append(True)
                if current_proc:
                    proc = current_proc[0]
                    try:
                        proc.kill()
                        note = f"killed pid {proc.pid}"
                    except Exception as exc:
                        ok = False
                        note = f"kill failed: {exc}"
                else:
                    note = "no active proc; runner will exit on next tick"
                if not drain_flag:
                    drain_flag.append(True)
            elif kind == "pause":
                if not pause_flag:
                    pause_flag.append(True)
                note = "paused"
            elif kind == "resume":
                pause_flag.clear()
                note = "resumed"
            elif kind == "suspend":
                if current_proc:
                    if not suspend_flag:
                        suspend_flag.append(True)
                    note = "suspend requested (terminate current run)"
                else:
                    note = "no active proc; suspend is a no-op until a run starts"
            elif kind == "resume_run":
                pause_flag.clear()
                resume_run_target.clear()
                qid = args.get("queue_id")
                if qid:
                    resume_run_target.append(str(qid))
                note = (
                    f"resume_run queued for {qid}"
                    if qid
                    else "resume_run queued (next checkpointed item)"
                )
            elif kind == "kick":
                qid = args.get("queue_id")
                if not qid:
                    ok = False
                    note = "missing args.queue_id"
                else:
                    ok, note = _kick_queue(queue_file, qid)
            elif kind == "release_claim":
                qid = args.get("queue_id")
                if not qid:
                    ok = False
                    note = "missing args.queue_id"
                else:
                    ok, note = _release_claim_in_queue(queue_file, qid)
            else:
                ok = False
                note = f"unsupported kind: {kind!r}"
        except Exception as exc:
            ok = False
            note = f"exception: {exc}"

        cmd["status"] = "done" if ok else "failed"
        cmd["completed_at_utc"] = _now_utc()
        cmd["result_note"] = note
        if not ok:
            cmd["error"] = note
        processed.append(cmd)
        print(f"[remote-control] cmd {cmd['id']} {kind} -> "
              f"{cmd['status']} ({note})", flush=True)

    written = write_commands_file(ree_assembly_path, machine, data)
    if auto_sync and written is not None:
        push_commands(ree_assembly_path, written, label="commands-ack")
    return processed
