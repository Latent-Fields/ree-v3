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


HEARTBEAT_SUBPATH = Path("evidence") / "experiments" / "runner_heartbeats"
COMMANDS_SUBPATH = Path("evidence") / "experiments" / "runner_commands"

VALID_COMMAND_KINDS = (
    "stop", "force_stop", "pause", "resume", "kick", "release_claim",
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
) -> Path | None:
    """Write a heartbeat snapshot for this runner. Never raises.

    Returns the path written, or None on failure.
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
    return path


def push_heartbeat(ree_assembly_path: Path, path: Path) -> None:
    """Stage + commit + push a single heartbeat file. Best-effort, never raises.

    Used when --auto-sync is on. Failure (e.g. concurrent push) is logged but
    does not interrupt the runner -- the next tick will rewrite + retry.
    """
    if not path or not path.exists():
        return
    try:
        rel = path.relative_to(ree_assembly_path)
    except ValueError:
        return
    try:
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash"],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=20,
        )
        add = subprocess.run(
            ["git", "add", str(rel)],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=10,
        )
        if add.returncode != 0:
            return
        # Skip commit if nothing staged (heartbeat unchanged within tick window).
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=10,
        )
        if diff.returncode == 0:
            return
        commit = subprocess.run(
            ["git", "commit", "-m", f"heartbeat: {path.stem}"],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=15,
        )
        if commit.returncode != 0:
            return
        push = subprocess.run(
            ["git", "push", "origin", "HEAD:master"],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=30,
        )
        if push.returncode != 0:
            stderr = (push.stderr or "").strip().splitlines()
            tail = stderr[-1] if stderr else "?"
            print(f"[remote-control] heartbeat push warn: {tail}", flush=True)
    except Exception as exc:
        print(f"[remote-control] heartbeat push error: {exc}", flush=True)


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
    """Stage + commit + push a commands file (best-effort, never raises)."""
    if not path or not path.exists():
        return
    try:
        rel = path.relative_to(ree_assembly_path)
    except ValueError:
        return
    try:
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash"],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=20,
        )
        add = subprocess.run(
            ["git", "add", str(rel)],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=10,
        )
        if add.returncode != 0:
            return
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=10,
        )
        if diff.returncode == 0:
            return
        commit = subprocess.run(
            ["git", "commit", "-m", f"{label}: {path.stem}"],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=15,
        )
        if commit.returncode != 0:
            return
        push = subprocess.run(
            ["git", "push", "origin", "HEAD:master"],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=30,
        )
        if push.returncode != 0:
            tail = (push.stderr or "").strip().splitlines()
            tail_msg = tail[-1] if tail else "?"
            print(f"[remote-control] {label} push warn: {tail_msg}", flush=True)
    except Exception as exc:
        print(f"[remote-control] {label} push error: {exc}", flush=True)


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
