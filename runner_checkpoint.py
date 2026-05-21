"""Checkpoint paths and resume args for runner suspend/resume_run.

Experiments that opt in via queue item fields:
  checkpoint_resumable: true
  checkpoint_experiment_type: <dir name under evidence/experiments/_partial/>

Scripts should write partial JSON at the default path and accept --resume.
See v3_exq_590_isef004_novelty_bonus_goldilocks_v3.py for the reference pattern.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PARTIAL_SUBPATH = Path("evidence") / "experiments" / "_partial"


def experiment_type_for_item(item: dict) -> str:
    """Directory name under _partial/ for this queue item."""
    explicit = (
        (item.get("checkpoint_experiment_type") or "")
        or (item.get("experiment_type") or "")
    ).strip()
    if explicit:
        return explicit
    script = (item.get("script") or "").replace("\\", "/")
    base = script.rsplit("/", 1)[-1]
    if base.endswith(".py"):
        base = base[:-3]
    return base or item.get("queue_id", "unknown")


def partial_checkpoint_path(
    ree_assembly_path: Path,
    queue_id: str,
    experiment_type: str,
) -> Path:
    return (
        ree_assembly_path
        / PARTIAL_SUBPATH
        / experiment_type
        / f"{queue_id}.json"
    )


def load_partial_checkpoint(path: Path, queue_id: str) -> dict[str, Any] | None:
    """Return parsed checkpoint if present and matches queue_id."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if payload.get("queue_id") != queue_id:
        return None
    return payload


def is_resumable_partial(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    if payload.get("complete") is True:
        return False
    return True


def resume_args_for_item(item: dict, ree_assembly_path: Path | None) -> list[str]:
    """CLI flags to pass when restarting a checkpointed experiment."""
    if not item.get("checkpoint_resumable"):
        return []
    if not ree_assembly_path:
        return ["--resume"]
    qid = item.get("queue_id", "")
    exp_type = experiment_type_for_item(item)
    ckpt = partial_checkpoint_path(ree_assembly_path, qid, exp_type)
    payload = load_partial_checkpoint(ckpt, qid)
    if not is_resumable_partial(payload):
        return []
    return ["--resume", f"--checkpoint-path={ckpt}"]
