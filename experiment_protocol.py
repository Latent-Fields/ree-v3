"""
Experiment-runner conformance contract.

Every experiment script's `if __name__ == "__main__":` block MUST end with a
single call to `emit_outcome(...)`. The call atomically writes a sentinel
JSON file that the runner reads after the subprocess exits.

This replaces the old stdout-regex-scraping handshake (RE_DONE_OUTCOME,
RE_SAVED_TO, etc.) which was fragile to script-level wording changes and
caused the 2026-05-08 silent-drop incidents on the Hetzner cloud workers.

Contract:
    1. The runner sets REE_QUEUE_ID + REE_RUNNER_SIGNAL_DIR in the
       subprocess environment.
    2. emit_outcome(outcome="PASS"|"FAIL", manifest_path=..., ...) writes
       <signal_dir>/<queue_id>.json atomically (write-tmp + os.replace).
    3. Runner reads the sentinel after proc.wait(). File present + valid
       -> trust outcome and manifest_path. File missing -> classify ERROR
       (NOT UNKNOWN -> NOT removed from queue).

Manual invocations (not under the runner) auto-fall back to a timestamped
filename under <signal_dir>/_manual/ so a developer running a script by
hand still produces a sentinel for inspection.

Keep this module tiny + ASCII-safe + zero non-stdlib imports. All
experiment scripts must be able to import it without dragging the
ree_core dependency tree.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

SCHEMA_VERSION = "experiment_runner_signal/v1"
DEFAULT_SIGNAL_DIRNAME = "_runner_signals"
VALID_OUTCOMES = ("PASS", "FAIL")
VALID_EXIT_REASONS = ("ok", "fail", "error", "interrupted", "skipped")


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _find_ree_assembly() -> Optional[Path]:
    """Locate REE_assembly/ for sentinel-dir auto-discovery."""
    candidates = []
    here = Path(__file__).resolve()
    candidates.append(here.parent.parent / "REE_assembly")
    candidates.append(Path.home() / "REE_Working" / "REE_assembly")
    p = here.parent
    for _ in range(6):
        sib = p.parent / "REE_assembly"
        if sib.is_dir():
            candidates.append(sib)
        p = p.parent
    for c in candidates:
        if c.is_dir() and (c / "evidence" / "experiments").is_dir():
            return c
    return None


def _resolve_signal_dir() -> Path:
    """REE_RUNNER_SIGNAL_DIR env var wins; else REE_assembly auto-discovery."""
    env_dir = os.environ.get("REE_RUNNER_SIGNAL_DIR")
    if env_dir:
        d = Path(env_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d
    assembly = _find_ree_assembly()
    if assembly is None:
        d = Path(tempfile.gettempdir()) / "ree_runner_signals"
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = assembly / "evidence" / "experiments" / DEFAULT_SIGNAL_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def emit_outcome(
    outcome: str,
    manifest_path: Optional[Union[str, Path]] = None,
    *,
    run_id: Optional[str] = None,
    queue_id: Optional[str] = None,
    exit_reason: str = "ok",
    extra: Optional[Mapping[str, Any]] = None,
    signal_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Emit the runner-conformance sentinel.

    Args:
        outcome: "PASS" or "FAIL". An experiment that ran-to-completion but
            did not meet acceptance criteria is FAIL. A code crash is NOT
            FAIL -- those should propagate as exceptions and the runner
            classifies them ERROR via subprocess returncode + missing
            sentinel.
        manifest_path: Path to the result-manifest JSON the script wrote
            (or None if the script writes no manifest, e.g. dry-runs).
            The runner verifies this path exists before removing the
            queue item.
        run_id: Optional script-level run identifier (e.g.
            "v3_exq_476c_..._20260508T125433Z_v3"). Logged for governance
            cross-reference; not used for the sentinel filename.
        queue_id: Override for REE_QUEUE_ID env var. Use only when the
            script is launched outside the runner.
        exit_reason: Free-form short label for the explorer ("ok" |
            "fail" | "error" | "interrupted" | "skipped"). Strict on
            VALID_EXIT_REASONS; arbitrary strings are accepted but flagged
            for the explorer to surface.
        extra: Optional dict merged into the sentinel under "extra".
            Use for experiment-side diagnostics that don't fit the
            fixed schema.
        signal_dir: Override sentinel directory. Test/diagnostic only;
            default is REE_RUNNER_SIGNAL_DIR / REE_assembly auto-discover.

    Returns:
        Path of the written sentinel.

    Raises:
        ValueError: outcome not in VALID_OUTCOMES.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"emit_outcome: outcome must be one of {VALID_OUTCOMES}, got {outcome!r}"
        )
    if exit_reason not in VALID_EXIT_REASONS:
        # Don't raise -- the runner will surface the literal string. But
        # nudge the user via stderr so unusual reasons show up in dev logs.
        print(
            f"[experiment_protocol] note: exit_reason={exit_reason!r} "
            f"is non-canonical; canonical values are {VALID_EXIT_REASONS}",
            file=sys.stderr,
            flush=True,
        )

    qid = queue_id or os.environ.get("REE_QUEUE_ID")
    sig_dir = Path(signal_dir) if signal_dir is not None else _resolve_signal_dir()
    if qid:
        out_path = sig_dir / f"{qid}.json"
    else:
        manual = sig_dir / "_manual"
        manual.mkdir(parents=True, exist_ok=True)
        stem = run_id or _utc_compact_now()
        out_path = manual / f"{stem}.json"

    manifest_str: Optional[str] = None
    if manifest_path is not None:
        manifest_str = str(Path(manifest_path).resolve())

    payload = {
        "schema_version": SCHEMA_VERSION,
        "queue_id": qid,
        "run_id": run_id,
        "outcome": outcome,
        "exit_reason": exit_reason,
        "manifest_path": manifest_str,
        "emitted_at": _utc_iso_now(),
        "pid": os.getpid(),
        "script": str(Path(sys.argv[0]).resolve()) if sys.argv and sys.argv[0] else None,
    }
    if extra is not None:
        payload["extra"] = dict(extra)

    _atomic_write_json(out_path, payload)
    print(
        f"[experiment_protocol] sentinel written: outcome={outcome} "
        f"queue_id={qid or '<manual>'} -> {out_path}",
        flush=True,
    )
    return out_path
