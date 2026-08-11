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
import shutil
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


DRY_RUN_SCRATCH_DIRNAME = "ree_dry_run_manifests"


def _iter_companion_files(manifest_path: Path) -> list[Path]:
    """Enumerate a manifest's companion side-files (e.g. fishtank episode logs).

    Mirrors experiment_runner._collect_companion_files's two sources -- kept as
    an independent, stdlib-only copy rather than an import, because that
    module drags in the full (torch-importing) runner stack this one is
    deliberately kept free of:
      1. A `companion_files` list declared in the manifest body, each entry
         resolved relative to the MANIFEST's own directory (absolute paths
         taken as-is). This is the write_flat_manifest case: the manifest
         lands flat in evidence/experiments/ while the declared entry points
         back into the per-experiment-type subdirectory, e.g.
         "v3_exq_913_.../v3_exq_913..._episode_log.json".
      2. Auto-discovery of THIS manifest's own `<stem>_episode_log.json` next
         to it -- the direct-write drivers (no write_flat_manifest) name the
         sidecar as exactly the manifest's stem plus that suffix and declare
         no companion_files. Deliberately scoped to this one expected name,
         NOT a directory-wide `*_episode_log.json` glob: a per-experiment-type
         directory accumulates one manifest + sidecar per historical run, and
         a wildcard would sweep in a PRIOR REAL run's sidecar and relocate
         (i.e. delete from evidence/) evidence that has nothing to do with the
         manifest being processed. Confirmed live 2026-08-11: an early version
         of this function using `manifest_dir.glob("*_episode_log.json")`
         moved a real, already-committed 2026-04-25 episode_log out of
         evidence/experiments/v3_exq_483_sd037_broadcast_override_4arm/ while
         relocating an unrelated --dry-run smoke's manifest from the same
         directory; recovered via `git checkout HEAD --` before landing.

    Best-effort: a missing/unparseable manifest yields an empty list rather
    than raising, so a malformed manifest never blocks its own relocation.
    """
    companions: list[Path] = []
    seen: set[Path] = set()
    manifest_dir = manifest_path.parent
    try:
        manifest_resolved = manifest_path.resolve()
    except OSError:
        return companions

    def _add(p: Path) -> None:
        try:
            rp = p.resolve()
        except OSError:
            return
        if rp == manifest_resolved or rp in seen or not rp.is_file():
            return
        seen.add(rp)
        companions.append(rp)

    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        doc = None
    declared = doc.get("companion_files") if isinstance(doc, dict) else None
    if isinstance(declared, (list, tuple)):
        for entry in declared:
            if isinstance(entry, str) and entry:
                p = Path(entry)
                _add(p if p.is_absolute() else manifest_dir / entry)

    _add(manifest_dir / f"{manifest_path.stem}_episode_log.json")

    return companions


def _relocate_dry_run_manifest(manifest_path: Path) -> Path:
    """Move a --dry-run smoke manifest out of the evidence/experiments tree.

    A --dry-run smoke is NOT evidence (e.g. 1 seed, a handful of toy episodes),
    but most experiment scripts' write_manifest() writes to
    evidence/experiments/ unconditionally. Left there, the manifest is reachable
    by build_experiment_indexes.py and surfaces in pending_review.md as an
    action-required FAIL (incident V3-EXQ-696, 2026-06-21). Relocating it to a
    throwaway scratch dir under the system tempdir means it never reaches the
    indexer or the review queue at all.

    Also relocates the manifest's companion side-files (see
    _iter_companion_files), e.g. the `*_episode_log.json` several fishtank-feed
    drivers write directly to evidence/experiments/ alongside (or, for
    write_flat_manifest callers, in a different subdirectory than) the
    manifest, before calling this. Left behind, that sidecar is a permanent
    orphan in the real evidence tree with no manifest to explain it -- twice
    mistaken for a crashed run with a missing manifest rather than an ordinary
    --dry-run leftover (see this fix's changelog entry). Companions are moved
    BEFORE the manifest itself, since manifest-relative companion_files entries
    are resolved against the manifest's pre-move directory.

    Best-effort throughout: on any error (e.g. missing file, permission) the
    original manifest path is returned unchanged, and a single companion
    failing to move never blocks the manifest's own relocation.
    generate_pending_review.py independently excludes dry_run-flagged
    manifests, so this relocation is the going-forward belt-and-suspenders,
    not the sole guard.
    """
    try:
        if not manifest_path.exists():
            return manifest_path
        scratch = Path(tempfile.gettempdir()) / DRY_RUN_SCRATCH_DIRNAME
        scratch.mkdir(parents=True, exist_ok=True)

        for companion in _iter_companion_files(manifest_path):
            try:
                cdest = scratch / companion.name
                if cdest.exists():
                    cdest.unlink()
                shutil.move(str(companion), str(cdest))
            except Exception:
                pass

        dest = scratch / manifest_path.name
        # shutil.move handles the cross-filesystem case (tempdir vs evidence on
        # a different mount) that os.replace would reject.
        if dest.exists():
            dest.unlink()
        shutil.move(str(manifest_path), str(dest))
        return Path(dest)
    except Exception:
        return manifest_path


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
    dry_run: bool = False,
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
        dry_run: Pass args.dry_run. When True and manifest_path is set, the
            just-written manifest is relocated out of evidence/experiments/ to
            a throwaway scratch dir (system tempdir / ree_dry_run_manifests/)
            BEFORE the sentinel is written, and the sentinel records the scratch
            path. This keeps --dry-run smoke manifests out of the indexer and
            pending_review.md (incident V3-EXQ-696). Default False is
            bit-identical to the pre-existing behaviour.

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
        resolved_manifest = Path(manifest_path).resolve()
        if dry_run:
            # Move the dry-run manifest out of evidence/ so it never reaches the
            # indexer or pending_review; record the scratch path in the sentinel.
            resolved_manifest = _relocate_dry_run_manifest(resolved_manifest).resolve()
        manifest_str = str(resolved_manifest)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "queue_id": qid,
        "run_id": run_id,
        "outcome": outcome,
        "exit_reason": exit_reason,
        "manifest_path": manifest_str,
        "dry_run": dry_run,
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
