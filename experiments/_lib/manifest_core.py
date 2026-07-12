"""
Always-record core stamper for the Experimental Recording Standard (V3+).

Standard: REE_assembly/evidence/planning/experimental_recording_standard_2026-07-12.md
(section 3b "ALWAYS-record core"). This is the highest-value hardening item: the
2026-07-12 corpus audit found 0% of flat manifests record a substrate hash, which is
precisely why no historical baseline can be safely reused
(arm_reuse_fingerprint_plan.md:128-133).

Purpose
-------
`stamp_recording_core(manifest, config=..., seeds=..., script_path=...)` merges the
mandatory always-core provenance/reproducibility fields onto an experiment manifest in
ONE line, so an author cannot forget them and every manifest carries the same fixed
skeleton (standard principle 1: small mandatory identity+provenance core). It is a
NO-OP-SAFE additive merge: by default it fills only fields that are absent/empty, never
clobbering a value the script already set (pass `overwrite=True` to force).

Always-core fields it stamps (standard 3b)
------------------------------------------
  recording_schema : "rec/v1"  -- the self-declaring manifest-shape version.
  substrate_hash   : content hash over ree_core/** + env + _lib/** (+ the driver
                     script). For a MULTI-ARM run it is HOISTED from
                     arm_results[i].arm_fingerprint.substrate_hash (already computed by
                     the arm-fingerprint machinery) so the top-level value matches the
                     per-cell fingerprints by construction; for a single-arm run it is
                     computed fresh via experiments/_lib/arm_fingerprint.py.
  machine          : socket.gethostname() (or a caller override -- the hub records
                     "ree-cloud-1" although its hostname is "ree-worker-1").
  machine_class    : arm_fingerprint.machine_class() -- fingerprint equality is
                     machine-class-bound, so this is the class the substrate_hash is
                     valid within.
  elapsed_seconds  : wallclock. Pass it directly, or pass started_at (a perf_counter()
                     value captured at run start) and the helper computes the delta.
  config           : the full config snapshot (env params + hyperparameters + schedule).
  seeds            : the explicit seed LIST (a single int is coerced to [int]).

Design (standard principle 4: additive, forward-compatible)
-----------------------------------------------------------
Additive-only and non-destructive: unknown/older manifests keep every field they had;
new fields are only ADDED. Safe to call unconditionally at manifest-build time in any
experiment_purpose (evidence / diagnostic / baseline).

ASCII-only output (repo rule). Stdlib + arm_fingerprint (itself stdlib-only), so this is
importable without torch/ree_core.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

# arm_fingerprint is a sibling module in this package and stdlib-only. Import it
# robustly across the several ways experiment scripts put experiments/ on sys.path
# (package import, _lib-on-path, cwd=experiments/).
try:  # normal package import (scripts do `from experiments._lib... import ...`)
    from experiments._lib import arm_fingerprint as _afp  # type: ignore
except Exception:  # pragma: no cover - path-dependent fallbacks
    try:
        from . import arm_fingerprint as _afp  # type: ignore
    except Exception:
        import arm_fingerprint as _afp  # type: ignore

RECORDING_SCHEMA = "rec/v1"

# The always-core keys this helper is responsible for. Kept as a named tuple so a
# validator (validate_recording.py) can import the canonical list rather than
# re-hardcoding it.
ALWAYS_CORE_KEYS: Sequence[str] = (
    "recording_schema",
    "substrate_hash",
    "machine",
    "machine_class",
    "elapsed_seconds",
    "config",
    "seeds",
)


def _is_empty(value: Any) -> bool:
    """A field counts as absent-for-fill purposes when it is None or an empty
    container/string. A meaningful 0 / False is NOT empty (so an explicit
    elapsed_seconds=0.0 is respected)."""
    if value is None:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict, set)) and len(value) == 0:
        return True
    return False


def _coerce_seed_list(seeds: Any) -> Optional[List[int]]:
    """Normalise seeds to an explicit list of ints, or None if not derivable.

    Accepts a single int, an iterable of ints, or None. Non-int members are kept
    verbatim (best-effort) rather than dropped, so a caller passing e.g. string
    seeds still records something rather than nothing.
    """
    if seeds is None:
        return None
    if isinstance(seeds, bool):  # bool is an int subclass -- treat as a scalar seed
        return [int(seeds)]
    if isinstance(seeds, int):
        return [seeds]
    if isinstance(seeds, (list, tuple, set)):
        out: List[Any] = []
        for s in seeds:
            try:
                out.append(int(s))
            except (TypeError, ValueError):
                out.append(s)
        return out
    return None


def _hoist_multi_arm_substrate_hash(manifest: Mapping[str, Any]) -> Optional[str]:
    """Return arm_results[i].arm_fingerprint.substrate_hash for the first arm that
    carries one, or None if this is not a multi-arm manifest / no arm carries a hash.

    All arms of one run execute against the same substrate, so the first present
    hash is authoritative -- hoisting it keeps the top-level value byte-identical to
    the per-cell fingerprints (standard 3b: "hoist one copy to the top level").
    """
    arm_results = manifest.get("arm_results")
    if not isinstance(arm_results, list):
        return None
    for cell in arm_results:
        if not isinstance(cell, dict):
            continue
        fp = cell.get("arm_fingerprint")
        if isinstance(fp, dict):
            sh = fp.get("substrate_hash")
            if isinstance(sh, str) and sh:
                return sh
    return None


def compute_single_arm_substrate_hash(
    script_path: Optional[Union[str, Path]] = None,
    extra_substrate_paths: Optional[Iterable[Union[str, Path]]] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> str:
    """Compute a top-level substrate_hash for a single-arm run.

    Hashes ree_core/** + env + _lib/** (the arm-fingerprint substrate glob) plus the
    driver script (so a driver edit correctly flips the hash), matching the
    include_driver_script_in_hash=True default of the arm-fingerprint machinery.
    """
    extra: List[Path] = []
    if script_path:
        extra.append(Path(script_path))
    if extra_substrate_paths:
        extra.extend(Path(p) for p in extra_substrate_paths)
    sub = _afp.compute_substrate_hash(
        extra_paths=extra or None,
        repo_root=Path(repo_root) if repo_root else None,
    )
    return str(sub["substrate_hash"])


def stamp_recording_core(
    manifest: Dict[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    seeds: Any = None,
    script_path: Optional[Union[str, Path]] = None,
    *,
    elapsed_seconds: Optional[float] = None,
    started_at: Optional[float] = None,
    machine: Optional[str] = None,
    extra_substrate_paths: Optional[Iterable[Union[str, Path]]] = None,
    repo_root: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Merge the always-record core onto `manifest` in place and return it.

    NO-OP-SAFE: by default only absent/empty fields are filled, so an existing value
    the script deliberately set is preserved (pass overwrite=True to force). A
    meaningful 0/False is NOT treated as empty.

    Parameters
    ----------
    manifest
        The manifest dict being built (mutated in place; also returned for chaining).
    config
        The full config snapshot. Recorded verbatim under `config` (standard 3b
        reproducibility core). If None and the manifest already carries `config`, it
        is left as-is.
    seeds
        The explicit seed list (a single int is coerced to [int]).
    script_path
        The driver script -- `Path(__file__)` from the experiment. Folded into a
        freshly-computed single-arm substrate_hash. Ignored for the hoist path.
    elapsed_seconds
        Wallclock seconds. Takes precedence over started_at.
    started_at
        A time.perf_counter() value captured at run start; elapsed is computed as
        perf_counter() - started_at when elapsed_seconds is not given.
    machine
        Override for the recorded machine name (default socket.gethostname()).
    extra_substrate_paths, repo_root
        Passed through to the single-arm substrate-hash computation.
    overwrite
        Force-overwrite already-present fields (default False -> fill-only).

    Returns the same manifest dict.
    """

    def _fill(key: str, value: Any) -> None:
        if value is None:
            return
        if overwrite or _is_empty(manifest.get(key)):
            manifest[key] = value

    # recording_schema -- the self-declaring version primitive.
    _fill("recording_schema", RECORDING_SCHEMA)

    # substrate_hash -- hoist from the per-arm fingerprints for a multi-arm run,
    # else compute fresh for a single-arm run. Only compute if we would actually
    # fill (avoid the file-hashing cost when the field is already present).
    if overwrite or _is_empty(manifest.get("substrate_hash")):
        hoisted = _hoist_multi_arm_substrate_hash(manifest)
        if hoisted:
            _fill("substrate_hash", hoisted)
        else:
            try:
                _fill(
                    "substrate_hash",
                    compute_single_arm_substrate_hash(
                        script_path=script_path,
                        extra_substrate_paths=extra_substrate_paths,
                        repo_root=repo_root,
                    ),
                )
            except Exception:
                # Never let provenance stamping crash an experiment. A missing
                # substrate_hash is a soft-validate WARN, not a run failure.
                pass

    # machine / machine_class -- where it ran + the class the hash is valid within.
    _fill("machine", machine if machine else socket.gethostname())
    try:
        _fill("machine_class", _afp.machine_class())
    except Exception:
        pass

    # elapsed_seconds -- explicit value wins; else derive from started_at.
    if elapsed_seconds is not None:
        _fill("elapsed_seconds", float(elapsed_seconds))
    elif started_at is not None:
        _fill("elapsed_seconds", float(time.perf_counter() - started_at))

    # config -- the full config snapshot (reproducibility + fingerprinting).
    if config is not None:
        _fill("config", dict(config))

    # seeds -- the explicit seed list.
    seed_list = _coerce_seed_list(seeds)
    if seed_list is not None:
        _fill("seeds", seed_list)

    return manifest


def missing_core_fields(manifest: Mapping[str, Any]) -> List[str]:
    """Return the always-core keys absent/empty on `manifest` (soft-validate helper).

    Used by validate_recording.py to WARN on an under-recorded manifest without
    re-hardcoding the key list. A field present but empty ([] / "" / {}) counts as
    missing; a meaningful 0/False does not.
    """
    return [k for k in ALWAYS_CORE_KEYS if _is_empty(manifest.get(k, None))]


__all__ = [
    "RECORDING_SCHEMA",
    "ALWAYS_CORE_KEYS",
    "compute_single_arm_substrate_hash",
    "stamp_recording_core",
    "missing_core_fields",
]
