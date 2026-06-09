"""
Arm-reuse consumer (Phase 1 -- refuse-by-default cite-baseline reuse).

Design plan: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md section 9.

Purpose
-------
Phase 0 (experiments/_lib/arm_fingerprint.py) RECORDS a per-cell arm_fingerprint
into experiment manifests. This module is the Phase 1 CONSUMER: it lets a new
iteration (e.g. V3-EXQ-643b / 610g) SKIP re-training its OFF/baseline arm by
reusing a previously-minted cell -- but ONLY under a strict refuse-by-default
gate so it can never substitute a non-identical baseline.

Governing asymmetry (plan section 2): a false cache-HIT corrupts a scientific
conclusion; a false cache-MISS only wastes compute. So every uncertainty resolves
to REFUSE. When in doubt, return None and let the caller run the arm normally.

HARD PREREQUISITE GATE (plan section 9.0)
-----------------------------------------
Reuse must NOT be enabled until the cross-instance determinism check has passed
(the V3-EXQ-610 OFF baseline minted on cloud-2 AND cloud-3, compared within a
written tolerance). This module is INERT until an experiment actually calls
try_reuse_cell with a cite_run_id; no experiment is wired to call it until the
gate passes. Building + unit-testing this module before the gate is explicitly
permitted (the index/helper/tests are inert); wiring an experiment to skip an arm
is not.

How a HIT actually happens (plan section 9.2 / 9.4)
--------------------------------------------------
try_reuse_cell recomputes the requesting cell's fingerprint with the SAME
compute_arm_fingerprint that Phase 0 emitted, then looks it up in
arm_fingerprint_index.json (built by the indexer). Fingerprint EQUALITY already
implies same substrate_hash + config_slice + seed + machine_class + regime, so
the machine-class and substrate guards are intrinsic: a Mac-run iteration cannot
match a cloud-minted baseline -- the fingerprints differ and it simply re-runs.

Driver-script coupling and how to defeat it (the recommended path):
By default the fingerprint's substrate_hash folds in the calling `script_path`'s
content, so a HIT would require the consumer to recompute with the SAME
`script_path` the mint used -- impractical, since a real consumer (610g / 643c)
has its own driver. The supported fix is `include_driver_script_in_hash=False`
(passed to BOTH the mint's compute_arm_fingerprint and this consumer): the driver
script is then excluded from the reuse-critical hash and the OFF cell is anchored
on the canonical baseline module under experiments/_lib/** (already in the
substrate-hash glob) + config_slice + seed + machine_class. A mint and a later
consumer with DIFFERENT drivers -- both built from the same canonical module --
then produce the SAME fingerprint and this consumer HITs. The flag must match on
both sides (an excluded-driver fingerprint can never collide with an included-driver
one). If it does not match, the fingerprint is not in the index and reuse is
REFUSED (the safe outcome). See the /queue-experiment opt-in step.

ASCII-only output (repo rule). Stdlib only.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Import the Phase 0 fingerprint primitives. arm_fingerprint.py is stdlib-only and
# importable standalone; add this _lib dir to sys.path so the import is robust
# regardless of how the calling experiment set up its own path.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from arm_fingerprint import (  # noqa: E402
    FINGERPRINT_SCHEMA,
    compute_arm_fingerprint,
)

# Default location of the index the indexer writes (REE_assembly is a sibling of
# ree-v3 under the REE_Working umbrella). parents: [_lib, experiments, ree-v3,
# REE_Working]. The index lives under REE_assembly/evidence/experiments/.
_DEFAULT_INDEX = (
    Path(__file__).resolve().parents[3]
    / "REE_assembly" / "evidence" / "experiments" / "arm_fingerprint_index.json"
)

INDEX_SCHEMA = "arm_fp_index/v1"

# Refuse reasons (returned in ReuseDecision.reason; logged as "reuse_refused: <reason>").
REFUSE_NO_INDEX = "no_index"
REFUSE_FP_NOT_IN_INDEX = "fingerprint_not_in_index"
REFUSE_CITE_MISMATCH = "cite_run_id_mismatch"
REFUSE_NOT_ELIGIBLE = "cached_not_reuse_eligible"
REFUSE_PARENT_ERROR = "parent_outcome_error"
REFUSE_SUPERSEDED = "superseded"
REFUSE_NEEDED_KEYS = "needed_keys_not_subset"
REFUSE_SCHEMA = "schema_mismatch"
REFUSE_MANIFEST_UNREADABLE = "manifest_unreadable"
REFUSE_CELL_NOT_FOUND = "cell_not_found_in_manifest"


class ReuseDecision:
    """Structured outcome of a reuse evaluation (one cell)."""

    __slots__ = ("reused", "reason", "cell", "fingerprint", "source_run_id")

    def __init__(
        self,
        reused: bool,
        reason: str,
        fingerprint: str,
        cell: Optional[Dict[str, Any]] = None,
        source_run_id: Optional[str] = None,
    ) -> None:
        self.reused = reused
        self.reason = reason
        self.cell = cell
        self.fingerprint = fingerprint
        self.source_run_id = source_run_id

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            "ReuseDecision(reused=%r, reason=%r, fp=%r, source_run_id=%r)"
            % (self.reused, self.reason, (self.fingerprint or "")[:12], self.source_run_id)
        )


def _utc_iso_now() -> str:
    # Runtime stamp for provenance; ASCII, UTC.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_index(index_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(index_path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _resolve_manifest_path(
    manifest_rel: str, index_path: Path, assembly_root: Optional[Path]
) -> Path:
    """Resolve a manifest path stored in the index relative to REE_assembly root.

    The index stores manifest_path relative to the REE_assembly repo root
    (e.g. "evidence/experiments/<run_id>.json"). Default assembly_root is derived
    from the index location (index is at <assembly>/evidence/experiments/...).
    """
    root = assembly_root if assembly_root is not None else index_path.resolve().parents[2]
    p = Path(manifest_rel)
    return p if p.is_absolute() else (root / p)


def _find_cell_in_manifest(manifest: Mapping[str, Any], fingerprint: str) -> Optional[Dict[str, Any]]:
    """Locate the arm_results cell whose arm_fingerprint.arm_fingerprint == fingerprint."""
    arm_results = manifest.get("arm_results")
    if not isinstance(arm_results, list):
        return None
    for row in arm_results:
        if not isinstance(row, dict):
            continue
        fp = row.get("arm_fingerprint")
        if isinstance(fp, dict) and fp.get("arm_fingerprint") == fingerprint:
            return row
    return None


def evaluate_reuse(
    *,
    config_slice: Mapping[str, Any],
    seed: int,
    script_path: Optional[Path],
    needed_keys: Sequence[str],
    cite_run_id: Optional[str] = None,
    index_path: Optional[Path] = None,
    assembly_root: Optional[Path] = None,
    extra_substrate_paths: Optional[Sequence[Path]] = None,
    repo_root: Optional[Path] = None,
    include_driver_script_in_hash: bool = True,
) -> ReuseDecision:
    """Decide whether the requesting cell may reuse a cached cell. Pure / no I/O side effects.

    Recomputes the requesting fingerprint via compute_arm_fingerprint (the SAME
    function Phase 0 emits) and applies EVERY plan-section-9.2 rule. Returns a
    ReuseDecision; on reuse, .cell is the cached cell stamped with provenance.

    Refuses (reused=False) unless ALL hold:
      1. an index entry exists for the recomputed fingerprint;
      2. cite_run_id (if given) matches the entry's run_id;
      3. cached reuse_eligible AND parent outcome != ERROR AND not superseded;
      4. set(needed_keys) subset of set(cell_keys);
      5. schema matches (arm_fp/v1).
    """
    idx_path = Path(index_path) if index_path is not None else _DEFAULT_INDEX

    # Recompute the requesting cell's fingerprint. rng_fully_reset / declared do NOT
    # enter the hash (only schema/substrate_hash/config_slice/seed/machine_class/
    # regime do), so they are immaterial to the lookup key; we pass True/True for a
    # faithful emit-equivalent payload.
    fp_payload = compute_arm_fingerprint(
        config_slice=config_slice,
        seed=seed,
        script_path=Path(script_path) if script_path is not None else None,
        rng_fully_reset=True,
        config_slice_declared=True,
        extra_substrate_paths=extra_substrate_paths,
        repo_root=repo_root,
        include_driver_script_in_hash=include_driver_script_in_hash,
    )
    fingerprint = fp_payload["arm_fingerprint"]
    requesting_schema = fp_payload["schema"]

    if requesting_schema != FINGERPRINT_SCHEMA:
        # Local module mismatch -- refuse rather than risk a cross-schema hit.
        return ReuseDecision(False, REFUSE_SCHEMA, fingerprint)

    index = _load_index(idx_path)
    if index is None:
        return ReuseDecision(False, REFUSE_NO_INDEX, fingerprint)

    by_fp = index.get("by_fingerprint")
    entry = by_fp.get(fingerprint) if isinstance(by_fp, dict) else None
    if not isinstance(entry, dict):
        return ReuseDecision(False, REFUSE_FP_NOT_IN_INDEX, fingerprint)

    # Rule 5 (schema of the cached entry).
    if str(entry.get("fingerprint_schema")) != FINGERPRINT_SCHEMA:
        return ReuseDecision(False, REFUSE_SCHEMA, fingerprint)

    # Rule 2 (explicit cite -- Phase 1 default, auditable, low blast radius).
    entry_run_id = entry.get("run_id")
    if cite_run_id is not None and str(cite_run_id) != str(entry_run_id):
        return ReuseDecision(False, REFUSE_CITE_MISMATCH, fingerprint, source_run_id=entry_run_id)

    # Rule 3 (eligible, non-ERROR, non-superseded).
    if not bool(entry.get("reuse_eligible", False)):
        return ReuseDecision(False, REFUSE_NOT_ELIGIBLE, fingerprint, source_run_id=entry_run_id)
    if str(entry.get("outcome", "")).upper() == "ERROR":
        return ReuseDecision(False, REFUSE_PARENT_ERROR, fingerprint, source_run_id=entry_run_id)
    if bool(entry.get("superseded", False)):
        return ReuseDecision(False, REFUSE_SUPERSEDED, fingerprint, source_run_id=entry_run_id)

    # Rule 4 (the easy-to-miss correctness trap): the cached cell must actually
    # have recorded every metric this experiment reads off its OFF arm.
    cell_keys = entry.get("cell_keys")
    cell_keys_set = set(cell_keys) if isinstance(cell_keys, list) else set()
    if not set(needed_keys).issubset(cell_keys_set):
        return ReuseDecision(False, REFUSE_NEEDED_KEYS, fingerprint, source_run_id=entry_run_id)

    # All gates passed -- load the manifest and extract the actual cell.
    manifest_rel = entry.get("manifest_path")
    if not isinstance(manifest_rel, str) or not manifest_rel:
        return ReuseDecision(False, REFUSE_MANIFEST_UNREADABLE, fingerprint, source_run_id=entry_run_id)
    manifest_path = _resolve_manifest_path(manifest_rel, idx_path, assembly_root)
    try:
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return ReuseDecision(False, REFUSE_MANIFEST_UNREADABLE, fingerprint, source_run_id=entry_run_id)

    cell = _find_cell_in_manifest(manifest, fingerprint)
    if cell is None:
        return ReuseDecision(False, REFUSE_CELL_NOT_FOUND, fingerprint, source_run_id=entry_run_id)

    # Stamp provenance so the consuming manifest is self-describing (plan 9.3): a
    # reviewer sees exactly what was reused vs freshly computed.
    reused_cell = json.loads(json.dumps(cell, default=str))
    reused_cell["reused_from_run_id"] = entry_run_id
    reused_cell["reused_fingerprint"] = fingerprint
    reused_cell["reused_at_utc"] = _utc_iso_now()
    return ReuseDecision(True, "ok", fingerprint, cell=reused_cell, source_run_id=entry_run_id)


def try_reuse_cell(
    config_slice: Mapping[str, Any],
    seed: int,
    script_path: Optional[Path],
    needed_keys: Sequence[str],
    cite_run_id: Optional[str] = None,
    *,
    index_path: Optional[Path] = None,
    assembly_root: Optional[Path] = None,
    extra_substrate_paths: Optional[Sequence[Path]] = None,
    repo_root: Optional[Path] = None,
    include_driver_script_in_hash: bool = True,
    logger: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Return a cached OFF-arm cell to reuse, or None if reuse is refused.

    Positional signature matches plan section 9.2:
        try_reuse_cell(config_slice, seed, script_path, needed_keys, cite_run_id=None)

    On None, the caller MUST run the arm normally and log "reuse_refused: <reason>"
    (this helper already logs it; the reason is also discoverable via evaluate_reuse).

    Parameters
    ----------
    config_slice, seed, script_path
        The same inputs Phase 0 passed to compute_arm_fingerprint for the OFF cell.
        To get a HIT, these MUST reproduce the mint's fingerprint. The practical
        recipe: build the OFF arm from the lineage's canonical baseline module and
        pass include_driver_script_in_hash=False on BOTH mint and consumer, so the
        differing driver scripts drop out of the reuse key; otherwise reuse is
        refused (safe).
    include_driver_script_in_hash
        Must equal the flag the mint used (default True). Pass False for the
        canonical-baseline-module reuse path described above.
    needed_keys
        The OFF-arm metric keys this experiment reads. If the cached cell did not
        record all of them, reuse is refused (the section-9.2 correctness trap).
    cite_run_id
        Phase 1 default: the mint run_id declared via `reuse_baseline_from`. Must
        match the index entry's run_id or reuse is refused.
    logger
        Optional callable(str); defaults to print. Receives one line per refusal /
        hit, ASCII-only.
    """
    decision = evaluate_reuse(
        config_slice=config_slice,
        seed=seed,
        script_path=script_path,
        needed_keys=needed_keys,
        cite_run_id=cite_run_id,
        index_path=index_path,
        assembly_root=assembly_root,
        extra_substrate_paths=extra_substrate_paths,
        repo_root=repo_root,
        include_driver_script_in_hash=include_driver_script_in_hash,
    )
    log = logger if callable(logger) else print
    fp12 = (decision.fingerprint or "")[:12]
    if decision.reused:
        log(
            "reuse_HIT: seed=%s fp=%s from_run_id=%s (OFF cell reused; treatment arms run fresh)"
            % (seed, fp12, decision.source_run_id)
        )
        return decision.cell
    log("reuse_refused: %s (seed=%s fp=%s)" % (decision.reason, seed, fp12))
    return None


__all__ = [
    "INDEX_SCHEMA",
    "ReuseDecision",
    "evaluate_reuse",
    "try_reuse_cell",
    "REFUSE_NO_INDEX",
    "REFUSE_FP_NOT_IN_INDEX",
    "REFUSE_CITE_MISMATCH",
    "REFUSE_NOT_ELIGIBLE",
    "REFUSE_PARENT_ERROR",
    "REFUSE_SUPERSEDED",
    "REFUSE_NEEDED_KEYS",
    "REFUSE_SCHEMA",
    "REFUSE_MANIFEST_UNREADABLE",
    "REFUSE_CELL_NOT_FOUND",
]
