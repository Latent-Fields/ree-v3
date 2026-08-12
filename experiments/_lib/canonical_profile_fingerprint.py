"""
Canonical-profile freeze + persistence (Option C of
REE_assembly/evidence/planning/architecture_epoch_investigation.md section 7-8).

Reuses arm_fingerprint.py's `compute_substrate_hash()` -- the exact whole-tree
content hash arm_fingerprint mints per experiment cell -- to bind a declared
canonical profile's content to the substrate it was declared against, then
PERSISTS the result as a named, versioned JSON artifact under
REE_assembly/docs/architecture/canonical_profiles/<name>.json.

WHY PERSISTED, WHEN ARM_FINGERPRINT'S OWN HASHES ARE NOT. arm_fingerprint.py's
own module docstring (machine_class()) states its arm-reuse fingerprint is
"persisted NOWHERE ... a fingerprint tag change is a hard cut" -- an accepted
trade for an ephemeral per-cell cache key recomputed fresh every run. A
canonical profile is the opposite case: its whole point is to be a stable
identity a manifest can point back to months later, so it must be written down
once and never silently recomputed differently. This module is the
persistence that gap needed, scoped to profiles only -- it does not touch or
migrate the general arm-reuse cache arm_fingerprint/arm_reuse serve.

Deliberately stdlib-only, mirroring arm_fingerprint.py's own guarantee
("Stdlib only (importable without ree_core)"). It works entirely off the
plain-data {"name", "version", "description", "overrides", "notes"} shape
`ree_core.utils.canonical_profile.CanonicalProfileSpec.as_dict()` produces --
this module never imports ree_core, so a caller can freeze/persist a profile
without torch installed.

ASCII-only output (repo rule).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# arm_fingerprint is a sibling module in this package and stdlib-only. Same
# triple-fallback import shape manifest_core.py and arm_reuse.py use.
try:  # normal package import
    from experiments._lib import arm_fingerprint as _afp  # type: ignore
except Exception:  # pragma: no cover - path-dependent fallbacks
    try:
        from . import arm_fingerprint as _afp  # type: ignore
    except Exception:
        import arm_fingerprint as _afp  # type: ignore

# pack_writer lives one level up (experiments/, not experiments/_lib/). Its
# resolve_evidence_experiments_dir() is the worktree-aware REE_assembly
# locator (see that function's own docstring for the nested-worktree bug it
# exists to fix); reused here rather than re-deriving REE_assembly's path with
# a hardcoded parents[N] arithmetic, which is exactly the bug class it fixes.
try:  # normal package import
    from experiments import pack_writer as _pack_writer  # type: ignore
except Exception:  # pragma: no cover - path-dependent fallbacks
    try:
        from .. import pack_writer as _pack_writer  # type: ignore
    except Exception:
        _experiments_dir = str(Path(__file__).resolve().parents[1])
        if _experiments_dir not in sys.path:
            sys.path.insert(0, _experiments_dir)
        import pack_writer as _pack_writer  # type: ignore

FREEZE_SCHEMA = "canonical_profile_freeze/v1"

CANONICAL_PROFILES_SUBPATH = ("docs", "architecture", "canonical_profiles")


def _canonical_json(obj: Any) -> str:
    # Same convention as arm_fingerprint._canonical_json: sorted keys, no
    # whitespace jitter, ASCII-escaped -- stable across processes.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_canonical_profiles_dir(script_path: Any) -> Path:
    """REE_assembly/docs/architecture/canonical_profiles/, worktree-aware.

    Derived from pack_writer.resolve_evidence_experiments_dir(script_path),
    which returns .../REE_assembly/evidence/experiments -- its grandparent is
    the REE_assembly repo root, joined here with docs/architecture/
    canonical_profiles instead of evidence/experiments.
    """
    evidence_experiments = _pack_writer.resolve_evidence_experiments_dir(script_path)
    assembly_root = evidence_experiments.parent.parent
    return assembly_root.joinpath(*CANONICAL_PROFILES_SUBPATH)


def freeze_profile(
    profile: Mapping[str, Any],
    *,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute the frozen, hash-addressed artifact for one declared profile version.

    `profile` is the plain-data shape CanonicalProfileSpec.as_dict() produces:
    {"schema", "name", "version", "description", "overrides", "notes"}.
    Reuses arm_fingerprint.compute_substrate_hash() UNSCOPED (whole-tree, the
    same safe default arm_fingerprint itself uses) to bind the profile to the
    substrate it was declared against: a profile's OVERRIDES dict can be
    content-identical across two commits while the substrate those overrides
    apply to changes meaning underneath them, and canonical_profile_hash must
    change when that happens -- exactly the reasoning that puts substrate_hash
    inside compute_arm_fingerprint's own fp_input.

    Returns a JSON-serialisable dict. Does not write anything -- see
    persist_profile_freeze for that.
    """
    name = str(profile["name"])
    version = str(profile["version"])
    overrides = dict(profile.get("overrides") or {})
    sub = _afp.compute_substrate_hash(repo_root=repo_root)
    fp_input = {
        "schema": FREEZE_SCHEMA,
        "name": name,
        "version": version,
        "overrides": overrides,
        "substrate_hash": sub["substrate_hash"],
    }
    canonical_profile_hash = _sha256_hex(_canonical_json(fp_input))
    return {
        "schema": FREEZE_SCHEMA,
        "name": name,
        "version": version,
        "description": str(profile.get("description", "")),
        "notes": str(profile.get("notes", "")),
        "overrides": overrides,
        "substrate_hash": sub["substrate_hash"],
        "substrate_n_files": sub["n_files"],
        "canonical_profile_hash": canonical_profile_hash,
    }


def _profile_file_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.json"


def persist_profile_freeze(
    freeze: Mapping[str, Any],
    *,
    out_dir: Path,
    frozen_at_utc: str,
    force: bool = False,
) -> Path:
    """Write (or update) the persisted artifact for `freeze["name"]`.

    One file per profile NAME (`<name>.json`), holding every declared
    VERSION's freeze under a top-level `"versions"` map -- so a profile's
    whole version history lives in one reviewable artifact instead of one
    file per version.

    IMMUTABILITY: re-persisting an ALREADY-recorded version whose
    canonical_profile_hash differs from what is on disk is REFUSED (raises
    ValueError) unless force=True -- a frozen version's content must not
    silently change; that is the entire point of freezing it. Re-persisting
    with an IDENTICAL hash is always allowed and is a no-op beyond confirming
    agreement (idempotent re-run of the freeze).

    `frozen_at_utc` is caller-supplied (repo convention: `date -u
    +"%Y-%m-%dT%H:%M:%SZ"`, per CLAUDE.md's Timestamps rule -- this module
    never calls a wall-clock API itself) and is stamped only the FIRST time a
    version is recorded; re-persisting an unchanged version leaves the
    original `frozen_at_utc` alone rather than overwriting it with the
    re-check time.
    """
    name = str(freeze["name"])
    version = str(freeze["version"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _profile_file_path(out_dir, name)

    if path.exists():
        with open(path) as fh:
            doc = json.load(fh)
    else:
        doc = {"schema": FREEZE_SCHEMA, "name": name, "versions": {}}

    existing = doc.get("versions", {}).get(version)
    if existing is not None:
        if existing.get("canonical_profile_hash") != freeze["canonical_profile_hash"] and not force:
            raise ValueError(
                f"refusing to overwrite frozen profile {name}@{version}: "
                f"recorded hash {existing.get('canonical_profile_hash')!r} != "
                f"new hash {freeze['canonical_profile_hash']!r} (pass force=True "
                "to knowingly replace a frozen version)"
            )
        record = dict(freeze)
        record["frozen_at_utc"] = existing.get("frozen_at_utc", frozen_at_utc)
    else:
        record = dict(freeze)
        record["frozen_at_utc"] = frozen_at_utc

    doc.setdefault("versions", {})[version] = record
    doc["name"] = name
    doc["schema"] = FREEZE_SCHEMA

    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)
    return path


def load_profile_freeze(out_dir: Path, name: str, version: str) -> Optional[Dict[str, Any]]:
    """Read back one persisted version's freeze, or None if not recorded."""
    path = _profile_file_path(out_dir, name)
    if not path.exists():
        return None
    with open(path) as fh:
        doc = json.load(fh)
    return doc.get("versions", {}).get(version)


__all__ = [
    "FREEZE_SCHEMA",
    "CANONICAL_PROFILES_SUBPATH",
    "resolve_canonical_profiles_dir",
    "freeze_profile",
    "persist_profile_freeze",
    "load_profile_freeze",
]
