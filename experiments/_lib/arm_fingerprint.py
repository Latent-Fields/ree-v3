"""
Arm-reuse fingerprint (Phase 0 -- INSTRUMENT ONLY).

Design plan: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md

Purpose
-------
Compute a content-addressed fingerprint for a single (seed x arm) experiment
cell so that, in a LATER phase, a cell that is provably the same random variable
as a previously-computed one need not be re-run. PHASE 0 EMITS THE FINGERPRINT
ONLY -- nothing here ever skips, caches, or reuses a computation. It is pure and
side-effect-free except for reading source files to hash them.

Governing asymmetry (see plan section 2): a false cache-HIT corrupts a scientific
conclusion; a false cache-MISS only wastes compute. So the fingerprint is
deliberately OVER-inclusive -- when unsure whether something affects the cell, it
goes into the hash. Over-inclusion only causes (cheap) false misses.

What the fingerprint binds (Regime A, plan section 2.3 / 3.2)
------------------------------------------------------------
  substrate_hash : sha256 over the CONTENT of the source the cell executes
                   (ree_core/**, the env module, _harness.py, _lib/**, and the
                   calling experiment script). Content hash, NOT git SHA --
                   this workflow runs dirty trees constantly, so a commit SHA
                   would falsely match across uncommitted edits.
  config_slice   : the resolved config the cell reads. WHOLE config by default
                   (decision 3, whole-config default); callers may pass a
                   narrowed dict, recorded via config_slice_declared.
  seed           : int.
  machine_class  : platform/arch tag -- Regime A only reuses within one class.
  regime         : "A".

reuse_eligible
--------------
A cell is only a candidate for FUTURE reuse if it is a pure function of
(substrate, config_slice, seed) -- independent of iteration order and global
mutable state (plan section 2.2: the 643 "ARM_A runs first" hazard). The caller
asserts this by passing rng_fully_reset=True AFTER calling reset_all_rng() (or an
equivalent complete reset) at cell entry. If it cannot, the fingerprint is still
emitted for observability but flagged reuse_eligible=False so no future phase can
ever serve it.

ASCII-only output (repo rule). Stdlib only (importable without ree_core).
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

FINGERPRINT_SCHEMA = "arm_fp/v1"
REGIME = "A"

_REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/

# Source trees whose CONTENT defines a cell's computation. Globs are relative to
# the ree-v3 repo root. Kept deliberately broad (over-inclusion -> false misses
# only). The calling script's own path is added dynamically in compute_*.
_SUBSTRATE_GLOBS: Sequence[str] = (
    "ree_core/**/*.py",
    "experiments/_harness.py",
    "experiments/_metrics.py",
    "experiments/_lib/**/*.py",
)


def _canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, no whitespace jitter, ASCII-escaped.

    Non-JSON-native values (e.g. numpy scalars, tuples) are coerced to a stable
    string form so the hash is reproducible across processes.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=_json_default)


def _json_default(o: Any) -> Any:
    # tuples -> lists handled by json natively; this catches numpy/torch scalars
    # and anything else without leaking memory addresses into the hash.
    for attr in ("item", "tolist"):
        fn = getattr(o, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return f"<{type(o).__name__}:{o!r}>"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def machine_class() -> str:
    """Coarse architecture tag. Regime A only reuses within one class.

    Intentionally coarse (system + machine arch + py major.minor). Two hosts of
    the same OS/arch/python are treated as one class; float-rounding determinism
    is assumed stable within a class and NOT across classes.
    """
    return "{sys}-{arch}-py{maj}.{minr}".format(
        sys=platform.system().lower(),
        arch=(platform.machine() or "unknown").lower(),
        maj=sys.version_info.major,
        minr=sys.version_info.minor,
    )


def compute_substrate_hash(
    extra_paths: Optional[Iterable[Path]] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Hash the CONTENT of every substrate source file the cell can execute.

    Returns a dict: {"substrate_hash", "n_files", "missing": [...]}. Files are
    sorted by repo-relative path before hashing so the result is order-stable.
    A missing expected file is recorded (not silently skipped) so a stripped
    checkout produces a visibly different hash rather than a false match.
    """
    root = (repo_root or _REPO_ROOT).resolve()
    paths: Dict[str, Path] = {}
    for g in _SUBSTRATE_GLOBS:
        for p in root.glob(g):
            if p.is_file():
                paths[str(p.relative_to(root))] = p
    for ep in extra_paths or ():
        ep = Path(ep).resolve()
        try:
            rel = str(ep.relative_to(root))
        except ValueError:
            rel = ep.name  # outside repo root: name-only key, still hashed
        if ep.is_file():
            paths[rel] = ep

    h = hashlib.sha256()
    missing: List[str] = []
    for rel in sorted(paths):
        p = paths[rel]
        try:
            content = p.read_bytes()
        except OSError:
            missing.append(rel)
            continue
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(_sha256_hex(content).encode("ascii"))
        h.update(b"\n")
    for rel in missing:
        h.update(b"MISSING\0")
        h.update(rel.encode("utf-8"))
        h.update(b"\n")
    return {
        "substrate_hash": h.hexdigest(),
        "n_files": len(paths) - len(missing),
        "missing": missing,
    }


def reset_all_rng(seed: int) -> None:
    """Complete per-cell RNG reset (plan section 2.2 hardening fix).

    Resets every RNG a cell can touch so the cell is order-independent:
    Python `random`, numpy, torch (+cuda), and the harness module-level
    `_action_random`. Call this at cell entry; then pass rng_fully_reset=True to
    compute_arm_fingerprint. Imports of numpy/torch/_harness are lazy so this
    module stays importable without them.
    """
    random.seed(seed)
    try:
        import numpy as _np  # noqa
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch  # noqa
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # Reseed the harness fallback RNG (module-level, unseeded by default:
    # experiments/_harness.py:70) so a cell that falls back to it is still a
    # pure function of `seed`.
    try:
        from experiments import _harness as _h  # type: ignore
        _h._action_random.seed(seed)
    except Exception:
        try:
            import _harness as _h2  # type: ignore
            _h2._action_random.seed(seed)
        except Exception:
            pass


def compute_arm_fingerprint(
    *,
    config_slice: Mapping[str, Any],
    seed: int,
    script_path: Optional[Path] = None,
    rng_fully_reset: bool,
    config_slice_declared: bool = False,
    extra_substrate_paths: Optional[Iterable[Path]] = None,
    repo_root: Optional[Path] = None,
    extra_ineligible_reasons: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute the per-cell fingerprint payload to embed in a manifest arm row.

    Parameters
    ----------
    config_slice
        The resolved config the cell reads. Pass the WHOLE config for the
        default (safe) behaviour; pass a narrowed dict + config_slice_declared=
        True only if you have deliberately verified the OFF path reads nothing
        outside it.
    seed
        The cell seed.
    script_path
        The calling experiment script (its content joins the substrate hash).
        Pass `Path(__file__)` from the experiment.
    rng_fully_reset
        True iff reset_all_rng(seed) (or equivalent) ran at cell entry. False
        forces reuse_eligible=False.
    extra_ineligible_reasons
        Any caller-known reasons the cell must never be reused (e.g.
        "shared_optimizer_across_arms"). Their presence forces
        reuse_eligible=False.

    Returns a JSON-serialisable dict. Embed it as arm_results[i]["arm_fingerprint"].
    NOTHING here reads or consults any cache -- Phase 0 is emit-only.
    """
    sub = compute_substrate_hash(
        extra_paths=[script_path] if script_path else None,
        repo_root=repo_root,
    )
    if extra_substrate_paths:
        # fold any additional declared deps into the hash deterministically
        extra = compute_substrate_hash(extra_paths=extra_substrate_paths,
                                       repo_root=repo_root)
        sub["substrate_hash"] = _sha256_hex(
            (sub["substrate_hash"] + ":" + extra["substrate_hash"]).encode("ascii")
        )

    mc = machine_class()
    fp_input = {
        "schema": FINGERPRINT_SCHEMA,
        "substrate_hash": sub["substrate_hash"],
        "config_slice": dict(config_slice),
        "seed": int(seed),
        "machine_class": mc,
        "regime": REGIME,
    }
    fingerprint = _sha256_hex(_canonical_json(fp_input).encode("utf-8"))

    reasons: List[str] = list(extra_ineligible_reasons or ())
    if not rng_fully_reset:
        reasons.append("incomplete_rng_reset")
    if sub["missing"]:
        reasons.append("substrate_files_missing")
    reuse_eligible = len(reasons) == 0

    return {
        "schema": FINGERPRINT_SCHEMA,
        "arm_fingerprint": fingerprint,
        "substrate_hash": sub["substrate_hash"],
        "substrate_n_files": sub["n_files"],
        "machine_class": mc,
        "regime": REGIME,
        "seed": int(seed),
        "config_slice_declared": bool(config_slice_declared),
        "reuse_eligible": reuse_eligible,
        "reuse_ineligible_reasons": reasons,
        # NOTE: Phase 0 is emit-only. No reuse is ever performed off this value.
    }


__all__ = [
    "FINGERPRINT_SCHEMA",
    "REGIME",
    "machine_class",
    "compute_substrate_hash",
    "reset_all_rng",
    "compute_arm_fingerprint",
]
