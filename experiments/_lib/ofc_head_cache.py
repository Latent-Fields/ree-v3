"""Reusable trained-OFC-head cache (SD-033b commitment_closure:GAP-8-affordability).

Design ref: chip-20260730-ofc-deval-affordable. GOVERNING PROBLEM (measured, not
assumed -- see experiments/_lib/baselines/arc071_chunking.py `run_cell` and
ree-v3/experiments/v3_exq_841_mech163_q085_grain_dose_response.py "DEVIATION FROM
THE claims.yaml DESIGN"): the SD-033b OFC-analog devaluation head
(`ree_core/pfc/ofc_analog.py`) emits exactly zero bias until deliberately TRAINED
(`train_state_bias_head` / `train_devaluation_head`, last Linear zeroed at init).
Training requires gradient tracking through the live agent loop, measured on the
hub at roughly 2 s/step against ~0.9 s/step for the same loop under
`torch.no_grad()` -- a per-cell trained-head grid (e.g. 6 arms x 8 seeds on
CausalGridWorldV2) is therefore ~28 h against ~7 h for a no-grad substitute, which
is why claims.yaml's own named design ("SD-033b OFC-analog devaluation; existing
driver lineage V3-EXQ-485g/i/k/m") has repeatedly been substituted for a
policy-independent env-side devaluation in practice (most recently V3-EXQ-841,
`devaluation_route: env_resource_benefit_to_zero`).

WHAT THIS MODULE DOES NOT DO. It does not train a head, does not pick a training
protocol, and does not know anything about CausalGridWorldV2, the SD-054 reef/
forage env the 485-lineage actually trained on, or any specific grid design. That
is deliberately out of scope here (per-experiment scientific judgement belongs in
a /queue-experiment driver, not in shared substrate infra) -- and it matters
concretely, because the 485-lineage's existing trained heads were fit on SD-054
reef/forage, a DIFFERENT env from CausalGridWorldV2 (the arc071_chunking /
Q-085/MECH-163 lineage), so those weights are NOT assumed portable and this module
never ships or bundles them.

WHAT THIS MODULE DOES. A content-addressed, machine-local cache for a trained
OFCAnalog head's weights (`OFCAnalog.head_state_dict()` /
`load_head_state_dict()`), keyed via the SAME content-hash discipline
`experiments/_lib/arm_fingerprint.py` uses for per-cell metric fingerprints:
substrate identity (frozen once per process, see `resolve_substrate_identity`),
`machine_class()`, a caller-declared `config_slice`, and `seed`. A caller pays the
expensive gradient-tracking training cost ONCE for a given key and every later
cell -- in the SAME run or, if the key is still valid, a LATER run -- that shares
the key gets the trained weights via an ordinary forward pass in `no_grad` mode.

GOVERNING ASYMMETRY (mirrors arm_fingerprint.py and
experiments/_lib/baselines/maturation_curriculum.py): a false cache HIT (serving a
head trained under materially different conditions) corrupts a scientific
conclusion; a false MISS only wastes compute by retraining. `config_slice`
therefore defaults to the WHOLE resolved config a caller passes -- narrowing it is
an opt-in the CALLER must justify, exactly like arm_fingerprint's own
`config_slice_declared` contract.

THE AFFORDABILITY LEVER THIS ENABLES (why config_slice narrowing is the point, not
an afterthought). `OFCAnalog.compute_bias` / `compute_devaluation_bias` read only
`(state_code, candidate_world_summaries)`; `state_code` is a gate-modulated EMA of
`world_proj(z_world)` [+ `outcome_pool_weight * outcome_proj(z_harm)`] -- see
`ree_core/pfc/ofc_analog.py` `OFCAnalog.update()`. Nothing in that closure reads
`policy_chunking` / chunk-proposal-injection / `chunk_max_size` or any other
composition-dose parameter. So a multi-arm grid whose ARMS vary only the dose
(e.g. Q-085's A_HIER_S2/S3/S5) can train ONE head per seed under a `config_slice`
that omits the dose field, and reuse it across all dose arms for that seed --
turning an O(arms x seeds) training cost into O(seeds). This is a REUSE OPPORTUNITY
a caller must deliberately construct (by choosing what goes in `config_slice`), not
something this module verifies for you: SOUNDNESS OF A NARROWED config_slice IS
THE CALLER'S OBLIGATION, exactly as it is for arm_fingerprint.compute_arm_fingerprint
itself. Omitting a field the training loop's loss or data generation actually reads
is a false-HIT bug (the ONLY failure direction that matters here); including too
much is always safe, it just narrows the reuse (a cheap false miss). Before
narrowing config_slice to drop a field, re-derive -- do not assume -- that
OFCAnalog's read surface still excludes it; the reasoning above holds only for the
substrate as it exists when this module was written (2026-07-31).

CROSS-DRIVER REUSE (`include_driver_script_in_hash`). Defaults to False here,
the OPPOSITE of arm_fingerprint.compute_arm_fingerprint's own default, because the
whole point of this cache is that a LATER, differently-authored grid experiment
should be able to reuse a head trained by an EARLIER script -- the same
"mint-as-you-go" convention already used for arm-reuse baselines (REE_Working/
CLAUDE.md "Saving a baseline for reuse": `include_driver_script_in_hash=False` so a
mint and a later differently-driven consumer can HIT). Pass True only when the
calling driver's own content IS the training recipe and must be part of the key.

STORAGE. `torch.save`d blobs under `~/.ree_ofc_head_cache` (override via
`REE_OFC_HEAD_CACHE_DIR`), one file per key, atomic write (`os.replace`), a
defensive key-mismatch check on load (any mismatch/corruption/partial write is a
MISS, never a crash), and `REE_OFC_HEAD_CACHE_DISABLE=1` as a global escape hatch.
This mirrors `experiments/_lib/baselines/maturation_curriculum.py`'s frozen-prefix
cache mechanics exactly, on purpose -- same asymmetry, same failure-mode handling,
same operator knobs.

ASCII-only output (repo rule). torch is required (state_dicts are tensors).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch

_LIB_DIR = Path(__file__).resolve().parent            # experiments/_lib
_EXP_DIR = _LIB_DIR.parent                             # experiments
_REPO_ROOT = _EXP_DIR.parent                           # ree-v3
for _p in (str(_REPO_ROOT), str(_EXP_DIR), str(_LIB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the arm_fingerprint content-hash + substrate-identity primitives so this
# cache's key obeys the same over-inclusion discipline and the same
# executed-substrate freeze (a mid-run checkout move must not key a saved head to
# a substrate it was not trained on -- identical reasoning to
# maturation_curriculum's _prefix_key, see that module's docstring).
from arm_fingerprint import (  # noqa: E402
    _canonical_json,
    _sha256_hex,
    machine_class,
    resolve_substrate_identity,
)

OFC_HEAD_CACHE_SCHEMA = "ofc_head_cache/v1"


# ---------------------------------------------------------------------------
# Cache directory / disable knobs (mirrors maturation_curriculum.py).
# ---------------------------------------------------------------------------

def _cache_dir(cache_dir: Optional[Path]) -> Path:
    if cache_dir is not None:
        d = Path(cache_dir)
    else:
        env = os.environ.get("REE_OFC_HEAD_CACHE_DIR")
        d = Path(env) if env else (Path.home() / ".ree_ofc_head_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_disabled() -> bool:
    return os.environ.get("REE_OFC_HEAD_CACHE_DISABLE", "").strip() not in (
        "", "0", "false", "False",
    )


# ---------------------------------------------------------------------------
# Key computation.
# ---------------------------------------------------------------------------

def compute_head_key(
    *,
    config_slice: Mapping[str, Any],
    seed: int,
    config_slice_declared: bool = False,
    substrate_scope: Optional[Sequence[str]] = None,
) -> str:
    """Content-addressed key for a trained OFC head.

    `config_slice` -- the resolved config the CALLER's training loop reads. Pass
    the WHOLE relevant config for the safe default; pass a narrowed dict +
    config_slice_declared=True only once you have verified (per this module's
    docstring) that the omitted fields are genuinely inert for OFC head training.
    `substrate_scope` -- None (default) hashes the whole ree_core/**+
    experiments/_lib/** trees via resolve_substrate_identity, the same safe
    default arm_fingerprint itself uses. A narrower author-declared scope may be
    passed once a caller has proven it conservative (see
    experiments/_lib/substrate_scope_guard.py), mirroring
    maturation_curriculum's dependency-scoped hashing -- not attempted here
    because no training driver exists yet to derive a provable scope from.
    """
    scope = list(substrate_scope) if substrate_scope is not None else None
    sub = resolve_substrate_identity(scope=scope)
    payload: Dict[str, Any] = {
        "schema": OFC_HEAD_CACHE_SCHEMA,
        "substrate_hash": sub["substrate_hash"],
        "substrate_scope_declared": scope is not None,
        "substrate_scope": scope,
        "machine_class": machine_class(),
        "config_slice": dict(config_slice),
        "config_slice_declared": bool(config_slice_declared),
        "seed": int(seed),
    }
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


# ---------------------------------------------------------------------------
# Blob load / store (mirrors maturation_curriculum._cache_load / _cache_store).
# ---------------------------------------------------------------------------

def _default_logger(msg: str) -> None:
    print(msg)


def load_blob(
    key: str,
    cache_dir: Optional[Path] = None,
    logger: Callable[[str], None] = _default_logger,
) -> Optional[Dict[str, Any]]:
    """Return the stored blob for `key`, or None on MISS (never raises).

    A MISS covers: cache disabled, no file, unreadable/corrupt file, or a stored
    key that does not match `key` (defends against a filename collision or a
    truncated write) -- all treated identically, per the governing asymmetry: a
    false HIT is the only outcome that must never happen.
    """
    if _cache_disabled():
        return None
    path = _cache_dir(cache_dir) / f"{key}.pt"
    if not path.is_file():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # corrupt / partial / version skew -> MISS
        logger("ofc_head_cache: unreadable (%s) -> MISS: %s" % (path.name, exc))
        return None
    if not isinstance(blob, dict) or blob.get("key") != key:
        logger("ofc_head_cache: key mismatch on %s -> MISS" % path.name)
        return None
    return blob


def store_blob(
    key: str,
    blob: Dict[str, Any],
    cache_dir: Optional[Path] = None,
    logger: Callable[[str], None] = _default_logger,
) -> None:
    """Persist `blob` (must already carry blob["key"] == key) under `key`.

    Atomic within a filesystem (`os.replace`) so a parallel worker never observes
    a partially-written file; a write failure is logged and swallowed (a store
    failure only costs a future re-train, never correctness).
    """
    if _cache_disabled():
        return
    d = _cache_dir(cache_dir)
    path = d / f"{key}.pt"
    tmp = d / f"{key}.pt.tmp.{os.getpid()}"
    try:
        torch.save(blob, tmp)
        os.replace(tmp, path)
    except Exception as exc:
        logger("ofc_head_cache: store failed for %s: %s" % (path.name, exc))
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# OFCAnalog-facing convenience API.
# ---------------------------------------------------------------------------

def load_into(
    ofc: Any,
    key: str,
    cache_dir: Optional[Path] = None,
    logger: Callable[[str], None] = _default_logger,
) -> bool:
    """Load a cached trained head's weights into `ofc` in place.

    Returns True on HIT (weights loaded via `ofc.load_head_state_dict`), False on
    MISS (`ofc` left completely untouched -- still its as-constructed init, e.g.
    the zeroed last Linear if train_* was False). `ofc` is expected to expose
    `head_state_dict()` / `load_head_state_dict()` (see
    `ree_core/pfc/ofc_analog.py OFCAnalog`), duck-typed rather than imported so
    this module stays usable without a hard `ree_core` dependency at import time.
    """
    blob = load_blob(key, cache_dir=cache_dir, logger=logger)
    if blob is None:
        return False
    ofc.load_head_state_dict(blob["head_state_dict"])
    return True


def store_from(
    ofc: Any,
    key: str,
    *,
    provenance: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Path] = None,
    logger: Callable[[str], None] = _default_logger,
) -> None:
    """Persist `ofc`'s current trained head weights under `key`.

    `provenance` is free-form caller metadata (e.g. training episode count, final
    loss, arm/experiment name) embedded alongside the weights purely for human
    triage on a later cache inspection -- never consulted by load_blob/load_into.
    """
    blob: Dict[str, Any] = {
        "key": key,
        "schema": OFC_HEAD_CACHE_SCHEMA,
        "head_state_dict": ofc.head_state_dict(),
        "provenance": dict(provenance) if provenance is not None else {},
    }
    store_blob(key, blob, cache_dir=cache_dir, logger=logger)


def get_or_train(
    ofc: Any,
    *,
    config_slice: Mapping[str, Any],
    seed: int,
    train_fn: Callable[[], None],
    config_slice_declared: bool = False,
    substrate_scope: Optional[Sequence[str]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Path] = None,
    logger: Callable[[str], None] = _default_logger,
) -> Dict[str, Any]:
    """The single entry point most callers want: HIT -> load, MISS -> train + store.

    `train_fn` is called with NO arguments and must train `ofc`'s heads IN PLACE
    (e.g. build an `optim.Adam(list(ofc.bias_head_parameters()) +
    list(ofc.devaluation_bias_head_parameters()), lr=...)` and step it under
    `torch.enable_grad()` / `train_mode=True`, exactly as the existing 485-lineage
    scripts already do). This module deliberately does not run or shape that loop
    itself -- see the module docstring's "WHAT THIS MODULE DOES NOT DO".

    Returns {"cache": "hit"|"miss"|"disabled", "key": <hex>} for the caller's own
    manifest/provenance recording (mirrors maturation_curriculum.
    mature_and_collect_world's return shape).
    """
    key = compute_head_key(
        config_slice=config_slice,
        seed=seed,
        config_slice_declared=config_slice_declared,
        substrate_scope=substrate_scope,
    )
    if load_into(ofc, key, cache_dir=cache_dir, logger=logger):
        logger("ofc_head_cache: HIT key=%s" % key[:12])
        return {"cache": "hit", "key": key}

    train_fn()

    cache_state = "disabled" if _cache_disabled() else "miss"
    store_from(ofc, key, provenance=provenance, cache_dir=cache_dir, logger=logger)
    logger("ofc_head_cache: %s-store key=%s" % (cache_state.upper(), key[:12]))
    return {"cache": cache_state, "key": key}


__all__ = [
    "OFC_HEAD_CACHE_SCHEMA",
    "compute_head_key",
    "load_blob",
    "store_blob",
    "load_into",
    "store_from",
    "get_or_train",
]
