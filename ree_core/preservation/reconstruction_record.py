"""ReconstructionRecord -- birth-replay preservation of a REE organism.

A ReconstructionRecord is an immutable, self-describing, integrity-checked bundle
of exactly the inputs that deterministically produce one REE life, plus the
provenance and contemporaneous understanding needed to interpret it later:

  * config    -- the full REEConfig, serialized faithfully to plain JSON;
  * seed      -- the integer seed the life was constructed under;
  * environment -- a {class, params} spec sufficient to rebuild the world;
  * provenance  -- substrate_hash, substrate_commit, machine_class,
                   architecture_epoch (fill these from experiments/_lib/
                   manifest_core on the experiment side and pass them in);
  * understanding -- a free-form dict pointer to the claims / governance state
                   and metrics we believed relevant at capture time;
  * reason_for_ending / lifetime -- optional developmental context.

Design commitments (see the plan doc for the full rationale):
  * PURE: this module imports only ree_core + stdlib. Provenance strings are
    passed IN by the caller, so ree_core never depends on the experiments layer.
  * FAITHFUL: REEConfig round-trips exactly, including tuple-typed fields that
    JSON would otherwise flatten to lists. Verified by an equality assert in
    capture() and by the contract test.
  * IMMUTABLE / APPEND-ONLY: write_record refuses to overwrite an existing
    record. load_record re-checks the integrity hash and raises on mismatch.
  * HONEST ABOUT FIDELITY: reconstruction is bit-exact for organism
    *construction* (weight init from seed is machine-robust). Multi-step *replay*
    equivalence holds only WITHIN a machine class, because torch.multinomial in
    the E3 selector diverges across machine classes -- machine_class is stamped
    on every record so a future reconstructor knows the boundary.

ASCII-only in anything that prints. Timestamps are UTC.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

# ree_core-only import (same layer; no experiments dependency).
from ree_core.utils.config import REEConfig

SCHEMA_VERSION = "reconstruction_record.v1"


class IntegrityError(Exception):
    """Raised when a record's stored integrity hash does not match its content."""


class RecordExistsError(Exception):
    """Raised when writing would overwrite an existing (immutable) record."""


# --------------------------------------------------------------------------- #
# REEConfig <-> plain-JSON (faithful, tuple-preserving)                        #
# --------------------------------------------------------------------------- #

# We serialize dataclasses and tuples with explicit, self-describing tags rather
# than relying on type annotations. Annotation-guided reconstruction is not
# sufficient for REEConfig: several fields are typed as a bare `list` (e.g.
# EventSegmenterConfig.scales, a list of nested dataclasses) or hold tuples that
# JSON cannot distinguish from lists. Tagging is annotation-independent, exact,
# and -- for a record meant to be read by a future system -- a feature: the config
# block says what each object is. Tag keys are namespaced to avoid colliding with
# real config keys.
_DC_TAG = "__dataclass__"
_TUPLE_TAG = "__tuple__"
# Records may only reconstruct dataclasses from ree_core (never import an
# arbitrary dotted path named in a file).
_ALLOWED_IMPORT_PREFIX = "ree_core."


def _import_dataclass(path: str) -> Any:
    if not path.startswith(_ALLOWED_IMPORT_PREFIX):
        raise ValueError(
            "refusing to import non-ree_core class named in a record: %r" % path
        )
    import importlib

    module_path, _, cls_name = path.rpartition(".")
    cls = getattr(importlib.import_module(module_path), cls_name)
    if not dataclasses.is_dataclass(cls):
        raise ValueError("%r is not a dataclass" % path)
    return cls


def _ser(obj: Any) -> Any:
    """Recursively serialize to JSON-native form, tagging dataclasses and tuples."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: Dict[str, Any] = {
            _DC_TAG: "%s.%s" % (type(obj).__module__, type(obj).__qualname__)
        }
        for f in dataclasses.fields(obj):
            out[f.name] = _ser(getattr(obj, f.name))
        return out
    if isinstance(obj, tuple):
        return {_TUPLE_TAG: [_ser(v) for v in obj]}
    if isinstance(obj, list):
        return [_ser(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _ser(v) for k, v in obj.items()}
    return obj


def _de(obj: Any) -> Any:
    """Inverse of _ser: rebuild tagged dataclasses and tuples exactly."""
    if isinstance(obj, dict):
        if _DC_TAG in obj:
            cls = _import_dataclass(obj[_DC_TAG])
            init_names = {f.name for f in dataclasses.fields(cls) if f.init}
            kwargs = {
                k: _de(v)
                for k, v in obj.items()
                if k != _DC_TAG and k in init_names
            }
            return cls(**kwargs)
        if _TUPLE_TAG in obj:
            return tuple(_de(v) for v in obj[_TUPLE_TAG])
        return {k: _de(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_de(v) for v in obj]
    return obj


def config_to_dict(cfg: REEConfig) -> Dict[str, Any]:
    """Serialize a REEConfig to a plain, JSON-native, self-describing dict."""
    return _ser(cfg)


def config_from_dict(data: Dict[str, Any]) -> REEConfig:
    """Reconstruct a REEConfig from a config_to_dict / JSON dict, exactly."""
    out = _de(data)
    if not isinstance(out, REEConfig):
        raise ValueError(
            "config_from_dict did not reconstruct a REEConfig (got %s); the record's "
            "config block is malformed or not a serialized REEConfig." % type(out).__name__
        )
    return out


# --------------------------------------------------------------------------- #
# The record                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class ReconstructionRecord:
    """One immutable birth-replay preservation record.

    `integrity` is a sha256 over the canonical JSON of every other field; it is
    filled by compute_integrity() and checked by verify_integrity(). Do not set
    it by hand.
    """

    record_id: str
    seed: int
    config: Dict[str, Any]
    environment: Dict[str, Any]                    # {"class": dotted.path, "params": {...}}
    provenance: Dict[str, Any]                     # substrate_hash / substrate_commit / machine_class / architecture_epoch
    understanding: Dict[str, Any] = field(default_factory=dict)
    reason_for_ending: Optional[str] = None
    lifetime: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    schema_version: str = SCHEMA_VERSION
    integrity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _canonical_bytes(payload: Dict[str, Any]) -> bytes:
    """Deterministic serialization for hashing (sorted keys, no whitespace)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_integrity(record: ReconstructionRecord) -> str:
    """Return the sha256 hex of the record with its `integrity` field excluded."""
    payload = record.to_dict()
    payload.pop("integrity", None)
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def verify_integrity(record: ReconstructionRecord) -> bool:
    """True iff the record's stored integrity hash matches its content."""
    return bool(record.integrity) and record.integrity == compute_integrity(record)


def capture(
    *,
    record_id: str,
    seed: int,
    config: REEConfig,
    environment: Dict[str, Any],
    substrate_hash: str,
    machine_class: str,
    substrate_commit: Optional[str] = None,
    architecture_epoch: Optional[str] = None,
    understanding: Optional[Dict[str, Any]] = None,
    reason_for_ending: Optional[str] = None,
    lifetime: Optional[Dict[str, Any]] = None,
    created_at: Optional[str] = None,
) -> ReconstructionRecord:
    """Build a stamped, integrity-sealed ReconstructionRecord.

    Provenance strings (substrate_hash / substrate_commit / machine_class /
    architecture_epoch) are supplied by the caller -- on the experiment side fill
    them from experiments/_lib/manifest_core so this module stays pure.

    Asserts the config round-trips exactly before sealing, so a record can never
    be written with a config it cannot faithfully reconstruct.
    """
    cfg_dict = config_to_dict(config)
    # Fail loudly at capture time if this config does not round-trip exactly.
    if config_from_dict(cfg_dict) != config:
        raise ValueError(
            "REEConfig did not round-trip exactly; refusing to seal a record whose "
            "config cannot be faithfully reconstructed. This usually means a new "
            "config field type the deserializer does not restore (see _rebuild)."
        )

    provenance = {
        "substrate_hash": substrate_hash,
        "substrate_commit": substrate_commit,
        "machine_class": machine_class,
        "architecture_epoch": architecture_epoch,
    }

    record = ReconstructionRecord(
        record_id=record_id,
        seed=int(seed),
        config=cfg_dict,
        environment=dict(environment),
        provenance=provenance,
        understanding=dict(understanding or {}),
        reason_for_ending=reason_for_ending,
        lifetime=dict(lifetime or {}),
        created_at=created_at or (datetime.utcnow().isoformat() + "Z"),
        schema_version=SCHEMA_VERSION,
        integrity=None,
    )
    record.integrity = compute_integrity(record)
    return record


def reconstruct_config(record: ReconstructionRecord) -> REEConfig:
    """Return the REEConfig this record was captured under, exactly.

    Combine with the same seeded-construction path the life was built under to
    re-derive the birth organism, e.g. (experiment side):

        from experiments._lib.arm_fingerprint import seeded_construct
        from ree_core.agent import REEAgent
        agent = seeded_construct(record.seed, lambda: REEAgent(reconstruct_config(record)))
    """
    return config_from_dict(record.config)


# --------------------------------------------------------------------------- #
# (De)serialization + immutable, integrity-checked storage                    #
# --------------------------------------------------------------------------- #

def to_json(record: ReconstructionRecord) -> str:
    return json.dumps(record.to_dict(), indent=2, sort_keys=True)


def from_json(text: str) -> ReconstructionRecord:
    data = json.loads(text)
    return ReconstructionRecord(**data)


def write_record(record: ReconstructionRecord, archive_dir: str) -> str:
    """Write the record immutably to <archive_dir>/<record_id>/reconstruction_record.json.

    Append-only: refuses to overwrite an existing record (RecordExistsError).
    Verifies integrity before writing. Returns the written path.
    """
    if not verify_integrity(record):
        raise IntegrityError(
            "refusing to write a record whose integrity hash is missing or stale; "
            "build it via capture() (which seals it) rather than by hand."
        )
    unit_dir = os.path.join(archive_dir, record.record_id)
    path = os.path.join(unit_dir, "reconstruction_record.json")
    if os.path.exists(path):
        raise RecordExistsError(
            "a reconstruction record already exists at %s; records are immutable "
            "and append-only -- never overwrite one." % path
        )
    os.makedirs(unit_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(to_json(record))
    os.replace(tmp, path)  # atomic within the directory
    return path


def load_record(path: str) -> ReconstructionRecord:
    """Load a record and re-verify its integrity hash (raises on mismatch)."""
    with open(path, "r", encoding="utf-8") as fh:
        record = from_json(fh.read())
    if not verify_integrity(record):
        raise IntegrityError(
            "integrity check failed for %s: the record's content does not match "
            "its stored hash (corruption or tampering)." % path
        )
    return record
