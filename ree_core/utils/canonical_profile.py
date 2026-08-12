"""
Canonical organism profile mechanism (Option B of
REE_assembly/evidence/planning/architecture_epoch_investigation.md section 7-8).

WHAT THIS IS. A versioned, named, human-curated bundle of REEConfig field
overrides, layered ON TOP OF the unchanged, backward-compatible bare
`REEConfig()` defaults. It answers "which curated organism, if any, did this
run use" -- a question `architecture_epoch` was never designed to answer (see
the investigation doc section 1: that field is a coarse generation-family tag,
copy-pasted once per script, carrying zero config information) and that today
has four different, non-interchangeable answers depending which of (a) bare
defaults, (b) the epoch-tagged corpus, (c) one script's hand-assembled bundle,
or (d) the unwritten "on by convention" set you pick (section 6).

WHAT THIS IS NOT (yet). This module does NOT decide which flags belong in a
canonical profile -- that is a governance decision gated on admission criteria
(investigation doc section 9) and, for anything E3-adjacent, on the in-flight
F-dominance investigation (section 15). `_PROFILES` below deliberately holds
exactly one profile with EMPTY overrides: the mechanism, not a populated
organism. Populating it is future work, not this module's job.

`REEConfig()` bare defaults are UNCHANGED by this module's existence -- no
field's coded default is touched, and importing this module has zero effect on
`REEConfig()` instances built without going through it. `build_config` always
starts from a fresh (or caller-supplied) `REEConfig()` and applies overrides on
top; it never mutates the class-level defaults.

See also: `experiments/_lib/canonical_profile_fingerprint.py`, which freezes
and persists a declared profile's content hash using the existing
`arm_fingerprint.compute_substrate_hash` machinery -- that module is
stdlib-only (mirrors arm_fingerprint's own import guarantee) and therefore
cannot import this one; the two are joined only by the plain-data
`CanonicalProfileSpec.overrides` dict, which is JSON-serialisable.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict, List, Mapping, Optional

from ree_core.utils.config import REEConfig

CANONICAL_PROFILE_SCHEMA = "canonical_profile/v1"


@dataclasses.dataclass(frozen=True)
class CanonicalProfileSpec:
    """One declared, human-curated canonical profile version.

    `overrides` is a flat mapping of dotted field paths (e.g.
    "shy_enabled" or "e3.some_nested_field") to the value that field takes
    under this profile -- the SAME dotted-path convention
    `manifest_core.enabled_default_off_flags` already uses when reading a
    live config back out, so a profile's declared overrides and a run's
    recorded enabled-flags are directly comparable without a translation
    layer.

    Deliberately a plain, JSON-round-trippable shape (str/str/str/dict) so a
    frozen artifact (see canonical_profile_fingerprint.py) can be written
    and re-read without importing this module or `ree_core` at all.
    """

    name: str
    version: str
    description: str
    overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": CANONICAL_PROFILE_SCHEMA,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "overrides": dict(self.overrides),
            "notes": self.notes,
        }

    @property
    def qualified_name(self) -> str:
        """"<name>@<version>" -- what a manifest's `canonical_profile` field records."""
        return f"{self.name}@{self.version}"


# ---------------------------------------------------------------------------
# Registry -- exactly one profile today: the mechanism's own placeholder.
# ---------------------------------------------------------------------------
#
# EMPTY overrides is not a stand-in for "not yet written" -- it is the correct,
# intentional content of v0. Admitting real members is a separate governance
# decision (investigation doc section 9), explicitly deferred by the task that
# built this module. A future admission pass adds a NEW CanonicalProfileSpec
# (e.g. name="ree_v3_baseline", version="v1") rather than mutating v0 -- a
# frozen version's overrides must never change in place (see
# canonical_profile_fingerprint.persist_profile_freeze's immutability check).

_PROFILES: Dict[str, CanonicalProfileSpec] = {}


def _register(spec: CanonicalProfileSpec) -> CanonicalProfileSpec:
    _PROFILES[spec.qualified_name] = spec
    return spec


PLACEHOLDER_PROFILE = _register(
    CanonicalProfileSpec(
        name="ree_v3_baseline",
        version="v0",
        description=(
            "Placeholder canonical profile -- the mechanism only, zero admitted "
            "overrides. Layers cleanly onto REEConfig() bare defaults (identical "
            "config) until a governance pass admits real members per "
            "architecture_epoch_investigation.md section 9."
        ),
        overrides={},
        notes=(
            "Do not add overrides here without running the admission criteria "
            "in the investigation doc (corpus enablement, cited-evidence-run "
            "check, non-degeneracy, known-interaction check) and, for anything "
            "E3-adjacent, waiting on the F-dominance investigation (section 15)."
        ),
    )
)


def get_profile(name: str, version: str) -> CanonicalProfileSpec:
    key = f"{name}@{version}"
    try:
        return _PROFILES[key]
    except KeyError:
        raise KeyError(
            f"no canonical profile registered as {key!r}; known: "
            f"{sorted(_PROFILES)}"
        ) from None


def list_profiles() -> List[CanonicalProfileSpec]:
    return list(_PROFILES.values())


# ---------------------------------------------------------------------------
# Dotted-path get/set over a (possibly nested) REEConfig instance.
# ---------------------------------------------------------------------------


def _get_dotted(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


def _set_dotted(obj: Any, dotted: str, value: Any) -> None:
    """Set a dotted attribute path, raising AttributeError on an unknown final
    field name.

    Intermediate segments already fail loudly via `getattr` (a bad prefix like
    "nope.foo" raises on `nope`). The final segment needs an explicit check:
    `setattr` on a plain (non-slotted) dataclass instance does NOT raise for an
    unrecognised attribute name -- it silently creates a new instance
    attribute, which would make a mistyped or stale override in a profile
    apply silently as a no-op instead of failing the way this mechanism
    requires (a canonical profile that can misreport what it changed is worse
    than one that crashes on load).
    """
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    last = parts[-1]
    if not hasattr(cur, last):
        raise AttributeError(
            f"canonical profile override path {dotted!r} does not resolve: "
            f"{type(cur).__name__!r} has no field {last!r}"
        )
    setattr(cur, last, value)


def build_config(
    profile: CanonicalProfileSpec, base: Optional[REEConfig] = None
) -> REEConfig:
    """A REEConfig built from `base` (default: a fresh `REEConfig()`) with
    `profile.overrides` applied on top.

    `base` is deep-copied before any override is applied, so neither the
    caller's `base` instance nor the `REEConfig` class-level defaults are ever
    mutated -- calling this repeatedly, or not at all, has zero effect on what
    a bare `REEConfig()` looks like elsewhere in the process.

    Raises AttributeError (not a silent no-op) if a declared dotted path does
    not resolve against REEConfig's actual fields -- a stale or mistyped
    override in a profile must fail loudly, not silently apply nothing.
    """
    cfg = copy.deepcopy(base) if base is not None else REEConfig()
    for dotted, value in profile.overrides.items():
        _set_dotted(cfg, dotted, value)
    return cfg


def diff_from_bare_defaults(
    config: Any, stock: Optional[Any] = None, _prefix: str = ""
) -> Dict[str, Any]:
    """{dotted_field_name: value} for every field of `config` that differs from
    `stock` (default: a fresh instance of `type(config)`), regardless of
    whether the stock value is falsy.

    Generalises `manifest_core.enabled_default_off_flags` (which only reports
    a field that moved FALSE/0 -> something else) to ANY change, which is what
    a profile-vs-bare-defaults comparison needs -- a profile could just as
    well flip a field OFF or change a numeric default. Recurses into nested
    dataclass fields with the same dotted-path convention as
    `enabled_default_off_flags` and `CanonicalProfileSpec.overrides`, so the
    two are directly comparable (see test_canonical_profile.py).

    Non-dataclass input returns {} rather than raising -- best-effort, mirrors
    `enabled_default_off_flags`'s own posture.
    """
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        return {}
    stock = stock if stock is not None else type(config)()
    out: Dict[str, Any] = {}
    for f in dataclasses.fields(config):
        try:
            val = getattr(config, f.name)
            stock_val = getattr(stock, f.name)
        except AttributeError:
            continue
        dotted = f"{_prefix}{f.name}"
        if dataclasses.is_dataclass(val):
            out.update(diff_from_bare_defaults(val, stock_val, dotted + "."))
            continue
        if val != stock_val:
            out[dotted] = val
    return out


__all__ = [
    "CANONICAL_PROFILE_SCHEMA",
    "CanonicalProfileSpec",
    "PLACEHOLDER_PROFILE",
    "get_profile",
    "list_profiles",
    "build_config",
    "diff_from_bare_defaults",
]
