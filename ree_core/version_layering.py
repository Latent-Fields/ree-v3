"""Version-layering registry: the single anchor for higher-version (V4/V5) flags.

V3 closure is primary. Any V4/V5 substrate work is PREPARATORY only and must not
change V3 *default* execution behaviour. The enforcement convention (see
REE_assembly/docs/architecture/version_layering_doctrine.md) is:

  - A V4/V5 substrate change to shared code (ree_core/) MUST be flag-gated and
    no-op / bit-identical by default.
  - A V4/V5 call-site into a shared path MUST be conditional on the feature flag,
    never unconditional (the 2026-06-17 V3-EXQ-654e DR-12 incident: an
    unconditional ``e2_forward_pe_per_candidate=`` kwarg crashed the default V3
    path against a skewed/older ``e3_selector.select()`` that lacked the param).

This module enumerates every generation-tagged master flag so the guards
(`tests/v3_parity_smoke.py`, `tests/contracts/test_version_layering_noop_default.py`,
and the runner preflight) can mechanically assert each one defaults OFF.

WHEN YOU ADD A V4/V5 FLAG: append a `GenerationFlag` entry here in the SAME pass
that adds the flag. That is what keeps the no-op-default guard exhaustive.
"""

from __future__ import annotations

from typing import Any, List, NamedTuple

# ASCII-only in any printed output (Windows cp1252 terminals).


class GenerationFlag(NamedTuple):
    """A higher-version master gate on a shared (ree_core/) code path.

    generation:  "v4" | "v5" -- the generation this flag belongs to.
    config_path: dotted attribute path under a built REEConfig, e.g.
                 "e3.use_pe_confidence_weighting".
    default_off: the bit-identical / no-op value (almost always False). The flag's
                 DEFAULT must equal this for V3 default behaviour to be preserved.
    claim:       the owning claim id (e.g. "DR-12") for traceability.
    note:        one-line human description.
    """

    generation: str
    config_path: str
    default_off: Any
    claim: str
    note: str


# ---------------------------------------------------------------------------
# Registry. Append (never silently retag) as V4/V5 substrate flags land.
# ---------------------------------------------------------------------------
GENERATION_FLAGS: List[GenerationFlag] = [
    GenerationFlag(
        generation="v4",
        config_path="e3.use_pe_confidence_weighting",
        default_off=False,
        claim="DR-12",
        note="self_model_v4:SELF-4 E2 forward-PE -> E3 confidence down-weight; "
        "first V4 substrate (2026-06-17). Gates the e2_forward_pe_per_candidate "
        "penalty in e3_selector.score_trajectory AND the conditional call-site in "
        "agent.select_action.",
    ),
]


def _read_path(config: Any, dotted: str) -> Any:
    """Walk a dotted attribute path under a config object.

    Raises AttributeError if any segment is missing -- a missing path is itself a
    drift signal the guard should surface, not silently pass.
    """
    obj = config
    for seg in dotted.split("."):
        obj = getattr(obj, seg)
    return obj


def iter_generation_flags(*generations: str) -> List[GenerationFlag]:
    """Return registered flags, optionally filtered to the given generation(s)."""
    if not generations:
        return list(GENERATION_FLAGS)
    wanted = set(generations)
    return [f for f in GENERATION_FLAGS if f.generation in wanted]


def find_flags_on(config: Any, *generations: str) -> List[str]:
    """Return dotted paths of any registered V4/V5 flag NOT at its no-op default.

    An empty list means the config preserves V3 default behaviour for every
    registered higher-version flag. Used by the V3-parity smoke and the
    no-op-default contract test.
    """
    on: List[str] = []
    for flag in iter_generation_flags(*generations):
        try:
            value = _read_path(config, flag.config_path)
        except AttributeError:
            # Path missing -> treat as drift the caller must see.
            on.append(flag.config_path + " (MISSING)")
            continue
        if value != flag.default_off:
            on.append(flag.config_path)
    return on


def assert_all_off(config: Any, *generations: str) -> None:
    """Raise AssertionError if any registered V4/V5 flag is not at its no-op default.

    The error message names the offending paths so a failing preflight / contract
    points straight at the flag that breaks V3-default safety.
    """
    on = find_flags_on(config, *generations)
    assert not on, (
        "version-layering violation: higher-version (V4/V5) flag(s) not at "
        "no-op default on this config: " + ", ".join(on)
    )
