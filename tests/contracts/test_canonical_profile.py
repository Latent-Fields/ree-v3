"""Contract tests for the canonical-profile mechanism (Option B+C).

Design brief: REE_assembly/evidence/planning/architecture_epoch_investigation.md
(2026-08-12), sections 7-8. This mechanism is infrastructure only -- it ships with
exactly one profile (v0, zero overrides); admitting real members is a separate,
future governance decision (section 9) and is explicitly NOT what these tests
exercise or validate.

Proves the three properties the task that built this mechanism required:

  (a) REEConfig() bare defaults are bit-identical whether or not the canonical-
      profile machinery has ever been used -- the profile layer changes nothing
      about REEConfig's class-level defaults.
  (b) A profile-constructed config differs from bare defaults in EXACTLY the
      fields the profile declared -- nothing more, nothing less.
  (c) A profile's fingerprint is stable across repeated freezes, persists and
      reloads correctly, and re-persisting a version under DIFFERENT content is
      refused rather than silently overwriting a frozen artifact.

ASCII-only (repo rule). Run: pytest tests/contracts/test_canonical_profile.py -q
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from experiments._lib import canonical_profile_fingerprint as cpf
from ree_core.utils import canonical_profile as cp
from ree_core.utils.config import REEConfig


# ---------------------------------------------------------------------------
# Registry / placeholder
# ---------------------------------------------------------------------------


def test_placeholder_profile_is_registered_empty():
    """The only profile shipped with this mechanism is v0, with zero overrides
    -- the task that built this module explicitly deferred admitting real
    members to a separate governance pass."""
    placeholder = cp.get_profile("ree_v3_baseline", "v0")
    assert placeholder is cp.PLACEHOLDER_PROFILE
    assert placeholder.overrides == {}
    assert placeholder.qualified_name == "ree_v3_baseline@v0"
    assert placeholder in cp.list_profiles()


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        cp.get_profile("does_not_exist", "v0")


# ---------------------------------------------------------------------------
# (a) Bare REEConfig() defaults are unaffected by the profile mechanism.
# ---------------------------------------------------------------------------


def test_bare_defaults_unaffected_by_profile_mechanism():
    """REEConfig() is bit-identical to itself whether or not the
    canonical-profile machinery has been exercised -- building (and discarding)
    a profile config must never mutate REEConfig's class-level defaults (e.g.
    by editing a shared mutable default_factory instance in place instead of a
    fresh deep copy)."""
    before = dataclasses.asdict(REEConfig())
    cp.build_config(cp.PLACEHOLDER_PROFILE)
    _synthetic_nonempty_config()  # also exercises a non-empty override profile
    after = dataclasses.asdict(REEConfig())
    assert before == after


def test_v0_profile_config_equals_bare_defaults():
    """v0's overrides are empty, so its built config must be indistinguishable
    from a bare REEConfig() -- the "layers cleanly onto unchanged defaults"
    property, made concrete for the one profile that exists today."""
    built = cp.build_config(cp.PLACEHOLDER_PROFILE)
    assert dataclasses.asdict(built) == dataclasses.asdict(REEConfig())
    assert cp.diff_from_bare_defaults(built) == {}


# ---------------------------------------------------------------------------
# (b) A profile-constructed config differs only in its declared overrides.
# ---------------------------------------------------------------------------


def _synthetic_profile() -> cp.CanonicalProfileSpec:
    """A non-empty profile constructed LOCALLY for this test only -- never
    registered in the real `_PROFILES` registry, so running this test cannot
    be mistaken for admitting a real mechanism member (the task explicitly
    forbids that). Exercises both a top-level scalar field and a dotted path
    into a nested sub-config, deliberately choosing fields outside
    ree_core/predictors/e3_selector.py's territory."""
    return cp.CanonicalProfileSpec(
        name="__test_only_profile__",
        version="v0",
        description="synthetic, test-local only -- never registered",
        overrides={
            "shy_enabled": True,       # top-level scalar bool field
            "device": "cuda",          # top-level scalar str field
            "latent.harm_dim": 8,      # nested dotted path (LatentStackConfig)
        },
    )


def _synthetic_nonempty_config() -> REEConfig:
    return cp.build_config(_synthetic_profile())


def test_profile_differs_only_in_declared_overrides():
    profile = _synthetic_profile()
    built = cp.build_config(profile)

    # Every declared override actually landed.
    assert built.shy_enabled is True
    assert built.device == "cuda"
    assert built.latent.harm_dim == 8

    # The diff against bare defaults reproduces the override dict EXACTLY --
    # nothing else moved.
    diff = cp.diff_from_bare_defaults(built)
    assert diff == dict(profile.overrides)


def test_profile_config_is_independent_of_base_and_stays_pure():
    """build_config must not mutate a caller-supplied `base`, and two configs
    built from the same profile must not share nested-object identity (else
    mutating one's sub-config would leak into the other)."""
    base = REEConfig()
    base_before = dataclasses.asdict(base)
    profile = _synthetic_profile()

    built = cp.build_config(profile, base=base)
    assert dataclasses.asdict(base) == base_before  # base untouched
    assert built is not base
    assert built.latent is not base.latent

    built_again = cp.build_config(profile, base=base)
    assert built_again.latent is not built.latent
    built.latent.harm_dim = 999
    assert built_again.latent.harm_dim == 8  # no shared nested object


def test_build_config_raises_on_unknown_dotted_path():
    """A stale or mistyped override must fail loudly, not silently apply
    nothing -- a canonical profile's whole purpose is being trustworthy."""
    bogus = cp.CanonicalProfileSpec(
        name="__test_only_bogus__",
        version="v0",
        description="synthetic, test-local only",
        overrides={"this_field_does_not_exist_anywhere": True},
    )
    with pytest.raises(AttributeError):
        cp.build_config(bogus)


# ---------------------------------------------------------------------------
# (c) Fingerprint stability + persistence.
# ---------------------------------------------------------------------------


def test_freeze_is_stable_across_repeated_calls():
    profile = _synthetic_profile().as_dict()
    freeze_1 = cpf.freeze_profile(profile)
    freeze_2 = cpf.freeze_profile(profile)
    assert freeze_1["canonical_profile_hash"] == freeze_2["canonical_profile_hash"]
    assert freeze_1["substrate_hash"] == freeze_2["substrate_hash"]


def test_freeze_hash_changes_with_overrides():
    p1 = _synthetic_profile().as_dict()
    p2 = cp.CanonicalProfileSpec(
        name="__test_only_profile__",
        version="v0",
        description="synthetic, test-local only",
        overrides={**p1["overrides"], "shy_enabled": False},
    ).as_dict()
    f1 = cpf.freeze_profile(p1)
    f2 = cpf.freeze_profile(p2)
    assert f1["canonical_profile_hash"] != f2["canonical_profile_hash"]
    # Same substrate -- only the declared overrides differ.
    assert f1["substrate_hash"] == f2["substrate_hash"]


def test_persist_and_reload_roundtrip(tmp_path):
    profile = _synthetic_profile().as_dict()
    freeze = cpf.freeze_profile(profile)
    out_dir = tmp_path / "canonical_profiles"

    path = cpf.persist_profile_freeze(
        freeze, out_dir=out_dir, frozen_at_utc="2026-08-12T00:00:00Z"
    )
    assert path == out_dir / "__test_only_profile__.json"
    assert path.exists()

    with open(path) as fh:
        on_disk = json.load(fh)
    assert on_disk["name"] == "__test_only_profile__"
    recorded = on_disk["versions"]["v0"]
    assert recorded["canonical_profile_hash"] == freeze["canonical_profile_hash"]
    assert recorded["overrides"] == freeze["overrides"]
    assert recorded["frozen_at_utc"] == "2026-08-12T00:00:00Z"

    reloaded = cpf.load_profile_freeze(out_dir, "__test_only_profile__", "v0")
    assert reloaded == recorded


def test_persist_is_idempotent_for_identical_content(tmp_path):
    profile = _synthetic_profile().as_dict()
    freeze = cpf.freeze_profile(profile)
    out_dir = tmp_path / "canonical_profiles"

    cpf.persist_profile_freeze(freeze, out_dir=out_dir, frozen_at_utc="2026-08-12T00:00:00Z")
    # Re-persisting the SAME content later must not change the recorded
    # frozen_at_utc -- freezing is a once-only stamp, not a re-check clock.
    path = cpf.persist_profile_freeze(
        freeze, out_dir=out_dir, frozen_at_utc="2026-08-13T00:00:00Z"
    )
    with open(path) as fh:
        on_disk = json.load(fh)
    assert on_disk["versions"]["v0"]["frozen_at_utc"] == "2026-08-12T00:00:00Z"


def test_persist_refuses_to_overwrite_changed_frozen_version(tmp_path):
    """A frozen version's content must never silently change -- that is the
    entire point of freezing it."""
    p1 = _synthetic_profile().as_dict()
    freeze_1 = cpf.freeze_profile(p1)
    out_dir = tmp_path / "canonical_profiles"
    cpf.persist_profile_freeze(freeze_1, out_dir=out_dir, frozen_at_utc="2026-08-12T00:00:00Z")

    p2 = dict(p1)
    p2["overrides"] = {**p1["overrides"], "shy_enabled": False}
    freeze_2 = cpf.freeze_profile(p2)  # same name+version, different content
    assert freeze_2["version"] == freeze_1["version"]
    assert freeze_2["canonical_profile_hash"] != freeze_1["canonical_profile_hash"]

    with pytest.raises(ValueError):
        cpf.persist_profile_freeze(
            freeze_2, out_dir=out_dir, frozen_at_utc="2026-08-13T00:00:00Z"
        )

    # force=True is an explicit, deliberate override.
    path = cpf.persist_profile_freeze(
        freeze_2, out_dir=out_dir, frozen_at_utc="2026-08-13T00:00:00Z", force=True
    )
    with open(path) as fh:
        on_disk = json.load(fh)
    assert on_disk["versions"]["v0"]["canonical_profile_hash"] == freeze_2["canonical_profile_hash"]


# ---------------------------------------------------------------------------
# manifest_core wiring
# ---------------------------------------------------------------------------


def test_stamp_recording_core_records_canonical_profile_fields():
    from experiments._lib import manifest_core as mc

    manifest: dict = {}
    mc.stamp_recording_core(
        manifest,
        canonical_profile="ree_v3_baseline@v0",
        canonical_profile_hash="deadbeef",
    )
    assert manifest["canonical_profile"] == "ree_v3_baseline@v0"
    assert manifest["canonical_profile_hash"] == "deadbeef"
    # Not core -- absent entirely when the caller doesn't supply it.
    manifest2: dict = {}
    mc.stamp_recording_core(manifest2)
    assert "canonical_profile" not in manifest2
    assert "canonical_profile_hash" not in manifest2
    assert "canonical_profile" not in mc.ALWAYS_CORE_KEYS
    assert "canonical_profile_hash" not in mc.ALWAYS_CORE_KEYS
