"""MECH-307 reachability-through-`REEConfig.from_dims()` contract.

A `REEConfig` knob needs THREE sites, not one:

  1. the dataclass field on `REEConfig`
  2. a named parameter in the `from_dims()` signature
  3. a post-`cls()` re-apply of any `__post_init__` resolver, inside `from_dims`

Site 3 is independently required because `from_dims` assigns its fields AFTER
`cls()`, so `__post_init__` has already run against the defaults by the time the
master flag is set -- without the re-apply the sub-flags are never forced True.

Until 2026-08-07 MECH-307 had only site 1. All twelve of its parameters fell
into `from_dims`'s `**kwargs` and were silently dropped, so
`from_dims(use_mech307_conjunction=True)` returned a config with all four gaps
OFF and raised nothing. 84 experiment drivers requested MECH-307 that way and
ran with it entirely off.

The pre-existing MECH-307 contracts (`test_mech307_conjunction_contract.py`,
`test_mech307_consumer_conjunction.py`) could not catch this: they build a
flagless config and `setattr` each SUB-flag afterwards. Sub-flag setattr works,
so those contracts genuinely verify the four gaps -- the untested route was the
master-flag-through-the-factory one, which is exactly how every driver asks.

So these tests deliberately assert the FACTORY route (site 2 + site 3), each
against a direct-construction parity case, so sites 2 and 3 cannot regress
independently: dropping the signature entry fails the reachability tests, and
dropping the re-apply fails the resolver tests while the reachability tests for
the master flag itself would still pass.

Write-up: REE_assembly/evidence/planning/mech307_from_dims_unreachable_2026-08-07.md
General failure mode: [memory] reference-reeconfig-from-dims-silent-kwargs
"""
from __future__ import annotations

import pytest

from ree_core.utils.config import REEConfig


# The three substrate-side sub-flags the master flag resolves True (OR-only).
# The consumer-side flag is deliberately NOT auto-set -- it is a downstream
# wiring decision, not a substrate change.
MECH307_SUB_FLAGS = (
    "use_mech307_split_surprise",
    "use_mech307_schema_multichannel",
    "use_mech307_predicted_location_write",
)

# Every MECH-307 parameter that must be reachable through the factory.
MECH307_PARAMS = (
    "use_mech307_conjunction",
    "use_mech307_split_surprise",
    "use_mech307_schema_multichannel",
    "use_mech307_predicted_location_write",
    "use_mech307_signed_pe",
    "use_mech307_consumer_conjunction_read",
    "mech307_anticipatory_liking_gain",
    "mech307_z_beta_schema_gain",
    "mech307_conjunction_gain",
    "mech307_conjunction_wanting_threshold",
    "mech307_conjunction_liking_threshold",
    "mech307_conjunction_z_beta_threshold",
)

DIMS = dict(
    body_obs_dim=8,
    world_obs_dim=32,
    action_dim=5,
    self_dim=32,
    world_dim=32,
)


def _from_dims(**flags) -> REEConfig:
    return REEConfig.from_dims(**DIMS, **flags)


# --------------------------------------------------------------------------
# Site 3: the master flag lights the sub-flags through the FACTORY
# --------------------------------------------------------------------------

def test_master_flag_through_from_dims_lights_all_three_sub_flags():
    """The defect, stated as an assertion.

    This is the route 84 drivers use. Before 2026-08-07 every one of these
    was False.
    """
    config = _from_dims(use_mech307_conjunction=True)
    assert config.use_mech307_conjunction is True
    for flag in MECH307_SUB_FLAGS:
        assert getattr(config, flag) is True, (
            f"{flag} not resolved True by the master flag through from_dims() "
            "-- the post-cls() resolver re-apply is missing (site 3)"
        )


def test_master_flag_direct_construction_parity():
    """Direct construction and the factory must agree on all twelve.

    The parity case is what makes site 2 and site 3 non-independent: a config
    built either way is the same config.
    """
    direct = REEConfig(use_mech307_conjunction=True)
    factory = _from_dims(use_mech307_conjunction=True)
    for param in MECH307_PARAMS:
        assert getattr(factory, param) == getattr(direct, param), (
            f"{param} differs between REEConfig(...) and REEConfig.from_dims(...)"
        )


def test_defaults_parity_bit_identical_off():
    """With no MECH-307 argument, the factory matches a default config.

    Guards the no-op default: adding the twelve to the signature must not move
    any default, so drivers that never mention MECH-307 are unaffected.
    """
    direct = REEConfig()
    factory = _from_dims()
    for param in MECH307_PARAMS:
        assert getattr(factory, param) == getattr(direct, param), (
            f"{param} default changed on the from_dims() path"
        )
    for flag in MECH307_SUB_FLAGS:
        assert getattr(factory, flag) is False


# --------------------------------------------------------------------------
# Site 2: every parameter is individually reachable through the factory
# --------------------------------------------------------------------------

@pytest.mark.parametrize("param", MECH307_PARAMS)
def test_each_mech307_param_is_reachable_through_from_dims(param):
    """A non-default value passed to from_dims() must actually land.

    Parametrised over all twelve so a future partial wiring (say, the six bool
    flags added and the six numeric gains forgotten) fails loudly rather than
    being swallowed by **kwargs.
    """
    default = getattr(REEConfig(), param)
    if isinstance(default, bool):
        value = not default
    else:
        value = float(default) + 0.125

    config = _from_dims(**{param: value})
    assert getattr(config, param) == value, (
        f"{param} was dropped by from_dims() -- it is absent from the "
        "signature and fell into **kwargs (site 2)"
    )


# --------------------------------------------------------------------------
# Resolver semantics: OR-only, no clobber
# --------------------------------------------------------------------------

def test_sub_flag_alone_through_from_dims_does_not_set_the_master():
    """A sub-flag set without the master keeps working, and stays alone.

    The resolver only flips False -> True under the master; it must not infer
    the master from a sub-flag, and must not touch the sibling sub-flags.
    """
    config = _from_dims(use_mech307_split_surprise=True)
    assert config.use_mech307_split_surprise is True
    assert config.use_mech307_conjunction is False
    assert config.use_mech307_schema_multichannel is False
    assert config.use_mech307_predicted_location_write is False


def test_master_flag_overrides_explicit_sub_flag_false():
    """OR-only means the master wins over an explicit False on a sub-flag.

    This mirrors the documented dataclass behaviour ("Setting this master flag
    overrides any explicit False on the three sub-flags above") and is the
    reason the re-apply must come after the sub-flag assignments in the
    from_dims body, not before.
    """
    config = _from_dims(
        use_mech307_conjunction=True,
        use_mech307_split_surprise=False,
        use_mech307_schema_multichannel=False,
        use_mech307_predicted_location_write=False,
    )
    for flag in MECH307_SUB_FLAGS:
        assert getattr(config, flag) is True


def test_master_flag_does_not_set_the_consumer_side_read():
    """The consumer-side flag is explicitly NOT part of the master bundle."""
    config = _from_dims(use_mech307_conjunction=True)
    assert config.use_mech307_consumer_conjunction_read is False


# --------------------------------------------------------------------------
# Interaction with the goal-stream bundle preset (unchanged precedence)
# --------------------------------------------------------------------------

def test_goal_stream_bundle_precedence_is_unchanged():
    """enable_goal_stream() still wins over the new from_dims assignments.

    The MECH-307 block is assigned BEFORE the goal_stream_enabled block in
    from_dims precisely so this preset keeps the precedence it has always had
    (same ordering the MECH-295 block relies on). If the block were moved after
    the preset, `goal_stream_enabled=True` would start returning the bare
    parameter defaults for the gains and the consumer read -- a silent
    behaviour change for every goal-stream driver.
    """
    config = _from_dims(goal_stream_enabled=True)
    for flag in MECH307_SUB_FLAGS:
        assert getattr(config, flag) is True
    assert config.use_mech307_consumer_conjunction_read is True
    assert config.mech307_conjunction_gain == 1.0
    assert config.mech307_conjunction_wanting_threshold == pytest.approx(0.10)
    assert config.mech307_conjunction_liking_threshold == pytest.approx(0.05)
    assert config.mech307_conjunction_z_beta_threshold == pytest.approx(0.10)
