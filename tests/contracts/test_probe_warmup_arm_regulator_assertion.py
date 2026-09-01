"""Contract: probe_warmup asserts, after warmup, that an agent's arm-conditional
regulators still match the arm flags the caller declared (SD-PROBE-WARMUP,
failure_autopsy_V3-EXQ-963_2026-08-30 recommendation 4 of 4).

WHAT THIS PINS, AND WHY IT IS A SEPARATE LAYER FROM
test_probe_warmup_cache_key_restore.py.

That file pins the two PREVENTIVE halves of the SD-PROBE-WARMUP repair, landed as
ree-v3 d614a9c: (a) `_warmup_key` folds the caller's arm flags into the cache key,
so differently-configured arms stop sharing one blob; (b) `_restore_cached_surface`
refuses to write a cached attribute whose type disagrees with the live HIT agent's
own value, which is the primary defence and holds even for a caller that never
passes `arm_key`.

This file pins the third, DETECTION half the V3-EXQ-963 autopsy asked for in as
many words -- "Add an assertion that every arm-conditional regulator is non-None
after restore when its flag is set". It is not redundant with (a)/(b), because
both of those are keyed on assumptions a future caller can fall outside of:

  * (a) only separates arms for a caller that actually PASSES `arm_key`. Every
    driver in the tree at the time of writing -- including V3-EXQ-963's own --
    does not, and therefore still shares one blob across its whole grid.
  * (b) protects only attributes ALREADY LIVE on the HIT agent.
    `_restore_cached_surface` Case 1 restores an attribute the HIT instance never
    set verbatim and with no type check, deliberately, so a regulator created
    lazily rather than declared in `__init__` sits outside the guard's cover in
    the install direction.

The assertion reads the agent warmup actually produced and asks whether it is
still the arm the caller says it is, so it fails loudly on any route to the
corruption, including routes that do not exist yet. The failure it exists to stop
is a run that COMPLETES and writes a claim-tagged manifest that looks valid while
the manipulation it claims to test was silently absent -- V3-EXQ-963 recorded
noise_floor_temp_lift_mean 0.0 on all 20 cells, including all 10 with
use_noise_floor=True, and was only caught by comparing a predecessor's per-cell
instrument reading.

BIT-IDENTICAL FOR EXISTING CALLERS is itself pinned below: with `arm_key` absent
(what every pre-fix driver passes) there are no declared flags, nothing is
checked, and nothing raises.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from experiments._lib.probe_warmup import (
    ArmRegulatorMismatch,
    assert_arm_regulators_live,
    warm_agent,
)


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #

def _mk_agent(use_noise_floor: bool = False, use_phasic_burst: bool = False):
    """A real, minimal REEAgent with the two V3-EXQ-963 arm axes toggled.

    Same shape as test_probe_warmup_cache_key_restore.py::_mk_agent -- a real
    agent, not a stub, because the property under test is precisely that
    REEAgent.__init__ declares these regulators Optional-and-always-present and
    constructs them only under their flag.
    """
    from ree_core.agent import REEAgent
    from ree_core.environment.causal_grid_world import CausalGridWorldV2
    from ree_core.utils.config import REEConfig

    env = CausalGridWorldV2(seed=11, size=5, num_hazards=1, num_resources=1)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    cfg.use_noise_floor = use_noise_floor
    cfg.use_phasic_burst = use_phasic_burst
    torch.manual_seed(0)
    return REEAgent(cfg)


def _noop_logger(_msg: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# the V3-EXQ-963 direction: flag SET, regulator silently gone                  #
# --------------------------------------------------------------------------- #

def test_assertion_fires_on_the_v3_exq_963_signature():
    """use_noise_floor=True but agent.noise_floor is None -- the exact state a
    restore-clobber leaves behind, and the state 20 of 20 V3-EXQ-963 cells ran in.
    """
    agent = _mk_agent(use_noise_floor=True)
    assert agent.noise_floor is not None, "sanity: the flag really built one"

    # Reproduce the clobber the fixed restore now refuses, to prove the assertion
    # catches the END STATE regardless of which stage produced it.
    object.__setattr__(agent, "noise_floor", None)

    with pytest.raises(ArmRegulatorMismatch) as exc:
        assert_arm_regulators_live(
            agent, {"use_noise_floor": True}, logger=_noop_logger
        )
    assert "noise_floor" in str(exc.value)


def test_assertion_message_names_the_attribute_and_the_shape():
    agent = _mk_agent(use_noise_floor=True)
    object.__setattr__(agent, "noise_floor", None)
    with pytest.raises(ArmRegulatorMismatch) as exc:
        assert_arm_regulators_live(
            agent, {"use_noise_floor": True}, label="T1P0 seed=3",
            logger=_noop_logger,
        )
    msg = str(exc.value)
    assert "noise_floor" in msg
    assert "T1P0 seed=3" in msg, "the failing arm must be identifiable from the message"
    assert "V3-EXQ-963" in msg, "the message must point at the known defect shape"


# --------------------------------------------------------------------------- #
# the inverse direction: flag CLEAR, regulator silently installed              #
# --------------------------------------------------------------------------- #

def test_assertion_fires_when_an_unrequested_regulator_is_present():
    """The OFF arm of an ON/OFF contrast silently carrying the regulator is the
    same corruption pointing the other way -- it destroys the contrast rather
    than the manipulation, and is the direction _restore_cached_surface Case 1
    (attribute never set on this instance -> restored verbatim) does not cover.
    """
    off_agent = _mk_agent(use_noise_floor=False)
    donor = _mk_agent(use_noise_floor=True)
    assert off_agent.noise_floor is None
    object.__setattr__(off_agent, "noise_floor", donor.noise_floor)

    with pytest.raises(ArmRegulatorMismatch) as exc:
        assert_arm_regulators_live(
            off_agent, {"use_noise_floor": False}, logger=_noop_logger
        )
    assert "noise_floor" in str(exc.value)


# --------------------------------------------------------------------------- #
# passing cases -- an assertion that fires on correct usage gets switched off  #
# --------------------------------------------------------------------------- #

def test_assertion_passes_when_flags_and_regulators_agree_both_ways():
    agent = _mk_agent(use_noise_floor=True, use_phasic_burst=False)
    report = assert_arm_regulators_live(
        agent,
        {"use_noise_floor": True, "use_phasic_burst": False},
        logger=_noop_logger,
    )
    assert not report["violations"]
    assert set(report["checked"]) == {"noise_floor", "phasic_burst"}, (
        "BOTH flags must actually be checked -- a pass that silently checked "
        "nothing is the failure mode this assertion is supposed to remove"
    )


def test_assertion_checks_the_full_2x2_of_flag_states():
    """All four (flag, regulator) combinations across the real 963 axes."""
    for use_nf in (False, True):
        for use_pb in (False, True):
            agent = _mk_agent(use_noise_floor=use_nf, use_phasic_burst=use_pb)
            report = assert_arm_regulators_live(
                agent,
                {"use_noise_floor": use_nf, "use_phasic_burst": use_pb},
                logger=_noop_logger,
            )
            assert not report["violations"], (
                "a correctly-constructed arm must never trip the assertion "
                "(use_noise_floor=%r use_phasic_burst=%r)" % (use_nf, use_pb)
            )
            assert len(report["checked"]) == 2


def test_assertion_reports_every_violation_not_only_the_first():
    """A corrupted grid usually loses a whole axis, so a one-at-a-time assertion
    would need N reruns to surface N faults."""
    agent = _mk_agent(use_noise_floor=True, use_phasic_burst=True)
    object.__setattr__(agent, "noise_floor", None)
    object.__setattr__(agent, "phasic_burst", None)
    with pytest.raises(ArmRegulatorMismatch) as exc:
        assert_arm_regulators_live(
            agent,
            {"use_noise_floor": True, "use_phasic_burst": True},
            logger=_noop_logger,
        )
    msg = str(exc.value)
    assert "noise_floor" in msg and "phasic_burst" in msg


# --------------------------------------------------------------------------- #
# negative controls -- what the assertion must deliberately NOT do             #
# --------------------------------------------------------------------------- #

def test_no_arm_key_is_a_strict_noop():
    """THE BACKWARD-COMPATIBILITY CONTRACT. Every pre-fix caller passes no
    arm_key; with no declared flags there is nothing to check and nothing may
    raise -- even on an agent that IS corrupted."""
    agent = _mk_agent(use_noise_floor=True)
    object.__setattr__(agent, "noise_floor", None)  # corrupted on purpose
    for empty in (None, {}):
        report = assert_arm_regulators_live(agent, empty, logger=_noop_logger)
        assert report["violations"] == []
        assert report["checked"] == []


def test_non_use_prefixed_flags_are_skipped_not_raised():
    """`arm_key` is a free-form mapping; a caller may fold in non-regulator
    identity (a seed tag, a source name). Those do not name an attribute and
    must not be treated as violations."""
    agent = _mk_agent(use_noise_floor=True)
    report = assert_arm_regulators_live(
        agent,
        {"use_noise_floor": True, "phasic_burst_signal_source": "instantaneous_pe"},
        logger=_noop_logger,
    )
    assert not report["violations"]
    assert report["skipped_not_use_prefixed"] == ["phasic_burst_signal_source"]


def test_use_flag_with_no_matching_attribute_is_skipped_not_raised():
    """A flag that gates behaviour rather than constructing a root-level object
    has no `agent.<attr>` to check. Skipped, and COUNTED so an unexpectedly-empty
    check is visible rather than reading as a pass."""
    agent = _mk_agent()
    report = assert_arm_regulators_live(
        agent, {"use_no_such_regulator_xyz": True}, logger=_noop_logger
    )
    assert not report["violations"]
    assert report["skipped_no_such_attribute"] == ["use_no_such_regulator_xyz"]
    assert report["checked"] == []


# --------------------------------------------------------------------------- #
# wiring: warm_agent actually runs the check                                   #
# --------------------------------------------------------------------------- #

def test_warm_agent_exposes_the_assertion_parameter_defaulting_on():
    sig = inspect.signature(warm_agent)
    assert "assert_arm_regulators" in sig.parameters, (
        "warm_agent must run the check itself -- a helper nobody calls protects "
        "nothing, which is how the original _fresh_regulator asymmetry survived"
    )
    param = sig.parameters["assert_arm_regulators"]
    assert param.default is True, (
        "the check must be ON by default; a corrupting-severity guard that has "
        "to be opted into reproduces the defect it closes"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_warm_agent_calls_the_assertion_on_both_cache_paths():
    """Source-level pin: the call sits AFTER the hit/miss branch converges, so a
    HIT (restore) and a MISS (fresh train) are both covered. Pinned as source
    because exercising a real cache HIT needs a full warmup train, which is far
    too slow for a contract."""
    src = inspect.getsource(warm_agent)
    assert "assert_arm_regulators_live(" in src
    body = src.split("if assert_arm_regulators:", 1)
    assert len(body) == 2, "the call must be guarded by its own flag"
    head = body[0]
    assert "warmup trained fresh" in head and "warmup restored from cache" in head, (
        "the assertion must run AFTER both the HIT and MISS branches, not inside "
        "one of them -- a check on only the restore path would miss a MISS-path "
        "regression, and a check on only the miss path would miss V3-EXQ-963"
    )
