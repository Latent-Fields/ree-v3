"""Contracts for scripts/authority_trace_probe.py.

WHAT THIS DEFENDS. The probe's whole output is a table of verdicts, and every
interesting verdict is a NEGATIVE claim -- "this signal's effect dies here".
A negative claim from an instrument that cannot detect anything is
indistinguishable from a negative claim from an instrument that works, and the
three incidents the probe is built from (V3-EXQ-689d, EXP-0155,
V3-EXQ-931/932) are all cases where exactly that confusion cost a run or a
diagnostic cycle. So the assertions here are overwhelmingly about the
instrument's ability to FAIL LOUDLY rather than about any particular verdict.

Roughly half are NEGATIVE CONTROLS. Those are the ones that stop a later
session "simplifying" a guard until an unmeasurable result starts reading as a
finding:

  * a bogus flag name must be PRECONDITION_FAIL, never INERT (the
    `from_dims` silent-kwarg-swallow trap);
  * a flag that exists only as a constructor SIGNATURE default must not be
    reported as live-on (the 4-space/8-space source-parse trap, which inflated
    the population ~2x while this probe was being written);
  * a dirty null control must produce UNMEASURABLE regardless of how large the
    measured deltas are;
  * a known-live flag must NOT come back INERT (detectability).

ASCII-only. Run: pytest tests/contracts/test_authority_trace_probe.py -q
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_REE_V3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_REE_V3, "scripts", "authority_trace_probe.py")


def _load():
    """Import the probe as a real module.

    Registering in sys.modules BEFORE exec_module is load-bearing: @dataclass
    resolves annotations through sys.modules[cls.__module__], which raises
    AttributeError on a spec-loaded module that was never registered.
    """
    name = "_atp_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


atp = _load()


# --------------------------------------------------------------------------
# helpers -- synthetic captures, so the pure logic is tested without rollouts
# --------------------------------------------------------------------------

def _rec(ticks, actions):
    r = atp.Recorder.__new__(atp.Recorder)
    r.ticks = ticks
    r.actions = actions
    r.harm_sum = 0.0
    r.n_steps = len(actions)
    return r


def _tick(step, raw, sel, n_elig=None, arb=False):
    return atp.TickCapture(
        step=step, raw_scores=raw, scores=raw, selected_idx=sel,
        n_eligible=n_elig, arbitration_ran=arb, action=None,
    )


# --------------------------------------------------------------------------
# NEGATIVE CONTROLS -- the instrument must fail loudly
# --------------------------------------------------------------------------

def test_bogus_flag_is_precondition_fail_not_inert():
    """The from_dims swallow trap. THE most important assertion in this file.

    `REEConfig.from_dims(**kwargs)` silently ignores an unknown kwarg, so a
    typo'd flag yields an OFF arm identical to ON. Without the precondition
    check that reads back as INERT -- a false negative dressed as a finding.
    """
    r = atp.probe_flag("use_this_flag_does_not_exist_anywhere", [11], 4)
    assert r["verdict"] == "PRECONDITION_FAIL"
    assert r["verdict"] != "INERT"


def test_constructor_signature_default_is_not_reported_live_on():
    """The 4-space/8-space parse trap, pinned by example.

    `use_resource_encoder` is a dataclass field defaulting False AND an
    `enable_goal_stream` parameter defaulting True. A source parse that
    accepts any indentation reports it default-on; a bare default-constructed
    config says otherwise, and the config is what a run actually gets.
    """
    on = atp.discover_default_on_flags()
    assert "use_resource_encoder" not in on
    assert atp.live_default("use_resource_encoder") is False


def test_flag_absent_from_every_config_object_has_no_default():
    """`use_consumer_conjunction_read` is a constructor parameter only.

    It is never an attribute of any live config object, so it is outside the
    population this probe can speak about -- and must say so rather than
    silently scoring it.
    """
    assert atp.live_default("use_consumer_conjunction_read") is None
    r = atp.probe_flag("use_consumer_conjunction_read", [11], 4)
    assert r["verdict"] == "PRECONDITION_FAIL"


def test_dirty_null_control_forces_unmeasurable():
    """A dirty ON-vs-ON control outranks any measured delta."""
    big = {"n_scoring_diff": 99, "n_eligibility_diff": 9, "n_arbitration_diff": 9,
           "n_committed_diff": 9, "realised_actions_differ": True}
    assert atp.classify(big, control_clean=False) == "UNMEASURABLE"
    assert atp.classify(big, control_clean=True) == "REALISED"


def test_no_delta_anywhere_is_inert():
    zero = {"n_scoring_diff": 0, "n_eligibility_diff": 0, "n_arbitration_diff": 0,
            "n_committed_diff": 0, "realised_actions_differ": False}
    assert atp.classify(zero, control_clean=True) == "INERT"


def test_probe_does_not_mutate_the_substrate_defaults():
    """Running the probe must leave a freshly built default config unchanged."""
    before = atp.discover_default_on_flags()
    atp.probe_flag("use_bla_analog", [11], 4)
    after = atp.discover_default_on_flags()
    assert before == after


# --------------------------------------------------------------------------
# the SCORING_ONLY shape -- authority without throughput (V3-EXQ-931/932)
# --------------------------------------------------------------------------

def test_scores_differ_but_commit_never_moves_is_scoring_only():
    """The CEM cluster's exact signature: loud trace, nil effect.

    A verdict of REALISED here would erase the distinction the whole probe
    exists to draw.
    """
    d = {"n_scoring_diff": 12, "n_eligibility_diff": 0, "n_arbitration_diff": 0,
         "n_committed_diff": 0, "realised_actions_differ": False}
    assert atp.classify(d, control_clean=True) == "SCORING_ONLY"


def test_committed_move_outranks_scoring():
    d = {"n_scoring_diff": 12, "n_eligibility_diff": 0, "n_arbitration_diff": 0,
         "n_committed_diff": 1, "realised_actions_differ": False}
    assert atp.classify(d, control_clean=True) == "REALISED"


# --------------------------------------------------------------------------
# comparison window -- divergence bounds the comparable prefix
# --------------------------------------------------------------------------

def test_comparison_stops_at_the_first_divergent_action():
    """After divergence the two agents are in different world states.

    Comparing scores past that point compares different problems, so those
    ticks must be excluded from the denominator rather than counted as
    agreement (which would dilute every ratio toward zero).
    """
    a = _rec([_tick(1, [1.0, 2.0], 0), _tick(2, [1.0, 2.0], 0)],
             [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)])
    b = _rec([_tick(1, [1.0, 2.0], 0), _tick(2, [5.0, 9.0], 1)],
             [(1.0, 0.0), (0.0, 1.0), (0.0, 1.0)])
    d = atp._pairwise_stage_delta(a, b)
    assert d["divergence_step"] == 2
    assert d["n_comparable_ticks"] == 1
    assert d["n_scoring_diff"] == 0        # tick 2 excluded, and tick 1 agrees
    assert d["realised_actions_differ"] is True


def test_identical_rollouts_report_zero_everywhere():
    ticks = [_tick(1, [1.0, 2.0], 0), _tick(2, [3.0, 1.0], 1)]
    acts = [(1.0, 0.0), (0.0, 1.0)]
    d = atp._pairwise_stage_delta(_rec(list(ticks), list(acts)),
                                  _rec(list(ticks), list(acts)))
    assert d["n_comparable_ticks"] == 2
    assert d["divergence_step"] is None
    assert d["n_scoring_diff"] == 0
    assert d["n_committed_diff"] == 0
    assert d["authority_spread_ratio_mean"] == 0.0
    assert d["realised_actions_differ"] is False


def test_spread_ratio_is_relative_not_absolute():
    """`authority_spread_ratio` semantics: the signal's cross-candidate spread
    against the dominant term's. V3-EXQ-931 read ~0.0037 -- nonzero, and not
    competitive. An absolute delta cannot express that.
    """
    a = _rec([_tick(1, [0.0, 10.0], 0)], [(1.0, 0.0)])
    b = _rec([_tick(1, [0.0, 11.0], 0)], [(1.0, 0.0)])
    d = atp._pairwise_stage_delta(a, b)
    # delta spread 1.0 against base spread 10.0
    assert d["authority_spread_ratio_mean"] == pytest.approx(0.1)
    assert d["max_abs_score_delta"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# direction awareness + detectability
# --------------------------------------------------------------------------

def test_contrast_direction_follows_the_live_default():
    """A default-OFF flag is contrasted in the ON direction.

    Without this the probe is unusable on 160 of the 173 flags, and no
    positive control is possible at all -- every default-off flag would be
    trivially INERT because its "off" arm IS the default.
    """
    assert atp.live_default("use_dualsystem_arbitration") is False
    r = atp.probe_flag("use_dualsystem_arbitration", [11], 4)
    assert r["contrast"] == "False -> True"
    on = atp.discover_default_on_flags()
    assert "use_bla_analog" in on
    r2 = atp.probe_flag("use_bla_analog", [11], 4)
    assert r2["contrast"] == "True -> False"


def test_positive_control_a_known_live_flag_is_not_inert():
    """DETECTABILITY. If this ever reports INERT the whole table is worthless.

    `use_dualsystem_arbitration` replaces `scores` wholesale via
    `_arbitrate_dual_system` (e3_selector.py:2750), so its ablation contrast
    must be visible at the scoring stage at minimum.
    """
    r = atp.probe_flag("use_dualsystem_arbitration", [11], 60)
    assert r["verdict"] not in ("INERT", "PRECONDITION_FAIL", "UNMEASURABLE")


def test_freshness_gating_not_one_capture_per_env_step():
    """The hold-weighted readout defect (699 / 689d).

    `heartbeat.e3_steps_per_tick` defaults >1 and agent.py returns the HELD
    action on a non-E3 tick before select() is reached. One capture per env
    step would replicate a single genuine selection across the whole hold.
    """
    steps = 60
    rec = atp.rollout(11, {}, steps, "atp_freshness_test")
    assert 0 < len(rec.ticks) < steps
    assert len(rec.actions) == steps
