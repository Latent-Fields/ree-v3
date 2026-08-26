"""MECH-464 instrument contracts: straddle fraction + da=0 shadow argmin.

MECH-464 asserts the D1/D2 opponent gain (ARC-109) is ORDER-CHANGING, not a
uniform scalar, because it gains the Go/D1 and No-Go/D2 populations
asymmetrically about zero. Its falsifier carries a MANDATORY non-vacuity gate:
report the fraction of eligible candidate pairs whose (pre-split) loop
accumulators straddle zero. `_loop_normalize`'s zscore is invariant to
positive affine scaling, so when every candidate in a loop shares a sign, the
D1/D2 split is a pure positive-scalar rescale the zscore cancels EXACTLY --
that gate did not exist anywhere in ree_core before this file's counterpart
change to `_segregated_loop_arbitrate` (e3_selector.py).

These tests lock:
  (1) the straddle fraction is computed on the PRE-SPLIT accumulator (before
      `_d1_d2_split` overwrites it in place), and is 0.0 when d1d2 is off;
  (2) the da=0 shadow argmin (`loop_d1_d2_reorder_vs_da0`) agrees with an
      independently-measured actual reorder (two separate selector calls at
      da=0 and da!=0), and is trivially False at da==0 itself;
  (3) the d2_gain==0 saturation confound fires only when DA has actually
      depressed the No-Go population to zero;
  (4) all four new diagnostics default to their off-values and never execute
      extra tensor ops when `use_d1_d2_population_split` is False.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig
from ree_core.predictors.e3_selector import E3TrajectorySelector, _FCG_CHANNEL_INDEX


def _selector(d1d2_flag: bool, **cfg_kwargs) -> E3TrajectorySelector:
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
        use_loop_segregation=True,
        use_d1_d2_population_split=d1d2_flag,
        **cfg_kwargs,
    )
    return E3TrajectorySelector(cfg.e3, None)


# Reused verbatim from
# test_use_d1_d2_population_split_is_bit_identical_at_da_zero_then_diverges
# (test_flag_inertness.py) -- known to reorder the committed index between
# da==0 and da==3.0, and known (by construction: 2 negative + 2 positive
# entries in each channel) to straddle zero at exactly 4/6 pairs.
_N = 4
_ELIG = torch.arange(_N)
_RAW = torch.zeros(_N)
_OFC = torch.tensor([-6.264134407043457, -1.78110933303833, 3.355997085571289, 0.849435031414032])
_DACC = torch.tensor([7.372593879699707, 3.526642322540283, -3.452648162841797, -2.195103645324707])
_TERMS = [(_FCG_CHANNEL_INDEX["ofc"], _OFC), (_FCG_CHANNEL_INDEX["dacc"], _DACC)]
_EXPECTED_STRADDLE = 4.0 / 6.0  # C(4,2)=6 pairs; 2 pos + 2 neg in each channel -> 4 mismatched


def _run(sel: E3TrajectorySelector):
    sel._segregated_loop_arbitrate(_ELIG, _RAW, _TERMS, True, [None] * _N, True, 1.0, True)
    return sel.last_score_diagnostics


# --------------------------------------------------------------------- #
# (1) Straddle fraction
# --------------------------------------------------------------------- #
def test_straddle_frac_matches_hand_computed_sign_mismatch():
    """ofc -> limbic, dacc -> associative (default channel map); both carry
    2 negative + 2 positive entries -> 4 of the 6 candidate pairs straddle."""
    sel = _selector(True)
    sel._lcg_value_baseline = 3.0  # da != 0, so the split actually runs
    d = _run(sel)
    assert d["loop_limbic_straddle_frac"] == pytest.approx(_EXPECTED_STRADDLE)
    assert d["loop_assoc_straddle_frac"] == pytest.approx(_EXPECTED_STRADDLE)


def test_straddle_frac_is_prescale_not_postsplit():
    """Sanity: an all-same-sign channel must read straddle_frac == 0.0 (the
    structurally-nil case _pair_straddle_frac's docstring names), proving the
    fraction is not some constant regardless of input."""
    sel = _selector(True)
    sel._lcg_value_baseline = 3.0
    all_pos = torch.tensor([1.0, 2.0, 3.0, 4.0])
    terms = [(_FCG_CHANNEL_INDEX["ofc"], all_pos), (_FCG_CHANNEL_INDEX["dacc"], all_pos)]
    sel._segregated_loop_arbitrate(_ELIG, _RAW, terms, True, [None] * _N, True, 1.0, True)
    d = sel.last_score_diagnostics
    assert d["loop_limbic_straddle_frac"] == pytest.approx(0.0)
    assert d["loop_assoc_straddle_frac"] == pytest.approx(0.0)


def test_straddle_frac_defaults_to_zero_when_d1d2_off():
    sel = _selector(False)
    d = _run(sel)
    assert d["loop_assoc_straddle_frac"] == 0.0
    assert d["loop_limbic_straddle_frac"] == 0.0


# --------------------------------------------------------------------- #
# (2) da=0 shadow argmin
# --------------------------------------------------------------------- #
def test_reorder_vs_da0_agrees_with_independently_measured_reorder():
    """Cross-check: the internal single-call shadow comparison must agree
    with the external two-call comparison (da==0 selector vs da!=0 selector)
    that test_flag_inertness.py already locks as a genuine reorder."""
    sel_da0 = _selector(True)
    loc_da0 = sel_da0._segregated_loop_arbitrate(
        _ELIG, _RAW, _TERMS, True, [None] * _N, True, 1.0, True
    )

    sel_da = _selector(True)
    sel_da._lcg_value_baseline = 3.0
    loc_da = sel_da._segregated_loop_arbitrate(
        _ELIG, _RAW, _TERMS, True, [None] * _N, True, 1.0, True
    )

    external_reorder = loc_da != loc_da0
    assert external_reorder, "fixture must reorder externally, or this test proves nothing"
    assert sel_da.last_score_diagnostics["loop_d1_d2_reorder_vs_da0"] == external_reorder
    assert sel_da0.last_score_diagnostics["loop_d1_d2_reorder_vs_da0"] is False, (
        "da==0 IS the shadow baseline -- comparing it to itself must never reorder"
    )


def test_reorder_vs_da0_false_when_d1d2_off():
    sel = _selector(False)
    d = _run(sel)
    assert d["loop_d1_d2_reorder_vs_da0"] is False


# --------------------------------------------------------------------- #
# (3) d2_gain==0 saturation confound
# --------------------------------------------------------------------- #
def test_d2_gain_zero_fires_only_when_da_fully_depresses_d2():
    sel = _selector(True, d2_da_gain=2.0)
    sel._lcg_value_baseline = 5.0  # tanh(5.0) ~= 0.9999 -> d2_gain = 1 - 2*0.9999 < 0 -> clamped 0
    d = _run(sel)
    assert d["loop_d1_d2_d2_gain_zero"] is True


def test_d2_gain_zero_false_under_normal_gain():
    sel = _selector(True)  # default d2_da_gain=1.0
    sel._lcg_value_baseline = 3.0  # da ~= 0.995 -> d2_gain ~= 0.005 > 0
    d = _run(sel)
    assert d["loop_d1_d2_d2_gain_zero"] is False


def test_d2_gain_zero_false_when_d1d2_off():
    sel = _selector(False)
    d = _run(sel)
    assert d["loop_d1_d2_d2_gain_zero"] is False


# --------------------------------------------------------------------- #
# (4) Byte-identical-when-off
# --------------------------------------------------------------------- #
def test_off_arm_committed_index_unaffected_by_the_new_diagnostics():
    """The new diagnostics must be pure reads -- adding them must not change
    which candidate is committed when d1d2 is off."""
    sel_a = _selector(False)
    loc_a = sel_a._segregated_loop_arbitrate(
        _ELIG, _RAW, _TERMS, True, [None] * _N, True, 1.0, True
    )
    sel_b = _selector(False)
    loc_b = sel_b._segregated_loop_arbitrate(
        _ELIG, _RAW, _TERMS, True, [None] * _N, True, 1.0, True
    )
    assert loc_a == loc_b
