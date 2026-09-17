"""Contract tests for the MECH-320 v_raw baseline (2026-09-17 substrate build).

THE DEFECT THIS BUILD ADDRESSES
-------------------------------
`TonicVigor.update_score_receipt` set `reward_signal = -float(score)`, and
`compute_score_bias` reads `max(0.0, self._v_raw)`. REE scores are
lower-is-better COSTS and are essentially always positive, so that EWMA is a
pure NEGATIVE accumulator: `_v_raw` can never exceed 0, `max(0.0, _v_raw)` is
identically zero, and `v_t` is pinned at `v_t_floor` before any gate is read.
V3-EXQ-951c measured exactly that -- v_raw max EXACTLY 0.0, means -14.26 and
-22.97 -- i.e. MECH-320's own scalar was dead at the v_t_floor=0.0 default.

THE LEVER
---------
`TonicVigorConfig.baseline_mode` (`REEConfig.tonic_vigor_baseline_mode`):

  "none" (DEFAULT) : reward_signal = -score              [pre-2026-09-17]
  "ewma"           : reward_signal = score_baseline - score, where
                     score_baseline is an EWMA of the score stream at
                     `baseline_half_life`.

CONTRACTS
  B1: default is "none" on both TonicVigorConfig and REEConfig, and an agent
      built with use_tonic_vigor=True inherits "none".
  B2: OFF-PATH BIT-IDENTITY. Under baseline_mode="none" the EWMA trajectory is
      EXACTLY (float ==, not approx) the pre-change reference arithmetic
      v_raw <- (1-alpha)*v_raw + alpha*(-score), over a long mixed-sign stream.
  B3: OFF-PATH DEAD-SCALAR REPRODUCTION. Under "none" with a positive-cost
      score stream, v_raw stays <= 0 on every tick, so max(0, v_raw) is
      identically 0 -- the V3-EXQ-951c signature.
  B4: ON-PATH. Under "ewma" on an improving (falling-cost) score stream,
      v_raw goes strictly positive, so v_t is no longer pinned at v_t_floor.
  B5: ON-PATH seeds its baseline on the FIRST receipt (first advantage is
      exactly 0.0), not from the 0.0 initialisation.
  B6: validation -- bogus baseline_mode and non-positive baseline_half_life
      raise ValueError at construction.
  B7: reset() clears baseline state on both paths.
  B8: MECH-094 simulation gate still holds on the new path (no baseline
      advance, no EWMA advance, only the skip counter moves).
  B9: the new path actually reaches compute_score_bias -- a positive v_raw
      produces a non-zero bias where the "none" path produces zeros.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.policy.tonic_vigor import TonicVigor, TonicVigorConfig
from ree_core.utils.config import REEConfig


def _reference_pre_change_v_raw(scores, half_life):
    """Verbatim pre-2026-09-17 arithmetic, recomputed independently."""
    alpha = 1.0 - math.pow(0.5, 1.0 / float(half_life))
    v_raw = 0.0
    out = []
    for s in scores:
        reward_signal = -float(s)
        v_raw = (1.0 - alpha) * v_raw + alpha * reward_signal
        out.append(v_raw)
    return out


def _mixed_stream(n=400):
    """Deterministic mixed-magnitude score stream (no RNG dependency)."""
    return [
        (3.0 + 2.5 * math.sin(i * 0.37) + 0.011 * i) for i in range(n)
    ]


def _falling_stream(n=400):
    """Improving agent: costs fall steadily from ~20 to ~2."""
    return [20.0 - 18.0 * (i / float(n - 1)) for i in range(n)]


# ----------------------------------------------------------------------
# B1 -- default is the pre-change path
# ----------------------------------------------------------------------
def test_b1_default_baseline_mode_is_none():
    assert TonicVigorConfig().baseline_mode == "none"
    assert REEConfig().tonic_vigor_baseline_mode == "none"


def test_b1_agent_inherits_none_default():
    cfg = REEConfig()
    cfg.use_tonic_vigor = True
    from ree_core.agent import REEAgent

    agent = REEAgent(config=cfg)
    assert agent.tonic_vigor is not None
    assert agent.tonic_vigor.config.baseline_mode == "none"


# ----------------------------------------------------------------------
# B2 -- off-path bit-identity against the pre-change reference
# ----------------------------------------------------------------------
@pytest.mark.parametrize("half_life", [10.0, 100.0, 500.0])
def test_b2_off_path_is_bit_identical_to_pre_change_reference(half_life):
    scores = _mixed_stream()
    expected = _reference_pre_change_v_raw(scores, half_life)

    tv = TonicVigor(TonicVigorConfig(use_tonic_vigor=True, half_life=half_life))
    for i, s in enumerate(scores):
        tv.update_score_receipt(s)
        # EXACT equality, not approx: this is the bit-identity claim.
        assert tv._v_raw == expected[i], (
            "off-path EWMA diverged from the pre-change reference at tick "
            f"{i}: {tv._v_raw!r} != {expected[i]!r}"
        )


def test_b2_off_path_baseline_state_is_never_touched():
    tv = TonicVigor(TonicVigorConfig(use_tonic_vigor=True, half_life=50.0))
    for s in _mixed_stream(120):
        tv.update_score_receipt(s)
    assert tv._score_baseline == 0.0
    assert tv._baseline_initialised is False


# ----------------------------------------------------------------------
# B3 -- off-path reproduces the dead-scalar signature
# ----------------------------------------------------------------------
def test_b3_off_path_v_raw_never_exceeds_zero_on_positive_costs():
    tv = TonicVigor(TonicVigorConfig(use_tonic_vigor=True, half_life=100.0))
    seen_max = -float("inf")
    for s in _falling_stream():
        tv.update_score_receipt(s)
        seen_max = max(seen_max, tv._v_raw)
        assert max(0.0, tv._v_raw) == 0.0
    assert seen_max <= 0.0
    assert tv._v_raw < 0.0


# ----------------------------------------------------------------------
# B4 -- on-path lets v_raw go positive
# ----------------------------------------------------------------------
def test_b4_ewma_path_drives_v_raw_positive_on_improving_stream():
    tv = TonicVigor(
        TonicVigorConfig(
            use_tonic_vigor=True,
            half_life=20.0,
            baseline_mode="ewma",
            baseline_half_life=200.0,
        )
    )
    positives = 0
    for s in _falling_stream():
        tv.update_score_receipt(s)
        if tv._v_raw > 0.0:
            positives += 1
    assert tv._v_raw > 0.0, "improving stream must leave v_raw positive"
    assert positives > 0
    assert max(0.0, tv._v_raw) > 0.0, "the gate input is no longer pinned at 0"


def test_b4_ewma_and_none_diverge_on_the_same_stream():
    scores = _falling_stream(200)
    kw = dict(use_tonic_vigor=True, half_life=20.0)
    off = TonicVigor(TonicVigorConfig(**kw))
    on = TonicVigor(
        TonicVigorConfig(baseline_mode="ewma", baseline_half_life=200.0, **kw)
    )
    for s in scores:
        off.update_score_receipt(s)
        on.update_score_receipt(s)
    assert off._v_raw < 0.0 < on._v_raw


# ----------------------------------------------------------------------
# B5 -- first receipt seeds the baseline
# ----------------------------------------------------------------------
def test_b5_first_receipt_gives_exactly_zero_advantage():
    tv = TonicVigor(
        TonicVigorConfig(
            use_tonic_vigor=True, baseline_mode="ewma", baseline_half_life=100.0
        )
    )
    assert tv._baseline_initialised is False
    tv.update_score_receipt(17.5)
    assert tv._baseline_initialised is True
    # advantage was baseline(=17.5) - 17.5 == 0.0, so the EWMA did not move.
    assert tv._v_raw == 0.0
    # ... but the baseline itself has advanced off the seed by one EWMA step
    # only if the seed differed from the score; here it did not.
    assert tv._score_baseline == 17.5


# ----------------------------------------------------------------------
# B6 -- validation
# ----------------------------------------------------------------------
def test_b6_bogus_baseline_mode_raises():
    with pytest.raises(ValueError, match="baseline_mode"):
        TonicVigor(TonicVigorConfig(baseline_mode="bogus"))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_b6_non_positive_baseline_half_life_raises(bad):
    with pytest.raises(ValueError, match="baseline_half_life"):
        TonicVigor(TonicVigorConfig(baseline_half_life=bad))


# ----------------------------------------------------------------------
# B7 -- reset
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["none", "ewma"])
def test_b7_reset_clears_baseline_state(mode):
    tv = TonicVigor(
        TonicVigorConfig(use_tonic_vigor=True, baseline_mode=mode)
    )
    for s in _falling_stream(50):
        tv.update_score_receipt(s)
    tv.reset()
    assert tv._v_raw == 0.0
    assert tv._score_baseline == 0.0
    assert tv._baseline_initialised is False


# ----------------------------------------------------------------------
# B8 -- MECH-094 simulation gate holds on the new path
# ----------------------------------------------------------------------
def test_b8_simulation_mode_does_not_advance_baseline():
    tv = TonicVigor(
        TonicVigorConfig(use_tonic_vigor=True, baseline_mode="ewma")
    )
    tv.update_score_receipt(10.0, simulation_mode=False)
    v_after_waking = tv._v_raw
    base_after_waking = tv._score_baseline
    for _ in range(25):
        tv.update_score_receipt(1.0, simulation_mode=True)
    assert tv._v_raw == v_after_waking
    assert tv._score_baseline == base_after_waking
    assert tv._n_simulation_score_skips == 25
    assert tv._n_waking_score_updates == 1


# ----------------------------------------------------------------------
# B9 -- the scalar reaches the bias path
# ----------------------------------------------------------------------
def test_b9_positive_v_raw_produces_non_zero_bias():
    scores = _falling_stream(300)
    kw = dict(
        use_tonic_vigor=True, half_life=20.0, w_action=0.1, w_passive=0.1
    )
    off = TonicVigor(TonicVigorConfig(**kw))
    on = TonicVigor(
        TonicVigorConfig(baseline_mode="ewma", baseline_half_life=200.0, **kw)
    )
    for s in scores:
        off.update_score_receipt(s)
        on.update_score_receipt(s)

    cand = torch.tensor([1.0, 2.0, 3.0, 4.0])
    classes = torch.tensor([0, 1, 2, 3])
    off_bias = off.compute_score_bias(cand, classes, 1.0, 0.0, 0.0)
    on_bias = on.compute_score_bias(cand, classes, 1.0, 0.0, 0.0)

    assert torch.all(off_bias == 0.0), "off path must stay inert (v_t pinned)"
    assert float(on_bias.abs().max().item()) > 0.0
    # sign convention: action classes get a NEGATIVE bias, noop a POSITIVE one.
    assert float(on_bias[0].item()) > 0.0
    assert float(on_bias[1].item()) < 0.0


def test_b9_get_state_exposes_baseline_fields():
    tv = TonicVigor(
        TonicVigorConfig(use_tonic_vigor=True, baseline_mode="ewma")
    )
    tv.update_score_receipt(5.0)
    st = tv.get_state()
    assert st["baseline_mode"] == "ewma"
    assert st["score_baseline"] == 5.0
    assert st["baseline_initialised"] is True
    assert st["baseline_alpha_derived"] > 0.0
