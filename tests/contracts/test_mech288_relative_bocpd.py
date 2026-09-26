"""Contract tests: MECH-288 magnitude-relative slow-scale BOCPD trigger.

substrate_queue MECH288-SLOW-SCALE-BOCPD-RAIL-UNREACHABLE (IGW-20260925-220).
Design + red-team: REE_assembly/evidence/planning/
mech288_slow_scale_rail_redesign_20260926.md.

Guarantees:
  R1. Default path is the canonical detector, pinned against golden values
      recorded from the PRE-CHANGE event_segmenter.py (ree-v3 7f08512) on a
      deterministic stream -- including that it does NOT fire on a 20%
      relative shift at 0.3 scale (the defect this build answers).
  R2. The D3 identity still holds on the default path: P(r_t=0) == hazard
      (up to top-k renormalisation), so the p0 readout never fires.
  R3. Relative mode fires on that same 20% shift within 1 tick.
  R4. Relative mode is scale-equivariant (x1000 -> identical fire ticks).
  R5. Relative mode is silent on pure geometric decay (the in-agent
      between-write z_goal stream).
  R6. short_run_mass burn-in: no fire in the first readout_lag+1
      observations, nor in the readout_lag+1 after a reseed.
  R7. The rollout stream always gets the canonical detector, even with a
      relative scale config (repeated identical z_goal would otherwise
      collapse run variance and fire at the min_segment_length ceiling).
  R8. HippocampalModule passes the bocpd_* scale-config fields through.
  R9. Unknown scale_mode / readout raise.
"""

from __future__ import annotations

import math

import pytest
import torch

from ree_core.hippocampal.event_segmenter import (
    EventSegmenter,
    Scale,
    _BOCPDGaussianDetector,
)

HAZARD = 1.0 / 40.0


def _stream():
    # Planted levels with a deterministic wobble; the last shift (0.30 ->
    # 0.36 at t=150) is a 20% relative move, far below the ~6.18 absolute rail.
    xs = []
    for k, level in enumerate([5.0, 15.0, 25.0, 10.0, 0.3, 0.36]):
        for i in range(30):
            xs.append(level + 0.02 * math.sin(1.7 * i + k))
    return xs


def _drive(det, xs):
    return [det.step(("z_goal",), {"z_goal": torch.tensor([x])}, {})[:2] for x in xs]


def _slow(**kw):
    return Scale(
        name="slow", streams=("z_goal",), algorithm="bocpd_gaussian", tau=40,
        min_segment_length=15, hazard=HAZARD, posterior_threshold=0.5, **kw,
    )


def _relative_det(**kw):
    return _BOCPDGaussianDetector(
        hazard=HAZARD, posterior_threshold=0.5, scale_mode="relative",
        readout="short_run_mass", **kw,
    )


def _fire_ticks(out):
    return [t for t, (fired, _p) in enumerate(out) if fired]


def test_r1_default_path_matches_pre_change_golden():
    det = _BOCPDGaussianDetector(hazard=HAZARD, posterior_threshold=0.5)
    assert det.scale_mode == "absolute" and det.readout == "p0"
    out = _drive(det, _stream())
    # Golden fire ticks recorded from ree-v3 7f08512 (pre-change) on _stream().
    assert _fire_ticks(out) == [30, 60, 90, 120]
    assert 150 not in _fire_ticks(out)          # the unreachable-rail defect
    for t in (30, 60, 90, 120):
        assert out[t][1] == 1.0


def test_r2_default_p0_equals_hazard():
    det = _BOCPDGaussianDetector(hazard=HAZARD, posterior_threshold=0.5)
    out = _drive(det, _stream())
    posterior_ticks = [t for t in range(1, len(out)) if t not in (30, 60, 90, 120)]
    # == hazard up to top-k renormalisation (drift ~1e-9 measured), so the
    # p0 readout can never approach posterior_threshold=0.5.
    for t in posterior_ticks:
        assert out[t][1] == pytest.approx(HAZARD, rel=1e-4)
        assert out[t][0] is False


def test_r3_relative_fires_on_20pct_shift_within_one_tick():
    out = _drive(_relative_det(), _stream())
    fires = _fire_ticks(out)
    assert any(150 <= t <= 151 for t in fires), fires
    # and it still sees every planted absolute jump
    for cp in (30, 60, 90, 120):
        assert any(cp <= t <= cp + 1 for t in fires), (cp, fires)


def test_r4_relative_is_scale_equivariant():
    xs = _stream()
    a = _fire_ticks(_drive(_relative_det(), xs))
    b = _fire_ticks(_drive(_relative_det(), [1000.0 * x for x in xs]))
    assert a == b and a


def test_r5_relative_silent_on_geometric_decay():
    xs = [0.3 * 0.995 ** t for t in range(600)]
    assert _fire_ticks(_drive(_relative_det(), xs)) == []


def test_r6_burn_in_after_start_and_after_reseed():
    lag = 3
    det = _relative_det(readout_lag=lag)
    # start: every run is short, so short-run mass is 1 -- must not fire.
    start = _drive(det, [0.1] * (lag + 2))
    assert not any(f for f, _ in start)
    assert start[1][1] == pytest.approx(1.0)
    # force a reseed with a large jump, then hold the new level.
    det2 = _relative_det(readout_lag=lag)
    out = _drive(det2, [0.1] * 40 + [0.5] + [0.5] * (lag + 2))
    assert out[40][0] is True                    # the jump itself fires
    after = out[41:41 + lag]
    assert not any(f for f, _ in after)          # held off during burn-in


def test_r7_rollout_stream_forced_canonical():
    seg = EventSegmenter(scales=[_slow(scale_mode="relative", readout="short_run_mass")])
    assert seg._detectors["observation"]["slow"].scale_mode == "relative"
    roll = seg._detectors["rollout"]["slow"]
    assert roll.scale_mode == "absolute" and roll.readout == "p0"
    fired = 0
    for t, x in enumerate(_stream()[120:]):          # small-scale stretch
        z = torch.tensor([[x]])
        for _ in range(8):                           # repeated per tick, as the MECH-321 probe does
            fired += len(seg.step(latent_dict={"z_goal": z}, pe_dict=None, t=t,
                                  input_stream="rollout"))
    assert fired == 0


def _agent_slow_detector(**slow_attrs):
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4,
                              alpha_world=0.3, use_event_segmenter=True)
    slow = [sc for sc in cfg.hippocampal.event_segmenter.scales if sc.name == "slow"][0]
    for k, v in slow_attrs.items():
        setattr(slow, k, v)
    agent = REEAgent(cfg)
    return agent.hippocampal.event_segmenter._detectors


def test_r8_hippocampal_module_passes_fields_through():
    dets = _agent_slow_detector(bocpd_scale_mode="relative",
                                bocpd_readout="short_run_mass", bocpd_readout_lag=5)
    obs = dets["observation"]["slow"]
    assert obs.scale_mode == "relative"
    assert obs.readout == "short_run_mass" and obs.readout_lag == 5
    assert dets["rollout"]["slow"].scale_mode == "absolute"
    default = _agent_slow_detector()["observation"]["slow"]
    assert default.scale_mode == "absolute" and default.readout == "p0"


def test_r9_invalid_modes_raise():
    with pytest.raises(ValueError):
        _BOCPDGaussianDetector(hazard=HAZARD, posterior_threshold=0.5, scale_mode="log")
    with pytest.raises(ValueError):
        _BOCPDGaussianDetector(hazard=HAZARD, posterior_threshold=0.5, readout="map")


def test_r10_from_dims_knob_reaches_slow_scale():
    """REEConfig.from_dims(event_segmenter_slow_relative_trigger=True) must reach
    the slow scale's detector -- from_dims silently swallows unknown kwargs
    (MECH-307 precedent), so a missing signature entry reads as OFF with no
    error. The fast scale and the rollout stream stay canonical."""
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent

    def dets(**kw):
        cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4,
                                  alpha_world=0.3, use_event_segmenter=True, **kw)
        return REEAgent(cfg).hippocampal.event_segmenter._detectors

    on = dets(event_segmenter_slow_relative_trigger=True)
    assert on["observation"]["slow"].scale_mode == "relative"
    assert on["observation"]["slow"].readout == "short_run_mass"
    assert on["rollout"]["slow"].scale_mode == "absolute"
    off = dets()
    assert off["observation"]["slow"].scale_mode == "absolute"
    assert off["observation"]["slow"].readout == "p0"
