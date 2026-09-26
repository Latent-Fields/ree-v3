"""
Contract tests for suffering-derivative-comparator-refractory (2026-09-26,
IGW-20260924-224; failure_autopsy_gflag0452-D1-cluster_2026-09-24).

SufferingDerivativeComparator.tick() fires on every tick whose rolling-window
drop clears drop_threshold, so one damage->heal trajectory emits a TRAIN of
relief-completion events (~9-17 per scheduled injection in V3-EXQ-517d). The
opt-in re-arm-on-rise latch (suffering_event_latch_enabled) makes one descent
yield one event.

Contracts:
  R1  Defaults: config fields default OFF / None; from_dims wires both knobs
      through to the agent's comparator (from_dims silently swallows unknown
      kwargs, so this is checked on the constructed agent, not the signature).
  R2  Latch OFF is bit-identical to the pre-2026-09-26 tick() (oracle below,
      copied verbatim from the pre-change module) on noisy multi-descent streams.
  R3  Latch ON: one event per damage->heal descent; latch OFF emits a train.
  R4  Latch ON fire ticks are a strict SUBSET of latch OFF fire ticks, and
      the first fire of a descent lands on the same tick as unlatched.
  R5  Re-arm: a rise >= rearm_rise above the post-fire trough re-arms (next
      descent fires); a smaller bump does not.
  R6  simulation_mode=True returns False and leaves buffer + latch untouched
      (MECH-094).
  R7  reset() clears latch and counters; negative rearm_rise raises.
"""

import random

import pytest

from ree_core.comparator.suffering_derivative_comparator import (
    SufferingDerivativeComparator,
)
from ree_core.utils.config import REEConfig

# V3-EXQ-517d comparator settings.
W, THR, MIN0 = 30, 0.005, 0.01


def _legacy_fires(norms, window, thr, min0):
    """Pre-2026-09-26 SufferingDerivativeComparator.tick(), verbatim logic."""
    buf, out = [], []
    for n in norms:
        buf.append(n)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) < window:
            out.append(False)
            continue
        if buf[0] < min0:
            out.append(False)
            continue
        out.append((buf[0] - buf[-1]) >= thr)
    return out


def _run(comp, norms):
    return [comp.tick(n) for n in norms]


def _heal_stream(injection_ticks, n_ticks, magnitude=0.4, heal=0.002, base=0.02,
                 gain=1.0, noise=0.0, seed=0):
    """Scalar harm-norm proxy: base + gain * damage, damage healing geometrically
    (SD-022 limb_damage *= 1 - heal_rate), topped up at each injection tick."""
    rng = random.Random(seed)
    dmg, out = 0.0, []
    inj = set(injection_ticks)
    for t in range(n_ticks):
        if t in inj:
            dmg = min(1.0, dmg + magnitude)
        dmg *= (1.0 - heal)
        out.append(base + gain * dmg + (rng.gauss(0.0, noise) if noise else 0.0))
    return out


# ---------------------------------------------------------------------------
# R1  Defaults and from_dims wiring
# ---------------------------------------------------------------------------

def test_r1_config_defaults_off():
    cfg = REEConfig()
    assert cfg.suffering_event_latch_enabled is False
    assert cfg.suffering_rearm_rise is None
    c = SufferingDerivativeComparator()
    assert c.latch_enabled is False
    assert c.rearm_rise == pytest.approx(c.drop_threshold)


def test_r1_from_dims_reaches_agent_comparator():
    from ree_core.agent import REEAgent

    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=5,
        use_suffering_derivative_comparator=True,
        suffering_drop_threshold=THR,
        suffering_event_latch_enabled=True,
        suffering_rearm_rise=0.02,
    )
    assert cfg.suffering_event_latch_enabled is True
    assert cfg.suffering_rearm_rise == 0.02
    agent = REEAgent(cfg)
    assert agent.suffering_comparator is not None
    assert agent.suffering_comparator.latch_enabled is True
    assert agent.suffering_comparator.rearm_rise == pytest.approx(0.02)

    cfg_off = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=5,
        use_suffering_derivative_comparator=True,
    )
    agent_off = REEAgent(cfg_off)
    assert agent_off.suffering_comparator.latch_enabled is False


# ---------------------------------------------------------------------------
# R2  Latch OFF bit-identical to legacy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_r2_latch_off_matches_legacy_oracle(seed):
    rng = random.Random(seed)
    inj = sorted(rng.sample(range(20, 1400), 8))
    norms = _heal_stream(inj, 1500, noise=0.003, seed=seed)
    for (w, thr, m0) in [(W, THR, MIN0), (5, 0.10, 0.05), (10, 0.02, 0.0)]:
        comp = SufferingDerivativeComparator(w, thr, m0)  # latch default OFF
        got = _run(comp, norms)
        assert got == _legacy_fires(norms, w, thr, m0)
        assert comp.suppressed_count == 0
        assert comp.fire_count == sum(got)


# ---------------------------------------------------------------------------
# R3  One event per descent ON; train OFF
# ---------------------------------------------------------------------------

def test_r3_one_event_per_descent():
    inj = [50, 450, 850]
    norms = _heal_stream(inj, 1250)
    off = _run(SufferingDerivativeComparator(W, THR, MIN0), norms)
    comp_on = SufferingDerivativeComparator(W, THR, MIN0, latch_enabled=True)
    on = _run(comp_on, norms)
    assert sum(off) > 3 * len(inj), "precondition: unlatched comparator emits a train"
    assert sum(on) == len(inj)
    # exactly one fire inside each injection's heal segment
    bounds = inj + [len(norms)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        assert sum(on[a:b]) == 1
    assert comp_on.suppressed_count == sum(off) - sum(on)


# ---------------------------------------------------------------------------
# R4  Subset + same first-fire tick
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_r4_latched_fires_subset_of_unlatched(seed):
    rng = random.Random(100 + seed)
    inj = sorted(rng.sample(range(10, 1900), 12))
    norms = _heal_stream(inj, 2000, noise=0.004, seed=seed)
    off = _run(SufferingDerivativeComparator(W, THR, MIN0), norms)
    on = _run(SufferingDerivativeComparator(W, THR, MIN0, latch_enabled=True), norms)
    assert sum(on) > 0
    assert all(o for o, n in zip(off, on) if n), "latched fired where unlatched did not"
    assert on.index(True) == off.index(True)


# ---------------------------------------------------------------------------
# R5  Re-arm threshold
# ---------------------------------------------------------------------------

def test_r5_rearm_requires_rise():
    down = [1.0 - 0.02 * i for i in range(20)]            # 1.00 -> 0.62
    small_bump = [down[-1] + 0.03] + [down[-1] + 0.03 - 0.02 * i for i in range(1, 15)]
    comp = SufferingDerivativeComparator(5, 0.05, 0.01, latch_enabled=True, rearm_rise=0.05)
    fires = _run(comp, down + small_bump)
    assert sum(fires) == 1, "bump of 0.03 < rearm_rise 0.05 must not re-arm"

    big_bump = [down[-1] + 0.30] + [down[-1] + 0.30 - 0.02 * i for i in range(1, 15)]
    comp = SufferingDerivativeComparator(5, 0.05, 0.01, latch_enabled=True, rearm_rise=0.05)
    fires = _run(comp, down + big_bump)
    assert sum(fires) == 2
    # re-arm restarts the window: no fire until a full fresh window has elapsed
    second = [i for i, f in enumerate(fires) if f][1]
    assert second >= len(down) + comp.window_length - 1


# ---------------------------------------------------------------------------
# R6  Simulation gating
# ---------------------------------------------------------------------------

def test_r6_simulation_mode_is_inert():
    comp = SufferingDerivativeComparator(5, 0.05, 0.01, latch_enabled=True)
    _run(comp, [1.0 - 0.05 * i for i in range(8)])
    assert comp.latched
    buf, trough = list(comp._norm_buffer), comp._trough
    for n in (5.0, 0.0, 5.0):
        assert comp.tick(n, simulation_mode=True) is False
    assert comp.latched and comp._norm_buffer == buf and comp._trough == trough


# ---------------------------------------------------------------------------
# R7  reset + validation
# ---------------------------------------------------------------------------

def test_r7_reset_clears_latch_and_counters():
    comp = SufferingDerivativeComparator(5, 0.05, 0.01, latch_enabled=True)
    _run(comp, [1.0 - 0.05 * i for i in range(12)])
    assert comp.latched and comp.fire_count == 1 and comp.suppressed_count > 0
    comp.reset()
    assert not comp.latched and comp.fire_count == 0 and comp.suppressed_count == 0
    assert comp._norm_buffer == []


def test_r7_negative_rearm_rise_raises():
    with pytest.raises(ValueError):
        SufferingDerivativeComparator(latch_enabled=True, rearm_rise=-0.1)
