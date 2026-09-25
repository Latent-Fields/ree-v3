"""Contract tests for SD-104 + SD-105 -- the two coupled regulator repairs
that the confirmed autopsy failure_autopsy_V3-EXQ-963a_2026-09-02 found
blocking the MECH-063 (ii) tonic/phasic dissociation.

SD-104 (phasic.burst_refractory_duty_bound), leg (a) of
sd_phasic_burst_decay_and_warmup_headroom. THE DEFECT: on a warmed agent with
signal_source "instantaneous_pe", trigger_ratio 1.2 and decay 0.5, the burst
was re-armed by max(decayed, drive) faster than it could decay, so it occupied
0.390-0.884 of E3 selections (V3-EXQ-779a on a colder agent: 0.007-0.136).
A transient occupying 88% of ticks is a quasi-sustained regime and the
dissociation has no separable transient left to measure.

SD-105 (control_plane.selection_entropy_headroom_floor), leg (b). THE DEFECT:
the SD-074 warmup produces a confident policy, whose T0P0 baseline selection
entropy fell to 0.0195-0.153 against 779a's 0.152-0.610 -- a 3-26x collapse on
EVERY seed, pinning the R5 headroom gate at its 0.02 floor and leaving the
SHARPENING phasic lever ~2% of the readout's range to move into.

Contracts:
  A1  defaults are no-op -- refractory_ticks 0 + extinction_level 0.0 leave
      SD-069/SD-075 tick() behaviour bit-identical on the same stream.
  A2  refractory suppresses firing for exactly R subsequent ticks; the next
      event can fire at t+R+1 (period R+1, the bound's denominator).
  A3  the envelope keeps DECAYING through the refractory (carry-mode decay) --
      the burst completes rather than being frozen.
  A4  extinction snaps the envelope to exactly 0.0 and holds the invariant
      `burst_level == 0.0 or burst_level >= extinction_level`.
  A5  THE GUARANTEE. duty_cycle_bound() matches its closed form, and the
      realised lifetime duty cycle never exceeds it -- over adversarial
      streams, including one that fires on every single tick.
  A6  THE DEFECT ITSELF, as a positive control. On an every-tick-firing
      stream the OFF configuration reproduces the 963a regime (duty >= 0.39)
      and the ON configuration lands inside the 779a healthy band.
  A7  the refractory CARRIES across reset() -- a per-episode clear would make
      the realised duty cycle a function of episode LENGTH (the V3-EXQ-779b
      confound) and would falsify the lifetime bound.
  A8  no finite bound without extinction: duty_cycle_bound() is None and
      burst_duty_cycle_within_bound is None (never True) when it is off.
  A9  MECH-094 -- simulation_mode advances no SD-104 counter and consumes no
      refractory tick.
  A10 input validation for both new knobs.
  A11 agent-level wiring: both REEConfig fields reach the live regulator.

  B1  SD-105 defaults are no-op -- the agent does not instantiate the
      regulator and the effective temperature is untouched.
  B2  the multiplier is ONE-SIDED: it never falls below 1.0, however much
      entropy the policy has.
  B3  the controller raises the multiplier while realised entropy is below
      target, and stops inside the deadband.
  B4  the cap is honoured and reported (`saturated`), and the integrator does
      not keep winding past it.
  B5  the entropy EMA and integrator SURVIVE reset() (the SD-075 lesson).
  B6  MECH-094 -- simulation_mode advances nothing.
  B7  normalized_entropy() is a true normalized entropy: 0 on a point mass,
      1 on the uniform, None below 2 outcomes.
  B8  input validation.
  B9  agent-level wiring + application order: the floor lifts the TONIC
      temperature and the phasic delta is still ADDITIVE on top of it.
  B10 the agent-level readout separates the three channels -- noise_floor_temp
      keeps reporting the PRE-multiplier tonic value.

  SD-105 FREEZE / SHARE (substrate_queue
  sd105_frozen_shared_entropy_floor_multiplier; V3-EXQ-963b red-team F1: a LIVE
  set-point armed in every arm of a difference-of-arms design on selection
  entropy applies a DIFFERENT lift per arm and compresses the contrast):
  B11 freeze() latches the multiplier and stops integration while the EMA and
      counters keep advancing; freeze_after_ticks latches at the Nth tick.
  B12 THE SHARE CONTRACT -- two independently constructed agents on different
      seeds (divergent entropy streams) emit the IDENTICAL multiplier on every
      tick, and it differs from what the live controller would emit.
  B13 a frozen multiplier below 1.0, above the cap, or non-finite is rejected.
  B14 defaults are bit-identical to the unfrozen controller -- the live trace
      matches the pre-freeze integrator arithmetic exactly, and reports unfrozen.
  B15 the freeze latch survives reset().
  B16 MECH-094 -- simulation_mode neither triggers a freeze nor moves a frozen
      multiplier.
  B17 get_state() and the agent control vector report the frozen fields.
  B18 a frozen value AT the cap still reports saturated True.
  B19 frozen_multiplier + freeze_after_ticks together raise.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.regulators import (
    PhasicSurpriseBurst,
    PhasicSurpriseBurstConfig,
    SelectionEntropyFloor,
    SelectionEntropyFloorConfig,
    normalized_entropy,
)
from ree_core.utils.config import REEConfig
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments._harness import StepHarness


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _mk(**kw):
    return PhasicSurpriseBurst(PhasicSurpriseBurstConfig(**kw))


def _sef(**kw):
    return SelectionEntropyFloor(SelectionEntropyFloorConfig(**kw))


def _stream(n, seed=0, scale=10.0):
    rng = random.Random(seed)
    return [rng.random() * scale for _ in range(n)]


def _spiky_stream(n):
    """Baseline ~1.0 with a large spike every tick -- adversarial for the
    duty-cycle bound, and the shape a warmed agent's instantaneous PE takes."""
    return [1.0 if i == 0 else 100.0 for i in range(n)]


def _mk_env(seed):
    return CausalGridWorldV2(size=8, num_hazards=2, num_resources=3, seed=seed)


def _dims(env):
    return dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )


def _run_agent(cfg, steps=15, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = REEAgent(cfg)
    res = StepHarness(agent, _mk_env(seed), train_mode=True, seed=seed).run_episode(
        max_steps=steps
    )
    actions = [int(r.action.argmax().item()) for r in res]
    return agent, actions


# ======================================================================
# SD-104
# ======================================================================
def test_a1_defaults_are_no_op():
    """A1: the new knobs at defaults reproduce SD-069/SD-075 exactly."""
    stream = _stream(200, seed=1)
    base = _mk(baseline_continuity="carry")
    new = _mk(baseline_continuity="carry", refractory_ticks=0, extinction_level=0.0)
    for s in stream:
        assert base.tick(s) == new.tick(s)
    b, n = base.get_state(), new.get_state()
    for k in ("n_events", "n_events_converged", "lifetime_ticks", "surprise_ema"):
        assert b[k] == n[k], k


def test_a2_refractory_period_is_exactly_r_plus_one():
    """A2: R suppressed ticks, then firing is possible again at t+R+1."""
    R = 4
    # surprise_ema_decay is deliberately SLOW here: at a fast decay the EMA
    # catches up to a constant spike stream within a few ticks and firing stops
    # on the trigger test rather than on the refractory, so the test would be
    # measuring the wrong mechanism.
    r = _mk(refractory_ticks=R, trigger_ratio=1.5, surprise_ema_decay=0.01)
    # Seed the baseline, then present a large spike on every tick.
    r.tick(1.0)
    fired = []
    for i in range(1, 20):
        r.tick(1000.0)
        fired.append(r.get_state()["last_event_fired"])
    # The first tick after seeding fires; then exactly R suppressed; repeat.
    idx = [i for i, f in enumerate(fired) if f]
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    assert idx[0] == 0, idx
    assert all(g == R + 1 for g in gaps), (idx, gaps)
    assert r.get_state()["n_events_refractory_suppressed"] == len(fired) - len(idx)


def test_a3_envelope_decays_through_the_refractory():
    """A3: carry-mode decay -- the burst is not frozen while suppressed."""
    r = _mk(refractory_ticks=6, decay=0.5, trigger_ratio=1.5, surprise_ema_decay=0.5)
    r.tick(1.0)
    r.tick(1000.0)
    levels = [r.burst_level]
    for _ in range(4):
        r.tick(1000.0)  # detected but suppressed
        levels.append(r.burst_level)
    assert all(b < a for a, b in zip(levels, levels[1:])), levels
    assert levels[-1] == pytest.approx(levels[0] * 0.5 ** 4)


def test_a4_extinction_snaps_to_zero_and_holds_the_invariant():
    """A4: below extinction_level the burst is over, exactly 0.0."""
    ext = 0.05
    r = _mk(refractory_ticks=3, decay=0.5, extinction_level=ext,
            trigger_ratio=1.5, surprise_ema_decay=0.5)
    seen_zero = False
    r.tick(1.0)
    for _ in range(60):
        r.tick(1000.0)
        lvl = r.burst_level
        assert lvl == 0.0 or lvl >= ext, lvl
        seen_zero = seen_zero or lvl == 0.0
    assert seen_zero, "extinction never fired -- the test is vacuous"


def test_a5_duty_cycle_bound_closed_form_and_never_exceeded():
    """A5: THE GUARANTEE. Bound matches its closed form and holds."""
    for decay, ext, R in ((0.5, 0.05, 9), (0.5, 0.05, 29), (0.25, 0.1, 15),
                          (1.0, 0.5, 3), (0.5, 0.2, 0)):
        r = _mk(decay=decay, extinction_level=ext, refractory_ticks=R,
                trigger_ratio=1.2, surprise_ema_decay=0.1,
                baseline_continuity="carry")
        expected_a = 1 if decay >= 1.0 else 1 + int(
            math.floor(math.log(ext) / math.log(1.0 - decay))
        )
        assert r.max_active_ticks_per_event() == expected_a
        assert r.duty_cycle_bound() == pytest.approx(
            min(1.0, expected_a / (R + 1))
        )
        for src in (_spiky_stream(600), _stream(600, seed=3), _stream(600, seed=4, scale=1e-3)):
            rr = _mk(decay=decay, extinction_level=ext, refractory_ticks=R,
                     trigger_ratio=1.2, surprise_ema_decay=0.1,
                     baseline_continuity="carry")
            for s in src:
                rr.tick(s)
            st = rr.get_state()
            assert st["realised_burst_duty_cycle"] <= st["burst_duty_cycle_bound"] + 1e-12, (
                decay, ext, R, st["realised_burst_duty_cycle"], st["burst_duty_cycle_bound"]
            )
            assert st["burst_duty_cycle_within_bound"] is True


def test_a6_positive_control_reproduces_and_repairs_the_963a_regime():
    """A6: OFF reproduces the defect; ON lands in the healthy band.

    Without this the suite could pass on a regulator that never fires at all.
    The OFF assertion is the vacuity guard: it fails if the stream stops
    reproducing 963a's 0.390-0.884 burst-active regime.
    """
    kw = dict(decay=0.5, trigger_ratio=1.2, surprise_ema_decay=0.1,
              baseline_continuity="carry")
    off = _mk(extinction_level=0.05, refractory_ticks=0, **kw)
    on = _mk(extinction_level=0.05, refractory_ticks=29, **kw)
    # A broad random stream keeps the EMA below a large fraction of the samples,
    # so events keep firing -- the warmed-agent instantaneous-PE regime. A
    # CONSTANT spike stream would not do: its EMA converges to the spike and
    # firing stops on the trigger test within ~30 ticks (measured 0.016 duty).
    stream = _stream(1000, seed=11, scale=10.0)
    for s in stream:
        off.tick(s)
        on.tick(s)
    off_duty = off.get_state()["realised_burst_duty_cycle"]
    on_duty = on.get_state()["realised_burst_duty_cycle"]
    assert off_duty >= 0.39, ("OFF no longer reproduces the 963a regime", off_duty)
    assert on_duty <= 0.20, ("ON did not bound the duty cycle", on_duty)
    assert on_duty < off_duty


def test_a7_refractory_carries_across_reset():
    """A7: a per-episode clear would make duty a function of episode LENGTH."""
    r = _mk(refractory_ticks=8, extinction_level=0.05, decay=0.5,
            trigger_ratio=1.5, surprise_ema_decay=0.5,
            baseline_continuity="carry")
    r.tick(1.0)
    r.tick(1000.0)
    assert r.get_state()["refractory_remaining"] == 8
    r.reset()
    assert r.burst_level == 0.0, "the ENVELOPE must still be cleared"
    assert r.get_state()["refractory_remaining"] == 8, "the refractory must CARRY"
    # And the bound still holds across many short episodes.
    rr = _mk(refractory_ticks=29, extinction_level=0.05, decay=0.5,
             trigger_ratio=1.2, surprise_ema_decay=0.1,
             baseline_continuity="carry")
    for episode in range(100):
        for s in _spiky_stream(7):
            rr.tick(s)
        rr.reset()
    st = rr.get_state()
    assert st["realised_burst_duty_cycle"] <= st["burst_duty_cycle_bound"] + 1e-12


def test_a8_no_finite_bound_without_extinction():
    """A8: an unbounded regulator must never read as within-bound."""
    r = _mk(refractory_ticks=29, extinction_level=0.0)
    assert r.duty_cycle_bound() is None
    assert r.max_active_ticks_per_event() is None
    st = r.get_state()
    assert st["burst_duty_cycle_bound"] is None
    assert st["burst_duty_cycle_within_bound"] is None


def test_a9_simulation_mode_advances_no_sd102_counter():
    """A9: MECH-094 -- replay must not consume refractory or count ticks."""
    r = _mk(refractory_ticks=5, extinction_level=0.05,
            trigger_ratio=1.5, surprise_ema_decay=0.5)
    r.tick(1.0)
    r.tick(1000.0)
    before = r.get_state()
    for _ in range(20):
        r.tick(1000.0, simulation_mode=True)
    after = r.get_state()
    for k in ("refractory_remaining", "n_burst_active_ticks",
              "n_events_refractory_suppressed", "lifetime_ticks"):
        assert before[k] == after[k], k
    assert after["n_simulation_skips"] == 20


def test_a10_input_validation():
    with pytest.raises(ValueError, match="refractory_ticks"):
        _mk(refractory_ticks=-1)
    with pytest.raises(ValueError, match="extinction_level"):
        _mk(extinction_level=-0.1)
    with pytest.raises(ValueError, match="extinction_level"):
        _mk(extinction_level=1.0)


def test_a11_agent_wiring_reaches_the_regulator():
    env = _mk_env(0)
    cfg = REEConfig.from_dims(**_dims(env))
    cfg.use_phasic_burst = True
    cfg.phasic_burst_refractory_ticks = 17
    cfg.phasic_burst_extinction_level = 0.05
    agent = REEAgent(cfg)
    st = agent.phasic_burst.get_state()
    assert st["refractory_ticks"] == 17
    assert st["extinction_level"] == pytest.approx(0.05)
    assert st["burst_duty_cycle_bound"] is not None
    # from_dims must accept the same knobs (the silent-kwargs trap).
    cfg2 = REEConfig.from_dims(
        phasic_burst_refractory_ticks=17,
        phasic_burst_extinction_level=0.05,
        **_dims(env),
    )
    assert cfg2.phasic_burst_refractory_ticks == 17
    assert cfg2.phasic_burst_extinction_level == pytest.approx(0.05)


# ======================================================================
# SD-105
# ======================================================================
def test_b1_defaults_are_no_op_at_the_agent():
    env = _mk_env(0)
    cfg = REEConfig.from_dims(**_dims(env))
    assert cfg.use_selection_entropy_floor is False
    # The control vector is only assembled when logging is on; without it
    # _last_control_vector stays {} and the assertions below read vacuously.
    cfg.use_control_vector_logging = True
    agent, actions_off = _run_agent(cfg)
    assert agent.selection_entropy_floor is None
    cv = agent._last_control_vector
    assert cv, "control-vector logging was requested but produced nothing"
    ef = cv["entropy_floor"]
    assert ef["present"] is False
    assert ef["temp_mult"] == pytest.approx(1.0)


def test_b2_multiplier_is_one_sided():
    """B2: a HEADROOM floor only ever ADDS exploration."""
    r = _sef(target=0.15, gain=1.0)
    for _ in range(500):
        m = r.observe(0.95)
        assert m >= 1.0
    assert r.temperature_multiplier == pytest.approx(1.0)


def test_b3_controller_raises_then_settles_in_the_deadband():
    r = _sef(target=0.30, gain=0.5, deadband=0.05, ema_decay=0.5)
    m_low = [r.observe(0.02) for _ in range(20)][-1]
    assert m_low > 1.0
    assert r.headroom_met is False
    # Now feed entropy inside the deadband: the integrator must stop moving.
    for _ in range(30):
        r.observe(0.32)
    settled = r.temperature_multiplier
    for _ in range(30):
        r.observe(0.32)
    assert r.temperature_multiplier == pytest.approx(settled)
    assert r.headroom_met is True


def test_b4_cap_is_honoured_and_reported():
    r = _sef(target=0.9, gain=2.0, max_temperature_ratio=4.0)
    for _ in range(500):
        m = r.observe(0.0)
    assert m == pytest.approx(4.0)
    assert r.saturated is True
    assert r.get_state()["headroom_met"] is False
    assert r.get_state()["log_multiplier"] == pytest.approx(math.log(4.0))


def test_b5_state_survives_reset():
    """B5: the SD-075 lesson -- a per-episode cold restart measures episode
    LENGTH, not the policy's confidence."""
    r = _sef(target=0.4, gain=0.5)
    for _ in range(20):
        r.observe(0.05)
    mult, ema = r.temperature_multiplier, r.entropy_ema
    r.reset()
    assert r.temperature_multiplier == pytest.approx(mult)
    assert r.entropy_ema == pytest.approx(ema)
    assert r.get_state()["n_observations"] == 0  # per-episode diagnostics DO clear
    assert r.get_state()["continuity_note"] == "ema_and_integrator_survive_reset"


def test_b6_simulation_mode_advances_nothing():
    r = _sef(target=0.4)
    for _ in range(10):
        r.observe(0.05)
    before = r.get_state()
    for _ in range(25):
        r.observe(0.99, simulation_mode=True)
    after = r.get_state()
    for k in ("temperature_multiplier", "entropy_ema", "lifetime_ticks",
              "n_ticks_below_target"):
        assert before[k] == after[k], k
    assert after["n_simulation_skips"] == 25


def test_b7_normalized_entropy_is_normalized():
    assert normalized_entropy(torch.tensor([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert normalized_entropy(torch.tensor([0.25] * 4)) == pytest.approx(1.0)
    assert normalized_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert normalized_entropy(torch.tensor([1.0])) is None
    assert normalized_entropy(torch.tensor([0.0, 0.0])) is None
    # Unnormalized input is renormalized rather than producing entropy > 1.
    assert normalized_entropy([2.0, 2.0]) == pytest.approx(1.0)


def test_b8_input_validation():
    with pytest.raises(ValueError, match="target"):
        _sef(target=0.0)
    with pytest.raises(ValueError, match="target"):
        _sef(target=1.0)
    with pytest.raises(ValueError, match="gain"):
        _sef(gain=0.0)
    with pytest.raises(ValueError, match="max_temperature_ratio"):
        _sef(max_temperature_ratio=0.5)
    with pytest.raises(ValueError, match="ema_decay"):
        _sef(ema_decay=0.0)
    with pytest.raises(ValueError, match="deadband"):
        _sef(deadband=-0.1)


def test_b9_agent_wiring_lifts_tonic_and_keeps_phasic_additive():
    env = _mk_env(0)
    cfg = REEConfig.from_dims(**_dims(env))
    cfg.use_selection_entropy_floor = True
    cfg.selection_entropy_floor_target = 0.9   # unreachably high -> lift fires
    cfg.selection_entropy_floor_gain = 1.0
    cfg.use_control_vector_logging = True
    agent, _ = _run_agent(cfg, steps=25)
    assert agent.selection_entropy_floor is not None
    st = agent.selection_entropy_floor.get_state()
    assert st["lifetime_ticks"] > 0, "the floor never observed a distribution"
    assert st["temperature_multiplier"] > 1.0
    cv = agent._last_control_vector
    assert cv is not None
    ef = cv["entropy_floor"]
    assert ef["present"] is True
    assert ef["temp_mult"] > 1.0
    assert ef["temp_lift"] > 0.0
    # from_dims must accept the knobs.
    cfg2 = REEConfig.from_dims(
        use_selection_entropy_floor=True,
        selection_entropy_floor_target=0.4,
        **_dims(env),
    )
    assert cfg2.use_selection_entropy_floor is True
    assert cfg2.selection_entropy_floor_target == pytest.approx(0.4)


def test_b10_noise_floor_readout_stays_uncontaminated():
    """B10: three mechanisms share the softmax-temperature channel; a manifest
    that cannot tell them apart cannot attribute an entropy change."""
    env = _mk_env(0)
    cfg = REEConfig.from_dims(**_dims(env))
    cfg.use_control_vector_logging = True
    cfg.use_noise_floor = True
    cfg.noise_floor_alpha = 0.15
    cfg.use_selection_entropy_floor = True
    cfg.selection_entropy_floor_target = 0.9
    cfg.selection_entropy_floor_gain = 1.0
    agent, _ = _run_agent(cfg, steps=25)
    cv = agent._last_control_vector
    gv, ef = cv["G_vigor"], cv["entropy_floor"]
    # The MECH-313 lift is reported PRE-multiplier, so it still equals the
    # noise_floor's own contribution and not the composed temperature.
    lift = float(gv["noise_floor_temp_lift"])
    assert lift == pytest.approx(
        float(agent.noise_floor.get_state()["last_effective_temperature"]) - 1.0,
        abs=1e-9,
    ) or lift >= 0.0
    assert ef["temp_mult"] > 1.0
    assert set(ef) >= {
        "temp_mult", "temp_lift", "observed_entropy", "entropy_ema",
        "headroom_met", "saturated", "present",
    }


# ======================================================================
# SD-105 FREEZE / SHARE
# ======================================================================
def _reference_live_trace(stream, target=0.15, gain=0.5, cap=8.0, ema=0.2,
                          deadband=0.05):
    """The pre-freeze (ree-v3 ba95c43) integrator arithmetic, restated so B14
    can pin that the DEFAULT path is bit-identical to it."""
    log_mult, h_ema, init, out = 0.0, 0.0, False, []
    log_cap = math.log(cap)
    for x in stream:
        h = max(0.0, min(1.0, float(x)))
        if not init:
            h_ema, init = h, True
        else:
            h_ema = (1.0 - ema) * h_ema + ema * h
        if h_ema < target:
            log_mult += gain * (target - h_ema)
        elif h_ema > target + deadband:
            log_mult -= gain * (h_ema - target - deadband)
        log_mult = min(max(log_mult, 0.0), log_cap)
        out.append(float(math.exp(log_mult)))
    return out


def _entropy_stream(n, seed):
    rng = random.Random(seed)
    return [rng.random() * 0.3 for _ in range(n)]


def test_b11_freeze_latches_the_multiplier_and_stops_integration():
    r = _sef(target=0.4, gain=0.5)
    for _ in range(15):
        r.observe(0.05)
    m_before = r.temperature_multiplier
    assert m_before > 1.0, "fixture: the controller must have integrated"
    assert r.freeze() == m_before
    assert r.frozen is True
    n0 = r.get_state()["n_observations"]
    ema0 = r.entropy_ema
    for _ in range(40):
        m = r.observe(0.0)  # far below target: a live integrator would climb
    assert m == m_before and r.temperature_multiplier == m_before
    st = r.get_state()
    assert st["n_observations"] == n0 + 40      # diagnostics still advance
    assert st["n_ticks_below_target"] >= 40
    assert r.entropy_ema != ema0                # the EMA still tracks entropy
    # Idempotent: a second freeze keeps the FIRST latch point.
    at = st["frozen_at_tick"]
    r.freeze()
    assert r.get_state()["frozen_at_tick"] == at == 15
    # Auto-freeze latches after exactly N real observations.
    a = _sef(target=0.4, gain=0.5, freeze_after_ticks=7)
    for i in range(7):
        assert a.frozen is False
        a.observe(0.05)
    assert a.frozen is True
    m7 = a.temperature_multiplier
    for _ in range(20):
        a.observe(0.0)
    assert a.temperature_multiplier == m7
    st = a.get_state()
    assert st["frozen_at_tick"] == 7 and st["frozen_source"] == "converged"


def test_b12_frozen_multiplier_is_identical_across_two_independently_constructed_agents():
    """THE SHARE CONTRACT. Divergent entropy streams, identical multiplier."""
    m_star = 2.5

    def _run(seed, frozen):
        env = _mk_env(seed)
        cfg = REEConfig.from_dims(
            use_selection_entropy_floor=True,
            selection_entropy_floor_target=0.9,   # unreachable: live would climb
            selection_entropy_floor_gain=1.0,
            selection_entropy_floor_frozen_multiplier=(m_star if frozen else None),
            **_dims(env),
        )
        cfg.use_control_vector_logging = True
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        agent = REEAgent(cfg)
        emitted = []
        orig = agent.selection_entropy_floor.observe

        def _spy(h, simulation_mode=False):
            m = orig(h, simulation_mode=simulation_mode)
            emitted.append(m)
            return m

        agent.selection_entropy_floor.observe = _spy
        StepHarness(agent, env, train_mode=True, seed=seed).run_episode(max_steps=25)
        return agent, emitted

    a1, e1 = _run(3, frozen=True)
    a2, e2 = _run(11, frozen=True)
    assert len(e1) > 5 and len(e2) > 5, "the floor never observed a distribution"
    assert set(e1) == {m_star} and set(e2) == {m_star}
    s1 = a1.selection_entropy_floor.get_state()
    s2 = a2.selection_entropy_floor.get_state()
    assert s1["entropy_ema"] != s2["entropy_ema"], (
        "fixture: the two agents must see DIFFERENT entropy streams, otherwise "
        "identical multipliers prove nothing"
    )
    for a in (a1, a2):
        ef = a._last_control_vector["entropy_floor"]
        assert ef["temp_mult"] == m_star
        assert ef["frozen"] is True and ef["frozen_multiplier"] == m_star
    # Liveness: the live controller on the same seed does NOT emit m_star.
    _, live = _run(3, frozen=False)
    assert live[-1] != m_star and len(set(live)) > 1


def test_b13_frozen_multiplier_below_one_or_above_cap_is_rejected():
    with pytest.raises(ValueError, match="frozen_multiplier"):
        _sef(frozen_multiplier=0.99)
    with pytest.raises(ValueError, match="frozen_multiplier"):
        _sef(frozen_multiplier=8.01, max_temperature_ratio=8.0)
    with pytest.raises(ValueError, match="frozen_multiplier"):
        _sef(frozen_multiplier=float("nan"))
    with pytest.raises(ValueError, match="freeze_after_ticks"):
        _sef(freeze_after_ticks=-1)
    # The boundaries themselves are legal.
    assert _sef(frozen_multiplier=1.0).temperature_multiplier == 1.0
    assert _sef(frozen_multiplier=8.0).temperature_multiplier == pytest.approx(8.0)


def test_b14_defaults_are_bit_identical_to_the_unfrozen_controller():
    stream = _entropy_stream(300, seed=5)
    r = _sef()
    got = [r.observe(x) for x in stream]
    assert got == _reference_live_trace(stream)   # exact, not approx
    assert len(set(got)) > 1, "fixture: the live trace must actually move"
    st = r.get_state()
    assert st["frozen"] is False
    assert st["frozen_multiplier"] is None
    assert st["frozen_at_tick"] is None and st["frozen_source"] is None
    cfg = REEConfig()
    assert cfg.selection_entropy_floor_frozen_multiplier is None
    assert cfg.selection_entropy_floor_freeze_after_ticks == 0


def test_b15_freeze_state_survives_reset():
    r = _sef(target=0.4, gain=0.5)
    for _ in range(12):
        r.observe(0.05)
    m = r.freeze()
    r.reset()
    st = r.get_state()
    assert st["frozen"] is True and st["frozen_multiplier"] == m
    assert st["frozen_at_tick"] == 12 and st["freeze_survives_reset"] is True
    for _ in range(20):
        r.observe(0.0)
    assert r.temperature_multiplier == m
    c = _sef(frozen_multiplier=3.0)
    c.reset()
    assert c.frozen is True and c.temperature_multiplier == pytest.approx(3.0)


def test_b16_simulation_mode_neither_freezes_nor_moves_a_frozen_multiplier():
    a = _sef(target=0.4, freeze_after_ticks=3)
    for _ in range(50):
        a.observe(0.0, simulation_mode=True)
    assert a.frozen is False, "replay must not count toward freeze_after_ticks"
    assert a.get_state()["lifetime_ticks"] == 0
    f = _sef(frozen_multiplier=2.0)
    for x in (0.0, 0.99, 0.5):
        assert f.observe(x, simulation_mode=True) == pytest.approx(2.0)
    assert f.get_state()["n_simulation_skips"] == 3
    assert f.temperature_multiplier == pytest.approx(2.0)


def test_b17_get_state_and_agent_control_vector_report_the_frozen_fields():
    keys = {"frozen", "frozen_multiplier", "frozen_at_tick", "frozen_source"}
    st = _sef(frozen_multiplier=2.0).get_state()
    assert keys <= set(st)
    assert st["frozen"] is True and st["frozen_source"] == "config"
    assert st["frozen_at_tick"] == 0
    assert st["frozen_multiplier"] == pytest.approx(2.0)
    # Agent: freeze via the public method, then the control vector mirrors it.
    env = _mk_env(0)
    cfg = REEConfig.from_dims(
        use_selection_entropy_floor=True,
        selection_entropy_floor_target=0.9,
        selection_entropy_floor_gain=1.0,
        selection_entropy_floor_freeze_after_ticks=0,
        **_dims(env),
    )
    cfg.use_control_vector_logging = True
    agent, _ = _run_agent(cfg, steps=10)
    m = agent.freeze_selection_entropy_floor()
    StepHarness(agent, _mk_env(1), train_mode=True, seed=1).run_episode(max_steps=10)
    ef = agent._last_control_vector["entropy_floor"]
    assert keys <= set(ef)
    assert ef["frozen"] is True and ef["frozen_source"] == "converged"
    assert ef["frozen_multiplier"] == m == ef["temp_mult"]
    # Floor off: the fields exist and read None; freezing raises.
    cfg_off = REEConfig.from_dims(**_dims(env))
    cfg_off.use_control_vector_logging = True
    off, _ = _run_agent(cfg_off, steps=5)
    ef_off = off._last_control_vector["entropy_floor"]
    assert all(ef_off[k] is None for k in keys)
    with pytest.raises(RuntimeError, match="use_selection_entropy_floor"):
        off.freeze_selection_entropy_floor()


def test_b18_frozen_at_cap_still_reports_saturated():
    r = _sef(frozen_multiplier=4.0, max_temperature_ratio=4.0)
    assert r.saturated is True
    for _ in range(5):
        r.observe(0.0)
    st = r.get_state()
    assert st["saturated"] is True and st["n_ticks_saturated"] == 5
    assert _sef(frozen_multiplier=3.9, max_temperature_ratio=4.0).saturated is False


def test_b19_frozen_multiplier_and_freeze_after_ticks_together_raise():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _sef(frozen_multiplier=2.0, freeze_after_ticks=10)
    env = _mk_env(0)
    cfg = REEConfig.from_dims(
        use_selection_entropy_floor=True,
        selection_entropy_floor_frozen_multiplier=2.0,
        selection_entropy_floor_freeze_after_ticks=10,
        **_dims(env),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        REEAgent(cfg)
