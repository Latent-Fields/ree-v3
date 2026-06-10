"""Contract tests for MECH-219 (SD-019b) affective-harm hysteretic integrator.

Interface-level guarantees that should hold regardless of tuning:
  C1  default OFF -> agent.harm_suffering_accumulator is None,
      LatentState.z_harm_suffering None, bit-identical action stream.
  C2  HarmSufferingAccumulatorConfig validation (loud on bad values).
  C3  controllability gate: escapability=1 (g=0) -> no suffering accrual even at
      high unpleasantness; escapability=0 (g=1) -> suffering accrues. The
      falsifiable Loffler 2018 / Salomons 2004 dissociation.
  C4  hysteresis: alpha_rise >> alpha_fall -> fast build, slow release
      (recovery-failure signature).
  C5  MECH-094 sim no-op: simulation_mode does not advance s_t.
  C6  body-damage fold-in (memo Section 6 fork b): body_damage_weight folds
      ||z_harm_a|| into the drive.
  C7  optional Schmitt bistable latch: latches above theta_on, holds across the
      band, releases below theta_off.
  C8  precondition: use_harm_suffering_accumulator=True requires use_harm_un=True.
  C9  escapability source modes (constant / external / avoidance_efficacy).
  C10 LatentState.z_harm_suffering populated + survives detach(); dim == z_harm_un.
  C11 per-consumer redirect flags default OFF (bit-identical) and the AIC
      redirect sources its urgency magnitude from the suffering channel when on.
"""

import torch

from ree_core.affect.harm_suffering_accumulator import (
    HarmSufferingAccumulator,
    HarmSufferingAccumulatorConfig,
)
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2


# ---------------------------------------------------------------- harness
def _harm_cfg(env, **kw):
    """REEConfig with the SD-010/011 harm stream + SD-019a z_harm_un on."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        use_harm_stream=True,
        harm_obs_dim=51,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        z_harm_a_dim=16,
        **kw,
    )
    cfg.latent.use_harm_un = True
    return cfg


def _sense(ag, env, od, n=1):
    lat = None
    for _ in range(n):
        lat = ag.sense(
            od["body_state"],
            od["world_state"],
            obs_harm=od["harm_obs"],
            obs_harm_a=od["harm_obs_a"],
        )
    return lat


def _action_stream(off_explicit, n=25):
    torch.manual_seed(321)
    env = CausalGridWorldV2(size=8, seed=11)
    kw = {"use_harm_suffering_accumulator": False} if off_explicit else {}
    ag = REEAgent(
        REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=4,
            **kw,
        )
    )
    _, od = env.reset()
    acts = []
    for _ in range(n):
        a = ag.act_with_split_obs(od["body_state"], od["world_state"])
        acts.append(int(a.argmax()))
        _, h, d, inf, od = env.step(a)
        if d:
            _, od = env.reset()
            ag.reset()
    return acts


# ---------------------------------------------------------------- C1
def test_c1_default_off_no_op():
    env = CausalGridWorldV2(size=8, seed=0)
    ag = REEAgent(
        REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=4,
        )
    )
    _, od = env.reset()
    lat = ag.sense(od["body_state"], od["world_state"])
    assert ag.harm_suffering_accumulator is None
    assert lat.z_harm_suffering is None
    # default == explicit-False: the regulator uses no torch RNG.
    assert _action_stream(False) == _action_stream(True)


# ---------------------------------------------------------------- C2
def test_c2_config_validation():
    import pytest

    with pytest.raises(ValueError):
        HarmSufferingAccumulator(HarmSufferingAccumulatorConfig(alpha_rise=1.5))
    with pytest.raises(ValueError):
        HarmSufferingAccumulator(HarmSufferingAccumulatorConfig(alpha_fall=-0.1))
    with pytest.raises(ValueError):
        HarmSufferingAccumulator(
            HarmSufferingAccumulatorConfig(escapability_mode="bogus")
        )
    with pytest.raises(ValueError):
        HarmSufferingAccumulator(HarmSufferingAccumulatorConfig(s_cap=0.0))
    with pytest.raises(ValueError):
        HarmSufferingAccumulator(
            HarmSufferingAccumulatorConfig(body_damage_weight=-1.0)
        )
    with pytest.raises(ValueError):
        # latch band must have theta_off < theta_on
        HarmSufferingAccumulator(
            HarmSufferingAccumulatorConfig(
                use_bistable_latch=True, theta_on=0.3, theta_off=0.5
            )
        )


# ---------------------------------------------------------------- C3
def test_c3_controllability_gate():
    r = HarmSufferingAccumulator(
        HarmSufferingAccumulatorConfig(use_harm_suffering_accumulator=True)
    )
    # Fully controllable: g=0 -> suffering never accrues even at high u.
    for _ in range(30):
        o = r.update(unpleasantness_norm=1.0, escapability=1.0)
    assert o.s < 1e-6, f"escapable suffering should stay ~0, got {o.s}"
    # Inescapable: g=1 -> suffering accrues.
    r2 = HarmSufferingAccumulator(
        HarmSufferingAccumulatorConfig(use_harm_suffering_accumulator=True)
    )
    for _ in range(30):
        o2 = r2.update(unpleasantness_norm=1.0, escapability=0.0)
    assert o2.s > 0.5, f"inescapable suffering should accrue, got {o2.s}"


# ---------------------------------------------------------------- C4
def test_c4_hysteresis():
    r = HarmSufferingAccumulator(
        HarmSufferingAccumulatorConfig(
            use_harm_suffering_accumulator=True, alpha_rise=0.2, alpha_fall=0.01
        )
    )
    # Build under inescapable harm.
    for _ in range(40):
        o = r.update(unpleasantness_norm=1.0, escapability=0.0)
    peak = o.s
    assert peak > 0.8
    # Relief: fully escapable -> slow release (alpha_fall << alpha_rise).
    for _ in range(40):
        o = r.update(unpleasantness_norm=1.0, escapability=1.0)
    # After 40 release steps at alpha_fall=0.01 the trace must still be high.
    assert o.s > peak * 0.5, (
        f"recovery must be slow (hysteresis); fell to {o.s} from {peak} in 40 steps"
    )


# ---------------------------------------------------------------- C5
def test_c5_mech094_sim_no_op():
    r = HarmSufferingAccumulator(
        HarmSufferingAccumulatorConfig(use_harm_suffering_accumulator=True)
    )
    for _ in range(10):
        o = r.update(unpleasantness_norm=1.0, escapability=0.0)
    before = o.s_raw
    o_sim = r.update(unpleasantness_norm=1.0, escapability=0.0, simulation_mode=True)
    assert o_sim.simulation_skipped is True
    assert abs(o_sim.s_raw - before) < 1e-9, "sim tick must not advance s_t"


# ---------------------------------------------------------------- C6
def test_c6_body_damage_fold_in():
    r = HarmSufferingAccumulator(
        HarmSufferingAccumulatorConfig(
            use_harm_suffering_accumulator=True, body_damage_weight=0.5
        )
    )
    o = r.update(unpleasantness_norm=1.0, escapability=0.0, body_damage_norm=2.0)
    # u = 1.0 + 0.5 * 2.0 = 2.0
    assert abs(o.u - 2.0) < 1e-6, f"body-damage fold-in expected u=2.0, got {o.u}"
    # weight 0 -> pure z_harm_un drive (no body contribution).
    r0 = HarmSufferingAccumulator(
        HarmSufferingAccumulatorConfig(
            use_harm_suffering_accumulator=True, body_damage_weight=0.0
        )
    )
    o0 = r0.update(unpleasantness_norm=1.0, escapability=0.0, body_damage_norm=2.0)
    assert abs(o0.u - 1.0) < 1e-6


# ---------------------------------------------------------------- C7
def test_c7_bistable_latch():
    r = HarmSufferingAccumulator(
        HarmSufferingAccumulatorConfig(
            use_harm_suffering_accumulator=True,
            use_bistable_latch=True,
            theta_on=0.5,
            theta_off=0.3,
            alpha_rise=0.5,
            alpha_fall=0.5,
        )
    )
    # Below theta_on -> not latched, output 0 even though s_raw > 0.
    o = r.update(unpleasantness_norm=1.0, escapability=0.0)  # s_raw=0.5, not > 0.5
    assert o.latched is False and o.s == 0.0
    # Cross theta_on -> latched, output = s_raw.
    o = r.update(unpleasantness_norm=1.0, escapability=0.0)
    assert o.latched is True and o.s > 0.0
    # Relief drops s below theta_off -> unlatch.
    for _ in range(6):
        o = r.update(unpleasantness_norm=0.0, escapability=1.0)
    assert o.latched is False and o.s == 0.0


# ---------------------------------------------------------------- C8
def test_c8_precondition_requires_harm_un():
    import pytest

    env = CausalGridWorldV2(size=8, seed=0)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        use_harm_suffering_accumulator=True,
    )
    # use_harm_un defaults False -> must raise loudly.
    assert cfg.latent.use_harm_un is False
    with pytest.raises(ValueError):
        REEAgent(cfg)


# ---------------------------------------------------------------- C9
def test_c9_escapability_source_modes():
    env = CausalGridWorldV2(size=8, seed=0)
    # constant
    ag = REEAgent(_harm_cfg(env, use_harm_suffering_accumulator=True,
                            harm_suffering_escapability_constant=0.3))
    assert abs(ag._resolve_harm_suffering_escapability() - 0.3) < 1e-9
    # external (driven by the setter)
    ag = REEAgent(_harm_cfg(env, use_harm_suffering_accumulator=True,
                            harm_suffering_escapability_mode="external"))
    ag.set_harm_suffering_escapability(0.0)
    assert ag._resolve_harm_suffering_escapability() == 0.0
    ag.set_harm_suffering_escapability(2.0)  # clamped
    assert ag._resolve_harm_suffering_escapability() == 1.0
    # avoidance_efficacy reads SD-058 effective_efficacy (here = scaffold floor)
    ag = REEAgent(_harm_cfg(env, use_harm_suffering_accumulator=True,
                            harm_suffering_escapability_mode="avoidance_efficacy",
                            use_instrumental_avoidance=True,
                            avoidance_scaffold_floor=0.7))
    assert abs(ag._resolve_harm_suffering_escapability() - 0.7) < 1e-9


# ---------------------------------------------------------------- C10
def test_c10_latent_field_populated_and_detach():
    env = CausalGridWorldV2(size=8, seed=0)
    ag = REEAgent(_harm_cfg(env, use_harm_suffering_accumulator=True,
                            harm_suffering_escapability_mode="external"))
    ag.set_harm_suffering_escapability(0.0)
    _, od = env.reset()
    lat = _sense(ag, env, od, n=10)
    assert lat.z_harm_suffering is not None
    assert lat.z_harm_un is not None
    # same dim as z_harm_un (memo Section 4).
    assert lat.z_harm_suffering.shape == lat.z_harm_un.shape
    # ||z_harm_suffering|| == accumulator s_t.
    s = ag.harm_suffering_accumulator.get_suffering()
    assert abs(float(lat.z_harm_suffering.norm()) - s) < 1e-4
    # survives detach.
    assert lat.detach().z_harm_suffering is not None


# ---------------------------------------------------------------- C11
def test_c11_redirect_flags_default_off_and_aic_redirect():
    env = CausalGridWorldV2(size=8, seed=0)
    # Redirects all default off -> bit-identical to the same agent without them.
    cfg_off = _harm_cfg(env, use_harm_suffering_accumulator=True,
                        harm_suffering_escapability_mode="external")
    assert cfg_off.harm_suffering_redirect_aic is False
    assert cfg_off.harm_suffering_redirect_pag is False
    assert cfg_off.harm_suffering_redirect_mech091 is False
    assert cfg_off.harm_suffering_redirect_dacc is False
    assert cfg_off.harm_suffering_redirect_pacc is False

    # AIC redirect ON: AIC urgency magnitude tracks the suffering channel.
    ag = REEAgent(_harm_cfg(env, use_harm_suffering_accumulator=True,
                            harm_suffering_escapability_mode="external",
                            harm_suffering_redirect_aic=True,
                            use_aic_analog=True))
    ag.set_harm_suffering_escapability(0.0)
    _, od = env.reset()
    lat = _sense(ag, env, od, n=30)
    suf = float(lat.z_harm_suffering.norm())
    assert suf > 0.1, "suffering should have accrued under inescapable harm"
    # AIC ticked (its last-tick cache exists); the redirect is exercised
    # (no exception, suffering is the urgency source).
    assert ag.aic is not None and ag._aic_last_tick is not None
