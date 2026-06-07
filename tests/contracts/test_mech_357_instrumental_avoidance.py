"""Contract tests for SD-058 / MECH-357 instrumental-avoidance acquisition.

Interface-level guarantees that should hold regardless of tuning:
  C1  default OFF -> agent.instrumental_avoidance is None; bit-identical action
      stream (the gate uses no RNG and adds zero bias at zero efficacy/floor).
  C2  eligibility-trace learning: directed action under threat that drops
      z_harm_a credits efficacy; freezing / failed avoidance under threat decays
      it; below threat -> no change.
  C3  action bias: penalise the no-op / freeze class under threat proportional
      to effective_efficacy * threat_scale; zero below threat / at zero efficacy;
      clamped to bias_scale.
  C4  freeze-suppression: should_suppress_freeze True iff eff*threat >= threshold.
  C5  reset() PRESERVES learned efficacy (cross-episode developmental
      acquisition) but clears the within-episode threat trace.
  C6  MECH-094: update() and compute_action_bias() are no-ops under
      simulation_mode.
  C7  config no-op defaults + from_dims propagation; scaffold-floor anneal hook.
"""

import torch

from ree_core.pfc.infralimbic_avoidance_gate import (
    InstrumentalAvoidanceGate,
    InstrumentalAvoidanceGateConfig,
)
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2


def _build(env, **kw):
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        **kw,
    )


def _action_stream(kw, n=25):
    torch.manual_seed(321)
    env = CausalGridWorldV2(size=8, seed=11)
    ag = REEAgent(_build(env, **kw))
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
    ag = REEAgent(_build(env))
    assert ag.instrumental_avoidance is None
    # use_instrumental_avoidance=True at zero efficacy/floor must NOT perturb the
    # action stream (no RNG; zero bias under no threat).
    assert _action_stream({}) == _action_stream({"use_instrumental_avoidance": True})


# ---------------------------------------------------------------- C2
def test_c2_eligibility_trace_credit_and_decay():
    g = InstrumentalAvoidanceGate(
        InstrumentalAvoidanceGateConfig(threat_floor=0.1, learn_rate=0.2, leak_rate=0.2)
    )
    # below threat -> no learning
    g.update(0.05, True)
    assert g.avoidance_efficacy == 0.0
    # seed prev with an above-threat reading
    g.update(0.6, True)
    assert g.avoidance_efficacy == 0.0  # prev was below floor on this call
    # directed action under threat AND harm dropped (0.6 -> 0.2) -> credit
    g.update(0.2, True)
    credited = g.avoidance_efficacy
    assert credited > 0.0
    # directed under threat but harm did NOT drop (0.2 -> 0.6) -> decay
    g.update(0.6, True)
    assert g.avoidance_efficacy < credited
    # froze (no-op) under threat -> decay (freezing is not credited)
    before = g.avoidance_efficacy
    g.update(0.2, False)  # prev=0.6>floor, but action not directed -> decay
    assert g.avoidance_efficacy <= before


# ---------------------------------------------------------------- C3
def test_c3_action_bias_penalises_noop_under_threat():
    g = InstrumentalAvoidanceGate(
        InstrumentalAvoidanceGateConfig(
            scaffold_floor=0.5, threat_floor=0.1, threat_ref=0.5,
            action_bias_gain=0.2, bias_scale=0.1, noop_class=0,
        )
    )
    # under full threat: no-op class penalised (positive), directed classes 0
    b = g.compute_action_bias(0.5, [0, 1, 2, 3], noop_class=0)
    assert float(b[0]) > 0.0
    assert float(b[1]) == 0.0 and float(b[2]) == 0.0 and float(b[3]) == 0.0
    # clamp respected (float32 tolerance)
    assert float(b[0]) <= 0.1 + 1e-6
    # below threat -> zero everywhere
    assert float(g.compute_action_bias(0.0, [0, 1, 2, 3]).abs().sum()) == 0.0
    # zero effective efficacy -> zero bias even under threat
    g0 = InstrumentalAvoidanceGate(InstrumentalAvoidanceGateConfig(scaffold_floor=0.0))
    assert float(g0.compute_action_bias(0.6, [0, 1]).abs().sum()) == 0.0


# ---------------------------------------------------------------- C4
def test_c4_freeze_suppression_threshold():
    g = InstrumentalAvoidanceGate(
        InstrumentalAvoidanceGateConfig(
            scaffold_floor=0.8, threat_floor=0.1, threat_ref=0.5,
            suppression_threshold=0.5,
        )
    )
    # full threat * eff 0.8 = 0.8 >= 0.5 -> suppress
    assert g.should_suppress_freeze(0.5) is True
    # below threat -> no suppression
    assert g.should_suppress_freeze(0.0) is False
    # low effective efficacy -> no suppression even under threat
    g2 = InstrumentalAvoidanceGate(
        InstrumentalAvoidanceGateConfig(scaffold_floor=0.2, suppression_threshold=0.5)
    )
    assert g2.should_suppress_freeze(0.5) is False


# ---------------------------------------------------------------- C5
def test_c5_reset_preserves_learned_efficacy():
    g = InstrumentalAvoidanceGate(
        InstrumentalAvoidanceGateConfig(threat_floor=0.1, learn_rate=0.3)
    )
    g.update(0.6, True)
    g.update(0.1, True)  # credit
    learned = g.avoidance_efficacy
    assert learned > 0.0
    g.reset()
    # learned efficacy survives the episode boundary; threat trace cleared
    assert g.avoidance_efficacy == learned
    assert g._z_harm_a_prev is None
    assert g._last_action_directed is False


# ---------------------------------------------------------------- C6
def test_c6_mech094_simulation_no_op():
    g = InstrumentalAvoidanceGate(
        InstrumentalAvoidanceGateConfig(scaffold_floor=0.8, threat_floor=0.1)
    )
    g.update(0.6, True)
    g.update(0.1, True)
    e = g.avoidance_efficacy
    # simulation_mode update must not advance efficacy
    g.update(0.6, True, simulation_mode=True)
    assert g.avoidance_efficacy == e
    # simulation_mode bias must be zero
    assert float(g.compute_action_bias(0.6, [0, 1], simulation_mode=True).abs().sum()) == 0.0
    # simulation_mode suppression must be False
    assert g.should_suppress_freeze(0.6, simulation_mode=True) is False


# ---------------------------------------------------------------- C7
def test_c7_config_defaults_and_from_dims_propagation():
    env = CausalGridWorldV2(size=8, seed=0)
    # default OFF
    cfg_off = _build(env)
    assert cfg_off.use_instrumental_avoidance is False
    assert cfg_off.avoidance_scaffold_floor == 0.0
    assert cfg_off.avoidance_noop_class == 0
    # from_dims propagates the knobs onto the agent's gate config
    cfg_on = _build(
        env,
        use_instrumental_avoidance=True,
        avoidance_learn_rate=0.07,
        avoidance_scaffold_floor=0.6,
        avoidance_suppression_threshold=0.4,
    )
    ag = REEAgent(cfg_on)
    assert ag.instrumental_avoidance is not None
    c = ag.instrumental_avoidance.config
    assert abs(c.learn_rate - 0.07) < 1e-9
    assert abs(c.scaffold_floor - 0.6) < 1e-9
    assert abs(c.suppression_threshold - 0.4) < 1e-9
    # scaffold-floor anneal hook (curriculum driver)
    ag.instrumental_avoidance.set_scaffold_floor(0.3)
    assert abs(ag.instrumental_avoidance.config.scaffold_floor - 0.3) < 1e-9
    assert abs(ag.instrumental_avoidance.effective_efficacy() - 0.3) < 1e-9
