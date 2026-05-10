"""Contract tests for MECH-320 tonic_vigor_coupling_score_bias (ARC-066 child).

Ten contracts (C1-C10) covering substrate-readiness invariants:
  C1: default-off no-op. With use_tonic_vigor=False, agent.tonic_vigor is
      None and the select_action MECH-320 block is skipped entirely. 281
      preexisting contracts validate end-to-end behaviour with master OFF.
  C2: EWMA arithmetic. With constant reward stream r, v_raw -> r in steady
      state. Reward step from r1 to r2 crosses (r1+r2)/2 at t = half_life.
      REE-low-is-better convention: caller passes raw E3 score; module
      internally negates to give v_raw the standard high-is-good sign.
  C3: gate composition. Gates multiply; energy below e_min linearly suppresses
      v_t; drive above d_max linearly suppresses; PE above pe_max linearly
      suppresses; saturated-bad gate values produce v_t=0.
  C4: additive form correctness. bias[i] = -w_action * v_t on action classes,
      +w_passive * v_t on noop class, clamped to [-bias_scale, +bias_scale].
  C5: multiplicative form correctness. bias[i] = (-w_action * v_t) *
      |scores[i]| on action classes, (+w_passive * v_t) * |scores[i]| on
      noop class, clamped. Distinguishable from additive form on a
      pre-existing-preference held-out batch.
  C6: form-switch validation. tonic_vigor_form='bogus' raises ValueError at
      construction; negative w_action / w_passive / non-positive bias_scale /
      non-positive half_life all raise ValueError.
  C7: select_action wiring contract. With master ON, agent.tonic_vigor is
      populated, single sense() + propose_trajectories + select_action tick
      runs without error, and update_score_receipt advanced n_waking_score_updates
      counter on the realised-score feedback path.
  C8: reset clears EWMA + diagnostic counters. After reset(), v_raw=0, v_t=0,
      all counters reset to zero.
  C9: MECH-094 simulation gate. compute_score_bias(simulation_mode=True)
      returns zeros + increments only the simulation-skip counter.
      update_score_receipt(simulation_mode=True) does NOT advance EWMA +
      increments only simulation-skip counter.
  C10: backward-compat across config matrix. Toggling use_tonic_vigor with
      sibling regulator flags (noise_floor, structured_curiosity, gated_policy)
      does not raise during agent construction or first sense() tick.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.policy import TonicVigor, TonicVigorConfig
from ree_core.utils.config import REEConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_agent_and_one_tick(seed: int = 7, **flags):
    """Build a small REEAgent and run one sense() tick. Returns
    (agent, latent_state)."""
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1,
        use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16, world_dim=16,
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    kwargs = {"obs_body": body, "obs_world": world}
    for key in ("harm_obs", "harm_obs_a"):
        v = obs_dict.get(key)
        if v is not None:
            attr = "use_harm_stream" if key == "harm_obs" else "use_affective_harm_stream"
            if getattr(cfg.latent, attr, False):
                if v.dim() == 1:
                    v = v.unsqueeze(0)
                kwargs[key.replace("harm_obs", "obs_harm")] = v
    with torch.no_grad():
        latent = agent.sense(**kwargs)
    return agent, latent


# ----------------------------------------------------------------------
# C1 default-off no-op
# ----------------------------------------------------------------------
def test_c1_default_off_no_op():
    agent, _ = _build_agent_and_one_tick()
    assert agent.tonic_vigor is None, (
        "default config should produce tonic_vigor=None"
    )


# ----------------------------------------------------------------------
# C2 EWMA arithmetic
# ----------------------------------------------------------------------
def test_c2_ewma_steady_state():
    """Constant reward of 1.0 (REE score=-1) drives v_raw -> 1.0."""
    tv = TonicVigor(TonicVigorConfig(
        use_tonic_vigor=True, half_life=10.0,
    ))
    for _ in range(200):
        tv.update_score_receipt(score=-1.0)
    # 200 ticks at half_life=10 is 20 half-lives; v_raw should be very
    # close to 1.0.
    assert tv._v_raw == pytest.approx(1.0, abs=1e-3)


def test_c2_ewma_step_crossing_at_half_life():
    """Reward step from 0 to 1 crosses 0.5 at t = half_life."""
    tv = TonicVigor(TonicVigorConfig(
        use_tonic_vigor=True, half_life=20.0,
    ))
    # Settle at 0.
    for _ in range(100):
        tv.update_score_receipt(score=0.0)
    assert tv._v_raw == pytest.approx(0.0, abs=1e-9)
    # Step to 1 (REE score=-1) for one half-life.
    for _ in range(20):
        tv.update_score_receipt(score=-1.0)
    # By definition of half-life: after H ticks, v_raw should be at
    # 1 - 0.5 = 0.5.
    assert tv._v_raw == pytest.approx(0.5, abs=1e-3)


def test_c2_sign_convention_low_score_is_high_vigor():
    """REE-convention test: very negative scores (rich rewards) drive
    v_raw UP; very positive scores (bad outcomes) drive v_raw DOWN."""
    tv = TonicVigor(TonicVigorConfig(
        use_tonic_vigor=True, half_life=5.0,
    ))
    # Reward-rich: scores -2.0 -> v_raw climbs toward +2.0.
    for _ in range(50):
        tv.update_score_receipt(score=-2.0)
    assert tv._v_raw > 1.5
    # Reward-poor: scores +1.0 -> v_raw falls.
    for _ in range(50):
        tv.update_score_receipt(score=1.0)
    assert tv._v_raw < 0.0


# ----------------------------------------------------------------------
# C3 gate composition
# ----------------------------------------------------------------------
def test_c3_energy_gate_linear_below_threshold():
    tv = TonicVigor(TonicVigorConfig(
        use_tonic_vigor=True, half_life=5.0,
        gate_energy_min=0.5,
        bias_scale=10.0,  # disable clamp
        w_action=1.0, w_passive=1.0,
    ))
    for _ in range(50):
        tv.update_score_receipt(score=-1.0)
    # energy=1.0 -> gate=1.0
    tv.compute_score_bias(
        torch.zeros(2), torch.tensor([0, 1]),
        energy=1.0, drive=0.0, recent_pe=0.0,
    )
    v_full = tv._last_v_t
    # energy=0.25 -> gate = 0.25/0.5 = 0.5
    tv.compute_score_bias(
        torch.zeros(2), torch.tensor([0, 1]),
        energy=0.25, drive=0.0, recent_pe=0.0,
    )
    v_half = tv._last_v_t
    assert v_half == pytest.approx(v_full * 0.5, rel=1e-3)


def test_c3_drive_gate_suppresses_above_threshold():
    tv = TonicVigor(TonicVigorConfig(
        use_tonic_vigor=True, half_life=5.0,
        gate_drive_max=0.7,
        bias_scale=10.0,
    ))
    for _ in range(50):
        tv.update_score_receipt(score=-1.0)
    # drive=0.0 -> gate=1.0
    tv.compute_score_bias(
        torch.zeros(1), torch.tensor([0]),
        energy=1.0, drive=0.0, recent_pe=0.0,
    )
    v_low = tv._last_v_t
    # drive=1.0 (fully depleted) -> gate=0.0
    tv.compute_score_bias(
        torch.zeros(1), torch.tensor([0]),
        energy=1.0, drive=1.0, recent_pe=0.0,
    )
    v_high = tv._last_v_t
    assert v_high == pytest.approx(0.0, abs=1e-6)
    assert v_low > 0.0


def test_c3_pe_gate_suppresses_above_threshold():
    tv = TonicVigor(TonicVigorConfig(
        use_tonic_vigor=True, half_life=5.0,
        gate_pe_max=1.0,
        bias_scale=10.0,
    ))
    for _ in range(50):
        tv.update_score_receipt(score=-1.0)
    tv.compute_score_bias(
        torch.zeros(1), torch.tensor([0]),
        energy=1.0, drive=0.0, recent_pe=0.0,
    )
    v_low = tv._last_v_t
    # pe=2.0 (>=2*pe_max) -> gate=0.0
    tv.compute_score_bias(
        torch.zeros(1), torch.tensor([0]),
        energy=1.0, drive=0.0, recent_pe=2.0,
    )
    v_high = tv._last_v_t
    assert v_high == pytest.approx(0.0, abs=1e-6)
    assert v_low > 0.0


# ----------------------------------------------------------------------
# C4 additive form correctness
# ----------------------------------------------------------------------
def test_c4_additive_bias_sign_and_magnitude():
    tv = TonicVigor(TonicVigorConfig(
        use_tonic_vigor=True, half_life=5.0,
        w_action=0.05, w_passive=0.05,
        bias_scale=1.0,  # large enough to not clamp
        form="additive",
    ))
    for _ in range(50):
        tv.update_score_receipt(score=-1.0)
    bias = tv.compute_score_bias(
        torch.tensor([0.5, -0.3, 0.7, 0.0]),
        torch.tensor([0, 1, 1, 2]),
        energy=1.0, drive=0.0, recent_pe=0.0,
    )
    v_t = tv._last_v_t
    # noop class 0: bias = +w_passive * v_t = +0.05 * v_t
    # action classes 1, 2: bias = -w_action * v_t = -0.05 * v_t
    expected = torch.tensor([
        +0.05 * v_t, -0.05 * v_t, -0.05 * v_t, -0.05 * v_t,
    ])
    assert torch.allclose(bias, expected, atol=1e-5)


def test_c4_additive_clamp():
    tv = TonicVigor(TonicVigorConfig(
        use_tonic_vigor=True, half_life=5.0,
        w_action=0.5, w_passive=0.5,
        bias_scale=0.1,  # tight clamp
        form="additive",
    ))
    for _ in range(50):
        tv.update_score_receipt(score=-1.0)
    bias = tv.compute_score_bias(
        torch.zeros(4), torch.tensor([0, 1, 1, 2]),
        energy=1.0, drive=0.0, recent_pe=0.0,
    )
    # All values clamped to +/- 0.1
    assert torch.all(bias.abs() <= 0.1 + 1e-6)
    # noop=0 sign positive; action=1,2 sign negative
    assert bias[0] > 0
    assert bias[1] < 0


# ----------------------------------------------------------------------
# C5 multiplicative form correctness
# ----------------------------------------------------------------------
def test_c5_multiplicative_form_scales_with_score_magnitude():
    tv = TonicVigor(TonicVigorConfig(
        use_tonic_vigor=True, half_life=5.0,
        w_action=0.05, w_passive=0.05,
        bias_scale=10.0,
        form="multiplicative",
    ))
    for _ in range(50):
        tv.update_score_receipt(score=-1.0)
    scores = torch.tensor([0.5, -0.3, 0.7, 0.0])
    classes = torch.tensor([0, 1, 1, 2])
    bias = tv.compute_score_bias(
        scores, classes,
        energy=1.0, drive=0.0, recent_pe=0.0,
    )
    v_t = tv._last_v_t
    # noop class 0: bias = +0.05 * v_t * |0.5|
    # action class 1: bias = -0.05 * v_t * |-0.3|
    # action class 1: bias = -0.05 * v_t * |0.7|
    # action class 2: bias = -0.05 * v_t * |0.0| = 0
    expected = torch.tensor([
        +0.05 * v_t * 0.5,
        -0.05 * v_t * 0.3,
        -0.05 * v_t * 0.7,
        0.0,
    ])
    assert torch.allclose(bias, expected, atol=1e-4)


def test_c5_multiplicative_distinguishable_from_additive():
    """On a held-out batch with non-uniform |scores|, multiplicative form
    produces per-candidate-magnitude-varying bias whereas additive
    produces a uniform-per-class bias."""
    cfg_kwargs = dict(
        use_tonic_vigor=True, half_life=5.0,
        w_action=0.05, w_passive=0.05,
        bias_scale=10.0,
    )
    tv_a = TonicVigor(TonicVigorConfig(form="additive", **cfg_kwargs))
    tv_m = TonicVigor(TonicVigorConfig(form="multiplicative", **cfg_kwargs))
    for _ in range(50):
        tv_a.update_score_receipt(score=-1.0)
        tv_m.update_score_receipt(score=-1.0)
    # Two action candidates with different |score|: 0.1 vs 0.9.
    scores = torch.tensor([0.1, 0.9])
    classes = torch.tensor([1, 1])  # both action class
    bias_a = tv_a.compute_score_bias(
        scores, classes, energy=1.0, drive=0.0, recent_pe=0.0,
    )
    bias_m = tv_m.compute_score_bias(
        scores, classes, energy=1.0, drive=0.0, recent_pe=0.0,
    )
    # Additive: both candidates get the same negative bias.
    assert bias_a[0] == pytest.approx(bias_a[1], abs=1e-6)
    # Multiplicative: |bias[1]| > |bias[0]| because |score[1]| > |score[0]|.
    assert abs(bias_m[1]) > abs(bias_m[0])


# ----------------------------------------------------------------------
# C6 form-switch validation
# ----------------------------------------------------------------------
def test_c6_form_validation():
    with pytest.raises(ValueError, match="form must be one of"):
        TonicVigor(TonicVigorConfig(form="bogus"))


def test_c6_negative_w_action_raises():
    with pytest.raises(ValueError, match="w_action"):
        TonicVigor(TonicVigorConfig(w_action=-0.1))


def test_c6_negative_w_passive_raises():
    with pytest.raises(ValueError, match="w_passive"):
        TonicVigor(TonicVigorConfig(w_passive=-0.1))


def test_c6_non_positive_bias_scale_raises():
    with pytest.raises(ValueError, match="bias_scale"):
        TonicVigor(TonicVigorConfig(bias_scale=0.0))


def test_c6_non_positive_half_life_raises():
    with pytest.raises(ValueError, match="half_life"):
        TonicVigor(TonicVigorConfig(half_life=0.0))


# ----------------------------------------------------------------------
# C7 select_action wiring contract
# ----------------------------------------------------------------------
def test_c7_agent_wires_tonic_vigor_when_flag_on():
    agent, _ = _build_agent_and_one_tick(use_tonic_vigor=True)
    assert agent.tonic_vigor is not None
    assert isinstance(agent.tonic_vigor, TonicVigor)
    assert agent.tonic_vigor.config.use_tonic_vigor is True


def test_c7_select_action_advances_ewma_counter():
    """One-tick boot through act_with_split_obs advances
    n_waking_score_updates."""
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(7)
    env = CausalGridWorldV2(
        seed=7, size=5, num_hazards=1, num_resources=1,
        use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16, world_dim=16,
        use_tonic_vigor=True,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    n_pre = agent.tonic_vigor._n_waking_score_updates
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    with torch.no_grad():
        action = agent.act_with_split_obs(
            obs_body=body, obs_world=world,
        )
    n_post = agent.tonic_vigor._n_waking_score_updates
    assert n_post == n_pre + 1, (
        f"expected exactly +1 EWMA waking advance per act tick; "
        f"got {n_post - n_pre}"
    )
    # Bias-call counter also advances.
    assert agent.tonic_vigor._n_waking_bias_calls >= 1


# ----------------------------------------------------------------------
# C8 reset clears state
# ----------------------------------------------------------------------
def test_c8_reset_clears_ewma_and_counters():
    tv = TonicVigor(TonicVigorConfig(use_tonic_vigor=True))
    for _ in range(50):
        tv.update_score_receipt(score=-1.0)
    tv.compute_score_bias(
        torch.zeros(2), torch.tensor([0, 1]),
        energy=1.0, drive=0.0, recent_pe=0.0,
    )
    assert tv._v_raw != 0.0
    assert tv._n_waking_score_updates > 0
    assert tv._n_waking_bias_calls > 0
    tv.reset()
    assert tv._v_raw == 0.0
    assert tv._last_v_t == 0.0
    assert tv._n_waking_score_updates == 0
    assert tv._n_simulation_score_skips == 0
    assert tv._n_waking_bias_calls == 0
    assert tv._n_simulation_bias_skips == 0


# ----------------------------------------------------------------------
# C9 MECH-094 simulation gate
# ----------------------------------------------------------------------
def test_c9_compute_score_bias_sim_returns_zeros():
    tv = TonicVigor(TonicVigorConfig(use_tonic_vigor=True))
    for _ in range(50):
        tv.update_score_receipt(score=-1.0)
    bias = tv.compute_score_bias(
        torch.zeros(3), torch.tensor([0, 1, 2]),
        energy=1.0, drive=0.0, recent_pe=0.0,
        simulation_mode=True,
    )
    assert torch.all(bias == 0.0)
    assert tv._n_simulation_bias_skips == 1
    assert tv._n_waking_bias_calls == 0


def test_c9_update_score_receipt_sim_does_not_advance_ewma():
    tv = TonicVigor(TonicVigorConfig(use_tonic_vigor=True))
    for _ in range(50):
        tv.update_score_receipt(score=-1.0)
    v_pre = tv._v_raw
    tv.update_score_receipt(score=-100.0, simulation_mode=True)
    assert tv._v_raw == v_pre
    assert tv._n_simulation_score_skips == 1


def test_c9_sim_skip_counter_independent_of_waking():
    tv = TonicVigor(TonicVigorConfig(use_tonic_vigor=True))
    tv.update_score_receipt(score=-1.0)  # waking
    tv.update_score_receipt(score=-1.0, simulation_mode=True)  # sim
    tv.update_score_receipt(score=-1.0)  # waking
    assert tv._n_waking_score_updates == 2
    assert tv._n_simulation_score_skips == 1


# ----------------------------------------------------------------------
# C10 backward-compat across config matrix
# ----------------------------------------------------------------------
@pytest.mark.parametrize("flags", [
    {"use_tonic_vigor": True},
    {"use_tonic_vigor": True, "use_noise_floor": True},
    {"use_tonic_vigor": True, "use_structured_curiosity": True},
    {"use_tonic_vigor": True, "tonic_vigor_form": "multiplicative"},
    {"use_tonic_vigor": True, "use_gated_policy": True},
    {"use_tonic_vigor": True, "use_dacc": True, "dacc_weight": 0.0},
])
def test_c10_backward_compat_config_matrix(flags):
    agent, latent = _build_agent_and_one_tick(**flags)
    assert agent is not None
    assert latent is not None
    if flags.get("use_tonic_vigor"):
        assert agent.tonic_vigor is not None
