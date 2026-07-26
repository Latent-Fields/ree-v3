"""SD-082 contract: common-mode-invariant trained rule_state -> action-bias read-out.

Pins the substrate fix for the V3-EXQ-822 propagation zero:
  * the flag is reachable end-to-end through REEConfig.from_dims -> LateralPFCConfig
    (the documented silently-unreachable-flag hazard),
  * OFF is bit-identical to the SD-033a landing (hard clamp on the raw summary input),
  * ON un-zeroes propagation in the SD-008 common-mode cone that produced the exact
    0.0 (the manipulation check), and
  * ON restores the head gradient the saturated hard clamp kills (trainability).

See REE_assembly/docs/architecture/sd_082_rule_selection_action_consumer.md.
"""
from __future__ import annotations

import itertools

import torch

from ree_core.pfc.lateral_pfc_analog import LateralPFCAnalog, LateralPFCConfig
from ree_core.utils.config import REEConfig


RULE_DIM = 16
WORLD_DIM = 32
K = 6
BIAS_SCALE = 0.1


def _common_mode_summaries(seed: int = 1) -> torch.Tensor:
    """K near-collinear candidate summaries (the SD-008 ~0.98-cosine cone)."""
    g = torch.Generator().manual_seed(seed)
    common = torch.randn(1, WORLD_DIM, generator=g) * 4.0
    summ = common + torch.randn(K, WORLD_DIM, generator=g) * 0.15
    n = torch.nn.functional.normalize(summ, dim=-1)
    cos = n @ n.t()
    min_cos = min(float(cos[i, j]) for i, j in itertools.combinations(range(K), 2))
    assert min_cos > 0.9, f"fixture not a common-mode cone (min cosine {min_cos})"
    return summ


def _make_head(consumer: bool) -> LateralPFCAnalog:
    torch.manual_seed(7)
    cfg = LateralPFCConfig(
        use_lateral_pfc_analog=True,
        rule_dim=RULE_DIM,
        hidden_dim=32,
        bias_scale=BIAS_SCALE,
        train_rule_bias_head=True,
        use_candidate_rule_source=True,
        rule_readout_consumer=consumer,
        readout_init_scale=0.25,
    )
    return LateralPFCAnalog(delta_dim=8, world_dim=WORLD_DIM, config=cfg)


def _prop_delta(lp: LateralPFCAnalog, summ: torch.Tensor) -> float:
    """Propagation counterfactual: mean |bias(rule_state) - bias(0)| (654f pattern)."""
    lp.rule_state.copy_(torch.randn(1, RULE_DIM) * 1.0)
    with torch.no_grad():
        b_field = lp.compute_bias(summ).clone()
        saved = lp.rule_state.clone()
        lp.rule_state.zero_()
        b_zero = lp.compute_bias(summ).clone()
        lp.rule_state.copy_(saved)
    return float((b_field - b_zero).abs().mean())


def test_flag_reachable_through_from_dims():
    """REEConfig.lateral_pfc_rule_readout_consumer must reach LateralPFCConfig.

    Guards the silently-unreachable-flag hazard: a flag that stops at REEConfig and
    never reaches the module is the confirmed failure mode this substrate depends on
    NOT hitting.
    """
    from ree_core.agent import REEAgent
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    env = CausalGridWorldV2(seed=101, size=8, num_hazards=2, num_resources=6,
                            use_proxy_fields=True)

    def build(flag: bool) -> REEAgent:
        return REEAgent(REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
            action_dim=4, use_lateral_pfc_analog=True,
            lateral_pfc_train_rule_bias_head=True,
            lateral_pfc_rule_readout_consumer=flag,
        ))

    assert build(True).lateral_pfc.config.rule_readout_consumer is True
    assert build(False).lateral_pfc.config.rule_readout_consumer is False
    # Absent -> default False (no-op backward compat).
    default = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=4, use_lateral_pfc_analog=True)
    assert default.lateral_pfc_rule_readout_consumer is False


def test_off_is_bit_identical_hard_clamp():
    """Consumer OFF == the SD-033a landing (hard clamp on the raw summary input)."""
    lp = _make_head(consumer=False)
    summ = _common_mode_summaries()
    lp.rule_state.copy_(torch.randn(1, RULE_DIM) * 0.5)
    with torch.no_grad():
        got = lp.compute_bias(summ)
        rule_rep = lp.rule_state.expand(K, -1)
        raw = lp.rule_bias_head(torch.cat([rule_rep, summ], dim=-1)).squeeze(-1)
        ref = raw.clamp(-BIAS_SCALE, BIAS_SCALE)
    assert torch.allclose(got, ref)


def test_off_reproduces_structural_zero_in_cone():
    """The bug: OFF propagation is a structural zero in the common-mode cone."""
    assert _prop_delta(_make_head(consumer=False), _common_mode_summaries()) == 0.0


def test_on_unzeroes_propagation_in_cone():
    """The fix: ON propagation clears the 1e-3 readiness floor in the same cone."""
    delta = _prop_delta(_make_head(consumer=True), _common_mode_summaries())
    assert delta > 1e-3, f"ON prop_delta {delta} did not clear the readiness floor"


def test_on_bound_magnitude_preserved():
    """Soft-tanh keeps the |bias| < bias_scale guarantee (cannot dominate E3)."""
    lp = _make_head(consumer=True)
    lp.rule_state.copy_(torch.randn(1, RULE_DIM) * 3.0)
    with torch.no_grad():
        bias = lp.compute_bias(_common_mode_summaries())
    assert float(bias.abs().max()) <= BIAS_SCALE + 1e-6


def test_on_restores_head_gradient():
    """Trainability: the saturated hard clamp has zero head gradient; tanh does not."""
    def grad_norm(consumer: bool) -> float:
        lp = _make_head(consumer)
        lp.rule_state.copy_(torch.randn(1, RULE_DIM) * 1.0)
        bias = lp.compute_bias(_common_mode_summaries())
        loss = (bias * torch.randn(K)).sum()
        loss.backward()
        return sum(float(p.grad.norm()) for p in lp.rule_bias_head.parameters()
                   if p.grad is not None)

    assert grad_norm(False) == 0.0
    assert grad_norm(True) > 1e-3
