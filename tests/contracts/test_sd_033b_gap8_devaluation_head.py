"""SD-033b commitment_closure:GAP-8 -- decoupled OFC devaluation head.

Contract for the failure_autopsy_V3-EXQ-485l (2026-06-22) repair: the single
shared OFCAnalog.state_bias_head under the +/-bias_scale clamp has no feasible
gain band -- a gain large enough to differentiate the devalued re-ranking
saturates the clamp (485k range 0.0) while an in-band gain undershoots the 0.05
readout floor (485l 0.031), because the same head + clamp must also carry the C2
high-threat discrimination range. The fix DECOUPLES the devaluation re-ranking
into a SECOND head (devaluation_bias_head) with its OWN independent clamp
(devaluation_bias_scale), read via compute_devaluation_bias().

C1 default (use_devaluation_head off) -> head None, compute_devaluation_bias
   returns zeros, no params -> bit-identical to the original SD-033b landing.
C2 trainable decoupled head -> differentiated supra-floor devalued range at its
   own (larger) clamp, while the C2 state_bias_head clamp is untouched (untraded).
C3 dev-only optimizer trains the devaluation head and leaves state_bias_head
   frozen (genuine decouple, not a shared gradient).
C4 untrained decoupled head (train_devaluation_head off) -> zeroed last Linear ->
   exactly-zero output until deliberately trained; get_state exposes the fields.
C5 from_dims wiring: default builds no dev head; explicit-on builds it with the
   configured scale and reachable params.
C6 own clamp respected; C2 head still clamped to bias_scale.
"""

import torch

from ree_core.pfc.ofc_analog import OFCAnalog, OFCConfig
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

WORLD_DIM = 32
K = 8
DEVAL_FLOOR = 0.05  # the 485l devalued-range readout floor the decouple must clear


def _bank(seed: int = 7) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(K, WORLD_DIM)


def _primed_ofc(**cfg_kw) -> OFCAnalog:
    cfg = OFCConfig(use_ofc_analog=True, harm_dim=0, **cfg_kw)
    ofc = OFCAnalog(world_dim=WORLD_DIM, config=cfg)
    torch.manual_seed(0)
    ofc.update(z_world=torch.randn(1, WORLD_DIM), gate=1.0)
    return ofc


def test_c1_default_off_is_no_op():
    """No devaluation head by default; compute_devaluation_bias is zeros."""
    ofc = _primed_ofc(train_state_bias_head=True)
    assert ofc.devaluation_bias_head is None
    bank = _bank()
    dv = ofc.compute_devaluation_bias(bank)
    assert dv.shape == (K,)
    assert torch.all(dv == 0.0)
    assert len(list(ofc.devaluation_bias_head_parameters())) == 0


def test_c2_trainable_decoupled_head_clears_floor_untraded():
    """Trainable dev head differentiates above the 485l floor at its own clamp;
    the C2 discrimination head stays clamped to bias_scale (magnitude untraded)."""
    ofc = _primed_ofc(
        train_state_bias_head=True,
        use_devaluation_head=True,
        devaluation_bias_scale=2.0,
        train_devaluation_head=True,
    )
    assert ofc.devaluation_bias_head is not None
    assert ofc.state_bias_head is not ofc.devaluation_bias_head
    bank = _bank()
    dv = ofc.compute_devaluation_bias(bank).detach()
    dv_range = float(dv.max() - dv.min())
    # the decoupled head produces a differentiated, supra-floor devalued range
    assert dv_range > DEVAL_FLOOR
    # respects its OWN larger clamp
    assert float(dv.abs().max()) <= 2.0 + 1e-6
    # C2 head independent: still clamped to the (small) bias_scale, untraded
    b = ofc.compute_bias(bank)
    assert float(b.abs().max()) <= ofc.config.bias_scale + 1e-6


def test_c3_dev_only_optimizer_trains_dev_head_freezes_c2():
    ofc = _primed_ofc(
        train_state_bias_head=True,
        use_devaluation_head=True,
        devaluation_bias_scale=2.0,
        train_devaluation_head=True,
    )
    bank = _bank()
    opt = torch.optim.Adam(list(ofc.devaluation_bias_head_parameters()), lr=0.05)
    dev_w0 = ofc.devaluation_bias_head[-1].weight.detach().clone()
    c2_w0 = ofc.state_bias_head[-1].weight.detach().clone()
    loss = ofc.compute_devaluation_bias(bank).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert not torch.allclose(ofc.devaluation_bias_head[-1].weight, dev_w0)
    assert torch.allclose(ofc.state_bias_head[-1].weight, c2_w0)


def test_c4_untrained_head_zeroed_and_get_state_fields():
    ofc = _primed_ofc(use_devaluation_head=True, train_devaluation_head=False)
    bank = _bank()
    assert torch.all(ofc.compute_devaluation_bias(bank) == 0.0)
    gs = ofc.get_state()
    assert gs["use_devaluation_head"] is True
    assert gs["train_devaluation_head"] is False
    assert gs["devaluation_bias_scale"] == 2.0
    assert "last_devaluation_bias_abs_mean" in gs


def test_c5_from_dims_wiring():
    base = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_ofc_analog=True, ofc_harm_dim=0,
    )
    a = REEAgent(base)
    assert a.ofc is not None and a.ofc.devaluation_bias_head is None

    on = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_ofc_analog=True, ofc_harm_dim=0,
        use_ofc_devaluation_head=True,
        ofc_devaluation_bias_scale=2.0,
        ofc_train_devaluation_head=True,
    )
    a2 = REEAgent(on)
    assert a2.ofc.devaluation_bias_head is not None
    assert len(list(a2.ofc.devaluation_bias_head_parameters())) == 4
    assert abs(float(a2.ofc.config.devaluation_bias_scale) - 2.0) < 1e-9


def test_c6_dev_head_off_when_master_ofc_off():
    """use_devaluation_head is inert if the OFC analog itself is off."""
    cfg = OFCConfig(use_ofc_analog=False, use_devaluation_head=True)
    ofc = OFCAnalog(world_dim=WORLD_DIM, config=cfg)
    bank = _bank()
    # compute_devaluation_bias short-circuits on use_ofc_analog False
    assert torch.all(ofc.compute_devaluation_bias(bank) == 0.0)
