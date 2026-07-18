"""Contracts for the MECH-457 distributional (two-hot / HL-Gauss) critic.

Substrate: ree_core/action_learning/distributional_value.py + the use_distributional_critic
path in ActorCriticPolicy. Unblocks hypothesis H-retention-critic (competence_floor).

The contracts that matter here are (a) OFF is byte-identical to the pre-change scalar critic,
(b) the swap is confined to the VALUE ESTIMATOR (the anti-alias constraint that keeps this leg
readable against mech457_policy_kl_anchor), and (c) the ON arm is actually TRAINED -- the
degenerate-arm trap that the successor-feature critic falls into (a zero-init reward_w read as
an alternative critic) is exactly what a retention verdict must not be built on.
"""

from __future__ import annotations

import torch

from ree_core.action_learning.actor_critic import ActorCriticPolicy
from ree_core.action_learning.distributional_value import ValueBins, symexp, symlog


# --- C1: OFF is the unchanged scalar critic -------------------------------------
def test_c1_off_is_scalar_critic_unchanged():
    torch.manual_seed(0)
    p = ActorCriticPolicy(world_dim=25, action_dim=5)
    z = torch.randn(4, 25)
    logits, value, phi, psi = p.forward(z)
    assert p.use_distributional_critic is False
    assert p.value_bins is None
    assert p.value_head.out_features == 1
    assert value.shape == (4,)
    assert phi is None and psi is None
    assert p.select(z).value_logits is None


# --- C2: ON changes ONLY the critic (anti-alias) --------------------------------
def test_c2_on_leaves_actor_bit_identical():
    """Same seed, same trunk + policy head. The distributional swap must not perturb the
    actor's parameters, or the leg aliases with an update-rule intervention."""
    torch.manual_seed(0)
    off = ActorCriticPolicy(world_dim=25, action_dim=5)
    torch.manual_seed(0)
    on = ActorCriticPolicy(world_dim=25, action_dim=5, use_distributional_critic=True)
    for a, b in zip(off.trunk.parameters(), on.trunk.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(off.policy_head.weight, on.policy_head.weight)
    assert torch.equal(off.policy_head.bias, on.policy_head.bias)


def test_c2b_critic_loss_leaves_policy_head_ungradiented():
    """The CE critic loss must produce no gradient on the policy head. (The shared trunk does
    receive gradient -- identically to the scalar critic's MSE term, so this is not a change
    in the update rule.)"""
    torch.manual_seed(0)
    on = ActorCriticPolicy(world_dim=25, action_dim=5, use_distributional_critic=True)
    z = torch.randn(3, 25)
    loss = on.critic_loss(on.forward_value_logits(z), torch.tensor([1.0, -2.0, 5.0]))
    loss.backward()
    assert on.policy_head.weight.grad is None
    assert on.value_head.weight.grad is not None
    assert torch.isfinite(on.value_head.weight.grad).all()


# --- C3: the scalar contract survives the swap ----------------------------------
def test_c3_value_stays_scalar_for_gae():
    """GAE, the bootstrap value, the credit-replay TD priority and eval all consume a scalar.
    The distributional critic must decode to one, or every downstream consumer breaks."""
    torch.manual_seed(0)
    on = ActorCriticPolicy(world_dim=25, action_dim=5, use_distributional_critic=True)
    z = torch.randn(6, 25)
    _logits, value, _phi, _psi = on.forward(z)
    step = on.select(z)
    assert value.shape == (6,)
    assert step.value.shape == (6,)
    assert step.value_logits.shape == (6, 41)
    assert torch.isfinite(value).all()


# --- C4: projection round-trips ---------------------------------------------------
def test_c4_hl_gauss_and_two_hot_round_trip():
    targets = torch.tensor([0.0, 1.0, -3.5, 27.0, -140.0])
    for sigma in (0.0, 0.75):
        bins = ValueBins(n_bins=41, limit=10.0, sigma_ratio=sigma)
        p = bins.project(targets)
        assert torch.allclose(p.sum(dim=-1), torch.ones(5), atol=1e-5)
        assert (p >= 0).all()
        decoded = symexp((p * bins.support).sum(dim=-1))
        assert torch.allclose(decoded, targets, atol=0.05, rtol=0.02)


def test_c4b_two_hot_puts_mass_on_at_most_two_bins():
    bins = ValueBins(n_bins=41, limit=10.0, sigma_ratio=0.0)
    p = bins.project(torch.tensor([1.0, -3.5, 27.0]))
    for row in p:
        assert int((row > 1e-9).sum()) <= 2


def test_c4c_symlog_symexp_are_inverse():
    x = torch.tensor([-1e4, -7.5, 0.0, 0.3, 1e4])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-3, rtol=1e-4)


# --- C5: the ON arm is actually trainable (no degenerate arm) --------------------
def test_c5_critic_fits_a_known_return():
    """The successor-feature trap: a critic that is enabled but never trained reads as an
    alternative estimator while being identically uninformed. Assert the CE objective moves
    the decoded value onto a known target."""
    torch.manual_seed(0)
    on = ActorCriticPolicy(world_dim=25, action_dim=5, use_distributional_critic=True)
    opt = torch.optim.Adam(on.parameters(), lr=0.05)
    z = torch.randn(1, 25)
    target = torch.tensor([7.5])
    for _ in range(200):
        loss = on.critic_loss(on.forward_value_logits(z), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert abs(float(on.forward(z)[1].item()) - 7.5) < 0.5


def test_c5b_scalar_critic_refuses_critic_loss():
    """A mis-wired ON arm must fail loudly, not silently train nothing."""
    on = ActorCriticPolicy(world_dim=25, action_dim=5)
    try:
        on.critic_loss(torch.randn(2, 41), torch.tensor([1.0, 2.0]))
    except RuntimeError:
        return
    raise AssertionError("scalar critic accepted critic_loss without raising")


# --- C6: the _lib dispatch falls back to the identical scalar MSE ----------------
def test_c6_dispatch_matches_scalar_mse_when_off():
    from experiments._lib import mech457_fanout as fan

    torch.manual_seed(0)
    off = ActorCriticPolicy(world_dim=25, action_dim=5)
    value_t = torch.randn(8)
    ret_t = torch.randn(8)
    expected = fan.AC_VALUE_COEF * 0.5 * (value_t - ret_t.detach()).pow(2).mean()
    got = fan.critic_value_loss(off, None, value_t, ret_t)
    assert torch.allclose(got, expected)
    # A None policy (or absent logits) must also take the scalar branch.
    assert torch.allclose(fan.critic_value_loss(None, None, value_t, ret_t), expected)


def test_c6b_dispatch_uses_cross_entropy_when_on():
    from experiments._lib import mech457_fanout as fan

    torch.manual_seed(0)
    on = ActorCriticPolicy(world_dim=25, action_dim=5, use_distributional_critic=True)
    z = torch.randn(8, 25)
    vlogits = on.forward_value_logits(z)
    value_t = on.forward(z)[1]
    ret_t = torch.randn(8)
    got = fan.critic_value_loss(on, vlogits, value_t, ret_t)
    expected = fan.AC_VALUE_COEF * on.critic_loss(vlogits, ret_t)
    assert torch.allclose(got, expected)


# --- C7: config plumbing carries the flag into the fingerprint -------------------
def test_c7_bootstrap_config_default_off_and_declared():
    from experiments._lib.mech457_bootstrap_explorer import BootstrapExplorerConfig

    assert BootstrapExplorerConfig().use_distributional_critic is False
    assert BootstrapExplorerConfig().as_slice()["use_distributional_critic"] is False
    on = BootstrapExplorerConfig(use_distributional_critic=True)
    assert on.as_slice()["use_distributional_critic"] is True
