"""SD-013 / SD-031: the interventional margin loss has an ABSORBING collapse point.

Measured 2026-09-22 (chip-20260922-sd013-margin-loss-collapse-point) against
E2HarmSForward.compute_interventional_loss (SD-013) and
E2WorldForward.compute_interventional_loss (the SD-031 z_world analogue).

WHAT IS PINNED HERE, and why each test can fail:

1. The loss is a NEGATIVE INSTRUMENT. At exact action-invariance it returns
   `margin` -- maximum alarm -- with ZERO gradient. It reads identically to the
   fixable violating case. Pinned so the property stays visible.

2. interventional_loss_is_live() separates those two states. This is the
   explicit cannot-determine category CLAUDE.md requires of any instrument
   whose negative authorises a decision.

3. THE GRADIENT MUST STAY UNIT-NORM NEAR COLLAPSE. grad ||d|| = d/||d|| has
   norm 1 for every nonzero d -- it does NOT blow up, so the usual
   "epsilon under the sqrt guards a derivative blow-up" rationale is FALSE
   here. Adding eps=1e-12 ATTENUATES the gradient by ~10x at ||d||=1e-7 and
   ~1e4x at ||d||=1e-10, i.e. exactly where the head most needs it. This test
   FAILS if anyone applies that epsilon to these two heads. See
   test_blind_spot_* below for the measurement that proves it can fail.

4. The collapsed state is REACHABLE by ordinary weight decay -- it is not the
   measure-zero curiosity a random-init argument suggests.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ree_core.predictors.e2_harm_s import E2HarmSForward, E2HarmSConfig
from ree_core.predictors.e2_world import E2WorldForward, E2WorldConfig

MARGIN_TOL = 1e-6


def _heads():
    return [
        ("E2HarmSForward", E2HarmSForward(E2HarmSConfig())),
        ("E2WorldForward", E2WorldForward(E2WorldConfig(z_world_dim=128))),
    ]


def _zdim(cfg):
    return getattr(cfg, "z_harm_dim", None) or cfg.z_world_dim


def _actions(n, adim):
    a1 = torch.zeros(n, adim)
    a1[:, 0] = 1.0
    a2 = torch.zeros(n, adim)
    a2[:, 1] = 1.0
    return a1, a2


def _collapse_action_path(head, zd):
    """Put the head in EXACT action-invariance (the absorbing state)."""
    rf = head._residual_fwd
    with torch.no_grad():
        rf.action_encoder.weight.zero_()
        rf.action_encoder.bias.zero_()
        rf.transition_net[0].weight[:, zd:].zero_()


def _param_grad_norm(head):
    return sum(
        float(p.grad.pow(2).sum()) for p in head.parameters() if p.grad is not None
    ) ** 0.5


@pytest.mark.parametrize("label,head", _heads())
def test_loss_reports_full_violation_with_zero_gradient_at_exact_collapse(label, head):
    """The negative-instrument property itself: maximum alarm, no gradient."""
    torch.manual_seed(0)
    cfg = head.config
    zd = _zdim(cfg)
    _collapse_action_path(head, zd)
    z = torch.randn(16, zd)
    a1, a2 = _actions(16, cfg.action_dim)

    loss = head.compute_interventional_loss(z, a1, a2)
    head.zero_grad()
    loss.backward()

    assert float(loss.detach()) == pytest.approx(cfg.interventional_margin, abs=MARGIN_TOL), (
        f"{label}: expected full-violation value == margin"
    )
    assert _param_grad_norm(head) == 0.0, (
        f"{label}: exact collapse must deliver ZERO gradient -- if this now fails, "
        f"the loss was changed; read the SD-013 COLLAPSE POINT note in "
        f"compute_interventional_loss before 'fixing' it."
    )


@pytest.mark.parametrize("label,head", _heads())
def test_is_live_separates_collapsed_from_merely_violating(label, head):
    """The cannot-determine category: True when trainable, False when absorbing."""
    torch.manual_seed(0)
    cfg = head.config
    zd = _zdim(cfg)
    z = torch.randn(16, zd)
    a1, a2 = _actions(16, cfg.action_dim)

    assert head.interventional_loss_is_live(z, a1, a2) is True, (
        f"{label}: a randomly initialised head is trainable, not collapsed"
    )

    _collapse_action_path(head, zd)
    assert head.interventional_loss_is_live(z, a1, a2) is False, (
        f"{label}: exact action-invariance must report NOT live"
    )
    # ... and the loss still reports maximum alarm, which is the whole point.
    assert float(
        head.compute_interventional_loss(z, a1, a2).detach()
    ) == pytest.approx(cfg.interventional_margin, abs=MARGIN_TOL)


@pytest.mark.parametrize("mag,min_norm", [(1e-4, 0.99), (1e-6, 0.99), (1e-7, 0.99), (1e-10, 0.99)])
def test_gradient_stays_unit_norm_near_collapse_no_epsilon(mag, min_norm):
    """FAILS if an epsilon-under-the-sqrt is applied. d/||d|| is already bounded."""
    d = torch.zeros(1, 8)
    d[0, 0] = mag
    d = d.clone().requires_grad_(True)
    F.relu(0.5 - d.norm(dim=-1)).mean().backward()
    assert float(d.grad.norm()) >= min_norm, (
        f"gradient collapsed to {float(d.grad.norm()):.3e} at ||d||={mag:.0e}; "
        f"an epsilon under the sqrt does this and is a pessimisation, not a fix"
    )


def test_blind_spot_epsilon_form_would_fail_the_unit_norm_test():
    """MEASURES THE BLIND SPOT of the test above: the defect it must catch.

    If this test's numbers ever stop holding, the unit-norm test above has gone
    vacuous and stops guarding anything.
    """
    for mag, worst in ((1e-7, 0.2), (1e-10, 1e-3)):
        d = torch.zeros(1, 8)
        d[0, 0] = mag
        d = d.clone().requires_grad_(True)
        F.relu(0.5 - (d.pow(2).sum(dim=-1) + 1e-12).sqrt()).mean().backward()
        got = float(d.grad.norm())
        assert got < worst, (
            f"eps form at ||d||={mag:.0e} gave {got:.3e}, expected < {worst}; "
            f"the unit-norm contract above would no longer be able to fail"
        )


def test_blind_spot_float64_distance_does_not_rescue_either():
    """The other plausible 'fix', measured: the predictions are already
    bit-identical when they leave transition_net, so casting afterwards is a
    no-op. Pins that the escape must come from OUTSIDE the loss."""
    torch.manual_seed(0)
    head = E2HarmSForward(E2HarmSConfig())
    cfg = head.config
    zd = _zdim(cfg)
    _collapse_action_path(head, zd)
    z = torch.randn(16, zd)
    a1, a2 = _actions(16, cfg.action_dim)

    rf = head._residual_fwd
    d64 = (rf(z, a1) - rf(z, a2)).double()
    loss = F.relu(cfg.interventional_margin - d64.norm(dim=-1)).mean().float()
    head.zero_grad()
    loss.backward()
    assert _param_grad_norm(head) == 0.0, "float64 must not rescue the absorbing state"


def test_collapse_is_reachable_by_weight_decay_not_measure_zero():
    """Reachability: ordinary weight decay walks into the absorbing state."""
    torch.manual_seed(7)
    g = torch.Generator().manual_seed(7)
    head = E2HarmSForward(E2HarmSConfig())
    cfg = head.config
    zd, adim = _zdim(cfg), cfg.action_dim
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=0.1)
    for _ in range(600):
        z = torch.randn(64, zd, generator=g)
        idx = torch.randint(0, adim, (64,), generator=g)
        a = torch.zeros(64, adim)
        a[torch.arange(64), idx] = 1.0
        z_next = z + 0.5 * torch.randn(64, zd, generator=g) + 0.001 * idx.float().unsqueeze(-1)
        loss = head.compute_loss(head(z, a), z_next.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()

    z = torch.randn(128, zd, generator=g)
    a1, a2 = _actions(128, adim)
    assert head.interventional_loss_is_live(z, a1, a2) is False, (
        "weight decay no longer reaches exact collapse on this architecture -- "
        "re-measure before relaxing the SD-013 COLLAPSE POINT note"
    )


def _tiny_diff_head(target_lo=1e-9, target_hi=1e-6):
    """A REAL head whose two predictions differ by a tiny but NONZERO amount.

    Returns (head, cfg, z, a1, a2, dist_fp32) or None if no scale lands in the
    window (explicit cannot-determine -- the caller skips rather than passing
    vacuously).
    """
    torch.manual_seed(3)
    head = E2HarmSForward(E2HarmSConfig())
    cfg = head.config
    zd = _zdim(cfg)
    z = torch.randn(4, zd)
    a1, a2 = _actions(4, cfg.action_dim)
    rf = head._residual_fwd
    base = rf.transition_net[0].weight[:, zd:].detach().clone()
    for scale in [10.0 ** -k for k in range(2, 14)]:
        with torch.no_grad():
            rf.transition_net[0].weight[:, zd:] = base * scale
            d = (rf(z, a1) - rf(z, a2)).norm(dim=-1)
        dmin = float(d.min())
        if target_lo <= dmin <= target_hi and float(d.max()) <= target_hi:
            return head, cfg, z, a1, a2, d
    return None


def test_real_head_loss_carries_no_epsilon_at_tiny_distance():
    """NOT a restatement of the formula: reads the distance from the MODULE
    UNDER TEST's own predictions and asserts the loss it returns equals
    margin - ||d|| exactly. An eps=1e-12 under the sqrt inflates the distance
    to ~1e-6 and fails this. This is the test that actually guards the file."""
    got = _tiny_diff_head()
    if got is None:
        pytest.skip("cannot determine: no weight scale produced a tiny nonzero ||d||")
    head, cfg, z, a1, a2, d = got
    expected = F.relu(cfg.interventional_margin - d).mean()
    actual = head.compute_interventional_loss(z, a1, a2)
    assert float(actual.detach()) == pytest.approx(float(expected.detach()), abs=1e-9), (
        f"loss {float(actual.detach()):.12f} != margin - ||d|| "
        f"{float(expected.detach()):.12f} at ||d||~{float(d.min()):.3e}; "
        f"an epsilon under the sqrt produces exactly this discrepancy"
    )
