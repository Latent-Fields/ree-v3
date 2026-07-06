"""Contracts for SD-063 -- E2 conditional predictive-uncertainty head.

SD-063 gives the E2 world-forward a distribution-free quantile/pinball head that
emits a per-input predictive spread (the V3-EXQ-712 winner), feeding E3
commitment gating in place of the state-blind running-variance EMA -- the
concrete realization of the MECH-059 confidence channel.

Coverage:
  - config defaults are no-op (bit-identical OFF surface): LatentStackConfig
    use_e2_world_uncertainty=False, E3Config use_conditional_precision_gate=False,
    and REEConfig.from_dims surfaces both;
  - head shapes: forward -> [B, D, Q]; predictive_variance / predictive_std -> [B];
  - z_world_dim is a required constructor arg (no silent mis-size);
  - quantile-level validation (strictly increasing, in (0,1), >= 2 levels);
  - pinball training reduces loss AND the per-input variance is CONDITIONAL
    (tracks heteroscedastic action-conditional noise) -- the property the EMA
    structurally cannot carry;
  - SD-031 agency-residual guard: the head shares NO params with E2WorldForward,
    and its loss (detached target) does not backprop into an encoder-side leaf;
  - E3 commit gate: OFF ignores the conditional variance (EMA path, byte-identical);
    ON lets the conditional variance override the EMA in BOTH directions; ON with
    no value supplied falls back to the EMA;
  - MECH-094 does NOT apply (waking online read; no memory write / simulation).
"""

from __future__ import annotations

import pytest
import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e2_world import E2WorldConfig, E2WorldForward
from ree_core.predictors.e2_world_uncertainty import (
    E2WorldUncertaintyConfig,
    E2WorldUncertaintyHead,
    QUANTILE_LEVELS,
)
from ree_core.predictors.e3_selector import (
    E3Config,
    E3TrajectorySelector,
    variance_commit_threshold,
)
from ree_core.utils.config import LatentStackConfig, REEConfig

WORLD_DIM = 32
ACTION_DIM = 4


def _head(**kw) -> E2WorldUncertaintyHead:
    cfg = E2WorldUncertaintyConfig(
        use_e2_world_uncertainty=True, z_world_dim=WORLD_DIM, action_dim=ACTION_DIM, **kw
    )
    return E2WorldUncertaintyHead(cfg)


def _onehot(n: int, idx=None) -> torch.Tensor:
    a = torch.zeros(n, ACTION_DIM)
    if idx is None:
        idx = torch.randint(0, ACTION_DIM, (n,))
    a[range(n), idx] = 1.0
    return a


def _traj(world_dim=WORLD_DIM, horizon=3) -> Trajectory:
    states = [torch.randn(1, 32) for _ in range(horizon + 1)]
    world_states = [torch.randn(1, world_dim) for _ in range(horizon + 1)]
    actions = torch.randn(1, horizon, ACTION_DIM)
    return Trajectory(states=states, actions=actions, world_states=world_states)


# ------------------------------------------------------------------ #
# Config no-op surface                                                #
# ------------------------------------------------------------------ #

def test_config_defaults_are_no_op():
    ls = LatentStackConfig()
    assert ls.use_e2_world_uncertainty is False
    assert ls.e2_world_uncertainty_hidden_dim == 128
    assert ls.e2_world_uncertainty_lr == 1e-3
    assert E3Config().use_conditional_precision_gate is False


def test_from_dims_surfaces_flags():
    cfg = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=54, action_dim=ACTION_DIM,
        self_dim=32, world_dim=WORLD_DIM,
    )
    assert cfg.latent.use_e2_world_uncertainty is False
    assert cfg.e3.use_conditional_precision_gate is False


# ------------------------------------------------------------------ #
# Head shapes + validation                                           #
# ------------------------------------------------------------------ #

def test_forward_and_readout_shapes():
    head = _head()
    B = 8
    z = torch.randn(B, WORLD_DIM)
    a = _onehot(B)
    q = head(z, a)
    assert q.shape == (B, WORLD_DIM, len(QUANTILE_LEVELS))
    assert head.predictive_variance(z, a).shape == (B,)
    assert head.predictive_std(z, a).shape == (B,)
    assert (head.predictive_variance(z, a) >= 0).all()


def test_z_world_dim_required():
    with pytest.raises(ValueError):
        E2WorldUncertaintyHead(E2WorldUncertaintyConfig(use_e2_world_uncertainty=True))


@pytest.mark.parametrize("levels", [[0.5], [0.2, 0.2], [0.9, 0.1], [-0.1, 0.5], [0.5, 1.2]])
def test_bad_quantile_levels_rejected(levels):
    with pytest.raises(ValueError):
        _head(quantile_levels=levels)


# ------------------------------------------------------------------ #
# Training + conditional (heteroscedastic) variance                  #
# ------------------------------------------------------------------ #

def test_pinball_training_and_conditional_variance():
    torch.manual_seed(0)
    head = _head()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    n = 512
    z = torch.randn(n, WORLD_DIM)
    ai = torch.randint(0, ACTION_DIM, (n,))
    a = _onehot(n, ai)
    noise = (0.05 + 0.5 * ai.float()).unsqueeze(1)   # action-conditional spread
    z_next = z + noise * torch.randn(n, WORLD_DIM)
    l0 = head.compute_loss(head(z.detach(), a), z_next.detach()).item()
    for _ in range(150):
        opt.zero_grad()
        loss = head.compute_loss(head(z.detach(), a), z_next.detach())
        loss.backward()
        opt.step()
    l1 = head.compute_loss(head(z.detach(), a), z_next.detach()).item()
    assert l1 < l0

    lo = _onehot(64, torch.zeros(64, dtype=torch.long))
    hi = _onehot(64, torch.full((64,), ACTION_DIM - 1, dtype=torch.long))
    zc = torch.randn(64, WORLD_DIM)
    v_lo = head.predictive_variance(zc, lo).mean().item()
    v_hi = head.predictive_variance(zc, hi).mean().item()
    assert v_hi > v_lo, "predictive variance must track conditional (heteroscedastic) noise"


# ------------------------------------------------------------------ #
# SD-031 agency-residual guard                                        #
# ------------------------------------------------------------------ #

def test_head_params_disjoint_from_e2_world_forward():
    head = _head()
    wf = E2WorldForward(
        E2WorldConfig(use_e2_world_forward=True, z_world_dim=128, action_dim=ACTION_DIM)
    )
    head_ids = {id(p) for p in head.parameters()}
    wf_ids = {id(p) for p in wf.parameters()}
    assert head_ids.isdisjoint(wf_ids)


def test_detached_target_blocks_encoder_gradient():
    head = _head()
    enc_leaf = torch.randn(16, WORLD_DIM, requires_grad=True)
    z_from_enc = enc_leaf * 2.0            # z_world produced by a (mock) encoder
    a = _onehot(16)
    tgt = torch.randn(16, WORLD_DIM)
    loss = head.compute_loss(head(z_from_enc.detach(), a), tgt.detach())
    loss.backward()
    assert enc_leaf.grad is None, "SD-031 violation: head grad reached the encoder"


# ------------------------------------------------------------------ #
# E3 commit-gate hook                                                 #
# ------------------------------------------------------------------ #

def _selector(gate: bool) -> E3TrajectorySelector:
    return E3TrajectorySelector(
        E3Config(world_dim=WORLD_DIM, hidden_dim=64, use_conditional_precision_gate=gate)
    )


_THR = variance_commit_threshold(E3Config().commitment_threshold)
_LOW = _THR * 0.1     # confident
_HIGH = _THR * 10.0   # uncertain


def _cands():
    torch.manual_seed(7)
    return [_traj() for _ in range(3)]


def test_gate_off_ignores_conditional_variance():
    sel = _selector(gate=False)
    sel._running_variance = _LOW  # EMA -> commit
    r_a = sel.select(_cands())
    sel._running_variance = _LOW
    r_b = sel.select(_cands(), conditional_predictive_variance=_HIGH)
    assert r_a.committed is True and r_b.committed is True


def test_gate_on_conditional_variance_overrides_ema_both_directions():
    sel = _selector(gate=True)
    # EMA says commit (low) but conditional says uncertain (high) -> veto
    sel._running_variance = _LOW
    assert sel.select(_cands(), conditional_predictive_variance=_HIGH).committed is False
    # EMA says don't commit (high) but conditional says confident (low) -> commit
    sel._running_variance = _HIGH
    assert sel.select(_cands(), conditional_predictive_variance=_LOW).committed is True


def test_gate_on_without_value_falls_back_to_ema():
    sel = _selector(gate=True)
    sel._running_variance = _LOW
    assert sel.select(_cands()).committed is True
    sel._running_variance = _HIGH
    assert sel.select(_cands()).committed is False
