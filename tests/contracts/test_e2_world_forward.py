"""Contract tests for SD-031 E2WorldForward (z_world single-pass comparator).

Six contracts (C1-C6):
  C1: bit-identical OFF. With use_e2_world_forward=False, agent.e2_world is
      None and the agent's action stream is identical to an explicitly-OFF
      agent across several steps (default == explicit-False).
  C2: shape / wiring. forward() returns [batch, z_world_dim]; comparator_residual
      is finite and responds to the action (different actions -> different
      residuals).
  C3: dim-guard. Constructing E2WorldForward at world_dim=32 with the flag ON
      RAISES ValueError citing the 2026-06-06 autopsy / world_dim>=128; z_world_dim
      omitted (None) RAISES; allow_subthreshold_dim=True constructs at dim=32 but
      attribution_ready is False and comparator_residual returns zeros.
  C4: delta-not-identity. The residual-delta architecture does not collapse to
      identity at init: forward(z, a) != z and forward(z, a0) != forward(z, a1)
      (action-conditional), including on an autocorrelated synthetic z_world.
  C5: MECH-094 simulation gate. comparator_residual(simulation_mode=True) returns
      a zeroed residual while forward() is unaffected; a waking comparator_residual
      with observed != predicted is nonzero.
  C6: agent-path guard. Building a REEAgent with use_e2_world_forward=True at the
      default world_dim (32) RAISES via the agent constructor; at world_dim=128 it
      constructs agent.e2_world with attribution_ready True.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.predictors.e2_world import (
    E2WorldConfig,
    E2WorldForward,
    MIN_DISCRIMINATIVE_WORLD_DIM,
)
from ree_core.utils.config import REEConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _onehot(idx: int, dim: int = 4) -> torch.Tensor:
    a = torch.zeros(1, dim)
    a[0, idx] = 1.0
    return a


def _action_stream(seed: int, steps: int, **flags):
    """Build a small agent and return its first-action argmax over N steps."""
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True
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
    actions = []
    for _ in range(steps):
        body = obs_dict["body_state"]
        world = obs_dict["world_state"]
        if body.dim() == 1:
            body = body.unsqueeze(0)
        if world.dim() == 1:
            world = world.unsqueeze(0)
        with torch.no_grad():
            action = agent.act_with_split_obs(obs_body=body, obs_world=world)
        actions.append(int(action.argmax().item()))
        _flat, obs_dict, _r, done, _info = env.step(action)
        if done:
            _flat, obs_dict = env.reset()
    return actions


# ----------------------------------------------------------------------
# C1: bit-identical OFF
# ----------------------------------------------------------------------
def test_c1_bit_identical_off():
    # Default (flag absent) agent has no e2_world.
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    env = CausalGridWorldV2(seed=1, size=5, num_hazards=1, num_resources=1,
                            use_proxy_fields=True)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=4, self_dim=16, world_dim=16,
    )
    agent = REEAgent(cfg)
    assert agent.e2_world is None

    # default == explicit-False action stream (bit-identical OFF).
    default_stream = _action_stream(seed=3, steps=6)
    explicit_off_stream = _action_stream(seed=3, steps=6, use_e2_world_forward=False)
    assert default_stream == explicit_off_stream


# ----------------------------------------------------------------------
# C2: shape / wiring; residual responds to action
# ----------------------------------------------------------------------
def test_c2_shape_and_action_response():
    torch.manual_seed(0)
    cfg = E2WorldConfig(use_e2_world_forward=True, z_world_dim=128, action_dim=4)
    model = E2WorldForward(cfg)

    z_prev = torch.randn(1, 128)
    z_obs = torch.randn(1, 128)

    pred = model(z_prev, _onehot(0))
    assert pred.shape == (1, 128)
    assert torch.isfinite(pred).all()

    res0 = model.comparator_residual(z_obs, z_prev, _onehot(0))
    res1 = model.comparator_residual(z_obs, z_prev, _onehot(1))
    assert res0.shape == (1, 128)
    assert torch.isfinite(res0).all()
    # Residual responds to the action (different actions -> different residuals).
    assert not torch.allclose(res0, res1)


# ----------------------------------------------------------------------
# C3: dim-guard
# ----------------------------------------------------------------------
def test_c3_dim_guard_raises_below_threshold():
    with pytest.raises(ValueError) as ei:
        E2WorldForward(E2WorldConfig(use_e2_world_forward=True, z_world_dim=32))
    msg = str(ei.value)
    assert str(MIN_DISCRIMINATIVE_WORLD_DIM) in msg
    assert "2026-06-06" in msg


def test_c3_dim_guard_requires_z_world_dim():
    with pytest.raises(ValueError):
        E2WorldForward(E2WorldConfig(use_e2_world_forward=True, z_world_dim=None))


def test_c3_subthreshold_optin_is_not_attribution_ready():
    model = E2WorldForward(
        E2WorldConfig(use_e2_world_forward=True, z_world_dim=32,
                      allow_subthreshold_dim=True)
    )
    assert model.attribution_ready is False
    z_prev = torch.randn(1, 32)
    z_obs = torch.randn(1, 32)
    res = model.comparator_residual(z_obs, z_prev, _onehot(0))
    # Not-ready -> zeroed sentinel residual, NOT a genuine zero-gap attribution.
    assert torch.count_nonzero(res) == 0


# ----------------------------------------------------------------------
# C4: delta-not-identity (no identity collapse)
# ----------------------------------------------------------------------
def test_c4_delta_not_identity():
    torch.manual_seed(1)
    model = E2WorldForward(E2WorldConfig(use_e2_world_forward=True, z_world_dim=128))
    # Autocorrelated synthetic z_world (high self-similarity, the identity-collapse
    # trap that motivated the residual-delta architecture).
    base = torch.randn(1, 128)
    z = 0.95 * base + 0.05 * torch.randn(1, 128)

    pred0 = model(z, _onehot(0))
    pred1 = model(z, _onehot(1))
    # Prediction is not pure identity ...
    assert not torch.allclose(pred0, z)
    # ... and is action-conditional.
    assert not torch.allclose(pred0, pred1)


# ----------------------------------------------------------------------
# C5: MECH-094 simulation gate
# ----------------------------------------------------------------------
def test_c5_mech094_simulation_gate():
    torch.manual_seed(2)
    model = E2WorldForward(E2WorldConfig(use_e2_world_forward=True, z_world_dim=128))
    z_prev = torch.randn(1, 128)
    z_obs = torch.randn(1, 128)

    sim_res = model.comparator_residual(z_obs, z_prev, _onehot(0), simulation_mode=True)
    assert torch.count_nonzero(sim_res) == 0

    # forward() (evaluator/rollout mode) is NOT gated.
    assert torch.isfinite(model(z_prev, _onehot(0))).all()

    # Waking comparator with observed != predicted is nonzero.
    wake_res = model.comparator_residual(z_obs, z_prev, _onehot(0), simulation_mode=False)
    assert torch.count_nonzero(wake_res) > 0


# ----------------------------------------------------------------------
# C6: agent-path guard
# ----------------------------------------------------------------------
def test_c6_agent_default_dim_guard_raises():
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    env = CausalGridWorldV2(seed=1, size=5, num_hazards=1, num_resources=1,
                            use_proxy_fields=True)
    # Default world_dim is 32 -> agent construction must raise when the flag is on.
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=4, self_dim=32, world_dim=32,
        use_e2_world_forward=True,
    )
    with pytest.raises(ValueError):
        REEAgent(cfg)


def test_c6_agent_at_128_constructs_attribution_ready():
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    env = CausalGridWorldV2(seed=1, size=5, num_hazards=1, num_resources=1,
                            use_proxy_fields=True)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=4, self_dim=128, world_dim=128,
        use_e2_world_forward=True,
    )
    agent = REEAgent(cfg)
    assert agent.e2_world is not None
    assert agent.e2_world.attribution_ready is True
