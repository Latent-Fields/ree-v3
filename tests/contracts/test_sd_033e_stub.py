"""Contract tests for SD-033e FrontopolarAnalog stub (V4-scope).

Guarantees enforced here:
  C1. Module and dataclass importable without side effects.
  C2. Default config has use_frontopolar_analog=False (backward-compat).
  C3. When disabled (default), all public compute_* methods return zero /
      uniform tensors of the correct shape. No exceptions.
  C4. When enabled without an implementation, all public compute_* methods
      raise NotImplementedError. This is the contract that prevents the
      stub from silently contaminating ablation baselines.
  C5. Zero-init contract on last Linear layers of the two heads (matches
      SD-033a pattern). This is what will make the enabled path
      bit-identical to disabled once the guard is lifted.
  C6. reset() clears diagnostics without raising.
  C7. get_state() returns a dict with stub=True marker.

These contracts test interface shape only. They must not assert any
behavioural magnitude, because the stub is explicitly behaviour-free.
Replace these contracts when the SD-033e design doc lands.
"""

from __future__ import annotations

import pytest
import torch


def test_c1_module_importable():
    """C1: module and dataclass importable without side effects."""
    from ree_core.pfc import FrontopolarAnalog, FrontopolarConfig
    from ree_core.pfc.frontopolar_analog import (
        FrontopolarAnalog as FA,
        FrontopolarConfig as FC,
    )
    assert FrontopolarAnalog is FA
    assert FrontopolarConfig is FC


def test_c2_default_config_is_backward_compatible():
    """C2: default config has use_frontopolar_analog=False."""
    from ree_core.pfc import FrontopolarConfig
    cfg = FrontopolarConfig()
    assert cfg.use_frontopolar_analog is False
    # Interface-shape defaults should match the module docstring.
    assert cfg.counterfactual_value_dim == 1
    assert cfg.importance_vector_dim == 2
    assert cfg.gateway_mode == "continuous"
    assert cfg.hidden_dim == 32
    assert cfg.disengagement_scale == 0.1


def test_c3_disabled_methods_return_zero_shapes():
    """C3: when disabled, compute_* methods return correct-shape zeros."""
    from ree_core.pfc import FrontopolarAnalog, FrontopolarConfig
    world_dim = 32
    goal_dim = 16
    action_dim = 4
    k = 3
    batch = 5

    cfg = FrontopolarConfig(
        use_frontopolar_analog=False,
        counterfactual_value_dim=1,
        importance_vector_dim=k,
    )
    fp = FrontopolarAnalog(
        world_dim=world_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        config=cfg,
    )

    z_chosen = torch.randn(batch, world_dim)
    z_alt = torch.randn(batch, world_dim)
    z_goal = torch.randn(batch, goal_dim)
    z_world = torch.randn(batch, world_dim)
    z_goals = torch.randn(batch, k, goal_dim)

    cfv = fp.compute_counterfactual_value(z_chosen, z_alt, z_goal)
    assert cfv.shape == (batch, 1)
    assert torch.all(cfv == 0)

    imp = fp.compute_relative_importance(z_world, z_goals)
    assert imp.shape == (batch, k)
    # Uniform 1/K when disabled.
    assert torch.allclose(imp, torch.full_like(imp, 1.0 / k))

    dis = fp.compute_disengagement_bias(cfv, imp)
    assert dis.shape == (batch,)
    assert torch.all(dis == 0)


def test_c4_enabled_methods_raise_not_implemented():
    """C4: enabling the stub without an implementation raises on compute_*.

    This is the contract that prevents the stub from silently producing
    plausible-looking numbers when accidentally enabled before the design
    doc lands.
    """
    from ree_core.pfc import FrontopolarAnalog, FrontopolarConfig
    cfg = FrontopolarConfig(use_frontopolar_analog=True)
    fp = FrontopolarAnalog(world_dim=32, goal_dim=16, action_dim=4, config=cfg)

    z_chosen = torch.randn(2, 32)
    z_alt = torch.randn(2, 32)
    z_goal = torch.randn(2, 16)
    z_world = torch.randn(2, 32)
    z_goals = torch.randn(2, 2, 16)

    with pytest.raises(NotImplementedError):
        fp.compute_counterfactual_value(z_chosen, z_alt, z_goal)

    with pytest.raises(NotImplementedError):
        fp.compute_relative_importance(z_world, z_goals)

    # Disengagement bias is computed from the outputs of the other two;
    # the no-op path short-circuits when disabled, but when enabled we
    # still raise regardless of inputs (the implementation isn't there).
    dummy_cfv = torch.zeros(2, 1)
    dummy_imp = torch.zeros(2, 2)
    with pytest.raises(NotImplementedError):
        fp.compute_disengagement_bias(dummy_cfv, dummy_imp)


def test_c5_last_linear_zeroed_at_init():
    """C5: zero-init on last Linear of each head (SD-033a contract)."""
    from ree_core.pfc import FrontopolarAnalog, FrontopolarConfig
    fp = FrontopolarAnalog(
        world_dim=32,
        goal_dim=16,
        action_dim=4,
        config=FrontopolarConfig(),
    )
    last_cfv = fp.counterfactual_value_head[-1]
    last_imp = fp.importance_monitor_head[-1]
    assert torch.all(last_cfv.weight == 0)
    assert torch.all(last_cfv.bias == 0)
    assert torch.all(last_imp.weight == 0)
    assert torch.all(last_imp.bias == 0)


def test_c6_reset_is_safe():
    """C6: reset() clears diagnostics and does not raise."""
    from ree_core.pfc import FrontopolarAnalog, FrontopolarConfig
    fp = FrontopolarAnalog(
        world_dim=32,
        goal_dim=16,
        action_dim=4,
        config=FrontopolarConfig(),
    )
    fp._last_counterfactual_value = 0.5
    fp._last_importance_entropy = 0.9
    fp.reset()
    assert fp._last_counterfactual_value == 0.0
    assert fp._last_importance_entropy == 0.0


def test_c7_get_state_marks_stub():
    """C7: get_state() includes the stub=True marker."""
    from ree_core.pfc import FrontopolarAnalog, FrontopolarConfig
    fp = FrontopolarAnalog(
        world_dim=32,
        goal_dim=16,
        action_dim=4,
        config=FrontopolarConfig(),
    )
    state = fp.get_state()
    assert isinstance(state, dict)
    assert state.get("stub") is True
    for key in (
        "last_gate",
        "last_counterfactual_value",
        "last_importance_entropy",
        # SD-033e (5e54781) renamed the de-commit diagnostic key
        # last_disengagement_bias -> last_decommit_pressure in get_state()
        # (the underlying attribute stays _last_disengagement_bias).
        "last_decommit_pressure",
    ):
        assert key in state
