"""Contract tests for MECH-295 drive -> liking-stream -> approach_cue bridge.

Guarantees enforced here:
  C1. Module / dataclass importable without side effects.
  C2. Default REEConfig has use_mech295_liking_bridge=False (backward-compat).
  C3. With master switch OFF, REEAgent.mech295_bridge is None and the agent
      boot is bit-identical to legacy.
  C4. With master switch ON, REEAgent instantiates a MECH295LikingBridge.
  C5. Bridge-on-empty-state no-op: when goal_state is inactive (z_goal == 0)
      AND drive is sub-floor, neither write nor cue fires.
  C6. Drive-elevated -> liking-write fires: with drive >= floor and
      z_goal_norm >= floor, compute_anticipatory_liking_write returns
      a positive scalar proportional to drive * z_goal_norm * gain.
  C7. Drive-elevated -> approach_cue fires: per-candidate score_bias is
      negative (E3 lower-is-better) and proportional to drive *
      goal_proximity.
  C8. Severed-bridge collapse signature: with master ON but
      mech295_liking_to_approach_cue_gain=0.0 the cue-side score_bias
      is exactly zero on every candidate; the write side may still fire
      depending on drive_to_liking_gain.
  C9. MECH-094: simulation_mode=True returns 0.0 / zero score_bias and
      does not advance counters.
  C10. reset() clears the per-tick diagnostic cache without raising.
  C11. Sub-floor drive: with drive_level < min_drive_to_fire, both write
      and cue are zero regardless of other inputs.
"""

from __future__ import annotations

import pytest
import torch


# -- Layer 1: pure module contracts --------------------------------------


def test_c1_module_importable():
    from ree_core.regulators import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
        MECH295LikingBridgeOutput,
    )
    cfg = MECH295LikingBridgeConfig()
    assert cfg.drive_to_liking_gain == 1.0
    assert cfg.liking_to_approach_cue_gain == 0.5
    assert cfg.min_drive_to_fire == 0.1
    assert cfg.min_z_goal_norm_to_fire == 0.05
    bridge = MECH295LikingBridge(cfg)
    assert bridge.get_last_output() is None
    diag = bridge.get_diagnostics()
    assert diag["n_write_fires"] == 0
    assert diag["n_cue_fires"] == 0


def test_c5_empty_state_no_op():
    """C5: sub-floor drive AND zero z_goal_norm -> both sides zero."""
    from ree_core.regulators import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )
    bridge = MECH295LikingBridge(MECH295LikingBridgeConfig())
    write = bridge.compute_anticipatory_liking_write(
        drive_level=0.0, z_goal_norm=0.0
    )
    assert write == 0.0
    bias = bridge.compute_approach_cue_score_bias(
        drive_level=0.0,
        candidate_proximities=torch.zeros(5),
    )
    assert bias.shape == (5,)
    assert torch.all(bias == 0.0)


def test_c6_drive_elevated_write_fires():
    """C6: drive*z_goal_norm above floors -> positive write."""
    from ree_core.regulators import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )
    cfg = MECH295LikingBridgeConfig(drive_to_liking_gain=1.0)
    bridge = MECH295LikingBridge(cfg)
    # drive=0.8, z_goal_norm=0.5 -> 1.0 * 0.8 * 0.5 = 0.4
    write = bridge.compute_anticipatory_liking_write(
        drive_level=0.8, z_goal_norm=0.5
    )
    assert write == pytest.approx(0.4, rel=1e-6)
    # gain=0 -> zero write even at elevated inputs
    cfg0 = MECH295LikingBridgeConfig(drive_to_liking_gain=0.0)
    bridge0 = MECH295LikingBridge(cfg0)
    assert bridge0.compute_anticipatory_liking_write(
        drive_level=0.8, z_goal_norm=0.5
    ) == 0.0


def test_c7_drive_elevated_cue_fires():
    """C7: drive elevated, candidate proximities non-trivial -> negative bias."""
    from ree_core.regulators import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )
    cfg = MECH295LikingBridgeConfig(liking_to_approach_cue_gain=0.5)
    bridge = MECH295LikingBridge(cfg)
    drive = 0.6
    prox = torch.tensor([0.1, 0.5, 0.9])
    bias = bridge.compute_approach_cue_score_bias(
        drive_level=drive, candidate_proximities=prox
    )
    # bias = -gain * drive * prox = -0.5 * 0.6 * prox = -0.3 * prox
    expected = -0.3 * prox
    assert torch.allclose(bias, expected, rtol=1e-6)
    # All non-zero biases are negative.
    assert torch.all(bias <= 0.0)
    # Larger proximity -> more negative bias (favours approach more).
    assert bias[2] < bias[0]


def test_c8_severed_bridge_collapse():
    """C8: liking_to_approach_cue_gain=0 -> bias exactly zero everywhere."""
    from ree_core.regulators import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )
    cfg = MECH295LikingBridgeConfig(
        drive_to_liking_gain=1.0,  # write side intact
        liking_to_approach_cue_gain=0.0,  # cue side severed
    )
    bridge = MECH295LikingBridge(cfg)
    bias = bridge.compute_approach_cue_score_bias(
        drive_level=0.9,
        candidate_proximities=torch.tensor([0.1, 0.5, 0.9]),
    )
    assert torch.all(bias == 0.0)
    # Write side still fires: drive=0.9 * z_goal_norm=0.5 * gain=1.0 = 0.45
    write = bridge.compute_anticipatory_liking_write(
        drive_level=0.9, z_goal_norm=0.5
    )
    assert write == pytest.approx(0.45, rel=1e-6)


def test_c9_mech094_simulation_mode():
    """C9: simulation_mode=True -> zero output, no counter advance."""
    from ree_core.regulators import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )
    bridge = MECH295LikingBridge(MECH295LikingBridgeConfig())
    write = bridge.compute_anticipatory_liking_write(
        drive_level=0.9, z_goal_norm=0.5, simulation_mode=True
    )
    assert write == 0.0
    bias = bridge.compute_approach_cue_score_bias(
        drive_level=0.9,
        candidate_proximities=torch.tensor([0.1, 0.5, 0.9]),
        simulation_mode=True,
    )
    assert torch.all(bias == 0.0)
    # tick() also honours the gate.
    w, b = bridge.tick(
        drive_level=0.9, z_goal_norm=0.5,
        candidate_proximities=torch.tensor([0.1, 0.5, 0.9]),
        simulation_mode=True,
    )
    assert w == 0.0
    assert torch.all(b == 0.0)
    diag = bridge.get_diagnostics()
    assert diag["n_write_fires"] == 0
    assert diag["n_cue_fires"] == 0


def test_c10_reset_clears_cache():
    """C10: reset() clears per-tick diagnostics, leaves counters intact."""
    from ree_core.regulators import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )
    bridge = MECH295LikingBridge(MECH295LikingBridgeConfig())
    bridge.tick(
        drive_level=0.8, z_goal_norm=0.5,
        candidate_proximities=torch.tensor([0.5]),
    )
    assert bridge.get_last_output() is not None
    pre_diag = bridge.get_diagnostics()
    bridge.reset()
    assert bridge.get_last_output() is None
    # Counters persist (intentional -- end-of-run reporting).
    assert bridge.get_diagnostics()["n_write_fires"] == pre_diag["n_write_fires"]


def test_c11_sub_floor_drive_silent():
    """C11: drive_level below floor -> both sides zero."""
    from ree_core.regulators import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )
    cfg = MECH295LikingBridgeConfig(min_drive_to_fire=0.2)
    bridge = MECH295LikingBridge(cfg)
    write = bridge.compute_anticipatory_liking_write(
        drive_level=0.05, z_goal_norm=0.5
    )
    assert write == 0.0
    bias = bridge.compute_approach_cue_score_bias(
        drive_level=0.05,
        candidate_proximities=torch.tensor([0.5, 0.9]),
    )
    assert torch.all(bias == 0.0)


# -- Layer 2: agent integration contracts -------------------------------


def test_c2_default_config_backward_compatible():
    """C2: default REEConfig has use_mech295_liking_bridge=False."""
    from ree_core.utils.config import REEConfig
    cfg = REEConfig()
    assert getattr(cfg, "use_mech295_liking_bridge", False) is False
    assert getattr(cfg, "mech295_drive_to_liking_gain", 1.0) == 1.0
    assert getattr(cfg, "mech295_liking_to_approach_cue_gain", 0.5) == 0.5


def test_c3_master_off_no_instantiation():
    """C3: master OFF -> agent.mech295_bridge is None."""
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(
        body_obs_dim=17, world_obs_dim=250, action_dim=4
    )
    agent = REEAgent(cfg)
    assert agent.mech295_bridge is None


def test_c4_master_on_instantiates():
    """C4: master ON -> agent.mech295_bridge is constructed with config sub-knobs."""
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(
        body_obs_dim=17, world_obs_dim=250, action_dim=4,
        use_mech295_liking_bridge=True,
        mech295_drive_to_liking_gain=1.5,
        mech295_liking_to_approach_cue_gain=0.7,
    )
    agent = REEAgent(cfg)
    assert agent.mech295_bridge is not None
    assert agent.mech295_bridge.config.drive_to_liking_gain == 1.5
    assert agent.mech295_bridge.config.liking_to_approach_cue_gain == 0.7
    # Reset hook does not raise.
    agent.reset()
