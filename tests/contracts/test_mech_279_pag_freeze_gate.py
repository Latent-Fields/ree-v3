"""Contract tests for MECH-279 PAG freeze-gate.

Guarantees enforced here:
  C1. Module + dataclass importable without side effects.
  C2. Default REEConfig has use_pag_freeze_gate=False (backward-compat).
  C3. With master switch OFF, REEAgent.pag_freeze_gate is None.
  C4. Freeze commit logic: z_harm_a_norm * duration_above_threshold > theta_freeze.
  C5. Freeze exit logic: z_harm_a_norm < theta_freeze * gaba_tone (after min_freeze_duration).
  C6. gaba_tone modulation: higher tone -> higher exit threshold -> easier exit.
  C7. MECH-094: simulation_mode=True returns zeroed output without state update.
  C8. reset() clears per-episode state.
  C9. Duration counter resets when z_harm_a falls below duration_input_threshold.
  C10. max_freeze_duration cap forces release.
"""

from __future__ import annotations

import pytest


def test_c1_module_importable():
    from ree_core.pag import (
        PAGFreezeGate,
        PAGFreezeGateConfig,
        PAGFreezeGateOutput,
    )
    assert PAGFreezeGateConfig().enabled is True


def test_c2_default_config_backward_compatible():
    from ree_core.utils.config import REEConfig
    cfg = REEConfig()
    assert getattr(cfg, "use_pag_freeze_gate", False) is False
    assert cfg.pag_theta_freeze == 2.0
    assert cfg.pag_duration_input_threshold == 0.4


def test_c3_master_switch_off_no_instantiation():
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    agent = REEAgent(cfg)
    assert agent.pag_freeze_gate is None


def test_c4_freeze_commit_logic():
    """C4: freeze_commit = (z * duration) > theta_freeze."""
    from ree_core.pag import PAGFreezeGate, PAGFreezeGateConfig

    # theta_freeze=2.0, duration_input_threshold=0.4 -> z=1.0 needs duration>=2.
    cfg = PAGFreezeGateConfig(
        enabled=True,
        theta_freeze=2.0,
        duration_input_threshold=0.4,
        min_freeze_duration=0,
        max_freeze_duration=0,
    )
    gate = PAGFreezeGate(cfg)

    # Tick 1: z=1.0 above input_threshold -> duration=1, z*dur=1.0 < 2.0 -> no commit.
    out1 = gate.tick(z_harm_a_norm=1.0, gaba_tone=1.0)
    assert out1.duration_above_threshold == 1
    assert out1.freeze_commit is False
    assert out1.freeze_active is False

    # Tick 2: z=1.0, duration=2, z*dur=2.0 NOT > 2.0 -> still no commit (strict >).
    out2 = gate.tick(z_harm_a_norm=1.0, gaba_tone=1.0)
    assert out2.duration_above_threshold == 2
    assert out2.freeze_commit is False

    # Tick 3: z=1.0, duration=3, z*dur=3.0 > 2.0 -> commit.
    out3 = gate.tick(z_harm_a_norm=1.0, gaba_tone=1.0)
    assert out3.duration_above_threshold == 3
    assert out3.freeze_commit is True
    assert out3.freeze_active is True


def test_c5_freeze_exit_logic():
    """C5: freeze exits when z_harm_a < theta_freeze * gaba_tone."""
    from ree_core.pag import PAGFreezeGate, PAGFreezeGateConfig

    cfg = PAGFreezeGateConfig(
        enabled=True,
        theta_freeze=2.0,
        duration_input_threshold=0.4,
        min_freeze_duration=0,
    )
    gate = PAGFreezeGate(cfg)

    # Force commit by sustained high input. theta=2, exit_thresh = 2 * 1.0 = 2.0.
    for _ in range(5):
        gate.tick(z_harm_a_norm=1.5, gaba_tone=1.0)
    assert gate.is_active is True

    # Now drop z below exit_threshold (2.0) -> exit.
    out = gate.tick(z_harm_a_norm=1.0, gaba_tone=1.0)
    assert out.freeze_release is True
    assert out.freeze_active is False
    assert gate.is_active is False


def test_c6_gaba_tone_modulates_exit():
    """C6: higher gaba_tone raises exit_threshold -> easier exit."""
    from ree_core.pag import PAGFreezeGate, PAGFreezeGateConfig

    cfg = PAGFreezeGateConfig(
        enabled=True, theta_freeze=2.0,
        duration_input_threshold=0.4, min_freeze_duration=0,
    )

    def _exit_at_z(z, tone):
        gate = PAGFreezeGate(cfg)
        for _ in range(5):
            gate.tick(z_harm_a_norm=1.5, gaba_tone=tone)
        # Now test exit at given z.
        out = gate.tick(z_harm_a_norm=z, gaba_tone=tone)
        return out.freeze_active

    # At tone=1.0, exit_thresh=2.0; z=1.5 < 2.0 -> exits.
    # At tone=0.5, exit_thresh=1.0; z=1.5 > 1.0 -> stays in freeze.
    # At tone=2.0, exit_thresh=4.0; z=3.0 < 4.0 -> exits.
    assert _exit_at_z(z=1.5, tone=1.0) is False  # exits
    assert _exit_at_z(z=1.5, tone=0.5) is True   # stuck
    assert _exit_at_z(z=3.0, tone=2.0) is False  # benzo accelerates exit


def test_c7_mech094_simulation_mode():
    """C7: simulation_mode=True returns zeroed output without state update."""
    from ree_core.pag import PAGFreezeGate, PAGFreezeGateConfig

    cfg = PAGFreezeGateConfig(enabled=True, theta_freeze=2.0)
    gate = PAGFreezeGate(cfg)
    out = gate.tick(z_harm_a_norm=10.0, gaba_tone=1.0, simulation_mode=True)
    assert out.freeze_active is False
    assert out.freeze_commit is False
    # Diagnostic n_ticks counter should NOT advance.
    assert gate.diagnostics["n_ticks"] == 0
    assert gate.diagnostics["duration_above_threshold"] == 0


def test_c8_reset_clears_state():
    """C8: reset() clears freeze state and counters."""
    from ree_core.pag import PAGFreezeGate, PAGFreezeGateConfig

    cfg = PAGFreezeGateConfig(enabled=True, theta_freeze=2.0)
    gate = PAGFreezeGate(cfg)
    for _ in range(5):
        gate.tick(z_harm_a_norm=1.5, gaba_tone=1.0)
    assert gate.is_active is True
    gate.reset()
    assert gate.is_active is False
    assert gate.diagnostics["duration_above_threshold"] == 0
    assert gate.diagnostics["ticks_in_freeze"] == 0


def test_c9_duration_resets_below_threshold():
    """C9: duration counter resets when z falls below duration_input_threshold."""
    from ree_core.pag import PAGFreezeGate, PAGFreezeGateConfig

    cfg = PAGFreezeGateConfig(
        enabled=True, theta_freeze=10.0,  # high enough that we don't commit
        duration_input_threshold=0.4,
    )
    gate = PAGFreezeGate(cfg)
    for _ in range(3):
        gate.tick(z_harm_a_norm=1.0, gaba_tone=1.0)
    assert gate.diagnostics["duration_above_threshold"] == 3
    # Drop below threshold.
    out = gate.tick(z_harm_a_norm=0.1, gaba_tone=1.0)
    assert out.duration_above_threshold == 0


def test_c10_max_freeze_duration_cap():
    """C10: max_freeze_duration forces release once reached."""
    from ree_core.pag import PAGFreezeGate, PAGFreezeGateConfig

    cfg = PAGFreezeGateConfig(
        enabled=True, theta_freeze=2.0,
        duration_input_threshold=0.4,
        max_freeze_duration=5,
    )
    gate = PAGFreezeGate(cfg)
    # Commit then sustain high input.
    for _ in range(10):
        out = gate.tick(z_harm_a_norm=10.0, gaba_tone=1.0)
        if out.freeze_release:
            # Released by max-cap.
            assert gate.diagnostics["n_releases"] == 1
            return
    pytest.fail("max_freeze_duration cap did not force release")


def test_backward_compat_agent_boot():
    """Backward-compat: REEAgent boot with default config has no PAG gate."""
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    agent = REEAgent(cfg)
    assert agent.pag_freeze_gate is None
    agent.reset()
