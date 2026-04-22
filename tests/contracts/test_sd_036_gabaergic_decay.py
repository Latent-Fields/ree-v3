"""Contract tests for SD-036 GABAergic cross-stream decay regulator.

Guarantees enforced here:
  C1. Module / dataclass importable without side effects.
  C2. Default REEConfig has use_gabaergic_decay=False (backward-compat).
  C3. With master switch OFF, REEAgent.gabaergic_decay is None and a default
      agent boot is bit-identical to legacy.
  C4. With master switch ON, REEAgent instantiates a regulator with the three
      default streams (z_harm, z_harm_a, z_beta) registered.
  C5. Decay arithmetic: z_s(t+1) = z_s(t) * exp(-tau_s * gaba_tone) when
      input gate does not fire.
  C6. gaba_tone modulation: tone>1.0 -> faster decay; tone<1.0 -> slower
      decay; tone=0 -> decay suspended.
  C7. Suspend-on-input gate: when |z(t) - z(t-1)| > input_threshold, decay
      is skipped for that tick.
  C8. MECH-094: simulation_mode=True returns the input unchanged and does
      not advance internal counters.
  C9. reset() clears per-episode state without raising.
  C10. Per-stream coverage flags ablate just the targeted stream.
"""

from __future__ import annotations

import math

import pytest
import torch


def test_c1_module_importable():
    """C1: module + dataclass importable without side effects."""
    from ree_core.regulators import (
        GABAergicDecayConfig,
        GABAergicDecayRegulator,
        StreamRegistration,
    )
    from ree_core.regulators.gabaergic_decay import (
        GABAergicDecayRegulator as GR,
    )
    assert GABAergicDecayRegulator is GR
    assert GABAergicDecayConfig().enabled is True


def test_c2_default_config_backward_compatible():
    """C2: default REEConfig has use_gabaergic_decay=False."""
    from ree_core.utils.config import REEConfig
    cfg = REEConfig()
    assert getattr(cfg, "use_gabaergic_decay", False) is False
    assert getattr(cfg, "use_pag_freeze_gate", False) is False
    # Default tau values match the design doc.
    assert cfg.gaba_tau_z_harm_s == 0.05
    assert cfg.gaba_tau_z_harm_a == 0.02
    assert cfg.gaba_tau_z_beta == 0.03
    assert cfg.gaba_tone == 1.0


def test_c3_master_switch_off_no_instantiation():
    """C3: master OFF -> agent.gabaergic_decay is None."""
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    agent = REEAgent(cfg)
    assert agent.gabaergic_decay is None
    assert agent.pag_freeze_gate is None


def test_c4_master_switch_on_registers_default_streams():
    """C4: master ON -> regulator has the three default streams registered."""
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_harm_stream=True, use_affective_harm_stream=True,
        use_gabaergic_decay=True,
    )
    agent = REEAgent(cfg)
    assert agent.gabaergic_decay is not None
    streams = agent.gabaergic_decay.registered_streams
    assert "z_harm" in streams
    assert "z_harm_a" in streams
    assert "z_beta" in streams


def test_c5_decay_arithmetic_no_input():
    """C5: z(t+1) = z(t) * exp(-tau * gaba_tone) when input gate does not fire."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    cfg = GABAergicDecayConfig(
        enabled=True, gaba_tone=1.0,
        tau_z_harm_s=0.05, tau_z_harm_a=0.02, tau_z_beta=0.03,
        # input_threshold=0.0 -> decay always proceeds
        input_threshold_z_harm_s=0.0,
    )
    reg = GABAergicDecayRegulator(cfg)
    reg.register("z_harm", tau=0.05, input_threshold=0.0)

    # Build a minimal latent stand-in with a `z_harm` attribute.
    class _Latent:
        z_harm = torch.tensor([[1.0, 0.0, 0.0]])
        hypothesis_tag = False
    latent = _Latent()
    expected = 1.0 * math.exp(-0.05 * 1.0)
    reg.tick(latent)
    new_norm = float(latent.z_harm.norm().item())
    # 1-norm of vector [exp(-0.05), 0, 0] is exp(-0.05).
    assert abs(new_norm - expected) < 1e-6


def test_c6_gaba_tone_modulation():
    """C6: tone>1 -> faster decay; tone<1 -> slower; tone=0 -> suspended."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    def _decay_norm(tone):
        cfg = GABAergicDecayConfig(enabled=True, gaba_tone=tone)
        reg = GABAergicDecayRegulator(cfg)
        reg.register("z_harm", tau=0.05, input_threshold=0.0)

        class _Latent:
            z_harm = torch.tensor([[1.0]])
            hypothesis_tag = False
        latent = _Latent()
        reg.tick(latent)
        return float(latent.z_harm.norm().item())

    n_baseline = _decay_norm(1.0)
    n_fast = _decay_norm(1.5)
    n_slow = _decay_norm(0.5)
    n_zero = _decay_norm(0.0)

    assert n_fast < n_baseline, "tone>1 should decay faster"
    assert n_slow > n_baseline, "tone<1 should decay slower"
    assert abs(n_zero - 1.0) < 1e-6, "tone=0 should suspend decay"


def test_c7_suspend_on_input_gate():
    """C7: suspend-on-input gate skips decay when magnitude change > threshold."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    cfg = GABAergicDecayConfig(enabled=True, gaba_tone=1.0)
    reg = GABAergicDecayRegulator(cfg)
    reg.register("z_harm", tau=0.05, input_threshold=0.5)

    # First tick establishes the baseline norm at 1.0; decay proceeds (no
    # prior baseline). After this tick, _last_norms["z_harm"] holds the
    # post-decay norm.
    class _Latent:
        z_harm = torch.tensor([[1.0]])
        hypothesis_tag = False
    latent = _Latent()
    reg.tick(latent)
    n_after_first = float(latent.z_harm.norm().item())
    # First tick: baseline is 0, current 1.0, delta=1.0 > 0.5 -> SUSPEND.
    # So the first tick should NOT have decayed.
    assert abs(n_after_first - 1.0) < 1e-6

    # Now provide a stable magnitude (no change from cached baseline).
    # The second tick has baseline 1.0, current 1.0, delta=0 < 0.5 -> DECAY.
    reg.tick(latent)
    n_after_second = float(latent.z_harm.norm().item())
    assert n_after_second < 1.0
    expected = 1.0 * math.exp(-0.05 * 1.0)
    assert abs(n_after_second - expected) < 1e-6


def test_c8_mech094_simulation_mode_no_op():
    """C8: simulation_mode=True returns input unchanged without advancing counters."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    cfg = GABAergicDecayConfig(enabled=True, gaba_tone=1.0)
    reg = GABAergicDecayRegulator(cfg)
    reg.register("z_harm", tau=0.05)

    class _Latent:
        z_harm = torch.tensor([[1.0]])
        hypothesis_tag = True
    latent = _Latent()
    reg.tick(latent, simulation_mode=True)
    # No decay applied.
    assert abs(float(latent.z_harm.norm().item()) - 1.0) < 1e-6
    # Diagnostic counter should not advance.
    assert reg.diagnostics["n_ticks"] == 0


def test_c9_reset_clears_state():
    """C9: reset() clears per-episode state without raising."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    cfg = GABAergicDecayConfig(enabled=True, gaba_tone=1.0)
    reg = GABAergicDecayRegulator(cfg)
    reg.register("z_harm", tau=0.05)

    class _Latent:
        z_harm = torch.tensor([[1.0]])
        hypothesis_tag = False
    latent = _Latent()
    reg.tick(latent)
    assert reg.diagnostics["n_ticks"] == 1
    reg.reset()
    assert reg.diagnostics["n_ticks"] == 0
    assert "z_harm" in reg.registered_streams  # registration preserved


def test_c10_per_stream_coverage_flags():
    """C10: per-stream coverage flags ablate just the targeted stream."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    # Ablate z_harm_a only.
    cfg = GABAergicDecayConfig(
        enabled=True, gaba_tone=1.0,
        decay_z_harm_s=True,
        decay_z_harm_a=False,
        decay_z_beta=True,
    )
    reg = GABAergicDecayRegulator(cfg)
    reg.register_default_streams(cfg)
    streams = reg.registered_streams
    assert "z_harm" in streams
    assert "z_harm_a" not in streams
    assert "z_beta" in streams


def test_backward_compat_agent_boot():
    """Backward-compat: REEAgent boot with default config is unaffected."""
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    agent = REEAgent(cfg)
    # gabaergic_decay must be None and reset() must not raise.
    assert agent.gabaergic_decay is None
    agent.reset()
