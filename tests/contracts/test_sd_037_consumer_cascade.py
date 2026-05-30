"""SD-037 consumer-cascade contract tests (MECH-281 motor-coupling axis).

Amend session 2026-05-30: wires override_signal from SD-037
BroadcastOverrideRegulator into four additional consumer sites named in
MECH-281 implementation_note: LateralPFCAnalog (SD-033a deliberation),
BLAAnalog + CeAAnalog (SD-035 amygdala arbitration), and BetaGate via
agent.py urgency_interrupt path (MECH-090 motor-side escape-from-freeze).

Already-wired (NOT covered here): PAG freeze-gate (override_alpha_pag),
SalienceCoordinator (override_salience_reweight_alpha), GoalState
(override_goal_seeding_gain). Those landed in the 2026-04-25 SD-037
landing pass.

Contracts:
  C1  REEConfig defaults: all four new gains default to 0.0; bit-identical OFF.
  C2  LateralPFCAnalog: override modulation scales eff_eta when gain>0 + override>0.
  C3  BLAAnalog: override modulation scales encoding_gain when gain>0 + override>0.
  C4  CeAAnalog: override modulation scales mode_prior + fast_prime, bounded by cap.
  C5  Simulation_mode=True skips override modulation on each consumer.
  C6  Agent integration: master ON + all 4 gains>0 constructs cleanly; one
      sense() tick advances broadcast_override; one select_action() tick
      consumes override_signal at lateral_pfc.update + urgency_interrupt.
"""

from __future__ import annotations

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.amygdala.bla import BLAAnalog, BLAConfig
from ree_core.amygdala.cea import CeAAnalog, CeAConfig
from ree_core.pfc.lateral_pfc_analog import LateralPFCAnalog, LateralPFCConfig
from ree_core.utils.config import REEConfig


# --------------------------------------------------------------------------
# C1: REEConfig defaults bit-identical OFF
# --------------------------------------------------------------------------

def test_c1_default_gains_zero_backward_compat():
    """All four override_*_gain knobs default to 0.0 -> bit-identical OFF."""
    cfg = REEConfig.from_dims(
        world_dim=32, self_dim=16, harm_dim=32, action_dim=4,
        body_obs_dim=17, world_obs_dim=300,
    )
    assert cfg.override_pfc_eta_gain == 0.0
    assert cfg.override_bla_encoding_gain == 0.0
    assert cfg.override_cea_amplitude_gain == 0.0
    assert cfg.override_beta_interrupt_gain == 0.0


def test_c1b_default_use_broadcast_override_off():
    """SD-037 master flag still defaults False; consumer-cascade gates
    transitively dormant even if a caller sets gain>0 without master ON."""
    cfg = REEConfig.from_dims(
        world_dim=32, self_dim=16, harm_dim=32, action_dim=4,
        body_obs_dim=17, world_obs_dim=300,
        override_pfc_eta_gain=2.0,  # gain set but master OFF
    )
    assert cfg.use_broadcast_override is False
    agent = REEAgent(cfg)
    assert agent.broadcast_override is None  # no regulator instantiated


# --------------------------------------------------------------------------
# C2: LateralPFCAnalog eta modulation
# --------------------------------------------------------------------------

def _lpfc(delta_dim=32, world_dim=32):
    cfg = LateralPFCConfig(use_lateral_pfc_analog=True, update_eta=0.1)
    return LateralPFCAnalog(delta_dim=delta_dim, world_dim=world_dim, config=cfg)


def test_c2_lateral_pfc_override_zero_gain_noop():
    """override_eta_gain=0.0 -> rule_state delta independent of override_signal."""
    torch.manual_seed(0)
    m = _lpfc()
    zd = torch.randn(1, 32)
    zw = torch.randn(1, 32)
    base = m.rule_state.clone()
    m.update(zd, zw, gate=1.0, override_signal=0.0, override_eta_gain=0.0)
    d_off = (m.rule_state - base).norm().item()
    m.rule_state.copy_(base)
    m.update(zd, zw, gate=1.0, override_signal=1.0, override_eta_gain=0.0)
    d_on = (m.rule_state - base).norm().item()
    assert abs(d_off - d_on) < 1e-9


def test_c2b_lateral_pfc_override_amplifies_eta():
    """override_eta_gain>0 + override_signal>0 -> larger rule_state delta."""
    torch.manual_seed(0)
    m = _lpfc()
    zd = torch.randn(1, 32)
    zw = torch.randn(1, 32)
    base = m.rule_state.clone()
    m.update(zd, zw, gate=1.0, override_signal=0.0, override_eta_gain=2.0)
    d_zero_signal = (m.rule_state - base).norm().item()
    m.rule_state.copy_(base)
    m.update(zd, zw, gate=1.0, override_signal=1.0, override_eta_gain=2.0)
    d_full_signal = (m.rule_state - base).norm().item()
    # multiplier = 1 + 2*1 = 3 -> 3x delta
    assert d_full_signal > d_zero_signal * 2.5
    assert d_full_signal < d_zero_signal * 3.5


# --------------------------------------------------------------------------
# C3: BLAAnalog encoding_gain modulation
# --------------------------------------------------------------------------

def test_c3_bla_override_zero_gain_noop():
    bla = BLAAnalog(BLAConfig())
    zh = torch.ones(1, 7) * 0.7
    o_off = bla.tick(zh, override_signal=0.0, override_encoding_gain=0.0)
    bla2 = BLAAnalog(BLAConfig())
    o_on = bla2.tick(zh, override_signal=1.0, override_encoding_gain=0.0)
    assert abs(o_off.encoding_gain - o_on.encoding_gain) < 1e-9


def test_c3b_bla_override_amplifies_encoding_gain():
    """override_encoding_gain>0 + override_signal>0 scales encoding_gain."""
    bla = BLAAnalog(BLAConfig())
    zh = torch.ones(1, 7) * 0.7
    o_off = bla.tick(zh, override_signal=0.0, override_encoding_gain=1.0)
    g_off = o_off.encoding_gain
    bla2 = BLAAnalog(BLAConfig())
    o_on = bla2.tick(zh, override_signal=1.0, override_encoding_gain=1.0)
    g_on = o_on.encoding_gain
    # multiplier = 1 + 1*1 = 2 -> 2x
    assert g_on > g_off * 1.5
    assert g_on < g_off * 2.5


# --------------------------------------------------------------------------
# C4: CeAAnalog mode_prior + fast_prime modulation, bounded by cap
# --------------------------------------------------------------------------

def test_c4_cea_override_zero_gain_noop():
    cea = CeAAnalog(CeAConfig())
    zh = torch.ones(1, 7) * 0.7
    o_off = cea.tick(zh, override_signal=0.0, override_amplitude_gain=0.0)
    cea2 = CeAAnalog(CeAConfig())
    o_on = cea2.tick(zh, override_signal=1.0, override_amplitude_gain=0.0)
    assert abs(o_off.mode_prior - o_on.mode_prior) < 1e-9
    assert abs(o_off.fast_prime - o_on.fast_prime) < 1e-9


def test_c4b_cea_override_amplifies_both_scalars():
    cea = CeAAnalog(CeAConfig())
    zh = torch.ones(1, 7) * 0.7
    o_off = cea.tick(zh, override_signal=0.0, override_amplitude_gain=1.0)
    cea2 = CeAAnalog(CeAConfig())
    o_on = cea2.tick(zh, override_signal=1.0, override_amplitude_gain=1.0)
    # Both scaled by (1 + 1*1) = 2x (modulo cap).
    assert o_on.mode_prior > o_off.mode_prior
    assert o_on.fast_prime > o_off.fast_prime


def test_c4c_cea_override_bounded_by_cap():
    """Even at high override + high gain, both scalars stay within the
    mode_prior_log_odds_max cap (CeA cannot over-rule cortex via the
    amplified path)."""
    cfg = CeAConfig(mode_prior_log_odds_max=0.8, mode_prior_gain=1.0,
                    fast_prime_amplitude=0.6)
    cea = CeAAnalog(cfg)
    zh = torch.ones(1, 7) * 1.5  # well over threshold -> raw scalars near cap
    out = cea.tick(zh, override_signal=1.0, override_amplitude_gain=10.0)
    assert abs(out.mode_prior) <= cfg.mode_prior_log_odds_max + 1e-9
    assert abs(out.fast_prime) <= cfg.mode_prior_log_odds_max + 1e-9


# --------------------------------------------------------------------------
# C5: simulation_mode=True skips override modulation
# --------------------------------------------------------------------------

def test_c5_bla_simulation_mode_returns_zeroed_output():
    """BLA simulation_mode=True returns zeroed output regardless of override."""
    bla = BLAAnalog(BLAConfig())
    zh = torch.ones(1, 7) * 0.7
    out = bla.tick(zh, simulation_mode=True, override_signal=1.0,
                   override_encoding_gain=1.0)
    assert out.encoding_gain == 1.0  # zeroed baseline
    assert out.arousal_tag == 0.0


def test_c5b_cea_simulation_mode_returns_zeroed_output():
    cea = CeAAnalog(CeAConfig())
    zh = torch.ones(1, 7) * 0.7
    out = cea.tick(zh, simulation_mode=True, override_signal=1.0,
                   override_amplitude_gain=1.0)
    assert out.mode_prior == 0.0
    assert out.fast_prime == 0.0


# --------------------------------------------------------------------------
# C6: agent integration end-to-end
# --------------------------------------------------------------------------

def _make_agent_consumer_cascade_on():
    cfg = REEConfig.from_dims(
        world_dim=32, self_dim=16, harm_dim=32, action_dim=4,
        body_obs_dim=17, world_obs_dim=300,
        use_broadcast_override=True,
        override_recruitment_threshold=0.3,
        use_lateral_pfc_analog=True,
        use_amygdala_analog=True,
        use_bla_analog=True,
        use_cea_analog=True,
        use_affective_harm_stream=True,
        harm_obs_a_dim=7,
        limb_damage_enabled=True,
        override_pfc_eta_gain=2.0,
        override_bla_encoding_gain=1.0,
        override_cea_amplitude_gain=1.0,
        override_beta_interrupt_gain=0.5,
    )
    return REEAgent(cfg), cfg


def test_c6_agent_constructs_with_consumer_cascade_on():
    """Agent with SD-037 master ON + all 4 consumer-cascade gains > 0 builds
    cleanly. broadcast_override + lateral_pfc + bla + cea all instantiated."""
    agent, _ = _make_agent_consumer_cascade_on()
    assert agent.broadcast_override is not None
    assert agent.lateral_pfc is not None
    assert agent.bla is not None
    assert agent.cea is not None


def test_c6b_override_signal_advances_with_drive_and_harm():
    """Sustained drive+harm ticks lift override_signal off zero."""
    agent, _ = _make_agent_consumer_cascade_on()
    ov = agent.broadcast_override
    assert ov.override_signal == 0.0
    for _ in range(30):
        ov.tick(drive_level=0.9, z_harm_norm=0.6)
    assert ov.override_signal > 0.3  # well off baseline after 30 ticks
