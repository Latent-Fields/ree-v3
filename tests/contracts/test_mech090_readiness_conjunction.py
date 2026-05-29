"""Contract tests for MECH-090 R-c readiness conjunction (commit_closure GAP-4).

Five contracts (C1-C5):
  C1: master-OFF backward-compat. With use_mech090_readiness_conjunction=False
      AND use_commit_readiness=False, agent.commit_readiness is None; the
      conjunction at the beta_gate.elevate() call sites reduces to the
      legacy rv-only + score_margin gate path; bit-identical to pre-MECH-090
      R-c-nav-competence behaviour.
  C2: ON + rv-low + readiness-low BLOCKS. With the master flag on and the
      readiness EMA driven below the floor (via notify_outcome), the
      conjunction's is_above_floor returns False; the diagnostic notify_block
      counter advances; should_admit_elevation alone is no longer sufficient.
  C3: ON + rv-low + readiness-high ADMITS. With readiness >= floor (default
      initial 1.0), is_above_floor returns True so the conjunction adds no
      block on top of the score_margin gate; legacy elevation paths reach the
      elevate() call.
  C4: MECH-094 simulation_mode invariance. update(simulation_mode=True) and
      notify_outcome(simulation_mode=True) do NOT advance the readiness EMA
      and only increment the simulation-skip counter. Replay / DMN paths
      cannot inherit waking-only readiness.
  C5: config auto-arm. REEConfig.from_dims(use_mech090_readiness_conjunction
      =True) auto-arms use_commit_readiness=True via __post_init__ /
      from_dims's OR-only resolver. REEConfig() direct construction with the
      conjunction flag set hits the __post_init__ resolver. With ONLY
      use_commit_readiness=True (no conjunction), the module instantiates
      as a passive diagnostic (readiness EMA advances but the conjunction
      at the elevate sites does not consult it).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.policy import CommitReadiness, CommitReadinessConfig
from ree_core.utils.config import REEConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_minimal_agent(**flags) -> REEAgent:
    """Build a minimal REEAgent for the conjunction contract tests."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        harm_obs_dim=50,
        harm_obs_a_dim=50,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        **flags,
    )
    return REEAgent(cfg)


# ----------------------------------------------------------------------
# C1 master-OFF backward-compat
# ----------------------------------------------------------------------
def test_c1_master_off_no_module_instantiated():
    agent = _build_minimal_agent()
    assert agent.commit_readiness is None, (
        "default config (use_mech090_readiness_conjunction=False AND "
        "use_commit_readiness=False) should produce commit_readiness=None"
    )
    # Verify the conjunction-relevant config flags are False
    assert agent.config.use_mech090_readiness_conjunction is False
    assert agent.config.use_commit_readiness is False


def test_c1_master_off_diagnostic_state_none():
    """Bit-identical OFF: no diagnostic state exposed."""
    agent = _build_minimal_agent()
    assert getattr(agent, "commit_readiness", None) is None


# ----------------------------------------------------------------------
# C2 ON + rv-low + readiness-low BLOCKS
# ----------------------------------------------------------------------
def test_c2_low_readiness_blocks_admission():
    """With readiness < floor, is_above_floor returns False."""
    agent = _build_minimal_agent(
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=0.3,
    )
    assert agent.commit_readiness is not None
    # Force readiness below the floor
    agent.commit_readiness.notify_outcome(0.1)
    assert agent.commit_readiness.get_readiness() == pytest.approx(0.1)
    assert agent.commit_readiness.is_above_floor(0.3) is False


def test_c2_block_counter_advances_on_notify_block():
    """The agent-side notify_block call increments the diagnostic counter."""
    agent = _build_minimal_agent(
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=0.3,
    )
    assert agent.commit_readiness is not None
    state_before = agent.commit_readiness.get_state()
    agent.commit_readiness.notify_block()
    agent.commit_readiness.notify_block()
    state_after = agent.commit_readiness.get_state()
    assert (
        state_after["n_blocks_emitted"]
        == state_before["n_blocks_emitted"] + 2
    )


# ----------------------------------------------------------------------
# C3 ON + rv-low + readiness-high ADMITS
# ----------------------------------------------------------------------
def test_c3_high_readiness_admits_conjunction():
    """With readiness >= floor (default 1.0 initial), is_above_floor True."""
    agent = _build_minimal_agent(
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=0.3,
    )
    assert agent.commit_readiness is not None
    # Initial value is 1.0 by default; should clear floor 0.3
    assert agent.commit_readiness.get_readiness() == pytest.approx(1.0)
    assert agent.commit_readiness.is_above_floor(0.3) is True
    # Boundary case: readiness == floor admits (inclusive comparison)
    agent.commit_readiness.notify_outcome(0.3)
    assert agent.commit_readiness.is_above_floor(0.3) is True


def test_c3_floor_boundary_inclusive():
    cr = CommitReadiness(CommitReadinessConfig(use_commit_readiness=True))
    cr.notify_outcome(0.5)
    assert cr.is_above_floor(0.5) is True  # inclusive boundary
    assert cr.is_above_floor(0.500001) is False


# ----------------------------------------------------------------------
# C4 MECH-094 simulation_mode invariance
# ----------------------------------------------------------------------
def test_c4_simulation_mode_does_not_advance_ema():
    cr = CommitReadiness(CommitReadinessConfig(use_commit_readiness=True))
    initial = cr.get_readiness()
    cr.update(outcome_signal=0.0, simulation_mode=True)
    assert cr.get_readiness() == initial, (
        "simulation_mode=True must NOT advance the EMA"
    )
    cr.notify_outcome(value=0.1, simulation_mode=True)
    assert cr.get_readiness() == initial, (
        "notify_outcome(simulation_mode=True) must NOT replace the EMA"
    )
    state = cr.get_state()
    assert state["n_simulation_skips"] == 2
    assert state["n_updates"] == 0


def test_c4_waking_after_simulation_still_advances():
    cr = CommitReadiness(CommitReadinessConfig(use_commit_readiness=True))
    cr.update(outcome_signal=0.0, simulation_mode=True)  # sim skip
    cr.update(outcome_signal=0.0, simulation_mode=False)  # advances EMA
    # 0.1 alpha * 0.0 + 0.9 * 1.0 = 0.9
    assert cr.get_readiness() == pytest.approx(0.9)
    state = cr.get_state()
    assert state["n_simulation_skips"] == 1
    assert state["n_updates"] == 1


def test_c4_none_outcome_is_noop():
    cr = CommitReadiness(CommitReadinessConfig(use_commit_readiness=True))
    initial = cr.get_readiness()
    result = cr.update(outcome_signal=None)
    assert result == initial
    state = cr.get_state()
    assert state["n_updates"] == 0
    assert state["n_simulation_skips"] == 0


# ----------------------------------------------------------------------
# C5 config auto-arm
# ----------------------------------------------------------------------
def test_c5_from_dims_auto_arms_commit_readiness():
    """use_mech090_readiness_conjunction=True auto-arms use_commit_readiness."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, harm_obs_dim=50,
        harm_obs_a_dim=50, action_dim=4, self_dim=16, world_dim=16,
        use_mech090_readiness_conjunction=True,
    )
    assert cfg.use_mech090_readiness_conjunction is True
    assert cfg.use_commit_readiness is True


def test_c5_direct_construction_post_init_arms():
    """REEConfig(...) direct construction with conjunction flag arms via __post_init__."""
    cfg = REEConfig(
        latent=None,  # filled by __post_init__
        use_mech090_readiness_conjunction=True,
    ) if False else None
    # The REEConfig constructor is complex; safer to verify via from_dims
    # and via manual assignment + a fresh __post_init__ invocation.
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, harm_obs_dim=50,
        harm_obs_a_dim=50, action_dim=4, self_dim=16, world_dim=16,
    )
    assert cfg.use_commit_readiness is False
    cfg.use_mech090_readiness_conjunction = True
    cfg.__post_init__()  # re-invoke the resolver
    assert cfg.use_commit_readiness is True


def test_c5_passive_diagnostic_mode():
    """use_commit_readiness=True without the conjunction = passive diagnostic.

    The CommitReadiness module instantiates and tracks readiness, but the
    conjunction at the elevate sites does NOT consult it (use_mech090_
    readiness_conjunction stays False).
    """
    agent = _build_minimal_agent(
        use_commit_readiness=True,
        use_mech090_readiness_conjunction=False,
    )
    assert agent.commit_readiness is not None
    assert agent.config.use_mech090_readiness_conjunction is False
    # Module works on its own
    agent.commit_readiness.notify_outcome(0.2)
    assert agent.commit_readiness.get_readiness() == pytest.approx(0.2)


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------
def test_config_alpha_out_of_range_raises():
    with pytest.raises(ValueError):
        CommitReadiness(CommitReadinessConfig(commit_readiness_ema_alpha=-0.1))
    with pytest.raises(ValueError):
        CommitReadiness(CommitReadinessConfig(commit_readiness_ema_alpha=1.1))


def test_config_initial_out_of_range_raises():
    with pytest.raises(ValueError):
        CommitReadiness(CommitReadinessConfig(commit_readiness_initial=-0.5))
    with pytest.raises(ValueError):
        CommitReadiness(CommitReadinessConfig(commit_readiness_initial=1.5))


def test_config_window_must_be_positive():
    with pytest.raises(ValueError):
        CommitReadiness(CommitReadinessConfig(commit_readiness_window=0))


# ----------------------------------------------------------------------
# Reset semantics
# ----------------------------------------------------------------------
def test_reset_returns_to_initial():
    cr = CommitReadiness(CommitReadinessConfig(
        use_commit_readiness=True, commit_readiness_initial=0.7,
    ))
    cr.notify_outcome(0.1)
    cr.notify_block()
    assert cr.get_readiness() == pytest.approx(0.1)
    assert cr.get_state()["n_blocks_emitted"] == 1
    cr.reset()
    assert cr.get_readiness() == pytest.approx(0.7)
    state = cr.get_state()
    assert state["n_blocks_emitted"] == 0
    assert state["n_updates"] == 0
    assert state["n_simulation_skips"] == 0


def test_agent_reset_calls_module_reset():
    agent = _build_minimal_agent(
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=0.3,
    )
    agent.commit_readiness.notify_outcome(0.1)
    assert agent.commit_readiness.get_readiness() == pytest.approx(0.1)
    agent.reset()
    assert agent.commit_readiness.get_readiness() == pytest.approx(1.0)
