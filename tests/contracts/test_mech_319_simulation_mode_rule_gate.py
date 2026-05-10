"""Contract tests for MECH-319 simulation_mode_rule_write_gate (arc_062 GAP-K).

Substrate-level instantiation of MECH-094 at the rule-arbitration layer.
Tests cover the truth-table semantics, agent wiring, falsifier control,
and MECH-094 invariance.

Five+ contracts:
  C1: default-off no-op. With use_simulation_mode_rule_gate=False
      (default), agent.simulation_mode_rule_gate is None and the
      arbitration-write call sites pass simulation_mode unchanged
      (bit-identical to pre-MECH-319).
  C2: waking writes admitted. With master ON + admit_writes=False
      (default) + caller_sim=False (waking), gate returns False (admit)
      and the n_waking_admitted counter increments.
  C3: simulation writes blocked (MECH-319 normal). With master ON +
      admit_writes=False + caller_sim=True (replay/DMN), gate returns
      True (block) and n_simulation_blocked increments.
  C4: admit_writes flag inverts (V3-EXQ-543c falsifier). With master
      ON + admit_writes=True + caller_sim=True, gate returns False
      (admit despite tag) and n_simulation_admitted increments. Waking
      calls under the falsifier flag are still admitted (no change).
  C5: MECH-094 invariance. The gate's behaviour exactly matches the
      pre-MECH-319 simulation_mode argument semantics in the master-OFF
      regime; in the master-ON regime, the wrap is bit-identical for
      waking call sites (the only sites currently wired). Specifically:
      gated_policy.forward sees the same simulation_mode argument value
      whether MECH-319 is OFF or ON-with-waking-caller.
  C6: per-site counter isolation. Distinct site labels accumulate
      independently in the per_site_* dicts.
  C7: reset() clears counters but preserves config.
  C8: precondition raises. Constructing the regulator with admit_writes=
      True and use_simulation_mode_rule_gate=False raises ValueError
      (loud-not-silent guard).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.regulators import (
    SimulationModeRuleGate,
    SimulationModeRuleGateConfig,
    SITE_DEFAULT,
    SITE_GATED_POLICY,
    SITE_LATERAL_PFC,
)
from ree_core.utils.config import REEConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_agent_and_one_tick(seed: int = 7, **flags):
    """Build a small REEAgent and run one sense() tick."""
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1,
        use_proxy_fields=True,
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
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    kwargs = {"obs_body": body, "obs_world": world}
    for key in ("harm_obs", "harm_obs_a"):
        v = obs_dict.get(key)
        if v is not None:
            attr = "use_harm_stream" if key == "harm_obs" else "use_affective_harm_stream"
            if getattr(cfg.latent, attr, False):
                if v.dim() == 1:
                    v = v.unsqueeze(0)
                kwargs[key.replace("harm_obs", "obs_harm")] = v
    with torch.no_grad():
        latent = agent.sense(**kwargs)
    return agent, latent


# ----------------------------------------------------------------------
# C1 default-off no-op
# ----------------------------------------------------------------------
def test_c1_default_off_no_op_agent():
    agent, latent = _build_agent_and_one_tick()
    assert agent.simulation_mode_rule_gate is None, (
        "default config should produce simulation_mode_rule_gate=None"
    )
    assert latent is not None
    assert torch.isfinite(latent.z_world).all()


def test_c1_default_off_truth_table_identity():
    """Master-OFF gate is identity for both waking and simulation calls."""
    gate = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(use_simulation_mode_rule_gate=False)
    )
    # Both False and True pass through unchanged.
    assert gate.effective_simulation_mode(False, site="x") is False
    assert gate.effective_simulation_mode(True, site="x") is True
    # Counters are NOT advanced (gate is architecturally absent in OFF).
    assert gate.diagnostics.n_calls_total == 0


# ----------------------------------------------------------------------
# C2 waking writes admitted
# ----------------------------------------------------------------------
def test_c2_waking_writes_admitted_master_on_default():
    gate = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(use_simulation_mode_rule_gate=True)
    )
    out = gate.effective_simulation_mode(simulation_mode=False, site=SITE_GATED_POLICY)
    assert out is False, "waking caller_sim=False -> output False (admit)"
    assert gate.diagnostics.n_waking_admitted == 1
    assert gate.diagnostics.n_simulation_blocked == 0
    assert gate.diagnostics.n_simulation_admitted == 0
    assert gate.diagnostics.n_calls_total == 1


def test_c2_agent_wires_gate_when_flag_on():
    agent, _ = _build_agent_and_one_tick(use_simulation_mode_rule_gate=True)
    assert agent.simulation_mode_rule_gate is not None
    assert isinstance(agent.simulation_mode_rule_gate, SimulationModeRuleGate)
    assert agent.simulation_mode_rule_gate.config.use_simulation_mode_rule_gate is True
    assert agent.simulation_mode_rule_gate.config.admit_writes is False


# ----------------------------------------------------------------------
# C3 simulation writes blocked (MECH-319 normal)
# ----------------------------------------------------------------------
def test_c3_simulation_writes_blocked_default():
    gate = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(use_simulation_mode_rule_gate=True)
    )
    out = gate.effective_simulation_mode(simulation_mode=True, site=SITE_LATERAL_PFC)
    assert out is True, "caller_sim=True + admit_writes=False -> output True (block)"
    assert gate.diagnostics.n_simulation_blocked == 1
    assert gate.diagnostics.n_simulation_admitted == 0
    assert gate.diagnostics.n_waking_admitted == 0


# ----------------------------------------------------------------------
# C4 admit_writes flag inverts (V3-EXQ-543c falsifier)
# ----------------------------------------------------------------------
def test_c4_falsifier_admits_simulation_writes():
    gate = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(
            use_simulation_mode_rule_gate=True,
            admit_writes=True,
        )
    )
    out = gate.effective_simulation_mode(simulation_mode=True, site=SITE_LATERAL_PFC)
    assert out is False, "admit_writes=True + caller_sim=True -> output False (admit)"
    assert gate.diagnostics.n_simulation_admitted == 1
    assert gate.diagnostics.n_simulation_blocked == 0
    assert gate.diagnostics.n_waking_admitted == 0


def test_c4_falsifier_does_not_change_waking_behaviour():
    """admit_writes=True should not change the answer for waking calls."""
    gate = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(
            use_simulation_mode_rule_gate=True,
            admit_writes=True,
        )
    )
    out = gate.effective_simulation_mode(simulation_mode=False, site=SITE_GATED_POLICY)
    assert out is False, "waking caller is always admitted regardless of admit_writes"
    assert gate.diagnostics.n_waking_admitted == 1
    assert gate.diagnostics.n_simulation_admitted == 0


def test_c4_agent_falsifier_construction():
    """admit_writes=True wires through REEConfig.from_dims and constructs."""
    agent, _ = _build_agent_and_one_tick(
        use_simulation_mode_rule_gate=True,
        simulation_mode_rule_gate_admit_writes=True,
    )
    assert agent.simulation_mode_rule_gate is not None
    assert agent.simulation_mode_rule_gate.config.admit_writes is True


# ----------------------------------------------------------------------
# C5 MECH-094 invariance
# ----------------------------------------------------------------------
def test_c5_mech094_invariance_gate_off_passes_through():
    """Master OFF: the gate is fully transparent for any caller_sim."""
    gate = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(use_simulation_mode_rule_gate=False)
    )
    # The gate-OFF path is the bit-identical legacy semantic. For the
    # waking arbitration call sites currently wired (gated_policy +
    # lateral_pfc) the caller passes False; the gate returns False
    # exactly as the legacy literal simulation_mode=False would.
    assert gate.effective_simulation_mode(False, site=SITE_GATED_POLICY) is False


def test_c5_mech094_invariance_gate_on_waking_matches_legacy():
    """Master ON, waking caller: gate output matches the legacy literal.

    The arbitration-write call sites in select_action pass simulation_mode=False
    (waking action selection). With MECH-319 ON, the gate must still return
    False for those sites so behaviour is bit-identical to MECH-319 OFF on
    the waking path. The asymmetry surfaces only when caller_sim=True
    (replay paths, not currently wired in select_action).
    """
    for admit_writes in (False, True):
        gate = SimulationModeRuleGate(
            SimulationModeRuleGateConfig(
                use_simulation_mode_rule_gate=True,
                admit_writes=admit_writes,
            )
        )
        for site in (SITE_GATED_POLICY, SITE_LATERAL_PFC):
            assert gate.effective_simulation_mode(False, site=site) is False, (
                f"waking call must be admitted; admit_writes={admit_writes} site={site}"
            )


# ----------------------------------------------------------------------
# C6 per-site counter isolation
# ----------------------------------------------------------------------
def test_c6_per_site_counters_independent():
    gate = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(use_simulation_mode_rule_gate=True)
    )
    gate.effective_simulation_mode(False, site=SITE_GATED_POLICY)
    gate.effective_simulation_mode(False, site=SITE_GATED_POLICY)
    gate.effective_simulation_mode(False, site=SITE_LATERAL_PFC)
    gate.effective_simulation_mode(True, site=SITE_LATERAL_PFC)  # blocked
    d = gate.diagnostics
    assert d.per_site_calls == {SITE_GATED_POLICY: 2, SITE_LATERAL_PFC: 2}
    assert d.per_site_waking_admitted == {SITE_GATED_POLICY: 2, SITE_LATERAL_PFC: 1}
    assert d.per_site_simulation_blocked == {SITE_LATERAL_PFC: 1}
    assert d.per_site_simulation_admitted == {}
    assert d.n_calls_total == 4
    assert d.n_waking_admitted == 3
    assert d.n_simulation_blocked == 1


# ----------------------------------------------------------------------
# C7 reset() clears counters but preserves config
# ----------------------------------------------------------------------
def test_c7_reset_clears_counters_preserves_config():
    gate = SimulationModeRuleGate(
        SimulationModeRuleGateConfig(
            use_simulation_mode_rule_gate=True,
            admit_writes=True,
        )
    )
    gate.effective_simulation_mode(True, site=SITE_GATED_POLICY)
    gate.effective_simulation_mode(False, site=SITE_LATERAL_PFC)
    assert gate.diagnostics.n_calls_total == 2
    gate.reset()
    assert gate.diagnostics.n_calls_total == 0
    assert gate.diagnostics.n_waking_admitted == 0
    assert gate.diagnostics.n_simulation_admitted == 0
    assert gate.diagnostics.per_site_calls == {}
    # Config preserved.
    assert gate.config.use_simulation_mode_rule_gate is True
    assert gate.config.admit_writes is True


def test_c7_agent_reset_clears_gate_counters():
    agent, _ = _build_agent_and_one_tick(use_simulation_mode_rule_gate=True)
    # Pump a few calls through the gate.
    agent.simulation_mode_rule_gate.effective_simulation_mode(False, site=SITE_DEFAULT)
    agent.simulation_mode_rule_gate.effective_simulation_mode(True, site=SITE_DEFAULT)
    assert agent.simulation_mode_rule_gate.diagnostics.n_calls_total == 2
    agent.reset()
    assert agent.simulation_mode_rule_gate.diagnostics.n_calls_total == 0


# ----------------------------------------------------------------------
# C8 precondition raises (admit_writes=True without master is invalid)
# ----------------------------------------------------------------------
def test_c8_precondition_admit_writes_without_master_raises():
    with pytest.raises(ValueError, match="MECH-319"):
        SimulationModeRuleGate(
            SimulationModeRuleGateConfig(
                use_simulation_mode_rule_gate=False,
                admit_writes=True,
            )
        )


# ----------------------------------------------------------------------
# Integration smoke: agent select_action with master ON + waking caller
# is bit-identical at the simulation_mode argument level. We don't run
# select_action here (it requires a fuller env scaffold + heads that
# may not be wired in the smoke env); the wiring contract is covered
# by the C5 tests + the V3-EXQ-546 substrate-readiness experiment.
# ----------------------------------------------------------------------
def test_truth_table_full_matrix():
    """Exhaustive sweep of (master, admit_writes, caller_sim) -> output."""
    cases = [
        # (master, admit_writes, caller_sim, expected_output)
        (False, False, False, False),
        (False, False, True, True),
        # admit_writes=True with master OFF raises -- skipped here, see C8
        (True, False, False, False),
        (True, False, True, True),
        (True, True, False, False),
        (True, True, True, False),
    ]
    for master, admit, caller, expected in cases:
        gate = SimulationModeRuleGate(
            SimulationModeRuleGateConfig(
                use_simulation_mode_rule_gate=master,
                admit_writes=admit,
            )
        )
        actual = gate.effective_simulation_mode(simulation_mode=caller, site="t")
        assert actual is expected, (
            f"truth-table mismatch: master={master} admit={admit} "
            f"caller_sim={caller} -> expected {expected} got {actual}"
        )
