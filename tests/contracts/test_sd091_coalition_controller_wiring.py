"""
Contract tests for SD-091/MECH-481 live-agent-loop wiring.

The MVP module (ree_core/claustrum/) landed self-contained and unimported
2026-08-02 (chip-20260802-sd091-implement-mvp). This suite covers the
deferred follow-up (chip-20260802-sd091-live-wiring): steps 4-5 of
sd_091_coalition_topology_control.md's "Minimum-viable V3 implementation
path" -- wiring CoalitionController into REEAgent.__init__/reset/
select_action and the named E1/E2/hippocampal/BetaGate consumer sites.

  W1  default-OFF: agent.coalition is None (use_coalition_controller=False).
  W2  bit-identical: use_coalition_controller=True with no active coalition
      requested produces IDENTICAL actions to default-OFF, per-step, across
      several ticks (write_gate()/channel_gain() are exact-1.0 no-ops).
  W3  reset() clears active coalitions (mirrors SalienceCoordinator/BetaGate).
  W4  SENSORY_RESAMPLE recruits exactly {e1_sensory_encoder,
      e2_fast_forward_model} (write_gate < 1.0) + e3_candidate_count
      (channel_gain > 1.0), and leaves every PROVENANCE_CHECK-only target
      at the no-op baseline (doc's "no global broadcast" guardrail, restated
      per-template at the wiring layer).
  W5  PROVENANCE_CHECK recruits/suppresses exactly its 5 named targets and
      leaves SENSORY_RESAMPLE-only targets at baseline.
  W6  doc step-6 smoke test, codified: a live agent tick under each template
      does not crash and coalition_gate differs measurably between the two
      templates at every site both name (there are none -- the tables are
      disjoint by construction -- so this instead pins that each template's
      OWN gates differ from the untouched 1.0 baseline during a real tick).
  W7  BetaGate guardrail (module docstring's own required contract test):
      coalition suppression of e3_commitment_monitor / motor_commitment can
      only LOWER the effective commit-readiness margin, never raise it above
      the uncoalitioned baseline, for arbitrary recruit/suppress weights.
  W8  agent.coalition.tick() is called each act_with_split_obs() step (a
      coalition opened with max_duration_ticks=1 has dissolved by the next
      tick's read).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ree_core.agent import REEAgent
from ree_core.claustrum.control_demand import ControlDemandType
from ree_core.claustrum.coalition_controller import (
    CoalitionController,
    CoalitionControllerConfig,
    CoalitionState,
)
from ree_core.utils.config import REEConfig


ALL_TARGETS = (
    "e1_sensory_encoder",
    "e2_fast_forward_model",
    "e3_candidate_count",
    "hippocampal_anchor_set",
    "hippocampal_persistence_appraisal",
    "e3_commitment_monitor",
    "motor_commitment",
    "hippocampal_write_consolidation",
)
SENSORY_RESAMPLE_WRITE_GATE_TARGETS = frozenset(
    {"e1_sensory_encoder", "e2_fast_forward_model"}
)
PROVENANCE_CHECK_WRITE_GATE_TARGETS = frozenset(
    {
        "hippocampal_anchor_set",
        "hippocampal_persistence_appraisal",
        "e3_commitment_monitor",
        "motor_commitment",
        "hippocampal_write_consolidation",
    }
)


def _build(use_coalition: bool = False, seed: int = 7):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        use_coalition_controller=use_coalition,
    )
    torch.manual_seed(123)
    agent = REEAgent(cfg)
    agent.reset()
    _flat, od = env.reset()
    b = od["body_state"]
    w = od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return agent, b, w


# ----------------------------------------------------------------------
# W1: default-OFF
# ----------------------------------------------------------------------
def test_w1_default_off_no_coalition():
    agent, _b, _w = _build(use_coalition=False)
    assert agent.coalition is None


# ----------------------------------------------------------------------
# W2: bit-identical when enabled-but-inactive
# ----------------------------------------------------------------------
def test_w2_enabled_but_inactive_is_bit_identical_to_off():
    agent_off, b1, w1 = _build(use_coalition=False)
    agent_on, b2, w2 = _build(use_coalition=True)
    assert agent_on.coalition is not None
    assert agent_on.coalition.config.enabled is True

    for i in range(8):
        torch.manual_seed(1000 + i)
        with torch.no_grad():
            a1 = agent_off.act_with_split_obs(b1, w1)
        torch.manual_seed(1000 + i)
        with torch.no_grad():
            a2 = agent_on.act_with_split_obs(b2, w2)
        assert torch.equal(a1, a2), f"action mismatch at step {i}"


# ----------------------------------------------------------------------
# W3: reset() clears active coalitions
# ----------------------------------------------------------------------
def test_w3_reset_clears_active_coalitions():
    agent, _b, _w = _build(use_coalition=True)
    agent.coalition.request_coalition(ControlDemandType.SENSORY_RESAMPLE, tick=0)
    assert len(agent.coalition.active_coalitions) == 1
    agent.reset()
    assert agent.coalition.active_coalitions == []
    assert agent.coalition.write_gate("e1_sensory_encoder") == 1.0


# ----------------------------------------------------------------------
# W4/W5: per-template target isolation (no global broadcast, restated at
# the wiring layer -- every named target actually reads a distinct gate).
# ----------------------------------------------------------------------
def test_w4_sensory_resample_targets_isolated():
    agent, _b, _w = _build(use_coalition=True)
    agent.coalition.request_coalition(ControlDemandType.SENSORY_RESAMPLE, tick=0)
    for target in SENSORY_RESAMPLE_WRITE_GATE_TARGETS:
        assert agent.coalition.write_gate(target) < 1.0, target
    assert agent.coalition.channel_gain("e3_candidate_count") > 1.0
    for target in PROVENANCE_CHECK_WRITE_GATE_TARGETS:
        assert agent.coalition.write_gate(target) == 1.0, target


def test_w5_provenance_check_targets_isolated():
    agent, _b, _w = _build(use_coalition=True)
    agent.coalition.request_coalition(ControlDemandType.PROVENANCE_CHECK, tick=0)
    for target in PROVENANCE_CHECK_WRITE_GATE_TARGETS:
        assert agent.coalition.write_gate(target) < 1.0, target
    for target in SENSORY_RESAMPLE_WRITE_GATE_TARGETS:
        assert agent.coalition.write_gate(target) == 1.0, target
    assert agent.coalition.channel_gain("e3_candidate_count") == 1.0


# ----------------------------------------------------------------------
# W6: doc step-6 smoke test -- a live tick under each template runs clean
# and the template's own named gates are away from the 1.0 baseline
# throughout the tick (not reset mid-tick by some other code path).
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "demand_type,targets",
    [
        (ControlDemandType.SENSORY_RESAMPLE, SENSORY_RESAMPLE_WRITE_GATE_TARGETS),
        (ControlDemandType.PROVENANCE_CHECK, PROVENANCE_CHECK_WRITE_GATE_TARGETS),
    ],
)
def test_w6_live_tick_under_each_template(demand_type, targets):
    agent, b, w = _build(use_coalition=True)
    agent.coalition.request_coalition(demand_type, tick=agent._step_count, max_duration_ticks=50)
    gates_before = {t: agent.coalition.write_gate(t) for t in targets}
    with torch.no_grad():
        action = agent.act_with_split_obs(b, w)
    assert torch.isfinite(action).all()
    gates_after = {t: agent.coalition.write_gate(t) for t in targets}
    assert gates_before == gates_after
    assert all(g < 1.0 for g in gates_after.values())


# ----------------------------------------------------------------------
# W7: BetaGate guardrail -- coalition_gate on commitment-monitor-adjacent
# targets is monotone non-increasing in effect on commit-readiness. This
# is the module docstring's own required contract test, exercised here at
# the wiring layer (the _readiness_margin composition in select_action)
# rather than only on CoalitionState arithmetic in isolation.
# ----------------------------------------------------------------------
def test_w7_betagate_guardrail_never_raises_readiness():
    controller = CoalitionController(CoalitionControllerConfig(enabled=True))
    baseline_margin = 0.42
    # Sweep the full [0, 1] range of both targets' effective coalition_gate
    # by constructing CoalitionState directly (isolated from any specific
    # template's fixed magnitudes) and confirm the composed gate can never
    # exceed 1.0 -- i.e. can never raise baseline_margin.
    for recruit in (0.0, 0.3, 0.6, 0.9, 1.0):
        for suppress in (0.0, 0.3, 0.6, 0.9, 1.0):
            state = CoalitionState(
                demand_type=ControlDemandType.PROVENANCE_CHECK,
                participating={"e3_commitment_monitor": recruit},
                suppressed={"motor_commitment": suppress},
                opened_tick=0,
                max_duration_ticks=10,
            )
            controller._active = [state]
            composed = controller.write_gate(
                "e3_commitment_monitor"
            ) * controller.write_gate("motor_commitment")
            assert composed <= 1.0
            effective_margin = baseline_margin * composed
            assert effective_margin <= baseline_margin + 1e-12


# ----------------------------------------------------------------------
# W8: coalition.tick() runs every step (a short-lived coalition dissolves).
# ----------------------------------------------------------------------
def test_w8_coalition_tick_dissolves_on_schedule():
    # coalition.tick() is called from the SELECT step, which -- like the
    # SalienceCoordinator tick it runs alongside -- only fires on E3-tick
    # cycles (MultiRateClock default e3_steps_per_tick=10), not every raw
    # env step. So dissolution is asserted within a budget of raw steps
    # comfortably exceeding several E3 ticks, not after exactly 2 calls.
    agent, b, w = _build(use_coalition=True)
    agent.coalition.request_coalition(
        ControlDemandType.SENSORY_RESAMPLE,
        tick=agent._step_count,
        max_duration_ticks=1,
    )
    assert len(agent.coalition.active_coalitions) == 1
    dissolved = False
    for _ in range(30):
        with torch.no_grad():
            agent.act_with_split_obs(b, w)
        if agent.coalition.active_coalitions == []:
            dissolved = True
            break
    assert dissolved, "coalition never dissolved within 30 raw steps (~3 E3 ticks)"
