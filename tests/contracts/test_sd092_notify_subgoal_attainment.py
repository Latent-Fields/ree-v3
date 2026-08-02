"""SD-092 call-site wiring contracts: REEAgent.notify_subgoal_attainment.

Purpose
-------
SD-092 built GoalState.credit_subgoal_attainment (the cross-level credit
primitive for MECH-427/MECH-428) but left it unreachable from the live agent
loop (docs/architecture/sd_092_cross_level_subgoal_credit.md, "What remains"
item 1). This adds the explicit harness hook REEAgent.notify_subgoal_attainment,
mirroring the notify_env_completion (SD-034) convention: the experiment
harness calls it right after env.step() with info["transition_type"], and it
is a no-op unless both the hook's own preconditions AND
GoalConfig.use_hierarchical_goal_credit are satisfied.

Acceptance
  C1  goal_state is None (z_goal_enabled=False) -> {} no-op, no exception.
  C2  goal_state present, use_hierarchical_goal_credit default False -> {}
      no-op (GoalState's own gate; bit-identical to pre-wiring behavior).
  C3  transition_type not a subgoal-attainment event (e.g. "none",
      "resource_contact") -> {} no-op even with the flag on.
  C4  Flag ON + "waypoint"/"sequence_complete" -> routes into
      credit_subgoal_attainment, parent_goal_norm() rises, non-empty dict
      returned.
  C5  child_representation omitted -> defaults to the current latent's
      z_world (post-act() state).
  C6  Explicit child_representation overrides the default.

Run: /opt/local/bin/python3 -m pytest tests/contracts/test_sd092_notify_subgoal_attainment.py -q
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent

OBS_DIM = 262  # body_obs_dim=12 + world_obs_dim=250 (matches existing SD-034 hook test)


def _build_agent(use_hierarchical_goal_credit=False, **kw):
    cfg = REEConfig.from_dims(
        world_obs_dim=250,
        body_obs_dim=12,
        action_dim=4,
        z_goal_enabled=True,
        **kw,
    )
    agent = REEAgent(cfg)
    agent.goal_state.config.use_hierarchical_goal_credit = use_hierarchical_goal_credit
    return agent


def test_c1_no_goal_state_is_noop():
    cfg = REEConfig.from_dims(
        world_obs_dim=250, body_obs_dim=12, action_dim=4, z_goal_enabled=False,
    )
    agent = REEAgent(cfg)
    assert agent.goal_state is None
    result = agent.notify_subgoal_attainment("sequence_complete")
    assert result == {}


def test_c2_flag_off_is_noop_bit_identical():
    agent = _build_agent(use_hierarchical_goal_credit=False)
    agent.act(torch.zeros(OBS_DIM))
    result = agent.notify_subgoal_attainment("sequence_complete")
    assert result == {}, f"C2 FAIL: expected {{}} no-op, got {result}"
    assert agent.goal_state._z_goal_parent is None, (
        "C2 FAIL: parent tensor allocated despite flag off"
    )


def test_c3_non_subgoal_transition_is_noop():
    agent = _build_agent(use_hierarchical_goal_credit=True)
    agent.act(torch.zeros(OBS_DIM))
    for ttype in ("none", "resource_contact", "hazard_approach", "benefit_approach"):
        result = agent.notify_subgoal_attainment(ttype)
        assert result == {}, f"C3 FAIL: transition_type={ttype!r} should be a no-op, got {result}"
    assert agent.goal_state.parent_goal_norm() == 0.0


def test_c4_flag_on_waypoint_event_credits_parent():
    agent = _build_agent(use_hierarchical_goal_credit=True)
    agent.act(torch.zeros(OBS_DIM))
    assert agent.goal_state.parent_goal_norm() == 0.0
    result = agent.notify_subgoal_attainment("waypoint")
    assert result != {}, "C4 FAIL: expected a non-empty credit result"
    assert result["n_subgoal_credits"] == 1
    assert agent.goal_state.parent_goal_norm() > 0.0

    result2 = agent.notify_subgoal_attainment("sequence_complete")
    assert result2["n_subgoal_credits"] == 2


def test_c5_default_child_representation_is_current_latent_z_world():
    agent = _build_agent(use_hierarchical_goal_credit=True)
    agent.act(torch.zeros(OBS_DIM))
    expected = agent._current_latent.z_world.detach()
    agent.notify_subgoal_attainment("sequence_complete")
    # Parent starts at zero, so one credit call yields exactly
    # parent_goal_alpha * credit * expected (see GoalState.credit_subgoal_attainment).
    alpha = agent.goal_state.config.parent_goal_alpha
    assert torch.allclose(agent.goal_state.z_goal_parent, alpha * expected, atol=1e-6), (
        "C5 FAIL: default child_representation was not the current latent's z_world"
    )


def test_c6_explicit_child_representation_overrides_default():
    agent = _build_agent(use_hierarchical_goal_credit=True)
    agent.act(torch.zeros(OBS_DIM))
    # goal_dim is kept in sync with world_dim (REEConfig.from_dims), so the
    # current latent's z_world shape is a safe stand-in for goal_dim.
    goal_dim = agent._current_latent.z_world.shape[-1]
    custom = torch.full((1, goal_dim), 3.0)
    agent.notify_subgoal_attainment("waypoint", child_representation=custom)
    cos = torch.nn.functional.cosine_similarity(
        agent.goal_state.z_goal_parent, custom, dim=-1
    ).item()
    assert cos > 0.99, f"C6 FAIL: parent not pulled toward the explicit override (cos={cos:.4f})"


if __name__ == "__main__":
    test_c1_no_goal_state_is_noop()
    test_c2_flag_off_is_noop_bit_identical()
    test_c3_non_subgoal_transition_is_noop()
    test_c4_flag_on_waypoint_event_credits_parent()
    test_c5_default_child_representation_is_current_latent_z_world()
    test_c6_explicit_child_representation_overrides_default()
    print("All SD-092 notify_subgoal_attainment call-site contracts PASS")
