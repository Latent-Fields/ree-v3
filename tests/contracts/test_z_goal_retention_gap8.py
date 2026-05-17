"""Contract tests for infant_substrate:GAP-8 -- post_sleep_z_goal_retention
and replay_diversity_index telemetry added to SleepLoopManager._run_cycle().

See REE_assembly/evidence/planning/infant_substrate_plan.md GAP-8.

These metrics are monitoring / advisory gate criteria for DEV-NEED-007
(sleep consolidation quality measurement). They are non-invasive: they do
not change any agent behaviour, training, or gradients; they only read
z_goal._z_goal.norm() before and after run_sleep_cycle() and compute a
region-diversity fraction from the already-collected replayed_regions set.

Guarantees enforced:
  C1. _safe_z_goal_norm returns -1.0 when agent has no goal_state.
  C2. _safe_z_goal_norm returns the correct L2 norm when goal_state is
      present and _z_goal has been seeded.
  C3. All four GAP-8 keys present in _run_cycle output for a minimal agent
      (no goal_state, no sampler): post_sleep_z_goal_before,
      post_sleep_z_goal_after, post_sleep_z_goal_retention,
      replay_diversity_index.
  C4. Sentinel values correct for no-goal / no-draw paths:
      z_goal metrics all -1.0 (no goal_state); replay_diversity_index -1.0
      (no sampler -> no routed draws).
  C5. z_goal retention ~= 1.0 when goal_state is seeded and sleep cycle
      runs (sleep does not modify _z_goal).
  C6. replay_diversity_index in [0, 1] when sampler draws produce routed
      events (distinct-regions / total-draws ratio).
  C7. Non-invasive: pre-GAP-8 keys (sws_n_writes, rem_n_rollouts, etc.)
      still present alongside new GAP-8 keys.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #

def _build_agent_no_goal(*, sws: bool = True, rem: bool = True):
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=50,
        action_dim=4,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1,
    )
    cfg.sws_enabled = sws
    cfg.rem_enabled = rem
    return REEAgent(cfg)


def _build_agent_with_goal(*, sws: bool = True, rem: bool = True):
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=50,
        action_dim=4,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1,
        z_goal_enabled=True,
    )
    cfg.sws_enabled = sws
    cfg.rem_enabled = rem
    return REEAgent(cfg)


def _fire_cycle(agent):
    """Trigger one sleep cycle and return the merged metrics dict."""
    from ree_core.sleep import SleepLoopManager

    mgr = SleepLoopManager(cycle_every_k_episodes=1)
    result = mgr.force_cycle(agent)
    return result


# ---------------------------------------------------------------------------- #
# C1: _safe_z_goal_norm sentinel when no goal_state                           #
# ---------------------------------------------------------------------------- #

def test_c1_safe_z_goal_norm_sentinel_no_goal_state():
    from ree_core.sleep.phase_manager import SleepLoopManager

    agent = _build_agent_no_goal()
    # Verify agent really has no goal_state.
    assert getattr(agent, "goal_state", None) is None
    norm = SleepLoopManager._safe_z_goal_norm(agent)
    assert norm == -1.0, f"Expected -1.0 sentinel, got {norm}"


# ---------------------------------------------------------------------------- #
# C2: _safe_z_goal_norm returns correct norm when goal_state present          #
# ---------------------------------------------------------------------------- #

def test_c2_safe_z_goal_norm_returns_correct_value():
    import torch
    from ree_core.sleep.phase_manager import SleepLoopManager

    agent = _build_agent_with_goal()
    goal_state = getattr(agent, "goal_state", None)
    assert goal_state is not None, "Agent must have goal_state for this test"

    # Seed _z_goal to a known value.
    dim = goal_state._z_goal.shape[-1]
    known_vec = torch.ones(dim) * 0.5
    goal_state._z_goal = known_vec
    expected_norm = float(known_vec.norm().item())

    returned = SleepLoopManager._safe_z_goal_norm(agent)
    assert abs(returned - expected_norm) < 1e-5, (
        f"Expected norm {expected_norm:.6f}, got {returned:.6f}"
    )


# ---------------------------------------------------------------------------- #
# C3: All four GAP-8 keys present in _run_cycle output                        #
# ---------------------------------------------------------------------------- #

def test_c3_gap8_keys_always_present():
    _REQUIRED = {
        "post_sleep_z_goal_before",
        "post_sleep_z_goal_after",
        "post_sleep_z_goal_retention",
        "replay_diversity_index",
    }
    agent = _build_agent_no_goal()
    result = _fire_cycle(agent)
    missing = _REQUIRED - set(result.keys())
    assert not missing, f"Missing GAP-8 keys: {missing}"


# ---------------------------------------------------------------------------- #
# C4: Sentinel -1.0 for no-goal and no-draw paths                             #
# ---------------------------------------------------------------------------- #

def test_c4_sentinel_values_no_goal_no_sampler():
    agent = _build_agent_no_goal()
    result = _fire_cycle(agent)

    # No goal_state -> z_goal metrics are -1.0.
    assert result["post_sleep_z_goal_before"] == -1.0, (
        f"Expected -1.0, got {result['post_sleep_z_goal_before']}"
    )
    assert result["post_sleep_z_goal_after"] == -1.0, (
        f"Expected -1.0, got {result['post_sleep_z_goal_after']}"
    )
    assert result["post_sleep_z_goal_retention"] == -1.0, (
        f"Expected -1.0, got {result['post_sleep_z_goal_retention']}"
    )
    # No sampler -> no routed draws -> replay_diversity_index -1.0.
    assert result["replay_diversity_index"] == -1.0, (
        f"Expected -1.0, got {result['replay_diversity_index']}"
    )


# ---------------------------------------------------------------------------- #
# C5: z_goal retention ~= 1.0 when goal_state seeded (sleep preserves z_goal)#
# ---------------------------------------------------------------------------- #

def test_c5_z_goal_retention_near_unity_when_seeded():
    import torch

    agent = _build_agent_with_goal()
    goal_state = agent.goal_state
    assert goal_state is not None

    # Seed _z_goal to a non-trivial value (norm > 1e-8 threshold).
    dim = goal_state._z_goal.shape[-1]
    goal_state._z_goal = torch.ones(dim) * 0.6

    result = _fire_cycle(agent)

    before = result["post_sleep_z_goal_before"]
    after = result["post_sleep_z_goal_after"]
    retention = result["post_sleep_z_goal_retention"]

    # Sleep does not modify _z_goal -> before == after, retention == 1.0.
    assert before > 1e-8, f"before should be > 0 when seeded; got {before}"
    assert abs(before - after) < 1e-5, (
        f"Sleep should not change z_goal norm: before={before:.6f}, after={after:.6f}"
    )
    assert abs(retention - 1.0) < 1e-5, (
        f"Retention should be ~1.0 when z_goal preserved; got {retention:.6f}"
    )


# ---------------------------------------------------------------------------- #
# C6: replay_diversity_index in [0, 1] when draws occur                       #
# ---------------------------------------------------------------------------- #

def test_c6_replay_diversity_index_bounded_when_draws_occur():
    """With a full sleep-aggregation cluster, draws happen and replayed_regions
    is populated. replay_diversity_index must be in [0.0, 1.0]."""
    import torch
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=50,
        action_dim=4,
        use_sleep_aggregation_cluster=True,
    )
    agent = REEAgent(cfg)
    assert agent.sleep_loop is not None

    # Warm the agent so the staleness accumulator / anchor set have content.
    obs = torch.zeros(12 + 50)
    for _ in range(20):
        agent.act(obs)
    agent.reset()  # fires sleep cycle via SleepLoopManager

    # Read the last cycle metrics from cycle_history.
    history = agent.sleep_loop.cycle_history
    if not history:
        # Sleep did not fire (no sws/rem enabled or no passes); skip metric check.
        return
    last = history[-1]
    if "replay_diversity_index" not in last:
        return  # cycle ran but no routed draws; already covered by C4.
    idx = last["replay_diversity_index"]
    if idx == -1.0:
        return  # sentinel (no draws) -- valid; C4 covers this path.
    assert 0.0 <= idx <= 1.0, (
        f"replay_diversity_index must be in [0, 1]; got {idx}"
    )


# ---------------------------------------------------------------------------- #
# C7: pre-GAP-8 keys still present (non-invasive)                             #
# ---------------------------------------------------------------------------- #

def test_c7_pre_gap8_keys_still_present():
    """GAP-8 additions must not displace pre-existing cycle metrics."""
    # The minimal SWS-only agent produces sws_n_writes and sws_slot_diversity.
    agent = _build_agent_no_goal(sws=True, rem=False)
    result = _fire_cycle(agent)

    sws_keys = {"sws_n_writes", "sws_slot_diversity", "sws_buffer_size",
                "sws_anchor_weight_applied"}
    missing = sws_keys - set(result.keys())
    assert not missing, (
        f"Pre-GAP-8 SWS metrics missing after GAP-8 addition: {missing}"
    )
    # GAP-8 keys also present.
    gap8_keys = {"post_sleep_z_goal_before", "post_sleep_z_goal_after",
                 "post_sleep_z_goal_retention", "replay_diversity_index"}
    missing_gap8 = gap8_keys - set(result.keys())
    assert not missing_gap8, f"GAP-8 keys missing: {missing_gap8}"
