"""
Contract tests for infant_substrate:GAP-1 -- harm gradient env feature.

Four contracts:
  C1  Off by default -- legacy CausalGridWorldV2 is bit-identical; sentinel
      info keys always present but zeroed.
  C2  Gradient fires and formula is correct when agent steps into the radial
      band (inner_radius, outer_radius].
  C3  Gradient is suppressed when transition_type != 'none' (direct hazard
      contact or proxy approach takes precedence).
  C4  inner_radius exclusion -- agent inside inner_radius does not trigger
      gradient even though it is inside outer_radius.
"""
import math
import pytest
from ree_core.environment.causal_grid_world import CausalGridWorldV2


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_env(proxy=False, **kw):
    """Return a CausalGridWorldV2 with proxy fields optionally disabled."""
    return CausalGridWorldV2(size=12, seed=7, num_hazards=1,
                             use_proxy_fields=proxy, **kw)


def _place(env, agent_xy, hazard_xy):
    """Force agent and single hazard to known positions for formula tests."""
    env.hazards = [list(hazard_xy)]
    env.agent_x, env.agent_y = agent_xy
    env.grid[agent_xy[0], agent_xy[1]] = env.ENTITY_TYPES["agent"]


def _step_east(env):
    """Action 3 moves (dy=+1); returns info."""
    _, _, _, info, _ = env.step(3)
    return info


# ──────────────────────────────────────────────────────────────────────────────
# C1 -- Off by default
# ──────────────────────────────────────────────────────────────────────────────

def test_c1_off_by_default_sentinel_keys():
    """Sentinel keys always present and zeroed when gradient is disabled."""
    env = _make_env()
    env.reset()
    _, _, _, info, _ = env.step(0)

    assert info["harm_gradient_enabled"] is False
    assert info["harm_gradient_reward_this_tick"] == 0.0
    assert info["harm_gradient_dist_to_nearest"] == -1.0


def test_c1_off_by_default_no_reward():
    """No gradient reward emitted across many steps with default config."""
    env = _make_env()
    env.reset()
    for _ in range(200):
        _, _, done, info, _ = env.step(0)
        assert info["harm_gradient_reward_this_tick"] == 0.0
        if done:
            env.reset()


# ──────────────────────────────────────────────────────────────────────────────
# C2 -- Gradient fires and formula is correct
# ──────────────────────────────────────────────────────────────────────────────

def test_c2_formula_correctness():
    """
    Agent at (2,2), hazard at (2,5).  After action east (dy=+1):
    new position = (2,3), dist to hazard = 2.0.
    Expected reward = -hazard_harm * (1 - 2.0/4.0)^2 * scale.
    """
    env = _make_env(
        harm_gradient_enabled=True,
        harm_gradient_outer_radius=4.0,
        harm_gradient_inner_radius=0.0,
        harm_gradient_scale=1.0,
    )
    env.reset()
    _place(env, (2, 2), (2, 5))
    info = _step_east(env)

    assert info["harm_gradient_enabled"] is True
    assert info["transition_type"] == "harm_gradient"

    expected_dist = 2.0
    expected_reward = -env.hazard_harm * (1.0 - expected_dist / 4.0) ** 2
    assert abs(info["harm_gradient_dist_to_nearest"] - expected_dist) < 1e-5
    assert abs(info["harm_gradient_reward_this_tick"] - expected_reward) < 1e-5


def test_c2_scale_parameter():
    """harm_gradient_scale multiplies the reward linearly."""
    base = _make_env(
        harm_gradient_enabled=True,
        harm_gradient_outer_radius=4.0,
        harm_gradient_inner_radius=0.0,
        harm_gradient_scale=1.0,
    )
    base.reset()
    _place(base, (2, 2), (2, 5))
    info1 = _step_east(base)

    scaled = _make_env(
        harm_gradient_enabled=True,
        harm_gradient_outer_radius=4.0,
        harm_gradient_inner_radius=0.0,
        harm_gradient_scale=3.0,
    )
    scaled.reset()
    _place(scaled, (2, 2), (2, 5))
    info3 = _step_east(scaled)

    assert abs(info3["harm_gradient_reward_this_tick"] -
               3.0 * info1["harm_gradient_reward_this_tick"]) < 1e-5


def test_c2_outside_outer_radius_no_fire():
    """Agent farther than outer_radius: gradient reward == 0."""
    env = _make_env(
        harm_gradient_enabled=True,
        harm_gradient_outer_radius=3.0,
        harm_gradient_inner_radius=0.0,
        harm_gradient_scale=1.0,
    )
    env.reset()
    # dist after east step: (2,3) to (2,8) = 5.0 > 3.0
    _place(env, (2, 2), (2, 8))
    info = _step_east(env)

    assert info["harm_gradient_reward_this_tick"] == 0.0
    assert info["harm_gradient_dist_to_nearest"] > 3.0


def test_c2_gradient_reward_is_negative():
    """Gradient is always a harm (negative reward)."""
    env = _make_env(
        harm_gradient_enabled=True,
        harm_gradient_outer_radius=4.0,
        harm_gradient_inner_radius=0.0,
        harm_gradient_scale=1.0,
    )
    env.reset()
    _place(env, (2, 2), (2, 5))
    info = _step_east(env)

    assert info["harm_gradient_reward_this_tick"] < 0.0


def test_c2_no_health_deduction():
    """Gradient reward does not reduce agent_health."""
    env = _make_env(
        harm_gradient_enabled=True,
        harm_gradient_outer_radius=4.0,
        harm_gradient_inner_radius=0.0,
        harm_gradient_scale=1.0,
    )
    env.reset()
    initial_health = env.agent_health
    _place(env, (2, 2), (2, 5))
    info = _step_east(env)

    assert info["harm_gradient_reward_this_tick"] < 0.0
    assert info["health"] == initial_health


# ──────────────────────────────────────────────────────────────────────────────
# C3 -- Suppression by direct contact / proxy approach
# ──────────────────────────────────────────────────────────────────────────────

def test_c3_suppressed_on_hazard_contact():
    """
    When a hazard contact fires (transition_type='hazard'), gradient must
    remain zero even if the agent is within outer_radius.
    """
    env = _make_env(
        harm_gradient_enabled=True,
        harm_gradient_outer_radius=5.0,
        harm_gradient_inner_radius=0.0,
        harm_gradient_scale=1.0,
    )
    env.reset()
    # Step over many transitions; whenever direct contact fires, gradient should be 0.
    double_signal = False
    for _ in range(400):
        _, _, done, info, _ = env.step(0)
        if info["transition_type"] == "hazard" and info["harm_gradient_reward_this_tick"] != 0.0:
            double_signal = True
            break
        if done:
            env.reset()
    assert not double_signal, "Gradient fired alongside direct hazard contact"


def test_c3_suppressed_by_proxy_approach():
    """
    When proxy approach fires (use_proxy_fields=True), gradient stays zero.
    """
    env = CausalGridWorldV2(
        size=12, seed=7, num_hazards=1,
        use_proxy_fields=True,  # proxy enabled
        harm_gradient_enabled=True,
        harm_gradient_outer_radius=4.0,
        harm_gradient_inner_radius=0.0,
        harm_gradient_scale=1.0,
    )
    env.reset()
    for _ in range(300):
        _, _, done, info, _ = env.step(0)
        if info["transition_type"] == "hazard_approach":
            assert info["harm_gradient_reward_this_tick"] == 0.0, \
                "Gradient fired alongside hazard_approach"
        if done:
            env.reset()


# ──────────────────────────────────────────────────────────────────────────────
# C4 -- inner_radius exclusion
# ──────────────────────────────────────────────────────────────────────────────

def test_c4_inner_radius_exclusion():
    """
    With inner_radius=2.5: agent at dist=2.0 (inside inner zone) -> no gradient.
    Agent at dist=3.0 (outside inner zone) -> gradient fires.
    """
    env = _make_env(
        harm_gradient_enabled=True,
        harm_gradient_outer_radius=4.0,
        harm_gradient_inner_radius=2.5,
        harm_gradient_scale=1.0,
    )
    env.reset()

    # dist after east step = 2.0 (< inner_radius=2.5) -> no fire
    _place(env, (2, 2), (2, 5))
    info = _step_east(env)
    assert info["harm_gradient_reward_this_tick"] == 0.0, \
        f"Gradient fired inside inner_radius: {info['harm_gradient_reward_this_tick']}"

    # dist after east step from (4,2) to (4,5) stepping east = (4,3), dist=2.0 -> still inside
    # Use dist=3.0: agent at (2,2), hazard at (2,6), east step -> (2,3), dist=3.0 > inner=2.5
    _place(env, (2, 2), (2, 6))
    info2 = _step_east(env)
    assert info2["harm_gradient_reward_this_tick"] < 0.0, \
        "Gradient should fire at dist=3.0 (> inner_radius=2.5)"
    assert abs(info2["harm_gradient_dist_to_nearest"] - 3.0) < 1e-5
