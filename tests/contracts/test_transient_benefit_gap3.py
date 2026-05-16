"""
Contract tests for infant_substrate:GAP-3 -- transient benefit patches env feature.

Five contracts:
  C1  Off by default -- legacy CausalGridWorldV2 is bit-identical (same RNG
      sequence -> same reset layout AND same stepped trajectory; no extra RNG
      draws when disabled); sentinel info keys always present but inert.
  C2  Spawn -- with transient_benefit_prob=1.0 a patch is placed every tick:
      tagged as a resource on the grid, registered in self.resources +
      self._transient_benefits (with expiry = spawn_step + duration) +
      self._transient_benefit_cells; n_spawned increments.
  C3  Expiry -- a patch is removed exactly transient_benefit_duration step()
      calls after it spawned: grid cell cleared, dropped from self.resources
      and tracking, n_expired increments.
  C4  Contact reward -- stepping onto a transient patch yields
      resource_benefit * transient_benefit_multiplier (not the plain
      resource_benefit); transition_type == "resource"; contact diagnostics
      set; the patch is consumed (removed from tracking).
  C5  reset() clears per-episode transient state and diagnostic counters.
"""
import numpy as np
import pytest
from ree_core.environment.causal_grid_world import CausalGridWorldV2


def _layout(env):
    """Sorted agent + hazard + resource positions after a reset()."""
    env.reset()
    return (
        (env.agent_x, env.agent_y),
        sorted(tuple(h) for h in env.hazards),
        sorted(tuple(r) for r in env.resources),
    )


# --------------------------------------------------------------------------- #
# C1 -- Off by default (bit-identical legacy behaviour)
# --------------------------------------------------------------------------- #

def test_c1_sentinel_keys_inert_when_disabled():
    env = CausalGridWorldV2(size=12, seed=7, num_hazards=2)
    env.reset()
    _, _, _, info, _ = env.step(0)
    assert info["transient_benefit_enabled"] is False
    assert info["transient_benefit_n_active"] == 0
    assert info["transient_benefit_n_spawned"] == 0
    assert info["transient_benefit_n_contacted"] == 0
    assert info["transient_benefit_n_expired"] == 0
    assert info["transient_benefit_contact_this_tick"] == 0.0
    assert env._transient_benefits == []
    assert env._transient_benefit_cells == set()


def test_c1_bit_identical_off_reset_and_step():
    """Default env and explicit transient_benefit_enabled=False produce the
    identical reset layout AND stepped trajectory from the same seed (no
    extra RNG draws anywhere when OFF)."""
    a = CausalGridWorldV2(size=12, seed=42, num_hazards=3, num_resources=4)
    b = CausalGridWorldV2(
        size=12, seed=42, num_hazards=3, num_resources=4,
        transient_benefit_enabled=False,
    )
    for _ in range(3):
        assert _layout(a) == _layout(b)
    # Reset both to a common episode, then step in lockstep.
    a.reset()
    b.reset()
    for t in range(60):
        act = t % 5
        _, ha, _, _, _ = a.step(act)
        _, hb, _, _, _ = b.step(act)
        assert (a.agent_x, a.agent_y) == (b.agent_x, b.agent_y)
        assert sorted(map(tuple, a.hazards)) == sorted(map(tuple, b.hazards))
        assert ha == hb


def test_c1_no_patches_across_episode_when_disabled():
    env = CausalGridWorldV2(size=12, seed=7, num_hazards=2)
    env.reset()
    for _ in range(200):
        _, _, done, info, _ = env.step(0)
        assert info["transient_benefit_enabled"] is False
        assert info["transient_benefit_n_active"] == 0
        if done:
            env.reset()


# --------------------------------------------------------------------------- #
# C2 -- Spawn
# --------------------------------------------------------------------------- #

def test_c2_spawn_every_tick_at_prob_one():
    """prob=1.0 -> a patch spawns on every step(); each is grid-tagged as a
    resource, in self.resources, in tracking, with the correct expiry."""
    env = CausalGridWorldV2(
        size=10, seed=3, num_hazards=0, num_resources=0,
        use_proxy_fields=False, toroidal=True,
        transient_benefit_enabled=True, transient_benefit_prob=1.0,
        transient_benefit_duration=1000,  # long enough that none expire here
    )
    env.reset()
    for n in range(5):
        spawn_step = env.steps  # pre-increment clock used by the spawn block
        _, _, _, info, _ = env.step(4)  # stay -- agent never contacts a patch
        assert info["transient_benefit_n_spawned"] == n + 1
        assert info["transient_benefit_n_active"] == n + 1
        assert len(env._transient_benefits) == n + 1
        assert len(env._transient_benefit_cells) == n + 1
        # Every tracked patch is on the grid as a resource and in resources.
        for (px, py, exp) in env._transient_benefits:
            assert env.grid[px, py] == env.ENTITY_TYPES["resource"]
            assert [px, py] in env.resources
            assert (px, py) in env._transient_benefit_cells
        # The patch spawned this tick carries expiry = spawn_step + duration.
        newest = env._transient_benefits[-1]
        assert newest[2] == spawn_step + env.transient_benefit_duration


def test_c2_no_spawn_at_prob_zero():
    env = CausalGridWorldV2(
        size=10, seed=3, num_hazards=0, num_resources=0,
        use_proxy_fields=False, toroidal=True,
        transient_benefit_enabled=True, transient_benefit_prob=0.0,
    )
    env.reset()
    for _ in range(50):
        _, _, _, info, _ = env.step(4)
        assert info["transient_benefit_n_spawned"] == 0
        assert info["transient_benefit_n_active"] == 0


# --------------------------------------------------------------------------- #
# C3 -- Expiry
# --------------------------------------------------------------------------- #

def test_c3_patch_expires_after_duration():
    """A single patch spawned at the first step() is removed exactly
    `duration` step() calls later (grid cleared, dropped from resources +
    tracking, n_expired incremented)."""
    duration = 3
    env = CausalGridWorldV2(
        size=10, seed=9, num_hazards=0, num_resources=0,
        use_proxy_fields=False, toroidal=True,
        transient_benefit_enabled=True, transient_benefit_prob=1.0,
        transient_benefit_duration=duration,
    )
    env.reset()
    # First step() (self.steps == 0 in the spawn block): patch P, expiry=3.
    _, _, _, info0, _ = env.step(4)
    assert info0["transient_benefit_n_spawned"] == 1
    px, py, exp = env._transient_benefits[0]
    assert exp == 0 + duration
    # Stop further spawns so exactly one patch is in flight.
    env.transient_benefit_prob = 0.0
    # steps==1, then 2: still alive (1 < 3, 2 < 3).
    for expected_step in (1, 2):
        _, _, _, info, _ = env.step(4)
        assert info["transient_benefit_n_active"] == 1
        assert info["transient_benefit_n_expired"] == 0
        assert env.grid[px, py] == env.ENTITY_TYPES["resource"]
    # steps==3 >= expiry 3 -> P expires this tick.
    _, _, _, info_exp, _ = env.step(4)
    assert info_exp["transient_benefit_n_active"] == 0
    assert info_exp["transient_benefit_n_expired"] == 1
    assert env._transient_benefits == []
    assert (px, py) not in env._transient_benefit_cells
    assert env.grid[px, py] == env.ENTITY_TYPES["empty"]
    assert [px, py] not in env.resources


# --------------------------------------------------------------------------- #
# C4 -- Contact reward
# --------------------------------------------------------------------------- #

def test_c4_contact_applies_multiplier():
    """Stepping onto a transient patch yields resource_benefit * multiplier
    (no proxy fields -> harm_signal == contact_benefit exactly)."""
    env = CausalGridWorldV2(
        size=12, seed=4, num_hazards=0, num_resources=0,
        use_proxy_fields=False, toroidal=True, resource_benefit=0.3,
        transient_benefit_enabled=True, transient_benefit_prob=0.0,
        transient_benefit_multiplier=2.0, transient_benefit_duration=50,
    )
    env.reset()
    # Manually inject one patch immediately east of the agent.
    env.agent_x, env.agent_y = 5, 5
    env.grid[:, :] = env.ENTITY_TYPES["empty"]
    env.grid[5, 5] = env.ENTITY_TYPES["agent"]
    env.grid[5, 6] = env.ENTITY_TYPES["resource"]
    env.resources = [[5, 6]]
    env._transient_benefits = [[5, 6, env.steps + 50]]
    env._transient_benefit_cells = {(5, 6)}

    # Action 3 -> (0, +1): move onto (5, 6).
    _, harm_signal, _, info, _ = env.step(3)
    expected = 0.3 * 2.0
    assert abs(harm_signal - expected) < 1e-9, harm_signal
    assert info["transition_type"] == "resource"
    assert abs(info["transient_benefit_contact_this_tick"] - expected) < 1e-9
    assert info["transient_benefit_n_contacted"] == 1
    # Patch consumed: dropped from tracking + resources.
    assert (5, 6) not in env._transient_benefit_cells
    assert all(not (tb[0] == 5 and tb[1] == 6) for tb in env._transient_benefits)
    assert [5, 6] not in env.resources


def test_c4_plain_resource_unaffected_by_multiplier():
    """A normal (non-transient) resource still pays the plain
    resource_benefit even with the transient feature enabled."""
    env = CausalGridWorldV2(
        size=12, seed=4, num_hazards=0, num_resources=0,
        use_proxy_fields=False, toroidal=True, resource_benefit=0.3,
        transient_benefit_enabled=True, transient_benefit_prob=0.0,
        transient_benefit_multiplier=5.0,
    )
    env.reset()
    env.agent_x, env.agent_y = 5, 5
    env.grid[:, :] = env.ENTITY_TYPES["empty"]
    env.grid[5, 5] = env.ENTITY_TYPES["agent"]
    env.grid[5, 6] = env.ENTITY_TYPES["resource"]
    env.resources = [[5, 6]]
    # NOT registered as a transient patch -> plain reward.
    _, harm_signal, _, info, _ = env.step(3)
    assert abs(harm_signal - 0.3) < 1e-9, harm_signal
    assert info["transient_benefit_n_contacted"] == 0
    assert info["transient_benefit_contact_this_tick"] == 0.0


# --------------------------------------------------------------------------- #
# C5 -- reset() clears per-episode transient state
# --------------------------------------------------------------------------- #

def test_c5_reset_clears_transient_state():
    env = CausalGridWorldV2(
        size=10, seed=2, num_hazards=0, num_resources=0,
        use_proxy_fields=False, toroidal=True,
        transient_benefit_enabled=True, transient_benefit_prob=1.0,
        transient_benefit_duration=1000,
    )
    env.reset()
    for _ in range(4):
        env.step(4)
    assert len(env._transient_benefits) > 0
    assert env._transient_benefit_n_spawned > 0

    env.reset()
    assert env._transient_benefits == []
    assert env._transient_benefit_cells == set()
    assert env._transient_benefit_n_spawned == 0
    assert env._transient_benefit_n_contacted == 0
    assert env._transient_benefit_n_expired == 0
    assert env._transient_benefit_contact_this_tick == 0.0
