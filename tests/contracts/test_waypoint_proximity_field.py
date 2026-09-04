"""
Contract tests for SD-WAYPOINT-FIELD: the directional waypoint proximity field
view in CausalGridWorld (subgoal_mode), landed 2026-09-04 from
chip-20260902-waypoint-proximity-field-observable.

WHY THIS SUBSTRATE EXISTS.
In subgoal_mode the waypoints reach the agent ONLY as entity-type channel 6 of
the 5x5x7 local view, so a waypoint more than 2 cells away is not observable at
all. V3-EXQ-977 (INV-086 goal_maintenance_feedback_necessity) measured the
consequence directly on 2026-09-02: 3 waypoints, 12x12 grid, 400 steps,
waypoint_visit_reward=0, seeds 42/43/44 -- the agent's own policy visited 0/0/1
waypoints and completed 0 sequences, indistinguishable from a random walk. Every
navigation-dependent DV (goal maintenance, subgoal seeding, completion rate) is
therefore pinned at chance, which is why all 27 existing subgoal_mode drivers
either script the walk or use a non-completion DV. This channel supplies the
missing DIRECTIONAL signal: a monotone gradient toward the pending waypoint,
readable from anywhere on the grid.

Eight contracts:
  C1  Off by default -- bit-identical to pre-extension behaviour; info-dict
      sentinels always present but inert; no obs channel; world_obs_dim
      unchanged.
  C2  Preconditions: the flag requires use_proxy_fields AND subgoal_mode, and
      waypoint_field_decay must be > 0. All three raise ValueError.
  C3  Channel geometry: enabling grows world_obs_dim by exactly 25; the view is
      the TRAILING 25 dims of world_state (appended last, so no existing channel
      offset moves and the stack.py / zworld_p0.py prefix slice constants stay
      valid); the obs key is present ON and absent OFF.
  C4  Kernel correctness: every cell of the 5x5 patch equals
      1 / (1 + decay * manhattan(cell, pending_waypoint)) exactly, 1.0 on the
      target cell itself, and out-of-bounds cells stay 0.0 (non-toroidal).
  C5  DIRECTIONALITY -- the contract this build exists for. The patch is
      strictly monotone toward the pending waypoint, the at-agent value rises
      strictly along a route that walks the target in, and a target FAR outside
      the 5x5 local view (the V3-EXQ-977 regime) is still discriminable: the
      field distinguishes two positions that the local view renders identically.
  C6  Target tracking: the field re-points when the pending index advances, is
      all-zero when there is no pending waypoint, and its per-episode tags reset
      on both reset() and reset_to().
  C7  RNG isolation: enabling the flag (a pure read of self.waypoints) draws no
      env RNG -- the world_state PREFIX and the harm signal are bit-identical to
      a control differing only by the flag.
  C8  Toroidal correctness: on a wrapped world a target across the seam reads as
      NEAR, not far -- the field uses the shortest wrap-around distance, so it
      never points the agent the long way round.
"""

import numpy as np
import pytest
import torch

from ree_core.environment.causal_grid_world import CausalGridWorld as Env

# Actions, per CausalGridWorld.ACTIONS: 0=-x, 1=+x, 2=-y, 3=+y, 4=stay.
X_DOWN, X_UP, Y_DOWN, Y_UP, STAY = 0, 1, 2, 3, 4


def _mk(**kw):
    """A proxy-mode subgoal env with hazards/resources/contamination off, so
    these contracts isolate the field from every other channel's dynamics."""
    base = dict(
        size=10,
        num_hazards=0,
        num_resources=0,
        contamination_spread=0.0,
        use_proxy_fields=True,
        subgoal_mode=True,
        num_waypoints=3,
        # Long enough that the commitment timeout never fires mid-test and
        # respawns the waypoints out from under the assertions.
        sequence_commitment_timeout=10_000,
        seed=1,
    )
    base.update(kw)
    return Env(**base)


def _place(env, waypoints, agent_pos=(0, 0)):
    """Deterministic layout (the SD-094 contract's helper): agent at agent_pos,
    waypoints in the given ORDER (waypoints[0] is the first target).
    reset_to() clears self.waypoints, so they are placed afterwards."""
    env.reset_to(agent_pos=agent_pos, hazard_positions=[], resource_positions=[])
    env.grid[:, :] = env.ENTITY_TYPES["empty"]
    env.waypoints = [[int(x), int(y)] for x, y in waypoints]
    env._next_waypoint_idx = 0
    env._sequence_in_progress = False
    for wx, wy in env.waypoints:
        env.grid[wx, wy] = env.ENTITY_TYPES["waypoint"]
    ax, ay = int(agent_pos[0]), int(agent_pos[1])
    env.agent_x, env.agent_y = ax, ay
    env.grid[ax, ay] = env.ENTITY_TYPES["agent"]
    return env


def _field(env):
    """The current 5x5 waypoint field as a (5, 5) tensor, read without stepping."""
    obs = env._get_observation_dict()
    return obs["waypoint_proximity_field_view"].reshape(5, 5)


def _at_agent(env):
    """The at-agent (patch centre) value."""
    return float(_field(env)[2, 2])


# ---------------------------------------------------------------------------
# C1  Off by default; sentinels present but inert; no channel; dim unchanged
# ---------------------------------------------------------------------------

def test_c1_off_by_default_inert():
    env = _mk()
    assert env.waypoint_proximity_field_enabled is False
    assert env._waypoint_field_at_agent == 0.0
    assert env._waypoint_field_target_idx == -1

    baseline_dim = _mk().world_obs_dim
    assert env.world_obs_dim == baseline_dim  # no dim change when OFF

    _flat, _h, _d, info, obs = env.step(X_UP)
    # Info sentinels always present but inert.
    assert info["waypoint_proximity_field_enabled"] is False
    assert info["waypoint_field_at_agent"] == 0.0
    assert info["waypoint_field_target_idx"] == -1
    # No obs channel leaked.
    assert "waypoint_proximity_field_view" not in obs

    # Setting the tuning knob alone, with the master switch OFF, changes nothing.
    tuned = _mk(waypoint_field_decay=0.9)
    assert tuned.world_obs_dim == baseline_dim
    o_base, _ = _mk().reset()
    o_tuned, d_tuned = tuned.reset()
    assert torch.equal(o_base, o_tuned)
    assert "waypoint_proximity_field_view" not in d_tuned


# ---------------------------------------------------------------------------
# C2  Preconditions raise, loud-not-silent
# ---------------------------------------------------------------------------

def test_c2_requires_proxy_fields():
    with pytest.raises(ValueError, match="use_proxy_fields"):
        _mk(waypoint_proximity_field_enabled=True, use_proxy_fields=False)


def test_c2_requires_subgoal_mode():
    # Without subgoal_mode self.waypoints is never populated, so the channel
    # would be identically zero and the experiment would measure nothing.
    with pytest.raises(ValueError, match="subgoal_mode"):
        _mk(waypoint_proximity_field_enabled=True, subgoal_mode=False)


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_c2_requires_positive_decay(bad):
    # A non-positive decay makes the kernel constant -> no gradient -> the
    # channel silently stops being directional, which is the entire point.
    with pytest.raises(ValueError, match="waypoint_field_decay"):
        _mk(waypoint_proximity_field_enabled=True, waypoint_field_decay=bad)


def test_c2_master_flag_alone_is_legal():
    env = _mk(waypoint_proximity_field_enabled=True)
    assert env.waypoint_proximity_field_enabled is True


# ---------------------------------------------------------------------------
# C3  Channel geometry: +25, appended LAST, key present ON / absent OFF
# ---------------------------------------------------------------------------

def test_c3_channel_geometry():
    off = _mk()
    on = _mk(waypoint_proximity_field_enabled=True)
    assert on.world_obs_dim == off.world_obs_dim + 25

    _obs, od = on.reset()
    view = od["waypoint_proximity_field_view"]
    assert view.shape == (25,)
    assert view.dtype == torch.float32
    # Appended LAST: this is what keeps stack.py's HAZARD_INDICES /
    # CONTAMINATION_SLICE / RESOURCE_FIELD_SLICE prefix constants valid.
    assert torch.allclose(od["world_state"][-25:], view)
    assert od["world_state"].shape[0] == on.world_obs_dim

    # ... and the prefix is byte-for-byte the OFF world_state.
    _obs_off, od_off = _mk().reset()
    assert torch.equal(od["world_state"][: off.world_obs_dim], od_off["world_state"])


# ---------------------------------------------------------------------------
# C4  Kernel correctness against a hand-computed layout
# ---------------------------------------------------------------------------

def test_c4_kernel_matches_reciprocal_manhattan_decay():
    decay = 0.25
    env = _mk(waypoint_proximity_field_enabled=True, waypoint_field_decay=decay)
    # Agent at (5, 5), pending waypoint 4 cells away at (5, 9).
    _place(env, [(5, 9), (1, 1), (8, 8)], agent_pos=(5, 5))
    f = _field(env)
    tx, ty = 5, 9
    for di in range(-2, 3):
        for dj in range(-2, 3):
            ni, nj = 5 + di, 5 + dj
            d = abs(ni - tx) + abs(nj - ty)
            expected = 1.0 / (1.0 + decay * d)
            assert float(f[di + 2, dj + 2]) == pytest.approx(expected, abs=1e-6)


def test_c4_exactly_one_on_the_target_cell():
    env = _mk(waypoint_proximity_field_enabled=True)
    # Standing ON the pending waypoint: the patch centre is the d=0 value.
    # Read without stepping, because arriving advances the pending index (C6).
    _place(env, [(4, 4), (1, 1), (8, 8)], agent_pos=(4, 4))
    assert _at_agent(env) == pytest.approx(1.0, abs=1e-6)


def test_c4_out_of_bounds_cells_stay_zero_non_toroidal():
    env = _mk(waypoint_proximity_field_enabled=True, toroidal=False)
    # Agent in the corner: the patch's left/top two rows/cols are off-grid.
    _place(env, [(5, 5), (1, 1), (8, 8)], agent_pos=(0, 0))
    f = _field(env)
    assert float(f[0, 0]) == 0.0
    assert float(f[1, 1]) == 0.0
    assert float(f[2, 2]) > 0.0  # the agent's own (in-bounds) cell


# ---------------------------------------------------------------------------
# C5  DIRECTIONALITY -- the contract this build exists for
# ---------------------------------------------------------------------------

def test_c5_patch_is_monotone_toward_the_pending_waypoint():
    env = _mk(waypoint_proximity_field_enabled=True)
    _place(env, [(5, 9), (1, 1), (8, 8)], agent_pos=(5, 5))
    f = _field(env)
    # Along the +y axis (toward the target at y=9) the value strictly increases.
    col = [float(f[2, j]) for j in range(5)]
    assert all(col[j + 1] > col[j] for j in range(4)), col
    # The maximum of the patch is the cell nearest the target.
    flat_idx = int(torch.argmax(f))
    assert (flat_idx // 5, flat_idx % 5) == (2, 4)


def test_c5_at_agent_value_rises_strictly_along_an_approach_route():
    env = _mk(waypoint_proximity_field_enabled=True, waypoint_field_decay=0.25)
    _place(env, [(5, 9), (1, 1), (8, 8)], agent_pos=(5, 5))
    target_idx = env._next_waypoint_idx
    vals = [_at_agent(env)]
    for _ in range(4):
        _f, _h, _d, info, obs = env.step(Y_UP)
        if info["waypoint_field_target_idx"] != target_idx:
            break  # arrived; the field has re-pointed (see C6)
        vals.append(float(obs["waypoint_proximity_field_view"][12]))
    assert len(vals) >= 4
    assert all(vals[i + 1] > vals[i] for i in range(len(vals) - 1)), vals
    # Exact expected series for d = 4, 3, 2, 1 at decay 0.25.
    assert vals[:4] == pytest.approx([0.5, 4.0 / 7.0, 2.0 / 3.0, 0.8], abs=1e-6)


def test_c5_target_beyond_the_local_view_is_still_discriminable():
    """The V3-EXQ-977 regime: the waypoint is far outside the 5x5 window, so the
    local view is IDENTICAL at two different distances and carries no signal.
    The field must separate them -- otherwise nothing has been fixed."""
    near = _mk(waypoint_proximity_field_enabled=True)
    far = _mk(waypoint_proximity_field_enabled=True)
    # Same empty surroundings, target 5 vs 9 cells away -- both far outside the
    # 5x5 (radius-2) local view, so neither agent can see a waypoint at all.
    _place(near, [(0, 5), (1, 1), (8, 8)], agent_pos=(0, 0))
    _place(far, [(0, 9), (1, 1), (8, 8)], agent_pos=(0, 0))

    od_near = near._get_observation_dict()
    od_far = far._get_observation_dict()
    # Precondition of the test: the local views really are identical.
    assert torch.equal(od_near["world_state"][:175], od_far["world_state"][:175])
    # ... and the field really does separate them, monotonically in distance.
    v_near = float(od_near["waypoint_proximity_field_view"][12])
    v_far = float(od_far["waypoint_proximity_field_view"][12])
    assert v_near > v_far
    assert v_near == pytest.approx(1.0 / (1.0 + 0.25 * 5), abs=1e-6)
    assert v_far == pytest.approx(1.0 / (1.0 + 0.25 * 9), abs=1e-6)


# ---------------------------------------------------------------------------
# C6  Target tracking, empty-set handling, per-episode reset
# ---------------------------------------------------------------------------

def test_c6_field_repoints_when_the_pending_index_advances():
    env = _mk(waypoint_proximity_field_enabled=True)
    # Target 0 is one step away at (5, 6); target 1 sits back at (5, 0).
    _place(env, [(5, 6), (5, 0), (8, 8)], agent_pos=(5, 5))
    assert env._next_waypoint_idx == 0
    before = _at_agent(env)
    _f, _h, _d, info, obs = env.step(Y_UP)  # arrive on target 0
    assert env._next_waypoint_idx == 1
    assert info["waypoint_field_target_idx"] == 1
    after = float(obs["waypoint_proximity_field_view"][12])
    # Re-pointed to the DISTANT second waypoint, so the at-agent value drops
    # even though the agent just moved (this is correct, not a regression:
    # arrival and re-targeting happen within the same tick).
    assert after < before


def test_c6_no_pending_waypoint_gives_an_all_zero_field():
    env = _mk(waypoint_proximity_field_enabled=True)
    _place(env, [(5, 9), (1, 1), (8, 8)], agent_pos=(5, 5))
    env._next_waypoint_idx = len(env.waypoints)  # sequence exhausted
    f = _field(env)
    assert torch.count_nonzero(f) == 0
    assert env._waypoint_field_target_idx == -1
    assert env._waypoint_field_at_agent == 0.0


def test_c6_reset_tags_track_the_new_episode_not_the_old_one():
    """reset() ends by building an observation, so the tags legitimately carry
    the FRESH episode's field rather than zeros. What must never happen is a
    tag surviving from the previous episode -- assert against the returned obs,
    which is the ground truth for that tick."""
    env = _mk(waypoint_proximity_field_enabled=True)
    _place(env, [(5, 9), (1, 1), (8, 8)], agent_pos=(5, 5))
    env.step(Y_UP)
    assert env._waypoint_field_at_agent > 0.0
    assert env._waypoint_field_target_idx == 0

    _obs, od = env.reset()
    fresh = float(od["waypoint_proximity_field_view"][12])
    assert env._waypoint_field_at_agent == pytest.approx(fresh, abs=0.0)
    # reset() repopulates self.waypoints, so the sequence restarts at index 0.
    assert env._waypoint_field_target_idx == 0
    assert env._next_waypoint_idx == 0


def test_c6_reset_to_clears_the_tags():
    """reset_to() clears self.waypoints (it is the scripted-eval path), so there
    is no pending target and the tags return to their inert values."""
    env = _mk(waypoint_proximity_field_enabled=True)
    _place(env, [(5, 9), (1, 1), (8, 8)], agent_pos=(5, 5))
    env.step(Y_UP)
    assert env._waypoint_field_at_agent > 0.0

    _obs, od = env.reset_to(
        agent_pos=(0, 0), hazard_positions=[], resource_positions=[]
    )
    assert env.waypoints == []
    assert env._waypoint_field_at_agent == 0.0
    assert env._waypoint_field_target_idx == -1
    assert torch.count_nonzero(od["waypoint_proximity_field_view"]) == 0


# ---------------------------------------------------------------------------
# C7  RNG isolation
# ---------------------------------------------------------------------------

def test_c7_rng_isolation():
    """The field is a pure read of self.waypoints, so enabling it must draw no
    env RNG: the shared world_state PREFIX and the harm signal stay
    bit-identical to a control differing only by the flag."""
    off = _mk(num_hazards=3, num_resources=3, seed=99)
    on = _mk(num_hazards=3, num_resources=3, seed=99,
             waypoint_proximity_field_enabled=True)
    off.reset()
    on.reset()
    prefix = off.world_obs_dim
    rng = np.random.RandomState(7)
    for t in range(60):
        a = int(rng.randint(0, 5))
        _f1, h1, d1, _i1, o1 = off.step(a)
        _f2, h2, d2, _i2, o2 = on.step(a)
        assert torch.equal(o1["world_state"], o2["world_state"][:prefix]), t
        assert h1 == pytest.approx(h2, abs=0.0), t
        assert d1 == d2, t
        if d1:
            break


# ---------------------------------------------------------------------------
# C8  Toroidal correctness
# ---------------------------------------------------------------------------

def test_c8_toroidal_target_across_the_seam_reads_near():
    """On a wrapped world the shortest path to a target at y=9 from y=0 is ONE
    step backwards, not nine forwards. A plain-Manhattan field would point the
    agent the long way round; this one must not."""
    size = 10
    env = _mk(size=size, toroidal=True, waypoint_proximity_field_enabled=True,
              waypoint_field_decay=0.25)
    _place(env, [(0, 9), (1, 1), (8, 8)], agent_pos=(0, 0))
    f = _field(env)
    # Wrap distance from (0,0) to (0,9) is 1, not 9.
    assert float(f[2, 2]) == pytest.approx(1.0 / (1.0 + 0.25 * 1), abs=1e-6)
    # Stepping to y=9 (the -y direction, which wraps) must INCREASE the value.
    before = float(f[2, 2])
    _f, _h, _d, _i, obs = env.step(Y_DOWN)
    assert (env.agent_x, env.agent_y) == (0, 9)
    # Arrival re-points the target, so assert on the pre-step gradient instead:
    # the neighbouring cell toward the seam was strictly nearer than the one away.
    toward = float(f[2, 1])   # dj = -1 -> y = 9 (wrapped), the target itself
    away = float(f[2, 3])     # dj = +1 -> y = 1
    assert toward > before > away
