"""Contracts for the SD-094 CausalGridWorld subgoal/hazard-free probe fixes.

Substrate: ree_core/environment/causal_grid_world.py CausalGridWorld.step().
Both defects were confirmed live-reproduced on 2026-08-03 against the
V3-EXQ-884 failure autopsy (num_hazards=0, num_resources=0, subgoal_mode=True,
400 configured steps; episodes actually terminated at 32/19/90 steps on seeds
42/43/44).

Defect 1 -- waypoint arrival is coupled to the GRID CELL TYPE.
  The arrival branch keys on `target_type == ENTITY_TYPES["waypoint"]`. But
  every move writes `self.grid[new_x, new_y] = ENTITY_TYPES["agent"]` over the
  destination cell, and the vacated-cell logic reverts the cell the agent just
  left to "empty" -- never back to "waypoint". So the first transit of ANY
  waypoint cell erases that waypoint's marker for the rest of the sequence,
  including (and especially) a waypoint merely passed through on the way to the
  current target. The agent can then stand exactly on
  `self.waypoints[self._next_waypoint_idx]` and register transition_type
  "none". Fix: opt-in `subgoal_arrival_position_check=True`, which (a) also
  recognises arrival by comparing the post-move position against the pending
  waypoint and (b) restores the marker on a vacated live-waypoint cell,
  mirroring the existing consummatory_act_enabled resource-marker restore.

Defect 2 -- "hazard-free" probes contaminate themselves to death.
  contamination_spread (default 0.5) applies to every cell entered and is
  independent of num_hazards, so num_hazards=0 does NOT yield a harm-free
  world: revisited cells cross contamination_threshold, become
  ENTITY_TYPES["contaminated"], and drain agent_health until
  `done = agent_health <= 0` fires. The V3-EXQ-513 precedent is to pass
  contamination_spread=0.0 by hand. Fix: opt-in
  `hazard_free_contamination_gate=True` makes a num_hazards=0 config do that
  for itself.

Defect 3 (recording gap) -- `done` carried no cause, so a manifest could not
  distinguish "ran the full configured budget" from "died early". Fix:
  always-present `info["done_cause"]` and `info["episode_steps"]`.

The contracts that matter here:
  (a) both flags default OFF and the OFF path reproduces the confirmed buggy
      behaviour exactly -- the backward-compat invariant that keeps the ~1150
      existing scripts constructing this env (and their already-collected
      evidence) unaffected;
  (b) ON, a legitimate arrival on a previously-transited waypoint registers,
      and a full sequence that the legacy path could not close now closes;
  (c) ON does not manufacture arrivals -- a NON-pending waypoint still does not
      advance the sequence, and the elif priority ordering is unchanged;
  (d) the contamination gate is scoped: it fires only for num_hazards == 0 and
      is inert otherwise, and reports whether it actually did anything;
  (e) the V3-EXQ-884 regression itself: OFF dies far short of its budget with
      self-inflicted contamination, ON runs the full budget.
"""

from __future__ import annotations

import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorld as Env

# Actions, per CausalGridWorld.ACTIONS: 0=-x, 1=+x, 2=-y, 3=+y, 4=stay.
Y_UP, Y_DOWN = 3, 2


def _mk(**kw):
    """A subgoal env with contamination off, so these tests isolate the
    waypoint-arrival defect from the (separately tested) contamination one."""
    base = dict(
        size=10,
        num_hazards=0,
        num_resources=0,
        contamination_spread=0.0,
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
    """Deterministic layout: agent at agent_pos, waypoints in the given ORDER
    (waypoints[0] is the first target). reset_to() clears self.waypoints, so
    they are placed afterwards."""
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


def _walk(env, actions):
    """Take the given actions; return the transition_type of each step."""
    out = []
    for a in actions:
        _, _, _, info, _ = env.step(a)
        out.append(info["transition_type"])
    return out


# The canonical scripted layout for defect 1. Waypoints lie on the x=0 column,
# with the SECOND target at y=1 -- i.e. between the agent (y=0) and the first
# target (y=3), so walking to target 0 necessarily transits target 1 first and
# erases its marker. Order: [0,3] then [0,1] then [0,5].
_LAYOUT = [(0, 3), (0, 1), (0, 5)]
# up, up, up  -> transits (0,1), arrives (0,3) = target 0
# down, down  -> returns to (0,1) = target 1
# up x4       -> up to (0,5) = target 2 -> sequence complete
_ROUTE = [Y_UP] * 3 + [Y_DOWN] * 2 + [Y_UP] * 4


# --- (a) OFF reproduces the confirmed defect ---------------------------------


def test_flags_are_off_by_default():
    """The backward-compat invariant. If either default flips, ~1150 existing
    experiment scripts silently change dynamics."""
    env = Env(seed=1)
    assert env.subgoal_arrival_position_check is False
    assert env.hazard_free_contamination_gate is False


def test_off_transited_waypoint_marker_is_destroyed():
    """The mechanism itself: passing through a not-yet-current waypoint leaves
    the cell "empty", desynchronising self.grid from self.waypoints."""
    env = _place(_mk(), _LAYOUT)
    _walk(env, [Y_UP])  # agent moves onto (0,1), which is waypoints[1]
    _walk(env, [Y_UP])  # and leaves it again
    assert env.waypoints[1] == [0, 1]
    assert env.grid[0, 1] == env.ENTITY_TYPES["empty"], (
        "legacy path must still erase the marker -- this is the defect being "
        "pinned, not the fix"
    )


def test_off_misses_arrival_on_a_previously_transited_waypoint():
    """The confirmed V3-EXQ-884 symptom: the agent stands exactly on the pending
    waypoint and no arrival registers."""
    env = _place(_mk(), _LAYOUT)
    # Walk only as far as the return to (0,1), so the position assertion below
    # is made AT the moment of the missed arrival rather than after the route.
    head = _walk(env, _ROUTE[:5])
    assert head[2] == "waypoint", "arrival on the untouched target 0 must fire"
    assert (env.agent_x, env.agent_y) == (0, 1), "agent is ON the pending target"
    assert env.waypoints[env._next_waypoint_idx] == [0, 1]
    assert head[4] == "none", "legacy path misses the arrival"
    tail = _walk(env, _ROUTE[5:])
    assert env._next_waypoint_idx == 1, "sequence stalls on target 1"
    assert env._sequences_completed == 0
    assert "sequence_complete" not in head + tail


# --- (b) ON registers the arrival and closes the sequence ---------------------


def test_on_registers_arrival_on_a_previously_transited_waypoint():
    env = _place(_mk(subgoal_arrival_position_check=True), _LAYOUT)
    types = _walk(env, _ROUTE)
    assert types[2] == "waypoint"
    assert types[4] == "waypoint", "position-based arrival must fire"
    assert types[8] == "sequence_complete", "the full sequence must now close"
    assert env._sequences_completed == 1


def test_on_restores_the_marker_on_a_vacated_live_waypoint_cell():
    """(b) of the fix: the grid -- and therefore the world observation the agent
    sees -- stays consistent with self.waypoints across departures."""
    env = _place(_mk(subgoal_arrival_position_check=True), _LAYOUT)
    _walk(env, [Y_UP, Y_UP])
    assert env.grid[0, 1] == env.ENTITY_TYPES["waypoint"]


def test_on_restores_the_marker_after_an_actual_arrival_too():
    """Waypoints are not consumed on visit -- the arrival branch only advances
    _next_waypoint_idx -- so an already-visited waypoint is still a live entry
    of self.waypoints and its marker must come back as well."""
    env = _place(_mk(subgoal_arrival_position_check=True), _LAYOUT)
    _walk(env, [Y_UP] * 3)  # arrive target 0 at (0,3)
    assert env._next_waypoint_idx == 1
    _walk(env, [Y_DOWN])  # leave it
    assert env.grid[0, 3] == env.ENTITY_TYPES["waypoint"]


# --- (c) ON does not manufacture arrivals ------------------------------------


def test_on_does_not_advance_on_a_non_pending_waypoint():
    """Standing on waypoints[1] while target 0 is pending must NOT advance the
    sequence -- the position check only ever adds the PENDING waypoint."""
    env = _place(_mk(subgoal_arrival_position_check=True), _LAYOUT)
    types = _walk(env, [Y_UP])  # onto (0,1) == waypoints[1], target is still 0
    assert types[0] == "none"
    assert env._next_waypoint_idx == 0
    assert env._sequences_completed == 0


def test_on_registers_a_no_move_arrival_when_standing_on_the_pending_waypoint():
    """Documented, deliberate consequence of detecting by POSITION: a STAY on the
    pending waypoint registers, where the legacy cell-type path cannot (the
    agent's own cell reads ENTITY_TYPES["agent"]). This is reachable in normal
    play -- _respawn_waypoints() can place a new waypoint on the cell the agent
    is standing on -- and it matches the semantics of the substrate's existing
    position-based completion_tolerance path at T=0. It fires ONCE: the arrival
    advances _next_waypoint_idx, and no two waypoints share a cell."""
    env = _place(_mk(subgoal_arrival_position_check=True), [(0, 0), (0, 5)],
                 agent_pos=(0, 0))
    types = _walk(env, [4, 4])  # stay, stay
    assert types[0] == "waypoint"
    assert env._next_waypoint_idx == 1
    assert types[1] == "none", "must not re-fire while parked"


def test_on_is_inert_outside_subgoal_mode():
    """subgoal_mode=False must be wholly unaffected: no waypoints exist, so the
    flag has nothing to key on and the rollout is identical either way."""
    kw = dict(size=10, num_hazards=3, num_resources=3, subgoal_mode=False, seed=5)
    off, on = Env(**kw), Env(**kw, subgoal_arrival_position_check=True)
    off.reset()
    on.reset()
    rng = np.random.RandomState(5)
    for _ in range(120):
        a = int(rng.randint(0, off.action_dim))
        fo, ho, do_, io, _ = off.step(a)
        fn, hn, dn, iN, _ = on.step(a)
        assert bool(np.array_equal(fo.numpy(), fn.numpy()))
        assert float(ho) == float(hn)
        assert bool(do_) == bool(dn)
        assert io["transition_type"] == iN["transition_type"]
        if do_:
            break


def test_on_preserves_the_hazard_priority_ordering():
    """The elif chain is deliberately unchanged: a hazard sitting on the pending
    waypoint cell still classifies as env_caused_hazard, not as an arrival."""
    env = _mk(subgoal_arrival_position_check=True)
    env.reset_to(agent_pos=(0, 0), hazard_positions=[(0, 1)], resource_positions=[])
    env.waypoints = [[0, 1]]
    env._next_waypoint_idx = 0
    env._sequence_in_progress = False
    types = _walk(env, [Y_UP])
    assert types[0] == "env_caused_hazard"
    assert env._next_waypoint_idx == 0


# --- (d) the contamination gate is correctly scoped --------------------------


def test_gate_zeroes_contamination_only_for_hazard_free_configs():
    on0 = Env(size=10, num_hazards=0, hazard_free_contamination_gate=True, seed=1)
    assert on0.contamination_spread == 0.0
    assert on0._contamination_gate_applied is True

    on3 = Env(size=10, num_hazards=3, hazard_free_contamination_gate=True, seed=1)
    assert on3.contamination_spread == 0.5, "must be inert when hazards exist"
    assert on3._contamination_gate_applied is False


def test_gate_reports_no_op_when_contamination_was_already_zero():
    """Distinguishes "the gate protected this probe" from "the flag was set but
    there was nothing to do" -- so a probe author can assert the former."""
    env = Env(
        size=10,
        num_hazards=0,
        contamination_spread=0.0,
        hazard_free_contamination_gate=True,
        seed=1,
    )
    assert env.contamination_spread == 0.0
    assert env._contamination_gate_applied is False


def test_gate_off_by_default_leaves_hazard_free_configs_contaminating():
    env = Env(size=10, num_hazards=0, seed=1)
    assert env.contamination_spread == 0.5
    assert env._contamination_gate_applied is False


# --- (e) the V3-EXQ-884 regression itself ------------------------------------


def _exq884_episode(seed, n_steps=400, **kw):
    """The exact V3-EXQ-884 env config, driven by a fixed random policy."""
    env = Env(
        size=10,
        num_hazards=0,
        num_resources=0,
        subgoal_mode=True,
        num_waypoints=3,
        seed=seed,
        **kw,
    )
    env.reset()
    rng = np.random.RandomState(seed)
    for _ in range(n_steps):
        _, _, done, info, _ = env.step(int(rng.randint(0, env.action_dim)))
        if done:
            return info["episode_steps"], info["done_cause"], float(env.agent_health)
    return info["episode_steps"], info["done_cause"], float(env.agent_health)


def test_exq884_config_dies_of_self_contamination_by_default():
    """Pins the reported failure. num_hazards=0 is NOT a harm-free world."""
    for seed in (42, 43, 44):
        steps, cause, health = _exq884_episode(seed)
        assert cause == "health_depleted", f"seed {seed}: cause={cause}"
        assert health == 0.0
        assert steps < 200, (
            f"seed {seed}: episode ended at {steps} of 400 configured steps "
            "-- the confirmed early-termination symptom"
        )


def test_exq884_config_runs_its_full_budget_with_the_gate_on():
    for seed in (42, 43, 44):
        steps, cause, health = _exq884_episode(
            seed, hazard_free_contamination_gate=True
        )
        assert steps == 400, f"seed {seed}: ran {steps} of 400"
        assert cause == "", "the episode must not have terminated"
        assert health == 1.0, "no harm source exists once the gate applies"


# --- (f) the recording gap ---------------------------------------------------


def test_done_cause_and_episode_steps_are_always_present():
    env = Env(size=10, num_hazards=3, seed=3)
    env.reset()
    for i in range(5):
        _, _, done, info, _ = env.step(4)
        assert "done_cause" in info and "episode_steps" in info
        assert info["episode_steps"] == i + 1 == info["steps"]
        if not done:
            assert info["done_cause"] == "", "empty while in flight"


def test_done_cause_reports_step_limit_at_the_cap():
    """The other terminal branch: 500 steps with no harm source available."""
    env = Env(
        size=10,
        num_hazards=0,
        num_resources=0,
        contamination_spread=0.0,
        seed=4,
    )
    env.reset()
    cause = None
    for _ in range(500):
        _, _, done, info, _ = env.step(4)  # stay: no contamination, no harm
        cause = info["done_cause"]
        if done:
            break
    assert done is True
    assert cause == "step_limit"
    assert info["episode_steps"] == 500
    assert env.agent_health > 0.0


# --- config-snapshot surface -------------------------------------------------


def test_flag_state_is_surfaced_in_the_step_info_snapshot():
    """A run's manifest can record whether either fix was active, without the
    experiment script having to re-derive it from its own kwargs."""
    env = Env(
        size=10,
        num_hazards=0,
        subgoal_mode=True,
        subgoal_arrival_position_check=True,
        hazard_free_contamination_gate=True,
        seed=1,
    )
    env.reset()
    _, _, _, info, _ = env.step(4)
    assert info["subgoal_arrival_position_check"] is True
    assert info["hazard_free_contamination_gate"] is True
    assert info["hazard_free_contamination_gate_applied"] is True
