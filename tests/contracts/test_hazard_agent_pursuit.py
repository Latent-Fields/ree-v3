"""Contracts for hazard_agent_pursuit (MECH-357 pressure fallback).

Substrate: ree_core/environment/causal_grid_world.py CausalGridWorld._drift_hazards().
Scoped in REE_assembly/evidence/planning/mech357_freeze_incompatible_pressure_scoping_
2026-08-10.md SS3, following /failure-autopsy on V3-EXQ-603s (MECH-357): 603s's
undirected env_drift_interval/env_drift_prob mobility increase never made hazard
movement depend on the agent's position, so a Pavlovian-instrumental "predator"
pressure paradigm had no genuine agent-directed threat to build on. This adds a
hazard_agent_pursuit: float sibling to the existing hazard_food_attraction bias --
same reef_enabled gate, same per-drift-tick probability roll, same sort-candidate-
directions-by-Manhattan-distance mechanism -- just targeting the agent's current
cell instead of the nearest food resource.

The contracts that matter here:
  (a) hazard_agent_pursuit defaults to 0.0 (OFF), and OFF draws no extra RNG and
      changes no existing trajectory -- the same backward-compat invariant every
      other opt-in knob in this env carries. Verified two ways: the raw default
      value, and (harder) that an RNG draw is never spent on the pursuit roll when
      it is disabled, even while a sibling knob (hazard_food_attraction) IS active
      and rolling its own check every tick -- the confound the design must avoid,
      since any extra draw would shift the RNG stream for every existing
      hazard_food_attraction-using script.
  (b) ON (hazard_agent_pursuit=1.0), a hazard's drift is deterministically biased
      toward the agent's current cell -- contrasted against the OFF/pure-random
      baseline, which is not.
  (c) Reef-cell exclusion holds under pursuit exactly as it does for the existing
      food-attraction/random branches: a pursuing hazard must not step into a
      reef cell even when that cell is the distance-minimising choice.
  (d) hazard_food_attraction keeps precedence when both biases are configured
      nonzero for the same hazard/tick (documented in the constructor comment
      and _drift_hazards docstring) -- its roll is checked first, and success
      there is followed even when it moves the hazard AWAY from the agent.
"""

from __future__ import annotations

import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorld as Env


def _mk(seed=1, **kw):
    base = dict(
        size=10,
        num_hazards=0,
        num_resources=0,
        reef_enabled=True,
        # Force every hazard to attempt a drift every tick so the tests below
        # isolate the direction-selection logic from the outer env_drift_prob
        # roll.
        env_drift_prob=1.0,
        seed=seed,
    )
    base.update(kw)
    return Env(**base)


def _place(env, agent_pos, hazard_positions, reef_cells=()):
    """Deterministic layout via reset_to(), then overlay reef cells the same
    way _place_reef_patches does in production: _reef_cells membership only,
    grid type stays "empty" (reef is not a distinct ENTITY_TYPES value)."""
    env.reset_to(agent_pos=agent_pos, hazard_positions=hazard_positions, resource_positions=[])
    env._reef_cells = set(reef_cells)
    return env


class _CountingRNG:
    """Delegates to a real numpy Generator; counts .random() calls only.

    Swapped in for env._rng AFTER placement so it never affects reset_to()'s
    (RNG-free) deterministic layout -- only _drift_hazards()'s own draws are
    counted.
    """

    def __init__(self, rng):
        self._inner = rng
        self.random_calls = 0

    def random(self, *a, **kw):
        self.random_calls += 1
        return self._inner.random(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# --- (a) default off, and off costs no RNG draw even with a sibling bias active


def test_hazard_agent_pursuit_defaults_to_off():
    env = Env(seed=1)
    assert env.hazard_agent_pursuit == 0.0


def test_default_off_draws_no_extra_rng_when_food_attraction_active():
    """The confound this design must avoid: with hazard_agent_pursuit at its
    default (0.0), _drift_hazards must never reach the pursuit probability
    roll, even while hazard_food_attraction is nonzero and rolling its own
    check every tick. Exactly 2 random() draws per hazard per tick (the outer
    env_drift_prob roll + the food-attraction roll) -- never a 3rd -- proves
    the elif's `hazard_agent_pursuit > 0.0` guard short-circuits before any
    RNG use, so no existing hazard_food_attraction script's trajectory shifts
    under this change.
    """
    env = _place(
        _mk(hazard_food_attraction=0.7, hazard_agent_pursuit=0.0),
        agent_pos=(5, 5),
        hazard_positions=[(1, 1)],
    )
    env.resources = [[8, 8]]
    counting = _CountingRNG(np.random.default_rng(123))
    env._rng = counting
    n_ticks = 20
    for _ in range(n_ticks):
        env._drift_hazards()
    assert counting.random_calls == 2 * n_ticks


def test_default_off_no_directional_bias_toward_agent():
    """Baseline (pursuit=0.0, food_attraction=0.0): a single drift tick from a
    strict diagonal offset should decrease agent-distance on roughly half of
    independent trials (2 of the 4 candidate directions reduce Manhattan
    distance from a pure diagonal offset; a fair shuffle picks each with
    equal probability). Run enough independent seeds that a genuinely-biased
    implementation (which would show ~100%, see the pursuit=1.0 contract
    below) is caught without the test itself being flaky at p=0.5.
    """
    n_trials = 30
    n_decrease = 0
    for trial_seed in range(n_trials):
        env = _place(
            _mk(seed=trial_seed, hazard_agent_pursuit=0.0),
            agent_pos=(5, 5),
            hazard_positions=[(3, 3)],
        )
        start_dist = abs(3 - 5) + abs(3 - 5)
        env._drift_hazards()
        hx, hy = env.hazards[0]
        end_dist = abs(hx - 5) + abs(hy - 5)
        if end_dist < start_dist:
            n_decrease += 1
    # A fair p=0.5 process exceeding 26/30 has probability ~1e-4 (binomial
    # tail) -- generous margin against flakiness while still failing hard on
    # a systematically-biased implementation.
    assert n_decrease < 27, (
        f"{n_decrease}/{n_trials} trials moved toward the agent with "
        "hazard_agent_pursuit=0.0 -- suggests an unintended directional bias"
    )


# --- (b) ON: deterministic bias toward the agent -----------------------------


def test_hazard_agent_pursuit_biases_every_drift_toward_agent():
    """hazard_agent_pursuit=1.0 -> the pursuit roll always succeeds (and
    hazard_food_attraction stays at its 0.0 default, so the food branch never
    preempts it). From a strict diagonal offset, sorted-by-distance ties
    resolve deterministically (stable sort over available_dirs, no RNG in the
    tie-break itself) to a distance-reducing move on every trial -- unlike
    the pursuit=0.0 baseline above, this is not a statistical claim.
    """
    for trial_seed in range(10):
        env = _place(
            _mk(seed=trial_seed, hazard_agent_pursuit=1.0),
            agent_pos=(5, 5),
            hazard_positions=[(3, 3)],
        )
        start_dist = abs(3 - 5) + abs(3 - 5)
        env._drift_hazards()
        hx, hy = env.hazards[0]
        end_dist = abs(hx - 5) + abs(hy - 5)
        assert end_dist < start_dist, (
            f"seed={trial_seed}: pursuit-biased hazard did not move closer "
            f"to the agent (start={start_dist}, end={end_dist})"
        )


def test_hazard_agent_pursuit_converges_over_many_ticks():
    """Sustained pursuit (not a one-shot bias): starting far from the agent
    on an open board, repeated drift ticks should bring the hazard into the
    agent's immediate neighbourhood. It cannot reach distance 0 -- the
    agent's cell is never "empty" so a hazard can never step onto it."""
    env = _place(
        _mk(seed=7, hazard_agent_pursuit=1.0),
        agent_pos=(8, 8),
        hazard_positions=[(1, 1)],
    )
    for _ in range(30):
        env._drift_hazards()
    hx, hy = env.hazards[0]
    end_dist = abs(hx - 8) + abs(hy - 8)
    assert end_dist <= 2, (
        f"sustained pursuit did not converge on the agent: final distance {end_dist}"
    )


# --- (c) reef-cell exclusion holds under pursuit -----------------------------


def test_pursuit_respects_reef_exclusion():
    """The distance-minimising step (4, 5) is a reef cell. A pursuing hazard
    must skip it exactly like the food-attraction/random branches already do
    (the collision loop's `(nx, ny) not in self._reef_cells` check is shared
    code, unmodified by this change) -- landing on the next-best open,
    non-reef direction instead.
    """
    env = _place(
        _mk(seed=1, hazard_agent_pursuit=1.0),
        agent_pos=(5, 5),
        hazard_positions=[(3, 5)],
        reef_cells=[(4, 5)],
    )
    env._drift_hazards()
    hx, hy = env.hazards[0]
    assert (hx, hy) != (4, 5), "pursuing hazard entered an excluded reef cell"
    assert (hx, hy) not in env._reef_cells


# --- (d) hazard_food_attraction keeps precedence over hazard_agent_pursuit ---


def test_food_attraction_takes_precedence_over_agent_pursuit():
    """Both biases configured nonzero and both rolls forced to succeed
    (probability 1.0 each): food is at (0, 0), the agent is at (8, 8) -- the
    two targets pull in opposite directions from the hazard at (2, 2), so the
    branch actually taken is unambiguous from the resulting move. Per the
    constructor/docstring precedence rule, food-attraction is checked (and
    therefore wins) first.
    """
    env = _place(
        _mk(seed=1, hazard_food_attraction=1.0, hazard_agent_pursuit=1.0),
        agent_pos=(8, 8),
        hazard_positions=[(2, 2)],
    )
    env.resources = [[0, 0]]
    agent_dist_before = abs(2 - 8) + abs(2 - 8)
    food_dist_before = abs(2 - 0) + abs(2 - 0)
    env._drift_hazards()
    hx, hy = env.hazards[0]
    agent_dist_after = abs(hx - 8) + abs(hy - 8)
    food_dist_after = abs(hx - 0) + abs(hy - 0)
    assert food_dist_after < food_dist_before, "expected the food-attraction branch to fire"
    assert agent_dist_after > agent_dist_before, (
        "hazard moved toward the agent -- pursuit fired instead of "
        "food-attraction, which should have taken precedence"
    )
