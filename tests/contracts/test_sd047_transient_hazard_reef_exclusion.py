"""Contract for reef-cell exclusion in SD-047 transient hazard appearance.

Substrate: ree_core/environment/causal_grid_world.py
CausalGridWorld._step_transient_events().

Found while building V3-EXQ-916 (Fishtank relief/safety telemetry). Every
other hazard-position mutator in this env excludes reef cells -- initial
placement (_place_reef_patches) and drift (_drift_hazards, `(nx, ny) not in
self._reef_cells`) -- but _step_transient_events's appearance loop only
filtered on `self.grid[i, j] == empty`, never checking `_reef_cells`. A
transient hazard could therefore spawn directly inside a reef cell, silently
violating the reef substrate's documented occupancy invariant ("hazards
excluded from reef cells") for any future experiment enabling BOTH
multi_source_dynamics_enabled and transient_events_enabled together with
reef_enabled=True. No live driver enables that combination today (both
default False), so this was latent, not an active-incident fix. Agent-
experienced harm was never affected -- a separate guard (step()'s reef-cell
harm zeroing) already makes the agent immune to contact damage while
standing in a reef cell regardless of hazard identity.

The contracts that matter here:
  (a) A transient hazard must never appear inside a reef cell, exactly like
      _drift_hazards already guarantees for hazard movement.
  (b) reef_enabled=False (the default) must leave _reef_cells empty and
      therefore spend exactly the same number of RNG draws as before this
      fix existed -- one per empty candidate cell, never fewer -- so every
      existing non-reef experiment's RNG stream and trajectory is
      byte-identical.
"""

from __future__ import annotations

import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorld as Env


def _mk(seed=1, **kw):
    base = dict(
        size=12,
        num_hazards=0,
        num_resources=0,
        multi_source_dynamics_enabled=True,
        transient_events_enabled=True,
        transient_p_appear=0.3,
        transient_p_disappear=0.1,
        seed=seed,
    )
    base.update(kw)
    return Env(**base)


class _CountingRNG:
    """Delegates to a real numpy Generator; counts .random() calls only.

    Swapped in AFTER reset() so it never affects reset()'s own draws --
    only _step_transient_events()'s draws are counted.
    """

    def __init__(self, rng):
        self._inner = rng
        self.random_calls = 0

    def random(self, *a, **kw):
        self.random_calls += 1
        return self._inner.random(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# --- (a) reef-cell exclusion holds for transient appearance -----------------


def test_transient_hazard_never_spawns_in_reef_cell():
    """Dense reef coverage (~1/3 of interior cells) + high appear probability
    over many ticks: any reef violation shows up fast if the exclusion is
    missing. This test fails against the pre-fix code (verified by stashing
    the fix and re-running: violations > 0 within a handful of ticks).
    """
    env = _mk(seed=3, reef_enabled=True, n_reef_patches=3, reef_patch_radius=2)
    env.reset()
    reef = set(env._reef_cells)
    assert reef, "expected reef cells to exist for this check to be meaningful"

    violations = 0
    total_appearances = 0
    for _ in range(300):
        n_appeared, _n_disappeared = env._step_transient_events()
        total_appearances += n_appeared
        for (tx, ty, _age) in env._transient_hazards:
            if (tx, ty) in reef:
                violations += 1

    assert total_appearances > 0, "test did not exercise the appearance path at all"
    assert violations == 0, (
        f"{violations} transient hazard(s) occupied a reef cell across "
        f"{total_appearances} appearances -- reef exclusion is not holding"
    )


def test_transient_hazard_reef_exclusion_holds_on_toroidal_board():
    """The toroidal and non-toroidal appearance loops are separate branches
    in _step_transient_events -- exercise both, not just the default."""
    env = _mk(seed=4, reef_enabled=True, n_reef_patches=3, reef_patch_radius=2, toroidal=True)
    env.reset()
    reef = set(env._reef_cells)
    assert reef

    violations = 0
    for _ in range(300):
        env._step_transient_events()
        violations += sum(1 for (tx, ty, _age) in env._transient_hazards if (tx, ty) in reef)

    assert violations == 0


# --- (b) reef_enabled=False costs zero extra RNG draws -----------------------


def test_transient_appearance_draws_no_extra_rng_when_reef_disabled():
    """reef_enabled=False must leave _reef_cells empty, so the appearance
    loop's `(i, j) not in self._reef_cells` filter removes nothing from the
    candidate list and therefore spends exactly one RNG draw per empty
    candidate cell -- the same draw count the loop had before this filter
    was added. This is the mechanism behind the fix's bit-identical-when-off
    guarantee (independently confirmed via a full-rollout hash comparison
    against the pre-fix code during development).
    """
    env = _mk(seed=5, reef_enabled=False)
    env.reset()
    assert env._reef_cells == set()

    if env.toroidal:
        n_candidates = int(np.sum(env.grid == env.ENTITY_TYPES["empty"]))
    else:
        interior = env.grid[1: env.size - 1, 1: env.size - 1]
        n_candidates = int(np.sum(interior == env.ENTITY_TYPES["empty"]))

    counting = _CountingRNG(np.random.default_rng(99))
    env._rng = counting
    env._step_transient_events()
    assert counting.random_calls == n_candidates, (
        f"expected exactly {n_candidates} RNG draws (one per empty candidate "
        f"cell, reef filter inert when off) but got {counting.random_calls}"
    )
