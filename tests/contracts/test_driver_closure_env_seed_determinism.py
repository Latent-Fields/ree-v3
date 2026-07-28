"""Contracts for the DRIVER-OWNED env-seed knob on V3-EXQ-460c.

WHAT THIS PINS, and why it is separate from
`test_scaffold_env_seed_determinism.py`. That contract covers
`ScaffoldedSD054OnboardingConfig.scaffold_env_seed`, which seeds the envs the
SCHEDULER builds. It deliberately does not cover a driver that builds its own
eval env -- 460c's `_build_closure_env` is the named residual. Measured on the
460c dry-run with `scaffold_env_seed` set: cross-process divergence fell from 31
differing result leaves to 12, and ALL 12 were downstream of `_build_closure_env`
(10 inside the ARM_CLOSURE_ON / ARM_CLOSURE_OFF blocks, plus
`commitment_completion_non_vacuity` and `criteria/C4`, which derive from them).
Every training-stage output was already identical. So the training curriculum
reproduced and the closure eval arms did not; this closes that half.

Mechanism, restated because it is the whole reason a driver-side fix was needed:
`CausalGridWorldV2` is a FACTORY FUNCTION forwarding to `CausalGridWorld.__init__`,
whose only RNG is `self._rng = np.random.default_rng(seed)`.
`np.random.default_rng(None)` takes OS entropy AT CONSTRUCTION and does NOT consume
the numpy global RNG, so `_run_seed`'s `np.random.seed(seed)` /
`torch.manual_seed(seed)` never reached the eval env. The defect is per
CONSTRUCTION, not per process.

Triage:
REE_assembly/evidence/planning/scaffold_goal_freeze_e3_read_path_triage_2026-07-27.md

BOTH directions are pinned here and both matter:
  * knob UNSET -> `seed=None` still reaches the factory, and `scaffold_env_seed`
    stays None, i.e. bit-identical to the landed 460c run;
  * knob SET   -> deterministic, distinct, reproducible seeds that cannot collide
    with the scheduler's own streams, and a same-seed env reproduces bitwise.

Env-level rather than whole-curriculum on purpose: the causal chain is
seed -> env layout -> observations, and a full 460c dry-run costs minutes.
"""
import os
import sys

import numpy as np
import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_REPO, os.path.join(_REPO, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import v3_exq_460c_sd034_verified_but_not_released_behavioural as drv  # noqa: E402
from scaffolded_sd054_onboarding import _derive_env_seed  # noqa: E402

# Stream ids, restated as constants so a future edit that renumbers them fails
# here rather than silently colliding two callers onto one env layout.
CURRICULUM_STREAM = 0   # scaffolded_sd054_onboarding scheduler builds
PROBE_STREAM = 1        # scaffolded_sd054_onboarding harm-discriminativeness probe
DRIVER_STREAM = 2       # this driver's own eval envs


# --------------------------------------------------------------------------- #
# Default path: bit-identical to the landed 460c run
# --------------------------------------------------------------------------- #

def test_build_closure_env_seed_defaults_to_none():
    """The knob must default OFF. Pinning unconditionally would change the eval
    env layout, and therefore the result, for a run already on the record."""
    import inspect
    assert inspect.signature(drv._build_closure_env).parameters["seed"].default is None


def test_default_hands_seed_none_to_the_factory(monkeypatch):
    """Not merely 'omits seed' -- the factory must still SEE None, which is what
    makes the explicit pass-through bit-identical to the pre-knob call."""
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return object()

    # 460c imports the factory into its OWN namespace, so patch it there --
    # patching ree_core.environment.causal_grid_world would not be seen.
    monkeypatch.setattr(drv, "CausalGridWorldV2", _capture)
    drv._build_closure_env(drv._make_scaffold_cfg(dry_run=True))
    assert "seed" in seen and seen["seed"] is None


def test_default_scaffold_cfg_leaves_the_scheduler_knob_unset():
    """The driver must not switch the SCHEDULER's envs on by accident: with no
    --env-seed, `scaffold_env_seed` stays None and the scheduler's own default
    path (documented in test_scaffold_env_seed_determinism.py) is untouched."""
    assert drv._make_scaffold_cfg(dry_run=True).scaffold_env_seed is None


def test_none_base_derives_none_for_both_driver_constructions():
    """The two `_build_closure_env` call sites in `_run_seed` (idx 0 = the
    world_obs_dim parity probe env, idx 1 = the closure eval env)."""
    for idx in (0, 1):
        assert _derive_env_seed(None, stream=DRIVER_STREAM, idx=idx) is None


# --------------------------------------------------------------------------- #
# Knob SET: deterministic, distinct, collision-free
# --------------------------------------------------------------------------- #

def test_set_scaffold_cfg_threads_the_base_through():
    assert drv._make_scaffold_cfg(dry_run=True, env_seed=4242).scaffold_env_seed == 4242


def test_build_closure_env_forwards_the_seed(monkeypatch):
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(drv, "CausalGridWorldV2", _capture)
    drv._build_closure_env(drv._make_scaffold_cfg(dry_run=True), seed=4242)
    assert seen["seed"] == 4242


def test_driver_stream_cannot_collide_with_the_scheduler_streams():
    """The reason the driver takes a stream of its own. If these ever collide,
    the eval env silently shares a layout with a curriculum stage."""
    base = 99
    driver = {_derive_env_seed(base, stream=DRIVER_STREAM, idx=i) for i in range(64)}
    scheduler = {
        _derive_env_seed(base, stream=s, idx=i)
        for s in (CURRICULUM_STREAM, PROBE_STREAM)
        for i in range(64)
    }
    assert not (driver & scheduler)


def test_the_two_driver_constructions_get_distinct_seeds():
    """Probe env and closure eval env must not share a layout -- they are built
    for different purposes and one is stepped by a trained agent."""
    base = 7
    assert (_derive_env_seed(base, stream=DRIVER_STREAM, idx=0)
            != _derive_env_seed(base, stream=DRIVER_STREAM, idx=1))


def test_run_seeds_do_not_collapse_onto_one_layout():
    """`_run_seed` folds the run seed into the base (`env_seed_base + seed`).
    Without that, one --env-seed would give all three SEEDS the same world and
    the seed dimension would be silently destroyed."""
    base = 1000
    per_seed = {
        _derive_env_seed(base + s, stream=DRIVER_STREAM, idx=idx)
        for s in drv.SEEDS
        for idx in (0, 1)
    }
    assert len(per_seed) == 2 * len(drv.SEEDS)


# --------------------------------------------------------------------------- #
# The property that actually matters: same seed -> same env trajectory
# --------------------------------------------------------------------------- #

def _rollout_fingerprint(seed, n_steps=12):
    """Deterministic scripted rollout of the CLOSURE env -- no policy, so only
    the env is under test."""
    env = drv._build_closure_env(drv._make_scaffold_cfg(dry_run=True), seed=seed)
    obs, _info = env.reset()
    arr = obs.detach().cpu().numpy() if torch.is_tensor(obs) else np.asarray(obs)
    parts = [arr.copy(), np.array([env.agent_x, env.agent_y], dtype=np.float64)]
    for t in range(n_steps):
        nxt, _reward, done, _info, _obs_dict = env.step(t % env.action_dim)
        parts.append(
            nxt.detach().cpu().numpy() if torch.is_tensor(nxt) else np.asarray(nxt)
        )
        parts.append(np.array([env.agent_x, env.agent_y], dtype=np.float64))
        if done:
            # Length is part of the fingerprint -- an earlier termination gives a
            # shorter vector, which array_equal already catches.
            break
    return np.concatenate([p.ravel().astype(np.float64) for p in parts])


def test_same_seed_gives_bit_identical_closure_env_trajectory():
    """The causal link the knob buys: seed -> layout -> observations."""
    a = _rollout_fingerprint(20260728)
    b = _rollout_fingerprint(20260728)
    assert np.array_equal(a, b), (
        "two closure envs built with the SAME explicit seed diverged -- the env "
        "carries an entropy source beyond its own _rng, and seeding the driver "
        "cannot make the 460c eval arms reproducible"
    )


def test_global_reseeding_does_not_rescue_the_unseeded_closure_env():
    """Pins the DEFECT itself, deterministically, on the 460c call path.

    `np.random.default_rng(None)` ignores the numpy global RNG, so reseeding the
    globals identically before each construction -- exactly what `_run_seed` does
    -- cannot make unseeded eval envs agree. Asserted as 'not all identical' over
    several builds rather than 'the first two differ', so a chance layout
    collision cannot flake it.
    """
    fps = []
    for _ in range(6):
        np.random.seed(42)
        torch.manual_seed(42)
        fps.append(_rollout_fingerprint(None, n_steps=4))
    assert not all(np.array_equal(fps[0], f) for f in fps[1:]), (
        "unseeded closure envs became reproducible under global reseeding -- if "
        "the env RNG was genuinely made seed-following, retire this test and the "
        "knob rather than weakening the assertion"
    )


def test_seeded_and_unseeded_closure_envs_coexist():
    """A seeded build must not perturb a later unseeded one into determinism, or
    vice versa -- the default path has to survive alongside a pinned one."""
    seeded_1 = _rollout_fingerprint(777, n_steps=4)
    _ = _rollout_fingerprint(None, n_steps=4)
    seeded_2 = _rollout_fingerprint(777, n_steps=4)
    assert np.array_equal(seeded_1, seeded_2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
