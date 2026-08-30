"""Contract: probe_warmup's cache KEY and surface RESTORE do not cross-contaminate
arms with different regulator configurations (IGW-20260830-222 / SD-PROBE-WARMUP,
confirmed by failure_autopsy_V3-EXQ-963_2026-08-30).

THE DEFECT THIS PINS (two parts, both fixed here, in probe_warmup.py only --
ree_core is healthy per the red-team pass this fix is routed from).

  (a) CACHE KEY EXCLUDED ARM FLAGS. `_warmup_key` hashed only (seed, recipe,
      env_kwargs). WarmupRecipe.as_dict() carries the warmup TRAINING schedule,
      never the caller's regulator flags (use_noise_floor, use_phasic_burst, ...),
      so a 2x2 grid sharing one seed+recipe+env hashed IDENTICALLY across all four
      arms regardless of which regulators each arm's agent was built with.

  (b) RESTORE UNCONDITIONALLY OVERWROTE. `_restore_cached_surface` did
      `object.__setattr__(module, name, value)` for every cached attribute with no
      check against what the live (already-constructed-per-its-OWN-config) HIT
      agent already had. Because probe_warmup's cache is keyed WITHOUT arm flags,
      the FIRST arm to run always mints and every later arm --including ones with
      DIFFERENT regulator flags-- silently cache-HITs and has the mint arm's
      regulator presence (real instance vs None) stamped over its own, live,
      already-correct one.

  Confirmed real-world instance: V3-EXQ-963's T0P0 arm (use_noise_floor=False)
  always minted first; `_restore_cached_surface` then wrote T0P0's
  `agent.noise_floor = None` over T1P0/T1P1's real, just-constructed NoiseFloor
  instances, silently zeroing the entire TONIC axis for the run
  (noise_floor_temp_lift_mean 0.0 on all 20 cells, including all 10
  use_noise_floor=True cells).

THE FIX, both parts:
  (a) `_warmup_key` now accepts an `arm_key: Optional[Mapping[str, Any]]` folded
      into the hashed payload, so two arms with different regulator flags mint
      distinct cache entries once a caller passes them.
  (b) `_restore_cached_surface` now refuses to write a cached attribute whose TYPE
      disagrees with the live HIT agent's OWN value for that attribute name
      (type(None) counts as a type, so None-vs-None still matches). This is
      SYMMETRIC -- it protects a live regulator from being zeroed by a cached
      None, AND protects a live None (an intentionally-absent regulator) from
      having an unrelated regulator installed by a cached non-None value. This is
      the PRIMARY defence: it holds even when a caller never passes `arm_key`,
      because REEAgent.__init__ has already built each arm's own regulator set,
      correctly, before warm_agent() is ever invoked -- the restore's only job is
      to not clobber that.

ASCII-only. Run: pytest tests/contracts/test_probe_warmup_cache_key_restore.py -q
"""

from __future__ import annotations

import torch

# conftest puts ree-v3 root on sys.path -> `experiments._lib.*` importable.
from experiments._lib.probe_warmup import (
    PROBE_WARMUP_SCHEMA,
    WarmupRecipe,
    _capture_cached_surface,
    _restore_cached_surface,
    _warmup_key,
)


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #

def _mk_agent(use_noise_floor: bool):
    """A real, minimal REEAgent with use_noise_floor toggled -- the exact V3-EXQ-963
    arm axis. Mirrors test_probe_warmup_nondestructive.py's `_mk_env_agent` shape.
    """
    from ree_core.agent import REEAgent
    from ree_core.environment.causal_grid_world import CausalGridWorldV2
    from ree_core.utils.config import REEConfig

    env = CausalGridWorldV2(seed=11, size=5, num_hazards=1, num_resources=1)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    cfg.use_noise_floor = use_noise_floor
    torch.manual_seed(0)
    return REEAgent(cfg)


def _noop_logger(_msg: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# (a) cache key differs across arm-flag combinations                          #
# --------------------------------------------------------------------------- #

def test_warmup_key_schema_is_v3_post_fix():
    # The schema bump forces every pre-fix (v2 and earlier) cache blob to MISS
    # regardless of whether a given caller opts into arm_key -- the fix must not
    # be silently defeated by a stale on-disk cache entry.
    assert PROBE_WARMUP_SCHEMA == "probe_warmup.v3"


def test_warmup_key_differs_across_arm_flag_values():
    recipe = WarmupRecipe(num_episodes=1, steps_per_episode=1)
    env_kwargs = {"size": 5, "num_hazards": 1}

    key_off = _warmup_key(
        seed=7, recipe=recipe, env_kwargs=env_kwargs,
        arm_key={"use_noise_floor": False},
    )
    key_on = _warmup_key(
        seed=7, recipe=recipe, env_kwargs=env_kwargs,
        arm_key={"use_noise_floor": True},
    )
    assert key_off != key_on, (
        "two arms differing ONLY in use_noise_floor must mint DISTINCT cache "
        "entries -- this is the V3-EXQ-963 defect (a): the mint/HIT collision "
        "across regulator-differing arms."
    )


def test_warmup_key_differs_across_multiple_flag_axes():
    recipe = WarmupRecipe(num_episodes=1, steps_per_episode=1)
    env_kwargs = {"size": 5}
    combos = [
        {"use_noise_floor": False, "use_phasic_burst": False},
        {"use_noise_floor": False, "use_phasic_burst": True},
        {"use_noise_floor": True, "use_phasic_burst": False},
        {"use_noise_floor": True, "use_phasic_burst": True},
    ]
    keys = [
        _warmup_key(seed=1, recipe=recipe, env_kwargs=env_kwargs, arm_key=combo)
        for combo in combos
    ]
    assert len(set(keys)) == len(combos), (
        "the full 2x2 (the literal V3-EXQ-963 grid shape) must mint FOUR distinct "
        "keys, not collapse onto one shared entry."
    )


def test_warmup_key_omitting_arm_key_equals_explicit_empty_dict():
    # None (the default, what every un-migrated caller passes today) must behave
    # identically to an explicit empty mapping -- no silent divergence between the
    # two spellings of "no arm-conditional flags supplied".
    recipe = WarmupRecipe(num_episodes=2, steps_per_episode=3)
    env_kwargs = {"size": 6}
    key_default = _warmup_key(seed=4, recipe=recipe, env_kwargs=env_kwargs)
    key_explicit_empty = _warmup_key(
        seed=4, recipe=recipe, env_kwargs=env_kwargs, arm_key={},
    )
    assert key_default == key_explicit_empty


def test_warmup_key_same_arm_key_reproducible():
    # Two calls with an IDENTICAL arm_key (the legitimate re-run / cache-reuse
    # case) must still produce the same key -- the fix must not turn every call
    # into a forced MISS.
    recipe = WarmupRecipe(num_episodes=3, steps_per_episode=4)
    env_kwargs = {"size": 5, "num_hazards": 2}
    arm_key = {"use_noise_floor": True, "use_phasic_burst": False}
    key_1 = _warmup_key(seed=9, recipe=recipe, env_kwargs=env_kwargs, arm_key=arm_key)
    key_2 = _warmup_key(seed=9, recipe=recipe, env_kwargs=env_kwargs, arm_key=dict(arm_key))
    assert key_1 == key_2


# --------------------------------------------------------------------------- #
# (b) restore never clobbers a configured regulator                           #
# --------------------------------------------------------------------------- #

def test_restore_does_not_zero_a_live_regulator_with_a_cached_none():
    """The exact V3-EXQ-963 shape: mint arm has NO regulator (None), restoring
    its surface onto a HIT agent that DOES have one must leave the HIT agent's
    real regulator instance intact."""
    mint_agent = _mk_agent(use_noise_floor=False)
    hit_agent = _mk_agent(use_noise_floor=True)

    assert mint_agent.noise_floor is None
    assert hit_agent.noise_floor is not None
    live_noise_floor = hit_agent.noise_floor

    cached_surface = _capture_cached_surface(mint_agent, _noop_logger)
    assert cached_surface["attrs"].get("", {}).get("noise_floor", "MISSING") is None, (
        "sanity: the mint agent's captured surface really does carry "
        "noise_floor=None at the root module path"
    )

    _restore_cached_surface(hit_agent, cached_surface, _noop_logger)

    assert hit_agent.noise_floor is not None, (
        "the HIT agent's live, correctly-constructed NoiseFloor regulator must "
        "survive a cache restore from a differently-configured mint -- this is "
        "the exact V3-EXQ-963 corruption (noise_floor silently zeroed, "
        "noise_floor_temp_lift_mean pinned at 0.0 on every cell)"
    )
    assert hit_agent.noise_floor is live_noise_floor, (
        "not merely non-None -- it must be the SAME live instance the HIT "
        "agent's own __init__ constructed, untouched by the restore"
    )


def test_restore_does_not_install_an_unconfigured_regulator():
    """The symmetric direction: a mint WITH a regulator restoring onto a HIT
    agent that was deliberately built WITHOUT one must not install it."""
    mint_agent = _mk_agent(use_noise_floor=True)
    hit_agent = _mk_agent(use_noise_floor=False)

    assert mint_agent.noise_floor is not None
    assert hit_agent.noise_floor is None

    cached_surface = _capture_cached_surface(mint_agent, _noop_logger)
    _restore_cached_surface(hit_agent, cached_surface, _noop_logger)

    assert hit_agent.noise_floor is None, (
        "a HIT agent whose own config never requested a regulator must not have "
        "one silently installed by a differently-configured mint's cached surface"
    )


def test_restore_still_applies_matching_type_state():
    """Non-regression: when the mint and HIT agents share the SAME regulator
    configuration (the legitimate cache-reuse case), the cached (warmed) regulator
    state must still be restored -- the fix must not turn every restore into a
    no-op."""
    mint_agent = _mk_agent(use_noise_floor=True)
    hit_agent = _mk_agent(use_noise_floor=True)

    assert mint_agent.noise_floor is not None
    assert hit_agent.noise_floor is not None
    original_hit_noise_floor = hit_agent.noise_floor

    cached_surface = _capture_cached_surface(mint_agent, _noop_logger)
    _restore_cached_surface(hit_agent, cached_surface, _noop_logger)

    assert hit_agent.noise_floor is not None
    # Same-type restore DOES replace the live instance's *content* with the
    # cached (mint's) state -- that is the whole point of a warm-start cache.
    # `_capture_cached_surface` deep-copies (via pickle round-trip), so this is
    # never the SAME object as the mint's -- only equal in type and state.
    assert type(hit_agent.noise_floor) is type(mint_agent.noise_floor)
    assert hit_agent.noise_floor is not original_hit_noise_floor
    assert hit_agent.noise_floor is not mint_agent.noise_floor


def test_restore_logs_clobber_skips_not_silent():
    mint_agent = _mk_agent(use_noise_floor=False)
    hit_agent = _mk_agent(use_noise_floor=True)

    messages = []
    _restore_cached_surface(
        hit_agent,
        _capture_cached_surface(mint_agent, _noop_logger),
        messages.append,
    )
    assert any("NOT restored" in m and "noise_floor" in m for m in messages), (
        "a skipped clobber must be reported, not swallowed"
    )
