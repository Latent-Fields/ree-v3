"""Contract: SD-MEL-PRODUCER non-converging-world action-map re-permutation.

Background -- the MECH-180 / INV-050 producer gap. The claim splits into
link (i) novelty -> graded above-reference waking MEL and link (ii) MEL ->
graded offline duration. Link (ii) is BUILT + PROVEN (SD-MEL-CONSUMER;
V3-EXQ-718a injection positive control). Link (i) has never been shown, and
three runs establish it cannot be shown with `env_drift_interval`:

  V3-EXQ-677   high- vs low-novelty mean E1 prediction error differed by
               8.8e-07 against a 0.01 threshold -- NO novelty gradient.
  V3-EXQ-718   C1 novelty-label monotonicity 0/3.
  V3-EXQ-718a  ecological measured MEL ~1e-5, noise-level and scrambled vs
               novelty level; conv_rel_drop ~0.98.

Root cause: `_drift_hazards()` only MOVES hazards. The optimal prediction of
a random walk is its mean, so the world-forward model learns that fast and PE
floors at the irreducible noise level. Drift adds sampling noise, not
learning load.

This knob instead re-permutes the action -> displacement map, so
E2.world_forward(z_world, a) -- which takes the action as an input -- becomes
systematically wrong until re-learned. Learnable between shifts, invalidated
at each shift.

The tests below pin the four properties that are load-bearing and were each
either a live defect during implementation or a hazard that would silently
corrupt other work. See REE_assembly/docs/architecture/sd_mel_producer.md.
"""

import hashlib

import numpy as np
import pytest

from ree_core.environment.causal_grid_world import CausalGridWorld


IDENTITY_ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}


def _rollout(seed, n_steps=300, n_episodes=1, **kw):
    """Deterministic rollout; returns (trajectory_digest, env)."""
    env = CausalGridWorld(size=10, seed=seed, use_proxy_fields=True, **kw)
    rng = np.random.default_rng(1234)
    h = hashlib.sha256()
    for _ep in range(n_episodes):
        env.reset()
        for _t in range(n_steps):
            action = int(rng.integers(0, 5))
            env.step(action)
            h.update(
                f"{env.agent_x},{env.agent_y},"
                f"{env.total_harm:.6f},{env.total_benefit:.6f};".encode()
            )
    return h.hexdigest(), env


# ---------------------------------------------------------------------------
# 1. Backward compatibility. THE defect this guards against is silent: a stray
#    RNG draw in the disabled path desynchronises every existing experiment's
#    seeded rollout, with no error and no visible signal. Every run predating
#    SD-MEL-PRODUCER depends on the disabled env being bit-identical.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 123, 456])
def test_disabled_env_is_bit_identical_to_absent_feature(seed):
    """Explicitly-off (with interval/depth set) == feature absent entirely."""
    digest_absent, env_absent = _rollout(seed)
    digest_off, env_off = _rollout(
        seed,
        world_rule_shift_enabled=False,
        world_rule_shift_interval=20,
        world_rule_shift_depth=2,
    )
    assert digest_absent == digest_off, (
        "disabled world_rule_shift changed the seeded trajectory -- a guarded "
        "RNG draw has leaked into the disabled path"
    )
    assert env_absent._world_rule_shift_count == 0
    assert env_off._world_rule_shift_count == 0


def test_disabled_env_consumes_no_randomness():
    """The interval/depth values must not matter at all when disabled."""
    digests = {
        _rollout(
            42,
            world_rule_shift_enabled=False,
            world_rule_shift_interval=interval,
            world_rule_shift_depth=depth,
        )[0]
        for interval, depth in ((0, 0), (1, 4), (7, 2), (999, 1))
    }
    assert len(digests) == 1, "disabled path is sensitive to its own parameters"


# ---------------------------------------------------------------------------
# 2. The class-level ACTIONS dict must never be mutated. It is a CLASS
#    attribute, so an in-place permutation would leak into every other
#    CausalGridWorld instance in the process -- silently corrupting unrelated
#    experiments sharing the interpreter.
# ---------------------------------------------------------------------------

def test_class_actions_dict_is_never_mutated():
    _, env = _rollout(
        42,
        n_steps=400,
        world_rule_shift_enabled=True,
        world_rule_shift_interval=5,
        world_rule_shift_depth=2,
    )
    assert env._world_rule_shift_count > 0, "test did not exercise any shift"
    assert CausalGridWorld.ACTIONS == IDENTITY_ACTIONS, (
        "the class-level ACTIONS dict was mutated -- the permutation will leak "
        "into every other env instance in this process"
    )
    assert env._action_map != CausalGridWorld.ACTIONS, (
        "shifts fired but the instance map is still identity"
    )


def test_two_envs_do_not_share_an_action_map():
    _, shifted = _rollout(
        42,
        n_steps=400,
        world_rule_shift_enabled=True,
        world_rule_shift_interval=5,
        world_rule_shift_depth=2,
    )
    fresh = CausalGridWorld(size=10, seed=99, use_proxy_fields=True)
    assert shifted._world_rule_shift_count > 0
    assert fresh._action_map == IDENTITY_ACTIONS, (
        "a fresh env inherited another env's permuted action map"
    )


# ---------------------------------------------------------------------------
# 3. The schedule must key off CUMULATIVE world time, not episode-local
#    self.steps. This was a live defect: because episode length itself
#    collapses as the world becomes less predictable, an episode-relative
#    schedule makes the nominal interval stop controlling the actual shift
#    RATE. Measured on the first probe, intervals 60/30/15/8/5 produced
#    2/2/3/20/21 shifts -- non-monotone, taking the MEL ladder with it.
# ---------------------------------------------------------------------------

def test_shift_count_scales_monotonically_with_rate_across_episodes():
    """Shorter interval => strictly more shifts, over a FIXED step budget
    split across several episodes (the case episode-relative scheduling got
    wrong)."""
    counts = []
    for interval in (60, 30, 15, 5):
        _, env = _rollout(
            42,
            n_steps=40,
            n_episodes=6,
            world_rule_shift_enabled=True,
            world_rule_shift_interval=interval,
            world_rule_shift_depth=2,
        )
        counts.append(env._world_rule_shift_count)
    assert all(counts[i] < counts[i + 1] for i in range(len(counts) - 1)), (
        f"shift count not monotone in shift rate across episodes: {counts} -- "
        "the schedule has regressed to episode-local self.steps"
    )


def test_shift_schedule_survives_episode_reset():
    """An interval LONGER than one episode must still fire."""
    _, env = _rollout(
        42,
        n_steps=10,
        n_episodes=10,
        world_rule_shift_enabled=True,
        world_rule_shift_interval=25,
        world_rule_shift_depth=2,
    )
    assert env._world_rule_shift_count >= 3, (
        f"interval 25 over ~100 world-steps fired only "
        f"{env._world_rule_shift_count} shifts -- episode-local scheduling"
    )


def test_action_map_is_not_reset_per_episode():
    """The action map is world causal structure, not episode state. Resetting
    it to identity each episode would hand the forward model a fixed anchor to
    re-converge on and defeat the point of a non-converging world."""
    env = CausalGridWorld(
        size=10, seed=42, use_proxy_fields=True,
        world_rule_shift_enabled=True,
        world_rule_shift_interval=5,
        world_rule_shift_depth=2,
    )
    rng = np.random.default_rng(0)
    env.reset()
    for _t in range(40):
        env.step(int(rng.integers(0, 5)))
    permuted = dict(env._action_map)
    assert permuted != IDENTITY_ACTIONS, "test did not achieve a permutation"
    count_before = env._world_rule_shift_count

    env.reset()
    assert env._action_map == permuted, (
        "reset() restored the identity action map -- the world's causal "
        "structure must persist across episodes"
    )
    assert env._world_rule_shift_count == count_before


# ---------------------------------------------------------------------------
# 4. The instrument. steps_since_world_rule_shift is what lets a consumer
#    distinguish genuine re-learning load (PE decays within a stationary
#    window) from graded NOISE (PE does not decay). Without it, a graded MEL
#    gradient is a DV-symmetry artifact -- the delta fixed before the run,
#    the class that held V3-EXQ-683.
# ---------------------------------------------------------------------------

def test_info_exposes_the_learnability_instrument():
    env = CausalGridWorld(
        size=10, seed=42, use_proxy_fields=True,
        world_rule_shift_enabled=True,
        world_rule_shift_interval=5,
        world_rule_shift_depth=2,
    )
    env.reset()
    rng = np.random.default_rng(0)
    seen_shift = False
    max_since = 0
    for _t in range(60):
        info = env.step(int(rng.integers(0, 5)))[3]
        for key in (
            "world_rule_shift_occurred",
            "world_rule_shift_count",
            "steps_since_world_rule_shift",
        ):
            assert key in info, f"info is missing {key}"
        if info["world_rule_shift_occurred"]:
            seen_shift = True
            assert info["steps_since_world_rule_shift"] == 0, (
                "steps_since_world_rule_shift must be 0 on the shift step"
            )
        max_since = max(max_since, info["steps_since_world_rule_shift"])
    assert seen_shift, "no shift fired in 60 steps at interval 5"
    assert max_since > 0, "steps_since_world_rule_shift never advanced"


def test_depth_zero_shifts_schedule_without_changing_the_rule():
    """Depth 0 is legal on purpose: it isolates the ACT of shifting from the
    CONTENT of the shift, which is the control arm for 'does the schedule
    itself perturb anything'."""
    env = CausalGridWorld(
        size=10, seed=42, use_proxy_fields=True,
        world_rule_shift_enabled=True,
        world_rule_shift_interval=5,
        world_rule_shift_depth=0,
    )
    env.reset()
    rng = np.random.default_rng(0)
    for _t in range(60):
        env.step(int(rng.integers(0, 5)))
    assert env._world_rule_shift_count > 0, "depth-0 schedule did not fire"
    assert env._action_map == IDENTITY_ACTIONS, (
        "depth 0 changed the action map"
    )


def test_invalid_scope_is_rejected():
    with pytest.raises(ValueError, match="world_rule_shift_scope"):
        CausalGridWorld(size=10, seed=1, world_rule_shift_scope="bogus")
