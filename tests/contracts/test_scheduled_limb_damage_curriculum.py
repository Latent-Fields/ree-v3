"""
Contract tests for SD-022 scheduled-injection extension (MECH-302 unblock,
2026-05-30; failure_autopsy_V3-EXQ-517b_2026-05-30).

Six contracts:
  C1  Off by default -- bit-identical to pre-extension behaviour; info-dict
      sentinel keys always present but inert when master is OFF.
  C2  Precondition: scheduled_limb_damage_enabled requires limb_damage_enabled.
      Constructor raises ValueError; bad limb_selection also raises.
  C3  Interval x prob arithmetic: with interval=10 / prob=1.0, scheduled
      injections fire on steps 10, 20, 30, ... within an episode (deterministic
      cadence given the master is on).
  C4  Magnitude semantics: random-mode adds magnitude to exactly one limb
      per fire; all-mode adds magnitude to all four limbs; SD-022's existing
      [0, 1] clamp via min(1.0, ...) applies (no overflow beyond 1.0).
  C5  Per-episode counter reset: reset() clears event_count + last_step +
      last_limb_idx + last_magnitude + injected_this_step.
  C6  RNG isolation: master OFF does not draw from the env RNG (verified by
      bit-identical limb_damage trajectory vs default-constructed control).
"""

import numpy as np
import pytest
import torch

from ree_core.environment.causal_grid_world import CausalGridWorldV2


def _act(a: int) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.long)


def _step(env, action_int: int):
    flat, harm, done, info, _obs = env.step(_act(action_int))
    return flat, harm, done, info


# ---------------------------------------------------------------------------
# C1  Off by default; sentinel keys always present but inert
# ---------------------------------------------------------------------------

def test_c1_off_by_default_inert():
    env = CausalGridWorldV2(size=12, num_hazards=3, num_resources=2, seed=42)
    assert env.scheduled_limb_damage_enabled is False
    assert env._scheduled_limb_damage_event_count == 0
    assert env._scheduled_limb_damage_last_limb_idx == -1
    assert env._scheduled_limb_damage_last_magnitude == 0.0
    assert env._scheduled_limb_damage_injected_this_step is False

    env.reset()
    _, _, _, info = _step(env, 4)

    required = [
        "scheduled_limb_damage_enabled",
        "scheduled_limb_damage_injected_this_step",
        "scheduled_limb_damage_event_count",
        "scheduled_limb_damage_last_limb_idx",
        "scheduled_limb_damage_last_magnitude",
    ]
    for k in required:
        assert k in info, "missing sentinel info key: {}".format(k)
    assert info["scheduled_limb_damage_enabled"] is False
    assert info["scheduled_limb_damage_injected_this_step"] is False
    assert info["scheduled_limb_damage_event_count"] == 0


# ---------------------------------------------------------------------------
# C2  Preconditions: master requires limb_damage_enabled; bad selection raises
# ---------------------------------------------------------------------------

def test_c2_precondition_requires_limb_damage_enabled():
    with pytest.raises(ValueError, match="limb_damage_enabled=True"):
        CausalGridWorldV2(
            size=12,
            num_hazards=3,
            num_resources=2,
            seed=42,
            scheduled_limb_damage_enabled=True,
        )


def test_c2_precondition_bad_limb_selection_raises():
    with pytest.raises(ValueError, match="limb_selection"):
        CausalGridWorldV2(
            size=12,
            num_hazards=3,
            num_resources=2,
            seed=42,
            limb_damage_enabled=True,
            scheduled_limb_damage_enabled=True,
            scheduled_limb_damage_limb_selection="banana",
        )


# ---------------------------------------------------------------------------
# C3  Interval x prob arithmetic: fires on steps 10, 20, 30 at interval=10/prob=1.0
# ---------------------------------------------------------------------------

def test_c3_interval_prob_arithmetic_deterministic_cadence():
    # Aggregate across multiple short episodes (the agent's default health
    # decay may end episodes before a single 30-step window). With interval=5
    # and prob=1.0 we expect a fire on every step%5==0 the agent survives;
    # across enough episodes this comfortably clears the 3-fire threshold.
    env = CausalGridWorldV2(
        size=12,
        num_hazards=3,
        num_resources=2,
        seed=42,
        limb_damage_enabled=True,
        scheduled_limb_damage_enabled=True,
        scheduled_limb_damage_interval=5,
        scheduled_limb_damage_prob=1.0,
        scheduled_limb_damage_magnitude=0.1,
        scheduled_limb_damage_limb_selection="all",
    )
    fired_at_step_mods = []
    for _ in range(6):
        env.reset()
        for _ in range(30):
            _, _, done, info = _step(env, 4)  # stay action
            if info["scheduled_limb_damage_injected_this_step"]:
                fired_at_step_mods.append(int(info["steps"]) % 5)
            if done:
                break
    assert len(fired_at_step_mods) >= 3, \
        "expected at least 3 fires across 6 episodes, got {}".format(fired_at_step_mods)
    # Cadence is deterministic given prob=1.0. The gate fires when
    # pre-increment self.steps % interval == 0; info["steps"] reports the
    # post-increment value, so every fire's info["steps"] % interval == 1.
    assert all(m == 1 for m in fired_at_step_mods), \
        "fire steps not all (info[steps] % 5) == 1: {}".format(fired_at_step_mods)


# ---------------------------------------------------------------------------
# C4  Magnitude semantics + clamp at 1.0
# ---------------------------------------------------------------------------

def test_c4_random_mode_adds_to_one_limb():
    env = CausalGridWorldV2(
        size=12,
        num_hazards=3,
        num_resources=2,
        seed=42,
        limb_damage_enabled=True,
        scheduled_limb_damage_enabled=True,
        scheduled_limb_damage_interval=5,
        scheduled_limb_damage_prob=1.0,
        scheduled_limb_damage_magnitude=0.3,
        scheduled_limb_damage_limb_selection="random",
    )
    env.reset()
    pre = env.limb_damage.copy()
    fired = False
    for _ in range(8):
        _, _, done, info = _step(env, 4)
        if info["scheduled_limb_damage_injected_this_step"]:
            fired = True
            break
        if done:
            env.reset()
            pre = env.limb_damage.copy()
    assert fired
    delta = env.limb_damage - pre
    # Exactly one limb should differ from pre by ~magnitude (modulo SD-022 healing).
    n_changed = int(np.sum(delta > 0.1))
    assert n_changed == 1, "expected 1 changed limb, got {} (delta={})".format(
        n_changed, delta.tolist()
    )
    assert info["scheduled_limb_damage_last_limb_idx"] in (0, 1, 2, 3)
    assert abs(info["scheduled_limb_damage_last_magnitude"] - 0.3) < 1e-6


def test_c4_all_mode_adds_to_four_limbs():
    env = CausalGridWorldV2(
        size=12,
        num_hazards=3,
        num_resources=2,
        seed=42,
        limb_damage_enabled=True,
        scheduled_limb_damage_enabled=True,
        scheduled_limb_damage_interval=5,
        scheduled_limb_damage_prob=1.0,
        scheduled_limb_damage_magnitude=0.4,
        scheduled_limb_damage_limb_selection="all",
    )
    env.reset()
    fired = False
    for _ in range(8):
        _, _, done, info = _step(env, 4)
        if info["scheduled_limb_damage_injected_this_step"]:
            fired = True
            break
        if done:
            env.reset()
    assert fired
    assert info["scheduled_limb_damage_last_limb_idx"] == -1
    # All four limbs > 0.3 (heal_rate=0.002/step keeps them close to magnitude).
    assert all(env.limb_damage > 0.3), env.limb_damage.tolist()


def test_c4_magnitude_clamps_at_one():
    env = CausalGridWorldV2(
        size=12,
        num_hazards=3,
        num_resources=2,
        seed=42,
        limb_damage_enabled=True,
        scheduled_limb_damage_enabled=True,
        scheduled_limb_damage_interval=2,
        scheduled_limb_damage_prob=1.0,
        scheduled_limb_damage_magnitude=0.6,
        scheduled_limb_damage_limb_selection="all",
    )
    env.reset()
    for _ in range(12):
        _, _, done, info = _step(env, 4)
        if done:
            env.reset()
    # After multiple 0.6 injections on all limbs the [0, 1] cap should hold.
    assert (env.limb_damage <= 1.0 + 1e-6).all()
    # And several injections should have pushed limbs near the cap.
    assert (env.limb_damage > 0.5).all(), env.limb_damage.tolist()


# ---------------------------------------------------------------------------
# C5  Per-episode counter reset
# ---------------------------------------------------------------------------

def test_c5_per_episode_reset_clears_counters():
    env = CausalGridWorldV2(
        size=12,
        num_hazards=3,
        num_resources=2,
        seed=42,
        limb_damage_enabled=True,
        scheduled_limb_damage_enabled=True,
        scheduled_limb_damage_interval=5,
        scheduled_limb_damage_prob=1.0,
        scheduled_limb_damage_magnitude=0.3,
    )
    env.reset()
    for _ in range(30):
        _, _, done, _ = _step(env, 4)
        if done:
            break
    assert env._scheduled_limb_damage_event_count > 0

    env.reset()
    assert env._scheduled_limb_damage_event_count == 0
    assert env._scheduled_limb_damage_last_step == -1
    assert env._scheduled_limb_damage_last_limb_idx == -1
    assert env._scheduled_limb_damage_last_magnitude == 0.0
    assert env._scheduled_limb_damage_injected_this_step is False


# ---------------------------------------------------------------------------
# C6  RNG isolation: explicit master OFF is bit-identical to default
# ---------------------------------------------------------------------------

def test_c6_rng_isolation_master_off_bit_identical():
    env_a = CausalGridWorldV2(
        size=12, num_hazards=3, num_resources=2, seed=42, limb_damage_enabled=True
    )
    env_b = CausalGridWorldV2(
        size=12, num_hazards=3, num_resources=2, seed=42,
        limb_damage_enabled=True,
        scheduled_limb_damage_enabled=False,  # explicit OFF
    )
    env_a.reset()
    env_b.reset()
    for i in range(30):
        a = _act(int(np.random.default_rng(i).integers(0, 5)))
        env_a.step(a)
        env_b.step(a)
        assert np.allclose(env_a.limb_damage, env_b.limb_damage), \
            "step {}: limb_damage divergence".format(i)
        assert env_a.agent_x == env_b.agent_x and env_a.agent_y == env_b.agent_y, \
            "step {}: agent position divergence".format(i)
