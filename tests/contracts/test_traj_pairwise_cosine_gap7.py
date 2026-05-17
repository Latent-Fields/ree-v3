"""
Contract tests for infant_substrate:GAP-7 -- traj_pairwise_cosine_mean telemetry.

Five contracts:

  C1  Off when explicitly disabled -- sentinel info keys present but inert
      (-1.0 / 0 stored); default-ON vs explicit-OFF produce bit-identical
      env layouts (telemetry has no RNG in env dynamics and never feeds back
      into obs/reward/done).

  C2  Store growth -- each completed episode adds one entry (capped at
      traj_max_stored); traj_n_episodes_stored echoes the buffer length;
      metric stays at -1.0 sentinel until at least 2 episodes complete.

  C3  Metric update timing -- traj_pairwise_cosine_mean updates only on the
      done=True step, not on intermediate steps; value returned on the done
      step matches the cached value on subsequent steps in the next episode.

  C4  Cosine metric properties -- two identical episodes yield mean ~0.0;
      two episodes with non-overlapping trajectories yield mean > 0.0;
      metric is in [0, 1] once 2+ episodes are stored.

  C5  reset() clears _traj_current but NOT _traj_store or the cached metric;
      reset_to() has the same behaviour.
"""
import math

import numpy as np
import pytest

from ree_core.environment.causal_grid_world import CausalGridWorldV2


def _run_episode(env, action_seq=None, n_steps=None):
    """Step the env for one full episode. Returns list of info dicts."""
    infos = []
    step = 0
    while True:
        if action_seq is not None:
            action = action_seq[step % len(action_seq)]
        else:
            action = np.random.randint(0, 4)
        _, _, done, info, _ = env.step(action)
        infos.append(info)
        step += 1
        if done:
            break
        if n_steps is not None and step >= n_steps:
            break
    return infos


def _env_layout(env):
    """Sorted agent + hazard + resource positions after a reset()."""
    env.reset()
    return (
        (env.agent_x, env.agent_y),
        sorted(tuple(h) for h in env.hazards),
        sorted(tuple(r) for r in env.resources),
    )


# --------------------------------------------------------------------------- #
# C1 -- Off when disabled (sentinels + bit-identical layout)
# --------------------------------------------------------------------------- #

def test_c1_sentinel_keys_inert_when_disabled():
    env = CausalGridWorldV2(
        size=12, seed=7, num_hazards=2, traj_telemetry_enabled=False
    )
    env.reset()
    _, _, _, info, _ = env.step(0)
    assert info["traj_telemetry_enabled"] is False
    assert info["traj_pairwise_cosine_mean"] == -1.0
    assert info["traj_n_episodes_stored"] == 0
    assert env._traj_store == []
    assert env._traj_current == []


def test_c1_default_on_vs_explicit_off_bit_identical_layout():
    """Telemetry has no feedback into env dynamics; layouts must match."""
    a = CausalGridWorldV2(size=12, seed=42, num_hazards=3, num_resources=4)
    b = CausalGridWorldV2(
        size=12, seed=42, num_hazards=3, num_resources=4,
        traj_telemetry_enabled=False,
    )
    for _ in range(5):
        assert _env_layout(a) == _env_layout(b)


def test_c1_no_telemetry_across_episode_when_disabled():
    env = CausalGridWorldV2(
        size=12, seed=7, num_hazards=2, traj_telemetry_enabled=False
    )
    env.reset()
    episodes = 0
    for _ in range(2000):
        _, _, done, info, _ = env.step(np.random.randint(0, 4))
        assert info["traj_telemetry_enabled"] is False
        assert info["traj_pairwise_cosine_mean"] == -1.0
        assert info["traj_n_episodes_stored"] == 0
        if done:
            episodes += 1
            if episodes >= 3:
                break
            env.reset()
    assert env._traj_store == []
    assert env._traj_current == []


# --------------------------------------------------------------------------- #
# C2 -- Store growth (capped at traj_max_stored; sentinel until 2 episodes)
# --------------------------------------------------------------------------- #

def test_c2_store_grows_one_per_episode():
    env = CausalGridWorldV2(
        size=10, seed=1, num_hazards=0, num_resources=0,
        traj_max_stored=5,
    )
    env.reset()
    for ep in range(1, 7):
        infos = _run_episode(env)
        stored = infos[-1]["traj_n_episodes_stored"]
        expected = min(ep, 5)
        assert stored == expected, f"ep={ep}: expected {expected}, got {stored}"
        env.reset()


def test_c2_sentinel_until_two_episodes():
    env = CausalGridWorldV2(
        size=10, seed=2, num_hazards=0, num_resources=0,
    )
    env.reset()
    # Episode 1: all intermediate steps should report -1.0; final step too
    # (store has 1 entry after ep1, still sentinel).
    infos_ep1 = _run_episode(env)
    for info in infos_ep1:
        if info["traj_n_episodes_stored"] < 2:
            assert info["traj_pairwise_cosine_mean"] == -1.0
    env.reset()
    # Episode 2: final step should have traj_n_episodes_stored == 2 and
    # traj_pairwise_cosine_mean != -1.0.
    infos_ep2 = _run_episode(env)
    last = infos_ep2[-1]
    assert last["traj_n_episodes_stored"] == 2
    assert last["traj_pairwise_cosine_mean"] != -1.0


def test_c2_store_caps_at_max_stored():
    max_stored = 4
    env = CausalGridWorldV2(
        size=10, seed=3, num_hazards=0, num_resources=0,
        traj_max_stored=max_stored,
    )
    env.reset()
    for _ in range(max_stored + 3):
        _run_episode(env)
        env.reset()
    assert len(env._traj_store) == max_stored


# --------------------------------------------------------------------------- #
# C3 -- Metric update timing (only on done step; cached across episode)
# --------------------------------------------------------------------------- #

def test_c3_metric_updates_only_at_episode_end():
    """Intermediate steps within an episode must not change the metric."""
    env = CausalGridWorldV2(
        size=10, seed=5, num_hazards=0, num_resources=0,
    )
    env.reset()
    # Run two episodes to seed the store.
    _run_episode(env)
    env.reset()
    _run_episode(env)
    env.reset()
    # Third episode: collect all metric values; only the done step should
    # differ from the value inherited from the end of episode 2.
    baseline = env._traj_pairwise_cosine_mean
    done_values = []
    mid_values = []
    for _ in range(600):
        _, _, done, info, _ = env.step(np.random.randint(0, 4))
        if done:
            done_values.append(info["traj_pairwise_cosine_mean"])
            break
        else:
            mid_values.append(info["traj_pairwise_cosine_mean"])
    # All mid-episode steps return the baseline (not yet updated).
    for v in mid_values:
        assert v == pytest.approx(baseline, abs=1e-9)
    # The done step returns the newly-computed value.
    assert len(done_values) == 1
    # (Could equal baseline if episodes are identical, so just check range.)
    assert 0.0 <= done_values[0] <= 1.0


def test_c3_cached_value_persists_into_next_episode():
    """Value from episode-end persists on first steps of the next episode."""
    env = CausalGridWorldV2(
        size=10, seed=6, num_hazards=0, num_resources=0,
    )
    env.reset()
    _run_episode(env)
    env.reset()
    infos = _run_episode(env)
    end_val = infos[-1]["traj_pairwise_cosine_mean"]
    env.reset()
    _, _, _, first_info, _ = env.step(0)
    assert first_info["traj_pairwise_cosine_mean"] == pytest.approx(end_val, abs=1e-9)


# --------------------------------------------------------------------------- #
# C4 -- Cosine metric properties
# --------------------------------------------------------------------------- #

def test_c4_identical_episodes_zero_distance():
    """Two identical histograms give cosine distance 0.

    We inject the histograms directly rather than driving two env episodes,
    because reset() randomises the start position (changing the trajectory
    even with the same action sequence). This isolates the cosine math."""
    env = CausalGridWorldV2(size=10, seed=0, num_hazards=0, num_resources=0)
    env.reset()
    # Construct a non-trivial histogram (uniform over 4 cells).
    hist = np.zeros(10 * 10, dtype=np.float32)
    for cell in [11, 22, 33, 44]:
        hist[cell] = 0.25
    env._traj_store = [hist.copy(), hist.copy()]
    val = env._compute_traj_cosine_mean()
    assert val == pytest.approx(0.0, abs=1e-5)


def test_c4_different_episodes_nonzero_distance():
    """Two episodes with different action sequences give mean dist > 0."""
    env = CausalGridWorldV2(
        size=14, seed=1, num_hazards=0, num_resources=0,
    )
    env.reset()
    _run_episode(env, action_seq=[0] * 500)  # always move up
    env.reset()
    infos = _run_episode(env, action_seq=[1] * 500)  # always move down
    val = infos[-1]["traj_pairwise_cosine_mean"]
    assert val > 0.01


def test_c4_metric_in_range():
    """After multiple episodes the metric is in [0, 1]."""
    env = CausalGridWorldV2(
        size=12, seed=9, num_hazards=2, num_resources=2,
    )
    env.reset()
    for _ in range(5):
        _run_episode(env)
        env.reset()
    val = env._traj_pairwise_cosine_mean
    assert 0.0 <= val <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# C5 -- reset() / reset_to() clear _traj_current but not _traj_store
# --------------------------------------------------------------------------- #

def test_c5_reset_clears_traj_current_preserves_store():
    env = CausalGridWorldV2(
        size=12, seed=4, num_hazards=1, num_resources=1,
    )
    env.reset()
    # Complete two episodes to build up a store.
    _run_episode(env)
    env.reset()
    _run_episode(env)
    store_len = len(env._traj_store)
    cached_metric = env._traj_pairwise_cosine_mean
    assert store_len == 2

    # Start a third episode and step a bit.
    env.reset()
    for _ in range(20):
        env.step(0)
    assert len(env._traj_current) > 0

    # reset() clears _traj_current but not the store or the cached metric.
    env.reset()
    assert env._traj_current == []
    assert len(env._traj_store) == store_len
    assert env._traj_pairwise_cosine_mean == pytest.approx(cached_metric, abs=1e-9)


def test_c5_reset_to_clears_traj_current_preserves_store():
    env = CausalGridWorldV2(
        size=12, seed=4, num_hazards=1, num_resources=0,
    )
    env.reset()
    _run_episode(env)
    env.reset()
    _run_episode(env)
    store_len = len(env._traj_store)

    env.reset()
    for _ in range(10):
        env.step(0)
    assert len(env._traj_current) > 0

    env.reset_to(agent_pos=(5, 5), hazard_positions=[(2, 2)])
    assert env._traj_current == []
    assert len(env._traj_store) == store_len
