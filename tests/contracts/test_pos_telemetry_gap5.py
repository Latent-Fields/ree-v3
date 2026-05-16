"""
Contract tests for infant_substrate:GAP-5 -- H_pos / zone_coverage telemetry.

Five contracts:
  C1  Off when explicitly disabled -- sentinel info keys present but inert
      (pos_entropy=-1.0, zone_coverage={}); default-ON vs explicit-OFF
      produce a bit-identical layout (telemetry has no RNG and never feeds
      back into env/agent/obs dynamics).
  C2  pos_entropy correctness -- stationary agent yields 0.0; a uniform
      spread over K distinct cells yields ln(K) (nats); the rolling window
      caps the histogram at pos_entropy_window most-recent positions.
  C3  zone_coverage with GAP-2 active -- keys subset of {0,1,2,3}, each
      fraction in [0, 1], a fully-swept single-zone grid yields fraction
      1.0, and coverage is monotone non-decreasing within an episode.
  C4  zone_coverage single-zone stub -- microhabitat disabled + stub on
      yields a single zone 0 in [0, 1] that grows with exploration; stub
      off yields {}.
  C5  reset() and reset_to() clear the rolling window + visited set;
      pos_entropy returns to the empty-window sentinel right after reset.
"""
import math

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
# C1 -- Off when explicitly disabled (sentinels + bit-identical layout)
# --------------------------------------------------------------------------- #

def test_c1_sentinel_keys_inert_when_disabled():
    env = CausalGridWorldV2(
        size=12, seed=7, num_hazards=2, pos_telemetry_enabled=False
    )
    env.reset()
    _, _, _, info, _ = env.step(0)
    assert info["pos_telemetry_enabled"] is False
    assert info["pos_entropy"] == -1.0
    assert info["zone_coverage"] == {}
    # The window length is still echoed even when disabled.
    assert info["pos_entropy_window"] == 100
    assert env._pos_window == []
    assert env._visited_cells == set()


def test_c1_default_on_vs_explicit_off_bit_identical_layout():
    """Telemetry has no RNG and no dynamics feedback, so the default-ON env
    and an explicit pos_telemetry_enabled=False env produce the identical
    layout from the same seed across many resets."""
    a = CausalGridWorldV2(size=12, seed=42, num_hazards=3, num_resources=4)
    b = CausalGridWorldV2(
        size=12, seed=42, num_hazards=3, num_resources=4,
        pos_telemetry_enabled=False,
    )
    for _ in range(5):
        assert _layout(a) == _layout(b)


def test_c1_no_telemetry_across_episode_when_disabled():
    env = CausalGridWorldV2(
        size=12, seed=7, num_hazards=2, pos_telemetry_enabled=False
    )
    env.reset()
    for _ in range(150):
        _, _, done, info, _ = env.step(np.random.randint(0, 4))
        assert info["pos_telemetry_enabled"] is False
        assert info["pos_entropy"] == -1.0
        assert info["zone_coverage"] == {}
        if done:
            env.reset()
    assert env._pos_window == []
    assert env._visited_cells == set()


# --------------------------------------------------------------------------- #
# C2 -- pos_entropy correctness (nats; stationary 0; uniform ln K; window cap)
# --------------------------------------------------------------------------- #

def test_c2_stationary_zero_entropy():
    env = CausalGridWorldV2(size=10, seed=1, num_hazards=0, num_resources=0)
    env.reset()
    env._pos_window = [(5, 5)] * 20
    assert env._pos_entropy() == 0.0


def test_c2_uniform_spread_is_ln_k():
    env = CausalGridWorldV2(size=10, seed=1, num_hazards=0, num_resources=0)
    env.reset()
    for k in (2, 3, 4, 8):
        env._pos_window = [(i, i) for i in range(k)]
        assert abs(env._pos_entropy() - math.log(k)) < 1e-9


def test_c2_window_caps_histogram():
    """Only the pos_entropy_window most-recent positions count. With a
    window of 4, an old run of one cell is evicted by 4 fresh distinct
    cells -> entropy becomes ln(4), not a mixture including the old cell."""
    env = CausalGridWorldV2(
        size=10, seed=1, num_hazards=0, num_resources=0,
        pos_entropy_window=4,
    )
    env.reset()
    env._reset_pos_telemetry()
    # Simulate the step() append+trim contract directly.
    seq = [(0, 0)] * 6 + [(1, 1), (2, 2), (3, 3), (4, 4)]
    for cell in seq:
        env._pos_window.append(cell)
        if len(env._pos_window) > env.pos_entropy_window:
            env._pos_window.pop(0)
    assert len(env._pos_window) == 4
    assert abs(env._pos_entropy() - math.log(4)) < 1e-9


def test_c2_empty_window_sentinel():
    env = CausalGridWorldV2(size=10, seed=1, num_hazards=0, num_resources=0)
    env.reset()
    # Fresh episode, no steps taken yet -> empty window -> sentinel.
    assert env._pos_window == []
    assert env._pos_entropy() == -1.0


# --------------------------------------------------------------------------- #
# C3 -- zone_coverage with GAP-2 active
# --------------------------------------------------------------------------- #

def test_c3_keys_and_range_with_microhabitat():
    env = CausalGridWorldV2(
        size=14, seed=3, num_hazards=2,
        microhabitat_enabled=True, n_microhabitats=3,
    )
    env.reset()
    for _ in range(80):
        env.step(np.random.randint(0, 4))
    _, _, _, info, _ = env.step(3)
    zc = info["zone_coverage"]
    assert set(zc.keys()).issubset({0, 1, 2, 3})
    assert len(zc) >= 1
    assert all(0.0 <= v <= 1.0 for v in zc.values())


def test_c3_full_sweep_is_one():
    env = CausalGridWorldV2(
        size=10, seed=2, num_hazards=0, num_resources=0,
        use_proxy_fields=False, toroidal=True,
        microhabitat_enabled=True,
    )
    env.reset()
    # Force the whole grid into zone C, then mark every cell visited.
    env._zone_map[:, :] = 2
    env._visited_cells = {
        (x, y) for x in range(env.size) for y in range(env.size)
    }
    zc = env._zone_coverage()
    assert set(zc.keys()) == {2}  # only zone with a non-zero denominator
    assert zc[2] == 1.0


def test_c3_monotone_non_decreasing():
    env = CausalGridWorldV2(
        size=14, seed=5, num_hazards=1,
        microhabitat_enabled=True, n_microhabitats=3,
    )
    env.reset()
    for _ in range(30):
        env.step(np.random.randint(0, 4))
    early = dict(env._zone_coverage())
    for _ in range(80):
        env.step(np.random.randint(0, 4))
    late = env._zone_coverage()
    # Visited set only grows and denominators are constant within an
    # episode, so every previously-seen zone fraction is non-decreasing.
    for z, v in early.items():
        assert late.get(z, 0.0) >= v - 1e-12


# --------------------------------------------------------------------------- #
# C4 -- zone_coverage single-zone stub when microhabitat disabled
# --------------------------------------------------------------------------- #

def test_c4_stub_single_zone_grows():
    env = CausalGridWorldV2(size=12, seed=11, num_hazards=1, num_resources=1)
    env.reset()
    assert env._zone_map is None
    for _ in range(20):
        env.step(np.random.randint(0, 4))
    cov_early = dict(env._zone_coverage())
    assert set(cov_early.keys()) == {0}
    assert 0.0 <= cov_early[0] <= 1.0
    for _ in range(120):
        env.step(np.random.randint(0, 4))
    cov_late = env._zone_coverage()
    assert set(cov_late.keys()) == {0}
    assert cov_late[0] >= cov_early[0] - 1e-12
    assert 0.0 <= cov_late[0] <= 1.0


def test_c4_stub_off_returns_empty():
    env = CausalGridWorldV2(
        size=12, seed=11, num_hazards=1,
        zone_coverage_stub_single_zone=False,
    )
    env.reset()
    assert env._zone_map is None
    for _ in range(10):
        _, _, _, info, _ = env.step(np.random.randint(0, 4))
    assert info["zone_coverage"] == {}


# --------------------------------------------------------------------------- #
# C5 -- reset() and reset_to() clear the window + visited set
# --------------------------------------------------------------------------- #

def test_c5_reset_clears_telemetry():
    env = CausalGridWorldV2(size=12, seed=2, num_hazards=1, num_resources=1)
    env.reset()
    for _ in range(40):
        env.step(np.random.randint(0, 4))
    assert len(env._pos_window) > 0
    assert len(env._visited_cells) > 0

    env.reset()
    assert env._pos_window == []
    assert env._visited_cells == set()
    # Empty window right after reset -> sentinel entropy.
    assert env._pos_entropy() == -1.0


def test_c5_reset_to_clears_telemetry():
    env = CausalGridWorldV2(size=12, seed=2, num_hazards=1, num_resources=0)
    env.reset()
    for _ in range(30):
        env.step(np.random.randint(0, 4))
    assert len(env._pos_window) > 0

    env.reset_to(agent_pos=(5, 5), hazard_positions=[(2, 2)])
    assert env._pos_window == []
    assert env._visited_cells == set()
    assert env._pos_entropy() == -1.0
