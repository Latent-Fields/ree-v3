"""
Contract tests for infant_substrate:GAP-2 -- microhabitat zones env feature.

Five contracts:
  C1  Off by default -- legacy CausalGridWorldV2 is bit-identical (same RNG
      sequence -> same agent/hazard/resource layout); sentinel info keys
      always present but inert (_zone_map is None).
  C2  Zone map covers every interior cell with the configured Voronoi zones
      plus the automatic D border zone; codes are well-formed.
  C3  Zone-weighted spawn -- with zone_C_hazard_factor=0 no hazard ever
      spawns in zone C, and zone-B hazard density exceeds zone-A density
      under the default high-B / low-A contrast.
  C4  Zone-C ambient bonus fires only in zone C, equals bonus * decay^visits,
      and is zero in non-C zones.
  C5  reset() clears per-episode zone state and rebuilds / nulls the map
      according to the master switch.
"""
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
# C1 -- Off by default (bit-identical legacy behaviour)
# --------------------------------------------------------------------------- #

def test_c1_sentinel_keys_inert_when_disabled():
    env = CausalGridWorldV2(size=12, seed=7, num_hazards=2)
    env.reset()
    _, _, _, info, _ = env.step(0)
    assert info["microhabitat_enabled"] is False
    assert info["microhabitat_zone_at_agent"] == -1
    assert info["microhabitat_zone_c_ambient_this_tick"] == 0.0
    assert info["microhabitat_zone_counts"] == [0, 0, 0, 0]
    assert env._zone_map is None


def test_c1_bit_identical_off():
    """Default env and explicit microhabitat_enabled=False produce the
    identical layout from the same seed (no extra RNG draws when OFF)."""
    a = CausalGridWorldV2(size=12, seed=42, num_hazards=3, num_resources=4)
    b = CausalGridWorldV2(
        size=12, seed=42, num_hazards=3, num_resources=4,
        microhabitat_enabled=False,
    )
    for _ in range(5):
        assert _layout(a) == _layout(b)


def test_c1_no_ambient_across_episode_when_disabled():
    env = CausalGridWorldV2(size=12, seed=7, num_hazards=2)
    env.reset()
    for _ in range(200):
        _, _, done, info, _ = env.step(0)
        assert info["microhabitat_zone_c_ambient_this_tick"] == 0.0
        assert info["microhabitat_enabled"] is False
        if done:
            env.reset()


# --------------------------------------------------------------------------- #
# C2 -- Zone map coverage and well-formedness
# --------------------------------------------------------------------------- #

def test_c2_zone_map_covers_interior():
    env = CausalGridWorldV2(
        size=14, seed=3, num_hazards=2, microhabitat_enabled=True,
        n_microhabitats=3,
    )
    env.reset()
    zm = env._zone_map
    assert zm is not None
    assert zm.shape == (14, 14)
    # Every non-toroidal interior cell must be assigned (code != -1).
    interior = zm[1:-1, 1:-1]
    assert (interior != -1).all(), "unassigned interior cell"
    codes = set(int(v) for v in np.unique(interior))
    # Three base zones (0,1,2) all present at this grid size + seed.
    assert {0, 1, 2}.issubset(codes), f"missing base zones, got {codes}"
    # Only valid codes appear.
    assert codes.issubset({0, 1, 2, 3}), f"unexpected zone code in {codes}"
    # Border zone D arises where base zones meet.
    assert 3 in codes, "expected at least one D-border cell"


def test_c2_clamp_n_microhabitats():
    env = CausalGridWorldV2(
        size=10, seed=1, microhabitat_enabled=True, n_microhabitats=0,
    )
    assert env.n_microhabitats == 1
    env.reset()
    assert env._zone_map is not None


# --------------------------------------------------------------------------- #
# C3 -- Zone-weighted spawn density
# --------------------------------------------------------------------------- #

def test_c3_zone_c_zero_hazard_factor_excludes_hazards():
    """zone_C_hazard_factor=0.0 (default) -> no hazard ever in zone C, and
    zone-B hazard density > zone-A density under default high-B/low-A."""
    env = CausalGridWorldV2(
        size=14, seed=11, num_hazards=6, num_resources=0,
        microhabitat_enabled=True, n_microhabitats=3,
    )
    haz_by_zone = np.zeros(4, dtype=np.int64)
    cells_by_zone = np.zeros(4, dtype=np.int64)
    n_resets = 40
    for _ in range(n_resets):
        env.reset()
        zm = env._zone_map
        for z in range(4):
            cells_by_zone[z] += int((zm == z).sum())
        for hx, hy in env.hazards:
            z = int(zm[hx, hy])
            if 0 <= z < 4:
                haz_by_zone[z] += 1
    # Hard guarantee: zone C (factor 0.0) never receives a hazard.
    assert haz_by_zone[2] == 0, f"hazard spawned in zone C: {haz_by_zone}"
    # Density = hazards per cell; zone B (1.8) should exceed zone A (0.3).
    dens = haz_by_zone / np.maximum(cells_by_zone, 1)
    assert dens[1] > dens[0], f"zone-B density {dens[1]} !> zone-A {dens[0]}"


def test_c3_resource_bias_toward_zone_a():
    """Default zone_A_resource_factor=1.5 > zone_C=0.3 -> zone-A resource
    density exceeds zone-C density across many resets."""
    env = CausalGridWorldV2(
        size=14, seed=5, num_hazards=0, num_resources=8,
        microhabitat_enabled=True, n_microhabitats=3,
    )
    res_by_zone = np.zeros(4, dtype=np.int64)
    cells_by_zone = np.zeros(4, dtype=np.int64)
    for _ in range(40):
        env.reset()
        zm = env._zone_map
        for z in range(4):
            cells_by_zone[z] += int((zm == z).sum())
        for rx, ry in env.resources:
            z = int(zm[rx, ry])
            if 0 <= z < 4:
                res_by_zone[z] += 1
    dens = res_by_zone / np.maximum(cells_by_zone, 1)
    assert dens[0] > dens[2], f"zone-A res density {dens[0]} !> zone-C {dens[2]}"


# --------------------------------------------------------------------------- #
# C4 -- Zone-C ambient bonus
# --------------------------------------------------------------------------- #

def test_c4_ambient_fires_in_zone_c_and_decays():
    env = CausalGridWorldV2(
        size=10, seed=2, num_hazards=0, num_resources=0,
        use_proxy_fields=False, toroidal=True,
        microhabitat_enabled=True, zone_C_ambient_bonus=0.2,
        zone_novelty_decay=0.5,
    )
    env.reset()
    # Force the entire grid to zone C so every move enters a C cell.
    env._zone_map[:, :] = 2
    env._zone_c_visit_count = 0

    _, r0, _, info0, _ = env.step(3)
    assert info0["transition_type"] == "zone_c_ambient"
    assert abs(info0["microhabitat_zone_c_ambient_this_tick"] - 0.2) < 1e-9
    assert env._zone_c_visit_count == 1

    _, r1, _, info1, _ = env.step(3)
    # Second visit: bonus * decay^1 = 0.2 * 0.5 = 0.1
    assert abs(info1["microhabitat_zone_c_ambient_this_tick"] - 0.1) < 1e-9
    assert env._zone_c_visit_count == 2


def test_c4_no_ambient_outside_zone_c():
    env = CausalGridWorldV2(
        size=10, seed=2, num_hazards=0, num_resources=0,
        use_proxy_fields=False, toroidal=True,
        microhabitat_enabled=True, zone_C_ambient_bonus=0.2,
    )
    env.reset()
    env._zone_map[:, :] = 0  # all zone A
    for _ in range(10):
        _, _, _, info, _ = env.step(3)
        assert info["microhabitat_zone_c_ambient_this_tick"] == 0.0
        assert info["transition_type"] != "zone_c_ambient"
    assert env._zone_c_visit_count == 0


# --------------------------------------------------------------------------- #
# C5 -- reset() clears per-episode zone state
# --------------------------------------------------------------------------- #

def test_c5_reset_clears_zone_state():
    env = CausalGridWorldV2(
        size=10, seed=2, num_hazards=0, num_resources=0,
        use_proxy_fields=False, toroidal=True,
        microhabitat_enabled=True, zone_C_ambient_bonus=0.2,
    )
    env.reset()
    env._zone_map[:, :] = 2
    env._zone_c_visit_count = 0
    env.step(3)
    env.step(3)
    assert env._zone_c_visit_count == 2

    env.reset()
    assert env._zone_c_visit_count == 0
    assert env._zone_c_ambient_this_tick == 0.0
    assert env._zone_map is not None  # rebuilt while enabled


def test_c5_disabled_reset_nulls_map():
    env = CausalGridWorldV2(size=10, seed=2, microhabitat_enabled=False)
    env.reset()
    assert env._zone_map is None


# --------------------------------------------------------------------------- #
# C6 -- Degenerate-seeding redraw guard (GAP-2 V3-EXQ-577 false-negative)
# --------------------------------------------------------------------------- #

# Exact config the V3-EXQ-577 autopsy reproduced the collapse with
# (failure_autopsy_EXQ-577_2026-05-16): seeds 0/1/2 x 100 episodes saw
# missing_012 = 2/2/3 -- one base Voronoi niche fully absorbed into the D
# ecotone in ~2-3% of stochastic episodes. With the redraw guard active
# (default microhabitat_max_seed_redraws=8) the collapse rate must be 0
# and strict per-episode {0,1,2} presence must hold.
_AUTOPSY_SEEDS = (0, 1, 2)
_AUTOPSY_EPISODES = 100


def test_c6_redraw_guard_eliminates_base_zone_collapse():
    """Over the exact autopsy repro (size=14, n=3, seeds 0/1/2, 100 eps
    each) no episode loses a base zone; codes stay well-formed; the guard
    demonstrably engages (>=1 redraw across the 300 episodes -- the same
    first-draws that previously collapsed) and never exhausts its cap."""
    collapses = 0
    total_redraws = 0
    exhausted_any = False
    for seed in _AUTOPSY_SEEDS:
        env = CausalGridWorldV2(
            size=14, seed=seed, num_hazards=6, num_resources=0,
            use_proxy_fields=False, microhabitat_enabled=True,
            n_microhabitats=3,
        )
        for _ in range(_AUTOPSY_EPISODES):
            env.reset()
            zm = env._zone_map
            assert zm is not None
            interior = zm[1:-1, 1:-1]
            codes = set(int(v) for v in np.unique(interior))
            assert -1 not in codes, "unassigned interior cell"
            assert codes.issubset({0, 1, 2, 3}), f"bad zone code in {codes}"
            assert 3 in codes, "expected a D-border cell"
            if not {0, 1, 2}.issubset(codes):
                collapses += 1
            total_redraws += int(env._microhabitat_redraw_count)
            exhausted_any |= bool(env._microhabitat_redraw_exhausted)
    assert collapses == 0, f"base-zone collapse persisted: {collapses}/300"
    assert not exhausted_any, "redraw cap exhausted -- raise the cap"
    assert total_redraws >= 1, (
        "guard never engaged -- the autopsy config had degenerate "
        "first-draws, so the guard must have performed >=1 redraw"
    )


def test_c6_redraw_diagnostics_inert_when_disabled():
    """The two new info keys are always present and inert when the master
    switch is off (no extra RNG -- bit-identical OFF is covered by C1)."""
    env = CausalGridWorldV2(size=12, seed=7, num_hazards=2)
    env.reset()
    _, _, _, info, _ = env.step(0)
    assert info["microhabitat_redraw_count"] == 0
    assert info["microhabitat_redraw_exhausted"] is False


def test_c6_zero_cap_disables_redraw():
    """microhabitat_max_seed_redraws=0 is the escape hatch: the guard never
    redraws (redraw_count stays 0), the first draw is always kept, and a
    still-degenerate draw is surfaced via the exhausted flag rather than
    fixed. The zone map is still built."""
    env = CausalGridWorldV2(
        size=14, seed=0, num_hazards=6, num_resources=0,
        use_proxy_fields=False, microhabitat_enabled=True,
        n_microhabitats=3, microhabitat_max_seed_redraws=0,
    )
    assert env.microhabitat_max_seed_redraws == 0
    for _ in range(_AUTOPSY_EPISODES):
        env.reset()
        assert env._zone_map is not None
        assert env._microhabitat_redraw_count == 0, "cap=0 must not redraw"
