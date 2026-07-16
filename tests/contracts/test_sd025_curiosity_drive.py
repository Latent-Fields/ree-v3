"""
Contract tests for SD-025 curiosity drive (ARC-057 component 2, 2026-07-16).

SD-025 adds an information-seeking bias to hippocampal CEM trajectory scoring that
favours regions of higher REPRESENTATIONAL DENSITY in the SD-024 benefit RBF map:

    novelty(z) = density(z) * (1 - familiarity(z))
    score     -= curiosity_weight * mean_over_trajectory(novelty)

density = HippocampalModule.compute_representational_density (the SD-024 weight-
INDEPENDENT active-center count -> the drive follows representational QUALITY, not
a positive-valence gradient). familiarity = a visit-count EMA (FamiliarityTracker)
that rises on revisit so novelty decays there (anti-perseveration).

SCOPE (drive mechanism, NOT the full ARC-057 ecological claim). The CausalGridWorld
cannot faithfully test ARC-057 approach-emergence (a cell is a cell; there is
nothing more to discover at higher resolution -- see claims.yaml ARC-057 SUBSTRATE
CONSTRAINT). These contracts validate the DRIVE: that the curiosity signal
propagates into CEM selection (the thing MECH-111's broken broadcast-novelty->E3
path could not do -- EXQ-141b/590a), reads density weight-independently, and is
discounted by familiarity.

Contracts:
  C1  Off by default -- curiosity_weight=0.0 -> no FamiliarityTracker, _score_trajectory
      bit-identical to the residue-only terrain score, update_familiarity a no-op.
  C2  FamiliarityTracker mechanics -- empty=0; rises toward 1 on revisit; low far
      away; bounded [0,1].
  C3  Curiosity PROPAGATES into scoring -- with curiosity on, a trajectory through a
      high-density region scores strictly lower (better) than its curiosity-off
      terrain score; the bonus is exactly curiosity_weight * mean(novelty).
  C4  Density-selectivity + weight-independence -- the bonus is larger through the
      dense cluster than through a sparse region, and is UNCHANGED when benefit
      weights are zeroed (inherits the SD-024 MECH-232 discriminator).
  C5  Familiarity discount -- repeated WAKING visits to the dense region raise
      familiarity there and shrink the curiosity bonus (anti-perseveration);
      use_curiosity_familiarity=False removes the discount (novelty = density).
  C6  MECH-094 gate -- update_familiarity(is_waking=False) does not change
      familiarity; the density read during scoring writes no memory.
"""

import torch

from ree_core.utils.config import (
    HippocampalConfig,
    ResidueConfig,
    E2Config,
)
from ree_core.residue.field import ResidueField
from ree_core.predictors.e2_fast import E2FastPredictor, Trajectory
from ree_core.hippocampal.module import HippocampalModule
from ree_core.hippocampal.curiosity import FamiliarityTracker

WORLD_DIM = 8
ACTION_DIM = 4
ACTION_OBJECT_DIM = 16


def _residue(da=True, scale=6.0, base_centers=16, bandwidth=1.0):
    cfg = ResidueConfig(world_dim=WORLD_DIM)
    cfg.num_basis_functions = base_centers
    cfg.kernel_bandwidth = bandwidth
    cfg.benefit_terrain_enabled = True
    cfg.use_da_modulated_rbf_density = da
    cfg.da_allocation_scale = scale
    cfg.da_jitter_radius = 0.1
    return ResidueField(cfg)


def _hip(residue, curiosity_weight=0.0, ema_alpha=0.2, use_familiarity=True):
    cfg = HippocampalConfig(
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM,
        hidden_dim=64,
        horizon=5,
        num_candidates=4,
        num_cem_iterations=1,
        curiosity_weight=curiosity_weight,
        familiarity_ema_alpha=ema_alpha,
        use_curiosity_familiarity=use_familiarity,
        familiarity_bandwidth=1.0,
    )
    e2 = E2FastPredictor(E2Config(
        self_dim=6, world_dim=WORLD_DIM, action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM, rollout_horizon=5, num_candidates=4,
    ))
    return HippocampalModule(cfg, e2, residue)


def _loc(coord, dim0=0):
    z = torch.zeros(1, WORLD_DIM)
    z[0, dim0] = coord
    return z


def _populate_dense(residue, at, n_events=3):
    """Allocate a dense DA cluster of benefit centers near `at`."""
    for _ in range(n_events):
        residue.accumulate_benefit(at, torch.tensor([1.0]), dopamine_signal=1.0)


def _traj_through(coord, horizon=5, dim0=0):
    """A trajectory whose world states all sit at coord along dim0."""
    world_states = [_loc(coord, dim0) for _ in range(horizon + 1)]
    states = [torch.zeros(1, 6) for _ in range(horizon + 1)]
    actions = torch.zeros(1, horizon, ACTION_DIM)
    return Trajectory(states=states, actions=actions, world_states=world_states)


# ---------------------------------------------------------------------------
# C1: off by default -- bit-identical
# ---------------------------------------------------------------------------

def test_c1_off_by_default_bit_identical():
    residue = _residue(da=True)
    _populate_dense(residue, _loc(2.0))
    hip = _hip(residue, curiosity_weight=0.0)
    assert hip.familiarity_tracker is None

    traj = _traj_through(2.0)
    world_seq = traj.get_world_state_sequence()
    terrain_only = residue.evaluate_trajectory(world_seq).sum()
    score = hip._score_trajectory(traj)
    # No curiosity path taken -> exactly the residue terrain score.
    assert torch.equal(score, terrain_only)
    # update_familiarity is a no-op (does not raise, changes nothing).
    assert hip.update_familiarity(_loc(2.0)) is None


# ---------------------------------------------------------------------------
# C2: FamiliarityTracker mechanics
# ---------------------------------------------------------------------------

def test_c2_familiarity_tracker_mechanics():
    ft = FamiliarityTracker(world_dim=WORLD_DIM, num_anchors=32,
                            ema_alpha=0.2, bandwidth=1.0)
    z = _loc(0.0)
    assert float(ft.query(z)[0]) == 0.0            # empty -> everything novel
    for _ in range(30):
        ft.update(z)
    f_near = float(ft.query(z)[0])
    f_far = float(ft.query(_loc(6.0))[0])
    assert f_near > 0.5                            # revisited -> familiar
    assert f_far < 0.05                            # untouched region -> novel
    assert 0.0 <= f_near <= 1.0 and 0.0 <= f_far <= 1.0


# ---------------------------------------------------------------------------
# C3: curiosity propagates into CEM scoring
# ---------------------------------------------------------------------------

def test_c3_curiosity_propagates_into_scoring():
    residue = _residue(da=True)
    _populate_dense(residue, _loc(2.0))
    cw = 0.5
    hip = _hip(residue, curiosity_weight=cw)

    traj = _traj_through(2.0)
    world_seq = traj.get_world_state_sequence()
    terrain_only = residue.evaluate_trajectory(world_seq).sum()
    score_on = hip._score_trajectory(traj)
    bonus = hip._curiosity_bonus(world_seq)

    assert float(bonus) > 0.0                      # dense region -> positive novelty
    # Selection actually changes: score is lowered (CEM minimises) by exactly the bonus.
    assert torch.allclose(score_on, terrain_only - bonus, atol=1e-6)
    assert float(score_on) < float(terrain_only)


# ---------------------------------------------------------------------------
# C4: density-selectivity + weight-independence (inherits SD-024 discriminator)
# ---------------------------------------------------------------------------

def test_c4_density_selective_and_weight_independent():
    residue = _residue(da=True)
    _populate_dense(residue, _loc(2.0))            # dense cluster at coord 2.0
    hip = _hip(residue, curiosity_weight=0.5)

    dense_bonus = float(hip._curiosity_bonus(_traj_through(2.0).get_world_state_sequence()))
    sparse_bonus = float(hip._curiosity_bonus(_traj_through(9.0).get_world_state_sequence()))
    assert dense_bonus > sparse_bonus              # curiosity favours the denser region

    # Weight-independence: zero the benefit RBF weights -> value collapses but the
    # density-driven curiosity bonus is unchanged (the MECH-232 discriminator).
    with torch.no_grad():
        residue.benefit_rbf_field.weights.zero_()
    dense_bonus_zeroed = float(
        hip._curiosity_bonus(_traj_through(2.0).get_world_state_sequence())
    )
    assert abs(dense_bonus_zeroed - dense_bonus) < 1e-6


# ---------------------------------------------------------------------------
# C5: familiarity discount (anti-perseveration) + ablation
# ---------------------------------------------------------------------------

def test_c5_familiarity_discount_reduces_bonus():
    residue = _residue(da=True)
    _populate_dense(residue, _loc(2.0))
    hip = _hip(residue, curiosity_weight=0.5, ema_alpha=0.3, use_familiarity=True)

    world_seq = _traj_through(2.0).get_world_state_sequence()
    bonus_fresh = float(hip._curiosity_bonus(world_seq))
    # Repeated WAKING visits to the dense region raise familiarity there.
    for _ in range(40):
        hip.update_familiarity(_loc(2.0), is_waking=True)
    bonus_familiar = float(hip._curiosity_bonus(world_seq))
    assert bonus_familiar < bonus_fresh            # novelty decays -> anti-perseveration
    assert bonus_familiar >= 0.0


def test_c5_ablation_no_familiarity_uses_pure_density():
    residue = _residue(da=True)
    _populate_dense(residue, _loc(2.0))
    hip = _hip(residue, curiosity_weight=0.5, ema_alpha=0.3, use_familiarity=False)

    world_seq = _traj_through(2.0).get_world_state_sequence()
    bonus_fresh = float(hip._curiosity_bonus(world_seq))
    for _ in range(40):
        hip.update_familiarity(_loc(2.0), is_waking=True)
    bonus_after = float(hip._curiosity_bonus(world_seq))
    # No familiarity discount -> bonus is pure density, unchanged by visits.
    assert abs(bonus_after - bonus_fresh) < 1e-6


# ---------------------------------------------------------------------------
# C6: MECH-094 gate -- replay/sim visits do not raise familiarity
# ---------------------------------------------------------------------------

def test_c6_mech094_replay_does_not_write_familiarity():
    residue = _residue(da=True)
    _populate_dense(residue, _loc(2.0))
    hip = _hip(residue, curiosity_weight=0.5, ema_alpha=0.3)

    z = _loc(2.0)
    before = float(hip.familiarity_tracker.query(z)[0])
    for _ in range(40):
        hip.update_familiarity(z, is_waking=False)   # replay / simulation
    after = float(hip.familiarity_tracker.query(z)[0])
    assert after == before                           # MECH-094: no real memory write
