"""
Contract tests for SD-024 DA-modulated RBF center density (MECH-232, 2026-07-16).

SD-024 is the DIAGNOSTIC substrate that resolves MECH-232: dopamine at reward-
associated locations produces representational EXPANSION (more RBF centers, finer
per-center bandwidth) in the hippocampal benefit terrain -- rather than writing an
explicit positive-valence gradient. The load-bearing property (MECH-232's
discriminator against a valence-tag mechanism) is that the resulting
compute_local_density read is WEIGHT-INDEPENDENT: DA raises representational
density even when the summed benefit value (evaluate) is held flat, so approach
that follows density demonstrates approach from representational quality alone.

Modulation touches ONLY the benefit_rbf_field -- the harm/safety fields keep
single-center standard-bandwidth allocation, preserving the MECH-233 asymmetry.

Contracts:
  C1  Off by default -- use_da_modulated_rbf_density False -> benefit field built
      with num_basis_functions centers, no per-center bandwidth buffer, single-
      center accumulation (dopamine_signal ignored). Bit-identical OFF.
  C2  DA cluster allocation -- with the master switch on, a reward event with a
      positive dopamine_signal allocates n = 1 + int(da * allocation_scale)
      jittered centers; da_benefit_num_centers overrides only the benefit field.
  C3  Density elevation + WEIGHT-INDEPENDENCE (the MECH-232 discriminator): local
      density is higher at the reward cluster than far away, and is unchanged when
      the benefit weights are zeroed (while evaluate_benefit collapses to ~0).
  C4  Per-center bandwidth narrowing -- DA narrows the cluster's per-center
      bandwidth toward 0.5 * base (floored), giving finer place-field resolution.
  C5  MECH-094 gate + asymmetry -- hypothesis_tag blocks DA expansion; the harm
      field is never DA-modulated (no per-center bandwidth, single-center harm).
  C6  Hippocampal read-through -- HippocampalModule.compute_representational_density
      delegates to the benefit-terrain density read.
"""

import torch

from ree_core.utils.config import ResidueConfig
from ree_core.residue.field import ResidueField, RBFLayer


def _cfg(da=False, scale=0.0, jitter=0.1, narrow=0.0, benefit_centers=None,
         world_dim=8, base_centers=16, bandwidth=1.0):
    cfg = ResidueConfig()
    cfg.world_dim = world_dim
    cfg.num_basis_functions = base_centers
    cfg.kernel_bandwidth = bandwidth
    cfg.benefit_terrain_enabled = True
    cfg.use_da_modulated_rbf_density = da
    cfg.da_allocation_scale = scale
    cfg.da_jitter_radius = jitter
    cfg.da_bandwidth_narrowing = narrow
    cfg.da_benefit_num_centers = benefit_centers
    return cfg


def _reward_loc(world_dim=8, coord=2.0):
    z = torch.zeros(1, world_dim)
    z[0, 0] = coord
    return z


# ---------------------------------------------------------------------------
# C1: off by default -- bit-identical OFF
# ---------------------------------------------------------------------------

def test_c1_off_by_default_bit_identical():
    # RBFLayer default registers no per-center bandwidth buffer.
    r = RBFLayer(world_dim=8, num_centers=16)
    assert r.per_center_bandwidth is False
    assert "center_bandwidths" not in dict(r.named_buffers())

    rf = ResidueField(_cfg(da=False))
    assert rf.benefit_rbf_field.num_centers == 16
    assert rf.benefit_rbf_field.per_center_bandwidth is False
    assert not any("center_bandwidths" in k for k in rf.state_dict().keys())

    # A supplied dopamine_signal is ignored when the master switch is off.
    z = _reward_loc()
    rf.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=5.0)
    assert int(rf.benefit_rbf_field.active_mask.sum()) == 1


def test_c1_off_forward_unchanged():
    # The default forward() read is byte-identical to the scalar-bandwidth path.
    torch.manual_seed(7)
    a = RBFLayer(8, 16)
    a.add_residue(torch.randn(8), 1.0)
    torch.manual_seed(7)
    b = RBFLayer(8, 16)
    b.add_residue(torch.randn(8), 1.0)
    q = torch.randn(4, 8)
    assert torch.equal(a(q), b(q))


# ---------------------------------------------------------------------------
# C2: DA cluster allocation
# ---------------------------------------------------------------------------

def test_c2_da_cluster_allocation_count():
    rf = ResidueField(_cfg(da=True, scale=2.0, benefit_centers=64))
    assert rf.benefit_rbf_field.num_centers == 64          # override applied
    assert rf.benefit_rbf_field.per_center_bandwidth is True
    z = _reward_loc()
    # DA = 1.0, scale = 2.0 -> n = 1 + int(2) = 3
    rf.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=1.0)
    assert int(rf.benefit_rbf_field.active_mask.sum()) == 3


def test_c2_scale_zero_is_single_center():
    # Master on but allocation_scale 0 -> single center (no expansion).
    rf = ResidueField(_cfg(da=True, scale=0.0))
    z = _reward_loc()
    rf.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=3.0)
    assert int(rf.benefit_rbf_field.active_mask.sum()) == 1


def test_c2_da_benefit_num_centers_none_uses_base():
    rf = ResidueField(_cfg(da=True, scale=2.0, benefit_centers=None, base_centers=16))
    assert rf.benefit_rbf_field.num_centers == 16


# ---------------------------------------------------------------------------
# C3: density elevation + weight-independence (MECH-232 discriminator)
# ---------------------------------------------------------------------------

def test_c3_density_higher_at_reward_cluster():
    torch.manual_seed(0)
    rf = ResidueField(_cfg(da=True, scale=3.0, benefit_centers=64))
    z = _reward_loc()
    rf.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=1.0)
    far = torch.zeros(1, 8)
    far[0, 0] = -5.0
    assert rf.compute_benefit_density(z).item() > rf.compute_benefit_density(far).item()


def test_c3_density_is_weight_independent():
    # The load-bearing MECH-232 property: density depends on center STRUCTURE,
    # not on benefit WEIGHTS. Zeroing weights leaves density unchanged while the
    # weighted benefit value collapses -- so a density-following approach reads a
    # representational-quality signal, not a positive-valence gradient.
    torch.manual_seed(0)
    rf = ResidueField(_cfg(da=True, scale=3.0, benefit_centers=64))
    z = _reward_loc()
    rf.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=1.0)
    dens_pre = rf.compute_benefit_density(z).item()
    with torch.no_grad():
        rf.benefit_rbf_field.weights.zero_()
    dens_post = rf.compute_benefit_density(z).item()
    assert abs(dens_pre - dens_post) < 1e-9
    assert abs(rf.evaluate_benefit(z).item()) < 1e-6


def test_c3_da_on_denser_than_da_off_same_events():
    # Same reward events, DA-ON vs DA-OFF: ON accumulates more representational
    # density at the reward location (the leg-1 expansion measurement).
    torch.manual_seed(1)
    rf_off = ResidueField(_cfg(da=False, benefit_centers=None))
    torch.manual_seed(1)
    rf_on = ResidueField(_cfg(da=True, scale=3.0, benefit_centers=64))
    z = _reward_loc()
    for _ in range(4):
        rf_off.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=1.0)
        rf_on.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=1.0)
    assert rf_on.compute_benefit_density(z).item() > rf_off.compute_benefit_density(z).item()


# ---------------------------------------------------------------------------
# C4: per-center bandwidth narrowing (finer resolution)
# ---------------------------------------------------------------------------

def test_c4_bandwidth_narrowing_floored():
    torch.manual_seed(0)
    rf = ResidueField(_cfg(da=True, scale=2.0, narrow=0.5, benefit_centers=64, bandwidth=1.0))
    z = _reward_loc()
    rf.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=1.0)
    active = rf.benefit_rbf_field.active_mask
    bws = rf.benefit_rbf_field.center_bandwidths[active]
    assert float(bws.min()) < 1.0                # narrowed below base
    assert float(bws.min()) >= 0.5 - 1e-6        # floored at 0.5 * base


def test_c4_no_narrowing_keeps_base():
    torch.manual_seed(0)
    rf = ResidueField(_cfg(da=True, scale=2.0, narrow=0.0, benefit_centers=64, bandwidth=1.0))
    z = _reward_loc()
    rf.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=1.0)
    active = rf.benefit_rbf_field.active_mask
    bws = rf.benefit_rbf_field.center_bandwidths[active]
    assert torch.allclose(bws, torch.ones_like(bws))


# ---------------------------------------------------------------------------
# C5: MECH-094 gate + MECH-233 asymmetry (harm never modulated)
# ---------------------------------------------------------------------------

def test_c5_hypothesis_tag_blocks_da_expansion():
    rf = ResidueField(_cfg(da=True, scale=2.0, benefit_centers=64))
    z = _reward_loc()
    rf.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=1.0, hypothesis_tag=True)
    assert int(rf.benefit_rbf_field.active_mask.sum()) == 0


def test_c5_harm_field_never_da_modulated():
    rf = ResidueField(_cfg(da=True, scale=2.0, benefit_centers=64))
    # The harm residue field keeps single-center, scalar-bandwidth allocation.
    assert rf.rbf_field.per_center_bandwidth is False
    z = _reward_loc()
    rf.accumulate(z, harm_magnitude=1.0)
    assert int(rf.rbf_field.active_mask.sum()) == 1


# ---------------------------------------------------------------------------
# C6: hippocampal read-through
# ---------------------------------------------------------------------------

def test_c6_hippocampal_density_delegate():
    from ree_core.utils.config import E2Config, HippocampalConfig
    from ree_core.predictors.e2_fast import E2FastPredictor
    from ree_core.hippocampal.module import HippocampalModule

    world_dim = 8
    e2 = E2FastPredictor(E2Config(
        self_dim=8, world_dim=world_dim, action_dim=4,
        action_object_dim=16, rollout_horizon=5, num_candidates=4,
    ))
    hcfg = HippocampalConfig(
        world_dim=world_dim, action_dim=4, action_object_dim=16,
        hidden_dim=64, horizon=5, num_candidates=4, num_cem_iterations=1,
    )
    residue = ResidueField(_cfg(da=True, scale=3.0, benefit_centers=64, world_dim=world_dim))
    hippo = HippocampalModule(hcfg, e2, residue)

    z = torch.zeros(1, world_dim)
    z[0, 0] = 2.0
    residue.accumulate_benefit(z, benefit_magnitude=1.0, dopamine_signal=1.0)
    d_hippo = hippo.compute_representational_density(z)
    d_field = residue.compute_benefit_density(z)
    assert torch.allclose(d_hippo, d_field)
    assert float(d_hippo.item()) > 0.0
