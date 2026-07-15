"""
Contract tests for SD-067 dedicated safety-terrain RBF bandwidth (MECH-303
behavioural promote-to-active gate, 2026-07-15).

The safety terrain read (evaluate_safety) is an RBF over accumulated z_world
centers. The shared kernel_bandwidth (1.0) is ~15x too wide for the z_world
residual scale (~0.065 between safe/unsafe contexts under SD-008 under-
differentiation), so the RBF saturates: it reads ~identically in every context
and the fixed contextual_safety_release_threshold cannot resolve the safe-vs-
unsafe signal into an absolute gap (the discrimination exists as a rank -- AUC
~0.83, V3-EXQ-760 -- but the behavioural release fires indiscriminately). A
dedicated tighter safety_terrain_bandwidth resolves it into an absolute gap.
Centering (a translation) is a no-op here because the RBF is Euclidean-distance-
based (translation-invariant), unlike SD-066's cosine store -- so bandwidth, not
centering, is the correct lever.

Four contracts:
  C1  Off by default -- safety_terrain_bandwidth None -> the safety RBF uses
      kernel_bandwidth (bit-identical to the pre-SD-067 behaviour).
  C2  When set, the safety RBF uses the dedicated (tighter) bandwidth; the
      benefit/harm RBFs are untouched (still kernel_bandwidth).
  C3  Resolution: with a safe cluster accumulated and a nearby off-cluster query,
      the WIDE (kernel) bandwidth reads ~identically at both (gap tiny) while the
      TIGHT bandwidth separates them into an absolute gap.
  C4  Config default + from_dims wiring: residue.safety_terrain_bandwidth
      defaults None; from_dims threads the value only when the terrain is on.
"""

import torch

from ree_core.utils.config import ResidueConfig, REEConfig
from ree_core.residue.field import ResidueField


def _residue_cfg(world_dim=8, bandwidth=1.0, safety_bw=None):
    cfg = ResidueConfig()
    cfg.world_dim = world_dim
    cfg.num_basis_functions = 32
    cfg.kernel_bandwidth = bandwidth
    cfg.safety_terrain_enabled = True
    cfg.safety_terrain_bandwidth = safety_bw
    return cfg


# ---------------------------------------------------------------------------
# C1  Off by default -- safety RBF falls back to kernel_bandwidth
# ---------------------------------------------------------------------------

def test_c1_none_falls_back_to_kernel_bandwidth():
    rf = ResidueField(_residue_cfg(bandwidth=1.0, safety_bw=None))
    assert rf.safety_terrain_rbf_field.bandwidth == 1.0


def test_c1_config_default_is_none():
    assert ResidueConfig().safety_terrain_bandwidth is None


# ---------------------------------------------------------------------------
# C2  When set, the safety RBF uses the dedicated bandwidth (others untouched)
# ---------------------------------------------------------------------------

def test_c2_dedicated_bandwidth_used():
    rf = ResidueField(_residue_cfg(bandwidth=1.0, safety_bw=0.03))
    assert rf.safety_terrain_rbf_field.bandwidth == 0.03
    # the general residue RBF (harm terrain) still uses kernel_bandwidth
    assert rf.rbf_field.bandwidth == 1.0


# ---------------------------------------------------------------------------
# C3  Tight bandwidth resolves a small z_world separation the wide one cannot
# ---------------------------------------------------------------------------

def test_c3_tight_bandwidth_resolves_small_separation():
    world_dim = 8
    torch.manual_seed(0)
    # A "safe" cluster near a base point; an "unsafe" query a small residual away.
    base = torch.zeros(world_dim)
    base[0] = 0.4
    safe_points = [base + 0.01 * torch.randn(world_dim) for _ in range(20)]
    unsafe_query = base.clone()
    unsafe_query[1] += 0.065   # the ~0.065 residual measured between safe/unsafe z_world
    safe_query = base + 0.01 * torch.randn(world_dim)

    def read(safety_bw):
        rf = ResidueField(_residue_cfg(world_dim=world_dim, bandwidth=1.0, safety_bw=safety_bw))
        for p in safe_points:
            rf.accumulate_safety(p, safety_magnitude=0.05)
        s = float(rf.evaluate_safety(safe_query.unsqueeze(0)).mean().detach())
        u = float(rf.evaluate_safety(unsafe_query.unsqueeze(0)).mean().detach())
        return s, u

    wide_s, wide_u = read(None)     # kernel_bandwidth = 1.0 -> saturated
    tight_s, tight_u = read(0.03)   # dedicated -> resolves

    # Wide: reads ~identically at safe and unsafe (ratio near 1 -> indiscriminate).
    assert wide_u / (wide_s + 1e-9) > 0.9
    # Tight: safe reads clearly higher than unsafe (absolute gap the release gate can use).
    assert tight_s > tight_u
    assert tight_u / (tight_s + 1e-9) < 0.7


# ---------------------------------------------------------------------------
# C4  Config default + from_dims wiring
# ---------------------------------------------------------------------------

def test_c4_from_dims_default_none():
    cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=5,
                              use_contextual_safety_terrain=True)
    assert cfg.residue.safety_terrain_bandwidth is None


def test_c4_from_dims_threads_value():
    cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=5,
                              use_contextual_safety_terrain=True,
                              safety_terrain_bandwidth=0.03)
    assert cfg.residue.safety_terrain_bandwidth == 0.03
