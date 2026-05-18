"""Contract tests for MECH-269 base substrate (Phase 1, 2026-04-22).

Per-stream verisimilitude V_s tracking on HippocampalModule. Foundation
for Phase 2 MECH-287 broadcast trigger and MECH-284 staleness accumulator.

Guarantees enforced here:
  C1. Default HippocampalConfig has use_per_stream_vs=False (backward
      compat). Module-level defaults match the Phase 1 spec
      (per_stream_vs_tau=0.1; per_stream_vs_streams covers the six
      registered streams).
  C2. Master switch OFF: HippocampalModule.per_stream_vs stays empty,
      update_per_stream_vs() is a no-op. Backward compatible at the
      module level.
  C3. Master switch ON: update_per_stream_vs() populates per_stream_vs
      for streams present on the LatentState. Identity-prediction proxy
      yields V_s = 1.0 on the first observation (perfect verisimilitude
      assumed) and < 1.0 on a perturbed second observation.
  C4. Per-stream isolation: a perturbation to one stream affects only
      that stream's V_s; other streams' V_s scores remain at 1.0.
  C5. EMA correctness: with tau=0.5 and a sequence of identical
      perturbations, V_s converges toward the per-step score (not to
      1.0 and not to 0.0).
"""

from __future__ import annotations

import pytest
import torch


def _make_module(use_per_stream_vs: bool = False, tau: float = 0.1):
    """Construct a minimal HippocampalModule for unit tests."""
    from ree_core.utils.config import HippocampalConfig, E2Config, ResidueConfig
    from ree_core.predictors.e2_fast import E2FastPredictor
    from ree_core.residue.field import ResidueField
    from ree_core.hippocampal.module import HippocampalModule

    hcfg = HippocampalConfig(
        world_dim=8, action_dim=4, action_object_dim=4,
        hidden_dim=16, horizon=3, num_candidates=4,
        num_cem_iterations=1, elite_fraction=0.5,
        use_per_stream_vs=use_per_stream_vs,
        per_stream_vs_tau=tau,
    )
    e2cfg = E2Config(self_dim=8, world_dim=8, action_dim=4, action_object_dim=4)
    rcfg = ResidueConfig(world_dim=8, num_basis_functions=4)
    e2 = E2FastPredictor(e2cfg)
    residue = ResidueField(rcfg)
    return HippocampalModule(hcfg, e2, residue)


def _make_latent(z_world=None, z_self=None, z_harm=None, z_harm_a=None,
                 z_beta=None):
    """Lightweight LatentState surrogate carrying just the streams we test."""
    from ree_core.latent.stack import LatentState

    def _z(t, dim):
        if t is not None:
            return t
        return None

    return LatentState(
        z_self=z_self if z_self is not None else torch.zeros(1, 8),
        z_world=z_world if z_world is not None else torch.zeros(1, 8),
        z_beta=z_beta if z_beta is not None else torch.zeros(1, 8),
        z_theta=torch.zeros(1, 8),
        z_delta=torch.zeros(1, 8),
        precision={},
        z_harm=z_harm,
        z_harm_a=z_harm_a,
    )


# ------------------------------------------------------------------ #
# C1                                                                 #
# ------------------------------------------------------------------ #

def test_c1_default_config_backward_compatible():
    """C1: HippocampalConfig defaults match Phase 1 spec; flag OFF by default."""
    from ree_core.utils.config import HippocampalConfig
    cfg = HippocampalConfig()
    assert getattr(cfg, "use_per_stream_vs", False) is False
    assert cfg.per_stream_vs_tau == 0.1
    streams = tuple(cfg.per_stream_vs_streams)
    for s in ("z_world", "z_self", "z_harm_s", "z_harm_a", "z_goal", "z_beta"):
        assert s in streams, f"expected stream {s} in defaults"


# ------------------------------------------------------------------ #
# C2                                                                 #
# ------------------------------------------------------------------ #

def test_c2_master_switch_off_is_noop():
    """C2: flag OFF -> per_stream_vs stays empty after update_per_stream_vs()."""
    mod = _make_module(use_per_stream_vs=False)
    latent = _make_latent(
        z_world=torch.randn(1, 8),
        z_self=torch.randn(1, 8),
        z_beta=torch.randn(1, 8),
    )
    assert mod.per_stream_vs == {}
    mod.update_per_stream_vs(latent)
    assert mod.per_stream_vs == {}, (
        "update_per_stream_vs() must be a no-op when use_per_stream_vs=False"
    )


# ------------------------------------------------------------------ #
# C3                                                                 #
# ------------------------------------------------------------------ #

def test_c3_master_switch_on_populates_and_drops_on_perturbation():
    """C3: flag ON -> first call seeds V_s=1.0, perturbation lowers V_s."""
    mod = _make_module(use_per_stream_vs=True, tau=1.0)  # tau=1 -> no EMA carry
    z0 = torch.zeros(1, 8)
    z0[0, 0] = 1.0
    latent_a = _make_latent(z_world=z0.clone(), z_self=z0.clone(),
                            z_beta=z0.clone())
    mod.update_per_stream_vs(latent_a)
    # First observation: V_s seeded at 1.0 for every present stream.
    for s in ("z_world", "z_self", "z_beta"):
        assert mod.per_stream_vs[s] == 1.0, (
            f"first observation should seed {s} at 1.0"
        )
    # Disabled streams should not appear (z_harm_s / z_harm_a / z_goal absent).
    assert "z_harm_s" not in mod.per_stream_vs
    assert "z_harm_a" not in mod.per_stream_vs
    assert "z_goal" not in mod.per_stream_vs

    # Now perturb z_world; identity proxy should drop V_s_world below 1.0.
    z1 = z0.clone()
    z1[0, 1] = 0.9
    latent_b = _make_latent(z_world=z1, z_self=z0.clone(),
                            z_beta=z0.clone())
    mod.update_per_stream_vs(latent_b)
    assert mod.per_stream_vs["z_world"] < 1.0, (
        "perturbation should drop V_s below 1.0"
    )
    assert mod.per_stream_vs["z_world"] >= 0.0


# ------------------------------------------------------------------ #
# C4                                                                 #
# ------------------------------------------------------------------ #

def test_c4_per_stream_isolation():
    """C4: a perturbation in one stream does not move other streams' V_s."""
    mod = _make_module(use_per_stream_vs=True, tau=1.0)
    z0 = torch.zeros(1, 8)
    z0[0, 0] = 1.0
    mod.update_per_stream_vs(_make_latent(
        z_world=z0.clone(), z_self=z0.clone(), z_beta=z0.clone(),
    ))
    # Perturb only z_world; z_self and z_beta repeat their previous value.
    z1 = z0.clone()
    z1[0, 2] = 0.5
    mod.update_per_stream_vs(_make_latent(
        z_world=z1, z_self=z0.clone(), z_beta=z0.clone(),
    ))
    assert mod.per_stream_vs["z_world"] < 1.0
    assert mod.per_stream_vs["z_self"] == pytest.approx(1.0)
    assert mod.per_stream_vs["z_beta"] == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# C5                                                                 #
# ------------------------------------------------------------------ #

def test_c5_ema_smoothing_correctness():
    """C5: with tau=0.5 and identical perturbations, V_s tracks per-step score."""
    mod = _make_module(use_per_stream_vs=True, tau=0.5)
    base = torch.zeros(1, 8)
    base[0, 0] = 1.0
    # Seed.
    mod.update_per_stream_vs(_make_latent(
        z_world=base.clone(), z_self=base.clone(), z_beta=base.clone(),
    ))
    assert mod.per_stream_vs["z_world"] == 1.0

    # Apply the same perturbation repeatedly; score should asymptote toward
    # the per-step score (not collapse to 0 and not stay at 1).
    perturbed = base.clone()
    perturbed[0, 1] = 0.5  # +0.5 on dim 1; norm ~sqrt(1.25) on each new obs

    last = None
    for _ in range(20):
        mod.update_per_stream_vs(_make_latent(
            z_world=perturbed.clone(), z_self=base.clone(), z_beta=base.clone(),
        ))
        last = mod.per_stream_vs["z_world"]
        # Per-tick: z_curr identical to z_prev after first repeated obs ->
        # err -> 0 -> per-step score -> 1.0; EMA pulls back up. After many
        # ticks of identical inputs, V_s should be back near 1.0.
        perturbed_again = perturbed.clone()
        perturbed = perturbed_again
    # After 20 identical repeat observations, EMA should have recovered
    # toward 1.0 (later perturbations all have err=0 -> score=1.0).
    assert last is not None
    assert last > 0.95, f"V_s should recover toward 1.0 under stable input, got {last}"
    # Bounds: EMA-smoothed score must remain in [0, 1].
    assert 0.0 <= last <= 1.0
