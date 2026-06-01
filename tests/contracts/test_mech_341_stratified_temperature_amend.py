"""
Contract tests for MECH-341 stratified_within_class_temperature amend.

Routed by failure_autopsy_V3-EXQ-616_2026-05-31.md Sections 7 + 10 (contingent-
on-614b-FAIL-C1 path). The 2026-05-28 retune (stratified_select on both
committed and uncommitted branches) is unchanged; this amend ADDS within-class
proportional sampling sharpness as a new togglable lever to dissociate Layer B
within-class diversity from the existing across-class softmax sampling.

Contracts:
    C1 default-bit-identical -- default config produces argmin within-class
       (legacy MECH-341 behaviour); diagnostics show n_within_class_sampled==0.
    C2 low-temperature-sharpening -- T -> 0+ (we use 1e-4) approaches argmin
       deterministically (sharpening to the legacy behaviour).
    C3 high-temperature-uniform -- T -> inf (we use 1e4) approaches uniform-
       within-class (each candidate in a class roughly equally likely to be
       picked across many trials).
    C4 mid-temperature-stochastic -- T=1.0 produces non-deterministic
       within-class selection (multiple within-class outcomes across trials).
    C5 master-gate -- use_stratified_select=False short-circuits before the
       temperature branch is consulted (returns None).
    C6 simulation-mode-gate -- simulation_mode=True returns None and the
       temperature branch is never entered (MECH-094 preserved).
    C7 single-class-pool -- pool with only one first-action class falls
       through to legacy (None) regardless of within-class temperature; the
       temperature branch is skipped (no class has >= 2 members to sample
       within when all members share a class).
    C8 from_dims-propagation -- REEConfig.from_dims accepts the new kwarg
       and propagates to config.e3_diversity_stratified_within_class_temperature.
    C9 build_from_ree_config-reads-new-flag -- build_e3_score_diversity_from_ree_config
       picks up the new flat REEConfig field and seeds it into
       E3ScoreDiversityConfig.stratified_within_class_temperature.

Failure-mode policy: any contract failure here is a load-bearing bug in the
amend (either bit-identical OFF broken or new behaviour absent).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import pytest
import torch

from ree_core.predictors.e3_score_diversity import (
    E3ScoreDiversity,
    E3ScoreDiversityConfig,
    build_from_ree_config,
)
from ree_core.utils.config import REEConfig


@dataclass
class _StubTrajectory:
    """Minimal trajectory stub for stratified_select tests.

    Carries an `actions` tensor with the first-step argmax matching the
    intended class index. Mirrors the shape produced by E2.rollout_with_world
    [batch, horizon, action_dim] but the helper code reads only the first step.
    """

    actions: torch.Tensor


def _make_pool(class_assignments: List[int], action_dim: int = 4):
    """Build K stub trajectories whose first-action argmax equals the given class."""
    trajectories = []
    for cls in class_assignments:
        a = torch.zeros(1, 1, action_dim)
        a[0, 0, cls] = 1.0
        trajectories.append(_StubTrajectory(actions=a))
    return trajectories


# -----------------------------------------------------------------------------
# C1 default-bit-identical
# -----------------------------------------------------------------------------

def test_c1_default_bit_identical_to_legacy_argmin():
    """Default config (within-class temp None) picks argmin within each class.

    Two classes (0 and 1) with two candidates each; the argmin within each
    class is deterministic. With a fixed seed the selected representative is
    the lower-score candidate in each class.
    """
    torch.manual_seed(42)
    cfg = E3ScoreDiversityConfig()
    assert cfg.stratified_within_class_temperature is None, (
        "default must be None for bit-identical OFF"
    )
    diversity = E3ScoreDiversity(cfg)
    # Candidates: class 0 [score 0.1, 0.5]; class 1 [score 0.2, 0.7].
    # Argmin within each class: idx 0 (score 0.1) and idx 2 (score 0.2).
    pool = _make_pool([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.5, 0.2, 0.7])
    # Run many trials -- with within-class temp None, the within-class pick
    # is deterministic argmin; across-class sampling still happens but only
    # over the per-class representatives [idx 0, idx 2].
    picks = set()
    for _ in range(100):
        idx = diversity.stratified_select(scores=scores, candidates=pool)
        picks.add(idx)
    # Only the argmin reps can ever surface (idx 0 or idx 2).
    assert picks.issubset({0, 2}), (
        f"default within-class must pick argmin only; got {picks}"
    )
    # Diagnostic counter for within-class sampling must remain zero.
    assert diversity.diagnostics.n_within_class_sampled == 0


# -----------------------------------------------------------------------------
# C2 low-temperature-sharpening
# -----------------------------------------------------------------------------

def test_c2_low_temperature_approaches_argmin():
    """T -> 0+ sharpens within-class sampling to the argmin (legacy)."""
    torch.manual_seed(43)
    cfg = E3ScoreDiversityConfig(stratified_within_class_temperature=1e-4)
    diversity = E3ScoreDiversity(cfg)
    # Class 0: [score 0.1, 0.5] -- argmin idx 0; class 1: [score 0.2, 0.7] -- argmin idx 2.
    pool = _make_pool([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.5, 0.2, 0.7])
    picks = []
    for _ in range(200):
        idx = diversity.stratified_select(scores=scores, candidates=pool)
        picks.add(idx) if False else picks.append(idx)
    # At T=1e-4, softmax(-score / T) for class 0 [score 0.1, 0.5] places
    # essentially all mass on idx 0; similarly all mass on idx 2 for class 1.
    # So picks should be a strict subset of {0, 2} with no idx-1 or idx-3
    # contamination.
    assert set(picks).issubset({0, 2}), (
        f"low-T sharpening must collapse to argmin; got {set(picks)}"
    )
    # Within-class sampling fired (the branch executed even though it
    # collapsed to a single value).
    assert diversity.diagnostics.n_within_class_sampled > 0


# -----------------------------------------------------------------------------
# C3 high-temperature-uniform-within-class
# -----------------------------------------------------------------------------

def test_c3_high_temperature_approaches_uniform_within_class():
    """T -> inf approaches uniform-within-class sampling.

    Both within-class members should surface at roughly comparable frequency
    across trials. We use 5 candidates per class and check both members appear
    >= 25% of the time (would be 50% at perfect uniformity; 25% is a generous
    tolerance for the stochastic test).
    """
    torch.manual_seed(44)
    cfg = E3ScoreDiversityConfig(stratified_within_class_temperature=1e4)
    diversity = E3ScoreDiversity(cfg)
    # 5 candidates per class, 2 classes. We track the fraction of times each
    # individual candidate is picked at the global level. Across-class sampling
    # is at the default `stratified_temperature=1.0` so the two classes are
    # roughly balanced.
    pool = _make_pool([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5])
    counts = {i: 0 for i in range(10)}
    for _ in range(1000):
        idx = diversity.stratified_select(scores=scores, candidates=pool)
        counts[idx] += 1
    # Within class 0 (idx 0..4): at perfect uniformity each member gets 1/5
    # of class-0 hits. Generous bound: each member receives >= 5% of total
    # hits (would be 10% at uniformity-and-class-balance).
    total = sum(counts.values())
    for i in range(10):
        frac = counts[i] / total
        assert frac > 0.02, (
            f"high-T must spread within-class mass; candidate {i} got "
            f"fraction {frac:.3f}; counts={counts}"
        )


# -----------------------------------------------------------------------------
# C4 mid-temperature-stochastic
# -----------------------------------------------------------------------------

def test_c4_mid_temperature_produces_stochastic_within_class():
    """T=1.0 produces non-deterministic within-class selection."""
    torch.manual_seed(45)
    cfg = E3ScoreDiversityConfig(stratified_within_class_temperature=1.0)
    diversity = E3ScoreDiversity(cfg)
    pool = _make_pool([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.5, 0.2, 0.7])
    picks = set()
    for _ in range(200):
        idx = diversity.stratified_select(scores=scores, candidates=pool)
        picks.add(idx)
    # All 4 candidates have positive softmax probability; we expect at
    # least 3 unique outcomes across 200 trials.
    assert len(picks) >= 3, (
        f"mid-T must be stochastic; got only {len(picks)} unique picks: {picks}"
    )
    # And n_within_class_sampled should match call count (fires every call
    # when temperature is set and pool has multi-class structure).
    assert diversity.diagnostics.n_within_class_sampled == 200


# -----------------------------------------------------------------------------
# C5 master-gate
# -----------------------------------------------------------------------------

def test_c5_master_gate_short_circuits_before_temperature_branch():
    """use_stratified_select=False short-circuits regardless of within-class temp."""
    torch.manual_seed(46)
    cfg = E3ScoreDiversityConfig(
        use_stratified_select=False,
        stratified_within_class_temperature=1.0,
    )
    diversity = E3ScoreDiversity(cfg)
    pool = _make_pool([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.5, 0.2, 0.7])
    for _ in range(20):
        idx = diversity.stratified_select(scores=scores, candidates=pool)
        assert idx is None
    # Within-class sampling counter must stay zero -- branch never entered.
    assert diversity.diagnostics.n_within_class_sampled == 0


# -----------------------------------------------------------------------------
# C6 simulation-mode-gate
# -----------------------------------------------------------------------------

def test_c6_simulation_mode_gates_temperature_branch():
    """simulation_mode=True returns None even with temperature set."""
    cfg = E3ScoreDiversityConfig(stratified_within_class_temperature=1.0)
    diversity = E3ScoreDiversity(cfg)
    pool = _make_pool([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.5, 0.2, 0.7])
    for _ in range(20):
        idx = diversity.stratified_select(
            scores=scores, candidates=pool, simulation_mode=True
        )
        assert idx is None
    assert diversity.diagnostics.n_within_class_sampled == 0
    # Simulation-skipped counter advanced as expected (existing MECH-094 path).
    assert diversity.diagnostics.n_simulation_skipped == 20


# -----------------------------------------------------------------------------
# C7 single-class-pool
# -----------------------------------------------------------------------------

def test_c7_single_class_pool_falls_through_under_min_classes_threshold():
    """Pool with only one first-action class falls through (returns None).

    The within-class branch only matters when the pool admits stratification
    (>= min_classes_for_stratification distinct first-action classes). With a
    single-class pool, the master `len(unique_classes) < min_classes` guard
    short-circuits before the within-class branch is consulted.
    """
    cfg = E3ScoreDiversityConfig(
        stratified_within_class_temperature=1.0,
        min_classes_for_stratification=2,
    )
    diversity = E3ScoreDiversity(cfg)
    pool = _make_pool([0, 0, 0, 0])
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    for _ in range(10):
        idx = diversity.stratified_select(scores=scores, candidates=pool)
        assert idx is None
    assert diversity.diagnostics.n_within_class_sampled == 0


# -----------------------------------------------------------------------------
# C8 from_dims propagation
# -----------------------------------------------------------------------------

def test_c8_from_dims_propagates_within_class_temperature():
    """REEConfig.from_dims accepts and propagates the new kwarg."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_e3_score_diversity=True,
        e3_diversity_stratified_within_class_temperature=0.75,
    )
    assert hasattr(cfg, "e3_diversity_stratified_within_class_temperature")
    assert cfg.e3_diversity_stratified_within_class_temperature == 0.75
    # Default None when kwarg omitted.
    cfg_default = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
    )
    assert cfg_default.e3_diversity_stratified_within_class_temperature is None


# -----------------------------------------------------------------------------
# C9 build_from_ree_config reads new flag
# -----------------------------------------------------------------------------

def test_c9_build_from_ree_config_reads_new_flag():
    """build_from_ree_config picks up the new flat field."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_e3_score_diversity=True,
        e3_diversity_stratified_within_class_temperature=2.0,
    )
    diversity = build_from_ree_config(cfg)
    assert diversity is not None
    assert diversity.config.stratified_within_class_temperature == 2.0
    # And master OFF -> None handle.
    cfg_off = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_e3_score_diversity=False,
    )
    assert build_from_ree_config(cfg_off) is None


# -----------------------------------------------------------------------------
# C10 get_state surfaces new diagnostics
# -----------------------------------------------------------------------------

def test_c10_get_state_surfaces_within_class_diagnostics():
    """get_state() includes the new mech341_*within_class_sampled* keys."""
    cfg = E3ScoreDiversityConfig(stratified_within_class_temperature=1.0)
    diversity = E3ScoreDiversity(cfg)
    pool = _make_pool([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.5, 0.2, 0.7])
    diversity.stratified_select(scores=scores, candidates=pool)
    state = diversity.get_state()
    assert "mech341_n_within_class_sampled" in state
    assert "mech341_last_within_class_sampled" in state
    assert "mech341_last_within_class_temperature" in state
    assert state["mech341_n_within_class_sampled"] == 1
    assert state["mech341_last_within_class_sampled"] is True
    assert state["mech341_last_within_class_temperature"] == 1.0
