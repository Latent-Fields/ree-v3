"""Contracts for DR-12 (self_model_v4:SELF-4): E2 forward-PE -> E3 confidence down-weight.

FIRST V4 substrate build. The lever lives in E3TrajectorySelector.score_trajectory()
and is threaded per-candidate via select(e2_forward_pe_per_candidate=...). It must be
bit-identical OFF, monotone in PE magnitude, per-candidate (a uniform PE is
argmin-invariant -- the V3-EXQ-571 lesson), and decisive (a differential PE changes the
committed argmin), with non-vacuity diagnostics for the pilot.
"""

from __future__ import annotations

import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector


def _candidate(action_class: int, action_dim: int = 5) -> Trajectory:
    world_dim = 6
    horizon = 3
    states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    world_states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _selector(**cfg_kw) -> E3TrajectorySelector:
    sel = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8, **cfg_kw))
    sel._running_variance = 0.0  # deterministic committed argmin path
    return sel


# ---------------------------------------------------------------------------
# C1: bit-identical OFF
# ---------------------------------------------------------------------------

def test_c1_lever_off_supplied_pe_is_noop():
    """Lever OFF (default): supplying a per-candidate PE changes nothing."""
    sel = _selector()  # use_pe_confidence_weighting defaults False
    cands = [_candidate(0), _candidate(1), _candidate(2)]
    base = sel.select(cands, temperature=1.0)
    with_pe = sel.select(
        cands, temperature=1.0,
        e2_forward_pe_per_candidate=torch.tensor([10.0, 0.0, 0.0]),
    )
    assert torch.allclose(base.scores, with_pe.scores)
    assert base.selected_index == with_pe.selected_index
    assert sel.last_score_diagnostics["pe_confidence_active"] is False


def test_c1_weight_zero_is_noop_even_when_enabled():
    """Master ON but weight 0.0 -> the `!= 0.0` guard keeps it bit-identical."""
    sel = _selector(use_pe_confidence_weighting=True, pe_confidence_weight=0.0)
    cands = [_candidate(0), _candidate(1), _candidate(2)]
    base = sel.select(cands, temperature=1.0)
    with_pe = sel.select(
        cands, temperature=1.0,
        e2_forward_pe_per_candidate=torch.tensor([10.0, 0.0, 0.0]),
    )
    assert torch.allclose(base.scores, with_pe.scores)
    assert sel.last_score_diagnostics["pe_confidence_active"] is False


def test_c1_enabled_but_no_pe_supplied_is_noop():
    """Master ON + weight>0 but no per-candidate PE -> no penalty (None passthrough).

    Mutate the flag on ONE selector instance so the random-init scorer heads are held
    fixed (comparing scores across two freshly-built selectors would differ by random
    init, not the lever).
    """
    sel = _selector()
    cands = [_candidate(0), _candidate(1), _candidate(2)]
    before = sel.select(cands, temperature=1.0)
    sel.config.use_pe_confidence_weighting = True
    sel.config.pe_confidence_weight = 1.0
    after = sel.select(cands, temperature=1.0)  # still no e2_forward_pe_per_candidate
    assert torch.allclose(before.scores, after.scores)
    assert sel.last_score_diagnostics["pe_confidence_active"] is False


# ---------------------------------------------------------------------------
# C2: differential PE changes the committed argmin (decisiveness)
# ---------------------------------------------------------------------------

def test_c2_high_pe_on_primary_best_flips_selection():
    """Candidate 0 is primary-best via score_bias; a high PE on it flips selection."""
    cands = [_candidate(0), _candidate(1), _candidate(2)]
    bias = torch.tensor([-1.0, 0.0, 0.0])  # candidate 0 favoured by the primary

    sel_off = _selector()
    off = sel_off.select(cands, temperature=1.0, score_bias=bias)
    assert off.selected_index == 0  # unconditional-trust baseline picks the primary-best

    sel_on = _selector(use_pe_confidence_weighting=True, pe_confidence_weight=1.0)
    on = sel_on.select(
        cands, temperature=1.0, score_bias=bias,
        e2_forward_pe_per_candidate=torch.tensor([10.0, 0.0, 0.0]),  # high PE on cand 0
    )
    assert on.selected_index != 0  # discounted away from the poorly-modelled region
    assert sel_on.last_score_diagnostics["pe_confidence_active"] is True
    assert sel_on.last_score_diagnostics["e2_forward_pe_range"] == 10.0
    assert sel_on.last_score_diagnostics["pe_confidence_penalty_range"] > 0.0


# ---------------------------------------------------------------------------
# C3: per-candidate, not uniform -- a flat PE is argmin-invariant (V3-EXQ-571 lesson)
# ---------------------------------------------------------------------------

def test_c3_uniform_pe_does_not_change_selection():
    cands = [_candidate(0), _candidate(1), _candidate(2)]
    bias = torch.tensor([-1.0, 0.0, 0.0])
    sel_off = _selector()
    off = sel_off.select(cands, temperature=1.0, score_bias=bias)

    sel_on = _selector(use_pe_confidence_weighting=True, pe_confidence_weight=1.0)
    on = sel_on.select(
        cands, temperature=1.0, score_bias=bias,
        e2_forward_pe_per_candidate=torch.tensor([5.0, 5.0, 5.0]),  # uniform
    )
    # A uniform penalty shifts every candidate equally -> argmin unchanged.
    assert on.selected_index == off.selected_index
    assert sel_on.last_score_diagnostics["e2_forward_pe_range"] == 0.0


# ---------------------------------------------------------------------------
# C4: monotone in PE magnitude (score_trajectory level)
# ---------------------------------------------------------------------------

def test_c4_linear_penalty_monotone_and_equals_weight_times_pe():
    sel = _selector(
        use_pe_confidence_weighting=True, pe_confidence_weight=2.0,
        pe_confidence_mode="linear",
    )
    c = _candidate(0)
    s0 = float(sel.score_trajectory(c, e2_forward_pe=torch.tensor(0.0)).detach().mean())
    s1 = float(sel.score_trajectory(c, e2_forward_pe=torch.tensor(1.0)).detach().mean())
    s5 = float(sel.score_trajectory(c, e2_forward_pe=torch.tensor(5.0)).detach().mean())
    assert s1 > s0
    assert s5 > s1
    # linear: penalty == weight * pe -> the delta from pe=0 to pe=1 is exactly weight.
    assert abs((s1 - s0) - 2.0) < 1e-5
    assert abs((s5 - s0) - 10.0) < 1e-5


def test_c4_saturating_penalty_bounded_and_monotone():
    sel = _selector(
        use_pe_confidence_weighting=True, pe_confidence_weight=3.0,
        pe_confidence_mode="saturating", pe_confidence_scale=1.0,
    )
    c = _candidate(0)
    base = float(sel.score_trajectory(c, e2_forward_pe=torch.tensor(0.0)).detach().mean())
    big = float(sel.score_trajectory(c, e2_forward_pe=torch.tensor(1e6)).detach().mean())
    mid = float(sel.score_trajectory(c, e2_forward_pe=torch.tensor(1.0)).detach().mean())
    # saturating penalty in [0, weight); monotone; bounded by weight.
    assert mid > base
    assert big > mid
    delta_big = big - base
    assert 0.0 < delta_big < 3.0 + 1e-6  # < weight (strictly bounded)


# ---------------------------------------------------------------------------
# C5: PE magnitude is non-negative-monotone (negative PE clamped to 0)
# ---------------------------------------------------------------------------

def test_c5_negative_pe_clamped_no_reward():
    """A negative PE value cannot REWARD a trajectory (penalty clamps at 0)."""
    sel = _selector(use_pe_confidence_weighting=True, pe_confidence_weight=1.0)
    c = _candidate(0)
    s_zero = sel.score_trajectory(c, e2_forward_pe=torch.tensor(0.0))
    s_neg = sel.score_trajectory(c, e2_forward_pe=torch.tensor(-5.0))
    assert torch.allclose(s_zero, s_neg)  # clamp(min=0) -> no negative penalty
