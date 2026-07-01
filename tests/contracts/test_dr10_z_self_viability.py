"""Contracts for DR-10 (self_model_v4:SELF-3): z_self enters E3 viability scoring.

The E3-scoring half of the MECH-215 unblock, built on the DR-13 stateful z_self
(SELF-1). A no-op-default lever in E3TrajectorySelector.score_trajectory() adds a
per-candidate self-viability COST (derived from the stateful z_self -- capacity /
affect / damage state of THIS agent) so a trajectory less viable for the current
bodily state is discounted. Sibling to the DR-12 (SELF-4) PE lever on the same
machinery; no learned parameters.

Contract shape mirrors DR-12: bit-identical OFF; monotone in the cost; per-candidate
(a uniform signal is argmin-invariant -- the V3-EXQ-571 lesson); decisive (a
differential signal changes the committed argmin); with non-vacuity diagnostics.

generation:v4 -- off the V3 critical path; PROMOTES NOTHING.
"""

from __future__ import annotations

import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector


def _candidate(action_class: int, action_dim: int = 5) -> Trajectory:
    world_dim = 6
    horizon = 3
    states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    world_states = [torch.randn(1, world_dim) * 0.3 for _ in range(horizon + 1)]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class % action_dim] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _selector(**cfg_kw) -> E3TrajectorySelector:
    sel = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8, **cfg_kw))
    sel._running_variance = 0.0  # deterministic committed argmin path
    return sel


def _cands(seed: int = 0, k: int = 6):
    torch.manual_seed(seed)
    return [_candidate(i) for i in range(k)]


# ---------------------------------------------------------------------------
# C1: bit-identical OFF
# ---------------------------------------------------------------------------

def test_c1_lever_off_supplied_self_viability_is_noop():
    sel = _selector()  # use_self_viability_weighting defaults False
    cands = _cands()
    base = sel.select(cands, temperature=1.0)
    withsv = sel.select(
        cands, temperature=1.0,
        self_viability_per_candidate=torch.tensor([9.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert torch.allclose(base.scores, withsv.scores)
    assert base.selected_index == withsv.selected_index
    assert sel.last_score_diagnostics["self_viability_active"] is False


def test_c1_weight_zero_is_noop_even_when_enabled():
    sel = _selector(use_self_viability_weighting=True, self_viability_weight=0.0)
    cands = _cands()
    base = sel.select(cands, temperature=1.0)
    withsv = sel.select(
        cands, temperature=1.0,
        self_viability_per_candidate=torch.tensor([9.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert torch.allclose(base.scores, withsv.scores)
    assert sel.last_score_diagnostics["self_viability_active"] is False


def test_c1_enabled_but_no_signal_supplied_is_noop():
    sel = _selector(use_self_viability_weighting=True, self_viability_weight=1.0)
    cands = _cands()
    base = sel.select(cands, temperature=1.0)
    again = sel.select(cands, temperature=1.0)  # no self_viability supplied
    assert torch.allclose(base.scores, again.scores)
    assert sel.last_score_diagnostics["self_viability_active"] is False


# ---------------------------------------------------------------------------
# C2: a differential self-viability cost changes the committed argmin
# ---------------------------------------------------------------------------

def test_c2_differential_self_viability_flips_selection():
    sel = _selector(use_self_viability_weighting=True, self_viability_weight=1.0)
    cands = _cands()
    off = sel.select(cands, temperature=1.0)
    off_idx = int(off.selected_index)
    sorted_s = torch.sort(off.scores).values
    gap = float((sorted_s[1] - sorted_s[0]).item())
    sv = torch.zeros(len(cands))
    sv[off_idx] = gap + 1.0  # decisive penalty on the current best
    on = sel.select(cands, temperature=1.0, self_viability_per_candidate=sv)
    d = sel.last_score_diagnostics
    assert d["self_viability_active"] is True
    assert d["self_viability_range"] > 0.0
    assert d["self_viability_penalty_range"] > 0.0
    assert int(on.selected_index) != off_idx


# ---------------------------------------------------------------------------
# C3: a uniform self-viability cost is argmin-invariant (control)
# ---------------------------------------------------------------------------

def test_c3_uniform_self_viability_is_argmin_invariant():
    sel = _selector(use_self_viability_weighting=True, self_viability_weight=1.0)
    cands = _cands()
    off = sel.select(cands, temperature=1.0)
    off_idx = int(off.selected_index)
    on = sel.select(
        cands, temperature=1.0,
        self_viability_per_candidate=torch.full((len(cands),), 5.0),
    )
    assert int(on.selected_index) == off_idx
    assert sel.last_score_diagnostics["self_viability_range"] == 0.0


# ---------------------------------------------------------------------------
# C4: monotone penalty forms
# ---------------------------------------------------------------------------

def test_c4_linear_penalty_is_the_cost():
    sel = _selector(use_self_viability_weighting=True, self_viability_weight=1.0,
                    self_viability_mode="linear")
    p = sel._self_viability_penalty(torch.tensor(0.7), device="cpu", dtype=torch.float32)
    assert abs(float(p) - 0.7) < 1e-6


def test_c4_saturating_penalty_is_bounded_and_monotone():
    sel = _selector(use_self_viability_weighting=True, self_viability_weight=1.0,
                    self_viability_mode="saturating", self_viability_scale=1.0)
    p_small = float(sel._self_viability_penalty(torch.tensor(0.5), device="cpu", dtype=torch.float32))
    p_big = float(sel._self_viability_penalty(torch.tensor(3.0), device="cpu", dtype=torch.float32))
    assert 0.0 <= p_small < p_big <= 1.0


def test_c4_negative_cost_clamped_no_reward():
    """A negative self-viability value must not become a reward (clamped >= 0)."""
    sel = _selector(use_self_viability_weighting=True, self_viability_weight=1.0)
    p = sel._self_viability_penalty(torch.tensor(-5.0), device="cpu", dtype=torch.float32)
    assert float(p) == 0.0
