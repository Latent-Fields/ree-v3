"""Contracts for EXQ-563 candidate-support diagnosis.

These tests separate E3 score-bias wiring from upstream candidate support.
"""

from __future__ import annotations

import torch

from ree_core.agent import candidate_support_preflight
from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector


def _candidate(action_class: int, action_dim: int = 5) -> Trajectory:
    world_dim = 6
    horizon = 3
    states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    world_states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class] = 1.0
    return Trajectory(
        states=states,
        actions=actions,
        world_states=world_states,
    )


def test_e3_score_bias_selects_manual_multi_action_candidate():
    selector = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8))
    selector._running_variance = 0.0  # deterministic argmin path
    candidates = [_candidate(0), _candidate(1), _candidate(2)]

    baseline = selector.select(
        candidates,
        temperature=1.0,
        score_bias=torch.tensor([0.0, 0.0, 0.0]),
    )
    biased = selector.select(
        candidates,
        temperature=1.0,
        score_bias=torch.tensor([0.0, 0.0, -100.0]),
    )

    assert baseline.selected_index == 0
    assert int(baseline.selected_action.argmax(dim=-1).item()) == 0
    assert biased.selected_index == 2
    assert int(biased.selected_action.argmax(dim=-1).item()) == 2
    assert biased.scores[2] < biased.scores[0]


def test_candidate_support_preflight_marks_single_class_not_run():
    candidates = [_candidate(4), _candidate(4), _candidate(4)]

    diag = candidate_support_preflight(
        candidates,
        forced_score_bias_per_class=[-2.0, 0.0, 0.0, 0.0],
    )

    assert diag["candidate_first_action_counts"] == {4: 3}
    assert diag["candidate_unique_first_action_classes"] == 1
    assert diag["candidate_first_action_entropy"] == 0.0
    assert diag["forced_bias_abs_mean"] == 0.0
    assert diag["preflight_status"] == "NOT_RUN"
    assert diag["not_run_reason"] == "candidate_support_collapse"
    assert diag["interpretation"] == "NOT_RUN: candidate_support_collapse"


def test_candidate_support_preflight_allows_multi_class_surface():
    candidates = [_candidate(0), _candidate(1), _candidate(4)]

    diag = candidate_support_preflight(
        candidates,
        forced_score_bias_per_class=[-2.0, 0.0, 0.0, 0.0],
    )

    assert diag["candidate_unique_first_action_classes"] == 3
    assert diag["candidate_first_action_entropy"] > 0.0
    assert diag["forced_bias_abs_mean"] == 2.0 / 3.0
    assert diag["forced_bias_nonzero_candidate_count"] == 1
    assert diag["preflight_status"] == "RUN"
