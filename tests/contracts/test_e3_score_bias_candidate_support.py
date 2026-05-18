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


# --- V3-EXQ-563c: score/bias diagnostics tests ---


def test_score_diagnostics_populated_after_select():
    """last_score_diagnostics is populated on every select() call."""
    selector = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8))
    selector._running_variance = 0.0
    candidates = [_candidate(0), _candidate(1), _candidate(2)]

    selector.select(candidates, temperature=1.0)

    diag = selector.last_score_diagnostics
    assert "e3_raw_score_range_mean" in diag
    assert "e3_raw_score_std_mean" in diag
    assert "score_bias_abs_mean" in diag
    assert "score_bias_range_mean" in diag
    assert "score_bias_to_raw_range_ratio" in diag
    assert "normalize_score_bias_active" in diag
    assert "selected_candidate_rank_before_bias" in diag
    assert "selected_candidate_rank_after_bias" in diag
    # No bias supplied: abs_mean == 0, ratio == 0.
    assert diag["score_bias_abs_mean"] == 0.0
    assert diag["score_bias_to_raw_range_ratio"] == 0.0
    assert diag["normalize_score_bias_active"] is False


def test_bias_normalisation_rescales_bias_proportionally():
    """When normalize_score_bias_to_e3_range=True, bias magnitude is rescaled."""
    from ree_core.utils.config import E3Config as FullE3Config

    cfg = FullE3Config(world_dim=6, hidden_dim=8, normalize_score_bias_to_e3_range=True)
    selector_norm = E3TrajectorySelector(cfg)
    selector_plain = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8))

    # Fix running variance to 0 for deterministic argmin.
    selector_norm._running_variance = 0.0
    selector_plain._running_variance = 0.0

    candidates = [_candidate(0), _candidate(1), _candidate(2), _candidate(3)]
    # Tiny bias relative to score range -- normalisation will amplify it.
    bias = torch.tensor([0.0, 0.0, 0.0, -0.001])

    result_norm = selector_norm.select(candidates, temperature=1.0, score_bias=bias)
    result_plain = selector_plain.select(candidates, temperature=1.0, score_bias=bias)

    diag_norm = selector_norm.last_score_diagnostics
    diag_plain = selector_plain.last_score_diagnostics

    # Normalisation flag should be True when raw_score_range > 1e-6.
    if diag_norm["e3_raw_score_range_mean"] > 1e-6:
        assert diag_norm["normalize_score_bias_active"] is True
        # The rescaled bias-to-range ratio should equal 1.0 (by definition: bias_range
        # is scaled to match raw_score_range).
        assert abs(diag_norm["score_bias_to_raw_range_ratio"] - 1.0) < 1e-4
    else:
        # Degenerate case: flat scores, normalisation stays off.
        assert diag_norm["normalize_score_bias_active"] is False

    assert diag_plain["normalize_score_bias_active"] is False


def test_selected_rank_after_bias_reflects_bias_effect():
    """selected_candidate_rank_after_bias changes when a large bias alters selection."""
    selector = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8))
    selector._running_variance = 0.0  # committed path: argmin

    candidates = [_candidate(0), _candidate(1), _candidate(2), _candidate(3)]

    # Without bias, candidate 0 is selected (argmin of raw scores).
    result_no_bias = selector.select(candidates, temperature=1.0)
    diag_no_bias = dict(selector.last_score_diagnostics)

    # With a strong bias favouring candidate 3, it should now win.
    bias = torch.tensor([0.0, 0.0, 0.0, -1000.0])
    result_biased = selector.select(candidates, temperature=1.0, score_bias=bias)
    diag_biased = dict(selector.last_score_diagnostics)

    assert result_biased.selected_index == 3
    # Rank AFTER bias for the biased winner should be 0 (best after bias applied).
    assert diag_biased["selected_candidate_rank_after_bias"] == 0
    # The selected_candidate_rank_before_bias and after_bias diagnostics are present.
    assert diag_no_bias["selected_candidate_rank_before_bias"] == 0
    assert diag_no_bias["selected_candidate_rank_after_bias"] == 0
