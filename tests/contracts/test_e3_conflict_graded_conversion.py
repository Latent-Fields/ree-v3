"""Contracts for the MECH-439 F-dominance conflict-grade conversion levers.

Two no-op-default renderings of the BG hyperdirect conflict-grade in
E3TrajectorySelector.select():

  FACTOR A -- conflict-graded shortlist width: k = clamp(round(k_max -
    (k_max-1)*gap_norm), 1, K), where gap_norm in [0, 1] is the normalized
    top-F gap. Near-ties widen k; a decisive F-gap narrows k to 1. F gates
    ELIGIBILITY only; it is absent from the within-set arbitration.

  FACTOR B -- gap-scaled entropy-regularized commit: the committed argmin is
    softened into multinomial(softmax(-cost / T_eff)) with
    T_eff = base_T + alpha*(1 - gap_norm), over the F-bounded eligible set
    (shortlist) or an F-eligibility envelope (standalone safety gate).

Coverage: OFF bit-identical; graded quantity (k, T_eff) monotone in the F-gap
and clamps; F absent from Factor-A within-set arbitration; Factor-A and
Factor-B safety gates (a clearly-harmful candidate is never selected).
"""

from __future__ import annotations

import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector


def _candidate(action_class: int, action_dim: int = 8) -> Trajectory:
    world_dim = 6
    horizon = 3
    states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    world_states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class % action_dim] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _selector(**e3_kwargs) -> E3TrajectorySelector:
    sel = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8, **e3_kwargs))
    sel._running_variance = 0.0  # committed path (deterministic argmin baseline)
    return sel


def _patch_raw(selector, candidates, raw_costs):
    """Force score_trajectory to return a known per-candidate raw cost."""
    raw_map = {id(c): torch.tensor([float(v)]) for c, v in zip(candidates, raw_costs)}
    selector.score_trajectory = lambda cand, **kw: raw_map[id(cand)]


# --------------------------------------------------------------------------- #
# OFF bit-identical                                                           #
# --------------------------------------------------------------------------- #


def test_off_is_bit_identical_to_legacy_top_k_argmin():
    """Both factors default OFF: top_k shortlist takes the legacy fixed-k
    argmin-over-modulatory-channel path, deterministically."""
    sel = _selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=3,
    )
    candidates = [_candidate(i) for i in range(8)]
    # F-best three are indices {0,1,2}; within them the modulatory-min is idx 1.
    _patch_raw(sel, candidates, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    bias = torch.tensor([0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    r1 = sel.select(candidates, temperature=1.0, score_bias=bias)
    r2 = sel.select(candidates, temperature=1.0, score_bias=bias)

    assert r1.selected_index == 1  # legacy argmin(mod_eligible) over top-3
    assert r2.selected_index == 1  # deterministic
    d = sel.last_score_diagnostics
    assert d["modulatory_shortlist_conflict_graded"] is False
    assert d["gap_scaled_commit_active"] is False
    assert d["modulatory_shortlist_size"] == 3


def test_off_default_flags_present_and_false():
    """The new flags exist and default to no-op on E3Config."""
    cfg = E3Config(world_dim=6, hidden_dim=8)
    assert cfg.modulatory_shortlist_conflict_graded is False
    assert cfg.modulatory_shortlist_k_max == 6
    assert cfg.use_gap_scaled_commit_temperature is False
    assert cfg.gap_scaled_commit_entropy_alpha == 1.0
    assert cfg.gap_scaled_commit_harm_floor == 0.25


# --------------------------------------------------------------------------- #
# FACTOR A -- conflict-graded width: monotone in gap + clamps                  #
# --------------------------------------------------------------------------- #


def _effective_k(gap_norm_costs):
    sel = _selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_conflict_graded=True,
        modulatory_shortlist_k_max=6,
    )
    candidates = [_candidate(i) for i in range(8)]
    _patch_raw(sel, candidates, gap_norm_costs)
    sel.select(
        candidates,
        temperature=1.0,
        score_bias=torch.zeros(8),
    )
    d = sel.last_score_diagnostics
    return d["modulatory_shortlist_k_effective"], d["modulatory_shortlist_gap_norm"]


def test_factor_a_width_clamps_to_k_max_at_near_tie():
    # range 7.0, top-F gap 0.0 -> gap_norm 0 -> k = k_max = 6.
    costs = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    k, gap = _effective_k(costs)
    assert abs(gap - 0.0) < 1e-6
    assert k == 6


def test_factor_a_width_clamps_to_one_at_decisive_gap():
    # range 1.0, top-F gap 1.0 -> gap_norm 1 -> k = 1.
    costs = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    k, gap = _effective_k(costs)
    assert abs(gap - 1.0) < 1e-6
    assert k == 1


def test_factor_a_width_monotone_non_increasing_in_gap():
    # range fixed at 1.0 (min 0.0, max 1.0); widen the top-F gap from 0 -> 1.
    # All non-first entries sit at the max so the second-smallest is exactly g.
    base = [0.0, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ks = []
    for g in (0.0, 0.2, 0.5, 0.8, 1.0):
        costs = list(base)
        costs[1] = g  # second-smallest -> top-F gap == g (range stays 1.0)
        k, gap = _effective_k(costs)
        assert abs(gap - g) < 1e-6
        ks.append(k)
    assert ks == sorted(ks, reverse=True)  # non-increasing
    assert ks[0] == 6 and ks[-1] == 1  # endpoints clamp
    assert all(1 <= k <= 8 for k in ks)  # within [1, K]


# --------------------------------------------------------------------------- #
# FACTOR A -- F absent from within-set arbitration + safety                    #
# --------------------------------------------------------------------------- #


def test_factor_a_f_absent_from_within_set_arbitration():
    """Within the F-bounded eligible set, selection is by the MODULATORY
    channel, NOT by F. The F-best candidate is not picked when a different
    eligible candidate carries the lower modulatory cost."""
    sel = _selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_conflict_graded=True,
        modulatory_shortlist_k_max=4,
    )
    candidates = [_candidate(i) for i in range(8)]
    # near-tie among the F-best so width is wide (k=k_max=4); eligible {0,1,2,3}.
    _patch_raw(sel, candidates, [0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0])
    # F-best is idx 0; modulatory-min within eligible is idx 3.
    bias = torch.tensor([0.0, 0.0, 0.0, -9.0, 0.0, 0.0, 0.0, 0.0])

    r = sel.select(candidates, temperature=1.0, score_bias=bias)

    assert r.selected_index == 3  # modulatory-min, not F-min (0)


def test_factor_a_safety_clearly_harmful_never_admitted():
    """A clearly-harmful candidate (large F-gap above the best, outside top-k)
    is never selected, even with an overwhelmingly favourable modulatory bias."""
    sel = _selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_conflict_graded=True,
        modulatory_shortlist_k_max=3,
    )
    candidates = [_candidate(i) for i in range(8)]
    # idx 7 is the worst by F (far outside any conflict-graded top-k); make it
    # the single most attractive modulatory candidate.
    _patch_raw(sel, candidates, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 50.0])
    bias = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1000.0])

    for _ in range(8):
        r = sel.select(candidates, temperature=1.0, score_bias=bias)
        assert r.selected_index != 7


# --------------------------------------------------------------------------- #
# FACTOR B -- gap-scaled commit temperature                                    #
# --------------------------------------------------------------------------- #


def _t_eff(gap_norm_costs, alpha=2.0, base_t=1.0):
    sel = _selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=4,
        use_gap_scaled_commit_temperature=True,
        gap_scaled_commit_entropy_alpha=alpha,
    )
    candidates = [_candidate(i) for i in range(8)]
    _patch_raw(sel, candidates, gap_norm_costs)
    torch.manual_seed(0)
    sel.select(candidates, temperature=base_t, score_bias=torch.zeros(8))
    d = sel.last_score_diagnostics
    return d["gap_scaled_commit_temperature_eff"], d["gap_scaled_commit_gap_norm"]


def test_factor_b_temperature_monotone_and_gap_scaling_load_bearing():
    # range 1.0; near-tie (gap 0) vs decisive (gap 1).
    t_hot, g_hot = _t_eff([0.0, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], alpha=2.0)
    t_cold, g_cold = _t_eff([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], alpha=2.0)
    assert abs(g_hot - 0.0) < 1e-6 and abs(g_cold - 1.0) < 1e-6
    # T_eff = base + alpha*(1 - gap_norm): hot at near-tie, base at decisive gap.
    assert abs(t_hot - (1.0 + 2.0 * 1.0)) < 1e-6
    assert abs(t_cold - 1.0) < 1e-6
    # The (1 - gap_norm) scaling is load-bearing: a gap-blind T would be flat.
    assert t_hot > t_cold + 1e-6


def test_factor_b_cold_preserves_decisive_winner():
    """At a decisive gap with a small base temperature, the committed pick
    concentrates on the modulatory argmin (recovers the hard argmin)."""
    sel = _selector()
    cost = torch.tensor([0.0, 100.0, 100.0, 100.0])
    for seed in range(6):
        torch.manual_seed(seed)
        # gap_norm 1.0, base 0.01 -> T_eff 0.01 -> softmax mass ~all on idx 0.
        local = sel._gap_scaled_commit_pick(cost, gap_norm=1.0, base_temperature=0.01)
        assert local == 0


def test_factor_b_hot_softens_near_tie():
    """At a near-tie with a hot temperature, the committed pick spreads across
    candidates instead of always taking the argmin."""
    sel = _selector()
    cost = torch.tensor([0.0, 0.01, 0.02, 0.03])
    picks = set()
    for seed in range(60):
        torch.manual_seed(seed)
        picks.add(
            sel._gap_scaled_commit_pick(cost, gap_norm=0.0, base_temperature=1.0)
        )
    assert len(picks) >= 2  # softened argmax, not collapsed


def test_factor_b_standalone_safety_gate_excludes_harmful():
    """With no Factor-A shortlist, Factor B restricts the gap-scaled softmax to
    an F-eligibility envelope so a hot commit-T can never softmax-promote a
    clearly-harmful candidate, even when its composed score is the lowest."""
    sel = _selector(
        use_gap_scaled_commit_temperature=True,
        gap_scaled_commit_entropy_alpha=5.0,
        gap_scaled_commit_harm_floor=0.25,
    )
    candidates = [_candidate(i) for i in range(8)]
    # range 8.0; idx 7 raw=8.0 is far outside the 0.25*range=2.0 envelope.
    _patch_raw(sel, candidates, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0])
    # idx 7 has an overwhelmingly favourable composed score (bias -1000).
    bias = torch.zeros(8)
    bias[7] = -1000.0

    for seed in range(12):
        torch.manual_seed(seed)
        r = sel.select(candidates, temperature=1.0, score_bias=bias)
        assert r.selected_index != 7  # harmful candidate never promoted


def test_factor_b_standalone_off_is_hard_argmin():
    """Factor B OFF (no shortlist): committed selection is the legacy
    argmin over composed scores."""
    sel = _selector()  # both factors off, no shortlist
    candidates = [_candidate(i) for i in range(5)]
    _patch_raw(sel, candidates, [3.0, 1.0, 2.0, 4.0, 5.0])
    bias = torch.zeros(5)
    r = sel.select(candidates, temperature=1.0, score_bias=bias)
    assert r.selected_index == 1  # argmin(scores)
