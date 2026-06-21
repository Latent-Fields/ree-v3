"""Contracts for MECH-449 / ARC-107 Go/No-Go eligibility constitution.

The bounded Go/No-Go pressure set acts on the MECH-448 eligible set (the
shortlist/envelope) BEFORE the within-eligible _modulatory_accum arbitration:
No-Go suppresses unsafe/stale/perseverative/low-viability candidates on an axis
ORTHOGONAL to F-rank (which rank-preserving demotion structurally cannot), and
bounded Go re-admits a lawfully-eligible demoted candidate. Master flag default
False -> bit-identical OFF. PROMOTES NOTHING.

C2/C3/C5/C6 use a flat-F candidate bank so the MECH-448 envelope is WIDE
(low-conflict -> all eligible) and the Go/No-Go gate is the ISOLATED variable.
C4 uses a distinct-F bank + top_k shortlist so the shortlist genuinely EXCLUDES
candidates that bounded Go must re-admit.
"""

from __future__ import annotations

import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector


def _candidate(action_class: int, world_val: float = 0.0,
               action_dim: int = 5, world_dim: int = 6,
               horizon: int = 3) -> Trajectory:
    states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    world_states = [
        torch.full((1, world_dim), float(world_val)) for _ in range(horizon + 1)
    ]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _selector(**overrides) -> E3TrajectorySelector:
    sel = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8, **overrides))
    sel._running_variance = 0.0  # deterministic committed argmin path
    return sel


def _flat_cands():
    # Flat F -> MECH-448 envelope returns ALL eligible (wide); the gate isolates.
    return [_candidate(i, 0.0) for i in range(4)]


def _distinct_cands():
    # Distinct world values -> distinct raw F so top_k strictly subsets.
    return [
        _candidate(0, 0.0),
        _candidate(1, 0.5),
        _candidate(2, 1.0),
        _candidate(3, 1.5),
    ]


def test_c1_master_off_bit_identical_even_with_signals():
    """Gate OFF (default): passing go_nogo_signals is ignored -> bit-identical."""
    sel_a = _selector()
    sel_b = _selector()
    bias = torch.tensor([0.0, 0.0, -5.0, 0.0])  # modulatory favours idx2
    r_no_sig = sel_a.select(_flat_cands(), temperature=1.0, score_bias=bias)
    r_with_sig = sel_b.select(
        _flat_cands(), temperature=1.0, score_bias=bias,
        go_nogo_signals={"safety": torch.tensor([0.0, 0.0, 0.9, 0.0])},
    )
    assert r_no_sig.selected_index == r_with_sig.selected_index == 2
    assert "go_nogo_constitution_active" not in sel_b.last_score_diagnostics


def test_c2_nogo_suppresses_within_eligible():
    """No-Go drops the modulatory-favoured candidate from the eligible set."""
    bias = torch.tensor([0.0, 0.0, -5.0, 0.0])  # idx2 is the modulatory winner
    base = _selector(
        use_f_eligibility_demotion=True, use_go_nogo_constitution=True
    ).select(_flat_cands(), temperature=1.0, score_bias=bias)
    assert base.selected_index == 2  # without a No-Go signal, idx2 wins
    sel = _selector(
        use_f_eligibility_demotion=True, use_go_nogo_constitution=True
    )
    gated = sel.select(
        _flat_cands(), temperature=1.0, score_bias=bias,
        go_nogo_signals={"staleness": torch.tensor([0.0, 0.0, 0.9, 0.0])},
    )
    assert gated.selected_index != 2
    assert sel.last_score_diagnostics["go_nogo_n_soft_applied"] >= 1


def test_c3_safety_holds_under_overwhelming_modulatory_pull():
    """SAFETY: a safety-No-Go'd candidate is never selected, even when the
    modulatory pull toward it is overwhelming (orthogonal-to-F guarantee)."""
    sel = _selector(
        use_f_eligibility_demotion=True, use_go_nogo_constitution=True
    )
    bias = torch.tensor([0.0, 0.0, -1.0e6, 0.0])  # idx2 dominates modulation
    gated = sel.select(
        _flat_cands(), temperature=1.0, score_bias=bias,
        go_nogo_signals={"safety": torch.tensor([0.0, 0.0, 0.9, 0.0])},
    )
    assert gated.selected_index != 2
    assert sel.last_score_diagnostics["go_nogo_n_safety_nogo"] >= 1


def test_c4_go_promotes_demoted_candidate():
    """Bounded Go re-admits a candidate the F top_k shortlist demoted out, so a
    strong modulatory bias on it can win once promoted."""
    # Use ONE selector instance throughout: each E3TrajectorySelector has random
    # head init, so the worst-F (shortlist-excluded) index must be discovered on
    # the SAME instance the gated select runs on. With zero bias + no Go signal
    # the SelectionResult scores ARE the raw F scores.
    sel = _selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k", modulatory_shortlist_k=2,
        use_go_nogo_constitution=True, gng_go_threshold=0.5, gng_go_max_promote=2,
    )
    probe = sel.select(
        _distinct_cands(), temperature=1.0, score_bias=torch.zeros(4),
    )
    assert sel.last_score_diagnostics["modulatory_shortlist_size"] == 2
    raw = torch.tensor([float(s.detach()) for s in probe.scores])
    target = int(torch.topk(raw, 2, largest=True).indices.tolist()[0])  # worst-F excluded
    go_sig = torch.zeros(4)
    go_sig[target] = 0.9
    bias = torch.zeros(4)
    bias[target] = -50.0  # strong modulatory pull once promoted
    gated = sel.select(
        _distinct_cands(), temperature=1.0, score_bias=bias,
        go_nogo_signals={"go": go_sig},
    )
    assert sel.last_score_diagnostics["go_nogo_n_go_promoted"] >= 1
    assert gated.selected_index == target


def test_c5_composes_with_mech448_envelope():
    """The gate runs ON the MECH-448 f_demotion eligible set (both active)."""
    sel = _selector(
        use_f_eligibility_demotion=True, use_go_nogo_constitution=True
    )
    sel.select(
        _flat_cands(), temperature=1.0,
        score_bias=torch.tensor([0.0, 0.0, -1.0, 0.0]),
        go_nogo_signals={"perseveration": torch.tensor([0.0, 0.0, 0.9, 0.0])},
    )
    assert sel.last_score_diagnostics.get("f_eligibility_demotion_active") is True
    assert sel.last_score_diagnostics.get("go_nogo_constitution_active") is True


def test_c6_nogo_never_empties_set_failopen():
    """Soft No-Go on every eligible candidate keeps the strongest-F survivor
    (fail-open guard against No-Go-over-pressure deadlock)."""
    sel = _selector(
        use_f_eligibility_demotion=True, use_go_nogo_constitution=True,
        gng_protect_min_eligible=1,
    )
    gated = sel.select(
        _flat_cands(), temperature=1.0, score_bias=torch.zeros(4),
        go_nogo_signals={"staleness": torch.tensor([0.9, 0.9, 0.9, 0.9])},
    )
    assert gated.selected_index is not None
    assert sel.last_score_diagnostics["go_nogo_envelope_size"] >= 1
