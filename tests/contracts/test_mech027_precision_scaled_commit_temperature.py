"""Contracts for the MECH-027 precision-scaled commit temperature lever.

MECH-027 (claims.yaml) needs current_precision / running_variance to have a
GRADED behavioral effect for its hypervigilance falsifier. Before this lever,
the ONLY consumer of precision was the binary commit gate (committed =
commit_variance < effective_threshold, ARC-016) -- and that gate saturates in
a trained baseline (running_variance empirically ~125x below threshold), so
once committed, further lowering variance (an elevated-gain/hypervigilance
push) had no further observable effect anywhere downstream.

use_precision_scaled_commit_temperature (E3Config) softens the committed
argmin into multinomial(softmax(-cost / T_eff)) with T_eff = base_temperature
+ precision_scaled_commit_entropy_alpha * (1 - precision_margin_norm), where
precision_margin_norm = clamp(1 - commit_variance/effective_threshold, 0, 1):
0 at the threshold (barely committed) -> hot; 1 as commit_variance -> 0
(maximally confident) -> cold (recovers the hard argmin). This is the same
softening shape as MECH-439 Factor B (gap-scaled commit temperature), with
the F-gap quantity replaced by the precision-margin quantity, and reuses
Factor B's F-eligibility-envelope safety gate.

Coverage: OFF bit-identical; graded T_eff monotone in precision_margin_norm;
cold/confident recovers the decisive argmin; hot/barely-committed spreads
across candidates; standalone safety gate excludes a clearly-harmful
candidate; gap-scaled commit temperature takes precedence when both levers
are enabled (preserves existing Factor-B-first composition).
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
    sel._running_variance = 0.0
    return sel


def _patch_raw(selector, candidates, raw_costs):
    raw_map = {id(c): torch.tensor([float(v)]) for c, v in zip(candidates, raw_costs)}
    selector.score_trajectory = lambda cand, **kw: raw_map[id(cand)]


# --------------------------------------------------------------------------- #
# OFF bit-identical                                                           #
# --------------------------------------------------------------------------- #


def test_off_default_flags_present_and_false():
    cfg = E3Config(world_dim=6, hidden_dim=8)
    assert cfg.use_precision_scaled_commit_temperature is False
    assert cfg.precision_scaled_commit_entropy_alpha == 1.0
    assert cfg.precision_scaled_commit_harm_floor == 0.25


def test_off_is_hard_argmin():
    sel = _selector()  # flag off
    candidates = [_candidate(i) for i in range(5)]
    _patch_raw(sel, candidates, [3.0, 1.0, 2.0, 4.0, 5.0])
    r = sel.select(candidates, temperature=1.0, score_bias=torch.zeros(5))
    assert r.selected_index == 1  # argmin(scores)
    d = sel.last_score_diagnostics
    assert d["precision_scaled_commit_active"] is False


def test_precision_margin_norm_always_populated_in_world_variance_mode():
    """Diagnostic is set regardless of whether the lever is active, mirroring
    conflict_gap_norm's always-set convention (per-tick regression falsifier
    needs to bin every tick uniformly)."""
    sel = _selector(commitment_threshold=0.40)
    sel._running_variance = 0.10  # committed; margin = 1 - 0.10/0.40 = 0.75
    candidates = [_candidate(i) for i in range(3)]
    _patch_raw(sel, candidates, [0.0, 1.0, 2.0])
    sel.select(candidates, temperature=1.0, score_bias=torch.zeros(3))
    d = sel.last_score_diagnostics
    assert abs(d["precision_margin_norm"] - 0.75) < 1e-6


# --------------------------------------------------------------------------- #
# _precision_scaled_commit_pick: graded temperature, direct unit coverage     #
# --------------------------------------------------------------------------- #


def test_temperature_monotone_and_precision_scaling_load_bearing():
    sel = _selector(precision_scaled_commit_entropy_alpha=2.0)
    sel._precision_scaled_commit_pick(
        torch.tensor([0.0, 1.0]), precision_norm=0.0, base_temperature=1.0
    )
    t_hot = sel.last_score_diagnostics["precision_scaled_commit_temperature_eff"]
    sel._precision_scaled_commit_pick(
        torch.tensor([0.0, 1.0]), precision_norm=1.0, base_temperature=1.0
    )
    t_cold = sel.last_score_diagnostics["precision_scaled_commit_temperature_eff"]
    # T_eff = base + alpha*(1 - precision_norm): hot at margin 0, base at margin 1.
    assert abs(t_hot - (1.0 + 2.0 * 1.0)) < 1e-6
    assert abs(t_cold - 1.0) < 1e-6
    assert t_hot > t_cold + 1e-6


def test_cold_confident_recovers_decisive_winner():
    """At precision_norm=1 (maximally confident) with a small base temperature,
    the committed pick concentrates on the argmin -- recovering the hard
    argmin the binary gate alone already gave, i.e. the OFF baseline."""
    sel = _selector()
    cost = torch.tensor([0.0, 100.0, 100.0, 100.0])
    for seed in range(6):
        torch.manual_seed(seed)
        local = sel._precision_scaled_commit_pick(
            cost, precision_norm=1.0, base_temperature=0.01
        )
        assert local == 0


def test_hot_barely_committed_softens_pick():
    """At precision_norm=0 (barely committed, at the threshold) with a hot
    temperature, the committed pick spreads across candidates -- the graded
    effect the binary gate alone could never produce once already committed."""
    sel = _selector()
    cost = torch.tensor([0.0, 0.01, 0.02, 0.03])
    picks = set()
    for seed in range(60):
        torch.manual_seed(seed)
        picks.add(
            sel._precision_scaled_commit_pick(
                cost, precision_norm=0.0, base_temperature=1.0
            )
        )
    assert len(picks) >= 2  # softened argmax, not collapsed


# --------------------------------------------------------------------------- #
# End-to-end through select(): safety gate + precedence                       #
# --------------------------------------------------------------------------- #


def test_end_to_end_spreads_when_barely_committed_and_envelope_wide():
    sel = _selector(
        use_precision_scaled_commit_temperature=True,
        precision_scaled_commit_entropy_alpha=5.0,
        precision_scaled_commit_harm_floor=1.0,  # wide envelope -- all eligible
        commitment_threshold=0.40,
    )
    sel._running_variance = 0.39  # barely committed -> margin ~0.025 -> hot
    candidates = [_candidate(i) for i in range(4)]
    _patch_raw(sel, candidates, [0.0, 0.01, 0.02, 0.03])
    picks = set()
    for seed in range(60):
        torch.manual_seed(seed)
        r = sel.select(candidates, temperature=1.0, score_bias=torch.zeros(4))
        picks.add(r.selected_index)
    assert len(picks) >= 2
    assert sel.last_score_diagnostics["precision_scaled_commit_active"] is True


def test_standalone_safety_gate_excludes_harmful():
    """A clearly-harmful candidate (far outside the F-eligibility envelope) is
    never softmax-promoted even under an overwhelming modulatory bias."""
    sel = _selector(
        use_precision_scaled_commit_temperature=True,
        precision_scaled_commit_entropy_alpha=5.0,
        precision_scaled_commit_harm_floor=0.25,
        commitment_threshold=0.40,
    )
    sel._running_variance = 0.39
    candidates = [_candidate(i) for i in range(8)]
    _patch_raw(sel, candidates, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0])
    bias = torch.zeros(8)
    bias[7] = -1000.0
    for seed in range(12):
        torch.manual_seed(seed)
        r = sel.select(candidates, temperature=1.0, score_bias=bias)
        assert r.selected_index != 7


def test_gap_scaled_commit_temperature_takes_precedence_when_both_enabled():
    """When both MECH-439 Factor B and the MECH-027 precision lever are on,
    gap-scaling wins -- preserves the pre-existing Factor-B-first composition
    semantics exactly (no new combinatorial behavior for existing users of
    use_gap_scaled_commit_temperature)."""
    sel = _selector(
        use_precision_scaled_commit_temperature=True,
        use_gap_scaled_commit_temperature=True,
        gap_scaled_commit_harm_floor=1.0,
        precision_scaled_commit_harm_floor=1.0,
        commitment_threshold=0.40,
    )
    sel._running_variance = 0.39
    candidates = [_candidate(i) for i in range(4)]
    _patch_raw(sel, candidates, [0.0, 0.01, 0.02, 0.03])
    torch.manual_seed(0)
    sel.select(candidates, temperature=1.0, score_bias=torch.zeros(4))
    d = sel.last_score_diagnostics
    assert d["gap_scaled_commit_active"] is True
    assert d["precision_scaled_commit_active"] is False
