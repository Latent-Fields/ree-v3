"""Contracts for MECH-448 / ARC-107 rank-preserving F->eligibility demotion.

The LEAD lever of the basal-ganglia E3-selector constitution. F (the primary
harm/goal score) decides who is ELIGIBLE to compete, not who wins: F is
renormalised against the competing field by a divisive-normalisation analog
(share of pooled merit, absolute floor), producing a graded, conflict-scaled,
rank-preserving eligibility envelope; the existing _modulatory_accum then
arbitrates the committed action WITHIN the envelope with F REMOVED from the
final argmin.

Coverage:
  - config defaults are no-op (bit-identical OFF guarantee surface);
  - OFF == legacy committed argmin (F+bias decides);
  - graded width: a near-tie envelope is wider than a decisive one;
  - NON-DEGENERACY: the envelope actually excludes on a divergent pool;
  - SAFETY: a clearly-harmful candidate outside the envelope is never selected
    even under overwhelming modulatory pull toward it;
  - F removed from the within-eligible argmin (winner is the modulatory argmin,
    not the F-argmin);
  - rank-preserving: the eligible set is an F-rank prefix;
  - the lever requires a modulatory channel (no-op when _modulatory_accum None);
  - an exact N-way tie -> wide fallback (excluded_count 0; non-degeneracy=false);
  - from_dims surfaces the flags onto config.e3.
"""

from __future__ import annotations

import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector
from ree_core.utils.config import REEConfig


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
    sel._running_variance = 0.0  # force committed path (deterministic argmin)
    return sel


def _patch_raw(selector, candidates, raw_costs):
    """Force score_trajectory to return a known per-candidate raw F cost."""
    raw_map = {id(c): torch.tensor([float(v)]) for c, v in zip(candidates, raw_costs)}
    selector.score_trajectory = lambda cand, **kw: raw_map[id(cand)]


# --------------------------------------------------------------------------- #
# Config defaults / wiring                                                     #
# --------------------------------------------------------------------------- #


def test_config_defaults_are_noop():
    cfg = E3Config(world_dim=6, hidden_dim=8)
    assert cfg.use_f_eligibility_demotion is False
    assert cfg.f_eligibility_envelope_floor == 0.30
    assert cfg.f_eligibility_dn_sigma == 0.0


def test_from_dims_surfaces_flags_onto_e3():
    cfg = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4,
        self_dim=16, world_dim=16,
        use_f_eligibility_demotion=True,
        f_eligibility_envelope_floor=0.4,
        f_eligibility_dn_sigma=0.1,
    )
    assert cfg.e3.use_f_eligibility_demotion is True
    assert cfg.e3.f_eligibility_envelope_floor == 0.4
    assert cfg.e3.f_eligibility_dn_sigma == 0.1
    # Default from_dims leaves the lever OFF.
    cfg_off = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4,
        self_dim=16, world_dim=16,
    )
    assert cfg_off.e3.use_f_eligibility_demotion is False


# --------------------------------------------------------------------------- #
# OFF bit-identical                                                            #
# --------------------------------------------------------------------------- #


def test_off_is_legacy_committed_argmin():
    """Lever OFF (default): committed selection is argmin(scores) = argmin(F+bias).
    A strong bias toward the F-worst candidate WINS (no eligibility envelope)."""
    sel = _selector()  # use_f_eligibility_demotion=False
    candidates = [_candidate(i) for i in range(4)]
    _patch_raw(sel, candidates, [0.0, 0.1, 0.2, 0.3])
    bias = torch.tensor([0.0, 0.0, 0.0, -10.0])  # pull toward F-worst idx3
    r = sel.select(candidates, temperature=1.0, score_bias=bias)
    assert r.selected_index == 3  # F+bias argmin (lever off)
    d = sel.last_score_diagnostics
    assert d["f_eligibility_demotion_active"] is False
    assert d["f_eligibility_envelope_size"] == -1  # pre-seed untouched


# --------------------------------------------------------------------------- #
# Graded, conflict-scaled width                                               #
# --------------------------------------------------------------------------- #


def test_graded_width_near_tie_wider_than_decisive():
    """The envelope width scales with conflict: a near-tie admits more candidates
    than a decisive F-winner pool (the hyperdirect conflict-grade)."""
    near_costs = [0.0, 0.05, 0.10, 0.15]
    dec_costs = [0.0, 1.0, 1.0, 1.0]

    sel_near = _selector(use_f_eligibility_demotion=True)
    cands_near = [_candidate(i) for i in range(4)]
    _patch_raw(sel_near, cands_near, near_costs)
    sel_near.select(cands_near, temperature=1.0, score_bias=torch.zeros(4))
    n_near = sel_near.last_score_diagnostics["f_eligibility_envelope_size"]

    sel_dec = _selector(use_f_eligibility_demotion=True)
    cands_dec = [_candidate(i) for i in range(4)]
    _patch_raw(sel_dec, cands_dec, dec_costs)
    sel_dec.select(cands_dec, temperature=1.0, score_bias=torch.zeros(4))
    n_dec = sel_dec.last_score_diagnostics["f_eligibility_envelope_size"]

    assert n_near > n_dec
    assert n_dec == 1  # a single clear F-winner -> envelope of one


# --------------------------------------------------------------------------- #
# Non-degeneracy: the envelope actually excludes                              #
# --------------------------------------------------------------------------- #


def test_envelope_excludes_on_divergent_pool():
    sel = _selector(use_f_eligibility_demotion=True)
    candidates = [_candidate(i) for i in range(4)]
    _patch_raw(sel, candidates, [0.0, 1.0, 1.0, 1.0])  # decisive -> excludes 3
    sel.select(candidates, temperature=1.0, score_bias=torch.zeros(4))
    d = sel.last_score_diagnostics
    assert d["f_eligibility_demotion_active"] is True
    assert d["f_eligibility_excluded_count"] > 0


# --------------------------------------------------------------------------- #
# Safety: a clearly-harmful candidate is never selectable                     #
# --------------------------------------------------------------------------- #


def test_safety_harmful_outlier_never_selected():
    """A clearly-harmful candidate (huge F cost) is excluded by the envelope and
    cannot be selected even under an overwhelming modulatory pull toward it (no
    global disinhibition)."""
    sel = _selector(use_f_eligibility_demotion=True)
    candidates = [_candidate(i) for i in range(4)]
    _patch_raw(sel, candidates, [0.0, 0.1, 0.2, 5.0])  # idx3 clearly harmful
    bias = torch.tensor([0.0, 0.0, 0.0, -100.0])  # overwhelming pull toward idx3
    r = sel.select(candidates, temperature=1.0, score_bias=bias)
    assert r.selected_index != 3
    d = sel.last_score_diagnostics
    assert d["f_eligibility_excluded_count"] >= 1  # idx3 excluded


# --------------------------------------------------------------------------- #
# F removed from the within-eligible argmin                                   #
# --------------------------------------------------------------------------- #


def test_f_removed_from_argmin_winner_is_modulatory():
    """Within the eligible set the committed winner is the modulatory argmin, NOT
    the F-argmin -- F is removed at commit (the rank-altering divergence)."""
    sel = _selector(use_f_eligibility_demotion=True)
    candidates = [_candidate(i) for i in range(4)]
    # idx3 harmful (excluded); within {0,1,2} modulation favours idx2 (F-worst eligible)
    _patch_raw(sel, candidates, [0.0, 0.1, 0.2, 3.0])
    bias = torch.tensor([0.0, 0.0, -5.0, 0.0])
    r = sel.select(candidates, temperature=1.0, score_bias=bias)
    assert r.selected_index == 2  # modulatory winner within the envelope
    d = sel.last_score_diagnostics
    assert d["f_eligibility_winner_neq_f_argmin"] is True  # F-argmin is 0
    assert d["f_eligibility_demotion_active"] is True


# --------------------------------------------------------------------------- #
# Rank-preserving: eligible set is an F-rank prefix                           #
# --------------------------------------------------------------------------- #


def test_rank_preserving_eligible_is_f_prefix():
    sel = _selector(use_f_eligibility_demotion=True)
    candidates = [_candidate(i) for i in range(4)]
    _patch_raw(sel, candidates, [0.0, 0.1, 0.2, 3.0])
    sel.select(candidates, temperature=1.0, score_bias=torch.zeros(4))
    d = sel.last_score_diagnostics
    assert d["f_eligibility_rank_preserving"] is True
    # eligible {0,1,2}; idx3 (highest cost) excluded -> a strict F-rank prefix
    assert d["f_eligibility_envelope_size"] == 3
    assert d["f_eligibility_excluded_count"] == 1


# --------------------------------------------------------------------------- #
# Requires a modulatory channel                                               #
# --------------------------------------------------------------------------- #


def test_requires_modulatory_channel_no_op_without_bias():
    """With the lever ON but no modulatory channel (_modulatory_accum None), the
    block is skipped -> legacy F argmin, demotion inactive (nothing to demote
    F to)."""
    sel = _selector(use_f_eligibility_demotion=True)
    candidates = [_candidate(i) for i in range(4)]
    _patch_raw(sel, candidates, [0.3, 0.0, 0.2, 0.1])  # F-argmin = idx1
    r = sel.select(candidates, temperature=1.0)  # no score_bias
    assert r.selected_index == 1  # legacy argmin(F)
    assert sel.last_score_diagnostics["f_eligibility_demotion_active"] is False


# --------------------------------------------------------------------------- #
# Exact N-way tie -> wide fallback (non-degeneracy = false)                   #
# --------------------------------------------------------------------------- #


def test_exact_tie_wide_fallback_non_degenerate_false():
    sel = _selector(use_f_eligibility_demotion=True)
    candidates = [_candidate(i) for i in range(4)]
    _patch_raw(sel, candidates, [0.5, 0.5, 0.5, 0.5])  # exact tie -> all eligible
    sel.select(candidates, temperature=1.0, score_bias=torch.zeros(4))
    d = sel.last_score_diagnostics
    assert d["f_eligibility_demotion_active"] is True
    assert d["f_eligibility_envelope_size"] == 4
    assert d["f_eligibility_excluded_count"] == 0  # the vacuity signal
    assert d["f_eligibility_rank_preserving"] is True
