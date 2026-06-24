"""Contracts for MECH-451: finer-channel-granularity selection-gating.

The cheap V3 rung BETWEEN ARC-108's single global w_chan over the pre-COMPRESSED
score_bias blend and ARC-110's full V4 segregated loops. When
use_finer_channel_gating is on, the single ARC-108 "score_bias" learned-gating
channel is EXPLODED into its constituents (ofc / dacc / lpfc / vigour / liking /
gated_policy + a "residual" lump of everything else summed into score_bias), each
with its OWN learned w_chan_finer entry trained by the SAME ARC-108 signed-RPE
three-factor rule. ONE shared arena (NOT ARC-110 per-loop competition). Tests
whether the F-dominance conversion ceiling (MECH-439) is REPRESENTATIONAL
COMPRESSION rather than a need for full per-loop competition.

Coverage:
  - C1 config default no-op; from_dims surfaces the flag onto config.e3;
        w_chan_finer initialises to softplus-unity at the finer registry size.
  - C2 bit-identical: finer-ON-at-init produces the EXACT same scores + selection
        as the legacy single-channel path (softplus(w_init)==1.0, named + residual
        sum to score_bias exactly), and the ARC-108 w_chan is untouched.
  - C3 activation: under a non-flat delta_t w_chan_finer MOVES from init when ON;
        the ARC-108 w_chan stays at init (finer is a PARALLEL buffer -- the
        V3-frozen ARC-108 path is never written by the finer mode).
  - C4 waking-only gate (MECH-094): a simulation_mode select records no finer
        eligibility, so the following post_action_update writes no w_chan_finer.
  - C5 residual exhaustiveness: only SOME named channels supplied -> the unsupplied
        contribution lands in "residual", so Sum(finer)==score_bias and the
        recompose at init still reproduces the summed bias EXACTLY.
  - C6 cross-channel dissociation (EXP-0391 non-degeneracy): per-channel biases
        that earn DIFFERENT eligibility drive the finer weights APART (range > 0);
        identical-eligibility channels move together (the degenerate
        re-labelled-blend the noise guard must catch).
  - C7 envelope intact / safety: the F-bounded MECH-448 eligible set is
        w_chan_finer-invariant -- a learned finer weight can never re-admit an
        F-excluded (No-Go-suppressed) candidate.
"""

from __future__ import annotations

import math

import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import (
    E3Config,
    E3TrajectorySelector,
    _LCG_W_INIT,
    _FCG_CHANNEL_NAMES,
    FINER_NAMED_CHANNELS,
)
from ree_core.utils.config import REEConfig


WORLD_DIM = 6


def _candidate(action_class: int, action_dim: int = 8) -> Trajectory:
    horizon = 3
    states = [torch.zeros(1, WORLD_DIM) for _ in range(horizon + 1)]
    world_states = [torch.zeros(1, WORLD_DIM) for _ in range(horizon + 1)]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class % action_dim] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _selector(**e3_kwargs) -> E3TrajectorySelector:
    sel = E3TrajectorySelector(E3Config(world_dim=WORLD_DIM, hidden_dim=8, **e3_kwargs))
    sel._running_variance = 0.0  # force the committed (deterministic) path
    return sel


def _patch_raw(selector, candidates, raw_costs):
    raw_map = {id(c): torch.tensor([float(v)]) for c, v in zip(candidates, raw_costs)}
    selector.score_trajectory = lambda cand, **kw: raw_map[id(cand)]


class _ConstHead(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, zw: torch.Tensor) -> torch.Tensor:
        return torch.full((zw.shape[0], 1), self.value)


def _patch_heads(selector, benefit: float, harm: float):
    selector.benefit_eval_head = _ConstHead(benefit)
    selector.harm_eval_head = _ConstHead(harm)


_N_FINER = len(_FCG_CHANNEL_NAMES)


# --------------------------------------------------------------------------- #
# C1 config / wiring                                                           #
# --------------------------------------------------------------------------- #

def test_c1_config_default_is_noop():
    cfg = E3Config(world_dim=WORLD_DIM, hidden_dim=8)
    assert cfg.use_finer_channel_gating is False


def test_c1_from_dims_surfaces_flag_onto_e3():
    cfg = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4,
        self_dim=16, world_dim=16,
        use_finer_channel_gating=True,
    )
    assert cfg.e3.use_finer_channel_gating is True
    base = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4,
        self_dim=16, world_dim=16,
    )
    assert base.e3.use_finer_channel_gating is False


def test_c1_w_chan_finer_initialises_to_softplus_unity():
    sel = _selector(use_finer_channel_gating=True)
    assert sel.w_chan_finer.shape == (_N_FINER,)
    sp = torch.nn.functional.softplus(sel.w_chan_finer)
    assert torch.allclose(sp, torch.ones(_N_FINER))
    # registry: 6 named + residual + mech341 + route
    assert FINER_NAMED_CHANNELS == ("ofc", "dacc", "lpfc", "vigour", "liking", "gated_policy")
    assert _FCG_CHANNEL_NAMES[len(FINER_NAMED_CHANNELS)] == "residual"
    assert "mech341" in _FCG_CHANNEL_NAMES and "route" in _FCG_CHANNEL_NAMES


# --------------------------------------------------------------------------- #
# C2 bit-identical OFF vs finer-ON-at-init                                     #
# --------------------------------------------------------------------------- #

def _finer_channels():
    # Named per-head biases (4 candidates); the rest of score_bias lands in residual.
    return {
        "ofc": torch.tensor([0.0, -0.2, 0.1, 0.0]),
        "dacc": torch.tensor([0.0, -0.1, 0.0, -0.1]),
        "lpfc": torch.tensor([0.0, 0.0, 0.1, 0.0]),
    }


def _finer_channels3():
    # 3-candidate variant for the K=3 tests.
    return {
        "ofc": torch.tensor([0.0, -0.2, 0.1]),
        "dacc": torch.tensor([0.0, -0.1, 0.0]),
        "lpfc": torch.tensor([0.0, 0.0, 0.1]),
    }


def test_c2_finer_on_at_init_is_bit_identical_to_legacy():
    # Authority ON so the recomposed _modulatory_accum actually reaches scores.
    score_bias = torch.tensor([0.0, -0.3, 0.2, -0.1])
    raw = [0.0, 0.4, 0.9, 0.2]
    cands = [_candidate(i) for i in range(4)]

    sel_off = _selector(use_finer_channel_gating=False,
                        use_modulatory_selection_authority=True)
    sel_on = _selector(use_finer_channel_gating=True,
                       use_modulatory_selection_authority=True)
    _patch_raw(sel_off, cands, raw)
    _patch_raw(sel_on, cands, raw)

    r_off = sel_off.select(cands, score_bias=score_bias.clone())
    r_on = sel_on.select(
        cands, score_bias=score_bias.clone(),
        score_bias_channels=_finer_channels(),
    )

    assert r_off.selected_index == r_on.selected_index
    assert torch.equal(r_off.scores, r_on.scores)  # EXACT, not approximate
    assert torch.equal(sel_on.w_chan_finer, torch.full((_N_FINER,), _LCG_W_INIT))


def test_c2_finer_on_does_not_touch_arc108_w_chan():
    cands = [_candidate(i) for i in range(3)]
    sel = _selector(use_finer_channel_gating=True)
    _patch_raw(sel, cands, [0.0, 0.5, 1.0])
    sel.select(cands, score_bias=torch.tensor([0.0, -0.5, 0.3]),
               score_bias_channels=_finer_channels3())
    sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
    # the ARC-108 buffer is parallel + untouched by the finer path
    assert torch.equal(sel.w_chan, torch.full((3,), _LCG_W_INIT))
    assert sel._lcg_pending is False


# --------------------------------------------------------------------------- #
# C3 activation: w_chan_finer moves; w_chan stays                             #
# --------------------------------------------------------------------------- #

def test_c3_w_chan_finer_moves_under_nonflat_delta():
    cands = [_candidate(i) for i in range(3)]
    sel = _selector(use_finer_channel_gating=True, learned_channel_gating_eta=0.5)
    _patch_raw(sel, cands, [0.0, 0.5, 1.0])
    _patch_heads(sel, benefit=1.0, harm=0.0)  # R_t = +1 -> non-zero delta_t

    # The committed argmin here is candidate 0; its per-channel biases must be
    # non-zero so it earns eligibility (eligibility_c = |channel_bias_c[selected]|).
    chans = {
        "ofc": torch.tensor([-0.3, -0.1, 0.1]),
        "dacc": torch.tensor([-0.1, 0.0, 0.0]),
        "lpfc": torch.tensor([-0.2, 0.0, 0.1]),
    }
    before = sel.w_chan_finer.clone()
    for _ in range(5):
        sel.select(cands, score_bias=torch.tensor([-0.6, -0.1, 0.2]),
                   score_bias_channels=chans)
        sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
    assert not torch.equal(sel.w_chan_finer, before)
    # ARC-108 global buffer never moves under the finer path
    assert torch.equal(sel.w_chan, torch.full((3,), _LCG_W_INIT))


def test_c3_off_w_chan_finer_stays_at_init():
    cands = [_candidate(i) for i in range(3)]
    sel = _selector(use_finer_channel_gating=False)
    _patch_raw(sel, cands, [0.0, 0.5, 1.0])
    _patch_heads(sel, benefit=1.0, harm=0.0)
    sel.select(cands, score_bias=torch.tensor([0.0, -0.4, 0.2]))
    sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
    assert torch.equal(sel.w_chan_finer, torch.full((_N_FINER,), _LCG_W_INIT))


# --------------------------------------------------------------------------- #
# C4 waking-only gate (MECH-094)                                              #
# --------------------------------------------------------------------------- #

def test_c4_simulation_mode_writes_no_w_chan_finer():
    cands = [_candidate(i) for i in range(3)]
    sel = _selector(use_finer_channel_gating=True, learned_channel_gating_eta=0.5)
    _patch_raw(sel, cands, [0.0, 0.5, 1.0])
    _patch_heads(sel, benefit=1.0, harm=0.0)
    sel.select(cands, score_bias=torch.tensor([0.0, -0.4, 0.2]),
               score_bias_channels=_finer_channels3(), simulation_mode=True)
    assert sel._fcg_pending is False
    sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
    assert torch.equal(sel.w_chan_finer, torch.full((_N_FINER,), _LCG_W_INIT))


# --------------------------------------------------------------------------- #
# C5 residual exhaustiveness                                                   #
# --------------------------------------------------------------------------- #

def test_c5_residual_absorbs_unsupplied_contribution():
    # score_bias carries MORE range than the named channels sum to; "residual"
    # absorbs the difference so the recompose at init still reproduces score_bias.
    score_bias = torch.tensor([0.0, -0.6, 0.5, -0.2])
    partial = {"ofc": torch.tensor([0.0, -0.1, 0.1, 0.0])}  # only one named channel
    raw = [0.0, 0.4, 0.9, 0.2]
    cands = [_candidate(i) for i in range(4)]

    sel_off = _selector(use_finer_channel_gating=False,
                        use_modulatory_selection_authority=True)
    sel_on = _selector(use_finer_channel_gating=True,
                       use_modulatory_selection_authority=True)
    _patch_raw(sel_off, cands, raw)
    _patch_raw(sel_on, cands, raw)

    r_off = sel_off.select(cands, score_bias=score_bias.clone())
    r_on = sel_on.select(cands, score_bias=score_bias.clone(),
                         score_bias_channels=partial)
    assert torch.equal(r_off.scores, r_on.scores)


# --------------------------------------------------------------------------- #
# C6 cross-channel dissociation (EXP-0391 non-degeneracy)                      #
# --------------------------------------------------------------------------- #

def test_c6_dissociable_channels_drive_weights_apart():
    cands = [_candidate(i) for i in range(3)]
    sel = _selector(use_finer_channel_gating=True, learned_channel_gating_eta=0.5)
    _patch_raw(sel, cands, [0.0, 0.5, 1.0])
    _patch_heads(sel, benefit=1.0, harm=0.0)
    # DIFFERENT per-channel biases -> different |bias[selected]| -> different
    # eligibility -> the finer weights must move APART (range > 0).
    diss = {
        "ofc": torch.tensor([0.0, -0.6, 0.3]),     # large magnitude
        "dacc": torch.tensor([0.0, -0.05, 0.02]),  # small magnitude
        "lpfc": torch.tensor([0.0, 0.0, 0.0]),     # silent (no eligibility)
    }
    for _ in range(5):
        sel.select(cands, score_bias=torch.tensor([0.0, -0.65, 0.32]),
                   score_bias_channels=diss)
        sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
    rng = float((sel.w_chan_finer.max() - sel.w_chan_finer.min()).item())
    assert rng > 1e-4  # channels dissociated, NOT the re-labelled blend


def test_c6_identical_channels_move_together():
    cands = [_candidate(i) for i in range(3)]
    sel = _selector(use_finer_channel_gating=True, learned_channel_gating_eta=0.5)
    _patch_raw(sel, cands, [0.0, 0.5, 1.0])
    _patch_heads(sel, benefit=1.0, harm=0.0)
    same = torch.tensor([0.0, -0.3, 0.15])
    identical = {"ofc": same.clone(), "dacc": same.clone(), "lpfc": same.clone()}
    for _ in range(5):
        sel.select(cands, score_bias=3.0 * same,
                   score_bias_channels=identical)
        sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
    # the three named channels earned IDENTICAL eligibility -> identical weight
    idx = {n: i for i, n in enumerate(_FCG_CHANNEL_NAMES)}
    w = sel.w_chan_finer
    assert torch.allclose(w[idx["ofc"]], w[idx["dacc"]], atol=1e-6)
    assert torch.allclose(w[idx["ofc"]], w[idx["lpfc"]], atol=1e-6)


# --------------------------------------------------------------------------- #
# C7 envelope intact / safety                                                  #
# --------------------------------------------------------------------------- #

def test_c7_finer_weight_cannot_readmit_excluded_candidate():
    # MECH-448 demotion ON: candidate 2 has a large F-cost (excluded from the
    # eligible set). Even a finer channel overwhelmingly favouring it cannot make
    # it the committed action -- the envelope is w_chan_finer-invariant.
    cands = [_candidate(i) for i in range(3)]
    sel = _selector(
        use_finer_channel_gating=True,
        use_modulatory_selection_authority=True,
        use_f_eligibility_demotion=True,
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=2,
    )
    _patch_raw(sel, cands, [0.0, 0.1, 10.0])  # cand 2 clearly worst on F
    # finer channel screams for the excluded candidate 2
    chans = {"ofc": torch.tensor([0.0, 0.0, -100.0])}
    # hand-train the ofc weight way up so the recompose strongly favours cand 2
    sel.w_chan_finer.data[0] = 5.0  # ofc index 0
    r = sel.select(cands, score_bias=torch.tensor([0.0, 0.0, -100.0]),
                   score_bias_channels=chans)
    assert r.selected_index != 2
