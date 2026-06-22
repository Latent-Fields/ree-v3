"""Contracts for ARC-108 JOB-1 step-1: learned dopamine-gated E3 gating.

The minimal V3 learned-gating mechanism (dopamine_into_gating design 2026-06-22
secs 2-4): a single LEARNED per-channel selection-weight vector w_chan over the
modulatory channels that feed the E3 selector's _modulatory_accum, composed as
_modulatory_accum = sum_c softplus(w_chan[c]) * channel_bias_c. At init
softplus(w_chan[c]) == 1.0 so the accumulator is bit-identical to the unweighted
sum. w_chan is updated by a three-factor rule driven by a SIGNED dopaminergic-RPE
analog delta_t = R_t - V-hat_t (R_t = benefit_eval - harm_eval at the realised
state from the already-trained valuation heads), with a D1-LTP/D2-LTD-analog
asymmetric gain.

Coverage:
  - C1 config defaults are no-op; from_dims surfaces flags onto config.e3.
  - C2 OFF bit-identical: flag-OFF leaves w_chan at init and writes nothing; and
        flag-ON-at-init produces the EXACT same scores + selection as OFF
        (softplus(w_init)==1.0, 1.0*x==x, matching add order).
  - C3 activation: under a non-flat delta_t sequence w_chan MOVES from init when
        ON; stays at init when OFF.
  - C4 waking-only gate (MECH-094): a simulation_mode select records no
        eligibility, so the following post_action_update writes no w_chan.
  - C5 signed-RPE is load-bearing (divergence B5): a positive delta_t POTENTIATES
        the voting channel while a negative delta_t DEPRESSES it -- opposite-sign
        w_chan changes that an unsigned-magnitude substitution cannot produce.
  - C6 envelope intact / safety: the F-bounded MECH-448 eligible set is
        w_chan-invariant, so a learned weight can never re-admit an F-excluded
        (No-Go-suppressed) candidate however strongly it favours that channel.
  - C7 unsigned-RPE ablation (ARC-108 sec-7 C3 / divergence B5): with
        learned_channel_rpe_mode="unsigned" the teaching signal is the always->=0
        ARC-016 prediction-error magnitude, so a good outcome and a bad outcome
        move w_chan in the SAME direction -- the unsigned mode CANNOT produce the
        opposite-sign moves C5 shows for the signed delta_t (the structural reason
        unsigned must fail to convert directed credit).
"""

from __future__ import annotations

import math

import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import (
    E3Config,
    E3TrajectorySelector,
    _LCG_W_INIT,
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
    """A valuation head stub returning a constant value (heads are nn.Module
    children, so a stub must itself be a Module)."""

    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, zw: torch.Tensor) -> torch.Tensor:
        return torch.full((zw.shape[0], 1), self.value)


def _patch_heads(selector, benefit: float, harm: float):
    """Force a known realised outcome valence R_t = benefit - harm."""
    selector.benefit_eval_head = _ConstHead(benefit)
    selector.harm_eval_head = _ConstHead(harm)


# --------------------------------------------------------------------------- #
# C1 config defaults / wiring                                                  #
# --------------------------------------------------------------------------- #

def test_c1_config_defaults_are_noop():
    cfg = E3Config(world_dim=WORLD_DIM, hidden_dim=8)
    assert cfg.use_learned_channel_gating is False
    assert cfg.learned_channel_gating_eta == 0.01
    assert cfg.learned_channel_gating_elig_decay == 0.9
    assert cfg.learned_channel_value_baseline_beta == 0.05
    assert cfg.learned_channel_asym_potentiation == 1.0
    assert cfg.learned_channel_asym_depression == 0.5
    assert cfg.learned_channel_rpe_mode == "signed"


def test_c1_from_dims_surfaces_flags_onto_e3():
    cfg = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4,
        self_dim=16, world_dim=16,
        use_learned_channel_gating=True,
        learned_channel_gating_eta=0.05,
        learned_channel_asym_depression=0.25,
    )
    assert cfg.e3.use_learned_channel_gating is True
    assert cfg.e3.learned_channel_gating_eta == 0.05
    assert cfg.e3.learned_channel_asym_depression == 0.25
    # Default from_dims leaves the lever OFF.
    base = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4,
        self_dim=16, world_dim=16,
    )
    assert base.e3.use_learned_channel_gating is False


def test_c1_w_chan_initialises_to_softplus_unity():
    sel = _selector(use_learned_channel_gating=True)
    assert sel.w_chan.shape == (3,)
    sp = torch.nn.functional.softplus(sel.w_chan)
    assert torch.allclose(sp, torch.ones(3))
    # init value is exactly ln(e - 1)
    assert abs(_LCG_W_INIT - math.log(math.e - 1.0)) < 1e-12


# --------------------------------------------------------------------------- #
# C2 bit-identical OFF (and ON-at-init)                                        #
# --------------------------------------------------------------------------- #

def test_c2_off_is_bit_identical_to_on_at_init():
    # Authority ON so _modulatory_accum reaches scores; a score_bias channel is
    # present so the learned recompose has a channel to weight.
    score_bias = torch.tensor([0.0, -0.3, 0.2, -0.1])
    raw = [0.0, 0.4, 0.9, 0.2]

    sel_off = _selector(use_learned_channel_gating=False,
                        use_modulatory_selection_authority=True)
    sel_on = _selector(use_learned_channel_gating=True,
                       use_modulatory_selection_authority=True)
    cands = [_candidate(i) for i in range(4)]
    _patch_raw(sel_off, cands, raw)
    _patch_raw(sel_on, cands, raw)

    r_off = sel_off.select(cands, score_bias=score_bias.clone())
    r_on = sel_on.select(cands, score_bias=score_bias.clone())

    assert r_off.selected_index == r_on.selected_index
    assert torch.equal(r_off.scores, r_on.scores)  # EXACT, not approximate
    # OFF leaves w_chan at init and never arms an update.
    assert torch.equal(sel_off.w_chan, torch.full((3,), _LCG_W_INIT))
    assert sel_off._lcg_pending is False


def test_c2_off_writes_no_w_chan_on_post_action():
    sel = _selector(use_learned_channel_gating=False)
    cands = [_candidate(i) for i in range(3)]
    _patch_raw(sel, cands, [0.0, 0.5, 1.0])
    sel.select(cands, score_bias=torch.tensor([0.0, -0.5, 0.3]))
    sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
    assert torch.equal(sel.w_chan, torch.full((3,), _LCG_W_INIT))


# --------------------------------------------------------------------------- #
# C3 activation: w_chan moves under a non-flat delta_t when ON                 #
# --------------------------------------------------------------------------- #

def test_c3_w_chan_moves_when_on_and_stays_when_off():
    cands = [_candidate(i) for i in range(3)]

    def _run(flag, n=6):
        sel = _selector(use_learned_channel_gating=flag,
                        learned_channel_gating_eta=0.1)
        _patch_raw(sel, cands, [0.0, 0.5, 1.0])
        for t in range(n):
            # active score_bias channel (non-zero at the selected candidate)
            sel.select(cands, score_bias=torch.tensor([-1.0, 0.2, 0.3]))
            # non-flat realised outcome: alternate good / bad states
            benefit, harm = (0.9, 0.1) if t % 2 == 0 else (0.1, 0.7)
            _patch_heads(sel, benefit, harm)
            sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
        return sel.w_chan.clone()

    w_on = _run(True)
    w_off = _run(False)
    init = torch.full((3,), _LCG_W_INIT)
    assert not torch.allclose(w_on, init)   # learning moved w_chan
    assert torch.equal(w_off, init)         # OFF never moves


# --------------------------------------------------------------------------- #
# C4 waking-only gate (MECH-094)                                              #
# --------------------------------------------------------------------------- #

def test_c4_simulation_tick_writes_no_w_chan():
    sel = _selector(use_learned_channel_gating=True, learned_channel_gating_eta=0.1)
    cands = [_candidate(i) for i in range(3)]
    _patch_raw(sel, cands, [0.0, 0.5, 1.0])
    _patch_heads(sel, 0.9, 0.1)  # would give a large positive delta_t
    # A replay/DMN tick: records no eligibility -> forms no delta_t.
    sel.select(cands, score_bias=torch.tensor([-1.0, 0.2, 0.3]), simulation_mode=True)
    assert sel._lcg_pending is False
    sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
    assert torch.equal(sel.w_chan, torch.full((3,), _LCG_W_INIT))
    assert sel._lcg_n_updates == 0


# --------------------------------------------------------------------------- #
# C5 signed-RPE is load-bearing (divergence B5)                                #
# --------------------------------------------------------------------------- #

def test_c5_signed_rpe_potentiates_vs_depresses():
    cands = [_candidate(i) for i in range(3)]
    bias = torch.tensor([-1.0, 0.2, 0.3])  # channel 0 (score_bias) votes for cand 0

    def _one(benefit, harm):
        sel = _selector(use_learned_channel_gating=True,
                        learned_channel_gating_eta=0.1,
                        learned_channel_value_baseline_beta=0.0)  # hold V-hat at 0
        _patch_raw(sel, cands, [0.0, 0.5, 1.0])
        sel.select(cands, score_bias=bias.clone())
        _patch_heads(sel, benefit, harm)
        sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
        return float(sel.w_chan[0] - _LCG_W_INIT)  # change on the voting channel

    dw_pos = _one(0.9, 0.1)   # R_t = +0.8 > baseline 0 -> delta_t > 0 -> potentiate
    dw_neg = _one(0.1, 0.9)   # R_t = -0.8 < baseline 0 -> delta_t < 0 -> depress
    assert dw_pos > 0.0       # positive RPE raises the voting channel's weight
    assert dw_neg < 0.0       # negative RPE LOWERS it (an unsigned |delta| could not)
    # opposite signs: the directional credit only a SIGNED delta_t supplies.
    assert dw_pos * dw_neg < 0.0


# --------------------------------------------------------------------------- #
# C6 envelope intact / safety: w_chan cannot re-admit an F-excluded candidate  #
# --------------------------------------------------------------------------- #

def test_c6_learned_weight_cannot_readmit_excluded_candidate():
    # cand 2 is F-excluded by the MECH-448 envelope (clearly-worse raw cost).
    # The modulatory channel strongly favours cand 2; even with w_chan maximally
    # amplifying that channel, cand 2 must never be selected (safety inherited
    # from the F-bounded eligible set; learning composes INSIDE it).
    cands = [_candidate(i) for i in range(3)]
    raw = [0.0, 0.1, 10.0]                       # cand 2 clearly worst F
    score_bias = torch.tensor([0.0, -1.0, -100.0])  # modulatory loves cand 2

    def _sel(w0):
        sel = _selector(use_learned_channel_gating=True,
                        use_f_eligibility_demotion=True,
                        use_modulatory_shortlist_then_modulate=True)
        _patch_raw(sel, cands, raw)
        with torch.no_grad():
            sel.w_chan[0] = float(w0)  # arbitrarily strong score_bias channel weight
        return sel.select(cands, score_bias=score_bias.clone())

    r_init = _sel(_LCG_W_INIT)
    r_huge = _sel(20.0)  # softplus(20) ~ 20: 20x amplified modulatory pull
    assert r_init.selected_index != 2
    assert r_huge.selected_index != 2  # excluded candidate never re-admitted


# --------------------------------------------------------------------------- #
# C7 unsigned-RPE ablation (ARC-108 sec-7 C3 / divergence B5)                  #
# --------------------------------------------------------------------------- #

def test_c7_unsigned_rpe_cannot_potentiate_vs_depress():
    # Mirror of C5, but with learned_channel_rpe_mode="unsigned": the teaching
    # signal is the always->=0 ARC-016 prediction-error magnitude
    # (e3._running_variance), so a good outcome and a bad outcome move the voting
    # channel's weight in the SAME direction. The opposite-sign credit C5 shows
    # for the signed delta_t is structurally impossible under the unsigned mode.
    cands = [_candidate(i) for i in range(3)]
    bias = torch.tensor([-1.0, 0.2, 0.3])  # channel 0 (score_bias) votes for cand 0

    def _one(benefit, harm):
        sel = _selector(use_learned_channel_gating=True,
                        learned_channel_gating_eta=0.1,
                        learned_channel_value_baseline_beta=0.0,  # hold V-hat at 0
                        learned_channel_rpe_mode="unsigned")
        _patch_raw(sel, cands, [0.0, 0.5, 1.0])
        sel.select(cands, score_bias=bias.clone())  # committed (rv pinned to 0)
        _patch_heads(sel, benefit, harm)
        # A non-zero realised state drives a positive ARC-016 running variance
        # (the unsigned teaching signal). R_t is fixed by the constant heads, so
        # it is independent of this state -- only the unsigned magnitude varies.
        sel.post_action_update(torch.full((1, WORLD_DIM), 0.5), harm_occurred=False)
        return float(sel.w_chan[0] - _LCG_W_INIT)  # change on the voting channel

    dw_pos = _one(0.9, 0.1)   # R_t = +0.8 (a signed delta_t would POTENTIATE)
    dw_neg = _one(0.1, 0.9)   # R_t = -0.8 (a signed delta_t would DEPRESS)
    # Both move the same way: the unsigned magnitude carries no sign.
    assert dw_pos != 0.0      # the unsigned signal is non-zero (rv > 0)
    assert dw_neg != 0.0
    assert dw_pos * dw_neg > 0.0   # SAME sign -- C5's signed mode gives < 0
    # No directional credit: the realised-outcome sign did not flip the move.
    assert (dw_pos > 0.0) == (dw_neg > 0.0)
