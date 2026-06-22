"""Contracts for MECH-450 (ARC-108 JOB-1 step-2): learned recurrent-settling step.

The SECOND factor of the learned-gating 2x2 (dopamine_into_gating design 2026-06-22
sec 4), coupled to the JOB-1 w_chan step and sharing the SAME signed-RPE delta_t.
A bounded recurrent lateral-inhibition SETTLING competition over the F-bounded
eligible set runs BEFORE the within-eligible commit:

    accum = _modulatory_accum[eligible_idx]
    for r in range(R):
        a       = softmax(-accum / T)            # support over eligible candidates
        a_class = onehot.T @ a                   # per-action-class aggregated support
        accum   = accum + onehot @ (W_lat @ a_class)   # learned lateral inhibition
    commit = argmin(settled accum) / sample(softmax(-settled accum / T))

W_lat is a LEARNED [C, C] inhibition matrix over candidate first-action CLASSES
(a stable object; per-candidate [K, K] is impossible -- K varies). Init W_lat == 0
-> the settling step is a no-op -> bit-identical OFF and at init. W_lat is learned
by the SAME three-factor Hebbian x signed-RPE x D1/D2-asym rule as w_chan.

Coverage:
  - C1 config defaults are no-op; from_dims surfaces flags onto config.e3; W_lat
        initialises to a zero [C, C] buffer.
  - C2 OFF == ON-at-init: with the settling engaged (a shortlist eligible set
        present) but W_lat == 0, the selection is bit-identical to settling-OFF, and
        the settling pass is a no-op (round_delta == 0); OFF never arms a W_lat write.
  - C3 activation: under a non-flat delta_t sequence W_lat MOVES from init when ON
        (even with w_chan OFF -- the settling learns independently); stays at init
        when OFF.
  - C4 waking-only gate (MECH-094): a simulation_mode select records no settling
        trace, so the following post_action_update writes no W_lat.
  - C5 non-degeneracy / the settling actually MOVES the field: with a non-zero W_lat
        the settled accumulator differs from its input across rounds (round_delta > 0);
        at init (W_lat == 0) the pass is an exact no-op.
  - C6 envelope intact / safety: the F-bounded eligible set is W_lat-invariant, so an
        arbitrarily strong learned W_lat can never re-admit an F-excluded candidate.
  - C7 shared delta_t: with BOTH levers ON (the A3 arm of the 2x2) w_chan AND W_lat
        both move off init under one signed RPE.
"""

from __future__ import annotations

import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector
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


# A top_k shortlist engages the within-eligible arbitration (where the settling
# lives) with a predictable, >= 2-candidate eligible set + a modulatory channel.
_SHORTLIST = dict(
    use_modulatory_shortlist_then_modulate=True,
    modulatory_shortlist_mode="top_k",
    modulatory_shortlist_k=3,
)


# --------------------------------------------------------------------------- #
# C1 config defaults / wiring                                                  #
# --------------------------------------------------------------------------- #

def test_c1_config_defaults_are_noop():
    cfg = E3Config(world_dim=WORLD_DIM, hidden_dim=8)
    assert cfg.use_learned_settling_step is False
    assert cfg.learned_settling_rounds == 3
    assert cfg.learned_settling_temperature == 1.0
    assert cfg.learned_settling_eta == 0.01
    assert cfg.learned_settling_elig_decay == 0.9
    assert cfg.learned_settling_n_action_classes == 8


def test_c1_from_dims_surfaces_flags_onto_e3():
    cfg = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4,
        self_dim=16, world_dim=16,
        use_learned_settling_step=True,
        learned_settling_rounds=4,
        learned_settling_eta=0.05,
    )
    assert cfg.e3.use_learned_settling_step is True
    assert cfg.e3.learned_settling_rounds == 4
    assert cfg.e3.learned_settling_eta == 0.05
    base = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4,
        self_dim=16, world_dim=16,
    )
    assert base.e3.use_learned_settling_step is False


def test_c1_w_lat_initialises_to_zero_square_buffer():
    sel = _selector(use_learned_settling_step=True,
                    learned_settling_n_action_classes=5)
    assert sel.W_lat.shape == (5, 5)
    assert torch.count_nonzero(sel.W_lat).item() == 0
    assert sel._wlat_pending is False


# --------------------------------------------------------------------------- #
# C2 bit-identical OFF (and ON-at-init, W_lat == 0)                            #
# --------------------------------------------------------------------------- #

def test_c2_off_is_bit_identical_to_on_at_init():
    score_bias = torch.tensor([0.0, -0.3, 0.2])
    raw = [0.0, 0.4, 0.9]
    cands = [_candidate(i) for i in range(3)]

    sel_off = _selector(use_learned_settling_step=False, **_SHORTLIST)
    sel_on = _selector(use_learned_settling_step=True, **_SHORTLIST)
    _patch_raw(sel_off, cands, raw)
    _patch_raw(sel_on, cands, raw)

    r_off = sel_off.select(cands, score_bias=score_bias.clone())
    r_on = sel_on.select(cands, score_bias=score_bias.clone())

    assert r_off.selected_index == r_on.selected_index
    assert torch.equal(r_off.scores, r_on.scores)   # settling never touches `scores`
    # ON-at-init: the settling ran but was a no-op (W_lat == 0).
    assert sel_on.last_score_diagnostics["learned_settling_active"] is True
    assert sel_on.last_score_diagnostics["learned_settling_round_delta"] == 0.0
    # OFF never engages the settling and never arms a W_lat write.
    assert sel_off.last_score_diagnostics["learned_settling_active"] is False
    assert sel_off._wlat_pending is False
    assert torch.count_nonzero(sel_off.W_lat).item() == 0


def test_c2_off_writes_no_w_lat_on_post_action():
    sel = _selector(use_learned_settling_step=False, **_SHORTLIST)
    cands = [_candidate(i) for i in range(3)]
    _patch_raw(sel, cands, [0.0, 0.5, 1.0])
    sel.select(cands, score_bias=torch.tensor([0.0, -0.5, 0.3]))
    sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
    assert torch.count_nonzero(sel.W_lat).item() == 0


# --------------------------------------------------------------------------- #
# C3 activation: W_lat moves under a non-flat delta_t when ON (w_chan OFF)     #
# --------------------------------------------------------------------------- #

def test_c3_w_lat_moves_when_on_and_stays_when_off():
    cands = [_candidate(i) for i in range(3)]

    def _run(flag, n=6):
        sel = _selector(use_learned_settling_step=flag,
                        learned_settling_eta=0.2,
                        # w_chan OFF: the settling learns its W_lat independently.
                        use_learned_channel_gating=False,
                        **_SHORTLIST)
        _patch_raw(sel, cands, [0.0, 0.2, 0.4])  # near-tie -> top-3 all eligible
        for t in range(n):
            sel.select(cands, score_bias=torch.tensor([-1.0, 0.2, 0.3]))
            benefit, harm = (0.9, 0.1) if t % 2 == 0 else (0.1, 0.7)
            _patch_heads(sel, benefit, harm)
            sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
        return sel.W_lat.clone()

    w_on = _run(True)
    w_off = _run(False)
    assert torch.count_nonzero(w_on).item() > 0   # learning moved W_lat off zero
    assert torch.count_nonzero(w_off).item() == 0  # OFF never moves


# --------------------------------------------------------------------------- #
# C4 waking-only gate (MECH-094)                                              #
# --------------------------------------------------------------------------- #

def test_c4_simulation_tick_writes_no_w_lat():
    sel = _selector(use_learned_settling_step=True, learned_settling_eta=0.2,
                    **_SHORTLIST)
    cands = [_candidate(i) for i in range(3)]
    _patch_raw(sel, cands, [0.0, 0.2, 0.4])
    _patch_heads(sel, 0.9, 0.1)  # would give a large positive delta_t
    # A replay/DMN tick: no settling, records no trace -> forms no delta_t.
    sel.select(cands, score_bias=torch.tensor([-1.0, 0.2, 0.3]), simulation_mode=True)
    assert sel._wlat_pending is False
    sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
    assert torch.count_nonzero(sel.W_lat).item() == 0
    assert sel._wlat_n_updates == 0


# --------------------------------------------------------------------------- #
# C5 non-degeneracy: the settling actually MOVES _modulatory_accum            #
# --------------------------------------------------------------------------- #

def test_c5_settling_moves_field_when_w_lat_nonzero_noop_at_init():
    cands = [_candidate(i) for i in range(3)]
    raw = [0.0, 0.2, 0.4]
    bias = torch.tensor([-0.2, 0.1, 0.3])

    # At init (W_lat == 0): the settling pass is an exact no-op.
    sel0 = _selector(use_learned_settling_step=True, **_SHORTLIST)
    _patch_raw(sel0, cands, raw)
    sel0.select(cands, score_bias=bias.clone())
    assert sel0._wlat_last_settle_delta == 0.0

    # With a non-zero W_lat (an inhibition matrix between distinct classes): the
    # settled accumulator differs from its input -> live cross-round movement.
    sel1 = _selector(use_learned_settling_step=True, **_SHORTLIST)
    _patch_raw(sel1, cands, raw)
    with torch.no_grad():
        sel1.W_lat.fill_(0.0)
        # off-diagonal inhibition between the 3 used action classes (0, 1, 2)
        for i in range(3):
            for j in range(3):
                if i != j:
                    sel1.W_lat[i, j] = 0.5
    sel1.select(cands, score_bias=bias.clone())
    assert sel1._wlat_last_settle_delta > 0.0
    assert sel1.last_score_diagnostics["learned_settling_round_delta"] > 0.0


# --------------------------------------------------------------------------- #
# C6 envelope intact / safety: W_lat cannot re-admit an F-excluded candidate   #
# --------------------------------------------------------------------------- #

def test_c6_settling_cannot_readmit_excluded_candidate():
    # cand 2 is F-excluded by the MECH-448 envelope (clearly-worse raw cost).
    # An arbitrarily strong learned W_lat settles only over the eligible subset, so
    # cand 2 is never touched and never selectable.
    cands = [_candidate(i) for i in range(3)]
    raw = [0.0, 0.1, 10.0]                        # cand 2 clearly worst F
    score_bias = torch.tensor([0.0, -1.0, -100.0])  # modulatory loves cand 2

    def _sel(fill):
        sel = _selector(use_learned_settling_step=True,
                        use_f_eligibility_demotion=True,
                        use_modulatory_shortlist_then_modulate=True)
        _patch_raw(sel, cands, raw)
        with torch.no_grad():
            sel.W_lat.fill_(float(fill))
        return sel.select(cands, score_bias=score_bias.clone())

    r_zero = _sel(0.0)
    r_huge = _sel(50.0)   # an overwhelming (and unrealistic) inhibition strength
    assert r_zero.selected_index != 2
    assert r_huge.selected_index != 2  # excluded candidate never re-admitted


# --------------------------------------------------------------------------- #
# C7 shared delta_t: both levers ON (the A3 arm) move off init together        #
# --------------------------------------------------------------------------- #

def test_c7_shared_delta_t_moves_both_w_chan_and_w_lat():
    cands = [_candidate(i) for i in range(3)]
    sel = _selector(use_learned_channel_gating=True,
                    use_learned_settling_step=True,
                    learned_channel_gating_eta=0.2,
                    learned_settling_eta=0.2,
                    **_SHORTLIST)
    _patch_raw(sel, cands, [0.0, 0.2, 0.4])
    from ree_core.predictors.e3_selector import _LCG_W_INIT
    w_chan_init = sel.w_chan.clone()
    for t in range(6):
        sel.select(cands, score_bias=torch.tensor([-1.0, 0.2, 0.3]))
        benefit, harm = (0.9, 0.1) if t % 2 == 0 else (0.1, 0.7)
        _patch_heads(sel, benefit, harm)
        m = sel.post_action_update(torch.zeros(1, WORLD_DIM), harm_occurred=False)
        # one signed RPE drives both updates this tick
        if "lcg_delta_t" in m and "wlat_delta_t" in m:
            assert float(m["lcg_delta_t"].item()) == float(m["wlat_delta_t"].item())
    assert not torch.allclose(sel.w_chan, w_chan_init)   # w_chan moved
    assert torch.count_nonzero(sel.W_lat).item() > 0     # W_lat moved
    assert abs(_LCG_W_INIT) > 0  # sanity: the init constant is the shared one
