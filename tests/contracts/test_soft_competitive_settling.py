"""Contracts for MECH-140 x MECH-450 disinhibitory soft-competitive settling.

The PARAMETER-FREE, always-graded complement to the LEARNED W_lat settling
(use_learned_settling_step, which is a no-op at init). A few rounds of
soft-competitive lateral inhibition over the F + MECH-448/449 within-eligible
field BEFORE the commit, so the committed action emerges from a bounded recurrent
SETTLING competition rather than a one-shot argmin (MECH-450), with losing options
down-weighted GRADED-ly but never silenced (MECH-140 -- soft-competitive
disinhibition, not winner-take-all).

The lateral-inhibition kernel is the parameter-free class-surround kernel
(1.0 within a first-action class, cross_class < 1 across classes, 0 on the
diagonal -- surround inhibition between competing motor programs, Mink 1996;
the SAME structure W_lat learns, here FIXED and always-on). Because the kernel
encodes candidate-vs-candidate STRUCTURE, the settling can REORDER: a candidate
crowded by same-class rivals accrues more lateral inhibition than an isolated
slightly-worse one and can lose to it -- the behavioural non-vacuity (an attractor
flip the one-shot argmin structurally lacks) the MECH-439 conversion ceiling needs.

Coverage:
  - config defaults are no-op (bit-identical OFF guarantee surface);
  - from_dims surfaces the 5 flags onto config.e3;
  - BYTE-IDENTICAL: flag ON with gain 0.0 == flag OFF, selected_index AND scores,
    across >= 12 seeds and varied pools (the "byte-identical at flag-off / at-init"
    contract; gain 0.0 is the default even when the master flag is on);
  - NON-VACUITY: with gain > 0 the settling MOVES the readout -- flips the committed
    winner vs the one-shot within-eligible argmin, and round_delta > 0;
  - GRADED-not-WTA: after settling every loser keeps strictly-positive activation
    (< winner), even at high gain / many rounds -- nothing is silenced;
  - SAFETY: the settled winner stays inside the F + Go/No-Go eligible set (a harmful
    outlier excluded by the envelope is never selected, even under settling + a strong
    pull toward it) and >= 1 candidate always survives;
  - MECH-094 waking gate: a simulation/replay tick does NOT settle;
  - composes with the segregated-loop (ARC-110) path.
"""

from __future__ import annotations

import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import (
    E3Config,
    E3TrajectorySelector,
    _FCG_CHANNEL_INDEX,
)
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
    assert cfg.use_soft_competitive_settling is False
    assert cfg.soft_competitive_settling_rounds == 3
    assert cfg.soft_competitive_settling_gain == 0.0
    assert cfg.soft_competitive_settling_temperature == 1.0
    assert cfg.soft_competitive_settling_cross_class == 0.25


def test_from_dims_surfaces_flags_onto_e3():
    cfg = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4, self_dim=16, world_dim=16,
        use_soft_competitive_settling=True,
        soft_competitive_settling_rounds=5,
        soft_competitive_settling_gain=1.5,
        soft_competitive_settling_temperature=0.5,
        soft_competitive_settling_cross_class=0.1,
    )
    assert cfg.e3.use_soft_competitive_settling is True
    assert cfg.e3.soft_competitive_settling_rounds == 5
    assert cfg.e3.soft_competitive_settling_gain == 1.5
    assert cfg.e3.soft_competitive_settling_temperature == 0.5
    assert cfg.e3.soft_competitive_settling_cross_class == 0.1
    # Default from_dims leaves the lever OFF.
    cfg_off = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4, self_dim=16, world_dim=16,
    )
    assert cfg_off.e3.use_soft_competitive_settling is False


# --------------------------------------------------------------------------- #
# BYTE-IDENTICAL: flag ON + gain 0.0 == flag OFF, across >= 12 seeds           #
# --------------------------------------------------------------------------- #


def test_byte_identical_off_across_seeds():
    """Flag ON with gain 0.0 (the default even when the master flag is on) is an
    EXACT no-op: identical committed index and scores to flag OFF, on the
    F-eligibility-demotion stack, across 12 seeds and random divergent pools."""
    for seed in range(12):
        torch.manual_seed(seed)
        raw = torch.rand(5).tolist()
        bias = (torch.rand(5) - 0.5) * 2.0

        sel_off = _selector(use_f_eligibility_demotion=True)
        c_off = [_candidate(i % 3) for i in range(5)]  # some class-crowding present
        _patch_raw(sel_off, c_off, raw)
        r_off = sel_off.select(c_off, temperature=1.0, score_bias=bias.clone())

        sel_on = _selector(
            use_f_eligibility_demotion=True,
            use_soft_competitive_settling=True,  # ON, but gain 0.0 -> exact no-op
        )
        c_on = [_candidate(i % 3) for i in range(5)]
        _patch_raw(sel_on, c_on, raw)
        r_on = sel_on.select(c_on, temperature=1.0, score_bias=bias.clone())

        assert r_on.selected_index == r_off.selected_index, (
            f"seed {seed}: ON-at-default-gain ({r_on.selected_index}) must match OFF "
            f"({r_off.selected_index})"
        )
        assert torch.equal(r_on.scores, r_off.scores), f"seed {seed}: scores diverged"
        # At gain 0.0 the field never MOVES: round_delta is 0.0 (settling ran as an
        # exact no-op) or -1.0 (envelope of one -> settling skipped) -- never > 0.
        assert sel_on.last_score_diagnostics["soft_competitive_settling_round_delta"] <= 0.0


# --------------------------------------------------------------------------- #
# NON-VACUITY: the settling MOVES the readout (flips the winner)               #
# --------------------------------------------------------------------------- #


def test_settling_flips_the_committed_winner():
    """A candidate that is the modulatory argmin but sits in a CROWDED first-action
    class loses, under soft-competitive settling, to an isolated slightly-worse
    candidate -- the readout the one-shot argmin could not produce."""
    # raw F is a flat near-tie -> the envelope admits all 4 (wide, low-conflict).
    raw = [0.0, 0.0, 0.0, 0.0]
    # Classes: 0,1,2 share class A (crowded); 3 is class B (isolated).
    classes = [0, 0, 0, 1]
    # Modulatory bias (COST, lower=better): idx0 is best, idx3 is worst.
    bias = torch.tensor([-1.0, -0.9, -0.9, -0.8])

    # OFF: one-shot within-eligible argmin over the bias -> idx0 (best bias).
    sel_off = _selector(use_f_eligibility_demotion=True)
    c_off = [_candidate(k) for k in classes]
    _patch_raw(sel_off, c_off, raw)
    r_off = sel_off.select(c_off, temperature=1.0, score_bias=bias.clone())
    assert r_off.selected_index == 0

    # ON: settling suppresses the crowded class-A cluster -> isolated idx3 wins.
    sel_on = _selector(
        use_f_eligibility_demotion=True,
        use_soft_competitive_settling=True,
        soft_competitive_settling_gain=2.0,
        soft_competitive_settling_rounds=4,
    )
    c_on = [_candidate(k) for k in classes]
    _patch_raw(sel_on, c_on, raw)
    r_on = sel_on.select(c_on, temperature=1.0, score_bias=bias.clone())
    assert r_on.selected_index == 3, (
        f"soft-competitive settling must flip the crowded-class winner to the isolated "
        f"candidate; got {r_on.selected_index}"
    )
    d = sel_on.last_score_diagnostics
    assert d["soft_competitive_settling_active"] is True
    assert d["soft_competitive_settling_round_delta"] > 0.0, "the field must MOVE"


def test_method_level_reorder_and_round_delta():
    """Direct _soft_competitive_settle: the settled COST argmin differs from the
    input argmin when crowding warrants it, and the round-delta is recorded."""
    sel = _selector(
        use_soft_competitive_settling=True,
        soft_competitive_settling_gain=2.0,
        soft_competitive_settling_rounds=4,
    )
    classes = [0, 0, 0, 1]
    cands = [_candidate(k) for k in classes]
    field = torch.tensor([-1.0, -0.9, -0.9, -0.8])  # COST: idx0 best pre-settle
    elig = torch.arange(4)
    settled = sel._soft_competitive_settle(field, cands, elig)
    assert int(field.argmin().item()) == 0
    assert int(settled.argmin().item()) == 3, "crowded idx0 loses to isolated idx3"
    assert sel._scs_last_round_delta > 0.0


# --------------------------------------------------------------------------- #
# GRADED, not WTA: losers reduced but never silenced                          #
# --------------------------------------------------------------------------- #


def test_graded_not_wta_loser_stays_nonzero():
    """After settling, in a near-tie, the loser retains strictly-positive (reduced)
    activation -- soft-competitive disinhibition, not winner-take-all -- even at
    high gain and many rounds."""
    sel = _selector(
        use_soft_competitive_settling=True,
        soft_competitive_settling_gain=5.0,
        soft_competitive_settling_rounds=10,
    )
    cands = [_candidate(0), _candidate(1)]  # distinct classes, minimal cross-inhibition
    field = torch.tensor([0.0, 0.001])  # idx0 marginally better
    sel._soft_competitive_settle(field, cands, torch.arange(2))
    support = sel._scs_last_support
    assert support is not None
    assert float(support.min().item()) > 0.0, "no candidate is silenced (graded)"
    assert support[0] > support[1], "the winner retains more activation than the loser"


# --------------------------------------------------------------------------- #
# SAFETY: settled winner stays inside the eligible set                        #
# --------------------------------------------------------------------------- #


def test_safety_harmful_outlier_never_selected_under_settling():
    """A clearly-harmful candidate (huge F cost) is excluded by the MECH-448
    envelope; settling transforms ONLY the eligible subset, so it can never be
    re-admitted -- even under an overwhelming modulatory pull toward it."""
    sel = _selector(
        use_f_eligibility_demotion=True,
        use_soft_competitive_settling=True,
        soft_competitive_settling_gain=3.0,
        soft_competitive_settling_rounds=5,
    )
    candidates = [_candidate(i) for i in range(4)]
    _patch_raw(sel, candidates, [0.0, 0.1, 0.2, 5.0])  # idx3 clearly harmful
    bias = torch.tensor([0.0, 0.0, 0.0, -100.0])  # overwhelming pull toward idx3
    r = sel.select(candidates, temperature=1.0, score_bias=bias)
    assert r.selected_index != 3
    assert sel.last_score_diagnostics["f_eligibility_excluded_count"] >= 1


def test_single_survivor_is_safe_noop():
    """A decisive F-winner leaves an envelope of one; settling needs >= 2 eligible,
    so it is skipped and the lone survivor is committed (no crash, >= 1 survivor)."""
    sel = _selector(
        use_f_eligibility_demotion=True,
        use_soft_competitive_settling=True,
        soft_competitive_settling_gain=3.0,
    )
    candidates = [_candidate(i) for i in range(4)]
    _patch_raw(sel, candidates, [0.0, 1.0, 1.0, 1.0])  # decisive -> envelope {0}
    r = sel.select(candidates, temperature=1.0, score_bias=torch.zeros(4))
    assert r.selected_index == 0
    assert sel.last_score_diagnostics["soft_competitive_settling_active"] is False


# --------------------------------------------------------------------------- #
# MECH-094 waking gate                                                         #
# --------------------------------------------------------------------------- #


def test_simulation_tick_does_not_settle():
    """simulation_mode=True -> settling is skipped -> the readout is the one-shot
    argmin (the crowded-class winner), NOT the flipped isolated candidate."""
    raw = [0.0, 0.0, 0.0, 0.0]
    classes = [0, 0, 0, 1]
    bias = torch.tensor([-1.0, -0.9, -0.9, -0.8])
    sel = _selector(
        use_f_eligibility_demotion=True,
        use_soft_competitive_settling=True,
        soft_competitive_settling_gain=2.0,
        soft_competitive_settling_rounds=4,
    )
    candidates = [_candidate(k) for k in classes]
    _patch_raw(sel, candidates, raw)
    r = sel.select(
        candidates, temperature=1.0, score_bias=bias.clone(), simulation_mode=True
    )
    assert r.selected_index == 0, "a simulation tick must not settle (waking-only)"
    assert sel.last_score_diagnostics["soft_competitive_settling_active"] is False


# --------------------------------------------------------------------------- #
# Composes with the segregated-loop (ARC-110) path                            #
# --------------------------------------------------------------------------- #


def _seg_selector(**e3_kwargs) -> E3TrajectorySelector:
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=8, self_dim=32, world_dim=32,
        use_loop_segregation=True,
        **e3_kwargs,
    )
    return E3TrajectorySelector(cfg.e3, None)


def test_segregated_path_byte_identical_off():
    """Under loop segregation, settling ON-at-gain-0 matches settling OFF (the
    arbitrated `final` field is settled by an exact no-op)."""
    n = 6
    elig = torch.arange(n)
    classes = [i % 3 for i in range(n)]
    cands = [_candidate(k) for k in classes]
    for seed in range(12):
        torch.manual_seed(seed)
        raw = torch.randn(n)
        terms = [
            (_FCG_CHANNEL_INDEX["ofc"], torch.randn(n)),
            (_FCG_CHANNEL_INDEX["dacc"], torch.randn(n)),
        ]
        loc_off = _seg_selector(
            use_soft_competitive_settling=False
        )._segregated_loop_arbitrate(elig, raw, terms, True, cands, True, 1.0, False)
        loc_on = _seg_selector(
            use_soft_competitive_settling=True,  # gain 0.0 -> no-op
        )._segregated_loop_arbitrate(elig, raw, terms, True, cands, True, 1.0, False)
        assert loc_off == loc_on, f"seed {seed}: segregated settling no-op diverged"


def test_segregated_path_settles_when_on():
    """Under loop segregation with gain > 0 and a waking tick, the settling runs on
    the arbitrated `final` field (round_delta recorded > 0)."""
    n = 6
    classes = [i % 2 for i in range(n)]  # two classes -> crowding + cross-inhibition
    cands = [_candidate(k) for k in classes]
    terms = [
        (_FCG_CHANNEL_INDEX["ofc"], torch.randn(n)),
        (_FCG_CHANNEL_INDEX["dacc"], torch.randn(n)),
    ]
    sel = _seg_selector(
        use_soft_competitive_settling=True,
        soft_competitive_settling_gain=2.0,
        soft_competitive_settling_rounds=4,
    )
    torch.manual_seed(0)
    sel._segregated_loop_arbitrate(
        torch.arange(n), torch.randn(n), terms, True, cands, True, 1.0, False
    )
    d = sel.last_score_diagnostics
    assert d["soft_competitive_settling_active"] is True
    assert d["soft_competitive_settling_round_delta"] > 0.0
