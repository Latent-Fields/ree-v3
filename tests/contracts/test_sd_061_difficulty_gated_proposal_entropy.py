"""SD-061: difficulty-gated proposal-entropy regulator contracts.

Detector (StuckStateDetector) + regulator (DifficultyGatedProposalEntropy) +
agent wiring. C1 default-OFF bit-identical; C2 detector rises under
impasse-with-goal; C3 goal-salience guard; C4 hysteretic decay; C5 regulator
gain mapping + safety; C6 MECH-094 simulation no-op; C7 agent wiring (build +
tick + reset); C8 from_dims surfaces the knobs.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.cingulate.stuck_state_detector import (
    StuckStateAxisUnavailable,
    StuckStateDetector,
    StuckStateDetectorConfig,
)
from ree_core.policy.difficulty_gated_proposal_entropy import (
    DifficultyGatedProposalEntropy,
    DifficultyGatedProposalEntropyConfig,
)
from ree_core.utils.config import REEConfig

DIMS = dict(
    body_obs_dim=4,
    world_obs_dim=8,
    action_dim=4,
    self_dim=8,
    world_dim=8,
    alpha_world=0.9,
    use_sleep_loop=False,
    sws_enabled=False,
    rem_enabled=False,
    use_sleep_aggregation_cluster=False,
)


def _detector(**ov) -> StuckStateDetector:
    return StuckStateDetector(
        StuckStateDetectorConfig(use_stuck_state_detector=True, **ov)
    )


def _regulator(**ov) -> DifficultyGatedProposalEntropy:
    return DifficultyGatedProposalEntropy(
        DifficultyGatedProposalEntropyConfig(
            use_difficulty_gated_proposal_entropy=True, **ov
        )
    )


def _run_agent(use_dgpe, seed=0, steps=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if use_dgpe is None:
        cfg = REEConfig.from_dims(**DIMS)
    else:
        cfg = REEConfig.from_dims(
            use_difficulty_gated_proposal_entropy=use_dgpe, **DIMS
        )
    agent = REEAgent(cfg)
    agent.reset()
    torch.manual_seed(seed)
    outs = []
    for _ in range(steps):
        act = agent.act_with_split_obs(torch.randn(1, 4), torch.randn(1, 8))
        outs.append(int(torch.argmax(act[0]).item()))
    return outs, agent


# --------------------------------------------------------------------------
# C1: default-OFF bit-identical
# --------------------------------------------------------------------------
def test_c1_default_off_bit_identical():
    o_default, a_default = _run_agent(None)
    o_explicit, a_explicit = _run_agent(False)
    assert a_default.stuck_state_detector is None
    assert a_default.difficulty_gated_proposal_entropy is None
    assert a_explicit.stuck_state_detector is None
    assert o_default == o_explicit


# --------------------------------------------------------------------------
# C2: detector rises under sustained impasse-with-goal
# --------------------------------------------------------------------------
def test_c2_detector_rises_under_impasse():
    d = _detector()
    s = 0.0
    for _ in range(15):
        s = d.update(
            goal_proximity=0.30,
            score_margin=0.0,
            n_candidates=8,
            committed_action_class=3,  # locked-in class
            choice_difficulty=0.0,  # ambiguous (small EV spread)
            goal_salience=0.8,
        )
    assert s > 0.5
    assert d.is_stuck()


# --------------------------------------------------------------------------
# C3: goal-salience guard -- no goal -> not stuck even with all impasse signals
# --------------------------------------------------------------------------
def test_c3_goal_salience_guard():
    d = _detector()
    s = 0.0
    for _ in range(15):
        s = d.update(
            goal_proximity=None,
            score_margin=0.0,
            n_candidates=8,
            committed_action_class=3,
            choice_difficulty=0.0,
            goal_salience=0.0,  # no goal pursued
        )
    assert s == 0.0
    assert not d.is_stuck()


# --------------------------------------------------------------------------
# C4: hysteretic decay -- stuck recovers under relief (progress + margin + div)
# --------------------------------------------------------------------------
def test_c4_hysteretic_decay():
    d = _detector()
    for _ in range(15):
        d.update(
            goal_proximity=0.30,
            score_margin=0.0,
            n_candidates=8,
            committed_action_class=3,
            choice_difficulty=0.0,
            goal_salience=0.8,
        )
    peak = d.get_stuck_score()
    assert peak > 0.5
    s = peak
    for k in range(40):
        s = d.update(
            goal_proximity=0.30 + 0.02 * k,  # advancing
            score_margin=1.0,  # decisive
            n_candidates=8,
            committed_action_class=k % 4,  # diverse
            choice_difficulty=1.0,  # easy
            goal_salience=0.8,
        )
    assert s < peak
    assert s < 0.5
    # rise alpha >> fall alpha -> decay slower than the rise (hysteresis)
    assert d.config.ema_alpha_rise > d.config.ema_alpha_fall


# --------------------------------------------------------------------------
# C5: regulator gain mapping + identity at s=0
# --------------------------------------------------------------------------
def test_c5_regulator_gain_mapping():
    r = _regulator(candidate_widen_max=8, temperature_gain_max=1.0)
    assert r.compute_proposal_gain(0.0) == (0, 1.0)  # identity when not stuck
    assert r.compute_proposal_gain(1.0) == (8, 2.0)  # full gain
    extra, temp = r.compute_proposal_gain(0.5)
    assert extra == 4 and abs(temp - 1.5) < 1e-9
    # monotone non-decreasing in stuck_score
    e_lo, t_lo = r.compute_proposal_gain(0.25)
    e_hi, t_hi = r.compute_proposal_gain(0.75)
    assert e_lo <= e_hi and t_lo <= t_hi
    # clamps out-of-range input
    assert r.compute_proposal_gain(2.0) == (8, 2.0)
    assert r.compute_proposal_gain(-1.0) == (0, 1.0)


# --------------------------------------------------------------------------
# C6: MECH-094 simulation no-op (both modules)
# --------------------------------------------------------------------------
def test_c6_simulation_no_op():
    d = _detector()
    d.update(goal_proximity=0.3, score_margin=0.0, n_candidates=8, goal_salience=0.8)
    before = d.get_stuck_score()
    out = d.update(
        goal_proximity=0.3,
        score_margin=0.0,
        n_candidates=8,
        goal_salience=0.8,
        simulation_mode=True,
    )
    assert out == before
    assert d.get_stuck_score() == before
    r = _regulator()
    assert r.compute_proposal_gain(1.0, simulation_mode=True) == (0, 1.0)


# --------------------------------------------------------------------------
# C7: agent wiring -- builds, ticks, resets
# --------------------------------------------------------------------------
def test_c7_agent_wiring():
    _, agent = _run_agent(True)
    assert agent.stuck_state_detector is not None
    assert agent.difficulty_gated_proposal_entropy is not None
    st = agent.stuck_state_detector.get_state()
    assert st["sd061_n_ticks"] > 0  # detector ticked over the run
    assert agent.difficulty_gated_proposal_entropy.get_state()["sd061_dgpe_n_calls"] > 0
    # reset clears state
    agent.reset()
    assert agent.stuck_state_detector.get_state()["sd061_n_ticks"] == 0
    assert agent._last_stuck_score == 0.0


# --------------------------------------------------------------------------
# C8: from_dims surfaces the SD-061 knobs onto the config
# --------------------------------------------------------------------------
def test_c8_from_dims_surfaces_knobs():
    cfg = REEConfig.from_dims(
        use_difficulty_gated_proposal_entropy=True,
        stuck_threshold=0.42,
        dgpe_candidate_widen_max=5,
        dgpe_temperature_gain_max=0.7,
        stuck_combine_mode="max",
        **DIMS,
    )
    assert cfg.use_difficulty_gated_proposal_entropy is True
    assert cfg.stuck_threshold == 0.42
    assert cfg.dgpe_candidate_widen_max == 5
    assert cfg.dgpe_temperature_gain_max == 0.7
    assert cfg.stuck_combine_mode == "max"
    agent = REEAgent(cfg)
    assert agent.stuck_state_detector.config.stuck_threshold == 0.42
    assert (
        agent.difficulty_gated_proposal_entropy.config.candidate_widen_max == 5
    )


# --------------------------------------------------------------------------
# C9-C13 (2026-09-18, GFLAG-0352): the temperature half and its consumer.
#
# differentiable_cem_temperature is read at exactly ONE place --
# HippocampalModule's CEM refit, inside SD-055's `if use_differentiable_cem`,
# default False -- so SD-061's temperature lever was inert at every config its
# design record named, and V3-EXQ-694's C2 certified the COUNT half alone.
# dgpe_enable_differentiable_cem couples the two; the get_state() diagnostics
# make the difference visible in a manifest either way.
# --------------------------------------------------------------------------
def test_c9_temperature_consumer_off_by_default_and_reported_inert():
    """SD-061 ON at shipped defaults: SD-055 stays off and the run says so."""
    cfg = REEConfig.from_dims(
        use_difficulty_gated_proposal_entropy=True, **DIMS
    )
    # The knob defaults False -> bit-identical with the pre-2026-09-18 tree,
    # so V3-EXQ-694 still reproduces exactly.
    assert cfg.dgpe_enable_differentiable_cem is False
    agent = REEAgent(cfg)
    assert agent.hippocampal.config.use_differentiable_cem is False
    st = agent.difficulty_gated_proposal_entropy.get_state()
    assert st["sd061_temperature_lever_consumer_live"] is False
    # temperature_gain_max defaults 1.0 (> 0), so the half IS inert here.
    assert st["sd061_temperature_half_inert"] is True


def test_c10_knob_couples_the_temperature_half_to_its_consumer():
    """dgpe_enable_differentiable_cem=True turns SD-055's consumer on."""
    cfg = REEConfig.from_dims(
        use_difficulty_gated_proposal_entropy=True,
        dgpe_enable_differentiable_cem=True,
        **DIMS,
    )
    assert cfg.dgpe_enable_differentiable_cem is True
    agent = REEAgent(cfg)
    assert agent.hippocampal.config.use_differentiable_cem is True
    st = agent.difficulty_gated_proposal_entropy.get_state()
    assert st["sd061_temperature_lever_consumer_live"] is True
    assert st["sd061_temperature_half_inert"] is False


def test_c11_knob_is_inert_when_the_sd061_master_flag_is_off():
    """The coupling lives inside the SD-061 construction block -- no leakage."""
    cfg = REEConfig.from_dims(
        use_difficulty_gated_proposal_entropy=False,
        dgpe_enable_differentiable_cem=True,
        **DIMS,
    )
    agent = REEAgent(cfg)
    assert agent.difficulty_gated_proposal_entropy is None
    assert agent.hippocampal.config.use_differentiable_cem is False


def test_c12_inert_flag_is_false_when_there_is_no_temperature_half():
    """temperature_gain_max=0 disables the lever, so nothing is 'inert'."""
    reg = _regulator(temperature_gain_max=0.0)
    st = reg.get_state()
    assert st["sd061_temperature_half_inert"] is False
    # ... and the declaration itself changes no arithmetic.
    live = _regulator(temperature_gain_max=0.6, temperature_lever_consumer_live=True)
    dead = _regulator(temperature_gain_max=0.6, temperature_lever_consumer_live=False)
    for s in (0.0, 0.25, 0.5, 1.0):
        assert live.compute_proposal_gain(s) == dead.compute_proposal_gain(s)


def test_c13_temperature_half_is_provably_inert_until_coupled():
    """The load-bearing contract: does the temperature half do ANYTHING?

    Isolates it by turning the COUNT lever off (candidate_widen_max=0), so the
    only thing stuck_score can still drive is the temperature. Then:

      * UNCOUPLED (SD-055 off, the shipped default and the state V3-EXQ-694 ran
        in): the proposed candidate set is BIT-IDENTICAL at stuck_score 1.0 and
        0.0. Nothing the regulator did reached the proposer. This is the fact
        behind GFLAG-0352.
      * COUPLED (dgpe_enable_differentiable_cem=True): the action-object content
        of the proposal DOES change with stuck_score. The lever is live.

    Both halves are asserted, because only the pair distinguishes "the coupling
    works" from "the probe is insensitive".
    """
    def _proposals(enable_cem, forced, seed=0, steps=8):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cfg = REEConfig.from_dims(
            use_difficulty_gated_proposal_entropy=True,
            dgpe_enable_differentiable_cem=enable_cem,
            dgpe_candidate_widen_max=0,      # COUNT lever OFF -- isolate temperature
            dgpe_temperature_gain_max=1.0,
            **DIMS,
        )
        agent = REEAgent(cfg)
        agent.reset()
        seen = []
        orig = agent.hippocampal.propose_trajectories

        def wrapped(*a, **k):
            out = orig(*a, **k)
            aos = [t.get_action_object_sequence() for t in out]
            aos = [x.detach().clone() for x in aos if x is not None]
            seen.append(torch.stack(aos) if aos else None)
            return out

        agent.hippocampal.propose_trajectories = wrapped
        torch.manual_seed(seed)
        for _ in range(steps):
            agent._last_stuck_score = forced
            agent.act_with_split_obs(torch.randn(1, 4), torch.randn(1, 8))
        return seen

    def _maxdiff(a, b):
        assert len(a) == len(b) and a, "no proposals observed"
        out = []
        for x, y in zip(a, b):
            assert x is not None and y is not None and x.shape == y.shape
            out.append(float((x - y).abs().max()))
        return max(out)

    # UNCOUPLED: stuck_score is invisible to the proposer. Exactly zero.
    off_hot = _proposals(False, 1.0)
    off_cold = _proposals(False, 0.0)
    assert _maxdiff(off_hot, off_cold) == 0.0, (
        "the temperature half is supposed to be INERT without SD-055; if this "
        "fires, a second consumer of differentiable_cem_temperature has "
        "appeared and GFLAG-0352's premise needs re-deriving"
    )

    # COUPLED: it is live.
    on_hot = _proposals(True, 1.0)
    on_cold = _proposals(True, 0.0)
    assert _maxdiff(on_hot, on_cold) > 0.0, (
        "dgpe_enable_differentiable_cem=True did not make the temperature half "
        "reach the proposer -- the coupling is broken"
    )


def test_c13b_coupled_temperature_does_not_move_first_action_class_entropy():
    """MEASURED SCOPE of the coupled lever -- pinned so nobody over-reads it.

    With the count lever off, the coupled temperature perturbs action-OBJECT
    content by ~1e-5 and does NOT move the candidate first-action-CLASS
    distribution, which is the DV that SD-061's what_would_answer criterion (3)
    and MECH-343's upstream leg (a) both name. The class is a coarse argmax of
    the sampled trajectory and a perturbation that small essentially never flips
    it.

    This is NOT an assertion that the coupling is useless -- it is the honest
    record of what it does and does not reach, so that a future Q-056 run cannot
    repeat V3-EXQ-694's error of reading a lever's presence as evidence that the
    DV could have moved. If this test ever FAILS, that is good news and the
    finding should be re-derived rather than the test relaxed.
    """
    def _class_entropies(forced, seed=0, steps=8):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cfg = REEConfig.from_dims(
            use_difficulty_gated_proposal_entropy=True,
            dgpe_enable_differentiable_cem=True,
            dgpe_candidate_widen_max=0,
            dgpe_temperature_gain_max=1.0,
            **DIMS,
        )
        agent = REEAgent(cfg)
        agent.reset()
        seen = []
        orig = agent.hippocampal.propose_trajectories

        def wrapped(*a, **k):
            out = orig(*a, **k)
            seen.append(
                tuple(int(torch.argmax(t.actions[0, 0, :]).item()) for t in out)
            )
            return out

        agent.hippocampal.propose_trajectories = wrapped
        torch.manual_seed(seed)
        for _ in range(steps):
            agent._last_stuck_score = forced
            agent.act_with_split_obs(torch.randn(1, 4), torch.randn(1, 8))
        return seen

    hot = _class_entropies(1.0)
    cold = _class_entropies(0.0)
    assert hot and len(hot) == len(cold)
    assert hot == cold, (
        "the coupled temperature half now MOVES the first-action-class "
        "distribution. That contradicts the 2026-09-18 measurement behind "
        "GFLAG-0352 and materially changes what Q-056's upstream leg (a) can "
        "measure -- re-derive the finding, do not relax this test"
    )


# --------------------------------------------------------------------------
# C14-C16 (2026-09-18, GFLAG-0352): axis-presence diagnostics.
#
# last_deficit_* is 0.0 for BOTH an absent axis and a present-but-zero one, and
# the combine is a mean over PRESENT axes -- so which axes arrive sets the
# attainable maximum of stuck_score. These counters make a null attributable.
# --------------------------------------------------------------------------
def test_c14_axis_presence_separates_absent_from_present_zero():
    d = _detector()
    # Only the two axes an act_with_split_obs-style loop actually supplies.
    d.update(goal_proximity=0.4, goal_salience=0.8)
    d.update(goal_proximity=0.4, goal_salience=0.8)
    st = d.get_state()
    assert st["sd061_last_present_progress"] is True
    assert st["sd061_last_present_margin"] is False
    assert st["sd061_last_present_diversity"] is False
    assert st["sd061_last_present_difficulty"] is False
    assert st["sd061_n_axes_present_last"] == 1
    # The diversity axis reads deficit 0.0 when PRESENT and fully diverse --
    # indistinguishable from absent on last_deficit_diversity alone.
    d2 = _detector()
    d2.update(goal_proximity=0.4, goal_salience=0.8, committed_action_class=0)
    d2.update(goal_proximity=0.4, goal_salience=0.8, committed_action_class=1)
    st2 = d2.get_state()
    assert st2["last_deficit_diversity"] == 0.0
    assert st2["sd061_last_present_diversity"] is True
    assert st2["sd061_n_present_diversity"] == 2


def test_c15_axis_presence_counters_accumulate_and_reset():
    d = _detector()
    for _ in range(3):
        d.update(
            goal_proximity=0.4,
            goal_salience=0.8,
            score_margin=0.02,
            n_candidates=4,
        )
    st = d.get_state()
    assert st["sd061_n_present_progress"] == 2  # first tick has no history yet
    assert st["sd061_n_present_margin"] == 3
    assert st["sd061_n_present_difficulty"] == 0
    d.reset()
    st = d.get_state()
    assert st["sd061_n_present_progress"] == 0
    assert st["sd061_n_present_margin"] == 0
    assert st["sd061_n_axes_present_last"] == 0


def test_c16_presence_diagnostics_do_not_change_the_score():
    """The counters are read-only -- stuck_score is unchanged by their addition.

    Pinned against the closed-form the combine defines, so this fails if a
    later edit lets a presence counter leak into the arithmetic.
    """
    d = _detector(ema_alpha_rise=0.5, stuck_threshold=0.5)
    # progress deficit saturates at 1.0, margin deficit 0.0 -> evidence 0.5
    s = None
    for _ in range(4):
        s = d.update(
            goal_proximity=0.3,
            goal_salience=0.9,
            score_margin=0.99,
            n_candidates=4,
        )
    st = d.get_state()
    assert st["sd061_n_axes_present_last"] == 2
    assert st["last_deficit_progress"] == 1.0
    assert st["last_deficit_margin"] == 0.0
    assert abs(st["last_combined_deficit"] - 0.5) < 1e-12
    # EMA toward 0.5 from 0.0 at alpha 0.5, three advancing ticks: 0.5*(1-0.5^3)
    assert 0.0 < s < 0.5
    assert d.is_stuck() is False


# --------------------------------------------------------------------------
# C17-C24 (2026-09-19): SD-061 (c) -- the DECLARED AXIS MASK.
#
# User decision 2026-09-19 (option 1 of the (c) chip): each run declares which
# detector axes are in scope; the combination is taken over the DECLARED set;
# a declared-but-absent axis is an ERROR, never a silent rescale. Threshold
# recalibration was explicitly REJECTED.
# --------------------------------------------------------------------------
def test_c17_declared_axes_none_is_the_legacy_behaviour_bit_identical():
    """Default-preserving: declared_axes=None must reproduce mean-over-present."""
    legacy = _detector()
    masked_off = _detector(declared_axes=None)
    out_l, out_m = [], []
    for i in range(6):
        kw = dict(goal_proximity=0.3 + 0.001 * i, goal_salience=0.9,
                  score_margin=0.02, n_candidates=4)
        out_l.append(legacy.update(**kw))
        out_m.append(masked_off.update(**kw))
    assert out_l == out_m
    assert legacy.get_state()["sd061_declared_axes"] is None
    assert masked_off.get_state()["sd061_declared_axes"] is None
    # ...and the agent path defaults to None too.
    cfg = REEConfig.from_dims(use_difficulty_gated_proposal_entropy=True, **DIMS)
    assert cfg.stuck_declared_axes is None
    agent = REEAgent(cfg)
    assert agent.stuck_state_detector.config.declared_axes is None


def test_c18_declared_but_unwired_axis_raises_rather_than_rescaling():
    """The load-bearing contract of decision (c).

    An axis that NEVER arrives is a mis-declared run and must refuse. The
    refusal waits out declared_axis_grace_ticks first, because some axes are
    legitimately absent on the opening tick(s) (see C25); inside that window the
    detector is UNDETERMINED -- it never forms a partial combination, so the
    anti-rescale guarantee holds from the very first tick regardless.
    """
    d = _detector(declared_axes=("progress", "difficulty"),
                  declared_axis_grace_ticks=2)
    # 'difficulty' is exactly the axis that never arrives in practice.
    # Inside the grace window: no advance, no partial combine, no raise.
    for _ in range(2):
        assert d.update(goal_proximity=0.3, goal_salience=0.9) == 0.0
    st = d.get_state()
    assert st["sd061_n_ticks"] == 0            # never advanced
    assert st["sd061_n_undetermined_ticks"] == 2
    assert st["last_combined_deficit"] == 0.0  # no partial combination formed
    # Past the window: refuse.
    with pytest.raises(StuckStateAxisUnavailable) as ei:
        d.update(goal_proximity=0.3, goal_salience=0.9)
    msg = str(ei.value)
    assert "difficulty" in msg and "never arrived" in msg
    # The message must be actionable, not just a name.
    assert "use_dacc" in msg or "docstring" in msg
    # A REFUSED tick leaves the detector untouched -- the raise happens before
    # any window append or counter bump.
    st2 = d.get_state()
    assert st2["sd061_n_ticks"] == 0
    assert st2["stuck_score"] == 0.0


def test_c18b_axis_that_arrives_then_stops_refuses_immediately():
    """Wiring breaking mid-run is not warmup -- no grace applies to it."""
    d = _detector(declared_axes=("difficulty",), declared_axis_grace_ticks=99)
    d.update(goal_proximity=0.3, goal_salience=0.9, choice_difficulty=0.01)
    assert d.get_state()["sd061_axes_ever_seen"] == ["difficulty"]
    with pytest.raises(StuckStateAxisUnavailable, match="has now stopped"):
        d.update(goal_proximity=0.3, goal_salience=0.9)


def test_c19_combination_is_over_the_declared_set_not_the_present_set():
    """An UNDECLARED axis is ignored even when its input arrives.

    This is what fixes the drifting denominator: with mean-over-present, an
    extra axis showing up silently changes the attainable MAXIMUM of
    stuck_score. Run at the SHIPPED ema_alpha_rise (0.3), because that is the
    configuration the 2026-09-18 ecological measurement behind GFLAG-0352 was
    taken in -- and the asymptotic approach is part of why the legacy path
    never fires (mean(1.0, 0.0) = 0.5 is reached only in the limit).
    """
    # Declare progress only. Feed BOTH progress and margin.
    only_prog = _detector(declared_axes=("progress",))
    legacy = _detector()
    s_masked = s_legacy = 0.0
    for _ in range(6):
        kw = dict(goal_proximity=0.3, goal_salience=0.9,
                  score_margin=0.99, n_candidates=4)   # margin deficit 0.0
        s_masked = only_prog.update(**kw)
        s_legacy = legacy.update(**kw)
    # progress deficit saturates at 1.0; margin deficit is 0.0.
    assert only_prog.get_state()["last_deficit_progress"] == 1.0
    assert only_prog.get_state()["last_deficit_margin"] == 0.0
    # Declared-progress-only combines over {1.0}: the evidence is 1.0, so the
    # score climbs past the threshold and the gate FIRES.
    assert only_prog.get_state()["last_combined_deficit"] == 1.0
    assert s_masked > 0.5
    assert only_prog.is_stuck() is True
    # Legacy combines over {1.0, 0.0} -> evidence 0.5, which the EMA can only
    # approach from below, so the gate NEVER fires. This is the exact failure
    # GFLAG-0352 recorded (measured duty cycle 0.000 in every arm).
    assert legacy.get_state()["last_combined_deficit"] == 0.5
    assert s_legacy < 0.5
    assert legacy.is_stuck() is False
    # ...and the separation is the mask's doing, not a threshold change: both
    # detectors ran at the same stuck_threshold.
    assert only_prog.config.stuck_threshold == legacy.config.stuck_threshold


def test_c20_declared_axes_validated_at_construction():
    with pytest.raises(ValueError, match="unknown axis"):
        _detector(declared_axes=("progres",))          # typo
    with pytest.raises(ValueError, match="at least one axis"):
        _detector(declared_axes=())
    with pytest.raises(ValueError, match="must not repeat"):
        _detector(declared_axes=("progress", "progress"))
    # Declaration order does not matter -- canonicalised to AXIS_NAMES order.
    a = _detector(declared_axes=("difficulty", "progress"))
    b = _detector(declared_axes=("progress", "difficulty"))
    assert a.get_state()["sd061_declared_axes"] == b.get_state()["sd061_declared_axes"]
    assert a.get_state()["sd061_declared_axes"] == ["progress", "difficulty"]


def test_c21_undetermined_tick_does_not_advance_and_does_not_partially_combine():
    """Warmup is NOT a refusal and NOT a rescale -- the EMA simply does not move.

    A declared axis whose INPUT is wired but whose deficit needs history (the
    progress window needs two samples) must not produce a partial combination.
    """
    d = _detector(declared_axes=("progress", "margin"), ema_alpha_rise=1.0)
    # Tick 1: progress has its input but only one sample -> undetermined.
    s1 = d.update(goal_proximity=0.3, goal_salience=0.9,
                  score_margin=0.0, n_candidates=4)
    st = d.get_state()
    assert s1 == 0.0                       # no advance
    assert st["sd061_n_undetermined_ticks"] == 1
    assert st["sd061_last_undetermined"] is True
    assert st["sd061_last_undetermined_axes"] == ["progress"]
    assert st["sd061_n_ticks"] == 0        # the tick was not counted as advanced
    # Tick 2: both determinable -> it advances.
    s2 = d.update(goal_proximity=0.3, goal_salience=0.9,
                  score_margin=0.0, n_candidates=4)
    st = d.get_state()
    assert s2 > 0.0
    assert st["sd061_n_undetermined_ticks"] == 1
    assert st["sd061_last_undetermined"] is False
    assert st["sd061_n_ticks"] == 1


def test_c22_mask_reaches_the_detector_through_from_dims_and_the_agent():
    cfg = REEConfig.from_dims(
        use_difficulty_gated_proposal_entropy=True,
        stuck_declared_axes=["margin", "progress"],
        **DIMS,
    )
    assert cfg.stuck_declared_axes == ["margin", "progress"]
    agent = REEAgent(cfg)
    # Normalised to a tuple so the caller's list cannot mutate it afterwards.
    assert agent.stuck_state_detector.config.declared_axes == ("margin", "progress")
    cfg.stuck_declared_axes.append("difficulty")
    assert agent.stuck_state_detector.config.declared_axes == ("margin", "progress")
    assert agent.stuck_state_detector.get_state()["sd061_declared_axes"] == [
        "progress", "margin"
    ]


def test_c23_reset_clears_the_mask_counters_but_not_the_declaration():
    d = _detector(declared_axes=("progress",))
    d.update(goal_proximity=0.3, goal_salience=0.9)   # undetermined (warmup)
    assert d.get_state()["sd061_n_undetermined_ticks"] == 1
    d.reset()
    st = d.get_state()
    assert st["sd061_n_undetermined_ticks"] == 0
    assert st["sd061_last_undetermined"] is False
    assert st["sd061_last_undetermined_axes"] == []
    # The DECLARATION survives reset -- it is configuration, not episode state.
    assert st["sd061_declared_axes"] == ["progress"]


def test_c24_mask_refusal_propagates_through_the_agent_tick():
    """End-to-end: a mis-declared run FAILS rather than returning a number.

    The agent's select_action feeds the detector; a declared axis the driver
    cannot wire must surface as an error there, which the runner records as
    ERROR -- not as a plausible-looking null. 40 ticks comfortably exceeds the
    default 8-tick grace window.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    cfg = REEConfig.from_dims(
        use_difficulty_gated_proposal_entropy=True,
        stuck_declared_axes=["difficulty"],   # unreachable via act_with_split_obs
        **DIMS,
    )
    agent = REEAgent(cfg)
    agent.reset()
    with pytest.raises(StuckStateAxisUnavailable):
        for _ in range(40):
            agent.act_with_split_obs(torch.randn(1, 4), torch.randn(1, 8))


def test_c25_an_axis_absent_only_at_the_start_is_declarable():
    """The regression the ecological probe caught in the first implementation.

    `score_margin` is None until the first E3 selection has happened -- measured
    99/100 ticks present, absent on the first only. A refusal keyed to a single
    tick's inputs made `margin` undeclarable by ANY driver, which is not what
    the decision asks for. Warmup absence must be survivable; only a
    never-arriving axis refuses.
    """
    d = _detector(declared_axes=("margin",), declared_axis_grace_ticks=8)
    # Tick 1: no selection yet -> input absent, but inside the window.
    assert d.update(goal_proximity=0.3, goal_salience=0.9) == 0.0
    assert d.get_state()["sd061_n_undetermined_ticks"] == 1
    # Ticks 2+: the margin arrives and the detector advances normally.
    for _ in range(4):
        d.update(goal_proximity=0.3, goal_salience=0.9,
                 score_margin=0.0, n_candidates=4)
    st = d.get_state()
    assert st["sd061_n_ticks"] == 4
    assert st["sd061_axes_ever_seen"] == ["margin"]
    assert d.get_stuck_score() > 0.0


def test_c26_grace_window_cannot_change_a_measured_quantity():
    """The grace knob decides only WHEN a mis-wired run is told.

    A correctly-wired run must be bit-identical across grace settings, because
    it never reaches either the refusal or the undetermined branch after warmup.
    """
    def _series(grace):
        d = _detector(declared_axes=("progress", "margin"),
                      declared_axis_grace_ticks=grace)
        out = []
        for i in range(8):
            out.append(d.update(goal_proximity=0.3 + 0.001 * i, goal_salience=0.9,
                                score_margin=0.01, n_candidates=4))
        return out

    assert _series(0) == _series(8) == _series(64)
