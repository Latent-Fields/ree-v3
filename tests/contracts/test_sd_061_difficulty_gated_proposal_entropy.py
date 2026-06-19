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
import torch

from ree_core.agent import REEAgent
from ree_core.cingulate.stuck_state_detector import (
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
