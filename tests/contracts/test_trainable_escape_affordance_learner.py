"""Contracts for the post-603i trainable escape-affordance learner scaffold."""

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.pfc.trainable_escape_affordance_learner import (
    TrainableEscapeAffordanceLearner,
    TrainableEscapeAffordanceLearnerConfig,
)
from ree_core.utils.config import REEConfig


def _learner(**kw):
    params = dict(
        enabled=True,
        n_action_classes=5,
        bias_scale=0.2,
        relief_learn_rate=0.5,
        safety_learn_rate=0.5,
        leak_rate=0.0,
        relief_reward_floor=0.05,
        threat_floor=0.1,
        noop_class=0,
    )
    params.update(kw)
    cfg = TrainableEscapeAffordanceLearnerConfig(**params)
    return TrainableEscapeAffordanceLearner(cfg)


def test_disabled_path_zero_bias_no_updates_and_agent_default_off():
    learner = _learner(enabled=False)
    learner.update(0.4, last_action_class=2)
    learner.update(0.0, last_action_class=2)
    bias = learner.compute_approach_bias(0.4, [0, 1, 2, 3, 4])
    assert float(bias.abs().max()) == 0.0
    assert learner.relief_values == [0.0] * 5
    assert learner.safety_values == [0.0] * 5
    assert learner.get_state()["trainable_escape_n_updates"] == 0

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = REEConfig.from_dims(
        world_obs_dim=250,
        body_obs_dim=12,
        harm_obs_dim=50,
        harm_obs_a_dim=50,
        action_dim=5,
    )
    agent = REEAgent(cfg)
    assert agent.trainable_escape_affordance_learner is None


def test_relief_positive_only_for_directed_action_under_threat_harm_drop():
    learner = _learner()
    learner.update(0.4, last_action_class=2)
    out = learner.update(0.1, last_action_class=2)
    assert out.relief_target == 1.0
    assert learner.relief_values[2] > 0.0
    assert all(learner.relief_values[i] == 0.0 for i in (0, 1, 3, 4))

    no_drop = _learner()
    no_drop.update(0.4, last_action_class=2)
    no_drop.update(0.39, last_action_class=2)
    assert no_drop.relief_values[2] == 0.0
    assert no_drop.get_state()["trainable_escape_n_relief_positive"] == 0

    noop = _learner()
    noop.update(0.4, last_action_class=0)
    noop.update(0.0, last_action_class=0)
    assert noop.relief_values[0] == 0.0
    assert noop.safety_values[0] == 0.0
    assert noop.get_state()["trainable_escape_n_noop_skipped"] > 0


def test_safety_positive_and_threat_recurrence_extinction():
    learner = _learner()
    learner.update(0.4, last_action_class=3)
    out = learner.update(0.0, last_action_class=3)
    assert out.safety_target == 1.0
    assert learner.safety_values[3] > 0.0

    before = learner.safety_values[3]
    learner.update(0.5, last_action_class=3)
    assert learner.safety_values[3] < before
    assert learner.get_state()["trainable_escape_n_safety_negative"] > 0


def test_simulation_and_hypothesis_guard_prevents_learning():
    sim = _learner()
    sim.update(0.4, last_action_class=2)
    sim.update(0.0, last_action_class=2, simulation_mode=True)
    assert sim.relief_values[2] == 0.0
    assert sim.safety_values[2] == 0.0

    hyp = _learner()
    hyp.update(0.4, last_action_class=2)
    hyp.update(0.0, last_action_class=2, hypothesis_tag=True)
    assert hyp.relief_values[2] == 0.0
    assert hyp.safety_values[2] == 0.0


def test_bias_is_threat_gated_clamped_and_noop_gets_no_bonus():
    learner = _learner(bias_scale=0.05)
    learner.update(0.4, last_action_class=2)
    learner.update(0.0, last_action_class=2)

    safe_bias = learner.compute_approach_bias(0.0, [0, 1, 2, 3, 4])
    assert float(safe_bias.abs().max()) == 0.0

    threat_bias = learner.compute_approach_bias(0.4, [0, 1, 2, 3, 4])
    assert threat_bias[2] < 0.0
    assert float(threat_bias[0]) == 0.0
    assert float(threat_bias.abs().max()) <= learner.config.bias_scale + 1e-9


def test_leak_and_negative_relief_extinction_decay_stale_predictions():
    learner = _learner(leak_rate=0.2)
    learner.update(0.4, last_action_class=2)
    learner.update(0.0, last_action_class=2)
    before_leak = learner.relief_values[2]
    learner.update(0.0, last_action_class=2)
    assert learner.relief_values[2] < before_leak

    learner.update(0.4, last_action_class=2)
    before_fail = learner.relief_values[2]
    learner.update(0.39, last_action_class=2)
    assert learner.relief_values[2] < before_fail
    assert learner.get_state()["trainable_escape_n_relief_negative"] > 0
