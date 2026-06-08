"""Contracts for the post-603i trainable escape-affordance learner."""

import math

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.pfc.trainable_escape_affordance_learner import (
    TrainableEscapeAffordanceLearner,
    TrainableEscapeAffordanceLearnerConfig,
)
from ree_core.utils.config import REEConfig


def _learner(**kw):
    torch.manual_seed(7)
    params = dict(
        enabled=True,
        n_action_classes=5,
        bias_scale=0.2,
        relief_learn_rate=1.0,
        safety_learn_rate=1.0,
        optimizer_lr=0.05,
        leak_rate=0.0,
        relief_reward_floor=0.05,
        relief_target_scale=0.3,
        threat_floor=0.1,
        noop_class=0,
        hidden_dim=32,
        action_embedding_dim=6,
        prediction_floor=0.02,
    )
    params.update(kw)
    cfg = TrainableEscapeAffordanceLearnerConfig(**params)
    return TrainableEscapeAffordanceLearner(cfg)


def _tick(learner, harm, action=2, directed=True, **kw):
    z_world = kw.pop("z_world", torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32))
    z_self = kw.pop("z_self", torch.tensor([0.3, -0.2], dtype=torch.float32))
    z_harm_a = kw.pop("z_harm_a", torch.tensor([float(harm)], dtype=torch.float32))
    return learner.update(
        float(harm),
        last_action_class=action,
        z_world=z_world,
        z_self=z_self,
        z_harm_a=z_harm_a,
        last_action_directed=directed,
        **kw,
    )


def _state(learner, harm):
    return learner.build_state_vector(
        z_world=torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32),
        z_self=torch.tensor([0.3, -0.2], dtype=torch.float32),
        z_harm_a=torch.tensor([float(harm)], dtype=torch.float32),
        z_harm_a_norm=float(harm),
        threat_scale=learner.threat_scale(float(harm)),
    )


def _train_relief_positive(learner, action=2, n=80):
    for _ in range(n):
        _tick(learner, 0.6, action=action)
        _tick(learner, 0.1, action=action)


def _train_safety_positive(learner, action=3, n=80):
    for _ in range(n):
        _tick(learner, 0.6, action=action)
        _tick(learner, 0.0, action=action)


def test_disabled_path_zero_bias_no_model_no_updates_and_agent_default_off():
    learner = _learner(enabled=False)
    assert learner.model is None
    assert learner.optimizer is None

    _tick(learner, 0.4, action=2)
    _tick(learner, 0.0, action=2)
    bias = learner.compute_approach_bias(0.4, [0, 1, 2, 3, 4])
    state = learner.get_state()

    assert float(bias.abs().max()) == 0.0
    assert learner.model is None
    assert learner.optimizer is None
    assert state["trainable_escape_n_updates"] == 0
    assert state["trainable_escape_n_optimizer_steps"] == 0

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


def test_relief_head_prediction_increases_after_positive_examples():
    learner = _learner()
    threat_state = _state(learner, 0.6)
    before = learner.predict_relief(2, threat_state)

    _train_relief_positive(learner, action=2, n=80)

    after = learner.predict_relief(2, threat_state)
    state = learner.get_state()
    assert learner.model is not None
    assert after > before + 0.15
    assert state["trainable_escape_n_optimizer_steps"] > 0
    assert state["trainable_escape_n_relief_positive"] > 0
    assert state["trainable_escape_relief_max_prediction"] > 0.0


def test_relief_head_prediction_decreases_after_failed_relief_examples():
    learner = _learner()
    threat_state = _state(learner, 0.6)
    _train_relief_positive(learner, action=2, n=80)
    before = learner.predict_relief(2, threat_state)

    for _ in range(100):
        _tick(learner, 0.6, action=2)
        _tick(learner, 0.6, action=2)

    after = learner.predict_relief(2, threat_state)
    assert after < before - 0.05
    assert learner.get_state()["trainable_escape_n_relief_negative"] > 0


def test_safety_head_prediction_increases_after_response_produced_safety():
    learner = _learner()
    threat_state = _state(learner, 0.6)
    before = learner.predict_safety(3, threat_state)

    _train_safety_positive(learner, action=3, n=80)

    after = learner.predict_safety(3, threat_state)
    state = learner.get_state()
    assert after > before + 0.15
    assert state["trainable_escape_n_safety_positive"] > 0
    assert state["trainable_escape_safety_max_prediction"] > 0.0


def test_safety_head_prediction_decreases_after_threat_recurrence():
    learner = _learner()
    threat_state = _state(learner, 0.6)
    _train_safety_positive(learner, action=3, n=80)
    before = learner.predict_safety(3, threat_state)

    for _ in range(100):
        _tick(learner, 0.6, action=3)
        _tick(learner, 0.6, action=3)

    after = learner.predict_safety(3, threat_state)
    assert after < before - 0.05
    assert learner.get_state()["trainable_escape_n_safety_negative"] > 0


def test_simulation_and_hypothesis_guard_prevents_optimizer_steps():
    sim = _learner()
    _tick(sim, 0.4, action=2)
    _tick(sim, 0.0, action=2, simulation_mode=True)
    assert sim.model is None
    assert sim.get_state()["trainable_escape_n_optimizer_steps"] == 0

    hyp = _learner()
    _tick(hyp, 0.4, action=2)
    _tick(hyp, 0.0, action=2, hypothesis_tag=True)
    assert hyp.model is None
    assert hyp.get_state()["trainable_escape_n_optimizer_steps"] == 0


def test_noop_freeze_gets_no_learning_or_bias_by_default():
    learner = _learner()
    _tick(learner, 0.4, action=0)
    _tick(learner, 0.0, action=0)
    bias = learner.compute_approach_bias(0.4, [0, 1, 2])
    state = learner.get_state()

    assert learner.model is None
    assert state["trainable_escape_n_optimizer_steps"] == 0
    assert state["trainable_escape_n_noop_skipped"] > 0
    assert float(bias[0]) == 0.0
    assert float(bias.abs().max()) == 0.0


def test_bias_is_threat_gated_clamped_and_lower_is_better():
    learner = _learner(bias_scale=0.05)
    _train_relief_positive(learner, action=2, n=80)

    safe_bias = learner.compute_approach_bias(0.0, [0, 1, 2, 3, 4])
    assert float(safe_bias.abs().max()) == 0.0

    threat_bias = learner.compute_approach_bias(0.6, [0, 1, 2, 3, 4])
    assert threat_bias[2] < 0.0
    assert float(threat_bias[0]) == 0.0
    assert float(threat_bias.abs().max()) <= learner.config.bias_scale + 1e-9
    assert learner.get_state()["trainable_escape_n_bias_fires"] > 0


def test_model_inputs_are_detached_from_upstream_latents():
    learner = _learner()
    z_world = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32, requires_grad=True)
    z_self = torch.tensor([0.3, -0.2], dtype=torch.float32, requires_grad=True)
    z_harm_a = torch.tensor([0.6], dtype=torch.float32, requires_grad=True)

    _tick(learner, 0.6, action=2, z_world=z_world, z_self=z_self, z_harm_a=z_harm_a)
    _tick(learner, 0.1, action=2, z_world=z_world, z_self=z_self, z_harm_a=z_harm_a)

    assert learner.model is not None
    assert z_world.grad is None
    assert z_self.grad is None
    assert z_harm_a.grad is None


def test_episode_reset_preserves_learned_head_weights_but_clears_traces():
    learner = _learner()
    threat_state = _state(learner, 0.6)
    _train_relief_positive(learner, action=2, n=80)
    model_before = learner.model
    prediction_before = learner.predict_relief(2, threat_state)

    learner.reset()

    prediction_after = learner.predict_relief(2, threat_state)
    assert learner.model is model_before
    assert math.isclose(prediction_after, prediction_before, rel_tol=0.0, abs_tol=1e-7)
    assert learner._prev_z_harm_a_norm is None
    assert learner._prev_state_vector is None
