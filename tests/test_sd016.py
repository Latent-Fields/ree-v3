"""
Tests for SD-016: Frontal Cue Integration Circuit (MECH-150/151/152, ARC-041)

Covers:
  - E1Config: sd016_enabled flag and action_object_dim field
  - E1DeepPredictor: world_query_proj/cue_action_proj/cue_terrain_proj instantiation
  - E1DeepPredictor.extract_cue_context(): output shapes, sigmoid bound, backward compat
  - E2FastPredictor.action_object(): action_bias additive effect
  - E2FastPredictor.rollout_with_world(): action_bias propagated to each action_object call
  - HippocampalModule.propose_trajectories(): action_bias accepted, trajectories unchanged shape
  - E3TrajectorySelector.score_trajectory(): terrain_weight scales harm/benefit terms
  - E3TrajectorySelector.select(): terrain_weight passed through to score_trajectory
  - REEAgent: _cue_action_bias/_cue_terrain_weight lifecycle (init, reset, e1_tick)
  - REEAgent: backward compat -- sd016_enabled=False leaves all cue signals None
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import (
    E1Config, E2Config, E3Config, HippocampalConfig, ResidueConfig, REEConfig,
)
from ree_core.predictors.e1_deep import E1DeepPredictor, ContextMemory
from ree_core.predictors.e2_fast import E2FastPredictor, Trajectory
from ree_core.predictors.e3_selector import E3TrajectorySelector
from ree_core.hippocampal.module import HippocampalModule
from ree_core.residue.field import ResidueField
from ree_core.agent import REEAgent


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

SELF_DIM  = 32
WORLD_DIM = 32
ACTION_DIM = 4
ACTION_OBJECT_DIM = 16
BATCH = 2


def make_e1(sd016_enabled: bool = False, action_object_dim: int = 16) -> E1DeepPredictor:
    cfg = E1Config(
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        latent_dim=SELF_DIM + WORLD_DIM,
        hidden_dim=128,
        sd016_enabled=sd016_enabled,
        action_object_dim=action_object_dim,
    )
    return E1DeepPredictor(cfg)


def make_e2() -> E2FastPredictor:
    cfg = E2Config(
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM,
        rollout_horizon=5,
        num_candidates=4,
    )
    return E2FastPredictor(cfg)


def make_e3() -> E3TrajectorySelector:
    cfg = E3Config(world_dim=WORLD_DIM, hidden_dim=64)
    return E3TrajectorySelector(cfg)


def make_hippocampal(e2: E2FastPredictor) -> HippocampalModule:
    cfg = HippocampalConfig(
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM,
        hidden_dim=64,
        horizon=5,
        num_candidates=4,
        num_cem_iterations=1,
    )
    residue_cfg = ResidueConfig(world_dim=WORLD_DIM)
    residue = ResidueField(residue_cfg)
    return HippocampalModule(cfg, e2, residue)


# ------------------------------------------------------------------ #
# E1Config tests                                                       #
# ------------------------------------------------------------------ #

class TestE1Config:
    def test_sd016_default_false(self):
        cfg = E1Config()
        assert cfg.sd016_enabled is False

    def test_sd016_enabled_flag(self):
        cfg = E1Config(sd016_enabled=True, action_object_dim=16)
        assert cfg.sd016_enabled is True
        assert cfg.action_object_dim == 16

    def test_from_dims_sd016_disabled_by_default(self):
        config = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=54, action_dim=4)
        assert config.e1.sd016_enabled is False

    def test_from_dims_sd016_enabled(self):
        config = REEConfig.from_dims(
            body_obs_dim=10, world_obs_dim=54, action_dim=4, sd016_enabled=True
        )
        assert config.e1.sd016_enabled is True
        assert config.e1.action_object_dim == 16  # default action_object_dim


# ------------------------------------------------------------------ #
# E1DeepPredictor SD-016 projection instantiation                     #
# ------------------------------------------------------------------ #

class TestE1Projections:
    def test_no_projections_when_disabled(self):
        e1 = make_e1(sd016_enabled=False)
        assert not hasattr(e1, 'world_query_proj'), "world_query_proj should not exist"
        assert not hasattr(e1, 'cue_action_proj'),  "cue_action_proj should not exist"
        assert not hasattr(e1, 'cue_terrain_proj'), "cue_terrain_proj should not exist"

    def test_projections_exist_when_enabled(self):
        e1 = make_e1(sd016_enabled=True)
        assert hasattr(e1, 'world_query_proj'), "world_query_proj missing"
        assert hasattr(e1, 'cue_action_proj'),  "cue_action_proj missing"
        assert hasattr(e1, 'cue_terrain_proj'), "cue_terrain_proj missing"

    def test_world_query_proj_dimensions(self):
        e1 = make_e1(sd016_enabled=True)
        # world_query_proj: world_dim -> hidden_dim (= memory_dim)
        assert e1.world_query_proj.in_features  == WORLD_DIM
        assert e1.world_query_proj.out_features == 128  # hidden_dim

    def test_cue_action_proj_dimensions(self):
        e1 = make_e1(sd016_enabled=True, action_object_dim=ACTION_OBJECT_DIM)
        # cue_action_proj: latent_dim (64) -> action_object_dim (16)
        assert e1.cue_action_proj.in_features  == SELF_DIM + WORLD_DIM
        assert e1.cue_action_proj.out_features == ACTION_OBJECT_DIM

    def test_cue_terrain_proj_dimensions(self):
        e1 = make_e1(sd016_enabled=True)
        # cue_terrain_proj: latent_dim (64) -> 2
        assert e1.cue_terrain_proj.in_features  == SELF_DIM + WORLD_DIM
        assert e1.cue_terrain_proj.out_features == 2

    def test_generate_prior_unaffected(self):
        """generate_prior() contract is unchanged by SD-016."""
        e1 = make_e1(sd016_enabled=True)
        total_state = torch.randn(BATCH, SELF_DIM + WORLD_DIM)
        prior = e1.generate_prior(total_state)
        assert prior.shape == (BATCH, WORLD_DIM)


# ------------------------------------------------------------------ #
# E1DeepPredictor.extract_cue_context()                               #
# ------------------------------------------------------------------ #

class TestExtractCueContext:
    def test_output_shapes(self):
        e1 = make_e1(sd016_enabled=True, action_object_dim=ACTION_OBJECT_DIM)
        z_world = torch.randn(BATCH, WORLD_DIM)
        action_bias, terrain_weight = e1.extract_cue_context(z_world)
        assert action_bias.shape    == (BATCH, ACTION_OBJECT_DIM), \
            f"action_bias shape {action_bias.shape}"
        assert terrain_weight.shape == (BATCH, 2), \
            f"terrain_weight shape {terrain_weight.shape}"

    def test_terrain_weight_sigmoid_bounded(self):
        """terrain_weight must be in (0, 1) -- sigmoid applied."""
        e1 = make_e1(sd016_enabled=True)
        z_world = torch.randn(BATCH, WORLD_DIM)
        _, terrain_weight = e1.extract_cue_context(z_world)
        assert terrain_weight.min().item() > 0.0, "terrain_weight has non-positive values"
        assert terrain_weight.max().item() < 1.0, "terrain_weight has values >= 1"

    def test_action_bias_unbounded(self):
        """action_bias has no activation -- must allow negative values."""
        e1 = make_e1(sd016_enabled=True)
        z_world = torch.randn(BATCH, WORLD_DIM)
        action_bias, _ = e1.extract_cue_context(z_world)
        # With random init some values should be negative (statistically very likely)
        # Run a few times to make it near-certain
        found_negative = False
        for _ in range(10):
            ab, _ = e1.extract_cue_context(torch.randn(BATCH, WORLD_DIM))
            if ab.min().item() < 0:
                found_negative = True
                break
        assert found_negative, "action_bias appears bounded (no negative values found)"

    def test_different_z_world_different_output(self):
        """Different z_world inputs should produce different cue contexts."""
        e1 = make_e1(sd016_enabled=True)
        z_a = torch.randn(1, WORLD_DIM)
        z_b = torch.randn(1, WORLD_DIM)
        ab_a, tw_a = e1.extract_cue_context(z_a)
        ab_b, tw_b = e1.extract_cue_context(z_b)
        # The two action_biases should differ (different queries produce different results)
        assert not torch.allclose(ab_a, ab_b), "Different z_world produced identical action_bias"

    def test_batch_size_1(self):
        e1 = make_e1(sd016_enabled=True)
        z_world = torch.randn(1, WORLD_DIM)
        action_bias, terrain_weight = e1.extract_cue_context(z_world)
        assert action_bias.shape    == (1, ACTION_OBJECT_DIM)
        assert terrain_weight.shape == (1, 2)

    def test_gradients_flow_to_projections(self):
        """Gradients must flow from extract_cue_context outputs back to projections."""
        e1 = make_e1(sd016_enabled=True)
        z_world = torch.randn(BATCH, WORLD_DIM)
        action_bias, terrain_weight = e1.extract_cue_context(z_world)
        loss = action_bias.sum() + terrain_weight.sum()
        loss.backward()
        assert e1.cue_action_proj.weight.grad  is not None, "No grad on cue_action_proj"
        assert e1.cue_terrain_proj.weight.grad is not None, "No grad on cue_terrain_proj"
        assert e1.world_query_proj.weight.grad is not None, "No grad on world_query_proj"


# ------------------------------------------------------------------ #
# E2FastPredictor.action_object() -- action_bias                      #
# ------------------------------------------------------------------ #

class TestE2ActionBias:
    def test_no_bias_unchanged(self):
        """action_object without bias should be identical to old behavior."""
        e2 = make_e2()
        z_world = torch.randn(BATCH, WORLD_DIM)
        action  = torch.randn(BATCH, ACTION_DIM)
        o_new = e2.action_object(z_world, action, action_bias=None)
        o_old = e2.action_object(z_world, action)  # original signature
        assert torch.allclose(o_new, o_old)

    def test_bias_adds_to_output(self):
        """action_object(z, a, bias) == action_object(z, a) + bias."""
        e2 = make_e2()
        z_world = torch.randn(BATCH, WORLD_DIM)
        action  = torch.randn(BATCH, ACTION_DIM)
        bias    = torch.randn(BATCH, ACTION_OBJECT_DIM)
        o_no_bias   = e2.action_object(z_world, action)
        o_with_bias = e2.action_object(z_world, action, action_bias=bias)
        assert torch.allclose(o_with_bias, o_no_bias + bias)

    def test_rollout_with_world_bias_shape(self):
        """rollout_with_world with action_bias produces Trajectory with correct action_object shapes."""
        e2 = make_e2()
        z_self  = torch.randn(1, SELF_DIM)
        z_world = torch.randn(1, WORLD_DIM)
        actions = torch.randn(1, 5, ACTION_DIM)
        bias    = torch.randn(1, ACTION_OBJECT_DIM)
        traj = e2.rollout_with_world(z_self, z_world, actions, compute_action_objects=True, action_bias=bias)
        assert traj.action_objects is not None
        for ao in traj.action_objects:
            assert ao.shape == (1, ACTION_OBJECT_DIM), f"action object shape {ao.shape}"

    def test_rollout_bias_affects_action_objects(self):
        """With action_bias, action_objects in trajectory should differ from bias=None."""
        e2 = make_e2()
        z_self  = torch.randn(1, SELF_DIM)
        z_world = torch.randn(1, WORLD_DIM)
        actions = torch.randn(1, 5, ACTION_DIM)
        bias    = torch.ones(1, ACTION_OBJECT_DIM) * 10.0  # large bias for detectability

        traj_no  = e2.rollout_with_world(z_self, z_world, actions, compute_action_objects=True)
        traj_yes = e2.rollout_with_world(z_self, z_world, actions, compute_action_objects=True, action_bias=bias)

        ao_no  = traj_no.action_objects[0]
        ao_yes = traj_yes.action_objects[0]
        assert not torch.allclose(ao_no, ao_yes), "action_bias had no effect on action_objects"
        # Difference should equal the bias
        assert torch.allclose(ao_yes - ao_no, bias, atol=1e-5)


# ------------------------------------------------------------------ #
# HippocampalModule.propose_trajectories() -- action_bias             #
# ------------------------------------------------------------------ #

class TestHippocampalActionBias:
    def test_propose_no_bias_baseline(self):
        """propose_trajectories without bias returns expected number of trajectories."""
        e2  = make_e2()
        hip = make_hippocampal(e2)
        z_world = torch.randn(1, WORLD_DIM)
        z_self  = torch.randn(1, SELF_DIM)
        trajs = hip.propose_trajectories(z_world, z_self, num_candidates=4)
        assert len(trajs) == 4

    def test_propose_with_bias_returns_same_count(self):
        """propose_trajectories with action_bias returns same number of trajectories."""
        e2  = make_e2()
        hip = make_hippocampal(e2)
        z_world    = torch.randn(1, WORLD_DIM)
        z_self     = torch.randn(1, SELF_DIM)
        action_bias = torch.randn(1, ACTION_OBJECT_DIM)
        trajs = hip.propose_trajectories(z_world, z_self, num_candidates=4, action_bias=action_bias)
        assert len(trajs) == 4

    def test_propose_action_bias_none_backward_compat(self):
        """Default (action_bias=None) must behave identically to pre-SD-016 calls."""
        e2  = make_e2()
        hip = make_hippocampal(e2)
        z_world = torch.randn(1, WORLD_DIM)
        z_self  = torch.randn(1, SELF_DIM)
        # Must not raise
        trajs = hip.propose_trajectories(z_world, z_self, num_candidates=2)
        assert len(trajs) == 2


# ------------------------------------------------------------------ #
# E3TrajectorySelector.score_trajectory() -- terrain_weight           #
# ------------------------------------------------------------------ #

class TestE3TerrainWeight:
    def _make_trajectory(self, batch: int = 1, horizon: int = 3) -> Trajectory:
        states = [torch.randn(batch, SELF_DIM) for _ in range(horizon + 1)]
        world_states = [torch.randn(batch, WORLD_DIM) for _ in range(horizon + 1)]
        actions = torch.randn(batch, horizon, ACTION_DIM)
        return Trajectory(states=states, actions=actions, world_states=world_states)

    def test_no_terrain_weight_unchanged(self):
        """score_trajectory(terrain_weight=None) must equal default call."""
        e3 = make_e3()
        traj = self._make_trajectory()
        s_default = e3.score_trajectory(traj)
        s_none    = e3.score_trajectory(traj, terrain_weight=None)
        assert torch.allclose(s_default, s_none)

    def test_terrain_weight_w_harm_scales_ethical_cost(self):
        """Reducing w_harm (terrain_weight[:,0]) should reduce the harm contribution to score."""
        e3   = make_e3()
        # High ethical lambda to make M(z) dominate
        e3.config.lambda_ethical = 10.0
        e3.config.rho_residue    = 0.0
        traj = self._make_trajectory()

        score_no_weight  = e3.score_trajectory(traj)
        # w_harm=0.5 (half scaling), w_goal=1.0 (unchanged)
        tw = torch.tensor([[0.5, 1.0]])  # [1, 2]
        score_low_harm = e3.score_trajectory(traj, terrain_weight=tw)

        # With lambda_ethical=10 and w_harm=0.5, the score should decrease
        # (harm cost halved -> lower J -> better trajectory)
        assert score_low_harm.item() < score_no_weight.item(), \
            f"Low w_harm should reduce score: {score_low_harm.item()} vs {score_no_weight.item()}"

    def test_terrain_weight_shape_batch1(self):
        """terrain_weight [1, 2] should work with batch-1 trajectory."""
        e3 = make_e3()
        traj = self._make_trajectory(batch=1)
        tw   = torch.sigmoid(torch.randn(1, 2))
        score = e3.score_trajectory(traj, terrain_weight=tw)
        assert score.shape[0] == 1

    def test_select_terrain_weight_passed_through(self):
        """select() with terrain_weight should produce different scores than without."""
        e3 = make_e3()
        e3.config.lambda_ethical = 5.0
        trajs = [self._make_trajectory() for _ in range(3)]

        result_no  = e3.select(trajs)
        # Extreme terrain_weight: w_harm=0.01, w_goal=0.99
        tw_low_harm = torch.tensor([[0.01, 0.99]])
        result_tw  = e3.select(trajs, terrain_weight=tw_low_harm)

        # Scores should differ (terrain_weight modifies M(z) contribution)
        # The selected index may or may not change, but the scores tensor must differ
        assert not torch.allclose(result_no.scores, result_tw.scores), \
            "terrain_weight had no effect on scores"


# ------------------------------------------------------------------ #
# REEAgent SD-016 wiring                                              #
# ------------------------------------------------------------------ #

class TestAgentSD016Wiring:
    def _make_agent(self, sd016_enabled: bool) -> REEAgent:
        config = REEConfig.from_dims(
            body_obs_dim=10, world_obs_dim=54, action_dim=4,
            sd016_enabled=sd016_enabled,
        )
        return REEAgent(config)

    def test_cue_signals_none_on_init_disabled(self):
        agent = self._make_agent(sd016_enabled=False)
        assert agent._cue_action_bias    is None
        assert agent._cue_terrain_weight is None

    def test_cue_signals_none_on_init_enabled(self):
        """Before any e1_tick, cue signals must be None even when sd016 enabled."""
        agent = self._make_agent(sd016_enabled=True)
        assert agent._cue_action_bias    is None
        assert agent._cue_terrain_weight is None

    def test_cue_signals_populated_after_e1_tick_enabled(self):
        agent = self._make_agent(sd016_enabled=True)
        agent.reset()
        obs_body  = torch.randn(1, 10)
        obs_world = torch.randn(1, 54)
        latent = agent.sense(obs_body, obs_world)
        _ = agent._e1_tick(latent)
        assert agent._cue_action_bias    is not None, "_cue_action_bias not set after e1_tick"
        assert agent._cue_terrain_weight is not None, "_cue_terrain_weight not set after e1_tick"
        assert agent._cue_action_bias.shape    == (1, 16)
        assert agent._cue_terrain_weight.shape == (1, 2)

    def test_cue_signals_none_after_e1_tick_disabled(self):
        """When sd016 disabled, cue signals must remain None after e1_tick."""
        agent = self._make_agent(sd016_enabled=False)
        agent.reset()
        obs_body  = torch.randn(1, 10)
        obs_world = torch.randn(1, 54)
        latent = agent.sense(obs_body, obs_world)
        _ = agent._e1_tick(latent)
        assert agent._cue_action_bias    is None
        assert agent._cue_terrain_weight is None

    def test_cue_signals_reset_on_episode_reset(self):
        agent = self._make_agent(sd016_enabled=True)
        agent.reset()
        obs_body  = torch.randn(1, 10)
        obs_world = torch.randn(1, 54)
        latent = agent.sense(obs_body, obs_world)
        _ = agent._e1_tick(latent)
        # Cue signals populated
        assert agent._cue_action_bias is not None

        # Now reset
        agent.reset()
        assert agent._cue_action_bias    is None, "_cue_action_bias not cleared on reset"
        assert agent._cue_terrain_weight is None, "_cue_terrain_weight not cleared on reset"

    def test_cue_signals_detached(self):
        """Cached cue signals must have no grad_fn (detached from compute graph)."""
        agent = self._make_agent(sd016_enabled=True)
        agent.reset()
        obs_body  = torch.randn(1, 10)
        obs_world = torch.randn(1, 54)
        latent = agent.sense(obs_body, obs_world)
        _ = agent._e1_tick(latent)
        assert agent._cue_action_bias.grad_fn    is None, "action_bias not detached"
        assert agent._cue_terrain_weight.grad_fn is None, "terrain_weight not detached"

    def test_full_act_loop_sd016_disabled(self):
        """Full act() loop with sd016=False must not raise."""
        agent = self._make_agent(sd016_enabled=False)
        agent.reset()
        obs = torch.randn(1, 64)  # body_obs_dim + world_obs_dim = 10+54
        action = agent.act(obs)
        assert action.shape[-1] == 4

    def test_full_act_loop_sd016_enabled(self):
        """Full act() loop with sd016=True must not raise."""
        agent = self._make_agent(sd016_enabled=True)
        agent.reset()
        obs = torch.randn(1, 64)
        action = agent.act(obs)
        assert action.shape[-1] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
