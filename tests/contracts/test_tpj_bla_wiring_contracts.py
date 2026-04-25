"""Focused contracts for the TPJ comparator and BLA consumer wiring."""

import torch

from ree_core.agent import REEAgent
from ree_core.predictors.e2_fast import Trajectory

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_env import make_tiny_env


def _batch(obs: torch.Tensor) -> torch.Tensor:
    return obs.unsqueeze(0) if obs.dim() == 1 else obs


def test_tpj_comparator_resolves_on_next_sense():
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env, use_tpj_comparator=True)
    agent = REEAgent(cfg)
    agent.reset()

    _flat, obs = env.reset()
    body = _batch(obs["body_state"])
    world = _batch(obs["world_state"])

    action = agent.act_with_split_obs(body, world)
    assert agent._tpj_predicted_z_self is not None

    action_idx = int(action.argmax(dim=-1).item())
    _flat_next, _harm, _done, _info, next_obs = env.step(action_idx)

    next_body = _batch(next_obs["body_state"])
    next_world = _batch(next_obs["world_state"])
    agent.sense(next_body, next_world)

    assert agent._tpj_predicted_z_self is None
    assert agent._tpj_last_agency_signal is not None
    assert agent._tpj_last_is_self_caused is not None
    agency = float(agent._tpj_last_agency_signal.squeeze().item())
    assert 0.0 <= agency <= 1.0, f"agency signal out of range: {agency}"
    assert agent._tpj_last_is_self_caused.dtype == torch.bool


def test_bla_wires_retrieval_bias_reverse_replay_and_context_remap():
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(
        env,
        use_amygdala_analog=True,
        use_affective_harm_stream=True,
        replay_diversity_enabled=True,
    )
    agent = REEAgent(cfg)
    agent.reset()

    z_self = torch.zeros(1, cfg.latent.self_dim)
    z_world = torch.zeros(1, cfg.latent.world_dim)
    actions = torch.zeros(1, 1, cfg.e2.action_dim)
    actions[0, 0, 0] = 1.0
    hi = Trajectory(
        states=[z_self.clone(), z_self.clone()],
        actions=actions.clone(),
        world_states=[z_world.clone(), z_world.clone()],
        arousal_tag=1.0,
        memory_strength=2.0,
    )
    lo = Trajectory(
        states=[z_self.clone(), z_self.clone()],
        actions=actions.clone(),
        world_states=[z_world.clone(), z_world.clone()],
        arousal_tag=0.0,
        memory_strength=1.0,
    )
    agent.hippocampal.record_exploration_trajectory(hi)
    agent.hippocampal.record_exploration_trajectory(lo)

    _flat, obs = env.reset()
    body = _batch(obs["body_state"])
    world = _batch(obs["world_state"])
    harm_a = torch.ones(1, cfg.latent.harm_obs_a_dim)
    agent._harm_a_pred_prev = torch.full((1, cfg.latent.z_harm_a_dim), -5.0)

    mem_before = agent.e1.context_memory.memory.detach().clone()
    agent.sense(body, world, obs_harm_a=harm_a)

    bla_out = agent._bla_last_output
    assert bla_out is not None
    assert bla_out.retrieval_bias is not None
    assert bla_out.retrieval_bias.shape[0] == 2
    assert float(bla_out.retrieval_bias[0]) > float(bla_out.retrieval_bias[1])
    assert len(bla_out.remap_signal) > 0
    assert not torch.allclose(mem_before, agent.e1.context_memory.memory)

    recent = torch.zeros(2, 1, cfg.latent.world_dim)
    hi_count = 0
    lo_count = 0
    for _ in range(64):
        traj = agent.hippocampal.diverse_replay(
            recent,
            num_replay_steps=1,
            mode="reverse",
            retrieval_bias=bla_out.retrieval_bias,
        )[0]
        if float(traj.arousal_tag) > 0.5:
            hi_count += 1
        else:
            lo_count += 1

    assert hi_count > lo_count, (
        f"BLA retrieval bias did not favour tagged traces: hi={hi_count} lo={lo_count}"
    )
