"""C1: default REEAgent survives a short episode.

Gross integration regression guard: boots agent, runs 10 steps through the
real sense -> clock -> e1_tick -> generate_trajectories -> select_action ->
env.step -> update_residue loop, checks the latent stays finite end to end.
"""

import torch

from ree_core.agent import REEAgent

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_loop import run_episode


def test_default_agent_survives_10_steps():
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)

    actions = run_episode(agent, env, steps=10)

    assert len(actions) == 10
    assert all(0 <= a < cfg.e2.action_dim for a in actions), \
        f"action index out of range: {actions}"

    latent = agent._current_latent
    assert latent is not None
    assert torch.isfinite(latent.z_world).all(), "z_world contains NaN/Inf after 10 steps"
    assert torch.isfinite(latent.z_self).all(), "z_self contains NaN/Inf after 10 steps"
