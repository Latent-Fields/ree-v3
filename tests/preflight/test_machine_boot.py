"""P3: machine boot smoke.

Instantiate a default REEAgent on the current host, reset a tiny env, run
one sense+select_action step. Catches the class of "setup crash on machine
X, immediate requeue" failures before the runner burns queue time.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from tests.fixtures.seed_utils import set_all_seeds  # noqa: E402
from tests.fixtures.tiny_env import make_tiny_env  # noqa: E402
from tests.fixtures.tiny_configs import make_tiny_config  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402


def _obs_to_tensors(obs_dict):
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body, world


def test_agent_instantiates_on_host():
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)
    assert agent.config.e2.action_dim == 4


def test_tiny_episode_sense_produces_latent():
    """Agent.reset() + env.reset() + one sense() on this host.

    select_action() requires candidate trajectories and a clock tick dict that
    this preflight does not build -- the contract-layer BG gating test (PR 2)
    covers that path. Here we just verify the agent boots and encodes one obs
    without crashing or producing NaN, which catches the "setup crash on
    machine X" failure class the preflight layer targets.
    """
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)
    agent.reset()

    _flat, obs_dict = env.reset()
    body, world = _obs_to_tensors(obs_dict)

    with torch.no_grad():
        latent = agent.sense(obs_body=body, obs_world=world)

    assert latent is not None, "sense() returned None"
    assert latent.z_world is not None, "latent.z_world is None after sense()"
    assert torch.isfinite(latent.z_world).all(), "z_world contains NaN or Inf"
    assert latent.z_world.shape[-1] == cfg.latent.world_dim, \
        f"z_world last-dim {latent.z_world.shape[-1]} vs latent.world_dim {cfg.latent.world_dim}"
