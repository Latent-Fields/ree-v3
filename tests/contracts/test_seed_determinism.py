"""C3: same seed -> same first N actions and same z_world checksum.

Catches accidental introductions of unseeded randomness (a fresh nn.Parameter
initialised post-set_all_seeds, an unseeded torch op, a reliance on Python
dict iteration order, etc.).
"""

import torch

from ree_core.agent import REEAgent

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_loop import run_episode

N_STEPS = 10


def _run(seed: int):
    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)
    actions = run_episode(agent, env, steps=N_STEPS)
    z_world = agent._current_latent.z_world.detach()
    checksum = float(z_world.abs().sum().item())
    return actions, checksum


def test_same_seed_same_actions():
    a1, _ = _run(0)
    a2, _ = _run(0)
    assert a1 == a2, f"nondeterministic actions: {a1} vs {a2}"


def test_same_seed_same_z_world_checksum():
    _, c1 = _run(0)
    _, c2 = _run(0)
    assert c1 == c2, f"z_world checksum drifted across identical runs: {c1} vs {c2}"


def test_different_seeds_diverge():
    """Sanity: two different seeds don't collapse to identical trajectories."""
    a1, c1 = _run(0)
    a2, c2 = _run(1)
    assert a1 != a2 or c1 != c2, \
        "two different seeds produced bit-identical results -- seeding is broken"
