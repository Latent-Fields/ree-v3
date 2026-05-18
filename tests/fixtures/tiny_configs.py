"""Tiny REEConfig factory for regression tests.

Always builds via REEConfig.from_dims(env.body_obs_dim, env.world_obs_dim, ACTION_DIM)
so the baseline stays aligned with the real factory path used by experiment scripts.
Callers overlay flags via kwargs.
"""

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

ACTION_DIM = 4


def make_tiny_config(env: CausalGridWorldV2, **overrides) -> REEConfig:
    kwargs = dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=16,
        world_dim=16,
    )
    kwargs.update(overrides)
    return REEConfig.from_dims(**kwargs)
