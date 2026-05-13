"""Goal-stream configuration wiring contracts.

`REEConfig.from_dims(goal_weight=...)` is the factory path used by experiment
scripts. E3 trajectory scoring reads `config.e3.goal_weight`, not
`config.goal.goal_weight`, so both copies must stay in sync.
"""

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def _make_cfg(**overrides):
    kwargs = dict(
        body_obs_dim=12,
        world_obs_dim=54,
        action_dim=4,
        self_dim=16,
        world_dim=16,
    )
    kwargs.update(overrides)
    return REEConfig.from_dims(**kwargs)


def test_from_dims_goal_weight_reaches_e3_selector_config():
    cfg = _make_cfg(z_goal_enabled=True, goal_weight=0.37)

    assert cfg.goal.goal_weight == 0.37
    assert cfg.e3.goal_weight == 0.37

    agent = REEAgent(cfg)
    assert agent.e3.config.goal_weight == 0.37


def test_from_dims_goal_weight_zero_disables_e3_goal_scoring():
    cfg = _make_cfg(z_goal_enabled=True, goal_weight=0.0)

    assert cfg.goal.goal_weight == 0.0
    assert cfg.e3.goal_weight == 0.0


def test_default_goal_weight_is_consistent_across_goal_and_e3():
    cfg = _make_cfg(z_goal_enabled=True)

    assert cfg.e3.goal_weight == cfg.goal.goal_weight
    assert cfg.e3.goal_weight > 0.0
