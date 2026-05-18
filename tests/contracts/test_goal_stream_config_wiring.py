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


def test_from_dims_goal_stream_flag_enables_coherent_bundle():
    cfg = _make_cfg(
        goal_stream_enabled=True,
        goal_weight=0.42,
        wanting_weight=0.33,
        benefit_threshold=0.07,
        schema_wanting_threshold=0.12,
        schema_wanting_gain=0.66,
    )

    assert cfg.goal_stream_enabled is True
    assert cfg.goal.z_goal_enabled is True
    assert cfg.goal.goal_weight == 0.42
    assert cfg.e3.goal_weight == 0.42
    assert cfg.goal.benefit_threshold == 0.07
    assert cfg.e1.schema_wanting_enabled is True
    assert cfg.schema_wanting_threshold == 0.12
    assert cfg.schema_wanting_gain == 0.66
    assert cfg.hippocampal.wanting_weight == 0.33
    assert cfg.latent.use_resource_encoder is True
    assert cfg.latent.z_resource_dim == cfg.goal.goal_dim
    assert cfg.use_mech295_liking_bridge is True
    assert cfg.surprise_gated_replay is True
    assert cfg.use_mech307_split_surprise is True
    assert cfg.use_mech307_schema_multichannel is True
    assert cfg.use_mech307_predicted_location_write is True
    assert cfg.use_mech307_consumer_conjunction_read is True


def test_goal_stream_preset_matches_heartbeat_bundle_defaults():
    cfg = REEConfig.goal_stream(
        body_obs_dim=12,
        world_obs_dim=54,
        action_dim=4,
        self_dim=16,
        world_dim=16,
    )

    assert cfg.goal_stream_enabled is True
    assert cfg.goal.goal_weight == 0.5
    assert cfg.e3.goal_weight == 0.5
    assert cfg.hippocampal.wanting_weight == 0.5
    assert cfg.goal.benefit_threshold == 0.05
    assert cfg.goal.drive_weight == 2.0
    assert cfg.schema_wanting_threshold == 0.10
    assert cfg.schema_wanting_gain == 0.60
    assert cfg.mech295_min_drive_to_fire == 0.01
    assert cfg.mech295_min_z_goal_norm_to_fire == 0.03
    assert cfg.mech307_conjunction_wanting_threshold == 0.10
    assert cfg.mech307_conjunction_liking_threshold == 0.05
    assert cfg.mech307_conjunction_z_beta_threshold == 0.10
