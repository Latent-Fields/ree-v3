"""C2: REEAgent boots under representative feature-flag combinations.

Many V3 features are default-off for backward compatibility. This matrix
catches wiring regressions in each flag's constructor path before they
burn queue time. One step with sense() is enough for "does this config
instantiate and encode an observation"; full behavioural assays live in
individual subsystem contract tests and EXQs.

We deliberately DO NOT assert post-flag behaviour here -- only that the
agent constructs and runs a sense() step. Flag semantics are tested in
dedicated contract files (bg_gating, imagined_acted_isolation, etc.).
"""

import pytest
import torch

from ree_core.agent import REEAgent

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config


FLAG_MATRIX = [
    ("baseline", {}),
    ("resource_proximity_head",
     {"use_resource_proximity_head": True}),
    ("harm_stream",
     {"use_harm_stream": True}),
    ("affective_harm_stream",
     {"use_harm_stream": True, "use_affective_harm_stream": True}),
    ("beta_gate_bistable",
     {}),  # set via nested config below
    ("sleep_sws_rem",
     {"sws_enabled": True, "rem_enabled": True}),
    ("dacc_and_salience",
     {"use_dacc": True, "use_salience_coordinator": True}),
    ("aic_pcc_pacc",
     {"use_aic_analog": True, "use_pcc_analog": True, "use_pacc_analog": True}),
]


def _build(name, overrides):
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env, **overrides)
    if name == "beta_gate_bistable":
        cfg.heartbeat.beta_gate_bistable = True
    return env, cfg


@pytest.mark.parametrize("name,overrides", FLAG_MATRIX, ids=[m[0] for m in FLAG_MATRIX])
def test_agent_boots_under_flags(name, overrides):
    env, cfg = _build(name, overrides)
    agent = REEAgent(cfg)
    agent.reset()

    _flat, obs_dict = env.reset()
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)

    kwargs = {"obs_body": body, "obs_world": world}
    obs_harm = obs_dict.get("harm_obs")
    if obs_harm is not None and getattr(cfg.latent, "use_harm_stream", False):
        if obs_harm.dim() == 1:
            obs_harm = obs_harm.unsqueeze(0)
        kwargs["obs_harm"] = obs_harm
    obs_harm_a = obs_dict.get("harm_obs_a")
    if obs_harm_a is not None and getattr(cfg.latent, "use_affective_harm_stream", False):
        if obs_harm_a.dim() == 1:
            obs_harm_a = obs_harm_a.unsqueeze(0)
        kwargs["obs_harm_a"] = obs_harm_a

    with torch.no_grad():
        latent = agent.sense(**kwargs)

    assert latent is not None
    assert torch.isfinite(latent.z_world).all(), f"{name}: z_world NaN/Inf"
    assert torch.isfinite(latent.z_self).all(), f"{name}: z_self NaN/Inf"
