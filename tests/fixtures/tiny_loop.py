"""Minimal env+agent step helper for contract tests.

Mirrors the inner loop used by real experiment scripts
(see e.g. experiments/v3_exq_433a_*): sense -> clock.advance -> e1_tick ->
generate_trajectories -> select_action -> env.step -> update_residue.

Kept small so contract tests can focus on the specific assertion they
own rather than re-implementing the stepping dance.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2


def _obs_tensors(obs_dict) -> Tuple[torch.Tensor, torch.Tensor]:
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body, world


def step_once(
    agent: REEAgent,
    env: CausalGridWorldV2,
    obs_dict,
    *,
    update_residue: bool = True,
    hypothesis_tag: bool = False,
) -> Tuple[torch.Tensor, int, dict, dict]:
    """Run one full agent+env step under no_grad.

    Returns (action, action_idx, ticks, next_obs_dict).
    """
    body, world = _obs_tensors(obs_dict)
    sense_kwargs = {"obs_body": body, "obs_world": world}
    latent_cfg = agent.config.latent
    obs_harm = obs_dict.get("harm_obs") if isinstance(obs_dict, dict) else None
    if obs_harm is not None and getattr(latent_cfg, "use_harm_stream", False):
        if obs_harm.dim() == 1:
            obs_harm = obs_harm.unsqueeze(0)
        sense_kwargs["obs_harm"] = obs_harm
    obs_harm_a = obs_dict.get("harm_obs_a") if isinstance(obs_dict, dict) else None
    if obs_harm_a is not None and getattr(latent_cfg, "use_affective_harm_stream", False):
        if obs_harm_a.dim() == 1:
            obs_harm_a = obs_harm_a.unsqueeze(0)
        sense_kwargs["obs_harm_a"] = obs_harm_a

    with torch.no_grad():
        latent = agent.sense(**sense_kwargs)
        ticks = agent.clock.advance()
        world_dim = agent.config.latent.world_dim
        if ticks.get("e1_tick", True):
            e1_prior = agent._e1_tick(latent)
        else:
            e1_prior = torch.zeros(1, world_dim)
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)

    action_idx = int(action.argmax(dim=-1).item())
    _flat, harm_signal, _done, _info, next_obs = env.step(action_idx)

    if update_residue:
        agent.update_residue(float(harm_signal), hypothesis_tag=hypothesis_tag)

    return action, action_idx, ticks, next_obs


def run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
) -> list:
    """Reset env+agent and run `steps` steps. Returns list of action indices."""
    agent.reset()
    _flat, obs_dict = env.reset()
    action_indices: list[int] = []
    for _ in range(steps):
        _action, action_idx, _ticks, obs_dict = step_once(agent, env, obs_dict)
        action_indices.append(action_idx)
    return action_indices
