"""Contract: a single merged optimizer over agent.parameters() can backprop an
UNDETACHED combined E1+E2+E3 objective across consecutive agent steps when
REEConfig.detach_carried_prev_action is True (ARC-021 H2 MERGED arm).

Background (chip-20260911-arc021-h2-contextmemory-inplace-write, pre-flight
2026-09-24). select_action stores its emitted action in _last_action with
requires_grad=True; the next sense() passes it as prev_action into the SD-007
reafference correction, so step t's z_world carried step t-1's selection graph
(E1 prior -> ContextMemory.read, E2, E3). A merged-optimizer backward on an
undetached z_world loss then walked back across step t-1's Adam step (an
in-place parameter update) and crashed with "modified by an inplace operation".
ContextMemory.write's .data write was the originally suspected mutator and was
measured NOT to be the cause.

What these tests guard (they are NOT stubbed -- they drive the real agent loop
exactly like experiments/_scratch/arc021_h2_merged_optimizer_runnability_probe.py):

  1. flag ON  -> MERGED arm runs N steps clean, AND e1.context_memory.memory
     still receives a gradient on at least one step (it remains a real
     participant of the merged objective; the fix does not exclude it).
  2. flag OFF -> the legacy behaviour is preserved: the MERGED arm still
     crashes on a later step (the defect this flag exists to cut). If this ever
     starts passing, the default path changed -- investigate, do not delete.
  3. flag ON leaves forward values bit-identical: z_world after N steps with a
     SEPARATE-style (no-backward) rollout is equal ON vs OFF.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

N_STEPS = 4


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=10, num_hazards=3, num_resources=5, hazard_harm=0.02,
        env_drift_interval=10, env_drift_prob=0.05, proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03, proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _make_agent(env, detach: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, reafference_action_dim=env.action_dim,
        detach_carried_prev_action=detach,
    )
    assert cfg.detach_carried_prev_action is detach  # from_dims reaches the field
    return REEAgent(cfg)


def _run(detach: bool, merged: bool, n_steps: int = N_STEPS):
    """Drive the spark-driver train loop. Returns (steps_clean, mem_grad_steps, z_world_trace)."""
    torch.manual_seed(0)
    random.seed(0)
    env = _make_env(0)
    agent = _make_agent(env, detach)
    world_dim = agent.config.latent.world_dim
    opt = optim.Adam(agent.parameters(), lr=1e-3) if merged else None
    agent.train()
    _, obs_dict = env.reset()
    agent.reset()
    z_self_t = None
    mem = agent.e1.context_memory.memory
    mem_grad_steps = 0
    z_trace = []
    for i in range(n_steps):
        obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
        obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
        if agent._current_latent is not None:
            z_self_t = agent._current_latent.z_self.detach().clone()
        latent = agent.sense(obs_body, obs_world)
        z_trace.append(latent.z_world.detach().clone())
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks["e1_tick"]
            else torch.zeros(1, world_dim, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)
        if z_self_t is not None:
            agent.record_transition(z_self_t, action, latent.z_self.detach())
        _, reward, done, info, obs_dict = env.step(action)
        harm_signal = float(reward) if reward < 0 else 0.0
        if merged:
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            harm_target = torch.tensor([[1.0 if harm_signal < 0 else 0.0]])
            harm_loss = F.mse_loss(agent.e3.harm_eval(latent.z_world), harm_target)
            combined = (
                (e1_loss if e1_loss.requires_grad else torch.tensor(0.0))
                + (e2_loss if e2_loss.requires_grad else torch.tensor(0.0))
                + harm_loss
            )
            if combined.requires_grad:
                opt.zero_grad()
                try:
                    combined.backward()
                except RuntimeError:
                    return i, mem_grad_steps, z_trace
                if mem.grad is not None and float(mem.grad.abs().sum()) > 0.0:
                    mem_grad_steps += 1
                opt.step()
        agent.update_residue(harm_signal)
        if done:
            break
    return n_steps, mem_grad_steps, z_trace


def test_merged_optimizer_runs_clean_with_flag_on():
    steps_clean, mem_grad_steps, _ = _run(detach=True, merged=True)
    assert steps_clean == N_STEPS, (
        f"MERGED arm crashed on step {steps_clean} with detach_carried_prev_action=True"
    )
    assert mem_grad_steps >= 1, (
        "e1.context_memory.memory received no gradient -- it must remain a real "
        "participant of the merged objective"
    )


def test_merged_optimizer_legacy_default_still_crashes():
    steps_clean, _, _ = _run(detach=False, merged=True)
    assert steps_clean < N_STEPS, (
        "default path no longer carries the cross-step graph; the legacy "
        "behaviour changed -- investigate before touching this test"
    )


def test_flag_is_forward_bit_identical():
    _, _, z_off = _run(detach=False, merged=False)
    _, _, z_on = _run(detach=True, merged=False)
    assert len(z_off) == len(z_on) == N_STEPS
    for a, b in zip(z_off, z_on):
        assert torch.equal(a, b)


def test_default_is_off():
    assert REEConfig().detach_carried_prev_action is False
