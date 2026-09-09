"""Contracts for `experiments/_lib/gradient_reencode.py`.

The mechanism this module exists to provide (see its own module docstring for
the full background): a training loss built on a rolling buffer of PAST
latents cannot move `agent.latent_stack` (the encoder) when that buffer stores
its entries `.detach().clone()`'d -- the deliberate, codebase-wide convention
(`ree_core/agent.py:5835-5836`) that severed the graph V3-EXQ-972a measured
frozen (failure_autopsy_dv-headroom-diagnostics-cluster_2026-09-07, REE_assembly
cb4a71fbd9; user-ratified 2026-09-08 by governance-20260908-0703). This file
pins two things: (1) the DEFECT reproduces at the unit level -- training on a
detached buffer genuinely leaves the encoder bit-for-bit unmoved; (2) the FIX
mechanism works -- training on `reencode_batch_z`'s output moves it. It also
pins that the re-encode path reproduces the live `sense()` forward pass
exactly (no silent numerical drift), so adopting it in a driver would not
change what the encoder computes, only whether its gradient is retained.

Wiring the 970/971/972/970a drivers to this module is a separate
`/queue-experiment`-mediated follow-up (CLAUDE.md "Experiment Scripts": logic
changes to an experiment script must go through that skill) -- these contracts
pin the shared mechanism only, not any driver's use of it.
"""

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments._lib.gradient_reencode import (
    ObservationCapture,
    reencode_batch_z,
    reencode_z_with_grad,
    latent_stack_param_snapshot,
    latent_stack_moved,
)

SEED = 3
NUM_SLOTS = 16


def _build_agent(seed: int = SEED):
    torch.manual_seed(seed)
    env = CausalGridWorldV2(seed=seed, size=5, num_hazards=1, num_resources=1)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=16,
        world_dim=16,
        contextmemory_write_selection="gumbel_learned",
        contextmemory_write_addressing_loss_weight=0.5,
    )
    agent = REEAgent(cfg)
    assert agent.e1.context_memory.write_addr_tagger is not None, (
        "gumbel_learned did not construct write_addr_tagger")
    agent.reset()
    return env, agent, cfg


def _step_and_capture(env, agent, n_steps, capture: bool):
    """Drive `n_steps` env ticks. When `capture`, snapshot each observation
    via ObservationCapture BEFORE calling sense() (matching the module's
    documented ordering); always also collect the DETACHED buffered latent
    the drivers currently store (970a:876's pattern), for the baseline case.
    """
    obs_dict = env.reset()[1]
    captures = []
    detached_rows = []
    for _ in range(n_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        if capture:
            captures.append(ObservationCapture.capture(agent, obs_body, obs_world))
        latent = agent.sense(obs_body, obs_world)
        detached_rows.append(
            torch.cat([latent.z_self.detach(), latent.z_world.detach()], dim=-1)
        )
        action = torch.zeros(1, env.action_dim)
        action[0, 0] = 1.0
        _, _, done, _info, obs_dict = env.step(action)
        if done:
            obs_dict = env.reset()[1]
    return captures, detached_rows


def test_reencode_reproduces_live_sense_output_exactly():
    """Re-encoding a just-captured observation must match the original
    sense() output bit-for-bit (same weights, same inputs, same recurrent
    context) -- the fix must not change what the encoder computes."""
    env, agent, _ = _build_agent()
    obs_dict = env.reset()[1]
    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]

    cap = ObservationCapture.capture(agent, obs_body, obs_world)
    live_latent = agent.sense(obs_body, obs_world)

    z_self_re, z_world_re = reencode_z_with_grad(agent, cap)

    assert torch.equal(z_self_re, live_latent.z_self.detach())
    assert torch.equal(z_world_re, live_latent.z_world.detach())


def test_baseline_detached_buffer_does_not_move_encoder():
    """Reproduces the confirmed defect at unit scale: training
    write_addr_tagger's addressing loss on a batch pulled from the
    conventional DETACHED buffer leaves every agent.latent_stack parameter
    bit-for-bit unmoved, even though the optimizer's param list includes
    them (mirrors 970a:984-986's standard_params construction)."""
    env, agent, _ = _build_agent()
    _, detached_rows = _step_and_capture(env, agent, n_steps=6, capture=False)

    before = latent_stack_param_snapshot(agent)

    standard_params = [p for n, p in agent.named_parameters()
                        if "harm_eval_head" not in n and "context_memory.memory" not in n]
    optimizer = torch.optim.Adam(standard_params, lr=1e-2)

    batch = torch.cat(detached_rows, dim=0)
    loss = agent.e1.context_memory.compute_write_addressing_loss(batch)
    assert loss.requires_grad, "tagger loss should build a graph reaching the tagger"

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not latent_stack_moved(agent, before), (
        "baseline (detached-buffer) training path moved latent_stack -- "
        "the defect this contract pins no longer reproduces; re-check "
        "whether the drivers' actual buffering still matches this pattern "
        "before treating this as a regression in the FIX rather than a "
        "change in the baseline it is contrasted against."
    )


def test_reencode_batch_moves_encoder():
    """The fix mechanism: training the SAME addressing loss on
    reencode_batch_z's output (re-run through the encoder at training time,
    per this module's docstring) DOES move agent.latent_stack."""
    env, agent, _ = _build_agent()
    captures, _ = _step_and_capture(env, agent, n_steps=6, capture=True)

    before = latent_stack_param_snapshot(agent)

    standard_params = [p for n, p in agent.named_parameters()
                        if "harm_eval_head" not in n and "context_memory.memory" not in n]
    optimizer = torch.optim.Adam(standard_params, lr=1e-2)

    batch = reencode_batch_z(agent, captures)
    assert batch.requires_grad, "re-encoded batch should be connected to the encoder"

    loss = agent.e1.context_memory.compute_write_addressing_loss(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert latent_stack_moved(agent, before), (
        "reencode_batch_z's output did not move latent_stack after a "
        "training step -- the fix mechanism is not delivering gradient to "
        "the encoder as designed"
    )


def test_reencode_does_not_touch_live_recurrent_state():
    """Re-encoding buffered observations must not perturb the agent's actual
    online trajectory: agent._current_latent after a batch of reencode calls
    must be exactly what it was before (only agent.sense() may advance it)."""
    env, agent, _ = _build_agent()
    captures, _ = _step_and_capture(env, agent, n_steps=4, capture=True)

    before_z_self = agent._current_latent.z_self.clone()
    before_z_world = agent._current_latent.z_world.clone()

    _ = reencode_batch_z(agent, captures)

    assert torch.equal(agent._current_latent.z_self, before_z_self)
    assert torch.equal(agent._current_latent.z_world, before_z_world)
