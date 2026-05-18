"""C4: BG gating contracts (MECH-090 Layer 1 + MECH-091).

- MECH-090 Layer 1: when beta_gate is elevated and a committed trajectory
  exists, select_action() should step through actions[:, idx, :] advancing
  _committed_step_idx each non-E3-tick.
- MECH-091: when beta_gate is elevated and z_harm_a.norm() exceeds
  urgency_interrupt_threshold, select_action() should release the gate and
  reset _committed_step_idx to 0.

These are interface-level contracts for the committed-action state machine.
Threshold values (urgency_interrupt_threshold, specific trajectory shapes)
come from the config under test, NOT from EXQ evidence.
"""

import torch

from ree_core.agent import REEAgent
from ree_core.predictors.e2_fast import Trajectory

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_loop import _obs_tensors


HORIZON = 3


def _seed_latent_and_commit(agent: REEAgent, env, *, z_harm_a_norm: float = 0.0):
    """Drive one sense() so _current_latent exists, then manually install a
    committed trajectory and elevate beta_gate. Returns (action_seq,)."""
    _flat, obs_dict = env.reset()
    body, world = _obs_tensors(obs_dict)
    with torch.no_grad():
        agent.sense(obs_body=body, obs_world=world)

    # Build a deterministic 3-step one-hot action sequence.
    action_dim = agent.config.e2.action_dim
    action_seq = torch.zeros(1, HORIZON, action_dim)
    for t in range(HORIZON):
        action_seq[0, t, t % action_dim] = 1.0

    z_self_dim = agent.config.latent.self_dim
    states = [torch.zeros(1, z_self_dim) for _ in range(HORIZON + 1)]
    traj = Trajectory(states=states, actions=action_seq)

    agent.e3._committed_trajectory = traj
    agent.beta_gate.elevate()
    agent._committed_step_idx = 0
    agent._last_action = action_seq[:, 0, :].clone()

    if z_harm_a_norm > 0 and agent._current_latent is not None:
        # Synthesise z_harm_a with the requested norm so MECH-091 can trigger.
        dim = agent.config.latent.z_harm_a_dim or 16
        vec = torch.ones(1, dim)
        vec = vec / vec.norm() * z_harm_a_norm
        agent._current_latent.z_harm_a = vec

    return action_seq


def test_committed_trajectory_steps_through_actions():
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)
    agent.reset()

    action_seq = _seed_latent_and_commit(agent, env)
    ticks_hold = {"e1_tick": False, "e2_tick": False, "e3_tick": False}

    # Three held ticks: _committed_step_idx should advance 0 -> 1 -> 2 -> 3,
    # and the emitted action should match actions[:, step_idx, :] at each tick.
    observed = []
    for _ in range(HORIZON):
        with torch.no_grad():
            a = agent.select_action(candidates=[], ticks=ticks_hold)
        observed.append(a.argmax(dim=-1).item())

    expected = [int(action_seq[0, t].argmax().item()) for t in range(HORIZON)]
    assert observed == expected, \
        f"committed stepping: observed {observed} vs expected {expected}"
    assert agent._committed_step_idx == HORIZON, \
        f"step idx {agent._committed_step_idx} after {HORIZON} ticks"


def test_urgency_interrupt_releases_gate_and_resets_idx():
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    # Enable affective harm stream so z_harm_a is allocated in LatentState.
    cfg = make_tiny_config(env, use_harm_stream=True, use_affective_harm_stream=True)
    agent = REEAgent(cfg)
    agent.reset()

    threshold = cfg.e3.urgency_interrupt_threshold
    assert threshold > 0, "urgency_interrupt_threshold must be positive"

    _seed_latent_and_commit(agent, env, z_harm_a_norm=threshold * 2.0)
    # Advance step idx so the reset is observable.
    agent._committed_step_idx = 2
    assert agent.beta_gate.is_elevated

    ticks_hold = {"e1_tick": False, "e2_tick": False, "e3_tick": False}
    with torch.no_grad():
        agent.select_action(candidates=[], ticks=ticks_hold)

    assert not agent.beta_gate.is_elevated, \
        "MECH-091: beta_gate should release on urgency interrupt"
    assert agent._committed_step_idx == 0, \
        f"MECH-091: _committed_step_idx should reset to 0, got {agent._committed_step_idx}"
