"""Contract: REEAgent.offline_integration() replays E1 on [z_self ++ z_world]
concatenated on the FEATURE axis, for equal and unequal self/world dims.

Defect (found by V3-EXQ-996, 2026-09-03): offline_integration() built the E1
replay buffer as ``torch.cat([s, w])`` with torch.cat's default dim=0. The two
experience buffers hold ``[1, D]`` tensors (leading batch dim, appended from
``latent_state.z_self / z_world`` in sense()), so:

  * self_dim == world_dim  -> a silent ``[2, D]`` tensor instead of ``[1, 2D]``;
    E1.integrate_experience() then crashed with
    ``RuntimeError: mat1 and mat2 shapes cannot be multiplied`` once the
    buffer exceeded 10 entries (the branch's own threshold).
  * self_dim != world_dim  -> torch.cat raised at the concatenation itself.

The sibling three lines above the buffer append (``total_state = torch.cat(
[z_self, z_world], dim=-1)``) and REEAgent.compute_prediction_loss() both use
the feature axis. No experiment in ree-v3/experiments/ ever called
offline_integration(), which is why the defect survived.

Second half of the same defect: with ``dim=-1`` alone the buffer entries are
``[1, 2D]``; E1.integrate_experience() torch.stack()s them into ``[T, 1, 2D]``
and, lacking a 3-D branch, read that as T batch rows of horizon 0 -- a silently
degenerate replay rather than a crash. integrate_experience() now transposes a
3-D stack to ``[B, T, D]``; the legacy 1-D-entry path (``[T, D]`` ->
``unsqueeze(0)``) is unchanged, which the equivalence test below pins.

These tests fail on the pre-fix code and pass on the fix.
"""

import math

import pytest
import torch

from ree_core.agent import REEAgent

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_loop import step_once

# offline_integration()'s E1-replay branch fires only when the buffer holds
# MORE than 10 entries; 15 steps puts us safely past the gate.
N_STEPS = 15


def _agent_with_filled_buffers(self_dim: int, world_dim: int, seed: int = 0) -> REEAgent:
    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    cfg = make_tiny_config(env, self_dim=self_dim, world_dim=world_dim)
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    for _ in range(N_STEPS):
        _a, _i, _t, obs_dict = step_once(agent, env, obs_dict)
    assert len(agent._world_experience_buffer) == N_STEPS
    assert len(agent._self_experience_buffer) == N_STEPS
    return agent


@pytest.mark.parametrize(
    "self_dim,world_dim",
    [
        pytest.param(16, 16, id="self_dim==world_dim"),
        pytest.param(16, 24, id="self_dim!=world_dim"),
    ],
)
def test_offline_integration_runs_e1_replay_once_buffer_is_full(self_dim, world_dim):
    agent = _agent_with_filled_buffers(self_dim, world_dim)
    # Pin the precondition the defect depended on: buffer entries carry a
    # leading batch dim, so a dim=0 cat is the wrong axis.
    assert tuple(agent._self_experience_buffer[0].shape) == (1, self_dim)
    assert tuple(agent._world_experience_buffer[0].shape) == (1, world_dim)

    metrics = agent.offline_integration()  # crashed pre-fix

    assert "e1_integration_loss" in metrics, (
        "offline_integration() did not reach the E1 replay branch: "
        f"keys={sorted(metrics)}"
    )
    loss = metrics["e1_integration_loss"]
    assert math.isfinite(loss), f"E1 integration loss is not finite: {loss!r}"
    # A degenerate horizon-0 replay (the [T, 1, 2D] mis-stack) yields an
    # MSE over an empty target and comes back nan, which the isfinite check
    # above already rejects; a genuine replay over a random-init E1 on real
    # latents is strictly positive.
    assert loss > 0.0, f"E1 integration loss degenerate (== 0): {loss!r}"


def test_offline_integration_feeds_e1_the_feature_concatenation():
    """The sequence handed to E1 must be [1, self_dim + world_dim] per entry --
    the same layout as sense()'s total_state -- not [2, D]."""
    self_dim, world_dim = 16, 16
    agent = _agent_with_filled_buffers(self_dim, world_dim)
    seen = {}
    real = agent.e1.integrate_experience

    def spy(experience_buffer, *args, **kwargs):
        seen["shapes"] = {tuple(e.shape) for e in experience_buffer}
        seen["n"] = len(experience_buffer)
        return real(experience_buffer, *args, **kwargs)

    agent.e1.integrate_experience = spy
    try:
        agent.offline_integration()
    finally:
        agent.e1.integrate_experience = real

    assert seen["n"] == N_STEPS
    assert seen["shapes"] == {(1, self_dim + world_dim)}, (
        f"expected every replay entry to be [1, {self_dim + world_dim}], got {seen['shapes']}"
    )


def test_integrate_experience_batch1_entries_match_legacy_1d_entries():
    """E1.integrate_experience() must treat [1, D] entries exactly as it treats
    the legacy 1-D [D] entries (same loss under the same seed), so the agent's
    batch-carrying buffers and any older 1-D caller share one code path."""
    agent = _agent_with_filled_buffers(16, 16)
    e1 = agent.e1
    buf_2d = [
        torch.cat([s, w], dim=-1)
        for s, w in zip(agent._self_experience_buffer, agent._world_experience_buffer)
    ]
    buf_1d = [e.reshape(-1) for e in buf_2d]
    acts = agent._action_experience_buffer

    torch.manual_seed(123)
    out_2d = e1.integrate_experience(buf_2d, num_iterations=3, action_buffer=acts)
    torch.manual_seed(123)
    out_1d = e1.integrate_experience(buf_1d, num_iterations=3, action_buffer=acts)

    l2, l1 = out_2d["integration_loss"], out_1d["integration_loss"]
    assert math.isfinite(l1) and l1 > 0.0, f"legacy 1-D path degenerate: {l1!r}"
    assert math.isclose(l2, l1, rel_tol=1e-6), (
        f"[1, D] entries and [D] entries diverged: {l2!r} vs {l1!r}"
    )
