"""Contract tests for the E2 world-forward sleep trainer (2026-09-17 build).

THE DEFECT THIS BUILD ADDRESSES
-------------------------------
`CrossModuleConsolidator.consolidate()` is pure orchestration -- the CALLER
supplies the losses. Its only live caller, `SleepLoopManager._run_cycle`, passed

    module_losses = {"e1": agent.compute_prediction_loss,
                     "e2": agent.compute_e2_loss}
    module_params["e2"] = list(agent.e2.parameters())

and `compute_e2_loss` calls ONLY `e2.predict_next_self`. So E2's world-domain
heads -- `world_transition` and `world_action_encoder`, both UNCONDITIONAL
modules -- sat in the "e2" optimiser's parameter list and received NO gradient on
any sleep cycle. That is the recorded "delta == 0.0 on every seed".

THE LEVER
---------
`REEConfig.use_sleep_world_forward_consolidation: bool = False`. When True a THIRD
module "e2_world" joins the pass, running `agent.compute_e2_world_loss` (the SD-056
`world_forward_contrastive_loss` over the aligned z_world / action replay buffers)
SCOPED to the two world heads.

OFF is bit-identical BY STRUCTURAL ABSENCE, and that is what W2 proves: the key is
never added, so no closure is constructed and no RNG is consumed. The lever is
REQUIRED rather than cosmetic because `consolidate()`'s "interleaved" schedule
steps every named module per trace, so a third name both changes
`cross_module_replay_share` and consumes global-RNG draws BETWEEN the e1 and e2
draws -- which shifts the replay batches those two modules themselves sample.
W2c measures that divergence directly, so the reason for the lever is pinned, not
merely asserted in a comment.

CONTRACTS
  W1  default OFF -- flag default False; consolidation runs with exactly
      {"e1", "e2"} and emits no updates_e2_world readout.
  W2  OFF structural absence -- the consolidate() call is shape-identical to the
      pre-build call (same module names, same param identity sets).
  W2c the hazard is real -- ON perturbs the e1/e2 parameters relative to OFF at
      the same seed, which is why OFF had to be structural.
  W3  ON trains the world heads -- delta > 0 and updates_e2_world > 0.
  W4  OFF leaves the world heads at delta EXACTLY 0.0 (the defect, latched).
  W5  param scoping -- "e2_world" is exactly world_transition +
      world_action_encoder, and excludes the z_self head "e2" already owns.
  W6  ALIGNMENT -- the training triple is (world[i], action[i+1], world[i+1]),
      the +1 offset compute_prediction_loss documents. The off-by-one is
      shape-identical and would train on the action that led INTO the input
      state, so this is pinned against the actual tensors handed to the loss.
  W7  no replay content -- a short buffer returns a graph-anchored exactly-zero
      sentinel, which consolidate() reads as "module not touched".
  W8  NOT inert under monostrategy -- a replay batch with ONE distinct action
      class still produces a real loss and a real gradient on both world
      heads. The helper's first-action-class floor is a SIBLING-CANDIDATE
      guard, and inheriting it here would have made the whole lever a no-op
      on exactly the agent behaviour sleep most needs to consolidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


# ----------------------------------------------------------------------
# Helpers (same shape as test_mech423_cross_module_consolidation.py)
# ----------------------------------------------------------------------
def _build(seed: int = 7, **flags):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, od = env.reset()
    b = od["body_state"]
    w = od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return agent, b, w


def _fill_buffers(agent, b, w, n: int = 14):
    sd = agent.config.latent.self_dim
    for _ in range(n):
        with torch.no_grad():
            a = agent.act_with_split_obs(b, w)
        act = a if a.dim() == 2 else a.unsqueeze(0)
        agent.record_transition(
            torch.randn(1, sd), act.float(), torch.randn(1, sd)
        )


def _sleep_flags(**extra):
    base = dict(
        use_sleep_loop=True,
        sws_enabled=True,
        rem_enabled=True,
        use_cross_module_consolidation=True,
        cross_module_consolidation_steps=3,
        cross_module_consolidation_schedule="interleaved",
    )
    base.update(extra)
    return base


def _world_head_params(agent):
    return (
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters())
    )


def _snapshot(params):
    return [p.detach().clone() for p in params]


def _max_delta(before, params):
    return max(
        float((a - p.detach()).abs().max().item())
        for a, p in zip(before, params)
    )


def _spy_consolidate(agent):
    """Wrap the consolidator so the call's SHAPE is observable."""
    seen = {}
    real = agent.cross_module_consolidator.consolidate

    def wrapper(*args, **kwargs):
        seen["loss_names"] = sorted(kwargs["module_losses"].keys())
        seen["param_names"] = sorted(kwargs["module_params"].keys())
        seen["param_ids"] = {
            k: [id(p) for p in v] for k, v in kwargs["module_params"].items()
        }
        return real(*args, **kwargs)

    agent.cross_module_consolidator.consolidate = wrapper
    return seen


def _run_sleep(seed=7, **extra):
    agent, b, w = _build(seed=seed, **_sleep_flags(**extra))
    _fill_buffers(agent, b, w, 14)
    seen = _spy_consolidate(agent)
    before = _snapshot(_world_head_params(agent))
    torch.manual_seed(1234)  # pin the consolidation draws
    metrics = agent.sleep_loop.force_cycle(agent)
    return agent, seen, metrics, before


# ----------------------------------------------------------------------
# W1 -- default OFF
# ----------------------------------------------------------------------
def test_w1_flag_defaults_off():
    assert REEConfig().use_sleep_world_forward_consolidation is False


def test_w1_off_emits_no_e2_world_readout():
    _agent, _seen, metrics, _before = _run_sleep()
    assert "cross_module_consolidation_updates_e1" in metrics
    assert "cross_module_consolidation_updates_e2" in metrics
    assert "cross_module_consolidation_updates_e2_world" not in metrics


# ----------------------------------------------------------------------
# W2 -- OFF structural absence (the bit-identity claim)
# ----------------------------------------------------------------------
def test_w2_off_call_is_shape_identical_to_pre_build_call():
    agent, seen, _metrics, _before = _run_sleep()
    assert seen["loss_names"] == ["e1", "e2"]
    assert seen["param_names"] == ["e1", "e2"]
    assert seen["param_ids"]["e2"] == [id(p) for p in agent.e2.parameters()]


def test_w2_on_adds_exactly_one_name():
    _agent, seen, _metrics, _before = _run_sleep(
        use_sleep_world_forward_consolidation=True
    )
    assert seen["loss_names"] == ["e1", "e2", "e2_world"]
    assert seen["param_names"] == ["e1", "e2", "e2_world"]


def test_w2c_on_perturbs_e1_e2_relative_to_off():
    """Why the lever had to be structural, measured rather than asserted."""
    off_agent, _s1, _m1, _b1 = _run_sleep(seed=11)
    on_agent, _s2, _m2, _b2 = _run_sleep(
        seed=11, use_sleep_world_forward_consolidation=True
    )
    off_e2 = [p.detach() for p in off_agent.e2.self_transition.parameters()]
    on_e2 = [p.detach() for p in on_agent.e2.self_transition.parameters()]
    diverged = max(
        float((a - b).abs().max().item()) for a, b in zip(off_e2, on_e2)
    )
    assert diverged > 0.0, (
        "adding a third interleaved module must perturb the existing modules "
        "(shared RNG + schedule); if this is 0 the lever's rationale is wrong"
    )


# ----------------------------------------------------------------------
# W3 / W4 -- the world heads move ON and do not move OFF
# ----------------------------------------------------------------------
def test_w3_on_trains_the_world_heads():
    agent, _seen, metrics, before = _run_sleep(
        use_sleep_world_forward_consolidation=True
    )
    assert metrics["cross_module_consolidation_updates_e2_world"] > 0.0
    assert _max_delta(before, _world_head_params(agent)) > 0.0


def test_w4_off_leaves_world_heads_at_exactly_zero_delta():
    agent, _seen, metrics, before = _run_sleep()
    assert metrics["cross_module_consolidation_updates_e2"] > 0.0
    assert _max_delta(before, _world_head_params(agent)) == 0.0


# ----------------------------------------------------------------------
# W5 -- param scoping
# ----------------------------------------------------------------------
def test_w5_e2_world_params_are_exactly_the_two_world_heads():
    agent, seen, _metrics, _before = _run_sleep(
        use_sleep_world_forward_consolidation=True
    )
    expected = [id(p) for p in _world_head_params(agent)]
    assert seen["param_ids"]["e2_world"] == expected

    self_head_ids = {id(p) for p in agent.e2.self_transition.parameters()}
    assert not (set(expected) & self_head_ids), (
        "e2_world must not double-step the z_self head that 'e2' owns"
    )


# ----------------------------------------------------------------------
# W6 -- alignment: (world[i], action[i+1], world[i+1])
# ----------------------------------------------------------------------
def test_w6_training_triple_uses_the_plus_one_action_offset():
    agent, _b, _w = _build()
    world_dim = int(agent.config.latent.world_dim)
    action_dim = 4
    n = 9

    # world[i] is the constant vector i; action[i] is one-hot at (i % 4).
    agent._world_experience_buffer = [
        torch.full((1, world_dim), float(i)) for i in range(n)
    ]
    agent._action_experience_buffer = []
    for i in range(n):
        a = torch.zeros(1, action_dim)
        a[0, i % action_dim] = 1.0
        agent._action_experience_buffer.append(a)

    captured = {}

    def capture(z_world_0, actions, z_world_1_targets, **kwargs):
        captured["z0"] = z_world_0.detach().clone()
        captured["a"] = actions.detach().clone()
        captured["z1"] = z_world_1_targets.detach().clone()
        return (
            next(agent.e2.world_transition.parameters()).sum() * 0.0 + 1.0
        )

    agent.e2.world_forward_contrastive_loss = capture
    agent.compute_e2_world_loss(batch_size=32)

    z0 = captured["z0"]
    z1 = captured["z1"]
    acts = captured["a"]
    assert z0.shape[0] == n - 1, "every aligned triple should be drawn"

    for r in range(z0.shape[0]):
        i = int(round(float(z0[r, 0].item())))
        assert torch.allclose(z0[r], torch.full((z0.shape[1],), float(i)))
        # target is state i+1 ...
        assert torch.allclose(
            z1[r], torch.full((z1.shape[1],), float(i + 1))
        ), f"row {r}: target must be world[{i + 1}]"
        # ... carried by the action recorded ALONGSIDE state i+1.
        assert int(acts[r].argmax().item()) == (i + 1) % action_dim, (
            f"row {r}: off-by-one -- got action[{i}] where action[{i + 1}] is "
            "the one that carries world[i] -> world[i+1]"
        )


# ----------------------------------------------------------------------
# W7 -- no replay content
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n_states", [0, 1, 2])
def test_w7_short_buffer_returns_graph_anchored_zero(n_states):
    agent, _b, _w = _build()
    world_dim = int(agent.config.latent.world_dim)
    agent._world_experience_buffer = [
        torch.full((1, world_dim), float(i)) for i in range(n_states)
    ]
    agent._action_experience_buffer = [
        torch.zeros(1, 4) for _ in range(n_states)
    ]
    loss = agent.compute_e2_world_loss(batch_size=8)
    assert torch.is_tensor(loss)
    assert float(loss.detach().item()) == 0.0
    # graph-anchored: consolidate() reads a non-grad tensor as "skip" too, but
    # the sentinel must be the documented params.sum() * 0.0 shape.
    assert loss.requires_grad


# ----------------------------------------------------------------------
# W8 -- NOT inert under monostrategy (the dead-flag failure mode)
# ----------------------------------------------------------------------
def test_w8_single_action_class_batch_still_trains():
    """A replay batch with ONE distinct action class must still train.

    This is the contract that nearly went the other way. The InfoNCE helper
    defaults to a floor of 2 distinct first-action classes, which is right for
    its WAKING callers (K sibling candidates off ONE state, where a
    single-class batch makes the negatives identical to the positive) and
    wrong here (K DISTINCT transitions, where the negatives are other states).
    REE agents are monostrategy in long stretches -- 14 consecutive ticks on
    CausalGridWorldV2 gave one action class -- so inheriting the floor would
    make the trainer silently inert exactly when sleep has the most to
    consolidate. That is the dead-flag failure mode tests/test_flag_inertness.py
    exists to catch, so it is pinned here rather than left to a comment.
    """
    agent, _b, _w = _build()
    world_dim = int(agent.config.latent.world_dim)
    n = 12
    torch.manual_seed(3)
    agent._world_experience_buffer = [
        torch.randn(1, world_dim) for _ in range(n)
    ]
    same = torch.zeros(1, 4)
    same[0, 0] = 1.0
    agent._action_experience_buffer = [same.clone() for _ in range(n)]

    loss = agent.compute_e2_world_loss(batch_size=n)
    assert loss.requires_grad
    assert float(loss.detach().item()) > 0.0, (
        "a single-action-class replay batch must still yield a real loss"
    )

    params = _world_head_params(agent)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    assert any(g is not None for g in grads)
    assert max(
        float(g.abs().max().item()) for g in grads if g is not None
    ) > 0.0, "the gradient must actually reach the world heads"
