"""Contract tests for SD-016 ContextMemory write-path fix (2026-04-25).

EXQ-477 follow-up. Two fixes locked in:
  Part A: ContextMemory.key_proj has bias=False (structural absence).
          The default-init Linear bias dominated W_K @ memory_s for every
          slot when memory init scale was 0.01, collapsing all keys to ~b_K
          and softmax to uniform attention (entropy ~ ln(num_slots)).
  Part B: E1Config.sd016_writepath_mode flag with values
          {"off", "train_only", "sense_only", "both"} routes observation-
          conditioned writes into ContextMemory at one or both of two hooks:
            B1  REEAgent.compute_prediction_loss
                (training-time write via E1.update_from_observation,
                 internally _offline_mode-gated)
            B2  REEAgent.sense()
                (per-tick write via context_memory.write(), gated at the
                 call site on E1._offline_mode)

Guarantees enforced here:
  C1 (Part A):
    ContextMemory instances built via E1DeepPredictor have
    self.key_proj.bias is None -- structural, not a numerical default.
  C2 (Part A):
    With memory slots spread across content space and queries directly
    at memory_dim (bypassing query_proj's own bias to isolate the
    key_proj contract), softmax over scores yields attn_entropy_mean
    strictly below ln(num_slots) - 0.2.
  C3 (Part A):
    keys_pair_sim_content_only -- mean pairwise cosine similarity of
    W_K @ memory_s rows -- is below 0.95. With the dropped bias the
    keys are no longer dominated by a shared offset and become
    differentiable.
  C4 (Part B mode routing):
    sd016_writepath_mode="off"           -> no writes from sense() OR
                                            compute_prediction_loss
    sd016_writepath_mode="sense_only"    -> sense() writes; compute_prediction_loss
                                            does NOT
    sd016_writepath_mode="train_only"    -> compute_prediction_loss writes;
                                            sense() does NOT
    sd016_writepath_mode="both"          -> both hooks write
    Detection: ContextMemory.memory snapshot before/after each hook --
    a write changes at least one slot row; "no writes" means snapshot
    bit-identical.
  C5 (Part B offline gate):
    With sd016_writepath_mode="both" AND e1._offline_mode=True, neither
    sense() NOR compute_prediction_loss writes (B1 routes through
    update_from_observation which is internally gated; B2 has an explicit
    not-offline guard at the call site).

See REE_assembly/docs/architecture/context_memory_writepath_fix.md.
"""

from __future__ import annotations

import math

import torch

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config


# ------------------------------------------------------------------ #
# Helpers shared by all C4 / C5 tests                                #
# ------------------------------------------------------------------ #


def _obs_from_env_step(obs_dict, device=None):
    """Pull (obs_body, obs_world) out of CausalGridWorldV2's obs_dict."""
    ob = obs_dict["body_state"]
    ow = obs_dict["world_state"]
    if not torch.is_tensor(ob):
        ob = torch.as_tensor(ob, dtype=torch.float32)
    if not torch.is_tensor(ow):
        ow = torch.as_tensor(ow, dtype=torch.float32)
    if ob.dim() == 1:
        ob = ob.unsqueeze(0)
    if ow.dim() == 1:
        ow = ow.unsqueeze(0)
    return ob, ow


def _sense_one_tick(agent, env, obs_dict):
    """Call agent.sense() with the (body, world) split from obs_dict."""
    ob, ow = _obs_from_env_step(obs_dict)
    agent.sense(ob, ow)


def _step_random_action(env, action_dim: int = 4):
    """Take a no-op-ish step (action_class=0) and return the next obs_dict."""
    action = torch.zeros(1, action_dim)
    action[0, 0] = 1.0
    _, _harm, _done, _info, obs_dict = env.step(action)
    return obs_dict


# ------------------------------------------------------------------ #
# C1 -- Part A structural: key_proj.bias is None                     #
# ------------------------------------------------------------------ #


def test_c1_key_proj_bias_is_structurally_absent():
    """key_proj is constructed with bias=False; .bias is therefore None."""
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    from ree_core.agent import REEAgent

    agent = REEAgent(cfg)
    cm = agent.e1.context_memory

    assert cm.key_proj.bias is None, (
        "ContextMemory.key_proj must be constructed with bias=False "
        "(SD-016 Part A); found a bias parameter."
    )


# ------------------------------------------------------------------ #
# C2 -- Part A behavioural: attn_entropy below uniform reference     #
# ------------------------------------------------------------------ #


def test_c2_attention_is_content_conditioned():
    """With memory spread + queries at memory_dim, attn_entropy < ln(num_slots) - 0.2.

    We bypass cm.query_proj here on purpose: query_proj is a separate Linear
    with its own (kept) bias, and any test through query_proj also probes
    that path. Part A is the key_proj.bias=False contract; the cleanest
    check is to directly construct queries at memory_dim and verify that
    softmax(q @ key_proj(memory).T / sqrt(d)) is non-uniform when memory
    rows are differentiated. If key_proj.bias were not zero, all keys would
    share a dominating offset and softmax would collapse to uniform
    regardless of query content.
    """
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    from ree_core.agent import REEAgent

    agent = REEAgent(cfg)
    cm = agent.e1.context_memory

    # Spread memory aggressively so each slot carries genuinely distinct
    # content. Scale 3.0 ensures scores have meaningful variance after
    # the sqrt(memory_dim) normalisation in the attention forward path.
    with torch.no_grad():
        cm.memory.copy_(torch.randn_like(cm.memory) * 3.0)

    # Directly construct random queries at memory_dim (bypass query_proj
    # so this test isolates the key_proj.bias=False contract). Scale 3.0
    # to match memory.
    queries = torch.randn(16, cm.memory_dim) * 3.0

    keys = cm.key_proj(cm.memory)  # [num_slots, memory_dim]
    scores = queries @ keys.t() / (cm.memory_dim ** 0.5)  # [16, num_slots]
    weights = torch.softmax(scores, dim=-1)

    eps = 1e-12
    entropies = -(weights * (weights + eps).log()).sum(dim=-1)
    mean_entropy = float(entropies.mean().item())

    uniform_ref = math.log(cm.num_slots)
    assert mean_entropy < uniform_ref - 0.2, (
        f"attn_entropy_mean={mean_entropy:.4f} too close to uniform reference "
        f"{uniform_ref:.4f} (num_slots={cm.num_slots}). bias=False alone is "
        "not sufficient to restore content-conditioned attention; check that "
        "key_proj.bias is None and that memory slots carry distinct content."
    )


# ------------------------------------------------------------------ #
# C3 -- Part A behavioural: keys are differentiable across slots     #
# ------------------------------------------------------------------ #


def test_c3_keys_pair_sim_below_threshold():
    """Mean pairwise cosine similarity of W_K @ memory_s rows < 0.95."""
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    from ree_core.agent import REEAgent

    agent = REEAgent(cfg)
    cm = agent.e1.context_memory

    with torch.no_grad():
        cm.memory.copy_(torch.randn_like(cm.memory))

    keys = cm.key_proj(cm.memory)  # [num_slots, memory_dim]
    keys_norm = keys / (keys.norm(dim=-1, keepdim=True) + 1e-12)
    sim = keys_norm @ keys_norm.t()  # [num_slots, num_slots]
    n = sim.shape[0]
    off = sim - torch.eye(n) * sim.diag()
    pair_sim = float(off.sum().item() / (n * (n - 1)))

    assert pair_sim < 0.95, (
        f"keys_pair_sim_content_only={pair_sim:.4f} above 0.95 -- keys are "
        "not differentiable across slots. Check that key_proj is content-"
        "only (no bias) and that memory slots carry distinct content."
    )


# ------------------------------------------------------------------ #
# C4 -- Part B mode routing                                          #
# ------------------------------------------------------------------ #


def _build_agent_with_writepath_mode(mode: str):
    """Construct a tiny REEAgent with the given sd016_writepath_mode."""
    from ree_core.agent import REEAgent

    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env, sd016_writepath_mode=mode)
    return REEAgent(cfg), env


def _memory_snapshot(agent) -> torch.Tensor:
    """Return a clone of the ContextMemory slot tensor."""
    return agent.e1.context_memory.memory.detach().clone()


def _populate_buffers_for_train_loss(agent, env, n_steps: int = 4):
    """Populate the world / self experience buffers so compute_prediction_loss
    has at least 2 frames to sample from.

    REEAgent.sense() encodes obs but does NOT push to the experience buffers
    -- _e1_tick is what populates them, and that runs inside select_action.
    For a contract test we don't want to invoke the full select_action path
    (heartbeat, BG, residue, etc. -- side effects far beyond what we're
    testing). Instead we run sense() to obtain a real LatentState and then
    manually push z_self / z_world copies into the buffers, mirroring what
    _e1_tick would do.
    """
    _, obs_dict = env.reset()
    for _ in range(n_steps):
        ob, ow = _obs_from_env_step(obs_dict)
        latent = agent.sense(ob, ow)
        agent._self_experience_buffer.append(latent.z_self.detach().clone())
        agent._world_experience_buffer.append(latent.z_world.detach().clone())
        obs_dict = _step_random_action(env)


def test_c4_off_mode_writes_zero():
    """sd016_writepath_mode='off' -> neither sense() nor compute_prediction_loss writes."""
    agent, env = _build_agent_with_writepath_mode("off")

    pre = _memory_snapshot(agent)
    _, obs_dict = env.reset()
    _sense_one_tick(agent, env, obs_dict)
    post_sense = _memory_snapshot(agent)
    assert torch.equal(pre, post_sense), (
        "off mode: sense() must not modify ContextMemory.memory."
    )

    _populate_buffers_for_train_loss(agent, env, n_steps=3)
    pre_train = _memory_snapshot(agent)
    _ = agent.compute_prediction_loss()
    post_train = _memory_snapshot(agent)
    assert torch.equal(pre_train, post_train), (
        "off mode: compute_prediction_loss() must not modify ContextMemory.memory."
    )


def test_c4_sense_only_mode_writes_on_sense_not_train():
    """sd016_writepath_mode='sense_only' -> sense() writes; compute_prediction_loss does not."""
    agent, env = _build_agent_with_writepath_mode("sense_only")

    pre = _memory_snapshot(agent)
    _, obs_dict = env.reset()
    _sense_one_tick(agent, env, obs_dict)
    post_sense = _memory_snapshot(agent)
    assert not torch.equal(pre, post_sense), (
        "sense_only mode: sense() must modify ContextMemory.memory."
    )

    _populate_buffers_for_train_loss(agent, env, n_steps=3)
    pre_train = _memory_snapshot(agent)
    _ = agent.compute_prediction_loss()
    post_train = _memory_snapshot(agent)
    assert torch.equal(pre_train, post_train), (
        "sense_only mode: compute_prediction_loss() must not modify ContextMemory.memory."
    )


def test_c4_train_only_mode_writes_on_train_not_sense():
    """sd016_writepath_mode='train_only' -> compute_prediction_loss writes; sense() does not."""
    agent, env = _build_agent_with_writepath_mode("train_only")

    pre = _memory_snapshot(agent)
    _, obs_dict = env.reset()
    _sense_one_tick(agent, env, obs_dict)
    post_sense = _memory_snapshot(agent)
    assert torch.equal(pre, post_sense), (
        "train_only mode: sense() must not modify ContextMemory.memory."
    )

    _populate_buffers_for_train_loss(agent, env, n_steps=3)
    pre_train = _memory_snapshot(agent)
    _ = agent.compute_prediction_loss()
    post_train = _memory_snapshot(agent)
    assert not torch.equal(pre_train, post_train), (
        "train_only mode: compute_prediction_loss() must modify ContextMemory.memory."
    )


def test_c4_both_mode_writes_on_both_hooks():
    """sd016_writepath_mode='both' -> sense() AND compute_prediction_loss write."""
    agent, env = _build_agent_with_writepath_mode("both")

    pre = _memory_snapshot(agent)
    _, obs_dict = env.reset()
    _sense_one_tick(agent, env, obs_dict)
    post_sense = _memory_snapshot(agent)
    assert not torch.equal(pre, post_sense), (
        "both mode: sense() must modify ContextMemory.memory."
    )

    _populate_buffers_for_train_loss(agent, env, n_steps=3)
    pre_train = _memory_snapshot(agent)
    _ = agent.compute_prediction_loss()
    post_train = _memory_snapshot(agent)
    assert not torch.equal(pre_train, post_train), (
        "both mode: compute_prediction_loss() must modify ContextMemory.memory."
    )


# ------------------------------------------------------------------ #
# C5 -- Part B offline gate                                          #
# ------------------------------------------------------------------ #


def test_c5_offline_mode_suppresses_all_writes():
    """e1._offline_mode=True with mode='both' -> no writes from either hook.

    B1 is internally _offline_mode-gated via E1.update_from_observation;
    B2 has an explicit not-offline guard at the call site in sense().
    """
    agent, env = _build_agent_with_writepath_mode("both")
    agent.e1._offline_mode = True

    pre = _memory_snapshot(agent)
    _, obs_dict = env.reset()
    _sense_one_tick(agent, env, obs_dict)
    post_sense = _memory_snapshot(agent)
    assert torch.equal(pre, post_sense), (
        "offline_mode=True: sense() must not modify ContextMemory.memory "
        "even when sd016_writepath_mode='both'."
    )

    _populate_buffers_for_train_loss(agent, env, n_steps=3)
    pre_train = _memory_snapshot(agent)
    _ = agent.compute_prediction_loss()
    post_train = _memory_snapshot(agent)
    assert torch.equal(pre_train, post_train), (
        "offline_mode=True: compute_prediction_loss() must not modify "
        "ContextMemory.memory even when sd016_writepath_mode='both'."
    )
