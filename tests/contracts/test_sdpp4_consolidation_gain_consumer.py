"""Integration contracts for SD-PP-1..4 wired into REEAgent (2026-09-22).

The four modules each have their own unit suites (test_sdpp1..4_*). This file
pins the INTEGRATION the session wrote in agent.py / config.py /
phase_manager.py, on the real waking + sleep path the V3-EXQ-1063 driver uses:

  I1  DEFAULT OFF is structural absence -- all four agent attributes are None,
      the SleepLoopManager flags are False, and a forced sleep cycle emits none
      of the new metric keys.
  I2  A/B NEUTRALITY -- recording provenance (SD-PP-1/2/3 on, gain off) leaves
      the world experience buffer AND the post-sleep world-head parameters
      BITWISE identical to the no-provenance agent at the same seed. The
      producers draw no RNG and the gain-off sleep pass is the pre-build pass.
  I3  ALIGNMENT through the agent -- len(packets) == len(_world_experience_buffer)
      after waking, and every has_prev packet's pred_at_test equals
      e2.world_forward(world[j-1], action[j]) for its position j.
  I4  GAIN IS LIVE -- with the gain on, the sleep pass emits the step_scale_*
      keys, the per-step trace carries 8 e2_world steps, and the world-head
      displacement differs from the gain-off agent.
  I5  LIVENESS (the MECH-572 pin, on the REAL pass) -- mode "global" at
      global_scale 0.1 / 1.0 / 2.0 gives strictly increasing displacement.
  I6  FLAG CONSISTENCY -- use_replay_precision_provenance without both
      producers, and the gain without provenance + the world-forward trainer,
      raise ValueError at construction.
  I7  notify_env_reset -- the packet recorded on the first tick after a reset
      is a placeholder (has_prev False) and the estimator drops its frame.
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

PROV = dict(
    use_observation_reliability=True,
    use_world_forward_epistemic_precision=True,
    use_replay_precision_provenance=True,
)


def _build(seed: int, **flags):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=16,
        world_dim=16,
        sws_enabled=True,
        rem_enabled=True,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1_000_000,
        use_sleep_aggregation_cluster=True,
        use_cross_module_consolidation=True,
        cross_module_consolidation_schedule="interleaved",
        cross_module_consolidation_steps=8,
        cross_module_consolidation_lr=1e-3,
        cross_module_consolidation_batch=16,
        use_sleep_world_forward_consolidation=True,
        surprise_gated_replay=True,
        pe_ema_alpha=0.02,
        use_mel_consumer=False,
        use_entry_pressure=False,
        use_within_life_sleep_trigger=False,
        cross_module_consolidation_record_trace=True,
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return env, agent


def _wake(env, agent, n: int = 40, seed: int = 1) -> None:
    """The V3-EXQ-1063 P1 loop, verbatim in shape."""
    _, od = env.reset()
    agent.notify_env_reset()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n):
        b = od["body_state"].unsqueeze(0)
        w = od["world_state"].unsqueeze(0)
        harm = od.get("harm_obs")
        if harm is not None and harm.dim() == 1:
            harm = harm.unsqueeze(0)
        latent = agent.sense(b, w, obs_harm=harm)
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick", False)
            else torch.zeros(1, 16)
        )
        cands = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(cands, ticks)
        if action is None:
            action = torch.zeros(1, env.action_dim)
            action[0, int(torch.randint(0, env.action_dim, (1,), generator=gen))] = 1.0
            agent._last_action = action
        agent.record_executed_action(action)
        _, harm_s, done, _info, od = env.step(action)
        with torch.no_grad():
            agent.update_residue(
                harm_signal=float(harm_s), world_delta=None,
                hypothesis_tag=False, owned=True,
            )
        if done:
            _, od = env.reset()
            agent.notify_env_reset()


def _world_params(agent):
    return [
        p.detach().clone()
        for m in ("world_transition", "world_action_encoder")
        for p in getattr(agent.e2, m).parameters()
    ]


def _hash_tensors(ts) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _buffer_hash(agent) -> str:
    return _hash_tensors([torch.cat(agent._world_experience_buffer)])


def _max_delta(before, after) -> float:
    return max(float((a - b).abs().max().item()) for a, b in zip(before, after))


NEW_KEY_MARKERS = ("step_scale", "provenance_gain_last", "obs_reliability_",
                   "wf_precision_", "replay_provenance_")


# ---------------------------------------------------------------- I1
def test_i1_default_off_is_structural_absence():
    env, agent = _build(7)
    assert agent.observation_reliability is None
    assert agent.world_forward_precision is None
    assert agent.replay_provenance is None
    assert agent.provenance_gain_config is None
    assert agent.sleep_loop.provenance_consolidation_gain is False
    _wake(env, agent, 30)
    m = agent.sleep_loop.force_cycle(agent)
    leaked = [k for k in m if any(mk in k for mk in NEW_KEY_MARKERS)]
    assert leaked == [], leaked


# ---------------------------------------------------------------- I2
def test_i2_store_only_is_bitwise_neutral():
    env_a, a = _build(42)
    _wake(env_a, a, 60)
    env_b, b = _build(42, **PROV)
    _wake(env_b, b, 60)
    assert _buffer_hash(a) == _buffer_hash(b)
    assert _hash_tensors(_world_params(a)) == _hash_tensors(_world_params(b))
    rs_a = torch.get_rng_state()
    a.sleep_loop.force_cycle(a)
    rs_b = torch.get_rng_state()
    # B must enter its pass with the same RNG state A did (producers draw none).
    torch.set_rng_state(rs_a)
    b.sleep_loop.force_cycle(b)
    assert _hash_tensors(_world_params(a)) == _hash_tensors(_world_params(b))
    del rs_b


# ---------------------------------------------------------------- I3
def test_i3_packets_align_with_buffer_through_agent():
    env, agent = _build(3, **PROV)
    _wake(agent_env := env, agent, 50)  # noqa: F841 (walrus keeps the env alive)
    rec = agent.replay_provenance
    wb = agent._world_experience_buffer
    ab = agent._action_experience_buffer
    assert len(rec.packets) == len(wb) == len(ab)
    n_checked = 0
    with torch.no_grad():
        for j in range(1, len(wb)):
            p = rec.get(j)
            assert p is not None and p.buffer_index == j
            if not p.has_prev:
                continue
            pred = agent.e2.world_forward(wb[j - 1][:1], ab[j][:1])
            assert torch.equal(p.pred_at_test, pred.detach()), j
            n_checked += 1
    assert n_checked >= 40


# ---------------------------------------------------------------- I4
def test_i4_gain_is_live_and_instrumented():
    env_b, b = _build(42, **PROV)
    _wake(env_b, b, 60)
    env_c, c = _build(42, **PROV, use_provenance_conditioned_consolidation_gain=True)
    _wake(env_c, c, 60)
    rs = torch.get_rng_state()
    before_b = _world_params(b)
    b.sleep_loop.force_cycle(b)
    torch.set_rng_state(rs)
    before_c = _world_params(c)
    m = c.sleep_loop.force_cycle(c)
    assert "cross_module_consolidation_step_scale_mean_e2_world" in m
    assert "provenance_gain_last_gain_mean" in m
    trace = c.cross_module_consolidator.last_step_trace
    e2w = [t for t in trace if t["module"] == "e2_world"]
    assert len(e2w) == 8
    assert all(t["step_scale"] > 0.0 for t in e2w)
    assert c._last_consolidation_gain is not None
    assert c._last_consolidation_gain.get("n_missing", 0.0) == 0.0
    d_b = _max_delta(before_b, _world_params(b))
    d_c = _max_delta(before_c, _world_params(c))
    assert d_b > 0.0 and d_c > 0.0 and d_b != d_c


# ---------------------------------------------------------------- I5
def test_i5_global_scale_liveness_on_real_pass():
    deltas = []
    for scale in (0.1, 1.0, 2.0):
        env, agent = _build(
            42, **PROV,
            use_provenance_conditioned_consolidation_gain=True,
            provenance_gain_mode="global",
            provenance_gain_global_scale=scale,
        )
        _wake(env, agent, 60)
        torch.manual_seed(999)
        before = _world_params(agent)
        agent.sleep_loop.force_cycle(agent)
        deltas.append(_max_delta(before, _world_params(agent)))
    assert deltas[0] < deltas[1] < deltas[2], deltas


# ---------------------------------------------------------------- I6
def test_i6_flag_consistency_is_enforced():
    with pytest.raises(ValueError):
        _build(1, use_replay_precision_provenance=True)
    with pytest.raises(ValueError):
        _build(1, use_observation_reliability=True,
               use_world_forward_epistemic_precision=True,
               use_provenance_conditioned_consolidation_gain=True)


# ---------------------------------------------------------------- I7
def test_i7_env_reset_yields_placeholder_and_drops_frame():
    env, agent = _build(5, **PROV)
    _wake(env, agent, 20)
    n0 = len(agent.replay_provenance.packets)
    agent.notify_env_reset()
    assert agent.observation_reliability._prev_obs is None if hasattr(
        agent.observation_reliability, "_prev_obs") else True
    _wake(env, agent, 1)  # one tick after the reset
    p = agent.replay_provenance.packets[n0]
    assert p.has_prev is False
    assert len(agent.replay_provenance.packets) == len(agent._world_experience_buffer)
