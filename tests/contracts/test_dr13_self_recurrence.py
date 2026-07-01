"""Contracts for SELF-1 / DR-13 (self_model_v4): z_self temporal depth.

The substrate floor of the self_model_v4 plan: z_self stops being an
instantaneous body snapshot (single-MLP + fixed-alpha EMA) and becomes a
STATEFUL temporal self-model via a dedicated gated self-recurrence
(SelfRecurrenceCell) whose dynamics are anchored by E1 generative feedback.

Mechanism (HYBRID, ARC-081 notes 2026-06-14): a light dedicated self-recurrence
REGULARISED/ANCHORED by the E1 predicted-next z_self, blend weight
self_recurrence_e1_coupling (0 = pure recurrence / Option A; 1 = pure
E1-feedback / Option B; light default = hybrid).

Contract shape mirrors the DR-12 (SELF-4) contracts: bit-identical OFF; the
lever populates a readout when ON; the stateful value departs from the
instantaneous encode; the E1 anchor blends exactly; the self subject is
perturbation-isolated from z_world; coupling=0 ignores the anchor; and the
agent-level plumbing (E1-tick anchor cache -> next encode) runs end-to-end.

generation:v4 -- off the V3 closure path; PROMOTES NOTHING.
"""

from __future__ import annotations

import torch

from ree_core.utils.config import LatentStackConfig, REEConfig
from ree_core.latent.stack import LatentStack
from ree_core.latent.self_recurrence import SelfRecurrenceCell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2


def _obs_dim(cfg: LatentStackConfig) -> int:
    return cfg.body_obs_dim + cfg.world_obs_dim


def _stack(seed: int = 0, **kw) -> LatentStack:
    torch.manual_seed(seed)
    return LatentStack(LatentStackConfig(**kw))


# ---------------------------------------------------------------------------
# C1: bit-identical OFF (default)
# ---------------------------------------------------------------------------

def test_c1_off_no_module_no_diag():
    s = _stack()
    assert s.self_recurrence is None
    torch.manual_seed(1)
    obs = torch.randn(1, _obs_dim(s.config))
    st = s.encode(obs)
    st2 = s.encode(obs, st)
    assert st2.self_recurrence_diag is None


def test_c1_off_supplying_anchor_is_noop():
    """OFF path ignores a supplied self_e1_anchor (bit-identical)."""
    s = _stack()
    torch.manual_seed(2)
    obs = torch.randn(1, _obs_dim(s.config))
    st = s.encode(obs)
    a = torch.randn(1, s.config.self_dim)
    base = s.encode(obs, st)
    with_anchor = s.encode(obs, st, self_e1_anchor=a)
    assert torch.allclose(base.z_self, with_anchor.z_self)
    assert with_anchor.self_recurrence_diag is None


def test_c1_off_deterministic_across_identical_builds():
    s1 = _stack(0)
    s2 = _stack(0)
    torch.manual_seed(3)
    obs = torch.randn(1, _obs_dim(s1.config))
    a1 = s1.encode(obs, s1.encode(obs))
    a2 = s2.encode(obs, s2.encode(obs))
    assert torch.allclose(a1.z_self, a2.z_self)


# ---------------------------------------------------------------------------
# C2: ON populates the module + readout
# ---------------------------------------------------------------------------

def test_c2_on_instantiates_module_and_diag():
    s = _stack(use_self_recurrence=True)
    assert isinstance(s.self_recurrence, SelfRecurrenceCell)
    torch.manual_seed(4)
    obs = torch.randn(1, _obs_dim(s.config))
    q = s.encode(obs, s.encode(obs))
    d = q.self_recurrence_diag
    assert d is not None and d["active"] is True
    assert d["e1_coupling"] == s.config.self_recurrence_e1_coupling
    assert torch.isfinite(q.z_self).all()


# ---------------------------------------------------------------------------
# C3: the stateful z_self departs from the instantaneous encode (temporal depth)
# ---------------------------------------------------------------------------

def test_c3_stateful_departs_from_instantaneous():
    s = _stack(use_self_recurrence=True)
    torch.manual_seed(5)
    # feed a varying observation sequence so history matters
    prev = None
    departures = []
    for _ in range(4):
        obs = torch.randn(1, _obs_dim(s.config))
        st = s.encode(obs, prev)
        departures.append(st.self_recurrence_diag["state_departure"])
        prev = st
    assert all(x >= 0.0 for x in departures)
    assert max(departures) > 0.0  # the recurrence carries state the snapshot does not


# ---------------------------------------------------------------------------
# C4: E1-anchor blend is exact: z_self = (1-c) * recur + c * anchor
# ---------------------------------------------------------------------------

def test_c4_anchor_blend_exact():
    s = _stack(use_self_recurrence=True)
    c = s.config.self_recurrence_e1_coupling
    assert c > 0.0
    torch.manual_seed(6)
    obs = torch.randn(1, _obs_dim(s.config))
    prev = s.encode(obs)
    anchor = torch.randn(1, s.config.self_dim)
    recur_only = s.encode(obs, prev, self_e1_anchor=None)
    blended = s.encode(obs, prev, self_e1_anchor=anchor)
    expected = (1.0 - c) * recur_only.z_self + c * anchor
    assert torch.allclose(blended.z_self, expected, atol=1e-5)
    assert blended.self_recurrence_diag["anchor_present"] is True
    assert recur_only.self_recurrence_diag["anchor_present"] is False


def test_c4_anchor_shape_mismatch_falls_back_to_recurrence():
    s = _stack(use_self_recurrence=True)
    torch.manual_seed(7)
    obs = torch.randn(1, _obs_dim(s.config))
    prev = s.encode(obs)
    recur_only = s.encode(obs, prev, self_e1_anchor=None)
    bad = torch.randn(1, s.config.self_dim + 3)  # wrong dim
    out = s.encode(obs, prev, self_e1_anchor=bad)
    assert torch.allclose(out.z_self, recur_only.z_self)
    assert out.self_recurrence_diag["anchor_present"] is False


# ---------------------------------------------------------------------------
# C5: the self subject is perturbation-isolated from z_world
# ---------------------------------------------------------------------------

def test_c5_self_perturbation_does_not_leak_to_world():
    s = _stack(use_self_recurrence=True)
    torch.manual_seed(8)
    obs = torch.randn(1, _obs_dim(s.config))
    base = s.encode(obs)
    pert = base.detach()
    pert.z_self = base.z_self + 5.0  # perturb ONLY the self subject
    r_base = s.encode(obs, base)
    r_pert = s.encode(obs, pert)
    self_change = (r_base.z_self - r_pert.z_self).abs().max().item()
    world_leak = (r_base.z_world - r_pert.z_world).abs().max().item()
    assert self_change > 1e-4
    assert world_leak < 1e-6


# ---------------------------------------------------------------------------
# C6: coupling=0 -> pure recurrence even with an anchor supplied (Option A)
# ---------------------------------------------------------------------------

def test_c6_coupling_zero_ignores_anchor():
    s = _stack(use_self_recurrence=True, self_recurrence_e1_coupling=0.0)
    torch.manual_seed(9)
    obs = torch.randn(1, _obs_dim(s.config))
    prev = s.encode(obs)
    no_anchor = s.encode(obs, prev, self_e1_anchor=None)
    with_anchor = s.encode(obs, prev, self_e1_anchor=torch.randn(1, s.config.self_dim))
    assert torch.allclose(no_anchor.z_self, with_anchor.z_self)
    assert with_anchor.self_recurrence_diag["anchor_present"] is False


# ---------------------------------------------------------------------------
# Agent-level plumbing: E1-tick anchor cache -> next encode
# ---------------------------------------------------------------------------

def _agent(seed: int = 0, **flags):
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
    return agent, env, b, w


def test_agent_off_never_populates_anchor_cache():
    agent, _env, b, w = _agent()
    assert agent.latent_stack.self_recurrence is None
    for _ in range(6):
        with torch.no_grad():
            agent.act_with_split_obs(b, w)
    assert agent._e1_predicted_next_z_self is None
    assert agent._current_latent.self_recurrence_diag is None


def test_agent_on_populates_anchor_and_runs_end_to_end():
    agent, _env, b, w = _agent(use_self_recurrence=True)
    assert agent.latent_stack.self_recurrence is not None
    # run several full waking ticks (sense + _e1_tick + ...)
    for _ in range(6):
        with torch.no_grad():
            agent.act_with_split_obs(b, w)
    # the E1-tick has cached a predicted-next z_self for the self-recurrence anchor
    assert agent._e1_predicted_next_z_self is not None
    assert agent._e1_predicted_next_z_self.shape[-1] == agent.config.latent.self_dim
    # and the live latent carries the DR-13 readout
    d = agent._current_latent.self_recurrence_diag
    assert d is not None and d["active"] is True
    assert torch.isfinite(agent._current_latent.z_self).all()
