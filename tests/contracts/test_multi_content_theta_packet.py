"""Contract tests for MECH-294 multi-content theta-burst packet.

Design memo (the contract):
  REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md

Contracts (memo S8):
  C1: default-OFF no-op / bit-identical. With use_multi_content_theta_packet
      False, agent.multi_content_theta_packet is None and the MECH-089
      ThetaBuffer behaviour + action stream is bit-identical.
  C2: ON builds a sealed packet at the E3 boundary with the populated
      content sub-slots (risk_sensory, risk_affective, state_summary,
      action_proposal under the full-sense path).
  C3: V_s-held substitution fires when a stream's V_s drops below the hold
      threshold; the component is marked stale with a non-zero vintage age and
      carries the prior snapshot value.
  C4: joint vs alternation vs shuffled produce structurally distinct packets
      from the SAME input stream (the substrate-side discriminability the
      validation experiment depends on).
  C5: MECH-094 simulation_mode no-op (observe / observe_action_proposal / seal
      all no-op under simulation_mode=True).
  C6: action_conditioned_on returns the action slot annotated with the
      same-cycle goal + risk.
  C7: agent precondition -- use_multi_content_theta_packet=True without
      use_per_stream_vs=True raises ValueError at construction (loud-not-silent).
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
from ree_core.latent.multi_content_theta_packet import (
    MultiContentThetaPacket,
    MultiContentThetaPacketConfig,
    ThetaPacket,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _packet(mode: str = "joint", refresh: float = 0.5, hold: float = 0.4):
    return MultiContentThetaPacket(
        MultiContentThetaPacketConfig(
            binding_mode=mode,
            snapshot_refresh_threshold=refresh,
            hold_threshold=hold,
        )
    )


_HIGH_VS = {"z_goal": 0.9, "z_harm_s": 0.9, "z_harm_a": 0.9, "z_world": 0.9}


def _feed(p, g, hs, ha, st, act, vs=None, sim=False):
    p.observe(g, hs, ha, vs if vs is not None else _HIGH_VS, simulation_mode=sim)
    p.observe_action_proposal(act, simulation_mode=sim)
    return p.seal(st, simulation_mode=sim)


def _build_run(seed, n, **flags):
    """Seed -> build -> run n waking ticks via act_with_split_obs. Returns the
    action stream (for bit-identical comparisons)."""
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=6, num_hazards=2, num_resources=2, use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=4, self_dim=16, world_dim=16, **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _f, obs = env.reset()
    acts = []
    for _ in range(n):
        body = obs["body_state"]
        world = obs["world_state"]
        if body.dim() == 1:
            body = body.unsqueeze(0)
        if world.dim() == 1:
            world = world.unsqueeze(0)
        with torch.no_grad():
            act = agent.act_with_split_obs(obs_body=body, obs_world=world)
        acts.append(act.detach().clone())
        ai = int(act.reshape(-1).argmax().item())
        _f, _r, _d, _i, obs = env.step(ai)
    return agent, acts


# ----------------------------------------------------------------------
# C1 default-OFF no-op / bit-identical
# ----------------------------------------------------------------------
def test_c1_default_off_no_op_and_bit_identical():
    agent, _ = _build_run(11, 8)
    assert agent.multi_content_theta_packet is None
    assert agent.last_theta_packet is None

    _, s_default = _build_run(11, 12)
    _, s_explicit = _build_run(11, 12, use_multi_content_theta_packet=False)
    assert all(torch.equal(a, b) for a, b in zip(s_default, s_explicit)), (
        "OFF must be bit-identical (default vs explicit-False)"
    )


# ----------------------------------------------------------------------
# C2 ON builds a sealed packet with content sub-slots
# ----------------------------------------------------------------------
def test_c2_on_builds_sealed_packet_with_slots():
    p = _packet("joint")
    g = torch.randn(1, 8)
    hs = torch.randn(1, 4)
    ha = torch.randn(1, 4)
    st = torch.randn(1, 16)
    act = torch.randn(1, 4)
    pk = _feed(p, g, hs, ha, st, act)
    assert isinstance(pk, ThetaPacket)
    assert pk.goal_latent is not None
    assert pk.risk_sensory is not None
    assert pk.risk_affective is not None
    assert pk.state_summary is not None
    assert pk.action_proposal is not None
    assert pk.is_complete()
    assert p.n_seals == 1
    # type-separated sub-slots (NOT pre-collapsed): risk keeps both SD-011 streams
    assert pk.risk_vector().shape[-1] == hs.shape[-1] + ha.shape[-1]


def test_c2_agent_seal_at_e3_boundary():
    agent, _ = _build_run(
        7, 16, use_multi_content_theta_packet=True, use_per_stream_vs=True,
    )
    assert agent.multi_content_theta_packet is not None
    diag = agent.multi_content_theta_packet.get_diagnostics()
    assert diag["mech294_n_seals"] > 0, "expected at least one E3-boundary seal"
    assert agent.last_theta_packet is not None
    # state_summary always populates (MECH-089 averaged z_world is always present).
    assert agent.last_theta_packet.state_summary is not None


# ----------------------------------------------------------------------
# C3 V_s-held substitution + stale vintage
# ----------------------------------------------------------------------
def test_c3_low_vs_holds_snapshot_and_marks_stale():
    p = _packet("joint", refresh=0.5, hold=0.4)
    g0 = torch.randn(1, 8)
    hs = torch.randn(1, 4)
    ha = torch.randn(1, 4)
    st = torch.randn(1, 16)
    act = torch.randn(1, 4)
    # Cycle 1: high V_s -> goal is current, snapshot refreshed.
    _feed(p, g0, hs, ha, st, act, vs=_HIGH_VS)
    # Cycle 2: drop z_goal V_s below hold; goal must substitute the prior snapshot.
    g1 = torch.randn(1, 8)
    low = dict(_HIGH_VS)
    low["z_goal"] = 0.1
    pk = _feed(p, g1, hs, ha, st, act, vs=low)
    assert pk.is_component_stale("goal_latent"), "low V_s goal must be held/stale"
    assert torch.allclose(pk.goal_latent, g0), "held goal must be the prior snapshot"
    assert pk.vintage["goal_latent"].age_ticks >= 1
    assert not pk.is_component_stale("state_summary"), "high-V_s state stays current"
    assert pk.n_distinct_vintages() >= 2, "packet must carry >= 2 distinct vintages"


# ----------------------------------------------------------------------
# C4 joint / alternation / shuffled structurally distinct
# ----------------------------------------------------------------------
def test_c4_binding_modes_structurally_distinct():
    torch.manual_seed(0)
    seq = [
        (torch.randn(1, 8), torch.randn(1, 4), torch.randn(1, 4),
         torch.randn(1, 16), torch.randn(1, 4))
        for _ in range(8)
    ]

    def run_mode(mode):
        p = _packet(mode)
        pk = None
        for (g, hs, ha, st, act) in seq:
            pk = _feed(p, g, hs, ha, st, act, vs=_HIGH_VS)
        return pk

    J = run_mode("joint")
    A = run_mode("alternation")
    S = run_mode("shuffled")

    def dist(a, b):
        d = 0.0
        for f in ("goal_latent", "risk_sensory", "risk_affective", "state_summary"):
            ta, tb = getattr(a, f), getattr(b, f)
            if ta is not None and tb is not None:
                d += (ta.reshape(-1) - tb.reshape(-1)).abs().sum().item()
        return d

    assert dist(J, A) > 1e-3, "alternation must differ structurally from joint"
    assert dist(J, S) > 1e-3, "shuffled must differ structurally from joint"
    # joint binds the SAME-cycle current goal (within-cycle co-binding).
    assert torch.allclose(J.goal_latent.reshape(-1), seq[-1][0].reshape(-1))


# ----------------------------------------------------------------------
# C5 MECH-094 simulation_mode no-op
# ----------------------------------------------------------------------
def test_c5_simulation_mode_no_op():
    p = _packet("joint")
    n0 = p.n_seals
    pk = _feed(
        p, torch.randn(1, 8), torch.randn(1, 4), torch.randn(1, 4),
        torch.randn(1, 16), torch.randn(1, 4), vs=_HIGH_VS, sim=True,
    )
    assert pk is None, "seal must be a no-op under simulation_mode"
    assert p.n_seals == n0, "no seal may be counted under simulation_mode"
    assert p.last_packet is None
    assert p.n_simulation_skipped >= 3  # observe + observe_action + seal


# ----------------------------------------------------------------------
# C6 action_conditioned_on annotates the action with same-cycle goal+risk
# ----------------------------------------------------------------------
def test_c6_action_conditioned_on_joint_read():
    p = _packet("joint")
    g = torch.randn(1, 8)
    hs = torch.randn(1, 4)
    ha = torch.randn(1, 4)
    st = torch.randn(1, 16)
    act = torch.randn(1, 4)
    pk = _feed(p, g, hs, ha, st, act, vs=_HIGH_VS)
    out = pk.action_conditioned_on(goal=True, risk=True)
    # action_proposal (4) + goal (8) + risk_sensory (4) + risk_affective (4)
    assert out.shape[-1] == 4 + 8 + 4 + 4
    # the leading action_proposal slice is exactly the co-bound action
    assert torch.allclose(out[:, :4], act)
    # action-only read drops goal + risk
    assert pk.action_conditioned_on(goal=False, risk=False).shape[-1] == 4


# ----------------------------------------------------------------------
# C7 agent precondition
# ----------------------------------------------------------------------
def test_c7_precondition_requires_per_stream_vs():
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    env = CausalGridWorldV2(seed=1, size=5, num_hazards=1, num_resources=1,
                            use_proxy_fields=True)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=4, self_dim=16, world_dim=16,
        use_multi_content_theta_packet=True,  # no use_per_stream_vs -> must raise
    )
    with pytest.raises(ValueError):
        REEAgent(cfg)
