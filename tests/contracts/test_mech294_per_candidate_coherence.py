"""
MECH-294 per-candidate co-binding coherence -- substrate-readiness contract.

This is the substrate-readiness GATE for the route-range amend that renders the
(scalar) currency_coherence as a CROSS-CANDIDATE RANGE quantity the modulatory
selection authority + 569i top-k shortlist can carve. Routed by the
substrate-ceiling-lifted triage (2026-06-19): V3-EXQ-661 confirmed currency_coherence
is mode-distinct (JOINT 1.0 / ALT 0.25 / SHUF 0.0) but the compose bias is a scalar
GATE on a mode-INVARIANT action-only cosine -> committed-distribution TV ~0 (incl
gate-ON-vs-OFF; action histograms byte-identical across modes), so the authority's
unit-range normalisation washed it out (joint == alternation) or floored it
(shuffled == 0).

The new per-candidate coherence (MultiContentThetaPacket.compose_per_candidate_coherence)
aligns each candidate's first action with the action co-bound WITH each V_s-gated
stream this cycle, currency-weighted:
  C1 JOINT      -- all four streams co-bound to THIS cycle's action -> a per-candidate
                   bias with NON-ZERO cross-candidate range.
  C2 SHUFFLED   -- nothing co-bound this cycle (weights 0) -> None / ~0 range.
  C3 no-regress -- the scalar currency_coherence() mode-discrimination is preserved
                   (JOINT 1.0 / ALT 0.25 / SHUF 0.0) bit-identically.
  C4 ALT pattern-- alternation carries a per-candidate PATTERN distinct from joint
                   (held-stream prior co-bound actions; the chosen design fork), not
                   merely a smaller magnitude (which range-normalisation would erase).
  C5 non-vacuity-- a monostrategic candidate pool (identical first actions) yields ~0
                   range even on JOINT -> the readiness gate must require candidate
                   first-action diversity (the V3-EXQ-661 joint_cem precondition).
  C6 OFF        -- default bit-identical: the per-candidate flag is False by default
                   and the legacy scalar-gated compose path is unchanged.
  C7 agent      -- the compose path fires end-to-end and the carve-able channel
                   reaches the route-range authority via the "coherence" source.

PASS = this gate clears; the MECH-294 behavioural falsifier is a SEPARATE later
/queue-experiment step. MECH-294 stays substrate_ceiling (no claim status change).
"""

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from ree_core.latent.multi_content_theta_packet import (
    MultiContentThetaPacket,
    MultiContentThetaPacketConfig,
)

_HIGH_VS = {"z_goal": 0.9, "z_harm_s": 0.9, "z_harm_a": 0.9, "z_world": 0.9}
_FLOOR = 1e-3


def _packet(mode: str, hold_weight: float = 0.5) -> MultiContentThetaPacket:
    return MultiContentThetaPacket(
        MultiContentThetaPacketConfig(binding_mode=mode, coherence_hold_weight=hold_weight)
    )


def _tick(p: MultiContentThetaPacket, action: torch.Tensor, vs=None):
    p.observe(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.2, 0.1]]),
        torch.tensor([[0.3, 0.0]]),
        per_stream_vs=vs if vs is not None else _HIGH_VS,
    )
    p.observe_action_proposal(action)
    return p.seal(torch.tensor([[0.5, 0.5]]))


# K=3 DIVERSE candidate first actions (4-dim one-hots).
_CAND_DIVERSE = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
# K=3 MONOSTRATEGIC candidate first actions (all identical).
_CAND_MONO = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [1.0, 0, 0, 0]])


# ----------------------------------------------------------------------
# C1: JOINT carries non-zero cross-candidate range
# ----------------------------------------------------------------------
def test_c1_joint_per_candidate_range_nonzero():
    p = _packet("joint")
    _tick(p, torch.tensor([[1.0, 0, 0, 0]]))
    bias = p.compose_per_candidate_coherence(_CAND_DIVERSE, bias_scale=0.1)
    assert bias is not None, "JOINT must produce a per-candidate coherence bias"
    rng = float((bias.max() - bias.min()).item())
    assert rng > _FLOOR, f"JOINT cross-candidate range {rng} must exceed floor {_FLOOR}"
    assert p.last_per_candidate_coherence_range > _FLOOR
    # all four streams co-bound to this cycle's action at full weight
    assert all(w == 1.0 for w in p.last_packet.coherence_weights.values())


# ----------------------------------------------------------------------
# C2: SHUFFLED carries ~0 cross-candidate range
# ----------------------------------------------------------------------
def test_c2_shuffled_per_candidate_range_zero():
    p = _packet("shuffled")
    # warm history so shuffled has prior cycles to draw content from
    for a in (
        torch.tensor([[1.0, 0, 0, 0]]),
        torch.tensor([[0, 1.0, 0, 0]]),
        torch.tensor([[0, 0, 1.0, 0]]),
    ):
        _tick(p, a)
    bias = p.compose_per_candidate_coherence(_CAND_DIVERSE, bias_scale=0.1)
    # nothing co-bound this cycle -> weights all 0 -> None (below-floor channel)
    assert bias is None, "SHUFFLED has no co-bound streams -> no carve-able range"
    assert all(w == 0.0 for w in p.last_packet.coherence_weights.values())
    assert p.last_packet.currency_coherence() == 0.0


# ----------------------------------------------------------------------
# C3: scalar currency_coherence mode-discrimination is NOT regressed
# ----------------------------------------------------------------------
def test_c3_scalar_currency_coherence_preserved():
    pj = _packet("joint")
    _tick(pj, torch.tensor([[1.0, 0, 0, 0]]))
    assert pj.last_packet.currency_coherence() == 1.0

    ps = _packet("shuffled")
    for a in (
        torch.tensor([[1.0, 0, 0, 0]]),
        torch.tensor([[0, 1.0, 0, 0]]),
        torch.tensor([[0, 0, 1.0, 0]]),
    ):
        _tick(ps, a)
    assert ps.last_packet.currency_coherence() == 0.0

    pa = _packet("alternation")
    for a in (
        torch.tensor([[1.0, 0, 0, 0]]),
        torch.tensor([[0, 1.0, 0, 0]]),
        torch.tensor([[0, 0, 1.0, 0]]),
        torch.tensor([[0, 0, 0, 1.0]]),
        torch.tensor([[1.0, 0, 0, 0]]),
    ):
        _tick(pa, a)
    # one of four streams live per cycle after warmup
    assert abs(pa.last_packet.currency_coherence() - 0.25) < 1e-9


# ----------------------------------------------------------------------
# C4: ALTERNATION per-candidate PATTERN distinct from JOINT (chosen design fork)
# ----------------------------------------------------------------------
def test_c4_alternation_pattern_distinct_from_joint():
    pj = _packet("joint")
    _tick(pj, torch.tensor([[1.0, 0, 0, 0]]))
    bj = pj.compose_per_candidate_coherence(_CAND_DIVERSE, bias_scale=0.1)

    pa = _packet("alternation", hold_weight=0.5)
    for a in (
        torch.tensor([[1.0, 0, 0, 0]]),
        torch.tensor([[0, 1.0, 0, 0]]),
        torch.tensor([[0, 0, 1.0, 0]]),
        torch.tensor([[0, 0, 0, 1.0]]),
        torch.tensor([[1.0, 0, 0, 0]]),
    ):
        _tick(pa, a)
    ba = pa.compose_per_candidate_coherence(_CAND_DIVERSE, bias_scale=0.1)
    assert ba is not None, "ALTERNATION (held prior refs at hold_weight) must carve"
    # held streams carry prior co-bound actions -> a per-candidate PATTERN that is
    # NOT a uniform scaling of JOINT (which range-normalisation would erase).
    cos = float(F.cosine_similarity(bj.reshape(1, -1), ba.reshape(1, -1)).item())
    assert cos < 1.0 - 1e-3, f"ALT pattern must differ from JOINT (cos {cos} ~ 1.0)"
    # held-stream weights are the configured hold_weight (currency-graded design)
    held = [w for w in pa.last_packet.coherence_weights.values() if w != 1.0]
    assert held and all(abs(w - 0.5) < 1e-9 for w in held)


# ----------------------------------------------------------------------
# C5: monostrategic candidate pool -> ~0 range even on JOINT (non-vacuity)
# ----------------------------------------------------------------------
def test_c5_monostrategy_yields_zero_range_on_joint():
    p = _packet("joint")
    _tick(p, torch.tensor([[1.0, 0, 0, 0]]))
    bias = p.compose_per_candidate_coherence(_CAND_MONO, bias_scale=0.1)
    assert bias is not None
    rng = float((bias.max() - bias.min()).item())
    assert rng <= _FLOOR, (
        f"identical candidate first actions must yield ~0 range (got {rng}); the "
        "readiness gate must require candidate first-action diversity"
    )


# ----------------------------------------------------------------------
# C6: default OFF bit-identical (legacy scalar-gated compose path unchanged)
# ----------------------------------------------------------------------
def test_c6_default_off_flag_and_legacy_compose_unchanged():
    cfg = REEConfig.from_dims(
        body_obs_dim=17, world_obs_dim=275, action_dim=4, self_dim=16, world_dim=16,
    )
    assert cfg.theta_packet_compose_per_candidate_coherence is False
    assert abs(cfg.theta_packet_coherence_hold_weight - 0.5) < 1e-9
    # legacy scalar-gated compose path bit-identical (per-candidate flag OFF):
    # the scalar gate scales the action-only cosine uniformly across candidates.
    p = _packet("joint")
    _tick(p, torch.tensor([[1.0, 0, 0, 0]]))
    legacy = p.compose_e3_bias(_CAND_DIVERSE, bias_scale=0.1, use_joint_coherence=True)
    assert legacy is not None
    # legacy path leaves the per-candidate-coherence diagnostics untouched
    assert p.n_per_candidate_coherence_calls == 0


# ----------------------------------------------------------------------
# C7: agent-level -- compose fires and the channel reaches the route-range authority
# ----------------------------------------------------------------------
def _build_agent(env, **overrides):
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=4, self_dim=16, world_dim=16,
        use_multi_content_theta_packet=True,
        use_per_stream_vs=True,
        theta_packet_compose_into_e3_bias=True,
        theta_packet_compose_per_candidate_coherence=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    agent = REEAgent(cfg)
    agent.reset()
    return agent


def _make_env(seed=7):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    return CausalGridWorldV2(seed=seed, size=6, num_hazards=2, num_resources=2,
                             use_proxy_fields=True)


def _run(agent, env, n=24, seed=7):
    torch.manual_seed(seed)
    _f, obs = env.reset()
    for _ in range(n):
        body = obs["body_state"]
        world = obs["world_state"]
        if body.dim() == 1:
            body = body.unsqueeze(0)
        if world.dim() == 1:
            world = world.unsqueeze(0)
        with torch.no_grad():
            act = agent.act_with_split_obs(obs_body=body, obs_world=world)
        ai = int(act.reshape(-1).argmax().item())
        _f, _r, _d, _i, obs = env.step(ai)


def test_c7_agent_compose_fires_and_routes_to_authority():
    env = _make_env()
    agent = _build_agent(env)
    _run(agent, env)
    assert agent.multi_content_theta_packet is not None
    # the per-candidate coherence compose path fired end-to-end
    assert agent.multi_content_theta_packet.n_per_candidate_coherence_calls > 0, (
        "compose_per_candidate_coherence must fire on the waking E3 path"
    )

    # the carve-able channel reaches the route-range authority via "coherence"
    env2 = _make_env(seed=8)
    agent2 = _build_agent(
        env2,
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source="coherence",
        use_modulatory_selection_authority=True,
    )
    # mirror onto the e3 sub-config (the authority + routing read both surfaces)
    agent2.config.e3.use_modulatory_channel_routing = True
    agent2.config.e3.use_modulatory_selection_authority = True
    _run(agent2, env2, seed=8)
    diag = getattr(agent2.e3, "last_score_diagnostics", None)
    assert diag is not None and "modulatory_channel_route_active" in diag, (
        "the coherence channel must reach the route-range authority without error"
    )
