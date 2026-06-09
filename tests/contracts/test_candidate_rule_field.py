"""Contract tests for ARC-063 v1 CandidateRuleField (GAP-B rule-creator).

C1  default-off no-op: agent.candidate_rule_field is None; from_dims default OFF.
C2  precondition: field requires use_lateral_pfc_analog (loud ValueError).
C3  CREATE: minting fires after recurrence on >=2 distinct context regimes,
    with distinct (subspace-partitioned) rule embeddings.
C4  OUTPUT: distinct contexts yield distinct (differentiated) rule_state vectors
    -- the structural inversion of the 598b C3 trainable_not_monomodal collapse.
C5  GATE: tolerance threshold is conflict-sensitive (rises with competing
    context-matched rules); single matched rule (no conflict) is admitted.
C6  CREDIT: a success outcome raises an eligible rule's availability.
C7  MECH-094: simulation_mode is a no-op (returns zeros, no mint, no credit).
C8  agent ON: lateral_pfc sources rule_state from the field; field mints over an
    episode and SD-033a rule_state is populated.
"""

import sys

import torch

sys.path.insert(0, "/Users/dgolden/REE_Working/ree-v3")

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.policy.candidate_rule_field import (
    CandidateRule,
    CandidateRuleField,
    CandidateRuleFieldConfig,
)
from ree_core.utils.config import REEConfig


def _field(**kw):
    cfg = CandidateRuleFieldConfig(use_candidate_rule_field=True, **kw)
    return CandidateRuleField(context_dim=16, config=cfg)


def _build_agent(seed=7, **flags):
    torch.manual_seed(seed)
    env = CausalGridWorldV2(seed=seed, size=5, num_hazards=1, num_resources=1,
                            use_proxy_fields=True)
    cfg = REEConfig.from_dims(body_obs_dim=env.body_obs_dim,
                              world_obs_dim=env.world_obs_dim, action_dim=4,
                              self_dim=16, world_dim=16, **flags)
    agent = REEAgent(cfg)
    agent.reset()
    return agent, env


# ----------------------------------------------------------------------
def test_c1_default_off_no_op():
    agent, _ = _build_agent()
    assert agent.candidate_rule_field is None
    cfg = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=20, action_dim=4)
    assert cfg.use_candidate_rule_field is False


def test_c2_precondition_requires_lateral_pfc():
    raised = False
    try:
        _build_agent(use_candidate_rule_field=True)  # no lateral_pfc
    except ValueError:
        raised = True
    assert raised, "field without use_lateral_pfc_analog must raise"


def test_c3_create_mints_distinct_context_rules():
    f = _field(n_slots=8, rule_dim=16, mint_recurrence_threshold=3,
               context_match_threshold=0.5)
    a = torch.zeros(16); a[0] = 1.0; a[1] = 1.0
    b = torch.zeros(16); b[0] = -1.0; b[1] = -1.0
    for _ in range(4):
        f.step(a, action_object_idx=0)
    for _ in range(4):
        f.step(b, action_object_idx=1)
    st = f.get_state()
    assert st["crf_n_slots_minted"] >= 2
    assert st["crf_max_pairwise_rule_dist"] > 0.1  # distinct pinned directions


def test_c4_output_differentiated_rule_state():
    f = _field(n_slots=8, rule_dim=16, mint_recurrence_threshold=3,
               context_match_threshold=0.5)
    a = torch.zeros(16); a[0] = 1.0; a[1] = 1.0
    b = torch.zeros(16); b[0] = -1.0; b[1] = -1.0
    for _ in range(4):
        f.step(a, action_object_idx=0)
    for _ in range(4):
        f.step(b, action_object_idx=1)
    sA = f.step(a, action_object_idx=0)
    sB = f.step(b, action_object_idx=1)
    assert tuple(sA.shape) == (1, 16)
    assert float((sA - sB).norm().item()) > 1e-4  # differentiated by context


def test_c5_gate_conflict_sensitive():
    f = _field(n_slots=8, rule_dim=16, mint_recurrence_threshold=1,
               tolerance_floor=0.3, tolerance_conflict_gain=1.0,
               context_match_threshold=0.3)
    cx = torch.zeros(16); cx[0] = 1.0
    f.step(cx, action_object_idx=0)
    # Insert a second rule sharing the same context (bypass the covered-guard).
    f._rules[1] = CandidateRule(
        rule_embedding=f._pinned_directions[1].clone(),
        context_tag=cx.clone(), availability=0.3, eligibility=0.0, minted_step=0)
    assert len(f.gate_and_select(cx)) == 0, "conflict raises theta -> both held out"
    f._rules.pop(1)
    assert len(f.gate_and_select(cx)) == 1, "single matched rule is admitted"


def test_c6_credit_raises_availability_on_success():
    f = _field(n_slots=8, rule_dim=16, mint_recurrence_threshold=1,
               context_match_threshold=0.3, availability_alpha=0.5)
    cx = torch.zeros(16); cx[0] = 1.0
    f.step(cx, action_object_idx=0)
    rule = list(f._rules.values())[0]
    av0 = rule.availability
    f.gate_and_select(cx)  # marks active + eligibility=1
    f.credit(outcome_signal=1.0)
    assert rule.availability > av0


def test_c7_mech094_simulation_no_op():
    f = _field(n_slots=8, rule_dim=16, mint_recurrence_threshold=1)
    cx = torch.zeros(16); cx[0] = 1.0
    out = f.step(cx, action_object_idx=0, simulation_mode=True)
    assert float(out.abs().sum()) == 0.0
    assert f.get_state()["crf_n_minted_total"] == 0
    assert f.get_state()["crf_n_simulation_skipped"] == 1


def test_c8_agent_on_sources_and_mints():
    agent, env = _build_agent(use_candidate_rule_field=True,
                              use_lateral_pfc_analog=True,
                              crf_mint_recurrence_threshold=2)
    assert agent.candidate_rule_field is not None
    assert agent.lateral_pfc.config.use_candidate_rule_source is True
    _flat, obs = env.reset()
    body = obs["body_state"]; world = obs["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    with torch.no_grad():
        for _ in range(25):
            act = agent.act_with_split_obs(body, world)
            _flat, _h, _d, _i, obs = env.step(int(act.argmax().item()))
            body = obs["body_state"]; world = obs["world_state"]
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)
    st = agent.candidate_rule_field.get_state()
    assert st["crf_n_minted_total"] >= 1
    assert float(agent.lateral_pfc.rule_state.norm().item()) > 0.0


# ----------------------------------------------------------------------
# ARC-062 amend (V3-EXQ-654 GAP-B maturity): cross-episode rule persistence.
# ----------------------------------------------------------------------
def _matured_field(persist):
    """Build a field with two recurring context regimes minted, return it."""
    f = _field(n_slots=8, rule_dim=16, mint_recurrence_threshold=3,
               context_match_threshold=0.5,
               persist_rules_across_episode_reset=persist)
    a = torch.zeros(16); a[0] = 1.0; a[1] = 1.0
    b = torch.zeros(16); b[0] = -1.0; b[1] = -1.0
    for _ in range(4):
        f.step(a, action_object_idx=0)
    for _ in range(4):
        f.step(b, action_object_idx=1)
    return f


def test_c9_persist_default_off_is_bit_identical_wipe():
    # Default OFF: reset() clears the live pool + recurrence + clock (legacy).
    cfg = CandidateRuleFieldConfig(use_candidate_rule_field=True)
    assert cfg.persist_rules_across_episode_reset is False
    f = _matured_field(persist=False)
    assert f.get_state()["crf_n_slots_minted"] >= 2
    f.reset()
    st = f.get_state()
    assert st["crf_n_slots_minted"] == 0
    assert len(f._rules) == 0
    assert len(f._recurrence) == 0
    assert st["crf_step"] == 0


def test_c10_persist_on_pool_survives_reset_and_keeps_maturing():
    # ON: reset() is a no-op -- pool, recurrence counters, and clock persist
    # across the per-episode boundary so the field matures across episodes.
    f = _matured_field(persist=True)
    minted_before = f.get_state()["crf_n_slots_minted"]
    step_before = f.get_state()["crf_step"]
    rec_before = dict(f._recurrence)
    assert minted_before >= 2
    f.reset()  # the per-episode wipe -- must NOT clear when persisting
    st = f.get_state()
    assert st["crf_n_slots_minted"] == minted_before
    assert len(f._rules) == minted_before
    assert f._recurrence == rec_before
    assert st["crf_step"] == step_before  # clock continues monotonically
    # A post-reset tick in a previously-seen regime keeps the live pool alive
    # (recurrence already cleared the mint threshold; no cold-start).
    a = torch.zeros(16); a[0] = 1.0; a[1] = 1.0
    out = f.step(a, action_object_idx=0)
    assert tuple(out.shape) == (1, 16)
    assert f.get_state()["crf_n_slots_minted"] >= minted_before


def test_c11_config_wiring_from_dims_and_agent():
    # from_dims default OFF; explicit True propagates onto the agent's field.
    cfg_off = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=20, action_dim=4)
    assert cfg_off.crf_persist_rules_across_episode_reset is False
    agent, _ = _build_agent(use_candidate_rule_field=True,
                            use_lateral_pfc_analog=True,
                            crf_persist_rules_across_episode_reset=True)
    assert agent.candidate_rule_field is not None
    assert (agent.candidate_rule_field.config
            .persist_rules_across_episode_reset is True)
