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


# ----------------------------------------------------------------------
# ARC-063 amend (V3-EXQ-654b GAP-B maturity): mature-pool gate/credit/retire
# dynamics so a differentiated, persistently-active pool of >=2 rules can form.
# ----------------------------------------------------------------------
def _collapsed_regime():
    """Two context regimes at cosine ~0.7 (collapsed-but-distinguishable z_world,
    the 654b monostrategy signature). Distinct recurrence keys come from the
    differing action class, not the bucket."""
    import math
    torch.manual_seed(1)
    base = torch.randn(16); base = base / base.norm()
    perp = torch.randn(16); perp = perp - (perp @ base) * base; perp = perp / perp.norm()
    ab = 0.7
    ctxA = base.clone()
    ctxB = ab * base + math.sqrt(1 - ab * ab) * perp
    return ctxA / ctxA.norm(), ctxB / ctxB.norm()


def _run_two_regime(f, ticks=400):
    ctxA, ctxB = _collapsed_regime()
    for t in range(ticks):
        ctx, ao = (ctxA, 0) if t % 2 == 0 else (ctxB, 1)
        outcome = -0.6 if (t % 5 == 0) else 0.2  # frequent hazard-env negatives
        f.step(context=ctx, action_object_idx=ao, outcome_signal=outcome)
    return f.get_state()


def test_c12_mature_default_off_bit_identical_and_frac_active_readout():
    # mature_* knobs default to recalibrated values but are INERT when the master
    # flag is off -> a field with absurd mature_* values but mature_pool_dynamics
    # False is bit-identical to the legacy default field on a fixed sequence.
    cfg = CandidateRuleFieldConfig(use_candidate_rule_field=True)
    assert cfg.mature_pool_dynamics is False
    legacy = _field()
    inert = _field(mature_pool_dynamics=False, mature_retire_floor=0.99,
                   mature_tolerance_floor=5.0, mature_mint_block_threshold=0.0)
    s_legacy = _run_two_regime(legacy)
    s_inert = _run_two_regime(inert)
    assert s_legacy["crf_n_slots_minted"] == s_inert["crf_n_slots_minted"]
    assert s_legacy["crf_n_minted_total"] == s_inert["crf_n_minted_total"]
    assert abs(s_legacy["crf_max_pairwise_rule_dist"]
               - s_inert["crf_max_pairwise_rule_dist"]) < 1e-9
    # frac_active readout (the CRF-readiness gate input) is present + in [0, 1].
    assert "crf_frac_active" in s_legacy
    assert 0.0 <= s_legacy["crf_frac_active"] <= 1.0


def test_c13_mature_conflict_gate_admits_two_matched_rules():
    # The latent deadlock: legacy theta = 0.3 + 1.0*n_competing -> 1.3 for two
    # matched rules -> NEITHER can be active. Mature 0.15 + 0.25*n -> 0.40 ->
    # both fire if availability >= 0.4.
    cx = torch.zeros(16); cx[0] = 1.0

    def _two_matched(mature):
        f = _field(n_slots=8, mint_recurrence_threshold=1,
                   context_match_threshold=0.3, mature_pool_dynamics=mature)
        for idx in (0, 1):
            f._rules[idx] = CandidateRule(
                rule_embedding=f._pinned_directions[idx].clone(),
                context_tag=cx.clone(), availability=0.5, eligibility=0.0,
                minted_step=0)
        return f

    assert len(_two_matched(False).gate_and_select(cx)) == 0, \
        "legacy conflict gate deadlocks >=2 matched rules"
    assert len(_two_matched(True).gate_and_select(cx)) == 2, \
        "mature conflict gate admits >=2 matched rules"


def test_c14_mature_breaks_654b_signature_and_clears_readiness_gate():
    # The headline 654b inversion under the collapsed-z_world regime: legacy
    # mint-block (cos 0.7 >= 0.5) blocks the 2nd mint -> 1 rule present ->
    # crf_max_pairwise_rule_dist 0.0 (READY=False). Mature mint-block at 0.8
    # admits the 2nd -> >=2 differentiated co-present rules (READY=True).
    s_legacy = _run_two_regime(_field())
    s_mature = _run_two_regime(_field(mature_pool_dynamics=True))
    assert s_legacy["crf_max_pairwise_rule_dist"] == 0.0
    assert s_legacy["crf_n_slots_minted"] == 1
    assert s_mature["crf_n_slots_minted"] >= 2
    assert s_mature["crf_max_pairwise_rule_dist"] > 0.0
    # CRF-readiness gate: max_pairwise_dist > floor AND frac_active >= 0.30.
    legacy_ready = (s_legacy["crf_max_pairwise_rule_dist"] > 0.0
                    and s_legacy["crf_frac_active"] >= 0.30)
    mature_ready = (s_mature["crf_max_pairwise_rule_dist"] > 0.0
                    and s_mature["crf_frac_active"] >= 0.30)
    assert legacy_ready is False
    assert mature_ready is True


def test_c15_mature_retire_churn_youth_protection_and_asymmetric_credit():
    cx = torch.zeros(16); cx[0] = 1.0

    # Mint-youth protection: a fresh rule below the retire floor survives within
    # mint_protection_ticks under mature; legacy retires it immediately.
    def _fresh_below_floor(mature):
        f = _field(mint_recurrence_threshold=1, mature_pool_dynamics=mature,
                   mature_mint_protection_ticks=30, mature_retire_floor=0.05)
        f._rules[0] = CandidateRule(
            rule_embedding=f._pinned_directions[0].clone(),
            context_tag=cx.clone(), availability=0.02, eligibility=0.0,
            minted_step=0)
        f.credit(outcome_signal=-1.0, step=5)  # 5 ticks old -> within protection
        return 0 in f._rules

    assert _fresh_below_floor(True) is True, "mature protects a fresh below-floor rule"
    assert _fresh_below_floor(False) is False, "legacy retires it (floor 0.15)"

    # Asymmetric negative credit: a negative outcome erodes availability less
    # under mature (alpha_neg 0.02) than legacy (alpha 0.1).
    def _neg_credit(mature):
        f = _field(mature_pool_dynamics=mature,
                   mature_availability_alpha_negative=0.02)
        f._rules[0] = CandidateRule(
            rule_embedding=f._pinned_directions[0].clone(),
            context_tag=cx.clone(), availability=0.8, eligibility=1.0,
            minted_step=0)
        f.credit(outcome_signal=-1.0, step=1)
        return f._rules[0].availability

    assert _neg_credit(True) > _neg_credit(False), \
        "mature negative credit is gentler -> higher residual availability"


def test_c16_mature_and_context_flags_from_dims_and_agent_wiring():
    # from_dims default OFF for both new flags.
    cfg_off = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=20, action_dim=4)
    assert cfg_off.crf_mature_pool_dynamics is False
    assert cfg_off.crf_context_from_e2_world_forward is False
    # Explicit True propagates onto the agent's field config (mature) + REEConfig
    # (context routing, read at the tick site), and the agent runs an act tick
    # exercising the mature dynamics + e2-world-forward context path without error.
    agent, env = _build_agent(use_candidate_rule_field=True,
                              use_lateral_pfc_analog=True,
                              crf_mint_recurrence_threshold=2,
                              crf_mature_pool_dynamics=True,
                              crf_context_from_e2_world_forward=True)
    assert agent.candidate_rule_field.config.mature_pool_dynamics is True
    assert agent.config.crf_context_from_e2_world_forward is True
    _flat, obs = env.reset()
    body = obs["body_state"]; world = obs["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    with torch.no_grad():
        for _ in range(20):
            act = agent.act_with_split_obs(body, world)
            _flat, _h, _d, _i, obs = env.step(int(act.argmax().item()))
            body = obs["body_state"]; world = obs["world_state"]
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)
    st = agent.candidate_rule_field.get_state()
    assert "crf_frac_active" in st
    assert st["crf_step"] > 0


# ----------------------------------------------------------------------
# crf-availability-maintenance (V3-EXQ-666 successor; ARC-063 amend)
# ----------------------------------------------------------------------
def _mint_two_then_silence(f, silent_ticks):
    """Mint two distinct rules (A, B) then run many context-absent ticks feeding a
    third orthogonal context C with a UNIQUE action-object each tick -- so C never
    mints and never matches A/B. This is the sparse-matching regime the
    V3-EXQ-666 differentiation<->persistence tension lives in: A and B are
    differentiated but their context does not recur, so without maintenance their
    availability erodes under the per-tick silence decay and they fall out of the
    reactivatable pool (legacy mature), while activity-silent maintenance HOLDS
    them (Mongillo)."""
    A = torch.zeros(16); A[0] = 1.0; A[1] = 1.0; A = A / A.norm()
    B = torch.zeros(16); B[2] = 1.0; B[3] = 1.0; B = B / B.norm()
    C = torch.zeros(16); C[4] = 1.0; C[5] = -1.0; C = C / C.norm()
    for _ in range(4):
        f.step(A, action_object_idx=0, outcome_signal=0.5)
    for _ in range(4):
        f.step(B, action_object_idx=1, outcome_signal=0.5)
    for t in range(silent_ticks):
        f.step(C, action_object_idx=1000 + t, outcome_signal=0.0)
    return f.get_state()


def test_c17_maintenance_default_off_bit_identical_and_readout_keys():
    # Default OFF (and inert even with absurd maintenance_* values when the master
    # flag is off) -> bit-identical to legacy on a fixed sequence; new maintained-
    # pool readout keys are present, consistent, and bounded.
    cfg = CandidateRuleFieldConfig(use_candidate_rule_field=True)
    assert cfg.availability_maintenance is False
    legacy = _field(mature_pool_dynamics=True)
    inert = _field(mature_pool_dynamics=True, availability_maintenance=False,
                   maintenance_floor=0.99, maintenance_decay=0.5,
                   engaged_sustain=True, engaged_sustain_rate=0.9)
    s_legacy = _run_two_regime(legacy)
    s_inert = _run_two_regime(inert)
    assert s_legacy["crf_n_slots_minted"] == s_inert["crf_n_slots_minted"]
    assert s_legacy["crf_n_minted_total"] == s_inert["crf_n_minted_total"]
    assert abs(s_legacy["crf_frac_active"] - s_inert["crf_frac_active"]) < 1e-9
    assert abs(s_legacy["crf_max_pairwise_rule_dist"]
               - s_inert["crf_max_pairwise_rule_dist"]) < 1e-9
    # maintained-pool readout keys present + consistent (always emitted).
    for k in ("crf_n_maintained_reactivatable", "crf_maintained_pairwise_dist",
              "crf_frac_maintained", "crf_maintained_reactivation_threshold"):
        assert k in s_legacy
    assert s_legacy["crf_n_maintained_reactivatable"] == len(
        legacy.maintained_reactivatable_rules())
    assert 0.0 <= s_legacy["crf_frac_maintained"] <= 1.0


def test_c18_maintenance_holds_differentiated_pool_under_sparse_matching():
    # The 666 fix: under sparse matching, activity-silent maintenance HOLDS a
    # differentiated >=2-rule reactivatable pool where the legacy mature path lets
    # it erode out of the reactivatable set.
    legacy = _field(mature_pool_dynamics=True, availability_maintenance=False)
    maint = _field(mature_pool_dynamics=True, availability_maintenance=True,
                   maintenance_floor=0.45, maintenance_decay=0.0)
    s_legacy = _mint_two_then_silence(legacy, silent_ticks=3000)
    s_maint = _mint_two_then_silence(maint, silent_ticks=3000)
    # Both mint two differentiated rules at the start.
    assert s_maint["crf_max_pairwise_rule_dist"] > 0.1
    # Maintenance: both rules stay maintained-and-reactivatable across the silence.
    assert s_maint["crf_n_maintained_reactivatable"] >= 2
    assert s_maint["crf_maintained_pairwise_dist"] > 0.1  # the readiness gate
    # Legacy mature: silence erodes the pool below the reactivation floor (and/or
    # retires it) -> fewer than two maintained-reactivatable rules.
    assert s_legacy["crf_n_maintained_reactivatable"] < 2
    # The CRF-readiness gate (maintained-pool form) clears for maintenance, not legacy.
    maint_ready = (s_maint["crf_maintained_pairwise_dist"] > 0.1
                   and s_maint["crf_n_maintained_reactivatable"] >= 2)
    legacy_ready = (s_legacy["crf_maintained_pairwise_dist"] > 0.1
                    and s_legacy["crf_n_maintained_reactivatable"] >= 2)
    assert maint_ready is True
    assert legacy_ready is False


def test_c19_maintenance_flags_from_dims_and_agent_wiring():
    cfg_off = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=20, action_dim=4)
    assert cfg_off.crf_availability_maintenance is False
    agent, env = _build_agent(use_candidate_rule_field=True,
                              use_lateral_pfc_analog=True,
                              crf_mint_recurrence_threshold=2,
                              crf_mature_pool_dynamics=True,
                              crf_context_from_e2_world_forward=True,
                              crf_availability_maintenance=True,
                              crf_maintenance_floor=0.5)
    f = agent.candidate_rule_field
    assert f.config.availability_maintenance is True
    assert abs(f.config.maintenance_floor - 0.5) < 1e-9
    _flat, obs = env.reset()
    body = obs["body_state"]; world = obs["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    with torch.no_grad():
        for _ in range(20):
            act = agent.act_with_split_obs(body, world)
            _flat, _h, _d, _i, obs = env.step(int(act.argmax().item()))
            body = obs["body_state"]; world = obs["world_state"]
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)
    st = f.get_state()
    assert "crf_n_maintained_reactivatable" in st
    assert "crf_maintained_pairwise_dist" in st
