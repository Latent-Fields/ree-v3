## ARC-063 v1: distributed CandidateRule field (GAP-B non-Bayesian rule-creator) (2026-06-04)
- ARC-063 (v1): policy.rule_apprehension_layer.candidate_rule_field -- IMPLEMENTED
  2026-06-04. The non-Bayesian rule-CREATOR that resolves
  arc_062_rule_apprehension:GAP-B (MECH-309: "trainers weight rules they do not
  invent"). Mint-then-weight over a subspace-partitioned field: CREATION is a
  non-gradient structural mint event; WEIGHTING is eligibility-trace credit on an
  existing rule's availability. The structural fix for the 543/598b rule_state
  collapse (598b C3 trainable_not_monomodal FAIL: two scalar scoring heads sharing
  a return gradient have an inert head_0==head_1 equilibrium -> the rule_state
  handed to SD-033a collapses). ARC-063 makes differentiation STRUCTURAL: distinct
  rules exist because they were minted as distinct pinned subspace directions
  (Weber 2023), so the rule_state is differentiated by construction.
  Brought V4->V3 2026-05-18; V3-tractable design landed 2026-06-04
  (docs/architecture/arc_063_candidate_rule_field.md). Sleep-vs-waking rule-field
  refinement is a V3 follow-on (NOT V4); social in-face (ARC-077/MECH-337) is the
  only genuinely V4-deferred face.
  Module: ree_core/policy/candidate_rule_field.py (CandidateRuleField +
    CandidateRule + CandidateRuleFieldConfig). Pure-arithmetic regulator (no
    nn.Module, no trained parameters, no gradient flow); sibling pattern to
    noise_floor.py (MECH-313) / structured_curiosity.py (MECH-314) /
    tonic_vigor.py (MECH-320). v1 has NO new trained encoder -> NO phased training.
  Five faces (the CandidateRule unit = rule_embedding [rule_dim] + context_tag
    [world_dim] + availability + eligibility + bookkeeping):
    1. CREATE (MECH-349): mint a slot on a recurring (context-bucket ->
       action-object) regularity >= crf_mint_recurrence_threshold times when no
       existing rule's context_tag already covers it; bottom-up (ARC-064) +
       optional top-down ARC-062 discriminator seed (gating_weight nudges initial
       availability when crf_seed_from_arc062).
    2. REPRESENT (MECH-350): pinned-distinct unit-vector slot directions
       (deterministic seeded basis) -> distinct minted rules occupy distinct
       subspace directions (the anti-monomodal geometry; Weber 2023 / Wallis 2001).
    3. GATE (MECH-351): tolerance-gated availability -- a rule is AVAILABLE only
       when availability >= theta = crf_tolerance_floor + crf_tolerance_conflict_gain
       * n_competing_context_matched_rules (Frank 2006 conflict-graded threshold;
       Cavanagh 2011). availability != selection (Frank/O'Reilly 2001).
    4. SELECT (MECH-338): cue-driven context-bound retrieval -- cosine(context,
       context_tag) >= crf_context_match_threshold.
    5. OUTPUT + CREDIT (MECH-352): the available-AND-context-matched rules combine
       (availability-weighted sum of embeddings) into a differentiated [1, rule_dim]
       rule_state vector for SD-033a; an eligibility trace credits the availability
       of rules active when an outcome arrived (raise on success, lower on
       exception; Brzosko 2015 / Kovach 2012), with slow decay + retire-below-floor.
  GAP-B wiring: LateralPFCAnalog (SD-033a) gains config.use_candidate_rule_source
    (set True at REEAgent.__init__ when the field is on) + an optional crf_source
    kwarg on update(). When supplied, crf_source REPLACES the legacy
    delta_proj/world_proj EMA source -- so rule_state tracks the field's
    differentiated active-rule stack (the literal 598b fix). The agent ticks the
    field in select_action at the lateral_pfc block: builds z_world context,
    credits the previous tick's active rules with this tick's outcome proxy
    (lower harm = success), mints on recurrence (keyed on the prev-tick chosen
    action class, stashed after E3 selection), gates+selects, returns crf_source.
  Config (REEConfig + from_dims; all no-op default, bit-identical OFF):
    use_candidate_rule_field (False) master, crf_n_slots (16), crf_rule_dim (16,
    matches lateral_pfc rule_dim), crf_mint_recurrence_threshold (3),
    crf_tolerance_floor (0.3), crf_tolerance_conflict_gain (1.0),
    crf_availability_alpha (0.1), crf_availability_decay (0.005),
    crf_eligibility_window (20), crf_context_match_threshold (0.5),
    crf_seed_from_arc062 (True).
  Precondition (loud ValueError at __init__): use_candidate_rule_field=True
    requires use_lateral_pfc_analog=True (SD-033a is the consumer). Matches the
    use_closure_operator / MECH-269b / MECH-293 precondition pattern.
  MECH-094: every state-advancing method takes simulation_mode and is a no-op when
    True (returns zeros, no mint, no credit); the existing MECH-319 _lpfc_skip gate
    also covers the lateral_pfc write site. Replay/DMN paths never mint or credit.
  Backward compatible: use_candidate_rule_field=False by default ->
    agent.candidate_rule_field is None; lateral_pfc.update sees crf_source=None and
    runs the legacy source; field bit-identical OFF. Regression: 7/7 preflight +
    775/782 contracts PASS (the 7 fails are the pre-existing local-git-env
    runner-conflict-recovery artifact "Not a valid object name master", zero
    overlap with this change). New contracts:
    tests/contracts/test_candidate_rule_field.py 8/8 (C1 default-off no-op / C2
    precondition / C3 CREATE distinct mints / C4 OUTPUT differentiated rule_state /
    C5 GATE conflict-sensitive / C6 CREDIT raises availability / C7 MECH-094 sim
    no-op / C8 agent ON sources + mints).
  Activation smoke (2026-06-04): field mints distinct context rules, rule_state
    differentiates across contexts (the C1 falsifier inversion), conflict gate
    holds under-supported rules out, credit raises availability, sim_mode no-op,
    agent ON populates SD-033a rule_state norm > 0.
  Phased training: NOT required (pure-arithmetic + buffers; no learned params).
    The learned-affordance rule-embedding upgrade WOULD need P0/P1/P2.
  DEFERRED (NOT in v1): sleep-vs-waking rule-field refinement pass (a V3 follow-on
    within ARC-063 reusing the MECH-272/273/285 sleep cluster -- NOT V4); full
    per-action moral-residue evidence records; learned rule-embedding; the social
    in-face (ARC-077/MECH-337, the only V4-by-substrate-necessity face).
  Validation experiment: V3-EXQ-639 substrate-readiness diagnostic (claim_ids=[];
    C1 differentiated rule_state >=2 distinct active rule vectors / C2 minting
    fires >=2 distinct context rules / C3 tolerance gate conflict-sensitive /
    C4 OFF bit-identical). The C4 ARC-062 GAP-B behavioural re-run on the
    field-populated substrate is the governance-weighting successor, queued
    separately.
  Design doc: REE_assembly/docs/architecture/arc_063_candidate_rule_field.md
  See ARC-063 (this), MECH-349 (CREATE) / MECH-350 (REPRESENT) / MECH-351 (GATE) /
    MECH-352 (CREDIT), MECH-338 (SELECT face), MECH-309 (logical necessity -- the
    creator is its answer), ARC-062 (weak-reading top-down source; exhausted
    alone -- GatedPolicy), ARC-064 (bottom-up source), SD-033a (rule_state
    consumer -- GAP-B target), MECH-318 (rule_state abstraction; retire-vs-promote
    downstream), MECH-319 (simulation-mode rule-write gate; covers the lateral_pfc
    site), ARC-077/MECH-337 (social in-face, V4), SD-057/MECH-347 (affective
    cue-recall sibling -- same retrieval-by-context motif), MECH-094 (call-site
    scoping).
