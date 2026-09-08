## SD-078: policy.common_mode_invariant_candidate_rule_field_context_key -- IMPLEMENTED (2026-07-22)
- SD-078: policy.common_mode_invariant_candidate_rule_field_context_key — IMPLEMENTED 2026-07-22.
  ARC-063 CandidateRuleField, ree_core/policy/candidate_rule_field.py.
  Config: CandidateRuleFieldConfig.cue_centering (default False; set True to enable) +
  cue_baseline_alpha (default 0.02). REEConfig knobs crf_cue_centering /
  crf_cue_baseline_alpha, plumbed through all three from_dims sites and read in
  agent.py's CRF config build.
  Data flow: z_world context -> slow-EMA common-mode baseline (advanced in step(),
  waking only) -> centered residual -> mint-block cosine + _context_bucket sign
  pattern + gate_and_select cosine. context_tags stored RAW, centered at comparison
  time; baseline PERSISTS across reset() (cue geometry, not rule content).
  Backward compatible: disabled by default; OFF allocates no baseline and _centered()
  is the identity.
  Why: under SD-008 the raw context key sits in a ~0.98-cosine cone, so the mint-block
  fires against the first rule for every later context at ANY expressible threshold —
  the pool is structurally capped at ONE rule and crf_max_pairwise_rule_dist == 0.0 is
  a tautology, not a churn symptom. Measured on the V3-EXQ-669b nursery: 1 rule /
  dist 0.0000 raw vs 9 rules / dist 1.7011 centered.
  The two 654b-amend mitigations are measured INEFFECTIVE and deliberately left in
  place unchanged: mature_mint_block_threshold=0.8 cannot clear a 0.9426 floor, and
  crf_context_from_e2_world_forward routes to a context carrying the same offset
  (still 1 rule, dist 0.0000). Do not re-tune either as the fix — contract C3 pins it.
  Phased training required: no (pure stateful tensor store, no nn.Module).
  Contracts: tests/contracts/test_sd_078_centered_candidate_rule_field_context_key.py
  (C0 fixture geometry guard, C1 default-off bit-identity + reproduced 654b signature,
  C2 centering separates, C3 no mint-block threshold in [0.5, 0.94] substitutes,
  C4 lazy seed + MECH-094, C5 raw-tag self-match under baseline drift, C6 bucket
  is centered too).
  See SD-066, SD-077, SD-079, SD-008, SD-070, ARC-063, SD-033a, MECH-262.
