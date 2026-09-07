## ARC-062 Phase 1 gated-policy (GAP-A, 2026-05-09)
- ARC-062: policy.gated_policy -- IMPLEMENTED 2026-05-09 (arc_062_rule_apprehension
  plan GAP-A). Module: ree_core/policy/gated_policy.py (GatedPolicy +
  GatedPolicyConfig). Two scoring heads (symmetry-broken init) plus 3-stream
  context discriminator (z_world, z_self, z_harm_a) -> sigmoid w in [0, 1];
  gated_score_bias = w*head_0 + (1-w)*head_1 (plus mode_separation_floor when
  enabled). Config: REEConfig.use_gated_policy (default False, bit-identical OFF);
  gated_policy_disc_hidden=24, gated_policy_n_heads=2, gated_policy_disc_init_scale=0.1.
  Data flow: candidate_features [K, world_dim] -> heads -> composed bias ->
  REEAgent.select_action() score_bias (parallel to lateral_pfc / dacc / ofc).
  Phase 1 does NOT wire SD-033a (that is GAP-C/D, commitment_closure GAP-1).
  Contracts: tests/contracts/test_gated_policy.py C1-C5 (7/7 PASS).
  MECH-094: simulation_mode returns neutral (0.5, zeros); no internal state buffer.
  Phased training: required for Phase 2/3 trained-policy falsifiers (P0/P1/P2).
  Validation experiment: V3-EXQ-542a substrate-readiness PASS (UC1-UC6 incl one-hot
  UC6; supersedes V3-EXQ-542). See MECH-309, SD-054, rule_apprehension_layer.md,
  evidence/planning/arc_062_rule_apprehension_plan.md.
