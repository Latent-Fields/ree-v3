## ARC-063 AMEND: cross-episode rule-persistence flag (V3-EXQ-654 GAP-B maturity) (2026-06-09)
- ARC-063 amend: policy.rule_apprehension_layer.candidate_rule_field cross-episode
  rule persistence -- IMPLEMENTED 2026-06-09. Module:
  ree_core/policy/candidate_rule_field.py (CandidateRuleFieldConfig +
  CandidateRuleField.reset()). Routed by the confirmed
  failure_autopsy_V3-EXQ-654_2026-06-09 (applied in the 2026-06-09 governance
  cycle; substrate_queue ARC-062 amend, impl target ARC-063 CandidateRuleField).
  ROOT CAUSE (code-grounded, NOT a falsification): the V3-EXQ-654 arc_062 GAP-B
  behavioural falsifier (MECH-309/ARC-062) FAILed its C1c readiness precondition
  (arm_on_rule_field_differentiated), so the C2 falsifier DV never ran. agent.reset()
  -> candidate_rule_field.reset() (agent.py:1908), called every ~26-tick episode,
  wiped self._rules + self._recurrence, so the live pool cold-started each episode
  and never matured a differentiated rule pool (crf_frac_active 0.116/0.123/0.115
  < 0.30 floor; crf_max_pairwise_rule_dist 0.0 all ARM_ON seeds; seed-42 ARM_ON
  committed-class byte-identical to ARM_OFF). The recurrence-threshold mint (>=3
  within ONE episode) + per-episode wipe starves the pool at behavioural-runtime
  episode lengths despite cumulative crf_n_minted 131-408. Biology accumulates
  task-set rule structure across experiences and is NOT reset per trial (Collins &
  Frank 2014; Mansouri rule-selective persistence) -- the per-episode wipe is a
  translation failure, a discovered prerequisite. V3-EXQ-639 PASS proves the field
  DOES differentiate when its pool matures.
  THE FIX (no-op-default; bit-identical OFF):
    Config: CandidateRuleFieldConfig.persist_rules_across_episode_reset (bool,
      default False). REEConfig.crf_persist_rules_across_episode_reset (default
      False) + REEConfig.from_dims passthrough; agent.py forwards it into the
      CandidateRuleFieldConfig at the field build site (getattr fallback, so an
      absent flat attr is bit-identical).
    Semantics: CandidateRuleField.reset() returns early (a no-op) when the flag
      is set -- the live rule pool (_rules), recurrence counters (_recurrence),
      AND the clock (_step, kept monotonic so minted_step/last_active_step stay
      ordered) PERSIST across the per-episode agent.reset(). The agent.reset()
      call itself is UNCHANGED; only the field's reset() semantics change. With
      the flag set, the field accumulates a live differentiated pool across the
      P1 measurement episodes so crf_frac_active can clear the 0.30 C1c floor at
      ~26-tick episode lengths.
  Default OFF = bit-identical per-episode wipe -- _rules/_recurrence/_step cleared
  exactly as before. The credit/eligibility/decay math uses no step deltas, so a
  continuous clock is safe (last_active_step/minted_step are bookkeeping-only).
  Backward compatible: persist_rules_across_episode_reset=False by default; every
  existing experiment cold-starts each episode unchanged. 11/11 CRF contracts
  (8 prior + 3 new: C9 default-OFF bit-identical wipe, C10 pool survives reset +
  keeps maturing, C11 from_dims + agent config wiring) + 7/7 preflight PASS; full
  contract suite 959 passed (the single control_vector C4 failure is a pre-existing
  baseline flake -- fails 4/5 on a clean tree, unrelated, documented above).
  Phased training: N/A (pure-arithmetic regulator; no learned parameters; the flag
  changes only episode-boundary pool lifecycle). MECH-094: unchanged -- the
  field's per-tick state-advancing methods keep their simulation_mode no-op gate;
  the persistence flag affects only the episode-boundary reset, not replay writes.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every
  existing experiment uses the default (per-episode wipe), so no dependent claim's
  measured mechanism changed. KEEP all evidence.
  GOVERNANCE: MECH-309/ARC-062 NEITHER validated NOR weakened; they stay candidate /
  substrate_ceiling / v3_pending / pending_retest_after_substrate. claims.yaml /
  governance state UNTOUCHED (substrate-only amend).
  Validation experiment: V3-EXQ-654a is the SECONDARY follow-up, queued SEPARATELY
  via /queue-experiment and GATED on this amend -- the same single-variable ARM_OFF
  (use_candidate_rule_field=False) vs ARM_ON (differentiated crf_source) GAP-B
  falsifier with crf_persist_rules_across_episode_reset=True + a TRAINED-bias-head
  P1 arm (GAP-D rule_bias_head trainability, landed 2026-05-17) + a propagation
  non-vacuity precondition (ARM_ON lateral_pfc bias must differ from ARM_OFF on
  firing ticks); committed-class entropy stays the PRIMARY DV. NOT queued in this
  implement-substrate session (user-scoped to the amend only).
  Design doc: REE_assembly/docs/architecture/arc_063_candidate_rule_field.md
  (cross-episode-persistence amend section, 2026-06-09).
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-654_2026-06-09.json.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json ARC-062 entry.
  See ARC-063 (parent v1 entry above), MECH-309 / ARC-062 (the GAP-B claims this
  matures the falsifier for; unweakened), SD-033a (rule_state consumer), GAP-D
  rule_bias_head trainability (the 654a propagation arm), V3-EXQ-654 (the FAIL this
  amend addresses), V3-EXQ-639 (PASS proving the field differentiates when matured),
  V3-EXQ-654a (validation; separate session), MECH-094 (call-site scoping; unchanged).
