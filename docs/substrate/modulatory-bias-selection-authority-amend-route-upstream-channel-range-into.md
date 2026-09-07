## modulatory-bias-selection-authority AMEND: route upstream-channel range into the bias the authority rescales (569f/661/654a, 2026-06-10)
- modulatory-bias-selection-authority route-range amend -- IMPLEMENTED 2026-06-10.
  Modules: ree_core/predictors/e3_selector.py (project_channel_range helper +
  channel_route_bias param + accumulator fold + P0 diagnostic), ree_core/agent.py
  (route source selection at the e3.select() site), ree_core/utils/config.py (flags).
  Routed by failure_autopsy_569f-661-654a_2026-06-10 (confirmed; user-adjudicated
  2026-06-10 governance cycle).
  ROOT CAUSE (one structural property, not three bugs): the 2026-06-03/06-06 authority
  rescales _modulatory_accum (the composed score_bias chain + MECH-341 bonus). The
  569f/661/654a cluster showed that a channel whose REPRESENTATION carries genuine
  cross-candidate range (569f consumed world-summary spread 0.196; 654a minted
  rule_state 268/549/220; 661 coherence JOINT 1.0/ALT 0.25/SHUF 0.0) still does NOT
  move committed action -- because that range is flattened by the consuming bias head
  (e.g. the SD-033a/b zeroed-last-layer heads) before it reaches _modulatory_accum, so
  the authority has nothing to amplify (569f selected-action entropy bit-identical
  0.549141 across e2wf / proposer / matched-noise). V3-EXQ-643 established "no range ->
  no authority"; this cluster extends it one link: the channel range must be ROUTED
  into the per-candidate modulatory bias the authority rescales, not merely exist in
  the representation.
  THE FIX (parameter-free; no-op default; bit-identical OFF):
    (1) project_channel_range(features) in e3_selector.py -- a deterministic,
      range-preserving projection of a channel's per-candidate representation into a
      per-candidate scalar bias [K]. For [K, D] (e.g. cand_world_summaries): center
      across the K candidates, project onto the leading right-singular vector of the
      centered matrix -> [K] signed scalar capturing the dominant cross-candidate
      variation (SVD on a detached copy; numerical fallback to the mean-deviation
      axis). For [K] (an already-per-candidate bias): identity. K<2 / flat input ->
      zeroed (below-floor) vector. The singular-vector sign is arbitrary: routing
      makes the channel range REACH and MOVE the committed argmax (the readiness
      property), NOT necessarily move it BENEFICIALLY -- that is the channel's own
      trained head (the separate per-claim evidence retest).
    (2) E3TrajectorySelector.select() gains channel_route_bias: Optional[Tensor] [K].
      When use_modulatory_channel_routing is on and it is supplied, it is normalised
      to unit zero-mean range (so the contribution stays bounded even when the
      authority is OFF; the authority re-normalises the combined accumulator to
      gain*raw_score_range regardless), scaled by modulatory_channel_route_weight, and
      folded into BOTH scores and _modulatory_accum BEFORE the authority's range
      computation -- so the channel's range reaches the bias term the authority
      rescales. New P0 diagnostics on last_score_diagnostics:
      modulatory_channel_route_range (the RAW cross-candidate range of the routed bias,
      pre-normalise/pre-rescale -- the gate signal) + modulatory_channel_route_active.
    (3) REEAgent.select_action builds channel_route_bias from the channel-under-test's
      per-candidate representation (project_channel_range) and passes it. Source
      selectable via modulatory_channel_route_source: "cand_world_summary" (the
      [K, world_dim] world-summary channel -- 569f cluster lead, the genuine
      projection case, sourced from cand_world_summaries / the ARC-065 GAP-A helper) /
      "curiosity" / "gated_policy" / "mech295" / "coherence" / "lateral_pfc" (each an
      already-computed per-candidate [K] bias, identity-routed; the MECH-294 compose
      bias stashed as _bdc_coherence; "lateral_pfc" (2026-07-31, ARC-062/MECH-309 P-A
      erratum fix) is the SD-033a LateralPFCAnalog rule-apprehension bias stashed as
      _bdc_lpfc -- the channel use_candidate_rule_field actually differentiates.
      NOT "gated_policy": that routes the unrelated ARC-062 Phase-1 GatedPolicy
      module's bias (_bdc_gp), which has no connection to LateralPFCAnalog/
      CandidateRuleField -- see arc_062_conversion_fanout_2026-07-29.md erratum).
      Default "none" -> channel_route_bias=None -> bit-identical.
  P0 readiness gate: modulatory_channel_route_range lets a retest assert the
  modulatory bias ITSELF carries cross-candidate range derived from the channel under
  test BEFORE any behavioural falsifier is scored (so an unrouted channel cannot
  self-route a false negative -- the autopsy's explicit requirement).
  Config (E3Config + REEConfig mirror + from_dims; all no-op default, bit-identical OFF):
    use_modulatory_channel_routing (False, master; E3Config + REEConfig mirror),
    modulatory_channel_route_min_range_floor (1e-6; E3Config, the P0 gate floor),
    modulatory_channel_route_weight (1.0; E3Config, routed-vs-legacy proportion in the
    accumulator), modulatory_channel_route_source (str "none"; REEConfig, agent source
    select). All wired through REEConfig.from_dims.
  HONEST SCOPE: routing makes the channel range reach + move the committed argmax (the
  P0 property). For COHERENCE specifically (661): currency_coherence is a SCALAR
  (uniform across candidates) -- its per-candidate range lives only in the compose
  cosine; routing the compose [K] bias gives it range, but joint-vs-alternation
  MODE-DISCRIMINATION (different per-candidate PATTERN, not just magnitude) is a
  MECH-294-side concern, out of scope here. A scalar gate is rescale-invisible by
  construction; the P0 gate correctly flags a channel that carries no cross-candidate
  range as substrate_not_ready.
  Backward compatible: use_modulatory_channel_routing=False by default -> the routing
  block is skipped, channel_route_bias is None, _modulatory_accum + authority unchanged
  -> bit-identical. 989 contracts (985 prior + 4 new in
  tests/contracts/test_e3_score_bias_candidate_support.py: project_channel_range
  range-preserve/identity/degenerate; routing-OFF bit-identical; routing-ON P0 range +
  scores-reach-rescaled-accumulator; below-floor inactive) + 7/7 preflight PASS.
  Activation smoke 2026-06-10 (StepHarness, world-summary source, authority ON):
  default == explicit-OFF bit-identical; routing-ON modulatory_channel_route_range
  0.04-0.11 (> floor), modulatory_channel_route_active True, committed argmax moves vs
  OFF (the 569f bit-identical-entropy washout broken).
  Phased training: N/A (pure arithmetic on the waking committed-selection path; no
  learned parameters). MECH-094: N/A (no replay/memory write surface; the SVD read is
  on a detached copy). Evidence-staleness (Step 8.5): NOT triggered -- no-op-default
  flag; every existing experiment uses the default (routing off), so no dependent
  claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: this amend resolves NO dependency on its own (validation pending). The
  unblocked claims (ARC-065 / MECH-294 / ARC-062 / MECH-309 / MECH-341 + the existing
  MECH-314/320/Q-044/etc.) stay candidate / v3_pending / pending_retest_after_substrate;
  claims.yaml NOT modified.
  Validation experiment: V3-EXQ-663 substrate-readiness diagnostic (claim_ids=[];
  ARM_0 routing OFF vs ARM_1 routing ON, both authority ON + e2_world_forward + SD-056
  online). Acceptance: READINESS (load-bearing, RANGE) ARM_1 route_range > floor on
  >=2/3 seeds; C1 (load-bearing, same range stat) ARM_1 active + ARM_0 inactive;
  C2 (secondary, behavioural reach) committed-class TV ARM_1-vs-ARM_0 > floor.
  PASS = READINESS AND C1. Below-floor -> substrate_not_ready_requeue. Dry-run runs
  end-to-end (self-routes substrate_not_ready at toy P0, as designed). PASS unblocks
  the SEPARATE per-claim behavioural retests (NOT queued here).
  Design doc: REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
  (route-range amend section). Autopsy: REE_assembly/evidence/planning/failure_autopsy_569f-661-654a_2026-06-10.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (modulatory-bias-selection-authority entry; +3 failure records 569f/661/654a applied
  by the 2026-06-10 governance cycle).
  See modulatory-bias-selection-authority (parent substrate; gap-relative scaling
  2026-06-03 + 643a float32 fix 2026-06-06), V3-EXQ-643 (predecessor: no range -> no
  authority), V3-EXQ-569f/661/654a (the cluster this amend addresses), ARC-065 GAP-A
  (the cand_world_summary e2_world_forward source the world-summary route reads),
  MECH-341 (entropy bonus -- a channel that already carries range in the accumulator),
  ARC-062/MECH-309 GAP-D (trainable rule_bias_head -- the beneficial-selection half for
  the rule_state behavioural retest), MECH-294 compose-coherence (the scalar-gate
  channel the P0 gate correctly flags), V3-EXQ-663 (validation), MECH-090 (admission
  gate; unchanged), MECH-094 (N/A).
