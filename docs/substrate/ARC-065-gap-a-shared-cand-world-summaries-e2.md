## ARC-065 GAP-A: shared cand_world_summaries e2.world_forward source (V3-EXQ-614e autopsy, 2026-06-07)
- ARC-065 GAP-A: policy.candidate_pool_per_candidate_signal_preservation.shared_channel
  -- IMPLEMENTED 2026-06-07. Routed by failure_autopsy_V3-EXQ-614e_2026-06-07
  (confirmed; non_contributory / substrate_ceiling / pending_retest_after_substrate
  gated on ARC-065 GAP-A). The shared-channel sibling of the MECH-314a Phase-2
  curiosity amend (landed earlier the same day): that pass fixed ONLY the curiosity
  channel's consumed representation; this pass extends the identical e2.world_forward
  re-sourcing to the SHARED per-candidate cand_world_summaries consumed by ALL the
  other E3-side bias channels (lateral_pfc / ofc / mech295 / gated_policy / tonic_vigor).
  ROOT CAUSE (V3-EXQ-614e): with the modulatory-bias-selection-authority gate now
  PROVEN operative (V3-EXQ-643a PASS), committed-class diversity still showed no lift
  (committed_class_entropy byte-identical across within-class T=0.5/1.0/2.0). The
  autopsy relocated the bottleneck from the authority gate (GAP-B, resolved) to the
  UPSTREAM candidate pool (GAP-A): all K CEM candidates produce identical z_world after
  one E2 world-forward step (cand_world_pairwise_dist=0.0000) despite differing first
  actions, so every E3-side bias channel sees a class-uniform pool. The shared
  cand_world_summaries was built from the collapsed proposer first-step z_world
  (trajectory.world_states[:,0,:]) at five fresh-build sites in select_action; the
  fix re-sources it from the SD-056-trained action-conditional e2.world_forward(z0, a_i)
  predictions (cross-candidate spread ~0.11 when SD-056 is trained), the representation
  the SD-056 readiness gate already validates and the GAP-B first-action-onehot fix
  bypassed only for GatedPolicy.
  THE FIX (no-op-default; bit-identical OFF):
    Module: ree_core/agent.py (new REEAgent._candidate_world_summaries(candidates)
      helper -- shared-channel sibling of _curiosity_candidate_summaries; consulted
      FIRST at all five cand_world_summaries fresh-build sites: gated_policy block,
      lateral_pfc fallback, ofc fallback, mech295 fallback, and via the reuse-chain
      the tonic_vigor anchor). ree_core/utils/config.py (1 new flag + from_dims
      passthrough).
    Config: REEConfig.candidate_summary_source: Literal["proposer","e2_world_forward"]
      = "proposer" (default; wired through REEConfig.from_dims). "proposer" -> helper
      returns None -> legacy collapsed proposer-summary build runs unchanged
      (bit-identical). "e2_world_forward" -> cand_world_summaries =
      e2.world_forward(z0.expand(K,-1), first_actions_K) [K, world_dim], the same
      construction _curiosity_candidate_summaries (648a) + the SD-056
      cand_world_pairwise_dist readiness diagnostic use. Kept SEPARATE from
      curiosity_candidate_source (the in-flight 648a validation) so the two compose
      without perturbing each other; a future pass may unify.
  Data flow (e2_world_forward ON): current z_world + per-candidate first action ->
    e2.world_forward (no_grad) -> [K, world_dim] action-divergent shared summaries ->
    lateral_pfc / ofc / mech295 / gated_policy / tonic_vigor bias channels ->
    dacc_score_bias -> e3.select(score_bias=...) -> modulatory-bias-selection-authority
    (643a) gap-relative authority over the committed argmin -> MECH-341 stratified
    across-class selection -> committed-class diversity expressible.
  Backward compatible: default "proposer" -> bit-identical (no e2 call; existing
    five build paths unchanged). 889/889 contracts (883 prior + 6 new) + 7/7 preflight
    PASS; V3-EXQ-614e --dry-run unchanged (default proposer path; same dry-scale
    outcome). New contracts tests/contracts/test_arc065_gapa_candidate_summary_source.py
    (G1 default-proposer-helper-None bit-identical + default==explicit / G2
    e2_world_forward [K,world_dim] shape / G3 divergent-world_forward shared spread >
    collapsed proposer spread -- the 614e cand_world_pairwise_dist=0.0000 fix made
    deterministic / G4 master-off helper None / G5 select_action end-to-end with source
    ON + lateral_pfc/ofc/mech295 ON).
  Phased training: N/A new params; reuses the already-trained e2.world_forward (SD-056).
    DETECTOR DEPENDS ON A TRAINED e2: on an untrained/under-trained e2 the predictions
    re-collapse -> the validation EXQ trains e2 (SD-056 online) in P0 and a
    cand_world_pairwise_dist readiness precondition guards vacuity (substrate_not_ready_requeue),
    same pattern as 648a.
  MECH-094: the e2.world_forward read is torch.no_grad() on the waking select_action
    path (no replay / memory write surface); not implicated.
  Evidence-staleness: NOT triggered -- no-op-default flag; every existing experiment
    uses the default "proposer" source, so no dependent claim's measured mechanism
    changed. KEEP all evidence.
  Validation experiment: V3-EXQ-649 (queued via /queue-experiment) -- substrate-readiness
    diagnostic (claim_ids=[]) enabling candidate_summary_source="e2_world_forward" + a
    cand_world_pairwise_dist readiness precondition + a shared-bias-channel per-candidate
    range readout (proposer ~0 vs e2_world_forward > floor). PASS unblocks the MECH-341
    committed-class diversity re-test (the within-class-REPRESENTATIVE-diversity readout,
    NOT committed-class entropy -- autopsy Learning #2), which is the governance-weighting
    successor queued separately. MECH-341 stays candidate / v3_pending; not weakened.
  Design doc: REE_assembly/docs/architecture/mech_314a_phase2_novelty_source_design.md
    (Candidate 1 source; GAP-A shared-channel extension section).
  Plan-of-record: REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
    (GAP-A node) + substrate_queue.json (ARC-065 entry, 614e failure_record).
  See ARC-065 (parent architectural commitment), behavioral_diversity_isolation:GAP-A
    (closure node), MECH-314a Phase-2 (curiosity-channel sibling; curiosity_candidate_source),
    SD-056 (e2.world_forward action-conditional divergence; the representation now consumed
    by the shared channel), ARC-062 GAP-B (the GatedPolicy first-action-onehot fix this
    generalises beyond GatedPolicy), modulatory-bias-selection-authority / V3-EXQ-643a
    (the authority gate that lets the now-divergent bias reach the committed argmin),
    MECH-341 (committed-class diversity re-test, pending_retest_after_substrate),
    V3-EXQ-614e (the FAIL this addresses), V3-EXQ-649 (validation), MECH-094 (call-site
    scoping; preserved).
