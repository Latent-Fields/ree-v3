## MECH-314a Phase-2 AMEND: e2.world_forward novelty-candidate-source (V3-EXQ-648 autopsy, 2026-06-07)
- MECH-314a Phase-2 amend: policy.structured_curiosity_bonus.e2_world_forward_novelty_source
  -- IMPLEMENTED 2026-06-07. Routed by failure_autopsy_V3-EXQ-648_2026-06-07.json
  (precondition_unmet; routing=implement-substrate; recommended_substrate_queue_entry.action
  =amend on MECH-314a-Phase-2-impl). Design-doc Candidate 1 source on the landed
  Candidate-5A machinery (REE_assembly/docs/architecture/mech_314a_phase2_novelty_source_design.md
  section 3 Candidate 1).
  ROOT CAUSE (V3-EXQ-648 FAIL, run v3_exq_648_..._20260607T025417Z): the MECH-314a
  per-candidate novelty AND the auto-augmentation _candidate_spread key were computed
  in agent.select_action from the hippocampal proposer's first-step z_world
  (trajectory.world_states[:,0,:], cur_summaries), whose cross-candidate spread is
  <0.01 under monostrategy -> curiosity_bias_range=0.0 and curiosity_std_across_K=0.0
  in EVERY arm incl the ARM_1 positive control. The SD-056-trained e2.world_forward(z0,a_i)
  predictions carry spread ~0.1147 (the representation the SD-056 readiness gate already
  validates) but were NOT the consumed signal. The 648 readiness precondition measured
  e2.cand_world_pairwise_dist (0.1147, PASS) while C2 routed on the proposer-derived bias
  (<0.01) -> false READY -> self-route mislabelled a collapsed-input artefact as a wiring
  null (canonical V3-EXQ-642 same-statistic pattern).
  THE FIX (no-op-default; bit-identical OFF):
    Module: ree_core/agent.py (new REEAgent._curiosity_candidate_summaries(candidates)
      helper + curiosity block in select_action consults it first), ree_core/utils/config.py
      (1 new flag + from_dims passthrough). ree_core/policy/structured_curiosity.py UNCHANGED
      -- both _compute_novelty and _candidate_spread already key on the candidate_world_summaries
      argument, so rebuilding that argument fixes both at once.
    Config: REEConfig.curiosity_candidate_source: Literal["proposer","e2_world_forward"]
      = "proposer" (default; wired through REEConfig.from_dims). "proposer" -> helper returns
      None -> legacy proposer-summary reuse-chain runs unchanged (bit-identical).
      "e2_world_forward" -> cur_summaries = e2.world_forward(z0.expand(K,-1), first_actions_K)
      [K, world_dim] (z0 = current observed z_world; first_actions_K = stack(c.actions[:,0,:])),
      the same construction the SD-056 cand_world_pairwise_dist readiness diagnostic uses.
  Data flow (e2_world_forward ON): current latent z_world + per-candidate first action ->
    e2.world_forward -> [K, world_dim] action-divergent predictions -> StructuredCuriosity.
    compute_score_bias(candidate_world_summaries=...) -> BOTH 314a RBF novelty (vs visitation
    buffer / residue centers) AND the auto-augmentation _candidate_spread key now key on the
    action-divergent representation.
  Backward compatible: default "proposer" -> bit-identical (no e2 call; existing reuse-chain).
    877/877 contracts + 7/7 preflight PASS; V3-EXQ-648 --dry-run unchanged (default proposer
    path; dry-scale substrate_not_ready_requeue as before). New C6 contracts (4) in
    tests/contracts/test_mech_314_curiosity.py: proposer returns None (bit-identical);
    e2_world_forward returns [K,world_dim]; with a monkeypatched divergent world_forward the
    curiosity-consumed _candidate_spread exceeds the collapsed proposer spread (the 648
    root-cause made deterministic); proposer-collapsed-baseline sanity.
  Phased training: N/A -- no new learned parameters; reuses the already-trained
    e2.world_forward (SD-056). The validation experiment trains e2 via the SD-056 online
    contrastive warmup in P0 (as V3-EXQ-648 already does). DETECTOR DEPENDS ON A TRAINED
    e2: on an untrained/under-trained e2 the predictions re-collapse -> the V3-EXQ-648a
    readiness precondition (re-targeted to THIS consumed representation) catches it and
    self-routes substrate_not_ready_requeue rather than mislabelling a wiring null.
  MECH-094: preserved. The e2.world_forward read is torch.no_grad() on the waking
    select_action path (no replay / memory write surface); the visitation-buffer write in
    sense() is untouched and stays MECH-094-gated; compute_score_bias's simulation_mode
    gate is unchanged.
  Evidence-staleness: NOT triggered -- no-op-default flag; every existing experiment uses
    the default "proposer" source, so no dependent claim's measured mechanism changed.
  Validation experiment: V3-EXQ-648a (supersedes V3-EXQ-648) -- the corrected
    substrate-readiness diagnostic enabling curiosity_candidate_source="e2_world_forward"
    + re-targeting the readiness precondition to the consumed representation, queued via
    /queue-experiment. PASS gates V3-EXQ-590b + the section-8 MECH-314a/MECH-314/ARC-065
    governance/claims updates (which stay GATED on PASS, not applied here).
  Design doc: REE_assembly/docs/architecture/mech_314a_phase2_novelty_source_design.md
    (Candidate 1 source; V3-EXQ-648a amend section).
  See MECH-314 (parent), MECH-314a (sub-flavour), ARC-065 (parent architectural commitment),
    SD-056 (e2.world_forward action-conditional divergence; the representation now consumed),
    V3-EXQ-648 (the FAIL this amend addresses), V3-EXQ-648a (validation), MECH-094 (call-site
    scoping; preserved).
