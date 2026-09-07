## modulatory-bias-selection-authority: gap-relative E3.select authority (2026-06-03)
- modulatory-bias-selection-authority: ethics_engine_3.modulatory_bias_selection_authority
  -- IMPLEMENTED 2026-06-03 (substrate-readiness validation pending V3-EXQ).
  Gives the modulatory / diversity score-bias channels genuine but BOUNDED authority
  over the committed argmin at E3.select. Root cause (604a/624a/614d cluster autopsy):
  fixed small modulatory magnitudes (~0.05-0.1) added to primary scores whose
  raw_score_range is much larger never change the argmin -- 604a curiosity_bias=0.0
  every arm, 624a vigor action_density byte-identical ON==OFF, 614d within-class
  temperature -> committed-class entropy byte-identical across T=0.5/1.0/2.0.
  Approach (b) gap-relative scaling (user-confirmed AskUserQuestion 2026-06-03):
    Modules: ree_core/predictors/e3_selector.py (additive authority),
      ree_core/predictors/e3_score_diversity.py (stratified across-class normalization),
      ree_core/utils/config.py (flags).
    Site 1 (e3_selector.select): after the composed score_bias chain (dACC +
      lateral_pfc + ofc + mech295 + MECH-314 curiosity + MECH-320 vigor) and the
      MECH-341 entropy bonus are added, compute mod = scores - raw_scores and rescale
      so range(mod) == modulatory_authority_gain * raw_score_range, then
      scores = raw_scores + rescaled_mod. Takes precedence over the legacy
      normalize_score_bias_to_e3_range (skipped when this flag is on).
    Site 2 (e3_score_diversity.stratified_select): normalize class-representative
      scores to UNIT range before the stratified_temperature softmax (614d C2 fix --
      absolute class-rep gap no longer collapses committed-class selection).
  SAFETY: primary scores NOT modified -> commit-threshold / running_variance /
    softmax-temperature / urgency-interrupt / MECH-090 admission semantics unchanged.
    gain=0.5 < 1.0 keeps modulatory competitive in near-ties but subdominant when the
    primary harm/goal gap exceeds gain*range (clearly-harmful candidate stays rejected).
  Config (REEConfig + from_dims + E3Config, all default no-op / bit-identical OFF):
    use_modulatory_selection_authority (bool, False; E3Config + REEConfig top-level
      mirror so build_from_ree_config arms the stratified site),
    modulatory_authority_gain (float, 0.5),
    modulatory_authority_min_range_floor (float, 1e-6).
  Diagnostics: e3_selector.last_score_diagnostics gains modulatory_authority_active +
    modulatory_authority_scale_factor; e3_score_diversity.get_state gains
    mech341_n_authority_normalized + mech341_last_authority_normalized +
    mech341_last_rep_score_range.
  Backward compatible: 734/734 contracts + 7/7 preflight PASS with flag OFF, verified
    under two pytest-randomly orderings; bit-identical baseline. Smoke (flag ON, large
    class gap): stratified across-class restores ~24% share to the worse class vs
    OFF collapse 399/1.
  NECESSARY-BUT-NOT-SUFFICIENT for the curiosity lever: 624a/614d are pure drowning
    (fixed directly); 604a had curiosity_bias=0.0 (genuinely zero -- MECH-314a no active
    residue centers + 314b/c broadcast-by-design), so scaling zero is still zero. The
    validation EXQ must guard curiosity_bias_abs_mean > 0 before testing curiosity.
  MECH-094: pure arithmetic on the waking committed-selection path; stratified_select
    carries simulation_mode; no replay write surface. Phased training: N/A.
  Contracts: tests/contracts/test_e3_score_bias_candidate_support.py
    (test_modulatory_authority_ON_rescales_bias + _min_range_floor_prevents_degenerate_scale,
    seeded for order-robustness).
  Validation experiment: substrate-readiness diagnostic queued via /queue-experiment
    (claim_ids=[]; vigor/curiosity/within-class arms OFF vs ON + curiosity non-degeneracy
    guard + no-harm-increase). PASS unblocks the per-claim evidence retests of
    MECH-314/320/341 and the MECH-343 hypothesis.
  Design doc: REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
  Concurrency note: clearing the substrate gate during the 614d review auto-spawned
    IGW-024 for this substrate; the two sessions converged on the identical design and
    the joint working-tree implementation was landed from the interactive session
    (igw-024 stood down, empty worktree).
  See modulatory-bias-selection-authority (substrate_queue), MECH-314 / MECH-320 /
    MECH-341 (unblocked levers, stay v3_pending), Q-044 / ARC-068 (unblocked),
    MECH-343 (difficulty-gated proposal entropy; downstream), failure_autopsy_604a-624a-630_2026-06-03,
    MECH-090 (admission gate; unchanged), MECH-094 (call-site scoping).
