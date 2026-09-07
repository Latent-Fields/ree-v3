## MECH-341 (ARC-065 Layer-B child): E3 Score Diversity Preservation (2026-05-27)
- MECH-341: ethics_engine_3.scoring_trajectory_class_diversity_preservation
  -- IMPLEMENTED 2026-05-27.
  Module: ree_core/predictors/e3_score_diversity.py (E3ScoreDiversity +
  E3ScoreDiversityConfig + E3ScoreDiversityDiagnostics + build_from_ree_config).
  Layer-B (post-CEM scoring) diversity-preservation substrate. Triggered by
  V3-EXQ-608 P2 (2026-05-26T02:58Z) majority R2a_e3_collapse_confirmed_large_gap:
  with SP-CEM main-path delivering frac_pre_ge2=1.0 (>= 2 first-action classes
  in the candidate pool every measured tick), E3 scoring collapsed to a single
  class with mean_top2_class_gap 0.27-0.60 (LARGE-gap; rules out option 3
  jittered tie-breaking; routes to options 1 + 2 per
  behavioral_diversity_isolation_plan.md "Substrate design options" section).
  Two togglable sub-flavours under one master, mirroring MECH-314a/b/c precedent
  so Q-054 falsifier can dissociate which one carries the load.
    Option 1 (entropy_bonus): per-candidate POSITIVE bias proportional to the
      first-action class's frequency in the pool (REE lower-is-better convention,
      so positive bias penalises over-represented classes). Composed into scores
      AFTER the dACC / lateral_pfc / ofc / mech295 / curiosity / tonic_vigor
      score_bias chain and BEFORE last_scores / softmax. Mnih 2016 A3C-style
      entropy regularisation adapted to local candidate-pool first-action
      categorical axis (not global policy entropy).
    Option 2 (stratified_select): partition candidates by first-action class,
      pick argmin within each class as the class representative, softmax-sample
      across class-representatives with temperature stratified_temperature.
      Replaces argmin in the committed-path selection at e3_selector.py:811-820.
      Falls through to legacy argmin when fewer than
      min_classes_for_stratification unique classes are present.
  Pure-arithmetic regulator (no nn.Module inheritance, no learned parameters).
  Sibling pattern to MECH-313 NoiseFloor, MECH-314 StructuredCuriosity, MECH-320
  TonicVigor.
  Config: REEConfig.use_e3_score_diversity (bool, default False; bit-identical
  OFF master). Sub-knobs (all consulted only when master ON; all wired through
  REEConfig.from_dims): use_e3_diversity_entropy_bonus (True),
  use_e3_diversity_stratified_select (True), e3_diversity_entropy_lambda (0.05;
  Q-054 calibrates), e3_diversity_entropy_bias_scale (0.1; mirrors
  lateral_pfc / curiosity / tonic_vigor bias_scale), e3_diversity_stratified_temperature
  (1.0), e3_diversity_min_classes_for_stratification (2).
  Agent wiring (REEAgent.__init__): instantiate via
  build_e3_score_diversity_from_ree_config(config) when master ON; None
  otherwise. reset() clears diagnostic counters. select_action passes
  score_diversity=self.score_diversity kwarg to e3.select(...).
  E3TrajectorySelector.select() gains a score_diversity: Optional[Any] = None
  kwarg; Option 1 composed at the same site as score_bias addition (before
  last_scores); Option 2 consulted in the committed branch before falling
  through to argmin.
  Data flow: scores (per-candidate harm + benefit + goal + residue) -> +
  dacc_score_bias chain -> + MECH-341 Option 1 entropy bonus -> last_scores
  diagnostics -> softmax(-scores / effective_temperature) [MECH-313 lifts
  temperature] -> if committed: MECH-341 Option 2 stratified pick OR legacy
  argmin / else: multinomial(probs).
  Backward compatible: use_e3_score_diversity=False by default;
  agent.score_diversity is None and both call sites are skipped (bit-identical
  to baseline). 506/506 contracts + 7/7 preflight PASS with flag OFF
  (regression-clean 2026-05-27). Single-class candidate pools produce zero
  bonus and fall through to argmin (no-op even when master ON).
  Lit-pull verdicts (defaults per behavioral_diversity_isolation_plan.md):
    Rigotti et al. 2013 -- mixed selectivity in PFC encodes diverse trajectory
      contingencies; preservation across scoring layers required for downstream
      behavioural flexibility.
    Padoa-Schioppa & Conen 2017 -- OFC value comparison preserves option-distinct
      value signals through the comparison stage; collapse to single rank is
      pathological.
  Both options are valid biological renderings; Option 1 is the soft-bias /
  entropy-pressure reading, Option 2 is the OFC categorical-preservation reading.
  Togglable-both architecture lets the Q-054 falsifier dissociate empirically.
  MECH-094: both methods accept simulation_mode argument; when True,
  apply_entropy_bonus returns zeros[K] and stratified_select returns None
  (caller falls through to legacy argmin). Inline gates are defensive (the
  wired call site E3Selector.select is currently invoked only from waking
  REEAgent.select_action paths). Diagnostic counter mech341_n_simulation_skipped
  tracks both paths.
  Phased training: not applicable (pure-arithmetic regulator; no learned
  parameters; no gradient flow). Q-054 calibration of entropy_lambda is a
  parametric sweep, not phased latent-target training.
  Validation experiment: V3-EXQ-611 queued (4-arm substrate-readiness
  diagnostic: ALL_OFF / OPT1_ONLY / OPT2_ONLY / BOTH_ON on the EXQ-608 env
  + metric stack; acceptance criteria reuse EXQ-608's mean_top2_class_gap +
  frac_pre_ge2 + selected_classes_count metrics; PASS = either single-option
  arm produces selected_action_classes_count >= 2 with frac_pre_ge2 >= 0.5).
  V3-EXQ-611 FAILed substrate-readiness 2026-05-27T13:02Z (manifest
  v3_exq_611_mech341_substrate_readiness_4arm_20260527T130213Z_v3.json):
  ARM_1/3 entropy_bonus_max_abs ~0.023-0.044 dwarfed by mean_top2_class_gap
  0.27-1.96; ARM_2 n_stratified_fired=0 across all 3 seeds because the
  committed branch was never entered during measurement and the prior
  implementation gated stratified_select to the committed path only.
  Behavioural validation (Phase P3 B_only / ablate_B / ALL_ON arms per the
  isolation plan + R2.c rule) deferred to a successor queued after the
  retune validation PASS.
  Retune (2026-05-28): two-action substrate update under user-confirmed scope
  via AskUserQuestion (TASK_CLAIMS session
  implement-substrate-mech-341-retune-20260528T165000Z). (a) Module change:
  ree_core/predictors/e3_selector.py applies stratified_select on BOTH
  committed and uncommitted branches (was committed-only); fixes the
  zero-fires issue surfaced by V3-EXQ-611 ARM_2. Bit-identical when
  score_diversity is None or sub-flag is False -- stratified_select returns
  None and caller falls through to legacy argmin (committed) or multinomial
  (uncommitted). MECH-094 preserved via the existing simulation_mode kwarg.
  506/506 contracts PASS post-edit. (b) Parameter sweep: V3-EXQ-611b queued
  (6-arm factorial: 3 option groups x 2 entropy_bias_scale values 1.0/2.0)
  to test scale-commensurability of the entropy bonus against the observed
  score-gap range. NO config default changes per implement-substrate skill
  rule -- scales passed via cfg_overrides per arm. Acceptance criteria:
  C1 stratified fires across all OPT2/BOTH arm seeds (direct test of the
  call-site expansion); C2 entropy_bonus_max_abs >= 0.7 * scale on majority
  of seeds in entropy-ON arms; C3 selected_classes >= 2 with frac_pre_ge2
  >= 0.5 on majority of seeds in at least one arm; R2.c readiness threshold
  cleared by at least one arm. Sentinel routing: PASS -> behavioural
  successor; FAIL with C1=false -> /diagnose-errors on e3_selector wiring;
  FAIL with C1=true and C2/C3=false -> substrate revisit (algorithm-level
  Option-2 redesign or floor adjustment).
  Design doc: REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md
  Plan-of-record: REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
  See MECH-341 (this claim), ARC-065 (parent diversity-generation pathway --
  Layer A), MECH-313 / MECH-314 / MECH-318 / MECH-319 (sibling ARC-065 / ARC-064
  child substrates at proposal / action-selection layers), Q-054 (minimum
  trajectory-class diversity floor for ARC-062 -- entropy_lambda calibration),
  INV-076 (diversity as structural prerequisite for ethical counterfactual
  evaluation -- companion universal invariant), ARC-062 (rule apprehension
  via gated policy; GAP-B downstream beneficiary), SD-003 (superseded;
  counterfactual pipeline retained via MECH-256 / SD-029 / MECH-257),
  MECH-094 (hypothesis_tag invariant; call-site-scoped via simulation_mode
  argument).
