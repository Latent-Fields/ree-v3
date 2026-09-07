## MECH-341 Retune: stratified_select call-site expansion + 6-arm validation (2026-05-28)
- MECH-341 retune -- IMPLEMENTED 2026-05-28. Resolves V3-EXQ-611
  substrate-readiness FAIL (2026-05-27T13:02Z) via two parallel actions:
  (a) module-level call-site expansion in ree_core/predictors/e3_selector.py
  lines 848-870, and (b) 6-arm parameter sweep V3-EXQ-611b. The original
  MECH-341 module (e3_score_diversity.py) and its config knobs are
  unchanged -- the retune respects the implement-substrate skill's "Never
  change defaults of existing params" rule.
  Module change: stratified_select previously fired ONLY in the committed
  branch of E3TrajectorySelector.select() (where it replaced argmin); the
  uncommitted branch used multinomial sampling without any class-axis
  preservation. V3-EXQ-611 ARM_2 (OPT2 stratified_select only) measured
  mech341_n_stratified_fired=0 across all 3 seeds: the committed branch
  was never entered in the validation episodes (running_variance never
  fell below effective_threshold), so OPT2 had no opportunity to fire.
  The 2026-05-28 patch invokes stratified_select in BOTH branches: when
  the substrate is enabled AND the pool admits >=2 first-action classes,
  stratified_select replaces the corresponding fallback (argmin in the
  committed branch / multinomial in the uncommitted branch). When
  score_diversity is None or the sub-flag is False, stratified_select
  returns None and both branches fall through to their legacy selection
  rule -- bit-identical to pre-retune behaviour. MECH-094 preserved by
  simulation_mode=False kwarg at both call sites (the gate at
  E3Selector.select is only invoked from waking REEAgent.select_action).
  Architectural rationale: the substrate_queue's failure-record language
  ("substrate-natural pool diversity gets preserved through softmax-sample-
  across-class-representatives") describes a categorical-preservation
  semantic that applies regardless of commit state. The uncommitted-branch
  multinomial path provides stochasticity over ALL candidates by their
  softmax probabilities -- it can concentrate within a single first-action
  class when multiple candidates from that class score well. Stratified
  selection delivers exactly one representative per class with probability
  proportional to softmax(-best_per_class_score / temperature). This is a
  stronger class-diversity guarantee. Restricting it to the committed
  branch mismatched the substrate's intent.
  Config: NO new flags. Existing flags (use_e3_score_diversity master,
  use_e3_diversity_entropy_bonus / use_e3_diversity_stratified_select sub-
  flavours, e3_diversity_entropy_bias_scale, e3_diversity_stratified_temperature,
  e3_diversity_min_classes_for_stratification) all retain 2026-05-27 defaults.
  Backward compatible: 506/506 contracts + 7/7 preflight PASS post-edit;
  bit-identical OFF guarantee verified.
  Validation: V3-EXQ-611b 6-arm factorial (priority 250, machine_affinity
  DLAPTOP-4.local, estimated 200 min). 3 option groups (OPT1_only,
  OPT2_only, BOTH) x 2 entropy_bias_scale values (1.0, 2.0) per the
  substrate_queue retune-target sweep (0.5 dropped as too small per the
  V3-EXQ-611 gap-magnitude analysis). ARMs:
    ARM_1_OPT1_S1   entropy_bonus ON, stratified OFF, scale=1.0
    ARM_2_OPT1_S2   entropy_bonus ON, stratified OFF, scale=2.0
    ARM_3_OPT2_S1   entropy_bonus OFF, stratified ON, scale=1.0 (scale unused on OPT2)
    ARM_4_OPT2_S2   entropy_bonus OFF, stratified ON, scale=2.0 (scale unused on OPT2)
    ARM_5_BOTH_S1   entropy_bonus ON, stratified ON, scale=1.0
    ARM_6_BOTH_S2   entropy_bonus ON, stratified ON, scale=2.0
  ALL_OFF baseline anchored to V3-EXQ-611 ARM_0_ALL_OFF manifest already on
  origin/master. Acceptance: C1 (primary -- n_stratified_fired > 0 across
  all OPT2/BOTH seeds; direct test of the call-site expansion) AND (C2
  bonus scale-commensurate OR C3 selected-class diversity preserved on
  majority of seeds). Phased training matches V3-EXQ-611 budget (P0=30 ep
  warmup, P1=20 ep measurement) so per-arm comparisons are calibrated.
  experiment_purpose=diagnostic; substrate-readiness retunes do NOT weight
  claim confidence per Phase-3 governance rules (Q-054 behavioural
  falsifier is the governance-weighting signal).
  Dry-run smoke 2026-05-28T17:25Z: 6/6 arms run to completion at reduced
  scale (P0=2 ep, P1=2 ep, 30 steps/ep); C1 confirmed firing
  (n_stratified_fired > 0 in OPT2/BOTH arms); C2/C3/R2c not satisfied at
  dry-run scale (requires full P1 budget). Substrate-side validation:
  passed.
  Plan-of-record: REE_assembly/evidence/planning/substrate_queue.json
  MECH-341 entry status pending_retune -> retune_implemented_pending_validation
  with full implementation_log. behavioral_diversity_isolation_plan.md row 2
  GAP-B updated to reflect retune-implemented state.
  Concurrent-session coordination: pathspec-limited commits avoid sweeping
  IGW-008 (plan-doc row 1), IGW-010 (plan-doc row 3 + workset regen), and
  IGW-011 (plan-doc row 4) edits to behavioral_diversity_isolation_plan.md.
  My touch on row 2 (GAP-B) is disjoint from theirs.
  See MECH-341 (this claim), V3-EXQ-611 (substrate-readiness FAIL the
  retune addresses), V3-EXQ-611b (validation experiment),
  REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md
  retune section, REE_assembly/evidence/planning/substrate_queue.json
  implementation_log block, IGW-20260528-025 (workset entry routing the
  retune via /implement-substrate).
