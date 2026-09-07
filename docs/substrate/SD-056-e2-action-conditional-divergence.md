## SD-056: E2 action-conditional divergence preservation (contrastive next-state) (2026-05-29)
- SD-056: e2.action_conditional_divergence_contrastive -- IMPLEMENTED 2026-05-29.
  Module: ree_core/predictors/e2_fast.py (E2FastPredictor.cand_world_pairwise_dist
  + E2FastPredictor.world_forward_contrastive_loss). ree_core/utils/config.py
  (E2Config + REEConfig.from_dims 4 knobs). Substrate-level fix for the
  V3-EXQ-571 root cause finding (2026-05-25): under reconstruction-shaped
  training, E2.world_forward fitted the action contribution to zero
  (cand_world_pairwise_dist = 0.0000 across K=8 candidates differing only in
  first action), collapsing per-candidate signal to every downstream consumer
  of cand_world_summaries. Same root cause as 2026-05-17 ARC-062 GAP-B autopsy;
  the GAP-B fix (gated_policy_use_first_action_onehot) was scoped only to
  GatedPolicy. SD-056 is the architecturally-faithful generalisation: fix the
  predictor's training objective rather than bypass it at each consumer.
  Algorithm: auxiliary InfoNCE-style contrastive loss on world_forward.
    For each anchor i in [K]:
      positive:  (z_world_0, a_i) -> predicted z_world_1[i]
      negatives: (z_world_0, a_j) for j != i (sibling CEM candidates)
      L_contrast_i = -log(exp(-||pred_i - target_i||^2/tau)
                        / sum_j exp(-||pred_j - target_i||^2/tau))
    L_E2 = L_recon + w_contrast * mean_i(L_contrast_i)
  Equivalent to cross-entropy over logits[i,j] = -||pred_j - target_i||^2/tau
  with label i. Asymmetric (anchor-to-prediction); symmetric would double
  cost without architectural gain. Negatives drawn from in-batch sibling CEM
  candidates (z_world_0 shared, first action differs) -- informative by
  construction, no negative-mining sweep needed.
  Scope: applies to world_forward only, not predict_next_self. z_self is not
  the collapse site (V3-EXQ-571 measured cand_world_pairwise_dist on z_world).
  world_forward signature and body unchanged; world_transition and
  world_action_encoder shapes and inits unchanged; predict_next_state,
  action_object, forward, forward_counterfactual unchanged; all downstream
  consumers (E1, E3, hippocampal, residue field) unchanged; existing
  rollout-loss machinery unchanged -- contrastive term is additive.
  Two new helpers on E2FastPredictor:
    cand_world_pairwise_dist(z_world_0, candidate_actions) -> scalar.
      Mean pairwise L2 across K predicted z_world_1 outputs. Headline
      substrate-readiness diagnostic per design memo; named by the
      2026-05-28 lit-pull SYNTHESIS verdict 3 as a methodological gap
      worth publishing as a standalone novel measurement. Accepts
      z_world_0 in [world_dim] / [1, world_dim] / [K, world_dim] forms;
      returns tensor(0.0) on K < 2.
    world_forward_contrastive_loss(z_world_0, actions, z_world_1_targets,
                                   weight=None, temperature=None,
                                   min_batch_classes=None,
                                   simulation_mode=False) -> scalar.
      Returns unweighted CE; caller multiplies by
      config.e2.e2_action_contrastive_weight before adding to L_E2 (matches
      the SD-019 / MECH-258 / MECH-273 auxiliary-helper pattern). Returns
      tensor(0.0) when (a) simulation_mode=True, (b) K < 2, or (c) fewer
      than min_batch_classes distinct first-action classes.
  Config: E2Config gains 4 fields (defaults all no-op):
    e2_action_contrastive_enabled (bool, False) -- master switch.
    e2_action_contrastive_weight (float, 0.01) -- w_contrast in L_E2.
    e2_action_contrastive_temperature (float, 0.1) -- InfoNCE tau.
    e2_action_contrastive_min_batch_classes (int, 2) -- first-action-class
      floor below which the loss returns 0 (no informative negatives).
  REEConfig.from_dims mirrors all four with the same defaults and assigns
  onto config.e2.* after the dataclass build (matches the MECH-313 /
  MECH-314 / MECH-320 wiring pattern, NOT the SD-049 env-only pattern --
  this is a regulator-style training-objective change on a learned
  substrate).
  Backward compatible: e2_action_contrastive_enabled=False by default;
  both helpers exist but are not invoked by any existing call site;
  world_forward signature and body unchanged. 539/539 contracts + 7/7
  preflight PASS with master OFF (regression-clean 2026-05-29). With
  defaults, no existing experiment's behaviour or output changes.
  Biological basis: cerebellar internal model (Tanaka et al. 2020),
  prefrontal counterfactual rollout (Miyamoto / Rushworth / Shea 2023),
  vestibular cerebellum corollary discharge (Cullen 2023) all preserve
  action-specificity at the prediction step via dedicated structural
  mechanisms. The contrastive loss enforces this same property -- actions
  must be discriminable in the predicted z_world. ML/AI engineering
  anchors: Srivastava et al. 2021 contrastive RSSM (lever B technique);
  Saanum / Dayan / Schulz 2024 PLSM (failure-mode diagnosis "lack of
  systematic representation of action effects"); Qiu et al. 2026 SWIRL
  (lever C fallback); InfoNCE temperature 0.1 standard literature value.
  Phased training: NOT required at the substrate level. Unlike encoder-
  head-on-frozen-latent patterns (EXQ-166b/c/d historical), both L_recon
  and L_contrast target the same predictor weights (world_transition +
  world_action_encoder) with compatible objectives. Joint training is the
  designed-for case.
  MECH-094: world_forward_contrastive_loss accepts simulation_mode kwarg
  and returns tensor(0.0) when True. Same defensive pattern as SD-035,
  MECH-279, MECH-313, MECH-314, MECH-319, MECH-320, MECH-341. Called only
  from waking E2 training paths; simulation_mode arg is forward-compat for
  any future replay-driven E2 training.
  Activation smoke (2026-05-29): contract test C7 confirms direction-of-
  change after 200 SGD steps on synthetic K=8 sibling batches:
  cand_world_pairwise_dist rises from random-init baseline by >= 2x or
  clears the 0.01 minimum-observable threshold. UC3 magnitude calibration
  (suggested threshold >= 0.05 in normalised units) lives in V3-EXQ-NEW-1.
  Validation experiment: V3-EXQ-NEW-1 substrate-readiness diagnostic
  (UC1-UC5 covering module surface, master-OFF backward-compat,
  cand_world_pairwise_dist direction-of-change, contrastive-task accuracy
  > 50% on held-out batch -- random baseline 1/K = 12.5% for K=8 --
  MECH-094 simulation gate). Diagnostic claim_ids=[] (substrate-readiness,
  NOT governance evidence yet). Behavioural validation V3-EXQ-569a
  (matched-entropy FP-2 falsifier on the fixed substrate; GAP-A R1.a/R1.b
  decision rule per behavioral_diversity_isolation_plan.md) queued
  separately per plan-of-record sequencing.
  Design doc: REE_assembly/docs/architecture/sd_056_e2_action_conditional_divergence.md
  Plan-of-record memo: REE_assembly/evidence/planning/e2_action_divergence_substrate_design.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_e2_forward_model_action_divergence/SYNTHESIS.md
  See SD-056 (this claim), V3-EXQ-571 (substrate-readiness FAIL the substrate
  addresses), V3-EXQ-NEW-1 (substrate-readiness validation), V3-EXQ-569a
  (behavioural falsifier on fixed substrate), ARC-062 GAP-B (tactical
  first-action one-hot bypass on GatedPolicy that SD-056 generalises),
  MECH-256 (single-pass forward-model comparator family; SD-056 sits at
  the world_forward training-objective layer), MECH-309 (logical-necessity
  diversity claim that SD-056 unblocks at the substrate level), MECH-314a /
  MECH-320 / MECH-295 / SD-033a / SD-033b (downstream bias-channel consumers
  of cand_world_summaries that recover per-candidate signal once SD-056
  lands), MECH-094 (simulation_mode argument standard pattern), SD-005
  (z_world / z_self split; substrate dependency), ARC-033 (E2_harm_s
  forward family; sibling per-stream forward predictor, not subject to this
  SD -- z_world is the collapse site, not z_harm_s).
