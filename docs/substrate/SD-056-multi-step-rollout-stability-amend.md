## SD-056 multi-step rollout stability amend (2026-05-31)
- SD-056 amend: e2.action_conditional_divergence_contrastive multi-step rollout
  stability -- IMPLEMENTED 2026-05-31. Amends -- does not supersede -- the
  SD-056 t=1 substrate landed 2026-05-29.
  Module: ree_core/predictors/e2_fast.py (new helper
  E2FastPredictor.world_forward_contrastive_loss_multistep; per-step output
  norm clamp inside E2FastPredictor.rollout_with_world).
  Triggered by V3-EXQ-569e Pathway A vs B mechanism-probe autopsy 2026-05-31
  (verdict_cell=INSTRUMENTATION_FAILURE): SD-056 contrastive training produced
  numerically explosive E2 rollouts (1e16+ magnitudes) on most ON-arm seeds
  at the behavioural-runtime episode length (P1 50 ep / 200 steps). 569d t=1
  measurements at the SAME contrastive weights {0.01, 0.05, 0.20} clean
  (rollout_skipped_nonfinite=0, top2_class_gap NaN-fraction=0.0). The
  substrate is stable at its t=1 training horizon; the missing piece is
  iterated multi-step rollout stability over the full-horizon
  E2.get_world_state_sequence() consumer surface that the M1 / M3 / M4 / M5
  569e measurement channels depend on.
  Two togglable levers (both default OFF; bit-identical to pre-amend SD-056):
    Lever (a) MULTI-STEP CONTRASTIVE (PRIMARY): extends the t=1 InfoNCE
      objective to an h-step rollout horizon (Dreamer / PlaNet / Srivastava
      2021 contrastive RSSM anchor). Helper
      world_forward_contrastive_loss_multistep(z_world_0, action_sequences,
      z_world_targets, ...) returns the horizon-mean cross-entropy with the
      same MECH-094 / K<2 / min_batch_classes / simulation_mode defensive
      returns as the t=1 helper. Same caller composition pattern: caller
      multiplies by e2_action_contrastive_weight at the loss-summation site.
    Lever (b) PER-STEP OUTPUT NORM CLAMP (DEFENSIVE): inside
      rollout_with_world loop, B2-anchor clamp predicted z_world_{t+1}
      against ratio * ||z_world_0|| (NOT ratio * ||z_t||) so the bound does
      not compound across the rollout horizon. initial_z_world.detach() for
      the threshold so gradient does not flow into the anchor. Default
      ratio=2.0 matches the autopsy acceptance criterion (rollout magnitudes
      within 2x of OFF baseline).
  Config (E2Config + REEConfig.from_dims; 5 new fields):
    e2_action_contrastive_multistep_enabled (default False) -- lever (a) master.
    e2_action_contrastive_horizon (default 5) -- Dreamer-default; calibratable.
    e2_action_contrastive_horizon_weights_decay (default 1.0) -- per-step weight
      decay; 1.0 = uniform across rollout horizon.
    e2_rollout_output_norm_clamp_enabled (default False) -- lever (b) master.
    e2_rollout_output_norm_clamp_ratio (default 2.0) -- B2 anchor: max
      ||z_t|| / ||z_world_0||.
  Architectural choice: implement BOTH levers (autopsy ranks (a) primary,
  (b) tactical; not mutually exclusive per autopsy Section 9). Lever (a)
  trains to bound the substrate at the training horizon scale (architecturally
  correct fix). Lever (b) provides an inference-time hard guarantee on every
  probe tick regardless of training state or OOD probe configurations. The
  acceptance criterion (max-NaN-fraction < 0.05 + rollout magnitudes within
  2x of OFF baseline) is met by (a) on average, by (b) as a hard guarantee.
  Lever (c) consumer-side bounding in M1 metric is cosmetic per autopsy and
  is NOT implemented at the substrate level (the script-side acceptance-
  criteria fixes mentioned in autopsy Section 6 land in the post-amend
  /queue-experiment session, not here).
  Scope (NOT changed): t=1 world_forward_contrastive_loss helper unchanged;
  cand_world_pairwise_dist diagnostic unchanged; world_forward signature +
  body unchanged; world_transition / world_action_encoder shapes + inits
  unchanged; predict_next_state / predict_next_self / action_object / forward /
  forward_counterfactual unchanged; E1, E3, hippocampal module, residue field,
  all downstream consumers unchanged.
  Backward compatible: all five new fields default to no-op; 590/590
  contracts + 7/7 preflight PASS with master OFF (regression-clean 2026-05-31;
  was 580 + 10 new MECH/SD-056-amend contracts in
  tests/contracts/test_sd_056_multistep_amend.py covering A1 config defaults,
  A2 from_dims propagation, A3 helper surface + grad-flow, A4 MECH-094 gate,
  A5 K<2 short-circuit, A6 min_batch_classes floor, A7 horizon clamps
  gracefully, A8 rollout clamp OFF bit-identical, A9 clamp ON enforces B2
  bound, A10 clamp blocks NaN/Inf under stress at 200-step horizon).
  Activation smoke (2026-05-31, both levers ON, h=5, ratio=2.0,
  world_transition weights 10x-amplified, 50-step rollout): multistep loss
  4.75 with grad-norm 1073.7 on world_transition (gradient flows); max
  ||z_world_t|| = 10.91 across all 50 steps equals 2.0 * max(||z_world_0||)
  exactly (B2 bound tight); 0 NaN/Inf anywhere along the rollout. Confirms
  end-to-end wiring with both levers under deliberately-unstable training
  state.
  Phased training: NOT required at substrate level. Multi-step contrastive
  trains the same world_transition + world_action_encoder weights as
  L_recon and the existing t=1 contrastive. Joint training is the
  designed-for case (matches the 2026-05-29 SD-056 landing rationale).
  MECH-094: world_forward_contrastive_loss_multistep accepts simulation_mode
  kwarg returning tensor(0.0); same defensive pattern as the t=1 helper,
  SD-035, MECH-279, MECH-313, MECH-314, MECH-319, MECH-320, MECH-341.
  Rollout clamp is a numerical guard (bounds a forward computation, not
  memory content); not gated by MECH-094.
  Downstream beneficiaries (the load-bearing rationale for the amend):
    ARC-065 GAP-A -- behavioural diversity Pathway A vs B mechanism
      dissociation (the V3-EXQ-569e probe blocked by the instability).
    MECH-309 -- logical-necessity claim for behavioural diversity;
      downstream consumer of action-discriminability at the rollout horizon.
    MECH-341 + ARC-062 GAP-B -- per-candidate signal preservation for the
      ARC-062 gated-policy heads + lateral-PFC consumers. The t=1 path
      already works for these via 569d; multi-step consumers also need
      stability post-amend.
  569c headline reading (~2.4x C3 lift over matched-noise control) remains
  the load-bearing finding on ARC-065 GAP-A pending the amend-and-re-run
  cycle.
  Validation experiment: substrate-readiness diagnostic V3-EXQ-NEW (3-arm
  probe: SD-056-OFF baseline / multi-step ON + clamp OFF / both ON) at the
  569e-equivalent P1 budget (50 ep / 200 steps, 3 seeds). Diagnostic-purpose;
  claim_ids=[]. Acceptance: max-NaN-fraction < 0.05 across both ON arms
  AND rollout magnitudes within 2x of ARM_0 OFF baseline. Queued at the end
  of this implement-substrate session (Step 8 of the skill).
  Behavioural validation (the full 8-arm V3-EXQ-569e-equivalent Pathway A
  vs B falsifier on the amended substrate, bundled with the three script-
  side acceptance-criteria fixes from autopsy Section 6) is the next
  /queue-experiment session per autopsy Section 8 -- NOT bundled in this
  implement-substrate session per the chip's explicit scope.
  Design doc: REE_assembly/docs/architecture/sd_056_e2_action_conditional_divergence.md
  (new section "Multi-step rollout stability amend (2026-05-31)").
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-569e_2026-05-31.md
  See SD-056 (parent claim; t=1 substrate this extends), V3-EXQ-569e
  (autopsy that routed this amend session via INSTRUMENTATION_FAILURE
  verdict + Section 9 amend options ranking), V3-EXQ-569c (the load-bearing
  ARC-065 GAP-A reading the amend preserves; ~2.4x C3 lift), V3-EXQ-569d
  (sister floor-recalibrated falsifier; PASS evidence that the t=1 substrate
  is sound), ARC-065 GAP-A (the behavioural-diversity validation surface the
  amend unblocks), MECH-309 / MECH-341 / ARC-062 GAP-B (downstream multi-step
  consumers), Srivastava et al. 2021 contrastive RSSM (lit-pull anchor for
  lever (a); the t=1 SD-056 already grounds in this paper), Dreamer family
  (Hafner et al. 2019/2020; multi-step latent-dynamics training pattern),
  MECH-094 (simulation_mode argument standard pattern), SD-005 (z_world /
  z_self split; substrate dependency unchanged).
