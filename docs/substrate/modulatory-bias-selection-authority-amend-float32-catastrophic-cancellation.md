## modulatory-bias-selection-authority AMEND: float32 catastrophic-cancellation fix (V3-EXQ-643a, 2026-06-06)
- modulatory-bias-selection-authority amend -- IMPLEMENTED 2026-06-06. Module:
  ree_core/predictors/e3_selector.py (E3TrajectorySelector.select authority block).
  Routed by the V3-EXQ-643 substrate-readiness validation FAIL (2026-06-06,
  modulatory_authority_active_frac=0.0 + scale_factor_mean=0.0 on BOTH ON arms,
  0/3 seeds). The 2026-06-03 substrate computed the combined modulatory
  contribution as `modulatory_total = scores - scores_raw` -- reconstructing it by
  SUBTRACTING two primary-magnitude score tensors. CODE-CONFIRMED ROOT CAUSE
  (corrects failure_autopsy_V3-EXQ-643_2026-06-06, whose stated root cause
  "the modulatory bias is uniform across the K candidates ... range < 1e-6" is
  NUMERICALLY WRONG): the MECH-341 entropy bonus DOES carry a real ~0.17
  cross-candidate range (it keys on first-action class, NOT z_world, so it is NOT
  uniform). The 643 harness trains SD-056 online (e2_action_contrastive) which
  drove the primary E3 scores to ~1e32 (z_world instability; the SD-056 multistep
  rollout-norm clamp is OFF in 643). At ~1e32 the float32 ULP is ~1e25, far above
  0.17, so `scores - scores_raw` collapses to EXACTLY 0.0 -> modulatory_range <
  floor -> the gate never fired. Diagnostic probe confirmed: SD-056-training-OFF
  raw_score_range ~4.5 (gate fires, modrange ~0.25, active=True); SD-056-training-ON
  raw_score_range ~1e32 (modrange 0.0, never fires). The substrate is FUNCTIONALLY
  CORRECT at normal score magnitude; 643 failed from a harness-induced numerical
  degeneracy.
  THE FIX (numerically robust; bit-identical OFF; bit-identical when scores small):
  track the combined modulatory contribution EXPLICITLY as a small accumulated
  tensor `_modulatory_accum` = (score_bias actually added) + (MECH-341 entropy
  bonus actually added), captured at the two add sites. The authority block now
  measures `modulatory_total = _modulatory_accum` (range computed from the small
  ~0.17 bias tensor, independent of primary-score magnitude) instead of
  reconstructing it by subtraction. Mathematically identical to (scores -
  scores_raw) in exact arithmetic; immune to large-score cancellation. The apply
  line `scores = scores_raw + scale_factor * modulatory_total` is unchanged. New
  diagnostic `modulatory_authority_range` on E3Selector.last_score_diagnostics
  exposes the true range the gate keyed on (0.17, not 0) for validation.
  Backward compatible: the whole block is gated on use_modulatory_selection_authority
  (default False) -> bit-identical OFF. When ON with small primary scores the change
  is mathematically identical (differs only in low-order float bits; argmin
  unchanged). 836 contracts (835 prior + 1 new regression) + 7/7 preflight PASS.
  New contract test_modulatory_authority_survives_large_primary_scores
  (tests/contracts/test_e3_score_bias_candidate_support.py): large per-candidate
  world states -> raw_score_range > 1e10, a 0.5-range bias is annihilated by the
  pre-fix subtraction but the post-fix gate reports modulatory_authority_range==0.5
  and active=True (the pre-fix code FAILS this test).
  Proof-of-fix probe (643 config + harness at reduced budget): pre-fix
  authority_active_frac=0.000 / modrange=0; post-fix authority_active_frac=1.000 /
  modrange=0.16 -- the real entropy-bonus range now reaches the gate. NOTE the
  scale_factor is ~1e27 at 1e26-magnitude scores (a VACUOUS fire on degenerate
  scores): the substrate fire is now correct, but a MEANINGFUL 643a REQUIRES the
  harness to keep primary scores bounded -- the V3-EXQ-643a re-validation enables
  the SD-056 rollout-norm clamp (e2_rollout_output_norm_clamp_enabled) and adds a
  raw_score_range readiness precondition (the run is substrate_not_ready_requeue if
  e3_raw_score_range_mean exceeds a sane bound -- a non-vacuity gate, not an
  authority verdict).
  Phased training: N/A (pure arithmetic; no learned parameters). MECH-094: N/A
  (waking committed-selection path; no replay write surface; unchanged).
  Validation experiment: V3-EXQ-643a (supersedes V3-EXQ-643) -- 643 re-run with
  SD-056 online-training numerical stability (rollout clamp) + raw_score_range
  non-vacuity precondition + the 604a curiosity non-degeneracy guard retained.
  claim_ids=[] (substrate-readiness). PASS unblocks the per-claim evidence retests
  of MECH-314/320/341. substrate_queue.ready stays FALSE until 643a clears the
  non-vacuity gate AND the authority changes selection on bounded scores.
  Design doc: REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
  (V3-EXQ-643a fix section). Autopsy correction:
  failure_autopsy_V3-EXQ-643_2026-06-06 Section 2 root cause superseded.
  See modulatory-bias-selection-authority (parent substrate; gap-relative scaling
  landed 2026-06-03), V3-EXQ-643 (the FAIL this amend addresses), V3-EXQ-643a
  (validation), MECH-341 (entropy bonus -- the modulatory channel that genuinely
  carries cross-candidate range), SD-056 (e2_action_contrastive online-training
  instability that exploded the scores; rollout clamp is the harness fix),
  MECH-314 / MECH-320 (z_world-derived channels -- uniform under cand_pairwise=0,
  but NOT the binding cause), MECH-090 (admission gate; unchanged), MECH-094 (N/A).
