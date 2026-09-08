## SD-063: E2 Conditional Predictive-Uncertainty Head (z_world quantile) (2026-07-05)
- SD-063: e2.conditional_predictive_uncertainty_head -- IMPLEMENTED 2026-07-05
  (substrate; v3_pending until a validation experiment shows the head's per-point
  predictive variance improves E3 commitment gating over the running-variance EMA
  AND the SD-031 agency residual is preserved under joint training. PROMOTES
  NOTHING). The concrete realization of the MECH-059 confidence channel: a
  distribution-free quantile/pinball head over (z_world, action) that emits a
  per-input predictive spread tracking realized error, feeding E3's commit gate
  in place of the state-blind, global running-variance EMA. Winner of the
  V3-EXQ-712 diagnostic (quantile CRPS 0.00486 vs point 0.00514;
  precision_error_corr 0.379 vs the EMA null 0.0; Gaussian-family heads -- hetero,
  mixture -- did WORSE than the point baseline, so the distribution-free form is
  load-bearing).
  Module: ree_core/predictors/e2_world_uncertainty.py (E2WorldUncertaintyHead +
  E2WorldUncertaintyConfig; QUANTILE_LEVELS = 9 levels 0.1..0.9, the 712 winner;
  IQR_TO_STD_10_90 = 2.5631). Trunk = 2-layer MLP(ReLU) -> Linear(D*Q) -> [B,D,Q]
  (matches the 712 QuantileHead). compute_loss = pinball; predictive_variance /
  predictive_std = monotone-rearranged (torch.sort, anti-crossing) [q0.1,q0.9] IQR
  -> Gaussian-reference variance, meaned over dims, per batch item, under no_grad.
  Data flow: z_world_t.detach() + a_onehot -> E2WorldUncertaintyHead -> pinball
  (P1) / predictive_variance (read) -> E3.select(conditional_predictive_variance=)
  commit gate.
  Config: LatentStackConfig.use_e2_world_uncertainty (bool, default False;
  bit-identical OFF) + e2_world_uncertainty_hidden_dim (128) +
  e2_world_uncertainty_lr (1e-3); E3Config.use_conditional_precision_gate (bool,
  default False). Both surfaced by REEConfig.from_dims. Like use_e2_world_forward,
  the flag signals intent; the head is instantiated at the experiment/agent level
  -- LatentStack.encode() is UNTOUCHED, so no new LatentState field and OFF is
  byte-identical. z_world_dim is REQUIRED at construction (no literal default);
  unlike E2WorldForward there is NO world_dim>=128 assert (this is a predictive-
  spread readout, not the SD-031 discriminative comparator; the 712 diagnostic ran
  at world_dim=32).
  E3 consumer (e3_selector.py select()): new kwarg conditional_predictive_variance
  (default None). When E3Config.use_conditional_precision_gate is True AND a value
  is supplied, the ARC-016 commit decision compares that per-input variance against
  effective_threshold INSTEAD of self._running_variance; None or flag-off -> EMA
  fallback (byte-identical). Does not touch the use_harm_variance_commit path.
  SD-031 AGENCY-RESIDUAL GUARD (the load-bearing caveat): the head is a SEPARATE
  nn.Module sharing NO parameters with E2WorldForward or the encoder, and its P1
  loss reads DETACHED z_world inputs AND a DETACHED z_world_next target. Its
  gradients therefore never reach the forward model that produces the SD-031
  agency residual -> it cannot explain the residual away by construction. The
  validation experiment must still confirm this empirically under joint training.
  Phased training (validation, not the substrate): P0 z_world encoder warmup
  (SD-009 + SD-018); P1 head on frozen z_world (detach inputs + target); P2 eval
  CRPS + precision_error_corr + agency-residual preservation.
  MECH-094: DOES NOT APPLY -- waking online forward-model read for commitment
  gating; no memory write, no simulation/replay.
  Backward compatible: both switches False by default; agent hot path unchanged;
  1381/1385 suite PASS (the 4 fails are pre-existing, unrelated: E1 SD-016 proj-dim,
  control-vector bit-identical, 2x runner fail-branch -- all fail on the clean base
  tree). 15/15 new contracts in tests/contracts/test_sd063_conditional_uncertainty_head.py
  (config no-op + from_dims surface / head shapes + dim-required + level validation /
  pinball-trains + heteroscedastic conditional variance / SD-031 param-disjoint +
  detach-blocks-encoder-grad / E3 gate OFF-ignores + ON-overrides-both-directions +
  ON-no-value-EMA-fallback).
  Validation experiment: V3-EXQ-716 queued (see below) -- diagnostic falsifier,
  PROMOTES NOTHING.
  Design doc: REE_assembly/docs/architecture/sd_063_e2_conditional_uncertainty_head.md
  See MECH-059 (confidence channel; instantiated), SD-031 / E2WorldForward (the
  agency residual this must not disturb; dep), MECH-256 (comparator family),
  V3-EXQ-712 (motivating diagnostic), ARC-016 (dynamic-precision commit gate this
  feeds).
