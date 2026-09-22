## SD-PP-2 World-Forward Epistemic Precision (2026-09-22)
- SD-PP-2: precision.world_forward_epistemic_precision -- IMPLEMENTED 2026-09-22.
  The MODEL-precision producer of the precision-provenance set: it estimates how
  precise the organism's own world-forward model (`E2FastPredictor.world_forward`)
  is about the next z_world, and splits that estimate into a total predictive
  variance `v_tot`, an ALEATORIC part `v_ale` attributed to the evidence channel
  (SD-PP-1), and the EPISTEMIC remainder `v_epi = max(v_tot - v_ale, v_floor)`
  with `pi_epi = 1 / v_epi`. It is the second of the three quantities the
  motivating intake keeps distinct (evidence precision = SD-PP-1; model precision
  = here; episode-linked prediction error = the SD-PP-3 packet's `pe`).
  THE HISTORICAL/CURRENT DISTINCTION IS ONE PRODUCER READ AT TWO TIMES, which is
  what makes it operational rather than nominal: `precision_at(z_prev, a)` called
  BEFORE the outcome of that transition is HISTORICAL model precision (rides in
  the packet as `pi_hist`); `current_read()` at sleep entry is CURRENT model
  precision (`pi_cur` for the SD-PP-4 gain rule). There is no second estimator
  whose disagreement with the first could be an artefact of a different statistic.
  Module: ree_core/precision/world_forward_epistemic_precision.py
  (`WorldForwardEpistemicPrecision` + `WorldForwardEpistemicPrecisionConfig` +
  `PrecisionRead`; module constants `READ_SOURCES = ("ema","sd063")`,
  `CONFIG_SOURCES = ("ema","sd063","sd063_or_ema")`).
  API: `prediction_at_test(e2, z_prev, a_onehot) -> [1,D]` (exactly
  `e2.world_forward(z_prev[:1], a_onehot[:1])` under no_grad, returned as a
  detached clone); `precision_at(z_prev, a_onehot, head=None) -> PrecisionRead`
  (CURRENT state only -- never the outcome, never a mutation);
  `observe_outcome(pred, z_now, evidence_variance_z) -> float pe` (reads ONLY its
  arguments); `current_read() -> PrecisionRead`; properties `v_tot`, `v_noise`,
  `v_epi`, `pi_cur`, `n_obs`; `snapshot()` / `get_metrics()` (prefix
  `wf_precision_`). `PrecisionRead` carries `pi_epi, v_tot, v_ale, v_epi, source`.
  SOURCES + FALLBACK RULE: `"ema"` is the global calibrated EMA of the per-dim
  mean squared PE of `world_forward` over waking transitions (alpha
  `pe_ema_alpha`, initialised to `v_init`), always available but STATE-BLIND.
  `"sd063"` is the per-(z,a) read `head.predictive_variance(z, a)[0]` from the
  SD-063 `E2WorldUncertaintyHead`, used only when `head is not None and
  head.training_ready`. `"sd063_or_ema"` (the default) takes the head when ready
  and the EMA otherwise. Source `"sd063"` with no head, or a head whose
  `training_ready` is False, ALSO falls back to the EMA read and REPORTS
  `source="ema"` -- `PrecisionRead.source` always names what actually served the
  read, never the configured intent, which is what lets the packet distinguish a
  per-state read from a state-blind one. `current_read()` is ALWAYS the EMA read
  by construction (a global quantity has no (z,a) to condition on).
  NOISE SPLIT: `v_noise <- (1-alpha) v_noise + alpha * (noise_gain *
  evidence_variance_z)` (same alpha as v_tot, initialised 0.0; 0.0 when SD-PP-1
  is absent, which collapses `v_ale` to 0 and makes `v_epi == v_tot`), then
  `v_epi = max(v_tot - v_noise, v_floor)` and `pi_epi = 1/v_epi`.
  `noise_gain = 2.0` is PRE-REGISTERED FROM FIRST PRINCIPLES, not fitted:
  observation noise enters the PE through BOTH the input (z_t) and the target
  (z_{t+1}), and for a near-identity head -- which MECH-573 measures the
  converged head to be, skill ~0 -- the PE noise variance is ~2x the per-frame
  z-noise variance. NAMED ASSUMPTION, recorded here as required by the contract.
  Config: `WorldForwardEpistemicPrecisionConfig` --
  `use_world_forward_epistemic_precision` (bool, default False),
  `source` (str, default "sd063_or_ema"), `pe_ema_alpha` (float, 0.05),
  `v_floor` (float, 1e-6, per-dim z units), `noise_gain` (float, 2.0),
  `v_init` (float, 1e-2 -- the fresh-base residual scale measured in
  V3-EXQ-1063). `__post_init__` validates: source in CONFIG_SOURCES,
  0 < pe_ema_alpha <= 1, v_floor > 0, noise_gain >= 0, v_init > 0.
  Enabled by `REEConfig.use_world_forward_epistemic_precision=True` with the
  surfaced knobs `world_forward_precision_source`,
  `world_forward_precision_pe_ema_alpha`, `world_forward_precision_v_floor`,
  `world_forward_precision_noise_gain` (WIRED BY THE SESSION, single writer --
  this module ships the producer only and touches no agent/config file).
  Data flow: `_e1_tick` -> `prediction_at_test(e2, z_prev, a)` ->
  `precision_at(z_prev, a, head=e2_world_uncertainty)` BEFORE the outcome ->
  `observe_outcome(pred, z_now, evidence_variance_z)` AFTER it -> the read and
  the pe ride in the `ReplayProvenancePacket` (SD-PP-3, index-bound to
  `agent._world_experience_buffer`) -> at sleep, `current_read().pi_epi` is
  `pi_cur` for `compute_provenance_gains` and the consolidation step scale
  (SD-PP-4). ORDERING IS THE CONTRACT: the read must precede the update, and
  `observe_outcome` reads nothing but its arguments, so the historical read is
  provably free of future information (test T5).
  Backward compatible: disabled by default; OFF is bit-identical BY STRUCTURAL
  ABSENCE -- with the flag False the integration layer constructs no estimator
  and makes no call. Pure float/tensor arithmetic, NO RNG draws anywhere (test
  T7 pins `torch.get_rng_state()` byte-identical across 100 read+update calls);
  no tensor with `requires_grad` is ever stored; no checkpoint or serialisation
  format change (the estimator's whole state is two floats and a counter).
  ASCII-only output.
  Biological basis: confidence is maintained per LEARNED RELATION and combined
  with surprise rather than replacing it -- Meyniel & Dehaene (2017) show rIFG /
  IPS carrying a confidence signal about a learned transition statistic that is
  read out jointly with the prediction error, which is exactly the
  `(pi_hist, pe)` pair this module produces and SD-PP-4 consumes. CA1 mismatch
  responses are scaled by the STRENGTH of the prediction being violated (Chen
  2015), the hippocampal correlate of weighting a contradiction by the precision
  of the belief it contradicts -- the anti-self-sealing direction SD-PP-4 pins
  (historical precision never enters as a multiplier < 1).
  Phased training: this module trains NOTHING. When the `sd063` source serves a
  read it inherits SD-063's own P0/P1/P2 discipline (P0 z_world encoder warmup,
  P1 head on frozen/detached z_world inputs AND targets, P2 frozen in eval), and
  the SD-031 agency-residual guard therefore continues to hold unchanged -- the
  read is `predictive_variance`, computed under the head's own `no_grad`, so no
  gradient path is created by consuming it here.
  Validation experiment: V3-EXQ-1073 (reserved). Preregistration:
  REE_assembly/evidence/planning/precision_provenance_consolidation_gain_design_20260922.md.
  Contract: REE_assembly/docs/architecture/precision_provenance_substrate_spec.md
  section 3. Tests: 8/8 in
  tests/contracts/test_sdpp2_world_forward_epistemic_precision.py (default OFF +
  config validation / EMA maths exact against a hand-rolled EMA / v_epi floored /
  noise split halves v_epi at evidence_variance_z = v_tot/(2*noise_gain) /
  no-future-info ordering / sd063 fallback both directions incl. source="ema"
  never consulting a ready head / no RNG consumption / prediction_at_test bitwise
  equal to e2.world_forward with no grad).
  LIMITATIONS (named, not papered over):
  (a) the `"ema"` source is GLOBAL and STATE-BLIND -- one scalar for the whole
      policy, the same property SD-063 criticises in the E3 running-variance EMA;
      a historical read at a hard (z,a) is indistinguishable from one at an easy
      (z,a), so on the EMA source `pi_hist` varies only with TIME, not state.
  (b) the SD-063 head's `predictive_variance` is TOTAL predictive spread -- it
      absorbs observation noise along with model uncertainty, so it cannot by
      itself separate model precision from evidence precision. THE EPISTEMIC
      SPLIT HERE IS THEREFORE BY SUBTRACTION of an EMA'd, gain-scaled evidence
      variance: an estimate, not a measurement. A per-state estimator that
      separates epistemic from aleatoric natively is spec substrate-necessity (f)
      and is NOT built here.
  (c) `v_ale` is a GLOBAL EMA even when `v_tot` is per-state, so the subtraction
      applies an average noise attribution to an instantaneous total.
  (d) `noise_gain = 2.0` inherits the near-identity-head assumption; if the world
      head ever departs from ~identity the factor is wrong in an unmeasured
      direction. It is a config field precisely so the preregistration can freeze
      or revise it.
  (e) `evidence_variance_z` itself inherits SD-PP-1's kappa limitation (measured
      on real motion, applied to noise -> OVER-estimates z-noise), so the
      aleatoric attribution is conservative and `v_epi` is biased LOW (precision
      biased HIGH) under noise.
  See MECH-572 (lead: consolidation gain is provenance-blind), MECH-573
  (near-identity converged world head), MECH-016, ARC-055, MECH-059 (the
  confidence channel this instantiates for the world-forward model), SD-063
  (`E2WorldUncertaintyHead`, the optional per-state source), SD-PP-1
  (`ObservationReliabilityEstimator`, the `evidence_variance_z` producer),
  SD-PP-3 (`ReplayProvenancePacket`, the carrier), SD-PP-4
  (`provenance_gain`, the consumer).
