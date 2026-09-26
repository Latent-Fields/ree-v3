## SD-032b AMENDMENT: dACC affective-PE scale normalisation (MECH-268 calibration) (2026-09-26)

- SD-032b amendment: cingulate.dacc_pe_scale_normalisation -- IMPLEMENTED 2026-09-26
  (substrate_queue `dacc-pe-scale-normalisation`, IGW-20260925-219; user decision
  `dec-20260923T185804-MECH-268` option 2, `rec-20260925-02281b11`).
  `DACCAdaptiveControl._normalise_pe_u()` in `ree_core/cingulate/dacc.py`, called from
  `_affective_pe()` between the raw affective quantity and the MECH-258 precision gain.
  Config: `REEConfig.dacc_pe_norm_enabled` (default False, no-op; set True to enable),
  `dacc_pe_norm_target` (0.5), `dacc_pe_norm_alpha` (0.001), `dacc_pe_norm_floor` (0.1).
  All three config sites (dataclass, `from_dims()` signature, assignment) plus the
  `DACCConfig` propagation in `REEAgent.__init__`.
  Data flow: `pe_u` (`||z_harm_a - pred||`, or `||z_harm_a||` when no `E2HarmAForward`
  prediction) -> per-statistic running mean `scale_k` (k = pred / nopred; exact cumulative
  mean for the first 1/alpha updates, then EMA) -> `pe_u * target / max(scale_k, floor)` ->
  `* (1 + prec_norm)` (MECH-258, unchanged) -> SD-034 cap -> MECH-268 f_sat -> bundle `pe` ->
  `SalienceCoordinator` `dacc_pe` (affinity + salience), `foraging_value`, Shenhav
  `control_required`.
  Estimator reads `pe_u` only (never the capped or saturated value), so it cannot undo the
  SD-034 cap or MECH-268 saturation. Buffers `_pe_norm_scale` / `_pe_norm_count` ([2],
  nopred/pred) are registered unconditionally and never touched when OFF;
  `_load_from_state_dict` fills them when absent, so pre-build checkpoints strict-load and an
  ON snapshot strict-loads into an OFF arm. NOT cleared by `reset()`, `reset_episode_pe()`
  or `reset_outcome_history()` (cross-episode operating-point estimate).
  `freeze_pe_norm(frozen=True)` stops updates (for eval arms sharing one snapshot).
  Diagnostics: `dacc.pe_norm_scale`, `dacc.pe_norm_updates`; ON-only bundle keys
  `pe_prenorm_unsaturated`, `pe_norm_scale` (divisor this tick), `pe_norm_updates`.
  Why: MECH-268's f_sat floor (0.357 at s=0.3) is fixed while `dacc_pe`'s scale drifts with
  training and seed (V3-EXQ-1089 trained seeds' unsaturated pe p50 2.12-5.61; ~16 on
  464d/467d), so the floor released the SD-032a register on some seeds and not others. With
  precision capped (all 1089 seeds) steady-state unsaturated `dacc_pe` = target * 4 = ~2.0,
  ~0.71 at the s=0.3 floor, on every seed (red-team simulation on 1089's per-tick streams:
  p50 1.98-2.21, floor-release fraction 0.99-1.00). Rule of thumb when moving
  `external_task_bias`: target * 4 * floor(s) < critical margin < target * 4.
  On configs without `use_e2_harm_a` the normalised quantity is the harm-latent magnitude,
  not a prediction error.
  Backward compatible: default OFF bundle keys/values and pe stream bit-identical.
  Phased training required: no (no learned parameters). MECH-094: not implicated
  (waking control-plane signal, no memory write).
  Contracts: `tests/contracts/test_dacc_pe_scale_normalisation.py` (19: OFF identity,
  scale invariance, steady state, warm-up, saturation/cap not undone, floor, per-statistic
  scales, persistence + cross-flag strict loads, freeze, from_dims plumbing, the 1089
  seed-spread property, live-agent liveness). Flag-inertness: `dacc_pe_norm_enabled` in PROBED.
  Design + red-team record: REE_assembly `docs/architecture/dacc_pe_scale_normalisation.md`.
  Validation experiment: V3-EXQ-1089a (not yet queued; chipped to /queue-experiment -- must
  run ON at s=0.3, freeze the scale at eval start, log `pe_norm_scale` per arm, gate
  readiness on `pe_norm_updates > 0`, keep a normaliser-OFF arm).
  See MECH-268, MECH-258, SD-032a, SD-034.
