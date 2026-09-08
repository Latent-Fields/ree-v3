## SD-068: sleep.consolidation_pipeline_lesion_harness -- IMPLEMENTED (2026-07-17)
- SD-068: sleep.consolidation_pipeline_lesion_harness -- IMPLEMENTED 2026-07-17.
  EXPERIMENT-LAYER harness (NO ree_core change):
  ree-v3/experiments/_lib/consolidation_lesion_harness.py. Per-phase-damageable +
  per-phase-functional-readout instrumentation on the MECH-120 (SWS denoising) ->
  MECH-121 (NREM slot-filling) -> MECH-123 (REM precision) offline-consolidation
  pipeline, so the MECH-168/INV-047/MECH-169 staged-decline-under-uniform-damage
  falsifier is buildable. Config: none added -- builds agents via existing no-op
  flags (shy_enabled/sws_enabled/rem_enabled/use_sleep_aggregation_cluster). Data
  flow: inject known clean content onto each phase's operative substrate
  (context_memory slots / consolidation params / E3 precision reference; V3-EXQ-702
  injected-content precedent, sidesteps the failure_autopsy_V3-EXQ-538a
  encoding-starvation ceiling) -> apply one UNIFORM diffuse-damage sigma
  (RMS-scaled Gaussian, identical per phase) -> read denoising-SNR (MECH-120) /
  transfer-fidelity (MECH-121) / precision-calibration-error (MECH-123) against the
  injection. NON-VACUITY: per-phase errors normalised to fractional-of-own-range
  degradation and ranked by damage-TOLERANCE (crossing sigma) for staging order,
  plus a REM passthrough-nudge-vs-generative-pass contrast (the amplify/attenuate
  indicator). SUBSTRATE FINDING: the three phases operate on DISJOINT state, so a
  faithful cross-phase content-propagation pipe is not directly instrumentable --
  non-vacuity is carried by staging-order + the REM contrast, not a fake pipe.
  Backward compatible: pure instrumentation, no existing behaviour changes. MECH-094:
  no new simulate-then-commit surface. MECH-121 hold RESPECTED (not lifted): the
  validation run is EXPERIMENT_PURPOSE=diagnostic and does NOT tag MECH-121 as
  promotion evidence; NREM leg is substrate-plumbing-fidelity only. Glymphatic/
  amyloid structural half of MECH-169 has NO V3 analog -- OUT OF SCOPE. Phased
  training: N/A (no encoder head trained on moving latents). Smoke: run_staged_sweep
  deterministic; observed tolerance order (nrem, rem, sws) stable across seeds
  42/7 (partial match to reverse-dependency prediction; REM generative sensitivity
  null). Validation experiment: V3-EXQ-778 (diagnostic staged-damage sweep, queued 2026-07-17).
  Design doc: REE_assembly/docs/architecture/sd_068_consolidation_lesion_harness.md
  See MECH-120, MECH-121 (held), MECH-123, MECH-168, INV-047, MECH-169, SD-017.
  >> SWS READOUT REBUILT 2026-07-18 (ree-v3 main 8b18338). V3-EXQ-778c's null control
  measured null_slope_ratio 1.0000 (sd 2.7e-8) on 8/8 seeds for the sws leg: `_shy` is
  AFFINE, so shy(clean+n) - shy(clean) = shy_centred(n) independent of `clean`, making
  `noise_power` content-free and the content term a constant offset that differentiates
  away. `denoising_snr_db` therefore NEVER measured content fidelity. The scored series
  is now `_sws_pattern_completion`: cosine retrieval margin of the post-SHY store
  against the injected prototypes, probed with the UNSCALED base so the null arm
  receives a real arm-identical probe that is simply not planted (Bar et al. 2020
  "same odour, no prior pairing") rather than a 0/0-degenerate zero -- the failure mode
  the rem leg already exhibits. Same repair pattern as rem_terrain_variance ->
  rem_generative_fidelity (da873a1). Backward compatible: denoising_snr_db /
  signal_power / noise_power still emitted as TELEMETRY and the error_propagation_gain
  driver keys are unchanged, so V3-EXQ-778/778a drivers still run. NOTE their staging
  numbers are NOT reproducible across this change -- tolerance_sigma_sws flows through
  the repaired series. That is intentional; those numbers were retracted as staging
  evidence by the 778c autopsy. Local smoke ONLY (seeds 42/7/123): null_slope_ratio_sws
  0.116/0.171/0.152 vs the 0.25 ceiling, injected slope 0.32-0.36, confounded_phases
  now ['rem'] alone. CAVEAT on that pass: the readout is cosine-based and therefore
  scale-invariant, so its null arm is flat in sigma partly BY CONSTRUCTION -- a
  content-scale ladder (0.0/0.25/0.5/1.0) was run as the independent check and shows
  zero response at content_scale=0 and a large amplitude-dependent response above it.
  NOT YET VALIDATED at seed scale: V3-EXQ-778g queued 2026-07-18 re-runs the null
  control on the repaired readout at the full 8-seed 778a set. Until 778g reports, do
  NOT treat the sws leg as a validated instrument.
