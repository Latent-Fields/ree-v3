## ARC-071 / MECH-324: chunk dissolution is SUPPRESSION-WITH-RETENTION, not erasure -- IMPLEMENTED (2026-07-27)
- ARC-071 / MECH-324: chunk dissolution is SUPPRESSION-WITH-RETENTION, not erasure --
  IMPLEMENTED 2026-07-27 (ree-v3 6c3e67e). A structural correction to the DISSOLVED state
  of the 2026-07-22 build above, not a new mechanism.
  ree_core/policy/policy_chunking.py (ChunkedPrimitive.n_dissolutions / n_reacquisitions /
  reacquisition_repetitions + is_dormant property, PolicyChunkingConfig
  .reacquisition_min_repetitions, ChunkLibrary revival path, module-docstring
  dissolution-is-suppression section), ree_core/utils/config.py (both knob declaration and
  from_dims forwarding).
  Config: REEConfig.use_chunk_dissolution_retention (default False; set True to enable) plus
  REEConfig.chunk_reacquisition_repetition_factor 0.25. NESTED UNDER use_chunk_maintenance --
  retention with maintenance OFF RAISES rather than running silently inert (nothing dissolves
  with maintenance off, so the flag would otherwise read enabled in a manifest while never
  firing). The factor must scale DOWN (validated, <= 1.0).
  THE BUG WAS WORSE THAN ERASURE, WHICH IS THE THING TO UNDERSTAND HERE. DISSOLVED was
  documented as terminal-but-retained-for-the-audit-trail, and the retention was itself the
  trap: PolicyChunking.note_outcome() skips any sequence already present in the library, and
  note_real_execution() had no DISSOLVED branch, so the tombstone permanently BLOCKED its own
  sequence from re-forming at ANY number of repetitions and ANY outcome consistency. Erasure
  would at least have permitted re-formation at R_min. Measured on the contract fixture: after
  forcing a crystallised chunk to DISSOLVED, 200 further trials of the same perfectly consistent
  above-baseline regime produced zero re-formations.
  Data flow (retention ON): CRYSTALLISED -> [variance > F_high for T_dissolve trials] ->
  DISSOLVING -> DISSOLVED, which zeroes reacquisition_repetitions and increments n_dissolutions
  -> the chunk stays in the library UNSELECTABLE (is_selectable is UNCHANGED by retention; a
  dormant chunk is SUPPRESSED, so retention buys a cheaper route back, not a free pass back into
  the proposal pool) -> subsequent real executions of the same sequence increment
  reacquisition_repetitions -> the MECH-323 joint formation gate is re-entered with ONLY the
  repetition term substituted, ceil(R_min * chunk_reacquisition_repetition_factor) against that
  post-dissolution counter, while variance < F_low and mean > baseline + margin apply UNCHANGED
  -> DISSOLVED -> FORMING (not CRYSTALLISED: rapid reacquisition is a claim about the FORMATION
  threshold, while C_min is the separate Smith & Graybiel 2013 IL sub-mechanism and re-runs from
  zero) -> n_reacquisitions increments.
  REPETITIONS ARE COUNTED SINCE DISSOLUTION, NOT AS THE ACCUMULATOR'S TALLY LENGTH, and this is
  not incidental: the per-sequence tally is a sliding window capped at chunk_window_trials and a
  long-lived chunk's bucket sits saturated at that cap, so a naive "compare the bucket length
  against a lowered R_min" would clear the reduced bar on the very first post-dissolution trial
  and measure nothing at all.
  MECH-322 REPLAY-ORIGIN CHUNKS ARE NEVER REVIVED (fails closed). A chunk retired on its
  corroboration deadline died precisely because real waking execution never corroborated it;
  letting it return by a REDUCED bar would be a shortcut around the MECH-094 posture the
  carve-out exists to preserve. is_dormant is state-only and excludes replay_origin, so it reads
  correctly regardless of the flag; the flag decides only whether anything ACTS on it.
  Backward compatible: disabled by default and bit-identical when OFF -- with retention off
  DISSOLVED behaves exactly as it did at the 2026-07-22 landing (terminal and blocking), which
  is what contract test_c10_off_dissolved_is_an_absorbing_tombstone pins.
  Phased training required: no (pure arithmetic, no learned parameters, no gradients).
  Biological basis: Barnes, Kubota, Hu, Jin & Graybiel 2005 (Nature, 10.1038/nature04053) --
  sensorimotor striatal ensemble patterns across acquisition / extinction / reacquisition are
  "successively formed, reversed and then re-emerged", and "regaining a habit can occur quickly,
  with even one or a few exposures"; Bouton, Winterbauer & Todd 2012 (Behav Processes,
  10.1016/j.beproc.2012.03.004) on INSTRUMENTAL extinction, i.e. at the action level ARC-071
  operates on -- extinction "weakens behavior without erasing the original learning", installing
  new context-dependent learning ALONGSIDE the old, from which three relapse effects follow
  (renewal, resurgence, rapid reacquisition). Targeted lit-pull 2026-07-27, REE_assembly
  36901c01ed.
  f_reacq = 0.25 IS AN UNCALIBRATED ENGINEERING DEFAULT, exactly the status of
  chunk_variance_high = 0.45. Both sources establish the DIRECTION (reacquisition is materially
  faster than acquisition) and NEITHER quantifies the magnitude. Do not cite 0.25 as
  literature-derived, and do not calibrate it against F_high as though they were one quantity.
  NOT IMPLEMENTED -- the other two relapse effects are design questions, not builds, and are
  recorded in the module docstring so the gap stays visible. RENEWAL needs dissolution gated
  against the chunk's initiation_set: formation is context-conditioned while dissolution is
  context-BLIND, so a chunk dissolved in one context is dissolved everywhere and REE cannot
  exhibit renewal at all -- closing this needs per-context dissolution state, not a parameter.
  RESURGENCE needs a chunk to return because a COMPETITOR was extinguished rather than because
  its own evidence improved, which requires dissolution state shared across competing library
  candidates; registration is per-sequence and produces no such coupling.
  Contracts: 10 further tests in tests/contracts/test_arc071_policy_chunking.py (the C10 block,
  14 collected cases: OFF-by-default + maintenance precondition, from_dims forwarding, factor
  validation, the ceil() bar across R_min x factor, the OFF absorbing-tombstone baseline, ON
  re-formation below R_min, reacquisition still requiring consistency AND contrast, replay-origin
  never revived, revive failing closed on every refusal path, dormant chunks still suppressed).
  Registered PROBED in tests/test_flag_inertness.py. Green on ree-cloud-2 (63 passed targeted)
  and on a full local pre-commit gate run (2331 contracts passed).
  Validation experiment: RAN 2026-07-27T17:05:39Z AND THE FALSIFIER FAILED. V3-EXQ-829
  (script + queue entry ree-v3 77e3ddc; run
  v3_exq_829_mech324_rapid_reacquisition_falsifier_20260727T170539Z_v3 on ree-cloud-2, 66 cells =
  6 seeds x 11 arms) returned outcome FAIL, evidence_direction MIXED for MECH-324 and SUPPORTS
  for MECH-323, non-degenerate, self-routing retention_real_but_rapid_reacquisition_falsified.
  NOT YET AUTOPSIED and NOT marked reviewed -- that is /failure-autopsy and governance work.
  C3 HELD, so THE STRUCTURAL CORRECTION ABOVE IS REAL: with retention OFF all 12 of 12 dissolved
  cells stayed dead and with it ON they revived, which is exactly the absorbing-tombstone claim.
  C5 (mild-dissolution robustness) also held.
  C1 FAILED, and it is the load-bearing one: reacquisition was SLOWER than acquisition, median
  r_reacq 90 against a measured median r_acq of 20 (= R_min exactly). C2 also failed -- r_reacq
  was FLAT at 90 across every f_reacq in {1.0, 0.5, 0.25, 0.1} (forced bars 20 / 10 / 5 / 2), so
  the bar this correction introduced had NO measurable effect on the DV it was built to move.
  LIKELY READING, a hypothesis for the autopsy and NOT a verdict: revival is rate-limited by the
  VARIANCE WINDOW FLUSHING, not by the repetition bar. r_reacq / W = 0.908 +/- 0.029, and the
  pre-registered W control separates the readings cleanly (median r_reacq 90 at W = 100, 28 at
  W = 30 -- the DV tracks the WINDOW, not R_min and not f_reacq). Reaching DISSOLVED needs
  chunk_dissolve_trials = 50 trials of supra-F_high variance, so at dissolution the variance
  window is necessarily contaminated by the stream that caused it, and variance < F_low cannot
  clear until that ages out -- ~0.9 W trials, always dominating a repetition term of at most
  R_min = 20. If that holds, chunk_reacquisition_repetition_factor is INERT BY CONSTRUCTION
  rather than mis-valued, and the fix is structural (the reacquisition path needs the variance
  gate scoped to post-dissolution trials too, not just the repetition counter), not a
  recalibration. The arithmetic-identity degeneracy check passed
  (all_on_cells_sit_on_forced_bar = false), so the flatness is a real measurement.
  It is deliberately an OPERATOR-LEVEL driver rather than an agent run: V3-EXQ-810 measured the
  MECH-323 accumulator SILENT under agent control (chunk_accumulator_silent), so an agent-level
  reacquisition run would have measured that readiness gap rather than MECH-324. Evidence
  attaches to the OPERATOR, not to chunk-driven behaviour.
  See MECH-324, MECH-323, MECH-322, ARC-071, MECH-094, and the
  dissolution-is-suppression-with-retention section of policy_chunking.py's module docstring.
