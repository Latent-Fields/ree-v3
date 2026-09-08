## f_dominance_conversion_ceiling rung 3 / SD-E3-CHANNEL-COMMENSURABILITY (MECH-439) -- E3 channel-commensurability operator -- IMPLEMENTED (2026-09-07)
- f_dominance_conversion_ceiling rung 3 / SD-E3-CHANNEL-COMMENSURABILITY (MECH-439) --
  E3 channel-commensurability operator -- IMPLEMENTED 2026-09-07.
  `ree_core/predictors/e3_selector.py` (`score_trajectory`, `select`,
  `_commensurability_scale`, `_update_channel_scale_estimates`,
  `_COMMENSURABILITY_CHANNELS`).
  PROBLEM (confirmed `failure_autopsy_V3-EXQ-571c_2026-09-02`, ratified by /governance
  2026-09-02 REE_assembly `0ade914d46`): `score_trajectory` sums its channels in their
  NATIVE units, so channel authority is decided by units, not content. 571c's within-tick
  cross-candidate partition found ONE channel holding 0.98-0.99999 of the variance in 15 of
  16 cells -- `residue_weighted` in all 8 residue-fed cells (F's share 4e-06 to 1.1e-05),
  F/`harm_weighted` in 7 of 8 starved cells (0.994-0.9998). The load-bearing observation:
  every competing channel cleared the 1e-12 ABSOLUTE variance floor and failed only the 1e-3
  RELATIVE share floor -- the monopoly is a SCALE phenomenon, not dead channels. So
  `n_live_channels=1` red-gated all four arms before criteria evaluation. A bound on F alone
  would merely hand the monopoly to residue, hence the JOINT channel scale.
  OPERATOR: per-channel divisive normalisation against a RUNNING scale estimate -- each
  declared channel's per-candidate term divided by an EMA of that channel's own
  cross-candidate standard deviation before the additive sum. Running (cross-tick) because
  `score_trajectory` scores ONE candidate at a time, so the tick's cross-candidate spread
  does not exist at scoring time; the estimate is folded in AFTER the candidate loop, so a
  tick is always scored against PRIOR ticks' estimate -- causal, never self-referential.
  Config: `E3Config.use_e3_channel_commensurability` (default False = bit-identical),
  `.e3_commensurability_ema_alpha` (0.05), `.e3_commensurability_warmup_ticks` (20),
  `.e3_commensurability_floor` (1e-12, deliberately 571c's own MIN_LIVE_CHANNEL_VARIANCE so
  operator floor and instrument liveness floor agree). NOT wired through
  `REEConfig.from_dims()` -- follows the `f_weight` / `e3_include_untrained_fallback_scorers`
  precedent; set per-arm as `cfg.e3.<field> = X`.
  Data flow: raw channel terms -> `_last_commensurability_raw` (ungated by the decomp flag)
  -> divided by `max(scale_ema[c], floor)` -> additive sum -> score; `select()` folds the
  tick's cross-candidate std into `_chan_scale_ema` and publishes
  `last_channel_scale_estimates`.
  SCOPE: `f_weighted`, `harm_weighted`, `residue_weighted`, `benefit_weighted`,
  `goal_weighted` -- exactly 571c's declared SCORE_COMPONENTS minus the structural zero.
  EXCLUDED on purpose: `novelty_weighted` (hardcoded 0.0, MECH-111 branch deleted
  2026-05-25) and the v4 `pe_confidence` / `self_viability` penalties (outside 571c's
  partition; normalising them would redefine what a successor measures against).
  `residue/field.py` is NOT touched -- the entry names `add_residue` / `RBFLayer.forward`
  because residue is the monopolising OCCUPANT, not because the operator lives there;
  normalisation is scale-invariant to the accumulator's magnitude by construction. (And
  `field.py::update_valence` is explicitly NOT this path -- SD-RESIDUE-VALENCE-BOUND clamps
  a different accumulator.) User-confirmed scope at the design gate 2026-09-07.
  THREE PROPERTIES EASY TO GET WRONG, each with its own contract + negative control:
  (1) the term capture is ungated by `e3_score_decomp_enabled` -- reusing
  `_last_traj_components` would make a LIVE selection-path behaviour depend on a DIAGNOSTIC
  switch; (2) the EMA is fed from RAW terms, never normalised ones -- otherwise it chases its
  own tail, every scale converges to 1.0 and the exposure is a lie; (3) a dead channel keeps
  unit scale via the absolute floor rather than being divided by its own near-zero spread.
  Rank-preserving: each channel is divided by a POSITIVE constant within a tick, so each
  channel's own candidate ordering is preserved; only authority BETWEEN channels moves.
  Verification exposure (spec-mandated): `last_channel_scale_estimates` = scales, n_updates,
  engaged, warmup_ticks, floor, ema_alpha, channels -- so a successor can VERIFY
  commensurability rather than assume it.
  READINESS TARGET (verbatim from the autopsy): >=2 channels simultaneously above a 1e-3
  relative cross-candidate share. Measured at build time on a bare selector (40 ticks, 6
  candidates, the substrate's NATIVE f/harm disparity, not an injected one): OFF n_live=1
  top_share 0.999518; ON n_live=2 top_share 0.500000, the suppressed `harm_weighted` moving
  4.82e-04 -> 5.00e-01.
  Backward compatible: bit-identical OFF, verified at full float64 against pristine
  `origin/main` over 25 `select()` ticks -- scores, selected indices and every per-channel
  decomp value, 1250 values, zero differences.
  Not a learning module -- no encoder, no learned parameters, no phased training, MECH-094
  N/A. A BatchNorm-style learned affine was deliberately NOT adopted: untrained parameters on
  the live selection path is exactly the SD-E3-SCORER-COMPLETION defect.
  Contracts: `tests/contracts/test_e3_channel_commensurability.py` (13 tests). The file OPENS
  with `test_monopoly_is_reproduced_with_the_operator_off`, the negative control for the
  whole file -- if the substrate stops exhibiting the 571c monopoly that test fails, so the
  suite cannot pass vacuously.
  Validation experiment: OWED, not yet queued -- the regime-level (936) validation was
  pacing-gated at build time (4 un-adjudicated PASS results against the wave-4 limit of 3);
  chipped for queueing once governance clears the backlog. The bare-selector measurement
  above is a build-time smoke check, NOT the regime-level validation.
  Biological basis: divisive normalisation as a canonical cortical computation (Carandini &
  Heeger 2012); the running-scale form is its adaptation face (contrast-gain adaptation,
  Fairhall et al. 2001). Consistent with `targeted_review_connectome_mech_439`.
  See MECH-439 (primary), ARC-062, MECH-309, MECH-341, MECH-448/449 (the eligibility-face
  levers this rung sits beside), SD-085, SD-E3-SCORER-COMPLETION,
  `REE_assembly/docs/architecture/sd_e3_channel_commensurability.md`,
  `substrate_queue.json::f_dominance_conversion_ceiling`.
