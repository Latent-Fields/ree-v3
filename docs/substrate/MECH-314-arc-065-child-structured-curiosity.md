## MECH-314 (ARC-065 child): Structured Curiosity Bonus + 3 Sub-Flavours (2026-05-10)
- MECH-314: policy.structured_curiosity_bonus_parent + MECH-314a/b/c sub-flavours
  -- IMPLEMENTED 2026-05-10. Module: ree_core/policy/structured_curiosity.py
  (StructuredCuriosity + StructuredCuriosityConfig). Second of four ARC-065 child
  substrates (MECH-313 noise-floor landed earlier the same day; MECH-318 / MECH-319
  remain separate spawned tasks). Pure-arithmetic, no learned parameters, no
  nn.Module inheritance; sibling to MECH-313 NoiseFloor in the ree_core.policy
  package. Three sub-flavours implemented as a single module with master + 3
  independently-togglable sub-flavour switches per Pull 1 R3 verdict NOT to
  collapse them prematurely; Q-044 holds the empirical resolution path.
  Three sub-flavours:
    MECH-314a striatal novelty (Wittmann 2008): per-candidate min-distance from
      candidate's first-step z_world to nearest ACTIVE ResidueField RBF center,
      normalised by candidate-pool mean norm. Genuinely per-candidate [K].
    MECH-314b frontopolar uncertainty (Daw 2006 / Friston 2010/2015 EFE):
      e3._running_variance scalar. Phase 1 BROADCAST scalar across [K]
      (per-candidate refinement requires E1 forward-variance head, deferred
      to Phase 2 follow-on).
    MECH-314c learning progress (Schmidhuber 1991 / Pathak 2017): EMA of
      |PE_t - PE_{t-K}| where PE feed is e3._running_variance per tick.
      Phase 1 BROADCAST scalar across [K] (per-candidate refinement
      deferred). Pull 1 R3 flagged 314c as least biologically anchored;
      Q-044 outcome may retire it without architectural cost.
  Algorithm at the e3.select() call site in REEAgent.select_action():
    total = zeros[K]
    if 314a ON and residue has active centers:
        total += -w_a * normalised_min_RBF_distance[K]
    if 314b ON: total += -w_b * unc * ones[K]
    if 314c ON and lp_seeded: total += -w_c * lp_ema * ones[K]
    return clamp(total, [-bias_scale, +bias_scale])
  Bonus is non-positive in E3's lower-is-better convention (curiosity makes
  novel/uncertain/LP-rich candidates more attractive). Composed additively into
  dacc_score_bias immediately AFTER the MECH-295 liking-bridge block and
  BEFORE the MECH-313 noise_floor temperature lift (curiosity affects scores;
  noise floor affects temperature; orthogonal). LP feed:
  curiosity.update_prediction_error(e3._running_variance, simulation_mode=False)
  called after each e3.select cycle in select_action (advances 314c LP
  buffer for next tick).
  Config: REEConfig.use_structured_curiosity (default False; bit-identical OFF
  master) + use_curiosity_novelty / _uncertainty / _learning_progress (defaults
  True; consulted only when master ON) + curiosity_novelty_weight /
  curiosity_uncertainty_weight / curiosity_learning_progress_weight (default
  0.05 each; Q-043 / Q-044 calibrate) + curiosity_bias_scale (default 0.1;
  mirrors lateral_pfc_bias_scale) + curiosity_lp_ema_alpha (default 0.1;
  ~10-tick window) + curiosity_lp_window_k (default 5; Schmidhuber 1991
  first-difference). All wired through REEConfig.from_dims().
  MECH-094: compute_score_bias(simulation_mode=True) returns zeros[K] +
  increments only the simulation-skip counter; update_prediction_error(
  simulation_mode=True) no-op on the LP buffer. Match the SD-035 / MECH-279 /
  gated_policy / MECH-313 simulation_mode pattern.
  Phase-1 architectural-placement note (mirrors MECH-313): a SEPARATE
  StructuredCuriosity module at the e3.select() call site, in parallel with
  MECH-313 NoiseFloor and the GatedPolicy bias chain. Whether the policy-
  layer regulators ultimately consolidate into one module is OPEN pending
  MECH-318 / MECH-319 substrates and Q-043 / Q-044 calibration. The
  separate-module choice keeps each sub-flavour independently togglable
  (which is what Q-044 needs).
  Phase 1 honest-scoping caveat: 314a is genuinely per-candidate; 314b and
  314c are state-dependent global scalars broadcast across [K] in Phase 1.
  The architectural shape is correct (bonus magnitude varies with global
  uncertainty / LP; substrate exposes the falsification surface), and
  Q-044's three-arm ablation IS a flag-set decision. What Phase 1 does NOT
  deliver: distinguishable behavioural signatures per sub-flavour at the
  candidate-selection level (broadcast-scalar 314b/c shifts every candidate's
  score by the same amount and does not change selection ordering).
  Per-candidate refinement is a Phase 2 follow-on, deferred until Q-044
  surfaces concrete need.
  Backward compatible: use_structured_curiosity=False by default; agent.curiosity
  is None and select_action skips the entire block + LP feed -> bit-identical
  to baseline. 273/273 contracts + 7/7 preflight PASS with master OFF
  (regression-clean, was 253 + 13 new MECH-314 tests + 7 preflight).
  Phased training: not applicable (pure-arithmetic regulator; no learned
  parameters; no gradient flow).
  Validation experiment: V3-EXQ-545 substrate-readiness diagnostic (5 sub-tests
  UC1-UC5 covering instantiation, master-OFF backward-compat, sub-flavour
  flag-set isolation -- the architectural prerequisite making Q-044 three-arm
  ablation a flag-set decision -- select_action wiring contract via
  act_with_split_obs, MECH-094 simulation gate). Smoke 5/5 PASS 2026-05-10
  (manifest scrubbed; runner will write the canonical PASS manifest from the
  queued entry). 13 contract tests in tests/contracts/test_mech_314_curiosity.py
  PASS. Behavioural validation deferred to Q-044 three-arm ablation (314a-OFF
  / 314b-OFF / 314c-OFF + all-on baseline) on V3-EXQ-543b/c successors AFTER
  MECH-318 / MECH-319 absorption-check sessions complete.
  Design doc: REE_assembly/docs/architecture/mech_314_structured_curiosity_bonus.md.
  See MECH-314 (parent claim), MECH-314a / MECH-314b / MECH-314c (sub-flavours),
    ARC-065 (parent architectural commitment), MECH-313 (sibling noise-floor
    under same parent; substrate-landed earlier the same day),
    MECH-104 (LC-NE phasic complement; substrate-landed),
    MECH-260 (anti-monostrategy related claim; Q-045 collapse falsifier),
    MECH-295 (sibling score-bias contributor composed before MECH-314 in chain),
    Q-043 (relative weight calibration MECH-313 vs MECH-314 -- parametric sweep),
    Q-044 (MECH-314a/b/c sub-flavour independence -- three-arm ablation),
    Q-045 (MECH-313 vs MECH-260 collapse falsifier),
    MECH-318 / MECH-319 (sibling ARC-065 / ARC-064 substrates; separate spawned tasks),
    MECH-094 (simulation_mode argument; call-site scoping for waking-only effects).
