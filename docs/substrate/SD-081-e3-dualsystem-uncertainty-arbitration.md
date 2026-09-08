## SD-081: e3.dualsystem_uncertainty_arbitration -- IMPLEMENTED (2026-07-22)
- SD-081: e3.dualsystem_uncertainty_arbitration -- IMPLEMENTED 2026-07-22.
  ree_core/predictors/e3_selector.py (_arbitrate_dual_system + _get_world_states depth
  limit), ree_core/agent.py (u_habit resolution in select_action).
  Config: E3Config.use_dualsystem_arbitration (default False; set True to enable), plus
  dualsystem_arbitration_gain 4.0, dualsystem_arbitration_bias 0.0,
  dualsystem_uncertainty_ema_alpha 0.05, dualsystem_habit_depth 2.
  Data flow: familiarity(z_world) -> u_habit (fallback E1 novelty EMA) + E3
  _running_variance -> u_planned -> per-pathway EMA normalisation u/(u+ema) ->
  w = sigmoid(gain*(u_habit_n - u_planned_n) + bias) -> blend of the z-scored HABIT
  (depth-2) and PLANNED (full-horizon) score vectors -> E3 select() scores, upstream of
  raw_scores / score_bias / commit gate / argmin.
  Backward compatible: disabled by default; the block is skipped entirely (no second
  scoring pass, no familiarity query, last_arbitration stays None).
  Biological basis: Daw, Niv & Dayan 2005, Nature Neuroscience 8(12):1704-1711 (conf
  0.79) -- control is allocated to whichever controller is less uncertain; differential
  recruitment is the OUTPUT of an arbitrator, not a property of having two pathways.
  Phased training required: no (no learned parameters -- deliberately NOT a learned gate,
  which would confound "the arbitrator works" with "the gate trained"). MECH-094 N/A
  (nothing written to memory).
  THE PARAMS LIVE ON E3Config, NOT REEConfig. E3Selector.config IS the E3Config, so a
  REEConfig-level field reads as a missing attribute in the selector and defaults to
  False -- the silently-unreachable-flag hazard one level below the documented from_dims
  one. This build tripped exactly that (arbitrator wired, kwargs arriving, 45 select()
  calls, zero arbitrations, no error). Both levels are pinned by contract.
  HABIT DEPTH IS 2 AND FLOORED AT 2 -- do not "simplify" it to 1. Index 0 of the z_world
  sequence is the CURRENT state, shared by every candidate, so a depth-1 score vector has
  cross-candidate range EXACTLY 0.0 and carries no ranking. This is not hypothetical: it
  is the confirmed defect in V3-EXQ-786a's recruitment DV, whose first-step vector was
  constant (n_unique=1 over 32 candidates on every tick, measured under 786a's own
  config). Its _spearman degeneracy guard could not fire, because it tests the std of the
  RANKS and double-argsort of a constant vector is a permutation of 0..K-1. The DV
  therefore measured tie-break noise: simulated mean 1.0173 sd 0.1871 against the
  manifest's reported 1.01725 with per-layout sds 0.149-0.207. Re-adjudicating that run
  is /failure-autopsy + /governance work, NOT settled here.
  Contracts: tests/contracts/test_sd081_dualsystem_arbitration.py (defaults off, flag
  reachable through from_dims at BOTH config levels, OFF bit-identity + no state written,
  ON changes the action stream with a live paired series, weight monotone in relative
  uncertainty = MECH-477's mandatory manipulation check, depth floor blocks the 786a
  degeneracy, depth limit never leaks out of the habit pass).
  Validation experiment: V3-EXQ-811 (script
  experiments/v3_exq_811_mech477_dualsystem_arbitration_falsifier.py, lineage module
  experiments/_lib/baselines/mech477_dualsystem_arbitration.py) ran 2026-07-23 (run_id
  v3_exq_811_mech477_dualsystem_arbitration_falsifier_20260723T054309Z_v3),
  evidence_direction=non_contributory, interpretation=substrate_not_ready_requeue -- both
  arms measured an exact-zero score range against the readiness gate
  (full_score_range_non_degenerate and habit_score_range_non_degenerate both failed on
  arm_off and arm_on). A /failure-autopsy has been separately spawned to root-cause the
  discrepancy against the smoke test's non-zero ranges; this citation should be revisited
  once that lands. NOTE the OFF arm cannot be 786a as-run (degenerate DV, see above); both
  arms needed a fresh run with the depth-2 habit read.
  See MECH-477, MECH-163, ARC-071 (transfer -- DISTINCT from this allocation mechanism;
  BUILT 2026-07-22, see the entry below), ARC-007, ARC-016, MECH-112,
  REE_assembly/docs/architecture/sd_081_dualsystem_uncertainty_arbitration.md.
