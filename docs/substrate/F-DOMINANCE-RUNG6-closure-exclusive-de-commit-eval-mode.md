## Closure-exclusive de-commit eval mode (rung-6 BUILD of f_dominance_conversion_ceiling; the named dissociable substrate from V3-EXQ-460j) (2026-06-22)
- control_plane.closure_exclusive_decommit_eval -- IMPLEMENTED 2026-06-22 (substrate;
  PROMOTES NOTHING; claims.yaml NOT modified; MECH-445/446 stay candidate / standard /
  v3_pending / pending_retest_after_substrate). The named awaited substrate the rung-6
  natural-commit-occupancy-release lever was PARKED on 2026-06-22 (failure_autopsy_V3-EXQ-460j,
  user-adjudicated "Park + amend, name the substrate"). Routed via /implement-substrate
  commitment_closure:GAP-4; the mandatory pre-build rethink re-confirmed buildable
  (user-confirmed the closure-exclusive design fork: suppress the natural path).
  ROOT CAUSE (re-confirmed in code): the natural-commit latch-hold (2026-06-21 amend)
  arms ONLY on a decisive natural commit (result.committed), which does not form on the
  full closure-coupling substrate (460j ARM_LEVER_OFF 3/3: ncl_hold_reassert_total=0,
  max_consecutive_beta_run=1, sd034_n_closure_commit_intent=0). The SD-034 closure
  de-commit control-plane fragments the latch to ~1-tick blips even with the rung-6
  release OFF, so natural-commit and closure-de-commit are NON-DISSOCIABLE: no sustained
  natural-commit occupancy exists for the de-commit to act on, and no fair test of
  MECH-445 (commit-intent) / MECH-446 (occupancy-drop) is reachable. A plain yield-clause
  patch (a "460k") was REFUSED -- it targets the release/yield logic, the WRONG cause; the
  actual cause is the ARM SOURCE of the occupancy.
  THE FIX (no-op default; bit-identical OFF): a closure-exclusive de-commit eval mode that
  re-points the occupancy's arm source onto the closure plane.
    Module: ree_core/agent.py (bistable elevate block + __init__ precondition + a
      _ncl_hold_closure_armed_count readout), ree_core/utils/config.py (1 no-op-default
      flag + from_dims). NO new module; reuses the existing latch-hold + re-assertion +
      yield + BetaGate.committed_run_length (ARC-106 G2 reuse-before-duplicate).
    (a) Closure-EXCLUSIVE elevation: when closure_exclusive_decommit_eval is set,
      _commit_for_beta is driven ONLY by _closure_commit_active (the closure->beta
      coupling) -- the fragile F-driven result.committed path is SUPPRESSED from beta
      elevation. So the beta occupancy is provably closure-formed (the literal "beta
      elevates ONLY via _closure_commit_active during the eval").
    (b) Closure-coupled hold-arm: the natural-commit latch-hold ARMS on
      _closure_commit_active (a closure-plane commitment forming) in addition to
      result.committed, GUARDED by beta_gate.refractory_remaining == 0 so it does NOT
      re-arm while an SD-034 closure de-commit is actively holding beta down (the hold
      yields to the de-commit, preserving the MECH-446 occupancy-drop DV). The closure
      commitment persists across ticks (unlike the fragile result.committed), so the
      hold's existing re-assertion sustains the occupancy reliably on all seeds.
  DISSOCIATION (the whole point): a closure commitment forms -> the hold arms + sustains a
  beta occupancy -> the SD-034 closure FIRES -> release() + refractory -> the hold yields
  -> the occupancy drops -> the refractory expires -> the next closure commit re-arms.
  Occupancy FORMATION (closure-coupled latch-hold) is now dissociable from closure
  DE-COMMIT (the SD-034 refractory), making MECH-445 commit-intent + MECH-446
  occupancy-drop co-measurable on the same seed -- dissolving the 460h disjoint-certifier
  problem. The fragile, seed-dependent F-driven natural commit is removed from the
  occupancy entirely.
  PRECONDITIONS (loud ValueError at REEAgent.__init__): closure_exclusive_decommit_eval=True
  requires use_closure_commit_beta_coupling=True AND use_natural_commit_latch_hold=True
  (the eval has no path to form / sustain the occupancy otherwise) -- the MECH-269b/293
  precondition pattern.
  Backward compatible: closure_exclusive_decommit_eval=False by default ->
  _commit_for_beta is the legacy result.committed OR _closure_commit_active, the hold arms
  only on result.committed -> bit-identical (verified: default == explicit-OFF action
  stream). 7/7 new contracts in tests/contracts/test_closure_exclusive_decommit_eval.py
  (C1 config defaults / C2 preconditions raise / C3 closure-coupled commit arms the hold
  under eval [LOAD-BEARING] + legacy does not / C4 natural-commit suppressed under eval /
  C5 yield to the closure refractory preserved / C6 bit-identical OFF) + 8/8 preflight +
  31/31 sibling closure/latch/beta-gate contracts PASS. V3-EXQ-460j --dry-run reproduces
  its eval-OFF baseline (off_occ~4, reassert=0, sustained_hold=False); activation: eval ON
  arms+re-asserts (closure_armed=1, reassert=8) where eval OFF stays 0 (the 460j signature).
  Phased training: N/A (control-state wiring; no learned parameters). MECH-094: N/A (waking
  select_action control-state transition; no replay/memory write surface). Evidence-staleness
  (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment uses the default
  (eval off), so no dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-445/446 stay candidate / standard / v3_pending /
  pending_retest_after_substrate; SD-034 / MECH-090 / MECH-342 / MECH-448 / MECH-439
  untouched. claims.yaml NOT modified (substrate-only build).
  Validation experiment: NOT queued here -- a 460-lineage successor (NEW letter, supersedes
  V3-EXQ-460j; queued SEPARATELY via /queue-experiment AFTER this build lands) runs in the
  closure-exclusive de-commit eval mode and gates on (a) the ARM_LEVER_OFF baseline
  sustains a natural-commit occupancy on >=2/3 seeds, THEN (b) the rung-6 release shortens
  it AND MECH-445 commit-intent + MECH-446 occupancy-drop co-occur on the same seeds.
  Design doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md
  (closure-exclusive de-commit eval mode section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460j_2026-06-21.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (f_dominance_conversion_ceiling rung-6 implementation_log + status PARKED -> BUILT).
  See natural_commit_occupancy_release (the rung-6 RELEASE this eval mode supports) +
  the natural-commit LATCH-HOLD amend above (the hold whose arm source this re-points),
  SD-034 / MECH-090 / MECH-445 / MECH-446 (the commitment-closure-control-plane cluster
  this dissociates; all candidate, unweakened), MECH-091 (genuine-threat interrupt the
  hold never overrides), MECH-439 / MECH-448 (F-dominance front; the duration face is
  PARALLEL), V3-EXQ-460j (the FAIL this addresses), MECH-094 (N/A).
