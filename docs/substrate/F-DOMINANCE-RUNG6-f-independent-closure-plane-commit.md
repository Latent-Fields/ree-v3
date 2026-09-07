## F-independent closure-plane commit-ENTRY primitive (rung-6 amend; the F-INDEPENDENT arm source the closure-exclusive de-commit eval lacked; closes the 460k/460l ncl_hold_closure_armed_total=0 signature) (2026-06-23)
- control_plane.closure_commit_entry -- IMPLEMENTED 2026-06-23 (substrate;
  PROMOTES NOTHING; claims.yaml carries an implementation_note at most; MECH-445/446 stay
  candidate / standard / v3_pending / pending_retest_after_substrate). The F-INDEPENDENT
  commit-ENTRY the closure plane lacked. Routed by the confirmed
  failure_autopsy_V3-EXQ-460k_2026-06-22 + failure_autopsy_V3-EXQ-460l_2026-06-23 via
  /implement-substrate (commitment_closure:GAP-4; rung-6 of f_dominance_conversion_ceiling).
  ROOT CAUSE (code-confirmed; the autopsies already name the circularity): the 2026-06-22
  closure-exclusive de-commit eval arms the latch-hold ONLY via _closure_commit_active
  (agent.py:6365), which gates on e3._committed_trajectory is not None -- whose ONLY
  non-None writer in all of ree_core/ is e3_selector.py:1926 under `if committed:`
  (committed = self._running_variance < effective_threshold, pure variance/F). On the 460j
  substrate the F-driven natural commit never sustains (off_baseline_not_sustained, 0/3
  seeds), so _committed_trajectory is rarely non-None -> the eval rarely elevates -> the
  latch-hold rarely arms = the 460k/460l signature closure_exclusive_eval_did_not_arm_hold /
  ncl_hold_closure_armed_total=0. The design semantics ("a closure-plane commitment forming,
  F-independent") and the implementation (F-commit trajectory presence) CONTRADICT. The
  closure plane today has only a completion/RELEASE event (ClosureOperator.tick / emit_closure
  / habenula_tick); it had NO commit-ENTRY event -- a 460k-successor on the old code re-fails
  identically.
  THE FIX (Option A, user-confirmed via AskUserQuestion 2026-06-23; no-op-default;
  bit-identical OFF):
    (1) New F-INDEPENDENT latch e3._closure_committed_active (bool, init False;
      ree_core/predictors/e3_selector.py __init__). Unlike _committed_trajectory (set ONLY
      under `if committed:` and torn down every tick by post_action_update), this latch is
      STICKY across ticks until a principled closure teardown.
    (2) SET site (ree_core/agent.py REEAgent.select_action, right after e3.select returns,
      BEFORE the bistable _closure_commit_active computation), guarded by
      getattr(config, "use_closure_commit_entry", False): SET when a goal-active,
      rule-directed commitment forms -- goal_state.is_active() AND a trajectory was selected
      toward it (the hippocampal proposer is goal-seeded under an active goal) AND a rule is
      being followed (lateral_pfc.rule_state.norm() >= closure_commit_entry_rule_norm_floor,
      mirroring the SD-034 ClosureOperator's rule-stability precursor). Faithful to SD-034:
      closure governs rule-directed commitments. MECH-094: select_action is the waking path
      (simulation_mode=False, as the neighbouring tonic_vigor/closure-coupling sites assume);
      no replay/memory write surface (a waking control-state transition).
    (3) Redefinition (agent.py:6365): _closure_commit_active = use_closure_commit_beta_coupling
      AND (e3._committed_trajectory is not None OR e3._closure_committed_active) -- the UNION.
      The legacy F-commit path is preserved exactly (latch never set when the flag is off ->
      _closure_commit_active reduces to `_committed_trajectory is not None` -> bit-identical),
      and the eval can now arm WITHOUT a sustained F-commit.
    (4) CLEAR sites (ree_core/agent.py): the four existing `_committed_trajectory = None`
      de-commit sites (MECH-342 maintenance release / NaturalCommitUrgencyRelease duration
      release / MECH-353 blocked-agency abort / habenula de-commit), the SD-034 closure fire
      (ClosureOperator.tick auto-detector + the notify_env_completion emit_closure hook, when
      _fire installs the de-commit refractory + releases beta), and agent.reset() (episode
      boundary).
    (5) LATCH-HOLD persistence UNION (the one necessary yield-clause extension): the rung-6
      latch-hold re-assertion (agent.py) yields when e3._committed_trajectory is None. Since
      the F-independent latch deliberately leaves _committed_trajectory None (the closure
      plane installs NO trajectory -- the user's explicit NOTE), the persistence sub-condition
      is made UNION-aware (_ncl_commit_present = _committed_trajectory is not None OR
      _closure_committed_active) so the closure-formed occupancy sustains instead of arming
      then immediately yielding. Bit-identical OFF (latch never set -> reduces to the legacy
      `_committed_trajectory is None`). The other principled yields (closure refractory /
      MECH-091 / rung-6 lever / max-ticks) are UNCHANGED, so the SD-034 de-commit still tears
      the hold down (the MECH-446 occupancy-drop DV intact). LEFT UNCHANGED: the elevate
      conjunction, the closure-arm, the rest of the yield logic.
  Config (REEConfig + from_dims, all no-op default -> bit-identical OFF):
  use_closure_commit_entry (False, master) + closure_commit_entry_rule_norm_floor (0.01,
  the "rule is being followed" floor). PRECONDITIONS (loud ValueError at REEAgent.__init__,
  the MECH-269b/SD-058 pattern): use_closure_commit_entry=True requires
  use_closure_commit_beta_coupling=True (the coupling is the latch's consumer) AND
  use_natural_commit_latch_hold=True (the hold sustains the closure-formed occupancy).
  Backward compatible: use_closure_commit_entry=False by default -> e3._closure_committed_active
  is never set -> _closure_commit_active reduces to the legacy `_committed_trajectory is not
  None` and the persistence union reduces to the legacy `_committed_trajectory is None` ->
  bit-identical for every existing run. preflight 8/8 + the closure/latch/beta-gate/decommit
  cluster (58, incl 6 new) + the full contract suite PASS. New contracts:
  tests/contracts/test_closure_commit_entry.py (C1 config defaults + latch attr init False +
  default agent never arms / C2 preconditions raise / C3 C-KEY LOAD-BEARING: entry ON + eval
  ON + committed never True [_committed_trajectory stays None] + goal-active rule-directed ->
  the latch arms [ncl_hold_closure_armed_count > 0] AND the hold SUSTAINS beta [multi-tick
  committed_run_length + hold stays armed], while the SAME scenario with entry OFF == the
  pre-fix legacy eval arms EXACTLY 0 and beta never sustains / C4 default-OFF bit-identical
  action stream / C5 episode-reset clears the latch).
  Phased training: N/A (control-state wiring + bool flags; no learned parameters). MECH-094:
  the SET is a waking control-state transition (select_action waking-only by call-site
  scoping); no replay/memory write surface (the latch is not memory content). Evidence-
  staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment uses
  the default (entry off), so no dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-445/446 stay candidate / standard / v3_pending /
  pending_retest_after_substrate; SD-034 / MECH-090 / MECH-342 / MECH-439 / MECH-448 untouched.
  claims.yaml carries an implementation_note at most (no flag/confidence/status change).
  Validation experiment: V3-EXQ-460m claim-free substrate-readiness diagnostic (queued via
  /queue-experiment) -- on the closure-exclusive de-commit eval substrate, gates in order
  (a) ARM_LEVER_OFF baseline SUSTAINS a closure-formed occupancy on >=2/3 seeds with ZERO
  F-commits (the precondition every prior 460 run failed); (b) the rung-6 release shortens it;
  (c) MECH-445 commit-intent + MECH-446 occupancy-drop co-occur on the same seeds. A 460-lineage
  successor (NEW letter, supersedes the parked 460k/460l line) runs the full de-commit falsifier
  once (a) clears. Self-routes substrate_not_ready_requeue if (a) still fails. Do NOT re-author
  460d/e/f/g/h/i/j/k/l.
  Design doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md
  (closure-plane commit-ENTRY primitive section). Autopsies:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460k_2026-06-22.{md,json} +
  failure_autopsy_V3-EXQ-460l_2026-06-23.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (f_dominance_conversion_ceiling rung-6 implementation_log).
  See the closure-exclusive de-commit eval mode + natural-commit LATCH-HOLD amend +
  natural_commit_occupancy_release (the rung-6 cluster this completes), SD-034 / MECH-090 /
  MECH-445 / MECH-446 (the commitment-closure-control-plane cluster; all candidate, unweakened),
  SD-033a lateral_pfc rule_state (the "rule is being followed" source), MECH-091 (genuine-threat
  interrupt the hold never overrides), MECH-439 / MECH-448 (F-dominance front; this is the
  duration/de-commit face), V3-EXQ-460k/460l (the FAILs this addresses), V3-EXQ-460m (validation),
  MECH-094 (N/A).
