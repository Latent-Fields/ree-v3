## F-independent closure-plane commit-ENTRY TRAJECTORY primitive (C-STEP extension of the bool latch; the between-tick path now STEPS a closure-formed committed program, not repeats _last_action) (2026-06-23)
- control_plane.closure_commit_entry_trajectory -- IMPLEMENTED 2026-06-23 (substrate;
  PROMOTES NOTHING; claims.yaml carries an implementation_note at most; MECH-445/446 stay
  candidate / standard / v3_pending / pending_retest_after_substrate). The C-STEP extension of
  the same-day bool-latch commit-ENTRY primitive (use_closure_commit_entry, ree-v3 84c1e7c).
  Routed via /implement-substrate (commitment_closure:GAP-4; rung-6 of
  f_dominance_conversion_ceiling) on top of the bool latch, after the bool-vs-trajectory
  buildability rethink (user-confirmed "build both": queue a readiness test for the bool latch +
  build the corrected trajectory latch the evidence can feed into).
  ROOT CAUSE (the residual one link past the bool latch): the bool latch e3._closure_committed_active
  arms + SUSTAINS the closure-formed beta occupancy (C-KEY -- the latch-hold yield/persistence
  check is union-aware, so the hold re-asserts beta), but a bare BOOL cannot be STEPPED. The
  between-E3-tick path (agent.py select_action) reads e3._committed_trajectory to advance a
  committed PROGRAM; on the closure-exclusive de-commit eval _committed_trajectory stays None, so
  a closure-armed hold falls through to `action = self._last_action` -- it holds beta but executes
  NO closure-formed committed program (the C-STEP gap). A bool fundamentally cannot fill the
  stepping site no matter how it is widened -- closing C-STEP REQUIRES a parallel trajectory the
  de-commit machinery consults.
  THE FIX (no-op default; bit-identical OFF; rides the bool flag):
    Module: ree_core/predictors/e3_selector.py (new field e3._closure_committed_trajectory +
      union in get_commitment_state is_committed telemetry), ree_core/agent.py (SET site +
      _closure_commit_active arm-gate union + _ncl_commit_present latch-hold-persistence union +
      the between-tick stepping union + the de-commit / closure-fire / reset clears +
      get_state is_committed telemetry + the precondition), ree_core/utils/config.py (1 new
      no-op-default flag + from_dims).
    A new sub-flag use_closure_commit_entry_trajectory (default False; PRECONDITION: requires
      use_closure_commit_entry, the bool latch it extends -- loud ValueError otherwise, the
      MECH-269b/SD-058 pattern). When on, REEAgent.select_action (at the SAME Option-A SET
      predicate where the bool is set: goal_state.is_active() AND a trajectory selected toward it
      AND lateral_pfc rule_state norm >= floor) ALSO installs the goal/rule-directed
      result.selected_trajectory into a PARALLEL STICKY latch e3._closure_committed_trajectory
      (resetting _committed_step_idx on a FRESH arm so stepping starts at the program head;
      subsequent E3 ticks refresh the trajectory but the counter keeps advancing across the held
      occupancy, mirroring the F-commit stepping which resets the counter only on beta release /
      reset). Unlike _committed_trajectory it is NOT torn down by post_action_update -- it
      persists across ticks until a principled closure teardown.
    Three UNION sites now read (_committed_trajectory OR _closure_committed_trajectory) -- all
      bit-identical when the trajectory latch is None (flag off):
      (1) agent.py _closure_commit_active arm gate -- a closure-FORMED trajectory also arms the
          coupling (alongside the legacy F-commit + the bool latch);
      (2) agent.py _ncl_commit_present latch-hold persistence -- a closure trajectory keeps the
          hold present so it re-asserts beta instead of yielding;
      (3) agent.py between-tick stepping (`_step_traj = _committed_trajectory or
          _closure_committed_trajectory`) -- the closure-armed hold ADVANCES the closure-formed
          committed PROGRAM (C-STEP) instead of repeating _last_action.
    CLEARED at the SAME sites as the bool latch (mirrored exactly): the three agent de-commit
      sites, the SD-034 auto-closure-fire teardown, and agent.reset() (episode boundary). NOT
      cleared at the e3_selector post_action_update teardown -- the closure latch must survive
      post_action_update (stickiness), same as the bool.
    is_committed telemetry (e3_selector.get_commitment_state + agent.get_state) widened with
      `or _closure_committed_trajectory is not None` so the closure-formed commit reads honestly
      on the closure-exclusive eval where _committed_trajectory stays None. The residue-write /
      hippocampal-record sites are LEFT UNCHANGED (the design wants occupancy + stepping, not
      memory recording of closure-formed commits -- matching the bool latch's scope).
  Config (REEConfig + from_dims, no-op default -> bit-identical OFF):
    use_closure_commit_entry_trajectory (False). PRECONDITION: requires use_closure_commit_entry
    (which itself requires use_closure_commit_beta_coupling + use_natural_commit_latch_hold).
  Backward compatible: use_closure_commit_entry_trajectory=False by default ->
    e3._closure_committed_trajectory is never installed -> every union reduces to the bool-latch
    behaviour (use_closure_commit_entry-only) -> bit-identical (the trajectory flag OFF reproduces
    A's 84c1e7c path byte-for-byte; verified by contract). preflight 8/8 + the closure/latch/beta-
    gate cluster (44, incl A's 6 bool-latch contracts + my 6 trajectory contracts) PASS; full
    contract suite 1253 passed (the 3 failures -- control_vector C4 + 2 runner_fail_branch -- are
    the documented pre-existing flakes, CONFIRMED failing identically with the 3 core edits
    stashed; control_vector C4 is the order-dependent flake that passes in isolation). New
    contracts: tests/contracts/test_closure_commit_entry_trajectory.py (C-KEY the F-independent
    TRAJECTORY latch installs + the hold sustains beta with ZERO F-commits / C-STEP LOAD-BEARING:
    the between-tick path STEPS the closure trajectory [proven with a trajectory whose per-step
    actions differ -- t=1 class 1, not the repeated _last_action t=0 class 0] / C-YIELD the hold
    still yields to the SD-034 de-commit refractory / C-OFF default-OFF bit-identical to the bool
    latch + reset clears the trajectory latch + the precondition raises).
  Phased training: N/A (control-state wiring; no learned parameters). MECH-094: the SET is a
    waking control-state transition (select_action waking-only by call-site scoping); no
    replay/memory write surface. Evidence-staleness (Step 8.5): NOT triggered -- no-op-default
    flag; every existing experiment uses the default (trajectory off), so no dependent claim's
    measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-445/446 stay candidate / standard / v3_pending /
    pending_retest_after_substrate; SD-034 / MECH-090 / MECH-342 / MECH-439 / MECH-448 untouched.
    claims.yaml carries an implementation_note at most (DEFERRED -- the governance-cycle session
    holds claims.yaml + substrate_queue at build time).
  Validation experiment: V3-EXQ-460m claim-free substrate-readiness diagnostic queued via
    /queue-experiment -- on the closure-exclusive de-commit eval substrate, an ARM_BOOL_LATCH
    (use_closure_commit_entry, no trajectory) vs an ARM_TRAJECTORY (+ use_closure_commit_entry_-
    trajectory) vs an ARM_ENTRY_OFF control, measuring (a) the closure-formed occupancy sustains
    with ZERO F-commits (ncl_hold_closure_armed_count>0, max consecutive beta run >> 1) on >=2/3
    seeds for BOTH latch arms where ARM_ENTRY_OFF shows 0 (the 460k/460l signature), and (b) the
    bool-vs-trajectory comparison (does the trajectory arm execute a stepped committed program the
    bool arm cannot) so the evidence informs which latch the de-commit falsifier uses. The
    460-lineage de-commit SUCCESSOR (NEW letter, supersedes the parked 460k/460l line) is NOT
    queued here -- blocked on this landing + the readiness result.
  Design doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md
    (closure-plane commit-ENTRY TRAJECTORY / C-STEP extension section). Autopsies:
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460k_2026-06-22.{md,json} +
    failure_autopsy_V3-EXQ-460l_2026-06-23.{md,json}.
  See the F-independent closure-plane commit-ENTRY primitive (the bool latch this extends; landed
    84c1e7c earlier 2026-06-23) + the closure-exclusive de-commit eval mode + natural-commit
    LATCH-HOLD amend + natural_commit_occupancy_release (the rung-6 cluster this completes),
    SD-034 / MECH-090 / MECH-445 / MECH-446 (the commitment-closure-control-plane cluster; all
    candidate, unweakened), SD-033a lateral_pfc rule_state (the "rule is being followed" source),
    MECH-091 (genuine-threat interrupt the hold never overrides), MECH-439 / MECH-448 (F-dominance
    front; this is the duration/de-commit face), V3-EXQ-460k/460l (the FAILs the cluster
    addresses), V3-EXQ-460m (validation), MECH-094 (N/A).
