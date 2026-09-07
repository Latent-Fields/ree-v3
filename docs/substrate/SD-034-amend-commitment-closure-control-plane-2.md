## SD-034 AMEND: commitment-closure-control-plane BETA-ENGAGEMENT (couple closure->beta elevation) (2026-06-17)
- commitment-closure-control-plane beta-engagement amend -- IMPLEMENTED 2026-06-17
  (substrate; SD-034/MECH-260/MECH-261 stay provisional/candidate + non_contributory +
  pending_retest_after_substrate -- PROMOTES NOTHING until the post-amend V3-EXQ-460f
  returns a contributory PASS). Mechanism (a) of the beta-engagement deliverable
  (user-confirmed (a)-only; (b)/(c) deferred). Routed by the confirmed
  failure_autopsy_V3-EXQ-460e_2026-06-17.
  ROOT CAUSE (code-confirmed, agent.py:5855-5868 bistable elevate block): the 460e config
  sets only cfg.heartbeat.beta_gate_bistable=True (both MECH-090 R-c gates OFF), so the
  bistable BetaGate elevates iff result.committed (running_variance < commit_threshold) --
  a decisive NATURAL commit-entry that fires on only 1/3 seeds on the 603n foraging
  substrate. But total_committed_steps counts e3._committed_trajectory is not None, which
  the Leg-A env-completion hook / closure control-plane populates INDEPENDENTLY -> the
  commit-without-beta dissociation (seeds 42/43 committed_steps 2415/2019 but
  total_beta_elevated=0; closures fire 7/6; seed 44 engages beta both arms -> C2 PASS, the
  existence proof). The Leg-C trained rule_bias_head biases per-candidate SCORING, not the
  running_variance that gates commit-ENTRY, so it cannot rescue engagement; the de-commit
  DV (ON<OFF latch occupancy) has nothing to measure when beta never elevates.
  THE FIX (no-op-default; bit-identical OFF):
    Config: REEConfig.use_closure_commit_beta_coupling (bool, default False) + from_dims
      passthrough (ree_core/utils/config.py). Mirrors the use_closure_env_completion_hook /
      closure_decommit_hold_ticks closure-plane flag precedent.
    Agent (ree_core/agent.py, bistable elevate block ~5855): compute
      _closure_commit_active = (flag AND self.e3._committed_trajectory is not None) and
      _commit_for_beta = bool(result.committed) OR _closure_commit_active; the elevate
      condition uses _commit_for_beta instead of result.committed, KEEPING the full
      should_admit_elevation(score_margin, K) AND _readiness_admits conjunction (so the
      coupling composes with MECH-090 when those gates are on; both are permissively True
      on the coupled path when result.committed is False, which is the 460e gates-off case).
      So beta occupancy tracks the closure-plane commitment on every seed where one forms,
      and the Leg-B de-commit refractory then produces a measurable ON<OFF latch-occupancy
      drop (the 460f DV). After elevate(), when _closure_commit_active and not
      result.committed, calls beta_gate.note_closure_coupled_elevation().
    BetaGate (ree_core/heartbeat/beta_gate.py): note_closure_coupled_elevation() +
      _n_closure_coupled_elevations counter (pure diagnostic; does not change gate state) +
      sd034_n_closure_coupled_elevations in get_state() + per-episode reset(). The 460f
      non-vacuity readout (the coupling, not the fragile natural commit-entry, engaged the
      latch).
  Backward compatible: use_closure_commit_beta_coupling=False by default -> _commit_for_beta
    == result.committed -> the bistable block is bit-identical; the diagnostic counter stays
    0. 5/5 new contracts (tests/contracts/test_sd034_closure_beta_coupling.py: C1 BetaGate
    primitive counter/get_state/reset / C2 config default + from_dims / C3 bit-identical-OFF
    action stream over a 12-step run / C4 coupling-ON elevates under a forced closure-plane
    commit WITHOUT a natural commit + counter>=1 / C5 coupling-OFF does NOT elevate under the
    same forced dissociation, counter==0) + 7/7 preflight + full contract suite 1084 passed
    (the 2 failures are the concurrent runner-observability session's experiment_runner.py
    ERROR-branch tests in test_runner_fail_branch_persists_result.py, outside this change set).
  Phased training: N/A (pure control-state wiring + a diagnostic counter; no learned
    parameters). MECH-094: N/A -- waking select_action control-state transition; no
    replay/memory write surface (same scope decision as the Leg-B refractory). Evidence-
    staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment uses
    the default (coupling off), so no dependent claim's measured mechanism changed. KEEP all
    evidence.
  GOVERNANCE: PROMOTES NOTHING. SD-034 stays provisional, MECH-260 candidate, MECH-261
    stable; all stay non_contributory + pending_retest_after_substrate (never fairly tested --
    the 460e readiness gate self-routed before the C2 de-commit DV ran). claims.yaml NOT
    modified (substrate-only amend; the amend record lands in substrate_queue.json
    commitment-closure-control-plane implementation_log).
  Validation experiments: V3-EXQ-460f (de-commit retest on the non-cap-pinned ON<OFF
    latch-occupancy DV) + V3-EXQ-468e (MECH-090 commit-entry conjunction under the trained
    head) -- queued TOGETHER via /queue-experiment on the amended substrate, BOTH arming
    beta_gate_bistable=True + use_closure_commit_beta_coupling=True + the Leg-A env-completion
    hook + Leg-B decommit hold + the Leg-C scaffold_train_rule_bias_head. Do NOT re-author
    460d/468d (superseded). Mechanism (b) commit_threshold adaptation stays available as an
    experiment-config lever if 460f still self-routes on beta engagement. substrate_queue
    commitment-closure-control-plane ready STAYS false until 460f scores a contributory PASS.
  Design doc: REE_assembly/docs/architecture/sd_034_governance_closure_operator.md.
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460e_2026-06-17.md.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
    (commitment-closure-control-plane).
  See SD-034 (parent + Legs A/B 2026-06-12 + Leg C 2026-06-16), MECH-090 (BetaGate -- the
    latch this couples the closure-plane commit to; the should_admit_elevation/_readiness
    conjunction is preserved), MECH-260 (No-Go), MECH-261 (mode-conditioning), the Leg-B
    de-commit refractory (the occupancy-drop actuator the coupling makes measurable), the
    Leg-C scaffold_train_rule_bias_head (per-candidate scoring; orthogonal to commit-entry
    decisiveness), V3-EXQ-460e (the FAIL this amend addresses), V3-EXQ-460f/468e (validation),
    MECH-094 (N/A).
