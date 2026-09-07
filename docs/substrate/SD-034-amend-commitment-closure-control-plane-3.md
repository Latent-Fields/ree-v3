## SD-034 AMEND: commitment-closure-control-plane DE-COMMIT-AUTHORITY MAGNITUDE (committed-run-scaled Leg-B refractory) (2026-06-19)
- commitment-closure-control-plane de-commit-authority magnitude amend -- IMPLEMENTED
  2026-06-19 (substrate; SD-034/MECH-260/MECH-261 stay provisional/candidate/stable +
  non_contributory + pending_retest_after_substrate -- PROMOTES NOTHING until the
  post-amend V3-EXQ-460g returns a contributory PASS). Part (a) of the de-commit-magnitude
  deliverable; part (b) (within-arm C2 DV + sd034_n_closure_coupled_elevations>0 non-vacuity
  gate) is experiment-side in the V3-EXQ-460g re-issue. Routed by the confirmed
  failure_autopsy_V3-EXQ-460f_2026-06-18 (user-adjudicated 2026-06-18T08:04Z governance
  cycle; substrate_queue commitment-closure-control-plane implementation_hint 460f amend).
  ROOT CAUSE (460f, the residual one link past beta-engagement): the 2026-06-17
  beta-engagement amend WORKED -- all 4 readiness gates cleared and the C2 de-commit
  occupancy-drop DV ran for the first time (PASS seed 42: ON 23.73 < OFF 35.67, -33.5%;
  FAIL 2/3) -- but on strong-natural-commit seeds the closure->beta coupling was INERT
  (sd034_n_closure_coupled_elevations 36/52 seed 42 vs 0/0 seeds 43/44), so the DV reduced
  to the bare Leg-B 5-tick refractory whose magnitude (~20-35 tick-blocks) is SWAMPED by the
  ~530-560 natural-commit elevated steps. NOT a falsification (seed 42 + 460e seed 44 are
  existence proofs of the correct de-commit SIGN); the gap is de-commit-authority MAGNITUDE.
  THE FIX (no-op-default; bit-identical OFF): scale the Leg-B de-commit refractory installed
  at a closure fire by the COMMITTED-RUN LENGTH captured from the BetaGate BEFORE the
  closure's own release(), so a long committed run -- the exact source of the swamping latch
  occupancy -- triggers a proportionally long post-closure hold:
    n = closure_decommit_hold_ticks + round(closure_decommit_hold_scale_with_run * run_length),
    clamped to closure_decommit_hold_max_ticks (0 = uncapped).
  Modules:
    ree_core/heartbeat/beta_gate.py -- BetaGate gains a per-run _committed_run_length counter
      + committed_run_length property + sd034_committed_run_length get_state key. Incremented
      once per propagate() tick while elevated; reset to 0 on a FRESH elevate()
      (not-elevated -> elevated transition; a re-elevate while already elevated leaves it
      unchanged) and on release(); cleared in reset(). Pure bookkeeping -- never read unless
      the lever is armed, so bit-identical when off.
    ree_core/governance/closure_operator.py -- ClosureOperatorConfig gains
      decommit_hold_scale_with_run (0.0) + decommit_hold_max_ticks (0). _fire() captures
      run_length_at_fire = beta_gate.committed_run_length BEFORE step (a) release() (which
      resets it), then step (a.2) installs the scaled hold. With scale 0.0 (default),
      hold_ticks == decommit_hold_ticks unchanged -> bit-identical to the fixed-hold path
      (and with decommit_hold_ticks 0 too, no refractory at all).
    ree_core/utils/config.py -- REEConfig.closure_decommit_hold_scale_with_run (0.0) +
      closure_decommit_hold_max_ticks (0) + from_dims signature + assignment.
    ree_core/agent.py -- the ClosureOperatorConfig build site forwards both via getattr
      fallback (absent flat attr -> bit-identical), mirroring the closure_decommit_hold_ticks
      precedent.
  NOTE on the autopsy's "EITHER...OR" (committed-run-scaled refractory OR active
  MECH-342-style release-pressure): the closure _fire() ALREADY calls beta_gate.release()
  (drops the latch at the fire), so option B's "drive the latch DOWN rather than block
  re-entry" distinction is moot for this gap -- the latch is already down at fire; the only
  lever is HOW LONG to keep it down, which is exactly the refractory. So the committed-run-
  scaled refractory is the faithful magnitude lever; an active release-pressure event would
  duplicate MECH-342 with no distinct mechanism here (user-confirmed A-only, AskUserQuestion
  2026-06-19).
  Backward compatible: closure_decommit_hold_scale_with_run=0.0 by default -> the refractory
    uses closure_decommit_hold_ticks exactly as the 2026-06-12 Leg-B landing -> bit-identical;
    the run-length counter increments but is never read. 6 new contracts in
    tests/contracts/test_sd034_decommit_magnitude.py (C1 committed_run_length counter
    increments-while-elevated / resets-on-release / resets-on-fresh-elevate-not-re-elevate /
    reset() clears + get_state key; C2 scale 0.0 bit-identical to the fixed hold independent
    of a 40-tick run; C3 scale>0 -> n = base + round(scale*run_length) captured before
    release + longer run -> longer hold; C4 max_ticks clamp; C5 from_dims surfaces both flags
    + agent forwards them; C6 agent action stream bit-identical default vs explicit scale=0.0
    over an 8-step loop) + 7/7 preflight + full contract suite 1101 passed (the 3 failures --
    control_vector C4 + 2 runner_fail_branch -- are the documented pre-existing flakes,
    CONFIRMED failing identically on a clean stash, outside this change set). Activation smoke
    2026-06-19 (the 460f scenario, 530-step committed run then a closure fire): FIXED
    base5/scale0 -> 5 ticks suppressed (swamped); SCALED base5/scale0.1 -> 58 ticks (capped 60
    -> 58) -- the de-commit authority now scales with the latch occupancy it must overcome.
  Phased training: N/A (control-state counter + scalar arithmetic; no learned parameters).
    MECH-094: N/A -- waking select_action control-state transition; no replay/memory write
    surface (same scope as the Leg-B refractory the lever extends; the run-length counter only
    advances on the waking propagate() path). Evidence-staleness (Step 8.5): NOT triggered --
    no-op-default lever; every existing experiment uses the default (scale 0.0), so no
    dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. SD-034 stays provisional, MECH-260 candidate, MECH-261 stable;
    all stay non_contributory + pending_retest_after_substrate. claims.yaml NOT modified
    (substrate-only amend; the amend record lands in substrate_queue.json
    commitment-closure-control-plane implementation_log).
  Validation experiment: V3-EXQ-460g (supersedes V3-EXQ-460f), queued via /queue-experiment --
    the de-commit retest arming the magnitude lever (closure_decommit_hold_scale_with_run +
    max_ticks) ON TOP of the beta-engagement-amended substrate (beta_gate_bistable +
    use_closure_commit_beta_coupling + Leg-A env-completion hook + Leg-B hold + Leg-C
    scaffold_train_rule_bias_head), with (b) the C2 DV redesigned to a WITHIN-ARM
    around-closure occupancy delta (pre-vs-post-closure window on the ON arm) and the
    non-vacuity gate tightened to sd034_n_closure_coupled_elevations > 0 on scored seeds.
    Acceptance: ON<OFF de-commit on the within-arm non-cap-pinned statistic on >=2/3 seeds
    with the coupling non-vacuity gate met. Do NOT re-author 460d/460e/460f; the parallel
    V3-EXQ-468e (MECH-090 commit-entry conjunction) leg is separately owed. substrate_queue
    commitment-closure-control-plane ready STAYS false until 460g scores a contributory PASS.
  Design doc: REE_assembly/docs/architecture/sd_034_governance_closure_operator.md
    (de-commit-authority magnitude amend section). Autopsy:
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460f_2026-06-18.md.
    Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
    (commitment-closure-control-plane).
  See SD-034 (parent + Legs A/B 2026-06-12 + Leg C 2026-06-16 + beta-engagement 2026-06-17),
    MECH-090 (BetaGate -- the latch the refractory holds down; the run-length counter rides
    its elevate/propagate/release lifecycle), MECH-342 (the active-release-pressure sibling
    the autopsy named as the alternative; redundant here since _fire already releases),
    MECH-260 (No-Go), MECH-261 (mode-conditioning), the Leg-B de-commit refractory (the
    actuator this lever scales), the closure->beta coupling (the engagement the de-commit DV
    rides), V3-EXQ-460f (the FAIL this amend addresses), V3-EXQ-460g (validation),
    V3-EXQ-468e (separate MECH-090 leg), MECH-094 (N/A).
