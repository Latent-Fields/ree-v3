## SD-034 AMEND: commitment-closure-control-plane REFRACTORY-INDEPENDENT coupling certifier (decouple the de-commit lever from its non-vacuity metric) (2026-06-19)
- commitment-closure-control-plane refractory-independent coupling-certifier amend --
  IMPLEMENTED 2026-06-19 (substrate; MECH-445/446 stay candidate/v3_pending/
  pending_retest_after_substrate -- PROMOTES NOTHING; the SD-034 closure cluster was
  decomposed into MECH-445 coupling-engagement + MECH-446 de-commit-magnitude the same
  day). Routed by the confirmed failure_autopsy_V3-EXQ-460g_2026-06-19
  recommended_substrate_queue_entry (SECONDARY action; the PRIMARY /claim-synthesis
  decomposition is its precondition).
  ROOT CAUSE (code-confirmed, the S5 self-defeating entanglement): the 460f-prescribed
  coupling non-vacuity gate keys on sd034_n_closure_coupled_elevations, which counts only
  closure-coupled beta elevations the gate ENTERS -- but note_closure_coupled_elevation
  is called INSIDE the bistable elevate if-block guarded by `not beta_gate.is_elevated`
  (agent.py:6082-6094), so once the closure-coupled commit latches beta elevated for the
  long committed run (~530-560 steps) the per-ENTRY counter is frozen, and the 460g
  committed-run-scaled de-commit-MAGNITUDE lever (apply_refractory cap 60) blocks
  re-elevation so it cannot re-fire as a transition. Net: scaling the de-commit authority
  UP suppresses its own coupling certifier (the counter collapsed 36 -> 0 on seed 42)
  even though the de-commit DID act (seed-42 within-arm occupancy 0.333 -> 0.0, C2 PASS).
  The two 460f amends (magnitude lever + coupling gate) are mutually self-defeating.
  THE FIX (no-op-default; bit-identical OFF; rides use_closure_commit_beta_coupling):
    ree_core/heartbeat/beta_gate.py -- BetaGate gains _n_closure_commit_intent +
      note_closure_commit_intent() + sd034_n_closure_commit_intent in get_state() +
      per-episode reset(). Pure diagnostic; does not change gate state.
    ree_core/agent.py -- the bistable elevate block calls note_closure_commit_intent()
      when `_closure_commit_active and not result.committed` BEFORE the elevate/refractory
      gate (before the `not is_elevated` + should_admit + readiness conjunction), so the
      closure-plane commit INTENT is certified every E3 tick a closure-coupled commitment
      forms without a natural running_variance crossing -- REGARDLESS of whether the latch
      is held elevated OR the de-commit-magnitude refractory then blocks the elevate. The
      existing sd034_n_closure_coupled_elevations stays (it now measures refractory-/latch-
      surviving elevations); the new counter is the refractory-INDEPENDENT MECH-445
      coupling-engagement certifier the magnitude lever (MECH-446) cannot zero.
  Backward compatible: note_closure_commit_intent only increments when
    use_closure_commit_beta_coupling is on (_closure_commit_active stays False otherwise),
    AND it is a pure int readout with no gate-state effect, so the action stream is
    bit-identical both with the coupling flag OFF (every existing experiment) and ON.
    20/20 SD-034 closure contracts (17 prior + 3 new C6/C7/C8 in
    tests/contracts/test_sd034_closure_beta_coupling.py: C6 primitive counter advances
    under an active refractory + get_state + reset; C7 the load-bearing 460h property --
    a held/blocked elevate gate freezes the coupled counter at 0 while the intent counter
    keeps certifying coupling; C8 coupling-OFF intent counter stays 0) + 8/8 preflight +
    full contract suite 1148 passed (the 3 fails -- control_vector C4 + 2 runner_fail_branch
    -- are the documented pre-existing flakes, untouched by this change). Activation smoke
    2026-06-19 (full amended substrate beta_gate_bistable + coupling + de-commit hold +
    magnitude lever, pinned refractory): sd034_n_closure_commit_intent populates and is
    > 0 where the elevate gate is suppressed.
  Phased training: N/A (control-state counter; no learned parameters). MECH-094: N/A --
    waking select_action control-state readout; no replay/memory write surface.
    Evidence-staleness (Step 8.5): NOT triggered -- no-op-default readout; every existing
    experiment uses the default (coupling off) and the counter has no behavioural effect,
    so no dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-445/446 stay candidate / v3_pending /
    pending_retest_after_substrate; SD-034 umbrella narrowed (2026-06-19 decomposition).
    claims.yaml NOT modified (substrate-only amend; the amend record lands in
    substrate_queue.json commitment-closure-control-plane implementation_log).
  Validation experiment: V3-EXQ-460h (supersedes the de-commit lineage; do NOT re-author
    460d/e/f/g), queued via /queue-experiment -- the de-commit retest arming the full
    amended substrate (beta_gate_bistable + use_closure_commit_beta_coupling + Leg-A
    env-completion hook + Leg-B committed-run-scaled refractory magnitude lever + Leg-C
    scaffold_train_rule_bias_head), keeping the 460g within-ARM around-closure
    occupancy-delta C2 DV but gating non-vacuity on the NEW sd034_n_closure_commit_intent
    > 0 (NOT sd034_n_closure_coupled_elevations). Acceptance: closure-coupled commit-intent
    > 0 on >= 2/3 scored seeds (MECH-445 precondition) AND ON within-arm post-closure
    occupancy < pre-closure by >= DECOMMIT_MIN_DROP_FRAC on >= 2/3 seeds (MECH-446 scored);
    the five readiness gates self-route substrate_not_ready_requeue when unmet (never a
    false weakens). MECH-261 stays non_contributory unless the automatic mode-conditioned
    detector fires (n_automatic_fires > 0 -- the Leg-A hook bypasses mode-conditioning).
    claim_ids=[MECH-446] (scored) + MECH-445 (coupling-engagement non-vacuity precondition).
    substrate_queue commitment-closure-control-plane ready STAYS false until 460h scores a
    contributory PASS.
  Design doc: REE_assembly/docs/architecture/sd_034_governance_closure_operator.md
    (refractory-independent commit-intent counter amend section). Autopsy:
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460g_2026-06-19.{md,json}.
    Decomposition: REE_assembly/evidence/planning/claim_synthesis_SD-034-closure_2026-06-19.md.
  See SD-034 (parent + Legs A/B 2026-06-12 + Leg C 2026-06-16 + beta-engagement 2026-06-17
    + de-commit-magnitude 2026-06-19), MECH-445 (closure->beta coupling engagement -- the
    child this counter certifies), MECH-446 (de-commit-authority magnitude -- the scored
    child the lever serves; cannot zero this certifier now), MECH-090 (BetaGate -- the latch
    whose held-elevated / refractory-blocked gate suppressed the old counter),
    sd034_n_closure_coupled_elevations (the 460f counter now measuring refractory-surviving
    elevations only), V3-EXQ-460g (the FAIL this amend addresses), V3-EXQ-460h (validation),
    MECH-261 (mode-conditioning; protect the stable claim), MECH-094 (N/A).
