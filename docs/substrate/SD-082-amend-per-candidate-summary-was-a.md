## SD-082 AMEND: per-candidate summary was a shared constant, not per-candidate (the CORRUPTING defect V3-EXQ-822c confirmed) (2026-08-29)
- SD-082 candidate-summary post-action fix -- IMPLEMENTED 2026-08-29. Routed by the
  confirmed failure_autopsy_V3-EXQ-822c_2026-08-29 (Section 6, "Routing --
  implement-substrate, amend SD-082"). HEADLINE: the "structural zero"
  (on/off_prop_delta_mean == 0.0) that drove this whole lineage for a month was NEVER
  measured -- it is the empty-list default of statistics.fmean(prop_deltas) if prop_deltas
  else 0.0, n_prop_samples is 0 in ALL 18 cells across V3-EXQ-822/822a/822b, because the
  drivers' _candidate_summaries() called only agent._candidate_world_summaries(candidates),
  which returns None on the default candidate_summary_source="proposer" -- starving both
  the P1 REINFORCE buffer and the P2 propagation measurement in every prior run.
  ROOT CAUSE (now measured, V3-EXQ-822c): with the measurement gap fixed, compute_bias's
  per-candidate summary is trajectory.world_states[:, 0, :] -- but
  ree_core/predictors/e2_fast.py's rollout_with_world() seeds world_states=[initial_z_world],
  so index 0 is the rollout's SHARED initial world state, bit-identical across all K
  candidates by construction (candidates differ only in actions applied from t>=1). SD-082's
  own centering step (summaries - summaries.mean(dim=0)) then annihilates this constant to
  float32 cancellation noise: rule_summary_magnitude_ratio 2.8e6-4.5e6, ~4000x the driver's
  own 1e3 in-range ceiling, in every 822c cell, both arms. SEVERITY CORRUPTING: on the
  default config, prop_delta clears the 1e-3 non-vacuity floor (0.001662) while carrying
  ZERO candidate-discriminating information -- an authentic-looking but meaningless number,
  worse than the "0.0 vacuous" reading the prior autopsies (wrongly) took at face value.
  THE FIX (both no-op default; bit-identical OFF; did NOT change the candidate_summary_source
  DEFAULT, which the autopsy flagged as option (b) needing its own sweep -- out of scope
  here):
    ree_core/utils/config.py + ree_core/agent.py -- candidate_summary_source gains a THIRD
      value, "proposer_post_action" (default stays "proposer"). Still proposer-rollout-based
      (not "e2_world_forward", a different fix for a different problem -- see the field's own
      docstring), but the new agent._proposer_post_action_summaries() reads
      world_states[:, 1:, :].mean(0) (the POST-ACTION states, one per candidate, reflecting
      that candidate's own action sequence) instead of world_states[:, 0, :], at zero extra
      model calls -- same rollout, different read-out index. Falls back to world_states[0]
      only for a degenerate zero-horizon rollout (no post-action state exists).
      _candidate_world_summaries() dispatches to it and returns non-None, so EVERY caller's
      manual ws[0, 0, :] fallback loop (gated_policy / lateral_pfc / ofc / mech295 /
      tonic_vigor -- all consumers of the one SHARED cand_world_summaries, per the
      ARC-065 GAP-A docstring) is bypassed uniformly when opted in -- exactly as
      "e2_world_forward" already does, not a lateral_pfc-only patch.
    ree_core/pfc/lateral_pfc_analog.py -- compute_bias() gains a centering-degeneracy guard.
      New LateralPFCConfig.candidate_summary_degeneracy_floor (default 1e-4): whenever
      rule_readout_consumer centers a >=2-candidate summary, unconditionally (independent of
      capture_head_diagnostics) records candidate_summary_norm_pre_centering /
      candidate_summary_norm_post_centering and sets candidate_summary_degenerate = True when
      post_norm <= floor * pre_norm -- flags, never raises or refuses, so no existing run's
      behaviour changes. get_state() exposes all three; reset() clears them. This makes the
      exact 822c failure mode directly measurable going forward rather than only inferable
      post hoc from rule_summary_magnitude_ratio.
  Backward compatible: candidate_summary_source default stays "proposer" (every existing
    experiment, including 822/822a/822b/822c, is bit-identical); the degeneracy guard only
    ever sets new diagnostic get_state() fields, never the returned bias tensor.
  CORRECTION (2026-08-30, chip-20260829-sd082-percandidate-summary-fix verify-and-land):
    this entry originally claimed "28 new contracts in
    tests/contracts/test_sd082_candidate_summary_post_action_amend.py" below, but the file
    was never actually committed -- the build session that wrote this CLAUDE.md entry died
    (spend-limit kill) before committing it. The verify-and-land session found the gap,
    authored the missing coverage, and committed it as a separate follow-up commit on this
    same branch (22 tests, not 28 -- see that commit's own message for the exact count and
    rationale). Corrected in place below rather than left wrong.
  22 new contracts in tests/contracts/test_sd082_candidate_summary_post_action_amend.py
    (run alongside the existing 14-total test_sd082_rule_readout_consumer.py, both green):
    proposer_post_action summaries are candidate-discriminating (the regression test against
    the exact bug) vs. a baseline confirming the fixture reproduces the ws[0,0,:] constant
    shape, zero-horizon degenerate fallback, None-world_states fallback,
    dispatch-through-_candidate_world_summaries parity (including the pre-existing
    e2_world_forward branch, confirmed untouched), "proposer" default still returns None
    (backward-compat), REEConfig.from_dims reachability (+absent-default), the degeneracy
    guard fires on an exactly-constant summary and does NOT false-positive on a genuinely
    differentiated one, degeneracy fields stay at default when rule_readout_consumer is OFF
    or k<2, the guard never changes the returned bias, get_state() exposes the new fields,
    reset() clears them, the floor threshold is configurable (same fixture flagged under a
    lenient floor, not flagged under the strict default), and an end-to-end section
    confirming the actual V3-EXQ-822c failure signature: collapsed proposer summaries drive
    the cross-candidate bias RANGE to exactly 0.0 (zero discrimination, despite a nonzero
    rule-state-ablation delta -- the corrupting defect is NOT bias-is-always-zero), and
    proposer_post_action summaries restore a nonzero range.
  Phased training: unchanged from the SD-082 landing (P0/P1/P2 for the trainable head).
    MECH-094: N/A -- unchanged, pure forward read on the waking select_action path.
  Failure record: V3-EXQ-822c marked resolved (this fix); 822/822a/822b stay superseded per
    the autopsy's own routing (nothing was fixed there -- the reads were wrong).
  GOVERNANCE: PROMOTES NOTHING, DEMOTES NOTHING. SD-078 and SD-082 both stay
    candidate_substrate_landed / non_contributory / pending_retest_after_substrate=true.
    Per the autopsy: SD-082's centering mechanism is NOT falsified by this history -- it was
    UNTESTED, because its input never carried the cross-candidate variance it was designed to
    preserve.
  Validation experiment: V3-EXQ-822d PLANNED but NOT YET QUEUED (corrected 2026-08-30 --
    this entry originally said "queued"; experiment_queue.json carries no V3-EXQ-822d entry
    as of the verify-and-land landing. A /queue-experiment follow-on session is needed):
    candidate_summary_source="proposer_post_action" + rule_readout_consumer=True on both
    arms; asserts n_prop_samples > 0 as a readiness gate BEFORE trusting any prop_delta
    aggregate -- the exact measurement-starvation gap 822/822a/822b fell into. See SD-078, SD-033a, ARC-063, SD-008,
    REE_assembly/docs/architecture/sd_082_rule_selection_action_consumer.md (Amendment
    section), REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-822c_2026-08-29.md.
  Does NOT queue a new experiment. Per the autopsy's own judgment (re-derive brake did not
    formally fire -- R3 counts only substrate_ceiling readings, this is
    competence_implementation_gap -- but two consecutive identical-gate failures argue for
    a substrate amend over a third blind re-queue letter regardless), the next step is a
    follow-on consumer-validation queue-experiment session (V3-EXQ-822b or similar, with
    lateral_pfc_capture_head_diagnostics=True on the trained arm) to actually root-cause
    the dead-ReLU/insensitivity hypothesis this instrumentation now makes measurable --
    NOT performed in this session (chip, not inline, per session-land Phase 3).
  See SD-082, SD-078, SD-033a, ARC-063,
    REE_assembly/evidence/planning/failure_autopsy_batch-822a-826-817a-827_2026-07-26.md
    Section 2, REE_assembly/docs/architecture/sd_082_rule_selection_action_consumer.md.
