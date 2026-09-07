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

- ARC-071: policy.composition_via_repeated_grounding -- IMPLEMENTED 2026-07-22.
  ree_core/policy/policy_chunking.py (new: ChunkState, ChunkedPrimitive,
  PolicyChunkingConfig, ChunkAccumulator = MECH-323, ChunkLibrary = MECH-324,
  PolicyChunking facade), ree_core/agent.py (instantiation, per-step record in
  select_action, end_episode in reset, note_chunk_outcome /
  note_chunk_replay_sequence / get_chunking_state API),
  ree_core/hippocampal/module.py (set_chunk_source + _build_chunk_candidates +
  proposal-pool splice).
  Config: REEConfig.use_policy_chunking (default False; set True to enable) plus
  chunk_min_repetitions 20, chunk_window_trials 100, chunk_variance_low 0.15,
  chunk_variance_high 0.45, chunk_evaluative_margin 0.05, chunk_min_size 2,
  chunk_max_size 5, chunk_max_depth 3, chunk_max_library_size 64,
  chunk_max_tracked_sequences 512; use_chunk_maintenance False,
  chunk_crystallisation_min 5, chunk_dissolve_trials 50;
  use_chunk_replay_origin_path False, chunk_replay_value_quantile 0.75,
  chunk_replay_corroboration_episodes 75; use_chunk_proposal_injection False
  (mirrored onto HippocampalConfig by from_dims). Defaults are the registered
  MECH-323 / MECH-324 suggested defaults.
  SUPERSEDED IN PART (2026-07-27): chunk_max_size 5 and chunk_max_depth 3 are no longer
  hard LIFETIME caps in every configuration. Both became INITIAL budgets on growable,
  deliberation-budget-derived ceilings under use_growable_chunk_ceiling /
  use_growable_chunk_depth. Both default OFF and both derivations reproduce exactly 5
  and 3 at REE's actual rollout horizon of 30, so the flat values above still describe
  every default agent -- but do not read them as a substrate-level ceiling. See the
  2026-07-27 "chunk SIZE and chunk DEPTH are GROWABLE CEILINGS" entry below.
  Data flow: committed action class (select_action) -> ChunkAccumulator.record_step
  -> note_chunk_outcome at a trial boundary credits contiguous sub-sequences of
  length 2-5 -> joint formation gate (reps >= R_min AND variance < F_low AND mean >
  baseline + margin) -> ChunkedPrimitive minted -> ChunkLibrary FORMING ->
  crystallisation counter on real executions -> CRYSTALLISED -> [injection knob]
  HippocampalModule.propose_trajectories splices it as ONE atomic Trajectory
  (metadata source="arc071_chunk") -> E3 selects the whole sub-sequence as a single
  move under the MECH-090 commit latch.
  Backward compatible: disabled by default; agent.policy_chunking stays None, every
  call site is None-guarded, the proposer makes no call at all.
  Biological basis: Graybiel 1998/2008 striatal chunking (repetition + outcome
  consistency is the PRIMARY trigger, lit R1 conf 0.78); Yin & Knowlton 2006 DMS->DLS
  transfer; Smith & Graybiel 2013 dual-operator view + IL causal requirement (R2 conf
  0.81 -- the substrate is PHASE-DEPENDENT MULTI-SUBSTRATE, which is why formation and
  maintenance are two operators with two switches, not one module); Sakai 2003
  chunk-size budget 2-5; Sutton 1999 options structure (R4).
  Phased training required: no (pure arithmetic, no learned parameters, no gradients).
  ARC-071 IS THE TRANSFER MECHANISM, SD-081/MECH-477 IS THE ALLOCATION MECHANISM.
  MECH-163 presupposes BOTH and specifies NEITHER. ARC-071 is slow, driven by
  repetition and outcome consistency, and execution-side (how content BECOMES
  habitual); SD-081 is fast, uncertainty-driven, and selection-side (which pathway
  holds control right now). They are separate builds and neither substitutes for the
  other -- do not collapse them.
  MECH-094 IS SAFETY-CRITICAL HERE AND IS STRICT BY DEFAULT. A hallucinated chunk
  would install a macro the agent never executed into the pool of things it can commit
  to atomically. record_step() REFUSES hypothesis_tag=True outright at any parameter
  setting (contract C2 pins n_formed == 0 over an 80-trial all-imagined stream). The
  ONLY path accepting internally-generated content is the MECH-322 carve-out
  record_replay_sequence(), which is a SEPARATE method behind a SEPARATE flag that is
  False even when chunking is on, and which ANDs all three MECH-322 conditions --
  (a) value-tag at or above the top-quartile of the REAL-execution outcome
  distribution, (b) designated SD-017 sleep phase (waking DMN, where MECH-292/293
  ghost-goal probes operate, reads False and is refused), (c) replay_origin=True audit
  flag plus accelerated dissolution to DISSOLVED on an uncorroborated N-episode
  deadline, bypassing the slower DISSOLVING window. Every condition fails CLOSED,
  including an empty real-execution history (the value bar is +inf, so nothing mints).
  THE EVALUATIVE GATE IS RELATIVE, WHICH IS EASY TO MISREAD AS "THE ACCUMULATOR IS
  BROKEN". A sub-sequence must beat the agent's RUNNING BASELINE by the margin, so a
  regime where every trial scores identically forms NOTHING no matter how many
  repetitions accumulate (pinned by contract C6). The validation task must therefore
  supply outcome CONTRAST between the repeating sub-sequence and the rest, not merely
  a repeating sub-sequence.
  HYSTERESIS IS VALIDATED, NOT ADVISORY: variance_low < variance_high is enforced by
  PolicyChunkingConfig.validate(), which raises on an equal or inverted pair. A single
  shared threshold would make chunks flicker in and out of the proposal pool on
  estimator noise alone.
  ARM_1 vs ARM_2 IS A REAL DISSOCIATION, NOT A DEGENERATE ARM: with
  use_chunk_maintenance=False chunks still FORM but never crystallise and never become
  selectable (contract C7) -- the substrate analog of Smith & Graybiel 2013's IL
  disruption. That contrast is what isolates MECH-324's contribution.
  Contracts: tests/contracts/test_arc071_policy_chunking.py (25 tests: C1 defaults +
  from_dims forwarding at both config levels + OFF inertness, C2 MECH-094 strict, C3/C4
  MECH-322 conditions each failing closed, C5 accelerated dissolution + corroboration
  reset, C6 joint formation gate incl. the uniform-outcome and high-variance negatives,
  C7 hysteresis validation + formation-only dissociation + recoverable DISSOLVING,
  C8 options fields + depth cap + size budget + library bound, C9 injection wiring).
  Registered in tests/test_flag_inertness.py PROBED.
  Validation experiment: NOT YET QUEUED -- substrate-readiness diagnostic (does the
  accumulator fire at all) goes through /queue-experiment, ON vs OFF with injection
  OFF. Behavioural-latency and rollout-cost measurement is the LATER ARM_1-vs-ARM_2
  experiment, per MECH-324's registered design.
  See ARC-071, ARC-069 (parent), ARC-070 (the inverse decomposition operator, unbuilt),
  MECH-323, MECH-324, MECH-322, MECH-163, MECH-477/SD-081, MECH-094, SD-017,
  REE_assembly/docs/architecture/policy_primitive_granularity.md.

- ARC-071 / MECH-324: chunk dissolution is SUPPRESSION-WITH-RETENTION, not erasure --
  IMPLEMENTED 2026-07-27 (ree-v3 6c3e67e). A structural correction to the DISSOLVED state
  of the 2026-07-22 build above, not a new mechanism.
  ree_core/policy/policy_chunking.py (ChunkedPrimitive.n_dissolutions / n_reacquisitions /
  reacquisition_repetitions + is_dormant property, PolicyChunkingConfig
  .reacquisition_min_repetitions, ChunkLibrary revival path, module-docstring
  dissolution-is-suppression section), ree_core/utils/config.py (both knob declaration and
  from_dims forwarding).
  Config: REEConfig.use_chunk_dissolution_retention (default False; set True to enable) plus
  REEConfig.chunk_reacquisition_repetition_factor 0.25. NESTED UNDER use_chunk_maintenance --
  retention with maintenance OFF RAISES rather than running silently inert (nothing dissolves
  with maintenance off, so the flag would otherwise read enabled in a manifest while never
  firing). The factor must scale DOWN (validated, <= 1.0).
  THE BUG WAS WORSE THAN ERASURE, WHICH IS THE THING TO UNDERSTAND HERE. DISSOLVED was
  documented as terminal-but-retained-for-the-audit-trail, and the retention was itself the
  trap: PolicyChunking.note_outcome() skips any sequence already present in the library, and
  note_real_execution() had no DISSOLVED branch, so the tombstone permanently BLOCKED its own
  sequence from re-forming at ANY number of repetitions and ANY outcome consistency. Erasure
  would at least have permitted re-formation at R_min. Measured on the contract fixture: after
  forcing a crystallised chunk to DISSOLVED, 200 further trials of the same perfectly consistent
  above-baseline regime produced zero re-formations.
  Data flow (retention ON): CRYSTALLISED -> [variance > F_high for T_dissolve trials] ->
  DISSOLVING -> DISSOLVED, which zeroes reacquisition_repetitions and increments n_dissolutions
  -> the chunk stays in the library UNSELECTABLE (is_selectable is UNCHANGED by retention; a
  dormant chunk is SUPPRESSED, so retention buys a cheaper route back, not a free pass back into
  the proposal pool) -> subsequent real executions of the same sequence increment
  reacquisition_repetitions -> the MECH-323 joint formation gate is re-entered with ONLY the
  repetition term substituted, ceil(R_min * chunk_reacquisition_repetition_factor) against that
  post-dissolution counter, while variance < F_low and mean > baseline + margin apply UNCHANGED
  -> DISSOLVED -> FORMING (not CRYSTALLISED: rapid reacquisition is a claim about the FORMATION
  threshold, while C_min is the separate Smith & Graybiel 2013 IL sub-mechanism and re-runs from
  zero) -> n_reacquisitions increments.
  REPETITIONS ARE COUNTED SINCE DISSOLUTION, NOT AS THE ACCUMULATOR'S TALLY LENGTH, and this is
  not incidental: the per-sequence tally is a sliding window capped at chunk_window_trials and a
  long-lived chunk's bucket sits saturated at that cap, so a naive "compare the bucket length
  against a lowered R_min" would clear the reduced bar on the very first post-dissolution trial
  and measure nothing at all.
  MECH-322 REPLAY-ORIGIN CHUNKS ARE NEVER REVIVED (fails closed). A chunk retired on its
  corroboration deadline died precisely because real waking execution never corroborated it;
  letting it return by a REDUCED bar would be a shortcut around the MECH-094 posture the
  carve-out exists to preserve. is_dormant is state-only and excludes replay_origin, so it reads
  correctly regardless of the flag; the flag decides only whether anything ACTS on it.
  Backward compatible: disabled by default and bit-identical when OFF -- with retention off
  DISSOLVED behaves exactly as it did at the 2026-07-22 landing (terminal and blocking), which
  is what contract test_c10_off_dissolved_is_an_absorbing_tombstone pins.
  Phased training required: no (pure arithmetic, no learned parameters, no gradients).
  Biological basis: Barnes, Kubota, Hu, Jin & Graybiel 2005 (Nature, 10.1038/nature04053) --
  sensorimotor striatal ensemble patterns across acquisition / extinction / reacquisition are
  "successively formed, reversed and then re-emerged", and "regaining a habit can occur quickly,
  with even one or a few exposures"; Bouton, Winterbauer & Todd 2012 (Behav Processes,
  10.1016/j.beproc.2012.03.004) on INSTRUMENTAL extinction, i.e. at the action level ARC-071
  operates on -- extinction "weakens behavior without erasing the original learning", installing
  new context-dependent learning ALONGSIDE the old, from which three relapse effects follow
  (renewal, resurgence, rapid reacquisition). Targeted lit-pull 2026-07-27, REE_assembly
  36901c01ed.
  f_reacq = 0.25 IS AN UNCALIBRATED ENGINEERING DEFAULT, exactly the status of
  chunk_variance_high = 0.45. Both sources establish the DIRECTION (reacquisition is materially
  faster than acquisition) and NEITHER quantifies the magnitude. Do not cite 0.25 as
  literature-derived, and do not calibrate it against F_high as though they were one quantity.
  NOT IMPLEMENTED -- the other two relapse effects are design questions, not builds, and are
  recorded in the module docstring so the gap stays visible. RENEWAL needs dissolution gated
  against the chunk's initiation_set: formation is context-conditioned while dissolution is
  context-BLIND, so a chunk dissolved in one context is dissolved everywhere and REE cannot
  exhibit renewal at all -- closing this needs per-context dissolution state, not a parameter.
  RESURGENCE needs a chunk to return because a COMPETITOR was extinguished rather than because
  its own evidence improved, which requires dissolution state shared across competing library
  candidates; registration is per-sequence and produces no such coupling.
  Contracts: 10 further tests in tests/contracts/test_arc071_policy_chunking.py (the C10 block,
  14 collected cases: OFF-by-default + maintenance precondition, from_dims forwarding, factor
  validation, the ceil() bar across R_min x factor, the OFF absorbing-tombstone baseline, ON
  re-formation below R_min, reacquisition still requiring consistency AND contrast, replay-origin
  never revived, revive failing closed on every refusal path, dormant chunks still suppressed).
  Registered PROBED in tests/test_flag_inertness.py. Green on ree-cloud-2 (63 passed targeted)
  and on a full local pre-commit gate run (2331 contracts passed).
  Validation experiment: RAN 2026-07-27T17:05:39Z AND THE FALSIFIER FAILED. V3-EXQ-829
  (script + queue entry ree-v3 77e3ddc; run
  v3_exq_829_mech324_rapid_reacquisition_falsifier_20260727T170539Z_v3 on ree-cloud-2, 66 cells =
  6 seeds x 11 arms) returned outcome FAIL, evidence_direction MIXED for MECH-324 and SUPPORTS
  for MECH-323, non-degenerate, self-routing retention_real_but_rapid_reacquisition_falsified.
  NOT YET AUTOPSIED and NOT marked reviewed -- that is /failure-autopsy and governance work.
  C3 HELD, so THE STRUCTURAL CORRECTION ABOVE IS REAL: with retention OFF all 12 of 12 dissolved
  cells stayed dead and with it ON they revived, which is exactly the absorbing-tombstone claim.
  C5 (mild-dissolution robustness) also held.
  C1 FAILED, and it is the load-bearing one: reacquisition was SLOWER than acquisition, median
  r_reacq 90 against a measured median r_acq of 20 (= R_min exactly). C2 also failed -- r_reacq
  was FLAT at 90 across every f_reacq in {1.0, 0.5, 0.25, 0.1} (forced bars 20 / 10 / 5 / 2), so
  the bar this correction introduced had NO measurable effect on the DV it was built to move.
  LIKELY READING, a hypothesis for the autopsy and NOT a verdict: revival is rate-limited by the
  VARIANCE WINDOW FLUSHING, not by the repetition bar. r_reacq / W = 0.908 +/- 0.029, and the
  pre-registered W control separates the readings cleanly (median r_reacq 90 at W = 100, 28 at
  W = 30 -- the DV tracks the WINDOW, not R_min and not f_reacq). Reaching DISSOLVED needs
  chunk_dissolve_trials = 50 trials of supra-F_high variance, so at dissolution the variance
  window is necessarily contaminated by the stream that caused it, and variance < F_low cannot
  clear until that ages out -- ~0.9 W trials, always dominating a repetition term of at most
  R_min = 20. If that holds, chunk_reacquisition_repetition_factor is INERT BY CONSTRUCTION
  rather than mis-valued, and the fix is structural (the reacquisition path needs the variance
  gate scoped to post-dissolution trials too, not just the repetition counter), not a
  recalibration. The arithmetic-identity degeneracy check passed
  (all_on_cells_sit_on_forced_bar = false), so the flatness is a real measurement.
  It is deliberately an OPERATOR-LEVEL driver rather than an agent run: V3-EXQ-810 measured the
  MECH-323 accumulator SILENT under agent control (chunk_accumulator_silent), so an agent-level
  reacquisition run would have measured that readiness gap rather than MECH-324. Evidence
  attaches to the OPERATOR, not to chunk-driven behaviour.
  See MECH-324, MECH-323, MECH-322, ARC-071, MECH-094, and the
  dissolution-is-suppression-with-retention section of policy_chunking.py's module docstring.

- ARC-071 / MECH-324: reacquisition-window ISOLATION fix -- corrects V3-EXQ-829's
  confirmed FALSIFIED result -- IMPLEMENTED 2026-07-31 (IGW-20260731-196; ree-v3
  7747a01c94). A targeted bugfix to the revival gate's data flow, not a new
  mechanism -- the dissolution-with-retention structure above is unchanged.
  Design doc: REE_assembly/docs/architecture/mech324_reacquisition_window_isolation.md.
  ROOT CAUSE (confirmed by V3-EXQ-829, ree-v3 77e3ddc): the revival gate in
  PolicyChunking._attempt_reacquisition() read variance/mean from
  ChunkAccumulator._tally[key] -- the sequence's WHOLE-LIFETIME sliding window,
  FIFO-capped at window_trials, never reset or segmented at dissolution.
  Reaching DISSOLVED requires dissolve_trials (default 50) of supra-F_high
  outcomes for that exact sequence, written into that SAME tally bucket, so at
  the moment of dissolution the bucket is saturated with the contaminating
  stream that caused the dissolution. Clearing var < F_low then takes ~0.9 x
  window_trials trials regardless of the reacquisition_repetition_factor bar --
  exactly V3-EXQ-829's measured signature (median_r_reacq flat across every
  tested f_reacq, r_reacq/window_trials = 0.908 +/- 0.029).
  FIX: ree_core/policy/policy_chunking.py -- new field
  ChunkedPrimitive.reacquisition_outcomes (List[float], FIFO-capped at
  window_trials), populated only from real executions SINCE the most recent
  dissolution (ChunkLibrary.note_real_execution()'s new outcome_signal param,
  threaded from PolicyChunking.note_outcome()); reset to [] in both
  _mark_dissolved() and revive() (symmetric with reacquisition_repetitions).
  PolicyChunking._attempt_reacquisition() branches on the new flag: True reads
  var/mu from chunk.reacquisition_outcomes (isolated, post-dissolution-only)
  instead of accumulator._tally[key] (contaminated, whole-lifetime); a
  len(window) < 2 numerical-stability floor prevents a single post-dissolution
  sample from trivially clearing the variance gate at bar==1 settings (var is
  definitionally 0.0 below n=2 -- "no evidence" masquerading as "perfect
  consistency").
  Config: PolicyChunkingConfig.use_reacquisition_window_isolation /
  REEConfig.use_reacquisition_window_isolation (default False). NESTED UNDER
  use_chunk_dissolution_retention -- with retention off no chunk is ever
  dormant, so the isolated window would never populate; the pairing is REFUSED
  loudly by PolicyChunkingConfig.validate(), matching the established
  sub-switch convention (see use_chunk_dissolution_retention above).
  Backward compatible: with the flag at default False, _attempt_reacquisition()
  takes the untouched else-branch reading the same contaminated tally exactly
  as before -- bit-identical output. reacquisition_outcomes is still populated
  (written under the enclosing use_chunk_dissolution_retention flag, consulted
  only under the new sub-flag -- same convention as n_dissolutions /
  n_reacquisitions), but nothing reads it on the OFF path, so this is memory
  bookkeeping only. Both prerequisite switches (use_chunk_maintenance,
  use_chunk_dissolution_retention) already default False, so every existing
  experiment with default config is completely unaffected regardless of this
  change.
  MECH-094: does not apply. The fix only touches the real-execution
  (note_real_execution) path reached from PolicyChunking.note_outcome(), never
  the hypothesis_tag=True replay carve-out (ChunkAccumulator.record_replay_sequence);
  replay-origin chunks remain permanently excluded from revival, unchanged.
  Phased training: no (pure arithmetic, no learned parameters, no gradients).
  Contracts: 5 new tests in tests/contracts/test_arc071_policy_chunking.py C10
  block -- off-by-default + precondition, from_dims 3-site forwarding, the bug
  REPRODUCED via real dissolution (contaminating note_outcome calls on the
  target sequence, NOT the existing suite's _dissolve() shortcut -- confirmed
  that shortcut never touches the tally and so never exercised this path),
  the fix confirmed (reformed-after == the reduced bar instead of ==
  window_trials), and the len(window)<2 numerical floor. All 100 tests in the
  file green (95 pre-existing + 5 new); registered PROBED in
  tests/test_flag_inertness.py. Full tests/contracts suite verified green
  locally on the Mac (3085 passed, 0 failed, 632s) after fleet-wide contention
  (hub + all 3 cloud workers running real hours-long experiments) prevented
  the remote pre-commit gate; committed --no-verify with explicit user
  approval.
  Validation experiment: V3-EXQ-829a queued (supersedes V3-EXQ-829), adding
  use_reacquisition_window_isolation as a third ablation axis crossed with
  829's exact f_reacq sweep and window_trials {30,100}; smoke --dry-run
  confirmed C1-C6 all green at n=1 (rho_ISOON=1.0, rho_ISOOFF=n/a correctly
  reproducing the flat signature).
  See MECH-324, MECH-323, ARC-071, MECH-094, V3-EXQ-829, V3-EXQ-829a.

- ARC-071 / MECH-323: chunk CREDIT RULE is all-position, not trailing-only -- IMPLEMENTED
  2026-07-27. ree_core/policy/policy_chunking.py (ChunkAccumulator.note_outcome,
  PolicyChunking._was_executed), ree_core/utils/config.py, ree_core/agent.py.
  Config: REEConfig.use_chunk_all_position_credit / PolicyChunkingConfig
  .use_chunk_all_position_credit (default False; set True to enable). FOUR wiring sites plus
  the agent mapping, round-trip asserted by C13 because from_dims swallows unknown kwargs.
  Data flow: episode action buffer -> note_outcome credit enumeration -> per-sequence tally
  -> formation_candidates. Backward compatible: OFF is asserted BIT-IDENTICAL against a
  hand-recomputed trailing-only tally (C13), not merely "similar".
  WHAT IT CHANGES. As built, note_outcome credited only the sub-sequences ENDING at the
  current position (actions[-size:]), so with a buffer of L symbols it tallied one key per
  size instead of L-size+1 and NEVER tallied a leading sub-sequence. ON credits every
  contiguous sub-sequence in the buffer, each DISTINCT key at most ONCE per outcome.
  _was_executed moves with it under the same flag (matches anywhere in the buffer, not just
  the tail) -- untied, chunks formed at non-trailing positions could never be corroborated,
  converting a C1 formation failure into a C2 crystallisation one.
  THE PER-OUTCOME DEDUP IS LOAD-BEARING, not an optimisation. A sub-sequence can recur
  within one episode; crediting per occurrence would let a single 5-long HELD run advance the
  tally by 4 and drive variance to 0, manufacturing the (reps >= R_min AND var < F_low)
  conjunction the gate exists to test. Measured, not theoretical: crediting the raw executed
  stream at every position without dedup mints 52-86 "chunks" that are overwhelmingly
  (1,1,1,1,1) / (2,2,2,2,2) with variance identically 0.0000. A readiness criterion would
  PASS on those spuriously. Pinned by two C13 contracts.
  HONEST EFFECT -- IT DOES NOT ON ITS OWN CLOSE V3-EXQ-810's C1. Measured on the exact 810
  readiness cells (ARM_FULL, seeds 101/202/303, 120 eps x 24 steps), with the as-built
  baseline first reproduced bit-for-bit off the manifest (formed 7/0/0, crystallised 6/0/0,
  form_seed_frac 0.333): flag ON gives seed 101 formed 7->8 and crystallised 6->7, and leaves
  seeds 202/303 at ZERO with tracked UNCHANGED at 6. On a degenerate stream the rule is a
  genuine no-op -- with a 3-symbol buffer the extra start positions yield keys the trailing
  rule already had, and the dedup collapses them.
  WHY, AND WHERE THE REAL BOUND IS. record_step is reached only on the E3 deliberation path
  (the non-E3 hold path returns earlier in select_action at agent.py), and E3 ticks every
  e3_steps_per_tick = 10 env steps. At 810's 24-step episodes that is THREE symbols per
  episode, measured buf=3.00. Two consequences: (a) note_outcome breaks out at
  len(actions) < size, so chunk sizes 4 and 5 are STRUCTURALLY UNREACHABLE -- max_chunk_size=5,
  the MECH-323 growable ceiling and the ARC-071 growable depth are all inert at that episode
  length, confirmed by formed-chunk lengths {2: 4, 3: 4} with nothing above 3; and (b) the
  behavioural repertoire is too thin to chunk -- on the failing seeds 43/120 and 37/120
  episodes consist of a SINGLE held action, against 16/120 on the seed that forms, and the
  whole run yields only 17-19 distinct commitment tuples against 36. Lowering R_min cannot
  help: repetition already reaches 37 against a bar of 5, and variance passes too. ONLY the
  evaluative gate blocks, on sequences that are the modal behaviour.
  MEASURED: flag ON at 810's own 24-step config leaves form_seed_frac at 0.333, IDENTICAL to
  as-built. Stated plainly because it is the point: this change is enabling, not sufficient.
  THIRD STARVATION CAUSE -- THE 810 DRIVER RUNS A REDUCED AGENT LOOP. MECH-091's
  salient-event E3 phase_reset() lives in REEAgent.update_residue, and 810's loop is
  sense -> clock.advance -> _e1_tick -> generate_trajectories -> select_action -> env.step ->
  note_chunk_outcome. It never calls update_residue or observe_outcome, so no salient event
  can ever shorten the E3 interval and the tick is PERFECTLY periodic -- which is why the
  measured buffer is exactly 3.00 on every seed rather than fluctuating. Compounded by
  num_hazards=0 in the same config, which would have been the main source of those events.
  CONSEQUENCE FOR THE RE-RUN: V3-EXQ-810a needs THREE changes, and flipping this flag is not
  one of them. (1) LENGTHEN EPISODES -- E3 ticks per episode is what sets the buffer, so ~72
  steps gives ~7 symbols and makes sizes 4-5 reachable at all. MEASURED at 72 steps (60 eps,
  flag ON): buf=8.00 and formed-chunk lengths {2:4, 3:3, 4:6, 5:2} -- sizes 4 and 5 DO form,
  so the growable ceiling and growable depth stop being inert, and the forming seed improves
  hard (formed 8->15, crystallised 7->12, tracked 25->134). BUT form_seed_frac STAYS 0.333:
  seeds 202/303 still form nothing (tracked 4 and 46). So longer episodes are NECESSARY and
  NOT SUFFICIENT -- they fix the structural inertness, not C1. Caveat on that negative: the
  72-step probe ran 60 episodes rather than 120, so those seeds got half the outcome reports;
  the positive is unaffected (seed 101 formed MORE in half the episodes). The residual blocker
  on 202/303 is behavioural, which is what (2) targets. (2) DRIVE THE FULL AGENT LOOP
  (update_residue/observe_outcome) and consider num_hazards > 0, so salient-event phase resets
  actually fire. (3) RAISE THE SEED COUNT -- C1 is a 3-seed binary vote on a stochastic event,
  so at a true per-seed formation rate of 0.8 it still fails ~10% of the time.
  ALSO FOUND -- 810's OWN INSTRUMENTATION IS DEFECTIVE: the manifest reports
  chunk_acc_n_steps = 0 on rows that also report chunk_acc_n_formed = 7. Those cannot both be
  true (formation requires recorded steps). Fix the step readout in 810a before drawing any
  inference from it.
  New readouts: chunk_acc_n_credit_events, chunk_acc_all_position_credit (so the arms are
  separable in a manifest without reading config).
  Validation experiment: V3-EXQ-810a NOT yet queued -- the C1 precondition is not met at 24
  steps, so the successor needs a longer-episode design rather than a flag flip.
  See ARC-071, MECH-323, MECH-324, MECH-094, and the credit-rule section of
  policy_chunking.py's module docstring.

- ARC-071 / MECH-323: chunk SIZE and chunk DEPTH are GROWABLE CEILINGS DERIVED FROM THE
  DELIBERATION BUDGET, not fiat constants -- IMPLEMENTED 2026-07-27 (size ree-v3 c74434f,
  pre-rebase sha 95f9376932; depth ree-v3 7c201f7). Two landings on ONE compute-versus-
  efficiency trade-off, documented together because they share a budget and are coupled
  structurally. ree_core/policy/policy_chunking.py (ChunkAccumulator.effective_max_chunk_size
  / consider_ceiling_growth, PolicyChunking.effective_max_depth / structural_max_depth /
  consider_depth_growth, ChunkAccumulatorConfig.derived_chunk_ceiling),
  ree_core/policy/policy_decomposition.py (depth_cap_config_issues takes the derived bound),
  ree_core/utils/config.py, ree_core/agent.py (_resolve_chunk_deliberation_horizon + the
  ChunkAccumulator/PolicyChunking build sites).
  Config: REEConfig.use_growable_chunk_ceiling (default False = chunk_max_size is a hard
  lifetime cap, as-first-built, bit-identical; set True to enable) + chunk_ceiling_budget_fraction
  0.1667 + chunk_ceiling_returns_threshold 0.10 + chunk_ceiling_hard_max 12; and
  REEConfig.use_growable_chunk_depth (default False = chunk_max_depth is a hard lifetime cap;
  set True to enable) + chunk_depth_budget_fraction 0.1 + chunk_depth_returns_threshold 0.10 +
  chunk_depth_hard_max 6. BOTH read ONE shared budget knob, REEConfig.chunk_deliberation_horizon
  (default 0). All nine plumbed through the three from_dims sites (dataclass field, signature,
  body) plus the agent build site, and registered PROBED in tests/test_flag_inertness.py.
  NESTING: both are sub-switches of use_policy_chunking (the accumulator/library are never
  constructed without it), so either flag ON with chunking OFF is silently inert rather than an
  error -- unlike use_chunk_dissolution_retention, which RAISES under use_chunk_maintenance=False.
  THE SHARED BUDGET KNOB IS SENTINEL-DEFAULTED, and that is a live-knob guarantee rather than a
  convenience: chunk_deliberation_horizon = 0 MIRRORS HippocampalConfig.horizon (the real rollout
  budget; falls back to 10 with no hippocampal block), >= 1 is an explicit experimental override
  honoured verbatim (ree_core/agent.py:_resolve_chunk_deliberation_horizon). Unconditionally
  mirroring the hippocampal horizon would have shipped a from_dims knob that accepts a value and
  silently discards it -- the same dead-knob class as from_dims swallowing an unknown kwarg, one
  layer further down and harder to see.
  Data flow: HippocampalConfig.horizon (or the override) -> derived_chunk_ceiling =
  floor(horizon * chunk_ceiling_budget_fraction) and derived_max_depth = floor(horizon *
  chunk_depth_budget_fraction) -> effective ceilings START at chunk_max_size / chunk_max_depth
  and grow ONE element / ONE level at a time toward those bounds -> each step licensed by a
  realised marginal outcome gain >= the returns threshold -> the raised ceiling takes effect from
  the NEXT note_outcome. Growth is evaluated in PolicyChunking.note_outcome AFTER the tally,
  reacquisition and maintenance passes, size BEFORE depth (the structural bound reads the live
  size ceiling, so evaluating depth first would judge it against a stale size budget and refuse a
  growth the same trial had just licensed).
  Backward compatible: both default False; the effective ceilings are pinned at the inherited
  constants and no growth path runs, so a default agent is bit-identical to the 2026-07-22
  landing. VERIFIED, and this is the point of the anchoring below: at REE's ACTUAL rollout horizon
  of 30 the derivations return exactly 5 and exactly 3, so even flag-ON at today's budget starts
  where the fiat constants sat.
  NEITHER NUMBER IS REPLACED. Neither source licenses a replacement constant, so none is asserted
  -- the fractions are ANCHORED (0.1667 = 5/30, 0.1 = 3/30) to reproduce the inherited Sakai
  budget and the inherited R4 recursion cap at the budget the agent actually has. Only a LARGER
  deliberation budget derives a larger ceiling. What changed is the parameter's SHAPE, from
  constant to function. Read the anchors as calibration: the papers fix the shape of the
  relationship and nothing about its scale, so the scale is pinned to the one quantity that does
  carry warrant rather than to a number invented here. The depth anchor is pinned by a contract
  that builds a REAL REEAgent, because HippocampalConfig.horizon has a dataclass default of 10
  that no built agent uses -- anchoring against 10 would have been a large raise disguised as a
  derivation.
  THE BRAKE IS THE GROWTH RULE, not a second parameter. What bounds chunk growth empirically is
  DIMINISHING RETURNS (Ramkumar's monkeys never collapse the whole sequence into one chunk), so a
  merge must have actually PAID: size growth needs a realised gain over the best shorter context,
  depth growth a realised gain from composing AT the ceiling over the best chunk one level
  shallower that it contains. An accumulator that grew monotonically with practice would
  eventually collapse everything into one unit whatever number it stopped at -- the fixed-constant
  failure in a slower disguise. Means come from the LIVE tally, never the frozen value_tag, so a
  chunk whose returns have since collapsed cannot keep licensing growth; and "no evidence" is not
  a gain of zero (both contract-pinned).
  DECOUPLED FROM R_min BY CONSTRUCTION, on both ceilings. Bo 2009 found the capacity-to-chunk-
  LENGTH correlation in both age groups but capacity-to-learning-RATE only in the young, so size
  and formation rate are SEPARABLE quantities. The obvious implementation -- grow the ceiling as
  repetitions accumulate -- would silently re-couple them and re-introduce exactly the confound
  that dissociation rules out. Growth reads realised marginal outcome gain and the deliberation
  budget; chunk_min_repetitions enters only as a judge-ability filter (is this sequence attested
  enough to be MEASURED), never as the thing being measured. Do not "simplify" it into a practice
  counter.
  THE STRUCTURAL COUPLING IS THE SHARP PART, and it is mechanical rather than by analogy. A chunk
  composes another only by CONTAINING it, so a depth-D chain needs D distinct sequence lengths
  drawn from [min_chunk_size, effective ceiling] and the deepest hierarchy this substrate can
  physically mint is structural_max_depth = effective_max_chunk_size - min_chunk_size + 1 (4 at
  the 2-5 budget). consider_depth_growth() refuses to grow past that bound and reads the LIVE
  (possibly grown) size ceiling, so raising the size budget is what makes deeper hierarchies
  reachable at all. Refusing an inert raise is how this avoids repeating the defect the 2026-07-27
  MECH-321 scoping spike found in decomposition_depth_cap: a depth knob with no degree of freedom
  over which a literature argument was nonetheless conducted. Note max_depth=3 is genuinely
  BINDING today -- at the 2-5 budget the substrate would otherwise mint a depth-4 chunk.
  DEPTH IS LINEAR IN THE BUDGET, NOT LOGARITHMIC. The log intuition comes from hierarchies whose
  span MULTIPLIES per level (b**D primitives at branching factor b). REE's descent is not that:
  each level is one more sequential re-tiling pass, and HippocampalModule._recursive_leaf_tiles is
  bounded by `iterations < depth_cap`, an ITERATION count -- so depth D costs linearly in D and the
  affordable depth is linear in the budget. The fraction absorbs the unknown per-level cost
  constant.
  MECH-321 COUPLING: decomposition_depth_cap is a DERIVED MIRROR of chunk_max_depth, so
  depth_cap_config_issues() now takes derived_max_depth as an optional third argument and stops
  warning INERT about a cap the growing ceiling will in fact reach. Omitted, or not higher than
  the static bound (the default, and every shipped MECH-321 run), it is byte-identical.
  TRACTABILITY IS PRESERVED, worth stating because the original 2-5 bound was justified by it.
  note_outcome's enumeration is O(L * ceiling), not combinatorial, L is capped at
  max(effective_max_chunk_size * 4, 32) by record_step, max_tracked_sequences (FIFO) is the hard
  memory bound and is unchanged, and chunk_ceiling_hard_max / chunk_depth_hard_max are absolute
  backstops.
  NOT IMPLEMENTED -- SHRINKAGE, on both ceilings, recorded as a gap rather than guessed at.
  Nothing lowers a ceiling once raised. Bo 2009's cross-agent half is covered by the derivation;
  the within-lifetime declining-capacity half needs a signal REE does not have.
  Biological basis: Ramkumar, Acuna, Berniker, Grafton, Turner & Kording 2016 (Nat Commun,
  10.1038/ncomms12176) -- chunking is the OUTPUT of an efficiency/computation trade-off, not a
  fixed capacity; two macaques learning a ten-element reaching sequence over months MERGE chunks
  as practice lowers computation cost, optimising "over increasingly longer horizons". Bo, Borza &
  Seidler 2009 (J Neurophysiol, 10.1152/jn.00393.2009) -- chunk length tracks visuospatial
  working-memory capacity and DECLINES with age, constraining from the other side. Solway, Diuk,
  Cordova, Yee, Barto, Niv & Botvinick 2014 (PLoS Comput Biol, 10.1371/journal.pcbi.1003779) --
  filed in the corpus as a deliberate NULL: the normative account of what makes one action
  hierarchy better than another is the paper that WOULD carry a principled policy-grain depth
  limit, and it declines to give one, capping its own analysis at one level for stated
  tractability reasons and stating the framework generalises to deeper hierarchies unaltered. The
  R3 sources supplying 3-4 (Badre & D'Esposito 2009, Koechlin & Summerfield 2007) are ANATOMICAL
  grain, not policy grain -- how deep a brain's control hierarchy runs, not how deep THIS agent
  can afford to search.
  New readouts (so an arm is separable in a manifest without reading config):
  chunk_acc_effective_ceiling / chunk_acc_ceiling_derived_max / chunk_acc_n_ceiling_growths /
  chunk_acc_last_ceiling_gain on the accumulator; chunk_effective_max_depth /
  chunk_derived_max_depth / chunk_structural_max_depth / chunk_n_depth_growths on the facade.
  Phased training required: no (pure arithmetic, no learned parameters, no gradients).
  MECH-094: not applicable directly -- growth is evaluated inside note_outcome, which is reached
  only from the waking outcome path, and the accumulator's existing hypothesis_tag refusal on
  record_step is untouched.
  Contracts: tests/contracts/test_arc071_policy_chunking.py -- the C11 block (14 contracts, 21
  cases: off-by-default and pinned when off, the derivation reproducing 5 at horizon 30, scaling
  with the budget, growth requiring a realised return, no-growth when a shorter context already
  predicts, no-evidence-is-not-a-gain-of-zero, the brake plateauing below the derived maximum,
  never exceeding the derived bound, decoupling from the repetition tally, an end-to-end
  not-inert check, reset returning the initial budget, incoherent-config refusal, from_dims
  forwarding, sentinel horizon mirroring the hippocampal budget) and the C12 block (20 contracts,
  28 cases: the same battery for depth, plus depth-and-size reading the SAME budget, the inherited
  cap being binding today, the structural bound being set by the size ceiling AND rising with a
  grown one, returns reading the live tally not the frozen value_tag, and get_state reporting both
  bounds separately); plus 4 C17b contracts in tests/contracts/test_arc070_policy_decomposition.py
  for the MECH-321 derived-bound handoff.
  Validation experiment: NOT YET QUEUED. Both ceilings are prerequisites for the coupled-parameter
  experiment (do decomposition depth and composition parameters move together as compute changes),
  which needed an independent variable that could move; chunk_deliberation_horizon is now that
  single variable. NOTE the standing V3-EXQ-810 bound: at 24-step episodes the buffer holds ~3
  symbols, so sizes 4-5 and any depth above the resulting structural bound are unreachable
  whatever these flags are set to -- a successor must run LONGER EPISODES, not merely flip a flag.
  MECH-323 / ARC-071 stay candidate/v3_pending; this build promotes and demotes nothing.
  See ARC-071, MECH-323, MECH-324, MECH-321/ARC-070 (the derived-mirror depth_cap), MECH-094, and
  the "CHUNK SIZE IS A GROWABLE CEILING" / "CHUNK DEPTH IS ALSO BUDGET-DERIVED" sections of
  policy_chunking.py's module docstring.

- ARC-070 / MECH-321: policy.decomposition_via_event_segmenter -- IMPLEMENTED 2026-07-24.
  ree_core/policy/policy_decomposition.py (new: DecompositionDecision, PolicyDecomposition,
  PolicyDecompositionConfig; pure decision logic, no learned parameters, mirrors
  ChunkAccumulator/ChunkLibrary in shape), ree_core/hippocampal/module.py
  (set_decomposition_source, _region_vs, _evaluate_decomposition_ticks,
  _recursive_leaf_tiles, _rollout_tile, _apply_policy_decomposition -- wired into
  propose_trajectories right after the ARC-071 chunk splice), ree_core/agent.py
  (instantiation with a loud precondition on hippocampal.use_event_segmenter,
  set_decomposition_source call, a mid-execution abort block beside the rung-6
  natural-commit-urgency release, get_policy_decomposition_state API).
  Config: REEConfig.use_policy_decomposition (default False; set True to enable),
  decomposition_vs_threshold 0.4 (matches this codebase's existing vs_gate_e1/e2_threshold
  convention), decomposition_depth_cap 3 (mirrors ARC-071's chunk_max_depth default so the
  inverse operations stay symmetric). NO hippocampal sub-config mirror -- unlike
  use_chunk_proposal_injection, HippocampalModule never reads use_policy_decomposition
  from its own config; it acts purely on whether set_decomposition_source() was called
  (same external-source-injection pattern as use_policy_chunking / _chunk_source), so the
  three-site REEConfig.from_dims hazard reduces to three sites here, not four.
  Data flow (pre-commit, R4 first phase): HippocampalModule.propose_trajectories builds
  ARC-071 chunk candidates as usual -> _apply_policy_decomposition sweeps up to 8 ticks of
  each candidate's own rolled-out states through PolicyDecomposition.evaluate(), which
  calls MECH-288 EventSegmenter.boundary_on(stream="rollout", ...) at each tick -> R1 OR
  trigger (region V_s -- mean of HippocampalModule.per_stream_vs, the SAME aggregate
  already used for anchor-write last_vs -- below decomposition_vs_threshold, OR the
  boundary firing) -> if depth < depth_cap and PolicyDecomposition.decompose_sequence()
  produces tiles (library-aware longest-match against shallower registered chunks,
  falling back to raw single actions), the candidate is REPLACED by its leaf-tile
  Trajectories (bounded recursive descent up to depth_cap levels); if depth >= depth_cap
  or no tiles exist, the candidate is EXCLUDED from the pool this tick rather than
  offered blind. Chunk injection is additive (the flat-grain CEM pool is untouched
  either way), so an excluded/replaced chunk never leaves E3 with nothing to select.
  Data flow (mid-execution, R4 second phase): while beta_gate.is_elevated and
  e3._committed_trajectory originated from chunk injection / MECH-321 decomposition,
  agent.select_action re-evaluates the REMAINING (unexecuted) tail of the committed
  sequence against the CURRENT observed latent on every tick, still via
  boundary_on(stream="rollout", ...) (the remaining content is, by definition, still a
  prediction, not an observation) -- a triggering remainder releases the commit latch
  (beta_gate.release() + the same five-field clear as the rung-6 release) so the NEXT
  tick's _e3_tick replans, rather than blindly finishing the remainder. This is a FOURTH
  principled release, alongside MECH-091 (safety, never overridden), the rung-6 duration
  release, and SD-034 closure de-commit.
  Backward compatible: disabled by default; agent.policy_decomposition stays None, every
  call site is None-guarded, HippocampalModule never registers a decomposition source and
  never calls boundary_on(stream="rollout", ...).
  Biological basis: Zacks et al. 2007 event segmentation theory (PE is the canonical
  boundary trigger, framework substrate-agnostic about observed-vs-imagined streams, R1
  conf 0.78); Schacter/Addis/Buckner 2008 constructive episodic simulation (the SAME core
  network supports remembering past and imagining future events -- R2 LOAD-BEARING
  empirical anchor for the bidirectional-substrate design, conf 0.74); Badre & D'Esposito
  2009 rostro-caudal prefrontal hierarchy + Koechlin & Summerfield 2007 cascaded cognitive
  control (R3 multi-level decomposition, depth cap 3-4, conf 0.78); McGovern & Barto 2001
  bottleneck-state subgoal discovery is an explicit FOIL, not the primary trigger (R5).
  Phased training required: no (pure arithmetic, no learned parameters, no gradients).
  THE ASYMMETRY WITH ARC-071 IS THE MOST IMPORTANT THING TO GET RIGHT HERE. ARC-071 is
  SLOW, repetition-driven, execution-side, and its default write path REFUSES
  hypothesis_tag=True outright (a hallucinated chunk would install a macro the agent never
  executed). ARC-070 is FAST, V_s-driven, simulation-side, and LEGITIMATELY FIRES under
  hypothesis_tag=True during rollout deliberation -- that is its PRIMARY phase, not an
  exception to be gated against. PolicyDecomposition.evaluate() has NO MECH-094 refusal
  branch at all: it never writes residue, never updates MECH-269 anchor sets, never
  touches MECH-287 broadcast, so there is nothing to refuse. The R4 hypothesis_tag
  argument is a pure diagnostics label distinguishing pre-commit (True) from
  mid-execution (False) call counts -- do not "fix" this by adding a write-gate that
  mirrors ChunkAccumulator.record_step; that would be importing the wrong claim's
  constraint. See policy_decomposition.py's "asymmetry with ARC-071" module docstring.
  A DEPTH-1 CHUNK MUST STILL BE ABLE TO DECOMPOSE -- CAUGHT BY THIS SESSION'S OWN
  ACTIVATION SMOKE TEST. decompose_sequence()'s first draft short-circuited to () whenever
  depth<=1 (reasoning: "nothing shallower than depth 1 to tile against"), which is correct
  about the library lookup but wrong about the OUTCOME: it made the single most common
  chunk shape (depth 1, composed directly of raw actions -- most ARC-071 formations never
  reach depth 2) permanently un-decomposable, landing every triggering depth-1 chunk as
  marked_unreliable/dropped instead of genuinely re-segmented. Fixed to skip only the
  library-lookup phase at depth<=1 while still falling through to the raw-action tiling
  loop (contract C3 pins this as a regression guard).
  Contracts: tests/contracts/test_arc070_policy_decomposition.py (21 tests: C1 defaults +
  from_dims forwarding + OFF inertness (no sub-config mirror needed), C2 loud precondition
  on use_event_segmenter, C3 depth-1 regression pin, C4 library-aware tiling incl.
  DISSOLVED chunks excluded, C5 atomic sequence has no tiles, C6 R1 OR-trigger (V_s alone
  / boundary alone / neither), C7 depth_cap marks unreliable instead of decomposing, C8
  live pre-commit withhold-and-replace, C9 live mid-execution latch release, C10 additive
  passthrough when nothing triggers incl. the no-decomposition-source bit-identical path;
  C11-C16 = the R5 bottleneck trigger mode below).
  Registered in tests/test_flag_inertness.py PROBED.
  R5 BOTTLENECK TRIGGER MODE -- ADDED 2026-07-25 (unblocks ARM_2 of the discriminative
  validation). The R1 trigger above is now selectable against an R5 alternative via
  REEConfig.decomposition_trigger_mode: "vs_boundary" (DEFAULT = the R1 OR trigger, unchanged
  and bit-identical) | "bottleneck" (R5: decompose ONLY at bottleneck states, REGARDLESS OF
  V_s). ARM_1 uses "vs_boundary"; ARM_2 uses "bottleneck". Bottleneck detection is an ONLINE
  INCREMENTAL diverse-density accumulator (PolicyDecomposition._update_and_test_bottleneck):
  a region is a bottleneck once it has BOTH recurred >= decomposition_bottleneck_min_visits
  times (default 3 -- the "repeated traversals" gate, McGovern & Barto 2001, and the source
  of ARM_2's predicted one-shot-rare signature) AND been entered-from/exited-to >=
  decomposition_bottleneck_min_distinct_neighbors distinct regions (default 2 -- funnel
  topology). Region key = a coarse FIXED quantisation of z_world content (round(z_world[:dims]
  / quant); decomposition_bottleneck_region_quant default 1.0, decomposition_bottleneck_region_dims
  default 8) -- NOT MECH-288's segment_id, which is monotonic and never recurs so could never
  register a bottleneck (documented in the module docstring "REGION KEY"). boundary_on(stream=
  "rollout", ...) is still called every evaluate() (R2 shared-substrate consumer + advances the
  rollout stream) but in bottleneck mode feeds only the audit fields, not the decision; V_s /
  boundary counters keep incrementing for audit (so a manifest can show "V_s WAS low yet the
  bottleneck trigger did not fire on it" -- the ARM_2 discriminative evidence).
  FAITHFULNESS CHOICE (documented deliberately): MECH-321's functional_restatement frames R5
  as a candidate OFFLINE/CONSOLIDATION-PHASE mechanism (batch analysis of the ARC-071
  ChunkLibrary for bottleneck topology). This landing operationalises the SAME statistical
  signal ONLINE/incrementally for the ARM_2 primary-trigger requirement; it does NOT claim to
  be the offline consolidation mechanism, which stays DEFERRED per R5 (see
  policy_primitive_granularity.md and the module docstring "R5 BOTTLENECK TRIGGER MODE").
  Default OFF / bit-identical: the accumulator is never allocated in "vs_boundary" mode (C15
  pins regions_tracked==0). New diagnostics in get_policy_decomposition_state():
  decomp_trigger_mode, decomp_n_bottleneck_fires, decomp_n_bottleneck_regions_tracked. Three-
  site REEConfig.from_dims wiring for all five new params (dataclass field, from_dims signature,
  from_dims body) + the agent.py PolicyDecompositionConfig construction site.
  Contracts for the mode: C11 default+validate (rejects unknown mode / non-positive gates) +
  from_dims forwarding; C12 bottleneck mode ignores V_s (low V_s alone does not fire, audit
  counter still increments); C13 one-shot-rare vs repeated-fires; C14 diagnostics surfaced;
  C15 vs_boundary leaves the accumulator untouched (bit-identity); C16 depth_cap still respected.
  Validation experiment: V3-EXQ-TBD queued via /queue-experiment (substrate-readiness
  diagnostic -- does decomposition fire and correctly withhold/replace a chunk candidate
  under an artificially induced low-V_s / boundary-firing rollout region). The full
  ARM_0/ARM_1/ARM_2 discriminative-pair design from MECH-321's claims.yaml
  functional_restatement (V_s-drop primary vs bottleneck-state primary vs OFF baseline,
  measuring execution-time prediction-failure rate) is a LATER experiment, mirroring how
  ARC-071's behavioural-latency measurement was deferred past its own substrate-readiness
  diagnostic.
  See ARC-070, ARC-069 (parent), ARC-071 (the inverse composition operator, BUILT
  2026-07-22 -- see the entry above), MECH-288 (substrate consumed), MECH-269 (V_s trigger
  source), MECH-094, REE_assembly/docs/architecture/policy_primitive_granularity.md.

- stdlib-`random` seeding in ree_core (`REEConfig.stdlib_rng_seed`) -- IMPLEMENTED 2026-07-28.
  NOT an SD -- a reproducibility knob over existing mechanisms, same class as
  `ScaffoldedSD054OnboardingConfig.scaffold_env_seed` (which likewise got contracts, no SD doc).
  THE GAP: three ree_core call sites draw from the stdlib `random` module, which auto-seeds
  from OS entropy at import, while experiment drivers seed only torch and numpy (`_run_seed`
  in a typical driver calls `torch.manual_seed` + `np.random.seed` and nothing else). Any run
  reaching one of those sites is therefore NOT reproducible across processes. `ree_core` seeds
  stdlib `random` nowhere -- verified by grep across the whole package.
  NOT the scaffold-curriculum non-determinism. That was the unseeded env
  (`np.random.default_rng(None)`), fixed in `7afea288fa`; on the 460c dry_run curriculum this
  path is DORMANT (stdlib-random state never advanced across all 60 Stage-0 select_action
  calls). Do not re-open that question -- it is closed. This landing closes a DIFFERENT,
  still-open exposure that would look identical if it ever fired.
  MEASURED LIVENESS (2026-07-28, per-site call counters on the module-level `random` binding
  each site resolves through -- not inferred from code reading):
    S1  hippocampal/module.py `diverse_replay(mode="auto")` per-step mode roll -- FIRES ONCE
        PER REPLAY STEP whenever `replay_diversity_enabled=True` (measured 5 rolls for a
        5-step replay; 0 with the flag off). Broadest exposure of the three: one config flag,
        and agent.py's SWS replay path passes `mode="auto"` explicitly.
    S2  hippocampal/module.py `_sample_exploration_trajectory` zero-weight fallback -- narrow:
        needs `retrieval_bias` supplied AND summing to zero against the memory_strength
        weights (measured: bias=None -> 0 calls, bias=[1,2] -> 0, bias=[0,0] -> 1). Reachable
        via a degenerate BLA retrieval bias, and called directly by V3-EXQ-659.
    S3  sleep/self_model_aggregator.py `offline_gradient_pass` waking-pair sample -- fires
        whenever the harm replay buffer is non-empty (measured: empty -> 0, non-empty -> 1)
        under `use_mech273_self_model=True`, which `REEConfig.enable_sleep_aggregation_cluster()`
        turns on as one of its eight flags.
  THE FIX (opt-in; default bit-identical): `REEConfig.stdlib_rng_seed: Optional[int] = None`,
  three-site from_dims wiring (dataclass field, signature, assignment) plus a mirror onto
  `HippocampalConfig.replay_rng_seed`, guarding the silently-unreachable-flag hazard.
    Each consumer holds an `_rng` source that DEFAULTS TO THE `random` MODULE ITSELF, so
      `self._rng.random()` / `.choice()` / `.choices()` resolve to the very same bound methods
      of the process-global Random instance the previous bare `random.*` calls used. The
      default is bit-identical BY CONSTRUCTION, not by equivalence.
    When set, each consumer gets its own `random.Random` instance. `random.seed()` is NEVER
      called -- a module-local instance, so seeding an agent cannot perturb the host process's
      global RNG or any other stdlib-random consumer in it.
    `derive_stdlib_rng_seed(base, stream)` namespaces the consumers (stream 0 = hippocampal
      replay, stream 1 = self-model writeback) so one base seed never hands two consumers a
      correlated sequence. Same idiom as `_derive_env_seed` in scaffolded_sd054_onboarding.py.
    `HippocampalModule.seed_replay_rng(seed)` is a no-op on None, so both construction paths
      (from_dims mirror, and a hand-built REEConfig setting only the top-level knob) honour it.
  Backward compatible -- verified TWO ways, not just asserted:
    (a) A/B against pre-change HEAD (`25bf07e`) in a throwaway worktree, knob unset, global
        stdlib RNG pinned: identical replay digest
        (`efcc3ecd07b5f910222d7854a8dc7330643c0a3c2a37c8685b0beaa72c80471e`), identical
        reverse-count (5/12), identical 20-element fallback pick sequence.
    (b) Identity assertions in the contracts (`_rng is random`), which is the structural
        reason (a) must hold.
  DELIBERATE BEHAVIOUR CHANGE WHEN SET: a seeded run reaches different draws and is NOT
  comparable to a landed run. Pin it deliberately, within one experiment, as a seeded pair.
  Do NOT retro-apply to landed runs or re-score claims on it.
  Contracts: `tests/contracts/test_stdlib_rng_seed_determinism.py` (12 tests), pinning BOTH
  directions per the scaffold_env_seed precedent -- D1/D2 default-is-the-global-instance
  (identity), D3 negative control that the unseeded sites really DO consume the global RNG,
  D3b None passthrough, D4 reproducible across constructions, D5 independent of global RNG
  state, D6 distinct seeds -> distinct streams, D7 seeding never perturbs the global RNG,
  D8 S2 fallback, D9 S3 writeback (both with unseeded negative controls), D10 stream
  namespacing, D11 the from_dims three-site mirror. Mutation-checked in both directions:
  making `seed_replay_rng` a no-op fails D4/D5/D7/D8; making the default a fresh
  `random.Random()` fails D1/D3.
  No validation experiment queued: this is a reproducibility knob whose default is proven
  bit-identical and whose ON path is pinned by contracts -- there is no substrate behaviour
  to validate empirically. The natural first consumer is any future cross-process determinism
  check that needs `replay_diversity_enabled=True`.
  CROSS-PROCESS REPRODUCTION VERIFIED 2026-07-28 -- the 12 contracts pin determinism WITHIN
  one process only, so "does a pinned run actually reproduce in a SECOND process, or is there
  a FOURTH entropy source still unfound?" was open until measured.
  `scripts/stdlib_rng_cross_process_probe.py` runs a real agent+env+sleep loop on a config
  reaching ALL THREE sites and digests the full trajectory + sleep metrics + every parameter
  byte. Measured on ree-worker-3 (linux-x86_64, py3.10.12, torch 2.12.0+cpu):
    ARM pinned   (stdlib_rng_seed + env seed + torch/numpy seeded) -- 3 processes, ONE digest
      `8ff7ac0e1d803eb733f7bfd8aa0ebc48539df519496a571edd601e03b467f88d`, identical
      state_dict. Replicated on ree-worker-2: identical in-machine AND byte-identical to
      worker-3's digest (same machine class). Draw counts S1=15 / S2=12 / S3=1 every run.
    ARM unpinned (the typical-driver state: torch+numpy seeded, stdlib `random` on OS
      entropy, everything else held) -- 3 processes, THREE DIFFERENT digests and three
      different state_dicts.
  RE-MEASURED 2026-07-28 ON THE REAL SLEEP ROUTE. The measurement above drove S2 and S3
  directly at their call sites, because Phase E early-returns with
  `mech273_writeback_regions = 0` unless `replay_sampler.draw()` returns non-None -- which
  needs the MECH-269 anchor-set substrate `enable_sleep_aggregation_cluster()` deliberately
  does not bundle. The probe now closes that chain (`use_anchor_sets`, `use_event_segmenter`,
  `use_per_stream_vs`, `use_invalidation_trigger`, `use_staleness_accumulator`,
  `mech285_draws_per_cycle`, `use_affective_harm_stream`, and `sleep_loop_episodes_K` above
  the episode count so `reset()` cannot fire an implicit second cycle), so **S3 now fires
  through `SleepLoopManager.force_cycle` -> Phase-E writeback**, not by direct call. Only S2
  remains direct (`_sample_exploration_trajectory` needs an all-zero retrieval_bias the waking
  route does not produce), still via the agent's own seeded consumer over its real buffer.
  New counts, every run of both arms: `mech273_writeback_regions = 4` in all 3 cycles,
  `self_model.choices = 3` (one per cycle). S1 falls 15 -> 5 and S2 stays 12; the S1 drop is
  the config change altering how many e3_quiescent replay ticks occur, not lost coverage.
    ARM pinned   -- 3 processes on ree-worker-3, ONE digest
      `027b3acdae1177324a9970a0b779ef5cef96a6a3d836680774d567738313e5d0`, identical
      state_dict. Replicated on ree-worker-2: same digest, again byte-identical across the
      two boxes of the same machine class.
    ARM unpinned -- 3 processes, THREE DIFFERENT digests and state_dicts on EACH worker
      (six distinct digests across the two). Sensitivity is visible in the counters too:
      unpinned draws S1=10 rather than 5, and one run diverged far enough to install a 5th
      anchor (`wb_regions [4,4,5]`).
  ANCHORS ARE SEEDED, NOT WAKING-DERIVED -- worth stating plainly. The natural route was
  tried first and measured: over all 36 waking ticks the MECH-288 segmenter emitted ZERO
  BoundaryEvents, so the pool stayed empty and every `draw()` returned None. That is the
  probe's tiny task, not a substrate defect: the policy collapses to one repeated action in
  episode 1, the z_world/z_self deltas then decay monotonically, and a monotonically
  decreasing series never rises 0.65 sigma above its own trailing window mean, which is what
  the fast scale's `pe_threshold` detector needs. So `_seed_anchors` writes the pool directly,
  as V3-EXQ-574 (the run that validated MECH-273 Phase E) does -- with a deterministic arange
  ramp rather than 574's `torch.randn`, since a reproducibility probe should not add an RNG
  consumer it does not need. Anchor provenance is upstream of every RNG site under test; what
  had to be real is the sleep-side route, and that is now entirely `_run_cycle`.
  The coverage check is no longer advisory: `main()` now EXITS 3 if `self_model.choices` is
  absent or `mech273_writeback_regions` is 0 in every cycle, so a config regression fails
  loudly instead of printing a reassuring hash.
  So: NO fourth entropy source, and the unpinned arm is what makes that statement mean
  anything -- it proves the probe is sensitive to exactly the entropy this knob closes.
  Do NOT drop the unpinned arm: without it, two matching pinned digests are equally
  consistent with "the seeding works" and with "the sites never fired". The draw counters
  are the second, direct check on the same thing, and they earned their keep -- the FIRST
  two probe builds reported ZERO draws (the `tiny_loop.step_once` path never reaches
  `_do_replay`, the only waking route to S1; and `agent.run_sleep_cycle()` stops before the
  Phase-E WRITEBACK that reaches S3, which needs `SleepLoopManager.force_cycle`). Both would
  have produced two matching, entirely meaningless digests.
  SCOPE: same machine class only. `torch.multinomial` is not reproducible across
  darwin-arm64/torch 2.10 vs linux-x86_64/torch 2.12 (see the "Running the test suite"
  cross-machine-class note in the umbrella CLAUDE.md), and that is unaffected by this knob.
  One footnote worth keeping, and it still costs time if forgotten: `harm_dim` must equal
  `E2HarmSConfig.z_harm_dim` (32), because `offline_gradient_pass` slices waking pairs to
  that width and a narrower stream slices SHORT rather than erroring, dying later on a
  shape mismatch deep in the transition net.
  See REE_assembly/evidence/planning/scaffold_goal_freeze_e3_read_path_triage_2026-07-27.md
  ("Follow-on: the residual entropy source is NAMED and FIXED", point 4), MECH-165 (S1/S2),
  MECH-273 (S3), and `experiments/scaffolded_sd054_onboarding.py` scaffold_env_seed (the
  design precedent this follows).
