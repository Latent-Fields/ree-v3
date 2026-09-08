## ARC-071 / MECH-324: reacquisition-window ISOLATION fix -- corrects V3-EXQ-829's confirmed FALSIFIED result -- IMPLEMENTED (2026-07-31)
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
