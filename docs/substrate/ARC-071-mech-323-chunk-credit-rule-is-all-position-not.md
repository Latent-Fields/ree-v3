## ARC-071 / MECH-323: chunk CREDIT RULE is all-position, not trailing-only -- IMPLEMENTED (2026-07-27)
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
