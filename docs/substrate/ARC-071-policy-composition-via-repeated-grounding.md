## ARC-071: policy.composition_via_repeated_grounding -- IMPLEMENTED (2026-07-22)
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
