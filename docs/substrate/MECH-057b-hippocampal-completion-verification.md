## MECH-057b: agentic_extension.thought_loop_trajectory_promotion_gate -- IMPLEMENTED (2026-09-14)
- MECH-057b: agentic_extension.thought_loop_trajectory_promotion_gate -- IMPLEMENTED 2026-09-14.
  Module: ree_core/hippocampal/module.py (HippocampalModule.verify_sequence_completion,
  HippocampalModule.promote_candidates), ree_core/hippocampal/visitation.py
  (VisitationCounter.query -- new lawful consumer, see its module docstring).
  Config: HippocampalConfig.use_completion_promotion_gate (default False; no-op),
  .completion_verification_tau (default 1.0), .completion_promotion_verification_floor
  (default 0.3), .completion_promotion_drop_fraction (default 0.4),
  .completion_promotion_min_candidates (default 2). All five forwarded through
  REEConfig.from_dims (same names).

  Problem this replaces: the only prior MECH-057b test (V3-EXQ-672b) implemented its
  "completion gate" entirely in EXPERIMENT-HARNESS code
  (experiments/v3_exq_672b_mech057b_trajectory_promotion_gate.py::
  _filter_trajectories_by_completion), ranking candidates directly on
  hippocampal._score_trajectory -- the ARC-007 STRICT terrain/residue cost HippocampalModule
  uses for CEM elite selection, which carries NO completion signal (see that method's own
  docstring). failure_autopsy_V3-EXQ-672-series_2026-06-15 (CONFIRMED) diagnosed this as
  "the gate has the symbol of completion-verification... but not the functional role", and
  the harness proxy inherited the ARC-065 GAP-A candidate-pool collapse (cross-candidate
  spread ~0.009, completion_signal pinned ~0.3439 across arms) instead of measuring
  completion at all. This landing is the first REAL, substrate-resident verification signal.

  Data flow: HippocampalModule.propose_trajectories()'s rolled-out z_world sequence per
  candidate -> verify_sequence_completion() reads VisitationCounter.query() (read-only;
  never writes) at each waypoint, mapping visit count -> a saturating confidence curve
  count/(count+tau), taking the MINIMUM over the sequence (a completed SEQUENCE is only as
  reliable as its least-encoded waypoint) -> promote_candidates() withholds candidates below
  completion_promotion_verification_floor (subject to a deadlock guard and a drop_fraction
  cap) -> the survivors are the pool propose_trajectories() returns, i.e. what becomes
  "eligible for E3 selection" (MECH-057b's claim text). Applied LAST inside
  propose_trajectories(), after ghost-mixing / support-preserving injection / scaffold /
  chunk splicing, over the truly final candidate pool.

  Genuinely distinct from CEM scoring, on purpose: verify_sequence_completion() reads
  visitation memory (how well-ENCODED a region is -- CA3 autoassociative pattern completion,
  Lisman & Grace 2005; Pfeiffer & Foster 2013), never the ResidueField terrain/valence
  _score_trajectory reads. A candidate can have a favourable terrain score and zero
  completion confidence (never-visited region) -- the two channels can and do disagree; see
  test_verify_sequence_completion_distinct_from_score_trajectory.

  Deadlock guard: promote_candidates() never reduces the pool below
  completion_promotion_min_candidates, so a cold-start / sparse-memory tick (every candidate
  reads confidence 0.0) still leaves E3 a usable pool rather than starving it -- expected and
  common early in an episode or in training.

  Exempt sources: candidates tagged with a structural-injection provenance
  (support_preserving_cem_injected, action_class_scaffold, arc071_chunk,
  mech321_decomposed, mech293_ghost_probe -- see _PROMOTION_EXEMPT_SOURCES) are NEVER
  suppressed by this gate and never counted against the drop-fraction cap: those injectors
  exist to guarantee structural pool properties (e.g. action-class coverage) that
  visitation-based filtering would directly undermine (a scaffold's whole purpose is to
  cover an under-explored action class -- exactly where visitation confidence reads lowest).

  Backward compatible: use_completion_promotion_gate defaults False ->
  promote_candidates() returns its input list unchanged (identity, not a copy) and
  verify_sequence_completion() is never called from propose_trajectories() -> bit-identical
  to every landed run. VisitationCounter.query()'s other (telemetry) callers are unaffected;
  update() (the write side) is still never called from any scoring or selection path.

  Biological basis: CA3 autoassociative pattern completion (Lisman & Grace 2005,
  subiculum->NAc->VP->VTA completion-to-dopamine loop) and Pfeiffer & Foster 2013 (place-cell
  sequences depict future paths to remembered goals, checked for plausibility before
  emission) -- both already cited as MECH-057b's literature evidence in claims.yaml.

  MECH-094: not applicable to the gate mechanism itself (arithmetic filter over already-
  rolled-out proposal trajectories, no memory write). The read side (VisitationCounter.query)
  is explicitly read-only; the write side (update()) remains gated on real waking visits
  exactly as before this landing, unchanged.

  Phased training: none (no learned parameters; the gate is a deterministic arithmetic
  filter over an existing tracker).

  Validation: contract suite tests/contracts/test_mech057b_completion_promotion.py (22
  tests: config reachability incl. from_dims forwarding, verification-channel distinctness
  from _score_trajectory, min-over-waypoints-not-mean, selective suppression, deadlock
  guard, exempt-source protection, end-to-end propose_trajectories wiring and OFF-path
  bit-identity) plus the pre-existing hippocampal/visitation/CEM contract suite (348 tests
  total across both, all green, ree-cloud-4 hub run 2026-09-14). No behavioural EXQ queued
  from this landing by design: EXP-0594 (REE_assembly evidence/planning/
  manual_proposals.v1.json) is the pre-registered retest, and its release_condition
  requires BOTH this substrate (now landed) AND ARC-065 GAP-A candidate-pool
  discriminability (still open, substrate_queue.json priority 1) -- queueing a MECH-057b
  experiment against only the first half would re-trigger the exact re-derive-brake
  degeneracy the 672-series autopsy diagnosed. See substrate_queue.json sd_id
  "mech057b-hippocampal-completion-verification-promotion-policy" and claims.yaml
  MECH-057b's implementation_note.

  See MECH-057b, MECH-057a (action-loop BetaGate completion gate, distinct claim), ARC-028,
  MECH-105 (hippocampal->BetaGate completion coupling, unaffected by this landing), ARC-065
  (candidate-pool discriminability, the second release-condition blocker),
  failure_autopsy_V3-EXQ-672-series_2026-06-15.json,
  REE_assembly evidence/planning/manual_proposals.v1.json EXP-0594.
