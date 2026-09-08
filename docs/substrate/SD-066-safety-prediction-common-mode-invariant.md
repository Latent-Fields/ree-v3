## SD-066: safety_prediction.common_mode_invariant_conditioned_safety_readout -- IMPLEMENTED (2026-07-15)
- SD-066: safety_prediction.common_mode_invariant_conditioned_safety_readout --
  IMPLEMENTED 2026-07-15. Module: ree_core/safety/conditioned_safety_store.py
  (ConditionedSafetyStore, non-parametric; NOT the V4 trainable-contrastive head).
  Opt-in centered readout that lifts the z_world common-mode CEILING blocking
  MECH-304's behavioural gate: under SD-008 z_world under-differentiation every
  z_world sits at cosine ~0.99, so the store's raw-cosine gate (sigmoid(gain*cos) >
  0.5 == cos>0) fires UNCONDITIONALLY once the prototype is non-empty -- the cue
  (~0.006 cosine) is unresolvable behaviourally (diagnosed 2026-07-15: raw store
  arm A(cue) == arm C(nocue) release, sig(cue)~sig(nocue)). Fix: the store keeps a
  slow EMA `baseline` of z_world (the common-mode) and does BOTH prototype
  accumulation and querying on the CENTERED residual z_world - baseline, so the cue
  residual dominates the cosine. Config: REEConfig.safety_store_centered (bool,
  default False -> bit-identical raw cosine) + safety_store_baseline_alpha (0.02),
  surfaced by from_dims; agent.py passes them to the store constructor. Baseline
  advances inside store.update() (already called every tick in sense()) -> NO agent
  hot-path/wiring change. sim_mode (MECH-094) does not advance the baseline. Data
  flow unchanged except the internal cosine is centered. Backward compatible:
  centered=False byte-identical (contract C1 re-implements the raw arithmetic and
  matches every update() return). Phased training: NOT REQUIRED (non-parametric).
  5/5 new contracts in tests/contracts/test_sd066_centered_safety_readout.py (raw
  bit-identical + never-touches-baseline / baseline lifecycle + reset / common-mode
  separation / sim_mode gate / config default); full pytest tests/ 1463 passed.
  VALIDATED end-to-end: with centered ON the MECH-304 behavioural gate becomes
  cue-specific (arm A cue release 1.0 / sig 0.7-0.94 vs arm C nocue release ~0 /
  sig 0.17-0.48; A-C ~1.0) -- V3-EXQ-763 PASS.
  Validation experiment: V3-EXQ-763 (MECH-304 promote-to-active behavioural falsifier).
  Design doc: REE_assembly/docs/architecture/sd_066_centered_conditioned_safety_readout.md
  See SD-051 (the store this extends), SD-065 (the cue channel it resolves), SD-008
  (the z_world under-differentiation it works around), MECH-304 (the gate it unblocks).
