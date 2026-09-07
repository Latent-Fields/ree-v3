## INV091-NULL-VALIDATION-RUN-LENGTH: standing default eval budget for the INV-091 driver family

- INV091-NULL-VALIDATION-RUN-LENGTH (substrate_queue.json sd_id, not a numbered SD --
  a run-length default fix, not a new architectural feature) -- IMPLEMENTED 2026-07-31.
  New `experiments/_lib/inv091_driver_defaults.py`: standing default eval-phase episode/step
  budget (`INV091_WARMUP_EPISODES=40`, `INV091_EVAL_EPISODES=24` [was 10],
  `INV091_STEPS_PER_EPISODE=150`) for the INV-091 cross-stream-similarity-band driver family
  (v3_exq_827 -> 827a -> 828 lineage). All three runs to date were executed at
  `EVAL_EPISODES=10` (1500 eval-phase steps) and every one found `null_validation.checked=False`
  -- too short for `q081_surrogate.plan_blocks` to build a valid constrained-realisation null.
  The worst POST-phase-sync-fix deficit recorded (V3-EXQ-828, 6 arms) was 2848 steps needed
  against 1500 supplied; 827's 12000-step reading is excluded from the sizing basis (it was
  driven by the pre-fix lockstep clock-collapse bug that 827a's redesign already corrected, not
  a property of a correctly-built driver). The new default (24 * 150 = 3600 eval-phase steps)
  clears the recorded 2848-step floor with 26% margin.
  Config: no new `REEConfig` / `ree_core` flag -- these are experiment-driver constants, not
  substrate config. `q081_surrogate.py`'s null-validation THRESHOLD (`DEFAULT_SAFETY_FACTOR`,
  `DEFAULT_MIN_BLOCKS`) is untouched; only the caller's run length is bumped to clear it.
  Data flow: n/a (no new latent, encoder, or agent-loop wiring).
  Backward compatible: no already-run experiment's recorded result changes; 827/827a/828 stay
  exactly as reviewed and scored in claims.yaml. A future successor script in this family opts
  in by importing the new constants (see the module docstring for the exact import).
  Not a new architectural feature, so no SD doc / phased-training / MECH-094 considerations
  apply.
  Validation: 5 new contracts, `tests/contracts/test_inv091_null_validation_run_length.py`
  (positive floor-clearance check with a documented margin threshold; warmup/step-length
  unchanged from the original driver; a negative control against the original undersized
  1500-step configuration; the module's own top-level assert fires on a regressed constant;
  the exact import path the docstring documents) -- all pass, alongside the existing
  `test_q081_surrogate_null.py` + `test_q081_landmark_removal.py` (66 passed total).
  Validation experiment: none queued in this session (scope discipline -- this is a substrate
  readiness fix, not the experiment itself). A `/queue-experiment` follow-on for the next
  INV-091 driver-family successor, importing these constants, is the natural next step and is
  chipped rather than done here.
  See INV-091, `REE_assembly/evidence/planning/substrate_queue.json` sd_id
  INV091-NULL-VALIDATION-RUN-LENGTH.
