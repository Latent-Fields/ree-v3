## SD-105 freeze/share API: frozen, shared selection-entropy floor multiplier (2026-09-25)
- SD-105: control_plane.selection_entropy_headroom_floor -- FREEZE / SHARE API IMPLEMENTED 2026-09-25.
  substrate_queue entry `sd105_frozen_shared_entropy_floor_multiplier` (priority 1, severity
  CORRUPTING), build ratified by governance 2026-09-23 (GFLAG-0331 item 6) and reaffirmed
  2026-09-24 (rec-20260924-17600b54). Design note:
  `REE_assembly/evidence/planning/sd105_freeze_share_api_design_note_20260909.md`.
  WHY: the SD-105 live closed-loop set-point cannot be armed in a difference-of-arms design whose
  DV is realised selection entropy -- it applies a DIFFERENT lift per arm (arms start at different
  entropies) and compresses the contrast (V3-EXQ-963b red-team F1, ree-v3 d2104f88f4, verified on
  963a's per-arm data). The fix is to converge the multiplier ONCE and apply it as a single frozen,
  shared constant to every arm.
  Module: `ree_core/regulators/selection_entropy_floor.py` (`SelectionEntropyFloor.freeze()`,
  `.frozen`, `SelectionEntropyFloorConfig.frozen_multiplier` / `.freeze_after_ticks`).
  Agent: `REEAgent.freeze_selection_entropy_floor()` (raises when the floor is off).
  Config: `REEConfig.selection_entropy_floor_frozen_multiplier` (default None = live controller;
  a float in [1.0, max_temperature_ratio] builds the regulator ALREADY frozen at that value --
  the SHARE path) and `REEConfig.selection_entropy_floor_freeze_after_ticks` (default 0 = never;
  N auto-latches after the Nth real observation -- CONVERGE-THEN-FREEZE). Both wired through the
  dataclass field, the `from_dims()` signature and the `from_dims()` assignment. Setting both is a
  ValueError.
  Semantics: freeze is a one-way latch (no unfreeze; a second freeze() keeps the first latch
  point). After a freeze, observe() still advances the entropy EMA and every diagnostic counter
  (so a driver can see whether the frozen value HELD entropy at the floor) but log_mult does not
  move. The latch SURVIVES reset() (the V3-EXQ-779b episode-length confound). simulation_mode
  (MECH-094) returns before any of it: replay never freezes and never moves a frozen value. A
  frozen value is always >= 1.0 (one-sidedness) and <= the cap; frozen AT the cap still reports
  `saturated` True.
  Reporting: `get_state()` adds `frozen`, `frozen_multiplier` (None while live), `frozen_at_tick`,
  `frozen_source` ("converged" | "config" | None), `freeze_survives_reset`; the agent's
  `entropy_floor` control-vector dict mirrors the first four (None when the floor is off).
  `continuity_note` is unchanged.
  Data flow: warmup agent (live controller) -> `freeze_selection_entropy_floor()` -> m* ->
  every arm built with `selection_entropy_floor_frozen_multiplier=m*` -> unchanged application
  site in `select_action()` (tonic_T -> noise_floor -> [frozen multiplier] -> phasic delta).
  Consumer contract (V3-EXQ-963c): R6 changes KIND -- from a spread tolerance that self-routes
  requeue to a design ASSERTION (max(mult) - min(mult) == 0.0 across a seed's arms, `frozen`
  True everywhere), so a violation is a harness bug, not a scientific outcome. The design's own
  validation is claims.yaml SD-105 what_would_answer leg (ii): dS_tonic under the frozen
  multiplier must be paired-indistinguishable from dS_tonic with the multiplier OFF, which needs a
  multiplier-OFF paired arm set (4 arms -> 8, or a 2-arm probe).
  Backward compatible: floor OFF and floor ON-but-unfrozen are bit-identical to ree-v3 origin/main
  dbc6db8 -- verified DIFFERENTIALLY (action + per-tick multiplier trace hashes, 2 seeds x 2
  episodes x 40 steps, identical), and pinned by contract B14 against the pre-freeze integrator
  arithmetic. Liveness: frozen m=2.0 vs the live controller on the same seed changes the action
  sequence.
  Flag registry: neither knob matches the `use_*` / `*_enabled` scan of
  `test_flag_registry_is_current`, and registering them in PROBED would trip its stale-entry
  check, so they are deliberately NOT registered; B11-B19 are their probes.
  Contracts: B11-B19 in `tests/contracts/test_sd104_sd105_burst_decay_and_entropy_headroom.py`
  (all nine FAIL on the pre-build tree, pass on the build).
  Pure arithmetic, no RNG, no gradient -- phased training does not apply.
  Validation experiment: V3-EXQ-963c (successor to V3-EXQ-963b, not yet authored; authoring
  chipped as /queue-experiment work). See SD-104 record, MECH-063 (ii), SD-069.
