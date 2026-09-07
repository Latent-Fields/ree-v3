## SD-093: Progress-Velocity Effort/Persistence Modulation -- IMPLEMENTED (2026-08-02)
- SD-093 (architecture) / MECH-426 (mechanism): goal.progress_velocity_maintenance --
  IMPLEMENTED 2026-08-02 (chip `chip-20260802-mech426-velocity-substrate`). Unlike SD-091 above,
  this is a COMPLETE minimal landing -- all three pieces (velocity computation, E3 consumption,
  agent.py recording call site) are wired end-to-end; nothing further is required before
  `/queue-experiment` can queue EXP-0384.
  `ree_core/goal.py` (`GoalState`): a bounded rolling-window deque
  (`maxlen=progress_velocity_window`, default 5) of `goal_proximity()` readings, appended by
  `record_progress(z_world)`. `progress_velocity` = `(newest - oldest) / (window_length - 1)` --
  the temporal derivative of the on-path progress estimate (Carver & Scheier 1990 second-order
  "velocity" control loop). `progress_velocity_effort_modulation` (a property) =
  `clip(-progress_velocity_effort_gain * progress_velocity, -progress_velocity_effort_max,
  +progress_velocity_effort_max)`.
  Config (`GoalConfig`): `use_progress_velocity_effort_modulation` (bool, default `False`,
  master switch), `progress_velocity_window` (int, default `5`, clamped >= 2),
  `progress_velocity_effort_gain` (float, default `1.0`), `progress_velocity_effort_max` (float,
  default `0.3`, symmetric saturation cap).
  Data flow: `agent.py._e3_tick()`'s one-shot `z_world_for_e3` (theta-buffer summary, the SAME
  signal `_compute_persistence_appraisal` reads -- NOT the per-candidate trajectory rollouts
  `compute_goal_score()` scores) -> `GoalState.record_progress()` (once per E3 tick, gated on
  `goal_state.is_active()`) -> `progress_velocity_effort_modulation` ->
  `E3TrajectorySelector.select()` (`ree_core/predictors/e3_selector.py`), consumed immediately
  after the existing SD-011 `urgency_applied` block:
  `effective_threshold = effective_threshold * (1.0 + velocity_effort)`.
  Sign convention (CRITICAL, verified against this method's own commit rule `committed = variance
  < effective_threshold`, where a HIGHER threshold is MORE permissive): `velocity > 0`
  (approaching the goal faster than the implicit zero reference rate) ->
  `effort_modulation < 0` -> LOWERS `effective_threshold` (stricter bar, more readily kicked back
  into deliberation) -- licenses coasting/redeployment. `velocity < 0` (falling behind) ->
  `effort_modulation > 0` -> RAISES `effective_threshold` (more permissive, locks in and pushes
  through) -- boosts persistence. This is deliberately the OPPOSITE of "positive progress ->
  bonus": per the claim's own Carver & Scheier coasting caveat, a same-goal-commitment bonus on
  positive velocity would invert the theory. `compute_goal_score()`/`goal_proximity()` (goal
  VALUE / trajectory score) are completely untouched by this flag -- verified byte-identical by
  contract test, not merely asserted.
  `with_injection()` (the MECH-188 `z_goal_inject > 0` scoring-only wrapper `E3.select()`
  actually receives whenever that pre-existing, independently-gated feature is active) now
  propagates `_progress_history`/`_progress_velocity` into the injected view -- previously it did
  not copy this (nor `_drive_trace`), which would have silently zeroed the modulation under that
  combination. `reset()` clears both -- per-episode state, mirroring the existing `_drive_trace`
  reset (a stalled-vs-progressing read from a PRIOR episode must not leak into a fresh one).
  Backward compatible: `use_progress_velocity_effort_modulation=False` (default) -> `record_progress()`
  is a true no-op (history never populated, returns `0.0`) and `progress_velocity_effort_modulation`
  always returns `0.0` -> `select()`'s modulation branch is never entered -> bit-identical to every
  existing run. Verified by running `experiments/v3_exq_869a_...py --dry-run` unchanged, plus 84
  existing contracts across `ree_core/goal.py` / `e3_selector.py` consumers (drive-floor, incentive
  bank, super-ordinal anchors, cue recall, natural-commit-urgency, conflict-graded conversion, dead
  z_goal stream lint, frozen z_goal scaffold, goal-stream config wiring, MECH-266/321/353), all pass.
  Not a learning module -- pure rolling-window arithmetic and a deterministic clip function, no
  `nn.Module`, no parameters, no phased training. MECH-094 N/A -- `record_progress()` records only
  the WAKING-tick `goal_proximity()` reading from `_e3_tick` (the `select_action` path); it is not
  a simulation/replay content write to any memory store.
  Distinct from MECH-217 (`goal.replay_wanting_spread`, an offline/training-time credit-assignment
  mechanism over the same `GoalState`, already implemented 2026-07-30) and from MECH-340/Q-053's
  `PersistenceAppraisal` (`ree_core/hippocampal/persistence_appraisal_compute.py`, a separate,
  independently-gated `GhostGoalBank` license computation also reading instantaneous
  `goal_proximity`) -- neither is modified by this landing.
  Contracts: `tests/contracts/test_mech426_progress_velocity.py` (18 tests, all pass) -- config
  defaults (C1); flag-off true no-op (C2); velocity magnitude against the documented
  rolling-window arithmetic on both an approach and a recede sequence (C3); effort-modulation
  sign in both directions, explicitly guarding against the theory-inverting bug (C4); saturation
  cap (C5); `with_injection()` propagation (C6); `reset()` clears velocity state (C7);
  `E3TrajectorySelector.select()` wiring -- stalling raises `effective_threshold` (commits MORE
  readily at a fixed variance), coasting lowers it (commits LESS readily), flag-off and
  inactive-goal_state are both inert (C8); `goal_proximity()`/`goal_distance()` byte-identical
  regardless of the flag (C9).
  EXP-0384 reset `blocked_substrate` -> `proposed` in
  `REE_assembly/evidence/planning/experiment_proposals.v1.json` (`resolved_note` added, mirroring
  the MECH-423/EXP-0384-prior convention). No `substrate_queue.json` entry added -- unlike SD-091,
  this item was never registered there as a pending queue entry (it was tracked only via
  `experiment_proposals.v1.json`'s `blocked_substrate` status), and since the landing is complete
  (no remaining steps), there is nothing left to track.
  Concurrency note: built via an isolated throwaway `git worktree` on both `ree-v3` and
  `REE_assembly` (rather than editing the live, actively-dirty shared checkouts directly) because
  a concurrent sibling chip (`chip-20260802-mech427-parent-subgoal-credit`, SD-092) was
  simultaneously editing `ree_core/goal.py` / `claims.yaml` / `experiment_proposals.v1.json` for a
  genuinely different mechanism (parent/subgoal credit propagation) -- see TASK_CLAIMS.json
  `--allow-overlap` claim `insights-34f9b4-mech426`.
  See MECH-426, SD-093, `REE_assembly/docs/architecture/sd_093_progress_velocity_maintenance.md`,
  INV-086, INV-034, MECH-217, MECH-116, MECH-340/Q-053.
