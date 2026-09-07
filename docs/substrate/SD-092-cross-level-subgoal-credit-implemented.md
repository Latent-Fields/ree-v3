## SD-092: Cross-Level Subgoal Credit -- IMPLEMENTED (primitive + agent-loop call site, 2026-08-02)
- SD-092: goal.cross_level_subgoal_credit -- primitive IMPLEMENTED 2026-08-02; agent-loop call
  site (`REEAgent.notify_subgoal_attainment`, commit `d9e05864a1`) landed same day. Serves
  MECH-427 (`cross_level_subgoal_credit`, maintenance-direction) and
  MECH-428 (`subgoal_bootstrapped_goal_seeding`, formation-direction) -- both were confirmed
  `blocked_substrate` on exactly this gap on 2026-08-02 (zero hits anywhere in `ree_core` for
  `MECH-427`/`parent_goal`/any credit-propagation-to-parent code; `GoalState` held exactly one
  attractor).
  `ree_core/goal.py` `GoalState` + `GoalConfig`. Config: `GoalConfig.use_hierarchical_goal_credit`
  (default `False`, bit-identical off -- no parent tensor allocated, no extra branch entered in
  `update()`), `parent_goal_alpha` (0.05, mirrors `alpha_goal`), `parent_goal_decay` (0.005,
  mirrors `decay_goal`), `subgoal_credit_min` (0.0, no-op gate default).
  New state: `GoalState._z_goal_parent` (`None` unless enabled -- mirrors SD-057's
  `IncentiveTokenBank` None-unless-enabled pattern), new method
  `credit_subgoal_attainment(child_representation, credit=1.0)` -- EMA-pulls the parent toward a
  caller-supplied child representation, pull fraction `min(1.0, parent_goal_alpha * credit)`; new
  readouts `parent_goal_norm()` / `z_goal_parent` property / `parent_is_active()`; parent decay
  added to `update()` (gated identically to the flag); `reset()` clears the parent per-episode
  (distinct from MECH-189's cross-episode `SuperOrdinalGoalMemory`, which is untouched);
  `state_dict()`/`load_state_dict()` extended, backward-compatible (a pre-SD-092 checkpoint dict
  missing the new keys loads fine, parent left unallocated); `with_injection()` (MECH-188's
  lightweight view) carries the parent state through so it does not raise `AttributeError`.
  Data flow (once a call site is wired -- **NOT this pass**): subgoal-attainment event ->
  caller-chosen representation (e.g. `causal_grid_world.py` `subgoal_mode`'s waypoint/
  sequence_complete signal, or the agent's own settled child-level `z_goal`) ->
  `credit_subgoal_attainment()` -> `_z_goal_parent` -> `parent_goal_norm()` (the metric both
  EXP-0385 and EXP-0390 name as their primary DV).
  MECH-427 (an already-seeded parent gets reinforced) and MECH-428 (a near-zero parent gets
  bootstrapped) are the **same call** against different starting states of `_z_goal_parent` -- no
  separate formation-mode code path exists or was needed.
  Backward compatible: disabled by default; every existing `GoalState` consumer (there are many --
  `agent.py`, `e3_selector.py`, `hippocampal/module.py`, etc.) is unaffected by construction, since
  every new branch is gated on the same flag, not merely on the parent tensor's existence.
  Not a learning module: no `nn.Module`, no trainable parameters, no phased training. MECH-094 N/A
  (no simulation/replay content; whatever a future call site credits carries the same status the
  existing single-level `z_goal` `update()` already has).
  **What remains:**
  (1) ~~no call site in `agent.py`~~ DONE 2026-08-02: `REEAgent.notify_subgoal_attainment(
  transition_type, child_representation=None, credit=1.0)` mirrors the `notify_env_completion`
  (SD-034) explicit-hook convention -- the harness calls it right after `env.step()` with
  `info["transition_type"]`; no-op unless `use_hierarchical_goal_credit` is on. Contracts:
  `tests/contracts/test_sd092_notify_subgoal_attainment.py` (C1-C6, all pass).
  (2) still no AUTOMATIC environment wiring, by design (mirrors `notify_env_completion`: the env
  stays agnostic of `GoalState`) -- `causal_grid_world.py`'s `subgoal_mode` waypoint/
  sequence_complete events reach the hook only when an `experiments/` driver reads
  `info["transition_type"]` and calls the hook; that driver-side wiring + representation choice
  is `/queue-experiment` Step-2.5's job (V3-EXQ-883 does this for EXP-0385/MECH-427);
  (3) no `REEConfig.from_dims` passthrough -- the new `GoalConfig` fields are set directly
  (`cfg.goal.use_hierarchical_goal_credit = True`), matching how
  `test_goalstate_forced_seed_positive_control.py` already exercises `GoalConfig` directly and
  avoiding the "from_dims swallows unknown kwargs" hazard for a consumer that does not exist yet.
  Deciding which representation to pass as `child_representation` (raw env waypoint z_world vs.
  the agent's settled child-level z_goal at attainment) is an experiment-design decision for
  `/queue-experiment`'s Step-2.5, left open by design (the primitive's signature is agnostic).
  Contracts: `tests/contracts/test_sd092_cross_level_subgoal_credit.py` (19 tests, all pass) --
  flag-off bit-identical/no-op regression guard (R1); zero-credit baseline (R2); MECH-428
  bootstrap shape vs. a matched no-credit control (R3); MECH-427 maintenance/reinforcement shape
  vs. a matched no-further-credit control (R4); credit gating on `credit<=0` and
  `subgoal_credit_min` (R5); parent decay between credit events (R6); independence -- crediting
  the parent never perturbs the child-level `_z_goal`/`goal_norm()` (R7); `reset()` clears parent
  state + counters (R8); `state_dict`/`load_state_dict` round-trip + pre-SD-092 checkpoint
  backward-compat (R9); `with_injection()` carries parent state without raising (R10); lazy
  allocation when the flag is flipped on after construction (R11).
  Validation experiments: EXP-0385 (MECH-427) and EXP-0390 (MECH-428) flipped
  `blocked_substrate` -> `proposed` in `experiment_proposals.v1.json` on the basis that the
  primitive they were blocked on now exists and is tested -- neither is queued yet
  (`/queue-experiment` Step-2.5 still needs to design/verify the harness-side attainment-event
  call site before either can run). `substrate_queue.json` SD-092 entry added, status
  `partially_implemented_pending_consumer_wiring` (mirrors SD-091's status convention for the
  identical "primitive built, consumer wiring deferred" shape).
  See MECH-427, MECH-428, INV-086, ARC-051, MECH-217 (the within-level precedent this SD's
  call/gate/return-dict style mirrors), MECH-112, MECH-230,
  `REE_assembly/docs/architecture/sd_092_cross_level_subgoal_credit.md`.
