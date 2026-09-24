## SD-092 Residual: Parent-Attractor E3 Scoring Consumer (2026-09-24)

- SD-092 (residual): e3.parent_goal_score -- IMPLEMENTED 2026-09-24.
  `ree_core/predictors/e3_selector.py` `E3TrajectorySelector.score_trajectory` (new additive term)
  + `compute_parent_goal_score`; `ree_core/goal.py` `GoalState.parent_goal_proximity`.
  Config: `E3Config.parent_goal_weight` (default `0.0`; set `> 0` to enable). `from_dims` now also
  passes through the SD-092 GoalConfig knobs `use_hierarchical_goal_credit`, `parent_goal_alpha`,
  `parent_goal_decay`, `subgoal_credit_min` (previously deferred "until a consumer exists"), so an
  experiment can arm both halves; the canonical builders (`large`, goal-stream presets) forward
  `**kwargs` to `from_dims`.
  Problem: `GoalState._z_goal_parent` (MECH-427 cross-level subgoal credit) had no reader in
  ree_core; E3's only goal channel was child-only (`compute_goal_score` -> `goal_proximity` reads
  `_z_goal`), so restoring or ablating cross-level credit was DV-invariant by construction
  (V3-EXQ-977 / INV-086 blocked_substrate; EXP-0710 build (b); SD-092's own EXP-0381).
  Form: `score -= parent_goal_weight * sum_t 1/(1 + MSE_sum(z_t, _z_goal_parent))`, terrain-scaled by
  `w_goal` like the child term. ADDITIVE -- not a blend inside `goal_proximity`, whose many other
  callers (SD-093 progress, dACC goal readout, ...) must not move. Gated on its own weight AND
  `goal_state.parent_is_active()`, deliberately NOT inside the child `is_active() and goal_weight > 0`
  block, so it stays live when the child attractor is inactive or `goal_weight=0` (MECH-428
  formation regime). Outside `_COMMENSURABILITY_CHANNELS` (documented in its DELIBERATELY EXCLUDED
  note); enters the score unscaled. `with_injection()` views already carry the parent state.
  Data flow: `credit_subgoal_attainment` (harness via `REEAgent.notify_subgoal_attainment`) ->
  `_z_goal_parent` -> `parent_goal_proximity` over each candidate's z_world rollout -> E3 score.
  Backward compatible: weight 0.0 -> the branch is skipped -> bit-identical.
  Liveness (real rollout, CausalGridWorldV2 8x8, z_goal_enabled, goal_weight 1.0, parent credited
  once from a real z_world, untrained): ON-OFF score delta has cross-candidate range 0.0356 mean
  (min 0.0330) over 39 ticks -- not a uniform shift; E3 argmin flips 0/39 at weight 1.0 (untrained
  agent), so behavioural reach at a trained operating point is NOT yet shown.
  Contract: `tests/contracts/test_sd092_parent_goal_e3_consumer.py` (parent != child fixture; the
  range-and-argmin test FAILS on the pre-change tree with "parent term is a uniform shift").
  Phased training required: no. MECH-094: not applicable (waking scoring only).
  Validation: OWED -- EXP-0710 (INV-086) build (b) release: an arm contrast restoring vs ablating
  cross-level credit with `parent_goal_weight > 0` and `use_hierarchical_goal_credit=True`, with the
  harness calling `notify_subgoal_attainment`; the DV must be committed-candidate change, and the
  parent must be checked to differ from the child attractor (it is an EMA of credited z_world and
  can sit near it). Not queued by this build.
  See SD-092, MECH-427, MECH-428, INV-086, chip-20260902-zgoal-parent-e3-consumer.
