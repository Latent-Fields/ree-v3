## SD-018 AMEND: encoder.resource_field_supervision (directional resource-field head) -- IMPLEMENTED (2026-09-02)
- SD-018 amend: encoder.resource_field_supervision -- IMPLEMENTED 2026-09-02. Design doc (new,
  covers the original scalar head and this amend):
  `REE_assembly/docs/architecture/sd_018_resource_proximity_supervision.md`.
  Routed by confirmed `failure_autopsy_V3-EXQ-948_2026-08-25` (H-observation-interface CONFIRMED,
  user-confirmed 2026-08-25; the one named un-owned build on the v3 critical path per
  `cross_plan_root_cause_synthesis_20260902.md` section 0 / GFLAG-0114). 948's finding: with the
  scalar SD-018 head ALREADY active (the x734/737/808/948 family base config), a PPO reader of
  z_world alone forages 0.5 res/ep against the 1.0 D3 floor, while the same reader given z_world +
  the full 25-dim `resource_field_view` clears it 3/3 (2.23). That field is `world_obs[225:250]`,
  i.e. already INSIDE z_world's own input -- z_world discards it. A scalar `max(field)` target
  supervises magnitude only; foraging needs the directional gradient.
  Shape chosen (skill Step 3, user-confirmed 2026-09-02, recommendation-ledger entry): (a) a
  DIRECTIONAL auxiliary head on z_world, generalising the scalar head from 1 to 25 dims -- NOT
  (b) routing the raw field as a side-channel past z_world (large blast radius on every
  world_dim consumer, and it bypasses the interface the synthesis says must carry direction
  through E1/E2 rollouts). (b) stays the fallback if (a)'s validation nulls.
  Config (LatentStackConfig; all three `from_dims` sites wired -- field, kwarg, assignment):
  `use_resource_field_head` (bool, default False), `resource_field_weight` (float, 0.5; online
  P1 loss weight), `resource_field_dim` (int, 25). P0: `ZWorldP0Config.resource_field_weight`
  (float, default 0.0 = leg off).
  SplitEncoder: `resource_field_head = Linear(world_dim, resource_field_dim) + Sigmoid`
  (field is max-normalised, values in [0,1]); `RESOURCE_FIELD_SLICE = slice(225, 250)`
  (mirrored by `zworld_p0.RESOURCE_FIELD_SLICE`, contract-pinned to the
  CausalGridWorldV2 `use_proxy_fields=True` layout). `forward()` now returns an 8-tuple
  (`..., resource_prox_pred, resource_field_pred`); the three in-module call sites updated.
  LatentState: `resource_field_pred` [batch, 25], None when off; carried through `detach()`.
  Agent: `compute_resource_field_loss(resource_field_target, latent_state)` -> MSE, zero-with-grad
  when off, loud ValueError on a width mismatch; same pass-the-sense()-LatentState rule as the
  scalar loss. P0: `ZWorldP0Trainer` derives the target from the buffered `world_obs` slice
  itself (no `observe()` signature change), trains the head jointly with the world path when
  `resource_field_weight > 0`, and reports `used_resource_field_head` + a held-out
  `resource_field_holdout` {mse, mean_predictor_mse, r2} readout.
  Data flow: world_obs[225:250] (target) -> resource_field_head(z_world) -> LatentState.
  resource_field_pred -> MSE -> backprop INTO world_encoder. No downstream consumer reads the
  prediction; the effect is on z_world itself (that is the hypothesis under test).
  Backward compatible: bit-identical OFF -- verified by a same-seed sense() hash against HEAD in
  a throwaway worktree (identical outputs, identical 51 state-dict keys). Phased training: yes,
  same P0/P1 rule as the scalar head. MECH-094: N/A (waking supervision, no replay write).
  ML note (skill Layer 7): plain auxiliary-task supervision; the hazard is a 25-dim target
  dominating P0 and collapsing a 32-dim z_world onto the field -- mitigated by the existing
  VICReg variance/covariance terms and the reconstruction head in P0; validation must report
  P0 participation ratio + held-out accuracies (already in the trainer stats).
  Contracts: `tests/contracts/test_sd018_resource_field_head.py` (C1 OFF inert incl.
  state-dict/param-order identity + P0 leg skipped at weight 0; C2 ON shape/range + grad reaches
  world_encoder; C3 from_dims plumbs all three knobs; C4 slice constants agree with the env layout
  + P0 leg learns a decodable field, held-out r2 > 0; C5 width mismatch is loud);
  `test_feature_flag_boot_matrix` row `resource_field_head`; `test_flag_inertness` probe
  `test_use_resource_field_head_populates_resource_field_pred_only_when_enabled` + PROBED entry.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; no dependent claim's
  measured mechanism changed. KEEP all evidence.
  Validation experiment: NOT queued here (per the routing chip); design reported for
  /queue-experiment -- 948-shape diagnostic, shared P0 with `resource_field_weight > 0` on the
  ON arm, arms = field head OFF vs ON (scalar head on in both, as in the x734 family), PPO reader
  of z_world alone, DV res/ep vs the 1.0 D3 floor on a seed majority, plus held-out linear decode
  of `resource_field_view` from z_world as the mechanism check; claims INV-088 + MECH-457
  read-across; if it nulls, (b) is the next build.
  See SD-018 (original), V3-EXQ-948, V3-EXQ-813, GFLAG-0114, INV-088, MECH-457, ARC-065,
  `conversion_ceiling_root` (H-observation-interface).

- SD-WAYPOINT-FIELD: environment.waypoint_proximity_field -- IMPLEMENTED 2026-09-04.
  `ree_core/environment/causal_grid_world.py` (env-only; no agent-side change). Full design:
  `REE_assembly/docs/architecture/sd_waypoint_proximity_field.md`. Routed by
  `chip-20260902-waypoint-proximity-field-observable` (campaign C3 item 2 of
  `evidence/planning/science_wave_campaign_plan_20260904.md`).
  Problem: in `subgoal_mode` waypoints reach the agent ONLY as entity-type channel 6 of the
  5x5x7 local view (radius 2), so a waypoint more than two cells away is absent from the
  observation entirely. Measured by V3-EXQ-977 (INV-086), adjudicated `blocked_substrate`
  2026-09-02: 3 waypoints, 12x12, 400 steps, `waypoint_visit_reward=0`, seeds 42/43/44 -- the
  agent's own policy visited 0/0/1 waypoints and completed 0 sequences, i.e. random walk. Every
  navigation-dependent DV is pinned at chance; 883/884 of the 27 `subgoal_mode` drivers script
  the walk and the 460-series relies on a ~25%-of-grid Chebyshev tolerance band.
  Config (constructor kwargs; env-only, NOT surfaced through `REEConfig.from_dims`, same
  convention as SD-065/SD-023): `waypoint_proximity_field_enabled` (bool, default False),
  `waypoint_field_decay` (float, 0.25; must be > 0).
  Kernel: `1 / (1 + decay * d)` with `d` Manhattan to `self.waypoints[self._next_waypoint_idx]`,
  torus-aware when `self.toroidal` (shortest wrap distance -- `_compute_proximity_fields()` uses
  plain Manhattan and would point the agent the long way round; this channel deliberately does
  not). Single source, so the field self-normalises: exactly 1.0 on the target, comparable across
  episodes and seeds, and at decay 0.25 the range on a 12x12 is 1.0 down to 0.154 at the diagonal
  (discriminable at long range, which the exponential reef/landmark kernels are not).
  Data flow: `self.waypoints[_next_waypoint_idx]` -> reciprocal-decay 5x5 patch centred on the
  agent -> `world_parts` (appended LAST, +25 dims) -> `world_state` -> SplitEncoder / z_world.
  No consumer change is owed: drivers already size the encoder from `env.world_obs_dim`.
  Computed ON DEMAND in `_get_observation_dict()`, not cached as a grid array like reef/landmark:
  the source MOVES (the pending index advances mid-step, `_respawn_waypoints()` relocates the
  set), so a cached field would be one tick stale exactly when the target changes. Sourced from
  `self.waypoints`, never `self.grid` -- a grid-sourced field would inherit the SD-094
  marker-erasure defect.
  Placement is load-bearing: appended last, after the SD-065 arm, with the matching `+25` last in
  `world_obs_dim`. `latent/stack.py` (`HAZARD_INDICES`, `CONTAMINATION_SLICE`,
  `RESOURCE_FIELD_SLICE = slice(225,250)`) and `latent/zworld_p0.py` are PREFIX slice constants
  pinned by `test_sd018_resource_field_head.py::test_c4_slice_constants_agree_with_env_layout`;
  inserting before index 250 would break them.
  Preconditions raise ValueError (loud-not-silent): the flag requires `use_proxy_fields` (the
  channel rides world_state) and `subgoal_mode` (`self.waypoints` is otherwise never populated,
  so the channel would be identically zero and the experiment would silently measure nothing);
  `waypoint_field_decay <= 0` is rejected (a constant kernel is not a gradient).
  Info sentinels (always present, inert OFF): `waypoint_proximity_field_enabled`,
  `waypoint_field_at_agent`, `waypoint_field_target_idx`.
  Driver note: the at-agent value NEVER reads 1.0 in normal play -- arrival and re-targeting
  happen in the same tick, so the arrival step's observation already points at the next waypoint.
  Use `transition_type` / `waypoint_field_target_idx` for arrival, the field for approach.
  Backward compatible: bit-identical OFF -- `world_obs_dim` unchanged (250 in proxy mode), obs key
  absent (mech090 absent-when-disabled precedent), zero env RNG draws, and with the flag ON the
  `world_state[:250]` prefix is byte-for-byte the OFF `world_state`. Setting the decay knob alone
  with the master switch off changes nothing.
  Phased training: N/A (environment observable; no encoder head, no loss). MECH-094: N/A (live
  exteroceptive observation; no memory write, no replay or simulated content).
  ML note (skill Layer 7): the problem is reward-free goal reachability under partial
  observability. Adopted a goal-distance OBSERVATION rather than potential-based reward shaping
  -- shaping would alter the reward channel and contaminate the valence/commitment/residue DVs
  the blocked experiments measure; and a bounded `1/(1+kd)` kernel so a moving source cannot
  destabilise encoder input scale. Deliberately NOT a learned goal embedding or attention
  read-out: this is an observability fix, and representational machinery here would confound the
  DVs it exists to free.
  Contracts: `tests/contracts/test_waypoint_proximity_field.py` (19 tests, C1-C8: C1 OFF inert
  + dim unchanged + knob-alone no-op; C2 three preconditions raise; C3 +25 exactly, trailing 25
  dims, ON prefix == OFF world_state; C4 kernel exact cell-by-cell, 1.0 on target, OOB cells 0.0;
  C5 DIRECTIONALITY -- patch monotone toward the target, at-agent series `0.5, 4/7, 2/3, 0.8` on
  approach, and a target far outside the 5x5 discriminable where the local view is identical;
  C6 re-points on index advance, all-zero with no pending target, reset/reset_to tag semantics;
  C7 RNG isolation over 60 steps; C8 toroidal seam).
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default env flag; no dependent claim's
  measured mechanism changed. KEEP all evidence.
  Validation experiment: NOT queued in the landing session -- `/governance`
  (`governance-20260904-1347`) held an exact-file pause on `experiment_queue.json`,
  `substrate_queue.json` and `claims.yaml`. Design + the drafted substrate_queue entry are in
  `REE_assembly/evidence/planning/c3_deferred_writes_20260904.md`: diagnostic-purpose, arms =
  flag ON vs OFF at matched seeds, DV = waypoints visited / sequences completed by the agent's
  OWN policy (no scripted walk), OFF-arm expectation the measured 0/0/1 visits and 0 completions,
  bar pre-registered inside the DV's measured range.
  See INV-086, MECH-428 (EXP-0390), V3-EXQ-977, SD-094, SD-023, SD-065, SD-018, SD-005.
