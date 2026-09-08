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
