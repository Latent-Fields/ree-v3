## SD-065: environment.conditioned_safety_cue_channel -- IMPLEMENTED (2026-07-14)
- SD-065: environment.conditioned_safety_cue_channel -- IMPLEMENTED 2026-07-14.
  Module: ree_core/environment/causal_grid_world.py (env-only; NOT surfaced through
  REEConfig.from_dims). The controllable, observable Pavlovian safety CS the SD-051
  ConditionedSafetyStore needs so MECH-304's promote-to-active BEHAVIOURAL falsifier
  can run (classical conditioned inhibition). Ambient (uniform-when-active) 25-dim
  field view appended LAST to world_state (feeds z_world, which the store keys on)
  + obs_dict["safety_cue_field_view"] (absent when disabled). Ambient not spatial so
  the cue's z_world contribution is position-independent -- reliably present at every
  relief tick and reproducible at test (removes a navigation confound). Adds NO
  entity type (NUM_ENTITY_TYPES stays 7; 5x5x7 local_view width fleet-invariant).
  Config (env kwargs): safety_cue_enabled (bool, default False -- master; grows
  world_obs_dim by exactly 25 when on) + safety_cue_scale (1.0, clipped [0,1]) +
  safety_cue_on_relief (False) + safety_cue_heal_floor (0.05). Two activation paths:
  (1) safety_cue_on_relief auto-activates the cue across the SD-022 damage->heal
  (MECH-302 relief) window -- active exactly when sum(limb_damage) > heal_floor; the
  store still writes its prototype ONLY on the real event_fired tick inside it, so
  the pairing is real, not synthesized; (2) set_safety_cue(active) tri-state manual
  override (test-phase API; takes precedence, persists across reset) to present
  {threat + cue} concurrently in one tick. Preconditions (loud-not-silent, mirroring
  SD-022): safety_cue_enabled requires use_proxy_fields; safety_cue_on_relief
  requires limb_damage_enabled (both raise ValueError). Info sentinels always present
  (safety_cue_enabled / safety_cue_active / safety_cue_event_count; 0/False OFF).
  Data flow: step() computes _safety_cue_active (manual override > relief window >
  inactive) -> world_state 25-dim view -> z_world -> ConditionedSafetyStore.update
  (SD-051) -> select_action beta_gate.release when signal > safety_store_threshold.
  MECH-094: DOES NOT APPLY at the env level (the MECH-094 sim-gate lives in the
  store.update sim_mode path, unchanged). Phased training: NOT REQUIRED (no new
  encoder head; env channel only). Backward compatible: safety_cue_enabled=False by
  default -- no channel, world_obs_dim unchanged, zero RNG draws, bit-identical OFF.
  10/10 new contracts in tests/contracts/test_scheduled_safety_cue_curriculum.py
  (off-inert + dim-unchanged / preconditions / geometry +25 / manual-override + scale
  clip / relief-pairing tracks damage window / reset clears dynamics + override
  persists / RNG isolation).
  Validation experiment: V3-EXQ-763 queued (MECH-304 promote-to-active behavioural
  falsifier; see below).
  Design doc: REE_assembly/docs/architecture/sd_065_conditioned_safety_cue_channel.md
  See SD-051 / MECH-304 (the consumer + the claim this makes testable), MECH-302 /
  SD-050 (relief teaching signal), MECH-303 (sister contextual-safety pathway; DV2
  sparing control), SD-022 (build-pattern precedent + the relief generator).
