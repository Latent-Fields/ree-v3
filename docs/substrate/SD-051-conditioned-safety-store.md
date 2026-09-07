## SD-051: Conditioned Safety Store (2026-05-04)
- SD-051 / MECH-304: safety_prediction.cue_specific_conditioned_inhibition_substrate -- IMPLEMENTED 2026-05-04.
  Module: ree_core/safety/conditioned_safety_store.py (ConditionedSafetyStore -- non-trainable,
    no nn.Module inheritance; pure arithmetic). New package ree_core/safety/__init__.py.
  Config: REEConfig.use_conditioned_safety_store (default False; set True to enable).
    Additional params: safety_store_ema_alpha (default 0.1), safety_store_decay_rate (default 0.001),
    safety_store_min_norm (default 0.1), safety_store_threshold (default 0.5),
    safety_store_commitment_weight (default 1.0). All defaults are no-op.
  Data flow: sense() -> z_world [world_dim] -> conditioned_safety_store.update(z_world,
    event_fired=_relief_completion_event, sim_mode=hypothesis_tag) -> _conditioned_safety_signal: float
    -> select_action(): when _conditioned_safety_signal > threshold AND beta_gate elevated ->
    beta_gate.release() + optional VALENCE_LIKING write (MECH-094).
  Encoding pathway (dorsal striatum / dlPFC analog): EMA prototype of z_world at MECH-302 event ticks.
    Per-step decay toward zero (safety_store_decay_rate) provides forgetting without reinforcement.
  Expression pathway (IL->CeA analog): cosine similarity between current z_world and prototype ->
    sigmoid -> safety_prediction scalar -> commitment-release gate.
  MECH-094 compliance: update(sim_mode=True) returns 0.0 without advancing the prototype
    (waking-path signal only; replay/DMN ticks are silent).
  Backward compatible: disabled by default; conditioned_safety_store is None when disabled;
    sense() and select_action() skip all SD-051 blocks entirely; bit-identical OFF.
  Phased training: not applicable (non-trainable).
  V4-deferred: (1) Approach attractor toward safety-signaling cues (requires V4 multi-step
    planning infrastructure, MECH-163 V3 completion gate). (2) Contrastive cue-specific
    learning (requires trainable encoder head + phased training; V3 prototype may
    over-generalise in stable-environment settings). See v3_v4_transition_boundary.md.
  Validation experiment: V3-EXQ-519 queued 2026-05-04 (4-arm substrate-readiness diagnostic).
  See MECH-304 (the claim this SD implements), MECH-303 (sister contextual pathway),
    MECH-302 / SD-050 (teaching signal source), MECH-057a (commitment-release pipeline reused),
    MECH-094 (simulation gate), SD-011 (z_harm_a source for MECH-302 event).
