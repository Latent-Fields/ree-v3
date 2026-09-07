## SD-029: Balanced Hazard-Event Curriculum (2026-04-21)
- SD-029: self_attribution.comparator_z_harm_s -- CURRICULUM-LEVEL BALANCED HAZARD-EVENT SUPPORT IMPLEMENTED 2026-04-21.
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorldV2).
  The substrate for the z_harm_s comparator (E2_harm_s forward model, ARC-033) and its
  interventional training (SD-013) already pass C1 (forward_r2 >= 0.9) and C2 (partial
  attenuation). The remaining blocker for C3/C4 (event-conditioned SNR on approach-to-harm
  events, with n_self >= 20 AND n_ext >= 20 per seed) is curriculum-level: the default env
  produces highly imbalanced hazard-event densities (some seeds near-zero self-caused,
  others near-zero externally-caused). This substrate change adds a scheduled
  externally-caused hazard injection curriculum to the env.
  Mechanism: when scheduled_external_hazard_enabled=True, every
  scheduled_external_hazard_interval steps, with probability scheduled_external_hazard_prob,
  an existing hazard is moved (or new one spawned) to a cell adjacent to the agent
  (or any empty cell when adjacent_only=False). Purely curriculum-level: the agent did
  not initiate the encounter, so subsequent harm is externally-caused in the
  self-vs-externally-caused taxonomy. Agent and latent code unchanged.
  Config (CausalGridWorldV2 __init__): scheduled_external_hazard_enabled (bool,
  default False -- no-op); scheduled_external_hazard_interval (int, default 50);
  scheduled_external_hazard_prob (float, default 0.5);
  scheduled_external_hazard_adjacent_only (bool, default True -- if no empty neighbour
  and False, falls back to any empty cell; if True, a tick with no adjacency is skipped).
  New env state and info keys:
    self._external_hazard_event_count: per-episode counter (reset in reset()).
    info["external_hazard_injected"]: bool, True on the step the injection fired.
    info["external_hazard_event_count"]: int, cumulative this episode.
  Data flow: step() -> after agent move / env drift checks -> [enabled and steps%interval==0
  and rng<prob] -> _inject_external_hazard() -> hazard relocated/spawned ->
  info tags set -> proximity fields recomputed.
  Backward compatible: scheduled_external_hazard_enabled=False by default; env state
  is unchanged relative to legacy behaviour (_drift_hazards, _respawn_resource unaffected).
  Info dict tags are always present (value 0 / False when disabled), but existing
  experiments that don't read them are unaffected.
  Biological basis: none required (curriculum-level env augmentation). The distinction
  this supports (self-caused vs externally-caused harm events) is a prerequisite of the
  Blakemore / Shergill / Frith comparator literature already grounding SD-029.
  Phased training: not applicable (env only; no new trainable parameters).
  MECH-094: not applicable (env observation stream, not replay content).
  Validation experiment: V3-EXQ-470 queued (diagnostic ablation:
  SCHEDULED vs BASELINE, 4 seeds, confirms per-seed n_ext >= 20 under the curriculum
  and that balanced event counts preserve C1/C2 while enabling C3/C4 measurement).
  See SD-029, MECH-256, ARC-033, SD-013.
  Design-doc reference: REE_assembly/docs/architecture/self_attribution_per_stream.md.
