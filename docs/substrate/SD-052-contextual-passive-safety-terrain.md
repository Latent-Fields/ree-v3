## SD-052: Contextual Passive Safety Terrain (2026-05-04)
- SD-052 / MECH-303: safety_prediction.contextual_passive_substrate -- IMPLEMENTED 2026-05-04.
  Module: ree_core/residue/field.py (ResidueField -- extended with safety_terrain_rbf_field,
    accumulate_safety(), evaluate_safety()). No new file; extends existing ResidueField.
  Config: REEConfig.use_contextual_safety_terrain (bool, default False; set True to enable).
    ResidueConfig.safety_terrain_enabled (bool, default False; auto-set by from_dims when
    use_contextual_safety_terrain=True).
    Additional params: contextual_safety_accum_weight (float, default 0.01),
    contextual_safety_harm_threshold (float, default 0.05),
    contextual_safety_release_threshold (float, default 1.0). All defaults are no-op.
  Data flow: sense() -> z_harm_a.norm() < contextual_safety_harm_threshold AND
    hypothesis_tag=False -> residue_field.accumulate_safety(z_world, accum_weight) ->
    safety_terrain_rbf_field.add_residue() [incremental per-step].
    select_action() when beta_gate.is_elevated -> residue_field.evaluate_safety(z_world)
    -> if mean >= contextual_safety_release_threshold -> beta_gate.release() +
    _committed_step_idx=0 + _committed_anchor_keys=None.
  Architecture: follows same RBF pattern as benefit_terrain (ARC-030 / MECH-117).
    Separate safety_terrain_rbf_field -- no sharing with benefit_terrain or residue values.
    Biological analog: vmPFC + hippocampus (vHipp-to-PL). Slow/diffuse (0.01/step);
    contrasts with MECH-304 fast event-driven update.
  MECH-094 compliance: accumulate_safety(hypothesis_tag=True) returns immediately --
    safety accumulation is waking-only; simulation/replay ticks are silent.
  Backward compatible: disabled by default; safety_terrain_rbf_field not instantiated
    when disabled; all agent.py blocks guarded by getattr config check; bit-identical OFF.
  Phased training: not applicable (non-trainable accumulator).
  Validation experiment: V3-EXQ-520 queued 2026-05-04 (4-arm substrate-readiness diagnostic).
  See MECH-303 (the claim this SD implements), MECH-304 / SD-051 (cue-specific sister),
    MECH-302 / SD-050 (harm source monitored by harm_threshold gate), SD-011 (z_harm_a stream),
    ARC-007 (contextual encoding substrate), ARC-030 / MECH-117 (benefit terrain parallel),
    MECH-094 (simulation gate).
