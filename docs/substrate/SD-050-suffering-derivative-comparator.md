## SD-050: Suffering-Derivative Comparator (2026-05-04)
- SD-050 / MECH-302: relief.suffering_derivative_comparator -- IMPLEMENTED 2026-05-04.
  Module: ree_core/comparator/suffering_derivative_comparator.py
    (SufferingDerivativeComparator -- non-trainable, no nn.Module inheritance).
  Config: REEConfig.use_suffering_derivative_comparator (default False; set True to enable).
    Additional params: suffering_window_length (default 5), suffering_drop_threshold
    (default 0.10), suffering_min_initial_norm (default 0.05),
    relief_completion_weight (default 1.0). All defaults are no-op.
  Data flow: sense() -> suffering_comparator.tick(z_harm_a.norm(), sim_mode)
    -> _relief_completion_event: bool (ephemeral flag, not in LatentState)
    -> select_action() -> beta_gate.release() + committed_step_idx reset
    + residue_field.update_valence(z_world, VALENCE_LIKING, relief_weight,
    hypothesis_tag=False) when valence_liking_enabled.
  Pipeline reuse: identical to MECH-057a goal-completion pipeline -- commitment
    release + MECH-094 categorical tag write -- triggered by suffering-stream
    descent rather than goal attainment. Architecturally adjacent to MECH-091
    urgency block in select_action(); opposite polarity.
  MECH-094 compliance: tick(simulation_mode=True) returns False without advancing
    the buffer (waking-path signal only; replay/DMN must not trigger events).
  Backward compatible: disabled by default; existing experiments unaffected;
    bit-identical OFF (no RNG draws; pure arithmetic).
  Phased training: not applicable (non-trainable).
  Validation experiments: V3-EXQ-515 PASS 2026-05-04 (comparator logic: 4-arm
    synthetic-norm unit diagnostic). V3-EXQ-516 queued (agent-loop integration:
    ARM_0 OFF backward-compat, ARM_1 event fires, ARM_2 valence write, ARM_3
    flat signal no false fires).
  See MECH-302, MECH-057a (commitment release pipeline reused), MECH-091
    (urgency block -- adjacent, opposite polarity), MECH-094 (simulation gate),
    SD-011 (z_harm_a source stream).
