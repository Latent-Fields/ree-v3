## SD-017: Minimal Sleep-Phase Infrastructure -- SWS/REM Passes (2026-04-09)
- SD-017: sleep_phase.minimal_sleep_infrastructure_v3 -- SWS-ANALOG + REM-ANALOG IMPLEMENTED 2026-04-09.
  Two new first-class methods added to REEAgent (ree_core/agent.py):
  (1) run_sws_schema_pass(): SWS-analog schema installation (hippocampus-to-cortex direction).
      Samples diverse z_world prototypes from _world_experience_buffer (stratified across
      buffer history), constructs [z_self, z_world] E1 input, writes to ContextMemory
      bypassing the offline gate (offline gate blocks waking obs; schema writes are
      intentional offline content). Returns: sws_n_writes, sws_slot_diversity (mean pairwise
      cosine distance of ContextMemory slots -- higher = more differentiated), sws_buffer_size.
  (2) run_rem_attribution_pass(): REM-analog attribution replay (slot-filling, MECH-166).
      Replays recent theta_buffer content via hippocampal.replay() (forward) and
      hippocampal.diverse_replay(mode="reverse") (reverse/ARC-045 bidirectional proxy).
      Evaluates residue terrain per trajectory without accumulating new residue
      (hypothesis_tag=True per MECH-094). Returns: rem_n_rollouts, rem_mean_harm_terrain,
      rem_terrain_variance, rem_n_reverse.
  (3) run_sleep_cycle(): Convenience method running SWS then REM in sequence with correct
      mode transitions (enter_sws_mode -> run_sws_schema_pass -> exit_sleep_mode ->
      enter_rem_mode -> run_rem_attribution_pass -> exit_sleep_mode). Returns merged metrics.
  Config (REEConfig, ree_core/utils/config.py):
      sws_enabled (bool, default False), sws_consolidation_steps (int, default 5),
      sws_schema_weight (float, default 0.1), rem_enabled (bool, default False),
      rem_attribution_steps (int, default 10). All wired through REEConfig.from_dims().
  Backward compatible: all switches default False; existing experiments unaffected.
  No trainable parameters. No gradient flow in pass bodies. No phased training needed.
  Prerequisites satisfied: MECH-092 (waking quiescent replay), MECH-120 SHY wiring
  (enter_sws_mode calls shy_normalise), serotonin module (MECH-203/204), enter_offline_mode.
  Distinguishes from EXQ-242: EXQ-242 used proxy hooks (standalone functions, non_contributory).
  This implementation adds first-class REEAgent methods experiments can call directly.
  MECH-094: hypothesis_tag=True in rem_attribution_pass (terrain scoring only; no residue writes).
  Validation experiment: V3-EXQ-265 queued (SD-017 activation + slot differentiation ablation,
  2 conditions x 3 seeds, ~45 min on Mac).
  See SD-017, ARC-045, MECH-166, MECH-120 (SHY gated within enter_sws_mode).
