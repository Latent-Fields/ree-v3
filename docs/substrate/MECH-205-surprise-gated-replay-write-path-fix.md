## MECH-205: Surprise-Gated Replay Write Path Fix (2026-04-09)
- MECH-205: hippocampal.surprise_gated_generative_replay -- WRITE PATH FIXED 2026-04-09.
  Tier 1 implementation (2026-04-07) wired PE EMA tracking + VALENCE_SURPRISE write in
  agent.py update_residue(). EXQ-258 FAIL (P1: surprise_tag_populated=False) had two
  root causes: (1) experiment script checked nonexistent `_rbf_layer` attr (should be
  `rbf_field`); (2) pe_ema_alpha=0.1 tracked PE so fast that surprise stayed near zero.
  Fix: pe_ema_alpha moved from hardcoded 0.1 to config (default 0.02, ~50-step window).
  Added pe_surprise_threshold (default 0.001) gate before update_valence(). Added
  _surprise_write_count diagnostic counter + mech205_write_count metric.
  Config: REEConfig.pe_ema_alpha (float, default 0.02), REEConfig.pe_surprise_threshold
  (float, default 0.001). Both wired through from_dims().
  Data flow: E3.post_action_update() -> e3_metrics["prediction_error"] -> PE EMA ->
  surprise = max(0, pe_mag - pe_ema) -> [gate: > threshold] ->
  ResidueField.update_valence(z_world, VALENCE_SURPRISE, surprise).
  Backward compatible: surprise_gated_replay=False by default; write block never entered.
  No trainable parameters. No phased training needed.
  MECH-094: not applicable (waking observation stream, not replay content).
  Validation experiment: EXQ-258a queued.
  See MECH-205, INV-052 (indirect).
