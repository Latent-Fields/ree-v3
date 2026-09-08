## SD-082: pfc.lateral_pfc.rule_selection_action_consumer -- IMPLEMENTED (2026-07-26)
- SD-082: pfc.lateral_pfc.rule_selection_action_consumer -- IMPLEMENTED 2026-07-26.
  ree_core/pfc/lateral_pfc_analog.py (LateralPFCAnalog.compute_bias + __init__).
  Config: LateralPFCConfig.rule_readout_consumer (default False; set True to enable) +
  readout_init_scale (default 0.25); REEConfig.lateral_pfc_rule_readout_consumer /
  lateral_pfc_readout_init_scale plumbed through all three from_dims sites and read in
  agent.py's LateralPFCConfig build (getattr-fallback).
  Data flow: SD-078 centered CandidateRuleField -> differentiated rule_state ->
  LateralPFCAnalog.rule_state -> compute_bias: (i) subtract the common mode from the
  per-candidate z_world summaries (when K>=2) so the SD-008 ~0.98-cosine cone no longer
  saturates every candidate to the same rail, then (ii) bound with bias_scale*tanh(raw/
  bias_scale) instead of a hard clamp -> E3 per-candidate score_bias.
  WHY: V3-EXQ-822 found the differentiated rule_state (on_rule_state_diff 0.644) was
  behaviourally SILENT -- propagation to the action bias was exactly 0.0 on BOTH arms.
  Root cause (reproduced): the hard clamp on the raw common-mode-dominated summaries maps
  every candidate to the identical rail, so zeroing rule_state changes nothing (structural
  zero) AND the clamp's flat region has zero gradient so REINFORCE cannot train the head
  (the observed 70ep null). Centering de-saturates (robust prop) + tanh restores the
  gradient (grad-norm 0.0 hard-clamp -> ~6.1 soft-tanh). Same magnitude bound
  (|bias| < bias_scale), so the SD-033a "bias cannot dominate E3" guarantee is preserved.
  Backward compatible: disabled by default; OFF path is bit-identical to the SD-033a
  landing (hard clamp on raw input) -- verified torch.allclose, and the existing
  v3_exq_822 script (flag absent) still reproduces prop=0.0 on both arms.
  Biological basis: corticostriatal rule-to-action mapping -- selection without a trained
  read-out to action is inert. Completes the SD-033a signature-(iv) trained-head variant
  (DESIGN ALTERNATIVE A2) that the landing deferred.
  Phased training required: yes (P0 warmup, P1 frozen-encoder+CRF bias-head REINFORCE, P2
  eval -- the existing 822 P0/P1/P2 protocol). MECH-094 N/A (pure forward read; no memory
  write; rule_state is MECH-261 gate-protected).
  NOT captured by test_flag_registry_is_current (name has the lateral_pfc_ prefix, not a
  use_*/`*_enabled` name -- same convention as lateral_pfc_train_rule_bias_head).
  Validation experiment: V3-EXQ-822a queued (re-run of 822, same question, consumer flag
  ON on both arms). Acceptance: on_prop_delta_mean >= 0.001 with an ON>OFF contrast.
  See SD-033a, SD-078, ARC-063 (GAP-B/GAP-D), SD-008, SD-066/SD-077,
  REE_assembly/docs/architecture/sd_082_rule_selection_action_consumer.md.
