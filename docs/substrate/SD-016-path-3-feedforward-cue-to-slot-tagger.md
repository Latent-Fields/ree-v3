## SD-016 Path 3: Feedforward cue->slot tagger (2026-06-05)
- SD-016 Path 3: e1.cue_slot_tagger -- IMPLEMENTED 2026-06-05.
  Module: ree_core/predictors/e1_deep.py (E1DeepPredictor.__init__ +
  extract_cue_context), ree_core/utils/config.py (E1Config + from_dims).
  Root cause it addresses (V3-EXQ-418i, the recommended div-weight sweep at
  1.0/2.0/5.0): Path 1 (auxiliary diversification loss) is "insufficient
  regardless of weight; the attention bottleneck is categorically in query
  selectivity, not slot orthogonality." The z_world-only q.k attention in
  extract_cue_context (e1_deep.py) is pinned at the uniform ln(num_slots)
  saddle because key_proj(memory) with memory init 0.01 yields near-identical
  keys -> softmax stays uniform -> the softmax Jacobian at uniform is a flat
  saddle that the cue_terrain_proj terrain_loss gradient cannot escape.
  THE FIX: a fresh feedforward MLP cue_slot_tagger (Linear(world_dim, hidden)
  -> ReLU -> Linear(hidden, num_slots)) replaces ONLY the slot-SELECTION
  scores; the slot-CONTENT path (value_proj -> output_proj -> cue_context)
  and both downstream projections (cue_action_proj retaining the 449a z_world
  concat band-aid; cue_terrain_proj) are untouched. A random MLP produces
  non-uniform logits from step 0, so it sits OFF the saddle and the existing
  terrain_loss gradient flows back into it and shapes contextual selectivity.
  No new supervised target is invented -- the tagger is a better-conditioned
  replacement for the saddle-stuck attention; the gradient source is the same
  terrain_loss that already trains cue_terrain_proj.
  Config (E1Config + REEConfig.from_dims; all no-op defaults):
    sd016_cue_slot_tagger (bool, default False) -- master switch (requires
      sd016_enabled=True; when False the legacy q.k attention branch runs
      verbatim, bit-identical).
    sd016_cue_slot_tagger_hidden (int, default 32) -- tagger MLP hidden width.
    sd016_cue_slot_tagger_temperature (float, default 1.0) -- softmax temp on
      the tagger logits (selectivity-sharpness knob; sweepable at eval without
      retraining).
  Data flow (Path 3 ON): z_world -> cue_slot_tagger -> slot_logits[num_slots]
    -> softmax(logits/temperature) [REPLACES bmm(q,k)/scale softmax] ->
    weights @ value_proj(memory) -> output_proj -> cue_context (unchanged) ->
    cue_action_proj([cue_context, z_world]) + cue_terrain_proj(cue_context).
  Read-only diagnostic: extract_cue_context caches the last selection
    distribution on E1DeepPredictor._last_cue_slot_weights [batch, num_slots]
    so a validation experiment can measure selection entropy (the V3-EXQ-418i
    bottleneck metric: uniform == ln(16) ~ 2.773).
  Backward compatible: sd016_cue_slot_tagger=False by default ->
    cue_slot_tagger is None and extract_cue_context takes the legacy attention
    branch (verified bit-identical: OFF selection entropy == ln(16) exactly).
    7/7 preflight + predictor-subsystem contracts PASS; 5/5 new contracts in
    tests/contracts/test_sd016_cue_slot_tagger.py PASS.
  Smoke (2026-06-05): OFF entropy 2.7726 (== ln16, on the saddle); ON entropy
    2.7407 at init (off the saddle) with per-context std 0.0098; terrain_loss
    backprops into the tagger (grad-sum > 0); under terrain SGD the tagger
    drives selection entropy toward 0 and fits the terrain target. (A toy with
    cleanly-separable clusters does NOT discriminate Path 3 from legacy; the
    real env -- where V3-EXQ-418i measured the legacy attention stuck at ~2.76
    across training -- is the discriminator. The validation EXQ runs there.)
  Phased training: NOT required -- the tagger trains jointly with cue_terrain_proj
    on the same terrain_loss (compatible objective, same gradient path).
    Experiments enabling it include terrain_loss in their E1 loop (the
    established SD-016 pattern, EXQ-182/187a/194).
  MECH-094: N/A -- extract_cue_context is a waking E1 query (_e1_tick); no
    replay/simulation write surface.
  HONEST SCOPE: Path 3 restores RETRIEVAL (cue_context) selectivity. Full
    behavioural action_bias_div >= 0.05 propagation ALSO depends on
    cue_action_proj, whose gradient path is the separate SD-055 differentiable-
    CEM / ARC-065 concern. The validation measures action_bias_div as a
    secondary/diagnostic, gated on SD-055.
  Validation experiment: V3-EXQ substrate-readiness diagnostic (claim_ids=[];
    OFF vs ON ablation; PRIMARY acceptance = mean selection entropy < 2.5 vs
    the pinned ln(16)=2.773 with the tagger ON; SECONDARY diagnostic =
    cue_context per-channel std + safe-vs-dangerous action_bias_div). Queued
    via /queue-experiment.
  Design doc: REE_assembly/docs/architecture/sd_016_writepath_v3_diversification_loss.md
    (Path 3 section, appended 2026-06-05).
  See SD-016, MECH-150 / MECH-151 / MECH-152, ARC-041, V3-EXQ-418i (the
    div-weight sweep that exhausted Path 1 + named the query-selectivity
    bottleneck), V3-EXQ-449a/449b (the constant-output fix + the residual
    contextual-selectivity gap this addresses), SD-016 Path 1 (exhausted
    auxiliary-diversification approach above), SD-055 (differentiable CEM;
    the cue_action_proj behavioural-propagation half), EXP-0155 (the original
    cue_action_proj forward-path diagnostic), MECH-094 (N/A -- waking query).
