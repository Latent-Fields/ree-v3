## SD-011 Second Source: Harm History Input (2026-04-08)
- SD-011 second source: harm_stream.affective_harm_history_input -- IMPLEMENTED 2026-04-08.
  AffectiveHarmEncoder (latent/stack.py) extended with harm_history input: rolling FIFO
  of past harm_exposure scalars from CausalGridWorldV2. Encoder input grows from
  harm_obs_a_dim to harm_obs_a_dim + harm_history_len when harm_history_len > 0.
  Auxiliary harm_accum_head (Linear+Sigmoid) predicts accumulated harm scalar, forcing
  z_harm_a to integrate temporal information that z_harm_s does not receive. This
  resolves the monotone redundancy confirmed by EXQ-241 (D3 reversal: z_harm_a predicted
  sensory target better than z_harm_s because both received the same spatial signal).
  Config: LatentStackConfig.harm_history_len (int, default 0; set 10 to enable).
  LatentStackConfig.z_harm_a_aux_loss_weight (float, default 0.1).
  CausalGridWorldV2 harm_history_len param (mirrors config; default 0).
  Data flow: env step() -> _harm_history FIFO -> obs_dict["harm_history"] ->
  agent.sense(obs_harm_history=...) -> encode(harm_history=...) ->
  AffectiveHarmEncoder(harm_obs_a, harm_history) -> z_harm_a + harm_accum_pred.
  Agent method: compute_harm_accum_loss(accumulated_harm_target, latent_state) -> loss.
  LatentState: harm_accum_pred field (Optional[Tensor], None when disabled).
  Backward compatible: harm_history_len=0 by default; existing experiments unaffected.
  Encoder hidden dim increased from 32 to 64 (input dim grew from 50 to 60).
  Phased training recommended but not strictly required (aux target is env scalar).
  MECH-094: not applicable (waking observation stream, not replay content).
  Validation experiment: V3-EXQ-241a queued (2-condition ablation, 3 seeds, ~60 min).
  See SD-011, MECH-112, ARC-030, ARC-032, MECH-029, Q-034.
