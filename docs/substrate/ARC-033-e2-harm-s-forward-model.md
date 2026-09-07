## ARC-033: E2_harm_s Forward Model (2026-04-09)
- ARC-033: harm_stream.sensory_discriminative_forward_model -- IMPLEMENTED 2026-04-09.
  E2HarmSForward in ree_core/predictors/e2_harm_s.py. f(z_harm_s_t, a_t) -> z_harm_s_pred_{t+1}.
  Wraps ResidualHarmForward (ree_core/latent/stack.py) -- residual delta architecture
  avoids identity collapse on autocorrelated z_harm_s signals (r~0.9).
  Config: E2HarmSConfig (standalone dataclass in e2_harm_s.py):
    use_e2_harm_s_forward (bool, default False), z_harm_dim (int, default 32),
    action_dim (int, default 4), hidden_dim (int, default 128),
    action_enc_dim (int, default 16), learning_rate (float, default 5e-4).
  LatentStackConfig.use_e2_harm_s_forward (bool, default False) added to config.py.
  REEConfig.from_dims() param: use_e2_harm_s_forward (default False).
  Data flow: HarmEncoder(harm_obs) -> z_harm_s + action_onehot -> E2HarmSForward -> z_harm_s_pred.
  SD-003 counterfactual pipeline:
    z_harm_s_cf = harm_fwd.counterfactual_forward(z_harm_s_t, a_cf)
    causal_sig  = E3.harm_eval_z_harm_head(z_harm_s_actual) - E3.harm_eval_z_harm_head(z_harm_s_cf)
  Backward compatible: disabled by default; existing experiments unaffected.
  Phased training required: YES (stop-gradient on z_harm_s inputs during P1).
    P0: HarmEncoder warmup (harm proximity supervision).
    P1: E2HarmSForward trains on frozen z_harm_s (z_b.detach(), z1_b.detach()).
    P2: Evaluation (forward_r2, harm_s_cf_gap).
  Biological basis: Keltner et al. (2006, J Neurosci) -- predictability suppresses
    sensory-discriminative (S1/S2) but not affective (ACC) pain responses.
    Forward model cancellation applies to z_harm_s (A-delta analog) not z_harm_a (C-fiber).
  MECH-094: not applicable (waking observation stream, not replay content).
  EXQ-195 evidence: harm_forward_r2=0.914 (forward model component working).
  Validation experiment: V3-EXQ-264 queued.
  Design doc: REE_assembly/docs/architecture/arc_033_e2_harm_s_forward_model.md
  See ARC-033, SD-003, SD-010, SD-011.
