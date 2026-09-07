## MECH-216: E1 Predictive Wanting / Schema Readout (2026-04-09)
- MECH-216: e1_predictive_wanting -- IMPLEMENTED 2026-04-09.
  E1DeepPredictor.schema_readout_head (Linear(hidden_dim, 1) + Sigmoid) reads LSTM
  top-layer hidden state -> schema_salience [0,1]. Agent caches in _schema_salience
  via _e1_tick(), seeds VALENCE_WANTING when > threshold via update_schema_wanting().
  Zhang/Berridge: W_m = kappa (drive_level) x V_hat (schema_salience).
  Config: E1Config.schema_wanting_enabled (False default), REEConfig.schema_wanting_threshold
  (0.3), schema_wanting_gain (0.5). Training: compute_schema_readout_loss(resource_proximity_target).
  Data flow: E1.predict_long_horizon() -> hidden[0][-1] -> schema_readout_head -> schema_salience
  -> agent._e1_tick() caches -> agent.update_schema_wanting(drive_level) -> ResidueField.update_valence(
  z_world, VALENCE_WANTING, sal * gain * drive).
  Backward compatible: schema_wanting_enabled=False by default; existing experiments unaffected.
  Literature: Berridge 2012 (incentive salience), Zhang et al 2009 (computational model),
  Gershman 2018 (successor representation), Garvert et al 2023 (spatio-predictive maps).
  Validation experiment: EXQ-263 queued (2-condition ablation, 3 seeds, ~100 min).
  See MECH-216, INV-065 (proxy goal necessity), ARC-051 (multi-level wanting hierarchy).
  Signal chain provenance: REE_assembly/docs/architecture/goal_wanting_signal_chain.md
