## SD-018: Resource Proximity Supervision (2026-04-07)
- SD-018: encoder.resource_proximity_supervision — IMPLEMENTED 2026-04-07.
  Auxiliary Sigmoid regression head on z_world predicting max(resource_field_view)
  in [0,1]. MSE loss backprops through encoder, forcing z_world to represent resource
  proximity. Without this, benefit_eval_head produces R2=-0.004 (EXQ-085m) and the
  entire benefit/goal pathway (goal_proximity, z_goal seeding, drive modulation,
  dual systems) operates on noise. This is the benefit-side analog of SD-009.
  Config: use_resource_proximity_head (bool, default False),
  resource_proximity_weight (float, default 0.5).
  Agent: compute_resource_proximity_loss(target, latent_state) -> MSE.
  SplitEncoder: resource_proximity_head = Linear(world_dim, 1) + Sigmoid.
  LatentState: resource_prox_pred field. All backward-compatible (disabled default).
  EXQ-257 queued: WITH vs WITHOUT ablation pair, 3 seeds, phased training.
  ALL new benefit/goal experiments MUST set use_resource_proximity_head=True.
