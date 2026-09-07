## Post-603i successor scaffold: trainable relief/safety escape-affordance learner (2026-06-08)
- trainable_escape_affordance_learner -- SCAFFOLDED 2026-06-08, not validated
  substrate. Module: ree_core/pfc/trainable_escape_affordance_learner.py
  (TrainableEscapeAffordanceLearner + Config + Output). This is a successor option
  inspired by REE_assembly/docs/thoughts/2026-06-08_Trainable_Releif_and_Safety.md,
  not a replacement for the active SD-059 / MECH-358 arithmetic bridge and not a
  change to V3-EXQ-603i. Config (REEConfig + from_dims; master default OFF):
  use_trainable_escape_affordance_learner (False),
  use_trainable_relief_critic (True), use_trainable_safety_predictor (True),
  trainable_escape_bias_scale (0.1), trainable_escape_relief_learn_rate (0.1),
  trainable_escape_safety_learn_rate (0.1), trainable_escape_leak_rate (0.01),
  trainable_escape_relief_reward_floor (1e-4),
  trainable_escape_relief_target_scale (0.3), trainable_escape_threat_floor
  (0.1), trainable_escape_noop_class (0), trainable_escape_hidden_dim (32),
  trainable_escape_action_embedding_dim (8), trainable_escape_optimizer_lr
  (0.03), trainable_escape_prediction_floor (0.02). Data flow: sense()
  detached z_world/z_self/z_harm_a + last action class -> local PyTorch
  relief/safety heads update through AdamW (one-tick lag, MECH-094 no-op under
  hypothesis_tag) -> select_action() per-candidate first action classes +
  current z_harm_a -> bounded negative E3 score-bias from model predictions.
  If the arithmetic bridge and trainable learner are both enabled, they compose
  additively, each under its own clamp. Guards: zero when disabled, zero when
  safe, simulation/hypothesis no-op, no-op/freeze receives no credit or approach
  bonus, failed relief and threat recurrence train extinction targets. Episode
  reset clears only one-tick traces and preserves learned head weights. Local note:
  docs/substrate_plans/trainable_escape_affordance_learner.md. No queue entry,
  no claims/governance effect; do not use for promotion until an explicit
  successor experiment is queued and reviewed.
