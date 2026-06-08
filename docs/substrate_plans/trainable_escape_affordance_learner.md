# Trainable Escape-Affordance Learner Scaffold

Status: post-603i successor scaffold, not validated substrate.
Registered locally: 2026-06-08.

## Scope

`ree_core/pfc/trainable_escape_affordance_learner.py` adds a feature-flagged
successor option for relief/safety escape affordances. It is inspired by
`REE_assembly/docs/thoughts/2026-06-08_Trainable_Releif_and_Safety.md`.

This scaffold does not replace or alter the current SD-059 / MECH-358 arithmetic
`EscapeAffordanceBridge`, and it does not change the active V3-EXQ-603i validation
path. All new behavior is disabled by default through
`REEConfig.use_trainable_escape_affordance_learner = False`.

## Design Intent

The scaffold separates two trainable-ready components:

- `Q_relief(state, action, threat_context)`: action-contingent expected reduction
  in harm/suffering after directed action under threat.
- `P_safety(state, cue_or_context, action)`: expected threat absence or
  response-produced safety, with extinction when threat recurs.

The first implementation uses compact scalar/prototype tables rather than neural
heads. The public hooks accept compact `z_world`, `z_self`, `z_harm_a`, threat
scale, action class, and optional outcome features so a later experiment can replace
the tables with small PyTorch heads without changing the update contract.

## Guards

- Master flag off by default; disabled learner produces zero bias and no state
  updates.
- Simulation and `hypothesis_tag` updates are no-ops.
- No-op/freeze does not receive credit or approach bonus.
- Bias is bounded by `trainable_escape_bias_scale`.
- Bias is threat-gated and zero when safe.
- Relief requires directed action under threat followed by a z_harm_a drop above
  `trainable_escape_relief_reward_floor`.
- Safety credit requires previous threat and subsequent threat absence, so safety
  is not learned as mere low harm.
- Failed relief and threat recurrence extinguish stale predictions; leak decays
  unused predictions.

## Governance Status

This is not validated substrate and should not be used for governance promotion,
confidence changes, or claim validation. A successor experiment must be explicitly
queued and reviewed before it can affect SD-059, MECH-358, MECH-302, MECH-303, or
MECH-304.
