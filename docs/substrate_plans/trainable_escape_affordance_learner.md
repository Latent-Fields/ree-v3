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

The scaffold separates two trainable components:

- `Q_relief(state, action, threat_context)`: action-contingent expected reduction
  in harm/suffering after directed action under threat.
- `P_safety(state, cue_or_context, action)`: expected threat absence or
  response-produced safety, with extinction when threat recurs.

The enabled implementation now uses actual PyTorch heads rather than
scalar/prototype credit tables. A small shared trunk consumes detached compact
state/context features (`z_world`, `z_self`, `z_harm_a`, `z_harm_a_norm`, threat
scale, and optional outcome features) together with a learned action embedding.
The relief and safety heads are sigmoid bounded to `[0, 1]` and train through a
local AdamW optimizer. The detach boundary is deliberate: by default, relief and
safety updates do not backpropagate into E1/E2/E3 encoders.

## Guards

- Master flag off by default; disabled learner produces zero bias and no state
  updates.
- Simulation and `hypothesis_tag` updates are no-ops.
- No-op/freeze does not receive credit or approach bonus.
- Bias is bounded by `trainable_escape_bias_scale`.
- Bias is threat-gated and zero when safe.
- Relief requires directed action under threat followed by a z_harm_a drop above
  `trainable_escape_relief_reward_floor`; the target is continuous via
  `trainable_escape_relief_target_scale`.
- Safety credit requires previous threat and subsequent threat absence, so safety
  is not learned as mere low harm.
- Failed relief and threat recurrence train extinction targets toward zero.
- Learned head weights persist across episode reset; reset clears only one-tick
  traces and cached previous state/action.
- Diagnostics include relief/safety loss, optimizer steps, positive/negative
  target counts, max relief/safety predictions, and bias fire count.

## Governance Status

This is not validated substrate and should not be used for governance promotion,
confidence changes, or claim validation. A successor experiment must be explicitly
queued and reviewed before it can affect SD-059, MECH-358, MECH-302, MECH-303, or
MECH-304. This note has no queue effect, no governance effect, and does not alter
the active V3-EXQ-603i path.
