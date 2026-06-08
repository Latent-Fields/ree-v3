# Post-603i E2 Escape-Affordance Linkage

Status: post-603i prerequisite substrate scaffold, **not** a validated substrate.
Registered locally: 2026-06-08.

## What 603i showed (and did not show)

V3-EXQ-603i landed **FAIL**, route `substrate_not_ready_requeue`,
`evidence_direction = non_contributory`, with **no claim weakening**.

- It did **not** falsify SD-059 / MECH-358. The relief/safety escape-affordance
  bridge could not be meaningfully adjudicated.
- The base defensive chain engaged correctly: Pavlovian freezing / PAG analogue
  present; the ilPFC-analogue instrumental-avoidance gate (SD-058 / MECH-357)
  engaged and suppressed freezing; the Stage-0 goal positive control passed.
- The failure was **downstream**: the fixed arithmetic relief/safety bridge did
  not lift hazard survival; safety credit did not fire; relief credit fired in
  some seeds but did not translate to survival; navigation-control did not clear.
  The manifest states the relief detector needs a **trained encoder / world-
  forward**.

603i therefore discovered a **prerequisite representation / linkage gap**, not a
falsification: the representation of "where out is" under threat — a directed
escape affordance — was not ready for the affect heads to label.

## The correction: reuse, do not duplicate

The missing piece must **reuse the existing E2 (cerebellar-analogue) action-
consequence forward model**, not duplicate a standalone fast predictor.
`E2FastPredictor.world_forward` / `E2WorldForward` already predict action
consequences; building a second "fast forward predictor" would duplicate Engine 2.

The decomposition this scaffold commits to:

| Region (biological)        | Role                                                              | Owner |
|----------------------------|------------------------------------------------------------------|-------|
| E2 / cerebellum            | fast forward prediction of action consequences                   | **E2 (reused)** |
| Hippocampus                | relational viability map over action-consequence coordinates     | viability-readout scaffold (index only; no trajectory generation, no reward) |
| Amygdala / affect streams  | relief and safety learning, kept **distinct**                    | TrainableEscapeAffordanceLearner heads (+ linker readouts) |
| Prefrontal gate / PAG      | freeze suppression + defensive execution                          | SD-058 / MECH-357 / MECH-279 (**not rebuilt**) |
| Basal ganglia / E3         | selection bias + commitment, not representation                  | E3 (receives bounded threat-gated bias only) |

The correct shape is: **E2 predicts. Hippocampus indexes viability. Relief/safety
label the consequence. E3 selects under bounded, threat-gated bias.** No
monolithic "escape brain."

## What this scaffold adds

Module: `ree_core/pfc/e2_escape_affordance_linker.py`
(`E2EscapeAffordanceLinker` / `E2EscapeAffordanceLinkerConfig` /
`E2EscapeAffordanceLinkerOutput`).

It is a **readout / linkage layer over detached E2 action-consequence features**,
not a forward predictor:

- **Inputs (detached):** the E2 forward feature for the executed `(prev_z_world,
  action)` pair — `[predicted_next_z_world, delta]` from `E2.world_forward`
  (the agent computes this under `torch.no_grad()` and passes it in; the linker
  never calls a predictor) — with compact `z_world` / `z_self` / `z_harm_a` as a
  fallback when no E2 forward model is exposed, plus threat scale, the
  executed/candidate action class, and optional refuge / hazard / survival
  signals when available.
- **Readouts (small trainable heads on top of the detached E2 geometry):**
  `predicted_harm_delta`, `predicted_threat_termination`,
  `predicted_safety_transition`, `predicted_refuge_reachability`, and an optional
  `predicted_survival_step`. Relief side = harm-delta + threat-termination;
  safety side = safety-transition + refuge-reachability. Kept distinct.
- **`escape_affordance_features`:** the trunk activation — the representational
  substrate the relief/safety affect heads can optionally consume.
- **Hippocampal-style viability index:** a per-action-class EMA of escape success
  ("where can this action lead?"). A readout / index only — it does **not**
  generate trajectories and does **not** compute reward or select actions.
- **Bounded threat-gated E3 bias:** behind its own flag, a negative (favoured)
  per-candidate score-bias toward the predicted escape action — clamped to
  `bias_scale`, exactly zero when safe, never applied to the no-op class.

### Training target

The linker learns **only** action-contingent viability labels on top of the E2
geometry. A *positive escape* target fires when the previous state was under
threat, the action was directed and non-noop, **and** harm decreased / threat
terminated / refuge became closer-or-reached / survival improved. A *negative
(extinction)* target fires when a directed escape attempt under prior threat did
not reduce harm/threat, or when predicted safety was followed by threat
recurrence. No learning occurs for the no-op/freeze class, under
simulation/hypothesis mode, when no threat transition occurred, or when inputs
are insufficient for a valid target.

### Integration with the relief/safety learner

`TrainableEscapeAffordanceLearner` is **not** replaced. It gains an optional
`extra_features` input (default `None` → bit-identical) so it can consume the
linker's `escape_affordance_features`. The relief head stays relief-specific and
the safety head stays safety-specific; the linker provides representational
substrate, the heads provide the affective labels.

## Guarantees (all enforced + contract-tested)

- OFF by default at the agent/config layer
  (`use_e2_escape_affordance_linker = False`); disabled construction instantiates
  no neural state and emits zero bias; the agent path is bit-identical OFF
  (949 contracts pass).
- E2 / latent inputs are **detached** — no backprop into E1/E2/E3 encoders by
  default.
- No learning under simulation / `hypothesis_tag` mode (MECH-094).
- No credit to the no-op/freeze class by default.
- Relief and safety remain distinct readouts.
- Bias is threat-gated (exactly zero when safe) and clamped to `bias_scale`.
- Learned head weights and the viability index persist across episode reset;
  reset clears only the one-tick action-contingent traces.

## Config flags (all no-op defaults)

`use_e2_escape_affordance_linker` (master), `use_e2_escape_linker_for_relief_safety`
(feed features to the learner), `use_e2_escape_linker_e3_bias` (emit the bounded
E3 bias), plus `escape_linker_learn_rate`, `escape_linker_optimizer_lr`,
`escape_linker_leak_rate`, `escape_linker_hidden_dim`,
`escape_linker_action_embedding_dim`, `escape_linker_bias_scale`,
`escape_linker_threat_floor`, `escape_linker_threat_ref`,
`escape_linker_noop_class`, `escape_linker_relief_reward_floor`,
`escape_linker_harm_delta_scale`, `escape_linker_prediction_floor`,
`escape_linker_block_hypothesis_learning`.

## Microdiagnostic (forced-choice readiness — not an ecological survival claim)

`tests/contracts/test_e2_escape_affordance_linker.py::test_forced_choice_readiness_microdiagnostic`
is a tiny forced-choice probe: four action classes (no-op / harm-worsening /
escape-producing / neutral), each non-noop action carrying a distinct E2
consequence feature and its true outcome. Readiness gates (2/3 seeds unless
noted): the readout predicts the escape action better after training; the
no-op/freeze class is uncredited; the threat-gated bias points toward the learned
escape action; the bias is exactly zero when safe (all seeds); learning is
blocked under hypothesis/simulation mode; learned weights persist across reset.
This is **substrate readiness only** — no ecological survival claim is made.

## Successor sequence (deliberately not done here)

A successor experiment should validate **encoder/linker readiness before** any
bridge re-validation. Specifically:

- A formal **failure autopsy** of V3-EXQ-603i and a **biological-fidelity
  review** of this linkage decomposition still need to be done.
- The hippocampal viability-mapping scaffold should be matured to genuinely
  index E2 action-object coordinates.
- Only after a readiness diagnostic passes should a bridge-validation EXQ be
  queued. **Do not queue a full 603j bridge re-run from this scaffold.**

## Governance status

This is not validated substrate. It must **not** be used for governance
promotion, confidence changes, or claim validation. It does **not** mark SD-059 /
MECH-358 (or MECH-302 / MECH-303 / MECH-304) as validated or weakened. This note
has **no queue effect, no governance effect**, and does **not** alter the active
V3-EXQ-603i path. A successor experiment must be explicitly queued and reviewed
before this layer can affect any claim.
