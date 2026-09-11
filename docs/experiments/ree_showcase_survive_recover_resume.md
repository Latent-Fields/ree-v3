# REE in a changing reef: survive, recover, resume

Date: 2026-09-11. Status: **design complete; implementation and queueing held at substrate preflight**.
Owner of this design: `codex-showcase-20260911`.
No EXQ number has been reserved, no experiment script has been written, and no runner has been started.

## The demonstration

A small agent learns a reef, encounters a period of increased danger, and gets an
opportunity to return to feeding after that danger recedes. Show the same sequence
beside simpler policies and a matched REE ablation. Viewers should be able to see
what the agent does, why its controller changed course, and whether that change
helped it stay alive and obtain resources.

**Question:** can a trained, explicitly specified V3 configuration combine its
demonstrated control mechanisms into useful behavior during a change in threat?
The missing fact is integrated behavioral performance under a new, paired
challenge: `complex (probe-gated)`, `puzzle (known rules)`.

"Best thus far" means a configuration selected on separate development seeds,
with every enabled mechanism and training budget disclosed. It does not mean
every optional flag enabled, or the most attractive surviving trajectory.

This combines a visual demonstration with a small benchmark. It is a
**baseline-purpose integration benchmark**, with `claim_ids: []` and
`evidence_direction: non_contributory`. Results cannot promote component claims
or establish that REE outperforms modern reinforcement learning generally.

## What existing results justify

These four run IDs are explicitly recorded as reviewed in
`REE_assembly/evidence/experiments/review_tracker.json`. The figures below were
read from the individual manifests, not inferred from the word PASS.

| Foundation | Recorded observation | Limit on the inference |
| --- | --- | --- |
| EXQ-944b, salient-event cycle boundaries | Mean straddling fraction 0.000 for aligned resets versus 0.9211 for rate-matched resets, across five scored seeds of six attempted | Strong evidence for timely controller boundaries; not evidence of better survival or feeding |
| EXQ-603v, avoidance eligibility repair | PASS for persistence of the learned avoidance trace through its scoring window | Instrument repair; its driver explicitly does not establish an intact-versus-lesioned survival advantage |
| EXQ-916a, relief/safety fishtank | Relief events 58, 0, 2 across three seeds; wanting and cue-safety signals vary | Contextual safety terrain and vigor remain flat in this run; variable channels do not establish competent behavior |
| EXQ-910b, orienting readout repair | 21 actual overrides: 19 approach, 2 withdraw, 0 resume; 11,025 fresh and 56,173 latched ticks | Establishes a correctly counted, non-degenerate two-class readout; does not demonstrate recovery/resume |

Source manifests, relative to the umbrella root:

- `REE_assembly/evidence/experiments/v3_exq_944b_mech091_salient_event_cycle_boundary_20260825T205738Z_v3.json`
- `REE_assembly/evidence/experiments/v3_exq_603v_mech357_eligibility_trace_repair_validation_20260827T184708Z_v3.json`
- `REE_assembly/evidence/experiments/v3_exq_916a_relief_safety_fishtank_showcase_20260811T194142Z_v3.json`
- `REE_assembly/evidence/experiments/v3_exq_910b_mech489_orienting_decision_at_override_tick_retest_20260822T235826Z_v3.json`

The prior fishtank PASS gates primarily concern channel liveness. EXQ-665, for
example, passed while recording only 50 evaluation steps and a zero hazard
survival gate. Reusing those gates would not answer the new question.

Existing-evidence check: none of the four inspected manifests contains the
proposed paired **safe recovery success** endpoint under this three-phase
schedule and these comparator arms. Their substrate hashes also differ. They
provide design precedents, not a pooled best-agent benchmark. Before reserving an
EXQ, run the repository-wide `reanalysis_query.py` check for this endpoint and
its raw inputs; the targeted check above is not a claim of exhaustive search.

## Proposed experiment

Use `CausalGridWorldV2` with a visible, bounded reef and renewable resources.
Keep the map and the organism continuous within each episode. No resurrection,
forced feeding, action advice, or state reset occurs during evaluation.

| Phase | Intended steps | What changes | Observable question |
| --- | --- | --- | --- |
| Settle and forage | 0-99 | Mild background danger | Does the agent obtain resources without needless harm? |
| Threat | 100-199 | Pre-generated increase in hazard activity | Does it interrupt its current behavior and reduce exposure? |
| Recovery | 200-299 | Hazard activity returns to its initial regime | Does it resume useful resource seeking and survive? |

The phase schedule is supplied to the environment and the display, never to the
policy. Exact hazard parameters must be selected on calibration seeds and frozen
before held-out evaluation. Preserve a physically attainable escape route.
Avoid making the test solvable only by a hidden global map or a phase label.

Use five held-out training seeds `[107, 211, 307, 401, 503]`, each with 12 new
evaluation maps. Development seeds are `[42, 43, 44]`. Reserve explicit,
non-overlapping environment, policy, training and analysis RNG streams. Treat
the five trained agents as the independent units for uncertainty estimates;
60 episodes and thousands of frames are not 60 or thousands of training seeds.

### Comparators

1. **REE candidate:** the best eligible configuration on development seeds,
   using staged encoder/head training, verified live goals, and calibrated harm
   signals. Evaluate the actual chosen configuration, recorded in full.
2. **Matched REE with salient-event cycle resets disabled:** start from the same
   trained state, change only the reset intervention, and preserve all other
   parameters, learned traces, goals and controller state. This tests a narrow
   integration contribution supported by EXQ-944b. It is not "REE without memory".
3. **Reactive local policy:** a transparent, memoryless controller using the
   same current sensory fields and legal actions. Choose its harm/resource
   tradeoff on the development split. Log its policy and information access.
4. **Uniform random legal actions:** an interpretable lower reference.

A separate full-state planner is an **environment attainability control** only.
Its privileged access must be explicit. It does not appear as an equal-information
competitor or establish the learner's observable ceiling.

For paired arms, generate environmental randomness independently of action
selection. Equal integer seeds alone do not produce matched hazard realizations
when different policies consume RNG differently. Archive the exogenous tapes;
action-dependent outcomes should still differ legitimately.

Do not clone solely with `REEAgent.state_dict()`: GoalState and other Python-side
state can be omitted. Verify the full clone and a no-intervention action replay
before changing the reset flag. Start each evaluation episode from an identical
post-training snapshot, so evaluation order cannot become additional training.
Keep network weights frozen; within-episode online state updates are permitted
and must be identical across the paired REE arms apart from the intervention.

## Success and interpretation

Primary endpoint: **safe recovery success**, one binary outcome for every
intended episode. It is 1 only when the agent:

- remains alive through step 299;
- obtains at least one resource during recovery (steps 200-299); and
- incurs recovery damage no greater than 10% of its initial health budget.

Otherwise it is 0. Death before recovery is a failure, not an excluded episode.
The implementation must define damage from the actual environment damage
accounting; cumulative negative reward is not an interchangeable proxy.

Proposed pre-registration, to freeze before held-out execution:

- **C1, absolute competence:** mean seed-level safe recovery success >= 0.60.
- **C2, useful lower-reference improvement:** paired mean advantage over random
  >= 0.20, with a seed-block bootstrap 95% interval lower bound above 0.
- **C3, controller contribution:** paired mean advantage over the reset-disabled
  REE arm >= 0.10, with its seed-block bootstrap 95% interval lower bound above 0.

Report C1-C3 separately. An overall showcase PASS requires all three. Comparison
with the reactive policy is fully reported but does not silently become a new
PASS criterion after seeing the data. With only five training seeds, intervals
are exploratory and may be wide; a null is not proof of equivalence.

Secondary measures: survival fraction, total damage, resource intake per
intended step, threat-phase exposure, and restricted time to first recovery
contact. Assign the full 100-step recovery latency to deaths and non-recoverers.
Show all denominators. Never turn missing samples into zero-harm successes.

Instrument readiness is separate from scientific success. On development seeds,
check the same primary endpoint under the attainability control and random
policy, plus successful phase transitions and actual reset opportunities in
both paired arms. Record measured values, numeric thresholds, and the exact
observation access of each control. A zero event count or unattainable task routes
to `instrument_not_ready`; it cannot falsify REE or earn a PASS. A full-state
control passing alone does not establish partial-observation learnability.

Do not tune on held-out results or replace failed seeds. If a changed environment,
threshold or configuration is needed after evaluation, register a new iteration.

## Visual and recording deliverables

- A synchronized four-panel replay with phase labels, health, resource contacts,
  damage and elapsed steps. Select the **median REE episode by the primary
  endpoint and then total resource intake**, using seed/map order to break ties;
  state this selection rule. Provide every other episode in the selector too.
- A behavior timeline linking actual fresh controller events to actions, with
  internal harm/drive/relief signals labeled as model variables. No claim that
  those traces show subjective feelings.
- A summary panel with all seed-level scores, paired differences, intervals,
  failures, and unresolved readiness conditions. A poor result remains visible.
- Standard `_v3` manifest, `ree_hybrid_guardrails_v1` architecture epoch, flat
  numeric readouts, explicit intended/realized denominators, per-arm fingerprints,
  full configuration, substrate hash, training budgets, and companion trajectory
  logs. Match the existing fishtank log schema where practical.

Use fresh-event instrumentation rather than repeatedly counting latched values.
Capture consume-once relief events before action selection clears them. Record
z_goal liveness and training representation quality. No internal-signal variance
criterion substitutes for the external behavioral endpoint.

## Why implementation is held

Checked against ree-v3 `1bbcf19ec9a9fd7a6fe4a8e39a33dd2c8f8b14f2` and fetched
REE_assembly origin/master `b5c1fe043db15cdd6ee085e9b46600b4de4c3fc0` on 2026-09-11.
These open `substrate_queue.json` records overlap the integrated driver:

| Record | Recorded state | Overlap |
| --- | --- | --- |
| `mode-governance-engagement` | `implemented_pending_validation`, `corrupting` | `ree_core/agent.py`, `ree_core/utils/config.py`, salience coordinator |
| `SD-e1-rollout-consistency-training` | validation owed, `corrupting` | `ree_core/predictors/e1_deep.py` |
| `contextmemory-write-path-addressing-degeneracy` | `implemented_pending_validation`, `corrupting` | `ree_core/predictors/e1_deep.py::ContextMemory.write` |

Queue-experiment Step 2.5c matches at module granularity and explicitly says:
**"Do NOT write the script. Do NOT add a queue entry."** The integrated agent
imports E1, so merely avoiding a direct driver import does not remove the overlap.
Disabling an optional feature may narrow scientific exposure, but the skill does
not automatically grant a module-gate exception for it.

The mode-governance record may partly lag a recent repair (WORKSPACE_STATE cites
ree-v3 `606aea2` and GFLAG-0262). It nevertheless remains open on fetched origin.
The other two records also remain open. This session does not close or downgrade
governance records to enable a showcase. EXQ-1010 is undergoing review in another
session and may further inform the configuration choice.

### Resumption checklist

1. Resolve or explicitly authorize a scoped exception to the three recorded
   module holds. Document what remains unvalidated and keep this benchmark
   non-contributory. This is a governance/authorization decision, not a request
   for the user to debug the code.
2. Recheck current source, live task claims, the completed EXQ-1010 review, and
   the full recorded-evidence reuse surface. Reserve the next EXQ only then.
3. Prove configuration reachability with a minimal probe; calibrate the task on
   development seeds, lock the specification, and implement the actual driver.
4. Smoke-test state cloning, matched randomness, death accounting, events,
   manifest/trajectory writing, and a nonzero primary endpoint on its control.
5. Complete the queue skill's code review, independent design review, strict
   script/recording/queue validation, commit and push, then confirm coordinator
   ingestion. No smoke test or independent review has yet been performed.
6. The user starts the runner through Explorer unless explicitly authorizing a
   direct start. Estimate full runtime from measured pilot throughput before
   scheduling the training grid; no credible runtime estimate exists yet.
