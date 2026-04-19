# V3 Regression Suite Proposal

**Created:** 2026-04-19  
**Status:** Proposal  
**Scope:** REE V3 engineering QA layer for compute-saving prechecks and regression detection  
**Audience:** solo REE development and experiment operations

---

## 1. Purpose

This document proposes a **small, fast regression suite** for `ree-v3` that sits
between:

- architectural/governance review
- full EXQ experiment runs

The suite is not intended to prove claims. Its job is to answer a narrower
question:

**Is the current substrate fit to generate evidence without wasting queue time on
setup failures, broken wiring, or obvious behavioral regressions?**

This is a compute-saving measure, not a scientific shortcut.

---

## 2. What The Suite Must And Must Not Do

### Must do

- catch setup and host-specific failures before long runs
- catch wiring regressions in goal, harm, gating, imagined-vs-acted, and sleep
- use tiny deterministic toy settings
- run cheaply on CPU
- stay stable across machines

### Must not do

- encode paper conclusions as tests
- replace EXQ experiments
- depend on long training
- assert narrow scientific thresholds copied from experiments
- lock REE into a temporary implementation detail

The suite should test **contracts**, not conclusions.

---

## 3. Design Principles

1. **Behavioral contract over implementation detail**
   The test should encode what must be true at the interface/behavior level.
   Example: imagined trajectories must not produce action-side effects or ethical
   residue writes. The current `hypothesis_tag` mechanism may be one way to do
   this, but the regression suite should not require that exact mechanism.

2. **Deterministic where possible**
   Use fixed seeds, tiny worlds, small horizons, and coarse directional checks.

3. **Three-layer runtime budget**
   - `preflight`: 10-30 seconds
   - `contracts`: 30-90 seconds total
   - `microprobes`: 2-10 minutes only when touched subsystems require them

4. **Fail cheap, fail early**
   Any issue that can be discovered in 30 seconds should never consume a
   2-4 hour queued run.

5. **Subsystem-local activation**
   If a change only touches sleep or gating, only the relevant microprobes need
   to run before queueing.

6. **No drift pressure**
   The suite should not force architecture decisions by accident. It should
   protect invariants already believed to be load-bearing.

---

## 4. Proposed Directory Layout

```text
ree-v3/
  tests/
    conftest.py
    fixtures/
      tiny_env.py
      tiny_configs.py
      seed_utils.py
    preflight/
      test_runner_preflight.py
      test_queue_integrity.py
      test_machine_boot.py
    contracts/
      test_agent_boot.py
      test_feature_flag_boot_matrix.py
      test_seed_determinism.py
      test_bg_gating_contracts.py
      test_imagined_acted_isolation.py
      test_goal_contracts.py
      test_harm_contracts.py
      test_sleep_contracts.py
      test_hippocampal_contracts.py
    microprobes/
      test_goal_lift_micro.py
      test_harm_avoidance_micro.py
      test_gating_micro.py
      test_sleep_ordering_micro.py
      test_hippocampal_planning_micro.py
  scripts/
    run_regression_suite.py
```

### Notes

- `fixtures/` contains tiny deterministic helpers shared across tests.
- `preflight/` is host- and runner-oriented.
- `contracts/` protects low-cost architectural invariants.
- `microprobes/` are tiny behavioral checks, still much cheaper than EXQ runs.
- `scripts/run_regression_suite.py` is a convenience wrapper for staged execution:
  `preflight`, `contracts`, or `changed-subsystems`.

---

## 5. Proposed Test Inventory

## 5.1 Preflight Layer

### P1. `test_runner_preflight.py`

Checks:
- runner imports succeed
- queue file parses
- runner status path resolves
- every queued script path exists
- dry startup path succeeds

Why:
- catches setup/path failures before runner queue time is burned

Expected runtime:
- under 10 seconds

### P2. `test_queue_integrity.py`

Checks:
- `experiment_queue.json` schema valid
- duplicate queue IDs absent
- supersedes references are syntactically sane
- claimed items still point at existing scripts

Why:
- cheap protection against manual queue breakage

Expected runtime:
- under 5 seconds

### P3. `test_machine_boot.py`

Checks:
- `REEAgent` instantiates on current host
- minimal environment reset/step works
- device selection and output directories work

Why:
- catches the class of "setup crash on machine X, immediate requeue"

Expected runtime:
- under 15 seconds

---

## 5.2 Contract Layer

### C1. `test_agent_boot.py`

Checks:
- default `REEAgent` boots, resets, and survives a tiny fixed episode

Protects:
- gross integration regressions in core wiring

### C2. `test_feature_flag_boot_matrix.py`

Checks a small matrix of important configurations:
- default baseline
- goal on
- harm stream / harm modulation on
- gating on
- sleep on
- combined "current V3" style config

Protects:
- flag interaction regressions

Comment:
- especially important because many V3 features are default-off for backward
  compatibility

### C3. `test_seed_determinism.py`

Checks:
- same seed -> same first N actions and same cheap summary metrics in tiny env

Protects:
- reproducibility and debuggability

### C4. `test_bg_gating_contracts.py`

Checks:
- commit elevates beta gate
- completion releases it
- committed trajectory steps through `a0 -> a1 -> a2` rather than repeating `a0`
- urgency interrupt breaks a held commitment and resets stepping state

Protects:
- basal ganglia gating / committed-action path

### C5. `test_imagined_acted_isolation.py`

Checks:
- imagined/replay paths do not emit external action-side effects
- imagined/replay paths do not write ethical residue
- imagined mode can update allowed internal replay bookkeeping only

Protects:
- imagined-vs-acted distinction

Important framing:
- this file should **not** require `hypothesis_tag` specifically
- the contract is behavioral isolation, not one chosen mechanism

### C6. `test_goal_contracts.py`

Checks:
- `z_goal` updates on benefit
- `z_goal` decays without benefit
- disabled goal mode stays inert
- enabling goal changes trajectory scoring/ranking in intended direction on a toy case

Protects:
- goal representation existence separate from full lift experiments

### C7. `test_harm_contracts.py`

Checks:
- obvious hazard ranks worse than obvious safe in a tiny case
- descending modulation only attenuates under intended gating conditions
- attenuation is not accidentally global

Protects:
- harm avoidance substrate

### C8. `test_sleep_contracts.py`

Checks:
- entering SWS/REM changes the right mode/state flags
- sleep-only write paths are gated to sleep
- waking path does not accidentally trigger sleep updates

Protects:
- sleep scaffolding as an engineering substrate

### C9. `test_hippocampal_contracts.py`

Checks:
- terrain/residue affects proposal generation
- ablated terrain weakens or flattens the proposal difference
- hippocampal proposal path remains distinct from direct value head behavior

Protects:
- hippocampal planner identity

---

## 5.3 Microprobe Layer

These are still cheap, but they are behavioral mini-assays rather than pure
contracts.

### M1. `test_goal_lift_micro.py`

Tiny assay:
- oracle or handcrafted goal cue vs null
- very small world, few seeds, very short training

Question:
- does the goal path produce any positive lift at all in a toy regime?

Purpose:
- pre-EXQ sanity check for goal-directed behavior

### M2. `test_harm_avoidance_micro.py`

Tiny assay:
- hazard-present vs safe route
- measure immediate harmful contact difference

Question:
- does the agent avoid obvious harm under current scoring/gating?

### M3. `test_gating_micro.py`

Tiny assay:
- compare committed vs non-committed path following under an interrupt condition

Question:
- does the gating system produce distinct behavior, not just internal state changes?

### M4. `test_sleep_ordering_micro.py`

Tiny assay:
- toy context/memory setup
- SWS-before-REM vs reversed or missing order

Question:
- is the sleep machinery structurally ordered in a way that produces different outcomes?

Purpose:
- protects the engineering substrate behind the Bayesian-prior-before-posterior
  paper claim without prematurely encoding the paper result as a unit test

### M5. `test_hippocampal_planning_micro.py`

Tiny assay:
- one-step greedy policy vs short multi-step proposal task

Question:
- can the hippocampal planning path solve a toy case that greedy one-step cannot?

Purpose:
- preflight for the V3 full completion gate around multi-step planning

---

## 6. First Implementation Tranche

The first tranche should be chosen by **compute saved per hour of engineering**,
not by architectural elegance.

### Phase A: highest immediate payoff

1. `test_runner_preflight.py`
2. `test_machine_boot.py`
3. `test_feature_flag_boot_matrix.py`
4. `test_bg_gating_contracts.py`
5. `test_goal_contracts.py`
6. `test_imagined_acted_isolation.py`

Reason:
- these target the current failure modes most likely to waste queue time:
  setup failure, feature-flag interaction, gating regressions, goal path breakage,
  and imagined/acted contamination

### Phase B: next priority

7. `test_harm_contracts.py`
8. `test_sleep_contracts.py`
9. `test_sleep_ordering_micro.py`
10. `test_hippocampal_contracts.py`

### Phase C: targeted research microprobes

11. `test_goal_lift_micro.py`
12. `test_harm_avoidance_micro.py`
13. `test_gating_micro.py`
14. `test_hippocampal_planning_micro.py`

---

## 7. Suggested Run Policy

### Before starting a runner on any machine

Run:
- preflight layer

### Before queueing after shared substrate edits

Run:
- preflight layer
- contract layer for touched subsystems

### Before queueing high-cost paper-gate experiments

Run:
- preflight layer
- full contract layer
- relevant microprobe(s)

### Before governance

Do not require the regression suite.

Reason:
- governance interprets evidence already generated
- the suite is for substrate fitness before evidence generation

---

## 8. Expected Compute Savings

This suite does not make experiments shorter. It saves compute by converting:

- machine-specific startup crashes -> 10-second preflight failures
- broken substrate wiring -> 30-second contract failures
- obviously dead paper-gate runs -> small microprobe failures

Even a modest reduction in failed or requeued runs would repay the engineering time.
For a solo workflow with limited hardware windows, the main gain is not raw FLOPs but
**better use of scarce runner time**.

---

## 9. Out Of Scope

The regression suite should not attempt to settle:

- whether the present `hypothesis_tag` mechanism is the final imagined/acted solution
- final scientific thresholds for the three candidate papers
- long-horizon training behavior
- governance promotion/demotion decisions
- V4-only mechanisms not yet admitted into V3 scope

Those belong to experiments, architecture decisions, and governance.

---

## 10. Summary

The regression suite should be treated as:

- **engineering QA for research substrate**
- **compute protection for scarce runner time**
- **implementation-agnostic where architecture is still unsettled**

The most important constraint is discipline:

**test contracts, not conclusions**

If that line holds, the suite should reduce compute waste without introducing
scientific drift or locking REE into temporary mechanisms.
