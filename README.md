# REE V3

REE V3 is the active implementation substrate for the Reflective-Ethical Engine
(REE). It is where architectural claims become experiments. The canonical claim,
evidence, planning, and governance records live in the adjacent
[`REE_assembly`](https://github.com/Latent-Fields/REE_assembly) repository.

REE is exploratory research. Nothing in this repository has been accepted for
peer-reviewed publication. A prototype, an implementation, or a result from a
bounded experiment is not a validated safety system, clinical tool, or evidence
of general scientific truth.

## Start Here

Choose the shortest route that matches your task:

- **Understand the research programme:**
  [How REE develops](https://github.com/Latent-Fields/REE_assembly/blob/master/docs/START_HERE_HOW_REE_DEVELOPS.md).
- **See the current research picture:**
  [Current Front](https://github.com/Latent-Fields/REE_assembly/blob/master/docs/CURRENT_FRONT.md).
- **Understand V3's architecture and substrate status:**
  [V3 specification](docs/ree-v3-spec.md).
- **Find a dated operational snapshot:**
  [V3 status records](docs/status/README.md).
- **Inspect the public research record:**
  [REE Development Map](https://latent-fields.github.io/REE_assembly/development_map.html)
  and the [Lab Window](https://latent-fields.github.io/REE_assembly/public_explorer/).
- **Work on experiments, the runner, or shared state:** read
  [CLAUDE.md](CLAUDE.md) before editing.

## What V3 Is For

V3 does not attempt to demonstrate complete ethical agency. It is a research
substrate for testing the prerequisites that would have to hold before richer
ethical questions could be investigated: self/world separation, causal agency
attribution, harm modelling, motivational persistence, commitment, residual
consequences, and offline integration.

The design commitment is architectural: these concerns should affect the state
from which an agent generates and commits to actions, rather than being added as
a post-hoc score or safety filter. V3 lets those commitments be made explicit,
instrumented, and falsified one at a time.

## Architecture At A Glance

- **E1:** persistent predictive context and temporal coherence.
- **E2:** self-relevant forward modelling, affordances, and action-conditioned
  consequences.
- **E3:** evaluates candidate trajectories, applies commitment constraints, and
  selects actions within those constraints.
- **Hippocampal and residue systems:** retain experience, support trajectory
  proposals, and carry the consequences of earlier actions forward.
- **Control, harm, and goal pathways:** supply the signals that make a candidate
  action relevant to viability, care, and continuing commitments.

The detailed component contracts, flags, implementation state, and evidence
references are maintained in the [V3 specification](docs/ree-v3-spec.md), not
here.

## Scope And Ethics Boundary

REE V3 is not claimed to be sentient, conscious, or a moral patient. It contains
welfare-relevant primitives, but it is pre-ethical substrate work rather than a
candidate mind. This boundary is a maintained governance question, not a
permanent conclusion.

REE V3 is also not a clinical, therapeutic, diagnostic, risk-prediction, or
patient-facing tool. Do not use it or describe it as one. No patient-identifiable
data belongs in public repositories, and no clinical deployment follows from
this work. The author being a clinician does not supply clinical validation.

The governing records are the
[sentience and welfare risk register](https://github.com/Latent-Fields/REE_assembly/blob/master/docs/governance/sentience_welfare_risk_register.md)
and the [ethics perimeter plan](https://github.com/Latent-Fields/REE_assembly/blob/master/evidence/planning/ethics_perimeter_plan.md).

## Repository Map

| Location | Purpose |
| --- | --- |
| `ree_core/` | V3 agent, models, environment interfaces, and control mechanisms. |
| `experiments/` | Pre-registered experiment scripts and shared harnesses. |
| `tests/` | Unit, contract, and flag-inertness coverage. |
| `docs/ree-v3-spec.md` | Current component and substrate reference. |
| `docs/status/` | Dated operational status history. |
| `experiment_queue.json` | Local queue snapshot; the Phase 3 coordinator is authoritative. |
| `../REE_assembly/` | Claims, evidence, planning, governance, and public documentation. |

## Working Safely

The experiment runner and shared evidence data are coordinated across machines.
Read [CLAUDE.md](CLAUDE.md) before changing experiment scripts, queue state,
runner behaviour, shared evidence, or governance records. In particular:

- use the established experiment and review workflows rather than editing the
  queue opportunistically;
- preserve no-op defaults and run the relevant focused tests before landing a
  substrate change;
- treat `REE_assembly` as canonical for claims, reviewed evidence, and planning;
- do not start runners unless the work explicitly calls for it.

## Status Update Boundary

This README is a stable front door. Do not prepend daily or nightly narrative
here. Add dated snapshots to [the status history](docs/status/nightly_history.md)
and update this file only when the stable purpose, boundaries, repository map,
or entry points change.

## License And Citation

Apache License 2.0. See [LICENSE](LICENSE).

For canonical architectural attribution, cite the
[REE specification](https://github.com/Latent-Fields/REE_assembly/) and identify
the exact V3 repository revision used.
