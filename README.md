# ree-v3

Active V3 implementation substrate for REE. Primary development target as of 2026-03-18 following formal V3 transition triggered by V2 hard-stop criteria.

## Why V3

V2 experiments (EXQ-014–028) produced three architectural failures requiring new substrate:

- **EXQ-027 FAIL**: E2 cannot discriminate agent-caused harm in z_gamma → z_self/z_world split required (SD-005)
- **EXQ-023 FAIL**: E1/E2 timescale not separable in shared latent → SD-005 needed
- **EXQ-021 FAIL**: Hippocampal map navigation fails without action-object backbone (SD-004)

## Core Design Decisions

**Founding substrate decisions (co-designed):**
- **SD-004**: Action objects as hippocampal map backbone — E2 produces compressed action-object representations that HippocampalModule maps over, enabling planning horizons far beyond raw state space.
- **SD-005**: Self/World latent split — z_gamma replaced by z_self (E2 domain: motor-sensory, proprioceptive) and z_world (E3/Hippocampus domain: residue, moral attribution, causal footprint). SD-004 and SD-005 must be co-designed.
- **SD-006**: Asynchronous multi-rate loop execution — E1, E2, E3 run at characteristic rates rather than lockstep.

**V3 substrate refinements (all implemented):**
- **SD-007**: Perspective-corrected z_world via ReafferencePredictor (implemented 2026-03-18, validated EXQ-027).
- **SD-008**: alpha_world >= 0.9 in LatentStackConfig — prevents EMA suppression of event responses (validated EXQ-040).
- **SD-009**: Event-contrastive CE auxiliary loss for z_world encoder — forces hazard vs empty discrimination (validated EXQ-020).
- **SD-010**: Harm stream separated as dedicated pathway (z_harm) — prerequisite for SD-003 counterfactual (validated EXQ-056c/058b/059c).
- **SD-011**: Dual nociceptive streams (z_harm_s + z_harm_a) — sensory-discriminative and affective-motivational streams (implemented 2026-03-30, validated EXQ-178b).
- **SD-012**: Homeostatic drive modulation for z_goal seeding — drive_weight scales benefit_exposure by depletion level (implemented 2026-04-02).
- **SD-014**: Hippocampal valence vector node recording — 4-component valence vector [wanting, liking, harm_discriminative, surprise] in RBFLayer and ResidueField (implemented 2026-04-04). Prerequisite for ARC-036 and replay prioritisation.
- **ARC-028 + MECH-105**: HippocampalModule completion signal + BetaGate coupling — implements Lisman & Grace 2005 subiculum->NAc->VP->VTA dopamine loop (implemented 2026-04-04).

SD-004 and SD-005 are interdependent: action objects require z_world to exist; z_world's separation from z_self is what makes action-object attribution meaningful.

## Current Status (2026-04-04)

~292 experiment scripts authored. SD-004 through SD-015 implemented (SD-015 in progress/experimental). EXQ-223 PASS (2026-04-03): minimal vertebrate ablation confirmed E1+E2+hippocampus core loop is sufficient for navigation, harm avoidance, and resource acquisition — named-structure match to zebrafish larva (5-7 dpf). Current focus: first-paper gate experiments — EXQ-074e (wanting/liking dissociation), EXQ-076e (E1 goal conditioning), EXQ-195 (SD-003 z_harm_s counterfactual post-SD-011). SD-014 valence vector and ARC-028/MECH-105 hippocampal-BetaGate coupling now implemented.

Full specification with substrate status table: [`docs/ree-v3-spec.md`](docs/ree-v3-spec.md)

## Experiment Tagging

- `run_id` must end in `_v3`
- `architecture_epoch` must be `"ree_hybrid_guardrails_v1"`
- Results go to `REE_assembly/evidence/experiments/`

## Governance

From `REE_assembly/` root:

```bash
bash scripts/governance.sh
```

Or manually:
```bash
python evidence/experiments/scripts/build_experiment_indexes.py
python scripts/generate_pending_review.py
```

## License

Apache License 2.0 (see `LICENSE`).

## Citation

- Cite this repository using `CITATION.cff`.
- For canonical architectural attribution, cite Daniel Golden's REE specification in `https://github.com/Latent-Fields/REE_assembly/`.
