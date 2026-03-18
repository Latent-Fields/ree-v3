# ree-v3

## Python
Use /opt/local/bin/python3 for all execution (has torch 2.10.0).
Use sys.executable for subprocesses within experiment runners.

## Branch Policy
No feature branches. All work to `main` directly.
Push: `git push origin HEAD:main`

## Governance
Run packs go to REE_assembly/evidence/experiments/.
run_id must end _v3. architecture_epoch must be "ree_hybrid_guardrails_v1".
After experiments complete: run sync_v3_results.py then build_experiment_indexes.py.

## Key Architecture Constraints
- E2 trains on motor-sensory error (z_self). NOT harm/goal error.
- E3 is the harm evaluator. harm_eval() belongs on E3Selector.
- ResidueField accumulates world_delta (z_world). NOT z_gamma.
- HippocampalModule navigates action-object space O. NOT raw z_world.
- All replay/simulation content must carry hypothesis_tag=True (MECH-094).
- Precision is E3-derived (E3 prediction error variance). NOT hardcoded.

## Q-020 Decision (2026-03-16)
ARC-007 STRICT: HippocampalModule generates value-flat proposals.
Terrain sensitivity = consequence of navigating residue-shaped z_world, not a separate hippocampal value computation.
MECH-073 reframed as consequence of ARC-013 applied to z_world.
MECH-074 (amygdala write interface) is valid but not a HippocampalModule prerequisite.

## SD Design Decisions Implemented
- SD-004: E2 action objects; HippocampalModule navigates action-object space O
- SD-005: z_gamma split into z_self (E2 domain) + z_world (E3/Hippocampal/ResidueField domain)
- SD-006: Asynchronous multi-rate loop execution (phase 1: time-multiplexed)

## SD Design Decisions Implemented (V3) — continued
- SD-007: encoder.perspective_corrected_world_latent — IMPLEMENTED 2026-03-18.
  ReafferencePredictor in ree_core/latent/stack.py. Enabled via reafference_action_dim
  in LatentStackConfig (0=disabled default; set to action_dim to enable). Applied in
  LatentStack.encode(): z_world_corrected = z_world_raw - ReafferencePredictor(z_self_prev, a_prev).
  LatentState.z_world_raw stores uncorrected value for training/diagnostic use.
  Biological basis: MSTd congruent/incongruent neurons (Gu et al. 2008). See MECH-098.

## SD Design Decisions Pending (V3)
- SD-008: encoder.z_world_alpha_correction — LatentStack.encode() EMA alpha for z_world
  must be >= 0.9 (not 0.3). MECH-089 theta buffer already handles temporal integration;
  the 0.3 encoder EMA double-smoothes z_world into a ~3-step average, suppressing event
  responses (Δz_world ≈ 0 on all events), trivialising E2_world prediction (MSE ≈ 0.005
  invariant to env perturbation), and preventing ARC-016 from firing (precision stuck at
  ~188). alpha_self may remain low (body state is genuinely autocorrelated). Evidence:
  EXQ-013 (event selectivity ≈ 0), EXQ-018 (precision invariant to drift_prob), EXQ-019
  (z_self more autocorrelated than z_world — backwards). See MECH-100.
  Config: LatentStackConfig.alpha_world (default 0.3 for compat; set to 0.9 or 1.0).
- SD-009: encoder.event_contrastive_supervision — z_world encoder requires event-type
  cross-entropy auxiliary loss during training (MECH-100). Reconstruction + E1-prediction
  losses are invariant to harm-relevance; only supervised event discrimination forces
  z_world to represent hazard-vs-empty distinctions. See EXQ-020.

## Experiment IDs
V3 experiments: V3-EXQ-001 onward
First priority: V3-EXQ-001 → V3-EXQ-002 → V3-EXQ-003 + V3-EXQ-004 (parallel)
