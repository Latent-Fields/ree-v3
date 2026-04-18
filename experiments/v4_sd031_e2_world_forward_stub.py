"""
V4-SD031-STUB: E2_world Causal-Footprint Forward Model Validation (DESIGN STUB)

THIS IS NOT A RUNNABLE V3 EXPERIMENT. It is a documented design stub for the
V4 validation of SD-031 (self_attribution.comparator_z_world).

PURPOSE:
  Concrete structural placeholder so that when V4 factors E2.world_forward
  into a dedicated E2WorldForward module, the experiment scaffold is ready to
  be filled in. Mirrors the ARC-033 / EXQ-264 pattern.

DO NOT QUEUE THIS SCRIPT. It intentionally raises NotImplementedError at
the substrate-construction step. It is checked into ree-v3/experiments/ so
that the design is discoverable next to the V4-active experiment scripts.

MECHANISM UNDER TEST (at V4):
  MECH-256 (general single-pass comparator) instantiated on z_world:
    predicted_z_world = E2WorldForward(z_world_{t-1}, a_actual)
    residual_z_world  = z_world_observed - predicted_z_world
    causal_sig_world  = f(residual_z_world, context)

  Supersedes SD-003 two-pass counterfactual on z_world (validated EXQ-030b
  PASS, world_forward_r2=0.947) by re-reading the same substrate single-pass
  per MECH-256. Bookkeeping change at the mechanism level; substrate already
  demonstrated adequate.

V4 PREREQUISITES:
  P1: SD-005 expansion -- z_world must be a first-class latent stream
      independently of z_gamma / z_self. Currently z_world is the world-
      component of the split latent but E2 still operates on z_gamma.
  P2: Factor E2FastPredictor.world_forward out into a dedicated
      E2WorldForward module at ree_core/predictors/e2_world.py with
      E2WorldConfig dataclass, residual-delta architecture, stop-gradient
      training discipline. Mirrors ree_core/predictors/e2_harm_s.py.
  P3: Environment extension -- CausalGridWorldV4 must support an
      "exogenous perturbation" condition where world state changes happen
      independently of the agent's action (e.g. a block moves without the
      agent pushing it). Without this, the world_causal_gap metric (P2)
      cannot be computed because all z_world deltas would be agent-caused.
  P4: SD-013 interventional training extension applicable here with high
      priority (world state has strong ambient correlations; observational
      training risks compressing causal_sig).

EXPERIMENT DESIGN (at V4):
  Two conditions, ablation pair:
    WITH_WORLD_FWD:    E2WorldForward factored out + single-pass comparator
                       reading; residual signal available to E3 + ResidueField
    WITHOUT_WORLD_FWD: baseline (implicit E2.world_forward only, no
                       comparator signal, legacy V3 residue integration)

  Phased training:
    P0 (already V3-established): z_world encoder (event-CE per SD-009,
                                 resource-proximity per SD-018). No additional
                                 warmup needed.
    P1 ( 80 episodes): E2WorldForward training on frozen z_world
                       (stop-gradient on z_world_t inputs; identity-collapse
                       check via action-shuffled control)
    P2 ( 20 episodes): Evaluation (world_forward_r2 + world_causal_gap)
    P3 (optional, 40 episodes): interventional training with margin loss,
                                re-test world_causal_gap

ACCEPTANCE CRITERIA (at V4):
  C1: world_forward_r2 > 0.9 on held-out trajectories (WITH_WORLD_FWD)
      (EXQ-030b baseline 0.947 confirms achievable)
  C2: world_forward_r2_shuffled < 0.4 on action-shuffled control
      (identity-collapse check)
  C3: world_causal_gap > 0.1
      world_causal_gap = mean(||residual||_perturbed) - mean(||residual||_self)
      (exogenous world-state changes produce larger residuals than
      self-caused changes)
  C4 (interventional, optional): world_causal_gap_interventional >
      world_causal_gap_observational
      (confounded-state compression recovered by margin loss)

DOWNSTREAM (at V4+):
  A working E2WorldForward unblocks:
    - Moral attribution for non-harm consequences (property damage, broken
      promises, agent-caused world changes generally)
    - Residue weighting by attribution strength -- ResidueField stops
      integrating externally-caused transitions with full weight
    - Causal chain tracking across multi-step action sequences
    - Decoupling of world attribution from harm attribution (finalises
      post-SD-010 topology; replaces infeasible HarmBridge design)

CAUSAL CHAIN TEST (V4+ extension, not in baseline validation):
  Scripted trial where agent's action at t-2 produces a world change at t that
  requires chaining two forward predictions to attribute. Residue-field
  integration with SD-031 attribution weights tracks multi-step credit;
  ablated residue field without attribution weights does not.

MORAL ATTRIBUTION TASK (V4 social extension):
  Multi-agent environment where the agent must avoid blame for events caused
  by other agents operating in the same state space. Agents with working
  SD-031 show selective avoidance (large world_residual = other agent did
  it, not my responsibility). Ablated agents cannot discriminate.

RELATED CLAIMS:
  - MECH-256: general comparator mechanism (parent)
  - MECH-257: dual-function gated readout (comparator + evaluator)
  - SD-029: z_harm_s sibling (V3-active)
  - SD-030: z_self sibling (V4)
  - SD-003: z_world counterfactual (superseded by SD-031 single-pass reading)
  - ARC-033: reference implementation template (on z_harm_s)
  - SD-005: self/world latent split (prerequisite)
  - SD-010: harm-stream separation (prerequisite -- ensures z_world distinct
    from z_harm so SD-031 reading is unambiguous)
  - ARC-017 / MECH-096 / MECH-103: ResidueField downstream consumer
  - SD-013: interventional training extension

DESIGN DOC: REE_assembly/docs/architecture/sd_031_e2_world_forward_model.md
"""

import sys

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v4_sd031_e2_world_forward_stub"
CLAIM_IDS = ["SD-031", "MECH-256"]
V4_STUB = True  # Marker: do not queue

V4_PREREQ_MESSAGE = (
    "SD-031 E2_world forward model validation requires V4 module refactor "
    "that does not exist in V3. Prerequisites:\n"
    "  P1: SD-005 expansion to dedicated z_world stream\n"
    "  P2: ree_core/predictors/e2_world.py module (factor out "
    "E2FastPredictor.world_forward)\n"
    "  P3: CausalGridWorldV4 exogenous perturbation condition\n"
    "  P4: SD-013 interventional training extension (high priority for world "
    "stream due to ambient correlations)\n"
    "See REE_assembly/docs/architecture/sd_031_e2_world_forward_model.md\n"
    "Note: EXQ-030b already validated the counterfactual version on the V3 "
    "substrate (world_forward_r2=0.947). SD-031 is a re-reading of that "
    "substrate as a single-pass comparator per MECH-256."
)


def main():
    """Stub entry point. Fails loudly with V4-prerequisite message."""
    print("[v4_sd031_e2_world_forward_stub] DESIGN STUB -- not runnable in V3.")
    print("")
    print(V4_PREREQ_MESSAGE)
    print("")
    print("This script is a design placeholder for V4 SD-031 validation.")
    print("It is checked in so the experiment scaffold is discoverable next")
    print("to the V3-active ARC-033 validation (v3_exq_264_arc033_e2_harm_s_forward.py).")
    raise NotImplementedError(V4_PREREQ_MESSAGE)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print("NotImplementedError:", e, file=sys.stderr)
        sys.exit(2)
