"""
V4-SD030-STUB: E2_self Motor-Proprioceptive Forward Model Validation (DESIGN STUB)

THIS IS NOT A RUNNABLE V3 EXPERIMENT. It is a documented design stub for the
V4 validation of SD-030 (self_attribution.comparator_z_self).

PURPOSE:
  Concrete structural placeholder so that when V4 materialises the z_self
  substrate (SD-005 expansion + DR-13 temporal depth), the experiment scaffold
  is ready to be filled in. Mirrors the ARC-033 / EXQ-264 pattern.

DO NOT QUEUE THIS SCRIPT. It intentionally raises NotImplementedError at
the substrate-construction step. It is checked into ree-v3/experiments/ so
that the design is discoverable next to the V4-active experiment scripts.

MECHANISM UNDER TEST (at V4):
  MECH-256 (general single-pass comparator) instantiated on z_self:
    predicted_z_self = E2SelfForward(z_self_{t-1}, a_actual)
    residual_z_self  = z_self_observed - predicted_z_self
    causal_sig_self  = f(residual_z_self, context)

  The comparator is stream-agnostic computationally (MECH-256). The substrate
  (E2SelfForward) is stream-specific. This experiment validates the substrate
  on the z_self stream following the same structural pattern that ARC-033 /
  EXQ-264 validate on z_harm_s.

V4 PREREQUISITES (none of which exist in V3 as of 2026-04-18):
  P1: SD-005 expansion -- z_self must be a first-class latent stream with a
      dedicated encoder and a stream-specific forward model substrate. Currently
      z_self exists as a single-layer MLP projection inside LatentStack; E2
      operates on z_gamma (combined).
  P2: DR-13 temporal depth -- z_self encoder must have recurrence or E1-feedback
      enrichment. A flat MLP is insufficient input for a forward model to
      predict meaningful transitions.
  P3: Dedicated E2SelfForward module at ree_core/predictors/e2_self.py with
      E2SelfConfig dataclass, residual-delta architecture, stop-gradient
      training discipline. Mirrors ree_core/predictors/e2_harm_s.py.
  P4: Environment extension -- CausalGridWorldV4 must support a "pushed"
      perturbation condition that modifies body state independently of the
      agent's action. Without this, the self_causal_gap metric (P2) cannot
      be computed because all z_self deltas would be self-initiated.
  P5: Body-frame action encoding (optional, quality improvement) -- richer
      than 4-direction one-hot; limb-specific motor commands so E2_self has
      discriminable action structure.

EXPERIMENT DESIGN (at V4):
  Two conditions, ablation pair:
    WITH_SELF_FWD:    E2SelfForward enabled; residual signal available to E3
    WITHOUT_SELF_FWD: baseline (z_self without comparator)

  Phased training:
    P0 (100 episodes): z_self encoder warmup (body-state supervision --
                       energy, limb_damage, residual_pain per SD-022)
    P1 ( 80 episodes): E2SelfForward training on frozen z_self
                       (stop-gradient on z_self_t inputs; identity-collapse
                       check via action-shuffled control)
    P2 ( 20 episodes): Evaluation (self_forward_r2 + self_causal_gap)

ACCEPTANCE CRITERIA (at V4):
  C1: self_forward_r2 > 0.85 on held-out trajectories (WITH_SELF_FWD)
  C2: self_forward_r2_shuffled < 0.3 on action-shuffled control
      (identity-collapse check: model must be using the action, not
      autocorrelation)
  C3: self_causal_gap > 0.1
      self_causal_gap = mean(||residual||_pushed) - mean(||residual||_self)
      (externally-imposed body-state changes produce larger residuals than
      self-initiated changes)

OPTIONAL INTERVENTIONAL EXTENSION (SD-013 analogue):
  After C1/C2/C3 pass, enable margin loss:
    margin_loss = ReLU(margin - ||E2_self(z_self, a_i) - E2_self(z_self, a_j)||)
  for a_i != a_j sampled from same state. Test prediction: observational
  training compresses causal_sig in confounded states; interventional
  restores identifiability.

DOWNSTREAM (at V4+):
  A working E2SelfForward unblocks:
    DR-12: E2 prediction error -> E3 confidence modulation on self-transitions
    DR-11: z_self-domain goal representation (z_goal_self seeded from z_self,
           evaluated via E2SelfForward rollout in evaluator mode per MECH-257)
    MECH-215: self-model capacity estimation for agentive prediction

RELATED CLAIMS:
  - MECH-256: general comparator mechanism (parent)
  - MECH-257: dual-function gated readout (comparator + evaluator)
  - SD-029: z_harm_s sibling (V3-active)
  - SD-031: z_world sibling (V4)
  - ARC-033: reference implementation template (on z_harm_s)
  - SD-005: self/world latent split (prerequisite)
  - SD-022: directional limb damage (body-state signal for z_self encoder warmup)

DESIGN DOC: REE_assembly/docs/architecture/sd_030_e2_self_forward_model.md
"""

import sys

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v4_sd030_e2_self_forward_stub"
CLAIM_IDS = ["SD-030", "MECH-256"]
V4_STUB = True  # Marker: do not queue

V4_PREREQ_MESSAGE = (
    "SD-030 E2_self forward model validation requires V4 substrate that does "
    "not exist in V3. Prerequisites:\n"
    "  P1: SD-005 expansion to dedicated z_self stream with its own forward model\n"
    "  P2: DR-13 z_self temporal depth (recurrence or E1-feedback enrichment)\n"
    "  P3: ree_core/predictors/e2_self.py module (E2SelfForward + E2SelfConfig)\n"
    "  P4: CausalGridWorldV4 'pushed' perturbation condition\n"
    "  P5 (optional): richer body-frame action encoding\n"
    "See REE_assembly/docs/architecture/sd_030_e2_self_forward_model.md"
)


def main():
    """Stub entry point. Fails loudly with V4-prerequisite message."""
    print("[v4_sd030_e2_self_forward_stub] DESIGN STUB -- not runnable in V3.")
    print("")
    print(V4_PREREQ_MESSAGE)
    print("")
    print("This script is a design placeholder for V4 SD-030 validation.")
    print("It is checked in so the experiment scaffold is discoverable next")
    print("to the V3-active ARC-033 validation (v3_exq_264_arc033_e2_harm_s_forward.py).")
    raise NotImplementedError(V4_PREREQ_MESSAGE)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        # Exit cleanly with non-zero to ensure this never accidentally
        # produces a "PASS" manifest if invoked.
        print("NotImplementedError:", e, file=sys.stderr)
        sys.exit(2)
