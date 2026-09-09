"""V3-EXQ-1017 -- Regulatory anchoring vs a matched arbitrary auxiliary: what survives the
observation -> z_world compression, read at the consumer (INV-104 / ARC-138, GOV-MATCHAUX-1).

red-team (opus): BLOCKING on the first pass (11 findings), re-reviewed once after the fixes below.
  F1/F2 (BLOCKING) no in-run witness that ANY P0a objective moves the adapter DV, so an insensitive
  instrument routed to "both claims weaken" -> FIXED: criterion C0 (sensitivity witness) and the
  pre-registered route p0a_objective_invisible_to_adapter_dv (non_contributory, non_degenerate
  false) ahead of every claim branch. F3 C3's non-degeneracy flag was the negation of C3 -> FIXED
  (keyed to C0). F4 C5 override fired inside the PASS branch -> FIXED (outside PASS only). F5 the
  matched target (per-cell wall/empty functional) predicted the oracle's ACTION 5-17 points above
  chance -> FIXED: cell-permutation-invariant COUNT functional (measured -0.03..-0.01 elevation)
  plus gate matched_target_action_irrelevant. F6 one-sided R2 floor admitted a B/C supervision
  mismatch -> FIXED: paired gate aux_supervision_matched. F7 matching certified on the calibration
  draw, not the training rollout -> FIXED: certify() on the identical training rollout. F8 the gate
  reads the trainer path, the DV the sense path -> ADDRESSED: sense-path own-target R2 and the
  arm-invariant relative residual recorded; the gate stays on the trainer path (it certifies that
  supervision was delivered to the part of the DV the manipulation can touch). F9 C4/C6 unread ->
  named as witnesses. F10 falsy-zero readbacks -> FIXED. F11 CFULL reproduction not gated ->
  witness criterion C9 (a failed reproduction does not invalidate the within-run paired contrast).
  RE-REVIEW (fresh opus reviewer): CONTESTED. (a) C0 admitted |A-UNT|/|CFULL-UNT|, so a purely
  destructive encoder move would pass it while every objective-choice contrast sat at zero ->
  FIXED: C0 now reads only contrasts BETWEEN trained arms. NEW nan-unsafe elevation readback ->
  FIXED (isfinite guard). NEW: KS gate and B's aux_target_learned floor unmeasured against the
  count target at full scale -> KS measured on the training rollout at 90 calibration episodes
  (0.147 / 0.071 / 0.179 on seeds 42/43/44 vs 0.25); B's own-target R2 measured once at full P0a
  scale before queueing (see the queue-entry note). F8 stays a recorded dismissal: the gate
  certifies that supervision was delivered at the one site the manipulation can reach; the
  sense-path R2 and the arm-invariant relative residual are recorded so a flat result is read
  against them. F7's identical-rollout claim was verified line by line by the re-reviewer.
SLEEP DRIVER: none (no sleep flag is set; the run never enters a non-waking state).

QUESTION (new registry qid `regulatory_anchoring_matched_aux`). The 2026-09-04 regulation-first
cluster (thought_intake_2026-09-04_regulation_first_organizing_subjective_experience.md) predicts
that anchoring z_world to ORGANISM-RELEVANT structure organises the latent differently from
generic extra supervision of equal capacity and cost. V3-EXQ-978 read a 1.5x reweighting as
supervision-vs-none and had no matched control (its autopsy, section 4); V3-EXQ-1002 then showed
the frozen lineage z_world does NOT carry the oracle's decision content in a reader-usable form
(0.67 vs a 0.80 bar, below the untrained control), V3-EXQ-1008 that the loss is CONTENT
discarded at encode (no information-preserving re-basis lifts it; PCA-32 of the encoder's own
input reaches 0.87). So the question is asked exactly where the content is lost: does the P0a
OBJECTIVE decide what survives the encoder, and is an organism-relevant target special?

THREE CONDITIONS, identical everywhere except the P0a objective (the chip's A/B/C):
  A  zworld_p0_generic           SD-070 anti-collapse + reconstruction ONLY (presence/distance/
                                 proximity weights 0). The generic-only compression INV-104's
                                 falsifying route F1 is a statement about -- previously declared
                                 unconstructible (EVB-1712) because run_zworld_p0 hardcoded the
                                 default config; constructible now via the config= seam.
  B  zworld_p0_matched_arbitrary A + the SD-018 proximity HEAD trained at proximity_weight 0.5 on
                                 a MATCHED ARBITRARY target: a fixed linear functional of the
                                 per-entity COUNTS over the 25 local-view cells (non-resource,
                                 non-hazard slots) -- cell-permutation-invariant, so it cannot
                                 point, and the oracle's action is a direction -- decorrelated
                                 from the resource-proximity target and quantile-mapped onto its
                                 marginal (experiments/_lib/matched_aux_targets.py). Same head,
                                 loss, weight, cadence, examples as C; organism-irrelevant
                                 meaning. Matching is MEASURED on the P0a training rollout itself
                                 and gated (KS <= 0.25, |r| <= 0.20, oracle-action decodability
                                 elevation <= 0.04, |R2_C - R2_B| <= 0.20), never assumed.
  C  zworld_p0_regulatory_prox   A + the same head at the same weight on the SD-018 resource
                                 proximity target (the organism-relevant scalar the lineage
                                 already supervises).
plus three anchors: CFULL zworld_p0_sd070_default (the full SD-070 recipe = V3-EXQ-1008's
zworld_off arm, expected to REPRODUCE 1008's 0.672/0.674/0.661 bit-for-bit -- same agent build,
same seeds, same P0a, same dataset, same adapter); UNT zworld_untrained (the architectural floor,
1002/1008 measured 0.699/0.681/0.704); PCA ws250_pca (an information-preserving LINEAR 250->32
compression of the encoder's own input, 1008 measured 0.877/0.858/0.870 -- the ceiling at the
encoder's width, and the achievable-range witness for the pairwise criterion); RAW
rawfield_ceiling (the adapter on the raw 25-dim field, 1002/1008 0.985/0.980/0.973 -- the
instrument's positive control).

WHY P0b / P1 ARE NOT RUN (a deliberate departure from 1002/1008's warmup). The lineage's P0b
(e2 contrastive) and P1 (REINFORCE) train NO latent_stack parameter (zworld_p0_warmup.py
docstring: no optimizer group covers it), and nothing this run reads depends on e2 or the
policy: the DV is the frozen SENSE-time z_world read by a fresh supervised adapter, the rollout
secondary drives the ADAPTER not E3, and the P0a stats come from the trainer. P0a runs FIRST
in _train_all_on_agent under RNG neutrality, so the encoder this run trains is the one 1008's
warmup produced; CFULL's reproduction of 1008 is the recorded check on that claim
(`reproduces_1008_within` per seed). Skipping P0b/P1 removes ~290 CEM-planning episodes per
warmup and nothing else.

DV (primary, all arms): held-out oracle-action agreement of a capacity-matched supervised
adapter (x1002's 21k-parameter PPOPolicyNet, cross-entropy on the local_view_greedy oracle's
action) reading the FROZEN sense-time z_world (x737._agent_zworld), on 1002's episode-split
dataset, standardised on the train split -- 1002's instrument, unchanged. Auxiliary heads are
never read at inference (the adapter sees z_world only) -- the GOV-MATCHAUX-1 head-removal
requirement holds by construction. Secondaries: participation ratio of the raw latent; each
arm's own-target held-out R2 through the trained encoder + proximity head (the "B learns its
nuisance target as well as C learns its own" witness); P0a holdout lifts; rolled-out foraging of
the adapter; the retained fraction R = (arm - UNT) / (PCA - UNT) (EXP-1397's R_k, class 1).

PRE-REGISTERED CRITERIA (paired by seed, majority 2 of 3; DELTA_MIN = 0.05, FLAT_EPS = 0.02):
  C0  objective_reaches_dv      max(|CFULL-A|, |B-A|, |C-A|, |C-B|) >= DELTA_MIN
                                (the SENSITIVITY WITNESS: some CHOICE of P0a objective is
                                visible to the adapter DV; trained-vs-untrained is a witness
                                only -- a destructive move is not sensitivity; 1008's full-
                                recipe-vs-untrained was -0.026, so this is genuinely uncertain)
  C1  anchoring_beats_matched   agreement(C) - agreement(B) >= DELTA_MIN     [load-bearing]
  C2  anchoring_beats_generic   agreement(C) - agreement(A) >= DELTA_MIN     [load-bearing]
  C3  matched_control_flat      |agreement(C) - agreement(B)| < FLAT_EPS
  C4  matched_beats_generic     agreement(B) - agreement(A) >= DELTA_MIN
  C5  generic_preserves_f1      R(A) >= 0.75                                  (INV-104 F1)
  C6  destruction_real_all      R(A), R(C), R(CFULL) all <= 0.25              (INV-104 (i))
VERDICT GRID (pure function, --self-test):
  not C0               -> p0a_objective_invisible_to_adapter_dv            FAIL
                          both non_contributory, non_degenerate false: "matched control matches
                          anchoring" and "the DV cannot see any objective" are indistinguishable
                          here, so neither is recorded
  C1 and C2            -> regulatory_anchoring_beats_matched_control      PASS
                          ARC-138 supports; INV-104 supports
  C3                   -> matched_control_matches_anchoring               FAIL
                          ARC-138 weakens; INV-104 weakens (the class-1 scaffold does nothing an
                          arbitrary one does not -- the cluster's own "reduces pressure to
                          complicate z_world" reading)
  C2 and not C1        -> generic_auxiliary_supervision_effect            FAIL
                          ARC-138 non_contributory (pre-registered "not evidence"); INV-104 mixed
  otherwise            -> inconsistent_across_seeds                       FAIL, both mixed
  C5 additionally      -> INV-104 weakens on every FAIL branch (generic already preserves >= 75%
                          of the width ceiling's room and anchoring adds nothing material: the
                          preservation requirement is vacuous for class 1 at this site); NOT on
                          the PASS branch, where the extra access anchoring delivers is real
  any readiness gate red -> substrate_not_ready_requeue / instrument_not_ready, FAIL,
                          both non_contributory, non_degenerate false.
DV HEADROOM (declared, measured in-run): the paired criterion needs a 0.05 movement of a 32-dim
latent's agreement; the achievable movement at that width on this dataset is PCA - UNT (1008:
~0.17 per seed), passed to dv_headroom_check as `achievable` with margin 2.0.

READINESS GATES: 1002's six (raw-field positive control >= 0.60; oracle label majority <= 0.60;
>= 500 held-out steps; encoder weight delta > 1e-6 on trained arms; participation ratio >= 2 on
z arms; local_view_greedy clears the D3 floor) + six new, each scoped to the arms it certifies:
ws250_pca_reaches_bar (>= 0.80, z arms: the R denominator); on B, all certified on the P0a
TRAINING ROLLOUT: matched_target_marginal_matched (KS <= 0.25), matched_target_decorrelated
(|r| <= 0.20), matched_target_action_irrelevant (oracle-action decodability elevation <= 0.04),
aux_supervision_matched (|own-target R2_C - R2_B| <= 0.20); aux_target_learned (own-target
held-out R2 >= 0.10, B / C / CFULL). Witnesses recorded, not gated: sense-path own-target R2 and
the sense-vs-encoder-path relative residual (the arm-invariant top-down + smoothing component
the P0a objective cannot touch); CFULL's reproduction of 1008 (C9).

CLAIMS. claim_ids = [INV-104, ARC-138]; GOV-MATCHAUX-1 is the design rule, not tested. NEVER
INV-088 / MECH-457 (re-derive brake fired at the 978 autopsy). Read-across only: SD-070, SD-018,
MECH-517, MECH-523. EXPERIMENT_PURPOSE = "evidence" -- the grid routes claim verdicts directly;
every readiness gate is emitted as a precondition so an unready run is scoring_excluded.

GOV-REUSE-1: decisive readout = paired held-out oracle_action_agreement between P0a-objective
arms. No recorded run carries a generic-only or matched-arbitrary P0a arm (the seam did not exist);
UNT / PCA / RAW / CFULL are re-run rather than cited because the arms are paired per seed on the
same re-collected dataset and re-running them costs minutes, not hours.

Ethics preflight: all false / not_applicable, decision allow (V3, pre-ethical instrumentation).
ASCII-only output. Runner conformance: emit_outcome at the end of __main__.
"""
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import (  # noqa: E402
    P0NotReady,
    dv_headroom_check,
    p0_readiness_gate,
)
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import Policy, RandomPolicy, evaluate_seed  # noqa: E402
from experiments._lib.matched_aux_targets import (  # noqa: E402
    MatchedArbitraryTarget,
    collect_calibration_obs,
)
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    latent_stack_weight_delta,
)
from experiments._lib.zworld_p0_warmup import resource_prox_target, run_zworld_p0  # noqa: E402
from ree_core.latent.zworld_p0 import ZWorldP0Config  # noqa: E402

# IMPORTED, NEVER REDEFINED: dataset recipe, adapter, standardiser, scoring, agent build (1002);
# projection + cell helpers (1008). This run's anchors are reproductions of theirs.
import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
import experiments.v3_exq_1008_zworld_adequacy_portfolio_ws250_rebasis as x1008  # noqa: E402
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_737_ree_latent_policy_head_competence_probe as x737  # noqa: E402
import experiments.v3_exq_808_return_decomposition_objective_misspecification as x808  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1017_inv104_arc138_regulatory_anchoring_matched_aux"
QUEUE_ID = "V3-EXQ-1017"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = ["INV-104", "ARC-138"]
HYPOTHESIS_QID = "regulatory_anchoring_matched_aux"
HYPOTHESIS_HIDS = {
    "anchoring": "H-RA-anchoring-organises",
    "generic_supervision": "H-RA-generic-supervision-effect",
    "matched_null": "H-RA-matched-control-null",
    "f1": "H-RA-generic-already-preserves",
}
DEVICE = x1002.DEVICE

SEEDS: List[int] = list(x1002.SEEDS)                # [42, 43, 44] -- the 978/1002/1008 seeds

# ---- IMPORTED CONSTANTS (never re-typed) ----------------------------------------------------
ZWORLD_P0_EPISODES = x1002.ZWORLD_P0_EPISODES        # 60
STEPS_PER_EPISODE = x1002.STEPS_PER_EPISODE          # 200
EVAL_EPISODES = x1002.EVAL_EPISODES                  # 20
BC_EPISODES = x1002.BC_EPISODES                      # 40
BC_RANDOM_EPISODES = x1002.BC_RANDOM_EPISODES        # 20
ADAPTER_PASSES = x1002.ADAPTER_PASSES                # 60
SEED_MAJORITY = x1002.SEED_MAJORITY                  # 2 of 3
RUNG, RUNG_ID, LEVEL_ID = x1002.RUNG, x1002.RUNG_ID, x1002.LEVEL_ID
RESOURCE_FIELD_DIM = x1002.RESOURCE_FIELD_DIM        # 25
AGREEMENT_BAR = x1002.AGREEMENT_BAR                  # 0.80
AGREEMENT_ELEVATION_MIN = x1002.AGREEMENT_ELEVATION_MIN  # 0.20
RAW_FIELD_CONTROL_FLOOR = x1002.RAW_FIELD_CONTROL_FLOOR  # 0.60
PARTICIPATION_RATIO_FLOOR = x1002.PARTICIPATION_RATIO_FLOOR  # 2.0
PROJECTION_DIM = x1008.PROJECTION_DIM                # 32

# ---- NEW pre-registered constants -------------------------------------------------------------
DELTA_MIN = 0.05          # paired agreement difference that counts as "beats"
FLAT_EPS = 0.02           # |C - B| below this on the seed majority = matched control matches
RETAINED_HIGH = 0.75      # R(A) at or above this = INV-104 F1 (generic already preserves)
RETAINED_LOW = 0.25       # R at or below this = destruction real (EXP-1397 condition (i))
KS_MAX = 0.25             # matched-target marginal vs prox marginal, held-out calibration half
R_MAX = 0.20              # |Pearson r| of the matched target with prox, held-out calibration half
AUX_R2_FLOOR = 0.10       # own-target held-out R2 through encoder + prox head, aux-trained arms
AUX_R2_GAP_MAX = 0.20     # |R2(C) - R2(B)|: achieved-supervision mismatch the matched pair may carry
ACTION_ELEV_MAX = 0.04    # matched target's oracle-ACTION decodability elevation over majority class
CALIBRATION_EPISODES = 90 # random-walk episodes the matched target is fitted/certified on
HOLDOUT_EPISODES = 10     # random-walk episodes the own-target R2 is measured on
P0A_PROX_WEIGHT = 0.5     # == ZWorldP0Config.proximity_weight default; B and C share it
# V3-EXQ-1008 zworld_off_diag per seed (42, 43, 44) -- CFULL's reproduction witness, recorded
# alongside the fresh value. Not a gate.
X1008_ZWORLD_OFF = {42: 0.6718, 43: 0.6735, 44: 0.6606}
REPRODUCTION_TOL = 0.005

# ---- arms ---------------------------------------------------------------------------------------
ARM_RAW = x1002.ARM_RAW                  # "rawfield_ceiling"
ARM_PCA = "ws250_pca"
ARM_UNT = "zworld_untrained"
ARM_A = "zworld_p0_generic"
ARM_B = "zworld_p0_matched_arbitrary"
ARM_C = "zworld_p0_regulatory_prox"
ARM_CFULL = "zworld_p0_sd070_default"
ARM_IDS = [ARM_RAW, ARM_PCA, ARM_UNT, ARM_A, ARM_B, ARM_C, ARM_CFULL]
Z_ARMS = [ARM_UNT, ARM_A, ARM_B, ARM_C, ARM_CFULL]
TRAINED_Z_ARMS = [ARM_A, ARM_B, ARM_C, ARM_CFULL]
AUX_TRAINED_ARMS = [ARM_B, ARM_C, ARM_CFULL]
ANCHOR_ARMS = [ARM_RAW, ARM_UNT, ARM_CFULL]        # get 1002's unstandardised secondary

DRY_RUN_SEEDS = [42]
DRY_RUN_ZWORLD_P0 = x1002.DRY_RUN_ZWORLD_P0
DRY_RUN_EVAL = x1002.DRY_RUN_EVAL
DRY_RUN_STEPS = x1002.DRY_RUN_STEPS
DRY_RUN_BC_EPISODES = x1002.DRY_RUN_BC_EPISODES
DRY_RUN_BC_RANDOM_EPISODES = x1002.DRY_RUN_BC_RANDOM_EPISODES
DRY_RUN_ADAPTER_PASSES = x1002.DRY_RUN_ADAPTER_PASSES
DRY_RUN_CALIBRATION_EPISODES = 6
DRY_RUN_HOLDOUT_EPISODES = 3

_EXTRA_SUBSTRATE_PATHS = [Path(m.__file__) for m in (x1002, x1008, x724, x734, x737, x808)]
# The linter cannot see through `_config_slice(base, arm_id)` to `_base_slice`, where
# P0A_PROX_WEIGHT IS declared (`p0a_prox_weight`), and it is declared a second time per arm as
# `arm_p0a_config.proximity_weight`. Both are in every cell's slice.
CONFIG_SLICE_DECLARATION_EXEMPT = (
    "P0A_PROX_WEIGHT is declared in the slice as p0a_prox_weight (base slice) and as "
    "arm_p0a_config.proximity_weight (per arm)")
_ZG = ZGoalStreamAccumulator()


# --------------------------------------------------------------------------------------
# P0a OBJECTIVES per arm -- the ONLY thing that differs between A / B / C / CFULL
# --------------------------------------------------------------------------------------
def p0a_config_for(arm_id: str) -> Optional[ZWorldP0Config]:
    if arm_id == ARM_A:
        return ZWorldP0Config(presence_weight=0.0, distance_weight=0.0, proximity_weight=0.0)
    if arm_id in (ARM_B, ARM_C):
        return ZWorldP0Config(presence_weight=0.0, distance_weight=0.0,
                              proximity_weight=float(P0A_PROX_WEIGHT))
    if arm_id == ARM_CFULL:
        return ZWorldP0Config()          # the SD-070 default = 1008's zworld_off P0a
    return None


def p0a_target_kind(arm_id: str) -> str:
    return {ARM_B: "matched_arbitrary", ARM_C: "resource_proximity",
            ARM_CFULL: "resource_proximity"}.get(arm_id, "none")


# --------------------------------------------------------------------------------------
# PRECONDITIONS: 1002's six + four new, each scoped to the arms it certifies
# --------------------------------------------------------------------------------------
def _arm_ctx(aid: str) -> Dict[str, Any]:
    return {"id": aid,
            "has_encoder": aid in Z_ARMS,
            "trained_encoder": aid in TRAINED_Z_ARMS,
            "matched": aid == ARM_B,
            "aux_trained": aid in AUX_TRAINED_ARMS,
            "projected": aid == ARM_PCA}


def _arm_contexts() -> List[Dict[str, Any]]:
    return [_arm_ctx(a) for a in ARM_IDS]


NEW_PRECONDITION_SPECS = [
    PreconditionSpec(
        name="ws250_pca_reaches_bar",
        description=("RETAINED-FRACTION DENOMINATOR. An information-preserving LINEAR 250->32 "
                     "compression of the encoder's own input must reach the verdict bar on the "
                     "worst seed, or R = (arm - UNT)/(PCA - UNT) has no ceiling to be a fraction "
                     "of and the pairwise headroom witness is not licensed. 1008 measured "
                     "0.858-0.877."),
        control="ws250_pca worst-seed held-out oracle_action_agreement",
        threshold=float(AGREEMENT_BAR), direction="lower", kind="readiness",
        applies_to=lambda ctx: bool(ctx["has_encoder"]),
        applies_note="certifies the width ceiling the z_world arms are read against",
        structural_max=lambda ctx: 1.0,
    ),
    PreconditionSpec(
        name="matched_target_marginal_matched",
        description=("GOV-MATCHAUX-1 MATCHING (i). Kolmogorov-Smirnov distance between the "
                     "matched arbitrary target's marginal and the resource-proximity target's, "
                     "certified on the P0a TRAINING ROLLOUT itself (same env seed + RandomPolicy "
                     "seed reproduce the identical random walk; red-team finding 7), worst = "
                     "largest seed. Equal marginals = equal variance and entropy of the regression "
                     "target. The held-out calibration-half value is reported alongside."),
        control="largest-seed KS distance on the training-rollout certification set",
        threshold=float(KS_MAX), direction="upper", kind="readiness",
        applies_to=lambda ctx: bool(ctx["matched"]),
        applies_note="only the matched-arbitrary arm has a matched target to certify",
        structural_min=lambda ctx: 0.0,
    ),
    PreconditionSpec(
        name="matched_target_decorrelated",
        description=("GOV-MATCHAUX-1 MATCHING (ii). |Pearson r| between the matched arbitrary "
                     "target and the resource-proximity target on the P0a training rollout "
                     "(worst = largest seed). Bounds how much of the organism-relevant SCALAR the "
                     "'irrelevant' target carries linearly."),
        control="largest-seed |r| on the training-rollout certification set",
        threshold=float(R_MAX), direction="upper", kind="readiness",
        applies_to=lambda ctx: bool(ctx["matched"]),
        applies_note="only the matched-arbitrary arm has a matched target to certify",
        structural_min=lambda ctx: 0.0,
    ),
    PreconditionSpec(
        name="matched_target_action_irrelevant",
        description=("GOV-MATCHAUX-1 MATCHING (iii) -- the LEAK witness (red-team finding 5). How "
                     "much of the DV's own label (the local_view_greedy oracle's ACTION) the "
                     "matched target predicts beyond the majority class, quantile-bin majorities "
                     "fitted on the calibration half and scored on the training rollout (worst = "
                     "largest seed). Decorrelation from the proximity SCALAR does not bound this: "
                     "a per-cell wall/empty functional predicted the action 5-17 points above "
                     "chance at authoring time, which is why the target is a cell-permutation-"
                     "invariant COUNT functional. The proximity target's own elevation is "
                     "recorded alongside as the reference."),
        control="largest-seed action-decodability elevation of the matched target on the training rollout",
        threshold=float(ACTION_ELEV_MAX), direction="upper", kind="readiness",
        applies_to=lambda ctx: bool(ctx["matched"]),
        applies_note="only the matched-arbitrary arm has a nuisance target that could leak the label",
        structural_min=lambda ctx: 0.0,
    ),
    PreconditionSpec(
        name="aux_supervision_matched",
        description=("GOV-MATCHAUX-1 MATCHING (iv) -- achieved supervision is PAIRED, not merely "
                     "floored (red-team finding 6): |R2_C - R2_B| of own-target held-out R2 "
                     "through encoder + prox head, per seed (worst = largest). A one-sided floor "
                     "would admit B learning a tenth of what C learned, the difficulty mismatch "
                     "the rule exists to exclude."),
        control="largest-seed |own-target R2(C) - own-target R2(B)|",
        threshold=float(AUX_R2_GAP_MAX), direction="upper", kind="readiness",
        applies_to=lambda ctx: bool(ctx["matched"]),
        applies_note="a property of the B/C PAIR, recorded on the matched arm",
        structural_min=lambda ctx: 0.0,
    ),
    PreconditionSpec(
        name="aux_target_learned",
        description=("The auxiliary head actually learned its OWN target through the trained "
                     "encoder: held-out R2 of resource_proximity_head(z_world) against the arm's "
                     "target on fresh random-walk episodes (worst = smallest seed). A B arm whose "
                     "head learned nothing is not a matched control; a C arm whose head learned "
                     "nothing received no regulatory pressure. SD-070 validation measured prox "
                     "R2 0.20-0.38 through the trained encoder."),
        control="smallest-seed own-target held-out R2",
        threshold=float(AUX_R2_FLOOR), direction="lower", kind="readiness",
        applies_to=lambda ctx: bool(ctx["aux_trained"]),
        applies_note="A trains no auxiliary head; UNT trains nothing; PCA/RAW have no encoder",
        structural_max=lambda ctx: 1.0,
    ),
]
PRECONDITION_SPECS = list(x1002.PRECONDITION_SPECS) + NEW_PRECONDITION_SPECS


# --------------------------------------------------------------------------------------
# CONFIG SLICE
# --------------------------------------------------------------------------------------
def _base_slice(sched: Dict[str, int], dry_run: bool) -> Dict[str, Any]:
    return {
        "env_kwargs": x734._env_kwargs_for_rung(RUNG), "rung_id": RUNG_ID, "level_id": LEVEL_ID,
        "zworld_p0_episodes": int(sched["zworld_p0"]), "steps_per_episode": int(sched["steps"]),
        "p0b_p1_skipped": True,
        "calibration_episodes": int(sched["calib"]), "holdout_episodes": int(sched["holdout"]),
        "agent_build": "x1002._make_agent (use_resource_field_head, field_dim 25, P0a field weight 0.5 declared)",
        "use_resource_field_head": True, "resource_field_dim": int(RESOURCE_FIELD_DIM),
        "p0a_prox_weight": float(P0A_PROX_WEIGHT),
        "bc_episodes": int(sched["bc_eps"]), "bc_random_episodes": int(sched["bc_rand"]),
        "bc_train_frac": float(x1002.BC_TRAIN_FRAC),
        "adapter_passes": int(sched["passes"]), "adapter_batch": int(x1002.ADAPTER_BATCH),
        "adapter_lr": float(x1002.ADAPTER_LR), "adapter_trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
        "adapter_init_reseeded_per_fit": True,
        "feature_standardisation": ("train_split_zscore" if x1002.STANDARDISE_FEATURES else "none"),
        "standardiser_eps": float(x1002.STANDARDISER_EPS),
        "eval_episodes": int(sched["eval_eps"]), "projection_dim": int(PROJECTION_DIM),
        "dry_run": bool(dry_run),
    }


def _config_slice(base: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    d = dict(base)
    d["arm_id"] = arm_id
    d["arm_input"] = ("resource_field_view" if arm_id == ARM_RAW
                      else "world_state" if arm_id == ARM_PCA else "z_world")
    d["arm_projection"] = "pca" if arm_id == ARM_PCA else "none"
    cfg = p0a_config_for(arm_id)
    d["arm_p0a_config"] = (None if cfg is None else {
        "variance_weight": cfg.variance_weight, "covariance_weight": cfg.covariance_weight,
        "presence_weight": cfg.presence_weight, "distance_weight": cfg.distance_weight,
        "proximity_weight": cfg.proximity_weight, "resource_field_weight": cfg.resource_field_weight,
        "reconstruction_weight": cfg.reconstruction_weight, "learning_rate": cfg.learning_rate,
        "epochs": cfg.epochs, "batch_size": cfg.batch_size})
    d["arm_p0a_target"] = p0a_target_kind(arm_id)
    d["arm_warmup_skipped"] = bool(arm_id in Z_ARMS and arm_id not in TRAINED_Z_ARMS)
    return d


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
def _r2(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    if pred.size < 3:
        return None
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None


def _own_target_r2(agent, target_fn: Callable[[Dict[str, Any]], Optional[float]],
                   holdout_obs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Held-out R2 of resource_proximity_head(trainer z_world path) against `target_fn`."""
    se = agent.latent_stack.split_encoder
    head = getattr(se, "resource_proximity_head", None)
    if head is None:
        return {"r2": None, "n": 0, "reason": "no_resource_proximity_head"}
    ws, ts = [], []
    for o in holdout_obs:
        t = target_fn(o)
        if t is None or not np.isfinite(t):
            continue
        ws.append(o["world_state"].detach().float().reshape(-1))
        ts.append(float(t))
    if len(ws) < 3:
        return {"r2": None, "n": len(ws), "reason": "too_few_labelled_steps"}
    with torch.no_grad():
        z = se.world_encoder(torch.stack(ws)) * torch.sigmoid(se.world_precision_logit).unsqueeze(0)
        pred = head(z).reshape(-1).cpu().numpy().astype(np.float64)
    tgt = np.asarray(ts, dtype=np.float64)
    return {"r2": _r2(pred, tgt), "n": int(len(ts)),
            "target_mean": float(tgt.mean()), "target_std": float(tgt.std()),
            "pred_mean": float(pred.mean()), "pred_std": float(pred.std())}


def _sense_features_for(agent, obs_list: List[Dict[str, Any]]) -> torch.Tensor:
    """Sense-time z_world for a stored rollout, episode-reset like _zworld_features."""
    zs = []
    last_ep = None
    for o in obs_list:
        ep = o.get("_calib_episode")
        if ep != last_ep:
            agent.reset()
            last_ep = ep
        zs.append(x737._agent_zworld(agent, o).reshape(-1).detach().cpu())
    return torch.stack(zs) if zs else torch.zeros(0, 1)


def _own_target_r2_sense(agent, target_fn, holdout_obs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Same head, applied to the SENSE-time z_world the adapter reads (not the trainer path)."""
    se = agent.latent_stack.split_encoder
    head = getattr(se, "resource_proximity_head", None)
    if head is None:
        return {"r2": None, "n": 0, "reason": "no_resource_proximity_head"}
    keep = [o for o in holdout_obs if target_fn(o) is not None and np.isfinite(target_fn(o))]
    if len(keep) < 3:
        return {"r2": None, "n": len(keep), "reason": "too_few_labelled_steps"}
    z = _sense_features_for(agent, keep)
    with torch.no_grad():
        pred = head(z.to(se.world_precision_logit.device)).reshape(-1).cpu().numpy().astype(np.float64)
    tgt = np.asarray([float(target_fn(o)) for o in keep], dtype=np.float64)
    return {"r2": _r2(pred, tgt), "n": int(len(keep))}


def _sense_vs_encoder_residual(agent, holdout_obs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """||z_sense - z_encoder_path|| / ||z_encoder_path||: how much of what the adapter reads is
    NOT the encoder's output (untrained top-down conditioning + alpha_world smoothing), which the
    P0a objective cannot touch. Arm-invariant by construction; recorded so a flat result can be
    read against it (red-team finding 8)."""
    se = agent.latent_stack.split_encoder
    if not holdout_obs:
        return {"relative_residual": None, "n": 0}
    z_sense = _sense_features_for(agent, holdout_obs)
    with torch.no_grad():
        w = torch.stack([o["world_state"].detach().float().reshape(-1) for o in holdout_obs])
        z_enc = (se.world_encoder(w) * torch.sigmoid(se.world_precision_logit).unsqueeze(0)).cpu()
    num = float((z_sense - z_enc).norm(dim=1).mean())
    den = float(z_enc.norm(dim=1).mean())
    return {"relative_residual": (num / den) if den > 0 else None, "n": int(z_enc.shape[0]),
            "mean_norm_encoder_path": den, "mean_norm_sense_path": float(z_sense.norm(dim=1).mean())}


def _warm_agent(arm_id: str, seed: int, env_kwargs: Dict[str, Any], sched: Dict[str, int],
                dry_run: bool) -> Tuple[Any, Dict[str, Any]]:
    """Build the lineage agent and run P0a with this arm's objective. UNT: no warmup at all."""
    warm_env = x734._make_env(seed, env_kwargs)
    agent = x1002._make_agent(warm_env)
    before = latent_stack_snapshot(agent)
    extra: Dict[str, Any] = {"p0a": {"ran": False}, "matched_target": None, "own_target_r2": None,
                             "matched_target_certification_training_rollout": None,
                             "own_target_r2_sense_path": None, "sense_vs_encoder_path": None}
    if arm_id == ARM_UNT:
        print("  [%s] no warmup (architectural floor)" % arm_id, flush=True)
        extra["zworld_weight_delta"] = latent_stack_weight_delta(agent, before)
        return agent, extra
    cfg = p0a_config_for(arm_id)
    target_fn: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None
    if arm_id == ARM_B:
        # Calibration on a DEDICATED env + policy (the warmup env's RNG stream is untouched).
        calib = collect_calibration_obs(x734._make_env(seed + 7001, env_kwargs),
                                        RandomPolicy(seed + 7001), sched["calib"], sched["steps"],
                                        label_policy=x1008.LocalViewGreedyPolicy(seed + 7001))
        tgt = MatchedArbitraryTarget(seed)
        extra["matched_target"] = tgt.fit(calib)
        # CERTIFY on the training distribution: the same env seed + RandomPolicy seed the P0a
        # warmup below uses reproduce the identical random walk (both env and policy are
        # self-seeded), so this IS the rollout the head trains on (red-team finding 7).
        train_dist = collect_calibration_obs(x734._make_env(seed, env_kwargs), RandomPolicy(seed),
                                             sched["zworld_p0"], sched["steps"],
                                             label_policy=x1008.LocalViewGreedyPolicy(seed))
        extra["matched_target_certification_training_rollout"] = tgt.certify(train_dist)
        target_fn = tgt
    print("Seed %d Condition %s:%s:warmup" % (seed, RUNG_ID, arm_id), flush=True)
    p0a = run_zworld_p0(agent, x734._make_env(seed, env_kwargs), seed, sched["zworld_p0"],
                        sched["steps"], policy=RandomPolicy(seed),
                        label="%s" % arm_id, dry_run=dry_run, config=cfg, target_fn=target_fn)
    extra["p0a"] = {
        "ran": bool(p0a.get("p0a_ran")), "reason": p0a.get("p0a_reason"),
        "config": p0a.get("p0a_config"), "target": p0a.get("p0a_target"),
        "n_buffered": p0a.get("p0a_n_buffered"), "n_steps": p0a.get("p0a_n_steps"),
        "mean_loss": p0a.get("p0a_mean_loss"), "final_loss": p0a.get("p0a_final_loss"),
        "variance_term": p0a.get("p0a_variance_term"),
        "covariance_term": p0a.get("p0a_covariance_term"),
        "used_proximity_head": p0a.get("p0a_used_proximity_head"),
        "used_reconstruction_head": p0a.get("p0a_used_reconstruction_head"),
        "used_resource_field_head": p0a.get("p0a_used_resource_field_head"),
        "grounding_label_balance": p0a.get("p0a_grounding_label_balance"),
        "holdout": p0a.get("p0a_holdout"),
    }
    extra["zworld_weight_delta"] = latent_stack_weight_delta(agent, before)
    if arm_id in AUX_TRAINED_ARMS:
        hold = collect_calibration_obs(x734._make_env(seed + 7002, env_kwargs),
                                       RandomPolicy(seed + 7002), sched["holdout"], sched["steps"])
        own_fn = target_fn if target_fn is not None else resource_prox_target
        extra["own_target_r2"] = _own_target_r2(agent, own_fn, hold)
        # The DV reads SENSE-time z_world (encoder + untrained top-down + smoothing), the gate
        # above reads the trainer path; report both, and how large the arm-invariant
        # non-encoder component is relative to the encoder's (red-team finding 8).
        extra["own_target_r2_sense_path"] = _own_target_r2_sense(agent, own_fn, hold)
        extra["sense_vs_encoder_path"] = _sense_vs_encoder_residual(agent, hold)
        # For B also report how well the SAME head reads the RESOURCE proximity target -- the
        # amount of organism-relevant scalar the matched arm's head ended up carrying.
        if arm_id == ARM_B:
            extra["prox_r2_through_matched_head"] = _own_target_r2(agent, resource_prox_target, hold)
    return agent, extra


# --------------------------------------------------------------------------------------
# ONE CELL
# --------------------------------------------------------------------------------------
def run_cell(arm_id: str, seed: int, data: Dict[str, Any], feats: Dict[str, Any],
             action_dim: int, sched: Dict[str, int], env_kwargs: Dict[str, Any],
             cfg_base: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    print("Seed %d Condition %s:%s" % (seed, RUNG_ID, arm_id), flush=True)
    y_tr, y_te, yr_te = feats["y_tr"], feats["y_te"], feats["yr_te"]
    with arm_cell(seed, config_slice=_config_slice(cfg_base, arm_id), script_path=Path(__file__),
                  config_slice_declared=True, extra_substrate_paths=_EXTRA_SUBSTRATE_PATHS,
                  include_driver_script_in_hash=False) as cell:
        extra: Dict[str, Any] = {}
        agent = None
        if arm_id == ARM_RAW:
            x = feats["field"]
            transform = x1008._DiagZ(x["tr"])
        elif arm_id == ARM_PCA:
            x = feats["ws"]
            W, pstats = x1008._world_state_pca_stats(x["tr"], PROJECTION_DIM)
            transform = x1008._LinearProjection(x["tr"], W, "pca_projection", {"pca": pstats})
            extra["world_state_pca_stats"] = pstats
        else:
            agent, warm = _warm_agent(arm_id, seed, env_kwargs, sched, dry_run)
            # ENCODER FROZEN from here: every z_world read is under torch.no_grad.
            z_tr, _ = x1002._zworld_features(agent, data["train"])
            z_te, _ = x1002._zworld_features(agent, data["test"])
            zr_te, _ = x1002._zworld_features(agent, data["random"])
            x = {"tr": z_tr, "te": z_te, "r": zr_te}
            transform = x1008._DiagZ(x["tr"])
            extra.update({
                "warmup_skipped": bool(arm_id == ARM_UNT),
                "zworld_weight_delta": warm["zworld_weight_delta"],
                "zworld_participation_ratio": x1002._participation_ratio(x["tr"]),
                "p0a": warm["p0a"],
                "matched_target": warm["matched_target"],
                "matched_target_certification_training_rollout": warm["matched_target_certification_training_rollout"],
                "own_target_r2": warm["own_target_r2"],
                "own_target_r2_sense_path": warm["own_target_r2_sense_path"],
                "sense_vs_encoder_path": warm["sense_vs_encoder_path"],
                "prox_r2_through_matched_head": warm.get("prox_r2_through_matched_head"),
            })
            if int(x["tr"].shape[0]):
                assert int(x["tr"].shape[1]) == PROJECTION_DIM, (
                    "z_world dim %d != PROJECTION_DIM %d" % (int(x["tr"].shape[1]), PROJECTION_DIM))
        net, row = x1008._fit_and_score(arm_id, seed, data, action_dim, sched["passes"],
                                        x["tr"], y_tr, x["te"], y_te, x["r"], yr_te,
                                        transform, unstd_secondary=(arm_id in ANCHOR_ARMS))
        row.update(extra)
        tnet = x1008._TransformedNet(net, transform)
        if arm_id == ARM_RAW:
            pol: Policy = x1002.RawFieldAdapterPolicy(tnet)
        elif arm_id == ARM_PCA:
            pol = x1008.WorldStateAdapterPolicy(tnet)
        else:
            pol = x737.LatentPPOEvalPolicy(tnet, agent)
        row.update(x1008._rollout_row(pol, seed, env_kwargs, sched["eval_eps"], sched["steps"]))
        if agent is not None:
            _ZG.observe(agent)
        cell.stamp(row)
    x1008._print_verdict(row)
    return row


# --------------------------------------------------------------------------------------
# THE VERDICT GRID -- a pure function of the per-seed agreement table; --self-test exercises it
# --------------------------------------------------------------------------------------
def _seed_metrics(agr: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    def d(a: str, b: str) -> Optional[float]:
        return (agr[a] - agr[b]) if (agr.get(a) is not None and agr.get(b) is not None) else None

    def r(arm: str) -> Optional[float]:
        den = d(ARM_PCA, ARM_UNT)
        num = d(arm, ARM_UNT)
        if den is None or num is None or abs(den) < 1e-6:
            return None
        return num / den
    return {"d_C_minus_B": d(ARM_C, ARM_B), "d_C_minus_A": d(ARM_C, ARM_A),
            "d_B_minus_A": d(ARM_B, ARM_A), "d_CFULL_minus_A": d(ARM_CFULL, ARM_A),
            "d_A_minus_UNT": d(ARM_A, ARM_UNT), "d_CFULL_minus_UNT": d(ARM_CFULL, ARM_UNT),
            "d_PCA_minus_UNT": d(ARM_PCA, ARM_UNT),
            "R_A": r(ARM_A), "R_B": r(ARM_B), "R_C": r(ARM_C), "R_CFULL": r(ARM_CFULL)}


def _majority(flags: List[Optional[bool]], majority: int) -> bool:
    return sum(1 for f in flags if f) >= majority


def adjudicate(per_seed: List[Dict[str, Optional[float]]], majority: int,
               gate_green: bool, ready: bool, seeds_sufficient: bool) -> Dict[str, Any]:
    def ge(key: str, thr: float) -> List[Optional[bool]]:
        return [(m[key] is not None and m[key] >= thr) for m in per_seed]

    def le(key: str, thr: float) -> List[Optional[bool]]:
        return [(m[key] is not None and m[key] <= thr) for m in per_seed]

    def _sens(m: Dict[str, Optional[float]]) -> Optional[bool]:
        # OBJECTIVE-CHOICE contrasts only -- all between TRAINED arms. Trained-vs-untrained
        # (A-UNT, CFULL-UNT) is deliberately NOT in the set: a purely destructive move (a
        # trained arm 0.05 below the untrained floor, the lineage's actual direction) would
        # satisfy it while every choice of objective lands identically, and the run would then
        # fall through to "matched control matches anchoring" (re-review finding (a)). Those
        # two contrasts are recorded as witnesses.
        ds = [m.get(k) for k in ("d_CFULL_minus_A", "d_B_minus_A", "d_C_minus_A", "d_C_minus_B")]
        ds = [abs(d) for d in ds if d is not None]
        return (max(ds) >= DELTA_MIN) if ds else None
    objective_reaches_dv = _majority([_sens(m) for m in per_seed], majority)
    c_beats_b = _majority(ge("d_C_minus_B", DELTA_MIN), majority)
    c_beats_a = _majority(ge("d_C_minus_A", DELTA_MIN), majority)
    b_matches_c = _majority([(m["d_C_minus_B"] is not None and abs(m["d_C_minus_B"]) < FLAT_EPS)
                             for m in per_seed], majority)
    b_beats_a = _majority(ge("d_B_minus_A", DELTA_MIN), majority)
    f1 = _majority(ge("R_A", RETAINED_HIGH), majority)
    destruction_all = _majority([(m["R_A"] is not None and m["R_C"] is not None
                                  and m["R_CFULL"] is not None and m["R_A"] <= RETAINED_LOW
                                  and m["R_C"] <= RETAINED_LOW and m["R_CFULL"] <= RETAINED_LOW)
                                 for m in per_seed], majority)
    classes = {"C0_objective_reaches_dv": objective_reaches_dv,
               "C1_anchoring_beats_matched": c_beats_b, "C2_anchoring_beats_generic": c_beats_a,
               "C3_matched_control_flat": b_matches_c, "C4_matched_beats_generic": b_beats_a,
               "C5_generic_preserves_f1": f1, "C6_destruction_real_all": destruction_all}
    if not seeds_sufficient:
        label, outcome = "insufficient_seeds_for_majority", "FAIL"
        arc, inv = "non_contributory", "non_contributory"
    elif not gate_green:
        label, outcome = "substrate_not_ready_requeue", "FAIL"
        arc, inv = "non_contributory", "non_contributory"
    elif not ready:
        label, outcome = "instrument_not_ready", "FAIL"
        arc, inv = "non_contributory", "non_contributory"
    elif not objective_reaches_dv:
        # No P0a objective change -- not even the full SD-070 recipe vs generic-only, nor any
        # trained arm vs the untrained floor -- moved the adapter DV by DELTA_MIN on the seed
        # majority. The instrument cannot see the manipulation class; "matched control matches
        # anchoring" and "the DV cannot see any objective" would be indistinguishable, so
        # neither is recorded (red-team findings 1 and 2).
        label, outcome = "p0a_objective_invisible_to_adapter_dv", "FAIL"
        arc, inv = "non_contributory", "non_contributory"
    elif c_beats_b and c_beats_a:
        label, outcome = "regulatory_anchoring_beats_matched_control", "PASS"
        arc, inv = "supports", "supports"
    elif b_matches_c:
        label, outcome = "matched_control_matches_anchoring", "FAIL"
        arc, inv = "weakens", "weakens"
    elif c_beats_a and not c_beats_b:
        label, outcome = "generic_auxiliary_supervision_effect", "FAIL"
        arc, inv = "non_contributory", "mixed"
    else:
        label, outcome = "inconsistent_across_seeds", "FAIL"
        arc, inv = "mixed", "mixed"
    passed = (label == "regulatory_anchoring_beats_matched_control")
    if seeds_sufficient and gate_green and ready and objective_reaches_dv and f1 and not passed:
        # Generic-only already preserves >= 75% of the width ceiling's room AND anchoring adds no
        # material access beyond it: the class-1 preservation requirement is vacuous at this
        # site. When anchoring DOES beat generic by DELTA_MIN (the PASS branch) the extra access
        # is real and the override does not apply (red-team finding 4).
        inv = "weakens"
    hid = (HYPOTHESIS_HIDS["anchoring"] if label == "regulatory_anchoring_beats_matched_control"
           else HYPOTHESIS_HIDS["matched_null"] if label == "matched_control_matches_anchoring"
           else HYPOTHESIS_HIDS["generic_supervision"] if label == "generic_auxiliary_supervision_effect"
           else None)
    hid_f1: Optional[str] = (HYPOTHESIS_HIDS["f1"]
                             if (seeds_sufficient and gate_green and ready and objective_reaches_dv and f1)
                             else None)
    directions = {"ARC-138": arc, "INV-104": inv}
    if arc == inv:
        overall = arc
    elif "weakens" in (arc, inv):
        overall = "weakens"
    elif "supports" in (arc, inv):
        overall = "mixed"
    else:
        overall = "mixed"
    return {"label": label, "outcome": outcome, "classes": classes,
            "evidence_direction_per_claim": directions, "evidence_direction": overall,
            "resolved_hid": hid, "f1_hid": hid_f1}


def _run_self_test() -> int:
    """Synthetic agreement tables through the grid. Seconds, no env."""
    def table(a, b, c, cf, unt=0.69, pca=0.87):
        return [_seed_metrics({ARM_A: a[i], ARM_B: b[i], ARM_C: c[i], ARM_CFULL: cf[i],
                               ARM_UNT: unt, ARM_PCA: pca}) for i in range(3)]
    cases = [
        ("anchoring", table([0.66] * 3, [0.67] * 3, [0.75] * 3, [0.67] * 3),
         "regulatory_anchoring_beats_matched_control", "PASS", {"ARC-138": "supports", "INV-104": "supports"}),
        ("matched", table([0.66] * 3, [0.72] * 3, [0.725] * 3, [0.67] * 3),
         "matched_control_matches_anchoring", "FAIL", {"ARC-138": "weakens", "INV-104": "weakens"}),
        ("generic-supervision", table([0.66] * 3, [0.70] * 3, [0.73] * 3, [0.67] * 3),
         "generic_auxiliary_supervision_effect", "FAIL", {"ARC-138": "non_contributory", "INV-104": "mixed"}),
        ("inconsistent", table([0.66, 0.70, 0.62], [0.70, 0.66, 0.70], [0.72, 0.66, 0.66], [0.67] * 3),
         "inconsistent_across_seeds", "FAIL", {"ARC-138": "mixed", "INV-104": "mixed"}),
        ("f1-flat", table([0.85] * 3, [0.85] * 3, [0.86] * 3, [0.67] * 3),
         "matched_control_matches_anchoring", "FAIL", {"ARC-138": "weakens", "INV-104": "weakens"}),
        ("f1-anchoring", table([0.83] * 3, [0.78] * 3, [0.90] * 3, [0.67] * 3),
         "regulatory_anchoring_beats_matched_control", "PASS", {"ARC-138": "supports", "INV-104": "supports"}),
        ("f1-generic-only", table([0.83] * 3, [0.84] * 3, [0.845] * 3, [0.67] * 3),
         "matched_control_matches_anchoring", "FAIL", {"ARC-138": "weakens", "INV-104": "weakens"}),
        ("all-flat-insensitive", table([0.69] * 3, [0.695] * 3, [0.70] * 3, [0.685] * 3),
         "p0a_objective_invisible_to_adapter_dv", "FAIL", {"ARC-138": "non_contributory", "INV-104": "non_contributory"}),
        ("destructive-only", table([0.62] * 3, [0.625] * 3, [0.63] * 3, [0.615] * 3),
         "p0a_objective_invisible_to_adapter_dv", "FAIL", {"ARC-138": "non_contributory", "INV-104": "non_contributory"}),
        ("exact-tie", table([0.325] * 3, [0.325] * 3, [0.325] * 3, [0.325] * 3, unt=0.325, pca=0.45),
         "p0a_objective_invisible_to_adapter_dv", "FAIL", {"ARC-138": "non_contributory", "INV-104": "non_contributory"}),
    ]
    ok = True
    for name, ps, label, outcome, dirs in cases:
        v = adjudicate(ps, 2, True, True, True)
        good = (v["label"] == label and v["outcome"] == outcome
                and v["evidence_direction_per_claim"] == dirs)
        print("[self-test] %-20s %s label=%s outcome=%s dirs=%s"
              % (name, "ok" if good else "MISMATCH", v["label"], v["outcome"],
                 v["evidence_direction_per_claim"]))
        ok = ok and good
    v = adjudicate(cases[0][1], 2, False, True, True)
    red_ok = v["label"] == "substrate_not_ready_requeue" and v["outcome"] == "FAIL" \
        and v["evidence_direction_per_claim"] == {"ARC-138": "non_contributory", "INV-104": "non_contributory"}
    print("[self-test] %-20s %s" % ("gate-red", "ok" if red_ok else "MISMATCH"))
    v = adjudicate(cases[0][1], 2, True, False, True)
    ready_ok = v["label"] == "instrument_not_ready" and v["outcome"] == "FAIL"
    print("[self-test] %-20s %s" % ("instrument-red", "ok" if ready_ok else "MISMATCH"))
    v = adjudicate(cases[0][1][:1], 2, True, True, False)
    seeds_ok = v["label"] == "insufficient_seeds_for_majority"
    print("[self-test] %-20s %s" % ("one-seed", "ok" if seeds_ok else "MISMATCH"))
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, _arm_contexts())
    print("[self-test] %-20s ok" % "gate-structure")
    ok = ok and red_ok and ready_ok and seeds_ok
    print("[self-test] %s" % ("ALL OK" if ok else "FAILURES"))
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
# RUN
# --------------------------------------------------------------------------------------
def run_experiment(seeds: List[int], dry_run: bool = False) -> Dict[str, Any]:
    sched = {
        "zworld_p0": DRY_RUN_ZWORLD_P0 if dry_run else ZWORLD_P0_EPISODES,
        "eval_eps": DRY_RUN_EVAL if dry_run else EVAL_EPISODES,
        "steps": DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE,
        "bc_eps": DRY_RUN_BC_EPISODES if dry_run else BC_EPISODES,
        "bc_rand": DRY_RUN_BC_RANDOM_EPISODES if dry_run else BC_RANDOM_EPISODES,
        "passes": DRY_RUN_ADAPTER_PASSES if dry_run else ADAPTER_PASSES,
        "calib": DRY_RUN_CALIBRATION_EPISODES if dry_run else CALIBRATION_EPISODES,
        "holdout": DRY_RUN_HOLDOUT_EPISODES if dry_run else HOLDOUT_EPISODES,
    }
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, _arm_contexts())
    seeds_sufficient = bool(len(seeds) >= SEED_MAJORITY)
    majority = int(SEED_MAJORITY)
    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    cfg_base = _base_slice(sched, dry_run)
    probe_env = x734._make_env(seeds[0], env_kwargs)
    action_dim = int(probe_env.action_dim)

    # ---- the 1002 dataset, re-collected from its deterministic recipe, once per seed --------
    per_seed_data: Dict[int, Dict[str, Any]] = {}
    per_seed_feats: Dict[int, Dict[str, Any]] = {}
    anchor_rows: List[Dict[str, Any]] = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        oracle_eps = x1002._collect_episodes(s, env_kwargs, "oracle", sched["bc_eps"], sched["steps"])
        rand_eps = x1002._collect_episodes(s, env_kwargs, "random", sched["bc_rand"], sched["steps"])
        tr, te = x1002._split_episodes(oracle_eps)
        per_seed_data[s] = {"train": tr, "test": te, "random": rand_eps}
        f_tr, y_tr = x1002._rawfield_features(tr)
        f_te, y_te = x1002._rawfield_features(te)
        fr_te, yr_te = x1002._rawfield_features(rand_eps)
        w_tr, _ = x1008._world_state_features(tr)
        w_te, _ = x1008._world_state_features(te)
        wr_te, _ = x1008._world_state_features(rand_eps)
        per_seed_feats[s] = {"y_tr": y_tr, "y_te": y_te, "yr_te": yr_te,
                             "field": {"tr": f_tr, "te": f_te, "r": fr_te},
                             "ws": {"tr": w_tr, "te": w_te, "r": wr_te}}
        ev = evaluate_seed(x1008.LocalViewGreedyPolicy(s), x734._make_env(s, env_kwargs),
                           sched["eval_eps"], sched["steps"])
        anchor_rows.append({"cell_id": "local_view_greedy|seed%d" % s, "anchor_id": "local_view_greedy",
                            "seed": int(s), "foraging_competence": float(ev["foraging_competence"]),
                            "competence_supra_floor": bool(ev["competence_supra_floor"]),
                            "n_train_episodes": len(tr), "n_test_episodes": len(te),
                            "n_random_episodes": len(rand_eps),
                            "n_train_steps": int(f_tr.shape[0]), "n_heldout_steps": int(f_te.shape[0])})

    # ---- positive control + width ceiling first, every seed -----------------------------------
    all_rows: List[Dict[str, Any]] = []
    for s in seeds:
        all_rows.append(run_cell(ARM_RAW, s, per_seed_data[s], per_seed_feats[s], action_dim,
                                 sched, env_kwargs, cfg_base, dry_run))
        all_rows.append(run_cell(ARM_PCA, s, per_seed_data[s], per_seed_feats[s], action_dim,
                                 sched, env_kwargs, cfg_base, dry_run))

    def _rows(aid: str) -> List[Dict[str, Any]]:
        return [r for r in all_rows if r["arm_id"] == aid]

    raw_worst, _rc = x1002._worst_cell(_rows(ARM_RAW), "oracle_action_agreement", "min")
    pca_worst, _pc = x1002._worst_cell(_rows(ARM_PCA), "oracle_action_agreement", "min")
    instrument_ready = bool(raw_worst >= RAW_FIELD_CONTROL_FLOOR)
    # `or dry_run`: the smoke must execute every arm (V3-EXQ-591g); at dry scale the control
    # cannot reach its floor and the gate would otherwise short-circuit past the warmups.
    for s in seeds:
        for aid in Z_ARMS:
            if instrument_ready or dry_run:
                all_rows.append(run_cell(aid, s, per_seed_data[s], per_seed_feats[s], action_dim,
                                         sched, env_kwargs, cfg_base, dry_run))
            else:
                print("Seed %d Condition %s:%s" % (s, RUNG_ID, aid), flush=True)
                print("  [skip] instrument not certified; arm not run", flush=True)
                print("verdict: FAIL", flush=True)

    # ---- per-seed agreement table -> grid inputs ----------------------------------------------
    def _by_seed(aid: str, key: str) -> Dict[int, Any]:
        return {r["seed"]: r.get(key) for r in _rows(aid)}

    agr = {aid: _by_seed(aid, "oracle_action_agreement") for aid in ARM_IDS}
    per_seed_metrics: List[Dict[str, Any]] = []
    for s in seeds:
        m = _seed_metrics({aid: agr[aid].get(s) for aid in ARM_IDS})
        m["seed"] = int(s)
        m.update({"agreement_%s" % aid: agr[aid].get(s) for aid in ARM_IDS})
        cf = agr[ARM_CFULL].get(s)
        m["reproduces_1008_within"] = ((abs(cf - X1008_ZWORLD_OFF[s]) <= REPRODUCTION_TOL)
                                       if (cf is not None and s in X1008_ZWORLD_OFF) else None)
        m["x1008_zworld_off_agreement"] = X1008_ZWORLD_OFF.get(s)
        per_seed_metrics.append(m)

    # ---- DV headroom (declared): the paired criterion vs the achievable movement at width 32 --
    pca_minus_unt = [m["d_PCA_minus_UNT"] for m in per_seed_metrics if m["d_PCA_minus_UNT"] is not None]
    dv_preconditions: List[Dict[str, Any]] = []
    dv_gate_green = True
    if pca_minus_unt:
        try:
            dv_preconditions = p0_readiness_gate([dv_headroom_check(
                "dv_headroom_pairwise_delta",
                dv_name="paired held-out agreement difference between P0a-objective arms",
                criterion_threshold=float(DELTA_MIN),
                achievable=float(np.mean(pca_minus_unt)), margin=2.0,
                measured_cells=["%s-minus-%s|seed%d" % (ARM_PCA, ARM_UNT, m["seed"])
                                for m in per_seed_metrics if m["d_PCA_minus_UNT"] is not None],
                control=("mean over seeds of ws250_pca minus zworld_untrained agreement: the "
                         "movement an information-preserving linear compression at the encoder's "
                         "width demonstrably makes over the untrained floor on this dataset"),
                description=("DELTA_MIN=0.05 must sit at most half the achievable movement away "
                             "(margin 2.0). 1008 measured PCA-32 0.858-0.877 vs untrained "
                             "0.681-0.704, i.e. ~0.17 of room."))])
        except P0NotReady as e:
            dv_preconditions = list(e.preconditions)
            dv_gate_green = False

    # ---- per-arm readiness gates (never AND'd across arms) -------------------------------------
    maj_worst, _mc = x1002._worst_cell(_rows(ARM_RAW), "train_majority_class_share", "max")
    nte_worst, _nc = x1002._worst_cell(_rows(ARM_RAW), "n_heldout_steps", "min")
    lvg_worst, _lc = x1002._worst_cell(anchor_rows, "foraging_competence", "min")
    arm_gates = []
    for aid in ARM_IDS:
        rows = _rows(aid)
        ctx = _arm_ctx(aid)
        measured: Dict[str, float] = {
            "adapter_capacity_sufficient_on_raw_field": raw_worst,
            "oracle_labels_non_degenerate": maj_worst,
            "heldout_split_sufficient": nte_worst,
            "d3_local_view_greedy_clears_floor": lvg_worst,
        }
        if ctx["has_encoder"]:
            prmin, _p = x1002._worst_cell(rows, "zworld_participation_ratio", "min") if rows else (0.0, None)
            measured["zworld_not_collapsed"] = prmin
            measured["ws250_pca_reaches_bar"] = pca_worst
        if ctx["trained_encoder"]:
            dmin, _d = x1002._worst_cell(
                [{"cell_id": r["cell_id"],
                  "d": float((r.get("zworld_weight_delta") or {}).get("world_encoder_max_abs_delta", 0.0) or 0.0)}
                 for r in rows], "d", "min") if rows else (0.0, None)
            measured["zworld_encoder_trained_in_p0"] = dmin
        if ctx["matched"]:
            def _cert(r: Dict[str, Any], key: str, default: float) -> float:
                v = (r.get("matched_target_certification_training_rollout") or {}).get(key)
                return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else default
            ks_worst, _k = x1002._worst_cell(
                [{"cell_id": r["cell_id"], "ks": _cert(r, "ks_distance_mapped_vs_prox", 1.0)}
                 for r in rows], "ks", "max") if rows else (1.0, None)
            r_worst, _r = x1002._worst_cell(
                [{"cell_id": r["cell_id"], "r": abs(_cert(r, "pearson_r_with_prox", 1.0))}
                 for r in rows], "r", "max") if rows else (1.0, None)
            def _elev(r: Dict[str, Any]) -> float:
                v = ((r.get("matched_target_certification_training_rollout") or {})
                     .get("action_decodability_matched") or {}).get("elevation")
                return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else 1.0
            elev_worst, _e = x1002._worst_cell(
                [{"cell_id": r["cell_id"], "e": _elev(r)} for r in rows], "e", "max") if rows else (1.0, None)
            # PAIRED achieved supervision: |R2(C) - R2(B)| per seed, read across the two arms.
            r2_b = {r["seed"]: ((r.get("own_target_r2") or {}).get("r2")) for r in rows}
            r2_c = {r["seed"]: ((r.get("own_target_r2") or {}).get("r2")) for r in _rows(ARM_C)}
            gaps = [{"cell_id": "%s-vs-%s|seed%d" % (ARM_C, ARM_B, sd),
                     "g": (abs(float(r2_c[sd]) - float(r2_b[sd]))
                           if (r2_c.get(sd) is not None and r2_b.get(sd) is not None) else 1.0)}
                    for sd in r2_b]
            gap_worst, _g = x1002._worst_cell(gaps, "g", "max") if gaps else (1.0, None)
            measured["matched_target_marginal_matched"] = ks_worst
            measured["matched_target_decorrelated"] = r_worst
            measured["matched_target_action_irrelevant"] = elev_worst
            measured["aux_supervision_matched"] = gap_worst
        if ctx["aux_trained"]:
            r2_worst, _q = x1002._worst_cell(
                [{"cell_id": r["cell_id"],
                  "r2": float(((r.get("own_target_r2") or {}).get("r2")) if (r.get("own_target_r2") or {}).get("r2") is not None else -1.0)}
                 for r in rows], "r2", "min") if rows else (-1.0, None)
            measured["aux_target_learned"] = r2_worst
        arm_gates.append(evaluate_arm_gate(aid, ctx, PRECONDITION_SPECS, measured))
    gate = aggregate_arm_gates(arm_gates)
    green = set(gate.get("green_arms") or [])
    verdict_arms_green = all(a in green for a in (ARM_A, ARM_B, ARM_C, ARM_UNT, ARM_PCA))
    gate_green = bool(gate["non_degenerate"]) and dv_gate_green and verdict_arms_green

    verdict = adjudicate(per_seed_metrics, majority, gate_green, instrument_ready, seeds_sufficient)
    cl = verdict["classes"]

    def _arm_summary(aid: str) -> Dict[str, Any]:
        rows = _rows(aid)
        vals = [r["oracle_action_agreement"] for r in rows if r.get("oracle_action_agreement") is not None]
        return {"arm_id": aid, "ran": bool(rows), "n_seeds": len(rows),
                "mean_oracle_action_agreement": (x1002._mean(vals) if vals else None),
                "per_seed_oracle_action_agreement": [r.get("oracle_action_agreement") for r in rows],
                "per_seed_agreement_elevation": [r.get("agreement_elevation") for r in rows],
                "per_seed_oracle_action_agreement_turn_states": [r.get("oracle_action_agreement_turn_states") for r in rows],
                "per_seed_oracle_action_agreement_random_states": [r.get("oracle_action_agreement_random_states") for r in rows],
                "per_seed_trivial_baseline": [r.get("trivial_baseline") for r in rows],
                "per_seed_participation_ratio": [r.get("zworld_participation_ratio") for r in rows],
                "per_seed_own_target_r2": [((r.get("own_target_r2") or {}).get("r2")) for r in rows],
                "per_seed_own_target_r2_sense_path": [((r.get("own_target_r2_sense_path") or {}).get("r2")) for r in rows],
                "per_seed_sense_vs_encoder_relative_residual": [((r.get("sense_vs_encoder_path") or {}).get("relative_residual")) for r in rows],
                "per_seed_prox_r2_through_matched_head": [((r.get("prox_r2_through_matched_head") or {}).get("r2")) for r in rows],
                "per_seed_p0a_final_loss": [((r.get("p0a") or {}).get("final_loss")) for r in rows],
                "per_seed_cloned_foraging_competence": [r.get("cloned_foraging_competence") for r in rows],
                "per_seed_cloned_death_rate": [r.get("cloned_death_rate") for r in rows],
                "per_seed_final_ce_loss": [(r.get("adapter_training") or {}).get("final_ce_loss") for r in rows],
                "action_path_params": (rows[0]["capacity_match"]["action_path_params"] if rows else None),
                "feature_dim": (rows[0].get("feature_dim") if rows else None),
                "p0a_config": (rows[0].get("p0a") or {}).get("config") if rows else None,
                "p0a_target": p0a_target_kind(aid)}

    per_arm = {aid: _arm_summary(aid) for aid in ARM_IDS}
    matched_diag = {int(r["seed"]): {"calibration": r.get("matched_target"),
                                     "training_rollout_certification": r.get("matched_target_certification_training_rollout")}
                    for r in _rows(ARM_B)}

    sensitive = bool(cl["C0_objective_reaches_dv"])
    nd_c1 = bool(gate_green and sensitive)
    nd_c2 = bool(gate_green and sensitive)
    repro = [m.get("reproduces_1008_within") for m in per_seed_metrics]
    criteria = [
        {"name": "C0_objective_reaches_dv", "load_bearing": False, "passed": bool(cl["C0_objective_reaches_dv"]),
         "description": ("SENSITIVITY WITNESS: on >= %d of %d seeds at least one OBJECTIVE-CHOICE contrast "
                         "between trained arms (CFULL-A, B-A, C-A, C-B) moves agreement by >= %.2f in "
                         "absolute value; trained-vs-untrained (A-UNT, CFULL-UNT) is recorded but excluded "
                         "so a purely destructive move cannot stand in for sensitivity. FALSE routes the run "
                         "to p0a_objective_invisible_to_adapter_dv (non_contributory) instead of any claim "
                         "verdict." % (majority, len(seeds), DELTA_MIN))},
        {"name": "C1_anchoring_beats_matched", "load_bearing": True, "passed": bool(cl["C1_anchoring_beats_matched"]),
         "description": "agreement(%s) - agreement(%s) >= %.2f on >= %d of %d seeds" % (ARM_C, ARM_B, DELTA_MIN, majority, len(seeds))},
        {"name": "C2_anchoring_beats_generic", "load_bearing": True, "passed": bool(cl["C2_anchoring_beats_generic"]),
         "description": "agreement(%s) - agreement(%s) >= %.2f on >= %d of %d seeds" % (ARM_C, ARM_A, DELTA_MIN, majority, len(seeds))},
        {"name": "C3_matched_control_flat", "load_bearing": False, "passed": bool(cl["C3_matched_control_flat"]),
         "description": "|agreement(%s) - agreement(%s)| < %.2f on the seed majority (matched control matches anchoring)" % (ARM_C, ARM_B, FLAT_EPS)},
        {"name": "C4_matched_beats_generic", "load_bearing": False, "passed": bool(cl["C4_matched_beats_generic"]),
         "description": "agreement(%s) - agreement(%s) >= %.2f on the seed majority (any auxiliary helps)" % (ARM_B, ARM_A, DELTA_MIN)},
        {"name": "C5_generic_preserves_f1", "load_bearing": False, "passed": bool(cl["C5_generic_preserves_f1"]),
         "description": "R(%s) = (A - UNT)/(PCA - UNT) >= %.2f on the seed majority (INV-104 F1: generic-only compression already preserves class-1 access)" % (ARM_A, RETAINED_HIGH)},
        {"name": "C6_destruction_real_all", "load_bearing": False, "passed": bool(cl["C6_destruction_real_all"]),
         "description": "R(A), R(C), R(CFULL) all <= %.2f on the seed majority, NEGATIVE R (below the untrained floor) included -- EXP-1397 condition (i) holds in every trained arm; a witness, not a conjunct" % RETAINED_LOW},
        {"name": "C7_positive_control_learns_from_raw_field", "load_bearing": False, "passed": bool(instrument_ready),
         "description": "the same adapter reaches >= %.2f from the raw 25-dim field on every seed" % RAW_FIELD_CONTROL_FLOOR},
        {"name": "C8_ws250_pca_reaches_bar", "load_bearing": False, "passed": bool(pca_worst >= AGREEMENT_BAR),
         "description": "the PCA-32 width ceiling reaches >= %.2f on every seed (the retained-fraction denominator)" % AGREEMENT_BAR},
        {"name": "C9_cfull_reproduces_1008", "load_bearing": False,
         "passed": bool(repro and all(x is True for x in repro)),
         "description": ("zworld_p0_sd070_default reproduces V3-EXQ-1008's zworld_off_diag agreement within "
                         "%.3f on every seed (%s) -- the recorded check that skipping P0b/P1 left the frozen "
                         "encoder the lineage's; a witness on the premise, never a conjunct" % (REPRODUCTION_TOL, X1008_ZWORLD_OFF))},
    ]
    combination_rule = (
        "Pure function `adjudicate` of the per-seed agreement table (contract-tested by --self-test). "
        "C1 AND C2 -> regulatory_anchoring_beats_matched_control (PASS; ARC-138 supports, INV-104 "
        "supports). Else C3 -> matched_control_matches_anchoring (FAIL; both weaken). Else C2 without "
        "C1 -> generic_auxiliary_supervision_effect (FAIL; ARC-138 non_contributory, INV-104 mixed). "
        "Else inconsistent_across_seeds (FAIL; both mixed). C5 forces INV-104 -> weakens ONLY outside the "
        "PASS branch. BEFORE any of these: C0 (sensitivity witness) FALSE -> p0a_objective_invisible_to_adapter_dv "
        "(FAIL, both non_contributory, non_degenerate false). Any red readiness gate on a verdict arm, or a red "
        "DV-headroom gate -> substrate_not_ready_requeue (FAIL, both non_contributory); raw-field positive control "
        "below %.2f -> instrument_not_ready. C4/C6/C7/C8/C9 are recorded witnesses, never conjuncts." % RAW_FIELD_CONTROL_FLOOR)

    summary_lines = ["V3-EXQ-1017 %s -> %s" % (verdict["outcome"], verdict["label"]),
                     "per-arm mean agreement: " + ", ".join(
                         "%s=%.3f" % (aid, per_arm[aid]["mean_oracle_action_agreement"])
                         for aid in ARM_IDS if per_arm[aid]["mean_oracle_action_agreement"] is not None),
                     "classes: " + ", ".join("%s=%s" % (k, v) for k, v in cl.items()),
                     "directions: %s" % verdict["evidence_direction_per_claim"],
                     "gate: green_arms=%s dv_headroom_green=%s instrument_ready=%s"
                     % (sorted(green), dv_gate_green, instrument_ready)]

    result: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE, "queue_id": QUEUE_ID,
        "experiment_purpose": EXPERIMENT_PURPOSE, "claim_ids": list(CLAIM_IDS),
        "hypothesis_qid": HYPOTHESIS_QID, "hypothesis_hids": dict(HYPOTHESIS_HIDS),
        # The qid is NEW and its registry pre-registration was OWED at queue time: the
        # authoring session was refused the registry file by an active /failure-autopsy claim
        # (fa-20260909-batch) and did not hand-edit around it. The adjudicating autopsy
        # registers the qid + the four hids above (Step 9b) before writing any resolution.
        "hypothesis_qid_status": "pre_registration_owed_at_queue_time",
        "resolved_hid": verdict["resolved_hid"], "f1_hid": verdict["f1_hid"],
        "outcome": verdict["outcome"], "overall_pass": bool(verdict["outcome"] == "PASS"),
        "evidence_direction": verdict["evidence_direction"],
        "evidence_direction_per_claim": verdict["evidence_direction_per_claim"],
        "sleep_driver_pattern": "none",
        "rung_id": RUNG_ID, "level_id": LEVEL_ID,
        "interpretation": {
            "label": verdict["label"],
            "preconditions": list(gate["adjudication_preconditions"]) + list(dv_preconditions),
            "criteria_non_degenerate": {
                "C1_anchoring_beats_matched": nd_c1, "C2_anchoring_beats_generic": nd_c2,
                "C0_objective_reaches_dv": bool(gate_green),
                "C3_matched_control_flat": bool(gate_green and sensitive),
                "C4_matched_beats_generic": bool(gate_green and sensitive),
                "C5_generic_preserves_f1": bool(gate_green and sensitive and pca_worst >= AGREEMENT_BAR),
                "C6_destruction_real_all": bool(gate_green and sensitive and pca_worst >= AGREEMENT_BAR),
            },
            "combination_rule": combination_rule,
        },
        "criteria": criteria, "combination_rule": combination_rule,
        "per_arm_gate": gate["per_arm_gate"],
        "non_degenerate": bool(gate_green and instrument_ready and seeds_sufficient and sensitive),
        "degeneracy_reason": (None if (gate_green and instrument_ready and seeds_sufficient and sensitive)
                              else verdict["label"]),
        "per_seed_metrics": per_seed_metrics,
        "per_arm": per_arm, "arm_results": all_rows, "anchor_results": anchor_rows,
        "matched_control_diagnostics": matched_diag,
        "dv_headroom": dv_preconditions,
        "pre_registered": {"delta_min": DELTA_MIN, "flat_eps": FLAT_EPS, "retained_high": RETAINED_HIGH,
                           "retained_low": RETAINED_LOW, "ks_max": KS_MAX, "r_max": R_MAX,
                           "aux_r2_floor": AUX_R2_FLOOR, "aux_r2_gap_max": AUX_R2_GAP_MAX,
                           "action_elev_max": ACTION_ELEV_MAX, "seed_majority": majority,
                           "x1008_zworld_off_agreement": X1008_ZWORLD_OFF,
                           "reproduction_tol": REPRODUCTION_TOL},
        "summary_markdown": "\n".join(summary_lines),
        "ethics_preflight": {"involves_negative_valence": False, "involves_suffering_like_state": False,
                             "involves_self_model": False, "involves_inescapability_or_helplessness": False,
                             "involves_offline_replay_over_harm": False, "involves_social_mind_or_language": False,
                             "involves_human_data_or_clinical_context": False, "decision": "allow"},
    }
    return result


if __name__ == "__main__":
    import argparse
    import time
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="Push synthetic agreement tables through the verdict grid and check the "
                             "gate structure, then exit. Seconds, no env.")
    args = parser.parse_args()
    if args.self_test:
        sys.exit(_run_self_test())

    t0 = time.perf_counter()
    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else SEEDS)
    result = run_experiment(seeds=seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["run_id"] = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)
    result["architecture_epoch"] = ARCHITECTURE_EPOCH

    full_config = dict(_base_slice({
        "zworld_p0": (DRY_RUN_ZWORLD_P0 if args.dry_run else ZWORLD_P0_EPISODES),
        "eval_eps": (DRY_RUN_EVAL if args.dry_run else EVAL_EPISODES),
        "steps": (DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE),
        "bc_eps": (DRY_RUN_BC_EPISODES if args.dry_run else BC_EPISODES),
        "bc_rand": (DRY_RUN_BC_RANDOM_EPISODES if args.dry_run else BC_RANDOM_EPISODES),
        "passes": (DRY_RUN_ADAPTER_PASSES if args.dry_run else ADAPTER_PASSES),
        "calib": (DRY_RUN_CALIBRATION_EPISODES if args.dry_run else CALIBRATION_EPISODES),
        "holdout": (DRY_RUN_HOLDOUT_EPISODES if args.dry_run else HOLDOUT_EPISODES),
    }, args.dry_run))
    full_config.update({
        "rung": RUNG, "arms": ARM_IDS,
        "arm_p0a_configs": {aid: _config_slice({}, aid)["arm_p0a_config"] for aid in ARM_IDS},
        "arm_p0a_targets": {aid: p0a_target_kind(aid) for aid in ARM_IDS},
        "delta_min": DELTA_MIN, "flat_eps": FLAT_EPS, "retained_high": RETAINED_HIGH,
        "retained_low": RETAINED_LOW, "ks_max": KS_MAX, "r_max": R_MAX, "aux_r2_floor": AUX_R2_FLOOR,
        "aux_r2_gap_max": AUX_R2_GAP_MAX, "action_elev_max": ACTION_ELEV_MAX,
        "agreement_bar": AGREEMENT_BAR, "agreement_elevation_min": AGREEMENT_ELEVATION_MIN,
        "raw_field_control_floor": RAW_FIELD_CONTROL_FLOOR,
        "participation_ratio_floor": PARTICIPATION_RATIO_FLOOR,
        "oracle_majority_ceiling": x1002.ORACLE_MAJORITY_CEILING,
        "heldout_min_steps": x1002.HELDOUT_MIN_STEPS, "seed_majority": SEED_MAJORITY,
        "adapter_class": "experiments.v3_exq_734...PPOPolicyNet",
    })
    out_path = write_flat_manifest(
        result, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__),
        started_at=t0, z_goal_stream_stats=_ZG.stats(),
    )
    print("outcome: %s (%s)" % (result["outcome"], result["interpretation"]["label"]), flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
