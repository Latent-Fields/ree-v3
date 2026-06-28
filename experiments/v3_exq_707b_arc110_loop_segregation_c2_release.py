#!/opt/local/bin/python3
"""
V3-EXQ-707b -- ARC-110 LOOP-SEGREGATION C2 RELEASE VALIDATION (diagnostic).

SUPERSEDES V3-EXQ-707a (and 707). 707/707a left C2 (limbic-loop load-bearing) SUBSTRATE-
BLOCKED: ARM_DROP_LIMBIC was BYTE-IDENTICAL to A1_LOOPS on every seed, so the per-loop
ablation produced ZERO change in the committed readout and C2 could not be tested (707a's
additive c2_drop_differs_from_a1 diagnostic flagged exactly this byte-identity). The 707
code autopsy (2026-06-28) traced it to a stack of three defects, the deepest being the
MECH-191 PHASIC-EXTERNALISATION GAP: even with the named limbic input modules enabled, the
named cortical bias HEADS (OFC/dACC/lPFC/vigour/liking) emit per-candidate-FLAT output --
the OFC head's INPUT carries real per-candidate range (~0.028) but its OUTPUT range is
exactly 0.0 -- so under per-loop zscore a flat channel is INERT and the limbic loop carried
no per-candidate competition. The only channels carrying per-candidate range were the lumped
residual/mech341/route (the GAP-A project_channel_range path), all in the associative loop.

THE C2 RELEASE (landed 2026-06-28 via /implement-substrate, ree-v3 ree_core): a no-op-default
E3Config flag `use_named_channel_routing` routes each named channel's per-candidate
REPRESENTATION (OFC/lPFC world-summaries [K,D], liking goal-proximity [K], vigour first-action
one-hots [K,A], dACC payoff/effort [K,2], gated_policy summaries [K,D]) through the SAME
parameter-free, range-preserving project_channel_range projection that keeps the lumped
`route` channel phasic, and substitutes that routed term for the flat bias-head scalar in the
segregated-loop accumulation ONLY (the _lcg_terms eligibility traces / authority recompose /
F-score commit path are UNCHANGED; byte-identical OFF). So the limbic loop (ofc/liking/vigour)
now carries REAL per-candidate range and ARM_DROP_LIMBIC can genuinely differ from A1_LOOPS --
making C2 testable. Regression guard: ree-v3/tests/test_arc110_loop_segregation.py
(TestNamedChannelRoutingC2Release + TestRoutedRepsReachSelectorThroughAgent).

CHANGES vs 707a: (1) the loop arms enable `use_named_channel_routing=True` AND the limbic
input modules (use_ofc_analog / use_mech295_liking_bridge / use_tonic_vigor) -- matched as a
CONSTANT on ALL arms incl. A0 so the only swept factor stays loop segregation; (2) a new
per-NAMED-channel non-degeneracy PRECONDITION (`named_channel_routing_live`): on A1_LOOPS at
least one LIMBIC channel's routed per-candidate range must clear a substantive floor on a
strict-majority of divergent seeds, evaluated BEFORE C2 is scored -- so the vacuous DROP==A1
condition self-routes substrate_not_ready_requeue, NEVER a false weakens; (3) C1 (A1 strict-
above max(A0, valid in-layer null ARM_NOISE)+margin) UNCHANGED, C2 (A1 strict-above
ARM_DROP_LIMBIC) re-tested now that DROP can genuinely differ. The c2_drop_differs_from_a1
diagnostic is retained (it should now be TRUE when the gate passes). The 707a in-layer-null
LIVENESS gate (null perturbs committed selection, not null>A0) is also UNCHANGED.

Substrate-readiness validation for the v4_loop_segregation substrate (ARC-110 + the
coupled ARC-109 D1/D2 split + MECH-452 loop-local traces + the S2 in-layer null) built
2026-06-27 via /implement-substrate. EXPERIMENT_PURPOSE=diagnostic; PROMOTES NOTHING.
Routed by the CONFIRMED cluster autopsy
REE_assembly/evidence/planning/failure_autopsy_704b-706b-conversion-ceiling_2026-06-27.{md,json}:
the conversion-ceiling lineage converged from four angles (700b/700c learned-gating+settling,
704b finer-channel, 706b double-gated curiosity) on the conclusion that the V3 SINGLE E3
foraging arena structurally denies (a) committed-action-class conversion from non-motor
channels and (b) a VALID same-layer null. The gate cleared by V3-EXQ-704b FAIL-to-convert
(the pre-registered positive-evidence-FOR-ARC-110 outcome: representational compression is
NOT the binding constraint).

THE QUESTION (ARC-110)
----------------------
Is the F-dominance conversion ceiling (MECH-439) partly an ARTEFACT of the single-arena
collapse? The hypothesis: with N>=3 parallel segregated cortico-BG-thalamic loops (motor /
associative / limbic), each running within-loop competition FIRST and cross-loop arbitration
AFTER, F dominates only the MOTOR loop and cannot drown the limbic "is this worth committing
to" value -- so committed-action diversity converts where the single F-dominated arena
plateaued. AND the in-layer (eligibility/settling-field) null can now VERIFY-LIFT, the valid
same-layer null the single arena could not construct (the 704b/706b binding constraint).

THE 4 ARMS (all carry the SAME landed arithmetic envelope + finer-channel gating + learned
settling as a MATCHED CONSTANT; the ONLY swept factor is loop segregation):
  A0_SINGLE_ARENA  : the single-arena baseline -- use_loop_segregation=False; finer-channel
                     gating + learned settling ON in ONE shared E3 arena (= the V3-EXQ-704b
                     A2 lineage that plateaued). The conversion baseline.
  A1_LOOPS         : use_loop_segregation=True (motor=F / associative=dACC+lPFC /
                     limbic=OFC+liking+vigour) + use_d1_d2_population_split + use_loop_local_
                     eligibility_traces. Within-loop competition first, Haber ascending-spiral
                     arbitration after, per-loop zscore normalisation (strips F's magnitude
                     advantage). The conversion arm.
  ARM_NOISE        : A1_LOOPS + loop_segregation_noise_on=True -- the S2 IN-LAYER same-layer
                     null: each non-motor loop accumulator is replaced by a magnitude-matched
                     random-structure perturbation at the SAME layer the loops settle on (NOT
                     policy temperature, the decoupled 700-lineage null). This is the VALID
                     same-layer null the single arena could not construct; a lift is
                     attributable to learned loop STRUCTURE only if it beats this null.
  ARM_DROP_LIMBIC  : A1_LOOPS with the LIMBIC loop ablated (its channels remapped to the
                     associative loop, so the limbic loop is empty) -- the ARC-106 per-loop
                     load-bearing falsifier: if dropping the limbic loop does NOT remove the
                     A1 lift, the limbic loop is DECORATIVE.
6 seeds. PRIMARY DV = committed-action-class entropy (nats), measured over P2.
claim_ids = [ARC-110]. experiment_purpose = diagnostic (substrate-readiness; PROMOTES NOTHING).

PRE-REGISTERED OUTCOME MAP (decisive either way)
------------------------------------------------
  PASS / supports ARC-110 (single-arena collapse WAS a binding constraint):
    A1_LOOPS lifts committed-action-class entropy strict-above A0_SINGLE_ARENA AND
    strict-above the VALID in-layer null ARM_NOISE on a strict-majority of DIVERGENT seeds,
    with live cross-loop variance (a non-motor loop actually FLIPPED the within-eligible
    winner). If ARM_DROP_LIMBIC does NOT reproduce the lift, the limbic loop is load-bearing.

  WEAKENED / route-elsewhere (collapse was NOT the binding constraint):
    a VALID same-layer null can now be constructed (ARM_NOISE is a LIVE perturbation of the
    committed selection) and loops carry live cross-loop variance, BUT A1_LOOPS does NOT
    convert committed diversity strict-above A0/the null. The conversion ceiling is INTRINSIC,
    not an artefact of single-arena collapse -> weakens ARC-110.

NON-VACUITY READINESS GATES (self-route substrate_not_ready_requeue, NEVER a false weakens):
  (1) candidate pool DIVERGENT: GAP-A guard cand_world_pairwise_dist > floor (per seed).
  (2) LOOPS carry live CROSS-LOOP VARIANCE on A1: loop_committed_neq_motor_winner OR
      loop_cross_loop_winner_disagreement on a majority of divergent seeds, AND per-loop
      loop_assoc_pref_range / loop_limbic_pref_range > 0. A "segregated" loop pinned to the
      motor winner is a vacuous split -> requeue (the ARC-110 what_would_answer guard).
  (3) the VALID in-layer null (ARM_NOISE) is a LIVE same-layer perturbation -- on a strict-
      majority of divergent seeds it actively perturbs the committed selection
      (loop_noise_active_ticks > 0 AND loop_frac_committed_neq_motor or loop_frac_disagree >
      the cross-loop-variance floor). This is the SAME-LAYER null the single arena could not
      construct; if even the in-layer null is INERT (does not reach the committed-class
      readout) the null-validity problem persists -> requeue, not weakens. NOTE (707->707a):
      the RETIRED proxy required the null to LIFT entropy strict-above A0 -- the 700-lineage
      TEMPERATURE-null liveness test, INVALID for a structured-accumulator null (random
      magnitude-matched loop content lands at baseline, not above it; structured beats random,
      random ~ A0). null-LIVENESS, not null-LIFT, is the correct non-vacuity guard, and it
      keeps the WEAKENS branch reachable. The path-artifact hypothesis (A1's lift is a zscore-
      path artefact) is ruled out by C1's "A1 strict-above the null", NOT by null-lift.
  (4) signed delta_t NON-FLAT + finer w_chan_finer entries actually MOVE (learning engaged).

ARC-106 load-bearing-vs-decorative: ARM_DROP_LIMBIC is the per-loop ablation. If dropping the
limbic loop leaves the A1 lift unchanged, the limbic loop is DECORATIVE (recorded; C2).
Phased training is NOT required in the encoder sense (reuses trained valuation heads; learned
objects use the ARC-108 LOCAL three-factor update), but the P0/P1/P2 phasing is kept for a
fair comparison with the 704b baseline. MECH-094: learning writes are waking-only (inherited);
the in-layer null is selection-only (no memory write).

See REE_assembly/docs/architecture/sd_v4_loop_segregation.md (ARC-110 design-of-record + IMPLEMENTED block),
    REE_assembly/evidence/planning/failure_autopsy_704b-706b-conversion-ceiling_2026-06-27.{md,json} (routes this build),
    REE_assembly/evidence/planning/substrate_queue.json (sd_id v4_loop_segregation),
    ree-v3/ree_core/predictors/e3_selector.py (_segregated_loop_arbitrate / _d1_d2_split / _loop_inlayer_null),
    experiments/v3_exq_704b_mech451_finer_channel_granularity_falsifier.py (matched-substrate sibling).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import compute_arm_fingerprint, reset_all_rng
from experiments._lib.arm_reuse import try_reuse_cell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_707b_arc110_loop_segregation_c2_release"
QUEUE_ID = "V3-EXQ-707b"
SUPERSEDES = "V3-EXQ-707a"   # 707b lands the C2 release (per-named-channel range-preserving routing) so the limbic loop is load-bearing
BACKLOG_ID = None   # no proposal; routed by failure_autopsy_704b-706b-conversion-ceiling_2026-06-27
CLAIM_IDS: List[str] = ["ARC-110"]
EXPERIMENT_PURPOSE = "diagnostic"   # substrate-readiness validation; PROMOTES NOTHING

# softplus-unity init for w_chan_finer (softplus(_FCG_W_INIT) == 1.0).
_FCG_W_INIT = math.log(math.e - 1.0)

# CRF-gate calibration levers (matured CRF stack; ported verbatim from 700c,
# matched on all arms -- the differentiated conversion source).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# ----- Acceptance thresholds (pre-registered) -----
# C1 conversion: strict-above margin on committed-class entropy (nats).
CONVERSION_MARGIN = 0.05
# C2 grows-over-training: second-half P2 entropy strict-above first-half margin.
GROWTH_MARGIN = 0.02

# ----- Per-seed-divergent gating (701a-style) -----
MIN_DIVERGENT_SEEDS = 3          # of 6: fewer divergent seeds => substrate_not_ready_requeue
DIVERGENT_PASS_FRACTION = 0.5    # strict-majority-ish gate within the divergent seeds
MIN_SEEDS_FOR_PASS = 2           # absolute floor of divergent seeds clearing a criterion

# ARC-110 non-degeneracy: loops must carry LIVE cross-loop variance on the loop arms.
# A non-motor loop must FLIP the within-eligible winner (or loops disagree) on at least this
# fraction of P2 select ticks, AND a non-motor loop must carry per-loop preference spread > 0.
LOOP_CROSS_VARIANCE_FRAC_FLOOR = 0.05
LOOP_PREF_RANGE_FLOOR = 1e-6
# ARC-110 C2 RELEASE non-degeneracy gate (707b): on A1_LOOPS, at least one LIMBIC channel
# (ofc/liking/vigour) must reach the segregated-loop arbitration carrying a per-candidate
# routed range above this SUBSTANTIVE floor (peak over P2 ticks), on a strict-majority of
# divergent seeds, BEFORE C2 (limbic load-bearing) is scored. This is the precondition the
# 707 vacuous DROP==A1 lacked: if the named limbic channels are still per-candidate-FLAT
# (the MECH-191 phasic gap), the limbic loop carries no competition and the per-loop ablation
# is a no-op -> self-route substrate_not_ready_requeue, NEVER a false weakens. Set well above
# the 1e-6 inert floor (so a collapsed/flat channel fails) but below the routed range a
# genuine world-summary / proximity / action-class representation yields (the OFC input range
# was ~0.028; project_channel_range preserves that order). 1e-3 ~= 36x the inert floor.
LIMBIC_ROUTED_RANGE_FLOOR = 1e-3
# Named channels assigned to the limbic loop (must match e3_selector _LOOP_DEFAULT_CHANNEL_MAP).
LIMBIC_NAMED_CHANNELS = ("ofc", "liking", "vigour")

# C1(a) readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# Non-vacuity (b): GAP-A consumed-summary divergence (649 statistic + 643a ceiling).
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# Non-vacuity (c): delta_t carries cross-tick variance (outcome variance to learn from).
DELTA_T_STD_FLOOR = 1e-4
# Non-vacuity (1): the finer w_chan_finer entries MOVED + are DISSOCIABLE (cross-channel
# range above floor -- finer entries that move identically are the blend re-labelled).
W_CHAN_FINER_RANGE_FLOOR = 1e-4   # softplus-unity init => range 0; >floor == reorganised + dissociable
# 704b near-vacuity guard (autopsy Step 7): the A2 LEARNED finer-channel range must be
# SUBSTANTIVE, not merely floor-clearing. V3-EXQ-704's A2 dissociation was 0.00148 (~15x the
# bare 1e-4 floor) -- the autopsy flagged that as near the "compressed blend re-labelled"
# vacuity boundary, where even a correctly magnitude-matched strict-above-noise bar is a
# knife-edge tiny-structure-vs-tiny-matched-noise non-test. A dissociation below this
# SUBSTANTIVE floor on a strict-majority of seeds self-routes substrate_not_ready_requeue
# (NEVER a false weakens). 5e-3 ~= 3.4x the prior near-vacuous 704 realised range (0.00148);
# erring high only yields an honest "not ready", never a misleading weakens.
W_CHAN_FINER_SUBSTANTIVE_RANGE_FLOOR = 5e-3
# A1 load-bearing-ablation non-vacuity: the single global w_chan also genuinely learned.
W_CHAN_RANGE_FLOOR = 1e-4

# CRF maturity readiness (matched constant; the differentiated source must be present).
CRF_MIN_MINTED = 2
CRF_N_ACTIVE_FLOOR = 1
CRF_FRAC_ACTIVE_FLOOR = 0.30

SEEDS = [42, 43, 44, 45, 46, 47]
P0_WARMUP_EPISODES = 100
P1_BIAS_TRAIN_EPISODES = 50
P2_MEASUREMENT_EPISODES = 100   # binned first/second half for C2 grows-over-training
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 4   # >= 2 so the P2 first/second-half split is exercised in the smoke
DRY_RUN_STEPS = 30

# ARC-110 S2 same-layer null (ARM_NOISE): the magnitude-matched random-structure perturbation
# is generated IN-SELECTOR per-tick at the non-motor loop accumulators (config.e3.
# loop_segregation_noise_on -> _loop_inlayer_null), NOT injected here. No manual w_chan_finer
# seeding is needed -- the null is a property of the loop substrate, not the experiment driver.

# Mint run_id to cite for arm-reuse; None => run all arms fresh (the safe default for a
# fresh lineage). A future sibling sets this to a prior 704 run_id after one lands.
REUSE_BASELINE_FROM = None

# Stable arm metric keys the acceptance logic reads from a reusable row (a reused cell
# MUST have recorded all of them -- the section-9.2 correctness trap).
REUSE_NEEDED_KEYS = [
    "committed_class_entropy_nats",
    "committed_class_entropy_p2_first_half_nats",
    "committed_class_entropy_p2_second_half_nats",
    "gapa_divergence",
    "frac_pre_ge2",
    "consumed_summary_pairwise_dist_mean",
    "consumed_summary_pairwise_dist_max",
    "crf_differentiated",
    "crf_frac_active_ge_floor",
    "crf_n_minted_total",
    "fcg_moved",
    "fcg_delta_nonflat",
    "fcg_w_chan_finer_range_max",
    "fcg_w_chan_finer_std_max",
    "fcg_delta_t_std",
    "lcg_moved",
    "lcg_delta_nonflat",
    "lcg_w_chan_range_max",
    "lcg_delta_t_std",
    "n_unique_committed_classes",
    "error_note",
]

# --- Matched-stack lever constants (identical on ALL arms; the landed envelope) ---
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
USE_MODULATORY_CHANNEL_ROUTING = True
MODULATORY_CHANNEL_ROUTE_SOURCE = "cand_world_summary"
MODULATORY_CHANNEL_ROUTE_WEIGHT = 1.0
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6
USE_MODULATORY_SHORTLIST_THEN_MODULATE = True
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_SHORTLIST_K = 3
USE_F_ELIGIBILITY_DEMOTION = True
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30
F_ELIGIBILITY_DN_SIGMA = 0.0
USE_F_ELIGIBILITY_ADAPTIVE_FLOOR = True
F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR = 1.0
USE_GO_NOGO_CONSTITUTION = True
USE_DACC = True
GNG_PERSEVERATION_FLOOR = 0.5
GNG_SAFETY_FLOOR = 0.5
GNG_PROTECT_MIN_ELIGIBLE = 1
MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

USE_CANDIDATE_RULE_FIELD = True

# ----- ARC-108 JOB-1 step-1 learned-gating knobs (substrate defaults; matched when armed) -----
# Reused VERBATIM by the MECH-451 finer path (same eta/elig/baseline/asym), so
# A1_GLOBAL_WCHAN vs A2_FINER differ ONLY in channel granularity (single-variable design).
LCG_ETA = 0.01
LCG_ELIG_DECAY = 0.9
LCG_VALUE_BASELINE_BETA = 0.05
LCG_ASYM_POTENTIATION = 1.0
LCG_ASYM_DEPRESSION = 0.5

# ----- MECH-450 settling (ON on all arms; the within-loop settling each loop runs) -----
LEARNED_SETTLING_ROUNDS = 3
LEARNED_SETTLING_TEMPERATURE = 1.0
LEARNED_SETTLING_ETA = 0.01
LEARNED_SETTLING_ELIG_DECAY = 0.9

# ----- ARC-110 loop-segregation knobs (matched on all loop arms) -----
LOOP_SEGREGATION_NORMALIZE = "zscore"   # per-loop preference normalisation (strips F's magnitude)
LOOP_SEGREGATION_NOISE_ALPHA = 1.0      # S2 in-layer null: range == alpha x the real loop range

# SD-056 online e2 training (mirror 700c).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
SD056_MULTISTEP_CONTRASTIVE = True
SD056_CONTRASTIVE_HORIZON = 5
SD056_OUTPUT_NORM_CLAMP = True
SD056_OUTPUT_NORM_CLAMP_RATIO = 2.0

# P1 bias-head REINFORCE training (mirror 700c).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9


# IDENTICAL env to 700c (the GAP-A reef-bipartite foraging bank).
ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)


# The 4 arms. ALL carry finer-channel gating + learned settling as a MATCHED CONSTANT;
# the ONLY swept factor is loop segregation (loop_seg) + its noise / drop-limbic variants.
# A0_SINGLE_ARENA is the single-arena baseline (= the 704b A2 lineage). ARM_NOISE adds the
# S2 in-layer null; ARM_DROP_LIMBIC ablates the limbic loop (per-loop load-bearing falsifier).
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "A0_SINGLE_ARENA",
        "label": "single_arena_baseline_finer_gating_settling_no_loop_segregation",
        "finer_on": True,
        "loop_seg": False,
        "loop_noise": False,
        "drop_limbic": False,
    },
    {
        "arm_id": "A1_LOOPS",
        "label": "arc110_segregated_loops_motor_assoc_limbic_d1d2_loop_local_traces",
        "finer_on": True,
        "loop_seg": True,
        "loop_noise": False,
        "drop_limbic": False,
    },
    {
        "arm_id": "ARM_NOISE",
        "label": "arc110_s2_in_layer_same_layer_null_magnitude_matched_random_structure",
        "finer_on": True,
        "loop_seg": True,
        "loop_noise": True,
        "drop_limbic": False,
    },
    {
        "arm_id": "ARM_DROP_LIMBIC",
        "label": "arc110_limbic_loop_ablated_per_loop_load_bearing_falsifier",
        "finer_on": True,
        "loop_seg": True,
        "loop_noise": False,
        "drop_limbic": True,
    },
]

# A0_SINGLE_ARENA is a stable single-arena baseline (self-mint eligible). The loop arms ride
# the just-built loop substrate (in flux for this lineage) -- not minted as reusable baselines.
REUSABLE_ARM_IDS_LOCAL = ("A0_SINGLE_ARENA",)

# ARM_DROP_LIMBIC remaps the limbic channels into the associative loop so the limbic loop is
# empty (the ARC-106 per-loop ablation). Motor stays F; associative absorbs the rest.
_DROP_LIMBIC_CHANNEL_MAP = {
    "dacc": "associative",
    "lpfc": "associative",
    "ofc": "associative",
    "liking": "associative",
    "vigour": "associative",
}


def _arm_config_slice(
    arm: Dict[str, Any],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Declared reuse fingerprint slice: ONLY what an arm's computation reads -- the
    swept arm flags + the matched arithmetic envelope every arm runs + the env + the
    schedule. NEVER acceptance thresholds. Same slice for the consumer fingerprint and
    a future mint, so they match by construction (settling ON, finer gating ON on all arms;
    loop segregation is the swept factor)."""
    return {
        "arm_id": arm["arm_id"],
        "finer_on": bool(arm["finer_on"]),
        "loop_seg": bool(arm.get("loop_seg", False)),
        "loop_noise": bool(arm.get("loop_noise", False)),
        "drop_limbic": bool(arm.get("drop_limbic", False)),
        "use_learned_settling_step": True,
        "use_d1_d2_population_split": bool(arm.get("loop_seg", False)),
        "use_loop_local_eligibility_traces": bool(arm.get("loop_seg", False)),
        # ARC-110 C2 RELEASE (707b): routing (swept with loop_seg) + limbic input modules
        # (matched constant on all arms) -- both change the arm's computation, so the
        # fingerprint must declare them (else a future consumer mis-matches the mint).
        "use_named_channel_routing": bool(arm.get("loop_seg", False)),
        "use_ofc_analog": True,
        "use_mech295_liking_bridge": True,
        "use_tonic_vigor": True,
        "learned_channel_gating_eta": LCG_ETA,
        "lcg_elig_decay": LCG_ELIG_DECAY,
        "lcg_value_baseline_beta": LCG_VALUE_BASELINE_BETA,
        "lcg_asym_potentiation": LCG_ASYM_POTENTIATION,
        "lcg_asym_depression": LCG_ASYM_DEPRESSION,
        "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
        "use_f_eligibility_adaptive_floor": bool(USE_F_ELIGIBILITY_ADAPTIVE_FLOOR),
        "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
        "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
        "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
        "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
        "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
        "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
        "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
        "modulatory_channel_route_source": str(MODULATORY_CHANNEL_ROUTE_SOURCE),
        "use_candidate_rule_field": bool(USE_CANDIDATE_RULE_FIELD),
        "use_dacc": bool(USE_DACC),
        "env_kwargs": dict(ENV_KWARGS),
        "sd056_weight": float(SD056_WEIGHT),
        "lr_lpfc_bias": float(LR_LPFC_BIAS),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
    }


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Matched-stack agent. The landed arithmetic envelope (demotion + adaptive floor +
    Go/No-Go + authority + routing + top_k shortlist) + the diversity stack (MECH-341,
    SD-056, CRF, trained lateral_pfc bias head, use_dacc) + finer-channel gating + learned
    settling are MATCHED CONSTANTS on ALL arms. The ONLY swept factor is loop segregation:
    A0_SINGLE_ARENA runs the legacy single-arena within-eligible argmin (use_loop_segregation
    =False); A1_LOOPS / ARM_NOISE / ARM_DROP_LIMBIC run the ARC-110 segregated loops
    (+ ARC-109 D1/D2 + MECH-452 loop-local traces). ARM_NOISE adds loop_segregation_noise_on
    (the S2 in-layer same-layer null); ARM_DROP_LIMBIC remaps the limbic channels into the
    associative loop (the per-loop ablation)."""
    finer_on = bool(arm["finer_on"])
    loop_seg = bool(arm.get("loop_seg", False))
    loop_noise = bool(arm.get("loop_noise", False))
    drop_limbic = bool(arm.get("drop_limbic", False))
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        # --- Matched stack (identical on all arms) ---
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # GAP-A (V3-EXQ-649): shared per-candidate signal from e2.world_forward.
        candidate_summary_source="e2_world_forward",
        # modulatory-bias-selection-authority (643a) + channel routing.
        use_modulatory_selection_authority=USE_MODULATORY_SELECTION_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_channel_routing=USE_MODULATORY_CHANNEL_ROUTING,
        modulatory_channel_route_source=MODULATORY_CHANNEL_ROUTE_SOURCE,
        modulatory_channel_route_weight=MODULATORY_CHANNEL_ROUTE_WEIGHT,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        # 569i TOP-K shortlist scaffold (the eligible set the gating acts inside).
        use_modulatory_shortlist_then_modulate=USE_MODULATORY_SHORTLIST_THEN_MODULATE,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        # MECH-448 demotion envelope (CONSTANT ON) + channel-adaptive floor (689e).
        use_f_eligibility_demotion=USE_F_ELIGIBILITY_DEMOTION,
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        use_f_eligibility_adaptive_floor=USE_F_ELIGIBILITY_ADAPTIVE_FLOOR,
        f_eligibility_adaptive_mean_factor=F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
        # MECH-449 Go/No-Go constitution (CONSTANT ON). use_dacc feeds the MECH-260
        # perseveration No-Go axis ecologically.
        use_dacc=USE_DACC,
        use_go_nogo_constitution=USE_GO_NOGO_CONSTITUTION,
        gng_perseveration_floor=GNG_PERSEVERATION_FLOOR,
        gng_safety_floor=GNG_SAFETY_FLOOR,
        gng_protect_min_eligible=GNG_PROTECT_MIN_ELIGIBLE,
        # MECH-341 (stratified across-class; within-class temperature default).
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=MECH341_ENTROPY_BIAS_SCALE,
        e3_diversity_stratified_within_class_temperature=None,
        # MECH-313 noise floor -- OFF on EVERY arm (the ARM_NOISE null is a same-layer
        # finer-gating null, NOT a policy-temperature lift).
        use_noise_floor=False,
        # V_s minimal stack.
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        # ARC-062 GatedPolicy (matched; symmetry-broken bias) -- a finer channel source.
        use_gated_policy=True,
        # SD-033a LateralPFCAnalog with the bias head UN-ZEROED + trainable (GAP-D) -- a
        # finer channel source.
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        # SD-056 (e2 action-conditional divergence; trained online in P0).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
        # --- CRF maturity + maintenance levers (MATCHED; the differentiated source) ---
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_context_from_e2_world_forward=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=CRF_MAINTENANCE_FLOOR,
        crf_maintenance_decay=CRF_MAINTENANCE_DECAY,
        crf_mature_context_match_threshold=CRF_MATURE_CONTEXT_MATCH_THRESHOLD,
        crf_tolerance_conflict_cap=CRF_TOLERANCE_CONFLICT_CAP,
        crf_maintenance_couple_to_theta=CRF_MAINTENANCE_COUPLE_TO_THETA,
        use_candidate_rule_field=USE_CANDIDATE_RULE_FIELD,
        # --- MECH-451: FINER separately-learnable channels (ON on ALL arms; the loops
        # partition these finer channels into motor/associative/limbic). ---
        use_finer_channel_gating=finer_on,
        use_learned_channel_gating=False,
        # Shared three-factor knobs (used by the finer w_chan_finer path on all arms).
        learned_channel_gating_eta=LCG_ETA,
        learned_channel_gating_elig_decay=LCG_ELIG_DECAY,
        learned_channel_value_baseline_beta=LCG_VALUE_BASELINE_BETA,
        learned_channel_asym_potentiation=LCG_ASYM_POTENTIATION,
        learned_channel_asym_depression=LCG_ASYM_DEPRESSION,
        # signed RPE (no C3 unsigned ablation in this experiment).
        learned_channel_rpe_mode="signed",
        # --- MECH-450 recurrent settling: ON on ALL arms (the within-loop settling each
        # segregated loop runs; on A0 it is the legacy single-arena settling). ---
        use_learned_settling_step=True,
        learned_settling_rounds=LEARNED_SETTLING_ROUNDS,
        learned_settling_temperature=LEARNED_SETTLING_TEMPERATURE,
        learned_settling_eta=LEARNED_SETTLING_ETA,
        learned_settling_elig_decay=LEARNED_SETTLING_ELIG_DECAY,
        # --- ARC-110 parallel segregated loops (the SWEPT factor) + coupled ARC-109 /
        # MECH-452 / S2 in-layer null. Default-off on A0_SINGLE_ARENA -> legacy single
        # arena. ARM_DROP_LIMBIC remaps the limbic channels into the associative loop. ---
        use_loop_segregation=loop_seg,
        loop_segregation_channel_map=(dict(_DROP_LIMBIC_CHANNEL_MAP) if drop_limbic else {}),
        loop_segregation_normalize=LOOP_SEGREGATION_NORMALIZE,
        loop_segregation_noise_on=loop_noise,
        loop_segregation_noise_alpha=LOOP_SEGREGATION_NOISE_ALPHA,
        use_d1_d2_population_split=loop_seg,
        use_loop_local_eligibility_traces=loop_seg,
        # --- ARC-110 C2 RELEASE (707b): per-named-channel range-preserving routing into the
        # segregated loops, so the named limbic channels carry per-candidate range (the flat
        # bias-head scalars are inert under per-loop zscore -- the MECH-191 phasic gap). ON on
        # the loop arms only (A0 has loop_seg=False so the override is built but never consumed
        # by the legacy single-arena path -> A0 stays the true single-arena baseline). ---
        use_named_channel_routing=loop_seg,
        # --- Limbic-loop INPUT modules: MATCHED CONSTANT on ALL arms (incl. A0) so the only
        # swept factor stays loop segregation. Without these the limbic channels carry NO live
        # representation to route. OFC-devaluation value / MECH-295 drive->liking->approach /
        # MECH-320 tonic vigour -- the three limbic-loop value sources (ofc/liking/vigour). ---
        use_ofc_analog=True,
        use_mech295_liking_bridge=True,
        use_tonic_vigor=True,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (verbatim from 700c)
# ---------------------------------------------------------------------------


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen_classes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen_classes:
            seen_classes[cls] = tup
        if len(seen_classes) >= k:
            break
    if len(seen_classes) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen_classes.values())
    picked_ids = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked_ids:
            continue
        samples.append(tup)
        picked_ids.add(id(tup))
    return samples


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K,
        actions=actions_K,
        z_world_1_targets=z1_K,
        simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    weighted = SD056_WEIGHT * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# Per-tick measurement helpers
# ---------------------------------------------------------------------------


def _traj_first_action_class(traj) -> int:
    return int(traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item())


def _consumed_summaries(agent: REEAgent, candidates) -> Optional[torch.Tensor]:
    """Per-candidate cand_world_summaries the bias channels consume (GAP-A
    e2.world_forward source; matched on all arms)."""
    summ = agent._candidate_world_summaries(candidates)
    if summ is not None:
        return summ.detach()
    rows: List[torch.Tensor] = []
    for c in candidates:
        if c.world_states is not None:
            rows.append(c.get_world_state_sequence()[0, 0, :].detach())
        elif agent._current_latent is not None:
            rows.append(agent._current_latent.z_world[0].detach())
        else:
            return None
    return torch.stack(rows, dim=0) if rows else None


def _mean_pairwise_l2(summ: torch.Tensor) -> float:
    summ = summ.detach()
    k = summ.shape[0]
    if k < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += float(torch.linalg.vector_norm(summ[i] - summ[j]))
            n += 1
    return total / max(n, 1)


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


def _entropy_from_int_counts(counts: Dict[int, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


# ---------------------------------------------------------------------------
# P1 bias-head REINFORCE training (verbatim from 700c)
# ---------------------------------------------------------------------------


def _lpfc_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
) -> torch.Tensor:
    if agent.lateral_pfc is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    n = len(outcome_buf)
    idxs = np.random.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False)
    terms: List[torch.Tensor] = []
    for i in idxs:
        cand_features, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        bias = agent.lateral_pfc.compute_bias(cand_features.to(device))
        log_p = F.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)
    env = _make_env(seed)
    agent = _make_agent(env, arm)

    # ARC-110 S2 in-layer null is IN-SELECTOR (config.e3.loop_segregation_noise_on), so
    # ARM_NOISE needs no manual buffer injection -- the magnitude-matched random structure
    # is generated per-tick at the non-motor loop accumulators inside _segregated_loop_arbitrate.

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = torch.optim.Adam(
        list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS
    )
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes + p2_episodes
    p1_start = p0_episodes
    p2_start = p0_episodes + p1_episodes
    p2_mid = p2_start + (p2_episodes // 2)  # P2 first/second-half split (by episode)
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p2_ticks = 0
    n_p0_contrastive_steps = 0
    n_p1_bias_updates = 0

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    # PRIMARY DV: committed first-action class counts over P2, split first/second half.
    committed_class_counts: Dict[int, int] = {}
    committed_class_counts_p2a: Dict[int, int] = {}  # P2 first half
    committed_class_counts_p2b: Dict[int, int] = {}  # P2 second half
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0

    # CRF maturity readiness (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_minted_total_last = 0

    # ----- ARC-108 (A1) / MECH-451 (A2) learning diagnostics (accumulated all phases) -----
    lcg_delta_ts: List[float] = []
    lcg_w_chan_range_max = 0.0
    fcg_delta_ts: List[float] = []
    fcg_w_chan_finer_range_max = 0.0
    fcg_w_chan_finer_std_max = 0.0
    n_select_ticks = 0

    # ----- ARC-110 loop-segregation diagnostics (P2 select ticks; the non-degeneracy net) -----
    loop_active_ticks = 0
    loop_noise_active_ticks = 0
    loop_d1d2_active_ticks = 0
    loop_committed_neq_motor_ticks = 0   # a non-motor loop flipped the within-eligible winner
    loop_disagree_ticks = 0              # any loop's within-loop winner != motor winner
    loop_assoc_range_sum = 0.0
    loop_limbic_range_sum = 0.0
    loop_d1d2_conflict_sum = 0.0
    loop_local_credited_sum = 0.0
    loop_local_credited_n = 0
    n_loop_diag_ticks = 0
    # ARC-110 C2 RELEASE (707b) per-named-channel routing diagnostics (peak over P2 ticks).
    loop_named_routing_active_ticks = 0
    loop_limbic_routed_range_peak = 0.0          # peak limbic-loop routed per-candidate range
    loop_named_routed_range_peaks: Dict[str, float] = {}   # per-named-channel peak routed range

    for ep in range(total_train_eps):
        is_p1 = (p1_start <= ep < p2_start)
        is_p2 = (ep >= p2_start)
        is_p2_second_half = (ep >= p2_mid)
        phase_label = "P2" if is_p2 else ("P1" if is_p1 else "P0")

        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

        ep_reward = 0.0
        ep_buf: List[Tuple[torch.Tensor, int]] = []

        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs_harm(obs_dict),
                obs_harm_a=_obs_harm_a(obs_dict),
                obs_harm_history=_obs_harm_history(obs_dict),
            )

            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(
                    z_self_prev, action_prev, latent.z_self.detach()
                )

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            pre_e3_classes: List[int] = []
            if is_p2 and candidates:
                pre_e3_classes = sorted({
                    _traj_first_action_class(t) for t in candidates
                })

            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            action = agent.select_action(candidates, ticks)
            n_select_ticks += 1
            # ARC-110: read the segregated-loop diagnostics from the last e3 select (P2 only,
            # when the loop path actually ran). These are the non-degeneracy net: a loop that
            # FLIPPED the within-eligible winner (loop_committed_neq_motor_winner) / disagreed
            # with the motor loop (loop_cross_loop_winner_disagreement) carries live cross-loop
            # variance; a "segregated" loop pinned to the motor winner is a vacuous split.
            if is_p2:
                diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
                if diag.get("loop_segregation_active", False):
                    n_loop_diag_ticks += 1
                    loop_active_ticks += 1
                    if diag.get("loop_segregation_noise_active", False):
                        loop_noise_active_ticks += 1
                    if diag.get("loop_d1_d2_active", False):
                        loop_d1d2_active_ticks += 1
                    if diag.get("loop_committed_neq_motor_winner", False):
                        loop_committed_neq_motor_ticks += 1
                    if diag.get("loop_cross_loop_winner_disagreement", False):
                        loop_disagree_ticks += 1
                    loop_assoc_range_sum += float(diag.get("loop_assoc_pref_range", 0.0) or 0.0)
                    loop_limbic_range_sum += float(diag.get("loop_limbic_pref_range", 0.0) or 0.0)
                    loop_d1d2_conflict_sum += float(diag.get("loop_d1_d2_conflict_signal", 0.0) or 0.0)
                    _lc = diag.get("loop_local_credited_channels", -1)
                    if _lc is not None and int(_lc) >= 0:
                        loop_local_credited_sum += float(_lc)
                        loop_local_credited_n += 1
                    # ARC-110 C2 RELEASE (707b): per-named-channel routed per-candidate range
                    # (the non-degeneracy gate reads this). Peak over P2 ticks per channel +
                    # the limbic-loop max. A flat (~0) limbic routed range == the MECH-191
                    # phasic gap unfixed -> the gate self-routes substrate_not_ready_requeue.
                    if diag.get("loop_named_channel_routing_active", False):
                        loop_named_routing_active_ticks += 1
                        loop_limbic_routed_range_peak = max(
                            loop_limbic_routed_range_peak,
                            float(diag.get("loop_limbic_routed_max_range", 0.0) or 0.0),
                        )
                        for _nm, _rg in (diag.get("loop_named_channel_routed_ranges", {}) or {}).items():
                            loop_named_routed_range_peaks[_nm] = max(
                                loop_named_routed_range_peaks.get(_nm, 0.0), float(_rg or 0.0)
                            )
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at arm={arm['arm_id']} seed={seed} "
                        f"phase={phase_label} ep={ep} step={_step}"
                    )
                break

            committed_class = int(action[0].argmax().item())

            if is_p1 and p1_snap_summaries is not None:
                sel = 0
                for ci, c in enumerate(candidates):
                    if (
                        getattr(c, "actions", None) is not None
                        and c.actions.shape[1] >= 1
                        and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                        == committed_class
                    ):
                        sel = min(ci, p1_snap_summaries.shape[0] - 1)
                        break
                ep_buf.append((p1_snap_summaries, sel))

            if is_p2:
                n_p2_ticks += 1
                committed_class_counts[committed_class] = (
                    committed_class_counts.get(committed_class, 0) + 1
                )
                if is_p2_second_half:
                    committed_class_counts_p2b[committed_class] = (
                        committed_class_counts_p2b.get(committed_class, 0) + 1
                    )
                else:
                    committed_class_counts_p2a[committed_class] = (
                        committed_class_counts_p2a.get(committed_class, 0) + 1
                    )
                if len(pre_e3_classes) >= 2:
                    n_p2_pre_ge2 += 1

                if candidates and len(candidates) >= 2:
                    consumed = _consumed_summaries(agent, candidates)
                    if consumed is not None and torch.isfinite(consumed).all():
                        d = _mean_pairwise_l2(consumed)
                        if math.isfinite(d):
                            consumed_dists.append(d)
                            consumed_dist_max = max(consumed_dist_max, d)

                crf = getattr(agent, "candidate_rule_field", None)
                if crf is not None:
                    st = crf.get_state()
                    crf_n_active_per_tick.append(int(st.get("crf_n_active_last", 0)))
                    crf_n_minted_total_last = int(st.get("crf_n_minted_total", 0))
            elif is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 ONLY (e2 frozen in P1/P2 for stable measurement).
            if (not is_p1) and (not is_p2) and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val):
                    n_p0_contrastive_steps += 1

            _, _harm_signal, done, info, obs_dict = env.step(action)
            if is_p1:
                ep_reward += float(_harm_signal)
            # update_residue drives e3.post_action_update -> the ARC-108 (A1 w_chan) /
            # MECH-451 (A2 w_chan_finer) three-factor learning fires here on EVERY waking
            # tick (all phases). On ARM_NOISE learned_channel_gating_eta=0.0 -> w_chan_finer
            # unchanged (frozen at the injected random vector).
            with torch.no_grad():
                resid_metrics = agent.update_residue(
                    harm_signal=float(_harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )
            # Capture the learning diagnostics surfaced by post_action_update (e3_ prefix).
            ldt = resid_metrics.get("e3_lcg_delta_t")
            if ldt is not None:
                lcg_delta_ts.append(float(ldt.item()))
            lwr = resid_metrics.get("e3_lcg_w_chan_range")
            if lwr is not None:
                lcg_w_chan_range_max = max(lcg_w_chan_range_max, float(lwr.item()))
            fdt = resid_metrics.get("e3_fcg_delta_t")
            if fdt is not None:
                fcg_delta_ts.append(float(fdt.item()))
            fwr = resid_metrics.get("e3_fcg_w_chan_finer_range")
            if fwr is not None:
                fcg_w_chan_finer_range_max = max(
                    fcg_w_chan_finer_range_max, float(fwr.item())
                )
            fws = resid_metrics.get("e3_fcg_w_chan_finer_std")
            if fws is not None:
                fcg_w_chan_finer_std_max = max(
                    fcg_w_chan_finer_std_max, float(fws.item())
                )

            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            tick_in_ep += 1
            if done:
                break

        if is_p1:
            reinforce_baseline = (
                EMA_DECAY * reinforce_baseline + (1.0 - EMA_DECAY) * ep_reward
            )
            for cand_features, sel in ep_buf:
                outcome_buf.append((cand_features, sel, ep_reward))
            if len(outcome_buf) > OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
            l_loss = _lpfc_reinforce_loss(
                agent, outcome_buf, reinforce_baseline, agent.device
            )
            if l_loss.requires_grad:
                bias_opt.zero_grad()
                l_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.lateral_pfc.bias_head_parameters(), 1.0
                )
                bias_opt.step()
                n_p1_bias_updates += 1

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

    # ----- Per-seed aggregation (over P2) -----
    committed_class_entropy = _entropy_from_int_counts(committed_class_counts)
    committed_class_entropy_p2a = _entropy_from_int_counts(committed_class_counts_p2a)
    committed_class_entropy_p2b = _entropy_from_int_counts(committed_class_counts_p2b)

    frac_pre_ge2 = float(n_p2_pre_ge2 / n_p2_ticks) if n_p2_ticks > 0 else 0.0
    consumed_spread_mean = (
        float(sum(consumed_dists) / len(consumed_dists)) if consumed_dists else 0.0
    )

    if crf_n_active_per_tick:
        frac_crf_active_ge_floor = float(
            sum(1 for n in crf_n_active_per_tick if n >= CRF_N_ACTIVE_FLOOR)
            / len(crf_n_active_per_tick)
        )
    else:
        frac_crf_active_ge_floor = 0.0
    crf_differentiated = bool(
        crf_n_minted_total_last >= CRF_MIN_MINTED
        and frac_crf_active_ge_floor >= CRF_FRAC_ACTIVE_FLOOR
    )

    # Learning non-vacuity per-seed.
    lcg_delta_t_std = float(statistics.pstdev(lcg_delta_ts)) if len(lcg_delta_ts) >= 2 else 0.0
    fcg_delta_t_std = float(statistics.pstdev(fcg_delta_ts)) if len(fcg_delta_ts) >= 2 else 0.0

    seed_class_axis_exercisable = bool(frac_pre_ge2 > FRAC_PRE_GE2_FLOOR)
    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )

    # Per-arm learning-engaged non-vacuity (only meaningful on the armed arms).
    lcg_moved = bool(lcg_w_chan_range_max > W_CHAN_RANGE_FLOOR)
    lcg_delta_nonflat = bool(lcg_delta_t_std > DELTA_T_STD_FLOOR)
    # MECH-451 gate (1): finer channels MOVED + DISSOCIABLE (range above floor).
    fcg_moved = bool(fcg_w_chan_finer_range_max > W_CHAN_FINER_RANGE_FLOOR)
    fcg_delta_nonflat = bool(fcg_delta_t_std > DELTA_T_STD_FLOOR)

    # ----- ARC-110 loop-segregation per-seed aggregation (over P2 select ticks) -----
    loop_n = max(n_loop_diag_ticks, 1)
    loop_frac_committed_neq_motor = float(loop_committed_neq_motor_ticks / loop_n) if n_loop_diag_ticks else 0.0
    loop_frac_disagree = float(loop_disagree_ticks / loop_n) if n_loop_diag_ticks else 0.0
    loop_assoc_range_mean = float(loop_assoc_range_sum / loop_n) if n_loop_diag_ticks else 0.0
    loop_limbic_range_mean = float(loop_limbic_range_sum / loop_n) if n_loop_diag_ticks else 0.0
    loop_d1d2_conflict_mean = float(loop_d1d2_conflict_sum / loop_n) if n_loop_diag_ticks else 0.0
    loop_local_credited_mean = (
        float(loop_local_credited_sum / loop_local_credited_n) if loop_local_credited_n else -1.0
    )
    # Live cross-loop variance: a non-motor loop FLIPPED the commit OR loops disagreed on a
    # non-trivial fraction of ticks, AND at least one non-motor loop carries pref range > 0.
    seed_loop_cross_variance = bool(
        n_loop_diag_ticks > 0
        and (loop_frac_committed_neq_motor > LOOP_CROSS_VARIANCE_FRAC_FLOOR
             or loop_frac_disagree > LOOP_CROSS_VARIANCE_FRAC_FLOOR)
        and (loop_assoc_range_mean > LOOP_PREF_RANGE_FLOOR
             or loop_limbic_range_mean > LOOP_PREF_RANGE_FLOOR)
    )
    # ARC-110 C2 RELEASE (707b) per-seed non-degeneracy: at least one LIMBIC channel reached
    # the arbitration carrying a routed per-candidate range above the substantive floor (peak
    # over P2 ticks). This is the C2-specific precondition the 707 vacuous DROP==A1 lacked.
    limbic_routed_peaks = [
        loop_named_routed_range_peaks.get(nm, 0.0) for nm in LIMBIC_NAMED_CHANNELS
    ]
    loop_limbic_routed_range_max = float(max(limbic_routed_peaks)) if limbic_routed_peaks else 0.0
    seed_named_channel_routing_live = bool(
        loop_named_routing_active_ticks > 0
        and loop_limbic_routed_range_max > LIMBIC_ROUTED_RANGE_FLOOR
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "finer_on": bool(arm["finer_on"]),
        "loop_seg": bool(arm.get("loop_seg", False)),
        "loop_noise": bool(arm.get("loop_noise", False)),
        "drop_limbic": bool(arm.get("drop_limbic", False)),
        # ----- ARC-110 loop diagnostics -----
        "loop_active_ticks": int(loop_active_ticks),
        "loop_noise_active_ticks": int(loop_noise_active_ticks),
        "loop_d1d2_active_ticks": int(loop_d1d2_active_ticks),
        "loop_frac_committed_neq_motor": round(loop_frac_committed_neq_motor, 6),
        "loop_frac_disagree": round(loop_frac_disagree, 6),
        "loop_assoc_pref_range": round(loop_assoc_range_mean, 6),
        "loop_limbic_pref_range": round(loop_limbic_range_mean, 6),
        "loop_d1d2_conflict_signal": round(loop_d1d2_conflict_mean, 6),
        "loop_local_credited_channels_mean": round(loop_local_credited_mean, 4),
        "loop_cross_variance": seed_loop_cross_variance,
        # ----- ARC-110 C2 RELEASE (707b) per-named-channel routing diagnostics -----
        "loop_named_routing_active_ticks": int(loop_named_routing_active_ticks),
        "loop_limbic_routed_range_max": round(loop_limbic_routed_range_max, 6),
        "loop_named_routed_range_peaks": {
            str(k): round(float(v), 6) for k, v in sorted(loop_named_routed_range_peaks.items())
        },
        "named_channel_routing_live": seed_named_channel_routing_live,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
        # ----- PRIMARY DV (committed-class entropy) -----
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "committed_class_entropy_p2_first_half_nats": round(committed_class_entropy_p2a, 6),
        "committed_class_entropy_p2_second_half_nats": round(committed_class_entropy_p2b, 6),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        # ----- Readiness / non-vacuity -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "class_axis_exercisable": seed_class_axis_exercisable,
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "crf_differentiated": crf_differentiated,
        # ----- A1 global-w_chan learning diagnostics -----
        "lcg_n_updates": int(len(lcg_delta_ts)),
        "lcg_delta_t_std": round(lcg_delta_t_std, 8),
        "lcg_w_chan_range_max": round(lcg_w_chan_range_max, 8),
        "lcg_moved": lcg_moved,
        "lcg_delta_nonflat": lcg_delta_nonflat,
        # ----- A2 / ARM_NOISE finer-channel learning diagnostics (MECH-451) -----
        "fcg_n_updates": int(len(fcg_delta_ts)),
        "fcg_delta_t_std": round(fcg_delta_t_std, 8),
        "fcg_w_chan_finer_range_max": round(fcg_w_chan_finer_range_max, 8),
        "fcg_w_chan_finer_std_max": round(fcg_w_chan_finer_std_max, 8),
        "fcg_moved": fcg_moved,
        "fcg_delta_nonflat": fcg_delta_nonflat,
    }


def _arm_rows(arm_results: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [
        r for r in arm_results
        if r["arm_id"] == arm_id and r["error_note"] is None
    ]


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _median(vals: List[float]) -> float:
    return float(statistics.median(vals)) if vals else 0.0


def _by_seed(rows: List[Dict[str, Any]], key: str) -> Dict[int, float]:
    return {int(r["seed"]): float(r[key]) for r in rows}


def _gap_by_seed(rows: List[Dict[str, Any]]) -> Dict[int, bool]:
    return {int(r["seed"]): bool(r["gapa_divergence"]) for r in rows}


def _div_pass(n_ok: int, n_div: int) -> bool:
    """A criterion PASSES on the divergent seeds iff there are >= MIN_DIVERGENT_SEEDS
    divergent seeds AND the criterion holds on a strict-majority-ish fraction of them
    (>= ceil(DIVERGENT_PASS_FRACTION * n_div)), with an absolute floor of
    MIN_SEEDS_FOR_PASS divergent seeds."""
    if n_div < MIN_DIVERGENT_SEEDS:
        return False
    needed = max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * n_div)))
    return n_ok >= needed


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arm_results: List[Dict[str, Any]] = []
    script_path = Path(__file__).resolve()
    n_reuse_hits = 0

    for arm in ARMS:
        print(
            f"Arm {arm['arm_id']} ({arm['label']}) loop_seg={arm.get('loop_seg', False)} "
            f"loop_noise={arm.get('loop_noise', False)} drop_limbic={arm.get('drop_limbic', False)} "
            f"finer_on={arm['finer_on']} "
            f"(P0={p0_episodes} ep e2-train, P1={p1_episodes} ep bias-train, "
            f"P2={p2_episodes} ep measure, steps_per_episode={steps_per_episode}, "
            f"dry_run={dry_run})",
            flush=True,
        )
        reusable = arm["arm_id"] in REUSABLE_ARM_IDS_LOCAL
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)

            # ----- ARM-REUSE (consumer side), gated + safe-by-default -----
            # Only A0_SINGLE_ARENA is reuse-eligible, and only IFF a mint is cited. With
            # REUSE_BASELINE_FROM=None (the default) this is skipped; every arm runs fresh.
            row: Optional[Dict[str, Any]] = None
            if REUSE_BASELINE_FROM is not None and reusable:
                cell = try_reuse_cell(
                    config_slice=_arm_config_slice(
                        arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                    ),
                    seed=s,
                    script_path=script_path,
                    needed_keys=REUSE_NEEDED_KEYS,
                    cite_run_id=REUSE_BASELINE_FROM,
                    include_driver_script_in_hash=False,
                )
                if cell is not None:
                    row = dict(cell)  # reuse hit; provenance stamped by try_reuse_cell
                    n_reuse_hits += 1

            if row is None:
                row = _run_seed_arm(
                    arm, s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode,
                )

            # Per-cell fingerprint. A0_SINGLE_ARENA emits a REUSE-ELIGIBLE fingerprint
            # (MINT-AS-YOU-GO: config slice declared, include_driver_script_in_hash=False).
            # The loop arms ride the just-built loop substrate (in flux) -- not reusable.
            if reusable:
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(
                        arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                    ),
                    seed=s,
                    script_path=script_path,
                    rng_fully_reset=True,
                    config_slice_declared=True,
                    include_driver_script_in_hash=False,
                )
            else:
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(
                        arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                    ),
                    seed=s,
                    script_path=script_path,
                    rng_fully_reset=True,
                    extra_ineligible_reasons=[
                        "arc110_loop_substrate_just_built_in_flux_not_a_reusable_baseline",
                    ],
                )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    a0_rows = _arm_rows(arm_results, "A0_SINGLE_ARENA")
    a1_rows = _arm_rows(arm_results, "A1_LOOPS")
    noise_rows = _arm_rows(arm_results, "ARM_NOISE")
    drop_rows = _arm_rows(arm_results, "ARM_DROP_LIMBIC")
    all_rows = a0_rows + a1_rows + noise_rows + drop_rows

    def _maj(rows: List[Dict[str, Any]], pred) -> bool:
        return sum(1 for r in rows if pred(r)) >= MIN_SEEDS_FOR_PASS

    a0_ent = _by_seed(a0_rows, "committed_class_entropy_nats")
    a1_ent = _by_seed(a1_rows, "committed_class_entropy_nats")
    noise_ent = _by_seed(noise_rows, "committed_class_entropy_nats")
    drop_ent = _by_seed(drop_rows, "committed_class_entropy_nats")

    a0_gap = _gap_by_seed(a0_rows)
    a1_gap = _gap_by_seed(a1_rows)
    noise_gap = _gap_by_seed(noise_rows)
    drop_gap = _gap_by_seed(drop_rows)

    # ----- Per-seed-divergent gating: seeds whose pool is divergent on the C1 comparison
    # arms (A0 + A1 + ARM_NOISE all divergent). A degenerate cell self-excludes. -----
    primary_div = [
        s for s in sorted(set(a0_gap) & set(a1_gap) & set(noise_gap))
        if a0_gap.get(s) and a1_gap.get(s) and noise_gap.get(s)
    ]
    n_primary_div = len(primary_div)
    enough_divergent = n_primary_div >= MIN_DIVERGENT_SEEDS

    # ----- Precondition (ARC-110 non-degeneracy): loops carry LIVE cross-loop variance on
    # A1 (a non-motor loop flipped the within-eligible winner / loops disagreed, AND a
    # non-motor loop carries pref range > 0) on a majority of seeds. A "segregated" loop
    # pinned to the motor winner is a vacuous split -> requeue. -----
    loop_cross_variance_ok = _maj(a1_rows, lambda r: r.get("loop_cross_variance", False))
    a1_loop_flip_min = float(min([r.get("loop_frac_committed_neq_motor", 0.0) for r in a1_rows] or [0.0]))
    a1_loop_disagree_min = float(min([r.get("loop_frac_disagree", 0.0) for r in a1_rows] or [0.0]))

    # ----- Precondition (707a FIX): the S2 in-layer null (ARM_NOISE) is a LIVE same-layer
    # perturbation -- on a strict-majority of divergent seeds it actively perturbs the
    # committed selection (loop_noise_active_ticks > 0 AND loop_frac_committed_neq_motor or
    # loop_frac_disagree > the cross-loop-variance floor). THIS is the same-layer null the
    # single arena could not construct (the 704b/706b binding constraint); if even the in-layer
    # null is INERT (does not reach the committed-class readout) the null-validity problem
    # persists -> requeue, NOT a weakens.
    #
    # WHY null-LIVENESS, not null-LIFT (the 707->707a fix): 707 required the null to LIFT
    # committed-class entropy strict-ABOVE A0 (n_noise_lifts = noise_ent > a0_ent + margin).
    # That is the 700-lineage TEMPERATURE-null liveness proxy (raising softmax temperature
    # trivially raises entropy) and is INVALID for a structured-accumulator null: random
    # magnitude-matched content routed through the SAME loop path (per-loop zscore + Haber
    # spiral, motor never nulled) lands committed-class entropy near BASELINE (structured loops
    # beat random; random ~ A0), NOT above it. In 707 the null was demonstrably live (it
    # perturbed selection on 4/4 divergent seeds) yet sat at/below A0 on 3/4, so the run was
    # spuriously requeued and the pre-registered WEAKENS branch was unreachable. The non-vacuity
    # the null actually has to provide is that "A1 strict-above the null" (C1) is a real test --
    # i.e. the null is a LIVE magnitude-matched alternative, not a dead control. The path-
    # artifact hypothesis (A1's lift is a zscore-path artefact) is ruled out by C1 itself (a
    # live null on the SAME zscore path would match A1 and fail C1), NOT by null-lift.
    # n_noise_lifts is still RECORDED for continuity but no longer gates.
    noise_by_seed = {int(r["seed"]): r for r in noise_rows}

    def _noise_live(s: int) -> bool:
        r = noise_by_seed.get(s)
        if not r:
            return False
        return bool(
            int(r.get("loop_noise_active_ticks", 0)) > 0
            and (
                float(r.get("loop_frac_committed_neq_motor", 0.0)) > LOOP_CROSS_VARIANCE_FRAC_FLOOR
                or float(r.get("loop_frac_disagree", 0.0)) > LOOP_CROSS_VARIANCE_FRAC_FLOOR
            )
        )

    n_noise_live = sum(1 for s in primary_div if _noise_live(s))
    n_noise_lifts = sum(
        1 for s in primary_div if noise_ent.get(s, 0.0) > a0_ent.get(s, 0.0) + CONVERSION_MARGIN
    )  # recorded for continuity with 707; NOT a gate in 707a
    noise_verified_lifting = bool(enough_divergent and _div_pass(n_noise_live, n_primary_div))

    # ----- Precondition (ARC-110 C2 RELEASE, 707b): per-NAMED-channel routing is LIVE on A1.
    # At least one LIMBIC channel (ofc/liking/vigour) reached the arbitration carrying a routed
    # per-candidate range above the substantive floor, on a strict-majority of DIVERGENT seeds.
    # This is evaluated BEFORE C2 is scored (it is part of preconditions_met, and C2 is only
    # read in the c1_holds PASS branch). If the named limbic channels are still per-candidate-
    # FLAT (MECH-191 phasic gap unfixed), the limbic loop carries no competition, ARM_DROP_LIMBIC
    # is a no-op (the 707 vacuous DROP==A1), and the run self-routes substrate_not_ready_requeue
    # -- NEVER a false weakens. -----
    named_routing_live_div = [
        s for s in primary_div
        if next((r for r in a1_rows if int(r["seed"]) == s), {}).get("named_channel_routing_live", False)
    ]
    n_named_routing_live = len(named_routing_live_div)
    named_channel_routing_live = bool(enough_divergent and _div_pass(n_named_routing_live, n_primary_div))
    a1_limbic_routed_range_max = float(
        max([r.get("loop_limbic_routed_range_max", 0.0) for r in a1_rows] or [0.0])
    )

    # ----- Precondition: learning engaged on A1 (finer channels dissociable + delta_t nonflat) -----
    fcg_moved_ok = _maj(a1_rows, lambda r: r.get("fcg_moved", False))
    fcg_delta_nonflat_ok = _maj(a1_rows, lambda r: r.get("fcg_delta_nonflat", False))

    # CRF maturity (matched constant; majority of seeds on all arms).
    crf_matured = all(
        _maj(rows, lambda r: r["crf_differentiated"]) for rows in
        (a0_rows, a1_rows, noise_rows, drop_rows)
    )

    preconditions_met = bool(
        enough_divergent
        and loop_cross_variance_ok
        and named_channel_routing_live    # ARC-110 C2 RELEASE (707b): limbic channels carry routed range
        and noise_verified_lifting
        and fcg_moved_ok and fcg_delta_nonflat_ok
        and crf_matured
    )

    # ----- C1 (loop-segregated conversion): A1_LOOPS committed-class entropy strict-above
    # BOTH A0_SINGLE_ARENA AND the VALID in-layer null ARM_NOISE, on a strict-majority of
    # divergent seeds. -----
    c1_seeds: List[int] = []
    for s in primary_div:
        bar = max(a0_ent.get(s, 0.0), noise_ent.get(s, 0.0)) + CONVERSION_MARGIN
        if a1_ent.get(s, 0.0) > bar:
            c1_seeds.append(s)
    n_c1 = len(c1_seeds)
    c1_holds = _div_pass(n_c1, n_primary_div)

    # ----- C2 (ARC-106 per-loop load-bearing): A1_LOOPS strict-above ARM_DROP_LIMBIC on a
    # strict-majority of divergent seeds where DROP is divergent. If dropping the limbic loop
    # does NOT remove the lift, the limbic loop is DECORATIVE. -----
    c2_div = [s for s in primary_div if drop_gap.get(s)]
    c2_seeds = [
        s for s in c2_div
        if a1_ent.get(s, 0.0) > drop_ent.get(s, 0.0) + CONVERSION_MARGIN
    ]
    c2_holds = bool(c1_holds and len(c2_div) >= MIN_SEEDS_FOR_PASS and _div_pass(len(c2_seeds), len(c2_div)))

    # ----- C2 NON-DEGENERACY DIAGNOSTIC (707a, additive; does NOT gate any verdict). 707
    # showed ARM_DROP_LIMBIC byte-identical to A1_LOOPS on every seed (same committed-class
    # entropy + same flip/disagree fractions), i.e. the limbic-loop ablation produced ZERO
    # change in the committed readout, so C2 (limbic load-bearing) was untestable -- it could
    # only ever read passed=false for a degenerate reason, not because the limbic loop is
    # decorative. Record per-seed whether DROP actually DIFFERS from A1 so a future read /
    # autopsy can tell "limbic decorative" (DROP differs, A1 not above it) from "ablation
    # inert / untestable" (DROP == A1). C2 is load_bearing=false and this flag changes NO
    # outcome; it only annotates the manifest. -----
    DROP_DIFF_EPS = 1e-6
    c2_drop_differs_seeds = [
        s for s in primary_div
        if abs(a1_ent.get(s, 0.0) - drop_ent.get(s, 0.0)) > DROP_DIFF_EPS
    ]
    c2_drop_differs_from_a1 = bool(c2_drop_differs_seeds)

    # ----- Outcome map (decisive either way) -----
    if not preconditions_met:
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "ARC-110 loop-segregated conversion could NOT be validly measured: a precondition "
            "is unmet (too few divergent seeds / loops carry NO live cross-loop variance = a "
            "vacuous split pinned to the motor winner / NAMED limbic channels carry NO routed "
            "per-candidate range = the MECH-191 phasic gap unfixed, so DROP_LIMBIC is a no-op "
            "(the 707 vacuous DROP==A1) / the in-layer null is INERT = does not perturb the "
            "committed selection / finer channels not dissociable / delta_t flat). NOT a "
            "falsification."
        )
        per_claim = {"ARC-110": "non_contributory"}
    elif c1_holds:
        outcome = "PASS"
        overall_direction = "supports"
        non_degenerate = True
        degeneracy_reason = ""
        if c2_holds:
            label = "loop_segregation_converts_limbic_loop_load_bearing_supports_arc110"
        else:
            label = "loop_segregation_converts_limbic_not_sole_driver_supports_arc110"
        per_claim = {"ARC-110": "supports"}
    else:
        outcome = "FAIL"
        overall_direction = "weakens"
        non_degenerate = True
        degeneracy_reason = ""
        label = "valid_null_loops_vary_but_no_conversion_ceiling_intrinsic_weakens_arc110"
        per_claim = {"ARC-110": "weakens"}

    a0_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a0_rows])
    a1_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a1_rows])
    noise_mean_dv = _mean([r["committed_class_entropy_nats"] for r in noise_rows])
    drop_mean_dv = _mean([r["committed_class_entropy_nats"] for r in drop_rows])

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "enough_divergent_seeds",
                "kind": "readiness",
                "description": (
                    "number of seeds whose candidate pool is DIVERGENT on ALL C1 comparison "
                    "arms (A0 + A1 + ARM_NOISE) >= MIN_DIVERGENT_SEEDS. Per-seed-divergent "
                    "gating; too few => substrate_not_ready_requeue (pool too collapsed to "
                    "test conversion)."
                ),
                "control": "consumed cand_world_summary pairwise spread > floor (GAP-A); per-seed",
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                "met": bool(enough_divergent),
            },
            {
                "name": "loops_carry_live_cross_loop_variance",
                "kind": "readiness",
                "description": (
                    "ARC-110 NON-DEGENERACY guard (what_would_answer): on A1_LOOPS a non-motor "
                    "loop must FLIP the within-eligible winner (loop_committed_neq_motor_winner) "
                    "or the loops must DISAGREE (loop_cross_loop_winner_disagreement) on a "
                    "non-trivial fraction of P2 ticks, AND a non-motor loop must carry per-loop "
                    "preference RANGE > 0 -- on a majority of seeds. A loop pinned to the motor "
                    "winner is a vacuous split => substrate_not_ready_requeue. measured = min "
                    "across A1 seeds of the flip fraction."
                ),
                "control": "A1_LOOPS loop_frac_committed_neq_motor / loop_frac_disagree + per-loop pref range",
                "measured": float(round(max(a1_loop_flip_min, a1_loop_disagree_min), 6)),
                "threshold": float(LOOP_CROSS_VARIANCE_FRAC_FLOOR),
                "met": bool(loop_cross_variance_ok),
            },
            {
                "name": "named_channel_routing_live",
                "kind": "readiness",
                "description": (
                    "ARC-110 C2 RELEASE non-degeneracy gate (707b): on A1_LOOPS at least one "
                    "LIMBIC channel (ofc/liking/vigour) must reach the segregated-loop "
                    "arbitration carrying a routed per-candidate RANGE > LIMBIC_ROUTED_RANGE_FLOOR "
                    "(peak over P2 ticks), on a strict-majority of DIVERGENT seeds. This is the "
                    "MECH-191 phasic-externalisation gate: the named bias HEADS emit "
                    "per-candidate-FLAT output, so without the use_named_channel_routing release "
                    "the limbic loop is inert and ARM_DROP_LIMBIC is byte-identical to A1 (the "
                    "707 vacuous DROP==A1). Evaluated BEFORE C2 is scored. measured = max across "
                    "A1 seeds of the limbic routed range; SAME statistic (per-candidate range) "
                    "the load-bearing C2 ablation depends on, not a magnitude proxy."
                ),
                "control": "A1_LOOPS loop_limbic_routed_range_max (peak limbic routed per-candidate range over P2)",
                "measured": float(round(a1_limbic_routed_range_max, 6)),
                "threshold": float(LIMBIC_ROUTED_RANGE_FLOOR),
                "met": bool(named_channel_routing_live),
            },
            {
                "name": "in_layer_null_live",
                "kind": "readiness",
                "description": (
                    "707a FIX (supersedes in_layer_null_verified_lifting): the S2 IN-LAYER "
                    "same-layer null (ARM_NOISE: magnitude-matched random structure at the "
                    "non-motor loop accumulators) is a LIVE perturbation -- it actively perturbs "
                    "the committed selection (loop_noise_active_ticks > 0 AND "
                    "loop_frac_committed_neq_motor or loop_frac_disagree > the cross-loop-variance "
                    "floor) on a strict-majority of DIVERGENT seeds. This is the SAME-LAYER null "
                    "the single arena could NOT construct (704b/706b binding constraint); a null "
                    "that does NOT reach the committed-class readout makes C1's strict-above-null "
                    "bar meaningless => substrate_not_ready_requeue (NOT a weakens). The RETIRED "
                    "707 proxy required the null to LIFT entropy strict-above A0 -- the 700-lineage "
                    "TEMPERATURE-null liveness test, INVALID for a structured-accumulator null: "
                    "random magnitude-matched loop content lands at baseline, not above it "
                    "(structured beats random; random ~ A0), so it spuriously requeued a valid "
                    "result and made the WEAKENS branch unreachable. The path-artifact hypothesis "
                    "is ruled out by C1 (A1 strict-above the live null), NOT by null-lift. "
                    "measured = n divergent seeds the null is LIVE on (n_noise_lifts recorded "
                    "separately for continuity, no longer gates)."
                ),
                "control": "ARM_NOISE loop_frac_committed_neq_motor / loop_frac_disagree + loop_noise_active_ticks, divergent seeds",
                "measured": float(n_noise_live),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "met": bool(noise_verified_lifting),
            },
            {
                "name": "learning_engaged_finer_channels_dissociable_delta_nonflat",
                "kind": "readiness",
                "description": (
                    "on A1_LOOPS the finer w_chan_finer entries MOVED + carry cross-channel "
                    "range above floor AND the signed-RPE delta_t carries cross-tick variance, "
                    "on a majority of seeds -- learning is engaged (else the loops are reading "
                    "an un-trained gate). Below floor => substrate_not_ready_requeue."
                ),
                "control": "A1 fcg_w_chan_finer_range_max + fcg_delta_t_std",
                "measured": float(min([r["fcg_w_chan_finer_range_max"] for r in a1_rows] or [0.0])),
                "threshold": float(W_CHAN_FINER_RANGE_FLOOR),
                "met": bool(fcg_moved_ok and fcg_delta_nonflat_ok),
            },
            {
                "name": "candidate_pool_divergent",
                "kind": "readiness",
                "description": (
                    "consumed cand_world_summaries (e2.world_forward) per-candidate SPREAD "
                    "clears the floor on a majority of seeds (GAP-A non-vacuity)."
                ),
                "control": "SD-056 e2 trained online in P0; candidate_summary_source=e2_world_forward",
                "measured": float(min([r["consumed_summary_pairwise_dist_mean"] for r in all_rows] or [0.0])),
                "threshold": float(CONSUMED_SPREAD_FLOOR),
                "met": bool(enough_divergent),
            },
        ],
        "criteria": [
            {
                "name": "C1_A1_loops_strict_above_A0_and_in_layer_null",
                "load_bearing": True,
                "passed": bool(c1_holds),
            },
            {
                "name": "C2_limbic_loop_load_bearing_A1_above_drop_limbic",
                "load_bearing": False,
                "passed": bool(c2_holds),
            },
        ],
        "criteria_non_degenerate": {
            "preconditions_met": bool(preconditions_met),
            "enough_divergent_seeds": bool(enough_divergent),
            "loops_carry_live_cross_loop_variance": bool(loop_cross_variance_ok),
            "named_channel_routing_live": bool(named_channel_routing_live),
            "in_layer_null_live": bool(noise_verified_lifting),
            "learning_engaged": bool(fcg_moved_ok and fcg_delta_nonflat_ok),
            "crf_matured": bool(crf_matured),
            "c2_drop_differs_from_a1": bool(c2_drop_differs_from_a1),
        },
    }

    total_seeds = len(ARMS) * len(seeds)
    total_completed = len(all_rows)

    manifest_core = {
        "outcome": outcome,
        "overall_direction": overall_direction,
        "evidence_direction_per_claim": per_claim,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": seeds,
        "n_arms": len(ARMS),
        "n_reuse_hits": int(n_reuse_hits),
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "conversion_margin": float(CONVERSION_MARGIN),
            "growth_margin": float(GROWTH_MARGIN),
            "min_divergent_seeds": int(MIN_DIVERGENT_SEEDS),
            "divergent_pass_fraction": float(DIVERGENT_PASS_FRACTION),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "loop_cross_variance_frac_floor": float(LOOP_CROSS_VARIANCE_FRAC_FLOOR),
            "loop_pref_range_floor": float(LOOP_PREF_RANGE_FLOOR),
            "limbic_routed_range_floor": float(LIMBIC_ROUTED_RANGE_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "delta_t_std_floor": float(DELTA_T_STD_FLOOR),
            "w_chan_finer_range_floor": float(W_CHAN_FINER_RANGE_FLOOR),
            "loop_segregation_normalize": str(LOOP_SEGREGATION_NORMALIZE),
            "loop_segregation_noise_alpha": float(LOOP_SEGREGATION_NOISE_ALPHA),
            "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
            "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
            "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
        },
        "acceptance_criteria": {
            "preconditions_met": preconditions_met,
            "n_divergent_seeds": int(n_primary_div),
            "enough_divergent_seeds": enough_divergent,
            "crf_matured": crf_matured,
            "loops_carry_live_cross_loop_variance": loop_cross_variance_ok,
            "a1_loop_flip_frac_min": round(a1_loop_flip_min, 6),
            "a1_loop_disagree_frac_min": round(a1_loop_disagree_min, 6),
            "named_channel_routing_live": named_channel_routing_live,
            "n_named_routing_live_over_divergent": int(n_named_routing_live),
            "a1_limbic_routed_range_max": round(a1_limbic_routed_range_max, 6),
            "limbic_routed_range_floor": float(LIMBIC_ROUTED_RANGE_FLOOR),
            "in_layer_null_live": noise_verified_lifting,
            "n_noise_live_over_divergent": int(n_noise_live),
            "n_noise_lifts_over_a0": int(n_noise_lifts),
            "learning_engaged_fcg_moved": fcg_moved_ok,
            "learning_engaged_fcg_delta_nonflat": fcg_delta_nonflat_ok,
            "C1_loop_conversion_a1_above_a0_and_null": c1_holds,
            "C1_a1_n_seeds": int(n_c1),
            "C1_n_divergent": int(n_primary_div),
            "C2_limbic_load_bearing_a1_above_drop": c2_holds,
            "C2_n_seeds": int(len(c2_seeds)),
            "C2_n_divergent": int(len(c2_div)),
            "C2_drop_differs_from_a1": c2_drop_differs_from_a1,
            "C2_n_drop_differs_seeds": int(len(c2_drop_differs_seeds)),
            "mean_committed_class_entropy_a0_single_arena": round(a0_mean_dv, 6),
            "mean_committed_class_entropy_a1_loops": round(a1_mean_dv, 6),
            "mean_committed_class_entropy_arm_noise_in_layer_null": round(noise_mean_dv, 6),
            "mean_committed_class_entropy_arm_drop_limbic": round(drop_mean_dv, 6),
            "mean_loop_d1d2_conflict_signal_a1": round(_mean([r.get("loop_d1d2_conflict_signal", 0.0) for r in a1_rows]), 6),
            "mean_loop_local_credited_channels_a1": round(_mean([r.get("loop_local_credited_channels_mean", -1.0) for r in a1_rows]), 4),
        },
        "interpretation_grid": {
            "PASS_loop_segregation_converts_limbic_loop_load_bearing_supports_arc110": (
                "preconditions met (ENOUGH divergent seeds + loops carry LIVE cross-loop "
                "variance + the in-layer null is LIVE + learning engaged) AND C1 (A1_LOOPS "
                "committed-class entropy strict-above A0_SINGLE_ARENA AND the valid in-layer null "
                "on a strict-majority of divergent seeds) AND C2 (A1 strict-above ARM_DROP_LIMBIC "
                "-- the limbic loop is load-bearing). Loop segregation CONVERTS committed-action "
                "diversity where the single F-dominated arena plateaued -> the single-arena "
                "collapse WAS a binding constraint on the F-dominance conversion ceiling "
                "(MECH-439) -> supports ARC-110."
            ),
            "PASS_loop_segregation_converts_limbic_not_sole_driver_supports_arc110": (
                "preconditions + C1 met BUT NOT C2 (dropping the limbic loop does NOT remove the "
                "lift). Loop segregation converts, but the associative loop (or the loop structure "
                "per se) carries it -- the limbic loop is not the sole driver. Still supports "
                "ARC-110 (segregation converts); the per-loop load-bearing attribution is partial."
            ),
            "FAIL_valid_null_loops_vary_but_no_conversion_ceiling_intrinsic_weakens_arc110": (
                "DECISIVE. preconditions met (a VALID in-layer null can now be constructed -- "
                "ARM_NOISE is a LIVE perturbation of the committed selection -- and the loops "
                "carry live cross-loop variance) BUT A1_LOOPS does NOT lift committed-class "
                "entropy strict-above A0/the null. The single-arena collapse was NOT the binding "
                "constraint -> the F-dominance conversion ceiling is INTRINSIC, not an artefact "
                "of collapse -> weakens ARC-110 (the loop-segregation-as-artefact hypothesis)."
            ),
            "FAIL_substrate_not_ready_requeue": (
                "A precondition is unmet: too FEW divergent seeds (pool collapsed), OR the loops "
                "carry NO live cross-loop variance (a vacuous split pinned to the motor winner -- "
                "the ARC-110 what_would_answer guard), OR the NAMED limbic channels carry NO routed "
                "per-candidate range (named_channel_routing_live=false -- the MECH-191 phasic gap "
                "unfixed, so ARM_DROP_LIMBIC is a no-op = the 707 vacuous DROP==A1, and C2 cannot "
                "be validly scored), OR the S2 in-layer null is INERT (does NOT perturb the "
                "committed selection even with loops -> null-validity problem persists), OR "
                "learning was not engaged (finer channels not dissociable / delta_t flat). The "
                "conversion question could NOT be measured -- NOT a falsification."
            ),
        },
        "arm_results": arm_results,
    }
    return manifest_core

def _build_manifest(
    result: Dict[str, Any],
    timestamp_utc: str,
    dry_run: bool,
) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "backlog_id": BACKLOG_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "non_degenerate": bool(result["non_degenerate"]),
        "degeneracy_reason": result["degeneracy_reason"],
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "same_layer_null": {
            "layer": (
                "ARC-110 S2: the eligibility/settling field at the non-motor loop accumulators "
                "(the SAME layer the per-loop settling acts on; NOT policy softmax temperature, "
                "the decoupled 700-lineage null)"
            ),
            "mechanism": (
                "config.e3.loop_segregation_noise_on=True -> _loop_inlayer_null replaces each "
                "non-motor loop accumulator with a magnitude-matched random-structure (gaussian) "
                "perturbation, range == loop_segregation_noise_alpha x the real loop accumulator "
                "range. Motor (F) is never nulled. Selection-only (no memory write)."
            ),
            "noise_alpha": LOOP_SEGREGATION_NOISE_ALPHA,
            "match_mode": "in_selector_per_tick_range_matched_to_real_loop_accumulator",
        },
        "reuse_baseline_from": REUSE_BASELINE_FROM,
        "evidence_direction_note": (
            f"V3-EXQ-707b ARC-110 LOOP-SEGREGATION C2 RELEASE VALIDATION "
            f"(supersedes V3-EXQ-707a/707; experiment_purpose=diagnostic; claim_ids=[ARC-110]; "
            f"PROMOTES NOTHING). 707b LANDS THE C2 RELEASE: a no-op-default flag "
            f"use_named_channel_routing routes each named channel's per-candidate REPRESENTATION "
            f"through project_channel_range (range-preserving, the GAP-A path) into the segregated "
            f"loops, so the limbic loop (ofc/liking/vigour) carries REAL per-candidate range and "
            f"ARM_DROP_LIMBIC can differ from A1 -- making C2 (limbic load-bearing) testable. 707/"
            f"707a left C2 substrate-blocked: the named bias HEADS emit per-candidate-FLAT output "
            f"(MECH-191 phasic gap), so DROP_LIMBIC was byte-identical to A1 (vacuous). A NEW "
            f"per-named-channel non-degeneracy precondition (named_channel_routing_live: a limbic "
            f"channel's routed range > floor on a strict-majority of divergent seeds) is evaluated "
            f"BEFORE C2 is scored, so the vacuous DROP==A1 self-routes substrate_not_ready_requeue "
            f"(NEVER a false weakens). The 707a in-layer-null LIVENESS gate (null must PERTURB "
            f"committed selection, not raise entropy above A0) and C1 (A1 strict-above "
            f"max(A0, the live null)+margin) are UNCHANGED. "
            f"Substrate-readiness validation for the v4_loop_segregation substrate (ARC-110 + "
            f"ARC-109 D1/D2 + MECH-452 loop-local traces + the S2 in-layer null) built "
            f"2026-06-27 via /implement-substrate. Routed by the confirmed cluster autopsy "
            f"failure_autopsy_704b-706b-conversion-ceiling_2026-06-27: the conversion-ceiling "
            f"lineage converged from four angles (700b/700c learned-gating+settling, 704b finer-"
            f"channel, 706b double-gated curiosity) on the conclusion that the V3 SINGLE E3 "
            f"foraging arena structurally denies (a) committed-action-class conversion from non-"
            f"motor channels and (b) a VALID same-layer null; the gate cleared by 704b FAIL-to-"
            f"convert (positive-evidence-FOR-ARC-110). 4 arms on the SAME GAP-A reef-bipartite "
            f"foraging substrate + the SAME landed arithmetic envelope + finer-channel gating + "
            f"learned settling as a MATCHED CONSTANT; the ONLY swept factor is loop segregation: "
            f"A0_SINGLE_ARENA (use_loop_segregation=False -- the 704b-A2 single-arena baseline) / "
            f"A1_LOOPS (ARC-110 motor/assoc/limbic + ARC-109 D1/D2 + MECH-452 loop-local traces) "
            f"/ ARM_NOISE (A1 + the S2 in-layer same-layer null) / ARM_DROP_LIMBIC (A1 with the "
            f"limbic loop ablated -- the ARC-106 per-loop load-bearing falsifier). PRE-REGISTERED "
            f"decisive either way: A1 committed-class entropy strict-above A0 AND the VALID in-"
            f"layer null on a strict-majority of divergent seeds, with live cross-loop variance "
            f"=> single-arena collapse WAS a binding constraint -> supports ARC-110 (limbic loop "
            f"load-bearing if A1 also strict-above ARM_DROP_LIMBIC); a LIVE null + "
            f"loops vary BUT A1 does NOT convert => the ceiling is INTRINSIC -> weakens ARC-110. "
            f"Non-vacuity self-route substrate_not_ready_requeue (NEVER a false weakens): loops "
            f"must carry LIVE cross-loop variance (not a vacuous split pinned to the motor "
            f"winner), the NAMED limbic channels must carry routed per-candidate range > floor "
            f"(named_channel_routing_live -- else DROP_LIMBIC is a no-op = the 707 vacuous "
            f"DROP==A1 and C2 cannot be scored), the in-layer null must be LIVE (perturb the "
            f"committed selection, else the null-validity problem persists), the pool divergent "
            f"(GAP-A), finer channels dissociable + delta_t non-"
            f"flat. PROMOTES NOTHING (ARC-110/ARC-109/MECH-452 candidate/substrate_conditional/v3). "
            f"outcome={result['outcome']}; label={result['interpretation_label']}; "
            f"per_claim={result['evidence_direction_per_claim']}."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "4-arm ARC-110 loop-segregation validation (A0_SINGLE_ARENA / A1_LOOPS / ARM_NOISE in-layer null / ARM_DROP_LIMBIC) + per-seed-divergent gating + live-cross-loop-variance non-degeneracy gate + LIVE in-layer same-layer null gate (707a: null-liveness, not null-lift)",
            "arms": "A0_SINGLE_ARENA (use_loop_segregation=False) / A1_LOOPS (ARC-110 motor/assoc/limbic + ARC-109 D1/D2 + MECH-452 loop-local traces) / ARM_NOISE (A1 + loop_segregation_noise_on) / ARM_DROP_LIMBIC (A1, limbic loop remapped into associative)",
            "swept_variables": "use_loop_segregation (A0 off, loop arms on) + loop_segregation_noise_on (ARM_NOISE) + limbic-loop channel-map ablation (ARM_DROP_LIMBIC). Finer-channel gating + learned settling ON on ALL arms.",
            "the_isolated_factor": (
                "loop segregation only: A0 runs the single-arena within-eligible argmin over the "
                "finer channels; A1 partitions the SAME finer channels into motor=F / "
                "associative=dACC+lPFC / limbic=OFC+liking+vigour loops with within-loop "
                "competition first + Haber ascending-spiral arbitration after + per-loop zscore "
                "normalisation (strips F's magnitude advantage). D1/D2 + loop-local traces are "
                "inert without loops, so the functional swept variable IS loop segregation."
            ),
            "matched_constant_arithmetic_envelope": (
                "use_f_eligibility_demotion=True + use_f_eligibility_adaptive_floor=True (689e) + "
                "use_go_nogo_constitution=True (689g) + use_modulatory_selection_authority=True (643a) + "
                "use_modulatory_channel_routing (cand_world_summary) + top_k shortlist (k=3, 569i)"
            ),
            "matched_diversity_stack": (
                "MECH-341 stratified + use_dacc (MECH-260 perseveration No-Go feed) + use_gated_policy + "
                "use_lateral_pfc_analog (lateral_pfc_train_rule_bias_head=True, TRAINED in P1 REINFORCE) + "
                "SD-056 all levers + the matured/maintained CRF pool + use_candidate_rule_field + "
                "use_finer_channel_gating=True + use_learned_settling_step=True"
            ),
            "settling_W_lat": "ON on ALL arms (the within-loop settling each segregated loop runs; on A0 the legacy single-arena settling)",
            "primary_dv": "committed-action-class entropy (nats), interpreted on divergent seeds only",
            "phases": "P0 e2-train (CRF matures, finer gating ON) -> P1 frozen-encoder bias-head REINFORCE -> P2 e2+bias frozen, gating + loops KEEP adapting",
            "learning_wiring": "w_chan_finer learns via e3.post_action_update driven by agent.update_residue every waking tick; on the loop arms credit is loop-local (MECH-452); the S2 null is selection-only (no learning write)",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "loop_segregation_normalize": LOOP_SEGREGATION_NORMALIZE,
            "loop_segregation_noise_alpha": LOOP_SEGREGATION_NOISE_ALPHA,
            "loop_default_channel_map": "motor=F; associative=dACC+lPFC; limbic=OFC+liking+vigour (built-in _LOOP_DEFAULT_CHANNEL_MAP)",
            "drop_limbic_channel_map": "limbic channels remapped into associative -> limbic loop empty (per-loop ablation)",
            "reuse_baseline_from": REUSE_BASELINE_FROM,
            "reusable_arm_ids": list(REUSABLE_ARM_IDS_LOCAL),
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
            "arc110_relationship": "this IS the ARC-110 validation -- the gate cleared by 704b FAIL-to-convert; supports => single-arena collapse was a binding constraint, weakens => ceiling intrinsic",
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-707 ARC-110 parallel segregated loop-segregation validation"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        p2 = DRY_RUN_P2
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1 = P1_BIAS_TRAIN_EPISODES
        p2 = P2_MEASUREMENT_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_episodes=p1,
        p2_episodes=p2,
        steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    if args.dry_run:
        out_path = out_dir / f"_dry_{manifest['run_id']}.json"

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    _ac = result["acceptance_criteria"]
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"reuse_hits={result['n_reuse_hits']} "
        f"preconditions_met={_ac['preconditions_met']} "
        f"n_divergent={_ac['n_divergent_seeds']} "
        f"loop_cross_variance={_ac['loops_carry_live_cross_loop_variance']} "
        f"in_layer_null_live={_ac['in_layer_null_live']} "
        f"n_noise_live={_ac['n_noise_live_over_divergent']} n_noise_lifts={_ac['n_noise_lifts_over_a0']} "
        f"C1_loop_conversion={_ac['C1_loop_conversion_a1_above_a0_and_null']} "
        f"C2_limbic_load_bearing={_ac['C2_limbic_load_bearing_a1_above_drop']} "
        f"C2_drop_differs={_ac['C2_drop_differs_from_a1']} "
        f"label={result['interpretation_label']}",
        flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
    sys.exit(0)
