#!/opt/local/bin/python3
"""
V3-EXQ-713 -- ARC-110 x ARC-108 BOUNDED PARITY-CONTROLLER VALIDATION (V3-EXQ-711 runaway repair).

The SEPARATE new-EXQ successor falsifier the V3-EXQ-711 confirmed autopsy routed to as the
load-bearing V3-closure build. NEW EXQ number, NEW mechanism (the BOUNDED target-parity controller,
landed 2026-07-04) -- a redesign of a DIFFERENT mechanism (the controller), NOT a raw-gain-magnitude
re-letter of 711. EXPERIMENT_PURPOSE=evidence (governance falsifier, NOT a diagnostic). Clears a
re-derive brake by testing a NEWLY-BUILT substrate (the 711 brake explicitly REFUSES a same-claim
raw-gain re-queue; a controller redesign is the allowed BUILD path -- see queue note).

THE QUESTION (MECH-439 x ARC-108 x ARC-110)
-------------------------------------------
V3-EXQ-709 built the LEARNED [3,3] cross-loop matrix W_cross = I + M_cross fully live but the limbic
loop reached the motor loop's EFFECTIVE COLUMN WEIGHT w_eff[j]=sum_i gain_i*W_cross[i,j] on only 1/4
divergent seeds (ascending coupling ~0.03, too WEAK). V3-EXQ-711 then tried a RAW-SCALAR ascending
gain (20x-forward x 5x-plasticity) and it RAN AWAY: the two unbounded multiplicative gains compounded
through the positive-feedback plastic M_cross loop -> M_cross range peak 4897.8, w_eff[limbic] 10-2274x
w_eff[motor] across the 3 divergent seeds -- a limbic MONOPOLY that merely replaces the F/motor-pinning,
not a fair parity win; committed-class entropy FELL below baseline on 2/3 divergent seeds. The bare
w_eff[limbic] >= w_eff[motor] win-gate could not tell a 1.01x fair win from a 2274x blow-up, so the
conversion was tested under a DEGENERATE (saturated) arbitration and self-routed a decisive FAIL that
the confirmed autopsy REJECTED (non_contributory). The 709->711 pattern: the raw scalar has NO stable
parity regime -- sub-threshold or runaway; the mechanism was MISSING A CONTROLLER.

The substrate landed 2026-07-04 (ree-v3 main; learned_cross_loop_arbitration.md "Sub-addendum:
bounded parity controller") adds a BOUNDED ascending gain -- a target-PARITY controller
(use_ascending_parity_controller): a per-step forward gain SOLVED so w_eff[limbic] is LIFTED toward
but HARD-CAPPED at PARITY_CEILING_RATIO x w_eff[motor] (motor column gain-invariant -> the fixed
parity reference), + a clamp on the ascending M_cross maturation (anti-windup). It bounds the
w_eff[limbic]/w_eff[motor] RATIO -> a FAIR within-eligible reorder, never a monopoly. Haber 2000's
striato-nigro-striatal spiral is a graded, bounded, parity-restoring modulation; the raw 711 scalar
had the symbol without that bounding dependency.

Does the BOUNDED controller let a non-motor (limbic) loop reach a FAIR PARITY win (in-band, not a
blow-up) on a strict-majority of divergent seeds -- and, once it can, does that CONVERT committed-
action diversity where the un-gained learned arbitration (709) could not?

THE 2 ARMS (both learned-arbitration loop arms; the ONLY swept factor is use_ascending_parity_controller)
--------------------------------------------------------------------------------------------------------
  A_ASCENDING_OFF : loop segregation ON + learned cross-loop arbitration ON, bounded controller OFF.
                    == the V3-EXQ-709 ceiling baseline (limbic reached motor weight on ~1/4 divergent).
  A_ASCENDING_ON  : identical + use_ascending_parity_controller ON (forward gain capped at the parity
                    ceiling + clamped ascending maturation). The mechanism under test. The raw scalar
                    (use_ascending_spiral_gain) is OFF on BOTH arms.

BOTH arms carry the SAME landed envelope + finer-channel gating + learned settling +
per-named-channel routing + limbic INPUT modules + LEARNED cross-loop arbitration as a MATCHED
CONSTANT. The ONLY difference is whether the bounded controller is on. Inert / OFF -> the arms are
BIT-IDENTICAL (at init M_cross==0 -> W_cross==I regardless).

6 seeds. PRIMARY DV = committed-action-class entropy (nats), measured over P2.
claim_ids = [MECH-439, ARC-108, ARC-110]. experiment_purpose = evidence (governance falsifier).

PRE-REGISTERED OUTCOME MAP (decisive either way)
------------------------------------------------
  LOAD-BEARING PRECONDITION (the autopsy's target, SATURATION-GUARDED): on A_ASCENDING_ON, the limbic
  loop reaches a FAIR PARITY WIN -- w_eff[motor] <= w_eff[limbic] <= PARITY_BAND_HI x w_eff[motor]
  (in-band) AND no blow-up (w_eff/M_cross under the saturation ceiling) -- on a STRICT-MAJORITY
  (>= 3/4) of DIVERGENT seeds. This is the exact 711 repair: the raw scalar met a BARE w_eff >= via a
  monopoly, which the band+ceiling now reject.

  SATURATION GUARD: if the controller BLOWS UP on ANY divergent seed (w_eff[limbic]/w_eff[motor] or
  M_cross range past the ceiling), the run self-routes substrate_not_ready_requeue (non_contributory,
  non_degenerate=False) -- NEVER a decisive FAIL/weakens. This is the exact hole the 711 win-gate
  lacked. With a correctly tuned controller it never fires.

  If the precondition is UNMET (the controller cannot deliver a fair parity win on >= 3/4 divergent
  seeds -- lift too weak), the run self-routes substrate_not_ready_requeue -- NEVER a false weakens.

  If the precondition IS met (fair parity win, no saturation), C1 becomes validly evaluable:
  C1 (PASS): A_ASCENDING_ON committed-class entropy STRICT-ABOVE A_ASCENDING_OFF + margin on a
  strict-majority (>= 2/3) of DIVERGENT seeds.

  PASS (precondition met + C1) -> the bounded controller LIFTS the F-dominance conversion ceiling
    under a FAIR parity arbitration:
      MECH-439: weakens  (the ceiling is NOT immutable -- a bounded parity lift converts it)
      ARC-108 : supports (the learned DA-gated gating, at fair parity, converts)
      ARC-110 : supports (segregated loops become load-bearing once a non-motor loop wins fairly)
      overall : mixed (MECH-439 opposes ARC-108/ARC-110 by construction)

  FAIL (precondition met, C1 fails) -> DECISIVE and now VALID (the win was a fair BOUNDED parity win,
    not the 711 saturated monopoly): even at fair competitive parity, diversity does NOT lift:
      MECH-439: supports (the ceiling is INTRINSIC -- winning fairly does not convert to diversity)
      ARC-108 : weakens  (learned gating at fair parity does not convert)
      ARC-110 : weakens  (the loop route at fair parity does not convert)
      overall : mixed

Plus the 707b/709 substrate-liveness preconditions (matched, both arms): enough divergent seeds
(GAP-A pool spread), loops carry live cross-loop variance, named_channel_routing_live on the ON arm,
the learned M_cross weights moved off init on the ON arm, learning engaged (finer channels
dissociable + delta_t non-flat), CRF matured. Any unmet -> substrate_not_ready_requeue, NEVER a weakens.

Phased training is kept (P0 e2-train -> P1 frozen-encoder bias-head REINFORCE -> P2 e2+bias frozen,
gating + loops + M_cross KEEP adapting) matched to 709/711. MECH-094: the M_cross three-factor update
(and its bounded maturation + clamp) is waking-only (a simulation/replay tick arms no trace, writes
no M_cross). Safety inherited: the arbitration runs strictly within the F + MECH-448/449 Go/No-Go
eligible set (the controller reorders within-eligible candidates but can never re-admit a suppressed one).

See REE_assembly/docs/architecture/learned_cross_loop_arbitration.md ("Sub-addendum: bounded parity controller"),
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-711_2026-07-04.md (the escalation source),
    ree-v3/CLAUDE.md "ARC-110 x ARC-108: BOUNDED ascending-spiral gain -- target-PARITY controller" (build entry),
    ree-v3/ree_core/predictors/e3_selector.py (_parity_forward_gain + the W_cross assembly + the M_cross clamp),
    experiments/v3_exq_711_ascending_spiral_gain_validation.py (the raw-scalar predecessor this controller repairs).
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
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_713_bounded_parity_controller_validation"
QUEUE_ID = "V3-EXQ-713"
SUPERSEDES = None   # tests a DIFFERENT mechanism than 711 (the bounded CONTROLLER, not the raw scalar) -- not a supersession
BACKLOG_ID = None   # no proposal; routed by failure_autopsy_V3-EXQ-711_2026-07-04 (implement-substrate; re-derive brake RELEASED: controller built)
CLAIM_IDS: List[str] = ["MECH-439", "ARC-108", "ARC-110"]
EXPERIMENT_PURPOSE = "evidence"   # governance-evidence falsifier

# softplus-unity init for w_chan_finer (softplus(_FCG_W_INIT) == 1.0).
_FCG_W_INIT = math.log(math.e - 1.0)

# CRF-gate calibration levers (matured CRF stack; ported verbatim from 707b/700c,
# matched on all arms -- the differentiated conversion source).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# ----- Acceptance thresholds (pre-registered) -----
# C1 conversion: strict-above margin on committed-class entropy (nats).
CONVERSION_MARGIN = 0.05

# ----- Per-seed-divergent gating (707b-style) -----
MIN_DIVERGENT_SEEDS = 3          # of 6: fewer divergent seeds => substrate_not_ready_requeue
DIVERGENT_PASS_FRACTION = 0.5    # strict-majority-ish gate within the divergent seeds
MIN_SEEDS_FOR_PASS = 2           # absolute floor of divergent seeds clearing a criterion

# ARC-110 non-degeneracy: loops must carry LIVE cross-loop variance on the loop arms.
LOOP_CROSS_VARIANCE_FRAC_FLOOR = 0.05
LOOP_PREF_RANGE_FLOOR = 1e-6
# 707b C2-release limbic-routing non-degeneracy: on the LEARNED arm at least one LIMBIC channel
# (ofc/liking/vigour) must reach the arbitration carrying routed per-candidate range above this
# floor (peak over P2 ticks) -- else the limbic loop is inert and the arbitration cannot learn to
# route through it (the MECH-191 phasic gap).
LIMBIC_ROUTED_RANGE_FLOOR = 1e-3
LIMBIC_NAMED_CHANNELS = ("ofc", "liking", "vigour")

# ----- ARC-108 x ARC-110 learned-cross-loop NON-VACUITY thresholds (the mechanism gate) -----
# (W) the learned M_cross weights MOVED off their zero init (loop_cross_loop_m_range +
#     post_action_update clg_m_cross_range). At init M_cross==0 -> range 0 -> LEARNED is
#     bit-identical to STATIC; a range above this floor == the arbitration is genuinely learning.
M_CROSS_RANGE_FLOOR = 1e-6
# (L) the limbic loop CAN win: loop_cross_loop_limbic_ge_motor reachable (w_eff[limbic] reaches
#     w_eff[motor]) on >= 1 tick. Recorded strict-exceed count is the stronger signal.

# C1(a) readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# Non-vacuity (b): GAP-A consumed-summary divergence (649 statistic + 643a ceiling).
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# Non-vacuity: delta_t carries cross-tick variance (outcome variance to learn from).
DELTA_T_STD_FLOOR = 1e-4
# Non-vacuity: the finer w_chan_finer entries MOVED + are DISSOCIABLE (cross-channel range).
W_CHAN_FINER_RANGE_FLOOR = 1e-4

# CRF maturity readiness (matched constant; the differentiated source must be present).
CRF_MIN_MINTED = 2
CRF_N_ACTIVE_FLOOR = 1
CRF_FRAC_ACTIVE_FLOOR = 0.30

SEEDS = [42, 43, 44, 45, 46, 47]
P0_WARMUP_EPISODES = 100
P1_BIAS_TRAIN_EPISODES = 50
P2_MEASUREMENT_EPISODES = 100
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 4
DRY_RUN_STEPS = 30

# ----- ARC-108 JOB-1 learned-gating knobs (substrate defaults; matched on both arms) -----
LCG_ETA = 0.01
LCG_ELIG_DECAY = 0.9
LCG_VALUE_BASELINE_BETA = 0.05
LCG_ASYM_POTENTIATION = 1.0
LCG_ASYM_DEPRESSION = 0.5

# ----- ARC-108 x ARC-110 learned cross-loop arbitration (the SWEPT factor's learning rate) -----
LEARNED_CROSS_LOOP_ETA = 0.01   # M_cross three-factor learning rate (substrate default)

# ----- BOUNDED ascending-spiral gain -- target-PARITY controller (the SWEPT factor here) -----
# The V3-EXQ-711 runaway repair: instead of the raw scalar (20x-fwd x 5x-plasticity) that ran
# w_eff[limbic] to 10-2274x w_eff[motor] (a limbic MONOPOLY), the ON arm uses the bounded
# controller (use_ascending_parity_controller): a per-step forward gain SOLVED to hold
# w_eff[limbic] toward but under PARITY_CEILING_RATIO * w_eff[motor], + a clamp on the ascending
# M_cross entries. OFF / inert -> bit-identical to the un-gained 709 learned arm. The magnitudes
# lift the ~0.03 709 coupling to a FAIR parity, never a monopoly; safety inherited (reorder only).
PARITY_FORWARD_GAIN = 8.0       # ascending lift strength g_raw (ceiling caps it)
PARITY_CEILING_RATIO = 1.25     # cap w_eff[limbic] <= ratio * w_eff[motor] (the parity band upper edge)
PARITY_PLASTICITY_GAIN = 2.0    # bounded ascending maturation rate (paired with the clamp)
PARITY_M_CROSS_CLAMP = 0.5      # |ascending M_cross| bound post-update (anti-windup)

# ----- SATURATION GUARD thresholds (the load-bearing 711 repair on the win-gate) -----
# A valid PARITY win = a tick with w_eff[motor] <= w_eff[limbic] <= PARITY_BAND_HI * w_eff[motor]
# AND no blow-up. A blow-up (w_eff[limbic] PEAK > SATURATION_W_EFF_CEILING * w_eff[motor] peak, or
# clg_m_range PEAK > SATURATION_M_CROSS_CEILING) means the arbitration SATURATED (a monopoly, not a
# fair win) -> the run self-routes substrate_not_ready_requeue, NEVER a decisive FAIL/weakens. With
# a working controller these ceilings are never breached (ceiling_ratio 1.25 caps w_eff by
# construction; the clamp bounds M_cross); a breach signals a controller failure -> requeue.
PARITY_BAND_HI = 1.25                 # w_eff[limbic]/w_eff[motor] upper edge of a valid parity win
SATURATION_W_EFF_CEILING = 3.0        # w_eff[limbic]/w_eff[motor] peak above this = blow-up
SATURATION_M_CROSS_CEILING = 3.0      # clg_m_range peak above this = blow-up
_SAT_EPS = 1e-6                       # floor for the w_eff[motor] denominator

# The autopsy target: a WORKING impl must let a non-motor loop reach motor effective-weight PARITY
# (a fair BAND win, not a blow-up) on a STRICT-MAJORITY (>= 3/4) of divergent seeds so C1 can be
# validly evaluated. This is the limbic_loop_can_win gate, now saturation-guarded.
LIMBIC_WIN_PASS_FRACTION = 0.75

# ----- MECH-450 settling (ON on both arms; the within-loop settling each loop runs) -----
LEARNED_SETTLING_ROUNDS = 3
LEARNED_SETTLING_TEMPERATURE = 1.0
LEARNED_SETTLING_ETA = 0.01
LEARNED_SETTLING_ELIG_DECAY = 0.9

# ----- ARC-110 loop-segregation knobs (matched on both loop arms) -----
LOOP_SEGREGATION_NORMALIZE = "zscore"

# SD-056 online e2 training (mirror 707b/700c).
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

# P1 bias-head REINFORCE training (mirror 707b/700c).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

# Matched-stack lever constants (identical on both arms; the landed envelope).
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


# IDENTICAL env to 707b/700c (the GAP-A reef-bipartite foraging bank).
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


# The 2 arms. BOTH are loop arms carrying the SAME landed envelope + finer gating + learned
# settling + per-named-channel routing + limbic input modules as a MATCHED CONSTANT; the ONLY
# swept factor is learn_cross (use_learned_cross_loop_arbitration).
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "A_ASCENDING_OFF",
        "label": "arc108_x_arc110_learned_cross_loop_matrix_bounded_parity_controller_OFF_709_ceiling_baseline",
        # BOTH arms run the LEARNED cross-loop matrix; the swept factor is the bounded controller.
        "learn_cross": True,
        "parity": False,
    },
    {
        "arm_id": "A_ASCENDING_ON",
        "label": "arc110_x_arc108_bounded_parity_controller_ON_forward_ceiling_plus_clamped_maturation",
        "learn_cross": True,
        "parity": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Matched-stack agent. The landed arithmetic envelope + diversity stack + finer-channel
    gating + learned settling + ARC-110 loop segregation (+ ARC-109 D1/D2 + MECH-452 loop-local
    traces) + per-named-channel routing + the limbic INPUT modules + the LEARNED cross-loop
    arbitration matrix are MATCHED CONSTANTS on BOTH arms. The ONLY swept factor is
    use_ascending_spiral_gain: A_ASCENDING_OFF runs the un-gained 709 learned combine (W_cross =
    I + M_cross); A_ASCENDING_ON scales the ascending (upper-triangular) M_cross entries in the
    forward W_cross and the maturation update, so a non-motor loop can reach motor effective weight."""
    learn_cross = bool(arm["learn_cross"])
    parity = bool(arm.get("parity", False))
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
        # --- Matched stack (identical on both arms) ---
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        candidate_summary_source="e2_world_forward",
        use_modulatory_selection_authority=USE_MODULATORY_SELECTION_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_channel_routing=USE_MODULATORY_CHANNEL_ROUTING,
        modulatory_channel_route_source=MODULATORY_CHANNEL_ROUTE_SOURCE,
        modulatory_channel_route_weight=MODULATORY_CHANNEL_ROUTE_WEIGHT,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        use_modulatory_shortlist_then_modulate=USE_MODULATORY_SHORTLIST_THEN_MODULATE,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        use_f_eligibility_demotion=USE_F_ELIGIBILITY_DEMOTION,
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        use_f_eligibility_adaptive_floor=USE_F_ELIGIBILITY_ADAPTIVE_FLOOR,
        f_eligibility_adaptive_mean_factor=F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
        use_dacc=USE_DACC,
        use_go_nogo_constitution=USE_GO_NOGO_CONSTITUTION,
        gng_perseveration_floor=GNG_PERSEVERATION_FLOOR,
        gng_safety_floor=GNG_SAFETY_FLOOR,
        gng_protect_min_eligible=GNG_PROTECT_MIN_ELIGIBLE,
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=MECH341_ENTROPY_BIAS_SCALE,
        e3_diversity_stratified_within_class_temperature=None,
        use_noise_floor=False,
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        use_gated_policy=True,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
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
        # --- MECH-451 FINER separately-learnable channels (ON on both arms; loops partition
        # these into motor/associative/limbic). ---
        use_finer_channel_gating=True,
        use_learned_channel_gating=False,
        learned_channel_gating_eta=LCG_ETA,
        learned_channel_gating_elig_decay=LCG_ELIG_DECAY,
        learned_channel_value_baseline_beta=LCG_VALUE_BASELINE_BETA,
        learned_channel_asym_potentiation=LCG_ASYM_POTENTIATION,
        learned_channel_asym_depression=LCG_ASYM_DEPRESSION,
        learned_channel_rpe_mode="signed",
        # --- MECH-450 recurrent settling: ON on both arms (the within-loop settling each
        # segregated loop runs). ---
        use_learned_settling_step=True,
        learned_settling_rounds=LEARNED_SETTLING_ROUNDS,
        learned_settling_temperature=LEARNED_SETTLING_TEMPERATURE,
        learned_settling_eta=LEARNED_SETTLING_ETA,
        learned_settling_elig_decay=LEARNED_SETTLING_ELIG_DECAY,
        # --- ARC-110 parallel segregated loops (ON on BOTH arms) + coupled ARC-109 / MECH-452.
        # Both arms are loop arms; loop segregation is a MATCHED CONSTANT here (unlike 707b, where
        # it was the swept factor). ---
        use_loop_segregation=True,
        loop_segregation_channel_map={},
        loop_segregation_normalize=LOOP_SEGREGATION_NORMALIZE,
        loop_segregation_noise_on=False,
        use_d1_d2_population_split=True,
        use_loop_local_eligibility_traces=True,
        # --- ARC-110 C2 RELEASE (707b): per-named-channel range-preserving routing into the
        # segregated loops, so the named limbic channels carry per-candidate range. ON on BOTH
        # arms (matched constant) so the limbic loop is live regardless of learn_cross. ---
        use_named_channel_routing=True,
        # --- Limbic-loop INPUT modules: MATCHED CONSTANT on both arms. ---
        use_ofc_analog=True,
        use_mech295_liking_bridge=True,
        use_tonic_vigor=True,
        # --- ARC-108 x ARC-110 LEARNED (dopamine-gated) cross-loop arbitration -- THE SWEPT
        # MATCHED CONSTANT here (unlike 709 where it was the swept factor): ON on BOTH arms, so
        # the [3,3] W_cross = I + M_cross matrix learns on both. The ascending gain (below) is the
        # 711 swept factor. ---
        use_learned_cross_loop_arbitration=learn_cross,
        learned_cross_loop_eta=LEARNED_CROSS_LOOP_ETA,
        # --- BOUNDED ascending-spiral gain -- target-PARITY controller -- THE SWEPT FACTOR here.
        # OFF -> the un-gained 709 learned arm (loop-effective-weight ceiling); ON -> the bounded
        # controller: a forward gain SOLVED to hold w_eff[limbic] under PARITY_CEILING_RATIO x
        # w_eff[motor] + a clamped ascending M_cross maturation. The raw scalar path stays OFF on
        # BOTH arms (this falsifier tests the CONTROLLER, not the 711 raw scalar). Inert / OFF ->
        # the two arms are bit-identical. ---
        use_ascending_spiral_gain=False,
        use_ascending_parity_controller=parity,
        loop_segregation_parity_forward_gain=(PARITY_FORWARD_GAIN if parity else 1.0),
        loop_segregation_parity_ceiling_ratio=(PARITY_CEILING_RATIO if parity else 0.0),
        loop_segregation_parity_plasticity_gain=(PARITY_PLASTICITY_GAIN if parity else 1.0),
        loop_segregation_m_cross_clamp=(PARITY_M_CROSS_CLAMP if parity else 0.0),
    )
    return REEAgent(cfg)


def _arm_config_slice(
    arm: Dict[str, Any],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Declared fingerprint slice: ONLY what an arm's computation reads -- the swept
    learn_cross flag + the matched loop/envelope config every arm runs + the env + the
    schedule. NEVER acceptance thresholds. Both arms ride the just-built learned-cross-loop /
    loop-segregation substrate, which is in ACTIVE FLUX for this lineage (the conversion-ceiling
    campaign is iterating e3_selector.py) -- so neither arm is minted as a reusable baseline."""
    _parity = bool(arm.get("parity", False))
    return {
        "arm_id": arm["arm_id"],
        "learn_cross": bool(arm["learn_cross"]),
        "learned_cross_loop_eta": float(LEARNED_CROSS_LOOP_ETA),
        "use_ascending_parity_controller": _parity,
        "loop_segregation_parity_forward_gain": (
            float(PARITY_FORWARD_GAIN) if _parity else 1.0
        ),
        "loop_segregation_parity_ceiling_ratio": (
            float(PARITY_CEILING_RATIO) if _parity else 0.0
        ),
        "loop_segregation_parity_plasticity_gain": (
            float(PARITY_PLASTICITY_GAIN) if _parity else 1.0
        ),
        "loop_segregation_m_cross_clamp": (
            float(PARITY_M_CROSS_CLAMP) if _parity else 0.0
        ),
        "use_loop_segregation": True,
        "use_named_channel_routing": True,
        "use_ofc_analog": True,
        "use_mech295_liking_bridge": True,
        "use_tonic_vigor": True,
        "use_d1_d2_population_split": True,
        "use_loop_local_eligibility_traces": True,
        "use_learned_settling_step": True,
        "use_finer_channel_gating": True,
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
        "loop_segregation_normalize": str(LOOP_SEGREGATION_NORMALIZE),
        "env_kwargs": dict(ENV_KWARGS),
        "sd056_weight": float(SD056_WEIGHT),
        "lr_lpfc_bias": float(LR_LPFC_BIAS),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
    }


# ---------------------------------------------------------------------------
# SD-056 online e2 training (verbatim from 707b)
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
# P1 bias-head REINFORCE training (verbatim from 707b)
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
    learn_cross = bool(arm["learn_cross"])

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
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p2_ticks = 0
    n_p0_contrastive_steps = 0
    n_p1_bias_updates = 0

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    # PRIMARY DV: committed first-action class counts over P2.
    committed_class_counts: Dict[int, int] = {}
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0

    # CRF maturity readiness (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_minted_total_last = 0

    # ----- MECH-451 finer-channel learning diagnostics (accumulated all phases; matched constant) -----
    fcg_delta_ts: List[float] = []
    fcg_w_chan_finer_range_max = 0.0
    fcg_w_chan_finer_std_max = 0.0

    # ----- ARC-110 loop diagnostics (P2 select ticks; the non-degeneracy net) -----
    loop_active_ticks = 0
    loop_committed_neq_motor_ticks = 0
    loop_disagree_ticks = 0
    loop_assoc_range_sum = 0.0
    loop_limbic_range_sum = 0.0
    n_loop_diag_ticks = 0
    # 707b C2-release per-named-channel routing (peak over P2 ticks).
    loop_named_routing_active_ticks = 0
    loop_limbic_routed_range_peak = 0.0
    loop_named_routed_range_peaks: Dict[str, float] = {}

    # ----- ARC-108 x ARC-110 learned cross-loop diagnostics (P2 select ticks; LEARNED arm) -----
    clg_active_ticks = 0
    clg_m_range_peak = 0.0                 # peak loop_cross_loop_m_range over P2 ticks
    clg_limbic_ge_motor_ticks = 0          # ticks where w_eff[limbic] >= w_eff[motor]
    clg_w_limbic_exceeds_motor_ticks = 0   # ticks where w_eff[limbic] > w_eff[motor] (strict)
    clg_limbic_parity_band_ticks = 0       # ticks where w_motor <= w_limbic <= BAND_HI*w_motor (fair PARITY win)
    clg_saturation_ticks = 0               # ticks where w_limbic or m_range breached the saturation ceiling (blow-up)
    clg_w_limbic_eff_peak = 0.0
    clg_w_motor_eff_peak = 0.0
    clg_limbic_to_motor_peak = 0.0         # peak M_cross[motor, limbic]
    # ----- learned cross-loop UPDATE diagnostics (from post_action_update, LEARNED arm) -----
    clg_update_delta_ts: List[float] = []
    clg_update_m_cross_range_max = 0.0
    clg_update_n_updates_last = 0

    for ep in range(total_train_eps):
        is_p1 = (p1_start <= ep < p2_start)
        is_p2 = (ep >= p2_start)
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
            # ARC-110 + ARC-108 x ARC-110: read the segregated-loop + learned-cross-loop
            # diagnostics from the last e3 select (P2 only, when the loop path ran).
            if is_p2:
                diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
                if diag.get("loop_segregation_active", False):
                    n_loop_diag_ticks += 1
                    loop_active_ticks += 1
                    if diag.get("loop_committed_neq_motor_winner", False):
                        loop_committed_neq_motor_ticks += 1
                    if diag.get("loop_cross_loop_winner_disagreement", False):
                        loop_disagree_ticks += 1
                    loop_assoc_range_sum += float(diag.get("loop_assoc_pref_range", 0.0) or 0.0)
                    loop_limbic_range_sum += float(diag.get("loop_limbic_pref_range", 0.0) or 0.0)
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
                    # ARC-108 x ARC-110 learned cross-loop non-vacuity + mechanism gates.
                    # Only emitted on the LEARNED arm (loop_learned_cross_loop_active True).
                    if diag.get("loop_learned_cross_loop_active", False):
                        clg_active_ticks += 1
                        _wm = float(diag.get("loop_cross_loop_w_motor_eff", 0.0) or 0.0)
                        _wl = float(diag.get("loop_cross_loop_w_limbic_eff", 0.0) or 0.0)
                        clg_m_range_peak = max(
                            clg_m_range_peak,
                            float(diag.get("loop_cross_loop_m_range", 0.0) or 0.0),
                        )
                        clg_w_motor_eff_peak = max(clg_w_motor_eff_peak, _wm)
                        clg_w_limbic_eff_peak = max(clg_w_limbic_eff_peak, _wl)
                        clg_limbic_to_motor_peak = max(
                            clg_limbic_to_motor_peak,
                            float(diag.get("loop_cross_loop_limbic_to_motor", 0.0) or 0.0),
                        )
                        if bool(diag.get("loop_cross_loop_limbic_ge_motor", False)):
                            clg_limbic_ge_motor_ticks += 1
                        if _wl > _wm:
                            clg_w_limbic_exceeds_motor_ticks += 1
                        # SATURATION GUARD (V3-EXQ-711 repair): a fair PARITY win is a tick where
                        # w_eff[limbic] reaches motor parity but stays within the band; a blow-up
                        # (past the saturation ceiling on w_eff OR the per-tick m_range) is a
                        # monopoly, NOT a fair win.
                        _mden = _wm if abs(_wm) > _SAT_EPS else _SAT_EPS
                        _mrange_tick = float(diag.get("loop_cross_loop_m_range", 0.0) or 0.0)
                        if _wm <= _wl <= PARITY_BAND_HI * _wm:
                            clg_limbic_parity_band_ticks += 1
                        if (_wl > SATURATION_W_EFF_CEILING * _mden) or (
                            _mrange_tick > SATURATION_M_CROSS_CEILING
                        ):
                            clg_saturation_ticks += 1
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
            # update_residue drives e3.post_action_update -> the ARC-108 (finer w_chan_finer) AND
            # the ARC-108 x ARC-110 learned cross-loop (M_cross) three-factor updates fire here on
            # EVERY waking tick (all phases). On the STATIC arm the M_cross path is inert (flag OFF).
            with torch.no_grad():
                resid_metrics = agent.update_residue(
                    harm_signal=float(_harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )
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
            # ARC-108 x ARC-110 learned cross-loop UPDATE diagnostics (post_action_update; the
            # (W) weights-moved gate reads these). Only present on the LEARNED arm when a waking
            # trace was pending (e3_clg_ prefix, via agent metrics.update({f"e3_{k}": v ...})).
            cdt = resid_metrics.get("e3_clg_delta_t")
            if cdt is not None:
                clg_update_delta_ts.append(float(cdt.item()))
            cmr = resid_metrics.get("e3_clg_m_cross_range")
            if cmr is not None:
                clg_update_m_cross_range_max = max(
                    clg_update_m_cross_range_max, float(cmr.item())
                )
            cnu = resid_metrics.get("e3_clg_n_updates")
            if cnu is not None:
                clg_update_n_updates_last = int(cnu.item())

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

    fcg_delta_t_std = float(statistics.pstdev(fcg_delta_ts)) if len(fcg_delta_ts) >= 2 else 0.0
    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )
    fcg_moved = bool(fcg_w_chan_finer_range_max > W_CHAN_FINER_RANGE_FLOOR)
    fcg_delta_nonflat = bool(fcg_delta_t_std > DELTA_T_STD_FLOOR)

    # ----- ARC-110 loop-segregation per-seed aggregation (over P2 select ticks) -----
    loop_n = max(n_loop_diag_ticks, 1)
    loop_frac_committed_neq_motor = float(loop_committed_neq_motor_ticks / loop_n) if n_loop_diag_ticks else 0.0
    loop_frac_disagree = float(loop_disagree_ticks / loop_n) if n_loop_diag_ticks else 0.0
    loop_assoc_range_mean = float(loop_assoc_range_sum / loop_n) if n_loop_diag_ticks else 0.0
    loop_limbic_range_mean = float(loop_limbic_range_sum / loop_n) if n_loop_diag_ticks else 0.0
    seed_loop_cross_variance = bool(
        n_loop_diag_ticks > 0
        and (loop_frac_committed_neq_motor > LOOP_CROSS_VARIANCE_FRAC_FLOOR
             or loop_frac_disagree > LOOP_CROSS_VARIANCE_FRAC_FLOOR)
        and (loop_assoc_range_mean > LOOP_PREF_RANGE_FLOOR
             or loop_limbic_range_mean > LOOP_PREF_RANGE_FLOOR)
    )
    limbic_routed_peaks = [
        loop_named_routed_range_peaks.get(nm, 0.0) for nm in LIMBIC_NAMED_CHANNELS
    ]
    loop_limbic_routed_range_max = float(max(limbic_routed_peaks)) if limbic_routed_peaks else 0.0
    seed_named_channel_routing_live = bool(
        loop_named_routing_active_ticks > 0
        and loop_limbic_routed_range_max > LIMBIC_ROUTED_RANGE_FLOOR
    )

    # ----- ARC-108 x ARC-110 learned cross-loop per-seed non-vacuity (LEARNED arm only) -----
    clg_update_delta_std = (
        float(statistics.pstdev(clg_update_delta_ts)) if len(clg_update_delta_ts) >= 2 else 0.0
    )
    # (W) the learned weights MOVED off init: M_cross range cleared the floor (from BOTH the
    # per-tick score diagnostic AND the post_action_update update diagnostic) AND updates fired.
    seed_clg_weights_moved = bool(
        learn_cross
        and clg_m_range_peak > M_CROSS_RANGE_FLOOR
        and clg_update_m_cross_range_max > M_CROSS_RANGE_FLOOR
        and clg_update_n_updates_last > 0
    )
    # SATURATION GUARD (V3-EXQ-711 repair): a seed SATURATED if w_eff[limbic] peak breached the
    # ceiling (blow-up ratio to w_eff[motor] peak) OR the M_cross range peak breached its ceiling.
    _wm_den = clg_w_motor_eff_peak if abs(clg_w_motor_eff_peak) > _SAT_EPS else _SAT_EPS
    seed_clg_saturated = bool(
        learn_cross and (
            clg_w_limbic_eff_peak > SATURATION_W_EFF_CEILING * _wm_den
            or clg_m_range_peak > SATURATION_M_CROSS_CEILING
            or clg_saturation_ticks > 0
        )
    )
    # (L) the limbic loop reaches a FAIR PARITY win: it entered the parity BAND (w_motor <=
    # w_limbic <= BAND_HI*w_motor) on >= 1 tick AND did NOT blow up. A saturated seed is NOT a
    # valid win (the 711 monopoly the raw scalar produced) -- it self-routes requeue, not a FAIL.
    seed_clg_limbic_parity_win = bool(
        learn_cross and clg_limbic_parity_band_ticks > 0 and not seed_clg_saturated
    )
    # Retained (un-guarded) reach signal for audit/back-compat.
    seed_clg_limbic_can_win = bool(learn_cross and clg_limbic_ge_motor_ticks > 0)

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "learn_cross": learn_cross,
        # ----- ARC-110 loop diagnostics -----
        "loop_active_ticks": int(loop_active_ticks),
        "loop_frac_committed_neq_motor": round(loop_frac_committed_neq_motor, 6),
        "loop_frac_disagree": round(loop_frac_disagree, 6),
        "loop_assoc_pref_range": round(loop_assoc_range_mean, 6),
        "loop_limbic_pref_range": round(loop_limbic_range_mean, 6),
        "loop_cross_variance": seed_loop_cross_variance,
        "loop_named_routing_active_ticks": int(loop_named_routing_active_ticks),
        "loop_limbic_routed_range_max": round(loop_limbic_routed_range_max, 6),
        "loop_named_routed_range_peaks": {
            str(k): round(float(v), 6) for k, v in sorted(loop_named_routed_range_peaks.items())
        },
        "named_channel_routing_live": seed_named_channel_routing_live,
        # ----- ARC-108 x ARC-110 learned cross-loop diagnostics -----
        "clg_active_ticks": int(clg_active_ticks),
        "clg_m_range_peak": round(clg_m_range_peak, 8),
        "clg_update_m_cross_range_max": round(clg_update_m_cross_range_max, 8),
        "clg_update_n_updates": int(clg_update_n_updates_last),
        "clg_update_delta_std": round(clg_update_delta_std, 8),
        "clg_limbic_ge_motor_ticks": int(clg_limbic_ge_motor_ticks),
        "clg_w_limbic_exceeds_motor_ticks": int(clg_w_limbic_exceeds_motor_ticks),
        "clg_limbic_parity_band_ticks": int(clg_limbic_parity_band_ticks),
        "clg_saturation_ticks": int(clg_saturation_ticks),
        "clg_w_motor_eff_peak": round(clg_w_motor_eff_peak, 6),
        "clg_w_limbic_eff_peak": round(clg_w_limbic_eff_peak, 6),
        "clg_limbic_to_motor_peak": round(clg_limbic_to_motor_peak, 6),
        "clg_weights_moved": seed_clg_weights_moved,
        "clg_limbic_can_win": seed_clg_limbic_can_win,
        "clg_limbic_parity_win": seed_clg_limbic_parity_win,
        "clg_saturated": seed_clg_saturated,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
        # ----- PRIMARY DV (committed-class entropy) -----
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        # ----- Readiness / non-vacuity -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "crf_differentiated": crf_differentiated,
        # ----- finer-channel learning diagnostics (MECH-451; matched constant) -----
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


def _by_seed(rows: List[Dict[str, Any]], key: str) -> Dict[int, float]:
    return {int(r["seed"]): float(r[key]) for r in rows}


def _gap_by_seed(rows: List[Dict[str, Any]]) -> Dict[int, bool]:
    return {int(r["seed"]): bool(r["gapa_divergence"]) for r in rows}


def _div_pass(n_ok: int, n_div: int) -> bool:
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

    for arm in ARMS:
        print(
            f"Arm {arm['arm_id']} ({arm['label']}) learn_cross={arm['learn_cross']} "
            f"(P0={p0_episodes} ep e2-train, P1={p1_episodes} ep bias-train, "
            f"P2={p2_episodes} ep measure, steps_per_episode={steps_per_episode}, "
            f"dry_run={dry_run})",
            flush=True,
        )
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)
            row = _run_seed_arm(
                arm, s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode,
            )
            # Per-cell fingerprint. BOTH arms ride the just-built learned-cross-loop /
            # loop-segregation substrate, which is in ACTIVE FLUX for this lineage (the
            # conversion-ceiling campaign is iterating e3_selector.py), so neither is a
            # reusable baseline -- stamped reuse-ineligible with the documented reason.
            row["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice=_arm_config_slice(
                    arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                ),
                seed=s,
                script_path=script_path,
                rng_fully_reset=True,
                extra_ineligible_reasons=[
                    "learned_cross_loop_arbitration_substrate_in_active_flux_conversion_ceiling_campaign",
                ],
            )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    static_rows = _arm_rows(arm_results, "A_ASCENDING_OFF")
    learned_rows = _arm_rows(arm_results, "A_ASCENDING_ON")
    all_rows = static_rows + learned_rows

    def _maj(rows: List[Dict[str, Any]], pred) -> bool:
        return sum(1 for r in rows if pred(r)) >= MIN_SEEDS_FOR_PASS

    static_ent = _by_seed(static_rows, "committed_class_entropy_nats")
    learned_ent = _by_seed(learned_rows, "committed_class_entropy_nats")
    static_gap = _gap_by_seed(static_rows)
    learned_gap = _gap_by_seed(learned_rows)

    # ----- Per-seed-divergent gating: seeds whose pool is divergent on BOTH arms. -----
    primary_div = [
        s for s in sorted(set(static_gap) & set(learned_gap))
        if static_gap.get(s) and learned_gap.get(s)
    ]
    n_primary_div = len(primary_div)
    enough_divergent = n_primary_div >= MIN_DIVERGENT_SEEDS

    # ----- Precondition (ARC-110 non-degeneracy): loops carry LIVE cross-loop variance on BOTH
    # loop arms (a non-motor loop flipped the within-eligible winner / loops disagreed, AND a
    # non-motor loop carries pref range > 0) on a majority of seeds. -----
    loop_cross_variance_ok = bool(
        _maj(static_rows, lambda r: r.get("loop_cross_variance", False))
        and _maj(learned_rows, lambda r: r.get("loop_cross_variance", False))
    )

    # ----- Precondition (707b C2 release): per-NAMED-channel routing is LIVE on the LEARNED arm
    # (at least one LIMBIC channel carries routed per-candidate range > floor), on a strict-
    # majority of DIVERGENT seeds. Else the limbic loop is inert and the arbitration cannot learn
    # to route through it (the MECH-191 phasic gap). -----
    named_routing_live_div = [
        s for s in primary_div
        if next((r for r in learned_rows if int(r["seed"]) == s), {}).get("named_channel_routing_live", False)
    ]
    named_channel_routing_live = bool(enough_divergent and _div_pass(len(named_routing_live_div), n_primary_div))
    learned_limbic_routed_range_max = float(
        max([r.get("loop_limbic_routed_range_max", 0.0) for r in learned_rows] or [0.0])
    )

    # ----- Precondition (ARC-108 x ARC-110 MECHANISM non-vacuity): the learned cross-loop
    # weights ACTUALLY MOVED off init (W), and the limbic loop CAN win (L), on the LEARNED arm,
    # on a strict-majority of DIVERGENT seeds. If M_cross never moved, LEARNED is bit-identical
    # to STATIC and a "no lift" is meaningless -> substrate_not_ready_requeue, NEVER a weakens. -----
    weights_moved_div = [
        s for s in primary_div
        if next((r for r in learned_rows if int(r["seed"]) == s), {}).get("clg_weights_moved", False)
    ]
    learned_weights_moved = bool(enough_divergent and _div_pass(len(weights_moved_div), n_primary_div))
    # SATURATION GUARD aggregate (V3-EXQ-711 repair): divergent seeds on the ON arm that BLEW UP
    # (w_eff/M_cross past the ceiling -- a monopoly, not a fair win). ANY saturated divergent seed
    # means the arbitration was degenerate on this run -> self-route substrate_not_ready_requeue,
    # NEVER a decisive FAIL/weakens (the exact hole the raw-scalar 711 win-gate lacked).
    saturated_div = [
        s for s in primary_div
        if next((r for r in learned_rows if int(r["seed"]) == s), {}).get("clg_saturated", False)
    ]
    saturation_detected = bool(len(saturated_div) > 0)
    limbic_can_win_div = [
        s for s in primary_div
        if next((r for r in learned_rows if int(r["seed"]) == s), {}).get("clg_limbic_parity_win", False)
    ]
    # The LOAD-BEARING gate: on the ON arm the limbic loop must reach motor effective-weight PARITY
    # (a fair BAND win, NOT a blow-up) on a STRICT-MAJORITY (>= 3/4) of divergent seeds -- the
    # autopsy's target, tighter than the generic _div_pass. Unmet -> substrate_not_ready_requeue
    # (the lift was insufficient), NEVER a weakens.
    def _win_pass(n_ok: int, n_div: int) -> bool:
        if n_div < MIN_DIVERGENT_SEEDS:
            return False
        needed = max(MIN_SEEDS_FOR_PASS, int(math.ceil(LIMBIC_WIN_PASS_FRACTION * n_div)))
        return n_ok >= needed
    learned_limbic_can_win = bool(enough_divergent and _win_pass(len(limbic_can_win_div), n_primary_div))
    learned_m_range_peak_max = float(max([r.get("clg_m_range_peak", 0.0) for r in learned_rows] or [0.0]))
    learned_n_updates_max = int(max([r.get("clg_update_n_updates", 0) for r in learned_rows] or [0]))
    learned_w_limbic_exceeds_motor_total = int(
        sum(r.get("clg_w_limbic_exceeds_motor_ticks", 0) for r in learned_rows)
    )

    # ----- Precondition: learning engaged on BOTH arms (finer channels + delta_t) -----
    fcg_moved_ok = bool(
        _maj(static_rows, lambda r: r.get("fcg_moved", False))
        and _maj(learned_rows, lambda r: r.get("fcg_moved", False))
    )
    fcg_delta_nonflat_ok = bool(
        _maj(static_rows, lambda r: r.get("fcg_delta_nonflat", False))
        and _maj(learned_rows, lambda r: r.get("fcg_delta_nonflat", False))
    )
    # Per-leg COUNTS behind the _maj calls above, reduced across arms by MIN. Each leg is
    # `_maj(static) and _maj(learned)`, i.e. TWO counts against the SAME MIN_SEEDS_FOR_PASS
    # threshold, so `min(n_static, n_learned) >= MIN_SEEDS_FOR_PASS` reproduces the
    # conjunction exactly from a single (measured, threshold) pair. The two legs
    # (moved / delta-nonflat) are DIFFERENT statistics and are therefore declared as two
    # separate recomputable preconditions -- the single entry that used to carry both had
    # `met = fcg_moved_ok and fcg_delta_nonflat_ok`, which no single pair can reproduce.
    n_fcg_moved_min_arm = min(
        sum(1 for r in static_rows if r.get("fcg_moved", False)),
        sum(1 for r in learned_rows if r.get("fcg_moved", False)),
    )
    n_fcg_delta_nonflat_min_arm = min(
        sum(1 for r in static_rows if r.get("fcg_delta_nonflat", False)),
        sum(1 for r in learned_rows if r.get("fcg_delta_nonflat", False)),
    )

    # CRF maturity (matched constant; majority of seeds on both arms).
    crf_matured = bool(
        _maj(static_rows, lambda r: r["crf_differentiated"])
        and _maj(learned_rows, lambda r: r["crf_differentiated"])
    )

    preconditions_met = bool(
        enough_divergent
        and loop_cross_variance_ok
        and named_channel_routing_live
        and learned_weights_moved      # (W) M_cross moved off init
        and learned_limbic_can_win     # (L) limbic loop can win
        and fcg_moved_ok and fcg_delta_nonflat_ok
        and crf_matured
    )

    # ----- C1 (learned-arbitration conversion): A_ASCENDING_ON committed-class entropy
    # strict-above A_ASCENDING_OFF + margin, on a strict-majority of divergent seeds. -----
    c1_seeds: List[int] = []
    for s in primary_div:
        if learned_ent.get(s, 0.0) > static_ent.get(s, 0.0) + CONVERSION_MARGIN:
            c1_seeds.append(s)
    n_c1 = len(c1_seeds)
    c1_holds = _div_pass(n_c1, n_primary_div)

    static_mean_dv = _mean([r["committed_class_entropy_nats"] for r in static_rows])
    learned_mean_dv = _mean([r["committed_class_entropy_nats"] for r in learned_rows])

    # ----- Outcome map (decisive either way) -----
    if saturation_detected:
        # SATURATION GUARD FIRES (the load-bearing 711 repair): the bounded controller BLEW UP on
        # at least one divergent seed (w_eff[limbic]/w_eff[motor] or M_cross range past the ceiling)
        # -- a limbic MONOPOLY, not a fair parity win. The conversion question was tested under a
        # DEGENERATE (saturated) arbitration, exactly as in the confirmed 711 autopsy -> the run
        # self-routes substrate_not_ready_requeue, NEVER a decisive FAIL/weakens. (With a correctly
        # tuned controller this branch never fires; a fire means the controller is mis-configured.)
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "bounded_parity_controller_saturated_substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "The bounded parity controller SATURATED on "
            + str(len(saturated_div)) + "/" + str(n_primary_div) + " divergent seed(s): "
            "w_eff[limbic] peak breached SATURATION_W_EFF_CEILING x w_eff[motor] peak, or the "
            "M_cross range peak breached SATURATION_M_CROSS_CEILING -- a limbic MONOPOLY, not a fair "
            "bounded parity win. The conversion DV was tested under a DEGENERATE (saturated) "
            "arbitration (the confirmed V3-EXQ-711 signature) -> non_contributory / "
            "substrate_not_ready_requeue. NOT a falsification: MECH-439 is NOT shown intrinsic and "
            "ARC-108/ARC-110 are NOT weakened -- the controller needs re-tuning (lower forward gain "
            "/ tighter ceiling / tighter clamp) before the conversion can be validly measured."
        )
        per_claim = {"MECH-439": "non_contributory", "ARC-108": "non_contributory", "ARC-110": "non_contributory"}
    elif not preconditions_met:
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "The bounded parity controller could NOT be validly measured against the conversion DV: "
            "a precondition is unmet (too few divergent seeds / loops carry NO live cross-loop "
            "variance / NAMED limbic channels carry NO routed per-candidate range = the MECH-191 "
            "phasic gap so the limbic loop is inert / the learned M_cross weights did NOT move off "
            "init on the ON arm / the LOAD-BEARING gate is unmet: even WITH the controller the "
            "limbic loop reached a FAIR PARITY win (band, not blow-up) on FEWER than 3/4 of "
            "divergent seeds, so the conversion question was not measured / finer channels not "
            "dissociable / delta_t flat / CRF not matured). NOT a falsification: MECH-439 is NOT "
            "shown intrinsic and ARC-108/ARC-110 are NOT weakened -- the lift needs strengthening "
            "(within the ceiling) or a further lever."
        )
        per_claim = {"MECH-439": "non_contributory", "ARC-108": "non_contributory", "ARC-110": "non_contributory"}
    elif c1_holds:
        outcome = "PASS"
        overall_direction = "mixed"
        non_degenerate = True
        degeneracy_reason = ""
        label = "bounded_parity_controller_lifts_conversion_ceiling_supports_arc108_arc110"
        # PASS: once the bounded controller lets a non-motor (limbic) loop reach a FAIR PARITY win
        # (band, not blow-up) on >= 3/4 divergent seeds, A_ASCENDING_ON converts committed-action
        # diversity strict-above the un-gained A_ASCENDING_OFF (709) baseline -> the F-dominance
        # ceiling is liftable (weakens MECH-439) via the ARC-108 learned gating at the ARC-110
        # cross-loop arbitration, under a BOUNDED (fair, non-monopoly) parity arbitration.
        per_claim = {"MECH-439": "weakens", "ARC-108": "supports", "ARC-110": "supports"}
    else:
        outcome = "FAIL"
        overall_direction = "mixed"
        non_degenerate = True
        degeneracy_reason = ""
        label = "bounded_parity_win_but_does_not_convert_ceiling_intrinsic_weakens_arc108_arc110"
        # FAIL (decisive -- and now VALID, because the win was a fair BOUNDED parity win, not the
        # 711 saturated monopoly): the controller DID lift the limbic loop to a fair parity win on
        # >= 3/4 divergent seeds (load-bearing gate MET, no saturation), M_cross moved, loops live
        # -- BUT A_ASCENDING_ON still does NOT lift committed-class entropy strict-above
        # A_ASCENDING_OFF. Even when a non-motor loop reaches FAIR competitive parity, committed-
        # action diversity does not convert -> the F-dominance conversion ceiling is INTRINSIC
        # (supports MECH-439), and the ARC-108 x ARC-110 loop-arbitration route does not deliver.
        per_claim = {"MECH-439": "supports", "ARC-108": "weakens", "ARC-110": "weakens"}

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "enough_divergent_seeds",
                "kind": "readiness",
                "description": (
                    "number of seeds whose candidate pool is DIVERGENT on BOTH arms >= "
                    "MIN_DIVERGENT_SEEDS. Per-seed-divergent gating; too few => "
                    "substrate_not_ready_requeue."
                ),
                "control": "consumed cand_world_summary pairwise spread > floor (GAP-A); per-seed",
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                # COUNT-shaped, INCLUSIVE floor: `met` is exactly
                # `n_primary_div >= MIN_DIVERGENT_SEEDS`. Declared rather than left to the
                # indexer's default so the boundary case is explicit.
                "comparator": ">=",
                "direction": "lower",
                "met": bool(enough_divergent),
            },
            {
                "name": "loops_carry_live_cross_loop_variance",
                "kind": "readiness",
                "description": (
                    "ARC-110 NON-DEGENERACY guard: on BOTH loop arms a non-motor loop FLIPS the "
                    "within-eligible winner or the loops DISAGREE on a non-trivial fraction of P2 "
                    "ticks, AND a non-motor loop carries per-loop preference RANGE > 0, on a "
                    "majority of seeds. A loop pinned to the motor winner is a vacuous split."
                ),
                "control": "loop_frac_committed_neq_motor / loop_frac_disagree + per-loop pref range (both arms)",
                # BOOLEAN INDICATOR, INCLUSIVE floor. Unlike 707/707a/707b (which reported a
                # min-fraction statistic that could not reproduce a majority COUNT), this entry
                # already reports the shipped predicate itself as a 0/1 indicator, so
                # `measured >= 1.0` reproduces `met` exactly even though the predicate is an AND
                # of two per-arm counts over a per-seed conjunction. Comparator declared so the
                # indexer does not have to fall back on its default.
                "measured": 1.0 if loop_cross_variance_ok else 0.0,
                "threshold": 1.0,
                "comparator": ">=",
                "direction": "lower",
                "observed_n_static_loop_cross_variance": int(
                    sum(1 for r in static_rows if r.get("loop_cross_variance", False))
                ),
                "observed_n_learned_loop_cross_variance": int(
                    sum(1 for r in learned_rows if r.get("loop_cross_variance", False))
                ),
                "met": bool(loop_cross_variance_ok),
            },
            {
                "name": "named_channel_routing_live",
                "kind": "readiness",
                "description": (
                    "707b C2-release non-degeneracy: on the LEARNED arm at least one LIMBIC "
                    "channel (ofc/liking/vigour) reaches the arbitration carrying routed "
                    "per-candidate RANGE > LIMBIC_ROUTED_RANGE_FLOOR (peak over P2 ticks), on a "
                    "strict-majority of DIVERGENT seeds. Without live limbic range the arbitration "
                    "has no limbic signal to learn to route through (the MECH-191 phasic gap). "
                    "measured = the NUMBER of DIVERGENT seeds on which a limbic channel is "
                    "routing-live; the per-seed test uses the SAME statistic (per-candidate "
                    "range) the conversion depends on, not a magnitude proxy."
                ),
                "control": "LEARNED loop_limbic_routed_range_max (peak limbic routed per-candidate range over P2)",
                # COUNT-shaped, INCLUSIVE floor: `met` is
                # `enough_divergent and _div_pass(len(named_routing_live_div), n_primary_div)`,
                # and _div_pass is `n_ok >= max(MIN_SEEDS_FOR_PASS, ceil(FRACTION * n_div))` --
                # the threshold reported here -- guarded by `n_div >= MIN_DIVERGENT_SEEDS`,
                # which is the same leg as `enough_divergent`, declared separately as
                # `enough_divergent_seeds`. This entry previously reported MAX across LEARNED
                # seeds of the limbic routed range against LIMBIC_ROUTED_RANGE_FLOOR, which is
                # strictly LOOSER than the shipped strict-majority count (one seed clearing the
                # floor satisfies a max), and the per-seed boolean is itself a CONJUNCTION
                # (routing_active_ticks > 0 AND range > floor), so no range statistic can
                # reproduce `met`. The max-range number is kept as a NON-BOUND diagnostic.
                "measured": float(len(named_routing_live_div)),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "comparator": ">=",
                "direction": "lower",
                "observed_learned_limbic_routed_range_max": float(round(learned_limbic_routed_range_max, 6)),
                "observed_limbic_routed_range_floor": float(LIMBIC_ROUTED_RANGE_FLOOR),
                "met": bool(named_channel_routing_live),
            },
            {
                "name": "learned_cross_loop_weights_moved_off_init",
                "kind": "readiness",
                "description": (
                    "ARC-108 x ARC-110 MECHANISM non-vacuity (W): on the LEARNED arm the learned "
                    "cross-loop matrix M_cross MOVED off its zero init -- loop_cross_loop_m_range "
                    "(per-tick) AND post_action_update clg_m_cross_range both > M_CROSS_RANGE_FLOOR "
                    "AND clg_n_updates > 0 -- on a strict-majority of DIVERGENT seeds. At init "
                    "M_cross==0 -> W_cross==I -> LEARNED is BIT-IDENTICAL to STATIC; if the weights "
                    "never moved a 'no lift' is meaningless => substrate_not_ready_requeue (NEVER "
                    "a weakens). measured = the NUMBER of DIVERGENT seeds on which the LEARNED "
                    "arm's M_cross moved off init."
                ),
                "control": "LEARNED clg_m_range_peak + clg_update_m_cross_range_max + clg_update_n_updates",
                # COUNT-shaped, INCLUSIVE floor: `met` is
                # `enough_divergent and _div_pass(len(weights_moved_div), n_primary_div)`,
                # threshold as reported; the `n_div >= MIN_DIVERGENT_SEEDS` leg is declared
                # separately as `enough_divergent_seeds`. This entry previously reported MAX
                # across LEARNED seeds of clg_m_range_peak against M_CROSS_RANGE_FLOOR, strictly
                # LOOSER than the shipped strict-majority count, and the per-seed
                # `clg_weights_moved` boolean is a THREE-way conjunction (per-tick M_cross range
                # AND post-update range both > floor AND n_updates > 0), which no single range
                # statistic can reproduce. The max-range number is kept as a NON-BOUND diagnostic.
                "measured": float(len(weights_moved_div)),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "comparator": ">=",
                "direction": "lower",
                "observed_learned_m_range_peak_max": float(round(learned_m_range_peak_max, 8)),
                "observed_m_cross_range_floor": float(M_CROSS_RANGE_FLOOR),
                "met": bool(learned_weights_moved),
            },
            {
                "name": "limbic_loop_parity_win",
                "kind": "readiness",
                "description": (
                    "THE LOAD-BEARING gate (the 711 autopsy target), NOW SATURATION-GUARDED: on the "
                    "A_ASCENDING_ON arm the limbic loop's effective column weight w_eff[limbic] "
                    "(from the PARITY-CONTROLLED W_cross) reaches a FAIR PARITY WIN -- it enters the "
                    "parity BAND (w_eff[motor] <= w_eff[limbic] <= PARITY_BAND_HI x w_eff[motor]) on "
                    ">= 1 P2 tick AND does NOT blow up (w_eff/M_cross under the saturation ceiling) -- "
                    "on a STRICT-MAJORITY (>= LIMBIC_WIN_PASS_FRACTION = 3/4) of DIVERGENT seeds. "
                    "This is the exact repair of the 711 hole: the raw scalar met a BARE "
                    "w_eff[limbic] >= w_eff[motor] via a 10-2274x MONOPOLY (saturation), which the "
                    "band+ceiling now rejects. If the controller cannot deliver a fair parity win on "
                    ">= 3/4 divergent seeds (lift too weak), the conversion question is not measured "
                    "=> substrate_not_ready_requeue, NEVER a weakens. measured = n divergent seeds "
                    "with a fair parity win; SAME statistic (effective column weight) the conversion "
                    "depends on."
                ),
                "control": "A_ASCENDING_ON clg_limbic_parity_win (band win AND not saturated) over P2",
                "measured": float(len(limbic_can_win_div)),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(LIMBIC_WIN_PASS_FRACTION * max(n_primary_div, 1))))),
                # COUNT-shaped, INCLUSIVE floor: `met` is
                # `enough_divergent and _win_pass(len(limbic_can_win_div), n_primary_div)`, and
                # _win_pass is `n_ok >= max(MIN_SEEDS_FOR_PASS, ceil(LIMBIC_WIN_PASS_FRACTION *
                # n_div))` -- exactly the threshold reported here, the TIGHTER 3/4 gate, not the
                # generic _div_pass -- guarded by `n_div >= MIN_DIVERGENT_SEEDS`, the same leg as
                # `enough_divergent`, declared separately as `enough_divergent_seeds`.
                "comparator": ">=",
                "direction": "lower",
                "met": bool(learned_limbic_can_win),
            },
            {
                "name": "no_saturation_blowup",
                "kind": "readiness",
                "description": (
                    "SATURATION GUARD (the load-bearing 711 repair): the number of DIVERGENT seeds "
                    "on the A_ASCENDING_ON arm whose arbitration BLEW UP -- w_eff[limbic] peak past "
                    "SATURATION_W_EFF_CEILING x w_eff[motor] peak, or M_cross range peak past "
                    "SATURATION_M_CROSS_CEILING -- must be ZERO (upper-bound: measured <= threshold). "
                    "Any saturated seed means a limbic MONOPOLY (not a fair win) and the conversion "
                    "was tested under a degenerate arbitration => substrate_not_ready_requeue, NEVER "
                    "a decisive FAIL/weakens. With a correctly tuned controller this is always 0."
                ),
                "control": "A_ASCENDING_ON clg_saturated count over divergent seeds (0 expected)",
                "measured": float(len(saturated_div)),
                "threshold": 0.0,
                # CEILING-shaped and INCLUSIVE: `met` is `not (len(saturated_div) > 0)`, i.e.
                # `len(saturated_div) <= 0`. `direction` was already declared (so this entry did
                # NOT carry the 2026-06-07 V3-EXQ-648a/649 floor-default bug); the comparator is
                # added to mirror the shipped predicate's strictness explicitly, since an upper
                # bound with a strict "<" would reject the expected measured == threshold == 0.
                "comparator": "<=",
                "direction": "upper",
                "met": bool(not saturation_detected),
            },
            {
                "name": "learning_engaged_finer_channels_dissociable",
                "kind": "readiness",
                "description": (
                    "on BOTH arms the finer w_chan_finer entries MOVED + carry cross-channel range "
                    "above floor, on a majority of seeds -- learning is engaged. measured = the "
                    "WORSE of the two arms' counts of fcg_moved seeds. Below floor => "
                    "substrate_not_ready_requeue."
                ),
                "control": "fcg_w_chan_finer_range_max (both arms)",
                # COUNT-shaped, INCLUSIVE floor: `met` is
                # `_maj(static, fcg_moved) and _maj(learned, fcg_moved)` -- two counts against
                # the SAME MIN_SEEDS_FOR_PASS threshold -- so the MIN of the two counts
                # reproduces the conjunction exactly. SPLIT from the delta_t leg below, which is
                # a DIFFERENT statistic: the single entry that used to carry both had
                # `met = fcg_moved_ok and fcg_delta_nonflat_ok`, which no one pair can
                # reproduce, and it reported min over all rows of fcg_w_chan_finer_range_max --
                # a per-seed magnitude strictly harsher than a majority count, silent on the
                # delta_t leg. The min-magnitude number is kept as a NON-BOUND diagnostic.
                "measured": float(n_fcg_moved_min_arm),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_w_chan_finer_range_max": float(
                    min([r["fcg_w_chan_finer_range_max"] for r in all_rows] or [0.0])
                ),
                "observed_w_chan_finer_range_floor": float(W_CHAN_FINER_RANGE_FLOOR),
                "met": bool(fcg_moved_ok),
            },
            {
                "name": "learning_engaged_delta_nonflat",
                "kind": "readiness",
                "description": (
                    "on BOTH arms the signed-RPE delta_t carries cross-tick variance above floor "
                    "on a majority of seeds -- the second leg of the learning-engaged guard. "
                    "measured = the WORSE of the two arms' counts of fcg_delta_nonflat seeds."
                ),
                "control": "fcg_delta_t_std (both arms)",
                # COUNT-shaped, INCLUSIVE floor: `met` is
                # `_maj(static, fcg_delta_nonflat) and _maj(learned, fcg_delta_nonflat)`, again
                # two counts against the same threshold, so the MIN reproduces it exactly. See
                # the entry above for why the two legs are declared separately; their
                # conjunction is the shipped predicate and the routing still reads the booleans.
                "measured": float(n_fcg_delta_nonflat_min_arm),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_delta_t_std": float(
                    min([r["fcg_delta_t_std"] for r in all_rows] or [0.0])
                ),
                "observed_delta_t_std_floor": float(DELTA_T_STD_FLOOR),
                "met": bool(fcg_delta_nonflat_ok),
            },
            {
                "name": "candidate_pool_divergent",
                "kind": "readiness",
                "description": (
                    "consumed cand_world_summaries (e2.world_forward) per-candidate SPREAD clears "
                    "the GAP-A non-vacuity floor on enough seeds: measured = the NUMBER of seeds "
                    "DIVERGENT on BOTH arms, threshold = MIN_DIVERGENT_SEEDS."
                ),
                "control": "SD-056 e2 trained online in P0; candidate_summary_source=e2_world_forward",
                # COUNT-shaped, INCLUSIVE floor: `met` is `enough_divergent`, i.e.
                # `n_primary_div >= MIN_DIVERGENT_SEEDS` -- a COUNT of seeds divergent on BOTH
                # arms. This entry previously reported min over all_rows of
                # consumed_summary_pairwise_dist_mean against CONSUMED_SPREAD_FLOOR, strictly
                # HARSHER than the shipped count predicate (a majority of seeds can clear the
                # floor while the min does not), so the indexer's authoritative recompute
                # wrongly flagged sound diagnostics precondition_unmet. No spread statistic CAN
                # reproduce `met`: the per-seed divergence boolean is itself a CONJUNCTION
                # (spread > CONSUMED_SPREAD_FLOOR and dist_max < CONSUMED_MAGNITUDE_CEIL)
                # evaluated on EACH arm, and a count over a conjunction does not distribute into
                # per-leg counts. The min-spread number is preserved as a NON-BOUND diagnostic.
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_consumed_spread": float(
                    min([r["consumed_summary_pairwise_dist_mean"] for r in all_rows] or [0.0])
                ),
                "observed_consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
                "met": bool(enough_divergent),
            },
        ],
        "criteria": [
            {
                "name": "C1_learned_strict_above_static",
                "load_bearing": True,
                "passed": bool(c1_holds),
            },
        ],
        "criteria_non_degenerate": {
            "preconditions_met": bool(preconditions_met),
            "enough_divergent_seeds": bool(enough_divergent),
            "loops_carry_live_cross_loop_variance": bool(loop_cross_variance_ok),
            "named_channel_routing_live": bool(named_channel_routing_live),
            "learned_cross_loop_weights_moved": bool(learned_weights_moved),
            "limbic_loop_parity_win": bool(learned_limbic_can_win),
            "no_saturation_blowup": bool(not saturation_detected),
            "learning_engaged": bool(fcg_moved_ok and fcg_delta_nonflat_ok),
            "crf_matured": bool(crf_matured),
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
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "conversion_margin": float(CONVERSION_MARGIN),
            "min_divergent_seeds": int(MIN_DIVERGENT_SEEDS),
            "divergent_pass_fraction": float(DIVERGENT_PASS_FRACTION),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "loop_cross_variance_frac_floor": float(LOOP_CROSS_VARIANCE_FRAC_FLOOR),
            "loop_pref_range_floor": float(LOOP_PREF_RANGE_FLOOR),
            "limbic_routed_range_floor": float(LIMBIC_ROUTED_RANGE_FLOOR),
            "m_cross_range_floor": float(M_CROSS_RANGE_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "delta_t_std_floor": float(DELTA_T_STD_FLOOR),
            "w_chan_finer_range_floor": float(W_CHAN_FINER_RANGE_FLOOR),
            "learned_cross_loop_eta": float(LEARNED_CROSS_LOOP_ETA),
            "loop_segregation_normalize": str(LOOP_SEGREGATION_NORMALIZE),
        },
        "acceptance_criteria": {
            "preconditions_met": preconditions_met,
            "n_divergent_seeds": int(n_primary_div),
            "enough_divergent_seeds": enough_divergent,
            "crf_matured": crf_matured,
            "loops_carry_live_cross_loop_variance": loop_cross_variance_ok,
            "named_channel_routing_live": named_channel_routing_live,
            "n_named_routing_live_over_divergent": int(len(named_routing_live_div)),
            "learned_limbic_routed_range_max": round(learned_limbic_routed_range_max, 6),
            "learned_cross_loop_weights_moved": learned_weights_moved,
            "n_weights_moved_over_divergent": int(len(weights_moved_div)),
            "learned_m_range_peak_max": round(learned_m_range_peak_max, 8),
            "learned_n_updates_max": int(learned_n_updates_max),
            "limbic_loop_parity_win": learned_limbic_can_win,
            "n_limbic_parity_win_over_divergent": int(len(limbic_can_win_div)),
            "saturation_detected": saturation_detected,
            "n_saturated_over_divergent": int(len(saturated_div)),
            "learned_w_limbic_exceeds_motor_total_ticks": learned_w_limbic_exceeds_motor_total,
            "learning_engaged_fcg_moved": fcg_moved_ok,
            "learning_engaged_fcg_delta_nonflat": fcg_delta_nonflat_ok,
            "C1_ascending_on_above_off": c1_holds,
            "C1_n_seeds": int(n_c1),
            "C1_n_divergent": int(n_primary_div),
            "mean_committed_class_entropy_ascending_off": round(static_mean_dv, 6),
            "mean_committed_class_entropy_ascending_on": round(learned_mean_dv, 6),
        },
        "interpretation_grid": {
            "PASS_ascending_spiral_gain_lifts_conversion_ceiling_supports_arc108_arc110": (
                "preconditions met (divergent seeds + loops carry LIVE cross-loop variance + "
                "named-channel routing live + the M_cross weights MOVED off init on the ON arm + "
                "the LOAD-BEARING gate: the ascending gain lets the limbic loop reach >= motor "
                "effective weight on >= 3/4 divergent seeds + learning engaged) AND C1 "
                "(A_ASCENDING_ON committed-class entropy strict-above A_ASCENDING_OFF + margin on a "
                "strict-majority of divergent seeds). Once a non-motor loop CAN win, the ascending "
                "spiral gain CONVERTS committed-action diversity where the un-gained 709 learned "
                "arm could not -> the F-dominance conversion ceiling is LIFTABLE (weakens MECH-439), "
                "via ARC-108 learned gating at the ARC-110 cross-loop arbitration (supports both). "
                "First evidence that a strengthened ascending spiral lifts the loop-effective-weight ceiling."
            ),
            "FAIL_ascending_spiral_gain_lets_limbic_win_but_does_not_convert_ceiling_intrinsic_weakens_arc108_arc110": (
                "DECISIVE. preconditions met (the M_cross MOVED, and the ascending gain DID lift the "
                "limbic loop to >= motor effective weight on >= 3/4 divergent seeds -- the "
                "load-bearing gate MET, loops live) BUT A_ASCENDING_ON still does NOT lift "
                "committed-class entropy strict-above A_ASCENDING_OFF. Even when a non-motor loop "
                "CAN win the arbitration, committed-action diversity does NOT convert -> the "
                "F-dominance conversion ceiling is INTRINSIC (supports MECH-439); the ARC-108 x "
                "ARC-110 loop-arbitration route does not deliver (weakens both)."
            ),
            "FAIL_substrate_not_ready_requeue": (
                "A precondition is unmet: too FEW divergent seeds, OR the loops carry NO live "
                "cross-loop variance, OR the NAMED limbic channels carry NO routed per-candidate "
                "range (MECH-191 phasic gap -- limbic loop inert), OR the M_cross weights did NOT "
                "move off init on the ON arm, OR the LOAD-BEARING gate is unmet (even WITH the "
                "ascending gain the limbic loop reached >= motor effective weight on FEWER than 3/4 "
                "divergent seeds -- the gain needs strengthening), OR learning was not engaged / CRF "
                "not matured. The conversion question could NOT be measured -- NOT a falsification."
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
        "evidence_direction_note": (
            f"V3-EXQ-711 ARC-110 x ARC-108 ASCENDING-SPIRAL GAIN VALIDATION (experiment_purpose="
            f"evidence; claim_ids=[MECH-439, ARC-108, ARC-110]). The SEPARATE new-EXQ falsifier BOTH "
            f"the V3-EXQ-709 AND V3-EXQ-710 confirmed autopsies routed to as the load-bearing "
            f"V3-closure build (clears a re-derive brake -- new EXQ number + a NEWLY-BUILT mechanism "
            f"on a freshly-built substrate). 709 built the LEARNED [3,3] M_cross cross-loop matrix "
            f"fully live (6/7 gates; M_cross range 0.116; limbic routing 1.414) but the limbic loop "
            f"reached >= motor EFFECTIVE COLUMN WEIGHT on only 1/4 divergent seeds -- the ascending "
            f"path M_cross[motor,limbic] peaked ~0.03, functionally too weak to lift a non-motor loop "
            f"above the F-pinned motor loop, so C1 (conversion) could not be validly evaluated. The "
            f"substrate landed 2026-07-04 (V3-EXQ-711 runaway repair) adds a BOUNDED ascending gain -- "
            f"a target-PARITY controller (use_ascending_parity_controller): a per-step forward gain "
            f"SOLVED to hold w_eff[limbic] toward but under PARITY_CEILING_RATIO x w_eff[motor], + a "
            f"clamp on the ascending M_cross entries. It replaces the raw 711 scalar that ran "
            f"w_eff[limbic] to 10-2274x w_eff[motor] (a limbic MONOPOLY). 2 arms on the SAME GAP-A "
            f"reef-bipartite substrate + the SAME matched envelope (finer gating + learned settling + "
            f"named-channel routing + limbic input modules + LEARNED cross-loop arbitration ON on "
            f"BOTH); the ONLY swept factor is use_ascending_parity_controller: A_ASCENDING_OFF (the "
            f"un-gained 709 ceiling baseline) vs A_ASCENDING_ON (bounded controller: forward gain "
            f"{PARITY_FORWARD_GAIN} capped at ratio {PARITY_CEILING_RATIO} + plasticity gain "
            f"{PARITY_PLASTICITY_GAIN} clamped at {PARITY_M_CROSS_CLAMP}). Inert / OFF -> the arms are "
            f"bit-identical. LOAD-BEARING PRECONDITION (the autopsy target, SATURATION-GUARDED): on "
            f"A_ASCENDING_ON the limbic loop reaches a FAIR PARITY win (band, not blow-up) on a "
            f"STRICT-MAJORITY (>=3/4) of divergent seeds. PRE-REGISTERED decisive: C1 = A_ASCENDING_ON "
            f"committed-class entropy strict-above A_ASCENDING_OFF + margin on a strict-majority "
            f"(>=2/3) of divergent seeds. PASS (precondition met + C1) => the bounded controller LIFTS "
            f"the F-dominance conversion ceiling under a FAIR parity arbitration -> MECH-439 weakens / "
            f"ARC-108 supports / ARC-110 supports. FAIL (precondition met, C1 fails; decisive and now "
            f"VALID) => even at fair bounded parity, diversity does NOT convert -> MECH-439 supports "
            f"(ceiling INTRINSIC) / ARC-108 + ARC-110 weakens. Non-vacuity self-route "
            f"substrate_not_ready_requeue (non_contributory, non_degenerate=False, NEVER a false "
            f"weakens) when the controller SATURATES (blow-up on any divergent seed) OR cannot deliver "
            f"a fair parity win on >=3/4 divergent seeds OR the M_cross weights did NOT move OR the "
            f"liveness preconditions fail. PROMOTES NOTHING "
            f"(MECH-439 candidate/substrate_ceiling; ARC-108/ARC-110 candidate/substrate_conditional/v3). "
            f"outcome={result['outcome']}; label={result['interpretation_label']}; "
            f"per_claim={result['evidence_direction_per_claim']}."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "2-arm ARC-110 x ARC-108 BOUNDED parity-controller validation (A_ASCENDING_OFF vs A_ASCENDING_ON) + per-seed-divergent gating + the load-bearing SATURATION-GUARDED limbic-loop-parity-win (>=3/4) gate + the 709 mechanism non-vacuity gates (M_cross weights-moved)",
            "arms": "A_ASCENDING_OFF (use_ascending_parity_controller=False -- the un-gained 709 learned-arbitration ceiling baseline) / A_ASCENDING_ON (use_ascending_parity_controller=True -- bounded target-parity controller: forward gain capped at the parity ceiling + clamped ascending M_cross maturation). Learned cross-loop arbitration ON on BOTH arms; the raw scalar (use_ascending_spiral_gain) OFF on BOTH.",
            "swept_variable": "use_ascending_parity_controller ONLY (bounded forward parity-ceiling gain + clamped plasticity maturation on the ascending upper-triangular M_cross entries). Learned cross-loop arbitration + loop segregation + ARC-109 D1/D2 + MECH-452 loop-local traces + named-channel routing + limbic input modules + finer gating + learned settling ON on BOTH arms.",
            "the_isolated_factor": (
                "whether the ascending (strict-upper-triangle) M_cross entries are BOUNDED-GAINED by "
                "the parity controller: OFF runs W_cross = I + M_cross (the un-gained 709 learned "
                "combine); ON runs W_cross = I + (G(g) .* M_cross) where g is SOLVED per-step to hold "
                "w_eff[limbic] toward but under PARITY_CEILING_RATIO x w_eff[motor] (a fair parity, "
                "not a monopoly), and clamps the ascending M_cross maturation to +/- m_cross_clamp. "
                "Raises w_eff[limbic]/w_eff[assoc] WITHOUT touching w_eff[motor], bounded. At init "
                "M_cross==0 or inert params -> bit-identical to OFF."
            ),
            "parity_forward_gain": PARITY_FORWARD_GAIN,
            "parity_ceiling_ratio": PARITY_CEILING_RATIO,
            "parity_plasticity_gain": PARITY_PLASTICITY_GAIN,
            "parity_m_cross_clamp": PARITY_M_CROSS_CLAMP,
            "parity_band_hi": PARITY_BAND_HI,
            "saturation_w_eff_ceiling": SATURATION_W_EFF_CEILING,
            "saturation_m_cross_ceiling": SATURATION_M_CROSS_CEILING,
            "limbic_win_pass_fraction": LIMBIC_WIN_PASS_FRACTION,
            "matched_constant_arithmetic_envelope": (
                "use_f_eligibility_demotion + use_f_eligibility_adaptive_floor (689e) + "
                "use_go_nogo_constitution (689g) + use_modulatory_selection_authority (643a) + "
                "use_modulatory_channel_routing (cand_world_summary) + top_k shortlist (k=3, 569i)"
            ),
            "matched_loop_substrate": (
                "use_loop_segregation=True + use_d1_d2_population_split + use_loop_local_eligibility_traces "
                "+ use_named_channel_routing + use_ofc_analog + use_mech295_liking_bridge + use_tonic_vigor "
                "(the 707b C2-release stack: the limbic loop carries live per-candidate range on both arms)"
            ),
            "matched_diversity_stack": (
                "MECH-341 stratified + use_dacc + use_gated_policy + use_lateral_pfc_analog (trained P1 "
                "REINFORCE) + SD-056 all levers + matured/maintained CRF + use_candidate_rule_field + "
                "use_finer_channel_gating + use_learned_settling_step"
            ),
            "primary_dv": "committed-action-class entropy (nats), interpreted on divergent seeds only",
            "phases": "P0 e2-train (CRF matures) -> P1 frozen-encoder bias-head REINFORCE -> P2 e2+bias frozen; finer gating + loops + M_cross KEEP adapting",
            "learning_wiring": "M_cross updated via e3.post_action_update driven by agent.update_residue every WAKING tick (MECH-094); shares the ARC-108 delta_t with w_chan_finer + W_lat; no autograd (register_buffer)",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "learned_cross_loop_eta": LEARNED_CROSS_LOOP_ETA,
            "loop_segregation_normalize": LOOP_SEGREGATION_NORMALIZE,
            "loop_default_channel_map": "motor=F; associative=dACC+lPFC; limbic=OFC+liking+vigour (built-in _LOOP_DEFAULT_CHANNEL_MAP)",
            "reusable_arm_ids": [],
            "reuse_note": "both arms ride the in-flux learned-cross-loop / loop-segregation substrate -> neither minted as a reusable baseline (extra_ineligible_reasons set)",
            "safety": "arbitration runs strictly within the F + MECH-448/449 Go/No-Go eligible set; a learned weight reorders within-eligible candidates but can never re-admit a suppressed one",
            "mech439_arc108_arc110_relationship": "this is the ascending-spiral-gain repair of the loop-effective-weight ceiling the 709/710 autopsies both routed to; PASS (limbic can win on >=3/4 + C1) => ceiling liftable (MECH-439 weakens, ARC-108/110 supports), FAIL (limbic can win but no conversion) => ceiling intrinsic (MECH-439 supports, ARC-108/110 weakens), limbic can't win on >=3/4 => substrate_not_ready_requeue (non_contributory, never a weakens)",
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-711 ARC-110 x ARC-108 ascending-spiral gain validation"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

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
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    _ac = result["acceptance_criteria"]
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"preconditions_met={_ac['preconditions_met']} "
        f"n_divergent={_ac['n_divergent_seeds']} "
        f"loop_cross_variance={_ac['loops_carry_live_cross_loop_variance']} "
        f"named_routing_live={_ac['named_channel_routing_live']} "
        f"weights_moved={_ac['learned_cross_loop_weights_moved']} (m_range_peak={_ac['learned_m_range_peak_max']}, n_updates={_ac['learned_n_updates_max']}) "
        f"limbic_parity_win(>=3/4)={_ac['limbic_loop_parity_win']} (n_saturated={_ac['n_saturated_over_divergent']}, strict_exceed_ticks={_ac['learned_w_limbic_exceeds_motor_total_ticks']}) "
        f"C1_ascending_on_above_off={_ac['C1_ascending_on_above_off']} "
        f"(off={_ac['mean_committed_class_entropy_ascending_off']}, on={_ac['mean_committed_class_entropy_ascending_on']}) "
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
