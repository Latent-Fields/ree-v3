#!/opt/local/bin/python3
"""
V3-EXQ-700 -- ARC-108 sec-7 SELECTION 2x2 LEARNED-GATING FALSIFIER. Does a
LEARNED, signed-RPE-driven adjustment of E3 per-channel selection weights
(JOB-1 step-1 w_chan) and/or a learned recurrent lateral-inhibition settling
step (MECH-450 step-2 W_lat) CONVERT committed-action-class diversity into
committed action where the PURE-ARITHMETIC envelope plateaus -- or is there no
lift? The next MECH-439 (F-dominance) attack: LEARNED, not arithmetic.

DESIGN-OF-RECORD (implement sec 5 EXACTLY; do NOT re-derive):
  REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md sec 5
  + REE_assembly/docs/architecture/dopamine_into_gating.md ("What remains").

WHY THIS, WHY NOW. The ARC-107 arbitration layer (MECH-447 conflict-grade,
MECH-448 rank-preserving F->eligibility demotion, MECH-449 Go/No-Go) is PURE
ARITHMETIC with no learned parameters -- the assembly-map A.4 learning gap and
the predicted root of F-dominance (F monopolises 88-89%% of E3 committed-
selection variance, V3-EXQ-571 / MECH-439). The campaign's own failure history
(structural bounding works, parametric tuning does not, each structural lever
needs hand-calibration per channel) is the signature of a system that needs a
selection rule that LEARNS its weights. The learned-gating mechanism is now
FULLY BUILT in ree-v3 main: JOB-1 step-1 w_chan + signed-RPE delta_t landed
ae907b5; MECH-450 step-2 learned-W_lat settling landed 9839492018. Both are
no-op-default flags on E3Config / REEConfig.from_dims.

THE 2x2 (all arms carry the landed arithmetic envelope as a MATCHED CONSTANT --
exactly as V3-EXQ-699: use_f_eligibility_demotion=True +
use_f_eligibility_adaptive_floor=True + use_go_nogo_constitution=True + the
modulatory-authority/top-k shortlist scaffold; the swept variables are ONLY the
two learned levers):
  A0 (envelope-only control) : use_learned_channel_gating=False, use_learned_settling_step=False
  A1 (learned w_chan)        : use_learned_channel_gating=True,  use_learned_settling_step=False
  A2 (learned W_lat settling): use_learned_channel_gating=False, use_learned_settling_step=True
  A3 (both)                  : use_learned_channel_gating=True,  use_learned_settling_step=True
plus
  ARM_NOISE (verified-lifting matched-noise temperature control; envelope-only +
             use_noise_floor): the 569g/684 lesson -- an entropy lift a noise
             control ALSO produces is NOT lawful conversion, so a VERIFIED-LIFTING
             matched-noise control is the bar the learning arms must EXCEED.
3 seeds. claim_ids = [MECH-439, ARC-108, MECH-450]. experiment_purpose = evidence.
PROMOTES NOTHING until it scores -- MECH-439 is substrate_ceiling and
ARC-108 / MECH-450 are substrate_conditional / v3, so promote/demote is
SUPPRESSED on all three (a PASS feeds confidence; it does not promote).

C3 DEFERRED (user-confirmed 2026-06-22): the signed-vs-unsigned-RPE ablation
(swap the signed delta_t for the unsigned ARC-016 variance -- must FAIL to
convert, falsifies B5) has NO substrate config knob -- delta_t = R_t - V-hat_t
is hard-wired in e3_selector.post_action_update. Implementing it here would
require an experiment-side override of substrate internals (fragile). It is
routed to a small /implement-substrate flag (e.g. learned_channel_rpe_mode:
signed|unsigned) + a follow-up arm. C1 (conversion) + C2 (learning is load-
bearing) deliver the primary "converts-where-arithmetic-plateaus + learning-is-
load-bearing" verdict on the built substrate; C3 is the next step.

SUBSTRATE -- the GAP-A-ready foraging bank (the non-vacuity precondition every
conversion-ceiling experiment now requires): SD-056-trained e2.world_forward +
ARC-065 GAP-A candidate_summary_source="e2_world_forward" -> a genuinely
DIVERGENT candidate pool. Scaffold mirrors V3-EXQ-699 (the same matured/
maintained CRF pool, top_k shortlist-then-modulate, modulatory authority +
routing, MECH-341 stratified, MECH-313 noise floor available, V_s minimal,
use_gated_policy, use_lateral_pfc_analog with the SD-033a bias head un-zeroed +
TRAINED in a frozen-encoder P1 REINFORCE window, SD-056 all levers, use_dacc ->
the MECH-260 recency-share vector feeding the Go/No-Go perseveration axis).

R_t -- the signed-RPE delta_t reads benefit_eval - harm_eval on actual_z_world.
Those heads are at the substrate's standard (untrained-on-this-bank) state, as
in the ENTIRE 571/689/699 campaign that established the F-dominance this attacks
-- so R_t is "the realised outcome valence the substrate computes," internally
consistent with the F it must beat. The C1-delta_t non-vacuity precondition
(delta_t carries cross-tick variance) protects validity: a degenerate / flat R_t
self-routes substrate_not_ready_requeue rather than producing a false verdict.

LEARNING WIRING. w_chan / W_lat learn via the three-factor rule inside
e3.post_action_update, which agent.update_residue (called every waking tick in
all phases) drives automatically -- no explicit call needed. select() records
the eligibility / settling co-activation trace (waking, simulation_mode=False);
the next update_residue consumes it (MECH-094). The learned weights + V-hat_t
PERSIST across episodes; agent.reset() clears only the within-episode credit
window. use_habenula_decommit stays default OFF so the JOB-2 branch never fires.

Phases / budget
---------------
P0 (e2 TRAINED online via SD-056 contrastive; bias head NOT trained; CRF matures
   across episodes; learned gating ON for the learning arms from the start --
   the three-factor update is automatic on the waking path).
P1 (encoder FROZEN, bias head TRAINED via outcome-coupled REINFORCE; learned
   gating still adapting): the GAP-D trained-bias-head window.
P2 (e2 + bias head FROZEN; the LEARNED gating KEEPS ADAPTING; instrumentation ON):
   the behavioural measurement window. P2 is binned first-half / second-half so
   C2 can read whether the committed-class entropy lift GROWS as the learned gate
   keeps adapting on a frozen encoder + frozen bias head.

PRIMARY ACCEPTANCE (design sec 5.2)
-----------------------------------
  C1 (conversion): committed-action-class entropy on a LEARNING arm (A1 and/or
      A3 for w_chan; A2/A3 for the settling step) strict-above BOTH the A0
      envelope-only arm AND the VERIFIED-LIFTING matched-noise control, by
      CONVERSION_MARGIN on >= 2/3 seeds.
  C2 (learning is load-bearing -- the core discriminator): the lift GROWS over
      training (second-half P2 committed-class entropy strict-above first-half on
      the learning arms, by GROWTH_MARGIN on >= 2/3 seeds) AND the learned weights
      MOVED from init (w_chan via lcg_w_chan_range > floor on A1/A3; W_lat via
      wlat_range > floor AND the settling MOVED the field
      learned_settling_round_delta > floor on A2/A3). A lift present from tick 0
      that does not grow is a STATIC re-weighting -> fold into the arithmetic
      lever, do NOT credit learning.
  C3 (signed-RPE load-bearing): DEFERRED to a substrate flag (see above).

NON-VACUITY / SELF-ROUTE (never a false weakening; design sec 5.3)
-----------------------------------------------------------------
  self-route substrate_not_ready_requeue if ANY precondition is unmet:
   (a) candidate pool NOT divergent (consumed cand_world_pairwise_dist below
       floor on the arms) -- nothing to convert;
   (b) delta_t FLAT (no outcome variance to learn from) on the learning arms;
   (c) w_chan / W_lat never MOVE (eligibility trace never credited) on their
       armed arms, or the settling never moves the field;
   (d) the matched-noise control does NOT lift above A0 (an unverified bar --
       the 569g/684 control-failure mode).
  A PRECONDITIONS-MET FAIL (learning engaged, weights moved, pool divergent, but
  NO entropy lift over A0/noise) is the GENUINE "no lift" outcome -> routes to
  the V4 full-loop scope bet (design sec 6); it does NOT falsify ARC-107 (the
  arithmetic envelope still holds) and does NOT promote learned gating.

Four-way verdict grid (design sec 5.4)
--------------------------------------
  C1 yes + C2 yes -> learned_gating_converts_where_arithmetic_plateaus
                     (PASS; supports MECH-439 + the learning claim(s) that converted)
  C1 yes + C2 no  -> static_reweighting_fold_into_arithmetic
                     (PASS; MECH-439 supports [ceiling liftable]; ARC-108/MECH-450
                     mixed -- the lift is NOT from learning)
  C1 no (precond met) -> no_lift_preconditions_met_escalate_v4_full_loop
                     (FAIL; non_contributory for all -- NOT a weakening)
  C1 no (precond unmet) -> substrate_not_ready_requeue (vacuous; re-queue)

See REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md,
    REE_assembly/evidence/planning/unified_dopamine_substrate_design_2026-06-22.md,
    REE_assembly/docs/architecture/dopamine_into_gating.md,
    REE_assembly/evidence/planning/conversion_ceiling_campaign_plan.md,
    experiments/v3_exq_699_pcomp_demotion_x_gonogo_composition.py (scaffold mirrored),
    tests/contracts/test_arc108_learned_channel_gating.py,
    tests/contracts/test_mech450_learned_settling_step.py.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, deque
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


EXPERIMENT_TYPE = "v3_exq_700_arc108_sec7_learned_gating_2x2"
QUEUE_ID = "V3-EXQ-700"
CLAIM_IDS: List[str] = ["MECH-439", "ARC-108", "MECH-450"]
EXPERIMENT_PURPOSE = "evidence"

# CRF-gate calibration levers (matured CRF stack; ported verbatim from V3-EXQ-699,
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
MIN_SEEDS_FOR_PASS = 2  # of 3

# C1(a) readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# Non-vacuity (a): GAP-A consumed-summary divergence (649 statistic + 643a ceiling).
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# Non-vacuity (b): delta_t carries cross-tick variance (outcome variance to learn from).
DELTA_T_STD_FLOOR = 1e-4
# Non-vacuity (c): the learned weights MOVED from init / the settling MOVED the field.
W_CHAN_RANGE_FLOOR = 1e-4   # softplus-unity init => range 0; >floor == reorganised
WLAT_RANGE_FLOOR = 1e-4     # zero init => range 0; >floor == non-zero inhibition learned
SETTLING_ROUND_DELTA_FLOOR = 1e-4  # the settling step actually moved _modulatory_accum

# CRF maturity readiness (matched constant; the differentiated source must be present).
CRF_MIN_MINTED = 2
CRF_N_ACTIVE_FLOOR = 1
CRF_FRAC_ACTIVE_FLOOR = 0.30

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 100
P1_BIAS_TRAIN_EPISODES = 50
P2_MEASUREMENT_EPISODES = 100   # binned first/second half for C2 grows-over-training
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 4   # >= 2 so the P2 first/second-half split is exercised in the smoke
DRY_RUN_STEPS = 30

# --- Matched-stack lever constants (identical on ALL arms; the landed envelope) ---
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
USE_MODULATORY_CHANNEL_ROUTING = True
MODULATORY_CHANNEL_ROUTE_SOURCE = "cand_world_summary"
MODULATORY_CHANNEL_ROUTE_WEIGHT = 1.0
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6
# 569i TOP-K shortlist scaffold -- the eligible set the settling step needs
# (_modulatory_accum[eligible_idx] with n_eligible >= 2). MATCHED CONSTANT.
USE_MODULATORY_SHORTLIST_THEN_MODULATE = True
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_SHORTLIST_K = 3
# MECH-448 demotion envelope (the landed arithmetic envelope CONSTANT on all arms).
USE_F_ELIGIBILITY_DEMOTION = True
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30   # legacy absolute floor (ignored under the adaptive floor)
F_ELIGIBILITY_DN_SIGMA = 0.0
USE_F_ELIGIBILITY_ADAPTIVE_FLOOR = True
F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR = 1.0
# MECH-449 Go/No-Go constitution (landed envelope CONSTANT on all arms).
USE_GO_NOGO_CONSTITUTION = True
USE_DACC = True
GNG_PERSEVERATION_FLOOR = 0.5
GNG_SAFETY_FLOOR = 0.5
GNG_PROTECT_MIN_ELIGIBLE = 1
MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

# use_candidate_rule_field is a MATCHED CONSTANT (the differentiated conversion source).
USE_CANDIDATE_RULE_FIELD = True

# ----- ARC-108 JOB-1 learned-gating knobs (substrate defaults; matched when armed) -----
LCG_ETA = 0.01
LCG_ELIG_DECAY = 0.9
LCG_VALUE_BASELINE_BETA = 0.05
LCG_ASYM_POTENTIATION = 1.0
LCG_ASYM_DEPRESSION = 0.5
# MECH-450 settling knobs (substrate defaults; matched when armed).
SETTLING_ROUNDS = 3
SETTLING_TEMPERATURE = 1.0
SETTLING_ETA = 0.01
SETTLING_ELIG_DECAY = 0.9
SETTLING_N_ACTION_CLASSES = 8

# ARM_NOISE matched-noise temperature control: a STRONG, verified-lifting noise
# floor on the envelope-only arm (MECH-313 LC-NE tonic temperature lift). The bar
# the learning arms must EXCEED so a lift is attributable to learned STRUCTURE,
# not unstructured temperature.
NOISE_FLOOR_ALPHA = 1.0
NOISE_FLOOR_MIN_TEMPERATURE = 1.0

# SD-056 online e2 training (mirror V3-EXQ-649/654j/699).
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

# P1 bias-head REINFORCE training (mirror V3-EXQ-598b/654j/699).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9


# IDENTICAL env to V3-EXQ-699 / 654j (the GAP-A foraging bank).
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


# The 5 arms. The ONLY swept config is (lcg_on, settle_on, noise_on). The whole
# arithmetic envelope + the diversity stack are matched constants.
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "A0_ENVELOPE_ONLY",
        "label": "envelope_only_control_no_learned_gating",
        "lcg_on": False,
        "settle_on": False,
        "noise_on": False,
    },
    {
        "arm_id": "A1_LEARNED_GATING",
        "label": "learned_w_chan_signed_rpe_per_channel_weight",
        "lcg_on": True,
        "settle_on": False,
        "noise_on": False,
    },
    {
        "arm_id": "A2_LEARNED_SETTLING",
        "label": "learned_W_lat_recurrent_settling_step",
        "lcg_on": False,
        "settle_on": True,
        "noise_on": False,
    },
    {
        "arm_id": "A3_BOTH",
        "label": "learned_w_chan_plus_learned_W_lat_settling",
        "lcg_on": True,
        "settle_on": True,
        "noise_on": False,
    },
    {
        "arm_id": "ARM_NOISE",
        "label": "verified_lifting_matched_noise_temperature_control",
        "lcg_on": False,
        "settle_on": False,
        "noise_on": True,
    },
]

LEARNING_ARM_IDS = ("A1_LEARNED_GATING", "A2_LEARNED_SETTLING", "A3_BOTH")
LCG_ARM_IDS = ("A1_LEARNED_GATING", "A3_BOTH")
SETTLE_ARM_IDS = ("A2_LEARNED_SETTLING", "A3_BOTH")


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Matched-stack agent. The landed arithmetic envelope (demotion + adaptive
    floor + Go/No-Go + authority + routing + top_k shortlist) + the diversity
    stack (MECH-341, SD-056, CRF, trained lateral_pfc bias head, use_dacc) are
    MATCHED CONSTANTS on all arms. The ONLY toggled flags are the two learned
    levers (use_learned_channel_gating, use_learned_settling_step) and -- on
    ARM_NOISE only -- the matched-noise temperature floor (use_noise_floor)."""
    lcg_on = bool(arm["lcg_on"])
    settle_on = bool(arm["settle_on"])
    noise_on = bool(arm["noise_on"])
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
        # 569i TOP-K shortlist scaffold (the eligible set the settling acts inside).
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
        # MECH-313 noise floor -- ARMED only on ARM_NOISE (the matched-noise control);
        # OFF on A0/A1/A2/A3 so the learning lift is not confounded by tonic noise.
        use_noise_floor=noise_on,
        noise_floor_alpha=(NOISE_FLOOR_ALPHA if noise_on else 0.1),
        noise_floor_min_temperature=NOISE_FLOOR_MIN_TEMPERATURE,
        # V_s minimal stack.
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        # ARC-062 GatedPolicy (matched; symmetry-broken bias).
        use_gated_policy=True,
        # SD-033a LateralPFCAnalog with the bias head UN-ZEROED + trainable (GAP-D).
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
        # --- ARC-108 JOB-1 step-1: learned per-channel selection gating (TOGGLED) ---
        use_learned_channel_gating=lcg_on,
        learned_channel_gating_eta=LCG_ETA,
        learned_channel_gating_elig_decay=LCG_ELIG_DECAY,
        learned_channel_value_baseline_beta=LCG_VALUE_BASELINE_BETA,
        learned_channel_asym_potentiation=LCG_ASYM_POTENTIATION,
        learned_channel_asym_depression=LCG_ASYM_DEPRESSION,
        # --- MECH-450 JOB-1 step-2: learned recurrent-settling step W_lat (TOGGLED) ---
        use_learned_settling_step=settle_on,
        learned_settling_rounds=SETTLING_ROUNDS,
        learned_settling_temperature=SETTLING_TEMPERATURE,
        learned_settling_eta=SETTLING_ETA,
        learned_settling_elig_decay=SETTLING_ELIG_DECAY,
        learned_settling_n_action_classes=SETTLING_N_ACTION_CLASSES,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (verbatim from V3-EXQ-699)
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
# P1 bias-head REINFORCE training (verbatim from V3-EXQ-699)
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

    # ----- ARC-108 / MECH-450 learning diagnostics (accumulated across ALL phases;
    # the three-factor update runs on every waking tick via update_residue) -----
    lcg_delta_ts: List[float] = []
    lcg_w_chan_range_max = 0.0
    wlat_delta_ts: List[float] = []
    wlat_range_max = 0.0
    settling_active_ticks = 0
    settling_round_deltas: List[float] = []
    n_select_ticks = 0

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

            # MECH-450 settling diagnostics (read every select tick; armed only when ON).
            diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
            if bool(diag.get("learned_settling_active", False)):
                settling_active_ticks += 1
                rd = float(diag.get("learned_settling_round_delta", -1.0))
                if math.isfinite(rd) and rd >= 0.0:
                    settling_round_deltas.append(rd)

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
            # update_residue drives e3.post_action_update -> the ARC-108/MECH-450
            # three-factor learning fires here on EVERY waking tick (all phases).
            with torch.no_grad():
                resid_metrics = agent.update_residue(
                    harm_signal=float(_harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )
            # Capture the learning diagnostics surfaced by post_action_update.
            ldt = resid_metrics.get("e3_lcg_delta_t")
            if ldt is not None:
                lcg_delta_ts.append(float(ldt.item()))
            lwr = resid_metrics.get("e3_lcg_w_chan_range")
            if lwr is not None:
                lcg_w_chan_range_max = max(lcg_w_chan_range_max, float(lwr.item()))
            wdt = resid_metrics.get("e3_wlat_delta_t")
            if wdt is not None:
                wlat_delta_ts.append(float(wdt.item()))
            wr = resid_metrics.get("e3_wlat_range")
            if wr is not None:
                wlat_range_max = max(wlat_range_max, float(wr.item()))

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
    wlat_delta_t_std = float(statistics.pstdev(wlat_delta_ts)) if len(wlat_delta_ts) >= 2 else 0.0
    mean_settling_round_delta = (
        float(sum(settling_round_deltas) / len(settling_round_deltas))
        if settling_round_deltas else 0.0
    )
    settling_active_frac = (
        float(settling_active_ticks) / float(n_select_ticks) if n_select_ticks > 0 else 0.0
    )

    seed_class_axis_exercisable = bool(frac_pre_ge2 > FRAC_PRE_GE2_FLOOR)
    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )

    # Per-arm learning-engaged non-vacuity (only meaningful on the armed arms).
    lcg_moved = bool(lcg_w_chan_range_max > W_CHAN_RANGE_FLOOR)
    lcg_delta_nonflat = bool(lcg_delta_t_std > DELTA_T_STD_FLOOR)
    wlat_moved = bool(wlat_range_max > WLAT_RANGE_FLOOR)
    wlat_delta_nonflat = bool(wlat_delta_t_std > DELTA_T_STD_FLOOR)
    settling_moved_field = bool(mean_settling_round_delta > SETTLING_ROUND_DELTA_FLOOR)

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "lcg_on": bool(arm["lcg_on"]),
        "settle_on": bool(arm["settle_on"]),
        "noise_on": bool(arm["noise_on"]),
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
        # ----- ARC-108 / MECH-450 learning diagnostics -----
        "lcg_n_updates": int(len(lcg_delta_ts)),
        "lcg_delta_t_std": round(lcg_delta_t_std, 8),
        "lcg_w_chan_range_max": round(lcg_w_chan_range_max, 8),
        "lcg_moved": lcg_moved,
        "lcg_delta_nonflat": lcg_delta_nonflat,
        "wlat_n_updates": int(len(wlat_delta_ts)),
        "wlat_delta_t_std": round(wlat_delta_t_std, 8),
        "wlat_range_max": round(wlat_range_max, 8),
        "wlat_moved": wlat_moved,
        "wlat_delta_nonflat": wlat_delta_nonflat,
        "settling_active_frac": round(settling_active_frac, 6),
        "mean_settling_round_delta": round(mean_settling_round_delta, 8),
        "settling_moved_field": settling_moved_field,
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
            f"Arm {arm['arm_id']} ({arm['label']}) lcg_on={arm['lcg_on']} "
            f"settle_on={arm['settle_on']} noise_on={arm['noise_on']} "
            f"(P0={p0_episodes} ep e2-train, P1={p1_episodes} ep bias-train, "
            f"P2={p2_episodes} ep measure, steps_per_episode={steps_per_episode}, "
            f"dry_run={dry_run})",
            flush=True,
        )
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)
            row = _run_seed_arm(
                arm, s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
            )
            row["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm_id": arm["arm_id"],
                    "lcg_on": bool(arm["lcg_on"]),
                    "settle_on": bool(arm["settle_on"]),
                    "noise_on": bool(arm["noise_on"]),
                    "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
                    "use_f_eligibility_adaptive_floor": bool(USE_F_ELIGIBILITY_ADAPTIVE_FLOOR),
                    "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
                    "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
                    "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
                    "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
                    "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
                    "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
                    "modulatory_channel_route_source": str(MODULATORY_CHANNEL_ROUTE_SOURCE),
                    "use_candidate_rule_field": bool(USE_CANDIDATE_RULE_FIELD),
                    "use_dacc": bool(USE_DACC),
                    "learned_channel_gating_eta": float(LCG_ETA),
                    "learned_channel_asym_potentiation": float(LCG_ASYM_POTENTIATION),
                    "learned_channel_asym_depression": float(LCG_ASYM_DEPRESSION),
                    "learned_settling_rounds": int(SETTLING_ROUNDS),
                    "learned_settling_eta": float(SETTLING_ETA),
                    "noise_floor_alpha": float(NOISE_FLOOR_ALPHA if arm["noise_on"] else 0.1),
                    "env_kwargs": dict(ENV_KWARGS),
                    "sd056_weight": SD056_WEIGHT,
                    "lr_lpfc_bias": LR_LPFC_BIAS,
                    "p0_episodes": int(p0_episodes),
                    "p1_episodes": int(p1_episodes),
                    "p2_episodes": int(p2_episodes),
                    "steps_per_episode": int(steps_per_episode),
                },
                seed=s,
                script_path=script_path,
                rng_fully_reset=True,
                extra_ineligible_reasons=[
                    "online_e2_training_stateful_per_cell",
                    "p1_bias_head_reinforce_training_stateful_per_cell",
                    "learned_channel_gating_state_persists_across_episodes",
                ],
            )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    a0_rows = _arm_rows(arm_results, "A0_ENVELOPE_ONLY")
    a1_rows = _arm_rows(arm_results, "A1_LEARNED_GATING")
    a2_rows = _arm_rows(arm_results, "A2_LEARNED_SETTLING")
    a3_rows = _arm_rows(arm_results, "A3_BOTH")
    noise_rows = _arm_rows(arm_results, "ARM_NOISE")
    all_rows = a0_rows + a1_rows + a2_rows + a3_rows + noise_rows
    learning_rows = a1_rows + a2_rows + a3_rows

    def _maj(rows: List[Dict[str, Any]], pred) -> bool:
        return sum(1 for r in rows if pred(r)) >= MIN_SEEDS_FOR_PASS

    # ----- Non-vacuity preconditions (design sec 5.3) -----
    # (a) candidate pool divergent on a majority of seeds in ALL arms.
    pool_divergent = all(
        _maj(rows, lambda r: r["gapa_divergence"]) for rows in
        (a0_rows, a1_rows, a2_rows, a3_rows, noise_rows)
    )
    class_axis_ok = all(
        _maj(rows, lambda r: r["class_axis_exercisable"]) for rows in
        (a0_rows, a1_rows, a2_rows, a3_rows, noise_rows)
    )
    crf_matured = all(
        _maj(rows, lambda r: r["crf_differentiated"]) for rows in
        (a0_rows, a1_rows, a2_rows, a3_rows, noise_rows)
    )
    # (b) delta_t carries variance on the LCG / settling armed arms.
    lcg_delta_nonflat_ok = all(_maj(rows, lambda r: r["lcg_delta_nonflat"]) for rows in (a1_rows, a3_rows))
    wlat_delta_nonflat_ok = all(_maj(rows, lambda r: r["wlat_delta_nonflat"]) for rows in (a2_rows, a3_rows))
    # (c) the learned weights actually MOVED (eligibility trace credited).
    lcg_moved_ok = all(_maj(rows, lambda r: r["lcg_moved"]) for rows in (a1_rows, a3_rows))
    wlat_moved_ok = all(
        _maj(rows, lambda r: r["wlat_moved"] and r["settling_moved_field"])
        for rows in (a2_rows, a3_rows)
    )
    # (d) the matched-noise control VERIFIED-LIFTING above A0 (paired seeds).
    a0_ent = _by_seed(a0_rows, "committed_class_entropy_nats")
    noise_ent = _by_seed(noise_rows, "committed_class_entropy_nats")
    shared_n = sorted(set(a0_ent) & set(noise_ent))
    n_noise_lifts = sum(1 for s in shared_n if noise_ent[s] > a0_ent[s] + CONVERSION_MARGIN)
    noise_verified_lifting = bool(shared_n and n_noise_lifts >= MIN_SEEDS_FOR_PASS)

    preconditions_met = bool(
        pool_divergent and class_axis_ok and crf_matured
        and lcg_delta_nonflat_ok and wlat_delta_nonflat_ok
        and lcg_moved_ok and wlat_moved_ok and noise_verified_lifting
    )

    # ----- C1 (conversion): a learning arm strict-above BOTH A0 AND the noise control -----
    a1_ent = _by_seed(a1_rows, "committed_class_entropy_nats")
    a2_ent = _by_seed(a2_rows, "committed_class_entropy_nats")
    a3_ent = _by_seed(a3_rows, "committed_class_entropy_nats")

    def _converts(arm_ent: Dict[int, float]) -> Tuple[int, List[int]]:
        seeds_ok: List[int] = []
        sh = sorted(set(arm_ent) & set(a0_ent) & set(noise_ent))
        for s in sh:
            bar = max(a0_ent[s], noise_ent[s]) + CONVERSION_MARGIN
            if arm_ent[s] > bar:
                seeds_ok.append(s)
        return len(seeds_ok), seeds_ok

    n_c1_a1, c1_a1_seeds = _converts(a1_ent)
    n_c1_a2, c1_a2_seeds = _converts(a2_ent)
    n_c1_a3, c1_a3_seeds = _converts(a3_ent)
    c1_a1 = n_c1_a1 >= MIN_SEEDS_FOR_PASS
    c1_a2 = n_c1_a2 >= MIN_SEEDS_FOR_PASS
    c1_a3 = n_c1_a3 >= MIN_SEEDS_FOR_PASS
    c1_holds = bool(c1_a1 or c1_a2 or c1_a3)

    # ----- C2 (learning load-bearing): grows over training on the converting arms -----
    def _grows(rows: List[Dict[str, Any]]) -> Tuple[int, List[int]]:
        seeds_ok: List[int] = []
        for r in rows:
            if (
                r["committed_class_entropy_p2_second_half_nats"]
                > r["committed_class_entropy_p2_first_half_nats"] + GROWTH_MARGIN
            ):
                seeds_ok.append(int(r["seed"]))
        return len(seeds_ok), seeds_ok

    n_grow_a1, grow_a1_seeds = _grows(a1_rows)
    n_grow_a2, grow_a2_seeds = _grows(a2_rows)
    n_grow_a3, grow_a3_seeds = _grows(a3_rows)
    # weights-moved is already a precondition (lcg_moved_ok / wlat_moved_ok); C2 adds growth.
    c2_a1 = bool(c1_a1 and n_grow_a1 >= MIN_SEEDS_FOR_PASS)
    c2_a2 = bool(c1_a2 and n_grow_a2 >= MIN_SEEDS_FOR_PASS)
    c2_a3 = bool(c1_a3 and n_grow_a3 >= MIN_SEEDS_FOR_PASS)
    c2_holds = bool(c2_a1 or c2_a2 or c2_a3)

    # ----- Outcome map (design sec 5.4) -----
    if not preconditions_met:
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        per_claim = {c: "non_contributory" for c in CLAIM_IDS}
    elif not c1_holds:
        # preconditions met (learning engaged, weights moved, pool divergent) but
        # NO entropy lift over A0/noise -> genuine no-lift -> V4 full-loop scope bet.
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "no_lift_preconditions_met_escalate_v4_full_loop"
        per_claim = {c: "non_contributory" for c in CLAIM_IDS}
    elif not c2_holds:
        # converts but does NOT grow with training -> STATIC re-weighting; fold into
        # the arithmetic lever, do NOT credit learning.
        outcome = "PASS"
        overall_direction = "mixed"
        label = "static_reweighting_fold_into_arithmetic"
        per_claim = {
            "MECH-439": "supports",        # the conversion ceiling IS liftable
            "ARC-108": "mixed",            # converts but not via learning (static)
            "MECH-450": "mixed",
        }
    else:
        # C1 + C2 -> learned gating converts where arithmetic plateaus.
        outcome = "PASS"
        overall_direction = "supports"
        label = "learned_gating_converts_where_arithmetic_plateaus"
        # Per-arm attribution: ARC-108 (w_chan) supported if A1 (or A3) converts+grows;
        # MECH-450 (W_lat settling) supported if A2 converts+grows OR A3 adds lift over A1.
        a1_mean = _mean([r["committed_class_entropy_nats"] for r in a1_rows])
        a3_mean = _mean([r["committed_class_entropy_nats"] for r in a3_rows])
        arc108_supported = bool(c2_a1 or c2_a3)
        settling_adds = bool(c2_a2 or (c2_a3 and a3_mean > a1_mean + CONVERSION_MARGIN))
        per_claim = {
            "MECH-439": "supports",
            "ARC-108": "supports" if arc108_supported else "mixed",
            "MECH-450": "supports" if settling_adds else "mixed",
        }

    a0_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a0_rows])
    a1_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a1_rows])
    a2_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a2_rows])
    a3_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a3_rows])
    noise_mean_dv = _mean([r["committed_class_entropy_nats"] for r in noise_rows])

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "candidate_pool_divergent_all_arms",
                "kind": "readiness",
                "description": (
                    "consumed cand_world_summaries (e2.world_forward) per-candidate SPREAD "
                    "clears the floor on a majority of seeds in ALL arms (GAP-A non-vacuity). "
                    "Below floor => nothing to convert => substrate_not_ready_requeue."
                ),
                "control": "SD-056 e2 trained online in P0; candidate_summary_source=e2_world_forward",
                "measured": float(min([r["consumed_summary_pairwise_dist_mean"] for r in all_rows] or [0.0])),
                "threshold": float(CONSUMED_SPREAD_FLOOR),
                "met": bool(pool_divergent),
            },
            {
                "name": "committed_class_axis_exercisable_all_arms",
                "kind": "readiness",
                "description": (
                    "frac of P2 ticks with >= 2 candidate first-action classes exceeds floor "
                    "on a majority of seeds in ALL arms (the committed-class DV is exercisable)."
                ),
                "control": "SP-CEM multi-class candidate pool, all arms",
                "measured": float(min([r["frac_pre_ge2"] for r in all_rows] or [0.0])),
                "threshold": float(FRAC_PRE_GE2_FLOOR),
                "met": bool(class_axis_ok),
            },
            {
                "name": "delta_t_carries_variance_on_learning_arms",
                "kind": "readiness",
                "description": (
                    "the signed-RPE delta_t (= benefit_eval - harm_eval - V-hat_t) carries "
                    "cross-tick STD above floor on a majority of seeds on the LCG-armed "
                    "(A1, A3) and settling-armed (A2, A3) arms -- there is outcome variance "
                    "to learn from. Flat delta_t => substrate_not_ready_requeue (design 5.3)."
                ),
                "control": "lcg_delta_t_std / wlat_delta_t_std on the armed arms",
                "measured": float(min(
                    [r["lcg_delta_t_std"] for r in (a1_rows + a3_rows)]
                    + [r["wlat_delta_t_std"] for r in (a2_rows + a3_rows)] or [0.0]
                )),
                "threshold": float(DELTA_T_STD_FLOOR),
                "met": bool(lcg_delta_nonflat_ok and wlat_delta_nonflat_ok),
            },
            {
                "name": "learned_weights_moved_from_init_on_armed_arms",
                "kind": "readiness",
                "description": (
                    "w_chan range > floor on a majority of seeds on the LCG-armed arms "
                    "(A1, A3; softplus-unity init => range 0) AND W_lat range > floor AND "
                    "the settling MOVED the field (learned_settling_round_delta > floor) on "
                    "the settling-armed arms (A2, A3; zero init => range 0). Weights never "
                    "moving => eligibility never credited => substrate_not_ready_requeue."
                ),
                "control": "lcg_w_chan_range_max / wlat_range_max + settling_round_delta on the armed arms",
                "measured": float(min(
                    [r["lcg_w_chan_range_max"] for r in (a1_rows + a3_rows)]
                    + [r["wlat_range_max"] for r in (a2_rows + a3_rows)] or [0.0]
                )),
                "threshold": float(W_CHAN_RANGE_FLOOR),
                "met": bool(lcg_moved_ok and wlat_moved_ok),
            },
            {
                "name": "matched_noise_control_verified_lifting",
                "kind": "readiness",
                "description": (
                    "the ARM_NOISE matched-noise temperature control lifts committed-class "
                    "entropy strict-above A0 by margin on a majority of seeds -- the 569g/684 "
                    "lesson: a non-lifting noise control is an UNVERIFIED bar (C1 would be "
                    "trivially satisfiable). Control not lifting => substrate_not_ready_requeue "
                    "(re-tune the noise alpha)."
                ),
                "control": "ARM_NOISE committed-class entropy vs A0, paired seeds",
                "measured": float(n_noise_lifts),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "met": bool(noise_verified_lifting),
            },
        ],
        "criteria": [
            {
                "name": "C1_conversion_learning_arm_above_A0_and_noise",
                "load_bearing": True,
                "passed": bool(c1_holds),
            },
            {
                "name": "C2_learning_load_bearing_grows_with_training_and_weights_moved",
                "load_bearing": True,
                "passed": bool(c2_holds),
            },
        ],
        "criteria_non_degenerate": {
            "preconditions_met": bool(preconditions_met),
            "pool_divergent": bool(pool_divergent),
            "noise_verified_lifting": bool(noise_verified_lifting),
            "lcg_moved": bool(lcg_moved_ok),
            "wlat_moved": bool(wlat_moved_ok),
            "delta_t_nonflat": bool(lcg_delta_nonflat_ok and wlat_delta_nonflat_ok),
        },
    }

    total_seeds = len(ARMS) * len(seeds)
    total_completed = len(all_rows)

    return {
        "outcome": outcome,
        "overall_direction": overall_direction,
        "evidence_direction_per_claim": per_claim,
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
            "growth_margin": float(GROWTH_MARGIN),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "frac_pre_ge2_floor": float(FRAC_PRE_GE2_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "consumed_magnitude_ceil": float(CONSUMED_MAGNITUDE_CEIL),
            "delta_t_std_floor": float(DELTA_T_STD_FLOOR),
            "w_chan_range_floor": float(W_CHAN_RANGE_FLOOR),
            "wlat_range_floor": float(WLAT_RANGE_FLOOR),
            "settling_round_delta_floor": float(SETTLING_ROUND_DELTA_FLOOR),
            "crf_min_minted": int(CRF_MIN_MINTED),
            "crf_frac_active_floor": float(CRF_FRAC_ACTIVE_FLOOR),
            "noise_floor_alpha": float(NOISE_FLOOR_ALPHA),
            "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
            "use_f_eligibility_adaptive_floor": bool(USE_F_ELIGIBILITY_ADAPTIVE_FLOOR),
            "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
            "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
            "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
            "lr_lpfc_bias": float(LR_LPFC_BIAS),
            "sd056_weight": float(SD056_WEIGHT),
        },
        "acceptance_criteria": {
            "preconditions_met": preconditions_met,
            "pool_divergent": pool_divergent,
            "class_axis_exercisable": class_axis_ok,
            "crf_matured": crf_matured,
            "lcg_delta_nonflat": lcg_delta_nonflat_ok,
            "wlat_delta_nonflat": wlat_delta_nonflat_ok,
            "lcg_moved": lcg_moved_ok,
            "wlat_moved": wlat_moved_ok,
            "noise_verified_lifting": noise_verified_lifting,
            "n_noise_lifts_over_a0": int(n_noise_lifts),
            "C1_conversion": c1_holds,
            "C1_a1_w_chan_converts": c1_a1,
            "C1_a2_settling_converts": c1_a2,
            "C1_a3_both_converts": c1_a3,
            "C1_a1_n_seeds": int(n_c1_a1),
            "C1_a2_n_seeds": int(n_c1_a2),
            "C1_a3_n_seeds": int(n_c1_a3),
            "C2_learning_load_bearing": c2_holds,
            "C2_a1_grows": c2_a1,
            "C2_a2_grows": c2_a2,
            "C2_a3_grows": c2_a3,
            "C2_a1_n_grow_seeds": int(n_grow_a1),
            "C2_a2_n_grow_seeds": int(n_grow_a2),
            "C2_a3_n_grow_seeds": int(n_grow_a3),
            "mean_committed_class_entropy_a0": round(a0_mean_dv, 6),
            "mean_committed_class_entropy_a1": round(a1_mean_dv, 6),
            "mean_committed_class_entropy_a2": round(a2_mean_dv, 6),
            "mean_committed_class_entropy_a3": round(a3_mean_dv, 6),
            "mean_committed_class_entropy_noise": round(noise_mean_dv, 6),
            "C3_signed_vs_unsigned_rpe_ablation": "DEFERRED_TO_SUBSTRATE_FLAG",
        },
        "c3_deferral_note": (
            "C3 (signed-vs-unsigned-RPE ablation; falsifies B5) is DEFERRED: delta_t = "
            "R_t - V-hat_t is hard-wired in e3_selector.post_action_update with no runtime "
            "knob to substitute the unsigned ARC-016 variance. Routed to a small "
            "/implement-substrate flag (e.g. learned_channel_rpe_mode: signed|unsigned) + a "
            "follow-up arm (user-confirmed 2026-06-22). This run delivers C1 (conversion) + "
            "C2 (learning load-bearing) on the built substrate."
        ),
        "interpretation_grid": {
            "PASS_learned_gating_converts_where_arithmetic_plateaus": (
                "preconditions met AND C1 (a learning arm committed-class entropy strict-above "
                "BOTH A0 and the verified-lifting matched-noise control on >=2/3 seeds) AND C2 "
                "(the lift GROWS over training + the learned weights moved from init). The "
                "learned, signed-RPE-driven channel re-weighting (ARC-108 w_chan) and/or the "
                "learned settling step (MECH-450 W_lat) CONVERTS committed-action diversity "
                "where the pure-arithmetic envelope plateaus -> supports MECH-439 (the "
                "conversion ceiling is liftable by learning) + the learning claim(s) that "
                "converted. Route to /governance + /thought-digestion (mint the learned-gating "
                "claims). PROMOTES NOTHING here (substrate_ceiling / substrate_conditional)."
            ),
            "PASS_static_reweighting_fold_into_arithmetic": (
                "preconditions met AND C1 (converts) BUT NOT C2 (the lift is present from "
                "tick 0 and does NOT grow with training). The conversion is a STATIC "
                "re-weighting, not learning -> MECH-439 supports (ceiling liftable) but "
                "ARC-108 / MECH-450 are MIXED (the lift is not from learning); fold the "
                "winning static weights into the arithmetic lever, do NOT mint a learning claim."
            ),
            "FAIL_no_lift_preconditions_met_escalate_v4_full_loop": (
                "preconditions met (learning engaged, weights moved, pool divergent, noise "
                "control verified-lifting) BUT NO entropy lift over A0/noise on the learning "
                "arms. The genuine 'no lift' outcome -> the single collapsed arena is the "
                "binding constraint -> escalate to the V4 full BG-thalamo-cortical loop / "
                "loop-segregation (design sec 6). This is NOT a falsification of ARC-107 (the "
                "arithmetic envelope still holds) and does NOT promote learned gating. "
                "non_contributory for all claims."
            ),
            "FAIL_substrate_not_ready_requeue": (
                "a precondition is unmet: candidate pool not divergent, OR delta_t flat (no "
                "outcome variance), OR the learned weights never moved (eligibility never "
                "credited) / the settling never moved the field, OR the matched-noise control "
                "did not lift above A0 (unverified bar). The conversion question could NOT be "
                "measured -- NOT a falsification. Re-queue at an adequate substrate."
            ),
        },
        "arm_results": arm_results,
    }


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
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "evidence_direction_note": (
            f"V3-EXQ-700 ARC-108 sec-7 SELECTION 2x2 LEARNED-GATING FALSIFIER "
            f"(experiment_purpose=evidence; claim_ids=[MECH-439, ARC-108, MECH-450]). The next "
            f"MECH-439 (F-dominance) attack: does a LEARNED, signed-RPE-driven adjustment of E3 "
            f"per-channel selection weights (ARC-108 w_chan) and/or a learned recurrent "
            f"lateral-inhibition settling step (MECH-450 W_lat) CONVERT committed-action-class "
            f"diversity where the PURE-ARITHMETIC envelope plateaus? 2x2 (use_learned_channel_gating "
            f"{{OFF,ON}} x use_learned_settling_step {{OFF,ON}}) over a MATCHED stack: the landed "
            f"arithmetic envelope (use_f_eligibility_demotion + use_f_eligibility_adaptive_floor + "
            f"use_go_nogo_constitution + the modulatory-authority/top_k shortlist scaffold) is a "
            f"MATCHED CONSTANT on ALL arms (exactly as V3-EXQ-699); the ONLY toggled flags are the "
            f"two learned levers; ARM_NOISE adds a verified-lifting matched-noise temperature "
            f"control (the 569g/684 bar). PRIMARY DV = committed-action-class entropy. C1 "
            f"(conversion) = a learning arm strict-above BOTH A0 AND the verified-lifting noise "
            f"control by {CONVERSION_MARGIN} on >=2/3 seeds. C2 (learning load-bearing) = the lift "
            f"GROWS over training (second-half P2 entropy strict-above first-half) AND the learned "
            f"weights MOVED from init (w_chan via lcg_w_chan_range; W_lat via wlat_range + the "
            f"settling moved the field). C3 (signed-vs-unsigned-RPE ablation) DEFERRED to a "
            f"substrate flag (no runtime knob exists; user-confirmed). NON-VACUITY self-route "
            f"substrate_not_ready_requeue if the pool is not divergent / delta_t is flat / the "
            f"learned weights never move / the noise control does not lift. A preconditions-met "
            f"no-lift FAIL routes to the V4 full-loop scope bet -- it does NOT falsify ARC-107 and "
            f"does NOT promote learned gating. PROMOTES NOTHING until it scores (MECH-439 "
            f"substrate_ceiling; ARC-108/MECH-450 substrate_conditional -> promote/demote "
            f"suppressed). outcome={result['outcome']}; label={result['interpretation_label']}; "
            f"per_claim={result['evidence_direction_per_claim']}."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "2x2 (use_learned_channel_gating {OFF,ON} x use_learned_settling_step {OFF,ON}) + verified-lifting matched-noise control",
            "arms": "A0_ENVELOPE_ONLY / A1_LEARNED_GATING (w_chan) / A2_LEARNED_SETTLING (W_lat) / A3_BOTH / ARM_NOISE",
            "swept_variables": "use_learned_channel_gating x use_learned_settling_step (+ use_noise_floor on ARM_NOISE only)",
            "matched_constant_arithmetic_envelope": (
                "use_f_eligibility_demotion=True + use_f_eligibility_adaptive_floor=True (689e) + "
                "use_go_nogo_constitution=True (689g) + use_modulatory_selection_authority=True (643a) + "
                "use_modulatory_channel_routing (cand_world_summary) + top_k shortlist (k=3, 569i)"
            ),
            "matched_diversity_stack": (
                "MECH-341 stratified + use_dacc (MECH-260 perseveration No-Go feed) + use_gated_policy + "
                "use_lateral_pfc_analog (lateral_pfc_train_rule_bias_head=True, TRAINED in P1 REINFORCE) + "
                "SD-056 all levers + the matured/maintained CRF pool + use_candidate_rule_field"
            ),
            "primary_dv": "committed-action-class entropy (nats)",
            "phases": "P0 e2-train (CRF matures, learned gating ON) -> P1 frozen-encoder bias-head REINFORCE -> P2 e2+bias frozen, learned gating KEEPS adapting (binned first/second half for C2)",
            "learning_wiring": "w_chan/W_lat learn via e3.post_action_update driven by agent.update_residue every waking tick (all phases); use_habenula_decommit OFF",
            "c3_status": "DEFERRED to a substrate flag (learned_channel_rpe_mode: signed|unsigned) + follow-up arm",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "noise_floor_alpha_arm_noise": NOISE_FLOOR_ALPHA,
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-700 ARC-108 sec-7 learned-gating 2x2 falsifier (committed-class entropy)"
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
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"preconditions_met={result['acceptance_criteria']['preconditions_met']} "
        f"C1={result['acceptance_criteria']['C1_conversion']} "
        f"C2={result['acceptance_criteria']['C2_learning_load_bearing']} "
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
