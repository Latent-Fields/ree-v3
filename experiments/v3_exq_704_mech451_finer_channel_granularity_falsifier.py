#!/opt/local/bin/python3
"""
V3-EXQ-704 (EXP-0391) -- MECH-451 FINER-CHANNEL-GRANULARITY VALIDATION FALSIFIER.

SIBLING of the V3-EXQ-700 lineage on the SAME GAP-A-ready reef-bipartite foraging
substrate + the SAME landed arithmetic envelope as a MATCHED CONSTANT on all arms.
The ONLY change across arms is the score_bias channel GRANULARITY feeding the
ARC-108 learned-gating layer. NEW number (not a 700 letter): MECH-451 is a DIFFERENT
scientific question than the ARC-108/MECH-450 settling lineage (claim_ids=[MECH-451]).

THE QUESTION (MECH-451)
-----------------------
Is the E3 F-dominance conversion ceiling (MECH-439) REPRESENTATIONAL COMPRESSION of
the modulatory blend, or a genuine need for full per-loop competition (ARC-110)?
ARC-108 added a SINGLE global learned w_chan over the modulatory channels, but at the
_modulatory_accum site "score_bias" is ALREADY the COMPRESSED dACC+lPFC+OFC+MECH-295+
MECH-320+gated_policy blend (summed UPSTREAM in agent.py). A learner that can only
re-weight a pre-compressed blend cannot dissociate the control functions compression
fused. MECH-451 explodes that single slot into SEVERAL separately-learnable FINER
channels, each with its OWN learned w_chan_finer entry trained by the SAME ARC-108
signed-RPE three-factor rule, keeping ONE shared arena (NOT ARC-110 per-loop
competition). The cheap V3 rung BETWEEN ARC-108's one global weight vector and
ARC-110's V4 segregated loops -- a PASS pre-empts the expensive V4 build.

THE 4 ARMS (settling W_lat OFF on ALL arms -- isolate the channel-granularity factor;
all carry the landed arithmetic envelope as a MATCHED CONSTANT: use_f_eligibility_
demotion + adaptive_floor + go_nogo + modulatory-authority/top_k shortlist k=3):
  A0_ENVELOPE_ONLY : conversion baseline -- no learned gating at all.
  A1_GLOBAL_WCHAN  : use_learned_channel_gating=True -- the SINGLE global w_chan over
                     the compressed blend = the V3-EXQ-700 A1 reference AND the ARC-106
                     collapse-to-blend ablation (the load-bearing ablation: if A1
                     reproduces A2's lift, the decomposition is NOT load-bearing).
  A2_FINER_CHANNELS: use_finer_channel_gating=True (MECH-451) -- score_bias exposed as
                     >=3 separately-learnable finer channels, each with its own
                     w_chan_finer entry; SAME signed-RPE three-factor trained.
  ARM_NOISE        : verified-lifting matched-noise control -- a SAME-LAYER null at the
                     finer-channel-gating layer: use_finer_channel_gating=True with
                     learned_channel_gating_eta=0.0 (the finer update is a no-op) and a
                     FROZEN magnitude-matched random w_chan_finer injected once into
                     agent.e3.w_chan_finer. So ARM_NOISE re-weights the SAME finer
                     channels by RANDOM (not learned) structure at the SAME layer A2
                     learns -- the better-layered bar (a lift is attributable to learned
                     STRUCTURE only if it beats unstructured random structure at the same
                     layer; the 569g/700b decoupled-null lesson: a null that does not
                     verify-lift makes the strict-above bar meaningless).
6 seeds. PRIMARY DV = committed-action-class entropy (nats), measured over P2.
claim_ids = [MECH-451]. experiment_purpose = evidence.

PRE-REGISTERED OUTCOME MAP (decisive either way)
------------------------------------------------
  PASS / supports MECH-451 (REPRESENTATIONAL COMPRESSION is the binding constraint):
    A2_FINER_CHANNELS lifts committed-action-class entropy strict-above BOTH A0 AND
    A1_GLOBAL_WCHAN AND the verified-lifting noise control, converting >=1 previously-
    F-dominated NON-MOTOR control function to committed authority, on a strict-majority
    of DIVERGENT seeds. This PRE-EMPTS the V4 ARC-110 segregated-loop build.

  WEAKENED / route-to-ARC-110 (compression is NOT the binding constraint):
    A2 finer-weights MOVE (fcg_w_chan_finer_range/_std > floor = dissociable channels)
    but produce NO committed-conversion lift beyond A1_GLOBAL_WCHAN. This is POSITIVE
    evidence FOR ARC-110 (full per-loop competition implicated), NOT against ARC-108 --
    the learning works, the single-arena channel exposure is insufficient. Route to the
    v4_loop_segregation build.

NON-VACUITY READINESS GATES (self-route substrate_not_ready_requeue, NEVER a false
weakens):
  (1) finer channels DISSOCIABLE: A2 fcg_w_chan_finer_range/_std > floor -- finer
      w_chan_finer entries that move IDENTICALLY are the compressed blend re-labelled.
  (2) candidate pool DIVERGENT: GAP-A guard cand_world_pairwise_dist > floor.
  (3) signed delta_t NON-FLAT + finer w_chan_finer entries actually MOVE.
  (4) the verified-lifting noise control (ARM_NOISE) actually raises entropy above A0
      on a strict-majority of divergent seeds (the 569g/700b lesson).
  (5) ARM_NOISE frozen-random w_chan_finer magnitude MATCHED to the median A2 learned
      w_chan_finer range (the legitimate one-time re-tune knob = FCG_NOISE_SCALE).

ARC-106 cargo-cult guard: the load-bearing ablation IS A1_GLOBAL_WCHAN (collapse-to-
blend = one global w_chan over the sum). If A1 reproduces A2's lift, the decomposition
is NOT load-bearing. A1 *is* an EXP-0391 arm and sits IN the C1 strict-above bar.

See REE_assembly/docs/architecture/mech_451_finer_channel_granularity.md,
    REE_assembly/evidence/planning/manual_proposals.v1.json (EXP-0391),
    REE_assembly/docs/architecture/sd_v4_loop_segregation.md (ARC-110 -- pre-empted on PASS),
    experiments/v3_exq_700c_arc108_sec7_learned_gating_settling_samelayer_null.py (scaffold),
    tests/contracts/test_mech451_finer_channel_gating.py.
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_704_mech451_finer_channel_granularity_falsifier"
QUEUE_ID = "V3-EXQ-704"
BACKLOG_ID = "EVB-0391"   # provenance only; manual_proposals proposal_id EXP-0391
CLAIM_IDS: List[str] = ["MECH-451"]
EXPERIMENT_PURPOSE = "evidence"

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

# ----- SAME-LAYER null at the FINER-CHANNEL-GATING layer (ARM_NOISE) -----
# ARM_NOISE injects a FROZEN, magnitude-matched, random-STRUCTURE perturbation at the
# finer-channel-gating layer (the exact layer the MECH-451 finer decomposition acts on):
# the finer-gating code path runs (use_finer_channel_gating=True) but the three-factor
# update is a no-op (learned_channel_gating_eta=0.0) and w_chan_finer is seeded once to a
# random vector around the softplus-unity init. NOT a policy temperature null (that was
# 700b's decoupled null) and NOT a settling-field null (700c -- settling is OFF here).
FCG_NOISE_SCALE = 0.5            # pre-registered scale of the frozen random w_chan_finer (magnitude-matched bar)
FCG_NOISE_SEED_OFFSET = 100000   # reproducible-per-cell random w_chan_finer distinct from the cell seed's other RNG
FCG_NOISE_MAGNITUDE_MATCH_LO = 0.25  # ARM_NOISE w_chan_finer range must be within [LO,HI] x median A2 learned range
FCG_NOISE_MAGNITUDE_MATCH_HI = 4.0

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


# The 4 arms. The ONLY swept config is (lcg_on, finer_on, fcg_noise). Settling OFF on
# ALL arms. A0/A1/A2 carry NO fcg_noise (reusable / self-mint eligible). ARM_NOISE is the
# changed arm: a SAME-LAYER finer-gating null (finer_on=True, fcg_noise=True -> eta=0 +
# a frozen magnitude-matched random w_chan_finer).
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "A0_ENVELOPE_ONLY",
        "label": "envelope_only_control_no_learned_gating",
        "lcg_on": False,
        "finer_on": False,
        "fcg_noise": False,
    },
    {
        "arm_id": "A1_GLOBAL_WCHAN",
        "label": "single_global_w_chan_over_compressed_blend_arc106_collapse_ablation",
        "lcg_on": True,
        "finer_on": False,
        "fcg_noise": False,
    },
    {
        "arm_id": "A2_FINER_CHANNELS",
        "label": "mech451_finer_separately_learnable_channels_focus_lever",
        "lcg_on": False,
        "finer_on": True,
        "fcg_noise": False,
    },
    {
        "arm_id": "ARM_NOISE",
        "label": "matched_magnitude_same_layer_finer_gating_null_frozen_random_w_chan_finer",
        "lcg_on": False,
        "finer_on": True,
        "fcg_noise": True,
    },
]

# A0/A1/A2 are reuse-ELIGIBLE (self-mint as we go); ARM_NOISE is the changed arm (never reused).
REUSABLE_ARM_IDS_LOCAL = ("A0_ENVELOPE_ONLY", "A1_GLOBAL_WCHAN", "A2_FINER_CHANNELS")


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
    a future mint, so they match by construction (settling OFF on all arms)."""
    return {
        "arm_id": arm["arm_id"],
        "lcg_on": bool(arm["lcg_on"]),
        "finer_on": bool(arm["finer_on"]),
        "fcg_noise": bool(arm.get("fcg_noise", False)),
        "use_learned_settling_step": False,
        "learned_channel_gating_eta": 0.0 if arm.get("fcg_noise", False) else LCG_ETA,
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
    SD-056, CRF, trained lateral_pfc bias head, use_dacc) are MATCHED CONSTANTS on all
    arms. Settling W_lat is OFF on EVERY arm. The ONLY toggled flags are the channel-
    granularity levers (use_learned_channel_gating for A1 vs use_finer_channel_gating for
    A2/ARM_NOISE) and -- on ARM_NOISE only -- learned_channel_gating_eta=0.0 (so the
    once-seeded random w_chan_finer is a frozen magnitude-matched random re-weighting of
    the SAME finer channels A2 learns)."""
    lcg_on = bool(arm["lcg_on"])
    finer_on = bool(arm["finer_on"])
    fcg_noise = bool(arm.get("fcg_noise", False))
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
        # --- ARC-108 JOB-1 step-1: SINGLE global learned per-channel gating (A1 only) ---
        use_learned_channel_gating=lcg_on,
        # --- MECH-451: FINER separately-learnable channels (A2 + ARM_NOISE) ---
        use_finer_channel_gating=finer_on,
        # Shared three-factor knobs (used by BOTH the global w_chan AND the finer
        # w_chan_finer path). On ARM_NOISE eta=0.0 freezes the once-seeded random
        # w_chan_finer (the same-layer matched-noise control).
        learned_channel_gating_eta=(0.0 if fcg_noise else LCG_ETA),
        learned_channel_gating_elig_decay=LCG_ELIG_DECAY,
        learned_channel_value_baseline_beta=LCG_VALUE_BASELINE_BETA,
        learned_channel_asym_potentiation=LCG_ASYM_POTENTIATION,
        learned_channel_asym_depression=LCG_ASYM_DEPRESSION,
        # signed RPE (no C3 unsigned ablation in this experiment).
        learned_channel_rpe_mode="signed",
        # --- MECH-450 recurrent settling: OFF on ALL arms (isolate channel granularity) ---
        use_learned_settling_step=False,
    )
    return REEAgent(cfg)


def _inject_fcg_noise(agent: REEAgent, seed: int) -> float:
    """SAME-LAYER null injection at the finer-channel-gating layer: seed the FROZEN
    random w_chan_finer once.

    Writes a magnitude-matched random vector (around the softplus-unity init) into
    agent.e3.w_chan_finer (the register_buffer the MECH-451 finer recompose reads). This
    is ONLY meaningful when learned_channel_gating_eta=0.0 (w_chan_finer frozen) -- which
    _make_agent guarantees on the noise arm -- so the random vector is a stable random
    per-channel re-weighting of the SAME finer channels A2 learns, NOT a learned object.

    Reproducible per cell via FCG_NOISE_SEED_OFFSET (distinct from the cell seed's other
    RNG). Returns the injected w_chan_finer range (max - min) for magnitude logging.
    """
    buf = agent.e3.w_chan_finer
    n = int(buf.shape[0])
    g = torch.Generator(device="cpu").manual_seed(int(seed) + FCG_NOISE_SEED_OFFSET)
    w = _FCG_W_INIT + torch.randn((n,), generator=g) * FCG_NOISE_SCALE
    buf.copy_(w.to(dtype=buf.dtype, device=buf.device))
    return float((w.max() - w.min()).item())


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

    # SAME-LAYER null: inject the frozen magnitude-matched random w_chan_finer ONCE on
    # the noise arm, after the agent is built and before training. eta=0 keeps it frozen
    # across all P0/P1/P2 episodes (w_chan_finer is not reset by agent.reset()).
    fcg_noise_injected_range = 0.0
    if arm.get("fcg_noise"):
        fcg_noise_injected_range = _inject_fcg_noise(agent, seed)

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

    # On ARM_NOISE the random w_chan_finer is FROZEN (eta=0), so the emitted range reflects
    # the INJECTED random structure. Use the injected magnitude as the authoritative range
    # for the magnitude match if no update surfaced one.
    if arm.get("fcg_noise") and fcg_noise_injected_range > fcg_w_chan_finer_range_max:
        fcg_w_chan_finer_range_max = fcg_noise_injected_range

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

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "lcg_on": bool(arm["lcg_on"]),
        "finer_on": bool(arm["finer_on"]),
        "fcg_noise": bool(arm.get("fcg_noise", False)),
        "fcg_noise_injected_range": round(float(fcg_noise_injected_range), 8),
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
            f"Arm {arm['arm_id']} ({arm['label']}) lcg_on={arm['lcg_on']} "
            f"finer_on={arm['finer_on']} fcg_noise={arm.get('fcg_noise', False)} "
            f"(P0={p0_episodes} ep e2-train, P1={p1_episodes} ep bias-train, "
            f"P2={p2_episodes} ep measure, steps_per_episode={steps_per_episode}, "
            f"dry_run={dry_run})",
            flush=True,
        )
        reusable = arm["arm_id"] in REUSABLE_ARM_IDS_LOCAL
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)

            # ----- ARM-REUSE (consumer side), gated + safe-by-default -----
            # Only the 3 unchanged arms (A0/A1/A2) are reuse-eligible, and only IFF a mint
            # is cited. With REUSE_BASELINE_FROM=None (the default) this is skipped and
            # every arm runs fresh (the false cache-miss is free).
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
                    arm, s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                )

            # Per-cell fingerprint. The 3 unchanged arms emit a REUSE-ELIGIBLE fingerprint
            # (MINT-AS-YOU-GO: config slice declared, include_driver_script_in_hash=False)
            # so a future sibling could reuse them. ARM_NOISE is the changed arm with a
            # frozen-random injection -- never reused.
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
                    config_slice={
                        **_arm_config_slice(
                            arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                        ),
                        "fcg_noise_scale": float(FCG_NOISE_SCALE),
                        "fcg_noise_seed_offset": int(FCG_NOISE_SEED_OFFSET),
                    },
                    seed=s,
                    script_path=script_path,
                    rng_fully_reset=True,
                    extra_ineligible_reasons=[
                        "fcg_noise_frozen_random_w_chan_finer_not_a_reusable_baseline",
                    ],
                )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    a0_rows = _arm_rows(arm_results, "A0_ENVELOPE_ONLY")
    a1_rows = _arm_rows(arm_results, "A1_GLOBAL_WCHAN")
    a2_rows = _arm_rows(arm_results, "A2_FINER_CHANNELS")
    noise_rows = _arm_rows(arm_results, "ARM_NOISE")
    all_rows = a0_rows + a1_rows + a2_rows + noise_rows

    def _maj(rows: List[Dict[str, Any]], pred) -> bool:
        return sum(1 for r in rows if pred(r)) >= MIN_SEEDS_FOR_PASS

    a0_ent = _by_seed(a0_rows, "committed_class_entropy_nats")
    a1_ent = _by_seed(a1_rows, "committed_class_entropy_nats")
    a2_ent = _by_seed(a2_rows, "committed_class_entropy_nats")
    noise_ent = _by_seed(noise_rows, "committed_class_entropy_nats")

    a0_gap = _gap_by_seed(a0_rows)
    a1_gap = _gap_by_seed(a1_rows)
    a2_gap = _gap_by_seed(a2_rows)
    noise_gap = _gap_by_seed(noise_rows)

    # ----- Per-seed-divergent gating -----
    # The PRIMARY divergence set = seeds whose pool is divergent on the C1 comparison
    # arms (A0 + A1 + A2 + ARM_NOISE all divergent). A degenerate cell self-excludes.
    primary_div = [
        s for s in sorted(set(a0_gap) & set(a1_gap) & set(a2_gap) & set(noise_gap))
        if a0_gap.get(s) and a1_gap.get(s) and a2_gap.get(s) and noise_gap.get(s)
    ]
    n_primary_div = len(primary_div)
    enough_divergent = n_primary_div >= MIN_DIVERGENT_SEEDS

    # ----- Precondition (4): the SAME-LAYER matched-noise control VERIFIED-LIFTING above A0 -----
    n_noise_lifts = sum(
        1 for s in primary_div if noise_ent.get(s, 0.0) > a0_ent.get(s, 0.0) + CONVERSION_MARGIN
    )
    noise_verified_lifting = bool(enough_divergent and _div_pass(n_noise_lifts, n_primary_div))

    # ----- Precondition (5): fcg_noise_magnitude_matched -----
    # ARM_NOISE's median frozen-random w_chan_finer range must be within [LO,HI] x the
    # median A2 LEARNED w_chan_finer range -- so the random null is magnitude-matched to
    # the structure A2 learns. An unmatched magnitude => substrate_not_ready_requeue
    # (re-tune the FINER-NOISE scale FCG_NOISE_SCALE, the legitimate ONE-time knob).
    noise_fcg_ranges = [float(r["fcg_w_chan_finer_range_max"]) for r in noise_rows]
    a2_fcg_ranges = [float(r["fcg_w_chan_finer_range_max"]) for r in a2_rows]
    median_noise_fcg_range = _median(noise_fcg_ranges)
    median_a2_fcg_range = _median(a2_fcg_ranges)
    if median_a2_fcg_range > 0.0:
        fcg_noise_magnitude_ratio = median_noise_fcg_range / median_a2_fcg_range
    else:
        fcg_noise_magnitude_ratio = 0.0
    fcg_noise_magnitude_matched = bool(
        median_a2_fcg_range > 0.0
        and FCG_NOISE_MAGNITUDE_MATCH_LO <= fcg_noise_magnitude_ratio <= FCG_NOISE_MAGNITUDE_MATCH_HI
    )

    # ----- Precondition (1)/(3): A2 finer channels DISSOCIABLE + delta_t non-flat -----
    # The KEY MECH-451 non-vacuity: the finer w_chan_finer entries MOVED + carry
    # cross-channel range above floor (finer entries that move identically are the
    # compressed blend re-labelled), and the teaching signal carried variance.
    fcg_dissociable_ok = _maj(a2_rows, lambda r: r["fcg_moved"])
    fcg_delta_nonflat_ok = _maj(a2_rows, lambda r: r["fcg_delta_nonflat"])
    # ----- A1 load-bearing ablation non-vacuity: the global w_chan also genuinely learned -----
    lcg_moved_ok = _maj(a1_rows, lambda r: r["lcg_moved"])
    lcg_delta_nonflat_ok = _maj(a1_rows, lambda r: r["lcg_delta_nonflat"])

    # CRF maturity readiness (matched constant; on a majority of seeds across all arms).
    crf_matured = all(
        _maj(rows, lambda r: r["crf_differentiated"]) for rows in
        (a0_rows, a1_rows, a2_rows, noise_rows)
    )

    preconditions_met = bool(
        enough_divergent
        and noise_verified_lifting
        and fcg_noise_magnitude_matched
        and fcg_dissociable_ok and fcg_delta_nonflat_ok
        and lcg_moved_ok and lcg_delta_nonflat_ok
        and crf_matured
    )

    # ----- COUNT-shaped restatements of the majority-of-seeds preconditions -----
    # The indexer RECOMPUTES interpretation.preconditions[].met from the reported
    # (measured, threshold) pair and treats the recompute as AUTHORITATIVE over the
    # author's `met` (build_experiment_indexes._precondition_unmet).
    #
    # `fcg_*_ok` / `lcg_*_ok` are k-of-n COUNTS over per-seed predicates
    # (`_maj` == ">= MIN_SEEDS_FOR_PASS seeds satisfy pred"). A min over the
    # underlying statistic cannot reproduce them: min(...) over rows is strictly
    # HARSHER than "a majority of seeds" (2-of-3 seeds can clear the floor while the
    # min does not, flagging a sound run precondition_unmet), and where `met` is a
    # conjunction of counts over two DIFFERENT statistics on two DIFFERENT arms, no
    # single pooled statistic exists at all. Reported instead as the satisfying-seed
    # COUNT minimised over arm groups vs MIN_SEEDS_FOR_PASS with comparator ">=" --
    # exact, because min(counts) >= k iff every count >= k. The old min numbers are
    # preserved on each entry as NON-BOUND diagnostic keys (extra keys are ignored by
    # the recompute), so no information is lost.
    def _n_seeds(rows: List[Dict[str, Any]], pred) -> int:
        return sum(1 for r in rows if pred(r))

    n_delta_nonflat_min_arm = min([
        _n_seeds(a2_rows, lambda r: r["fcg_delta_nonflat"]),
        _n_seeds(a1_rows, lambda r: r["lcg_delta_nonflat"]),
    ])
    n_weights_moved_min_arm = min([
        _n_seeds(a2_rows, lambda r: r["fcg_moved"]),
        _n_seeds(a1_rows, lambda r: r["lcg_moved"]),
    ])
    n_fcg_dissociable = _n_seeds(a2_rows, lambda r: r["fcg_moved"])
    # The fraction leg of `noise_verified_lifting`, split out so it is recomputable
    # on its own bounds -- see the entry below.
    noise_lift_needed = max(
        MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1)))
    )
    noise_lift_fraction_ok = bool(n_noise_lifts >= noise_lift_needed)

    # ----- C1 (conversion): A2 strict-above BOTH A0 AND A1 AND the noise control, on the
    # per-seed divergent seeds. A1 IN THE BAR is the ARC-106 cargo-cult guard (collapse-
    # to-blend ablation: if the single global w_chan reproduces the lift, the finer
    # decomposition is NOT load-bearing). -----
    a2_div = [
        s for s in sorted(set(a0_gap) & set(a1_gap) & set(a2_gap) & set(noise_gap))
        if a0_gap.get(s) and a1_gap.get(s) and a2_gap.get(s) and noise_gap.get(s)
    ]
    c1_a2_seeds: List[int] = []
    for s in a2_div:
        bar = max(a0_ent.get(s, 0.0), a1_ent.get(s, 0.0), noise_ent.get(s, 0.0)) + CONVERSION_MARGIN
        if a2_ent.get(s, 0.0) > bar:
            c1_a2_seeds.append(s)
    n_c1_a2 = len(c1_a2_seeds)
    n_div_a2 = len(a2_div)
    c1_holds = _div_pass(n_c1_a2, n_div_a2)

    # ----- C2 (learning load-bearing): the A2 lift GROWS over training (second-half P2
    # entropy strict-above first-half) on the divergent seeds where A2 converted. -----
    grow_a2_seeds: List[int] = []
    for r in a2_rows:
        s = int(r["seed"])
        if s not in a2_div:
            continue
        if (
            r["committed_class_entropy_p2_second_half_nats"]
            > r["committed_class_entropy_p2_first_half_nats"] + GROWTH_MARGIN
        ):
            grow_a2_seeds.append(s)
    n_grow_a2 = len(grow_a2_seeds)
    c2_holds = bool(c1_holds and _div_pass(n_grow_a2, n_div_a2))

    # ----- A1-reproduces-A2 (ARC-106 ablation readout) -----
    # Diagnostic: did the single global w_chan ALSO convert (strict-above A0 + noise)?
    # If A1 reproduces the lift, the decomposition is not load-bearing (recorded; the C1
    # bar already includes A1, so c1_holds REQUIRES A2 > A1).
    a1_div = [
        s for s in sorted(set(a0_gap) & set(a1_gap) & set(noise_gap))
        if a0_gap.get(s) and a1_gap.get(s) and noise_gap.get(s)
    ]
    a1_converts_seeds = [
        s for s in a1_div
        if a1_ent.get(s, 0.0) > max(a0_ent.get(s, 0.0), noise_ent.get(s, 0.0)) + CONVERSION_MARGIN
    ]
    a1_reproduces = _div_pass(len(a1_converts_seeds), len(a1_div))

    # ----- Outcome map (decisive either way) -----
    if not preconditions_met:
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "MECH-451 conversion could NOT be measured: a precondition is unmet "
            "(too few divergent seeds / finer channels not dissociable / delta_t flat / "
            "weights never moved / matched-noise control did not verify-lift / "
            "fcg-noise magnitude unmatched). NOT a falsification."
        )
        per_claim = {"MECH-451": "non_contributory"}
    elif c1_holds:
        # A2 strict-above A0 AND A1 AND noise on a strict-majority of divergent seeds:
        # representational compression IS the binding constraint -> supports MECH-451,
        # pre-empts the V4 ARC-110 segregated-loop build. C2 distinguishes learned
        # conversion (grows over training) from a static finer reweighting.
        outcome = "PASS"
        overall_direction = "supports"
        non_degenerate = True
        degeneracy_reason = ""
        if c2_holds:
            label = "finer_channels_convert_representational_compression_confirmed_preempt_arc110"
        else:
            label = "finer_static_reweighting_converts_compression_confirmed_fold_in"
        per_claim = {"MECH-451": "supports"}
    else:
        # preconditions met (finer channels DISSOCIABLE + verified-lifting noise bar +
        # magnitude-matched + A1 learned) BUT A2 does NOT lift strict-above A0/A1/noise:
        # the finer weights MOVE but produce no committed-conversion lift beyond the
        # single global w_chan -> compression is NOT the binding constraint -> POSITIVE
        # evidence FOR ARC-110 (full per-loop competition implicated), NOT against ARC-108.
        outcome = "FAIL"
        overall_direction = "weakens"
        non_degenerate = True
        degeneracy_reason = ""
        label = "finer_weights_move_no_lift_route_to_arc110_v4_loop_segregation"
        per_claim = {"MECH-451": "weakens"}

    a0_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a0_rows])
    a1_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a1_rows])
    a2_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a2_rows])
    noise_mean_dv = _mean([r["committed_class_entropy_nats"] for r in noise_rows])

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "enough_divergent_seeds",
                "kind": "readiness",
                "description": (
                    "the number of seeds whose candidate pool is DIVERGENT on ALL C1 "
                    "comparison arms (A0 + A1 + A2 + ARM_NOISE) is >= MIN_DIVERGENT_SEEDS. "
                    "Per-seed-divergent gating: a degenerate cell self-excludes. Too few "
                    "divergent seeds => substrate_not_ready_requeue (pool too collapsed to "
                    "test conversion)."
                ),
                "control": "consumed cand_world_summary pairwise spread > floor (GAP-A); per-seed",
                # COUNT-shaped and already correct; comparator declared so the
                # recompute mirrors the source (`n_primary_div >= MIN_DIVERGENT_SEEDS`)
                # rather than taking the default.
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(enough_divergent),
            },
            {
                "name": "finer_channels_dissociable",
                "kind": "readiness",
                "description": (
                    "MECH-451 NON-DEGENERACY gate (1): the A2 finer w_chan_finer entries "
                    "MOVED from the softplus-unity init AND carry cross-channel RANGE above "
                    "floor on a strict-majority of seeds -- finer entries that move "
                    "IDENTICALLY (range ~0) are the compressed blend re-labelled, NOT a "
                    "genuine decomposition. Below floor => substrate_not_ready_requeue."
                ),
                "control": "A2 fcg_w_chan_finer_range_max vs floor (softplus-unity init => range 0)",
                # COUNT-shaped: `met` is `_maj(a2_rows, fcg_moved)`, a k-of-n count of
                # A2 seeds whose finer range clears the floor. Single-leg predicate
                # (`fcg_moved` == `fcg_w_chan_finer_range_max > W_CHAN_FINER_RANGE_FLOOR`),
                # but still a count, so min(range) over A2 is strictly harsher than the
                # shipped bar and would flag a sound 2-of-3 run precondition_unmet.
                "measured": float(n_fcg_dissociable),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_a2_fcg_range": float(
                    min([r["fcg_w_chan_finer_range_max"] for r in a2_rows] or [0.0])
                ),
                "observed_w_chan_finer_range_floor": float(W_CHAN_FINER_RANGE_FLOOR),
                "met": bool(fcg_dissociable_ok),
            },
            {
                "name": "delta_t_carries_variance_on_armed_arms",
                "kind": "readiness",
                "description": (
                    "the signed-RPE delta_t (= benefit_eval - harm_eval - V-hat_t) carries "
                    "cross-tick STD above floor on a majority of seeds on the A2 finer arm "
                    "AND the A1 global-w_chan arm -- there is outcome variance to learn from. "
                    "Flat => substrate_not_ready_requeue."
                ),
                "control": "fcg_delta_t_std (A2) + lcg_delta_t_std (A1) on the armed arms",
                # COUNT-shaped: `met` is the conjunction of two per-arm majority counts
                # over two DIFFERENT statistics (fcg_delta_t_std on A2, lcg_delta_t_std
                # on A1), so no single pooled min reproduces it. Counted per arm group
                # against its own leg, then minimised.
                "measured": float(n_delta_nonflat_min_arm),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_delta_t_std": float(min(
                    [r["fcg_delta_t_std"] for r in a2_rows]
                    + [r["lcg_delta_t_std"] for r in a1_rows] or [0.0]
                )),
                "observed_delta_t_std_floor": float(DELTA_T_STD_FLOOR),
                "met": bool(fcg_delta_nonflat_ok and lcg_delta_nonflat_ok),
            },
            {
                "name": "learned_weights_moved_from_init_on_armed_arms",
                "kind": "readiness",
                "description": (
                    "A2 finer w_chan_finer range > floor (softplus-unity init => range 0) "
                    "AND A1 global w_chan range > floor (the load-bearing collapse-to-blend "
                    "ablation must itself be a real learner). Weights never moving => "
                    "eligibility never credited => substrate_not_ready_requeue."
                ),
                "control": "A2 fcg_w_chan_finer_range_max + A1 lcg_w_chan_range_max",
                # COUNT-shaped: `met` is the conjunction of two per-arm majority counts
                # over two DIFFERENT statistics against two DIFFERENT floors
                # (fcg_w_chan_finer_range_max vs W_CHAN_FINER_RANGE_FLOOR on A2,
                # lcg_w_chan_range_max vs W_CHAN_RANGE_FLOOR on A1), so no single pooled
                # min against one floor reproduces it.
                "measured": float(n_weights_moved_min_arm),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_learned_weight_range": float(min(
                    [r["fcg_w_chan_finer_range_max"] for r in a2_rows]
                    + [r["lcg_w_chan_range_max"] for r in a1_rows] or [0.0]
                )),
                "observed_w_chan_finer_range_floor": float(W_CHAN_FINER_RANGE_FLOOR),
                "observed_w_chan_range_floor": float(W_CHAN_RANGE_FLOOR),
                "met": bool(fcg_dissociable_ok and lcg_moved_ok),
            },
            {
                "name": "matched_noise_control_verified_lifting",
                "kind": "readiness",
                "description": (
                    "the SAME-LAYER finer-gating null (ARM_NOISE: frozen magnitude-matched "
                    "random w_chan_finer with eta=0) lifts committed-class entropy strict-"
                    "above A0 by margin on a strict-majority of DIVERGENT seeds. A null that "
                    "does NOT verify-lift makes the strict-above bar meaningless (the "
                    "569g/700b lesson). Control not verify-lifting => substrate_not_ready_"
                    "requeue (re-tune FCG_NOISE_SCALE, the correct layer)."
                ),
                "control": "ARM_NOISE committed-class entropy vs A0, divergent seeds, paired",
                # COUNT-shaped. `noise_verified_lifting` is `enough_divergent and
                # _div_pass(...)`, a CONJUNCTION of (i) n_primary_div >=
                # MIN_DIVERGENT_SEEDS and (ii) n_noise_lifts >= noise_lift_needed.
                # Only (ii) is expressible on this entry's bounds, so with the old
                # declaration the recompute could say "met" on a run with 2 divergent
                # seeds that both lifted while the shipped predicate said unmet. Leg
                # (i) is ALREADY declared as its own recomputable precondition
                # (`enough_divergent_seeds` above), so this entry now carries leg (ii)
                # alone -- the same split as SD-068 c7d398c2e0. The conjunction is
                # unchanged and still routes the label via `noise_verified_lifting` /
                # preconditions_met, which are computed from the underlying booleans,
                # not from these entries.
                "measured": float(n_noise_lifts),
                "threshold": float(noise_lift_needed),
                "comparator": ">=",
                "direction": "lower",
                "observed_enough_divergent_seeds": bool(enough_divergent),
                "observed_noise_verified_lifting_conjunction": bool(noise_verified_lifting),
                "met": bool(noise_lift_fraction_ok),
            },
            {
                "name": "fcg_noise_magnitude_matched",
                "kind": "readiness",
                "description": (
                    "ARM_NOISE's median frozen-random w_chan_finer range is within "
                    "[FCG_NOISE_MAGNITUDE_MATCH_LO, FCG_NOISE_MAGNITUDE_MATCH_HI] x the median "
                    "A2 LEARNED w_chan_finer range -- so the random null is magnitude-matched "
                    "to the structure A2 learns at the SAME layer. measured = the ratio. An "
                    "unmatched magnitude => substrate_not_ready_requeue (the legitimate ONE-"
                    "time re-tune is FCG_NOISE_SCALE)."
                ),
                "control": "median ARM_NOISE fcg_w_chan_finer_range_max vs median A2 learned range",
                # TWO-SIDED (interval). A bare list `threshold` is not a numeric bound
                # spec, so the recompute returned None and fell through to the legacy
                # author-trusted path -- legible now as the indexer's INTERVAL shape.
                # Both legs are INCLUSIVE, mirroring the source predicate
                # `LO <= ratio <= HI`. The guard leg (`median_a2_fcg_range > 0.0`) needs
                # no separate declaration: when it fails the ratio is set to 0.0, which
                # is below LO=0.25, so the interval already reports unmet.
                "measured": float(round(fcg_noise_magnitude_ratio, 6)),
                "threshold": [float(FCG_NOISE_MAGNITUDE_MATCH_LO), float(FCG_NOISE_MAGNITUDE_MATCH_HI)],
                "threshold_low": float(FCG_NOISE_MAGNITUDE_MATCH_LO),
                "threshold_high": float(FCG_NOISE_MAGNITUDE_MATCH_HI),
                "comparator_low": ">=",
                "comparator_high": "<=",
                "direction": "interval",
                "met": bool(fcg_noise_magnitude_matched),
            },
            {
                "name": "candidate_pool_divergent_focus_arms",
                "kind": "readiness",
                "description": (
                    "consumed cand_world_summaries (e2.world_forward) per-candidate SPREAD "
                    "clears the floor on a majority of seeds in the focus arms (GAP-A "
                    "non-vacuity). This is the per-seed divergence the gating reads."
                ),
                "control": "SD-056 e2 trained online in P0; candidate_summary_source=e2_world_forward",
                # COUNT-shaped. `met` is `enough_divergent`, i.e. a k-of-n COUNT of
                # seeds divergent on ALL C1 comparison arms -- NOT a bound on the
                # pool-spread statistic. Two reasons the old min-spread declaration
                # could not reproduce it: (1) min(spread) over all rows is strictly
                # HARSHER than "a majority of seeds", so a sound run with 3 of 6 seeds
                # divergent was flagged precondition_unmet; (2) per-seed
                # `gapa_divergence` is a CONJUNCTION (`consumed_spread_mean >
                # CONSUMED_SPREAD_FLOOR and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL`),
                # and a count over a conjunction does not distribute into per-leg
                # counts, so no single spread statistic exists that could carry it.
                # Reported as the same divergent-seed count `enough_divergent` is
                # defined on -- exact by construction. (It therefore duplicates
                # `enough_divergent_seeds` above; that is the shipped predicate, which
                # is unchanged.)
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                "comparator": ">=",
                "direction": "lower",
                # Non-bound diagnostics: the statistics the divergence count ranges
                # over, preserved verbatim from the pre-fix declaration. Ignored by the
                # indexer's recompute.
                "observed_min_consumed_spread": float(
                    min([r["consumed_summary_pairwise_dist_mean"] for r in all_rows] or [0.0])
                ),
                "observed_consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
                "observed_max_consumed_dist": float(
                    max([r["consumed_summary_pairwise_dist_max"] for r in all_rows] or [0.0])
                ),
                "observed_consumed_magnitude_ceil": float(CONSUMED_MAGNITUDE_CEIL),
                "met": bool(enough_divergent),
            },
        ],
        "criteria": [
            {
                "name": "C1_A2_finer_strict_above_A0_and_A1_and_noise",
                "load_bearing": True,
                "passed": bool(c1_holds),
            },
            {
                "name": "C2_A2_lift_grows_over_training",
                "load_bearing": False,
                "passed": bool(c2_holds),
            },
        ],
        "criteria_non_degenerate": {
            "preconditions_met": bool(preconditions_met),
            "enough_divergent_seeds": bool(enough_divergent),
            "finer_channels_dissociable": bool(fcg_dissociable_ok),
            "noise_verified_lifting": bool(noise_verified_lifting),
            "fcg_noise_magnitude_matched": bool(fcg_noise_magnitude_matched),
            "fcg_moved": bool(fcg_dissociable_ok),
            "lcg_moved": bool(lcg_moved_ok),
            "delta_t_nonflat": bool(fcg_delta_nonflat_ok and lcg_delta_nonflat_ok),
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
            "frac_pre_ge2_floor": float(FRAC_PRE_GE2_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "consumed_magnitude_ceil": float(CONSUMED_MAGNITUDE_CEIL),
            "delta_t_std_floor": float(DELTA_T_STD_FLOOR),
            "w_chan_finer_range_floor": float(W_CHAN_FINER_RANGE_FLOOR),
            "w_chan_range_floor": float(W_CHAN_RANGE_FLOOR),
            "crf_min_minted": int(CRF_MIN_MINTED),
            "crf_frac_active_floor": float(CRF_FRAC_ACTIVE_FLOOR),
            "fcg_noise_scale": float(FCG_NOISE_SCALE),
            "fcg_noise_magnitude_match_lo": float(FCG_NOISE_MAGNITUDE_MATCH_LO),
            "fcg_noise_magnitude_match_hi": float(FCG_NOISE_MAGNITUDE_MATCH_HI),
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
            "n_divergent_seeds": int(n_primary_div),
            "enough_divergent_seeds": enough_divergent,
            "crf_matured": crf_matured,
            "finer_channels_dissociable": fcg_dissociable_ok,
            "fcg_delta_nonflat": fcg_delta_nonflat_ok,
            "fcg_moved": fcg_dissociable_ok,
            "lcg_moved": lcg_moved_ok,
            "lcg_delta_nonflat": lcg_delta_nonflat_ok,
            "noise_verified_lifting": noise_verified_lifting,
            "n_noise_lifts_over_a0": int(n_noise_lifts),
            "fcg_noise_magnitude_matched": fcg_noise_magnitude_matched,
            "fcg_noise_magnitude_ratio": round(float(fcg_noise_magnitude_ratio), 6),
            "median_noise_fcg_range": round(float(median_noise_fcg_range), 8),
            "median_a2_fcg_range": round(float(median_a2_fcg_range), 8),
            "C1_conversion_a2_above_a0_a1_noise": c1_holds,
            "C1_a2_n_seeds": int(n_c1_a2),
            "C1_a2_n_divergent": int(n_div_a2),
            "C2_learning_load_bearing_grows": c2_holds,
            "C2_a2_n_grow_seeds": int(n_grow_a2),
            "A1_reproduces_lift_arc106_ablation": a1_reproduces,
            "A1_n_converts_seeds": int(len(a1_converts_seeds)),
            "mean_committed_class_entropy_a0": round(a0_mean_dv, 6),
            "mean_committed_class_entropy_a1_global_wchan": round(a1_mean_dv, 6),
            "mean_committed_class_entropy_a2_finer_channels": round(a2_mean_dv, 6),
            "mean_committed_class_entropy_noise_same_layer": round(noise_mean_dv, 6),
        },
        "interpretation_grid": {
            "PASS_finer_channels_convert_representational_compression_confirmed_preempt_arc110": (
                "preconditions met (ENOUGH divergent seeds + finer channels DISSOCIABLE + "
                "VERIFIED-LIFTING magnitude-matched SAME-LAYER finer-gating noise bar + "
                "delta_t non-flat + A1 learned) AND C1 (A2 committed-class entropy strict-"
                "above BOTH A0 AND A1_GLOBAL_WCHAN AND the verified noise control on a "
                "strict-majority of DIVERGENT seeds). The finer per-head decomposition "
                "CONVERTS committed-action diversity where the single global w_chan (and a "
                "matched RANDOM finer re-weighting) plateaus -> the conversion ceiling IS "
                "REPRESENTATIONAL COMPRESSION -> supports MECH-451 + PRE-EMPTS the V4 ARC-110 "
                "segregated-loop build. C2 (grows over training) distinguishes learned "
                "conversion from a static finer reweighting (fold-in if static)."
            ),
            "PASS_finer_static_reweighting_converts_compression_confirmed_fold_in": (
                "preconditions met AND C1 (A2 converts strict-above A0/A1/noise) BUT NOT C2 "
                "(the lift is present from tick 0 and does NOT grow with training). The "
                "conversion is a STATIC finer re-weighting, not learning -> MECH-451 supports "
                "(compression IS the constraint, the decomposition matters) but fold the "
                "winning static finer weights into the arithmetic lever."
            ),
            "FAIL_finer_weights_move_no_lift_route_to_arc110_v4_loop_segregation": (
                "DECISIVE. preconditions met (finer channels DISSOCIABLE, verified-lifting "
                "magnitude-matched noise bar, A1 learned) BUT A2 does NOT lift committed-class "
                "entropy strict-above A0/A1/noise. The finer weights MOVE but produce NO "
                "committed-conversion lift beyond the single global w_chan -> representational "
                "compression is NOT the binding constraint -> POSITIVE evidence FOR ARC-110 "
                "(full per-loop competition implicated; route to the v4_loop_segregation "
                "build), NOT evidence against ARC-108 (the learning works, the single-arena "
                "channel exposure is insufficient). weakens MECH-451 (the compression "
                "hypothesis)."
            ),
            "FAIL_substrate_not_ready_requeue": (
                "A precondition is unmet: too FEW divergent seeds (pool collapsed), OR the A2 "
                "finer channels are NOT dissociable (move identically = the compressed blend "
                "re-labelled), OR delta_t flat (no outcome variance), OR the learned weights "
                "never moved (eligibility never credited), OR the SAME-LAYER matched-noise "
                "control did not VERIFY-LIFT above A0 on a strict-majority of divergent seeds, "
                "OR the frozen random w_chan_finer magnitude did not MATCH the A2 learned "
                "range. The conversion question could NOT be measured -- NOT a falsification. "
                "Re-tune FCG_NOISE_SCALE (the correct layer) ONCE; persistent failure to "
                "verify-lift/match-magnitude routes to the V4 ARC-110 build."
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
        "backlog_id": BACKLOG_ID,
        "proposal_id": "EXP-0391",
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
                "finer-channel-gating layer (w_chan_finer over the finer registry, the "
                "exact layer the MECH-451 finer decomposition acts on)"
            ),
            "mechanism": (
                "frozen magnitude-matched random w_chan_finer with "
                "learned_channel_gating_eta=0.0 (the finer three-factor update a no-op), "
                "use_finer_channel_gating=True so the recompose uses the random vector"
            ),
            "scale": FCG_NOISE_SCALE,
        },
        "reuse_baseline_from": REUSE_BASELINE_FROM,
        "evidence_direction_note": (
            f"V3-EXQ-704 (EXP-0391) MECH-451 FINER-CHANNEL-GRANULARITY VALIDATION FALSIFIER "
            f"(experiment_purpose=evidence; claim_ids=[MECH-451]). SIBLING of the V3-EXQ-700 "
            f"lineage on the SAME GAP-A-ready reef-bipartite foraging substrate + the SAME "
            f"landed arithmetic envelope (use_f_eligibility_demotion + adaptive_floor + "
            f"go_nogo + modulatory-authority/top_k k=3) as a MATCHED CONSTANT on all arms; "
            f"settling W_lat OFF on ALL arms. The ONLY change across arms is the score_bias "
            f"channel GRANULARITY feeding the ARC-108 learned-gating layer: A0_ENVELOPE_ONLY "
            f"(no learned gating) / A1_GLOBAL_WCHAN (single global w_chan over the compressed "
            f"blend = the V3-EXQ-700 A1 reference AND the ARC-106 collapse-to-blend ablation) "
            f"/ A2_FINER_CHANNELS (use_finer_channel_gating, MECH-451) / ARM_NOISE (a SAME-"
            f"LAYER finer-gating null = frozen magnitude-matched random w_chan_finer, eta=0). "
            f"PRE-REGISTERED decisive either way: A2 strict-above A0 AND A1 AND noise => "
            f"REPRESENTATIONAL COMPRESSION confirmed -> supports MECH-451 + PRE-EMPTS the V4 "
            f"ARC-110 loop-segregation build; A2 finer-weights-MOVE but no lift beyond A1 => "
            f"compression NOT the binding constraint -> POSITIVE evidence FOR ARC-110 (route "
            f"to v4_loop_segregation), NOT against ARC-108, weakens the MECH-451 compression "
            f"hypothesis. ARC-106 cargo-cult guard: A1 sits IN the C1 strict-above bar (if "
            f"the single global w_chan reproduces the lift, the finer decomposition is NOT "
            f"load-bearing). Non-vacuity self-route substrate_not_ready_requeue (NEVER a false "
            f"weakens): finer channels must be DISSOCIABLE (fcg_w_chan_finer_range > floor), "
            f"the pool divergent (GAP-A), delta_t non-flat + finer weights moved, AND the "
            f"verified-lifting magnitude-matched same-layer noise bar. PROMOTES NOTHING until "
            f"it scores (MECH-451 candidate / substrate_conditional / v3). "
            f"outcome={result['outcome']}; label={result['interpretation_label']}; "
            f"per_claim={result['evidence_direction_per_claim']}."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "4-arm channel-granularity falsifier (A0_ENVELOPE_ONLY / A1_GLOBAL_WCHAN / A2_FINER_CHANNELS / ARM_NOISE same-layer finer-gating null) + per-seed-divergent gating + dissociable-finer-channels gate + verified-lifting magnitude-matched same-layer noise bar",
            "arms": "A0_ENVELOPE_ONLY (no learned gating) / A1_GLOBAL_WCHAN (use_learned_channel_gating; ARC-106 collapse-to-blend ablation) / A2_FINER_CHANNELS (use_finer_channel_gating, MECH-451) / ARM_NOISE (frozen magnitude-matched random w_chan_finer, eta=0 -- same-layer finer-gating null)",
            "swept_variables": "use_learned_channel_gating (A1) vs use_finer_channel_gating (A2/ARM_NOISE) vs neither (A0); ARM_NOISE: learned_channel_gating_eta=0.0 + injected random w_chan_finer. use_learned_settling_step OFF on every arm.",
            "the_isolated_factor": (
                "channel GRANULARITY only: A1 = ONE global w_chan over the compressed "
                "dACC+lPFC+OFC+gated_policy+residual blend; A2 = SEPARATELY-learnable finer "
                "w_chan_finer per named channel; trained by the SAME ARC-108 signed-RPE "
                "three-factor rule (same eta/elig/baseline/asym), so A1-vs-A2 is single-variable."
            ),
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
            "settling_W_lat": "OFF on ALL arms (use_learned_settling_step=False) -- isolate channel granularity",
            "primary_dv": "committed-action-class entropy (nats), interpreted on divergent seeds only",
            "phases": "P0 e2-train (CRF matures, channel gating ON) -> P1 frozen-encoder bias-head REINFORCE -> P2 e2+bias frozen, channel gating KEEPS adapting (binned first/second half for C2)",
            "learning_wiring": "w_chan (A1) / w_chan_finer (A2) learn via e3.post_action_update driven by agent.update_residue every waking tick (all phases); ARM_NOISE eta=0 -> w_chan_finer frozen at the injected random vector",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "fcg_noise_scale": FCG_NOISE_SCALE,
            "fcg_noise_magnitude_match_band": [FCG_NOISE_MAGNITUDE_MATCH_LO, FCG_NOISE_MAGNITUDE_MATCH_HI],
            "reuse_baseline_from": REUSE_BASELINE_FROM,
            "reusable_arm_ids": list(REUSABLE_ARM_IDS_LOCAL),
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
            "arc110_relationship": "MECH-451 is the cheap V3 rung BEFORE ARC-110/v4_loop_segregation; a PASS pre-empts the V4 build, a route-to-arc110 FAIL is positive evidence FOR it",
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-704 (EXP-0391) MECH-451 finer-channel-granularity falsifier"
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
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"reuse_hits={result['n_reuse_hits']} "
        f"preconditions_met={result['acceptance_criteria']['preconditions_met']} "
        f"n_divergent={result['acceptance_criteria']['n_divergent_seeds']} "
        f"finer_dissociable={result['acceptance_criteria']['finer_channels_dissociable']} "
        f"noise_verified_lifting={result['acceptance_criteria']['noise_verified_lifting']} "
        f"fcg_noise_magnitude_matched={result['acceptance_criteria']['fcg_noise_magnitude_matched']} "
        f"C1={result['acceptance_criteria']['C1_conversion_a2_above_a0_a1_noise']} "
        f"C2={result['acceptance_criteria']['C2_learning_load_bearing_grows']} "
        f"A1_reproduces={result['acceptance_criteria']['A1_reproduces_lift_arc106_ablation']} "
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
