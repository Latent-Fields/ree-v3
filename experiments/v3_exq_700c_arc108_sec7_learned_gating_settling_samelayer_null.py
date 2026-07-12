#!/opt/local/bin/python3
"""
V3-EXQ-700c -- ARC-108 sec-7 LEARNED-GATING CONVERSION FALSIFIER, brake-EXEMPT
TERMINAL RE-RUN with a SAME-LAYER null. SUPERSEDES V3-EXQ-700b.

THE ONE SCIENTIFIC CHANGE FROM 700b
-----------------------------------
700b's matched-noise control (ARM_NOISE) was a POLICY-SOFTMAX-TEMPERATURE lift
(MECH-313 use_noise_floor, NOISE_FLOOR_ALPHA=2.0). The 700b autopsy
(failure_autopsy_V3-EXQ-700b_2026-06-24) found that null is at the WRONG LAYER:
under the F-bounded eligibility constitution (MECH-448 demotion + MECH-449 go/no-go +
the top_k shortlist), a tonic softmax-temperature lift is DECOUPLED from the
committed-class DV -- it perturbs the sampling temperature, not the within-eligible
accumulator the committed argmin reads. So the temperature null could not VERIFY-LIFT
committed-class entropy (the recurring 569g/684 control-failure mode), and the C1
"strict-above-noise" bar stayed UNVERIFIED.

700c REPLACES that decoupled null with a SAME-LAYER null: a FROZEN, magnitude-matched,
random-STRUCTURE perturbation injected at the EXACT layer MECH-450 settling acts on --
the eligibility/settling field. Concretely, ARM_NOISE runs the learned-settling code
path (use_learned_settling_step=True) but with learned_settling_eta=0.0 (W_lat FROZEN,
Delta=0) and W_lat seeded once to a magnitude-matched RANDOM [C,C] matrix. The settling
step then injects inhib = onehot @ (W_lat @ a_class) into accum = mod_eligible.clone()
-- the within-eligible field the committed argmin reads. This is a random-structure
lateral-inhibition operator in the SAME computation the learned arms use; it is
structurally COUPLED to the committed-class DV (the temperature null was not).

WHY THIS VERIFY-LIFTS WHERE 700b's TEMPERATURE NULL COULD NOT: a settling-field
perturbation acts INSIDE the eligible set on the same accumulator the committed argmin
reads, so a random structure moves the committed class. The learning arms must then
exceed a magnitude-matched RANDOM settling-field perturbation -- a far better-layered
bar: a lift is attributable to learned STRUCTURE only if it beats unstructured
random structure at the same layer.

WHY THIS IS BRAKE-EXEMPT (user-adjudicated 2026-06-24, re_derive_brake.exempt=true):
MECH-439 has multiple prior non_contributory/substrate_ceiling autopsies (the
mechanical re-derive brake), but this run is EXEMPT: it runs CONCURRENTLY with the
V4 loop-segregation escalation (the autopsy's other adjudicated route -- it cannot be
delayed), it is a genuine null REDESIGN (a same-layer null, NOT another alpha-bump
iteration of the same lever), and it is PRE-REGISTERED TERMINAL. The brake REFUSES any
alpha-bump same-lever requeue; this is not that.

TERMINALITY
-----------
PRE-REGISTERED TERMINAL. The field-noise scale may be re-tuned AT MOST ONCE
(FIELD_NOISE_WLAT_SCALE, the CORRECT-layer knob -- NOT an alpha bump). A persistent
failure to verify-lift / match magnitude at the correct layer ESCALATES to the V4
loop-segregation substrate (implement-substrate), NEVER another V3 letter.

THE 5 ARMS (all carry the landed arithmetic envelope as a MATCHED CONSTANT, exactly as
700b: use_f_eligibility_demotion + adaptive_floor + go_nogo + modulatory-authority/top_k
shortlist; the swept variables are ONLY the learned levers + rpe_mode + the same-layer
field-noise on ARM_NOISE):
  A0_ENVELOPE_ONLY       : envelope-only control (no learning, no noise).
  A2_SETTLING_SIGNED     : learned W_lat settling, signed RPE (the FOCUS / converting lever).
  A3_BOTH_SIGNED         : learned w_chan + learned W_lat settling, signed RPE.
  C3_SETTLING_UNSIGNED   : learned W_lat settling, UNSIGNED RPE (B5 ablation; must FAIL to convert).
  ARM_NOISE              : matched-magnitude SAME-LAYER settling-field null (FROZEN random W_lat,
                           learned_settling_eta=0.0) -- the verify-lifting bar the learning arms exceed.
6 seeds. claim_ids = [MECH-439, ARC-108, MECH-450]. experiment_purpose = evidence.
PROMOTES NOTHING until it scores -- MECH-439 is substrate_ceiling and ARC-108 / MECH-450
are substrate_conditional, so promote/demote is SUPPRESSED on all three.

ARM-REUSE: the 4 unchanged arms (A0/A2/A3/C3) are byte-identical to 700b (same env +
matched envelope + schedule); they emit a REUSE-ELIGIBLE per-cell fingerprint (config
slice declared from experiments/_lib/baselines/exq700_arc108_settling_baseline.py,
include_driver_script_in_hash=False) so a future mint could match. With REUSE_BASELINE_FROM
=None (the safe default for a terminal run) every arm runs fresh. ARM_NOISE is the changed
arm and is never reusable (extra_ineligible_reasons).

PRIMARY ACCEPTANCE (unchanged from 700b structure; only the noise control's LAYER changed)
------------------------------------------------------------------------------------------
  PRIMARY DV = committed-action-class entropy (nats), measured over P2.
  Interpreted ONLY on DIVERGENT seeds (per-seed-divergent gating).
  C1 (conversion): a learning arm's committed-class entropy strict-above BOTH the A0
      envelope-only arm AND the VERIFIED-LIFTING SAME-LAYER matched-noise control (frozen
      magnitude-matched random W_lat at the eligibility/settling field), by CONVERSION_MARGIN,
      on a strict-majority of DIVERGENT seeds.
  C2 (learning is load-bearing): the lift GROWS over training (second-half P2 entropy
      strict-above first-half on the converting arm) AND the learned weights MOVED.
  C3 (signed-RPE load-bearing -- divergence B5): A2_SETTLING_SIGNED CONVERTS (C3a) AND
      C3_SETTLING_UNSIGNED does NOT converge (C3b).

NON-VACUITY / SELF-ROUTE (never a false weakening)
--------------------------------------------------
  self-route substrate_not_ready_requeue if ANY precondition is unmet:
   (a) too few DIVERGENT seeds; (b) delta_t FLAT; (c) W_lat / w_chan never MOVE / the
   settling never moves the field; (d) the SAME-LAYER matched-noise control does NOT
   verify-lift above A0 on a strict-majority of DIVERGENT seeds (the better-layered bar);
   (e) field_noise_magnitude_matched -- ARM_NOISE's frozen random W_lat range is within
   [LO,HI] x the median settling-arm learned W_lat range (the legitimate one-time re-tune
   knob is the FIELD-NOISE scale, the CORRECT layer -- NOT an alpha bump).

DECISIVE-OR-ESCALATE (the HARD STOP)
------------------------------------
  A PRECONDITIONS-MET FAIL (learning engaged, weights moved, ENOUGH divergent seeds,
  the SAME-LAYER null VERIFIED-LIFTING + magnitude-matched, but NO entropy lift over
  A0/noise) is the GENUINE, DECISIVE "no lift" outcome -> route to the V4 full
  BG-thalamo-cortical loop / loop-segregation scope bet
  (no_lift_preconditions_met_escalate_v4_full_loop), NOT another V3 letter. The same-layer
  null verify-lifting AND the learned settling arms NOT beating it => learned STRUCTURE adds
  nothing over a matched random settling-field perturbation -> escalate V4 (no more V3 letters).
  It does NOT falsify ARC-107 (the arithmetic envelope still holds) and does NOT promote
  learned gating.

See REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-700b_2026-06-24.{md,json},
    REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md,
    experiments/v3_exq_700b_arc108_sec7_learned_gating_settling_c3.py (the superseded harness),
    experiments/_lib/baselines/exq700_arc108_settling_baseline.py (the reusable-arm config slice),
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
from experiments._lib.arm_reuse import try_reuse_cell
from experiments._lib.baselines.exq700_arc108_settling_baseline import (
    REUSABLE_ARM_IDS,
    arm_config_slice,
)
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_700c_arc108_sec7_learned_gating_settling_samelayer_null"
QUEUE_ID = "V3-EXQ-700c"
SUPERSEDES = "V3-EXQ-700b"
CLAIM_IDS: List[str] = ["MECH-439", "ARC-108", "MECH-450"]
EXPERIMENT_PURPOSE = "evidence"

# CRF-gate calibration levers (matured CRF stack; ported verbatim from 700b,
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

# ----- SAME-LAYER null (700c redesign) -----
# ARM_NOISE injects a FROZEN, magnitude-matched, random-STRUCTURE perturbation at the
# eligibility/settling field (the exact layer MECH-450 W_lat settling acts on): the
# learned-settling code path runs (use_learned_settling_step=True) but W_lat is FROZEN
# (learned_settling_eta=0.0) and seeded once to a random [C,C] matrix. NOT a policy
# temperature lift (that was 700b's decoupled null). Only meaningful when eta=0 (frozen).
FIELD_NOISE_WLAT_SCALE = 0.5          # pre-registered scale of the frozen random W_lat (magnitude-matched bar)
FIELD_NOISE_SEED_OFFSET = 100000      # reproducible-per-cell random W_lat distinct from the cell seed's other RNG
FIELD_NOISE_MAGNITUDE_MATCH_LO = 0.25  # ARM_NOISE wlat_range_max must be within [LO,HI] x median settling-arm range
FIELD_NOISE_MAGNITUDE_MATCH_HI = 4.0

# Mint run_id to cite for arm-reuse; None => run all arms fresh (the safe default for a
# terminal run). The parent session sets this only AFTER the V3-EXQ-700c-mint completes.
REUSE_BASELINE_FROM = None

# OFF/settling-arm metric keys the acceptance logic reads from the reusable rows (so a
# reused cell must have recorded all of them -- the section-9.2 correctness trap).
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
    "wlat_moved",
    "wlat_delta_nonflat",
    "settling_moved_field",
    "wlat_range_max",
    "wlat_delta_t_std",
    "lcg_moved",
    "lcg_delta_nonflat",
    "lcg_w_chan_range_max",
    "lcg_delta_t_std",
    "mean_settling_round_delta",
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

# MECH-313 noise floor -- removed as the null on ARM_NOISE (700c uses the same-layer null
# instead). The floor stays inert on every arm now; defaults retained for the off path.
NOISE_FLOOR_ALPHA = 2.0
NOISE_FLOOR_MIN_TEMPERATURE = 1.0

# SD-056 online e2 training (mirror 700b).
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

# P1 bias-head REINFORCE training (mirror 700b).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9


# IDENTICAL env to 700b (the GAP-A reef-bipartite foraging bank).
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


# The 5 arms. The ONLY swept config is (lcg_on, settle_on, noise_on, field_noise, rpe_mode).
# A0/A2/A3/C3 are BYTE-IDENTICAL to 700b (no field_noise key; defaults False). The 700c
# redesign is ARM_NOISE: a SAME-LAYER settling-field null (settle_on=True, noise_on=False,
# field_noise=True) replacing 700b's policy-temperature null (noise_on=True).
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "A0_ENVELOPE_ONLY",
        "label": "envelope_only_control_no_learned_gating",
        "lcg_on": False,
        "settle_on": False,
        "noise_on": False,
        "rpe_mode": "signed",   # irrelevant (no learning); default
    },
    {
        "arm_id": "A2_SETTLING_SIGNED",
        "label": "learned_W_lat_recurrent_settling_signed_rpe_focus_lever",
        "lcg_on": False,
        "settle_on": True,
        "noise_on": False,
        "rpe_mode": "signed",   # signed delta_t = R_t - V-hat_t (the converting lever)
    },
    {
        "arm_id": "A3_BOTH_SIGNED",
        "label": "learned_w_chan_plus_learned_W_lat_settling_signed_rpe",
        "lcg_on": True,
        "settle_on": True,
        "noise_on": False,
        "rpe_mode": "signed",   # does w_chan ADD over the settling lever?
    },
    {
        "arm_id": "C3_SETTLING_UNSIGNED",
        "label": "learned_W_lat_settling_unsigned_rpe_B5_ablation",
        "lcg_on": False,
        "settle_on": True,
        "noise_on": False,
        "rpe_mode": "unsigned",  # abs ARC-016 running_variance substituted for delta_t (B5)
    },
    {
        "arm_id": "ARM_NOISE",
        "label": "matched_magnitude_same_layer_settling_field_null_frozen_random_wlat",
        "lcg_on": False,
        "settle_on": True,       # run the settling code path...
        "noise_on": False,       # ...with the MECH-313 temperature null REMOVED (the 700b layer)
        "field_noise": True,     # ...and a FROZEN magnitude-matched random W_lat (eta=0) at the settling field
        "rpe_mode": "signed",    # irrelevant (eta=0, no learning); default
    },
]

LEARNING_ARM_IDS = ("A2_SETTLING_SIGNED", "A3_BOTH_SIGNED", "C3_SETTLING_UNSIGNED")
LCG_ARM_IDS = ("A3_BOTH_SIGNED",)   # only A3 arms the w_chan lever
SETTLE_ARM_IDS = ("A2_SETTLING_SIGNED", "A3_BOTH_SIGNED", "C3_SETTLING_UNSIGNED")
# The four reusable arms (A0/A2/A3/C3); ARM_NOISE is the changed arm (never reused).
REUSABLE_ARM_IDS_LOCAL = tuple(REUSABLE_ARM_IDS)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Matched-stack agent. The landed arithmetic envelope (demotion + adaptive floor +
    Go/No-Go + authority + routing + top_k shortlist) + the diversity stack (MECH-341,
    SD-056, CRF, trained lateral_pfc bias head, use_dacc) are MATCHED CONSTANTS on all
    arms. The ONLY toggled flags are the two learned levers
    (use_learned_channel_gating, use_learned_settling_step), the C3 ablation knob
    (learned_channel_rpe_mode), and -- on ARM_NOISE only -- the SAME-LAYER field-noise:
    the settling step runs (settle_on True) with learned_settling_eta=0.0 (W_lat FROZEN)
    so the once-seeded random W_lat is a magnitude-matched random-structure perturbation
    in the SAME within-eligible field the learned arms settle. use_noise_floor is OFF
    on every arm now (the 700b policy-temperature null is removed)."""
    lcg_on = bool(arm["lcg_on"])
    settle_on = bool(arm["settle_on"])
    noise_on = bool(arm["noise_on"])
    field_noise = bool(arm.get("field_noise", False))
    rpe_mode = str(arm.get("rpe_mode", "signed"))
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
        # MECH-313 noise floor -- OFF on EVERY arm (the 700b policy-temperature null is
        # removed; 700c uses the same-layer field-noise on ARM_NOISE instead).
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
        # --- ARC-108 sec-7 C3 (divergence B5): signed-vs-unsigned-RPE ablation knob. ---
        learned_channel_rpe_mode=rpe_mode,
        # --- MECH-450 JOB-1 step-2: learned recurrent-settling step W_lat (TOGGLED) ---
        # ARM_NOISE runs the settling code path (settle_on True) but with eta=0.0 so its
        # once-seeded random W_lat stays FROZEN -- the same-layer magnitude-matched null.
        use_learned_settling_step=settle_on,
        learned_settling_rounds=SETTLING_ROUNDS,
        learned_settling_temperature=SETTLING_TEMPERATURE,
        learned_settling_eta=(0.0 if field_noise else SETTLING_ETA),
        learned_settling_elig_decay=SETTLING_ELIG_DECAY,
        learned_settling_n_action_classes=SETTLING_N_ACTION_CLASSES,
    )
    return REEAgent(cfg)


def _inject_field_noise_wlat(agent: REEAgent, seed: int) -> float:
    """SAME-LAYER null injection: seed the FROZEN random W_lat once.

    Writes a magnitude-matched random [C,C] matrix into agent.e3.W_lat (the
    register_buffer the MECH-450 settling step reads). This is ONLY meaningful when
    learned_settling_eta=0.0 (W_lat frozen) -- which _make_agent guarantees on the
    field-noise arm -- so the random matrix is a stable random-structure
    lateral-inhibition operator that the settling step injects into the within-eligible
    field every tick (the layer the committed argmin reads), NOT a learned object.

    Reproducible per cell via FIELD_NOISE_SEED_OFFSET (distinct from the cell seed's
    other RNG). Returns the injected W_lat's range (max - min) for magnitude logging.
    """
    C = int(SETTLING_N_ACTION_CLASSES)
    g = torch.Generator(device="cpu").manual_seed(int(seed) + FIELD_NOISE_SEED_OFFSET)
    W = torch.randn((C, C), generator=g) * FIELD_NOISE_WLAT_SCALE
    agent.e3.W_lat.copy_(W.to(dtype=agent.e3.W_lat.dtype, device=agent.e3.W_lat.device))
    return float((W.max() - W.min()).item())


# ---------------------------------------------------------------------------
# SD-056 online e2 training (verbatim from 700b)
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
# P1 bias-head REINFORCE training (verbatim from 700b)
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

    # SAME-LAYER null: inject the frozen magnitude-matched random W_lat ONCE on the
    # field-noise arm, after the agent is built and before training. eta=0 keeps it
    # frozen across all P0/P1/P2 episodes (W_lat is not reset by agent.reset()).
    field_noise_wlat_range = 0.0
    if arm.get("field_noise"):
        field_noise_wlat_range = _inject_field_noise_wlat(agent, seed)

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

    # ----- ARC-108 / MECH-450 learning diagnostics (accumulated across ALL phases) -----
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
            # three-factor learning fires here on EVERY waking tick (all phases). On
            # the field-noise arm learned_settling_eta=0.0 -> W_lat unchanged (frozen).
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

    # On the field-noise arm the random W_lat is FROZEN, so wlat_range_max reflects the
    # INJECTED random structure (eta=0 => zero updates => range comes from the seeded
    # matrix). Use the injected magnitude as the authoritative range for the magnitude
    # match if no update ever surfaced one.
    if arm.get("field_noise") and field_noise_wlat_range > wlat_range_max:
        wlat_range_max = field_noise_wlat_range

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
        "field_noise": bool(arm.get("field_noise", False)),
        "rpe_mode": str(arm.get("rpe_mode", "signed")),
        "field_noise_wlat_injected_range": round(float(field_noise_wlat_range), 8),
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
            f"settle_on={arm['settle_on']} noise_on={arm['noise_on']} "
            f"field_noise={arm.get('field_noise', False)} "
            f"rpe_mode={arm.get('rpe_mode', 'signed')} "
            f"(P0={p0_episodes} ep e2-train, P1={p1_episodes} ep bias-train, "
            f"P2={p2_episodes} ep measure, steps_per_episode={steps_per_episode}, "
            f"dry_run={dry_run})",
            flush=True,
        )
        reusable = arm["arm_id"] in REUSABLE_ARM_IDS_LOCAL
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)

            # ----- ARM-REUSE (consumer side), gated + safe-by-default -----
            # Only the 4 unchanged arms are reuse-eligible, and only IFF a mint is cited.
            # With REUSE_BASELINE_FROM=None (the terminal default) this is skipped and
            # every arm runs fresh.
            row: Optional[Dict[str, Any]] = None
            if REUSE_BASELINE_FROM is not None and reusable:
                cell = try_reuse_cell(
                    config_slice=arm_config_slice(
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

            # Per-cell fingerprint. The 4 reusable arms emit a REUSE-ELIGIBLE fingerprint
            # (config slice declared from the canonical baseline module,
            # include_driver_script_in_hash=False) so a future mint could match. ARM_NOISE
            # is the changed arm -- never reused.
            if reusable:
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=arm_config_slice(
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
                        "arm_id": arm["arm_id"],
                        "lcg_on": bool(arm["lcg_on"]),
                        "settle_on": bool(arm["settle_on"]),
                        "noise_on": bool(arm["noise_on"]),
                        "field_noise": bool(arm.get("field_noise", False)),
                        "field_noise_wlat_scale": float(FIELD_NOISE_WLAT_SCALE),
                        "field_noise_seed_offset": int(FIELD_NOISE_SEED_OFFSET),
                        "rpe_mode": str(arm.get("rpe_mode", "signed")),
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
                        "learned_settling_rounds": int(SETTLING_ROUNDS),
                        "learned_settling_eta": 0.0,
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
                        "field_noise_frozen_random_wlat_not_a_reusable_baseline",
                    ],
                )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    a0_rows = _arm_rows(arm_results, "A0_ENVELOPE_ONLY")
    a2_rows = _arm_rows(arm_results, "A2_SETTLING_SIGNED")
    a3_rows = _arm_rows(arm_results, "A3_BOTH_SIGNED")
    c3u_rows = _arm_rows(arm_results, "C3_SETTLING_UNSIGNED")
    noise_rows = _arm_rows(arm_results, "ARM_NOISE")
    all_rows = a0_rows + a2_rows + a3_rows + c3u_rows + noise_rows
    learning_rows = a2_rows + a3_rows + c3u_rows

    def _maj(rows: List[Dict[str, Any]], pred) -> bool:
        return sum(1 for r in rows if pred(r)) >= MIN_SEEDS_FOR_PASS

    a0_ent = _by_seed(a0_rows, "committed_class_entropy_nats")
    a2_ent = _by_seed(a2_rows, "committed_class_entropy_nats")
    a3_ent = _by_seed(a3_rows, "committed_class_entropy_nats")
    c3u_ent = _by_seed(c3u_rows, "committed_class_entropy_nats")
    noise_ent = _by_seed(noise_rows, "committed_class_entropy_nats")

    a0_gap = _gap_by_seed(a0_rows)
    a2_gap = _gap_by_seed(a2_rows)
    a3_gap = _gap_by_seed(a3_rows)
    c3u_gap = _gap_by_seed(c3u_rows)
    noise_gap = _gap_by_seed(noise_rows)

    # ----- Per-seed-divergent gating -----
    # The PRIMARY divergence set = seeds whose pool is divergent on the focus C1
    # comparison arms (A0 + A2 settling + ARM_NOISE). A degenerate cell self-excludes.
    primary_div = [
        s for s in sorted(set(a0_gap) & set(a2_gap) & set(noise_gap))
        if a0_gap.get(s) and a2_gap.get(s) and noise_gap.get(s)
    ]
    n_primary_div = len(primary_div)
    enough_divergent = n_primary_div >= MIN_DIVERGENT_SEEDS

    # ----- Precondition (d): the SAME-LAYER matched-noise control VERIFIED-LIFTING above A0 -----
    # 700c: ARM_NOISE is now a frozen magnitude-matched random W_lat at the settling field
    # (NOT a policy-temperature lift). The verify-lift logic is unchanged; it now tests the
    # better-layered bar (the same-layer null must lift committed-class entropy strict-above
    # A0 on a strict-majority of divergent seeds; the autopsy root cause was that 700b's
    # temperature null was decoupled from the committed-class DV and could not).
    n_noise_lifts = sum(
        1 for s in primary_div if noise_ent.get(s, 0.0) > a0_ent.get(s, 0.0) + CONVERSION_MARGIN
    )
    noise_verified_lifting = bool(enough_divergent and _div_pass(n_noise_lifts, n_primary_div))

    # ----- Precondition (e): field_noise_magnitude_matched (the legitimate one-time re-tune knob) -----
    # ARM_NOISE's median frozen-random W_lat range must be within [LO,HI] x the median
    # learned settling-arm (A2+A3+C3) W_lat range -- so the random null is magnitude-matched
    # to the structure the learning arms learn. An unmatched magnitude => substrate_not_ready_requeue
    # (re-tune the FIELD-NOISE scale FIELD_NOISE_WLAT_SCALE, the CORRECT layer -- NOT an alpha bump).
    noise_wlat_ranges = [float(r["wlat_range_max"]) for r in noise_rows]
    settling_wlat_ranges = [
        float(r["wlat_range_max"]) for r in (a2_rows + a3_rows + c3u_rows)
    ]
    median_noise_wlat_range = _median(noise_wlat_ranges)
    median_settling_wlat_range = _median(settling_wlat_ranges)
    if median_settling_wlat_range > 0.0:
        field_noise_magnitude_ratio = median_noise_wlat_range / median_settling_wlat_range
    else:
        field_noise_magnitude_ratio = 0.0
    field_noise_magnitude_matched = bool(
        median_settling_wlat_range > 0.0
        and FIELD_NOISE_MAGNITUDE_MATCH_LO <= field_noise_magnitude_ratio <= FIELD_NOISE_MAGNITUDE_MATCH_HI
    )

    # ----- Precondition (b)/(c): learning engaged on the armed arms -----
    # Settling-armed arms (A2, A3, C3): W_lat moved + settling moved field + delta nonflat.
    wlat_moved_ok = all(
        _maj(rows, lambda r: r["wlat_moved"] and r["settling_moved_field"])
        for rows in (a2_rows, a3_rows, c3u_rows)
    )
    wlat_delta_nonflat_ok = all(
        _maj(rows, lambda r: r["wlat_delta_nonflat"]) for rows in (a2_rows, a3_rows, c3u_rows)
    )
    # w_chan-armed arm (A3 only).
    lcg_moved_ok = _maj(a3_rows, lambda r: r["lcg_moved"])
    lcg_delta_nonflat_ok = _maj(a3_rows, lambda r: r["lcg_delta_nonflat"])

    # CRF maturity readiness (matched constant; on a majority of seeds across all arms).
    crf_matured = all(
        _maj(rows, lambda r: r["crf_differentiated"]) for rows in
        (a0_rows, a2_rows, a3_rows, c3u_rows, noise_rows)
    )

    preconditions_met = bool(
        enough_divergent
        and noise_verified_lifting
        and field_noise_magnitude_matched
        and wlat_moved_ok and wlat_delta_nonflat_ok
        and lcg_moved_ok and lcg_delta_nonflat_ok
        and crf_matured
    )

    # ----- C1 (conversion): a learning arm strict-above BOTH A0 AND the noise control,
    # on the per-arm divergent seeds (A0 AND that arm AND noise all divergent) -----
    def _converts(arm_ent: Dict[int, float], arm_gap: Dict[int, bool]) -> Tuple[int, int, List[int]]:
        div = [
            s for s in sorted(set(a0_gap) & set(arm_gap) & set(noise_gap))
            if a0_gap.get(s) and arm_gap.get(s) and noise_gap.get(s)
        ]
        seeds_ok: List[int] = []
        for s in div:
            bar = max(a0_ent.get(s, 0.0), noise_ent.get(s, 0.0)) + CONVERSION_MARGIN
            if arm_ent.get(s, 0.0) > bar:
                seeds_ok.append(s)
        return len(seeds_ok), len(div), seeds_ok

    n_c1_a2, n_div_a2, c1_a2_seeds = _converts(a2_ent, a2_gap)
    n_c1_a3, n_div_a3, c1_a3_seeds = _converts(a3_ent, a3_gap)
    n_c1_c3u, n_div_c3u, c1_c3u_seeds = _converts(c3u_ent, c3u_gap)
    c1_a2 = _div_pass(n_c1_a2, n_div_a2)
    c1_a3 = _div_pass(n_c1_a3, n_div_a3)
    c1_c3u = _div_pass(n_c1_c3u, n_div_c3u)
    # The CONVERTING (signed settling) levers only -- the unsigned arm converting is the
    # C3-REFUTED signal, not the conversion criterion.
    c1_holds = bool(c1_a2 or c1_a3)

    # ----- C2 (learning load-bearing): grows over training, on the per-arm divergent
    # seeds where the arm converted -----
    def _grows(rows: List[Dict[str, Any]], arm_gap: Dict[int, bool]) -> Tuple[int, int, List[int]]:
        div = [
            s for s in sorted(set(a0_gap) & set(arm_gap) & set(noise_gap))
            if a0_gap.get(s) and arm_gap.get(s) and noise_gap.get(s)
        ]
        seeds_ok: List[int] = []
        for r in rows:
            s = int(r["seed"])
            if s not in div:
                continue
            if (
                r["committed_class_entropy_p2_second_half_nats"]
                > r["committed_class_entropy_p2_first_half_nats"] + GROWTH_MARGIN
            ):
                seeds_ok.append(s)
        return len(seeds_ok), len(div), seeds_ok

    n_grow_a2, _, grow_a2_seeds = _grows(a2_rows, a2_gap)
    n_grow_a3, _, grow_a3_seeds = _grows(a3_rows, a3_gap)
    # weights-moved is already a precondition; C2 adds growth on the converting arm.
    c2_a2 = bool(c1_a2 and _div_pass(n_grow_a2, n_div_a2))
    c2_a3 = bool(c1_a3 and _div_pass(n_grow_a3, n_div_a3))
    c2_holds = bool(c2_a2 or c2_a3)

    # ----- C3 (signed-RPE load-bearing -- divergence B5) -----
    c3_unsigned_engaged = bool(
        _maj(c3u_rows, lambda r: r["wlat_moved"] and r["settling_moved_field"])
        and _maj(c3u_rows, lambda r: r["wlat_delta_nonflat"])
    )
    c3a_signed_converts = bool(c1_a2)
    c3b_unsigned_fails = bool(not c1_c3u)
    c3_scoreable = bool(c3_unsigned_engaged and enough_divergent)
    c3_signed_load_bearing = bool(c3_scoreable and c3a_signed_converts and c3b_unsigned_fails)
    c3_refuted = bool(c3_scoreable and c3a_signed_converts and c1_c3u)

    # ----- Outcome map -----
    if not preconditions_met:
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        per_claim = {c: "non_contributory" for c in CLAIM_IDS}
    elif not c1_holds:
        # preconditions met (learning engaged, weights moved, ENOUGH divergent seeds,
        # SAME-LAYER null VERIFIED-LIFTING + magnitude-matched) but NO entropy lift over
        # A0/noise -> the GENUINE, DECISIVE no-lift outcome: the same-layer null verify-
        # lifts AND the learned settling arms do NOT beat it -> learned STRUCTURE adds
        # nothing over a matched random settling-field perturbation -> escalate V4
        # loop-segregation (NO more V3 letters).
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "no_lift_preconditions_met_escalate_v4_full_loop"
        per_claim = {c: "non_contributory" for c in CLAIM_IDS}
    elif not c2_holds:
        outcome = "PASS"
        overall_direction = "mixed"
        label = "static_reweighting_fold_into_arithmetic"
        per_claim = {
            "MECH-439": "supports",        # the conversion ceiling IS liftable
            "ARC-108": "mixed",            # converts but not via learning (static)
            "MECH-450": "mixed",
        }
    else:
        outcome = "PASS"
        overall_direction = "supports"
        label = "learned_gating_converts_where_arithmetic_plateaus"
        settling_supported = bool(c2_a2 or c2_a3)
        if c3_signed_load_bearing:
            arc108_dir = "supports"
        elif c3_refuted:
            arc108_dir = "mixed"
        else:
            arc108_dir = "mixed"   # C3 unscoreable (unsigned arm starved); w_chan additivity unclear
        per_claim = {
            "MECH-439": "supports",
            "ARC-108": arc108_dir,
            "MECH-450": "supports" if settling_supported else "mixed",
        }

    a0_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a0_rows])
    a2_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a2_rows])
    a3_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a3_rows])
    c3u_mean_dv = _mean([r["committed_class_entropy_nats"] for r in c3u_rows])
    noise_mean_dv = _mean([r["committed_class_entropy_nats"] for r in noise_rows])

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "enough_divergent_seeds",
                "kind": "readiness",
                "description": (
                    "the number of seeds whose candidate pool is DIVERGENT on the focus "
                    "C1 comparison arms (A0 + A2 settling + ARM_NOISE) is >= "
                    "MIN_DIVERGENT_SEEDS. Per-seed-divergent gating (701a-style): a "
                    "degenerate cell self-excludes. Too few divergent seeds => "
                    "substrate_not_ready_requeue (pool too collapsed to test conversion)."
                ),
                "control": "consumed cand_world_summary pairwise spread > floor (GAP-A); per-seed",
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                "met": bool(enough_divergent),
            },
            {
                "name": "matched_noise_control_verified_lifting",
                "kind": "readiness",
                "description": (
                    "700c SAME-LAYER null: the ARM_NOISE matched-magnitude FROZEN random "
                    "W_lat at the eligibility/settling field (the layer MECH-450 settling "
                    "acts on) lifts committed-class entropy strict-above A0 by margin on a "
                    "strict-majority of DIVERGENT seeds. This is the BETTER-LAYERED bar: a "
                    "settling-field perturbation is COUPLED to the committed-class DV (the "
                    "700b policy-temperature null was DECOUPLED by the F-bounded eligibility "
                    "constitution and could not verify-lift). Control not verify-lifting => "
                    "substrate_not_ready_requeue (re-tune the FIELD-NOISE scale, the correct "
                    "layer -- NOT an alpha bump)."
                ),
                "control": "ARM_NOISE committed-class entropy vs A0, divergent seeds, paired",
                "measured": float(n_noise_lifts),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "met": bool(noise_verified_lifting),
            },
            {
                "name": "field_noise_magnitude_matched",
                "kind": "readiness",
                "description": (
                    "700c SAME-LAYER null magnitude match: ARM_NOISE's median frozen-random "
                    "W_lat range is within [FIELD_NOISE_MAGNITUDE_MATCH_LO, "
                    "FIELD_NOISE_MAGNITUDE_MATCH_HI] x the median LEARNED settling-arm "
                    "(A2+A3+C3) W_lat range -- so the random null is magnitude-matched to the "
                    "structure the learning arms learn. measured = the ratio. An unmatched "
                    "magnitude => substrate_not_ready_requeue (the legitimate ONE-TIME re-tune "
                    "is the FIELD-NOISE scale FIELD_NOISE_WLAT_SCALE, the CORRECT layer -- NOT "
                    "an alpha bump)."
                ),
                "control": "median ARM_NOISE wlat_range_max vs median A2/A3/C3 learned wlat_range_max",
                "measured": float(round(field_noise_magnitude_ratio, 6)),
                "threshold": [float(FIELD_NOISE_MAGNITUDE_MATCH_LO), float(FIELD_NOISE_MAGNITUDE_MATCH_HI)],
                "met": bool(field_noise_magnitude_matched),
            },
            {
                "name": "delta_t_carries_variance_on_learning_arms",
                "kind": "readiness",
                "description": (
                    "the signed-RPE delta_t (= benefit_eval - harm_eval - V-hat_t; OR the "
                    "unsigned abs ARC-016 variance on C3_SETTLING_UNSIGNED) carries cross-tick "
                    "STD above floor on a majority of seeds on the settling-armed (A2, A3, C3) "
                    "arms AND the w_chan-armed (A3) arm -- there is outcome variance to learn "
                    "from. Flat => substrate_not_ready_requeue (the C3 unsigned arm flat is "
                    "STARVED, not 'unsigned fails')."
                ),
                "control": "wlat_delta_t_std (A2/A3/C3) + lcg_delta_t_std (A3) on the armed arms",
                "measured": float(min(
                    [r["wlat_delta_t_std"] for r in (a2_rows + a3_rows + c3u_rows)]
                    + [r["lcg_delta_t_std"] for r in a3_rows] or [0.0]
                )),
                "threshold": float(DELTA_T_STD_FLOOR),
                "met": bool(wlat_delta_nonflat_ok and lcg_delta_nonflat_ok),
            },
            {
                "name": "learned_weights_moved_from_init_on_armed_arms",
                "kind": "readiness",
                "description": (
                    "W_lat range > floor AND the settling MOVED the field "
                    "(learned_settling_round_delta > floor) on the settling-armed arms "
                    "(A2, A3, C3; zero init => range 0) AND w_chan range > floor on the "
                    "w_chan-armed arm (A3; softplus-unity init => range 0). Weights never "
                    "moving => eligibility never credited => substrate_not_ready_requeue."
                ),
                "control": "wlat_range_max + settling_round_delta (A2/A3/C3) + lcg_w_chan_range_max (A3)",
                "measured": float(min(
                    [r["wlat_range_max"] for r in (a2_rows + a3_rows + c3u_rows)]
                    + [r["lcg_w_chan_range_max"] for r in a3_rows] or [0.0]
                )),
                "threshold": float(WLAT_RANGE_FLOOR),
                "met": bool(wlat_moved_ok and lcg_moved_ok),
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
                "measured": float(min([r["consumed_summary_pairwise_dist_mean"] for r in all_rows] or [0.0])),
                "threshold": float(CONSUMED_SPREAD_FLOOR),
                "met": bool(enough_divergent),
            },
        ],
        "criteria": [
            {
                "name": "C1_conversion_settling_arm_above_A0_and_noise",
                "load_bearing": True,
                "passed": bool(c1_holds),
            },
            {
                "name": "C2_learning_load_bearing_grows_with_training_and_weights_moved",
                "load_bearing": True,
                "passed": bool(c2_holds),
            },
            {
                "name": "C3_signed_rpe_load_bearing_signed_converts_unsigned_fails",
                "load_bearing": False,
                "passed": bool(c3_signed_load_bearing),
            },
        ],
        "criteria_non_degenerate": {
            "preconditions_met": bool(preconditions_met),
            "enough_divergent_seeds": bool(enough_divergent),
            "noise_verified_lifting": bool(noise_verified_lifting),
            "field_noise_magnitude_matched": bool(field_noise_magnitude_matched),
            "wlat_moved": bool(wlat_moved_ok),
            "lcg_moved": bool(lcg_moved_ok),
            "delta_t_nonflat": bool(wlat_delta_nonflat_ok and lcg_delta_nonflat_ok),
            "c3_unsigned_engaged": bool(c3_unsigned_engaged),
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
            "w_chan_range_floor": float(W_CHAN_RANGE_FLOOR),
            "wlat_range_floor": float(WLAT_RANGE_FLOOR),
            "settling_round_delta_floor": float(SETTLING_ROUND_DELTA_FLOOR),
            "crf_min_minted": int(CRF_MIN_MINTED),
            "crf_frac_active_floor": float(CRF_FRAC_ACTIVE_FLOOR),
            "field_noise_wlat_scale": float(FIELD_NOISE_WLAT_SCALE),
            "field_noise_magnitude_match_lo": float(FIELD_NOISE_MAGNITUDE_MATCH_LO),
            "field_noise_magnitude_match_hi": float(FIELD_NOISE_MAGNITUDE_MATCH_HI),
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
            "wlat_delta_nonflat": wlat_delta_nonflat_ok,
            "lcg_delta_nonflat": lcg_delta_nonflat_ok,
            "wlat_moved": wlat_moved_ok,
            "lcg_moved": lcg_moved_ok,
            "noise_verified_lifting": noise_verified_lifting,
            "n_noise_lifts_over_a0": int(n_noise_lifts),
            "field_noise_magnitude_matched": field_noise_magnitude_matched,
            "field_noise_magnitude_ratio": round(float(field_noise_magnitude_ratio), 6),
            "median_noise_wlat_range": round(float(median_noise_wlat_range), 8),
            "median_settling_wlat_range": round(float(median_settling_wlat_range), 8),
            "C1_conversion": c1_holds,
            "C1_a2_settling_signed_converts": c1_a2,
            "C1_a3_both_signed_converts": c1_a3,
            "C1_c3_settling_unsigned_converts": c1_c3u,
            "C1_a2_n_seeds": int(n_c1_a2),
            "C1_a2_n_divergent": int(n_div_a2),
            "C1_a3_n_seeds": int(n_c1_a3),
            "C1_a3_n_divergent": int(n_div_a3),
            "C1_c3u_n_seeds": int(n_c1_c3u),
            "C1_c3u_n_divergent": int(n_div_c3u),
            "C2_learning_load_bearing": c2_holds,
            "C2_a2_grows": c2_a2,
            "C2_a3_grows": c2_a3,
            "C2_a2_n_grow_seeds": int(n_grow_a2),
            "C2_a3_n_grow_seeds": int(n_grow_a3),
            "C3_unsigned_engaged_non_vacuity": c3_unsigned_engaged,
            "C3_scoreable": c3_scoreable,
            "C3a_signed_converts": c3a_signed_converts,
            "C3b_unsigned_fails": c3b_unsigned_fails,
            "C3_signed_rpe_load_bearing": c3_signed_load_bearing,
            "C3_refuted_precision_reweighting": c3_refuted,
            "mean_committed_class_entropy_a0": round(a0_mean_dv, 6),
            "mean_committed_class_entropy_a2_settling_signed": round(a2_mean_dv, 6),
            "mean_committed_class_entropy_a3_both_signed": round(a3_mean_dv, 6),
            "mean_committed_class_entropy_c3_settling_unsigned": round(c3u_mean_dv, 6),
            "mean_committed_class_entropy_noise_same_layer": round(noise_mean_dv, 6),
        },
        "interpretation_grid": {
            "PASS_learned_gating_converts_where_arithmetic_plateaus": (
                "preconditions met (ENOUGH divergent seeds + VERIFIED-LIFTING SAME-LAYER "
                "noise bar + magnitude-matched + learning engaged) AND C1 (a settling arm "
                "committed-class entropy strict-above BOTH A0 and the verified same-layer "
                "noise control on a strict-majority of DIVERGENT seeds) AND C2 (the lift "
                "GROWS over training + the learned weights moved). The learned recurrent "
                "settling step (MECH-450 W_lat) -- and/or the learned channel re-weighting "
                "(ARC-108 w_chan via A3) -- CONVERTS committed-action diversity where the "
                "pure-arithmetic envelope (and a matched RANDOM settling-field perturbation) "
                "plateaus -> supports MECH-439 (ceiling liftable) + the learning claim(s). "
                "C3 (signed-RPE load-bearing): if C3a (signed converts) AND C3b (unsigned "
                "fails) with the unsigned arm engaged, the SIGN of the RPE is load-bearing -> "
                "ARC-108 supports; if the unsigned arm ALSO converts, route back to ARC-016 "
                "(ARC-108 mixed). PROMOTES NOTHING here (substrate_ceiling / substrate_conditional)."
            ),
            "PASS_static_reweighting_fold_into_arithmetic": (
                "preconditions met AND C1 (converts) BUT NOT C2 (the lift is present from "
                "tick 0 and does NOT grow with training). The conversion is a STATIC "
                "re-weighting, not learning -> MECH-439 supports (ceiling liftable) but "
                "ARC-108 / MECH-450 are MIXED; fold the winning static weights into the "
                "arithmetic lever, do NOT mint a learning claim."
            ),
            "FAIL_no_lift_preconditions_met_escalate_v4_full_loop": (
                "DECISIVE. preconditions met (learning engaged, weights moved, ENOUGH "
                "divergent seeds, the SAME-LAYER null VERIFIED-LIFTING + magnitude-matched) "
                "BUT NO entropy lift over A0/noise on the settling arms. The SAME-LAYER null "
                "verify-lifts AND the learned settling arms do NOT beat it => learned "
                "STRUCTURE adds nothing over a matched random settling-field perturbation at "
                "the same layer -> escalate to the V4 full BG-thalamo-cortical loop / "
                "loop-segregation (no more V3 letters; this run is PRE-REGISTERED TERMINAL). "
                "This is NOT a falsification of ARC-107 (the arithmetic envelope still holds) "
                "and does NOT promote learned gating. non_contributory for all claims."
            ),
            "FAIL_substrate_not_ready_requeue": (
                "PRE-REGISTERED TERMINAL: the field-noise scale may be re-tuned AT MOST ONCE. "
                "A precondition is unmet: too FEW divergent seeds (pool collapsed), OR delta_t "
                "flat (no outcome variance), OR the learned weights never moved (eligibility "
                "never credited) / the settling never moved the field, OR the SAME-LAYER "
                "matched-noise control did not VERIFY-LIFT above A0 on a strict-majority of "
                "divergent seeds, OR the frozen random W_lat magnitude did not MATCH the "
                "learned settling-arm range. The conversion question could NOT be measured -- "
                "NOT a falsification. Re-tune the FIELD-NOISE scale (the CORRECT layer) ONCE; "
                "a persistent failure to verify-lift / match magnitude at the correct layer "
                "ESCALATES to V4 loop-segregation, NEVER another V3 letter."
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
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "pre_registered_terminal": True,
        "terminal_escalation": (
            "V4 loop-segregation substrate (implement-substrate); NO further V3 letters"
        ),
        "re_derive_brake": {
            "exempt": True,
            "rationale": (
                "user-adjudicated 2026-06-24: runs concurrently with the V4 escalation "
                "(cannot delay it), genuine null REDESIGN (same-layer, not another alpha-bump "
                "iteration of the same lever), pre-registered terminal"
            ),
            "refuses": "any alpha-bump same-lever requeue",
        },
        "same_layer_null": {
            "layer": (
                "eligibility/settling field (_modulatory_accum[eligible_idx], the layer "
                "MECH-450 W_lat settling acts on)"
            ),
            "mechanism": (
                "frozen magnitude-matched random W_lat with learned_settling_eta=0.0, gated "
                "behind per-arm field_noise so A0/A2/A3/C3 stay byte-identical to 700b"
            ),
            "scale": FIELD_NOISE_WLAT_SCALE,
        },
        "reuse_baseline_from": REUSE_BASELINE_FROM,
        "evidence_direction_note": (
            f"V3-EXQ-700c ARC-108 sec-7 LEARNED-GATING CONVERSION FALSIFIER, brake-EXEMPT "
            f"TERMINAL RE-RUN with a SAME-LAYER null (experiment_purpose=evidence; "
            f"claim_ids=[MECH-439, ARC-108, MECH-450]; supersedes V3-EXQ-700b). Routed by the "
            f"user-adjudicated failure_autopsy_V3-EXQ-700b_2026-06-24 (CONCURRENT routing: V4 "
            f"loop-segregation + this terminal V3 700c). THE ONE SCIENTIFIC CHANGE FROM 700b: "
            f"replace the policy-softmax-temperature ARM_NOISE (MECH-313 use_noise_floor, the "
            f"WRONG layer -- decoupled from the committed-class DV by the F-bounded eligibility "
            f"constitution) with a SAME-LAYER null -- a FROZEN, magnitude-matched, random-"
            f"structure perturbation (random W_lat with learned_settling_eta=0.0) injected at "
            f"the eligibility/settling field, the EXACT layer MECH-450 settling acts on. A "
            f"settling-field perturbation is structurally COUPLED to the committed-class DV, so "
            f"it verify-lifts where the temperature null could not (alpha is the wrong knob; the "
            f"redesign moves the null into the coupled layer). PRE-REGISTERED TERMINAL: the "
            f"field-noise scale may be re-tuned AT MOST ONCE (the FIELD-NOISE scale, the correct "
            f"layer -- NOT an alpha bump); a persistent failure to verify-lift/match-magnitude "
            f"escalates to V4 loop-segregation, never another V3 letter. RE-DERIVE BRAKE EXEMPT "
            f"(user-adjudicated 2026-06-24: concurrent with the V4 escalation, genuine null "
            f"redesign, pre-registered terminal; the brake refuses any alpha-bump same-lever "
            f"requeue). The landed arithmetic envelope (use_f_eligibility_demotion + "
            f"adaptive_floor + go_nogo + modulatory-authority/top_k k=3) is a MATCHED CONSTANT "
            f"on all arms exactly as 700b; A0/A2/A3/C3 are byte-identical to 700b. PROMOTES "
            f"NOTHING until it scores (MECH-439 substrate_ceiling; ARC-108/MECH-450 "
            f"substrate_conditional; pending_retest_after_substrate). outcome={result['outcome']}; "
            f"label={result['interpretation_label']}; per_claim={result['evidence_direction_per_claim']}."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "5-arm focus-settling + C3 fold-in (A0 / A2_SETTLING_SIGNED / A3_BOTH_SIGNED / C3_SETTLING_UNSIGNED / ARM_NOISE same-layer null) + per-seed-divergent gating + verified-lifting magnitude-matched same-layer field-noise bar",
            "arms": "A0_ENVELOPE_ONLY / A2_SETTLING_SIGNED (W_lat, focus) / A3_BOTH_SIGNED (w_chan+W_lat) / C3_SETTLING_UNSIGNED (B5 ablation) / ARM_NOISE (frozen magnitude-matched random W_lat settling-field null, eta=0 -- NOT use_noise_floor)",
            "swept_variables": "use_learned_settling_step x use_learned_channel_gating x learned_channel_rpe_mode (+ same-layer field_noise frozen-random-W_lat on ARM_NOISE only; use_noise_floor OFF on every arm)",
            "the_one_change_from_700b": (
                "ARM_NOISE: a SAME-LAYER settling-field null (frozen magnitude-matched random "
                "W_lat with learned_settling_eta=0.0) replaces 700b's policy-softmax-temperature "
                "null (MECH-313 use_noise_floor). The new null acts INSIDE the eligible set on "
                "the same accumulator the committed argmin reads, so it is COUPLED to the "
                "committed-class DV (the temperature null was decoupled by the F-bounded "
                "eligibility constitution). A0/A2/A3/C3 are byte-identical to 700b."
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
            "primary_dv": "committed-action-class entropy (nats), interpreted on divergent seeds only",
            "phases": "P0 e2-train (CRF matures, learned gating ON) -> P1 frozen-encoder bias-head REINFORCE -> P2 e2+bias frozen, learned gating KEEPS adapting (binned first/second half for C2)",
            "learning_wiring": "w_chan/W_lat learn via e3.post_action_update driven by agent.update_residue every waking tick (all phases); use_habenula_decommit OFF; ARM_NOISE settling_eta=0 -> W_lat frozen",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "field_noise_wlat_scale": FIELD_NOISE_WLAT_SCALE,
            "field_noise_magnitude_match_band": [FIELD_NOISE_MAGNITUDE_MATCH_LO, FIELD_NOISE_MAGNITUDE_MATCH_HI],
            "reuse_baseline_from": REUSE_BASELINE_FROM,
            "reusable_arm_ids": list(REUSABLE_ARM_IDS_LOCAL),
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
            "supersedes": SUPERSEDES,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-700c ARC-108 sec-7 learned-gating terminal re-run (same-layer null)"
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
        f"noise_verified_lifting={result['acceptance_criteria']['noise_verified_lifting']} "
        f"field_noise_magnitude_matched={result['acceptance_criteria']['field_noise_magnitude_matched']} "
        f"C1={result['acceptance_criteria']['C1_conversion']} "
        f"C2={result['acceptance_criteria']['C2_learning_load_bearing']} "
        f"C3_load_bearing={result['acceptance_criteria']['C3_signed_rpe_load_bearing']} "
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
