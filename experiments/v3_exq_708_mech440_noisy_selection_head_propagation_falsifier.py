#!/opt/local/bin/python3
"""
V3-EXQ-708 -- MECH-440 NOISY-SELECTION-HEAD PROPAGATION FALSIFIER (diagnostic).

Falsifier for MECH-440 (state_conditioned_self_annealing_noise_floor; NoisyNet propagating
selection-head weight noise), built 2026-06-27 via /implement-substrate
(ree_core/policy/noisy_selection_head.py). EXPERIMENT_PURPOSE=diagnostic; PROMOTES NOTHING.
Decision-of-record: REE_assembly/evidence/decisions/cpkt_tonic_exploration_noise_build_decision_2026-06-27.md.

THE QUESTION (MECH-440)
----------------------
MECH-313's tonic exploration floor lifts the softmax TEMPERATURE; V3-EXQ-687 found that floor
NON-PROPAGATING -- the temperature was invisible to the committed argmax (selected_action_entropy
= 0.0, the r1a_entropy_only_artefact). MECH-440 asserts the floor must instead be injected as
LEARNED PER-PARAMETER (factorised-Gaussian) WEIGHT NOISE at the E3 selection head so it
PROPAGATES into the committed action. Does it? Specifically: does selection-head weight noise
raise COMMITTED-action-class entropy STRICTLY ABOVE a matched-pre-commit-variance TEMPERATURE
control (the 687 non-propagating arm) -- i.e. does the noise CARVE through to the commit rather
than THRASH at pre-commit and wash out at the argmax?

AND (the same-day-autopsy disambiguation, decision could_be_wrong_if #4): is the injection-SITE
the binding constraint, or is it the SINGLE-ARENA collapse (-> ARC-110)? The cluster autopsy
failure_autopsy_704b-706b-conversion-ceiling_2026-06-27 re-rooted the conversion ceiling to the
single F-dominated arena. So we must disambiguate the injection-site axis (weight noise vs
temperature) from the single-arena/ARC-110 axis (loop segregation OFF vs ON). If propagating
weight noise reproduces the single-arena ceiling (washes at argmax even on the SOTA stack),
route to /implement-substrate on ARC-110, NOT a deeper noise build.

THE 4 ARMS (all carry the SAME landed SOTA conversion stack -- 569i top-k shortlist +
MECH-448 demotion + Go/No-Go + modulatory authority + routing + finer gating + learned settling
-- as a MATCHED CONSTANT; the ONLY swept factor is the exploration-injection site):
  A0_OFF            : no exploration injection (use_noise_floor=False, use_noisy_selection_head
                      =False), single arena. The committed-entropy + pre-commit-entropy floor.
  ARM_TEMP          : the 687 NON-PROPAGATING matched-pre-commit-variance TEMPERATURE control --
                      use_noise_floor=True (a softmax-temperature lift that raises the PRE-COMMIT
                      sampling-distribution entropy), use_noisy_selection_head=False, single arena.
  ARM_NOISE_SINGLE  : use_noisy_selection_head=True (sigma_init>0) -- factorised-Gaussian weight
                      noise into the committed within-eligible argmin -- single arena
                      (use_loop_segregation=False). The MECH-440 injection-site arm.
  ARM_NOISE_LOOPSEG : ARM_NOISE_SINGLE + use_loop_segregation=True (ARC-110 motor/assoc/limbic +
                      ARC-109 D1/D2 + MECH-452 loop-local). The disambiguation arm: does the
                      weight noise convert only under the multi-arena substrate?
6 seeds. PRIMARY DV = committed-action-class entropy (nats), P2. SECONDARY DV = PRE-COMMIT
sampling-class entropy (nats; from e3.last_precommit_probs) -- the thrash-vs-carve discriminator.
claim_ids = [MECH-440]. experiment_purpose = diagnostic (PROMOTES NOTHING).

PRE-REGISTERED OUTCOME MAP (decisive either way)
------------------------------------------------
  PASS / supports MECH-440 (injection-site fix works on the single arena):
    ARM_NOISE_SINGLE committed-class entropy strict-above ARM_TEMP (the matched-pre-commit-
    variance temperature control) on a strict-majority of DIVERGENT seeds. The weight noise
    CARVES through to the committed action where the temperature THRASHES at pre-commit.

  ROUTE-TO-ARC-110 (injection-site fix gated on the multi-arena):
    ARM_NOISE_SINGLE does NOT convert (committed ~ ARM_TEMP) BUT ARM_NOISE_LOOPSEG does
    (committed strict-above ARM_TEMP AND strict-above ARM_NOISE_SINGLE, with live cross-loop
    variance). The propagation needs loop segregation -> route /implement-substrate ARC-110.

  WEAKENED / route-to-ARC-110 (single-arena collapse subsumes the injection-locus fix):
    NEITHER ARM_NOISE_SINGLE NOR ARM_NOISE_LOOPSEG converts committed diversity strict-above
    ARM_TEMP, despite both genuinely raising PRE-COMMIT entropy (matched to ARM_TEMP). The
    weight noise washes at the argmax even on the SOTA stack -- thrash, not carve -- so the
    binding constraint is the single F-dominated arena, not the injection site (the autopsy's
    could_be_wrong_if #4 realised) -> weakens MECH-440 as a standalone fix; route ARC-110.

NON-VACUITY READINESS GATES (self-route substrate_not_ready_requeue, NEVER a false weakens):
  (1) candidate pool DIVERGENT: GAP-A cand_world_pairwise_dist > floor (per seed).
  (2) BOTH ARM_TEMP and the NOISE arms genuinely RAISE pre-commit entropy strict-above A0_OFF
      (the "matched-pre-commit-variance" non-vacuity: a temperature control that does not lift
      pre-commit, or a weight noise that injects no pre-commit variance, makes the carve test
      meaningless). measured per arm.
  (3) the NOISE arms inject a SUPRA-FLOOR per-candidate noise bias range (noisy_selection_
      bias_range > floor): a noise too small to ever flip the within-eligible argmin is a
      vacuous test -> requeue.
  (4) dACC non-vacuity: dacc_max_suppression > 0 on the dACC-bearing arms (the Go/No-Go
      perseveration axis is live; a flat dACC means the constitution arm is inert).
  (5) ARM_NOISE_LOOPSEG carries LIVE cross-loop variance (loop_committed_neq_motor_winner OR
      loop_cross_loop_winner_disagreement + per-loop pref range > 0) -- else the loop-seg
      disambiguation arm is a vacuous split pinned to the motor winner.

ARC-106: MECH-440's load-bearing-vs-decorative ablation is A0_OFF (the noise OFF reproduces the
687 non-propagation) vs ARM_NOISE_SINGLE (ON propagates). Divergences logged in the grounding
ledger (per-parameter sigma below the systems-level tonic/phasic gate; sigma annealed by LOCAL
EMA not RL gradient). MECH-094: the noise is waking-only (simulation_mode -> zero perturbation).
Phased P0/P1/P2 kept for a fair comparison with the 707/704b sibling harness.

COORDINATE (do not duplicate): the empirical 687-successor (behavioral_diversity_isolation:GAP-C
leg) tests the EMPIRICAL conversion; this tests the MECH-440 INJECTION-SITE mechanism. The
705/706 lineage is the CURIOSITY channel (MECH-441/MECH-314), a different leg.

See REE_assembly/docs/architecture/state_conditioned_exploration_noise_floor.md (#mech-440),
    ree-v3/ree_core/policy/noisy_selection_head.py (NoisySelectionHead),
    experiments/v3_exq_707_arc110_loop_segregation_validation.py (matched-substrate sibling harness).
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


EXPERIMENT_TYPE = "v3_exq_708_mech440_noisy_selection_head_propagation_falsifier"
QUEUE_ID = "V3-EXQ-708"
SUPERSEDES = None
BACKLOG_ID = None   # routed by cpkt_tonic_exploration_noise_build_decision_2026-06-27
CLAIM_IDS: List[str] = ["MECH-440"]
EXPERIMENT_PURPOSE = "diagnostic"   # PROMOTES NOTHING

_FCG_W_INIT = math.log(math.e - 1.0)

# CRF-gate calibration levers (matured CRF stack; ported from 707, matched on all arms).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# ----- Acceptance thresholds (pre-registered) -----
# C1 conversion: strict-above margin on committed-class entropy (nats).
CONVERSION_MARGIN = 0.05
# Non-vacuity: an arm "raises pre-commit entropy" iff its pre-commit-class entropy is at least
# this margin above A0_OFF's pre-commit-class entropy.
PRECOMMIT_LIFT_MARGIN = 0.05

# ----- Per-seed-divergent gating -----
MIN_DIVERGENT_SEEDS = 3          # of 6: fewer divergent seeds => substrate_not_ready_requeue
DIVERGENT_PASS_FRACTION = 0.5    # strict-majority-ish gate within the divergent seeds
MIN_SEEDS_FOR_PASS = 2           # absolute floor of divergent seeds clearing a criterion

# Non-vacuity: the noise arms must inject a per-candidate bias range above this floor, AND it
# must be a non-trivial fraction of the raw-score range (so it CAN flip the within-eligible
# argmin). A noise too small to ever compete is a vacuous test -> requeue.
NOISE_BIAS_RANGE_FLOOR = 1e-4
NOISE_BIAS_TO_RAW_RANGE_FRAC_FLOOR = 0.02

# dACC non-vacuity: the Go/No-Go perseveration axis must be live (max suppression > 0).
DACC_MAX_SUPPRESSION_FLOOR = 0.0   # strictly > this

# ARC-110 loop non-degeneracy (only on ARM_NOISE_LOOPSEG).
LOOP_CROSS_VARIANCE_FRAC_FLOOR = 0.05
LOOP_PREF_RANGE_FLOOR = 1e-6

# C1(a) readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# Non-vacuity (b): GAP-A consumed-summary divergence.
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# Non-vacuity (c): delta_t carries cross-tick variance.
DELTA_T_STD_FLOOR = 1e-4
# Finer-channel learning engaged (matched constant).
W_CHAN_FINER_RANGE_FLOOR = 1e-4

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

# --- MECH-440 injection-site lever constants ---
# Weight-noise arms: factorised-Gaussian sigma_init + overall weight on the per-candidate bias.
NOISY_SELECTION_SIGMA_INIT = 1.0
NOISY_SELECTION_WEIGHT = 1.0
NOISY_SELECTION_ANNEAL = True
NOISY_SELECTION_ANNEAL_FLOOR = 0.1
NOISY_SELECTION_ANNEAL_EMA_ALPHA = 0.01
# ARM_TEMP: the matched-pre-commit-variance temperature control (the 687 non-propagating lift).
# effective_T = max(base_T + alpha, min_temperature); base_T defaults to 1.0 (select_action).
TEMP_NOISE_FLOOR_ALPHA = 1.5
TEMP_NOISE_FLOOR_MIN_TEMPERATURE = 1.0

# --- Matched-stack lever constants (identical on ALL arms; the landed SOTA conversion stack) ---
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

LCG_ETA = 0.01
LCG_ELIG_DECAY = 0.9
LCG_VALUE_BASELINE_BETA = 0.05
LCG_ASYM_POTENTIATION = 1.0
LCG_ASYM_DEPRESSION = 0.5

LEARNED_SETTLING_ROUNDS = 3
LEARNED_SETTLING_TEMPERATURE = 1.0
LEARNED_SETTLING_ETA = 0.01
LEARNED_SETTLING_ELIG_DECAY = 0.9

LOOP_SEGREGATION_NORMALIZE = "zscore"

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

LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9


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


# The 4 arms. ALL carry the matched SOTA conversion stack + finer gating + learned settling;
# the ONLY swept factor is the exploration-injection site (none / temperature / weight-noise /
# weight-noise+loop-segregation).
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "A0_OFF",
        "label": "no_exploration_injection_single_arena_baseline",
        "noise_head": False, "temp": False, "loop_seg": False,
    },
    {
        "arm_id": "ARM_TEMP",
        "label": "matched_pre_commit_variance_temperature_control_687_non_propagating",
        "noise_head": False, "temp": True, "loop_seg": False,
    },
    {
        "arm_id": "ARM_NOISE_SINGLE",
        "label": "mech440_noisy_selection_head_weight_noise_single_arena",
        "noise_head": True, "temp": False, "loop_seg": False,
    },
    {
        "arm_id": "ARM_NOISE_LOOPSEG",
        "label": "mech440_noisy_selection_head_weight_noise_plus_arc110_loop_segregation",
        "noise_head": True, "temp": False, "loop_seg": True,
    },
]


def _arm_config_slice(
    arm: Dict[str, Any],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Declared reuse fingerprint slice: ONLY what an arm's computation reads."""
    return {
        "arm_id": arm["arm_id"],
        "noise_head": bool(arm["noise_head"]),
        "temp": bool(arm["temp"]),
        "loop_seg": bool(arm["loop_seg"]),
        "noisy_selection_sigma_init": float(NOISY_SELECTION_SIGMA_INIT),
        "noisy_selection_weight": float(NOISY_SELECTION_WEIGHT),
        "temp_noise_floor_alpha": float(TEMP_NOISE_FLOOR_ALPHA),
        "temp_noise_floor_min_temperature": float(TEMP_NOISE_FLOOR_MIN_TEMPERATURE),
        "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
        "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
        "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
        "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
        "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
        "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
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
    """Matched-stack agent. The landed SOTA conversion stack (569i top-k + MECH-448 demotion +
    adaptive floor + Go/No-Go + modulatory authority + routing) + the diversity stack (MECH-341,
    SD-056, CRF, trained lateral_pfc bias head, use_dacc) + finer-channel gating + learned
    settling are MATCHED CONSTANTS on ALL arms. The ONLY swept factor is the exploration-
    injection site: ARM_TEMP arms MECH-313 noise_floor (a softmax-temperature lift); the
    NOISE arms arm MECH-440 use_noisy_selection_head (factorised-Gaussian weight noise into the
    committed argmin); ARM_NOISE_LOOPSEG also arms ARC-110 loop segregation."""
    noise_head = bool(arm["noise_head"])
    temp = bool(arm["temp"])
    loop_seg = bool(arm["loop_seg"])
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
        # --- Matched stack ---
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
        # --- MECH-313 temperature noise floor: ARMED ONLY on ARM_TEMP (the 687 control) ---
        use_noise_floor=temp,
        noise_floor_alpha=(TEMP_NOISE_FLOOR_ALPHA if temp else 0.1),
        noise_floor_min_temperature=(TEMP_NOISE_FLOOR_MIN_TEMPERATURE if temp else 1.0),
        # --- MECH-440 NoisyNet propagating selection-head weight noise: ARMED on the NOISE arms ---
        use_noisy_selection_head=noise_head,
        noisy_selection_sigma_init=(NOISY_SELECTION_SIGMA_INIT if noise_head else 0.0),
        noisy_selection_weight=NOISY_SELECTION_WEIGHT,
        noisy_selection_anneal=NOISY_SELECTION_ANNEAL,
        noisy_selection_anneal_floor=NOISY_SELECTION_ANNEAL_FLOOR,
        noisy_selection_anneal_ema_alpha=NOISY_SELECTION_ANNEAL_EMA_ALPHA,
        # MECH-441 OFF (this falsifier is the MECH-440 leg only).
        use_model_disagreement_curiosity=False,
        # V_s minimal stack.
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
        # MECH-451 finer channels ON on all arms (the loops partition these; matched constant).
        use_finer_channel_gating=True,
        use_learned_channel_gating=False,
        learned_channel_gating_eta=LCG_ETA,
        learned_channel_gating_elig_decay=LCG_ELIG_DECAY,
        learned_channel_value_baseline_beta=LCG_VALUE_BASELINE_BETA,
        learned_channel_asym_potentiation=LCG_ASYM_POTENTIATION,
        learned_channel_asym_depression=LCG_ASYM_DEPRESSION,
        learned_channel_rpe_mode="signed",
        use_learned_settling_step=True,
        learned_settling_rounds=LEARNED_SETTLING_ROUNDS,
        learned_settling_temperature=LEARNED_SETTLING_TEMPERATURE,
        learned_settling_eta=LEARNED_SETTLING_ETA,
        learned_settling_elig_decay=LEARNED_SETTLING_ELIG_DECAY,
        # --- ARC-110 loop segregation: ARMED ONLY on ARM_NOISE_LOOPSEG (the disambiguation arm) ---
        use_loop_segregation=loop_seg,
        loop_segregation_normalize=LOOP_SEGREGATION_NORMALIZE,
        use_d1_d2_population_split=loop_seg,
        use_loop_local_eligibility_traces=loop_seg,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (verbatim from 707)
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
        z_world_0=z0_K, actions=actions_K, z_world_1_targets=z1_K, simulation_mode=False,
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


def _entropy_from_prob_dict(probs: Dict[int, float]) -> float:
    tot = sum(probs.values())
    if tot <= 0:
        return 0.0
    h = 0.0
    for p in probs.values():
        q = p / tot
        if q > 0:
            h -= q * math.log(q)
    return float(h)


# ---------------------------------------------------------------------------
# P1 bias-head REINFORCE training (verbatim from 707)
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
    # SECONDARY DV: PRE-COMMIT sampling-class marginal (sum of softmax class-mass over ticks).
    precommit_class_mass: Dict[int, float] = {}
    n_precommit_ticks = 0
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0

    # MECH-440 noise diagnostics (P2).
    noise_bias_ranges: List[float] = []
    raw_score_ranges: List[float] = []
    dacc_max_suppression = 0.0

    # CRF maturity readiness (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_minted_total_last = 0

    # Learning diagnostics.
    lcg_delta_ts: List[float] = []
    fcg_delta_ts: List[float] = []
    fcg_w_chan_finer_range_max = 0.0
    n_select_ticks = 0

    # ARC-110 loop diagnostics (only meaningful on ARM_NOISE_LOOPSEG).
    loop_committed_neq_motor_ticks = 0
    loop_disagree_ticks = 0
    loop_assoc_range_sum = 0.0
    loop_limbic_range_sum = 0.0
    n_loop_diag_ticks = 0

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
            n_select_ticks += 1

            if is_p2:
                diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
                # MECH-440 noise non-vacuity: per-candidate bias range + raw-score range.
                nbr = float(diag.get("noisy_selection_bias_range", 0.0) or 0.0)
                if nbr > 0.0:
                    noise_bias_ranges.append(nbr)
                rsr = float(diag.get("e3_raw_score_range_mean", 0.0) or 0.0)
                if rsr > 0.0:
                    raw_score_ranges.append(rsr)
                # ARC-110 loop non-degeneracy (loop arm only).
                if diag.get("loop_segregation_active", False):
                    n_loop_diag_ticks += 1
                    if diag.get("loop_committed_neq_motor_winner", False):
                        loop_committed_neq_motor_ticks += 1
                    if diag.get("loop_cross_loop_winner_disagreement", False):
                        loop_disagree_ticks += 1
                    loop_assoc_range_sum += float(diag.get("loop_assoc_pref_range", 0.0) or 0.0)
                    loop_limbic_range_sum += float(diag.get("loop_limbic_pref_range", 0.0) or 0.0)
                # dACC non-vacuity: max perseveration suppression magnitude.
                bundle = getattr(agent, "_dacc_last_bundle", None)
                if bundle is not None:
                    supp = bundle.get("suppression", None)
                    if supp is not None:
                        try:
                            dacc_max_suppression = max(
                                dacc_max_suppression, float(supp.detach().abs().max().item())
                            )
                        except Exception:
                            pass
                # SECONDARY DV: pre-commit sampling-class marginal from last_precommit_probs.
                pp = getattr(agent.e3, "last_precommit_probs", None)
                if pp is not None and candidates is not None and len(candidates) >= 1:
                    try:
                        ppv = pp.detach().reshape(-1)
                        if ppv.numel() == len(candidates) and torch.isfinite(ppv).all():
                            n_precommit_ticks += 1
                            for ci, c in enumerate(candidates):
                                cls = _traj_first_action_class(c)
                                precommit_class_mass[cls] = (
                                    precommit_class_mass.get(cls, 0.0) + float(ppv[ci].item())
                                )
                    except Exception:
                        pass

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
            with torch.no_grad():
                resid_metrics = agent.update_residue(
                    harm_signal=float(_harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )
            ldt = resid_metrics.get("e3_lcg_delta_t")
            if ldt is not None:
                lcg_delta_ts.append(float(ldt.item()))
            fdt = resid_metrics.get("e3_fcg_delta_t")
            if fdt is not None:
                fcg_delta_ts.append(float(fdt.item()))
            fwr = resid_metrics.get("e3_fcg_w_chan_finer_range")
            if fwr is not None:
                fcg_w_chan_finer_range_max = max(
                    fcg_w_chan_finer_range_max, float(fwr.item())
                )

            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure, drive_level=drive_level,
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

    committed_class_entropy = _entropy_from_int_counts(committed_class_counts)
    precommit_class_entropy = _entropy_from_prob_dict(precommit_class_mass)

    frac_pre_ge2 = float(n_p2_pre_ge2 / n_p2_ticks) if n_p2_ticks > 0 else 0.0
    consumed_spread_mean = (
        float(sum(consumed_dists) / len(consumed_dists)) if consumed_dists else 0.0
    )

    if crf_n_active_per_tick:
        frac_crf_active_ge_floor = float(
            sum(1 for n in crf_n_active_per_tick if n >= 1) / len(crf_n_active_per_tick)
        )
    else:
        frac_crf_active_ge_floor = 0.0

    lcg_delta_t_std = float(statistics.pstdev(lcg_delta_ts)) if len(lcg_delta_ts) >= 2 else 0.0
    fcg_delta_t_std = float(statistics.pstdev(fcg_delta_ts)) if len(fcg_delta_ts) >= 2 else 0.0

    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )
    fcg_moved = bool(fcg_w_chan_finer_range_max > W_CHAN_FINER_RANGE_FLOOR)
    fcg_delta_nonflat = bool(fcg_delta_t_std > DELTA_T_STD_FLOOR)

    noise_bias_range_mean = (
        float(sum(noise_bias_ranges) / len(noise_bias_ranges)) if noise_bias_ranges else 0.0
    )
    raw_score_range_mean = (
        float(sum(raw_score_ranges) / len(raw_score_ranges)) if raw_score_ranges else 0.0
    )
    noise_to_raw_frac = (
        noise_bias_range_mean / raw_score_range_mean if raw_score_range_mean > 1e-9 else 0.0
    )

    loop_n = max(n_loop_diag_ticks, 1)
    loop_frac_committed_neq_motor = (
        float(loop_committed_neq_motor_ticks / loop_n) if n_loop_diag_ticks else 0.0
    )
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

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "noise_head": bool(arm["noise_head"]),
        "temp": bool(arm["temp"]),
        "loop_seg": bool(arm["loop_seg"]),
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
        # ----- PRIMARY + SECONDARY DV -----
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "precommit_class_entropy_nats": round(precommit_class_entropy, 6),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        "n_precommit_ticks": int(n_precommit_ticks),
        # ----- MECH-440 noise non-vacuity -----
        "noise_bias_range_mean": round(noise_bias_range_mean, 8),
        "raw_score_range_mean": round(raw_score_range_mean, 8),
        "noise_to_raw_range_frac": round(noise_to_raw_frac, 6),
        "dacc_max_suppression": round(dacc_max_suppression, 8),
        # ----- ARC-110 loop diagnostics (loop arm only) -----
        "loop_frac_committed_neq_motor": round(loop_frac_committed_neq_motor, 6),
        "loop_frac_disagree": round(loop_frac_disagree, 6),
        "loop_assoc_pref_range": round(loop_assoc_range_mean, 6),
        "loop_limbic_pref_range": round(loop_limbic_range_mean, 6),
        "loop_cross_variance": seed_loop_cross_variance,
        # ----- Readiness / non-vacuity -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "lcg_delta_t_std": round(lcg_delta_t_std, 8),
        "fcg_delta_t_std": round(fcg_delta_t_std, 8),
        "fcg_w_chan_finer_range_max": round(fcg_w_chan_finer_range_max, 8),
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
            f"Arm {arm['arm_id']} ({arm['label']}) noise_head={arm['noise_head']} "
            f"temp={arm['temp']} loop_seg={arm['loop_seg']} "
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
            # Per-cell fingerprint. A0_OFF is a stable single-arena baseline (self-mint
            # eligible). The exploration arms ride the just-built MECH-440 / ARC-110
            # substrate (in flux for this lineage) -- not reusable baselines.
            if arm["arm_id"] == "A0_OFF":
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(
                        arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                    ),
                    seed=s, script_path=script_path,
                    rng_fully_reset=True, config_slice_declared=True,
                    include_driver_script_in_hash=False,
                )
            else:
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(
                        arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                    ),
                    seed=s, script_path=script_path,
                    rng_fully_reset=True,
                    extra_ineligible_reasons=[
                        "mech440_noise_substrate_just_built_in_flux_not_a_reusable_baseline",
                    ],
                )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    off_rows = _arm_rows(arm_results, "A0_OFF")
    temp_rows = _arm_rows(arm_results, "ARM_TEMP")
    nsingle_rows = _arm_rows(arm_results, "ARM_NOISE_SINGLE")
    nloop_rows = _arm_rows(arm_results, "ARM_NOISE_LOOPSEG")
    all_rows = off_rows + temp_rows + nsingle_rows + nloop_rows

    def _maj(rows: List[Dict[str, Any]], pred) -> bool:
        return sum(1 for r in rows if pred(r)) >= MIN_SEEDS_FOR_PASS

    off_ent = _by_seed(off_rows, "committed_class_entropy_nats")
    temp_ent = _by_seed(temp_rows, "committed_class_entropy_nats")
    nsingle_ent = _by_seed(nsingle_rows, "committed_class_entropy_nats")
    nloop_ent = _by_seed(nloop_rows, "committed_class_entropy_nats")

    off_pre = _by_seed(off_rows, "precommit_class_entropy_nats")
    temp_pre = _by_seed(temp_rows, "precommit_class_entropy_nats")
    nsingle_pre = _by_seed(nsingle_rows, "precommit_class_entropy_nats")

    off_gap = _gap_by_seed(off_rows)
    temp_gap = _gap_by_seed(temp_rows)
    nsingle_gap = _gap_by_seed(nsingle_rows)

    # Divergent seeds: pool divergent on the C1 comparison arms (OFF + TEMP + NOISE_SINGLE).
    primary_div = [
        s for s in sorted(set(off_gap) & set(temp_gap) & set(nsingle_gap))
        if off_gap.get(s) and temp_gap.get(s) and nsingle_gap.get(s)
    ]
    n_primary_div = len(primary_div)
    enough_divergent = n_primary_div >= MIN_DIVERGENT_SEEDS

    # ----- Precondition: matched-pre-commit-variance is NON-VACUOUS -- BOTH ARM_TEMP and
    # ARM_NOISE_SINGLE genuinely RAISE pre-commit entropy strict-above A0_OFF on a
    # strict-majority of divergent seeds. A temperature control that doesn't lift pre-commit
    # (or a weight noise that injects no pre-commit variance) makes the carve test meaningless. -----
    n_temp_pre_lift = sum(
        1 for s in primary_div if temp_pre.get(s, 0.0) > off_pre.get(s, 0.0) + PRECOMMIT_LIFT_MARGIN
    )
    n_noise_pre_lift = sum(
        1 for s in primary_div if nsingle_pre.get(s, 0.0) > off_pre.get(s, 0.0) + PRECOMMIT_LIFT_MARGIN
    )
    temp_lifts_precommit = bool(enough_divergent and _div_pass(n_temp_pre_lift, n_primary_div))
    noise_lifts_precommit = bool(enough_divergent and _div_pass(n_noise_pre_lift, n_primary_div))

    # ----- Precondition: the NOISE arms inject a SUPRA-FLOOR per-candidate bias range AND it
    # is a non-trivial fraction of the raw-score range (can flip the within-eligible argmin). -----
    noise_bias_supra_floor = _maj(
        nsingle_rows,
        lambda r: r.get("noise_bias_range_mean", 0.0) > NOISE_BIAS_RANGE_FLOOR
        and r.get("noise_to_raw_range_frac", 0.0) > NOISE_BIAS_TO_RAW_RANGE_FRAC_FLOOR,
    )

    # ----- Precondition: dACC non-vacuity (Go/No-Go perseveration axis live) on a majority
    # of seeds across the dACC-bearing arms (all arms use dACC). -----
    dacc_live = all(
        _maj(rows, lambda r: r.get("dacc_max_suppression", 0.0) > DACC_MAX_SUPPRESSION_FLOOR)
        for rows in (off_rows, temp_rows, nsingle_rows, nloop_rows) if rows
    )

    # ----- Precondition: learning engaged (finer channels dissociable + delta_t nonflat). -----
    fcg_moved_ok = _maj(nsingle_rows, lambda r: r.get("fcg_moved", False))
    fcg_delta_nonflat_ok = _maj(nsingle_rows, lambda r: r.get("fcg_delta_nonflat", False))

    # ----- Precondition (ARC-110 disambiguation arm non-degeneracy): ARM_NOISE_LOOPSEG carries
    # LIVE cross-loop variance (else the loop-seg disambiguation is a vacuous split). -----
    loop_cross_variance_ok = _maj(nloop_rows, lambda r: r.get("loop_cross_variance", False))

    preconditions_met = bool(
        enough_divergent
        and temp_lifts_precommit and noise_lifts_precommit
        and noise_bias_supra_floor
        and dacc_live
        and fcg_moved_ok and fcg_delta_nonflat_ok
    )

    # ----- C1 (MECH-440 single-arena propagation): ARM_NOISE_SINGLE committed-class entropy
    # strict-above ARM_TEMP (the matched-pre-commit-variance temperature control) on a
    # strict-majority of divergent seeds. The weight noise CARVES where the temperature THRASHES. -----
    c1_seeds = [
        s for s in primary_div
        if nsingle_ent.get(s, 0.0) > temp_ent.get(s, 0.0) + CONVERSION_MARGIN
    ]
    c1_holds = _div_pass(len(c1_seeds), n_primary_div)

    # ----- Disambiguation: ARM_NOISE_LOOPSEG converts (committed strict-above ARM_TEMP AND
    # strict-above ARM_NOISE_SINGLE) on a strict-majority of divergent seeds where the loop
    # arm is divergent + carries live cross-loop variance. -----
    loop_div = [
        s for s in primary_div
        if s in nloop_ent and any(r["seed"] == s and r.get("loop_cross_variance", False) for r in nloop_rows)
    ]
    loop_conv_seeds = [
        s for s in loop_div
        if nloop_ent.get(s, 0.0) > temp_ent.get(s, 0.0) + CONVERSION_MARGIN
        and nloop_ent.get(s, 0.0) > nsingle_ent.get(s, 0.0) + CONVERSION_MARGIN
    ]
    loop_converts = bool(
        len(loop_div) >= MIN_SEEDS_FOR_PASS and _div_pass(len(loop_conv_seeds), len(loop_div))
    )

    # ----- Outcome map -----
    if not preconditions_met:
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "MECH-440 propagation could NOT be validly measured: a precondition is unmet "
            "(too few divergent seeds / ARM_TEMP did not raise pre-commit entropy / the weight "
            "noise injected no pre-commit variance / noise bias range below floor or trivial vs "
            "raw range / dACC suppression flat / finer channels not dissociable / delta_t flat). "
            "NOT a falsification."
        )
        per_claim = {"MECH-440": "non_contributory"}
    elif c1_holds:
        outcome = "PASS"
        overall_direction = "supports"
        non_degenerate = True
        degeneracy_reason = ""
        label = "weight_noise_carves_committed_diversity_above_matched_temperature_supports_mech440"
        per_claim = {"MECH-440": "supports"}
    elif loop_converts:
        outcome = "FAIL"
        overall_direction = "non_contributory"
        non_degenerate = True
        degeneracy_reason = ""
        label = "injection_site_fix_gated_on_arc110_route_implement_substrate_loop_segregation"
        per_claim = {"MECH-440": "non_contributory"}
    else:
        outcome = "FAIL"
        overall_direction = "weakens"
        non_degenerate = True
        degeneracy_reason = ""
        label = "weight_noise_washes_at_argmax_single_arena_subsumes_injection_locus_route_arc110_weakens_mech440"
        per_claim = {"MECH-440": "weakens"}

    off_mean_dv = _mean([r["committed_class_entropy_nats"] for r in off_rows])
    temp_mean_dv = _mean([r["committed_class_entropy_nats"] for r in temp_rows])
    nsingle_mean_dv = _mean([r["committed_class_entropy_nats"] for r in nsingle_rows])
    nloop_mean_dv = _mean([r["committed_class_entropy_nats"] for r in nloop_rows])

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "enough_divergent_seeds",
                "kind": "readiness",
                "description": (
                    "seeds whose candidate pool is DIVERGENT on ALL C1 comparison arms "
                    "(A0_OFF + ARM_TEMP + ARM_NOISE_SINGLE) >= MIN_DIVERGENT_SEEDS. Too few "
                    "=> substrate_not_ready_requeue."
                ),
                "control": "consumed cand_world_summary pairwise spread > floor (GAP-A); per-seed",
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                "met": bool(enough_divergent),
            },
            {
                "name": "temperature_control_raises_precommit_entropy",
                "kind": "readiness",
                "description": (
                    "ARM_TEMP (the 687 temperature control) raises PRE-COMMIT sampling-class "
                    "entropy strict-above A0_OFF by margin on a strict-majority of divergent "
                    "seeds. A temperature control that does NOT lift pre-commit is not a valid "
                    "matched-pre-commit-variance control => substrate_not_ready_requeue. measured "
                    "= n seeds ARM_TEMP lifts pre-commit."
                ),
                "control": "ARM_TEMP precommit_class_entropy vs A0_OFF, divergent seeds, paired",
                "measured": float(n_temp_pre_lift),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "met": bool(temp_lifts_precommit),
            },
            {
                "name": "weight_noise_raises_precommit_entropy",
                "kind": "readiness",
                "description": (
                    "ARM_NOISE_SINGLE raises PRE-COMMIT sampling-class entropy strict-above "
                    "A0_OFF by margin on a strict-majority of divergent seeds -- the weight noise "
                    "genuinely injects pre-commit variance (so the matched comparison with "
                    "ARM_TEMP is fair). Below floor => substrate_not_ready_requeue. measured = n "
                    "seeds ARM_NOISE_SINGLE lifts pre-commit."
                ),
                "control": "ARM_NOISE_SINGLE precommit_class_entropy vs A0_OFF, divergent seeds, paired",
                "measured": float(n_noise_pre_lift),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "met": bool(noise_lifts_precommit),
            },
            {
                "name": "noise_bias_range_supra_floor_vs_raw",
                "kind": "readiness",
                "description": (
                    "ARM_NOISE_SINGLE per-candidate noise bias range > floor AND a non-trivial "
                    "fraction of the raw-score range (so the noise CAN flip the within-eligible "
                    "argmin). A noise too small to ever compete is a vacuous test => requeue. "
                    "measured = min noise_bias_range_mean across ARM_NOISE_SINGLE seeds."
                ),
                "control": "ARM_NOISE_SINGLE noisy_selection_bias_range / e3_raw_score_range",
                "measured": float(min([r.get("noise_bias_range_mean", 0.0) for r in nsingle_rows] or [0.0])),
                "threshold": float(NOISE_BIAS_RANGE_FLOOR),
                "met": bool(noise_bias_supra_floor),
            },
            {
                "name": "dacc_suppression_live",
                "kind": "readiness",
                "description": (
                    "dACC perseveration No-Go axis is LIVE (dacc_max_suppression > 0) on a "
                    "majority of seeds across the dACC-bearing arms. A flat dACC means the "
                    "Go/No-Go constitution is inert => the SOTA stack is not actually engaged."
                ),
                "control": "agent._dacc_last_bundle['suppression'] max over P2",
                "measured": float(min([r.get("dacc_max_suppression", 0.0) for r in nsingle_rows] or [0.0])),
                "threshold": float(DACC_MAX_SUPPRESSION_FLOOR),
                "met": bool(dacc_live),
            },
            {
                "name": "loopseg_arm_carries_live_cross_loop_variance",
                "kind": "readiness",
                "description": (
                    "ARM_NOISE_LOOPSEG carries LIVE cross-loop variance (a non-motor loop "
                    "flipped the within-eligible winner / loops disagreed + per-loop pref range "
                    "> 0) on a majority of seeds -- so the ARC-110 disambiguation arm is not a "
                    "vacuous split pinned to the motor winner. (Gates the loop_converts branch "
                    "only; not required for the C1 single-arena verdict.)"
                ),
                "control": "ARM_NOISE_LOOPSEG loop_frac_committed_neq_motor / loop_frac_disagree + per-loop range",
                "measured": float(sum(1 for r in nloop_rows if r.get("loop_cross_variance", False))),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "met": bool(loop_cross_variance_ok),
            },
            {
                "name": "learning_engaged_finer_channels_dissociable",
                "kind": "readiness",
                "description": (
                    "on ARM_NOISE_SINGLE the finer w_chan_finer entries MOVED + the signed-RPE "
                    "delta_t carries cross-tick variance, on a majority of seeds. Below floor => "
                    "substrate_not_ready_requeue."
                ),
                "control": "ARM_NOISE_SINGLE fcg_w_chan_finer_range_max + fcg_delta_t_std",
                "measured": float(min([r["fcg_w_chan_finer_range_max"] for r in nsingle_rows] or [0.0])),
                "threshold": float(W_CHAN_FINER_RANGE_FLOOR),
                "met": bool(fcg_moved_ok and fcg_delta_nonflat_ok),
            },
        ],
        "criteria": [
            {
                "name": "C1_noise_single_committed_strict_above_temp",
                "load_bearing": True,
                "passed": bool(c1_holds),
            },
            {
                "name": "DISAMBIG_loopseg_converts_above_temp_and_single",
                "load_bearing": False,
                "passed": bool(loop_converts),
            },
        ],
        "criteria_non_degenerate": {
            "preconditions_met": bool(preconditions_met),
            "enough_divergent_seeds": bool(enough_divergent),
            "temp_lifts_precommit": bool(temp_lifts_precommit),
            "noise_lifts_precommit": bool(noise_lifts_precommit),
            "noise_bias_supra_floor": bool(noise_bias_supra_floor),
            "dacc_live": bool(dacc_live),
            "learning_engaged": bool(fcg_moved_ok and fcg_delta_nonflat_ok),
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
            "precommit_lift_margin": float(PRECOMMIT_LIFT_MARGIN),
            "min_divergent_seeds": int(MIN_DIVERGENT_SEEDS),
            "divergent_pass_fraction": float(DIVERGENT_PASS_FRACTION),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "noise_bias_range_floor": float(NOISE_BIAS_RANGE_FLOOR),
            "noise_bias_to_raw_range_frac_floor": float(NOISE_BIAS_TO_RAW_RANGE_FRAC_FLOOR),
            "dacc_max_suppression_floor": float(DACC_MAX_SUPPRESSION_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "noisy_selection_sigma_init": float(NOISY_SELECTION_SIGMA_INIT),
            "noisy_selection_weight": float(NOISY_SELECTION_WEIGHT),
            "temp_noise_floor_alpha": float(TEMP_NOISE_FLOOR_ALPHA),
            "temp_noise_floor_min_temperature": float(TEMP_NOISE_FLOOR_MIN_TEMPERATURE),
            "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
            "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
        },
        "acceptance_criteria": {
            "preconditions_met": preconditions_met,
            "n_divergent_seeds": int(n_primary_div),
            "enough_divergent_seeds": enough_divergent,
            "temp_lifts_precommit": temp_lifts_precommit,
            "noise_lifts_precommit": noise_lifts_precommit,
            "noise_bias_supra_floor": noise_bias_supra_floor,
            "dacc_live": dacc_live,
            "learning_engaged_fcg_moved": fcg_moved_ok,
            "learning_engaged_fcg_delta_nonflat": fcg_delta_nonflat_ok,
            "loopseg_carries_live_cross_loop_variance": loop_cross_variance_ok,
            "C1_noise_single_above_temp": c1_holds,
            "C1_n_seeds": int(len(c1_seeds)),
            "C1_n_divergent": int(n_primary_div),
            "DISAMBIG_loopseg_converts": loop_converts,
            "DISAMBIG_n_seeds": int(len(loop_conv_seeds)),
            "DISAMBIG_n_divergent": int(len(loop_div)),
            "mean_committed_class_entropy_a0_off": round(off_mean_dv, 6),
            "mean_committed_class_entropy_arm_temp": round(temp_mean_dv, 6),
            "mean_committed_class_entropy_arm_noise_single": round(nsingle_mean_dv, 6),
            "mean_committed_class_entropy_arm_noise_loopseg": round(nloop_mean_dv, 6),
            "mean_precommit_class_entropy_a0_off": round(_mean([r["precommit_class_entropy_nats"] for r in off_rows]), 6),
            "mean_precommit_class_entropy_arm_temp": round(_mean([r["precommit_class_entropy_nats"] for r in temp_rows]), 6),
            "mean_precommit_class_entropy_arm_noise_single": round(_mean([r["precommit_class_entropy_nats"] for r in nsingle_rows]), 6),
        },
        "interpretation_grid": {
            "PASS_weight_noise_carves_committed_diversity_above_matched_temperature_supports_mech440": (
                "preconditions met (divergent seeds + BOTH ARM_TEMP and ARM_NOISE_SINGLE raise "
                "pre-commit entropy + supra-floor noise bias + dACC live + learning engaged) AND "
                "C1 (ARM_NOISE_SINGLE committed-class entropy strict-above the matched-pre-commit-"
                "variance temperature control ARM_TEMP on a strict-majority of divergent seeds). "
                "The factorised-Gaussian weight noise CARVES through to the committed action where "
                "the temperature THRASHES at pre-commit and washes out at the argmax (the "
                "V3-EXQ-687 non-propagation FIX) -> supports MECH-440."
            ),
            "FAIL_injection_site_fix_gated_on_arc110_route_implement_substrate_loop_segregation": (
                "preconditions met BUT NOT C1 (ARM_NOISE_SINGLE ~ ARM_TEMP at the commit) WHILE "
                "ARM_NOISE_LOOPSEG converts (committed strict-above ARM_TEMP AND ARM_NOISE_SINGLE, "
                "with live cross-loop variance). The propagating weight noise needs the multi-"
                "arena substrate to convert -> route /implement-substrate ARC-110, NOT a deeper "
                "noise build. non_contributory for MECH-440 standalone."
            ),
            "FAIL_weight_noise_washes_at_argmax_single_arena_subsumes_injection_locus_route_arc110_weakens_mech440": (
                "DECISIVE. preconditions met (both arms genuinely raise pre-commit entropy) BUT "
                "NEITHER ARM_NOISE_SINGLE NOR ARM_NOISE_LOOPSEG lifts committed-class entropy "
                "strict-above ARM_TEMP. The weight noise washes at the argmax even on the SOTA "
                "stack -- thrash, not carve -- so the binding constraint is the single F-dominated "
                "arena, not the injection site (the decision could_be_wrong_if #4 realised) -> "
                "weakens MECH-440 as a standalone fix; route /implement-substrate ARC-110."
            ),
            "FAIL_substrate_not_ready_requeue": (
                "A precondition is unmet: too FEW divergent seeds, OR ARM_TEMP did not raise pre-"
                "commit entropy (not a valid matched control), OR the weight noise injected no "
                "pre-commit variance / a sub-floor bias range, OR dACC suppression was flat, OR "
                "learning was not engaged. The propagation question could NOT be measured -- NOT a "
                "falsification."
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
            f"V3-EXQ-708 MECH-440 NOISY-SELECTION-HEAD PROPAGATION FALSIFIER "
            f"(experiment_purpose=diagnostic; claim_ids=[MECH-440]; PROMOTES NOTHING). "
            f"Falsifier for MECH-440 (NoisyNet propagating selection-head weight noise) built "
            f"2026-06-27 via /implement-substrate. Decision-of-record "
            f"cpkt_tonic_exploration_noise_build_decision_2026-06-27. 4 arms on the SAME GAP-A "
            f"reef-bipartite foraging substrate + the SAME landed SOTA conversion stack (569i "
            f"top-k + MECH-448 demotion + Go/No-Go + authority + routing + finer gating + learned "
            f"settling) as a MATCHED CONSTANT; the ONLY swept factor is the exploration-injection "
            f"site: A0_OFF / ARM_TEMP (the 687 matched-pre-commit-variance temperature control, "
            f"use_noise_floor) / ARM_NOISE_SINGLE (MECH-440 use_noisy_selection_head, single "
            f"arena) / ARM_NOISE_LOOPSEG (MECH-440 + ARC-110 loop segregation). PRIMARY DV = "
            f"committed-action-class entropy; SECONDARY DV = pre-commit sampling-class entropy "
            f"(the thrash-vs-carve discriminator). PRE-REGISTERED decisive: C1 ARM_NOISE_SINGLE "
            f"committed strict-above ARM_TEMP on a strict-majority of divergent seeds => the "
            f"weight noise carves through to commit (the 687 non-propagation fix) -> supports "
            f"MECH-440; if NOT C1 but ARM_NOISE_LOOPSEG converts => injection-site fix gated on "
            f"ARC-110 -> route /implement-substrate loop segregation (non_contributory); if "
            f"NEITHER converts despite both raising pre-commit => weight noise washes at the "
            f"argmax, the single-arena collapse subsumes the injection locus (decision "
            f"could_be_wrong_if #4) -> weakens MECH-440 + route ARC-110. Non-vacuity self-route "
            f"substrate_not_ready_requeue (NEVER a false weakens): both arms must genuinely raise "
            f"pre-commit entropy, the noise bias range supra-floor vs raw, dACC live, learning "
            f"engaged, the pool divergent. COORDINATE (no duplicate): the empirical 687-successor "
            f"(GAP-C leg) tests EMPIRICAL conversion; the 705/706 lineage is the curiosity channel "
            f"(MECH-441). PROMOTES NOTHING (MECH-440 candidate/substrate_ceiling/v3_pending). "
            f"outcome={result['outcome']}; label={result['interpretation_label']}; "
            f"per_claim={result['evidence_direction_per_claim']}."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "4-arm MECH-440 noise-injection-site falsifier (A0_OFF / ARM_TEMP / ARM_NOISE_SINGLE / ARM_NOISE_LOOPSEG) on the 569i top-k + MECH-448 demotion SOTA conversion stack + per-seed-divergent gating + matched-pre-commit-variance non-vacuity + thrash-vs-carve pre-commit-vs-committed entropy discriminator",
            "swept_variables": "exploration-injection site only: none (A0_OFF) / softmax temperature (ARM_TEMP, MECH-313 noise_floor) / selection-head weight noise (ARM_NOISE_SINGLE, MECH-440) / weight noise + loop segregation (ARM_NOISE_LOOPSEG, MECH-440 + ARC-110)",
            "the_isolated_factor": (
                "all arms run the SAME SOTA conversion stack; ARM_TEMP raises the softmax "
                "temperature (pre-commit only, the 687 non-propagating control); ARM_NOISE_SINGLE "
                "injects factorised-Gaussian weight noise into _modulatory_accum BEFORE the "
                "committed within-eligible argmin (propagates); ARM_NOISE_LOOPSEG also arms the "
                "ARC-110 segregated loops to disambiguate injection-site from single-arena."
            ),
            "matched_constant_sota_stack": (
                "use_f_eligibility_demotion=True + use_f_eligibility_adaptive_floor=True + "
                "use_go_nogo_constitution=True + use_modulatory_selection_authority=True + "
                "use_modulatory_channel_routing + top_k shortlist (k=3, 569i) + use_dacc + "
                "use_finer_channel_gating + use_learned_settling_step"
            ),
            "primary_dv": "committed-action-class entropy (nats), divergent seeds only",
            "secondary_dv": "pre-commit sampling-class entropy (nats) from e3.last_precommit_probs (thrash-vs-carve)",
            "phases": "P0 e2-train (CRF matures, finer gating ON) -> P1 frozen-encoder bias-head REINFORCE -> P2 measure (e2+bias frozen; gating keeps adapting)",
            "mech440_noise": f"use_noisy_selection_head=True on the NOISE arms; sigma_init={NOISY_SELECTION_SIGMA_INIT}, weight={NOISY_SELECTION_WEIGHT}, local confidence-EMA self-anneal",
            "temp_control": f"use_noise_floor=True on ARM_TEMP; alpha={TEMP_NOISE_FLOOR_ALPHA}, min_temperature={TEMP_NOISE_FLOOR_MIN_TEMPERATURE} (effective_T ~ {1.0 + TEMP_NOISE_FLOOR_ALPHA})",
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "mech440_relationship": "this IS the MECH-440 injection-site falsifier; supports => weight noise propagates where temperature does not; weakens/route => single-arena collapse (ARC-110) subsumes the injection locus",
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-708 MECH-440 noisy-selection-head propagation falsifier"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, p2, steps = DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_P2, DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0, p1, p2, steps = (
            P0_WARMUP_EPISODES, P1_BIAS_TRAIN_EPISODES, P2_MEASUREMENT_EPISODES, STEPS_PER_EPISODE
        )

    result = run_experiment(
        seeds=seeds, p0_episodes=p0, p1_episodes=p1, p2_episodes=p2,
        steps_per_episode=steps, dry_run=bool(args.dry_run),
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
        f"temp_pre_lift={_ac['temp_lifts_precommit']} noise_pre_lift={_ac['noise_lifts_precommit']} "
        f"noise_bias_ok={_ac['noise_bias_supra_floor']} dacc_live={_ac['dacc_live']} "
        f"C1_noise_above_temp={_ac['C1_noise_single_above_temp']} "
        f"DISAMBIG_loopseg_converts={_ac['DISAMBIG_loopseg_converts']} "
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
