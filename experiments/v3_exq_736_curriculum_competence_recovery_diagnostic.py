#!/opt/local/bin/python3
"""
V3-EXQ-736 -- CURRICULUM COMPETENCE-RECOVERY DIAGNOSTIC (Track-1c of the post-724 campaign).

WHY THIS EXISTS (a DIAGNOSTIC, not an evidence falsifier; the self-routed label below is a
HYPOTHESIS the adjudication pipeline can falsify, NOT a governance verdict; claim_ids=[],
experiment_purpose=diagnostic, non_contributory). The 719a/724/728/732a stream established
that the integrated all-ON REE-v3 agent cannot forage the hard reef-bipartite env (0.065 /
0.0 / 0.455 res/ep in 719a; 0.17 in 728; below the 1.0 competence floor). V3-EXQ-724
localized nothing -- its A4 recovery arm (minimal + P1-LONG 300ep + e2-unfrozen) ALSO stayed
below the floor -- so "more flat RL of the same kind on the hard env" is RULED OUT (724
diffuse; every single-factor and the multi-factor recovery arm sub-floor). But 724 NEVER
tested a CURRICULUM: staged environment difficulty (train easy first, then transfer to hard).

HYPOTHESIS (Track-1c): the competence deficit is a training-CURRICULUM gap, not a capacity
gap. Two-stage training -- Stage-A train the all-ON stack to forage a decisively EASY env
(hazards OFF, hazard_food_attraction 0, reef_bipartite OFF, proximity_harm 0), THEN Stage-B
fine-tune / transfer the SAME agent into the hard 724 env. QUESTION: does hard-env foraging
clear the 1.0 floor post-curriculum, where flat P1-long (724 A1/A4, 300ep) did not?

BRAKE-EXEMPT. This is a COMPETENCE / training-curriculum diagnostic, NOT a conversion or
de-commit falsifier -- the conversion_ceiling_campaign_plan.md re-derive brakes do NOT apply
(and it tags NO claim, so the /failure-autopsy re-derive brake counter is zero). New EXQ
NUMBER (a different question -- curriculum-vs-flat -- than 724's OFAT localization).

SIBLING CAMPAIGN TRACKS (post-724 competence recovery): V3-EXQ-734 = Track-1a (env-difficulty
competence-recovery sweep -- does foraging recover as the env de-risks) and V3-EXQ-735 =
Track-1b (drive/reward-balance approach-weighting sweep). This is Track-1c (curriculum).
Track-1a had NOT yet produced results when this was queued, so the EASY Stage-A config here was
chosen independently (hazards / hazard_food_attraction / bipartite / proximity-harm neutralized,
reef structural channels kept ON for obs-dim parity) and is validated in-run by the Stage-A
readiness gate. If Track-1a later lands its lowest-difficulty forageable cell, a lettered
successor could adopt that exact cell as Stage-A.

------------------------------------------------------------------------------------------
DESIGN -- flat-hard control vs curriculum arms at MATCHED total P1 budget (300ep), so the
comparison isolates CURRICULUM-vs-FLAT and NOT total training budget.

  A0  flat_hard_p1full_frozen  -- CONTROL. all-ON, trained FLAT on the HARD env only:
        P0=200 (hard) + P1=300 (hard), e2 encoder FROZEN in P1. The budget-matched flat
        comparator (== 724 A1 recipe, known 0.3 res/ep sub-floor; 724 A4's flat-unfrozen
        300ep also failed at 0.18). Reproduces the incompetence at the matched budget.
  C1  curriculum_easy_hard_frozen_stageB -- all-ON CURRICULUM, e2 FROZEN in Stage-B.
        Stage-A: P0=200 (easy) + P1=150 (easy, e2 TRAINS -- learn the easy world model).
        Stage-B: P1=150 (hard, e2 FROZEN -- transfer the easy-learned world model, adapt
        only the policy heads on the hard env). Total P1 = 300 (matched to A0).
  C2  curriculum_easy_hard_unfrozen_stageB -- all-ON CURRICULUM, e2 UNFROZEN in Stage-B.
        Identical to C1 but the SD-056 e2 world-forward keeps training THROUGH Stage-B (the
        world model continues adapting to the hard env). Total P1 = 300.
  C3  curriculum_easy_med_hard_unfrozen -- all-ON CURRICULUM, 3 rungs, e2 unfrozen throughout.
        Stage-A P1=100 (easy) -> Stage-mid P1=100 (medium) -> Stage-B P1=100 (hard). The
        most gradual / strongest-form curriculum. Total P1 = 300.

Curriculum transfer semantics: the SAME agent (all encoder / planner / head WEIGHTS) carries
across stage env switches. The e2 transition-replay buffer, the REINFORCE outcome buffer, and
the REINFORCE baseline are RESET at every env change (stale easy-env transitions / returns
would poison hard-env world-model / policy adaptation); only the learned weights transfer.

DV (load-bearing): HARD-env P2 mean_resources_per_episode (mean over completed P2 eval
episodes of env.step() ticks with info.transition_type == 'resource'), measured on the SAME
statistic and the SAME 1.0 floor as 724/728. STAGE-A foraging (easy-env eval after Stage-A)
is ALSO measured on the SAME statistic -- it is the load-bearing PREMISE (confirm the agent
CAN forage when the env is easy; if it cannot, the curriculum has no foundation to transfer).

POSITIVE CONTROLS (readiness). Greedy nearest-resource ORACLE foragers (no agent) on BOTH the
hard env (724/728 convention -- floor achievable on hard) AND the easy env (floor achievable
on easy). Same statistic as the agent DV.

PRE-REGISTERED SELF-ROUTE (HYPOTHESIS, not a verdict):
  * READINESS fails -> label `substrate_not_ready_requeue`. Readiness = hard oracle clears
    floor AND easy oracle clears floor AND CURRICULUM PREMISE MET (at least one curriculum arm
    forages the EASY env >= floor on a majority of seeds after Stage-A) AND A0 flat REPRODUCES
    the incompetence (majority sub-floor on hard) AND every cell logged >= MIN_P2_EPISODES. If
    NO curriculum arm can forage even the easy env after Stage-A, the curriculum premise is
    UNMET -- draw NO conclusion about transfer; re-examine the Stage-A regime and re-queue.
    (This is the direct guard the 732a stream warns about: a learner too weak to forage the
    easy env cannot license a transfer conclusion.)
  * CURRICULUM-BUILDABLE: readiness holds AND some curriculum arm that cleared its Stage-A easy
    floor ALSO clears the HARD floor on a majority of seeds while the flat control A0 does not
    -> label `competence_curriculum_buildable` (PASS). Competence is curriculum-BUILDABLE on
    the existing substrate. HYPOTHESIS: route to /implement-substrate on a staged training
    regime for the f_dominance_conversion_ceiling competence build.
  * TRANSFER CEILING: readiness holds, the curriculum premise is met (the agent DOES forage the
    easy env), but NO curriculum arm clears the HARD floor -> label
    `transfer_capacity_ceiling` (FAIL). Competence is buildable in the easy env but does NOT
    transfer to the hard env under this staged regime -> transfer / capacity is the true
    ceiling, not the training curriculum. HYPOTHESIS: route to a different substrate
    investigation (observation encoding / capacity), NOT a staged-training build.

EVIDENCE-FOR / EVIDENCE-AGAINST (diagnostic-description requirement):
  * EVIDENCE that competence is CURRICULUM-BUILDABLE: a curriculum arm clears the hard floor
    (majority) while flat A0 does not, with its Stage-A easy premise met.
  * EVIDENCE that the true ceiling is TRANSFER / CAPACITY: the agent forages the easy env
    competently (premise met) but no curriculum arm clears the hard floor.
  * EVIDENCE AGAINST any transfer conclusion (substrate_not_ready_requeue): an oracle cannot
    clear the floor (env does not permit it) OR NO curriculum arm forages even the easy env
    after Stage-A (premise unmet) OR A0 already clears the hard floor (incompetence not
    reproduced) OR insufficient P2 episodes. No conclusion licensed.
  The self-routed label is a HYPOTHESIS for /failure-autopsy adjudication; this experiment
  PROMOTES / DEMOTES NOTHING and tags NO claim.

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

All-ON matched-stack config sourced from V3-EXQ-714 ARM_ON (via V3-EXQ-724 / 719a):
experiments/v3_exq_724_competence_localization_diagnostic.py (all-ON stack + harness + oracle),
experiments/v3_exq_719a_conversion_ceiling_dissociation_diagnostic.py,
experiments/v3_exq_714_fullstack_selection_valuation_conversion_falsifier.py (ARM_ON source),
experiments/v3_exq_732a_policy_learning_discriminator.py (sanity/easy env reference),
ree_core/environment/causal_grid_world.py (ACTIONS / env.resources / agent_x/agent_y / step() info),
ree_core/agent.py (select_action / sense / generate_trajectories / update_z_goal / update_residue),
ree_core/utils/config.py (gating-flag defaults).
See REE_assembly/evidence/planning/conversion_ceiling_campaign_plan.md (node CAMPAIGN),
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-724_*, .../failure_autopsy_V3-EXQ-732a_2026-07-10.md.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_736_curriculum_competence_recovery_diagnostic"
QUEUE_ID = "V3-EXQ-736"
CLAIM_IDS: List[str] = []                 # tags NO claim -- pure diagnostic
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# Budget (P1 total held at 300 across arms so curriculum-vs-flat is not
# confounded with total training budget -- matches 724 A1/A4 P1_LONG=300).
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45]
P0_WARMUP_EPISODES = 200          # encoder / e2 warmup (mirrors 719a / 724 P0)
P1_FULL = 300                     # flat control: all 300 P1 episodes on the hard env
P1_HALF = 150                     # 2-stage curriculum: 150 easy + 150 hard
P1_THIRD = 100                    # 3-rung curriculum: 100 easy + 100 med + 100 hard
P2_EVAL_EPISODES = 20             # HARD-env competence eval (load-bearing DV) per cell
STAGE_A_EVAL_EPISODES = 20        # EASY-env foraging eval after Stage-A (premise) per cell
STEPS_PER_EPISODE = 200
N_ORACLE_EPISODES = 20            # positive-control oracle episodes per seed per env

# Pre-registered behavioural competence floor (shared with 724/728/732a).
COMPETENCE_RESOURCE_FLOOR = 1.0
MIN_P2_EPISODES = 5               # per cell: below this the DV is not estimable

# ---------------------------------------------------------------------------
# Dry-run budget (tiny; smoke stays fast)
# ---------------------------------------------------------------------------
DRY_RUN_SEEDS = [42, 43]
DRY_RUN_P0 = 2
DRY_RUN_P1_FULL = 4
DRY_RUN_P1_HALF = 2
DRY_RUN_P1_THIRD = 2
DRY_RUN_P2 = 2
DRY_RUN_STAGE_A = 2
DRY_RUN_STEPS = 30
DRY_RUN_ORACLE_EPS = 2


def _min_seeds(n_seeds: int) -> int:
    """Strict majority: 3 of 4, 2 of 3, 2 of 2."""
    return n_seeds // 2 + 1


# ---------------------------------------------------------------------------
# Environments. HARD == identical to V3-EXQ-714 / 719a / 724. EASY / MED are
# graded-difficulty curriculum rungs. (EASY mirrors the V3-EXQ-732a sanity env:
# hazards / reef / contamination OFF, so a resource-seeking policy forages freely.)
# ---------------------------------------------------------------------------
HARD_ENV_KWARGS = dict(
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

# EASY rung. CRITICAL: the observation SPACE must be identical across curriculum rungs
# (a transferred agent has a fixed-width world_obs_encoder), and reef_enabled toggles
# world_obs_dim (reef OFF -> 250, reef ON -> 275). So EASY keeps the reef STRUCTURAL
# channels ON (obs-dim parity at 275 with hard/med) but neutralizes every DIFFICULTY
# driver: no hazards, no harm, no hazard_food_attraction, no bipartite agent/resource
# split, no proximity harm. A resource-seeking policy forages it freely.
EASY_ENV_KWARGS = dict(
    size=12,
    num_hazards=0,
    num_resources=5,
    hazard_harm=0.0,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.0,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.0,
    reef_bipartite_layout=False,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

# Intermediate rung: a couple of hazards, mild food-attraction, reef on but NOT bipartite.
MED_ENV_KWARGS = dict(
    size=12,
    num_hazards=2,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.35,
    reef_bipartite_layout=False,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

ENV_KWARGS_BY_KIND: Dict[str, Dict[str, Any]] = {
    "easy": EASY_ENV_KWARGS,
    "med": MED_ENV_KWARGS,
    "hard": HARD_ENV_KWARGS,
}


def _make_env(env_kind: str, seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS_BY_KIND[env_kind])


# ---------------------------------------------------------------------------
# Arm table. Each arm = P0 warmup env + ordered training STAGES (env, P1 length,
# e2_train flag). Curriculum arms run a Stage-A EASY foraging eval after their
# first stage; every arm runs the HARD P2 eval at the end.
# ---------------------------------------------------------------------------
def _arms(p1_full: int, p1_half: int, p1_third: int) -> List[Dict[str, Any]]:
    return [
        {
            "arm_id": "A0_flat_hard_p1full_frozen",
            "role": "control_flat_hard",
            "p0_env": "hard",
            "is_curriculum": False,
            "stages": [
                {"env": "hard", "p1": p1_full, "e2_train": False},
            ],
        },
        {
            "arm_id": "C1_curriculum_easy_hard_frozen_stageB",
            "role": "curriculum_frozen_stageB",
            "p0_env": "easy",
            "is_curriculum": True,
            "stages": [
                {"env": "easy", "p1": p1_half, "e2_train": True},
                {"env": "hard", "p1": p1_half, "e2_train": False},
            ],
        },
        {
            "arm_id": "C2_curriculum_easy_hard_unfrozen_stageB",
            "role": "curriculum_unfrozen_stageB",
            "p0_env": "easy",
            "is_curriculum": True,
            "stages": [
                {"env": "easy", "p1": p1_half, "e2_train": True},
                {"env": "hard", "p1": p1_half, "e2_train": True},
            ],
        },
        {
            "arm_id": "C3_curriculum_easy_med_hard_unfrozen",
            "role": "curriculum_3rung_unfrozen",
            "p0_env": "easy",
            "is_curriculum": True,
            "stages": [
                {"env": "easy", "p1": p1_third, "e2_train": True},
                {"env": "med", "p1": p1_third, "e2_train": True},
                {"env": "hard", "p1": p1_third, "e2_train": True},
            ],
        },
    ]


CONTROL_ARM_ID = "A0_flat_hard_p1full_frozen"
CURRICULUM_ARM_IDS = (
    "C1_curriculum_easy_hard_frozen_stageB",
    "C2_curriculum_easy_hard_unfrozen_stageB",
    "C3_curriculum_easy_med_hard_unfrozen",
)

# ---------------------------------------------------------------------------
# All-ON matched-stack constants (sourced from V3-EXQ-714 ARM_ON via 724 / 719a)
# ---------------------------------------------------------------------------
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
MODULATORY_CHANNEL_ROUTE_SOURCE = "cand_world_summary"
MODULATORY_CHANNEL_ROUTE_WEIGHT = 1.0
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_SHORTLIST_K = 3
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30
F_ELIGIBILITY_DN_SIGMA = 0.0
F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR = 1.0
GNG_PERSEVERATION_FLOOR = 0.5
GNG_SAFETY_FLOOR = 0.5
GNG_PROTECT_MIN_ELIGIBLE = 1
MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

OFC_STATE_DIM = 16
OFC_HARM_DIM = 32
OFC_BIAS_SCALE = 0.5
OFC_DEVAL_BIAS_SCALE = 2.0
LR_OFC_DEVAL = 2e-3
GNG_VIABILITY_FLOOR = 0.1

# SD-056 online e2 training (714 harness).
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

# P1 bias-head REINFORCE training (714).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9


# ---------------------------------------------------------------------------
# Agent config: all-ON (714 ARM_ON via 724). alpha_world=0.9 (z_world fidelity).
# ---------------------------------------------------------------------------
def _base_config_kwargs(env: CausalGridWorldV2) -> Dict[str, Any]:
    return dict(
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
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        candidate_summary_source="e2_world_forward",
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
    )


def _all_on_extra_kwargs() -> Dict[str, Any]:
    """The gating / valuation / modulation superstructure (V3-EXQ-714 ARM_ON)."""
    return dict(
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source=MODULATORY_CHANNEL_ROUTE_SOURCE,
        modulatory_channel_route_weight=MODULATORY_CHANNEL_ROUTE_WEIGHT,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        use_f_eligibility_demotion=True,
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        use_f_eligibility_adaptive_floor=True,
        f_eligibility_adaptive_mean_factor=F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
        use_dacc=True,
        use_go_nogo_constitution=True,
        gng_perseveration_floor=GNG_PERSEVERATION_FLOOR,
        gng_safety_floor=GNG_SAFETY_FLOOR,
        gng_protect_min_eligible=GNG_PROTECT_MIN_ELIGIBLE,
        gng_viability_floor=GNG_VIABILITY_FLOOR,
        use_ofc_analog=True,
        ofc_state_dim=OFC_STATE_DIM,
        ofc_harm_dim=OFC_HARM_DIM,
        ofc_bias_scale=OFC_BIAS_SCALE,
        use_ofc_devaluation_head=True,
        ofc_devaluation_bias_scale=OFC_DEVAL_BIAS_SCALE,
        ofc_train_devaluation_head=True,
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=MECH341_ENTROPY_BIAS_SCALE,
        e3_diversity_stratified_within_class_temperature=None,
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        use_gated_policy=True,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_context_from_e2_world_forward=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=CRF_MAINTENANCE_FLOOR,
        crf_maintenance_decay=CRF_MAINTENANCE_DECAY,
        crf_mature_context_match_threshold=CRF_MATURE_CONTEXT_MATCH_THRESHOLD,
        crf_tolerance_conflict_cap=CRF_TOLERANCE_CONFLICT_CAP,
        crf_maintenance_couple_to_theta=CRF_MAINTENANCE_COUPLE_TO_THETA,
        use_candidate_rule_field=True,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    kwargs = _base_config_kwargs(env)
    kwargs.update(_all_on_extra_kwargs())
    cfg = REEConfig.from_dims(**kwargs)
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-724)
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
# obs helpers (mirror V3-EXQ-724)
# ---------------------------------------------------------------------------
def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


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


def _build_viability_nogo(bias_low: torch.Tensor) -> Optional[torch.Tensor]:
    """485m / 714 viability No-Go from trained OFC devalued valuation."""
    bl = bias_low.detach().reshape(-1)
    if bl.numel() < 2:
        return None
    rng = float((bl.max() - bl.min()).item())
    if rng < 1e-6:
        return None
    bln = (bl - bl.min()) / (bl.max() - bl.min())
    return (1.0 - bln).detach()


# ---------------------------------------------------------------------------
# P1 two-head REINFORCE (mirror V3-EXQ-724)
# ---------------------------------------------------------------------------
def _lpfc_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
) -> torch.Tensor:
    if getattr(agent, "lateral_pfc", None) is None or len(outcome_buf) < 2:
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
        log_p = torch.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


def _ofc_deval_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
) -> torch.Tensor:
    ofc = getattr(agent, "ofc", None)
    if ofc is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    n = len(outcome_buf)
    idxs = np.random.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False)
    terms: List[torch.Tensor] = []
    for i in idxs:
        cand_features, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        bias = ofc.compute_devaluation_bias(cand_features.to(device))
        if not bias.requires_grad or bias.shape[0] < 2:
            continue
        log_p = torch.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# Positive-control ORACLE forager (no agent; greedy to nearest resource).
# Measures the SAME statistic as the agent DV: mean resources/episode.
# ---------------------------------------------------------------------------
def _oracle_action(env: CausalGridWorldV2) -> int:
    resources = getattr(env, "resources", None)
    if not resources:
        return 4
    ax, ay = int(env.agent_x), int(env.agent_y)
    best = min(resources, key=lambda r: abs(int(r[0]) - ax) + abs(int(r[1]) - ay))
    rx, ry = int(best[0]), int(best[1])
    dx, dy = rx - ax, ry - ay
    if dx == 0 and dy == 0:
        return 4
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0
    return 3 if dy > 0 else 2


def _run_oracle(
    env_kind: str, seed: int, n_episodes: int, steps_per_episode: int
) -> Dict[str, Any]:
    env = _make_env(env_kind, seed)
    ep_resources: List[int] = []
    for _ep in range(n_episodes):
        env.reset()
        collected = 0
        for _step in range(steps_per_episode):
            a = _oracle_action(env)
            _, _harm, done, info, _obs = env.step(a)
            if str(info.get("transition_type", "none")) == "resource":
                collected += 1
            if done:
                break
        ep_resources.append(collected)
    mean_res = float(sum(ep_resources) / len(ep_resources)) if ep_resources else 0.0
    return {
        "env_kind": env_kind,
        "seed": int(seed),
        "n_episodes": int(len(ep_resources)),
        "mean_resources_per_episode": round(mean_res, 6),
        "max_resources_in_episode": int(max(ep_resources)) if ep_resources else 0,
    }


# ---------------------------------------------------------------------------
# Info-theory helper (marginal committed-class entropy for context)
# ---------------------------------------------------------------------------
def _marginal_entropy_nats(counts: Dict[Any, int]) -> float:
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
# Shared per-step primitive (sense -> plan -> select -> step), used by P0 warmup,
# P1 REINFORCE, and frozen eval. Handles the e2 transition capture, residue and
# z_goal deployed-policy updates. Returns the committed action-class + step signals.
# ---------------------------------------------------------------------------
class _StepState:
    """Mutable per-episode carry for the e2 world-forward transition capture."""

    def __init__(self) -> None:
        self.pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.z_self_prev: Optional[torch.Tensor] = None
        self.action_prev: Optional[torch.Tensor] = None


def _agent_step(
    agent: REEAgent,
    env: CausalGridWorldV2,
    obs_dict: Dict[str, Any],
    st: _StepState,
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    inject_ofc_viability: bool,
    want_p1_snapshot: bool,
) -> Dict[str, Any]:
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

    if st.pending_capture is not None:
        z0_prev, a_prev = st.pending_capture
        z1_obs = latent.z_world.detach().reshape(-1).clone()
        if (
            torch.isfinite(z0_prev).all()
            and torch.isfinite(a_prev).all()
            and torch.isfinite(z1_obs).all()
        ):
            transition_buffer.append((z0_prev, a_prev, z1_obs))
        st.pending_capture = None

    if st.z_self_prev is not None and st.action_prev is not None:
        agent.record_transition(st.z_self_prev, st.action_prev, latent.z_self.detach())

    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (
        agent._e1_tick(latent) if ticks.get("e1_tick", False)
        else torch.zeros(1, wdim, device=agent.device)
    )
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)

    p1_snap_summaries: Optional[torch.Tensor] = None
    if want_p1_snapshot and candidates and len(candidates) >= 2:
        cs = _consumed_summaries(agent, candidates)
        if cs is not None and torch.isfinite(cs).all():
            p1_snap_summaries = cs.clone()

    if inject_ofc_viability and getattr(agent, "ofc", None) is not None:
        if candidates and len(candidates) >= 2:
            ofc_summ = _consumed_summaries(agent, candidates)
            viability_sig: Optional[torch.Tensor] = None
            if ofc_summ is not None and torch.isfinite(ofc_summ).all():
                with torch.no_grad():
                    deval_bias = agent.ofc.compute_devaluation_bias(ofc_summ).detach()
                viability_sig = _build_viability_nogo(deval_bias)
            if viability_sig is not None:
                agent.set_injected_go_nogo_signals(
                    {"viability": viability_sig.to(agent.device)}
                )
            else:
                agent.set_injected_go_nogo_signals(None)
        else:
            agent.set_injected_go_nogo_signals(None)

    action = agent.select_action(candidates, ticks)
    if action is None:
        idx = int(np.random.randint(0, env.action_dim))
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, idx] = 1.0
        agent._last_action = action
    if not torch.isfinite(action).all():
        return {"non_finite": True}

    committed_class = int(action[0].argmax().item())

    sel_idx = 0
    if p1_snap_summaries is not None:
        for ci, c in enumerate(candidates):
            if (
                getattr(c, "actions", None) is not None
                and c.actions.shape[1] >= 1
                and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                == committed_class
            ):
                sel_idx = min(ci, p1_snap_summaries.shape[0] - 1)
                break

    if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
        st.pending_capture = (
            latent.z_world.detach().reshape(-1).clone(),
            action.detach().reshape(-1).clone(),
        )

    _, _harm_signal, done, info, next_obs = env.step(action)
    harm_signal = float(_harm_signal)

    with torch.no_grad():
        agent.update_residue(
            harm_signal=harm_signal, world_delta=None,
            hypothesis_tag=False, owned=True,
        )
    if agent.goal_state is not None:
        benefit_exposure = float(info.get("benefit_exposure", 0.0))
        energy = float(body[0, 3].item())
        drive_level = max(0.0, 1.0 - energy)
        agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

    st.z_self_prev = latent.z_self.detach()
    st.action_prev = action.detach()

    return {
        "non_finite": False,
        "committed_class": committed_class,
        "harm_signal": harm_signal,
        "done": bool(done),
        "info": info if isinstance(info, dict) else {},
        "next_obs": next_obs,
        "p1_snap_summaries": p1_snap_summaries,
        "sel_idx": sel_idx,
    }


# ---------------------------------------------------------------------------
# P0 warmup: e2 world-forward contrastive only (no REINFORCE, no eval).
# ---------------------------------------------------------------------------
def _run_p0_warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    e2_opt: torch.optim.Optimizer,
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    sample_rng: random.Random,
    n_episodes: int,
    steps_per_episode: int,
    ep_counter: List[int],
    total_eps: int,
    arm_id: str,
    seed: int,
) -> int:
    n_ticks = 0
    for _ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        st = _StepState()
        tick_in_ep = 0
        for _step in range(steps_per_episode):
            out = _agent_step(
                agent, env, obs_dict, st, transition_buffer,
                inject_ofc_viability=False, want_p1_snapshot=False,
            )
            if out.get("non_finite"):
                break
            n_ticks += 1
            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng)
            obs_dict = out["next_obs"]
            tick_in_ep += 1
            if out["done"]:
                break
        ep_counter[0] += 1
        cur = ep_counter[0]
        if cur % 25 == 0 or cur == total_eps:
            print(
                f"  [train] curriculum arm={arm_id} seed={seed} phase=P0 "
                f"ep {cur}/{total_eps}",
                flush=True,
            )
    return n_ticks


# ---------------------------------------------------------------------------
# P1 REINFORCE stage on one env (two-head lateral-PFC bias + OFC devaluation).
# ---------------------------------------------------------------------------
def _run_p1_stage(
    agent: REEAgent,
    env: CausalGridWorldV2,
    stage_env_kind: str,
    e2_opt: torch.optim.Optimizer,
    bias_opt: Optional[torch.optim.Optimizer],
    ofc_deval_opt: Optional[torch.optim.Optimizer],
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    sample_rng: random.Random,
    n_episodes: int,
    steps_per_episode: int,
    e2_train: bool,
    ep_counter: List[int],
    total_eps: int,
    arm_id: str,
    seed: int,
) -> int:
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None
    has_ofc = getattr(agent, "ofc", None) is not None
    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []
    n_ticks = 0

    for _ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        st = _StepState()
        tick_in_ep = 0
        ep_reward = 0.0
        ep_buf: List[Tuple[torch.Tensor, int]] = []

        for _step in range(steps_per_episode):
            out = _agent_step(
                agent, env, obs_dict, st, transition_buffer,
                inject_ofc_viability=False,
                want_p1_snapshot=bool(has_lpfc),
            )
            if out.get("non_finite"):
                break
            n_ticks += 1
            snap = out["p1_snap_summaries"]
            if snap is not None:
                ep_buf.append((snap, out["sel_idx"]))
            if e2_train and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng)
            ep_reward += out["harm_signal"]
            obs_dict = out["next_obs"]
            tick_in_ep += 1
            if out["done"]:
                break

        # End-of-episode two-head REINFORCE.
        if has_lpfc or has_ofc:
            reinforce_baseline = (
                EMA_DECAY * reinforce_baseline + (1.0 - EMA_DECAY) * ep_reward
            )
            for cand_features, sel in ep_buf:
                outcome_buf.append((cand_features, sel, ep_reward))
            if len(outcome_buf) > OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
            if has_lpfc and bias_opt is not None:
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
            if has_ofc and ofc_deval_opt is not None:
                ofc_loss = _ofc_deval_reinforce_loss(
                    agent, outcome_buf, reinforce_baseline, agent.device
                )
                if ofc_loss.requires_grad:
                    ofc_deval_opt.zero_grad()
                    ofc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.ofc.devaluation_bias_head_parameters(), 1.0
                    )
                    ofc_deval_opt.step()

        ep_counter[0] += 1
        cur = ep_counter[0]
        if cur % 25 == 0 or cur == total_eps:
            print(
                f"  [train] curriculum arm={arm_id} seed={seed} "
                f"phase=P1_{stage_env_kind} ep {cur}/{total_eps}",
                flush=True,
            )
    return n_ticks


# ---------------------------------------------------------------------------
# Frozen deployed-policy foraging eval on one env (OFC viability injected, as in
# 724's P2). Measures the load-bearing DV: mean resources/episode.
# ---------------------------------------------------------------------------
def _run_eval(
    agent: REEAgent,
    env: CausalGridWorldV2,
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_episodes: int,
    steps_per_episode: int,
    ep_counter: List[int],
    total_eps: int,
    arm_id: str,
    seed: int,
    phase_label: str,
) -> Dict[str, Any]:
    has_ofc = getattr(agent, "ofc", None) is not None
    ep_resources: List[int] = []
    ep_hazard_hits: List[int] = []
    ep_contaminations: List[int] = []
    ep_rewards: List[float] = []
    committed_class_counts: Dict[int, int] = {}
    n_eps_completed = 0
    error_note: Optional[str] = None

    for _ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        st = _StepState()
        res = 0
        haz = 0
        contam = 0
        rew = 0.0
        for _step in range(steps_per_episode):
            out = _agent_step(
                agent, env, obs_dict, st, transition_buffer,
                inject_ofc_viability=bool(has_ofc),
                want_p1_snapshot=False,
            )
            if out.get("non_finite"):
                error_note = f"non-finite action arm={arm_id} seed={seed} phase={phase_label}"
                break
            cc = out["committed_class"]
            committed_class_counts[cc] = committed_class_counts.get(cc, 0) + 1
            info = out["info"]
            rew += out["harm_signal"]
            ttype = str(info.get("transition_type", "none"))
            if ttype == "resource":
                res += 1
            elif ttype == "env_caused_hazard":
                haz += 1
            if ttype == "agent_caused_hazard" or float(
                info.get("contamination_delta", 0.0)
            ) > 0.0:
                contam += 1
            obs_dict = out["next_obs"]
            if out["done"]:
                break
        if error_note is not None:
            break
        ep_resources.append(res)
        ep_hazard_hits.append(haz)
        ep_contaminations.append(contam)
        ep_rewards.append(rew)
        n_eps_completed += 1
        ep_counter[0] += 1
        cur = ep_counter[0]
        if cur % 25 == 0 or cur == total_eps or phase_label.startswith("P2"):
            print(
                f"  [train] curriculum arm={arm_id} seed={seed} "
                f"phase={phase_label} ep {cur}/{total_eps}",
                flush=True,
            )

    mean_res = float(sum(ep_resources) / len(ep_resources)) if ep_resources else 0.0
    return {
        "n_eps_completed": int(n_eps_completed),
        "error_note": error_note,
        "mean_resources_per_episode": round(mean_res, 6),
        "competence_supra_floor": bool(mean_res >= COMPETENCE_RESOURCE_FLOOR),
        "mean_hazard_hits_per_episode": round(
            float(sum(ep_hazard_hits) / len(ep_hazard_hits)) if ep_hazard_hits else 0.0, 6
        ),
        "mean_contaminations_per_episode": round(
            float(sum(ep_contaminations) / len(ep_contaminations)) if ep_contaminations else 0.0, 6
        ),
        "mean_episode_reward": round(
            float(sum(ep_rewards) / len(ep_rewards)) if ep_rewards else 0.0, 6
        ),
        "marginal_committed_class_entropy_nats": round(
            _marginal_entropy_nats(committed_class_counts), 6
        ),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        "per_episode_resources": [int(x) for x in ep_resources],
    }


# ---------------------------------------------------------------------------
# Per-cell (arm x seed) staged-curriculum run.
# ---------------------------------------------------------------------------
def _run_cell(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    stage_a_eval_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    is_curriculum = bool(arm["is_curriculum"])
    stages: List[Dict[str, Any]] = arm["stages"]

    # One env instance per distinct env_kind this arm touches (seeded by cell seed).
    env_kinds = {arm["p0_env"]} | {st["env"] for st in stages} | {"hard"}
    envs: Dict[str, CausalGridWorldV2] = {k: _make_env(k, seed) for k in env_kinds}

    agent = _make_agent(envs[arm["p0_env"]])
    has_ofc = getattr(agent, "ofc", None) is not None
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = (
        torch.optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS)
        if has_lpfc else None
    )
    ofc_deval_opt = (
        torch.optim.Adam(list(agent.ofc.devaluation_bias_head_parameters()), lr=LR_OFC_DEVAL)
        if has_ofc else None
    )
    sample_rng = random.Random(seed)

    # Total episodes across this cell (progress denominator).
    total_eps = (
        p0_episodes
        + sum(int(st["p1"]) for st in stages)
        + (stage_a_eval_episodes if is_curriculum else 0)
        + p2_episodes
    )
    ep_counter = [0]

    n_p0_ticks = 0
    n_p1_ticks = 0
    error_note: Optional[str] = None
    stage_a_eval: Optional[Dict[str, Any]] = None

    # Transition buffer + prev-env-kind tracker (reset buffers on env change).
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)

    # ---- P0 warmup (on the arm's p0_env) ----
    p0_env_kind = arm["p0_env"]
    n_p0_ticks += _run_p0_warmup(
        agent, envs[p0_env_kind], e2_opt, transition_buffer, sample_rng,
        p0_episodes, steps_per_episode, ep_counter, total_eps, arm["arm_id"], seed,
    )
    prev_env_kind = p0_env_kind

    # ---- Training stages ----
    for si, st in enumerate(stages):
        stage_env_kind = st["env"]
        if stage_env_kind != prev_env_kind:
            # Env change: stale transitions do not transfer -- reset the replay buffer.
            transition_buffer.clear()
        n_p1_ticks += _run_p1_stage(
            agent, envs[stage_env_kind], stage_env_kind, e2_opt, bias_opt, ofc_deval_opt,
            transition_buffer, sample_rng, int(st["p1"]), steps_per_episode,
            bool(st["e2_train"]), ep_counter, total_eps, arm["arm_id"], seed,
        )
        prev_env_kind = stage_env_kind

        # Stage-A EASY foraging eval (curriculum premise) after the first stage.
        if is_curriculum and si == 0:
            stage_a_eval = _run_eval(
                agent, envs[stage_env_kind], transition_buffer,
                stage_a_eval_episodes, steps_per_episode, ep_counter, total_eps,
                arm["arm_id"], seed, phase_label=f"EVAL_A_{stage_env_kind}",
            )
            if stage_a_eval.get("error_note") and error_note is None:
                error_note = stage_a_eval["error_note"]

    # ---- Final HARD P2 competence eval (load-bearing DV) ----
    if prev_env_kind != "hard":
        transition_buffer.clear()
    hard_eval = _run_eval(
        agent, envs["hard"], transition_buffer,
        p2_episodes, steps_per_episode, ep_counter, total_eps,
        arm["arm_id"], seed, phase_label="P2_hard",
    )
    if hard_eval.get("error_note") and error_note is None:
        error_note = hard_eval["error_note"]

    row = {
        "arm_id": arm["arm_id"],
        "arm_role": arm["role"],
        "is_curriculum": is_curriculum,
        "seed": int(seed),
        "p0_env": p0_env_kind,
        "stages": [
            {"env": st["env"], "p1": int(st["p1"]), "e2_train": bool(st["e2_train"])}
            for st in stages
        ],
        "p0_episodes": int(p0_episodes),
        "p1_episodes_total": int(sum(int(st["p1"]) for st in stages)),
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "error_note": error_note,
        # ----- LOAD-BEARING DV: HARD-env foraging -----
        "n_p2_eps_completed": int(hard_eval["n_eps_completed"]),
        "mean_resources_per_episode": hard_eval["mean_resources_per_episode"],
        "competence_supra_floor": hard_eval["competence_supra_floor"],
        "mean_hazard_hits_per_episode": hard_eval["mean_hazard_hits_per_episode"],
        "mean_contaminations_per_episode": hard_eval["mean_contaminations_per_episode"],
        "mean_episode_reward": hard_eval["mean_episode_reward"],
        "marginal_committed_class_entropy_nats": hard_eval[
            "marginal_committed_class_entropy_nats"
        ],
        "n_unique_committed_classes": hard_eval["n_unique_committed_classes"],
        "committed_class_counts": hard_eval["committed_class_counts"],
        "per_p2_episode_resources": hard_eval["per_episode_resources"],
        # ----- PREMISE: Stage-A EASY foraging (curriculum arms only) -----
        "stage_a_n_eps_completed": (
            int(stage_a_eval["n_eps_completed"]) if stage_a_eval else None
        ),
        "stage_a_mean_resources_per_episode": (
            stage_a_eval["mean_resources_per_episode"] if stage_a_eval else None
        ),
        "stage_a_supra_floor": (
            bool(stage_a_eval["competence_supra_floor"]) if stage_a_eval else None
        ),
        "stage_a_per_episode_resources": (
            stage_a_eval["per_episode_resources"] if stage_a_eval else None
        ),
    }
    return row


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _arm_majority(flags: List[bool], min_seeds: int) -> bool:
    return bool(sum(1 for f in flags if f) >= min_seeds)


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_full: int,
    p1_half: int,
    p1_third: int,
    stage_a_eval_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    oracle_episodes: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arms = _arms(p1_full, p1_half, p1_third)
    min_seeds = _min_seeds(len(seeds))

    # Obs-space parity guard: a transferred agent has a fixed-width encoder, so every
    # curriculum rung MUST share body_obs_dim / world_obs_dim / action_dim. Fail loudly
    # here rather than with a cryptic matmul error deep in a multi-hour cloud run.
    _probe = {k: _make_env(k, seeds[0]) for k in ("easy", "med", "hard")}
    _sig = {
        k: (int(e.body_obs_dim), int(e.world_obs_dim), int(e.action_dim))
        for k, e in _probe.items()
    }
    if len({v for v in _sig.values()}) != 1:
        raise ValueError(
            "curriculum rungs have mismatched observation dims (transfer impossible): "
            + ", ".join(f"{k}={v}" for k, v in _sig.items())
        )

    print(
        f"Curriculum competence-recovery diagnostic ({len(arms)} arms x {len(seeds)} seeds; "
        f"P0={p0_episodes}, P1_full={p1_full}, P1_half={p1_half}, P1_third={p1_third}, "
        f"stageA_eval={stage_a_eval_episodes}, P2_eval={p2_episodes}, "
        f"steps={steps_per_episode}, oracle_eps={oracle_episodes}, "
        f"min_seeds={min_seeds}, dry_run={dry_run})",
        flush=True,
    )

    # ----- Positive-control oracles: hard + easy (per seed) -----
    hard_oracle_rows: List[Dict[str, Any]] = []
    easy_oracle_rows: List[Dict[str, Any]] = []
    for s in seeds:
        ho = _run_oracle("hard", s, oracle_episodes, steps_per_episode)
        eo = _run_oracle("easy", s, oracle_episodes, steps_per_episode)
        hard_oracle_rows.append(ho)
        easy_oracle_rows.append(eo)
        print(
            f"  [oracle] seed={s} hard/ep={ho['mean_resources_per_episode']} "
            f"easy/ep={eo['mean_resources_per_episode']}",
            flush=True,
        )
    hard_oracle_min = min(
        [o["mean_resources_per_episode"] for o in hard_oracle_rows], default=0.0
    )
    easy_oracle_min = min(
        [o["mean_resources_per_episode"] for o in easy_oracle_rows], default=0.0
    )
    hard_oracle_clears = bool(hard_oracle_min >= COMPETENCE_RESOURCE_FLOOR)
    easy_oracle_clears = bool(easy_oracle_min >= COMPETENCE_RESOURCE_FLOOR)

    # ----- Arm x seed cells -----
    cells: List[Dict[str, Any]] = []
    for arm in arms:
        for s in seeds:
            print(f"Seed {s} Condition {arm['arm_id']}", flush=True)
            slice_cfg = {
                "arm_id": arm["arm_id"],
                "p0_env": arm["p0_env"],
                "stages": [
                    {"env": st["env"], "p1": int(st["p1"]), "e2_train": bool(st["e2_train"])}
                    for st in arm["stages"]
                ],
                "p0_episodes": int(p0_episodes),
                "stage_a_eval_episodes": int(stage_a_eval_episodes),
                "p2_episodes": int(p2_episodes),
                "steps_per_episode": int(steps_per_episode),
                "env_kwargs_by_kind": {
                    k: dict(v) for k, v in ENV_KWARGS_BY_KIND.items()
                },
            }
            with arm_cell(
                s,
                config_slice=slice_cfg,
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = _run_cell(
                    arm, s, p0_episodes, stage_a_eval_episodes, p2_episodes,
                    steps_per_episode,
                )
                cell.stamp(row)
            cells.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(
                f"verdict: {verdict} (arm={arm['arm_id']} seed={s} "
                f"hard/ep={row['mean_resources_per_episode']} "
                f"supra_floor={row['competence_supra_floor']} "
                f"stageA/ep={row['stage_a_mean_resources_per_episode']})",
                flush=True,
            )

    # ----- Per-arm aggregation -----
    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        rows = [c for c in cells if c["arm_id"] == arm["arm_id"]]
        ok_rows = [r for r in rows if r["error_note"] is None]
        hard_flags = [bool(r["competence_supra_floor"]) for r in ok_rows]
        stage_a_flags = [
            bool(r["stage_a_supra_floor"]) for r in ok_rows
            if r["stage_a_supra_floor"] is not None
        ]
        per_arm[arm["arm_id"]] = {
            "arm_id": arm["arm_id"],
            "role": arm["role"],
            "is_curriculum": bool(arm["is_curriculum"]),
            "n_seeds_ok": int(len(ok_rows)),
            "n_seeds_min_p2": int(
                sum(1 for r in ok_rows if r["n_p2_eps_completed"] >= MIN_P2_EPISODES)
            ),
            # HARD DV
            "mean_resources_per_episode_mean": round(
                _mean([r["mean_resources_per_episode"] for r in ok_rows]), 6
            ),
            "n_seeds_supra_floor": int(sum(1 for f in hard_flags if f)),
            "majority_supra_floor": _arm_majority(hard_flags, min_seeds),
            "mean_hazard_hits_per_episode_mean": round(
                _mean([r["mean_hazard_hits_per_episode"] for r in ok_rows]), 6
            ),
            "mean_episode_reward_mean": round(
                _mean([r["mean_episode_reward"] for r in ok_rows]), 6
            ),
            # STAGE-A premise (curriculum arms only)
            "stage_a_mean_resources_per_episode_mean": (
                round(_mean([
                    r["stage_a_mean_resources_per_episode"] for r in ok_rows
                    if r["stage_a_mean_resources_per_episode"] is not None
                ]), 6) if stage_a_flags else None
            ),
            "stage_a_n_seeds_supra_floor": (
                int(sum(1 for f in stage_a_flags if f)) if stage_a_flags else None
            ),
            "stage_a_majority_supra_floor": (
                _arm_majority(stage_a_flags, min_seeds) if stage_a_flags else None
            ),
        }

    # ----- Readiness -----
    control_stats = per_arm[CONTROL_ARM_ID]
    a0_reproduces_incompetence = bool(not control_stats["majority_supra_floor"])

    # Curriculum premise: at least one curriculum arm forages the EASY env >= floor
    # on a majority of seeds after Stage-A.
    premise_arms = [
        aid for aid in CURRICULUM_ARM_IDS
        if per_arm[aid]["stage_a_majority_supra_floor"] is True
    ]
    curriculum_premise_met = bool(premise_arms)

    all_cells_ok = [c for c in cells if c["error_note"] is None]
    sufficient_p2 = bool(
        all_cells_ok
        and all(c["n_p2_eps_completed"] >= MIN_P2_EPISODES for c in all_cells_ok)
    )
    readiness_met = bool(
        hard_oracle_clears and easy_oracle_clears and curriculum_premise_met
        and a0_reproduces_incompetence and sufficient_p2
    )

    # ----- Discrimination: does a premise-met curriculum arm clear the HARD floor? -----
    recovering_arms = [
        aid for aid in CURRICULUM_ARM_IDS
        if per_arm[aid]["stage_a_majority_supra_floor"] is True
        and per_arm[aid]["majority_supra_floor"] is True
    ]
    curriculum_recovers = bool(recovering_arms and not control_stats["majority_supra_floor"])

    # ----- Self-route (HYPOTHESIS, not a verdict) -----
    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif curriculum_recovers:
        outcome = "PASS"
        label = "competence_curriculum_buildable"
    else:
        outcome = "FAIL"
        label = "transfer_capacity_ceiling"
    direction = "non_contributory"

    interpretation = {
        "label": label,
        "curriculum_recovering_arms": recovering_arms,
        "curriculum_premise_arms": premise_arms,
        "control_reproduces_incompetence": a0_reproduces_incompetence,
        "preconditions": [
            {
                "name": "hard_oracle_clears_floor",
                "kind": "readiness",
                "description": (
                    "A greedy nearest-resource ORACLE clears COMPETENCE_RESOURCE_FLOOR "
                    "resources/episode on the HARD env (724/728 convention). Same statistic as "
                    "the load-bearing DV. Below => floor not achievable on hard => "
                    "substrate_not_ready_requeue, NEVER a transfer verdict."
                ),
                "control": "greedy nearest-resource oracle, HARD env, same seed, no agent",
                "measured": float(round(hard_oracle_min, 6)),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(hard_oracle_clears),
            },
            {
                "name": "easy_oracle_clears_floor",
                "kind": "readiness",
                "description": (
                    "A greedy nearest-resource ORACLE clears the floor on the EASY (Stage-A) "
                    "env, proving the Stage-A floor is achievable. Same statistic as the DV. "
                    "Below => easy env too sparse => substrate_not_ready_requeue."
                ),
                "control": "greedy nearest-resource oracle, EASY env, same seed, no agent",
                "measured": float(round(easy_oracle_min, 6)),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(easy_oracle_clears),
            },
            {
                "name": "curriculum_premise_agent_forages_easy",
                "kind": "readiness",
                "description": (
                    "LOAD-BEARING PREMISE. At least one CURRICULUM arm must forage the EASY env "
                    ">= floor on a majority (min_seeds) of seeds AFTER Stage-A -- i.e. the all-ON "
                    "stack CAN forage when the env is easy. Measured as the per-seed count of "
                    "Stage-A easy foraging >= floor (SAME per-seed-supra-floor statistic the "
                    "load-bearing HARD criterion routes on), so met matches the indexer recompute. "
                    "If NO curriculum arm forages even the easy env, the curriculum has no "
                    "foundation to transfer => substrate_not_ready_requeue, NEVER a transfer-"
                    "ceiling verdict (this is the guard the 732a learner-adequacy stream warns "
                    "about)."
                ),
                "control": (
                    "max over curriculum arms of the count of seeds with Stage-A easy "
                    "mean_resources/ep >= floor; met when that count >= min_seeds"
                ),
                "measured": float(max(
                    [
                        (per_arm[aid]["stage_a_n_seeds_supra_floor"] or 0)
                        for aid in CURRICULUM_ARM_IDS
                    ] or [0]
                )),
                "threshold": float(min_seeds),
                "met": bool(curriculum_premise_met),
            },
            {
                "name": "control_reproduces_incompetence",
                "kind": "readiness",
                "description": (
                    "The flat-hard control A0 (matched 300ep budget) must forage BELOW the floor "
                    "on a majority of seeds -- i.e. the flat-RL incompetence must reproduce -- for "
                    "curriculum-vs-flat to be meaningful. If A0 already clears the hard floor the "
                    "premise is not reproduced => substrate_not_ready_requeue."
                ),
                "control": "A0 flat-hard mean_resources/ep vs floor (majority of seeds)",
                "measured": float(control_stats["mean_resources_per_episode_mean"]),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "direction": "upper",
                "met": bool(a0_reproduces_incompetence),
            },
            {
                "name": "sufficient_p2_episodes_all_cells",
                "kind": "readiness",
                "description": (
                    "Every completed cell must log >= MIN_P2_EPISODES HARD P2 eval episodes so "
                    "mean_resources_per_episode is estimable. Below => substrate_not_ready_requeue."
                ),
                "control": "min completed HARD P2 episodes across all cells",
                "measured": float(
                    min([c["n_p2_eps_completed"] for c in all_cells_ok], default=0)
                ),
                "threshold": float(MIN_P2_EPISODES),
                "met": bool(sufficient_p2),
            },
        ],
        "criteria": [
            {
                "name": "curriculum_recovers_hard_competence",
                "load_bearing": True,
                "passed": bool(curriculum_recovers),
            },
        ],
        "criteria_non_degenerate": {
            "hard_oracle_clears_floor": bool(hard_oracle_clears),
            "easy_oracle_clears_floor": bool(easy_oracle_clears),
            "curriculum_premise_met": bool(curriculum_premise_met),
            "control_reproduces_incompetence": bool(a0_reproduces_incompetence),
            "sufficient_p2_episodes": bool(sufficient_p2),
            "curriculum_recovers_hard": bool(curriculum_recovers),
        },
    }

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": seeds,
        "min_seeds": int(min_seeds),
        "p0_episodes": int(p0_episodes),
        "p1_full": int(p1_full),
        "p1_half": int(p1_half),
        "p1_third": int(p1_third),
        "stage_a_eval_episodes": int(stage_a_eval_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "oracle_episodes": int(oracle_episodes),
        "decision_rule_thresholds": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "min_seeds": int(min_seeds),
            "min_p2_episodes": int(MIN_P2_EPISODES),
            "control_arm_id": CONTROL_ARM_ID,
            "curriculum_arm_ids": list(CURRICULUM_ARM_IDS),
        },
        "readiness_gates": {
            "hard_oracle_clears_floor": hard_oracle_clears,
            "hard_oracle_min_resources_per_episode": round(hard_oracle_min, 6),
            "easy_oracle_clears_floor": easy_oracle_clears,
            "easy_oracle_min_resources_per_episode": round(easy_oracle_min, 6),
            "curriculum_premise_met": curriculum_premise_met,
            "curriculum_premise_arms": premise_arms,
            "control_reproduces_incompetence": a0_reproduces_incompetence,
            "control_mean_resources_per_episode": control_stats[
                "mean_resources_per_episode_mean"
            ],
            "sufficient_p2_episodes": sufficient_p2,
            "readiness_met": readiness_met,
        },
        "curriculum_gates": {
            "curriculum_recovering_arms": recovering_arms,
            "curriculum_recovers_hard_competence": curriculum_recovers,
        },
        "hard_oracle_results": hard_oracle_rows,
        "easy_oracle_results": easy_oracle_rows,
        "per_arm": per_arm,
        "arm_results": cells,
        "interpretation_grid": {
            "competence_curriculum_buildable": (
                "readiness holds AND a curriculum arm that cleared its Stage-A EASY floor ALSO "
                "clears the HARD floor on a majority of seeds while the flat control A0 does not. "
                "Competence is CURRICULUM-BUILDABLE on the existing substrate where flat P1-long "
                "(724 A1/A4) was not. HYPOTHESIS (not a verdict): route to /implement-substrate on "
                "a staged training regime for the f_dominance_conversion_ceiling competence build."
            ),
            "transfer_capacity_ceiling": (
                "readiness holds, the curriculum premise is met (the agent DOES forage the easy "
                "env after Stage-A), but NO curriculum arm clears the HARD floor. Competence is "
                "buildable in easy conditions but does NOT transfer to the hard env under this "
                "staged regime -> transfer / capacity is the true ceiling, not the training "
                "curriculum. HYPOTHESIS: route to a different substrate investigation (observation "
                "encoding / capacity), NOT a staged-training build."
            ),
            "substrate_not_ready_requeue": (
                "an oracle cannot clear the floor (env does not permit it), OR NO curriculum arm "
                "forages even the EASY env after Stage-A (premise unmet -- the 732a learner-weak "
                "guard), OR A0 already clears the hard floor (incompetence not reproduced), OR a "
                "cell logged fewer than MIN_P2_EPISODES eval episodes. NOT a verdict -- re-examine "
                "the Stage-A regime / env / floor / budget and re-queue. Draw NO conclusion about "
                "transfer."
            ),
        },
    }


def _build_manifest(
    result: Dict[str, Any],
    timestamp_utc: str,
    dry_run: bool,
) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    rg = result["readiness_gates"]
    cg = result["curriculum_gates"]
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
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "sleep_driver_pattern": "none",
        "evidence_direction_note": (
            f"V3-EXQ-736 CURRICULUM COMPETENCE-RECOVERY DIAGNOSTIC (Track-1c; "
            f"experiment_purpose=diagnostic, claim_ids=[], non_contributory -- EXCLUDED from "
            f"governance scoring; PROMOTES / DEMOTES NOTHING). Tests whether a two-stage "
            f"CURRICULUM (Stage-A forage an EASY env -> Stage-B fine-tune into the hard 724 env) "
            f"clears the {COMPETENCE_RESOURCE_FLOOR} foraging floor where FLAT P1-long (724 A1/A4, "
            f"300ep) did not -- the curriculum path 724 never tested. Arms at MATCHED 300ep total "
            f"P1: A0 flat-hard control (== 724 A1 recipe), C1 curriculum e2-frozen-in-StageB, C2 "
            f"curriculum e2-unfrozen-in-StageB, C3 3-rung easy->med->hard. all-ON = 714 ARM_ON via "
            f"724. Load-bearing DV: HARD-env P2 mean_resources_per_episode. Stage-A EASY foraging "
            f"is the load-bearing PREMISE (same statistic/floor). Positive controls: greedy oracle "
            f"on hard (min/ep={rg['hard_oracle_min_resources_per_episode']}, "
            f"clears={rg['hard_oracle_clears_floor']}) AND easy "
            f"(min/ep={rg['easy_oracle_min_resources_per_episode']}, "
            f"clears={rg['easy_oracle_clears_floor']}). Self-route (HYPOTHESIS, not a verdict): "
            f"readiness_met={rg['readiness_met']} (both oracles clear AND >=1 curriculum arm "
            f"forages easy >= floor [premise_arms={rg['curriculum_premise_arms']}] AND A0 flat "
            f"reproduces incompetence [{rg['control_reproduces_incompetence']}] AND all cells >= "
            f"MIN_P2_EPISODES); if a premise-met curriculum arm ALSO clears the HARD floor while "
            f"A0 does not -> competence_curriculum_buildable "
            f"(recovering_arms={cg['curriculum_recovering_arms']}); else "
            f"transfer_capacity_ceiling; if readiness fails -> substrate_not_ready_requeue. "
            f"interpretation_label={result['interpretation_label']}. Feeds the "
            f"conversion_ceiling_campaign:CAMPAIGN competence-recovery build decision. Route to "
            f"/failure-autopsy for adjudication before any governance action."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(HARD_ENV_KWARGS),
        "easy_env_kwargs": dict(EASY_ENV_KWARGS),
        "med_env_kwargs": dict(MED_ENV_KWARGS),
        "config_summary": {
            "design": (
                "curriculum-vs-flat competence recovery; 4 arms x seeds at matched 300ep total "
                "P1; greedy oracle positive controls on hard + easy"
            ),
            "arms": {
                "A0_flat_hard_p1full_frozen": "CONTROL: all-ON flat on hard env, P0=200 + P1=300 (hard, e2 frozen); == 724 A1 recipe",
                "C1_curriculum_easy_hard_frozen_stageB": "curriculum: Stage-A easy (P1=150, e2 trains) -> Stage-B hard (P1=150, e2 FROZEN)",
                "C2_curriculum_easy_hard_unfrozen_stageB": "curriculum: Stage-A easy (P1=150, e2 trains) -> Stage-B hard (P1=150, e2 UNFROZEN)",
                "C3_curriculum_easy_med_hard_unfrozen": "curriculum 3-rung: easy(P1=100)->med(P1=100)->hard(P1=100), e2 unfrozen throughout",
            },
            "curriculum_transfer_semantics": (
                "the SAME agent weights carry across stage env switches; the e2 transition-replay "
                "buffer + REINFORCE outcome buffer + baseline are RESET at every env change"
            ),
            "load_bearing_dv": "HARD-env P2 mean_resources_per_episode (env.step info transition_type=='resource')",
            "load_bearing_premise": "Stage-A EASY-env mean_resources_per_episode (same statistic/floor)",
            "positive_controls": "greedy nearest-resource oracle on hard + easy envs, same seed, no agent",
            "all_on_config_sourced_from": "V3-EXQ-714 ARM_ON via V3-EXQ-724",
            "easy_env_sourced_from": "V3-EXQ-732a sanity env (hazards/reef/contamination OFF)",
            "alpha_world": 0.9,
            "use_sleep_loop": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-736 curriculum competence-recovery DIAGNOSTIC (curriculum-vs-flat; "
            "claim_ids=[])"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1_full = DRY_RUN_P1_FULL
        p1_half = DRY_RUN_P1_HALF
        p1_third = DRY_RUN_P1_THIRD
        stage_a = DRY_RUN_STAGE_A
        p2 = DRY_RUN_P2
        steps = DRY_RUN_STEPS
        oracle_eps = DRY_RUN_ORACLE_EPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1_full = P1_FULL
        p1_half = P1_HALF
        p1_third = P1_THIRD
        stage_a = STAGE_A_EVAL_EPISODES
        p2 = P2_EVAL_EPISODES
        steps = STEPS_PER_EPISODE
        oracle_eps = N_ORACLE_EPISODES

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_full=p1_full,
        p1_half=p1_half,
        p1_third=p1_third,
        stage_a_eval_episodes=stage_a,
        p2_episodes=p2,
        steps_per_episode=steps,
        oracle_episodes=oracle_eps,
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
    rg = result["readiness_gates"]
    cg = result["curriculum_gates"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"readiness={rg['readiness_met']} "
        f"hard_oracle_min/ep={rg['hard_oracle_min_resources_per_episode']} "
        f"easy_oracle_min/ep={rg['easy_oracle_min_resources_per_episode']} "
        f"premise_met={rg['curriculum_premise_met']} "
        f"control/ep={rg['control_mean_resources_per_episode']} "
        f"recovering_arms={cg['curriculum_recovering_arms']}",
        flush=True,
    )
    for aid, stt in result["per_arm"].items():
        print(
            f"  ARM {aid}: hard/ep_mean={stt['mean_resources_per_episode_mean']} "
            f"hard_supra={stt['n_seeds_supra_floor']}/{stt['n_seeds_ok']} "
            f"majority={stt['majority_supra_floor']} "
            f"stageA/ep={stt['stage_a_mean_resources_per_episode_mean']} "
            f"stageA_majority={stt['stage_a_majority_supra_floor']}",
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
