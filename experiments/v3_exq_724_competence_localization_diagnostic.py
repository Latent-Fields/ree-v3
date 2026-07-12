#!/opt/local/bin/python3
"""
V3-EXQ-724 -- COMPETENCE-LOCALIZATION DIAGNOSTIC for the integrated all-ON REE-v3 agent.

WHY THIS EXISTS (a DIAGNOSTIC, not an evidence falsifier; the self-routed label below is a
HYPOTHESIS the adjudication pipeline can falsify, NOT a governance verdict). The failure
autopsy failure_autopsy_V3-EXQ-719a_2026-07-08 (REE_assembly master 05a7679e86) recorded the
FIRST DIRECT COMPETENCE MEASUREMENT of the integrated all-mechanisms-ON stack: in the 719a P2
eval it foraged mean_resources_per_episode = 0.065 / 0.0 / 0.455 across seeds 42/43/44 -- ALL
THREE seeds BELOW the COMPETENCE_RESOURCE_FLOOR of 1.0 resource/episode (0/3). An agent that
does not forage cannot exhibit committed-action DIVERSITY, so every conversion-ceiling
committed-action dependent variable is UNDEFINED on this substrate. That single fact is the
shared root of the 654h / 485i / 625e / 460h / 460i "substrate_not_ready" retest wall: those
experiments all read a committed-action DV off an agent that never commits to foraging.

Before any /implement-substrate build we must LOCALIZE which lever causes the incompetence.
This experiment does NOT test or weight any claim (claim_ids=[], experiment_purpose=diagnostic,
non_contributory) -- it discriminates AMONG candidate causes so the follow-on
f_dominance_conversion_ceiling substrate build targets the right one.

BRAKE-EXEMPT. This asks a DIFFERENT question ("why is the integrated agent incompetent at
foraging") than the conversion-ceiling claims ARC-062 / MECH-309 ("does the agent's committed
action collapse to a single class"). It is NOT a same-claim behavioural re-test of those
claims -- it tags NO claim -- so the /failure-autopsy re-derive brake does not apply.

------------------------------------------------------------------------------------------
DESIGN -- one-factor-at-a-time (OFAT) localization anchored on the 719a incompetent config.

The 719a harness (P0 encoder/e2 warmup -> P1 frozen-encoder bias-head REINFORCE -> P2 frozen
eval) is reproduced as the BASELINE arm. Each single-factor arm changes EXACTLY ONE candidate
cause relative to the baseline, so a competence recovery in that arm localizes the cause:

  A0  baseline_allon_p1short_frozen  -- all-ON, P1=P1_SHORT, e2 encoder FROZEN in P1.
        Reproduces the 719a config that foraged 0.065/0.0/0.455 (the thing to explain).
  A1  allon_p1long_frozen            -- FACTOR (a) THIN P1 BUDGET. all-ON, P1=P1_LONG, frozen.
        Only P1 length differs from A0. If competence recovers => the deficit was undertraining.
  A2  allon_p1short_unfrozen         -- FACTOR (b) FROZEN ENCODER. all-ON, P1=P1_SHORT, but the
        SD-056 e2 world-forward contrastive keeps training THROUGH P1 (not P0-only). Only the
        encoder-freeze differs from A0. If competence recovers => a frozen/under-fit world
        model was starving the planner.
  A3  minimal_p1short_frozen         -- FACTOR (c) ALL-ON MECHANISM INTERFERENCE. A "minimal
        forage-only" config: the SAME sensory encoder + z_goal/resource-proximity/benefit
        drives + SP-CEM planner + SD-056 e2 world model as A0, but with the ENTIRE gating /
        valuation / modulation SUPERSTRUCTURE removed (no Go/No-Go constitution, no dACC, no
        f_eligibility demotion / adaptive floor, no OFC devaluation viability withdrawal, no
        modulatory selection authority / channel routing, no e3 score-diversity, no noise
        floor, no V_s rollout gating, no lateral_pfc bias, no candidate_rule_field). P1=P1_SHORT,
        frozen. If competence recovers => the all-ON gating layer collectively SUPPRESSES /
        vetoes foraging actions.
  A4  recovery_minimal_p1long_unfrozen -- RECOVERY CEILING (all three competence-favouring
        levers at once: minimal config + P1=P1_LONG + e2 unfrozen). Best-effort "can this
        architecture forage at all". If EVEN THIS stays below the floor, the deficit is NOT
        attributable to any of the three levers -- it is diffuse (env lethality / reward
        shaping / drive wiring / DV definition), and the correct route is a different build.

Single-factor arms = {A1, A2, A3}; baseline = A0; recovery ceiling = A4. Run over
SEEDS = [42, 43, 44] in CausalGridWorldV2 with the IDENTICAL env config as 719a / 714.

POSITIVE CONTROL (readiness). A GREEDY nearest-resource ORACLE forager (no agent; steps the
env toward the nearest resource each tick) is run per seed on a fresh env with the SAME
ENV_KWARGS/seed, and its mean_resources_per_episode is measured with the SAME statistic
(env.step() info transition_type == 'resource') as the agent DV. This establishes that the
COMPETENCE_RESOURCE_FLOOR of 1.0 resource/episode is ACHIEVABLE by a resource-seeking policy
in this exact env. If even the oracle cannot clear the floor, the floor is not achievable here
(env too sparse / lethal) and NO agent-architecture conclusion is licensed => the run
self-routes substrate_not_ready_requeue, NEVER a "diffuse deficit" verdict.

DV (load-bearing): P2 mean_resources_per_episode (mean over completed P2 eval episodes of the
count of env.step() ticks with info.transition_type == 'resource'). Secondary context:
hazard-hits/ep, contamination/ep, mean episode reward, and the marginal committed-class
entropy + unique committed classes.

PRE-REGISTERED SELF-ROUTE (readiness / present / diffuse) -- HYPOTHESIS, not a verdict:
  * READINESS fails (oracle below floor OR baseline A0 does NOT reproduce incompetence -- i.e.
    A0 already forages >= floor -- OR too few completed P2 episodes to estimate the DV) ->
    label `substrate_not_ready_requeue`. The premise is not measurable / not reproduced; do
    NOT draw a localization conclusion.
  * PRESENT: readiness holds AND at least one SINGLE-FACTOR arm (A1 / A2 / A3) clears the floor
    on a majority of seeds while the baseline A0 does not -> label `competence_lever_localized`.
    The incompetence is localized to the named lever(s). Route to /implement-substrate on that
    lever.
  * DIFFUSE: readiness holds, baseline reproduces incompetence, and NO single-factor arm clears
    the floor -> label `competence_deficit_diffuse`. Sub-state recorded in the gates: whether
    the multi-factor recovery ceiling A4 clears the floor (recoverable only by combining levers)
    or not (deep deficit: env / reward / drive / DV). Route to a different substrate
    investigation, NOT a single-lever build.

EVIDENCE-FOR / EVIDENCE-AGAINST (diagnostic-description requirement):
  * EVIDENCE that the deficit is a TRAINING-REGIME artifact: A1 (more P1) and/or A2 (unfrozen
    encoder) clear the floor while A0 does not -- competence is recoverable by training the
    existing all-ON stack longer / with a live world model.
  * EVIDENCE that the deficit is MECHANISM INTERFERENCE: A3 (minimal, same P1 as A0) clears the
    floor while A0 does not -- the all-ON gating superstructure is vetoing foraging.
  * EVIDENCE that the deficit is DIFFUSE (no single lever): readiness holds but none of A1/A2/A3
    clears the floor. If A4 also fails, the root is below the three tested levers.
  * EVIDENCE AGAINST any localization (substrate_not_ready_requeue): the oracle cannot clear the
    floor (env does not permit it) OR A0 already forages >= floor (719a premise not reproduced)
    OR insufficient P2 episodes. No conclusion licensed.
  The self-routed label is a HYPOTHESIS for /failure-autopsy adjudication; this experiment
  PROMOTES / DEMOTES NOTHING and tags NO claim.

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

All-ON matched-stack config sourced from V3-EXQ-714 ARM_ON (via V3-EXQ-719a):
experiments/v3_exq_719a_conversion_ceiling_dissociation_diagnostic.py (harness + all-ON stack),
experiments/v3_exq_714_fullstack_selection_valuation_conversion_falsifier.py (ARM_ON source),
ree_core/environment/causal_grid_world.py (ACTIONS / env.resources / agent_x/agent_y / step() info),
ree_core/agent.py (select_action -> executed committed first-action class),
ree_core/utils/config.py (gating-flag defaults; all default False -> minimal omits them).
See REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-719a_2026-07-08.md,
REE_assembly/evidence/planning/conversion_ceiling_campaign_plan.md.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, deque
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_724_competence_localization_diagnostic"
QUEUE_ID = "V3-EXQ-724"
CLAIM_IDS: List[str] = []                 # tags NO claim -- pure diagnostic
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# Budget (tunable; the P1_LONG arms dominate all-ON cost)
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 200          # encoder / e2 warmup (mirrors 719a / 714 P0)
P1_SHORT = 90                     # baseline P1 (mirrors 719a / 714 P1)
P1_LONG = 300                     # ~3.3x P1 (factor (a): thin-budget test)
P2_EVAL_EPISODES = 20             # fixed competence-eval episodes per cell
STEPS_PER_EPISODE = 200
N_ORACLE_EPISODES = 20            # positive-control oracle episodes per seed

# Pre-registered competence floor (nats-free behavioural count). 719a's floor: a purely
# random walker on this 12x12 reef-bipartite forage layout collects well under 1 resource/
# episode over 200 steps once resource_respawn_on_consume is on; 1.0/episode is a
# conservative supra-random floor a decisive forager clears comfortably. Its ACHIEVABILITY
# in this exact env is validated by the greedy oracle positive control (readiness gate).
COMPETENCE_RESOURCE_FLOOR = 1.0
COMPETENCE_MIN_SEEDS = 2          # of 3 (majority)
MIN_P2_EPISODES = 5              # per cell: below this the DV is not estimable

# ---------------------------------------------------------------------------
# Dry-run budget (tiny; smoke stays fast)
# ---------------------------------------------------------------------------
DRY_RUN_SEEDS = [42, 43]
DRY_RUN_P0 = 2
DRY_RUN_P1_SHORT = 2
DRY_RUN_P1_LONG = 3
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 30
DRY_RUN_ORACLE_EPS = 2

# ---------------------------------------------------------------------------
# Arm table (OFAT). role in {baseline, factor_p1_budget, factor_encoder_unfrozen,
# factor_mechanism_ablation, recovery_ceiling}.
# ---------------------------------------------------------------------------
def _arms(p1_short: int, p1_long: int) -> List[Dict[str, Any]]:
    return [
        {"arm_id": "A0_baseline_allon_p1short_frozen", "kind": "all_on",
         "p1": p1_short, "e2_train_in_p1": False, "role": "baseline"},
        {"arm_id": "A1_allon_p1long_frozen", "kind": "all_on",
         "p1": p1_long, "e2_train_in_p1": False, "role": "factor_p1_budget"},
        {"arm_id": "A2_allon_p1short_unfrozen", "kind": "all_on",
         "p1": p1_short, "e2_train_in_p1": True, "role": "factor_encoder_unfrozen"},
        {"arm_id": "A3_minimal_p1short_frozen", "kind": "minimal",
         "p1": p1_short, "e2_train_in_p1": False, "role": "factor_mechanism_ablation"},
        {"arm_id": "A4_recovery_minimal_p1long_unfrozen", "kind": "minimal",
         "p1": p1_long, "e2_train_in_p1": True, "role": "recovery_ceiling"},
    ]


SINGLE_FACTOR_ARM_IDS = (
    "A1_allon_p1long_frozen",
    "A2_allon_p1short_unfrozen",
    "A3_minimal_p1short_frozen",
)
BASELINE_ARM_ID = "A0_baseline_allon_p1short_frozen"
RECOVERY_ARM_ID = "A4_recovery_minimal_p1long_unfrozen"

# ---------------------------------------------------------------------------
# All-ON matched-stack constants (sourced from V3-EXQ-714 ARM_ON via 719a)
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


# Identical env to V3-EXQ-714 / 719a (SD-054 reef + hazard_food_attraction + bipartite).
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


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


# ---------------------------------------------------------------------------
# Agent config: all-ON (719a ARM_ON) vs minimal forage-only.
# Both share the sensory encoder + z_goal / resource-proximity / benefit drives + SP-CEM
# planner + SD-056 e2 world model. The ONLY difference is the gating / valuation / modulation
# SUPERSTRUCTURE, which minimal omits (every such flag defaults False in REEConfig).
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
        # Shared planner + world-model readout (held FIXED across kinds).
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        candidate_summary_source="e2_world_forward",
        # Shared SD-056 e2 world-forward contrastive levers.
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


def _make_agent(env: CausalGridWorldV2, kind: str) -> REEAgent:
    kwargs = _base_config_kwargs(env)
    if kind == "all_on":
        kwargs.update(_all_on_extra_kwargs())
    elif kind != "minimal":
        raise ValueError(f"unknown agent kind {kind!r}")
    cfg = REEConfig.from_dims(**kwargs)
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-719a)
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
# obs helpers (mirror V3-EXQ-719a)
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
# P1 two-head REINFORCE (mirror V3-EXQ-719a; no-ops for minimal, heads absent)
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
# ACTIONS: {0:(-1,0)=N, 1:(1,0)=S, 2:(0,-1)=W, 3:(0,1)=E, 4:(0,0)=stay}, grid[x,y] x=row.
# Measures the SAME statistic as the agent DV: mean resources/episode via transition_type.
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


def _run_oracle(seed: int, n_episodes: int, steps_per_episode: int) -> Dict[str, Any]:
    env = _make_env(seed)
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
# Per-cell (arm x seed) run
# ---------------------------------------------------------------------------
def _run_cell(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    kind = arm["kind"]
    p1_episodes = int(arm["p1"])
    e2_train_in_p1 = bool(arm["e2_train_in_p1"])

    env = _make_env(seed)
    agent = _make_agent(env, kind)
    has_ofc = getattr(agent, "ofc", None) is not None
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = (
        torch.optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS)
        if has_lpfc else None
    )
    ofc_deval_opt = (
        torch.optim.Adam(
            list(agent.ofc.devaluation_bias_head_parameters()), lr=LR_OFC_DEVAL
        )
        if has_ofc else None
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

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    committed_class_counts: Dict[int, int] = {}
    p2_ep_resources: List[int] = []
    p2_ep_hazard_hits: List[int] = []
    p2_ep_contaminations: List[int] = []
    p2_ep_rewards: List[float] = []
    n_p2_eps_completed = 0

    for ep in range(total_train_eps):
        is_p1 = (p1_start <= ep < p2_start)
        is_p2 = (ep >= p2_start)
        is_p0 = (not is_p1 and not is_p2)
        phase_label = "P2" if is_p2 else ("P1" if is_p1 else "P0")

        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

        ep_reward = 0.0
        ep_buf: List[Tuple[torch.Tensor, int]] = []

        ep_resources = 0
        ep_hazard_hits = 0
        ep_contaminations = 0
        ep_reward_signal = 0.0

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

            # P1 REINFORCE snapshot of candidate summaries (all-ON only; heads present).
            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and has_lpfc and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            # P2: inject trained OFC devaluation viability into Go/No-Go (all-ON only).
            if is_p2 and has_ofc and candidates and len(candidates) >= 2:
                ofc_summ_p2 = _consumed_summaries(agent, candidates)
                viability_sig: Optional[torch.Tensor] = None
                if ofc_summ_p2 is not None and torch.isfinite(ofc_summ_p2).all():
                    with torch.no_grad():
                        deval_bias_p2 = agent.ofc.compute_devaluation_bias(
                            ofc_summ_p2
                        ).detach()
                    viability_sig = _build_viability_nogo(deval_bias_p2)
                if viability_sig is not None:
                    agent.set_injected_go_nogo_signals(
                        {"viability": viability_sig.to(agent.device)}
                    )
                else:
                    agent.set_injected_go_nogo_signals(None)
            elif is_p2 and has_ofc:
                agent.set_injected_go_nogo_signals(None)

            action = agent.select_action(candidates, ticks)
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
            elif is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 always; P1 only when the arm unfreezes the encoder;
            # NEVER in P2 (frozen for a clean competence measurement).
            train_e2_now = is_p0 or (is_p1 and e2_train_in_p1)
            if train_e2_now and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )

            _, _harm_signal, done, info, obs_dict = env.step(action)
            harm_signal = float(_harm_signal)
            if is_p1:
                ep_reward += harm_signal

            if is_p2:
                ep_reward_signal += harm_signal
                ttype = str(info.get("transition_type", "none"))
                if ttype == "resource":
                    ep_resources += 1
                elif ttype == "env_caused_hazard":
                    ep_hazard_hits += 1
                if ttype == "agent_caused_hazard" or float(
                    info.get("contamination_delta", 0.0)
                ) > 0.0:
                    ep_contaminations += 1

            with torch.no_grad():
                agent.update_residue(
                    harm_signal=harm_signal,
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
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

        # P1 end-of-episode: TWO-head REINFORCE (no-op when heads absent / minimal).
        if is_p1 and (has_lpfc or has_ofc):
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

        if is_p2 and error_note is None:
            p2_ep_resources.append(ep_resources)
            p2_ep_hazard_hits.append(ep_hazard_hits)
            p2_ep_contaminations.append(ep_contaminations)
            p2_ep_rewards.append(ep_reward_signal)
            n_p2_eps_completed += 1

        # Absolute per-cell progress print ('[train] ... ep N/M' the runner parses).
        cur = ep + 1
        if cur % 25 == 0 or cur == total_train_eps or phase_label == "P2":
            print(
                f"  [train] localize arm={arm['arm_id']} seed={seed} "
                f"phase={phase_label} ep {cur}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

    mean_resources_per_ep = (
        float(sum(p2_ep_resources) / len(p2_ep_resources)) if p2_ep_resources else 0.0
    )
    mean_hazard_hits_per_ep = (
        float(sum(p2_ep_hazard_hits) / len(p2_ep_hazard_hits)) if p2_ep_hazard_hits else 0.0
    )
    mean_contaminations_per_ep = (
        float(sum(p2_ep_contaminations) / len(p2_ep_contaminations))
        if p2_ep_contaminations else 0.0
    )
    mean_episode_reward = (
        float(sum(p2_ep_rewards) / len(p2_ep_rewards)) if p2_ep_rewards else 0.0
    )
    competence_supra_floor = bool(mean_resources_per_ep >= COMPETENCE_RESOURCE_FLOOR)
    marginal_entropy = _marginal_entropy_nats(committed_class_counts)

    return {
        "arm_id": arm["arm_id"],
        "arm_kind": kind,
        "arm_role": arm["role"],
        "seed": int(seed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "e2_train_in_p1": e2_train_in_p1,
        "p2_episodes_requested": int(p2_episodes),
        "n_p2_eps_completed": int(n_p2_eps_completed),
        "error_note": error_note,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        # ----- LOAD-BEARING DV -----
        "mean_resources_per_episode": round(mean_resources_per_ep, 6),
        "competence_supra_floor": competence_supra_floor,
        # ----- context -----
        "mean_hazard_hits_per_episode": round(mean_hazard_hits_per_ep, 6),
        "mean_contaminations_per_episode": round(mean_contaminations_per_ep, 6),
        "mean_episode_reward": round(mean_episode_reward, 6),
        "marginal_committed_class_entropy_nats": round(marginal_entropy, 6),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        "per_p2_episode_resources": [int(x) for x in p2_ep_resources],
    }


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _arm_majority_supra(rows: List[Dict[str, Any]], min_seeds: int) -> bool:
    n = sum(1 for r in rows if r.get("error_note") is None and r["competence_supra_floor"])
    return bool(n >= min_seeds)


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_short: int,
    p1_long: int,
    p2_episodes: int,
    steps_per_episode: int,
    oracle_episodes: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arms = _arms(p1_short, p1_long)

    print(
        f"Competence-localization diagnostic (OFAT; {len(arms)} arms x {len(seeds)} seeds; "
        f"P0={p0_episodes}, P1_short={p1_short}, P1_long={p1_long}, "
        f"P2_eval={p2_episodes}, steps={steps_per_episode}, "
        f"oracle_eps={oracle_episodes}, dry_run={dry_run})",
        flush=True,
    )

    # ----- Positive-control oracle (per seed) -----
    oracle_rows: List[Dict[str, Any]] = []
    for s in seeds:
        orow = _run_oracle(s, oracle_episodes, steps_per_episode)
        oracle_rows.append(orow)
        print(
            f"  [oracle] seed={s} mean_resources/ep="
            f"{orow['mean_resources_per_episode']} "
            f"max={orow['max_resources_in_episode']}",
            flush=True,
        )
    oracle_mean_resources = _mean([o["mean_resources_per_episode"] for o in oracle_rows])
    oracle_min_resources = min(
        [o["mean_resources_per_episode"] for o in oracle_rows], default=0.0
    )
    oracle_clears_floor = bool(oracle_min_resources >= COMPETENCE_RESOURCE_FLOOR)

    # ----- Arm x seed cells -----
    cells: List[Dict[str, Any]] = []
    for arm in arms:
        for s in seeds:
            print(f"Seed {s} Condition {arm['arm_id']}", flush=True)
            slice_cfg = {
                "arm_id": arm["arm_id"],
                "kind": arm["kind"],
                "p0_episodes": int(p0_episodes),
                "p1_episodes": int(arm["p1"]),
                "e2_train_in_p1": bool(arm["e2_train_in_p1"]),
                "p2_episodes": int(p2_episodes),
                "steps_per_episode": int(steps_per_episode),
                "env_kwargs": dict(ENV_KWARGS),
            }
            with arm_cell(
                s,
                config_slice=slice_cfg,
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = _run_cell(arm, s, p0_episodes, p2_episodes, steps_per_episode)
                cell.stamp(row)
            cells.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(
                f"verdict: {verdict} (arm={arm['arm_id']} seed={s} "
                f"resources/ep={row['mean_resources_per_episode']} "
                f"supra_floor={row['competence_supra_floor']})",
                flush=True,
            )

    # ----- Per-arm aggregation -----
    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        rows = [c for c in cells if c["arm_id"] == arm["arm_id"]]
        ok_rows = [r for r in rows if r["error_note"] is None]
        per_arm[arm["arm_id"]] = {
            "arm_id": arm["arm_id"],
            "kind": arm["kind"],
            "role": arm["role"],
            "p1_episodes": int(arm["p1"]),
            "e2_train_in_p1": bool(arm["e2_train_in_p1"]),
            "n_seeds_ok": int(len(ok_rows)),
            "n_seeds_min_p2": int(
                sum(1 for r in ok_rows if r["n_p2_eps_completed"] >= MIN_P2_EPISODES)
            ),
            "mean_resources_per_episode_mean": round(
                _mean([r["mean_resources_per_episode"] for r in ok_rows]), 6
            ),
            "n_seeds_supra_floor": int(
                sum(1 for r in ok_rows if r["competence_supra_floor"])
            ),
            "majority_supra_floor": _arm_majority_supra(ok_rows, COMPETENCE_MIN_SEEDS),
            "mean_hazard_hits_per_episode_mean": round(
                _mean([r["mean_hazard_hits_per_episode"] for r in ok_rows]), 6
            ),
            "mean_episode_reward_mean": round(
                _mean([r["mean_episode_reward"] for r in ok_rows]), 6
            ),
        }

    # ----- Readiness -----
    baseline_stats = per_arm[BASELINE_ARM_ID]
    baseline_reproduces_incompetence = bool(not baseline_stats["majority_supra_floor"])
    all_cells_ok = [c for c in cells if c["error_note"] is None]
    sufficient_p2 = bool(
        all_cells_ok
        and all(c["n_p2_eps_completed"] >= MIN_P2_EPISODES for c in all_cells_ok)
    )
    readiness_met = bool(
        oracle_clears_floor and baseline_reproduces_incompetence and sufficient_p2
    )

    # ----- Localization gate (single-factor arms) -----
    localizing_arms = [
        aid for aid in SINGLE_FACTOR_ARM_IDS if per_arm[aid]["majority_supra_floor"]
    ]
    single_lever_localized = bool(localizing_arms)
    recovery_supra = bool(per_arm[RECOVERY_ARM_ID]["majority_supra_floor"])

    # ----- Self-route (HYPOTHESIS, not a verdict) -----
    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif single_lever_localized:
        outcome = "PASS"
        label = "competence_lever_localized"
    else:
        outcome = "FAIL"
        label = "competence_deficit_diffuse"
    direction = "non_contributory"

    interpretation = {
        "label": label,
        "localizing_arms": localizing_arms,
        "recovery_ceiling_supra_floor": recovery_supra,
        "preconditions": [
            {
                "name": "oracle_resource_channel_clears_floor",
                "kind": "readiness",
                "description": (
                    "A greedy nearest-resource ORACLE (no agent) clears "
                    "COMPETENCE_RESOURCE_FLOOR resources/episode in this exact env, proving the "
                    "floor is ACHIEVABLE by a resource-seeking policy. Same statistic as the "
                    "agent DV (env.step info transition_type=='resource'). Below-floor => the "
                    "floor is not achievable here (env too sparse/lethal) => "
                    "substrate_not_ready_requeue, NEVER a diffuse-deficit verdict."
                ),
                "control": "greedy nearest-resource oracle forager, same ENV_KWARGS/seed, no agent",
                "measured": float(round(oracle_min_resources, 6)),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(oracle_clears_floor),
            },
            {
                "name": "baseline_reproduces_incompetence",
                "kind": "readiness",
                "description": (
                    "The baseline arm A0 (719a all-ON config) must forage BELOW the floor on a "
                    "majority of seeds -- i.e. the 719a incompetence must reproduce -- for a "
                    "localization to be meaningful. If A0 already clears the floor the premise "
                    "is not reproduced => substrate_not_ready_requeue, NOT a localization."
                ),
                "control": "A0 baseline mean_resources/ep vs floor (majority of seeds)",
                "measured": float(baseline_stats["mean_resources_per_episode_mean"]),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "direction": "upper",
                "met": bool(baseline_reproduces_incompetence),
            },
            {
                "name": "sufficient_p2_episodes_all_cells",
                "kind": "readiness",
                "description": (
                    "Every completed cell must log >= MIN_P2_EPISODES P2 eval episodes so "
                    "mean_resources_per_episode is estimable. Below => "
                    "substrate_not_ready_requeue."
                ),
                "control": "min completed P2 episodes across all cells",
                "measured": float(
                    min([c["n_p2_eps_completed"] for c in all_cells_ok], default=0)
                ),
                "threshold": float(MIN_P2_EPISODES),
                "met": bool(sufficient_p2),
            },
        ],
        "criteria": [
            {
                "name": "single_factor_arm_recovers_competence",
                "load_bearing": True,
                "passed": bool(single_lever_localized),
            },
        ],
        "criteria_non_degenerate": {
            "oracle_clears_floor": bool(oracle_clears_floor),
            "baseline_reproduces_incompetence": bool(baseline_reproduces_incompetence),
            "sufficient_p2_episodes": bool(sufficient_p2),
            "any_single_factor_arm_supra": bool(single_lever_localized),
            "recovery_ceiling_supra": bool(recovery_supra),
        },
    }

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": seeds,
        "p0_episodes": int(p0_episodes),
        "p1_short": int(p1_short),
        "p1_long": int(p1_long),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "oracle_episodes": int(oracle_episodes),
        "decision_rule_thresholds": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "competence_min_seeds": int(COMPETENCE_MIN_SEEDS),
            "min_p2_episodes": int(MIN_P2_EPISODES),
            "single_factor_arm_ids": list(SINGLE_FACTOR_ARM_IDS),
            "baseline_arm_id": BASELINE_ARM_ID,
            "recovery_arm_id": RECOVERY_ARM_ID,
        },
        "readiness_gates": {
            "oracle_clears_floor": oracle_clears_floor,
            "oracle_mean_resources_per_episode": round(oracle_mean_resources, 6),
            "oracle_min_resources_per_episode": round(oracle_min_resources, 6),
            "baseline_reproduces_incompetence": baseline_reproduces_incompetence,
            "baseline_mean_resources_per_episode": baseline_stats[
                "mean_resources_per_episode_mean"
            ],
            "sufficient_p2_episodes": sufficient_p2,
            "readiness_met": readiness_met,
        },
        "localization_gates": {
            "localizing_single_factor_arms": localizing_arms,
            "single_lever_localized": single_lever_localized,
            "recovery_ceiling_supra_floor": recovery_supra,
        },
        "oracle_results": oracle_rows,
        "per_arm": per_arm,
        "arm_results": cells,
        "interpretation_grid": {
            "competence_lever_localized": (
                "readiness holds AND at least one SINGLE-FACTOR arm (A1 more P1 / A2 unfrozen "
                "encoder / A3 minimal-config) clears the floor on a majority of seeds while the "
                "baseline A0 does not. The incompetence is localized to the named lever(s) "
                "(interpretation.localizing_arms). HYPOTHESIS (not a verdict): route to "
                "/implement-substrate on that lever for the f_dominance_conversion_ceiling build."
            ),
            "competence_deficit_diffuse": (
                "readiness holds, A0 reproduces the incompetence, and NO single-factor arm "
                "clears the floor. recovery_ceiling_supra_floor records whether the multi-factor "
                "A4 recovers (competence needs combined levers) or not (deep deficit: env "
                "lethality / reward shaping / drive wiring / DV definition). HYPOTHESIS: route to "
                "a different substrate investigation, NOT a single-lever build."
            ),
            "substrate_not_ready_requeue": (
                "the greedy oracle cannot clear the floor (env does not permit it), OR A0 "
                "already forages >= floor (719a premise not reproduced), OR a cell logged fewer "
                "than MIN_P2_EPISODES eval episodes. NOT a verdict -- re-examine env/floor/budget "
                "and re-queue. Draw NO conclusion about the competence root."
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
    lg = result["localization_gates"]
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
            f"V3-EXQ-724 COMPETENCE-LOCALIZATION DIAGNOSTIC (experiment_purpose=diagnostic, "
            f"claim_ids=[], non_contributory -- EXCLUDED from governance scoring; PROMOTES / "
            f"DEMOTES NOTHING). Localizes WHY the integrated all-ON REE-v3 agent foraged "
            f"0.065/0.0/0.455 resources/episode (below the {COMPETENCE_RESOURCE_FLOOR} floor, "
            f"0/3 seeds) in the failure_autopsy_V3-EXQ-719a_2026-07-08 competence measurement -- "
            f"the shared root of the 654h/485i/625e/460h/460i substrate_not_ready retest wall. "
            f"OFAT design anchored on the 719a config as baseline A0; single-factor arms isolate "
            f"(a) thin P1 budget [A1 P1={result['p1_long']} vs A0 P1={result['p1_short']}], "
            f"(b) frozen encoder [A2 e2 trained through P1], (c) all-ON mechanism interference "
            f"[A3 minimal forage-only config]; A4 is the multi-factor recovery ceiling. A greedy "
            f"nearest-resource ORACLE (positive control) validates the floor is achievable in "
            f"this env (oracle_min_resources/ep={rg['oracle_min_resources_per_episode']}, "
            f"clears_floor={rg['oracle_clears_floor']}). Load-bearing DV: P2 "
            f"mean_resources_per_episode. Self-route (HYPOTHESIS, not a verdict): "
            f"readiness_met={rg['readiness_met']} (oracle clears floor AND baseline reproduces "
            f"incompetence AND all cells >= MIN_P2_EPISODES); if a single-factor arm recovers "
            f"competence on a majority of seeds -> competence_lever_localized "
            f"(localizing_arms={lg['localizing_single_factor_arms']}); else "
            f"competence_deficit_diffuse (recovery_ceiling_supra="
            f"{lg['recovery_ceiling_supra_floor']}); if readiness fails -> "
            f"substrate_not_ready_requeue. interpretation_label="
            f"{result['interpretation_label']}. Feeds the /implement-substrate "
            f"f_dominance_conversion_ceiling competence/training-regime build. Route to "
            f"/failure-autopsy for adjudication before any governance action."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "OFAT competence localization; 5 arms x 3 seeds + greedy-oracle positive control",
            "arms": {
                "A0_baseline_allon_p1short_frozen": "all-ON (719a ARM_ON), P1 short, e2 frozen in P1 -- reproduces 719a incompetence",
                "A1_allon_p1long_frozen": "factor (a) thin-P1-budget: all-ON, P1 long, e2 frozen",
                "A2_allon_p1short_unfrozen": "factor (b) frozen-encoder: all-ON, P1 short, e2 trained through P1",
                "A3_minimal_p1short_frozen": "factor (c) mechanism-interference: minimal forage-only (drives+SP-CEM+e2, no gating superstructure), P1 short, e2 frozen",
                "A4_recovery_minimal_p1long_unfrozen": "recovery ceiling: minimal + P1 long + e2 unfrozen",
            },
            "minimal_config_definition": (
                "shared sensory encoder + z_goal/resource-proximity/benefit drives + SP-CEM "
                "planner + SD-056 e2 world model; ALL gating/valuation/modulation flags omitted "
                "(default False): Go/No-Go+dACC, f_eligibility demotion+adaptive floor, OFC "
                "analog+devaluation, modulatory authority+channel routing, e3 score-diversity, "
                "noise floor, V_s rollout gating, lateral_pfc bias, candidate_rule_field."
            ),
            "load_bearing_dv": "P2 mean_resources_per_episode (env.step info transition_type=='resource')",
            "positive_control": "greedy nearest-resource oracle forager, same ENV_KWARGS/seed, no agent",
            "all_on_config_sourced_from": "V3-EXQ-714 ARM_ON via V3-EXQ-719a",
            "alpha_world": 0.9,
            "reef_bipartite_layout": True,
            "use_sleep_loop": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-724 competence-localization DIAGNOSTIC (OFAT; why the all-ON agent "
            "does not forage; claim_ids=[])"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1_short = DRY_RUN_P1_SHORT
        p1_long = DRY_RUN_P1_LONG
        p2 = DRY_RUN_P2
        steps = DRY_RUN_STEPS
        oracle_eps = DRY_RUN_ORACLE_EPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1_short = P1_SHORT
        p1_long = P1_LONG
        p2 = P2_EVAL_EPISODES
        steps = STEPS_PER_EPISODE
        oracle_eps = N_ORACLE_EPISODES

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_short=p1_short,
        p1_long=p1_long,
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
    rg = result["readiness_gates"]
    lg = result["localization_gates"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"readiness={rg['readiness_met']} "
        f"oracle_min/ep={rg['oracle_min_resources_per_episode']} "
        f"baseline/ep={rg['baseline_mean_resources_per_episode']} "
        f"localizing_arms={lg['localizing_single_factor_arms']} "
        f"recovery_supra={lg['recovery_ceiling_supra_floor']}",
        flush=True,
    )
    for aid, st in result["per_arm"].items():
        print(
            f"  ARM {aid}: resources/ep_mean={st['mean_resources_per_episode_mean']} "
            f"supra_seeds={st['n_seeds_supra_floor']}/{st['n_seeds_ok']} "
            f"majority_supra={st['majority_supra_floor']}",
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
