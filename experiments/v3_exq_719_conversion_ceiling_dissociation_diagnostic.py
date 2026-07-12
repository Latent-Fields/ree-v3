#!/opt/local/bin/python3
"""
V3-EXQ-719 -- CONVERSION-CEILING DISSOCIATION DIAGNOSTIC for MECH-309 / ARC-062.

PURPOSE (a DIAGNOSTIC, not an evidence falsifier -- the self-routed label below is a
HYPOTHESIS, not a verdict). The conversion-ceiling campaign's dependent variable --
`committed_class_entropy` (the Shannon entropy of the MARGINAL distribution over the
agent's committed first-action class) -- CONFLATES two very different agent states that a
marginal-entropy metric cannot distinguish:

  (A) GENUINE MONOMODAL POLICY COLLAPSE -- the agent commits to the SAME first-action
      regardless of world-state (the real pathology claim MECH-309 names; low marginal
      entropy is a TRUE positive for collapse). vs
  (B) DECISIVE, STATE-APPROPRIATE COMMITMENT -- the agent commits to DIFFERENT actions in
      DIFFERENT states (a WORKING agent). Marginal entropy can ALSO be low here (if the
      agent is decisive within each state), so a marginal-entropy metric wrongly flags a
      competent state-conditioned policy as "collapsed".

Marginal entropy H(committed_action) cannot tell (A) from (B). The MUTUAL INFORMATION
I(state ; committed_action) CAN: it is ~0 for (A) (action independent of state) and > 0
for (B) (action driven by state). This run measures BOTH on the SAME ticks, plus
behavioural competence, to dissociate them.

DESIGN. A SINGLE integrated all-mechanisms-ON configuration (also a behavioural
demonstrator), run in CausalGridWorldV2 over SEEDS = [42, 43, 44]. Phases:
  P0 (encoder / e2 world-forward warmup; SD-056 online contrastive; CRF field matures) --
     the all-ON substrate needs training to be non-degenerate; mirrors V3-EXQ-714 P0=200.
  P1 (frozen-encoder outcome-coupled REINFORCE on the lateral_pfc bias head + the decoupled
     OFC devaluation head; field keeps maturing) -- the same TWO-head training the all-ON
     stack needs; mirrors V3-EXQ-714 P1=90.
  P2 (LONG eval / logging phase; all frozen; OFC devaluation viability injected into the
     Go/No-Go gate exactly as 714) -- 60 ep x 200 steps ~ 12000 ticks/seed so MI over up to
     ~25 state-bins is estimable.

Per P2 tick we log:
  * state_bin -- derived from GROUND-TRUTH env state (env.grid, env.agent_x/agent_y; this is
    INSTRUMENTATION, NOT fed to the agent). state_bin = (nearest-salient-entity-type) x
    (relative-direction-to-it). Salient types: resource, hazard, waypoint. Relative
    direction quantised to {N, S, E, W, on-top}. Up to 3 x 5 = 15 occupied bins, plus a
    fallback "empty/open" bin when no salient entity within SALIENT_RADIUS.
  * committed_class -- the agent's EXECUTED first-action class this tick
    (int(action.argmax()) of agent.select_action(...)), the clean committed proxy (the SAME
    quantity V3-EXQ-714 logs as committed_class).
  * behaviour -- resource_collected / hazard_hit / contamination_event, read from the env
    step() return `info` (transition_type / contamination_delta) + the harm_signal reward.

THREE DV FAMILIES (per seed, then aggregated):
  1. marginal_committed_class_entropy_nats -- Shannon entropy of the MARGINAL committed_class
     distribution over all P2 ticks (reproduces the campaign's "ceiling" number).
  2. mutual_information_state_committed_nats = I(state_bin ; committed_class); plus
     conditional_entropy_committed_given_state_nats = H(committed|state); plus normalized_mi
     = I / H(committed_class) (fraction of committed-action entropy explained by state).
     DEBIASED with a shuffle null: permute committed_class labels across ticks (breaking
     state-action coupling), recompute MI N_SHUFFLE=200 times -> mi_shuffle_null_mean,
     mi_shuffle_null_p95, mi_debiased = raw_mi - mi_shuffle_null_mean. Fraction of seeds
     with raw_mi > mi_shuffle_null_p95 (real, above-chance coupling).
  3. competence -- resources/episode, hazard-hits/episode, contamination/episode, mean
     episode reward.

PRE-REGISTERED DISSOCIATION (the self-routed interpretation.label is a HYPOTHESIS the
pipeline can falsify, NOT a verdict):
  * mi_debiased HIGH (state genuinely drives commitment: mi_debiased > MI_DEBIASED_FLOOR AND
    raw_mi > shuffle_p95 on >= MI_REAL_MIN_SEEDS/3 seeds) AND competence supra-random
    (resources/episode > COMPETENCE_RESOURCE_FLOOR on >= 2/3 seeds) -> label
    `decisive_state_appropriate_commitment` (=> the "conversion ceiling" is LARGELY a
    marginal-entropy artifact; reframes MECH-309 / ARC-062).
  * mi_debiased ~0 (agent does the same thing regardless of state: mi_debiased <=
    MI_DEBIASED_FLOOR AND NOT the real-coupling gate) -> label `genuine_monomodal_collapse`
    (=> warrants the GAP-A-divergence-survival substrate build).
  * mixed / insufficient samples (READINESS fails: too few total P2 ticks OR too few
    occupied state-bins to estimate MI) -> label `substrate_not_ready_requeue` (NOT a
    verdict; re-queue at a larger P2 budget).

EVIDENCE-FOR / EVIDENCE-AGAINST (diagnostic-description requirement).
  * EVIDENCE FOR genuine collapse (supports MECH-309 as a real pathology): mi_debiased ~0
    AND raw_mi NOT above the shuffle p95 -- the committed action is statistically independent
    of world-state; the low marginal entropy is a TRUE collapse signature.
  * EVIDENCE FOR the metric-artifact reframe (the ceiling is largely a measurement artifact):
    mi_debiased HIGH and above the shuffle null AND competence supra-random -- the agent
    commits to different, state-appropriate actions; the low MARGINAL entropy misreads a
    working state-conditioned policy as collapsed.
  * EVIDENCE AGAINST either conclusion (self-route substrate_not_ready_requeue, do NOT draw a
    verdict): total P2 ticks below TOTAL_TICKS_FLOOR or occupied state-bins below
    MIN_OCCUPIED_BINS -- MI is under-sampled and neither dissociation branch is licensed.
  The self-routed label is a HYPOTHESIS for the adjudication pipeline (/failure-autopsy),
  NOT a governance verdict; this experiment PROMOTES / DEMOTES NOTHING by itself.

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Claims: [MECH-309, ARC-062] -- experiment_purpose = "diagnostic" (EXCLUDED from governance
confidence / conflict scoring; context only). Single all-ON config (no OFF/treatment arm
grid) -> no arm_results key; per-seed results under per_seed_results; module-level
ARM_FINGERPRINT_EXEMPT set.

All-ON matched-stack config sourced from V3-EXQ-714
(experiments/v3_exq_714_fullstack_selection_valuation_conversion_falsifier.py) ARM_ON:
SP-CEM + candidate_summary_source=e2_world_forward (SD-056, GAP-A) +
use_modulatory_selection_authority (std basis) + channel routing +
use_f_eligibility_demotion + use_f_eligibility_adaptive_floor (MECH-448) +
use_go_nogo_constitution + use_dacc (MECH-449) + the P3 OFC valuation face
(use_ofc_analog + use_ofc_devaluation_head + ofc_train_devaluation_head, injected as the
Go/No-Go viability axis in P2) + MECH-341 stratified + MECH-313 noise floor + V_s minimal +
use_gated_policy + use_lateral_pfc_analog (bias head trained P1) + use_candidate_rule_field
(the differentiated crf_source, matured via crf_persist + the 666c/654e maintenance levers).

See REE_assembly/evidence/planning/conversion_ceiling_campaign_plan.md,
REE_assembly/evidence/planning/conversion_ceiling_prong_map.md,
REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md (GAP-I),
experiments/v3_exq_714_fullstack_selection_valuation_conversion_falsifier.py (all-ON config source),
ree_core/environment/causal_grid_world.py (env.grid / agent_x/agent_y / ENTITY_TYPES / step() info),
ree_core/agent.py (select_action -> executed committed action class),
ree_core/pfc/ofc_analog.py (the decoupled devaluation_bias_head + viability withdrawal).
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
import torch.nn.functional as F

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import reset_all_rng
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_719_conversion_ceiling_dissociation_diagnostic"
QUEUE_ID = "V3-EXQ-719"
SUPERSEDES = None
CLAIM_IDS: List[str] = ["MECH-309", "ARC-062"]
EXPERIMENT_PURPOSE = "diagnostic"

# Single all-ON config (no OFF/treatment arm grid); no arm_results key is written.
ARM_FINGERPRINT_EXEMPT = "single-config diagnostic; no OFF/treatment arm grid"

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 200         # encoder / e2 warmup + CRF maturation (mirrors 714 P0)
P1_BIAS_TRAIN_EPISODES = 90      # frozen-encoder TWO-head REINFORCE (mirrors 714 P1)
P2_EVAL_EPISODES = 60            # LONG eval/logging window; 60 x 200 = 12000 ticks/seed
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42, 43]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 3
DRY_RUN_STEPS = 40

# ---------------------------------------------------------------------------
# State-bin instrumentation (ground-truth; NOT fed to the agent)
# ---------------------------------------------------------------------------
# Salient entity types considered for the nearest-salient state factor.
SALIENT_ENTITY_NAMES: Tuple[str, ...] = ("resource", "hazard", "waypoint")
# Manhattan radius within which a salient entity counts as "nearby"; beyond it the
# tick is the fallback "empty/open" bin.
SALIENT_RADIUS = 6
# Relative-direction quantisation: N, S, E, W, or ON (on-top / same cell).
REL_DIRECTIONS: Tuple[str, ...] = ("N", "S", "E", "W", "ON")

# ---------------------------------------------------------------------------
# Pre-registered thresholds (constants; NOT derived from the run's own statistics)
# ---------------------------------------------------------------------------
# Shuffle-null debias.
N_SHUFFLE = 200                  # permutations of committed_class to build the MI null

# Dissociation gates.
# MI_DEBIASED_FLOOR (nats): a debiased I(state;action) at/below this is "state does not
# drive commitment" (collapse branch); strictly above it is a candidate for the
# state-appropriate branch. 0.05 nats mirrors the campaign's C2_LIFT_MARGIN_NATS entropy
# margin (V3-EXQ-714) -- the same "meaningful bits" scale used elsewhere in this lineage.
MI_DEBIASED_FLOOR = 0.05
# Real-coupling gate: raw_mi must exceed the per-seed shuffle p95 (above-chance coupling)
# on at least this many of the 3 seeds for the state-appropriate branch.
MI_REAL_MIN_SEEDS = 2            # of 3
# Competence: mean resources collected per episode must clear this floor on a majority of
# seeds for the "working agent" (state-appropriate) branch. A purely random walker on this
# 12x12 reef-bipartite forage layout collects well under 1 resource/episode over 200 steps
# once resource_respawn_on_consume is on; 1.0/episode is a conservative supra-random floor
# (a decisive forager clears it comfortably).
COMPETENCE_RESOURCE_FLOOR = 1.0
COMPETENCE_MIN_SEEDS = 2         # of 3

# READINESS (MI estimability). Below either floor -> substrate_not_ready_requeue.
# Need enough total P2 ticks AND enough occupied state-bins to estimate MI over up to
# ~15-16 bins x |action classes| without the estimate being dominated by finite-sample bias.
TOTAL_TICKS_FLOOR = 3000         # per seed: << the 12000 the full P2 delivers; guards a truncated run
MIN_OCCUPIED_BINS = 4            # distinct occupied state-bins required to estimate I(state;action)
READINESS_MIN_SEEDS = 2          # of 3 seeds must be MI-estimable to license a dissociation branch

MIN_SEEDS_FOR_PASS = 2           # of 3 (generic majority)

# ---------------------------------------------------------------------------
# All-ON matched-stack constants (sourced from V3-EXQ-714 ARM_ON)
# ---------------------------------------------------------------------------
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7   # 714 CRF-gate calibration amend (FAULT 1)
CRF_TOLERANCE_CONFLICT_CAP = 3             # 714 (FAULT 2a)
CRF_MAINTENANCE_COUPLE_TO_THETA = True     # 714 (FAULT 2b)
CRF_MAINTENANCE_FLOOR = 0.45               # 714 / 666c MAINTENANCE_FLOOR
CRF_MAINTENANCE_DECAY = 0.0                # 714 / 666c MAINTENANCE_DECAY

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

# P3 OFC valuation face (714).
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


# Identical env to V3-EXQ-714 (SD-054 reef + hazard_food_attraction + bipartite layout).
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


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """All-mechanisms-ON matched-stack agent (V3-EXQ-714 ARM_ON config,
    use_candidate_rule_field=True). This is the integrated substrate under test AND a
    behavioural demonstrator."""
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
        # --- All-ON matched stack (from 714 ARM_ON) ---
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
        use_candidate_rule_field=True,   # all-ON: the differentiated crf_source
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-714)
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
# obs helpers (mirror V3-EXQ-714)
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


def _ofc_deval_bias_range(agent: REEAgent, summaries: torch.Tensor) -> Optional[float]:
    ofc = getattr(agent, "ofc", None)
    if ofc is None or summaries is None:
        return None
    try:
        with torch.no_grad():
            bias = ofc.compute_devaluation_bias(summaries).detach().reshape(-1)
        if bias.numel() < 2:
            return None
        return float((bias.max() - bias.min()).item())
    except Exception:
        return None


def _build_viability_nogo(bias_low: torch.Tensor) -> Optional[torch.Tensor]:
    """485m / 714 _build_viability_nogo: trained OFC devalued valuation -> per-candidate
    viability No-Go (most-devalued -> viability ~0). Flat bias -> None (no withdrawal)."""
    bl = bias_low.detach().reshape(-1)
    if bl.numel() < 2:
        return None
    rng = float((bl.max() - bl.min()).item())
    if rng < 1e-6:
        return None
    bln = (bl - bl.min()) / (bl.max() - bl.min())
    return (1.0 - bln).detach()


# ---------------------------------------------------------------------------
# P1 two-head REINFORCE (mirror V3-EXQ-714)
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
        log_p = F.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# State-bin instrumentation (ground-truth; NOT fed to the agent)
# ---------------------------------------------------------------------------


def _entity_positions(env: CausalGridWorldV2, name: str) -> List[Tuple[int, int]]:
    """Ground-truth positions of a salient entity type (list of (x, y))."""
    if name == "resource":
        src = getattr(env, "resources", None)
    elif name == "hazard":
        src = getattr(env, "hazards", None)
    elif name == "waypoint":
        src = getattr(env, "waypoints", None)
    else:
        src = None
    if not src:
        return []
    out: List[Tuple[int, int]] = []
    for item in src:
        try:
            out.append((int(item[0]), int(item[1])))
        except (TypeError, IndexError, ValueError):
            continue
    return out


def _relative_direction(ax: int, ay: int, ex: int, ey: int) -> str:
    """Quantise the direction from agent (ax, ay) to entity (ex, ey) to
    {N, S, E, W, ON}. On-top when same cell. Otherwise the axis of larger
    displacement wins (ties broken toward the x-axis / N-S)."""
    dx = ex - ax
    dy = ey - ay
    if dx == 0 and dy == 0:
        return "ON"
    # In this env grid[x, y]: x is the row index (vertical), y is the column (horizontal).
    if abs(dx) >= abs(dy):
        return "S" if dx > 0 else "N"
    return "E" if dy > 0 else "W"


def _state_bin(env: CausalGridWorldV2) -> str:
    """GROUND-TRUTH state bin = (nearest salient entity type) x (relative direction).
    Fallback 'empty/open' when no salient entity within SALIENT_RADIUS. Pure instrumentation
    read off env.grid / env.agent_x / env.agent_y -- NEVER fed to the agent."""
    ax = int(env.agent_x)
    ay = int(env.agent_y)
    best_name: Optional[str] = None
    best_dist = SALIENT_RADIUS + 1
    best_pos: Optional[Tuple[int, int]] = None
    for name in SALIENT_ENTITY_NAMES:
        for (ex, ey) in _entity_positions(env, name):
            d = abs(ex - ax) + abs(ey - ay)
            if d < best_dist:
                best_dist = d
                best_name = name
                best_pos = (ex, ey)
    if best_name is None or best_pos is None or best_dist > SALIENT_RADIUS:
        return "empty_open"
    rel = _relative_direction(ax, ay, best_pos[0], best_pos[1])
    return f"{best_name}:{rel}"


# ---------------------------------------------------------------------------
# Information-theoretic DVs
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


def _mutual_information_nats(states: List[str], actions: List[int]) -> float:
    """Plug-in estimate of I(state ; action) in nats from paired samples.
    I = H(action) - H(action|state) = sum p(s,a) log( p(s,a) / (p(s)p(a)) )."""
    n = len(states)
    if n == 0 or n != len(actions):
        return 0.0
    joint: Counter = Counter()
    ps: Counter = Counter()
    pa: Counter = Counter()
    for s, a in zip(states, actions):
        joint[(s, a)] += 1
        ps[s] += 1
        pa[a] += 1
    mi = 0.0
    for (s, a), c in joint.items():
        p_sa = c / n
        p_s = ps[s] / n
        p_a = pa[a] / n
        if p_sa > 0.0 and p_s > 0.0 and p_a > 0.0:
            mi += p_sa * math.log(p_sa / (p_s * p_a))
    return float(max(0.0, mi))


def _conditional_entropy_nats(states: List[str], actions: List[int]) -> float:
    """H(action | state) in nats."""
    n = len(states)
    if n == 0:
        return 0.0
    by_state: Dict[str, Counter] = {}
    state_counts: Counter = Counter()
    for s, a in zip(states, actions):
        by_state.setdefault(s, Counter())[a] += 1
        state_counts[s] += 1
    h = 0.0
    for s, ac in by_state.items():
        p_s = state_counts[s] / n
        h += p_s * _marginal_entropy_nats(dict(ac))
    return float(h)


def _shuffle_null_mi(
    states: List[str], actions: List[int], n_shuffle: int, rng: random.Random
) -> Tuple[float, float, List[float]]:
    """Permute the action labels across ticks (break state-action coupling) and recompute
    MI n_shuffle times. Returns (null_mean, null_p95, null_values)."""
    if not states or not actions:
        return 0.0, 0.0, []
    perm = list(actions)
    vals: List[float] = []
    for _ in range(max(1, n_shuffle)):
        rng.shuffle(perm)
        vals.append(_mutual_information_nats(states, perm))
    vals_sorted = sorted(vals)
    null_mean = float(sum(vals) / len(vals))
    idx95 = min(len(vals_sorted) - 1, int(math.ceil(0.95 * len(vals_sorted)) - 1))
    null_p95 = float(vals_sorted[max(0, idx95)])
    return null_mean, null_p95, vals


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------


def _run_seed(
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)
    env = _make_env(seed)
    agent = _make_agent(env)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = torch.optim.Adam(
        list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS
    )
    ofc_deval_opt = torch.optim.Adam(
        list(agent.ofc.devaluation_bias_head_parameters()), lr=LR_OFC_DEVAL
    )
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)
    shuffle_rng = random.Random(seed + 100003)

    total_train_eps = p0_episodes + p1_episodes + p2_episodes
    p1_start = p0_episodes
    p2_start = p0_episodes + p1_episodes
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p2_ticks = 0

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    # P2 logging accumulators.
    p2_states: List[str] = []
    p2_committed: List[int] = []
    committed_class_counts: Dict[int, int] = {}
    state_bin_counts: Counter = Counter()

    # Behavioural competence (per P2 episode).
    p2_ep_resources: List[int] = []
    p2_ep_hazard_hits: List[int] = []
    p2_ep_contaminations: List[int] = []
    p2_ep_rewards: List[float] = []
    n_p2_eps_completed = 0

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

        # Per-P2-episode competence counters.
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

            # P1 REINFORCE snapshot of candidate summaries (same source compute_bias uses).
            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            # P2: inject the trained OFC devaluation viability into the Go/No-Go gate
            # (matched to 714's P2 composition; keeps the all-ON stack fully engaged).
            if is_p2 and candidates and len(candidates) >= 2:
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
            elif is_p2:
                agent.set_injected_go_nogo_signals(None)

            # GROUND-TRUTH state bin BEFORE the action executes (instrumentation only).
            state_bin = _state_bin(env) if is_p2 else None

            action = agent.select_action(candidates, ticks)
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at seed={seed} phase={phase_label} "
                        f"ep={ep} step={_step}"
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
                p2_states.append(state_bin)
                p2_committed.append(committed_class)
                committed_class_counts[committed_class] = (
                    committed_class_counts.get(committed_class, 0) + 1
                )
                state_bin_counts[state_bin] += 1
            elif is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 ONLY (frozen in P1/P2 for stable measurement).
            if (not is_p1) and (not is_p2) and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )

            _, _harm_signal, done, info, obs_dict = env.step(action)
            harm_signal = float(_harm_signal)
            if is_p1:
                ep_reward += harm_signal

            # Behavioural competence readout (P2), from the env step() return `info`.
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

        # P1 end-of-episode: TWO-head REINFORCE.
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

        # P2 competence bookkeeping (per completed eval episode).
        if is_p2 and error_note is None:
            p2_ep_resources.append(ep_resources)
            p2_ep_hazard_hits.append(ep_hazard_hits)
            p2_ep_contaminations.append(ep_contaminations)
            p2_ep_rewards.append(ep_reward_signal)
            n_p2_eps_completed += 1

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] diagnostic seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

    # ----- Per-seed DV computation -----
    # DV1: marginal committed-class entropy.
    marginal_entropy = _marginal_entropy_nats(committed_class_counts)

    # DV2: mutual information + debias.
    raw_mi = _mutual_information_nats(p2_states, p2_committed)
    cond_entropy = _conditional_entropy_nats(p2_states, p2_committed)
    normalized_mi = float(raw_mi / marginal_entropy) if marginal_entropy > 0.0 else 0.0
    mi_null_mean, mi_null_p95, _ = _shuffle_null_mi(
        p2_states, p2_committed, N_SHUFFLE, shuffle_rng
    )
    mi_debiased = float(raw_mi - mi_null_mean)
    mi_above_null_p95 = bool(raw_mi > mi_null_p95)

    n_occupied_bins = int(len([b for b, c in state_bin_counts.items() if c > 0]))
    mi_estimable = bool(
        n_p2_ticks >= TOTAL_TICKS_FLOOR and n_occupied_bins >= MIN_OCCUPIED_BINS
    )

    # DV3: competence.
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
    competence_supra_random = bool(mean_resources_per_ep > COMPETENCE_RESOURCE_FLOOR)

    return {
        "seed": int(seed),
        "error_note": error_note,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p2_eps_completed": int(n_p2_eps_completed),
        # ----- DV1: marginal committed-class entropy (the campaign ceiling) -----
        "marginal_committed_class_entropy_nats": round(marginal_entropy, 6),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        # ----- DV2: mutual information -----
        "mutual_information_state_committed_nats": round(raw_mi, 6),
        "conditional_entropy_committed_given_state_nats": round(cond_entropy, 6),
        "normalized_mi": round(normalized_mi, 6),
        "mi_shuffle_null_mean": round(mi_null_mean, 6),
        "mi_shuffle_null_p95": round(mi_null_p95, 6),
        "mi_debiased": round(mi_debiased, 6),
        "mi_above_shuffle_p95": mi_above_null_p95,
        "n_shuffle": int(N_SHUFFLE),
        "n_occupied_state_bins": n_occupied_bins,
        "state_bin_counts": {str(k): int(v) for k, v in sorted(state_bin_counts.items())},
        "mi_estimable": mi_estimable,
        # ----- DV3: competence -----
        "mean_resources_per_episode": round(mean_resources_per_ep, 6),
        "mean_hazard_hits_per_episode": round(mean_hazard_hits_per_ep, 6),
        "mean_contaminations_per_episode": round(mean_contaminations_per_ep, 6),
        "mean_episode_reward": round(mean_episode_reward, 6),
        "competence_supra_random": competence_supra_random,
    }


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    per_seed_results: List[Dict[str, Any]] = []

    print(
        f"Diagnostic conversion-ceiling dissociation "
        f"(all-ON single config; P0={p0_episodes} ep e2-warmup, "
        f"P1={p1_episodes} ep two-head REINFORCE, P2={p2_episodes} ep eval/logging, "
        f"steps_per_episode={steps_per_episode}, dry_run={dry_run})",
        flush=True,
    )
    for s in seeds:
        print(f"Seed {s} Condition all_on_diagnostic", flush=True)
        row = _run_seed(s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode)
        per_seed_results.append(row)
        verdict = "PASS" if row["error_note"] is None else "FAIL"
        print(f"verdict: {verdict}", flush=True)

    ok_rows = [r for r in per_seed_results if r["error_note"] is None]

    # ----- READINESS (MI estimability) -----
    n_estimable = sum(1 for r in ok_rows if r["mi_estimable"])
    readiness_met = bool(n_estimable >= READINESS_MIN_SEEDS)

    # ----- Dissociation gates (only licensed when readiness is met) -----
    # Real, above-chance coupling: raw_mi > shuffle p95.
    n_real_coupling = sum(1 for r in ok_rows if r["mi_above_shuffle_p95"])
    real_coupling_gate = bool(n_real_coupling >= MI_REAL_MIN_SEEDS)
    # Debiased MI clears the floor on a majority of seeds.
    n_mi_debiased_supra = sum(
        1 for r in ok_rows if r["mi_debiased"] > MI_DEBIASED_FLOOR
    )
    mi_debiased_gate = bool(n_mi_debiased_supra >= MIN_SEEDS_FOR_PASS)
    # Competence supra-random on a majority of seeds.
    n_competent = sum(1 for r in ok_rows if r["competence_supra_random"])
    competence_gate = bool(n_competent >= COMPETENCE_MIN_SEEDS)

    state_appropriate = bool(mi_debiased_gate and real_coupling_gate and competence_gate)
    # Genuine collapse: debiased MI at/below the floor AND not real above-chance coupling.
    n_collapse = sum(
        1 for r in ok_rows
        if (r["mi_debiased"] <= MI_DEBIASED_FLOOR and not r["mi_above_shuffle_p95"])
    )
    genuine_collapse = bool(n_collapse >= MIN_SEEDS_FOR_PASS)

    # ----- Self-route (HYPOTHESIS, not a verdict) -----
    if not readiness_met:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif state_appropriate:
        outcome = "PASS"
        direction = "non_contributory"   # diagnostic; never weights the claims
        label = "decisive_state_appropriate_commitment"
    elif genuine_collapse:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "genuine_monomodal_collapse"
    else:
        # Estimable but mixed (neither branch's gates cleanly met).
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"

    evidence_direction_per_claim = {"MECH-309": direction, "ARC-062": direction}

    # Aggregate DV means for readability.
    agg_marginal = _mean([r["marginal_committed_class_entropy_nats"] for r in ok_rows])
    agg_raw_mi = _mean([r["mutual_information_state_committed_nats"] for r in ok_rows])
    agg_mi_debiased = _mean([r["mi_debiased"] for r in ok_rows])
    agg_normalized_mi = _mean([r["normalized_mi"] for r in ok_rows])
    agg_resources = _mean([r["mean_resources_per_episode"] for r in ok_rows])
    agg_hazards = _mean([r["mean_hazard_hits_per_episode"] for r in ok_rows])
    agg_contam = _mean([r["mean_contaminations_per_episode"] for r in ok_rows])
    agg_reward = _mean([r["mean_episode_reward"] for r in ok_rows])
    min_ticks = min([r["n_p2_ticks"] for r in ok_rows], default=0)
    min_bins = min([r["n_occupied_state_bins"] for r in ok_rows], default=0)

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "mi_estimable_sufficient_ticks",
                "kind": "readiness",
                "description": (
                    "Per-seed total P2 ticks >= TOTAL_TICKS_FLOOR so I(state;committed) is "
                    "estimable without finite-sample bias dominating. Below-floor => "
                    "substrate_not_ready_requeue, NOT a dissociation verdict."
                ),
                "control": "P2 eval window tick count per seed",
                "measured": float(min_ticks),
                "threshold": float(TOTAL_TICKS_FLOOR),
                "met": bool(
                    all(r["n_p2_ticks"] >= TOTAL_TICKS_FLOOR for r in ok_rows) if ok_rows
                    else False
                ),
            },
            {
                "name": "mi_estimable_enough_occupied_state_bins",
                "kind": "readiness",
                "description": (
                    "Per-seed distinct occupied ground-truth state-bins >= MIN_OCCUPIED_BINS "
                    "so the joint p(state, committed) has support to estimate MI. Below-floor "
                    "=> substrate_not_ready_requeue, NOT a dissociation verdict."
                ),
                "control": "occupied (nearest-salient x rel-direction) state-bins per seed",
                "measured": float(min_bins),
                "threshold": float(MIN_OCCUPIED_BINS),
                "met": bool(
                    all(r["n_occupied_state_bins"] >= MIN_OCCUPIED_BINS for r in ok_rows)
                    if ok_rows else False
                ),
            },
            {
                "name": "readiness_majority_seeds_mi_estimable",
                "kind": "readiness",
                "description": (
                    "MI is estimable (both floors above) on a majority of seeds; only then is "
                    "either dissociation branch licensed. Below => substrate_not_ready_requeue."
                ),
                "control": "count of MI-estimable seeds",
                "measured": float(n_estimable),
                "threshold": float(READINESS_MIN_SEEDS),
                "met": bool(readiness_met),
            },
        ],
        "criteria": [
            {
                "name": "state_drives_commitment_mi_debiased_and_above_null",
                "load_bearing": True,
                "passed": bool(state_appropriate),
            },
        ],
        "criteria_non_degenerate": {
            "readiness_mi_estimable": bool(readiness_met),
            "mi_debiased_supra_floor_majority": bool(mi_debiased_gate),
            "raw_mi_above_shuffle_p95_majority": bool(real_coupling_gate),
            "competence_supra_random_majority": bool(competence_gate),
            "genuine_collapse_majority": bool(genuine_collapse),
        },
    }

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": seeds,
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "total_seeds_attempted": int(len(seeds)),
        "total_seeds_completed": int(len(ok_rows)),
        "decision_rule_thresholds": {
            "n_shuffle": int(N_SHUFFLE),
            "mi_debiased_floor_nats": float(MI_DEBIASED_FLOOR),
            "mi_real_min_seeds": int(MI_REAL_MIN_SEEDS),
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "competence_min_seeds": int(COMPETENCE_MIN_SEEDS),
            "total_ticks_floor": int(TOTAL_TICKS_FLOOR),
            "min_occupied_bins": int(MIN_OCCUPIED_BINS),
            "readiness_min_seeds": int(READINESS_MIN_SEEDS),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "salient_radius": int(SALIENT_RADIUS),
            "salient_entity_names": list(SALIENT_ENTITY_NAMES),
            "rel_directions": list(REL_DIRECTIONS),
        },
        "dissociation_gates": {
            "readiness_met": readiness_met,
            "n_estimable_seeds": int(n_estimable),
            "mi_debiased_gate": mi_debiased_gate,
            "n_mi_debiased_supra_floor": int(n_mi_debiased_supra),
            "real_coupling_gate": real_coupling_gate,
            "n_raw_mi_above_shuffle_p95": int(n_real_coupling),
            "competence_gate": competence_gate,
            "n_competent_seeds": int(n_competent),
            "state_appropriate_branch": state_appropriate,
            "genuine_collapse_branch": genuine_collapse,
            "n_collapse_seeds": int(n_collapse),
        },
        "aggregate_dvs": {
            "marginal_committed_class_entropy_nats_mean": round(agg_marginal, 6),
            "mutual_information_state_committed_nats_mean": round(agg_raw_mi, 6),
            "mi_debiased_mean": round(agg_mi_debiased, 6),
            "normalized_mi_mean": round(agg_normalized_mi, 6),
            "mean_resources_per_episode_mean": round(agg_resources, 6),
            "mean_hazard_hits_per_episode_mean": round(agg_hazards, 6),
            "mean_contaminations_per_episode_mean": round(agg_contam, 6),
            "mean_episode_reward_mean": round(agg_reward, 6),
        },
        "interpretation_grid": {
            "decisive_state_appropriate_commitment": (
                "mi_debiased HIGH (state genuinely drives commitment; > floor on a majority "
                "of seeds AND raw_mi above the shuffle p95 on >= MI_REAL_MIN_SEEDS) AND "
                "competence supra-random. The agent commits to DIFFERENT, state-appropriate "
                "first-actions in DIFFERENT states -- the low MARGINAL committed_class entropy "
                "the campaign reads as a 'ceiling' is LARGELY a marginal-entropy ARTIFACT of a "
                "working state-conditioned policy. HYPOTHESIS (not a verdict): reframes "
                "MECH-309 / ARC-062 -- route to /failure-autopsy for adjudication before any "
                "governance action."
            ),
            "genuine_monomodal_collapse": (
                "mi_debiased ~0 (<= floor) AND raw_mi NOT above the shuffle p95 -- the "
                "committed first-action is statistically independent of world-state. The low "
                "marginal entropy is a TRUE collapse signature: the agent does the same thing "
                "regardless of state. HYPOTHESIS (not a verdict): supports MECH-309 as a real "
                "pathology and warrants the GAP-A-divergence-survival substrate build -- route "
                "to /failure-autopsy for adjudication."
            ),
            "substrate_not_ready_requeue": (
                "MI under-sampled (too few total P2 ticks OR too few occupied state-bins on a "
                "majority of seeds), OR the gates were mixed (neither branch cleanly met). "
                "NOT a dissociation verdict -- re-queue at a larger P2 budget / more seeds. Do "
                "NOT draw a conclusion about MECH-309 / ARC-062."
            ),
        },
        "per_seed_results": per_seed_results,
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
        "sleep_driver_pattern": "none",
        "evidence_direction_note": (
            f"V3-EXQ-719 CONVERSION-CEILING DISSOCIATION DIAGNOSTIC (MECH-309 / ARC-062; "
            f"experiment_purpose=diagnostic, EXCLUDED from governance scoring). Dissociates two "
            f"agent states the campaign DV committed_class_entropy CONFLATES: (A) genuine "
            f"monomodal policy collapse (same first-action regardless of state -- the real "
            f"MECH-309 pathology) vs (B) decisive state-appropriate commitment (different "
            f"actions in different states -- a WORKING agent a marginal-entropy metric wrongly "
            f"flags as collapsed). Marginal entropy cannot separate them; I(state;committed) "
            f"can. Single all-ON config (V3-EXQ-714 ARM_ON stack: SP-CEM + e2_world_forward "
            f"SD-056 + MECH-448 demotion + MECH-449 Go/No-Go + P3 OFC valuation + lateral_pfc "
            f"CRF bias, use_candidate_rule_field=True) over seeds {result['seeds']}; P0 e2 "
            f"warmup -> P1 two-head REINFORCE -> P2 long eval/logging. Per P2 tick logs a "
            f"GROUND-TRUTH state_bin (nearest-salient-entity-type x relative-direction, read "
            f"off env.grid/agent_x/agent_y -- NOT fed to the agent), the executed committed "
            f"first-action class, and behavioural competence (resources/hazards/contamination "
            f"per episode, mean reward). THREE DV families: (1) marginal committed-class "
            f"entropy (reproduces the ceiling); (2) I(state;committed) + H(committed|state) + "
            f"normalized_mi, DEBIASED with a {N_SHUFFLE}-permutation shuffle null "
            f"(mi_debiased = raw - null_mean; real coupling = raw > null_p95); (3) competence. "
            f"Pre-registered dissociation (self-routed label is a HYPOTHESIS, NOT a verdict): "
            f"mi_debiased > {MI_DEBIASED_FLOOR} nats AND raw>p95 on >= {MI_REAL_MIN_SEEDS}/3 AND "
            f"competence supra-random -> decisive_state_appropriate_commitment (ceiling is "
            f"largely a metric artifact; reframes MECH-309/ARC-062); mi_debiased ~0 AND not "
            f"real-coupled -> genuine_monomodal_collapse (warrants GAP-A-divergence-survival "
            f"substrate build); under-sampled/mixed -> substrate_not_ready_requeue. "
            f"interpretation_label={result['interpretation_label']}. readiness_met="
            f"{result['dissociation_gates']['readiness_met']}. PROMOTES / DEMOTES NOTHING by "
            f"itself; route to /failure-autopsy for adjudication."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "single_config": "all-mechanisms-ON (V3-EXQ-714 ARM_ON), use_candidate_rule_field=True",
            "arm_fingerprint_exempt": ARM_FINGERPRINT_EXEMPT,
            "state_bin_definition": (
                "GROUND-TRUTH nearest-salient-entity-type (resource|hazard|waypoint within "
                f"Manhattan radius {SALIENT_RADIUS}) x relative-direction {{N,S,E,W,ON}}; "
                "fallback 'empty_open' bin when no salient entity nearby. Instrumentation only "
                "(env.grid / env.agent_x / env.agent_y) -- NEVER fed to the agent."
            ),
            "committed_class_source": "int(agent.select_action(...).argmax()) -- executed committed first-action class",
            "behaviour_source": "env.step() info: transition_type (resource / env_caused_hazard / agent_caused_hazard) + contamination_delta; harm_signal reward",
            "phases": "P0 e2-warmup (field matures) -> P1 frozen-encoder TWO-head REINFORCE (lateral_pfc + OFC devaluation) -> P2 frozen long eval/logging w/ OFC viability injection",
            "matched_stack": (
                "SP-CEM + candidate_summary_source=e2_world_forward (SD-056/GAP-A) + "
                "use_modulatory_selection_authority (std) + channel routing + MECH-448 "
                "f_eligibility demotion + channel-adaptive floor + MECH-449 Go/No-Go + dACC "
                "perseveration + P3 OFC valuation (use_ofc_devaluation_head, viability injected "
                "P2) + MECH-341 stratified + MECH-313 noise floor + V_s minimal + "
                "use_gated_policy + use_lateral_pfc_analog (bias head trained P1) + "
                "use_candidate_rule_field + SD-056 all levers"
            ),
            "all_on_config_sourced_from": "V3-EXQ-714 ARM_ON (v3_exq_714_fullstack_selection_valuation_conversion_falsifier.py)",
            "alpha_world": 0.9,
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "use_sleep_loop": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-719 conversion-ceiling dissociation DIAGNOSTIC "
            "(marginal entropy vs I(state;committed); MECH-309/ARC-062)"
        )
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
        p2 = P2_EVAL_EPISODES
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
    dg = result["dissociation_gates"]
    ag = result["aggregate_dvs"]
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"label={result['interpretation_label']} "
        f"readiness={dg['readiness_met']}",
        flush=True,
    )
    print(
        f"DV1 marginal_committed_class_entropy_nats_mean="
        f"{ag['marginal_committed_class_entropy_nats_mean']} | "
        f"DV2 raw_mi_mean={ag['mutual_information_state_committed_nats_mean']} "
        f"mi_debiased_mean={ag['mi_debiased_mean']} "
        f"normalized_mi_mean={ag['normalized_mi_mean']} | "
        f"DV3 resources/ep={ag['mean_resources_per_episode_mean']} "
        f"hazards/ep={ag['mean_hazard_hits_per_episode_mean']} "
        f"contam/ep={ag['mean_contaminations_per_episode_mean']} "
        f"reward/ep={ag['mean_episode_reward_mean']}",
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
