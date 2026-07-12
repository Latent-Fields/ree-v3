#!/opt/local/bin/python3
"""
V3-EXQ-723 -- REE-NATIVE J-LENS DISPOSITIONAL-READOUT DIAGNOSTIC (Experiment A of SD-064).

PURPOSE (a DIAGNOSTIC, not an evidence falsifier -- the self-routed label below is a
HYPOTHESIS, not a verdict; experiment_purpose=diagnostic, EXCLUDED from governance scoring).
Motivated by Anthropic's J-space / Jacobian-lens result (2026,
anthropic.com/research/global-workspace): a capable model, under no design pressure, grew a
COMPACT (few dozen concepts, < 10% of internal activity), REPORTABLE, causally-central
broadcast bottleneck -- a Global Workspace. SD-064 asserts REE already instantiates such an
access channel. This experiment ports the J-lens as a READOUT to ask the existence question
directly, on the EXISTING all-ON substrate, with NO new mechanism (measurement only):

  Does REE have a "J-space" -- a COMPACT, ACTION-COUPLED subspace of the latent state that
  (a) predicts the agent's future committed action above chance, and (b) occupies a small
  fraction of total latent activity (Anthropic's < 10%)?

DESIGN. A SINGLE all-mechanisms-ON configuration (V3-EXQ-714 ARM_ON stack,
use_candidate_rule_field=True -- identical to V3-EXQ-719a, the harness this adapts) in
CausalGridWorldV2 over SEEDS = [42, 43, 44]. Phases:
  P0 (encoder / e2 world-forward warmup; SD-056 online contrastive) -- mirrors 719a/714 P0=200.
  P1 (frozen-encoder TWO-head REINFORCE on lateral_pfc bias + OFC devaluation) -- mirrors P1=90.
  P2 (TICK-BUDGETED eval / logging; all frozen). Per P2 tick we log:
     * z_t -- the FULL latent state vector this tick: concat of the available latent streams
       (z_world, z_self, z_harm, z_harm_a), detached. This is the REE analogue of the residual
       stream the J-lens reads.
     * committed_class_t -- the EXECUTED first-action class (int(select_action(...).argmax())),
       the same committed proxy V3-EXQ-714/719a log.
     * a best-effort BROADCAST signal (MECH-287 hippocampal broadcast-queue depth / SD-037
       override magnitude if exposed by the substrate) as a "report" proxy; recorded only when
       the running all-ON config actually exposes it (else broadcast_alignment_available=false).

THE READOUT (post-P2, per seed; the "REE J-lens"). REE has an explicit action head, so the
dispositional map is z_t -> committed_class_{t+H}: which directions in latent state "vote" for
future committed action. For each HORIZON H in {1, 3, 5} we form within-episode pairs
(z_t, committed_class_{t+H}), standardize z on a temporal train split (first 70%), fit a
ridge-regularized LINEAR dispositional map W (D x K) by least-squares on one-hot targets
(W = (Z'Z + lambda I)^-1 Z'Y -- deterministic, no iterative fit), and compute on the held-out
tail (last 30%):
  * balanced_accuracy_full -- mean per-class recall of argmax(Z W); chance = majority-class
    recall baseline. LABEL-PERMUTATION NULL: permute held-out labels N_PERM=200 times, recompute
    balanced accuracy of the FIXED predictions -> null_mean, null_p95; above_null = acc > p95.
  * J-SPACE via SVD of W: W = U S V'. The left-singular vectors U (latent-space directions) are
    the J-space basis. jspace_dim = # singular directions capturing >= JSPACE_ENERGY (0.90) of
    W's action-predictive energy. jspace_activity_fraction = variance of z projected onto those
    dims / total z variance (Anthropic's < 10% test -> JSPACE_FRACTION_CEIL=0.10).
  * predictive_retention = balanced_accuracy(z reconstructed from the J-space only) /
    balanced_accuracy_full -- does the COMPACT subspace carry the predictive power?
  * H=1 (reactive/immediate) vs H=5 (integrative/future) compactness is reported as an
    automatic-vs-integrative contrast (GWT predicts integrative cognition concentrates in the
    workspace while immediate reactive responses are more distributed) -- reported, not gated.

PRE-REGISTERED SELF-ROUTE (HYPOTHESIS the pipeline can falsify, NOT a verdict), at PRIMARY_H=3:
  * READINESS fails (too few P2 ticks / too few usable train pairs / < 2 committed classes, OR
    the linear readout has NO above-null signal on a majority of seeds -- an undertrained-substrate
    signature, NOT a scientific null) -> label `substrate_not_ready_requeue`.
  * action predictable above null AND jspace_activity_fraction < 0.10 AND predictive_retention
    >= 0.80, each on a majority of seeds -> label `compact_action_coupled_subspace_present`
    (POSITIVE J-space existence -- raises the SD-064 prior; motivates the SD-027 V3 boundary-gate
    retrofit that unlocks the Experiment B ablation-cliff falsifier).
  * action predictable above null BUT the action-predictive structure is DIFFUSE (fraction >= 0.10
    OR retention < 0.80 on a majority) -> label `no_compact_workspace_diffuse` (evidence toward the
    SD-027-original pluralist / no-single-workspace reading).

EVIDENCE-FOR / EVIDENCE-AGAINST (diagnostic-description requirement):
  * EVIDENCE FOR a global workspace (supports SD-064): a compact (< 10% activity), action-coupled
    subspace that retains the predictive power -- the J-space analogue exists in REE.
  * EVIDENCE FOR the pluralist reading (SD-027 original hedge): action is predictable but the
    predictive structure is diffuse / high-dimensional -- no compact broadcast bottleneck.
  * EVIDENCE AGAINST either conclusion (self-route substrate_not_ready_requeue, draw NO verdict):
    under-sampled P2, too few classes, or no above-null predictive signal (a linear readout that
    cannot predict the agent's own next action is an undertraining signature, not a workspace null).
  The self-routed label is a HYPOTHESIS for /failure-autopsy adjudication; this experiment
  PROMOTES / DEMOTES NOTHING by itself.

NOTE ON claim_ids: [] (a diagnostic; excluded from scoring). SD-064 (the global-workspace claim)
and MECH-191 (cross-architecture signal legibility -- the dispositional readout is a candidate
unblock for its tonic-channel problem) are referenced for CONTEXT only; tagging them as claim_ids
would (wrongly) route diagnostic context into governance confidence. Experiment B (workspace-ablation
cliff) is SUBSTRATE-BLOCKED (SD-027 boundary gate not built in V3) and is NOT queued here.

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Single all-ON config (no OFF/treatment arm grid) -> no arm_results key; module-level
ARM_FINGERPRINT_EXEMPT set.

See REE_assembly/evidence/planning/global_workspace_jlens_plan.md (design; section 2 = this
experiment), REE_assembly/docs/claims/claims.yaml (SD-064),
experiments/v3_exq_719a_conversion_ceiling_dissociation_diagnostic.py (harness this adapts),
experiments/v3_exq_714_fullstack_selection_valuation_conversion_falsifier.py (all-ON config source),
ree_core/agent.py (select_action -> executed committed action class; latent streams).
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
import torch.nn.functional as F

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import reset_all_rng
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_723_jlens_dispositional_readout_diagnostic"
QUEUE_ID = "V3-EXQ-723"
CLAIM_IDS: List[str] = []           # diagnostic; SD-064 / MECH-191 referenced for CONTEXT only
EXPERIMENT_PURPOSE = "diagnostic"

# Single all-ON config (no OFF/treatment arm grid); no arm_results key is written.
ARM_FINGERPRINT_EXEMPT = "single-config diagnostic; no OFF/treatment arm grid"

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 200         # encoder / e2 warmup (mirrors 719a/714 P0)
P1_BIAS_TRAIN_EPISODES = 90      # frozen-encoder TWO-head REINFORCE (mirrors 719a/714 P1)
STEPS_PER_EPISODE = 200

# P2 is TICK-BUDGET-DRIVEN (mirrors 719a): run P2 episodes until P2_TICKS_TARGET committed P2
# ticks are collected OR a cap of P2_MAX_EPISODES is reached, whichever first. ~6000 ticks
# gives ~4200 train / ~1800 test pairs per horizon -- ample for a ~112-dim ridge map.
P2_TICKS_TARGET = 6000
P2_MAX_EPISODES = 400

DRY_RUN_SEEDS = [42, 43]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2_TICKS_TARGET = 120    # small but > readout minimums so the readout path executes
DRY_RUN_P2_MAX_EPISODES = 4
DRY_RUN_STEPS = 60

# ---------------------------------------------------------------------------
# Readout horizons
# ---------------------------------------------------------------------------
HORIZONS: Tuple[int, ...] = (1, 3, 5)   # committed_class_{t+H}
PRIMARY_H = 3                            # the horizon the self-route reads

# ---------------------------------------------------------------------------
# Pre-registered thresholds (constants; NOT derived from the run's own statistics)
# ---------------------------------------------------------------------------
RIDGE_LAMBDA = 1.0               # L2 on standardized features for the linear dispositional map
TRAIN_FRAC = 0.70                # temporal split: first 70% train, last 30% held-out
N_PERM = 200                     # label-permutation null for balanced-accuracy significance

JSPACE_ENERGY = 0.90             # cumulative singular-value energy defining jspace_dim
JSPACE_FRACTION_CEIL = 0.10      # Anthropic's "< 10% of internal activity" compactness test
PRED_RETENTION_FLOOR = 0.80      # jspace-only balanced-acc must retain >= 80% of full-state acc

# Readiness (readout estimability). Below any floor -> substrate_not_ready_requeue.
TOTAL_TICKS_FLOOR = 3000         # per seed P2 ticks (mirrors 719a; guards a truncated run)
TRAIN_PAIRS_FLOOR = 1000         # per seed usable (z_t, class_{t+H}) train pairs at PRIMARY_H
MIN_COMMITTED_CLASSES = 2        # need >= 2 committed classes to fit a classifier
READINESS_MIN_SEEDS = 2          # of 3 seeds must be readout-estimable to license a branch

MAJORITY_MIN_SEEDS = 2           # of 3 (generic majority for the gates)

# ---------------------------------------------------------------------------
# All-ON matched-stack constants (sourced from V3-EXQ-714 ARM_ON; identical to 719a)
# ---------------------------------------------------------------------------
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

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

OFC_STATE_DIM = 16
OFC_HARM_DIM = 32
OFC_BIAS_SCALE = 0.5
OFC_DEVAL_BIAS_SCALE = 2.0
LR_OFC_DEVAL = 2e-3
GNG_VIABILITY_FLOOR = 0.1

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


# Identical env to V3-EXQ-714 / 719a.
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
    """All-mechanisms-ON matched-stack agent (V3-EXQ-714 ARM_ON, use_candidate_rule_field=True);
    identical to the V3-EXQ-719a config this harness adapts."""
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
        use_candidate_rule_field=True,
    )
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
    bl = bias_low.detach().reshape(-1)
    if bl.numel() < 2:
        return None
    rng = float((bl.max() - bl.min()).item())
    if rng < 1e-6:
        return None
    bln = (bl - bl.min()) / (bl.max() - bl.min())
    return (1.0 - bln).detach()


# ---------------------------------------------------------------------------
# P1 two-head REINFORCE (mirror V3-EXQ-719a)
# ---------------------------------------------------------------------------


def _lpfc_reinforce_loss(agent, outcome_buf, baseline, device) -> torch.Tensor:
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


def _ofc_deval_reinforce_loss(agent, outcome_buf, baseline, device) -> torch.Tensor:
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
# Latent-state assembly (the REE "residual stream" the J-lens reads)
# ---------------------------------------------------------------------------


def _assemble_z(latent) -> Optional[np.ndarray]:
    """Concat the available latent streams (z_world, z_self, z_harm, z_harm_a) into one 1-D
    numpy vector. Streams that are absent/None/non-finite are skipped; the SET of present
    streams is fixed across a run (checked once at capture-order time), so the assembled
    vector has consistent dimensionality within a seed."""
    parts: List[np.ndarray] = []
    for name in ("z_world", "z_self", "z_harm", "z_harm_a"):
        t = getattr(latent, name, None)
        if t is None:
            continue
        try:
            v = t.detach().reshape(-1).float().cpu().numpy()
        except Exception:
            continue
        if v.size == 0 or not np.all(np.isfinite(v)):
            continue
        parts.append(v.astype(np.float64))
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def _broadcast_signal(agent: REEAgent) -> Optional[float]:
    """Best-effort scalar 'report' proxy from the broadcast substrate (MECH-287 hippocampal
    broadcast-queue depth or SD-037 override magnitude), IF the running all-ON config exposes
    it. Returns None when no broadcast channel is accessible (broadcast_alignment then reported
    unavailable). Purely read-only; never mutates the agent."""
    # MECH-287: hippocampal broadcast-event queue depth.
    hippo = getattr(agent, "hippocampal", None) or getattr(agent, "hippocampus", None)
    if hippo is not None:
        q = getattr(hippo, "_broadcast_event_queue", None)
        if q is not None:
            try:
                return float(len(q))
            except Exception:
                pass
    # SD-037: broadcast-override signal magnitude.
    override = getattr(agent, "broadcast_override", None)
    if override is not None:
        sig = getattr(override, "last_override_signal", None)
        if sig is not None:
            try:
                return float(abs(float(sig)))
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# J-lens readout math (deterministic; numpy)
# ---------------------------------------------------------------------------


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> float:
    """Mean per-class recall (robust to class imbalance)."""
    recalls: List[float] = []
    for c in classes:
        mask = (y_true == c)
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        recalls.append(float((y_pred[mask] == c).sum()) / n_c)
    return float(np.mean(recalls)) if recalls else 0.0


def _fit_ridge_onehot(Z: np.ndarray, y: np.ndarray, classes: np.ndarray, lam: float) -> np.ndarray:
    """Ridge least-squares linear map on one-hot targets: W = (Z'Z + lam I)^-1 Z'Y. (D, K)."""
    n, d = Z.shape
    k = classes.shape[0]
    Y = np.zeros((n, k), dtype=np.float64)
    cls_to_col = {int(c): j for j, c in enumerate(classes)}
    for i in range(n):
        Y[i, cls_to_col[int(y[i])]] = 1.0
    ztz = Z.T @ Z + lam * np.eye(d, dtype=np.float64)
    W = np.linalg.solve(ztz, Z.T @ Y)
    return W


def _predict(Z: np.ndarray, W: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return classes[np.argmax(Z @ W, axis=1)]


def _jlens_readout_one_horizon(
    z_all: np.ndarray,
    y_all: np.ndarray,
    lam: float,
    n_perm: int,
    perm_rng: np.random.Generator,
) -> Dict[str, Any]:
    """Fit z_t -> committed_class_{t+H}, evaluate on a temporal held-out tail, and characterise
    the J-space (SVD of the weight matrix). All deterministic given the caller's RNG."""
    n = z_all.shape[0]
    n_train = int(math.floor(TRAIN_FRAC * n))
    z_tr, z_te = z_all[:n_train], z_all[n_train:]
    y_tr, y_te = y_all[:n_train], y_all[n_train:]

    # Standardize on train stats.
    mu = z_tr.mean(axis=0, keepdims=True)
    sd = z_tr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z_tr_s = (z_tr - mu) / sd
    z_te_s = (z_te - mu) / sd

    classes = np.array(sorted(set(int(c) for c in y_tr)), dtype=np.int64)
    n_classes_train = int(classes.shape[0])
    # Test labels must be within the trained class set to be predictable.
    te_mask = np.isin(y_te, classes)
    z_te_s = z_te_s[te_mask]
    y_te = y_te[te_mask]

    result: Dict[str, Any] = {
        "n_pairs_total": int(n),
        "n_train": int(n_train),
        "n_test": int(z_te_s.shape[0]),
        "n_classes_train": n_classes_train,
        "readout_estimable": False,
    }
    if n_classes_train < MIN_COMMITTED_CLASSES or z_te_s.shape[0] < 20 or z_tr_s.shape[0] < 20:
        return result

    W = _fit_ridge_onehot(z_tr_s, y_tr, classes, lam)

    # Full-state predictive balanced accuracy + majority-class baseline.
    y_pred_full = _predict(z_te_s, W, classes)
    bal_acc_full = _balanced_accuracy(y_te, y_pred_full, classes)
    # Chance = balanced accuracy of always predicting the train-majority class.
    train_counts = np.array([(y_tr == c).sum() for c in classes])
    majority_cls = classes[int(np.argmax(train_counts))]
    y_pred_majority = np.full_like(y_te, fill_value=majority_cls)
    bal_acc_majority = _balanced_accuracy(y_te, y_pred_majority, classes)

    # Label-permutation null: permute held-out labels, recompute balanced acc of FIXED preds.
    null_vals = np.empty(max(1, n_perm), dtype=np.float64)
    y_te_perm = y_te.copy()
    for i in range(max(1, n_perm)):
        perm_rng.shuffle(y_te_perm)
        null_vals[i] = _balanced_accuracy(y_te_perm, y_pred_full, classes)
    null_mean = float(null_vals.mean())
    null_p95 = float(np.percentile(null_vals, 95))
    above_null = bool(bal_acc_full > null_p95)

    # J-space: SVD of the weight matrix W (D, K). Left-singular vectors U are latent directions.
    # jspace_dim = # singular dirs for JSPACE_ENERGY of the action-predictive energy (sv^2).
    U, S, _Vt = np.linalg.svd(W, full_matrices=False)
    sv_energy = S ** 2
    total_energy = float(sv_energy.sum())
    if total_energy <= 0.0:
        return {**result, "readout_estimable": True, "bal_acc_full": round(bal_acc_full, 6),
                "degenerate_map": True}
    cum = np.cumsum(sv_energy) / total_energy
    jspace_dim = int(np.searchsorted(cum, JSPACE_ENERGY) + 1)
    jspace_dim = max(1, min(jspace_dim, U.shape[1]))
    B = U[:, :jspace_dim]                      # (D, jspace_dim) latent-space J-space basis

    # jspace_activity_fraction: fraction of TOTAL latent-activity variance in the J-space.
    # Use standardized held-out activity (each dim ~ unit variance -> total variance ~ D).
    proj = z_te_s @ B                          # (n_test, jspace_dim)
    var_proj = float(np.sum(np.var(proj, axis=0)))
    var_total = float(np.sum(np.var(z_te_s, axis=0)))
    jspace_activity_fraction = float(var_proj / var_total) if var_total > 0 else 1.0

    # Predictive retention: balanced acc using ONLY the J-space projection of z.
    z_te_j = (z_te_s @ B) @ B.T                 # reconstruct from J-space only
    y_pred_j = _predict(z_te_j, W, classes)
    bal_acc_jspace = _balanced_accuracy(y_te, y_pred_j, classes)
    predictive_retention = float(bal_acc_jspace / bal_acc_full) if bal_acc_full > 1e-9 else 0.0

    result.update({
        "readout_estimable": True,
        "latent_dim": int(z_all.shape[1]),
        "bal_acc_full": round(bal_acc_full, 6),
        "bal_acc_majority_baseline": round(bal_acc_majority, 6),
        "null_mean": round(null_mean, 6),
        "null_p95": round(null_p95, 6),
        "above_null": above_null,
        "jspace_dim": jspace_dim,
        "jspace_activity_fraction": round(jspace_activity_fraction, 6),
        "jspace_compact": bool(jspace_activity_fraction < JSPACE_FRACTION_CEIL),
        "bal_acc_jspace": round(bal_acc_jspace, 6),
        "predictive_retention": round(predictive_retention, 6),
        "retention_ok": bool(predictive_retention >= PRED_RETENTION_FLOOR),
        "degenerate_map": False,
    })
    return result


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------


def _run_seed(
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    p2_ticks_target: int,
    p2_max_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)
    env = _make_env(seed)
    agent = _make_agent(env)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = torch.optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS)
    ofc_deval_opt = torch.optim.Adam(
        list(agent.ofc.devaluation_bias_head_parameters()), lr=LR_OFC_DEVAL
    )
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
        maxlen=TRANSITION_BUFFER_MAX
    )
    sample_rng = random.Random(seed)
    perm_rng = np.random.default_rng(seed + 90007)

    total_train_eps = p0_episodes + p1_episodes + p2_max_episodes
    p1_start = p0_episodes
    p2_start = p0_episodes + p1_episodes
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p2_ticks = 0

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    # P2 per-episode logs: each entry is (z_seq [list of np vectors], class_seq, broadcast_seq).
    p2_episode_logs: List[Tuple[List[np.ndarray], List[int], List[Optional[float]]]] = []
    committed_class_counts: Dict[int, int] = {}
    n_broadcast_available = 0

    # Behavioural competence (per P2 episode).
    p2_ep_resources: List[int] = []
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

        # Per-P2-episode J-lens logs + competence.
        ep_z_seq: List[np.ndarray] = []
        ep_class_seq: List[int] = []
        ep_broadcast_seq: List[Optional[float]] = []
        ep_resources = 0
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
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            if is_p2 and candidates and len(candidates) >= 2:
                ofc_summ_p2 = _consumed_summaries(agent, candidates)
                viability_sig: Optional[torch.Tensor] = None
                if ofc_summ_p2 is not None and torch.isfinite(ofc_summ_p2).all():
                    with torch.no_grad():
                        deval_bias_p2 = agent.ofc.compute_devaluation_bias(ofc_summ_p2).detach()
                    viability_sig = _build_viability_nogo(deval_bias_p2)
                if viability_sig is not None:
                    agent.set_injected_go_nogo_signals(
                        {"viability": viability_sig.to(agent.device)}
                    )
                else:
                    agent.set_injected_go_nogo_signals(None)
            elif is_p2:
                agent.set_injected_go_nogo_signals(None)

            # J-lens capture: FULL latent vector this tick, BEFORE the action executes.
            z_vec = _assemble_z(latent) if is_p2 else None
            bcast = _broadcast_signal(agent) if is_p2 else None

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
                committed_class_counts[committed_class] = (
                    committed_class_counts.get(committed_class, 0) + 1
                )
                if z_vec is not None:
                    ep_z_seq.append(z_vec)
                    ep_class_seq.append(committed_class)
                    ep_broadcast_seq.append(bcast)
                    if bcast is not None:
                        n_broadcast_available += 1
            elif is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 ONLY (frozen in P1/P2 for a stable measurement substrate).
            if (not is_p1) and (not is_p2) and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer, optimiser=e2_opt, rng=sample_rng
                )

            _, _harm_signal, done, info, obs_dict = env.step(action)
            harm_signal = float(_harm_signal)
            if is_p1:
                ep_reward += harm_signal

            if is_p2:
                ep_reward_signal += harm_signal
                if str(info.get("transition_type", "none")) == "resource":
                    ep_resources += 1

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

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            tick_in_ep += 1
            if done:
                break

        # P1 end-of-episode: TWO-head REINFORCE.
        if is_p1:
            reinforce_baseline = EMA_DECAY * reinforce_baseline + (1.0 - EMA_DECAY) * ep_reward
            for cand_features, sel in ep_buf:
                outcome_buf.append((cand_features, sel, ep_reward))
            if len(outcome_buf) > OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
            l_loss = _lpfc_reinforce_loss(agent, outcome_buf, reinforce_baseline, agent.device)
            if l_loss.requires_grad:
                bias_opt.zero_grad()
                l_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.lateral_pfc.bias_head_parameters(), 1.0)
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

        # P2 per-episode bookkeeping.
        if is_p2 and error_note is None:
            if len(ep_class_seq) >= 1:
                p2_episode_logs.append((ep_z_seq, ep_class_seq, ep_broadcast_seq))
            p2_ep_resources.append(ep_resources)
            p2_ep_rewards.append(ep_reward_signal)
            n_p2_eps_completed += 1

        # Progress print (phase-relative; runner parses '[train] ... ep N/M').
        if phase_label == "P2":
            phase_ep = ep - p2_start + 1
            phase_total = p2_max_episodes
        elif phase_label == "P1":
            phase_ep = ep - p1_start + 1
            phase_total = p1_episodes
        else:
            phase_ep = ep + 1
            phase_total = p0_episodes
        if phase_ep % 10 == 0 or phase_ep == phase_total or (ep + 1) == total_train_eps:
            print(
                f"  [train] jlens seed={seed} phase={phase_label} ep {phase_ep}/{phase_total}",
                flush=True,
            )

        if error_note is not None:
            break

        # P2 tick-budget termination.
        if is_p2 and n_p2_ticks >= p2_ticks_target:
            break

    # ----- Per-seed J-lens readout -----
    marginal_classes = sorted(committed_class_counts.keys())
    per_horizon: Dict[str, Any] = {}
    for h in HORIZONS:
        # Form within-episode (z_t, class_{t+H}) pairs.
        z_pairs: List[np.ndarray] = []
        y_pairs: List[int] = []
        for (z_seq, c_seq, _b_seq) in p2_episode_logs:
            m = len(c_seq)
            for i in range(m - h):
                z_pairs.append(z_seq[i])
                y_pairs.append(int(c_seq[i + h]))
        if len(z_pairs) >= 40:
            z_all = np.vstack(z_pairs)
            y_all = np.asarray(y_pairs, dtype=np.int64)
            per_horizon[str(h)] = _jlens_readout_one_horizon(
                z_all, y_all, RIDGE_LAMBDA, N_PERM, np.random.default_rng(seed + 700 + h)
            )
        else:
            per_horizon[str(h)] = {
                "n_pairs_total": int(len(z_pairs)), "readout_estimable": False
            }

    prim = per_horizon.get(str(PRIMARY_H), {"readout_estimable": False})
    n_train_pairs_primary = int(prim.get("n_train", 0))

    # Readiness: enough P2 ticks, enough primary-H train pairs, >= 2 committed classes,
    # AND the primary-H readout is estimable.
    readout_estimable_primary = bool(prim.get("readout_estimable", False))
    seed_readiness = bool(
        n_p2_ticks >= TOTAL_TICKS_FLOOR
        and n_train_pairs_primary >= TRAIN_PAIRS_FLOOR
        and len(marginal_classes) >= MIN_COMMITTED_CLASSES
        and readout_estimable_primary
    )

    # Primary-H J-space signature (only meaningful when estimable).
    prim_above_null = bool(prim.get("above_null", False))
    prim_compact = bool(prim.get("jspace_compact", False))
    prim_retention_ok = bool(prim.get("retention_ok", False))

    return {
        "seed": int(seed),
        "error_note": error_note,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p2_eps_completed": int(n_p2_eps_completed),
        "n_committed_classes": int(len(marginal_classes)),
        "committed_class_counts": {str(k): int(v) for k, v in sorted(committed_class_counts.items())},
        "broadcast_available_ticks": int(n_broadcast_available),
        "broadcast_alignment_available": bool(n_broadcast_available > 0),
        "n_train_pairs_primary": n_train_pairs_primary,
        "seed_readiness": seed_readiness,
        # Primary-H headline signature.
        "primary_h": int(PRIMARY_H),
        "primary_above_null": prim_above_null,
        "primary_jspace_compact": prim_compact,
        "primary_retention_ok": prim_retention_ok,
        "primary_bal_acc_full": prim.get("bal_acc_full"),
        "primary_null_p95": prim.get("null_p95"),
        "primary_jspace_dim": prim.get("jspace_dim"),
        "primary_jspace_activity_fraction": prim.get("jspace_activity_fraction"),
        "primary_predictive_retention": prim.get("predictive_retention"),
        "primary_latent_dim": prim.get("latent_dim"),
        "per_horizon": per_horizon,
        # Competence sanity.
        "mean_resources_per_episode": round(
            float(sum(p2_ep_resources) / len(p2_ep_resources)) if p2_ep_resources else 0.0, 6
        ),
        "mean_episode_reward": round(
            float(sum(p2_ep_rewards) / len(p2_ep_rewards)) if p2_ep_rewards else 0.0, 6
        ),
    }


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    p2_ticks_target: int,
    p2_max_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    per_seed_results: List[Dict[str, Any]] = []

    print(
        f"J-lens dispositional-readout diagnostic (all-ON single config; "
        f"P0={p0_episodes} e2-warmup, P1={p1_episodes} two-head REINFORCE, "
        f"P2 tick-budgeted (target={p2_ticks_target} / cap={p2_max_episodes} ep), "
        f"horizons={list(HORIZONS)} primary_H={PRIMARY_H}, "
        f"steps_per_episode={steps_per_episode}, dry_run={dry_run})",
        flush=True,
    )
    for s in seeds:
        print(f"Seed {s} Condition all_on_jlens", flush=True)
        row = _run_seed(
            s, p0_episodes, p1_episodes, p2_ticks_target, p2_max_episodes, steps_per_episode
        )
        per_seed_results.append(row)
        verdict = "PASS" if row["error_note"] is None else "FAIL"
        print(f"verdict: {verdict}", flush=True)

    ok_rows = [r for r in per_seed_results if r["error_note"] is None]

    # ----- READINESS -----
    n_ready = sum(1 for r in ok_rows if r["seed_readiness"])
    readiness_met = bool(n_ready >= READINESS_MIN_SEEDS)

    # ----- J-space signature gates (majority of seeds; only licensed when readiness met) -----
    n_above_null = sum(1 for r in ok_rows if r["primary_above_null"])
    signal_gate = bool(n_above_null >= MAJORITY_MIN_SEEDS)
    n_compact = sum(1 for r in ok_rows if r["primary_jspace_compact"])
    compact_gate = bool(n_compact >= MAJORITY_MIN_SEEDS)
    n_retention = sum(1 for r in ok_rows if r["primary_retention_ok"])
    retention_gate = bool(n_retention >= MAJORITY_MIN_SEEDS)

    jspace_present = bool(signal_gate and compact_gate and retention_gate)
    # Diffuse: there IS an above-null predictive signal, but it is not compact/retained.
    diffuse = bool(signal_gate and not (compact_gate and retention_gate))

    # ----- Self-route (HYPOTHESIS, not a verdict) -----
    if not readiness_met or not signal_gate:
        # No sampling / no above-null signal -> cannot assess compactness (undertraining signature).
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif jspace_present:
        outcome = "PASS"
        direction = "non_contributory"   # diagnostic; never weights any claim
        label = "compact_action_coupled_subspace_present"
    elif diffuse:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "no_compact_workspace_diffuse"
    else:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"

    # Aggregate headline DVs.
    def _agg(key: str) -> float:
        vals = [r[key] for r in ok_rows if isinstance(r.get(key), (int, float))]
        return _mean(vals)

    agg_bal_acc = _agg("primary_bal_acc_full")
    agg_null_p95 = _agg("primary_null_p95")
    agg_jspace_dim = _agg("primary_jspace_dim")
    agg_jspace_frac = _agg("primary_jspace_activity_fraction")
    agg_retention = _agg("primary_predictive_retention")
    agg_resources = _agg("mean_resources_per_episode")
    latent_dims = [r.get("primary_latent_dim") for r in ok_rows if r.get("primary_latent_dim")]
    latent_dim = int(latent_dims[0]) if latent_dims else None
    min_ticks = min([r["n_p2_ticks"] for r in ok_rows], default=0)
    min_train_pairs = min([r["n_train_pairs_primary"] for r in ok_rows], default=0)
    min_classes = min([r["n_committed_classes"] for r in ok_rows], default=0)
    any_broadcast = any(r["broadcast_alignment_available"] for r in ok_rows)

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "readout_sufficient_p2_ticks",
                "kind": "readiness",
                "description": (
                    "Per-seed P2 ticks >= TOTAL_TICKS_FLOOR so the dispositional map is "
                    "estimable. Below-floor => substrate_not_ready_requeue, NOT a workspace verdict."
                ),
                "control": "P2 eval window tick count per seed",
                "measured": float(min_ticks),
                "threshold": float(TOTAL_TICKS_FLOOR),
                "met": bool(all(r["n_p2_ticks"] >= TOTAL_TICKS_FLOOR for r in ok_rows) if ok_rows else False),
            },
            {
                "name": "readout_sufficient_train_pairs_primary_h",
                "kind": "readiness",
                "description": (
                    "Per-seed usable (z_t, committed_class_{t+H}) TRAIN pairs at PRIMARY_H >= "
                    "TRAIN_PAIRS_FLOOR so the ~112-dim ridge map is well-posed. Below-floor => "
                    "substrate_not_ready_requeue."
                ),
                "control": "within-episode training pairs at primary horizon",
                "measured": float(min_train_pairs),
                "threshold": float(TRAIN_PAIRS_FLOOR),
                "met": bool(all(r["n_train_pairs_primary"] >= TRAIN_PAIRS_FLOOR for r in ok_rows) if ok_rows else False),
            },
            {
                "name": "at_least_two_committed_classes",
                "kind": "readiness",
                "description": (
                    "Per-seed distinct committed classes >= MIN_COMMITTED_CLASSES; a classifier "
                    "needs >= 2 target classes. Below => substrate_not_ready_requeue."
                ),
                "control": "distinct executed committed-action classes per seed",
                "measured": float(min_classes),
                "threshold": float(MIN_COMMITTED_CLASSES),
                "met": bool(all(r["n_committed_classes"] >= MIN_COMMITTED_CLASSES for r in ok_rows) if ok_rows else False),
            },
            {
                "name": "action_predictable_above_null_majority",
                "kind": "readiness",
                "description": (
                    "The linear readout predicts committed_class_{t+H=PRIMARY} above the "
                    "label-permutation p95 on a majority of seeds. Absent this signal, compactness "
                    "is not assessable (an undertrained-substrate signature, NOT a workspace null) "
                    "=> substrate_not_ready_requeue."
                ),
                "control": "count of seeds with bal_acc_full > null_p95 at PRIMARY_H",
                "measured": float(n_above_null),
                "threshold": float(MAJORITY_MIN_SEEDS),
                "met": bool(signal_gate),
            },
            {
                "name": "readiness_majority_seeds_estimable",
                "kind": "readiness",
                "description": (
                    "Readout is estimable (all floors above) on a majority of seeds; only then is "
                    "a present/diffuse branch licensed. Below => substrate_not_ready_requeue."
                ),
                "control": "count of readout-estimable seeds",
                "measured": float(n_ready),
                "threshold": float(READINESS_MIN_SEEDS),
                "met": bool(readiness_met),
            },
        ],
        "criteria": [
            {
                "name": "compact_action_coupled_subspace_present",
                "load_bearing": True,
                "passed": bool(jspace_present),
            },
        ],
        "criteria_non_degenerate": {
            "readiness_estimable": bool(readiness_met),
            "action_predictable_above_null_majority": bool(signal_gate),
            "jspace_compact_majority": bool(compact_gate),
            "predictive_retention_majority": bool(retention_gate),
            "diffuse_branch": bool(diffuse),
        },
    }

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": seeds,
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_ticks_target": int(p2_ticks_target),
        "p2_max_episodes": int(p2_max_episodes),
        "steps_per_episode": int(steps_per_episode),
        "total_seeds_attempted": int(len(seeds)),
        "total_seeds_completed": int(len(ok_rows)),
        "latent_dim": latent_dim,
        "broadcast_alignment_available_any_seed": bool(any_broadcast),
        "decision_rule_thresholds": {
            "horizons": list(HORIZONS),
            "primary_h": int(PRIMARY_H),
            "ridge_lambda": float(RIDGE_LAMBDA),
            "train_frac": float(TRAIN_FRAC),
            "n_perm": int(N_PERM),
            "jspace_energy": float(JSPACE_ENERGY),
            "jspace_fraction_ceil": float(JSPACE_FRACTION_CEIL),
            "pred_retention_floor": float(PRED_RETENTION_FLOOR),
            "total_ticks_floor": int(TOTAL_TICKS_FLOOR),
            "train_pairs_floor": int(TRAIN_PAIRS_FLOOR),
            "min_committed_classes": int(MIN_COMMITTED_CLASSES),
            "readiness_min_seeds": int(READINESS_MIN_SEEDS),
            "majority_min_seeds": int(MAJORITY_MIN_SEEDS),
        },
        "signature_gates": {
            "readiness_met": readiness_met,
            "n_ready_seeds": int(n_ready),
            "signal_gate_above_null": signal_gate,
            "n_above_null_seeds": int(n_above_null),
            "compact_gate": compact_gate,
            "n_compact_seeds": int(n_compact),
            "retention_gate": retention_gate,
            "n_retention_seeds": int(n_retention),
            "jspace_present_branch": jspace_present,
            "diffuse_branch": diffuse,
        },
        "aggregate_dvs": {
            "primary_bal_acc_full_mean": round(agg_bal_acc, 6),
            "primary_null_p95_mean": round(agg_null_p95, 6),
            "primary_jspace_dim_mean": round(agg_jspace_dim, 6),
            "primary_jspace_activity_fraction_mean": round(agg_jspace_frac, 6),
            "primary_predictive_retention_mean": round(agg_retention, 6),
            "mean_resources_per_episode_mean": round(agg_resources, 6),
            "latent_dim": latent_dim,
        },
        "interpretation_grid": {
            "compact_action_coupled_subspace_present": (
                "action predictable above the label-permutation null AND jspace_activity_fraction "
                "< 0.10 AND predictive_retention >= 0.80, each on a majority of seeds. REE has a "
                "COMPACT, action-coupled, causally-central subspace -- the J-space analogue. "
                "HYPOTHESIS (not a verdict): raises the SD-064 (global-workspace) prior and "
                "motivates the SD-027 V3 boundary-gate retrofit that unlocks the Experiment B "
                "ablation-cliff falsifier. Route to /failure-autopsy for adjudication."
            ),
            "no_compact_workspace_diffuse": (
                "action IS predictable above null, but the action-predictive structure is DIFFUSE "
                "(jspace_activity_fraction >= 0.10 OR predictive_retention < 0.80 on a majority). "
                "HYPOTHESIS (not a verdict): evidence toward the SD-027-original pluralist / "
                "no-single-workspace reading -- no compact broadcast bottleneck. Route to "
                "/failure-autopsy."
            ),
            "substrate_not_ready_requeue": (
                "under-sampled P2 / too few train pairs / < 2 committed classes, OR NO above-null "
                "predictive signal on a majority of seeds (a linear readout that cannot predict the "
                "agent's own next action is an undertraining signature, NOT a workspace null). NOT "
                "a verdict -- re-queue at a larger P2 budget / more training. Draw NO conclusion "
                "about SD-064."
            ),
        },
        "per_seed_results": per_seed_results,
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
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
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "sleep_driver_pattern": "none",
        "evidence_direction_note": (
            f"V3-EXQ-723 REE-NATIVE J-LENS DISPOSITIONAL-READOUT DIAGNOSTIC (Experiment A of "
            f"SD-064; experiment_purpose=diagnostic, EXCLUDED from governance scoring; claim_ids=[] "
            f"-- SD-064 global-workspace claim + MECH-191 signal-legibility referenced for CONTEXT "
            f"only). Ports Anthropic's J-lens (2026) as a READOUT on the EXISTING all-ON substrate "
            f"(V3-EXQ-714 ARM_ON, identical to V3-EXQ-719a) -- NO new mechanism, measurement only. "
            f"Question: does REE have a J-space -- a COMPACT, action-coupled subspace of the latent "
            f"state that predicts the future committed action above chance while occupying < 10% of "
            f"latent activity? Per P2 tick logs the full latent vector z_t (concat z_world/z_self/"
            f"z_harm/z_harm_a) + executed committed_class + a best-effort broadcast 'report' proxy. "
            f"Readout (per seed, horizons {list(HORIZONS)}, primary_H={PRIMARY_H}): ridge linear map "
            f"z_t -> committed_class_{{t+H}} on a temporal train split; held-out balanced accuracy "
            f"vs a {N_PERM}-permutation label null; J-space via SVD of the weight matrix "
            f"(jspace_dim @ {JSPACE_ENERGY} energy; jspace_activity_fraction vs the {JSPACE_FRACTION_CEIL} "
            f"compactness ceiling; predictive_retention vs the {PRED_RETENTION_FLOOR} floor). "
            f"Pre-registered self-route (HYPOTHESIS, NOT a verdict): predictable-above-null + "
            f"fraction < 0.10 + retention >= 0.80 (majority of seeds) -> "
            f"compact_action_coupled_subspace_present (raises SD-064 prior; motivates the SD-027 V3 "
            f"gate retrofit unlocking Experiment B); predictable but diffuse -> "
            f"no_compact_workspace_diffuse (pluralist reading); under-sampled / no above-null signal "
            f"-> substrate_not_ready_requeue. Experiment B (workspace-ablation cliff) is "
            f"SUBSTRATE-BLOCKED (SD-027 boundary gate not built in V3) and is NOT queued. "
            f"interpretation_label={result['interpretation_label']}; readiness_met="
            f"{result['signature_gates']['readiness_met']}. PROMOTES / DEMOTES NOTHING; route to "
            f"/failure-autopsy for adjudication. Design: "
            f"REE_assembly/evidence/planning/global_workspace_jlens_plan.md section 2."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "single_config": "all-mechanisms-ON (V3-EXQ-714 ARM_ON), use_candidate_rule_field=True (identical to 719a)",
            "arm_fingerprint_exempt": ARM_FINGERPRINT_EXEMPT,
            "latent_vector": "concat of available streams z_world/z_self/z_harm/z_harm_a (detached), the REE 'residual stream'",
            "committed_class_source": "int(agent.select_action(...).argmax()) -- executed committed first-action class",
            "readout": (
                "ridge linear dispositional map z_t -> committed_class_{t+H} (W=(Z'Z+lambda I)^-1 Z'Y); "
                "temporal 70/30 split; balanced-accuracy vs label-permutation null; J-space = SVD of W; "
                "jspace_activity_fraction vs 0.10; predictive_retention vs 0.80"
            ),
            "phases": "P0 e2-warmup -> P1 frozen-encoder TWO-head REINFORCE (lateral_pfc + OFC devaluation) -> P2 frozen TICK-BUDGETED eval/logging",
            "all_on_config_sourced_from": "V3-EXQ-714 ARM_ON via V3-EXQ-719a",
            "alpha_world": 0.9,
            "use_sleep_loop": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-723 REE-native J-lens dispositional-readout DIAGNOSTIC "
            "(J-space existence on the all-ON substrate; SD-064 Experiment A)"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        p2_ticks = DRY_RUN_P2_TICKS_TARGET
        p2_max_eps = DRY_RUN_P2_MAX_EPISODES
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1 = P1_BIAS_TRAIN_EPISODES
        p2_ticks = P2_TICKS_TARGET
        p2_max_eps = P2_MAX_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_episodes=p1,
        p2_ticks_target=p2_ticks,
        p2_max_episodes=p2_max_eps,
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
    sg = result["signature_gates"]
    ag = result["aggregate_dvs"]
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"label={result['interpretation_label']} readiness={sg['readiness_met']} "
        f"latent_dim={result['latent_dim']} broadcast_proxy={result['broadcast_alignment_available_any_seed']}",
        flush=True,
    )
    print(
        f"J-lens@H={PRIMARY_H}: bal_acc_full_mean={ag['primary_bal_acc_full_mean']} "
        f"null_p95_mean={ag['primary_null_p95_mean']} "
        f"jspace_dim_mean={ag['primary_jspace_dim_mean']} "
        f"jspace_activity_fraction_mean={ag['primary_jspace_activity_fraction_mean']} "
        f"predictive_retention_mean={ag['primary_predictive_retention_mean']} | "
        f"present={sg['jspace_present_branch']} diffuse={sg['diffuse_branch']}",
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
