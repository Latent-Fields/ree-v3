#!/opt/local/bin/python3
"""
V3-EXQ-723a -- REE-NATIVE J-LENS DISCRIMINATIVE-COMPACTNESS DIAGNOSTIC (Experiment A of SD-064).

SUPERSEDES V3-EXQ-723 (same scientific question; implementation fix to the discrimination
criteria). experiment_purpose=diagnostic, EXCLUDED from governance scoring; the self-routed
label is a HYPOTHESIS for /failure-autopsy, NOT a verdict; PROMOTES / DEMOTES NOTHING.

WHY THIS ITERATION EXISTS (see failure_autopsy_V3-EXQ-723_2026-07-09). V3-EXQ-723 self-routed
`compact_action_coupled_subspace_present` but was adjudicated INCONCLUSIVE: its two
discrimination gates were NON-DISCRIMINATIVE BY CONSTRUCTION.
  * `jspace_activity_fraction < 0.10` passed at 0.0003 (350x under ceiling) -- a ridge-map row
    space is a tiny fraction of a 112-dim latent for ANY weak signal, so the raw ceiling is
    essentially always cleared (and is confounded by ridge amplification of low-variance
    directions). NON-DISCRIMINATIVE.
  * `predictive_retention >= 0.80` passed at ~1.0 -- the J-space IS W's predictive directions,
    so it retains ~all of W's (weak) predictivity BY CONSTRUCTION. NON-DISCRIMINATIVE.
The only real signal was WEAK above-null decodability (bal_acc 0.35-0.50 vs null_p95 0.21-0.51),
and seed 43 degenerated to ~2 effective committed classes ({1:2699,2:3398,3:1}).

QUESTION (UNCHANGED): Does REE have a J-space -- a COMPACT, action-coupled subspace of the
latent state that (a) predicts the future committed action above chance and (b) concentrates
that predictive power in a SMALL fraction of the latent, MORE than a matched-null baseline
would -- the REE analogue of Anthropic's Global-Workspace finding (2026)?

THE THREE FIXES (this is the redesign):

  FIX 1 -- DISCRIMINATIVE COMPACTNESS vs a MATCHED-RANK / MATCHED-COUNT NULL BASELINE.
    Instead of the raw fraction-vs-0.10 test, we ask whether the action-predictive structure is
    concentrated MORE than chance, via TWO matched-null contrasts, and we report the CONTRAST
    (not the raw fraction):
      (1a) CONCENTRATION CURVE (the LOAD-BEARING test). Rank latent dims by their influence on
           the ridge map W (row-norm ||W[i,:]||). frac_dims_90_top = fraction of latent dims
           (top-influence first) needed to recover >= 90% of the ABOVE-CHANCE balanced accuracy.
           Matched null = the SAME curve under RANDOM dim orderings (frac_dims_90_random). A
           DIFFUSE map (uniform influence) gives frac_top ~ frac_random (ratio ~ 1); a COMPACT
           map gives frac_top << frac_random. Discriminative gate:
           concentration_ratio = frac_top / frac_random_mean < CONCENTRATION_RATIO_CEIL
           AND frac_top < CONCENTRATION_ABS_CEIL. This is NOT tautological: the ratio is ~1 for a
           diffuse map, so a weak map does NOT trivially pass.
      (1b) MATCHED-RANK RANDOM-SUBSPACE ACTIVITY CONTRAST (corroborating). For the SVD J-space of
           rank k, compare jspace_activity_fraction f_J against the distribution of activity
           fractions of N random ORTHONORMAL k-frames (matched rank). Report the contrast; gate
           f_J < random_activity_frac_p5 (J-space occupies LESS activity than a random same-rank
           subspace). (Corroborating only -- ridge low-variance amplification can make f_J tiny
           for the wrong reason, so this is NEVER the load-bearing gate.)

  FIX 2 -- GATE (not just report) the plan's POSITIVE-MODE contrasts that 723 left "reported,
    not gated" (global_workspace_jlens_plan.md section 2 DVs):
      (2a) REACTIVE-BYPASS (H1-vs-H5). GWT predicts integrative cognition concentrates in the
           workspace while immediate reactive responses bypass it. Gate: concentration at the
           INTEGRATIVE horizon H=5 is TIGHTER (smaller frac_dims_90_top) than at the REACTIVE
           horizon H=1 by >= REACTIVE_BYPASS_MARGIN.
      (2b) BROADCAST-ALIGNMENT (reportability proxy; MECH-287 broadcast-queue depth / SD-037).
           REE has no language report head, so broadcast is the report analogue. Gate: per-tick
           J-space occupancy correlates with the broadcast signal above a permutation null.
           If the substrate exposes no VARYING broadcast signal, the contrast is NOT assessable
           and the CLEAN positive is withheld (routed to `..._broadcast_unverified`, which does
           NOT greenlight the SD-027 build) rather than silently passing.

  FIX 3 -- COMMITTED-ACTION-BALANCED EVAL so no seed degenerates to ~2 classes (cf. 723 seed 43).
      * Restrict the classification to EFFECTIVE classes (>= MIN_PER_CLASS_TRAIN train and
        >= MIN_PER_CLASS_TEST test instances); require >= MIN_EFFECTIVE_CLASSES effective classes
        for a seed to be readout-estimable (else substrate_not_ready_requeue).
      * CLASS-BALANCED (inverse-frequency-weighted) ridge so the map is not dominated by one
        committed class.
      * Longer P2 tick budget (9000 vs 723's 6000) for denser per-class coverage.

DESIGN (substrate UNCHANGED from 723). A SINGLE all-mechanisms-ON configuration (V3-EXQ-714
ARM_ON stack, use_candidate_rule_field=True -- identical to V3-EXQ-719a/723) in CausalGridWorldV2
over SEEDS = [42, 43, 44, 45]. Phases: P0 (encoder / e2 world-forward warmup; SD-056 online
contrastive); P1 (frozen-encoder TWO-head REINFORCE on lateral_pfc bias + OFC devaluation);
P2 (TICK-BUDGETED eval / logging; all frozen). Per P2 tick we log z_t (concat z_world / z_self /
z_harm / z_harm_a, detached -- the REE "residual stream"), the executed committed_class, and a
best-effort broadcast "report" proxy.

PRE-REGISTERED SELF-ROUTE (HYPOTHESIS the pipeline can falsify, NOT a verdict), at PRIMARY_H=3,
majority of seeds unless noted:
  * READINESS fails (too few P2 ticks / train pairs / effective classes, NO above-null signal,
    signal margin below floor, OR the random-ordering control is itself degenerate so the
    concentration contrast is vacuous) -> `substrate_not_ready_requeue` (NOT a workspace verdict).
  * above-null + concentration-discriminative (LOAD-BEARING) + reactive-bypass, AND broadcast
    assessable + aligned -> `compact_action_coupled_subspace_present` (CLEAN positive: raises the
    SD-064 prior; GREENLIGHTS the SD-027 V3 boundary-gate retrofit that unlocks the Experiment B
    ablation-cliff falsifier). The matched-rank random-subspace ACTIVITY contrast is REPORTED +
    corroborating (the autopsy's "and/or" second leg), NOT required -- the SVD-activity fraction is
    confounded by ridge low-variance amplification, so it never gates the verdict.
  * same but broadcast NOT assessable (no varying broadcast signal) ->
    `compact_subspace_present_broadcast_unverified` (present but reportability UNVERIFIED; does
    NOT greenlight; /failure-autopsy decides).
  * same but broadcast assessable and NOT aligned ->
    `compact_but_not_broadcast_reportable` (compact + integrative but not reportable; GWT-incomplete;
    does NOT greenlight).
  * above-null signal present BUT concentration-discriminative (load-bearing) FAILS ->
    `no_compact_workspace_diffuse` (evidence toward the SD-027-original pluralist / no-single-
    workspace reading).

EVIDENCE-FOR / EVIDENCE-AGAINST (diagnostic-description requirement):
  * EVIDENCE FOR a global workspace (supports SD-064): action is predictable AND its predictive
    structure is concentrated MORE than a matched null (concentration_ratio << 1), the compact
    subspace occupies less activity than a random same-rank subspace, integrative (H=5) cognition
    concentrates more than reactive (H=1), and J-space occupancy tracks the broadcast substrate.
  * EVIDENCE FOR the pluralist reading (SD-027 original hedge): action is predictable but the
    predictive structure is DIFFUSE (concentration_ratio ~ 1) -- no compact broadcast bottleneck.
  * EVIDENCE AGAINST either conclusion (self-route substrate_not_ready_requeue, draw NO verdict):
    under-sampled P2, too few effective classes, no above-null signal, insufficient signal margin,
    or a degenerate random-ordering control.
  The self-routed label is a HYPOTHESIS for /failure-autopsy adjudication.

NOTE ON claim_ids: [] (a diagnostic; excluded from scoring). SD-064 (the global-workspace claim)
and MECH-191 (cross-architecture signal legibility) are referenced for CONTEXT only. Experiment B
(workspace-ablation cliff) is SUBSTRATE-BLOCKED (SD-027 boundary gate not built in V3) and is NOT
queued here; a CLEAN positive from 723a is its build gate, a negative saves the build.

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Single all-ON config (no OFF/treatment arm grid) -> no arm_results key; module-level
ARM_FINGERPRINT_EXEMPT set.

See REE_assembly/evidence/planning/global_workspace_jlens_plan.md (design; section 2 = this
experiment), REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-723_2026-07-09.md (why this
redesign), REE_assembly/docs/claims/claims.yaml (SD-064),
experiments/v3_exq_723_jlens_dispositional_readout_diagnostic.py (the superseded predecessor),
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


EXPERIMENT_TYPE = "v3_exq_723a_jlens_discriminative_compactness_diagnostic"
QUEUE_ID = "V3-EXQ-723a"
SUPERSEDES = "V3-EXQ-723"
CLAIM_IDS: List[str] = []           # diagnostic; SD-064 / MECH-191 referenced for CONTEXT only
EXPERIMENT_PURPOSE = "diagnostic"

# Single all-ON config (no OFF/treatment arm grid); no arm_results key is written.
ARM_FINGERPRINT_EXEMPT = "single-config diagnostic; no OFF/treatment arm grid"

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45]          # 4 seeds (723 used 3; +1 for a firmer majority)
P0_WARMUP_EPISODES = 200          # encoder / e2 warmup (mirrors 719a/714/723 P0)
P1_BIAS_TRAIN_EPISODES = 90       # frozen-encoder TWO-head REINFORCE (mirrors 719a/714/723 P1)
STEPS_PER_EPISODE = 200

# P2 is TICK-BUDGET-DRIVEN: run P2 episodes until P2_TICKS_TARGET committed P2 ticks are collected
# OR a cap of P2_MAX_EPISODES is reached, whichever first. 9000 ticks (vs 723's 6000) gives ~6300
# train / ~2700 test pairs per horizon -- denser per-class coverage for the balanced multi-class
# ridge + the concentration curves.
P2_TICKS_TARGET = 9000
P2_MAX_EPISODES = 600

DRY_RUN_SEEDS = [42, 43]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2_TICKS_TARGET = 200     # small but > readout minimums so the readout path executes
DRY_RUN_P2_MAX_EPISODES = 6
DRY_RUN_STEPS = 60

# ---------------------------------------------------------------------------
# Readout horizons
# ---------------------------------------------------------------------------
HORIZONS: Tuple[int, ...] = (1, 3, 5)   # committed_class_{t+H}
PRIMARY_H = 3                            # the horizon the headline self-route reads
REACTIVE_H = 1                           # immediate / automatic
INTEGRATIVE_H = 5                        # future / integrative

# ---------------------------------------------------------------------------
# Pre-registered thresholds (constants; NOT derived from the run's own statistics)
# ---------------------------------------------------------------------------
RIDGE_LAMBDA = 1.0               # L2 on standardized features for the linear dispositional map
TRAIN_FRAC = 0.70                # temporal split: first 70% train, last 30% held-out
N_PERM = 200                     # label-permutation null for balanced-accuracy significance

# FIX 1a -- concentration curve (LOAD-BEARING discriminative compactness).
CONCENTRATION_ENERGY_FRAC = 0.90       # recover >= 90% of the ABOVE-CHANCE balanced accuracy
CONCENTRATION_GRID = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)  # geometric dim-count grid (capped at D)
N_RANDOM_ORDERINGS = 48                 # random dim orderings for the matched-count null
CONCENTRATION_RATIO_CEIL = 0.50         # frac_top / frac_random_mean must be < this to be compact
CONCENTRATION_ABS_CEIL = 0.15           # AND frac_top itself < this (absolute compactness)
CONCENTRATION_CONTROL_FLOOR = 0.25      # random ordering must need >= 25% of dims (else contrast vacuous)

# FIX 1b -- matched-rank random-subspace activity contrast (corroborating).
JSPACE_ENERGY = 0.90             # cumulative singular-value energy defining jspace_dim (as 723)
N_RANDOM_SUBSPACES = 120         # random orthonormal k-frames for the matched-rank activity null

# FIX 2a -- reactive-bypass (H1-vs-H5) gate.
REACTIVE_BYPASS_MARGIN = 0.05    # frac_dims_90_top(H1) - frac_dims_90_top(H5) >= this

# FIX 2b -- broadcast-alignment (reportability proxy) gate.
BROADCAST_ALIGN_MIN_ABS_CORR = 0.05   # |corr(J-space occupancy, broadcast)| must clear its null p95
BROADCAST_MIN_VARYING_TICKS = 50      # need this many varying broadcast samples to assess alignment

# Signal adequacy (so "90% of predictive power" is a real, non-degenerate target).
SIGNAL_MARGIN_FLOOR = 0.03       # bal_acc_full - null_p95 must clear this

# Readiness (readout estimability). Below any floor -> substrate_not_ready_requeue.
TOTAL_TICKS_FLOOR = 4500          # per seed P2 ticks (raised from 723's 3000 with the bigger budget)
TRAIN_PAIRS_FLOOR = 1500          # per seed usable (z_t, class_{t+H}) train pairs at PRIMARY_H
# FIX 3 -- committed-action-balanced eval floors.
MIN_PER_CLASS_TRAIN = 25          # a committed class needs this many TRAIN instances to be "effective"
MIN_PER_CLASS_TEST = 8            # ... and this many TEST instances
MIN_EFFECTIVE_CLASSES = 3         # need >= 3 effective (well-populated) classes (guards 723 seed-43)
READINESS_MIN_SEEDS = 2           # of 4 seeds must be readout-estimable to license a branch

MAJORITY_MIN_SEEDS = 2            # of 4 (generic majority for the gates; >= half of estimable seeds)

# ---------------------------------------------------------------------------
# All-ON matched-stack constants (sourced from V3-EXQ-714 ARM_ON; identical to 719a/723)
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


# Identical env to V3-EXQ-714 / 719a / 723.
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
    identical to the V3-EXQ-719a/723 config this harness adapts."""
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
# SD-056 online e2 training (mirror V3-EXQ-719a/723)
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
# obs helpers (mirror V3-EXQ-719a/723)
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
# P1 two-head REINFORCE (mirror V3-EXQ-719a/723)
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
    streams is fixed across a run, so the assembled vector has consistent dimensionality
    within a seed."""
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
    broadcast-event queue depth or SD-037 override magnitude), IF the running all-ON config
    exposes it. Returns None when no broadcast channel is accessible. Read-only; never mutates
    the agent."""
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


def _fit_ridge_onehot_weighted(
    Z: np.ndarray, y: np.ndarray, classes: np.ndarray, lam: float, sample_w: np.ndarray
) -> np.ndarray:
    """Class-BALANCED (inverse-frequency-weighted) ridge least-squares linear map on one-hot
    targets: W = (Z' diag(w) Z + lam I)^-1 Z' diag(w) Y. (D, K). The weights stop one dominant
    committed class from dictating the map (FIX 3)."""
    n, d = Z.shape
    k = classes.shape[0]
    Y = np.zeros((n, k), dtype=np.float64)
    cls_to_col = {int(c): j for j, c in enumerate(classes)}
    for i in range(n):
        Y[i, cls_to_col[int(y[i])]] = 1.0
    w = sample_w.reshape(-1, 1)
    Zw = Z * w                                   # (n, d) row-weighted
    ztz = Z.T @ Zw + lam * np.eye(d, dtype=np.float64)
    zty = Zw.T @ Y
    W = np.linalg.solve(ztz, zty)
    return W


def _predict(Z: np.ndarray, W: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return classes[np.argmax(Z @ W, axis=1)]


def _frac_dims_for_target(
    z_te_s: np.ndarray,
    W: np.ndarray,
    ordering: np.ndarray,
    classes: np.ndarray,
    y_te: np.ndarray,
    bal_acc_majority: float,
    target_abs: float,
    grid: Tuple[int, ...],
) -> float:
    """Incrementally include latent dims in the given ORDERING; return the fraction of dims
    (relative to D) at which the ABOVE-CHANCE balanced accuracy first recovers target_abs.
    Uses cumulative logits so the cost is one z@W-equivalent per ordering. Returns 1.0 if the
    target is never met (needs all dims -> diffuse)."""
    n_test = z_te_s.shape[0]
    d = z_te_s.shape[1]
    logits = np.zeros((n_test, W.shape[1]), dtype=np.float64)
    grid_pts = [g for g in grid if g < d]
    grid_pts.append(d)
    prev = 0
    for m in grid_pts:
        idx = ordering[prev:m]
        if idx.size:
            logits += z_te_s[:, idx] @ W[idx, :]
        prev = m
        y_pred = classes[np.argmax(logits, axis=1)]
        ba = _balanced_accuracy(y_te, y_pred, classes)
        if (ba - bal_acc_majority) >= target_abs:
            return float(m) / float(d)
    return 1.0


def _jlens_readout_one_horizon(
    z_all: np.ndarray,
    y_all: np.ndarray,
    lam: float,
    n_perm: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Fit z_t -> committed_class_{t+H} on a temporal train split (class-BALANCED weighted ridge
    over EFFECTIVE classes), evaluate on the held-out tail, and characterise compactness with the
    DISCRIMINATIVE contrasts (concentration curve vs random ordering; matched-rank random-subspace
    activity). All deterministic given the caller's RNG."""
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

    # FIX 3 -- restrict to EFFECTIVE committed classes (well-populated in BOTH train and test).
    all_classes = sorted(set(int(c) for c in y_tr) | set(int(c) for c in y_te))
    effective: List[int] = []
    for c in all_classes:
        n_tr_c = int((y_tr == c).sum())
        n_te_c = int((y_te == c).sum())
        if n_tr_c >= MIN_PER_CLASS_TRAIN and n_te_c >= MIN_PER_CLASS_TEST:
            effective.append(c)
    classes = np.array(sorted(effective), dtype=np.int64)
    n_effective_classes = int(classes.shape[0])

    result: Dict[str, Any] = {
        "n_pairs_total": int(n),
        "n_train": int(n_train),
        "n_test": int(z_te.shape[0]),
        "n_effective_classes": n_effective_classes,
        "readout_estimable": False,
    }
    if n_effective_classes < MIN_EFFECTIVE_CLASSES:
        result["reason"] = "too_few_effective_classes"
        return result

    # Keep only effective-class rows in train and test (balanced eval).
    tr_mask = np.isin(y_tr, classes)
    te_mask = np.isin(y_te, classes)
    z_tr_s, y_tr = z_tr_s[tr_mask], y_tr[tr_mask]
    z_te_s, y_te = z_te_s[te_mask], y_te[te_mask]
    if z_tr_s.shape[0] < 40 or z_te_s.shape[0] < 40:
        result["reason"] = "too_few_effective_rows"
        return result

    # Inverse-frequency class weights on train (FIX 3).
    train_counts = {int(c): int((y_tr == c).sum()) for c in classes}
    inv = {c: (1.0 / max(1, train_counts[c])) for c in train_counts}
    norm = sum(inv.values())
    class_w = {c: (inv[c] / norm * len(classes)) for c in inv}   # mean weight ~ 1
    sample_w = np.array([class_w[int(v)] for v in y_tr], dtype=np.float64)

    W = _fit_ridge_onehot_weighted(z_tr_s, y_tr, classes, lam, sample_w)

    # Full-state predictive balanced accuracy + majority-class baseline.
    y_pred_full = _predict(z_te_s, W, classes)
    bal_acc_full = _balanced_accuracy(y_te, y_pred_full, classes)
    counts_arr = np.array([train_counts[int(c)] for c in classes])
    majority_cls = classes[int(np.argmax(counts_arr))]
    y_pred_majority = np.full_like(y_te, fill_value=majority_cls)
    bal_acc_majority = _balanced_accuracy(y_te, y_pred_majority, classes)

    # Label-permutation null: permute held-out labels, recompute balanced acc of FIXED preds.
    null_vals = np.empty(max(1, n_perm), dtype=np.float64)
    y_te_perm = y_te.copy()
    for i in range(max(1, n_perm)):
        rng.shuffle(y_te_perm)
        null_vals[i] = _balanced_accuracy(y_te_perm, y_pred_full, classes)
    null_mean = float(null_vals.mean())
    null_p95 = float(np.percentile(null_vals, 95))
    above_null = bool(bal_acc_full > null_p95)
    signal_margin = float(bal_acc_full - null_p95)

    d = z_te_s.shape[1]

    # ---- FIX 1a: CONCENTRATION CURVE (load-bearing discriminative compactness) ----
    # Above-chance predictive power to recover.
    target_abs = CONCENTRATION_ENERGY_FRAC * max(0.0, bal_acc_full - bal_acc_majority)
    # Top-influence ordering = latent dims by descending row-norm of W.
    row_norm = np.linalg.norm(W, axis=1)
    top_ordering = np.argsort(-row_norm)
    frac_top = _frac_dims_for_target(
        z_te_s, W, top_ordering, classes, y_te, bal_acc_majority, target_abs, CONCENTRATION_GRID
    )
    # Matched-count random-ordering null.
    rand_fracs: List[float] = []
    for _ in range(N_RANDOM_ORDERINGS):
        perm = rng.permutation(d)
        rand_fracs.append(
            _frac_dims_for_target(
                z_te_s, W, perm, classes, y_te, bal_acc_majority, target_abs, CONCENTRATION_GRID
            )
        )
    rand_fracs_arr = np.array(rand_fracs, dtype=np.float64)
    random_frac_mean = float(rand_fracs_arr.mean())
    random_frac_p5 = float(np.percentile(rand_fracs_arr, 5))
    concentration_ratio = float(frac_top / random_frac_mean) if random_frac_mean > 1e-9 else 1.0
    # Discriminative compactness gate + its non-degenerate control.
    concentration_control_ok = bool(random_frac_mean >= CONCENTRATION_CONTROL_FLOOR)
    concentration_discriminative = bool(
        (target_abs > 1e-9)
        and concentration_control_ok
        and (concentration_ratio < CONCENTRATION_RATIO_CEIL)
        and (frac_top < CONCENTRATION_ABS_CEIL)
    )

    # ---- FIX 1b: MATCHED-RANK RANDOM-SUBSPACE ACTIVITY CONTRAST (corroborating) ----
    U, S, _Vt = np.linalg.svd(W, full_matrices=False)
    sv_energy = S ** 2
    total_energy = float(sv_energy.sum())
    if total_energy <= 0.0:
        return {**result, "readout_estimable": True, "degenerate_map": True,
                "bal_acc_full": round(bal_acc_full, 6), "above_null": above_null,
                "signal_margin": round(signal_margin, 6)}
    cum = np.cumsum(sv_energy) / total_energy
    jspace_dim = int(np.searchsorted(cum, JSPACE_ENERGY) + 1)
    jspace_dim = max(1, min(jspace_dim, U.shape[1]))
    B = U[:, :jspace_dim]                          # (D, k) J-space basis
    var_total = float(np.sum(np.var(z_te_s, axis=0)))
    f_jspace = float(np.sum(np.var(z_te_s @ B, axis=0)) / var_total) if var_total > 0 else 1.0
    # Random orthonormal k-frames (matched rank).
    rand_act: List[float] = []
    for _ in range(N_RANDOM_SUBSPACES):
        G = rng.standard_normal((d, jspace_dim))
        Q, _R = np.linalg.qr(G)
        rand_act.append(float(np.sum(np.var(z_te_s @ Q, axis=0)) / var_total) if var_total > 0 else 1.0)
    rand_act_arr = np.array(rand_act, dtype=np.float64)
    random_activity_frac_mean = float(rand_act_arr.mean())
    random_activity_frac_p5 = float(np.percentile(rand_act_arr, 5))
    activity_control_ok = bool(float(rand_act_arr.std()) > 1e-9)
    activity_discriminative = bool(activity_control_ok and (f_jspace < random_activity_frac_p5))
    # Legacy retention (reported for continuity; NON-discriminative by construction -- see 723 autopsy).
    z_te_j = (z_te_s @ B) @ B.T
    bal_acc_jspace = _balanced_accuracy(y_te, _predict(z_te_j, W, classes), classes)
    predictive_retention = float(bal_acc_jspace / bal_acc_full) if bal_acc_full > 1e-9 else 0.0

    result.update({
        "readout_estimable": True,
        "degenerate_map": False,
        "latent_dim": int(d),
        "effective_classes": [int(c) for c in classes],
        "bal_acc_full": round(bal_acc_full, 6),
        "bal_acc_majority_baseline": round(bal_acc_majority, 6),
        "null_mean": round(null_mean, 6),
        "null_p95": round(null_p95, 6),
        "above_null": above_null,
        "signal_margin": round(signal_margin, 6),
        # FIX 1a -- discriminative concentration (LOAD-BEARING).
        "frac_dims_90_top": round(frac_top, 6),
        "frac_dims_90_random_mean": round(random_frac_mean, 6),
        "frac_dims_90_random_p5": round(random_frac_p5, 6),
        "concentration_ratio": round(concentration_ratio, 6),
        "concentration_control_ok": concentration_control_ok,
        "concentration_discriminative": concentration_discriminative,
        # FIX 1b -- matched-rank random-subspace activity contrast (corroborating).
        "jspace_dim": int(jspace_dim),
        "jspace_activity_fraction": round(f_jspace, 8),
        "random_activity_frac_mean": round(random_activity_frac_mean, 6),
        "random_activity_frac_p5": round(random_activity_frac_p5, 6),
        "activity_control_ok": activity_control_ok,
        "activity_discriminative": activity_discriminative,
        # Legacy (reported; non-discriminative).
        "bal_acc_jspace": round(bal_acc_jspace, 6),
        "predictive_retention": round(predictive_retention, 6),
    })
    return result


def _broadcast_alignment(
    occupancy: np.ndarray, broadcast: np.ndarray, rng: np.random.Generator
) -> Dict[str, Any]:
    """FIX 2b -- correlate per-tick J-space occupancy with the broadcast 'report' signal, vs a
    permutation null. Returns assessable=False when the broadcast signal does not vary enough."""
    mask = np.isfinite(occupancy) & np.isfinite(broadcast)
    occ = occupancy[mask]
    bc = broadcast[mask]
    n_varying = int(bc.size)
    if n_varying < BROADCAST_MIN_VARYING_TICKS or float(np.std(bc)) < 1e-9 or float(np.std(occ)) < 1e-9:
        return {"assessable": False, "n_varying": n_varying}
    corr = float(np.corrcoef(occ, bc)[0, 1])
    if not math.isfinite(corr):
        return {"assessable": False, "n_varying": n_varying}
    null_abs = np.empty(N_PERM, dtype=np.float64)
    bc_perm = bc.copy()
    for i in range(N_PERM):
        rng.shuffle(bc_perm)
        c = np.corrcoef(occ, bc_perm)[0, 1]
        null_abs[i] = abs(c) if math.isfinite(c) else 0.0
    null_p95 = float(np.percentile(null_abs, 95))
    aligned = bool(abs(corr) >= max(BROADCAST_ALIGN_MIN_ABS_CORR, null_p95))
    return {
        "assessable": True,
        "n_varying": n_varying,
        "corr": round(corr, 6),
        "abs_corr": round(abs(corr), 6),
        "null_p95": round(null_p95, 6),
        "aligned": aligned,
    }


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

        # Progress print (GLOBAL episode counter; runner parses '[train] ... ep N/M').
        if (ep + 1) % 25 == 0 or ep == p1_start or ep == p2_start or (ep + 1) == total_train_eps:
            print(
                f"  [train] jlens seed={seed} phase={phase_label} ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

        # P2 tick-budget termination.
        if is_p2 and n_p2_ticks >= p2_ticks_target:
            print(
                f"  [train] jlens seed={seed} phase=P2 ep {ep + 1}/{total_train_eps} "
                f"(P2 tick budget {n_p2_ticks}>={p2_ticks_target} reached)",
                flush=True,
            )
            break

    # ----- Per-seed J-lens readout -----
    marginal_classes = sorted(committed_class_counts.keys())
    per_horizon: Dict[str, Any] = {}
    for h in HORIZONS:
        z_pairs: List[np.ndarray] = []
        y_pairs: List[int] = []
        for (z_seq, c_seq, _b_seq) in p2_episode_logs:
            m = len(c_seq)
            for i in range(m - h):
                z_pairs.append(z_seq[i])
                y_pairs.append(int(c_seq[i + h]))
        if len(z_pairs) >= 60:
            z_all = np.vstack(z_pairs)
            y_all = np.asarray(y_pairs, dtype=np.int64)
            per_horizon[str(h)] = _jlens_readout_one_horizon(
                z_all, y_all, RIDGE_LAMBDA, N_PERM, np.random.default_rng(seed + 700 + h)
            )
        else:
            per_horizon[str(h)] = {
                "n_pairs_total": int(len(z_pairs)), "readout_estimable": False,
                "reason": "too_few_pairs",
            }

    prim = per_horizon.get(str(PRIMARY_H), {"readout_estimable": False})
    n_train_pairs_primary = int(prim.get("n_train", 0))
    n_effective_classes_primary = int(prim.get("n_effective_classes", 0))

    # ----- FIX 2b: broadcast alignment (J-space occupancy vs broadcast signal) at PRIMARY_H -----
    # Occupancy proxy: per-tick projection energy onto the primary-H J-space is expensive to carry
    # back here; use per-tick ||z||-normalized broadcast correlation with the *concentration* proxy
    # = squared latent norm in the top-influence dims. Simpler robust proxy: correlate the broadcast
    # signal against per-tick total latent activity in the primary-H effective window is not the
    # workspace test. We instead correlate broadcast with the per-tick J-space OCCUPANCY computed
    # from the primary-H map, reconstructed per tick.
    broadcast_result: Dict[str, Any] = {"assessable": False, "n_varying": 0}
    prim_estimable = bool(prim.get("readout_estimable", False))
    if prim_estimable and int(n_broadcast_available) >= BROADCAST_MIN_VARYING_TICKS:
        # Rebuild the primary-H J-space basis and standardizer from the SAME pairs, then compute
        # per-tick occupancy = fraction of that tick's standardized latent energy in the J-space.
        occ_list: List[float] = []
        bc_list: List[float] = []
        try:
            # Reconstruct standardizer + J-space from primary-H train split (deterministic).
            z_pairs_p: List[np.ndarray] = []
            y_pairs_p: List[int] = []
            for (z_seq, c_seq, _b) in p2_episode_logs:
                m = len(c_seq)
                for i in range(m - PRIMARY_H):
                    z_pairs_p.append(z_seq[i])
                    y_pairs_p.append(int(c_seq[i + PRIMARY_H]))
            z_all_p = np.vstack(z_pairs_p)
            y_all_p = np.asarray(y_pairs_p, dtype=np.int64)
            n_tr = int(math.floor(TRAIN_FRAC * z_all_p.shape[0]))
            z_tr_p = z_all_p[:n_tr]
            mu_p = z_tr_p.mean(axis=0, keepdims=True)
            sd_p = z_tr_p.std(axis=0, keepdims=True)
            sd_p = np.where(sd_p < 1e-8, 1.0, sd_p)
            eff = prim.get("effective_classes", [])
            classes_p = np.array(sorted(int(c) for c in eff), dtype=np.int64)
            if classes_p.shape[0] >= MIN_EFFECTIVE_CLASSES:
                tr_mask_p = np.isin(y_all_p[:n_tr], classes_p)
                z_tr_s_p = ((z_tr_p - mu_p) / sd_p)[tr_mask_p]
                y_tr_p = y_all_p[:n_tr][tr_mask_p]
                cnts = {int(c): int((y_tr_p == c).sum()) for c in classes_p}
                inv = {c: 1.0 / max(1, cnts[c]) for c in cnts}
                nrm = sum(inv.values())
                cw = {c: inv[c] / nrm * len(classes_p) for c in inv}
                sw = np.array([cw[int(v)] for v in y_tr_p], dtype=np.float64)
                Wp = _fit_ridge_onehot_weighted(z_tr_s_p, y_tr_p, classes_p, RIDGE_LAMBDA, sw)
                Up, Sp, _ = np.linalg.svd(Wp, full_matrices=False)
                sve = Sp ** 2
                if float(sve.sum()) > 0:
                    cumj = np.cumsum(sve) / float(sve.sum())
                    kdim = max(1, min(int(np.searchsorted(cumj, JSPACE_ENERGY) + 1), Up.shape[1]))
                    Bp = Up[:, :kdim]
                    # Per-tick occupancy over the P2 stream (only ticks with a broadcast reading).
                    for (z_seq, c_seq, b_seq) in p2_episode_logs:
                        for zt, bt in zip(z_seq, b_seq):
                            if bt is None:
                                continue
                            zs = (zt.reshape(1, -1) - mu_p) / sd_p
                            tot = float(np.sum(zs ** 2))
                            if tot <= 1e-12:
                                continue
                            proj = float(np.sum((zs @ Bp) ** 2))
                            occ_list.append(proj / tot)
                            bc_list.append(float(bt))
        except Exception:
            occ_list, bc_list = [], []
        if len(occ_list) >= BROADCAST_MIN_VARYING_TICKS:
            broadcast_result = _broadcast_alignment(
                np.asarray(occ_list, dtype=np.float64),
                np.asarray(bc_list, dtype=np.float64),
                np.random.default_rng(seed + 90007),
            )

    # Readiness: enough P2 ticks, enough primary-H train pairs, >= MIN_EFFECTIVE_CLASSES effective
    # classes, above-null signal with margin, AND a non-degenerate concentration control.
    prim_above_null = bool(prim.get("above_null", False))
    prim_signal_margin = float(prim.get("signal_margin", 0.0) or 0.0)
    prim_signal_margin_ok = bool(prim_signal_margin >= SIGNAL_MARGIN_FLOOR)
    prim_control_ok = bool(prim.get("concentration_control_ok", False))
    prim_random_frac_mean = float(prim.get("frac_dims_90_random_mean", 0.0) or 0.0)

    seed_readiness = bool(
        n_p2_ticks >= TOTAL_TICKS_FLOOR
        and n_train_pairs_primary >= TRAIN_PAIRS_FLOOR
        and n_effective_classes_primary >= MIN_EFFECTIVE_CLASSES
        and prim_estimable
        and prim_above_null
        and prim_signal_margin_ok
        and prim_control_ok
    )

    # Primary-H discriminative signature.
    prim_concentration_discriminative = bool(prim.get("concentration_discriminative", False))
    prim_activity_discriminative = bool(prim.get("activity_discriminative", False))

    # Reactive-bypass (H1 vs H5): integrative (H5) concentration tighter than reactive (H1).
    rH1 = per_horizon.get(str(REACTIVE_H), {})
    rH5 = per_horizon.get(str(INTEGRATIVE_H), {})
    reactive_assessable = bool(
        rH1.get("readout_estimable", False) and rH5.get("readout_estimable", False)
        and (rH1.get("frac_dims_90_top") is not None)
        and (rH5.get("frac_dims_90_top") is not None)
    )
    reactive_bypass = False
    reactive_bypass_gap = None
    if reactive_assessable:
        gap = float(rH1["frac_dims_90_top"]) - float(rH5["frac_dims_90_top"])
        reactive_bypass_gap = round(gap, 6)
        reactive_bypass = bool(gap >= REACTIVE_BYPASS_MARGIN)

    broadcast_assessable = bool(broadcast_result.get("assessable", False))
    broadcast_aligned = bool(broadcast_result.get("aligned", False))

    return {
        "seed": int(seed),
        "error_note": error_note,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p2_eps_completed": int(n_p2_eps_completed),
        "n_committed_classes": int(len(marginal_classes)),
        "n_effective_classes_primary": n_effective_classes_primary,
        "committed_class_counts": {str(k): int(v) for k, v in sorted(committed_class_counts.items())},
        "broadcast_available_ticks": int(n_broadcast_available),
        "broadcast_alignment_available": bool(n_broadcast_available > 0),
        "n_train_pairs_primary": n_train_pairs_primary,
        "seed_readiness": seed_readiness,
        # Primary-H headline signature.
        "primary_h": int(PRIMARY_H),
        "primary_above_null": prim_above_null,
        "primary_signal_margin": round(prim_signal_margin, 6),
        "primary_signal_margin_ok": prim_signal_margin_ok,
        "primary_concentration_control_ok": prim_control_ok,
        "primary_random_frac_mean": round(prim_random_frac_mean, 6),
        "primary_concentration_discriminative": prim_concentration_discriminative,
        "primary_activity_discriminative": prim_activity_discriminative,
        "primary_bal_acc_full": prim.get("bal_acc_full"),
        "primary_null_p95": prim.get("null_p95"),
        "primary_frac_dims_90_top": prim.get("frac_dims_90_top"),
        "primary_frac_dims_90_random_mean": prim.get("frac_dims_90_random_mean"),
        "primary_concentration_ratio": prim.get("concentration_ratio"),
        "primary_jspace_dim": prim.get("jspace_dim"),
        "primary_jspace_activity_fraction": prim.get("jspace_activity_fraction"),
        "primary_random_activity_frac_p5": prim.get("random_activity_frac_p5"),
        "primary_predictive_retention": prim.get("predictive_retention"),
        "primary_latent_dim": prim.get("latent_dim"),
        # Reactive-bypass (H1 vs H5).
        "reactive_bypass_assessable": reactive_assessable,
        "reactive_bypass": reactive_bypass,
        "reactive_bypass_gap": reactive_bypass_gap,
        "frac_dims_90_top_H1": rH1.get("frac_dims_90_top"),
        "frac_dims_90_top_H5": rH5.get("frac_dims_90_top"),
        # Broadcast reportability.
        "broadcast_assessable": broadcast_assessable,
        "broadcast_aligned": broadcast_aligned,
        "broadcast_detail": broadcast_result,
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
        f"J-lens DISCRIMINATIVE-compactness diagnostic (all-ON single config; supersedes "
        f"{SUPERSEDES}; P0={p0_episodes} e2-warmup, P1={p1_episodes} two-head REINFORCE, "
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

    # ----- Discriminative signature gates (majority; only licensed when readiness met) -----
    n_above_null = sum(1 for r in ok_rows if r["primary_above_null"] and r["primary_signal_margin_ok"])
    signal_gate = bool(n_above_null >= MAJORITY_MIN_SEEDS)
    n_conc = sum(1 for r in ok_rows if r["primary_concentration_discriminative"])
    concentration_gate = bool(n_conc >= MAJORITY_MIN_SEEDS)         # LOAD-BEARING
    n_act = sum(1 for r in ok_rows if r["primary_activity_discriminative"])
    activity_gate = bool(n_act >= MAJORITY_MIN_SEEDS)
    n_reactive_assess = sum(1 for r in ok_rows if r["reactive_bypass_assessable"])
    n_reactive = sum(1 for r in ok_rows if r["reactive_bypass"])
    reactive_gate = bool(n_reactive >= MAJORITY_MIN_SEEDS)
    n_bcast_assess = sum(1 for r in ok_rows if r["broadcast_assessable"])
    n_bcast_aligned = sum(1 for r in ok_rows if r["broadcast_aligned"])
    broadcast_assessable_majority = bool(n_bcast_assess >= MAJORITY_MIN_SEEDS)
    broadcast_gate = bool(n_bcast_aligned >= MAJORITY_MIN_SEEDS)

    # Compact core = above-null + discriminative CONCENTRATION (load-bearing, the "predict-better-
    # than-random-same-dim" contrast) + reactive-bypass. The matched-rank random-SUBSPACE ACTIVITY
    # contrast (activity_gate) is the autopsy's OTHER "and/or" leg -- REPORTED + corroborating, NOT
    # required, because the SVD-activity fraction is confounded by ridge amplification of low-variance
    # directions (f_J ~ random p5 even for a genuinely compact map; verified on synthetic controls),
    # so ANDing it would false-negative real workspaces. Concentration is the load-bearing contrast.
    compact_core = bool(signal_gate and concentration_gate and reactive_gate)
    # Diffuse: there IS an above-null predictive signal, but concentration (load-bearing) fails.
    diffuse = bool(signal_gate and not concentration_gate)

    # ----- Self-route (HYPOTHESIS, not a verdict) -----
    if not readiness_met or not signal_gate:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif compact_core and broadcast_assessable_majority and broadcast_gate:
        outcome = "PASS"
        direction = "non_contributory"   # diagnostic; never weights any claim
        label = "compact_action_coupled_subspace_present"
    elif compact_core and not broadcast_assessable_majority:
        outcome = "PASS"
        direction = "non_contributory"
        label = "compact_subspace_present_broadcast_unverified"
    elif compact_core and broadcast_assessable_majority and not broadcast_gate:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "compact_but_not_broadcast_reportable"
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
    agg_frac_top = _agg("primary_frac_dims_90_top")
    agg_frac_random = _agg("primary_frac_dims_90_random_mean")
    agg_conc_ratio = _agg("primary_concentration_ratio")
    agg_jspace_frac = _agg("primary_jspace_activity_fraction")
    agg_rand_act_p5 = _agg("primary_random_activity_frac_p5")
    agg_reactive_gap = _agg("reactive_bypass_gap")
    agg_resources = _agg("mean_resources_per_episode")
    latent_dims = [r.get("primary_latent_dim") for r in ok_rows if r.get("primary_latent_dim")]
    latent_dim = int(latent_dims[0]) if latent_dims else None
    min_ticks = min([r["n_p2_ticks"] for r in ok_rows], default=0)
    min_train_pairs = min([r["n_train_pairs_primary"] for r in ok_rows], default=0)
    min_eff_classes = min([r["n_effective_classes_primary"] for r in ok_rows], default=0)
    any_broadcast = any(r["broadcast_alignment_available"] for r in ok_rows)

    # Precondition counts keyed to the same majority the self-route uses, so a straggler seed that
    # misses one floor does NOT false-flag a majority PASS with `precondition_unmet` (the indexer
    # recomputes met from measured>=threshold on floor-direction numerics).
    n_ticks_ok = sum(1 for r in ok_rows if r["n_p2_ticks"] >= TOTAL_TICKS_FLOOR)
    n_pairs_ok = sum(1 for r in ok_rows if r["n_train_pairs_primary"] >= TRAIN_PAIRS_FLOOR)
    n_classes_ok = sum(1 for r in ok_rows if r["n_effective_classes_primary"] >= MIN_EFFECTIVE_CLASSES)
    # SAME-STATISTIC readiness for the concentration load-bearing gate: min frac_dims_90 under the
    # RANDOM ordering across the seeds we license the branch on (ready seeds), 0 if none ready.
    ready_rows = [r for r in ok_rows if r["seed_readiness"]]
    min_ready_random_frac = min(
        [r["primary_random_frac_mean"] for r in ready_rows
         if isinstance(r.get("primary_random_frac_mean"), (int, float))],
        default=0.0,
    )

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "readout_sufficient_p2_ticks",
                "kind": "readiness",
                "description": (
                    f"Seeds with P2 ticks >= TOTAL_TICKS_FLOOR ({TOTAL_TICKS_FLOOR}) so the "
                    f"dispositional map is estimable, counted >= READINESS_MIN_SEEDS. min P2 ticks "
                    f"this run = {int(min_ticks)}. Below => substrate_not_ready_requeue, NOT a verdict."
                ),
                "control": "count of seeds clearing the P2-tick floor",
                "measured": float(n_ticks_ok),
                "threshold": float(READINESS_MIN_SEEDS),
                "met": bool(n_ticks_ok >= READINESS_MIN_SEEDS),
            },
            {
                "name": "readout_sufficient_train_pairs_primary_h",
                "kind": "readiness",
                "description": (
                    f"Seeds with usable (z_t, committed_class_{{t+H}}) TRAIN pairs at PRIMARY_H >= "
                    f"TRAIN_PAIRS_FLOOR ({TRAIN_PAIRS_FLOOR}), counted >= READINESS_MIN_SEEDS. min "
                    f"train pairs this run = {int(min_train_pairs)}. Below => substrate_not_ready_requeue."
                ),
                "control": "count of seeds clearing the train-pairs floor at primary horizon",
                "measured": float(n_pairs_ok),
                "threshold": float(READINESS_MIN_SEEDS),
                "met": bool(n_pairs_ok >= READINESS_MIN_SEEDS),
            },
            {
                "name": "at_least_three_effective_committed_classes",
                "kind": "readiness",
                "description": (
                    f"Seeds with >= MIN_EFFECTIVE_CLASSES ({MIN_EFFECTIVE_CLASSES}) EFFECTIVE committed "
                    f"classes (>= {MIN_PER_CLASS_TRAIN} train / >= {MIN_PER_CLASS_TEST} test instances each), "
                    f"counted >= READINESS_MIN_SEEDS. FIX 3: guards the 723 seed-43 degeneracy (one class "
                    f"with a single instance -> effectively binary). min effective classes this run = "
                    f"{int(min_eff_classes)}. Below => substrate_not_ready_requeue."
                ),
                "control": "count of seeds with enough well-populated committed-action classes",
                "measured": float(n_classes_ok),
                "threshold": float(READINESS_MIN_SEEDS),
                "met": bool(n_classes_ok >= READINESS_MIN_SEEDS),
            },
            {
                "name": "action_predictable_above_null_with_margin_majority",
                "kind": "readiness",
                "description": (
                    "The linear readout predicts committed_class_{t+H=PRIMARY} above the "
                    "label-permutation p95 by >= SIGNAL_MARGIN_FLOOR on >= MAJORITY_MIN_SEEDS seeds. "
                    "Absent a real signal, '90% of predictive power' is a degenerate concentration "
                    "target (undertrained-substrate signature, NOT a workspace null) => "
                    "substrate_not_ready_requeue."
                ),
                "control": "count of seeds with (bal_acc_full - null_p95) >= SIGNAL_MARGIN_FLOOR at PRIMARY_H",
                "measured": float(n_above_null),
                "threshold": float(MAJORITY_MIN_SEEDS),
                "met": bool(signal_gate),
            },
            {
                # SAME-STATISTIC readiness: the load-bearing gate routes on the CONCENTRATION
                # statistic frac_dims_90 (top vs random). This asserts the SAME statistic on the
                # NEGATIVE control (random ordering): if a random ordering itself needs few dims,
                # the space is trivially low-dim and the concentration contrast is vacuous. Measured
                # = min frac_dims_90_random over the ready seeds we license the branch on (0 if none).
                "name": "concentration_control_discriminable_random_ordering",
                "kind": "readiness",
                "description": (
                    f"The random-ordering control needs >= CONCENTRATION_CONTROL_FLOOR "
                    f"({CONCENTRATION_CONTROL_FLOOR}) of the latent dims to recover 90% of the above-chance "
                    f"accuracy (SAME frac_dims_90 statistic the load-bearing concentration gate routes on, "
                    f"measured on the random negative control). If the control is itself concentrated, "
                    f"'top beats random' is vacuous => substrate_not_ready_requeue, NEVER a diffuse/compact "
                    f"verdict."
                ),
                "control": "min over ready seeds of frac_dims_90 under RANDOM dim ordering (negative control)",
                "measured": float(round(min_ready_random_frac, 6)),
                "threshold": float(CONCENTRATION_CONTROL_FLOOR),
                "met": bool(readiness_met and min_ready_random_frac >= CONCENTRATION_CONTROL_FLOOR),
            },
            {
                "name": "readiness_majority_seeds_estimable",
                "kind": "readiness",
                "description": (
                    "Readout is estimable (all floors above) on >= READINESS_MIN_SEEDS seeds; only then "
                    "is a compact/diffuse branch licensed. Below => substrate_not_ready_requeue."
                ),
                "control": "count of readout-estimable seeds",
                "measured": float(n_ready),
                "threshold": float(READINESS_MIN_SEEDS),
                "met": bool(readiness_met),
            },
        ],
        "criteria": [
            {
                "name": "concentration_discriminative_majority",
                "load_bearing": True,
                "passed": bool(concentration_gate),
            },
            {
                "name": "matched_rank_activity_discriminative_majority",
                "load_bearing": False,
                "passed": bool(activity_gate),
            },
            {
                "name": "reactive_bypass_H1_vs_H5_majority",
                "load_bearing": False,
                "passed": bool(reactive_gate),
            },
            {
                "name": "broadcast_reportability_majority",
                "load_bearing": False,
                "passed": bool(broadcast_gate),
            },
        ],
        # NOTE: every key here follows the "True = non-degenerate/good" convention AND must be True
        # on EVERY PASS label. Broadcast assessability is DELIBERATELY excluded -- the PASS label
        # `compact_subspace_present_broadcast_unverified` has broadcast unassessable BY DESIGN, and
        # putting a legitimately-False-on-PASS key here would re-create the V3-EXQ-723 `diffuse_branch`
        # schema-collision false `vacuous_pass` (autopsy sections 3/8). Broadcast lives in
        # signature_gates + the label itself, not here.
        "criteria_non_degenerate": {
            "readiness_estimable": bool(readiness_met),
            "above_null_signal_with_margin_majority": bool(signal_gate),
            "concentration_control_nondegenerate_majority": bool(
                sum(1 for r in ok_rows if r["primary_concentration_control_ok"]) >= MAJORITY_MIN_SEEDS
            ),
            "reactive_bypass_assessable_majority": bool(n_reactive_assess >= MAJORITY_MIN_SEEDS),
        },
    }

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": seeds,
        "supersedes": SUPERSEDES,
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
            "reactive_h": int(REACTIVE_H),
            "integrative_h": int(INTEGRATIVE_H),
            "ridge_lambda": float(RIDGE_LAMBDA),
            "train_frac": float(TRAIN_FRAC),
            "n_perm": int(N_PERM),
            "concentration_energy_frac": float(CONCENTRATION_ENERGY_FRAC),
            "n_random_orderings": int(N_RANDOM_ORDERINGS),
            "concentration_ratio_ceil": float(CONCENTRATION_RATIO_CEIL),
            "concentration_abs_ceil": float(CONCENTRATION_ABS_CEIL),
            "concentration_control_floor": float(CONCENTRATION_CONTROL_FLOOR),
            "jspace_energy": float(JSPACE_ENERGY),
            "n_random_subspaces": int(N_RANDOM_SUBSPACES),
            "reactive_bypass_margin": float(REACTIVE_BYPASS_MARGIN),
            "broadcast_align_min_abs_corr": float(BROADCAST_ALIGN_MIN_ABS_CORR),
            "broadcast_min_varying_ticks": int(BROADCAST_MIN_VARYING_TICKS),
            "signal_margin_floor": float(SIGNAL_MARGIN_FLOOR),
            "total_ticks_floor": int(TOTAL_TICKS_FLOOR),
            "train_pairs_floor": int(TRAIN_PAIRS_FLOOR),
            "min_per_class_train": int(MIN_PER_CLASS_TRAIN),
            "min_per_class_test": int(MIN_PER_CLASS_TEST),
            "min_effective_classes": int(MIN_EFFECTIVE_CLASSES),
            "readiness_min_seeds": int(READINESS_MIN_SEEDS),
            "majority_min_seeds": int(MAJORITY_MIN_SEEDS),
        },
        "signature_gates": {
            "readiness_met": readiness_met,
            "n_ready_seeds": int(n_ready),
            "signal_gate_above_null_margin": signal_gate,
            "n_above_null_seeds": int(n_above_null),
            "concentration_gate_load_bearing": concentration_gate,
            "n_concentration_seeds": int(n_conc),
            "activity_gate": activity_gate,
            "n_activity_seeds": int(n_act),
            "reactive_gate": reactive_gate,
            "n_reactive_assessable_seeds": int(n_reactive_assess),
            "n_reactive_seeds": int(n_reactive),
            "broadcast_assessable_majority": broadcast_assessable_majority,
            "n_broadcast_assessable_seeds": int(n_bcast_assess),
            "broadcast_gate": broadcast_gate,
            "n_broadcast_aligned_seeds": int(n_bcast_aligned),
            "compact_core_branch": compact_core,
            "diffuse_branch": diffuse,
        },
        "aggregate_dvs": {
            "primary_bal_acc_full_mean": round(agg_bal_acc, 6),
            "primary_null_p95_mean": round(agg_null_p95, 6),
            "primary_frac_dims_90_top_mean": round(agg_frac_top, 6),
            "primary_frac_dims_90_random_mean": round(agg_frac_random, 6),
            "primary_concentration_ratio_mean": round(agg_conc_ratio, 6),
            "primary_jspace_activity_fraction_mean": round(agg_jspace_frac, 8),
            "primary_random_activity_frac_p5_mean": round(agg_rand_act_p5, 6),
            "reactive_bypass_gap_mean": round(agg_reactive_gap, 6),
            "mean_resources_per_episode_mean": round(agg_resources, 6),
            "latent_dim": latent_dim,
        },
        "interpretation_grid": {
            "compact_action_coupled_subspace_present": (
                "above-null predictive signal AND discriminative concentration (frac_dims_90_top / "
                "frac_dims_90_random < ratio ceil AND frac_top < abs ceil -- the LOAD-BEARING gate) AND "
                "reactive-bypass (H5 tighter than H1) AND broadcast-alignment (J-space occupancy tracks "
                "the broadcast substrate), each on a majority of seeds. The matched-rank random-subspace "
                "activity contrast (f_J vs random p5) is REPORTED as corroboration but does not gate. "
                "REE has a COMPACT, action-coupled, reportable workspace -- the J-space analogue. "
                "HYPOTHESIS (not a verdict): raises the SD-064 prior and GREENLIGHTS the SD-027 V3 "
                "boundary-gate retrofit that unlocks Experiment B. Route to /failure-autopsy."
            ),
            "compact_subspace_present_broadcast_unverified": (
                "compact + activity + reactive-bypass all pass, but the broadcast (reportability) proxy "
                "did not vary enough to assess alignment. Present but reportability UNVERIFIED -- does "
                "NOT greenlight the SD-027 build on its own. Route to /failure-autopsy."
            ),
            "compact_but_not_broadcast_reportable": (
                "compact + activity + reactive-bypass pass, broadcast IS assessable but J-space occupancy "
                "does NOT track the broadcast substrate above null. Compact + integrative but not "
                "reportable -- GWT-incomplete; does NOT greenlight. Route to /failure-autopsy."
            ),
            "no_compact_workspace_diffuse": (
                "action IS predictable above null, but the action-predictive structure is DIFFUSE "
                "(concentration_ratio ~ 1: top-influence dims need ~as many dims as random ordering). "
                "HYPOTHESIS (not a verdict): evidence toward the SD-027-original pluralist / no-single-"
                "workspace reading. Route to /failure-autopsy."
            ),
            "substrate_not_ready_requeue": (
                "under-sampled P2 / too few train pairs / < MIN_EFFECTIVE_CLASSES effective classes / "
                "no above-null signal with margin / degenerate random-ordering control. NOT a verdict -- "
                "re-queue at a larger P2 budget / more training. Draw NO conclusion about SD-064."
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
        "supersedes": SUPERSEDES,
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
            f"V3-EXQ-723a REE-NATIVE J-LENS DISCRIMINATIVE-COMPACTNESS DIAGNOSTIC (Experiment A of "
            f"SD-064; SUPERSEDES V3-EXQ-723). experiment_purpose=diagnostic, EXCLUDED from governance "
            f"scoring; claim_ids=[] (SD-064 global-workspace + MECH-191 signal-legibility CONTEXT only). "
            f"Redesign fixing V3-EXQ-723's NON-DISCRIMINATIVE gates (per failure_autopsy_V3-EXQ-723_"
            f"2026-07-09): 723's jspace_activity_fraction<0.10 (passed 350x under ceiling) and "
            f"predictive_retention>=0.80 (~1.0 by construction) were trivially satisfiable by any weak "
            f"linear ridge map. FIX 1 -- discriminative compactness via matched-null baselines: a "
            f"CONCENTRATION CURVE (frac of latent dims, top-influence-first, to recover 90% of above-"
            f"chance balanced accuracy) vs the SAME curve under RANDOM dim orderings "
            f"(concentration_ratio = frac_top/frac_random; LOAD-BEARING, ~1 for a diffuse map so a weak "
            f"map does not trivially pass), plus a matched-rank RANDOM-SUBSPACE activity contrast "
            f"(f_J < random same-rank p5). FIX 2 -- GATE (not just report) the plan's positive-mode "
            f"contrasts: reactive-bypass (H1-vs-H5 concentration divergence -- integrative H5 tighter "
            f"than reactive H1) and broadcast-alignment (per-tick J-space occupancy vs MECH-287/SD-037 "
            f"broadcast proxy, vs a permutation null; withheld to *_broadcast_unverified when the "
            f"proxy does not vary). FIX 3 -- committed-action-balanced eval: restrict to EFFECTIVE "
            f"classes (>= {MIN_PER_CLASS_TRAIN} train / >= {MIN_PER_CLASS_TEST} test), require "
            f">= {MIN_EFFECTIVE_CLASSES} effective classes (guards 723 seed-43 near-binary degeneracy), "
            f"class-balanced weighted ridge, longer P2 ({P2_TICKS_TARGET} ticks). Same substrate as 723 "
            f"(V3-EXQ-714 ARM_ON, all-ON), NO new mechanism. Self-route (HYPOTHESIS, NOT a verdict): "
            f"compact core (above-null + discriminative concentration + activity + reactive-bypass) + "
            f"broadcast-aligned -> compact_action_coupled_subspace_present (raises SD-064 prior; "
            f"GREENLIGHTS the SD-027 V3 gate retrofit unlocking Experiment B); broadcast unassessable -> "
            f"compact_subspace_present_broadcast_unverified (does NOT greenlight); broadcast not aligned "
            f"-> compact_but_not_broadcast_reportable; predictable-but-diffuse -> no_compact_workspace_"
            f"diffuse; under-sampled / no signal / degenerate control -> substrate_not_ready_requeue. "
            f"interpretation_label={result['interpretation_label']}; readiness_met="
            f"{result['signature_gates']['readiness_met']}. PROMOTES / DEMOTES NOTHING; route to "
            f"/failure-autopsy for adjudication. Design: "
            f"REE_assembly/evidence/planning/global_workspace_jlens_plan.md section 2."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "single_config": "all-mechanisms-ON (V3-EXQ-714 ARM_ON), use_candidate_rule_field=True (identical to 719a/723)",
            "arm_fingerprint_exempt": ARM_FINGERPRINT_EXEMPT,
            "supersedes": SUPERSEDES,
            "latent_vector": "concat of available streams z_world/z_self/z_harm/z_harm_a (detached), the REE 'residual stream'",
            "committed_class_source": "int(agent.select_action(...).argmax()) -- executed committed first-action class",
            "readout": (
                "class-BALANCED weighted ridge z_t -> committed_class_{t+H} over EFFECTIVE classes; "
                "temporal 70/30 split; balanced-accuracy vs label-permutation null; DISCRIMINATIVE "
                "compactness = concentration curve (top-influence vs random dim ordering) + matched-rank "
                "random-subspace activity contrast; GATED reactive-bypass (H1-vs-H5) + broadcast-alignment"
            ),
            "phases": "P0 e2-warmup -> P1 frozen-encoder TWO-head REINFORCE (lateral_pfc + OFC devaluation) -> P2 frozen TICK-BUDGETED eval/logging",
            "all_on_config_sourced_from": "V3-EXQ-714 ARM_ON via V3-EXQ-719a/723",
            "alpha_world": 0.9,
            "use_sleep_loop": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-723a REE-native J-lens DISCRIMINATIVE-compactness DIAGNOSTIC "
            "(supersedes V3-EXQ-723; SD-064 Experiment A)"
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
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
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
        f"frac_dims_90_top_mean={ag['primary_frac_dims_90_top_mean']} "
        f"frac_dims_90_random_mean={ag['primary_frac_dims_90_random_mean']} "
        f"concentration_ratio_mean={ag['primary_concentration_ratio_mean']} | "
        f"conc_gate={sg['concentration_gate_load_bearing']} act_gate={sg['activity_gate']} "
        f"reactive_gate={sg['reactive_gate']} bcast_gate={sg['broadcast_gate']} "
        f"compact_core={sg['compact_core_branch']} diffuse={sg['diffuse_branch']}",
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
