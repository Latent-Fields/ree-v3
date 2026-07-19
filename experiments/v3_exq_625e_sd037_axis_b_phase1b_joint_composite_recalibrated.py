"""
V3-EXQ-625e -- SD-037 axis (b) Phase 1b JOINT-COMPOSITE substrate-readiness
falsifier, RECALIBRATED for the TRAINED harm pathway (successor to V3-EXQ-625d;
supersedes it).

WHY 625e (per failure_autopsy_V3-EXQ-625d_2026-06-18 Section 7): 625d ran the full
joint composite + 569i conversion config and self-routed substrate_not_ready_requeue
(R3 0/3, R4 0/3) -- internally consistent, weakens nothing. The autopsy diagnosed an
OVER-DETERMINED committed-action monostrategy-lock (entropy 0.0 3/3) downstream of the
axis-(b) SATURATED-THREAT regime: z_harm_a pinned ~6 (the TRAINED harm pathway
over-drives the 625b-era overlay ~14x; 0 crossings of the 0.4 threshold), candidate-pool
collapse (cand_world_pairwise_dist < floor 2/3), and F-dominance over the modulatory
range (seed 42 route_range operative yet committed entropy still 0). The 569i conversion
PASS is ENV-CONDITIONAL (it rests on the reef-bipartite env's structural guarantee of
categorically-opposite first-action argmaxes, which the saturated-threat overlay removes).

625e recalibrates the axis-(b) MEASUREMENT env for the trained-pathway z_harm_a magnitude
so z_harm_a sits in a SUB-SATURATING, OSCILLATION-CAPABLE band that can cross 0.4 BOTH ways:
  (1) hazard_harm + proximity_harm_scale lowered ~10x (0.2 -> 0.02), AND
  (2) the SD-029 scheduled-threat made TIME-VARYING / PULSED (on/off duty-cycle windows)
      so z_harm_a builds during ON windows and RELIEVES (limb-damage healing) during OFF
      windows -- the C3a above->below crossing the saturated tonic regime made impossible.
  (3) the policy is trained on the 603q-STABILIZED harm-pathway config (decoupled encoder
      LR 3e-4 + LR warmup 250 + Stage-H hazard headroom num_hazards=6) so the base harm
      landscape forms on >=2/3 seeds (the 603p seed-fragility 625d hit on seeds 43/44).
The four non-vacuity self-route guards (R1-R4) are UNCHANGED and carry GAP-A conversion-
propagation as a HARD upstream gate: a recalibrated threat that STILL cannot clear the
candidate-pool collapse self-routes substrate_not_ready_requeue (R3/R4), NEVER a weakens.

PREDECESSOR (625d) docstring follows, with the env/harm-config deltas above applied:

PURPOSE (substrate readiness, NOT governance evidence; claim_ids=[]):
sd_037_axis_b:P1b. 625b/625c measured z_harm_a dynamic crossings on the axis-(b)
sustained-threat env overlay using the gap4-baseline `warmup_train` BASE policy
and found a MONOSTRATEGY lock (each seed froze on a single action attractor: seed
continuously-above OR continuously-below the duration_input_threshold, 0 dynamic
crossings) -- NOT because the env is incapable (625b seed 7 proved capability) but
because the committed policy never produced the diverse behaviour needed to drive
z_harm_a both UP across and back DOWN across the threshold. The plan-of-record
(sd_037_axis_b_sustained_threat_curriculum_plan.md, node P1b) reserved the
redesigned successor V3-EXQ-625d (JOINT-COMPOSITE-ON) for AFTER committed-action
diversity is demonstrated by the behavioral_diversity_isolation GAP-A lineage.

THE GATE IS NOW MET: V3-EXQ-569i (PASS/supports, non_degenerate, claim ARC-065,
2026-06-16) demonstrated committed-action diversity reaches COMMITTED ACTION on
the TOP-K shortlist conversion substrate -- ARM_1 (e2_world_forward source + top_k
shortlist k=3 + gain=2.0 std-basis authority) selected-action class entropy strict
above BOTH the matched-noise control AND the collapsed proposer on 2/3 seeds, with
the in-arm route-range gate (0.267 > 0.01) AND e2-divergence (0.082 > 0.03) met.
625d carries that exact 569i-validated conversion config into the axis-(b)
sustained-threat measurement, on a policy trained THROUGH the
scaffolded_sd054_onboarding scheduler path (NOT 625c's warmup_train base policy),
to ask: does the now-diverse committed policy produce OSCILLATING, non-monostrategy
z_harm_a (genuine dynamic crossings) under sustained threat?

JOINT COMPOSITE (per the P1b resume_condition):
  (i)   POLICY via the FULL scaffolded_sd054_onboarding curriculum (Stage-0 nursery
        -> Stage-0b consolidation -> P0 warm-up -> Stage-H isolated hazard avoidance
        with harm-pathway training -> P1 foraging), mirroring the validated 603n
        config -- a survival-AND-foraging-AND-avoidance-competent base policy (NOT
        625c's warmup_train base policy, which had no trained harm landscape and
        froze).
  (ii)  the 569i-validated CONVERSION config ON: use_modulatory_channel_routing +
        route_source=cand_world_summary + use_modulatory_selection_authority gain=2.0
        normalize_basis=std + use_modulatory_shortlist_then_modulate + mode=top_k +
        shortlist_k=3 + candidate_summary_source=e2_world_forward (the committed-
        action-diversity lever).
  (iii) MECH-341 (use_e3_score_diversity stratified+entropy) + SD-056
        (e2_action_contrastive, trained online on the axis-(b) env in a short warm-up
        so e2.world_forward carries action-conditional divergence under threat).
  (iv)  the axis-(b) SUSTAINED-THREAT env overlay (SD-029 scheduled_external_hazard
        interval=20 prob=0.7 adjacent_only + hazard_harm=0.2 + proximity_harm_scale=0.2)
        applied to a DIM-MATCHED scaffold-style measurement env.

SHARPENED C3 (per failure_autopsy_V3-EXQ-625b_2026-06-02 Section 8.3; the same C3a/C3b
the 625c redesign introduced):
  per seed, count directional crossings of z_harm_a across the duration_input_threshold:
    C3a >= 1 above->below transition AND C3b >= 1 below->above transition.
  EXPERIMENT C3 = dynamic-crossings pass on >= 2/3 seeds. A continuously-frozen-above
  policy has 0 above->below (FAILS C3a); a continuously-frozen-below policy has 0
  crossings of either kind (FAILS both). Only a policy whose committed behaviour
  transitions enough to drive z_harm_a up across AND back down across the threshold
  passes -- the non-monostrategy oscillation the joint composite is meant to produce.

PRE-REGISTERED NON-VACUITY GATES (else self-route substrate_not_ready_requeue, NEVER
a substrate verdict / weakens). A z_harm_a crossing measurement is interpretable as a
genuine axis-(b) oscillation verdict ONLY if all four readiness preconditions hold --
otherwise a 0-crossing read is "the lever that should drive oscillation was not
operative here" (re-derives 625c's monostrategy lock for a substrate reason), NOT
"axis-(b) sustained threat cannot produce oscillation":
  (R1) curriculum_fired      : external_hazard_event_count > 0 in 3/3 measurement
                               seeds (SD-029 sustained-threat curriculum confirmed
                               firing). Below -> env overlay mis-applied -> requeue.
  (R2) z_harm_a_nonzero      : z_harm_a NONZERO fraction >= 0.01 on >= 2/3 seeds
                               (the affective stream is populated; a flat-zero
                               z_harm_a cannot cross anything). Below -> affective
                               stream not engaged -> requeue.
  (R3) conversion_operative  : the 569i-validated conversion is operative HERE --
                               the IN-ARM modulatory_channel_route_range (the
                               V3-EXQ-662 statistic, read LIVE at the select tick)
                               > ROUTE_RANGE_FLOOR AND the e2.world_forward
                               per-candidate spread cand_world_pairwise_dist >
                               C1_PAIRWISE_DIST_FLOOR, each on >= 2/3 seeds (the SAME
                               RANGE statistics 569i's readiness gates on; SD-056
                               under-trained -> routed range ~0 -> requeue, NEVER a
                               verdict).
  (R4) committed_diversity   : the committed policy is NOT monostrategy-locked --
                               selected-action class entropy > C3_SELECTED_ENTROPY_FLOOR
                               on >= 2/3 seeds (the 569i C_R1B floor; the diversity
                               that MUST reach committed action for z_harm_a to be
                               able to oscillate). A monostrategy-locked policy
                               (entropy ~0) makes 0 crossings for a CONVERSION reason,
                               not an env-capability reason -> requeue.
With ALL FOUR met, a sub-2/3 C3 is a GENUINE residual verdict (the diverse committed
policy STILL does not oscillate z_harm_a under sustained threat) routed to
/failure-autopsy -- NOT substrate_not_ready, NOT a weakens (claim_ids=[]).

ON 625e PASS: the P1b gate clears and the chain P2 (deterministic p70 recalibration)
-> P3 (verification diagnostic) -> P4 (V3-EXQ-483f terminal validation, shared with
axis-a) becomes workable. (Flagged for a follow-on session; NOT done here.)

GUARDRAILS: SD-037 / MECH-280 / MECH-281 stay substrate_ceiling /
pending_retest_after_substrate -- this run weights NO claim (claim_ids=[], diagnostic).

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler; the axis-(b) measurement is a waking eval loop).

experiment_purpose: diagnostic
claim_ids: []  (substrate readiness; gates the axis-(b) P2->P3->P4 chain, weights no claim)
supersedes: V3-EXQ-625d (the saturated-threat tonic-immobility run that pinned z_harm_a
  ~6 with 0 crossings and self-routed substrate_not_ready; 625e keeps the joint composite
  + 569i conversion config + R1-R4 guards but recalibrates the axis-(b) threat -- 10x lower
  magnitude + pulsed duty cycle -- and trains on the 603q-stabilized harm pathway).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_625e_sd037_axis_b_phase1b_joint_composite_recalibrated"
QUEUE_ID = "V3-EXQ-625e"
CLAIM_IDS: List[str] = []  # substrate readiness; tags no claim
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "v3_exq_625d_sd037_axis_b_phase1b_joint_composite"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "SCAFFOLD_TRAINED_JOINT_COMPOSITE_569i_CONVERSION_AXIS_B_OVERLAY"

# --- Goal-pipeline / encoder dims (mirror 603n exactly). ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- Full scaffold curriculum budgets (mirror the validated 603n). ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4
HAZARD_STAGE_NUM_HAZARDS = 6  # 625e: 603q hazard headroom (was 4) so the base harm landscape is discriminative
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2  # the agent's PAG gate (603n-tuned); diagnostic crossing readout
HARM_PATHWAY_LR = 1e-3
# --- 603q harm-pathway STABILIZATION (decoupled encoder LR + LR warmup). ---
# Forms the base harm landscape on >=2/3 seeds (625d hit the 603p 1/3 seed-fragility).
HARM_PATHWAY_ENCODER_LR = 3e-4     # decoupled (lower) latent_stack encoder LR (603q)
HARM_PATHWAY_WARMUP_STEPS = 250    # linear LR warmup over the first N harm-pathway steps (603q)

# --- 569i-validated CONVERSION config (the committed-action-diversity lever). ---
CANDIDATE_SUMMARY_SOURCE = "e2_world_forward"
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_SHORTLIST_K = 3
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6

# --- SD-056 online contrastive training (mirror 569i harness). ---
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# --- Axis-(b) RECALIBRATED threat env overlay (625e; per 625d autopsy Section 7). ---
# Magnitude lowered ~10x from the 625b-era 0.2 so the TRAINED z_harm_a sits in a
# sub-saturating, oscillation-capable band (625d pinned z_harm_a ~6 = tonic immobility).
AXIS_B_INJECT_INTERVAL = 20
AXIS_B_INJECT_PROB = 0.7
AXIS_B_HAZARD_HARM = 0.02            # 625e: 10x lower than 625d's 0.2 (saturation fix)
AXIS_B_PROXIMITY_HARM_SCALE = 0.02  # 625e: 10x lower than 625d's 0.2 (saturation fix)
# --- Axis-(b) PULSED/TIME-VARYING threat duty cycle (625e; the C3a relief fix). ---
# scheduled_external_hazard toggled ON for PULSE_ON ticks then OFF for PULSE_OFF ticks
# so z_harm_a builds during ON and RELIEVES (limb-damage healing) during OFF -> can cross
# the 0.4 threshold DOWNWARD (625d's tonic regime had 0 above->below crossings).
AXIS_B_PULSE_ON_TICKS = 40
AXIS_B_PULSE_OFF_TICKS = 40

# --- Axis-(b) measurement phase budgets. ---
AXISB_WARMUP_EPS = 30   # SD-056 online training on the axis-(b) threat distribution
AXISB_MEASURE_EPS = 20  # frozen-policy z_harm_a crossing + readiness measurement

# --- C3 crossing definition (canonical; matches 625b/625c + plan section 2.3). ---
# The GATING crossing threshold is the DEFAULT PAG duration_input_threshold (0.4),
# the same z_harm_a level the 625b/625c sharpened C3a/C3b reference, for direct
# comparability with the predecessor. The agent's own (603n-tuned) PAG threshold
# (0.2) is reported as a SECONDARY non-gating diagnostic crossing readout.
CROSSING_Z_THRESHOLD = 0.4
CROSSING_Z_THRESHOLD_AGENT_PAG = PAG_DURATION_INPUT_THRESHOLD  # 0.2, diagnostic only

# --- Pre-registered gate floors (NOT derived from the run's own statistics). ---
ROUTE_RANGE_FLOOR = 0.01          # R3: modulatory_channel_route_range (V3-EXQ-662 statistic)
C1_PAIRWISE_DIST_FLOOR = 0.03     # R3: e2.world_forward per-candidate spread (SD-056 trained)
Z_HARM_A_NONZERO_FLOOR = 0.01     # R2: z_harm_a nonzero fraction floor
C3_SELECTED_ENTROPY_FLOOR = 0.3   # R4: committed-action class entropy floor (569i C_R1B)
MIN_SEEDS_FOR_PASS = 2            # of 3
MIN_FRACTION = 2.0 / 3.0
ZERO_FLOOR = 1e-9                 # below this -> counted as exactly zero


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, steps = 2, 2, 5, 5, 5, 30
    else:
        stage0, stage0b, p0, hazard, p1, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=2,  # scaffold p2 unused (we run our own axis-b measurement)
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        # 603q stabilization: decoupled (lower) encoder LR + LR warmup so the base
        # harm landscape forms discriminatively on >=2/3 seeds.
        scaffold_harm_pathway_encoder_lr=HARM_PATHWAY_ENCODER_LR,
        scaffold_harm_pathway_warmup_steps=HARM_PATHWAY_WARMUP_STEPS,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """603n full-curriculum config PLUS the 569i-validated conversion config and
    MECH-341 E3 score-diversity. The conversion config + MECH-341 are no-op-default
    flags; enabling them makes the scaffold-trained policy run under the
    committed-action-diversity arbitration that 569i validated."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        # ARC-065 SP-CEM (Layer A) main-path default (action-divergent pool)
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        # SHARED E3-side bias channel (consumes cand_world_summaries) -- on for the
        # conversion route to have a channel to route, as in 569i.
        use_lateral_pfc_analog=True,
        # SD-056 e2 action-conditional contrastive (trained online in the axis-b warmup)
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # MECH-279 PAG freeze-gate (603n-tuned).
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        # SD-058 / MECH-357 instrumental-avoidance gate.
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        # ===== MECH-341 E3 score-diversity (the plan's "MECH-341 lever ON") =====
        use_e3_score_diversity=True,
        use_e3_diversity_stratified_select=True,
        use_e3_diversity_entropy_bonus=True,
        # ===== 569i CONVERSION config (committed-action diversity reaches action) =====
        candidate_summary_source=CANDIDATE_SUMMARY_SOURCE,
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source="cand_world_summary",
        modulatory_channel_route_weight=1.0,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
    )
    cfg.latent.use_resource_encoder = True  # SD-015 (direct, not via from_dims)
    return cfg


# ---------------------------------------------------------------------------
# Measurement helpers (z_harm_a crossings + conversion readiness + SD-056 online)
# ---------------------------------------------------------------------------

def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _trajectory_first_action_class(traj) -> int:
    return int(traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item())


def _first_actions_K(candidates) -> torch.Tensor:
    rows = [traj.actions[:, 0, :].detach().reshape(-1) for traj in candidates]
    return torch.stack(rows, dim=0)


def _consumed_summaries(agent: REEAgent, candidates) -> Optional[torch.Tensor]:
    summ = agent._candidate_world_summaries(candidates)
    if summ is not None:
        return summ.detach()
    return None


def _mean_pairwise_l2(summ: torch.Tensor) -> float:
    summ = summ.detach()
    K = summ.shape[0]
    if K < 2:
        return 0.0
    total, n = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            total += float(torch.linalg.vector_norm(summ[i] - summ[j]))
            n += 1
    return total / max(n, 1)


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / n
            ent -= p * math.log(p)
    return ent


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen:
            seen[cls] = tup
        if len(seen) >= k:
            break
    if len(seen) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen.values())
    picked = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked:
            continue
        samples.append(tup)
        picked.add(id(tup))
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
        z_world_0=z0_K, actions=actions_K, z_world_1_targets=z1_K,
        simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    (SD056_WEIGHT * loss).backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


def compute_crossings(instant_vals: List[float], *, threshold: float) -> Dict[str, Any]:
    """Directional threshold crossings of the z_harm_a instant stream (the sharpened
    C3a/C3b from failure_autopsy_V3-EXQ-625b Section 8.3; identical convention to
    625c). A tick is 'above' when v > threshold (strict). A crossing is recorded
    whenever the above/below state flips between consecutive ticks. C3 requires BOTH
    directions present (>= 1 above->below AND >= 1 below->above)."""
    n_ab, n_ba, n_tot = 0, 0, 0
    prev_above: Optional[bool] = None
    for v in instant_vals:
        above = v > threshold
        if prev_above is not None and above != prev_above:
            n_tot += 1
            if prev_above and not above:
                n_ab += 1
            else:
                n_ba += 1
        prev_above = above
    return {
        "threshold": float(threshold),
        "n_ticks": int(len(instant_vals)),
        "n_above_to_below": int(n_ab),
        "n_below_to_above": int(n_ba),
        "n_total_transitions": int(n_tot),
        "dynamic_crossings_pass": bool(n_ab >= 1 and n_ba >= 1),
    }


def _build_axisb_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig, seed: int):
    """A DIM-MATCHED axis-(b) measurement env: the scaffold's own p2 env (same
    structural kwargs the shared agent was built/trained against -- reef-bipartite +
    SD-049 + limb_damage, guaranteeing world_obs_dim matches) with the axis-(b)
    sustained-threat overlay applied. All overlay knobs are plain instance attributes
    read LIVE in step()/proximity computation (causal_grid_world.py lines 491/508/
    535-538), so mutating them + reset() is correct and dim-safe."""
    env = _build_env(scaffold_cfg, "p2", seed=seed)
    env.scheduled_external_hazard_enabled = True
    env.scheduled_external_hazard_interval = AXIS_B_INJECT_INTERVAL
    env.scheduled_external_hazard_prob = AXIS_B_INJECT_PROB
    env.scheduled_external_hazard_adjacent_only = True
    env.hazard_harm = AXIS_B_HAZARD_HARM
    env.proximity_harm_scale = AXIS_B_PROXIMITY_HARM_SCALE
    return env


def _z_harm_a_norm(latent) -> float:
    za = getattr(latent, "z_harm_a", None)
    if za is None:
        return 0.0
    try:
        return float(za.norm().item())
    except Exception:
        return 0.0


def _axisb_measure(
    agent: REEAgent, env, seed: int, warmup_eps: int, measure_eps: int,
    steps_per_episode: int, ep_done_start: int, total_eps: int,
) -> Dict[str, Any]:
    """Axis-(b) phase: warmup_eps episodes train SD-056 online on the sustained-threat
    distribution, then measure_eps frozen-policy episodes measure z_harm_a crossings +
    the conversion-readiness statistics (route_range, cand_world_pairwise_dist,
    selected-action entropy)."""
    agent.train()  # SD-056 contrastive trains during the warmup; measure flips to eval()
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
        maxlen=TRANSITION_BUFFER_MAX
    )
    sample_rng = random.Random(seed)

    # Measurement accumulators (MEASURE phase only).
    z_harm_a_stream: List[float] = []      # z_harm_a_norm per measure tick (for crossings)
    z_harm_a_all: List[float] = []         # for nonzero-fraction
    route_ranges: List[float] = []
    pairwise_dists: List[float] = []
    selected_class_counts: Counter = Counter()
    authority_active_ticks = 0
    shortlist_sizes: List[float] = []
    n_measure_ticks = 0
    total_external_hazard_event_count = 0
    error_note: Optional[str] = None

    total_axisb_eps = warmup_eps + measure_eps
    ep_done = ep_done_start

    for ep in range(total_axisb_eps):
        is_measure = ep >= warmup_eps
        if is_measure:
            agent.eval()
        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0
        last_event_count = 0

        for _step in range(steps_per_episode):
            # 625e PULSED threat: toggle the SD-029 scheduled hazard ON/OFF by duty cycle
            # so z_harm_a builds during ON windows and RELIEVES (limb-damage healing)
            # during OFF windows -> can cross the 0.4 threshold in BOTH directions
            # (625d's tonic-saturated regime had 0 above->below crossings).
            _pulse_phase = tick_in_ep % (AXIS_B_PULSE_ON_TICKS + AXIS_B_PULSE_OFF_TICKS)
            env.scheduled_external_hazard_enabled = bool(_pulse_phase < AXIS_B_PULSE_ON_TICKS)

            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs(obs_dict, "harm_obs"),
                obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                obs_harm_history=_obs(obs_dict, "harm_history"),
            )

            # SD-056 transition capture (one-step-lagged z0 -> z1).
            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all()
                        and torch.isfinite(z1_obs).all()):
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

            # READINESS (b) / R3: e2.world_forward per-candidate spread (MEASURE phase).
            if is_measure and candidates and len(candidates) >= 2:
                actions_K = _first_actions_K(candidates).to(agent.device)
                z0 = latent.z_world.detach()
                with torch.no_grad():
                    pdist = float(agent.e2.cand_world_pairwise_dist(z0, actions_K).item())
                if math.isfinite(pdist):
                    pairwise_dists.append(pdist)

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                agent.update_z_goal(benefit_exposure=0.0, drive_level=max(0.0, 1.0 - energy))

            action = agent.select_action(candidates, ticks, temperature=1.0)

            # READINESS (a) / R3 IN-ARM route-range + C3 / R4 committed diversity (MEASURE).
            if is_measure:
                diag = agent.e3.last_score_diagnostics
                rr = float(diag.get("modulatory_channel_route_range", 0.0))
                if math.isfinite(rr):
                    route_ranges.append(rr)
                if bool(diag.get("modulatory_authority_active", False)):
                    authority_active_ticks += 1
                if bool(diag.get("modulatory_shortlist_active", False)):
                    sl = float(diag.get("modulatory_shortlist_size", 0))
                    if math.isfinite(sl):
                        shortlist_sizes.append(sl)

            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = f"non-finite action seed={seed} ep={ep} step={_step}"
                break

            if is_measure:
                # C3 z_harm_a instant stream + nonzero fraction (R2).
                z_norm = _z_harm_a_norm(latent)
                z_harm_a_stream.append(z_norm)
                z_harm_a_all.append(z_norm)
                # R4 committed-action diversity DV.
                selected_class_counts[int(action[0].argmax().item())] += 1
                n_measure_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 online contrastive step (WARMUP phase trains; MEASURE frozen).
            if (not is_measure) and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                _e2_contrastive_step(agent, transition_buffer, e2_opt, sample_rng)

            _, harm_signal, done, info, next_obs_dict = env.step(action)
            try:
                last_event_count = int((info or {}).get("external_hazard_event_count", 0))
            except (TypeError, ValueError):
                last_event_count = 0
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal), world_delta=None,
                    hypothesis_tag=False, owned=True,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action
            obs_dict = next_obs_dict
            tick_in_ep += 1
            if done:
                break

        if is_measure:
            total_external_hazard_event_count += last_event_count

        ep_done += 1
        if (ep + 1) % 10 == 0 or (ep + 1) == total_axisb_eps:
            phase = "axisb_measure" if is_measure else "axisb_warmup"
            print(f"  [train] {phase} seed={seed} ep {ep_done}/{total_eps}", flush=True)

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    n_all = len(z_harm_a_all)
    n_zero = sum(1 for v in z_harm_a_all if v < ZERO_FLOOR)
    nonzero_fraction = float(n_all - n_zero) / float(n_all) if n_all else 0.0

    crossings = compute_crossings(z_harm_a_stream, threshold=CROSSING_Z_THRESHOLD)
    crossings_agent_pag = compute_crossings(
        z_harm_a_stream, threshold=CROSSING_Z_THRESHOLD_AGENT_PAG
    )
    selected_entropy = _entropy_from_counts(dict(selected_class_counts))

    return {
        "ep_done": ep_done,
        "error_note": error_note,
        "n_measure_ticks": int(n_measure_ticks),
        "external_hazard_event_count": int(total_external_hazard_event_count),
        "z_harm_a_nonzero_fraction": round(nonzero_fraction, 6),
        "z_harm_a_mean": round(_mean(z_harm_a_all), 6),
        "z_harm_a_max": round(max(z_harm_a_all) if z_harm_a_all else 0.0, 6),
        "modulatory_channel_route_range_mean": round(_mean(route_ranges), 6),
        "modulatory_channel_route_range_max": round(max(route_ranges) if route_ranges else 0.0, 6),
        "modulatory_authority_active_ticks": int(authority_active_ticks),
        "modulatory_shortlist_size_mean": round(_mean(shortlist_sizes), 6),
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        "selected_action_class_entropy": round(selected_entropy, 6),
        "selected_class_counts": {str(k): int(v) for k, v in selected_class_counts.items()},
        # C3 (gating, threshold 0.4) + agent-PAG diagnostic (0.2).
        "crossings_summary": crossings,
        "crossings_summary_agent_pag": crossings_agent_pag,
    }


def _aborted_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "reached_p1": False, "p1_survival_pass": False,
        "harm_pathway_n_train_steps": 0, "harm_eval_range": 0.0,
        "external_hazard_event_count": 0, "z_harm_a_nonzero_fraction": 0.0,
        "z_harm_a_mean": 0.0, "z_harm_a_max": 0.0,
        "modulatory_channel_route_range_mean": 0.0, "cand_world_pairwise_dist_mean": 0.0,
        "selected_action_class_entropy": 0.0, "n_measure_ticks": 0,
        "c3a_above_to_below": 0, "c3b_below_to_above": 0, "c3_dynamic_crossings_pass": False,
        "r1_curriculum_fired": False, "r2_z_harm_a_nonzero": False,
        "r3_conversion_operative": False, "r4_committed_diversity": False,
        "crossings_summary": {}, "crossings_summary_agent_pag": {},
        "error_note": f"aborted_at={stage}:{reason}",
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")

    probe_env = _build_env(scaffold_cfg, "p2", seed=seed)
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0

    # --- Scaffold curriculum: Stage-0 -> Stage-0b -> P0 -> Stage-H -> P1. ---
    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
          f" z_goal_peak={s0.z_goal_norm_peak:.4f}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return _aborted_record(seed, "stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
          f" retention={s0b.retention_ratio:.3f}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return _aborted_record(seed, "stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return _aborted_record(seed, "p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    harm_diag = dict(getattr(hz, "harm_discriminativeness", {}) or {})
    harm_pathway_diag = dict(getattr(hz, "harm_pathway_diag", {}) or {})
    harm_eval_range = float(harm_diag.get("harm_eval_range", 0.0))
    harm_train_steps = int(harm_pathway_diag.get("n_train_steps", 0))
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}"
          f" harm_eval_range={harm_eval_range:.4f}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        rec = _aborted_record(seed, "hazard", hz.abort_reason)
        rec["harm_pathway_n_train_steps"] = harm_train_steps
        rec["harm_eval_range"] = harm_eval_range
        return rec

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

    # --- Axis-(b) sustained-threat measurement on a dim-matched scaffold-style env. ---
    if dry_run:
        warmup_eps, measure_eps = 2, 2
    else:
        warmup_eps, measure_eps = AXISB_WARMUP_EPS, AXISB_MEASURE_EPS
    axisb_env = _build_axisb_env(scaffold_cfg, seed)
    meas = _axisb_measure(
        agent, axisb_env, seed, warmup_eps, measure_eps,
        scaffold_cfg.scaffold_steps_per_episode, ep_done_start=done, total_eps=total_eps,
    )

    cr = meas["crossings_summary"]
    c3a = int(cr.get("n_above_to_below", 0))
    c3b = int(cr.get("n_below_to_above", 0))
    c3_pass = bool(c3a >= 1 and c3b >= 1)

    # Per-seed non-vacuity readiness flags.
    r1 = bool(meas["external_hazard_event_count"] > 0)
    r2 = bool(meas["z_harm_a_nonzero_fraction"] >= Z_HARM_A_NONZERO_FLOOR)
    r3 = bool(
        meas["modulatory_channel_route_range_mean"] > ROUTE_RANGE_FLOOR
        and meas["cand_world_pairwise_dist_mean"] > C1_PAIRWISE_DIST_FLOOR
    )
    r4 = bool(meas["selected_action_class_entropy"] > C3_SELECTED_ENTROPY_FLOOR)

    print(f"verdict: {'PASS' if c3_pass else 'FAIL'} seed={seed}"
          f" c3a={c3a} c3b={c3b} c3={c3_pass}"
          f" r1={r1} r2={r2} r3={r3} r4={r4}"
          f" route_range={meas['modulatory_channel_route_range_mean']:.4f}"
          f" sel_entropy={meas['selected_action_class_entropy']:.4f}"
          f" z_harm_a_max={meas['z_harm_a_max']:.4f}", flush=True)

    rec = {
        "seed": seed, "aborted_at": None, "abort_reason": "",
        "reached_p1": True, "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "harm_pathway_n_train_steps": harm_train_steps,
        "harm_eval_range": harm_eval_range,
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "c3a_above_to_below": c3a,
        "c3b_below_to_above": c3b,
        "c3_dynamic_crossings_pass": c3_pass,
        "r1_curriculum_fired": r1,
        "r2_z_harm_a_nonzero": r2,
        "r3_conversion_operative": r3,
        "r4_committed_diversity": r4,
    }
    rec.update(meas)
    return rec


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


# --- Precondition-DECLARATION helpers. No scientific predicate changes; these exist only
# so the adjudication gate can reproduce the shipped `met`. ---
#
# build_experiment_indexes._precondition_unmet RECOMPUTES each
# interpretation.preconditions[].met from that entry's own (measured, threshold) pair and
# treats the recompute as AUTHORITATIVE over the author's value. A k-of-n predicate
# ("holds on >= 2/3 seeds") therefore has to be reported as a statistic whose comparison
# against the threshold reproduces the k-of-n COUNT exactly. A max() over seeds does NOT:
# it is strictly LOOSER than "on >= k seeds" (a single good seed satisfies it), so the
# recompute silently CLEARS a genuinely-failed premise -- the dangerous direction, and the
# one these entries previously shipped.
def _min_count(n: int) -> int:
    """Smallest seed COUNT k for which `_frac` clears MIN_FRACTION -- the integer form of
    the k-of-n rule at this n (3 seeds -> 2, 1 dry-run seed -> 1). Derived by re-running
    _frac's OWN division rather than math.ceil(MIN_FRACTION * n), so the two agree
    bit-for-bit instead of risking an off-by-one on the float boundary."""
    for k in range(n + 1):
        if (float(k) / float(n) if n else 0.0) >= MIN_FRACTION:
            return k
    return n + 1


def _kth_best(values: List[float], k: int) -> float:
    """The k-th LARGEST value. For a per-seed FLOOR predicate, `_kth_best(vals, k) <op>
    floor` is EXACTLY "at least k seeds satisfied `value <op> floor`", so it reproduces
    the k-of-n `met` under the indexer's recompute where a max() cannot."""
    if not values or k < 1 or k > len(values):
        return 0.0
    return float(sorted(values, reverse=True)[k - 1])


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS

    if dry_run:
        scaffold_total = 2 + 2 + 5 + 5 + 5
        total_eps = scaffold_total + 2 + 2
    else:
        scaffold_total = STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET + P1_BUDGET
        total_eps = scaffold_total + AXISB_WARMUP_EPS + AXISB_MEASURE_EPS

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)

    # --- C3 (load-bearing): dynamic crossings (C3a AND C3b per seed) on >= 2/3 seeds. ---
    c3_flags = [r["c3_dynamic_crossings_pass"] for r in per_seed]
    c3_frac = _frac(c3_flags)
    c3_pass = c3_frac >= MIN_FRACTION

    # --- Pre-registered non-vacuity preconditions. ---
    r1_all = all(r["r1_curriculum_fired"] for r in per_seed)  # 3/3 required
    r2_frac = _frac([r["r2_z_harm_a_nonzero"] for r in per_seed])
    r3_frac = _frac([r["r3_conversion_operative"] for r in per_seed])
    r4_frac = _frac([r["r4_committed_diversity"] for r in per_seed])
    r2_pass = r2_frac >= MIN_FRACTION
    r3_pass = r3_frac >= MIN_FRACTION
    r4_pass = r4_frac >= MIN_FRACTION
    preconditions_met = bool(r1_all and r2_pass and r3_pass and r4_pass)

    route_range_max = max((r.get("modulatory_channel_route_range_mean", 0.0) for r in per_seed), default=0.0)
    pairwise_max = max((r.get("cand_world_pairwise_dist_mean", 0.0) for r in per_seed), default=0.0)
    entropy_max = max((r.get("selected_action_class_entropy", 0.0) for r in per_seed), default=0.0)

    # --- Recomputable forms of the R2/R3/R4 k-of-n aggregates, for the precondition
    # DECLARATIONS below. The *_max statistics above stay exactly as they are and keep
    # feeding `readiness` / criteria_non_degenerate; they are simply no longer the number
    # the adjudicator recomputes `met` from. ---
    k_seeds = _min_count(n)
    z_harm_nonzero_kth = _kth_best(
        [float(r.get("z_harm_a_nonzero_fraction", 0.0)) for r in per_seed], k_seeds)
    entropy_kth = _kth_best(
        [float(r.get("selected_action_class_entropy", 0.0)) for r in per_seed], k_seeds)
    route_range_kth = _kth_best(
        [float(r.get("modulatory_channel_route_range_mean", 0.0)) for r in per_seed], k_seeds)
    # R3's per-seed flag is a CONJUNCTION (route_range > floor AND cand_world pairwise
    # dist > floor), and a count over a conjunction does not distribute into per-leg
    # counts -- so NO single route-range statistic can reproduce r3_pass. The count of
    # satisfying seeds is reported instead; route_range_kth is carried alongside as a
    # non-bound diagnostic.
    n_r3_operative = sum(1 for r in per_seed if r["r3_conversion_operative"])

    # --- Routing (diagnostic adjudication gate). ---
    if not preconditions_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
    elif c3_pass:
        outcome = "PASS"
        readiness_route = "axis_b_oscillation_demonstrated_p1b_clears"
    else:
        outcome = "FAIL"
        readiness_route = "residual_no_oscillation_despite_diverse_committed_policy"

    # --- Diagnostic adjudication structures. ---
    # R3/R4 are RANGE/entropy statistics (the same-statistic discipline: the C3
    # crossing criterion can only fire if committed-action diversity reaches action,
    # which R3 route-range + R4 selected-entropy assert on a RANGE/ENTROPY basis).
    preconditions = [
        {
            "name": "r1_sustained_threat_curriculum_fired",
            "kind": "readiness",
            "description": "SD-029 scheduled_external_hazard must fire on the axis-(b) "
                           "measurement env (external_hazard_event_count > 0 in 3/3 seeds). "
                           "Below -> env overlay mis-applied -> substrate_not_ready_requeue.",
            "control": "axis-(b) overlay on a dim-matched scaffold p2 env "
                       "(scheduled_external_hazard interval=20 prob=0.7 adjacent_only).",
            "measured": float(min((r["external_hazard_event_count"] for r in per_seed), default=0)),
            "threshold": 1.0,
            # FLOOR-shaped, INCLUSIVE, and an ALL-seeds rule: `met` is
            # `all(count > 0)` over an integer count, which for the min over seeds is
            # exactly `min >= 1`. Declared rather than left to the indexer's default so
            # the boundary is explicit (the 2026-06-07 V3-EXQ-648a/649 directionality bug).
            "comparator": ">=",
            "direction": "lower",
            "met": bool(r1_all),
        },
        {
            "name": "r2_z_harm_a_nonzero",
            "kind": "readiness",
            "description": "z_harm_a NONZERO fraction must clear the floor on >= 2/3 seeds "
                           "(the affective stream is populated; a flat-zero z_harm_a cannot "
                           "cross). Below -> affective stream not engaged -> requeue.",
            "control": "use_affective_harm_stream=True + scaffold_feed_harm_stream=True; "
                       "axis-(b) hazard_harm=0.2 + proximity_harm_scale=0.2 lift the signal.",
            # k-of-n FLOOR, INCLUSIVE: the per-seed predicate is
            # `z_harm_a_nonzero_fraction >= Z_HARM_A_NONZERO_FLOOR` and `met` is that on
            # >= 2/3 seeds, so the k-th LARGEST per-seed value clearing the floor is
            # exactly "at least k seeds cleared it". This entry previously reported the
            # MAX over seeds, which is strictly LOOSER than the k-of-n rule -- one good
            # seed satisfies it -- so the indexer's authoritative recompute would have
            # silently CLEARED a failed premise. The max is preserved below as a
            # NON-BOUND diagnostic (extra keys are ignored by the recompute).
            "measured": float(z_harm_nonzero_kth),
            "threshold": float(Z_HARM_A_NONZERO_FLOOR),
            "comparator": ">=",
            "direction": "lower",
            "seeds_required": int(k_seeds),
            "observed_max_z_harm_a_nonzero_fraction": float(
                max((r["z_harm_a_nonzero_fraction"] for r in per_seed), default=0.0)),
            "met": bool(r2_pass),
        },
        {
            "name": "r3_conversion_route_range_operative",
            "kind": "readiness",
            "description": "The 569i conversion is OPERATIVE here: the IN-ARM "
                           "modulatory_channel_route_range (the V3-EXQ-662 RANGE statistic, "
                           "read LIVE at the select tick) > floor on >= 2/3 seeds. SAME RANGE "
                           "statistic 569i gates on. Below -> SD-056 under-trained / routing "
                           "not wired -> requeue, NEVER a verdict.",
            "control": "569i-validated conversion config ON + SD-056 trained online on the "
                       "axis-(b) env; positive control = e2_world_forward source with genuine "
                       "per-candidate range.",
            # COUNT-shaped, INCLUSIVE floor: `met` is `r3_frac >= MIN_FRACTION`, i.e.
            # `n_r3_operative >= k_seeds` -- a COUNT of seeds. This entry previously
            # reported the MAX route-range over seeds against ROUTE_RANGE_FLOOR, which is
            # strictly LOOSER than "on >= 2/3 seeds", so the indexer's authoritative
            # recompute silently CLEARED a premise the author had marked FAILED (625d
            # shipped 0.084518 vs a 0.01 floor with met=False; 625e 0.197728).
            # A count is used rather than the k-th-largest route range because the
            # per-seed flag is a CONJUNCTION (route_range > ROUTE_RANGE_FLOOR AND
            # cand_world_pairwise_dist > C1_PAIRWISE_DIST_FLOOR) and a count over a
            # conjunction does not distribute into per-leg counts -- no single
            # route-range statistic CAN reproduce r3_pass. Both route-range aggregates
            # are preserved below as NON-BOUND diagnostics.
            "measured": float(n_r3_operative),
            "threshold": float(k_seeds),
            "comparator": ">=",
            "direction": "lower",
            "observed_route_range_floor": float(ROUTE_RANGE_FLOOR),
            "observed_kth_best_route_range": float(route_range_kth),
            "observed_max_route_range": float(route_range_max),
            "met": bool(r3_pass),
        },
        {
            "name": "r4_committed_action_diversity_entropy",
            "kind": "readiness",
            "description": "The committed policy is NOT monostrategy-locked: selected-action "
                           "class ENTROPY > floor on >= 2/3 seeds (the 569i C_R1B floor). This "
                           "is the diversity that MUST reach committed action for z_harm_a to "
                           "be ABLE to oscillate -- a monostrategy-locked policy makes 0 "
                           "crossings for a CONVERSION reason, not an env-capability reason -> "
                           "requeue. ENTROPY statistic, matching the diversity the C3 crossing "
                           "criterion presupposes.",
            "control": "scaffold-trained competent policy + 569i conversion config; positive "
                       "control = 569i ARM_1 selected-action entropy strict-above controls.",
            # k-of-n FLOOR, STRICT: the per-seed predicate is
            # `selected_action_class_entropy > C3_SELECTED_ENTROPY_FLOOR` (strict) on
            # >= 2/3 seeds, so the k-th LARGEST per-seed entropy strictly above the floor
            # is exactly "at least k seeds cleared it". Same latent max()-is-looser defect
            # as R2/R3; the max is preserved as a NON-BOUND diagnostic.
            "measured": float(entropy_kth),
            "threshold": float(C3_SELECTED_ENTROPY_FLOOR),
            "comparator": ">",
            "direction": "lower",
            "seeds_required": int(k_seeds),
            "observed_max_selected_entropy": float(entropy_max),
            "met": bool(r4_pass),
        },
    ]
    criteria_non_degenerate = {
        "C3_dynamic_crossings": bool(n >= 2 and _frac([r["reached_p1"] for r in per_seed]) >= MIN_FRACTION),
        "R1_curriculum": bool(n >= 1),
        "R3_conversion": bool(pairwise_max > 0.0 or route_range_max > 0.0),
        "R4_diversity": bool(n >= 2),
    }
    criteria = [
        {"name": "C3_dynamic_crossings_2of3", "load_bearing": True, "passed": bool(c3_pass)},
    ]

    gate = {
        "c3_dynamic_crossings_pass": bool(c3_pass),
        "c3_fraction": float(c3_frac),
        "per_seed_c3": c3_flags,
        "per_seed_c3a_above_to_below": [r["c3a_above_to_below"] for r in per_seed],
        "per_seed_c3b_below_to_above": [r["c3b_below_to_above"] for r in per_seed],
        "crossing_z_threshold_gating": CROSSING_Z_THRESHOLD,
        "crossing_z_threshold_agent_pag_diagnostic": CROSSING_Z_THRESHOLD_AGENT_PAG,
        "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
        "min_fraction": MIN_FRACTION,
    }
    readiness = {
        "preconditions_met": preconditions_met,
        "r1_curriculum_fired_all3": bool(r1_all),
        "r2_z_harm_a_nonzero_frac": float(r2_frac),
        "r3_conversion_operative_frac": float(r3_frac),
        "r4_committed_diversity_frac": float(r4_frac),
        "route_range_floor": ROUTE_RANGE_FLOOR,
        "cand_world_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
        "z_harm_a_nonzero_floor": Z_HARM_A_NONZERO_FLOOR,
        "selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
        "route_range_max_over_seeds": float(route_range_max),
        "cand_world_pairwise_dist_max_over_seeds": float(pairwise_max),
        "selected_entropy_max_over_seeds": float(entropy_max),
    }

    print(f"[{EXPERIMENT_TYPE}] C3 dynamic-crossings frac={c3_frac:.2f} pass={c3_pass}"
          f" | preconditions_met={preconditions_met}"
          f" (r1={r1_all} r2={r2_pass} r3={r3_pass} r4={r4_pass})"
          f" -> outcome={outcome} route={readiness_route}", flush=True)

    return {
        "outcome": outcome,
        "c3_pass": c3_pass,
        "preconditions_met": preconditions_met,
        "c3_gate": gate,
        "readiness_gate": readiness,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
        },
        "per_seed": per_seed,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    result = run_experiment(dry_run=dry_run)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": "non_contributory",  # diagnostic; tags no claim
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler + waking axis-(b) eval)",
        "condition": CONDITION_LABEL,
        "design_note": "JOINT-COMPOSITE Phase-1b RECALIBRATED (625e): scaffold-trained competent "
                       "policy (603n full curriculum + 603q-stabilized harm pathway: encoder_lr "
                       "3e-4 + warmup 250 + Stage-H num_hazards=6) + 569i-validated conversion "
                       "config (top_k shortlist k=3 + authority gain=2.0 std-basis + e2_world_forward "
                       "source) + MECH-341 + SD-056 (online on axis-(b)) measured for z_harm_a dynamic "
                       "crossings under a RECALIBRATED axis-(b) threat: magnitude lowered ~10x "
                       "(hazard_harm/proximity_harm_scale 0.2 -> 0.02) AND time-varying/PULSED "
                       "(scheduled_external_hazard duty-cycle ON 40 / OFF 40 ticks) so the TRAINED "
                       "z_harm_a sits in a sub-saturating oscillation-capable band and can cross 0.4 "
                       "BOTH ways. Supersedes 625d (saturated-threat tonic-immobility regime, z_harm_a "
                       "~6, 0 crossings, self-routed substrate_not_ready). PASS = C3 dynamic crossings "
                       ">=2/3 with all four non-vacuity preconditions met. A non-vacuity miss "
                       "self-routes substrate_not_ready_requeue (NEVER a weakens; claim_ids=[]).",
        "axis_b_env_overlay": {
            "base": "scaffold p2 env (dim-matched: reef-bipartite + SD-049 + limb_damage)",
            "recalibrated_vs_625d": "magnitude 10x lower (0.2->0.02) + pulsed threat (per 625d autopsy S7)",
            "scheduled_external_hazard_enabled": "pulsed (duty cycle)",
            "scheduled_external_hazard_interval": AXIS_B_INJECT_INTERVAL,
            "scheduled_external_hazard_prob": AXIS_B_INJECT_PROB,
            "scheduled_external_hazard_adjacent_only": True,
            "pulse_on_ticks": AXIS_B_PULSE_ON_TICKS,
            "pulse_off_ticks": AXIS_B_PULSE_OFF_TICKS,
            "hazard_harm": AXIS_B_HAZARD_HARM,
            "proximity_harm_scale": AXIS_B_PROXIMITY_HARM_SCALE,
        },
        "conversion_config_569i": {
            "candidate_summary_source": CANDIDATE_SUMMARY_SOURCE,
            "use_modulatory_channel_routing": True,
            "modulatory_channel_route_source": "cand_world_summary",
            "use_modulatory_selection_authority": True,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
            "use_modulatory_shortlist_then_modulate": True,
            "modulatory_shortlist_mode": MODULATORY_SHORTLIST_MODE,
            "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
            "mech341_use_e3_score_diversity": True,
            "sd056_e2_action_contrastive_enabled": True,
        },
        "pre_registered_gates": {
            "c3_definition": "per seed: n_above_to_below >= 1 AND n_below_to_above >= 1 of "
                             "z_harm_a across the duration_input_threshold (0.4, canonical); "
                             "EXPERIMENT C3 PASS on >= 2/3 seeds (the sharpened C3a/C3b from "
                             "failure_autopsy_V3-EXQ-625b Section 8.3).",
            "crossing_z_threshold_gating": CROSSING_Z_THRESHOLD,
            "crossing_z_threshold_agent_pag_diagnostic": CROSSING_Z_THRESHOLD_AGENT_PAG,
            "r1_curriculum_fired": "external_hazard_event_count > 0 in 3/3 seeds",
            "r2_z_harm_a_nonzero_floor": Z_HARM_A_NONZERO_FLOOR,
            "r3_route_range_floor": ROUTE_RANGE_FLOOR,
            "r3_cand_world_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
            "r4_selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
            "min_fraction": MIN_FRACTION,
            "pass_rule": "PASS = C3 (>=2/3) AND preconditions_met (R1 3/3 AND R2/R3/R4 each >=2/3); "
                         "preconditions unmet -> substrate_not_ready_requeue; preconditions met "
                         "but C3 fail -> residual_no_oscillation (route /failure-autopsy).",
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "axisb_warmup_eps": AXISB_WARMUP_EPS,
            "axisb_measure_eps": AXISB_MEASURE_EPS, "train_steps": TRAIN_STEPS,
            "scaffold_train_harm_pathway": True, "scaffold_feed_harm_stream": True,
            "scaffold_harm_pathway_encoder_lr": HARM_PATHWAY_ENCODER_LR,
            "scaffold_harm_pathway_warmup_steps": HARM_PATHWAY_WARMUP_STEPS,
            "hazard_stage_num_hazards": HAZARD_STAGE_NUM_HAZARDS,
            "pag_theta_freeze": PAG_THETA_FREEZE,
            "pag_duration_input_threshold": PAG_DURATION_INPUT_THRESHOLD,
        },
    }
    manifest.update(result)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_res["manifest_path"],
        )
