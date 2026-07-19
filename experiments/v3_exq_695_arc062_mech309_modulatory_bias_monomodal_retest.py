#!/opt/local/bin/python3
"""
V3-EXQ-695 -- ARC-062 / MECH-309 modulatory-bias monomodal-collapse RETEST on the
NOW-IMPLEMENTED modulatory-bias / rule-apprehension channel, ported onto the
569i-VALIDATED TOP-K SHORTLIST conversion.

SUPERSEDES V3-EXQ-654g (non_contributory 2026-06-19, interpretation label
shared_selection_authority_conversion_ceiling_route_implement_substrate). 654g
tested exactly {MECH-309, ARC-062} on the matured CRF + 569i top-k shortlist
stack and routed into the shared MECH-439 F-dominance conversion ceiling. 695
re-asks the SAME scientific question -- does the implemented modulatory-bias /
rule-apprehension channel recover committed-action diversity (break monomodal
collapse) -- on the same matched stack, with the conversion-ceiling entanglement
made an EXPLICIT, same-statistic P0 readiness gate (below). This is a full
copy-and-modify treatment of 654g; everything is re-derived, no stale tags.

CLAIMS UNDER TEST (substrate-landing retest of two substrate_ceiling claims):
  ARC-062 -- rule-apprehension architectural slot (weak reading): a non-Bayesian
             rule-creator at the policy-granularity layer (the gated-policy heads +
             SD-033a LateralPFCAnalog + the ARC-063 CandidateRuleField crf_source).
  MECH-309 -- monomodal policy collapse is the equilibrium of a parametric-policy
             agent WITHOUT a rule substrate; with the modulatory-bias /
             rule-apprehension channel now implemented, does it recover committed-
             action-class diversity above the collapsed controls.
claim_ids = [MECH-309, ARC-062]; experiment_purpose = evidence; multi-claim ->
evidence_direction_per_claim emitted.

THE 569i-VALIDATED TOP-K SHORTLIST CONVERSION (the matched-stack CONSTANT on BOTH
arms): F (the raw primary harm/goal scores) filters the candidate pool to a SMALL
fixed top-k F-best near-tie set (use_modulatory_shortlist_then_modulate=True +
modulatory_shortlist_mode="top_k" + modulatory_shortlist_k=3); the modulatory
channel (the lateral_pfc CRF rule-bias + the routed cand_world_summary range) then
ARBITRATES the winner WITHIN that k-set (argmin committed). A SMALL fixed k gives an
eligible set whose MEMBERSHIP rotates with state, so argmin-of-the-routed-channel
within the rotating small set produces committed-action diversity that reflects
genuine per-candidate structure WITHOUT the modulatory channel having to
out-magnitude the F-dominated primary (V3-EXQ-571: F ~ 88-89%% of the committed-
selection variance). SAFETY preserved at any internal strength -- only the k F-best
are eligible.

ENTANGLEMENT WARNING (the reason for the explicit P0 gate)
----------------------------------------------------------
Per substrate_queue f_dominance_conversion_ceiling (status
..._build_rank_preserving_F_to_eligibility_demotion_MECH448_lead...; ready=False),
ARC-062/MECH-309's residual ceiling IS the shared MECH-439 F-dominance conversion
ceiling, now MID-BUILD (the MECH-448 rank-preserving F->eligibility demotion lever
landed today; its falsifier V3-EXQ-689d is queued on ree-cloud-3). 654g already
routed these two claims into this exact build. So a naive retest may LAND on the
mid-build ceiling and self-route. 695 therefore makes the lifting-channel
non-degeneracy a PRE-REGISTERED, SAME-STATISTIC P0 readiness gate:

  P0 READINESS (load-bearing, route-RANGE; the V3-EXQ-643 same-statistic rule):
  the load-bearing C2 criterion routes on committed-class-entropy LIFT produced by
  the modulatory route channel; a range-gated criterion MUST assert RANGE on a
  positive control. So the gate measures the SAME route-range statistic the
  conversion authority routes on -- e3.last_score_diagnostics
  ["modulatory_channel_route_range"] -- captured per-tick on the ARM_ON positive
  control during P2, plus the GAP-A consumed-summary spread (e2-pairwise-divergence)
  and the matured-pool differentiation, and the propagation non-vacuity. If ANY of
  these is degenerate (route_range / e2-pairwise-divergence / modulatory
  cross-candidate RANGE below floor), the run self-routes
  substrate_not_ready_requeue (NEVER a substrate-VERDICT label, NEVER a weakens) --
  it landed on the mid-build conversion ceiling, not on a fair MECH-309/ARC-062 test.

THIS RETEST IS DISTINCT FROM 689d (THE 689-FAMILY MECH-439 LEVERS ARE OFF)
--------------------------------------------------------------------------
695 is about the modulatory / rule-apprehension channel (ARC-062/MECH-309), NOT the
F-dominance demotion lever 689d tests. All MECH-439 conflict-grade / F-eligibility-
demotion levers are HELD OFF on every arm (use_f_eligibility_demotion=False,
modulatory_shortlist_conflict_graded=False, use_gap_scaled_commit_temperature=False,
use_natural_commit_urgency_release=False) so 695 stays a clean single-variable
modulatory-channel retest and does not double-count the 689d lever.

Single-variable arm contrast (matched stack; the ONLY swept variable is
use_candidate_rule_field, which auto-sets use_candidate_rule_source on the agent):
  ARM_OFF (baseline / control): use_candidate_rule_field=False -> LateralPFCAnalog
           rule_state is the LEGACY delta_proj(z_delta) + world_pool_weight *
           world_proj(z_world) EMA source -- the 543/598b COLLAPSED rule_state. The
           modulatory route channel + 569i top-k are armed identically, but the
           rule-apprehension creator is absent -> the collapsed-control arm.
  ARM_ON  (modulatory-bias / rule-apprehension ON): use_candidate_rule_field=True
           (+ crf_persist=True) -> LateralPFCAnalog consumes crf_source, the field's
           DIFFERENTIATED active-rule-stack rule_state, MATURED across episodes.

MATCHED-NOISE PROPOSER CONTROL (C_R1B non-vacuity, the 569i bar): committed-class
entropy is read against the PROPOSER first-action distribution (the collapsed
control the channel must beat) AND the ARM_OFF arm. The load-bearing criterion is:
does the channel recover committed-action-class diversity STRICTLY ABOVE the
collapsed controls (break monomodal collapse).

Dependent variable -- COMMITTED-CLASS diversity (mechanistically matched)
-------------------------------------------------------------------------
CODE-CONFIRMED (agent._candidate_world_summaries + lateral_pfc.compute_bias): the
per-candidate summaries the bias channels consume are keyed on the candidate's
FIRST action (e2.world_forward(z0, a_first)); compute_bias broadcasts one rule_state
across K. So the rule-creator moves WHICH CLASS is committed (the committed-class
axis), NOT within-class representative selection. PRIMARY DV = committed-class
entropy. Within-class-representative entropy is a SECONDARY NEGATIVE CONTROL
(expected ~null, confirming the bias is class-keyed).

Phases / budget
---------------
P0 (200 ep, e2 TRAINED online via SD-056 contrastive; bias head NOT trained; field
   matures across episodes via crf_persist + the 666c maintenance levers).
P1 (90 ep, encoder FROZEN, bias head TRAINED via outcome-coupled REINFORCE; field
   continues to mature + maintain): the GAP-D trained-bias-head window. No measurement.
P2 (60 ep, all FROZEN -- e2 + bias head; field persists + maintains; instrumentation
   ON): the behavioural measurement window + route-range readiness capture.
Budget: 2 arms x 3 seeds x 350 ep x 200 steps = 420k steps total (~10 h). All
comparisons within-seed -> machine_affinity "any" (we suggest ree-cloud-3).

Pre-registered acceptance criteria
----------------------------------
  P0 READINESS / non-vacuity (unmet -> substrate_not_ready_requeue, NEVER a weakens):
     R1 committed-class axis exercisable: frac_pre_ge2 > FRAC_PRE_GE2_FLOOR on a
        majority (>= 2/3) of seeds in BOTH arms.
     R2 GAP-A e2-divergence real: consumed_summary_pairwise_dist_mean >
        CONSUMED_SPREAD_FLOOR (and bounded) on a majority of seeds in BOTH arms.
     R3 ARM_ON manipulation live AND MATURED: crf_frac_active >= CRF_FRAC_ACTIVE_FLOOR
        AND >= CRF_MIN_MINTED distinct rules minted, on a majority of ARM_ON seeds.
     R4 PROPAGATION non-vacuity: paired |mean_lateral_pfc_bias_abs(ARM_ON) -
        mean_lateral_pfc_bias_abs(ARM_OFF)| > PROP_NONVAC_FLOOR on a majority of seeds.
     R5 MODULATORY ROUTE-RANGE non-vacuity (the SAME-statistic gate; the
        V3-EXQ-643 rule): ARM_ON mean modulatory_channel_route_range >
        ROUTE_RANGE_FLOOR on a majority of ARM_ON seeds. This is the SAME route-range
        statistic the conversion authority routes on -- if it is below floor the
        lifting channel is degenerate (landed on the mid-build MECH-439 ceiling) and
        the C2 lift cannot be interpreted.
  C2 (PRIMARY -- the load-bearing falsifier): paired-by-seed committed_class_entropy
     lift of ARM_ON over BOTH the ARM_OFF arm AND the proposer first-action entropy
     of at least C2_LIFT_MARGIN_NATS on a majority (>= 2/3) of seeds. Tag load_bearing.

Overall outcome (THREE branches; NO weakens -- the conversion ceiling is open)
------------------------------------------------------------------------------
  PASS  = P0-readiness (incl route-range R5) AND C2 (committed-class lift over both
          controls) -> supports MECH-309 + ARC-062.
  FAIL (readiness holds, C2 fails) = the matured + differentiated rule pool's bias
          REACHES committed action (R4) AND carries route range (R5) but does NOT lift
          committed-class diversity EVEN under the validated top-k shortlist ->
          DEEPER SHARED F-dominance CONVERSION CEILING (MECH-439; 569g/682). NOT a
          falsification. non_contributory; route to /implement-substrate (the
          F-dominance rebalance, MECH-448 lead). Do NOT weaken MECH-309 / ARC-062.
  FAIL (readiness fails) = substrate not exercisable / not matured / propagation
          vacuous / ROUTE-RANGE degenerate -> substrate_not_ready_requeue; re-queue.
          Do NOT weaken.

See REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md (GAP-B),
REE_assembly/evidence/planning/substrate_queue.json (f_dominance_conversion_ceiling),
REE_assembly/evidence/planning/conversion_ceiling_phase0_synthesis_2026-06-18.md
(MECH-439 F-dominance live root the off-ramp routes to),
experiments/v3_exq_654g_arc062_gapb_rule_apprehension_behavioural_falsifier.py (the
direct 2-claim predecessor this supersedes),
experiments/v3_exq_569i_gapa_conversion_topk_shortlist_falsifier.py (the 569i top-k
shortlist conversion config ported here),
experiments/v3_exq_689a_mech439_conflict_grade_gapblind_falsifier.py (the GAP-A
conversion config + readiness gates),
ree-v3/CLAUDE.md (modulatory-bias-selection-authority TOP-K shortlist amend +
ARC-063 crf-availability-maintenance + ARC-062 GAP-A/B/C/D + SD-056 entries).
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
from experiments._lib.arm_fingerprint import compute_arm_fingerprint, reset_all_rng
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_695_arc062_mech309_modulatory_bias_monomodal_retest"
QUEUE_ID = "V3-EXQ-695"
SUPERSEDES = "V3-EXQ-654g"
CLAIM_IDS: List[str] = ["MECH-309", "ARC-062"]
EXPERIMENT_PURPOSE = "evidence"

# --- CRF-gate calibration amend levers (the now-working CRF stack; 654f C1 fully met;
# landed ree-v3 main 42895f6, 2026-06-17). Inert in ARM_OFF (no field built). ---
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True

# crf-availability-maintenance substrate levers (666c ARM_2 flags; V3-EXQ-666c PASS).
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# Within-class-representative signature horizon (SECONDARY negative control).
H_SIGNATURE = 3

# C2 (PRIMARY): paired-by-seed committed-class entropy lift of ARM_ON over BOTH controls.
C2_LIFT_MARGIN_NATS = 0.05
C2_MIN_LIFT_SEEDS = 2  # of 3

# R1 readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# R2 readiness: GAP-A consumed-summary divergence (649 statistic + 643a numerical ceiling).
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# R3 readiness: ARM_ON rule field minted distinct rules AND matured.
CRF_MIN_MINTED = 2
CRF_N_ACTIVE_FLOOR = 1
CRF_FRAC_ACTIVE_FLOOR = 0.30
CRF_DIST_FLOOR = 1e-3
# R4 readiness: propagation non-vacuity (ARM_ON bias differs from ARM_OFF).
PROP_NONVAC_FLOOR = 1e-3
# R5 readiness: MODULATORY ROUTE-RANGE non-vacuity (the SAME-statistic gate; the
# V3-EXQ-643 rule -- assert RANGE on a positive control because C2 routes on a
# range-driven lift). Read from e3.last_score_diagnostics["modulatory_channel_route_range"]
# per-tick on the ARM_ON positive control. Floor matches the route min-range
# substrate floor (1e-6) lifted to a meaningful per-candidate spread.
ROUTE_RANGE_FLOOR = 1e-3

# Only classes committed to at least this many P2 ticks feed the unweighted mean
# within-class entropy (secondary negative control).
MIN_TICKS_PER_CLASS = 5

MIN_SEEDS_FOR_PASS = 2  # of 3

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 200
P1_BIAS_TRAIN_EPISODES = 90
P2_MEASUREMENT_EPISODES = 60
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 30

# --- Matched-stack lever constants (identical on BOTH arms). ---
# The 569i-VALIDATED conversion stack: std-basis authority + channel routing build
# _modulatory_accum; the TOP-K SHORTLIST arbitrates within F's small rotating
# near-tie set (and OVERRIDES the additive-authority selection).
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 2.0               # 569i (kept; shortlist overrides selection)
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"  # 569i std basis (kept; shortlist overrides)
USE_MODULATORY_CHANNEL_ROUTING = True
MODULATORY_CHANNEL_ROUTE_SOURCE = "cand_world_summary"
MODULATORY_CHANNEL_ROUTE_WEIGHT = 1.0
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6        # substrate numerical active/inactive floor
USE_MODULATORY_SHORTLIST_THEN_MODULATE = True
MODULATORY_SHORTLIST_MODE = "top_k"            # the 569i lever
MODULATORY_SHORTLIST_K = 3                      # small fixed k -> rotating eligible set
MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

# --- 689-FAMILY / MECH-439 conversion-ceiling levers: HELD OFF on every arm. ---
# 695 retests the modulatory/rule channel, NOT the F-dominance demotion lever 689d
# tests. Kept distinct so 695 does not double-count the 689-family lever.
USE_F_ELIGIBILITY_DEMOTION = False
MODULATORY_SHORTLIST_CONFLICT_GRADED = False
USE_GAP_SCALED_COMMIT_TEMPERATURE = False
USE_NATURAL_COMMIT_URGENCY_RELEASE = False

# SD-056 online e2 training (mirror V3-EXQ-649 / 654g harness).
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

# P1 bias-head REINFORCE training (mirror V3-EXQ-598b).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9


# IDENTICAL env to V3-EXQ-654g (SD-054 reef + hazard_food_attraction + bipartite
# layout) -- the behavioural-falsifier substrate.
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


ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_OFF",
        "label": "rule_creator_absent_legacy_collapsed_rule_state_control",
        "use_candidate_rule_field": False,
    },
    {
        "arm_id": "ARM_ON",
        "label": "modulatory_bias_rule_apprehension_present_matured_crf_source",
        "use_candidate_rule_field": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, use_candidate_rule_field: bool) -> REEAgent:
    """Matched-stack agent; the ONLY varied flag is use_candidate_rule_field.

    Both arms enable use_lateral_pfc_analog + use_gated_policy with the bias head
    un-zeroed (lateral_pfc_train_rule_bias_head=True) + the 569i TOP-K shortlist
    conversion + the modulatory route channel. candidate_summary_source =
    e2_world_forward on BOTH arms (GAP-A; e2 trained online in P0). All 689-family
    MECH-439 levers HELD OFF.
    """
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
        # --- Matched stack (identical on both arms) ---
        # Layer A: SP-CEM (candidate-pool first-action-class diversity).
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # GAP-A (V3-EXQ-649): shared per-candidate signal from e2.world_forward.
        candidate_summary_source="e2_world_forward",
        # modulatory-bias-selection-authority (643a float32 fix) ARMED with the
        # 569i TOP-K SHORTLIST conversion.
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
        # --- 689-FAMILY / MECH-439 conversion-ceiling levers: EXPLICITLY OFF ---
        use_f_eligibility_demotion=USE_F_ELIGIBILITY_DEMOTION,
        modulatory_shortlist_conflict_graded=MODULATORY_SHORTLIST_CONFLICT_GRADED,
        use_gap_scaled_commit_temperature=USE_GAP_SCALED_COMMIT_TEMPERATURE,
        use_natural_commit_urgency_release=USE_NATURAL_COMMIT_URGENCY_RELEASE,
        # MECH-341 (stratified across-class; within-class temperature default).
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=MECH341_ENTROPY_BIAS_SCALE,
        e3_diversity_stratified_within_class_temperature=None,
        # MECH-313 noise floor.
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        # V_s minimal stack.
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        # ARC-062 GatedPolicy (matched; symmetry-broken bias).
        use_gated_policy=True,
        # SD-033a LateralPFCAnalog with the bias head un-zeroed + trainable (GAP-D).
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        # SD-056 (e2 action-conditional divergence; trained online in P0).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
        # --- CRF maturity + maintenance levers (inert in ARM_OFF -- no field built) ---
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_context_from_e2_world_forward=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=CRF_MAINTENANCE_FLOOR,
        crf_maintenance_decay=CRF_MAINTENANCE_DECAY,
        crf_mature_context_match_threshold=CRF_MATURE_CONTEXT_MATCH_THRESHOLD,
        crf_tolerance_conflict_cap=CRF_TOLERANCE_CONFLICT_CAP,
        crf_maintenance_couple_to_theta=CRF_MAINTENANCE_COUPLE_TO_THETA,
        # --- The ONLY swept variable ---
        use_candidate_rule_field=bool(use_candidate_rule_field),
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-649 / 654g)
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


def _traj_rep_signature(traj) -> Tuple[int, ...]:
    acts = traj.actions[0]  # [horizon, action_dim]
    h = min(H_SIGNATURE, int(acts.shape[0]))
    classes = acts[:h, :].argmax(dim=-1).detach().reshape(-1).tolist()
    return tuple(int(c) for c in classes)


def _consumed_summaries(agent: REEAgent, candidates) -> Optional[torch.Tensor]:
    """Per-candidate cand_world_summaries the bias channels consume (GAP-A
    e2.world_forward source; both arms)."""
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


def _route_range_this_tick(agent: REEAgent) -> Optional[float]:
    """The SAME route-range statistic the conversion authority routes on:
    e3.last_score_diagnostics["modulatory_channel_route_range"] -- the RAW
    cross-candidate range of the routed modulatory bias at this select tick."""
    e3 = getattr(agent, "e3", None)
    if e3 is None:
        return None
    diag = getattr(e3, "last_score_diagnostics", None)
    if not isinstance(diag, dict):
        return None
    rr = diag.get("modulatory_channel_route_range", None)
    if rr is None:
        return None
    try:
        rr_f = float(rr)
    except (TypeError, ValueError):
        return None
    return rr_f if math.isfinite(rr_f) else None


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


def _entropy_from_counter(counter: Counter) -> float:
    n = sum(counter.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


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
# P1 bias-head REINFORCE training (mirror V3-EXQ-598b _lpfc_reinforce_loss)
# ---------------------------------------------------------------------------


def _lpfc_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
) -> torch.Tensor:
    """REINFORCE on the SD-033a bias head over stored (candidate_features, sel, return)."""
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


def _propagation_counterfactual_delta(
    agent: REEAgent, summaries: torch.Tensor
) -> Optional[float]:
    """Within-ARM_ON isolation: mean |bias(field rule_state) - bias(zeroed rule_state)|."""
    lpfc = getattr(agent, "lateral_pfc", None)
    if lpfc is None or summaries is None:
        return None
    try:
        with torch.no_grad():
            bias_field = lpfc.compute_bias(summaries).detach().clone()
            saved = lpfc.rule_state.detach().clone()
            lpfc.rule_state.zero_()
            bias_zero = lpfc.compute_bias(summaries).detach().clone()
            lpfc.rule_state.copy_(saved)
        return float((bias_field - bias_zero).abs().mean().item())
    except Exception:
        return None


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
    agent = _make_agent(env, bool(arm["use_candidate_rule_field"]))
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

    # P1 REINFORCE state.
    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    # PRIMARY DV + readiness accumulators (P2).
    committed_class_counts: Dict[int, int] = {}
    selected_class_counts: Dict[int, int] = {}
    proposer_class_counts: Dict[int, int] = {}  # matched-noise proposer control (C_R1B)
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0

    # SECONDARY negative control (within-class-representative; P2).
    per_class_rep_sigs: Dict[int, Counter] = {}
    all_rep_sigs: Counter = Counter()

    # ARM_ON differentiation + bias + ROUTE-RANGE diagnostics (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_matched_per_tick: List[int] = []
    crf_max_pairwise_rule_dist_max = 0.0
    crf_n_minted_total_last = 0
    lateral_pfc_bias_abs_vals: List[float] = []
    prop_counterfactual_deltas: List[float] = []
    route_range_vals: List[float] = []  # R5 same-statistic readiness capture

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

            # SD-056 transition capture (z0, a) this tick -> z1 next tick.
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

            # P1 REINFORCE snap (the same e2_world_forward source compute_bias consumes).
            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

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

            # P1: record (candidate_features, selected-candidate-index) snap.
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
                # Matched-noise proposer control (C_R1B): the proposer first-action
                # distribution the channel must beat. Count every candidate's first
                # action class this tick (the collapsed-control entropy source).
                if candidates:
                    for c in candidates:
                        pc = _traj_first_action_class(c)
                        proposer_class_counts[pc] = proposer_class_counts.get(pc, 0) + 1
                if len(pre_e3_classes) >= 2:
                    n_p2_pre_ge2 += 1

                if candidates and len(candidates) >= 2:
                    consumed = _consumed_summaries(agent, candidates)
                    if consumed is not None and torch.isfinite(consumed).all():
                        d = _mean_pairwise_l2(consumed)
                        if math.isfinite(d):
                            consumed_dists.append(d)
                            consumed_dist_max = max(consumed_dist_max, d)

                # R5: route-range this tick -- the SAME route-range statistic the
                # conversion authority routes on. Captured for the readiness gate.
                rr = _route_range_this_tick(agent)
                if rr is not None:
                    route_range_vals.append(rr)

                # SECONDARY negative control: within-class representative.
                sel_traj = getattr(agent.e3, "_last_selected_trajectory", None)
                if sel_traj is not None:
                    sel_class = _traj_first_action_class(sel_traj)
                    rep_sig = _traj_rep_signature(sel_traj)
                    selected_class_counts[sel_class] = (
                        selected_class_counts.get(sel_class, 0) + 1
                    )
                    per_class_rep_sigs.setdefault(sel_class, Counter())[rep_sig] += 1
                    all_rep_sigs[rep_sig] += 1

                # lateral_pfc bias magnitude (manipulation-reached-E3 context).
                lpfc = getattr(agent, "lateral_pfc", None)
                if lpfc is not None:
                    lb_mean = getattr(lpfc, "_last_bias_abs_mean", None)
                    if isinstance(lb_mean, (int, float)):
                        lateral_pfc_bias_abs_vals.append(float(lb_mean))

                # CandidateRuleField differentiation (ARM_ON).
                crf = getattr(agent, "candidate_rule_field", None)
                if crf is not None:
                    st = crf.get_state()
                    n_active = int(st.get("crf_n_active_last", 0))
                    crf_n_active_per_tick.append(n_active)
                    crf_n_matched_per_tick.append(
                        int(st.get("crf_n_matched_last", 0))
                    )
                    crf_max_pairwise_rule_dist_max = max(
                        crf_max_pairwise_rule_dist_max,
                        float(st.get("crf_max_pairwise_rule_dist", 0.0)),
                    )
                    crf_n_minted_total_last = int(st.get("crf_n_minted_total", 0))
                    if (
                        n_active >= CRF_N_ACTIVE_FLOOR
                        and candidates and len(candidates) >= 2
                    ):
                        cf_summ = _consumed_summaries(agent, candidates)
                        if cf_summ is not None and torch.isfinite(cf_summ).all():
                            d_cf = _propagation_counterfactual_delta(agent, cf_summ)
                            if d_cf is not None and math.isfinite(d_cf):
                                prop_counterfactual_deltas.append(d_cf)
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
                agent.update_residue(
                    harm_signal=float(_harm_signal),
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

        # P1 end-of-episode: REINFORCE update on the SD-033a bias head.
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
    selected_class_entropy = _entropy_from_int_counts(selected_class_counts)
    proposer_class_entropy = _entropy_from_int_counts(proposer_class_counts)

    frac_pre_ge2 = float(n_p2_pre_ge2 / n_p2_ticks) if n_p2_ticks > 0 else 0.0

    consumed_spread_mean = (
        float(sum(consumed_dists) / len(consumed_dists)) if consumed_dists else 0.0
    )

    # SECONDARY within-class-representative entropy (negative control).
    qualifying: List[float] = []
    within_class_entropies: Dict[int, float] = {}
    for cls, sig_counter in per_class_rep_sigs.items():
        ent = _entropy_from_counter(sig_counter)
        within_class_entropies[cls] = ent
        if sum(sig_counter.values()) >= MIN_TICKS_PER_CLASS:
            qualifying.append(ent)
    mean_within_class_rep_entropy = (
        float(sum(qualifying) / len(qualifying)) if qualifying else 0.0
    )

    # ARM_ON differentiation readiness.
    if crf_n_active_per_tick:
        frac_crf_active_ge_floor = float(
            sum(1 for n in crf_n_active_per_tick if n >= CRF_N_ACTIVE_FLOOR)
            / len(crf_n_active_per_tick)
        )
        mean_crf_n_active = float(
            sum(crf_n_active_per_tick) / len(crf_n_active_per_tick)
        )
    else:
        frac_crf_active_ge_floor = 0.0
        mean_crf_n_active = 0.0

    if crf_n_matched_per_tick:
        mean_crf_n_matched = float(
            sum(crf_n_matched_per_tick) / len(crf_n_matched_per_tick)
        )
        max_crf_n_matched = int(max(crf_n_matched_per_tick))
    else:
        mean_crf_n_matched = 0.0
        max_crf_n_matched = 0

    crf_present = bool(arm["use_candidate_rule_field"])
    crf_differentiated = bool(
        crf_present
        and crf_n_minted_total_last >= CRF_MIN_MINTED
        and frac_crf_active_ge_floor >= CRF_FRAC_ACTIVE_FLOOR
    )

    mean_lateral_pfc_bias_abs = (
        float(sum(lateral_pfc_bias_abs_vals) / len(lateral_pfc_bias_abs_vals))
        if lateral_pfc_bias_abs_vals else 0.0
    )
    mean_prop_counterfactual_delta = (
        float(sum(prop_counterfactual_deltas) / len(prop_counterfactual_deltas))
        if prop_counterfactual_deltas else 0.0
    )
    # R5 ROUTE-RANGE aggregate (the SAME route-range statistic the C2 criterion's
    # lifting channel routes on).
    mean_route_range = (
        float(sum(route_range_vals) / len(route_range_vals)) if route_range_vals else 0.0
    )
    max_route_range = float(max(route_range_vals)) if route_range_vals else 0.0

    # Per-seed readiness flags.
    seed_class_axis_exercisable = bool(frac_pre_ge2 > FRAC_PRE_GE2_FLOOR)
    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )
    seed_route_range_nonvac = bool(mean_route_range > ROUTE_RANGE_FLOOR)

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "use_candidate_rule_field": crf_present,
        "crf_persist_rules_across_episode_reset": True,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
        # ----- PRIMARY DV -----
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        # ----- proposer matched-noise control (C_R1B) -----
        "proposer_class_entropy_nats": round(proposer_class_entropy, 6),
        "n_unique_proposer_classes": int(len(proposer_class_counts)),
        # ----- R1 readiness -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "class_axis_exercisable": seed_class_axis_exercisable,
        # ----- R2 readiness -----
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        # ----- R3 readiness -----
        "crf_present": crf_present,
        "crf_mean_n_active": round(mean_crf_n_active, 6),
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_max_pairwise_rule_dist": round(crf_max_pairwise_rule_dist_max, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "crf_differentiated": crf_differentiated,
        "crf_mean_n_matched": round(mean_crf_n_matched, 6),
        "crf_max_n_matched": int(max_crf_n_matched),
        # ----- R4 propagation non-vacuity -----
        "mean_lateral_pfc_bias_abs": round(mean_lateral_pfc_bias_abs, 8),
        "mean_prop_counterfactual_delta": round(mean_prop_counterfactual_delta, 8),
        # ----- R5 ROUTE-RANGE non-vacuity (same-statistic gate) -----
        "mean_modulatory_route_range": round(mean_route_range, 8),
        "max_modulatory_route_range": round(max_route_range, 8),
        "route_range_nonvac": seed_route_range_nonvac,
        # ----- SECONDARY negative control (NOT load-bearing) -----
        "mean_within_class_rep_entropy_nats": round(mean_within_class_rep_entropy, 6),
        "n_distinct_rep_signatures_total": int(len(all_rep_sigs)),
        "selected_class_entropy_nats": round(selected_class_entropy, 6),
    }


def _arm_rows(arm_results: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [
        r for r in arm_results
        if r["arm_id"] == arm_id and r["error_note"] is None
    ]


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
    arm_results: List[Dict[str, Any]] = []
    script_path = Path(__file__).resolve()

    for arm in ARMS:
        print(
            f"Arm {arm['arm_id']} ({arm['label']}) "
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
                    "use_candidate_rule_field": bool(arm["use_candidate_rule_field"]),
                    "crf_persist_rules_across_episode_reset": True,
                    "crf_mature_pool_dynamics": True,
                    "crf_context_from_e2_world_forward": True,
                    "crf_availability_maintenance": True,
                    "crf_maintenance_floor": float(CRF_MAINTENANCE_FLOOR),
                    "crf_maintenance_decay": float(CRF_MAINTENANCE_DECAY),
                    "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
                    "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
                    "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
                    "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
                    "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
                    "modulatory_channel_route_source": str(MODULATORY_CHANNEL_ROUTE_SOURCE),
                    "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
                    "modulatory_shortlist_conflict_graded": bool(MODULATORY_SHORTLIST_CONFLICT_GRADED),
                    "use_gap_scaled_commit_temperature": bool(USE_GAP_SCALED_COMMIT_TEMPERATURE),
                    "use_natural_commit_urgency_release": bool(USE_NATURAL_COMMIT_URGENCY_RELEASE),
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
                ],
            )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    off_rows = _arm_rows(arm_results, "ARM_OFF")
    on_rows = _arm_rows(arm_results, "ARM_ON")

    # ---- P0 READINESS / non-vacuity (route via p0_readiness_gate, same-statistic) ----
    # R1: committed-class axis exercisable both arms.
    n_off_axis = sum(1 for r in off_rows if r["class_axis_exercisable"])
    n_on_axis = sum(1 for r in on_rows if r["class_axis_exercisable"])
    r1_holds = bool(n_off_axis >= MIN_SEEDS_FOR_PASS and n_on_axis >= MIN_SEEDS_FOR_PASS)

    # R2: GAP-A divergence real both arms.
    n_off_gapa = sum(1 for r in off_rows if r["gapa_divergence"])
    n_on_gapa = sum(1 for r in on_rows if r["gapa_divergence"])
    r2_holds = bool(n_off_gapa >= MIN_SEEDS_FOR_PASS and n_on_gapa >= MIN_SEEDS_FOR_PASS)

    # R3: ARM_ON differentiated + matured on majority of ARM_ON seeds.
    n_on_differentiated = sum(1 for r in on_rows if r["crf_differentiated"])
    r3_holds = bool(n_on_differentiated >= MIN_SEEDS_FOR_PASS)

    # R4: propagation non-vacuity -- paired |bias_ON - bias_OFF| > floor.
    off_bias_by_seed = {int(r["seed"]): r["mean_lateral_pfc_bias_abs"] for r in off_rows}
    on_bias_by_seed = {int(r["seed"]): r["mean_lateral_pfc_bias_abs"] for r in on_rows}
    prop_diff_by_seed: Dict[int, float] = {}
    n_prop_nonvac_seeds = 0
    for seed in sorted(set(off_bias_by_seed) & set(on_bias_by_seed)):
        diff = abs(on_bias_by_seed[seed] - off_bias_by_seed[seed])
        prop_diff_by_seed[seed] = round(diff, 8)
        if diff > PROP_NONVAC_FLOOR:
            n_prop_nonvac_seeds += 1
    r4_holds = bool(n_prop_nonvac_seeds >= MIN_SEEDS_FOR_PASS)

    # R5: MODULATORY ROUTE-RANGE non-vacuity (the SAME route-range statistic the C2
    # criterion's lifting channel routes on) on majority of ARM_ON seeds.
    n_on_route_range = sum(1 for r in on_rows if r["route_range_nonvac"])
    r5_holds = bool(n_on_route_range >= MIN_SEEDS_FOR_PASS)

    # Aggregate via p0_readiness_gate (manifest-ready preconditions[] payload). We use
    # the worst-case across-seeds measured value vs each floor; "lower" direction = floor
    # (met iff measured >= threshold) except the bounded-ceiling check.
    on_route_range_majority = (
        sorted([r["mean_modulatory_route_range"] for r in on_rows], reverse=True)[
            MIN_SEEDS_FOR_PASS - 1
        ]
        if len(on_rows) >= MIN_SEEDS_FOR_PASS else 0.0
    )
    on_crf_frac_majority = (
        sorted([r["crf_frac_active_ge_floor"] for r in on_rows], reverse=True)[
            MIN_SEEDS_FOR_PASS - 1
        ]
        if len(on_rows) >= MIN_SEEDS_FOR_PASS else 0.0
    )
    min_frac_pre_ge2 = float(min([r["frac_pre_ge2"] for r in (off_rows + on_rows)] or [0.0]))
    min_consumed_spread = float(
        min([r["consumed_summary_pairwise_dist_mean"] for r in (off_rows + on_rows)] or [0.0])
    )
    max_consumed_spread = float(
        max([r["consumed_summary_pairwise_dist_max"] for r in (off_rows + on_rows)] or [0.0])
    )
    max_prop_diff = float(max(list(prop_diff_by_seed.values()) or [0.0]))

    # Each R* below is a COUNT-of-seeds rule (r1..r5_holds above), so each check reports
    # the COUNT against MIN_SEEDS_FOR_PASS. p0_readiness_gate derives `met` from
    # (measured, threshold, direction) with floor semantics `measured >= threshold`,
    # and REE_assembly's build_experiment_indexes._precondition_unmet RECOMPUTES `met`
    # the same way and treats its recompute as AUTHORITATIVE -- so a count against
    # MIN_SEEDS_FOR_PASS reproduces r1..r5_holds EXACTLY, in both places.
    #
    # These previously reported continuous worst/best-case magnitudes against their
    # scientific floors, which did NOT reproduce the shipped booleans in either
    # direction:
    #   R1/R2  min over ALL cells vs the floor -- strictly HARSHER than "a majority of
    #          seeds in each arm" (one bad seed sinks it), so a sound run was flagged
    #          `precondition_unmet` (699's sibling entry shipped 0.004852 vs a 0.05
    #          floor while its per-arm majorities held).
    #   R4     max over seeds vs the floor -- strictly LOOSER than "on >= 2 seeds", so a
    #          genuinely-failed premise would have been silently CLEARED.
    #   R3/R5  already used the k-th-best form, which is exact for a SINGLE-leg floor --
    #          but `crf_differentiated` is a CONJUNCTION (crf_present AND n_minted >=
    #          CRF_MIN_MINTED AND frac_active >= floor) that a count over the frac_active
    #          leg alone cannot reproduce, and `route_range_nonvac` is STRICT (`>`) where
    #          the gate's floor is inclusive. Counts fix both.
    # No single continuous statistic CAN reproduce a count over a conjunction, since the
    # count does not distribute into per-leg counts. The original magnitudes are attached
    # below as NON-BOUND `observed_*` diagnostics (extra keys are ignored by the
    # recompute), so nothing is lost.
    readiness_checks = [
        {"name": "R1_committed_class_axis_exercisable_both_arms",
         "measured": float(min(n_off_axis, n_on_axis)),
         "threshold": float(MIN_SEEDS_FOR_PASS), "direction": "lower"},
        {"name": "R2_gapa_consumed_summary_divergence_both_arms",
         "measured": float(min(n_off_gapa, n_on_gapa)),
         "threshold": float(MIN_SEEDS_FOR_PASS), "direction": "lower"},
        # R2b is NOT a count rule: it is a standalone ceiling on the observed magnitude,
        # with no per-seed boolean behind it, so it keeps its magnitude form. (The same
        # ceiling also appears as the second leg of the per-seed `gapa_divergence`
        # conjunction that R2 counts.)
        {"name": "R2b_gapa_consumed_summary_bounded",
         "measured": max_consumed_spread, "threshold": float(CONSUMED_MAGNITUDE_CEIL),
         "direction": "upper"},
        {"name": "R3_arm_on_rule_field_differentiated_and_matured",
         "measured": float(n_on_differentiated),
         "threshold": float(MIN_SEEDS_FOR_PASS), "direction": "lower"},
        {"name": "R4_propagation_non_vacuity_arm_on_bias_differs_from_arm_off",
         "measured": float(n_prop_nonvac_seeds),
         "threshold": float(MIN_SEEDS_FOR_PASS), "direction": "lower"},
        {"name": "R5_modulatory_route_range_non_vacuity_same_statistic",
         "measured": float(n_on_route_range),
         "threshold": float(MIN_SEEDS_FOR_PASS), "direction": "lower"},
    ]
    # NON-BOUND diagnostics, merged onto the gate's payload below. The gate emits only
    # name/measured/threshold/direction/met/kind, so the magnitudes the checks used to
    # bind on are re-attached here rather than dropped.
    readiness_diagnostics = {
        "R1_committed_class_axis_exercisable_both_arms": {
            "observed_min_frac_pre_ge2": min_frac_pre_ge2,
            "observed_frac_pre_ge2_floor": float(FRAC_PRE_GE2_FLOOR),
        },
        "R2_gapa_consumed_summary_divergence_both_arms": {
            "observed_min_consumed_summary_pairwise_dist_mean": min_consumed_spread,
            "observed_consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
        },
        "R3_arm_on_rule_field_differentiated_and_matured": {
            "observed_kth_best_crf_frac_active": float(on_crf_frac_majority),
            "observed_crf_frac_active_floor": float(CRF_FRAC_ACTIVE_FLOOR),
        },
        "R4_propagation_non_vacuity_arm_on_bias_differs_from_arm_off": {
            "observed_max_paired_bias_diff": max_prop_diff,
            "observed_prop_nonvac_floor": float(PROP_NONVAC_FLOOR),
        },
        "R5_modulatory_route_range_non_vacuity_same_statistic": {
            "observed_kth_best_arm_on_route_range": float(on_route_range_majority),
            "observed_route_range_floor": float(ROUTE_RANGE_FLOOR),
        },
    }

    # Build the manifest-ready preconditions[] payload regardless of pass/fail, and the
    # composite readiness flag (use the per-seed majority logic, which is the binding
    # decision; p0_readiness_gate's measured/threshold mirror it for the indexer).
    preconditions: List[Dict[str, Any]]
    try:
        preconditions = p0_readiness_gate(readiness_checks)
        gate_raised = False
    except P0NotReady as exc:
        preconditions = exc.preconditions
        gate_raised = True
    for _pc in preconditions:
        _pc.update(readiness_diagnostics.get(_pc.get("name", ""), {}))

    readiness_holds = bool(r1_holds and r2_holds and r3_holds and r4_holds and r5_holds)

    # ---- C2 (PRIMARY): paired committed-class entropy lift over BOTH controls ----
    off_by_seed = {int(r["seed"]): r["committed_class_entropy_nats"] for r in off_rows}
    on_by_seed = {int(r["seed"]): r["committed_class_entropy_nats"] for r in on_rows}
    proposer_by_seed = {int(r["seed"]): r["proposer_class_entropy_nats"] for r in on_rows}
    paired_lifts_vs_off: Dict[int, float] = {}
    paired_lifts_vs_proposer: Dict[int, float] = {}
    n_lift_seeds = 0
    for seed in sorted(set(off_by_seed) & set(on_by_seed)):
        lift_off = on_by_seed[seed] - off_by_seed[seed]
        lift_prop = on_by_seed[seed] - proposer_by_seed.get(seed, 0.0)
        paired_lifts_vs_off[seed] = round(lift_off, 6)
        paired_lifts_vs_proposer[seed] = round(lift_prop, 6)
        # Strictly above BOTH controls by the margin (C_R1B + ARM_OFF).
        if lift_off >= C2_LIFT_MARGIN_NATS and lift_prop >= C2_LIFT_MARGIN_NATS:
            n_lift_seeds += 1
    c2_holds = bool(n_lift_seeds >= C2_MIN_LIFT_SEEDS)

    off_mean_dv = _mean([r["committed_class_entropy_nats"] for r in off_rows])
    on_mean_dv = _mean([r["committed_class_entropy_nats"] for r in on_rows])
    proposer_mean_dv = _mean([r["proposer_class_entropy_nats"] for r in on_rows])

    on_prop_cf = [r["mean_prop_counterfactual_delta"] for r in on_rows]
    n_on_prop_cf_nonzero = sum(1 for d in on_prop_cf if d > PROP_NONVAC_FLOOR)

    # ---- Non-degeneracy self-report (same-statistic: route_range + lift channel) ----
    degeneracy = check_degeneracy({
        "modulatory_route_range_arm_on": {
            "values": [r["mean_modulatory_route_range"] for r in on_rows],
            "floor": float(ROUTE_RANGE_FLOOR),
        },
        "committed_class_entropy_lift_paired": {
            "groups": [
                [on_by_seed[s], off_by_seed[s]]
                for s in sorted(set(off_by_seed) & set(on_by_seed))
            ],
        },
    })

    # ----- Outcome map (THREE branches; NO weakens) -----
    if not readiness_holds:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif c2_holds:
        outcome = "PASS"
        direction = "supports"
        label = "PASS_readiness_C2_modulatory_rule_channel_breaks_monomodal_collapse"
    else:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "shared_f_dominance_conversion_ceiling_route_implement_substrate_mech448"

    evidence_direction_per_claim = {"MECH-309": direction, "ARC-062": direction}

    interpretation = {
        "label": label,
        "preconditions": preconditions,
        "criteria": [
            {
                "name": "C2_committed_class_entropy_lift_over_both_controls",
                "load_bearing": True,
                "passed": bool(c2_holds),
            },
        ],
        "criteria_non_degenerate": {
            "R1_class_axis_exercisable": bool(r1_holds),
            "R2_gapa_divergence": bool(r2_holds),
            "R3_arm_on_differentiated_matured": bool(r3_holds),
            "R4_propagation_non_vacuity": bool(r4_holds),
            "R5_modulatory_route_range_non_vacuity": bool(r5_holds),
            "R4_within_arm_on_rule_state_counterfactual_nonzero": bool(
                n_on_prop_cf_nonzero >= MIN_SEEDS_FOR_PASS
            ),
            "C2_paired_lift": bool(c2_holds),
            "non_degenerate": bool(degeneracy["non_degenerate"]),
        },
    }

    total_seeds = len(ARMS) * len(seeds)
    total_completed = len(off_rows) + len(on_rows)

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "interpretation_label": label,
        "interpretation": interpretation,
        "non_degenerate": bool(degeneracy["non_degenerate"]),
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "degenerate_metrics": degeneracy["degenerate_metrics"],
        "p0_readiness_gate_raised": bool(gate_raised),
        "seeds": seeds,
        "n_arms": len(ARMS),
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "h_signature": int(H_SIGNATURE),
            "c2_lift_margin_nats": float(C2_LIFT_MARGIN_NATS),
            "c2_min_lift_seeds": int(C2_MIN_LIFT_SEEDS),
            "frac_pre_ge2_floor": float(FRAC_PRE_GE2_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "consumed_magnitude_ceil": float(CONSUMED_MAGNITUDE_CEIL),
            "crf_min_minted": int(CRF_MIN_MINTED),
            "crf_n_active_floor": int(CRF_N_ACTIVE_FLOOR),
            "crf_frac_active_floor": float(CRF_FRAC_ACTIVE_FLOOR),
            "crf_dist_floor": float(CRF_DIST_FLOOR),
            "prop_nonvac_floor": float(PROP_NONVAC_FLOOR),
            "route_range_floor": float(ROUTE_RANGE_FLOOR),
            "min_ticks_per_class": int(MIN_TICKS_PER_CLASS),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "lr_lpfc_bias": float(LR_LPFC_BIAS),
            "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
            "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
            "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
            "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
            "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
            "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
            "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
            "modulatory_shortlist_conflict_graded": bool(MODULATORY_SHORTLIST_CONFLICT_GRADED),
            "use_gap_scaled_commit_temperature": bool(USE_GAP_SCALED_COMMIT_TEMPERATURE),
            "use_natural_commit_urgency_release": bool(USE_NATURAL_COMMIT_URGENCY_RELEASE),
            "sd056_weight": float(SD056_WEIGHT),
            "crf_persist_rules_across_episode_reset": True,
        },
        "acceptance_criteria": {
            "READINESS_substrate_exercisable_and_route_range_nonvac": readiness_holds,
            "R1_class_axis_exercisable_both_arms": r1_holds,
            "R1_n_off_axis": int(n_off_axis),
            "R1_n_on_axis": int(n_on_axis),
            "R2_gapa_divergence_both_arms": r2_holds,
            "R2_n_off_gapa": int(n_off_gapa),
            "R2_n_on_gapa": int(n_on_gapa),
            "R3_arm_on_rule_field_differentiated_matured": r3_holds,
            "R3_n_on_differentiated": int(n_on_differentiated),
            "R4_propagation_non_vacuity": r4_holds,
            "R4_n_prop_nonvac_seeds": int(n_prop_nonvac_seeds),
            "R4_prop_diff_by_seed": prop_diff_by_seed,
            "R4_n_on_within_arm_counterfactual_nonzero": int(n_on_prop_cf_nonzero),
            "R5_modulatory_route_range_non_vacuity": r5_holds,
            "R5_n_on_route_range_nonvac": int(n_on_route_range),
            "R5_arm_on_mean_route_range_majority": round(on_route_range_majority, 8),
            "C2_committed_class_lift_over_both_controls": c2_holds,
            "C2_n_lift_seeds": int(n_lift_seeds),
            "C2_paired_lifts_vs_off_by_seed": paired_lifts_vs_off,
            "C2_paired_lifts_vs_proposer_by_seed": paired_lifts_vs_proposer,
            "C2_off_mean_committed_class_entropy": round(off_mean_dv, 6),
            "C2_on_mean_committed_class_entropy": round(on_mean_dv, 6),
            "C2_proposer_mean_first_action_entropy": round(proposer_mean_dv, 6),
        },
        "secondary_negative_control_not_load_bearing": {
            "note": (
                "Within-class-representative entropy is a NEGATIVE CONTROL: the rule "
                "bias is class-keyed (compute_bias broadcasts one rule_state across K), "
                "so it cannot move within-class selection -> ARM_ON ~ ARM_OFF is "
                "EXPECTED here, confirming the rule-creator's signal lives in the "
                "committed-class axis (the load-bearing C2 DV)."
            ),
            "arm_off_within_class_rep_entropy_mean": round(
                _mean([r["mean_within_class_rep_entropy_nats"] for r in off_rows]), 6
            ),
            "arm_on_within_class_rep_entropy_mean": round(
                _mean([r["mean_within_class_rep_entropy_nats"] for r in on_rows]), 6
            ),
            "arm_off_selected_class_entropy_mean": round(
                _mean([r["selected_class_entropy_nats"] for r in off_rows]), 6
            ),
            "arm_on_selected_class_entropy_mean": round(
                _mean([r["selected_class_entropy_nats"] for r in on_rows]), 6
            ),
            "arm_off_lateral_pfc_bias_abs_mean": round(
                _mean([r["mean_lateral_pfc_bias_abs"] for r in off_rows]), 8
            ),
            "arm_on_lateral_pfc_bias_abs_mean": round(
                _mean([r["mean_lateral_pfc_bias_abs"] for r in on_rows]), 8
            ),
            "arm_on_mean_route_range": round(
                _mean([r["mean_modulatory_route_range"] for r in on_rows]), 8
            ),
            "arm_on_within_arm_prop_counterfactual_delta_mean": round(
                _mean(on_prop_cf), 8
            ),
        },
        "interpretation_grid": {
            "PASS_readiness_C2": (
                "The implemented modulatory-bias / rule-apprehension channel's "
                "DIFFERENTIATED + MATURED rule_state carries cross-candidate route "
                "range (R5) and lifts committed-class diversity above BOTH the collapsed "
                "ARM_OFF control and the proposer first-action distribution (breaks "
                "monomodal collapse). supports MECH-309 + ARC-062. Route to /governance."
            ),
            "FAIL_readiness_holds_C2_fails": (
                "Class axis exercisable, GAP-A divergence real, ARM_ON differentiated + "
                "matured, propagation non-vacuous AND route range present (R5) -- the "
                "modulatory channel REACHES the committed argmax -- but it adds no "
                "marginal committed-class diversity over the controls. This is the SHARED "
                "F-dominance CONVERSION CEILING (MECH-439; behavioral_diversity_isolation:"
                "GAP-A; 569g/682), now mid-build (MECH-448 lead; V3-EXQ-689d queued). NOT "
                "a MECH-309 / ARC-062 falsification. non_contributory; route to "
                "/implement-substrate (the F-dominance rebalance). Do NOT weaken."
            ),
            "FAIL_readiness_substrate_not_ready_requeue": (
                "The committed-class axis was not exercisable, and/or GAP-A divergence "
                "absent, and/or ARM_ON did not mature a differentiated pool, and/or "
                "propagation vacuous, and/or the MODULATORY ROUTE-RANGE (the SAME "
                "statistic the C2 lift routes on) was degenerate -- the run landed on "
                "the mid-build MECH-439 conversion ceiling, not a fair test. The "
                "falsifier could not run -- NOT a falsification. Re-queue; do NOT weaken."
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
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "evidence_direction_note": (
            f"V3-EXQ-695 ARC-062 / MECH-309 modulatory-bias monomodal-collapse RETEST "
            f"on the now-implemented modulatory-bias / rule-apprehension channel, "
            f"supersedes V3-EXQ-654g (non_contributory 2026-06-19; conversion-ceiling "
            f"self-route). Single-variable: ARM_OFF (use_candidate_rule_field=False, "
            f"legacy collapsed rule_state control) vs ARM_ON (use_candidate_rule_field="
            f"True -> MATURED + MAINTAINED differentiated crf_source), both on the matched "
            f"649/643a/SD-056/SP-CEM/MECH-341 stack with the 569i TOP-K SHORTLIST "
            f"conversion (use_modulatory_shortlist_then_modulate=True + mode="
            f"{MODULATORY_SHORTLIST_MODE} + k={MODULATORY_SHORTLIST_K}) and the SD-033a "
            f"bias head un-zeroed AND TRAINED in a frozen-encoder P1 REINFORCE window "
            f"(GAP-D). PRIMARY DV = committed-class entropy, read against BOTH the ARM_OFF "
            f"control AND the proposer first-action distribution (the collapsed controls "
            f"the channel must beat). All 689-family / MECH-439 conversion-ceiling levers "
            f"(use_f_eligibility_demotion / modulatory_shortlist_conflict_graded / "
            f"use_gap_scaled_commit_temperature / use_natural_commit_urgency_release) "
            f"HELD OFF on every arm -- 695 is distinct from 689d. ENTANGLEMENT: per "
            f"substrate_queue f_dominance_conversion_ceiling the residual ceiling IS the "
            f"shared MECH-439 F-dominance ceiling, MID-BUILD (MECH-448 lead landed today; "
            f"V3-EXQ-689d queued). So 695 self-routes substrate_not_ready_requeue if the "
            f"lifting channel is degenerate via a SAME-STATISTIC P0 readiness gate -- R5 "
            f"asserts ARM_ON modulatory_channel_route_range (the SAME route-range statistic "
            f"the conversion authority routes on, per the V3-EXQ-643 rule) > "
            f"{ROUTE_RANGE_FLOOR} on a majority of seeds, alongside R1 class-axis / R2 "
            f"GAP-A divergence / R3 matured pool / R4 propagation non-vacuity. "
            f"interpretation_label={result['interpretation_label']}. "
            f"READINESS={result['acceptance_criteria']['READINESS_substrate_exercisable_and_route_range_nonvac']}, "
            f"C2={result['acceptance_criteria']['C2_committed_class_lift_over_both_controls']}, "
            f"non_degenerate={result['non_degenerate']}. ONLY the PASS branch weights "
            f"MECH-309/ARC-062 (as supports); readiness-fail self-routes "
            f"substrate_not_ready_requeue AND readiness-holds-C2-fail self-routes the "
            f"deeper F-dominance CONVERSION ceiling (MECH-439; MECH-448 lead) -> "
            f"/implement-substrate -- BOTH non_contributory, NEITHER a falsification "
            f"(NO weakens branch)."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "arms": "ARM_OFF (rule field off, collapsed-control) vs ARM_ON (modulatory-bias / rule-apprehension on)",
            "swept_variable": "use_candidate_rule_field",
            "supersedes": SUPERSEDES,
            "retest_of": "V3-EXQ-654g {MECH-309, ARC-062} on the now-built modulatory channel",
            "matched_stack": (
                "SP-CEM + candidate_summary_source=e2_world_forward (GAP-A/649) + "
                "use_modulatory_selection_authority (643a) + channel routing "
                "(cand_world_summary) + 569i TOP-K SHORTLIST (mode=top_k, k=3) + "
                "MECH-341 stratified + MECH-313 noise floor + V_s minimal + "
                "use_gated_policy + use_lateral_pfc_analog (rule_bias_head TRAINED in P1) "
                "+ SD-056 all levers + matured CRF stack (crf_persist + 666c maintenance "
                "+ CRF-gate amend)"
            ),
            "mech439_689_family_levers_held_off": {
                "use_f_eligibility_demotion": USE_F_ELIGIBILITY_DEMOTION,
                "modulatory_shortlist_conflict_graded": MODULATORY_SHORTLIST_CONFLICT_GRADED,
                "use_gap_scaled_commit_temperature": USE_GAP_SCALED_COMMIT_TEMPERATURE,
                "use_natural_commit_urgency_release": USE_NATURAL_COMMIT_URGENCY_RELEASE,
                "note": "695 retests the modulatory/rule channel, NOT the 689d F-dominance demotion lever",
            },
            "primary_dv": "committed-class entropy vs ARM_OFF + proposer first-action distribution",
            "secondary_negative_control": "within-class-representative entropy (expected ~null)",
            "phases": "P0 e2-train (field matures) -> P1 frozen-encoder bias-head REINFORCE -> P2 frozen measurement",
            "p0_readiness_same_statistic_route_range": {
                "statistic": "e3.last_score_diagnostics['modulatory_channel_route_range']",
                "floor": float(ROUTE_RANGE_FLOOR),
                "rule": "V3-EXQ-643: range-gated C2 criterion -> assert RANGE on the ARM_ON positive control",
            },
            "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
            "use_modulatory_channel_routing": USE_MODULATORY_CHANNEL_ROUTING,
            "modulatory_channel_route_source": MODULATORY_CHANNEL_ROUTE_SOURCE,
            "use_modulatory_shortlist_then_modulate": USE_MODULATORY_SHORTLIST_THEN_MODULATE,
            "modulatory_shortlist_mode": MODULATORY_SHORTLIST_MODE,
            "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
            "known_open_risk": "569i top-k 2/3-seed margin thin (MECH-439 F-dominance); C2 fail -> branch (b) deeper rebalance, NOT a falsification",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "sd056_output_norm_clamp": SD056_OUTPUT_NORM_CLAMP,
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-695 ARC-062/MECH-309 modulatory-bias monomodal retest (supersedes 654g)"
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
        f"READINESS={result['acceptance_criteria']['READINESS_substrate_exercisable_and_route_range_nonvac']} "
        f"C2={result['acceptance_criteria']['C2_committed_class_lift_over_both_controls']} "
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
    return outcome_emit, manifest_for_sentinel


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
