#!/opt/local/bin/python3
"""
V3-EXQ-654f -- arc_062_rule_apprehension:GAP-B behavioural falsifier for
MECH-309 / ARC-062, ported onto the CRF-GATE CALIBRATION AMEND
(crf-availability-maintenance at the CRF locus, UNGATED from GAP-A).

RECOVERY of V3-EXQ-654e (science UNCHANGED): 654e was queued (ree-v3 488ec03) and
claimed+started on the hub ree-cloud-1 (2026-06-17T08:53:05Z) but CRASHED ~3s later
on a TRANSIENT cross-checkout kwarg skew -- the DR-12 (first V4) build added the
e2_forward_pe_per_candidate param to E3TrajectorySelector.select() while the ARC-063
session added the UNCONDITIONAL call-site in agent.select_action(); at 08:53 the hub
runner checkout had the call-site WITHOUT the param. The runner caught the ERROR,
moved 654e to 'completed', and dropped it from the queue with NO manifest -- a phantom
DB row the same-id skip prevents re-running. The skew is now RESOLVED (origin/main +
hub runner checkout consistent: e3_selector param + agent call-site both present), so
654f re-runs the identical experiment under a new id (supersedes 654e).

SUPERSEDES V3-EXQ-654d (FAIL non_contributory, substrate_ceiling, 2026-06-16;
confirmed failure_autopsy_V3-EXQ-654d_2026-06-16). 654d ARMED the GAP-A de-collapse
lever (ARM_STD_G2: modulatory_authority_normalize_basis=std + authority_gain=2.0 +
routed e2_world_forward channel) on both arms AND recorded the discriminator
crf_mean_n_matched. RESULT (the load-bearing disambiguation): the GAP-A conversion
de-collapse was the WRONG LEVER for this gate. ARM_STD_G2 de-collapses the E3
SELECTION channel (consumed_summary spread cleared the 0.05 floor on 2/3 seeds),
but it does NOT touch the CRF rule-MATCH context key -- so the CRF still matched
7-8 rules per tick (crf_mean_n_matched 7.08/7.29/8.70, HIGHER than 654c's >=3) and
gated EVERY one out: gate_and_select theta = 0.15 + 0.25*(n_matched-1) ~= 1.65 >>
maintenance_floor 0.45 -> crf_frac_active = 0.0 on all 3 ARM_ON seeds.
mean_prop_counterfactual_delta = 0.0 confirms the gated-out rule_state never reaches
committed action. The gate-lockout is INDEPENDENT of GAP-A -- it persists on the
seeds where consumed_summary divergence clears the 0.05 floor (consumed_summary
spread = cross-candidate E3 geometry; n_matched = cross-rule CRF context-match
geometry; different vectors, different loci).

THE FIX (the CRF-gate calibration amend, landed ree-v3 main 42895f6 2026-06-17,
routed by the 654d autopsy's user-confirmed repair pathway): three no-op-default
levers at the CRF locus, ungated from GAP-A. ARM_ON here arms all three (inert in
ARM_OFF, where no field is built); the matched ARM_STD_G2 GAP-A constant is RETAINED
on both arms (harmless -- it de-collapses the E3 channel and gives the authority gain
for the C2 committed-argmax lever):
  FAULT 1 -- crf_mature_context_match_threshold=0.7: SHARPEN the gate match cutoff so
    fewer of the clustered context_tags co-match a per-tick context (n_matched 7-8 ->
    ~2-3). Decoupled from retrieval / mint-block; the differentiated pool stays minted.
  FAULT 2a -- crf_tolerance_conflict_cap=3: CAP n_competing so theta is bounded at
    0.15 + 0.25*3 = 0.90 < 1.0 (reachable under crowding spikes).
  FAULT 2b -- crf_maintenance_couple_to_theta=True: floor each maintained rule's
    availability to max(maintenance_floor, theta(n_matched)+eps) so the maintained
    DIFFERENTIATED pool CLEARS the (capped) gate rather than being suppressed
    wholesale -- electing the differentiated set to co-fire.
Substrate-side validation (ree-v3 contracts C20-C24 + activation smoke): on the SAME
crowded maintained pool that legacy gates out (frac_active 0.000), the armed levers
clear the gate (n_matched 8->3, theta 1.90->0.65, frac_active 0.98).

DESIGN CHANGES vs V3-EXQ-654d (everything else matched -- ARM_STD_G2 constant, the
CRF mature+maintenance+persist levers, the C1a-d preconditions, the committed-class
entropy PRIMARY DV, and the three-branch NO-weakens routing are UNCHANGED):
  1. ARM the three CRF-gate-amend levers on the ARM_ON agent (the actual GAP-B fix).
  2. crf_n_matched_last + crf_frac_active + crf_last_theta (the get_state diagnostics)
     remain recorded so the manifest shows the gate firing (n_matched ~2-3, theta < 1.0,
     frac_active >= 0.30) -- the inversion of the 654d lockout signature.

C1c GUARD (now EXPECTED to clear because of the amend): if crf_frac_active < 0.30
STILL, the amend was insufficient -> C1c self-routes substrate_not_ready_requeue
(NEVER a false weakens). CONVERSION-CEILING off-ramp: C1 holds (gate fires) but the
C2 committed-class entropy lift is absent -> conversion_ceiling_persists,
non_contributory + /implement-substrate or /claim-synthesis, NOT a falsification of
MECH-309 / ARC-062.

INHERITED from 654c (unchanged):
  - crf_persist_rules_across_episode_reset=True (cross-episode pool carry-over).
  - TRAINED-bias-head P1 phase (GAP-D): a frozen-encoder P1 window TRAINS the
    SD-033a rule_bias_head via outcome-coupled REINFORCE
    (agent.lateral_pfc.bias_head_parameters() in the P1 optimizer; mirrors
    v3_exq_598b._lpfc_reinforce_loss) so the matured rule_state reaches a non-zero
    per-candidate E3 score-bias.
  - PROPAGATION NON-VACUITY precondition (C1d): ARM_ON committed lateral_pfc bias
    must DIFFER from ARM_OFF on a paired-by-seed basis (the bias REACHES committed
    action). If it does not, self-route substrate_not_ready_requeue.

OFF-RAMP -- the standing committed-action CONVERSION ceiling (PRE-REGISTERED):
REE has a confirmed shared selection-authority CONVERSION ceiling
(behavioral_diversity_isolation:GAP-A; failure_autopsy_V3-EXQ-569g_2026-06-14 +
V3-EXQ-682): a modulatory channel's range REACHES the E3 authority accumulator
but does NOT move the F-dominated committed argmax (F ~ 88-89% of the score per
V3-EXQ-571). 654d ARMS the 684a-validated ARM_STD_G2 conversion lever (std basis,
gain 2.0, routed channel) precisely to overcome this. So if THIS run's MATURED +
DIFFERENTIATED + MAINTAINED rule pool clears the C1c floor AND its bias reaches
committed action (C1d) but STILL fails to lift committed-class entropy (C2) EVEN
WITH ARM_STD_G2 armed, that is the SAME SHARED CONVERSION CEILING persisting under
the validated lever -- NOT a MECH-309 / ARC-062 falsification. The legacy "weakens"
branch is therefore REPLACED here by a conversion-ceiling off-ramp that self-routes
non_contributory and routes to /implement-substrate (the same gain/contrast amend
as GAP-A). The only claim-weighting outcome is the PASS branch (C2 lift =
supports). A C2 fail cannot discriminate "rule-creator adds no diversity" from
"the shared conversion ceiling masks the lift" while the conversion ceiling is
open, so it does NOT weaken the claims.

Single-variable arm contrast
----------------------------
Both arms run the SAME matched stack -- SP-CEM (Layer A), GAP-A shared
candidate-summary source candidate_summary_source="e2_world_forward" (V3-EXQ-649)
with e2 TRAINED ONLINE in P0 (SD-056 contrastive; rollout-norm clamp ON per the
643a numerical-stability lesson), the modulatory-bias-selection-authority gate
(V3-EXQ-643a float32 fix), MECH-341 stratified-select, MECH-313 noise floor, the
V_s minimal stack, AND use_lateral_pfc_analog=True + use_gated_policy=True with
the SD-033a bias head UN-ZEROED (lateral_pfc_train_rule_bias_head=True) so the
rule_state actually reaches a non-zero per-candidate E3 score-bias.

The ONLY swept variable is use_candidate_rule_field (which auto-sets
use_candidate_rule_source on the agent):
  ARM_OFF: use_candidate_rule_field=False -> LateralPFCAnalog rule_state is the
           LEGACY delta_proj(z_delta) + world_pool_weight * world_proj(z_world)
           EMA source -- the 543/598b COLLAPSED rule_state.
  ARM_ON:  use_candidate_rule_field=True (+ crf_persist=True) -> LateralPFCAnalog
           consumes crf_source -- the field's DIFFERENTIATED active-rule-stack
           rule_state, now MATURED across episodes (the literal 598b C3
           trainable_not_monomodal fix on a non-cold-started pool).

Dependent variable -- COMMITTED-CLASS diversity (mechanistically matched)
-------------------------------------------------------------------------
CODE-CONFIRMED (agent._candidate_world_summaries + lateral_pfc.compute_bias):
the per-candidate summaries the bias channels consume are keyed on the candidate's
FIRST action (e2.world_forward(z0, a_first)), and compute_bias broadcasts a single
rule_state across all K candidates. So within a first-action class every candidate
receives an IDENTICAL rule bias. Therefore the rule-creator moves WHICH CLASS is
committed (the committed-class axis), NOT within-class representative selection.
The matched readout is therefore committed-class entropy (PRIMARY DV).
Within-class-representative entropy is retained as a SECONDARY NEGATIVE CONTROL
(expected ~null, confirming the bias is class-keyed).

Phases / budget
---------------
P0 (200 ep, e2 TRAINED online via SD-056 contrastive; bias head NOT trained; field
   matures across episodes via crf_persist + the 666c-validated maintenance levers):
   builds the action-conditional e2.world_forward divergence the GAP-A
   candidate_summary_source AND the e2-sourced CRF context consume, AND matures +
   MAINTAINS a differentiated rule pool. 654c ENLARGED maturation (was 150).
P1 (90 ep, encoder FROZEN, bias head TRAINED via outcome-coupled REINFORCE; field
   continues to mature + maintain): the GAP-D trained-bias-head window. No measurement.
P2 (60 ep, all FROZEN -- e2 + bias head; field persists + maintains; instrumentation
   ON): the behavioural measurement window.
Budget: 2 arms x 3 seeds x 350 ep x 200 steps = 420k steps total (~10 h). The
maturation window (P0+P1) is 290 ep so the matured + maintained pool clears
crf_frac_active >= 0.30 before P2 scoring. The C2 paired lift + C1d propagation
comparison are within-seed (same seed, same machine) -> machine_affinity "any".

Pre-registered acceptance criteria
----------------------------------
  C1 (READINESS / non-vacuity -- protects against a false weakens; C1 fail
      self-routes substrate_not_ready_requeue, NOT a falsification):
     (a) committed-class axis exercisable: frac_pre_ge2 > FRAC_PRE_GE2_FLOOR on a
         majority (>= 2/3) of seeds in BOTH arms.
     (b) GAP-A divergence real: consumed_summary_pairwise_dist_mean >
         CONSUMED_SPREAD_FLOOR (and bounded) on a majority of seeds in BOTH arms.
     (c) ARM_ON manipulation live AND MATURED: per-tick crf_n_active >=
         CRF_N_ACTIVE_FLOOR on >= CRF_FRAC_ACTIVE_FLOOR of P2 ticks (the matured
         pool clears the 654 cold-start floor) AND >= CRF_MIN_MINTED distinct rules
         minted, on a majority of ARM_ON seeds.
     (d) PROPAGATION non-vacuity: paired-by-seed |mean_lateral_pfc_bias_abs(ARM_ON)
         - mean_lateral_pfc_bias_abs(ARM_OFF)| > PROP_NONVAC_FLOOR on a majority of
         seeds (the trained head + matured field produce a DIFFERENT bias than the
         legacy collapsed source -- the seed-42 byte-identical washout did not
         recur). Within-ARM_ON rule_state counterfactual delta reported as a
         supporting diagnostic.
  C2 (PRIMARY -- the falsifier): paired-by-seed lift in committed_class_entropy_nats
     of ARM_ON over ARM_OFF of at least C2_LIFT_MARGIN_NATS on a majority
     (>= 2/3) of seeds.
  C3 (SECONDARY negative control): within-class-representative entropy +
     selected-class entropy + lateral_pfc bias range. Reported, not load-bearing.

Overall outcome (THREE branches; NO weakens -- the conversion ceiling is open)
------------------------------------------------------------------------------
  PASS  = C1 (non-vacuous) AND C2 (committed-class lift) -> supports MECH-309 + ARC-062.
  FAIL (C1 holds, C2 fails) = the matured + maintained + differentiated rule pool's
          bias REACHES committed action (C1d non-vacuous) but does NOT lift
          committed-class diversity -> SHARED selection-authority CONVERSION CEILING
          (behavioral_diversity_isolation:GAP-A; 569g/682), NOT a falsification.
          non_contributory; route to /implement-substrate (gain/contrast amend).
          Do NOT weaken MECH-309 / ARC-062.
  FAIL (C1 fails) = substrate not exercisable / not matured (C1c crf_frac_active <
          0.30) / propagation vacuous (C1d ARM_ON bias == ARM_OFF) / class axis or
          GAP-A divergence absent -> non_contributory / substrate_not_ready_requeue;
          re-queue. Do NOT weaken.

Claims: [MECH-309, ARC-062] (only the PASS branch weights them, as supports; both
FAIL branches are pre-registered non_contributory). experiment_purpose = evidence.
supersedes V3-EXQ-654e (recovery re-queue of the silently-stalled 654e run; science
unchanged; 654e itself superseded the wrong-lever V3-EXQ-654d).

See REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md (GAP-B),
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-654c_2026-06-15.json,
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-569g_2026-06-14.json (the
shared conversion ceiling the off-ramp routes to),
ree-v3/CLAUDE.md (ARC-063 AMEND crf-availability-maintenance + ARC-062 GAP-A/B/C/D
+ SD-056 entries),
experiments/v3_exq_666c_arc063_crf_availability_maintenance_readiness_fracgate.py
(the validated substrate ARM_2 flags ported here),
experiments/v3_exq_598b_gap1_sd033a_bias_head_trainable_ablation.py (P1 bias-head
REINFORCE pattern mirrored here).
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
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_654f_arc062_gapb_rule_apprehension_behavioural_falsifier"
QUEUE_ID = "V3-EXQ-654f"
SUPERSEDES = "V3-EXQ-654e"
CLAIM_IDS: List[str] = ["MECH-309", "ARC-062"]
EXPERIMENT_PURPOSE = "evidence"

# CRF-gate calibration amend levers (crf-availability-maintenance at the CRF locus;
# landed ree-v3 main 42895f6, 2026-06-17). The 654d FIX: 654d proved GAP-A
# de-collapse (ARM_STD_G2, armed below + kept as a matched-stack constant) was the
# WRONG lever -- crf_frac_active stayed EXACTLY 0.0 on all 3 ARM_ON seeds even where
# the consumed_summary spread cleared the 0.05 floor, because the real GAP-B blocker
# is the CRF conflict-gate lockout (7-8 rules co-match -> theta(7)=0.15+0.25*6=1.65
# >> maintenance_floor 0.45 -> every matched rule gated out), INDEPENDENT of GAP-A.
# These three levers (consulted only under crf_mature_pool_dynamics; inert in ARM_OFF
# where no field is built) clear that lockout:
#   FAULT 1 -- sharpen the gate match cutoff so fewer of the clustered context_tags
#     co-match a per-tick context (n_matched 7-8 -> ~2-3).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
#   FAULT 2a -- cap n_competing in theta so it stays reachable (theta_floor +
#     theta_gain*cap = 0.15 + 0.25*3 = 0.90 < 1.0).
CRF_TOLERANCE_CONFLICT_CAP = 3
#   FAULT 2b -- couple maintained availability to the (capped) per-tick theta so the
#     maintained differentiated pool CLEARS the gate (max(maintenance_floor, theta+eps)).
CRF_MAINTENANCE_COUPLE_TO_THETA = True

# crf-availability-maintenance substrate levers (the 666c ARM_2 flags, validated
# V3-EXQ-666c PASS 2026-06-15; substrate_queue crf-availability-maintenance ready=True).
CRF_MAINTENANCE_FLOOR = 0.45   # 666c MAINTENANCE_FLOOR
CRF_MAINTENANCE_DECAY = 0.0    # 666c MAINTENANCE_DECAY (pure activity-silent hold)

# Within-class-representative signature horizon (SECONDARY negative control).
H_SIGNATURE = 3

# C2 (PRIMARY): paired-by-seed committed-class entropy lift of ARM_ON over ARM_OFF.
C2_LIFT_MARGIN_NATS = 0.05
C2_MIN_LIFT_SEEDS = 2  # of 3

# C1(a) readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# C1(b) readiness: GAP-A consumed-summary divergence (649 statistic + 643a ceiling).
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# C1(c) readiness: ARM_ON rule field minted distinct rules AND -- the 654a fix --
# fired a non-zero differentiated rule_state on a meaningful fraction of MATURED P2
# ticks. The 654 cold-start kept crf_frac_active ~0.12; with crf_persist the pool
# persists across episodes so the floor is reachable.
CRF_MIN_MINTED = 2              # distinct rules created over the run
CRF_N_ACTIVE_FLOOR = 1          # >= 1 active rule => non-zero rule_state this tick
CRF_FRAC_ACTIVE_FLOOR = 0.30    # fraction of P2 ticks the field fired a rule_state
CRF_DIST_FLOOR = 1e-3           # reported diagnostic (pinned-distinct separability)

# C1(d) PROPAGATION non-vacuity: ARM_ON mean lateral_pfc bias must differ from
# ARM_OFF (the 654 seed-42 byte-identical washout had a ~4.4e-5 difference).
PROP_NONVAC_FLOOR = 1e-3

# Only classes committed to at least this many P2 ticks feed the unweighted mean
# within-class entropy (secondary negative control).
MIN_TICKS_PER_CLASS = 5

MIN_SEEDS_FOR_PASS = 2  # of 3

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 200         # 654c ENLARGED maturation (was 150): matches the 666c maturation budget; field matures + maintains
P1_BIAS_TRAIN_EPISODES = 90      # frozen-encoder bias-head REINFORCE (GAP-D); field keeps maturing + maintaining
P2_MEASUREMENT_EPISODES = 60     # all frozen; behavioural measurement
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 30

# Matched-stack lever constants (identical on BOTH arms).
USE_MODULATORY_SELECTION_AUTHORITY = True
# 654d: arm the 684a-validated ARM_STD_G2 GAP-A CONVERSION lever (the de-collapse).
# 654c used gain=0.5 + the default "range" basis + no channel routing -> the
# consumed-summary context collapsed (0.0089) and the CRF gate locked out at
# frac_active=0.0. ARM_STD_G2 (std basis, gain 2.0, routed e2_world_forward
# channel) is what 684a PASSed with (conversion_mechanism_identified).
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
USE_MODULATORY_CHANNEL_ROUTING = True
MODULATORY_CHANNEL_ROUTE_SOURCE = "cand_world_summary"
MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

# SD-056 online e2 training (mirror V3-EXQ-649 harness).
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


# IDENTICAL env to V3-EXQ-654 / 614e (SD-054 reef + hazard_food_attraction
# + bipartite layout) -- the behavioural falsifier substrate.
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
        "label": "rule_creator_absent_legacy_collapsed_rule_state",
        "use_candidate_rule_field": False,
    },
    {
        "arm_id": "ARM_ON",
        "label": "rule_creator_present_matured_persisted_crf_source",
        "use_candidate_rule_field": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, use_candidate_rule_field: bool) -> REEAgent:
    """Matched-stack agent; the ONLY varied flag is use_candidate_rule_field.

    Both arms enable use_lateral_pfc_analog + use_gated_policy with the bias head
    UN-ZEROED (lateral_pfc_train_rule_bias_head=True) so the rule_state reaches a
    non-zero per-candidate E3 score-bias AND the head can be trained in P1.
    crf_persist_rules_across_episode_reset=True (the 654a fix; inert in ARM_OFF
    where no field is built) so the ARM_ON rule pool matures across episodes.
    candidate_summary_source = e2_world_forward on BOTH arms (GAP-A; e2 trained
    online in P0).
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
        # modulatory-bias-selection-authority (V3-EXQ-643a float32 fix) ARMED with
        # the 684a-validated ARM_STD_G2 CONVERSION lever (the GAP-A de-collapse):
        # std basis + gain 2.0 + the routed e2_world_forward channel.
        use_modulatory_selection_authority=USE_MODULATORY_SELECTION_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_channel_routing=USE_MODULATORY_CHANNEL_ROUTING,
        modulatory_channel_route_source=MODULATORY_CHANNEL_ROUTE_SOURCE,
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
        # SD-033a LateralPFCAnalog with the bias head UN-ZEROED + trainable (GAP-D)
        # so the head can be trained in P1 and maps rule_state -> a non-zero
        # per-candidate bias.
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
        # 654a fix: cross-episode pool carry-over.
        crf_persist_rules_across_episode_reset=True,
        # 654b mature-pool amend: deadlock-free conflict gate + slow decay +
        # mint-youth protection + e2-world-forward CRF context (differentiation source).
        crf_mature_pool_dynamics=True,
        crf_context_from_e2_world_forward=True,
        # 654c: the 666c-validated availability-maintenance trace (activity-silent
        # hold of a differentiated rule across sparse-matching ticks). This is what
        # 654b lacked -- it is the lever V3-EXQ-666c PASS validated.
        crf_availability_maintenance=True,
        crf_maintenance_floor=CRF_MAINTENANCE_FLOOR,
        crf_maintenance_decay=CRF_MAINTENANCE_DECAY,
        # 654e: the CRF-gate calibration amend -- the actual GAP-B fix (654d proved
        # GAP-A de-collapse alone leaves crf_frac_active=0.0; the gate locks out the
        # whole maintained pool). Sharpen the gate match cutoff (FAULT 1) + cap theta
        # (FAULT 2a) + couple maintained availability to theta (FAULT 2b) so the
        # maintained differentiated pool CLEARS the conflict gate. Inert in ARM_OFF.
        crf_mature_context_match_threshold=CRF_MATURE_CONTEXT_MATCH_THRESHOLD,
        crf_tolerance_conflict_cap=CRF_TOLERANCE_CONFLICT_CAP,
        crf_maintenance_couple_to_theta=CRF_MAINTENANCE_COUPLE_TO_THETA,
        # --- The ONLY swept variable ---
        use_candidate_rule_field=bool(use_candidate_rule_field),
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-649)
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
    """REINFORCE on the SD-033a bias head over stored (candidate_features, sel, return).

    Re-runs compute_bias (differentiable w.r.t. rule_bias_head weights) with the
    CURRENT rule_state on stored candidate summaries, REINFORCE-weighted by the
    episode-return advantage. Mirrors v3_exq_598b._lpfc_reinforce_loss.
    """
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
    """Within-ARM_ON isolation: mean |bias(field rule_state) - bias(zeroed rule_state)|.

    Quantifies how much the field's rule_state actually shapes the per-candidate
    bias (independent of the trained head). Returned as a supporting diagnostic
    alongside the load-bearing cross-arm C1d precondition. Best-effort.
    """
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
        # Restore best-effort if anything went wrong mid-swap.
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
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0

    # SECONDARY negative control (within-class-representative; P2).
    per_class_rep_sigs: Dict[int, Counter] = {}
    all_rep_sigs: Counter = Counter()

    # ARM_ON differentiation + bias diagnostics (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_matched_per_tick: List[int] = []  # 654d: the 654c-autopsy discriminator
    crf_max_pairwise_rule_dist_max = 0.0
    crf_n_minted_total_last = 0
    lateral_pfc_bias_abs_vals: List[float] = []
    prop_counterfactual_deltas: List[float] = []

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

        # P1 per-episode REINFORCE buffers.
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

            # Capture candidate summaries BEFORE select_action for P1 REINFORCE
            # snap (the same e2_world_forward source compute_bias consumes).
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
                if len(pre_e3_classes) >= 2:
                    n_p2_pre_ge2 += 1

                if candidates and len(candidates) >= 2:
                    consumed = _consumed_summaries(agent, candidates)
                    if consumed is not None and torch.isfinite(consumed).all():
                        d = _mean_pairwise_l2(consumed)
                        if math.isfinite(d):
                            consumed_dists.append(d)
                            consumed_dist_max = max(consumed_dist_max, d)

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
                    # 654d: record n_matched -- distinguishes "never matched" from
                    # "matched but gated out" if C1c fails despite de-collapse.
                    crf_n_matched_per_tick.append(
                        int(st.get("crf_n_matched_last", 0))
                    )
                    crf_max_pairwise_rule_dist_max = max(
                        crf_max_pairwise_rule_dist_max,
                        float(st.get("crf_max_pairwise_rule_dist", 0.0)),
                    )
                    crf_n_minted_total_last = int(st.get("crf_n_minted_total", 0))
                    # Within-ARM_ON propagation counterfactual on firing ticks only.
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

            # Capture (z0, a) for the next-tick SD-056 transition.
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

    # 654d discriminator: n_matched aggregates. High mean_n_matched with
    # frac_active=0 == matched-but-gated-out (route to the maintenance-theta amend);
    # mean_n_matched ~0 == never-matched (a different fault).
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

    # Per-seed readiness flags.
    seed_class_axis_exercisable = bool(frac_pre_ge2 > FRAC_PRE_GE2_FLOOR)
    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )

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
        # ----- C1 readiness -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "class_axis_exercisable": seed_class_axis_exercisable,
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        "crf_present": crf_present,
        "crf_mean_n_active": round(mean_crf_n_active, 6),
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_max_pairwise_rule_dist": round(crf_max_pairwise_rule_dist_max, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "crf_differentiated": crf_differentiated,
        # 654d discriminator (matched-but-gated-out vs never-matched).
        "crf_mean_n_matched": round(mean_crf_n_matched, 6),
        "crf_max_n_matched": int(max_crf_n_matched),
        # ----- C1d propagation non-vacuity -----
        "mean_lateral_pfc_bias_abs": round(mean_lateral_pfc_bias_abs, 8),
        "mean_prop_counterfactual_delta": round(mean_prop_counterfactual_delta, 8),
        # ----- C3 SECONDARY negative control (NOT load-bearing) -----
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

    # C1(a): committed-class axis exercisable on majority of seeds in BOTH arms.
    n_off_axis = sum(1 for r in off_rows if r["class_axis_exercisable"])
    n_on_axis = sum(1 for r in on_rows if r["class_axis_exercisable"])
    c1a_holds = bool(n_off_axis >= MIN_SEEDS_FOR_PASS and n_on_axis >= MIN_SEEDS_FOR_PASS)

    # C1(b): GAP-A divergence real on majority of seeds in BOTH arms.
    n_off_gapa = sum(1 for r in off_rows if r["gapa_divergence"])
    n_on_gapa = sum(1 for r in on_rows if r["gapa_divergence"])
    c1b_holds = bool(n_off_gapa >= MIN_SEEDS_FOR_PASS and n_on_gapa >= MIN_SEEDS_FOR_PASS)

    # C1(c): ARM_ON field minted distinct rules AND matured on majority of ARM_ON seeds.
    n_on_differentiated = sum(1 for r in on_rows if r["crf_differentiated"])
    c1c_holds = bool(n_on_differentiated >= MIN_SEEDS_FOR_PASS)

    # C1(d): propagation non-vacuity -- paired |bias_ON - bias_OFF| > floor.
    off_bias_by_seed = {int(r["seed"]): r["mean_lateral_pfc_bias_abs"] for r in off_rows}
    on_bias_by_seed = {int(r["seed"]): r["mean_lateral_pfc_bias_abs"] for r in on_rows}
    prop_diff_by_seed: Dict[int, float] = {}
    n_prop_nonvac_seeds = 0
    for seed in sorted(set(off_bias_by_seed) & set(on_bias_by_seed)):
        diff = abs(on_bias_by_seed[seed] - off_bias_by_seed[seed])
        prop_diff_by_seed[seed] = round(diff, 8)
        if diff > PROP_NONVAC_FLOOR:
            n_prop_nonvac_seeds += 1
    c1d_holds = bool(n_prop_nonvac_seeds >= MIN_SEEDS_FOR_PASS)

    c1_holds = bool(c1a_holds and c1b_holds and c1c_holds and c1d_holds)

    # C2 (PRIMARY): paired-by-seed committed-class entropy lift.
    off_by_seed = {int(r["seed"]): r["committed_class_entropy_nats"] for r in off_rows}
    on_by_seed = {int(r["seed"]): r["committed_class_entropy_nats"] for r in on_rows}
    paired_lifts: Dict[int, float] = {}
    n_lift_seeds = 0
    for seed in sorted(set(off_by_seed) & set(on_by_seed)):
        lift = on_by_seed[seed] - off_by_seed[seed]
        paired_lifts[seed] = round(lift, 6)
        if lift >= C2_LIFT_MARGIN_NATS:
            n_lift_seeds += 1
    c2_holds = bool(n_lift_seeds >= C2_MIN_LIFT_SEEDS)

    off_mean_dv = _mean([r["committed_class_entropy_nats"] for r in off_rows])
    on_mean_dv = _mean([r["committed_class_entropy_nats"] for r in on_rows])

    # within-ARM_ON propagation counterfactual (supporting diagnostic).
    on_prop_cf = [r["mean_prop_counterfactual_delta"] for r in on_rows]
    n_on_prop_cf_nonzero = sum(1 for d in on_prop_cf if d > PROP_NONVAC_FLOOR)

    # ----- Outcome map (THREE branches; NO weakens -- the conversion ceiling is open) -----
    if not c1_holds:
        # C1 fail: substrate not exercisable / not matured (C1c) / propagation vacuous
        # (C1d) / class axis or GAP-A divergence absent -> re-queue, NOT a falsification.
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif c2_holds:
        outcome = "PASS"
        direction = "supports"
        label = "PASS_C1_C2_rule_creator_lifts_committed_class_diversity"
    else:
        # C1 holds (matured + maintained pool AND propagation non-vacuous: the bias
        # REACHES committed action) but C2 fails. While the SHARED selection-authority
        # CONVERSION ceiling (behavioral_diversity_isolation:GAP-A; 569g/682) is open,
        # this cannot discriminate "rule-creator adds no committed-class diversity"
        # from "the F-dominated argmax masks the lift" -> NOT a MECH-309/ARC-062
        # falsification. non_contributory; route to /implement-substrate (the same
        # gain/contrast amend as GAP-A). PRE-REGISTERED off-ramp (see docstring).
        outcome = "FAIL"
        direction = "non_contributory"
        label = "shared_selection_authority_conversion_ceiling_route_implement_substrate"

    evidence_direction_per_claim = {"MECH-309": direction, "ARC-062": direction}

    # ----- interpretation block (preconditions + non-degeneracy) -----
    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "committed_class_axis_exercisable_both_arms",
                "kind": "readiness",
                "description": (
                    "frac of P2 ticks with >= 2 candidate first-action classes "
                    "exceeds floor on a majority of seeds in BOTH arms. SAME-statistic "
                    "family as C2 (class multiplicity bounds class entropy)."
                ),
                "control": "SP-CEM multi-class candidate pool, both arms",
                # COUNT-shaped, INCLUSIVE floor: `met` is c1a_holds, i.e.
                # `n_off_axis >= MIN_SEEDS_FOR_PASS and n_on_axis >= MIN_SEEDS_FOR_PASS`,
                # and min(counts) >= k iff every count >= k, so `measured >= threshold`
                # reproduces `met` EXACTLY. This entry previously reported the min
                # frac_pre_ge2 across ALL cells against FRAC_PRE_GE2_FLOOR, which is
                # strictly HARSHER than "a majority of seeds clear the floor" -- the
                # indexer's authoritative recompute therefore wrongly flagged sound
                # diagnostics precondition_unmet (FALSE_UNMET). The min-fraction number
                # is preserved below as a NON-BOUND diagnostic (extra keys are ignored by
                # the recompute) so no information is lost.
                "measured": float(min(n_off_axis, n_on_axis)),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_frac_pre_ge2": float(
                    min([r["frac_pre_ge2"] for r in (off_rows + on_rows)] or [0.0])
                ),
                "observed_frac_pre_ge2_floor": float(FRAC_PRE_GE2_FLOOR),
                "met": bool(c1a_holds),
            },
            {
                "name": "gapa_consumed_summary_divergence_both_arms",
                "kind": "readiness",
                "description": (
                    "consumed cand_world_summaries (e2.world_forward) per-candidate "
                    "SPREAD clears the floor on a majority of seeds in BOTH arms -- "
                    "the class bias is non-degenerate. Same range statistic the 649 "
                    "GAP-A readiness asserts."
                ),
                "control": "ARM_OFF + ARM_ON: SD-056 e2 trained online in P0; candidate_summary_source=e2_world_forward",
                # COUNT-shaped, INCLUSIVE floor: `met` is c1b_holds, i.e.
                # `n_off_gapa >= MIN_SEEDS_FOR_PASS and n_on_gapa >= MIN_SEEDS_FOR_PASS`,
                # and min(counts) >= k iff every count >= k, so this reproduces `met`
                # EXACTLY. This entry previously reported min(consumed_summary_pairwise_
                # dist_mean) across all cells against CONSUMED_SPREAD_FLOOR -- strictly
                # HARSHER than "a majority of seeds", so the indexer's authoritative
                # recompute wrongly flagged sound diagnostics precondition_unmet
                # (confirmed live on 654h/654j). No single spread statistic CAN reproduce
                # `met`: the per-seed boolean is a CONJUNCTION (consumed_spread_mean >
                # FLOOR and consumed_dist_max < CEIL), and a count over a conjunction
                # does not distribute into per-leg counts. The min-spread number is
                # preserved as a NON-BOUND diagnostic.
                "measured": float(min(n_off_gapa, n_on_gapa)),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_consumed_spread": float(
                    min(
                        [r["consumed_summary_pairwise_dist_mean"] for r in (off_rows + on_rows)]
                        or [0.0]
                    )
                ),
                "observed_consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
                "met": bool(c1b_holds),
            },
            {
                "name": "gapa_consumed_summary_bounded",
                "kind": "readiness",
                "description": (
                    "consumed-summary spread stayed below the 643a explosion ceiling "
                    "(SD-056 online-training numerical stability; rollout-norm clamp ON)."
                ),
                "control": "max consumed_summary_pairwise_dist across all cells",
                "measured": float(
                    max(
                        [r["consumed_summary_pairwise_dist_max"] for r in (off_rows + on_rows)]
                        or [0.0]
                    )
                ),
                "threshold": float(CONSUMED_MAGNITUDE_CEIL),
                # CEILING-shaped, and STRICTLY so: both the per-seed leg and this
                # entry's own `met` are `< CONSUMED_MAGNITUDE_CEIL`. `direction` was
                # already declared; `comparator` was not, so the recompute defaulted to
                # an INCLUSIVE ceiling and would have flipped the measured == threshold
                # boundary. Aggregation is sound as-is: `met` here is the max over all
                # cells against the ceiling, which is exactly what is reported.
                "comparator": "<",
                "direction": "upper",
                "met": bool(
                    max(
                        [r["consumed_summary_pairwise_dist_max"] for r in (off_rows + on_rows)]
                        or [0.0]
                    ) < CONSUMED_MAGNITUDE_CEIL
                ),
            },
            {
                "name": "arm_on_rule_field_differentiated_and_matured",
                "kind": "readiness",
                "description": (
                    "ARM_ON CandidateRuleField minted >= CRF_MIN_MINTED distinct "
                    "rules AND fired a non-zero rule_state on >= CRF_FRAC_ACTIVE_FLOOR "
                    "of P2 ticks (the crf_persist-matured pool clears the 0.30 floor "
                    "the 654 per-episode cold-start could not), on a majority of "
                    "ARM_ON seeds. Below-floor => substrate not matured => "
                    "substrate_not_ready_requeue."
                ),
                "control": "ARM_ON crf frac-active (matured pool) + crf_n_minted_total",
                # COUNT-shaped, INCLUSIVE floor: `met` is c1c_holds, i.e.
                # `n_on_differentiated >= MIN_SEEDS_FOR_PASS` -- a COUNT over ARM_ON
                # seeds, so this single count reproduces `met` EXACTLY. This entry
                # previously reported the MAX crf_frac_active_ge_floor across ARM_ON
                # cells, which is strictly LAXER than "a majority of seeds" (one good
                # seed clears it), so the recompute could silently clear a genuine
                # premise failure (MISSED_UNMET). No frac statistic CAN reproduce `met`
                # either way: crf_differentiated is a CONJUNCTION (crf_present and
                # n_minted >= CRF_MIN_MINTED and frac_active >= CRF_FRAC_ACTIVE_FLOOR).
                # The max-frac number is preserved as a NON-BOUND diagnostic.
                "measured": float(n_on_differentiated),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_max_crf_frac_active": float(
                    max([r["crf_frac_active_ge_floor"] for r in on_rows] or [0.0])
                ),
                "observed_crf_frac_active_floor": float(CRF_FRAC_ACTIVE_FLOOR),
                "met": bool(c1c_holds),
            },
            {
                "name": "propagation_non_vacuity_arm_on_bias_differs_from_arm_off",
                "kind": "readiness",
                "description": (
                    "paired-by-seed |mean_lateral_pfc_bias_abs(ARM_ON) - "
                    "mean_lateral_pfc_bias_abs(ARM_OFF)| > floor on a majority of "
                    "seeds. The trained head + matured field produce a DIFFERENT bias "
                    "than the legacy collapsed source -- the 654 seed-42 byte-identical "
                    "(~4.4e-5) propagation washout did not recur. Below-floor => "
                    "vacuous propagation => substrate_not_ready_requeue (do NOT score "
                    "the falsifier). Supported by the within-ARM_ON rule_state "
                    "counterfactual delta (zeroing rule_state changes the bias)."
                ),
                "control": "paired ARM_ON vs ARM_OFF mean lateral_pfc bias on the matched stack",
                # COUNT-shaped, INCLUSIVE floor: `met` is c1d_holds, i.e.
                # `n_prop_nonvac_seeds >= MIN_SEEDS_FOR_PASS`, so this count reproduces
                # `met` EXACTLY. This entry previously reported the MAX paired diff
                # across seeds against PROP_NONVAC_FLOOR, which is strictly LAXER than
                # "a majority of seeds" (a single non-vacuous seed clears it), so the
                # recompute could silently clear a genuine premise failure
                # (MISSED_UNMET). The max-diff number is preserved as a NON-BOUND
                # diagnostic.
                "measured": float(n_prop_nonvac_seeds),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_max_paired_bias_diff": float(
                    max(list(prop_diff_by_seed.values()) or [0.0])
                ),
                "observed_prop_nonvac_floor": float(PROP_NONVAC_FLOOR),
                "met": bool(c1d_holds),
            },
        ],
        "criteria": [
            {
                "name": "C2_committed_class_entropy_lift",
                "load_bearing": True,
                "passed": bool(c2_holds),
            },
        ],
        "criteria_non_degenerate": {
            "C1a_class_axis_exercisable": bool(c1a_holds),
            "C1b_gapa_divergence": bool(c1b_holds),
            "C1c_arm_on_differentiated_matured": bool(c1c_holds),
            "C1d_propagation_non_vacuity": bool(c1d_holds),
            "C1d_within_arm_on_rule_state_counterfactual_nonzero": bool(
                n_on_prop_cf_nonzero >= MIN_SEEDS_FOR_PASS
            ),
            "C2_paired_lift": bool(c2_holds),
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
            "min_ticks_per_class": int(MIN_TICKS_PER_CLASS),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "lr_lpfc_bias": float(LR_LPFC_BIAS),
            "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
            "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
            "sd056_weight": float(SD056_WEIGHT),
            "crf_persist_rules_across_episode_reset": True,
        },
        "acceptance_criteria": {
            "C1_substrate_exercisable_and_manipulation_live": c1_holds,
            "C1a_class_axis_exercisable_both_arms": c1a_holds,
            "C1a_n_off_axis": int(n_off_axis),
            "C1a_n_on_axis": int(n_on_axis),
            "C1b_gapa_divergence_both_arms": c1b_holds,
            "C1b_n_off_gapa": int(n_off_gapa),
            "C1b_n_on_gapa": int(n_on_gapa),
            "C1c_arm_on_rule_field_differentiated_matured": c1c_holds,
            "C1c_n_on_differentiated": int(n_on_differentiated),
            "C1d_propagation_non_vacuity": c1d_holds,
            "C1d_n_prop_nonvac_seeds": int(n_prop_nonvac_seeds),
            "C1d_prop_diff_by_seed": prop_diff_by_seed,
            "C1d_n_on_within_arm_counterfactual_nonzero": int(n_on_prop_cf_nonzero),
            "C2_committed_class_lift": c2_holds,
            "C2_n_lift_seeds": int(n_lift_seeds),
            "C2_paired_lifts_by_seed": paired_lifts,
            "C2_off_mean_committed_class_entropy": round(off_mean_dv, 6),
            "C2_on_mean_committed_class_entropy": round(on_mean_dv, 6),
        },
        "secondary_negative_control_not_load_bearing": {
            "note": (
                "Within-class-representative entropy is a NEGATIVE CONTROL: the rule "
                "bias is class-keyed (per-candidate summary first-action-keyed; "
                "compute_bias broadcasts one rule_state across K), so it cannot move "
                "within-class selection -> ARM_ON ~ ARM_OFF is EXPECTED here, "
                "confirming the rule-creator's signal lives in the committed-class "
                "axis (the load-bearing C2 DV)."
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
            "arm_on_within_arm_prop_counterfactual_delta_mean": round(
                _mean(on_prop_cf), 8
            ),
        },
        "interpretation_grid": {
            "PASS_C1_C2": (
                "The non-Bayesian rule-creator's DIFFERENTIATED + MATURED rule_state "
                "propagates (via the trained bias head) to greater committed-class "
                "diversity than the collapsed legacy baseline. supports MECH-309 + "
                "ARC-062. Route to /governance: MECH-309 / ARC-062 supports evidence; "
                "consider GAP-B closure."
            ),
            "FAIL_C1_holds_C2_fails": (
                "Class axis exercisable, GAP-A divergence real, ARM_ON minted distinct "
                "rules + matured + MAINTAINED (frac_active >= 0.30), propagation "
                "non-vacuous (trained head, ARM_ON committed bias differs from "
                "ARM_OFF -- the bias REACHES committed action), but the differentiated "
                "rule_state adds no marginal committed-class diversity. This is the "
                "SHARED selection-authority CONVERSION ceiling "
                "(behavioral_diversity_isolation:GAP-A; failure_autopsy_V3-EXQ-569g + "
                "V3-EXQ-682: the channel range reaches the E3 accumulator but does not "
                "move the F-dominated committed argmax), NOT a MECH-309 / ARC-062 "
                "falsification. non_contributory; route to /implement-substrate (the "
                "same gain/contrast amend as GAP-A). Do NOT weaken the claims."
            ),
            "FAIL_C1_substrate_not_ready_requeue": (
                "The committed-class axis was not exercisable, and/or GAP-A "
                "consumed-summary divergence was absent, and/or ARM_ON did not mature "
                "a differentiated pool (frac_active < 0.30 -- crf_persist did not "
                "deliver), and/or propagation was vacuous (ARM_ON bias == ARM_OFF -- "
                "the 654 washout recurred). The falsifier could not run -- NOT an "
                "MECH-309 / ARC-062 falsification. Route to substrate enrichment / "
                "re-queue; do NOT weaken the claims."
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
        # Mirror the self-route block at the manifest top level so the indexer's
        # _compute_adjudication (build_experiment_indexes.py: reads
        # manifest.get("interpretation") at the TOP level) surfaces the label.
        # The block is also nested under result[...]; these two keys make it
        # visible to top-level readers (governance / pending_review) without
        # diving into the nested result dict. Emit-hygiene only (the adjudication
        # FLAG stays "n/a" for experiment_purpose="evidence"); per
        # failure_autopsy_V3-EXQ-654f_2026-06-18 Section 4.
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "evidence_direction_note": (
            f"V3-EXQ-654f arc_062 GAP-B behavioural falsifier (MECH-309 / ARC-062), "
            f"supersedes V3-EXQ-654e (recovery re-queue of the silently-stalled 654e run; science unchanged), ported onto the CRF-GATE CALIBRATION AMEND "
            f"(crf-availability-maintenance at the CRF locus, ungated from GAP-A; landed "
            f"ree-v3 main 42895f6). 654d proved the GAP-A de-collapse was the WRONG lever "
            f"(crf_frac_active 0.0 even where consumed_summary cleared the 0.05 floor; the "
            f"CRF conflict gate locked out 7-8 co-matching rules, theta(7)=1.65 >> floor "
            f"0.45, INDEPENDENT of GAP-A). ARM_ON arms the three CRF-gate-amend levers "
            f"(crf_mature_context_match_threshold={CRF_MATURE_CONTEXT_MATCH_THRESHOLD} + "
            f"crf_tolerance_conflict_cap={CRF_TOLERANCE_CONFLICT_CAP} + "
            f"crf_maintenance_couple_to_theta={CRF_MAINTENANCE_COUPLE_TO_THETA}) so the "
            f"maintained differentiated pool CLEARS the gate. The matched ARM_STD_G2 GAP-A "
            f"constant (modulatory_authority_normalize_basis={MODULATORY_AUTHORITY_NORMALIZE_BASIS}"
            f" + authority_gain={MODULATORY_AUTHORITY_GAIN} + use_modulatory_channel_routing="
            f"{USE_MODULATORY_CHANNEL_ROUTING} source={MODULATORY_CHANNEL_ROUTE_SOURCE}) is "
            f"RETAINED on BOTH arms (de-collapses the E3 channel + gives the C2 authority gain). "
            f"Single-variable: ARM_OFF (use_candidate_rule_field=False, legacy collapsed "
            f"rule_state) vs ARM_ON (use_candidate_rule_field=True + crf_persist + the 666c "
            f"maintenance levers + the CRF-gate amend, MATURED + MAINTAINED + NOW-ACTIVE "
            f"differentiated crf_source), "
            f"both on the matched 649/643a/SD-056/SP-CEM/MECH-341 stack with the SD-033a "
            f"bias head un-zeroed AND TRAINED in a frozen-encoder P1 REINFORCE window "
            f"(GAP-D); ENLARGED P0 200 ep. PRIMARY DV = COMMITTED-CLASS entropy. C1 "
            f"(non-vacuity) = committed-class axis exercisable AND GAP-A divergence real "
            f"(both arms) AND ARM_ON matured (crf_frac_active >= {CRF_FRAC_ACTIVE_FLOOR}) "
            f"AND propagation non-vacuous (ARM_ON committed bias differs from ARM_OFF -- "
            f"the bias reaches committed action); C2 (PRIMARY) = paired-by-seed ARM_ON > "
            f"ARM_OFF committed-class entropy lift (>= {C2_MIN_LIFT_SEEDS}/3 seeds). "
            f"interpretation_label={result['interpretation_label']}. "
            f"C1={result['acceptance_criteria']['C1_substrate_exercisable_and_manipulation_live']}, "
            f"C2={result['acceptance_criteria']['C2_committed_class_lift']}. ONLY the PASS "
            f"branch weights MECH-309/ARC-062 (as supports); C1-fail self-routes "
            f"substrate_not_ready_requeue AND C1-holds-C2-fail self-routes the shared "
            f"selection-authority CONVERSION ceiling (GAP-A 569g/682) -> /implement-substrate "
            f"-- BOTH non_contributory, NEITHER a falsification (NO weakens branch)."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "arms": "ARM_OFF (rule field off) vs ARM_ON (rule field on + crf_persist + 666c maintenance levers)",
            "swept_variable": "use_candidate_rule_field",
            "crf_persist_rules_across_episode_reset": True,
            "crf_mature_pool_dynamics": True,
            "crf_context_from_e2_world_forward": True,
            "crf_availability_maintenance": True,
            "crf_maintenance_floor": float(CRF_MAINTENANCE_FLOOR),
            "crf_maintenance_decay": float(CRF_MAINTENANCE_DECAY),
            "crf_substrate_validated_by": "V3-EXQ-666c PASS 2026-06-15 (ready=True)",
            # 654e CRF-gate calibration amend (the GAP-B fix; ungated from GAP-A):
            "crf_mature_context_match_threshold": float(CRF_MATURE_CONTEXT_MATCH_THRESHOLD),
            "crf_tolerance_conflict_cap": int(CRF_TOLERANCE_CONFLICT_CAP),
            "crf_maintenance_couple_to_theta": bool(CRF_MAINTENANCE_COUPLE_TO_THETA),
            "crf_gate_amend_landed": "ree-v3 main 42895f6 (2026-06-17)",
            "matched_stack": (
                "SP-CEM + candidate_summary_source=e2_world_forward (GAP-A/649, e2 "
                "trained online in P0) + use_modulatory_selection_authority (643a) + "
                "MECH-341 stratified + MECH-313 noise floor + V_s minimal + "
                "use_gated_policy + use_lateral_pfc_analog "
                "(lateral_pfc_train_rule_bias_head=True, TRAINED in P1) + SD-056 all levers"
            ),
            "primary_dv": "committed-class entropy (class-keyed rule bias)",
            "secondary_negative_control": "within-class-representative entropy (expected ~null)",
            "phases": "P0 e2-train (field matures) -> P1 frozen-encoder bias-head REINFORCE -> P2 frozen measurement",
            "p1_bias_head_trained_via_reinforce": True,
            "propagation_non_vacuity_precondition": True,
            "e2_trained_in_p0_frozen_in_p1_p2": True,
            "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
            "use_modulatory_channel_routing": USE_MODULATORY_CHANNEL_ROUTING,
            "modulatory_channel_route_source": MODULATORY_CHANNEL_ROUTE_SOURCE,
            "gapa_decollapse_lever": "ARM_STD_G2 (684a PASS conversion_mechanism_identified)",
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
    parser = argparse.ArgumentParser(description="V3-EXQ-654f GAP-B falsifier (on the CRF-gate calibration amend; ungated from GAP-A)")
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
        f"C1={result['acceptance_criteria']['C1_substrate_exercisable_and_manipulation_live']} "
        f"C2={result['acceptance_criteria']['C2_committed_class_lift']} "
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
