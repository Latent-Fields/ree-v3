#!/opt/local/bin/python3
"""
V3-EXQ-710 -- MECH-140 x MECH-450 DISINHIBITORY SOFT-COMPETITIVE SETTLING VALIDATION.

A 3-arm governance-evidence falsifier (EXPERIMENT_PURPOSE=evidence) for the disinhibitory
soft-competitive settling substrate that landed 2026-07-02 (ree-v3 main 8cc42bc), on the SAME
GAP-A reef-bipartite foraging substrate as V3-EXQ-709 but with a SINGLE-ARENA conversion envelope
(loop segregation OFF), so the settling runs in the WITHIN-ELIGIBLE single-arena path.

THE MECHANISM (MECH-140 x MECH-450)
-----------------------------------
The substrate replaces the one-shot within-eligible committed argmin (a hard argmin over an
F-dominated modulatory field ALWAYS returns the F-winner -- the MECH-439 conversion ceiling) with
a few rounds of PARAMETER-FREE soft-competitive lateral-inhibition SETTLING over the
`_modulatory_accum[eligible_idx]` field BEFORE the commit (e3_selector._soft_competitive_settle):

    x       = -field                          # activation (higher = better)
    for r in range(R):
        support = softmax(x / T)              # graded competitive support, ALL > 0
        inhib   = gain * (K @ support)        # lateral inhibition from competitors
        x       = x - inhib                   # disinhibit winner, reduce losers
    return -x                                 # back to COST units for the argmin

K is the PARAMETER-FREE class-surround kernel: 1.0 within a first-action class, `cross_class` (< 1)
across classes, 0 on the diagonal -- surround inhibition between competing motor programs (Mink
1996). Because K encodes candidate-vs-candidate STRUCTURE, the settling can REORDER: a candidate
crowded by same-class rivals accrues more lateral inhibition than an isolated slightly-worse one
and can lose to it (MECH-140 soft-competitive disinhibition -- graded, never winner-take-all; the
attractor flip the one-shot argmin structurally lacks -- MECH-450). WITH cross_class == 1.0 the
kernel is UNIFORM (all off-diagonal equal) -> the settling becomes RANK-PRESERVING (it cannot
reorder / flip the committed winner) -- the ablated inhibition-on-inhibition STRUCTURAL edge.

Does the disinhibitory soft-competitive settling lift committed-action diversity where the one-shot
argmin plateaued -- and is the STRUCTURED class-surround edge LOAD-BEARING (a uniform kernel
collapses the conversion while the base graded settling machinery still runs)?

THE 3 ARMS (the ONLY swept factor is the soft-competitive settling)
-------------------------------------------------------------------
  A0_OFF      : use_soft_competitive_settling=False. The baseline / plateau (one-shot argmin).
  A1_INTACT   : use_soft_competitive_settling=True, gain=1.0, rounds=3, temperature=1.0,
                cross_class=0.25 -- the STRUCTURED class-surround kernel = the intact
                inhibition-on-inhibition edge.
  A2_ABLATED  : use_soft_competitive_settling=True, same gain/rounds/temperature,
                cross_class=1.0 -- the UNIFORM kernel = the STRUCTURAL edge ablated ->
                rank-preserving (the settling machinery runs but cannot reorder).

BOTH settling arms carry the SAME landed 569i top-k + MECH-448 demotion + Go/No-Go SOTA conversion
envelope as a MATCHED CONSTANT; the ONLY difference is whether the settling runs, and (A1 vs A2)
whether its kernel is structured or uniform. Loop segregation is OFF (single arena, so the settling
runs the within-eligible path), and the LEARNED W_lat settling (use_learned_settling_step) and the
LEARNED cross-loop arbitration (use_learned_cross_loop_arbitration) are OFF on all arms so the ONLY
active settling is MY parameter-free soft-competitive settling.

6 seeds. PRIMARY DV = committed-action-class entropy (nats), measured over P2.
claim_ids = [MECH-140, MECH-450, MECH-439]. experiment_purpose = evidence (governance falsifier).

PRE-REGISTERED OUTCOME MAP (decisive either way)
------------------------------------------------
  C1 (VALIDATION): A1_INTACT committed-class entropy STRICT-ABOVE A0_OFF + margin on a
  strict-majority (>= 2/3) of DIVERGENT seeds.

  C1 PASS -> the disinhibitory soft-competitive settling LIFTS the F-dominance conversion ceiling:
      MECH-140: supports  (soft-competitive disinhibition converts committed-action diversity)
      MECH-450: supports  (the bounded recurrent settling step over the eligible field converts)
      MECH-439: weakens   (the ceiling is LIFTABLE by the settling)
      overall : mixed     (MECH-439 opposes MECH-140/MECH-450 by construction)

  C1 FAIL (non-vacuity met) -> the settling ALSO fails to lift:
      MECH-140: weakens   (soft-competitive disinhibition does not re-weight enough to convert)
      MECH-450: weakens   (even the settling step does not convert)
      MECH-439: supports  (the ceiling is INTRINSIC)
      overall : mixed

  C2 (PLOS-BIOLOGY ABLATION dissociation, SECONDARY): A1_INTACT committed-class entropy STRICT-ABOVE
  A2_ABLATED + margin on a strict-majority of divergent seeds. Tests that the STRUCTURED
  class-surround edge (the inhibition-on-inhibition structure) is LOAD-BEARING: ablating it to a
  uniform (rank-preserving) kernel collapses the conversion while the base graded settling machinery
  still runs -- the mouse-V1 inhibition-on-inhibition-silencing signature. C2 is REPORTED and feeds
  the interpretation; it does NOT flip the overall per-claim directions. If C1 PASSES AND C2 PASSES:
  "structured disinhibition is load-bearing -- the PLOS ablation signature".

NON-VACUITY GATES (self-route substrate_not_ready_requeue if unmet; NEVER a false weakens)
------------------------------------------------------------------------------------------
On A1_INTACT, over a strict-majority of DIVERGENT seeds:
  (M) the settling ACTUALLY MOVED the field: mean soft_competitive_settling_round_delta >
      ROUND_DELTA_FLOOR. If the field never moved, the settling was a no-op and any "no lift" is
      meaningless (LEARNED-arm bit-identical trap).
  (E) the eligible set is NON-DEGENERATE: the F-eligibility envelope excluded >= 1 candidate on a
      non-trivial fraction of ticks (f_eligibility_excluded_count > 0), so there was a real
      within-eligible competition for the settling to act on.
Plus the standard GAP-A pool-divergence / substrate-liveness preconditions 709 checks (enough
divergent seeds, committed-class axis exercisable, finer channels dissociable + delta_t non-flat,
CRF matured).

DIAGNOSTIC (not a hard gate): A2_ABLATED's uniform kernel should be rank-preserving -> A2 committed-
class entropy should be ~<= A1_INTACT. If A2 substantially EXCEEDS A1 the uniform kernel unexpectedly
reordered -- flagged in the manifest (a2_unexpectedly_reordered).

LIMITATION NOTE (reorder-rate): the selector does NOT expose the pre-settle within-eligible argmin
as a running diagnostic, so a literal "committed winner != one-shot argmin" reorder rate cannot be
recorded per-tick in-run. We record instead (a) the settling round_delta (the field-movement /
non-vacuity signal, from _soft_competitive_settle) and (b) the f_eligibility diagnostics
(excluded_count > 0 = real within-eligible competition; winner_neq_f_argmin = F demoted at commit).
The behavioural reorder is instead read at the DV level (A1 vs A0 committed-class entropy) and the
structural-edge dissociation (A1 vs A2).

Phased training kept (P0 e2-train -> P1 frozen-encoder bias-head REINFORCE -> P2 e2+bias frozen,
finer channels KEEP adapting) for a fair comparison. MECH-094: the settling is WAKING-ONLY (a
simulation/replay tick does not settle). Safety inherited: the settling transforms ONLY the F +
MECH-448/449 Go/No-Go eligible subset -- a suppressed candidate is never re-admitted however the
field moves.

See ree-v3/ree_core/predictors/e3_selector.py (_soft_competitive_settle + the single-arena
    within-eligible wiring at the `use_soft_competitive_settling` block after `_lateral_settle`),
    ree-v3/ree_core/utils/config.py (use_soft_competitive_settling config block),
    ree-v3/tests/contracts/test_soft_competitive_settling.py (the flag-behaviour contracts),
    experiments/v3_exq_709_learned_cross_loop_arbitration_validation.py (matched-substrate template).
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


EXPERIMENT_TYPE = "v3_exq_710_disinhibitory_soft_competitive_settling_validation"
QUEUE_ID = "V3-EXQ-710"
SUPERSEDES = None   # tests a DIFFERENT mechanism than 709 (soft-competitive settling, single arena)
BACKLOG_ID = None
CLAIM_IDS: List[str] = ["MECH-140", "MECH-450", "MECH-439"]
EXPERIMENT_PURPOSE = "evidence"   # governance-evidence falsifier

# softplus-unity init for w_chan_finer (softplus(_FCG_W_INIT) == 1.0).
_FCG_W_INIT = math.log(math.e - 1.0)

# CRF-gate calibration levers (matured CRF stack; ported verbatim from 709/707b/700c,
# matched on all arms -- the differentiated conversion source).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# ----- Acceptance thresholds (pre-registered) -----
# C1 conversion + C2 ablation dissociation: strict-above margin on committed-class entropy (nats).
CONVERSION_MARGIN = 0.05
ABLATION_MARGIN = 0.05
# A2-vs-A1 rank-preserving diagnostic: A2 should NOT substantially exceed A1 (uniform kernel is
# rank-preserving). If mean A2 entropy > mean A1 entropy + this slack, flag a2_unexpectedly_reordered.
A2_REORDER_FLAG_SLACK = 0.10

# ----- Per-seed-divergent gating (709-style) -----
MIN_DIVERGENT_SEEDS = 3          # of 6: fewer divergent seeds => substrate_not_ready_requeue
DIVERGENT_PASS_FRACTION = 0.5    # strict-majority-ish gate within the divergent seeds
MIN_SEEDS_FOR_PASS = 2           # absolute floor of divergent seeds clearing a criterion

# ----- MECH-140 x MECH-450 soft-competitive settling NON-VACUITY thresholds (the mechanism gate) -----
# (M) the settling ACTUALLY MOVED the field: mean soft_competitive_settling_round_delta over waking
# committed selections > this floor. At gain 0.0 (or when the field never moves) round_delta is
# 0.0/-1.0 -> a "no lift" would be meaningless; above this floor == the settling genuinely acted.
ROUND_DELTA_FLOOR = 1e-3
# (E) the eligible set is NON-DEGENERATE: the F-eligibility envelope excluded >= 1 candidate on a
# non-trivial fraction of P2 ticks (there was a real within-eligible competition to settle).
ELIG_EXCLUDED_FRAC_FLOOR = 0.05

# C1(a) readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# Non-vacuity (b): GAP-A consumed-summary divergence (649 statistic + 643a ceiling).
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# Non-vacuity: delta_t carries cross-tick variance (outcome variance to learn from).
DELTA_T_STD_FLOOR = 1e-4
# Non-vacuity: the finer w_chan_finer entries MOVED + are DISSOCIABLE (cross-channel range).
W_CHAN_FINER_RANGE_FLOOR = 1e-4

# CRF maturity readiness (matched constant; the differentiated source must be present).
CRF_MIN_MINTED = 2
CRF_N_ACTIVE_FLOOR = 1
CRF_FRAC_ACTIVE_FLOOR = 0.30

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

# ----- MECH-140 x MECH-450 soft-competitive settling knobs (the SWEPT factor) -----
SCS_GAIN = 1.0                    # inhibition strength (> 0 to activate)
SCS_ROUNDS = 3                    # mutual-inhibition rounds per tick
SCS_TEMPERATURE = 1.0             # softmax temperature for the graded support
SCS_CROSS_CLASS_INTACT = 0.25    # STRUCTURED class-surround kernel (intact edge)
SCS_CROSS_CLASS_ABLATED = 1.0    # UNIFORM kernel (structural edge ablated -> rank-preserving)

# ----- ARC-108 JOB-1 learned-gating knobs (substrate defaults; matched on all arms) -----
LCG_ETA = 0.01
LCG_ELIG_DECAY = 0.9
LCG_VALUE_BASELINE_BETA = 0.05
LCG_ASYM_POTENTIATION = 1.0
LCG_ASYM_DEPRESSION = 0.5

# SD-056 online e2 training (mirror 709/707b/700c).
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

# P1 bias-head REINFORCE training (mirror 709/707b/700c).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

# Matched-stack lever constants (identical on all arms; the landed 569i top-k + MECH-448 + Go/No-Go
# SOTA conversion envelope).
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


# IDENTICAL env to 709/707b/700c (the GAP-A reef-bipartite foraging bank).
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


# The 3 arms. ALL carry the SAME landed single-arena conversion envelope as a MATCHED CONSTANT; the
# ONLY swept factor is the soft-competitive settling (OFF / INTACT structured kernel / ABLATED
# uniform kernel).
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "A0_OFF",
        "label": "soft_competitive_settling_off_one_shot_argmin_baseline_plateau",
        "scs_on": False,
        "scs_cross_class": SCS_CROSS_CLASS_INTACT,   # unused when OFF
    },
    {
        "arm_id": "A1_INTACT",
        "label": "soft_competitive_settling_intact_structured_class_surround_kernel_cross025",
        "scs_on": True,
        "scs_cross_class": SCS_CROSS_CLASS_INTACT,
    },
    {
        "arm_id": "A2_ABLATED",
        "label": "soft_competitive_settling_ablated_uniform_kernel_rank_preserving_cross100",
        "scs_on": True,
        "scs_cross_class": SCS_CROSS_CLASS_ABLATED,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Matched-stack agent. The landed 569i top-k + MECH-448 demotion + Go/No-Go SOTA conversion
    envelope + finer-channel gating + the diversity stack + matured CRF are MATCHED CONSTANTS on ALL
    arms. Loop segregation is OFF (single arena -- so the settling runs the within-eligible path),
    and the LEARNED W_lat settling (use_learned_settling_step) and the LEARNED cross-loop
    arbitration (use_learned_cross_loop_arbitration) are OFF so the ONLY active settling is the
    parameter-free MECH-140 x MECH-450 soft-competitive settling. The ONLY swept factor is
    use_soft_competitive_settling (+ the cross_class kernel structure: 0.25 intact / 1.0 ablated)."""
    scs_on = bool(arm["scs_on"])
    scs_cross = float(arm["scs_cross_class"])
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
        use_noise_floor=False,
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
        # --- MECH-451 FINER separately-learnable channels (ON on all arms; matched constant). ---
        use_finer_channel_gating=True,
        use_learned_channel_gating=False,
        learned_channel_gating_eta=LCG_ETA,
        learned_channel_gating_elig_decay=LCG_ELIG_DECAY,
        learned_channel_value_baseline_beta=LCG_VALUE_BASELINE_BETA,
        learned_channel_asym_potentiation=LCG_ASYM_POTENTIATION,
        learned_channel_asym_depression=LCG_ASYM_DEPRESSION,
        learned_channel_rpe_mode="signed",
        # --- LEARNED W_lat settling: OFF on ALL arms so the ONLY active settling is MY
        # parameter-free soft-competitive settling (isolate the mechanism). ---
        use_learned_settling_step=False,
        # --- ARC-110 loop segregation: OFF on ALL arms (SINGLE ARENA) -- so the soft-competitive
        # settling runs the within-eligible single-arena path (transforms _modulatory_accum
        # [eligible_idx]) that I am validating. ---
        use_loop_segregation=False,
        # --- ARC-108 x ARC-110 learned cross-loop arbitration: OFF on ALL arms (single arena;
        # isolate my mechanism from the 709 learned-arbitration substrate). ---
        use_learned_cross_loop_arbitration=False,
        # --- MECH-140 x MECH-450 DISINHIBITORY SOFT-COMPETITIVE SETTLING -- THE SWEPT FACTOR.
        # OFF -> one-shot within-eligible argmin (A0 baseline / plateau). ON with cross_class=0.25
        # -> the STRUCTURED class-surround kernel (A1 intact edge). ON with cross_class=1.0 -> the
        # UNIFORM kernel (A2 ablated -> rank-preserving). gain > 0 activates; at gain 0.0 it is an
        # EXACT no-op even when the flag is on. ---
        use_soft_competitive_settling=scs_on,
        soft_competitive_settling_gain=(SCS_GAIN if scs_on else 0.0),
        soft_competitive_settling_rounds=SCS_ROUNDS,
        soft_competitive_settling_temperature=SCS_TEMPERATURE,
        soft_competitive_settling_cross_class=scs_cross,
    )
    return REEAgent(cfg)


def _arm_config_slice(
    arm: Dict[str, Any],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Declared fingerprint slice: ONLY what an arm's computation reads -- the swept settling
    flags (scs_on + cross_class kernel) + the matched single-arena envelope every arm runs + the
    env + the schedule. NEVER acceptance thresholds. Minted reuse-ELIGIBLE
    (include_driver_script_in_hash=False + config_slice_declared) so a later iteration that
    constructs the SAME arm from the same substrate + config_slice + seed can HIT (mint-as-you-go;
    terminality is unknowable)."""
    return {
        "arm_id": arm["arm_id"],
        "scs_on": bool(arm["scs_on"]),
        "scs_cross_class": float(arm["scs_cross_class"]) if arm["scs_on"] else None,
        "scs_gain": float(SCS_GAIN) if arm["scs_on"] else 0.0,
        "scs_rounds": int(SCS_ROUNDS),
        "scs_temperature": float(SCS_TEMPERATURE),
        "use_loop_segregation": False,
        "use_learned_settling_step": False,
        "use_learned_cross_loop_arbitration": False,
        "use_finer_channel_gating": True,
        "learned_channel_gating_eta": LCG_ETA,
        "lcg_elig_decay": LCG_ELIG_DECAY,
        "lcg_value_baseline_beta": LCG_VALUE_BASELINE_BETA,
        "lcg_asym_potentiation": LCG_ASYM_POTENTIATION,
        "lcg_asym_depression": LCG_ASYM_DEPRESSION,
        "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
        "use_f_eligibility_adaptive_floor": bool(USE_F_ELIGIBILITY_ADAPTIVE_FLOOR),
        "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
        "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
        "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
        "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
        "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
        "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
        "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
        "modulatory_channel_route_source": str(MODULATORY_CHANNEL_ROUTE_SOURCE),
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


# ---------------------------------------------------------------------------
# SD-056 online e2 training (verbatim from 709)
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
# P1 bias-head REINFORCE training (verbatim from 709)
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
    scs_on = bool(arm["scs_on"])

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
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0
    ep_reward_p2_sum = 0.0
    n_p2_episodes_done = 0

    # CRF maturity readiness (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_minted_total_last = 0

    # ----- MECH-451 finer-channel learning diagnostics (accumulated all phases; matched constant) -----
    fcg_delta_ts: List[float] = []
    fcg_w_chan_finer_range_max = 0.0
    fcg_w_chan_finer_std_max = 0.0

    # ----- MECH-140 x MECH-450 soft-competitive settling diagnostics (P2 select ticks) -----
    scs_active_ticks = 0                # ticks the settling actually RAN (>= 2 eligible, waking)
    scs_round_delta_sum = 0.0          # sum of soft_competitive_settling_round_delta over active ticks
    scs_round_delta_moved_ticks = 0    # ticks where round_delta > ROUND_DELTA_FLOOR (field moved)
    scs_round_delta_peak = 0.0
    # ----- MECH-448 f-eligibility (E) non-degeneracy diagnostics (P2 select ticks) -----
    f_elig_diag_ticks = 0
    f_elig_excluded_gt0_ticks = 0      # ticks the envelope excluded >= 1 candidate (real competition)
    f_elig_winner_neq_f_argmin_ticks = 0  # ticks F was demoted at commit (reorder proxy)

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
            # MECH-140 x MECH-450: read the soft-competitive settling + MECH-448 f-eligibility
            # diagnostics from the last e3 select (P2 only). The settling runs the single-arena
            # within-eligible path (loop segregation OFF).
            if is_p2:
                diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
                if diag.get("soft_competitive_settling_active", False):
                    scs_active_ticks += 1
                    _rd = float(diag.get("soft_competitive_settling_round_delta", 0.0) or 0.0)
                    if _rd > 0.0:
                        scs_round_delta_sum += _rd
                        scs_round_delta_peak = max(scs_round_delta_peak, _rd)
                        if _rd > ROUND_DELTA_FLOOR:
                            scs_round_delta_moved_ticks += 1
                # MECH-448 f-eligibility (E) non-degeneracy: the envelope excluded >= 1 candidate
                # (real within-eligible competition to settle) + F demoted at commit (reorder proxy).
                if diag.get("f_eligibility_demotion_active", False):
                    f_elig_diag_ticks += 1
                    _exc = int(diag.get("f_eligibility_excluded_count", -1) or -1)
                    if _exc > 0:
                        f_elig_excluded_gt0_ticks += 1
                    if bool(diag.get("f_eligibility_winner_neq_f_argmin", False)):
                        f_elig_winner_neq_f_argmin_ticks += 1
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
            if is_p2:
                ep_reward += float(_harm_signal)
            # update_residue drives e3.post_action_update -> the ARC-108 (finer w_chan_finer)
            # three-factor update fires here on EVERY waking tick (all phases). The soft-competitive
            # settling is PARAMETER-FREE (no update path -- it is a fixed graded competition).
            with torch.no_grad():
                resid_metrics = agent.update_residue(
                    harm_signal=float(_harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )
            fdt = resid_metrics.get("e3_fcg_delta_t")
            if fdt is not None:
                fcg_delta_ts.append(float(fdt.item()))
            fwr = resid_metrics.get("e3_fcg_w_chan_finer_range")
            if fwr is not None:
                fcg_w_chan_finer_range_max = max(
                    fcg_w_chan_finer_range_max, float(fwr.item())
                )
            fws = resid_metrics.get("e3_fcg_w_chan_finer_std")
            if fws is not None:
                fcg_w_chan_finer_std_max = max(
                    fcg_w_chan_finer_std_max, float(fws.item())
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

        if is_p2:
            ep_reward_p2_sum += ep_reward
            n_p2_episodes_done += 1

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

    frac_pre_ge2 = float(n_p2_pre_ge2 / n_p2_ticks) if n_p2_ticks > 0 else 0.0
    consumed_spread_mean = (
        float(sum(consumed_dists) / len(consumed_dists)) if consumed_dists else 0.0
    )
    mean_reward_p2 = (
        float(ep_reward_p2_sum / n_p2_episodes_done) if n_p2_episodes_done > 0 else 0.0
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

    fcg_delta_t_std = float(statistics.pstdev(fcg_delta_ts)) if len(fcg_delta_ts) >= 2 else 0.0
    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )
    fcg_moved = bool(fcg_w_chan_finer_range_max > W_CHAN_FINER_RANGE_FLOOR)
    fcg_delta_nonflat = bool(fcg_delta_t_std > DELTA_T_STD_FLOOR)

    # ----- MECH-140 x MECH-450 soft-competitive settling per-seed non-vacuity (settling arms) -----
    scs_mean_round_delta = (
        float(scs_round_delta_sum / scs_active_ticks) if scs_active_ticks > 0 else 0.0
    )
    # (M) the settling ACTUALLY MOVED the field: mean round_delta over active ticks > floor.
    seed_scs_field_moved = bool(
        scs_on and scs_active_ticks > 0 and scs_mean_round_delta > ROUND_DELTA_FLOOR
    )
    # (E) the eligible set is NON-DEGENERATE: the envelope excluded >= 1 candidate on a non-trivial
    # fraction of P2 f-eligibility ticks (there was a real within-eligible competition to settle).
    f_elig_frac_excluded_gt0 = (
        float(f_elig_excluded_gt0_ticks / f_elig_diag_ticks) if f_elig_diag_ticks > 0 else 0.0
    )
    f_elig_frac_winner_neq_argmin = (
        float(f_elig_winner_neq_f_argmin_ticks / f_elig_diag_ticks) if f_elig_diag_ticks > 0 else 0.0
    )
    seed_elig_non_degenerate = bool(
        f_elig_diag_ticks > 0 and f_elig_frac_excluded_gt0 > ELIG_EXCLUDED_FRAC_FLOOR
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "scs_on": scs_on,
        "scs_cross_class": (float(arm["scs_cross_class"]) if scs_on else None),
        # ----- MECH-140 x MECH-450 soft-competitive settling diagnostics -----
        "scs_active_ticks": int(scs_active_ticks),
        "scs_mean_round_delta": round(scs_mean_round_delta, 8),
        "scs_round_delta_peak": round(scs_round_delta_peak, 8),
        "scs_round_delta_moved_ticks": int(scs_round_delta_moved_ticks),
        "scs_field_moved": seed_scs_field_moved,
        # ----- MECH-448 f-eligibility (E) non-degeneracy diagnostics -----
        "f_elig_diag_ticks": int(f_elig_diag_ticks),
        "f_elig_frac_excluded_gt0": round(f_elig_frac_excluded_gt0, 6),
        "f_elig_frac_winner_neq_argmin": round(f_elig_frac_winner_neq_argmin, 6),
        "elig_non_degenerate": seed_elig_non_degenerate,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
        # ----- PRIMARY DV (committed-class entropy) + task performance -----
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        "mean_reward_p2": round(mean_reward_p2, 6),
        # ----- Readiness / non-vacuity -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "crf_differentiated": crf_differentiated,
        # ----- finer-channel learning diagnostics (MECH-451; matched constant) -----
        "fcg_n_updates": int(len(fcg_delta_ts)),
        "fcg_delta_t_std": round(fcg_delta_t_std, 8),
        "fcg_w_chan_finer_range_max": round(fcg_w_chan_finer_range_max, 8),
        "fcg_w_chan_finer_std_max": round(fcg_w_chan_finer_std_max, 8),
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
            f"Arm {arm['arm_id']} ({arm['label']}) scs_on={arm['scs_on']} "
            f"cross_class={arm['scs_cross_class'] if arm['scs_on'] else 'n/a'} "
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
            # Per-cell fingerprint. Minted reuse-ELIGIBLE (rng_fully_reset + config_slice_declared +
            # include_driver_script_in_hash=False) -- mint-as-you-go; terminality is unknowable, so a
            # later iteration reconstructing the SAME arm from substrate + config_slice + seed can HIT.
            row["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice=_arm_config_slice(
                    arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                ),
                seed=s,
                script_path=script_path,
                rng_fully_reset=True,
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    off_rows = _arm_rows(arm_results, "A0_OFF")
    intact_rows = _arm_rows(arm_results, "A1_INTACT")
    ablated_rows = _arm_rows(arm_results, "A2_ABLATED")
    all_rows = off_rows + intact_rows + ablated_rows

    def _maj(rows: List[Dict[str, Any]], pred) -> bool:
        return sum(1 for r in rows if pred(r)) >= MIN_SEEDS_FOR_PASS

    off_ent = _by_seed(off_rows, "committed_class_entropy_nats")
    intact_ent = _by_seed(intact_rows, "committed_class_entropy_nats")
    ablated_ent = _by_seed(ablated_rows, "committed_class_entropy_nats")
    off_gap = _gap_by_seed(off_rows)
    intact_gap = _gap_by_seed(intact_rows)
    ablated_gap = _gap_by_seed(ablated_rows)

    # ----- Per-seed-divergent gating: seeds whose pool is divergent on ALL THREE arms. -----
    primary_div = [
        s for s in sorted(set(off_gap) & set(intact_gap) & set(ablated_gap))
        if off_gap.get(s) and intact_gap.get(s) and ablated_gap.get(s)
    ]
    n_primary_div = len(primary_div)
    enough_divergent = n_primary_div >= MIN_DIVERGENT_SEEDS

    # ----- Precondition (MECH-140 x MECH-450 MECHANISM non-vacuity M): on the INTACT arm the
    # settling ACTUALLY MOVED the field (mean round_delta > floor) on a strict-majority of DIVERGENT
    # seeds. If the field never moved, the settling was a no-op and any "no lift" is meaningless. -----
    field_moved_div = [
        s for s in primary_div
        if next((r for r in intact_rows if int(r["seed"]) == s), {}).get("scs_field_moved", False)
    ]
    settling_field_moved = bool(enough_divergent and _div_pass(len(field_moved_div), n_primary_div))
    intact_mean_round_delta_max = float(
        max([r.get("scs_mean_round_delta", 0.0) for r in intact_rows] or [0.0])
    )

    # ----- Precondition (E) eligible set NON-DEGENERATE: on the INTACT arm the F-eligibility
    # envelope excluded >= 1 candidate on a non-trivial fraction of ticks (real within-eligible
    # competition to settle), on a strict-majority of DIVERGENT seeds. -----
    elig_nondeg_div = [
        s for s in primary_div
        if next((r for r in intact_rows if int(r["seed"]) == s), {}).get("elig_non_degenerate", False)
    ]
    elig_non_degenerate = bool(enough_divergent and _div_pass(len(elig_nondeg_div), n_primary_div))
    intact_frac_excluded_max = float(
        max([r.get("f_elig_frac_excluded_gt0", 0.0) for r in intact_rows] or [0.0])
    )

    # ----- Precondition: learning engaged on ALL arms (finer channels + delta_t) -----
    fcg_moved_ok = bool(
        _maj(off_rows, lambda r: r.get("fcg_moved", False))
        and _maj(intact_rows, lambda r: r.get("fcg_moved", False))
        and _maj(ablated_rows, lambda r: r.get("fcg_moved", False))
    )
    fcg_delta_nonflat_ok = bool(
        _maj(off_rows, lambda r: r.get("fcg_delta_nonflat", False))
        and _maj(intact_rows, lambda r: r.get("fcg_delta_nonflat", False))
        and _maj(ablated_rows, lambda r: r.get("fcg_delta_nonflat", False))
    )
    # Per-leg COUNTS behind the _maj calls above, reduced across arms by MIN. Each leg is an
    # AND of three per-arm counts against the SAME MIN_SEEDS_FOR_PASS threshold, so the MIN of
    # the three counts reproduces the conjunction exactly from a single (measured, threshold)
    # pair. The two legs (moved / delta-nonflat) are DIFFERENT statistics and are declared as
    # two separate recomputable preconditions -- the single entry that used to carry both had
    # `met = fcg_moved_ok and fcg_delta_nonflat_ok`, which no one pair can reproduce.
    n_fcg_moved_min_arm = min(
        sum(1 for r in rows if r.get("fcg_moved", False))
        for rows in (off_rows, intact_rows, ablated_rows)
    )
    n_fcg_delta_nonflat_min_arm = min(
        sum(1 for r in rows if r.get("fcg_delta_nonflat", False))
        for rows in (off_rows, intact_rows, ablated_rows)
    )

    # CRF maturity (matched constant; majority of seeds on all arms).
    crf_matured = bool(
        _maj(off_rows, lambda r: r["crf_differentiated"])
        and _maj(intact_rows, lambda r: r["crf_differentiated"])
        and _maj(ablated_rows, lambda r: r["crf_differentiated"])
    )
    # The per-arm COUNTS behind the _maj calls above, reported by the crf_matured
    # precondition entry so the indexer's authoritative recompute can reproduce `met`.
    # `crf_matured` is an all()/AND over per-arm k-of-n seed counts and _maj is
    # `count >= MIN_SEEDS_FOR_PASS`, so min(per-arm count) >= MIN_SEEDS_FOR_PASS is
    # EXACT: min(counts) >= k iff every count >= k. NOT split into one entry per arm --
    # a k-of-n COUNT does not distribute over the conjunction the way all() does (two
    # arms each cleared by k DIFFERENT seeds is not the conjunction), so a per-leg split
    # would be strictly LOOSER than the shipped gate.
    n_crf_differentiated_per_arm = [
        sum(1 for r in rows if r["crf_differentiated"])
        for rows in (off_rows, intact_rows, ablated_rows)
    ]
    n_crf_differentiated_min = int(min(n_crf_differentiated_per_arm))

    preconditions_met = bool(
        enough_divergent
        and settling_field_moved     # (M) the settling moved the field
        and elig_non_degenerate      # (E) real within-eligible competition
        and fcg_moved_ok and fcg_delta_nonflat_ok
        and crf_matured
    )

    # ----- C1 (VALIDATION): A1_INTACT committed-class entropy strict-above A0_OFF + margin, on a
    # strict-majority of divergent seeds. -----
    c1_seeds: List[int] = []
    for s in primary_div:
        if intact_ent.get(s, 0.0) > off_ent.get(s, 0.0) + CONVERSION_MARGIN:
            c1_seeds.append(s)
    n_c1 = len(c1_seeds)
    c1_holds = _div_pass(n_c1, n_primary_div)

    # ----- C2 (PLOS-BIOLOGY ABLATION dissociation, SECONDARY): A1_INTACT committed-class entropy
    # strict-above A2_ABLATED + margin, on a strict-majority of divergent seeds. -----
    c2_seeds: List[int] = []
    for s in primary_div:
        if intact_ent.get(s, 0.0) > ablated_ent.get(s, 0.0) + ABLATION_MARGIN:
            c2_seeds.append(s)
    n_c2 = len(c2_seeds)
    c2_holds = _div_pass(n_c2, n_primary_div)

    off_mean_dv = _mean([r["committed_class_entropy_nats"] for r in off_rows])
    intact_mean_dv = _mean([r["committed_class_entropy_nats"] for r in intact_rows])
    ablated_mean_dv = _mean([r["committed_class_entropy_nats"] for r in ablated_rows])

    # Diagnostic (not a hard gate): the uniform kernel (A2) is rank-preserving, so A2 should NOT
    # substantially EXCEED A1. If it does, flag it (the uniform kernel unexpectedly reordered).
    a2_unexpectedly_reordered = bool(
        ablated_mean_dv > intact_mean_dv + A2_REORDER_FLAG_SLACK
    )

    # ----- Outcome map (decisive either way) -----
    if not preconditions_met:
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "MECH-140 x MECH-450 disinhibitory soft-competitive settling could NOT be validly "
            "measured: a precondition is unmet (too few divergent seeds / the settling did NOT move "
            "the field on the INTACT arm = a no-op so a no-lift is meaningless / the F-eligibility "
            "envelope was DEGENERATE (all-admit, no within-eligible competition to settle) / finer "
            "channels not dissociable / delta_t flat / CRF not matured). NOT a falsification."
        )
        per_claim = {"MECH-140": "non_contributory", "MECH-450": "non_contributory", "MECH-439": "non_contributory"}
    elif c1_holds:
        outcome = "PASS"
        overall_direction = "mixed"
        non_degenerate = True
        degeneracy_reason = ""
        if c2_holds:
            label = "soft_competitive_settling_lifts_conversion_ceiling_structured_edge_load_bearing_plos_ablation_signature"
        else:
            label = "soft_competitive_settling_lifts_conversion_ceiling_supports_mech140_mech450"
        # C1 PASS: the disinhibitory soft-competitive settling converts committed-action diversity
        # where the one-shot argmin (A0) plateaued -> the F-dominance ceiling is LIFTABLE (weakens
        # MECH-439), via soft-competitive disinhibition (MECH-140) + the bounded recurrent settling
        # step (MECH-450).
        per_claim = {"MECH-140": "supports", "MECH-450": "supports", "MECH-439": "weakens"}
    else:
        outcome = "FAIL"
        overall_direction = "mixed"
        non_degenerate = True
        degeneracy_reason = ""
        label = "soft_competitive_settling_does_not_convert_ceiling_intrinsic_weakens_mech140_mech450"
        # C1 FAIL (decisive): the settling MOVED the field + the eligible set was non-degenerate,
        # BUT A1_INTACT does NOT lift committed-class entropy strict-above A0_OFF. Even the
        # soft-competitive disinhibition cannot convert -> the F-dominance conversion ceiling is
        # INTRINSIC (supports MECH-439); the MECH-140/MECH-450 settling route does not deliver.
        per_claim = {"MECH-140": "weakens", "MECH-450": "weakens", "MECH-439": "supports"}

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "crf_matured",
                "kind": "readiness",
                "description": (
                    "CRF maturity: the consumed matched constant must be DIFFERENTIATED "
                    "(per-seed crf_differentiated) on at least MIN_SEEDS_FOR_PASS seeds within "
                    "EVERY arm -- an undifferentiated constant makes the manipulation vacuous "
                    "=> substrate_not_ready_requeue. measured = the SMALLEST per-arm count of "
                    "seeds carrying crf_differentiated, over arms (off_rows, intact_rows, ablated_rows)."
                ),
                "control": "per-seed crf_differentiated, counted within each arm",
                # COUNT-shaped, INCLUSIVE floor, and EXACT for the shipped predicate:
                # `met` is all(_maj(rows, crf_differentiated) for rows in arms) with _maj ==
                # `count >= MIN_SEEDS_FOR_PASS`, and min(counts) >= k iff every count >= k.
                "measured": float(n_crf_differentiated_min),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                # Non-bound observable (inert to the recompute): the full per-arm counts,
                # so a reader can see WHICH arm failed, not merely that one did.
                "observed_crf_differentiated_counts_per_arm": [int(c) for c in n_crf_differentiated_per_arm],
                "met": bool(crf_matured),
            },
            {
                "name": "enough_divergent_seeds",
                "kind": "readiness",
                "description": (
                    "number of seeds whose candidate pool is DIVERGENT on ALL THREE arms >= "
                    "MIN_DIVERGENT_SEEDS. Per-seed-divergent gating; too few => "
                    "substrate_not_ready_requeue."
                ),
                "control": "consumed cand_world_summary pairwise spread > floor (GAP-A); per-seed",
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                # COUNT-shaped, INCLUSIVE floor: `met` is exactly
                # `n_primary_div >= MIN_DIVERGENT_SEEDS`. Declared rather than left to the
                # indexer's default so the boundary case is explicit.
                "comparator": ">=",
                "direction": "lower",
                "met": bool(enough_divergent),
            },
            {
                "name": "settling_field_moved_off_no_op",
                "kind": "readiness",
                "description": (
                    "MECH-140 x MECH-450 MECHANISM non-vacuity (M): on the INTACT arm the "
                    "soft-competitive settling ACTUALLY MOVED the eligible field -- mean "
                    "soft_competitive_settling_round_delta over active P2 ticks > ROUND_DELTA_FLOOR "
                    "-- on a strict-majority of DIVERGENT seeds. At gain 0.0 / when the field never "
                    "moves round_delta <= 0 -> the settling is a no-op and a 'no lift' is "
                    "meaningless => substrate_not_ready_requeue (NEVER a weakens). measured = the "
                    "NUMBER of DIVERGENT seeds on which the INTACT arm's field moved; the "
                    "per-seed test uses the SAME statistic (round_delta = L2 field movement) the "
                    "mechanism depends on."
                ),
                "control": "INTACT scs_mean_round_delta (mean soft_competitive_settling_round_delta over active P2 ticks)",
                # COUNT-shaped, INCLUSIVE floor: `met` is
                # `enough_divergent and _div_pass(len(field_moved_div), n_primary_div)`, and
                # _div_pass is `n_ok >= max(MIN_SEEDS_FOR_PASS, ceil(FRACTION * n_div))` -- the
                # threshold reported here -- guarded by `n_div >= MIN_DIVERGENT_SEEDS`, the same
                # leg as `enough_divergent`, declared separately as `enough_divergent_seeds`.
                # This entry previously reported MAX across INTACT seeds of scs_mean_round_delta
                # against ROUND_DELTA_FLOOR, strictly LOOSER than the shipped strict-majority
                # count (one seed clearing the floor satisfies a max), and the per-seed boolean
                # is a THREE-way conjunction (scs_on AND active_ticks > 0 AND round_delta >
                # floor), which no single delta statistic can reproduce. The max-delta number is
                # kept as a NON-BOUND diagnostic (extra keys are ignored by the recompute).
                "measured": float(len(field_moved_div)),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "comparator": ">=",
                "direction": "lower",
                "observed_intact_mean_round_delta_max": float(round(intact_mean_round_delta_max, 8)),
                "observed_round_delta_floor": float(ROUND_DELTA_FLOOR),
                "met": bool(settling_field_moved),
            },
            {
                "name": "eligible_set_non_degenerate",
                "kind": "readiness",
                "description": (
                    "MECH-448 (E) non-degeneracy: on the INTACT arm the F-eligibility envelope "
                    "EXCLUDED >= 1 candidate (f_eligibility_excluded_count > 0) on a fraction of P2 "
                    "ticks > ELIG_EXCLUDED_FRAC_FLOOR, on a strict-majority of DIVERGENT seeds -- so "
                    "there was a real within-eligible competition for the settling to act on. An "
                    "all-admit envelope leaves nothing to settle. measured = the NUMBER of "
                    "DIVERGENT seeds whose INTACT eligible set is non-degenerate."
                ),
                "control": "INTACT f_elig_frac_excluded_gt0 (fraction of P2 f-eligibility ticks with excluded_count > 0)",
                # COUNT-shaped, INCLUSIVE floor: `met` is
                # `enough_divergent and _div_pass(len(elig_nondeg_div), n_primary_div)`,
                # threshold as reported; the `n_div >= MIN_DIVERGENT_SEEDS` leg is declared
                # separately as `enough_divergent_seeds`. This entry previously reported MAX
                # across INTACT seeds of f_elig_frac_excluded_gt0 against
                # ELIG_EXCLUDED_FRAC_FLOOR, strictly LOOSER than the shipped strict-majority
                # count, and the per-seed boolean is a CONJUNCTION (diag_ticks > 0 AND frac >
                # floor), so no fraction statistic can reproduce `met`. The max-fraction number
                # is kept as a NON-BOUND diagnostic.
                "measured": float(len(elig_nondeg_div)),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "comparator": ">=",
                "direction": "lower",
                "observed_intact_frac_excluded_max": float(round(intact_frac_excluded_max, 6)),
                "observed_elig_excluded_frac_floor": float(ELIG_EXCLUDED_FRAC_FLOOR),
                "met": bool(elig_non_degenerate),
            },
            {
                "name": "learning_engaged_finer_channels_dissociable",
                "kind": "readiness",
                "description": (
                    "on ALL arms the finer w_chan_finer entries MOVED + carry cross-channel range "
                    "above floor, on a majority of seeds -- learning is engaged. measured = the "
                    "WORST arm's count of fcg_moved seeds. Below floor => "
                    "substrate_not_ready_requeue."
                ),
                "control": "fcg_w_chan_finer_range_max (all arms)",
                # COUNT-shaped, INCLUSIVE floor: `met` is `_maj(fcg_moved)` on EACH of the three
                # arms -- three counts against the SAME MIN_SEEDS_FOR_PASS threshold -- so the
                # MIN of the three counts reproduces the conjunction exactly. SPLIT from the
                # delta_t leg below, a DIFFERENT statistic: the single entry that used to carry
                # both had `met = fcg_moved_ok and fcg_delta_nonflat_ok`, which no one pair can
                # reproduce, and it reported min over all rows of fcg_w_chan_finer_range_max --
                # a per-seed magnitude strictly harsher than a majority count, and silent on the
                # delta_t leg. The min-magnitude number is kept as a NON-BOUND diagnostic.
                "measured": float(n_fcg_moved_min_arm),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_w_chan_finer_range_max": float(
                    min([r["fcg_w_chan_finer_range_max"] for r in all_rows] or [0.0])
                ),
                "observed_w_chan_finer_range_floor": float(W_CHAN_FINER_RANGE_FLOOR),
                "met": bool(fcg_moved_ok),
            },
            {
                "name": "learning_engaged_delta_nonflat",
                "kind": "readiness",
                "description": (
                    "on ALL arms the signed-RPE delta_t carries cross-tick variance above floor "
                    "on a majority of seeds -- the second leg of the learning-engaged guard. "
                    "measured = the WORST arm's count of fcg_delta_nonflat seeds."
                ),
                "control": "fcg_delta_t_std (all arms)",
                # COUNT-shaped, INCLUSIVE floor: `met` is `_maj(fcg_delta_nonflat)` on each of
                # the three arms, again three counts against the same threshold, so the MIN
                # reproduces it exactly. See the entry above for why the legs are separate;
                # their conjunction is the shipped predicate and routing still reads the
                # booleans.
                "measured": float(n_fcg_delta_nonflat_min_arm),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_delta_t_std": float(
                    min([r["fcg_delta_t_std"] for r in all_rows] or [0.0])
                ),
                "observed_delta_t_std_floor": float(DELTA_T_STD_FLOOR),
                "met": bool(fcg_delta_nonflat_ok),
            },
            {
                "name": "candidate_pool_divergent",
                "kind": "readiness",
                "description": (
                    "consumed cand_world_summaries (e2.world_forward) per-candidate SPREAD clears "
                    "the GAP-A non-vacuity floor on enough seeds: measured = the NUMBER of seeds "
                    "DIVERGENT on all three arms, threshold = MIN_DIVERGENT_SEEDS."
                ),
                "control": "SD-056 e2 trained online in P0; candidate_summary_source=e2_world_forward",
                # COUNT-shaped, INCLUSIVE floor: `met` is `enough_divergent`, i.e.
                # `n_primary_div >= MIN_DIVERGENT_SEEDS` -- a COUNT of seeds divergent on ALL
                # THREE arms. This entry previously reported min over all_rows of
                # consumed_summary_pairwise_dist_mean against CONSUMED_SPREAD_FLOOR, strictly
                # HARSHER than the shipped count predicate (a majority of seeds can clear the
                # floor while the min does not), so the indexer's authoritative recompute
                # wrongly flagged sound diagnostics precondition_unmet. No spread statistic CAN
                # reproduce `met`: the per-seed divergence boolean is itself a CONJUNCTION
                # (spread > CONSUMED_SPREAD_FLOOR and dist_max < CONSUMED_MAGNITUDE_CEIL)
                # evaluated on EACH arm, and a count over a conjunction does not distribute into
                # per-leg counts. The min-spread number is preserved as a NON-BOUND diagnostic.
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_consumed_spread": float(
                    min([r["consumed_summary_pairwise_dist_mean"] for r in all_rows] or [0.0])
                ),
                "observed_consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
                "met": bool(enough_divergent),
            },
        ],
        "criteria": [
            {
                "name": "C1_intact_strict_above_off",
                "load_bearing": True,
                "passed": bool(c1_holds),
            },
            {
                "name": "C2_intact_strict_above_ablated_structured_edge_load_bearing",
                "load_bearing": False,
                "passed": bool(c2_holds),
            },
        ],
        "criteria_non_degenerate": {
            "preconditions_met": bool(preconditions_met),
            "enough_divergent_seeds": bool(enough_divergent),
            "settling_field_moved": bool(settling_field_moved),
            "eligible_set_non_degenerate": bool(elig_non_degenerate),
            "learning_engaged": bool(fcg_moved_ok and fcg_delta_nonflat_ok),
            "crf_matured": bool(crf_matured),
        },
        "a2_ablated_rank_preserving_diagnostic": {
            "a2_unexpectedly_reordered": bool(a2_unexpectedly_reordered),
            "mean_committed_class_entropy_intact": round(intact_mean_dv, 6),
            "mean_committed_class_entropy_ablated": round(ablated_mean_dv, 6),
            "note": (
                "A2 uniform kernel is rank-preserving -> A2 entropy should be ~<= A1 entropy. "
                "a2_unexpectedly_reordered True means the uniform kernel reordered beyond the "
                "A2_REORDER_FLAG_SLACK slack (a substrate anomaly to inspect, NOT a claim direction)."
            ),
        },
        "reorder_rate_limitation_note": (
            "the selector does NOT expose the pre-settle within-eligible argmin as a running "
            "diagnostic, so a literal per-tick 'committed winner != one-shot argmin' reorder rate "
            "cannot be recorded in-run. Recorded instead: soft_competitive_settling_round_delta (the "
            "field-movement non-vacuity signal) and the f_eligibility diagnostics "
            "(f_eligibility_excluded_count > 0 = real within-eligible competition; "
            "f_eligibility_winner_neq_f_argmin = F demoted at commit). The behavioural reorder is "
            "read at the DV level (A1 vs A0) and the structural-edge dissociation (A1 vs A2)."
        ),
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
            "ablation_margin": float(ABLATION_MARGIN),
            "a2_reorder_flag_slack": float(A2_REORDER_FLAG_SLACK),
            "min_divergent_seeds": int(MIN_DIVERGENT_SEEDS),
            "divergent_pass_fraction": float(DIVERGENT_PASS_FRACTION),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "round_delta_floor": float(ROUND_DELTA_FLOOR),
            "elig_excluded_frac_floor": float(ELIG_EXCLUDED_FRAC_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "delta_t_std_floor": float(DELTA_T_STD_FLOOR),
            "w_chan_finer_range_floor": float(W_CHAN_FINER_RANGE_FLOOR),
            "scs_gain": float(SCS_GAIN),
            "scs_rounds": int(SCS_ROUNDS),
            "scs_temperature": float(SCS_TEMPERATURE),
            "scs_cross_class_intact": float(SCS_CROSS_CLASS_INTACT),
            "scs_cross_class_ablated": float(SCS_CROSS_CLASS_ABLATED),
        },
        "acceptance_criteria": {
            "preconditions_met": preconditions_met,
            "n_divergent_seeds": int(n_primary_div),
            "enough_divergent_seeds": enough_divergent,
            "crf_matured": crf_matured,
            "settling_field_moved": settling_field_moved,
            "n_field_moved_over_divergent": int(len(field_moved_div)),
            "intact_mean_round_delta_max": round(intact_mean_round_delta_max, 8),
            "eligible_set_non_degenerate": elig_non_degenerate,
            "n_elig_non_degenerate_over_divergent": int(len(elig_nondeg_div)),
            "intact_frac_excluded_max": round(intact_frac_excluded_max, 6),
            "learning_engaged_fcg_moved": fcg_moved_ok,
            "learning_engaged_fcg_delta_nonflat": fcg_delta_nonflat_ok,
            "C1_intact_above_off": c1_holds,
            "C1_n_seeds": int(n_c1),
            "C1_n_divergent": int(n_primary_div),
            "C2_intact_above_ablated": c2_holds,
            "C2_n_seeds": int(n_c2),
            "a2_unexpectedly_reordered": a2_unexpectedly_reordered,
            "mean_committed_class_entropy_off": round(off_mean_dv, 6),
            "mean_committed_class_entropy_intact": round(intact_mean_dv, 6),
            "mean_committed_class_entropy_ablated": round(ablated_mean_dv, 6),
        },
        "interpretation_grid": {
            "PASS_soft_competitive_settling_lifts_conversion_ceiling_supports_mech140_mech450": (
                "preconditions met (divergent seeds + the settling MOVED the field on the INTACT "
                "arm + the F-eligibility envelope was NON-DEGENERATE + learning engaged) AND C1 "
                "(A1_INTACT committed-class entropy strict-above A0_OFF + margin on a strict-majority "
                "of divergent seeds). The disinhibitory soft-competitive settling CONVERTS "
                "committed-action diversity where the one-shot argmin plateaued -> the F-dominance "
                "conversion ceiling is LIFTABLE (weakens MECH-439), via soft-competitive "
                "disinhibition (supports MECH-140) + the bounded recurrent settling step (supports "
                "MECH-450). If C2 also passes (A1 strict-above A2_ABLATED), the STRUCTURED "
                "class-surround edge is LOAD-BEARING -- the PLOS-biology inhibition-on-inhibition "
                "ablation signature (a uniform rank-preserving kernel collapses the conversion while "
                "the base graded settling machinery still runs)."
            ),
            "FAIL_soft_competitive_settling_does_not_convert_ceiling_intrinsic_weakens_mech140_mech450": (
                "DECISIVE. preconditions met (the settling MOVED the field, the eligible set was "
                "non-degenerate) BUT A1_INTACT does NOT lift committed-class entropy strict-above "
                "A0_OFF. Even the soft-competitive disinhibition cannot convert -> the F-dominance "
                "conversion ceiling is INTRINSIC (supports MECH-439); the MECH-140 x MECH-450 "
                "settling route does not deliver (weakens both)."
            ),
            "FAIL_substrate_not_ready_requeue": (
                "A precondition is unmet: too FEW divergent seeds, OR the settling did NOT move the "
                "field on the INTACT arm (a no-op -- no-lift meaningless), OR the F-eligibility "
                "envelope was DEGENERATE (all-admit -- nothing to settle), OR learning was not "
                "engaged / CRF not matured. The conversion question could NOT be measured -- NOT a "
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
            f"V3-EXQ-710 MECH-140 x MECH-450 DISINHIBITORY SOFT-COMPETITIVE SETTLING VALIDATION "
            f"(experiment_purpose=evidence; claim_ids=[MECH-140, MECH-450, MECH-439]). The substrate "
            f"landed 2026-07-02 (ree-v3 main 8cc42bc) replaces the one-shot within-eligible committed "
            f"argmin (a hard argmin over an F-dominated modulatory field always returns the F-winner "
            f"-- the MECH-439 conversion ceiling) with a few rounds of PARAMETER-FREE soft-competitive "
            f"lateral-inhibition SETTLING over _modulatory_accum[eligible_idx] BEFORE the commit "
            f"(_soft_competitive_settle). The class-surround kernel K is 1.0 within a first-action "
            f"class + cross_class across classes; because it encodes candidate-vs-candidate STRUCTURE "
            f"the settling can REORDER (a crowded candidate loses to an isolated slightly-worse one -- "
            f"graded, never WTA). 3 arms on the SAME GAP-A reef-bipartite foraging substrate + the "
            f"SAME matched 569i top-k + MECH-448 demotion + Go/No-Go SOTA single-arena envelope; the "
            f"ONLY swept factor is the settling: A0_OFF (one-shot argmin baseline / plateau) vs "
            f"A1_INTACT (cross_class=0.25 -- STRUCTURED class-surround kernel) vs A2_ABLATED "
            f"(cross_class=1.0 -- UNIFORM kernel = the structural inhibition-on-inhibition edge "
            f"ablated -> rank-preserving). Loop segregation OFF (single arena -> the settling runs "
            f"the within-eligible path), and the LEARNED W_lat settling + LEARNED cross-loop "
            f"arbitration OFF so the ONLY active settling is the parameter-free soft-competitive "
            f"settling. PRE-REGISTERED decisive: C1 = A1_INTACT committed-class entropy strict-above "
            f"A0_OFF + margin on a strict-majority (>=2/3) of divergent seeds. C1 PASS => the settling "
            f"LIFTS the F-dominance conversion ceiling -> MECH-140 supports / MECH-450 supports / "
            f"MECH-439 weakens (ceiling liftable). C1 FAIL (decisive) => even the settling cannot "
            f"convert -> MECH-140 weakens / MECH-450 weakens / MECH-439 supports (ceiling intrinsic). "
            f"C2 (SECONDARY, PLOS-biology ablation dissociation) = A1_INTACT strict-above A2_ABLATED + "
            f"margin: tests the STRUCTURED class-surround edge is LOAD-BEARING (a uniform "
            f"rank-preserving kernel collapses the conversion while the base graded settling machinery "
            f"still runs -- the mouse-V1 inhibition-on-inhibition-silencing signature); reported, does "
            f"NOT flip per-claim directions. Non-vacuity self-route substrate_not_ready_requeue "
            f"(non_contributory, non_degenerate=False, NEVER a false weakens) when the settling did "
            f"NOT move the field on the INTACT arm (a no-op -- no-lift meaningless) OR the "
            f"F-eligibility envelope was DEGENERATE (all-admit -- nothing to settle) OR the GAP-A "
            f"liveness preconditions fail. PROMOTES NOTHING (MECH-140/MECH-450 candidate; MECH-439 "
            f"candidate/substrate_ceiling). "
            f"outcome={result['outcome']}; label={result['interpretation_label']}; "
            f"per_claim={result['evidence_direction_per_claim']}."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "3-arm MECH-140 x MECH-450 disinhibitory soft-competitive settling validation (A0_OFF vs A1_INTACT structured kernel vs A2_ABLATED uniform kernel) on a SINGLE-ARENA conversion envelope + per-seed-divergent gating + settling mechanism non-vacuity gates (field-moved + eligible-set-non-degenerate)",
            "arms": "A0_OFF (use_soft_competitive_settling=False -- one-shot argmin baseline) / A1_INTACT (ON, cross_class=0.25 -- structured class-surround kernel) / A2_ABLATED (ON, cross_class=1.0 -- uniform rank-preserving kernel)",
            "swept_variable": "use_soft_competitive_settling ONLY (+ the cross_class kernel structure: 0.25 intact / 1.0 ablated). gain=1.0/rounds=3/temperature=1.0 held constant on both settling arms.",
            "the_isolated_factor": (
                "whether the committed selection emerges from a bounded recurrent soft-competitive "
                "SETTLING over the eligible _modulatory_accum field (graded lateral inhibition via a "
                "class-surround kernel) rather than a one-shot argmin; and (A1 vs A2) whether that "
                "kernel is STRUCTURED (class-surround, cross_class<1 -> can reorder) or UNIFORM "
                "(cross_class=1 -> rank-preserving). At gain 0.0 the settling is an EXACT no-op."
            ),
            "single_arena": (
                "use_loop_segregation=False on ALL arms -> the settling runs the within-eligible "
                "SINGLE-ARENA path (transforms _modulatory_accum[eligible_idx]); "
                "use_learned_settling_step=False + use_learned_cross_loop_arbitration=False so the "
                "ONLY active settling is the parameter-free soft-competitive settling"
            ),
            "matched_constant_conversion_envelope": (
                "use_f_eligibility_demotion + use_f_eligibility_adaptive_floor (689e) + "
                "use_go_nogo_constitution (689g) + use_modulatory_selection_authority (643a) + "
                "use_modulatory_channel_routing (cand_world_summary) + top_k shortlist (k=3, 569i)"
            ),
            "matched_diversity_stack": (
                "MECH-341 stratified + use_dacc + use_gated_policy + use_lateral_pfc_analog (trained P1 "
                "REINFORCE) + SD-056 all levers + matured/maintained CRF + use_candidate_rule_field + "
                "use_finer_channel_gating"
            ),
            "primary_dv": "committed-action-class entropy (nats), interpreted on divergent seeds only",
            "secondary_dvs": "mean soft_competitive_settling_round_delta (field movement), f_eligibility_excluded fraction, f_eligibility_winner_neq_f_argmin fraction (reorder proxy), mean_reward_p2 (task performance)",
            "phases": "P0 e2-train (CRF matures) -> P1 frozen-encoder bias-head REINFORCE -> P2 e2+bias frozen; finer gating KEEPS adapting",
            "settling_wiring": "PARAMETER-FREE -- no update path; a fixed graded class-surround competition applied WAKING-ONLY (MECH-094) over the eligible field before commit; no autograd (detached copy)",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "scs_gain": SCS_GAIN,
            "scs_rounds": SCS_ROUNDS,
            "scs_temperature": SCS_TEMPERATURE,
            "scs_cross_class_intact": SCS_CROSS_CLASS_INTACT,
            "scs_cross_class_ablated": SCS_CROSS_CLASS_ABLATED,
            "reusable_arm_ids": ["A0_OFF", "A1_INTACT", "A2_ABLATED"],
            "reuse_note": "all 3 arms minted reuse-ELIGIBLE (rng_fully_reset + config_slice_declared + include_driver_script_in_hash=False) -- mint-as-you-go; a later settling-iteration reconstructing the same arm from substrate + config_slice + seed can HIT",
            "reorder_rate_limitation": "the selector does not expose the pre-settle within-eligible argmin as a running diagnostic; the behavioural reorder is read at the DV level (A1 vs A0) + the structural-edge dissociation (A1 vs A2), with round_delta + f_eligibility diagnostics as the mechanism/non-vacuity signals",
            "safety": "the settling transforms strictly the F + MECH-448/449 Go/No-Go eligible subset; a suppressed candidate is never re-admitted however the field moves; >= 1 survivor always",
            "mech140_mech450_mech439_relationship": "C1 PASS => ceiling liftable (MECH-439 weakens, MECH-140/MECH-450 supports), C1 FAIL => ceiling intrinsic (MECH-439 supports, MECH-140/MECH-450 weakens); C2 tests the structured-edge is load-bearing (PLOS ablation signature)",
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-710 MECH-140 x MECH-450 disinhibitory soft-competitive settling validation"
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
    _ac = result["acceptance_criteria"]
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"preconditions_met={_ac['preconditions_met']} "
        f"n_divergent={_ac['n_divergent_seeds']} "
        f"field_moved={_ac['settling_field_moved']} (round_delta_max={_ac['intact_mean_round_delta_max']}) "
        f"elig_non_degenerate={_ac['eligible_set_non_degenerate']} (excluded_frac_max={_ac['intact_frac_excluded_max']}) "
        f"C1_intact_above_off={_ac['C1_intact_above_off']} "
        f"C2_intact_above_ablated={_ac['C2_intact_above_ablated']} "
        f"(off={_ac['mean_committed_class_entropy_off']}, intact={_ac['mean_committed_class_entropy_intact']}, ablated={_ac['mean_committed_class_entropy_ablated']}) "
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
