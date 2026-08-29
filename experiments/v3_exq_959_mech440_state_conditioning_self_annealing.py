#!/usr/bin/env python3
"""V3-EXQ-959 -- MECH-440 legs (ii)/(iii): STATE-CONDITIONING + SELF-ANNEALING second-order
falsifier, on the SAME armed raised-class-floor stack as V3-EXQ-955.

WHY THIS EXISTS
----------------
V3-EXQ-955 (autopsied `failure_autopsy_V3-EXQ-955_2026-08-29`, CONFIRMED, user-ratified
2026-08-29) established leg (i) PROPAGATION: MECH-440's factorised-Gaussian selection-head
weight noise reaches the committed action at the raised class floor
(support_preserving_min_first_action_classes = action_dim), with all armed-conversion
preconditions met and measured for the first time in the lineage. That run's own honest gap
(section "SECONDARY", carried verbatim into its manifest as secondary_checks_measured=false):
legs (ii) STATE-CONDITIONING (per-state noise magnitude covaries with state) and (iii)
SELF-ANNEALING (per-parameter sigma falls where the policy is confident, holds where near-ties
recur) are BOTH asserted by MECH-440's design (ree_core/policy/noisy_selection_head.py) but
NEITHER was measured. This run measures both, directly, on the same substrate. It is NOT a
955a re-run: propagation is not re-tested (already confirmed by reanalysis, no re-run needed
per GOV-REUSE-1 -- see below); this is a NEW EXQ number testing a different pair of hypotheses.

WHAT "STATE-CONDITIONING" AND "SELF-ANNEALING" ACTUALLY MEAN HERE, OPERATIONALIZED
-------------------------------------------------------------------------------------
A naive test -- "does the injected bias's magnitude correlate with some state descriptor" -- is
close to TAUTOLOGICAL: bias[k] = (sigma*eps_w)@x_k + sigma*eps_b*scale is a LINEAR function of
the per-candidate action-feature vector x_k (itself a function of the state) with the sampled
eps held fixed within a tick, so its RANGE across candidates mechanically tracks the spread of
x_k regardless of whether the noise is meaningfully "state-conditioned" in any functional sense
-- eps resampling ALONE (independent of x_k) would already make the bias vary tick to tick.
Magnitude covariance with state is therefore not a real falsifier; a REAL test asks whether the
noise's FUNCTIONAL IMPACT on the committed decision is state-dependent, which is a genuinely
falsifiable, non-definitional claim (and the biologically correct one per Aston-Jones & Cohen
2005 adaptive-gain theory: exploration should matter more under uncertainty, not uniformly).

  LEG (ii) STATE-CONDITIONING, operationalized as IMPACT CONCENTRATION: a lever arm's committed
  action should diverge from the yoked no-lever reference (A0_OFF) MORE OFTEN in states where
  A0_OFF's own pre-noise candidate scores are near-tied (low raw_score_range: little to lose by
  perturbing the argmin) than in states where they are decisive (high raw_score_range: the
  argmin is robust to a bounded perturbation). This is the LC-NE functional signature -- noise
  matters where the decision is genuinely uncertain -- and it is falsifiable: a state-BLIND
  noise source (e.g. a fixed schedule unrelated to candidate structure) would show no such
  concentration.

  LEG (iii) SELF-ANNEALING, operationalized CAUSALLY via an ablation, not a correlation: compare
  ARM_ANNEAL (noisy_selection_anneal=True, the real MECH-440 config, sigma scaled by the local
  confidence EMA) against ARM_NO_ANNEAL (identical in every other respect, noisy_selection_
  anneal=False, sigma pinned at sigma_init for the whole run -- the ONE flag that differs). Per
  the design (noisy_selection_head.py docstring point 3), annealing should SELECTIVELY suppress
  noise where the policy is confident (decisive states) while leaving it near-full where
  near-ties recur -- i.e. a NON-UNIFORM decay, not a global noise reduction. So: ARM_ANNEAL's
  divergence rate in DECISIVE states (A0_OFF-referenced raw_score_range at/above that seed's own
  median) should be MEASURABLY LOWER than ARM_NO_ANNEAL's in the same stratum, while the two
  arms' divergence rates in NEAR-TIE states (below median) should stay comparable. A uniform
  (non-selective) sigma reduction would instead lower BOTH strata's divergence roughly equally,
  or a broken/no-op anneal would show no differential at all -- both are distinguishable from the
  predicted selective-suppression signature.

Both criteria use the SAME per-tick state descriptor: A0_OFF's own raw_score_range at that tick
(agent.e3.last_raw_scores range, captured BEFORE any modulatory rescale or noise injection --
e3_selector.py:2858 sets last_raw_scores ahead of both). Using the UNTREATED reference agent's
own scoring as the stratification key (rather than each lever arm's own, differently-trained-by-
P0/P1, scoring) gives a state descriptor that is not itself contaminated by which lever is under
test, and is IDENTICAL across the ARM_ANNEAL/ARM_NO_ANNEAL comparison at any given tick (the
yoked pass steps every arm on the SAME observation sequence, driven by A0_OFF).

THE THREE ARMS (matched SOTA conversion stack as a MATCHED CONSTANT, verbatim from 955/708b
except as noted):
  A0_OFF        : no exploration injection. Yoked reference; drives the shared env walk; also
                  the source of the per-tick raw_score_range state descriptor for BOTH criteria.
  ARM_ANNEAL    : MECH-440 noisy selection head, noisy_selection_anneal=True (955's
                  ARM_NOISE_SINGLE, unchanged config) -- the real, as-shipped mechanism.
  ARM_NO_ANNEAL : identical to ARM_ANNEAL except noisy_selection_anneal=False -- sigma frozen at
                  NOISY_SELECTION_SIGMA_INIT for the entire run. The leg-(iii) ablation.
955's ARM_TEMP (the 687 non-propagating temperature control) is DROPPED -- propagation is
already confirmed; re-including it here would spend a third of the compute re-deriving a
question this run does not ask. 6 seeds x 3 arms (half 955's arm count at the same seed count).

SEED SWAP (documented departure from 955's [42,43,44,45,46,47]): seed 44 replaced with 48.
955's autopsy diagnoses seed 44 as a coupled anomaly SPECIFIC to that run's ceiling-headroom /
TEMP-divergence interaction (near-uniform committed behaviour compresses lift comparisons); this
run drops the TEMP arm entirely, so the TEMP-specific half of that coupling does not apply here,
but the underlying near-saturation tendency of seed 44 on this env/floor combination is still a
known source of noisy, harder-to-interpret per-tick stratification data for a first measurement
of a brand-new instrument (the raw_score_range-median stratification below). Swapping to a fresh
seed avoids re-litigating a documented anomaly inside a run that cannot itself re-adjudicate it.

CEILING-HEADROOM GATE FIX (955's own driver-fix note, applied here): 955's defect was
max-aggregating A0_OFF's fraction-of-ceiling across seeds for an UPPER-bound (saturation)
readiness check, so one anomalous seed (44, 0.973) alone reddened the whole arm despite a
0.833 arm MEAN and 5/6 seeds comfortably below the 0.90 threshold. This driver aggregates that
precondition's measured value as the MEAN across seeds (per the autopsy's explicit note:
"headroom gate per-seed (or mean), falsifying-branch only"), not the max. The precondition still
applies ONLY to A0_OFF (the falsifying/reference branch), per the original pre-registered scope.

ARMED-CONVERSION PRECONDITIONS (mandatory, MEASURED not assumed -- unchanged from 955, ARC-065's
what_would_answer; P1/P2 min-aggregation across seeds is UNCHANGED from 955 -- the identified
955 defect was specific to the max-aggregated ceiling-headroom check, not to P1/P2's
already-a-floor min-aggregation, which was not flagged and needs no fix):
  P1: mean_distinct_first_action_classes >= 0.9 * action_dim in EVERY arm.
  P2: authority_rel_deviation_mean > 0.05 in every LEVER-BEARING arm (ARM_ANNEAL, ARM_NO_ANNEAL).

VACUITY RULE (unchanged from 955): a null under an unmet precondition self-routes
substrate_not_ready_requeue, never weakens.

PASS (MECH-440 legs ii+iii support): with all preconditions armed, BOTH C_STATE and C_ANNEAL
(below) hold on >=2/3 of seeds with sufficient per-tick pairs. A PASS on only one of the two is
`mixed`, naming which leg confirmed. Neither holding is `weakens` for the ii/iii legs
specifically (propagation, leg i, is untouched by this run either way -- see SCOPE).

PRE-REGISTERED CRITERIA
  C_STATE (leg ii, load-bearing): for ARM_ANNEAL, the per-tick A0_OFF-referenced raw_score_range
  on ticks where ARM_ANNEAL's committed action DIVERGED from A0_OFF is, on average, at least
  STATE_COND_REL_MARGIN (10%) LOWER than on ticks where it did NOT diverge -- divergence
  concentrates in near-tie (low-differentiation) states, the LC-NE-style functional signature.
  ARM_NO_ANNEAL's same reading is recorded as corroborating (non-gating) context: the mechanism
  predicts state-conditioning is a property of the injection site (present regardless of anneal
  setting), so a matching pattern there strengthens the read without being required for PASS.

  C_ANNEAL (leg iii, load-bearing): stratify ticks per-seed into DECISIVE (A0_OFF
  raw_score_range >= that seed's own median) and NEAR-TIE (< median), using A0_OFF's per-tick
  value as the SAME classification for both comparison arms at each tick. Within DECISIVE ticks,
  ARM_ANNEAL's divergence rate is at least ANNEAL_SUPPRESSION_MARGIN (0.05) lower than
  ARM_NO_ANNEAL's; within NEAR-TIE ticks, the two arms' divergence rates differ by no more than
  NEAR_TIE_PARITY_TOLERANCE (0.10) -- selective suppression in confident states, not a uniform
  reduction (which a working-but-blunt anneal or a global sigma cut could also produce) and not
  no suppression at all (a broken/no-op anneal).

Both criteria require >= MIN_STATE_PAIRS (30) per-tick (raw_score_range, diverged) pairs for a
seed to count toward that criterion's per-seed majority vote (a seed with too few genuine fresh
E3 selects contributes no signal either way, per the same sample-size-integrity discipline as
955/708a/949 -- see below).

SECONDARY (non-gating, descriptive): ARM_ANNEAL's per-tick noisy_selection_sigma_scale is traced
alongside an EXTERNALLY-RECONSTRUCTED gap_norm (same formula the E3 selector itself uses to feed
observe_gap: sort agent.e3.last_raw_scores ascending, gap_norm = clamp((s[1]-s[0])/range, 0, 1)
-- reconstructed rather than read from an internal-only local variable, so labelled
"reconstructed_external", not claimed identical to the selector's own in-tick value). Reported
as a Spearman-style rank comparison (mean sigma_scale in the lowest vs highest gap_norm decile)
in custom_information, purely as sanity context that the anneal EMA is live and moving in the
documented direction -- NOT a pass criterion (the load-bearing test of self-annealing is the
causal C_ANNEAL ablation above, which does not depend on this reconstruction being exact).

SAMPLE-SIZE INTEGRITY (708a/949/955's repaired instrument, carried forward verbatim). E3-cadence
latches (last_score_diagnostics, last_precommit_probs, last_scores, last_raw_scores,
_last_explore_term) are cleared to None IMMEDIATELY before every select_action() call; a tick
counts toward any measurement ONLY on a genuine fresh (non-None) repopulation afterward.
n_fresh_select / n_latched recorded per cell.

RE-DERIVE BRAKE (2026-08-29 check, this driver): 0 substrate_ceiling-category autopsies for
MECH-440 in the corpus (955 itself resolved MEASURES FAILED, not substrate_ceiling -- an
adjudication-branch defect, ree:false, not chargeable to REE). NOT braked.

SUBSTRATE-PATH OVERLAP GATE (skill step 2.5c, 2026-08-29 check): same two OPEN substrate_queue
entries as 955 overlap this driver's imports (mode-governance-engagement: corrupting but NOT in
this driver's causal path, no mode-governance knob enabled anywhere in _make_agent, identical
carve-out to 955/949/947; contextmemory-write-path-addressing-degeneracy: corrupting, potentially
in path via E1, but applies IDENTICALLY to every one of the 3 arms and interacts with neither the
anneal factor nor the injection-site factor, biasing the load-bearing CONTRAST toward the null
rather than toward a false positive -- same reasoning as 955's identical carve-out). Two OPEN
degrading entries also overlap (mech357-freeze-incompatible-pressure-mechanism,
SD-MECH303-THRESHOLD-SOURCING) -- noted, non-blocking.

GOV-REUSE-1: the decisive readouts (per-tick A0_OFF-referenced raw_score_range paired with
lever-arm divergence, stratified by seed-median; ARM_ANNEAL-vs-ARM_NO_ANNEAL differential
divergence suppression by stratum) are recorded nowhere in the corpus -- 955 recorded per-cell
summary divergence fractions only, no per-tick pairing and no anneal-off ablation arm at all.
Not recoverable by reanalysis -> run. A0_OFF's cell is NOT reused from 955 despite being
config-slice-identical and reuse-ELIGIBLE there (955 stamped it with rng_fully_reset=True,
config_slice_declared=True, include_driver_script_in_hash=False): this run's yoked pass requires
A0_OFF as a LIVE, jointly-stepped agent object driving the shared observation sequence in
lockstep with the other two arms, not a cached scalar-metrics dict -- the arm-reuse mechanism
(experiments/_lib/arm_reuse.py) is built for skipping independently-scored arms, not for
reconstructing a live co-stepped reference agent, so it does not apply here (955 did not reuse
it either, for the identical reason).

claim_ids = [MECH-440]. related_claims (context only, not scored per-claim): ["ARC-065",
"MECH-441", "MECH-439", "ARC-110", "MECH-313", "MECH-458"] -- unchanged from 955, ARC-065 still
owns the ARMED-CONVERSION precondition this driver inherits and re-measures.
experiment_purpose = "evidence" (tests MECH-440's own claimed legs ii/iii directly).

SLEEP: none.

DV-SYMMETRY DECLARATION (mandatory, V3-EXQ-604c failure class), per arm:
  * A0_OFF: no exploration lever; reference for both the yoked walk and the raw_score_range
    state descriptor. Never diverges from itself (self-yoke instrument control).
  * ARM_ANNEAL: per-candidate factorised-Gaussian weight noise, resampled per tick, scaled by a
    confidence-EMA-driven anneal factor. Neither a uniform broadcast (varies per-candidate via
    x_k) nor a pure monotone rescaling of the un-noised order -- can flip the committed argmin by
    construction; yoked divergence measures this directly (identical reasoning to 955's
    ARM_NOISE_SINGLE declaration, which this arm is unchanged from).
  * ARM_NO_ANNEAL: identical injection mechanism to ARM_ANNEAL (same per-candidate, non-broadcast,
    non-monotone noise); the ONLY difference is sigma is pinned at sigma_init instead of being
    scaled by the anneal factor -- still not DV-symmetry-invariant for the committed-action DV
    for the same reason ARM_ANNEAL is not.

ETHICS PREFLIGHT: all-false / decision=allow (V3 has no live self-model, no autobiographical
memory, no social mind; SENT-0 boundary, pre-ethical instrumentation only).

See REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-955_2026-08-29.{md,json} (the routing
source for this run); REE_assembly/docs/claims/claims.yaml (MECH-440, ARC-065 what_would_answer);
ree-v3/experiments/v3_exq_955_mech440_armed_stack_raised_class_floor_falsifier.py (the base
design and matched stack this driver is forked from); ree_core/policy/noisy_selection_head.py
(the mechanism under test, both legs); ree_core/predictors/e3_selector.py:2858,3345-3427
(last_raw_scores capture point; noise-injection call site, gap_norm feed to observe_gap).
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
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
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_TYPE = "v3_exq_959_mech440_state_conditioning_self_annealing"
QUEUE_ID = "V3-EXQ-959"
CLAIM_IDS: List[str] = ["MECH-440"]
RELATED_CLAIMS: List[str] = [
    "ARC-065", "MECH-441", "MECH-439", "ARC-110", "MECH-313", "MECH-458",
]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ----- Acceptance thresholds (pre-registered) -----
ARMED_CONVERSION_CLASSES_FRAC = 0.9   # P1: mean_distinct_first_action_classes >= this * action_dim
AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR = 0.05  # P2: authority_rel_deviation_mean floor
CEILING_HEADROOM_MAX_FRAC = 0.90      # A0_OFF fraction-of-ceiling must sit below this (MEAN-aggregated)
MIN_DIVERGENT_SEEDS = 4               # >=4 seeds with usable data (2/3-of-6 majority convention)
DIVERGENT_PASS_FRACTION = (2.0 / 3.0)

# Non-vacuity: the lever arms must inject a per-candidate bias range above this floor.
NOISE_BIAS_RANGE_FLOOR = 1e-4

# dACC non-vacuity: the Go/No-Go perseveration axis must be live.
DACC_MAX_SUPPRESSION_FLOOR = 0.0

# Fresh-select sufficiency (708a/949/955's repaired-instrument readiness gate).
MIN_FRESH_SELECTS = 30

# leg ii / leg iii criteria thresholds.
STATE_COND_REL_MARGIN = 0.10          # C_STATE: diverged-mean rsr <= (1-margin)*non-diverged-mean
ANNEAL_SUPPRESSION_MARGIN = 0.05      # C_ANNEAL: decisive-stratum divergence rate gap (absolute)
NEAR_TIE_PARITY_TOLERANCE = 0.10      # C_ANNEAL: near-tie-stratum divergence rate must stay close
MIN_STATE_PAIRS = 30                  # per-seed minimum (raw_score_range, diverged) pairs to count

SEEDS = [42, 43, 45, 46, 47, 48]      # 44 swapped for 48 -- see SEED SWAP note above
P0_WARMUP_EPISODES = 100
P1_BIAS_TRAIN_EPISODES = 50
P2_YOKED_EPISODES = 100
STEPS_PER_EPISODE = 200

# Self-yoke instrument control: a SHORT independent P0/P1 training budget is sufficient (949/955
# precedent) -- the control's purpose is to prove the RUNNER's RNG isolation, not to reproduce
# full training.
CONTROL_P0_EPISODES = 1
CONTROL_P1_EPISODES = 1
CONTROL_STEPS = 15
CONTROL_EPISODES = 1

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 3
DRY_RUN_STEPS = 25

# --- MECH-440 injection-site lever constants (verbatim from 955/708b) ---
NOISY_SELECTION_SIGMA_INIT = 1.0
NOISY_SELECTION_WEIGHT = 1.0
NOISY_SELECTION_ANNEAL_FLOOR = 0.1
NOISY_SELECTION_ANNEAL_EMA_ALPHA = 0.01

# --- Matched-stack lever constants (identical on ALL arms; verbatim from 955/708b) ---
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

SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

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
        "arm_id": "A0_OFF",
        "label": "no_exploration_injection_yoked_reference",
        "noise_head": False, "anneal": False,
    },
    {
        "arm_id": "ARM_ANNEAL",
        "label": "mech440_noisy_selection_head_self_annealing_on",
        "noise_head": True, "anneal": True,
    },
    {
        "arm_id": "ARM_NO_ANNEAL",
        "label": "mech440_noisy_selection_head_self_annealing_off_frozen_sigma",
        "noise_head": True, "anneal": False,
    },
]


def _arm_config_slice(arm: Dict[str, Any], floor: int) -> Dict[str, Any]:
    """Declared reuse fingerprint slice: ONLY what an arm's computation reads."""
    return {
        "arm_id": arm["arm_id"],
        "noise_head": bool(arm["noise_head"]),
        "anneal": bool(arm["anneal"]),
        "support_preserving_min_first_action_classes": int(floor),
        "noisy_selection_sigma_init": float(NOISY_SELECTION_SIGMA_INIT),
        "noisy_selection_weight": float(NOISY_SELECTION_WEIGHT),
        "noisy_selection_anneal_floor": float(NOISY_SELECTION_ANNEAL_FLOOR),
        "noisy_selection_anneal_ema_alpha": float(NOISY_SELECTION_ANNEAL_EMA_ALPHA),
        "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
        "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
        "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
        "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
        "use_candidate_rule_field": bool(USE_CANDIDATE_RULE_FIELD),
        "use_dacc": bool(USE_DACC),
        "env_kwargs": dict(ENV_KWARGS),
        "sd056_weight": float(SD056_WEIGHT),
        "lr_lpfc_bias": float(LR_LPFC_BIAS),
        "min_state_pairs": int(MIN_STATE_PAIRS),
    }


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any], floor: int) -> REEAgent:
    """Matched-stack agent -- verbatim from 955/708b's _make_agent, EXCEPT the noisy-selection
    lever now parameterises BOTH noise_head presence and the anneal flag (955 hardcoded
    anneal=True always; this driver's leg-iii ablation needs it as a per-arm factor)."""
    noise_head = bool(arm["noise_head"])
    anneal = bool(arm["anneal"])
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
        support_preserving_min_first_action_classes=int(floor),
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
        # --- MECH-313 temperature noise floor: OFF -- this driver does not test propagation ---
        use_noise_floor=False,
        noise_floor_alpha=0.1,
        noise_floor_min_temperature=1.0,
        # --- MECH-440 NoisyNet propagating selection-head weight noise ---
        use_noisy_selection_head=noise_head,
        noisy_selection_sigma_init=(NOISY_SELECTION_SIGMA_INIT if noise_head else 0.0),
        noisy_selection_weight=NOISY_SELECTION_WEIGHT,
        noisy_selection_anneal=anneal,
        noisy_selection_anneal_floor=NOISY_SELECTION_ANNEAL_FLOOR,
        noisy_selection_anneal_ema_alpha=NOISY_SELECTION_ANNEAL_EMA_ALPHA,
        # MECH-441 OFF (this falsifier is the MECH-440 legs ii/iii only).
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
        e2_action_contrastive_multistep_enabled=True,
        e2_action_contrastive_horizon=5,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
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
        use_loop_segregation=False,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# P0 online e2 contrastive training helpers (verbatim from 955/708b/707)
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


# ---------------------------------------------------------------------------
# Per-tick measurement helpers
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


def _mean_or0(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _entropy_fraction_of_ceiling(entropy_nats: float, mean_distinct_classes: float) -> float:
    if mean_distinct_classes <= 1.0 + 1e-9:
        return 0.0
    ceiling = math.log(mean_distinct_classes)
    if ceiling <= 1e-9:
        return 0.0
    return float(entropy_nats / ceiling)


# ---------------------------------------------------------------------------
# P0/P1 phased training (verbatim mechanism from 955/708b, stops before P2)
# ---------------------------------------------------------------------------


def train_arm_p0_p1(
    arm: Dict[str, Any],
    seed: int,
    floor: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Tuple[REEAgent, Dict[str, Any]]:
    reset_all_rng(seed)
    env = _make_env(seed)
    agent = _make_agent(env, arm, floor)

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = torch.optim.Adam(
        list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS
    )
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
        maxlen=TRANSITION_BUFFER_MAX
    )
    sample_rng = random.Random(seed)

    total_eps = p0_episodes + p1_episodes
    p1_start = p0_episodes
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p0_contrastive_steps = 0
    n_p1_bias_updates = 0
    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    for ep in range(total_eps):
        is_p1 = ep >= p1_start
        phase_label = "P1" if is_p1 else "P0"

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
                cs = agent._candidate_world_summaries(candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.detach().clone()

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

            if is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            if (not is_p1) and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer, optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val):
                    n_p0_contrastive_steps += 1

            _, harm_signal, done, info, obs_dict = env.step(action)
            if is_p1:
                ep_reward += float(harm_signal)
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal), world_delta=None,
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
                n_p1_bias_updates += 1

        if (ep + 1) % 10 == 0 or (ep + 1) == total_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_eps}", flush=True,
            )
        if error_note is not None:
            break

    stats = {
        "arm_id": arm["arm_id"], "seed": int(seed),
        "n_p0_ticks": int(n_p0_ticks), "n_p1_ticks": int(n_p1_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
    }
    return agent, stats


# ---------------------------------------------------------------------------
# P2 YOKED measurement (949/955 design, extended with per-tick state-pair tracking for the
# leg-ii/leg-iii criteria). One runner wraps an ALREADY-TRAINED, FROZEN agent (no further
# training in P2) and owns its own private torch RNG stream.
# ---------------------------------------------------------------------------


class _YokedRunner:
    """Wraps an already-trained agent for the yoked P2 pass. OWNS ITS OWN RNG STREAM: snapshot
    and restore around every tick, episode reset, and residue update (949/955 mechanism)."""

    def __init__(self, agent: REEAgent) -> None:
        self.agent = agent
        self._rng_state = torch.get_rng_state()
        self.n_ticks = 0
        self.class_sum = 0
        self.committed_class_counts: Dict[int, int] = {}
        # Authority-engagement instrumentation.
        self.raw_score_ranges: List[float] = []
        self.post_score_ranges: List[float] = []
        self.n_e3_select_fires = 0
        self.n_e3_latched_ticks = 0
        # MECH-440 noise non-vacuity (955/708b instrument, recorded context).
        self.noise_bias_ranges: List[float] = []
        self.diag_raw_score_ranges: List[float] = []
        self.dacc_max_suppression = 0.0
        # THIS DRIVER'S NEW STATE: per-tick raw_score_range (this runner's OWN reading, used
        # only when this runner IS A0_OFF -- the reference stream that stratifies both
        # comparison arms). Also last-tick-fresh flag so run_yoked_measurement can gate on it.
        self.last_raw_score_range: Optional[float] = None
        self.last_tick_fresh: bool = False
        # Self-annealing descriptive trace (ARM_ANNEAL/ARM_NO_ANNEAL only): sigma_scale and a
        # reconstructed (not internal-exact) gap_norm, per fresh tick.
        self.sigma_scale_trace: List[float] = []
        self.ext_gap_norm_trace: List[float] = []

    def reset_episode(self) -> None:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            self.agent.reset()
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def choose(self, obs_dict: Dict[str, Any]) -> int:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            return self._choose_inner(obs_dict)
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def _choose_inner(self, obs_dict: Dict[str, Any]) -> int:
        agent = self.agent
        self.last_raw_score_range = None
        self.last_tick_fresh = False
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
        ticks = agent.clock.advance()
        wdim = latent.z_world.shape[-1]
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick", False)
            else torch.zeros(1, wdim, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        if candidates:
            self.n_ticks += 1
            classes = {int(c.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item())
                       for c in candidates}
            self.class_sum += len(classes)

        # Sample-size-integrity idiom (708a/949/955, mandatory): clear ALL E3-cadence latches
        # immediately before select_action(), record only on a genuine fresh fire afterward.
        agent.e3.last_score_diagnostics = None
        agent.e3.last_precommit_probs = None
        agent.e3.last_scores = None
        agent.e3.last_raw_scores = None
        agent.e3._last_explore_term = None

        action = agent.select_action(candidates, ticks)
        if action is None:
            adim = int(agent.config.e2.action_dim)
            idx = int(np.random.randint(0, adim))
            action = torch.zeros(1, adim, device=agent.device)
            action[0, idx] = 1.0

        diag = getattr(agent.e3, "last_score_diagnostics", None)
        if diag is not None:
            self.n_e3_select_fires += 1
            nbr = float(diag.get("noisy_selection_bias_range", 0.0) or 0.0)
            if nbr > 0.0:
                self.noise_bias_ranges.append(nbr)
            raw = getattr(agent.e3, "last_raw_scores", None)
            post = getattr(agent.e3, "last_scores", None)
            if raw is not None and post is not None and raw.numel() > 1:
                raw_flat = raw.detach().reshape(-1)
                raw_range = float((raw_flat.max() - raw_flat.min()).item())
                post_range = float((post.max() - post.min()).item())
                self.raw_score_ranges.append(raw_range)
                self.post_score_ranges.append(post_range)
                self.diag_raw_score_ranges.append(raw_range)
                self.last_raw_score_range = raw_range
                self.last_tick_fresh = True
                # Reconstructed (not internal-exact) gap_norm: same formula the E3 selector
                # itself applies to feed observe_gap, computed here from the SAME captured
                # last_raw_scores tensor (sort ascending; gap between best two / range).
                if raw_flat.numel() >= 2 and raw_range > 1e-9:
                    srt, _ = torch.sort(raw_flat)
                    gap = float((srt[1] - srt[0]).item())
                    gn = max(0.0, min(1.0, gap / (raw_range + 1e-8)))
                else:
                    gn = 0.0
                head = getattr(agent.e3, "noisy_selection_head", None)
                if head is not None:
                    try:
                        st = head.get_state()
                        self.sigma_scale_trace.append(float(st.get("noisy_selection_sigma_scale", 1.0)))
                        self.ext_gap_norm_trace.append(gn)
                    except Exception:
                        pass
        else:
            self.n_e3_latched_ticks += 1

        bundle = getattr(agent, "_dacc_last_bundle", None)
        if bundle is not None:
            supp = bundle.get("suppression", None)
            if supp is not None:
                try:
                    self.dacc_max_suppression = max(
                        self.dacc_max_suppression, float(supp.detach().abs().max().item())
                    )
                except Exception:
                    pass

        agent.update_z_goal(
            benefit_exposure=0.0,
            drive_level=REEAgent.compute_drive_level(body),
        )
        committed = int(action[0].argmax().item())
        self.committed_class_counts[committed] = self.committed_class_counts.get(committed, 0) + 1
        return committed

    def observe(self, harm_signal: float) -> None:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            with torch.no_grad():
                self.agent.update_residue(
                    harm_signal=float(harm_signal), world_delta=None,
                    hypothesis_tag=False, owned=True,
                )
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def authority_rel_deviation_mean(self) -> float:
        if not self.raw_score_ranges:
            return 0.0
        devs = [
            abs(p - r) / max(r, 1e-9)
            for r, p in zip(self.raw_score_ranges, self.post_score_ranges)
        ]
        return sum(devs) / len(devs)

    def summary(self) -> Dict[str, Any]:
        mean_distinct = (self.class_sum / self.n_ticks) if self.n_ticks else 0.0
        entropy = _entropy_from_int_counts(self.committed_class_counts)
        noise_bias_mean = _mean_or0(self.noise_bias_ranges)
        diag_raw_mean = _mean_or0(self.diag_raw_score_ranges)
        return {
            "n_candidate_ticks": int(self.n_ticks),
            "mean_distinct_first_action_classes": round(mean_distinct, 6),
            "committed_class_entropy_nats": round(entropy, 6),
            "committed_class_entropy_fraction_of_ceiling": round(
                _entropy_fraction_of_ceiling(entropy, mean_distinct), 6
            ),
            "n_unique_committed_classes": int(len(self.committed_class_counts)),
            "authority_rel_deviation_mean": round(self.authority_rel_deviation_mean(), 6),
            "raw_score_range_mean": round(_mean_or0(self.raw_score_ranges), 8),
            "post_score_range_mean": round(_mean_or0(self.post_score_ranges), 8),
            "n_e3_select_fires": int(self.n_e3_select_fires),
            "n_e3_latched_ticks": int(self.n_e3_latched_ticks),
            "fresh_selects_sufficient": bool(self.n_e3_select_fires >= MIN_FRESH_SELECTS),
            "noise_bias_range_mean": round(noise_bias_mean, 8),
            "diag_raw_score_range_mean": round(diag_raw_mean, 8),
            "noise_to_raw_range_frac": round(
                noise_bias_mean / diag_raw_mean if diag_raw_mean > 1e-9 else 0.0, 6
            ),
            "dacc_max_suppression": round(self.dacc_max_suppression, 8),
            "sigma_scale_mean": round(_mean_or0(self.sigma_scale_trace), 6),
        }


def run_yoked_measurement(
    seed: int,
    trained_agents: Dict[str, REEAgent],
    episodes: int,
    steps: int,
    zg: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """A0_OFF's trained agent drives the environment; ARM_ANNEAL and ARM_NO_ANNEAL's trained
    agents are each independently stepped on the SAME observation sequence. Per tick, ALSO
    records A0_OFF's own raw_score_range (the state descriptor) paired with each comparison
    arm's divergence indicator, for the leg-ii/leg-iii criteria computed in run_experiment."""
    reset_all_rng(seed)
    ref = _YokedRunner(trained_agents["A0_OFF"])
    reset_all_rng(seed)
    runner_anneal = _YokedRunner(trained_agents["ARM_ANNEAL"])
    reset_all_rng(seed)
    runner_no_anneal = _YokedRunner(trained_agents["ARM_NO_ANNEAL"])
    reset_all_rng(seed)

    comparison = [("ARM_ANNEAL", runner_anneal), ("ARM_NO_ANNEAL", runner_no_anneal)]
    env = _make_env(seed)
    n_cmp: Dict[str, int] = {aid: 0 for aid, _ in comparison}
    n_diff: Dict[str, int] = {aid: 0 for aid, _ in comparison}
    # NEW: per-arm list of (a0_off_raw_score_range, diverged_bool), fresh-ref-ticks only.
    state_pairs: Dict[str, List[Tuple[float, bool]]] = {aid: [] for aid, _ in comparison}

    for ep in range(episodes):
        _, obs = env.reset()
        ref.reset_episode()
        for _, runner in comparison:
            runner.reset_episode()
        ep_diff = {aid: 0 for aid, _ in comparison}
        for _step in range(steps):
            a_ref = ref.choose(obs)
            ref_rsr = ref.last_raw_score_range if ref.last_tick_fresh else None
            for aid, runner in comparison:
                a_cmp = runner.choose(obs)
                n_cmp[aid] += 1
                diverged = (a_cmp != a_ref)
                if diverged:
                    n_diff[aid] += 1
                    ep_diff[aid] += 1
                if ref_rsr is not None:
                    state_pairs[aid].append((ref_rsr, diverged))
            action_onehot = torch.zeros(1, env.action_dim)
            action_onehot[0, a_ref] = 1.0
            _, harm_signal, done, _info, obs = env.step(action_onehot)
            ref.observe(harm_signal)
            for _, runner in comparison:
                runner.observe(harm_signal)
            if done:
                break
        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            print(
                f"  [train] yoked seed={seed} ep {ep + 1}/{episodes} "
                f"diverged(ANNEAL/NO_ANNEAL)={ep_diff['ARM_ANNEAL']}/{ep_diff['ARM_NO_ANNEAL']}",
                flush=True,
            )

    zg.observe(ref.agent)
    for _, runner in comparison:
        zg.observe(runner.agent)

    out: Dict[str, Any] = {"A0_OFF": {"summary": ref.summary()}}
    for aid, runner in comparison:
        out[aid] = {
            "summary": runner.summary(),
            "yoked_n_compared": n_cmp[aid],
            "yoked_n_diverged": n_diff[aid],
            "yoked_divergence_frac": (n_diff[aid] / n_cmp[aid]) if n_cmp[aid] else 0.0,
            "state_pairs": state_pairs[aid],
        }
    return out


def paired_control_divergence(arm: Dict[str, Any], seed: int, floor: int) -> float:
    """INSTRUMENT CONTROL: two FRESHLY, IDENTICALLY trained agents for the same arm+seed,
    yoked against each other, MUST diverge on zero ticks (949/955 precedent)."""
    agent_a, _ = train_arm_p0_p1(arm, seed, floor, CONTROL_P0_EPISODES, CONTROL_P1_EPISODES, CONTROL_STEPS)
    agent_b, _ = train_arm_p0_p1(arm, seed, floor, CONTROL_P0_EPISODES, CONTROL_P1_EPISODES, CONTROL_STEPS)
    reset_all_rng(seed)
    a = _YokedRunner(agent_a)
    reset_all_rng(seed)
    b = _YokedRunner(agent_b)
    reset_all_rng(seed)
    env = _make_env(seed)
    n = d = 0
    for _ in range(CONTROL_EPISODES):
        _, obs = env.reset()
        a.reset_episode()
        b.reset_episode()
        for _ in range(CONTROL_STEPS):
            aa = a.choose(obs)
            bb = b.choose(obs)
            n += 1
            if aa != bb:
                d += 1
            action_onehot = torch.zeros(1, env.action_dim)
            action_onehot[0, aa] = 1.0
            _, harm_signal, done, _info, obs = env.step(action_onehot)
            a.observe(harm_signal)
            b.observe(harm_signal)
            if done:
                break
    return (d / n) if n else 0.0


# ---------------------------------------------------------------------------
# Preconditions (readiness gates, regime-conditioned per arm -- never whole-run ANDed)
# ---------------------------------------------------------------------------


def _arm_specs(action_dim: int) -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="paired_control_is_bit_identical",
            description=(
                "INSTRUMENT CONTROL: an arm yoked against ITSELF (freshly, identically "
                "trained) diverges on 0 ticks. Non-zero means the runner's RNG isolation "
                "is broken and the whole DV is void."
            ),
            control="same arm vs same arm, identical seed and config, short fresh training",
            threshold=1e-9,
            direction="upper",
            kind="readiness",
        ),
        PreconditionSpec(
            name="armed_conversion_p1_candidate_diversity",
            description=(
                "ARC-065 ARMED-CONVERSION P1: mean_distinct_first_action_classes >= "
                "0.9 * action_dim, MEASURED not assumed, in EVERY arm."
            ),
            control="raised support_preserving_min_first_action_classes=action_dim, live P2 rollout",
            threshold=ARMED_CONVERSION_CLASSES_FRAC * action_dim,
            direction="lower",
            kind="readiness",
            structural_max=lambda ctx: float(action_dim),
        ),
        PreconditionSpec(
            name="armed_conversion_p2_authority_engaged",
            description=(
                "ARC-065 ARMED-CONVERSION P2: authority_rel_deviation_mean > 0.05 on the "
                "lever-bearing arms."
            ),
            control="live E3 select() fires, last_raw_scores vs last_scores",
            threshold=AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR,
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] in ("ARM_ANNEAL", "ARM_NO_ANNEAL"),
            applies_note=(
                "A0_OFF is the yoked reference with no exploration lever; the "
                "ARMED-CONVERSION P2 question is only meaningful for the two lever-bearing "
                "arms. A0_OFF's authority_rel_deviation_mean is still recorded as context."
            ),
        ),
        PreconditionSpec(
            name="fresh_selects_sufficient",
            description="at least MIN_FRESH_SELECTS genuinely-fresh E3 select() calls per cell.",
            control="708a/955's repaired sample-size-integrity instrument",
            threshold=float(MIN_FRESH_SELECTS),
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="noise_bias_supra_floor",
            description=(
                "The lever arms' per-candidate bias range clears NOISE_BIAS_RANGE_FLOOR."
            ),
            control="live noisy_selection_head diagnostics",
            threshold=NOISE_BIAS_RANGE_FLOOR,
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] in ("ARM_ANNEAL", "ARM_NO_ANNEAL"),
            applies_note="the noise-magnitude non-vacuity check applies only to the lever arms.",
        ),
        PreconditionSpec(
            name="dacc_live",
            description="the Go/No-Go perseveration axis (dACC) is live (max suppression > 0).",
            control="live _dacc_last_bundle suppression tensor",
            threshold=DACC_MAX_SUPPRESSION_FLOOR,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="ceiling_headroom_below_saturation",
            description=(
                "A0_OFF's committed_class_entropy_fraction_of_ceiling sits materially below "
                "1.0 (MEAN-aggregated across seeds -- the 955 driver-fix: max-aggregation let "
                "one anomalous seed void an otherwise-informative arm)."
            ),
            control="A0_OFF's own yoked-pass committed-action distribution, MEAN across seeds",
            threshold=CEILING_HEADROOM_MAX_FRAC,
            direction="upper",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] == "A0_OFF",
            applies_note="the ceiling-headroom check is specific to the falsifying (OFF) arm.",
        ),
    ]


def _arm_ctx(arm_id: str, action_dim: int) -> Dict[str, Any]:
    return {"arm_id": arm_id, "action_dim": action_dim}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _stratified_stats(pairs: List[Tuple[float, bool]]) -> Dict[str, Any]:
    """From a list of (raw_score_range, diverged) pairs for ONE seed x ONE arm: diverged/
    non-diverged raw_score_range means (for C_STATE), and decisive/near-tie divergence rates
    split at THIS pair-list's own median (for C_ANNEAL, called with A0_OFF-referenced pairs
    shared identically across both comparison arms at each seed -- see run_experiment)."""
    n = len(pairs)
    if n == 0:
        return {
            "n_pairs": 0, "sufficient": False,
            "mean_rsr_diverged": 0.0, "mean_rsr_not_diverged": 0.0,
            "median_rsr": 0.0, "divergence_rate_decisive": 0.0, "divergence_rate_near_tie": 0.0,
            "n_decisive": 0, "n_near_tie": 0,
        }
    rsrs = [p[0] for p in pairs]
    median_rsr = float(statistics.median(rsrs))
    diverged_rsrs = [r for r, d in pairs if d]
    not_diverged_rsrs = [r for r, d in pairs if not d]
    decisive = [(r, d) for r, d in pairs if r >= median_rsr]
    near_tie = [(r, d) for r, d in pairs if r < median_rsr]
    return {
        "n_pairs": n,
        "sufficient": bool(n >= MIN_STATE_PAIRS),
        "mean_rsr_diverged": round(_mean_or0(diverged_rsrs), 8),
        "mean_rsr_not_diverged": round(_mean_or0(not_diverged_rsrs), 8),
        "n_diverged": len(diverged_rsrs),
        "n_not_diverged": len(not_diverged_rsrs),
        "median_rsr": round(median_rsr, 8),
        "n_decisive": len(decisive),
        "n_near_tie": len(near_tie),
        "divergence_rate_decisive": round(
            _mean_or0([1.0 if d else 0.0 for _, d in decisive]), 6
        ),
        "divergence_rate_near_tie": round(
            _mean_or0([1.0 if d else 0.0 for _, d in near_tie]), 6
        ),
    }


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    probe_env = _make_env(seeds[0])
    action_dim = int(probe_env.action_dim)
    floor = action_dim
    zg = ZGoalStreamAccumulator()
    script_path = Path(__file__).resolve()

    all_ctxs = [_arm_ctx(a["arm_id"], action_dim) for a in ARMS]
    assert_no_structurally_unsatisfiable_gate(
        _arm_specs(action_dim), all_ctxs, arm_id_key="arm_id"
    )

    # INSTRUMENT CONTROL, before any scored compute.
    control_div: Dict[str, float] = {}
    for arm in ARMS:
        control_div[arm["arm_id"]] = paired_control_divergence(arm, seeds[0], floor)
        print(
            f"[control] {arm['arm_id']}: self-yoked divergence = "
            f"{control_div[arm['arm_id']]:.6f} (must be 0)", flush=True,
        )

    arm_results: List[Dict[str, Any]] = []
    p0p1_stats: List[Dict[str, Any]] = []
    per_seed_state: Dict[str, Dict[int, Dict[str, Any]]] = {"ARM_ANNEAL": {}, "ARM_NO_ANNEAL": {}}

    for seed in seeds:
        trained_agents: Dict[str, REEAgent] = {}
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm['label']}_p0p1", flush=True)
            agent, stats = train_arm_p0_p1(
                arm, seed, floor, p0_episodes, p1_episodes, steps_per_episode
            )
            trained_agents[arm["arm_id"]] = agent
            p0p1_stats.append(stats)
            print(f"verdict: {'PASS' if stats['error_note'] is None else 'FAIL'}", flush=True)

        print(f"Seed {seed} Condition yoked_p2_measurement", flush=True)
        quad = run_yoked_measurement(seed, trained_agents, p2_episodes, steps_per_episode, zg)
        yoked_ok = all(
            quad[aid]["summary"]["n_candidate_ticks"] > 0
            for aid in ("A0_OFF", "ARM_ANNEAL", "ARM_NO_ANNEAL")
        )
        print(f"verdict: {'PASS' if yoked_ok else 'FAIL'}", flush=True)

        for arm in ARMS:
            aid = arm["arm_id"]
            cell_data = quad[aid]
            row: Dict[str, Any] = {
                "arm_id": aid, "label": arm["label"], "seed": int(seed),
                "noise_head": bool(arm["noise_head"]), "anneal": bool(arm["anneal"]),
                **cell_data["summary"],
            }
            if aid != "A0_OFF":
                row["yoked_n_compared"] = cell_data["yoked_n_compared"]
                row["yoked_n_diverged"] = cell_data["yoked_n_diverged"]
                row["yoked_divergence_frac"] = cell_data["yoked_divergence_frac"]
                strat = _stratified_stats(cell_data["state_pairs"])
                row["state_stratification"] = strat
                per_seed_state[aid][seed] = strat
            if aid == "A0_OFF":
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(arm, floor),
                    seed=seed, script_path=script_path,
                    rng_fully_reset=True, config_slice_declared=True,
                    include_driver_script_in_hash=False,
                )
            else:
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(arm, floor),
                    seed=seed, script_path=script_path,
                    rng_fully_reset=True,
                    extra_ineligible_reasons=[
                        "leg_ii_iii_ablation_design_point_new_not_yet_a_proven_reusable_baseline",
                    ],
                )
            arm_results.append(row)

    # ---- per-arm readiness gates (regime-conditioned, never whole-run ANDed) ----
    arm_gates = []
    for arm in ARMS:
        aid = arm["arm_id"]
        rows = [r for r in arm_results if r["arm_id"] == aid]
        if not rows:
            continue
        ctx = _arm_ctx(aid, action_dim)
        measured = {
            "paired_control_is_bit_identical": control_div.get(aid, 1.0),
            "armed_conversion_p1_candidate_diversity": min(
                r["mean_distinct_first_action_classes"] for r in rows
            ),
            "armed_conversion_p2_authority_engaged": min(
                r["authority_rel_deviation_mean"] for r in rows
            ),
            "fresh_selects_sufficient": min(r["n_e3_select_fires"] for r in rows),
            "noise_bias_supra_floor": min(r["noise_bias_range_mean"] for r in rows),
            "dacc_live": min(r["dacc_max_suppression"] for r in rows),
            # THE 955 FIX: mean, not max, across seeds for the falsifying-branch headroom gate.
            "ceiling_headroom_below_saturation": statistics.fmean(
                r["committed_class_entropy_fraction_of_ceiling"] for r in rows
            ),
        }
        arm_gates.append(evaluate_arm_gate(aid, ctx, _arm_specs(action_dim), measured))
    aggregate = aggregate_arm_gates(arm_gates)
    green = set(aggregate["green_arms"])

    # ---- pre-registered criteria: C_STATE (leg ii) + C_ANNEAL (leg iii) ----
    eligible_seeds_state = [
        s for s in seeds if per_seed_state["ARM_ANNEAL"].get(s, {}).get("sufficient")
    ]
    eligible_seeds_anneal = [
        s for s in seeds
        if per_seed_state["ARM_ANNEAL"].get(s, {}).get("sufficient")
        and per_seed_state["ARM_NO_ANNEAL"].get(s, {}).get("sufficient")
    ]

    def _state_seed_passes(s: int) -> bool:
        st = per_seed_state["ARM_ANNEAL"].get(s)
        if not st or not st["sufficient"]:
            return False
        mrd = st["mean_rsr_diverged"]
        mrn = st["mean_rsr_not_diverged"]
        if mrn <= 1e-9 or st["n_diverged"] == 0 or st["n_not_diverged"] == 0:
            return False
        return bool(mrd <= (1.0 - STATE_COND_REL_MARGIN) * mrn)

    def _anneal_seed_passes(s: int) -> bool:
        sa = per_seed_state["ARM_ANNEAL"].get(s)
        sn = per_seed_state["ARM_NO_ANNEAL"].get(s)
        if not sa or not sn or not sa["sufficient"] or not sn["sufficient"]:
            return False
        if sa["n_decisive"] == 0 or sn["n_decisive"] == 0:
            return False
        decisive_gap = sn["divergence_rate_decisive"] - sa["divergence_rate_decisive"]
        near_tie_gap = abs(sa["divergence_rate_near_tie"] - sn["divergence_rate_near_tie"])
        return bool(
            decisive_gap >= ANNEAL_SUPPRESSION_MARGIN
            and near_tie_gap <= NEAR_TIE_PARITY_TOLERANCE
        )

    n_state_eligible = len(eligible_seeds_state)
    n_state_pass = sum(1 for s in eligible_seeds_state if _state_seed_passes(s))
    state_needed = math.ceil(DIVERGENT_PASS_FRACTION * n_state_eligible) if n_state_eligible else 0
    c_state = bool(
        n_state_eligible >= MIN_DIVERGENT_SEEDS and n_state_pass >= state_needed
    )

    n_anneal_eligible = len(eligible_seeds_anneal)
    n_anneal_pass = sum(1 for s in eligible_seeds_anneal if _anneal_seed_passes(s))
    anneal_needed = math.ceil(DIVERGENT_PASS_FRACTION * n_anneal_eligible) if n_anneal_eligible else 0
    c_anneal = bool(
        n_anneal_eligible >= MIN_DIVERGENT_SEEDS and n_anneal_pass >= anneal_needed
    )

    # ARM_NO_ANNEAL's own C_STATE-shaped reading, recorded as non-gating corroboration.
    n_no_anneal_state_pass = sum(
        1 for s in seeds
        if (lambda st: (
            st is not None and st["sufficient"] and st["n_diverged"] > 0
            and st["n_not_diverged"] > 0 and st["mean_rsr_not_diverged"] > 1e-9
            and st["mean_rsr_diverged"] <= (1.0 - STATE_COND_REL_MARGIN) * st["mean_rsr_not_diverged"]
        ))(per_seed_state["ARM_NO_ANNEAL"].get(s))
    )

    criteria = [
        {
            "name": "C_STATE_leg_ii_state_conditioning_impact_concentration",
            "load_bearing": True, "passed": bool(c_state),
            "measured": {
                "n_seeds_pass": n_state_pass, "n_seeds_eligible": n_state_eligible,
                "needed": state_needed,
                "no_anneal_corroborating_seeds_pass": n_no_anneal_state_pass,
            },
            "threshold": (
                f">= 2/3 of eligible seeds: diverged-tick mean raw_score_range <= "
                f"(1-{STATE_COND_REL_MARGIN}) * non-diverged-tick mean, on ARM_ANNEAL"
            ),
        },
        {
            "name": "C_ANNEAL_leg_iii_self_annealing_selective_suppression",
            "load_bearing": True, "passed": bool(c_anneal),
            "measured": {
                "n_seeds_pass": n_anneal_pass, "n_seeds_eligible": n_anneal_eligible,
                "needed": anneal_needed,
            },
            "threshold": (
                f">= 2/3 of eligible seeds: decisive-stratum divergence rate ANNEAL <= "
                f"NO_ANNEAL - {ANNEAL_SUPPRESSION_MARGIN}, AND near-tie-stratum rates within "
                f"{NEAR_TIE_PARITY_TOLERANCE}"
            ),
        },
    ]
    combination_rule = (
        "overall_pass = C_STATE (leg ii) AND C_ANNEAL (leg iii) AND every arm's readiness gate "
        "is green. A pass on only one criterion is `mixed`, naming which leg confirmed. "
        "VACUITY RULE: if any arm's readiness gate is red, the run self-routes "
        "substrate_not_ready_requeue regardless of C_STATE/C_ANNEAL."
    )

    armed_ok = bool(
        "A0_OFF" in green and "ARM_ANNEAL" in green and "ARM_NO_ANNEAL" in green
    )

    if not armed_ok:
        direction = "unknown"
        label = "substrate_not_ready_requeue"
    elif c_state and c_anneal:
        direction = "supports"
        label = "mech440_legs_ii_iii_state_conditioning_and_self_annealing_confirmed"
    elif c_state and not c_anneal:
        direction = "mixed"
        label = "mech440_leg_ii_state_conditioning_confirmed_leg_iii_self_annealing_not_confirmed"
    elif c_anneal and not c_state:
        direction = "mixed"
        label = "mech440_leg_iii_self_annealing_confirmed_leg_ii_state_conditioning_not_confirmed"
    else:
        direction = "weakens"
        label = "mech440_legs_ii_iii_neither_confirmed_at_raised_floor"

    overall_pass = bool(armed_ok and c_state and c_anneal)

    criteria_nd = arm_criteria_non_degenerate(
        {
            "ARM_ANNEAL": [
                "C_STATE_leg_ii_state_conditioning_impact_concentration",
                "C_ANNEAL_leg_iii_self_annealing_selective_suppression",
            ],
            "ARM_NO_ANNEAL": [
                "C_ANNEAL_leg_iii_self_annealing_selective_suppression",
            ],
        },
        aggregate,
    )

    # ---- descriptive (non-gating) sigma_scale sanity trace: arm-level means only, taken
    # from each row's own summary (the full per-tick trace stays in the runner, not the
    # manifest, per Step 3's "generous but not unbounded" recording norm).
    sigma_means = [r["sigma_scale_mean"] for r in arm_results if r["arm_id"] == "ARM_ANNEAL"]
    sigma_means_no_anneal = [
        r["sigma_scale_mean"] for r in arm_results if r["arm_id"] == "ARM_NO_ANNEAL"
    ]

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "related_claims": RELATED_CLAIMS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": "PASS" if overall_pass else "FAIL",
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "dry_run": dry_run,
        "sleep_driver_pattern": None,
        "secondary_checks_measured": True,
        "secondary_checks_note": (
            "This run measures MECH-440 legs (ii) state-conditioning and (iii) self-annealing "
            "directly, closing the honest gap V3-EXQ-955 left open (955 confirmed leg (i) "
            "propagation only). Leg (i) is NOT re-tested here -- see the routing autopsy "
            "failure_autopsy_V3-EXQ-955_2026-08-29 and GOV-REUSE-1 note above."
        ),
        "metrics": {
            "action_dim": action_dim,
            "class_floor_used": floor,
            "n_state_eligible_seeds": n_state_eligible,
            "n_state_pass_seeds": n_state_pass,
            "n_anneal_eligible_seeds": n_anneal_eligible,
            "n_anneal_pass_seeds": n_anneal_pass,
            "no_anneal_corroborating_state_pass_seeds": n_no_anneal_state_pass,
            "yoked_divergence_frac_anneal_mean": _mean_or0(
                [r["yoked_divergence_frac"] for r in arm_results if r["arm_id"] == "ARM_ANNEAL"]
            ),
            "yoked_divergence_frac_no_anneal_mean": _mean_or0(
                [r["yoked_divergence_frac"] for r in arm_results if r["arm_id"] == "ARM_NO_ANNEAL"]
            ),
            "sigma_scale_mean_anneal": _mean_or0(sigma_means),
            "sigma_scale_mean_no_anneal": _mean_or0(sigma_means_no_anneal),
            "authority_rel_deviation_mean_off": _mean_or0(
                [r["authority_rel_deviation_mean"] for r in arm_results if r["arm_id"] == "A0_OFF"]
            ),
            "authority_rel_deviation_mean_anneal": _mean_or0(
                [r["authority_rel_deviation_mean"] for r in arm_results if r["arm_id"] == "ARM_ANNEAL"]
            ),
            "authority_rel_deviation_mean_no_anneal": _mean_or0(
                [r["authority_rel_deviation_mean"] for r in arm_results if r["arm_id"] == "ARM_NO_ANNEAL"]
            ),
            "mean_distinct_first_action_classes_off": _mean_or0(
                [r["mean_distinct_first_action_classes"] for r in arm_results if r["arm_id"] == "A0_OFF"]
            ),
            "committed_class_entropy_fraction_of_ceiling_off_mean": _mean_or0(
                [r["committed_class_entropy_fraction_of_ceiling"] for r in arm_results if r["arm_id"] == "A0_OFF"]
            ),
            "paired_control_divergence_max": max(control_div.values()) if control_div else 1.0,
        },
        "criteria": criteria,
        "combination_rule": combination_rule,
        "arm_results": arm_results,
        "p0p1_training_stats": p0p1_stats,
        "per_arm_gate": aggregate["per_arm_gate"],
        "non_degenerate": aggregate["non_degenerate"],
        "degeneracy_reason": aggregate["degeneracy_reason"],
        "interpretation": {
            "label": label,
            "preconditions": aggregate["adjudication_preconditions"],
            "preconditions_scope_note": aggregate.get("preconditions_scope_note", ""),
            "criteria_non_degenerate": criteria_nd,
        },
        "dv_symmetry_declaration": {
            "A0_OFF": (
                "reference; no exploration lever. Drives the yoked walk and supplies the "
                "raw_score_range state descriptor. Never diverges from itself."
            ),
            "ARM_ANNEAL": (
                "per-candidate factorised-Gaussian weight noise, resampled per tick, scaled by "
                "a confidence-EMA anneal factor. Not a uniform broadcast (varies per-candidate "
                "via x_k), not a pure monotone rescaling -- can flip the committed argmin by "
                "construction; yoked divergence measures this directly."
            ),
            "ARM_NO_ANNEAL": (
                "identical injection mechanism to ARM_ANNEAL; sigma pinned at sigma_init instead "
                "of anneal-scaled -- same DV-symmetry reasoning as ARM_ANNEAL."
            ),
        },
        "custom_information": {
            "re_derive_brake_note": (
                "0 substrate_ceiling-category autopsies for MECH-440 in the corpus (2026-08-29 "
                "check). 955 resolved MEASURES FAILED (adjudication-branch defect), not "
                "substrate_ceiling. NOT braked."
            ),
            "substrate_defect_note": (
                "Same two OPEN corrupting substrate_queue entries as 955 overlap this driver's "
                "imports: mode-governance-engagement (not in causal path, no mode-governance "
                "knob enabled) and contextmemory-write-path-addressing-degeneracy (potentially "
                "in path via E1 but applies identically to every arm, biasing the load-bearing "
                "CONTRAST toward the null, not a false positive) -- both carved out per 949/955's "
                "identical precedent."
            ),
            "gov_reuse_1_note": (
                "Per-tick A0_OFF-referenced raw_score_range paired with lever-arm divergence, "
                "stratified by seed-median, and an anneal-off ablation arm are recorded nowhere "
                "in the corpus; 955 recorded only per-cell summary divergence fractions with no "
                "anneal-off arm. Not recoverable by reanalysis -> run. A0_OFF is NOT reused from "
                "955 despite being config-slice-identical and reuse-eligible there -- this run's "
                "yoked pass requires a LIVE, jointly-stepped A0_OFF agent, which the arm-reuse "
                "mechanism (built for independently-scored arms) cannot supply."
            ),
            "seed_swap_note": (
                "Seed 44 (present in 955) replaced with 48 here -- 955's autopsy diagnoses seed "
                "44 as a coupled near-saturation anomaly on this env/floor combination; dropped "
                "to avoid noisy first-measurement data for the new per-tick stratification "
                "instrument introduced by this driver."
            ),
            "ceiling_headroom_fix_note": (
                "955's max-aggregated ceiling-headroom precondition let one anomalous seed void "
                "an otherwise-informative arm (autopsy-confirmed defect). This driver "
                "MEAN-aggregates the same precondition's measured value across seeds instead, "
                "per the autopsy's own driver-fix note."
            ),
            "state_conditioning_operationalization_note": (
                "C_STATE tests IMPACT CONCENTRATION (does divergence concentrate in near-tie "
                "states), not raw bias-magnitude covariance with state -- the latter is close to "
                "tautological given the noise head's linear dependence on candidate features "
                "within a fixed-eps tick. See module docstring for the full argument."
            ),
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }

    full_config = {
        "seeds": seeds,
        "p0_episodes": p0_episodes,
        "p1_episodes": p1_episodes,
        "p2_episodes": p2_episodes,
        "steps_per_episode": steps_per_episode,
        "action_dim": action_dim,
        "class_floor": floor,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "env_kwargs": dict(ENV_KWARGS),
        "thresholds": {
            "ARMED_CONVERSION_CLASSES_FRAC": ARMED_CONVERSION_CLASSES_FRAC,
            "AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR": AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR,
            "CEILING_HEADROOM_MAX_FRAC": CEILING_HEADROOM_MAX_FRAC,
            "MIN_DIVERGENT_SEEDS": MIN_DIVERGENT_SEEDS,
            "STATE_COND_REL_MARGIN": STATE_COND_REL_MARGIN,
            "ANNEAL_SUPPRESSION_MARGIN": ANNEAL_SUPPRESSION_MARGIN,
            "NEAR_TIE_PARITY_TOLERANCE": NEAR_TIE_PARITY_TOLERANCE,
            "MIN_STATE_PAIRS": MIN_STATE_PAIRS,
        },
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    return {"outcome": manifest["outcome"], "manifest": manifest, "out_path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, p2, steps = DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_P2, DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0, p1, p2, steps = (
            P0_WARMUP_EPISODES, P1_BIAS_TRAIN_EPISODES, P2_YOKED_EPISODES, STEPS_PER_EPISODE
        )

    result = run_experiment(
        seeds=seeds, p0_episodes=p0, p1_episodes=p1, p2_episodes=p2,
        steps_per_episode=steps, dry_run=bool(args.dry_run),
    )
    out_path = result["out_path"]
    print(f"manifest: {out_path}", flush=True)
    m = result["manifest"]["metrics"]
    print(
        f"outcome={result['outcome']} label={result['manifest']['interpretation']['label']} "
        f"state_pass={m['n_state_pass_seeds']}/{m['n_state_eligible_seeds']} "
        f"anneal_pass={m['n_anneal_pass_seeds']}/{m['n_anneal_eligible_seeds']} "
        f"sigma_scale(anneal/no_anneal)={m['sigma_scale_mean_anneal']:.4f}/"
        f"{m['sigma_scale_mean_no_anneal']:.4f} "
        f"paired_control_max={m['paired_control_divergence_max']:.6f}",
        flush=True,
    )
    if args.dry_run:
        checks = {
            "candidate diversity engaged (mean_distinct > 1 on A0_OFF)":
                m["mean_distinct_first_action_classes_off"] > 1.0,
            "INSTRUMENT CONTROL: self-yoked arms are bit-identical (==0)":
                m["paired_control_divergence_max"] == 0.0,
            "sigma_scale field present and numeric (anneal + no_anneal)": (
                m["sigma_scale_mean_anneal"] >= 0.0 and m["sigma_scale_mean_no_anneal"] >= 0.0
            ),
        }
        print("[smoke] decisive-readout engagement checks:", flush=True)
        for label_, ok in checks.items():
            print(f"  [{'OK' if ok else 'XX'}] {label_}", flush=True)

    raw = str(result["outcome"]).upper()
    return (raw if raw in ("PASS", "FAIL") else "FAIL"), out_path, args.dry_run


if __name__ == "__main__":
    _outcome_raw, _out_path, _dry = main()
    emit_outcome(outcome=_outcome_raw, manifest_path=_out_path, dry_run=_dry)
