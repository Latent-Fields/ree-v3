#!/opt/local/bin/python3
"""
V3-EXQ-708b -- MECH-440 PRE-COMMIT DISTRIBUTION *SHAPE* FALSIFIER (diagnostic).

SAME-QUESTION RE-RUN OF V3-EXQ-708a -- THE READOUT CHANGES, THE DESIGN DOES NOT.
supersedes: v3_exq_708a_mech440_noisy_selection_head_propagation_falsifier_20260720T211903Z_v3.

WHY THIS EXISTS (read before editing the measurement path)
----------------------------------------------------------
V3-EXQ-708a's instrument repair WORKED and is NOT in question. Its own numbers prove it:
862 genuinely-fresh E3 select() calls against a floor of 30, 4 divergent seeds, a
supra-floor and non-trivial-vs-raw noise bias range (0.221), live dACC suppression, and
dissociable finer channels. The 708 pseudo-replication defect is gone.

What 708a found instead is SUBSTANTIVE and is the reason for this run: on that
demonstrably-working instrument, NEITHER DESIGNED LEVER MOVES THE DEPENDENT VARIABLE.
ARM_TEMP (the V3-EXQ-687 temperature control) raises pre-commit sampling-class entropy
above A0_OFF on 0 of 4 divergent seeds; ARM_NOISE_SINGLE does so on 1 of 4; the bar was 2.
That is a failed MANIPULATION CHECK on a working instrument -- a finding about the system,
NOT an unready substrate (failure_autopsy_V3-EXQ-708a_2026-07-22, user-adjudicated
2026-07-22: "708a: record as a substantive finding").

THE TWO LIVE READINGS -- and why 708a's manifest cannot separate them
--------------------------------------------------------------------
  (i)  H-PRECOMMIT-SATURATED. The pre-commit distribution is already argmax-like, with no
       entropy headroom for any lever to exploit. This is the MECH-439 F-dominance
       signature, observed PRE-commit rather than post-commit.
  (ii) H-LEVER-DOWNSTREAM. Temperature and weight noise are applied downstream of where the
       pre-commit distribution is formed, so they cannot move it by construction. A wiring
       gap, not a substrate property.

708a recorded ONLY the between-arm entropy DELTA. A single scalar cannot tell
"already at ceiling" from "lever never arrived". That -- and only that -- is what changes here.

THE READOUT CHANGE (three additions; arms, seeds, phases, stack all UNCHANGED)
-----------------------------------------------------------------------------
  1. PRE-COMMIT DISTRIBUTION SHAPE, not just its entropy. Per genuinely-fresh select():
       max_prob                 -- the top candidate's pre-commit mass
       participation_ratio      -- 1 / sum(p_i^2), the continuous effective support
       eff_support              -- count of candidates with p_i >= PRECOMMIT_MASS_FLOOR
       sorted top-3 mass profile, n_candidates
     plus the CLASS-level analogues (max_class_mass, n_distinct_classes), because the DV is a
     CLASS entropy: candidate-level headroom can coexist with class-level collapse.
     Aggregated PER ARM PER SEED INTO THE MANIFEST -- deliberately NOT a per_tick_sink, since
     Phase 3 cloud workers transport only manifest_bytes (twice-confirmed: 785 and 708).

     THE DISCRIMINATOR, pre-registered:
       under (i)  max_prob is near 1 in EVERY arm INCLUDING A0_OFF -> saturated everywhere;
       under (ii) max_prob DIFFERS by arm while the class entropy does not -> the lever
                  reaches the distribution's shape but not the class marginal.

  2. THE LEVER'S OWN ARRIVAL, instrumented at the E3 call site -- WITHOUT touching ree_core.
     The applied softmax temperature is RECOVERED EXACTLY, per fresh select, by inverting the
     pre-commit softmax from two already-recorded substrate attributes:
         e3_selector.py:2742  probs  = F.softmax(-scores / temperature, dim=0)
         e3_selector.py:2712  self.last_scores        = scores.detach()   (POST-explore-term)
         e3_selector.py:2747  self.last_precommit_probs = probs.detach()
       => log p_i - log p_j = -(s_i - s_j) / T
       => T = (s_j - s_i) / (log p_i - log p_j)   [taken on the top-2 pair for stability]
     A recovered T on ARM_TEMP strictly above A0_OFF's IS direct evidence that the MECH-313
     temperature lift arrived AT the pre-commit softmax -- which is the exact claim reading
     (ii) denies. Likewise the weight noise: e3_selector.py:2698 does
     `scores = scores + _explore_term` and sets `self._last_explore_term`, and BOTH precede
     the softmax at :2742, so `_last_explore_term is not None` on a fresh select is a
     positive, per-tick witness that the MECH-440 perturbation entered the very score vector
     the pre-commit softmax normalises. Both are recorded as measured values, per arm-seed:
     lever_arrival_temperature_recovered / lever_arrival_explore_term_frac.

     STATIC READING OF THE SUBSTRATE (recorded for the reader, not a substitute for the
     measurement): both levers ARE upstream of the pre-commit softmax in the code as landed,
     which already argues against (ii). The point of measuring is that a static read cannot
     see a lever that is present but inert (a noise_floor that returns the base temperature, a
     head whose bias is dropped by the non-finite guard at :2694). The measurement can.

  3. RECLASSIFIED MANIPULATION CHECKS. `temperature_control_raises_precommit_entropy` and
     `weight_noise_raises_precommit_entropy` move OUT of the adjudicating
     `interpretation.preconditions[]` list and into `interpretation.recorded_preconditions[]`,
     with an explicit `preconditions_scope_note` (the V3-EXQ-737 pattern) and a descriptive
     `kind: "manipulation_check"`. RATIONALE: the schema has no manipulation_check kind, and
     build_experiment_indexes._compute_adjudication reads the flat `preconditions` list
     ARM-BLIND, returning whole-run `precondition_unmet` on the first unmet entry. In 708a
     that buried a working instrument's real finding as "substrate not ready". A null on a
     manipulation check IS this run's result and must report as one. They keep honest
     measured/threshold/met so any recompute agrees; they are surfaced, not adjudicating.

WHAT IS DELIBERATELY UNCHANGED (do not "improve" these)
-------------------------------------------------------
* THE 708a INSTRUMENT REPAIR, verbatim: agent.e3.last_score_diagnostics and
  .last_precommit_probs are cleared to None IMMEDIATELY before every agent.select_action and a
  row is recorded ONLY if select() repopulated them (pattern from v3_exq_785a, lifted into
  _lib at ree-v3 08e9955). 708b additionally clears .last_scores, which latches identically and
  is newly load-bearing for the temperature recovery. That repair is validated; it is not in
  question and must not be relaxed.
* The 3 arms, the 6 seeds, the P0/P1/P2 phase structure, the matched SOTA conversion stack,
  the GAP-A reef-bipartite env, and every lever constant.
* claim_ids = [MECH-440]. experiment_purpose = diagnostic. PROMOTES AND DEMOTES NOTHING.
* NO substrate build is warranted until (i) vs (ii) is settled: the 708a autopsy sets
  recommended_substrate_queue_entry.action = "none". This run does NOT route to
  /implement-substrate under ANY outcome. It routes to /failure-autopsy.

THE 3 ARMS (all carry the SAME landed SOTA conversion stack as a MATCHED CONSTANT; the ONLY
swept factor is the exploration-injection site):
  A0_OFF            : no exploration injection. The pre-commit shape FLOOR -- and, under
                      reading (i), the arm whose max_prob near 1 is decisive.
  ARM_TEMP          : the 687 non-propagating temperature control (use_noise_floor=True).
  ARM_NOISE_SINGLE  : MECH-440 factorised-Gaussian selection-head weight noise.
6 seeds x 3 arms = 18 cells. PRIMARY READOUT = pre-commit distribution SHAPE (per arm per
seed). The 708a DVs (committed-class entropy, pre-commit class entropy) are retained
unchanged so this run is directly comparable to its predecessor.

PRE-REGISTERED OUTCOME MAP (the load-bearing criterion is the DISCRIMINATION, not MECH-440)
-------------------------------------------------------------------------------------------
  PASS / discrimination achieved -- reading (i) H-PRECOMMIT-SATURATED:
    every arm INCLUDING A0_OFF sits at mean max_prob >= PRECOMMIT_SATURATION_MAX_PROB, AND
    both levers demonstrably ARRIVED (recovered T lifted on ARM_TEMP; explore-term witnessed
    on ARM_NOISE_SINGLE). The distribution has no headroom; the levers are not the problem.
    Routes the F-dominance reading PRE-commit. MECH-440 remains non_contributory (untested).

  PASS / discrimination achieved -- reading (ii) H-LEVER-DOWNSTREAM:
    a lever FAILED to arrive (ARM_TEMP's recovered pre-commit temperature is NOT lifted above
    A0_OFF's, and/or ARM_NOISE_SINGLE witnesses no explore term in the pre-commit score
    vector) while the shape is NOT saturated. A wiring gap. MECH-440 non_contributory.

  FAIL / discrimination NOT achieved (the third, genuinely-new state):
    the levers arrived AND the shape has headroom, yet neither entropy nor class marginal
    moves. Neither (i) nor (ii). Recorded as
    precommit_shape_headroom_unexplained -- with the candidate-vs-class shape split retained,
    which is where the next hypothesis will come from. MECH-440 non_contributory.

  PASS / supports MECH-440 (retained; reachable only if the manipulation now fires):
    both manipulation checks pass AND C1 holds (ARM_NOISE_SINGLE committed-class entropy
    strict-above ARM_TEMP on a strict-majority of divergent seeds).

  FAIL / substrate_not_ready_requeue: a genuine READINESS gate is unmet (too few divergent
    seeds / sub-floor noise bias range / flat dACC / learning not engaged / too few fresh
    selects / too few recoverable shape samples). NOT a falsification. Note the two
    manipulation checks can NO LONGER reach this branch -- that is the point of change 3.

RE-DERIVE BRAKE: NOT FIRED. The 708a autopsy records epistemic_category `measurement_gap`,
explicitly NOT `substrate_ceiling`, and recommended_substrate_queue_entry.action "none", so
the counting rule is not met. This is a readout change on a working instrument -- the same
shape as 708 -> 708a, one layer further in.

GOV-REUSE-1: the decisive readout is the per-fresh-select pre-commit distribution SHAPE
(max_prob / participation ratio / effective support) and the recovered applied temperature.
Checked v3_exq_708a_..._20260720T211903Z_v3 and v3_exq_708_..._20260628T220908Z_v3: both
record only the between-arm pre-commit class-entropy scalar, no shape moments, no per-tick
sink (Phase 3 transports manifest_bytes only). Not recoverable by reanalysis -> run.

See REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-708a_2026-07-22.md (sections 2, 6, 7),
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-708_2026-07-19.json (predecessor),
    ree-v3/ree_core/predictors/e3_selector.py:2698,2712,2742,2747 (the measured call site),
    ree-v3/ree_core/agent.py:6783-6829 (where the MECH-313 temperature lift is composed).
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


EXPERIMENT_TYPE = "v3_exq_708b_mech440_precommit_distribution_shape_falsifier"
QUEUE_ID = "V3-EXQ-708b"
# The 708a RUN, not the queue id: this run supersedes that specific landed manifest.
SUPERSEDES = (
    "v3_exq_708a_mech440_noisy_selection_head_propagation_falsifier_20260720T211903Z_v3"
)
BACKLOG_ID = None   # routed by cpkt_tonic_exploration_noise_build_decision_2026-06-27
CLAIM_IDS: List[str] = ["MECH-440"]
EXPERIMENT_PURPOSE = "diagnostic"   # PROMOTES NOTHING

# validate_experiments.py --checks anchor_reachability asks that an anchor-kind readiness gate
# prove its threshold is REACHABLE by its positive control, so an unmeetable-by-construction
# predicate cannot masquerade as a substrate verdict. Reachability here is established
# EMPIRICALLY by the predecessor's landed manifest rather than by a synthetic reference cell:
# 708b's readiness gates are the FOUR that V3-EXQ-708a's own landed manifest recorded met=True
# on a REPAIRED instrument over 18 completed arm-seeds of this exact substrate and config:
# fresh_selects_sufficient 862/30, enough_divergent_seeds 4/3,
# noise_bias_range_supra_floor_vs_raw 0.221/1e-4, dacc_suppression_live 1.0/0.0,
# learning_engaged_finer_channels_dissociable 0.00127/1e-4. Every one cleared with headroom.
# The two 708a entries that recorded met=False -- temperature_control_raises_precommit_entropy
# (0/4 seeds) and weight_noise_raises_precommit_entropy (1/4) -- are RECLASSIFIED here as
# MANIPULATION CHECKS carried under recorded_preconditions, so they no longer gate. Their null
# IS this run's finding; see the module docstring, change 3.
# The one NEW gate (shape_samples_sufficient) rides the same fresh-select denominator as
# fresh_selects_sufficient, which cleared 862 against 30, so it has the same ~30x headroom.
# CAUTION for the reader: enough_divergent_seeds cleared at 4 against a threshold of 3 in 708a
# (3.0/3.0 in 708) -- near-zero headroom, so a lost divergent seed self-routes this run to
# substrate_not_ready_requeue. That is correct conservative behaviour, not a defect.
ANCHOR_REACHABILITY_EXEMPT = (
    "reachability established empirically by the predecessor V3-EXQ-708a's landed manifest, "
    "which recorded met=True on ALL FOUR retained readiness gates over 18 completed arm-seeds "
    "on this exact substrate+config with a REPAIRED instrument (862 fresh selects vs a floor "
    "of 30). The 2 gates 708a recorded unmet are reclassified here as non-gating manipulation "
    "checks; the new shape_samples_sufficient floor rides the same 862-vs-30 denominator."
)

_FCG_W_INIT = math.log(math.e - 1.0)

# CRF-gate calibration levers (matured CRF stack; ported from 707, matched on all arms).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# ----- Acceptance thresholds (pre-registered) -----
# C1 conversion: strict-above margin on committed-class entropy (nats).
CONVERSION_MARGIN = 0.05
# Non-vacuity: an arm "raises pre-commit entropy" iff its pre-commit-class entropy is at least
# this margin above A0_OFF's pre-commit-class entropy.
PRECOMMIT_LIFT_MARGIN = 0.05

# ----- Per-seed-divergent gating -----
MIN_DIVERGENT_SEEDS = 3          # of 6: fewer divergent seeds => substrate_not_ready_requeue
DIVERGENT_PASS_FRACTION = 0.5    # strict-majority-ish gate within the divergent seeds
MIN_SEEDS_FOR_PASS = 2           # absolute floor of divergent seeds clearing a criterion

# Non-vacuity: the noise arms must inject a per-candidate bias range above this floor, AND it
# must be a non-trivial fraction of the raw-score range (so it CAN flip the within-eligible
# argmin). A noise too small to ever compete is a vacuous test -> requeue.
NOISE_BIAS_RANGE_FLOOR = 1e-4
NOISE_BIAS_TO_RAW_RANGE_FRAC_FLOOR = 0.02

# dACC non-vacuity: the Go/No-Go perseveration axis must be live (max suppression > 0).
DACC_MAX_SUPPRESSION_FLOOR = 0.0   # strictly > this

# --- 708a fresh-select sufficiency (the repaired instrument's own readiness gate) ---
# The DV now accumulates ONLY on verified-fresh E3 selections. E3 cadence is 5-20 env steps
# (MECH-093 arousal-modulated; default 10), so P2's honest denominator is roughly
# n_p2_ticks / 10. Below this floor the class marginal is too sparse to estimate and the
# correct route is substrate_not_ready_requeue, NEVER a weakens on an underpowered DV.
MIN_FRESH_SELECTS = 30
# Exposure imbalance vs A0_OFF beyond this fraction is REPORTED (never gating): it is the
# mechanism by which 708's per-env-step read distorted the between-arm delta
# (+1.7% / +57.5% / +32.1% on divergent seeds 44/45/46).
EXPOSURE_IMBALANCE_REPORT_FLOOR = 0.05

# C1(a) readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# Non-vacuity (b): GAP-A consumed-summary divergence.
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# Non-vacuity (c): delta_t carries cross-tick variance.
DELTA_T_STD_FLOOR = 1e-4
# Finer-channel learning engaged (matched constant).
W_CHAN_FINER_RANGE_FLOOR = 1e-4

# ----- 708b PRE-COMMIT SHAPE READOUT (the new instrument) -----
# A candidate carries "real" pre-commit mass at or above this floor; eff_support counts them.
PRECOMMIT_MASS_FLOOR = 0.05
# Reading (i) H-PRECOMMIT-SATURATED: an arm is SATURATED when its mean top-candidate mass is
# at or above this. Pre-registered at 0.95 -- at max_prob 0.95 the residual mass over ALL other
# candidates is 0.05, i.e. below the single-candidate mass floor above, so no lever acting on
# the score vector can redistribute enough mass to move a class marginal.
PRECOMMIT_SATURATION_MAX_PROB = 0.95
# Reading (ii) H-LEVER-DOWNSTREAM discriminator: an arm's mean max_prob "differs from A0_OFF"
# when it moves by at least this much. Deliberately an order of magnitude below the saturation
# residual, so a lever that reaches the SHAPE but not the class marginal is still detected.
MAXPROB_ARM_DELTA_FLOOR = 0.01
# ARM_TEMP lever arrival: the recovered pre-commit softmax temperature must exceed A0_OFF's by
# this RELATIVE fraction. The configured lift is alpha=1.5 on a base of 1.0 (a ~150% lift), so
# a 5% floor is ~30x inside the designed effect and cannot be met by recovery noise alone.
TEMP_ARRIVAL_REL_FLOOR = 0.05
# ARM_NOISE_SINGLE lever arrival: the fraction of fresh selects on which e3._last_explore_term
# is non-None -- a positive per-tick witness that the MECH-440 bias entered the very score
# vector the pre-commit softmax normalises (e3_selector.py:2698 precedes :2742).
EXPLORE_TERM_ARRIVAL_FRAC_FLOOR = 0.5
# Temperature recovery needs a strictly-positive second-largest probability; a p2 that has
# underflowed is itself evidence of saturation and is COUNTED, never silently dropped.
TEMP_RECOVERY_MIN_P2 = 1e-12
# Readiness: every arm-seed must recover at least this many usable shape samples. Rides the
# same fresh-select denominator as MIN_FRESH_SELECTS (708a measured 862 against 30).
MIN_SHAPE_SAMPLES = 30

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

# --- MECH-440 injection-site lever constants ---
# Weight-noise arms: factorised-Gaussian sigma_init + overall weight on the per-candidate bias.
NOISY_SELECTION_SIGMA_INIT = 1.0
NOISY_SELECTION_WEIGHT = 1.0
NOISY_SELECTION_ANNEAL = True
NOISY_SELECTION_ANNEAL_FLOOR = 0.1
NOISY_SELECTION_ANNEAL_EMA_ALPHA = 0.01
# ARM_TEMP: the matched-pre-commit-variance temperature control (the 687 non-propagating lift).
# effective_T = max(base_T + alpha, min_temperature); base_T defaults to 1.0 (select_action).
TEMP_NOISE_FLOOR_ALPHA = 1.5
TEMP_NOISE_FLOOR_MIN_TEMPERATURE = 1.0

# --- Matched-stack lever constants (identical on ALL arms; the landed SOTA conversion stack) ---
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

LOOP_SEGREGATION_NORMALIZE = "zscore"

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


# The 4 arms. ALL carry the matched SOTA conversion stack + finer gating + learned settling;
# the ONLY swept factor is the exploration-injection site (none / temperature / weight-noise /
# weight-noise+loop-segregation).
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "A0_OFF",
        "label": "no_exploration_injection_single_arena_baseline",
        "noise_head": False, "temp": False, "loop_seg": False,
    },
    {
        "arm_id": "ARM_TEMP",
        "label": "matched_pre_commit_variance_temperature_control_687_non_propagating",
        "noise_head": False, "temp": True, "loop_seg": False,
    },
    {
        "arm_id": "ARM_NOISE_SINGLE",
        "label": "mech440_noisy_selection_head_weight_noise_single_arena",
        "noise_head": True, "temp": False, "loop_seg": False,
    },
    # ARM_NOISE_LOOPSEG (ARC-110 loop segregation) DROPPED in 708a -- see the VENUE block in
    # the module docstring: V3-EXQ-707b answered the single-arena-artefact sub-hypothesis with
    # a valid null, so that arm would re-pose a settled question.
]


def _arm_config_slice(
    arm: Dict[str, Any],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Declared reuse fingerprint slice: ONLY what an arm's computation reads."""
    return {
        "arm_id": arm["arm_id"],
        "noise_head": bool(arm["noise_head"]),
        "temp": bool(arm["temp"]),
        "loop_seg": bool(arm["loop_seg"]),
        "noisy_selection_sigma_init": float(NOISY_SELECTION_SIGMA_INIT),
        "noisy_selection_weight": float(NOISY_SELECTION_WEIGHT),
        "temp_noise_floor_alpha": float(TEMP_NOISE_FLOOR_ALPHA),
        "temp_noise_floor_min_temperature": float(TEMP_NOISE_FLOOR_MIN_TEMPERATURE),
        "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
        "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
        "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
        "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
        "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
        "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
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


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Matched-stack agent. The landed SOTA conversion stack (569i top-k + MECH-448 demotion +
    adaptive floor + Go/No-Go + modulatory authority + routing) + the diversity stack (MECH-341,
    SD-056, CRF, trained lateral_pfc bias head, use_dacc) + finer-channel gating + learned
    settling are MATCHED CONSTANTS on ALL arms. The ONLY swept factor is the exploration-
    injection site: ARM_TEMP arms MECH-313 noise_floor (a softmax-temperature lift); the
    NOISE arms arm MECH-440 use_noisy_selection_head (factorised-Gaussian weight noise into the
    committed argmin). 708a runs no loop-segregation arm, so loop_seg is False on every arm."""
    noise_head = bool(arm["noise_head"])
    temp = bool(arm["temp"])
    loop_seg = bool(arm["loop_seg"])
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
        # --- MECH-313 temperature noise floor: ARMED ONLY on ARM_TEMP (the 687 control) ---
        use_noise_floor=temp,
        noise_floor_alpha=(TEMP_NOISE_FLOOR_ALPHA if temp else 0.1),
        noise_floor_min_temperature=(TEMP_NOISE_FLOOR_MIN_TEMPERATURE if temp else 1.0),
        # --- MECH-440 NoisyNet propagating selection-head weight noise: ARMED on the NOISE arms ---
        use_noisy_selection_head=noise_head,
        noisy_selection_sigma_init=(NOISY_SELECTION_SIGMA_INIT if noise_head else 0.0),
        noisy_selection_weight=NOISY_SELECTION_WEIGHT,
        noisy_selection_anneal=NOISY_SELECTION_ANNEAL,
        noisy_selection_anneal_floor=NOISY_SELECTION_ANNEAL_FLOOR,
        noisy_selection_anneal_ema_alpha=NOISY_SELECTION_ANNEAL_EMA_ALPHA,
        # MECH-441 OFF (this falsifier is the MECH-440 leg only).
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
        # MECH-451 finer channels ON on all arms (the loops partition these; matched constant).
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
        # --- ARC-110 loop segregation: OFF on every 708a arm (loop_seg is False throughout;
        # the ARM_NOISE_LOOPSEG arm was dropped -- see the VENUE block). Kept wired so the
        # config surface stays identical to 708's for a like-for-like comparison. ---
        use_loop_segregation=loop_seg,
        loop_segregation_normalize=LOOP_SEGREGATION_NORMALIZE,
        use_d1_d2_population_split=loop_seg,
        use_loop_local_eligibility_traces=loop_seg,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (verbatim from 707)
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


def _mean_or0(vals: List[float]) -> float:
    """Mean of a possibly-empty sample. Empty -> 0.0 (the row also carries its own n)."""
    return float(sum(vals) / len(vals)) if vals else 0.0


def _shape_of(ppv: torch.Tensor) -> Dict[str, Any]:
    """708b: the SHAPE moments of one pre-commit softmax distribution over candidates.

    708a recorded only the between-arm ENTROPY DELTA, which cannot distinguish "the
    distribution is already argmax-like and has no headroom" (reading i) from "the lever is
    applied downstream and never reaches this distribution" (reading ii). These moments can:
    under (i) max_prob is near 1 in EVERY arm INCLUDING the OFF baseline.

    participation_ratio = 1 / sum(p^2) is the continuous effective support (1 when one
    candidate holds all mass, n under a uniform distribution) and is reported alongside the
    thresholded eff_support count because the two fail differently: the count is blind to
    mass moving BELOW the floor, the ratio is blind to how many candidates carry it.
    """
    p = ppv.detach().reshape(-1)
    n = int(p.numel())
    srt, _ = torch.sort(p, descending=True)
    sq = float((p * p).sum().item())
    return {
        "max_prob": float(srt[0].item()),
        "participation_ratio": (1.0 / sq) if sq > 0.0 else float(n),
        "eff_support": int((p >= PRECOMMIT_MASS_FLOOR).sum().item()),
        "top1": float(srt[0].item()),
        "top2": float(srt[1].item()) if n >= 2 else 0.0,
        "top3": float(srt[2].item()) if n >= 3 else 0.0,
    }


def _recover_temperature(
    scores: torch.Tensor, probs: torch.Tensor
) -> Optional[float]:
    """708b: recover the APPLIED pre-commit softmax temperature, exactly, from the substrate.

    ree_core/predictors/e3_selector.py:2742 computes
        probs = F.softmax(-scores / temperature, dim=0)
    and :2712 / :2747 record `scores` (POST-explore-term) and `probs`. Therefore for any pair
    (i, j) of candidates:
        log p_i - log p_j = -(s_i - s_j) / T   =>   T = (s_j - s_i) / (log p_i - log p_j)
    Taken on the TOP-2 pair: their probabilities are the largest, so this is the numerically
    best-conditioned pair available and the one least exposed to float underflow.

    This is the whole of the "instrument the lever's arrival" requirement for ARM_TEMP, and it
    needs NO substrate edit -- deliberately, since the 708a autopsy sets
    recommended_substrate_queue_entry.action = "none".

    Returns None when the second-largest probability has underflowed (the recovery is
    ill-conditioned precisely BECAUSE the distribution is saturated -- the caller counts that
    case rather than discarding it, since it is itself evidence for reading (i)).
    """
    srt, idx = torch.sort(probs.detach().reshape(-1), descending=True)
    p1 = float(srt[0].item())
    p2 = float(srt[1].item())
    if p2 <= TEMP_RECOVERY_MIN_P2 or p1 <= 0.0:
        return None
    denom = math.log(p1) - math.log(p2)
    if denom <= 1e-12:
        # p1 == p2: the pair carries no information about T (any T reproduces a tie).
        return None
    s = scores.detach().reshape(-1)
    num = float(s[int(idx[1].item())].item()) - float(s[int(idx[0].item())].item())
    t = num / denom
    if not math.isfinite(t) or t <= 0.0:
        return None
    return t


def _entropy_from_prob_dict(probs: Dict[int, float]) -> float:
    tot = sum(probs.values())
    if tot <= 0:
        return 0.0
    h = 0.0
    for p in probs.values():
        q = p / tot
        if q > 0:
            h -= q * math.log(q)
    return float(h)


# ---------------------------------------------------------------------------
# P1 bias-head REINFORCE training (verbatim from 707)
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
    # SECONDARY DV: PRE-COMMIT sampling-class marginal (sum of softmax class-mass over ticks).
    precommit_class_mass: Dict[int, float] = {}
    n_precommit_ticks = 0
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0

    # MECH-440 noise diagnostics (P2).
    noise_bias_ranges: List[float] = []
    raw_score_ranges: List[float] = []
    dacc_max_suppression = 0.0

    # CRF maturity readiness (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_minted_total_last = 0

    # Learning diagnostics.
    lcg_delta_ts: List[float] = []
    fcg_delta_ts: List[float] = []
    fcg_w_chan_finer_range_max = 0.0
    n_select_ticks = 0

    # --- 708a fresh-select telemetry (autopsy requirement 2) ---
    # n_fresh_select: P2 ticks where E3 select() genuinely ran (diagnostics repopulated).
    # n_latched:      P2 ticks where the held/stepped action was returned (no fresh select).
    # These make the replication factor VISIBLE in the manifest instead of inferred; in 708
    # the ratio was ~1.0 by construction (every env step recorded) and the true yield was
    # invisible. n_fresh_without_precommit is a defensive tripwire (see the read site).
    n_fresh_select = 0
    n_latched = 0
    n_fresh_without_precommit = 0

    # --- 708b PRE-COMMIT SHAPE + LEVER-ARRIVAL accumulators (the new readout) ---
    # All accumulated ONLY on verified-fresh selects, i.e. on exactly the same denominator as
    # the pre-commit class-entropy DV, so shape and entropy are directly comparable per cell.
    # CANDIDATE level: does the pre-commit softmax itself have any headroom?
    shape_max_prob: List[float] = []          # top-1 candidate mass
    shape_participation_ratio: List[float] = []   # 1 / sum(p^2): continuous effective support
    shape_eff_support: List[int] = []         # count of candidates with p >= mass floor
    shape_top1: List[float] = []
    shape_top2: List[float] = []
    shape_top3: List[float] = []
    shape_n_candidates: List[int] = []
    # CLASS level: the DV is a CLASS entropy, so candidate headroom can coexist with class
    # collapse (several candidates sharing one first-action class). Recorded separately so the
    # two cannot be conflated -- this is the discriminator 708a's single scalar could not carry.
    shape_max_class_mass: List[float] = []
    shape_n_distinct_classes: List[int] = []
    # LEVER ARRIVAL. Recovered applied temperature (see module docstring change 2): exact
    # inversion of probs = softmax(-scores / T) on the top-2 pair.
    lever_temp_recovered: List[float] = []
    lever_precommit_score_range: List[float] = []
    lever_logit_spread_over_t: List[float] = []
    n_temp_unrecoverable = 0                  # p2 underflowed -> itself a saturation signal
    n_explore_term_present = 0                # MECH-440 bias witnessed IN the pre-commit scores

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

            # ---------------- 708a FRESHNESS GUARD (the V3-EXQ-708 defect) ---------------- #
            # E3 select() does NOT run every env step: agent.py:5429 returns the held/stepped
            # action when ticks["e3_tick"] is False, BEFORE the e3.select() call at
            # agent.py:7011, on a 5-20-step MECH-093-modulated cadence (clock.py:52-70,
            # default 10 at config.py:2017). Both attributes LATCH across held ticks, so a
            # read without this clear re-records one selection once per held step. 708 did
            # exactly that (n_precommit_ticks == n_p2_ticks on all 24 arm-seeds) and the
            # replication was ARM-DEPENDENT (up to +58%), which distorted the between-arm
            # delta rather than merely inflating n.
            #
            # Both markers are set unconditionally inside select() -- last_score_diagnostics
            # is rebuilt at e3_selector.py:2452 and last_precommit_probs assigned at :2687,
            # with no early return between them -- so "repopulated" is a sound freshness test.
            # NOTE: a numel()==len(candidates) check is NOT a freshness guard (708's line 812):
            # the candidate count is constant, so a stale vector passes it.
            #
            # 708b ADDS .last_scores to the clear set. It latches identically (assigned at
            # e3_selector.py:2712, inside select() only) and is newly LOAD-BEARING here: it is
            # the `scores` half of the temperature recovery, so a stale vector paired with a
            # fresh probs vector would yield a fabricated temperature. Clearing it makes the
            # pairing provably same-tick.
            agent.e3.last_score_diagnostics = None
            agent.e3.last_precommit_probs = None
            agent.e3.last_scores = None
            agent.e3._last_explore_term = None

            action = agent.select_action(candidates, ticks)
            n_select_ticks += 1

            _diag_fresh = getattr(agent.e3, "last_score_diagnostics", None)
            if is_p2 and _diag_fresh is None:
                # Commitment latch held / non-E3 tick: NO fresh selection. Record NOTHING.
                n_latched += 1

            if is_p2 and _diag_fresh is not None:
                n_fresh_select += 1
                diag = _diag_fresh
                # MECH-440 noise non-vacuity: per-candidate bias range + raw-score range.
                nbr = float(diag.get("noisy_selection_bias_range", 0.0) or 0.0)
                if nbr > 0.0:
                    noise_bias_ranges.append(nbr)
                rsr = float(diag.get("e3_raw_score_range_mean", 0.0) or 0.0)
                if rsr > 0.0:
                    raw_score_ranges.append(rsr)
                # (ARC-110 loop diagnostics removed with ARM_NOISE_LOOPSEG -- see VENUE block.)
                # dACC non-vacuity: max perseveration suppression magnitude.
                bundle = getattr(agent, "_dacc_last_bundle", None)
                if bundle is not None:
                    supp = bundle.get("suppression", None)
                    if supp is not None:
                        try:
                            dacc_max_suppression = max(
                                dacc_max_suppression, float(supp.detach().abs().max().item())
                            )
                        except Exception:
                            pass
                # SECONDARY DV: pre-commit sampling-class marginal from last_precommit_probs.
                # 708a: reached ONLY on a verified-fresh select (guarded above), so each
                # probability vector contributes EXACTLY ONCE -- a tick sample, not a
                # hold-duration-weighted mixture. This is the readout whose invalidity in
                # V3-EXQ-708 forced this re-run.
                pp = getattr(agent.e3, "last_precommit_probs", None)
                if pp is None:
                    # Defensive + auditable: select() ran (diagnostics repopulated) but the
                    # pre-commit vector did not. No such path exists between :2452 and :2687
                    # today; counted rather than silently dropped so a future substrate change
                    # that introduces one is visible in the manifest instead of biasing the DV.
                    n_fresh_without_precommit += 1
                elif candidates is not None and len(candidates) >= 1:
                    try:
                        ppv = pp.detach().reshape(-1)
                        if ppv.numel() == len(candidates) and torch.isfinite(ppv).all():
                            n_precommit_ticks += 1
                            tick_class_mass: Dict[int, float] = {}
                            for ci, c in enumerate(candidates):
                                cls = _traj_first_action_class(c)
                                m = float(ppv[ci].item())
                                precommit_class_mass[cls] = (
                                    precommit_class_mass.get(cls, 0.0) + m
                                )
                                tick_class_mass[cls] = tick_class_mass.get(cls, 0.0) + m
                            # ---- 708b: PRE-COMMIT DISTRIBUTION SHAPE (the new readout) ----
                            # THE DISCRIMINATOR. Under reading (i) max_prob is near 1 in EVERY
                            # arm including A0_OFF (no headroom for any lever); under reading
                            # (ii) it differs BY ARM while the class entropy does not.
                            _sc = _shape_of(ppv)
                            shape_max_prob.append(_sc["max_prob"])
                            shape_participation_ratio.append(_sc["participation_ratio"])
                            shape_eff_support.append(_sc["eff_support"])
                            shape_top1.append(_sc["top1"])
                            shape_top2.append(_sc["top2"])
                            shape_top3.append(_sc["top3"])
                            shape_n_candidates.append(int(ppv.numel()))
                            # CLASS level: candidate headroom can coexist with class collapse.
                            if tick_class_mass:
                                shape_max_class_mass.append(float(max(tick_class_mass.values())))
                                shape_n_distinct_classes.append(int(len(tick_class_mass)))
                            # ---- 708b: LEVER ARRIVAL at the pre-commit softmax ----
                            # Exact inversion of e3_selector.py:2742
                            #   probs = F.softmax(-scores / temperature, dim=0)
                            # against :2712 self.last_scores (POST-explore-term, same tick --
                            # both were cleared before this select_action). Taken on the top-2
                            # pair for numerical stability. A recovered T on ARM_TEMP strictly
                            # above A0_OFF's is DIRECT evidence the MECH-313 lift arrived AT the
                            # pre-commit softmax -- exactly what reading (ii) denies.
                            _sv = getattr(agent.e3, "last_scores", None)
                            if _sv is not None:
                                _svv = _sv.detach().reshape(-1)
                                if (
                                    _svv.numel() == ppv.numel()
                                    and _svv.numel() >= 2
                                    and torch.isfinite(_svv).all()
                                ):
                                    lever_precommit_score_range.append(
                                        float((_svv.max() - _svv.min()).item())
                                    )
                                    _t = _recover_temperature(_svv, ppv)
                                    if _t is None:
                                        # p2 underflowed: the recovery is unavailable BECAUSE
                                        # the distribution is saturated. Counted, not dropped --
                                        # it is itself evidence for reading (i).
                                        n_temp_unrecoverable += 1
                                    else:
                                        lever_temp_recovered.append(_t)
                                        lever_logit_spread_over_t.append(
                                            float((_svv.max() - _svv.min()).item()) / _t
                                        )
                            # Positive per-tick witness that the MECH-440 perturbation entered
                            # the score vector the softmax normalises: e3_selector.py:2698 sets
                            # _last_explore_term in the same branch that does
                            # `scores = scores + _explore_term`, and both precede :2742. Cleared
                            # before select_action, so non-None means THIS tick.
                            if getattr(agent.e3, "_last_explore_term", None) is not None:
                                n_explore_term_present += 1
                    except Exception:
                        pass

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
                resid_metrics = agent.update_residue(
                    harm_signal=float(_harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )
            ldt = resid_metrics.get("e3_lcg_delta_t")
            if ldt is not None:
                lcg_delta_ts.append(float(ldt.item()))
            fdt = resid_metrics.get("e3_fcg_delta_t")
            if fdt is not None:
                fcg_delta_ts.append(float(fdt.item()))
            fwr = resid_metrics.get("e3_fcg_w_chan_finer_range")
            if fwr is not None:
                fcg_w_chan_finer_range_max = max(
                    fcg_w_chan_finer_range_max, float(fwr.item())
                )

            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure, drive_level=drive_level,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            tick_in_ep += 1
            if done:
                break

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

    committed_class_entropy = _entropy_from_int_counts(committed_class_counts)
    precommit_class_entropy = _entropy_from_prob_dict(precommit_class_mass)

    frac_pre_ge2 = float(n_p2_pre_ge2 / n_p2_ticks) if n_p2_ticks > 0 else 0.0
    consumed_spread_mean = (
        float(sum(consumed_dists) / len(consumed_dists)) if consumed_dists else 0.0
    )

    if crf_n_active_per_tick:
        frac_crf_active_ge_floor = float(
            sum(1 for n in crf_n_active_per_tick if n >= 1) / len(crf_n_active_per_tick)
        )
    else:
        frac_crf_active_ge_floor = 0.0

    lcg_delta_t_std = float(statistics.pstdev(lcg_delta_ts)) if len(lcg_delta_ts) >= 2 else 0.0
    fcg_delta_t_std = float(statistics.pstdev(fcg_delta_ts)) if len(fcg_delta_ts) >= 2 else 0.0

    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )
    fcg_moved = bool(fcg_w_chan_finer_range_max > W_CHAN_FINER_RANGE_FLOOR)
    fcg_delta_nonflat = bool(fcg_delta_t_std > DELTA_T_STD_FLOOR)

    noise_bias_range_mean = (
        float(sum(noise_bias_ranges) / len(noise_bias_ranges)) if noise_bias_ranges else 0.0
    )
    raw_score_range_mean = (
        float(sum(raw_score_ranges) / len(raw_score_ranges)) if raw_score_ranges else 0.0
    )
    noise_to_raw_frac = (
        noise_bias_range_mean / raw_score_range_mean if raw_score_range_mean > 1e-9 else 0.0
    )

    # 708a: the honest yield of genuine E3 selections over P2 env steps. In 708 this was
    # 1.0 by construction (the defect); here it should land near 1/e3_steps_per_tick.
    fresh_select_yield = (
        float(n_fresh_select / n_p2_ticks) if n_p2_ticks > 0 else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "noise_head": bool(arm["noise_head"]),
        "temp": bool(arm["temp"]),
        "loop_seg": bool(arm["loop_seg"]),
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
        # ----- PRIMARY + SECONDARY DV -----
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "precommit_class_entropy_nats": round(precommit_class_entropy, 6),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        "n_precommit_ticks": int(n_precommit_ticks),
        # ----- 708a fresh-select telemetry (autopsy requirement 2 + 4) -----
        # Recorded PER ARM-SEED in the manifest ITSELF, deliberately NOT via a declared
        # per_tick_sink: Phase 3 cloud workers POST only manifest_bytes, so a declared sink
        # is not transported (twice-confirmed, V3-EXQ-785 and V3-EXQ-708).
        "n_fresh_select": int(n_fresh_select),
        "n_latched": int(n_latched),
        "fresh_select_yield": round(fresh_select_yield, 6),
        "n_fresh_without_precommit": int(n_fresh_without_precommit),
        "fresh_selects_sufficient": bool(n_fresh_select >= MIN_FRESH_SELECTS),
        # ----- 708b PRE-COMMIT DISTRIBUTION SHAPE (per arm per seed, IN THE MANIFEST) -----
        # The discriminator between reading (i) H-PRECOMMIT-SATURATED and reading (ii)
        # H-LEVER-DOWNSTREAM. Recorded here rather than via a declared per_tick_sink because
        # Phase 3 cloud workers POST only manifest_bytes (twice-confirmed: 785, 708).
        # min/max as well as the mean: a mean max_prob of 0.9 is a different object when the
        # per-tick minimum is 0.9 (uniformly saturated) than when it is 0.3 (bimodal).
        "n_shape_samples": int(len(shape_max_prob)),
        "precommit_max_prob_mean": round(_mean_or0(shape_max_prob), 6),
        "precommit_max_prob_min": round(min(shape_max_prob), 6) if shape_max_prob else 0.0,
        "precommit_max_prob_max": round(max(shape_max_prob), 6) if shape_max_prob else 0.0,
        "precommit_participation_ratio_mean": round(_mean_or0(shape_participation_ratio), 6),
        "precommit_eff_support_mean": round(
            _mean_or0([float(v) for v in shape_eff_support]), 6
        ),
        "precommit_top1_mean": round(_mean_or0(shape_top1), 6),
        "precommit_top2_mean": round(_mean_or0(shape_top2), 6),
        "precommit_top3_mean": round(_mean_or0(shape_top3), 6),
        "precommit_n_candidates_mean": round(
            _mean_or0([float(v) for v in shape_n_candidates]), 6
        ),
        # CLASS-level shape: the DV aggregates candidates by first-action class, so a
        # candidate-level distribution with headroom can still yield a collapsed class
        # marginal. Kept separate so the two are never conflated.
        "precommit_max_class_mass_mean": round(_mean_or0(shape_max_class_mass), 6),
        "precommit_n_distinct_classes_mean": round(
            _mean_or0([float(v) for v in shape_n_distinct_classes]), 6
        ),
        # ----- 708b LEVER ARRIVAL at the pre-commit softmax -----
        "precommit_temperature_recovered_mean": round(_mean_or0(lever_temp_recovered), 6),
        "n_temperature_recovered": int(len(lever_temp_recovered)),
        "n_temperature_unrecoverable": int(n_temp_unrecoverable),
        "precommit_score_range_mean": round(_mean_or0(lever_precommit_score_range), 8),
        # score_range / T: the logit spread the softmax actually sees. This is the direct,
        # units-free statement of the headroom argument -- a large spread over a small T
        # saturates the distribution no matter which arm produced it.
        "precommit_logit_spread_over_t_mean": round(
            _mean_or0(lever_logit_spread_over_t), 6
        ),
        "n_explore_term_present": int(n_explore_term_present),
        "explore_term_present_frac": round(
            float(n_explore_term_present / n_fresh_select) if n_fresh_select > 0 else 0.0, 6
        ),
        "shape_samples_sufficient": bool(len(shape_max_prob) >= MIN_SHAPE_SAMPLES),
        # ----- MECH-440 noise non-vacuity -----
        "noise_bias_range_mean": round(noise_bias_range_mean, 8),
        "raw_score_range_mean": round(raw_score_range_mean, 8),
        "noise_to_raw_range_frac": round(noise_to_raw_frac, 6),
        "dacc_max_suppression": round(dacc_max_suppression, 8),
        # ----- Readiness / non-vacuity -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "lcg_delta_t_std": round(lcg_delta_t_std, 8),
        "fcg_delta_t_std": round(fcg_delta_t_std, 8),
        "fcg_w_chan_finer_range_max": round(fcg_w_chan_finer_range_max, 8),
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
            f"Arm {arm['arm_id']} ({arm['label']}) noise_head={arm['noise_head']} "
            f"temp={arm['temp']} loop_seg={arm['loop_seg']} "
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
            # Per-cell fingerprint. A0_OFF is a stable single-arena baseline (self-mint
            # eligible). The exploration arms ride the just-built MECH-440 / ARC-110
            # substrate (in flux for this lineage) -- not reusable baselines.
            if arm["arm_id"] == "A0_OFF":
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(
                        arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                    ),
                    seed=s, script_path=script_path,
                    rng_fully_reset=True, config_slice_declared=True,
                    include_driver_script_in_hash=False,
                )
            else:
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(
                        arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                    ),
                    seed=s, script_path=script_path,
                    rng_fully_reset=True,
                    extra_ineligible_reasons=[
                        "mech440_noise_substrate_just_built_in_flux_not_a_reusable_baseline",
                    ],
                )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    off_rows = _arm_rows(arm_results, "A0_OFF")
    temp_rows = _arm_rows(arm_results, "ARM_TEMP")
    nsingle_rows = _arm_rows(arm_results, "ARM_NOISE_SINGLE")
    all_rows = off_rows + temp_rows + nsingle_rows

    def _maj(rows: List[Dict[str, Any]], pred) -> bool:
        return sum(1 for r in rows if pred(r)) >= MIN_SEEDS_FOR_PASS

    off_ent = _by_seed(off_rows, "committed_class_entropy_nats")
    temp_ent = _by_seed(temp_rows, "committed_class_entropy_nats")
    nsingle_ent = _by_seed(nsingle_rows, "committed_class_entropy_nats")

    off_pre = _by_seed(off_rows, "precommit_class_entropy_nats")
    temp_pre = _by_seed(temp_rows, "precommit_class_entropy_nats")
    nsingle_pre = _by_seed(nsingle_rows, "precommit_class_entropy_nats")

    off_gap = _gap_by_seed(off_rows)
    temp_gap = _gap_by_seed(temp_rows)
    nsingle_gap = _gap_by_seed(nsingle_rows)

    # Divergent seeds: pool divergent on the C1 comparison arms (OFF + TEMP + NOISE_SINGLE).
    primary_div = [
        s for s in sorted(set(off_gap) & set(temp_gap) & set(nsingle_gap))
        if off_gap.get(s) and temp_gap.get(s) and nsingle_gap.get(s)
    ]
    n_primary_div = len(primary_div)
    enough_divergent = n_primary_div >= MIN_DIVERGENT_SEEDS

    # ----- Precondition: matched-pre-commit-variance is NON-VACUOUS -- BOTH ARM_TEMP and
    # ARM_NOISE_SINGLE genuinely RAISE pre-commit entropy strict-above A0_OFF on a
    # strict-majority of divergent seeds. A temperature control that doesn't lift pre-commit
    # (or a weight noise that injects no pre-commit variance) makes the carve test meaningless. -----
    n_temp_pre_lift = sum(
        1 for s in primary_div if temp_pre.get(s, 0.0) > off_pre.get(s, 0.0) + PRECOMMIT_LIFT_MARGIN
    )
    n_noise_pre_lift = sum(
        1 for s in primary_div if nsingle_pre.get(s, 0.0) > off_pre.get(s, 0.0) + PRECOMMIT_LIFT_MARGIN
    )
    temp_lifts_precommit = bool(enough_divergent and _div_pass(n_temp_pre_lift, n_primary_div))
    noise_lifts_precommit = bool(enough_divergent and _div_pass(n_noise_pre_lift, n_primary_div))

    # ----- Precondition: the NOISE arms inject a SUPRA-FLOOR per-candidate bias range AND it
    # is a non-trivial fraction of the raw-score range (can flip the within-eligible argmin). -----
    noise_bias_supra_floor = _maj(
        nsingle_rows,
        lambda r: r.get("noise_bias_range_mean", 0.0) > NOISE_BIAS_RANGE_FLOOR
        and r.get("noise_to_raw_range_frac", 0.0) > NOISE_BIAS_TO_RAW_RANGE_FRAC_FLOOR,
    )

    # ----- Precondition: dACC non-vacuity (Go/No-Go perseveration axis live) on a majority
    # of seeds across the dACC-bearing arms (all arms use dACC). -----
    dacc_live = all(
        _maj(rows, lambda r: r.get("dacc_max_suppression", 0.0) > DACC_MAX_SUPPRESSION_FLOOR)
        for rows in (off_rows, temp_rows, nsingle_rows) if rows
    )

    # ----- Precondition: learning engaged (finer channels dissociable + delta_t nonflat). -----
    fcg_moved_ok = _maj(nsingle_rows, lambda r: r.get("fcg_moved", False))
    fcg_delta_nonflat_ok = _maj(nsingle_rows, lambda r: r.get("fcg_delta_nonflat", False))

    # ----- Precondition (708a, NEW): FRESH-SELECT SUFFICIENCY. The repaired DV accumulates
    # only on verified-fresh E3 selections, so its honest denominator is ~1/e3_steps_per_tick
    # of 708's. Every arm-seed must clear MIN_FRESH_SELECTS; an underpowered DV must route
    # substrate_not_ready_requeue, NEVER a weakens. Worst cell (not the mean) is the measured
    # quantity, so the recompute matches the all(...) quantifier this met expression uses. -----
    _fresh_counts = [(int(r.get("n_fresh_select", 0)), r["arm_id"], r["seed"]) for r in all_rows]
    _worst_fresh = min(_fresh_counts) if _fresh_counts else (0, "none", -1)
    fresh_selects_sufficient = bool(
        _fresh_counts and all(n >= MIN_FRESH_SELECTS for n, _a, _s in _fresh_counts)
    )

    # ----- Exposure imbalance vs A0_OFF (autopsy requirement 3): REPORTED, NEVER GATING.
    # This is the mechanism by which 708's per-env-step read distorted the between-arm delta
    # (OFF and TEMP tick-identical while NOISE_SINGLE diverged up to +57.5%). Under the
    # freshness guard the DV no longer inherits it, but it is recorded per arm-seed so a
    # reader can verify that directly rather than take it on trust. -----
    _off_fresh_by_seed = {r["seed"]: int(r.get("n_fresh_select", 0)) for r in off_rows}
    exposure_imbalance: List[Dict[str, Any]] = []
    for r in all_rows:
        base = _off_fresh_by_seed.get(r["seed"], 0)
        n_f = int(r.get("n_fresh_select", 0))
        frac = float((n_f - base) / base) if base > 0 else 0.0
        exposure_imbalance.append({
            "arm_id": r["arm_id"],
            "seed": int(r["seed"]),
            "n_fresh_select": n_f,
            "n_latched": int(r.get("n_latched", 0)),
            "n_p2_ticks": int(r.get("n_p2_ticks", 0)),
            "fresh_select_yield": r.get("fresh_select_yield", 0.0),
            "a0_off_n_fresh_select": base,
            "exposure_imbalance_vs_off": round(frac, 6),
            "beyond_report_floor": bool(abs(frac) > EXPOSURE_IMBALANCE_REPORT_FLOOR),
        })
    max_exposure_imbalance = max(
        (abs(e["exposure_imbalance_vs_off"]) for e in exposure_imbalance), default=0.0
    )

    # ----- Precondition (708b, NEW): SHAPE-SAMPLE SUFFICIENCY. The shape moments ride the same
    # verified-fresh denominator as the pre-commit DV; below the floor the distribution's shape
    # cannot be estimated and the discrimination is not attemptable => requeue, never a verdict.
    _shape_counts = [(int(r.get("n_shape_samples", 0)), r["arm_id"], r["seed"]) for r in all_rows]
    _worst_shape = min(_shape_counts) if _shape_counts else (0, "none", -1)
    shape_samples_sufficient = bool(
        _shape_counts and all(n >= MIN_SHAPE_SAMPLES for n, _a, _s in _shape_counts)
    )

    # ================= 708b: THE DISCRIMINATION (the load-bearing readout) =================
    # 708a established that neither lever moves the DV on a demonstrably-working instrument.
    # Two readings survive that observation and this block separates them. Everything below is
    # computed from the SHAPE moments and the RECOVERED applied temperature -- quantities that
    # exist in NO prior manifest, which is why this run exists.
    _arm_rowsets = {
        "A0_OFF": off_rows, "ARM_TEMP": temp_rows, "ARM_NOISE_SINGLE": nsingle_rows,
    }
    shape_by_arm: Dict[str, Dict[str, float]] = {}
    for _aid, _rows in _arm_rowsets.items():
        shape_by_arm[_aid] = {
            "n_seeds": float(len(_rows)),
            "max_prob_mean": _mean([r["precommit_max_prob_mean"] for r in _rows]),
            "max_prob_min_over_seeds": (
                min(r["precommit_max_prob_min"] for r in _rows) if _rows else 0.0
            ),
            "participation_ratio_mean": _mean(
                [r["precommit_participation_ratio_mean"] for r in _rows]
            ),
            "eff_support_mean": _mean([r["precommit_eff_support_mean"] for r in _rows]),
            "n_candidates_mean": _mean([r["precommit_n_candidates_mean"] for r in _rows]),
            "max_class_mass_mean": _mean(
                [r["precommit_max_class_mass_mean"] for r in _rows]
            ),
            "n_distinct_classes_mean": _mean(
                [r["precommit_n_distinct_classes_mean"] for r in _rows]
            ),
            "temperature_recovered_mean": _mean(
                [r["precommit_temperature_recovered_mean"] for r in _rows
                 if r["n_temperature_recovered"] > 0]
            ),
            "logit_spread_over_t_mean": _mean(
                [r["precommit_logit_spread_over_t_mean"] for r in _rows]
            ),
            "explore_term_present_frac_mean": _mean(
                [r["explore_term_present_frac"] for r in _rows]
            ),
            "frac_temperature_unrecoverable": _mean([
                float(r["n_temperature_unrecoverable"])
                / max(1.0, float(r["n_temperature_unrecoverable"] + r["n_temperature_recovered"]))
                for r in _rows
            ]),
        }

    _off_shape = shape_by_arm["A0_OFF"]

    # --- Reading (i) H-PRECOMMIT-SATURATED: no headroom ANYWHERE, the OFF baseline included.
    # The autopsy states the discriminator exactly: "under (i) max_prob is near 1 in EVERY arm
    # including A0_OFF". A0_OFF is the load-bearing cell -- an OFF arm at max_prob ~ 1 has a
    # distribution no lever could have moved, whatever the lever does.
    per_arm_saturated = {
        aid: bool(sh["max_prob_mean"] >= PRECOMMIT_SATURATION_MAX_PROB)
        for aid, sh in shape_by_arm.items() if sh["n_seeds"] > 0
    }
    all_arms_saturated = bool(per_arm_saturated) and all(per_arm_saturated.values())
    off_arm_saturated = bool(per_arm_saturated.get("A0_OFF", False))

    # --- Reading (ii) H-LEVER-DOWNSTREAM: did each lever actually ARRIVE at the pre-commit
    # softmax? ARM_TEMP arrives iff the RECOVERED applied temperature is lifted above A0_OFF's
    # (the softmax at e3_selector.py:2742 divides by exactly this T). ARM_NOISE_SINGLE arrives
    # iff the explore term is witnessed in the score vector that same softmax normalises.
    _off_t = _off_shape["temperature_recovered_mean"]
    _temp_t = shape_by_arm["ARM_TEMP"]["temperature_recovered_mean"]
    temp_arrival_rel_lift = (
        float((_temp_t - _off_t) / _off_t) if _off_t > 1e-12 else 0.0
    )
    temp_lever_arrives = bool(
        _off_t > 1e-12 and _temp_t > 1e-12
        and temp_arrival_rel_lift >= TEMP_ARRIVAL_REL_FLOOR
    )
    noise_explore_frac = shape_by_arm["ARM_NOISE_SINGLE"]["explore_term_present_frac_mean"]
    noise_lever_arrives = bool(noise_explore_frac >= EXPLORE_TERM_ARRIVAL_FRAC_FLOOR)
    both_levers_arrive = bool(temp_lever_arrives and noise_lever_arrives)

    # --- The second half of the autopsy's discriminator: "under (ii) max_prob DIFFERS by arm
    # while entropy does not". A lever that reaches the distribution's SHAPE but not its class
    # marginal is a distinct state from one that never reaches the distribution at all, and
    # this is what separates them.
    maxprob_delta_by_arm = {
        aid: round(sh["max_prob_mean"] - _off_shape["max_prob_mean"], 6)
        for aid, sh in shape_by_arm.items() if aid != "A0_OFF" and sh["n_seeds"] > 0
    }
    maxprob_differs_by_arm = bool(
        maxprob_delta_by_arm
        and max(abs(v) for v in maxprob_delta_by_arm.values()) >= MAXPROB_ARM_DELTA_FLOOR
    )

    preconditions_met = bool(
        enough_divergent
        and noise_bias_supra_floor
        and dacc_live
        and fcg_moved_ok and fcg_delta_nonflat_ok
        and fresh_selects_sufficient
        and shape_samples_sufficient
    )
    # NOTE (708b change 3): temp_lifts_precommit / noise_lifts_precommit are DELIBERATELY absent
    # from preconditions_met. They are MANIPULATION CHECKS, not readiness gates: a null there is
    # this run's RESULT, not a statement that the substrate was unready. They are still computed
    # above, still reported, and still carried with honest measured/threshold/met -- but under
    # interpretation.recorded_preconditions[], which the REE_assembly indexer surfaces without
    # adjudicating. Carrying them in the flat adjudicating list is what filed 708a's working
    # instrument as "substrate not ready". See the module docstring, change 3.
    manipulation_checks_pass = bool(temp_lifts_precommit and noise_lifts_precommit)

    # ----- C1 (MECH-440 single-arena propagation): ARM_NOISE_SINGLE committed-class entropy
    # strict-above ARM_TEMP (the matched-pre-commit-variance temperature control) on a
    # strict-majority of divergent seeds. The weight noise CARVES where the temperature THRASHES. -----
    c1_seeds = [
        s for s in primary_div
        if nsingle_ent.get(s, 0.0) > temp_ent.get(s, 0.0) + CONVERSION_MARGIN
    ]
    c1_holds = _div_pass(len(c1_seeds), n_primary_div)

    # (708a: the ARM_NOISE_LOOPSEG disambiguation branch is REMOVED. V3-EXQ-707b answered the
    # single-arena-artefact sub-hypothesis with a valid non_degenerate null, concluding the
    # conversion ceiling is intrinsic to MECH-439 F-dominance. Offering a route-to-ARC-110
    # outcome here would let this run self-route into a settled question.)

    # ----- Outcome map (708b) -----
    # The LOAD-BEARING criterion is the DISCRIMINATION -- (i) vs (ii) -- not MECH-440
    # propagation. MECH-440's propagation question is only reachable if the manipulation
    # actually fires this time; 708a says it does not, so that branch is retained but is not
    # what this run is powered for. Every non-supports branch leaves MECH-440
    # `non_contributory`: with no injected pre-commit variance there is nothing for the
    # propagation question to observe, which is exactly the 708a autopsy's reading.
    if not preconditions_met:
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "The (i)-vs-(ii) discrimination could NOT be attempted: a READINESS precondition is "
            "unmet (too few divergent seeds / noise bias range below floor or trivial vs raw "
            "range / dACC suppression flat / finer channels not dissociable / delta_t flat / too "
            "few genuinely-fresh E3 selections / too few recoverable pre-commit shape samples). "
            "NOT a falsification. NOTE: the two MANIPULATION CHECKS "
            "(temperature_control_raises_precommit_entropy, weight_noise_raises_precommit_entropy) "
            "can no longer reach this branch -- their null is a RESULT, not an unready substrate."
        )
        per_claim = {"MECH-440": "non_contributory"}
    elif manipulation_checks_pass and c1_holds:
        # Retained from 708a and reachable only if the manipulation NOW fires (708a: 0/4 and
        # 1/4 against a bar of 2). If it does, the propagation question is genuinely asked.
        outcome = "PASS"
        overall_direction = "supports"
        non_degenerate = True
        degeneracy_reason = ""
        label = "weight_noise_carves_committed_diversity_above_matched_temperature_supports_mech440"
        per_claim = {"MECH-440": "supports"}
    elif both_levers_arrive and all_arms_saturated:
        # READING (i) H-PRECOMMIT-SATURATED. Both levers demonstrably reach the pre-commit
        # softmax, yet every arm INCLUDING the OFF baseline sits at max_prob >= the saturation
        # threshold. The distribution has no headroom for any lever to exploit. This is the
        # MECH-439 F-dominance signature observed PRE-commit.
        outcome = "PASS"
        overall_direction = "non_contributory"
        non_degenerate = True
        degeneracy_reason = ""
        label = "precommit_distribution_saturated_no_headroom_levers_arrive"
        per_claim = {"MECH-440": "non_contributory"}
    elif not both_levers_arrive:
        # READING (ii) H-LEVER-DOWNSTREAM. At least one lever does NOT arrive at the pre-commit
        # softmax: ARM_TEMP's recovered applied temperature is not lifted above A0_OFF's,
        # and/or the MECH-440 explore term is not witnessed in the score vector that softmax
        # normalises. A wiring gap, not a substrate property.
        outcome = "PASS"
        overall_direction = "non_contributory"
        non_degenerate = True
        degeneracy_reason = ""
        label = "precommit_lever_applied_downstream_does_not_reach_precommit_softmax"
        per_claim = {"MECH-440": "non_contributory"}
    else:
        # NEITHER reading. The levers arrive AND the distribution has headroom, yet neither the
        # entropy nor the class marginal moves. This is a genuinely new state and the reason the
        # candidate-vs-class shape split is recorded separately: the most likely resolution is
        # that candidate-level headroom coexists with class-level collapse (several candidates
        # sharing one first-action class), which the retained shape moments can show directly.
        outcome = "FAIL"
        overall_direction = "non_contributory"
        non_degenerate = True
        degeneracy_reason = ""
        label = "precommit_shape_headroom_unexplained"
        per_claim = {"MECH-440": "non_contributory"}
    discrimination_achieved = bool(label in (
        "precommit_distribution_saturated_no_headroom_levers_arrive",
        "precommit_lever_applied_downstream_does_not_reach_precommit_softmax",
        "weight_noise_carves_committed_diversity_above_matched_temperature_supports_mech440",
    ))

    off_mean_dv = _mean([r["committed_class_entropy_nats"] for r in off_rows])
    temp_mean_dv = _mean([r["committed_class_entropy_nats"] for r in temp_rows])
    nsingle_mean_dv = _mean([r["committed_class_entropy_nats"] for r in nsingle_rows])

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "enough_divergent_seeds",
                "kind": "readiness",
                "description": (
                    "seeds whose candidate pool is DIVERGENT on ALL C1 comparison arms "
                    "(A0_OFF + ARM_TEMP + ARM_NOISE_SINGLE) >= MIN_DIVERGENT_SEEDS. Too few "
                    "=> substrate_not_ready_requeue."
                ),
                "control": "consumed cand_world_summary pairwise spread > floor (GAP-A); per-seed",
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                "direction": "lower",  # FLOOR: met iff measured >= threshold
                "met": bool(enough_divergent),
            },
            {
                "name": "noise_bias_range_supra_floor_vs_raw",
                "kind": "readiness",
                "description": (
                    "ARM_NOISE_SINGLE per-candidate noise bias range > floor AND a non-trivial "
                    "fraction of the raw-score range (so the noise CAN flip the within-eligible "
                    "argmin). A noise too small to ever compete is a vacuous test => requeue. "
                    "measured = min noise_bias_range_mean across ARM_NOISE_SINGLE seeds."
                ),
                "control": "ARM_NOISE_SINGLE noisy_selection_bias_range / e3_raw_score_range",
                "measured": float(min([r.get("noise_bias_range_mean", 0.0) for r in nsingle_rows] or [0.0])),
                "threshold": float(NOISE_BIAS_RANGE_FLOOR),
                "direction": "lower",  # FLOOR: met iff measured >= threshold
                "met": bool(noise_bias_supra_floor),
            },
            {
                "name": "dacc_suppression_live",
                "kind": "readiness",
                "description": (
                    "dACC perseveration No-Go axis is LIVE (dacc_max_suppression > 0) on a "
                    "majority of seeds across the dACC-bearing arms. A flat dACC means the "
                    "Go/No-Go constitution is inert => the SOTA stack is not actually engaged."
                ),
                "control": "agent._dacc_last_bundle['suppression'] max over P2",
                "measured": float(min([r.get("dacc_max_suppression", 0.0) for r in nsingle_rows] or [0.0])),
                "threshold": float(DACC_MAX_SUPPRESSION_FLOOR),
                "direction": "lower",  # FLOOR: met iff measured >= threshold
                "met": bool(dacc_live),
            },
            {
                "name": "fresh_selects_sufficient",
                "kind": "readiness",
                "description": (
                    "EVERY arm-seed recorded at least MIN_FRESH_SELECTS genuinely fresh E3 "
                    "selections in P2. 708a accumulates the pre-commit DV ONLY on verified-fresh "
                    "ticks (agent.e3.last_score_diagnostics / .last_precommit_probs cleared to "
                    "None before every select_action and required to repopulate), so the honest "
                    "denominator is ~1/e3_steps_per_tick of V3-EXQ-708's, whose per-env-step "
                    "read gave n_precommit_ticks == n_p2_ticks on all 24 arm-seeds. An honest-"
                    "but-underpowered DV must route substrate_not_ready_requeue, NEVER a weakens. "
                    "measured = the WORST cell (min across arm-seeds), matching the all(...) "
                    "quantifier this met expression uses, so the indexer's recompute agrees."
                ),
                "control": (
                    f"worst arm-seed n_fresh_select over P2; offending cell "
                    f"arm={_worst_fresh[1]} seed={_worst_fresh[2]}"
                ),
                "offending_cell": f"{_worst_fresh[1]}:seed{_worst_fresh[2]}",
                "measured": float(_worst_fresh[0]),
                "threshold": float(MIN_FRESH_SELECTS),
                "direction": "lower",  # FLOOR: met iff measured >= threshold
                "met": bool(fresh_selects_sufficient),
            },
            {
                "name": "learning_engaged_finer_channels_dissociable",
                "kind": "readiness",
                "description": (
                    "on ARM_NOISE_SINGLE the finer w_chan_finer entries MOVED + the signed-RPE "
                    "delta_t carries cross-tick variance, on a majority of seeds. Below floor => "
                    "substrate_not_ready_requeue."
                ),
                "control": "ARM_NOISE_SINGLE fcg_w_chan_finer_range_max + fcg_delta_t_std",
                "measured": float(min([r["fcg_w_chan_finer_range_max"] for r in nsingle_rows] or [0.0])),
                "threshold": float(W_CHAN_FINER_RANGE_FLOOR),
                "direction": "lower",  # FLOOR: met iff measured >= threshold
                "met": bool(fcg_moved_ok and fcg_delta_nonflat_ok),
            },
            {
                "name": "shape_samples_sufficient",
                "kind": "readiness",
                "description": (
                    "708b NEW. EVERY arm-seed recovered at least MIN_SHAPE_SAMPLES usable "
                    "pre-commit distributions (a finite probability vector matching the "
                    "candidate count, on a verified-FRESH select). The shape moments are this "
                    "run's load-bearing readout, so below the floor the (i)-vs-(ii) "
                    "discrimination cannot be attempted and the correct route is "
                    "substrate_not_ready_requeue, NEVER a verdict on an unestimated shape. "
                    "measured = the WORST cell (min across arm-seeds), matching the all(...) "
                    "quantifier this met expression uses, so the indexer's recompute agrees."
                ),
                "control": (
                    f"worst arm-seed n_shape_samples over P2; offending cell "
                    f"arm={_worst_shape[1]} seed={_worst_shape[2]}"
                ),
                "offending_cell": f"{_worst_shape[1]}:seed{_worst_shape[2]}",
                "measured": float(_worst_shape[0]),
                "threshold": float(MIN_SHAPE_SAMPLES),
                "direction": "lower",  # FLOOR: met iff measured >= threshold
                "met": bool(shape_samples_sufficient),
            },
        ],
        # ================= 708b change 3: MANIPULATION CHECKS, RECORDED NOT GATING =========
        # These two are the entries whose null IS this run's subject matter. In V3-EXQ-708a they
        # sat in the flat adjudicating `preconditions` list as kind: readiness, so
        # build_experiment_indexes._compute_adjudication -- which reads that list ARM-BLIND and
        # returns whole-run `precondition_unmet` on the first unmet entry -- filed a
        # demonstrably-working instrument (862 fresh selects vs a floor of 30) as "substrate not
        # ready", and the self-route was `substrate_not_ready_requeue`. That buried the finding.
        #
        # The schema has no `manipulation_check` kind, so they are carried here under
        # `recorded_preconditions` -- the auditable-but-non-adjudicating sibling list, the
        # V3-EXQ-737 pattern -- with `kind: "manipulation_check"` recorded descriptively. They
        # keep honest measured / threshold / met, so any recompute agrees with the run's own
        # verdict; they are SURFACED for governance, and they do not gate, do not set
        # substrate_not_ready_requeue, and do not falsify non_degenerate.
        "preconditions_scope_note": (
            "The two entries temperature_control_raises_precommit_entropy and "
            "weight_noise_raises_precommit_entropy are carried under 'recorded_preconditions', "
            "NOT under the adjudicating flat 'preconditions' list. They are MANIPULATION CHECKS, "
            "not readiness gates: they ask whether the designed lever MOVED the dependent "
            "variable, and a null there is a RESULT ABOUT THE SYSTEM, not a statement that the "
            "setup was inadequate. V3-EXQ-708a carried them as kind: readiness, so its null "
            "(ARM_TEMP 0/4 divergent seeds, ARM_NOISE_SINGLE 1/4, bar 2) self-routed "
            "substrate_not_ready_requeue and the REE_assembly indexer flagged the whole run "
            "precondition_unmet -- on an instrument its OWN numbers prove was working (862 "
            "genuinely-fresh E3 select() calls against a floor of 30, 4 divergent seeds, "
            "supra-floor noise bias range 0.221, live dACC suppression, dissociable finer "
            "channels). Adjudicated 2026-07-22 (failure_autopsy_V3-EXQ-708a_2026-07-22, "
            "user-confirmed): record as a substantive finding. The schema has no "
            "manipulation_check kind; kind is recorded descriptively on each entry. The genuine "
            "READINESS gates -- divergent seeds, noise bias range, dACC, learning engaged, fresh "
            "selects, shape samples -- remain in the adjudicating list and DO gate."
        ),
        "recorded_preconditions": [
            {
                "name": "temperature_control_raises_precommit_entropy",
                "kind": "manipulation_check",   # NOT readiness -- see preconditions_scope_note
                "description": (
                    "MANIPULATION CHECK (recorded, non-gating). Does ARM_TEMP (the V3-EXQ-687 "
                    "temperature control) raise PRE-COMMIT sampling-class entropy strict-above "
                    "A0_OFF by margin on a strict-majority of divergent seeds? V3-EXQ-708a "
                    "measured 0 of 4 against a bar of 2. A null here is this run's SUBJECT, and "
                    "the shape moments + the recovered applied temperature are what explain it. "
                    "measured = n divergent seeds on which ARM_TEMP lifts pre-commit entropy."
                ),
                "control": "ARM_TEMP precommit_class_entropy vs A0_OFF, divergent seeds, paired",
                "measured": float(n_temp_pre_lift),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "direction": "lower",  # FLOOR: met iff measured >= threshold
                "met": bool(temp_lifts_precommit),
            },
            {
                "name": "weight_noise_raises_precommit_entropy",
                "kind": "manipulation_check",   # NOT readiness -- see preconditions_scope_note
                "description": (
                    "MANIPULATION CHECK (recorded, non-gating). Does ARM_NOISE_SINGLE's "
                    "factorised-Gaussian selection-head weight noise raise PRE-COMMIT "
                    "sampling-class entropy strict-above A0_OFF by margin on a strict-majority "
                    "of divergent seeds? V3-EXQ-708a measured 1 of 4 against a bar of 2, WITH a "
                    "supra-floor bias range (0.221) -- i.e. the perturbation was injected and "
                    "the entropy still did not move, which is precisely the observation this "
                    "run exists to explain. measured = n divergent seeds lifting pre-commit."
                ),
                "control": "ARM_NOISE_SINGLE precommit_class_entropy vs A0_OFF, divergent seeds, paired",
                "measured": float(n_noise_pre_lift),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "direction": "lower",  # FLOOR: met iff measured >= threshold
                "met": bool(noise_lifts_precommit),
            },
        ],
        "criteria": [
            {
                # 708b's LOAD-BEARING criterion is the DISCRIMINATION, not MECH-440 propagation.
                # That is the honest statement of what this run is powered for: 708a already
                # showed the manipulation does not fire, so a criterion gated on propagation
                # would be unreachable-by-construction (the V3-EXQ-785 structural-vacuity shape).
                "name": "D1_precommit_shape_discriminates_saturation_vs_lever_arrival",
                "load_bearing": True,
                "passed": bool(discrimination_achieved),
            },
            {
                # Retained from 708a, explicitly NOT load-bearing: reachable only if the
                # manipulation now fires. Recorded so the two runs stay directly comparable.
                "name": "C1_noise_single_committed_strict_above_temp",
                "load_bearing": False,
                "passed": bool(c1_holds),
            },
        ],
        "criteria_non_degenerate": {
            # D1 discriminates only if the shape was actually estimable and the arms are not
            # bit-identical; a shape read off too few samples, or a saturation call made without
            # the OFF baseline, would be a vacuous discrimination.
            "D1_shape_estimable": bool(shape_samples_sufficient),
            "D1_off_baseline_present": bool(shape_by_arm["A0_OFF"]["n_seeds"] > 0),
            "D1_temperature_recoverable": bool(
                shape_by_arm["A0_OFF"]["temperature_recovered_mean"] > 1e-12
            ),
            "C1_preconditions_met": bool(preconditions_met),
            "C1_manipulation_fired": bool(manipulation_checks_pass),
            "enough_divergent_seeds": bool(enough_divergent),
            "noise_bias_supra_floor": bool(noise_bias_supra_floor),
            "dacc_live": bool(dacc_live),
            "learning_engaged": bool(fcg_moved_ok and fcg_delta_nonflat_ok),
            "fresh_selects_sufficient": bool(fresh_selects_sufficient),
        },
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
            "precommit_lift_margin": float(PRECOMMIT_LIFT_MARGIN),
            "min_divergent_seeds": int(MIN_DIVERGENT_SEEDS),
            "divergent_pass_fraction": float(DIVERGENT_PASS_FRACTION),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "min_fresh_selects": int(MIN_FRESH_SELECTS),
            "exposure_imbalance_report_floor": float(EXPOSURE_IMBALANCE_REPORT_FLOOR),
            "noise_bias_range_floor": float(NOISE_BIAS_RANGE_FLOOR),
            "noise_bias_to_raw_range_frac_floor": float(NOISE_BIAS_TO_RAW_RANGE_FRAC_FLOOR),
            "dacc_max_suppression_floor": float(DACC_MAX_SUPPRESSION_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "noisy_selection_sigma_init": float(NOISY_SELECTION_SIGMA_INIT),
            "noisy_selection_weight": float(NOISY_SELECTION_WEIGHT),
            "temp_noise_floor_alpha": float(TEMP_NOISE_FLOOR_ALPHA),
            "temp_noise_floor_min_temperature": float(TEMP_NOISE_FLOOR_MIN_TEMPERATURE),
            "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
            "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
        },
        "acceptance_criteria": {
            # ---- 708b: the DISCRIMINATION (load-bearing) ----
            "D1_discrimination_achieved": bool(discrimination_achieved),
            "reading_i_all_arms_saturated": bool(all_arms_saturated),
            "reading_i_off_arm_saturated": bool(off_arm_saturated),
            "reading_ii_temp_lever_arrives": bool(temp_lever_arrives),
            "reading_ii_noise_lever_arrives": bool(noise_lever_arrives),
            "reading_ii_both_levers_arrive": bool(both_levers_arrive),
            "temp_arrival_rel_lift": round(temp_arrival_rel_lift, 6),
            "noise_explore_term_present_frac": round(noise_explore_frac, 6),
            "maxprob_differs_by_arm": bool(maxprob_differs_by_arm),
            "maxprob_delta_by_arm_vs_off": maxprob_delta_by_arm,
            "per_arm_saturated": per_arm_saturated,
            "shape_samples_sufficient": bool(shape_samples_sufficient),
            "worst_cell_n_shape_samples": int(_worst_shape[0]),
            "worst_cell_shape_id": f"{_worst_shape[1]}:seed{_worst_shape[2]}",
            # ---- manipulation checks (recorded, NON-GATING) ----
            "manipulation_check_temp_lifts_precommit": bool(temp_lifts_precommit),
            "manipulation_check_noise_lifts_precommit": bool(noise_lifts_precommit),
            "manipulation_checks_pass": bool(manipulation_checks_pass),
            "preconditions_met": preconditions_met,
            "n_divergent_seeds": int(n_primary_div),
            "enough_divergent_seeds": enough_divergent,
            "temp_lifts_precommit": temp_lifts_precommit,
            "noise_lifts_precommit": noise_lifts_precommit,
            "noise_bias_supra_floor": noise_bias_supra_floor,
            "dacc_live": dacc_live,
            "learning_engaged_fcg_moved": fcg_moved_ok,
            "learning_engaged_fcg_delta_nonflat": fcg_delta_nonflat_ok,
            "fresh_selects_sufficient": fresh_selects_sufficient,
            "worst_cell_n_fresh_select": int(_worst_fresh[0]),
            "worst_cell_fresh_select_id": f"{_worst_fresh[1]}:seed{_worst_fresh[2]}",
            "max_exposure_imbalance_vs_off": round(max_exposure_imbalance, 6),
            "C1_noise_single_above_temp": c1_holds,
            "C1_n_seeds": int(len(c1_seeds)),
            "C1_n_divergent": int(n_primary_div),
            "mean_committed_class_entropy_a0_off": round(off_mean_dv, 6),
            "mean_committed_class_entropy_arm_temp": round(temp_mean_dv, 6),
            "mean_committed_class_entropy_arm_noise_single": round(nsingle_mean_dv, 6),
            "mean_precommit_class_entropy_a0_off": round(_mean([r["precommit_class_entropy_nats"] for r in off_rows]), 6),
            "mean_precommit_class_entropy_arm_temp": round(_mean([r["precommit_class_entropy_nats"] for r in temp_rows]), 6),
            "mean_precommit_class_entropy_arm_noise_single": round(_mean([r["precommit_class_entropy_nats"] for r in nsingle_rows]), 6),
        },
        "interpretation_grid": {
            "PASS_precommit_distribution_saturated_no_headroom_levers_arrive": (
                "READING (i) H-PRECOMMIT-SATURATED. Readiness gates met AND both levers "
                "demonstrably ARRIVE at the pre-commit softmax (ARM_TEMP's recovered applied "
                "temperature is lifted above A0_OFF's by at least TEMP_ARRIVAL_REL_FLOOR; the "
                "MECH-440 explore term is witnessed in the pre-commit score vector on at least "
                "EXPLORE_TERM_ARRIVAL_FRAC_FLOOR of fresh selects) AND every arm INCLUDING "
                "A0_OFF sits at mean max_prob >= PRECOMMIT_SATURATION_MAX_PROB. The pre-commit "
                "distribution is already argmax-like: there is no entropy headroom for any "
                "lever to exploit, which is why V3-EXQ-708a's manipulation check returned null "
                "on a working instrument. This is the MECH-439 F-dominance signature observed "
                "PRE-commit rather than post-commit. MECH-440 stays non_contributory -- with no "
                "injectable pre-commit variance the propagation question is still unasked. "
                "Routes to /failure-autopsy. Does NOT route to /implement-substrate."
            ),
            "PASS_precommit_lever_applied_downstream_does_not_reach_precommit_softmax": (
                "READING (ii) H-LEVER-DOWNSTREAM. Readiness gates met, the shape is NOT "
                "saturated in every arm, and at least one lever does NOT arrive: ARM_TEMP's "
                "recovered pre-commit temperature is not lifted above A0_OFF's, and/or the "
                "MECH-440 explore term is not witnessed in the score vector the pre-commit "
                "softmax normalises. The lever is applied downstream of where the pre-commit "
                "distribution is formed and cannot move it by construction -- a WIRING GAP, not "
                "a substrate property. MECH-440 non_contributory (untested, and for a reason "
                "that is repairable). Routes to /failure-autopsy, which owns the decision about "
                "whether a wiring repair is then warranted."
            ),
            "FAIL_precommit_shape_headroom_unexplained": (
                "NEITHER READING. Readiness gates met, both levers arrive, the distribution has "
                "headroom -- and still neither the pre-commit class entropy nor the class "
                "marginal moves. A genuinely new state that the 2026-07-22 autopsy did not "
                "enumerate. The candidate-level vs CLASS-level shape split is recorded "
                "separately precisely for this branch: the leading candidate explanation is "
                "that candidate-level headroom coexists with class-level collapse (several "
                "candidates sharing one first-action class), which precommit_max_class_mass_mean "
                "and precommit_n_distinct_classes_mean can show directly. MECH-440 "
                "non_contributory. Routes to /failure-autopsy."
            ),
            "PASS_weight_noise_carves_committed_diversity_above_matched_temperature_supports_mech440": (
                "RETAINED from V3-EXQ-708a and reachable ONLY if the manipulation NOW fires "
                "(708a measured 0/4 and 1/4 against a bar of 2). Both manipulation checks pass "
                "-- ARM_TEMP and ARM_NOISE_SINGLE each raise pre-commit sampling-class entropy "
                "strict-above A0_OFF on a strict-majority of divergent seeds -- AND C1 holds: "
                "ARM_NOISE_SINGLE committed-class entropy strict-above the matched-pre-commit-"
                "variance temperature control on a strict-majority of divergent seeds. The "
                "factorised-Gaussian weight noise CARVES through to the committed action where "
                "the temperature THRASHES at pre-commit (the V3-EXQ-687 non-propagation fix) "
                "-> supports MECH-440."
            ),
            "FAIL_substrate_not_ready_requeue": (
                "A genuine READINESS gate is unmet: too FEW divergent seeds, OR a sub-floor "
                "noise bias range (or one trivial vs the raw-score range), OR flat dACC "
                "suppression, OR learning not engaged, OR too FEW genuinely-fresh E3 selections, "
                "OR too FEW recoverable pre-commit shape samples. The discrimination could not "
                "be ATTEMPTED -- NOT a falsification. IMPORTANT: unlike V3-EXQ-708a, the two "
                "MANIPULATION CHECKS cannot reach this branch. A lever that fails to move the DV "
                "is a RESULT about the system, and routing it here is what buried 708a's "
                "finding as 'substrate not ready' on an instrument its own numbers proved was "
                "working."
            ),
        },
        # ---- 708a instrument telemetry (autopsy requirements 2, 3, 4) ----
        # Recorded in the MANIFEST ITSELF, deliberately NOT via a declared per_tick_sink:
        # Phase 3 cloud workers POST only manifest_bytes, so a declared sink is not
        # transported (twice-confirmed, V3-EXQ-785 and V3-EXQ-708 -- the latter is exactly
        # why 708's contamination was NOT recoverable by re-analysis and forced this re-run).
        "fresh_select_telemetry": {
            "note": (
                "n_fresh_select counts P2 ticks where E3 select() genuinely ran (diagnostics "
                "repopulated after an explicit None-clear); n_latched counts held/stepped "
                "ticks that recorded NOTHING. In V3-EXQ-708 the effective yield was 1.0 by "
                "construction -- every env step re-recorded the latched vector -- and the true "
                "denominator was invisible. exposure_imbalance_vs_off is the per-arm-seed "
                "fresh-selection count relative to that seed's A0_OFF arm: it is the mechanism "
                "by which the arm-dependent replication distorted 708's between-arm delta, and "
                "is REPORTED here (never gating) so a reader can verify the repaired DV does "
                "not inherit it."
            ),
            "min_fresh_selects_gate": int(MIN_FRESH_SELECTS),
            "fresh_selects_sufficient": bool(fresh_selects_sufficient),
            "worst_cell_n_fresh_select": int(_worst_fresh[0]),
            "worst_cell_id": f"{_worst_fresh[1]}:seed{_worst_fresh[2]}",
            "max_abs_exposure_imbalance_vs_off": round(max_exposure_imbalance, 6),
            "per_arm_seed": exposure_imbalance,
        },
        # ---- 708b: PRE-COMMIT SHAPE + LEVER ARRIVAL (the readout this run exists for) ----
        # Per arm per seed in arm_results[]; aggregated per arm here. In the MANIFEST, never a
        # declared per_tick_sink: Phase 3 cloud workers POST only manifest_bytes.
        "precommit_shape": {
            "note": (
                "THE DISCRIMINATOR between the two readings V3-EXQ-708a's manifest could not "
                "separate, because it recorded only the between-arm pre-commit class-entropy "
                "DELTA. (i) H-PRECOMMIT-SATURATED: max_prob_mean is at or above "
                "PRECOMMIT_SATURATION_MAX_PROB in EVERY arm INCLUDING A0_OFF -- the pre-commit "
                "distribution is argmax-like and has no headroom, so no lever acting on the "
                "score vector can move it (the MECH-439 F-dominance signature, observed "
                "PRE-commit). (ii) H-LEVER-DOWNSTREAM: max_prob differs BY ARM while the class "
                "entropy does not, and/or a lever does not arrive at all. All moments are "
                "accumulated on the SAME verified-fresh-select denominator as the entropy DV, "
                "so shape and entropy are directly comparable within each cell. The "
                "candidate-level and CLASS-level moments are kept separate on purpose: the DV "
                "aggregates candidates by first-action class, so candidate-level headroom can "
                "coexist with class-level collapse -- a third state neither reading names."
            ),
            "saturation_max_prob_threshold": float(PRECOMMIT_SATURATION_MAX_PROB),
            "mass_floor_for_eff_support": float(PRECOMMIT_MASS_FLOOR),
            "maxprob_arm_delta_floor": float(MAXPROB_ARM_DELTA_FLOOR),
            "min_shape_samples_gate": int(MIN_SHAPE_SAMPLES),
            "per_arm": shape_by_arm,
            "per_arm_saturated": per_arm_saturated,
            "maxprob_delta_by_arm_vs_off": maxprob_delta_by_arm,
            "maxprob_differs_by_arm": bool(maxprob_differs_by_arm),
            "all_arms_saturated": bool(all_arms_saturated),
        },
        "lever_arrival": {
            "note": (
                "Does each designed lever actually REACH the pre-commit softmax? Measured, not "
                "assumed -- a static code read cannot see a lever that is present but inert. "
                "ARM_TEMP: the APPLIED temperature is recovered EXACTLY per fresh select by "
                "inverting probs = F.softmax(-scores / T) (e3_selector.py:2742) against the "
                "same-tick last_scores (:2712, POST-explore-term) and last_precommit_probs "
                "(:2747), on the top-2 pair; both attributes are cleared before every "
                "select_action, so the pairing is provably same-tick. A recovered T on ARM_TEMP "
                "above A0_OFF's IS the MECH-313 lift arriving at the pre-commit softmax. "
                "ARM_NOISE_SINGLE: e3._last_explore_term non-None on a fresh select is a "
                "positive per-tick witness that the MECH-440 bias entered the score vector that "
                "softmax normalises -- :2698 does `scores = scores + _explore_term` and precedes "
                ":2742. n_temperature_unrecoverable counts ticks whose second-largest "
                "probability underflowed: that is NOT a measurement failure, it is direct "
                "evidence for reading (i), and it is counted rather than dropped. No ree_core "
                "edit is made or needed -- the 708a autopsy sets "
                "recommended_substrate_queue_entry.action = 'none'."
            ),
            "static_substrate_reading": (
                "As landed, BOTH levers are upstream of the pre-commit softmax: the MECH-313 "
                "temperature lift is composed in agent.py:6783-6829 into effective_temperature "
                "and passed to e3.select at agent.py:7044-7045, where it is the `temperature` "
                "divided by at e3_selector.py:2742 and is not reassigned in between; the "
                "MECH-440 explore term is added to `scores` at e3_selector.py:2698, before "
                "last_scores (:2712) and before that same softmax. This already argues against "
                "reading (ii) -- but a present-yet-inert lever (a noise_floor returning the base "
                "temperature, a head bias dropped by the non-finite guard at :2694) is invisible "
                "to a static read and visible to the measurement above. The measurement, not "
                "this note, decides."
            ),
            "temp_arrival_rel_floor": float(TEMP_ARRIVAL_REL_FLOOR),
            "explore_term_arrival_frac_floor": float(EXPLORE_TERM_ARRIVAL_FRAC_FLOOR),
            "a0_off_temperature_recovered_mean": round(_off_t, 6),
            "arm_temp_temperature_recovered_mean": round(_temp_t, 6),
            "temp_arrival_rel_lift": round(temp_arrival_rel_lift, 6),
            "temp_lever_arrives": bool(temp_lever_arrives),
            "noise_explore_term_present_frac": round(noise_explore_frac, 6),
            "noise_lever_arrives": bool(noise_lever_arrives),
            "both_levers_arrive": bool(both_levers_arrive),
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
    # 708a: surface arm_results at the TOP LEVEL (708 nested it under "result").
    # manifest_core._hoist_multi_arm_substrate_hash reads manifest["arm_results"], so with
    # it nested the hoist silently returns None and stamp_recording_core falls back to a
    # freshly-computed, DRIVER-INCLUSIVE single-arm hash that does NOT match the per-cell
    # arm_fingerprint.substrate_hash values -- the exact trap the recording standard warns
    # about (section 3b). Verified against the landed 708 manifest, which carries no
    # substrate_hash at all while every cell fingerprint has one. Moved rather than copied
    # so the payload is not duplicated in the JSON.
    _result_body = {k: v for k, v in result.items() if k != "arm_results"}
    _arm_results = result.get("arm_results") or []
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
            f"V3-EXQ-708b MECH-440 PRE-COMMIT DISTRIBUTION *SHAPE* FALSIFIER -- SAME-QUESTION "
            f"RE-RUN OF V3-EXQ-708a with a changed READOUT (supersedes the 708a run; "
            f"experiment_purpose=diagnostic; claim_ids=[MECH-440]; PROMOTES AND DEMOTES "
            f"NOTHING; routes to /failure-autopsy, NOT to /implement-substrate). WHY: 708a's "
            f"instrument repair WORKED -- 862 genuinely-fresh E3 select() calls against a floor "
            f"of 30, 4 divergent seeds, a supra-floor and non-trivial-vs-raw noise bias range "
            f"(0.221), live dACC suppression, dissociable finer channels. Its SUBSTANTIVE "
            f"finding is that NEITHER DESIGNED LEVER MOVES THE DEPENDENT VARIABLE: ARM_TEMP "
            f"(the 687 temperature control) raises pre-commit sampling-class entropy above "
            f"A0_OFF on 0 of 4 divergent seeds and ARM_NOISE_SINGLE on 1 of 4, against a bar of "
            f"2. That is a failed MANIPULATION CHECK on a demonstrably working instrument -- a "
            f"finding about the system, NOT an unready substrate "
            f"(failure_autopsy_V3-EXQ-708a_2026-07-22, user-adjudicated: record as a "
            f"substantive finding; epistemic_category measurement_gap, explicitly NOT "
            f"substrate_ceiling). TWO READINGS remain live and are NOT separable from 708a's "
            f"manifest, which records only the between-arm entropy DELTA: (i) the pre-commit "
            f"distribution is already argmax-like with no headroom for any lever (the MECH-439 "
            f"F-dominance signature, observed PRE-commit), or (ii) the levers are applied "
            f"downstream of where the pre-commit distribution is formed and cannot move it by "
            f"construction. THE READOUT CHANGE (design, arms, seeds, stack all UNCHANGED): "
            f"(1) the pre-commit distribution's SHAPE is recorded per genuinely-fresh select -- "
            f"max_prob, participation ratio 1/sum(p^2), effective support at a {PRECOMMIT_MASS_FLOOR} "
            f"mass floor, the top-3 mass profile, and the CLASS-level analogues -- aggregated "
            f"per arm per seed INTO THE MANIFEST (never a per_tick_sink: Phase 3 transports "
            f"only manifest_bytes). DISCRIMINATOR: under (i) max_prob is near 1 in EVERY arm "
            f"INCLUDING A0_OFF; under (ii) max_prob differs by arm while the entropy does not. "
            f"(2) each lever's ARRIVAL is instrumented at the E3 call site with NO ree_core "
            f"edit: the applied pre-commit softmax temperature is recovered EXACTLY by "
            f"inverting probs = softmax(-scores/T) (e3_selector.py:2742) against the same-tick "
            f"last_scores (:2712, post-explore-term) and last_precommit_probs (:2747), and the "
            f"MECH-440 explore term is witnessed per tick via e3._last_explore_term, set at "
            f":2698 where `scores = scores + _explore_term` -- both upstream of that softmax. "
            f"(3) temperature_control_raises_precommit_entropy and "
            f"weight_noise_raises_precommit_entropy are RECLASSIFIED from kind: readiness to "
            f"MANIPULATION CHECKS carried under interpretation.recorded_preconditions with an "
            f"explicit preconditions_scope_note (the V3-EXQ-737 pattern), because the "
            f"REE_assembly indexer's _compute_adjudication reads the flat preconditions list "
            f"ARM-BLIND and returns whole-run precondition_unmet on the first unmet entry -- "
            f"which is exactly what filed 708a's working instrument as 'substrate not ready'. A "
            f"null there now reports as a RESULT. UNCHANGED and NOT in question: the 708a "
            f"instrument repair (clear e3.last_score_diagnostics / .last_precommit_probs to "
            f"None immediately before every select_action and record ONLY if repopulated; 708b "
            f"additionally clears .last_scores, newly load-bearing for the temperature "
            f"recovery). LOAD-BEARING criterion is D1, the (i)-vs-(ii) DISCRIMINATION -- not "
            f"MECH-440 propagation, which 708a showed the manipulation cannot currently reach. "
            f"MECH-440 stays candidate / v3_pending and non_contributory in every branch except "
            f"the retained supports branch, which requires the manipulation to fire. NO "
            f"substrate build is warranted until (i) vs (ii) is settled. "
            f"outcome={result['outcome']}; label={result['interpretation_label']}; "
            f"per_claim={result['evidence_direction_per_claim']}."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "3-arm MECH-440 injection-site design UNCHANGED from V3-EXQ-708a (A0_OFF / ARM_TEMP / ARM_NOISE_SINGLE on the 569i top-k + MECH-448 demotion SOTA conversion stack, per-seed-divergent gating, 708a fresh-select freshness guard retained verbatim); the READOUT changes: pre-commit distribution SHAPE (max_prob / participation ratio / effective support / class-level analogues) plus per-tick LEVER-ARRIVAL instrumentation (exact recovery of the applied pre-commit softmax temperature; explore-term witness), which together discriminate H-precommit-saturated from H-lever-downstream",
            "swept_variables": "exploration-injection site only: none (A0_OFF) / softmax temperature (ARM_TEMP, MECH-313 noise_floor) / selection-head weight noise (ARM_NOISE_SINGLE, MECH-440)",
            "instrument_repair_vs_708_retained_verbatim": (
                "agent.e3.last_score_diagnostics and .last_precommit_probs are cleared to None "
                "immediately before EVERY agent.select_action, and a row is recorded ONLY if "
                "select() repopulated them. V3-EXQ-708 read both once per env step with no clear, "
                "so each vector was replicated by its hold duration under a 5-20-step "
                "MECH-093-modulated E3 cadence -- arm-dependently, hence directionally. "
                "n_fresh_select / n_latched / fresh_select_yield and per-arm-seed "
                "exposure_imbalance_vs_off are recorded in the manifest itself."
            ),
            "the_isolated_factor": (
                "all arms run the SAME SOTA conversion stack; ARM_TEMP raises the softmax "
                "temperature (pre-commit only, the 687 non-propagating control); ARM_NOISE_SINGLE "
                "injects factorised-Gaussian weight noise into _modulatory_accum BEFORE the "
                "committed within-eligible argmin (propagates)."
            ),
            "matched_constant_sota_stack": (
                "use_f_eligibility_demotion=True + use_f_eligibility_adaptive_floor=True + "
                "use_go_nogo_constitution=True + use_modulatory_selection_authority=True + "
                "use_modulatory_channel_routing + top_k shortlist (k=3, 569i) + use_dacc + "
                "use_finer_channel_gating + use_learned_settling_step"
            ),
            "primary_dv": "PRE-COMMIT DISTRIBUTION SHAPE per arm per seed (max_prob, participation ratio, effective support, class-level max mass and distinct-class count) on the verified-fresh-select denominator -- the (i)-vs-(ii) discriminator",
            "secondary_dv": "the two V3-EXQ-708a DVs retained UNCHANGED for direct comparability: committed-action-class entropy (nats) and pre-commit sampling-class entropy (nats), both on divergent seeds",
            "phases": "P0 e2-train (CRF matures, finer gating ON) -> P1 frozen-encoder bias-head REINFORCE -> P2 measure (e2+bias frozen; gating keeps adapting)",
            "mech440_noise": f"use_noisy_selection_head=True on the NOISE arms; sigma_init={NOISY_SELECTION_SIGMA_INIT}, weight={NOISY_SELECTION_WEIGHT}, local confidence-EMA self-anneal",
            "temp_control": f"use_noise_floor=True on ARM_TEMP; alpha={TEMP_NOISE_FLOOR_ALPHA}, min_temperature={TEMP_NOISE_FLOOR_MIN_TEMPERATURE} (effective_T ~ {1.0 + TEMP_NOISE_FLOOR_ALPHA})",
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "mech440_relationship": "this IS the MECH-440 injection-site falsifier; supports => weight noise propagates where temperature does not; weakens => the noise thrashes at pre-commit and washes at the committed argmax, so the injection site is not the binding constraint (NOT an ARC-110 route -- V3-EXQ-707b settled that branch)",
            "readout_change_vs_708a": (
                "arms, seeds, phases, env and every lever constant are IDENTICAL to V3-EXQ-708a. "
                "Three things change, all readout-side: (1) the pre-commit distribution's SHAPE "
                "is recorded, not only its entropy; (2) each lever's arrival at the pre-commit "
                "softmax is measured (recovered applied temperature; per-tick explore-term "
                "witness) rather than assumed; (3) the two entropy-lift entries are reclassified "
                "from gating readiness preconditions to recorded manipulation checks, so a null "
                "reports as a result instead of self-routing substrate_not_ready_requeue."
            ),
            "dv_symmetry_declaration": (
                "Per-arm, the symmetry group of the arm's DV and why the manipulation is NOT "
                "invariant under it (the V3-EXQ-604c check). A0_OFF: no manipulation (baseline). "
                "ARM_TEMP: the 708b primary DV is the pre-commit distribution SHAPE (max_prob, "
                "participation ratio, effective support), whose symmetry group is permutation of "
                "candidates plus any map preserving the probability vector. A temperature change "
                "rescales the logits and strictly changes max_prob for any non-constant score "
                "vector, so it is NOT invariant -- the shape DV can see the lever. "
                "ARM_NOISE_SINGLE: the MECH-440 term is a PER-CANDIDATE vector, not a broadcast "
                "scalar -- V3-EXQ-708a measured noisy_selection_bias_range = 0.221, a non-zero "
                "cross-candidate RANGE, which is the measured refutation of the broadcast-constant "
                "case, and max_prob is not invariant under a non-uniform additive perturbation of "
                "the scores. NOTE THE CONTRAST WITH THE OLD DV, which is a live hypothesis for "
                "708a's null rather than a defect in 708b: pre-commit CLASS entropy aggregates "
                "candidates by first-action class, so it IS invariant under any redistribution of "
                "mass WITHIN a class -- and if the k=3 top-k shortlist happens to hold candidates "
                "sharing one first-action class, class entropy is pinned at 0 for EVERY arm no "
                "matter what either lever does. precommit_n_distinct_classes_mean measures exactly "
                "that, which is why the class-level moments are recorded beside the "
                "candidate-level ones. The lever-arrival readouts (recovered temperature, "
                "explore-term witness) are INSTRUMENTS measuring the manipulation, not DVs testing "
                "it, so the invariance question does not arise for them."
            ),
            "no_substrate_edit": (
                "ree_core is NOT modified. The lever-arrival instrument is built entirely from "
                "attributes the substrate already records (e3.last_scores, "
                "e3.last_precommit_probs, e3._last_explore_term), by arithmetic inversion of the "
                "pre-commit softmax. The 708a autopsy sets recommended_substrate_queue_entry."
                "action = 'none': no substrate build is warranted until (i) vs (ii) is settled."
            ),
        },
        # Top-level so stamp_recording_core HOISTS substrate_hash from the per-cell
        # arm_fingerprints (see the note at the head of this function).
        "arm_results": _arm_results,
        "result": _result_body,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-708b MECH-440 pre-commit distribution shape falsifier"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, p2, steps = DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_P2, DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0, p1, p2, steps = (
            P0_WARMUP_EPISODES, P1_BIAS_TRAIN_EPISODES, P2_MEASUREMENT_EPISODES, STEPS_PER_EPISODE
        )

    result = run_experiment(
        seeds=seeds, p0_episodes=p0, p1_episodes=p1, p2_episodes=p2,
        steps_per_episode=steps, dry_run=bool(args.dry_run),
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
        # the seeds ACTUALLY run, not the module constant -- under --dry-run SEEDS would
        # record 6 seeds for a 1-seed smoke.
        seeds=seeds,
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
        f"D1_discriminates={_ac['D1_discrimination_achieved']} "
        f"all_arms_saturated={_ac['reading_i_all_arms_saturated']} "
        f"off_saturated={_ac['reading_i_off_arm_saturated']} "
        f"temp_arrives={_ac['reading_ii_temp_lever_arrives']}"
        f"(rel_lift={_ac['temp_arrival_rel_lift']}) "
        f"noise_arrives={_ac['reading_ii_noise_lever_arrives']}"
        f"(explore_frac={_ac['noise_explore_term_present_frac']}) "
        f"maxprob_differs_by_arm={_ac['maxprob_differs_by_arm']} "
        f"manip_temp={_ac['manipulation_check_temp_lifts_precommit']} "
        f"manip_noise={_ac['manipulation_check_noise_lifts_precommit']} "
        f"noise_bias_ok={_ac['noise_bias_supra_floor']} dacc_live={_ac['dacc_live']} "
        f"C1_noise_above_temp={_ac['C1_noise_single_above_temp']} "
        f"fresh_ok={_ac['fresh_selects_sufficient']} "
        f"worst_fresh={_ac['worst_cell_n_fresh_select']}@{_ac['worst_cell_fresh_select_id']} "
        f"shape_ok={_ac['shape_samples_sufficient']} "
        f"worst_shape={_ac['worst_cell_n_shape_samples']}@{_ac['worst_cell_shape_id']} "
        f"max_exposure_imbalance={_ac['max_exposure_imbalance_vs_off']} "
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
