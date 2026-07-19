#!/opt/local/bin/python3
"""V3-EXQ-722: CROSS-BANK GENERALITY falsifier for the f_dominance_conversion_ceiling
SELECTION face -- does the MECH-448 (ARC-107) rank-preserving F->eligibility DEMOTION
lever's off-substrate PERSISTENCE generalise, or is it an artifact of the
spread-F / envelope-floor-calibration crutch?

!! PARKED -- DO NOT QUEUE WITHOUT LIFTING THE RE-DERIVE BRAKE (status 2026-07-19) !!
------------------------------------------------------------------------------------
This script is COMPLETE and LANDED FOR PRESERVATION ONLY. It has NO entry in
ree-v3/experiment_queue.json and MUST NOT be given one in its current form.

It was authored ~2026-06-21/22 and never committed by its original session, which
almost certainly abandoned it on the finding that landed within a day or two of
authoring: failure_autopsy_V3-EXQ-654i + failure_autopsy_V3-EXQ-654j_2026-06-22
established that BOTH ARC-107 eligibility-governance legs (MECH-448 demotion,
MECH-449 Go/No-Go) PASS at the SELECTION face but NEITHER converts at the
BEHAVIOURAL committed-class-entropy face (C2) -- the exact DV this experiment
scores. The re-derive brake FIRED there (18th MECH-309 / 19th ARC-062) and REFUSES
any further eligibility-governance letter on the 689 / 485 / 654 lineages. It has
since fired again (20th, 2026-07-07, on V3-EXQ-714; behavioral_diversity_isolation_plan.md
records "fullstack re-queue REFUSED"). The diagnosis of record is that the conversion
ceiling is structural and DOWNSTREAM of selection (root C, commit-entry/latch), so a
cross-bank generality re-test of the SELECTION-face lever cannot be interpreted.

Separately, the whole f_dominance / MECH-439 campaign is PARKED behind two upstream
blocks -- MECH-457 (competence floor / missing first-class RPE actor-critic) and
INV-088 (z_world under-differentiation). CURRENT_FRONT.md states outright that the
live front is not "attack f_dominance".

REOPEN CONDITION: the brake is lifted AND the upstream parks clear (MECH-457 +
INV-088), OR this design is re-scoped to a face the brake does not cover. At that
point re-validate the readiness floors and arm config against the then-current
substrate before queuing -- do not queue it as-is on the strength of this file
compiling. Route any re-queue through /queue-experiment.

OWED ON REOPEN: this script predates the Experimental Recording Standard
(2026-07-12) and still writes its flat manifest with a raw json.dump. It carries
MANIFEST_WRITER_EXEMPT (below) on the archival not-re-run grounds above. Any reopen
MUST first migrate the manifest tail to experiments/pack_writer.write_flat_manifest
and drop that exemption.

Verified at landing (2026-07-19): compiles, all five substrate imports resolve,
use_f_eligibility_demotion still present in e3_selector.py, no TODO/stub markers.

BACKGROUND (the open question this experiment answers)
------------------------------------------------------
MECH-448 = rank-preserving F->eligibility DEMOTION (a divisive-normalisation envelope
with an ABSOLUTE merit-share floor). F decides ELIGIBILITY, not the winner; within the
F-eligible set a modulatory channel picks the committed action with F REMOVED from the
final argmin. MECH-449 = the Go/No-Go constitution follow-on.

  - On the GAP-A foraging substrate the demotion lever CONVERTS committed-action
    diversity: V3-EXQ-689d PASS (committed entropy 0.938 ON vs 0.371 OFF).
  - On two OFF-substrate BEHAVIOURAL banks it PERSISTS the ceiling once calibrated to
    FIRE: V3-EXQ-485j (OFC bank) and V3-EXQ-654i (arc_062 rule-apprehension bank).
    BOTH those banks had a SPREAD (non-divergent) per-candidate F pool, so the
    689d-tuned absolute floor (0.30) all-admitted (excluded_count==0, a STRUCTURAL
    no-op) until a per-(arm,seed) envelope-floor CALIBRATION was added to force the
    envelope to exclude a non-empty tail.

OPEN QUESTION: is off-substrate persistence GENERAL, or an artifact of the
spread-F / calibration crutch? To isolate that, this experiment tests the lever on a
THIRD candidate-bank type whose F pool is DIVERGENT-BY-CONSTRUCTION (peaked), so the
MECH-448 envelope FIRES CLEANLY under the DEFAULT floor (0.30) WITHOUT needing
calibration to rescue an all-admit, AND it puts a MATCHED GAP-A foraging arm IN THE
SAME RUN as the positive-control anchor (reproduce 689d qualitatively).

DESIGN -- a (bank_type x arm x seed) grid (2 banks x 3 arms x 3 seeds = 18 cells)
--------------------------------------------------------------------------------
bank_type:
  gapa_foraging  -- the V3-EXQ-689d reef-bipartite ENV_KWARGS VERBATIM;
    candidate_summary_source=e2_world_forward. POSITIVE-CONTROL ANCHOR: the substrate
    where the lever is KNOWN to CONVERT. Its F pool is peaked; the default floor 0.30
    already fires (689d READINESS(c) non-degeneracy held there).
  bankC_divergent -- a THIRD, genuinely distinct BEHAVIOURAL bank (NOT the OFC 485j
    bank, NOT the arc_062 654i bank, NOT the plain gapa foraging task): a
    "decisive-threat forced-choice" bank that REUSES the 654i rule-apprehension
    committed-class machinery (the differentiated rule-field modulatory channel + the
    committed_class_entropy readout) but HOSTS it on a CONCENTRATED HIGH-HAZARD env
    regime (raised num_hazards / hazard_harm / proximity_harm_scale, a non-bipartite
    concentrated hazard layout, reef OFF) tuned so ONE first-action class is decisively
    lowest-harm => the per-candidate harm-cost F is PEAKED => the MECH-448 envelope
    EXCLUDES a non-empty tail under the DEFAULT floor (0.30), i.e.
    DIVERGENT-BY-CONSTRUCTION, WITHOUT relying on calibration. R6 (below) is the
    structural-divergence check; if it cannot be achieved the calibration path is kept
    (belt-and-suspenders) and R6 self-routes substrate_not_ready for bankC.

arm (the V3-EXQ-485j 3-way dissociation: (head trained?) x (demotion on?)):
  ARM_trained_demotion_on  (TEST)     -- trained informative channel
    (use_candidate_rule_field=True + lateral_pfc_train_rule_bias_head=True) + demotion ON.
  ARM_trained_demotion_off (CEILING)  -- trained channel + demotion OFF (hard top_k
    F-prefix; F dominates the argmin).
  ARM_frozen_demotion_on   (SILENCE)  -- FROZEN channel
    (lateral_pfc_train_rule_bias_head=False, bias head left zero-init so its bias is
    ~uninformative; use_candidate_rule_field=True kept as a matched constant) + demotion ON.
Everything else is matched-constant (the now-working CRF stack, SD-056, SP-CEM,
MECH-341, MECH-313, V_s, candidate_summary_source per bank).

DV + PRE-REGISTERED ACCEPTANCE
------------------------------
DV = committed_class_entropy_nats per cell (654i _entropy_from_int_counts).

C_PRIMARY (LOAD-BEARING, computed PER BANK): ARM_trained_demotion_on committed-class
  entropy STRICTLY ABOVE BOTH ARM_trained_demotion_off AND ARM_frozen_demotion_on by
  C2_LIFT_MARGIN_NATS (0.05) on >= 2/3 seeds (paired by seed). "Converts" iff C_PRIMARY
  holds for that bank.

READINESS (per bank; ANY fail on a bank => that bank self-routes substrate_not_ready,
  NEVER a weakens):
  R1 committed-class axis exercisable (frac_pre_ge2 > FRAC_PRE_GE2_FLOOR) all arms >=2/3 seeds.
  R2 consumed-summary divergence real (consumed_summary_pairwise_dist_mean >
     CONSUMED_SPREAD_FLOOR, bounded) all arms >=2/3 seeds.
  R3 trained arms channel matured + bias head trained (head weight-delta > floor) on
     trained-on seeds.
  R4 propagation non-vacuity (trained-on bias abs-mean differs from frozen by > floor).
  R5 MECH-448 C1e non-degeneracy: f_eligibility_excluded_count > 0 on demotion-ON arms
     >=2/3 seeds (AFTER per-(arm,seed) calibration). All-admit => substrate_not_ready.
  R6 (bankC-only, THE divergent-by-construction check): on bankC demotion-ON arms,
     f_eligibility_excluded_count > 0 under the DEFAULT floor (0.30) BEFORE calibration
     on >=2/3 seeds -- proving the divergence is STRUCTURAL, not calibration-manufactured.
     Both the default-floor and post-calibration excluded_count are recorded. If R6
     fails, bankC self-routes substrate_not_ready (the bank was not built as intended).

The 485j-style per-(arm,seed) _calibrate_envelope_floor + ENVELOPE_KEEP_MIN/MAX and
dn_sigma=0.0 are kept (belt-and-suspenders even for bankC), so the lever fires even
where the raw pool is spread; R6 records whether bankC ALSO fires WITHOUT it.

INTERPRETATION GRID (self-route; experiment_purpose="evidence" with a falsifiable block)
----------------------------------------------------------------------------------------
  gapa positive control FAILS to convert (C_PRIMARY not met on gapa) OR gapa readiness
    unmet => substrate_not_ready_requeue (whole run uninterpretable; the harness is not
    reproducing 689d). NOT a weakens.
  gapa converts AND bankC readiness unmet (esp R5/R6) => substrate_not_ready_requeue for
    the cross-bank arm. NOT a weakens.
  gapa converts AND bankC converts => demotion_converts_cross_bank_divergent_F --
    persistence was tied to spread-F / calibration; supports MECH-448 / MECH-309 /
    ARC-062 GENERALITY. evidence_direction "supports".
  gapa converts AND bankC does NOT convert (readiness incl R6 MET) =>
    conversion_ceiling_general_persists_on_divergent_F_third_bank -- persistence is
    GENERAL, not a spread/calibration artifact; MECH-448 stays provisional; route
    MECH-449 / V4. evidence_direction "non_contributory" (persistence, NOT a
    falsification). PROMOTES NOTHING.

claim_ids = [MECH-309, ARC-062, MECH-448]. experiment_purpose = evidence. PROMOTES
NOTHING by itself in ALL branches; MECH-448 stays provisional. MECH-448 is tagged
because this IS the demotion-lever generality test: a bankC-converts result is the
FIRST cross-bank confirmation the lever generalises off GAP-A onto a divergent-F
behavioural bank; a bankC-persists result leaves MECH-448 provisional (the ceiling is
general). MECH-309 / ARC-062 are the committed-action-diversity / rule-apprehension
claims whose SELECTION face the conversion ceiling sits on.

See:
  experiments/v3_exq_654i_arc062_gapb_rule_apprehension_behavioural_falsifier.py (base),
  experiments/v3_exq_485j_sd033b_demotion_envelope_calibrated_behavioural.py (3-arm dissociation),
  experiments/v3_exq_689d_mech448_f_eligibility_demotion_falsifier.py (the GAP-A PASS),
  REE_assembly/docs/architecture/mech_448_f_eligibility_demotion.md,
  REE_assembly/evidence/planning/arc_107_selector_constitution_design_2026-06-20.md.

Usage:
  /opt/local/bin/python3 experiments/v3_exq_722_crossbank_conversion_ceiling_selection_falsifier.py --dry-run
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


EXPERIMENT_TYPE = "v3_exq_722_crossbank_conversion_ceiling_selection_falsifier"
QUEUE_ID = "V3-EXQ-722"
SUPERSEDES: Optional[str] = None  # NEW cross-bank generality question, not a fix.
CLAIM_IDS: List[str] = ["MECH-309", "ARC-062", "MECH-448"]
EXPERIMENT_PURPOSE = "evidence"

# Landed PARKED for preservation only, never queued, never to be re-run as-is (see the
# PARKED banner in the module docstring). Authored ~2026-06-21/22, i.e. BEFORE the
# Experimental Recording Standard (2026-07-12) made pack_writer.write_flat_manifest the
# single sanctioned flat-manifest writer. Migrating the writer now would be an
# unverifiable edit to a script that cannot be run for real under the re-derive brake, so
# the migration is DEFERRED and OWED as part of the reopen condition, not skipped.
MANIFEST_WRITER_EXEMPT = (
    "parked pre-standard script (authored ~2026-06-21/22, pre-2026-07-12 recording "
    "standard); landed for preservation only, not queued and not re-run -- migrate to "
    "pack_writer.write_flat_manifest as part of any reopen"
)

# --- CRF-gate calibration amend levers (the 654i now-working stack; matched constant) ---
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# Within-class-representative signature horizon (SECONDARY negative control).
H_SIGNATURE = 3

# C_PRIMARY (LOAD-BEARING, per bank): paired-by-seed committed-class entropy lift of
# ARM_trained_demotion_on over BOTH collapsed controls (ceiling + silence).
C2_LIFT_MARGIN_NATS = 0.05
C2_MIN_LIFT_SEEDS = 2  # of 3

# Readiness floors.
FRAC_PRE_GE2_FLOOR = 0.30          # R1: committed-class axis exercisable
CONSUMED_SPREAD_FLOOR = 0.05       # R2: consumed-summary divergence
CONSUMED_MAGNITUDE_CEIL = 1.0e6    # R2: 643a explosion ceiling
CRF_MIN_MINTED = 2                 # R3: distinct rules minted (trained arms)
CRF_N_ACTIVE_FLOOR = 1             # >= 1 active rule => non-zero rule_state this tick
CRF_FRAC_ACTIVE_FLOOR = 0.30       # R3: fraction of P2 ticks the field fired a rule_state
HEAD_DELTA_MIN = 1e-3              # R3: bias-head weight-delta floor (trained-on arms)
PROP_NONVAC_FLOOR = 1e-3           # R4: |bias_abs(trained_on) - bias_abs(frozen)|
MIN_TICKS_PER_CLASS = 5            # secondary within-class-entropy qualifier
MIN_SEEDS_FOR_PASS = 2             # of 3

# R5 MECH-448 demotion non-vacuity (after calibration).
DEMOTION_ACTIVE_FRAC_FLOOR = 0.8   # fraction of P2 ticks the f_demotion envelope is active
EXCLUDED_COUNT_FLOOR = 0.0         # mean f_eligibility_excluded_count strictly above this

# R6 (bankC-only) divergent-by-construction. THE structural-divergence check. NOTE the
# design lesson from the smoke test: raw excluded_count under the default absolute floor
# is DOMINATED BY CANDIDATE COUNT (with ~32 SP-CEM candidates and an absolute 0.30
# merit-share floor, almost every candidate falls below the floor -- so excluded_count>0
# is trivially true on ANY pool, peaked OR spread, and does NOT discriminate structural
# divergence). The COUNT-INVARIANT structural-divergence statistic is the TOP-1 MERIT
# SHARE: the fraction of total merit held by the single F-best candidate. A genuinely
# PEAKED (divergent-by-construction) F pool concentrates merit on one candidate (high
# top-1 share); a SPREAD pool distributes it (low top-1 share, ~1/K). R6 requires bankC's
# top-1 merit share to clear R6_TOP1_MERIT_SHARE_FLOOR, a threshold the gapa positive
# control does NOT meet -- proving bankC fires the envelope from STRUCTURAL peakedness,
# not the calibration crutch or the candidate-count artifact.
R6_TOP1_MERIT_SHARE_FLOOR = 0.25   # bankC must exceed; gapa's ~0.06 does not

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 120           # trimmed from the 654i 200 to keep 18 cells reasonable
P1_BIAS_TRAIN_EPISODES = 70        # frozen-encoder bias-head REINFORCE (GAP-D)
P2_MEASUREMENT_EPISODES = 50       # all frozen; behavioural measurement
STEPS_PER_EPISODE = 200
# episodes_per_run per cell = P0 + P1 + P2 = 240 (documented in the queue note).

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 3
DRY_RUN_P1 = 4
DRY_RUN_P2 = 3
DRY_RUN_STEPS = 30

# Matched-stack lever constants (identical on all arms/banks).
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
USE_MODULATORY_CHANNEL_ROUTING = True
MODULATORY_CHANNEL_ROUTE_SOURCE = "cand_world_summary"
MODULATORY_CHANNEL_ROUTE_WEIGHT = 1.0
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6
USE_MODULATORY_SHORTLIST_THEN_MODULATE = True
MODULATORY_SHORTLIST_MODE = "top_k"     # 569i lineage kept; f_demotion overrides it
MODULATORY_SHORTLIST_K = 3
# MECH-448 (ARC-107) rank-preserving F->eligibility demotion.
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30     # 689d / substrate DEFAULT absolute DN-share floor
F_ELIGIBILITY_DN_SIGMA = 0.0            # 689d / substrate default DN semi-saturation
# 485j-style envelope-floor calibration (belt-and-suspenders even for bankC).
ENVELOPE_KEEP_MIN = 2
ENVELOPE_KEEP_MAX = 4
MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

# SD-056 online e2 training (mirror V3-EXQ-649 / 654i harness).
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

# P1 bias-head REINFORCE training (mirror V3-EXQ-598b / 654i).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9


# ---------------------------------------------------------------------------
# Bank definitions
# ---------------------------------------------------------------------------
# gapa_foraging: the V3-EXQ-689d reef-bipartite ENV_KWARGS VERBATIM (the substrate
# where the demotion lever is known to CONVERT). candidate_summary_source=e2_world_forward.
GAPA_ENV_KWARGS: Dict[str, Any] = dict(
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

# bankC_divergent: a CONCENTRATED HIGH-HAZARD, NON-bipartite (reef OFF) regime tuned so
# one first-action class is decisively lowest-harm => the per-candidate harm-cost F is
# PEAKED (divergent-by-construction) => the MECH-448 envelope EXCLUDES a non-empty tail
# under the DEFAULT floor (0.30) WITHOUT calibration. Smaller grid + many high-harm
# hazards + high proximity-harm scale concentrate the hazard field so escaping-class
# candidates have decisively lower harm cost than approaching-class candidates.
# TUNE these in the smoke test until R6 (default-floor excluded_count > 0) holds
# (see the smoke-test / R6 readiness discussion in the docstring).
BANKC_ENV_KWARGS: Dict[str, Any] = dict(
    size=8,
    num_hazards=10,
    num_resources=3,
    hazard_harm=0.6,
    env_drift_interval=8,
    env_drift_prob=0.05,
    proximity_harm_scale=0.6,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.35,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=False,          # NON-reef concentrated-hazard layout (default)
    hazard_food_attraction=0.0,  # no reef => no bipartite food/hazard split
)

BANKS: List[Dict[str, Any]] = [
    {
        "bank_id": "gapa_foraging",
        "label": "gapa_foraging_reef_bipartite_positive_control_anchor",
        "env_kwargs": GAPA_ENV_KWARGS,
        "candidate_summary_source": "e2_world_forward",
        "divergent_by_construction": False,  # spread-F relies on calibration to fire
        "is_positive_control": True,
    },
    {
        "bank_id": "bankC_divergent",
        "label": "bankC_decisive_threat_forced_choice_concentrated_hazard_divergent_F",
        "env_kwargs": BANKC_ENV_KWARGS,
        "candidate_summary_source": "e2_world_forward",
        "divergent_by_construction": True,   # peaked-F: default floor should fire (R6)
        "is_positive_control": False,
    },
]

# The 485j 3-way dissociation: (head trained?) x (demotion on?).
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_trained_demotion_on",
        "label": "trained_informative_channel_demotion_on_TEST",
        "train_head": True,
        "demotion": True,
    },
    {
        "arm_id": "ARM_trained_demotion_off",
        "label": "trained_channel_demotion_off_F_dominance_ceiling",
        "train_head": True,
        "demotion": False,
    },
    {
        "arm_id": "ARM_frozen_demotion_on",
        "label": "frozen_zeroed_head_demotion_on_silence",
        "train_head": False,
        "demotion": True,
    },
]


def _make_env(bank: Dict[str, Any], seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **bank["env_kwargs"])


def _calibrate_envelope_floor(
    raw_scores: torch.Tensor,
    dn_sigma: float = F_ELIGIBILITY_DN_SIGMA,
    keep_min: int = ENVELOPE_KEEP_MIN,
    keep_max: int = ENVELOPE_KEEP_MAX,
) -> Optional[float]:
    """485j / 654i envelope-floor calibration (belt-and-suspenders).

    The MECH-448 envelope (e3_selector._f_eligibility_envelope) admits candidate i iff
    its merit-share elig[i] = merit[i] / (dn_sigma + sum(merit)) clears an ABSOLUTE floor
    (merit[i] = max(F) - F[i]; F = raw_scores, a per-candidate cost). On a SPREAD F pool
    the 689d-tuned floor 0.30 admits every candidate (excluded_count==0 -> demotion no-op).
    This CALIBRATES an ABSOLUTE floor to the bank's actual merit-share distribution: keep
    the F-best top-k (k in [keep_min, keep_max]) and exclude the rest, choosing the cut
    with the LARGEST share-gap. KEEP an ABSOLUTE share floor; KEEP dn_sigma=0.0. Returns
    None when no clean gap exists (flat / near-flat F) -- the caller then leaves the
    default floor so the envelope all-admits and the R5 (excluded_count>0) gate
    self-routes. dn_sigma MUST match the agent config value (0.0).
    """
    n = int(raw_scores.shape[0])
    if n < 2:
        return None
    merit = (raw_scores.max() - raw_scores).clamp(min=0.0)
    merit_sum = float(merit.sum().item())
    if merit_sum <= 1e-8:
        return None  # flat F -- no discrimination, no calibratable floor
    elig = (merit / (dn_sigma + merit_sum)).detach().cpu()
    sorted_desc, _ = torch.sort(elig, descending=True)
    k_hi = min(keep_max, n - 1)          # must exclude >= 1 candidate
    best_gap = -1.0
    best_floor: Optional[float] = None
    for keep in range(keep_min, k_hi + 1):
        hi = float(sorted_desc[keep - 1].item())   # smallest kept share
        lo = float(sorted_desc[keep].item())       # largest excluded share
        gap = hi - lo
        if gap > 1e-6 and gap > best_gap:
            best_gap = gap
            best_floor = 0.5 * (hi + lo)
    return best_floor


def _top1_merit_share(raw_scores: torch.Tensor) -> Optional[float]:
    """R6 count-invariant structural-divergence statistic: the fraction of TOTAL merit
    held by the single F-best candidate (merit[i] = max(F) - F[i]; F = raw_scores, a
    per-candidate cost). A PEAKED (divergent-by-construction) F pool concentrates merit
    on one candidate (top-1 share -> 1); a SPREAD pool distributes it (top-1 share ->
    ~1/K). Unlike the raw excluded_count under an absolute floor, this does NOT depend on
    the candidate count K, so it genuinely discriminates a peaked pool from a spread one.
    Returns None on a flat F (no discrimination)."""
    n = int(raw_scores.shape[0])
    if n < 2:
        return None
    merit = (raw_scores.max() - raw_scores).clamp(min=0.0)
    merit_sum = float(merit.sum().item())
    if merit_sum <= 1e-8:
        return None
    return float(merit.max().item()) / merit_sum


def _default_floor_excluded_count(raw_scores: torch.Tensor) -> int:
    """R6 divergent-by-construction probe: count how many candidates the MECH-448
    envelope would EXCLUDE under the DEFAULT floor (F_ELIGIBILITY_ENVELOPE_FLOOR),
    BEFORE any calibration. Reproduces e3_selector._f_eligibility_envelope's
    absolute-share-floor admission rule (dn_sigma=0.0). > 0 on a peaked pool proves the
    divergence is STRUCTURAL, not calibration-manufactured; == 0 on a spread pool means
    the default floor all-admits (the crutch the calibration exists to rescue)."""
    n = int(raw_scores.shape[0])
    if n < 2:
        return 0
    merit = (raw_scores.max() - raw_scores).clamp(min=0.0)
    merit_sum = float(merit.sum().item())
    if merit_sum <= 1e-8:
        return 0
    elig = merit / (F_ELIGIBILITY_DN_SIGMA + merit_sum)
    eligible = (elig >= F_ELIGIBILITY_ENVELOPE_FLOOR)
    n_eligible = int(eligible.sum().item())
    # The substrate guarantees at least one eligible (the F-best); excluded = n - eligible.
    return int(max(0, n - max(1, n_eligible)))


def _make_agent(
    bank: Dict[str, Any], train_head: bool, demotion: bool
) -> REEAgent:
    """Matched-stack agent. Varied factors: use_candidate_rule_field is TRUE on all
    arms (matched constant); the swept factors are (train_head?) x (demotion?):
      - train_head=True  -> lateral_pfc_train_rule_bias_head=True (informative channel,
        trainable head).
      - train_head=False -> lateral_pfc_train_rule_bias_head=False (FROZEN zero-init head
        => ~uninformative bias; the silence arm).
      - demotion         -> use_f_eligibility_demotion toggle.
    candidate_summary_source = e2_world_forward on both banks (e2 trained online in P0).
    """
    env = _make_env(bank, 0)  # dims only; per-cell env is rebuilt in _run_cell
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
        candidate_summary_source=str(bank["candidate_summary_source"]),
        # modulatory-bias-selection-authority (643a) + routing + 569i top_k lineage.
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
        # MECH-448 (ARC-107) rank-preserving F->eligibility demotion (SWEPT per arm).
        use_f_eligibility_demotion=bool(demotion),
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        # MECH-341 stratified.
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
        # ARC-062 GatedPolicy (matched).
        use_gated_policy=True,
        # SD-033a LateralPFCAnalog. train_head SWEPT: True -> trainable un-zeroed head
        # (informative channel); False -> frozen zero-init head (~uninformative; silence).
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=bool(train_head),
        # SD-056 (e2 action-conditional divergence; trained online in P0).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
        # --- CRF maturity + maintenance levers (matched constant; use_candidate_rule_field
        # is TRUE on all arms so the differentiated channel is present; the silence arm
        # differs only by the FROZEN uninformative head, not by the CRF field). ---
        use_candidate_rule_field=True,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_context_from_e2_world_forward=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=CRF_MAINTENANCE_FLOOR,
        crf_maintenance_decay=CRF_MAINTENANCE_DECAY,
        crf_mature_context_match_threshold=CRF_MATURE_CONTEXT_MATCH_THRESHOLD,
        crf_tolerance_conflict_cap=CRF_TOLERANCE_CONFLICT_CAP,
        crf_maintenance_couple_to_theta=CRF_MAINTENANCE_COUPLE_TO_THETA,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-649 / 654i)
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
# Per-tick measurement helpers (mirror 654i)
# ---------------------------------------------------------------------------


def _traj_first_action_class(traj) -> int:
    return int(traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item())


def _traj_rep_signature(traj) -> Tuple[int, ...]:
    acts = traj.actions[0]  # [horizon, action_dim]
    h = min(H_SIGNATURE, int(acts.shape[0]))
    classes = acts[:h, :].argmax(dim=-1).detach().reshape(-1).tolist()
    return tuple(int(c) for c in classes)


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


def _head_weight_vector(agent: REEAgent) -> torch.Tensor:
    lpfc = getattr(agent, "lateral_pfc", None)
    if lpfc is None:
        return torch.zeros(1)
    return torch.cat(
        [p.detach().reshape(-1).cpu() for p in lpfc.bias_head_parameters()]
    )


# ---------------------------------------------------------------------------
# P1 bias-head REINFORCE training (mirror V3-EXQ-598b / 654i)
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


def _propagation_counterfactual_delta(
    agent: REEAgent, summaries: torch.Tensor
) -> Optional[float]:
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


# ---------------------------------------------------------------------------
# Per-(bank, arm, seed) cell runner
# ---------------------------------------------------------------------------


def _run_cell(
    bank: Dict[str, Any],
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)
    env = _make_env(bank, seed)
    train_head = bool(arm["train_head"])
    demotion = bool(arm["demotion"])
    agent = _make_agent(bank, train_head, demotion)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = torch.optim.Adam(
        list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS
    )
    head_init = _head_weight_vector(agent)
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

    # PRIMARY DV + readiness accumulators (P2).
    committed_class_counts: Dict[int, int] = {}
    selected_class_counts: Dict[int, int] = {}
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0

    # SECONDARY negative control (within-class-representative; P2).
    per_class_rep_sigs: Dict[int, Counter] = {}
    all_rep_sigs: Counter = Counter()

    # CRF differentiation + bias diagnostics (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_matched_per_tick: List[int] = []
    crf_max_pairwise_rule_dist_max = 0.0
    crf_n_minted_total_last = 0
    lateral_pfc_bias_abs_vals: List[float] = []
    prop_counterfactual_deltas: List[float] = []

    # MECH-448 f_eligibility-demotion readouts (P2 select ticks). demotion-ON arms only.
    demotion_active_ticks = 0
    demotion_envelope_sizes: List[float] = []
    demotion_excluded_counts: List[float] = []
    demotion_winner_neq_f_argmin_ticks = 0
    demotion_rank_preserving_active_ticks = 0
    # R6 divergent-by-construction probes (BEFORE calibration): the count-invariant
    # top-1 merit share (the real structural-divergence statistic) + the raw default-floor
    # excluded_count (kept as a diagnostic; count-dominated, NOT the R6 gate).
    default_floor_excluded_counts: List[int] = []
    top1_merit_shares: List[float] = []

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
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at bank={bank['bank_id']} "
                        f"arm={arm['arm_id']} seed={seed} phase={phase_label} "
                        f"ep={ep} step={_step}"
                    )
                break

            committed_class = int(action[0].argmax().item())

            # R6 + calibration: select_action just populated agent.e3.last_raw_scores
            # (this tick's F pool). On demotion-ON arms in P2, FIRST record the
            # default-floor excluded_count (R6: does the peaked bankC pool fire the
            # envelope WITHOUT calibration?), THEN re-calibrate the ABSOLUTE floor so the
            # envelope excludes on the NEXT tick even where the raw pool is spread.
            if demotion:
                _raw = getattr(agent.e3, "last_raw_scores", None)
                if _raw is not None and _raw.numel() >= 2:
                    if is_p2:
                        default_floor_excluded_counts.append(
                            _default_floor_excluded_count(_raw.detach())
                        )
                        _t1 = _top1_merit_share(_raw.detach())
                        if _t1 is not None and math.isfinite(_t1):
                            top1_merit_shares.append(_t1)
                    _floor = _calibrate_envelope_floor(_raw.detach())
                    if _floor is not None:
                        agent.e3.config.f_eligibility_envelope_floor = float(_floor)

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

                sel_traj = getattr(agent.e3, "_last_selected_trajectory", None)
                if sel_traj is not None:
                    sel_class = _traj_first_action_class(sel_traj)
                    rep_sig = _traj_rep_signature(sel_traj)
                    selected_class_counts[sel_class] = (
                        selected_class_counts.get(sel_class, 0) + 1
                    )
                    per_class_rep_sigs.setdefault(sel_class, Counter())[rep_sig] += 1
                    all_rep_sigs[rep_sig] += 1

                lpfc = getattr(agent, "lateral_pfc", None)
                if lpfc is not None:
                    lb_mean = getattr(lpfc, "_last_bias_abs_mean", None)
                    if isinstance(lb_mean, (int, float)):
                        lateral_pfc_bias_abs_vals.append(float(lb_mean))

                diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
                if bool(diag.get("f_eligibility_demotion_active", False)):
                    demotion_active_ticks += 1
                    env_size = float(diag.get("f_eligibility_envelope_size", -1))
                    if math.isfinite(env_size) and env_size >= 0:
                        demotion_envelope_sizes.append(env_size)
                    excl = float(diag.get("f_eligibility_excluded_count", -1))
                    if math.isfinite(excl) and excl >= 0:
                        demotion_excluded_counts.append(excl)
                    if bool(diag.get("f_eligibility_winner_neq_f_argmin", False)):
                        demotion_winner_neq_f_argmin_ticks += 1
                    if bool(diag.get("f_eligibility_rank_preserving", True)):
                        demotion_rank_preserving_active_ticks += 1

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

        # P1 end-of-episode: REINFORCE update on the SD-033a bias head (trained arms only;
        # the frozen-head silence arm has train_head=False so bias_head_parameters carry
        # no informative gradient signal, but we still guard on train_head to skip it).
        if is_p1 and train_head:
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
                f"  [train] {bank['bank_id']}_{arm['arm_id']} seed={seed} "
                f"phase={phase_label} ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

    # ----- Per-cell aggregation (over P2) -----
    committed_class_entropy = _entropy_from_int_counts(committed_class_counts)
    selected_class_entropy = _entropy_from_int_counts(selected_class_counts)
    frac_pre_ge2 = float(n_p2_pre_ge2 / n_p2_ticks) if n_p2_ticks > 0 else 0.0
    consumed_spread_mean = (
        float(sum(consumed_dists) / len(consumed_dists)) if consumed_dists else 0.0
    )

    qualifying: List[float] = []
    for cls, sig_counter in per_class_rep_sigs.items():
        ent = _entropy_from_counter(sig_counter)
        if sum(sig_counter.values()) >= MIN_TICKS_PER_CLASS:
            qualifying.append(ent)
    mean_within_class_rep_entropy = (
        float(sum(qualifying) / len(qualifying)) if qualifying else 0.0
    )

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

    crf_differentiated = bool(
        crf_n_minted_total_last >= CRF_MIN_MINTED
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

    head_final = _head_weight_vector(agent)
    head_weight_delta = float(torch.norm(head_final - head_init).item())

    demotion_active_frac = (
        float(demotion_active_ticks) / float(n_p2_ticks) if n_p2_ticks > 0 else 0.0
    )
    demotion_excluded_count_mean = (
        float(sum(demotion_excluded_counts) / len(demotion_excluded_counts))
        if demotion_excluded_counts else 0.0
    )
    demotion_envelope_size_mean = (
        float(sum(demotion_envelope_sizes) / len(demotion_envelope_sizes))
        if demotion_envelope_sizes else 0.0
    )
    demotion_rank_preserving_frac = (
        float(demotion_rank_preserving_active_ticks) / float(demotion_active_ticks)
        if demotion_active_ticks > 0 else 0.0
    )
    # R5: post-calibration non-vacuity (envelope active AND actually excluding).
    seed_demotion_non_vacuous = bool(
        demotion_active_frac >= DEMOTION_ACTIVE_FRAC_FLOOR
        and demotion_excluded_count_mean > EXCLUDED_COUNT_FLOOR
    )
    # R6 diagnostics (pre-calibration). default_floor_excluded_count is count-dominated
    # (kept for triage only); the R6 GATE is the count-invariant top-1 merit share.
    default_floor_excluded_count_mean = (
        float(sum(default_floor_excluded_counts) / len(default_floor_excluded_counts))
        if default_floor_excluded_counts else 0.0
    )
    top1_merit_share_mean = (
        float(sum(top1_merit_shares) / len(top1_merit_shares))
        if top1_merit_shares else 0.0
    )
    # R6 structural-divergence flag: the F pool is PEAKED (top-1 merit share above the
    # count-invariant floor) -- divergent-by-construction, not calibration-manufactured
    # and not a candidate-count artifact.
    seed_default_floor_divergent = bool(top1_merit_share_mean > R6_TOP1_MERIT_SHARE_FLOOR)

    seed_class_axis_exercisable = bool(frac_pre_ge2 > FRAC_PRE_GE2_FLOOR)
    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )

    return {
        "bank_id": bank["bank_id"],
        "arm_id": arm["arm_id"],
        "label": f"{bank['bank_id']}_{arm['arm_id']}",
        "seed": int(seed),
        "train_head": train_head,
        "demotion": demotion,
        "divergent_by_construction": bool(bank["divergent_by_construction"]),
        "candidate_summary_source": bank["candidate_summary_source"],
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
        # ----- R1 / R2 readiness -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "class_axis_exercisable": seed_class_axis_exercisable,
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        # ----- R3 CRF + head maturity -----
        "crf_mean_n_active": round(mean_crf_n_active, 6),
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_max_pairwise_rule_dist": round(crf_max_pairwise_rule_dist_max, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "crf_differentiated": crf_differentiated,
        "crf_mean_n_matched": round(mean_crf_n_matched, 6),
        "crf_max_n_matched": int(max_crf_n_matched),
        "head_weight_delta_norm": round(head_weight_delta, 8),
        # ----- R4 propagation non-vacuity -----
        "mean_lateral_pfc_bias_abs": round(mean_lateral_pfc_bias_abs, 8),
        "mean_prop_counterfactual_delta": round(mean_prop_counterfactual_delta, 8),
        # ----- R5 MECH-448 demotion non-vacuity (post-calibration) -----
        "f_eligibility_demotion_active_ticks": int(demotion_active_ticks),
        "f_eligibility_demotion_active_frac": round(demotion_active_frac, 6),
        "f_eligibility_excluded_count_mean": round(demotion_excluded_count_mean, 6),
        "f_eligibility_envelope_size_mean": round(demotion_envelope_size_mean, 6),
        "f_eligibility_winner_neq_f_argmin_ticks": int(demotion_winner_neq_f_argmin_ticks),
        "f_eligibility_rank_preserving_frac": round(demotion_rank_preserving_frac, 6),
        "demotion_non_vacuous": seed_demotion_non_vacuous,
        # ----- R6 divergent-by-construction (pre-calibration) -----
        # top1_merit_share_mean is the count-invariant R6 GATE statistic;
        # default_floor_excluded_count_mean is a count-dominated diagnostic only.
        "top1_merit_share_mean": round(top1_merit_share_mean, 6),
        "default_floor_excluded_count_mean": round(default_floor_excluded_count_mean, 6),
        "default_floor_divergent": seed_default_floor_divergent,
        # ----- SECONDARY negative control (NOT load-bearing) -----
        "mean_within_class_rep_entropy_nats": round(mean_within_class_rep_entropy, 6),
        "n_distinct_rep_signatures_total": int(len(all_rep_sigs)),
        "selected_class_entropy_nats": round(selected_class_entropy, 6),
    }


# ---------------------------------------------------------------------------
# Aggregation + interpretation
# ---------------------------------------------------------------------------

ARM_TEST = "ARM_trained_demotion_on"
ARM_CEILING = "ARM_trained_demotion_off"
ARM_SILENCE = "ARM_frozen_demotion_on"


def _cells(rows: List[Dict[str, Any]], bank_id: str, arm_id: str) -> List[Dict[str, Any]]:
    return [
        r for r in rows
        if r["bank_id"] == bank_id and r["arm_id"] == arm_id and r["error_note"] is None
    ]


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _evaluate_bank(bank: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-bank readiness + C_PRIMARY. 'Converts' iff C_PRIMARY holds AND readiness met."""
    bank_id = bank["bank_id"]
    test = _cells(rows, bank_id, ARM_TEST)
    ceiling = _cells(rows, bank_id, ARM_CEILING)
    silence = _cells(rows, bank_id, ARM_SILENCE)
    all_arms = [test, ceiling, silence]
    demotion_on_arms = [test, silence]

    # R1: committed-class axis exercisable, all arms, >= 2/3 seeds each.
    r1_holds = all(
        sum(1 for r in arm if r["class_axis_exercisable"]) >= MIN_SEEDS_FOR_PASS
        for arm in all_arms
    )
    # R2: consumed-summary divergence real, all arms, >= 2/3 seeds each.
    r2_holds = all(
        sum(1 for r in arm if r["gapa_divergence"]) >= MIN_SEEDS_FOR_PASS
        for arm in all_arms
    )
    # R3: trained arms matured (CRF differentiated) AND bias head trained
    # (head_weight_delta > floor) on trained-on (TEST) seeds, >= 2/3.
    r3_seeds = sum(
        1 for r in test
        if r["crf_differentiated"] and r["head_weight_delta_norm"] > HEAD_DELTA_MIN
    )
    r3_holds = bool(r3_seeds >= MIN_SEEDS_FOR_PASS)
    # R4: propagation non-vacuity -- paired |bias_abs(TEST) - bias_abs(SILENCE)| > floor.
    test_bias_by_seed = {int(r["seed"]): r["mean_lateral_pfc_bias_abs"] for r in test}
    sil_bias_by_seed = {int(r["seed"]): r["mean_lateral_pfc_bias_abs"] for r in silence}
    prop_diff_by_seed: Dict[int, float] = {}
    n_prop_nonvac = 0
    for seed in sorted(set(test_bias_by_seed) & set(sil_bias_by_seed)):
        diff = abs(test_bias_by_seed[seed] - sil_bias_by_seed[seed])
        prop_diff_by_seed[seed] = round(diff, 8)
        if diff > PROP_NONVAC_FLOOR:
            n_prop_nonvac += 1
    r4_holds = bool(n_prop_nonvac >= MIN_SEEDS_FOR_PASS)
    # R5: MECH-448 demotion non-vacuity (post-calibration) on demotion-ON arms, >= 2/3 each.
    r5_holds = all(
        sum(1 for r in arm if r["demotion_non_vacuous"]) >= MIN_SEEDS_FOR_PASS
        for arm in demotion_on_arms
    )
    # R6: bankC-only -- default-floor (pre-calibration) divergence on demotion-ON arms,
    # >= 2/3 each. gapa (positive control) is NOT required to be default-floor divergent.
    if bool(bank["divergent_by_construction"]):
        r6_holds = all(
            sum(1 for r in arm if r["default_floor_divergent"]) >= MIN_SEEDS_FOR_PASS
            for arm in demotion_on_arms
        )
        r6_applicable = True
    else:
        r6_holds = True   # not applicable to the positive-control anchor
        r6_applicable = False

    readiness_ok = bool(r1_holds and r2_holds and r3_holds and r4_holds and r5_holds and r6_holds)

    # C_PRIMARY: TEST committed-class entropy STRICTLY ABOVE BOTH CEILING AND SILENCE by
    # margin, paired by seed, >= 2/3 seeds.
    ceil_by_seed = {int(r["seed"]): r["committed_class_entropy_nats"] for r in ceiling}
    sil_by_seed = {int(r["seed"]): r["committed_class_entropy_nats"] for r in silence}
    test_by_seed = {int(r["seed"]): r["committed_class_entropy_nats"] for r in test}
    paired_lifts: Dict[int, Dict[str, float]] = {}
    n_lift = 0
    for seed in sorted(set(test_by_seed) & set(ceil_by_seed) & set(sil_by_seed)):
        lift_vs_ceiling = test_by_seed[seed] - ceil_by_seed[seed]
        lift_vs_silence = test_by_seed[seed] - sil_by_seed[seed]
        paired_lifts[seed] = {
            "vs_ceiling": round(lift_vs_ceiling, 6),
            "vs_silence": round(lift_vs_silence, 6),
        }
        if lift_vs_ceiling >= C2_LIFT_MARGIN_NATS and lift_vs_silence >= C2_LIFT_MARGIN_NATS:
            n_lift += 1
    c_primary_holds = bool(n_lift >= C2_MIN_LIFT_SEEDS)

    converts = bool(readiness_ok and c_primary_holds)

    return {
        "bank_id": bank_id,
        "is_positive_control": bool(bank["is_positive_control"]),
        "divergent_by_construction": bool(bank["divergent_by_construction"]),
        "readiness_ok": readiness_ok,
        "R1_class_axis_exercisable": r1_holds,
        "R2_consumed_divergence": r2_holds,
        "R3_trained_matured_head_trained": r3_holds,
        "R3_n_seeds": int(r3_seeds),
        "R4_propagation_non_vacuity": r4_holds,
        "R4_n_prop_nonvac_seeds": int(n_prop_nonvac),
        "R4_prop_diff_by_seed": prop_diff_by_seed,
        "R5_demotion_non_vacuous_post_calibration": r5_holds,
        "R6_applicable": r6_applicable,
        "R6_default_floor_divergent": r6_holds,
        "c_primary_holds": c_primary_holds,
        "c_primary_n_lift_seeds": int(n_lift),
        "c_primary_paired_lifts_by_seed": paired_lifts,
        "converts": converts,
        # Diagnostics
        "test_mean_committed_class_entropy": round(
            _mean([r["committed_class_entropy_nats"] for r in test]), 6
        ),
        "ceiling_mean_committed_class_entropy": round(
            _mean([r["committed_class_entropy_nats"] for r in ceiling]), 6
        ),
        "silence_mean_committed_class_entropy": round(
            _mean([r["committed_class_entropy_nats"] for r in silence]), 6
        ),
        "test_top1_merit_share_mean": round(
            _mean([r["top1_merit_share_mean"] for r in test]), 6
        ),
        "silence_top1_merit_share_mean": round(
            _mean([r["top1_merit_share_mean"] for r in silence]), 6
        ),
        "r6_top1_merit_share_floor": float(R6_TOP1_MERIT_SHARE_FLOOR),
        "test_default_floor_excluded_count_mean": round(
            _mean([r["default_floor_excluded_count_mean"] for r in test]), 6
        ),
        "test_post_calib_excluded_count_mean": round(
            _mean([r["f_eligibility_excluded_count_mean"] for r in test]), 6
        ),
        "test_top1_merit_share_by_seed": {
            int(r["seed"]): r["top1_merit_share_mean"] for r in test
        },
    }


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

    for bank in BANKS:
        print(
            f"Bank {bank['bank_id']} ({bank['label']}) "
            f"(P0={p0_episodes} P1={p1_episodes} P2={p2_episodes} "
            f"steps={steps_per_episode} dry_run={dry_run})",
            flush=True,
        )
        for arm in ARMS:
            for s in seeds:
                print(
                    f"Seed {s} Condition {bank['bank_id']}_{arm['arm_id']}",
                    flush=True,
                )
                row = _run_cell(
                    bank, arm, s, p0_episodes, p1_episodes, p2_episodes,
                    steps_per_episode,
                )
                # Arm fingerprint. Reset RNG already happened at cell entry
                # (reset_all_rng inside _run_cell). The OFF/baseline arms (the demotion
                # -OFF ceiling arm + the frozen silence arm) are minted reuse-ELIGIBLE:
                # rng_fully_reset=True + config_slice declared. Online e2 + P1 REINFORCE
                # are stateful per-cell but self-contained (reset each cell), so no
                # cross-arm ineligibility applies.
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice={
                        "bank_id": bank["bank_id"],
                        "arm_id": arm["arm_id"],
                        "train_head": bool(arm["train_head"]),
                        "demotion": bool(arm["demotion"]),
                        "use_candidate_rule_field": True,
                        "candidate_summary_source": bank["candidate_summary_source"],
                        "crf_persist_rules_across_episode_reset": True,
                        "crf_mature_pool_dynamics": True,
                        "crf_context_from_e2_world_forward": True,
                        "crf_availability_maintenance": True,
                        "crf_maintenance_floor": float(CRF_MAINTENANCE_FLOOR),
                        "crf_maintenance_decay": float(CRF_MAINTENANCE_DECAY),
                        "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
                        "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
                        "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
                        "use_f_eligibility_demotion": bool(arm["demotion"]),
                        "f_eligibility_envelope_floor": float(F_ELIGIBILITY_ENVELOPE_FLOOR),
                        "f_eligibility_dn_sigma": float(F_ELIGIBILITY_DN_SIGMA),
                        "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
                        "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
                        "modulatory_channel_route_source": str(MODULATORY_CHANNEL_ROUTE_SOURCE),
                        "env_kwargs": dict(bank["env_kwargs"]),
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
                )
                arm_results.append(row)
                verdict = "PASS" if row["error_note"] is None else "FAIL"
                print(f"verdict: {verdict}", flush=True)

    bank_eval = {b["bank_id"]: _evaluate_bank(b, arm_results) for b in BANKS}
    gapa = bank_eval["gapa_foraging"]
    bankc = bank_eval["bankC_divergent"]

    # ----- Outcome map (self-route; PROMOTES NOTHING; MECH-448 stays provisional) -----
    gapa_converts = bool(gapa["converts"])
    bankc_converts = bool(bankc["converts"])
    bankc_readiness = bool(bankc["readiness_ok"])

    if not gapa_converts:
        # positive control failed to reproduce 689d (or gapa readiness unmet) -> the
        # whole run is uninterpretable; the harness is not reproducing the GAP-A PASS.
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif not bankc_readiness:
        # gapa converts but bankC readiness (esp R5/R6) unmet -> the cross-bank arm never
        # ran a genuinely-firing divergent-F test; re-queue with a better-tuned bankC.
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif bankc_converts:
        # gapa converts AND bankC converts -> the demotion lever CONVERTS on a THIRD,
        # divergent-by-construction behavioural bank; persistence was tied to the
        # spread-F / calibration crutch. supports MECH-448 / MECH-309 / ARC-062 generality.
        outcome = "PASS"
        direction = "supports"
        label = "demotion_converts_cross_bank_divergent_F"
    else:
        # gapa converts AND bankC does NOT convert (readiness incl R6 MET) -> the
        # conversion ceiling is GENERAL, not a spread/calibration artifact. MECH-448
        # stays provisional; route MECH-449 / V4. NOT a falsification (persistence).
        outcome = "FAIL"
        direction = "non_contributory"
        label = "conversion_ceiling_general_persists_on_divergent_F_third_bank"

    # Per-claim direction. On the supports branch all three are "supports"; otherwise all
    # non_contributory (persistence / not-ready) -- never a weakens (PROMOTES NOTHING).
    evidence_direction_per_claim = {
        "MECH-309": direction,
        "ARC-062": direction,
        "MECH-448": direction,
    }

    # ----- interpretation block (falsifiable; preconditions[] + criteria_non_degenerate
    # + criteria[] with C_PRIMARY load_bearing:true) -----
    def _min_over(bank_id: str, key: str) -> float:
        vals = [r[key] for r in arm_results if r["bank_id"] == bank_id and r["error_note"] is None]
        return float(min(vals)) if vals else 0.0

    def _max_over(bank_id: str, key: str) -> float:
        vals = [r[key] for r in arm_results if r["bank_id"] == bank_id and r["error_note"] is None]
        return float(max(vals)) if vals else 0.0

    preconditions = [
        {
            "name": "gapa_positive_control_committed_class_axis_exercisable",
            "kind": "readiness",
            "description": (
                "gapa (positive-control anchor) committed-class axis exercisable "
                "(frac of P2 ticks with >= 2 candidate first-action classes > floor) "
                "on a majority of seeds in all arms."
            ),
            "control": "gapa reef-bipartite SP-CEM multi-class candidate pool (689d ENV_KWARGS)",
            "measured": _min_over("gapa_foraging", "frac_pre_ge2"),
            "threshold": float(FRAC_PRE_GE2_FLOOR),
            "comparator": ">",
            "met": bool(gapa["R1_class_axis_exercisable"]),
        },
        {
            "name": "gapa_positive_control_mech448_demotion_live_and_excluding",
            "kind": "readiness",
            "description": (
                "gapa MECH-448 f_demotion envelope active on >= floor of P2 ticks AND "
                "actually excludes (mean f_eligibility_excluded_count > floor, "
                "post-calibration) on demotion-ON arms, majority of seeds. All-admit "
                "(excluded_count==0) => substrate_not_ready_requeue."
            ),
            "control": "gapa demotion-ON arms f_demotion diagnostics",
            "measured": _min_over("gapa_foraging", "f_eligibility_excluded_count_mean"),
            "threshold": float(EXCLUDED_COUNT_FLOOR),
            "comparator": ">",
            "met": bool(gapa["R5_demotion_non_vacuous_post_calibration"]),
        },
        {
            "name": "gapa_positive_control_converts_reproduces_689d",
            "kind": "readiness",
            "description": (
                "gapa C_PRIMARY holds: ARM_trained_demotion_on committed-class entropy "
                "strictly above BOTH ARM_trained_demotion_off AND ARM_frozen_demotion_on "
                "by margin on a majority of seeds. This is the anchor that the harness "
                "reproduces the V3-EXQ-689d GAP-A conversion. gapa NOT converting => the "
                "whole run is uninterpretable => substrate_not_ready_requeue."
            ),
            "control": "gapa 3-arm dissociation committed-class entropy",
            "measured": float(gapa["c_primary_n_lift_seeds"]),
            "threshold": float(C2_MIN_LIFT_SEEDS),
            "direction": "lower",
            "met": bool(gapa["converts"]),
        },
        {
            "name": "bankC_divergent_by_construction_top1_merit_share_R6",
            "kind": "readiness",
            "description": (
                "bankC (concentrated-hazard) F pool is PEAKED by construction: the "
                "COUNT-INVARIANT top-1 merit share (fraction of total merit on the single "
                "F-best candidate) clears R6_TOP1_MERIT_SHARE_FLOOR on demotion-ON arms, "
                "majority of seeds -- a threshold the gapa positive control does NOT meet. "
                "This is THE structural-divergence check: it proves the pool fires the "
                "envelope from real peakedness, NOT the calibration crutch and NOT the "
                "candidate-count artifact (raw excluded_count under an absolute floor is "
                "count-dominated with ~32 SP-CEM candidates, so it is a diagnostic only, "
                "not the gate). R6 fail => bankC's F pool is NOT divergent-by-construction "
                "(the concentrated-hazard tuning did not peak the per-candidate F) => "
                "substrate_not_ready_requeue for the cross-bank arm, NEVER a weakens."
            ),
            "control": "bankC demotion-ON arms top-1 merit share (pre-calibration, count-invariant)",
            "measured": _min_over("bankC_divergent", "top1_merit_share_mean"),
            "threshold": float(R6_TOP1_MERIT_SHARE_FLOOR),
            "comparator": ">",
            "met": bool(bankc["R6_default_floor_divergent"]),
        },
        {
            "name": "bankC_mech448_demotion_live_and_excluding",
            "kind": "readiness",
            "description": (
                "bankC MECH-448 f_demotion envelope active on >= floor of P2 ticks AND "
                "actually excludes (post-calibration belt-and-suspenders) on demotion-ON "
                "arms, majority of seeds. All-admit => substrate_not_ready_requeue for "
                "the cross-bank arm."
            ),
            "control": "bankC demotion-ON arms f_demotion diagnostics",
            "measured": _min_over("bankC_divergent", "f_eligibility_excluded_count_mean"),
            "threshold": float(EXCLUDED_COUNT_FLOOR),
            "comparator": ">",
            "met": bool(bankc["R5_demotion_non_vacuous_post_calibration"]),
        },
        {
            "name": "bankC_propagation_non_vacuity_trained_vs_frozen",
            "kind": "readiness",
            "description": (
                "bankC paired-by-seed |mean_lateral_pfc_bias_abs(ARM_trained_demotion_on) "
                "- mean_lateral_pfc_bias_abs(ARM_frozen_demotion_on)| > floor on a "
                "majority of seeds -- the trained informative channel produces a "
                "different bias than the frozen zeroed head. Below-floor => vacuous "
                "propagation => substrate_not_ready_requeue."
            ),
            "control": "bankC paired TEST vs SILENCE mean lateral_pfc bias",
            "measured": float(max(list(bankc["R4_prop_diff_by_seed"].values()) or [0.0])),
            "threshold": float(PROP_NONVAC_FLOOR),
            "comparator": ">",
            "met": bool(bankc["R4_propagation_non_vacuity"]),
        },
    ]

    criteria = [
        {
            "name": "C_PRIMARY_bankC_committed_class_entropy_lift_vs_both_controls",
            "load_bearing": True,
            "passed": bool(bankc["c_primary_holds"]),
        },
        {
            "name": "C_PRIMARY_gapa_positive_control_committed_class_entropy_lift",
            "load_bearing": True,
            "passed": bool(gapa["c_primary_holds"]),
        },
    ]

    criteria_non_degenerate = {
        "gapa_R1_class_axis_exercisable": bool(gapa["R1_class_axis_exercisable"]),
        "gapa_R2_consumed_divergence": bool(gapa["R2_consumed_divergence"]),
        "gapa_R3_trained_matured_head_trained": bool(gapa["R3_trained_matured_head_trained"]),
        "gapa_R4_propagation_non_vacuity": bool(gapa["R4_propagation_non_vacuity"]),
        "gapa_R5_demotion_non_vacuous": bool(gapa["R5_demotion_non_vacuous_post_calibration"]),
        "gapa_converts": bool(gapa["converts"]),
        "bankC_R1_class_axis_exercisable": bool(bankc["R1_class_axis_exercisable"]),
        "bankC_R2_consumed_divergence": bool(bankc["R2_consumed_divergence"]),
        "bankC_R3_trained_matured_head_trained": bool(bankc["R3_trained_matured_head_trained"]),
        "bankC_R4_propagation_non_vacuity": bool(bankc["R4_propagation_non_vacuity"]),
        "bankC_R5_demotion_non_vacuous_post_calibration": bool(bankc["R5_demotion_non_vacuous_post_calibration"]),
        "bankC_R6_default_floor_divergent_by_construction": bool(bankc["R6_default_floor_divergent"]),
        "bankC_readiness_ok": bool(bankc["readiness_ok"]),
        "bankC_c_primary_holds": bool(bankc["c_primary_holds"]),
    }

    interpretation = {
        "label": label,
        "preconditions": preconditions,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "note": (
            "Cross-bank generality falsifier for the MECH-448 demotion lever's "
            "off-substrate PERSISTENCE. gapa (positive-control anchor, 689d ENV_KWARGS) "
            "must CONVERT (reproduce the GAP-A PASS) for the run to be interpretable. "
            "bankC (divergent-by-construction, R6 default-floor excluded_count>0) tests "
            "whether the demotion lever converts on a THIRD behavioural bank whose F pool "
            "fires the envelope WITHOUT the calibration crutch. bankC-converts => "
            "supports generality; bankC-persists (readiness incl R6 met) => the ceiling "
            "is GENERAL, MECH-448 stays provisional, route MECH-449/V4 (persistence, NOT "
            "a falsification). ANY readiness fail self-routes substrate_not_ready_requeue. "
            "PROMOTES NOTHING by itself in all branches."
        ),
    }

    total_cells = len(BANKS) * len(ARMS) * len(seeds)
    total_completed = sum(1 for r in arm_results if r["error_note"] is None)

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": seeds,
        "n_banks": len(BANKS),
        "n_arms": len(ARMS),
        "total_cells_attempted": int(total_cells),
        "total_cells_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "episodes_per_run_per_cell": int(p0_episodes + p1_episodes + p2_episodes),
        "gapa_converts": gapa_converts,
        "bankC_converts": bankc_converts,
        "bankC_readiness_ok": bankc_readiness,
        "bank_evaluation": bank_eval,
        "decision_rule_thresholds": {
            "h_signature": int(H_SIGNATURE),
            "c2_lift_margin_nats": float(C2_LIFT_MARGIN_NATS),
            "c2_min_lift_seeds": int(C2_MIN_LIFT_SEEDS),
            "frac_pre_ge2_floor": float(FRAC_PRE_GE2_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "consumed_magnitude_ceil": float(CONSUMED_MAGNITUDE_CEIL),
            "crf_min_minted": int(CRF_MIN_MINTED),
            "crf_frac_active_floor": float(CRF_FRAC_ACTIVE_FLOOR),
            "head_delta_min": float(HEAD_DELTA_MIN),
            "prop_nonvac_floor": float(PROP_NONVAC_FLOOR),
            "demotion_active_frac_floor": float(DEMOTION_ACTIVE_FRAC_FLOOR),
            "excluded_count_floor": float(EXCLUDED_COUNT_FLOOR),
            "r6_top1_merit_share_floor": float(R6_TOP1_MERIT_SHARE_FLOOR),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "f_eligibility_envelope_floor_default": float(F_ELIGIBILITY_ENVELOPE_FLOOR),
            "f_eligibility_dn_sigma": float(F_ELIGIBILITY_DN_SIGMA),
            "envelope_keep_min": int(ENVELOPE_KEEP_MIN),
            "envelope_keep_max": int(ENVELOPE_KEEP_MAX),
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
        "schema_version": "v1",
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
        "non_degenerate": bool(
            result["bank_evaluation"]["gapa_foraging"]["readiness_ok"]
            and result["bank_evaluation"]["bankC_divergent"]["readiness_ok"]
        ),
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "evidence_direction_note": (
            f"V3-EXQ-722 CROSS-BANK GENERALITY falsifier for the "
            f"f_dominance_conversion_ceiling SELECTION face (claims MECH-309, ARC-062, "
            f"MECH-448). Tests whether the MECH-448 rank-preserving F->eligibility "
            f"demotion lever's off-substrate PERSISTENCE (V3-EXQ-485j OFC bank + "
            f"V3-EXQ-654i arc_062 bank, both on SPREAD-F pools rescued by envelope-floor "
            f"calibration) is GENERAL or a spread-F/calibration artifact. Grid = 2 banks "
            f"x 3 arms x 3 seeds = 18 cells. gapa_foraging = the V3-EXQ-689d reef-bipartite "
            f"ENV_KWARGS VERBATIM (positive-control anchor; the substrate where the lever "
            f"is KNOWN to CONVERT, committed entropy 0.938 ON vs 0.371 OFF). bankC_divergent "
            f"= a THIRD decisive-threat forced-choice behavioural bank on a CONCENTRATED "
            f"HIGH-HAZARD non-reef regime tuned so the per-candidate harm-cost F is PEAKED "
            f"=> the MECH-448 envelope excludes a non-empty tail under the DEFAULT floor "
            f"({F_ELIGIBILITY_ENVELOPE_FLOOR}) WITHOUT calibration (R6). Arms = the 485j "
            f"3-way dissociation (head trained?) x (demotion on?): "
            f"ARM_trained_demotion_on (TEST), ARM_trained_demotion_off (F-dominance "
            f"CEILING), ARM_frozen_demotion_on (SILENCE). Everything else matched. "
            f"PRIMARY DV = committed_class_entropy_nats. C_PRIMARY (LOAD-BEARING, per "
            f"bank) = TEST strict-above BOTH controls by {C2_LIFT_MARGIN_NATS} nats on "
            f">= {C2_MIN_LIFT_SEEDS}/3 seeds; 'converts' iff C_PRIMARY holds AND readiness "
            f"met. READINESS R1..R6 (per bank; ANY fail => that bank self-routes "
            f"substrate_not_ready, NEVER a weakens; R6 bankC-only default-floor "
            f"divergence). Outcome branches: gapa fails to converge OR gapa readiness "
            f"unmet => substrate_not_ready_requeue (harness not reproducing 689d); gapa "
            f"converts + bankC readiness unmet => substrate_not_ready_requeue "
            f"(cross-bank arm); gapa converts + bankC converts => "
            f"demotion_converts_cross_bank_divergent_F (SUPPORTS generality; the "
            f"persistence was tied to spread-F/calibration); gapa converts + bankC "
            f"persists (readiness incl R6 met) => "
            f"conversion_ceiling_general_persists_on_divergent_F_third_bank "
            f"(non_contributory PERSISTENCE, NOT a falsification; MECH-448 stays "
            f"provisional; route MECH-449 Go/No-Go constitution / V4). evidence_direction "
            f"per branch: supports on the converts branch, non_contributory otherwise; "
            f"NEVER a weakens. PROMOTES NOTHING by itself in ALL branches; MECH-448 stays "
            f"provisional. gapa_converts={result['gapa_converts']}, "
            f"bankC_converts={result['bankC_converts']}, "
            f"bankC_readiness_ok={result['bankC_readiness_ok']}. "
            f"interpretation_label={result['interpretation_label']}."
        ),
        "dry_run": bool(dry_run),
        "banks": [
            {
                "bank_id": b["bank_id"],
                "label": b["label"],
                "divergent_by_construction": bool(b["divergent_by_construction"]),
                "is_positive_control": bool(b["is_positive_control"]),
                "candidate_summary_source": b["candidate_summary_source"],
                "env_kwargs": dict(b["env_kwargs"]),
            }
            for b in BANKS
        ],
        "arms": [
            {
                "arm_id": a["arm_id"],
                "label": a["label"],
                "train_head": bool(a["train_head"]),
                "demotion": bool(a["demotion"]),
            }
            for a in ARMS
        ],
        "config_summary": {
            "grid": "2 banks x 3 arms x 3 seeds = 18 cells (dry-run: 1 seed = 6 cells)",
            "primary_dv": "committed_class_entropy_nats",
            "c_primary_load_bearing": True,
            "conversion_lever": (
                "MECH-448 (ARC-107) rank-preserving F->eligibility demotion (graded DN "
                "envelope; f_demotion mode overrides 569i top_k) -- the lever V3-EXQ-689d "
                "PASSed on GAP-A"
            ),
            "positive_control_anchor": "gapa_foraging (689d reef-bipartite ENV_KWARGS verbatim)",
            "third_bank_divergent_by_construction": "bankC_divergent (concentrated high-hazard, reef OFF)",
            "matched_stack": (
                "SP-CEM + candidate_summary_source=e2_world_forward (e2 trained online in "
                "P0, SD-056) + modulatory-selection-authority (643a) + channel routing + "
                "MECH-448 demotion (swept) + MECH-341 stratified + MECH-313 noise floor + "
                "V_s minimal + use_gated_policy + use_lateral_pfc_analog + "
                "use_candidate_rule_field (matched constant, CRF stack) + envelope-floor "
                "calibration (belt-and-suspenders)"
            ),
            "swept_factors": "(train_head?) x (demotion?) per the 485j 3-way dissociation",
            "phases": "P0 e2-train (field matures) -> P1 frozen-encoder bias-head REINFORCE -> P2 frozen measurement",
            "e2_trained_in_p0_frozen_in_p1_p2": True,
            "envelope_floor_calibration_kept": True,
            "r6_default_floor_divergence_probe": (
                "bankC records default_floor_excluded_count (pre-calibration) to prove "
                "STRUCTURAL divergence, not calibration-manufactured"
            ),
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-722 cross-bank conversion-ceiling SELECTION falsifier "
            "(MECH-448 demotion generality; gapa positive control + bankC divergent-by-construction)"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_cells_completed']}/{result['total_cells_attempted']} "
        f"gapa_converts={result['gapa_converts']} "
        f"bankC_converts={result['bankC_converts']} "
        f"bankC_readiness={result['bankC_readiness_ok']} "
        f"label={result['interpretation_label']}",
        flush=True,
    )

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
    sys.exit(0)
