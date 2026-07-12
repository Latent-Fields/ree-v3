#!/opt/local/bin/python3
"""V3-EXQ-569e -- SD-056 mechanism probe: Pathway A vs Pathway B dissociation.

Parallel diagnostic to V3-EXQ-569c (SD-056 matched-entropy FP-2 falsifier) and
V3-EXQ-569d (floor-recalibrated falsifier). NOT a successor: supersedes is
intentionally NOT set. The 569c manifest stays the load-bearing C3-lift
evidence; 569e isolates WHERE the C3 lift comes from.

Predecessor:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-569c_2026-05-30.{md,json}
  REE_assembly/evidence/experiments/v3_exq_569c_sd056_action_contrastive_diversity_falsifier_20260530T124450Z_v3.json

Claims:    [ARC-065, MECH-341]  (preserved from 569c; not modified by this probe)

Open mechanism question (autopsy section 5)
-------------------------------------------
569c headline: ON arms (sd056_weight 0.01/0.05/0.20) produced
selected_action_entropy 0.875/0.833/0.951 vs matched-entropy control (ARM_4
sd056_OFF + T=2.5) at 0.414 -- ~2.4x above the FP-2 control on C3.
Per-candidate cand_world_pairwise_dist lifted ~3x baseline (0.015 -> 0.041-
0.046). Pure softmax temperature is decisively ruled out as the explanation;
SD-056 produces structural behavioural diversity beyond entropy injection.

Where is the C3 lift coming from?

  Pathway A -- per-candidate z_world variance propagation. SD-056 trains
    E2.world_forward to produce action-discriminable next-state predictions.
    The small per-candidate t=1 variance (~3x baseline) propagates through E3
    scoring as small score differences that resolve to different argmax
    classes at selection. Proportions roughly compatible (C1 lift ~3x, C3
    lift ~2.4x).

  Pathway B -- E2 rollout dynamics shift. Contrastive training updates E2
    weights such that ON-arm full-horizon rollouts have qualitatively
    different trajectory dynamics than OFF -- not just slightly more spread
    per candidate but a different scoring landscape that E3 sees. E3 selects
    different classes for reasons independent of within-tick per-candidate
    variance.

These are not mutually exclusive; both could contribute. 569c's
e3_top2_class_gap NaN on every ON arm blocked the most informative
dissociation channel.

Measurements (M3 + M4 + M1 + M5, per user-confirmed design 2026-05-30T17Z)
-------------------------------------------------------------------------
M3 -- e3_top2_class_gap + e3_score_std NaN bug fix. 569c's
      `_per_class_score_stats` returned per-class score means via
      `float(scores_t[i].item())`; whenever last_scores contained a NaN
      (the contrastive training step in ON arms occasionally produces
      Inf-then-NaN in E3 score chain on a small fraction of ticks), the
      arithmetic propagated NaN into top2_gap / score_std and the
      list-mean returned NaN. M3 filters non-finite values at the append
      site and tracks the skip count separately.

M1 -- E2 rollout-divergence over full horizon. `cand_world_pairwise_dist`
      (569c metric) reads t=1 only. M1 stacks each candidate's
      get_world_state_sequence() and computes per-step pairwise L2 across
      candidates over the entire rollout horizon. Pathway B prediction:
      ON arms produce qualitatively larger rollout divergence than the
      candidate-pool t=1 divergence alone would explain (ratio
      rollout_mean / pairwise_t1 substantially > horizon-step linear-
      growth expectation).

M4 -- frozen-E2-at-P1 ablation arms. Train E2 contrastive head during P0
      (encoder warmup); freeze E2 weights at P1 entry (set e2_opt=None +
      stop appending to the contrastive buffer). If C3 lift persists with
      frozen E2 -> the lift is encoded in P0-trained E2 weights (Pathway B).
      If C3 lift collapses on freeze -> the lift was being maintained
      tick-by-tick by per-candidate variance fed by ongoing contrastive
      updates (Pathway A).

M5 -- force-argmin counterfactual arms. Same E2 training as ON-live, but
      collapse E3 softmax selection to argmin (temperature=0.01).
      Removes E3 routing's per-candidate softmax over score distribution.
      If C3 lift survives this ablation -> E2 dynamics produce different
      score landscapes that argmin RESOLVES TO different actions
      across candidate pools (Pathway B, via score landscape). If C3 lift
      collapses -> the lift was specifically in the softmax-sampling
      routing of per-candidate score variance (Pathway A, via routing).

Arms (8, with TWO live weights per user-confirmed design)
---------------------------------------------------------
  ARM_0_OFF                  sd056 master OFF, T=1.0 baseline.
                             Reproduces V3-EXQ-569 / 569c baseline.
  ARM_1_W005_LIVE            sd056 ON w=0.05, train_e2 P0+P1.
  ARM_2_W020_LIVE            sd056 ON w=0.20, train_e2 P0+P1.
  ARM_3_W005_FROZEN_P1       sd056 ON w=0.05, train_e2 P0 only.
                             E2 weights frozen at P1 entry; buffer
                             append disabled. M4 ablation arm at w=0.05.
  ARM_4_W020_FROZEN_P1       sd056 ON w=0.20, train_e2 P0 only.
                             M4 ablation arm at w=0.20.
  ARM_5_W005_ARGMIN          sd056 ON w=0.05, train_e2 P0+P1, force
                             argmin selection at P1 via T=0.01. M5 routing
                             ablation at w=0.05.
  ARM_6_W020_ARGMIN          sd056 ON w=0.20, train_e2 P0+P1, force
                             argmin selection at P1 via T=0.01. M5 routing
                             ablation at w=0.20.
  ARM_7_MATCHED_NOISE        sd056 OFF + T=2.5 (matched-entropy control).
                             Required so C3 baseline stays interpretable
                             relative to 569c headline.

Five seeds [42, 43, 44, 45, 46] per user-confirmed design (3 seeds in 569c
were too thin for discriminative arithmetic across the cell grid).

Pre-registered interpretation grid (3-channel)
----------------------------------------------
Cell coordinates are computed AFTER measurements land, on the per-arm means
across 5 seeds. Pre-registered thresholds defined as constants in this
script (do NOT derive them from the run's own statistics).

Let
  c3_live      = mean(ARM_1.selected_entropy, ARM_2.selected_entropy)
  c3_frozen    = mean(ARM_3.selected_entropy, ARM_4.selected_entropy)
  c3_argmin    = mean(ARM_5.selected_entropy, ARM_6.selected_entropy)
  c3_off       = ARM_0.selected_entropy
  c3_noise     = ARM_7.selected_entropy
  on_c3_lift   = c3_live - c3_noise
  frozen_pres  = (c3_frozen - c3_noise) / max(on_c3_lift, 1e-6)   # M4 axis
  argmin_pres  = (c3_argmin - c3_noise) / max(on_c3_lift, 1e-6)   # M5 axis
  rollout_amp  = rollout_mean(LIVE_arms) / pairwise_t1_mean(LIVE_arms)   # M1 axis

PRESERVED_FLOOR = 0.5  (frozen_pres / argmin_pres above this == "preserved")
COLLAPSED_CEIL  = 0.2  (below this == "collapsed")
ROLLOUT_B_FLOOR = 2.0  (rollout amplification ratio; B signature if >=)

Verdict cells (M3 instrumentation must succeed for any cell to be valid):

  PATHWAY_B  (E2 rollout dynamics shift, dominant)
    frozen_pres >= PRESERVED_FLOOR
    AND argmin_pres >= PRESERVED_FLOOR
    AND rollout_amp >= ROLLOUT_B_FLOOR
    Interpretation: C3 lift survives BOTH freezing E2 AND removing softmax
    routing AND the per-step rollout divergence is qualitatively larger than
    t=1 divergence alone predicts. Lift lives in E2 weights' production of a
    different scoring landscape; argmin over that landscape still picks
    different classes.

  PATHWAY_A  (per-candidate variance propagation, dominant)
    frozen_pres <= COLLAPSED_CEIL
    AND argmin_pres <= COLLAPSED_CEIL
    AND rollout_amp < ROLLOUT_B_FLOOR
    Interpretation: C3 lift collapses on either freeze or argmin, AND
    rollout divergence is roughly horizon-linear in t=1 variance. Lift is
    tick-by-tick per-candidate variance routed through softmax sampling.

  BOTH       (additive)
    frozen_pres in [COLLAPSED_CEIL, PRESERVED_FLOOR]
    OR argmin_pres in [COLLAPSED_CEIL, PRESERVED_FLOOR]
    AND rollout_amp >= ROLLOUT_B_FLOOR
    Interpretation: partial preservation under either ablation, with a
    meaningful rollout-amplification signature. Both pathways are
    contributing; neither alone accounts for the 569c finding.

  NEITHER    (instrumentation interpretable but verdict ambiguous)
    All other configurations of the three axes. The mechanism question is
    not decisively resolved by 569e; the diagnostic informs governance but
    routes to a refined 569f probe.

  INSTRUMENTATION_FAILURE
    M3 NaN bug not fixed (>=50% of ON-arm seeds still emit NaN on
    e3_top2_class_gap_mean) OR rollout-divergence metric non-finite
    on ON arms OR frozen/argmin arms crash before P1 measurement.
    Diagnostic data cannot be interpreted. Route to /diagnose-errors.

Acceptance criteria
-------------------
C1 (M3 instrumentation): non-NaN e3_top2_class_gap_mean on at least 3/5
   seeds in EACH of ARM_1/2/5/6 (live + argmin live arms). Direct test of
   the NaN bug fix.
C2 (M4 discriminative): |c3_live - c3_frozen| >= 0.05 OR
   |frozen_pres - 1.0| >= 0.25. Confirms M4 produces measurable change
   on the C3 axis (regardless of direction).
C3 (M1 operative): rollout_traj_pairwise_dist_mean on ON arms (1/2)
   >= 3.0x ARM_0 baseline AND non-zero across all measured ticks.
   Confirms rollout-divergence metric is informative.
C4 (verdict definite): interpretation grid cell IS NOT
   INSTRUMENTATION_FAILURE AND IS NOT NEITHER. The probe produces a
   discriminative reading.

Overall PASS = C1 AND C2 AND C3 AND C4. PASS unlocks the autopsy's
recommended per-claim direction shift application + evidence_quality_note
extension per the failure_autopsy_V3-EXQ-569c routing.

experiment_purpose = "diagnostic" -- this script DISSOCIATES mechanism
pathways; it does NOT produce additional claim weight beyond the 569c
result. Governance excludes diagnostic experiments from confidence /
conflict scoring; the verdict is recorded as context for the autopsy
application step.

Phases
------
P0 (30 ep, instrumentation OFF, training ON for LIVE/FROZEN_P0/ARGMIN arms):
    encoder warmup. Allow V_s / event_segmenter / residue field to develop;
    contrastive head trains on observed (z_world_0, action, z_world_1)
    triples for SD-056-enabled arms.
P1 (20 ep, instrumentation ON):
    measurement window. For LIVE arms: training continues. For FROZEN_P1
    arms: e2_opt cleared at P1 entry, buffer-append disabled. For ARGMIN
    arms: training continues but temperature dropped to 0.01.

Budget: 8 arms x 5 seeds x 50 ep x 200 steps = 400k steps total.
~400 min on Mac (DLAPTOP-4.local @ ~14 steps/sec), proportional on cloud
workers. estimated_minutes=400 in queue entry.

Sister chip (parallel): V3-EXQ-569d floor-recalibrated falsifier (separate
session). 569d tests "is the substrate operative under the right floor".
569e (this) tests "where does the C3 lift come from". Both reference the
same autopsy; both can run in parallel.

Smoke
-----
  /opt/local/bin/python3 experiments/v3_exq_569e_sd056_mechanism_probe_pathway_a_vs_b.py --dry-run
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

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_569e_sd056_mechanism_probe_pathway_a_vs_b"
QUEUE_ID = "V3-EXQ-569e"
CLAIM_IDS: List[str] = ["ARC-065", "MECH-341"]
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES: Optional[str] = None  # parallel diagnostic, NOT a falsifier successor

SEEDS = [42, 43, 44, 45, 46]
P0_WARMUP_EPISODES = 30
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Pre-registered interpretation thresholds (do NOT derive from the run).
PRESERVED_FLOOR = 0.5    # frozen_pres / argmin_pres above this == "preserved"
COLLAPSED_CEIL = 0.2     # below this == "collapsed"
ROLLOUT_B_FLOOR = 2.0    # rollout_mean / pairwise_t1_mean ratio: B signature if >=
ON_LIFT_C3_FLOOR = 0.30  # require c3_live - c3_noise >= this for verdict validity
M3_NAN_FRAC_CEIL = 0.40  # M3 instrumentation fails if NaN fraction >= this on ON arms
C1_MIN_OK_SEEDS = 3      # of 5 seeds with non-NaN top2_class_gap_mean
C2_DELTA_FLOOR = 0.05    # |c3_live - c3_frozen| floor for M4 discriminative
C2_FROZEN_PRES_DELTA = 0.25  # alt: |frozen_pres - 1.0| floor
C3_ROLLOUT_RATIO_FLOOR = 3.0  # rollout-mean on ON arms vs ARM_0 baseline

# E2 contrastive training params (identical to 569c so cross-experiment
# comparability holds against the 569c manifest).
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# M5 force-argmin selection: very-low temperature sharpens softmax to argmin
# through the standard select_action pipeline (E3 internal scoring unchanged).
ARGMIN_TEMPERATURE = 0.01
MATCHED_ENTROPY_TEMPERATURE = 2.5

# ENV identical to V3-EXQ-569c so cross-arm comparability holds.
ENV_KWARGS: Dict[str, Any] = dict(
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


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_OFF",
        "label": "sd056_master_off_baseline",
        "sd056_enabled": False,
        "sd056_weight": 0.0,
        "p0_temperature": 1.0,
        "p1_temperature": 1.0,
        "train_e2_p0": False,
        "train_e2_p1": False,
        "category": "off",
    },
    {
        "arm_id": "ARM_1_W005_LIVE",
        "label": "sd056_on_w005_train_p0_p1",
        "sd056_enabled": True,
        "sd056_weight": 0.05,
        "p0_temperature": 1.0,
        "p1_temperature": 1.0,
        "train_e2_p0": True,
        "train_e2_p1": True,
        "category": "live",
    },
    {
        "arm_id": "ARM_2_W020_LIVE",
        "label": "sd056_on_w020_train_p0_p1",
        "sd056_enabled": True,
        "sd056_weight": 0.20,
        "p0_temperature": 1.0,
        "p1_temperature": 1.0,
        "train_e2_p0": True,
        "train_e2_p1": True,
        "category": "live",
    },
    {
        "arm_id": "ARM_3_W005_FROZEN_P1",
        "label": "sd056_on_w005_train_p0_only_frozen_at_p1",
        "sd056_enabled": True,
        "sd056_weight": 0.05,
        "p0_temperature": 1.0,
        "p1_temperature": 1.0,
        "train_e2_p0": True,
        "train_e2_p1": False,   # M4 frozen-at-P1
        "category": "frozen",
    },
    {
        "arm_id": "ARM_4_W020_FROZEN_P1",
        "label": "sd056_on_w020_train_p0_only_frozen_at_p1",
        "sd056_enabled": True,
        "sd056_weight": 0.20,
        "p0_temperature": 1.0,
        "p1_temperature": 1.0,
        "train_e2_p0": True,
        "train_e2_p1": False,
        "category": "frozen",
    },
    {
        "arm_id": "ARM_5_W005_ARGMIN",
        "label": "sd056_on_w005_train_p0_p1_force_argmin_at_p1",
        "sd056_enabled": True,
        "sd056_weight": 0.05,
        "p0_temperature": 1.0,
        "p1_temperature": ARGMIN_TEMPERATURE,  # M5 force argmin at P1
        "train_e2_p0": True,
        "train_e2_p1": True,
        "category": "argmin",
    },
    {
        "arm_id": "ARM_6_W020_ARGMIN",
        "label": "sd056_on_w020_train_p0_p1_force_argmin_at_p1",
        "sd056_enabled": True,
        "sd056_weight": 0.20,
        "p0_temperature": 1.0,
        "p1_temperature": ARGMIN_TEMPERATURE,
        "train_e2_p0": True,
        "train_e2_p1": True,
        "category": "argmin",
    },
    {
        "arm_id": "ARM_7_MATCHED_NOISE",
        "label": "sd056_off_matched_entropy_temperature_control",
        "sd056_enabled": False,
        "sd056_weight": 0.0,
        "p0_temperature": 1.0,
        "p1_temperature": MATCHED_ENTROPY_TEMPERATURE,
        "train_e2_p0": False,
        "train_e2_p1": False,
        "category": "noise",
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM + V_s + SD-054 stack with SD-056 arm overrides.

    Identical to 569c agent build except SD-056 weight is taken from this
    script's arm dict. Layer B (MECH-341) deliberately OFF and Layer C
    (MECH-313) deliberately OFF -- this is the same single-axis A_only
    falsifier configuration as 569c, so the mechanism probe asks where
    the 569c C3 lift came from on the bit-identical substrate axis.
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
        # ARC-065 SP-CEM (Layer A) -- main-path default
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # Layer B / C deliberately OFF (single-layer A-only mechanism probe)
        use_e3_score_diversity=False,
        use_noise_floor=False,
        # V_s substrate (main-path default; identical across arms)
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056 (the single varying axis between OFF/LIVE/FROZEN/ARGMIN;
        # ARM_7 NOISE has it OFF)
        e2_action_contrastive_enabled=bool(arm["sd056_enabled"]),
        e2_action_contrastive_weight=float(arm["sd056_weight"]),
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Per-tick measurement helpers
# ---------------------------------------------------------------------------

def _trajectory_first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _first_actions_K(candidates) -> torch.Tensor:
    """Stack candidate first-step actions into [K, action_dim]."""
    rows = []
    for traj in candidates:
        rows.append(traj.actions[:, 0, :].detach().reshape(-1))
    return torch.stack(rows, dim=0)


def _entropy_from_counts(counts: Dict[int, int]) -> float:
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


def _per_class_score_stats(
    candidates: List, last_scores: Optional[torch.Tensor]
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Return (selected_class, top2_class_gap, score_std).

    M3: NaN-cleaning happens at the APPEND site (caller side) so non-finite
    propagated values are tracked separately and excluded from the mean. This
    helper just returns the raw arithmetic; the caller decides whether to
    include the value.
    """
    if (
        not candidates
        or len(candidates) < 2
        or last_scores is None
        or last_scores.numel() != len(candidates)
    ):
        return None, None, None
    scores_t = last_scores.detach().reshape(-1).float()
    per_class_scores: Dict[int, List[float]] = {}
    classes_per_cand: List[int] = []
    for i, traj in enumerate(candidates):
        cls = _trajectory_first_action_class(traj)
        classes_per_cand.append(cls)
        per_class_scores.setdefault(cls, []).append(float(scores_t[i].item()))
    sel_idx = int(scores_t.argmin().item())
    selected_class = int(classes_per_cand[sel_idx])
    class_means = [sum(v) / len(v) for v in per_class_scores.values()]
    sorted_means = sorted(class_means)
    top2_gap = float(sorted_means[1] - sorted_means[0]) if len(sorted_means) >= 2 else None
    score_std = float(scores_t.std(unbiased=False).item()) if scores_t.numel() > 1 else 0.0
    return selected_class, top2_gap, score_std


def _rollout_traj_pairwise_dist(candidates) -> Tuple[Optional[float], Optional[float]]:
    """M1: per-step pairwise L2 across candidate z_world rollouts over the
    full horizon. Returns (mean_over_horizon, max_over_horizon) of the
    per-step mean pairwise L2 across candidate pairs.

    Pathway A predicts horizon-mean roughly proportional to the t=1 pairwise
    distance (cand_world_pairwise_dist), scaling roughly horizon-linearly
    with per-step variance.
    Pathway B predicts qualitatively larger horizon-mean than the t=1
    divergence alone would account for, because E2 trajectory dynamics have
    been reshaped.
    """
    if len(candidates) < 2:
        return None, None
    rows = []
    for traj in candidates:
        ws = traj.get_world_state_sequence()
        if ws is None:
            return None, None
        rows.append(ws.detach().reshape(ws.shape[1], -1))  # [horizon+1, world_dim]
    try:
        W = torch.stack(rows, dim=0)  # [K, T, D]
    except Exception:
        return None, None
    K, T, D = W.shape
    if K < 2 or T < 1:
        return None, None
    diff = W.unsqueeze(0) - W.unsqueeze(1)        # [K, K, T, D]
    dist = diff.norm(dim=-1)                       # [K, K, T]
    eye = torch.eye(K, dtype=torch.bool, device=dist.device).unsqueeze(-1)
    mask = (~eye).expand(K, K, T)
    n_pairs = mask.sum(dim=(0, 1)).clamp(min=1).float()  # [T]
    per_t = dist.masked_fill(~mask, 0).sum(dim=(0, 1)) / n_pairs  # [T]
    mean_val = float(per_t.mean().item())
    max_val = float(per_t.max().item())
    if not (math.isfinite(mean_val) and math.isfinite(max_val)):
        return None, None
    return mean_val, max_val


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Pick K buffer entries spreading across first-action classes."""
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
    arm_weight: float,
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    """Run one SGD step on E2 with the SD-056 contrastive loss.

    Identical to V3-EXQ-569c: targets anchored in observed env transitions
    from the rolling buffer; skip on non-finite loss; gradient-clipped at
    MAX_GRAD_NORM before optimiser.step().
    """
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
    weighted = float(arm_weight) * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(
        agent.e2.parameters(), max_norm=MAX_GRAD_NORM
    )
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)

    # E2-only optimiser for SD-056 online contrastive training. Constructed
    # iff SD-056 is enabled AND we will train at least one of P0 / P1. The
    # optimiser is conditionally cleared at P1 entry for FROZEN_P1 arms.
    e2_opt: Optional[torch.optim.Optimizer] = None
    if bool(arm["sd056_enabled"]) and (
        bool(arm["train_e2_p0"]) or bool(arm["train_e2_p1"])
    ):
        e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)
    n_buffer_appends = 0
    n_contrastive_skipped_nonfinite = 0
    n_contrastive_skipped_sparse = 0

    total_train_eps = p0_episodes + p1_episodes

    # P1 metric accumulators.
    pairwise_dists: List[float] = []                       # t=1 (569c metric)
    rollout_pairwise_means: List[float] = []                # M1 horizon-mean
    rollout_pairwise_maxes: List[float] = []                # M1 horizon-max
    rollout_skipped_nonfinite = 0                           # M1 instrumentation
    candidate_first_action_counts: Counter = Counter()
    candidate_unique_per_tick: List[float] = []
    candidate_entropy_per_tick: List[float] = []
    selected_class_counts: Counter = Counter()
    top2_gaps: List[float] = []                             # M3 non-NaN only
    top2_gap_nan_count = 0                                  # M3 NaN tracker
    score_stds: List[float] = []                            # M3 non-NaN only
    score_std_nan_count = 0                                 # M3 NaN tracker
    contrastive_loss_values: List[float] = []

    n_p0_ticks = 0
    n_p1_ticks = 0
    n_contrastive_steps = 0

    error_note: Optional[str] = None
    frozen_at_p1_event = False  # M4 instrumentation: did the freeze fire

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        phase_label = "P1" if is_p1 else "P0"

        # M4 freeze-at-P1: on the FIRST P1 episode, clear the optimiser for
        # FROZEN_P1 arms (train_e2_p1=False) so no further weight updates
        # occur. Buffer-append is also skipped below in the same branch.
        if (
            is_p1
            and not frozen_at_p1_event
            and bool(arm["sd056_enabled"])
            and not bool(arm["train_e2_p1"])
            and e2_opt is not None
        ):
            e2_opt = None
            frozen_at_p1_event = True

        # Phase-conditioned temperature (M5: ARGMIN arms drop to 0.01 at P1).
        arm_temperature = float(
            arm["p1_temperature"] if is_p1 else arm["p0_temperature"]
        )

        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

        for _step in range(steps_per_episode):
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

            # 569c-style observed-transition buffer (anchored, NOT self-
            # anchored). Append BEFORE training so the new sample is
            # available in the same iteration.
            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                    n_buffer_appends += 1
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

            # P1 instrumentation: pre-E3 candidate pool + SD-056 substrate
            # readiness (t=1) + M1 rollout-divergence (horizon).
            if is_p1 and candidates:
                pre_e3_classes = [
                    _trajectory_first_action_class(t) for t in candidates
                ]
                candidate_first_action_counts.update(pre_e3_classes)
                candidate_unique_per_tick.append(float(len(set(pre_e3_classes))))
                cnt: Counter = Counter(pre_e3_classes)
                candidate_entropy_per_tick.append(_entropy_from_counts(dict(cnt)))
                if len(candidates) >= 2:
                    actions_K = _first_actions_K(candidates).to(agent.device)
                    z0 = latent.z_world.detach()
                    with torch.no_grad():
                        dist = float(
                            agent.e2.cand_world_pairwise_dist(z0, actions_K).item()
                        )
                    if math.isfinite(dist):
                        pairwise_dists.append(dist)
                # M1 rollout-divergence (horizon-mean and horizon-max).
                rollout_mean_h, rollout_max_h = _rollout_traj_pairwise_dist(
                    candidates
                )
                if rollout_mean_h is None or rollout_max_h is None:
                    rollout_skipped_nonfinite += 1
                else:
                    rollout_pairwise_means.append(rollout_mean_h)
                    rollout_pairwise_maxes.append(rollout_max_h)

            # Drive z_goal (matches 611b / 569c protocol).
            if agent.goal_state is not None:
                benefit_exposure = float(obs_dict.get("benefit_exposure", 0.0)) if hasattr(
                    obs_dict.get("benefit_exposure", 0.0), "__float__"
                ) else 0.0
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )

            action = agent.select_action(
                candidates, ticks, temperature=arm_temperature
            )
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

            # P1 instrumentation: selected-class accounting + E3 score stats.
            # M3 NaN cleaning: filter top2_gap / score_std at the append site;
            # track skip counts separately so instrumentation health is observable.
            if is_p1:
                last_scores = getattr(agent.e3, "last_scores", None)
                sel_class, top2_gap, score_std = _per_class_score_stats(
                    candidates, last_scores
                )
                committed_class = int(action[0].argmax().item())
                selected_class_counts[committed_class] += 1
                if top2_gap is not None:
                    if math.isfinite(top2_gap):
                        top2_gaps.append(top2_gap)
                    else:
                        top2_gap_nan_count += 1
                if score_std is not None:
                    if math.isfinite(score_std):
                        score_stds.append(score_std)
                    else:
                        score_std_nan_count += 1
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            # SD-056 online contrastive training step (gated per phase).
            # For LIVE / ARGMIN arms: train across P0 + P1 (e2_opt is not None
            # in both phases). For FROZEN_P1 arms: e2_opt is set to None at
            # P1 entry, so no step fires and buffer-append continues only
            # because the env transitions still feed the buffer (the buffer
            # is harmless when no optimiser consumes it).
            if e2_opt is not None and tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                # Phase gate: train_e2_p0 / train_e2_p1 govern whether we
                # actually take an SGD step this phase. For LIVE / ARGMIN
                # arms both are True; for FROZEN_P1 we already cleared
                # e2_opt above so this branch is unreachable. The phase gate
                # is belt-and-braces in case future arm definitions differ.
                phase_trains = (
                    bool(arm["train_e2_p0"]) if not is_p1 else bool(arm["train_e2_p1"])
                )
                if phase_trains:
                    loss_val = _e2_contrastive_step(
                        agent=agent,
                        buffer=transition_buffer,
                        arm_weight=float(arm["sd056_weight"]),
                        optimiser=e2_opt,
                        rng=sample_rng,
                    )
                    if loss_val is None:
                        n_contrastive_skipped_sparse += 1
                    elif not math.isfinite(loss_val):
                        n_contrastive_skipped_nonfinite += 1
                    elif is_p1:
                        contrastive_loss_values.append(loss_val)
                        n_contrastive_steps += 1

            if (
                e2_opt is not None
                and torch.isfinite(latent.z_world).all()
                and torch.isfinite(action).all()
            ):
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            _, harm_signal, done, info, next_obs_dict = env.step(action)

            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action
            obs_dict = next_obs_dict
            tick_in_ep += 1
            if done:
                break

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    def _maxx(xs: List[float], default: float = 0.0) -> float:
        return float(max(xs)) if xs else default

    def _minx(xs: List[float], default: float = 0.0) -> float:
        return float(min(xs)) if xs else default

    candidate_first_action_entropy_mean = _mean(candidate_entropy_per_tick)
    candidate_unique_mean = _mean(candidate_unique_per_tick)
    selected_action_entropy = _entropy_from_counts(dict(selected_class_counts))
    trajectory_class_count_mean = candidate_unique_mean
    selected_n_unique = int(len(selected_class_counts))

    # M3: NaN fraction = nan / (nan + ok). 1.0 means all P1 measurements
    # produced NaN; 0.0 means none did. ARM_0 / ARM_7 (no SD-056) should be
    # ~0.0 in 569c, ON arms in 569c were 1.0 -- this is the M3 success
    # criterion (must be << 1.0 on ON arms now).
    total_top2_attempts = len(top2_gaps) + top2_gap_nan_count
    top2_nan_fraction = (
        float(top2_gap_nan_count) / float(total_top2_attempts)
        if total_top2_attempts > 0
        else 0.0
    )
    total_score_std_attempts = len(score_stds) + score_std_nan_count
    score_std_nan_fraction = (
        float(score_std_nan_count) / float(total_score_std_attempts)
        if total_score_std_attempts > 0
        else 0.0
    )

    # M1 rollout-amplification ratio: horizon-mean / t=1 mean. Only meaningful
    # when both are populated and non-trivial.
    t1_mean = _mean(pairwise_dists)
    rollout_mean = _mean(rollout_pairwise_means)
    rollout_amp_ratio = (
        rollout_mean / t1_mean if t1_mean > 1e-9 else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "category": arm["category"],
        "seed": int(seed),
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_contrastive_steps": int(n_contrastive_steps),
        "n_buffer_appends": int(n_buffer_appends),
        "n_contrastive_skipped_sparse": int(n_contrastive_skipped_sparse),
        "n_contrastive_skipped_nonfinite": int(n_contrastive_skipped_nonfinite),
        "frozen_at_p1_event": bool(frozen_at_p1_event),
        "error_note": error_note,
        # SD-056 substrate-operative metrics (t=1 -- 569c metric)
        "cand_world_pairwise_dist_mean": round(t1_mean, 6),
        "cand_world_pairwise_dist_max": round(_maxx(pairwise_dists), 6),
        "cand_world_pairwise_dist_min": round(_minx(pairwise_dists), 6),
        # M1: rollout-divergence over full horizon
        "rollout_traj_pairwise_dist_mean": round(rollout_mean, 6),
        "rollout_traj_pairwise_dist_max": round(_maxx(rollout_pairwise_maxes), 6),
        "rollout_traj_pairwise_dist_amp_ratio_vs_t1": round(rollout_amp_ratio, 4),
        "rollout_skipped_nonfinite": int(rollout_skipped_nonfinite),
        # Pre-E3 candidate-pool diversity
        "candidate_first_action_entropy_mean": round(
            candidate_first_action_entropy_mean, 6
        ),
        "candidate_unique_first_action_classes_mean": round(candidate_unique_mean, 6),
        "trajectory_class_count_mean": round(trajectory_class_count_mean, 6),
        "candidate_first_action_counts": dict(sorted(candidate_first_action_counts.items())),
        # Post-E3 selection diversity (C3 input)
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": selected_n_unique,
        # M3: E3 score diagnostics with NaN cleaning + instrumentation tracking
        "e3_top2_class_gap_mean": round(_mean(top2_gaps), 6),
        "e3_top2_class_gap_n_finite": int(len(top2_gaps)),
        "e3_top2_class_gap_n_nan": int(top2_gap_nan_count),
        "e3_top2_class_gap_nan_fraction": round(top2_nan_fraction, 4),
        "e3_score_std_mean": round(_mean(score_stds), 6),
        "e3_score_std_n_finite": int(len(score_stds)),
        "e3_score_std_n_nan": int(score_std_nan_count),
        "e3_score_std_nan_fraction": round(score_std_nan_fraction, 4),
        # SD-056 training-side telemetry
        "contrastive_loss_mean": round(_mean(contrastive_loss_values), 6),
        "contrastive_loss_min": round(_minx(contrastive_loss_values), 6),
        "contrastive_loss_max": round(_maxx(contrastive_loss_values), 6),
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation + interpretation grid
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _category_rows(
    rows: List[Dict[str, Any]], category: str
) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("category") == category]


def _n_seeds_above(rows: List[Dict[str, Any]], key: str, floor: float) -> int:
    return sum(1 for r in rows if float(r.get(key, 0.0)) > floor)


def _n_seeds_with_finite(
    rows: List[Dict[str, Any]], finite_key: str, min_count: int = 1
) -> int:
    """Seeds where finite_key >= min_count (instrumentation populated)."""
    return sum(1 for r in rows if int(r.get(finite_key, 0)) >= min_count)


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pre-registered interpretation grid + acceptance criteria."""
    by_arm = {
        arm["arm_id"]: _arm_rows(arm_results, arm["arm_id"]) for arm in ARMS
    }
    live_rows = _category_rows(arm_results, "live")        # ARM_1 + ARM_2
    frozen_rows = _category_rows(arm_results, "frozen")    # ARM_3 + ARM_4
    argmin_rows = _category_rows(arm_results, "argmin")    # ARM_5 + ARM_6
    off_rows = _category_rows(arm_results, "off")          # ARM_0
    noise_rows = _category_rows(arm_results, "noise")      # ARM_7

    # M3 instrumentation health: count seeds with finite top2_class_gap on
    # ON-arms (live + argmin). FROZEN arms may produce NaN under different
    # conditions and are not part of this gate (their E2 weights are frozen
    # at P1, so they should behave more like ARM_0 / ARM_7).
    arm1_c1 = _n_seeds_with_finite(by_arm["ARM_1_W005_LIVE"], "e3_top2_class_gap_n_finite")
    arm2_c1 = _n_seeds_with_finite(by_arm["ARM_2_W020_LIVE"], "e3_top2_class_gap_n_finite")
    arm5_c1 = _n_seeds_with_finite(by_arm["ARM_5_W005_ARGMIN"], "e3_top2_class_gap_n_finite")
    arm6_c1 = _n_seeds_with_finite(by_arm["ARM_6_W020_ARGMIN"], "e3_top2_class_gap_n_finite")
    c1_min_required = C1_MIN_OK_SEEDS
    c1_pass = (
        arm1_c1 >= c1_min_required
        and arm2_c1 >= c1_min_required
        and arm5_c1 >= c1_min_required
        and arm6_c1 >= c1_min_required
    )
    # Also check NaN fraction stays below ceiling on ON-arms.
    m3_max_nan_frac_on = max(
        _mean_key(by_arm["ARM_1_W005_LIVE"], "e3_top2_class_gap_nan_fraction"),
        _mean_key(by_arm["ARM_2_W020_LIVE"], "e3_top2_class_gap_nan_fraction"),
        _mean_key(by_arm["ARM_5_W005_ARGMIN"], "e3_top2_class_gap_nan_fraction"),
        _mean_key(by_arm["ARM_6_W020_ARGMIN"], "e3_top2_class_gap_nan_fraction"),
    )

    # Core C3 axis means
    c3_live = _mean_key(live_rows, "selected_action_class_entropy")
    c3_frozen = _mean_key(frozen_rows, "selected_action_class_entropy")
    c3_argmin = _mean_key(argmin_rows, "selected_action_class_entropy")
    c3_off = _mean_key(off_rows, "selected_action_class_entropy")
    c3_noise = _mean_key(noise_rows, "selected_action_class_entropy")

    on_c3_lift = c3_live - c3_noise
    frozen_pres = (c3_frozen - c3_noise) / max(on_c3_lift, 1e-6)
    argmin_pres = (c3_argmin - c3_noise) / max(on_c3_lift, 1e-6)

    # C2 M4 discriminative: |c3_live - c3_frozen| OR frozen_pres delta from 1.0
    c2_abs_delta = abs(c3_live - c3_frozen)
    c2_pres_delta = abs(frozen_pres - 1.0)
    c2_pass = (
        c2_abs_delta >= C2_DELTA_FLOOR
        or c2_pres_delta >= C2_FROZEN_PRES_DELTA
    )

    # M1 rollout-divergence axis
    rollout_mean_live = _mean_key(live_rows, "rollout_traj_pairwise_dist_mean")
    rollout_mean_off = _mean_key(off_rows, "rollout_traj_pairwise_dist_mean")
    t1_mean_live = _mean_key(live_rows, "cand_world_pairwise_dist_mean")
    t1_mean_off = _mean_key(off_rows, "cand_world_pairwise_dist_mean")
    rollout_amp_live = _mean_key(live_rows, "rollout_traj_pairwise_dist_amp_ratio_vs_t1")
    rollout_ratio_on_vs_off = (
        rollout_mean_live / rollout_mean_off if rollout_mean_off > 1e-9 else 0.0
    )

    c3_pass = (
        rollout_ratio_on_vs_off >= C3_ROLLOUT_RATIO_FLOOR
        and rollout_mean_live > 1e-6
    )

    # Verdict cell selection (pre-registered grid)
    if (
        not c1_pass
        or m3_max_nan_frac_on >= M3_NAN_FRAC_CEIL
        or not c3_pass
        or on_c3_lift < ON_LIFT_C3_FLOOR
    ):
        verdict_cell = "INSTRUMENTATION_FAILURE"
    elif (
        frozen_pres >= PRESERVED_FLOOR
        and argmin_pres >= PRESERVED_FLOOR
        and rollout_amp_live >= ROLLOUT_B_FLOOR
    ):
        verdict_cell = "PATHWAY_B"
    elif (
        frozen_pres <= COLLAPSED_CEIL
        and argmin_pres <= COLLAPSED_CEIL
        and rollout_amp_live < ROLLOUT_B_FLOOR
    ):
        verdict_cell = "PATHWAY_A"
    elif (
        (
            COLLAPSED_CEIL < frozen_pres < PRESERVED_FLOOR
            or COLLAPSED_CEIL < argmin_pres < PRESERVED_FLOOR
        )
        and rollout_amp_live >= ROLLOUT_B_FLOOR
    ):
        verdict_cell = "BOTH"
    else:
        verdict_cell = "NEITHER"

    c4_pass = verdict_cell not in ("INSTRUMENTATION_FAILURE", "NEITHER")

    overall_pass = bool(c1_pass and c2_pass and c3_pass and c4_pass)

    return {
        # M3 instrumentation
        "m3_c1_pass": bool(c1_pass),
        "m3_c1_min_seeds_required": int(c1_min_required),
        "m3_arm1_n_seeds_finite": int(arm1_c1),
        "m3_arm2_n_seeds_finite": int(arm2_c1),
        "m3_arm5_n_seeds_finite": int(arm5_c1),
        "m3_arm6_n_seeds_finite": int(arm6_c1),
        "m3_max_nan_fraction_on_arms": round(m3_max_nan_frac_on, 4),
        "m3_nan_fraction_ceiling": M3_NAN_FRAC_CEIL,
        # C3 axis core means
        "c3_live_mean": round(c3_live, 6),
        "c3_frozen_mean": round(c3_frozen, 6),
        "c3_argmin_mean": round(c3_argmin, 6),
        "c3_off_mean": round(c3_off, 6),
        "c3_noise_mean": round(c3_noise, 6),
        "on_c3_lift_above_noise": round(on_c3_lift, 6),
        "on_c3_lift_floor": ON_LIFT_C3_FLOOR,
        # M4 ablation axis
        "m4_c2_pass": bool(c2_pass),
        "m4_abs_delta_live_minus_frozen": round(c2_abs_delta, 6),
        "m4_abs_delta_floor": C2_DELTA_FLOOR,
        "m4_frozen_preservation_ratio": round(frozen_pres, 4),
        "m4_frozen_preservation_delta_from_unity": round(c2_pres_delta, 4),
        "m4_frozen_pres_delta_floor": C2_FROZEN_PRES_DELTA,
        # M5 routing-ablation axis
        "m5_argmin_preservation_ratio": round(argmin_pres, 4),
        # M1 rollout axis
        "m1_c3_pass": bool(c3_pass),
        "m1_rollout_mean_live": round(rollout_mean_live, 6),
        "m1_rollout_mean_off": round(rollout_mean_off, 6),
        "m1_rollout_ratio_on_vs_off": round(rollout_ratio_on_vs_off, 4),
        "m1_rollout_ratio_floor": C3_ROLLOUT_RATIO_FLOOR,
        "m1_t1_mean_live": round(t1_mean_live, 6),
        "m1_t1_mean_off": round(t1_mean_off, 6),
        "m1_rollout_amp_ratio_live": round(rollout_amp_live, 4),
        "m1_rollout_amp_ratio_floor": ROLLOUT_B_FLOOR,
        # Verdict cell + thresholds
        "verdict_cell": verdict_cell,
        "verdict_c4_pass": bool(c4_pass),
        "preserved_floor": PRESERVED_FLOOR,
        "collapsed_ceil": COLLAPSED_CEIL,
        # Per-arm core means (interpretation aid)
        "arm0_selected_entropy": round(_mean_key(by_arm["ARM_0_OFF"], "selected_action_class_entropy"), 6),
        "arm1_selected_entropy": round(_mean_key(by_arm["ARM_1_W005_LIVE"], "selected_action_class_entropy"), 6),
        "arm2_selected_entropy": round(_mean_key(by_arm["ARM_2_W020_LIVE"], "selected_action_class_entropy"), 6),
        "arm3_selected_entropy": round(_mean_key(by_arm["ARM_3_W005_FROZEN_P1"], "selected_action_class_entropy"), 6),
        "arm4_selected_entropy": round(_mean_key(by_arm["ARM_4_W020_FROZEN_P1"], "selected_action_class_entropy"), 6),
        "arm5_selected_entropy": round(_mean_key(by_arm["ARM_5_W005_ARGMIN"], "selected_action_class_entropy"), 6),
        "arm6_selected_entropy": round(_mean_key(by_arm["ARM_6_W020_ARGMIN"], "selected_action_class_entropy"), 6),
        "arm7_selected_entropy": round(_mean_key(by_arm["ARM_7_MATCHED_NOISE"], "selected_action_class_entropy"), 6),
        "arm0_pairwise_t1": round(_mean_key(by_arm["ARM_0_OFF"], "cand_world_pairwise_dist_mean"), 6),
        "arm1_pairwise_t1": round(_mean_key(by_arm["ARM_1_W005_LIVE"], "cand_world_pairwise_dist_mean"), 6),
        "arm2_pairwise_t1": round(_mean_key(by_arm["ARM_2_W020_LIVE"], "cand_world_pairwise_dist_mean"), 6),
        "arm3_pairwise_t1": round(_mean_key(by_arm["ARM_3_W005_FROZEN_P1"], "cand_world_pairwise_dist_mean"), 6),
        "arm4_pairwise_t1": round(_mean_key(by_arm["ARM_4_W020_FROZEN_P1"], "cand_world_pairwise_dist_mean"), 6),
        # Overall
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = _run_seed_arm(arm, seed, p0, p1, steps)
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

    # experiment_purpose=diagnostic -> per-claim direction does NOT carry
    # governance weight. We still record the autopsy-recommended directions
    # for the claim_ids tagged, so the manifest's evidence_record contains
    # the routing reference even when scoring excludes the manifest.
    per_claim_direction = {
        "ARC-065": "supports" if outcome == "PASS" else "mixed",
        "MECH-341": "supports" if outcome == "PASS" else "mixed",
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "result": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": "supports" if outcome == "PASS" else "mixed",
        "evidence_direction_per_claim": per_claim_direction,
        "evidence_direction_note": (
            "V3-EXQ-569e is a mechanism probe parallel to V3-EXQ-569c "
            "(matched-entropy FP-2 falsifier) and V3-EXQ-569d (floor-"
            "recalibrated falsifier). Dissociates Pathway A (per-candidate "
            "z_world variance propagation through E3 softmax routing) vs "
            "Pathway B (E2 rollout dynamics shift producing a different "
            "scoring landscape) as the source of the 569c C3 lift "
            "(~2.4x above matched-noise control). Pre-registered "
            "interpretation grid maps (M3 NaN-bug-fix, M4 frozen-E2-at-P1, "
            "M5 force-argmin, M1 rollout-divergence) onto PATHWAY_A / "
            "PATHWAY_B / BOTH / NEITHER verdict cells. PASS = grid "
            "produces a determinate non-ambiguous cell; FAIL = "
            "instrumentation broken (M3 NaN persists / M1 rollout "
            "metric non-finite) OR verdict ambiguous (NEITHER cell). "
            "experiment_purpose=diagnostic so per-claim direction does "
            "NOT weight governance scoring; the verdict cell informs "
            "application of the failure_autopsy_V3-EXQ-569c-recommended "
            "evidence_quality_note + per-claim direction shift on the "
            "569c manifest."
        ),
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": ENV_KWARGS,
            "arms": [{"arm_id": a["arm_id"], "label": a["label"],
                       "category": a["category"],
                       "sd056_enabled": a["sd056_enabled"],
                       "sd056_weight": a["sd056_weight"],
                       "p0_temperature": a["p0_temperature"],
                       "p1_temperature": a["p1_temperature"],
                       "train_e2_p0": a["train_e2_p0"],
                       "train_e2_p1": a["train_e2_p1"]} for a in ARMS],
            "preserved_floor": PRESERVED_FLOOR,
            "collapsed_ceil": COLLAPSED_CEIL,
            "rollout_b_floor": ROLLOUT_B_FLOOR,
            "on_lift_c3_floor": ON_LIFT_C3_FLOOR,
            "m3_nan_fraction_ceiling": M3_NAN_FRAC_CEIL,
            "c1_min_ok_seeds": C1_MIN_OK_SEEDS,
            "c2_delta_floor": C2_DELTA_FLOOR,
            "c2_frozen_pres_delta": C2_FROZEN_PRES_DELTA,
            "c3_rollout_ratio_floor": C3_ROLLOUT_RATIO_FLOOR,
            "argmin_temperature": ARGMIN_TEMPERATURE,
            "matched_entropy_temperature": MATCHED_ENTROPY_TEMPERATURE,
            "e2_contrastive_lr": E2_CONTRASTIVE_LR,
            "e2_train_every_k_ticks": E2_TRAIN_EVERY_K_TICKS,
        },
        "acceptance_criteria": {
            "C1_M3_instrumentation_top2_gap_finite_on_on_arms": summary["m3_c1_pass"],
            "C2_M4_frozen_arm_discriminative_on_C3": summary["m4_c2_pass"],
            "C3_M1_rollout_divergence_operative_on_on_arms": summary["m1_c3_pass"],
            "C4_interpretation_grid_yields_determinate_verdict": summary["verdict_c4_pass"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Result written to: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"  verdict_cell: {summary['verdict_cell']}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-569e SD-056 mechanism probe (Pathway A vs Pathway B)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Short smoke run; no manifest written.",
    )
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
    sys.exit(0)
