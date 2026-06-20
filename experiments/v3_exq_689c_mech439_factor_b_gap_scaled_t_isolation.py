"""V3-EXQ-689c: MECH-439 Factor-B-alone (gap-scaled commit-temperature, A0B1)
isolation retest. Same scientific question (MECH-439's conflict-grade); does NOT
supersede anything (it isolates a sub-lever; 689a's combined-conflict-grade
verdict stands).

WHY 689c (re-point the primary, not a re-queue). 689a was an 8-arm 2x2 conflict-
grade falsifier whose PRIMARY arm was ARM_A1B1 (BOTH conflict-grade levers ON:
Factor A conflict-graded shortlist width + Factor B gap-scaled commit-T). 689a
FAILED -- A1B1 lost to all controls -- but its decomposition showed ARM_A0B1
(Factor B ALONE: factor_a OFF / factor_b ON) was the ONLY converter. 689c
re-points the pre-registered PRIMARY to ARM_A0B1, drops the two Factor-A-only
arms (ARM_A1B0, ARM_FIXED_KMAX), and keeps ARM_A1B1 as a CONFIRMATORY arm
(both levers; expected to LOSE to A0B1 -- Factor A poisons B). Same claim
(MECH-439), same env, same GAP-A conversion constants; ONLY the primary, the
arm set, the gating criteria, the identifiers, and this docstring change.

THE TEST (Factor-B-alone, gap-scaled commit-temperature). Factor B is the
gap-scaled entropy-regularized commit: T_eff = base + alpha*(1 - gap_norm), so
near-ties get a hotter (softer-argmax) committed selection and decisive gaps
stay cold. ARM_A0B1 (factor_a OFF, factor_b ON, alpha=1.5) is the isolated
gap-SCALED commit cell over the divergent e2_world_forward source. The
LOAD-BEARING control is ARM_FIXED_HOT_T (factor_a OFF, factor_b ON,
gap_scaled_alpha=0.0, base T=2.5 -> T_eff=2.5 CONSTANT): a uniformly-hot FLAT
softmax over the same divergent eligible set. If A0B1's committed-entropy lift
is matched by the gap-blind flat-hot control, the conversion is NOT
gap-scaling -- it is just a hotter flat softmax.
  LEAD (C_GAPBLIND_B, load-bearing) -- Factor B's gap-scaling is load-bearing
  IFF ARM_A0B1 (PRIMARY) is strict-above ARM_FIXED_HOT_T (the flat-hot Factor-B
  control) on committed-action-class entropy on >= 2/3 seeds: the conversion is
  gap-SCALING, not a hotter flat softmax.

PRESERVED non-vacuity gates (below floor still self-routes
substrate_not_ready_requeue, NEVER a false weakens):
  (a) IN-ARM route-range > ROUTE_RANGE_FLOOR (the divergent channel reaches the
      bias the shortlist arbitrates), RANGE statistic.
  (b) e2.world_forward per-candidate prediction spread > C1_PAIRWISE_DIST_FLOOR
      (SD-056 trained the action-conditional divergence), RANGE statistic.
  (c) Factor-B lever-engaged: ARM_A0B1 gap-scaled commit T_eff varies across P1
      ticks (Factor A is OFF, so k is CONSTANT by design -- k_varies is reported
      but NOT gated). t_eff-varies only.

THE FACTOR-B CONTROL MAPPING ONTO THE LANDED MECH-439 SUBSTRATE:
  ARM_A0B1 (PRIMARY) = factor_a OFF + factor_b ON
    (use_gap_scaled_commit_temperature) with gap_scaled_commit_entropy_alpha=1.5
    + base temperature=1.0 -> T_eff = 1.0 + 1.5*(1 - gap_norm), peak 2.5 at a
    near-tie. Scales the commit heat BY the gap.
  ARM_FIXED_HOT_T = factor_b ON but gap_scaled_commit_entropy_alpha=0.0 AND base
    temperature=2.5 -> T_eff = 2.5 + 0*(1 - gap_norm) = 2.5 CONSTANT (gap-blind).
    Same multinomial-over-eligible-set commit machinery as ARM_A0B1, but flat.
    Distinct from ARM_MATCHED_NOISE (flat-hot over the COLLAPSED proposer
    source); FIXED_HOT_T is flat-hot over the DIVERGENT e2_world_forward source.

ALL six arms share the GAP-A-ready conversion constant: SD-056 online-trained
e2.world_forward (rollout-norm clamp ON, 643a lesson) + ARC-065 GAP-A
candidate_summary_source=e2_world_forward (the divergent eligible set -- the
NON-VACUITY precondition) + route-range routing + top-k shortlist + SP-CEM
Layer A + shared E3-side bias channels (lateral_pfc + mech295). CRF stack off.

  ARM_PROPOSER_CTRL  proposer,  T=1.0, A off, B off, k=3  (collapsed-channel baseline; no-conversion-reaches control)
  ARM_MATCHED_NOISE  proposer,  T=2.5, A off, B off, k=3  (flat hot softmax over the COLLAPSED proposer channel; negative control)
  ARM_A0B0           e2wf,      T=1.0, A off, B off, k=3  (divergent pool; both levers OFF = the 569i baseline cell)
  ARM_A0B1 (PRIMARY) e2wf,      T=1.0, A off, B GAP-SCALED (Factor B ALONE -- gap-scaled commit-T, alpha=1.5)
  ARM_A1B1 (confirm) e2wf,      T=1.0, A GRADED, B GAP-SCALED (BOTH levers; should LOSE to A0B1 -- Factor A poisons B)
  ARM_FIXED_HOT_T    e2wf,      T=2.5, A off, B FLAT (alpha=0 -> T_eff=2.5 const) (GAP-BLIND Factor-B control -- flat hot softmax over the DIVERGENT pool)

ACCEPTANCE (evidence, claim_ids=[MECH-439]; supersedes nothing):
  READINESS (load-bearing non-vacuity; below any -> substrate_not_ready_requeue,
  NEVER a weakens -- RANGE statistics, the same-statistic discipline):
    (a) IN-ARM ROUTE-RANGE: ARM_A0B1 modulatory_channel_route_range mean (read
        LIVE at the select tick) > ROUTE_RANGE_FLOOR on >= MIN_SEEDS_FOR_PASS seeds.
    (b) E2-DIVERGENCE: ARM_A0B1 cand_world_pairwise_dist > C1_PAIRWISE_DIST_FLOOR
        on >= MIN_SEEDS_FOR_PASS seeds.
    (c) FACTOR-B LEVER-ENGAGED: ARM_A0B1 gap-scaled commit T_eff varies (spread
        > eps) on >= MIN_SEEDS_FOR_PASS seeds. Factor A is OFF so k_effective is
        CONSTANT by design (k_varies is computed + REPORTED, NOT gated).
  C_PRIMARY (gate, secondary off-ramp discriminator): ARM_A0B1
    selected_action_class_entropy STRICTLY ABOVE BOTH ARM_PROPOSER_CTRL AND
    ARM_MATCHED_NOISE on the SAME seed, on >= MIN_SEEDS_FOR_PASS seeds, AND
    ARM_A0B1 mean > C3_SELECTED_ENTROPY_FLOOR (no-lift-at-all off-ramp).
  C_GAPBLIND_B (LOAD-BEARING): ARM_A0B1 selected_action_class_entropy STRICTLY
    ABOVE ARM_FIXED_HOT_T (the flat-hot Factor-B control) on the SAME seed, on
    >= MIN_SEEDS_FOR_PASS seeds, AND ARM_A0B1 mean > C3_SELECTED_ENTROPY_FLOOR.
    The conversion is gap-SCALING (conflict-grading), not a hotter flat softmax.
  PASS = READINESS AND C_PRIMARY AND C_GAPBLIND_B ->
    factor_b_gap_scaled_t_converts_committed_diversity; evidence_direction=supports.

Interpretation grid:
| outcome                                              | label                                              | evidence_direction | next                                                                                       |
|------------------------------------------------------|----------------------------------------------------|--------------------|--------------------------------------------------------------------------------------------|
| READINESS + C_PRIMARY + C_GAPBLIND_B                 | factor_b_gap_scaled_t_converts_committed_diversity | supports           | Factor B isolated PASS; gap-scaled commit-T moves committed behaviour, lift is gap-scaled   |
| route/e2-div below floor OR t_eff not varied         | substrate_not_ready_requeue                        | non_contributory   | routing/SD-056 under-trained OR Factor-B lever inert; re-queue; NOT a weakens               |
| READINESS met, C_PRIMARY fail (no lift vs collapsed) | conversion_ceiling_persists_despite_conflict_grade | non_contributory   | OFF-RAMP -> V4 directions (divisive normalization / output-null / cross-tick QD archive); NOT a falsification |
| READINESS + C_PRIMARY met, C_GAPBLIND_B fail         | gap_scaled_t_lift_matches_flat_hot_control         | non_contributory   | the lift matches the gap-blind flat-hot-T -> NOT gap-concentrated -> the lift is not gap-scaling |

NOTE: NO weakens path. MECH-439 is candidate; a PASS moves it toward supports; a
preconditions-met fail routes to the V4 directions / the V3 fallback, NOT a dead
end. The 2x2 dissociation is reported per cell (informational); the PRIMARY gates
on ARM_A0B1 vs the flat-hot Factor-B control. claim_ids=[MECH-439] only;
ARC-065/MECH-341/ARC-062/MECH-309/MECH-294 untouched.

Usage:
  /opt/local/bin/python3 experiments/v3_exq_689c_mech439_factor_b_gap_scaled_t_isolation.py --dry-run
"""

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
from experiments._lib.arm_fingerprint import (  # noqa: E402
    compute_arm_fingerprint,
    reset_all_rng,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_689c_mech439_factor_b_gap_scaled_t_isolation"
QUEUE_ID = "V3-EXQ-689c"
SUPERSEDES = None
CLAIM_IDS: List[str] = ["MECH-439"]  # MECH-439 Factor-B-alone (gap-scaled commit-T) isolation
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60           # SD-056 contrastive warmup (569i proven budget)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Acceptance thresholds (pre-registered).
ROUTE_RANGE_FLOOR = 0.01          # READINESS (a): IN-ARM modulatory_channel_route_range
C1_PAIRWISE_DIST_FLOOR = 0.03     # READINESS (b): ARM_A1B1 e2.world_forward prediction spread
C3_SELECTED_ENTROPY_FLOOR = 0.3   # C_PRIMARY / C_GAPBLIND: ARM_A1B1 selected-action class entropy floor
MATCHED_ENTROPY_TEMPERATURE = 2.5
MIN_SEEDS_FOR_PASS = 2            # of 3

# F-gap secondary-readout (C_FGAP) parameters -- NON-GATING in 689a.
FGAP_NBINS = 4                    # gap_norm in [0,1] -> 4 bins (fixed-width AND quantile)
FGAP_MIN_BIN_TICKS = 50          # a bin counts toward the slope only if it has >= this many ticks
GAP_MIN_POPULATED_BINS = 3       # gap-spread DIAGNOSTIC (reported; does NOT gate readiness)
GAP_SLOPE_DELTA = 0.03           # secondary readout: ARM_A1B1 slope <= ARM_A0B0 slope - this

# Conflict-grade lever config (MECH-439).
MODULATORY_SHORTLIST_K = 3                    # fixed-k baseline (Factor A OFF)
MODULATORY_SHORTLIST_K_MAX = 6               # conflict-graded k_max (Factor A ON): k ranges 1..6
GAP_SCALED_COMMIT_ENTROPY_ALPHA = 1.5        # Factor B: T_eff peak = 1.0 + 1.5 = 2.5 (gap-concentrated)
GAP_SCALED_COMMIT_HARM_FLOOR = 0.25          # Factor B standalone safety envelope

# Shared conversion constant (ON all arms).
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6

# SD-056 online contrastive training (mirror 569i harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Behavioural-diversity env: SD-054 reef-bipartite hazard layout (matches 569i exactly).
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

# Arm ids. factor_a = conflict-graded width; factor_b = gap-scaled commit-T.
PRIMARY_ARM = "ARM_A0B1"      # Factor B ALONE (gap-scaled commit-T) -- 689c primary
CONFIRM_ARM = "ARM_A1B1"      # both-levers confirmatory arm (should LOSE to A0B1)
BASELINE_ARM = "ARM_A0B0"
PROPOSER_CTRL_ARM = "ARM_PROPOSER_CTRL"
MATCHED_NOISE_ARM = "ARM_MATCHED_NOISE"
FIXED_KMAX_ARM = "ARM_FIXED_KMAX"      # GAP-BLIND Factor-A control
FIXED_HOT_T_ARM = "ARM_FIXED_HOT_T"    # GAP-BLIND Factor-B control

# Per-arm fields: factor_a / factor_b toggle the gap-scaled levers; shortlist_k and
# gap_scaled_alpha are per-arm OVERRIDES (default to the global config) so the two
# gap-blind controls can pin a fixed-large-k / flat-hot-T over the divergent source.
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": PROPOSER_CTRL_ARM,
        "label": "proposer_collapsed_channel_baseline_control",
        "candidate_summary_source": "proposer",
        "temperature": 1.0,
        "factor_a": False,
        "factor_b": False,
        "shortlist_k": MODULATORY_SHORTLIST_K,
        "gap_scaled_alpha": GAP_SCALED_COMMIT_ENTROPY_ALPHA,
    },
    {
        "arm_id": MATCHED_NOISE_ARM,
        "label": "proposer_matched_entropy_flat_temperature_negative_control",
        "candidate_summary_source": "proposer",
        "temperature": MATCHED_ENTROPY_TEMPERATURE,
        "factor_a": False,
        "factor_b": False,
        "shortlist_k": MODULATORY_SHORTLIST_K,
        "gap_scaled_alpha": GAP_SCALED_COMMIT_ENTROPY_ALPHA,
    },
    {
        "arm_id": BASELINE_ARM,
        "label": "e2wf_fixed_k_hard_argmin_both_levers_off",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "factor_a": False,
        "factor_b": False,
        "shortlist_k": MODULATORY_SHORTLIST_K,
        "gap_scaled_alpha": GAP_SCALED_COMMIT_ENTROPY_ALPHA,
    },
    {
        "arm_id": "ARM_A0B1",
        "label": "e2wf_fixed_k_gap_scaled_commit_t_factor_b_only",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "factor_a": False,
        "factor_b": True,
        "shortlist_k": MODULATORY_SHORTLIST_K,
        "gap_scaled_alpha": GAP_SCALED_COMMIT_ENTROPY_ALPHA,
    },
    {
        "arm_id": CONFIRM_ARM,
        "label": "e2wf_conflict_graded_k_gap_scaled_commit_t_both_levers",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "factor_a": True,
        "factor_b": True,
        "shortlist_k": MODULATORY_SHORTLIST_K,
        "gap_scaled_alpha": GAP_SCALED_COMMIT_ENTROPY_ALPHA,
    },
    {
        # GAP-BLIND Factor-B control: flat hot commit-T over the DIVERGENT e2wf
        # source. factor_b ON (multinomial-over-eligible-set commit machinery)
        # but gap_scaled_alpha=0.0 AND base temperature=2.5 -> T_eff = 2.5 +
        # 0*(1-gap_norm) = 2.5 CONSTANT. Isolates "a hotter flat softmax" from
        # the gap-CONCENTRATED heat of ARM_A0B1 / ARM_A1B1. Distinct from
        # ARM_MATCHED_NOISE, which is flat-hot over the COLLAPSED proposer source.
        "arm_id": FIXED_HOT_T_ARM,
        "label": "e2wf_fixed_hot_t2p5_gap_blind_factor_b_control",
        "candidate_summary_source": "e2_world_forward",
        "temperature": MATCHED_ENTROPY_TEMPERATURE,   # base T = 2.5
        "factor_a": False,
        "factor_b": True,            # commit machinery ON ...
        "shortlist_k": MODULATORY_SHORTLIST_K,
        "gap_scaled_alpha": 0.0,     # ... but gap-BLIND -> T_eff = 2.5 constant
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """GAP-A-ready conversion stack (SP-CEM + SD-056 online + ARC-065 GAP-A
    candidate_summary_source + route-range routing + top-k shortlist + shared
    lateral_pfc/mech295 bias channels), with the MECH-439 conflict-grade levers
    toggled per arm. The two gap-blind controls pin a fixed-large-k
    (modulatory_shortlist_k=k_max, factor_a OFF) or a flat-hot-T
    (use_gap_scaled_commit_temperature with gap_scaled_alpha=0.0 + base T=2.5)
    over the divergent e2_world_forward source. Factor A / Factor B default OFF
    -> bit-identical to the 569i fixed-k/hard-argmin cell."""
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
        # ARC-065 SP-CEM (Layer A) -- main-path action-divergent pool
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # SHARED E3-side bias channels (consume cand_world_summaries)
        use_lateral_pfc_analog=True,
        use_mech295_liking_bridge=True,
        # Other policy-layer regulators + CRF stack OFF (the levers are the axis)
        use_structured_curiosity=False,
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_ofc_analog=False,
        use_gated_policy=False,
        use_candidate_rule_field=False,
        # SD-056 substrate trained online on every arm (e2.world_forward divergence)
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # ARC-065 GAP-A: divergent eligible set (the non-vacuity precondition for all e2wf arms)
        candidate_summary_source=str(arm["candidate_summary_source"]),
        # Shared route-range routing + authority + top-k shortlist conversion constant
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source="cand_world_summary",
        modulatory_channel_route_weight=1.0,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        # Per-arm shortlist width: baseline 3, ARM_FIXED_KMAX pins 6 (gap-blind wide).
        modulatory_shortlist_k=int(arm["shortlist_k"]),
        # --- MECH-439 FACTOR A: conflict-graded shortlist width (per arm) ---
        modulatory_shortlist_conflict_graded=bool(arm["factor_a"]),
        modulatory_shortlist_k_max=MODULATORY_SHORTLIST_K_MAX,
        # --- MECH-439 FACTOR B: gap-scaled entropy-regularized commit (per arm) ---
        #   ARM_FIXED_HOT_T sets factor_b True + gap_scaled_alpha 0.0 + base T 2.5
        #   -> T_eff = 2.5 + 0*(1-gap_norm) = 2.5 constant (gap-blind flat hot-T).
        use_gap_scaled_commit_temperature=bool(arm["factor_b"]),
        gap_scaled_commit_entropy_alpha=float(arm["gap_scaled_alpha"]),
        gap_scaled_commit_harm_floor=GAP_SCALED_COMMIT_HARM_FLOOR,
    )
    agent = REEAgent(cfg)
    return agent


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _trajectory_first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _first_actions_K(candidates) -> torch.Tensor:
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
    if not rows:
        return None
    return torch.stack(rows, dim=0)


def _mean_pairwise_l2(summ: torch.Tensor) -> float:
    summ = summ.detach()
    K = summ.shape[0]
    if K < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(K):
        for j in range(i + 1, K):
            total += float(torch.linalg.vector_norm(summ[i] - summ[j]))
            n += 1
    return total / max(n, 1)


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _gap_bin_index(gap_norm: float) -> int:
    if not math.isfinite(gap_norm) or gap_norm < 0.0:
        return -1
    b = int(gap_norm * FGAP_NBINS)
    if b >= FGAP_NBINS:
        b = FGAP_NBINS - 1
    if b < 0:
        b = 0
    return b


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float(0.5 * (s[mid - 1] + s[mid]))


def _lsq_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    n_used = len(xs)
    if n_used < 3:
        return None
    n = float(n_used)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 1e-12:
        return None
    return float(sxy / sxx)


def _binned_entropy_slope(
    bin_class_counts: List[Dict[int, int]],
) -> Tuple[Optional[float], int, List[Optional[float]]]:
    """Least-squares slope of per-FIXED-WIDTH-gap-bin committed-action entropy vs
    the bin midpoint, over bins with >= FGAP_MIN_BIN_TICKS ticks. SECONDARY,
    NON-GATING in 689a (kept for continuity with V3-EXQ-689). A NEGATIVE slope
    means the committed diversity is concentrated at low F-gap (near-ties) -- the
    conflict-grade signature; on the foraging regime this is usually uncomputable
    because the gap is pinned in the near-tie bin (the reason 689a uses the arm
    contrast as the load-bearing test)."""
    per_bin_entropy: List[Optional[float]] = []
    xs: List[float] = []
    ys: List[float] = []
    for b in range(FGAP_NBINS):
        counts = bin_class_counts[b] if b < len(bin_class_counts) else {}
        total = sum(counts.values())
        if total >= FGAP_MIN_BIN_TICKS:
            h = _entropy_from_counts(counts)
            per_bin_entropy.append(round(h, 6))
            mid = (b + 0.5) / FGAP_NBINS
            xs.append(mid)
            ys.append(h)
        else:
            per_bin_entropy.append(None)
    return _lsq_slope(xs, ys), len(xs), per_bin_entropy


def _quantile_bin_slope(
    gap_class_pairs: List[Tuple[float, int]],
) -> Tuple[Optional[float], int, List[Optional[float]], List[float]]:
    """SECONDARY readout (NON-GATING): quantile-adaptive per-gap-bin
    committed-entropy slope. Bins the per-tick (gap_norm, committed_class) pairs
    by gap QUANTILES (FGAP_NBINS equal-mass bins) rather than fixed-width edges,
    so even a near-tie-concentrated gap distribution yields >= 3 populated bins
    (the fix for the 689 uncomputable-regression failure). Returns
    (slope_or_None, n_bins_used, per_bin_entropy[NBINS], bin_median_gaps). A
    NEGATIVE slope = committed diversity concentrated at low F-gap (the
    conflict-grade signature)."""
    pairs = [(g, c) for (g, c) in gap_class_pairs if math.isfinite(g) and g >= 0.0]
    n = len(pairs)
    per_bin_entropy: List[Optional[float]] = [None] * FGAP_NBINS
    if n < FGAP_NBINS * 2:
        return None, 0, per_bin_entropy, []
    gaps_sorted = sorted(g for (g, _) in pairs)
    # FGAP_NBINS-1 interior quantile edges (equal-mass bins).
    edges: List[float] = []
    for i in range(1, FGAP_NBINS):
        idx = int(round(i * (n - 1) / FGAP_NBINS))
        edges.append(gaps_sorted[idx])
    bin_classes: List[List[int]] = [[] for _ in range(FGAP_NBINS)]
    bin_gaps: List[List[float]] = [[] for _ in range(FGAP_NBINS)]
    for g, c in pairs:
        b = 0
        while b < len(edges) and g > edges[b]:
            b += 1
        bin_classes[b].append(c)
        bin_gaps[b].append(g)
    xs: List[float] = []
    ys: List[float] = []
    median_gaps: List[float] = []
    for b in range(FGAP_NBINS):
        median_gaps.append(round(_median(bin_gaps[b]), 6) if bin_gaps[b] else -1.0)
        if len(bin_classes[b]) >= FGAP_MIN_BIN_TICKS:
            cnt: Counter = Counter(bin_classes[b])
            h = _entropy_from_counts(dict(cnt))
            per_bin_entropy[b] = round(h, 6)
            xs.append(_median(bin_gaps[b]))
            ys.append(h)
    return _lsq_slope(xs, ys), len(xs), per_bin_entropy, [round(e, 6) for e in edges]


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
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    arm_temperature = float(arm["temperature"])
    total_train_eps = p0_episodes + p1_episodes

    pairwise_dists: List[float] = []
    route_ranges: List[float] = []
    route_range_max = 0.0
    authority_active_ticks = 0
    shortlist_sizes: List[float] = []
    shortlist_active_ticks = 0
    shortlist_mode_seen: Optional[str] = None
    # MECH-439 grading-engaged non-vacuity readouts.
    k_eff_values: List[float] = []
    t_eff_values: List[float] = []
    # F-gap readouts. Fixed-width counts (continuity) + raw pairs (quantile readout).
    gap_bin_class_counts: List[Counter] = [Counter() for _ in range(FGAP_NBINS)]
    gap_bin_tick_counts: List[int] = [0 for _ in range(FGAP_NBINS)]
    gap_class_pairs: List[Tuple[float, int]] = []
    selected_class_counts: Counter = Counter()
    n_p1_ticks = 0
    n_contrastive_steps = 0
    error_note: Optional[str] = None

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        phase_label = "P1" if is_p1 else "P0"

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

            if is_p1 and candidates and len(candidates) >= 2:
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
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

            action = agent.select_action(
                candidates, ticks, temperature=arm_temperature
            )

            # MECH-439 readouts: read LIVE diagnostics at the select tick.
            if is_p1:
                diag = agent.e3.last_score_diagnostics
                rr = float(diag.get("modulatory_channel_route_range", 0.0))
                if math.isfinite(rr):
                    route_ranges.append(rr)
                    route_range_max = max(route_range_max, rr)
                if bool(diag.get("modulatory_authority_active", False)):
                    authority_active_ticks += 1
                if bool(diag.get("modulatory_shortlist_active", False)):
                    shortlist_active_ticks += 1
                    sl_size = float(diag.get("modulatory_shortlist_size", 0))
                    if math.isfinite(sl_size):
                        shortlist_sizes.append(sl_size)
                    if shortlist_mode_seen is None:
                        shortlist_mode_seen = str(
                            diag.get("modulatory_shortlist_mode", "")
                        )
                # Factor A effective k (only set when conflict-graded fired).
                k_eff = diag.get("modulatory_shortlist_k_effective", -1)
                if isinstance(k_eff, (int, float)) and k_eff >= 1:
                    k_eff_values.append(float(k_eff))
                # Factor B effective commit temperature (only set when gap-scaled commit fired).
                t_eff = diag.get("gap_scaled_commit_temperature_eff", -1.0)
                if isinstance(t_eff, (int, float)) and t_eff >= 0.0:
                    t_eff_values.append(float(t_eff))

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

            # C_PRIMARY / C_GAPBLIND behavioural DV: committed first-action class.
            # F-gap readouts: fixed-width bin counts + raw (gap, class) pairs.
            if is_p1:
                committed_class = int(action[0].argmax().item())
                selected_class_counts[committed_class] += 1
                n_p1_ticks += 1
                gap_norm = float(
                    agent.e3.last_score_diagnostics.get("conflict_gap_norm", -1.0)
                )
                b = _gap_bin_index(gap_norm)
                if b >= 0:
                    gap_bin_class_counts[b][committed_class] += 1
                    gap_bin_tick_counts[b] += 1
                    gap_class_pairs.append((gap_norm, committed_class))

            if (
                torch.isfinite(latent.z_world).all()
                and torch.isfinite(action).all()
            ):
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val) and is_p1:
                    n_contrastive_steps += 1

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

    def _spread(xs: List[float]) -> float:
        return float(max(xs) - min(xs)) if xs else 0.0

    selected_action_entropy = _entropy_from_counts(dict(selected_class_counts))
    n_populated_gap_bins = sum(
        1 for b in range(FGAP_NBINS) if gap_bin_tick_counts[b] >= FGAP_MIN_BIN_TICKS
    )
    # SECONDARY quantile-adaptive readout (computed per cell; raw pairs not stored).
    q_slope, q_nbins, q_per_bin_entropy, q_edges = _quantile_bin_slope(gap_class_pairs)

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "candidate_summary_source": arm["candidate_summary_source"],
        "temperature": arm_temperature,
        "factor_a_conflict_graded": bool(arm["factor_a"]),
        "factor_b_gap_scaled_commit": bool(arm["factor_b"]),
        "shortlist_k_config": int(arm["shortlist_k"]),
        "gap_scaled_alpha_config": float(arm["gap_scaled_alpha"]),
        "n_p1_ticks": int(n_p1_ticks),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # READINESS (a) / IN-ARM route-range.
        "modulatory_channel_route_range_mean": round(_mean(route_ranges), 6),
        "modulatory_channel_route_range_max": round(route_range_max, 6),
        "modulatory_authority_active_ticks": int(authority_active_ticks),
        # Shortlist discriminator (control-engagement diagnostic).
        "modulatory_shortlist_size_mean": round(_mean(shortlist_sizes), 6),
        "modulatory_shortlist_active_ticks": int(shortlist_active_ticks),
        "modulatory_shortlist_mode": shortlist_mode_seen or "",
        # READINESS (b) / e2.world_forward prediction spread.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        # READINESS (c) / levers-engaged non-vacuity + control-engagement diagnostics.
        "k_effective_mean": round(_mean(k_eff_values), 6),
        "k_effective_spread": round(_spread(k_eff_values), 6),
        "k_effective_n_ticks": int(len(k_eff_values)),
        "t_eff_mean": round(_mean(t_eff_values), 6),
        "t_eff_spread": round(_spread(t_eff_values), 6),
        "t_eff_n_ticks": int(len(t_eff_values)),
        # gap-spread DIAGNOSTIC (reported, NON-GATING in 689a).
        "n_populated_gap_bins": int(n_populated_gap_bins),
        # C_PRIMARY / C_GAPBLIND behavioural DV.
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": int(len(selected_class_counts)),
        # C_FGAP fixed-width (secondary, non-gating; continuity with 689).
        "gap_bin_class_counts": [dict(sorted(c.items())) for c in gap_bin_class_counts],
        "gap_bin_tick_counts": list(gap_bin_tick_counts),
        # C_FGAP quantile-adaptive (secondary readout; the 689 uncomputable fix).
        "gap_quantile_slope": (round(q_slope, 6) if q_slope is not None else None),
        "gap_quantile_bins_used": int(q_nbins),
        "gap_quantile_per_bin_entropy": q_per_bin_entropy,
        "gap_quantile_edges": q_edges,
        "gap_pairs_n": int(len(gap_class_pairs)),
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _n_seeds(rows: List[Dict[str, Any]], predicate) -> int:
    return sum(1 for r in rows if predicate(r))


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _mean_quantile_slope(rows: List[Dict[str, Any]]) -> Optional[float]:
    vals = [float(r["gap_quantile_slope"]) for r in rows
            if r.get("gap_quantile_slope") is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _pool_gap_bins(rows: List[Dict[str, Any]]) -> List[Dict[int, int]]:
    """Sum per-fixed-width-gap-bin committed-class counts across seeds for one arm."""
    pooled: List[Counter] = [Counter() for _ in range(FGAP_NBINS)]
    for r in rows:
        bins = r.get("gap_bin_class_counts") or []
        for b in range(min(FGAP_NBINS, len(bins))):
            for cls, c in (bins[b] or {}).items():
                pooled[b][int(cls)] += int(c)
    return [dict(c) for c in pooled]


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    proposer = _arm_rows(arm_results, PROPOSER_CTRL_ARM)
    noise = _arm_rows(arm_results, MATCHED_NOISE_ARM)
    a0b0 = _arm_rows(arm_results, BASELINE_ARM)
    a0b1 = _arm_rows(arm_results, "ARM_A0B1")
    # 689c: PRIMARY_ARM is ARM_A0B1 (Factor B alone). The variable name `a1b1` is
    # kept as-is throughout to avoid renaming bugs, but it now holds the PRIMARY
    # (A0B1) rows. `confirm` holds the both-levers confirmatory arm (real A1B1).
    a1b1 = _arm_rows(arm_results, PRIMARY_ARM)
    confirm = _arm_rows(arm_results, CONFIRM_ARM)
    fixed_hot_t = _arm_rows(arm_results, FIXED_HOT_T_ARM)

    proposer_by_seed = {r["seed"]: r for r in proposer}
    noise_by_seed = {r["seed"]: r for r in noise}
    fixed_hot_t_by_seed = {r["seed"]: r for r in fixed_hot_t}

    RDIST = "modulatory_channel_route_range_mean"
    PDIST = "cand_world_pairwise_dist_mean"
    SENT = "selected_action_class_entropy"

    # READINESS (a): IN-ARM route-range gate on the PRIMARY cell.
    a1b1_route_mean = _mean_key(a1b1, RDIST)
    route_seeds_ok = _n_seeds(a1b1, lambda r: float(r.get(RDIST, 0.0)) > ROUTE_RANGE_FLOOR)
    route_ready = bool(route_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # READINESS (b): e2.world_forward prediction spread on the PRIMARY cell.
    a1b1_pdist_mean = _mean_key(a1b1, PDIST)
    c1_seeds_ok = _n_seeds(a1b1, lambda r: float(r.get(PDIST, 0.0)) > C1_PAIRWISE_DIST_FLOOR)
    c1_pass = bool(c1_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # READINESS (c): Factor-B lever-engaged non-vacuity on the PRIMARY (A0B1)
    # cell. 689c isolates Factor B (factor_a OFF), so k_effective is CONSTANT by
    # design and k_varies would always fail -- it is computed + REPORTED but NOT
    # gated. The gate requires ONLY that the gap-scaled commit T_eff varies. A
    # flat T_eff means Factor B could not act -> substrate_not_ready_requeue.
    k_varies_seeds = _n_seeds(a1b1, lambda r: float(r.get("k_effective_spread", 0.0)) >= 1.0)
    t_varies_seeds = _n_seeds(a1b1, lambda r: float(r.get("t_eff_spread", 0.0)) > 1e-6)
    levers_engaged = bool(t_varies_seeds >= MIN_SEEDS_FOR_PASS)

    # gap-spread DIAGNOSTIC only (reported; does NOT gate readiness in 689a).
    gap_spread_seeds = _n_seeds(
        a1b1, lambda r: int(r.get("n_populated_gap_bins", 0)) >= GAP_MIN_POPULATED_BINS
    )

    readiness_ok = bool(route_ready and c1_pass and levers_engaged)

    # C_PRIMARY (gate, off-ramp): ARM_A1B1 selected-action entropy STRICTLY ABOVE
    # both the collapsed-proposer baseline AND the matched-noise control.
    def _strict_above_pair(r1: Dict[str, Any], by_seed_a, by_seed_b) -> bool:
        ra = by_seed_a.get(r1["seed"])
        rb = by_seed_b.get(r1["seed"])
        if ra is None or rb is None:
            return False
        e1 = float(r1.get(SENT, 0.0))
        return e1 > float(ra.get(SENT, 0.0)) and e1 > float(rb.get(SENT, 0.0))

    primary_seeds_ok = _n_seeds(
        a1b1, lambda r: _strict_above_pair(r, proposer_by_seed, noise_by_seed)
    )
    a1b1_sel_mean = _mean_key(a1b1, SENT)
    primary_floor_ok = bool(a1b1_sel_mean > C3_SELECTED_ENTROPY_FLOOR)
    primary_pass = bool(primary_seeds_ok >= MIN_SEEDS_FOR_PASS and primary_floor_ok)

    # C_GAPBLIND_B (LOAD-BEARING): ARM_A0B1 (PRIMARY) STRICTLY ABOVE the flat-hot
    # Factor-B control ARM_FIXED_HOT_T on the same seed. If the flat-hot control
    # matches A0B1, the conversion is NOT gap-SCALING -- it is just a hotter flat
    # softmax. (Single-control: _strict_above_pair against fixed_hot_t for BOTH
    # by_seed args yields "above fixed_hot_t AND above fixed_hot_t" = "above
    # fixed_hot_t".)
    gapblind_seeds_ok = _n_seeds(
        a1b1, lambda r: _strict_above_pair(r, fixed_hot_t_by_seed, fixed_hot_t_by_seed)
    )
    gapblind_pass = bool(gapblind_seeds_ok >= MIN_SEEDS_FOR_PASS and primary_floor_ok)

    # NEGATIVE-CONTROL sanity (informational, does NOT gate): matched-noise must
    # NOT lift committed entropy over the collapsed-proposer baseline.
    def _noise_lifts(rn: Dict[str, Any]) -> bool:
        rp = proposer_by_seed.get(rn["seed"])
        if rp is None:
            return False
        return float(rn.get(SENT, 0.0)) > float(rp.get(SENT, 0.0))
    noise_lift_seeds = _n_seeds(noise, _noise_lifts)
    negative_control_does_not_lift = bool(noise_lift_seeds == 0)

    # Control-engagement diagnostics (informational; confirm the gap-blind
    # Factor-B control applied its flat hot commit-T).
    a0b0_shortlist_mean = _mean_key(a0b0, "modulatory_shortlist_size_mean")
    fixed_hot_t_teff_mean = _mean_key(fixed_hot_t, "t_eff_mean")

    # Non-degeneracy: every measured arm produced P1 ticks.
    all_arms = [proposer, noise, a0b0, a0b1, a1b1, fixed_hot_t]
    non_degenerate = bool(
        all(len(a) > 0 for a in all_arms)
        and all(int(r.get("n_p1_ticks", 0)) > 0 for a in all_arms for r in a)
    )

    # C_FGAP secondary readouts (NON-GATING): fixed-width AND quantile-adaptive.
    a1b1_slope_fw, a1b1_nbins_fw, a1b1_bin_fw = _binned_entropy_slope(_pool_gap_bins(a1b1))
    a0b0_slope_fw, a0b0_nbins_fw, a0b0_bin_fw = _binned_entropy_slope(_pool_gap_bins(a0b0))
    fgap_fw_computable = bool(a1b1_slope_fw is not None and a0b0_slope_fw is not None)

    a1b1_q_slope = _mean_quantile_slope(a1b1)
    a0b0_q_slope = _mean_quantile_slope(a0b0)
    fgap_q_computable = bool(a1b1_q_slope is not None and a0b0_q_slope is not None)
    if fgap_q_computable:
        fgap_q_negative = bool(a1b1_q_slope < 0.0)
        fgap_q_more_concentrated = bool(a1b1_q_slope <= a0b0_q_slope - GAP_SLOPE_DELTA)
    else:
        fgap_q_negative = False
        fgap_q_more_concentrated = False

    # VERDICT resolver (689c): readiness -> C_PRIMARY -> C_GAPBLIND_B. The
    # load-bearing test is ARM_A0B1 strict-above the flat-hot Factor-B control.
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif not primary_pass:
        label = "conversion_ceiling_persists_despite_conflict_grade"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif not gapblind_pass:
        label = "gap_scaled_t_lift_matches_flat_hot_control"
        overall_pass = False
        evidence_direction = "non_contributory"
    else:
        label = "factor_b_gap_scaled_t_converts_committed_diversity"
        overall_pass = True
        evidence_direction = "supports"

    # Dissociation (informational): each cell's entropy + how many seeds it is
    # strict-above both collapsed controls + above the flat-hot Factor-B control.
    def _cell_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "selected_entropy_mean": round(_mean_key(rows, SENT), 6),
            "seeds_strict_above_both_collapsed_controls": int(
                _n_seeds(rows, lambda r: _strict_above_pair(r, proposer_by_seed, noise_by_seed))
            ),
            "seeds_strict_above_flat_hot_t_control": int(
                _n_seeds(rows, lambda r: _strict_above_pair(r, fixed_hot_t_by_seed, fixed_hot_t_by_seed))
            ),
        }

    return {
        "readiness": {
            "route_range_floor": ROUTE_RANGE_FLOOR,
            "a1b1_route_range_mean": round(a1b1_route_mean, 6),
            "a1b1_seeds_route_above_floor": int(route_seeds_ok),
            "route_ready": route_ready,
            "c1_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
            "a1b1_pairwise_dist_mean": round(a1b1_pdist_mean, 6),
            "a1b1_seeds_e2_divergent": int(c1_seeds_ok),
            "c1_pass": c1_pass,
            "levers_engaged": {
                "k_varies_seeds": int(k_varies_seeds),
                "t_eff_varies_seeds": int(t_varies_seeds),
                "levers_engaged_ok": levers_engaged,
                "note": (
                    "689c gates the Factor-B lever ONLY: t_eff must vary. Factor A "
                    "is OFF in the A0B1 primary, so k_effective is CONSTANT by "
                    "design -- k_varies_seeds is reported, NOT gated."
                ),
            },
            "gap_spread_diagnostic": {
                "gap_spread_seeds": int(gap_spread_seeds),
                "gap_min_populated_bins": GAP_MIN_POPULATED_BINS,
                "note": (
                    "DIAGNOSTIC ONLY in 689c -- DECOUPLED from the readiness gate. "
                    "A Factor-B-engaged + gap-pinned run reaches the flat-hot arm "
                    "contrast instead of self-routing substrate_not_ready_requeue."
                ),
            },
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "readiness_ok": readiness_ok,
        },
        "c_primary": {
            "a1b1_seeds_strict_above_both_collapsed_controls": int(primary_seeds_ok),
            "a1b1_selected_entropy_mean": round(a1b1_sel_mean, 6),
            "selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
            "primary_floor_ok": primary_floor_ok,
            "c_primary_pass": primary_pass,
            "note": (
                "Off-ramp discriminator: ARM_A0B1 (PRIMARY) strict-above BOTH "
                "collapsed controls (proposer + matched-noise). Fail = no lift at "
                "all -> conversion_ceiling_persists_despite_conflict_grade (V4 "
                "off-ramp)."
            ),
        },
        "c_gapblind_b": {
            "a1b1_seeds_strict_above_flat_hot_t_control": int(gapblind_seeds_ok),
            "a1b1_selected_entropy_mean": round(a1b1_sel_mean, 6),
            "fixed_hot_t_selected_entropy_mean": round(_mean_key(fixed_hot_t, SENT), 6),
            "selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
            "c_gapblind_b_pass": gapblind_pass,
            "note": (
                "LOAD-BEARING: ARM_A0B1 (PRIMARY, gap-SCALED commit-T) STRICTLY "
                "ABOVE the flat-hot Factor-B control ARM_FIXED_HOT_T (flat T=2.5 "
                "over the divergent source) -> the conversion is gap-SCALING, not "
                "a hotter flat softmax. Fail = the flat-hot control matches A0B1 "
                "-> the lift is not gap-scaling."
            ),
        },
        "control_engagement_diagnostic": {
            "a0b0_shortlist_size_mean": round(a0b0_shortlist_mean, 6),
            "fixed_hot_t_teff_mean": round(fixed_hot_t_teff_mean, 6),
            "note": (
                "Informational: confirm the flat-hot Factor-B control applied its "
                "fixed lever -- ARM_FIXED_HOT_T t_eff ~ 2.5 (flat hot commit-T). "
                "Does NOT gate."
            ),
        },
        "c_fgap_secondary": {
            "fixed_width": {
                "a1b1_gap_slope": (round(a1b1_slope_fw, 6) if a1b1_slope_fw is not None else None),
                "a1b1_gap_bins_used": int(a1b1_nbins_fw),
                "a1b1_per_bin_entropy": a1b1_bin_fw,
                "a0b0_gap_slope": (round(a0b0_slope_fw, 6) if a0b0_slope_fw is not None else None),
                "a0b0_gap_bins_used": int(a0b0_nbins_fw),
                "a0b0_per_bin_entropy": a0b0_bin_fw,
                "fgap_computable": fgap_fw_computable,
            },
            "quantile_adaptive": {
                "a1b1_gap_slope_mean": (round(a1b1_q_slope, 6) if a1b1_q_slope is not None else None),
                "a0b0_gap_slope_mean": (round(a0b0_q_slope, 6) if a0b0_q_slope is not None else None),
                "gap_slope_delta_required": GAP_SLOPE_DELTA,
                "fgap_computable": fgap_q_computable,
                "fgap_slope_negative": fgap_q_negative,
                "fgap_more_concentrated_than_fixed_k": fgap_q_more_concentrated,
            },
            "note": (
                "SECONDARY readout, NON-GATING in 689c. The fixed-width regression "
                "is the 689 metric (usually uncomputable on the near-tie-pinned "
                "foraging gap). The quantile-adaptive binning (equal-mass bins "
                "matched to the gap distribution) restores >= 3 populated bins so "
                "the entropy-vs-gap slope is computable even on a concentrated gap. "
                "The LOAD-BEARING test is the flat-hot arm contrast (c_gapblind_b)."
            ),
        },
        "negative_control": {
            "matched_noise_lift_seeds_over_proposer": int(noise_lift_seeds),
            "negative_control_does_not_lift": negative_control_does_not_lift,
            "negative_control_unexpectedly_lifted": bool(not negative_control_does_not_lift),
            "note": (
                "ARM_MATCHED_NOISE (proposer @ flat T=2.5) MUST NOT lift committed "
                "entropy over ARM_PROPOSER_CTRL (collapsed channel). Informational."
            ),
        },
        "two_by_two_dissociation": {
            "ARM_A0B0_both_off": _cell_summary(a0b0),
            "ARM_A0B1_factor_b_only_primary": _cell_summary(a0b1),
            "ARM_A1B1_both_on_confirmatory": _cell_summary(confirm),
            "ARM_FIXED_HOT_T_gap_blind_b": _cell_summary(fixed_hot_t),
            "note": (
                "Informational dissociation. The PRIMARY gates on ARM_A0B1 vs the "
                "flat-hot Factor-B control (c_gapblind_b); these per-cell readouts "
                "dissociate Factor B alone (A0B1, primary) vs both levers (A1B1, "
                "confirmatory; expected to LOSE) vs the gap-blind flat-hot control."
            ),
        },
        "selected_action_entropy_per_arm_mean": {
            PROPOSER_CTRL_ARM: round(_mean_key(proposer, SENT), 6),
            MATCHED_NOISE_ARM: round(_mean_key(noise, SENT), 6),
            BASELINE_ARM: round(_mean_key(a0b0, SENT), 6),
            PRIMARY_ARM: round(a1b1_sel_mean, 6),
            CONFIRM_ARM: round(_mean_key(confirm, SENT), 6),
            FIXED_HOT_T_ARM: round(_mean_key(fixed_hot_t, SENT), 6),
        },
        "route_range_per_arm_mean": {
            PROPOSER_CTRL_ARM: round(_mean_key(proposer, RDIST), 6),
            MATCHED_NOISE_ARM: round(_mean_key(noise, RDIST), 6),
            BASELINE_ARM: round(_mean_key(a0b0, RDIST), 6),
            PRIMARY_ARM: round(a1b1_route_mean, 6),
            CONFIRM_ARM: round(_mean_key(confirm, RDIST), 6),
            FIXED_HOT_T_ARM: round(_mean_key(fixed_hot_t, RDIST), 6),
        },
        "label": label,
        "evidence_direction": evidence_direction,
        "overall_pass": overall_pass,
        "preconditions": [
            {
                "name": "a0b1_modulatory_channel_route_range_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_A0B1 (PRIMARY) IN-ARM RAW cross-candidate RANGE of the "
                    "modulatory bias routed into the E3 selection authority "
                    "(modulatory_channel_route_range, read LIVE at the select "
                    "tick) clears the floor. SAME range statistic the route-range "
                    "substrate gates on. Below floor => routing not wired / e2 "
                    "under-trained => substrate_not_ready_requeue."
                ),
                "control": (
                    "ARM_A0B1: candidate_summary_source=e2_world_forward, "
                    "route-range routing + top-k shortlist ON"
                ),
                "measured": round(a1b1_route_mean, 6),
                "threshold": ROUTE_RANGE_FLOOR,
                "met": route_ready,
            },
            {
                "name": "a0b1_e2_world_forward_prediction_spread_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_A0B1 e2.world_forward(z0, a_i) per-candidate prediction "
                    "spread (cand_world_pairwise_dist) clears the floor -- SD-056 "
                    "trained the action-conditional divergence the eligible set "
                    "needs. RANGE statistic. Below floor => SD-056 under-trained "
                    "=> substrate_not_ready_requeue."
                ),
                "control": "ARM_A0B1: agent.e2.cand_world_pairwise_dist on SP-CEM candidates",
                "measured": round(a1b1_pdist_mean, 6),
                "threshold": C1_PAIRWISE_DIST_FLOOR,
                "met": c1_pass,
            },
            {
                "name": "a0b1_factor_b_lever_engaged_non_vacuous",
                "kind": "readiness",
                "description": (
                    "ARM_A0B1 gap-scaled commit-temperature t_eff varies across "
                    "P1 ticks (spread > eps). Factor A is OFF in the A0B1 primary, "
                    "so k_effective is CONSTANT by design (k_varies reported, NOT "
                    "gated). A flat T_eff means Factor B could not act => "
                    "substrate_not_ready_requeue. Count statistic (seeds meeting "
                    "t_eff-varies)."
                ),
                "control": (
                    "ARM_A0B1: factor_b ON (gap-scaled commit-T); t_eff_spread "
                    "read live; k_effective_spread reported"
                ),
                "measured": int(t_varies_seeds),
                "threshold": MIN_SEEDS_FOR_PASS,
                "met": levers_engaged,
            },
        ],
        "criteria": [
            {"name": "C1_a0b1_e2_world_forward_divergent", "load_bearing": True, "passed": c1_pass},
            {"name": "C_PRIMARY_a0b1_selected_entropy_strict_above_both_collapsed_controls",
             "load_bearing": True, "passed": primary_pass},
            {"name": "C_GAPBLIND_B_a0b1_selected_entropy_strict_above_flat_hot_t_control",
             "load_bearing": True, "passed": gapblind_pass},
            {"name": "C_FGAP_quantile_slope_correlates_with_F_gap",
             "load_bearing": False, "passed": bool(fgap_q_more_concentrated)},
        ],
        "criteria_non_degenerate": {
            "C1": non_degenerate,
            "C_PRIMARY": non_degenerate,
            "C_GAPBLIND_B": non_degenerate,
        },
        "non_degenerate": non_degenerate,
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
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {
                        k: arm[k]
                        for k in (
                            "arm_id", "candidate_summary_source", "temperature",
                            "factor_a", "factor_b", "shortlist_k", "gap_scaled_alpha",
                        )
                    },
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "conflict_grade_config": {
                        "modulatory_shortlist_k_max": MODULATORY_SHORTLIST_K_MAX,
                        "gap_scaled_commit_harm_floor": GAP_SCALED_COMMIT_HARM_FLOOR,
                    },
                    "conversion_constant": {
                        "use_modulatory_channel_routing": True,
                        "modulatory_channel_route_source": "cand_world_summary",
                        "use_modulatory_selection_authority": True,
                        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
                        "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
                        "use_modulatory_shortlist_then_modulate": True,
                        "modulatory_shortlist_mode": MODULATORY_SHORTLIST_MODE,
                    },
                    "p0_episodes": p0, "p1_episodes": p1, "steps_per_episode": steps,
                },
                seed=seed,
                script_path=Path(__file__),
                rng_fully_reset=True,
                extra_ineligible_reasons=["online_e2_training_stateful_per_cell"],
            )
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    evidence_direction = summary["evidence_direction"]

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "result": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-439": evidence_direction},
        "non_degenerate": summary.get("non_degenerate", True),
        "evidence_direction_note": (
            "MECH-439 Factor-B-alone (gap-scaled commit-temperature, A0B1) isolation "
            "retest. Does NOT supersede anything -- it isolates a sub-lever; 689a's "
            "combined-conflict-grade verdict stands. 689a was an 8-arm 2x2 falsifier "
            "whose PRIMARY (ARM_A1B1, both levers) FAILED, but its decomposition showed "
            "ARM_A0B1 (Factor B ALONE: factor_a OFF / factor_b ON) was the ONLY "
            "converter. 689c re-points the pre-registered PRIMARY to ARM_A0B1, drops "
            "the two Factor-A-only arms, and keeps ARM_A1B1 as a CONFIRMATORY arm "
            "(both levers; should LOSE to A0B1 -- Factor A poisons B). The LOAD-BEARING "
            "test is C_GAPBLIND_B: ARM_A0B1 strict-above the flat-hot Factor-B control "
            "ARM_FIXED_HOT_T (flat T=2.5 over the divergent e2_world_forward source) on "
            "committed-action class entropy on >=2/3 seeds -- the conversion is "
            "gap-SCALING, not a hotter flat softmax. PASS "
            "(label=factor_b_gap_scaled_t_converts_committed_diversity) = READINESS "
            "(IN-ARM route-range + e2-divergence + Factor-B lever-engaged: t_eff varies; "
            "k is CONSTANT by design, reported not gated) AND C_PRIMARY (strict-above "
            "both collapsed controls) AND C_GAPBLIND_B. Route/e2-div below floor OR "
            "t_eff not varied self-routes substrate_not_ready_requeue (non_contributory, "
            "NEVER a weakens). Readiness met, no lift vs collapsed => "
            "conversion_ceiling_persists_despite_conflict_grade (V4 off-ramp). Readiness "
            "+ lift but the flat-hot control matches A0B1 => "
            "gap_scaled_t_lift_matches_flat_hot_control (the lift is not gap-scaling). "
            "Quantile-adaptive gap binning + the fixed-width regression are SECONDARY "
            "readouts (c_fgap_secondary), NON-GATING. NO weakens path. claim_ids=[MECH-439] "
            "only; ARC-065/MECH-341/ARC-062/MECH-309/MECH-294 untouched."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "factor_b_gap_scaled_t_converts_committed_diversity": "PASS -> Factor B isolated; the committed-entropy lift is gap-SCALED (ARM_A0B1 beats the flat-hot Factor-B control); toward supports",
                "substrate_not_ready_requeue": "routing/SD-056 under-trained OR Factor-B t_eff lever not varied; re-queue; do NOT weaken MECH-439",
                "conversion_ceiling_persists_despite_conflict_grade": "OFF-RAMP -> the V4 directions in conversion_ceiling_phase0_synthesis (divisive normalization / output-null subspace / cross-tick QD archive); NOT a falsification of MECH-439",
                "gap_scaled_t_lift_matches_flat_hot_control": "the lift is real but the gap-blind flat-hot-T control matches A0B1 -> NOT gap-concentrated -> the lift is not gap-scaling",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": ENV_KWARGS,
            "arms": [
                {k: a[k] for k in (
                    "arm_id", "label", "candidate_summary_source", "temperature",
                    "factor_a", "factor_b", "shortlist_k", "gap_scaled_alpha",
                )}
                for a in ARMS
            ],
            "sd056_weight": SD056_WEIGHT,
            "matched_entropy_temperature": MATCHED_ENTROPY_TEMPERATURE,
            "conflict_grade_config": {
                "modulatory_shortlist_k_baseline": MODULATORY_SHORTLIST_K,
                "modulatory_shortlist_k_max": MODULATORY_SHORTLIST_K_MAX,
                "gap_scaled_commit_entropy_alpha": GAP_SCALED_COMMIT_ENTROPY_ALPHA,
                "gap_scaled_commit_harm_floor": GAP_SCALED_COMMIT_HARM_FLOOR,
            },
            "conversion_constant": {
                "use_modulatory_channel_routing": True,
                "modulatory_channel_route_source": "cand_world_summary",
                "use_modulatory_selection_authority": True,
                "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
                "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
                "use_modulatory_shortlist_then_modulate": True,
                "modulatory_shortlist_mode": MODULATORY_SHORTLIST_MODE,
            },
            "thresholds": {
                "route_range_floor": ROUTE_RANGE_FLOOR,
                "c1_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
                "c3_selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
                "gap_slope_delta_secondary": GAP_SLOPE_DELTA,
                "fgap_nbins": FGAP_NBINS,
                "fgap_min_bin_ticks": FGAP_MIN_BIN_TICKS,
                "gap_min_populated_bins_diagnostic": GAP_MIN_POPULATED_BINS,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_route_range_ready": summary["readiness"]["route_ready"],
            "readiness_e2_divergent_ready": summary["readiness"]["c1_pass"],
            "readiness_levers_engaged": summary["readiness"]["levers_engaged"]["levers_engaged_ok"],
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "negative_control_does_not_lift": summary["negative_control"]["negative_control_does_not_lift"],
            "C_PRIMARY_a0b1_strict_above_both_collapsed_controls": summary["c_primary"]["c_primary_pass"],
            "C_GAPBLIND_B_a0b1_strict_above_flat_hot_t_control": summary["c_gapblind_b"]["c_gapblind_b_pass"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} (label={summary['label']}, evidence_direction={evidence_direction})", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-689c MECH-439 Factor-B-alone (gap-scaled commit-T, A0B1) isolation retest"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
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
