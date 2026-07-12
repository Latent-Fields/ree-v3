"""V3-EXQ-689: MECH-439 F-dominance conflict-grade 2-factor (2x2) discriminating
falsifier on the GAP-A-ready foraging substrate -- FIRST falsifier for MECH-439.

conversion_ceiling_phase0_synthesis_2026-06-18.md (Phase 0 four-root + Phase 1
fork-resolution + Phase 1 backfill). The live root of the committed-action-diversity
conversion ceiling is F-DOMINANCE (MECH-439): the primary harm/goal score F carries
~88-89% of E3 committed-selection variance (V3-EXQ-571), UNCHANGED by the diversity
stack, so every diversity channel reaches the E3 accumulator but cannot move the
F-dominated committed argmax. The standalone 569i top-k PASS (2/3) only thinly cleared
(0.711 vs proposer 0.650). The backfill found the two V3-tractable levers are TWO
RENDERINGS OF ONE PRINCIPLE -- the BG hyperdirect conflict-grade (grade the committed
decision by the normalized top-F gap) -- and must be tested TOGETHER as a 2-factor
experiment.

THE TWO LEVERS (landed in e3_selector.py, both no-op-default behind own E3Config flags):
  FACTOR A -- conflict-graded shortlist width: the existing top-k shortlist k is
    replaced by k = clamp(round(k_max - (k_max-1)*gap_norm), 1, K). Near-ties widen k
    (slower commit); a decisive F-gap narrows k to 1. F gates ELIGIBILITY only; absent
    from the within-set arbitration. SAFETY: clearly-harmful candidates (large F-gap
    above the best) are never admitted.
  FACTOR B -- gap-scaled entropy-regularized commit: the committed argmin becomes
    multinomial(softmax(-q / T_eff)) over the eligible set, T_eff = base_T +
    entropy_alpha*(1 - gap_norm). Near-ties hotter; decisive gap cold (preserves the
    F-winner). q = routed modulatory channel within the F-bounded shortlist (safety).

DESIGN: a 2x2 treatment grid (Factor A {fixed-k=3 | conflict-graded k} x Factor B
{hard argmin | gap-scaled commit-T}) PLUS two pre-registered controls, all matched-seed.
ALL six arms share the GAP-A-ready conversion constant: SD-056 online-trained
e2.world_forward (rollout-norm clamp ON, 643a lesson) + ARC-065 GAP-A
candidate_summary_source=e2_world_forward (the divergent eligible set -- the NON-VACUITY
precondition) + route-range routing + top-k shortlist + SP-CEM Layer A + shared E3-side
bias channels (lateral_pfc + mech295). The CRF stack stays constant (off).
  ARM_PROPOSER_CTRL  source=proposer,         T=1.0, fixed-k, hard argmin  (collapsed-channel baseline; the no-conversion-reaches control)
  ARM_MATCHED_NOISE  source=proposer,         T=2.5, fixed-k, hard argmin  (matched-entropy NEGATIVE control -- flat hotter softmax over a collapsed channel)
  ARM_A0B0           source=e2_world_forward,  T=1.0, fixed-k, hard argmin  (divergent pool; both levers OFF = the 569i baseline cell)
  ARM_A1B0           source=e2_world_forward,  T=1.0, GRADED-k, hard argmin (Factor A only)
  ARM_A0B1           source=e2_world_forward,  T=1.0, fixed-k, GAP-SCALED-T (Factor B only)
  ARM_A1B1           source=e2_world_forward,  T=1.0, GRADED-k, GAP-SCALED-T (BOTH; the PRIMARY cell)

Factor B's gap-scaled T peaks at base + entropy_alpha at near-ties; with
entropy_alpha=1.5 + base 1.0 the peak is 2.5 -- the SAME peak as the flat matched-noise
control, but applied ONLY at near-ties. Beating the flat-2.5 control therefore isolates
the GAP-CONCENTRATION of the heat (the load-bearing property), not raw temperature.

ACCEPTANCE (evidence, claim_ids=[MECH-439]; this is MECH-439's FIRST falsifier):
  READINESS (load-bearing non-vacuity; below any -> substrate_not_ready_requeue, NEVER
  a weakens -- RANGE statistics, the same-statistic discipline):
    (a) IN-ARM ROUTE-RANGE: ARM_A1B1 modulatory_channel_route_range mean (read LIVE at
        the select tick) > ROUTE_RANGE_FLOOR on >= MIN_SEEDS_FOR_PASS seeds (the divergent
        channel reaches the bias the shortlist arbitrates).
    (b) E2-DIVERGENCE: ARM_A1B1 cand_world_pairwise_dist (e2.world_forward prediction
        spread) > C1_PAIRWISE_DIST_FLOOR on >= MIN_SEEDS_FOR_PASS seeds (SD-056 trained
        the action-conditional divergence the eligible set needs).
    (c) GRADING-ENGAGED non-vacuity: in ARM_A1B1, k_effective AND T_eff actually VARY
        across P1 ticks (range >= 1 for k, > eps for T_eff) AND the per-tick F-gap is
        spread across >= GAP_MIN_POPULATED_BINS gap bins. A flat k / flat T_eff / single-
        gap distribution means the grading could not act (the 684 margin-collapse / a
        monostrategy-pinned gap) -> the test is vacuous -> substrate_not_ready_requeue.
  C_PRIMARY (load-bearing): ARM_A1B1 selected_action_class_entropy STRICTLY ABOVE BOTH
    ARM_PROPOSER_CTRL AND ARM_MATCHED_NOISE on the SAME seed, on >= MIN_SEEDS_FOR_PASS
    seeds, AND ARM_A1B1 mean > C3_SELECTED_ENTROPY_FLOOR (the robustness bar 569i only
    thinly cleared at 0.711 vs 0.650).
  C_FGAP (load-bearing PRE-REGISTERED FALSIFIER, shared by both factors): bin P1 ticks by
    top-F-gap and regress committed-action class entropy on the gap bin -- the LIFT MUST
    correlate with per-tick F-gap. Operationally: ARM_A1B1 per-gap-bin committed-entropy
    SLOPE vs gap is NEGATIVE (lift concentrated at near-ties) AND MORE NEGATIVE than the
    fixed-k/hard-argmin ARM_A0B0 baseline by >= GAP_SLOPE_DELTA. A flat (uniform) lift =
    the grading adds nothing over a bigger fixed shortlist / hotter flat softmax -> the
    conflict-grading is NOT the mechanism.
  PASS = READINESS AND C_PRIMARY AND C_FGAP -> conflict_grade_converts_committed_diversity;
    evidence_direction=supports.

Interpretation grid:
| outcome                                          | label                                          | evidence_direction | next                                                                                  |
|--------------------------------------------------|------------------------------------------------|--------------------|---------------------------------------------------------------------------------------|
| READINESS + C_PRIMARY + C_FGAP                   | conflict_grade_converts_committed_diversity    | supports           | MECH-439 first falsifier PASS; conflict-grade moves committed behaviour, gap-correlated |
| route/e2-div below floor; k/T_eff/gap not spread | substrate_not_ready_requeue                    | non_contributory   | routing/SD-056 under-trained OR monostrategy-pinned gap; re-queue at higher P0; NOT a weakens |
| READINESS met, C_PRIMARY fail                    | conversion_ceiling_persists_despite_conflict_grade | non_contributory | OFF-RAMP -> V4 directions (divisive normalization / output-null / cross-tick QD archive) per synthesis; NOT a falsification |
| READINESS + C_PRIMARY met, C_FGAP fail (uniform) | uniform_lift_grading_non_load_bearing          | non_contributory   | the lift is real but NOT gap-concentrated -> "a bigger fixed shortlist / hotter flat softmax"; route to the rank_preserving_F_to_eligibility_demotion V3 fallback |

NOTE: NO weakens path. MECH-439 is candidate; a PASS moves it toward supports; a
preconditions-met fail routes to the V4 directions in the synthesis doc, NOT a dead end.
The 2x2 dissociates which factor carries the lift (reported per cell, informational); the
PRIMARY gates on the both-levers cell ARM_A1B1. claim_ids=[MECH-439] only;
ARC-065/MECH-341/ARC-062/MECH-309/MECH-294 untouched.

Usage:
  /opt/local/bin/python3 experiments/v3_exq_689_mech439_conflict_grade_2factor_falsifier.py --dry-run
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_689_mech439_conflict_grade_2factor_falsifier"
QUEUE_ID = "V3-EXQ-689"
CLAIM_IDS: List[str] = ["MECH-439"]  # MECH-439's FIRST falsifier
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
C3_SELECTED_ENTROPY_FLOOR = 0.3   # C_PRIMARY: ARM_A1B1 selected-action class entropy floor
MATCHED_ENTROPY_TEMPERATURE = 2.5
MIN_SEEDS_FOR_PASS = 2            # of 3

# F-gap regression falsifier (C_FGAP) parameters.
FGAP_NBINS = 4                    # gap_norm in [0,1] -> 4 equal bins
FGAP_MIN_BIN_TICKS = 50          # a bin counts toward the slope only if it has >= this many ticks
GAP_MIN_POPULATED_BINS = 3       # READINESS (c): the per-tick F-gap must span >= 3 bins
GAP_SLOPE_DELTA = 0.03           # C_FGAP: ARM_A1B1 slope must be <= ARM_A0B0 slope - this (more gap-concentrated)

# Conflict-grade lever config (MECH-439).
MODULATORY_SHORTLIST_K = 3                    # fixed-k baseline (Factor A OFF)
MODULATORY_SHORTLIST_K_MAX = 6               # conflict-graded k_max (Factor A ON): k ranges 1..6
GAP_SCALED_COMMIT_ENTROPY_ALPHA = 1.5        # Factor B: T_eff peak = 1.0 + 1.5 = 2.5 (== matched-noise flat T, but gap-concentrated)
GAP_SCALED_COMMIT_HARM_FLOOR = 0.25          # Factor B standalone safety envelope (unused under the shortlist)

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

# 2x2 treatment grid + 2 controls. factor_a = conflict-graded width; factor_b = gap-scaled commit-T.
PRIMARY_ARM = "ARM_A1B1"
BASELINE_ARM = "ARM_A0B0"
PROPOSER_CTRL_ARM = "ARM_PROPOSER_CTRL"
MATCHED_NOISE_ARM = "ARM_MATCHED_NOISE"

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": PROPOSER_CTRL_ARM,
        "label": "proposer_collapsed_channel_baseline_control",
        "candidate_summary_source": "proposer",
        "temperature": 1.0,
        "factor_a": False,
        "factor_b": False,
    },
    {
        "arm_id": MATCHED_NOISE_ARM,
        "label": "proposer_matched_entropy_flat_temperature_negative_control",
        "candidate_summary_source": "proposer",
        "temperature": MATCHED_ENTROPY_TEMPERATURE,
        "factor_a": False,
        "factor_b": False,
    },
    {
        "arm_id": BASELINE_ARM,
        "label": "e2wf_fixed_k_hard_argmin_both_levers_off",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "factor_a": False,
        "factor_b": False,
    },
    {
        "arm_id": "ARM_A1B0",
        "label": "e2wf_conflict_graded_k_hard_argmin_factor_a_only",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "factor_a": True,
        "factor_b": False,
    },
    {
        "arm_id": "ARM_A0B1",
        "label": "e2wf_fixed_k_gap_scaled_commit_t_factor_b_only",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "factor_a": False,
        "factor_b": True,
    },
    {
        "arm_id": PRIMARY_ARM,
        "label": "e2wf_conflict_graded_k_gap_scaled_commit_t_both_levers",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "factor_a": True,
        "factor_b": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """GAP-A-ready conversion stack (SP-CEM + SD-056 online + ARC-065 GAP-A
    candidate_summary_source + route-range routing + top-k shortlist + shared
    lateral_pfc/mech295 bias channels), with the MECH-439 conflict-grade levers
    toggled per arm. Factor A (conflict-graded width) and Factor B (gap-scaled
    commit-T) default OFF -> bit-identical to the 569i fixed-k/hard-argmin cell."""
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
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        # --- MECH-439 FACTOR A: conflict-graded shortlist width (per arm) ---
        modulatory_shortlist_conflict_graded=bool(arm["factor_a"]),
        modulatory_shortlist_k_max=MODULATORY_SHORTLIST_K_MAX,
        # --- MECH-439 FACTOR B: gap-scaled entropy-regularized commit (per arm) ---
        use_gap_scaled_commit_temperature=bool(arm["factor_b"]),
        gap_scaled_commit_entropy_alpha=GAP_SCALED_COMMIT_ENTROPY_ALPHA,
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


def _binned_entropy_slope(
    bin_class_counts: List[Dict[int, int]],
) -> Tuple[Optional[float], int, List[Optional[float]]]:
    """Least-squares slope of per-gap-bin committed-action entropy vs the bin
    midpoint, over bins with >= FGAP_MIN_BIN_TICKS ticks. Returns
    (slope_or_None, n_bins_used, per_bin_entropy[NBINS]). A NEGATIVE slope means
    the committed diversity is concentrated at low F-gap (near-ties) -- the
    conflict-grade signature."""
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
    n_used = len(xs)
    if n_used < 3:
        return None, n_used, per_bin_entropy
    n = float(n_used)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 1e-12:
        return None, n_used, per_bin_entropy
    return float(sxy / sxx), n_used, per_bin_entropy


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
    # F-gap regression falsifier: per-gap-bin committed-action class counts.
    gap_bin_class_counts: List[Counter] = [Counter() for _ in range(FGAP_NBINS)]
    gap_bin_tick_counts: List[int] = [0 for _ in range(FGAP_NBINS)]
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
                # Factor B effective commit temperature (only set when gap-scaled fired).
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

            # C_PRIMARY + C_FGAP behavioural DV: committed first-action class, binned by F-gap.
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

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "candidate_summary_source": arm["candidate_summary_source"],
        "temperature": arm_temperature,
        "factor_a_conflict_graded": bool(arm["factor_a"]),
        "factor_b_gap_scaled_commit": bool(arm["factor_b"]),
        "n_p1_ticks": int(n_p1_ticks),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # READINESS (a) / IN-ARM route-range.
        "modulatory_channel_route_range_mean": round(_mean(route_ranges), 6),
        "modulatory_channel_route_range_max": round(route_range_max, 6),
        "modulatory_authority_active_ticks": int(authority_active_ticks),
        # Shortlist discriminator.
        "modulatory_shortlist_size_mean": round(_mean(shortlist_sizes), 6),
        "modulatory_shortlist_active_ticks": int(shortlist_active_ticks),
        "modulatory_shortlist_mode": shortlist_mode_seen or "",
        # READINESS (b) / e2.world_forward prediction spread.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        # READINESS (c) / grading-engaged non-vacuity.
        "k_effective_mean": round(_mean(k_eff_values), 6),
        "k_effective_spread": round(_spread(k_eff_values), 6),
        "k_effective_n_ticks": int(len(k_eff_values)),
        "t_eff_mean": round(_mean(t_eff_values), 6),
        "t_eff_spread": round(_spread(t_eff_values), 6),
        "t_eff_n_ticks": int(len(t_eff_values)),
        "n_populated_gap_bins": int(n_populated_gap_bins),
        # C_PRIMARY behavioural DV.
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": int(len(selected_class_counts)),
        # C_FGAP: per-gap-bin committed-action class counts (for the regression).
        "gap_bin_class_counts": [dict(sorted(c.items())) for c in gap_bin_class_counts],
        "gap_bin_tick_counts": list(gap_bin_tick_counts),
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


def _pool_gap_bins(rows: List[Dict[str, Any]]) -> List[Dict[int, int]]:
    """Sum per-gap-bin committed-class counts across seeds for one arm."""
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
    a1b0 = _arm_rows(arm_results, "ARM_A1B0")
    a0b1 = _arm_rows(arm_results, "ARM_A0B1")
    a1b1 = _arm_rows(arm_results, PRIMARY_ARM)

    proposer_by_seed = {r["seed"]: r for r in proposer}
    noise_by_seed = {r["seed"]: r for r in noise}

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

    # READINESS (c): grading-engaged non-vacuity on the PRIMARY cell.
    #   k_effective must vary (Factor A engaged), T_eff must vary (Factor B engaged),
    #   and the per-tick F-gap must span >= GAP_MIN_POPULATED_BINS bins.
    k_varies_seeds = _n_seeds(a1b1, lambda r: float(r.get("k_effective_spread", 0.0)) >= 1.0)
    t_varies_seeds = _n_seeds(a1b1, lambda r: float(r.get("t_eff_spread", 0.0)) > 1e-6)
    gap_spread_seeds = _n_seeds(
        a1b1, lambda r: int(r.get("n_populated_gap_bins", 0)) >= GAP_MIN_POPULATED_BINS
    )
    grading_engaged = bool(
        k_varies_seeds >= MIN_SEEDS_FOR_PASS
        and t_varies_seeds >= MIN_SEEDS_FOR_PASS
        and gap_spread_seeds >= MIN_SEEDS_FOR_PASS
    )

    readiness_ok = bool(route_ready and c1_pass and grading_engaged)

    # C_PRIMARY (load-bearing): ARM_A1B1 selected-action entropy STRICTLY ABOVE both
    # the collapsed-proposer baseline AND the matched-noise control on the same seed.
    def _primary(r1: Dict[str, Any]) -> bool:
        rp = proposer_by_seed.get(r1["seed"])
        rn = noise_by_seed.get(r1["seed"])
        if rp is None or rn is None:
            return False
        e1 = float(r1.get(SENT, 0.0))
        return e1 > float(rp.get(SENT, 0.0)) and e1 > float(rn.get(SENT, 0.0))
    primary_seeds_ok = _n_seeds(a1b1, _primary)
    a1b1_sel_mean = _mean_key(a1b1, SENT)
    primary_floor_ok = bool(a1b1_sel_mean > C3_SELECTED_ENTROPY_FLOOR)
    primary_pass = bool(primary_seeds_ok >= MIN_SEEDS_FOR_PASS and primary_floor_ok)

    # C_FGAP (load-bearing): the lift must correlate with the per-tick F-gap. The
    # graded PRIMARY cell's per-gap-bin committed-entropy slope must be NEGATIVE (lift
    # concentrated at near-ties) AND MORE NEGATIVE than the fixed-k/hard-argmin
    # ARM_A0B0 baseline by >= GAP_SLOPE_DELTA (the grading adds gap-dependence the
    # fixed cell does not have). A uniform lift = a bigger fixed shortlist / hotter
    # flat softmax, NOT conflict-grading.
    a1b1_slope, a1b1_nbins, a1b1_bin_entropy = _binned_entropy_slope(_pool_gap_bins(a1b1))
    a0b0_slope, a0b0_nbins, a0b0_bin_entropy = _binned_entropy_slope(_pool_gap_bins(a0b0))
    fgap_computable = bool(a1b1_slope is not None and a0b0_slope is not None)
    if fgap_computable:
        fgap_negative = bool(a1b1_slope < 0.0)
        fgap_more_concentrated = bool(a1b1_slope <= a0b0_slope - GAP_SLOPE_DELTA)
        fgap_pass = bool(fgap_negative and fgap_more_concentrated)
    else:
        fgap_negative = False
        fgap_more_concentrated = False
        fgap_pass = False

    # NEGATIVE-CONTROL sanity (informational, does NOT gate): matched-noise must NOT
    # lift committed entropy over the collapsed-proposer baseline.
    def _noise_lifts(rn: Dict[str, Any]) -> bool:
        rp = proposer_by_seed.get(rn["seed"])
        if rp is None:
            return False
        return float(rn.get(SENT, 0.0)) > float(rp.get(SENT, 0.0))
    noise_lift_seeds = _n_seeds(noise, _noise_lifts)
    negative_control_does_not_lift = bool(noise_lift_seeds == 0)

    # Non-degeneracy: every measured arm produced P1 ticks.
    all_arms = [proposer, noise, a0b0, a1b0, a0b1, a1b1]
    non_degenerate = bool(
        all(len(a) > 0 for a in all_arms)
        and all(int(r.get("n_p1_ticks", 0)) > 0 for a in all_arms for r in a)
    )

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif not primary_pass:
        label = "conversion_ceiling_persists_despite_conflict_grade"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif not fgap_pass:
        label = "uniform_lift_grading_non_load_bearing"
        overall_pass = False
        evidence_direction = "non_contributory"
    else:
        label = "conflict_grade_converts_committed_diversity"
        overall_pass = True
        evidence_direction = "supports"

    # 2x2 dissociation (informational): each treatment cell vs BOTH controls.
    def _cell_vs_controls(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def _beats_both(r1: Dict[str, Any]) -> bool:
            rp = proposer_by_seed.get(r1["seed"])
            rn = noise_by_seed.get(r1["seed"])
            if rp is None or rn is None:
                return False
            e1 = float(r1.get(SENT, 0.0))
            return e1 > float(rp.get(SENT, 0.0)) and e1 > float(rn.get(SENT, 0.0))
        return {
            "selected_entropy_mean": round(_mean_key(rows, SENT), 6),
            "seeds_strict_above_both_controls": int(_n_seeds(rows, _beats_both)),
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
            "grading_engaged": {
                "k_varies_seeds": int(k_varies_seeds),
                "t_eff_varies_seeds": int(t_varies_seeds),
                "gap_spread_seeds": int(gap_spread_seeds),
                "gap_min_populated_bins": GAP_MIN_POPULATED_BINS,
                "grading_engaged_ok": grading_engaged,
            },
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "readiness_ok": readiness_ok,
        },
        "c_primary": {
            "a1b1_seeds_strict_above_both_controls": int(primary_seeds_ok),
            "a1b1_selected_entropy_mean": round(a1b1_sel_mean, 6),
            "selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
            "primary_floor_ok": primary_floor_ok,
            "c_primary_pass": primary_pass,
        },
        "c_fgap": {
            "a1b1_gap_slope": (round(a1b1_slope, 6) if a1b1_slope is not None else None),
            "a1b1_gap_bins_used": int(a1b1_nbins),
            "a1b1_per_bin_entropy": a1b1_bin_entropy,
            "a0b0_gap_slope": (round(a0b0_slope, 6) if a0b0_slope is not None else None),
            "a0b0_gap_bins_used": int(a0b0_nbins),
            "a0b0_per_bin_entropy": a0b0_bin_entropy,
            "gap_slope_delta_required": GAP_SLOPE_DELTA,
            "fgap_computable": fgap_computable,
            "fgap_slope_negative": fgap_negative,
            "fgap_more_concentrated_than_fixed_k": fgap_more_concentrated,
            "c_fgap_pass": fgap_pass,
            "note": (
                "Pre-registered LOAD-BEARING falsifier: the committed-entropy lift must "
                "correlate with the per-tick top-F gap. ARM_A1B1 per-gap-bin entropy slope "
                "must be NEGATIVE (lift concentrated at near-ties) AND more negative than "
                "the fixed-k/hard-argmin ARM_A0B0 baseline by >= gap_slope_delta. A uniform "
                "lift = a bigger fixed shortlist / hotter flat softmax, NOT conflict-grading."
            ),
        },
        "negative_control": {
            "matched_noise_lift_seeds_over_proposer": int(noise_lift_seeds),
            "negative_control_does_not_lift": negative_control_does_not_lift,
            "negative_control_unexpectedly_lifted": bool(not negative_control_does_not_lift),
            "note": (
                "ARM_MATCHED_NOISE (proposer @ flat T=2.5) MUST NOT lift committed entropy "
                "over ARM_PROPOSER_CTRL (collapsed channel). Informational sanity; does NOT "
                "gate the verdict."
            ),
        },
        "two_by_two_dissociation": {
            "ARM_A0B0_both_off": _cell_vs_controls(a0b0),
            "ARM_A1B0_factor_a_only": _cell_vs_controls(a1b0),
            "ARM_A0B1_factor_b_only": _cell_vs_controls(a0b1),
            "ARM_A1B1_both_on": _cell_vs_controls(a1b1),
            "note": (
                "Informational: which factor carries the lift. The PRIMARY gates on the "
                "both-levers cell ARM_A1B1; these per-cell readouts dissociate Factor A "
                "(conflict-graded width) vs Factor B (gap-scaled commit-T) vs both."
            ),
        },
        "selected_action_entropy_per_arm_mean": {
            PROPOSER_CTRL_ARM: round(_mean_key(proposer, SENT), 6),
            MATCHED_NOISE_ARM: round(_mean_key(noise, SENT), 6),
            BASELINE_ARM: round(_mean_key(a0b0, SENT), 6),
            "ARM_A1B0": round(_mean_key(a1b0, SENT), 6),
            "ARM_A0B1": round(_mean_key(a0b1, SENT), 6),
            PRIMARY_ARM: round(a1b1_sel_mean, 6),
        },
        "route_range_per_arm_mean": {
            PROPOSER_CTRL_ARM: round(_mean_key(proposer, RDIST), 6),
            MATCHED_NOISE_ARM: round(_mean_key(noise, RDIST), 6),
            BASELINE_ARM: round(_mean_key(a0b0, RDIST), 6),
            "ARM_A1B0": round(_mean_key(a1b0, RDIST), 6),
            "ARM_A0B1": round(_mean_key(a0b1, RDIST), 6),
            PRIMARY_ARM: round(a1b1_route_mean, 6),
        },
        "label": label,
        "evidence_direction": evidence_direction,
        "overall_pass": overall_pass,
        "preconditions": [
            {
                "name": "a1b1_modulatory_channel_route_range_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_A1B1 IN-ARM RAW cross-candidate RANGE of the modulatory bias "
                    "routed into the E3 selection authority (modulatory_channel_route_range, "
                    "read LIVE at the select tick) clears the floor: the divergent channel "
                    "REACHES the bias the conflict-graded shortlist arbitrates. SAME range "
                    "statistic the route-range substrate gates on. Below floor => routing "
                    "not wired / e2 under-trained => substrate_not_ready_requeue."
                ),
                "control": (
                    "ARM_A1B1: candidate_summary_source=e2_world_forward (genuinely "
                    "action-divergent), route-range routing + top-k shortlist ON"
                ),
                "measured": round(a1b1_route_mean, 6),
                "threshold": ROUTE_RANGE_FLOOR,
                "met": route_ready,
            },
            {
                "name": "a1b1_e2_world_forward_prediction_spread_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_A1B1 e2.world_forward(z0, a_i) per-candidate prediction spread "
                    "(cand_world_pairwise_dist) clears the floor -- SD-056 trained the "
                    "action-conditional divergence the eligible set needs. RANGE statistic. "
                    "Below floor => SD-056 under-trained => substrate_not_ready_requeue."
                ),
                "control": "ARM_A1B1: agent.e2.cand_world_pairwise_dist on SP-CEM candidates",
                "measured": round(a1b1_pdist_mean, 6),
                "threshold": C1_PAIRWISE_DIST_FLOOR,
                "met": c1_pass,
            },
            {
                "name": "a1b1_grading_quantities_and_gap_distribution_non_vacuous",
                "kind": "readiness",
                "description": (
                    "ARM_A1B1 conflict-graded k_effective varies across P1 ticks (spread "
                    ">= 1), the gap-scaled commit T_eff varies (spread > eps), AND the "
                    "per-tick F-gap spans >= gap_min_populated_bins gap bins. A flat k / "
                    "flat T_eff / single-gap distribution (monostrategy-pinned gap) means "
                    "the grading could not act -> the test is vacuous => "
                    "substrate_not_ready_requeue, never a verdict label. Count statistic "
                    "(seeds meeting all three)."
                ),
                "control": (
                    "ARM_A1B1: both MECH-439 levers ON; k_effective_spread / t_eff_spread / "
                    "n_populated_gap_bins read live"
                ),
                "measured": int(min(k_varies_seeds, t_varies_seeds, gap_spread_seeds)),
                "threshold": MIN_SEEDS_FOR_PASS,
                "met": grading_engaged,
            },
        ],
        "criteria": [
            {"name": "C1_a1b1_e2_world_forward_divergent", "load_bearing": True, "passed": c1_pass},
            {"name": "C_PRIMARY_a1b1_selected_entropy_strict_above_both_controls",
             "load_bearing": True, "passed": primary_pass},
            {"name": "C_FGAP_committed_entropy_lift_correlates_with_F_gap",
             "load_bearing": True, "passed": fgap_pass},
        ],
        "criteria_non_degenerate": {
            "C1": non_degenerate,
            "C_PRIMARY": non_degenerate,
            "C_FGAP": bool(fgap_computable),
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
                            "factor_a", "factor_b",
                        )
                    },
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "conflict_grade_config": {
                        "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
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
            "MECH-439 F-dominance conflict-grade 2-factor (2x2) discriminating falsifier -- "
            "MECH-439's FIRST falsifier. The conversion ceiling's live root is F-dominance "
            "(F = 88-89% of E3 committed-selection variance, V3-EXQ-571). Two no-op-default "
            "levers (Factor A conflict-graded shortlist width + Factor B gap-scaled commit-T) "
            "are TWO RENDERINGS OF ONE PRINCIPLE (the BG hyperdirect conflict-grade: grade "
            "the committed decision by the normalized top-F gap), tested as a 2x2 grid + two "
            "controls (collapsed-proposer + matched-noise) on the GAP-A-ready foraging "
            "substrate (SD-056-trained e2.world_forward + ARC-065 GAP-A "
            "candidate_summary_source=e2_world_forward = the divergent eligible set). PASS "
            "(label=conflict_grade_converts_committed_diversity) = ARM_A1B1 (both levers) "
            "committed-action class entropy strictly above BOTH the collapsed-proposer "
            "baseline AND the matched-noise control on >=2/3 seeds (mean > floor), with the "
            "IN-ARM route-range + e2-divergence + grading-engaged readiness gates met AND the "
            "LOAD-BEARING pre-registered F-gap falsifier (the committed-entropy lift "
            "correlates with the per-tick top-F gap: ARM_A1B1 per-gap-bin entropy slope "
            "negative AND more negative than the fixed-k ARM_A0B0 baseline). Route/e2-div "
            "below floor OR k/T_eff/gap not spread self-routes substrate_not_ready_requeue "
            "(non_contributory, NEVER a weakens). Readiness met but no strict lift => "
            "conversion_ceiling_persists_despite_conflict_grade (OFF-RAMP to the V4 "
            "directions in the synthesis doc, NOT a falsification). Readiness + lift but "
            "uniform (non-gap-correlated) lift => uniform_lift_grading_non_load_bearing "
            "(the lift is a bigger fixed shortlist / hotter flat softmax, NOT conflict-"
            "grading -> route to rank_preserving_F_to_eligibility_demotion). NO weakens "
            "path. claim_ids=[MECH-439] only; ARC-065/MECH-341/ARC-062/MECH-309/MECH-294 "
            "untouched."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "conflict_grade_converts_committed_diversity": "PASS -> MECH-439 first falsifier supports; conflict-grade moves committed behaviour and the lift is gap-correlated; toward supports",
                "substrate_not_ready_requeue": "routing/SD-056 under-trained OR monostrategy-pinned gap (k/T_eff/gap not spread); re-queue at higher P0; do NOT weaken MECH-439",
                "conversion_ceiling_persists_despite_conflict_grade": "OFF-RAMP -> the V4 directions in conversion_ceiling_phase0_synthesis (divisive normalization / output-null subspace / cross-tick QD archive); NOT a falsification of MECH-439",
                "uniform_lift_grading_non_load_bearing": "the lift is real but NOT gap-concentrated -> a bigger fixed shortlist / hotter flat softmax; route to the rank_preserving_F_to_eligibility_demotion V3 fallback",
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
                {k: a[k] for k in ("arm_id", "label", "candidate_summary_source", "temperature", "factor_a", "factor_b")}
                for a in ARMS
            ],
            "sd056_weight": SD056_WEIGHT,
            "matched_entropy_temperature": MATCHED_ENTROPY_TEMPERATURE,
            "conflict_grade_config": {
                "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
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
                "gap_slope_delta": GAP_SLOPE_DELTA,
                "fgap_nbins": FGAP_NBINS,
                "fgap_min_bin_ticks": FGAP_MIN_BIN_TICKS,
                "gap_min_populated_bins": GAP_MIN_POPULATED_BINS,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_route_range_ready": summary["readiness"]["route_ready"],
            "readiness_e2_divergent_ready": summary["readiness"]["c1_pass"],
            "readiness_grading_engaged": summary["readiness"]["grading_engaged"]["grading_engaged_ok"],
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "negative_control_does_not_lift": summary["negative_control"]["negative_control_does_not_lift"],
            "C_PRIMARY_a1b1_strict_above_both_controls": summary["c_primary"]["c_primary_pass"],
            "C_FGAP_lift_correlates_with_F_gap": summary["c_fgap"]["c_fgap_pass"],
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
        description="V3-EXQ-689 MECH-439 F-dominance conflict-grade 2-factor (2x2) falsifier"
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
