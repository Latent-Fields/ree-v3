#!/opt/local/bin/python3
"""
V3-EXQ-690 -- Q-054: ARC-062 minimum trajectory-class diversity floor sweep.

Answers open-question Q-054: what is the minimum Rung-1 first_action_entropy
floor for the ARC-062 context discriminator to learn a reliable reef-vs-foraging
cut, and does the SP-CEM 0.497-nat lift sit ABOVE (sufficient) or BELOW
(insufficient) that floor?

Design source (build exactly this):
  REE_assembly/evidence/planning/q054_q055_q056_buildability_triage_2026-06-19.md
  ("Q-054" section).

Primary base template: v3_exq_654g (ARC-062 gated-policy + 3-stream context
discriminator + 569i top-k shortlist conversion stack). DV / top-k conversion
arming patterned on v3_exq_569i.

WHY THIS IS NOW BUILDABLE (the hold is stale)
---------------------------------------------
The 2026-06-16 deferral was gated on the GAP-A conversion ceiling: committed
behaviour did not move with the channel range, so discriminator accuracy could
not be read as a function of REALISED first_action_entropy. That ceiling was
LIFTED 2026-06-17 (V3-EXQ-569i PASS -- the TOP-K shortlist conversion clears the
committed-action conversion frontier; ARC-065 substrate_ceiling -> standard). So
realised first-action entropy can now be made to MOVE committed behaviour via the
top-k shortlist, which is the exact precondition the deferral said was missing.
The top-k conversion stack is armed IDENTICALLY on every arm (the matched-stack
constant) so the only thing that varies across arms is the upstream noise that
sets realised first_action_entropy.

THE SWEEP (the only varied variable across arms)
------------------------------------------------
A graded entropy sweep via the MECH-313 noise floor, which lifts the E3
SELECTION softmax temperature (the committed / uncommitted multinomial in
select_action) -- so it directly raises REALISED committed first-action entropy.
We sweep noise_floor_min_temperature (a hard floor on the effective selection
temperature) across:
  ARM_T_LOW    min_temperature 0.30  -> realised first_action_entropy BELOW ~0.497
  ARM_T_SPCEM  min_temperature 1.00  -> ~0.497 nats (the SP-CEM baseline regime)
  ARM_T_MID    min_temperature 1.75  -> ~0.75 nats
  ARM_T_HIGH   min_temperature 2.75  -> ~1.0+ nats
(noise_floor_alpha is held at 0.0 on every arm so min_temperature is the clean
single lever; the realised per-arm entropy is MEASURED, not assumed.)

The MEASURED upstream first_action_entropy per arm is the x-axis; the DV is read
against the measured value, not the nominal temperature.

DEPENDENT VARIABLES (both, per arm, vs measured first_action_entropy)
---------------------------------------------------------------------
(1) ARC-062 context-discriminator accuracy (reef vs forage): the gated-policy
    discriminator emits a scalar gating_weight w in [0,1] each tick. We bin ticks
    by whether the agent is in the reef-half vs the forage-half of the SD-054
    bipartite layout (env.agent_x vs midline+radius on the horizontal axis), and
    score the discriminator's ability to separate the two contexts via balanced
    accuracy of the best-threshold cut on w (reef-vs-forage), reported alongside
    a chance baseline (0.5).
(2) TV distance TV(P(a|s_reef), P(a|s_forage)): the total-variation distance
    between the committed-first-action class distributions in reef-half ticks vs
    forage-half ticks. A discriminator that has learned the cut produces
    context-conditional behaviour -> high TV; a discriminator at chance produces
    context-uniform behaviour -> TV ~ 0.

THE FLOOR (the Q-054 answer)
----------------------------
Locate the entropy value below which discriminator accuracy is at chance and
above which it rises monotonically. Report whether 0.497 nats sits above
(SP-CEM sufficient) or below (insufficient) the bracketed floor.

NON-VACUITY / READINESS SELF-ROUTE GATES (never a verdict label if unmet)
-------------------------------------------------------------------------
R1  realised first_action_entropy actually VARIES across arms
    (max-arm - min-arm measured entropy > FLOOR_R1_ENTROPY_RANGE). Else the
    sweep is vacuous -> substrate_not_ready_requeue.
R2  top-k conversion non-vacuous: modulatory_shortlist_active_ticks > 0 AND
    shortlist mode == 'top_k' AND mean shortlist size ~= k (within tolerance) on
    a majority of arms. Else the conversion is not engaged -> requeue.
R3  discriminator non-degenerate: the gating_weight z-context VARIANCE / its
    cross-tick spread is not pinned (w_range > FLOOR_R3_W_RANGE) on a majority of
    arms; and both reef-half and forage-half ticks were actually sampled.
    Else the discriminator readout is degenerate -> requeue.

If discriminator accuracy is FLAT-AT-CHANCE across ALL arms WITH R1-R3 MET, this
is NOT a floor result: it is a residual conversion ceiling (F-dominance,
MECH-439) and self-routes to /implement-substrate, NOT a Q-054 floor verdict and
NOT an ARC-062 falsification.

claim_ids = ["ARC-062"] (tests the discriminator floor only).
experiment_purpose = "evidence".

See:
  REE_assembly/evidence/planning/q054_q055_q056_buildability_triage_2026-06-19.md
  experiments/v3_exq_654g_arc062_gapb_rule_apprehension_behavioural_falsifier.py
  experiments/v3_exq_569i_gapa_conversion_topk_shortlist_falsifier.py
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

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_690_q054_arc062_diversity_floor_sweep"
QUEUE_ID = "V3-EXQ-690"
CLAIM_IDS: List[str] = ["ARC-062"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds (module constants; NOT derived post-hoc)
# ---------------------------------------------------------------------------
# R1: realised first_action_entropy must vary across arms by at least this range.
FLOOR_R1_ENTROPY_RANGE = 0.20
# R2: top-k conversion engaged -- shortlist mean size within this tolerance of k.
SHORTLIST_SIZE_TOL = 0.75
# R3: discriminator non-degenerate -- gating_weight cross-tick range floor.
FLOOR_R3_W_RANGE = 1e-3
# Discriminator accuracy "above chance" margin (balanced accuracy).
ACC_ABOVE_CHANCE_MARGIN = 0.05   # bal-acc > 0.5 + margin counts as above chance
CHANCE_ACC = 0.5
# Floor-located criterion: accuracy must RISE monotonically once above the floor.
# A floor is BRACKETED if >= 1 arm is at chance AND >= 1 higher-entropy arm is
# above chance.
# Minimum reef-half AND forage-half ticks for a per-arm DV to be scored.
MIN_CONTEXT_TICKS = 10
# SP-CEM reference entropy (the value whose sufficiency Q-054 reports).
SPCEM_REFERENCE_ENTROPY_NATS = 0.497

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60          # SD-056 e2 contrastive warmup (569i-proven budget)
P1_TRAIN_EPISODES = 90           # frozen-encoder gated-policy + bias-head REINFORCE
P2_MEASUREMENT_EPISODES = 60     # all frozen; DV measurement window
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 30

# ---------------------------------------------------------------------------
# 569i-validated TOP-K SHORTLIST conversion stack (matched-stack CONSTANT on
# EVERY arm -- so realised first-action entropy can actually move committed
# behaviour; the precondition lifted 2026-06-17).
# ---------------------------------------------------------------------------
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

# SD-056 online e2 training (mirror 569i harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# P1 gated-policy / bias-head REINFORCE training (mirror 598b / 654g).
LR_GATED_POLICY = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

# Selection temperature baseline passed into select_action; the noise floor
# min_temperature lifts the EFFECTIVE selection temperature above this when
# higher (the sweep lever). Held at 1.0 on every arm; the floor does the work.
BASE_SELECT_TEMPERATURE = 1.0


# SD-054 reef-bipartite hazard layout (matches 654g / 569i).
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
# The graded entropy SWEEP -- the only varied variable is the noise-floor
# min_temperature (lifts the E3 selection softmax temperature => realised
# committed first-action entropy). 4 levels spanning below/at/above 0.497 nats.
# ---------------------------------------------------------------------------
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_T_LOW",
        "label": "noise_floor_min_temp_0p30_below_spcem_entropy",
        "noise_floor_min_temperature": 0.30,
    },
    {
        "arm_id": "ARM_T_SPCEM",
        "label": "noise_floor_min_temp_1p00_spcem_baseline_~0p497nats",
        "noise_floor_min_temperature": 1.00,
    },
    {
        "arm_id": "ARM_T_MID",
        "label": "noise_floor_min_temp_1p75_~0p75nats",
        "noise_floor_min_temperature": 1.75,
    },
    {
        "arm_id": "ARM_T_HIGH",
        "label": "noise_floor_min_temp_2p75_~1p0plus_nats",
        "noise_floor_min_temperature": 2.75,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Matched-stack ARC-062 gated-policy + 3-stream discriminator + MECH-341
    preserver + the 569i top-k shortlist conversion, ON identically on every arm.
    The ONLY varied flag is noise_floor_min_temperature (the entropy sweep)."""
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
        # --- ARC-065 SP-CEM (Layer A): action-divergent candidate pool ---
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # --- GAP-A shared per-candidate signal from e2.world_forward (649) ---
        candidate_summary_source="e2_world_forward",
        # --- modulatory-bias-selection-authority (643a) + 569i TOP-K shortlist ---
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
        # --- MECH-341 preserver (stratified across-class) ON ---
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=2.0,
        e3_diversity_stratified_within_class_temperature=None,
        # --- ARC-062 gated-policy heads + 3-stream context discriminator ON ---
        use_gated_policy=True,
        # --- MECH-313 noise floor: the SWEEP lever (min_temperature per arm) ---
        use_noise_floor=True,
        noise_floor_alpha=0.0,
        noise_floor_min_temperature=float(arm["noise_floor_min_temperature"]),
        # --- V_s minimal stack ---
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=0.5,
        vs_gate_e1_threshold=0.4,
        # --- SD-056 e2 action-conditional divergence (trained online in P0) ---
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=True,
        e2_action_contrastive_horizon=5,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror 569i)
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
# Helpers
# ---------------------------------------------------------------------------

def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _first_action_class(traj) -> int:
    return int(traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item())


def _first_actions_K(candidates) -> torch.Tensor:
    rows = [t.actions[:, 0, :].detach().reshape(-1) for t in candidates]
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


def _gated_policy_params(agent: REEAgent):
    """Trainable params of the gated policy (heads/base+delta + discriminator)."""
    gp = agent.gated_policy
    if gp is None:
        return []
    return [p for p in gp.parameters() if p.requires_grad]


def _is_reef_half(env: CausalGridWorldV2) -> bool:
    """Is the agent currently on the reef side of the SD-054 bipartite partition?

    Horizontal axis: reef occupies rows > midline + radius; the row index is
    env.agent_x. Vertical axis: reef occupies cols > midline + radius (agent_y).
    """
    midline = env.size // 2
    radius = env.reef_bipartite_agent_band_radius
    if env.reef_bipartite_axis == "horizontal":
        return int(env.agent_x) > midline + radius
    return int(env.agent_y) > midline + radius


def _tv_distance(counts_a: Dict[int, int], counts_b: Dict[int, int]) -> float:
    """Total-variation distance between two categorical action distributions."""
    na = sum(counts_a.values())
    nb = sum(counts_b.values())
    if na <= 0 or nb <= 0:
        return 0.0
    keys = set(counts_a) | set(counts_b)
    tv = 0.0
    for k in keys:
        pa = counts_a.get(k, 0) / na
        pb = counts_b.get(k, 0) / nb
        tv += abs(pa - pb)
    return 0.5 * tv


def _balanced_threshold_accuracy(
    w_reef: List[float], w_forage: List[float]
) -> float:
    """Best-threshold balanced accuracy separating reef vs forage gating_weight.

    Sweeps every candidate threshold (midpoints of the sorted pooled values) and
    both polarities; returns the maximum balanced accuracy. Chance == 0.5.
    """
    if not w_reef or not w_forage:
        return CHANCE_ACC
    pooled = sorted(set(w_reef + w_forage))
    if len(pooled) < 2:
        return CHANCE_ACC
    thresholds = [
        0.5 * (pooled[i] + pooled[i + 1]) for i in range(len(pooled) - 1)
    ]
    best = CHANCE_ACC
    for thr in thresholds:
        # polarity 1: reef predicted as w >= thr
        tpr = sum(1 for v in w_reef if v >= thr) / len(w_reef)
        tnr = sum(1 for v in w_forage if v < thr) / len(w_forage)
        best = max(best, 0.5 * (tpr + tnr))
        # polarity 2: reef predicted as w < thr
        tpr2 = sum(1 for v in w_reef if v < thr) / len(w_reef)
        tnr2 = sum(1 for v in w_forage if v >= thr) / len(w_forage)
        best = max(best, 0.5 * (tpr2 + tnr2))
    return float(best)


# ---------------------------------------------------------------------------
# P1 gated-policy REINFORCE training (mirror 598b _lpfc_reinforce_loss)
# ---------------------------------------------------------------------------

def _gated_policy_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]],
    baseline: float,
    device,
) -> torch.Tensor:
    """REINFORCE on the gated-policy heads+discriminator over stored snaps.

    Each stored snap is (z_world, z_self, z_harm_a, sel_idx, ep_return). Re-runs
    gated_policy.forward (differentiable w.r.t. its params) on the stored latents
    + candidate summaries, REINFORCE-weighted by the episode-return advantage.
    Mirrors v3_exq_598b/654g._lpfc_reinforce_loss pattern but for the gated
    policy bias.
    """
    gp = agent.gated_policy
    if gp is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    n = len(outcome_buf)
    idxs = np.random.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False)
    terms: List[torch.Tensor] = []
    for i in idxs:
        zw, zs, za, cand_feats, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        out = gp.forward(
            z_world=zw.to(device),
            z_self=zs.to(device),
            z_harm_a=za.to(device) if za is not None else None,
            candidate_features=cand_feats.to(device),
            simulation_mode=False,
        )
        bias = out.gated_score_bias
        if bias.shape[0] < 1:
            continue
        log_p = torch.nn.functional.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    gp_params = _gated_policy_params(agent)
    gp_opt = (
        torch.optim.Adam(gp_params, lr=LR_GATED_POLICY) if gp_params else None
    )
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes + p2_episodes
    p1_start = p0_episodes
    p2_start = p0_episodes + p1_episodes
    error_note: Optional[str] = None

    # P1 REINFORCE state.
    reinforce_baseline = 0.0
    outcome_buf: List[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float]
    ] = []
    n_p1_gp_updates = 0
    n_p0_contrastive_steps = 0

    # P2 measurement accumulators.
    committed_first_action_counts: Dict[int, int] = {}
    reef_action_counts: Dict[int, int] = {}
    forage_action_counts: Dict[int, int] = {}
    w_reef: List[float] = []
    w_forage: List[float] = []
    w_all: List[float] = []
    pairwise_dists: List[float] = []
    route_ranges: List[float] = []
    shortlist_sizes: List[float] = []
    shortlist_active_ticks = 0
    shortlist_mode_seen: Optional[str] = None
    n_p2_ticks = 0
    n_reef_ticks = 0
    n_forage_ticks = 0

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
        ep_buf: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
        ] = []

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

            # P1 snapshot for the gated-policy REINFORCE update: capture the
            # per-candidate summaries (the same e2_world_forward source the
            # gated policy consumes) BEFORE select_action.
            p1_snap: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]] = None
            if is_p1 and candidates and len(candidates) >= 2:
                cs = agent._candidate_world_summaries(candidates)
                if cs is None:
                    rows = []
                    for c in candidates:
                        if c.world_states is not None:
                            rows.append(c.get_world_state_sequence()[0, 0, :].detach())
                        else:
                            rows.append(latent.z_world[0].detach())
                    cs = torch.stack(rows, dim=0)
                if cs is not None and torch.isfinite(cs).all():
                    za = (
                        latent.z_harm_a.detach().clone()
                        if latent.z_harm_a is not None else None
                    )
                    p1_snap = (
                        latent.z_world.detach().clone(),
                        latent.z_self.detach().clone(),
                        za,
                        cs.detach().clone(),
                    )

            # P2 substrate-readiness reads (e2 divergence; informational).
            if is_p2 and candidates and len(candidates) >= 2:
                actions_K = _first_actions_K(candidates).to(agent.device)
                z0 = latent.z_world.detach()
                with torch.no_grad():
                    pdist = float(
                        agent.e2.cand_world_pairwise_dist(z0, actions_K).item()
                    )
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
                candidates, ticks, temperature=BASE_SELECT_TEMPERATURE
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

            committed_class = int(action[0].argmax().item())

            # P1: record (latents, candidate_features, selected-candidate-index).
            if is_p1 and p1_snap is not None:
                sel = 0
                for ci, c in enumerate(candidates):
                    if (
                        getattr(c, "actions", None) is not None
                        and c.actions.shape[1] >= 1
                        and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                        == committed_class
                    ):
                        sel = min(ci, p1_snap[3].shape[0] - 1)
                        break
                ep_buf.append((p1_snap[0], p1_snap[1], p1_snap[2], p1_snap[3], sel))

            # P2 DV measurement.
            if is_p2:
                n_p2_ticks += 1
                committed_first_action_counts[committed_class] = (
                    committed_first_action_counts.get(committed_class, 0) + 1
                )
                # ARC-062 discriminator output this tick.
                w = None
                if agent.gated_policy is not None:
                    w = float(getattr(agent.gated_policy, "_last_gating_weight", 0.5))
                # reef-vs-forage context label from the env geometry.
                in_reef = _is_reef_half(env)
                if w is not None and math.isfinite(w):
                    w_all.append(w)
                if in_reef:
                    n_reef_ticks += 1
                    reef_action_counts[committed_class] = (
                        reef_action_counts.get(committed_class, 0) + 1
                    )
                    if w is not None and math.isfinite(w):
                        w_reef.append(w)
                else:
                    n_forage_ticks += 1
                    forage_action_counts[committed_class] = (
                        forage_action_counts.get(committed_class, 0) + 1
                    )
                    if w is not None and math.isfinite(w):
                        w_forage.append(w)
                # top-k conversion non-vacuity (R2) + route-range (R3 support).
                diag = agent.e3.last_score_diagnostics
                rr = float(diag.get("modulatory_channel_route_range", 0.0))
                if math.isfinite(rr):
                    route_ranges.append(rr)
                if bool(diag.get("modulatory_shortlist_active", False)):
                    shortlist_active_ticks += 1
                    sl = float(diag.get("modulatory_shortlist_size", 0))
                    if math.isfinite(sl):
                        shortlist_sizes.append(sl)
                    if shortlist_mode_seen is None:
                        shortlist_mode_seen = str(
                            diag.get("modulatory_shortlist_mode", "")
                        )

            if (
                torch.isfinite(latent.z_world).all()
                and torch.isfinite(action).all()
            ):
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 ONLY (frozen in P1/P2 for stable measurement).
            if (not is_p1) and (not is_p2) and (
                tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0
            ):
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val):
                    n_p0_contrastive_steps += 1

            _, harm_signal, done, info, next_obs_dict = env.step(action)
            if is_p1:
                ep_reward += -float(harm_signal)  # lower harm = better outcome
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            obs_dict = next_obs_dict
            tick_in_ep += 1
            if done:
                break

        # P1 end-of-episode: REINFORCE update on the gated-policy params.
        if is_p1 and gp_opt is not None:
            reinforce_baseline = (
                EMA_DECAY * reinforce_baseline + (1.0 - EMA_DECAY) * ep_reward
            )
            for (zw, zs, za, cf, sel) in ep_buf:
                outcome_buf.append((zw, zs, za, cf, sel, ep_reward))
            if len(outcome_buf) > OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
            gp_loss = _gated_policy_reinforce_loss(
                agent, outcome_buf, reinforce_baseline, agent.device
            )
            if gp_loss.requires_grad:
                gp_opt.zero_grad()
                gp_loss.backward()
                torch.nn.utils.clip_grad_norm_(gp_params, MAX_GRAD_NORM)
                gp_opt.step()
                n_p1_gp_updates += 1

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    # ----- Per-seed-arm aggregation (over P2) -----
    realised_first_action_entropy = _entropy_from_counts(
        committed_first_action_counts
    )
    # DV (1): discriminator reef-vs-forage balanced accuracy.
    context_scorable = bool(
        n_reef_ticks >= MIN_CONTEXT_TICKS and n_forage_ticks >= MIN_CONTEXT_TICKS
    )
    disc_balanced_accuracy = (
        _balanced_threshold_accuracy(w_reef, w_forage)
        if context_scorable else CHANCE_ACC
    )
    # DV (2): TV(P(a|s_reef), P(a|s_forage)).
    tv_reef_forage = (
        _tv_distance(reef_action_counts, forage_action_counts)
        if context_scorable else 0.0
    )
    # R3 support: gating_weight cross-tick range.
    w_range = (max(w_all) - min(w_all)) if len(w_all) >= 2 else 0.0

    shortlist_size_mean = _mean(shortlist_sizes)
    shortlist_mode = shortlist_mode_seen or ""
    r2_arm_ok = bool(
        shortlist_active_ticks > 0
        and shortlist_mode == MODULATORY_SHORTLIST_MODE
        and abs(shortlist_size_mean - float(MODULATORY_SHORTLIST_K))
        <= SHORTLIST_SIZE_TOL
    )
    r3_arm_ok = bool(w_range > FLOOR_R3_W_RANGE and context_scorable)

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "noise_floor_min_temperature": float(arm["noise_floor_min_temperature"]),
        "error_note": error_note,
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_gp_updates": int(n_p1_gp_updates),
        "n_p2_ticks": int(n_p2_ticks),
        "n_reef_ticks": int(n_reef_ticks),
        "n_forage_ticks": int(n_forage_ticks),
        "context_scorable": context_scorable,
        # x-axis: MEASURED realised upstream first-action entropy.
        "realised_first_action_entropy_nats": round(realised_first_action_entropy, 6),
        "n_unique_committed_first_action_classes": int(
            len(committed_first_action_counts)
        ),
        # DV (1): ARC-062 discriminator reef-vs-forage accuracy.
        "disc_reef_forage_balanced_accuracy": round(disc_balanced_accuracy, 6),
        "disc_above_chance": bool(
            disc_balanced_accuracy > CHANCE_ACC + ACC_ABOVE_CHANCE_MARGIN
        ),
        # DV (2): TV distance of context-conditional action distributions.
        "tv_reef_forage": round(tv_reef_forage, 6),
        # R3 support: discriminator non-degeneracy.
        "gating_weight_range": round(w_range, 8),
        "gating_weight_mean": round(_mean(w_all), 6),
        # R2 support: top-k conversion engagement.
        "modulatory_shortlist_active_ticks": int(shortlist_active_ticks),
        "modulatory_shortlist_size_mean": round(shortlist_size_mean, 6),
        "modulatory_shortlist_mode": shortlist_mode,
        "r2_arm_topk_engaged": r2_arm_ok,
        "r3_arm_discriminator_nondegenerate": r3_arm_ok,
        # informational substrate readouts.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        "modulatory_channel_route_range_mean": round(_mean(route_ranges), 6),
        "reef_action_counts": {str(k): int(v) for k, v in sorted(reef_action_counts.items())},
        "forage_action_counts": {str(k): int(v) for k, v in sorted(forage_action_counts.items())},
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id and r.get("error_note") is None]


def _mean(xs: List[float], default: float = 0.0) -> float:
    return float(sum(xs) / len(xs)) if xs else default


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    arm_ids = [a["arm_id"] for a in ARMS]
    # Per-arm means (across seeds).
    per_arm: Dict[str, Dict[str, float]] = {}
    for aid in arm_ids:
        rows = _arm_rows(arm_results, aid)
        per_arm[aid] = {
            "realised_first_action_entropy_nats": round(
                _mean([r["realised_first_action_entropy_nats"] for r in rows]), 6
            ),
            "disc_reef_forage_balanced_accuracy": round(
                _mean([r["disc_reef_forage_balanced_accuracy"] for r in rows]), 6
            ),
            "tv_reef_forage": round(
                _mean([r["tv_reef_forage"] for r in rows]), 6
            ),
            "gating_weight_range": round(
                _mean([r["gating_weight_range"] for r in rows]), 8
            ),
            "shortlist_size_mean": round(
                _mean([r["modulatory_shortlist_size_mean"] for r in rows]), 6
            ),
            "n_seeds_above_chance": int(
                sum(1 for r in rows if r["disc_above_chance"])
            ),
            "n_seeds": int(len(rows)),
            "n_seeds_r2_ok": int(sum(1 for r in rows if r["r2_arm_topk_engaged"])),
            "n_seeds_r3_ok": int(
                sum(1 for r in rows if r["r3_arm_discriminator_nondegenerate"])
            ),
        }

    # ---- R1: realised entropy varies across arms ----
    arm_entropies = [
        per_arm[aid]["realised_first_action_entropy_nats"] for aid in arm_ids
    ]
    entropy_range = (max(arm_entropies) - min(arm_entropies)) if arm_entropies else 0.0
    r1_met = bool(entropy_range > FLOOR_R1_ENTROPY_RANGE)

    # ---- R2: top-k conversion non-vacuous on a majority of arms ----
    n_arms_r2 = sum(1 for aid in arm_ids if per_arm[aid]["n_seeds_r2_ok"] >= 2)
    r2_met = bool(n_arms_r2 >= (len(arm_ids) // 2 + 1))

    # ---- R3: discriminator non-degenerate on a majority of arms ----
    n_arms_r3 = sum(1 for aid in arm_ids if per_arm[aid]["n_seeds_r3_ok"] >= 2)
    r3_met = bool(n_arms_r3 >= (len(arm_ids) // 2 + 1))

    readiness_ok = bool(r1_met and r2_met and r3_met)

    # ---- Sort arms by MEASURED entropy (the x-axis) ----
    arms_by_entropy = sorted(
        arm_ids,
        key=lambda aid: per_arm[aid]["realised_first_action_entropy_nats"],
    )
    acc_curve = [
        {
            "arm_id": aid,
            "realised_first_action_entropy_nats": per_arm[aid][
                "realised_first_action_entropy_nats"
            ],
            "disc_reef_forage_balanced_accuracy": per_arm[aid][
                "disc_reef_forage_balanced_accuracy"
            ],
            "tv_reef_forage": per_arm[aid]["tv_reef_forage"],
            "above_chance": bool(
                per_arm[aid]["disc_reef_forage_balanced_accuracy"]
                > CHANCE_ACC + ACC_ABOVE_CHANCE_MARGIN
            ),
        }
        for aid in arms_by_entropy
    ]

    # ---- Floor location: bracket the entropy at which accuracy crosses chance ----
    below = [c for c in acc_curve if not c["above_chance"]]
    above = [c for c in acc_curve if c["above_chance"]]
    floor_bracketed = bool(below and above)
    # Monotone-rise check: among arms sorted by entropy, accuracy is
    # non-decreasing once it first goes above chance.
    accs_sorted = [c["disc_reef_forage_balanced_accuracy"] for c in acc_curve]
    monotone_after_floor = True
    seen_above = False
    prev = -1.0
    for c in acc_curve:
        if c["above_chance"]:
            seen_above = True
        if seen_above:
            if c["disc_reef_forage_balanced_accuracy"] < prev - 1e-6:
                monotone_after_floor = False
            prev = c["disc_reef_forage_balanced_accuracy"]

    floor_low = max(
        [c["realised_first_action_entropy_nats"] for c in below], default=0.0
    ) if below else 0.0
    floor_high = min(
        [c["realised_first_action_entropy_nats"] for c in above], default=0.0
    ) if above else 0.0

    all_at_chance = bool(len(above) == 0)
    all_above_chance = bool(len(below) == 0 and len(above) == len(acc_curve))

    # ---- Verdict ----
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "non_contributory"
        spcem_sufficient: Optional[bool] = None
    elif all_at_chance:
        # Readiness met (entropy varied, conversion engaged, discriminator
        # non-degenerate) but accuracy flat at chance across ALL arms -> NOT a
        # floor result. Residual conversion ceiling (F-dominance, MECH-439).
        label = "residual_conversion_ceiling_route_implement_substrate"
        outcome = "FAIL"
        direction = "non_contributory"
        spcem_sufficient = None
    elif floor_bracketed:
        label = "diversity_floor_bracketed"
        outcome = "PASS"
        direction = "supports"
        # SP-CEM 0.497 sufficient iff it sits at-or-above the bracketed floor_low
        # (the highest at-chance entropy below the first above-chance arm).
        spcem_sufficient = bool(SPCEM_REFERENCE_ENTROPY_NATS >= floor_low)
    else:
        # All arms above chance -> floor is BELOW the lowest swept entropy; the
        # floor is sub-bracketed (SP-CEM clearly sufficient) but not pinned.
        label = "floor_below_lowest_swept_entropy_spcem_sufficient"
        outcome = "PASS"
        direction = "supports"
        spcem_sufficient = True

    return {
        "label": label,
        "outcome": outcome,
        "evidence_direction": direction,
        "readiness_ok": readiness_ok,
        "readiness": {
            "R1_realised_entropy_varies_across_arms": r1_met,
            "R1_entropy_range_measured": round(entropy_range, 6),
            "R1_entropy_range_floor": float(FLOOR_R1_ENTROPY_RANGE),
            "R2_topk_conversion_nonvacuous": r2_met,
            "R2_n_arms_topk_engaged": int(n_arms_r2),
            "R3_discriminator_nondegenerate": r3_met,
            "R3_n_arms_nondegenerate": int(n_arms_r3),
        },
        "floor": {
            "bracketed": floor_bracketed,
            "monotone_after_floor": bool(monotone_after_floor),
            "floor_low_entropy_nats": round(floor_low, 6),
            "floor_high_entropy_nats": round(floor_high, 6),
            "all_at_chance": all_at_chance,
            "all_above_chance": all_above_chance,
        },
        "spcem_reference_entropy_nats": float(SPCEM_REFERENCE_ENTROPY_NATS),
        "spcem_sufficient": spcem_sufficient,
        "accuracy_curve_sorted_by_entropy": acc_curve,
        "per_arm_means": per_arm,
        "non_degenerate": bool(readiness_ok),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
            f"(P0={p0_episodes} P1={p1_episodes} P2={p2_episodes} "
            f"steps={steps_per_episode} dry_run={dry_run})",
            flush=True,
        )
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)
            config_slice = {
                "arm_id": arm["arm_id"],
                "noise_floor_min_temperature": float(
                    arm["noise_floor_min_temperature"]
                ),
                "noise_floor_alpha": 0.0,
                "use_modulatory_shortlist_then_modulate": bool(
                    USE_MODULATORY_SHORTLIST_THEN_MODULATE
                ),
                "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
                "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
                "modulatory_authority_normalize_basis": str(
                    MODULATORY_AUTHORITY_NORMALIZE_BASIS
                ),
                "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
                "candidate_summary_source": "e2_world_forward",
                "use_gated_policy": True,
                "use_e3_score_diversity": True,
                "env_kwargs": dict(ENV_KWARGS),
                "sd056_weight": SD056_WEIGHT,
                "lr_gated_policy": LR_GATED_POLICY,
                "p0_episodes": int(p0_episodes),
                "p1_episodes": int(p1_episodes),
                "p2_episodes": int(p2_episodes),
                "steps_per_episode": int(steps_per_episode),
            }
            with arm_cell(
                s,
                config_slice=config_slice,
                script_path=script_path,
                extra_ineligible_reasons=[
                    "online_e2_training_stateful_per_cell",
                    "p1_gated_policy_reinforce_training_stateful_per_cell",
                ],
            ) as cell:
                row = _run_seed_arm(
                    arm, s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                )
                cell.stamp(row)
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    summary = _evaluate(arm_results)

    total_seeds = len(ARMS) * len(seeds)
    total_completed = sum(1 for r in arm_results if r["error_note"] is None)

    # Pre-registered interpretation block (preconditions + non-degeneracy +
    # load-bearing criteria) for the indexer / governance readout.
    interpretation = {
        "label": summary["label"],
        "preconditions": [
            {
                "name": "R1_realised_first_action_entropy_varies_across_arms",
                "kind": "readiness",
                "description": (
                    "Measured committed first-action entropy RANGE across the 4 "
                    "noise-floor arms exceeds the floor -- the sweep actually moved "
                    "realised entropy (else the floor sweep is vacuous). SAME "
                    "statistic (first_action_entropy) the floor verdict locates."
                ),
                "control": "noise_floor_min_temperature graded 0.30/1.00/1.75/2.75",
                "measured": float(summary["readiness"]["R1_entropy_range_measured"]),
                "threshold": float(FLOOR_R1_ENTROPY_RANGE),
                "met": bool(summary["readiness"]["R1_realised_entropy_varies_across_arms"]),
            },
            {
                "name": "R2_topk_shortlist_conversion_nonvacuous",
                "kind": "readiness",
                "description": (
                    "569i top-k shortlist engaged: shortlist active, mode==top_k, "
                    "mean shortlist size ~= k on a majority of arms. Without the "
                    "conversion engaged, realised entropy cannot reach committed "
                    "behaviour and discriminator accuracy cannot be read as a "
                    "function of entropy."
                ),
                "control": "use_modulatory_shortlist_then_modulate + mode=top_k + k=3",
                "measured": float(summary["readiness"]["R2_n_arms_topk_engaged"]),
                "threshold": float(len(ARMS) // 2 + 1),
                "met": bool(summary["readiness"]["R2_topk_conversion_nonvacuous"]),
            },
            {
                "name": "R3_discriminator_nondegenerate",
                "kind": "readiness",
                "description": (
                    "ARC-062 gating_weight carries cross-tick RANGE (not pinned) "
                    "AND both reef-half and forage-half contexts were sampled, on a "
                    "majority of arms. A pinned discriminator gives a degenerate "
                    "accuracy readout. RANGE statistic."
                ),
                "control": "gated_policy gating_weight cross-tick range + context tick counts",
                "measured": float(summary["readiness"]["R3_n_arms_nondegenerate"]),
                "threshold": float(len(ARMS) // 2 + 1),
                "met": bool(summary["readiness"]["R3_discriminator_nondegenerate"]),
            },
        ],
        "criteria": [
            {
                "name": "diversity_floor_bracketed_or_sub_bracketed",
                "load_bearing": True,
                "passed": bool(summary["outcome"] == "PASS"),
            },
        ],
        "criteria_non_degenerate": {
            "R1_entropy_varies": bool(summary["readiness"]["R1_realised_entropy_varies_across_arms"]),
            "R2_conversion_engaged": bool(summary["readiness"]["R2_topk_conversion_nonvacuous"]),
            "R3_discriminator_nondegenerate": bool(summary["readiness"]["R3_discriminator_nondegenerate"]),
            "floor_bracketed": bool(summary["floor"]["bracketed"]),
        },
    }

    return {
        "outcome": summary["outcome"],
        "overall_direction": summary["evidence_direction"],
        "interpretation_label": summary["label"],
        "interpretation": interpretation,
        "summary": summary,
        "seeds": seeds,
        "n_arms": len(ARMS),
        # episodes_per_run MUST equal the TOTAL training-loop bound per seed-arm.
        "episodes_per_run": int(p0_episodes + p1_episodes + p2_episodes),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "decision_rule_thresholds": {
            "floor_r1_entropy_range": float(FLOOR_R1_ENTROPY_RANGE),
            "shortlist_size_tol": float(SHORTLIST_SIZE_TOL),
            "floor_r3_w_range": float(FLOOR_R3_W_RANGE),
            "acc_above_chance_margin": float(ACC_ABOVE_CHANCE_MARGIN),
            "chance_acc": float(CHANCE_ACC),
            "min_context_ticks": int(MIN_CONTEXT_TICKS),
            "spcem_reference_entropy_nats": float(SPCEM_REFERENCE_ENTROPY_NATS),
            "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
            "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
            "base_select_temperature": float(BASE_SELECT_TEMPERATURE),
        },
        "arm_results": arm_results,
    }


def _build_manifest(
    result: Dict[str, Any], timestamp_utc: str, dry_run: bool
) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    summary = result["summary"]
    return {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": {"ARC-062": result["overall_direction"]},
        "non_degenerate": bool(summary.get("non_degenerate", True)),
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "evidence_direction_note": (
            "V3-EXQ-690 Q-054 ARC-062 minimum trajectory-class diversity floor sweep. "
            "Graded entropy sweep via the MECH-313 noise-floor min_temperature "
            "(0.30/1.00/1.75/2.75) -- the only varied variable; the 569i-validated "
            "TOP-K shortlist conversion stack (use_modulatory_shortlist_then_modulate "
            f"+ mode={MODULATORY_SHORTLIST_MODE} + k={MODULATORY_SHORTLIST_K} + "
            "candidate_summary_source=e2_world_forward + 643a authority + MECH-341 "
            "preserver) armed IDENTICALLY on every arm so realised first-action entropy "
            "can reach committed behaviour (the precondition lifted 2026-06-17 by "
            "V3-EXQ-569i PASS). DV1 = ARC-062 context-discriminator reef-vs-forage "
            "balanced accuracy (gating_weight cut); DV2 = TV(P(a|s_reef),P(a|s_forage)); "
            "both read against the MEASURED upstream first_action_entropy per arm. The "
            "Q-054 answer = locate the entropy floor below which accuracy is at chance "
            "and above which it rises; report whether SP-CEM 0.497 nats sits above "
            "(sufficient) or below (insufficient). label="
            f"{result['interpretation_label']}; spcem_sufficient="
            f"{summary.get('spcem_sufficient')}. Self-routes: R1/R2/R3 unmet -> "
            "substrate_not_ready_requeue; flat-at-chance across ALL arms WITH R1-R3 met "
            "-> residual_conversion_ceiling (F-dominance / MECH-439, /implement-substrate), "
            "NOT a floor verdict and NOT an ARC-062 falsification. Only the floor-bracketed "
            "/ floor-below-lowest branches weight ARC-062 (as supports)."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "swept_variable": "noise_floor_min_temperature",
            "arms_min_temperature": [
                a["noise_floor_min_temperature"] for a in ARMS
            ],
            "noise_floor_alpha": 0.0,
            "conversion_stack": (
                "569i TOP-K shortlist (use_modulatory_shortlist_then_modulate + "
                "mode=top_k + k=3) + channel routing (cand_world_summary) + "
                "std-basis authority gain 2.0 + candidate_summary_source="
                "e2_world_forward + MECH-341 preserver + SP-CEM"
            ),
            "discriminator": "ARC-062 gated-policy 3-stream context discriminator (z_world, z_self, z_harm_a)",
            "dv1": "reef-vs-forage discriminator balanced accuracy (gating_weight cut)",
            "dv2": "TV(P(a|s_reef), P(a|s_forage)) committed-action distributions",
            "x_axis": "MEASURED realised committed first_action_entropy per arm",
            "phases": "P0 e2-train -> P1 frozen-encoder gated-policy REINFORCE -> P2 frozen measurement",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_bipartite_layout": True,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-690 Q-054 ARC-062 diversity-floor sweep"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, p2, steps = DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_P2, DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1 = P1_TRAIN_EPISODES
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

    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    if args.dry_run:
        out_path = out_dir / f"_dry_{manifest['run_id']}.json"

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"label={result['interpretation_label']} "
        f"spcem_sufficient={result['summary'].get('spcem_sufficient')}",
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
