"""V3-EXQ-569i: ARC-065 GAP-A committed-action diversity falsifier on the TOP-K
SHORTLIST conversion substrate, with an IN-ARM route-range gate.

behavioral_diversity_isolation:GAP-A. Successor to V3-EXQ-569h (supersedes it).
ROUTED BY the active failure_autopsy_V3-EXQ-569h_2026-06-16 (which hands off
/implement-substrate amend modulatory-bias-selection-authority) + the
critical_path_synthesis_2026-06-15 Path 2 recommendation (a real architectural
change -- shortlist-then-modulate -- not another gain tweak).

WHY 569i (the single substrate change vs 569h): 569h re-ran the matched-entropy
falsifier with the 684a-identified ADDITIVE conversion config (ARM_STD_G2 =
normalize_basis=std + authority_gain=2.0) ON all arms and landed FAIL/
non_contributory (conversion_ceiling_persists_despite_routing): readiness fully
MET (route_range 0.31 3/3 > floor; e2-divergence 3/3; non_degenerate; negative
control clean) but the load-bearing C_R1B (ARM_1 selected-action entropy strict-
above BOTH matched-noise and proposer) cleared only 1/3 seeds (need 2/3; arm1
selected_entropy 0.785). The 2026-06-15 MARGIN shortlist lever (b) ALSO failed:
V3-EXQ-684 ARM_SHORTLIST (margin 0.25) converted 0/3, committed entropy 0.337
BELOW the collapsed proposer 0.549. DIAGNOSIS (684 manifest): the margin shortlist
admitted modulatory_shortlist_size_mean 6.25-8.54 -- the cutoff
best_raw + 0.25*raw_score_range pulled in ~7 of ~8 candidates = a near-WHOLE,
state-STABLE eligible set, so the committed-branch argmin(_modulatory_accum)
collapsed to the modulatory channel's GLOBAL favourite (monostrategy). The
additive arms (which blend F) preserved MORE diversity than the margin shortlist
that let modulation solely decide over a near-whole set.

THE TOP-K FIX (landed ree-v3 dafc76a; new E3Config modulatory_shortlist_mode
'margin'|'top_k' + modulatory_shortlist_k): in 'top_k' the eligible set is the
k F-best candidates by PRIMARY score (k smallest raw_scores). A SMALL fixed k
gives an eligible set whose MEMBERSHIP ROTATES with state (the k F-best change as
the agent moves), so the within-set argmin of the routed modulatory channel
converts the per-candidate range into committed-action diversity that reflects
genuine per-candidate STRUCTURE -- and BEATS unstructured matched-noise (the
within-set rule stays deterministic, so the entropy is not sampling noise).
Safety preserved: only the k F-best are eligible -> clearly-harmful never
selectable. 569i is the CLAIM-WEIGHTING falsifier that re-runs the SAME 569h
matched-entropy design with the TOP-K shortlist conversion config ON across all
arms (the constant), so the upstream e2_world_forward diversity can finally
CONVERT to committed action through the rotating top-k set.

DESIGN: 3-arm single-variable design, matched seeds -- IDENTICAL in shape to
569h except the conversion constant is now the TOP-K shortlist
(use_modulatory_shortlist_then_modulate=True + modulatory_shortlist_mode='top_k'
+ modulatory_shortlist_k=3 + use_modulatory_channel_routing +
source=cand_world_summary + use_modulatory_selection_authority gain=2.0 std-basis
[harmless: the shortlist overrides selection; authority still arms
_modulatory_accum]), and candidate_summary_source stays the single swept variable
(plus temperature for the noise control). SP-CEM (Layer A) on; the SHARED E3-side
bias channels (lateral_pfc + mech295) ON; SD-056 online contrastive trained on
EVERY arm with the rollout-norm clamp ON (643a float32-cancellation lesson). Each
arm changes ONE thing relative to ARM_0:
  ARM_0_PROPOSER       candidate_summary_source="proposer",        temperature=1.0  (collapsed channel; routed range ~0 -- the no-conversion-reaches baseline)
  ARM_1_E2WF_TOPK      candidate_summary_source="e2_world_forward", temperature=1.0  (the 649 GAP-A fix routed + converted via TOP-K shortlist; routed range ~0.2-0.4; under test)
  ARM_2_MATCHED_NOISE  candidate_summary_source="proposer",        temperature=2.5  (matched-entropy NEGATIVE control; proposer routed range ~0 -> within-top-k argmin over a collapsed channel -> must NOT lift over ARM_0)

ARM_1 vs ARM_0 isolates the CONSUMPTION channel at matched temperature on the
top-k conversion substrate. ARM_1 vs ARM_2 isolates structured-vs-noise: the
higher-temperature proposer arm carries no routed range (collapsed channel), so
its within-top-k argmin is over an unstructured accumulator; the falsifier
requires the structured, route-reaching, TOP-K-converted e2_world_forward channel
to exceed BOTH the collapsed-proposer baseline AND the undirected-noise control.

ACCEPTANCE (evidence, claim_ids=[ARC-065]; plan decision rule R1.b on the top-k
conversion substrate):
  READINESS (load-bearing non-vacuity, both required; RANGE statistics, the
  same-statistic discipline):
    (a) IN-ARM ROUTE-RANGE gate: ARM_1 modulatory_channel_route_range mean
        (the V3-EXQ-662 statistic, read LIVE at the select tick from
        e3.last_score_diagnostics) > ROUTE_RANGE_FLOOR on >= MIN_SEEDS_FOR_PASS
        seeds. Below floor self-routes substrate_not_ready_requeue, NEVER weakens.
    (b) E2-DIVERGENCE gate (C1): ARM_1 cand_world_pairwise_dist
        (e2.world_forward prediction spread) > C1_PAIRWISE_DIST_FLOOR on
        >= MIN_SEEDS_FOR_PASS seeds -- confirms SD-056 trained the
        action-conditional divergence the routed channel re-sources.
  C_R1B PRIMARY (load-bearing): ARM_1 selected_action_class_entropy STRICTLY
    ABOVE BOTH ARM_2_MATCHED_NOISE AND ARM_0_PROPOSER on the SAME seed, on
    >= MIN_SEEDS_FOR_PASS seeds, AND ARM_1 mean > C3_SELECTED_ENTROPY_FLOOR.
  SHORTLIST-ENGAGED sanity (informational, does NOT gate): ARM_1
    modulatory_shortlist_size_mean ~= modulatory_shortlist_k (3) AND
    modulatory_shortlist_mode == 'top_k' -- confirms the top-k mode actually
    fired (the autopsy discriminator: top_k size ~= 3, vs the 684 margin 6.25-8.54
    near-whole set). A size far from k means the shortlist did not engage as
    configured; flagged, does not change the verdict.
  NEGATIVE-CONTROL sanity (informational, does NOT gate): ARM_2 must NOT lift
    committed entropy over ARM_0.
  PASS = READINESS(route+e2-divergence) AND C_R1B -> r1b_diversity_reaches_committed_action;
    ARC-065 GAP-A theory-1 reaches committed action on the top-k conversion
    substrate; evidence_direction=supports.

Interpretation grid (plan R1.b on the top-k conversion substrate):
| outcome                                              | label                                       | evidence_direction | next                                                                 |
|------------------------------------------------------|---------------------------------------------|--------------------|----------------------------------------------------------------------|
| READINESS(route+e2-divergence) + C_R1B               | r1b_diversity_reaches_committed_action      | supports           | R1.b cleared; GAP-A theory-1 confirmed; conversion reaches committed action |
| route-range below floor / non-finite                 | substrate_not_ready_requeue                 | non_contributory   | routing not wired / e2 under-trained; re-queue at higher P0, do NOT weaken |
| e2-divergence below floor (C1 fail)                  | substrate_not_ready_requeue                 | non_contributory   | SD-056 under-trained; re-queue at higher P0; do NOT weaken            |
| READINESS met, C_R1B fail (no strict lift)          | conversion_ceiling_persists_despite_routing | non_contributory   | OFF-RAMP: even the TOP-K shortlist (the architectural change) does not move committed behaviour -> deeper /implement-substrate or /claim-synthesis, NOT a falsification of ARC-065 |

NOTE: there is NO weakens path. ARC-065 is provisional/substrate_ceiling and the
GAP-A design treats "diversity present + routed but not reaching committed action"
as a CONVERSION CEILING (substrate insufficiency), NOT a refutation. The falsifier
teeth are the matched-noise NEGATIVE control + the IN-ARM route-range gate.

claim_ids=[ARC-065] ONLY. MECH-341 GAP-B is NOT active in this lineage
(use_e3_score_diversity=False). ARC-062 / MECH-309 / MECH-294 untouched.

Usage:
  /opt/local/bin/python3 experiments/v3_exq_569i_gapa_conversion_topk_shortlist_falsifier.py --dry-run
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

EXPERIMENT_TYPE = "v3_exq_569i_gapa_conversion_topk_shortlist_falsifier"
QUEUE_ID = "V3-EXQ-569i"
CLAIM_IDS: List[str] = ["ARC-065"]  # GAP-A theory-1 (SP-CEM child / shared-channel consumption)
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60           # SD-056 contrastive warmup (684a/649/569h proven budget)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Acceptance thresholds (pre-registered; mirror 569h).
ROUTE_RANGE_FLOOR = 0.01          # READINESS (a): ARM_1 modulatory_channel_route_range (the V3-EXQ-662 statistic)
C1_PAIRWISE_DIST_FLOOR = 0.03     # READINESS (b) / C1: ARM_1 e2.world_forward prediction spread (SD-056 trained)
C3_SELECTED_ENTROPY_FLOOR = 0.3   # C_R1B: ARM_1 selected-action class entropy floor
MATCHED_ENTROPY_TEMPERATURE = 2.5
MIN_SEEDS_FOR_PASS = 2            # of 3

# SD-056 online contrastive training (mirror 569h harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# TOP-K shortlist conversion config (the 569h-autopsy-routed architectural change;
# ON all arms = the constant). The shortlist takes precedence at SELECTION; the
# additive authority still arms _modulatory_accum (harmless under shortlist).
MODULATORY_SHORTLIST_MODE = "top_k"           # the new lever (vs 684 margin)
MODULATORY_SHORTLIST_K = 3                     # small fixed k -> rotating eligible set
MODULATORY_AUTHORITY_GAIN = 2.0               # 684/684a gain (kept; shortlist overrides selection)
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"  # std basis (kept)
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6       # substrate numerical active/inactive floor

# Behavioural-diversity env: SD-054 reef-bipartite hazard layout (matches 569h exactly).
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

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_PROPOSER",
        "label": "topk_summary_source_proposer_collapsed_baseline",
        "candidate_summary_source": "proposer",
        "temperature": 1.0,
    },
    {
        "arm_id": "ARM_1_E2WF_TOPK",
        "label": "topk_summary_source_e2_world_forward_conversion_under_test",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
    },
    {
        "arm_id": "ARM_2_MATCHED_NOISE",
        "label": "topk_proposer_matched_entropy_temperature_negative_control",
        "candidate_summary_source": "proposer",
        "temperature": MATCHED_ENTROPY_TEMPERATURE,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM stack with the SHARED E3-side bias channels (lateral_pfc +
    mech295) ON and candidate_summary_source set per arm. SD-056 contrastive is
    ENABLED on every arm with the rollout-norm clamp ON. The TOP-K SHORTLIST
    CONVERSION config is ON for EVERY arm (the constant): F (raw primary scores)
    filters to the k F-best candidates, then the routed modulatory channel
    (_modulatory_accum, fed by the e2.world_forward per-candidate range) arbitrates
    the within-set argmin, converting the channel range into committed action."""
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
        # ARC-065 SP-CEM (Layer A) -- main-path default (action-divergent pool)
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # SHARED E3-side bias channels under test (consume cand_world_summaries)
        use_lateral_pfc_analog=True,
        use_mech295_liking_bridge=True,
        # Other policy-layer regulators OFF (candidate_summary_source is the axis)
        use_structured_curiosity=False,
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_ofc_analog=False,
        use_gated_policy=False,
        # SD-056 substrate trained online on every arm (e2.world_forward divergence)
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # --- ARC-065 GAP-A: the swept axis ---
        candidate_summary_source=str(arm["candidate_summary_source"]),
        # --- TOP-K SHORTLIST CONVERSION config (ON all arms; the constant). The
        #     routed cand_world_summaries cross-candidate range feeds _modulatory_accum
        #     (route-range amend); the TOP-K shortlist (k F-best by primary score)
        #     lets the routed channel's within-set argmin convert the range into the
        #     committed selection. The additive authority (gain 2.0, std basis) still
        #     arms _modulatory_accum but the shortlist overrides selection. ARM_0/ARM_2
        #     (proposer source) collapse -> routed range ~0 (the matched controls). ---
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
    """The per-candidate cand_world_summaries the SHARED bias channels actually
    consume this arm: agent._candidate_world_summaries (e2.world_forward source)
    when candidate_summary_source='e2_world_forward'; else the proposer first-step
    z_world. This is the SAME representation the route-range substrate projects
    into the modulatory accumulator."""
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

    consumed_dists: List[float] = []
    consumed_dist_max = 0.0
    pairwise_dists: List[float] = []
    route_ranges: List[float] = []          # IN-ARM modulatory_channel_route_range readout
    route_range_max = 0.0
    authority_active_ticks = 0
    shortlist_sizes: List[float] = []        # IN-ARM modulatory_shortlist_size readout (top-k discriminator)
    shortlist_active_ticks = 0
    shortlist_mode_seen: Optional[str] = None
    candidate_unique_per_tick: List[float] = []
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
                # Candidate-pool first-action diversity (substrate-naive readout).
                pre_e3_classes = [_trajectory_first_action_class(t) for t in candidates]
                candidate_unique_per_tick.append(float(len(set(pre_e3_classes))))
                # C1 substrate-operative: e2.world_forward prediction spread.
                actions_K = _first_actions_K(candidates).to(agent.device)
                z0 = latent.z_world.detach()
                with torch.no_grad():
                    pdist = float(agent.e2.cand_world_pairwise_dist(z0, actions_K).item())
                if math.isfinite(pdist):
                    pairwise_dists.append(pdist)
                # GAP-A statistic: per-candidate spread of the CONSUMED summaries.
                consumed = _consumed_summaries(agent, candidates)
                if consumed is not None and torch.isfinite(consumed).all():
                    cdist = _mean_pairwise_l2(consumed)
                    if math.isfinite(cdist):
                        consumed_dists.append(cdist)
                        consumed_dist_max = max(consumed_dist_max, cdist)

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

            # READINESS (a) / IN-ARM route-range: read the RAW cross-candidate range
            # of the routed modulatory bias (the V3-EXQ-662 statistic), LIVE at the
            # select tick. Also read the TOP-K shortlist size (the autopsy
            # discriminator: top_k -> size ~= k=3; the 684 margin -> 6.25-8.54).
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

            # C_R1B behavioural DV: committed first-action class diversity.
            if is_p1:
                committed_class = int(action[0].argmax().item())
                selected_class_counts[committed_class] += 1
                n_p1_ticks += 1

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

    selected_action_entropy = _entropy_from_counts(dict(selected_class_counts))

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "candidate_summary_source": arm["candidate_summary_source"],
        "temperature": arm_temperature,
        "n_p1_ticks": int(n_p1_ticks),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # READINESS (a) / IN-ARM route-range: RAW cross-candidate range routed into the authority.
        "modulatory_channel_route_range_mean": round(_mean(route_ranges), 6),
        "modulatory_channel_route_range_max": round(route_range_max, 6),
        "modulatory_authority_active_ticks": int(authority_active_ticks),
        # TOP-K shortlist discriminator: top_k -> size ~= k (3); 684 margin -> 6.25-8.54.
        "modulatory_shortlist_size_mean": round(_mean(shortlist_sizes), 6),
        "modulatory_shortlist_active_ticks": int(shortlist_active_ticks),
        "modulatory_shortlist_mode": shortlist_mode_seen or "",
        # GAP-A statistic: per-candidate spread of the CONSUMED summaries (informational).
        "consumed_summary_pairwise_dist_mean": round(_mean(consumed_dists), 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        # READINESS (b) / C1: e2.world_forward prediction spread.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        # Candidate-pool first-action diversity (interpretation aid).
        "candidate_unique_first_action_classes_mean": round(_mean(candidate_unique_per_tick), 6),
        "trajectory_class_count_mean": round(_mean(candidate_unique_per_tick), 6),
        # C_R1B behavioural DV.
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": int(len(selected_class_counts)),
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


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    arm0 = _arm_rows(arm_results, "ARM_0_PROPOSER")
    arm1 = _arm_rows(arm_results, "ARM_1_E2WF_TOPK")
    arm2 = _arm_rows(arm_results, "ARM_2_MATCHED_NOISE")
    arm0_by_seed = {r["seed"]: r for r in arm0}
    arm2_by_seed = {r["seed"]: r for r in arm2}

    RDIST = "modulatory_channel_route_range_mean"
    PDIST = "cand_world_pairwise_dist_mean"
    SENT = "selected_action_class_entropy"
    SLSIZE = "modulatory_shortlist_size_mean"

    # READINESS (a): IN-ARM route-range gate (load-bearing non-vacuity; the
    # V3-EXQ-662 RANGE statistic the routing gates on -- read live at select).
    arm1_route_mean = _mean_key(arm1, RDIST)
    route_seeds_ok = _n_seeds(arm1, lambda r: float(r.get(RDIST, 0.0)) > ROUTE_RANGE_FLOOR)
    route_ready = bool(route_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # READINESS (b) / C1: e2.world_forward prediction spread (load-bearing
    # non-vacuity; RANGE statistic; SD-056 trained the action-conditional divergence).
    arm1_pdist_mean = _mean_key(arm1, PDIST)
    c1_seeds_ok = _n_seeds(arm1, lambda r: float(r.get(PDIST, 0.0)) > C1_PAIRWISE_DIST_FLOOR)
    c1_pass = bool(c1_seeds_ok >= MIN_SEEDS_FOR_PASS)

    readiness_ok = bool(route_ready and c1_pass)

    # SHORTLIST-ENGAGED sanity (informational, does NOT gate): top-k mode actually
    # fired with size ~= k (the autopsy discriminator vs the 684 margin near-whole set).
    arm1_shortlist_size_mean = _mean_key(arm1, SLSIZE)
    arm1_modes = {str(r.get("modulatory_shortlist_mode", "")) for r in arm1}
    top_k_engaged = bool(
        arm1_modes == {MODULATORY_SHORTLIST_MODE}
        and abs(arm1_shortlist_size_mean - float(MODULATORY_SHORTLIST_K)) <= 0.5
    )

    # C_R1B PRIMARY (load-bearing): ARM_1 selected-action class entropy STRICTLY
    # ABOVE both ARM_2_MATCHED_NOISE and ARM_0_PROPOSER on the same seed.
    def _r1b(r1: Dict[str, Any]) -> bool:
        r0 = arm0_by_seed.get(r1["seed"])
        r2 = arm2_by_seed.get(r1["seed"])
        if r0 is None or r2 is None:
            return False
        e1 = float(r1.get(SENT, 0.0))
        return e1 > float(r0.get(SENT, 0.0)) and e1 > float(r2.get(SENT, 0.0))
    r1b_seeds_ok = _n_seeds(arm1, _r1b)
    arm1_sel_mean = _mean_key(arm1, SENT)
    r1b_floor_ok = bool(arm1_sel_mean > C3_SELECTED_ENTROPY_FLOOR)
    r1b_pass = bool(r1b_seeds_ok >= MIN_SEEDS_FOR_PASS and r1b_floor_ok)

    # NEGATIVE-CONTROL sanity (informational, does NOT gate): ARM_2 must NOT lift
    # committed entropy over ARM_0.
    def _noise_lifts(r2: Dict[str, Any]) -> bool:
        r0 = arm0_by_seed.get(r2["seed"])
        if r0 is None:
            return False
        return float(r2.get(SENT, 0.0)) > float(r0.get(SENT, 0.0))
    noise_lift_seeds = _n_seeds(arm2, _noise_lifts)
    negative_control_does_not_lift = bool(noise_lift_seeds == 0)

    # Non-degeneracy: every measured arm produced P1 ticks (the C_R1B metric can move).
    all_arms = [arm0, arm1, arm2]
    non_degenerate = bool(
        all(len(a) > 0 for a in all_arms)
        and all(int(r.get("n_p1_ticks", 0)) > 0 for a in all_arms for r in a)
    )

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif r1b_pass:
        label = "r1b_diversity_reaches_committed_action"
        overall_pass = True
        evidence_direction = "supports"
    else:
        # OFF-RAMP: routed channel reaches the top-k shortlist (readiness met) but
        # no committed-action lift -> CONVERSION CEILING -> deeper /implement-substrate
        # or /claim-synthesis, NOT a falsification of ARC-065.
        label = "conversion_ceiling_persists_despite_routing"
        overall_pass = False
        evidence_direction = "non_contributory"

    return {
        "readiness": {
            "route_range_floor": ROUTE_RANGE_FLOOR,
            "arm1_route_range_mean": round(arm1_route_mean, 6),
            "arm1_seeds_route_above_floor": int(route_seeds_ok),
            "route_ready": route_ready,
            "c1_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
            "arm1_pairwise_dist_mean": round(arm1_pdist_mean, 6),
            "arm1_seeds_e2_divergent": int(c1_seeds_ok),
            "c1_pass": c1_pass,
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "readiness_ok": readiness_ok,
        },
        "shortlist_engaged": {
            "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
            "arm1_shortlist_size_mean": round(arm1_shortlist_size_mean, 6),
            "arm1_shortlist_modes": sorted(arm1_modes),
            "top_k_engaged": top_k_engaged,
            "note": (
                "Informational sanity: confirms the TOP-K shortlist fired with "
                "size ~= k=3 (the autopsy discriminator -- the 684 margin shortlist "
                "showed modulatory_shortlist_size_mean 6.25-8.54, a near-whole set). "
                "Does NOT gate the verdict."
            ),
        },
        "negative_control": {
            "matched_noise_lift_seeds_over_proposer": int(noise_lift_seeds),
            "negative_control_does_not_lift": negative_control_does_not_lift,
            "negative_control_unexpectedly_lifted": bool(not negative_control_does_not_lift),
            "note": (
                "ARM_2_MATCHED_NOISE (proposer @ T=2.5) is an UNDIRECTED negative "
                "control: it MUST NOT lift committed entropy over ARM_0_PROPOSER "
                "(its routed channel carries no range, so the within-top-k argmin is "
                "over an unstructured accumulator). Informational sanity only; does "
                "NOT gate the verdict."
            ),
        },
        "c_r1b": {
            "arm1_seeds_strict_above_both": int(r1b_seeds_ok),
            "arm1_selected_entropy_mean": round(arm1_sel_mean, 6),
            "selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
            "r1b_floor_ok": r1b_floor_ok,
            "c_r1b_pass": r1b_pass,
        },
        "route_range_per_arm_mean": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, RDIST), 6),
            "ARM_1_E2WF_TOPK": round(_mean_key(arm1, RDIST), 6),
            "ARM_2_MATCHED_NOISE": round(_mean_key(arm2, RDIST), 6),
        },
        "shortlist_size_per_arm_mean": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, SLSIZE), 6),
            "ARM_1_E2WF_TOPK": round(arm1_shortlist_size_mean, 6),
            "ARM_2_MATCHED_NOISE": round(_mean_key(arm2, SLSIZE), 6),
        },
        "selected_action_entropy_per_arm_mean": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, SENT), 6),
            "ARM_1_E2WF_TOPK": round(arm1_sel_mean, 6),
            "ARM_2_MATCHED_NOISE": round(_mean_key(arm2, SENT), 6),
        },
        "trajectory_class_count_per_arm_mean": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, "trajectory_class_count_mean"), 6),
            "ARM_1_E2WF_TOPK": round(_mean_key(arm1, "trajectory_class_count_mean"), 6),
            "ARM_2_MATCHED_NOISE": round(_mean_key(arm2, "trajectory_class_count_mean"), 6),
        },
        "label": label,
        "evidence_direction": evidence_direction,
        "overall_pass": overall_pass,
        "preconditions": [
            {
                "name": "arm1_modulatory_channel_route_range_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_1 (e2_world_forward source) IN-ARM RAW cross-candidate "
                    "RANGE of the modulatory bias ROUTED into the E3 selection "
                    "authority (modulatory_channel_route_range, the V3-EXQ-662 "
                    "statistic, read LIVE at the select tick) clears the floor: the "
                    "channel range REACHES the bias the top-k shortlist arbitrates. "
                    "SAME range statistic the route-range substrate gates on. Below "
                    "floor => routing not wired / e2 under-trained => "
                    "substrate_not_ready_requeue, never a weakens."
                ),
                "control": (
                    "ARM_1: use_modulatory_channel_routing + source=cand_world_summary "
                    "+ top-k shortlist (k=3); candidate_summary_source=e2_world_forward "
                    "(genuinely action-divergent routed channel)"
                ),
                "measured": round(arm1_route_mean, 6),
                "threshold": ROUTE_RANGE_FLOOR,
                "met": route_ready,
            },
            {
                "name": "arm1_e2_world_forward_prediction_spread_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_1 e2.world_forward(z0, a_i) per-candidate prediction spread "
                    "(cand_world_pairwise_dist) clears the floor -- confirms SD-056 "
                    "trained the action-conditional divergence the routed channel "
                    "re-sources. RANGE statistic. Below floor => SD-056 under-trained "
                    "=> substrate_not_ready_requeue."
                ),
                "control": "ARM_1: agent.e2.cand_world_pairwise_dist on SP-CEM candidates",
                "measured": round(arm1_pdist_mean, 6),
                "threshold": C1_PAIRWISE_DIST_FLOOR,
                "met": c1_pass,
            },
        ],
        "criteria": [
            {"name": "C1_arm1_e2_world_forward_divergent", "load_bearing": True, "passed": c1_pass},
            {"name": "C_R1B_selected_entropy_strict_above_matched_noise_and_proposer",
             "load_bearing": True, "passed": r1b_pass},
        ],
        "criteria_non_degenerate": {"C1": non_degenerate, "C_R1B": non_degenerate},
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
                    "arm": {k: arm[k] for k in ("arm_id", "candidate_summary_source", "temperature")},
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "conversion_config": {
                        "use_modulatory_channel_routing": True,
                        "modulatory_channel_route_source": "cand_world_summary",
                        "use_modulatory_selection_authority": True,
                        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
                        "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
                        "use_modulatory_shortlist_then_modulate": True,
                        "modulatory_shortlist_mode": MODULATORY_SHORTLIST_MODE,
                        "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
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
        "supersedes": "V3-EXQ-569h",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"ARC-065": evidence_direction},
        "non_degenerate": summary.get("non_degenerate", True),
        "evidence_direction_note": (
            "ARC-065 GAP-A committed-action diversity falsifier on the TOP-K SHORTLIST "
            "conversion substrate (use_modulatory_shortlist_then_modulate + "
            "modulatory_shortlist_mode=top_k + modulatory_shortlist_k=3 + "
            "use_modulatory_channel_routing + source=cand_world_summary + "
            "use_modulatory_selection_authority gain=2.0 std-basis = the constant ON all "
            "arms). ROUTED BY failure_autopsy_V3-EXQ-569h + critical_path_synthesis Path 2. "
            "supersedes V3-EXQ-569h (which proved REACH + CONVERSION via additive STD_G2 "
            "but cleared C_R1B only 1/3; the 684 MARGIN shortlist converted 0/3 because the "
            "margin admitted a near-whole state-stable set [size 6.25-8.54] -> argmin "
            "collapsed to the channel global favourite). 569i re-runs the SAME 3-arm "
            "matched-entropy falsifier with the TOP-K shortlist (k F-best by primary score "
            "-> rotating eligible set -> within-set argmin of the routed channel converts "
            "the range into committed diversity). PASS (label="
            "r1b_diversity_reaches_committed_action) = ARM_1 (e2_world_forward + top_k) "
            "selected-action entropy strictly above BOTH the matched-entropy noise control "
            "(proposer T=2.5) and the collapsed-proposer baseline on >=2/3 seeds, with the "
            "IN-ARM route-range readiness gate AND e2-divergence (SD-056 trained) met => "
            "R1.b fires; ARC-065 GAP-A theory-1 NOW reaches committed action on the top-k "
            "conversion substrate (supports). Route-range-below-floor OR e2-divergence-"
            "below-floor self-routes substrate_not_ready_requeue => non_contributory, NEVER "
            "a weakens. Readiness met but no strict lift => "
            "conversion_ceiling_persists_despite_routing => non_contributory + OFF-RAMP "
            "(even the TOP-K architectural change does not move committed behaviour -> "
            "deeper substrate enrichment or /claim-synthesis, NOT a falsification of "
            "ARC-065). NO weakens path. claim_ids=[ARC-065] only; MECH-341 GAP-B substrate "
            "NOT active (use_e3_score_diversity=False); ARC-062/MECH-309/MECH-294 untouched."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "r1b_diversity_reaches_committed_action": "PASS -> R1.b cleared; GAP-A theory-1 (ARC-065) confirmed as a real diversity contributor that reaches committed action on the top-k conversion substrate",
                "substrate_not_ready_requeue": "re-queue at higher P0 budget (or check route-range wiring / SD-056 training); do NOT weaken ARC-065",
                "conversion_ceiling_persists_despite_routing": "OFF-RAMP -> deeper /implement-substrate or /claim-synthesis: even the TOP-K shortlist (the architectural change) does NOT move committed behaviour; substrate enrichment, NOT a falsification of ARC-065",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": ENV_KWARGS,
            "arms": [{k: a[k] for k in ("arm_id", "label", "candidate_summary_source", "temperature")} for a in ARMS],
            "sd056_weight": SD056_WEIGHT,
            "matched_entropy_temperature": MATCHED_ENTROPY_TEMPERATURE,
            "conversion_config": {
                "use_modulatory_channel_routing": True,
                "modulatory_channel_route_source": "cand_world_summary",
                "modulatory_channel_route_weight": 1.0,
                "modulatory_channel_route_min_range_floor": MODULATORY_ROUTE_MIN_RANGE_FLOOR,
                "use_modulatory_selection_authority": True,
                "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
                "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
                "use_modulatory_shortlist_then_modulate": True,
                "modulatory_shortlist_mode": MODULATORY_SHORTLIST_MODE,
                "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
            },
            "thresholds": {
                "route_range_floor": ROUTE_RANGE_FLOOR,
                "c1_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
                "c3_selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_route_range_ready": summary["readiness"]["route_ready"],
            "readiness_e2_divergent_ready": summary["readiness"]["c1_pass"],
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "top_k_shortlist_engaged": summary["shortlist_engaged"]["top_k_engaged"],
            "negative_control_does_not_lift": summary["negative_control"]["negative_control_does_not_lift"],
            "C_R1B_selected_entropy_strict_above_both": summary["c_r1b"]["c_r1b_pass"],
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
        description="V3-EXQ-569i ARC-065 GAP-A committed-action diversity falsifier (top-k shortlist conversion substrate)"
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
