"""V3-EXQ-649: ARC-065 GAP-A shared cand_world_summaries e2.world_forward
substrate-readiness diagnostic.

Routed by failure_autopsy_V3-EXQ-614e_2026-06-07 (confirmed; substrate_ceiling /
GAP-A). Validates the SHARED-channel e2.world_forward re-sourcing landed
2026-06-07 (ree-v3 commit 71dfb2b; REEConfig.candidate_summary_source). The
MECH-314a Phase-2 amend (648a) fixed ONLY the curiosity channel's consumed
representation; GAP-A extends the identical re-sourcing to the SHARED
cand_world_summaries consumed by lateral_pfc / ofc / mech295 / gated_policy /
tonic_vigor.

ROOT CAUSE (V3-EXQ-614e): with the modulatory-bias-selection-authority gate
PROVEN operative (V3-EXQ-643a), committed-class diversity still showed no lift
because every E3-side bias channel sees a class-uniform candidate pool -- all K
CEM candidates produce identical z_world after one E2 world-forward step
(cand_world_pairwise_dist=0.0000) despite differing first actions. The SHARED
cand_world_summaries was built from the collapsed proposer first-step z_world
(trajectory.world_states[:,0,:]). The fix re-sources it from the SD-056-trained
action-conditional e2.world_forward(z0, a_i), which carries per-action spread.

DESIGN: 2-arm ablation, matched seeds, SP-CEM on, the SHARED bias channels
(lateral_pfc + mech295) ON. SD-056 online contrastive trained in BOTH arms (the
substrate the e2.world_forward divergence depends on; the SD-056 rollout-norm
clamp is ON per the 643a numerical-stability lesson). The ONLY swept axis is
candidate_summary_source.
  ARM_0_PROPOSER       candidate_summary_source="proposer"        (614e baseline)
  ARM_1_E2_WORLD_FWD   candidate_summary_source="e2_world_forward" (GAP-A fix)

The measured statistic is the per-candidate SPREAD of the representation the bias
channels ACTUALLY consume this arm (consumed_summary_pairwise_dist): for ARM_1 it
is e2.world_forward(z0, a_i); for ARM_0 it is the proposer world_states[:,0,:].

ACCEPTANCE (substrate-readiness, claim_ids=[] -- does NOT weight claim confidence):
  READINESS (load-bearing non-vacuity): ARM_1 consumed_summary_pairwise_dist mean
    > C0_SPREAD_FLOOR on >= MIN_SEEDS_FOR_PASS seeds, finite + below the 643a
    explosion ceiling. Below floor (under-trained e2 / amend not wired) self-routes
    substrate_not_ready_requeue, NEVER a substrate verdict.
  C1 backward-compat: ARM_0 proposer consumed spread collapses (<= C1_COLLAPSE_CEIL)
    on >= MIN_SEEDS_FOR_PASS seeds -- reproduces the 614e cand_world_pairwise_dist
    ~0 signature.
  C2 PRIMARY (load-bearing): ARM_1 consumed spread > ARM_0 consumed spread per
    matched seed on >= MIN_SEEDS_FOR_PASS seeds -- the GAP-A lift (e2_world_forward
    >> proposer). Routes on the SAME consumed_summary_pairwise_dist statistic the
    readiness precondition asserts (exact-statistic, range-based; no
    magnitude-vs-range mismatch).
  PASS = READINESS AND C1 AND C2.

PASS (label=gapa_shared_channel_ready) unblocks the MECH-341 committed-class
diversity re-test (a within-class-REPRESENTATIVE-diversity readout, NOT
committed-class entropy -- 614e autopsy Learning #2), queued separately. MECH-341
stays candidate / v3_pending; not weakened.

Interpretation grid:
| outcome                              | label                          | next                                                        |
|--------------------------------------|--------------------------------|-------------------------------------------------------------|
| READINESS+C1+C2                      | gapa_shared_channel_ready      | /queue-experiment MECH-341 within-class-representative retest |
| readiness leg below floor/non-finite | substrate_not_ready_requeue    | re-queue as 649a at higher P0 (or fix e2/SD-056 wiring); do NOT weaken |
| readiness ok but C1 or C2 fail       | gapa_shared_channel_inert      | /failure-autopsy on the failing criterion                   |

Usage:
  /opt/local/bin/python3 experiments/v3_exq_649_arc065_gapa_shared_candidate_summary_source.py --dry-run
"""

import argparse
import json
import math
import random
import sys
from collections import deque
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

EXPERIMENT_TYPE = "v3_exq_649_arc065_gapa_shared_candidate_summary_source"
QUEUE_ID = "V3-EXQ-649"
CLAIM_IDS: List[str] = []  # substrate-readiness diagnostic (gates the MECH-341 GAP-A retest)
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60           # SD-056 contrastive warmup (V3-EXQ-604a/648a proven budget)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200
MEASURE_AFTER_TICK = 20

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30
DRY_RUN_MEASURE_AFTER_TICK = 2

# Acceptance thresholds (pre-registered).
C0_SPREAD_FLOOR = 0.05            # readiness: ARM_1 consumed-summary pairwise spread (SD-056 magnitude)
C0_MAGNITUDE_CEIL = 1.0e6         # readiness: rolled-out z_world spread bounded (643a guard)
C1_COLLAPSE_CEIL = 0.05           # C1: ARM_0 proposer consumed spread collapses (614e signature)
MIN_SEEDS_FOR_PASS = 2            # of 3

# SD-056 online contrastive training (mirror V3-EXQ-604a / 648a harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# HARM-FREE env: SP-CEM + resources give action-divergent candidates for SD-056
# to train z_world divergence on; no hazards needed for the shared-summary test.
ENV_KWARGS: Dict[str, Any] = dict(
    size=12,
    num_hazards=0,
    num_resources=5,
    hazard_harm=0.0,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.0,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
)

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_PROPOSER",
        "label": "shared_summary_source_proposer_614e_baseline",
        "candidate_summary_source": "proposer",
    },
    {
        "arm_id": "ARM_1_E2_WORLD_FWD",
        "label": "shared_summary_source_e2_world_forward_gapa_fix",
        "candidate_summary_source": "e2_world_forward",
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM stack with the SHARED E3-side bias channels (lateral_pfc +
    mech295) ON and candidate_summary_source set per arm. SD-056 contrastive is
    ENABLED on every arm (the e2.world_forward divergence the GAP-A fix consumes
    depends on it) with the rollout-norm clamp ON (643a stability lesson)."""
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
    )
    agent = REEAgent(cfg)
    return agent


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _first_actions_K(candidates) -> torch.Tensor:
    rows = []
    for traj in candidates:
        rows.append(traj.actions[:, 0, :].detach().reshape(-1))
    return torch.stack(rows, dim=0)


def _consumed_summaries(agent: REEAgent, candidates) -> Optional[torch.Tensor]:
    """The per-candidate cand_world_summaries the SHARED bias channels actually
    consume this arm: agent._candidate_world_summaries (e2.world_forward source)
    when candidate_summary_source='e2_world_forward'; else the proposer first-step
    z_world (trajectory.world_states[:,0,:]) -- the exact pre/post-GAP-A contrast."""
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
    """Mean pairwise L2 over the K rows = cand_world_pairwise_dist (the 614e
    statistic, =0.0000 under monostrategy)."""
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
    measure_after_tick: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes

    consumed_dists: List[float] = []
    consumed_dist_max = 0.0
    goal_prox_ranges: List[float] = []
    n_p1_ticks_past_window = 0
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

            past_window = is_p1 and tick_in_ep >= measure_after_tick
            if past_window and candidates and len(candidates) >= 2:
                consumed = _consumed_summaries(agent, candidates)
                if consumed is not None and torch.isfinite(consumed).all():
                    dist = _mean_pairwise_l2(consumed)
                    if math.isfinite(dist):
                        consumed_dists.append(dist)
                        consumed_dist_max = max(consumed_dist_max, dist)
                        if agent.goal_state is not None:
                            with torch.no_grad():
                                gp = agent.goal_state.goal_proximity(consumed)
                            gp = gp.detach().reshape(-1)
                            if gp.numel() >= 2 and torch.isfinite(gp).all():
                                goal_prox_ranges.append(
                                    float(gp.max() - gp.min())
                                )
                    n_p1_ticks_past_window += 1

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

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

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "candidate_summary_source": arm["candidate_summary_source"],
        "n_p1_ticks_past_window": int(n_p1_ticks_past_window),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # The exact GAP-A statistic: per-candidate spread of the CONSUMED summaries.
        "consumed_summary_pairwise_dist_mean": round(_mean(consumed_dists), 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        # Secondary propagation readout: per-candidate goal_proximity range over the
        # consumed summaries (how much the divergent rep would vary a goal-conditioned
        # bias across candidates). Non-load-bearing diagnostic.
        "goal_proximity_range_mean": round(_mean(goal_prox_ranges), 8),
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
    arm1 = _arm_rows(arm_results, "ARM_1_E2_WORLD_FWD")
    arm0_by_seed = {r["seed"]: r for r in arm0}

    DIST = "consumed_summary_pairwise_dist_mean"

    # READINESS (load-bearing non-vacuity): ARM_1 consumed spread > floor, bounded.
    readiness_seeds_ok = _n_seeds(arm1, lambda r: float(r.get(DIST, 0.0)) > C0_SPREAD_FLOOR)
    arm1_dist_mean = _mean_key(arm1, DIST)
    max_dist = max(
        [float(r.get("consumed_summary_pairwise_dist_max", 0.0)) for r in arm_results] or [0.0]
    )
    magnitude_ok = bool(math.isfinite(max_dist) and max_dist < C0_MAGNITUDE_CEIL)
    readiness_ok = bool(readiness_seeds_ok >= MIN_SEEDS_FOR_PASS and magnitude_ok)

    # C1 backward-compat: ARM_0 proposer consumed spread collapses (614e signature).
    c1_seeds_ok = _n_seeds(arm0, lambda r: float(r.get(DIST, 0.0)) <= C1_COLLAPSE_CEIL)
    c1_pass = bool(c1_seeds_ok >= MIN_SEEDS_FOR_PASS)
    c1_non_degenerate = bool(arm1_dist_mean > C0_SPREAD_FLOOR)  # real contrast exists

    # C2 PRIMARY (load-bearing): ARM_1 consumed spread > ARM_0 per matched seed.
    def _c2(r1: Dict[str, Any]) -> bool:
        r0 = arm0_by_seed.get(r1["seed"])
        if r0 is None:
            return False
        return float(r1.get(DIST, 0.0)) > float(r0.get(DIST, 0.0))
    c2_seeds_ok = _n_seeds(arm1, _c2)
    c2_pass = bool(c2_seeds_ok >= MIN_SEEDS_FOR_PASS)
    c2_non_degenerate = bool(
        len(arm1) > 0 and all(int(r.get("n_p1_ticks_past_window", 0)) > 0 for r in arm1)
        and len(arm0) > 0 and all(int(r.get("n_p1_ticks_past_window", 0)) > 0 for r in arm0)
    )

    criteria_pass = {"C1": c1_pass, "C2": c2_pass}
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
    elif c1_pass and c2_pass:
        label = "gapa_shared_channel_ready"
        overall_pass = True
    else:
        label = "gapa_shared_channel_inert"
        overall_pass = False

    return {
        "readiness": {
            "c0_spread_floor": C0_SPREAD_FLOOR,
            "arm1_consumed_spread_mean": round(arm1_dist_mean, 6),
            "arm1_seeds_above_floor": int(readiness_seeds_ok),
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "max_consumed_spread_observed": round(max_dist, 6),
            "magnitude_ceil": C0_MAGNITUDE_CEIL,
            "magnitude_ok": magnitude_ok,
            "readiness_ok": readiness_ok,
        },
        "criteria_pass": criteria_pass,
        "c1_arm0_seeds_collapsed": int(c1_seeds_ok),
        "c2_arm1_seeds_lift_over_proposer": int(c2_seeds_ok),
        "consumed_spread_per_arm_mean": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, DIST), 6),
            "ARM_1_E2_WORLD_FWD": round(_mean_key(arm1, DIST), 6),
        },
        "goal_proximity_range_per_arm_mean": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, "goal_proximity_range_mean"), 8),
            "ARM_1_E2_WORLD_FWD": round(_mean_key(arm1, "goal_proximity_range_mean"), 8),
        },
        "label": label,
        "overall_pass": overall_pass,
        # Diagnostic adjudication structures (skill Step 3.5).
        "preconditions": [
            {
                "name": "arm1_consumed_summary_spread_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_1 (e2_world_forward source) per-candidate SPREAD of the "
                    "CONSUMED cand_world_summaries (the representation the SHARED bias "
                    "channels actually read) clears the floor -- a cross-candidate "
                    "RANGE statistic, the SAME one C1/C2 route on, NOT a magnitude. "
                    "Below floor => under-trained e2 / amend not wired => "
                    "substrate_not_ready_requeue, never a substrate verdict."
                ),
                "control": "ARM_1: SD-056 contrastive trained online; SP-CEM multi-class candidates; candidate_summary_source=e2_world_forward",
                "measured": round(arm1_dist_mean, 6),
                "threshold": C0_SPREAD_FLOOR,
                "met": bool(readiness_seeds_ok >= MIN_SEEDS_FOR_PASS),
            },
            {
                "name": "rolled_out_zworld_spread_bounded",
                "kind": "readiness",
                "description": (
                    "Consumed-summary spread stayed finite and below the 643a "
                    "explosion ceiling (SD-056 online training numerical stability; "
                    "rollout-norm clamp ON)."
                ),
                "control": "max consumed_summary_pairwise_dist across all arms",
                "measured": round(max_dist, 6),
                "threshold": C0_MAGNITUDE_CEIL,
                "met": magnitude_ok,
            },
        ],
        "criteria": [
            {"name": "C1_proposer_consumed_spread_collapsed", "load_bearing": False, "passed": c1_pass},
            {"name": "C2_e2_world_forward_lifts_consumed_spread_over_proposer",
             "load_bearing": True, "passed": c2_pass},
        ],
        "criteria_non_degenerate": {"C1": c1_non_degenerate, "C2": c2_non_degenerate},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    measure_after = DRY_RUN_MEASURE_AFTER_TICK if dry_run else MEASURE_AFTER_TICK

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = _run_seed_arm(arm, seed, p0, p1, steps, measure_after)
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {k: arm[k] for k in ("arm_id", "candidate_summary_source")},
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
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
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "ARC-065 GAP-A substrate-readiness diagnostic for the SHARED "
            "cand_world_summaries e2.world_forward source (ree-v3 71dfb2b; "
            "REEConfig.candidate_summary_source). Routed by "
            "failure_autopsy_V3-EXQ-614e_2026-06-07. claim_ids=[] (does NOT weight "
            "claim confidence). PASS (label=gapa_shared_channel_ready) unblocks the "
            "MECH-341 committed-class diversity re-test (within-class-REPRESENTATIVE "
            "diversity readout, NOT committed-class entropy -- 614e autopsy Learning "
            "#2). Readiness-below-floor (ARM_1 consumed-summary spread below floor / "
            "non-finite) self-routes substrate_not_ready_requeue, NOT a substrate "
            "verdict. MECH-341 stays candidate / v3_pending; not weakened."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "gapa_shared_channel_ready": "PASS -> /queue-experiment MECH-341 within-class-representative-diversity retest",
                "substrate_not_ready_requeue": "re-queue as V3-EXQ-649a at higher P0 budget (or fix e2/SD-056 wiring); do NOT weaken",
                "gapa_shared_channel_inert": "FAIL -> /failure-autopsy on the failing criterion",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "measure_after_tick": measure_after,
            "env_kwargs": ENV_KWARGS,
            "arms": [{k: a[k] for k in ("arm_id", "label", "candidate_summary_source")} for a in ARMS],
            "sd056_weight": SD056_WEIGHT,
            "thresholds": {
                "c0_spread_floor": C0_SPREAD_FLOOR,
                "c0_magnitude_ceil": C0_MAGNITUDE_CEIL,
                "c1_collapse_ceil": C1_COLLAPSE_CEIL,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "C1_proposer_consumed_spread_collapsed": summary["criteria_pass"]["C1"],
            "C2_e2_world_forward_lifts_consumed_spread": summary["criteria_pass"]["C2"],
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

    print(f"Outcome: {outcome} (label={summary['label']})", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-649 ARC-065 GAP-A shared cand_world_summaries e2.world_forward substrate-readiness diagnostic"
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
