#!/opt/local/bin/python3
"""
V3-EXQ-526 -- Q-034 Reef-Enabled Hazard/Resource Threshold Sweep

Claims: Q-034
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES: V3-EXQ-451 (monostrategy failure -- agent used fixed route so
goal_reach_rate did not decrease with hazard_density, C1 monotonicity failed)

=== SCIENTIFIC QUESTION ===

Q-034: at what hazard density / resource scarcity does an agent's
goal-directed behaviour collapse into pure avoidance?

=== WHY THE PREDECESSOR FAILED ===

V3-EXQ-451 FAILed on C1 (monotonicity): goal_reach_rate did not strictly
decrease with hazard_density in any column. The agent adopted a single
fixed-route strategy regardless of hazard level. When hazards happen to
block the route, goal_reach_rate collapses; otherwise it stays high. The
result is a step-function (or random variation) rather than a monotone
relationship.

=== REEF FIX ===

The reef substrate (SD-050, landed 2026-05-04) adds two behavioural
attractors to the environment:
  1. Reef safe zones: circular patches where hazards cannot enter, providing
     a safe refuge from harm. Agents can seek the reef when hazard pressure
     is high.
  2. Food-attracted hazards: hazards bias their drift toward food cells,
     making foraging actively more dangerous at high hazard densities.

Together these create a genuine hazard-density-dependent trade-off: at low
hazard density, foraging is feasible and goal_reach_rate is high; at high
hazard density, foraging through food-attracted hazards is costly and agents
retreat to the reef, lowering goal_reach_rate. This is the monotone
relationship that C1 requires.

=== DESIGN ===

Same 15-grid-point sweep as V3-EXQ-451:
  hazard_density in {0.05, 0.10, 0.15, 0.20, 0.25}
  resource_density in {0.04, 0.08, 0.12}

Grid upgraded from 8x8 to 10x10 to accommodate 3 reef patches of radius 2
without crowding the foraging space.

Object counts on 10x10 grid (size=10): num_objects = max(1, round(density * 100)).
  hd=0.05 -> 5;  hd=0.10 -> 10;  hd=0.15 -> 15;  hd=0.20 -> 20; hd=0.25 -> 25
  rd=0.04 -> 4;  rd=0.08 -> 8;   rd=0.12 -> 12

Reef: reef_enabled=True, n_reef_patches=3, reef_patch_radius=2,
      hazard_food_attraction=0.7

Seeds per point: 2. Episodes per run: 30 warmup + 30 eval = 60 total.
Steps per ep: 200. Total runs: 30 (15 conditions x 2 seeds).

=== ACCEPTANCE CRITERIA ===

C1 monotonicity: goal_reach_rate strictly decreasing in hazard_density within
  each resource_density column, holding in >= 2/3 columns.
C2 threshold_identifiable: at least one grid point >= 0.5 AND at least one < 0.5.
C3 coverage: all 15 grid points complete without error and produce non-null metrics.
C4 avoidance_signature (diagnostic): at hd=0.25,rd=0.04 harm_exposure_mean <
  harm_exposure_mean at hd=0.25,rd=0.12.

PASS: C1 AND C2 AND C3. C4 is diagnostic only.
evidence_direction: supports (PASS) or weakens (FAIL).

=== RUNTIME ===

30 runs x 60 episodes x 200 steps. ~220 min on Mac.
"""

import sys
import math
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_526_q034_reef_threshold_sweep"
CLAIM_IDS          = ["Q-034"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES_EXQ     = "V3-EXQ-451"

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
HAZARD_DENSITIES   = [0.05, 0.10, 0.15, 0.20, 0.25]
RESOURCE_DENSITIES = [0.04, 0.08, 0.12]
SEEDS              = [42, 7]

GRID_SIZE          = 10

# Reef parameters (SD-050)
REEF_ENABLED             = True
N_REEF_PATCHES           = 3
REEF_PATCH_RADIUS        = 2
HAZARD_FOOD_ATTRACTION   = 0.7

# ---------------------------------------------------------------------------
# Training / eval parameters
# ---------------------------------------------------------------------------
WARMUP_EPISODES = 30
EVAL_EPISODES   = 30
EPISODES_PER_RUN = WARMUP_EPISODES + EVAL_EPISODES  # 60
STEPS_PER_EP    = 200
GREEDY_FRAC     = 0.5

WORLD_DIM       = 32
BATCH_SIZE      = 16
MAX_BUF         = 2000
WF_BUF_MAX      = 1500

LR_E1           = 1e-3
LR_E2_WF        = 1e-3
LR_HARM         = 1e-4
LR_BENEFIT      = 1e-3
LR_RP           = 1e-3
LAMBDA_RESOURCE = 0.5

# ---------------------------------------------------------------------------
# Acceptance criteria thresholds
# ---------------------------------------------------------------------------
C1_MIN_MONOTONE_COLS = 2
C2_GOAL_REACH_MID    = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _density_to_count(density: float, size: int) -> int:
    return max(1, int(round(density * size * size)))


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_toward_resource(env) -> int:
    ax, ay = env.agent_x, env.agent_y
    if not env.resources:
        return random.randint(0, env.action_dim - 1)
    best_d = float("inf")
    nearest = None
    for r in env.resources:
        rx, ry = int(r[0]), int(r[1])
        d = abs(ax - rx) + abs(ay - ry)
        if d < best_d:
            best_d = d
            nearest = (rx, ry)
    if nearest is None or best_d == 0:
        return random.randint(0, env.action_dim - 1)
    rx, ry = nearest
    dx, dy = rx - ax, ry - ay
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0
    return 3 if dy > 0 else 2


def _dist_to_nearest_resource(env) -> int:
    if not env.resources:
        return 999
    ax, ay = env.agent_x, env.agent_y
    return min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)


def _get_field(obs_body: torch.Tensor, idx: int, default: float = 0.0) -> float:
    flat = obs_body.flatten()
    if flat.shape[0] > idx:
        return float(flat[idx].item())
    return default


def _get_harm_exposure(obs_body: torch.Tensor) -> float:
    return _get_field(obs_body, 10, 0.0)


def _get_benefit_exposure(obs_body: torch.Tensor) -> float:
    return _get_field(obs_body, 11, 0.0)


def _get_energy(obs_body: torch.Tensor) -> float:
    return _get_field(obs_body, 3, 1.0)


def _update_z_goal(agent: REEAgent, obs_body: torch.Tensor) -> None:
    b_exp = _get_benefit_exposure(obs_body)
    energy = _get_energy(obs_body)
    drive_level = max(0.0, 1.0 - energy)
    agent.update_z_goal(b_exp, drive_level=drive_level)


def _shannon_entropy_bits(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Environment / agent factories
# ---------------------------------------------------------------------------

def _make_env(num_hazards: int, num_resources: int, seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=num_hazards,
        num_resources=num_resources,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        hazard_harm=0.02,
        proximity_harm_scale=0.3,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        env_drift_interval=999,
        env_drift_prob=0.0,
        reef_enabled=REEF_ENABLED,
        n_reef_patches=N_REEF_PATCHES,
        reef_patch_radius=REEF_PATCH_RADIUS,
        hazard_food_attraction=HAZARD_FOOD_ATTRACTION,
    )


def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=16,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=True,
        benefit_weight=0.5,
        z_goal_enabled=True,
        e1_goal_conditioned=True,
        goal_weight=1.0,
        drive_weight=2.0,
        use_resource_proximity_head=True,
        resource_proximity_weight=LAMBDA_RESOURCE,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Warmup: trains E1/E2/harm/benefit/RP heads
# ---------------------------------------------------------------------------

def _warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    n_episodes: int,
    seed: int,
    hd: float,
    rd: float,
    ep_global_start: int,
    total_eps: int,
) -> None:
    device = agent.device
    n_act  = env.action_dim

    e1_params    = list(agent.e1.parameters())
    e2_wf_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    e1_opt      = optim.Adam(e1_params, lr=LR_E1)
    e2_wf_opt   = optim.Adam(e2_wf_params, lr=LR_E2_WF)
    harm_opt    = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_HARM)
    benefit_opt = optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=LR_BENEFIT)

    wf_buf:       List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_pos_buf: List[torch.Tensor] = []
    harm_neg_buf: List[torch.Tensor] = []
    ben_zw_buf:   List[torch.Tensor] = []
    ben_lbl_buf:  List[float]        = []

    random.seed(seed)
    agent.train()

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for _step_i in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)
            rfv       = obs_dict.get("resource_field_view", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            if ticks.get("e1_tick", False):
                _ = agent._e1_tick(latent)

            z_world_curr = latent.z_world.detach()

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev, action_prev, z_world_curr))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_toward_resource(env)
            else:
                action_idx = random.randint(0, n_act - 1)
            action_oh = _onehot(action_idx, n_act, device)
            agent._last_action = action_oh

            dist    = _dist_to_nearest_resource(env)
            is_near = 1.0 if dist <= 2 else 0.0

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            _update_z_goal(agent, obs_dict["body_state"])

            # E1 + SD-018 resource proximity loss
            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if rfv is not None:
                    rfv_t = rfv if isinstance(rfv, torch.Tensor) else torch.tensor(rfv)
                    rp_target = float(rfv_t.max().item())
                    rp_loss   = agent.compute_resource_proximity_loss(rp_target, latent)
                    e1_loss   = e1_loss + LAMBDA_RESOURCE * rp_loss
                if e1_loss.requires_grad:
                    e1_opt.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
                    e1_opt.step()

            # E2 world_forward
            if len(wf_buf) >= BATCH_SIZE:
                idxs  = random.sample(range(len(wf_buf)), min(BATCH_SIZE, len(wf_buf)))
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_opt.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(e2_wf_params, 1.0)
                    e2_wf_opt.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            # E3 harm_eval (stratified)
            if float(harm_signal) < 0:
                harm_pos_buf.append(z_world_curr)
                if len(harm_pos_buf) > MAX_BUF:
                    harm_pos_buf = harm_pos_buf[-MAX_BUF:]
            else:
                harm_neg_buf.append(z_world_curr)
                if len(harm_neg_buf) > MAX_BUF:
                    harm_neg_buf = harm_neg_buf[-MAX_BUF:]

            if len(harm_pos_buf) >= 4 and len(harm_neg_buf) >= 4:
                k_p  = min(BATCH_SIZE // 2, len(harm_pos_buf))
                k_n  = min(BATCH_SIZE // 2, len(harm_neg_buf))
                pi   = torch.randperm(len(harm_pos_buf))[:k_p].tolist()
                ni   = torch.randperm(len(harm_neg_buf))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_pos_buf[i] for i in pi] + [harm_neg_buf[i] for i in ni],
                    dim=0,
                )
                tgt  = torch.cat([
                    torch.ones(k_p,  1, device=device),
                    torch.zeros(k_n, 1, device=device),
                ], dim=0)
                pred  = agent.e3.harm_eval(zw_b)
                hloss = F.binary_cross_entropy(pred, tgt)
                if hloss.requires_grad:
                    harm_opt.zero_grad()
                    hloss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_opt.step()

            # E3 benefit_eval
            ben_zw_buf.append(z_world_curr)
            ben_lbl_buf.append(is_near)
            if len(ben_zw_buf) > MAX_BUF:
                ben_zw_buf  = ben_zw_buf[-MAX_BUF:]
                ben_lbl_buf = ben_lbl_buf[-MAX_BUF:]

            if len(ben_zw_buf) >= 32 and _step_i % 4 == 0:
                k    = min(32, len(ben_zw_buf))
                idxs = random.sample(range(len(ben_zw_buf)), k)
                zw_b = torch.cat([ben_zw_buf[i] for i in idxs], dim=0)
                lbl  = torch.tensor(
                    [ben_lbl_buf[i] for i in idxs],
                    dtype=torch.float32,
                ).unsqueeze(1).to(device)
                pred_b = agent.e3.benefit_eval(zw_b)
                bloss  = F.binary_cross_entropy(pred_b, lbl)
                if bloss.requires_grad:
                    benefit_opt.zero_grad()
                    bloss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.benefit_eval_head.parameters(), 0.5
                    )
                    benefit_opt.step()
                    agent.e3.record_benefit_sample(k)

            z_world_prev = z_world_curr
            action_prev  = action_oh

            if done:
                break

        global_ep = ep_global_start + ep + 1
        if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
            print(
                f"  [train] seed={seed} cond=haz={hd:.2f}_res={rd:.2f}"
                f" ep {global_ep}/{total_eps}  [warmup]",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Eval: frozen policy, collect grid-point metrics
# ---------------------------------------------------------------------------

def _eval_grid_point(
    agent: REEAgent,
    env: CausalGridWorldV2,
    n_episodes: int,
    hd: float,
    rd: float,
    seed: int,
    ep_global_start: int,
    total_eps: int,
) -> Dict:
    device = agent.device
    n_act  = env.action_dim

    agent.eval()

    reached_eps      = 0
    harm_exposures   = []
    benefit_exposures = []
    first_action_counts = [0] * n_act
    time_to_first_list: List[int] = []

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        ep_harm_sum     = 0.0
        ep_benefit_sum  = 0.0
        ep_steps        = 0
        got_resource    = False
        time_to_first   = -1
        first_action_captured = False

        prev_resources = len(env.resources)

        for step_i in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            if ticks.get("e1_tick", False):
                e1_prior = agent._e1_tick(latent)
            else:
                e1_prior = torch.zeros(1, WORLD_DIM, device=device)

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks, temperature=1.0)

            if action is None:
                action = _onehot(random.randint(0, n_act - 1), n_act, device)
                agent._last_action = action

            if not first_action_captured:
                a_idx = int(action.flatten().argmax().item())
                first_action_counts[a_idx] += 1
                first_action_captured = True

            _, harm_signal, done, info, obs_dict = env.step(action)

            with torch.no_grad():
                _update_z_goal(agent, obs_dict["body_state"])

            ep_harm_sum    += _get_harm_exposure(obs_dict["body_state"])
            ep_benefit_sum += _get_benefit_exposure(obs_dict["body_state"])
            ep_steps       += 1

            consumed = False
            if isinstance(info, dict):
                if info.get("resource_consumed", False):
                    consumed = True
            if not consumed:
                cur_resources = len(env.resources)
                if cur_resources < prev_resources:
                    consumed = True
                prev_resources = cur_resources
            if consumed and not got_resource:
                got_resource = True
                time_to_first = step_i + 1

            if done:
                break

        if got_resource:
            reached_eps += 1
            time_to_first_list.append(time_to_first)

        denom = max(1, ep_steps)
        harm_exposures.append(ep_harm_sum / denom)
        benefit_exposures.append(ep_benefit_sum / denom)

        global_ep = ep_global_start + ep + 1
        if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
            print(
                f"  [train] seed={seed} cond=haz={hd:.2f}_res={rd:.2f}"
                f" ep {global_ep}/{total_eps}  [eval]"
                f" reached={reached_eps}"
                f" harm_mean={(sum(harm_exposures)/len(harm_exposures)):.4f}",
                flush=True,
            )

    n_eps = max(1, n_episodes)
    goal_reach_rate     = reached_eps / n_eps
    harm_exposure_mean  = sum(harm_exposures) / n_eps
    benefit_exposure_mean = sum(benefit_exposures) / n_eps
    action_entropy      = _shannon_entropy_bits(first_action_counts)
    time_to_first_mean  = (
        sum(time_to_first_list) / len(time_to_first_list)
        if time_to_first_list else -1.0
    )

    return {
        "goal_reach_rate":         goal_reach_rate,
        "harm_exposure_mean":      harm_exposure_mean,
        "benefit_exposure_mean":   benefit_exposure_mean,
        "action_entropy_bits":     action_entropy,
        "time_to_first_resource":  time_to_first_mean,
        "n_eps":                   n_eps,
        "first_action_counts":     first_action_counts,
    }


# ---------------------------------------------------------------------------
# Per seed / per grid-point driver
# ---------------------------------------------------------------------------

def _run_seed_condition(
    hd: float,
    rd: float,
    seed: int,
    n_warmup: int,
    n_eval: int,
) -> Dict:
    total_eps = n_warmup + n_eval
    num_hazards   = _density_to_count(hd, GRID_SIZE)
    num_resources = _density_to_count(rd, GRID_SIZE)

    # Runner boundary marker -- resets episodes_in_run counter
    print(
        f"Seed {seed} Condition haz={hd:.2f}_res={rd:.2f}",
        flush=True,
    )
    print(
        f"  num_hazards={num_hazards} num_resources={num_resources}"
        f" reef_enabled={REEF_ENABLED} hazard_food_attraction={HAZARD_FOOD_ATTRACTION}",
        flush=True,
    )

    env   = _make_env(num_hazards, num_resources, seed)
    agent = _make_agent(env, seed)

    _warmup(agent, env, n_warmup, seed, hd, rd, ep_global_start=0, total_eps=total_eps)

    env_eval = _make_env(num_hazards, num_resources, seed + 10_000)
    metrics = _eval_grid_point(
        agent, env_eval, n_eval, hd, rd, seed,
        ep_global_start=n_warmup, total_eps=total_eps,
    )

    # verdict per run (PASS = data collected successfully)
    print(
        f"verdict: PASS"
        f"  seed={seed} haz={hd:.2f}_res={rd:.2f}"
        f" goal_reach_rate={metrics['goal_reach_rate']:.3f}"
        f" harm_exp_mean={metrics['harm_exposure_mean']:.4f}"
        f" action_entropy={metrics['action_entropy_bits']:.3f}",
        flush=True,
    )

    return {
        "seed":                    seed,
        "hazard_density":          hd,
        "resource_density":        rd,
        "num_hazards":             num_hazards,
        "num_resources":           num_resources,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Aggregation + criteria
# ---------------------------------------------------------------------------

def _aggregate_and_score(per_seed_results: List[Dict]) -> Dict:
    grid: Dict[Tuple[float, float], List[Dict]] = {}
    for r in per_seed_results:
        key = (r["hazard_density"], r["resource_density"])
        grid.setdefault(key, []).append(r)

    grid_rows: List[Dict] = []
    all_non_null = True
    for (hd, rd), seeds in sorted(grid.items(), key=lambda x: (x[0][1], x[0][0])):
        if not seeds:
            all_non_null = False
            continue
        grr  = sum(s["goal_reach_rate"] for s in seeds) / len(seeds)
        hx   = sum(s["harm_exposure_mean"] for s in seeds) / len(seeds)
        bx   = sum(s["benefit_exposure_mean"] for s in seeds) / len(seeds)
        ent  = sum(s["action_entropy_bits"] for s in seeds) / len(seeds)
        tt_list = [s["time_to_first_resource"] for s in seeds
                   if s["time_to_first_resource"] >= 0]
        ttf = sum(tt_list) / len(tt_list) if tt_list else -1.0

        row = {
            "hazard_density":          hd,
            "resource_density":        rd,
            "num_hazards":             seeds[0]["num_hazards"],
            "num_resources":           seeds[0]["num_resources"],
            "goal_reach_rate":         round(grr, 4),
            "harm_exposure_mean":      round(hx,  6),
            "benefit_exposure_mean":   round(bx,  6),
            "action_entropy_bits":     round(ent, 4),
            "time_to_first_resource":  round(ttf, 3) if ttf >= 0 else -1.0,
            "per_seed": [
                {
                    "seed":                   s["seed"],
                    "goal_reach_rate":        round(s["goal_reach_rate"], 4),
                    "harm_exposure_mean":     round(s["harm_exposure_mean"], 6),
                    "benefit_exposure_mean":  round(s["benefit_exposure_mean"], 6),
                    "action_entropy_bits":    round(s["action_entropy_bits"], 4),
                    "time_to_first_resource": round(s["time_to_first_resource"], 3)
                                              if s["time_to_first_resource"] >= 0
                                              else -1.0,
                }
                for s in seeds
            ],
        }
        grid_rows.append(row)

    # C1 monotonicity
    columns: Dict[float, List[Tuple[float, float]]] = {}
    for row in grid_rows:
        columns.setdefault(row["resource_density"], []).append(
            (row["hazard_density"], row["goal_reach_rate"])
        )
    n_monotone_cols = 0
    col_reports = []
    for rd_key, pairs in sorted(columns.items()):
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        vals = [p[1] for p in pairs_sorted]
        is_monotone = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
        col_reports.append({
            "resource_density":     rd_key,
            "goal_reach_by_hazard": pairs_sorted,
            "strictly_decreasing":  bool(is_monotone),
        })
        if is_monotone:
            n_monotone_cols += 1
    c1 = n_monotone_cols >= C1_MIN_MONOTONE_COLS

    # C2 threshold_identifiable
    any_above = any(r["goal_reach_rate"] >= C2_GOAL_REACH_MID for r in grid_rows)
    any_below = any(r["goal_reach_rate"] <  C2_GOAL_REACH_MID for r in grid_rows)
    c2 = bool(any_above and any_below)

    # C3 coverage
    expected_n = len(HAZARD_DENSITIES) * len(RESOURCE_DENSITIES)
    c3 = bool(all_non_null and len(grid_rows) == expected_n)

    # C4 diagnostic: harm_exposure lower in scarce-resource high-hazard regime
    def _lookup(hd: float, rd: float) -> Optional[Dict]:
        for r in grid_rows:
            if abs(r["hazard_density"] - hd) < 1e-9 and \
               abs(r["resource_density"] - rd) < 1e-9:
                return r
        return None

    avoid_scarce = _lookup(0.25, 0.04)
    avoid_rich   = _lookup(0.25, 0.12)
    c4 = None
    c4_detail: Dict = {}
    if avoid_scarce is not None and avoid_rich is not None:
        c4 = bool(avoid_scarce["harm_exposure_mean"] < avoid_rich["harm_exposure_mean"])
        c4_detail = {
            "harm_exposure_scarce_res04": avoid_scarce["harm_exposure_mean"],
            "harm_exposure_rich_res12":   avoid_rich["harm_exposure_mean"],
        }

    overall_pass = bool(c1 and c2 and c3)

    return {
        "grid_rows":          grid_rows,
        "c1_monotonicity": {
            "pass":               c1,
            "n_monotone_columns": n_monotone_cols,
            "required":           C1_MIN_MONOTONE_COLS,
            "columns":            col_reports,
        },
        "c2_threshold_identifiable": {
            "pass":      c2,
            "any_above": any_above,
            "any_below": any_below,
            "threshold": C2_GOAL_REACH_MID,
        },
        "c3_coverage": {
            "pass":         c3,
            "n_grid_rows":  len(grid_rows),
            "expected":     expected_n,
            "all_non_null": all_non_null,
        },
        "c4_avoidance_signature": {
            "pass":   c4,
            "detail": c4_detail,
        },
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        hazard_densities   = [0.05, 0.25]
        resource_densities = [0.08]
        seeds              = [42]
        n_warmup           = 2
        n_eval             = 2
    else:
        hazard_densities   = HAZARD_DENSITIES
        resource_densities = RESOURCE_DENSITIES
        seeds              = SEEDS
        n_warmup           = WARMUP_EPISODES
        n_eval             = EVAL_EPISODES

    total_conditions = len(hazard_densities) * len(resource_densities)
    total_runs       = total_conditions * len(seeds)

    print(
        f"[V3-EXQ-526] Q-034 Reef Threshold Sweep"
        f" dry_run={args.dry_run}"
        f" seeds={seeds}"
        f" warmup={n_warmup} eval={n_eval} steps={STEPS_PER_EP}"
        f" grid={GRID_SIZE}x{GRID_SIZE}"
        f" reef_enabled={REEF_ENABLED}"
        f" hazard_food_attraction={HAZARD_FOOD_ATTRACTION}",
        flush=True,
    )
    print(
        f"  Grid: {len(hazard_densities)} x {len(resource_densities)}"
        f" = {total_conditions} grid points x {len(seeds)} seeds"
        f" = {total_runs} total runs",
        flush=True,
    )

    per_seed_results: List[Dict] = []

    run_idx = 0
    for seed in seeds:
        for rd in resource_densities:
            for hd in hazard_densities:
                run_idx += 1
                print(
                    f"\n[V3-EXQ-526] ({run_idx}/{total_runs}) starting"
                    f" seed={seed} haz={hd:.2f} res={rd:.2f}",
                    flush=True,
                )
                try:
                    result = _run_seed_condition(hd, rd, seed, n_warmup, n_eval)
                    per_seed_results.append(result)
                except Exception as exc:
                    print(f"verdict: FAIL  error={exc}", flush=True)

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    agg = _aggregate_and_score(per_seed_results)
    outcome = "PASS" if agg["overall_pass"] else "FAIL"
    evidence_direction = "supports" if outcome == "PASS" else "weakens"

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"\n[V3-EXQ-526] === Results ===", flush=True)
    print(
        f"  {'hd':>5} {'rd':>5} {'grr':>6} {'harm':>7} {'benefit':>8}"
        f" {'entropy':>8} {'ttfr':>7}",
        flush=True,
    )
    for r in agg["grid_rows"]:
        print(
            f"  {r['hazard_density']:>5.2f} {r['resource_density']:>5.2f}"
            f" {r['goal_reach_rate']:>6.3f}"
            f" {r['harm_exposure_mean']:>7.4f}"
            f" {r['benefit_exposure_mean']:>8.4f}"
            f" {r['action_entropy_bits']:>8.3f}"
            f" {r['time_to_first_resource']:>7.2f}",
            flush=True,
        )
    print(
        f"\n  C1 monotonicity: {agg['c1_monotonicity']['pass']}"
        f" ({agg['c1_monotonicity']['n_monotone_columns']}"
        f"/{len(resource_densities)} columns)",
        flush=True,
    )
    print(
        f"  C2 threshold_identifiable: {agg['c2_threshold_identifiable']['pass']}",
        flush=True,
    )
    print(
        f"  C3 coverage: {agg['c3_coverage']['pass']}"
        f" ({agg['c3_coverage']['n_grid_rows']}"
        f"/{agg['c3_coverage']['expected']} grid points)",
        flush=True,
    )
    print(
        f"  C4 avoidance_signature (diagnostic): {agg['c4_avoidance_signature']['pass']}",
        flush=True,
    )
    print(f"\n  -> {outcome} (evidence_direction={evidence_direction})", flush=True)

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    # -----------------------------------------------------------------------
    # Write manifest
    # -----------------------------------------------------------------------
    ts_utc  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts_utc}_v3.json"

    manifest = {
        "run_id":                    f"{EXPERIMENT_TYPE}_{ts_utc}_v3",
        "experiment_type":           EXPERIMENT_TYPE,
        "architecture_epoch":        "ree_hybrid_guardrails_v1",
        "claim_ids":                 CLAIM_IDS,
        "experiment_purpose":        EXPERIMENT_PURPOSE,
        "supersedes":                SUPERSEDES_EXQ,
        "outcome":                   outcome,
        "evidence_direction":        evidence_direction,
        "evidence_direction_per_claim": {"Q-034": evidence_direction},
        "timestamp_utc":             ts_utc,

        # Sweep parameters
        "hazard_densities":          HAZARD_DENSITIES,
        "resource_densities":        RESOURCE_DENSITIES,
        "seeds":                     SEEDS,
        "grid_size":                 GRID_SIZE,
        "warmup_episodes":           WARMUP_EPISODES,
        "eval_episodes":             EVAL_EPISODES,
        "steps_per_ep":              STEPS_PER_EP,

        # Substrate flags
        "reef_enabled":              REEF_ENABLED,
        "n_reef_patches":            N_REEF_PATCHES,
        "reef_patch_radius":         REEF_PATCH_RADIUS,
        "hazard_food_attraction":    HAZARD_FOOD_ATTRACTION,
        "use_resource_proximity_head": True,
        "drive_weight":              2.0,
        "resource_proximity_weight": LAMBDA_RESOURCE,

        # Acceptance thresholds
        "c1_min_monotone_cols":      C1_MIN_MONOTONE_COLS,
        "c2_goal_reach_threshold":   C2_GOAL_REACH_MID,

        # Results
        "grid_rows":                 agg["grid_rows"],
        "c1_monotonicity":           agg["c1_monotonicity"],
        "c2_threshold_identifiable": agg["c2_threshold_identifiable"],
        "c3_coverage":               agg["c3_coverage"],
        "c4_avoidance_signature":    agg["c4_avoidance_signature"],
        "overall_pass":              agg["overall_pass"],

        # Summary
        "n_grid_points":             len(agg["grid_rows"]),
        "n_total_runs":              len(per_seed_results),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-526] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
