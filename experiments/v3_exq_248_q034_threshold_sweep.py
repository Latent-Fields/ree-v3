#!/opt/local/bin/python3
"""
V3-EXQ-248 -- Q-034: Hazard/Resource Ratio Threshold Sweep

Claims: Q-034
EXPERIMENT_PURPOSE = "evidence"

=== SCIENTIFIC QUESTION ===

At what hazard/resource ratio does the depression attractor form?
EXQ-237a showed PLANNED > HABIT in the LONG_HORIZON condition
(1 resource, 3 hazards). This sweep characterizes the boundary:
for which (hazard_count, resource_count) cells is
lift = PLANNED.resource_rate - HABIT.resource_rate < 0.02?

Key deliverable: threshold estimate (which cells show
behavioral equivalence between HABIT and PLANNED) rather than
just PASS/FAIL.

=== DESIGN ===

3x3 grid sweep: hazard_counts x resource_counts.
  hazard_counts  = [1, 2, 3]  (low -> high hazard pressure)
  resource_counts = [3, 2, 1]  (high -> low resource availability)
  → 9 cells total

3 seeds per cell. Both HABIT and PLANNED mode per seed.
  HABIT:   z_goal_enabled=False, drive_weight=0
  PLANNED: z_goal_enabled=True, e1_goal_conditioned=True, drive_weight=2

Key metric per cell: lift = PLANNED.resource_rate - HABIT.resource_rate
When lift < 0.02 in >=2/3 seeds: modes behaviorally equivalent.
This is the "depression attractor" region.

=== PRE-REGISTERED CRITERIA ===

PASS: At least 3 of the 9 cells have majority (>=2/3 seeds) lift < 0.02,
      AND these cells form a contiguous high-hazard/low-resource region,
      indicating a threshold exists.

FAIL: Either fewer than 3 cells show behavioral equivalence,
      or the cells do not form a coherent high-hazard/low-resource region.

Per-cell output: mean_lift, std_lift, z_goal_norm (PLANNED),
                 z_harm_a_mean (PLANNED), harm_rate (both modes),
                 pass_fraction_lt002 (fraction of seeds with lift < 0.02)

=== WARMUP ===

200 eps, 40% greedy toward nearest resource.
Trains: E1 (prediction), E2 world_forward, E3 harm_eval (stratified),
        E3 benefit_eval (proximity labels).
PLANNED additionally updates z_goal each step.

=== EVAL ===

100 eps. Full agent pipeline: generate_trajectories + select_action.
PLANNED continues z_goal updates (no gradients).
"""

import sys
import random
import json
import time
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
EXPERIMENT_TYPE    = "v3_exq_248_q034_threshold_sweep"
CLAIM_IDS          = ["Q-034"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
LIFT_EQUIV_THRESH = 0.02   # lift < this => behavioral equivalence (depression attractor)
MAJORITY_THRESH   = 2      # >= 2/3 seeds
MIN_EQUIV_CELLS   = 3      # at least this many cells must show equivalence for PASS

# ---------------------------------------------------------------------------
# Grid and episode parameters
# ---------------------------------------------------------------------------
GRID_SIZE       = 8
STEPS_PER_EP    = 150      # LONG_HORIZON base (same as EXQ-237a LONG_HORIZON)
HAZARD_COUNTS   = [1, 2, 3]
RESOURCE_COUNTS = [3, 2, 1]

# ---------------------------------------------------------------------------
# Training parameters (matching EXQ-237a LONG_HORIZON)
# ---------------------------------------------------------------------------
WARMUP_EPISODES = 200
EVAL_EPISODES   = 100
SEEDS           = [42, 7, 13]
GREEDY_FRAC     = 0.4
MAX_BUF         = 4000
WF_BUF_MAX      = 2000
WORLD_DIM       = 32
BATCH_SIZE      = 16

# Learning rates
LR_E1      = 1e-3
LR_E2_WF   = 1e-3
LR_HARM    = 1e-4
LR_BENEFIT = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_toward_resource(env) -> int:
    """Greedy action: move toward nearest resource (Manhattan)."""
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


def _get_benefit_exposure(obs_body: torch.Tensor) -> float:
    """Extract benefit_exposure from body_state obs (index 11 in proxy mode)."""
    flat = obs_body.flatten()
    if flat.shape[0] > 11:
        return float(flat[11].item())
    return 0.0


def _get_energy(obs_body: torch.Tensor) -> float:
    """Extract energy from body_state obs (index 3)."""
    flat = obs_body.flatten()
    if flat.shape[0] > 3:
        return float(flat[3].item())
    return 1.0


def _update_z_goal(agent: REEAgent, obs_body: torch.Tensor) -> None:
    """Update z_goal from current step's benefit_exposure and drive_level."""
    b_exp = _get_benefit_exposure(obs_body)
    energy = _get_energy(obs_body)
    drive_level = max(0.0, 1.0 - energy)
    agent.update_z_goal(b_exp, drive_level=drive_level)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(n_hazards: int, n_resources: int, seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=n_hazards,
        num_resources=n_resources,
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
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(condition: str, env: CausalGridWorldV2, seed: int) -> REEAgent:
    planned = (condition == "PLANNED")
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
        z_goal_enabled=planned,
        e1_goal_conditioned=planned,
        goal_weight=1.0 if planned else 0.0,
        drive_weight=2.0 if planned else 0.0,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    condition: str,
    warmup_episodes: int,
    steps_per_episode: int,
    seed: int,
    cell_label: str,
) -> Dict:
    """
    Mixed-policy warmup: 40% greedy, 60% random.

    Trains: E1 (prediction loss), E2 world_forward (explicit buffer),
    E3 harm_eval (stratified), E3 benefit_eval (proximity labels).
    PLANNED: additionally updates z_goal each step.
    """
    planned = (condition == "PLANNED")
    device  = agent.device
    n_act   = env.action_dim

    # Separate optimisers
    e1_params    = list(agent.e1.parameters())
    e2_wf_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    e1_opt      = optim.Adam(e1_params, lr=LR_E1)
    e2_wf_opt   = optim.Adam(e2_wf_params, lr=LR_E2_WF)
    harm_opt    = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_HARM)
    benefit_opt = optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=LR_BENEFIT)

    # Experience buffers
    wf_buf:       List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_pos_buf: List[torch.Tensor] = []
    harm_neg_buf: List[torch.Tensor] = []
    ben_zw_buf:   List[torch.Tensor] = []
    ben_lbl_buf:  List[float]        = []

    random.seed(seed)
    agent.train()

    total_eps = warmup_episodes + EVAL_EPISODES  # denominator for progress prints

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for step_i in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            # Sense
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            # E1 tick
            if ticks.get("e1_tick", False):
                _ = agent._e1_tick(latent)

            z_world_curr = latent.z_world.detach()

            # E2 world_forward buffer
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev, action_prev, z_world_curr))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            # Mixed warmup policy
            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_toward_resource(env)
            else:
                action_idx = random.randint(0, n_act - 1)
            action_oh = _onehot(action_idx, n_act, device)
            agent._last_action = action_oh

            # Benefit proximity label (before step)
            dist   = _dist_to_nearest_resource(env)
            is_near = 1.0 if dist <= 2 else 0.0

            # Env step
            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            # z_goal update (PLANNED only)
            if planned:
                _update_z_goal(agent, obs_dict["body_state"])

            # --- Train E1 ---
            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_opt.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
                    e1_opt.step()

            # --- Train E2 world_forward ---
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

            # --- Train E3 harm_eval (stratified) ---
            if float(harm_signal) < 0:
                harm_pos_buf.append(z_world_curr)
                if len(harm_pos_buf) > MAX_BUF:
                    harm_pos_buf = harm_pos_buf[-MAX_BUF:]
            else:
                harm_neg_buf.append(z_world_curr)
                if len(harm_neg_buf) > MAX_BUF:
                    harm_neg_buf = harm_neg_buf[-MAX_BUF:]

            if len(harm_pos_buf) >= 4 and len(harm_neg_buf) >= 4:
                k_p = min(BATCH_SIZE // 2, len(harm_pos_buf))
                k_n = min(BATCH_SIZE // 2, len(harm_neg_buf))
                pi  = torch.randperm(len(harm_pos_buf))[:k_p].tolist()
                ni  = torch.randperm(len(harm_neg_buf))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_pos_buf[i] for i in pi] + [harm_neg_buf[i] for i in ni],
                    dim=0,
                )
                tgt = torch.cat([
                    torch.ones(k_p,  1, device=device),
                    torch.zeros(k_n, 1, device=device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                hloss = F.binary_cross_entropy(pred, tgt)
                if hloss.requires_grad:
                    harm_opt.zero_grad()
                    hloss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_opt.step()

            # --- Train E3 benefit_eval (proximity labels) ---
            ben_zw_buf.append(z_world_curr)
            ben_lbl_buf.append(is_near)
            if len(ben_zw_buf) > MAX_BUF:
                ben_zw_buf  = ben_zw_buf[-MAX_BUF:]
                ben_lbl_buf = ben_lbl_buf[-MAX_BUF:]

            if len(ben_zw_buf) >= 32 and step_i % 4 == 0:
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

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            diag = agent.compute_goal_maintenance_diagnostic()
            print(
                f"    [train] {cell_label} seed={seed} ep {ep+1}/{total_eps}"
                f" harm_pos={len(harm_pos_buf)}"
                f" harm_neg={len(harm_neg_buf)}"
                f" ben_samples={agent.e3._benefit_samples_seen}"
                f" goal_norm={diag['goal_norm']:.3f}",
                flush=True,
            )

    diag_final = agent.compute_goal_maintenance_diagnostic()
    return {"goal_norm": float(diag_final["goal_norm"])}


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def _eval(
    agent: REEAgent,
    env: CausalGridWorldV2,
    condition: str,
    eval_episodes: int,
    steps_per_episode: int,
    warmup_done: int,
    total_eps: int,
    cell_label: str,
    seed: int,
) -> Dict:
    """
    Eval using the FULL agent pipeline.
    Also collects z_harm_a_mean for PLANNED (SD-011 affective stream).
    """
    planned = (condition == "PLANNED")
    agent.eval()

    device    = agent.device
    world_dim = WORLD_DIM
    n_act     = env.action_dim

    resource_counts:  List[int]   = []
    harm_rates:       List[float] = []
    z_harm_a_norms:   List[float] = []

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        ep_resources = 0
        ep_harm_sum  = 0.0
        ep_steps     = 0
        ep_z_harm_a  = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            # E1 tick
            if ticks.get("e1_tick", False):
                e1_prior = agent._e1_tick(latent)
            else:
                e1_prior = torch.zeros(1, world_dim, device=device)

            # Full hippocampal planning
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks, temperature=1.0)

            if action is None:
                action = _onehot(
                    random.randint(0, n_act - 1), n_act, device
                )
                agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)

            if info.get("transition_type") == "resource":
                ep_resources += 1
            if float(harm_signal) < 0:
                ep_harm_sum += abs(float(harm_signal))
            ep_steps += 1

            # z_harm_a norm (SD-011 affective stream)
            if hasattr(latent, "z_harm_a") and latent.z_harm_a is not None:
                ep_z_harm_a += float(latent.z_harm_a.norm().item())

            # z_goal maintenance (PLANNED only, no gradients)
            if planned:
                with torch.no_grad():
                    _update_z_goal(agent, obs_dict["body_state"])

            if done:
                break

        resource_counts.append(1 if ep_resources >= 1 else 0)
        harm_rates.append(ep_harm_sum / max(1, ep_steps))
        z_harm_a_norms.append(ep_z_harm_a / max(1, ep_steps))

        if (ep + 1) % 50 == 0 or ep == eval_episodes - 1:
            ep_global = warmup_done + ep + 1
            print(
                f"    [train] {cell_label} seed={seed} ep {ep_global}/{total_eps}"
                f" eval_ep={ep+1}/{eval_episodes}"
                f" cond={condition}",
                flush=True,
            )

    resource_rate  = sum(resource_counts) / max(1, len(resource_counts))
    harm_rate      = sum(harm_rates) / max(1, len(harm_rates))
    z_harm_a_mean  = sum(z_harm_a_norms) / max(1, len(z_harm_a_norms))

    return {
        "resource_rate": resource_rate,
        "harm_rate":     harm_rate,
        "z_harm_a_mean": z_harm_a_mean,
    }


# ---------------------------------------------------------------------------
# Per-seed, per-cell run
# ---------------------------------------------------------------------------

def _run_cell_seed(
    n_hazards:       int,
    n_resources:     int,
    seed:            int,
    warmup_episodes: int,
    eval_episodes:   int,
    steps_per_ep:    int,
) -> Dict:
    """Run HABIT and PLANNED for one (cell, seed) combination."""
    cell_label = f"h{n_hazards}_r{n_resources}"
    total_eps  = warmup_episodes + eval_episodes

    results: Dict = {}

    for condition in ["HABIT", "PLANNED"]:
        print(
            f"\n[V3-EXQ-248] Seed {seed} Condition {cell_label} Mode {condition}",
            flush=True,
        )

        env   = _make_env(n_hazards, n_resources, seed)
        agent = _make_agent(condition, env, seed)

        warmup_res = _warmup(
            agent, env, condition,
            warmup_episodes, steps_per_ep, seed, cell_label,
        )

        eval_res = _eval(
            agent, env, condition,
            eval_episodes, steps_per_ep,
            warmup_done=warmup_episodes,
            total_eps=total_eps,
            cell_label=cell_label,
            seed=seed,
        )

        print(
            f"  [eval done] {cell_label} seed={seed} cond={condition}"
            f" resource_rate={eval_res['resource_rate']:.3f}"
            f" harm_rate={eval_res['harm_rate']:.5f}"
            f" goal_norm={warmup_res['goal_norm']:.3f}"
            f" z_harm_a={eval_res['z_harm_a_mean']:.4f}",
            flush=True,
        )

        results[condition] = {
            "resource_rate": eval_res["resource_rate"],
            "harm_rate":     eval_res["harm_rate"],
            "z_harm_a_mean": eval_res["z_harm_a_mean"],
            "goal_norm":     warmup_res["goal_norm"],
        }

        lift = (results["PLANNED"]["resource_rate"] - results["HABIT"]["resource_rate"]
                if "PLANNED" in results and "HABIT" in results else float("nan"))
        if "PLANNED" in results and "HABIT" in results:
            seed_pass = lift < LIFT_EQUIV_THRESH
            print(
                f"  verdict: {'PASS' if seed_pass else 'FAIL'}"
                f" (lift={lift:+.4f}"
                f" thresh={LIFT_EQUIV_THRESH})",
                flush=True,
            )

    return results


# ---------------------------------------------------------------------------
# Aggregate per cell
# ---------------------------------------------------------------------------

def _aggregate_cell(
    n_hazards:   int,
    n_resources: int,
    seed_results: List[Dict],
) -> Dict:
    """Compute per-cell summary across seeds."""
    lifts = []
    z_goal_norms  = []
    z_harm_a_vals = []
    harm_rates_habit   = []
    harm_rates_planned = []

    for sr in seed_results:
        habit   = sr["HABIT"]
        planned = sr["PLANNED"]
        lift = planned["resource_rate"] - habit["resource_rate"]
        lifts.append(lift)
        z_goal_norms.append(planned["goal_norm"])
        z_harm_a_vals.append(planned["z_harm_a_mean"])
        harm_rates_habit.append(habit["harm_rate"])
        harm_rates_planned.append(planned["harm_rate"])

    n = len(lifts)
    mean_lift = sum(lifts) / max(1, n)
    std_lift  = (
        (sum((l - mean_lift) ** 2 for l in lifts) / max(1, n)) ** 0.5
    )
    pass_fraction = sum(1 for l in lifts if l < LIFT_EQUIV_THRESH) / max(1, n)
    majority_equiv = sum(1 for l in lifts if l < LIFT_EQUIV_THRESH) >= MAJORITY_THRESH

    return {
        "n_hazards":        n_hazards,
        "n_resources":      n_resources,
        "mean_lift":        mean_lift,
        "std_lift":         std_lift,
        "lifts":            lifts,
        "z_goal_norm":      sum(z_goal_norms) / max(1, n),
        "z_harm_a_mean":    sum(z_harm_a_vals) / max(1, n),
        "harm_rate_habit":  sum(harm_rates_habit) / max(1, n),
        "harm_rate_planned":sum(harm_rates_planned) / max(1, n),
        "pass_fraction_lt002": pass_fraction,
        "majority_equiv":   majority_equiv,
    }


# ---------------------------------------------------------------------------
# Decide outcome
# ---------------------------------------------------------------------------

def _decide(per_cell_results: Dict[str, Dict]) -> Tuple[str, str, str, bool, str]:
    """
    Returns (outcome, evidence_direction, decision, threshold_identified,
             threshold_region).
    PASS criterion: >=3 cells show majority equivalence AND they form
    a contiguous high-hazard/low-resource region.
    """
    equiv_cells = [k for k, v in per_cell_results.items() if v["majority_equiv"]]
    n_equiv = len(equiv_cells)

    # Check if equiv cells cluster in high-hazard, low-resource corner
    threshold_identified = False
    threshold_region     = "none"

    if n_equiv >= MIN_EQUIV_CELLS:
        # Check contiguity in high-hazard / low-resource direction
        high_hazard_cells = [k for k in equiv_cells if per_cell_results[k]["n_hazards"] >= 2]
        low_resource_cells = [k for k in equiv_cells if per_cell_results[k]["n_resources"] <= 2]
        corner_cells = [k for k in equiv_cells
                        if per_cell_results[k]["n_hazards"] >= 2
                        and per_cell_results[k]["n_resources"] <= 2]

        if len(corner_cells) >= 2:
            threshold_identified = True
            hazards_thresh   = min(per_cell_results[k]["n_hazards"] for k in equiv_cells)
            resources_thresh = max(per_cell_results[k]["n_resources"] for k in equiv_cells)
            threshold_region = f"h>={hazards_thresh} and r<={resources_thresh}"

        outcome   = "PASS" if threshold_identified else "FAIL"
        direction = "supports" if threshold_identified else "does_not_support"
        decision  = "retain_ree" if threshold_identified else "inconclusive"
    else:
        outcome   = "FAIL"
        direction = "does_not_support"
        decision  = "inconclusive"

    return outcome, direction, decision, threshold_identified, threshold_region


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    warmup    = 5    if args.dry_run else WARMUP_EPISODES
    n_eval    = 5    if args.dry_run else EVAL_EPISODES
    seeds     = [42] if args.dry_run else SEEDS
    h_counts  = [1]  if args.dry_run else HAZARD_COUNTS
    r_counts  = [1]  if args.dry_run else RESOURCE_COUNTS

    print(
        f"[V3-EXQ-248] Q-034 Hazard/Resource Ratio Threshold Sweep"
        f" dry_run={args.dry_run}"
        f" warmup={warmup} eval={n_eval} seeds={seeds}"
        f" hazard_counts={h_counts} resource_counts={r_counts}",
        flush=True,
    )
    print(
        f"  Grid: {GRID_SIZE}x{GRID_SIZE}, {STEPS_PER_EP} steps/ep"
        f" hazard_harm=0.02 resource_respawn=True",
        flush=True,
    )

    # Collect results: cell_label -> list of seed dicts
    cell_seed_results: Dict[str, List[Dict]] = {}

    for n_hazards in h_counts:
        for n_resources in r_counts:
            cell_label = f"h{n_hazards}_r{n_resources}"
            cell_seed_results[cell_label] = []

            for seed in seeds:
                print(
                    f"\n[V3-EXQ-248] === Cell {cell_label} seed={seed} ===",
                    flush=True,
                )
                sr = _run_cell_seed(
                    n_hazards=n_hazards,
                    n_resources=n_resources,
                    seed=seed,
                    warmup_episodes=warmup,
                    eval_episodes=n_eval,
                    steps_per_ep=STEPS_PER_EP,
                )
                cell_seed_results[cell_label].append(sr)

    # Aggregate per cell
    per_cell_agg: Dict[str, Dict] = {}
    for n_hazards in h_counts:
        for n_resources in r_counts:
            cell_label = f"h{n_hazards}_r{n_resources}"
            per_cell_agg[cell_label] = _aggregate_cell(
                n_hazards, n_resources, cell_seed_results[cell_label]
            )

    outcome, direction, decision, threshold_identified, threshold_region = _decide(
        per_cell_agg
    )

    # Print summary
    print(f"\n[V3-EXQ-248] === Results ===", flush=True)
    for cell_label, cell in sorted(per_cell_agg.items()):
        print(
            f"  {cell_label}: lift={cell['mean_lift']:+.4f}"
            f" +/-{cell['std_lift']:.4f}"
            f" equiv_frac={cell['pass_fraction_lt002']:.2f}"
            f" z_goal={cell['z_goal_norm']:.3f}"
            f" z_harm_a={cell['z_harm_a_mean']:.4f}"
            f" {'[EQUIV]' if cell['majority_equiv'] else '[LIFT]'}",
            flush=True,
        )
    print(
        f"  threshold_identified={threshold_identified}"
        f" threshold_region={threshold_region}",
        flush=True,
    )
    print(
        f"  -> {outcome} decision={decision} direction={direction}",
        flush=True,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    # Write output
    ts      = int(time.time())
    ts_str  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    # Build serialisable per_cell_results
    per_cell_serializable: Dict = {}
    for cell_label, cell in per_cell_agg.items():
        per_cell_serializable[cell_label] = {
            "n_hazards":            cell["n_hazards"],
            "n_resources":          cell["n_resources"],
            "mean_lift":            float(cell["mean_lift"]),
            "std_lift":             float(cell["std_lift"]),
            "lifts":                [float(l) for l in cell["lifts"]],
            "z_goal_norm":          float(cell["z_goal_norm"]),
            "z_harm_a_mean":        float(cell["z_harm_a_mean"]),
            "harm_rate_habit":      float(cell["harm_rate_habit"]),
            "harm_rate_planned":    float(cell["harm_rate_planned"]),
            "pass_fraction_lt002":  float(cell["pass_fraction_lt002"]),
            "majority_equiv":       bool(cell["majority_equiv"]),
        }

    manifest = {
        "run_id":             f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":    EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome":            outcome,
        "evidence_direction": direction,
        "decision":           decision,
        "timestamp_utc":      ts_str,
        "seeds":              seeds,
        # Parameters
        "warmup_episodes":       warmup,
        "eval_episodes":         n_eval,
        "steps_per_ep":          STEPS_PER_EP,
        "grid_size":             GRID_SIZE,
        "hazard_counts":         h_counts,
        "resource_counts":       r_counts,
        "hazard_harm":           0.02,
        "greedy_frac":           GREEDY_FRAC,
        # Thresholds
        "lift_equiv_thresh":     LIFT_EQUIV_THRESH,
        "majority_thresh":       MAJORITY_THRESH,
        "min_equiv_cells":       MIN_EQUIV_CELLS,
        # Key outputs
        "per_cell_results":      per_cell_serializable,
        "threshold_identified":  threshold_identified,
        "threshold_region":      threshold_region,
        "n_equiv_cells":         sum(1 for v in per_cell_agg.values() if v["majority_equiv"]),
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-248] Output written to: {out_path}", flush=True)
    print(f"[V3-EXQ-248] run_id: {manifest['run_id']}", flush=True)


if __name__ == "__main__":
    main()
