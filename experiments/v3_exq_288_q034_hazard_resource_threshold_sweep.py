#!/opt/local/bin/python3
"""
V3-EXQ-288 -- Q-034: Hazard/Resource Ratio Threshold Sweep (3x3 grid, 3 seeds)

Claims: Q-034
EXPERIMENT_PURPOSE = "evidence"

=== SCIENTIFIC QUESTION ===

Q-034 asks: above what hazard/resource ratio does behavioral equivalence
(|HABIT_harm_rate - PLANNED_harm_rate| < 0.02) break down? That is, at
what density ratio does the PLANNED system's advantage over HABIT become
detectable in harm avoidance?

EXQ-237a established the PLANNED advantage exists at extreme LONG_HORIZON
parameters. This sweep characterizes the boundary: the threshold ratio above
which the two systems diverge, and the ratio below which they converge
(HABIT sufficient, PLANNED adds no harm-avoidance benefit).

=== DESIGN ===

3x3 grid:
  hazard_rate in [0.1, 0.3, 0.5]  -> num_hazards in [1, 2, 4]  (on 8x8 grid)
  resource_rate in [0.1, 0.3, 0.5] -> num_resources in [1, 2, 4] (on 8x8 grid)

  Mapping: rate * GRID_SIZE = count (round to int, minimum 1).
  This gives hazard/resource ratios:
    0.1/0.5 = 0.2   (1 hazard, 4 resources -- resource-rich)
    0.1/0.3 = 0.33  (1 hazard, 2 resources)
    0.1/0.1 = 1.0   (1 hazard, 1 resource)
    0.3/0.5 = 0.6   (2 hazards, 4 resources)
    0.3/0.3 = 1.0   (2 hazards, 2 resources)
    0.3/0.1 = 3.0   (2 hazards, 1 resource -- hazard-heavy)
    0.5/0.5 = 1.0   (4 hazards, 4 resources)
    0.5/0.3 = 1.67  (4 hazards, 2 resources)
    0.5/0.1 = 5.0   (4 hazards, 1 resource -- extreme)

  3 seeds per cell, each seed run in both HABIT and PLANNED modes.

=== HABIT vs PLANNED ===

HABIT:   z_goal_enabled=False, drive_weight=0, benefit_eval_enabled=True.
         Value-flat HippocampalModule CEM proposals + harm_eval - benefit_eval.
         No goal attractor; no E1 goal-conditioning.

PLANNED: z_goal_enabled=True, e1_goal_conditioned=True, drive_weight=2.0.
         z_goal attractor seeded from benefit contacts, maintained by E1.
         E3 additionally subtracts goal_proximity from trajectory cost.

Both conditions: 200 training eps (40% greedy warmup), then 50 eval eps.
use_resource_proximity_head=True, SD-018 resource proximity supervision.

=== CRITERION ===

Per cell, per seed: gap = |HABIT_harm_rate - PLANNED_harm_rate|
  seeds_equivalent = count(gap < 0.02)  (out of 3)
  behavioral_equivalence_score = mean(gap)

Summary curve: sort cells by ratio. Identify first ratio above which
  seeds_equivalent >= 2/3 in majority (>= 5/9) of cells with ratio >= threshold.

PASS: transition_found = True (clear threshold ratio identified).
      At least 3 cells with ratio >= threshold have seeds_equivalent >= 2 AND
      at least 2 cells with ratio < threshold have seeds_equivalent < 2.

FAIL: no clear transition found.

Key deliverable: threshold_estimate + ratio_to_gap_curve (grid_results sorted by ratio).
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
EXPERIMENT_TYPE    = "v3_exq_288_q034_hazard_resource_threshold_sweep"
CLAIM_IDS          = ["Q-034"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
HAZARD_RATES   = [0.1, 0.3, 0.5]    # -> num_hazards = round(rate * GRID_SIZE), min 1
RESOURCE_RATES = [0.1, 0.3, 0.5]    # -> num_resources = round(rate * GRID_SIZE), min 1
SEEDS          = [42, 7, 13]

GRID_SIZE      = 8

# ---------------------------------------------------------------------------
# Training / eval parameters
# ---------------------------------------------------------------------------
TRAINING_EPISODES = 200
EVAL_EPISODES     = 50
STEPS_PER_EP      = 150
GREEDY_FRAC       = 0.4

MAX_BUF           = 4000
WF_BUF_MAX        = 2000
WORLD_DIM         = 32
BATCH_SIZE        = 16

LR_E1           = 1e-3
LR_E2_WF        = 1e-3
LR_HARM         = 1e-4
LR_BENEFIT      = 1e-3
LAMBDA_RESOURCE = 0.5

# ---------------------------------------------------------------------------
# Pass/fail threshold
# ---------------------------------------------------------------------------
EQUIVALENCE_GAP   = 0.02   # |HABIT_harm_rate - PLANNED_harm_rate| < this -> equivalent
MAJORITY_SEEDS    = 2      # seeds_equivalent >= this -> cell is "equivalent"
# Transition: at least this many high-ratio cells equivalent AND low-ratio cells not
MIN_HIGH_RATIO_EQUIV   = 3
MIN_LOW_RATIO_NOT_EQUIV = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rate_to_count(rate: float) -> int:
    """Map a density rate [0.1, 0.3, 0.5] to num objects on GRID_SIZE grid."""
    return max(1, round(rate * GRID_SIZE))


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_toward_resource(env) -> int:
    """Greedy action: move toward nearest resource (Manhattan distance)."""
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
    """Update z_goal from current step benefit_exposure and drive_level."""
    b_exp = _get_benefit_exposure(obs_body)
    energy = _get_energy(obs_body)
    drive_level = max(0.0, 1.0 - energy)
    agent.update_z_goal(b_exp, drive_level=drive_level)


# ---------------------------------------------------------------------------
# Environment factory
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
        use_resource_proximity_head=True,
        resource_proximity_weight=LAMBDA_RESOURCE,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Training (warmup)
# ---------------------------------------------------------------------------

def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    condition: str,
    n_episodes: int,
    seed: int,
    hz_rate: float,
    res_rate: float,
) -> Dict:
    """Mixed-policy training: 40% greedy, 60% random."""
    planned = (condition == "PLANNED")
    device  = agent.device
    n_act   = env.action_dim

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

        for step_i in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            rfv = obs_dict.get("resource_field_view", None)

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

            if planned:
                _update_z_goal(agent, obs_dict["body_state"])

            # Train E1 + SD-018 resource proximity
            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if rfv is not None:
                    rp_target = rfv[12].item()
                    rp_loss   = agent.compute_resource_proximity_loss(rp_target, latent)
                    e1_loss   = e1_loss + LAMBDA_RESOURCE * rp_loss
                if e1_loss.requires_grad:
                    e1_opt.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
                    e1_opt.step()

            # Train E2 world_forward
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

            # Train E3 harm_eval (stratified)
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

            # Train E3 benefit_eval (proximity labels)
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

        if (ep + 1) % 50 == 0 or ep == n_episodes - 1:
            print(
                f"    [train] condition={condition}"
                f" condition={hz_rate:.1f}x{res_rate:.1f}"
                f" seed={seed}"
                f" ep {ep+1}/{n_episodes}"
                f" harm_pos={len(harm_pos_buf)}"
                f" harm_neg={len(harm_neg_buf)}",
                flush=True,
            )

    diag = agent.compute_goal_maintenance_diagnostic()
    return {"goal_norm": float(diag["goal_norm"])}


# ---------------------------------------------------------------------------
# Eval (full agent pipeline)
# ---------------------------------------------------------------------------

def _eval(
    agent: REEAgent,
    env: CausalGridWorldV2,
    condition: str,
    n_episodes: int,
) -> Dict:
    """Eval using full agent pipeline: generate_trajectories + select_action."""
    planned = (condition == "PLANNED")
    agent.eval()

    device  = agent.device
    n_act   = env.action_dim

    harm_rates: List[float] = []

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        ep_harm_sum = 0.0
        ep_steps    = 0

        for _ in range(STEPS_PER_EP):
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

            _, harm_signal, done, info, obs_dict = env.step(action)

            if float(harm_signal) < 0:
                ep_harm_sum += abs(float(harm_signal))
            ep_steps += 1

            if planned:
                with torch.no_grad():
                    _update_z_goal(agent, obs_dict["body_state"])

            if done:
                break

        harm_rates.append(ep_harm_sum / max(1, ep_steps))

    harm_rate = sum(harm_rates) / max(1, len(harm_rates))
    return {"harm_rate": harm_rate}


# ---------------------------------------------------------------------------
# Per-seed, per-cell run
# ---------------------------------------------------------------------------

def _run_cell_seed(
    hz_rate: float,
    res_rate: float,
    seed: int,
    n_train: int,
    n_eval: int,
) -> Dict:
    """Run HABIT and PLANNED for one cell-seed combination."""
    num_hazards   = _rate_to_count(hz_rate)
    num_resources = _rate_to_count(res_rate)
    ratio         = hz_rate / res_rate

    print(
        f"\n[V3-EXQ-288] Seed {seed}"
        f" Condition h{hz_rate:.1f}r{res_rate:.1f}"
        f" (hazards={num_hazards} resources={num_resources} ratio={ratio:.2f})",
        flush=True,
    )

    results = {}
    for condition in ["HABIT", "PLANNED"]:
        env   = _make_env(num_hazards, num_resources, seed)
        agent = _make_agent(condition, env, seed)

        train_info = _train(agent, env, condition, n_train, seed, hz_rate, res_rate)
        eval_info  = _eval(agent, env, condition, n_eval)

        harm_rate = eval_info["harm_rate"]
        goal_norm = train_info["goal_norm"]

        print(
            f"  verdict: PASS"
            f" seed={seed} h{hz_rate:.1f}r{res_rate:.1f} {condition}"
            f" harm_rate={harm_rate:.5f}"
            f" goal_norm={goal_norm:.3f}",
            flush=True,
        )

        results[condition] = {
            "harm_rate": harm_rate,
            "goal_norm": goal_norm,
        }

    gap = abs(results["HABIT"]["harm_rate"] - results["PLANNED"]["harm_rate"])
    return {
        "seed":     seed,
        "gap":      gap,
        "habit_harm_rate":   results["HABIT"]["harm_rate"],
        "planned_harm_rate": results["PLANNED"]["harm_rate"],
        "goal_norm_planned": results["PLANNED"]["goal_norm"],
    }


# ---------------------------------------------------------------------------
# Aggregate grid results
# ---------------------------------------------------------------------------

def _aggregate_grid(
    grid_data: Dict,
) -> Tuple[List[Dict], Optional[float], bool]:
    """
    Aggregate per-cell results.

    grid_data: dict keyed by (hz_rate, res_rate) -> list of per-seed results

    Returns:
      grid_results: list of 9 dicts (one per cell)
      threshold_estimate: ratio at transition, or None
      transition_found: bool
    """
    grid_results = []

    for (hz_rate, res_rate), seed_results in sorted(
        grid_data.items(), key=lambda x: x[0][0] / x[0][1]
    ):
        ratio         = hz_rate / res_rate
        gaps          = [s["gap"] for s in seed_results]
        mean_gap      = sum(gaps) / max(1, len(gaps))
        seeds_equiv   = sum(1 for g in gaps if g < EQUIVALENCE_GAP)

        grid_results.append({
            "hazard_rate":                hz_rate,
            "resource_rate":              res_rate,
            "ratio":                      round(ratio, 4),
            "num_hazards":                _rate_to_count(hz_rate),
            "num_resources":              _rate_to_count(res_rate),
            "mean_gap":                   round(mean_gap, 6),
            "seeds_equivalent":           seeds_equiv,
            "behavioral_equivalence_score": round(mean_gap, 6),
            "per_seed":                   seed_results,
        })

    # Sort by ratio for threshold sweep
    grid_results_sorted = sorted(grid_results, key=lambda x: x["ratio"])

    # Identify transition: find ratio R* such that cells with ratio >= R* have
    # seeds_equivalent >= MAJORITY_SEEDS, and cells below do not.
    # We scan candidate thresholds (each unique ratio value).
    unique_ratios = sorted(set(r["ratio"] for r in grid_results_sorted))

    threshold_estimate = None
    transition_found   = False

    for candidate in unique_ratios:
        high_cells = [r for r in grid_results_sorted if r["ratio"] >= candidate]
        low_cells  = [r for r in grid_results_sorted if r["ratio"] < candidate]

        n_high_equiv    = sum(1 for r in high_cells if r["seeds_equivalent"] >= MAJORITY_SEEDS)
        n_low_not_equiv = sum(1 for r in low_cells  if r["seeds_equivalent"] < MAJORITY_SEEDS)

        if (n_high_equiv >= MIN_HIGH_RATIO_EQUIV and
                n_low_not_equiv >= MIN_LOW_RATIO_NOT_EQUIV):
            threshold_estimate = candidate
            transition_found   = True
            break

    return grid_results_sorted, threshold_estimate, transition_found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n_train = 2    if args.dry_run else TRAINING_EPISODES
    n_eval  = 2    if args.dry_run else EVAL_EPISODES
    seeds   = [42] if args.dry_run else SEEDS

    print(
        f"[V3-EXQ-288] Q-034 Hazard/Resource Ratio Threshold Sweep"
        f" dry_run={args.dry_run}"
        f" train={n_train} eval={n_eval}"
        f" seeds={seeds}"
        f" hazard_rates={HAZARD_RATES}"
        f" resource_rates={RESOURCE_RATES}",
        flush=True,
    )
    print(
        f"  Grid: {len(HAZARD_RATES)}x{len(RESOURCE_RATES)}={len(HAZARD_RATES)*len(RESOURCE_RATES)} cells"
        f" x {len(seeds)} seeds"
        f" x 2 conditions (HABIT/PLANNED)",
        flush=True,
    )
    print(
        f"  equivalence_gap={EQUIVALENCE_GAP}"
        f"  majority_seeds={MAJORITY_SEEDS}/{len(seeds)}"
        f"  min_high_equiv={MIN_HIGH_RATIO_EQUIV}"
        f"  min_low_not_equiv={MIN_LOW_RATIO_NOT_EQUIV}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Run all 9 cells x seeds
    # -----------------------------------------------------------------------
    grid_data: Dict = {}

    for hz_rate in HAZARD_RATES:
        for res_rate in RESOURCE_RATES:
            cell_results = []
            for seed in seeds:
                seed_result = _run_cell_seed(
                    hz_rate, res_rate, seed, n_train, n_eval
                )
                cell_results.append(seed_result)
            grid_data[(hz_rate, res_rate)] = cell_results

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    grid_results, threshold_estimate, transition_found = _aggregate_grid(grid_data)

    outcome = "PASS" if transition_found else "FAIL"

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"\n[V3-EXQ-288] === Results ===", flush=True)
    print(
        f"  {'hz_r':>4} {'res_r':>5} {'ratio':>6} {'mean_gap':>9}"
        f" {'equiv':>5} {'verdict':>7}",
        flush=True,
    )
    for r in grid_results:
        equiv_str = f"{r['seeds_equivalent']}/{len(seeds)}"
        verd = "equiv" if r["seeds_equivalent"] >= MAJORITY_SEEDS else "gap"
        print(
            f"  {r['hazard_rate']:>4.1f} {r['resource_rate']:>5.1f}"
            f" {r['ratio']:>6.2f} {r['mean_gap']:>9.5f}"
            f" {equiv_str:>5} {verd:>7}",
            flush=True,
        )

    if transition_found:
        print(
            f"\n  Transition found at ratio >= {threshold_estimate:.4f}",
            flush=True,
        )
    else:
        print(f"\n  No clear transition found.", flush=True)

    print(f"\n  -> {outcome}", flush=True)

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    ts      = int(time.time())
    ts_utc  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    # Serialize grid_results (remove per_seed detail for top-level, keep it nested)
    grid_results_out = []
    for r in grid_results:
        row = {k: v for k, v in r.items() if k != "per_seed"}
        row["per_seed"] = [
            {
                "seed":              s["seed"],
                "gap":               round(s["gap"], 6),
                "habit_harm_rate":   round(s["habit_harm_rate"], 6),
                "planned_harm_rate": round(s["planned_harm_rate"], 6),
                "goal_norm_planned": round(s["goal_norm_planned"], 4),
            }
            for s in r["per_seed"]
        ]
        grid_results_out.append(row)

    manifest = {
        "run_id":               f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":      EXPERIMENT_TYPE,
        "architecture_epoch":   "ree_hybrid_guardrails_v1",
        "claim_ids":            CLAIM_IDS,
        "experiment_purpose":   EXPERIMENT_PURPOSE,
        "outcome":              outcome,
        "timestamp_utc":        ts_utc,
        # Parameters
        "hazard_rates":         HAZARD_RATES,
        "resource_rates":       RESOURCE_RATES,
        "seeds":                seeds,
        "grid_size":            GRID_SIZE,
        "training_episodes":    n_train,
        "eval_episodes":        n_eval,
        "steps_per_ep":         STEPS_PER_EP,
        "equivalence_gap":      EQUIVALENCE_GAP,
        "majority_seeds":       MAJORITY_SEEDS,
        "drive_weight":         2.0,
        "use_resource_proximity_head": True,
        # Results
        "grid_results":         grid_results_out,
        "threshold_estimate":   threshold_estimate,
        "transition_found":     transition_found,
        # Summary metrics
        "n_cells":              len(grid_results),
        "n_cells_equivalent":   sum(
            1 for r in grid_results
            if r["seeds_equivalent"] >= MAJORITY_SEEDS
        ),
        "mean_gap_all_cells":   round(
            sum(r["mean_gap"] for r in grid_results) / max(1, len(grid_results)),
            6,
        ),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-288] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
