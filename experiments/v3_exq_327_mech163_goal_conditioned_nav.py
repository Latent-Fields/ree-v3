#!/opt/local/bin/python3
"""
V3-EXQ-327: MECH-163 Hippocampal Goal-Conditioned Navigation

experiment_purpose: evidence

Tests MECH-163 (z_goal in E3 trajectory scoring via goal_weight) and
SD-015 (z_resource encoder seeding enables goal-directed navigation).

Design: Does using z_goal as a scoring term in E3.score_trajectory()
(via E3Config.goal_weight) produce benefit_ratio >= 1.3x (paper gate)?

E3Config.goal_weight already exists in the substrate (default 0.0).
When goal_weight > 0.0 and goal_state.is_active(), E3.score_trajectory()
subtracts goal_weight * goal_proximity from the cost, biasing selection
toward goal-proximal trajectories.

Two conditions per seed:
  GOAL_CONDITIONED: goal_weight=1.0, z_goal_enabled=True, drive_weight=2.0,
                    use_resource_proximity_head=True (SD-018 prerequisite)
  GOAL_ABLATED:     goal_weight=0.0, z_goal_enabled=False (no goal term)
  Both use same env config (num_hazards=2, hazard_harm=0.1).

Phased training:
  P0 (100 ep): encoder + proximity head warmup (no goal updates)
  P1 (100 ep): full pipeline (z_goal seeding + goal_weight active in
               GOAL_CONDITIONED; ABLATED trains encoders only)
  P2 (50 ep):  evaluation (no training updates)

Pass criteria:
  C1: benefit_ratio >= 1.3x in GOAL_CONDITIONED (paper gate primary, >= 2/3 seeds)
  C2: prox_r2 >= 0.7 in GOAL_CONDITIONED (resource_prox_pred quality check)
  C3: GOAL_CONDITIONED benefit > GOAL_ABLATED benefit (goal term causal, >= 2/3 seeds)

PASS: C1 AND C3 across >= 2/3 seeds (C2 advisory).

Claims: MECH-163, SD-015
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import random
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.optim as optim
import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_327_mech163_goal_conditioned_nav"
CLAIM_IDS          = ["MECH-163", "SD-015"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS       = [42, 7, 13]
CONDITIONS  = ["GOAL_CONDITIONED", "GOAL_ABLATED"]

P0_EPISODES = 100
P1_EPISODES = 100
P2_EPISODES = 50
STEPS_PER_EP = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 2
HAZARD_HARM   = 0.1
DRIVE_WEIGHT  = 2.0
GOAL_WEIGHT   = 1.0

LR = 3e-4

C1_BENEFIT_RATIO  = 1.3
C3_MIN_SEEDS_PASS = 2
C1_MIN_SEEDS_PASS = 2

DRY_RUN_EPISODES = 3
DRY_RUN_STEPS    = 10


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        hazard_harm=HAZARD_HARM,
        resource_benefit=0.5,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.05,
        proximity_approach_threshold=0.15,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, condition: str, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    goal_on = (condition == "GOAL_CONDITIONED")
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        use_event_classifier=True,
        z_goal_enabled=goal_on,
        drive_weight=DRIVE_WEIGHT if goal_on else 0.0,
        goal_weight=GOAL_WEIGHT if goal_on else 0.0,
        benefit_eval_enabled=goal_on,
        benefit_weight=1.0,
        e1_goal_conditioned=goal_on,
        wanting_weight=0.3 if goal_on else 0.0,
    )
    return REEAgent(config)


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_condition(
    seed: int,
    condition: str,
    dry_run: bool = False,
) -> Dict:
    total_p0   = DRY_RUN_EPISODES if dry_run else P0_EPISODES
    total_p1   = DRY_RUN_EPISODES if dry_run else P1_EPISODES
    total_p2   = DRY_RUN_EPISODES if dry_run else P2_EPISODES
    steps_per  = DRY_RUN_STEPS    if dry_run else STEPS_PER_EP
    total_eps  = total_p0 + total_p1 + total_p2

    print(f"  Seed {seed} Condition {condition}")

    env   = _make_env(seed)
    agent = _make_agent(env, condition, seed)
    device = agent.device

    optimizer = optim.Adam(list(agent.parameters()), lr=LR)

    p2_resources:   List[float] = []
    p2_benefit:     List[float] = []
    prox_preds:     List[float] = []
    prox_targets:   List[float] = []
    prev_ttype = "none"

    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()

        phase = "P0" if ep < total_p0 else ("P1" if ep < total_p0 + total_p1 else "P2")
        in_eval = (phase == "P2")

        ep_resources = 0
        ep_benefit   = 0.0

        for _step in range(steps_per):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            z_self_prev: Optional[torch.Tensor] = None
            if agent._current_latent is not None:
                z_self_prev = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()

            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", True)
                else torch.zeros(1, 32, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks)

            # SD-012 goal seeding (P1/P2 GOAL_CONDITIONED only)
            if condition == "GOAL_CONDITIONED" and phase != "P0":
                benefit_raw = float(obs_body.flatten()[11].item()) if obs_body.shape[-1] > 11 else 0.0
                drive_level = REEAgent.compute_drive_level(obs_body)
                agent.update_z_goal(benefit_raw, drive_level)

            # Track resource_prox prediction quality
            rfv = obs_dict.get("resource_field_view", None)
            if rfv is not None and latent.resource_prox_pred is not None:
                rp_target = float(rfv.max().item())
                rp_pred   = float(latent.resource_prox_pred.squeeze().item())
                prox_targets.append(rp_target)
                prox_preds.append(rp_pred)

            action_idx = int(action.argmax(dim=-1).item())
            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            if ttype == "resource":
                ep_resources += 1
            ep_benefit += max(0.0, float(info.get("benefit_exposure", 0.0)))

            agent.update_residue(float(harm_signal))
            if z_self_prev is not None:
                agent.record_transition(z_self_prev, action, latent.z_self.detach())

            if not in_eval:
                optimizer.zero_grad()
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                loss = e1_loss + e2_loss

                if rfv is not None:
                    rp_target_f = float(rfv.max().item())
                    rp_loss = agent.compute_resource_proximity_loss(rp_target_f, latent)
                    loss = loss + rp_loss

                latent2 = agent.sense(obs_body, obs_world)
                ec_loss = agent.compute_event_contrastive_loss(prev_ttype, latent2)
                loss = loss + ec_loss

                if loss.requires_grad:
                    loss.backward()
                    import torch.nn as nn
                    nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                    optimizer.step()

            prev_ttype = ttype
            obs_dict = obs_dict_next

            if done:
                break

        if in_eval:
            p2_resources.append(float(ep_resources))
            p2_benefit.append(ep_benefit / max(1, steps_per))

        if (ep + 1) % 50 == 0:
            print(
                f"    [train] seed={seed} {condition} ep {ep+1}/{total_eps} "
                f"phase={phase} resources={ep_resources} benefit={ep_benefit:.3f}",
                flush=True,
            )

    resource_rate = float(np.mean(p2_resources)) if p2_resources else 0.0
    mean_benefit  = float(np.mean(p2_benefit))   if p2_benefit   else 0.0

    prox_r2 = 0.0
    if len(prox_preds) >= 10:
        try:
            tgt = np.array(prox_targets)
            prd = np.array(prox_preds)
            ss_res = float(np.sum((tgt - prd) ** 2))
            ss_tot = float(np.sum((tgt - tgt.mean()) ** 2))
            prox_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        except Exception:
            prox_r2 = 0.0

    print(f"  verdict: {'PASS' if resource_rate > 0 else 'FAIL'} "
          f"resource_rate={resource_rate:.3f} benefit={mean_benefit:.4f} prox_r2={prox_r2:.3f}")

    return {
        "seed": seed,
        "condition": condition,
        "resource_rate": resource_rate,
        "mean_benefit_exposure": mean_benefit,
        "prox_r2": prox_r2,
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_cond[r["condition"]].append(r)

    gc_list = sorted(by_cond.get("GOAL_CONDITIONED", []), key=lambda x: x["seed"])
    ab_list = sorted(by_cond.get("GOAL_ABLATED",     []), key=lambda x: x["seed"])

    # C1: benefit_ratio >= 1.3x
    c1_seeds = 0
    c1_ratios = []
    for gc, ab in zip(gc_list, ab_list):
        base = max(ab["mean_benefit_exposure"], 1e-6)
        ratio = gc["mean_benefit_exposure"] / base
        c1_ratios.append(ratio)
        if ratio >= C1_BENEFIT_RATIO:
            c1_seeds += 1
    c1_pass = c1_seeds >= C1_MIN_SEEDS_PASS

    # C2: prox_r2 >= 0.7 (advisory)
    c2_vals = [r["prox_r2"] for r in gc_list]
    c2_pass = all(v >= 0.7 for v in c2_vals)

    # C3: GOAL_CONDITIONED benefit > GOAL_ABLATED benefit
    c3_seeds = sum(
        gc["mean_benefit_exposure"] > ab["mean_benefit_exposure"]
        for gc, ab in zip(gc_list, ab_list)
    )
    c3_pass = c3_seeds >= C3_MIN_SEEDS_PASS

    overall_pass = c1_pass and c3_pass
    return {
        "c1_benefit_ratio_pass": c1_pass,
        "c1_ratios": c1_ratios,
        "c1_seeds_pass": c1_seeds,
        "c2_prox_r2_advisory": c2_pass,
        "c2_vals": c2_vals,
        "c3_goal_causal_pass": c3_pass,
        "c3_seeds_pass": c3_seeds,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"v3_exq_327_mech163_goal_conditioned_nav_dry_{ts}_v3"
        if args.dry_run
        else f"v3_exq_327_mech163_goal_conditioned_nav_{ts}_v3"
    )
    print(f"EXQ-327 start: {run_id}")

    all_results: List[Dict] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            result = run_condition(seed, condition, dry_run=args.dry_run)
            all_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome  = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== EXQ-327 {outcome} ===")
    print(f"C1 benefit_ratio: {criteria['c1_benefit_ratio_pass']} "
          f"(ratios={[f'{v:.3f}' for v in criteria['c1_ratios']]})")
    print(f"C2 prox_r2 (advisory): {criteria['c2_prox_r2_advisory']} "
          f"(vals={[f'{v:.3f}' for v in criteria['c2_vals']]})")
    print(f"C3 goal_causal: {criteria['c3_goal_causal_pass']}")

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction_per_claim": {
            "MECH-163": "supports" if criteria["overall_pass"] else "does_not_support",
            "SD-015":   "supports" if criteria["c1_benefit_ratio_pass"] else "does_not_support",
        },
        "evidence_direction": "supports" if criteria["overall_pass"] else "does_not_support",
        "outcome": outcome,
        "criteria": criteria,
        "results_per_condition": all_results,
        "config": {
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "p2_episodes": P2_EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "goal_weight": GOAL_WEIGHT,
            "drive_weight": DRIVE_WEIGHT,
        },
        "timestamp_utc": ts,
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
        EXPERIMENT_TYPE,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
