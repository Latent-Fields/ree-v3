#!/usr/bin/env python3
"""
V3-EXQ-259: VALENCE_WANTING Gradient Navigation

Tests whether wanting-gradient trajectory scoring (HippocampalModule.wanting_weight > 0)
enables directed navigation toward resources when VALENCE_WANTING is populated by
SerotoninModule.update_benefit_salience().

This resolves SD-015 (z_resource separation) via the VALENCE_WANTING landscape in the
residue field rather than a separate z_resource latent stream. The residue field
accumulates benefit-encounter history; _score_trajectory() prefers trajectories through
high-wanting regions; this biases CEM toward resource-proximal trajectories.

Two conditions per seed:
  A: WITH_WANTING  -- wanting_weight=0.4, tonic_5ht_enabled=True
  B: WITHOUT       -- wanting_weight=0.0, tonic_5ht_enabled=True (ablation baseline)

Both use: use_resource_proximity_head=True, use_event_classifier=True,
  use_harm_stream=True, z_goal_enabled=True, drive_weight=2.0,
  benefit_eval_enabled=True.

Training structure:
  P0 (100 ep): warmup -- VALENCE_WANTING builds up, encoders trained
  P1 (50 ep):  evaluation -- measure navigation quality

PASS criteria:
  C1: mean VALENCE_WANTING at visited locations > 0.01 in WITH_WANTING (confirms population)
  C2: resource_rate_WITH > resource_rate_WITHOUT * 1.1 in P1 (10% navigation lift),
      majority of seeds (>= 2/3)
  C3: mean_benefit_WITH > mean_benefit_WITHOUT in majority of seeds (>= 2/3)

Background: docs/substrate_plans/sd015_zresource_gap_note.md
Claims: SD-015, MECH-112, ARC-030, SD-012
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import torch
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.residue.field import VALENCE_WANTING

EXPERIMENT_PURPOSE = "evidence"

# -- Config --
SEEDS = [1, 2, 3]
CONDITIONS = ["WITH_WANTING", "WITHOUT"]
P0_EPISODES = 100    # warmup: VALENCE_WANTING populates
P1_EPISODES = 50     # evaluation: measure navigation quality
STEPS_PER_EP = 200
GRID_SIZE = 5
NUM_RESOURCES = 2
NUM_HAZARDS = 2
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
LR = 1e-3
LAMBDA_RESOURCE = 0.5
WANTING_WEIGHT = 0.4

# PASS thresholds
C1_WANTING_THRESHOLD = 0.01
C2_RESOURCE_LIFT = 1.1   # 10%
C2_MIN_SEEDS = 2
C3_MIN_SEEDS = 2


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=0.3,
        resource_benefit=0.5,
        use_proxy_fields=True,
        seed=seed,
    )


def make_config(condition: str) -> REEConfig:
    use_wanting = condition == "WITH_WANTING"
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=5,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_event_classifier=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=LAMBDA_RESOURCE,
        use_harm_stream=True,
        harm_obs_dim=51,
        z_harm_dim=32,
        z_goal_enabled=True,
        drive_weight=2.0,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        tonic_5ht_enabled=True,
        wanting_weight=WANTING_WEIGHT if use_wanting else 0.0,
    )


def run_condition(seed: int, condition: str, dry_run: bool = False) -> Dict:
    """Run one seed x condition pair."""
    print(f"  Seed {seed} {condition}")

    env = make_env(seed)
    cfg = make_config(condition)
    agent = REEAgent(cfg)
    device = agent.device

    optimizer = optim.Adam(list(agent.parameters()), lr=LR)

    total_eps = P0_EPISODES + P1_EPISODES
    eval_start = P0_EPISODES

    p1_resource_counts: List[float] = []
    p1_benefit_exposures: List[float] = []
    wanting_vals_visited: List[float] = []

    for ep in range(total_eps):
        if dry_run and ep >= 3:
            break

        obs, info = env.reset()
        agent.reset()

        body_obs = torch.tensor(obs[:BODY_OBS_DIM], dtype=torch.float32).unsqueeze(0)
        world_obs = torch.tensor(
            obs[BODY_OBS_DIM:BODY_OBS_DIM + WORLD_OBS_DIM], dtype=torch.float32
        ).unsqueeze(0)

        ep_resources = 0
        ep_benefit = 0.0

        for step in range(STEPS_PER_EP):
            latent = agent.sense(body_obs, world_obs)
            ticks = agent.clock.advance()

            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, cfg.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            # Extract signals from body obs
            benefit_exposure = (
                float(body_obs[0, 11]) if body_obs.shape[-1] > 11 else 0.0
            )
            drive_level = agent.compute_drive_level(body_obs)

            # Serotonin + goal update -- populates VALENCE_WANTING
            agent.serotonin_step(benefit_exposure)
            agent.update_z_goal(benefit_exposure, drive_level)
            agent.update_benefit_salience(benefit_exposure)

            # C1: sample VALENCE_WANTING at current location
            if latent is not None and latent.z_world is not None:
                with torch.no_grad():
                    valence = agent.residue_field.evaluate_valence(latent.z_world)
                wanting_vals_visited.append(float(valence[0, VALENCE_WANTING].item()))

            # Track resource encounters
            if benefit_exposure > 0.01:
                ep_resources += 1
            ep_benefit += benefit_exposure

            # Step environment
            action_idx = int(action.argmax(dim=-1).item())
            obs_next, reward, done, truncated, info_next = env.step(action_idx)

            # Residue update on harm
            harm_signal = info_next.get("harm", 0.0)
            if harm_signal != 0:
                agent.update_residue(harm_signal)

            # Training
            optimizer.zero_grad()
            loss = agent.compute_prediction_loss()
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                optimizer.step()

            # Advance obs
            body_obs = torch.tensor(
                obs_next[:BODY_OBS_DIM], dtype=torch.float32
            ).unsqueeze(0)
            world_obs = torch.tensor(
                obs_next[BODY_OBS_DIM:BODY_OBS_DIM + WORLD_OBS_DIM], dtype=torch.float32
            ).unsqueeze(0)

            if done or truncated:
                break

        if ep >= eval_start:
            p1_resource_counts.append(float(ep_resources))
            p1_benefit_exposures.append(
                ep_benefit / max(1, STEPS_PER_EP)
            )

        if (ep + 1) % 50 == 0:
            phase = "P0" if ep < eval_start else "P1"
            print(
                f"    {phase} ep {ep+1}/{total_eps} "
                f"resources={ep_resources} benefit={ep_benefit:.3f}"
            )

    resource_rate = float(np.mean(p1_resource_counts)) if p1_resource_counts else 0.0
    mean_benefit = float(np.mean(p1_benefit_exposures)) if p1_benefit_exposures else 0.0
    wanting_mean = (
        float(np.mean(wanting_vals_visited)) if wanting_vals_visited else 0.0
    )

    print(
        f"    -> resource_rate={resource_rate:.3f} "
        f"benefit={mean_benefit:.4f} "
        f"wanting_mean={wanting_mean:.4f}"
    )

    return {
        "seed": seed,
        "condition": condition,
        "resource_rate": resource_rate,
        "mean_benefit_exposure": mean_benefit,
        "valence_wanting_mean": wanting_mean,
        "p1_resource_counts": p1_resource_counts,
        "p1_benefit_exposures": p1_benefit_exposures,
    }


def evaluate_criteria(results: List[Dict]) -> Dict:
    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    with_list = sorted(by_cond.get("WITH_WANTING", []), key=lambda x: x["seed"])
    without_list = sorted(by_cond.get("WITHOUT", []), key=lambda x: x["seed"])

    # C1: VALENCE_WANTING populated in WITH_WANTING condition
    c1_vals = [r["valence_wanting_mean"] for r in with_list]
    c1_pass = all(v > C1_WANTING_THRESHOLD for v in c1_vals)

    # C2: resource_rate lift
    c2_lifts = []
    for w, wo in zip(with_list, without_list):
        base = max(wo["resource_rate"], 1e-6)
        c2_lifts.append(w["resource_rate"] / base)
    c2_pass = sum(1 for v in c2_lifts if v >= C2_RESOURCE_LIFT) >= C2_MIN_SEEDS

    # C3: mean_benefit lift
    c3_results = [
        w["mean_benefit_exposure"] > wo["mean_benefit_exposure"]
        for w, wo in zip(with_list, without_list)
    ]
    c3_pass = sum(c3_results) >= C3_MIN_SEEDS

    return {
        "c1_wanting_populated": c1_pass,
        "c1_vals": c1_vals,
        "c2_resource_rate_lift": c2_pass,
        "c2_lifts": c2_lifts,
        "c3_benefit_lift": c3_pass,
        "overall_pass": c1_pass and c2_pass and c3_pass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = (
        "v3_exq_259_wanting_gradient_navigation_dry"
        if args.dry_run
        else f"v3_exq_259_wanting_gradient_navigation_{int(time.time())}"
    )
    print(f"EXQ-259 start: {run_id}")

    all_results = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            result = run_condition(seed, condition, dry_run=args.dry_run)
            all_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== EXQ-259 {outcome} ===")
    print(f"C1 wanting_populated: {criteria['c1_wanting_populated']} "
          f"(vals={[f'{v:.4f}' for v in criteria['c1_vals']]})")
    print(f"C2 resource_rate_lift: {criteria['c2_resource_rate_lift']} "
          f"(lifts={[f'{v:.3f}' for v in criteria['c2_lifts']]})")
    print(f"C3 benefit_lift: {criteria['c3_benefit_lift']}")

    output = {
        "run_id": run_id,
        "experiment_type": "v3_exq_259_wanting_gradient_navigation",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": ["SD-015", "MECH-112", "ARC-030", "SD-012"],
        "evidence_direction_per_claim": {
            "SD-015": "supports" if criteria["overall_pass"] else "does_not_support",
            "MECH-112": "supports" if criteria["overall_pass"] else "does_not_support",
            "ARC-030": "supports" if criteria["overall_pass"] else "does_not_support",
            "SD-012": "supports" if criteria["overall_pass"] else "does_not_support",
        },
        "evidence_direction": (
            "supports" if criteria["overall_pass"] else "does_not_support"
        ),
        "outcome": outcome,
        "criteria": criteria,
        "results_per_condition": all_results,
        "config": {
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "wanting_weight": WANTING_WEIGHT,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "steps_per_ep": STEPS_PER_EP,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
        "v3_exq_259_wanting_gradient_navigation",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
