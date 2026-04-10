#!/opt/local/bin/python3
"""
V3-EXQ-322: SD-015 ResourceEncoder vs z_world Seeding of z_goal

experiment_purpose: evidence

Tests that seeding GoalState from z_resource (SD-015 ResourceEncoder) produces
a z_goal representation more correlated with resource proximity than z_world seeding,
which encodes the full scene without isolating object-type features.

Two conditions per seed:
  RESOURCE_SEED -- use_resource_encoder=True (z_goal seeded from z_resource)
  WORLD_SEED    -- use_resource_encoder=False (z_goal seeded from z_world, current baseline)

Key metrics:
  goal_resource_r    -- cosine similarity between z_goal and z_resource (after seeding)
  benefit_ratio      -- episodes where benefit_exposure > threshold / total episodes

Pass criterion (pre-registered):
  C1: goal_resource_r_resource_seed > goal_resource_r_world_seed (z_goal better captures resource)
  C2: goal_resource_r_resource_seed >= 0.3 (absolute threshold: z_goal has some resource content)
  C3: benefit_ratio_resource_seed >= benefit_ratio_world_seed (navigation not worse)

Experiment PASS: >= 3/5 seeds satisfy C1 and C2.

Claims: SD-015 (z_resource encoder enables goal-directed navigation), MECH-112
"""

import json
import sys
import random
import datetime
import argparse
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig, LatentStackConfig


EXPERIMENT_TYPE = "v3_exq_322_sd015_resource_encoder_seeding"
CLAIM_IDS = ["SD-015", "MECH-112"]

C1_threshold = 0.0    # resource_seed cosine > world_seed cosine
C2_threshold = 0.2    # absolute cosine floor
C3_threshold = 0.0    # benefit_ratio not worse (resource >= world)
PASS_MIN_SEEDS = 3
BENEFIT_THRESHOLD = 0.1

SEEDS = [42, 43, 44, 45, 46]
TRAIN_EPISODES = 80
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200
LR = 1e-3


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=10, num_hazards=2, num_resources=3,
        hazard_harm=0.02, resource_benefit=0.18,
        use_proxy_fields=True, seed=seed,
        proximity_benefit_scale=0.18,
    )


def make_config(use_resource_encoder: bool) -> REEConfig:
    lat = LatentStackConfig(
        world_dim=32,
        use_resource_encoder=use_resource_encoder,
        z_resource_dim=32,
    )
    return REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        alpha_world=0.9,
        latent=lat,
        drive_weight=2.0,
        z_goal_enabled=True,
    )


def run_training(agent: REEAgent, env: CausalGridWorldV2, device, n_eps: int):
    """Train agent including ResourceEncoder proximity supervision (when enabled)."""
    opt = optim.Adam(agent.parameters(), lr=LR)
    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent = agent.sense(obs_body, obs_world)

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, _, done, info, obs_dict = env.step(action_idx)

            opt.zero_grad()
            loss = agent.compute_prediction_loss()

            # ResourceEncoder proximity supervision (when enabled)
            if (
                agent.latent_stack.resource_encoder is not None
                and latent.resource_prox_pred_r is not None
            ):
                prox_target_val = float(info.get("resource_proximity", 0.0))
                prox_target = torch.tensor([[prox_target_val]], dtype=torch.float32, device=device)
                res_loss = agent.compute_resource_encoder_loss(prox_target, latent)
                loss = loss + res_loss

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

            # z_goal update
            benefit_exp = float(info.get("benefit_exposure", 0.0))
            drive_lvl = float(info.get("drive_level", 0.5))
            agent.update_z_goal(benefit_exp, drive_level=drive_lvl)

            if done:
                break


def eval_goal_quality(agent: REEAgent, env: CausalGridWorldV2, device, n_eps: int) -> Dict:
    """Measure goal_resource_r and benefit_ratio."""
    cosine_sims = []
    benefit_episodes = 0

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        ep_benefit = 0.0

        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent = agent.sense(obs_body, obs_world)

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            with torch.no_grad():
                action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, _, done, info, obs_dict = env.step(action_idx)

            benefit_exp = float(info.get("benefit_exposure", 0.0))
            ep_benefit += benefit_exp

            # z_goal vs z_resource cosine similarity (when both available)
            if (
                agent.goal_state.z_goal is not None
                and latent.z_resource is not None
            ):
                with torch.no_grad():
                    z_g = agent.goal_state.z_goal
                    z_r = latent.z_resource
                    if z_g.ndim == 1:
                        z_g = z_g.unsqueeze(0)
                    if z_r.ndim == 1:
                        z_r = z_r.unsqueeze(0)
                    if z_g.shape[-1] == z_r.shape[-1]:
                        cos = float(F.cosine_similarity(z_g, z_r, dim=-1).mean().item())
                        cosine_sims.append(cos)

            # Update z_goal
            drive_lvl = float(info.get("drive_level", 0.5))
            agent.update_z_goal(benefit_exp, drive_level=drive_lvl)

            if done:
                break

        if ep_benefit >= BENEFIT_THRESHOLD:
            benefit_episodes += 1

    goal_resource_r = float(np.mean(cosine_sims)) if cosine_sims else 0.0
    benefit_ratio = benefit_episodes / max(n_eps, 1)
    return {
        "goal_resource_r": goal_resource_r,
        "benefit_ratio": benefit_ratio,
        "n_cosine_samples": len(cosine_sims),
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_train = 5 if dry_run else TRAIN_EPISODES
    n_eval = 3 if dry_run else EVAL_EPISODES

    print(f"Seed {seed}")
    condition_results = {}
    for condition in ["RESOURCE_SEED", "WORLD_SEED"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_env(seed)
        cfg = make_config(use_resource_encoder=(condition == "RESOURCE_SEED"))
        agent = REEAgent(cfg)

        print(f"  {condition}: training {n_train} eps...")
        run_training(agent, env, device, n_train)
        print(f"  {condition}: eval {n_eval} eps...")
        metrics = eval_goal_quality(agent, env, device, n_eval)
        condition_results[condition] = metrics
        print(
            f"  {condition}: goal_resource_r={metrics['goal_resource_r']:.4f} "
            f"benefit_ratio={metrics['benefit_ratio']:.3f}"
        )

    rs = condition_results["RESOURCE_SEED"]
    ws = condition_results["WORLD_SEED"]
    c1_pass = rs["goal_resource_r"] > ws["goal_resource_r"]
    c2_pass = rs["goal_resource_r"] >= C2_threshold
    c3_pass = rs["benefit_ratio"] >= ws["benefit_ratio"]
    seed_pass = c1_pass and c2_pass

    print(f"  -> {'PASS' if seed_pass else 'FAIL'}")
    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "goal_resource_r_resource": rs["goal_resource_r"],
        "goal_resource_r_world": ws["goal_resource_r"],
        "benefit_ratio_resource": rs["benefit_ratio"],
        "benefit_ratio_world": ws["benefit_ratio"],
        "c1_resource_r_higher": c1_pass,
        "c2_abs_threshold": c2_pass,
        "c3_benefit_not_worse": c3_pass,
        "condition_results": condition_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        "v3_exq_322_sd015_resource_encoder_seeding_dry" if args.dry_run
        else f"v3_exq_322_sd015_resource_encoder_seeding_{timestamp}_v3"
    )
    print(f"EXQ-322 start: {run_id}")

    per_seed = [run_seed(s, dry_run=args.dry_run) for s in SEEDS]
    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-322 {outcome} ===")
    print(f"Seeds pass: {seeds_passing}/{len(SEEDS)}")
    for r in per_seed:
        s = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {s} "
            f"r_res={r['goal_resource_r_resource']:.4f} r_world={r['goal_resource_r_world']:.4f} "
            f"ben_res={r['benefit_ratio_resource']:.3f}"
        )

    evidence_direction = "supports" if experiment_passes else "does_not_support"
    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": "evidence",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {
            "SD-015": evidence_direction,
            "MECH-112": evidence_direction,
        },
        "outcome": outcome,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "registered_thresholds": {
            "C1_resource_r_higher": C1_threshold,
            "C2_abs_resource_r": C2_threshold,
            "C3_benefit_not_worse": C3_threshold,
            "seeds_needed": PASS_MIN_SEEDS,
        },
        "per_seed_results": per_seed,
        "seeds_passing": seeds_passing,
        "experiment_passes": experiment_passes,
    }

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
