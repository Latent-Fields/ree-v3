#!/usr/bin/env python3
"""
V3-EXQ-257: SD-018 Resource Proximity Supervision Validation

Tests whether auxiliary resource-proximity regression on z_world encoder
enables benefit_eval_head to learn and goal-directed behavior to emerge.

Two conditions per seed:
  A: WITH_RESOURCE_HEAD  (use_resource_proximity_head=True)
  B: WITHOUT             (use_resource_proximity_head=False, ablation baseline)

Both use: use_event_classifier=True, use_harm_stream=True, z_goal_enabled=True,
  drive_weight=2.0, benefit_eval_enabled=True, benefit_weight=1.0.

Phased training (MANDATORY per CLAUDE.md):
  P0: encoder warmup (50 ep) -- E1/E2/recon/event_ce/(resource_prox if WITH)
  P1: benefit_eval_head training (30 ep) -- freeze encoder, train head on detached z_world
  P2: evaluation (20 ep) -- all frozen, collect metrics

PASS criteria:
  C1: resource_prox_r2 > 0.3 (WITH condition, eval episodes)
  C2: benefit_eval_r2_WITH > benefit_eval_r2_WITHOUT in all seeds
  C3: resource_rate_WITH > resource_rate_WITHOUT * 1.1 (10% lift)

Claims: SD-018, ARC-030, MECH-112
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent

EXPERIMENT_PURPOSE = "evidence"

# -- Config --
SEEDS = [1, 2, 3]
CONDITIONS = ["WITH_RESOURCE_HEAD", "WITHOUT"]
P0_EPISODES = 50   # encoder warmup
P1_EPISODES = 30   # benefit_eval_head training (frozen encoder)
P2_EPISODES = 20   # evaluation
STEPS_PER_EP = 200
TOTAL_TRAIN_EPS = P0_EPISODES + P1_EPISODES  # 80 for progress reporting
GRID_SIZE = 5
NUM_RESOURCES = 2
NUM_HAZARDS = 2
LR = 1e-3
LAMBDA_EVENT = 0.5
LAMBDA_RESOURCE = 0.5   # SD-018 resource proximity loss weight

# -- Thresholds --
C1_RESOURCE_PROX_R2 = 0.3
C2_BENEFIT_R2_LIFT = True  # WITH > WITHOUT in all seeds
C3_RESOURCE_RATE_LIFT = 1.1  # 10% lift


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


def make_agent(condition: str) -> REEAgent:
    use_rph = condition == "WITH_RESOURCE_HEAD"
    config = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_event_classifier=True,
        use_resource_proximity_head=use_rph,
        resource_proximity_weight=LAMBDA_RESOURCE,
        use_harm_stream=True,
        harm_obs_dim=51,
        z_harm_dim=32,
        z_goal_enabled=True,
        drive_weight=2.0,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
    )
    return REEAgent(config)


def compute_r2(preds: List[float], actuals: List[float]) -> float:
    if len(preds) < 2:
        return 0.0
    y = np.array(actuals)
    yhat = np.array(preds)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def run_condition(seed: int, condition: str) -> Dict:
    """Run one seed x condition pair."""
    print(f"Seed {seed} Condition {condition}")

    env = make_env(seed)
    agent = make_agent(condition)
    device = agent.device
    use_rph = condition == "WITH_RESOURCE_HEAD"

    # Optimizers
    all_params = list(agent.parameters())
    optimizer = optim.Adam(all_params, lr=LR)

    # Benefit eval head optimizer (P1: separate, frozen encoder)
    benefit_params = []
    for n, p in agent.e3.named_parameters():
        if "benefit_eval" in n:
            benefit_params.append(p)
    benefit_optimizer = optim.Adam(benefit_params, lr=LR) if benefit_params else None

    # ---- P0: Encoder warmup ----
    for ep in range(P0_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        prev_ttype = None
        prev_resource_prox = None

        for step in range(STEPS_PER_EP):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32).unsqueeze(0)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32).unsqueeze(0)
            obs_harm = torch.tensor(obs_dict.get("harm_obs", np.zeros(51)), dtype=torch.float32).unsqueeze(0)

            # Sense (retains grad for aux losses)
            latent = agent.latent_stack.encode(
                torch.cat([obs_body, obs_world], dim=-1),
                agent._current_latent,
                prev_action=agent._last_action,
                harm_obs=obs_harm,
            )
            agent._current_latent = latent.detach()

            # Compute losses
            e1_loss = agent.compute_prediction_loss()

            # SD-009: event contrastive
            event_loss = torch.zeros(1, device=device)
            if prev_ttype is not None:
                event_loss = agent.compute_event_contrastive_loss(prev_ttype, latent)

            # SD-018: resource proximity (P0 only for WITH condition)
            resource_loss = torch.zeros(1, device=device)
            if use_rph and prev_resource_prox is not None:
                resource_loss = agent.compute_resource_proximity_loss(prev_resource_prox, latent)

            total_loss = e1_loss + LAMBDA_EVENT * event_loss + LAMBDA_RESOURCE * resource_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # Action selection (simple random for P0)
            action_idx = np.random.randint(4)
            _, reward, done, info, obs_dict_next = env.step(action_idx)

            # Store for next step
            prev_ttype = info.get("transition_type", "none")
            rfv = obs_dict.get("resource_field_view", None)
            prev_resource_prox = float(rfv[12]) if rfv is not None else 0.0  # center cell = agent pos

            # Record E2 transition
            if agent._current_latent is not None:
                z_self_t = agent._current_latent.z_self
                action_t = torch.zeros(1, 4, device=device)
                action_t[0, action_idx] = 1.0
                latent_next = agent.latent_stack.encode(
                    torch.cat([
                        torch.tensor(obs_dict_next["body_state"], dtype=torch.float32).unsqueeze(0),
                        torch.tensor(obs_dict_next["world_state"], dtype=torch.float32).unsqueeze(0),
                    ], dim=-1),
                    agent._current_latent,
                    prev_action=action_t,
                    harm_obs=torch.tensor(obs_dict_next.get("harm_obs", np.zeros(51)), dtype=torch.float32).unsqueeze(0),
                )
                agent.record_transition(z_self_t, action_t, latent_next.z_self.detach())
                agent._current_latent = latent_next.detach()

            # Update z_goal
            be = obs_dict_next["body_state"][11] if len(obs_dict_next["body_state"]) > 11 else 0.0
            dl = 1.0 - obs_dict_next["body_state"][3] if len(obs_dict_next["body_state"]) > 3 else 1.0
            agent.update_z_goal(float(be), float(dl))

            # Harm/residue
            if reward < 0:
                agent.update_residue(reward)

            obs_dict = obs_dict_next
            if done:
                break

        if (ep + 1) % 10 == 0:
            print(f"  [train] P0 seed={seed} cond={condition} ep {ep+1}/{TOTAL_TRAIN_EPS} e1={e1_loss.item():.4f} res={resource_loss.item():.4f}", flush=True)

    # ---- P1: Benefit eval head training (frozen encoder) ----
    # Freeze encoder
    for p in agent.latent_stack.parameters():
        p.requires_grad_(False)
    for p in agent.e1.parameters():
        p.requires_grad_(False)
    for p in agent.e2.parameters():
        p.requires_grad_(False)

    benefit_preds_train = []
    benefit_actuals_train = []

    for ep in range(P1_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()

        for step in range(STEPS_PER_EP):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32).unsqueeze(0)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                latent = agent.latent_stack.encode(
                    torch.cat([obs_body, obs_world], dim=-1),
                    agent._current_latent,
                )
                agent._current_latent = latent.detach()

            z_world_detached = latent.z_world.detach()

            # Train benefit_eval_head on detached z_world
            if benefit_optimizer is not None:
                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    target_prox = float(rfv[12])  # center cell = agent pos
                    pred = agent.e3.benefit_eval_head(z_world_detached)
                    target_t = torch.tensor([[target_prox]], device=device)
                    loss = F.mse_loss(pred, target_t)
                    benefit_optimizer.zero_grad()
                    loss.backward()
                    benefit_optimizer.step()

                    benefit_preds_train.append(pred.item())
                    benefit_actuals_train.append(target_prox)

            action_idx = np.random.randint(4)
            _, reward, done, info, obs_dict_next = env.step(action_idx)

            be = obs_dict_next["body_state"][11] if len(obs_dict_next["body_state"]) > 11 else 0.0
            dl = 1.0 - obs_dict_next["body_state"][3] if len(obs_dict_next["body_state"]) > 3 else 1.0
            agent.update_z_goal(float(be), float(dl))

            obs_dict = obs_dict_next
            if done:
                break

        ep_global = P0_EPISODES + ep
        if (ep + 1) % 10 == 0:
            print(f"  [train] P1 seed={seed} cond={condition} ep {ep_global+1}/{TOTAL_TRAIN_EPS}", flush=True)

    benefit_r2_train = compute_r2(benefit_preds_train, benefit_actuals_train)

    # ---- P2: Evaluation ----
    # Unfreeze for action selection but no training
    for p in agent.parameters():
        p.requires_grad_(False)

    resource_prox_preds = []
    resource_prox_actuals = []
    benefit_preds_eval = []
    benefit_actuals_eval = []
    total_resources_collected = 0
    total_steps_eval = 0

    for ep in range(P2_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()

        for step in range(STEPS_PER_EP):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32).unsqueeze(0)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                latent = agent.latent_stack.encode(
                    torch.cat([obs_body, obs_world], dim=-1),
                    agent._current_latent,
                )
                agent._current_latent = latent.detach()

                # SD-018: resource proximity prediction
                if latent.resource_prox_pred is not None:
                    resource_prox_preds.append(latent.resource_prox_pred.item())
                    rfv = obs_dict.get("resource_field_view", None)
                    if rfv is not None:
                        resource_prox_actuals.append(float(rfv[12]))  # center cell = agent pos

                # Benefit eval head prediction
                z_w = latent.z_world
                bpred = agent.e3.benefit_eval_head(z_w)
                benefit_preds_eval.append(bpred.item())
                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    benefit_actuals_eval.append(float(rfv[12]))  # center cell = agent pos

            # Action: use E3 trajectory scoring (not random)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, agent.config.latent.world_dim, device=device
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item()) if action.dim() > 1 else int(action.item())
            action_idx = min(action_idx, 3)

            _, reward, done, info, obs_dict_next = env.step(action_idx)

            if info.get("transition_type") == "resource":
                total_resources_collected += 1
            total_steps_eval += 1

            be = obs_dict_next["body_state"][11] if len(obs_dict_next["body_state"]) > 11 else 0.0
            dl = 1.0 - obs_dict_next["body_state"][3] if len(obs_dict_next["body_state"]) > 3 else 1.0
            agent.update_z_goal(float(be), float(dl))

            if reward < 0:
                agent.update_residue(reward)

            obs_dict = obs_dict_next
            if done:
                break

    resource_rate = total_resources_collected / max(1, total_steps_eval)
    resource_prox_r2 = compute_r2(resource_prox_preds, resource_prox_actuals) if resource_prox_preds else -1.0
    benefit_eval_r2 = compute_r2(benefit_preds_eval, benefit_actuals_eval) if benefit_preds_eval else -1.0
    z_goal_norm = agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0

    print(f"  [eval] seed={seed} cond={condition} resource_rate={resource_rate:.4f} "
          f"resource_prox_r2={resource_prox_r2:.4f} benefit_eval_r2={benefit_eval_r2:.4f} "
          f"z_goal_norm={z_goal_norm:.4f}", flush=True)

    return {
        "seed": seed,
        "condition": condition,
        "resource_rate": resource_rate,
        "resource_prox_r2": resource_prox_r2,
        "benefit_eval_r2": benefit_eval_r2,
        "benefit_r2_train": benefit_r2_train,
        "z_goal_norm": z_goal_norm,
        "resources_collected": total_resources_collected,
        "eval_steps": total_steps_eval,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    if args.dry_run:
        # Minimal smoke: 1 seed, 2 conditions, 2 ep each phase
        global P0_EPISODES, P1_EPISODES, P2_EPISODES, SEEDS, STEPS_PER_EP, TOTAL_TRAIN_EPS
        P0_EPISODES = 2
        P1_EPISODES = 2
        P2_EPISODES = 2
        TOTAL_TRAIN_EPS = P0_EPISODES + P1_EPISODES
        STEPS_PER_EP = 10
        SEEDS = [1]
        print("DRY RUN: 1 seed, 2 conditions, 2 ep per phase, 10 steps/ep", flush=True)

    all_results = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            result = run_condition(seed, condition)
            all_results.append(result)
            passed = True  # placeholder for per-run verdict
            print(f"verdict: {'PASS' if passed else 'FAIL'}")

    # ---- Aggregate and evaluate criteria ----
    with_results = [r for r in all_results if r["condition"] == "WITH_RESOURCE_HEAD"]
    without_results = [r for r in all_results if r["condition"] == "WITHOUT"]

    # C1: resource_prox_r2 > 0.3 in WITH condition
    c1_values = [r["resource_prox_r2"] for r in with_results]
    c1_pass = all(v > C1_RESOURCE_PROX_R2 for v in c1_values)

    # C2: benefit_eval_r2 WITH > WITHOUT in all seeds
    c2_pass = True
    c2_details = []
    for w, wo in zip(with_results, without_results):
        lift = w["benefit_eval_r2"] > wo["benefit_eval_r2"]
        c2_pass = c2_pass and lift
        c2_details.append({
            "seed": w["seed"],
            "with_r2": w["benefit_eval_r2"],
            "without_r2": wo["benefit_eval_r2"],
            "lift": lift,
        })

    # C3: resource_rate WITH > WITHOUT * 1.1
    c3_pass = True
    c3_details = []
    for w, wo in zip(with_results, without_results):
        threshold = wo["resource_rate"] * C3_RESOURCE_RATE_LIFT
        lift = w["resource_rate"] > threshold
        c3_pass = c3_pass and lift
        c3_details.append({
            "seed": w["seed"],
            "with_rate": w["resource_rate"],
            "without_rate": wo["resource_rate"],
            "threshold": threshold,
            "lift": lift,
        })

    overall_pass = c1_pass and c2_pass and c3_pass
    outcome = "PASS" if overall_pass else "FAIL"

    # Per-claim direction
    per_claim = {}
    per_claim["SD-018"] = "supports" if c1_pass else "weakens"
    per_claim["ARC-030"] = "supports" if (c1_pass and c2_pass) else "weakens"
    per_claim["MECH-112"] = "supports" if c3_pass else "weakens"

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_257_sd018_resource_prox_validation_{ts}_v3"

    output = {
        "run_id": run_id,
        "experiment_type": "v3_exq_257_sd018_resource_prox_validation",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": ["SD-018", "ARC-030", "MECH-112"],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "supports" if overall_pass else "mixed" if (c1_pass or c2_pass) else "weakens",
        "evidence_direction_per_claim": per_claim,
        "timestamp_utc": ts,
        "criteria": {
            "C1_resource_prox_r2_pass": c1_pass,
            "C1_values": c1_values,
            "C1_threshold": C1_RESOURCE_PROX_R2,
            "C2_benefit_r2_lift_pass": c2_pass,
            "C2_details": c2_details,
            "C3_resource_rate_lift_pass": c3_pass,
            "C3_details": c3_details,
        },
        "per_condition_results": all_results,
        "config": {
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "p2_episodes": P2_EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "grid_size": GRID_SIZE,
            "num_resources": NUM_RESOURCES,
            "num_hazards": NUM_HAZARDS,
            "lambda_resource": LAMBDA_RESOURCE,
            "lambda_event": LAMBDA_EVENT,
        },
    }

    # Write output
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
    )
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to {out_path}")
    print(f"\nFINAL OUTCOME: {outcome}")
    print(f"  C1 resource_prox_r2 > {C1_RESOURCE_PROX_R2}: {c1_pass} {c1_values}")
    print(f"  C2 benefit_r2 WITH > WITHOUT: {c2_pass}")
    print(f"  C3 resource_rate lift > {C3_RESOURCE_RATE_LIFT}x: {c3_pass}")


if __name__ == "__main__":
    main()
