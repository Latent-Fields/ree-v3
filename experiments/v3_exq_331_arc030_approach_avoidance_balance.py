#!/opt/local/bin/python3
"""
V3-EXQ-331: ARC-030 Approach-Avoidance Balance with Goal Seeding

experiment_purpose: evidence

Tests ARC-030 (Go/NoGo symmetric approach-avoidance evaluation).

Rationale: EXQ-138a failed because z_goal was never populated (SD-012
was not properly configured). With SD-012 working (drive_weight=2.0,
use_resource_proximity_head=True), the Go channel (benefit_eval_head)
now has something to score -- z_world contains resource proximity signal
and z_goal can be seeded on contact.

Design: With working goal seeding, does GOAL_HARM_JOINT (Go+NoGo symmetric)
produce better combined performance than HARM_ONLY (NoGo only)?

Two conditions per seed:
  JOINT:     benefit_eval_enabled=True, goal_weight=1.0, drive_weight=2.0,
             use_resource_proximity_head=True (Go + NoGo channels active)
  HARM_ONLY: benefit_eval_enabled=False, goal_weight=0.0, drive_weight=0.0,
             use_resource_proximity_head=True (NoGo channel only; proximity
             head still trained for fair encoder quality comparison)

Env: num_hazards=2, hazard_harm=0.1 (mild -- NOT 8 hazards which is volatile).
Standard phased training:
  P0 (100 ep): encoder + proximity head warmup (no goal/benefit scoring)
  P1 (100 ep): full pipeline (goal seeding + benefit_eval active in JOINT)
  P2 (50 ep):  evaluation (no training updates)

Pass criteria:
  C1: JOINT benefit_rate > HARM_ONLY benefit_rate (Go channel contributes, >= 2/3 seeds)
  C2: JOINT harm_rate <= HARM_ONLY harm_rate * 1.1 (avoidance not degraded, >= 2/3 seeds)
  C3: JOINT combined_score > HARM_ONLY combined_score (combined improvement, >= 2/3 seeds)
      where combined_score = benefit_rate - harm_rate

PASS: C1 AND C2 AND C3 across >= 2/3 seeds.

Claims: ARC-030
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
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
EXPERIMENT_TYPE    = "v3_exq_331_arc030_approach_avoidance_balance"
CLAIM_IDS          = ["ARC-030"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS      = [42, 7, 13]
CONDITIONS = ["JOINT", "HARM_ONLY"]

P0_EPISODES  = 100
P1_EPISODES  = 100
P2_EPISODES  = 50
STEPS_PER_EP = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 2    # standard -- NOT 8 (volatile/exploratory regime)
HAZARD_HARM   = 0.1  # mild

DRIVE_WEIGHT = 2.0
GOAL_WEIGHT  = 1.0

LR = 3e-4

# Pass thresholds
MIN_SEEDS_PASS  = 2
HARM_TOLERANCE  = 1.1  # JOINT harm_rate may be at most 10% higher than HARM_ONLY

DRY_RUN_EPISODES = 3
DRY_RUN_STEPS    = 20


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
    joint = (condition == "JOINT")
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_event_classifier=True,
        use_resource_proximity_head=True,   # both conditions (encoder fairness)
        resource_proximity_weight=0.5,
        # Go channel (JOINT only)
        benefit_eval_enabled=joint,
        benefit_weight=1.0 if joint else 0.0,
        # Goal / drive (JOINT only)
        z_goal_enabled=joint,
        drive_weight=DRIVE_WEIGHT if joint else 0.0,
        goal_weight=GOAL_WEIGHT if joint else 0.0,
        e1_goal_conditioned=joint,
        wanting_weight=0.3 if joint else 0.0,
        # NoGo channel (both conditions -- E3 harm_eval always active)
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_condition(
    seed: int,
    condition: str,
    dry_run: bool = False,
) -> Dict:
    total_p0  = DRY_RUN_EPISODES if dry_run else P0_EPISODES
    total_p1  = DRY_RUN_EPISODES if dry_run else P1_EPISODES
    total_p2  = DRY_RUN_EPISODES if dry_run else P2_EPISODES
    steps_per = DRY_RUN_STEPS    if dry_run else STEPS_PER_EP
    total_eps = total_p0 + total_p1 + total_p2

    print(f"  Seed {seed} Condition {condition}")

    env   = _make_env(seed)
    agent = _make_agent(env, condition, seed)
    device = agent.device

    optimizer = optim.Adam(list(agent.parameters()), lr=LR)

    p2_benefits:  List[float] = []
    p2_harms:     List[float] = []
    p2_resources: List[float] = []
    prev_ttype = "none"

    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()

        phase   = "P0" if ep < total_p0 else ("P1" if ep < total_p0 + total_p1 else "P2")
        in_eval = (phase == "P2")

        ep_benefit   = 0.0
        ep_harm      = 0.0
        ep_resources = 0

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
            action_idx = int(action.argmax(dim=-1).item())

            # Goal seeding (JOINT only, P1/P2)
            if condition == "JOINT" and phase != "P0":
                benefit_raw = float(obs_body.flatten()[11].item()) if obs_body.shape[-1] > 11 else 0.0
                drive_level = REEAgent.compute_drive_level(obs_body)
                agent.update_z_goal(benefit_raw, drive_level)

            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            harm_val    = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            benefit_val = max(0.0, float(info.get("benefit_exposure", 0.0)))

            ep_benefit   += benefit_val
            ep_harm      += harm_val
            if ttype == "resource":
                ep_resources += 1

            agent.update_residue(float(harm_signal))

            if z_self_prev is not None:
                agent.record_transition(z_self_prev, action, latent.z_self.detach())

            if not in_eval:
                optimizer.zero_grad()
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                loss    = e1_loss + e2_loss

                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    rp_t = float(rfv.max().item())
                    loss = loss + agent.compute_resource_proximity_loss(rp_t, latent)

                latent2 = agent.sense(obs_body, obs_world)
                ec_loss = agent.compute_event_contrastive_loss(prev_ttype, latent2)
                loss    = loss + ec_loss

                # Go channel: benefit_eval_head loss (JOINT only, P1)
                if condition == "JOINT" and phase == "P1":
                    benefit_t = torch.tensor([[benefit_val]], dtype=torch.float32, device=device)
                    bel = agent.compute_benefit_eval_loss(benefit_t)
                    loss = loss + bel

                if loss.requires_grad:
                    loss.backward()
                    import torch.nn as nn
                    nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                    optimizer.step()

            prev_ttype = ttype
            obs_dict   = obs_dict_next

            if done:
                break

        if in_eval:
            p2_benefits.append(ep_benefit / max(1, steps_per))
            p2_harms.append(ep_harm / max(1, steps_per))
            p2_resources.append(float(ep_resources))

        if (ep + 1) % 50 == 0:
            print(
                f"    [train] seed={seed} {condition} ep {ep+1}/{total_eps} "
                f"phase={phase} benefit={ep_benefit:.3f} harm={ep_harm:.3f}",
                flush=True,
            )

    benefit_rate  = float(np.mean(p2_benefits))  if p2_benefits  else 0.0
    harm_rate     = float(np.mean(p2_harms))      if p2_harms     else 0.0
    combined_score = benefit_rate - harm_rate
    resource_rate  = float(np.mean(p2_resources)) if p2_resources else 0.0

    verdict = "PASS" if benefit_rate > 0 else "FAIL"
    print(f"  verdict: {verdict} benefit={benefit_rate:.4f} harm={harm_rate:.4f} "
          f"combined={combined_score:.4f}")

    return {
        "seed": seed,
        "condition": condition,
        "benefit_rate": benefit_rate,
        "harm_rate": harm_rate,
        "combined_score": combined_score,
        "resource_rate": resource_rate,
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_cond[r["condition"]].append(r)

    joint_list = sorted(by_cond.get("JOINT",      []), key=lambda x: x["seed"])
    harm_list  = sorted(by_cond.get("HARM_ONLY",  []), key=lambda x: x["seed"])

    # C1: JOINT benefit_rate > HARM_ONLY benefit_rate
    c1_seeds = sum(
        j["benefit_rate"] > h["benefit_rate"]
        for j, h in zip(joint_list, harm_list)
    )
    c1_pass = c1_seeds >= MIN_SEEDS_PASS

    # C2: JOINT harm_rate <= HARM_ONLY harm_rate * HARM_TOLERANCE
    c2_seeds = sum(
        j["harm_rate"] <= h["harm_rate"] * HARM_TOLERANCE
        for j, h in zip(joint_list, harm_list)
    )
    c2_pass = c2_seeds >= MIN_SEEDS_PASS

    # C3: JOINT combined_score > HARM_ONLY combined_score
    c3_seeds = sum(
        j["combined_score"] > h["combined_score"]
        for j, h in zip(joint_list, harm_list)
    )
    c3_pass = c3_seeds >= MIN_SEEDS_PASS

    overall_pass = c1_pass and c2_pass and c3_pass
    return {
        "c1_benefit_improvement_pass": c1_pass,
        "c1_seeds_pass": c1_seeds,
        "c1_joint_benefits": [r["benefit_rate"] for r in joint_list],
        "c1_harm_only_benefits": [r["benefit_rate"] for r in harm_list],
        "c2_harm_preserved_pass": c2_pass,
        "c2_seeds_pass": c2_seeds,
        "c2_joint_harms": [r["harm_rate"] for r in joint_list],
        "c2_harm_only_harms": [r["harm_rate"] for r in harm_list],
        "c3_combined_score_pass": c3_pass,
        "c3_seeds_pass": c3_seeds,
        "c3_joint_scores": [r["combined_score"] for r in joint_list],
        "c3_harm_only_scores": [r["combined_score"] for r in harm_list],
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
        f"v3_exq_331_arc030_approach_avoidance_balance_dry_{ts}_v3"
        if args.dry_run
        else f"v3_exq_331_arc030_approach_avoidance_balance_{ts}_v3"
    )
    print(f"EXQ-331 start: {run_id}")

    all_results: List[Dict] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            result = run_condition(seed, condition, dry_run=args.dry_run)
            all_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome  = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== EXQ-331 {outcome} ===")
    print(f"C1 benefit_improvement: {criteria['c1_benefit_improvement_pass']} "
          f"joint={criteria['c1_joint_benefits']} harm_only={criteria['c1_harm_only_benefits']}")
    print(f"C2 harm_preserved: {criteria['c2_harm_preserved_pass']} "
          f"joint={criteria['c2_joint_harms']} harm_only={criteria['c2_harm_only_harms']}")
    print(f"C3 combined_score: {criteria['c3_combined_score_pass']}")

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
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
            "num_hazards": NUM_HAZARDS,
            "hazard_harm": HAZARD_HARM,
            "drive_weight": DRIVE_WEIGHT,
            "goal_weight": GOAL_WEIGHT,
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
