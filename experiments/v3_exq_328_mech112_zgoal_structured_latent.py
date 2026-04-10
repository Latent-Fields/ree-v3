#!/opt/local/bin/python3
"""
V3-EXQ-328: MECH-112 z_goal Structured Latent Discriminative Pair

experiment_purpose: evidence

Tests MECH-112 (z_goal as structured goal representation) and
SD-012 (homeostatic drive modulation of goal seeding).

Design: Does z_goal_norm stay elevated after resource contact when
drive_weight=2.0, and does it decay predictably? Is it absent
without drive?

Two conditions per seed:
  DRIVE_ACTIVE: drive_weight=2.0, z_goal_enabled=True, use_resource_proximity_head=True
  DRIVE_ABLATED: drive_weight=0.0, z_goal_enabled=True (no drive amplification)

After each resource contact event, sample z_goal_norm at steps
t=0, t=10, t=25, t=50 post-contact.

Training: 150 episodes encoder warmup, then 100 episodes of measurement.
The encoder must be trained enough to produce meaningful z_world
before goal seeding can work.

Pass criteria:
  C1: In DRIVE_ACTIVE, mean z_goal_norm post-contact > 0.1 at t=0
      (goal seeded at contact, >= 2/3 seeds)
  C2: In DRIVE_ACTIVE, mean z_goal_norm at t=50 > 0.05
      (goal persists, >= 2/3 seeds)
  C3: In DRIVE_ABLATED, mean z_goal_norm throughout < 0.05
      (goal absent without drive, >= 2/3 seeds)

PASS: C1 AND C2 AND C3 across >= 2/3 seeds.

Claims: MECH-112, SD-012
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
EXPERIMENT_TYPE    = "v3_exq_328_mech112_zgoal_structured_latent"
CLAIM_IDS          = ["MECH-112", "SD-012"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS      = [42, 7, 13]
CONDITIONS = ["DRIVE_ACTIVE", "DRIVE_ABLATED"]

P0_EPISODES  = 150    # encoder warmup (no goal seeding during warmup)
P1_EPISODES  = 100    # measurement phase (goal seeding active in DRIVE_ACTIVE)
STEPS_PER_EP = 200
POST_CONTACT_STEPS = [0, 10, 25, 50]

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 2
HAZARD_HARM   = 0.1
DRIVE_WEIGHT  = 2.0

LR = 3e-4

# Pass thresholds
C1_GOAL_NORM_AT_CONTACT  = 0.1   # z_goal_norm at t=0 must exceed this
C2_GOAL_NORM_AT_T50      = 0.05  # z_goal_norm at t=50 must exceed this
C3_GOAL_NORM_ABLATED_MAX = 0.05  # DRIVE_ABLATED z_goal_norm must stay below this
MIN_SEEDS_PASS           = 2

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
        proximity_benefit_scale=0.05,
        proximity_harm_scale=0.05,
        proximity_approach_threshold=0.15,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, condition: str, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    drive_on = (condition == "DRIVE_ACTIVE")
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
        z_goal_enabled=True,   # both conditions have z_goal enabled
        drive_weight=DRIVE_WEIGHT if drive_on else 0.0,
        goal_weight=0.0,       # not testing navigation here, just z_goal seeding
        benefit_eval_enabled=False,
        e1_goal_conditioned=True,
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
    steps_per = DRY_RUN_STEPS    if dry_run else STEPS_PER_EP
    total_eps = total_p0 + total_p1

    print(f"  Seed {seed} Condition {condition}")

    env   = _make_env(seed)
    agent = _make_agent(env, condition, seed)
    device = agent.device

    optimizer = optim.Adam(list(agent.parameters()), lr=LR)

    # Stores: for each resource contact, z_goal_norm trajectory
    contact_trajectories: List[Dict] = []
    ablated_norms: List[float] = []

    prev_ttype = "none"

    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()

        phase   = "P0" if ep < total_p0 else "P1"
        in_meas = (phase == "P1")

        # Track contact events within episode
        post_contact_buffer: Optional[Dict] = None  # {step_offset: norm}
        contact_step_in_ep: Optional[int] = None
        ep_step_count = 0

        for step in range(steps_per):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

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
            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            agent.update_residue(float(harm_signal))

            # Goal seeding on resource contact (P1 only)
            if in_meas and ttype == "resource":
                benefit_raw = float(obs_body.flatten()[11].item()) if obs_body.shape[-1] > 11 else 0.5
                drive_level = REEAgent.compute_drive_level(obs_body)
                agent.update_z_goal(benefit_raw, drive_level)
                # Record t=0 norm
                post_contact_buffer = {0: agent.goal_state.goal_norm() if agent.goal_state else 0.0}
                contact_step_in_ep = step

            # Record post-contact norms
            if in_meas and post_contact_buffer is not None and contact_step_in_ep is not None:
                offset = step - contact_step_in_ep
                if offset in POST_CONTACT_STEPS and offset not in post_contact_buffer:
                    norm_val = agent.goal_state.goal_norm() if agent.goal_state else 0.0
                    post_contact_buffer[offset] = norm_val
                # Close out trajectory if we have all needed offsets or reaching max
                if max(POST_CONTACT_STEPS) in post_contact_buffer:
                    contact_trajectories.append(dict(post_contact_buffer))
                    post_contact_buffer = None
                    contact_step_in_ep  = None

            # DRIVE_ABLATED: track z_goal_norm throughout (should stay low)
            if in_meas and condition == "DRIVE_ABLATED":
                norm_val = agent.goal_state.goal_norm() if agent.goal_state else 0.0
                ablated_norms.append(norm_val)

            if not in_meas:
                optimizer.zero_grad()
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                loss = e1_loss + e2_loss

                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    rp_t = float(rfv.max().item())
                    rp_loss = agent.compute_resource_proximity_loss(rp_t, latent)
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
            ep_step_count += 1

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"    [train] seed={seed} {condition} ep {ep+1}/{total_eps} "
                f"phase={phase} contacts={len(contact_trajectories)}",
                flush=True,
            )

    # Aggregate post-contact norms
    norms_by_offset: Dict[int, List[float]] = defaultdict(list)
    for traj in contact_trajectories:
        for offset, norm in traj.items():
            norms_by_offset[offset].append(norm)

    mean_norm_by_offset = {
        str(t): float(np.mean(norms_by_offset.get(t, [0.0])))
        for t in POST_CONTACT_STEPS
    }
    mean_ablated_norm = float(np.mean(ablated_norms)) if ablated_norms else 0.0

    c1_val = mean_norm_by_offset.get("0", 0.0)
    c2_val = mean_norm_by_offset.get("50", 0.0)

    print(f"  verdict: t0={c1_val:.4f} t50={c2_val:.4f} ablated={mean_ablated_norm:.4f}")

    return {
        "seed": seed,
        "condition": condition,
        "mean_norm_by_offset": mean_norm_by_offset,
        "mean_ablated_norm": mean_ablated_norm,
        "num_contact_events": len(contact_trajectories),
        "c1_norm_at_contact": c1_val,
        "c2_norm_at_t50": c2_val,
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_cond[r["condition"]].append(r)

    active_list  = sorted(by_cond.get("DRIVE_ACTIVE",  []), key=lambda x: x["seed"])
    ablated_list = sorted(by_cond.get("DRIVE_ABLATED", []), key=lambda x: x["seed"])

    c1_vals = [r["c1_norm_at_contact"] for r in active_list]
    c1_seeds = sum(v > C1_GOAL_NORM_AT_CONTACT for v in c1_vals)
    c1_pass  = c1_seeds >= MIN_SEEDS_PASS

    c2_vals = [r["c2_norm_at_t50"] for r in active_list]
    c2_seeds = sum(v > C2_GOAL_NORM_AT_T50 for v in c2_vals)
    c2_pass  = c2_seeds >= MIN_SEEDS_PASS

    c3_vals = [r["mean_ablated_norm"] for r in ablated_list]
    c3_seeds = sum(v < C3_GOAL_NORM_ABLATED_MAX for v in c3_vals)
    c3_pass  = c3_seeds >= MIN_SEEDS_PASS

    overall_pass = c1_pass and c2_pass and c3_pass
    return {
        "c1_goal_seeded_at_contact": c1_pass,
        "c1_vals": c1_vals,
        "c1_seeds_pass": c1_seeds,
        "c2_goal_persists": c2_pass,
        "c2_vals": c2_vals,
        "c2_seeds_pass": c2_seeds,
        "c3_ablated_absent": c3_pass,
        "c3_vals": c3_vals,
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
        f"v3_exq_328_mech112_zgoal_structured_latent_dry_{ts}_v3"
        if args.dry_run
        else f"v3_exq_328_mech112_zgoal_structured_latent_{ts}_v3"
    )
    print(f"EXQ-328 start: {run_id}")

    all_results: List[Dict] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            result = run_condition(seed, condition, dry_run=args.dry_run)
            all_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome  = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== EXQ-328 {outcome} ===")
    print(f"C1 goal_seeded: {criteria['c1_goal_seeded_at_contact']} "
          f"(vals={[f'{v:.4f}' for v in criteria['c1_vals']]})")
    print(f"C2 goal_persists: {criteria['c2_goal_persists']} "
          f"(vals={[f'{v:.4f}' for v in criteria['c2_vals']]})")
    print(f"C3 ablated_absent: {criteria['c3_ablated_absent']} "
          f"(vals={[f'{v:.4f}' for v in criteria['c3_vals']]})")

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction_per_claim": {
            "MECH-112": "supports" if criteria["overall_pass"] else "does_not_support",
            "SD-012":   "supports" if (criteria["c1_goal_seeded_at_contact"] and criteria["c3_ablated_absent"]) else "does_not_support",
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
            "steps_per_ep": STEPS_PER_EP,
            "drive_weight": DRIVE_WEIGHT,
            "post_contact_steps": POST_CONTACT_STEPS,
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
