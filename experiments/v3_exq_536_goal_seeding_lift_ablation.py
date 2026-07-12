#!/opt/local/bin/python3
"""V3-EXQ-536: Goal Seeding Lift Ablation

Tests whether the full goal seeding pipeline (SD-018 + z_goal seeding + benefit_eval_head
+ MECH-295 liking bridge) produces measurable approach-commit lift versus a baseline
with all three components disabled.

Claims: ARC-030, MECH-112, SD-018, SD-012, MECH-295

=== PIPELINE UNDER TEST ===

Full pipeline (ARM_1):
  1. SD-018 resource_proximity_head: trains z_world encoder to represent resource
     proximity (max resource_field_view regression). Without this, benefit_eval_head
     and z_goal operate on scene noise (EXQ-085m finding: R2=-0.004).
  2. benefit_eval_head (ARC-030/MECH-112): Go channel in E3.score_trajectory().
     Subtracted from E3 cost, biasing E3 toward resource-proximal trajectories.
  3. z_goal seeding (MECH-112): at resource contact (benefit_exposure > threshold),
     z_goal is seeded from z_world. E3 score_trajectory() subtracts goal_weight *
     goal_proximity. Goal_proximity = 1/(1+dist(z_world_traj, z_goal)).
  4. MECH-295 liking bridge: adds anticipatory approach_cue score_bias at E3 select()
     based on drive_level * per-candidate goal proximity. Sign convention: negative
     bias (lower = better), so approach-congruent candidates are favoured.

=== KNOWN WIRING BUG FIXED HERE ===

from_dims(goal_weight=1.0) sets config.goal.goal_weight (GoalConfig) only.
E3Selector reads self.config.goal_weight (E3Config), which defaults to 0.0.
Fix: cfg.e3.goal_weight = 1.0 must be set explicitly in ARM_1.

=== DESIGN ===

Two arms, 3 seeds each:
  ARM_0 (baseline): SD-018=OFF, z_goal_enabled=False, benefit_eval_enabled=False,
                    MECH-295=OFF. E3 uses only harm + residue scoring.
  ARM_1 (pipeline): SD-018=ON, z_goal_enabled=True, benefit_eval_enabled=True,
                    MECH-295=ON, cfg.e3.goal_weight=1.0.

Training phase (random policy):
  ARM_0: E1+E2 prediction loss only.
  ARM_1: E1+E2 loss + resource_proximity_loss (SD-018) + update_z_goal when
         benefit_exposure > 0 + record_benefit_sample() to pass warmup gate.

Eval phase (full E3 selection):
  Both arms: generate_trajectories + select_action per step.
  ARM_1: update_z_goal when benefit_exposure > 0.
  Per-episode: goal_state is explicitly reset for clean within-episode measurement.

BreathOscillator: both arms use breath_period=50 (from_dims default). This
periodically forces uncommitted windows so E3 re-evaluates trajectories.

=== METRICS ===

approach_commit_rate = fraction of eval steps where agent is committed
                       AND resource_proximity(env) >= APPROACH_THRESH (0.25).

goal_lift = ARM_1.approach_commit_rate / ARM_0.approach_commit_rate

Auxiliary:
  benefit_eval_r2: R^2 of benefit_eval_head predictions vs max(resource_field_view)
  z_goal_active_fraction: fraction of eval steps where goal_state.is_active()

=== PASS CRITERIA ===

C1: ARM_1 benefit_eval_r2 > 0.3       (SD-018 training successfully represents resource proximity in z_world)
C2: ARM_1 z_goal_active_fraction > 0.2 (z_goal is being seeded from resource contacts)
C3: goal_lift >= 1.5 in >= 2/3 seeds   (pipeline produces measurable approach-commit lift)

Overall PASS = C1 AND C2 AND C3

claim_ids: ["ARC-030", "MECH-112", "SD-018", "SD-012", "MECH-295"]
experiment_purpose: evidence
architecture_epoch: "ree_hybrid_guardrails_v1"
"""

import json
import sys
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_536_goal_seeding_lift_ablation"
QUEUE_ID = "V3-EXQ-536"
CLAIM_IDS = ["ARC-030", "MECH-112", "SD-018", "SD-012", "MECH-295"]

N_TRAIN_EPS   = 100
N_EVAL_EPS    = 40
N_STEPS       = 200
N_SEEDS       = 3
GRID_SIZE     = 12
APPROACH_THRESH = 0.25   # resource_proximity threshold for "near resource"

DRY_RUN = "--dry-run" in sys.argv
if DRY_RUN:
    N_TRAIN_EPS = 8
    N_EVAL_EPS  = 5
    N_SEEDS     = 1


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _obs_tensors(obs_dict):
    body  = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    return body, world


def _resource_proximity(env) -> float:
    """1 / (1 + min Manhattan dist to nearest resource). 0.0 if no resources."""
    if not env.resources:
        return 0.0
    ax, ay = env.agent_x, env.agent_y
    min_dist = min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)
    return 1.0 / (1.0 + min_dist)


def _benefit_drive(obs_dict):
    """Return (benefit_exposure, drive_level) from obs_dict body_state."""
    body_raw = obs_dict["body_state"]
    benefit  = float(body_raw[11].item()) if body_raw.numel() > 11 else 0.0
    energy   = float(body_raw[3].item())  if body_raw.numel() > 3  else 0.5
    drive    = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive


def _r_squared(pred: List[float], actual: List[float]) -> float:
    if len(pred) < 4:
        return 0.0
    mean_a = sum(actual) / len(actual)
    ss_tot = sum((a - mean_a) ** 2 for a in actual)
    ss_res = sum((p - a) ** 2 for p, a in zip(pred, actual))
    if ss_tot < 1e-9:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=4,
        num_resources=4,
        hazard_harm=0.02,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        reef_enabled=False,
    )


def _make_agent(env, use_goal_pipeline: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        world_obs_dim=env.world_obs_dim,
        body_obs_dim=env.body_obs_dim,
        action_dim=env.action_dim,
        use_resource_proximity_head=use_goal_pipeline,  # SD-018
        drive_weight=2.0,
        z_goal_enabled=use_goal_pipeline,
        benefit_eval_enabled=use_goal_pipeline,
        benefit_weight=2.0,
        use_mech295_liking_bridge=use_goal_pipeline,
        mech295_drive_to_liking_gain=1.0,
        mech295_liking_to_approach_cue_gain=0.5,
        # BreathOscillator: breath_period=50 is the from_dims default;
        # both arms get it (enables re-evaluation of trajectories).
    )
    # BUG FIX: from_dims routes goal_weight to config.goal (GoalConfig) only.
    # E3Selector reads config.e3.goal_weight, which defaults to 0.0.
    # Must set explicitly so goal proximity scoring is active in ARM_1.
    if use_goal_pipeline:
        cfg.e3.goal_weight = 1.0
    return REEAgent(cfg)


# ------------------------------------------------------------------ #
# Run one arm, one seed                                                #
# ------------------------------------------------------------------ #

def run_arm(use_goal_pipeline: bool, seed: int) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env   = _make_env(seed)
    agent = _make_agent(env, use_goal_pipeline)

    optimizer = optim.Adam(agent.parameters(), lr=3e-4)

    # Separate benefit_eval optimizer (ARM_1 only)
    benefit_eval_optimizer = None
    benefit_pred_vals:   List[float] = []
    benefit_target_vals: List[float] = []

    if use_goal_pipeline:
        benefit_eval_optimizer = optim.Adam(
            list(agent.e3.benefit_eval_head.parameters()), lr=1e-4
        )

    world_dim = agent.config.latent.world_dim

    # --------------------------------------------------------
    # Training phase: random policy
    # --------------------------------------------------------
    agent.train()

    for ep in range(N_TRAIN_EPS):
        _, obs_dict = env.reset()
        agent.reset()

        for _step in range(N_STEPS):
            body, world = _obs_tensors(obs_dict)

            # Keep latent for gradient flow into resource_proximity_head (SD-018).
            # agent.sense() also caches a detached copy in agent._current_latent.
            latent = agent.sense(obs_body=body, obs_world=world)
            agent.clock.advance()

            # Random action
            action_int = random.randint(0, env.action_dim - 1)
            action_oh = torch.zeros(1, env.action_dim)
            action_oh[0, action_int] = 1.0
            agent._last_action = action_oh

            _, _harm, done, _info, obs_dict = env.step(action_oh)

            benefit, drive = _benefit_drive(obs_dict)

            # Compute all losses
            pred_loss = agent.compute_prediction_loss()
            e2_loss   = agent.compute_e2_loss()
            total_loss = pred_loss + e2_loss

            if use_goal_pipeline:
                # SD-018: resource proximity regression on z_world encoder.
                # Use latent from sense() (has gradient), not agent._current_latent (detached).
                resource_field = obs_dict.get("resource_field_view", None)
                if resource_field is not None:
                    prox_target = float(resource_field.max().item())
                else:
                    prox_target = _resource_proximity(env)
                prox_loss = agent.compute_resource_proximity_loss(prox_target, latent)
                total_loss = total_loss + prox_loss

                # benefit_eval_head: resource proximity regression on z_world
                with torch.no_grad():
                    z_world_det = latent.z_world.detach()
                benefit_pred_train = agent.e3.benefit_eval_head(z_world_det)
                prox_t = torch.tensor([[prox_target]], dtype=torch.float32)
                b_loss = F.mse_loss(benefit_pred_train, prox_t)
                benefit_pred_vals.append(float(benefit_pred_train.item()))
                benefit_target_vals.append(prox_target)
                if len(benefit_pred_vals) > 2000:
                    benefit_pred_vals  = benefit_pred_vals[-2000:]
                    benefit_target_vals = benefit_target_vals[-2000:]

                # Warmup gate: must call record_benefit_sample to allow
                # benefit scoring in eval (gate threshold: 50 samples).
                agent.e3.record_benefit_sample(1)

                # MECH-112: seed z_goal from z_world at resource contact.
                if benefit > 0.01:
                    agent.update_z_goal(benefit, drive)

                # Train benefit_eval_head separately (own optimizer)
                if benefit_eval_optimizer is not None and b_loss.requires_grad:
                    benefit_eval_optimizer.zero_grad()
                    b_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e3.benefit_eval_head.parameters()), 0.5
                    )
                    benefit_eval_optimizer.step()

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        if not DRY_RUN and (ep + 1) % 25 == 0:
            arm_tag = "ARM_1" if use_goal_pipeline else "ARM_0"
            r2 = _r_squared(benefit_pred_vals[-500:], benefit_target_vals[-500:]) if benefit_pred_vals else 0.0
            goal_active = (
                agent.goal_state is not None and agent.goal_state.is_active()
            ) if use_goal_pipeline else False
            print(
                f"  [{arm_tag} seed={seed}] ep {ep+1}/{N_TRAIN_EPS}"
                f" benefit_r2={r2:.3f} z_goal_active={goal_active}",
                flush=True,
            )

    # Final benefit_eval R^2 from last 1000 training samples
    benefit_eval_r2 = _r_squared(benefit_pred_vals[-1000:], benefit_target_vals[-1000:]) if benefit_pred_vals else None

    # --------------------------------------------------------
    # Eval phase: full E3 selection
    # --------------------------------------------------------
    agent.eval()

    approach_commit_steps = 0
    total_eval_steps      = 0
    z_goal_active_steps   = 0
    resource_contacts     = 0

    for _ep in range(N_EVAL_EPS):
        _, obs_dict = env.reset()
        agent.reset()

        # Reset z_goal per episode so measurement is purely within-episode.
        # goal_state is NOT cleared by agent.reset() -- must be explicit.
        if use_goal_pipeline and agent.goal_state is not None:
            agent.goal_state.reset()

        for _step in range(N_STEPS):
            body, world = _obs_tensors(obs_dict)

            agent.sense(obs_body=body, obs_world=world)
            ticks = agent.clock.advance()

            e1_prior = (
                agent._e1_tick(agent._current_latent)
                if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(
                agent._current_latent, e1_prior, ticks
            )
            _action_tensor = agent.select_action(candidates, ticks)

            # Sample metrics before env step
            is_committed  = agent.e3._committed_trajectory is not None
            rp            = _resource_proximity(env)
            is_near_rsc   = rp >= APPROACH_THRESH

            total_eval_steps += 1
            if is_committed and is_near_rsc:
                approach_commit_steps += 1
            if use_goal_pipeline and agent.goal_state is not None:
                if agent.goal_state.is_active():
                    z_goal_active_steps += 1

            action_int = int(_action_tensor.argmax(dim=-1).item())
            action_oh  = torch.zeros(1, env.action_dim)
            action_oh[0, action_int] = 1.0

            _, _harm, done, info, obs_dict = env.step(action_oh)

            ttype = info.get("transition_type", "none")
            if ttype in ("resource", "benefit_approach"):
                resource_contacts += 1

            if use_goal_pipeline:
                benefit, drive = _benefit_drive(obs_dict)
                if benefit > 0.01:
                    agent.update_z_goal(benefit, drive)

            if done:
                break

    approach_commit_rate = (
        float(approach_commit_steps) / max(1, total_eval_steps)
    )
    z_goal_active_fraction = (
        float(z_goal_active_steps) / max(1, total_eval_steps)
        if use_goal_pipeline else None
    )
    resource_contacts_per_ep = float(resource_contacts) / max(1, N_EVAL_EPS)

    return {
        "approach_commit_rate":      approach_commit_rate,
        "z_goal_active_fraction":    z_goal_active_fraction,
        "benefit_eval_r2":           benefit_eval_r2,
        "resource_contacts_per_ep":  resource_contacts_per_ep,
        "total_eval_steps":          total_eval_steps,
        "approach_commit_steps":     approach_commit_steps,
    }


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    start_time = time.time()

    print("V3-EXQ-536 goal seeding lift ablation", flush=True)
    print(f"DRY_RUN={DRY_RUN} N_TRAIN={N_TRAIN_EPS} N_EVAL={N_EVAL_EPS} N_SEEDS={N_SEEDS}", flush=True)

    per_seed_lifts = []
    arm0_rates     = []
    arm1_rates     = []
    arm1_r2s       = []
    arm1_z_fracs   = []

    seed_details   = []

    for seed in range(N_SEEDS):
        print(f"\n-- Seed {seed} --", flush=True)

        r0 = run_arm(use_goal_pipeline=False, seed=seed)
        r1 = run_arm(use_goal_pipeline=True,  seed=seed)

        acr0 = r0["approach_commit_rate"]
        acr1 = r1["approach_commit_rate"]
        lift  = acr1 / max(acr0, 1e-6)

        arm0_rates.append(acr0)
        arm1_rates.append(acr1)
        per_seed_lifts.append(lift)

        if r1["benefit_eval_r2"] is not None:
            arm1_r2s.append(r1["benefit_eval_r2"])
        if r1["z_goal_active_fraction"] is not None:
            arm1_z_fracs.append(r1["z_goal_active_fraction"])

        seed_details.append({
            "seed": seed,
            "arm0": r0,
            "arm1": r1,
            "goal_lift": float(lift),
        })

        print(
            f"  ARM_0 approach_commit_rate={acr0:.4f}"
            f"  ARM_1 approach_commit_rate={acr1:.4f}"
            f"  lift={lift:.3f}",
            flush=True,
        )
        print(
            f"  ARM_1 benefit_eval_r2={r1['benefit_eval_r2']}"
            f"  z_goal_active_fraction={r1['z_goal_active_fraction']:.3f}"
            if r1["z_goal_active_fraction"] is not None else "",
            flush=True,
        )

    mean_lift    = float(np.mean(per_seed_lifts))
    seeds_pass   = sum(1 for l in per_seed_lifts if l >= 1.5)
    mean_r2      = float(np.mean(arm1_r2s)) if arm1_r2s else None
    mean_z_frac  = float(np.mean(arm1_z_fracs)) if arm1_z_fracs else None

    C1 = mean_r2 is not None and mean_r2 > 0.3
    C2 = mean_z_frac is not None and mean_z_frac > 0.2
    C3 = seeds_pass >= 2

    outcome = "PASS" if (C1 and C2 and C3) else "FAIL"
    if DRY_RUN:
        outcome = "DRY_RUN_COMPLETE"

    elapsed = time.time() - start_time

    print(f"\nOutcome: {outcome}", flush=True)
    print(f"C1 benefit_eval_r2>0.3: {C1} (mean_r2={mean_r2})", flush=True)
    print(f"C2 z_goal_active>0.2:   {C2} (mean_frac={mean_z_frac})", flush=True)
    print(f"C3 lift>=1.5 in >=2/3:  {C3} ({seeds_pass}/3 seeds, mean_lift={mean_lift:.3f})", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)

    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "criteria": {
            "C1_benefit_eval_r2_gt_0.3": C1,
            "C2_z_goal_active_fraction_gt_0.2": C2,
            "C3_goal_lift_ge_1.5_in_2of3_seeds": C3,
        },
        "mean_goal_lift":           mean_lift,
        "seeds_passing_lift":       seeds_pass,
        "per_seed_lifts":           per_seed_lifts,
        "arm1_mean_benefit_eval_r2": mean_r2,
        "arm1_mean_z_goal_active_fraction": mean_z_frac,
        "arm0_mean_approach_commit_rate": float(np.mean(arm0_rates)),
        "arm1_mean_approach_commit_rate": float(np.mean(arm1_rates)),
        "seed_details": seed_details,
        "config": {
            "n_train_eps":      N_TRAIN_EPS,
            "n_eval_eps":       N_EVAL_EPS,
            "n_steps":          N_STEPS,
            "n_seeds":          N_SEEDS,
            "grid_size":        GRID_SIZE,
            "approach_thresh":  APPROACH_THRESH,
            "dry_run":          DRY_RUN,
            "e3_goal_weight_fix": "cfg.e3.goal_weight=1.0 (from_dims does not set E3Config.goal_weight)",
        },
        "elapsed_seconds":  elapsed,
        "generated_utc":    datetime.utcnow().isoformat() + "Z",
    }

    if DRY_RUN:
        print("[DRY RUN] Not writing evidence.", flush=True)
        return

    evidence_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments"
        / EXPERIMENT_TYPE
    )
    manifest_path = write_flat_manifest(
        manifest,
        evidence_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Manifest written: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
