#!/opt/local/bin/python3
"""V3-EXQ-534: SD-016 Frontal Cue Terrain Training

Tests whether SD-016 cue_terrain_proj (E1's terrain precision weighting head) can
be trained via supervised terrain_loss (LAMBDA_TERRAIN=0.1), reducing terrain_loss
compared to sd016 active but without terrain_loss supervision.

NOTE: EXP-0155 confirmed cue_ACTION_proj receives zero gradient (argmax non-differentiable).
This experiment targets only cue_TERRAIN_proj, which IS trainable via terrain_loss.

Two-arm design:
  ARM_0: sd016_enabled=True, NO terrain_loss (sd016 wired but cue_terrain_proj untrained)
  ARM_1: sd016_enabled=True, WITH terrain_loss (LAMBDA_TERRAIN=0.1, trains cue_terrain_proj)

Training protocol:
  - 40 episodes random policy, both arms
  - ARM_0: only compute_prediction_loss()
  - ARM_1: compute_prediction_loss() + LAMBDA_TERRAIN * terrain_loss
  - Last 10 episodes: record terrain_loss per step for evaluation

Metrics:
  terrain_loss_final  -- mean terrain_loss over final 10 episodes
  terrain_loss_trend  -- terrain_loss over time (train diagnostics)

Acceptance criteria:
  C1: ARM_1 terrain_loss_final < ARM_0 terrain_loss_final * 0.8
      (supervised training reduces terrain_loss by >=20%)

Overall PASS = C1

claim_ids: ["SD-016"]
experiment_purpose: evidence
architecture_epoch: "ree_hybrid_guardrails_v1"
"""

import json
import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

MANIFEST_WRITER_EXEMPT = "archival early-era manifest (non-canonical filename not provably == run_id.json; superseded lineage, not re-run)"

EXPERIMENT_TYPE = "v3_exq_534_sd016_cue_terrain_training"
QUEUE_ID = "V3-EXQ-534"
CLAIM_IDS = ["SD-016"]

N_TRAIN_EPS  = 40
N_SEEDS      = 3
GRID_SIZE    = 12
N_STEPS      = 200
LAMBDA_TERRAIN = 0.1
N_EVAL_EPS   = 10  # last N episodes for terrain_loss measurement

DRY_RUN = "--dry-run" in sys.argv
if DRY_RUN:
    N_TRAIN_EPS = 5
    N_SEEDS     = 1
    N_EVAL_EPS  = 2


def _onehot(idx: int, dim: int, device) -> torch.Tensor:
    t = torch.zeros(1, dim, device=device)
    t[0, idx] = 1.0
    return t


def compute_terrain_loss(agent: REEAgent, z_world: torch.Tensor, hazard_max: float) -> torch.Tensor:
    """Supervised terrain_loss for cue_terrain_proj (extract_cue_context WITH gradients)."""
    _, terrain_weight = agent.e1.extract_cue_context(z_world)
    w_harm_target = 0.8 if hazard_max > 0.3 else 0.2
    w_goal_target = 0.8 if hazard_max < 0.1 else 0.3
    target = torch.tensor(
        [[w_harm_target, w_goal_target]],
        dtype=terrain_weight.dtype,
        device=terrain_weight.device,
    )
    return F.mse_loss(terrain_weight, target)


def run_arm(use_terrain_loss: bool, seed: int, device: str = "cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=4,
        num_resources=4,
        hazard_harm=0.02,
        reef_enabled=True,
        n_reef_patches=3,
        reef_patch_radius=2,
        hazard_food_attraction=0.7,
    )

    cfg = REEConfig.from_dims(
        world_obs_dim=env.world_obs_dim,
        body_obs_dim=env.body_obs_dim,
        action_dim=env.action_dim,
        sd016_enabled=True,
        drive_weight=2.0,
    )

    agent = REEAgent(cfg)
    agent.train()

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    eval_start_ep = N_TRAIN_EPS - N_EVAL_EPS
    terrain_losses_eval = []
    terrain_losses_all = []

    for ep in range(N_TRAIN_EPS):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        ep_terrain_losses = []
        for step in range(N_STEPS):
            obs_body = torch.tensor(
                obs_dict["body_state"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            obs_world = torch.tensor(
                obs_dict["world_state"], dtype=torch.float32, device=device
            ).unsqueeze(0)

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            # Compute hazard context for terrain supervision
            hazard_field = np.array(obs_dict.get("hazard_field_view", [0.0] * 25))
            hazard_max = float(np.max(hazard_field))

            action_idx = np.random.randint(0, env.action_dim)
            action = _onehot(action_idx, env.action_dim, device)
            _, _, done, info, obs_dict = env.step(action)

            pred_loss = agent.compute_prediction_loss()

            if use_terrain_loss:
                t_loss = compute_terrain_loss(agent, latent.z_world, hazard_max)
                total_loss = pred_loss + LAMBDA_TERRAIN * t_loss
                # Record terrain_loss without gradient for diagnostics
                ep_terrain_losses.append(float(t_loss.detach()))
            else:
                # Still compute terrain_loss for measurement (no_grad), don't backprop it
                with torch.no_grad():
                    t_loss_val = compute_terrain_loss(agent, latent.z_world, hazard_max)
                total_loss = pred_loss
                ep_terrain_losses.append(float(t_loss_val))

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        ep_mean_t_loss = float(np.mean(ep_terrain_losses)) if ep_terrain_losses else 0.0
        terrain_losses_all.append(ep_mean_t_loss)
        if ep >= eval_start_ep:
            terrain_losses_eval.append(ep_mean_t_loss)

    terrain_loss_final = float(np.mean(terrain_losses_eval)) if terrain_losses_eval else float("inf")
    return {
        "terrain_loss_final": terrain_loss_final,
        "terrain_losses_all": terrain_losses_all,
        "use_terrain_loss": use_terrain_loss,
    }


def main():
    start_time = time.time()
    device = "cpu"

    arms = [
        ("ARM_0_no_terrain_loss", False),
        ("ARM_1_terrain_loss", True),
    ]

    results = {}
    for arm_name, use_terrain_loss in arms:
        seed_results = []
        for seed in range(N_SEEDS):
            r = run_arm(use_terrain_loss, seed, device)
            seed_results.append(r)

        mean_tl = float(np.mean([r["terrain_loss_final"] for r in seed_results]))
        results[arm_name] = {
            "seed_results": seed_results,
            "mean_terrain_loss_final": mean_tl,
            "use_terrain_loss": use_terrain_loss,
        }

    arm0 = results["ARM_0_no_terrain_loss"]
    arm1 = results["ARM_1_terrain_loss"]

    arm0_tl = arm0["mean_terrain_loss_final"]
    arm1_tl = arm1["mean_terrain_loss_final"]

    C1 = arm1_tl < arm0_tl * 0.8

    outcome = "PASS" if C1 else "FAIL"
    if DRY_RUN:
        outcome = "DRY_RUN_COMPLETE"

    elapsed = time.time() - start_time

    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "experiment_purpose": "evidence",
        "criteria": {
            "C1_arm1_terrain_lt_arm0_x0.8": C1,
        },
        "arm0_terrain_loss_final": arm0_tl,
        "arm1_terrain_loss_final": arm1_tl,
        "terrain_loss_ratio": arm1_tl / (arm0_tl + 1e-8),
        "note": (
            "Tests only cue_terrain_proj (trainable via terrain_loss). "
            "cue_action_proj has confirmed zero gradient (EXP-0155) and is excluded. "
            "PASS = terrain_loss reduced >=20% by explicit supervision."
        ),
        "results": results,
        "config": {
            "n_train_eps": N_TRAIN_EPS,
            "n_seeds": N_SEEDS,
            "grid_size": GRID_SIZE,
            "lambda_terrain": LAMBDA_TERRAIN,
            "n_eval_eps": N_EVAL_EPS,
            "reef_enabled": True,
            "hazard_food_attraction": 0.7,
            "dry_run": DRY_RUN,
        },
        "elapsed_seconds": elapsed,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
    }

    print(f"Outcome: {outcome}")
    print(f"C1 ARM_1 terrain<ARM_0*0.8: {C1} (arm0={arm0_tl:.4f}, arm1={arm1_tl:.4f})")
    print(f"Elapsed: {elapsed:.1f}s")

    if DRY_RUN:
        print("[DRY RUN] Not writing evidence.")
        return

    evidence_dir = Path(__file__).parent.parent.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
