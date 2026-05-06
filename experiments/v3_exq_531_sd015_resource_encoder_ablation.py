#!/opt/local/bin/python3
"""V3-EXQ-531: SD-015 ResourceEncoder Ablation

Tests whether the SD-015 ResourceEncoder (world_obs -> z_resource) learns to predict
resource proximity when trained with the auxiliary resource_prox_head supervision.

Two-arm design:
  ARM_0: use_resource_encoder=False (baseline -- no ResourceEncoder)
  ARM_1: use_resource_encoder=True (ResourceEncoder active + proximity supervision)

Training protocol:
  - P0: 50 episodes random policy; ARM_1 computes resource_encoder_loss each step
  - P1: 20 eval episodes; record resource_prox_pred_r vs actual proximity

Metrics:
  resource_prox_r2  -- R^2 of resource_prox_pred_r vs max(resource_field_view)
  resource_prox_mae -- mean absolute error

Acceptance criteria:
  C1: ARM_1 resource_prox_r2 >= 0.5  (ResourceEncoder predicts resource proximity)
  C2: ARM_0 resource_prox_r2 is None (no encoder -> no prediction)

Overall PASS = C1

claim_ids: ["SD-015"]
experiment_purpose: evidence
architecture_epoch: "ree_hybrid_guardrails_v1"
"""

import json
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

EXPERIMENT_TYPE = "v3_exq_531_sd015_resource_encoder_ablation"
QUEUE_ID = "V3-EXQ-531"
CLAIM_IDS = ["SD-015"]

N_TRAIN_EPS = 50
N_EVAL_EPS  = 20
N_SEEDS     = 3
GRID_SIZE   = 12
N_STEPS     = 200
RESOURCE_ENC_LR = 5e-4

DRY_RUN = "--dry-run" in sys.argv
if DRY_RUN:
    N_TRAIN_EPS = 5
    N_EVAL_EPS  = 3
    N_SEEDS     = 1


def _onehot(idx: int, dim: int, device) -> torch.Tensor:
    t = torch.zeros(1, dim, device=device)
    t[0, idx] = 1.0
    return t


def run_arm(use_resource_encoder: bool, seed: int, device: str = "cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=4,
        num_resources=4,
        hazard_harm=0.02,
        reef_enabled=False,
    )

    cfg = REEConfig.from_dims(
        world_obs_dim=env.world_obs_dim,
        body_obs_dim=env.body_obs_dim,
        action_dim=env.action_dim,
        use_resource_proximity_head=False,  # SD-018 head on z_world -- keep OFF
        drive_weight=2.0,
    )
    # from_dims has no use_resource_encoder param; set directly on the latent config
    cfg.latent.use_resource_encoder = use_resource_encoder

    agent = REEAgent(cfg)
    agent.train()

    # Set up optimizer
    params = list(agent.parameters())
    optimizer = torch.optim.Adam(params, lr=RESOURCE_ENC_LR)

    # P0: Training phase
    resource_field_dim = 25  # 5x5 resource field

    for ep in range(N_TRAIN_EPS):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        for step in range(N_STEPS):
            obs_body = torch.tensor(
                obs_dict["body_state"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            obs_world = torch.tensor(
                obs_dict["world_state"], dtype=torch.float32, device=device
            ).unsqueeze(0)

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            action_idx = np.random.randint(0, env.action_dim)
            action = _onehot(action_idx, env.action_dim, device)
            _, _, done, info, obs_dict = env.step(action)

            # Main prediction loss
            pred_loss = agent.compute_prediction_loss()

            # Resource encoder loss (ARM_1 only)
            if use_resource_encoder and latent.resource_prox_pred_r is not None:
                resource_field = np.array(obs_dict.get("resource_field_view", [0.0] * resource_field_dim))
                resource_prox_target = float(np.max(resource_field))
                enc_loss = agent.compute_resource_encoder_loss(resource_prox_target, latent)
            else:
                enc_loss = torch.tensor(0.0)

            total_loss = pred_loss + enc_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

    # P1: Eval phase
    agent.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for ep in range(N_EVAL_EPS):
            flat_obs, obs_dict = env.reset()
            agent.reset()
            for step in range(N_STEPS):
                obs_body = torch.tensor(
                    obs_dict["body_state"], dtype=torch.float32, device=device
                ).unsqueeze(0)
                obs_world = torch.tensor(
                    obs_dict["world_state"], dtype=torch.float32, device=device
                ).unsqueeze(0)

                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                if use_resource_encoder and latent.resource_prox_pred_r is not None:
                    pred_val = float(latent.resource_prox_pred_r.item())
                    resource_field = np.array(obs_dict.get("resource_field_view", [0.0] * resource_field_dim))
                    target_val = float(np.max(resource_field))
                    preds.append(pred_val)
                    targets.append(target_val)

                action_idx = np.random.randint(0, env.action_dim)
                action = _onehot(action_idx, env.action_dim, device)
                _, _, done, info, obs_dict = env.step(action)
                if done:
                    break

    # Compute R^2
    if preds and len(preds) > 5:
        preds_np = np.array(preds)
        targets_np = np.array(targets)
        ss_res = np.sum((preds_np - targets_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        mae = float(np.mean(np.abs(preds_np - targets_np)))
        n_samples = len(preds)
    else:
        r2 = None
        mae = None
        n_samples = 0

    return {
        "resource_prox_r2": r2,
        "resource_prox_mae": mae,
        "n_samples": n_samples,
    }


def main():
    start_time = time.time()
    device = "cpu"

    arms = [
        ("ARM_0_no_encoder", False),
        ("ARM_1_resource_encoder", True),
    ]

    results = {}
    for arm_name, use_resource_encoder in arms:
        seed_results = []
        for seed in range(N_SEEDS):
            r = run_arm(use_resource_encoder, seed, device)
            seed_results.append(r)

        arm_r2s = [r["resource_prox_r2"] for r in seed_results if r["resource_prox_r2"] is not None]
        arm_maes = [r["resource_prox_mae"] for r in seed_results if r["resource_prox_mae"] is not None]

        results[arm_name] = {
            "seed_results": seed_results,
            "mean_r2": float(np.mean(arm_r2s)) if arm_r2s else None,
            "mean_mae": float(np.mean(arm_maes)) if arm_maes else None,
            "use_resource_encoder": use_resource_encoder,
        }

    # Evaluate criteria
    arm1 = results["ARM_1_resource_encoder"]
    arm0 = results["ARM_0_no_encoder"]

    arm1_r2 = arm1.get("mean_r2")
    arm0_r2 = arm0.get("mean_r2")

    C1 = arm1_r2 is not None and arm1_r2 >= 0.5
    C2 = arm0_r2 is None  # Baseline has no encoder

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
        "criteria": {
            "C1_arm1_r2_ge_0.5": C1,
            "C2_arm0_no_prediction": C2,
        },
        "arm1_r2": arm1_r2,
        "arm0_r2": arm0_r2,
        "arm1_mae": arm1.get("mean_mae"),
        "results": results,
        "config": {
            "n_train_eps": N_TRAIN_EPS,
            "n_eval_eps": N_EVAL_EPS,
            "n_seeds": N_SEEDS,
            "grid_size": GRID_SIZE,
            "dry_run": DRY_RUN,
        },
        "elapsed_seconds": elapsed,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
    }

    print(f"Outcome: {outcome}")
    print(f"C1 ARM_1 r2>=0.5: {C1} (arm1_r2={arm1_r2})")
    print(f"C2 ARM_0 no pred: {C2} (arm0_r2={arm0_r2})")
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
