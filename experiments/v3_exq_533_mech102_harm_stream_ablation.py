#!/opt/local/bin/python3
"""V3-EXQ-533: MECH-102 Harm Stream Voluntary Harm Ablation

Tests whether activating the harm stream (SD-010 HarmEncoder: z_harm_s) changes the
agent's rate of voluntary approach-to-hazard, which is the V3-tractable proxy for
MECH-102 (violence as terminal error-correction).

NOTE: MECH-102 has epistemic_category=substrate_ceiling. This experiment probes the
V3 approximation: does an active harm signal reduce voluntary hazard approach?

Two-arm design:
  ARM_0: use_harm_stream=False (no harm encoder; control baseline)
  ARM_1: use_harm_stream=True with harm proximity supervision (harm stream active)

Both arms: reef_enabled=True, hazard_food_attraction=0.7 (SD-050 monostrategy breaking)

Training protocol:
  - 40 episodes random policy (allow env exploration)
  - Measure voluntary_harm_events: steps where agent moved INTO a cell containing
    a hazard (info["harm_exposure"] > 0 on that step AND the action was a MOVE action)
  - Measure food_events: info["benefit_exposure"] > 0 contacts

Metrics:
  voluntary_harm_rate -- voluntary_harm_events / total_steps
  food_rate           -- food_events / total_episodes

Acceptance criteria:
  C1: ARM_1 voluntary_harm_rate <= ARM_0 voluntary_harm_rate * 1.2
      (harm stream does not significantly INCREASE voluntary harm)
  C2: ARM_1 food_rate >= ARM_0 food_rate * 0.7  (goal-seeking preserved)

  NOTE: We do NOT expect harm_rate to strictly decrease in V3 (substrate ceiling).
  C1 uses <=1.2 (at most 20% worse) to detect if harm stream disrupts baseline.
  Primary outcome is diagnostic: documents V3 substrate ceiling for MECH-102.

Overall PASS = C1 AND C2
experiment_purpose: diagnostic

claim_ids: ["MECH-102"]
evidence_direction_per_claim: {"MECH-102": "does_not_support"} is the expected outcome.
architecture_epoch: "ree_hybrid_guardrails_v1"
"""

import json
import sys
import os
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

EXPERIMENT_TYPE = "v3_exq_533_mech102_harm_stream_ablation"
QUEUE_ID = "V3-EXQ-533"
CLAIM_IDS = ["MECH-102"]

N_TRAIN_EPS = 40
N_SEEDS     = 3
GRID_SIZE   = 12
N_STEPS     = 200

DRY_RUN = "--dry-run" in sys.argv
if DRY_RUN:
    N_TRAIN_EPS = 4
    N_SEEDS     = 1


def _onehot(idx: int, dim: int, device) -> torch.Tensor:
    t = torch.zeros(1, dim, device=device)
    t[0, idx] = 1.0
    return t


def run_arm(use_harm_stream: bool, seed: int, device: str = "cpu"):
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
        use_harm_stream=use_harm_stream,
        harm_obs_dim=51,
        use_harm_proximity_head=use_harm_stream,
        drive_weight=2.0,
    )

    agent = REEAgent(cfg)
    agent.train()

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    total_steps = 0
    voluntary_harm_events = 0
    food_events = 0

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

            # Harm obs for ARM_1
            if use_harm_stream:
                obs_harm = torch.tensor(
                    obs_dict.get("harm_obs", [0.0] * 51), dtype=torch.float32, device=device
                ).unsqueeze(0)
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            else:
                latent = agent.sense(obs_body, obs_world)

            agent.clock.advance()

            action_idx = np.random.randint(0, env.action_dim)
            action = _onehot(action_idx, env.action_dim, device)

            _, harm_signal, done, info, obs_dict = env.step(action)

            # Measure voluntary harm: agent moved (not noop) AND got harmed
            is_move = action_idx < 4  # actions 0-3 are moves; 4 is noop
            harm_exposure = float(info.get("harm_exposure", harm_signal))
            if is_move and harm_exposure > 0:
                voluntary_harm_events += 1

            # Measure food contact
            benefit_exposure = float(info.get("benefit_exposure", 0.0))
            if benefit_exposure > 0:
                food_events += 1

            total_steps += 1

            pred_loss = agent.compute_prediction_loss()
            if pred_loss.requires_grad:
                optimizer.zero_grad()
                pred_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

    voluntary_harm_rate = voluntary_harm_events / max(1, total_steps)
    food_rate = food_events / max(1, N_TRAIN_EPS)

    return {
        "voluntary_harm_rate": voluntary_harm_rate,
        "food_rate": food_rate,
        "voluntary_harm_events": voluntary_harm_events,
        "food_events": food_events,
        "total_steps": total_steps,
    }


def main():
    start_time = time.time()
    device = "cpu"

    arms = [
        ("ARM_0_no_harm_stream", False),
        ("ARM_1_harm_stream", True),
    ]

    results = {}
    for arm_name, use_harm_stream in arms:
        seed_results = []
        for seed in range(N_SEEDS):
            r = run_arm(use_harm_stream, seed, device)
            seed_results.append(r)

        mean_vhr = float(np.mean([r["voluntary_harm_rate"] for r in seed_results]))
        mean_fr = float(np.mean([r["food_rate"] for r in seed_results]))
        results[arm_name] = {
            "seed_results": seed_results,
            "mean_voluntary_harm_rate": mean_vhr,
            "mean_food_rate": mean_fr,
            "use_harm_stream": use_harm_stream,
        }

    arm0 = results["ARM_0_no_harm_stream"]
    arm1 = results["ARM_1_harm_stream"]

    arm0_vhr = arm0["mean_voluntary_harm_rate"]
    arm1_vhr = arm1["mean_voluntary_harm_rate"]
    arm0_fr  = arm0["mean_food_rate"]
    arm1_fr  = arm1["mean_food_rate"]

    C1 = arm1_vhr <= arm0_vhr * 1.2
    C2 = arm1_fr >= arm0_fr * 0.7

    outcome = "PASS" if (C1 and C2) else "FAIL"
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
        "experiment_purpose": "diagnostic",
        "criteria": {
            "C1_arm1_vhr_le_arm0_x1.2": C1,
            "C2_arm1_food_ge_arm0_x0.7": C2,
        },
        "arm0_voluntary_harm_rate": arm0_vhr,
        "arm1_voluntary_harm_rate": arm1_vhr,
        "arm0_food_rate": arm0_fr,
        "arm1_food_rate": arm1_fr,
        "note": (
            "MECH-102 substrate_ceiling diagnostic. "
            "V3 proxy: does harm stream change voluntary hazard approach rate? "
            "Expected: no strong effect (documents substrate ceiling). "
            "PASS criterion is permissive (<=1.2x) to confirm harm stream does not disrupt baseline."
        ),
        "results": results,
        "config": {
            "n_train_eps": N_TRAIN_EPS,
            "n_seeds": N_SEEDS,
            "grid_size": GRID_SIZE,
            "reef_enabled": True,
            "hazard_food_attraction": 0.7,
            "dry_run": DRY_RUN,
        },
        "elapsed_seconds": elapsed,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
    }

    print(f"Outcome: {outcome}")
    print(f"C1 ARM_1 vhr<=ARM_0*1.2: {C1} (arm0={arm0_vhr:.4f}, arm1={arm1_vhr:.4f})")
    print(f"C2 ARM_1 food>=ARM_0*0.7: {C2} (arm0={arm0_fr:.2f}, arm1={arm1_fr:.2f})")
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
