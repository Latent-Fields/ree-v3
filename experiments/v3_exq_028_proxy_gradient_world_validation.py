"""
V3-EXQ-028: CausalGridWorldV2 Proxy-Gradient Field Validation

Tests ARC-024: does the new world generate observable harm/benefit gradient signals
that precede contact events?

This is a prerequisite validation before running SD-003 experiments on the new world.
Inverts the EXQ-006 failure condition:

EXQ-006 failure: mean_dz_world_agent_hazard < mean_dz_world_empty
  (stepping toward hazard produces LESS z_world change than empty locomotion)
  Root cause: binary contact-only harm → E2 can't learn approach dynamics

EXQ-028 PASS: mean_dz_world_hazard_approach > mean_dz_world_none
  (approach to hazard produces MORE z_world change than empty locomotion)
  Evidence that proxy fields create visible gradient signal in z_world

Criteria:
  C1 (gradient_dominance): mean_dz_world_hazard_approach > mean_dz_world_none
  C2 (gradient_magnitude): mean_dz_world_hazard_approach > 0.02
  C3 (benefit_gradient): mean_dz_world_benefit_approach > mean_dz_world_none  (if sufficient samples)
  C4 (ema_accumulation): mean_harm_exposure_at_contact > mean_harm_exposure_at_none
     (interoceptive EMA accumulates before hazard contact)

Architecture: ARC-024, INV-025, INV-026, INV-027
Claims: ARC-024
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ree_core.environment.causal_grid_world import CausalGridWorldV2


def run(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    env = CausalGridWorldV2(
        size=args.size,
        num_hazards=args.num_hazards,
        num_resources=args.num_resources,
        hazard_harm=args.harm_scale,
        env_drift_interval=args.drift_interval,
        env_drift_prob=0.1,
        seed=args.seed,
        hazard_field_decay=args.field_decay,
        resource_field_decay=args.field_decay,
        proximity_harm_scale=args.proximity_scale,
        proximity_benefit_scale=args.proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
    )

    print(f"\n=== V3-EXQ-028: CausalGridWorldV2 Proxy-Gradient Validation ===")
    print(f"body_obs_dim={env.body_obs_dim}, world_obs_dim={env.world_obs_dim}")
    print(f"hazard_field_decay={args.field_decay}, proximity_scale={args.proximity_scale}")
    print(f"Running {args.episodes} episodes × {args.steps} steps")

    # Collect raw z_world changes per transition type
    dz_world_by_type = defaultdict(list)
    harm_exposure_by_type = defaultdict(list)
    harm_signal_by_type = defaultdict(list)
    field_val_by_type = defaultdict(list)
    counts = defaultdict(int)

    prev_world = None
    total_steps = 0

    for ep in range(args.episodes):
        flat, obs_dict = env.reset()
        prev_world = obs_dict["world_state"].numpy().copy()

        for step in range(args.steps):
            action = rng.integers(0, env.action_dim)
            flat2, hs, done, info, obs_dict2 = env.step(torch.tensor(action))

            tt = info["transition_type"]
            curr_world = obs_dict2["world_state"].numpy()

            dz = float(np.mean(np.abs(curr_world - prev_world)))
            dz_world_by_type[tt].append(dz)
            harm_exposure_by_type[tt].append(info.get("harm_exposure", 0.0))
            harm_signal_by_type[tt].append(abs(hs))
            field_val_by_type[tt].append(info.get("hazard_field_at_agent", 0.0))
            counts[tt] += 1

            prev_world = curr_world
            total_steps += 1

            if done:
                break

    print(f"\nTotal steps: {total_steps}")
    print(f"\nTransition counts: {dict(counts)}")

    means = {}
    stds = {}
    for tt, vals in dz_world_by_type.items():
        means[tt] = float(np.mean(vals)) if vals else 0.0
        stds[tt] = float(np.std(vals)) if vals else 0.0
        print(f"  {tt:25s}: mean_dz={means[tt]:.4f} ± {stds[tt]:.4f}  "
              f"harm_sig={np.mean(harm_signal_by_type[tt]):.4f}  "
              f"field={np.mean(field_val_by_type[tt]):.3f}  "
              f"exposure={np.mean(harm_exposure_by_type[tt]):.4f}  n={counts[tt]}")

    # --- Criteria evaluation ---
    none_dz = means.get("none", 0.0)
    hazard_approach_dz = means.get("hazard_approach", 0.0)
    benefit_approach_dz = means.get("benefit_approach", 0.0)
    contact_dz = means.get("env_caused_hazard", 0.0)

    n_approach = counts.get("hazard_approach", 0)
    n_benefit = counts.get("benefit_approach", 0)
    n_none = counts.get("none", 0)

    # C1: approach produces more z_world change than locomotion
    c1_pass = hazard_approach_dz > none_dz and n_approach >= 50
    # C2: gradient magnitude is non-trivial
    c2_pass = hazard_approach_dz > 0.02 and n_approach >= 50
    # C3: benefit gradient visible (optional — may not be enough samples)
    c3_pass = (benefit_approach_dz > none_dz and n_benefit >= 20) if n_benefit >= 20 else None
    # C4: harm_exposure EMA accumulates on approach vs none
    exposure_approach = np.mean(harm_exposure_by_type.get("hazard_approach", [0.0]))
    exposure_none = np.mean(harm_exposure_by_type.get("none", [0.0]))
    c4_pass = exposure_approach > exposure_none and n_approach >= 50

    print(f"\n--- Criteria ---")
    print(f"C1 (gradient_dominance):  hazard_approach_dz({hazard_approach_dz:.4f}) > none_dz({none_dz:.4f})  → {'PASS' if c1_pass else 'FAIL'}")
    print(f"C2 (gradient_magnitude):  hazard_approach_dz({hazard_approach_dz:.4f}) > 0.02               → {'PASS' if c2_pass else 'FAIL'}")
    if c3_pass is not None:
        print(f"C3 (benefit_gradient):    benefit_approach_dz({benefit_approach_dz:.4f}) > none_dz({none_dz:.4f})  → {'PASS' if c3_pass else 'FAIL'}")
    else:
        print(f"C3 (benefit_gradient):    skipped (n_benefit={n_benefit} < 20)")
    print(f"C4 (ema_accumulation):    exposure_approach({exposure_approach:.4f}) > exposure_none({exposure_none:.4f})  → {'PASS' if c4_pass else 'FAIL'}")

    required_pass = c1_pass and c2_pass and c4_pass
    final_verdict = "PASS" if required_pass else "FAIL"
    print(f"\nFinal verdict: {final_verdict}")

    # --- Write run pack ---
    run_id = f"v3_exq_028_proxy_gradient_world_validation_s{args.seed}_v3"
    run_pack = {
        "run_id": run_id,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "queue_id": "V3-EXQ-028",
        "claim_id": "ARC-024",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "config": {
            "size": args.size,
            "num_hazards": args.num_hazards,
            "num_resources": args.num_resources,
            "harm_scale": args.harm_scale,
            "proximity_scale": args.proximity_scale,
            "field_decay": args.field_decay,
            "episodes": args.episodes,
            "steps": args.steps,
        },
        "metrics": {
            "mean_dz_world_none": none_dz,
            "mean_dz_world_hazard_approach": hazard_approach_dz,
            "mean_dz_world_benefit_approach": benefit_approach_dz,
            "mean_dz_world_env_caused_hazard": contact_dz,
            "exposure_approach": float(exposure_approach),
            "exposure_none": float(exposure_none),
            "n_hazard_approach": n_approach,
            "n_benefit_approach": n_benefit,
            "n_none": n_none,
            "total_steps": total_steps,
            "transition_counts": dict(counts),
        },
        "criteria": {
            "C1_gradient_dominance": bool(c1_pass),
            "C2_gradient_magnitude": bool(c2_pass),
            "C3_benefit_gradient": bool(c3_pass) if c3_pass is not None else None,
            "C4_ema_accumulation": bool(c4_pass),
        },
        "final_verdict": final_verdict,
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(run_pack, f, indent=2)
    print(f"\nRun pack written: {out_path}")
    return run_pack


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--num-hazards", type=int, default=4)
    parser.add_argument("--num-resources", type=int, default=5)
    parser.add_argument("--harm-scale", type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--field-decay", type=float, default=0.5)
    parser.add_argument("--drift-interval", type=int, default=10)
    args = parser.parse_args()
    run(args)
