"""
V3-EXQ-576: GAP-1 harm gradient env feature substrate validation.
ARM_0: harm_gradient_enabled=False -- no gradient reward fires.
ARM_1: harm_gradient_enabled=True  -- gradient reward fires and is negative.
PASS = C1 (ARM_1 mean_gradient_reward < THRESH_C1) AND C2 (ARM_0 == 0.0) across all seeds.
experiment_purpose: diagnostic (substrate readiness test, not a claim hypothesis test)
claim_ids: [] (env feature validation only)
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiment_protocol import emit_outcome
from ree_core.environment.causal_grid_world import CausalGridWorldV2

EXPERIMENT_PURPOSE = "diagnostic"
QUEUE_ID = "V3-EXQ-576"

SEEDS = [0, 1, 2]
N_EPISODES = 100
N_STEPS = 200
OUTER_RADIUS = 3.0
INNER_RADIUS = 0.0
SCALE = 1.0

THRESH_C1 = -1e-4   # ARM_1 mean gradient reward must be below this (negative)
THRESH_C2 = 0.0     # ARM_0 must be exactly 0.0

CONDITIONS = [
    ("ARM_0_disabled", False),
    ("ARM_1_enabled", True),
]


def run_experiment(n_episodes, dry_run=False):
    results_by_seed = {}
    c1_by_seed = []
    c2_by_seed = []

    for seed in SEEDS:
        results_by_seed[seed] = {}
        for cond_label, gradient_enabled in CONDITIONS:
            print(f"Seed {seed} Condition {cond_label}", flush=True)

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            env = CausalGridWorldV2(
                size=12,
                seed=seed,
                num_hazards=1,
                use_proxy_fields=False,
                harm_gradient_enabled=gradient_enabled,
                harm_gradient_outer_radius=OUTER_RADIUS,
                harm_gradient_inner_radius=INNER_RADIUS,
                harm_gradient_scale=SCALE,
            )

            total_gradient_reward = 0.0
            total_steps = 0
            n_gradient_fires = 0

            for ep in range(n_episodes):
                env.reset()
                done = False
                for _ in range(N_STEPS):
                    if done:
                        break
                    action = np.random.randint(0, 4)
                    _, _, done, info, _ = env.step(action)
                    gr = info["harm_gradient_reward_this_tick"]
                    total_gradient_reward += gr
                    total_steps += 1
                    if gr != 0.0:
                        n_gradient_fires += 1

                print_interval = max(1, n_episodes // 5)
                if (ep + 1) % print_interval == 0:
                    print(
                        f"  [train] seed={seed} cond={cond_label} ep {ep + 1}/{n_episodes}",
                        flush=True,
                    )

            mean_gradient_reward = total_gradient_reward / max(total_steps, 1)

            results_by_seed[seed][cond_label] = {
                "mean_gradient_reward": mean_gradient_reward,
                "n_gradient_fires": n_gradient_fires,
                "total_steps": total_steps,
                "gradient_enabled": gradient_enabled,
            }

            if gradient_enabled:
                passed = mean_gradient_reward < THRESH_C1
                c1_by_seed.append(passed)
            else:
                passed = mean_gradient_reward == THRESH_C2
                c2_by_seed.append(passed)

            verdict_str = "PASS" if passed else "FAIL"
            print(f"verdict: {verdict_str}", flush=True)

    c1_pass = all(c1_by_seed) if c1_by_seed else False
    c2_pass = all(c2_by_seed) if c2_by_seed else False
    overall_pass = c1_pass and c2_pass

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c1_by_seed": c1_by_seed,
        "c2_by_seed": c2_by_seed,
        "results_by_seed": results_by_seed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dry_run = args.dry_run
    n_episodes = 5 if dry_run else N_EPISODES

    print(f"V3-EXQ-576 GAP-1 harm gradient validation", flush=True)
    print(f"  dry_run={dry_run} n_episodes={n_episodes} seeds={SEEDS}", flush=True)

    result = run_experiment(n_episodes=n_episodes, dry_run=dry_run)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_576_gap1_harm_gradient_validation_{timestamp}_v3"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": "gap1_harm_gradient_validation",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": [],
        "evidence_direction": "non_contributory",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "dry_run": dry_run,
        "config": {
            "seeds": SEEDS,
            "n_episodes": n_episodes,
            "n_steps": N_STEPS,
            "outer_radius": OUTER_RADIUS,
            "inner_radius": INNER_RADIUS,
            "scale": SCALE,
            "thresh_c1": THRESH_C1,
            "thresh_c2": THRESH_C2,
        },
        "acceptance_checks": {
            "C1_arm1_gradient_fires": result["c1_pass"],
            "C2_arm0_no_gradient": result["c2_pass"],
        },
        "c1_by_seed": result["c1_by_seed"],
        "c2_by_seed": result["c2_by_seed"],
        "results_by_seed": result["results_by_seed"],
    }

    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "REE_assembly", "evidence", "experiments"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written: {out_path}", flush=True)
    print(f"Outcome: {result['outcome']}", flush=True)
    print(f"  C1 (ARM_1 gradient fires): {result['c1_pass']}", flush=True)
    print(f"  C2 (ARM_0 no gradient): {result['c2_pass']}", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
