#!/opt/local/bin/python3
"""V3-EXQ-522: Reef Monostrategy Break -- Behavioral Diversity Validation

Tests whether the SD-050 reef enrichment substrate (reef safe zones + food-attracted
hazards) creates meaningful behavioral diversity in a harm-avoiding resource-seeking
agent. This is the behavioral follow-up to V3-EXQ-521 (substrate readiness PASS).

Monostrategy problem: in the baseline environment, a harm-avoiding agent learns one
fixed route to food, producing a monomodal policy that cannot generate balanced
self-vs-externally-caused event distributions (blocks SD-029 C2/C3 measurement).

Three-arm design:
  ARM_0: baseline (reef_enabled=False, hazard_food_attraction=0.0)
  ARM_1: reef + food-attracted hazards (reef_enabled=True, hazard_food_attraction=0.7)
  ARM_2: reef only (reef_enabled=True, hazard_food_attraction=0.0)

Agent policy: harm-avoiding heuristic with reef awareness --
  if nearest hazard within flee_threshold cells AND reef available: move toward reef
  else: move toward nearest food
  (fallback: random walk)

Acceptance criteria:
  C1: Reef used         -- ARM_1 reef_visit_fraction >= 0.05
  C2: Foraging persists -- ARM_1 foraging_fraction >= 0.15
  C3: Zone switching    -- ARM_1 zone_transitions_per_ep >= 1.0

Diagnostic (not hard gates):
  C4: ARM_2 reef_visit_fraction >= 0.05
  C5: ARM_1 zone_transitions_per_ep >= ARM_2 zone_transitions_per_ep
  C6: ARM_0 reef_visit_fraction == 0.0   (sanity: no reef cells, no reef visits)

Overall PASS = C1 AND C2 AND C3.

claim_ids: []
evidence_direction: "supports"
experiment_purpose: "diagnostic"
"""

import json
import sys
import os
import time
import numpy as np
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2

FLEE_THRESHOLD = 2  # Manhattan distance; retreat to reef when hazard within this range


def _move_toward(ax, ay, tx, ty):
    """Greedy single-step toward (tx, ty). Actions: 0:(-1,0) 1:(1,0) 2:(0,-1) 3:(0,1) 4:NOOP"""
    dx = tx - ax
    dy = ty - ay
    if dx == 0 and dy == 0:
        return 4
    if abs(dx) >= abs(dy):
        return 0 if dx < 0 else 1
    return 2 if dy < 0 else 3


def _heuristic_action(env, reef_cells, use_reef, rng):
    """Harm-avoiding, reef-aware, resource-seeking heuristic policy."""
    ax, ay = env.agent_x, env.agent_y
    nearest_hz = min(
        (abs(ax - h[0]) + abs(ay - h[1]) for h in env.hazards),
        default=999
    )
    if use_reef and nearest_hz <= FLEE_THRESHOLD and reef_cells:
        if (ax, ay) in reef_cells:
            return 4  # already safe; stay put
        target = min(reef_cells, key=lambda r: abs(ax - r[0]) + abs(ay - r[1]))
        return _move_toward(ax, ay, target[0], target[1])
    if env.resources:
        target = min(env.resources, key=lambda r: abs(ax - r[0]) + abs(ay - r[1]))
        return _move_toward(ax, ay, target[0], target[1])
    return int(rng.integers(0, env.action_dim))


def _run_arm(arm_id, env_kwargs, n_episodes, steps_per_ep, rng):
    """Run one ARM and collect behavioral diversity statistics."""
    env = CausalGridWorldV2(**env_kwargs)
    use_reef = env_kwargs.get("reef_enabled", False)

    total_steps = 0
    reef_steps = 0
    non_reef_steps = 0
    zone_transitions_total = 0
    ep_entropies = []
    ep_reef_fracs = []
    ep_trans_list = []

    for ep in range(n_episodes):
        env.reset()
        reef_cells = env._reef_cells

        ax, ay = env.agent_x, env.agent_y
        prev_in_reef = bool(reef_cells and (ax, ay) in reef_cells)
        ep_positions = []
        ep_reef = 0
        ep_transitions = 0

        for step in range(steps_per_ep):
            ax, ay = env.agent_x, env.agent_y
            in_reef = bool(reef_cells and (ax, ay) in reef_cells)

            ep_positions.append((ax, ay))
            if in_reef:
                ep_reef += 1
                reef_steps += 1
            else:
                non_reef_steps += 1
            total_steps += 1

            if in_reef != prev_in_reef:
                ep_transitions += 1
                zone_transitions_total += 1
            prev_in_reef = in_reef

            action = _heuristic_action(env, reef_cells, use_reef, rng)
            env.step(action)

        ep_steps = len(ep_positions)
        if ep_steps > 0:
            counts = Counter(ep_positions)
            tot = sum(counts.values())
            entropy = -sum((c / tot) * np.log(c / tot + 1e-12) for c in counts.values())
        else:
            entropy = 0.0

        ep_entropies.append(entropy)
        ep_reef_fracs.append(ep_reef / max(ep_steps, 1))
        ep_trans_list.append(ep_transitions)

    return {
        "arm_id": arm_id,
        "world_obs_dim": env.world_obs_dim,
        "n_reef_cells": len(env._reef_cells),
        "reef_visit_fraction": reef_steps / max(total_steps, 1),
        "foraging_fraction": non_reef_steps / max(total_steps, 1),
        "zone_transitions_per_ep": zone_transitions_total / max(n_episodes, 1),
        "mean_position_entropy": float(np.mean(ep_entropies)),
        "std_position_entropy": float(np.std(ep_entropies)),
        "mean_ep_reef_fraction": float(np.mean(ep_reef_fracs)),
        "mean_ep_transitions": float(np.mean(ep_trans_list)),
    }


def run_experiment(dry_run=False):
    n_episodes = 5 if dry_run else 30
    steps_per_ep = 50 if dry_run else 200
    n_seeds = 1 if dry_run else 3
    grid_size = 12

    arms_def = [
        ("ARM_0_baseline",
         dict(reef_enabled=False, hazard_food_attraction=0.0)),
        ("ARM_1_reef_food",
         dict(reef_enabled=True, n_reef_patches=3, reef_patch_radius=2,
              hazard_food_attraction=0.7)),
        ("ARM_2_reef_only",
         dict(reef_enabled=True, n_reef_patches=3, reef_patch_radius=2,
              hazard_food_attraction=0.0)),
    ]

    results_by_seed = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 2000)
        seed_results = []
        common = dict(
            size=grid_size, num_hazards=3, num_resources=5,
            use_proxy_fields=True,
            env_drift_prob=0.3, env_drift_interval=1,
        )
        for arm_id, arm_extra in arms_def:
            kwargs = dict(**common, seed=seed, **arm_extra)
            result = _run_arm(arm_id, kwargs, n_episodes, steps_per_ep, rng)
            seed_results.append(result)
            if dry_run:
                print(
                    f"  seed={seed} {arm_id}: "
                    f"reef_frac={result['reef_visit_fraction']:.3f} "
                    f"forage_frac={result['foraging_fraction']:.3f} "
                    f"transitions/ep={result['zone_transitions_per_ep']:.2f} "
                    f"entropy={result['mean_position_entropy']:.2f}"
                )
        results_by_seed.append(seed_results)

    per_arm = {}
    for arm_id, _ in arms_def:
        per_arm[arm_id] = {
            k: float(np.mean([r[k] for sr in results_by_seed for r in sr
                               if r["arm_id"] == arm_id]))
            for k in ["reef_visit_fraction", "foraging_fraction",
                      "zone_transitions_per_ep", "mean_position_entropy",
                      "n_reef_cells", "world_obs_dim"]
        }

    arm0 = per_arm["ARM_0_baseline"]
    arm1 = per_arm["ARM_1_reef_food"]
    arm2 = per_arm["ARM_2_reef_only"]

    c1 = arm1["reef_visit_fraction"] >= 0.05
    c2 = arm1["foraging_fraction"] >= 0.15
    c3 = arm1["zone_transitions_per_ep"] >= 1.0
    c4 = arm2["reef_visit_fraction"] >= 0.05
    c5 = arm1["zone_transitions_per_ep"] >= arm2["zone_transitions_per_ep"]
    c6 = arm0["reef_visit_fraction"] == 0.0

    overall_pass = c1 and c2 and c3

    return {
        "per_arm": per_arm,
        "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6},
        "overall_pass": overall_pass,
        "n_episodes": n_episodes,
        "steps_per_ep": steps_per_ep,
        "n_seeds": n_seeds,
    }


def write_result(result, run_id):
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "REE_assembly", "evidence", "experiments"
    )
    out_path = os.path.join(output_dir, f"{run_id}.json")
    manifest = {
        "run_id": run_id,
        "experiment_type": "v3_exq_522_reef_monostrategy_break",
        "queue_id": "V3-EXQ-522",
        "claim_ids": [],
        "evidence_direction": "supports",
        "experiment_purpose": "diagnostic",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": "PASS" if result["overall_pass"] else "FAIL",
        "metrics": result,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    print("V3-EXQ-522 Reef Monostrategy Break -- Behavioral Diversity")
    print(f"dry_run={args.dry_run}")

    result = run_experiment(dry_run=args.dry_run)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Criteria: {result['criteria']}")
    print(f"Overall: {'PASS' if result['overall_pass'] else 'FAIL'}")

    for arm_id, stats in result["per_arm"].items():
        print(
            f"  {arm_id}: "
            f"reef={stats['reef_visit_fraction']:.3f} "
            f"forage={stats['foraging_fraction']:.3f} "
            f"transitions/ep={stats['zone_transitions_per_ep']:.2f} "
            f"entropy={stats['mean_position_entropy']:.2f}"
        )

    if not args.dry_run:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"v3_exq_522_reef_monostrategy_break_{ts}_v3"
        write_result(result, run_id)
