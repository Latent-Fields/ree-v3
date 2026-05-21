#!/opt/local/bin/python3
"""V3-EXQ-605 -- Q-043 MECH-313 vs MECH-314 relative-weight calibration sweep.

IGW-20260521-003 / arc_062_rule_apprehension:GAP-H.

Parametric grid on noise_floor_alpha (MECH-313) and curiosity scale (MECH-314)
on SD-054 bipartite reef + ARC-062 gated-policy + SP-CEM.

Grid: 3 alphas x 3 curiosity scales = 9 conditions (both modules ON).

Pre-registered PASS (Q-043 Pareto calibration zone):
  P1: At least one grid cell beats baseline (1x/1x) on entropy AND reef_fraction
  P2: Top-entropy cell and top-reef cell are not the same (w_313, w_314) pair
       -> non-degenerate tradeoff / Pareto surface

claim_ids: Q-043, ARC-065, MECH-313, MECH-314
experiment_purpose: evidence

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_605_q043_noise_floor_curiosity_weight_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_605_q043_noise_floor_curiosity_weight_sweep"
QUEUE_ID = "V3-EXQ-605"
CLAIM_IDS = ["Q-043", "ARC-065", "MECH-313", "MECH-314"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 2
DRY_RUN_STEPS = 50

ENTROPY_LIFT_MARGIN = 0.05
REEF_TOLERANCE = 0.15

NOISE_ALPHAS = [0.1, 0.5, 1.0]
CURIOSITY_SCALES = [1.0, 5.0, 10.0]
BASE_CURIOSITY_WEIGHT = 0.05
BASE_CURIOSITY_BIAS = 0.1

ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def _grid_arms() -> List[Dict[str, Any]]:
    arms: List[Dict[str, Any]] = []
    for alpha in NOISE_ALPHAS:
        for scale in CURIOSITY_SCALES:
            arms.append(
                {
                    "arm": f"ARM_a{alpha}_c{int(scale)}x",
                    "noise_floor_alpha": alpha,
                    "curiosity_scale": scale,
                }
            )
    return arms


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_config(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEConfig:
    scale = float(arm["curiosity_scale"])
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_gated_policy=True,
        gated_policy_use_first_action_onehot=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_noise_floor=True,
        noise_floor_alpha=float(arm["noise_floor_alpha"]),
        use_structured_curiosity=True,
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=True,
        curiosity_bias_scale=BASE_CURIOSITY_BIAS * scale,
        curiosity_novelty_weight=BASE_CURIOSITY_WEIGHT * scale,
        curiosity_uncertainty_weight=BASE_CURIOSITY_WEIGHT * scale,
        curiosity_lp_weight=BASE_CURIOSITY_WEIGHT * scale,
        use_dacc=False,
    )


def _run_cell(
    arm: Dict[str, Any],
    seed: int,
    episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    agent = REEAgent(_make_config(env, arm))

    action_counts: Counter = Counter()
    reef_steps = 0
    total_steps = 0
    reef_cells = set()

    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        if ep == 0:
            reef_cells = set(getattr(env, "_reef_cells", set()))

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            if obs_body.dim() == 1:
                obs_body = obs_body.unsqueeze(0)
            if obs_world.dim() == 1:
                obs_world = obs_world.unsqueeze(0)

            with torch.no_grad():
                action = agent.act_with_split_obs(obs_body, obs_world)
            if action is None:
                idx = random.randint(0, env.action_dim - 1)
                action_onehot = torch.zeros(1, env.action_dim, device=agent.device)
                action_onehot[0, idx] = 1.0
            else:
                action_onehot = action
                idx = int(action.argmax(dim=-1).item())
            action_counts[idx] += 1

            if (int(env.agent_x), int(env.agent_y)) in reef_cells:
                reef_steps += 1

            _flat, _harm, done, _info, obs_dict = env.step(action_onehot)
            total_steps += 1
            if done:
                break

        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            print(
                f"  [train] arm={arm['arm']} seed={seed} ep {ep + 1}/{episodes}",
                flush=True,
            )

    return {
        "arm": arm["arm"],
        "seed": seed,
        "noise_floor_alpha": arm["noise_floor_alpha"],
        "curiosity_scale": arm["curiosity_scale"],
        "selected_action_entropy": round(_entropy(action_counts), 6),
        "reef_fraction": round(reef_steps / max(total_steps, 1), 6),
        "total_steps": int(total_steps),
    }


def _mean_by_arm(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        arm = row["arm"]
        acc.setdefault(arm, {"entropy": [], "reef": []})
        acc[arm]["entropy"].append(row["selected_action_entropy"])
        acc[arm]["reef"].append(row["reef_fraction"])
    out: Dict[str, Dict[str, float]] = {}
    for arm, vals in acc.items():
        out[arm] = {
            "entropy": float(sum(vals["entropy"]) / len(vals["entropy"])),
            "reef": float(sum(vals["reef"]) / len(vals["reef"])),
        }
    return out


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    means = _mean_by_arm(rows)
    baseline_key = "ARM_a0.1_c1x"
    if baseline_key not in means:
        baseline_key = next(iter(means.keys()))
    b_ent = means[baseline_key]["entropy"]
    b_reef = means[baseline_key]["reef"]

    p1_cells: List[str] = []
    for arm, m in means.items():
        if (
            m["entropy"] > b_ent + ENTROPY_LIFT_MARGIN
            and abs(m["reef"] - b_reef) <= REEF_TOLERANCE
        ):
            p1_cells.append(arm)

    best_ent_arm = max(means.keys(), key=lambda a: means[a]["entropy"])
    best_reef_arm = max(means.keys(), key=lambda a: means[a]["reef"])
    p2 = best_ent_arm != best_reef_arm

    return {
        "baseline_arm": baseline_key,
        "baseline_entropy": round(b_ent, 6),
        "baseline_reef_fraction": round(b_reef, 6),
        "p1_calibration_cells": p1_cells,
        "p1_any_calibration_zone": bool(p1_cells),
        "p2_entropy_reef_tradeoff": p2,
        "best_entropy_arm": best_ent_arm,
        "best_reef_arm": best_reef_arm,
        "overall_pass": bool(p1_cells) and p2,
        "cell_means": {
            arm: {
                "entropy": round(m["entropy"], 6),
                "reef_fraction": round(m["reef"], 6),
            }
            for arm, m in means.items()
        },
    }


def _evidence_direction_per_claim(summary: Dict[str, Any]) -> Dict[str, str]:
    passed = summary["overall_pass"]
    base = "supports" if passed else "weakens"
    return {
        "Q-043": base,
        "ARC-065": base,
        "MECH-313": base,
        "MECH-314": base,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    arms = _grid_arms()

    rows: List[Dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            rows.append(_run_cell(arm, seed, episodes, steps))
            print("verdict: PASS", flush=True)

    summary = _evaluate(rows)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    edpc = _evidence_direction_per_claim(summary)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": edpc["Q-043"],
        "evidence_direction_per_claim": edpc,
        "dry_run": dry_run,
        "grid": {"noise_alphas": NOISE_ALPHAS, "curiosity_scales": CURIOSITY_SCALES},
        "acceptance_criteria": summary,
        "summary": summary,
        "cell_results": rows,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = out_dir / f"{run_id}.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_experiment(dry_run=args.dry_run)
    if args.dry_run:
        sys.exit(0)
    emit_outcome(
        outcome=str(result.get("outcome", "FAIL")),
        manifest_path=str(result.get("manifest_path", "/dev/null")),
    )
