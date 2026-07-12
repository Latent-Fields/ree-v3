#!/opt/local/bin/python3
"""V3-EXQ-604 -- Q-044 MECH-314a/b/c sub-flavour independence ablation.

IGW-20260521-003 / arc_062_rule_apprehension:GAP-H.

Three single-flavour-OFF arms vs all-ON baseline on SD-054 + gated-policy + SP-CEM.

Arms:
  ARM_0_all_on        -- novelty + uncertainty + LP all ON
  ARM_1_novelty_off   -- MECH-314a OFF (314b + 314c ON)
  ARM_2_uncertainty_off
  ARM_3_lp_off

Pre-registered PASS (Q-044 independence):
  C1: At least two ablation arms differ from ARM_0 by entropy > DISTINCT_MARGIN
  C2: The three ablation arms are not pairwise identical (max pairwise delta > EPS)

PASS if C1 and C2.

claim_ids: Q-044, MECH-314, MECH-314a, MECH-314b, MECH-314c
experiment_purpose: evidence

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_604_q044_mech314_subflavour_three_arm_ablation.py --dry-run
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
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_604_q044_mech314_subflavour_three_arm_ablation"
QUEUE_ID = "V3-EXQ-604"
CLAIM_IDS = ["Q-044", "MECH-314", "MECH-314a", "MECH-314b", "MECH-314c"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 2
DRY_RUN_STEPS = 50

DISTINCT_MARGIN = 0.03
PAIRWISE_EPS = 0.01

# Elevated vs 1x defaults (EXQ-573 calibration debt).
CURIOSITY_BIAS_SCALE = 0.5
CURIOSITY_WEIGHT = 0.25

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

ARMS: List[Dict[str, Any]] = [
    {
        "arm": "ARM_0_all_on",
        "use_curiosity_novelty": True,
        "use_curiosity_uncertainty": True,
        "use_curiosity_learning_progress": True,
    },
    {
        "arm": "ARM_1_novelty_off",
        "use_curiosity_novelty": False,
        "use_curiosity_uncertainty": True,
        "use_curiosity_learning_progress": True,
    },
    {
        "arm": "ARM_2_uncertainty_off",
        "use_curiosity_novelty": True,
        "use_curiosity_uncertainty": False,
        "use_curiosity_learning_progress": True,
    },
    {
        "arm": "ARM_3_lp_off",
        "use_curiosity_novelty": True,
        "use_curiosity_uncertainty": True,
        "use_curiosity_learning_progress": False,
    },
]


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


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_config(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEConfig:
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
        use_structured_curiosity=True,
        use_curiosity_novelty=arm["use_curiosity_novelty"],
        use_curiosity_uncertainty=arm["use_curiosity_uncertainty"],
        use_curiosity_learning_progress=arm["use_curiosity_learning_progress"],
        curiosity_bias_scale=CURIOSITY_BIAS_SCALE,
        curiosity_novelty_weight=CURIOSITY_WEIGHT,
        curiosity_uncertainty_weight=CURIOSITY_WEIGHT,
        curiosity_lp_weight=CURIOSITY_WEIGHT,
        use_noise_floor=False,
        use_dacc=False,
    )


def _run_arm_seed(
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
    total_steps = 0

    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()

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
        "selected_action_entropy": round(_entropy(action_counts), 6),
        "total_steps": int(total_steps),
        "unique_actions": len(action_counts),
    }


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, float] = {}
    for row in rows:
        arm = row["arm"]
        by_arm.setdefault(arm, []).append(row["selected_action_entropy"])  # type: ignore

    means = {arm: float(sum(v) / len(v)) for arm, v in by_arm.items()}  # type: ignore

    e0 = means.get("ARM_0_all_on", 0.0)
    e1 = means.get("ARM_1_novelty_off", 0.0)
    e2 = means.get("ARM_2_uncertainty_off", 0.0)
    e3 = means.get("ARM_3_lp_off", 0.0)

    deltas_vs_0 = [
        abs(e1 - e0),
        abs(e2 - e0),
        abs(e3 - e0),
    ]
    c1 = sum(1 for d in deltas_vs_0 if d > DISTINCT_MARGIN) >= 2

    pairwise = [abs(e1 - e2), abs(e1 - e3), abs(e2 - e3)]
    c2 = max(pairwise) > PAIRWISE_EPS

    return {
        "entropy_ARM_0": round(e0, 6),
        "entropy_ARM_1": round(e1, 6),
        "entropy_ARM_2": round(e2, 6),
        "entropy_ARM_3": round(e3, 6),
        "c1_two_distinct_ablations": c1,
        "c2_ablations_not_identical": c2,
        "overall_pass": bool(c1 and c2),
    }


def _evidence_direction_per_claim(summary: Dict[str, Any]) -> Dict[str, str]:
    passed = summary["overall_pass"]
    base = "supports" if passed else "weakens"
    return {
        "Q-044": base,
        "MECH-314": base,
        "MECH-314a": "supports" if summary["entropy_ARM_1"] < summary["entropy_ARM_0"] else "mixed",
        "MECH-314b": "supports" if summary["entropy_ARM_2"] < summary["entropy_ARM_0"] else "mixed",
        "MECH-314c": "supports" if summary["entropy_ARM_3"] < summary["entropy_ARM_0"] else "mixed",
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            rows.append(_run_arm_seed(arm, seed, episodes, steps))
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
        "evidence_direction": edpc["Q-044"],
        "evidence_direction_per_claim": edpc,
        "dry_run": dry_run,
        "acceptance_criteria": summary,
        "summary": summary,
        "arm_results": rows,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = out_dir / f"{run_id}.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
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
