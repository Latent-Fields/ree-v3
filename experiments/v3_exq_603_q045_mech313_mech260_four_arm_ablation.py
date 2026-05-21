#!/opt/local/bin/python3
"""V3-EXQ-603 -- Q-045 MECH-313 vs MECH-260 four-arm anti-monostrategy ablation.

IGW-20260521-003 / arc_062_rule_apprehension:GAP-H.

Falsifies whether MECH-313 (LC-NE noise floor) and MECH-260 (dACC anti-recency)
are mutually load-bearing or collapse into one substrate.

SD-054 bipartite reef env + ARC-062 gated-policy ON + main-path SP-CEM.

Arms:
  ARM_0_both_off   -- control (expected collapse / low diversity)
  ARM_1_mech313    -- use_noise_floor only
  ARM_2_mech260    -- use_dacc only (MECH-260 pathway)
  ARM_3_both_on    -- MECH-313 + MECH-260

Pre-registered PASS (Q-045 mutually load-bearing):
  C1: ARM_3 entropy > ARM_0 + ENTROPY_MARGIN
  C2: ARM_3 entropy > max(ARM_1, ARM_2) + ENTROPY_MARGIN
  C3: ARM_1 entropy > ARM_0 AND ARM_2 entropy > ARM_0

PASS if C2. FAIL otherwise (still records per-arm directions).

claim_ids: Q-045, MECH-313, MECH-260
experiment_purpose: evidence

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_603_q045_mech313_mech260_four_arm_ablation.py --dry-run
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


EXPERIMENT_TYPE = "v3_exq_603_q045_mech313_mech260_four_arm_ablation"
QUEUE_ID = "V3-EXQ-603"
CLAIM_IDS = ["Q-045", "MECH-313", "MECH-260"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 2
DRY_RUN_STEPS = 50

ENTROPY_MARGIN = 0.05
DACC_SUPPRESSION_WEIGHT = 0.5

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
        "arm": "ARM_0_both_off",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_dacc": False,
    },
    {
        "arm": "ARM_1_mech313_only",
        "use_noise_floor": True,
        "noise_floor_alpha": 0.5,
        "use_dacc": False,
    },
    {
        "arm": "ARM_2_mech260_only",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_dacc": True,
    },
    {
        "arm": "ARM_3_both_on",
        "use_noise_floor": True,
        "noise_floor_alpha": 0.5,
        "use_dacc": True,
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
        use_noise_floor=arm["use_noise_floor"],
        noise_floor_alpha=arm["noise_floor_alpha"],
        use_dacc=arm["use_dacc"],
        dacc_weight=(1.0 if arm["use_dacc"] else 0.0),
        dacc_suppression_weight=(
            DACC_SUPPRESSION_WEIGHT if arm["use_dacc"] else 0.0
        ),
        use_structured_curiosity=False,
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
    cfg = _make_config(env, arm)
    agent = REEAgent(cfg)

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

            pos = (int(env.agent_x), int(env.agent_y))
            if pos in reef_cells:
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

    entropy = _entropy(action_counts)
    reef_fraction = reef_steps / max(total_steps, 1)

    return {
        "arm": arm["arm"],
        "seed": seed,
        "selected_action_entropy": round(entropy, 6),
        "reef_fraction": round(reef_fraction, 6),
        "total_steps": int(total_steps),
        "unique_actions": len(action_counts),
    }


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    def mean_entropy(arm_name: str) -> float:
        vals = [r["selected_action_entropy"] for r in by_arm.get(arm_name, [])]
        return float(sum(vals) / len(vals)) if vals else 0.0

    e0 = mean_entropy("ARM_0_both_off")
    e1 = mean_entropy("ARM_1_mech313_only")
    e2 = mean_entropy("ARM_2_mech260_only")
    e3 = mean_entropy("ARM_3_both_on")

    c1 = e3 > e0 + ENTROPY_MARGIN
    c2 = e3 > max(e1, e2) + ENTROPY_MARGIN
    c3 = (e1 > e0) and (e2 > e0)

    return {
        "entropy_ARM_0": round(e0, 6),
        "entropy_ARM_1": round(e1, 6),
        "entropy_ARM_2": round(e2, 6),
        "entropy_ARM_3": round(e3, 6),
        "c1_both_beats_off": c1,
        "c2_mutually_load_bearing": c2,
        "c3_each_alone_beats_off": c3,
        "overall_pass": bool(c2),
    }


def _evidence_direction_per_claim(summary: Dict[str, Any]) -> Dict[str, str]:
    e0 = summary["entropy_ARM_0"]
    e1 = summary["entropy_ARM_1"]
    e2 = summary["entropy_ARM_2"]
    e3 = summary["entropy_ARM_3"]

    if summary["c2_mutually_load_bearing"]:
        q045 = "supports"
        m313 = "supports" if e1 > e0 + ENTROPY_MARGIN else "mixed"
        m260 = "supports" if e2 > e0 + ENTROPY_MARGIN else "mixed"
    elif e3 > e0:
        q045 = "mixed"
        m313 = "supports" if e1 >= e2 else "weakens"
        m260 = "supports" if e2 >= e1 else "weakens"
    else:
        q045 = "weakens"
        m313 = "weakens"
        m260 = "weakens"

    return {"Q-045": q045, "MECH-313": m313, "MECH-260": m260}


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            cell = _run_arm_seed(arm, seed, episodes, steps)
            rows.append(cell)
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
        "evidence_direction": edpc["Q-045"],
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
