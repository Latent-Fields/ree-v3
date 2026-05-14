#!/opt/local/bin/python3
"""V3-EXQ-561 -- ARC-065 / MECH-313 / MECH-314 / MECH-320 diversity stack heartbeat.

Calibration debt Work Package B. Diagnostic only -- no governance weighting.

Purpose
-------
Substrates MECH-313 (noise floor), MECH-314 (structured curiosity), and
MECH-320 (tonic vigor) have all landed and pass contract tests. None has
produced a measurable behavioural diversity effect in prior experiments.
This 6-arm matched-entropy heartbeat localises where calibration fails:

  1. Does MECH-313 alone increase action class entropy above baseline?
  2. Does MECH-314 alone increase action class entropy above baseline?
  3. Does a high-temperature matched-entropy control (ARM_3) exceed both,
     indicating that the MECH-314 curiosity bias is weaker than raw noise?
  4. Does the combined MECH-313 + MECH-314 arm produce additive diversity?
  5. Does adding MECH-320 (full stack, ARM_5) help, and does v_t move at all
     (Rung-2 diagnosis: is the EWMA signal non-zero)?

Arms
----
ARM_0_baseline:
    No diversity modules. Action entropy floor.

ARM_1_mech313_only:
    MECH-313 noise floor (noise_floor_alpha=0.1, min_temperature=1.0).
    Tests whether tonic LC-NE temperature lift alone diversifies selection.

ARM_2_mech314_only:
    MECH-314 structured curiosity only (novelty + uncertainty + learning_progress).
    Tests whether the curiosity score bias alone diversifies selection.

ARM_3_entropy_ceiling:
    MECH-313 substrate used as entropy ceiling control (noise_floor_alpha=5.0,
    effective temperature ~6.0 -> near-uniform selection).
    This is the maximum-entropy achievable without architectural change.

ARM_4_mech313_mech314:
    MECH-313 + MECH-314 combined. Tests additive diversity.

ARM_5_full_stack:
    MECH-313 + MECH-314 + MECH-320 + MECH-260 (dACC anti-recency).
    Full ARC-065 diversity stack. Also instruments MECH-320 Rung-2 state.

PASS: diagnostic ran and produced finite metrics for all arms.
Evidence direction: non_contributory for all claims (diagnostic run).

Smoke
-----
  /opt/local/bin/python3 experiments/v3_exq_561_arc065_diversity_stack_heartbeat.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from _harness import StepHarness, StepHooks  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_561_arc065_diversity_stack_heartbeat"
QUEUE_ID = "V3-EXQ-561"
CLAIM_IDS: List[str] = ["ARC-065", "MECH-313", "MECH-314", "MECH-320"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [7, 42]
EVAL_EPISODES = 8
STEPS_PER_EPISODE = 150

DRY_RUN_SEEDS = [7]
DRY_RUN_EPISODES = 1
DRY_RUN_STEPS = 50

ARMS = [
    {
        "arm": "ARM_0_baseline",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_structured_curiosity": False,
        "use_tonic_vigor": False,
        "use_dacc": False,
        "dacc_suppression_weight": 0.0,
    },
    {
        "arm": "ARM_1_mech313_only",
        "use_noise_floor": True,
        "noise_floor_alpha": 0.1,
        "use_structured_curiosity": False,
        "use_tonic_vigor": False,
        "use_dacc": False,
        "dacc_suppression_weight": 0.0,
    },
    {
        "arm": "ARM_2_mech314_only",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_structured_curiosity": True,
        "use_tonic_vigor": False,
        "use_dacc": False,
        "dacc_suppression_weight": 0.0,
    },
    {
        "arm": "ARM_3_entropy_ceiling",
        "use_noise_floor": True,
        "noise_floor_alpha": 5.0,
        "use_structured_curiosity": False,
        "use_tonic_vigor": False,
        "use_dacc": False,
        "dacc_suppression_weight": 0.0,
    },
    {
        "arm": "ARM_4_mech313_mech314",
        "use_noise_floor": True,
        "noise_floor_alpha": 0.1,
        "use_structured_curiosity": True,
        "use_tonic_vigor": False,
        "use_dacc": False,
        "dacc_suppression_weight": 0.0,
    },
    {
        "arm": "ARM_5_full_stack",
        "use_noise_floor": True,
        "noise_floor_alpha": 0.1,
        "use_structured_curiosity": True,
        "use_tonic_vigor": True,
        "use_dacc": True,
        "dacc_suppression_weight": 0.5,
    },
]


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=8,
        num_hazards=2,
        num_resources=10,
        hazard_harm=0.01,
        resource_benefit=0.25,
        energy_decay=0.015,
        use_proxy_fields=True,
        proximity_benefit_scale=0.18,
        proximity_harm_scale=0.01,
        resource_respawn_on_consume=True,
        seed=seed,
    )


def _make_config(env: CausalGridWorld, arm: Dict) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        use_noise_floor=arm["use_noise_floor"],
        noise_floor_alpha=arm["noise_floor_alpha"],
        use_structured_curiosity=arm["use_structured_curiosity"],
        use_tonic_vigor=arm["use_tonic_vigor"],
        use_dacc=arm["use_dacc"],
        dacc_suppression_weight=arm["dacc_suppression_weight"],
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


def _run_arm_seed(
    arm: Dict,
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
    total_steps = 0

    def on_action(*, agent, latent, action, obs_dict, ticks, step, **_kw) -> None:
        idx = int(action.argmax(dim=-1).item())
        action_counts[idx] += 1

    hooks = StepHooks(on_action=on_action)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)

    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            total_steps += 1
            obs_dict = result.next_obs_dict
            if result.done:
                break
        print(
            f"  [train] arm={arm['arm']} seed={seed} ep {ep+1}/{episodes}",
            flush=True,
        )

    entropy = _entropy(action_counts)
    unique_actions = len(action_counts)

    metrics: Dict[str, Any] = {
        "arm": arm["arm"],
        "seed": seed,
        "total_steps": total_steps,
        "action_class_entropy": round(entropy, 6),
        "unique_actions_taken": unique_actions,
        "action_counts": dict(action_counts),
    }

    # MECH-320 Rung-2 diagnostic (ARM_5 only, but safe to collect anywhere)
    if arm["use_tonic_vigor"] and agent.tonic_vigor is not None:
        tv_state = agent.tonic_vigor.get_state()
        metrics["mech320_v_raw_final"] = round(float(tv_state.get("v_raw", 0.0)), 6)
        metrics["mech320_last_v_t"] = round(float(tv_state.get("last_v_t", 0.0)), 6)
        metrics["mech320_n_score_updates"] = int(
            tv_state.get("n_waking_score_updates", 0)
        )
        metrics["mech320_n_bias_calls"] = int(
            tv_state.get("n_waking_bias_calls", 0)
        )
        # Rung-2 passes if the EWMA ever produced a non-zero gated signal
        metrics["mech320_rung2_pass"] = bool(
            metrics["mech320_last_v_t"] > 0.0
            or metrics["mech320_v_raw_final"] > 0.0
        )

    return metrics


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    arm_entropy_means: Dict[str, float] = {}

    for arm in ARMS:
        seed_entropies = []
        for seed in seeds:
            print(f"Seed {seed} Arm {arm['arm']}", flush=True)
            cell = _run_arm_seed(arm, seed, episodes, steps)
            arm_results.append(cell)
            seed_entropies.append(cell["action_class_entropy"])
            passed = math.isfinite(cell["action_class_entropy"])
            print(
                f"verdict: {'PASS' if passed else 'FAIL'}",
                flush=True,
            )
        arm_entropy_means[arm["arm"]] = (
            sum(seed_entropies) / len(seed_entropies) if seed_entropies else 0.0
        )

    baseline_entropy = arm_entropy_means.get("ARM_0_baseline", 0.0)
    ceiling_entropy = arm_entropy_means.get("ARM_3_entropy_ceiling", 0.0)

    summary: Dict[str, Any] = {
        "arm_entropy_means": {k: round(v, 6) for k, v in arm_entropy_means.items()},
        "baseline_entropy": round(baseline_entropy, 6),
        "ceiling_entropy": round(ceiling_entropy, 6),
        "mech313_entropy_delta": round(
            arm_entropy_means.get("ARM_1_mech313_only", 0.0) - baseline_entropy, 6
        ),
        "mech314_entropy_delta": round(
            arm_entropy_means.get("ARM_2_mech314_only", 0.0) - baseline_entropy, 6
        ),
        "combined_entropy_delta": round(
            arm_entropy_means.get("ARM_4_mech313_mech314", 0.0) - baseline_entropy, 6
        ),
        "full_stack_entropy_delta": round(
            arm_entropy_means.get("ARM_5_full_stack", 0.0) - baseline_entropy, 6
        ),
        "curiosity_beats_noise": bool(
            arm_entropy_means.get("ARM_2_mech314_only", 0.0)
            > arm_entropy_means.get("ARM_1_mech313_only", 0.0)
        ),
        "combined_beats_ceiling": bool(
            arm_entropy_means.get("ARM_4_mech313_mech314", 0.0) > ceiling_entropy
        ),
    }

    # MECH-320 Rung-2 aggregate
    arm5_cells = [r for r in arm_results if r["arm"] == "ARM_5_full_stack"]
    if arm5_cells and "mech320_rung2_pass" in arm5_cells[0]:
        summary["mech320_rung2_pass_any_seed"] = any(
            c.get("mech320_rung2_pass", False) for c in arm5_cells
        )
        summary["mech320_n_score_updates_total"] = sum(
            c.get("mech320_n_score_updates", 0) for c in arm5_cells
        )
        summary["mech320_v_raw_final_mean"] = round(
            sum(c.get("mech320_v_raw_final", 0.0) for c in arm5_cells)
            / max(1, len(arm5_cells)),
            6,
        )

    # PASS = all cells produced finite entropy (diagnostic ran cleanly)
    all_finite = all(
        math.isfinite(r["action_class_entropy"]) for r in arm_results
    )
    outcome = "PASS" if all_finite else "FAIL"

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": {
            "ARC-065": "non_contributory",
            "MECH-313": "non_contributory",
            "MECH-314": "non_contributory",
            "MECH-320": "non_contributory",
        },
        "dry_run": dry_run,
        "config": {
            "seeds": seeds,
            "eval_episodes": episodes,
            "steps_per_episode": steps,
            "n_arms": len(ARMS),
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        print("Dry run -- manifest not written.", flush=True)
        out_path = Path("/dev/null")

    print(f"Outcome: {outcome}", flush=True)
    print(f"Baseline entropy: {baseline_entropy:.4f}", flush=True)
    print(f"Ceiling entropy:  {ceiling_entropy:.4f}", flush=True)
    print(f"MECH-313 delta:   {summary['mech313_entropy_delta']:+.4f}", flush=True)
    print(f"MECH-314 delta:   {summary['mech314_entropy_delta']:+.4f}", flush=True)
    print(f"Combined delta:   {summary['combined_entropy_delta']:+.4f}", flush=True)
    print(f"Full stack delta: {summary['full_stack_entropy_delta']:+.4f}", flush=True)
    if "mech320_rung2_pass_any_seed" in summary:
        print(
            f"MECH-320 Rung-2 pass (any seed): {summary['mech320_rung2_pass_any_seed']}",
            flush=True,
        )

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-561 ARC-065 diversity stack heartbeat"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Short smoke run (1 seed, 1 episode, 50 steps); no manifest written.",
    )
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    _manifest_path = result.get("manifest_path", str(Path("/dev/null")))

    emit_outcome(
        outcome=_outcome,
        manifest_path=_manifest_path,
    )
    sys.exit(0)
