"""
V3-EXQ-575: ISEF-001 Harm Gradient Baseline -- Warm-Start Gate Diagnostic.

DEV-NEED-029: Confirm that proximity_harm_scale=0.30 (ARM_1 treatment) produces
residue_coverage_pct >= 0.15 (at least 5 of 32 RBF centers active) after 200 episodes,
compared to proximity_harm_scale=0.05 (ARM_0 control, default cold-start).

This diagnostic gates ALL downstream diversity experiments (ARC-065 sweeps, Q-043/044/045
retests, INV-049 retest). EXQ-573 proved that cold-start environments produce null
diversity results. This experiment confirms the warm-start gate is achievable.

Interpretation grid:
  Outcome                                    | Diagnosis
  -------------------------------------------|------------------------------------------
  C1 FAIL (ARM_1 mean coverage < 0.15)       | proximity_harm_scale=0.30 insufficient;
                                             |   try higher scale or more episodes;
                                             |   check update_residue() call path
  C2 FAIL (ARM_1 not > ARM_0)               | Residue field not responding to harm
                                             |   gradient; check proximity_harm_scale
                                             |   interaction with CausalGridWorldV2
  C1+C2 PASS                                | Warm-start gate validated; ARC-065 and
                                             |   downstream experiments can proceed with
                                             |   proximity_harm_scale=0.30 pre-training
  ARM_0 also >= 0.15                         | Gate trivially met; cold-start may be
                                             |   sufficient; re-examine DEV-NEED-029
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_575_isef001_harm_gradient_baseline"
QUEUE_ID = "V3-EXQ-575"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44, 45, 46]
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4
N_EPISODES = 200
STEPS_PER_EPISODE = 200
NUM_BASIS_FUNCTIONS = 32

ARM_NAMES = ["ARM_0_control", "ARM_1_treatment"]
ARM_HARM_SCALES: Dict[str, float] = {
    "ARM_0_control": 0.05,
    "ARM_1_treatment": 0.30,
}

# Pre-registered acceptance thresholds
C1_MIN_ARM1_COVERAGE = 0.15   # ARM_1 mean coverage >= 15% of RBF centers
C2_ARM1_EXCEEDS_ARM0 = True   # ARM_1 mean > ARM_0 mean


def _build_agent() -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
    )
    cfg.latent.alpha_world = 0.9
    return REEAgent(cfg)


def _extract_split_obs(obs_dict: Dict) -> tuple:
    """Extract body and world tensors from obs_dict using correct env keys."""
    obs_body = obs_dict["body_state"].float()
    if obs_body.shape[0] < BODY_OBS_DIM:
        obs_body = torch.cat([obs_body, torch.zeros(BODY_OBS_DIM - obs_body.shape[0])])
    elif obs_body.shape[0] > BODY_OBS_DIM:
        obs_body = obs_body[:BODY_OBS_DIM]

    obs_world = obs_dict["world_state"].float()
    if obs_world.shape[0] < WORLD_OBS_DIM:
        obs_world = torch.cat([obs_world, torch.zeros(WORLD_OBS_DIM - obs_world.shape[0])])
    elif obs_world.shape[0] > WORLD_OBS_DIM:
        obs_world = obs_world[:WORLD_OBS_DIM]

    return obs_body, obs_world


def _run_arm(
    *,
    seed: int,
    arm_name: str,
    harm_scale: float,
    dry_run: bool,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    agent = _build_agent()

    env = CausalGridWorldV2(
        proximity_harm_scale=harm_scale,
        resource_respawn_on_consume=True,
    )

    n_episodes = 2 if dry_run else N_EPISODES
    coverage_per_ep: List[float] = []

    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        obs_body, obs_world = _extract_split_obs(obs_dict)

        for _step in range(STEPS_PER_EPISODE):
            with torch.no_grad():
                action = agent.act_with_split_obs(obs_body=obs_body, obs_world=obs_world)

            action_idx = int(action.argmax().item()) % env.action_dim
            _flat_obs, harm_signal, done, info, obs_dict = env.step(action_idx)
            agent.update_residue(float(harm_signal))

            if done:
                _flat, obs_dict = env.reset()

            obs_body, obs_world = _extract_split_obs(obs_dict)

        stats = agent.residue_field.get_statistics()
        active = float(stats["active_centers"].item())
        coverage = active / NUM_BASIS_FUNCTIONS
        coverage_per_ep.append(coverage)

        if (ep + 1) % 50 == 0 or (ep + 1) == n_episodes:
            print(
                f"  [train] {arm_name} ep {ep + 1}/{n_episodes} "
                f"seed={seed} coverage={coverage:.3f}",
                flush=True,
            )

    final_coverage = coverage_per_ep[-1] if coverage_per_ep else 0.0
    mean_coverage = sum(coverage_per_ep) / len(coverage_per_ep) if coverage_per_ep else 0.0

    return {
        "arm": arm_name,
        "harm_scale": harm_scale,
        "final_coverage": final_coverage,
        "mean_coverage": mean_coverage,
        "coverage_per_ep": coverage_per_ep if dry_run else coverage_per_ep[-10:],
    }


def _run_seed(*, seed: int, dry_run: bool) -> Dict[str, Any]:
    arm_results: Dict[str, Dict[str, Any]] = {}
    for arm_name in ARM_NAMES:
        harm_scale = ARM_HARM_SCALES[arm_name]
        print(f"Seed {seed} Condition {arm_name}", flush=True)
        result = _run_arm(
            seed=seed,
            arm_name=arm_name,
            harm_scale=harm_scale,
            dry_run=dry_run,
        )
        arm_results[arm_name] = result

        arm1_cov = result["final_coverage"]
        seed_arm_pass = arm1_cov >= 0.05
        print(
            f"verdict: {'PASS' if seed_arm_pass else 'FAIL'}",
            flush=True,
        )

    return {
        "seed": seed,
        "arm_results": arm_results,
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    print(f"V3-EXQ-575: ISEF-001 Harm Gradient Baseline -- Warm-Start Gate", flush=True)
    print(f"  seeds={seeds} dry_run={dry_run}", flush=True)
    print(
        f"  ARM_0 harm_scale={ARM_HARM_SCALES['ARM_0_control']} "
        f"ARM_1 harm_scale={ARM_HARM_SCALES['ARM_1_treatment']}",
        flush=True,
    )

    all_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        result = _run_seed(seed=seed, dry_run=dry_run)
        all_seed_results.append(result)

    arm0_coverages = [
        r["arm_results"]["ARM_0_control"]["final_coverage"]
        for r in all_seed_results
    ]
    arm1_coverages = [
        r["arm_results"]["ARM_1_treatment"]["final_coverage"]
        for r in all_seed_results
    ]
    arm0_mean = sum(arm0_coverages) / len(arm0_coverages)
    arm1_mean = sum(arm1_coverages) / len(arm1_coverages)

    if dry_run:
        c1_pass = True
        c2_pass = True
    else:
        c1_pass = arm1_mean >= C1_MIN_ARM1_COVERAGE
        c2_pass = arm1_mean > arm0_mean

    outcome = "PASS" if (c1_pass and c2_pass) else "FAIL"

    print(f"", flush=True)
    print(f"ARM_0 mean final coverage: {arm0_mean:.4f}", flush=True)
    print(f"ARM_1 mean final coverage: {arm1_mean:.4f}", flush=True)
    print(
        f"C1 (ARM_1 mean >= {C1_MIN_ARM1_COVERAGE}): {'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"C2 (ARM_1 > ARM_0): {'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "arm0_mean_coverage": arm0_mean,
        "arm1_mean_coverage": arm1_mean,
        "arm0_per_seed_coverages": arm0_coverages,
        "arm1_per_seed_coverages": arm1_coverages,
        "all_seed_results": all_seed_results,
    }


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = (
        f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    )
    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
        / EXPERIMENT_TYPE
    )

    out_path: Path
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
    else:
        out_path = out_dir / f"{run_id}.json"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "n_episodes": N_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "num_basis_functions": NUM_BASIS_FUNCTIONS,
            "arm_harm_scales": ARM_HARM_SCALES,
        },
        "acceptance_criteria": {
            "C1_arm1_min_mean_coverage": C1_MIN_ARM1_COVERAGE,
            "C2_arm1_exceeds_arm0": True,
        },
        "criteria_results": {
            "C1_pass": result["c1_pass"],
            "C2_pass": result["c2_pass"],
        },
        "metrics": {
            "arm0_mean_final_coverage": result["arm0_mean_coverage"],
            "arm1_mean_final_coverage": result["arm1_mean_coverage"],
            "arm0_per_seed_coverages": result["arm0_per_seed_coverages"],
            "arm1_per_seed_coverages": result["arm1_per_seed_coverages"],
        },
        "per_seed_results": result["all_seed_results"],
        "notes": (
            "DEV-NEED-029 warm-start gate diagnostic (ISEF-001). "
            "ARM_0: proximity_harm_scale=0.05 (default cold-start control). "
            "ARM_1: proximity_harm_scale=0.30 (treatment warm-start). "
            "Primary metric: residue_coverage_pct = active_centers / 32 RBF bases. "
            "PASS gates downstream ARC-065 diversity sweeps and Q-043/044/045 retests."
        ),
    }

    if not dry_run:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written", flush=True)
        print(json.dumps(manifest, indent=2), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
