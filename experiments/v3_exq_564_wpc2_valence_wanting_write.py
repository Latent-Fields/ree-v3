"""
V3-EXQ-564: WPC Rung 2 -- Forced VALENCE_WANTING write diagnostic.

Tests the write->read seam in the ResidueField valence pipeline.
Rung 2 of the WPC goal-stream calibration ladder (Work Package C).

Seam under test: after training populates RBF centers, can a direct call to
ResidueField.update_valence(z_world, VALENCE_WANTING, value) inject a signal
that evaluate_valence then returns?

Interpretation grid:
  Outcome                                                | Diagnosis
  -------------------------------------------------------|-------------------------------------------
  P1 FAIL (ARM_2 wanting <= 0.1)                        | write fires but read returns near-zero
                                                         |   next: check RBF center population;
                                                         |         verify update_valence sign/scale
  P1 PASS, P2 FAIL (ARM_2 barely above ARM_0)           | forced write does not exceed baseline
                                                         |   next: check VALENCE_WANTING index;
                                                         |         verify evaluate_valence indexing
  P1+P2 PASS, P3 FAIL (ARM_3 ~ ARM_2)                  | write fires but value magnitude ignored
                                                         |   next: check evaluate_valence
                                                         |         normalisation or clamping
  P1+P2+P3 all PASS                                     | write->read seam functional
                                                         |   next: proceed to Rung 3 (schema_salience)
  ARM_1_canonical wanting != 0 across seeds             | MECH-216 natural path active
                                                         |   next: measure effect size vs forced arm
  ARM_1_canonical wanting == 0.0 in all seeds           | MECH-216 natural path silent
                                                         |   next: diagnose schema_wanting_enabled
                                                         |         wiring (E1Config vs REEConfig)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.residue.field import VALENCE_WANTING  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_564_wpc2_valence_wanting_write"
QUEUE_ID = "V3-EXQ-564"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7]
TRAIN_EPISODES = 30
EVAL_EPISODES = 10
STEPS_PER_EPISODE = 120

DRY_RUN_SEEDS = [42]
DRY_RUN_TRAIN_EPISODES = 3
DRY_RUN_EVAL_EPISODES = 2
DRY_RUN_STEPS = 50

# Pre-registered acceptance thresholds
P1_WANTING_MIN = 0.1        # ARM_2 mean_valence_wanting must exceed this
P2_MARGIN_EPS = 0.01        # ARM_2 must exceed ARM_0 by at least this
P3_SCALE_EPS = 0.05         # ARM_3 must exceed ARM_2 by at least this

ARMS = [
    {
        "arm": "ARM_0_baseline",
        "schema_wanting_enabled": False,
        "force_injection": False,
        "injection_value": 0.0,
        "description": "baseline: no injection, no schema_wanting",
    },
    {
        "arm": "ARM_1_canonical",
        "schema_wanting_enabled": True,
        "force_injection": False,
        "injection_value": 0.0,
        "description": "canonical MECH-216 natural write via schema_wanting_enabled",
    },
    {
        "arm": "ARM_2_forced_low",
        "schema_wanting_enabled": False,
        "force_injection": True,
        "injection_value": 1.0,
        "description": "forced VALENCE_WANTING write at injection_value=1.0",
    },
    {
        "arm": "ARM_3_forced_high",
        "schema_wanting_enabled": False,
        "force_injection": True,
        "injection_value": 5.0,
        "description": "forced VALENCE_WANTING write at injection_value=5.0 (scale check)",
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


def _make_config(env: CausalGridWorld, arm: Dict[str, Any]) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        z_goal_enabled=True,
    )
    cfg.e1.schema_wanting_enabled = arm["schema_wanting_enabled"]
    return cfg


def _run_arm_seed(
    arm: Dict[str, Any],
    seed: int,
    train_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    env = _make_env(seed)
    cfg = _make_config(env, arm)
    agent = REEAgent(cfg)

    # Training phase: warm up encoder + populate RBF centers
    train_harness = StepHarness(agent, env, train_mode=True, seed=seed)
    for ep in range(train_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        train_harness.reset()
        for _ in range(steps_per_episode):
            result = train_harness.step(obs_dict)
            obs_dict = result.next_obs_dict
            if result.done:
                break
        if (ep + 1) % 10 == 0 or (ep + 1) == train_episodes:
            print(
                f"  [train] seed={seed} arm={arm['arm']} ep {ep + 1}/{train_episodes}",
                flush=True,
            )

    n_centers_post_train = int(
        agent.residue_field.rbf_field.active_mask.sum().item()
    )

    # Eval phase: inject and read back VALENCE_WANTING
    eval_harness = StepHarness(agent, env, train_mode=False, seed=seed)
    wanting_readings: List[float] = []
    n_centers_readings: List[int] = []
    steps_with_injection: int = 0
    steps_total: int = 0

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        eval_harness.reset()
        for _ in range(steps_per_episode):
            result = eval_harness.step(obs_dict)
            obs_dict = result.next_obs_dict

            z_world = getattr(result.latent, "z_world", None)
            if z_world is not None:
                n_centers = int(
                    agent.residue_field.rbf_field.active_mask.sum().item()
                )
                n_centers_readings.append(n_centers)
                steps_total += 1

                if arm["force_injection"] and n_centers > 0:
                    agent.residue_field.update_valence(
                        z_world.detach(),
                        VALENCE_WANTING,
                        arm["injection_value"],
                    )
                    steps_with_injection += 1

                valence = agent.residue_field.evaluate_valence(z_world.detach())
                wanting = float(valence[..., VALENCE_WANTING].mean().item())
                wanting_readings.append(wanting)

            if result.done:
                break

    mean_wanting = (
        float(sum(wanting_readings) / len(wanting_readings))
        if wanting_readings
        else 0.0
    )
    mean_n_centers = (
        float(sum(n_centers_readings) / len(n_centers_readings))
        if n_centers_readings
        else 0.0
    )
    injection_coverage = (
        float(steps_with_injection / steps_total) if steps_total > 0 else 0.0
    )

    return {
        "arm": arm["arm"],
        "seed": seed,
        "mean_valence_wanting": mean_wanting,
        "mean_n_active_centers": mean_n_centers,
        "n_centers_post_train": n_centers_post_train,
        "injection_coverage": injection_coverage,
        "steps_total": steps_total,
    }


def main(dry_run: bool = False):
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    train_episodes = DRY_RUN_TRAIN_EPISODES if dry_run else TRAIN_EPISODES
    eval_episodes = DRY_RUN_EVAL_EPISODES if dry_run else EVAL_EPISODES
    steps_per_episode = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            arm_result = _run_arm_seed(
                arm, seed, train_episodes, eval_episodes, steps_per_episode
            )
            all_results.append(arm_result)
            print(
                f"  result seed={seed} arm={arm['arm']}"
                f" mean_wanting={arm_result['mean_valence_wanting']:.4f}"
                f" n_centers={arm_result['mean_n_active_centers']:.1f}"
                f" inj_cov={arm_result['injection_coverage']:.2f}",
                flush=True,
            )
            print("verdict: PASS", flush=True)

    # Aggregate per arm across seeds
    arm_stats: Dict[str, Dict[str, float]] = {}
    for arm in ARMS:
        arm_name = arm["arm"]
        arm_rows = [r for r in all_results if r["arm"] == arm_name]
        arm_stats[arm_name] = {
            "mean_valence_wanting": (
                sum(r["mean_valence_wanting"] for r in arm_rows) / len(arm_rows)
            ),
            "mean_n_active_centers": (
                sum(r["mean_n_active_centers"] for r in arm_rows) / len(arm_rows)
            ),
            "mean_injection_coverage": (
                sum(r["injection_coverage"] for r in arm_rows) / len(arm_rows)
            ),
        }

    arm0_wanting = arm_stats["ARM_0_baseline"]["mean_valence_wanting"]
    arm2_wanting = arm_stats["ARM_2_forced_low"]["mean_valence_wanting"]
    arm3_wanting = arm_stats["ARM_3_forced_high"]["mean_valence_wanting"]

    p1_pass = arm2_wanting > P1_WANTING_MIN
    p2_pass = arm2_wanting > arm0_wanting + P2_MARGIN_EPS
    p3_pass = arm3_wanting > arm2_wanting + P3_SCALE_EPS

    outcome = "PASS" if (p1_pass and p2_pass and p3_pass) else "FAIL"

    print(
        f"P1 (ARM_2 wanting > {P1_WANTING_MIN}):"
        f" {'PASS' if p1_pass else 'FAIL'} ({arm2_wanting:.4f})",
        flush=True,
    )
    print(
        f"P2 (ARM_2 > ARM_0 + {P2_MARGIN_EPS}):"
        f" {'PASS' if p2_pass else 'FAIL'} ({arm2_wanting:.4f} vs {arm0_wanting:.4f})",
        flush=True,
    )
    print(
        f"P3 (ARM_3 > ARM_2 + {P3_SCALE_EPS}):"
        f" {'PASS' if p3_pass else 'FAIL'} ({arm3_wanting:.4f} vs {arm2_wanting:.4f})",
        flush=True,
    )
    print(f"Overall outcome: {outcome}", flush=True)

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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": seeds,
            "train_episodes": train_episodes,
            "eval_episodes": eval_episodes,
            "steps_per_episode": steps_per_episode,
        },
        "acceptance_criteria": {
            "P1_arm2_wanting_min": P1_WANTING_MIN,
            "P2_margin_eps": P2_MARGIN_EPS,
            "P3_scale_eps": P3_SCALE_EPS,
        },
        "criteria_results": {
            "P1_pass": p1_pass,
            "P2_pass": p2_pass,
            "P3_pass": p3_pass,
        },
        "arm_stats": arm_stats,
        "per_seed_arm_results": all_results,
        "notes": (
            "WPC Rung 2: forced VALENCE_WANTING write diagnostic. "
            "Tests write->read seam in ResidueField valence pipeline. "
            "ARM_2/ARM_3 inject via update_valence(); ARM_1 tests natural MECH-216 path."
        ),
    }

    if not dry_run:
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
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
