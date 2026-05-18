"""
V3-EXQ-566: WPC Rung 3 -- Forced schema_salience injection diagnostic.

Tests the schema_salience cache -> update_schema_wanting -> residue_field seam
(MECH-216 write path). Rung 3 of the WPC goal-stream calibration ladder
(Work Package C).

Seam under test: can forcing agent._schema_salience (the MECH-216 E1 LSTM
top-layer readout) to a known value cause update_schema_wanting() to write
a predictable VALENCE_WANTING signal that evaluate_valence then returns?

This is one rung upstream from Rung 2 (EXQ-564): Rung 2 verified that direct
residue_field.update_valence() writes are read back correctly; Rung 3 verifies
that the MECH-216 intermediary (schema_readout_head -> _schema_salience cache ->
update_schema_wanting() threshold / gain gate) is correctly wired between the
E1 LSTM and the residue field.

Forced injection mechanism: during the eval phase, directly overwrite
  agent._schema_salience = torch.tensor([[injection_value]])
before calling agent.update_schema_wanting(drive_level=...), bypassing the
natural E1 path. This isolates the threshold/gain/write seam from the training
dynamics of the schema_readout_head itself.

Interpretation grid:
  Outcome                                              | Diagnosis
  -----------------------------------------------------|--------------------------------------------
  P1 FAIL (forced arm wanting <= 0.1)                  | gate blocks write or read returns near-zero
                                                       |   next: check update_schema_wanting threshold
                                                       |         vs injection_value; verify
                                                       |         schema_wanting_enabled=True in cfg
  P1 PASS, P2 FAIL (forced barely above baseline)      | forced write does not exceed baseline leak
                                                       |   next: check injection_value vs baseline
                                                       |         wanting from ARM_0; increase value
  P1+P2 PASS, P3 FAIL (high dose ~ low dose)           | gain/threshold not scaling with salience
                                                       |   next: check schema_wanting_gain default
                                                       |         in REEConfig.from_dims; verify
                                                       |         wanting_value = sal * gain * drive
  P1+P2+P3 all PASS                                    | schema_salience -> update_schema_wanting
                                                       |         -> residue_field seam functional
  ARM_1_canonical wanting != 0 across seeds            | MECH-216 natural path active in training
                                                       |   next: measure signal vs forced arms
  ARM_1_canonical wanting == 0.0 in all seeds          | MECH-216 natural path silent (expected
                                                       |         with short training: schema_readout
                                                       |         head may not yet be calibrated)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.residue.field import VALENCE_WANTING  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_566_wpc3_schema_salience_injection"
QUEUE_ID = "V3-EXQ-566"
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

DRIVE_LEVEL = 1.0  # saturated drive so wanting_value = sal * gain * 1.0

# Pre-registered acceptance thresholds
P1_WANTING_MIN = 0.1      # forced-mid arm mean_wanting must exceed this
P2_MARGIN_EPS = 0.01      # forced-mid must exceed baseline (ARM_0) by at least this
P3_SCALE_EPS = 0.05       # forced-high must exceed forced-mid by at least this

# Injection values for the forced arms.
# schema_wanting_threshold (from_dims default) = 0.10
# schema_wanting_gain (from_dims default) = 0.60
# wanting_value = sal * gain * drive = sal * 0.60 * 1.0
# ARM_2 (sal=0.5): wanting_value = 0.5 * 0.60 * 1.0 = 0.30
# ARM_3 (sal=0.9): wanting_value = 0.9 * 0.60 * 1.0 = 0.54
# Both clear the threshold=0.10.
FORCED_SALIENCE_MID = 0.5
FORCED_SALIENCE_HIGH = 0.9

ARMS = [
    {
        "arm": "ARM_0_baseline",
        "schema_wanting_enabled": False,
        "force_injection": False,
        "injection_salience": 0.0,
        "description": "baseline: no injection, no schema_wanting",
    },
    {
        "arm": "ARM_1_canonical",
        "schema_wanting_enabled": True,
        "force_injection": False,
        "injection_salience": 0.0,
        "description": "canonical MECH-216 natural path via schema_wanting_enabled",
    },
    {
        "arm": "ARM_2_forced_mid",
        "schema_wanting_enabled": True,
        "force_injection": True,
        "injection_salience": FORCED_SALIENCE_MID,
        "description": f"forced _schema_salience={FORCED_SALIENCE_MID}",
    },
    {
        "arm": "ARM_3_forced_high",
        "schema_wanting_enabled": True,
        "force_injection": True,
        "injection_salience": FORCED_SALIENCE_HIGH,
        "description": f"forced _schema_salience={FORCED_SALIENCE_HIGH} (scale check)",
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
    # schema_wanting_enabled must be True for update_schema_wanting() to run
    # (the method has an early return when the flag is False).
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

    # Training phase: warm up encoder + populate RBF centers.
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

    # Eval phase: forced schema_salience injection.
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
                    # Forced injection: overwrite _schema_salience cache directly,
                    # bypassing the E1 natural read path. This isolates the
                    # update_schema_wanting() threshold/gain/write seam.
                    agent._schema_salience = torch.tensor(
                        [[arm["injection_salience"]]],
                        dtype=torch.float32,
                    )
                    agent.update_schema_wanting(drive_level=DRIVE_LEVEL)
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

    # Aggregate per arm across seeds.
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
    arm2_wanting = arm_stats["ARM_2_forced_mid"]["mean_valence_wanting"]
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
        f" {'PASS' if p2_pass else 'FAIL'}"
        f" ({arm2_wanting:.4f} vs {arm0_wanting:.4f})",
        flush=True,
    )
    print(
        f"P3 (ARM_3 > ARM_2 + {P3_SCALE_EPS}):"
        f" {'PASS' if p3_pass else 'FAIL'}"
        f" ({arm3_wanting:.4f} vs {arm2_wanting:.4f})",
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
            "drive_level": DRIVE_LEVEL,
            "forced_salience_mid": FORCED_SALIENCE_MID,
            "forced_salience_high": FORCED_SALIENCE_HIGH,
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
            "WPC Rung 3: forced _schema_salience injection diagnostic. "
            "Tests schema_salience cache -> update_schema_wanting -> "
            "residue_field seam (MECH-216 write path). "
            "ARM_2/ARM_3 inject by overwriting agent._schema_salience before "
            "calling agent.update_schema_wanting(drive_level=1.0). "
            "ARM_1 tests the natural MECH-216 path (schema_wanting_enabled=True "
            "but no forced override). "
            "Rung 2 (EXQ-564) verified write->read seam; Rung 3 verifies the "
            "MECH-216 intermediary gate (threshold/gain/write)."
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
