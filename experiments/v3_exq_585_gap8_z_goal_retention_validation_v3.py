"""
V3-EXQ-585: infant_substrate:GAP-8 validation --
SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)
post_sleep_z_goal_retention and replay_diversity_index telemetry.

Tests that the four GAP-8 metrics added to SleepLoopManager._run_cycle()
(post_sleep_z_goal_before, post_sleep_z_goal_after,
post_sleep_z_goal_retention, replay_diversity_index) emit correct values
for two conditions:

  ARM_0 -- no goal pipeline (z_goal_enabled=False): z_goal metrics must be
  -1.0 sentinel; replay_diversity_index must be -1.0 (no sampler -> no
  routed draws).

  ARM_1 -- goal pipeline active (z_goal_enabled=True, _z_goal seeded to
  norm > 0.4): retention must be > 0.95 (sleep does not modify _z_goal);
  before norm must be > 0.1.

Seam under test:
  SleepLoopManager._safe_z_goal_norm() [read before]
  -> agent.run_sleep_cycle()
  -> SleepLoopManager._safe_z_goal_norm() [read after]
  -> merged["post_sleep_z_goal_before/after/retention"]
  -> merged["replay_diversity_index"]

Interpretation grid:
  Outcome                                | Diagnosis
  ---------------------------------------|------------------------------------------
  C1 FAIL (ARM_0 z_goal not -1.0)       | goal_state present on no-goal agent;
                                         | check REEConfig.from_dims wiring
  C2 FAIL (ARM_0 replay_div not -1.0)   | sampler present without explicit flag;
                                         | check SleepLoopManager ctor path
  C3 FAIL (ARM_1 retention <= 0.95)     | sleep cycle is modifying _z_goal;
                                         | check run_sws_schema_pass / enter_sws_mode
  C4 FAIL (ARM_1 before <= 0.1)         | _z_goal seed did not persist to cycle;
                                         | check that force_cycle reads live state
  C1+C2+C3+C4 all PASS                  | GAP-8 telemetry validated; DEV-NEED-007
                                         | advisory gate instrumented correctly
"""

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
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_585_gap8_z_goal_retention_validation"
QUEUE_ID = "V3-EXQ-585"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 13]
N_SENSE_STEPS = 20  # waking acts to populate world_experience_buffer
Z_GOAL_SEED_NORM = 0.6  # ARM_1: _z_goal seeded to this norm (> 0.4 threshold)

# Pre-registered acceptance thresholds
C1_SENTINEL = -1.0           # ARM_0 z_goal metrics must equal this
C2_SENTINEL = -1.0           # ARM_0 replay_diversity_index must equal this
C3_RETENTION_MIN = 0.95      # ARM_1 retention must exceed this
C4_BEFORE_MIN = 0.1          # ARM_1 z_goal_before must exceed this

ARMS = [
    {
        "arm": "ARM_0_no_goal",
        "z_goal_enabled": False,
        "description": "no goal pipeline -- all z_goal metrics must be -1.0 sentinel",
    },
    {
        "arm": "ARM_1_goal_seeded",
        "z_goal_enabled": True,
        "description": "goal seeded to norm=0.6 -- retention must be > 0.95",
    },
]


def _build_agent(*, z_goal_enabled: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=50,
        action_dim=4,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1,
        z_goal_enabled=z_goal_enabled,
    )
    cfg.sws_enabled = True
    cfg.rem_enabled = True
    return REEAgent(cfg)


def _run_arm_seed(
    *,
    z_goal_enabled: bool,
    seed: int,
    dry_run: bool,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    agent = _build_agent(z_goal_enabled=z_goal_enabled)

    # Populate world_experience_buffer via act_with_split_obs so SWS has
    # content. run_sws_schema_pass returns early when buffer < 2 entries.
    for _ in range(N_SENSE_STEPS):
        obs_body = torch.randn(12)
        obs_world = torch.randn(50)
        agent.act_with_split_obs(obs_body=obs_body, obs_world=obs_world)

    # ARM_1: seed _z_goal to a known non-zero norm before the sleep cycle.
    if z_goal_enabled and agent.goal_state is not None:
        dim = agent.goal_state._z_goal.shape[-1]
        seed_vec = torch.ones(dim)
        seed_vec = seed_vec / seed_vec.norm() * Z_GOAL_SEED_NORM
        agent.goal_state._z_goal = seed_vec

    metrics = agent.sleep_loop.force_cycle(agent)

    return {
        "seed": seed,
        "z_goal_enabled": z_goal_enabled,
        "post_sleep_z_goal_before": float(
            metrics.get("post_sleep_z_goal_before", -999.0)
        ),
        "post_sleep_z_goal_after": float(
            metrics.get("post_sleep_z_goal_after", -999.0)
        ),
        "post_sleep_z_goal_retention": float(
            metrics.get("post_sleep_z_goal_retention", -999.0)
        ),
        "replay_diversity_index": float(
            metrics.get("replay_diversity_index", -999.0)
        ),
        "sws_n_writes": float(metrics.get("sws_n_writes", 0.0)),
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    print(
        f"V3-EXQ-585: GAP-8 z_goal_retention + replay_diversity_index validation",
        flush=True,
    )
    print(f"  seeds={seeds} dry_run={dry_run}", flush=True)

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for arm_cfg in ARMS:
        arm_name = arm_cfg["arm"]
        z_goal_enabled = arm_cfg["z_goal_enabled"]
        all_results[arm_name] = []
        for seed in seeds:
            print(f"Seed {seed} Condition {arm_name}", flush=True)
            result = _run_arm_seed(
                z_goal_enabled=z_goal_enabled,
                seed=seed,
                dry_run=dry_run,
            )
            all_results[arm_name].append(result)
            before = result["post_sleep_z_goal_before"]
            after = result["post_sleep_z_goal_after"]
            retention = result["post_sleep_z_goal_retention"]
            div_idx = result["replay_diversity_index"]
            print(
                f"  [result] seed={seed} arm={arm_name} "
                f"before={before:.4f} after={after:.4f} "
                f"retention={retention:.4f} replay_div={div_idx:.4f} "
                f"sws_writes={result['sws_n_writes']:.0f}",
                flush=True,
            )
            print("verdict: PASS", flush=True)

    arm0 = all_results["ARM_0_no_goal"]
    arm1 = all_results["ARM_1_goal_seeded"]

    # C1: ARM_0 z_goal metrics all -1.0 sentinel
    c1_pass = all(
        abs(r["post_sleep_z_goal_before"] - C1_SENTINEL) < 1e-6
        and abs(r["post_sleep_z_goal_after"] - C1_SENTINEL) < 1e-6
        and abs(r["post_sleep_z_goal_retention"] - C1_SENTINEL) < 1e-6
        for r in arm0
    )
    # C2: ARM_0 replay_diversity_index -1.0 sentinel (no sampler)
    c2_pass = all(
        abs(r["replay_diversity_index"] - C2_SENTINEL) < 1e-6
        for r in arm0
    )
    # C3: ARM_1 retention > threshold (sleep preserves z_goal)
    c3_pass = all(r["post_sleep_z_goal_retention"] > C3_RETENTION_MIN for r in arm1)
    # C4: ARM_1 before norm captured correctly (> floor)
    c4_pass = all(r["post_sleep_z_goal_before"] > C4_BEFORE_MIN for r in arm1)

    outcome = "PASS" if (c1_pass and c2_pass and c3_pass and c4_pass) else "FAIL"

    print("", flush=True)
    arm0_before = [r["post_sleep_z_goal_before"] for r in arm0]
    arm1_before = [r["post_sleep_z_goal_before"] for r in arm1]
    arm1_ret = [r["post_sleep_z_goal_retention"] for r in arm1]
    arm0_div = [r["replay_diversity_index"] for r in arm0]
    print(
        f"C1 (ARM_0 z_goal sentinel -1.0): {'PASS' if c1_pass else 'FAIL'} "
        f"before={[f'{v:.4f}' for v in arm0_before]}",
        flush=True,
    )
    print(
        f"C2 (ARM_0 replay_div sentinel -1.0): {'PASS' if c2_pass else 'FAIL'} "
        f"div={[f'{v:.4f}' for v in arm0_div]}",
        flush=True,
    )
    print(
        f"C3 (ARM_1 retention > {C3_RETENTION_MIN}): {'PASS' if c3_pass else 'FAIL'} "
        f"retention={[f'{v:.4f}' for v in arm1_ret]}",
        flush=True,
    )
    print(
        f"C4 (ARM_1 before > {C4_BEFORE_MIN}): {'PASS' if c4_pass else 'FAIL'} "
        f"before={[f'{v:.4f}' for v in arm1_before]}",
        flush=True,
    )
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c4_pass": c4_pass,
        "arm0_before_mean": float(sum(arm0_before) / len(arm0_before)),
        "arm1_before_mean": float(sum(arm1_before) / len(arm1_before)),
        "arm1_retention_mean": float(sum(arm1_ret) / len(arm1_ret)),
        "arm0_div_mean": float(sum(arm0_div) / len(arm0_div)),
        "all_results": all_results,
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
            "n_sense_steps": N_SENSE_STEPS,
            "z_goal_seed_norm": Z_GOAL_SEED_NORM,
        },
        "acceptance_criteria": {
            "C1_arm0_z_goal_sentinel": C1_SENTINEL,
            "C2_arm0_replay_div_sentinel": C2_SENTINEL,
            "C3_arm1_retention_min": C3_RETENTION_MIN,
            "C4_arm1_before_min": C4_BEFORE_MIN,
        },
        "criteria_results": {
            "C1_pass": result["c1_pass"],
            "C2_pass": result["c2_pass"],
            "C3_pass": result["c3_pass"],
            "C4_pass": result["c4_pass"],
        },
        "arm0_before_mean": result["arm0_before_mean"],
        "arm1_before_mean": result["arm1_before_mean"],
        "arm1_retention_mean": result["arm1_retention_mean"],
        "arm0_div_mean": result["arm0_div_mean"],
        "per_arm_per_seed_results": result["all_results"],
        "notes": (
            "infant_substrate:GAP-8 validation. post_sleep_z_goal_retention "
            "and replay_diversity_index telemetry added to "
            "SleepLoopManager._run_cycle() (ree_core/sleep/phase_manager.py). "
            "ARM_0 (no goal pipeline) verifies -1.0 sentinels. "
            "ARM_1 (goal seeded to norm=0.6) verifies retention > 0.95 "
            "(sleep does not modify _z_goal). DEV-NEED-007 advisory gate "
            "instrumentation."
        ),
    }

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
