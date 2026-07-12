#!/opt/local/bin/python3
"""
V3-EXQ-490g: MECH-295 drive->liking->approach cascade Tier-1 retest (goal_pipeline:GAP-4).

Supersedes the pre-GAP-3 / pre-StepHarness EXQ-490f confound chain. Runs under the
GAP-4 operating substrate: drive_floor=0.9 (MECH-306 / V3-EXQ-582a), canonical
goal_stream bundle (MECH-307 + MECH-295 + schema wanting), StepHarness eval loop,
main-path SP-CEM defaults.

Arms (2 x 3 seeds):
  ARM_0_legacy_collapsed: z_goal on, drive_floor=0, no goal_stream / bridge stack.
  ARM_1_gap4_operating: full GAP-4 operating config.

Pre-registered Tier-1 acceptance on ARM_1_gap4_operating (>=2/3 seeds):
  C1 bridge_cue_fires >= 1
  C2 dacc_bias_nonzero_steps >= 1
  C3 approach_commit_steps >= 1
  C4 goal_active_fraction >= 0.05
  C3b approach_commit_rate > ARM_0 per seed (>=2/3 seeds)

claim_ids: [MECH-295]
experiment_purpose: evidence
supersedes: V3-EXQ-490f
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from _lib.goal_pipeline_tier1 import (  # noqa: E402
    ArmSpec,
    ENV_FISHTANK_KWARGS,
    EVAL_EPISODES_DEFAULT,
    SEEDS_DEFAULT,
    STEPS_PER_EPISODE_DEFAULT,
    WARMUP_EPISODES_DEFAULT,
    evaluate_tier1_cohort,
    run_seed_arm,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_490g_mech295_cascade_gap4_tier1"
QUEUE_ID = "V3-EXQ-490g"
CLAIM_IDS = ["MECH-295"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-490f"

ARMS = [
    ArmSpec("ARM_0_legacy_collapsed", gap4_operating=False),
    ArmSpec("ARM_1_gap4_operating", gap4_operating=True),
]
GAP4_ARM = "ARM_1_gap4_operating"
BASE_ARM = "ARM_0_legacy_collapsed"


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(dry_run: bool = False) -> Tuple[str, Path] | int:
    seeds = [SEEDS_DEFAULT[0]] if dry_run else SEEDS_DEFAULT
    warmup = 8 if dry_run else WARMUP_EPISODES_DEFAULT
    eval_eps = 2 if dry_run else EVAL_EPISODES_DEFAULT
    steps = 40 if dry_run else STEPS_PER_EPISODE_DEFAULT

    print(f"[{EXPERIMENT_TYPE}] dry_run={dry_run}", flush=True)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        for arm in ARMS:
            rows.append(
                run_seed_arm(
                    seed,
                    arm,
                    env_kwargs=ENV_FISHTANK_KWARGS,
                    warmup_episodes=warmup,
                    eval_episodes=eval_eps,
                    steps_per_episode=steps,
                )
            )

    acceptance = evaluate_tier1_cohort(
        rows, gap4_arm_id=GAP4_ARM, baseline_arm_id=BASE_ARM
    )
    outcome = "PASS" if acceptance["pass"] else "FAIL"
    elapsed = time.time() - t0
    print(f"[{EXPERIMENT_TYPE}] acceptance={acceptance}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s", flush=True)

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; no manifest.", flush=True)
        return 0

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    out_dir = (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_compact(),
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": {
            "MECH-295": "supports" if outcome == "PASS" else "weakens",
        },
        "elapsed_seconds": elapsed,
        "gap4_operating": {
            "drive_floor": 0.9,
            "drive_ema_alpha": 1.0,
            "goal_stream": True,
        },
        "acceptance": acceptance,
        "per_run": rows,
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if result == 0:
        sys.exit(0)
    outcome, out_path = result
    emit_outcome(outcome=outcome, manifest_path=out_path)
    sys.exit(0)
