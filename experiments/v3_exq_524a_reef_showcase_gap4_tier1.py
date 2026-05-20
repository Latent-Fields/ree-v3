#!/opt/local/bin/python3
"""
V3-EXQ-524a: SD-054 reef fishtank showcase under GAP-4 operating substrate.

Re-runs EXQ-524 reef env (ARM_1_reef_food) with goal_stream + drive_floor=0.9 +
StepHarness. Single ARM_gap4_operating; Tier-1 C1-C4 must pass in >=2/3 seeds
(no legacy arm -- reef showcase is eval-only against tier1 thresholds).

claim_ids: [] (diagnostic showcase; SD-054 behavioural gate is informational)
experiment_purpose: diagnostic
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
    ENV_REEF_KWARGS,
    EVAL_EPISODES_DEFAULT,
    SEEDS_DEFAULT,
    STEPS_PER_EPISODE_DEFAULT,
    WARMUP_EPISODES_DEFAULT,
    run_seed_arm,
    tier1_seed_pass,
)

EXPERIMENT_TYPE = "v3_exq_524a_reef_showcase_gap4_tier1"
QUEUE_ID = "V3-EXQ-524a"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

ARM = ArmSpec("ARM_gap4_reef", gap4_operating=True)


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(dry_run: bool = False) -> Tuple[str, Path] | int:
    seeds = [SEEDS_DEFAULT[0]] if dry_run else SEEDS_DEFAULT
    warmup = 6 if dry_run else WARMUP_EPISODES_DEFAULT
    eval_eps = 2 if dry_run else EVAL_EPISODES_DEFAULT
    steps = 30 if dry_run else STEPS_PER_EPISODE_DEFAULT

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        rows.append(
            run_seed_arm(
                seed,
                ARM,
                env_kwargs=ENV_REEF_KWARGS,
                warmup_episodes=warmup,
                eval_episodes=eval_eps,
                steps_per_episode=steps,
            )
        )

    passes = [tier1_seed_pass(r) for r in rows]
    n_ok = sum(1 for p in passes if all(p.values()))
    acceptance = {
        "pass": n_ok >= 2,
        "seeds_passing_all_tier1": n_ok,
        "per_seed_checks": passes,
    }
    outcome = "PASS" if acceptance["pass"] else "FAIL"
    elapsed = time.time() - t0

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run outcome={outcome}", flush=True)
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
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_compact(),
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Reef fishtank diagnostic under GAP-4 operating substrate. "
            "Tier-1 cascade thresholds only; no claim_ids."
        ),
        "acceptance": acceptance,
        "per_run": rows,
        "elapsed_seconds": elapsed,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
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
