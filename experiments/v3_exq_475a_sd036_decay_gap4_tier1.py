#!/opt/local/bin/python3
"""
V3-EXQ-475a: SD-036 GABAergic decay unlock retest under GAP-4 operating substrate.

Matched to EXQ-475 (gaba/pag vs 471 baseline) but both arms use the GAP-4 goal
pipeline stack; the factorial is gabaergic decay OFF vs ON.

Arms:
  ARM_gap4_no_gaba: gap4 operating, use_gabaergic_decay=False
  ARM_gap4_gaba_on: gap4 operating, use_gabaergic_decay=True, use_pag_freeze_gate=True

Tier-1 acceptance on ARM_gap4_gaba_on vs ARM_gap4_no_gaba (same C1-C4 + C3 lift).

claim_ids: [SD-036, MECH-279]
experiment_purpose: evidence
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

EXPERIMENT_TYPE = "v3_exq_475a_sd036_decay_gap4_tier1"
QUEUE_ID = "V3-EXQ-475a"
CLAIM_IDS = ["SD-036", "MECH-279"]
EXPERIMENT_PURPOSE = "evidence"

ARMS = [
    ArmSpec("ARM_gap4_no_gaba", gap4_operating=True, use_gabaergic_decay=False),
    ArmSpec(
        "ARM_gap4_gaba_on",
        gap4_operating=True,
        use_gabaergic_decay=True,
        use_pag_freeze_gate=True,
    ),
]
GAP4_ARM = "ARM_gap4_gaba_on"
BASE_ARM = "ARM_gap4_no_gaba"


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(dry_run: bool = False) -> Tuple[str, Path] | int:
    seeds = [SEEDS_DEFAULT[0]] if dry_run else SEEDS_DEFAULT
    warmup = 8 if dry_run else WARMUP_EPISODES_DEFAULT
    eval_eps = 2 if dry_run else EVAL_EPISODES_DEFAULT
    steps = 40 if dry_run else STEPS_PER_EPISODE_DEFAULT

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
    per_claim = {
        "SD-036": "supports" if outcome == "PASS" else "weakens",
        "MECH-279": "unknown",
    }
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
        "evidence_direction": "mixed",
        "evidence_direction_per_claim": per_claim,
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
