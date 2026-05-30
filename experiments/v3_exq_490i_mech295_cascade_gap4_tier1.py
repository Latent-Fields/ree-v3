#!/opt/local/bin/python3
"""
V3-EXQ-490i: MECH-295 drive->liking->approach cascade Tier-1 retest under
the rebuilt GAP-4 Tier-1 library (post 2026-05-29 V3-EXQ-490g-cohort autopsy
Fork A library rebuild).

Supersedes V3-EXQ-490h (bit-identical scientific re-run). V3-EXQ-490h ran to
completion on ree-cloud-2 2026-05-29T21:46:08Z (elapsed 5662s; runner reported
FAIL via sentinel) but its manifest never reached disk or the coordinator
`results` table. Root cause is a runner-pipeline bug (FAIL branch of
ree-v3/experiment_runner.py:2299-2367 skipped git_push_results +
coordinator_client.report_result before the 2026-05-29 fix at commit 41c3411;
cloud-2's runner process was already-launched with pre-fix code in memory
when 490h started at 20:11Z), NOT a script-level issue. See autopsy artifact
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-490h-V3-EXQ-592b_2026-05-30.md.
This re-run is on the post-fix runner code; nothing in the experiment itself
has changed and the Outcome Interpretation Grid below applies as written.

Supersedes V3-EXQ-490g (FAIL on the un-rebuilt library; 490g and 490f are now
both stale -- 490g hit the use_dacc-not-set + approach_commit_rate-saturated-
baseline gap; 490f hit the pre-MECH-307 confound).

Rebuilt library:
  - cfg.use_dacc=True is now unconditional in build_config -- agent.dacc is
    instantiated in BOTH gap4_operating arms; C2_dacc_bias can fire.
  - evaluate_tier1_cohort C3_lift_vs_baseline default metric is
    goal_norm_peak delta vs baseline (substrate-side, cross-claim-comparable;
    range 0.09-0.36 observed in 483c/524a). approach_commit_rate retained
    only as a legacy debug knob -- ceiling-saturates at 1.0 in OFF_OFF
    under drive_floor=0.9 + goal_stream + reef.

claim_ids accuracy (CLAUDE.md rule):
  This experiment tests whether the MECH-295 cascade (drive amplification ->
  z_goal seeding -> liking-stream write -> approach-cue bias) produces
  measurable substrate-side and behavioural signal under realistic policy
  state. The rebuilt C3 metric (goal_norm_peak delta) measures the cascade's
  proximal substrate output -- z_goal seeding amplification -- directly,
  which is the load-bearing arithmetic for the MECH-295 implementation.
  C1 (bridge_cue_fires) measures the MECH-295 bridge's cue-side firing.
  Tagging MECH-295 only -- the predecessor 490g's claim_ids were the same
  single tag, and the cascade test does not test MECH-269b, ARC-030,
  MECH-117, or Q-040 directly at this metric resolution.

  Q-040 (cascade dominance vs MECH-269b) is a SEPARATE factorial done by
  the V3-EXQ-490b/c/e/f chain; this experiment is the GAP-4 closure
  retest on a working substrate, not a Q-040 re-factorial.

Arms (2 x 3 seeds):
  ARM_0_legacy_collapsed: z_goal on, drive_floor=0, no goal_stream / bridge stack.
  ARM_1_gap4_operating: full GAP-4 operating config (drive_floor=0.9 + goal_stream
    + relaxed MECH-295 floors + use_dacc=True via the rebuilt library default).

Pre-registered Tier-1 acceptance on ARM_1_gap4_operating (>=2/3 seeds):
  C1 bridge_cue_fires >= 1
  C2 dacc_bias_nonzero_steps >= 1
  C3 approach_commit_steps >= 1
  C4 goal_active_fraction >= 0.05
  C3_lift_vs_baseline: ARM_1 goal_norm_peak > ARM_0 goal_norm_peak +
    TIER1_GOAL_NORM_PEAK_DELTA (0.01) on per-seed basis, >=2/3 seeds.

Outcome interpretation grid (paste into per-claim direction at review):
  PASS (all five clear):
    -> MECH-295 substrate validated on goal_norm_peak delta + behavioural C3.
    -> evidence_direction_per_claim["MECH-295"] = supports.
    -> goal_pipeline:GAP-4 advances toward closure (additionally requires
       490i to land contributory result on origin/master).
  FAIL with C2=False, others PASS:
    -> dACC instantiation in the rebuilt library still not reaching the
       _last_bundle population path -- likely a downstream wiring gap
       under cfg.use_dacc=True. Route to /diagnose-errors targeting
       SD-032b cingulate.dacc bundle population.
  FAIL with C3_lift_vs_baseline=False, C1/C2/C4 PASS:
    -> Substrate fires but goal_norm_peak amplification is below the 0.01
       delta threshold under the GAP-4 operating config. Route to MECH-295
       sub-gain parametric sweep (mech295_drive_to_liking_gain,
       mech295_liking_to_approach_cue_gain) before declaring substrate
       inadequacy.
  FAIL with C3=False (approach_commit_steps=0), C1/C2/C4 PASS:
    -> Substrate amplification reaches the bias channel but is not large
       enough to trip the approach-commit threshold. Route to
       APPROACH_WANTING_THRESH sweep + MECH-295 gain sweep.
  FAIL with C1=False:
    -> mech295_bridge cue-side never fires. Route to MECH-295 bridge
       activation-floor + drive-amplification probe; likely substrate-
       integration regression rather than library gap.

claim_ids: [MECH-295]
experiment_purpose: evidence
supersedes: V3-EXQ-490h (post-runner-pipeline-fix re-run; transitively V3-EXQ-490g)
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

EXPERIMENT_TYPE = "v3_exq_490i_mech295_cascade_gap4_tier1"
QUEUE_ID = "V3-EXQ-490i"
CLAIM_IDS = ["MECH-295"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-490h"

ARMS = [
    ArmSpec("ARM_0_legacy_collapsed", gap4_operating=False),
    ArmSpec("ARM_1_gap4_operating", gap4_operating=True),
]
GAP4_ARM = "ARM_1_gap4_operating"
BASE_ARM = "ARM_0_legacy_collapsed"


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(dry_run: bool = False) -> "Tuple[str, Path] | int":
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
            "use_dacc": True,
            "library_rebuild": "2026-05-29 Fork A (V3-EXQ-490g-cohort autopsy)",
        },
        "acceptance": acceptance,
        "per_run": rows,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
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
