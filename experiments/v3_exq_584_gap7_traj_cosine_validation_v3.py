"""
V3-EXQ-584: infant_substrate:GAP-7 traj_pairwise_cosine_mean substrate-readiness diagnostic.

Two-arm diagnostic validating that the traj_pairwise_cosine_mean metric
(infant_substrate:GAP-7) is correctly implemented in CausalGridWorldV2.

ARM_0 (off): traj_telemetry_enabled=False.
  All steps must return traj_pairwise_cosine_mean==-1.0, traj_n_episodes_stored==0,
  traj_telemetry_enabled==False.

ARM_1 (on): traj_telemetry_enabled=True (default params).
  5 episodes with random actions per seed.
  C0: After ep1 done step -- traj_n_episodes_stored==1 AND metric==-1.0 (sentinel;
      store has 1 entry, < 2 needed to compute).
  C1: After ep2 done step -- traj_n_episodes_stored==2 AND metric in [0.0, 1.0].
  C2: After ep3 done step -- traj_n_episodes_stored==3 AND metric in [0.0, 1.0].
  C3: Every mid-episode step returns the same metric value as the one cached at the
      end of the previous episode (metric only updates on done steps).
  C4: Mean metric across seeds > 0.0 (random actions produce non-identical trajectories).

PASS = ARM_0 sentinel OK AND all C0-C4 hold across all 3 seeds.

experiment_purpose: diagnostic (substrate readiness test; no governance weighting).
claim_ids: [] (env telemetry validation only).
Unblocks: infant_substrate:GAP-7, DEV-NEED-002, DEV-NEED-005.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiment_protocol import emit_outcome
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from pathlib import Path  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
QUEUE_ID = "V3-EXQ-584"

SEEDS = [0, 1, 2]
N_EPISODES = 5
SIZE = 12
NUM_HAZARDS = 2
NUM_RESOURCES = 2


# --------------------------------------------------------------------------- #
# ARM helpers                                                                  #
# --------------------------------------------------------------------------- #

def run_arm_0(seed):
    """Sentinel checks: traj_telemetry_enabled=False."""
    rng = np.random.default_rng(seed)
    env = CausalGridWorldV2(
        size=SIZE,
        seed=seed,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        traj_telemetry_enabled=False,
    )
    env.reset()
    c_enabled_ok = True
    c_stored_ok = True
    c_value_ok = True
    ep = 0
    while ep < N_EPISODES:
        action = int(rng.integers(0, 4))
        _, _, done, info, _ = env.step(action)
        if info["traj_telemetry_enabled"] is not False:
            c_enabled_ok = False
        if info["traj_n_episodes_stored"] != 0:
            c_stored_ok = False
        if info["traj_pairwise_cosine_mean"] != -1.0:
            c_value_ok = False
        if done:
            ep += 1
            print(
                f"  [train] ARM_0 seed={seed} ep {ep}/{N_EPISODES} done",
                flush=True,
            )
            if ep < N_EPISODES:
                env.reset()
    passed = c_enabled_ok and c_stored_ok and c_value_ok
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return {
        "passed": passed,
        "c_enabled_ok": c_enabled_ok,
        "c_stored_ok": c_stored_ok,
        "c_value_ok": c_value_ok,
    }


def run_arm_1(seed):
    """Accumulation and timing checks: traj_telemetry_enabled=True."""
    rng = np.random.default_rng(seed + 100)
    env = CausalGridWorldV2(
        size=SIZE,
        seed=seed,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        traj_telemetry_enabled=True,
    )
    c0_ok = False
    c1_ok = False
    c2_ok = False
    c3_ok = True
    end_metrics = []

    for ep_idx in range(N_EPISODES):
        env.reset()
        prev_metric = float(env._traj_pairwise_cosine_mean)
        while True:
            action = int(rng.integers(0, 4))
            _, _, done, info, _ = env.step(action)
            metric = float(info["traj_pairwise_cosine_mean"])
            stored = int(info["traj_n_episodes_stored"])
            ep_num = ep_idx + 1
            if not done:
                if metric != prev_metric:
                    c3_ok = False
            if done:
                if ep_num == 1:
                    c0_ok = (stored == 1 and metric == -1.0)
                elif ep_num == 2:
                    c1_ok = (stored == 2 and 0.0 <= metric <= 1.0)
                elif ep_num == 3:
                    c2_ok = (stored == 3 and 0.0 <= metric <= 1.0)
                if stored >= 2 and metric != -1.0:
                    end_metrics.append(metric)
                print(
                    f"  [train] ARM_1 seed={seed} ep {ep_num}/{N_EPISODES} done"
                    f" stored={stored} metric={metric:.4f}",
                    flush=True,
                )
                break

    c4_ok = len(end_metrics) > 0 and float(np.mean(end_metrics)) > 0.0
    passed = c0_ok and c1_ok and c2_ok and c3_ok and c4_ok
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return {
        "passed": passed,
        "c0_ok": c0_ok,
        "c1_ok": c1_ok,
        "c2_ok": c2_ok,
        "c3_ok": c3_ok,
        "c4_ok": c4_ok,
        "end_metrics": [float(v) for v in end_metrics],
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run_experiment():
    arm0_results = {}
    arm1_results = {}

    for seed in SEEDS:
        print(f"Seed {seed} Condition ARM_0", flush=True)
        arm0_results[seed] = run_arm_0(seed)
        print(f"Seed {seed} Condition ARM_1", flush=True)
        arm1_results[seed] = run_arm_1(seed)

    all_arm0_pass = all(r["passed"] for r in arm0_results.values())
    all_arm1_pass = all(r["passed"] for r in arm1_results.values())
    all_pass = all_arm0_pass and all_arm1_pass
    outcome = "PASS" if all_pass else "FAIL"

    all_end_metrics = [
        v for r in arm1_results.values() for v in r.get("end_metrics", [])
    ]
    mean_metric_arm1 = float(np.mean(all_end_metrics)) if all_end_metrics else -1.0

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_584_gap7_traj_cosine_validation_{ts}_v3"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_type": "v3_exq_584_gap7_traj_cosine_validation",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": [],
        "outcome": outcome,
        "timestamp_utc": ts,
        "seeds": SEEDS,
        "n_episodes_per_arm": N_EPISODES,
        "mean_metric_arm1": mean_metric_arm1,
        "all_arm0_pass": all_arm0_pass,
        "all_arm1_pass": all_arm1_pass,
        "c0_sentinel_ok": all(r["c0_ok"] for r in arm1_results.values()),
        "c1_ep2_valid": all(r["c1_ok"] for r in arm1_results.values()),
        "c2_ep3_valid": all(r["c2_ok"] for r in arm1_results.values()),
        "c3_no_mid_update": all(r["c3_ok"] for r in arm1_results.values()),
        "c4_positive_mean": all(r["c4_ok"] for r in arm1_results.values()),
        "per_seed_arm0": {
            str(s): r for s, r in arm0_results.items()
        },
        "per_seed_arm1": {
            str(s): r for s, r in arm1_results.items()
        },
    }

    return manifest, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest, run_id = run_experiment()

    if args.dry_run:
        print("DRY RUN -- manifest preview:", flush=True)
        print(json.dumps({k: v for k, v in manifest.items() if k != "per_seed_arm1"}, indent=2))
        sys.exit(0)

    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "REE_assembly", "evidence", "experiments"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"Manifest written: {out_path}", flush=True)
    print(f"Outcome: {manifest['outcome']}", flush=True)
    print(f"  ARM_0 sentinel: {manifest['all_arm0_pass']}", flush=True)
    print(f"  C0 (ep1 sentinel): {manifest['c0_sentinel_ok']}", flush=True)
    print(f"  C1 (ep2 valid):    {manifest['c1_ep2_valid']}", flush=True)
    print(f"  C2 (ep3 valid):    {manifest['c2_ep3_valid']}", flush=True)
    print(f"  C3 (no mid upd):   {manifest['c3_no_mid_update']}", flush=True)
    print(f"  C4 (pos mean):     {manifest['c4_positive_mean']}", flush=True)
    print(f"  mean_metric_arm1:  {manifest['mean_metric_arm1']:.4f}", flush=True)

    _outcome = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
