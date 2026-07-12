"""
V3-EXQ-575a: ISEF-001 Harm Gradient Baseline -- Warm-Start Gate Diagnostic.

Supersedes V3-EXQ-575. Same scientific question, instrument fix only.

failure_autopsy_EXQ-575_2026-05-17 root cause: V3-EXQ-575 measured
`get_statistics()["active_centers"] / 32`. active_mask is flipped True
PERMANENTLY by a 32-slot ring buffer on every harm event; proximity_harm_scale
modulates residue WEIGHT MAGNITUDE, not WHICH centers are active. Over
40k steps/episode all 32 bits flip in both arms -> coverage pinned at 1.0,
C2 (ARM_1 > ARM_0) structurally inexpressible. epistemic_category =
measurement_artifact. The corrected instrument is GAP-6
ResidueField.get_coverage_telemetry()["residue_coverage_pct"] =
(active_mask & |weight| > 0.02*rsf).sum() / n_centers, which IS sensitive to
harm intensity. This script swaps to that metric and adds the magnitude
secondary metrics (harm_total, mean_weight) so the intensity separation is
visible even if thresholded coverage co-saturates at high harm_scale.

DEV-NEED-029: confirm proximity_harm_scale=0.30 (ARM_1 treatment) produces
residue_coverage_pct >= 0.15 after 200 episodes, exceeding
proximity_harm_scale=0.05 (ARM_0 control, default cold-start). This diagnostic
gates ALL downstream diversity experiments (ARC-065 sweeps, Q-043/044/045
retests, INV-049 retest). EXQ-573 proved cold-start environments produce null
diversity results.

Interpretation grid (instrument-corrected):
  Outcome                                     | Diagnosis / next action
  --------------------------------------------|---------------------------------------
  C1 PASS + C2 PASS                           | Warm-start gate VALIDATED;
                                              |   downstream ARC-065 / Q-043/044/045 /
                                              |   INV-049 may proceed with
                                              |   proximity_harm_scale=0.30 pre-train
  C1 FAIL (ARM_1 thresholded cov < 0.15)      | 0.30 insufficient under thr=0.02:
                                              |   raise proximity_harm_scale OR more
                                              |   episodes; residue path is live
                                              |   (verify via harm_total > 0)
  C2 FAIL with ARM_1 ~ ARM_0 ~ 1.0            | Thresholded coverage co-saturated:
                                              |   metric still ceilinged at thr=0.02.
                                              |   This is NOT a substrate conclusion --
                                              |   re-issue with lower contrast
                                              |   (ARM_0 -> 0.0) or higher threshold
                                              |   (residue_scale_factor up). Use the
                                              |   harm_total / mean_weight secondary
                                              |   metrics to confirm intensity DID
                                              |   separate (sanity that the path works)
  C2 FAIL with ARM_1 < ARM_0                  | Inverted: residue field not tracking
                                              |   harm gradient; check
                                              |   proximity_harm_scale wiring into
                                              |   CausalGridWorldV2 harm_signal
  Secondary: arm1 mean_weight > arm0          | Confirms intensity scaling is wired
  mean_weight (expected True regardless)      |   even when coverage co-saturates;
                                              |   informational, not a gate
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

EXPERIMENT_TYPE = "v3_exq_575a_isef001_harm_gradient_baseline"
QUEUE_ID = "V3-EXQ-575a"
SUPERSEDES = "V3-EXQ-575"
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

# Pre-registered acceptance thresholds (gates depend on these ONLY)
C1_MIN_ARM1_COVERAGE = 0.15   # ARM_1 mean thresholded coverage >= 15% of RBF centers
C2_ARM1_EXCEEDS_ARM0 = True   # ARM_1 mean thresholded coverage > ARM_0 mean


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
    harm_total_per_ep: List[float] = []
    mean_weight_per_ep: List[float] = []

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

        # Instrument fix: magnitude-thresholded coverage (GAP-6), not the
        # binary ring-buffer active-center count get_statistics() exposes.
        telemetry = agent.residue_field.get_coverage_telemetry()
        coverage = float(telemetry["residue_coverage_pct"])
        harm_total = float(telemetry["harm_total"])

        stats = agent.residue_field.get_statistics()
        mean_weight = float(stats["mean_weight"].item())

        coverage_per_ep.append(coverage)
        harm_total_per_ep.append(harm_total)
        mean_weight_per_ep.append(mean_weight)

        if (ep + 1) % 50 == 0 or (ep + 1) == n_episodes:
            print(
                f"  [train] {arm_name} ep {ep + 1}/{n_episodes} "
                f"seed={seed} coverage={coverage:.3f} "
                f"harm_total={harm_total:.3f} mean_weight={mean_weight:.4f}",
                flush=True,
            )

    final_coverage = coverage_per_ep[-1] if coverage_per_ep else 0.0
    mean_coverage = (
        sum(coverage_per_ep) / len(coverage_per_ep) if coverage_per_ep else 0.0
    )
    final_harm_total = harm_total_per_ep[-1] if harm_total_per_ep else 0.0
    final_mean_weight = mean_weight_per_ep[-1] if mean_weight_per_ep else 0.0

    return {
        "arm": arm_name,
        "harm_scale": harm_scale,
        "final_coverage": final_coverage,
        "mean_coverage": mean_coverage,
        "final_harm_total": final_harm_total,
        "final_mean_weight": final_mean_weight,
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

        arm_cov = result["final_coverage"]
        seed_arm_pass = arm_cov >= 0.05
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
    print(f"V3-EXQ-575a: ISEF-001 Harm Gradient Baseline -- Warm-Start Gate", flush=True)
    print(f"  supersedes={SUPERSEDES} dry_run={dry_run} seeds={seeds}", flush=True)
    print(
        f"  ARM_0 harm_scale={ARM_HARM_SCALES['ARM_0_control']} "
        f"ARM_1 harm_scale={ARM_HARM_SCALES['ARM_1_treatment']} "
        f"metric=get_coverage_telemetry.residue_coverage_pct (thr=0.02)",
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
    arm0_harm_totals = [
        r["arm_results"]["ARM_0_control"]["final_harm_total"]
        for r in all_seed_results
    ]
    arm1_harm_totals = [
        r["arm_results"]["ARM_1_treatment"]["final_harm_total"]
        for r in all_seed_results
    ]
    arm0_mean_weights = [
        r["arm_results"]["ARM_0_control"]["final_mean_weight"]
        for r in all_seed_results
    ]
    arm1_mean_weights = [
        r["arm_results"]["ARM_1_treatment"]["final_mean_weight"]
        for r in all_seed_results
    ]

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    arm0_mean = _mean(arm0_coverages)
    arm1_mean = _mean(arm1_coverages)
    arm0_harm_mean = _mean(arm0_harm_totals)
    arm1_harm_mean = _mean(arm1_harm_totals)
    arm0_w_mean = _mean(arm0_mean_weights)
    arm1_w_mean = _mean(arm1_mean_weights)

    if dry_run:
        c1_pass = True
        c2_pass = True
    else:
        c1_pass = arm1_mean >= C1_MIN_ARM1_COVERAGE
        c2_pass = arm1_mean > arm0_mean

    # C3 is INFORMATIONAL ONLY -- not part of the outcome gate. It confirms
    # the harm-intensity scaling is wired even when thresholded coverage
    # co-saturates (failure_autopsy_EXQ-575 secondary-metric requirement).
    c3_meanweight_separates = arm1_w_mean > arm0_w_mean

    outcome = "PASS" if (c1_pass and c2_pass) else "FAIL"

    print(f"", flush=True)
    print(f"ARM_0 mean thresholded coverage: {arm0_mean:.4f}", flush=True)
    print(f"ARM_1 mean thresholded coverage: {arm1_mean:.4f}", flush=True)
    print(
        f"ARM_0 mean harm_total={arm0_harm_mean:.3f} mean_weight={arm0_w_mean:.4f}",
        flush=True,
    )
    print(
        f"ARM_1 mean harm_total={arm1_harm_mean:.3f} mean_weight={arm1_w_mean:.4f}",
        flush=True,
    )
    print(
        f"C1 (ARM_1 mean >= {C1_MIN_ARM1_COVERAGE}): {'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"C2 (ARM_1 > ARM_0): {'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"C3 [info] (ARM_1 mean_weight > ARM_0): "
        f"{'YES' if c3_meanweight_separates else 'NO'}",
        flush=True,
    )
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_meanweight_separates": c3_meanweight_separates,
        "arm0_mean_coverage": arm0_mean,
        "arm1_mean_coverage": arm1_mean,
        "arm0_per_seed_coverages": arm0_coverages,
        "arm1_per_seed_coverages": arm1_coverages,
        "arm0_mean_harm_total": arm0_harm_mean,
        "arm1_mean_harm_total": arm1_harm_mean,
        "arm0_mean_weight": arm0_w_mean,
        "arm1_mean_weight": arm1_w_mean,
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
        "supersedes": SUPERSEDES,
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
            "metric": "get_coverage_telemetry.residue_coverage_pct",
            "coverage_threshold": 0.02,
        },
        "acceptance_criteria": {
            "C1_arm1_min_mean_coverage": C1_MIN_ARM1_COVERAGE,
            "C2_arm1_exceeds_arm0": True,
            "C3_arm1_meanweight_exceeds_arm0_informational": True,
        },
        "criteria_results": {
            "C1_pass": result["c1_pass"],
            "C2_pass": result["c2_pass"],
            "C3_meanweight_separates_informational": result[
                "c3_meanweight_separates"
            ],
        },
        "metrics": {
            "arm0_mean_final_coverage": result["arm0_mean_coverage"],
            "arm1_mean_final_coverage": result["arm1_mean_coverage"],
            "arm0_per_seed_coverages": result["arm0_per_seed_coverages"],
            "arm1_per_seed_coverages": result["arm1_per_seed_coverages"],
            "arm0_mean_harm_total": result["arm0_mean_harm_total"],
            "arm1_mean_harm_total": result["arm1_mean_harm_total"],
            "arm0_mean_weight": result["arm0_mean_weight"],
            "arm1_mean_weight": result["arm1_mean_weight"],
        },
        "per_seed_results": result["all_seed_results"],
        "notes": (
            "DEV-NEED-029 warm-start gate diagnostic (ISEF-001). "
            "Supersedes V3-EXQ-575 (instrument fix per "
            "failure_autopsy_EXQ-575_2026-05-17: get_statistics().active_centers "
            "is a 32-slot ring-buffer saturation counter blind to harm "
            "intensity). Primary metric: "
            "get_coverage_telemetry()['residue_coverage_pct'] = "
            "(active_mask & |weight| > 0.02) / 32. Secondary metrics "
            "harm_total / mean_weight expose the intensity separation even if "
            "thresholded coverage co-saturates; C3 is informational only and "
            "NOT part of the outcome gate. ARM_0 proximity_harm_scale=0.05 "
            "(cold-start control), ARM_1 0.30 (warm-start treatment). PASS "
            "gates downstream ARC-065 sweeps and Q-043/044/045 / INV-049 "
            "retests."
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
