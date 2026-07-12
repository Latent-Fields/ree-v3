"""
V3-EXQ-587: EXQ-ISEF-001 -- Harm Gradient vs Binary-Contact Residue Geography
Formation Speed (infant_substrate:GAP-10 closure experiment).

Scientific question: Does harm_gradient_enabled=True (approach signal, fires
a graduated negative reward when the agent is within outer_radius of a hazard
without contact) produce a more strongly-weighted residue field than
binary-contact-only (harm_gradient_enabled=False) under an identical episode
budget?

Design:
  ARM_0 (binary contact): harm_gradient_enabled=False (default). Residue
        accumulates only on direct hazard contact events.
  ARM_1 (gradient): harm_gradient_enabled=True, harm_gradient_scale=0.30.
        Residue accumulates from both contact events AND approach-signal events
        (graduated reward proportional to proximity).

Context from prior runs:
  V3-EXQ-575: tested proximity_harm_scale variation (wrong knob for GAP-10).
  V3-EXQ-575a: same wrong knob but instrument-corrected. Informational C3
        showed mean_weight separation ~2x (ARM_0=1.48 vs ARM_1=2.84) at
        proximity_harm_scale 0.05 vs 0.30. Coverage saturated at 1.0 for both
        arms -- mean_weight is the correct discriminating metric.
  This experiment (V3-EXQ-587) isolates harm_gradient_enabled True vs False,
  the actual GAP-10 manipulation.

Acceptance criteria:
  C1 (gate): At episode 1000 (final), ARM_1 mean_weight > 2x ARM_0 mean_weight
      in at least 4 of 5 seeds. The 2x threshold is pre-registered from the
      575a C3 informational result. PASS = C1.
  C2 (advisory): ARM_1 residue_coverage_pct at episode 100 > ARM_0. Tests
      whether approach signal accelerates early geography formation.
  C3 (advisory): ARM_1 final harm_total > ARM_0 final harm_total. Sanity:
      approach signal generates more harm events than binary contact alone.

Outcome is gated on C1 only. C2 and C3 are logged but do not affect the
PASS/FAIL decision.

Interpretation grid:
  Outcome                                       | Diagnosis / next action
  ----------------------------------------------|-------------------------------------
  C1 PASS (>=4/5 seeds, mean_weight ratio >=2x) | harm_gradient_enabled=True drives
                                                |   significantly denser residue;
                                                |   GAP-10 CLOSED; ARC-013 and
                                                |   DEV-NEED-004 unblocked
  C1 FAIL (ratio < 2x in majority of seeds)     | Gradient adds marginal residue vs
                                                |   binary contact at this scale;
                                                |   raise harm_gradient_scale or
                                                |   extend episodes; check that
                                                |   harm_gradient fires are reaching
                                                |   update_residue()
  C1 FAIL + C3 FAIL (harm_total ARM_1 < ARM_0)  | Approach signal not generating
                                                |   harm events; check
                                                |   harm_gradient_enabled wiring in
                                                |   CausalGridWorldV2.step()
  C1 FAIL + C3 PASS but ratio < 2x              | Gradient fires but residue
                                                |   accumulation per event is low;
                                                |   check update_residue() harm_scale
                                                |   vs binary contact harm magnitude

claim_ids: [] (diagnostic -- tests substrate feature, not a claim hypothesis)
experiment_purpose: diagnostic
Unblocks: infant_substrate:GAP-10, ARC-013, DEV-NEED-004.
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

EXPERIMENT_TYPE = "v3_exq_587_isef001_harm_gradient_enabled"
QUEUE_ID = "V3-EXQ-587"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44, 45, 46]
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4
N_EPISODES = 1000
STEPS_PER_EPISODE = 200

# Episode indices (0-indexed) at which to capture telemetry snapshots.
# ep 99 = episode 100, ep 499 = episode 500, ep 999 = episode 1000.
SNAPSHOT_EPS = [99, 499, 999]

ARM_NAMES = ["ARM_0_binary_contact", "ARM_1_harm_gradient"]

# CausalGridWorldV2 constructor kwargs per arm (merged with shared defaults).
ARM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ARM_0_binary_contact": {
        "harm_gradient_enabled": False,
    },
    "ARM_1_harm_gradient": {
        "harm_gradient_enabled": True,
        "harm_gradient_scale": 0.30,
    },
}

# Pre-registered acceptance thresholds.
# Ratio threshold derived from 575a C3 informational result (1.48 vs 2.84).
C1_MEAN_WEIGHT_RATIO_MIN = 2.0
C1_MIN_SEEDS_PASSING = 4   # >= 4 of 5 seeds must exceed the ratio


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_agent() -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
    )
    cfg.latent.alpha_world = 0.9
    return REEAgent(cfg)


def _extract_split_obs(obs_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
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


# ------------------------------------------------------------------
# Arm runner
# ------------------------------------------------------------------

def _run_arm(
    *,
    seed: int,
    arm_name: str,
    arm_config: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    agent = _build_agent()

    env = CausalGridWorldV2(
        resource_respawn_on_consume=True,
        **arm_config,
    )

    n_episodes = 2 if dry_run else N_EPISODES
    snapshots: Dict[int, Dict[str, float]] = {}

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

        # Collect telemetry at snapshot episodes and at progress-print boundaries.
        want_snapshot = ep in SNAPSHOT_EPS or ep == (n_episodes - 1)
        want_print = (ep + 1) % 100 == 0 or ep == (n_episodes - 1)

        if want_snapshot or want_print:
            telemetry = agent.residue_field.get_coverage_telemetry()
            stats = agent.residue_field.get_statistics()
            snap = {
                "residue_coverage_pct": float(telemetry["residue_coverage_pct"]),
                "harm_total": float(telemetry["harm_total"]),
                "mean_weight": float(stats["mean_weight"].item()),
            }
            if want_snapshot:
                snapshots[ep] = snap
            if want_print:
                print(
                    f"  [train] {arm_name} ep {ep + 1}/{n_episodes} "
                    f"seed={seed} coverage={snap['residue_coverage_pct']:.3f} "
                    f"mean_weight={snap['mean_weight']:.4f}",
                    flush=True,
                )

    # Extract key metrics from snapshots.
    final_ep = n_episodes - 1
    final_snap = snapshots.get(
        final_ep,
        {"residue_coverage_pct": 0.0, "harm_total": 0.0, "mean_weight": 0.0},
    )
    # Early snapshot: ep 99 (episode 100). Falls back to final in dry run.
    early_ep = 99 if not dry_run else final_ep
    early_snap = snapshots.get(
        early_ep,
        snapshots.get(final_ep, {"residue_coverage_pct": 0.0}),
    )

    return {
        "arm": arm_name,
        "final_mean_weight": final_snap["mean_weight"],
        "final_coverage": final_snap["residue_coverage_pct"],
        "final_harm_total": final_snap["harm_total"],
        "ep100_coverage": early_snap["residue_coverage_pct"],
        "snapshots": {str(k): v for k, v in snapshots.items()},
    }


# ------------------------------------------------------------------
# Seed runner
# ------------------------------------------------------------------

def _run_seed(*, seed: int, dry_run: bool) -> Dict[str, Any]:
    arm_results: Dict[str, Dict[str, Any]] = {}
    for arm_name in ARM_NAMES:
        arm_config = ARM_CONFIGS[arm_name]
        print(f"Seed {seed} Condition {arm_name}", flush=True)
        result = _run_arm(
            seed=seed,
            arm_name=arm_name,
            arm_config=arm_config,
            dry_run=dry_run,
        )
        arm_results[arm_name] = result
        # Per-arm sanity verdict: residue path is live if any weight accumulated.
        arm_pass = result["final_mean_weight"] > 0.0
        print(
            f"verdict: {'PASS' if arm_pass else 'FAIL'}",
            flush=True,
        )

    return {
        "seed": seed,
        "arm_results": arm_results,
    }


# ------------------------------------------------------------------
# Experiment
# ------------------------------------------------------------------

def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS

    print(f"V3-EXQ-587: EXQ-ISEF-001 harm_gradient_enabled True vs False", flush=True)
    print(f"  dry_run={dry_run} seeds={seeds} n_episodes={N_EPISODES}", flush=True)
    print(f"  ARM_0: harm_gradient_enabled=False (binary contact only)", flush=True)
    print(f"  ARM_1: harm_gradient_enabled=True harm_gradient_scale=0.30", flush=True)
    print(
        f"  C1 gate: ARM_1 mean_weight > {C1_MEAN_WEIGHT_RATIO_MIN}x ARM_0 "
        f"in >= {C1_MIN_SEEDS_PASSING}/5 seeds",
        flush=True,
    )

    all_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        result = _run_seed(seed=seed, dry_run=dry_run)
        all_seed_results.append(result)

    # ------------------------------------------------------------------
    # C1: per-seed mean_weight ratio at final episode
    # ------------------------------------------------------------------
    seed_c1_results: List[Dict[str, Any]] = []
    for r in all_seed_results:
        arm0_w = r["arm_results"]["ARM_0_binary_contact"]["final_mean_weight"]
        arm1_w = r["arm_results"]["ARM_1_harm_gradient"]["final_mean_weight"]
        # Cap at 999.0 to keep JSON valid (float("inf") is not JSON-serialisable).
        ratio = arm1_w / arm0_w if arm0_w > 1e-9 else 999.0
        seed_pass = ratio >= C1_MEAN_WEIGHT_RATIO_MIN
        seed_c1_results.append(
            {
                "seed": r["seed"],
                "arm0_final_mean_weight": arm0_w,
                "arm1_final_mean_weight": arm1_w,
                "ratio": ratio,
                "c1_pass": seed_pass,
            }
        )

    seeds_passing_c1 = sum(1 for s in seed_c1_results if s["c1_pass"])
    c1_pass = seeds_passing_c1 >= C1_MIN_SEEDS_PASSING or dry_run

    # ------------------------------------------------------------------
    # C2 (advisory): ARM_1 early coverage (ep100) > ARM_0
    # ------------------------------------------------------------------
    arm0_ep100_covs = [
        r["arm_results"]["ARM_0_binary_contact"]["ep100_coverage"]
        for r in all_seed_results
    ]
    arm1_ep100_covs = [
        r["arm_results"]["ARM_1_harm_gradient"]["ep100_coverage"]
        for r in all_seed_results
    ]

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    mean_arm0_ep100 = _mean(arm0_ep100_covs)
    mean_arm1_ep100 = _mean(arm1_ep100_covs)
    c2_arm1_early_exceeds_arm0 = mean_arm1_ep100 > mean_arm0_ep100

    # ------------------------------------------------------------------
    # C3 (advisory): ARM_1 final harm_total > ARM_0
    # ------------------------------------------------------------------
    arm0_harm_totals = [
        r["arm_results"]["ARM_0_binary_contact"]["final_harm_total"]
        for r in all_seed_results
    ]
    arm1_harm_totals = [
        r["arm_results"]["ARM_1_harm_gradient"]["final_harm_total"]
        for r in all_seed_results
    ]
    mean_arm0_harm = _mean(arm0_harm_totals)
    mean_arm1_harm = _mean(arm1_harm_totals)
    c3_arm1_harm_exceeds_arm0 = mean_arm1_harm > mean_arm0_harm

    outcome = "PASS" if c1_pass else "FAIL"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"", flush=True)
    print(f"=== V3-EXQ-587 Results ===", flush=True)
    print(
        f"C1 gate: ARM_1 mean_weight > {C1_MEAN_WEIGHT_RATIO_MIN}x ARM_0 "
        f"in >= {C1_MIN_SEEDS_PASSING}/5 seeds",
        flush=True,
    )
    print(f"  Seeds passing C1: {seeds_passing_c1}/{len(seeds)}", flush=True)
    for sc in seed_c1_results:
        print(
            f"  seed={sc['seed']} arm0_w={sc['arm0_final_mean_weight']:.4f} "
            f"arm1_w={sc['arm1_final_mean_weight']:.4f} ratio={sc['ratio']:.2f} "
            f"c1={'PASS' if sc['c1_pass'] else 'FAIL'}",
            flush=True,
        )
    print(f"C1 overall: {'PASS' if c1_pass else 'FAIL'}", flush=True)
    print(
        f"C2 [advisory] early coverage ep100: "
        f"arm0={mean_arm0_ep100:.4f} arm1={mean_arm1_ep100:.4f} "
        f"arm1_exceeds_arm0={'YES' if c2_arm1_early_exceeds_arm0 else 'NO'}",
        flush=True,
    )
    print(
        f"C3 [advisory] final harm_total: "
        f"arm0={mean_arm0_harm:.3f} arm1={mean_arm1_harm:.3f} "
        f"arm1_exceeds_arm0={'YES' if c3_arm1_harm_exceeds_arm0 else 'NO'}",
        flush=True,
    )
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "c1_pass": c1_pass,
        "seeds_passing_c1": seeds_passing_c1,
        "c2_arm1_early_exceeds_arm0": c2_arm1_early_exceeds_arm0,
        "c3_arm1_harm_exceeds_arm0": c3_arm1_harm_exceeds_arm0,
        "seed_c1_results": seed_c1_results,
        "mean_arm0_ep100_coverage": mean_arm0_ep100,
        "mean_arm1_ep100_coverage": mean_arm1_ep100,
        "mean_arm0_final_harm_total": mean_arm0_harm,
        "mean_arm1_final_harm_total": mean_arm1_harm,
        "all_seed_results": all_seed_results,
    }


# ------------------------------------------------------------------
# Main / manifest
# ------------------------------------------------------------------

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
            "n_episodes": N_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "snapshot_episodes": SNAPSHOT_EPS,
            "arm_configs": ARM_CONFIGS,
            "metric_primary": "mean_weight_ratio (ARM_1/ARM_0 at final episode)",
            "metric_secondary": ["residue_coverage_pct_ep100", "harm_total_final"],
        },
        "acceptance_criteria": {
            "C1_gate": (
                f"ARM_1 final mean_weight > {C1_MEAN_WEIGHT_RATIO_MIN}x ARM_0 "
                f"in >= {C1_MIN_SEEDS_PASSING}/5 seeds"
            ),
            "C2_advisory": "ARM_1 residue_coverage_pct at ep100 > ARM_0",
            "C3_advisory": "ARM_1 final harm_total > ARM_0 final harm_total",
        },
        "criteria_results": {
            "C1_pass": result["c1_pass"],
            "seeds_passing_c1": result["seeds_passing_c1"],
            "C2_arm1_early_exceeds_arm0_advisory": result["c2_arm1_early_exceeds_arm0"],
            "C3_arm1_harm_exceeds_arm0_advisory": result["c3_arm1_harm_exceeds_arm0"],
        },
        "metrics": {
            "mean_arm0_ep100_coverage": result["mean_arm0_ep100_coverage"],
            "mean_arm1_ep100_coverage": result["mean_arm1_ep100_coverage"],
            "mean_arm0_final_harm_total": result["mean_arm0_final_harm_total"],
            "mean_arm1_final_harm_total": result["mean_arm1_final_harm_total"],
        },
        "per_seed_c1_results": result["seed_c1_results"],
        "per_seed_results": result["all_seed_results"],
        "notes": (
            "EXQ-ISEF-001 proper (infant_substrate:GAP-10). "
            "V3-EXQ-575/575a tested proximity_harm_scale variation (different knob). "
            "This experiment isolates harm_gradient_enabled True vs False. "
            "Primary criterion: final mean_weight ratio >= 2x in >= 4/5 seeds. "
            "Threshold pre-registered from 575a C3 informational (arm0=1.48, arm1=2.84). "
            "Coverage saturates at 1.0 for both arms (575a finding) -- mean_weight "
            "is the correct discriminating metric. PASS closes GAP-10 and unblocks "
            "ARC-013, DEV-NEED-004."
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
        summary = {
            k: v for k, v in manifest.items() if k not in ("per_seed_results",)
        }
        print(json.dumps(summary, indent=2), flush=True)

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
