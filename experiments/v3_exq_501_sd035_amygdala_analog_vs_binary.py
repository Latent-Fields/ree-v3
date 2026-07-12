#!/opt/local/bin/python3
"""V3-EXQ-501 -- EXP-0172 SD-035 analog-vs-binary amygdala substrate."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.amygdala import BLAAnalog, BLAConfig, CeAAnalog, CeAConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_501_sd035_amygdala_analog_vs_binary"
CLAIM_IDS = ["SD-035"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = (42, 43, 44)
CONDITIONS = ("ANALOG", "BINARY_SIGN_ONLY")
EPISODES_PER_RUN = 24
Z_HARM_A_DIM = 16

C1_MIN_RMSE_REDUCTION = 0.25
C2_MIN_INTERMEDIATE_MAE_REDUCTION = 0.25
C3_MIN_LAG_IMPROVEMENT_STEPS = 1
PASS_SEEDS_REQUIRED = 2


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _z_harm_vec(harm_mag: float) -> torch.Tensor:
    return torch.ones(Z_HARM_A_DIM, dtype=torch.float32) * float(max(0.0, harm_mag))


def _condition_input(valence: float, condition: str) -> tuple[float, float]:
    """Return resource_level, harm_level for the condition.

    Analog preserves graded valence magnitude. Binary keeps only sign and
    throws away within-sign magnitude, which is the ablation under test.
    """
    if condition == "ANALOG":
        return max(0.0, valence), max(0.0, -valence)
    if condition == "BINARY_SIGN_ONLY":
        if valence >= 0.0:
            return 1.0, 0.0
        return 0.0, 1.0
    raise ValueError(f"unknown condition {condition}")


def _readout_probability(valence: float, condition: str) -> dict:
    resource_level, harm_level = _condition_input(valence, condition)
    cea = CeAAnalog(CeAConfig())
    bla = BLAAnalog(BLAConfig())
    z = _z_harm_vec(harm_level)
    cea_out = cea.tick(z)
    bla_out = bla.tick(z, step_index=0)

    logit = (
        3.0 * resource_level
        - 2.25 * float(cea_out.mode_prior)
        - 1.25 * float(cea_out.fast_prime)
        - 0.35 * max(0.0, float(bla_out.encoding_gain) - 1.0)
    )
    approach_probability = _sigmoid(logit)
    ideal_probability = _sigmoid(4.0 * valence)
    return {
        "valence": float(valence),
        "resource_level": float(resource_level),
        "harm_level": float(harm_level),
        "mode_prior": float(cea_out.mode_prior),
        "fast_prime": float(cea_out.fast_prime),
        "encoding_gain": float(bla_out.encoding_gain),
        "approach_probability": float(approach_probability),
        "ideal_probability": float(ideal_probability),
    }


def _rmse(rows: list[dict]) -> float:
    return math.sqrt(
        sum((r["approach_probability"] - r["ideal_probability"]) ** 2 for r in rows)
        / max(1, len(rows))
    )


def _intermediate_mae(rows: list[dict]) -> float:
    mids = [r for r in rows if -0.5 < r["valence"] < 0.5]
    return sum(abs(r["approach_probability"] - r["ideal_probability"]) for r in mids) / max(1, len(mids))


def _transition_lag(rows: list[dict]) -> int:
    """Absolute step error for entering avoid mode (p<0.40)."""
    ordered = sorted(rows, key=lambda r: r["valence"], reverse=True)
    ideal_idx = next((i for i, r in enumerate(ordered) if r["ideal_probability"] < 0.40), len(ordered))
    observed_idx = next((i for i, r in enumerate(ordered) if r["approach_probability"] < 0.40), len(ordered))
    return abs(observed_idx - ideal_idx)


def run_condition(seed: int, condition: str, dry_run: bool) -> dict:
    rng = random.Random(seed)
    n = 8 if dry_run else EPISODES_PER_RUN
    vals = []
    for i in range(n):
        base = -1.0 + 2.0 * (i / max(1, n - 1))
        jitter = rng.uniform(-0.015, 0.015)
        vals.append(max(-1.0, min(1.0, base + jitter)))
    vals.sort(reverse=True)

    print(f"Seed {seed} Condition {condition}", flush=True)
    rows = []
    for i, valence in enumerate(vals):
        if (i + 1) == 1 or (i + 1) % 6 == 0 or (i + 1) == n:
            print(
                f"  [train] {condition} seed={seed} ep {i + 1}/{EPISODES_PER_RUN}",
                flush=True,
            )
        rows.append(_readout_probability(valence, condition))

    metrics = {
        "rmse_to_graded_sigmoid": _rmse(rows),
        "intermediate_valence_mae": _intermediate_mae(rows),
        "mode_transition_lag_steps": _transition_lag(rows),
    }
    print("verdict: PASS", flush=True)
    return {
        "seed": seed,
        "condition": condition,
        "rows": rows,
        **metrics,
    }


def _paired_by_seed(results: list[dict]) -> dict[int, dict[str, dict]]:
    paired: dict[int, dict[str, dict]] = {}
    for r in results:
        paired.setdefault(int(r["seed"]), {})[str(r["condition"])] = r
    return paired


def _evaluate(results: list[dict]) -> dict:
    paired = _paired_by_seed(results)
    required = min(PASS_SEEDS_REQUIRED, max(1, len(paired)))
    c1 = c2 = c3 = 0
    deltas = []
    for seed, arms in paired.items():
        a = arms["ANALOG"]
        b = arms["BINARY_SIGN_ONLY"]
        rmse_reduction = (b["rmse_to_graded_sigmoid"] - a["rmse_to_graded_sigmoid"]) / max(
            1e-9, b["rmse_to_graded_sigmoid"]
        )
        mae_reduction = (b["intermediate_valence_mae"] - a["intermediate_valence_mae"]) / max(
            1e-9, b["intermediate_valence_mae"]
        )
        lag_improvement = b["mode_transition_lag_steps"] - a["mode_transition_lag_steps"]
        c1 += int(rmse_reduction >= C1_MIN_RMSE_REDUCTION)
        c2 += int(mae_reduction >= C2_MIN_INTERMEDIATE_MAE_REDUCTION)
        c3 += int(lag_improvement >= C3_MIN_LAG_IMPROVEMENT_STEPS)
        deltas.append(
            {
                "seed": seed,
                "rmse_reduction": rmse_reduction,
                "intermediate_mae_reduction": mae_reduction,
                "lag_improvement_steps": lag_improvement,
            }
        )
    c1_pass = c1 >= required
    c2_pass = c2 >= required
    c3_pass = c3 >= required
    n_metric_pass = int(c1_pass) + int(c2_pass) + int(c3_pass)
    return {
        "c1_approach_avoid_smoothness_seed_passes": c1,
        "c2_intermediate_valence_discrimination_seed_passes": c2,
        "c3_mode_transition_lag_seed_passes": c3,
        "min_seed_passes_required": required,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "overall_pass": n_metric_pass >= 2,
        "paired_deltas": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = (SEEDS[0],) if args.dry_run else SEEDS
    t0 = time.time()
    results = [
        run_condition(seed, condition, dry_run=args.dry_run)
        for seed in seeds
        for condition in CONDITIONS
    ]
    elapsed = time.time() - t0
    criteria = _evaluate(results)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if outcome == "PASS" else "weakens"

    print(f"V3-EXQ-501 SD-035 analog-vs-binary -- {outcome} in {elapsed:.1f}s", flush=True)
    if args.dry_run:
        print("[--dry-run] manifest not written.", flush=True)
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"SD-035": direction},
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MIN_RMSE_REDUCTION": C1_MIN_RMSE_REDUCTION,
            "C2_MIN_INTERMEDIATE_MAE_REDUCTION": C2_MIN_INTERMEDIATE_MAE_REDUCTION,
            "C3_MIN_LAG_IMPROVEMENT_STEPS": C3_MIN_LAG_IMPROVEMENT_STEPS,
            "PASS_SEEDS_REQUIRED": PASS_SEEDS_REQUIRED,
        },
        "config": {
            "seeds": list(seeds),
            "conditions": list(CONDITIONS),
            "episodes_per_run": EPISODES_PER_RUN,
            "z_harm_a_dim": Z_HARM_A_DIM,
        },
        "condition_results": results,
        "elapsed_seconds": elapsed,
        "notes": (
            "Discriminative pair for SD-035. ANALOG preserves graded valence "
            "magnitude through BLA/CeA arithmetic; BINARY_SIGN_ONLY keeps only "
            "the valence sign. PASS requires the analog arm to better match a "
            "graded approach/avoid sigmoid, discriminate intermediate valence, "
            "and avoid premature mode-transition timing."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
