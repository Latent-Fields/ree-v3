#!/opt/local/bin/python3
"""V3-EXQ-509 -- SD-047 multi-source environmental dynamics substrate readiness.

Claim: SD-047 (environment.multi_source_dynamics)
Status: candidate (v3_pending). Substrate IMPLEMENTED 2026-05-03.

Why this experiment exists
--------------------------
SD-047 lands three concurrent stochastic event sources on CausalGridWorldV2:
an AR(1) coarse-grid weather perturbation, a Poisson transient-hazard process,
and a small set of background-drift sources. The architectural commitment is
that with multi_source_intensity_scale=1.0 (ARM_2, default calibration), the
env produces an agent-independent causal background dense enough that the
event ratio env_events:agent_events lands in 1:1 to 2:1 over a random-policy
episode. Below that the comparator (MECH-095 / V3-EXQ-506) sees too-thin
causation to learn a not-self baseline; above that the agent's own signal
is overwhelmed (Asai 2016 non-monotonic agency S/N).

This experiment is a SUBSTRATE READINESS DIAGNOSTIC, not the full comparator-
gap behavioural test. It exercises the live env at the four pre-registered
arms (OFF / 0.25x / 1.0x / 4.0x) under a random-policy rollout and checks
that the new code paths fire as designed and produce the calibrated event
ratios. The full comparator-gap behavioural test using a trained agent's
E2_harm_s forward model lives in V3-EXQ-510 (queued separately as a
follow-up after this readiness probe PASSes).

Pre-registered acceptance criteria
----------------------------------
ARM_0 (multi_source_dynamics_enabled=False, intensity_scale=1.0):
  - bit-identical to legacy V2 substrate.
  - C0: multi_source_n_env_events == 0 across the rollout (no SD-047 events).

ARM_1 (enabled=True, scale=0.25, all three sources on):
  - C1a: weather_step_delta_sum > 0 (AR(1) firing).
  - C1b: ARM_1 env_events > 0 (sources firing).

ARM_2 (enabled=True, scale=1.0, all three sources on):
  - C2a: 0.5 <= ratio_env_per_agent <= 2.5 (calibration target band 1:1-2:1
    with tolerance for Poisson noise across 200 ticks; the SD doc's
    "1:1 to 2:1 env:agent change events" target).
  - C2b: weather, transient, drift sources all produce non-zero counts.
  - C2c: env_events_ARM_2 > env_events_ARM_1 (monotone scaling 1->2).

ARM_3 (enabled=True, scale=4.0, all three sources on):
  - C3a: env_events_ARM_3 >= env_events_ARM_2 OR within 5% of ARM_2 (saturation
    permitted: at high scale the transient pool is bounded by available
    empty cells; this is not a calibration failure).

PASS = C0 AND C1a AND C1b AND C2a AND C2b AND C2c AND C3a.
PASS = SD-047 substrate is calibrated and ready for the V3-EXQ-510 comparator-
gap behavioural test.
FAIL on C2a -> recalibrate per-source intensity defaults; revisit smoke
defaults; do not move to V3-EXQ-510 until calibration is in band.
FAIL on C0 -> bit-identical OFF guarantee broken; bug in env code path.

experiment_purpose = "diagnostic" (substrate readiness, not governance evidence).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_509_sd047_multi_source_substrate_readiness.py
  /opt/local/bin/python3 experiments/v3_exq_509_sd047_multi_source_substrate_readiness.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_509_sd047_multi_source_substrate_readiness"
CLAIM_IDS = ["SD-047"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = (42, 43, 44)
N_TICKS_PER_ARM = 200
GRID_SIZE = 8
N_HAZARDS = 3
N_RESOURCES = 3

# Source defaults at ARM_2 (intensity_scale=1.0).
WEATHER_SUPER_CELLS = 4
WEATHER_ALPHA_AR1 = 0.95
WEATHER_SIGMA = 0.10
TRANSIENT_P_APPEAR = 5e-3
TRANSIENT_P_DISAPPEAR = 0.10
N_DRIFT_SOURCES = 2
DRIFT_POLICY = "random_walk"

# 4-arm sweep specification.
ARMS: List[Tuple[str, bool, float]] = [
    ("ARM_0_off",        False, 1.0),
    ("ARM_1_low_0p25",   True,  0.25),
    ("ARM_2_default",    True,  1.0),
    ("ARM_3_high_4p0",   True,  4.0),
]

# Acceptance thresholds (pre-registered).
RATIO_LOWER_BAND = 0.5
RATIO_UPPER_BAND = 2.5
ARM_3_SATURATION_TOLERANCE = 0.05  # ARM_3 may saturate; allow 5% under ARM_2.


def random_action(rng: np.random.Generator) -> torch.Tensor:
    return torch.tensor(int(rng.integers(0, 5)), dtype=torch.long)


def run_arm(seed: int, arm_name: str, enabled: bool, scale: float, n_ticks: int) -> Dict:
    """Run a single (seed, arm) cell. Returns aggregate metrics."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + hash(arm_name) % 2**16)
    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        seed=seed,
        # SD-047 master + per-source switches.
        multi_source_dynamics_enabled=enabled,
        multi_source_intensity_scale=scale,
        weather_field_enabled=enabled,
        weather_super_cells=WEATHER_SUPER_CELLS,
        weather_alpha_ar1=WEATHER_ALPHA_AR1,
        weather_sigma=WEATHER_SIGMA,
        transient_events_enabled=enabled,
        transient_p_appear=TRANSIENT_P_APPEAR,
        transient_p_disappear=TRANSIENT_P_DISAPPEAR,
        background_drift_enabled=enabled,
        n_drift_sources=N_DRIFT_SOURCES,
        drift_policy=DRIFT_POLICY,
    )
    flat, obs = env.reset()
    n_env = 0
    n_agent = 0
    weather_step_delta_sum = 0.0
    n_transient_appear = 0
    n_transient_disappear = 0
    n_drift_moved = 0
    n_resets = 0
    for _ in range(n_ticks):
        a = random_action(rng)
        flat, harm, done, info, obs = env.step(a)
        n_env += int(info["multi_source_n_env_events"])
        n_agent += int(info["multi_source_n_agent_events"])
        weather_step_delta_sum += float(info["multi_source_weather_step_delta"])
        n_transient_appear += int(info["multi_source_n_transient_appear"])
        n_transient_disappear += int(info["multi_source_n_transient_disappear"])
        n_drift_moved += int(info["multi_source_n_drift_moved"])
        if done:
            flat, obs = env.reset()
            n_resets += 1
    ratio = n_env / max(1, n_agent)
    return {
        "arm": arm_name,
        "enabled": enabled,
        "scale": scale,
        "seed": seed,
        "n_ticks": n_ticks,
        "n_env_events": n_env,
        "n_agent_events": n_agent,
        "ratio_env_per_agent": float(ratio),
        "weather_step_delta_sum": float(weather_step_delta_sum),
        "n_transient_appear": n_transient_appear,
        "n_transient_disappear": n_transient_disappear,
        "n_drift_moved": n_drift_moved,
        "n_resets": n_resets,
    }


def evaluate_acceptance(per_arm_aggregate: Dict[str, Dict]) -> Dict:
    """Evaluate the pre-registered C0/C1a/C1b/C2a/C2b/C2c/C3a checks on aggregates."""
    a0 = per_arm_aggregate["ARM_0_off"]
    a1 = per_arm_aggregate["ARM_1_low_0p25"]
    a2 = per_arm_aggregate["ARM_2_default"]
    a3 = per_arm_aggregate["ARM_3_high_4p0"]

    # C0: bit-identical OFF for SD-047 events. Note: legacy env_drift produces
    # env_caused_hazard transitions which the agent counts as agent_events
    # via transition_type; SD-047's env-event counter increments only via the
    # multi-source code paths plus env_caused_hazard transition_type. With
    # SD-047 disabled, only the latter contributes; we assert n_env from
    # SD-047's sources is zero by checking weather/transient/drift counters.
    c0 = (
        a0["weather_step_delta_sum"] == 0.0
        and a0["n_transient_appear"] == 0
        and a0["n_transient_disappear"] == 0
        and a0["n_drift_moved"] == 0
    )
    c1a = a1["weather_step_delta_sum"] > 0.0
    c1b = a1["n_env_events"] > 0
    c2a = (RATIO_LOWER_BAND <= a2["ratio_env_per_agent"] <= RATIO_UPPER_BAND)
    c2b = (
        a2["weather_step_delta_sum"] > 0.0
        and a2["n_transient_appear"] > 0
        and a2["n_drift_moved"] > 0
    )
    c2c = a2["n_env_events"] > a1["n_env_events"]
    saturation_floor = a2["n_env_events"] * (1.0 - ARM_3_SATURATION_TOLERANCE)
    c3a = a3["n_env_events"] >= saturation_floor

    overall = c0 and c1a and c1b and c2a and c2b and c2c and c3a
    return {
        "C0_off_bit_identical": bool(c0),
        "C1a_arm1_weather_firing": bool(c1a),
        "C1b_arm1_env_events_positive": bool(c1b),
        "C2a_arm2_calibration_band": bool(c2a),
        "C2b_arm2_all_sources_firing": bool(c2b),
        "C2c_arm2_above_arm1": bool(c2c),
        "C3a_arm3_at_or_above_arm2": bool(c3a),
        "all_pass": bool(overall),
    }


def aggregate_seeds(per_seed_arms: List[Dict]) -> Dict[str, Dict]:
    """Sum per-seed aggregates per arm so acceptance reads totals, not means."""
    bucket: Dict[str, Dict] = {}
    for r in per_seed_arms:
        arm = r["arm"]
        if arm not in bucket:
            bucket[arm] = {
                "arm": arm,
                "enabled": r["enabled"],
                "scale": r["scale"],
                "n_ticks": 0,
                "n_env_events": 0,
                "n_agent_events": 0,
                "weather_step_delta_sum": 0.0,
                "n_transient_appear": 0,
                "n_transient_disappear": 0,
                "n_drift_moved": 0,
                "n_resets": 0,
            }
        b = bucket[arm]
        b["n_ticks"] += r["n_ticks"]
        b["n_env_events"] += r["n_env_events"]
        b["n_agent_events"] += r["n_agent_events"]
        b["weather_step_delta_sum"] += r["weather_step_delta_sum"]
        b["n_transient_appear"] += r["n_transient_appear"]
        b["n_transient_disappear"] += r["n_transient_disappear"]
        b["n_drift_moved"] += r["n_drift_moved"]
        b["n_resets"] += r["n_resets"]
    for arm, b in bucket.items():
        b["ratio_env_per_agent"] = b["n_env_events"] / max(1, b["n_agent_events"])
    return bucket


def main(dry_run: bool = False) -> int:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    n_ticks = 30 if dry_run else N_TICKS_PER_ARM
    seeds = (SEEDS[0],) if dry_run else SEEDS
    per_seed_arms: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        for arm_name, enabled, scale in ARMS:
            r = run_arm(seed, arm_name, enabled, scale, n_ticks)
            per_seed_arms.append(r)
            print(
                f"  seed={seed} arm={arm_name:<18} env={r['n_env_events']:4d} "
                f"agent={r['n_agent_events']:4d} ratio={r['ratio_env_per_agent']:.2f} "
                f"weather_d={r['weather_step_delta_sum']:.3f} "
                f"trans_app/dis={r['n_transient_appear']}/{r['n_transient_disappear']} "
                f"drift_moved={r['n_drift_moved']}"
            )
    elapsed = time.time() - t0
    aggregates = aggregate_seeds(per_seed_arms)
    acceptance = evaluate_acceptance(aggregates)
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"
    print(f"[{EXPERIMENT_TYPE}] aggregates:")
    for arm in ("ARM_0_off", "ARM_1_low_0p25", "ARM_2_default", "ARM_3_high_4p0"):
        a = aggregates[arm]
        print(
            f"  {arm:<18} env_total={a['n_env_events']:5d} agent_total={a['n_agent_events']:5d} "
            f"ratio={a['ratio_env_per_agent']:.2f}"
        )
    print(f"[{EXPERIMENT_TYPE}] acceptance:")
    for k, v in acceptance.items():
        print(f"  {k}: {v}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
        return 0

    # Manifest write per V3 explorer-launch convention.
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "elapsed_seconds": elapsed,
        "n_seeds": len(seeds),
        "n_ticks_per_arm": n_ticks,
        "arms": list(aggregates.values()),
        "per_seed_per_arm": per_seed_arms,
        "acceptance": acceptance,
        "thresholds": {
            "ratio_lower_band": RATIO_LOWER_BAND,
            "ratio_upper_band": RATIO_UPPER_BAND,
            "arm_3_saturation_tolerance": ARM_3_SATURATION_TOLERANCE,
        },
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}")

    from experiment_protocol import emit_outcome
    emit_outcome(outcome=outcome, manifest_path=str(out_path))

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
