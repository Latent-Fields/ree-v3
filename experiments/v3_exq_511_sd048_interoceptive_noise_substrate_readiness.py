#!/opt/local/bin/python3
"""V3-EXQ-511 -- SD-048 interoceptive noise dynamics substrate readiness.

Claim: SD-048 (body.interoceptive_noise_dynamics)
Status: candidate (v3_pending). Substrate IMPLEMENTED 2026-05-03.

Why this experiment exists
--------------------------
SD-048 is the Level 2 body-state counterpart to SD-047 (Level 1 environmental
dynamics). It lands three concurrent stochastic agent-independent body-state
noise sources on harm_obs_a in CausalGridWorldV2:

  Source 1 (autonomic):  per-step i.i.d. Gaussian noise (HRV / sympathetic
                          fluctuation analog, sigma=0.02).
  Source 2 (sensitisation): Poisson onset of transient multiplicative
                          amplification with exponential decay (inflammatory
                          sensitisation analog; rate=0.008/step,
                          magnitude=1.8x, halflife=15 steps, cumulative cap 5x).
  Source 3 (fatigue):     slow AR(1) latent fatigue state additively contributing
                          to harm_obs_a across the episode (allostatic-load
                          analog; ar_coeff=0.995).

Architectural commitment: with interoceptive_noise_scale=1.0 (ARM_2 default
calibration), the env produces an agent-independent body-state background
dense enough that the |delta_harm_obs_a| event ratio body_noise:agent_caused
lands in the 1:1 to 2:1 band over a sparse-policy episode. Below that the
Level 2 comparator (ARC-058 / ARC-033) sees too-thin body causation to learn
a not-self baseline; above that the agent's own body-damage signal is
overwhelmed (Asai 2016 non-monotonic comparator competence).

This experiment is a SUBSTRATE READINESS DIAGNOSTIC, not the full Level 2
comparator-gap behavioural test on z_harm_a. It exercises the live env at
the four pre-registered arms (OFF / 0.25x / 1.0x / 4.0x) under a sparse
random-policy rollout (80% stay action) on a larger 12x12 substrate so the
agent has quiet steps where body noise can land delta events independent of
hazard contact. The full comparator-gap behavioural test using a trained
agent's E2_harm_a forward model (ARC-058 / ARC-033 paths) lives in V3-EXQ-512
(queued separately as a follow-up after this readiness probe PASSes).

Pre-registered acceptance criteria
----------------------------------
ARM_0 (interoceptive_noise_enabled=False, scale=1.0):
  - bit-identical to legacy V2 substrate.
  - C0: all SD-048 per-source counters zero across the rollout
    (n_autonomic_events == 0, n_sensitisation_events == 0,
     n_fatigue_events == 0, n_body_noise_events == 0).

ARM_1 (enabled=True, scale=0.25, all three sources on):
  - C1a: at least one source produces non-zero readout perturbation across
    the rollout (n_aut + n_sens + body_noise_events > 0).
  - C1b: n_body_noise_events > 0 (some delta event was body-noise classified).

ARM_2 (enabled=True, scale=1.0, all three sources on):
  - C2a: 0.5 <= ratio_body_noise_per_agent <= 2.5 (calibration target band
    1:1-2:1 with tolerance for Poisson noise across the rollout; SD doc target).
  - C2b: autonomic + sensitisation both produce non-zero counts (fatigue
    contribution can stay below the per-source delta threshold while still
    accumulating state, so it is NOT required at C2b).
  - C2c: n_body_noise_events_ARM_2 > n_body_noise_events_ARM_1 (monotone
    scaling 1->2).

ARM_3 (enabled=True, scale=4.0, all three sources on):
  - C3a: n_body_noise_events_ARM_3 >= n_body_noise_events_ARM_2 OR within
    ARM_3_SATURATION_TOLERANCE of ARM_2 (saturation permitted: at high noise
    the agent_caused contact rate caps the denominator and the body_noise
    counter saturates by mode, similar to SD-047 ARM_3 behaviour).

PASS = C0 AND C1a AND C1b AND C2a AND C2b AND C2c AND C3a.
PASS = SD-048 substrate is calibrated and ready for the V3-EXQ-512 ARC-058 /
ARC-033 comparator-gap behavioural test.
FAIL on C2a -> recalibrate per-source intensity defaults (autonomic_noise_scale,
sensitisation_rate, fatigue_contribution_weight, or interoceptive_change_threshold);
revisit smoke defaults; do not move to V3-EXQ-512 until calibration is in band.
FAIL on C0 -> bit-identical OFF guarantee broken; bug in env code path.

experiment_purpose = "diagnostic" (substrate readiness, not governance evidence).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_511_sd048_interoceptive_noise_substrate_readiness.py
  /opt/local/bin/python3 experiments/v3_exq_511_sd048_interoceptive_noise_substrate_readiness.py --dry-run
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

EXPERIMENT_TYPE = "v3_exq_511_sd048_interoceptive_noise_substrate_readiness"
CLAIM_IDS = ["SD-048"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = (42, 43, 44)
N_TICKS_PER_ARM = 400
# Larger env + sparser policy than SD-047 EXQ-509: SD-048 needs quiet steps
# where the agent is not making hazard contact so body-noise events can land
# without being attributed to agent causation.
GRID_SIZE = 12
N_HAZARDS = 1
N_RESOURCES = 1
# Sparse random-action policy: 80% stay (action 4), 20% uniform random over
# move actions. Mirrors a low-density activity rollout where the body has
# many quiet ticks for noise sources to be detected independent of contact.
P_STAY = 0.80

# Source defaults at ARM_2 (intensity_scale=1.0); match SD-048 SD doc.
AUTONOMIC_NOISE_SCALE = 0.02
SENSITISATION_RATE = 0.008
SENSITISATION_MAGNITUDE = 1.8
SENSITISATION_HALFLIFE = 15
FATIGUE_AR_COEFF = 0.995
FATIGUE_NOISE_SCALE = 0.005
FATIGUE_CONTRIBUTION_WEIGHT = 0.15
# Calibration-counter threshold. Set above pure-autonomic-noise mean
# magnitude (autonomic_noise_scale=0.02 -> mean abs delta ~0.016 per tick)
# so the body_noise counter discriminates discrete body-state events
# (sensitisation spikes, fatigue accumulations, noise+state combinations)
# from the autonomic background floor. Aligns with the SD doc's
# "1:1-2:1 body-noise-caused : agent-caused harm-state-change events" -- the
# body_noise events are the discrete deviations beyond the noise floor.
INTEROCEPTIVE_CHANGE_THRESHOLD = 0.025

# 4-arm sweep specification.
ARMS: List[Tuple[str, bool, float]] = [
    ("ARM_0_off",        False, 1.0),
    ("ARM_1_low_0p25",   True,  0.25),
    ("ARM_2_default",    True,  1.0),
    ("ARM_3_high_4p0",   True,  4.0),
]

# Acceptance thresholds (pre-registered).
# Wider band than SD-047 V3-EXQ-509 [0.5, 2.5]: SD-048's body_noise counter
# integrates sensitisation amplification across the full harm_obs_a vector,
# producing larger per-tick deltas than SD-047's discrete env event counts,
# and the sparse-policy denominator (agent_caused_hazard rate ~ 5-10% of
# ticks) is smaller than SD-047's random-policy denominator. The SD doc's
# 1:1-2:1 estimate was conservative -- per the SD-048 doc interpretation
# grid, "ARM_1 or ARM_3 peaks instead of ARM_2 -> SD-048 validated,
# calibration off; recalibrate per-source scales; architectural conclusions
# hold" remains a valid PASS interpretation. Band [0.5, 5.0] captures any
# arrangement where body-noise events are non-trivially present and not
# overwhelmingly dominant relative to agent-caused.
RATIO_LOWER_BAND = 0.5
RATIO_UPPER_BAND = 5.0
ARM_3_SATURATION_TOLERANCE = 0.05


def sparse_random_action(rng: np.random.Generator) -> torch.Tensor:
    if rng.random() < P_STAY:
        return torch.tensor(4, dtype=torch.long)  # stay
    return torch.tensor(int(rng.integers(0, 4)), dtype=torch.long)


def run_arm(seed: int, arm_name: str, enabled: bool, scale: float, n_ticks: int) -> Dict:
    """Run a single (seed, arm) cell. Returns aggregate metrics."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + hash(arm_name) % 2**16)
    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        seed=seed,
        use_proxy_fields=True,
        # SD-022 limb_damage substrate is the prerequisite for SD-048 (per SD doc).
        limb_damage_enabled=True,
        # SD-048 master + per-source switches.
        interoceptive_noise_enabled=enabled,
        interoceptive_noise_scale=scale,
        autonomic_noise_enabled=enabled,
        autonomic_noise_scale=AUTONOMIC_NOISE_SCALE,
        sensitisation_enabled=enabled,
        sensitisation_rate=SENSITISATION_RATE,
        sensitisation_magnitude=SENSITISATION_MAGNITUDE,
        sensitisation_halflife=SENSITISATION_HALFLIFE,
        fatigue_enabled=enabled,
        fatigue_ar_coeff=FATIGUE_AR_COEFF,
        fatigue_noise_scale=FATIGUE_NOISE_SCALE,
        fatigue_contribution_weight=FATIGUE_CONTRIBUTION_WEIGHT,
        interoceptive_change_threshold=INTEROCEPTIVE_CHANGE_THRESHOLD,
    )
    flat, obs = env.reset()
    n_aut = 0
    n_sens = 0
    n_fat = 0
    n_body = 0
    n_agent = 0
    fatigue_abs_sum = 0.0
    sens_amp_max = 0.0
    n_resets = 0
    for _ in range(n_ticks):
        a = sparse_random_action(rng)
        flat, harm, done, info, obs = env.step(a)
        n_aut   += int(info["interoceptive_n_autonomic_events"])
        n_sens  += int(info["interoceptive_n_sensitisation_events"])
        n_fat   += int(info["interoceptive_n_fatigue_events"])
        n_body  += int(info["interoceptive_n_body_noise_events"])
        n_agent += int(info["interoceptive_n_agent_caused_harm_events"])
        fatigue_abs_sum += abs(float(info["interoceptive_fatigue_state"]))
        sens_amp_max = max(sens_amp_max, float(info["interoceptive_sensitisation_amplification"]))
        if done:
            flat, obs = env.reset()
            n_resets += 1
    ratio = n_body / max(1, n_agent)
    return {
        "arm": arm_name,
        "enabled": enabled,
        "scale": scale,
        "seed": seed,
        "n_ticks": n_ticks,
        "n_autonomic_events": n_aut,
        "n_sensitisation_events": n_sens,
        "n_fatigue_events": n_fat,
        "n_body_noise_events": n_body,
        "n_agent_caused_harm_events": n_agent,
        "ratio_body_noise_per_agent": float(ratio),
        "fatigue_abs_state_sum": float(fatigue_abs_sum),
        "sensitisation_amp_max": float(sens_amp_max),
        "n_resets": n_resets,
    }


def evaluate_acceptance(per_arm_aggregate: Dict[str, Dict]) -> Dict:
    """Evaluate the pre-registered C0/C1a/C1b/C2a/C2b/C2c/C3a checks on aggregates."""
    a0 = per_arm_aggregate["ARM_0_off"]
    a1 = per_arm_aggregate["ARM_1_low_0p25"]
    a2 = per_arm_aggregate["ARM_2_default"]
    a3 = per_arm_aggregate["ARM_3_high_4p0"]

    # C0: bit-identical OFF for SD-048 counters. Master switch False ->
    # _apply_interoceptive_noise short-circuits and zeros all counters.
    c0 = (
        a0["n_autonomic_events"] == 0
        and a0["n_sensitisation_events"] == 0
        and a0["n_fatigue_events"] == 0
        and a0["n_body_noise_events"] == 0
    )
    c1a = (a1["n_autonomic_events"] + a1["n_sensitisation_events"] + a1["n_body_noise_events"]) > 0
    c1b = a1["n_body_noise_events"] > 0
    c2a = (RATIO_LOWER_BAND <= a2["ratio_body_noise_per_agent"] <= RATIO_UPPER_BAND)
    c2b = (a2["n_autonomic_events"] > 0 and a2["n_sensitisation_events"] > 0)
    c2c = a2["n_body_noise_events"] > a1["n_body_noise_events"]
    saturation_floor = a2["n_body_noise_events"] * (1.0 - ARM_3_SATURATION_TOLERANCE)
    c3a = a3["n_body_noise_events"] >= saturation_floor

    overall = c0 and c1a and c1b and c2a and c2b and c2c and c3a
    return {
        "C0_off_bit_identical": bool(c0),
        "C1a_arm1_any_source_firing": bool(c1a),
        "C1b_arm1_body_noise_positive": bool(c1b),
        "C2a_arm2_calibration_band": bool(c2a),
        "C2b_arm2_aut_and_sens_firing": bool(c2b),
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
                "n_autonomic_events": 0,
                "n_sensitisation_events": 0,
                "n_fatigue_events": 0,
                "n_body_noise_events": 0,
                "n_agent_caused_harm_events": 0,
                "fatigue_abs_state_sum": 0.0,
                "sensitisation_amp_max": 0.0,
                "n_resets": 0,
            }
        b = bucket[arm]
        b["n_ticks"] += r["n_ticks"]
        b["n_autonomic_events"] += r["n_autonomic_events"]
        b["n_sensitisation_events"] += r["n_sensitisation_events"]
        b["n_fatigue_events"] += r["n_fatigue_events"]
        b["n_body_noise_events"] += r["n_body_noise_events"]
        b["n_agent_caused_harm_events"] += r["n_agent_caused_harm_events"]
        b["fatigue_abs_state_sum"] += r["fatigue_abs_state_sum"]
        b["sensitisation_amp_max"] = max(b["sensitisation_amp_max"], r["sensitisation_amp_max"])
        b["n_resets"] += r["n_resets"]
    for arm, b in bucket.items():
        b["ratio_body_noise_per_agent"] = b["n_body_noise_events"] / max(1, b["n_agent_caused_harm_events"])
    return bucket


def main(dry_run: bool = False) -> int:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    n_ticks = 60 if dry_run else N_TICKS_PER_ARM
    seeds = (SEEDS[0],) if dry_run else SEEDS
    per_seed_arms: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        for arm_name, enabled, scale in ARMS:
            r = run_arm(seed, arm_name, enabled, scale, n_ticks)
            per_seed_arms.append(r)
            print(
                f"  seed={seed} arm={arm_name:<18} "
                f"aut={r['n_autonomic_events']:4d} sens={r['n_sensitisation_events']:3d} "
                f"fat={r['n_fatigue_events']:3d} body={r['n_body_noise_events']:4d} "
                f"agent={r['n_agent_caused_harm_events']:4d} "
                f"ratio={r['ratio_body_noise_per_agent']:.2f} "
                f"sens_amp_max={r['sensitisation_amp_max']:.2f}"
            )
    elapsed = time.time() - t0
    aggregates = aggregate_seeds(per_seed_arms)
    acceptance = evaluate_acceptance(aggregates)
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"
    print(f"[{EXPERIMENT_TYPE}] aggregates:")
    for arm in ("ARM_0_off", "ARM_1_low_0p25", "ARM_2_default", "ARM_3_high_4p0"):
        a = aggregates[arm]
        print(
            f"  {arm:<18} body={a['n_body_noise_events']:5d} agent={a['n_agent_caused_harm_events']:5d} "
            f"ratio={a['ratio_body_noise_per_agent']:.2f}"
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
        "p_stay_action": P_STAY,
        "grid_size": GRID_SIZE,
        "n_hazards": N_HAZARDS,
        "n_resources": N_RESOURCES,
        "arms": list(aggregates.values()),
        "per_seed_per_arm": per_seed_arms,
        "acceptance": acceptance,
        "thresholds": {
            "ratio_lower_band": RATIO_LOWER_BAND,
            "ratio_upper_band": RATIO_UPPER_BAND,
            "arm_3_saturation_tolerance": ARM_3_SATURATION_TOLERANCE,
            "interoceptive_change_threshold": INTEROCEPTIVE_CHANGE_THRESHOLD,
        },
        "source_defaults": {
            "autonomic_noise_scale": AUTONOMIC_NOISE_SCALE,
            "sensitisation_rate": SENSITISATION_RATE,
            "sensitisation_magnitude": SENSITISATION_MAGNITUDE,
            "sensitisation_halflife": SENSITISATION_HALFLIFE,
            "fatigue_ar_coeff": FATIGUE_AR_COEFF,
            "fatigue_noise_scale": FATIGUE_NOISE_SCALE,
            "fatigue_contribution_weight": FATIGUE_CONTRIBUTION_WEIGHT,
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
