#!/opt/local/bin/python3
"""V3-EXQ-505 -- MECH-093 z_beta vs precision dissociation, substrate-level.

Claim: MECH-093 (affective.z_beta_e3_update_rate_modulation)
Status: candidate (exp_conf=0.68, 3 PASS / 7 FAIL across 14 runs)

Why this experiment exists
--------------------------
The claim asserts: z_beta modulates the E3 heartbeat update rate, AND this
mechanism is DISTINCT from precision-weighting (which modulates commit
thresholds, not rate). EXQ-097 FAILed (C1 p1_rate_gap=-0.74) -- the rate
gap between high-arousal and low-arousal episodes was NEGATIVE in an env
where high-harm was the predicted high-arousal condition, suggesting either
(i) z_beta in the env loop did not actually correlate with harm, or
(ii) some confound (e.g. action quiescence under harm) overwhelmed the
expected rate effect. EXQ-116 multiseed produced mixed evidence.

This experiment isolates the substrate by injecting controlled z_beta and
salience signals directly into the MultiRateClock (the substrate of MECH-093,
ree_core/heartbeat/clock.py:201 update_e3_rate_from_beta), bypassing any
env-loop confound. The hypothesis is sharply falsifiable at substrate level:
if z_beta directly controls e3_steps_per_tick and salience does NOT enter
the rate computation, then a 2x2 factorial should show a main effect of
z_beta, no main effect of salience, and no interaction.

2x2 factorial (3 seeds x 4 conditions = 12 runs)
------------------------------------------------
Each run advances a fresh MultiRateClock for N_STEPS env steps and counts
how many e3_ticks fired.

ARM_A (LOW_BETA, NO_SALIENCE):
  z_beta_norm ~ 0.1 each step. clock.update_e3_rate_from_beta(z_beta).
  No salience marking. Predicts ~ N_STEPS / max_steps e3 ticks (slow rate).

ARM_B (LOW_BETA, HIGH_SALIENCE):
  z_beta_norm ~ 0.1 each step. Same update_e3_rate_from_beta call.
  clock.mark_salient() called every step. Predicts e3 tick count
  approximately equal to ARM_A (salience does not modulate rate).

ARM_C (HIGH_BETA, NO_SALIENCE):
  z_beta_norm ~ 1.0 each step. update_e3_rate_from_beta drives rate up.
  Predicts ~ N_STEPS / min_steps e3 ticks (fast rate).

ARM_D (HIGH_BETA, HIGH_SALIENCE):
  z_beta_norm ~ 1.0 each step. Both update_e3_rate_from_beta and
  mark_salient called each step. Predicts e3 tick count approximately
  equal to ARM_C (the dissociation: precision channel does not bleed
  into rate channel).

Pre-registered metrics
----------------------
  e3_tick_count_<arm>: ticks fired over N_STEPS clock advances.
  beta_main_effect = mean(C, D) - mean(A, B)            -- expected POSITIVE and large
  salience_main_effect = mean(B, D) - mean(A, C)        -- expected NEAR ZERO
  interaction = (D - C) - (B - A)                       -- expected NEAR ZERO

PASS criteria (>= 2/3 seeds for each)
-------------------------------------
  C1 beta_main_effect: high-beta arms produce >= 1.5x as many ticks as
     low-beta arms in the same seed. Gap = (C+D)/2 - (A+B)/2 must be
     >= 0.5 * (C+D)/2 (relative gap >= 50%).
  C2 salience_main_effect_small: |salience_main_effect| / mean_tick_count
     <= 0.10 (salience changes total tick count by <=10% across seeds).
  C3 no_interaction: |interaction| / mean_tick_count <= 0.10.

PASS = C1 AND C2 AND C3.
PASS supports MECH-093 strong reading: z_beta is the rate channel, precision
is a separate channel; the two are dissociable at substrate.
FAIL with C1 alone PASSing -> z_beta moves rate but salience also leaks
in; the precision/rate decoupling is incomplete (architectural concern).
FAIL with C1 failing -> the substrate update_e3_rate_from_beta does not
produce the expected magnitude of rate change at the tested |z_beta|
levels; check beta_magnitude_scale calibration vs config defaults.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_505_mech093_zbeta_precision_dissociation.py
  /opt/local/bin/python3 experiments/v3_exq_505_mech093_zbeta_precision_dissociation.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.heartbeat.clock import MultiRateClock  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_505_mech093_zbeta_precision_dissociation"
CLAIM_IDS = ["MECH-093"]
EXPERIMENT_PURPOSE = "evidence"

# --- Config -------------------------------------------------------------
SEEDS = (42, 43, 44)
N_STEPS = 500
BETA_DIM = 16
LOW_BETA_NORM_TARGET = 0.1
HIGH_BETA_NORM_TARGET = 1.5

# Match ree_core defaults so the substrate is exercised at canonical
# calibration (HeartbeatConfig defaults from ree_core/utils/config.py:866
# clock are the agent's installed values; we instantiate the same shape).
E3_BASE_STEPS = 10
BETA_RATE_MIN_STEPS = 5
BETA_RATE_MAX_STEPS = 20
BETA_MAGNITUDE_SCALE = 1.0

# Pre-registered thresholds.
C1_MIN_RELATIVE_GAP = 0.50           # high-beta arms produce >=1.5x ticks
C2_MAX_SALIENCE_REL_EFFECT = 0.10
C3_MAX_INTERACTION_REL = 0.10
PASS_FRACTION_REQUIRED = 2.0 / 3.0


def _make_clock() -> MultiRateClock:
    return MultiRateClock(
        e1_steps_per_tick=1,
        e2_steps_per_tick=3,
        e3_steps_per_tick=E3_BASE_STEPS,
        beta_rate_min_steps=BETA_RATE_MIN_STEPS,
        beta_rate_max_steps=BETA_RATE_MAX_STEPS,
        beta_magnitude_scale=BETA_MAGNITUDE_SCALE,
    )


def _z_beta_with_target_norm(target_norm: float, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    raw = torch.randn(1, BETA_DIM, generator=g)
    cur = raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return raw * (target_norm / cur)


def run_arm(seed: int, arm_label: str, beta_target: float, mark_salience: bool,
            n_steps: int) -> Dict:
    """Drive a fresh clock for n_steps and tally e3 ticks."""
    torch.manual_seed(seed)
    clock = _make_clock()

    z_beta = _z_beta_with_target_norm(beta_target, seed)
    realized_norm = float(z_beta.norm(dim=-1).mean().item())

    e3_tick_count = 0
    e3_step_window_samples: List[int] = []  # snapshot e3_steps_per_tick over time

    for step in range(n_steps):
        # Update rate from controlled z_beta BEFORE advancing.
        clock.update_e3_rate_from_beta(z_beta)
        if mark_salience:
            clock.mark_salient()
        ticks = clock.advance()
        if ticks["e3_tick"]:
            e3_tick_count += 1
        if step % max(1, n_steps // 20) == 0:
            e3_step_window_samples.append(int(clock.e3_steps_per_tick))

    return {
        "seed": seed,
        "arm_label": arm_label,
        "beta_target_norm": float(beta_target),
        "beta_realized_norm": realized_norm,
        "mark_salience": bool(mark_salience),
        "e3_tick_count": int(e3_tick_count),
        "n_steps": int(n_steps),
        "current_e3_steps_samples": e3_step_window_samples,
    }


def _evaluate(arm_a: List[Dict], arm_b: List[Dict], arm_c: List[Dict],
              arm_d: List[Dict]) -> Dict:
    n = len(arm_a)
    required = math.ceil(n * PASS_FRACTION_REQUIRED)

    c1_passes = 0
    c2_passes = 0
    c3_passes = 0
    rows = []
    for a, b, c, d in zip(arm_a, arm_b, arm_c, arm_d):
        a_ct = a["e3_tick_count"]
        b_ct = b["e3_tick_count"]
        c_ct = c["e3_tick_count"]
        d_ct = d["e3_tick_count"]
        low_mean = (a_ct + b_ct) / 2.0
        high_mean = (c_ct + d_ct) / 2.0
        beta_main = high_mean - low_mean
        salience_main = ((b_ct + d_ct) / 2.0) - ((a_ct + c_ct) / 2.0)
        interaction = (d_ct - c_ct) - (b_ct - a_ct)
        mean_ct = max(1.0, (a_ct + b_ct + c_ct + d_ct) / 4.0)

        rel_gap = (beta_main / max(1.0, high_mean)) if high_mean > 0 else 0.0
        rel_salience = abs(salience_main) / mean_ct
        rel_interact = abs(interaction) / mean_ct

        c1 = rel_gap >= C1_MIN_RELATIVE_GAP
        c2 = rel_salience <= C2_MAX_SALIENCE_REL_EFFECT
        c3 = rel_interact <= C3_MAX_INTERACTION_REL
        c1_passes += int(c1)
        c2_passes += int(c2)
        c3_passes += int(c3)

        rows.append({
            "seed": a["seed"],
            "low_mean_ticks": low_mean,
            "high_mean_ticks": high_mean,
            "beta_main_effect": beta_main,
            "salience_main_effect": salience_main,
            "interaction": interaction,
            "rel_gap": rel_gap,
            "rel_salience": rel_salience,
            "rel_interaction": rel_interact,
            "c1_pass": c1, "c2_pass": c2, "c3_pass": c3,
        })

    return {
        "n_seeds": n,
        "min_seeds_required": required,
        "c1_seeds_pass": c1_passes,
        "c2_seeds_pass": c2_passes,
        "c3_seeds_pass": c3_passes,
        "c1_pass": c1_passes >= required,
        "c2_pass": c2_passes >= required,
        "c3_pass": c3_passes >= required,
        "overall_pass": (c1_passes >= required and c2_passes >= required
                         and c3_passes >= required),
        "per_seed_breakdown": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    args = parser.parse_args()

    seeds = (args.seeds[0],) if args.dry_run else tuple(args.seeds)
    n_steps = 60 if args.dry_run else args.n_steps
    if args.dry_run:
        print("[DRY-RUN] 1 seed, 60 steps -- smoke only.", flush=True)

    t0 = time.time()
    arm_a = [run_arm(s, "ARM_A_low_no_sal", LOW_BETA_NORM_TARGET, False, n_steps) for s in seeds]
    arm_b = [run_arm(s, "ARM_B_low_high_sal", LOW_BETA_NORM_TARGET, True, n_steps) for s in seeds]
    arm_c = [run_arm(s, "ARM_C_high_no_sal", HIGH_BETA_NORM_TARGET, False, n_steps) for s in seeds]
    arm_d = [run_arm(s, "ARM_D_high_high_sal", HIGH_BETA_NORM_TARGET, True, n_steps) for s in seeds]
    elapsed = time.time() - t0

    criteria = _evaluate(arm_a, arm_b, arm_c, arm_d)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    print(f"\nV3-EXQ-505 (MECH-093) -- {outcome} in {elapsed:.2f}s ({len(seeds)} seed(s))", flush=True)
    for label, results in (
        ("ARM_A_low_no_sal", arm_a), ("ARM_B_low_high_sal", arm_b),
        ("ARM_C_high_no_sal", arm_c), ("ARM_D_high_high_sal", arm_d),
    ):
        for r in results:
            print(f"  {label} seed={r['seed']}  "
                  f"beta_norm={r['beta_realized_norm']:.3f}  "
                  f"e3_ticks={r['e3_tick_count']}/{r['n_steps']}  "
                  f"current_e3_steps_samples={r['current_e3_steps_samples'][:5]}", flush=True)
    print(f"  C1 beta-main-effect (>=50% rel gap):   "
          f"{criteria['c1_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c1_pass'] else 'FAIL'}", flush=True)
    print(f"  C2 salience-main-effect (<=10%):       "
          f"{criteria['c2_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c2_pass'] else 'FAIL'}", flush=True)
    print(f"  C3 no-interaction (<=10%):             "
          f"{criteria['c3_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c3_pass'] else 'FAIL'}", flush=True)

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
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-093": direction},
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MIN_RELATIVE_GAP": C1_MIN_RELATIVE_GAP,
            "C2_MAX_SALIENCE_REL_EFFECT": C2_MAX_SALIENCE_REL_EFFECT,
            "C3_MAX_INTERACTION_REL": C3_MAX_INTERACTION_REL,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
        },
        "config": {
            "n_steps": n_steps, "beta_dim": BETA_DIM,
            "low_beta_norm_target": LOW_BETA_NORM_TARGET,
            "high_beta_norm_target": HIGH_BETA_NORM_TARGET,
            "e3_base_steps": E3_BASE_STEPS,
            "beta_rate_min_steps": BETA_RATE_MIN_STEPS,
            "beta_rate_max_steps": BETA_RATE_MAX_STEPS,
            "beta_magnitude_scale": BETA_MAGNITUDE_SCALE,
            "seeds": list(seeds),
        },
        "results_arm_a_low_no_sal": arm_a,
        "results_arm_b_low_high_sal": arm_b,
        "results_arm_c_high_no_sal": arm_c,
        "results_arm_d_high_high_sal": arm_d,
        "elapsed_seconds": elapsed,
        "notes": (
            "Substrate-level 2x2 factorial isolating MECH-093's z_beta -> "
            "e3_steps_per_tick channel from any precision/salience confound. "
            "Drives MultiRateClock directly via update_e3_rate_from_beta and "
            "mark_salient. PASS supports the strong-reading dissociation. "
            "Resolves the EXQ-097 negative rate-gap puzzle by separating "
            "substrate function from in-env z_beta-vs-harm correlation."
        ),
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
