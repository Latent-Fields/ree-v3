#!/opt/local/bin/python3
"""V3-EXQ-502 -- EXP-0173 MECH-062 tri-loop gate coordination."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from experiments.pack_writer import write_flat_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT_TYPE = "v3_exq_502_mech062_tri_loop_gate_coordination"
CLAIM_IDS = ["MECH-062"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = (42, 43, 44)
POLICIES = ("TRI_LOOP_COORDINATED", "SINGLE_LIMBIC_LOOP")
ENV_REGIMES = ("STABLE", "VOLATILE")
NUM_SEQUENCES = 120
SEQUENCE_LENGTH = 8
MAX_WAIT_STEPS = 24
GATE_THRESHOLD = 0.55
DISAGREEMENT_THRESHOLD = 0.25

C1_MIN_STABLE_GAP = 0.03
C2_MIN_VOLATILE_GAP = 0.12
C3_MIN_GAP_WIDENING = 0.08
C4_MIN_TRI_DISAGREEMENT_RATE = 0.15


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sample_gates(rng: random.Random, regime: str) -> tuple[float, float, float]:
    if regime == "STABLE":
        base = rng.uniform(0.64, 0.92)
        motor = _clip(base + rng.gauss(0.0, 0.06))
        cognitive = _clip(base + rng.gauss(0.0, 0.05))
        motivational = _clip(base + rng.gauss(0.0, 0.04))
        if rng.random() < 0.04:
            motor = rng.uniform(0.25, 0.52)
        if rng.random() < 0.04:
            cognitive = rng.uniform(0.25, 0.52)
        return motor, cognitive, motivational

    if regime == "VOLATILE":
        motivational = rng.uniform(0.58, 0.96)
        motor = rng.uniform(0.25, 0.94)
        cognitive = rng.uniform(0.25, 0.94)
        if rng.random() < 0.30:
            motor = rng.uniform(0.15, 0.48)
        if rng.random() < 0.26:
            cognitive = rng.uniform(0.15, 0.48)
        return motor, cognitive, motivational

    raise ValueError(f"unknown regime {regime}")


def _run_sequence(rng: random.Random, policy: str, regime: str) -> dict:
    committed_step = 0
    waits = 0
    steps = 0
    disagreements = 0
    premature = False

    while committed_step < SEQUENCE_LENGTH and waits <= MAX_WAIT_STEPS and steps < 200:
        motor, cognitive, motivational = _sample_gates(rng, regime)
        gates = (motor, cognitive, motivational)
        disagreement = (max(gates) - min(gates)) > DISAGREEMENT_THRESHOLD
        disagreements += int(disagreement)
        steps += 1

        if policy == "TRI_LOOP_COORDINATED":
            if all(g >= GATE_THRESHOLD for g in gates):
                committed_step += 1
                waits = 0
            else:
                waits += 1
            continue

        if policy == "SINGLE_LIMBIC_LOOP":
            if motivational >= GATE_THRESHOLD:
                if motor < GATE_THRESHOLD or cognitive < GATE_THRESHOLD:
                    premature = True
                    break
                committed_step += 1
                waits = 0
            else:
                waits += 1
            continue

        raise ValueError(f"unknown policy {policy}")

    completed = committed_step >= SEQUENCE_LENGTH and not premature
    return {
        "completed": bool(completed),
        "premature_termination": bool(premature),
        "steps": steps,
        "disagreement_steps": disagreements,
        "disagreement_rate": disagreements / max(1, steps),
    }


def run_condition(seed: int, policy: str, regime: str, dry_run: bool) -> dict:
    salt = sum(ord(ch) for ch in f"{policy}:{regime}")
    rng = random.Random(seed * 1009 + salt)
    n = 16 if dry_run else NUM_SEQUENCES
    condition = f"{policy}_{regime}"
    print(f"Seed {seed} Condition {condition}", flush=True)
    rows = []
    for i in range(n):
        if (i + 1) == 1 or (i + 1) % 30 == 0 or (i + 1) == n:
            print(f"  [train] {condition} seed={seed} ep {i + 1}/{NUM_SEQUENCES}", flush=True)
        rows.append(_run_sequence(rng, policy, regime))
    completed = sum(1 for r in rows if r["completed"])
    premature = sum(1 for r in rows if r["premature_termination"])
    stability = completed / max(1, n)
    disagreement_rate = statistics.mean(r["disagreement_rate"] for r in rows)
    condition_pass = stability >= 0.50
    print(f"verdict: {'PASS' if condition_pass else 'FAIL'}", flush=True)
    return {
        "seed": seed,
        "policy": policy,
        "regime": regime,
        "condition": condition,
        "n_sequences": n,
        "action_stability": stability,
        "premature_termination_rate": premature / max(1, n),
        "cross_loop_disagreement_rate": disagreement_rate,
    }


def _evaluate(results: list[dict]) -> dict:
    by_seed: dict[int, dict[str, dict]] = {}
    for r in results:
        key = f"{r['policy']}_{r['regime']}"
        by_seed.setdefault(int(r["seed"]), {})[key] = r

    required = min(2, max(1, len(by_seed)))
    stable_gap_pass = 0
    volatile_gap_pass = 0
    widening_pass = 0
    disagreement_pass = 0
    deltas = []
    for seed, arms in by_seed.items():
        tri_stable = arms["TRI_LOOP_COORDINATED_STABLE"]["action_stability"]
        single_stable = arms["SINGLE_LIMBIC_LOOP_STABLE"]["action_stability"]
        tri_volatile = arms["TRI_LOOP_COORDINATED_VOLATILE"]["action_stability"]
        single_volatile = arms["SINGLE_LIMBIC_LOOP_VOLATILE"]["action_stability"]
        stable_gap = tri_stable - single_stable
        volatile_gap = tri_volatile - single_volatile
        widening = volatile_gap - stable_gap
        tri_disagreement = (
            arms["TRI_LOOP_COORDINATED_STABLE"]["cross_loop_disagreement_rate"]
            + arms["TRI_LOOP_COORDINATED_VOLATILE"]["cross_loop_disagreement_rate"]
        ) / 2.0
        stable_gap_pass += int(stable_gap >= C1_MIN_STABLE_GAP)
        volatile_gap_pass += int(volatile_gap >= C2_MIN_VOLATILE_GAP)
        widening_pass += int(widening >= C3_MIN_GAP_WIDENING)
        disagreement_pass += int(tri_disagreement >= C4_MIN_TRI_DISAGREEMENT_RATE)
        deltas.append(
            {
                "seed": seed,
                "stable_gap": stable_gap,
                "volatile_gap": volatile_gap,
                "gap_widening": widening,
                "tri_disagreement_rate": tri_disagreement,
            }
        )

    c1 = stable_gap_pass >= required
    c2 = volatile_gap_pass >= required
    c3 = widening_pass >= required
    c4 = disagreement_pass >= required
    return {
        "c1_action_stability_stable_seed_passes": stable_gap_pass,
        "c2_volatile_resilience_seed_passes": volatile_gap_pass,
        "c3_gap_widens_in_volatile_seed_passes": widening_pass,
        "c4_cross_loop_disagreement_nontrivial_seed_passes": disagreement_pass,
        "min_seed_passes_required": required,
        "c1_pass": c1,
        "c2_pass": c2,
        "c3_pass": c3,
        "c4_pass": c4,
        "overall_pass": bool(c1 and c2 and c3 and c4),
        "paired_deltas": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = (SEEDS[0],) if args.dry_run else SEEDS
    t0 = time.time()
    results = [
        run_condition(seed, policy, regime, dry_run=args.dry_run)
        for seed in seeds
        for regime in ENV_REGIMES
        for policy in POLICIES
    ]
    elapsed = time.time() - t0
    criteria = _evaluate(results)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if outcome == "PASS" else "weakens"

    print(f"V3-EXQ-502 MECH-062 tri-loop coordination -- {outcome} in {elapsed:.1f}s", flush=True)
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
        "evidence_direction_per_claim": {"MECH-062": direction},
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MIN_STABLE_GAP": C1_MIN_STABLE_GAP,
            "C2_MIN_VOLATILE_GAP": C2_MIN_VOLATILE_GAP,
            "C3_MIN_GAP_WIDENING": C3_MIN_GAP_WIDENING,
            "C4_MIN_TRI_DISAGREEMENT_RATE": C4_MIN_TRI_DISAGREEMENT_RATE,
        },
        "config": {
            "seeds": list(seeds),
            "policies": list(POLICIES),
            "env_regimes": list(ENV_REGIMES),
            "num_sequences": NUM_SEQUENCES,
            "sequence_length": SEQUENCE_LENGTH,
            "max_wait_steps": MAX_WAIT_STEPS,
            "gate_threshold": GATE_THRESHOLD,
        },
        "condition_results": results,
        "elapsed_seconds": elapsed,
        "notes": (
            "Standalone mechanism experiment for MECH-062. The coordinated "
            "policy requires motor, cognitive-set, and motivational gates to "
            "agree before advancing a committed sequence; the ablation listens "
            "only to the limbic/motivational loop. PASS requires higher action "
            "stability in coordinated tri-loop gating, especially under volatile "
            "cross-loop disagreement."
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
