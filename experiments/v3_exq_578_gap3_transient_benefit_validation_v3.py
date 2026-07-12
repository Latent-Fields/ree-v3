"""
V3-EXQ-578: GAP-3 transient benefit patches env feature substrate validation.

ARM_0: transient_benefit_enabled=False -- no patch spawns; all transient_benefit_*
       counters stay exactly zero across the run.
ARM_1: transient_benefit_enabled=True (prob=0.15, duration=15, multiplier=2.0) --
       patches spawn at ~the configured rate, expire after `duration` steps, and
       any contact pays exactly resource_benefit * transient_benefit_multiplier.

PASS = C1 AND C2 AND C3 AND C4 across all seeds:
  C1  ARM_1 spawns at the configured rate: total n_spawned > 0 AND observed
      per-step spawn fraction within [SPAWN_LO, SPAWN_HI] of transient_benefit_prob.
  C2  ARM_0 fully silent: n_spawned == 0 AND n_contacted == 0 AND n_expired == 0.
  C3  ARM_1 patches expire: total n_expired > 0 (patches age out, not leak).
  C4  ARM_1 contact-reward invariant: on every contact tick,
      transient_benefit_contact_this_tick == resource_benefit * multiplier
      (within 1e-9). Vacuously satisfied if no contacts occur; any single
      mismatched contact tick fails C4.

experiment_purpose: diagnostic (substrate readiness test, not a claim hypothesis
test). Unblocks DEV-NEED-006 (z_goal seeding via accidental benefit contacts) and
MECH-189 (super-ordinal goal formation) per
REE_assembly/evidence/planning/infant_substrate_plan.md (GAP-3).
claim_ids: [] (env feature validation only; no governance weighting).
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiment_protocol import emit_outcome
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from pathlib import Path  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
QUEUE_ID = "V3-EXQ-578"

SEEDS = [0, 1, 2]
N_EPISODES = 100
N_STEPS = 200

RESOURCE_BENEFIT = 0.3
TB_PROB = 0.15
TB_DURATION = 15
TB_MULTIPLIER = 2.0

# C1 spawn-fraction band: observed per-step spawn fraction must land within a
# generous relative band of TB_PROB. The band is wide enough to absorb the
# episode-length variation introduced by random-walker contamination death
# (episodes end early, so the empties pool is always non-empty -> the spawn
# Bernoulli fires at ~TB_PROB) but tight enough to catch a broken spawn path
# (always-on, never-on, or wrong probability).
SPAWN_LO = 0.5 * TB_PROB   # 0.075
SPAWN_HI = 1.5 * TB_PROB   # 0.225

EXPECTED_CONTACT = RESOURCE_BENEFIT * TB_MULTIPLIER  # 0.6
CONTACT_TOL = 1e-9

CONDITIONS = [
    ("ARM_0_disabled", False),
    ("ARM_1_enabled", True),
]


def run_experiment(n_episodes, dry_run=False):
    results_by_seed = {}
    c1_by_seed = []
    c2_by_seed = []
    c3_by_seed = []
    c4_by_seed = []

    for seed in SEEDS:
        results_by_seed[seed] = {}
        for cond_label, tb_enabled in CONDITIONS:
            print(f"Seed {seed} Condition {cond_label}", flush=True)

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            env = CausalGridWorldV2(
                size=12,
                seed=seed,
                num_hazards=0,
                num_resources=2,
                use_proxy_fields=False,
                resource_benefit=RESOURCE_BENEFIT,
                transient_benefit_enabled=tb_enabled,
                transient_benefit_prob=TB_PROB,
                transient_benefit_duration=TB_DURATION,
                transient_benefit_multiplier=TB_MULTIPLIER,
            )

            total_steps = 0
            total_spawned = 0
            total_contacted = 0
            total_expired = 0
            max_contact_value = 0.0
            n_contact_ticks = 0
            n_contact_mismatch = 0

            for ep in range(n_episodes):
                env.reset()
                done = False
                last_info = None
                for _ in range(N_STEPS):
                    if done:
                        break
                    action = np.random.randint(0, 5)
                    _, _, done, info, _ = env.step(action)
                    last_info = info
                    total_steps += 1
                    cv = info["transient_benefit_contact_this_tick"]
                    if cv > 0.0:
                        n_contact_ticks += 1
                        max_contact_value = max(max_contact_value, cv)
                        if abs(cv - EXPECTED_CONTACT) > CONTACT_TOL:
                            n_contact_mismatch += 1
                # env counters are per-episode (reset on env.reset()); read the
                # final tick's cumulative episode counts and accumulate.
                if last_info is not None:
                    total_spawned += int(last_info["transient_benefit_n_spawned"])
                    total_contacted += int(
                        last_info["transient_benefit_n_contacted"]
                    )
                    total_expired += int(last_info["transient_benefit_n_expired"])

                print_interval = max(1, n_episodes // 5)
                if (ep + 1) % print_interval == 0:
                    print(
                        f"  [train] seed={seed} cond={cond_label} "
                        f"ep {ep + 1}/{n_episodes}",
                        flush=True,
                    )

            spawn_fraction = total_spawned / max(total_steps, 1)

            results_by_seed[seed][cond_label] = {
                "tb_enabled": tb_enabled,
                "total_steps": total_steps,
                "total_spawned": total_spawned,
                "total_contacted": total_contacted,
                "total_expired": total_expired,
                "spawn_fraction": spawn_fraction,
                "n_contact_ticks": n_contact_ticks,
                "n_contact_mismatch": n_contact_mismatch,
                "max_contact_value": max_contact_value,
            }

            if tb_enabled:
                # C1 spawn rate, C3 expiry, C4 contact-reward invariant.
                c1 = (
                    total_spawned > 0
                    and SPAWN_LO <= spawn_fraction <= SPAWN_HI
                )
                c3 = total_expired > 0
                c4 = n_contact_mismatch == 0
                c1_by_seed.append(c1)
                c3_by_seed.append(c3)
                c4_by_seed.append(c4)
                passed = c1 and c3 and c4
            else:
                # C2 fully silent when disabled.
                c2 = (
                    total_spawned == 0
                    and total_contacted == 0
                    and total_expired == 0
                    and n_contact_ticks == 0
                )
                c2_by_seed.append(c2)
                passed = c2

            verdict_str = "PASS" if passed else "FAIL"
            print(f"verdict: {verdict_str}", flush=True)

    c1_pass = all(c1_by_seed) if c1_by_seed else False
    c2_pass = all(c2_by_seed) if c2_by_seed else False
    c3_pass = all(c3_by_seed) if c3_by_seed else False
    c4_pass = all(c4_by_seed) if c4_by_seed else False
    overall_pass = c1_pass and c2_pass and c3_pass and c4_pass

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c4_pass": c4_pass,
        "c1_by_seed": c1_by_seed,
        "c2_by_seed": c2_by_seed,
        "c3_by_seed": c3_by_seed,
        "c4_by_seed": c4_by_seed,
        "results_by_seed": results_by_seed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dry_run = args.dry_run
    n_episodes = 5 if dry_run else N_EPISODES

    print("V3-EXQ-578 GAP-3 transient benefit patches validation", flush=True)
    print(
        f"  dry_run={dry_run} n_episodes={n_episodes} seeds={SEEDS} "
        f"prob={TB_PROB} duration={TB_DURATION} multiplier={TB_MULTIPLIER}",
        flush=True,
    )

    result = run_experiment(n_episodes=n_episodes, dry_run=dry_run)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_578_gap3_transient_benefit_validation_{timestamp}_v3"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": "gap3_transient_benefit_validation",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": [],
        "evidence_direction": "non_contributory",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "dry_run": dry_run,
        "config": {
            "seeds": SEEDS,
            "n_episodes": n_episodes,
            "n_steps": N_STEPS,
            "resource_benefit": RESOURCE_BENEFIT,
            "tb_prob": TB_PROB,
            "tb_duration": TB_DURATION,
            "tb_multiplier": TB_MULTIPLIER,
            "spawn_lo": SPAWN_LO,
            "spawn_hi": SPAWN_HI,
            "expected_contact": EXPECTED_CONTACT,
        },
        "acceptance_checks": {
            "C1_arm1_spawn_rate": result["c1_pass"],
            "C2_arm0_fully_silent": result["c2_pass"],
            "C3_arm1_patches_expire": result["c3_pass"],
            "C4_arm1_contact_multiplier_exact": result["c4_pass"],
        },
        "c1_by_seed": result["c1_by_seed"],
        "c2_by_seed": result["c2_by_seed"],
        "c3_by_seed": result["c3_by_seed"],
        "c4_by_seed": result["c4_by_seed"],
        "results_by_seed": result["results_by_seed"],
    }

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
    print(f"Outcome: {result['outcome']}", flush=True)
    print(f"  C1 (ARM_1 spawn rate):            {result['c1_pass']}", flush=True)
    print(f"  C2 (ARM_0 fully silent):          {result['c2_pass']}", flush=True)
    print(f"  C3 (ARM_1 patches expire):        {result['c3_pass']}", flush=True)
    print(f"  C4 (ARM_1 contact multiplier):    {result['c4_pass']}", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
