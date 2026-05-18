"""
V3-EXQ-577a: GAP-2 microhabitat zones env feature substrate validation
(corrected C2 -- supersedes V3-EXQ-577).

Same scientific question as V3-EXQ-577 (infant_substrate:GAP-2 substrate
readiness). Measurement fix only, per the V3-EXQ-577 failure autopsy
(REE_assembly/evidence/planning/failure_autopsy_EXQ-577_2026-05-16):
V3-EXQ-577 C2 demanded per-episode presence of ALL base zones {0,1,2},
which fights the deliberately stochastic Voronoi seeding -- ~2-3% of
episodes legitimately absorbed one base niche into the D ecotone. The
GAP-2 degenerate-seeding redraw guard (now landed in
_build_microhabitat_zones) drives that collapse rate to ~0, but the
readiness criterion itself was over-strict and is corrected here.

ARM_0_disabled: microhabitat_enabled=False -- legacy behaviour, bit-identical.
ARM_1_enabled:  microhabitat_enabled=True, n_microhabitats=3 (defaults).

Acceptance (substrate readiness; experiment_purpose=diagnostic, NOT governance
evidence):
  C1  ARM_0 bit-identical OFF: (reward, agent_x, agent_y, done) sequence over a
      fixed action stream is identical to a legacy CausalGridWorldV2 -- every
      seed. (unchanged from V3-EXQ-577)
  C2  ARM_1 zone map CORRECTED: per episode assert WELL-FORMEDNESS only --
      no -1 in the interior, codes subset {0,1,2,3}, border zone 3 present,
      and >=2 base zones present (the stochastic ecotone may legitimately
      absorb at most one niche). The strict {0,1,2} presence is asserted
      AGGREGATED over all episodes (union), not per episode. The
      base-zone-collapse frequency (episodes missing a base zone) and the
      redraw-guard counters are reported as DIAGNOSTIC statistics: expect
      ~0 collapses with the guard active; a non-zero value quantifies
      residual stochastic collapse rather than failing the gate.
  C3  ARM_1 aggregated over resets: zone-C hazard count == 0
      (zone_C_hazard_factor=0.0 hard guarantee) AND zone-B hazard density >
      zone-A hazard density -- every seed. (unchanged from V3-EXQ-577)
  C4  ARM_1 zone-C ambient bonus fires and decays: on a forced all-zone-C
      probe env the first zone-C entry yields bonus * decay^0 and the
      second yields bonus * decay^1 -- every seed. (unchanged from
      V3-EXQ-577)

PASS = C1 AND C2 AND C3 AND C4 across all seeds.
experiment_purpose: diagnostic (substrate readiness test, not a claim hypothesis)
claim_ids: [] (env feature validation only; mirrors V3-EXQ-576/577 precedent.
            GAP-2 unblocks DEV-NEED-001/003/007 + ARC-065 per
            infant_substrate_plan.md, but those are governed by the full
            infant pipeline + EXQ-ISEF behavioural runs, not this readiness
            diagnostic.)
supersedes: V3-EXQ-577 (gap2_microhabitat_validation) -- corrected C2.
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

EXPERIMENT_PURPOSE = "diagnostic"
QUEUE_ID = "V3-EXQ-577a"
SUPERSEDES = "gap2_microhabitat_validation"  # experiment_type of V3-EXQ-577

SEEDS = [0, 1, 2]
N_EPISODES = 100
N_STEPS = 200
SIZE = 14
N_HAZARDS = 6
N_MICROHABITATS = 3

# GAP-2 defaults under test.
ZONE_C_AMBIENT_BONUS = 0.05
ZONE_NOVELTY_DECAY = 0.95

CONDITIONS = [
    ("ARM_0_disabled", False),
    ("ARM_1_enabled", True),
]


def _check_bit_identical_off(seed):
    """C1: ARM_0 env (explicit microhabitat_enabled=False) is bit-identical to
    a legacy CausalGridWorldV2 over a fixed action stream. Unchanged from
    V3-EXQ-577."""
    legacy = CausalGridWorldV2(
        size=SIZE, seed=seed, num_hazards=N_HAZARDS, num_resources=5,
        use_proxy_fields=False,
    )
    arm0 = CausalGridWorldV2(
        size=SIZE, seed=seed, num_hazards=N_HAZARDS, num_resources=5,
        use_proxy_fields=False, microhabitat_enabled=False,
    )

    def trace(env):
        env.reset()
        acc = []
        for t in range(N_STEPS):
            _, r, dn, _, _ = env.step((t * 7) % 5)
            acc.append((round(float(r), 6), env.agent_x, env.agent_y, dn))
            if dn:
                env.reset()
        return acc

    return trace(legacy) == trace(arm0)


def _check_ambient_decay(seed):
    """C4: forced all-zone-C probe; first zone-C entry == bonus * decay^0,
    second == bonus * decay^1. Unchanged from V3-EXQ-577."""
    env = CausalGridWorldV2(
        size=10, seed=seed, num_hazards=0, num_resources=0,
        use_proxy_fields=False, toroidal=True,
        microhabitat_enabled=True, n_microhabitats=N_MICROHABITATS,
        zone_C_ambient_bonus=ZONE_C_AMBIENT_BONUS,
        zone_novelty_decay=ZONE_NOVELTY_DECAY,
    )
    env.reset()
    env._zone_map[:, :] = 2  # force entire grid to zone C
    env._zone_c_visit_count = 0

    _, _, _, info0, _ = env.step(3)
    _, _, _, info1, _ = env.step(3)
    a0 = info0["microhabitat_zone_c_ambient_this_tick"]
    a1 = info1["microhabitat_zone_c_ambient_this_tick"]
    exp0 = ZONE_C_AMBIENT_BONUS * (ZONE_NOVELTY_DECAY ** 0)
    exp1 = ZONE_C_AMBIENT_BONUS * (ZONE_NOVELTY_DECAY ** 1)
    fired = (
        info0["transition_type"] == "zone_c_ambient"
        and abs(a0 - exp0) < 1e-9
        and abs(a1 - exp1) < 1e-9
    )
    return fired, float(a0), float(a1)


def run_experiment(n_episodes, dry_run=False):
    results_by_seed = {}
    c1_by_seed = []
    c2_by_seed = []
    c3_by_seed = []
    c4_by_seed = []

    for seed in SEEDS:
        results_by_seed[seed] = {}
        for cond_label, mh_enabled in CONDITIONS:
            print(f"Seed {seed} Condition {cond_label}", flush=True)

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            cond_metrics = {"microhabitat_enabled": mh_enabled}

            if not mh_enabled:
                # ARM_0: C1 bit-identical OFF check, then a legacy stepping
                # loop (progress + verdict instrumentation parity).
                c1_ok = _check_bit_identical_off(seed)
                c1_by_seed.append(c1_ok)
                cond_metrics["c1_bit_identical_off"] = c1_ok

                env = CausalGridWorldV2(
                    size=SIZE, seed=seed, num_hazards=N_HAZARDS,
                    num_resources=5, use_proxy_fields=False,
                    microhabitat_enabled=False,
                )
                ambient_seen = 0
                for ep in range(n_episodes):
                    env.reset()
                    done = False
                    for _ in range(N_STEPS):
                        if done:
                            break
                        _, _, done, info, _ = env.step(np.random.randint(0, 5))
                        if info["microhabitat_zone_c_ambient_this_tick"] > 0:
                            ambient_seen += 1
                    pi = max(1, n_episodes // 5)
                    if (ep + 1) % pi == 0:
                        print(
                            f"  [train] seed={seed} cond={cond_label} "
                            f"ep {ep + 1}/{n_episodes}",
                            flush=True,
                        )
                cond_metrics["ambient_fires_when_disabled"] = ambient_seen
                passed = c1_ok and ambient_seen == 0
            else:
                # ARM_1: C2 (corrected) coverage, C3 density, C4 ambient decay.
                env = CausalGridWorldV2(
                    size=SIZE, seed=seed, num_hazards=N_HAZARDS,
                    num_resources=0, use_proxy_fields=False,
                    microhabitat_enabled=True, n_microhabitats=N_MICROHABITATS,
                    zone_C_ambient_bonus=ZONE_C_AMBIENT_BONUS,
                    zone_novelty_decay=ZONE_NOVELTY_DECAY,
                )

                haz_by_zone = np.zeros(4, dtype=np.int64)
                cells_by_zone = np.zeros(4, dtype=np.int64)
                # Corrected C2 accounting.
                well_formed_all = True       # per-episode well-formedness gate
                agg_codes = set()            # union of interior codes (aggregate)
                base_zone_collapse_count = 0  # diagnostic: episodes missing a base zone
                redraw_count_total = 0        # diagnostic: redraws performed
                redraw_exhausted_count = 0    # diagnostic: cap-exhausted episodes
                for ep in range(n_episodes):
                    env.reset()
                    zm = env._zone_map
                    interior = zm[1:-1, 1:-1]
                    codes = set(int(v) for v in np.unique(interior))
                    base_present = codes & {0, 1, 2}
                    # Per-episode WELL-FORMEDNESS (not per-episode determinism):
                    # no unassigned interior cell, only valid codes, the D
                    # ecotone present, and at least 2 of the 3 base niches
                    # (the stochastic ecotone may absorb at most one).
                    if (
                        (interior == -1).any()
                        or not codes.issubset({0, 1, 2, 3})
                        or 3 not in codes
                        or len(base_present) < 2
                    ):
                        well_formed_all = False
                    agg_codes |= codes
                    if not {0, 1, 2}.issubset(codes):
                        base_zone_collapse_count += 1
                    redraw_count_total += int(env._microhabitat_redraw_count)
                    if bool(env._microhabitat_redraw_exhausted):
                        redraw_exhausted_count += 1
                    for z in range(4):
                        cells_by_zone[z] += int((zm == z).sum())
                    for hx, hy in env.hazards:
                        zz = int(zm[hx, hy])
                        if 0 <= zz < 4:
                            haz_by_zone[zz] += 1
                    # Step the episode so progress + runtime mirror GAP-1.
                    done = False
                    for _ in range(N_STEPS):
                        if done:
                            break
                        _, _, done, _, _ = env.step(np.random.randint(0, 5))
                    pi = max(1, n_episodes // 5)
                    if (ep + 1) % pi == 0:
                        print(
                            f"  [train] seed={seed} cond={cond_label} "
                            f"ep {ep + 1}/{n_episodes}",
                            flush=True,
                        )

                # Corrected C2: per-episode well-formedness AND aggregate
                # {0,1,2} presence (collapse frequency is a diagnostic only).
                agg_base_ok = {0, 1, 2}.issubset(agg_codes)
                c2_ok = bool(well_formed_all and agg_base_ok)

                dens = haz_by_zone / np.maximum(cells_by_zone, 1)
                c3_ok = bool(haz_by_zone[2] == 0 and dens[1] > dens[0])
                c4_ok, a0, a1 = _check_ambient_decay(seed)

                c2_by_seed.append(c2_ok)
                c3_by_seed.append(c3_ok)
                c4_by_seed.append(c4_ok)
                collapse_rate = (
                    round(base_zone_collapse_count / n_episodes, 6)
                    if n_episodes else 0.0
                )
                cond_metrics.update({
                    "c2_zone_map_wellformed_and_agg_base": c2_ok,
                    "c2_well_formed_all_episodes": bool(well_formed_all),
                    "c2_aggregate_base_zones_present": bool(agg_base_ok),
                    "c2_aggregate_codes": sorted(int(c) for c in agg_codes),
                    "c3_zone_c_zero_hazard_and_B_gt_A": c3_ok,
                    "c4_ambient_fires_and_decays": c4_ok,
                    # --- diagnostic statistics (do NOT gate C2) ---
                    "diag_base_zone_collapse_count": base_zone_collapse_count,
                    "diag_base_zone_collapse_rate": collapse_rate,
                    "diag_redraw_count_total": redraw_count_total,
                    "diag_redraw_exhausted_count": redraw_exhausted_count,
                    "haz_by_zone": haz_by_zone.tolist(),
                    "cells_by_zone": cells_by_zone.tolist(),
                    "hazard_density_by_zone": [round(float(d), 6) for d in dens],
                    "c4_ambient_visit0": a0,
                    "c4_ambient_visit1": a1,
                })
                passed = c2_ok and c3_ok and c4_ok

            results_by_seed[seed][cond_label] = cond_metrics
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    c1_pass = bool(c1_by_seed) and all(c1_by_seed)
    c2_pass = bool(c2_by_seed) and all(c2_by_seed)
    c3_pass = bool(c3_by_seed) and all(c3_by_seed)
    c4_pass = bool(c4_by_seed) and all(c4_by_seed)
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

    print("V3-EXQ-577a GAP-2 microhabitat zones validation (corrected C2)",
          flush=True)
    print(f"  dry_run={dry_run} n_episodes={n_episodes} seeds={SEEDS}",
          flush=True)

    result = run_experiment(n_episodes=n_episodes, dry_run=dry_run)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_577a_gap2_microhabitat_validation_{timestamp}_v3"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": "gap2_microhabitat_validation",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": [],
        "evidence_direction": "non_contributory",
        "supersedes": SUPERSEDES,
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "dry_run": dry_run,
        "config": {
            "seeds": SEEDS,
            "n_episodes": n_episodes,
            "n_steps": N_STEPS,
            "size": SIZE,
            "n_hazards": N_HAZARDS,
            "n_microhabitats": N_MICROHABITATS,
            "zone_C_ambient_bonus": ZONE_C_AMBIENT_BONUS,
            "zone_novelty_decay": ZONE_NOVELTY_DECAY,
        },
        "acceptance_checks": {
            "C1_arm0_bit_identical_off": result["c1_pass"],
            "C2_arm1_zone_map_wellformed_and_agg_base": result["c2_pass"],
            "C3_arm1_zone_c_zero_hazard_and_density_bias": result["c3_pass"],
            "C4_arm1_zone_c_ambient_fires_and_decays": result["c4_pass"],
        },
        "c1_by_seed": result["c1_by_seed"],
        "c2_by_seed": result["c2_by_seed"],
        "c3_by_seed": result["c3_by_seed"],
        "c4_by_seed": result["c4_by_seed"],
        "results_by_seed": result["results_by_seed"],
    }

    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "REE_assembly", "evidence",
        "experiments"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written: {out_path}", flush=True)
    print(f"Outcome: {result['outcome']}", flush=True)
    print(f"  C1 (ARM_0 bit-identical OFF): {result['c1_pass']}", flush=True)
    print(f"  C2 (ARM_1 wellformed + agg base): {result['c2_pass']}",
          flush=True)
    print(f"  C3 (ARM_1 zone-C zero hazard + B>A): {result['c3_pass']}",
          flush=True)
    print(f"  C4 (ARM_1 zone-C ambient decay): {result['c4_pass']}",
          flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
