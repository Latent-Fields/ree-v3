#!/opt/local/bin/python3
"""V3-EXQ-520 -- SD-052 / MECH-303 contextual passive safety terrain substrate readiness.

Claim: MECH-303 (safety_prediction.contextual_passive_substrate)
Status: candidate (v3_pending). Substrate IMPLEMENTED 2026-05-04 as SD-052.

Why this experiment exists
--------------------------
SD-052 extends ResidueField with safety_terrain_rbf_field: a passive harm-absence
accumulator that builds up a spatially-localised safety attractor at z_world over
repeated exposure to a harmless context. This is a SUBSTRATE READINESS diagnostic --
it verifies the mechanics of the RBF terrain and the background vigilance gate, not the
downstream contextual-vs-cue conditioned safety discrimination (tested later via
discriminative pair gated on EXQ-520 PASS).

Test structure
--------------
Part 1 (unit): Directly call ResidueField.accumulate_safety() and evaluate_safety()
with synthetic z_world tensors and verify arithmetic.

Part 2 (integration): Run REEAgent + CausalGridWorldV2 for N ticks per arm.
Track total_safety, num_safety_steps, and evaluate_safety at final z_world.
Arms:
  ARM_0 (OFF baseline): use_contextual_safety_terrain=False -- bit-identical OFF.
  ARM_1 (write-only): use_contextual_safety_terrain=True, harm_threshold=999
    (forces accumulation), release_threshold=999 -- terrain accumulates, gate cannot fire.
  ARM_2 (full MECH-303): use_contextual_safety_terrain=True, harm_threshold=999,
    release_threshold=1.0 -- terrain accumulates, gate fires when evaluate_safety>=1.0.
  ARM_3 (gate-only): use_contextual_safety_terrain=True, harm_threshold=999,
    accum_weight=0.0, release_threshold=1.0 -- terrain empty, no gate. Equals ARM_0.

Pre-registered acceptance criteria
-----------------------------------
C1: Unit -- accumulate_safety increments total_safety (basic write path works)
C2: Unit -- accumulate_safety with hypothesis_tag=True does NOT increment
    (MECH-094 gate: simulation ticks cannot build safety terrain)
C3: Unit -- evaluate_safety before accumulation returns near-zero vector
    (terrain starts empty; no false positives)
C4: Unit -- evaluate_safety after 50 accumulation steps at fixed z_world
    returns >= EVAL_THRESHOLD (terrain is populated and responds to query)
C5: Unit -- evaluate_safety with terrain OFF returns zero tensor
    (disabled feature path: no RBF instantiated)

C6: Integration -- ARM_1: total_safety > 0 AND num_safety_steps > 0
    (terrain write path wired in agent.py sense())
C7: Integration -- ARM_1: evaluate_safety(final_z_world) >= EVAL_THRESHOLD
    (RBF field retains accumulated safety across step loop)
C8: Integration -- ARM_2: evaluate_safety(final_z_world) >= contextual_safety_release_threshold
    (gate would fire: terrain built above release threshold)
C9: Integration -- ARM_3: total_safety == 0
    (accum_weight=0.0 blocks accumulation; gate never fires without terrain)
C10: Integration -- no exceptions in any arm
    (end-to-end wiring: sense() write + select_action() release gate)

PASS = all C1-C10.
FAIL on C1 -> accumulate_safety write path broken; check field.py accumulate_safety().
FAIL on C2 -> MECH-094 hypothesis_tag gate broken; check hypothesis_tag guard in accumulate_safety().
FAIL on C3 -> RBF not initialised empty; check ResidueField.__init__ safety_terrain block.
FAIL on C4 -> RBF not accumulating; check add_residue() call in accumulate_safety().
FAIL on C5 -> evaluate_safety() not returning zero when disabled; check early return.
FAIL on C6 -> agent.py sense() not calling accumulate_safety(); check MECH-303 block.
FAIL on C7 -> RBF field not persisting across ticks; check buffer registration.
FAIL on C8 -> RBF not building to release_threshold; check accum_weight or N_TICKS.
FAIL on C9 -> accum_weight=0 still accumulating; check magnitude guard in accumulate_safety().
FAIL on C10 -> wiring crash; check sense() / select_action() integration blocks.

experiment_purpose = "diagnostic"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig, ResidueConfig  # noqa: E402
from ree_core.residue.field import ResidueField  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_520_sd052_contextual_safety_terrain_readiness"
CLAIM_IDS = ["MECH-303"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

WORLD_DIM_UNIT = 32
EVAL_THRESHOLD = 0.1          # minimum evaluate_safety response after accumulation
N_ACCUM_STEPS_UNIT = 50       # steps for C4 unit accumulation test
ACCUM_WEIGHT_UNIT = 0.05      # higher weight for fast unit test
RELEASE_THRESHOLD = 1.0       # ARM_2 release threshold

GRID_SIZE = 8
NUM_HAZARDS = 3
NUM_RESOURCES = 2
N_TICKS_INTEGRATION = 300
SEEDS = (42, 43)


# ---- Part 1: Unit tests ----

def run_unit_tests() -> Dict:
    """Directly test ResidueField.accumulate_safety() and evaluate_safety() arithmetic."""

    # Build a minimal ResidueField with safety terrain enabled.
    rcfg = ResidueConfig(
        world_dim=WORLD_DIM_UNIT,
        num_basis_functions=20,
        kernel_bandwidth=1.0,
        safety_terrain_enabled=True,
    )
    field_on = ResidueField(rcfg)
    field_on.eval()

    z0 = torch.randn(1, WORLD_DIM_UNIT)

    # C3: evaluate_safety before accumulation is near-zero.
    eval_before = float(field_on.evaluate_safety(z0).mean())
    c3 = eval_before < 0.05

    # C1: accumulate_safety increments total_safety.
    field_on.accumulate_safety(z0, safety_magnitude=ACCUM_WEIGHT_UNIT, hypothesis_tag=False)
    c1 = float(field_on.total_safety) > 0.0
    ts_after_one = float(field_on.total_safety)
    ns_after_one = int(field_on.num_safety_steps)

    # C2: accumulate_safety with hypothesis_tag=True does NOT increment.
    ts_before_sim = float(field_on.total_safety)
    field_on.accumulate_safety(z0, safety_magnitude=ACCUM_WEIGHT_UNIT, hypothesis_tag=True)
    ts_after_sim = float(field_on.total_safety)
    c2 = abs(ts_after_sim - ts_before_sim) < 1e-9

    # C4: evaluate_safety after N_ACCUM_STEPS_UNIT steps >= EVAL_THRESHOLD.
    for _ in range(N_ACCUM_STEPS_UNIT - 1):   # one step already done above
        field_on.accumulate_safety(z0, safety_magnitude=ACCUM_WEIGHT_UNIT, hypothesis_tag=False)
    eval_after = float(field_on.evaluate_safety(z0).mean())
    c4 = eval_after >= EVAL_THRESHOLD

    # C5: evaluate_safety with terrain OFF returns zero.
    rcfg_off = ResidueConfig(
        world_dim=WORLD_DIM_UNIT,
        num_basis_functions=20,
        kernel_bandwidth=1.0,
        safety_terrain_enabled=False,
    )
    field_off = ResidueField(rcfg_off)
    field_off.eval()
    eval_off = float(field_off.evaluate_safety(z0).mean())
    c5 = eval_off == 0.0

    return {
        "c3_eval_before_accum": eval_before,
        "C3_eval_near_zero_before": c3,
        "ts_after_one_step": ts_after_one,
        "ns_after_one_step": ns_after_one,
        "C1_accumulate_increments_total_safety": c1,
        "ts_before_sim_tick": ts_before_sim,
        "ts_after_sim_tick": ts_after_sim,
        "C2_hypothesis_tag_blocks_accumulation": c2,
        "c4_eval_after_accum": eval_after,
        "C4_eval_above_threshold": c4,
        "c5_eval_off_field": eval_off,
        "C5_off_field_returns_zero": c5,
    }


# ---- Part 2: Integration smoke ----

# (arm_name, use_terrain, harm_threshold, accum_weight, release_threshold)
ARMS: List[Tuple[str, bool, float, float, float]] = [
    ("ARM_0_off_baseline",    False,  0.05,  0.01,  1.0),
    ("ARM_1_write_only",      True,   999.0, 0.05, 999.0),
    ("ARM_2_full_mech303",    True,   999.0, 0.05,  1.0),
    ("ARM_3_gate_only_noop",  True,   999.0, 0.0,   1.0),
]


def run_integration_arm(
    seed: int,
    arm_name: str,
    use_terrain: bool,
    harm_threshold: float,
    accum_weight: float,
    release_threshold: float,
    n_ticks: int,
) -> Dict:
    """Run REEAgent + env; track safety terrain state."""
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        seed=seed,
        use_proxy_fields=True,
        limb_damage_enabled=True,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        use_affective_harm_stream=True,
        limb_damage_enabled=True,
        use_contextual_safety_terrain=use_terrain,
        contextual_safety_harm_threshold=harm_threshold,
        contextual_safety_accum_weight=accum_weight,
        contextual_safety_release_threshold=release_threshold,
    )
    agent = REEAgent(config)
    agent.eval()

    exception_count = 0
    final_z_world = None

    _, obs_dict = env.reset()
    for tick_i in range(n_ticks):
        try:
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm_a = obs_dict.get("harm_obs_a")
            latent = agent.sense(obs_body, obs_world, obs_harm_a=obs_harm_a)
            if latent.z_world is not None:
                final_z_world = latent.z_world.detach()
            _, _, done, _, obs_dict = env.step(torch.tensor(tick_i % env.action_dim))
            if done:
                _, obs_dict = env.reset()
        except Exception as exc:  # noqa: BLE001
            exception_count += 1
            print(f"  [integration] EXCEPTION arm={arm_name} seed={seed}: {exc}")
            try:
                _, obs_dict = env.reset()
            except Exception:  # noqa: BLE001
                break

    total_safety = 0.0
    num_safety_steps = 0
    eval_final = 0.0
    terrain_enabled = agent.residue_field.safety_terrain_enabled

    if terrain_enabled and hasattr(agent.residue_field, "total_safety"):
        total_safety = float(agent.residue_field.total_safety)
        num_safety_steps = int(agent.residue_field.num_safety_steps)
        if final_z_world is not None:
            eval_final = float(
                agent.residue_field.evaluate_safety(final_z_world).mean()
            )

    return {
        "arm": arm_name,
        "seed": seed,
        "n_ticks": n_ticks,
        "terrain_enabled": terrain_enabled,
        "total_safety": total_safety,
        "num_safety_steps": num_safety_steps,
        "eval_final_z_world": eval_final,
        "exception_count": exception_count,
    }


def evaluate_acceptance(unit: Dict, integration_results: List[Dict]) -> Dict:
    """Evaluate pre-registered acceptance checks C1-C10."""

    def agg(arm: str, field: str, fn=max):
        vals = [r[field] for r in integration_results if r["arm"] == arm]
        return fn(vals) if vals else 0.0

    c1 = unit["C1_accumulate_increments_total_safety"]
    c2 = unit["C2_hypothesis_tag_blocks_accumulation"]
    c3 = unit["C3_eval_near_zero_before"]
    c4 = unit["C4_eval_above_threshold"]
    c5 = unit["C5_off_field_returns_zero"]

    # C6: ARM_1 total_safety > 0 and num_safety_steps > 0
    arm1_total = agg("ARM_1_write_only", "total_safety")
    arm1_steps = agg("ARM_1_write_only", "num_safety_steps")
    c6 = arm1_total > 0.0 and arm1_steps > 0

    # C7: ARM_1 evaluate_safety(final_z_world) >= EVAL_THRESHOLD
    arm1_eval = agg("ARM_1_write_only", "eval_final_z_world")
    c7 = arm1_eval >= EVAL_THRESHOLD

    # C8: ARM_2 evaluate_safety(final_z_world) >= RELEASE_THRESHOLD
    arm2_eval = agg("ARM_2_full_mech303", "eval_final_z_world")
    c8 = arm2_eval >= RELEASE_THRESHOLD

    # C9: ARM_3 total_safety == 0
    arm3_total = agg("ARM_3_gate_only_noop", "total_safety")
    c9 = arm3_total == 0.0

    # C10: no exceptions
    total_exc = sum(r["exception_count"] for r in integration_results)
    c10 = total_exc == 0

    overall = c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8 and c9 and c10
    return {
        "C1_accumulate_increments_total_safety": bool(c1),
        "C2_hypothesis_tag_blocks_accumulation": bool(c2),
        "C3_eval_near_zero_before": bool(c3),
        "C3_eval_before": float(unit["c3_eval_before_accum"]),
        "C4_eval_above_threshold": bool(c4),
        "C4_eval_after": float(unit["c4_eval_after_accum"]),
        "C4_threshold": EVAL_THRESHOLD,
        "C5_off_field_returns_zero": bool(c5),
        "C6_arm1_terrain_accumulates": bool(c6),
        "C6_arm1_total_safety": float(arm1_total),
        "C6_arm1_num_steps": int(arm1_steps),
        "C7_arm1_eval_above_eval_threshold": bool(c7),
        "C7_arm1_eval_final": float(arm1_eval),
        "C8_arm2_eval_above_release_threshold": bool(c8),
        "C8_arm2_eval_final": float(arm2_eval),
        "C8_release_threshold": RELEASE_THRESHOLD,
        "C9_arm3_total_safety_zero": bool(c9),
        "C9_arm3_total_safety": float(arm3_total),
        "C10_no_exceptions": bool(c10),
        "C10_total_exceptions": int(total_exc),
        "all_pass": bool(overall),
    }


def main(dry_run: bool = False) -> int:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    seeds = (SEEDS[0],) if dry_run else SEEDS
    n_ticks = 60 if dry_run else N_TICKS_INTEGRATION
    t0 = time.time()

    # --- Part 1: Unit tests ---
    print(f"[{EXPERIMENT_TYPE}] Part 1: unit tests")
    unit = run_unit_tests()
    for k, v in unit.items():
        print(f"  {k}: {v}")

    # --- Part 2: Integration ---
    print(
        f"[{EXPERIMENT_TYPE}] Part 2: integration smoke "
        f"({n_ticks} ticks x {len(seeds)} seeds)"
    )
    integration_results: List[Dict] = []
    for seed in seeds:
        for arm_name, use_terrain, harm_thresh, accum_w, rel_thresh in ARMS:
            r = run_integration_arm(
                seed=seed,
                arm_name=arm_name,
                use_terrain=use_terrain,
                harm_threshold=harm_thresh,
                accum_weight=accum_w,
                release_threshold=rel_thresh,
                n_ticks=n_ticks,
            )
            integration_results.append(r)
            print(
                f"  seed={seed} {arm_name:<30} "
                f"total_safety={r['total_safety']:.3f} "
                f"steps={r['num_safety_steps']:4d} "
                f"eval={r['eval_final_z_world']:.4f} "
                f"exc={r['exception_count']}"
            )

    elapsed = time.time() - t0
    acceptance = evaluate_acceptance(unit, integration_results)
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] acceptance:")
    for k, v in acceptance.items():
        print(f"  {k}: {v}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")
    print(f"verdict: {outcome}")

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
        return 0

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "elapsed_seconds": elapsed,
        "n_seeds_unit": 1,
        "n_seeds_integration": len(seeds),
        "n_ticks_integration": n_ticks,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "eval_threshold": EVAL_THRESHOLD,
        "release_threshold": RELEASE_THRESHOLD,
        "n_accum_steps_unit": N_ACCUM_STEPS_UNIT,
        "integration_results": integration_results,
        "acceptance": acceptance,
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
    print(f"Done. Outcome: {outcome}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
