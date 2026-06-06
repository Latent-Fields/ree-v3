#!/opt/local/bin/python3
"""V3-EXQ-646 -- baseline MINT: modulatory-authority OFF/baseline arm (Phase 0).

experiment_purpose=baseline. claim_ids=[]. Records only -- this experiment SKIPS
nothing and REUSES nothing (arm-reuse is still at Phase 0, instrument-only).

WHAT THIS IS
------------
A baseline pre-mint for the V3-EXQ-643 modulatory-bias-selection-authority
lineage. It runs ONLY the OFF/baseline arm (ARM_A "authority_off_baseline",
use_modulatory_selection_authority=False) across SEEDS=[42,43,44] and records,
per cell, the arm-reuse fingerprint (experiments/_lib/arm_fingerprint.py). The
OFF arm is built from the canonical baseline module
experiments/_lib/baselines/exq643_modulatory_authority_baseline.py so its
identity is content-hashed and a FUTURE 643b/643c that builds its OFF arm from
the same module can (in a later phase) recognise this as the same draw and avoid
re-running the baseline.

WHY (user-directed, arm_reuse_fingerprint_plan.md 7b)
-----------------------------------------------------
643a/643b/etc. re-train a from-scratch OFF baseline that is a deliberate
replication of the previous iteration's OFF arm. On idle cloud workers this mint
records that baseline now so it is ready for reuse later. A stale mint is simply
refused by the fingerprint gate and the arm re-runs normally -- a pre-mint can
never corrupt a result; the only cost is (free) compute on an idle machine.

MACHINE CLASS: queued to ree-cloud-2 (linux-x86_64) so a future cloud-run 643b
can reuse it (Regime A reuses within one machine class only; plan 7b constraint 1
moved 643 to the cloud for this reason).

PHASE 0 GUARANTEE: this script consults NO cache and reuses NO computation. It
adds arm_fingerprint fields to the manifest and nothing else. It cannot skip or
invalidate any experiment.

architecture_epoch: "ree_hybrid_guardrails_v1"

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_646_mint_modulatory_authority_off_baseline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import compute_arm_fingerprint  # noqa: E402
from experiments._lib.baselines.exq643_modulatory_authority_baseline import (  # noqa: E402
    ARM_OFF,
    CANONICAL_BASELINE_ID,
    ENV_KWARGS,
    LINEAGE,
    P0_WARMUP_EPISODES,
    P1_MEASUREMENT_EPISODES,
    SEEDS,
    STEPS_PER_EPISODE,
    off_path_config_slice,
    run_off_cell,
)

EXPERIMENT_TYPE = "v3_exq_646_mint_modulatory_authority_off_baseline"
QUEUE_ID = "V3-EXQ-646"
CLAIM_IDS: List[str] = []          # baseline mint; weights no claim
EXPERIMENT_PURPOSE = "baseline"

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    config_slice = off_path_config_slice(p0=p0, p1=p1, steps=steps)

    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition {ARM_OFF['arm_id']}", flush=True)
        # run_off_cell calls reset_all_rng(seed) at cell entry (complete reset),
        # so rng_fully_reset=True is the truthful assertion the fingerprint needs.
        cell = run_off_cell(seed, p0_episodes=p0, p1_episodes=p1, steps_per_episode=steps)
        cell["arm_fingerprint"] = compute_arm_fingerprint(
            config_slice=config_slice,
            seed=seed,
            script_path=Path(__file__),
            rng_fully_reset=True,
            config_slice_declared=True,
        )
        arm_results.append(cell)
        passed = cell.get("error_note") is None
        print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    n_ok = sum(1 for c in arm_results if c.get("error_note") is None)
    all_ran = n_ok == len(seeds)
    all_reuse_eligible = all(
        bool(c.get("arm_fingerprint", {}).get("reuse_eligible")) for c in arm_results
    )
    outcome = "PASS" if all_ran else "FAIL"

    # Mint-level summary of the emitted fingerprints (observability only).
    fps = [c.get("arm_fingerprint", {}) for c in arm_results]
    substrate_hashes = sorted({fp.get("substrate_hash") for fp in fps if fp})
    machine_classes = sorted({fp.get("machine_class") for fp in fps if fp})

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    interpretation = {
        "label": "baseline_minted" if all_ran else "baseline_mint_incomplete",
        "preconditions": [
            {
                "name": "all_off_cells_completed",
                "description": (
                    "every OFF/baseline cell ran to completion without a non-finite "
                    "action (error_note is None) so the recorded baseline is usable"
                ),
                "measured": int(n_ok),
                "threshold": int(len(seeds)),
                "met": bool(all_ran),
            },
        ],
        "criteria_non_degenerate": {
            "all_cells_ran": bool(all_ran),
            "all_cells_reuse_eligible": bool(all_reuse_eligible),
        },
    }

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "result": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Baseline MINT (Phase 0, arm-reuse instrument-only). Runs ONLY the "
            "OFF/baseline arm of the V3-EXQ-643 modulatory-authority lineage "
            "(ARM_A authority_off_baseline, use_modulatory_selection_authority=False) "
            "across SEEDS=[42,43,44], built from the canonical baseline module "
            "experiments/_lib/baselines/exq643_modulatory_authority_baseline.py, and "
            "records each cell's arm_fingerprint. It SKIPS nothing and REUSES nothing "
            "-- it only records a content-addressed fingerprint so a later phase can "
            "recognise a re-run of this baseline. claim_ids=[]; excluded from "
            "governance confidence/conflict scoring."
        ),
        "interpretation": interpretation,
        "dry_run": bool(dry_run),
        # Mint provenance / reuse metadata.
        "baseline_id": CANONICAL_BASELINE_ID,
        "lineage": LINEAGE,
        "canonical_baseline_module": (
            "experiments/_lib/baselines/exq643_modulatory_authority_baseline.py"
        ),
        "arm_id_minted": ARM_OFF["arm_id"],
        "arm_label_minted": ARM_OFF["label"],
        "reuse_phase": 0,
        "substrate_hashes": substrate_hashes,
        "machine_classes": machine_classes,
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": ENV_KWARGS,
            "off_path_config_slice": config_slice,
        },
        "summary": {
            "n_cells": len(arm_results),
            "n_cells_ok": int(n_ok),
            "all_cells_ran": bool(all_ran),
            "all_cells_reuse_eligible": bool(all_reuse_eligible),
        },
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}  label: {interpretation['label']}", flush=True)
    print(f"  n_cells_ok: {n_ok}/{len(seeds)}", flush=True)
    print(f"  all_cells_reuse_eligible: {all_reuse_eligible}", flush=True)
    if substrate_hashes:
        print(f"  substrate_hash: {substrate_hashes[0][:16]}...", flush=True)
    if machine_classes:
        print(f"  machine_class: {machine_classes[0]}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-646 modulatory-authority OFF/baseline mint (Phase 0)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
    sys.exit(0)
