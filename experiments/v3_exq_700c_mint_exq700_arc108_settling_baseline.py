#!/opt/local/bin/python3
"""
V3-EXQ-700c-mint -- low-priority cloud BASELINE MINT for the ARC-108 sec-7 settling lineage.

Mints the four REUSABLE arms (A0_ENVELOPE_ONLY / A2_SETTLING_SIGNED / A3_BOTH_SIGNED /
C3_SETTLING_UNSIGNED) x the 6 seeds of the V3-EXQ-700c harness so a LATER iteration (a
700d / a sibling falsifier) can arm-reuse them instead of re-training. The fifth arm
(ARM_NOISE, the same-layer field-noise null) is the CHANGED arm in each lettered
iteration and is NEVER reusable, so it is excluded here.

BYTE-IDENTICAL COMPUTATION
--------------------------
This script imports `_run_seed_arm`, `ARMS`, `SEEDS`, the schedule, and the fingerprint
config-slice helper DIRECTLY from the V3-EXQ-700c consumer module, so each minted cell is
byte-identical to the cell the consumer would run fresh. The per-cell fingerprint is
emitted with `include_driver_script_in_hash=False` (MANDATORY) so the consumer's DIFFERENT
driver script still produces the SAME fingerprint and `arm_reuse.try_reuse_cell` can HIT.
Both sides build their fingerprint config_slice from the ONE shared
`arm_config_slice(...)` in experiments/_lib/baselines/exq700_arc108_settling_baseline.py,
so the slices match by construction.

This is a MINT (experiment_purpose="baseline"; claim_ids=[]): it has NO acceptance
criteria. The outcome is PASS iff all reusable cells ran to completion (no error_note),
FAIL otherwise. run_id ends _v3; architecture_epoch is "ree_hybrid_guardrails_v1".

Machine class: the mint should run on the same machine class as the future consumer
(linux-x86_64-py3.10; ree-cloud-4). A Mac-minted baseline cannot be reused by a cloud
consumer (machine_class enters the fingerprint) and vice-versa.

See experiments/v3_exq_700c_arc108_sec7_learned_gating_settling_samelayer_null.py (the
consumer harness this mints for), experiments/_lib/baselines/exq700_arc108_settling_baseline.py
(the shared config-slice helper), REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import compute_arm_fingerprint, reset_all_rng
from experiments._lib.baselines.exq700_arc108_settling_baseline import (
    REUSABLE_ARM_IDS,
    arm_config_slice,
)

# Byte-identical computation: import the harness + config from the 700c consumer.
from experiments.v3_exq_700c_arc108_sec7_learned_gating_settling_samelayer_null import (
    ARMS,
    SEEDS,
    P0_WARMUP_EPISODES,
    P1_BIAS_TRAIN_EPISODES,
    P2_MEASUREMENT_EPISODES,
    STEPS_PER_EPISODE,
    DRY_RUN_SEEDS,
    DRY_RUN_P0,
    DRY_RUN_P1,
    DRY_RUN_P2,
    DRY_RUN_STEPS,
    _run_seed_arm,
)

EXPERIMENT_TYPE = "v3_exq_700c_mint_exq700_arc108_settling_baseline"
QUEUE_ID = "V3-EXQ-700c-m"   # schema-valid (V<gen>-EXQ-<n><letter>(-<letter>)); the V3-EXQ-700c baseline mint
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "baseline"

# Only the four reusable arms are minted (ARM_NOISE is the changed arm; never reused).
_REUSABLE_ARMS: List[Dict[str, Any]] = [
    arm for arm in ARMS if arm["arm_id"] in REUSABLE_ARM_IDS
]


def run_mint(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arm_results: List[Dict[str, Any]] = []
    script_path = Path(__file__).resolve()
    n_cells = 0
    n_cells_ok = 0

    for arm in _REUSABLE_ARMS:
        print(
            f"Mint arm {arm['arm_id']} ({arm['label']}) lcg_on={arm['lcg_on']} "
            f"settle_on={arm['settle_on']} rpe_mode={arm.get('rpe_mode', 'signed')} "
            f"(P0={p0_episodes} ep e2-train, P1={p1_episodes} ep bias-train, "
            f"P2={p2_episodes} ep measure, steps_per_episode={steps_per_episode}, "
            f"dry_run={dry_run})",
            flush=True,
        )
        for s in seeds:
            print(f"Mint seed {s} Condition {arm['label']}", flush=True)
            # Complete per-cell RNG reset at cell entry (order-independence; the cell is a
            # pure function of seed so the Phase-0 fingerprint is reuse_eligible). This is
            # idempotent with the reset_all_rng(seed) _run_seed_arm performs as its first
            # statement -- a double reset to the same seed is a no-op, so the minted cell is
            # byte-identical to the consumer's fresh-run cell.
            reset_all_rng(s)
            # Byte-identical to the consumer's fresh-run cell.
            row = _run_seed_arm(
                arm, s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
            )
            # MANDATORY include_driver_script_in_hash=False so the consumer's different
            # driver still matches; config_slice from the SHARED arm_config_slice helper.
            row["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice=arm_config_slice(
                    arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                ),
                seed=s,
                script_path=script_path,
                rng_fully_reset=True,
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            )
            arm_results.append(row)
            n_cells += 1
            if row["error_note"] is None:
                n_cells_ok += 1
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    all_ran = bool(n_cells > 0 and n_cells_ok == n_cells)
    outcome = "PASS" if all_ran else "FAIL"

    return {
        "outcome": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "non_degenerate": "N/A (baseline mint; no acceptance criteria)",
        "seeds": seeds,
        "reusable_arm_ids": list(REUSABLE_ARM_IDS),
        "n_arms_minted": len(_REUSABLE_ARMS),
        "n_cells": int(n_cells),
        "n_cells_ok": int(n_cells_ok),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "arm_results": arm_results,
    }


def _build_manifest(
    result: Dict[str, Any],
    timestamp_utc: str,
    dry_run: bool,
) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": "non_contributory",
        "non_degenerate": result["non_degenerate"],
        "mint_note": (
            "Low-priority cloud BASELINE MINT for the ARC-108 sec-7 settling lineage "
            "(V3-EXQ-700c reusable arms A0/A2/A3/C3 x 6 seeds). Byte-identical computation "
            "to the V3-EXQ-700c consumer (imports _run_seed_arm + ARMS + schedule from it); "
            "per-cell fingerprint emitted with include_driver_script_in_hash=False so a later "
            "iteration with a different driver can arm-reuse these cells. ARM_NOISE (the "
            "same-layer field-noise null) is the changed arm and is NOT minted. PROMOTES "
            "NOTHING (experiment_purpose=baseline; claim_ids=[]). The parent records this "
            "run_id into the lineage baseline module for citation via REUSE_BASELINE_FROM."
        ),
        "dry_run": bool(dry_run),
        "config_summary": {
            "design": "baseline mint of the 4 reusable arms (A0/A2/A3/C3) x 6 seeds of V3-EXQ-700c",
            "reusable_arm_ids": list(REUSABLE_ARM_IDS),
            "excluded_arm": "ARM_NOISE (same-layer field-noise null; the changed arm, never reused)",
            "byte_identical_to": "experiments/v3_exq_700c_arc108_sec7_learned_gating_settling_samelayer_null.py",
            "fingerprint_include_driver_script_in_hash": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-700c-mint baseline mint for the ARC-108 sec-7 settling lineage"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        p2 = DRY_RUN_P2
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1 = P1_BIAS_TRAIN_EPISODES
        p2 = P2_MEASUREMENT_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_mint(
        seeds=seeds,
        p0_episodes=p0,
        p1_episodes=p1,
        p2_episodes=p2,
        steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    if args.dry_run:
        out_path = out_dir / f"_dry_{manifest['run_id']}.json"

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"cells_ok={result['n_cells_ok']}/{result['n_cells']} "
        f"arms={result['n_arms_minted']}",
        flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
    sys.exit(0)
