#!/opt/local/bin/python3
"""V3-EXQ-610 OFF/baseline arm MINT (arm-reuse Phase 0, INV-074 lineage).

Runs ONLY the OFF/baseline arm of the V3-EXQ-610 (INV-074 crystallization
necessity) lineage -- ARM_0_stripped_control (crystallize=False, phase-3
entropy floor 0.0, MECH-313/341/260 floors OFF) -- across the lineage's matched
SEEDS, and records each (seed) cell's reuse fingerprint.

EXPERIMENT_PURPOSE = "baseline"   (records a reference measurement; excluded
from governance confidence/conflict scoring). claim_ids = [] -- a single OFF
arm cannot test INV-074 (no treatment arm); this run only MINTS the OFF baseline
so a future 610g can reuse it instead of re-training it, AND doubles as the
Phase-0 cross-instance determinism check.

WHY (arm-reuse plan 7b)
-----------------------
The 610 lineage's OFF baseline is a deliberate replication that every 610x
re-trains from scratch. This mint computes it once on each of two separate
CX22 cloud instances (ree-cloud-2 and ree-cloud-3 -- the CORRECT machine-class:
610b/c/d ran on cloud-2, 610e/f on cloud-1, all linux-x86_64, so cloud is the
reuse class for a future cloud 610g). The OFF arm is built from the canonical
baseline module experiments/_lib/baselines/exq610_inv074_crystallization_baseline.py
-- the content-hashed contract a future 610g MUST build its OFF arm from. The
fingerprint's substrate_hash content-hashes experiments/_lib/**/*.py (the
baseline module included), so any drift in the baseline refuses a stale reuse
(safe: a refused reuse just re-runs the arm).

Queued TWICE (same script + config, machine_affinity ree-cloud-2 and
ree-cloud-3), LOW priority so real science preempts. After both run, the two
manifests' OFF metrics are compared: agreement within tolerance confirms
separate cloud instances are mutually deterministic on CPU torch (the
assumption Regime A reuse rests on -- Phase 0's "zero false-collision" exit
criterion). The arm_fingerprint hash is identical across the two instances by
construction (coarse machine_class = linux-x86_64-pyX.Y); the determinism
check is on the recorded METRICS.

PHASE 0 EMIT-ONLY: nothing here consults a cache, skips, or reuses any
computation. It records arm_results[i]["arm_fingerprint"] only.

Per-cell RNG reset: reset_all_rng(seed) at cell entry (closes the order-
dependence hazard so the cell is a pure function of (substrate, config, seed)
and is reuse_eligible). build_off_arm also re-seeds torch/numpy/random; the
explicit reset_all_rng ALSO reseeds the harness module-level _action_random.

Estimated runtime: ~1 OFF arm x 3 seeds x ~2500 ep @ 200 steps. 610e measured
~142 min/run on cloud-1; ~3 runs ~= 7-8h on a CX22.

Design plan: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md
Helper: experiments/_lib/arm_fingerprint.py
Canonical baseline: experiments/_lib/baselines/exq610_inv074_crystallization_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_protocol import emit_outcome
from _lib.arm_fingerprint import reset_all_rng, compute_arm_fingerprint
from _lib.baselines.exq610_inv074_crystallization_baseline import (
    BASELINE_ID,
    SOURCE_LINEAGE,
    SOURCE_SCRIPT,
    SEEDS,
    MAX_EPISODES,
    STEPS_PER_EPISODE,
    OFF_ARM,
    build_off_arm,
    train_off_arm,
    off_path_config_slice,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_610_inv074_crystallization_baseline_mint"
EXPERIMENT_PURPOSE = "baseline"
CLAIM_IDS: List[str] = []   # single OFF arm cannot test a claim; mint-only.


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_off_cell(seed: int, dry_run: bool) -> Dict:
    """Run one OFF/baseline cell and attach its arm_fingerprint."""
    # Runner progress: seed/condition boundary line + verdict line per cell.
    print(f"Seed {seed} Condition {OFF_ARM['label']}", flush=True)
    print(
        f"[V3-EXQ-610-MINT] OFF cell seed={seed} (crystallize=False, p3_eb=0.0, "
        f"floors OFF; baseline_id={BASELINE_ID})",
        flush=True,
    )

    # Complete per-cell RNG reset BEFORE building the agent/env so the cell is a
    # pure function of (substrate, config_slice, seed) -> reuse_eligible.
    reset_all_rng(seed)

    agent, env, scheduler = build_off_arm(seed=seed)
    metrics = train_off_arm(agent=agent, scheduler=scheduler, seed=seed, dry_run=dry_run)

    # Phase-0 emit-only fingerprint (records; never consults a cache).
    metrics["arm_fingerprint"] = compute_arm_fingerprint(
        config_slice=off_path_config_slice(),
        seed=seed,
        script_path=Path(__file__),
        rng_fully_reset=True,
        config_slice_declared=True,
    )

    print(f"verdict: PASS", flush=True)
    return metrics


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv

    qid = os.environ.get("REE_QUEUE_ID")  # set by runner; differs per cloud entry
    print(
        f"[V3-EXQ-610-MINT] OFF/baseline arm mint for {SOURCE_LINEAGE} "
        f"(source arm: {SOURCE_SCRIPT} ARM_0_stripped_control); queue_id={qid}",
        flush=True,
    )

    seeds = SEEDS if not dry else [42]

    results: List[Dict] = []
    for seed in seeds:
        results.append(_run_off_cell(seed, dry_run=dry))

    # Cross-instance determinism check is performed LATER by comparing the
    # ree-cloud-2 and ree-cloud-3 manifests' OFF metrics. This run records them.
    print("")
    print("[V3-EXQ-610-MINT] OFF cell metrics (compared across cloud-2/3 manifests):")
    for r in results:
        p2 = r["end_phase_2_entropy"]
        p3 = r["end_phase_3_entropy"]
        p2s = f"{p2:.6f}" if p2 is not None else "None"
        p3s = f"{p3:.6f}" if p3 is not None else "None"
        fp = r["arm_fingerprint"]
        print(
            f"  seed={r['seed']} end_p2_entropy={p2s} end_p3_entropy={p3s} "
            f"mean_reward={r['mean_reward']:.6f} final_phase={r['final_phase']} "
            f"reuse_eligible={fp['reuse_eligible']} fp={fp['arm_fingerprint'][:12]} "
            f"machine_class={fp['machine_class']}",
            flush=True,
        )

    n_complete = sum(
        1 for r in results
        if r.get("end_phase_3_entropy") is not None and r.get("total_episodes", 0) >= (5 if dry else MAX_EPISODES)
    )
    fingerprints_emitted = all("arm_fingerprint" in r for r in results)
    all_reuse_eligible = all(r["arm_fingerprint"]["reuse_eligible"] for r in results)

    if dry:
        print(
            f"[V3-EXQ-610-MINT] dry-run complete ({len(results)} cell(s)); "
            f"fingerprints_emitted={fingerprints_emitted} "
            f"all_reuse_eligible={all_reuse_eligible}. No manifest written.",
            flush=True,
        )
        sys.exit(0)

    # ---- Manifest (real run only) ----
    ts = _utc_compact_now()
    qid_clean = (qid or "manual").replace("-", "").lower()
    run_id = f"{EXPERIMENT_TYPE}_{qid_clean}_{ts}_v3"
    evidence_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    manifest_path = evidence_dir / f"{run_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Baseline adjudication block (Phase-0 mint: records only; drives no
    # governance verdict; reads no learned quantity for a self-route).
    interpretation = {
        "label": "off_baseline_minted_phase0_emit_only",
        "preconditions": [
            {
                "name": "phase0_emit_only_no_reuse_executed",
                "description": "mint records arm_fingerprint + OFF metrics only; no cache consulted, no reuse executed (Phase 0)",
                "measured": 1,
                "threshold": 1,
                "met": True,
            },
            {
                "name": "all_off_cells_completed",
                "description": "every OFF (seed) cell ran to MAX_EPISODES and produced a finite end_phase_3 entropy",
                "measured": n_complete,
                "threshold": len(seeds),
                "met": n_complete >= len(seeds),
            },
        ],
        "criteria_non_degenerate": {
            "off_cells_ran": n_complete >= len(seeds),
            "fingerprints_emitted": fingerprints_emitted,
            "all_reuse_eligible": all_reuse_eligible,
        },
    }

    outcome = "PASS" if (fingerprints_emitted and n_complete >= len(seeds)) else "FAIL"

    manifest = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "queue_id": qid,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "completed_at": _utc_iso_now(),
        "timestamp_utc": ts,
        "baseline_id": BASELINE_ID,
        "source_lineage": SOURCE_LINEAGE,
        "source_script": SOURCE_SCRIPT,
        "interpretation": interpretation,
        "arm_results": results,
        "config": {
            "seeds": SEEDS,
            "max_episodes": MAX_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "off_arm": dict(OFF_ARM),
            "off_path_config_slice": off_path_config_slice(),
            "note": (
                "Phase-0 arm-reuse mint of the V3-EXQ-610 OFF/baseline arm. "
                "Cross-instance determinism check: compare this manifest's "
                "arm_results end_phase_*_entropy + mean_reward against the "
                "sibling cloud manifest (ree-cloud-2 vs ree-cloud-3) within "
                "tolerance. Record the result in arm_reuse_fingerprint_plan.md 7b."
            ),
        },
    }

    manifest_path = write_flat_manifest(
        manifest,
        evidence_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )

    print(f"[V3-EXQ-610-MINT] Manifest written: {manifest_path}", flush=True)

    emit_outcome(
        outcome=outcome,
        manifest_path=str(manifest_path),
        run_id=run_id,
        # queue_id intentionally omitted -> emit_outcome reads REE_QUEUE_ID so
        # the cloud-2 and cloud-3 entries each write their own sentinel.
    )
