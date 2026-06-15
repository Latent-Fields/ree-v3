#!/opt/local/bin/python3
"""V3-EXQ-685 -- first AUTOMATED arm-reuse index-HIT in the wild (infrastructure demo).

EXPERIMENT_PURPOSE = "baseline"   (infrastructure demonstration; records a
reference measurement only -- excluded from governance confidence/conflict
scoring). claim_ids = [] -- this run tests NO scientific claim; it exercises the
arm-reuse machinery end-to-end.

WHY (arm_reuse_fingerprint_plan.md :: arm_reuse_fingerprint:P1-auto)
-------------------------------------------------------------------
The Phase-1 arm-reuse consumer (try_reuse_cell over arm_fingerprint_index.json)
is BUILT + unit-tested (P1-build / P1-fix), the cross-instance determinism gate
PASSED + was user-ratified (GATE), and an explicit-cite reuse ran live (P1-cite,
V3-EXQ-647). The single remaining node -- P1-auto -- is the first AUTOMATED
index-HIT in the wild: a consumer that RE-MINTS its OFF baseline AND consumes it
via try_reuse_cell(include_driver_script_in_hash=False), confirming reused_from_run_id
appears in-manifest AND that flipping one config byte flips back to a fresh run.

P1-auto was parked on "the next genuinely-needed iteration (610g / 643c)". That
trigger is now gone: V3-EXQ-655 (the 610f-redesign successor) landed 2026-06-13
and resolved the pre-registered fork (b) -- the stripped REINFORCE control did
NOT collapse, INV-074 was accepted as substrate_ceiling, and the user STOPPED the
610 cascade (no re-queue). So 610g will never run, and 643c is not needed. The
automated index-HIT will not arise naturally -> this is a minimal, purpose-built
consumer that mints a False-mode OFF cell and automated-HITs it, closing P1-auto.

The existing 644/645/646 mints are all include_driver_script_in_hash=True
(legacy), so no False-mode source entry exists in the index. This demo creates
one and consumes it, all in-process, side-effect-free.

WHAT IT DOES (4 passes, seed 42, ONE 610 canonical OFF cell)
------------------------------------------------------------
1. MINT (False-mode): reset_all_rng -> build_off_arm + train_off_arm (5-ep
   minimal-but-real OFF cell from the canonical baseline module
   experiments/_lib/baselines/exq610_inv074_crystallization_baseline.py) ->
   compute_arm_fingerprint(off_path_config_slice(), include_driver_script_in_hash
   =False). Written to a TEMP corpus manifest (NOT evidence/, so the committed
   governance index is never touched).
2. INDEX: run the REAL governance indexer writer
   (build_experiment_indexes._write_arm_fingerprint_index) over the temp corpus
   -> a temp arm_fingerprint_index.json with the False-mode source registered.
   This exercises the production indexer code, not a mock.
3. CONSUME (distinct driver): try_reuse_cell(off_path_config_slice(), seed=42,
   script_path=<a DIFFERENT driver file>, include_driver_script_in_hash=False,
   cite_run_id=<mint run_id>, index_path=<temp index>) -> assert HIT, assert the
   returned cell carries reused_from_run_id / reused_fingerprint / reused_at_utc.
   A genuinely different driver file HITS -- the P1-fix cross-driver property.
4. FLIP: bump ONE byte of config_slice and call try_reuse_cell again -> assert it
   returns None (refuse, fingerprint_not_in_index) so the arm would re-run fresh.
   (Bonus: a legacy include_driver_script_in_hash=True recompute with a different
   driver also REFUSES -- the section-9.7 mode-isolation property.)

PASS iff: mint cell reuse_eligible AND automated index-HIT fired AND provenance
stamped AND the one-byte flip refused. This is the section-9.5 step-6 acceptance.

NOTE: nothing here writes to the committed arm_fingerprint_index.json or to a
reusable arm_results source cell in evidence/. The result manifest records the
demonstration under a `demonstration` block; it carries NO reusable source cell,
so the next governance cycle does not index a transient False-mode source.

Design plan: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md
Helpers: experiments/_lib/arm_fingerprint.py, experiments/_lib/arm_reuse.py
Indexer: REE_assembly/evidence/experiments/scripts/build_experiment_indexes.py
Canonical baseline: experiments/_lib/baselines/exq610_inv074_crystallization_baseline.py
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]                       # ree-v3 root
EXPERIMENTS_DIR = HERE.parent                     # ree-v3/experiments
ASSEMBLY_ROOT = REPO_ROOT.parent / "REE_assembly"
INDEXER_SCRIPTS = ASSEMBLY_ROOT / "evidence" / "experiments" / "scripts"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(INDEXER_SCRIPTS))

from experiment_protocol import emit_outcome
from _lib.arm_fingerprint import reset_all_rng, compute_arm_fingerprint
from _lib.arm_reuse import try_reuse_cell, REFUSE_FP_NOT_IN_INDEX, evaluate_reuse
from _lib.baselines.exq610_inv074_crystallization_baseline import (
    build_off_arm,
    train_off_arm,
    off_path_config_slice,
    BASELINE_ID,
)

EXPERIMENT_TYPE = "v3_exq_685_arm_reuse_automated_hit_demo"
EXPERIMENT_PURPOSE = "baseline"
CLAIM_IDS: list = []

DEMO_SEED = 42
# The OFF metrics this "consumer" reads off the reused baseline cell. Must be a
# subset of the keys the mint cell actually records (section-9.2 needed_keys
# correctness trap); both keys come straight from train_off_arm's metrics dict.
NEEDED_KEYS = ["end_phase_3_entropy", "mean_reward"]


def _utc_compact_now() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mint_off_cell(seed: int) -> Dict[str, Any]:
    """Pass 1: mint a minimal-but-real OFF baseline cell with a FALSE-mode fingerprint."""
    print(f"[685] PASS 1 MINT: reset_all_rng + build/train OFF cell seed={seed} "
          f"(baseline_id={BASELINE_ID}, 5-ep minimal demo budget)", flush=True)
    reset_all_rng(seed)
    agent, env, scheduler = build_off_arm(seed=seed)
    # dry_run=True -> 5 real episodes: a genuine OFF baseline cell, kept cheap.
    metrics = train_off_arm(agent=agent, scheduler=scheduler, seed=seed, dry_run=True)
    metrics.setdefault("seed", seed)

    # FALSE-mode fingerprint: anchored on the canonical baseline module (already in
    # the experiments/_lib/** substrate glob) + config_slice + seed + machine_class;
    # the mint's own driver script is EXCLUDED from the reuse-critical hash so a
    # consumer with its own driver can match (the 2026-06-09 P1-fix enabler).
    metrics["arm_fingerprint"] = compute_arm_fingerprint(
        config_slice=off_path_config_slice(),
        seed=seed,
        script_path=HERE,
        rng_fully_reset=True,
        config_slice_declared=True,
        include_driver_script_in_hash=False,
    )
    fp = metrics["arm_fingerprint"]
    print(f"[685]   minted fp={fp['arm_fingerprint'][:12]} "
          f"reuse_eligible={fp['reuse_eligible']} machine_class={fp['machine_class']} "
          f"driver_in_hash={fp.get('driver_script_in_substrate_hash')}", flush=True)
    return metrics


def _write_temp_mint_manifest(cell: Dict[str, Any], tmp_assembly: Path) -> tuple:
    """Write the mint manifest into a temp REE_assembly/evidence/experiments corpus."""
    ts = _utc_compact_now()
    mint_run_id = f"v3_exq_685_offmint_{ts}_v3"
    evid = tmp_assembly / "evidence" / "experiments"
    evid.mkdir(parents=True, exist_ok=True)
    mint_manifest = {
        "experiment_type": "v3_exq_685_offmint",
        "run_id": mint_run_id,
        "claim_ids": [],
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": "baseline",
        "outcome": "PASS",
        "evidence_direction": "non_contributory",
        "timestamp_utc": ts,
        "completed_at": _utc_iso_now(),
        "baseline_id": BASELINE_ID,
        "arm_results": [cell],
    }
    path = evid / f"{mint_run_id}.json"
    with open(path, "w") as f:
        json.dump(mint_manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return mint_run_id, evid, path


def run_demo() -> Dict[str, Any]:
    """Run the 4-pass automated-index-HIT demonstration. Returns a result dict."""
    import build_experiment_indexes as indexer  # real governance indexer writer

    cell = _mint_off_cell(DEMO_SEED)
    mint_fp = cell["arm_fingerprint"]["arm_fingerprint"]
    mint_reuse_eligible = bool(cell["arm_fingerprint"]["reuse_eligible"])

    tmp_root = Path(tempfile.mkdtemp(prefix="v3_exq_685_"))
    try:
        tmp_assembly = tmp_root / "REE_assembly"
        mint_run_id, evid_dir, mint_path = _write_temp_mint_manifest(cell, tmp_assembly)

        # PASS 2: run the REAL indexer over the temp corpus -> temp index with the
        # False-mode source registered. base_dir.parent.parent == tmp_assembly, so
        # manifest_path is stored relative to it and resolves under try_reuse_cell.
        print("[685] PASS 2 INDEX: build_experiment_indexes._write_arm_fingerprint_index "
              "over temp corpus", flush=True)
        index = indexer._write_arm_fingerprint_index(evid_dir, _utc_compact_now())
        index_path = evid_dir / "arm_fingerprint_index.json"
        source_indexed = mint_fp in (index.get("by_fingerprint") or {})
        print(f"[685]   n_source_cells={index.get('n_source_cells')} "
              f"n_fingerprints={index.get('n_fingerprints')} "
              f"mint_fp_indexed={source_indexed}", flush=True)

        # PASS 3: AUTOMATED index-HIT with a DIFFERENT driver. Pointing script_path at
        # a genuinely different existing file proves the cross-driver HIT (P1-fix):
        # with include_driver_script_in_hash=False the driver drops out of the key.
        consumer_driver = EXPERIMENTS_DIR / "v3_exq_610_inv074_crystallization_baseline_mint.py"
        print(f"[685] PASS 3 CONSUME: try_reuse_cell (driver={consumer_driver.name}, "
              f"include_driver_script_in_hash=False)", flush=True)
        reused = try_reuse_cell(
            config_slice=off_path_config_slice(),
            seed=DEMO_SEED,
            script_path=consumer_driver,
            needed_keys=NEEDED_KEYS,
            cite_run_id=mint_run_id,
            index_path=index_path,
            assembly_root=tmp_assembly,
            include_driver_script_in_hash=False,
        )
        hit = reused is not None
        prov = {}
        if hit:
            prov = {
                "reused_from_run_id": reused.get("reused_from_run_id"),
                "reused_fingerprint": (reused.get("reused_fingerprint") or "")[:12],
                "reused_at_utc": reused.get("reused_at_utc"),
            }
        provenance_stamped = hit and bool(reused.get("reused_from_run_id")) \
            and bool(reused.get("reused_fingerprint")) and bool(reused.get("reused_at_utc"))
        print(f"[685]   automated_index_hit={hit} provenance={prov}", flush=True)

        # PASS 4: flip ONE config byte -> different fingerprint -> refuse (re-run fresh).
        flipped_slice = copy.deepcopy(off_path_config_slice())
        flipped_slice["__demo_flip_byte"] = 1
        print("[685] PASS 4 FLIP: bump one config byte -> try_reuse_cell expects refuse", flush=True)
        flip_decision = evaluate_reuse(
            config_slice=flipped_slice,
            seed=DEMO_SEED,
            script_path=consumer_driver,
            needed_keys=NEEDED_KEYS,
            cite_run_id=mint_run_id,
            index_path=index_path,
            assembly_root=tmp_assembly,
            include_driver_script_in_hash=False,
        )
        flip_refused = (not flip_decision.reused) and flip_decision.reason == REFUSE_FP_NOT_IN_INDEX
        print(f"[685]   flip_refused={flip_refused} reason={flip_decision.reason}", flush=True)

        # Bonus (section 9.7 mode-isolation): a legacy include_driver_script_in_hash=True
        # recompute with a different driver also REFUSES (the two modes never collide).
        legacy_decision = evaluate_reuse(
            config_slice=off_path_config_slice(),
            seed=DEMO_SEED,
            script_path=consumer_driver,
            needed_keys=NEEDED_KEYS,
            cite_run_id=mint_run_id,
            index_path=index_path,
            assembly_root=tmp_assembly,
            include_driver_script_in_hash=True,
        )
        default_mode_cross_driver_refused = not legacy_decision.reused
        print(f"[685]   default_mode_cross_driver_refused={default_mode_cross_driver_refused} "
              f"(reason={legacy_decision.reason})", flush=True)

        return {
            "mint_run_id": mint_run_id,
            "mint_fingerprint": mint_fp,
            "mint_reuse_eligible": mint_reuse_eligible,
            "false_mode_source_indexed": source_indexed,
            "automated_index_hit_fired": hit,
            "provenance_stamped": provenance_stamped,
            "provenance": prov,
            "flip_refused": flip_refused,
            "flip_refuse_reason": flip_decision.reason,
            "default_mode_cross_driver_refused": default_mode_cross_driver_refused,
            "consumer_driver": consumer_driver.name,
            "needed_keys": NEEDED_KEYS,
            "off_metrics": {k: cell.get(k) for k in ("end_phase_3_entropy", "mean_reward", "final_phase")},
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _build_interpretation(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": "arm_reuse_automated_index_hit_demonstrated"
                 if (d["automated_index_hit_fired"] and d["provenance_stamped"] and d["flip_refused"])
                 else "arm_reuse_automated_index_hit_not_demonstrated",
        "preconditions": [
            {
                "name": "mint_cell_reuse_eligible",
                "description": "the minted OFF cell is reuse_eligible (complete RNG reset, schema arm_fp/v1)",
                "measured": 1 if d["mint_reuse_eligible"] else 0,
                "threshold": 1,
                "met": d["mint_reuse_eligible"],
            },
            {
                "name": "false_mode_source_in_index",
                "description": "the real indexer registered the False-mode mint fingerprint as a reusable source",
                "measured": 1 if d["false_mode_source_indexed"] else 0,
                "threshold": 1,
                "met": d["false_mode_source_indexed"],
            },
        ],
        "criteria_non_degenerate": {
            "automated_index_hit_fired": d["automated_index_hit_fired"],
            "provenance_stamped": d["provenance_stamped"],
            "flip_refused": d["flip_refused"],
            "cross_driver_mode_isolation": d["default_mode_cross_driver_refused"],
        },
    }


def main(argv) -> int:
    dry = "--dry-run" in argv
    print(f"[685] arm-reuse automated index-HIT demonstration (arm_reuse_fingerprint:P1-auto)"
          f"{' [--dry-run]' if dry else ''}", flush=True)
    print(f"[train] arm_reuse_demo seed={DEMO_SEED} ep 1/1 (single OFF mint cell)", flush=True)

    d = run_demo()
    passed = bool(
        d["mint_reuse_eligible"]
        and d["false_mode_source_indexed"]
        and d["automated_index_hit_fired"]
        and d["provenance_stamped"]
        and d["flip_refused"]
    )
    outcome = "PASS" if passed else "FAIL"
    print(f"verdict: {outcome}", flush=True)

    if dry:
        print("[685] dry-run complete; full 4-pass demo exercised. No manifest written.", flush=True)
        return outcome, None

    ts = _utc_compact_now()
    qid = os.environ.get("REE_QUEUE_ID")
    qid_clean = (qid or "manual").replace("-", "").lower()
    run_id = f"{EXPERIMENT_TYPE}_{qid_clean}_{ts}_v3"
    evidence_dir = ASSEMBLY_ROOT / "evidence" / "experiments"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence_dir / f"{run_id}.json"

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
        "interpretation": _build_interpretation(d),
        # No reusable arm_results source cell -> the governance indexer indexes
        # nothing from this manifest (the mint source lived only in a temp corpus).
        "demonstration": d,
        "config": {
            "demo_seed": DEMO_SEED,
            "needed_keys": NEEDED_KEYS,
            "baseline_id": BASELINE_ID,
            "note": (
                "arm_reuse_fingerprint:P1-auto -- first AUTOMATED index-HIT in the wild. "
                "Mints a False-mode (include_driver_script_in_hash=False) OFF baseline cell, "
                "runs the real indexer over a temp corpus, consumes via try_reuse_cell from a "
                "DIFFERENT driver (HIT), and confirms a one-byte config flip refuses. "
                "Self-contained: never touches the committed arm_fingerprint_index.json."
            ),
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[685] Manifest written: {manifest_path}", flush=True)
    print(f"Result written to: {manifest_path}", flush=True)
    return outcome, str(manifest_path)


if __name__ == "__main__":
    _outcome, _manifest_path = main(sys.argv)
    if _manifest_path is not None:
        emit_outcome(
            outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
            manifest_path=_manifest_path,
            run_id=Path(_manifest_path).stem,
        )
    sys.exit(0)
