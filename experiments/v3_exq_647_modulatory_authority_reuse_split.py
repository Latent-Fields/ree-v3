#!/opt/local/bin/python3
"""V3-EXQ-647 -- 643a reconstructed on cloud-4 via arm-reuse (control reused, experimental fresh).

experiment_purpose=diagnostic. claim_ids=[]. This is an arm-reuse SYSTEM TEST + a
second (cloud-class) replication of V3-EXQ-643a.

THE TEST (user-directed 2026-06-06)
-----------------------------------
643a is a 3-arm experiment: ARM_A (authority OFF / control) + ARM_B (gain 0.5) +
ARM_C (gain 0.8), each x seeds [42,43,44]. V3-EXQ-646 already minted ARM_A on
cloud-4 (it IS 643a's control arm -- built from the canonical baseline module that
reproduces 643a's ARM_A byte-for-byte). So instead of re-running the control, this
experiment:
  1. REUSES the control arm from the latest V3-EXQ-646 mint manifest (explicit-cite
     Phase-1 reuse; provenance stamped on each reused cell), and
  2. runs ONLY the experimental arms (ARM_B, ARM_C) fresh on cloud-4 using 643a's
     own _run_seed_arm (byte-identical ON-arm computation), then
  3. assembles a full 643a arm_results = [reused control] + [fresh experimental] and
     runs 643a's own _evaluate() / interpretation over it.

WHY THIS IS A VALID 643a (not an approximation)
-----------------------------------------------
The reused control (646) and the fresh experimental arms both run on cloud-4
(same machine_class linux-x86_64), same seeds, same canonical config -- differing
ONLY in use_modulatory_selection_authority. In a full fresh cloud-4 643a, every
cell re-seeds torch+numpy at entry (order-independent), so ARM_A from 646 ==
ARM_A a full cloud-4 643a would compute, and ARM_B/ARM_C here == ARM_B/ARM_C a
full cloud-4 643a would compute. The reassembled experiment is therefore
cell-for-cell what a fully-fresh cloud-4 643a would produce -- the reuse is sound
by construction (the matched-control invariant holds: control + treatment share
seed + substrate + machine_class).

DUAL COMPARISON
---------------
- Within cloud-4 (rigorous): valid same-machine_class 643a by construction.
- vs the Mac full 643a (V3-EXQ-643a, darwin-arm64): cross machine_class, so the
  comparison is at the VERDICT level (readiness / C0 / C1 / C2) -- does reuse-
  assembled 643a reach the same conclusion as the fully-fresh Mac run?

REUSE PATH: explicit-cite (Phase-1 "OPT-IN CITE-BASELINE", auditable). This does
NOT rely on the automated arm_fingerprint_index HIT -- the Phase-0 fingerprint
folds the calling driver's content into substrate_hash, so an automated HIT keyed
on THIS driver would refuse against 646 (minted with the mint driver's path). That
automated decision is RECORDED here (reuse_test.automated_consumer_decision) to
document the script_path-coupling gap for the Phase-1 consumer build, but the
actual reuse is performed by explicit cite of the 646 manifest. NO governance
weight (claim_ids=[]); a supervised, user-directed first live cite-reuse.

architecture_epoch: "ree_hybrid_guardrails_v1"

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_647_modulatory_authority_reuse_split.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import compute_arm_fingerprint  # noqa: E402
from experiments._lib.baselines.exq643_modulatory_authority_baseline import (  # noqa: E402
    off_path_config_slice,
    run_off_cell,
)

EXPERIMENT_TYPE = "v3_exq_647_modulatory_authority_reuse_split"
QUEUE_ID = "V3-EXQ-647"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

# Lineage source for the reused control arm.
CITE_MINT_EXPERIMENT_TYPE = "v3_exq_646_mint_modulatory_authority_off_baseline"
EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# OFF-arm metric keys 643a's _evaluate() reads off the control cells. A reused
# control cell missing any of these is unusable -> refuse the reuse.
NEEDED_OFF_KEYS: Tuple[str, ...] = (
    "seed",
    "modulatory_authority_active_frac",
    "modulatory_authority_scale_factor_mean",
    "modulatory_authority_range_mean",
    "raw_score_range_mean",
    "raw_bounded_frac",
    "score_bias_abs_mean",
    "bias_changed_selection_frac",
    "visited_cells",
    "mean_episode_length",
    "selected_action_class_entropy",
    "n_positive_control_ticks",
)

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30


def _load_643a_module():
    """Import the 643a module so the ON-arm computation + evaluator are byte-identical."""
    path = REPO_ROOT / "experiments" / "v3_exq_643a_modulatory_authority_validation.py"
    spec = importlib.util.spec_from_file_location("exq643a_for_reuse", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _latest_mint_manifest() -> Optional[Path]:
    cands = sorted(EVIDENCE_DIR.glob(f"{CITE_MINT_EXPERIMENT_TYPE}_*_v3.json"))
    return cands[-1] if cands else None


def _record_automated_decision(off_cells_slice, seeds, p0, p1, steps) -> Dict[str, Any]:
    """Best-effort: record what the AUTOMATED Phase-1 consumer would decide, to
    document the script_path-coupling gap. Never raises; never changes behaviour."""
    try:
        from experiments._lib.arm_reuse import evaluate_reuse  # noqa: E402
    except Exception as exc:  # pragma: no cover
        return {"available": False, "note": f"arm_reuse import failed: {type(exc).__name__}"}
    out: Dict[str, Any] = {"available": True, "per_seed": []}
    cfg = off_path_config_slice(p0=p0, p1=p1, steps=steps)
    for seed in seeds:
        try:
            dec = evaluate_reuse(
                config_slice=cfg,
                seed=seed,
                script_path=Path(__file__),  # THIS driver -> demonstrates the coupling
                needed_keys=list(NEEDED_OFF_KEYS),
            )
            out["per_seed"].append({"seed": seed, "reused": bool(dec.reused), "reason": dec.reason})
        except Exception as exc:  # pragma: no cover
            out["per_seed"].append({"seed": seed, "reused": False, "reason": f"error:{type(exc).__name__}"})
    out["note"] = (
        "Automated index-HIT keyed on THIS driver's script_path; expected to REFUSE "
        "vs 646 (minted with the mint driver path) -- the explicit cite below performs "
        "the actual reuse. Documents the Phase-1 fingerprint script_path-coupling gap."
    )
    return out


def _reused_control_cells(
    seeds: List[int], p0: int, p1: int, steps: int, dry_run: bool
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Obtain ARM_A (control) cells by explicit-cite reuse of the latest 646 mint.

    Returns (off_cells, reuse_meta). Raises RuntimeError on a real (non-dry) run if
    no valid mint manifest is available -- the runner classifies that as ERROR and
    leaves the item in the queue (correct: do not assemble a 643a without its control).
    """
    manifest_path = _latest_mint_manifest()

    if manifest_path is None:
        if dry_run:
            print("Dry run -- no V3-EXQ-646 mint manifest found; computing control locally as a stand-in.", flush=True)
            off_cells = []
            for seed in seeds:
                print(f"Seed {seed} Condition ARM_A (dry-run local stand-in)", flush=True)
                cell = run_off_cell(seed, p0_episodes=p0, p1_episodes=p1, steps_per_episode=steps)
                cell["reused_from_run_id"] = "DRY_RUN_LOCAL"
                off_cells.append(cell)
                print("verdict: PASS", flush=True)
            return off_cells, {"reuse_source": "dry_run_local_off", "mint_manifest": None}
        raise RuntimeError(
            "No V3-EXQ-646 mint manifest found under evidence/experiments/ -- "
            "the control arm must be minted (646) before this reuse reconstruction can run."
        )

    with open(manifest_path) as fh:
        mint = json.load(fh)
    mint_run_id = mint.get("run_id")
    rows = [r for r in (mint.get("arm_results") or []) if isinstance(r, dict) and r.get("arm_id") == "ARM_A"]
    by_seed = {int(r["seed"]): r for r in rows if "seed" in r}

    off_cells: List[Dict[str, Any]] = []
    for seed in seeds:
        cell = by_seed.get(int(seed))
        if cell is None:
            raise RuntimeError(f"Mint {mint_run_id} has no ARM_A cell for seed {seed}; cannot reuse.")
        if cell.get("error_note") is not None:
            raise RuntimeError(f"Mint {mint_run_id} ARM_A seed {seed} carried error_note; refusing reuse.")
        missing = [k for k in NEEDED_OFF_KEYS if k not in cell]
        if missing:
            raise RuntimeError(f"Mint {mint_run_id} ARM_A seed {seed} missing keys {missing}; refusing reuse.")
        reused = json.loads(json.dumps(cell, default=str))
        fp = reused.get("arm_fingerprint") or {}
        reused["reused_from_run_id"] = mint_run_id
        reused["reused_fingerprint"] = fp.get("arm_fingerprint") if isinstance(fp, dict) else None
        reused["reused_at_utc"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        off_cells.append(reused)
        print(f"Seed {seed} Condition ARM_A (REUSED from {mint_run_id})", flush=True)
        print("verdict: PASS", flush=True)

    reuse_meta = {
        "reuse_source": "explicit_cite_646_mint",
        "mint_manifest": manifest_path.name,
        "mint_run_id": mint_run_id,
        "mint_machine_classes": mint.get("machine_classes"),
        "n_control_cells_reused": len(off_cells),
        "automated_consumer_decision": _record_automated_decision(off_cells, seeds, p0, p1, steps),
    }
    return off_cells, reuse_meta


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    m = _load_643a_module()
    seeds = m.DRY_RUN_SEEDS if dry_run else m.SEEDS
    p0 = m.DRY_RUN_P0 if dry_run else m.P0_WARMUP_EPISODES
    p1 = m.DRY_RUN_P1 if dry_run else m.P1_MEASUREMENT_EPISODES
    steps = m.DRY_RUN_STEPS if dry_run else m.STEPS_PER_EPISODE

    # 1. Control arm: REUSED from the 646 mint (explicit cite).
    off_cells, reuse_meta = _reused_control_cells(seeds, p0, p1, steps, dry_run)

    # 2. Experimental arms: fresh on cloud-4, byte-identical to 643a's ON arms.
    arm_results: List[Dict[str, Any]] = list(off_cells)
    on_arms = [a for a in m.ARMS if a["arm_id"] != "ARM_A"]
    for arm in on_arms:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = m._run_seed_arm(arm, seed, p0, p1, steps)
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    # 3. Evaluate with 643a's own evaluator (faithful 643a verdict).
    summary = m._evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    edir = m._evidence_direction(summary)
    label = m._interpretation_label(summary)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "modulatory_range_supra_floor",
                "description": (
                    "gate's true cross-candidate modulatory range (modulatory_authority_range) "
                    "clears the floor -- the SAME RANGE statistic the C1 authority criterion gates on"
                ),
                "control": "ON arms, ticks with >= 2 first-action classes",
                "measured": round(summary["p_range_measured"], 8),
                "threshold": m.MODULATORY_RANGE_FLOOR,
                "met": bool(summary["p_range_met"]),
            },
            {
                "name": "primary_scores_bounded",
                "description": (
                    "fraction of ON-arm P1 ticks with e3_raw_score_range_mean < "
                    f"{m.RAW_SCORE_RANGE_BOUND} -- non-vacuity guard"
                ),
                "control": "SD-056 rollout-norm clamp enabled",
                "measured": round(summary["p_bounded_measured"], 6),
                "threshold": m.BOUNDED_FRAC,
                "met": bool(summary["p_bounded_met"]),
            },
        ],
        "criteria_non_degenerate": {
            "C0": bool(summary["c0_pass"]),
            "C1": bool(summary["c1_non_degenerate"]),
            "C2": bool(summary["c2_non_degenerate"]),
        },
        "criteria": [
            {
                "name": "C1_authority_active_on_bounded_scores",
                "load_bearing": True,
                "passed": bool(summary["c1_pass"] and summary["readiness_met"]),
            },
        ],
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
            "Arm-reuse SYSTEM TEST + cloud-class replication of V3-EXQ-643a. The "
            "control arm (ARM_A authority_off_baseline) is REUSED by explicit cite from "
            "the V3-EXQ-646 mint (built from the canonical baseline module that "
            "reproduces 643a's ARM_A byte-for-byte); ONLY the experimental arms "
            "(ARM_B gain0.5 / ARM_C gain0.8) are run fresh on cloud-4 via 643a's own "
            "_run_seed_arm. Control + experimental arms share machine_class "
            "(linux-x86_64), seeds, and config (differ only in the authority flag), so "
            "the reassembled experiment is cell-for-cell what a fully-fresh cloud-4 643a "
            "would produce -- the reuse is sound by construction. Compare the verdict "
            "(readiness/C0/C1/C2) to the Mac full 643a (V3-EXQ-643a, darwin-arm64; "
            "cross machine_class, so verdict-level). diagnostic, claim_ids=[]; does NOT "
            "weight governance -- it tests the arm-reuse system and gives a second "
            "643a verdict. Supervised, user-directed first live explicit-cite reuse."
        ),
        "interpretation": interpretation,
        "dry_run": bool(dry_run),
        "reuse_test": {
            "reused_arm_id": "ARM_A",
            "fresh_arm_ids": [a["arm_id"] for a in on_arms],
            "control_arm_source": "V3-EXQ-646 mint (explicit cite)",
            "reuse_phase": 1,
            "reuse_mode": "explicit_cite",
            **reuse_meta,
            "compare_to_full_fresh": "V3-EXQ-643a (DLAPTOP-4.local, darwin-arm64)",
        },
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": m.ENV_KWARGS,
            "arms": [
                {
                    "arm_id": a["arm_id"],
                    "label": a["label"],
                    "use_modulatory_selection_authority": a["use_modulatory_selection_authority"],
                    "modulatory_authority_gain": a["modulatory_authority_gain"],
                    "source": "reused_from_646" if a["arm_id"] == "ARM_A" else "fresh_cloud4",
                }
                for a in m.ARMS
            ],
        },
        "acceptance_criteria": {
            "readiness_met": summary["readiness_met"],
            "C0_curiosity_non_degeneracy": summary["c0_pass"],
            "C1_authority_mechanism_active": summary["c1_pass"],
            "C2_authority_changes_selection_and_behaviour": summary["c2_pass"],
            "C3_dose_response_informative": summary["c3_dose_response_pass"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = EVIDENCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
        print(f"Result written to: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}  label: {label}", flush=True)
    print(f"  reuse_source: {reuse_meta.get('reuse_source')}", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-647 643a reuse-split reconstruction on cloud-4"
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
