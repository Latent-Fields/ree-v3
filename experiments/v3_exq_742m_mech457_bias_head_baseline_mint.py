#!/opt/local/bin/python3
"""V3-EXQ-742-m -- BASELINE MINT for the MECH-457 actor-critic ON/OFF lineage.

experiment_purpose=baseline; claim_ids=[]; PROMOTES / DEMOTES NOTHING; excluded from
governance scoring by construction. This is the SEPARATE, ORDER-INDEPENDENT baseline mint
(arm_reuse_fingerprint_plan.md sections 7b, 9) for the reusable OFF arm of the V3-EXQ-742
lineage: the ``bias_head_baseline`` arm (the 724-A0 all-ON incompetence control -- world-
model warmup P0 + two-head REINFORCE P1, SD-056 e2 frozen in P1, evaluated by
capability_eval.REEForwardPolicy). It re-trains ONLY that OFF arm x SEEDS x RUNGS and emits
a reuse-eligible per-cell arm fingerprint, so any strictly-later MECH-457 iteration (a
lettered successor, a sibling ablation, or a re-test) can skip re-training the byte-identical
OFF arm by citing this run via ``reuse_baseline_from``.

WHY A SEPARATE MINT (the correction the parent session first wrongly skipped): the V3-EXQ-742
consumer already self-mints its OFF cells IN-RUN, but that only helps successors that run
STRICTLY AFTER 742 lands a clean manifest. This mint banks the OFF baseline REGARDLESS of
whether/when 742 runs, completes, or is superseded -- the order-independent insurance.
Terminality is unknowable (a first-of-lineage experiment is exactly the paradigm mint case,
not a skip case); the cost of an unneeded mint is idle low-priority cloud compute, never
correctness. Pinned to a ree-cloud worker so the fingerprint is the reusable cloud
machine-class (a Mac mint would be a different class, dead on arrival). That class now
carries the TORCH BUILD as well -- currently `linux-x86_64-py3.10-torch2.5.1+cu121`; see
`machine_class()` in experiments/_lib/arm_fingerprint.py, which is the authority. A fleet
torch upgrade retires this mint exactly as an OS or python change always did, and any
baseline minted BEFORE the 2026-07-19 hard cut is dead and needs re-minting under the new
class (plan section 12).

Both this mint and the V3-EXQ-742 consumer construct the OFF cell + its fingerprint slice
from the ONE shared module experiments/_lib/baselines/exq742_mech457_bias_head_baseline.py
(run_off_cell + off_path_config_slice), so the fingerprints match BY CONSTRUCTION. The
include_driver_script_in_hash=False flag (MANDATORY on BOTH sides) excludes each side's
distinct driver script from the substrate hash, so the mint and the consumer -- which have
different drivers -- still collide on the same fingerprint and reuse HITs.

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

This module is ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, machine_class  # noqa: E402
from experiments._lib.baselines import exq742_mech457_bias_head_baseline as ac_baseline  # noqa: E402
from experiments._lib.capability_eval import COMPETENCE_RESOURCE_FLOOR  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_742m_mech457_bias_head_baseline_mint"
QUEUE_ID = "V3-EXQ-742-m"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "baseline"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Match the V3-EXQ-742 consumer's budget for the OFF arm exactly (so the fingerprint matches).
SEEDS: List[int] = [42, 43, 44]
ZWORLD_P0_EPISODES = x734.ZWORLD_P0_EPISODES        # 60 -- SD-070 z_world encoder warmup (P0a).
                                                    # MUST equal the 742 consumer's setting or
                                                    # the minted arm cannot cache-HIT it.
P0_WARMUP_EPISODES = x734.P0_WARMUP_EPISODES        # 200
P1_REINFORCE_EPISODES = x734.P1_REINFORCE_EPISODES  # 90
EVAL_EPISODES = x734.EVAL_EPISODES                  # 20
STEPS_PER_EPISODE = x734.STEPS_PER_EPISODE          # 200

# Same two rungs (D0 + D3) the 742 consumer evaluates the OFF arm on.
RUNGS = [x734.DIFFICULTY_RUNGS[0], x734.DIFFICULTY_RUNGS[-1]]

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_EVAL = 2
DRY_RUN_STEPS = 15


def run_experiment(
    seeds: List[int], p0: int, p1: int, eval_eps: int, steps: int,
    zworld_p0: int = 0, dry_run: bool = False,
) -> Dict[str, Any]:
    print(
        f"MECH-457 bias_head_baseline MINT ({len(RUNGS)} rungs x {len(seeds)} seeds; "
        f"P0a={zworld_p0}, P0={p0}, P1_reinforce={p1}, eval={eval_eps}, steps={steps})",
        flush=True,
    )
    cells: List[Dict[str, Any]] = []
    n_supra = 0
    for rung in RUNGS:
        rid = rung["rung_id"]
        env_kwargs = x734._env_kwargs_for_rung(rung)
        for seed in seeds:
            print(f"Seed {seed} Condition {rid}:bias_head_baseline_mint", flush=True)
            # MUST carry the same zworld_p0 as the 742 consumer, or the minted arm cannot
            # cache-HIT it (and, worse, a pre-SD-070 mint would satisfy a post-SD-070 consumer
            # if this key were absent -- see off_path_config_slice).
            slice_cfg = ac_baseline.off_path_config_slice(
                env_kwargs, p0, p1, eval_eps, steps, zworld_p0_episodes=zworld_p0,
            )
            with arm_cell(
                seed,
                config_slice=slice_cfg,
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,  # MUST match the 742 consumer's flag
            ) as cell:
                row = ac_baseline.run_off_cell(
                    env_kwargs, seed,
                    p0_warmup_episodes=p0, p1_reinforce_episodes=p1,
                    eval_episodes=eval_eps, steps_per_episode=steps, rung_id=rid,
                    zworld_p0_episodes=zworld_p0, zworld_p0_dry_run=dry_run,
                )
                row["rung_id"] = rid
                row["arm_id"] = ac_baseline.REUSABLE_ARM_ID
                row["seed"] = int(seed)
                cell.stamp(row)
            forage = float(row["foraging_competence"])
            if row.get("competence_supra_floor"):
                n_supra += 1
            cells.append(row)
            print(
                f"verdict: PASS (mint rung={rid} seed={seed} forage/ep={forage} "
                f"fingerprint_eligible={row.get('arm_fingerprint', {}).get('reuse_eligible')})",
                flush=True,
            )

    return {
        # A mint is a successful RUN if it produced all cells + fingerprints; its foraging
        # verdict is not a scientific PASS/FAIL (baseline purpose). We emit PASS to mean
        # "the baseline was minted", which is what the runner records.
        "outcome": "PASS",
        "n_cells": len(cells),
        "n_supra_floor": int(n_supra),
        "arm_results": cells,
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "lineage": ac_baseline.LINEAGE,
        "mints_arm": ac_baseline.REUSABLE_ARM_ID,
        # Computed, never hardcoded: the tag gained the torch version on 2026-07-19 and a
        # literal here would silently go stale again on the next fleet torch upgrade.
        "reuse_machine_class": machine_class(),
        "n_cells": result["n_cells"],
        "n_supra_floor": result["n_supra_floor"],
        "arm_results": result["arm_results"],
        "config": {
            "seeds": SEEDS if not dry_run else DRY_RUN_SEEDS,
            "rungs": [r["rung_id"] for r in RUNGS],
            "zworld_p0_episodes": (
                ZWORLD_P0_EPISODES if not dry_run else x734.DRY_RUN_ZWORLD_P0
            ),
            "p0_warmup_episodes": P0_WARMUP_EPISODES if not dry_run else DRY_RUN_P0,
            "p1_reinforce_episodes": P1_REINFORCE_EPISODES if not dry_run else DRY_RUN_P1,
            "eval_episodes": EVAL_EPISODES if not dry_run else DRY_RUN_EVAL,
            "steps_per_episode": STEPS_PER_EPISODE if not dry_run else DRY_RUN_STEPS,
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        },
        "notes": (
            "Baseline mint for the MECH-457 actor-critic ON/OFF lineage bias_head_baseline "
            "(724-A0 all-ON) OFF arm. PROMOTES/DEMOTES NOTHING; excluded from governance "
            "scoring. Order-independent reuse insurance: cite this run_id via reuse_baseline_from "
            "in a future MECH-457 iteration. include_driver_script_in_hash=False on both sides. "
            "Pinned to a ree-cloud worker for the reusable cloud machine-class (torch build "
            "included in the tag since 2026-07-19; see reuse_machine_class for this run's value)."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-742-m MECH-457 bias_head_baseline MINT (baseline; claim_ids=[])"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, eval_eps, steps = DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_EVAL, DRY_RUN_STEPS
        zworld_p0 = x734.DRY_RUN_ZWORLD_P0
    else:
        seeds = list(SEEDS)
        p0, p1, eval_eps, steps = P0_WARMUP_EPISODES, P1_REINFORCE_EPISODES, EVAL_EPISODES, STEPS_PER_EPISODE
        zworld_p0 = ZWORLD_P0_EPISODES

    result = run_experiment(
        seeds=seeds, p0=p0, p1=p1, eval_eps=eval_eps, steps=steps,
        zworld_p0=zworld_p0, dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} minted_cells={result['n_cells']} "
        f"n_supra_floor={result['n_supra_floor']} lineage={ac_baseline.LINEAGE}",
        flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return "PASS", manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
