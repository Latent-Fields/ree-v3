"""V4-EXQ-003: DR-10 z_self-in-E3 viability -- controlled wiring falsifier.

Third V4 experiment (self_model_v4:SELF-3; user-approved graduation 2026-07-01, built on
the same-day DR-13 stateful z_self). Same V4 conventions as V4-EXQ-001/002:
  - architecture_epoch = "ree_self_model_v1"
  - run_id suffix "_v4"; queue_id "V4-EXQ-003"; owner_exq -> self_model_v4:SELF-3.

PURPOSE (diagnostic / substrate-readiness; PROMOTES NOTHING; claim_ids=[]):
validate the no-op-default z_self-in-E3 viability lever landed 2026-07-01 in
ree_core/predictors/e3_selector.py (use_self_viability_weighting + score_trajectory
penalty + select(self_viability_per_candidate=...)). A CONTROLLED probe (caller-supplied
per-candidate self-viability cost -- the DR-13 stateful z_self is the SUBJECT the cost is
derived from; the ecological z_self-derived auto-source is a documented follow-on),
exercising the lever end-to-end through the harness path -- not just the unit contracts.
Structurally identical to the DR-12 pilot (V4-EXQ-001): DR-10 is DR-12's sibling lever on
the same score_trajectory + per-candidate select() machinery.

FALSIFIER (SELF-3 build path): if the z_self-derived self-viability weighting does NOT
change trajectory selection when a decisive per-candidate cost is supplied vs the OFF
baseline, DR-10 buys nothing and the wiring is inert.

THREE arms per seed (candidates bit-identical across arms via per-cell RNG reset):
  OFF          -- use_self_viability_weighting=False; baseline. Records the primary-best
                  candidate (committed argmin) + the best-vs-second primary gap.
  DIFFERENTIAL -- lever ON; HIGH self-viability cost assigned ONLY to the OFF primary-best,
                  low elsewhere. Penalty is constructed > gap (by DECISIVENESS_MARGIN) so a
                  WORKING lever MUST move selection away from the low-viability candidate.
                  A non-flip here = inert wiring.
  UNIFORM      -- lever ON; the SAME high cost on ALL candidates (range 0). Control: a
                  uniform penalty is argmin-invariant (the V3-EXQ-571 lesson), so selection
                  MUST be unchanged vs OFF. Confirms it is the PER-CANDIDATE (differential)
                  cost that carries the effect.

NON-VACUITY precondition (readiness): the DIFFERENTIAL arm's cross-candidate
self_viability_range >= SV_RANGE_FLOOR AND the applied penalty exceeds the primary gap
(decisiveness). A flat cost cannot change selection; an under-powered penalty cannot either.
Below-floor self-routes to substrate_not_ready_requeue -- NEVER a false negative.

INERT-WIRING off-ramp: precondition met but DIFFERENTIAL does NOT flip on the majority of
seeds -> label dr10_wiring_inert, non_contributory (DR-10 buys nothing).

GUARDRAILS: generation:v4, off the V3 critical path, promotes nothing in V3. No training
(synthetic deterministic E3-selection probe); no env; no encoder head.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from ree_core.predictors.e2_fast import Trajectory  # noqa: E402
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v4_exq_003_dr10_z_self_viability_falsifier"
ARCH_EPOCH = "ree_self_model_v1"  # V4 epoch (V4-EXQ-001/002 precedent)

SEEDS = [42, 43, 44]
ARMS = ["OFF", "DIFFERENTIAL", "UNIFORM"]
K = 6                      # candidate trajectories
WORLD_DIM = 6
HORIZON = 3
ACTION_DIM = 5
SELF_VIABILITY_WEIGHT = 1.0
DECISIVENESS_MARGIN = 1.0  # penalty = gap + margin -> a working lever MUST flip
SV_RANGE_FLOOR = 0.1       # non-vacuity floor for the differential self-viability range
MAJORITY = 2               # of 3 seeds

OUT_DIR = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"


def _build_candidates(k: int, world_dim: int) -> list:
    """k candidate trajectories with distinct (seeded) z_world content so primary scores
    genuinely differ. RNG is reset by the enclosing arm_cell -> deterministic per seed and
    bit-identical across arms within a seed."""
    cands = []
    for i in range(k):
        states = [torch.randn(1, world_dim) * 0.3 for _ in range(HORIZON + 1)]
        world_states = [torch.randn(1, world_dim) * 0.3 for _ in range(HORIZON + 1)]
        actions = torch.zeros(1, HORIZON, ACTION_DIM)
        actions[:, 0, i % ACTION_DIM] = 1.0
        cands.append(Trajectory(states=states, actions=actions, world_states=world_states))
    return cands


def _run_seed(seed: int) -> dict:
    arm_rows = {}
    off_idx = None
    gap = None
    high_sv = None

    for arm in ARMS:
        cfg_slice = {
            "arm": arm, "K": K, "world_dim": WORLD_DIM,
            "self_viability_weight": SELF_VIABILITY_WEIGHT,
            "decisiveness_margin": DECISIVENESS_MARGIN,
        }
        with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__)) as cell:
            selector = E3TrajectorySelector(E3Config(world_dim=WORLD_DIM, hidden_dim=8))
            selector._running_variance = 0.0  # deterministic committed argmin path
            candidates = _build_candidates(K, WORLD_DIM)

            # OFF reference (primary scores, no self-viability) -- identical across arms.
            selector.config.use_self_viability_weighting = False
            ref = selector.select(candidates, temperature=1.0)
            ref_scores = ref.scores.detach()
            ref_idx = int(ref_scores.argmin().item())
            sorted_s = torch.sort(ref_scores).values
            cell_gap = float((sorted_s[1] - sorted_s[0]).item()) if K >= 2 else 0.0

            if off_idx is None:
                off_idx = ref_idx
                gap = cell_gap
                high_sv = cell_gap + DECISIVENESS_MARGIN  # penalty target > gap

            if arm == "OFF":
                selected_idx = ref_idx
                sv_range = 0.0
                penalty_range = 0.0
                sv_active = False
            else:
                sv = torch.zeros(K)
                if arm == "DIFFERENTIAL":
                    sv[ref_idx] = high_sv       # high viability-cost only on the primary-best
                else:  # UNIFORM
                    sv[:] = high_sv             # uniform -> argmin-invariant control
                selector.config.use_self_viability_weighting = True
                selector.config.self_viability_weight = SELF_VIABILITY_WEIGHT
                sel = selector.select(
                    candidates, temperature=1.0, self_viability_per_candidate=sv
                )
                selected_idx = int(sel.selected_index)
                diag = selector.last_score_diagnostics
                sv_range = float(diag.get("self_viability_range", 0.0))
                penalty_range = float(diag.get("self_viability_penalty_range", 0.0))
                sv_active = bool(diag.get("self_viability_active", False))

            row = {
                "arm": arm,
                "seed": seed,
                "off_primary_best_idx": ref_idx,
                "selected_idx": selected_idx,
                "flipped_vs_off": bool(selected_idx != off_idx),
                "primary_gap": cell_gap,
                "high_self_viability": high_sv if arm != "OFF" else 0.0,
                "self_viability_range": sv_range,
                "self_viability_penalty_range": penalty_range,
                "self_viability_active": sv_active,
            }
            cell.stamp(row)  # writes row["arm_fingerprint"]
            arm_rows[arm] = row

    diff = arm_rows["DIFFERENTIAL"]
    unif = arm_rows["UNIFORM"]
    seed_pass = bool(diff["flipped_vs_off"] and (not unif["flipped_vs_off"]))
    return {
        "seed": seed,
        "off_primary_best_idx": off_idx,
        "primary_gap": gap,
        "high_self_viability": high_sv,
        "differential_flipped": diff["flipped_vs_off"],
        "uniform_flipped": unif["flipped_vs_off"],
        "differential_sv_range": diff["self_viability_range"],
        "differential_penalty_range": diff["self_viability_penalty_range"],
        "penalty_exceeds_gap": bool(high_sv > gap),  # decisiveness precondition
        "seed_pass": seed_pass,
        "arm_rows": [arm_rows[a] for a in ARMS],
    }


def run_experiment(seeds=None, dry_run: bool = False) -> dict:
    seeds = seeds if seeds is not None else (SEEDS[:1] if dry_run else SEEDS)
    seed_records = []
    for s in seeds:
        print(f"Seed {s} Condition dr10_probe", flush=True)
        print(f"  [train] dr10 seed={s} ep 1/1", flush=True)
        rec = _run_seed(s)
        seed_records.append(rec)
        print(f"verdict: {'PASS' if rec['seed_pass'] else 'FAIL'}", flush=True)

    n = len(seed_records)
    n_diff_flip = sum(1 for r in seed_records if r["differential_flipped"])
    n_unif_inert = sum(1 for r in seed_records if not r["uniform_flipped"])
    min_sv_range = min((r["differential_sv_range"] for r in seed_records), default=0.0)
    all_decisive = all(r["penalty_exceeds_gap"] for r in seed_records)
    majority = MAJORITY if not dry_run else 1

    precond_met = (min_sv_range >= SV_RANGE_FLOOR) and all_decisive
    c1_diff_flips = n_diff_flip >= majority           # load-bearing
    c2_unif_inert = n_unif_inert >= majority          # control

    if not precond_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif c1_diff_flips and c2_unif_inert:
        label = "dr10_z_self_viability_changes_selection"
        outcome = "PASS"
    elif c1_diff_flips and not c2_unif_inert:
        label = "dr10_uniform_sv_also_flips_anomaly"  # uniform should be inert
        outcome = "FAIL"
    else:
        label = "dr10_wiring_inert"  # FALSIFIER fired: DR-10 buys nothing
        outcome = "FAIL"

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v4"

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCH_EPOCH,
        "generation": "v4",
        "claim_ids": [],
        "unblocks_claims": ["MECH-215", "ARC-081"],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": ts,
        "owner_node": "self_model_v4:SELF-3",
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "differential_self_viability_range_supra_floor",
                    "description": "DIFFERENTIAL arm cross-candidate self-viability range (the "
                                   "same statistic the lever's per-candidate effect routes on) "
                                   "clears the non-vacuity floor on every seed.",
                    "measured": min_sv_range,
                    "threshold": SV_RANGE_FLOOR,
                    "control": "high self-viability cost assigned only to the OFF primary-best",
                    "met": bool(min_sv_range >= SV_RANGE_FLOOR),
                },
                {
                    "name": "penalty_exceeds_primary_gap",
                    "description": "Applied penalty (weight*high_self_viability) exceeds the "
                                   "best-vs-second primary gap on every seed, so a working lever "
                                   "MUST flip selection (decisiveness; under-powered penalty is "
                                   "not a falsification).",
                    "measured": 1.0 if all_decisive else 0.0,
                    "threshold": 1.0,
                    "met": bool(all_decisive),
                },
            ],
            "criteria_non_degenerate": {
                "C1_differential_flips": bool(K >= 2),
                "C2_uniform_inert": True,  # uniform self-viability range == 0 by construction
            },
            "criteria": [
                {"name": "C1_differential_flips", "load_bearing": True,
                 "passed": bool(c1_diff_flips),
                 "detail": f"{n_diff_flip}/{n} seeds flipped away from the low-viability candidate"},
                {"name": "C2_uniform_inert", "load_bearing": False,
                 "passed": bool(c2_unif_inert),
                 "detail": f"{n_unif_inert}/{n} seeds unchanged under uniform self-viability (control)"},
            ],
        },
        "summary": {
            "n_seeds": n,
            "n_differential_flipped": n_diff_flip,
            "n_uniform_inert": n_unif_inert,
            "min_differential_sv_range": min_sv_range,
            "all_decisive": all_decisive,
            "majority_required": majority,
        },
        "arm_results": [row for r in seed_records for row in r["arm_rows"]],
        "seed_records": seed_records,
        "config": {
            "seeds": seeds, "arms": ARMS, "K": K, "world_dim": WORLD_DIM,
            "horizon": HORIZON, "action_dim": ACTION_DIM,
            "self_viability_weight": SELF_VIABILITY_WEIGHT,
            "decisiveness_margin": DECISIVENESS_MARGIN,
            "sv_range_floor": SV_RANGE_FLOOR, "dry_run": dry_run,
        },
        "notes": "Third V4 experiment. Controlled caller-supplied self-viability wiring probe "
                 "for DR-10 (self_model_v4:SELF-3); DR-12's sibling lever on the same "
                 "score_trajectory + per-candidate select() machinery. FALSIFIER: a decisive "
                 "per-candidate z_self-derived self-viability must change selection vs the OFF "
                 "baseline. The ecological z_self-derived auto-source is a documented follow-on. "
                 "Baseline mint SKIPPED: synthetic no-training E3-selection probe (~seconds), no "
                 "expensive reusable trained OFF baseline. Promotes nothing; off the V3 critical "
                 "path.",
    }
    return manifest, run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest, run_id = run_experiment(dry_run=args.dry_run)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"outcome={manifest['outcome']} label={manifest['interpretation']['label']} "
          f"diff_flip={manifest['summary']['n_differential_flipped']}/{manifest['summary']['n_seeds']} "
          f"unif_inert={manifest['summary']['n_uniform_inert']}/{manifest['summary']['n_seeds']}",
          flush=True)
    print(f"wrote {out_path}", flush=True)

    _outcome = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=run_id,
        dry_run=args.dry_run,
    )
