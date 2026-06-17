"""V4-EXQ-001: DR-12 PE-conditioned E3 confidence -- controlled wiring falsifier.

THE FIRST-EVER V4 EXPERIMENT (self_model_v4:SELF-4; user-APPROVED graduation
2026-06-16). Precedent-setting conventions for the V4 generation:
  - architecture_epoch = "ree_self_model_v1"  (NEW V4 epoch parallel to V3's
    ree_hybrid_guardrails_v1; per docs/architecture/v4_spec.md:267 each V4 track
    gets its own epoch, e.g. ree_multi_agent_v1 for the multi-agent track).
  - run_id suffix "_v4" (parallel to V3 "_v3").
  - queue_id "V4-EXQ-001" (V4-EXQ-NNN namespace parallel to V3-EXQ-NNN).
  - owner_exq=V4-EXQ-001 assigned to the self_model_v4:SELF-4 node WHEN queued.

PURPOSE (diagnostic / substrate-readiness; PROMOTES NOTHING; claim_ids=[]):
validate the no-op-default E2-forward-PE -> E3 confidence down-weight lever landed
2026-06-17 in ree_core/predictors/e3_selector.py (use_pe_confidence_weighting +
score_trajectory penalty + select(e2_forward_pe_per_candidate=...)). This is a
CONTROLLED probe (caller-supplied per-candidate PE; the ecological region-PE
auto-source is a documented follow-on), exercising the lever end-to-end through
the experiment harness path -- not just the unit contracts.

FALSIFIER (SELF-4 graduation decision): if PE-conditioned confidence weighting
does NOT change trajectory selection in high-PE (poorly-modelled) regions vs the
unconditional-trust baseline, DR-12 buys nothing and the wiring is inert.

THREE arms per seed (candidates bit-identical across arms via per-cell RNG reset):
  OFF          -- use_pe_confidence_weighting=False; unconditional-trust baseline.
                  Records the primary-best candidate (committed argmin) + the
                  best-vs-second primary gap.
  DIFFERENTIAL -- lever ON; HIGH PE assigned ONLY to the OFF primary-best, low
                  elsewhere. Penalty is constructed > gap (by DECISIVENESS_MARGIN)
                  so a WORKING lever MUST move selection away from the high-PE
                  (poorly-modelled) region. A non-flip here = inert wiring.
  UNIFORM      -- lever ON; the SAME high PE on ALL candidates (range 0). Control:
                  a uniform penalty is argmin-invariant (the V3-EXQ-571 lesson), so
                  selection MUST be unchanged vs OFF. Confirms it is the
                  PER-CANDIDATE (differential) PE that carries the effect.

NON-VACUITY precondition (readiness): the DIFFERENTIAL arm's cross-candidate
e2_forward_pe_range >= PE_RANGE_FLOOR AND the applied penalty exceeds the primary
gap (decisiveness). A flat PE cannot change selection; an under-powered penalty
cannot either. Below-floor self-routes to substrate_not_ready_requeue -- NEVER a
false negative against the lever.

INERT-WIRING off-ramp: precondition met but DIFFERENTIAL does NOT flip on the
majority of seeds -> label dr12_wiring_inert, non_contributory (DR-12 buys nothing).

GUARDRAILS: generation:v4, off the V3 critical path, promotes nothing in V3.
No training (synthetic deterministic E3-selection probe); no env; no encoder head.
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
EXPERIMENT_TYPE = "v4_exq_001_dr12_pe_conditioned_confidence_falsifier"
ARCH_EPOCH = "ree_self_model_v1"  # NEW V4 epoch (precedent)

SEEDS = [42, 43, 44]
ARMS = ["OFF", "DIFFERENTIAL", "UNIFORM"]
K = 6                      # candidate trajectories
WORLD_DIM = 6
HORIZON = 3
ACTION_DIM = 5
PE_CONFIDENCE_WEIGHT = 1.0
DECISIVENESS_MARGIN = 1.0  # penalty = gap + margin -> a working lever MUST flip
PE_RANGE_FLOOR = 0.1       # non-vacuity floor for the differential PE range
MAJORITY = 2               # of 3 seeds

OUT_DIR = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"


def _build_candidates(k: int, world_dim: int) -> list:
    """Build k candidate trajectories with distinct (seeded) z_world content so the
    primary scores genuinely differ. RNG is reset by the enclosing arm_cell, so this
    is deterministic per seed and bit-identical across arms within a seed."""
    cands = []
    for i in range(k):
        states = [torch.randn(1, world_dim) * 0.3 for _ in range(HORIZON + 1)]
        world_states = [torch.randn(1, world_dim) * 0.3 for _ in range(HORIZON + 1)]
        actions = torch.zeros(1, HORIZON, ACTION_DIM)
        actions[:, 0, i % ACTION_DIM] = 1.0
        cands.append(Trajectory(states=states, actions=actions, world_states=world_states))
    return cands


def _run_seed(seed: int) -> dict:
    """Run all three arms for one seed and return the per-seed record."""
    arm_rows = {}
    off_idx = None
    gap = None
    high_pe = None

    for arm in ARMS:
        cfg_slice = {
            "arm": arm, "K": K, "world_dim": WORLD_DIM,
            "pe_confidence_weight": PE_CONFIDENCE_WEIGHT,
            "decisiveness_margin": DECISIVENESS_MARGIN,
        }
        with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__)) as cell:
            # arm_cell reset RNG to `seed` on enter -> selector init + candidates
            # are bit-identical across arms within this seed.
            selector = E3TrajectorySelector(E3Config(world_dim=WORLD_DIM, hidden_dim=8))
            selector._running_variance = 0.0  # deterministic committed argmin path
            candidates = _build_candidates(K, WORLD_DIM)

            # OFF reference (primary scores, no PE) -- identical across arms.
            selector.config.use_pe_confidence_weighting = False
            ref = selector.select(candidates, temperature=1.0)
            ref_scores = ref.scores.detach()
            ref_idx = int(ref_scores.argmin().item())
            sorted_s = torch.sort(ref_scores).values
            cell_gap = float((sorted_s[1] - sorted_s[0]).item()) if K >= 2 else 0.0

            if off_idx is None:
                off_idx = ref_idx
                gap = cell_gap
                high_pe = cell_gap + DECISIVENESS_MARGIN  # penalty target > gap

            if arm == "OFF":
                selected_idx = ref_idx
                pe_range = 0.0
                penalty_range = 0.0
                pe_active = False
            else:
                pe = torch.zeros(K)
                if arm == "DIFFERENTIAL":
                    pe[ref_idx] = high_pe       # high PE only on the primary-best
                else:  # UNIFORM
                    pe[:] = high_pe             # uniform -> argmin-invariant control
                selector.config.use_pe_confidence_weighting = True
                selector.config.pe_confidence_weight = PE_CONFIDENCE_WEIGHT
                sel = selector.select(
                    candidates, temperature=1.0, e2_forward_pe_per_candidate=pe
                )
                selected_idx = int(sel.selected_index)
                diag = selector.last_score_diagnostics
                pe_range = float(diag.get("e2_forward_pe_range", 0.0))
                penalty_range = float(diag.get("pe_confidence_penalty_range", 0.0))
                pe_active = bool(diag.get("pe_confidence_active", False))

            row = {
                "arm": arm,
                "seed": seed,
                "off_primary_best_idx": ref_idx,
                "selected_idx": selected_idx,
                "flipped_vs_off": bool(selected_idx != off_idx),
                "primary_gap": cell_gap,
                "high_pe": high_pe if arm != "OFF" else 0.0,
                "e2_forward_pe_range": pe_range,
                "pe_confidence_penalty_range": penalty_range,
                "pe_confidence_active": pe_active,
            }
            cell.stamp(row)  # writes row["arm_fingerprint"]
            arm_rows[arm] = row

    diff = arm_rows["DIFFERENTIAL"]
    unif = arm_rows["UNIFORM"]
    # per-seed: differential flips away from the high-PE primary-best AND uniform inert.
    seed_pass = bool(diff["flipped_vs_off"] and (not unif["flipped_vs_off"]))
    return {
        "seed": seed,
        "off_primary_best_idx": off_idx,
        "primary_gap": gap,
        "high_pe": high_pe,
        "differential_flipped": diff["flipped_vs_off"],
        "uniform_flipped": unif["flipped_vs_off"],
        "differential_pe_range": diff["e2_forward_pe_range"],
        "differential_penalty_range": diff["pe_confidence_penalty_range"],
        "penalty_exceeds_gap": bool(high_pe > gap),  # decisiveness precondition
        "seed_pass": seed_pass,
        "arm_rows": [arm_rows[a] for a in ARMS],
    }


def run_experiment(seeds=None, dry_run: bool = False) -> dict:
    seeds = seeds if seeds is not None else (SEEDS[:1] if dry_run else SEEDS)
    seed_records = []
    for s in seeds:
        print(f"Seed {s} Condition dr12_probe", flush=True)
        print(f"  [train] dr12 seed={s} ep 1/1", flush=True)
        rec = _run_seed(s)
        seed_records.append(rec)
        print(f"verdict: {'PASS' if rec['seed_pass'] else 'FAIL'}", flush=True)

    n = len(seed_records)
    n_diff_flip = sum(1 for r in seed_records if r["differential_flipped"])
    n_unif_inert = sum(1 for r in seed_records if not r["uniform_flipped"])
    # readiness preconditions (across seeds): differential PE range >= floor on every
    # seed AND penalty exceeds the primary gap on every seed (decisiveness).
    min_pe_range = min((r["differential_pe_range"] for r in seed_records), default=0.0)
    all_decisive = all(r["penalty_exceeds_gap"] for r in seed_records)
    majority = MAJORITY if not dry_run else 1

    precond_met = (min_pe_range >= PE_RANGE_FLOOR) and all_decisive
    c1_diff_flips = n_diff_flip >= majority           # load-bearing
    c2_unif_inert = n_unif_inert >= majority          # control

    if not precond_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif c1_diff_flips and c2_unif_inert:
        label = "dr12_pe_conditioning_changes_selection"
        outcome = "PASS"
    elif c1_diff_flips and not c2_unif_inert:
        label = "dr12_uniform_pe_also_flips_anomaly"  # uniform should be inert
        outcome = "FAIL"
    else:
        label = "dr12_wiring_inert"  # FALSIFIER fired: DR-12 buys nothing
        outcome = "FAIL"

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v4"  # V4 run_id suffix (precedent)

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCH_EPOCH,
        "generation": "v4",
        "claim_ids": [],
        "unblocks_claims": ["MECH-215"],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": ts,
        "owner_node": "self_model_v4:SELF-4",
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "differential_pe_range_supra_floor",
                    "description": "DIFFERENTIAL arm cross-candidate PE range (the same "
                                   "statistic the lever's per-candidate effect routes on) "
                                   "clears the non-vacuity floor on every seed.",
                    "measured": min_pe_range,
                    "threshold": PE_RANGE_FLOOR,
                    "control": "high PE assigned only to the OFF primary-best candidate",
                    "met": bool(min_pe_range >= PE_RANGE_FLOOR),
                },
                {
                    "name": "penalty_exceeds_primary_gap",
                    "description": "Applied penalty (weight*high_pe) exceeds the best-vs-"
                                   "second primary gap on every seed, so a working lever "
                                   "MUST flip selection (decisiveness; under-powered "
                                   "penalty is not a falsification).",
                    "measured": 1.0 if all_decisive else 0.0,
                    "threshold": 1.0,
                    "met": bool(all_decisive),
                },
            ],
            "criteria_non_degenerate": {
                "C1_differential_flips": bool(K >= 2),
                "C2_uniform_inert": True,  # uniform PE range == 0 by construction
            },
            "criteria": [
                {"name": "C1_differential_flips", "load_bearing": True,
                 "passed": bool(c1_diff_flips),
                 "detail": f"{n_diff_flip}/{n} seeds flipped away from the high-PE region"},
                {"name": "C2_uniform_inert", "load_bearing": False,
                 "passed": bool(c2_unif_inert),
                 "detail": f"{n_unif_inert}/{n} seeds unchanged under uniform PE (control)"},
            ],
        },
        "summary": {
            "n_seeds": n,
            "n_differential_flipped": n_diff_flip,
            "n_uniform_inert": n_unif_inert,
            "min_differential_pe_range": min_pe_range,
            "all_decisive": all_decisive,
            "majority_required": majority,
        },
        "arm_results": [row for r in seed_records for row in r["arm_rows"]],
        "seed_records": seed_records,
        "config": {
            "seeds": seeds, "arms": ARMS, "K": K, "world_dim": WORLD_DIM,
            "horizon": HORIZON, "action_dim": ACTION_DIM,
            "pe_confidence_weight": PE_CONFIDENCE_WEIGHT,
            "decisiveness_margin": DECISIVENESS_MARGIN,
            "pe_range_floor": PE_RANGE_FLOOR, "dry_run": dry_run,
        },
        "notes": "FIRST V4 experiment. Controlled caller-supplied-PE wiring probe for "
                 "DR-12 (self_model_v4:SELF-4). FALSIFIER: PE-conditioned weighting must "
                 "change selection in high-PE regions vs the unconditional-trust baseline. "
                 "ecological region-PE auto-source is a documented follow-on. Promotes "
                 "nothing; off the V3 critical path.",
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
    )
