"""
V3-EXQ-778: SD-068 consolidation-pipeline staged-damage DIAGNOSTIC.
SLEEP DRIVER: manual-cycle-loop (the SD-068 harness drives enter_sws_mode /
run_sws_schema_pass + enter_rem_mode / run_rem_attribution_pass +
recalibrate_precision_to directly per phase readout; no SleepLoopManager scheduling).

Validates the SD-068 consolidation-pipeline lesion harness
(experiments/_lib/consolidation_lesion_harness.py) end-to-end and reports the
MECH-168 / INV-047 / MECH-169 staged-decline-under-uniform-damage readout.

WHAT IT MEASURES (per seed, single condition):
  run_staged_sweep applies one UNIFORM diffuse-damage sigma
  [0.0, 0.25, 0.5, 1.0, 2.0] identically to each of the three consolidation
  phases and reads each phase's output-quality against INJECTED known content
  (V3-EXQ-702 injected-content precedent -> sidesteps the failure_autopsy_V3-EXQ-538a
  encoding-starvation ceiling):
    MECH-120 SWS  -> denoising-SNR
    MECH-121 NREM -> transfer-fidelity  (substrate-plumbing-fidelity only; see below)
    MECH-123 REM  -> precision-calibration-error (+ passthrough-vs-generative contrast)
  Non-vacuity: per-phase errors are normalised to fractional-of-own-range
  degradation and ranked by damage-TOLERANCE (crossing sigma) for the observed
  staged-failure order; the sigma=0 point is the intact zero-damage baseline.

WHY DIAGNOSTIC (not evidence):
  This is SD-068 substrate-readiness validation. It does NOT weight governance
  confidence. It tags MECH-168/INV-047/MECH-169 (the staged-decline claims the
  harness serves) and SD-068 (the harness itself). It deliberately does NOT tag
  MECH-121 as promotion evidence -- MECH-121 is candidate/substrate_conditional
  (hold_pending_v3_substrate); the NREM leg here is substrate-plumbing-fidelity on
  injected content, NOT MECH-121 behavioural validation. The glymphatic/amyloid
  structural half of MECH-169 has no V3 analog and is out of scope.

ACCEPTANCE (diagnostic, non-vacuous -- PASS means "the harness is a working
instrument", NOT "the staging prediction is confirmed"):
  On >= 2/3 seeds: (C1) each of the three phases shows MONOTONE degradation across
  sigma (positive sigma-error correlation AND non-trivial span), (C2) the intact
  (sigma=0) readouts are non-degenerate (the P0 positive control), and (C3) the
  staged-failure order + REM passthrough-vs-generative contrast are computable.
  The staging MATCH vs the reverse-dependency prediction (rem, nrem, sws) is
  REPORTED, never gated -- a partial-match / inversion is a VALID diagnostic
  outcome (the harness is designed to not rubber-stamp the prediction).
  Load-bearing criterion: C1 (monotone degradation). If the harness produces no
  monotone degradation, it measured nothing -> vacuous.
  P0 readiness: a below-floor intact readout self-routes to
  substrate_not_ready_requeue, never to a substrate verdict.
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib import consolidation_lesion_harness as H  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_sd068_consolidation_staged_damage_diagnostic"
QUEUE_ID = "V3-EXQ-778"
CLAIM_IDS: List[str] = ["SD-068", "MECH-168", "INV-047", "MECH-169"]
EXPERIMENT_PURPOSE = "diagnostic"
SLEEP_DRIVER_PATTERN = "manual-cycle-loop"

SEEDS = [42, 7, 123]
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
WARM_STEPS = 40

# Pre-registered acceptance floors.
SPAN_FLOOR = 1e-3            # min fractional-degradation span for a phase to count as "degrading"
MONOTONE_CORR_FLOOR = 0.5    # min Pearson corr(sigma, error) for "monotone degradation"
INTACT_SIGNAL_FLOOR = 1e-9   # sigma=0 readouts must be non-degenerate (positive control)
PASS_FRACTION = 2.0 / 3.0
PREDICTED_ORDER = list(H.REVERSE_DEPENDENCY_ORDER)  # ("rem","nrem","sws")


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return (num / (dx * dy)) if (dx > 1e-12 and dy > 1e-12) else 0.0


def _phase_error_frac(res: "H.StagedSweepResult", phase: str) -> Tuple[List[float], List[float]]:
    """Return (sigmas, fractional-degradation) for a phase from the swept integrity."""
    integ = res.integrity[phase]
    if phase == "sws":
        sig = integ.get("signal_power", [])
        noi = integ.get("noise_power", [])
        errs = [(n / s) if s > 1e-12 else float("nan") for s, n in zip(sig, noi)]
    elif phase == "nrem":
        fid = integ.get("transfer_fidelity", [])
        errs = [(1.0 - f) if f != H.UNAVAILABLE else float("nan") for f in fid]
    else:  # rem
        err = integ.get("calibration_error", [])
        cv = integ.get("clean_target_variance", [])
        errs = [(e / c) if (c > 1e-12) else float("nan") for e, c in zip(err, cv)]
    frac = H._normalise_degradation(errs)
    xs = [s for s, f in zip(res.sigmas, frac) if not math.isnan(f)]
    ys = [f for f in frac if not math.isnan(f)]
    return xs, ys


def _intact_nondegenerate(res: "H.StagedSweepResult") -> Dict[str, float]:
    """P0 positive control: at sigma=0 the readouts must be non-degenerate."""
    i0 = res.sigmas.index(0.0) if 0.0 in res.sigmas else 0
    sws_sig = res.integrity["sws"].get("signal_power", [0.0])[i0]
    nrem_gap = res.integrity["nrem"].get("gap_before", [0.0])[i0] if "gap_before" in res.integrity["nrem"] else 0.0
    rem_cv = res.integrity["rem"].get("clean_target_variance", [0.0])[i0]
    return {"sws_signal_power": float(sws_sig), "nrem_gap_before": float(nrem_gap), "rem_clean_variance": float(rem_cv)}


def _score_seed(res: "H.StagedSweepResult") -> Dict[str, Any]:
    monotone: Dict[str, bool] = {}
    corr: Dict[str, float] = {}
    span: Dict[str, float] = {}
    for phase in ("sws", "nrem", "rem"):
        xs, ys = _phase_error_frac(res, phase)
        c = _pearson(xs, ys)
        sp = (max(ys) - min(ys)) if ys else 0.0
        corr[phase] = c
        span[phase] = sp
        monotone[phase] = (c >= MONOTONE_CORR_FLOOR) and (sp >= SPAN_FLOOR)

    intact = _intact_nondegenerate(res)
    intact_ok = (
        intact["sws_signal_power"] > INTACT_SIGNAL_FLOOR
        and intact["rem_clean_variance"] > INTACT_SIGNAL_FLOOR
    )
    staging_computable = len(res.observed_order) == 3
    rem_contrast_computable = (
        res.gains.get("rem_passthrough_calibration_slope", H.UNAVAILABLE) != H.UNAVAILABLE
    )

    c1 = all(monotone.values())              # load-bearing: monotone degradation on all phases
    c2 = intact_ok                            # P0 positive control non-degenerate
    c3 = staging_computable and rem_contrast_computable
    return {
        "monotone": monotone,
        "corr": corr,
        "span": span,
        "intact": intact,
        "intact_ok": intact_ok,
        "C1_monotone_all_phases": c1,
        "C2_intact_nondegenerate": c2,
        "C3_staging_and_rem_contrast_computable": c3,
        "seed_pass": bool(c1 and c2 and c3),
        "observed_order": list(res.observed_order),
        "tolerance_sigma": {p: res.gains.get(f"tolerance_sigma_{p}", H.UNAVAILABLE) for p in ("sws", "nrem", "rem")},
        "rem_passthrough_slope": res.gains.get("rem_passthrough_calibration_slope", H.UNAVAILABLE),
        "rem_generative_slope": res.gains.get("rem_generative_output_slope", H.UNAVAILABLE),
        "rem_generative_available": res.gains.get("rem_generative_available", 0.0),
        "staging_matches_prediction": bool(res.staging_matches_prediction),
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    warm = 8 if dry_run else WARM_STEPS
    sigmas = [0.0, 1.0] if dry_run else SIGMAS
    print(f"V3-EXQ-778: SD-068 consolidation staged-damage diagnostic", flush=True)
    print(f"  seeds={seeds} sigmas={sigmas} warm_steps={warm} dry_run={dry_run}", flush=True)

    config_slice = {
        "sigmas": sigmas,
        "warm_steps": warm,
        "shy_decay_rate": 0.85,
        "body_obs_dim": H.BODY_OBS_DIM,
        "world_obs_dim": H.WORLD_OBS_DIM,
        "action_dim": H.ACTION_DIM,
        "harm_obs_dim": H.HARM_OBS_DIM,
    }

    arm_results: List[Dict[str, Any]] = []
    seed_scores: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition STAGED_SWEEP", flush=True)
        with arm_cell(
            seed,
            config_slice=config_slice,
            script_path=Path(__file__),
            config_slice_declared=True,
        ) as cell:
            res = H.run_staged_sweep(seed=seed, sigmas=list(sigmas), warm_steps=warm)
            score = _score_seed(res)
            row: Dict[str, Any] = {
                "seed": seed,
                "arm": "STAGED_SWEEP",
                "sigmas": list(res.sigmas),
                "integrity": res.integrity,
                "observed_order": score["observed_order"],
                "tolerance_sigma": score["tolerance_sigma"],
                "corr": score["corr"],
                "span": score["span"],
                "intact": score["intact"],
                "C1_monotone_all_phases": score["C1_monotone_all_phases"],
                "C2_intact_nondegenerate": score["C2_intact_nondegenerate"],
                "C3_staging_and_rem_contrast_computable": score["C3_staging_and_rem_contrast_computable"],
                "seed_pass": score["seed_pass"],
                "rem_passthrough_slope": score["rem_passthrough_slope"],
                "rem_generative_slope": score["rem_generative_slope"],
                "rem_generative_available": score["rem_generative_available"],
                "staging_matches_prediction": score["staging_matches_prediction"],
            }
            cell.stamp(row)
        arm_results.append(row)
        seed_scores.append(score)
        print(
            f"  [train] staged_sweep seed={seed} ep 1/1 "
            f"order={score['observed_order']} monotone_all={score['C1_monotone_all_phases']} "
            f"match_pred={score['staging_matches_prediction']}",
            flush=True,
        )
        print(f"verdict: {'PASS' if score['seed_pass'] else 'FAIL'}", flush=True)

    n = len(seed_scores)
    need = math.ceil(PASS_FRACTION * n)
    n_pass = sum(1 for s in seed_scores if s["seed_pass"])
    overall_pass = n_pass >= need

    # Staging summary across seeds (reported, NOT gated).
    from collections import Counter

    # Staging order is only DEFINED on seeds whose full 3-phase order was computable
    # (C3). A truncated observed_order cannot match the prediction by construction, so
    # pooling those seeds silently votes NO on seeds that were never eligible to vote --
    # and lets a 1- or 2-element tuple be elected modal_observed_order. Same
    # subgroup-superset defect as the V3-EXQ-778h C2 aggregation; scoped and the
    # exclusion emitted rather than left silent.
    staging_rows = [
        s for s in seed_scores if s["C3_staging_and_rem_contrast_computable"]
    ]
    staging_excluded_seeds = [
        s.get("seed") for s in seed_scores if not s["C3_staging_and_rem_contrast_computable"]
    ]
    n_staging = len(staging_rows)
    staging_need = math.ceil(PASS_FRACTION * n_staging)
    order_counter = Counter(tuple(s["observed_order"]) for s in staging_rows)
    modal_order, modal_count = (
        order_counter.most_common(1)[0] if order_counter else ((), 0)
    )
    n_match_pred = sum(1 for s in staging_rows if s["staging_matches_prediction"])

    # Interpretation: self-routed label (falsifiable via preconditions + non-degeneracy).
    intact_measured = min(
        min(s["intact"]["sws_signal_power"] for s in seed_scores),
        min(s["intact"]["rem_clean_variance"] for s in seed_scores),
    )
    substrate_ready = all(s["C2_intact_nondegenerate"] for s in seed_scores)
    if not substrate_ready:
        label = "substrate_not_ready_requeue"
    elif not overall_pass:
        label = "harness_nonmonotone_uninstrumented"
    elif n_staging and n_match_pred >= staging_need:
        label = "harness_operational_staging_matches_reverse_dependency"
    else:
        label = "harness_operational_staging_partial_or_inverted"

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "intact_readouts_nondegenerate",
                "description": "at sigma=0 the injected-content readouts (SWS signal_power, REM clean variance) are non-degenerate -- the P0 positive control that the phases actually operate",
                "measured": float(intact_measured),
                "threshold": INTACT_SIGNAL_FLOOR,
                # FLOOR-shaped, and STRICTLY so: C2 is `> INTACT_SIGNAL_FLOOR` per
                # seed, and `measured` is the min over seeds of both readouts, so
                # `measured > threshold` reproduces `met` exactly. Declared so the
                # indexer's recompute matches rather than defaulting (the 2026-06-07
                # V3-EXQ-648a/649 directionality bug).
                "comparator": ">",
                "direction": "lower",
                "control": "sigma=0 intact sweep point",
                "met": bool(substrate_ready),
            },
        ],
        "criteria_non_degenerate": {
            "C1_monotone_degradation": bool(all(s["C1_monotone_all_phases"] for s in seed_scores)),
            "C2_intact_nondegenerate": bool(substrate_ready),
            "C3_staging_and_rem_contrast_computable": bool(all(s["C3_staging_and_rem_contrast_computable"] for s in seed_scores)),
        },
        "criteria": [
            {"name": "C1_monotone_degradation_all_phases", "load_bearing": True, "passed": bool(overall_pass)},
        ],
        "staging_summary": {
            "predicted_reverse_dependency_order": PREDICTED_ORDER,
            "modal_observed_order": list(modal_order),
            "modal_order_seed_count": int(modal_count),
            "n_seeds_matching_prediction": int(n_match_pred),
            # Subgroup scoping, emitted so the narrowing is auditable.
            "staging_subgroup_predicate": "C3_staging_and_rem_contrast_computable",
            "staging_subgroup_n": int(n_staging),
            "staging_excluded_seeds": staging_excluded_seeds,
            "staging_match_threshold": int(staging_need),
            "note": "staging match is REPORTED, not gated; a partial-match/inversion is a valid diagnostic outcome. Order statistics are scoped to the C3-computable subgroup.",
        },
    }

    # Per-claim direction (diagnostic -> scoring-excluded; informational).
    direction = "supports" if overall_pass else "weakens"
    per_claim = {
        "SD-068": direction,   # the harness works as an instrument
        "MECH-168": "unknown",  # diagnostic; staging reported not adjudicated here
        "INV-047": "unknown",
        "MECH-169": "unknown",
    }

    print("", flush=True)
    print(f"seeds pass: {n_pass}/{n} (need {need}) -> overall {'PASS' if overall_pass else 'FAIL'}", flush=True)
    print(f"modal observed order: {list(modal_order)} (predicted {PREDICTED_ORDER}); seeds matching pred: {n_match_pred}/{n}", flush=True)
    print(f"self-route label: {label}", flush=True)

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "interpretation": interpretation,
        "arm_results": arm_results,
        "n_seeds_pass": n_pass,
        "need_seeds": need,
        "config": config_slice,
        "seeds": seeds,
    }


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    import time
    t0 = time.perf_counter()
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "interpretation": result["interpretation"],
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "n_seeds_pass": result["n_seeds_pass"],
        "need_seeds": result["need_seeds"],
        "acceptance_criteria": {
            "C1_monotone_degradation": f"corr(sigma,error)>={MONOTONE_CORR_FLOOR} AND span>={SPAN_FLOOR} on all 3 phases (LOAD-BEARING)",
            "C2_intact_nondegenerate": f"sigma=0 readouts > {INTACT_SIGNAL_FLOOR} (P0 positive control)",
            "C3_computable": "staged order (3 phases) + REM passthrough-vs-generative contrast computable",
            "pass_rule": f">= {PASS_FRACTION:.2f} of seeds; staging match REPORTED not gated",
        },
        "arm_results": result["arm_results"],
        "notes": (
            "SD-068 consolidation-pipeline lesion harness validation (diagnostic). Injected-content "
            "702-precedent; sidesteps 538a encoding-starvation ceiling. Non-vacuous via damage-tolerance "
            "staging order + REM passthrough-vs-generative contrast. MECH-121 NOT tagged for promotion "
            "(held; NREM leg is substrate-plumbing-fidelity only). Glymphatic half out of scope. "
            "GOV-REUSE-1: decisive readouts are new (harness did not exist) -> not recoverable, ran."
        ),
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=result["config"],
        seeds=result["seeds"],
        script_path=Path(__file__),
        started_at=t0,
        json_default=str,
    )
    if dry_run:
        print("[dry-run] manifest relocated out of evidence/ by emit_outcome", flush=True)
    else:
        print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=args.dry_run,
    )
    sys.exit(0)
