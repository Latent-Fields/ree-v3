"""
V3-EXQ-778b: SD-068 zero-injected-content NULL CONTROL for the consolidation-pipeline
lesion harness.
SLEEP DRIVER: manual-cycle-loop (the SD-068 harness drives enter_sws_mode /
run_sws_schema_pass + enter_rem_mode / run_rem_attribution_pass +
recalibrate_precision_to directly per phase readout; no SleepLoopManager scheduling).

WHAT THIS MEASURES
------------------
V3-EXQ-778 validated the SD-068 harness (PASS) and reported a seed-stable
damage-tolerance staging order of (nrem, rem, sws). That order is only meaningful if
each per-phase readout is tracking CONTENT FIDELITY. This run tests the alternative:
that a readout tracks PERTURBATION MAGNITUDE and would move with `sigma` even with no
known content injected at all -- in which case the staging order is an ordering of the
three phases' raw NOISE SENSITIVITY, not of their functional damage tolerance. That is
precisely the vacuity SD-068 claims to escape.

The control is the analog of the odour-contingency null in Bar et al. 2020 (Curr Biol,
DOI 10.1016/j.cub.2020.01.091) -- the methodological precedent SD-068 follows (inject
known content, apply a scoped perturbation, read out at the same scope). What made Bar
et al. convincing was that unilateral olfactory stimulation during sleep produced NO
memory effect and NO oscillatory effect when learning had occurred WITHOUT the
contextual odour: the perturbation alone does nothing; it acts only on injected
content. SD-068 had no such null until now
(lit entry: REE_assembly/evidence/literature/targeted_review_sd_068/entries/
2026-07-18_sd_068_local_tmr_injected_content_bar2020/).

METHOD. The identical sigma sweep [0.0, 0.25, 0.5, 1.0, 2.0] runs TWICE per seed on
the identical substrate, warm-up and RNG streams. Only the injected content differs:
  ARM INJECTED (content_scale=1.0) -- bit-identical to the V3-EXQ-778 sweep (verified).
  ARM NULL     (content_scale=0.0) -- no known content planted; each readout is
                                      exercised on noise alone.
The DELIVERED perturbation is held numerically IDENTICAL across arms (each readout
references its noise scale to the UNSCALED content), so the null arm is "same odour, no
prior pairing" rather than "weaker odour" -- otherwise the null would come out flat for
the wrong reason. Both arms are expressed in COMMON units using the INJECTED arm's
denominators and least-squares fitted against sigma.

REPORTED PER PHASE (a RATIO, not a bare pass/fail, so a PARTIAL null is visible):
    null_slope_ratio_<phase> = |null sigma-slope| / |injected sigma-slope|
      ~0.0 -> fully content-contingent (the readout is inert on noise)
      ~1.0 -> fully confounded (identical sigma-response with and without content)
    intermediate values are exactly that: intermediate.

A phase above NULL_SLOPE_RATIO_CEILING is CONFOUNDED. Confounded phases are NAMED and
flagged in the manifest and in the harness docstring's CONFOUND REGISTER; they are
NEVER silently dropped from the staging order -- dropping them would hide the confound
rather than surface it.

WHY DIAGNOSTIC (not evidence)
-----------------------------
This is an instrument-validity control on SD-068, not a test of a substrate
hypothesis. It does NOT weight governance confidence. It tags SD-068 (the harness whose
non-vacuity it audits) and the three staging claims the harness serves
(MECH-168 / INV-047 / MECH-169) as context. It deliberately does NOT tag MECH-121 --
MECH-121 is candidate/substrate_conditional (hold_pending_v3_substrate) and the NREM
leg here remains substrate-plumbing-fidelity on injected content, NOT MECH-121
behavioural validation.

ACCEPTANCE (pre-registered)
---------------------------
  C1 (LOAD-BEARING): all three phases content-contingent -- null_slope_ratio <=
     NULL_SLOPE_RATIO_CEILING (0.25) on >= PASS_FRACTION of seeds. PASS means the
     staging order survives the null control.
  C2 (readiness / non-degeneracy): the ratio is INTERPRETABLE on all three phases --
     the injected-arm slope (the ratio's denominator, the SAME statistic C1 routes on)
     clears INJECTED_SLOPE_FLOOR on the known-damaged positive control. A below-floor
     denominator means the sweep never damaged the readout, so the control cannot
     discriminate -> substrate_not_ready_requeue, NEVER a substrate verdict.
  C3: the confound verdict is stable across seeds.

A FAIL here is an INFORMATIVE outcome, not a broken run: it scopes SD-068's
non-vacuity honestly (the contract would then have to be carried by the REM
passthrough-vs-generative contrast alone) rather than withdrawing the claim.
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
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_sd068_null_content_control_diagnostic"
QUEUE_ID = "V3-EXQ-778b"
# SD-068 = the harness whose non-vacuity this audits. The three staging claims are
# tagged as context only. MECH-121 is deliberately ABSENT (held; the NREM leg is
# substrate-plumbing-fidelity only and must not accrue promotion evidence).
CLAIM_IDS: List[str] = ["SD-068", "MECH-168", "INV-047", "MECH-169"]
EXPERIMENT_PURPOSE = "diagnostic"
SLEEP_DRIVER_PATTERN = "manual-cycle-loop"

# Seeds 42 and 7: the pair on which V3-EXQ-778 reported a stable (nrem, rem, sws)
# order, so the control is directly comparable to the result it audits.
SEEDS = [42, 7]
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
WARM_STEPS = 40
ARMS = ["INJECTED", "NULL"]

# Pre-registered thresholds.
NULL_SLOPE_RATIO_CEILING = H.NULL_SLOPE_RATIO_CEILING  # 0.25
INJECTED_SLOPE_FLOOR = 1e-6   # C2 readiness: min |injected slope| for an interpretable ratio
PASS_FRACTION = 1.0           # both seeds must be clean for C1 (a control, not a vote)


def _fmt(v: float) -> str:
    """ASCII-safe float rendering that keeps UNAVAILABLE legible."""
    if v == H.UNAVAILABLE or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.4f}"


def _score_seed(control: Dict[str, float]) -> Dict[str, Any]:
    """Score one seed's null control. Returns per-phase ratios + criteria."""
    ratios: Dict[str, float] = {}
    inj_slopes: Dict[str, float] = {}
    null_slopes: Dict[str, float] = {}
    contingent: Dict[str, bool] = {}
    interpretable: Dict[str, bool] = {}

    for p in ("sws", "nrem", "rem"):
        ratios[p] = control.get(f"null_slope_ratio_{p}", H.UNAVAILABLE)
        inj_slopes[p] = control.get(f"injected_slope_{p}", H.UNAVAILABLE)
        null_slopes[p] = control.get(f"null_slope_{p}", H.UNAVAILABLE)
        contingent[p] = control.get(f"content_contingent_{p}", 0.0) >= 1.0
        # C2 readiness asserts the SAME statistic C1 routes on: the ratio's
        # denominator. A near-zero injected slope makes the ratio uninterpretable.
        inj = inj_slopes[p]
        interpretable[p] = (
            inj != H.UNAVAILABLE
            and not math.isnan(inj)
            and abs(inj) >= INJECTED_SLOPE_FLOOR
        )

    confounded = H.confounded_phase_names(control)
    c1 = all(contingent.values())
    c2 = all(interpretable.values())
    return {
        "rem_ratio_off_scale": bool(
            control.get("null_slope_ratio_rem_off_scale", 0.0) >= 1.0
        ),
        "null_rem_target_clamped_frac": float(
            control.get("null_rem_target_clamped_frac", 0.0)
        ),
        "null_slope_ratio": ratios,
        "injected_slope": inj_slopes,
        "null_slope": null_slopes,
        "content_contingent": contingent,
        "ratio_interpretable": interpretable,
        "confounded_phases": confounded,
        "C1_all_phases_content_contingent": c1,
        "C2_ratio_interpretable_all_phases": c2,
        "seed_pass": bool(c1 and c2),
        "min_injected_slope_abs": float(
            min(
                abs(v)
                for v in inj_slopes.values()
                if v != H.UNAVAILABLE and not math.isnan(v)
            )
            if any(
                v != H.UNAVAILABLE and not math.isnan(v) for v in inj_slopes.values()
            )
            else 0.0
        ),
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    warm = 8 if dry_run else WARM_STEPS
    sigmas = [0.0, 0.5, 2.0] if dry_run else SIGMAS

    print("V3-EXQ-778b: SD-068 zero-injected-content null control", flush=True)
    print(
        f"  seeds={seeds} sigmas={sigmas} warm_steps={warm} "
        f"arms={ARMS} dry_run={dry_run}",
        flush=True,
    )

    config_slice = {
        "sigmas": sigmas,
        "warm_steps": warm,
        "arms": list(ARMS),
        "null_slope_ratio_ceiling": NULL_SLOPE_RATIO_CEILING,
        "injected_slope_floor": INJECTED_SLOPE_FLOOR,
        "shy_decay_rate": 0.85,
        "body_obs_dim": H.BODY_OBS_DIM,
        "world_obs_dim": H.WORLD_OBS_DIM,
        "action_dim": H.ACTION_DIM,
        "harm_obs_dim": H.HARM_OBS_DIM,
    }

    arm_results: List[Dict[str, Any]] = []
    seed_scores: List[Dict[str, Any]] = []
    total_eps = len(sigmas)

    for seed in seeds:
        print(f"Seed {seed} Condition NULL_CONTENT_CONTROL", flush=True)
        with arm_cell(
            seed,
            config_slice=config_slice,
            script_path=Path(__file__),
            config_slice_declared=True,
        ) as cell:
            # Both arms swept together per sigma so the progress bar reflects real work.
            inj_pr: Dict[float, Dict[str, Dict[str, float]]] = {}
            null_pr: Dict[float, Dict[str, Dict[str, float]]] = {}
            for i, s in enumerate(sigmas):
                inj_pr[s] = H.phase_integrity_at_sigma(
                    seed=seed, sigma=s, warm_steps=warm, content_scale=1.0
                )
                null_pr[s] = H.phase_integrity_at_sigma(
                    seed=seed, sigma=s, warm_steps=warm, content_scale=0.0
                )
                print(
                    f"  [train] null_control seed={seed} ep {i + 1}/{total_eps} "
                    f"sigma={s}",
                    flush=True,
                )

            control = H.run_null_content_control(
                seed=seed,
                sigmas=list(sigmas),
                warm_steps=warm,
                injected_pr_by_sigma=inj_pr,
                null_pr_by_sigma=null_pr,
            )
            # The staging order this control audits, recomputed from the SAME
            # injected sweep so the two are exactly comparable.
            gains = H.error_propagation_gain(
                seed=seed,
                sigmas=list(sigmas),
                warm_steps=warm,
                pr_by_sigma=inj_pr,
            )
            observed_order = sorted(
                ("sws", "nrem", "rem"),
                key=lambda p: (
                    float("inf")
                    if gains.get(f"tolerance_sigma_{p}", H.UNAVAILABLE) == H.UNAVAILABLE
                    else float(gains[f"tolerance_sigma_{p}"]),
                    -float(gains.get(f"norm_degradation_slope_{p}", 0.0)),
                ),
            )

            score = _score_seed(control)
            row: Dict[str, Any] = {
                "seed": seed,
                "arm": "NULL_CONTENT_CONTROL",
                "arms_compared": list(ARMS),
                "sigmas": list(sigmas),
                "null_control": control,
                "null_slope_ratio": score["null_slope_ratio"],
                "injected_slope": score["injected_slope"],
                "null_slope": score["null_slope"],
                "content_contingent": score["content_contingent"],
                "ratio_interpretable": score["ratio_interpretable"],
                "confounded_phases": score["confounded_phases"],
                "observed_staging_order": observed_order,
                "tolerance_sigma": {
                    p: gains.get(f"tolerance_sigma_{p}", H.UNAVAILABLE)
                    for p in ("sws", "nrem", "rem")
                },
                # OFF/baseline (INJECTED) arm internals recorded as richly as the
                # NULL arm, per the Experimental Recording Standard.
                "integrity_injected": {
                    str(s): inj_pr[s] for s in sigmas
                },
                "integrity_null": {
                    str(s): null_pr[s] for s in sigmas
                },
                "C1_all_phases_content_contingent": score[
                    "C1_all_phases_content_contingent"
                ],
                "C2_ratio_interpretable_all_phases": score[
                    "C2_ratio_interpretable_all_phases"
                ],
                "seed_pass": score["seed_pass"],
            }
            cell.stamp(row)

        arm_results.append(row)
        seed_scores.append(score)

        r = score["null_slope_ratio"]
        print(
            f"  null_slope_ratio: sws={_fmt(r['sws'])} nrem={_fmt(r['nrem'])} "
            f"rem={_fmt(r['rem'])} (ceiling {NULL_SLOPE_RATIO_CEILING})",
            flush=True,
        )
        if score["rem_ratio_off_scale"]:
            print(
                f"  NOTE rem ratio is OFF-SCALE: the null arm's precision reference "
                f"hit the 1e-3 positivity floor on "
                f"{score['null_rem_target_clamped_frac']:.0%} of sigma points, so the "
                f"ratio reads 'structurally content-free', NOT a literal N-fold "
                f"noise sensitivity.",
                flush=True,
            )
        print(
            f"  confounded_phases: {score['confounded_phases'] or 'none'} "
            f"| staging order under audit: {observed_order}",
            flush=True,
        )
        print(f"verdict: {'PASS' if score['seed_pass'] else 'FAIL'}", flush=True)

    n = len(seed_scores)
    need = math.ceil(PASS_FRACTION * n)
    n_pass = sum(1 for s in seed_scores if s["seed_pass"])
    readiness_ok = all(s["C2_ratio_interpretable_all_phases"] for s in seed_scores)
    overall_pass = readiness_ok and n_pass >= need

    # Per-phase aggregation across seeds (mean ratio + confound stability).
    phase_summary: Dict[str, Any] = {}
    for p in ("sws", "nrem", "rem"):
        vals = [
            s["null_slope_ratio"][p]
            for s in seed_scores
            if s["null_slope_ratio"][p] != H.UNAVAILABLE
            and not math.isnan(s["null_slope_ratio"][p])
        ]
        n_conf = sum(1 for s in seed_scores if p in s["confounded_phases"])
        phase_summary[p] = {
            "mean_null_slope_ratio": (sum(vals) / len(vals)) if vals else H.UNAVAILABLE,
            "per_seed_null_slope_ratio": [s["null_slope_ratio"][p] for s in seed_scores],
            "n_seeds_confounded": n_conf,
            "confounded_all_seeds": bool(n_conf == n),
            "confound_verdict_stable": bool(n_conf == 0 or n_conf == n),
        }
    c3_stable = all(v["confound_verdict_stable"] for v in phase_summary.values())

    confounded_all = sorted(
        {p for s in seed_scores for p in s["confounded_phases"]}
    )

    # Self-routed label. A below-floor readiness measure routes ONLY to requeue.
    min_inj_slope = min(s["min_injected_slope_abs"] for s in seed_scores)
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
    elif overall_pass:
        label = "null_control_clean_staging_order_content_contingent"
    elif len(confounded_all) < 3:
        label = "null_control_partial_staging_order_partly_noise_sensitivity"
    else:
        label = "null_control_failed_staging_order_is_noise_sensitivity_ordering"

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "injected_arm_sigma_slope_supra_floor",
                "description": (
                    "the ratio's DENOMINATOR -- the injected-arm sigma-slope, the same "
                    "statistic C1 routes on -- clears the floor on the known-damaged "
                    "positive control, so null/injected is interpretable. Below floor "
                    "means the sweep never damaged the readout and the control cannot "
                    "discriminate content-contingency from inertness."
                ),
                "measured": float(min_inj_slope),
                "threshold": INJECTED_SLOPE_FLOOR,
                "direction": "lower",
                "control": "injected arm swept to sigma=max (known-damaged positive control)",
                "met": bool(readiness_ok),
            },
        ],
        "criteria_non_degenerate": {
            # Each is False if it passed/failed for a degenerate reason.
            "C1_all_phases_content_contingent": bool(readiness_ok),
            "C2_ratio_interpretable_all_phases": bool(
                min_inj_slope > 0.0
            ),
            "C3_confound_verdict_seed_stable": bool(len(seed_scores) >= 2),
        },
        "criteria": [
            {
                "name": "C1_all_phases_content_contingent",
                "load_bearing": True,
                "passed": bool(overall_pass),
            },
            {
                "name": "C2_ratio_interpretable_all_phases",
                "load_bearing": False,
                "passed": bool(readiness_ok),
            },
            {
                "name": "C3_confound_verdict_seed_stable",
                "load_bearing": False,
                "passed": bool(c3_stable),
            },
        ],
        "confound_register": {
            "confounded_phases": confounded_all,
            "per_phase": phase_summary,
            "ceiling": NULL_SLOPE_RATIO_CEILING,
            "rem_ratio_off_scale": bool(
                any(s["rem_ratio_off_scale"] for s in seed_scores)
            ),
            "rem_off_scale_note": (
                "When the rem null arm's precision reference collapses onto the 1e-3 "
                "positivity floor, 1/1e-3 = 1000 dominates the calibration error, so a "
                "large rem null_slope_ratio is OFF-SCALE -- read it as 'this leg is "
                "structurally content-free' (it is passthrough by construction at "
                "step=1.0), NEVER as a calibrated N-fold noise sensitivity."
            ),
            "note": (
                "A confounded phase's readout moves with sigma even with NO injected "
                "content, so its contribution to the V3-EXQ-778 damage-tolerance order "
                "(nrem, rem, sws) is raw noise sensitivity rather than functional damage "
                "tolerance. Confounded phases are REPORTED and flagged, never dropped "
                "from the staging order. Recorded in the harness module docstring's "
                "CONFOUND REGISTER."
            ),
        },
    }

    # Direction: SD-068's non-vacuity contract is what this audits. The three staging
    # claims are context only (diagnostic -> scoring-excluded regardless).
    if not readiness_ok:
        sd068_direction = "unknown"
    elif overall_pass:
        sd068_direction = "supports"
    else:
        sd068_direction = "weakens"
    per_claim = {
        "SD-068": sd068_direction,
        "MECH-168": "unknown",
        "INV-047": "unknown",
        "MECH-169": "unknown",
    }

    print("", flush=True)
    print(
        f"seeds pass: {n_pass}/{n} (need {need}) -> overall "
        f"{'PASS' if overall_pass else 'FAIL'}",
        flush=True,
    )
    for p in ("sws", "nrem", "rem"):
        ps = phase_summary[p]
        print(
            f"  {p:>4}: mean null_slope_ratio={_fmt(ps['mean_null_slope_ratio'])} "
            f"confounded_seeds={ps['n_seeds_confounded']}/{n} "
            f"stable={ps['confound_verdict_stable']}",
            flush=True,
        )
    print(f"confounded phases (any seed): {confounded_all or 'none'}", flush=True)
    print(f"self-route label: {label}", flush=True)

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": sd068_direction,
        "evidence_direction_per_claim": per_claim,
        "interpretation": interpretation,
        "arm_results": arm_results,
        "phase_summary": phase_summary,
        "confounded_phases": confounded_all,
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
        "phase_summary": result["phase_summary"],
        "confounded_phases": result["confounded_phases"],
        "acceptance_criteria": {
            "C1_all_phases_content_contingent": (
                f"null_slope_ratio <= {NULL_SLOPE_RATIO_CEILING} on all 3 phases, "
                f"on >= {PASS_FRACTION:.2f} of seeds (LOAD-BEARING)"
            ),
            "C2_ratio_interpretable": (
                f"|injected sigma-slope| >= {INJECTED_SLOPE_FLOOR} on all 3 phases "
                "(readiness; asserts the ratio's denominator, the same statistic C1 "
                "routes on; below-floor -> substrate_not_ready_requeue)"
            ),
            "C3_seed_stable": "confound verdict identical across seeds",
        },
        "arm_results": result["arm_results"],
        "notes": (
            "SD-068 zero-injected-content NULL CONTROL (diagnostic). Analog of the "
            "odour-contingency null in Bar et al. 2020 (Curr Biol, DOI "
            "10.1016/j.cub.2020.01.091), the methodological precedent SD-068 follows. "
            "Audits whether the V3-EXQ-778 damage-tolerance staging order (nrem, rem, "
            "sws) reflects content fidelity or raw noise sensitivity. Identical sigma "
            "sweep with content_scale 1.0 vs 0.0; delivered perturbation held "
            "numerically identical across arms via diffuse_perturb(rms_ref=...). "
            "Injected arm verified BIT-IDENTICAL to the pre-778b harness, so 778 is "
            "unaffected and the arms are directly comparable. Confounded phases are "
            "REPORTED and flagged in the harness CONFOUND REGISTER, never dropped from "
            "the staging order. MECH-121 deliberately NOT tagged (held; NREM leg is "
            "substrate-plumbing-fidelity only). Experiment-layer only; zero ree_core "
            "change. GOV-REUSE-1: decisive readout (null-arm sigma-slope) carried by 0 "
            "of 3 SD-068 manifests on substrate_hash 3bafe754/e9a22a91 -- the null arm "
            "has never run, not recoverable -> ran."
        ),
    }

    stamp_recording_core(
        manifest,
        config=result["config"],
        seeds=result["seeds"],
        script_path=Path(__file__),
        started_at=t0,
    )

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
