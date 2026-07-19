"""
V3-EXQ-778a: SD-068 consolidation-pipeline staged-damage POWER-UP DIAGNOSTIC.
SLEEP DRIVER: manual-cycle-loop (the SD-068 harness drives enter_sws_mode /
run_sws_schema_pass + enter_rem_mode / run_rem_attribution_pass +
recalibrate_precision_to directly per phase readout; no SleepLoopManager scheduling).

Successor to V3-EXQ-778 (which validated the SD-068 consolidation-lesion harness
operational, 3/3 seeds, but found the staged-failure ORDER seed-variable with n=3:
per-seed orders (rem,nrem,sws) / (nrem,sws,rem) / (nrem,rem,sws), only 1/3 matching
the predicted reverse-dependency order). With n=3 that ambiguity is unresolvable:
is the reverse-dependency staging genuinely NOT supported, or merely under-powered?

WHAT THIS ADDS OVER 778 (the whole point):
  1. 8 seeds (the original 3 + 5 new) instead of 3 -> a real n for the ordering.
  2. A PER-PHASE damage-TOLERANCE distribution across seeds (mean, SD, 95% CI).
  3. A STATISTICAL staging test, three complementary ways:
     (a) per-seed Spearman rank-correlation of the observed tolerance ranking vs the
         predicted (rem,nrem,sws) order; aggregate mean rho + 95% CI + sign test.
     (b) the two ADJACENCY predictions as paired per-seed tolerance differences --
         rem-fails-first (nrem_tol - rem_tol > 0) and nrem-before-sws
         (sws_tol - nrem_tol > 0) -- each with mean, 95% CI, and an exact sign test.
     (c) Kendall's W concordance (+ Friedman chi-square) across seeds: do the seeds
         agree on ANY consistent phase ordering at all, independent of the prediction?
  Everything else (harness, sigma grid, per-phase readouts, arm fingerprinting,
  recording core) is bit-identical to 778 so the two runs pool directly.

WHY DIAGNOSTIC (not evidence) -- unchanged from 778:
  SD-068 substrate-readiness / staging characterisation. Does NOT weight governance
  confidence. Tags MECH-168/INV-047/MECH-169 (the staged-decline claims the harness
  serves) and SD-068 (the harness itself). Deliberately does NOT tag MECH-121 for
  promotion -- MECH-121 is candidate/substrate_conditional (hold_pending_v3_substrate)
  and the NREM leg is substrate-plumbing-fidelity on injected content, NOT MECH-121
  behavioural validation. The glymphatic/amyloid structural half of MECH-169 has no
  V3 analog and is out of scope.

WHAT THE STATISTICAL VERDICT MEANS (REPORTED, never gated):
  A robust match (mean Spearman rho CI > 0 AND both adjacency CIs > 0) strengthens
  the INV-047/MECH-168 staged-decline prediction. A robust rejection of rem-first
  (the rem<nrem adjacency CI entirely < 0) is itself an informative finding: the V3
  substrate's consolidation phases do NOT show reverse-dependency staging under
  uniform damage. A CI straddling 0 on the contested axis = genuinely seed-variable /
  under-powered even at n=8 (also informative -- bounds the effect size).

ACCEPTANCE (diagnostic, non-vacuous -- PASS means "the harness is a working
instrument across seeds AND the staging question now has a real n", NOT "the staging
prediction is confirmed"):
  On >= 2/3 of seeds: (C1) each of the three phases shows MONOTONE degradation across
  sigma (positive sigma-error correlation AND non-trivial span), (C2) the intact
  (sigma=0) readouts are non-degenerate (the P0 positive control), and (C3) the
  staged-failure order + REM passthrough-vs-generative contrast are computable.
  The statistical staging test is ALWAYS computed and REPORTED; the match vs the
  reverse-dependency prediction is never gated -- a robust non-match / seed-variable
  result is a VALID diagnostic outcome.
  Load-bearing criterion: C1 (monotone degradation). If the harness produces no
  monotone degradation, it measured nothing -> vacuous.
  P0 readiness: a below-floor intact readout self-routes to
  substrate_not_ready_requeue, never to a substrate verdict.
"""

import argparse
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib import consolidation_lesion_harness as H  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_sd068_consolidation_staging_power_diagnostic"
QUEUE_ID = "V3-EXQ-778a"
SUPERSEDES = "V3-EXQ-778"
CLAIM_IDS: List[str] = ["SD-068", "MECH-168", "INV-047", "MECH-169"]
EXPERIMENT_PURPOSE = "diagnostic"
SLEEP_DRIVER_PATTERN = "manual-cycle-loop"

# 8 seeds: the original 778 three (42, 7, 123) first for direct poolability, then 5 new.
SEEDS = [42, 7, 123, 2024, 99, 7777, 314, 1000]
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]  # identical grid to 778 (poolable)
WARM_STEPS = 40
PHASES = ("sws", "nrem", "rem")

# Pre-registered acceptance floors (identical to 778).
SPAN_FLOOR = 1e-3            # min fractional-degradation span for a phase to count as "degrading"
MONOTONE_CORR_FLOOR = 0.5    # min Pearson corr(sigma, error) for "monotone degradation"
INTACT_SIGNAL_FLOOR = 1e-9   # sigma=0 readouts must be non-degenerate (positive control)
PASS_FRACTION = 2.0 / 3.0
PREDICTED_ORDER = list(H.REVERSE_DEPENDENCY_ORDER)  # ("rem","nrem","sws")

# Pre-registered thresholds for the statistical staging conclusion (REPORTED, not gated).
# A "robust" call requires the 95% CI to exclude 0 in the relevant direction.
STAGING_CI_LEVEL = 0.95


# --------------------------------------------------------------------------- #
# Small pure-python stats (no scipy; mirrors the harness's pure-python style)  #
# --------------------------------------------------------------------------- #

# Two-sided t-critical values at 95% by degrees of freedom (df = n-1).
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 20: 2.086, 30: 2.042,
}


def _t95(df: int) -> float:
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    lower = [k for k in keys if k <= df]
    if lower:
        return _T95[max(lower)]
    return 1.96


def _mean(xs: List[float]) -> float:
    return (sum(xs) / len(xs)) if xs else float("nan")


def _std(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _ci95(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"))
    m = _mean(xs)
    se = _std(xs) / math.sqrt(n)
    h = _t95(n - 1) * se
    return (m - h, m + h)


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


def _rank_asc(vals: List[float]) -> List[float]:
    """Ascending ranks (1 = smallest), average-ranked on ties. NaN sorts last."""
    idx = sorted(range(len(vals)), key=lambda i: (math.isnan(vals[i]), vals[i]))
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def _spearman(a: List[float], b: List[float]) -> float:
    return _pearson(_rank_asc(a), _rank_asc(b))


def _sign_test(diffs: List[float]) -> Dict[str, Any]:
    """Two-sided exact binomial sign test of median(diffs) == 0."""
    from math import comb
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"n_pos": 0, "n_neg": 0, "n_effective": 0, "p_two_sided": 1.0}
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(0, k + 1))
    p = min(1.0, 2.0 * tail / (2 ** n))
    return {"n_pos": int(pos), "n_neg": int(neg), "n_effective": int(n), "p_two_sided": float(p)}


def _kendall_w(rank_rows: List[List[float]], k: int) -> Dict[str, float]:
    """Kendall's coefficient of concordance W (+ Friedman chi-square) across seeds.

    rank_rows: one per-seed rank vector of length k (ascending ranks over the k phases).
    W in [0, 1]: 1 == perfect agreement on a single ordering, 0 == no agreement.
    Friedman chi2 = m*(k-1)*W with df = k-1.
    """
    m = len(rank_rows)
    if m < 1 or k < 2:
        return {"W": float("nan"), "friedman_chi2": float("nan"), "df": float(k - 1), "m_raters": float(m)}
    col_sums = [sum(row[j] for row in rank_rows) for j in range(k)]
    rbar = sum(col_sums) / k
    S = sum((c - rbar) ** 2 for c in col_sums)
    denom = (m ** 2) * (k ** 3 - k)
    W = (12.0 * S / denom) if denom > 0 else float("nan")
    chi2 = m * (k - 1) * W if not math.isnan(W) else float("nan")
    return {"W": float(W), "friedman_chi2": float(chi2), "df": float(k - 1), "m_raters": float(m)}


# --------------------------------------------------------------------------- #
# Per-seed scoring (identical logic to 778)                                   #
# --------------------------------------------------------------------------- #

def _phase_error_frac(res: "H.StagedSweepResult", phase: str) -> Tuple[List[float], List[float]]:
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
    i0 = res.sigmas.index(0.0) if 0.0 in res.sigmas else 0
    sws_sig = res.integrity["sws"].get("signal_power", [0.0])[i0]
    nrem_gap = res.integrity["nrem"].get("gap_before", [0.0])[i0] if "gap_before" in res.integrity["nrem"] else 0.0
    rem_cv = res.integrity["rem"].get("clean_target_variance", [0.0])[i0]
    return {"sws_signal_power": float(sws_sig), "nrem_gap_before": float(nrem_gap), "rem_clean_variance": float(rem_cv)}


def _score_seed(res: "H.StagedSweepResult") -> Dict[str, Any]:
    monotone: Dict[str, bool] = {}
    corr: Dict[str, float] = {}
    span: Dict[str, float] = {}
    for phase in PHASES:
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

    c1 = all(monotone.values())
    c2 = intact_ok
    c3 = staging_computable and rem_contrast_computable
    tol = {p: res.gains.get(f"tolerance_sigma_{p}", H.UNAVAILABLE) for p in PHASES}
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
        "tolerance_sigma": tol,
        "rem_passthrough_slope": res.gains.get("rem_passthrough_calibration_slope", H.UNAVAILABLE),
        "rem_generative_slope": res.gains.get("rem_generative_output_slope", H.UNAVAILABLE),
        "rem_generative_available": res.gains.get("rem_generative_available", 0.0),
        "staging_matches_prediction": bool(res.staging_matches_prediction),
    }


# --------------------------------------------------------------------------- #
# Cross-seed statistical staging test (the new content)                       #
# --------------------------------------------------------------------------- #

def _tolerance_distribution(seed_scores: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-phase damage-tolerance distribution across seeds (only real, available values)."""
    dist: Dict[str, Dict[str, Any]] = {}
    for phase in PHASES:
        vals = [
            float(s["tolerance_sigma"][phase])
            for s in seed_scores
            if s["tolerance_sigma"].get(phase, H.UNAVAILABLE) != H.UNAVAILABLE
        ]
        lo, hi = _ci95(vals)
        dist[phase] = {
            "values": vals,
            "n": len(vals),
            "mean": _mean(vals),
            "std": _std(vals),
            "ci95_low": lo,
            "ci95_high": hi,
        }
    return dist


def _staging_statistics(seed_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Three complementary staging tests over the per-seed tolerances.

    Predicted reverse-dependency order = (rem, nrem, sws): rem fails FIRST (lowest
    damage-tolerance), sws LAST (highest). Only seeds where all three phase
    tolerances are available contribute to the paired/rank tests.
    """
    # Predicted tolerance RANK per phase (ascending: rem fails first -> rank 1 ... sws last -> rank 3).
    # PREDICTED_ORDER = (rem, nrem, sws): i=0 rem -> 1, i=1 nrem -> 2, i=2 sws -> 3.
    pred_rank = {p: (i + 1) for i, p in enumerate(PREDICTED_ORDER)}
    pred_vec = [pred_rank[p] for p in PHASES]  # ordered as PHASES = (sws, nrem, rem) -> [3, 2, 1]

    per_seed_rho: List[float] = []
    rank_rows: List[List[float]] = []          # per-seed ascending tolerance ranks (for Kendall W)
    d_rem_first: List[float] = []              # nrem_tol - rem_tol  (>0 supports rem-fails-first)
    d_nrem_before_sws: List[float] = []        # sws_tol - nrem_tol  (>0 supports nrem-before-sws)

    for s in seed_scores:
        tol = s["tolerance_sigma"]
        if any(tol.get(p, H.UNAVAILABLE) == H.UNAVAILABLE for p in PHASES):
            continue
        obs = [float(tol[p]) for p in PHASES]   # ordered as PHASES
        per_seed_rho.append(_spearman(obs, pred_vec))
        rank_rows.append(_rank_asc(obs))
        d_rem_first.append(float(tol["nrem"]) - float(tol["rem"]))
        d_nrem_before_sws.append(float(tol["sws"]) - float(tol["nrem"]))

    rho_lo, rho_hi = _ci95(per_seed_rho)
    d1_lo, d1_hi = _ci95(d_rem_first)
    d2_lo, d2_hi = _ci95(d_nrem_before_sws)

    spearman = {
        "per_seed_rho": per_seed_rho,
        "mean_rho": _mean(per_seed_rho),
        "ci95_low": rho_lo,
        "ci95_high": rho_hi,
        "sign_test": _sign_test(per_seed_rho),
        "note": "rho of observed tolerance ranking vs predicted (rem,nrem,sws); +1 = exact match, -1 = full inversion",
    }
    adjacency = {
        "rem_fails_first": {  # nrem_tol - rem_tol > 0
            "diffs": d_rem_first,
            "mean": _mean(d_rem_first),
            "ci95_low": d1_lo,
            "ci95_high": d1_hi,
            "sign_test": _sign_test(d_rem_first),
            "note": "nrem_tolerance - rem_tolerance; >0 => REM fails at lower damage (prediction). This is the CONTESTED axis in 778.",
        },
        "nrem_before_sws": {  # sws_tol - nrem_tol > 0
            "diffs": d_nrem_before_sws,
            "mean": _mean(d_nrem_before_sws),
            "ci95_low": d2_lo,
            "ci95_high": d2_hi,
            "sign_test": _sign_test(d_nrem_before_sws),
            "note": "sws_tolerance - nrem_tolerance; >0 => NREM fails before SWS (prediction). 778 suggested this is robust.",
        },
    }
    concordance = _kendall_w(rank_rows, len(PHASES))

    # Robustness calls (REPORTED, never gated). CI-based, direction-aware.
    rho_robust_match = (not math.isnan(rho_lo)) and rho_lo > 0.0
    rem_first_robust_support = (not math.isnan(d1_lo)) and d1_lo > 0.0
    rem_first_robust_reject = (not math.isnan(d1_hi)) and d1_hi < 0.0
    nrem_before_sws_robust = (not math.isnan(d2_lo)) and d2_lo > 0.0

    return {
        "n_seeds_all_phases_available": len(per_seed_rho),
        "predicted_order": PREDICTED_ORDER,
        "spearman": spearman,
        "adjacency": adjacency,
        "concordance": concordance,
        "robustness": {
            "spearman_ci_excludes_zero_positive": bool(rho_robust_match),
            "rem_fails_first_ci_supports": bool(rem_first_robust_support),
            "rem_fails_first_ci_rejects": bool(rem_first_robust_reject),
            "nrem_before_sws_ci_supports": bool(nrem_before_sws_robust),
        },
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:2] if dry_run else SEEDS
    warm = 8 if dry_run else WARM_STEPS
    sigmas = [0.0, 1.0] if dry_run else SIGMAS
    print("V3-EXQ-778a: SD-068 consolidation staged-damage POWER-UP diagnostic", flush=True)
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
        # One run per seed, one episode per run (matches 778): denominator M=1, seeds=8 runs.
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

    # Per-phase tolerance distribution + the cross-seed statistical staging test.
    tol_dist = _tolerance_distribution(seed_scores)
    staging_stats = _staging_statistics(seed_scores)

    # Staging summary across seeds (reported, NOT gated).
    order_counter = Counter(tuple(s["observed_order"]) for s in seed_scores)
    modal_order, modal_count = order_counter.most_common(1)[0]
    n_match_pred = sum(1 for s in seed_scores if s["staging_matches_prediction"])

    intact_measured = min(
        min(s["intact"]["sws_signal_power"] for s in seed_scores),
        min(s["intact"]["rem_clean_variance"] for s in seed_scores),
    )
    substrate_ready = all(s["C2_intact_nondegenerate"] for s in seed_scores)

    rob = staging_stats["robustness"]
    if not substrate_ready:
        label = "substrate_not_ready_requeue"
    elif not overall_pass:
        label = "harness_nonmonotone_uninstrumented"
    elif rob["spearman_ci_excludes_zero_positive"] and rob["rem_fails_first_ci_supports"] and rob["nrem_before_sws_ci_supports"]:
        label = "staging_robustly_matches_reverse_dependency"
    elif rob["rem_fails_first_ci_rejects"]:
        # REM is robustly MORE damage-tolerant than NREM -> rem-first is falsified.
        label = "staging_robustly_rejects_rem_first"
    else:
        label = "staging_seed_variable_underpowered"

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
            "n_seeds": int(n),
            "note": "staging match is REPORTED, not gated; a robust non-match / seed-variability is a valid diagnostic outcome.",
        },
        "tolerance_distribution": tol_dist,
        "staging_statistics": staging_stats,
    }

    direction = "supports" if overall_pass else "weakens"
    per_claim = {
        "SD-068": direction,   # the harness works as an instrument
        "MECH-168": "unknown",  # diagnostic; staging reported, not adjudicated here
        "INV-047": "unknown",
        "MECH-169": "unknown",
    }

    print("", flush=True)
    print(f"seeds pass: {n_pass}/{n} (need {need}) -> overall {'PASS' if overall_pass else 'FAIL'}", flush=True)
    print(f"modal observed order: {list(modal_order)} (predicted {PREDICTED_ORDER}); seeds matching pred: {n_match_pred}/{n}", flush=True)
    for phase in PHASES:
        d = tol_dist[phase]
        print(
            f"  tolerance[{phase}] mean={d['mean']:.4f} sd={d['std']:.4f} "
            f"ci95=[{d['ci95_low']:.4f},{d['ci95_high']:.4f}] n={d['n']}",
            flush=True,
        )
    sp = staging_stats["spearman"]
    a1 = staging_stats["adjacency"]["rem_fails_first"]
    a2 = staging_stats["adjacency"]["nrem_before_sws"]
    kw = staging_stats["concordance"]
    print(
        f"  spearman mean_rho={sp['mean_rho']:.4f} ci95=[{sp['ci95_low']:.4f},{sp['ci95_high']:.4f}] "
        f"sign_p={sp['sign_test']['p_two_sided']:.4f}",
        flush=True,
    )
    print(
        f"  adj rem_fails_first mean={a1['mean']:.4f} ci95=[{a1['ci95_low']:.4f},{a1['ci95_high']:.4f}] "
        f"sign_p={a1['sign_test']['p_two_sided']:.4f}",
        flush=True,
    )
    print(
        f"  adj nrem_before_sws mean={a2['mean']:.4f} ci95=[{a2['ci95_low']:.4f},{a2['ci95_high']:.4f}] "
        f"sign_p={a2['sign_test']['p_two_sided']:.4f}",
        flush=True,
    )
    print(f"  kendall_W={kw['W']:.4f} friedman_chi2={kw['friedman_chi2']:.4f} (df={int(kw['df'])})", flush=True)
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
        "supersedes": SUPERSEDES,
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
            "pass_rule": f">= {PASS_FRACTION:.2f} of seeds; staging match + statistical test REPORTED not gated",
        },
        "arm_results": result["arm_results"],
        "notes": (
            "SD-068 consolidation-lesion harness POWER-UP diagnostic (successor to V3-EXQ-778). "
            "8 seeds (778's 3 + 5 new) resolve the seed-variable staging order 778 found at n=3. "
            "Adds per-phase damage-tolerance distribution (mean/SD/95% CI) + a 3-way statistical "
            "staging test: per-seed Spearman rho vs predicted (rem,nrem,sws) with sign test, the two "
            "adjacency predictions (rem-fails-first / nrem-before-sws) as paired tolerance-diff CIs + "
            "sign tests, and Kendall's W concordance. Same harness/sigma-grid/readouts as 778 -> pools "
            "directly. MECH-121 NOT tagged for promotion (held; NREM leg is substrate-plumbing-fidelity "
            "only). Glymphatic half of MECH-169 out of scope. Diagnostic -> non-scoring; staging verdict "
            "REPORTED, never gated. GOV-REUSE-1: n>=6 tolerance distribution is a new manipulation not "
            "recoverable from the 3-seed 778 manifest -> ran."
        ),
    }

    # write_flat_manifest internally calls stamp_recording_core AFTER arm_results is
    # assembled (pack_writer.py:369), so the multi-arm substrate_hash hoists from the
    # per-cell fingerprints -- matching the validated 778 pattern.
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
