"""
V3-EXQ-778e: SD-068 GOV-FANOUT-1 portfolio leg 2 of 3 -- REM de-clamped readout.
Hypothesis under test: H-rem-genuinely-content-free (axis: REPRESENTATION).
SLEEP DRIVER: manual-cycle-loop (the SD-068 harness drives recalibrate_precision_to +
enter_rem_mode / run_rem_attribution_pass directly per phase readout; no
SleepLoopManager scheduling).

WHAT THIS MEASURES
------------------
V3-EXQ-778c's REM leg came back DEGENERATE AT BOTH RAILS: `null_slope_ratio_rem` exactly
0.0 on 5/8 seeds (the null arm's `calibration_error` pinned at the constant
998.5009992509989 with `target_clamped` 1.0 -- an identically ZERO slope) and off-scale
1801-9143 on 3/8 (the null precision reference collapsed onto the 1e-3 positivity floor,
so 1/1e-3 = 1000 dominates). mean 1911.6, sd 3306.1, 95% CI [-379, 4203],
`ceiling_inside_ci95` true, `confound_verdict_stable` false. The 5 apparently-clean seeds
are clean only BY DEGENERACY -- a zero slope from a saturated constant is the absence of
a measurement, not evidence of content-contingency.

This is leg 2 of the GOV-FANOUT-1 discrimination portfolio routed by
`REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-778c_2026-07-18.json`
(`targets[0].fanout_recommendation`). It attacks the REPRESENTATION axis -- how the error
is REPRESENTED -- holding everything else at the exact 778c condition. Siblings:
V3-EXQ-778d (measurement: how the null is operationalised) and V3-EXQ-778f (observation:
the generative-gain readout). They run in PARALLEL, not in sequence, because a single
re-posed probe can inherit the prior confound and return a confident-but-wrong verdict.

HYPOTHESES UNDER TEST
---------------------
H-rem-genuinely-content-free (THIS LEG, axis=representation, pre-registered
  2026-07-18T08:41:15Z in hypothesis_space_registry.v1.json question
  `consolidation_readout_validity`):
    CLAIM: the rem calibration readout is GENUINELY content-independent -- like the sws
    leg, which V3-EXQ-778c eliminated at null_slope_ratio 1.0000 (sd 2.7e-8, 8/8 seeds)
    -- and the clamping is incidental. On this reading a de-clamped readout would STILL
    show a null slope ratio near 1.0, making rem a SECOND content-free leg and leaving
    SD-068's non-vacuity contract carried by the nrem leg alone.
    PROBE: score the rem leg against the injected precision target DIRECTLY, in
    PRECISION units, rather than via `running_variance_after` in variance units.
      calibration_error   (778c) = |rv_after - 1/clean_target|            [variance units]
      direct_precision_error (this) = |1/rv_after - clean_target| / clean_target
    Inverting back to precision UNDOES the 1/x blow-up: when the null reference lands on
    the 1e-3 floor the variance-units readout jumps to 999.001 and pins, whereas the
    precision-units readout is bounded and its sigma-response stays graded on both arms.
    Everything else is held at the exact 778c condition (step=1.0, zero-content null,
    same 8 seeds, same sigma grid, same RNG streams), so the READOUT DEFINITION is the
    ONLY thing that differs.
    EVIDENCE FOR: the de-clamped readout yields null_slope_ratio > ceiling -- the rem leg
      responds to sigma just as strongly with the content removed, so it is measuring
      perturbation magnitude, not content fidelity. rem joins sws as content-free.
    EVIDENCE AGAINST (DECLARED NULL): if the de-clamped readout yields
      null_slope_ratio <= NULL_SLOPE_RATIO_CEILING (0.25), this leg is REFUTED -- the rem
      readout DOES track content once it is scored in a non-degenerate representation,
      and the 778c degeneracy was an artifact rather than content-freeness.

INTERPRETATION GRID (self-routed label; a HYPOTHESIS, not a verdict)
--------------------------------------------------------------------
  readiness UNMET (injected-arm slope of the DE-CLAMPED readout below floor)
        -> `substrate_not_ready_requeue`  [NEVER a substrate verdict]
  de-clamped null series STILL degenerate (constant / near-zero spread)
        -> `measurement_still_degenerate_requeue`  [NEVER a substrate verdict]
           The representation swap did not buy a graded null. Nothing is adjudicated.
  de-clamped ratio > ceiling, stable across seeds
        -> `rem_readout_genuinely_content_free`
           H-rem-genuinely-content-free SUPPORTED; the parent H-rem-content-contingent
           is WEAKENED.
  de-clamped ratio <= ceiling, stable across seeds
        -> `rem_readout_content_contingent_when_declamped`
           H-rem-genuinely-content-free REFUTED (its declared null); the parent
           H-rem-content-contingent is SUPPORTED.
  verdict not stable across seeds (ceiling inside the 95% CI)
        -> `rem_declamped_verdict_seed_unstable_underpowered`
           Explicitly NOT a verdict either way -- the 778c failure mode, surfaced rather
           than papered over.

ANTI-ALIAS / DESIGN-AUDIT NOTES (GOV-FANOUT-1 step 4)
-----------------------------------------------------
1. WITHIN-RUN READOUT CONTROL. Both readouts are scored on the SAME cells in the SAME
   run: `calibration_error` (the 778c readout) and `direct_precision_error` (the
   de-clamped one). The 778c readout's ratio therefore acts as a replication anchor, and
   any difference between the two is attributable to the readout definition ALONE -- not
   to a run-to-run difference. Without this, a de-clamped ratio near 1.0 would alias
   "the readout is genuinely content-free" against "this run differs from 778c".
2. NON-DEGENERACY GATE ON THE NULL ARM ITSELF. A ratio of ~0.0 is ambiguous between "the
   readout is inert on noise" (content-contingent -- the finding) and "the null series is
   a saturated constant" (degenerate -- no finding). That ambiguity is exactly what left
   778c unresolved, so this leg gates on the NULL series' OWN spread and distinct-value
   count and routes a still-degenerate null to a requeue label, never to a verdict. A
   declared null that can be satisfied by a new degeneracy is not a declared null.
3. SEED-STABILITY IS A REPORTED OUTCOME, not an assumption. 778c's rem verdict flipped
   across seeds; a ceiling inside the 95% CI self-routes to the explicit
   `..._seed_unstable_underpowered` branch rather than reading the mean as a verdict.
4. SCOPE. rem phase ONLY (the sws repair is a single unambiguous build routed to
   /implement-substrate and exempt from GOV-FANOUT-1; nrem is already confirmed
   content-contingent at ratio 0.1445, 0/8 confounded).

WHY DIAGNOSTIC (not evidence)
-----------------------------
This discriminates WHY an instrument reads as it does; it tests no substrate hypothesis
and PROMOTES/DEMOTES NOTHING. `experiment_purpose="diagnostic"` excludes it from
governance confidence/conflict scoring. It tags SD-068 as the subject and
MECH-168 / INV-047 / MECH-169 as CONTEXT only. MECH-121 is deliberately NOT tagged:
MECH-121 is held (candidate/substrate_conditional, hold_pending_v3_substrate) and the
NREM leg is substrate-plumbing-fidelity only -- it must not accrue promotion evidence.
Resolution of the pre-registered hypothesis is via /failure-autopsy Step 9b Mode B
against the frozen ledger, NOT by this script's self-route.

Experiment-layer only; zero `ree_core` change.
Design + validity model: REE_assembly/docs/architecture/sd_068_consolidation_lesion_harness.md
Routing autopsy:        REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-778c_2026-07-18.json
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
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_sd068_rem_declamped_readout_diagnostic"
QUEUE_ID = "V3-EXQ-778e"
CLAIM_IDS: List[str] = ["SD-068", "MECH-168", "INV-047", "MECH-169"]
EXPERIMENT_PURPOSE = "diagnostic"
SLEEP_DRIVER_PATTERN = "manual-cycle-loop"
HYPOTHESIS_ID = "H-rem-genuinely-content-free"
HYPOTHESIS_AXIS = "representation"
HYPOTHESIS_QUESTION = "consolidation_readout_validity"

# The V3-EXQ-778a / 778c 8-seed set, reused EXACTLY so the within-run
# `calibration_error` control is a direct replication anchor against 778c.
SEEDS = [42, 7, 123, 2024, 99, 7777, 314, 1000]
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
WARM_STEPS = 40
ARMS = ["INJECTED", "NULL"]
# Held at the exact 778c condition so the READOUT is the only difference.
REM_STEP = 1.0

# The two readouts scored on the identical cells.
READOUT_DECLAMPED = "direct_precision_error"   # precision units -- the probe
READOUT_LEGACY = "calibration_error"           # variance units -- the 778c anchor

# Pre-registered thresholds.
NULL_SLOPE_RATIO_CEILING = H.NULL_SLOPE_RATIO_CEILING  # 0.25
INJECTED_SLOPE_FLOOR = 1e-6      # readiness on the DE-CLAMPED readout's own slope
NULL_SERIES_MIN_DISTINCT = 3     # non-degeneracy: the null series must not be constant
NULL_SERIES_MIN_SD = 1e-9        # ...and must carry real spread

# Both readiness anchors are `all(seeds)` gates. Expressed as the FRACTION of reference
# cells the reachability guard must see scored True, that is exactly 1.0. This is not a
# new gate: it is the existing all(...) semantics restated in the fraction units
# assert_anchor_reachable takes. A margin is unsatisfiable against a 1.0 fraction by
# construction (required = 1.0 + margin/n > 1.0), so margin_cells stays 0 -- KNOWN AND
# INTENDED, per readiness_anchor.py rule 4, rather than an unconsidered default.
ANCHOR_REACHABILITY_MIN_FRAC = 1.0


# --- THE SHIPPED PREDICATES ---------------------------------------------------------
# Both were INLINED in run_experiment. They are factored out here so the LIVE cells and
# the frozen reachability-guard references are scored through ONE callable each --
# scoring a guard with a re-implementation defeats its entire purpose, since the defect
# being guarded against IS a mis-specified predicate (V3-EXQ-778d, autopsy sec 2).
# Semantics, comparators and thresholds are UNCHANGED from the inlined originals.


def _injected_slope_supra_floor(cell: Dict[str, Any]) -> bool:
    """Did the known-damaged INJECTED arm move the de-clamped readout at all?

    THE SHIPPED PREDICATE for `declamped_injected_arm_sigma_slope_supra_floor`. This is
    the ratio's DENOMINATOR: below floor, no ratio is interpretable.
    """
    inj_slope = cell["injected_slope"]
    return bool(
        inj_slope != H.UNAVAILABLE
        and not math.isnan(inj_slope)
        and abs(inj_slope) >= INJECTED_SLOPE_FLOOR
    )


def _null_series_non_degenerate(cell: Dict[str, Any]) -> bool:
    """Does this de-clamped error series carry real spread, or is it a saturated constant?

    THE SHIPPED PREDICATE for `declamped_null_series_non_degenerate`.

    RAIL COVERAGE. V3-EXQ-778c's rem degeneracy had TWO rails -- (a) SATURATION (the
    reference clamped across the grid -> constant series -> zero slope) and (b)
    POSITIVITY-FLOOR COLLAPSE (the reference touched the 1e-3 floor and the
    VARIANCE-units fit went off-scale, |ratio| 1801-9143). V3-EXQ-778d's predicate
    tested rail (a) only and was therefore unmeetable. This predicate tests rail (a)
    directly (n_distinct / sd), and rail (b) is covered because the PRECISION-units
    readout this leg swaps in is BOUNDED: a floored reference (rv_after ~ 1/1e-3) maps
    to direct_precision_error = |1/rv_after - clean| / clean ~ 0.9995, a CONSTANT --
    i.e. under this readout rail (b) arrives AS rail (a) and is caught by the same
    test, instead of escaping as an off-scale-but-graded series. Confirmed on the
    recorded run: the 5 seeds 778c railed by saturation are exactly the 5 that come back
    constant here (n_distinct 1, sd 0), and the 3 seeds 778c railed by floor collapse
    (42/123/99) come back graded (n_distinct 5, sd 0.061-0.586). The non-finite /
    UNAVAILABLE sd case is handled explicitly, so there is no third escape.
    """
    null_sd = cell["null_series_sd"]
    return bool(
        cell["null_series_n_distinct"] >= NULL_SERIES_MIN_DISTINCT
        and null_sd != H.UNAVAILABLE
        and not math.isnan(null_sd)
        and null_sd >= NULL_SERIES_MIN_SD
    )


# --- FROZEN KNOWN-POSITIVE REFERENCES for the setup-time reachability guards ---------
# Recorded values, frozen as literals so the guards need zero compute and cannot drift
# with the substrate. Both are read off the completed run
# `v3_exq_sd068_rem_declamped_readout_diagnostic_20260718T163318Z_v3.json`
# (V3-EXQ-778e, this same script), whose INJECTED arm is the established known-positive
# control: it is the known-damaged arm, swept across the same sigma grid, and its
# de-clamped response was graded on 8/8 seeds.

# Reference A -- per-seed de-clamped injected sigma-slopes. The control this anchor's
# own `control` key names ("injected arm swept to sigma=max").
_REFERENCE_DECLAMPED_INJECTED_SLOPE: List[Dict[str, Any]] = [
    {"seed": 42, "injected_slope": 0.1953635811805725},
    {"seed": 7, "injected_slope": 0.35048487584800714},
    {"seed": 123, "injected_slope": 0.07712462544441223},
    {"seed": 2024, "injected_slope": 0.464908588053894},
    {"seed": 99, "injected_slope": 1.3731259107589722},
    {"seed": 7777, "injected_slope": 0.41504868281097407},
    {"seed": 314, "injected_slope": 0.4997496999999999},
    {"seed": 1000, "injected_slope": 0.4997496999999999},
]

# Reference B -- the spread statistics of the INJECTED arm's own de-clamped error series
# on the same 8 seeds, computed from the recorded `declamped.injected_series` by the
# identical statistic H.rem_null_slope_ratio applies to the null series (sample sd over
# the finite entries; distinct count over values rounded to 12 dp). The injected arm is
# used deliberately rather than the null arm's 3 healthy seeds: it is a control whose
# non-degeneracy is established INDEPENDENTLY of the quantity under test (it is the arm
# that by construction responds to sigma), and it exercises the gate at the full n=8 the
# live all(...) gate demands, instead of at a cherry-picked healthy subset.
_REFERENCE_DECLAMPED_GRADED_SERIES: List[Dict[str, Any]] = [
    {"seed": 42, "null_series_n_distinct": 5.0, "null_series_sd": 0.154448472094454},
    {"seed": 7, "null_series_n_distinct": 5.0, "null_series_sd": 0.27708262328051964},
    {"seed": 123, "null_series_n_distinct": 5.0, "null_series_sd": 0.06097237002292964},
    {"seed": 2024, "null_series_n_distinct": 4.0, "null_series_sd": 0.4341168261194272},
    {"seed": 99, "null_series_n_distinct": 5.0, "null_series_sd": 1.0855513480478642},
    {"seed": 7777, "null_series_n_distinct": 4.0, "null_series_sd": 0.4397063677455669},
    {"seed": 314, "null_series_n_distinct": 5.0, "null_series_sd": 0.4328804790823687},
    {"seed": 1000, "null_series_n_distinct": 5.0, "null_series_sd": 0.39656500561045827},
]

_REFERENCE_SOURCE_SLOPE = (
    "V3-EXQ-778e run_id v3_exq_sd068_rem_declamped_readout_diagnostic_"
    "20260718T163318Z_v3, arm_results[].declamped.injected_slope (de-clamped readout, "
    "INJECTED arm = the known-damaged positive control; recorded met=true on 8/8 seeds)"
)
_REFERENCE_SOURCE_SERIES = (
    "V3-EXQ-778e run_id v3_exq_sd068_rem_declamped_readout_diagnostic_"
    "20260718T163318Z_v3, spread statistics of arm_results[].declamped.injected_series "
    "(the INJECTED arm's own de-clamped error series -- a series whose gradedness is "
    "established independently of the null arm under test)"
)


def _fmt(v: float) -> str:
    """ASCII-safe float rendering that keeps UNAVAILABLE legible."""
    if v == H.UNAVAILABLE or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.4f}"


def _finite(vals: List[float]) -> List[float]:
    return [
        v for v in vals if v != H.UNAVAILABLE and not (isinstance(v, float) and math.isnan(v))
    ]


def _mean_sd_ci(vals: List[float]) -> Tuple[float, float, float, float]:
    """Mean, sample SD, and 95% CI bounds. UNAVAILABLE where undefined."""
    v = _finite(vals)
    if not v:
        return H.UNAVAILABLE, H.UNAVAILABLE, H.UNAVAILABLE, H.UNAVAILABLE
    mean = sum(v) / len(v)
    if len(v) < 2:
        return mean, H.UNAVAILABLE, H.UNAVAILABLE, H.UNAVAILABLE
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
    sem = sd / math.sqrt(len(v))
    return mean, sd, mean - 1.96 * sem, mean + 1.96 * sem


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    warm = 8 if dry_run else WARM_STEPS
    sigmas = [0.0, 0.5, 2.0] if dry_run else SIGMAS

    print(
        "V3-EXQ-778e: SD-068 REM de-clamped readout (H-rem-genuinely-content-free)",
        flush=True,
    )
    print(
        f"  seeds={seeds} sigmas={sigmas} warm_steps={warm} arms={ARMS} "
        f"step={REM_STEP} dry_run={dry_run}",
        flush=True,
    )

    # SETUP-TIME REACHABILITY GUARDS -- before any compute. Each replays a frozen,
    # known-positive reference through THE SHIPPED PREDICATE and refuses to run if the
    # gate exceeds what that reference can itself score. A precondition that a faithful
    # positive control CANNOT pass is a guaranteed false negative: it reports met=false
    # on every run and mislabels an instrument-specification gap as
    # `substrate_not_ready_requeue` / `measurement_still_degenerate_requeue`. This is the
    # check that would have caught the V3-EXQ-778d defect at design-audit time. Raises
    # AnchorUnreachable (an AssertionError) -> non-zero exit -> runner classifies ERROR,
    # which is the correct loud failure. The references are frozen, so the guards are
    # dry-run-invariant and the smoke test exercises them.
    slope_guard = assert_anchor_reachable(
        anchor_name="declamped_injected_arm_sigma_slope_supra_floor",
        reference_cells=_REFERENCE_DECLAMPED_INJECTED_SLOPE,
        score_fn=_injected_slope_supra_floor,
        threshold=ANCHOR_REACHABILITY_MIN_FRAC,
        reference_source=_REFERENCE_SOURCE_SLOPE,
    )
    print(
        f"  [guard] anchor reachability OK: "
        f"declamped_injected_arm_sigma_slope_supra_floor -- the known-damaged reference "
        f"scores {slope_guard['n_reference_scored_true']}/"
        f"{slope_guard['n_reference_cells']} = "
        f"{slope_guard['reference_score']:.3f} under the shipped predicate "
        f"(gate {ANCHOR_REACHABILITY_MIN_FRAC:.2f} = the live all-seeds gate)",
        flush=True,
    )
    series_guard = assert_anchor_reachable(
        anchor_name="declamped_null_series_non_degenerate",
        reference_cells=_REFERENCE_DECLAMPED_GRADED_SERIES,
        score_fn=_null_series_non_degenerate,
        threshold=ANCHOR_REACHABILITY_MIN_FRAC,
        reference_source=_REFERENCE_SOURCE_SERIES,
    )
    print(
        f"  [guard] anchor reachability OK: declamped_null_series_non_degenerate -- the "
        f"known-graded reference scores {series_guard['n_reference_scored_true']}/"
        f"{series_guard['n_reference_cells']} = "
        f"{series_guard['reference_score']:.3f} under the shipped predicate "
        f"(gate {ANCHOR_REACHABILITY_MIN_FRAC:.2f} = the live all-seeds gate); the "
        "predicate covers BOTH 778c rails, floor-collapse arriving as a bounded "
        "constant under the precision-units readout",
        flush=True,
    )

    config_slice = {
        "sigmas": sigmas,
        "warm_steps": warm,
        "arms": list(ARMS),
        "rem_step": REM_STEP,
        "readout_declamped": READOUT_DECLAMPED,
        "readout_legacy": READOUT_LEGACY,
        "null_slope_ratio_ceiling": NULL_SLOPE_RATIO_CEILING,
        "injected_slope_floor": INJECTED_SLOPE_FLOOR,
        "null_series_min_distinct": NULL_SERIES_MIN_DISTINCT,
        "null_series_min_sd": NULL_SERIES_MIN_SD,
        "shy_decay_rate": 0.85,
        "body_obs_dim": H.BODY_OBS_DIM,
        "world_obs_dim": H.WORLD_OBS_DIM,
        "action_dim": H.ACTION_DIM,
        "harm_obs_dim": H.HARM_OBS_DIM,
    }

    arm_results: List[Dict[str, Any]] = []
    total_eps = len(sigmas)

    for seed in seeds:
        print(f"Seed {seed} Condition REM_DECLAMPED_READOUT", flush=True)
        with arm_cell(
            seed,
            config_slice=config_slice,
            script_path=Path(__file__),
            config_slice_declared=True,
        ) as cell_ctx:
            inj_pr: Dict[float, Dict[str, Dict[str, float]]] = {}
            null_pr: Dict[float, Dict[str, Dict[str, float]]] = {}
            for i, s in enumerate(sigmas):
                inj_pr[s] = H.rem_only_integrity_at_sigma(
                    seed=seed,
                    sigma=s,
                    warm_steps=warm,
                    content_scale=1.0,
                    rem_step=REM_STEP,
                )
                null_pr[s] = H.rem_only_integrity_at_sigma(
                    seed=seed,
                    sigma=s,
                    warm_steps=warm,
                    content_scale=0.0,
                    rem_step=REM_STEP,
                )
                print(
                    f"  [train] declamped seed={seed} ep {i + 1}/{total_eps} sigma={s}",
                    flush=True,
                )

            # BOTH readouts on the IDENTICAL cells -- the legacy one is the within-run
            # replication anchor against 778c.
            declamped = H.rem_null_slope_ratio(
                sigmas=list(sigmas),
                injected_pr_by_sigma=inj_pr,
                null_pr_by_sigma=null_pr,
                rem_error_key=READOUT_DECLAMPED,
            )
            legacy = H.rem_null_slope_ratio(
                sigmas=list(sigmas),
                injected_pr_by_sigma=inj_pr,
                null_pr_by_sigma=null_pr,
                rem_error_key=READOUT_LEGACY,
            )

            # Scored through the SHIPPED predicates -- the same two callables the
            # setup-time reachability guards replay their frozen references through.
            readiness = _injected_slope_supra_floor(declamped)
            non_degenerate = _null_series_non_degenerate(declamped)
            ratio = declamped["null_slope_ratio"]
            content_free = bool(
                ratio != H.UNAVAILABLE
                and not math.isnan(ratio)
                and ratio > NULL_SLOPE_RATIO_CEILING
            )

            row: Dict[str, Any] = {
                "seed": seed,
                "arm": "REM_DECLAMPED_READOUT",
                "arms_compared": list(ARMS),
                "hypothesis_id": HYPOTHESIS_ID,
                "sigmas": list(sigmas),
                "declamped": declamped,
                "legacy_anchor": legacy,
                "null_slope_ratio_declamped": ratio,
                "null_slope_ratio_legacy": legacy["null_slope_ratio"],
                "readiness_met": readiness,
                "null_series_non_degenerate": non_degenerate,
                "content_free": content_free,
                "seed_pass": bool(readiness and non_degenerate and content_free),
                # BOTH arms' internals recorded as richly as each other.
                "integrity_injected": {str(s): inj_pr[s]["rem"] for s in sigmas},
                "integrity_null": {str(s): null_pr[s]["rem"] for s in sigmas},
            }
            cell_ctx.stamp(row)

        arm_results.append(row)
        print(
            f"  declamped ratio={_fmt(ratio)} (legacy anchor "
            f"{_fmt(legacy['null_slope_ratio'])}) "
            f"null_n_distinct={int(declamped['null_series_n_distinct'])} "
            f"null_sd={_fmt(declamped['null_series_sd'])} "
            f"non_degenerate={non_degenerate}",
            flush=True,
        )
        print(f"verdict: {'PASS' if row['seed_pass'] else 'FAIL'}", flush=True)

    n = len(arm_results)
    readiness_ok = all(r["readiness_met"] for r in arm_results)
    non_degenerate_ok = all(r["null_series_non_degenerate"] for r in arm_results)
    n_content_free = sum(1 for r in arm_results if r["content_free"])

    declamped_ratios = [r["null_slope_ratio_declamped"] for r in arm_results]
    legacy_ratios = [r["null_slope_ratio_legacy"] for r in arm_results]
    d_mean, d_sd, d_lo, d_hi = _mean_sd_ci(declamped_ratios)
    l_mean, l_sd, l_lo, l_hi = _mean_sd_ci(legacy_ratios)

    ceiling_inside_ci = bool(
        d_lo != H.UNAVAILABLE and d_lo <= NULL_SLOPE_RATIO_CEILING <= d_hi
    )
    verdict_stable = bool(n_content_free == 0 or n_content_free == n)

    readout_summary = {
        READOUT_DECLAMPED: {
            "mean_null_slope_ratio": d_mean,
            "sd_null_slope_ratio": d_sd,
            "ci95_low": d_lo,
            "ci95_high": d_hi,
            "per_seed_null_slope_ratio": declamped_ratios,
            "per_seed_null_series_sd": [
                r["declamped"]["null_series_sd"] for r in arm_results
            ],
            "per_seed_null_series_n_distinct": [
                r["declamped"]["null_series_n_distinct"] for r in arm_results
            ],
            "n_seeds_content_free": n_content_free,
            "ceiling_inside_ci95": ceiling_inside_ci,
            "verdict_stable": verdict_stable,
        },
        READOUT_LEGACY: {
            "mean_null_slope_ratio": l_mean,
            "sd_null_slope_ratio": l_sd,
            "ci95_low": l_lo,
            "ci95_high": l_hi,
            "per_seed_null_slope_ratio": legacy_ratios,
            "per_seed_null_series_n_distinct": [
                r["legacy_anchor"]["null_series_n_distinct"] for r in arm_results
            ],
            "note": (
                "WITHIN-RUN REPLICATION ANCHOR: the V3-EXQ-778c readout scored on the "
                "IDENTICAL cells. Any difference from the de-clamped readout is "
                "attributable to the readout DEFINITION alone, not to a run-to-run "
                "difference. 778c recorded exactly 0.0 on 5/8 seeds and 1801-9143 on "
                "3/8."
            ),
        },
    }

    # Self-routed label. Readiness / residual degeneracy route ONLY to requeue.
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
    elif not non_degenerate_ok:
        label = "measurement_still_degenerate_requeue"
    elif ceiling_inside_ci or not verdict_stable:
        label = "rem_declamped_verdict_seed_unstable_underpowered"
    elif n_content_free == n:
        label = "rem_readout_genuinely_content_free"
    else:
        label = "rem_readout_content_contingent_when_declamped"

    overall_pass = bool(
        readiness_ok
        and non_degenerate_ok
        and verdict_stable
        and not ceiling_inside_ci
        and n_content_free == n
    )

    min_inj_slope = min(
        abs(r["declamped"]["injected_slope"])
        for r in arm_results
        if r["declamped"]["injected_slope"] != H.UNAVAILABLE
        and not math.isnan(r["declamped"]["injected_slope"])
    )
    min_null_sd = min(
        _finite([r["declamped"]["null_series_sd"] for r in arm_results]) or [0.0]
    )

    interpretation = {
        "label": label,
        "hypothesis_id": HYPOTHESIS_ID,
        "hypothesis_axis": HYPOTHESIS_AXIS,
        "hypothesis_question": HYPOTHESIS_QUESTION,
        "declared_null": (
            f"if the de-clamped readout yields null_slope_ratio <= "
            f"{NULL_SLOPE_RATIO_CEILING}, H-rem-genuinely-content-free is REFUTED -- the "
            "rem readout DOES track content once scored in a non-degenerate "
            "representation, and the V3-EXQ-778c both-rails degeneracy was an artifact "
            "rather than content-freeness."
        ),
        "preconditions": [
            {
                "name": "declamped_injected_arm_sigma_slope_supra_floor",
                "description": (
                    "the ratio's DENOMINATOR under the DE-CLAMPED readout -- the same "
                    "statistic the content-freeness criterion routes on -- clears the "
                    "floor on the known-damaged positive control. Below floor means the "
                    "sweep never damaged the de-clamped readout, so its ratio cannot "
                    "discriminate content-freeness from inertness."
                ),
                "measured": float(min_inj_slope),
                "threshold": INJECTED_SLOPE_FLOOR,
                "direction": "lower",
                "control": "injected arm swept to sigma=max (known-damaged positive control)",
                "met": bool(readiness_ok),
            },
            {
                "name": "declamped_null_series_non_degenerate",
                "description": (
                    "the NULL arm's OWN error series under the de-clamped readout "
                    "carries real spread. A ratio near 0.0 is ambiguous between 'inert "
                    "on noise' (content-contingent -- the finding) and 'saturated "
                    "constant' (degenerate -- no finding); that ambiguity is exactly "
                    "what left V3-EXQ-778c's rem leg unresolved. Asserting the null "
                    "series' spread is what makes the declared null falsifiable rather "
                    "than satisfiable by a fresh degeneracy."
                ),
                "measured": float(min_null_sd),
                "threshold": NULL_SERIES_MIN_SD,
                "direction": "lower",
                "control": "null arm swept across the full sigma grid",
                "met": bool(non_degenerate_ok),
            },
        ],
        "criteria_non_degenerate": {
            "C1_declamped_ratio_above_ceiling": bool(readiness_ok and non_degenerate_ok),
            "C2_verdict_seed_stable": bool(n >= 2),
            "C3_legacy_anchor_scored": bool(
                len(_finite(legacy_ratios)) >= 1
            ),
        },
        "criteria": [
            {
                "name": "C1_declamped_ratio_above_ceiling",
                "load_bearing": True,
                "passed": bool(n_content_free == n),
            },
            {
                "name": "C2_verdict_seed_stable",
                "load_bearing": False,
                "passed": bool(verdict_stable and not ceiling_inside_ci),
            },
            {
                "name": "C3_legacy_anchor_scored",
                "load_bearing": False,
                "passed": bool(len(_finite(legacy_ratios)) >= 1),
            },
        ],
        "readout_summary": readout_summary,
        # Proof, recorded in the shipped artifact, that BOTH readiness gates are
        # reachable by their own known-positive references under the shipped predicates.
        "anchor_reachability_guards": {
            "declamped_injected_arm_sigma_slope_supra_floor": slope_guard,
            "declamped_null_series_non_degenerate": series_guard,
        },
        "portfolio": {
            "gov_rule": "GOV-FANOUT-1",
            "question": HYPOTHESIS_QUESTION,
            "this_leg": f"{HYPOTHESIS_ID} (axis={HYPOTHESIS_AXIS})",
            "sibling_legs": [
                "V3-EXQ-778d H-rem-clamp-artifact (axis=measurement)",
                "V3-EXQ-778f H-gen-gain-content-free (axis=observation)",
            ],
            "note": (
                "Read the three legs JOINTLY. Resolution via /failure-autopsy Step 9b "
                "Mode B against the frozen ledger, not by this self-route."
            ),
        },
    }

    per_claim = {
        "SD-068": "unknown",
        "MECH-168": "unknown",
        "INV-047": "unknown",
        "MECH-169": "unknown",
    }

    print("", flush=True)
    print(
        f"content-free seeds: {n_content_free}/{n} | readiness_ok={readiness_ok} "
        f"non_degenerate_ok={non_degenerate_ok} verdict_stable={verdict_stable}",
        flush=True,
    )
    print(
        f"  declamped ({READOUT_DECLAMPED}): mean={_fmt(d_mean)} sd={_fmt(d_sd)} "
        f"ci95=[{_fmt(d_lo)}, {_fmt(d_hi)}] (ceiling {NULL_SLOPE_RATIO_CEILING})"
        + ("  [CEILING INSIDE CI -- verdict unresolved at this n]" if ceiling_inside_ci else ""),
        flush=True,
    )
    print(
        f"  legacy anchor ({READOUT_LEGACY}): mean={_fmt(l_mean)} sd={_fmt(l_sd)}",
        flush=True,
    )
    print(f"self-route label: {label}", flush=True)

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": per_claim,
        "interpretation": interpretation,
        "arm_results": arm_results,
        "readout_summary": readout_summary,
        "config": config_slice,
        "seeds": seeds,
        "non_degenerate": bool(readiness_ok and non_degenerate_ok),
        "degeneracy_reason": (
            ""
            if (readiness_ok and non_degenerate_ok)
            else (
                "the de-clamped readout's injected slope fell below floor and/or its "
                "null series remained a saturated constant; the ratio does not "
                "discriminate content-freeness from inertness"
            )
        ),
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
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "readout_summary": result["readout_summary"],
        "hypothesis_id": HYPOTHESIS_ID,
        "hypothesis_axis": HYPOTHESIS_AXIS,
        "hypothesis_question": HYPOTHESIS_QUESTION,
        "acceptance_criteria": {
            "C1_declamped_ratio_above_ceiling": (
                f"null_slope_ratio under {READOUT_DECLAMPED} > "
                f"{NULL_SLOPE_RATIO_CEILING} on ALL seeds (LOAD-BEARING). At or below "
                "the ceiling REFUTES this leg -- its declared null."
            ),
            "C2_verdict_seed_stable": (
                "the content-free verdict is identical across seeds AND the ceiling "
                "does not fall inside the 95% CI (else -> "
                "rem_declamped_verdict_seed_unstable_underpowered, explicitly not a "
                "verdict)"
            ),
            "C3_legacy_anchor_scored": (
                f"{READOUT_LEGACY} scored on the identical cells as a within-run "
                "replication anchor against V3-EXQ-778c"
            ),
        },
        "arm_results": result["arm_results"],
        "notes": (
            "SD-068 GOV-FANOUT-1 discrimination portfolio, leg 2 of 3 "
            "(axis=REPRESENTATION, hypothesis H-rem-genuinely-content-free, "
            "pre-registered 2026-07-18T08:41:15Z in hypothesis_space_registry.v1.json "
            "question consolidation_readout_validity). Routed by "
            "failure_autopsy_V3-EXQ-778c_2026-07-18.json "
            "targets[0].fanout_recommendation. Scores the rem leg against the injected "
            "precision target DIRECTLY in PRECISION units "
            "(direct_precision_error = |1/rv_after - clean_target| / clean_target) "
            "instead of via running_variance_after in VARIANCE units "
            "(calibration_error = |rv_after - 1/clean_target|), which undoes the 1/x "
            "blow-up that pinned V3-EXQ-778c's null arm at the constant "
            "998.5009992509989 on 5/8 seeds and sent it off-scale 1801-9143 on 3/8. "
            "Everything else is held at the exact 778c condition (step=1.0, "
            "zero-content null, same 8 seeds, same sigma grid, same RNG streams), so "
            "the READOUT DEFINITION is the only difference; the 778c readout is ALSO "
            "scored on the identical cells as a within-run replication anchor. A "
            "NON-DEGENERACY GATE on the null arm's own spread routes a still-degenerate "
            "null to measurement_still_degenerate_requeue rather than letting a fresh "
            "degeneracy satisfy the declared null. rem phase ONLY (the sws repair is a "
            "single unambiguous build, exempt from GOV-FANOUT-1; nrem is already "
            "confirmed content-contingent at ratio 0.1445). DIAGNOSTIC: excluded from "
            "governance confidence/conflict scoring; PROMOTES/DEMOTES NOTHING. MECH-121 "
            "deliberately NOT tagged (held; NREM leg is substrate-plumbing-fidelity "
            "only). Siblings V3-EXQ-778d (measurement) and V3-EXQ-778f (observation) "
            "run in PARALLEL and are read jointly. Experiment-layer only; zero ree_core "
            "change. GOV-REUSE-1: the decisive readout is the null/injected sigma-slope "
            "ratio under a PRECISION-units error, which no recorded manifest carries -- "
            "778/778a/778b/778c all scored calibration_error in variance units only, "
            "and direct_precision_error did not exist before this leg -- so it is not "
            "recoverable by reanalysis -> run."
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
