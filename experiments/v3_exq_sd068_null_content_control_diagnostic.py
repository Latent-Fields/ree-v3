"""
V3-EXQ-778c: SD-068 zero-injected-content NULL CONTROL for the consolidation-pipeline
lesion harness -- 8-seed powered successor to V3-EXQ-778b.
SLEEP DRIVER: manual-cycle-loop (the SD-068 harness drives enter_sws_mode /
run_sws_schema_pass + enter_rem_mode / run_rem_attribution_pass +
recalibrate_precision_to directly per phase readout; no SleepLoopManager scheduling).

WHAT THIS MEASURES
------------------
V3-EXQ-778 validated the SD-068 harness (PASS) and V3-EXQ-778a powered its
damage-tolerance staging order up to 8 seeds. That order -- WHATEVER it turns out to
be -- is only meaningful if each per-phase readout is tracking CONTENT FIDELITY.

STATE OF THE ORDER (read from the manifests, not assumed). The order is NOT stable:
778 produced three DIFFERENT orders across its three seeds, and 778a self-routed
`staging_seed_variable_underpowered` -- modal order (rem, nrem, sws) at only 4/8
seeds, Spearman mean rho 0.375 with CI [-0.247, 0.997] spanning zero, and the
contested rem-fails-first axis at 4-pos/4-neg (sign test p = 1.0). So this control is
NOT auditing a settled ordering; it is asking the prior question of whether the
per-phase readouts the ordering is built from are measuring content at all. If a
readout is confounded, its seed-variability is not merely noise to be powered
through -- it is variability in a noise-sensitivity measurement.

This run tests the alternative:
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
  C3: the confound verdict is stable across all 8 seeds.

Per-phase ratios are reported as a DISTRIBUTION (mean / SD / 95% CI / per-seed values),
matching the V3-EXQ-778a treatment, because the underlying damage tolerance is strongly
heteroscedastic (778a: sws SD ~8.6e-9, nrem ~0.0014, rem ~0.396). A phase whose CI
STRADDLES the ceiling is flagged `ceiling_inside_ci95` -- its confound verdict is
UNRESOLVED at this n and must not be read as a clean pass or a clean confound.

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
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_sd068_null_content_control_diagnostic"
QUEUE_ID = "V3-EXQ-778c"
# 778c re-runs the identical control at the full V3-EXQ-778a 8-seed set. 778b ran at
# n=2 and left the rem leg UNRESOLVED (per-seed ratios [4348.47, 0.0], verdict flipped);
# its sws (1.0000) and nrem (0.144) legs were stable and are reproduced here.
SUPERSEDES = "V3-EXQ-778b"
# SD-068 = the harness whose non-vacuity this audits. The three staging claims are
# tagged as context only. MECH-121 is deliberately ABSENT (held; the NREM leg is
# substrate-plumbing-fidelity only and must not accrue promotion evidence).
CLAIM_IDS: List[str] = ["SD-068", "MECH-168", "INV-047", "MECH-169"]
EXPERIMENT_PURPOSE = "diagnostic"
SLEEP_DRIVER_PATTERN = "manual-cycle-loop"

# The V3-EXQ-778a 8-seed set, reused EXACTLY so the null ratios pool directly onto
# that run's damage-tolerance distribution.
#
# WHY 8 AND NOT 2 (corrected 2026-07-18, after V3-EXQ-778b ran and confirmed the
# problem empirically): this control was first authored at
# seeds [42, 7] on the belief that V3-EXQ-778 had found a seed-STABLE order of
# (nrem, rem, sws). The manifests refute that on every count -- 778 produced THREE
# DIFFERENT orders in its three seeds, (nrem, rem, sws) occurs in only 1 of 778a's 8
# seeds, the modal order is the PREDICTED (rem, nrem, sws) at 4/8, and 778a's own
# self-route label is `staging_seed_variable_underpowered` (Spearman mean rho 0.375,
# CI [-0.247, 0.997] spanning zero; the contested rem-fails-first axis 4-pos/4-neg,
# sign test p = 1.0). Seeds 42 and 7 specifically give DIFFERENT orders.
#
# That matters here because the per-phase damage tolerance is wildly heteroscedastic:
# sws SD ~ 8.6e-9 (deterministic -- its ratio is analytic and seed-invariant), nrem
# SD ~ 0.0014, but REM SD ~ 0.396. The rem leg's null ratio is genuinely seed-variable
# (its clamp saturation is itself a random draw), so n=2 would under-power exactly the
# phase most in need of characterisation -- repeating at the control layer the error
# 778a was queued to fix at the staging layer.
#
# V3-EXQ-778b then CONFIRMED this empirically at full fidelity: sws [1.0000, 1.0000]
# and nrem [0.1449, 0.1429] came back stable, but rem came back [4348.47, 0.0] -- the
# confound verdict FLIPPED between the two seeds (confound_verdict_stable=False). The
# one leg whose confound status is genuinely in question was left UNRESOLVED. Hence
# the full 8 here.
SEEDS = [42, 7, 123, 2024, 99, 7777, 314, 1000]
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
WARM_STEPS = 40
ARMS = ["INJECTED", "NULL"]

# Pre-registered thresholds.
NULL_SLOPE_RATIO_CEILING = H.NULL_SLOPE_RATIO_CEILING  # 0.25
# WIDE MARGIN, KNOWN AND INTENDED (audited 2026-07-19). Tightest recorded phase slope is
# ~0.0967 against this 1e-6 floor -- ~5 orders of headroom. NOT the rule-3 defect: this
# gates the literal DENOMINATOR of null_slope_ratio_<phase>, the statistic C1 routes on,
# so a tripwire orders below the working range is the correct shape. But `met: true` says
# only "the ratio was computable", not "the substrate was ready"; and it is largely
# REDUNDANT with the harness's own NULL_MIN_INJECTED_SLOPE = 1e-9 0/0 guard
# (consolidation_lesion_harness.py:1409), which already reports the phase UNAVAILABLE.
# Deliberately NOT retuned -- the run has already executed.
INJECTED_SLOPE_FLOOR = 1e-6   # C2 readiness: min |injected slope| for an interpretable ratio
PASS_FRACTION = 1.0           # ALL seeds must be clean for C1 (a control, not a vote)


def _fmt(v: float) -> str:
    """ASCII-safe float rendering that keeps UNAVAILABLE legible."""
    if v == H.UNAVAILABLE or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.4f}"


# --- THE SHIPPED READINESS PREDICATE ------------------------------------------------
# The single declared precondition, `injected_arm_sigma_slope_supra_floor`, is a PER-SEED
# boolean (`C2_ratio_interpretable_all_phases`) aggregated with `all(...)` over seeds
# (`readiness_ok` below), so the faithful re-expression for `assert_anchor_reachable` is:
# score_fn = the per-seed predicate, threshold = the FRACTION 1.0.
#
# These are factored to MODULE LEVEL precisely so the live scoring path in `_score_seed`
# and the setup-time reachability guard run THE SAME CALLABLE. Scoring the guard with a
# re-implementation would defeat its purpose -- the defect class being guarded against IS
# a mis-specified predicate (SD-068 REM fanout autopsy, Learning 1; see
# experiments/_lib/readiness_anchor.py).
#
# A `cell` is one seed's recorded values: {"seed": int, "injected_slope": {phase: float}}
# -- exactly the shape `arm_results[i].injected_slope` is written out with.


def _phase_slope_supra_floor(inj: Any) -> bool:
    """One phase's injected-arm sigma-slope clears the interpretability floor.

    FLOOR-shaped, INCLUSIVE (`>=`) -- comparator preserved exactly as shipped.
    """
    return bool(
        inj != H.UNAVAILABLE
        and not math.isnan(inj)
        and abs(inj) >= INJECTED_SLOPE_FLOOR
    )


def _injected_slopes_all_supra_floor(cell: Dict[str, Any]) -> bool:
    """C2 readiness, per seed: the ratio's DENOMINATOR is interpretable on ALL 3 phases.

    Precondition `injected_arm_sigma_slope_supra_floor`. A near-zero injected slope means
    the sweep never damaged that readout, so null/injected has no referent.
    """
    slopes = cell.get("injected_slope") or {}
    return all(
        _phase_slope_supra_floor(slopes.get(p, H.UNAVAILABLE))
        for p in ("sws", "nrem", "rem")
    )


# The KNOWN-DAMAGED POSITIVE CONTROL the precondition's own `control` key names ("injected
# arm swept to sigma=max"), frozen as a literal. These are the per-seed
# `arm_results[i].injected_slope` values recorded by the completed V3-EXQ-778c run
# `v3_exq_sd068_null_content_control_diagnostic_20260718T072318Z_v3` -- the same script,
# same 8 seeds, same sigma grid -- which reported `ratio_interpretable` true on 3/3 phases
# for 8/8 seeds. Frozen so the guard needs zero compute and cannot drift with the
# substrate. (Note the run's overall outcome was FAIL on C1 content-contingency; that is
# irrelevant here -- what makes it the right reference for THIS anchor is that its
# injected arm was demonstrably damaged, which is the only thing the precondition asserts.)
_REFERENCE_778C_INJECTED_SLOPES: List[Dict[str, Any]] = [
    {"seed": 42, "injected_slope": {
        "sws": 2.269717072399339, "nrem": 0.09671970039873905,
        "rem": 0.1386070204874681}},
    {"seed": 7, "injected_slope": {
        "sws": 2.151620459920367, "nrem": 0.09867377313414635,
        "rem": 1.1855791050165805}},
    {"seed": 123, "injected_slope": {
        "sws": 2.384814913120032, "nrem": 0.09667174055435042,
        "rem": 0.06648187008701933}},
    {"seed": 2024, "injected_slope": {
        "sws": 2.0791972446568705, "nrem": 0.0967143029598984,
        "rem": 1197.8934429238284}},
    {"seed": 99, "injected_slope": {
        "sws": 2.2589915022805984, "nrem": 0.09865533298549493,
        "rem": 0.3325760259971743}},
    {"seed": 7777, "injected_slope": {
        "sws": 2.2808525420241947, "nrem": 0.09687813173177665,
        "rem": 1196.8272982267972}},
    {"seed": 314, "injected_slope": {
        "sws": 2.2748340739382793, "nrem": 0.0974091445154601,
        "rem": 999.5480676256806}},
    {"seed": 1000, "injected_slope": {
        "sws": 2.343211152122328, "nrem": 0.09708835577665026,
        "rem": 998.5654418183228}},
]
_REFERENCE_SOURCE = (
    "V3-EXQ-778c completed run "
    "v3_exq_sd068_null_content_control_diagnostic_20260718T072318Z_v3 "
    "(same script, same 8 seeds, same sigma grid; ratio_interpretable true on 3/3 "
    "phases for 8/8 seeds; per-seed injected_slope recorded in arm_results)"
)
# The precondition is a per-seed boolean aggregated with `all(...)` over seeds, so the
# reachability threshold is the FRACTION 1.0 -- every reference cell must score.
ANCHOR_ALL_SEEDS_FRAC = 1.0


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
        # Scored through the MODULE-LEVEL shipped predicate -- the same callable the
        # setup-time reachability guard replays the frozen 778c reference through.
        interpretable[p] = _phase_slope_supra_floor(inj_slopes[p])

    confounded = H.confounded_phase_names(control)
    c1 = all(contingent.values())
    # The per-seed precondition, evaluated on a cell of exactly the shape the guard's
    # frozen reference cells carry.
    c2 = _injected_slopes_all_supra_floor({"injected_slope": inj_slopes})
    return {
        # Carried so H.subgroup_ratio_stats can NAME the seeds it excludes -- its
        # default seed_of reads this key, and a scoped statistic whose excluded_seeds
        # list is [null, null, ...] is not auditable.
        "seed": int(control.get("null_control_seed", -1)),
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

    print("V3-EXQ-778c: SD-068 zero-injected-content null control (8-seed)", flush=True)
    print(
        f"  seeds={seeds} sigmas={sigmas} warm_steps={warm} "
        f"arms={ARMS} dry_run={dry_run}",
        flush=True,
    )

    # ---- READINESS-ANCHOR REACHABILITY GUARD (setup-time, BEFORE any compute) --------
    # This script declares an anchor-kind readiness precondition
    # (`injected_arm_sigma_slope_supra_floor`) and self-routes on it to
    # `substrate_not_ready_requeue`. A precondition its own known-damaged positive control
    # CANNOT pass is a guaranteed false negative: it would report met=false on every run
    # and mislabel an instrument-specification gap as a substrate verdict (the confirmed
    # V3-EXQ-778d defect; see experiments/_lib/readiness_anchor.py).
    #
    # The guard replays the FROZEN V3-EXQ-778c reference through THE SHIPPED PREDICATE --
    # the same module-level callable `_score_seed` scores the live cells with. Raises
    # AnchorUnreachable (an AssertionError) -> non-zero exit -> runner classifies ERROR,
    # which is the correct loud failure. Runs on dry-run too: the reference is frozen, so
    # the guard is dry-run-invariant and the smoke test exercises it.
    anchor_guard = assert_anchor_reachable(
        anchor_name="injected_arm_sigma_slope_supra_floor",
        reference_cells=_REFERENCE_778C_INJECTED_SLOPES,
        score_fn=_injected_slopes_all_supra_floor,
        threshold=ANCHOR_ALL_SEEDS_FRAC,
        reference_source=_REFERENCE_SOURCE,
        # margin_cells=0 is KNOWN AND INTENDED, not an oversight (readiness_anchor rule
        # 4). The shipped aggregation is `all(...)` over seeds, so the gate is already the
        # maximum expressible fraction (1.0) and ANY margin would make required > 1.0 --
        # unsatisfiable by construction. The headroom that matters is instead at the
        # per-cell level, and it is enormous: the reference's tightest phase slope is
        # nrem ~0.0967 against a floor of 1e-6, five orders of magnitude clear.
        margin_cells=0,
    )
    print(
        f"  [guard] anchor reachable: injected_arm_sigma_slope_supra_floor -- the "
        f"known-damaged V3-EXQ-778c reference scores "
        f"{anchor_guard['n_reference_scored_true']}/"
        f"{anchor_guard['n_reference_cells']} = "
        f"{anchor_guard['reference_score']:.3f} under the shipped predicate "
        f"(gate {ANCHOR_ALL_SEEDS_FRAC:.2f}, floor {INJECTED_SLOPE_FLOOR})",
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

    # ---- Per-phase aggregation across seeds (mean ratio + confound stability). ----
    #
    # SCOPING DECISION (family audit follow-on to the V3-EXQ-778h C2 fix, ree-v3
    # b42f69ffa3). The rem leg's MAGNITUDE statistics -- mean/sd/ci95 and the derived
    # ceiling_inside_ci95 -- are scoped to the ON-SCALE subgroup, i.e. seeds whose
    # `null_slope_ratio_rem_off_scale` flag is clear.
    #
    # WHY. That flag is the harness's own statement (see the block setting
    # `null_slope_ratio_rem_off_scale` in consolidation_lesion_harness.py) that the
    # ratio is NOT on a common scale with the other seeds': when the null arm's rem
    # precision reference collapses onto the 1e-3 positivity floor, the 1/1e-3 = 1000
    # term dominates the calibration error, so the number reads "this leg is
    # structurally content-free", never a calibrated N-fold noise sensitivity. Pooling
    # a quantity with no common scale into a mean produces a number with no referent,
    # and -- because the off-scale values are 3-4 orders of magnitude out -- an SEM
    # large enough to swallow the ceiling, so `ceiling_inside_ci95` goes True and reads
    # as "underpowered, cannot conclude" regardless of what the on-scale seeds show.
    #
    # THE CONCRETE INSTANCE, run ..._20260718T072318Z_v3 (V3-EXQ-778c): all 8 seeds are
    # off-scale (5 at clamp_frac 1.0 reporting ratio 0.0, 3 at 0.2 reporting
    # 1801.6 / 4348.5 / 9142.8). Pooled that is mean 1911.6, sd 3306.1,
    # CI95 [-379.4, 4202.6], ceiling_inside_ci95 true -- a published point estimate and
    # interval for a quantity that was never measured on scale, plus a NEGATIVE lower
    # bound on a ratio of magnitudes. Scoped, subgroup_n is 0 and the mean/sd/CI are
    # UNAVAILABLE: "there is no on-scale rem ratio at this n" is the honest reading, and
    # it is the one the off-scale flag was written to convey.
    #
    # THE POOLING WAS NOT A DELIBERATE REGISTER CHOICE. The follow-on leg's own
    # docstring (v3_exq_sd068_rem_declamped_readout_diagnostic.py, V3-EXQ-778e) already
    # reads this run the scoped way in prose -- "DEGENERATE AT BOTH RAILS ... the 5
    # apparently-clean seeds are clean only BY DEGENERACY -- a zero slope from a
    # saturated constant is the absence of a measurement, not evidence of
    # content-contingency". The family had therefore already discounted the pooled
    # number; this change makes the register emit what its consumers were reading
    # around it, so the two cannot drift apart.
    #
    # WHAT IS *NOT* SCOPED. The per-seed confound VERDICT (`confounded_phases`,
    # n_seeds_confounded, confound_verdict_stable) stays over ALL seeds, per the
    # register's standing rule that confounded phases are reported and never dropped --
    # off-scale bears on the magnitude, not on whether the phase is confounded. The
    # per_seed_null_slope_ratio audit trail likewise stays complete, so a reader can
    # always see the values the narrowing excluded.
    #
    # sws/nrem have no off-scale concept (the flag is rem-specific in the harness), so
    # their predicate is trivially true and their statistics are unchanged. They are
    # routed through the same helper deliberately: a uniform call site is what stops a
    # future edit reintroducing an unscoped comprehension for one phase only.
    def _ratio_on_scale(phase: str):
        if phase != "rem":
            return lambda s: True
        return lambda s: not s["rem_ratio_off_scale"]

    phase_summary: Dict[str, Any] = {}
    for p in ("sws", "nrem", "rem"):
        # Distribution, not just a mean -- the per-phase damage tolerance is strongly
        # heteroscedastic across seeds (778a: sws SD ~8.6e-9 vs rem SD ~0.396), so a
        # bare mean would hide that the rem leg's ratio is the seed-variable one.
        stats = H.subgroup_ratio_stats(
            seed_scores,
            eligible=_ratio_on_scale(p),
            value=lambda s, _p=p: s["null_slope_ratio"][_p],
            ceiling=NULL_SLOPE_RATIO_CEILING,
        )
        n_conf = sum(1 for s in seed_scores if p in s["confounded_phases"])
        phase_summary[p] = {
            "mean_null_slope_ratio": stats["mean"],
            "sd_null_slope_ratio": stats["sd"],
            "ci95_low": stats["ci95_low"],
            "ci95_high": stats["ci95_high"],
            # The n BEHIND the mean/sd/CI -- i.e. the on-scale, finite subgroup.
            "n_seeds_with_ratio": stats["subgroup_n"],
            # The narrowing, emitted rather than silent (H.subgroup_ratio_stats
            # contract). ratio_subgroup_basis names the predicate in words so the
            # manifest is readable without the source.
            "ratio_subgroup_n": stats["subgroup_n"],
            "ratio_subgroup_n_eligible": stats["n_eligible"],
            "ratio_subgroup_n_non_finite": stats["n_non_finite"],
            "ratio_excluded_seeds": stats["excluded_seeds"],
            "ratio_n_excluded": stats["n_excluded"],
            "ratio_subgroup_basis": (
                "on-scale seeds only (null_slope_ratio_rem_off_scale clear)"
                if p == "rem"
                else "all seeds (no off-scale condition on this phase)"
            ),
            # Audit trail: ALL seeds, unscoped, so the exclusion is checkable.
            "per_seed_null_slope_ratio": [s["null_slope_ratio"][p] for s in seed_scores],
            "n_seeds_confounded": n_conf,
            "confounded_all_seeds": bool(n_conf == n),
            "confound_verdict_stable": bool(n_conf == 0 or n_conf == n),
            # A ceiling INSIDE the CI means the confound verdict is not resolved for
            # this phase at this n -- reported so it cannot read as a clean verdict.
            # With an empty subgroup there is no interval and this degrades to False;
            # ratio_subgroup_n == 0 is then the readout that carries the information.
            "ceiling_inside_ci95": stats["ceiling_inside_ci95"],
        }
        if p == "rem":
            phase_summary[p]["per_seed_rem_ratio_off_scale"] = [
                bool(s["rem_ratio_off_scale"]) for s in seed_scores
            ]
            phase_summary[p]["per_seed_null_rem_target_clamped_frac"] = [
                s["null_rem_target_clamped_frac"] for s in seed_scores
            ]
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
        # Provenance for the setup-time guard: proof, recorded in the shipped artifact,
        # that this run's declared readiness precondition is reachable by its own
        # known-damaged positive control under the SHIPPED predicate.
        "anchor_reachability_guard": anchor_guard,
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
                "step=1.0), NEVER as a calibrated N-fold noise sensitivity. Because "
                "such a ratio is not on a common scale with the on-scale seeds', the "
                "rem MAGNITUDE statistics (mean/sd/ci95/ceiling_inside_ci95) are "
                "scoped to the on-scale subgroup -- see per_phase.rem."
                "ratio_subgroup_n / ratio_excluded_seeds, and note that "
                "ratio_subgroup_n == 0 means NO on-scale rem ratio exists at this n, "
                "which is a stronger statement than a wide interval. The per-seed "
                "confound VERDICT is unscoped: off-scale bears on magnitude, not on "
                "whether the phase is confounded."
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
            f"sd={_fmt(ps['sd_null_slope_ratio'])} "
            f"ci95=[{_fmt(ps['ci95_low'])}, {_fmt(ps['ci95_high'])}] "
            f"confounded_seeds={ps['n_seeds_confounded']}/{n} "
            f"stable={ps['confound_verdict_stable']}"
            # The narrowing is announced on the console too, not only in the manifest:
            # a mean silently computed over a subset is the hazard H.subgroup_ratio_stats
            # exists to prevent, and n/a with no reason given is its quieter cousin.
            + (f" [magnitude over {ps['ratio_subgroup_n']}/{n} on-scale seeds;"
               f" excluded {ps['ratio_excluded_seeds']}]"
               if ps["ratio_n_excluded"] else "")
            + ("  [CEILING INSIDE CI -- verdict unresolved at this n]"
               if ps["ceiling_inside_ci95"] else ""),
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
