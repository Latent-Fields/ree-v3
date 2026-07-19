"""
V3-EXQ-778g: SD-068 validation of the REBUILT sws content-scored readout.
SLEEP DRIVER: manual-cycle-loop (the SD-068 harness drives enter_sws_mode /
run_sws_schema_pass directly per phase readout; no SleepLoopManager scheduling).

WHAT THIS MEASURES
------------------
V3-EXQ-778c ran the zero-injected-content null control and found the sws leg
CONTENT-FREE: `null_slope_ratio_sws` = 1.0000 (sd 2.7e-8) on 8/8 seeds. That was an
ANALYTIC identity, not a statistical near-miss. `denoising_snr_db` is
`10*log10(signal_power / noise_power)` where `noise_power = ||shy(damaged) -
shy(clean)||^2`; `_shy` is AFFINE, so `shy(clean + n) - shy(clean) = shy_centred(n)`
exactly, independent of `clean`. The corruption is referenced to the UNSCALED content
(`diffuse_perturb(rms_ref=...)`), so it is numerically identical across arms, and the
common-units series divides both arms by the same injected denominator. The content
term is a constant offset that DIFFERENTIATES AWAY. The readout never measured content
fidelity, and the staging order built on it was never evidence of staging.

The routed repair (`/implement-substrate`, ree-v3 main 8b18338) replaced the SCORED
sws series with `_sws_pattern_completion`: the cosine retrieval margin of the post-SHY
store against the injected prototypes,

    sims[i, j] = cos(probe_i, shy(store)_j)
    margin_i   = sims[i, i] - max_{j != i} sims[i, j]

which is a RATIO of correct-vs-incorrect similarity whose denominators carry `clean`,
so the affine cancellation no longer removes the content term. Probes are the UNSCALED
prototypes, so the null arm receives a real, arm-identical probe that is simply not
planted -- Bar et al. 2020's "same odour delivered, no prior pairing" -- rather than a
zero vector, which would make the readout 0/0-degenerate. That degeneracy is exactly
the rem leg's existing failure mode (5/8 of its "unconfounded" seeds are unconfounded
only BY DEGENERACY, off a saturated constant), and avoiding it is the point.

THIS RUN ASKS ONE QUESTION: does the repaired sws readout pass the null control it
previously failed 8/8?

WHY THE SWS LEG ONLY IS GATED (design decision, stated so it cannot be mistaken)
-------------------------------------------------------------------------------
778c's C1 required ALL THREE phases content-contingent. Re-using that here would be
uninformative: the rem leg is KNOWN degenerate at both clamp rails (778c: exactly 0.0
on 5/8 seeds off a saturated constant, off-scale 1801-9143 on 3/8), so an all-phase C1
would FAIL regardless of whether the sws repair worked, and the run would answer
nothing about the thing it was queued to test. The rem leg is owned by the GOV-FANOUT-1
portfolio V3-EXQ-778d/e/f and is deliberately NOT gated here.

So C1 is scoped to `sws`. The `nrem` and `rem` legs are still MEASURED and REPORTED in
full (nrem as the known-good comparator at ~0.1445, rem as the known-degenerate one) --
reporting them is what lets a reader see that the sws leg moved while the others did
not. They are context, not criteria.

THE CAVEAT THIS RUN EXISTS TO CLOSE (C3 -- do not drop it)
----------------------------------------------------------
The repaired readout is COSINE-BASED and therefore SCALE-INVARIANT. In the null arm the
store is `0 + sigma*noise`, so the sigma factor cancels out of the cosine entirely and
the null arm is flat in sigma PARTLY BY CONSTRUCTION. A low `null_slope_ratio_sws` is
therefore partly IMPLIED BY THE READOUT'S FORM rather than independently measured, which
makes the binary null control a weaker check here than the ratio alone suggests. Local
smoke saw exactly this: the null-arm margin was identical to six decimal places across
four sigmas.

That is the same species of tell 778c taught us to distrust -- a variance orders of
magnitude below its siblings is an instrument-validity flag, not a strength-of-effect
signal. So this run does NOT rest on the null ratio alone. C3 is a CONTENT-SCALE LADDER
(`content_scale` in {0.0, 0.25, 0.5, 1.0}) asking the independent question: does the
sigma-response track the AMOUNT of planted content?

  (a) zero sigma-response at content_scale = 0        -- the null-control property;
  (b) large response at every content_scale > 0        -- the readout is not inert;
  (c) response VARIES with content amplitude           -- it tracks content, rather
      than merely switching on the presence of a non-zero store.

NOTE ON THE EXPECTED DIRECTION of (c): the sigma-slope DECREASES as content strengthens
(local smoke: 0.449 / 0.430 / 0.325 at content_scale 0.25 / 0.5 / 1.0). That is the
correct physics, not a defect -- damage is referenced to `_rms(base)`, the UNSCALED
prototypes, so it is held at full strength regardless of content amplitude; weakly
planted content faces a proportionally larger perturbation and is destroyed faster. C3
therefore tests for VARIATION and a zero-content floor, NOT for a monotone increase. An
earlier draft of this criterion asserted monotone-increasing and was wrong.

WHY DIAGNOSTIC (not evidence)
-----------------------------
This is an instrument-validity control on SD-068, not a test of a substrate hypothesis.
It does NOT weight governance confidence. It tags SD-068 (the harness whose readout it
validates) and the three staging claims the harness serves (MECH-168 / INV-047 /
MECH-169) as CONTEXT only. It deliberately does NOT tag MECH-121: MECH-121 is
candidate/substrate_conditional (hold_pending_v3_substrate), the hold is in force, and
nothing here is MECH-121 behavioural validation.

ACCEPTANCE (pre-registered)
---------------------------
  C1 (LOAD-BEARING): `null_slope_ratio_sws` <= NULL_SLOPE_RATIO_CEILING (0.25) on
     >= PASS_FRACTION of seeds. This is the criterion 778c failed 8/8.
  C2 (readiness / positive control): the ratio is INTERPRETABLE -- the injected-arm sws
     sigma-slope (the ratio's DENOMINATOR, i.e. the SAME statistic C1 routes on) clears
     INJECTED_SLOPE_FLOOR. A below-floor denominator means the sweep never damaged the
     readout, so the control cannot discriminate -> `substrate_not_ready_requeue`,
     NEVER a substrate verdict.
  C3 (anti-artifact, LOAD-BEARING): the content-scale ladder shows (a) SIGNAL --
     |slope| at every content_scale>0 exceeds LADDER_SIGNAL_RATIO x |slope| at
     content_scale=0; and (b) SPREAD -- the spread across the content>0 slopes exceeds
     LADDER_SPREAD_FLOOR. Both are RELATIVE tests. An absolute floor on the
     zero-content slope was tried and rejected at authoring time: the zero-content arm
     has a real discontinuity at sigma=0 (store exactly zero) which gives it a small
     nonzero fitted slope no absolute bound can sensibly cover. See the
     LADDER_SIGNAL_RATIO calibration note.

A FAIL is INFORMATIVE, not a broken run:
  C1 fail            -> the repair did not work; the sws leg is still content-free.
  C3 fail with C1 pass -> the null-control pass is a scale-invariance artifact and the
                          readout does not actually track content. This is the outcome
                          the local smoke could not rule out, and is the reason C3 is
                          pre-registered rather than added post-hoc.
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

EXPERIMENT_TYPE = "v3_exq_sd068_sws_content_scored_readout_diagnostic"
QUEUE_ID = "V3-EXQ-778g"
# NOT a supersession. 778c's finding stands entirely -- it correctly established that the
# OLD readout was content-free. This validates the REPLACEMENT readout, a different
# instrument. Marking 778c superseded would erase the finding that motivated this build.
SUPERSEDES = None
# SD-068 = the harness whose readout this validates. The three staging claims are
# context only. MECH-121 deliberately ABSENT (held; nothing here is MECH-121 evidence).
CLAIM_IDS: List[str] = ["SD-068", "MECH-168", "INV-047", "MECH-169"]
EXPERIMENT_PURPOSE = "diagnostic"
SLEEP_DRIVER_PATTERN = "manual-cycle-loop"

# The V3-EXQ-778a / 778c 8-seed set, reused EXACTLY so the repaired readout's ratios
# pool directly onto 778c's recorded per-seed distribution for the SAME seeds. This is
# what makes "1.0000 before, X after" a within-seed comparison rather than two
# independent samples.
SEEDS = [42, 7, 123, 2024, 99, 7777, 314, 1000]
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
WARM_STEPS = 40
ARMS = ["INJECTED", "NULL"]
# Ladder rungs. 0.0 and 1.0 are ALREADY computed by the main sweep (they are the null
# and injected arms), and sws_only_integrity_at_sigma reproduces that cell exactly, so
# only the intermediate rungs cost extra compute.
LADDER_SCALES = [0.0, 0.25, 0.5, 1.0]
LADDER_EXTRA_SCALES = [0.25, 0.5]

# Pre-registered thresholds.
NULL_SLOPE_RATIO_CEILING = H.NULL_SLOPE_RATIO_CEILING  # 0.25
INJECTED_SLOPE_FLOOR = 1e-6   # C2 readiness: min |injected sws slope| for interpretability
PASS_FRACTION = 1.0           # ALL seeds must be clean for C1 (a control, not a vote)
# C3 thresholds. CALIBRATION NOTE (recorded so these are auditable rather than tuned):
# the first draft gated C3(a) on an ABSOLUTE floor (|slope| at content_scale=0 below
# 1e-6). The authoring dry-run showed that is the wrong test. The zero-content arm has a
# genuine DISCONTINUITY at sigma=0 -- the store is EXACTLY zero there (margin 0.0) but
# pure noise at every sigma>0 (margin ~-0.107, then flat, because cosine is
# scale-invariant). That single step gives the zero-content rung a small but nonzero
# fitted slope (~0.065 in the dry-run) which no absolute floor can sensibly bound.
#
# The principled test is RELATIVE: with no content planted there is nothing to lose, so
# the content-bearing response must be SEVERAL-FOLD the content-free one. 3.0x is that
# "several-fold" -- chosen as a conventional bar BEFORE the real run, not fitted to the
# observed value (the dry-run separation was ~6.8x, so there is real headroom; if the
# full run lands near 3x that is a genuine warning sign and C3 should fail).
# The absolute zero-slope magnitude is still RECORDED, just not gated on.
LADDER_SIGNAL_RATIO = 3.0     # C3(a): content>0 slopes must exceed this x the zero slope
LADDER_SPREAD_FLOOR = 0.01    # C3(b): spread across content>0 slopes must exceed this

# The phase C1 gates on. nrem/rem are measured and reported as CONTEXT, never gated --
# see "WHY THE SWS LEG ONLY IS GATED" in the module docstring.
GATED_PHASE = "sws"
CONTEXT_PHASES = ("nrem", "rem")


def _fmt(v: float) -> str:
    """ASCII-safe float rendering that keeps UNAVAILABLE legible."""
    if v == H.UNAVAILABLE or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.6f}"


def _finite(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and v != H.UNAVAILABLE
        and not math.isnan(float(v))
    )


def _ladder_slope(
    *, seed: int, content_scale: float, sigmas: List[float], warm: int,
    cached: Dict[float, Dict[str, Dict[str, float]]] = None,
) -> Tuple[float, List[float]]:
    """Sigma-slope of the sws completion error at one content_scale.

    Uses the INJECTED arm's own undamaged margin at this rung as the denominator, so
    each rung is expressed in its own units and the rungs are directly comparable as
    fractions-of-own-discriminability-lost. `cached` supplies already-computed sws rows
    (the main sweep's null / injected arms) so rungs 0.0 and 1.0 cost nothing.
    """
    margins: List[float] = []
    m_clean = None
    for s in sigmas:
        if cached is not None and s in cached:
            row = cached[s]["sws"]
        else:
            row = H.sws_only_integrity_at_sigma(
                seed=seed, sigma=s, warm_steps=warm, content_scale=content_scale
            )["sws"]
        margins.append(float(row.get("sws_completion_margin", float("nan"))))
        if m_clean is None:
            m_clean = float(row.get("sws_completion_margin_clean", 0.0))

    if m_clean is None or abs(m_clean) <= 1e-9:
        # content_scale = 0 -> no injected discriminability to lose. The error series is
        # referenced to the margin itself, which is ~0 at every sigma, so the slope is
        # ~0. That is the C3(a) property, measured rather than assumed.
        errs = [(-m) for m in margins]
    else:
        errs = [1.0 - (m / m_clean) for m in margins]

    xs = [s for s, e in zip(sigmas, errs) if not math.isnan(e)]
    ys = [e for e in errs if not math.isnan(e)]
    slope = float(H._lin_slope(xs, ys)) if len(ys) >= 2 else float("nan")
    return slope, errs


def _score_seed(
    control: Dict[str, float], ladder: Dict[float, float]
) -> Dict[str, Any]:
    """Score one seed: C1 on sws, C2 readiness, C3 ladder. nrem/rem recorded as context."""
    ratio = control.get(f"null_slope_ratio_{GATED_PHASE}", H.UNAVAILABLE)
    inj = control.get(f"injected_slope_{GATED_PHASE}", H.UNAVAILABLE)
    null = control.get(f"null_slope_{GATED_PHASE}", H.UNAVAILABLE)

    # C2 readiness asserts the SAME statistic C1 routes on: the ratio's denominator.
    c2 = _finite(inj) and abs(float(inj)) >= INJECTED_SLOPE_FLOOR
    c1 = bool(_finite(ratio) and float(ratio) <= NULL_SLOPE_RATIO_CEILING)

    zero_slope = ladder.get(0.0, float("nan"))
    pos = [ladder[c] for c in LADDER_SCALES if c > 0.0 and _finite(ladder.get(c))]
    # C3(a) SIGNAL -- relative, not absolute: see the LADDER_SIGNAL_RATIO calibration
    # note. The zero-content rung carries a small nonzero slope from the sigma=0
    # store-is-exactly-zero discontinuity, so what must hold is that every
    # content-bearing rung responds SEVERAL-FOLD more strongly than the content-free one.
    c3a = bool(pos) and _finite(zero_slope) and min(abs(v) for v in pos) > (
        LADDER_SIGNAL_RATIO * max(abs(zero_slope), 1e-12)
    )
    # C3(b) SPREAD -- the response must VARY with content amplitude, not merely switch
    # on. This is what separates "tracks content" from "detects a non-empty store".
    c3b = bool(pos) and (max(pos) - min(pos)) > LADDER_SPREAD_FLOOR
    c3 = bool(c3a and c3b)

    return {
        "null_slope_ratio_sws": ratio,
        "injected_slope_sws": inj,
        "null_slope_sws": null,
        "context_null_slope_ratio": {
            p: control.get(f"null_slope_ratio_{p}", H.UNAVAILABLE)
            for p in CONTEXT_PHASES
        },
        "context_content_contingent": {
            p: bool(control.get(f"content_contingent_{p}", 0.0) >= 1.0)
            for p in CONTEXT_PHASES
        },
        "ladder_slopes": {str(k): v for k, v in ladder.items()},
        "C1_sws_content_contingent": c1,
        "C2_ratio_interpretable": c2,
        "C3_ladder_tracks_content": c3,
        "C3_detail": {"signal": c3a, "spread": c3b, "zero_slope": zero_slope},
        "seed_pass": bool(c1 and c2 and c3),
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    warm = 8 if dry_run else WARM_STEPS
    sigmas = [0.0, 0.5, 2.0] if dry_run else SIGMAS
    extra_scales = [0.5] if dry_run else LADDER_EXTRA_SCALES
    ladder_scales = [0.0] + extra_scales + [1.0]

    print("V3-EXQ-778g: SD-068 rebuilt sws content-scored readout validation", flush=True)
    print(
        f"  seeds={seeds} sigmas={sigmas} warm_steps={warm} arms={ARMS} "
        f"ladder={ladder_scales} dry_run={dry_run}",
        flush=True,
    )

    config_slice = {
        "sigmas": sigmas,
        "warm_steps": warm,
        "arms": list(ARMS),
        "ladder_scales": ladder_scales,
        "null_slope_ratio_ceiling": NULL_SLOPE_RATIO_CEILING,
        "injected_slope_floor": INJECTED_SLOPE_FLOOR,
        "ladder_signal_ratio_note": (
            "C3 uses RELATIVE tests only; an absolute zero-content floor was tried and "
            "rejected at authoring time (sigma=0 store-is-exactly-zero discontinuity)"
        ),
        "ladder_signal_ratio": LADDER_SIGNAL_RATIO,
        "ladder_spread_floor": LADDER_SPREAD_FLOOR,
        "gated_phase": GATED_PHASE,
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
        print(f"Seed {seed} Condition SWS_READOUT_VALIDATION", flush=True)
        with arm_cell(
            seed,
            config_slice=config_slice,
            script_path=Path(__file__),
            config_slice_declared=True,
        ) as cell:
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
                    f"  [train] sws_validation seed={seed} ep {i + 1}/{total_eps} "
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

            # C3 ladder. Rungs 0.0 and 1.0 REUSE the main sweep's cells (identical RNG
            # stream by construction -- sws_only_integrity_at_sigma reproduces
            # phase_integrity_at_sigma's sws cell exactly), so only the intermediate
            # rungs cost extra compute.
            ladder: Dict[float, float] = {}
            ladder_series: Dict[str, List[float]] = {}
            for cs in ladder_scales:
                cache = None
                if cs == 0.0:
                    cache = null_pr
                elif cs == 1.0:
                    cache = inj_pr
                slope, errs = _ladder_slope(
                    seed=seed, content_scale=cs, sigmas=list(sigmas),
                    warm=warm, cached=cache,
                )
                ladder[cs] = slope
                ladder_series[str(cs)] = errs
                print(
                    f"  [ladder] seed={seed} content_scale={cs} sigma_slope={_fmt(slope)}",
                    flush=True,
                )

            score = _score_seed(control, ladder)
            row: Dict[str, Any] = {
                "seed": seed,
                "arm": "SWS_READOUT_VALIDATION",
                "arms_compared": list(ARMS),
                "sigmas": list(sigmas),
                "null_control": control,
                "ladder_scales": ladder_scales,
                "ladder_slopes": score["ladder_slopes"],
                "ladder_error_series": ladder_series,
                "null_slope_ratio_sws": score["null_slope_ratio_sws"],
                "injected_slope_sws": score["injected_slope_sws"],
                "null_slope_sws": score["null_slope_sws"],
                "context_null_slope_ratio": score["context_null_slope_ratio"],
                "context_content_contingent": score["context_content_contingent"],
                # Both arms' full per-sigma internals recorded, per the Experimental
                # Recording Standard (the OFF/NULL arm as richly as the INJECTED one).
                "integrity_injected": {str(s): inj_pr[s] for s in sigmas},
                "integrity_null": {str(s): null_pr[s] for s in sigmas},
                "C1_sws_content_contingent": score["C1_sws_content_contingent"],
                "C2_ratio_interpretable": score["C2_ratio_interpretable"],
                "C3_ladder_tracks_content": score["C3_ladder_tracks_content"],
                "C3_detail": score["C3_detail"],
                "seed_pass": score["seed_pass"],
            }
            cell.stamp(row)

        arm_results.append(row)
        seed_scores.append(score)

        ctx = score["context_null_slope_ratio"]
        print(
            f"  null_slope_ratio: sws={_fmt(score['null_slope_ratio_sws'])} "
            f"(GATED, ceiling {NULL_SLOPE_RATIO_CEILING}) | context: "
            f"nrem={_fmt(ctx['nrem'])} rem={_fmt(ctx['rem'])}",
            flush=True,
        )
        print(
            f"  C1={score['C1_sws_content_contingent']} "
            f"C2={score['C2_ratio_interpretable']} "
            f"C3={score['C3_ladder_tracks_content']} {score['C3_detail']}",
            flush=True,
        )
        print(f"verdict: {'PASS' if score['seed_pass'] else 'FAIL'}", flush=True)

    n = len(seed_scores)
    need = math.ceil(PASS_FRACTION * n)
    n_pass = sum(1 for s in seed_scores if s["seed_pass"])
    readiness_ok = all(s["C2_ratio_interpretable"] for s in seed_scores)
    c1_all = all(s["C1_sws_content_contingent"] for s in seed_scores)
    c3_all = all(s["C3_ladder_tracks_content"] for s in seed_scores)
    # C3's legs, aggregated separately for the two recomputable preconditions.
    # c3_all == (c3a_all and c3b_all): `all` over a conjunction distributes.
    c3a_all = all(s["C3_detail"]["signal"] for s in seed_scores)
    c3b_all = all(s["C3_detail"]["spread"] for s in seed_scores)
    overall_pass = bool(readiness_ok and n_pass >= need)

    ratios = [
        float(s["null_slope_ratio_sws"])
        for s in seed_scores
        if _finite(s["null_slope_ratio_sws"])
    ]
    if ratios:
        mean_r = sum(ratios) / len(ratios)
        if len(ratios) >= 2:
            var = sum((v - mean_r) ** 2 for v in ratios) / (len(ratios) - 1)
            sd_r = math.sqrt(var)
            sem = sd_r / math.sqrt(len(ratios))
            ci_lo, ci_hi = mean_r - 1.96 * sem, mean_r + 1.96 * sem
        else:
            sd_r = ci_lo = ci_hi = H.UNAVAILABLE
    else:
        mean_r = sd_r = ci_lo = ci_hi = H.UNAVAILABLE

    # A CI straddling the ceiling means the verdict is UNRESOLVED at this n -- the same
    # treatment 778c gave its rem leg. Reported so a mean under the ceiling cannot be
    # read as a clean pass when the interval says otherwise.
    ceiling_inside_ci95 = bool(
        _finite(ci_lo) and _finite(ci_hi)
        and ci_lo <= NULL_SLOPE_RATIO_CEILING <= ci_hi
    )

    sws_summary = {
        "mean_null_slope_ratio": mean_r,
        "sd_null_slope_ratio": sd_r,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "per_seed_null_slope_ratio": [s["null_slope_ratio_sws"] for s in seed_scores],
        "per_seed_injected_slope": [s["injected_slope_sws"] for s in seed_scores],
        "n_seeds_content_contingent": sum(
            1 for s in seed_scores if s["C1_sws_content_contingent"]
        ),
        "ceiling_inside_ci95": ceiling_inside_ci95,
        "prior_778c_ratio": 1.0000,
        "prior_778c_sd": 2.7e-8,
    }
    context_summary = {
        p: {
            "per_seed_null_slope_ratio": [
                s["context_null_slope_ratio"][p] for s in seed_scores
            ],
            "n_seeds_content_contingent": sum(
                1 for s in seed_scores if s["context_content_contingent"][p]
            ),
            "gated": False,
        }
        for p in CONTEXT_PHASES
    }

    # Self-route. Readiness dominates: a below-floor denominator means the control never
    # discriminated, which is a requeue, NEVER a substrate verdict.
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
    elif not c1_all:
        label = "sws_readout_still_content_free"
    elif not c3_all:
        # C1 passed but the ladder did not -- exactly the scale-invariance artifact C3
        # was pre-registered to catch.
        label = "sws_null_pass_is_scale_invariance_artifact"
    else:
        label = "sws_readout_content_contingent_validated"

    sd068_direction = "supports" if overall_pass else "weakens"
    per_claim = {
        "SD-068": sd068_direction,
        "MECH-168": "unknown",
        "INV-047": "unknown",
        "MECH-169": "unknown",
    }

    min_inj = min(
        (abs(float(s["injected_slope_sws"])) for s in seed_scores
         if _finite(s["injected_slope_sws"])),
        default=0.0,
    )
    # C3's two legs are aggregated SEPARATELY because each is declared as its own
    # adjudication precondition. `C3_ladder_tracks_content` is `c3a and c3b`, so a
    # single precondition carrying only the spread statistic could not reproduce
    # `met` from its own (measured, threshold) pair -- the signal leg would be
    # undeclared and the indexer's recompute would silently adjudicate on half the
    # check. A seed with NO content-bearing rung fails both legs by construction
    # (`bool(pos)` guards each), so it contributes a 0.0 worst case rather than
    # being skipped -- otherwise the min over seeds could clear a floor that the
    # shipped predicate did not.
    ladder_spreads = []
    ladder_signal_ratios = []
    for s in seed_scores:
        pos = [
            v for k, v in s["ladder_slopes"].items()
            if float(k) > 0.0 and _finite(v)
        ]
        zero_slope = s["C3_detail"]["zero_slope"]
        # c3b guards on bool(pos) alone; c3a additionally needs a finite zero rung.
        # The guards are mirrored separately so each reported statistic reproduces
        # its OWN predicate, not the conjunction.
        ladder_spreads.append(max(pos) - min(pos) if pos else 0.0)
        ladder_signal_ratios.append(
            min(abs(v) for v in pos) / max(abs(float(zero_slope)), 1e-12)
            if pos and _finite(zero_slope) else 0.0
        )

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "injected_arm_sws_sigma_slope_supra_floor",
                "description": (
                    "The ratio's DENOMINATOR -- the same statistic C1 routes on. "
                    "Measured on the known-damaged injected arm (the positive "
                    "control): if the sigma sweep never moved the repaired readout, "
                    "the ratio is 0/0 and the control cannot discriminate."
                ),
                "control": "injected arm (content_scale=1.0) across the full sigma grid",
                "measured": float(min_inj),
                "threshold": float(INJECTED_SLOPE_FLOOR),
                # FLOOR-shaped, INCLUSIVE: C2 is `abs(inj) >= INJECTED_SLOPE_FLOOR`
                # and `measured` is the min over seeds of that same absolute slope.
                "comparator": ">=",
                "direction": "lower",
                "met": bool(readiness_ok),
            },
            {
                # C3(a). Declared SEPARATELY from the spread leg below: C3 is
                # `signal and spread`, two different statistics, so one entry could
                # not carry both and the undeclared leg would drop out of the
                # indexer's recompute entirely.
                "name": "ladder_content_signal_ratio_supra_floor",
                "description": (
                    "C3(a). The content-bearing rungs must respond SEVERAL-FOLD "
                    "more strongly in sigma than the content-free rung. Relative, "
                    "not absolute: the zero-content rung carries a small nonzero "
                    "slope from the sigma=0 store-is-exactly-zero discontinuity, so "
                    "an absolute floor here would be met by that artifact alone."
                ),
                "control": "content_scale ladder on the injected path",
                "measured": float(min(ladder_signal_ratios)) if ladder_signal_ratios else 0.0,
                "threshold": float(LADDER_SIGNAL_RATIO),
                # FLOOR-shaped, STRICT: c3a is `min|pos| > RATIO * max(|zero|, 1e-12)`,
                # reported here in already-divided ratio form so the comparison is a
                # plain (measured, threshold) pair.
                "comparator": ">",
                "direction": "lower",
                "met": bool(c3a_all),
            },
            {
                "name": "ladder_content_slope_spread_supra_floor",
                "description": (
                    "C3(c). Guards the scale-invariance artifact: a cosine readout is "
                    "flat in sigma without content BY CONSTRUCTION, so the null ratio "
                    "alone is partly implied by the readout's form. If the sigma-slope "
                    "does not VARY with content amplitude, the readout is not tracking "
                    "content and a low null ratio means nothing."
                ),
                "control": "content_scale ladder on the injected path",
                "measured": float(min(ladder_spreads)) if ladder_spreads else 0.0,
                "threshold": float(LADDER_SPREAD_FLOOR),
                # FLOOR-shaped, STRICT: c3b is `(max(pos) - min(pos)) > LADDER_SPREAD_FLOOR`.
                # `met` is the SPREAD leg alone -- previously it carried the full C3
                # conjunction, which no (measured, threshold) pair on the spread
                # statistic could reproduce. The conjunction still routes the label
                # via criteria_non_degenerate["C3"] / c3_all; it is now expressed to
                # the adjudicator as two recomputable preconditions that must BOTH
                # hold, which is the same predicate.
                "comparator": ">",
                "direction": "lower",
                "met": bool(c3b_all),
            },
        ],
        "criteria_non_degenerate": {
            # C1 is degenerate if its denominator never cleared the floor (0/0).
            "C1": bool(readiness_ok),
            "C2": bool(_finite(min_inj)),
            # C3 is degenerate if the ladder produced no usable content>0 slopes.
            "C3": bool(ladder_spreads),
        },
        "criteria": [
            {
                "name": "C1_sws_content_contingent",
                "load_bearing": True,
                "passed": bool(c1_all),
            },
            {"name": "C2_ratio_interpretable", "load_bearing": False,
             "passed": bool(readiness_ok)},
            {"name": "C3_ladder_tracks_content", "load_bearing": True,
             "passed": bool(c3_all)},
        ],
        "gated_phase": GATED_PHASE,
        "context_phases_not_gated": list(CONTEXT_PHASES),
        "rem_leg_owner": "V3-EXQ-778d/e/f (GOV-FANOUT-1 portfolio)",
    }

    print("", flush=True)
    print(
        f"seeds pass: {n_pass}/{n} (need {need}) -> overall "
        f"{'PASS' if overall_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  sws (GATED): mean null_slope_ratio={_fmt(mean_r)} sd={_fmt(sd_r)} "
        f"ci95=[{_fmt(ci_lo)}, {_fmt(ci_hi)}] "
        f"content_contingent_seeds={sws_summary['n_seeds_content_contingent']}/{n}"
        + ("  [CEILING INSIDE CI -- verdict unresolved at this n]"
           if ceiling_inside_ci95 else ""),
        flush=True,
    )
    print(f"  prior 778c sws ratio was 1.0000 (sd 2.7e-8) on 8/8 seeds", flush=True)
    for p in CONTEXT_PHASES:
        cs = context_summary[p]
        print(
            f"  {p:>4} (context, NOT gated): content_contingent_seeds="
            f"{cs['n_seeds_content_contingent']}/{n}",
            flush=True,
        )
    print(f"self-route label: {label}", flush=True)

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": sd068_direction,
        "evidence_direction_per_claim": per_claim,
        "interpretation": interpretation,
        "arm_results": arm_results,
        "sws_summary": sws_summary,
        "context_summary": context_summary,
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
        "sws_summary": result["sws_summary"],
        "context_summary": result["context_summary"],
        "acceptance_criteria": {
            "C1_sws_content_contingent": (
                f"null_slope_ratio_sws <= {NULL_SLOPE_RATIO_CEILING} on "
                f">= {PASS_FRACTION:.2f} of seeds (LOAD-BEARING; the criterion "
                "V3-EXQ-778c failed 8/8 on the retired readout)"
            ),
            "C2_ratio_interpretable": (
                f"|injected sws sigma-slope| >= {INJECTED_SLOPE_FLOOR} (readiness; "
                "asserts the ratio's denominator, the same statistic C1 routes on; "
                "below-floor -> substrate_not_ready_requeue)"
            ),
            "C3_ladder_tracks_content": (
                f"content-scale ladder, RELATIVE tests: |slope| at every "
                f"content_scale>0 > {LADDER_SIGNAL_RATIO}x |slope| at content_scale=0; "
                f"AND spread across content>0 slopes > {LADDER_SPREAD_FLOOR} "
                "(LOAD-BEARING anti-artifact: the readout is cosine-based and therefore "
                "scale-invariant, so its null arm is flat in sigma partly BY "
                "CONSTRUCTION and the null ratio alone is a weaker check than it looks)"
            ),
        },
        "arm_results": result["arm_results"],
        "notes": (
            "SD-068 validation of the REBUILT sws content-scored readout "
            "(_sws_pattern_completion, landed ree-v3 main 8b18338 via "
            "/implement-substrate per failure_autopsy_V3-EXQ-778c_2026-07-18). "
            "V3-EXQ-778c found the OLD sws readout content-free: null_slope_ratio 1.0000 "
            "(sd 2.7e-8) on 8/8 seeds, an ANALYTIC identity -- _shy is affine, so "
            "noise_power is independent of the injected content and the content term "
            "differentiates away. The replacement scores a cosine retrieval margin of "
            "the post-SHY store against the injected prototypes, probed with the "
            "UNSCALED base so the null arm gets a real arm-identical probe that is "
            "simply not planted (Bar et al. 2020 'same odour, no prior pairing') rather "
            "than a 0/0-degenerate zero. NOT a supersession of 778c: 778c's finding "
            "stands and motivated this build; this validates a DIFFERENT instrument. "
            "C1 is scoped to the sws leg ALONE -- the rem leg is known degenerate at "
            "both clamp rails and is owned by the GOV-FANOUT-1 portfolio 778d/e/f, so "
            "gating on all three phases would FAIL regardless of whether the sws repair "
            "worked and would answer nothing. nrem/rem are measured and reported as "
            "context. C3 (content-scale ladder) is pre-registered because the repaired "
            "readout is cosine-based and therefore scale-invariant: its null arm is flat "
            "in sigma partly by construction, so a low null ratio is partly implied by "
            "the readout's form and needs an independent content-tracking check. "
            "MECH-121 deliberately NOT tagged (hold_pending_v3_substrate in force). "
            "Experiment-layer only; zero ree_core change; no substrate_queue entry "
            "(autopsy action: none). GOV-REUSE-1: decisive readout "
            "(null_slope_ratio_sws on _sws_pattern_completion) is carried by ZERO "
            "existing manifests -- the readout did not exist before 8b18338 landed "
            "today and the substrate_hash changed with it, so it is not recoverable "
            "from any recorded run -> ran."
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
