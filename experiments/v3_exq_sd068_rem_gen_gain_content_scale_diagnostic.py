"""
V3-EXQ-778f: SD-068 GOV-FANOUT-1 portfolio leg 3 of 3 -- REM generative-gain content
scaling. Hypothesis under test: H-gen-gain-content-free (axis: OBSERVATION).
SLEEP DRIVER: manual-cycle-loop (the SD-068 harness drives enter_rem_mode /
run_rem_attribution_pass + the e2.rollout_with_world re-derivation directly per phase
readout; no SleepLoopManager scheduling).

WHAT THIS MEASURES
------------------
This leg does NOT threaten the finding that the REM generative pass ATTENUATES. That
finding -- `rem_generative_gain` 0.149, attenuating on 8/8 seeds -- STANDS, and nothing
here can overturn it: the transfer function does attenuate. What is under test is the
narrower INTERPRETIVE GLOSS placed on it, that "the correction needs an intact seed"
(`REE_assembly/docs/architecture/sd_068_consolidation_lesion_harness.md`, the OPEN
QUESTION box in the "REM generative gain" section, flagged 2026-07-18). Keeping those two
apart is the whole point of this leg -- a result here revises a sentence of
interpretation, not a measured quantity.

THE OBSERVATION THAT MOTIVATES IT. In the V3-EXQ-778c manifest the NULL arm's
`rem_generative_gain` (seed 42: 0.182 / 0.184 / 0.188 / 0.209 across sigma) is CLOSE to
the INJECTED arm's (0.165 / 0.166 / 0.172 / 0.190) at `rem_gen_content_scale` 0.0 -- that
is, with a CONTENT-FREE (zero) rollout seed. If attenuation occurs just as strongly with
no seed content at all, then attenuation is a property of the rollout transfer function
rather than evidence that the correction needs an intact seed. This sits OUTSIDE the
scored C1 criteria of 778c and was recorded as an open question, NOT a verdict. It is
n=1-seed, eyeballed, and unpowered; this leg is what turns it into a measurement.

This is leg 3 of the GOV-FANOUT-1 discrimination portfolio routed by
`REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-778c_2026-07-18.json`
(`targets[0].fanout_recommendation`). It attacks the OBSERVATION axis -- a DIFFERENT
readout entirely (the generative rollout) from the calibration readout its two siblings
probe. Siblings: V3-EXQ-778d (measurement) and V3-EXQ-778e (representation). They run in
PARALLEL, not in sequence.

HYPOTHESES UNDER TEST
---------------------
H-gen-gain-content-free (THIS LEG, axis=observation, pre-registered
  2026-07-18T08:41:15Z in hypothesis_space_registry.v1.json question
  `consolidation_readout_validity`):
    CLAIM: `rem_generative_gain` is a CONTENT-FREE property of the rollout transfer
    function, NOT evidence that "correction needs an intact seed".
    PROBE: contrast the generative gain at `rem_gen_content_scale` in {0.0, 0.5, 1.0}
    across seeds; test whether gain varies with content scale AT ALL.
    EVIDENCE FOR: gain is FLAT in content scale -- |gain(1.0) - gain(0.0)| below the
      pre-registered separation bar. Attenuation then happens just as strongly with a
      content-free seed, so the doc's "needs an intact seed" gloss does not follow from
      it. The ATTENUATION FINDING ITSELF IS UNTOUCHED.
    EVIDENCE AGAINST (DECLARED NULL): if gain scales MONOTONICALLY with content scale
      (strictly ordered across 0.0 / 0.5 / 1.0) AND the endpoints are separated beyond
      the bar, this leg is REFUTED and the architecture doc's "correction needs an
      intact seed" gloss STANDS.

INTERPRETATION GRID (self-routed label; a HYPOTHESIS, not a verdict)
--------------------------------------------------------------------
  readiness UNMET (the seed-corruption range at content_scale=1.0 is too narrow to fit
      a gain slope through, or the gain is UNAVAILABLE)
        -> `substrate_not_ready_requeue`  [NEVER a substrate verdict]
  gain strictly monotonic in content scale AND endpoints separated beyond the bar
        -> `gen_gain_content_dependent_intact_seed_gloss_stands`
           H-gen-gain-content-free REFUTED (its declared null). The doc's gloss stands.
  endpoints NOT separated beyond the bar (gain flat in content scale)
        -> `gen_gain_content_free_intact_seed_gloss_unsupported`
           H-gen-gain-content-free SUPPORTED. The ATTENUATION finding is unaffected;
           only the "needs an intact seed" gloss loses its support.
  endpoints separated but NOT monotonic
        -> `gen_gain_content_scale_nonmonotonic_uninterpretable`
           Explicitly NOT a verdict either way -- a non-monotonic response to a
           monotone manipulation is an instrument flag, not a finding.

ANTI-ALIAS / DESIGN-AUDIT NOTES (GOV-FANOUT-1 step 4)
-----------------------------------------------------
1. THREE CONTENT SCALES, NOT TWO. The intermediate 0.5 point is what separates "gain is
   flat in content" from "gain differs between two arbitrary endpoints for some other
   reason". A two-point contrast cannot distinguish a genuine dose-response from a
   step change, so the monotonicity test needs the middle rung.
2. EFFECT-SIZE BAR SCALED ON THE SD OF THE DELTA, plus an absolute floor. The separation
   bar is max(GAIN_SEPARATION_ABS_FLOOR, GAIN_SEPARATION_SD_MULT * SEM of the per-seed
   delta), so a tiny but consistent difference cannot pass on consistency alone and a
   large but noisy one cannot pass on magnitude alone. A bare threshold on the mean
   would alias "real dose-response" against "noise that happened to order correctly".
3. PER-SEED DELTAS, PAIRED. gain(1.0) and gain(0.0) are computed on the SAME seed and the
   SAME RNG stream, so the delta is paired and the seed-to-seed variance in the transfer
   function itself cancels out of the contrast.
4. THE ATTENUATION FINDING IS RECORDED ALONGSIDE, NOT RE-LITIGATED. Per-content-scale
   gains are all reported, so it stays visible that every arm attenuates (gain < 1)
   whatever this leg concludes about content-dependence. This is the guard against the
   result being read as overturning the attenuation finding, which it cannot do.
5. SCOPE. rem generative readout ONLY.

WHY DIAGNOSTIC (not evidence)
-----------------------------
This discriminates what a readout is a property OF; it tests no substrate hypothesis and
PROMOTES/DEMOTES NOTHING. `experiment_purpose="diagnostic"` excludes it from governance
confidence/conflict scoring. It tags SD-068 as the subject and MECH-168 / INV-047 /
MECH-169 as CONTEXT only. MECH-121 is deliberately NOT tagged: MECH-121 is held
(candidate/substrate_conditional, hold_pending_v3_substrate) and the NREM leg is
substrate-plumbing-fidelity only -- it must not accrue promotion evidence. Resolution of
the pre-registered hypothesis is via /failure-autopsy Step 9b Mode B against the frozen
ledger, NOT by this script's self-route.

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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_sd068_rem_gen_gain_content_scale_diagnostic"
QUEUE_ID = "V3-EXQ-778f"
CLAIM_IDS: List[str] = ["SD-068", "MECH-168", "INV-047", "MECH-169"]
EXPERIMENT_PURPOSE = "diagnostic"
SLEEP_DRIVER_PATTERN = "manual-cycle-loop"
HYPOTHESIS_ID = "H-gen-gain-content-free"
HYPOTHESIS_AXIS = "observation"
HYPOTHESIS_QUESTION = "consolidation_readout_validity"

# The V3-EXQ-778a / 778c 8-seed set, so the content_scale 1.0 and 0.0 arms pool directly
# onto 778c's recorded per-seed generative gains.
SEEDS = [42, 7, 123, 2024, 99, 7777, 314, 1000]
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
WARM_STEPS = 40
# The manipulation: the rollout SEED's injected content, scaled.
CONTENT_SCALES = [0.0, 0.5, 1.0]
ARMS = ["CONTENT_0.0", "CONTENT_0.5", "CONTENT_1.0"]

# Pre-registered thresholds.
# Effect-size bar: scaled on the SD of the per-seed DELTA plus an absolute floor, so a
# consistent-but-tiny difference cannot pass on consistency alone.
GAIN_SEPARATION_ABS_FLOOR = 0.05
GAIN_SEPARATION_SD_MULT = 2.0
# Readiness: the gain is a SLOPE fitted over the sigma grid, so the readiness check is a
# RANGE check on that slope's x-axis (the injected seed's relative corruption) -- the
# same statistic the gain routes on, measured on the content_scale=1.0 positive control.
INPUT_CORRUPTION_RANGE_FLOOR = 0.05


def _fmt(v: float) -> str:
    """ASCII-safe float rendering that keeps UNAVAILABLE legible."""
    if v == H.UNAVAILABLE or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.4f}"


def _finite(vals: List[float]) -> List[float]:
    return [
        v
        for v in vals
        if v != H.UNAVAILABLE and not (isinstance(v, float) and math.isnan(v))
    ]


def _mean_sd_sem(vals: List[float]) -> Tuple[float, float, float]:
    v = _finite(vals)
    if not v:
        return H.UNAVAILABLE, H.UNAVAILABLE, H.UNAVAILABLE
    mean = sum(v) / len(v)
    if len(v) < 2:
        return mean, H.UNAVAILABLE, H.UNAVAILABLE
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
    return mean, sd, sd / math.sqrt(len(v))


def _sweep_content_scale(
    *, seed: int, content_scale: float, sigmas: List[float], warm: int
) -> Dict[str, Any]:
    """Generative-gain readout across the sigma grid at one rollout-seed content scale."""
    pr: Dict[float, Dict[str, Dict[str, float]]] = {}
    for s in sigmas:
        pr[s] = H.rem_only_integrity_at_sigma(
            seed=seed,
            sigma=s,
            warm_steps=warm,
            content_scale=content_scale,
            run_generative=True,
        )

    gains = H.error_propagation_gain(
        seed=seed,
        sigmas=list(sigmas),
        warm_steps=warm,
        pr_by_sigma={
            s: {
                # error_propagation_gain reads all three phases for the staging block;
                # only its rem generative terms are used here, so the two unswept phases
                # are supplied as explicitly-unavailable rather than fabricated.
                "sws": {"signal_power": 0.0, "noise_power": float("nan")},
                "nrem": {"transfer_fidelity": H.UNAVAILABLE},
                "rem": pr[s]["rem"],
            }
            for s in sigmas
        },
    )

    in_cor = _finite(
        [pr[s]["rem"].get("rem_gen_input_rel_corruption", H.UNAVAILABLE) for s in sigmas]
    )
    corruption_range = (max(in_cor) - min(in_cor)) if len(in_cor) >= 2 else 0.0

    return {
        "content_scale": float(content_scale),
        "rem_generative_gain": gains.get("rem_generative_gain", H.UNAVAILABLE),
        "rem_generative_gain_mean": gains.get("rem_generative_gain_mean", H.UNAVAILABLE),
        "rem_generative_output_slope": gains.get(
            "rem_generative_output_slope", H.UNAVAILABLE
        ),
        "rem_generative_attenuates": gains.get("rem_generative_attenuates", 0.0),
        "input_corruption_range": float(corruption_range),
        "per_sigma_gain": [
            pr[s]["rem"].get("rem_generative_gain", H.UNAVAILABLE) for s in sigmas
        ],
        "per_sigma_output_rel_dev": [
            pr[s]["rem"].get("rem_gen_output_rel_dev", H.UNAVAILABLE) for s in sigmas
        ],
        "per_sigma_input_rel_corruption": [
            pr[s]["rem"].get("rem_gen_input_rel_corruption", H.UNAVAILABLE)
            for s in sigmas
        ],
        # Every arm's internals recorded as richly as every other arm's.
        "integrity": {str(s): pr[s]["rem"] for s in sigmas},
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    warm = 8 if dry_run else WARM_STEPS
    sigmas = [0.0, 0.5, 2.0] if dry_run else SIGMAS
    scales = CONTENT_SCALES

    print(
        "V3-EXQ-778f: SD-068 REM generative gain vs content scale "
        "(H-gen-gain-content-free)",
        flush=True,
    )
    print(
        f"  seeds={seeds} sigmas={sigmas} content_scales={scales} warm_steps={warm} "
        f"arms={ARMS} dry_run={dry_run}",
        flush=True,
    )
    print(
        "  NOTE: this leg does NOT test the attenuation finding (gain 0.149, 8/8), "
        "only the 'correction needs an intact seed' gloss placed on it.",
        flush=True,
    )

    config_slice = {
        "sigmas": sigmas,
        "content_scales": list(scales),
        "warm_steps": warm,
        "arms": list(ARMS),
        "gain_separation_abs_floor": GAIN_SEPARATION_ABS_FLOOR,
        "gain_separation_sd_mult": GAIN_SEPARATION_SD_MULT,
        "input_corruption_range_floor": INPUT_CORRUPTION_RANGE_FLOOR,
        "shy_decay_rate": 0.85,
        "body_obs_dim": H.BODY_OBS_DIM,
        "world_obs_dim": H.WORLD_OBS_DIM,
        "action_dim": H.ACTION_DIM,
        "harm_obs_dim": H.HARM_OBS_DIM,
    }

    arm_results: List[Dict[str, Any]] = []
    total_eps = len(scales)

    for seed in seeds:
        print(f"Seed {seed} Condition REM_GEN_GAIN_CONTENT_SCALE", flush=True)
        with arm_cell(
            seed,
            config_slice=config_slice,
            script_path=Path(__file__),
            config_slice_declared=True,
        ) as cell_ctx:
            by_scale: Dict[str, Any] = {}
            for i, cs in enumerate(scales):
                a = _sweep_content_scale(
                    seed=seed, content_scale=cs, sigmas=sigmas, warm=warm
                )
                by_scale[str(cs)] = a
                print(
                    f"  [train] gen_gain seed={seed} ep {i + 1}/{total_eps} "
                    f"content_scale={cs} gain={_fmt(a['rem_generative_gain'])} "
                    f"attenuates={bool(a['rem_generative_attenuates'])} "
                    f"corruption_range={a['input_corruption_range']:.4f}",
                    flush=True,
                )

            g0 = by_scale[str(scales[0])]["rem_generative_gain"]
            gmid = by_scale[str(scales[1])]["rem_generative_gain"]
            g1 = by_scale[str(scales[-1])]["rem_generative_gain"]
            have_all = all(
                g != H.UNAVAILABLE and not math.isnan(g) for g in (g0, gmid, g1)
            )
            delta = (g1 - g0) if have_all else H.UNAVAILABLE
            monotonic = bool(have_all and ((g0 < gmid < g1) or (g0 > gmid > g1)))
            # Readiness is asserted on the POSITIVE CONTROL arm (content_scale=1.0):
            # the gain is a slope, so its x-range must be non-degenerate.
            ctrl_range = by_scale[str(scales[-1])]["input_corruption_range"]
            readiness = bool(
                have_all and ctrl_range >= INPUT_CORRUPTION_RANGE_FLOOR
            )
            attenuates_all = all(
                bool(by_scale[str(cs)]["rem_generative_attenuates"]) for cs in scales
            )

            row: Dict[str, Any] = {
                "seed": seed,
                "arm": "REM_GEN_GAIN_CONTENT_SCALE",
                "arms_compared": list(ARMS),
                "hypothesis_id": HYPOTHESIS_ID,
                "sigmas": list(sigmas),
                "content_scales": list(scales),
                "by_content_scale": by_scale,
                "gain_at_0": g0,
                "gain_at_mid": gmid,
                "gain_at_1": g1,
                "gain_delta_1_minus_0": delta,
                "monotonic_in_content_scale": monotonic,
                "attenuates_at_all_scales": attenuates_all,
                "control_input_corruption_range": float(ctrl_range),
                "readiness_met": readiness,
            }
            cell_ctx.stamp(row)

        arm_results.append(row)
        print(
            f"  gain: cs0={_fmt(g0)} cs_mid={_fmt(gmid)} cs1={_fmt(g1)} "
            f"delta={_fmt(delta)} monotonic={monotonic} "
            f"attenuates_all_scales={attenuates_all}",
            flush=True,
        )
        print(f"verdict: {'PASS' if monotonic else 'FAIL'}", flush=True)

    n = len(arm_results)
    readiness_ok = all(r["readiness_met"] for r in arm_results)
    deltas = [r["gain_delta_1_minus_0"] for r in arm_results]
    d_mean, d_sd, d_sem = _mean_sd_sem(deltas)
    n_monotonic = sum(1 for r in arm_results if r["monotonic_in_content_scale"])
    monotonic_majority = n_monotonic > n / 2

    # Effect-size bar: SD-of-the-delta scaled, plus an absolute floor.
    if d_sem != H.UNAVAILABLE and not math.isnan(d_sem):
        bar = max(GAIN_SEPARATION_ABS_FLOOR, GAIN_SEPARATION_SD_MULT * d_sem)
    else:
        bar = GAIN_SEPARATION_ABS_FLOOR
    separated = bool(
        d_mean != H.UNAVAILABLE and not math.isnan(d_mean) and abs(d_mean) >= bar
    )

    scale_summary: Dict[str, Any] = {}
    for cs in CONTENT_SCALES:
        k = str(cs)
        gains = [r["by_content_scale"][k]["rem_generative_gain"] for r in arm_results]
        m, sd, sem = _mean_sd_sem(gains)
        scale_summary[k] = {
            "mean_rem_generative_gain": m,
            "sd_rem_generative_gain": sd,
            "sem_rem_generative_gain": sem,
            "per_seed_rem_generative_gain": gains,
            "n_seeds_attenuating": sum(
                1
                for r in arm_results
                if bool(r["by_content_scale"][k]["rem_generative_attenuates"])
            ),
            "per_seed_input_corruption_range": [
                r["by_content_scale"][k]["input_corruption_range"] for r in arm_results
            ],
        }

    min_ctrl_range = min(
        (r["control_input_corruption_range"] for r in arm_results), default=0.0
    )

    # Self-routed label. Readiness routes ONLY to requeue.
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
    elif separated and monotonic_majority:
        label = "gen_gain_content_dependent_intact_seed_gloss_stands"
    elif not separated:
        label = "gen_gain_content_free_intact_seed_gloss_unsupported"
    else:
        label = "gen_gain_content_scale_nonmonotonic_uninterpretable"

    # PASS == this leg's hypothesis SUPPORTED (gain is content-free).
    overall_pass = bool(readiness_ok and not separated)

    interpretation = {
        "label": label,
        "hypothesis_id": HYPOTHESIS_ID,
        "hypothesis_axis": HYPOTHESIS_AXIS,
        "hypothesis_question": HYPOTHESIS_QUESTION,
        "declared_null": (
            "if gain scales MONOTONICALLY with content scale (strictly ordered across "
            "0.0 / 0.5 / 1.0) AND the endpoints separate beyond the pre-registered bar, "
            "H-gen-gain-content-free is REFUTED and the architecture doc's 'correction "
            "needs an intact seed' gloss STANDS."
        ),
        "scope_guard": (
            "This leg CANNOT overturn the attenuation finding (rem_generative_gain "
            "0.149, attenuating 8/8 seeds) and does not attempt to. Whatever it "
            "concludes, the transfer function still attenuates -- per-content-scale "
            "gains and per-scale attenuation counts are recorded alongside precisely so "
            "that stays visible. Only the interpretive gloss that the correction needs "
            "an INTACT SEED is under test."
        ),
        "preconditions": [
            {
                "name": "control_input_corruption_range_supra_floor",
                "description": (
                    "rem_generative_gain is a SLOPE of output relative deviation "
                    "against input relative seed corruption across the sigma grid, so "
                    "the readiness check is a RANGE check on that slope's x-axis -- the "
                    "SAME statistic the gain routes on -- measured on the "
                    "content_scale=1.0 positive control. A degenerate x-range means the "
                    "gain is fitted through a point cloud with no spread, so a "
                    "content-scale contrast between such gains is uninterpretable. "
                    "Range-gated criterion -> range readiness (NOT a magnitude proxy)."
                ),
                "measured": float(min_ctrl_range),
                "threshold": INPUT_CORRUPTION_RANGE_FLOOR,
                "direction": "lower",
                "control": "content_scale=1.0 arm across the full sigma grid (fully-injected positive control)",
                "met": bool(readiness_ok),
            },
        ],
        "criteria_non_degenerate": {
            "C1_gain_flat_in_content_scale": bool(readiness_ok),
            # Monotonicity is vacuous without three distinct content scales.
            "C2_monotonic_in_content_scale": bool(len(CONTENT_SCALES) >= 3),
            "C3_attenuation_recorded_at_all_scales": bool(
                all(
                    scale_summary[str(cs)]["mean_rem_generative_gain"] != H.UNAVAILABLE
                    for cs in CONTENT_SCALES
                )
            ),
        },
        "criteria": [
            {
                "name": "C1_gain_flat_in_content_scale",
                "load_bearing": True,
                "passed": bool(not separated),
            },
            {
                "name": "C2_monotonic_in_content_scale",
                "load_bearing": False,
                "passed": bool(monotonic_majority),
            },
            {
                "name": "C3_attenuation_recorded_at_all_scales",
                "load_bearing": False,
                "passed": bool(
                    all(
                        scale_summary[str(cs)]["mean_rem_generative_gain"]
                        != H.UNAVAILABLE
                        for cs in CONTENT_SCALES
                    )
                ),
            },
        ],
        "separation": {
            "mean_gain_delta_1_minus_0": d_mean,
            "sd_gain_delta": d_sd,
            "sem_gain_delta": d_sem,
            "separation_bar": float(bar),
            "bar_rule": (
                f"max({GAIN_SEPARATION_ABS_FLOOR} absolute floor, "
                f"{GAIN_SEPARATION_SD_MULT} x SEM of the per-seed paired delta) -- so a "
                "consistent-but-tiny difference cannot pass on consistency alone, nor a "
                "large-but-noisy one on magnitude alone"
            ),
            "separated": separated,
            "n_seeds_monotonic": n_monotonic,
            "per_seed_gain_delta": deltas,
        },
        "scale_summary": scale_summary,
        "portfolio": {
            "gov_rule": "GOV-FANOUT-1",
            "question": HYPOTHESIS_QUESTION,
            "this_leg": f"{HYPOTHESIS_ID} (axis={HYPOTHESIS_AXIS})",
            "sibling_legs": [
                "V3-EXQ-778d H-rem-clamp-artifact (axis=measurement)",
                "V3-EXQ-778e H-rem-genuinely-content-free (axis=representation)",
            ],
            "note": (
                "Read the three legs JOINTLY. This leg is on a DIFFERENT readout (the "
                "generative rollout) from its two siblings (the precision calibration), "
                "so it is independent of them by construction. Resolution via "
                "/failure-autopsy Step 9b Mode B against the frozen ledger, not by this "
                "self-route."
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
        f"monotonic seeds: {n_monotonic}/{n} | mean delta(gain@1.0 - gain@0.0)="
        f"{_fmt(d_mean)} sd={_fmt(d_sd)} bar={bar:.4f} separated={separated}",
        flush=True,
    )
    for cs in CONTENT_SCALES:
        ss = scale_summary[str(cs)]
        print(
            f"  content_scale={cs}: mean_gain={_fmt(ss['mean_rem_generative_gain'])} "
            f"sd={_fmt(ss['sd_rem_generative_gain'])} "
            f"attenuating={ss['n_seeds_attenuating']}/{n}",
            flush=True,
        )
    print(f"self-route label: {label}", flush=True)

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": per_claim,
        "interpretation": interpretation,
        "arm_results": arm_results,
        "scale_summary": scale_summary,
        "config": config_slice,
        "seeds": seeds,
        "non_degenerate": bool(readiness_ok),
        "degeneracy_reason": (
            ""
            if readiness_ok
            else (
                "the content_scale=1.0 control's input-corruption range fell below the "
                "floor, so rem_generative_gain is a slope fitted through a degenerate "
                "x-range and the content-scale contrast is uninterpretable"
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
        "scale_summary": result["scale_summary"],
        "hypothesis_id": HYPOTHESIS_ID,
        "hypothesis_axis": HYPOTHESIS_AXIS,
        "hypothesis_question": HYPOTHESIS_QUESTION,
        "acceptance_criteria": {
            "C1_gain_flat_in_content_scale": (
                "|mean(gain@1.0 - gain@0.0)| BELOW the separation bar "
                f"max({GAIN_SEPARATION_ABS_FLOOR}, "
                f"{GAIN_SEPARATION_SD_MULT} x SEM of the per-seed paired delta) "
                "(LOAD-BEARING). Above the bar WITH monotonicity REFUTES this leg -- its "
                "declared null."
            ),
            "C2_monotonic_in_content_scale": (
                "gain strictly ordered across content_scale 0.0 / 0.5 / 1.0 on a "
                "majority of seeds (the middle rung is what separates a genuine "
                "dose-response from a two-point step change)"
            ),
            "C3_attenuation_recorded_at_all_scales": (
                "per-content-scale gains and attenuation counts recorded, so the "
                "standing attenuation finding (0.149, 8/8) stays visible and is not "
                "read as overturned by this leg"
            ),
        },
        "arm_results": result["arm_results"],
        "notes": (
            "SD-068 GOV-FANOUT-1 discrimination portfolio, leg 3 of 3 (axis=OBSERVATION, "
            "hypothesis H-gen-gain-content-free, pre-registered 2026-07-18T08:41:15Z in "
            "hypothesis_space_registry.v1.json question consolidation_readout_validity). "
            "Routed by failure_autopsy_V3-EXQ-778c_2026-07-18.json "
            "targets[0].fanout_recommendation. SCOPE GUARD: this leg does NOT threaten "
            "the finding that the REM generative pass ATTENUATES (rem_generative_gain "
            "0.149, 8/8 seeds), which stands; it tests only the narrower interpretive "
            "gloss that 'correction needs an intact seed' "
            "(docs/architecture/sd_068_consolidation_lesion_harness.md, the OPEN "
            "QUESTION box in the REM generative gain section). Motivated by the 778c "
            "manifest observation -- OUTSIDE its scored C1 criteria and recorded as an "
            "open question, not a verdict -- that on seed 42 the NULL arm's "
            "rem_generative_gain (0.182/0.184/0.188/0.209 across sigma) is close to the "
            "INJECTED arm's (0.165/0.166/0.172/0.190) at rem_gen_content_scale 0.0. "
            "Contrasts gain at rollout-seed content_scale 0.0 / 0.5 / 1.0 across the 8 "
            "778a/778c seeds; the middle rung separates a genuine dose-response from a "
            "two-point step change, deltas are PAIRED within seed so transfer-function "
            "variance cancels, and the effect-size bar is scaled on the SD of the delta "
            "plus an absolute floor. rem generative readout ONLY. DIAGNOSTIC: excluded "
            "from governance confidence/conflict scoring; PROMOTES/DEMOTES NOTHING. "
            "MECH-121 deliberately NOT tagged (held; NREM leg is "
            "substrate-plumbing-fidelity only). Siblings V3-EXQ-778d (measurement) and "
            "V3-EXQ-778e (representation) run in PARALLEL and are read jointly; this leg "
            "is on a DIFFERENT readout from both, so it is independent by construction. "
            "Experiment-layer only; zero ree_core change. GOV-REUSE-1: the decisive "
            "readout is rem_generative_gain at an INTERMEDIATE content scale (0.5), "
            "which no recorded manifest carries -- 778c swept only 1.0 and 0.0, and its "
            "0.0 arm is a single unpowered seed-42 observation -- so the monotonicity "
            "test is not recoverable by reanalysis -> run."
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
