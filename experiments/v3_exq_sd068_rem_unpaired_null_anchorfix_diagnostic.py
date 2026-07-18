"""
V3-EXQ-778h: SD-068 REM unpaired-target null -- SAME-QUESTION re-run of V3-EXQ-778d
with the readiness-anchor specification defect corrected.
Hypothesis under test: H-rem-clamp-artifact (axis: MEASUREMENT).
SLEEP DRIVER: manual-cycle-loop (the SD-068 harness drives recalibrate_precision_to +
enter_rem_mode / run_rem_attribution_pass directly per phase readout; no
SleepLoopManager scheduling).

WHAT CHANGED FROM V3-EXQ-778d -- AND WHAT DID NOT
--------------------------------------------------
THE SCIENCE IS UNCHANGED AND IS NOT UNDER RE-ADJUDICATION. Same question, same arms,
same seeds, same sigma grid, same RNG streams, same thresholds, same criteria. The
unpaired-target null (ARM_NULL_UNPAIRED vs the ARM_NULL_ZERO anchor) is CORRECT and is
retained verbatim as the faithful Bar et al. 2020 "same odour, no prior pairing" analog.
The retired step-ladder probe is NOT revisited -- it was proven algebraically inert
(step cancels in the reported ratio; verified identical to 12 significant figures over
step in {0.1, 0.25, 0.5, 1.0}) and that retirement is recorded at
hypothesis_space_registry.v1.json -> consolidation_readout_validity ->
H-rem-clamp-artifact -> probe_amendments[0].

This is LEDGER HYGIENE: it makes the control provenance clean-AS-SHIPPED rather than
clean-as-rehabilitated. Exactly two things differ.

CHANGE 1 -- the `railed` predicate now counts BOTH rails.
V3-EXQ-778d's readiness anchor `null_zero_anchor_reproduces_778c_railed_signature` was
UNMEETABLE BY CONSTRUCTION. Its predicate was

    derailed = (clamp_frac <= 0.2 and n_distinct >= 3 and n_unclamped >= 3)
    railed   = not derailed

but V3-EXQ-778c's degeneracy sat at TWO rails, not one:

  (a) SATURATION -- `calibration_error` pinned at the constant 998.5009992509989 with
      `target_clamped` 1.0 on every sigma point, an identically ZERO slope. 5/8 seeds
      (7, 2024, 7777, 314, 1000).
  (b) POSITIVITY-FLOOR COLLAPSE -- the null precision reference collapsed onto the
      1e-3 floor at the low end of the grid, so the 1/1e-3 term dominates the fit and
      the ratio goes off-scale: 4348.47 / 9142.77 / 1801.65. 3/8 seeds (42, 123, 99).

The predicate detected only rail (a). The rail-(b) seeds have clamp_frac 0.2,
n_distinct 5, n_unclamped 4 -- so they SATISFY `derailed` and were scored HEALTHY. The
maximum anchor fraction achievable by a BIT-PERFECT replication of 778c was therefore
exactly 5/8 = 0.625, against a gate of ANCHOR_MIN_RAILED_SEEDS_FRAC = 0.75. It could
never report met, no matter how faithfully the anchor reproduced -- and it did
reproduce, bit-identically, across three independent runs (778c, 778d ARM_NULL_ZERO,
778e's within-run legacy anchor all give the same floats).

That mattered because `criteria_non_degenerate.C1_unpaired_null_derails` was
`bool(readiness_ok and anchor_ok)` with readiness_ok true on 8/8, so the entire
degeneracy flag on the LOAD-BEARING criterion traced to this one mis-specified
statistic -- and the self-route mislabelled the cause `substrate_not_ready_requeue`
when nothing about the substrate was unready. The correct classification is
`instrument_specification_gap`. Diagnosis:
`REE_assembly/evidence/planning/failure_autopsy_SD-068-rem-fanout-cluster_2026-07-18.md`
section 2.

`_railed()` below now scores a seed railed if it hit EITHER rail, with each rail tested
separately from the RECORDED INTEGRITY FIELDS (no hardcoded seed indices) and the rail
kind recorded per seed. Under it, the anchor is expected to clear 8/8.

CHANGE 2 -- a reachability guard, so this class of defect fails at setup.
`experiments/_lib/readiness_anchor.assert_anchor_reachable` replays the frozen,
known-degenerate 778c/778d reference through THE SHIPPED PREDICATE at setup and raises
if the gate exceeds what the reference itself can score. A precondition a perfect
replication cannot pass is a guaranteed false negative; this is the check that would
have caught the defect at design-audit time. It is factored into `_lib/` so other
readiness anchors can reuse it (Learning 1 of the autopsy).

EXPECTED RESULT: the anchor clears (8/8 railed), C1's non_degenerate flag becomes true,
and the leg's existing verdict reproduces clean-as-shipped -- C1_unpaired_null_derails
still PASSing at 7/8 seeds de-railed.

LEDGER NOTE: `H-rem-clamp-artifact` is ALREADY resolved `confirmed` in the frozen ledger
with `control_rehabilitated: true`, on this corrected both-rails predicate. This re-run
does NOT change that resolution. If it lands, a governance session may clear the
`control_rehabilitated` flag. This script does NOT and MUST NOT edit
`hypothesis_space_registry.v1.json` -- GOV-FROZEN-1 single-producer rule: only
/failure-autopsy Step 9b writes it.

WHAT THIS MEASURES
------------------
V3-EXQ-778c ran the SD-068 zero-injected-content null control at n=8 and left the REM
leg DEGENERATE AT BOTH RAILS rather than resolved. Per-seed `null_slope_ratio_rem` came
back exactly 0.0 on 5/8 seeds -- the null arm's `calibration_error` pinned at the
constant 998.5009992509989 with `target_clamped` 1.0, an identically ZERO slope -- and
off-scale 1801-9143 on the other 3/8, where the null arm's precision reference collapsed
onto the 1e-3 positivity floor so 1/1e-3 dominates. mean 1911.6, sd 3306.1, 95% CI
[-379, 4203], `ceiling_inside_ci95` true, `confound_verdict_stable` false.

The 5 apparently-clean seeds are clean only BY DEGENERACY. A zero slope produced by a
saturated constant is NOT evidence of content-contingency -- it is the absence of a
measurement. This script is leg 1 of the GOV-FANOUT-1 discrimination portfolio routed by
`REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-778c_2026-07-18.json`
(`targets[0].fanout_recommendation`), which enumerates three live hypotheses on three
DIFFERENT design axes, each with a declared null. This leg attacks the MEASUREMENT axis;
its siblings are V3-EXQ-778e (representation) and V3-EXQ-778f (observation). They run in
PARALLEL, not in sequence -- a single re-posed probe can inherit the prior confound and
return a confident-but-wrong verdict, which is what GOV-FANOUT-1 exists to prevent.

PROBE AMENDED BEFORE QUEUING -- READ THIS BEFORE COMPARING TO THE LEDGER
------------------------------------------------------------------------
The autopsy's sketched probe for this leg was a `step` LADDER (0.1/0.25/0.5/1.0), on the
reading that full adoption drives `calibration_error` into its rails. That probe is
PROVABLY INERT and was replaced during the pre-queue design audit, before any compute was
spent. The substrate computes (ree_core/predictors/e3_selector.py
recalibrate_precision_to):

    rv_after = (1 - step) * rv_before + step * (1 / (corrupt_target + 1e-6))

so (1) the `max(1e-3, raw_target)` positivity clamp is applied to corrupt_target UPSTREAM
of `step`, making the clamp fraction exactly invariant to step; and (2) both arms'
sigma-slopes carry the same linear factor `step`, so it CANCELS in the reported ratio
|null slope| / |injected slope|. Verified numerically over step in {0.1, 0.25, 0.5, 1.0}:
the ratio is identical to 12 significant figures (14569.1833719598-...602, float
round-off only) and the clamp fraction and distinct-value count are bit-identical. A step
ladder would therefore return this leg's DECLARED NULL by construction -- writing a FALSE
ELIMINATION into the frozen ledger rather than making a measurement. (A positivity-floor
ladder fails for the same reason: the null arm's reference is centred on ZERO, so it
clamps on ~half the draws at ANY floor value.)

The clamp's actual root is that MULTIPLICATIVE content scaling
(`injected_target = clean_target * content_scale`) sends the null arm's target precision
to ZERO -- a degenerate point of a strictly-positive parameterisation. So the
measurement-axis knob that genuinely bites is HOW THE NULL IS OPERATIONALISED, which is
what this leg now sweeps. The hypothesis it adjudicates (the degeneracy is an artifact of
the measurement rather than a property of the readout) is UNCHANGED; only the probe is.
The ledger entry for H-rem-clamp-artifact needs its `probe` and `declared_null` amended
to match -- NOT done by this session (see LEDGER note at the foot of this docstring).

HYPOTHESES UNDER TEST
---------------------
H-rem-clamp-artifact (THIS LEG, axis=measurement, pre-registered
  2026-07-18T08:41:15Z in hypothesis_space_registry.v1.json question
  `consolidation_readout_validity`):
    CLAIM: the both-rails degeneracy is an artifact of the MEASUREMENT -- specifically of
    operationalising "no content" as a target precision of ZERO, which is off a cliff of
    the parameterisation -- rather than a property of the rem readout itself.
    PROBE: contrast two nulls that remove the same content pairing but sit at different
    points of the parameterisation:
      ARM_NULL_ZERO     -- target = clean * 0.0 (the 778c null; REPLICATION ANCHOR)
      ARM_NULL_UNPAIRED -- target = an INDEPENDENT positive draw of the same magnitude
                           class, UNPAIRED with the clean target the error is scored
                           against. This is the faithful analog of the Bar et al. 2020
                           odour control SD-068 follows: "same odour, NO PRIOR PAIRING",
                           not "no odour". The delivered perturbation is unchanged
                           (jitter is always referenced to the UNSCALED clean target);
                           only the content PAIRING is removed.
    EVIDENCE FOR: ARM_NULL_UNPAIRED DE-RAILS -- clamp fraction on the scored grid at or
      below DERAIL_CLAMP_CEILING, at least MIN_UNCLAMPED_SIGMAS unclamped sigma points,
      and a non-constant error series -- while ARM_NULL_ZERO reproduces 778c's rails.
      The degeneracy is then an artifact of the zero-target operationalisation.
    EVIDENCE AGAINST (DECLARED NULL): if ARM_NULL_UNPAIRED stays railed too -- still
      clamping, still constant -- this leg is REFUTED. The degeneracy is then NOT an
      artifact of how the null was operationalised, and the measurement axis is
      eliminated from the question's live set.

INTERPRETATION GRID (self-routed label; a HYPOTHESIS, not a verdict)
--------------------------------------------------------------------
  readiness UNMET (injected-arm slope below floor, or ARM_NULL_ZERO fails to reproduce
      778c's railed signature)
        -> `substrate_not_ready_requeue`  [NEVER a substrate verdict]
  unpaired null DE-RAILS and its ratio is CONTENT-CONTINGENT (<= ceiling)
        -> `rem_clamp_artifact_confirmed_content_contingent_when_declamped`
           H-rem-clamp-artifact SUPPORTED; additionally bears on the parent
           H-rem-content-contingent (the readout DOES track content once measured at a
           non-degenerate point).
  unpaired null DE-RAILS but its ratio is CONFOUNDED (> ceiling)
        -> `rem_clamp_artifact_confirmed_but_readout_still_content_free`
           H-rem-clamp-artifact SUPPORTED (the zero-target null WAS an artifact) while
           the parent H-rem-content-contingent is WEAKENED. These are SEPARABLE outcomes
           and the grid keeps them separate -- collapsing them is precisely the
           verdict-aliasing this portfolio was audited against.
  unpaired null STAYS RAILED
        -> `rem_clamp_artifact_refuted_unpaired_null_also_railed`
           H-rem-clamp-artifact REFUTED (its declared null).

ANTI-ALIAS / DESIGN-AUDIT NOTES (GOV-FANOUT-1 step 4)
-----------------------------------------------------
1. REPLICATION ANCHOR. ARM_NULL_ZERO IS the V3-EXQ-778c rem condition, on the SAME 8
   seeds and the SAME RNG stream (`rem_only_integrity_at_sigma` reproduces
   `phase_integrity_at_sigma`'s `_gen(seed * 1009 + 3)` cell exactly). It must reproduce
   778c's railed signature. If it does NOT, something other than the null-mode differs
   between the runs and the contrast is uninterpretable -- so a failed anchor routes to
   `substrate_not_ready_requeue`, never to a verdict about the clamp. Without it, a
   de-railing in the unpaired arm would alias "the zero-target null was the artifact"
   against "this run simply differs from 778c".
2. RATIO REPORTED, NOT JUST THE CLAMP FRACTION. De-railing alone does not establish
   content-contingency: a de-railed null arm can still track sigma perfectly. Reporting
   the ratio keeps "the null was mis-operationalised" separable from "the readout tracks
   content" (the two middle grid branches above).
3. SCORED ON THE UNCLAMPED SUBGRID. At large sigma the jitter (sd = sigma * clean_target)
   can push even a positive unpaired reference negative, re-clamping it. The ratio is
   therefore fitted on the UNCLAMPED sigma points, with a readiness gate demanding at
   least MIN_UNCLAMPED_SIGMAS of them -- so a slope is never fitted through saturated
   points, and a leg that cannot get enough clean points says so instead of scoring.
4. SCOPE. rem phase ONLY. The sws repair is a single unambiguous build (routed to
   /implement-substrate, exempt from GOV-FANOUT-1) and the nrem leg is already confirmed
   content-contingent by 778c, so sweeping them would be compute answering nothing.

WHY DIAGNOSTIC (not evidence)
-----------------------------
This discriminates WHY an instrument reads degenerately; it tests no substrate hypothesis
and PROMOTES/DEMOTES NOTHING. `experiment_purpose="diagnostic"` excludes it from
governance confidence/conflict scoring. It tags SD-068 as the subject and
MECH-168 / INV-047 / MECH-169 as CONTEXT only. MECH-121 is deliberately NOT tagged:
MECH-121 is held (candidate/substrate_conditional, hold_pending_v3_substrate) and the
NREM leg is substrate-plumbing-fidelity only -- it must not accrue promotion evidence.
Resolution of the pre-registered hypothesis is via /failure-autopsy Step 9b Mode B
against the frozen ledger, NOT by this script's self-route.

LEDGER: the H-rem-clamp-artifact entry in hypothesis_space_registry.v1.json still records
the retired `step`-ladder probe and its declared null. It needs amending to the probe
above, citing the step-invariance proof. This session did NOT write it -- an active
TASK_CLAIMS entry (`axis-family convergence discriminator`, claimed 2026-07-18T08:05:00Z)
holds that file, and silently overwriting another session's claim is forbidden.

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

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib import consolidation_lesion_harness as H  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_sd068_rem_unpaired_null_anchorfix_diagnostic"
QUEUE_ID = "V3-EXQ-778h"
SUPERSEDES = "V3-EXQ-778d"
SUPERSEDES_RUN_ID = "v3_exq_sd068_rem_unpaired_null_diagnostic_20260718T124216Z_v3"
CLAIM_IDS: List[str] = ["SD-068", "MECH-168", "INV-047", "MECH-169"]
EXPERIMENT_PURPOSE = "diagnostic"
SLEEP_DRIVER_PATTERN = "manual-cycle-loop"
HYPOTHESIS_ID = "H-rem-clamp-artifact"
HYPOTHESIS_AXIS = "measurement"
HYPOTHESIS_QUESTION = "consolidation_readout_validity"

# The V3-EXQ-778a / 778c 8-seed set, reused EXACTLY so ARM_NULL_ZERO is a direct
# replication anchor against 778c's recorded per-seed rem degeneracy.
SEEDS = [42, 7, 123, 2024, 99, 7777, 314, 1000]
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
WARM_STEPS = 40
ARMS = ["INJECTED", "NULL_ZERO", "NULL_UNPAIRED"]

# The clean target precision the harness injects and scores against.
CLEAN_TARGET_PRECISION = 2.0
# The unpaired null's target is an INDEPENDENT positive draw of the same magnitude
# class: clean * exp(N(0, UNPAIRED_LOG_SD)). Log-normal so it is positive BY
# CONSTRUCTION (never needing the clamp that is under investigation), same order of
# magnitude, and drawn on a generator seeded independently of the scored target so it
# carries no information about it.
UNPAIRED_LOG_SD = 0.35
UNPAIRED_SEED_OFFSET = 90001

# Pre-registered thresholds.
NULL_SLOPE_RATIO_CEILING = H.NULL_SLOPE_RATIO_CEILING  # 0.25
INJECTED_SLOPE_FLOOR = 1e-6      # readiness: min |injected slope| for an interpretable ratio
DERAIL_CLAMP_CEILING = 0.2       # "de-railed" needs the reference off the floor on
                                 # at least 80% of the sigma grid
DERAIL_MIN_DISTINCT = 3          # ...AND a non-constant error series (of len(SIGMAS)=5)
MIN_UNCLAMPED_SIGMAS = 3         # ...AND enough clean points to fit a slope through
# Anchor: ARM_NULL_ZERO must reproduce 778c's railed signature on most seeds.
ANCHOR_MIN_RAILED_SEEDS_FRAC = 0.75

# --- CHANGE 1 constants: the SECOND rail (positivity-floor collapse) -----------------
# The strictly-positive parameterisation's floor, applied by
# ree_core/predictors/e3_selector.py recalibrate_precision_to as max(1e-3, raw_target).
POSITIVITY_FLOOR = 1e-3
# Tolerance for "sitting ON the floor" -- the recorded value is the clamp output
# exactly, so this only absorbs float round-trip through JSON.
POSITIVITY_FLOOR_TOL = 1e-9
# A slope fitted through a reference that touched the floor is dominated by the
# 1/1e-3 term and goes off-scale. Pre-registered at 100.0: 400x the
# content-contingency ceiling (0.25) that a healthy null must sit under, and 18x
# BELOW the smallest floor-collapse ratio V3-EXQ-778c actually produced (1801.65).
# Both rails are therefore separated from the healthy regime by orders of magnitude,
# not by a tuned cut.
FLOOR_COLLAPSE_RATIO_MIN = 100.0


def _railed(cell: Dict[str, Any]) -> bool:
    """BOTH-RAILS predicate: did this null arm hit EITHER degenerate rail?

    THE SHIPPED PREDICATE. The live cells and the frozen reachability-guard reference
    are both scored through this one callable -- scoring the guard with a re-implementation
    would defeat its purpose, since the defect being guarded against IS a mis-specified
    predicate.

    Rail (a) SATURATION: the reference is clamped across (nearly) the whole grid, so the
      error series is constant and the slope is identically zero. This is the ONLY rail
      V3-EXQ-778d tested -- it is `not derailed` under the original predicate.
    Rail (b) POSITIVITY-FLOOR COLLAPSE: the reference touched the 1e-3 floor somewhere on
      the grid, and the resulting fit is off-scale. These seeds look healthy on
      clamp_frac / n_distinct / n_unclamped (0.2 / 5 / 4) yet carry ratios of 1801-9143.

    Derived from the RECORDED INTEGRITY FIELDS, never from seed indices, so it travels
    to any seed set.
    """
    clamp_frac = float(cell["null_target_clamped_frac"])
    n_distinct = float(cell["null_series_n_distinct"])
    n_unclamped = int(cell["n_unclamped_sigmas"])
    ratio_full = float(cell["null_slope_ratio_full"])
    min_ref = float(cell["min_corrupt_target_precision"])

    saturation_railed = not (
        clamp_frac <= DERAIL_CLAMP_CEILING
        and n_distinct >= DERAIL_MIN_DISTINCT
        and n_unclamped >= MIN_UNCLAMPED_SIGMAS
    )
    touched_floor = min_ref <= POSITIVITY_FLOOR + POSITIVITY_FLOOR_TOL
    off_scale = (not math.isnan(ratio_full)) and abs(ratio_full) >= FLOOR_COLLAPSE_RATIO_MIN
    floor_collapsed = bool(touched_floor and off_scale)

    return bool(saturation_railed or floor_collapsed)


def _rail_kind(cell: Dict[str, Any]) -> str:
    """Which rail this cell hit -- recorded per seed so the anchor is auditable."""
    clamp_frac = float(cell["null_target_clamped_frac"])
    n_distinct = float(cell["null_series_n_distinct"])
    n_unclamped = int(cell["n_unclamped_sigmas"])
    ratio_full = float(cell["null_slope_ratio_full"])
    min_ref = float(cell["min_corrupt_target_precision"])
    sat = not (
        clamp_frac <= DERAIL_CLAMP_CEILING
        and n_distinct >= DERAIL_MIN_DISTINCT
        and n_unclamped >= MIN_UNCLAMPED_SIGMAS
    )
    floor = bool(
        min_ref <= POSITIVITY_FLOOR + POSITIVITY_FLOOR_TOL
        and (not math.isnan(ratio_full))
        and abs(ratio_full) >= FLOOR_COLLAPSE_RATIO_MIN
    )
    if sat and floor:
        return "both"
    if sat:
        return "saturation"
    if floor:
        return "positivity_floor_collapse"
    return "none"


# The known-degenerate POSITIVE CONTROL, frozen as a literal. These are the recorded
# ARM_NULL_ZERO values from
# `v3_exq_sd068_rem_unpaired_null_diagnostic_20260718T124216Z_v3.json` (V3-EXQ-778d),
# which replicated V3-EXQ-778c bit-identically -- the same floats also appear as
# V3-EXQ-778e's within-run legacy anchor. Frozen so the reachability guard needs zero
# compute and cannot drift with the substrate.
#
# Under the ORIGINAL 778d predicate these score 5/8 = 0.625 against a 0.75 gate (the
# defect). Under `_railed` above they score 8/8 = 1.0.
_REFERENCE_778C_NULL_ZERO: List[Dict[str, Any]] = [
    # seed 42  -- rail (b): looks healthy on clamp/distinct/unclamped, ratio 4348
    {"seed": 42, "null_target_clamped_frac": 0.2, "null_series_n_distinct": 5.0,
     "n_unclamped_sigmas": 4, "null_slope_ratio_full": 4348.4665081785715,
     "min_corrupt_target_precision": 0.001},
    # seed 7 -- rail (a): clamped on the whole grid, constant series, zero slope
    {"seed": 7, "null_target_clamped_frac": 1.0, "null_series_n_distinct": 1.0,
     "n_unclamped_sigmas": 0, "null_slope_ratio_full": 0.0,
     "min_corrupt_target_precision": 0.001},
    # seed 123 -- rail (b), ratio 9143
    {"seed": 123, "null_target_clamped_frac": 0.2, "null_series_n_distinct": 5.0,
     "n_unclamped_sigmas": 4, "null_slope_ratio_full": 9142.771353766831,
     "min_corrupt_target_precision": 0.001},
    {"seed": 2024, "null_target_clamped_frac": 1.0, "null_series_n_distinct": 1.0,
     "n_unclamped_sigmas": 0, "null_slope_ratio_full": 0.0,
     "min_corrupt_target_precision": 0.001},
    # seed 99 -- rail (b), ratio 1802 (the smallest floor-collapse ratio observed)
    {"seed": 99, "null_target_clamped_frac": 0.2, "null_series_n_distinct": 5.0,
     "n_unclamped_sigmas": 4, "null_slope_ratio_full": 1801.6453681003277,
     "min_corrupt_target_precision": 0.001},
    {"seed": 7777, "null_target_clamped_frac": 1.0, "null_series_n_distinct": 1.0,
     "n_unclamped_sigmas": 0, "null_slope_ratio_full": 0.0,
     "min_corrupt_target_precision": 0.001},
    {"seed": 314, "null_target_clamped_frac": 1.0, "null_series_n_distinct": 1.0,
     "n_unclamped_sigmas": 0, "null_slope_ratio_full": 0.0,
     "min_corrupt_target_precision": 0.001},
    {"seed": 1000, "null_target_clamped_frac": 1.0, "null_series_n_distinct": 1.0,
     "n_unclamped_sigmas": 0, "null_slope_ratio_full": 0.0,
     "min_corrupt_target_precision": 0.001},
]
_REFERENCE_SOURCE = (
    "V3-EXQ-778c rem null, replicated bit-identically by V3-EXQ-778d ARM_NULL_ZERO "
    "(run_id v3_exq_sd068_rem_unpaired_null_diagnostic_20260718T124216Z_v3) and by "
    "V3-EXQ-778e's within-run legacy calibration_error anchor"
)


def _fmt(v: float) -> str:
    """ASCII-safe float rendering that keeps UNAVAILABLE legible."""
    if v == H.UNAVAILABLE or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.4f}"


def _unpaired_target(seed: int) -> float:
    """An independent positive target of the same magnitude class as the clean one.

    Seeded off a DIFFERENT stream than anything the scored target or the damage uses, so
    it is statistically independent of the quantity the error is scored against -- the
    'no prior pairing' half of the Bar et al. control. Log-normal keeps it strictly
    positive, so this arm never needs the positivity clamp under investigation.
    """
    g = torch.Generator()
    g.manual_seed(int(seed) + UNPAIRED_SEED_OFFSET)
    z = float(torch.randn(1, generator=g).item())
    return float(CLEAN_TARGET_PRECISION * math.exp(UNPAIRED_LOG_SD * z))


def _unclamped_subgrid(
    sigmas: List[float], null_pr: Dict[float, Dict[str, Dict[str, float]]]
) -> List[float]:
    """Sigma points where the null arm's reference did NOT hit the positivity floor.

    Fitting a slope through saturated points measures the clamp, not the readout -- so
    the ratio is scored here and the readiness gate demands enough of these points.
    """
    return [
        s for s in sigmas if float(null_pr[s]["rem"].get("target_clamped", 0.0)) < 0.5
    ]


def _sweep_arm(
    *,
    seed: int,
    sigmas: List[float],
    warm: int,
    null_mode: str,
    inj_pr: Dict[float, Dict[str, Dict[str, float]]],
) -> Dict[str, Any]:
    """One null arm across the sigma grid, contrasted against the shared injected arm."""
    unpaired = _unpaired_target(seed) if null_mode == "unpaired_target" else None
    null_pr: Dict[float, Dict[str, Dict[str, float]]] = {}
    for s in sigmas:
        null_pr[s] = H.rem_only_integrity_at_sigma(
            seed=seed,
            sigma=s,
            warm_steps=warm,
            content_scale=0.0,
            null_mode=null_mode,
            unpaired_target_precision=unpaired,
        )

    # Score on the full grid (for comparability with 778c) AND on the unclamped subgrid
    # (the honest fit). The unclamped fit is what the criteria route on.
    full = H.rem_null_slope_ratio(
        sigmas=list(sigmas),
        injected_pr_by_sigma=inj_pr,
        null_pr_by_sigma=null_pr,
        rem_error_key="calibration_error",
    )
    clean_sigmas = _unclamped_subgrid(sigmas, null_pr)
    if len(clean_sigmas) >= 2:
        unclamped = H.rem_null_slope_ratio(
            sigmas=list(clean_sigmas),
            injected_pr_by_sigma=inj_pr,
            null_pr_by_sigma=null_pr,
            rem_error_key="calibration_error",
        )
    else:
        unclamped = {
            "null_slope_ratio": H.UNAVAILABLE,
            "injected_slope": H.UNAVAILABLE,
            "null_slope": H.UNAVAILABLE,
            "content_contingent": 0.0,
            "null_series_n_distinct": 0.0,
            "available": 0.0,
        }

    clamp_frac = float(full["null_target_clamped_frac"])
    n_distinct = float(full["null_series_n_distinct"])
    # Rail (b) evidence, read straight off the recorded integrity fields: the lowest
    # value the null arm's precision reference took anywhere on the sigma grid. Sitting
    # on POSITIVITY_FLOOR means the strictly-positive parameterisation clamped it.
    min_corrupt_target_precision = min(
        float(null_pr[s]["rem"].get("corrupt_target_precision", float("nan")))
        for s in sigmas
    )
    # Scored through the SHIPPED both-rails predicate -- the same callable the
    # reachability guard replays the frozen 778c reference through.
    rail_cell = {
        "null_target_clamped_frac": clamp_frac,
        "null_series_n_distinct": n_distinct,
        "n_unclamped_sigmas": len(clean_sigmas),
        "null_slope_ratio_full": full["null_slope_ratio"],
        "min_corrupt_target_precision": min_corrupt_target_precision,
    }
    railed = _railed(rail_cell)
    rail_kind = _rail_kind(rail_cell)
    derailed = not railed
    inj = full["injected_slope"]
    return {
        "null_mode": null_mode,
        "unpaired_target_precision": float(unpaired) if unpaired is not None else 0.0,
        "full_grid": full,
        "unclamped_grid": unclamped,
        "unclamped_sigmas": list(clean_sigmas),
        "n_unclamped_sigmas": len(clean_sigmas),
        "null_target_clamped_frac": clamp_frac,
        "null_series_n_distinct": n_distinct,
        "null_slope_ratio_full": full["null_slope_ratio"],
        "null_slope_ratio_unclamped": unclamped["null_slope_ratio"],
        "content_contingent_unclamped": float(unclamped["content_contingent"]) >= 1.0,
        "railed": bool(railed),
        "derailed": bool(derailed),
        "rail_kind": rail_kind,
        "min_corrupt_target_precision": min_corrupt_target_precision,
        "readiness_met": bool(
            inj != H.UNAVAILABLE
            and not math.isnan(inj)
            and abs(inj) >= INJECTED_SLOPE_FLOOR
        ),
        "integrity_null": {str(s): null_pr[s]["rem"] for s in sigmas},
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    warm = 8 if dry_run else WARM_STEPS
    sigmas = [0.0, 0.5, 2.0] if dry_run else SIGMAS

    print(
        "V3-EXQ-778h: SD-068 REM unpaired-target null, anchor-fix re-run of V3-EXQ-778d "
        "(H-rem-clamp-artifact)",
        flush=True,
    )
    print(
        f"  seeds={seeds} sigmas={sigmas} warm_steps={warm} arms={ARMS} "
        f"dry_run={dry_run}",
        flush=True,
    )

    # CHANGE 2 -- reachability guard. Replay the frozen, known-degenerate 778c reference
    # through the SHIPPED both-rails predicate BEFORE spending any compute, and refuse to
    # run if the gate exceeds what that reference can itself score. A precondition a
    # bit-perfect replication cannot pass is a guaranteed false negative; this is the
    # check that would have caught the 778d defect at design-audit time. Raises
    # AnchorUnreachable (an AssertionError) -> non-zero exit -> runner classifies ERROR,
    # which is the correct loud failure. Runs on dry-run too: the reference is frozen,
    # so the guard is dry-run-invariant and the smoke test exercises it.
    anchor_guard = assert_anchor_reachable(
        anchor_name="null_zero_anchor_reproduces_778c_railed_signature",
        reference_cells=_REFERENCE_778C_NULL_ZERO,
        score_fn=_railed,
        threshold=ANCHOR_MIN_RAILED_SEEDS_FRAC,
        reference_source=_REFERENCE_SOURCE,
    )
    print(
        f"  [guard] anchor reachability OK: the known-degenerate reference scores "
        f"{anchor_guard['n_reference_scored_true']}/"
        f"{anchor_guard['n_reference_cells']} = "
        f"{anchor_guard['reference_score']:.3f} under the shipped both-rails predicate "
        f"(gate {ANCHOR_MIN_RAILED_SEEDS_FRAC:.2f}); under the V3-EXQ-778d "
        "saturation-only predicate it could reach at most 0.625 -- unmeetable.",
        flush=True,
    )

    config_slice = {
        "sigmas": sigmas,
        "warm_steps": warm,
        "arms": list(ARMS),
        "clean_target_precision": CLEAN_TARGET_PRECISION,
        "unpaired_log_sd": UNPAIRED_LOG_SD,
        "unpaired_seed_offset": UNPAIRED_SEED_OFFSET,
        "null_slope_ratio_ceiling": NULL_SLOPE_RATIO_CEILING,
        "injected_slope_floor": INJECTED_SLOPE_FLOOR,
        "derail_clamp_ceiling": DERAIL_CLAMP_CEILING,
        "derail_min_distinct": DERAIL_MIN_DISTINCT,
        "min_unclamped_sigmas": MIN_UNCLAMPED_SIGMAS,
        "rem_error_key": "calibration_error",
        "shy_decay_rate": 0.85,
        "body_obs_dim": H.BODY_OBS_DIM,
        "world_obs_dim": H.WORLD_OBS_DIM,
        "action_dim": H.ACTION_DIM,
        "harm_obs_dim": H.HARM_OBS_DIM,
    }

    arm_results: List[Dict[str, Any]] = []
    total_eps = 2  # two null arms swept per seed against the shared injected arm

    for seed in seeds:
        print(f"Seed {seed} Condition REM_UNPAIRED_NULL", flush=True)
        with arm_cell(
            seed,
            config_slice=config_slice,
            script_path=Path(__file__),
            config_slice_declared=True,
        ) as cell_ctx:
            # Shared injected arm -- swept ONCE, contrasted against both nulls.
            inj_pr: Dict[float, Dict[str, Dict[str, float]]] = {
                s: H.rem_only_integrity_at_sigma(
                    seed=seed, sigma=s, warm_steps=warm, content_scale=1.0
                )
                for s in sigmas
            }

            arms: Dict[str, Any] = {}
            for i, (name, mode) in enumerate(
                (("NULL_ZERO", "zero_content"), ("NULL_UNPAIRED", "unpaired_target"))
            ):
                a = _sweep_arm(
                    seed=seed, sigmas=sigmas, warm=warm, null_mode=mode, inj_pr=inj_pr
                )
                arms[name] = a
                print(
                    f"  [train] unpaired_null seed={seed} ep {i + 1}/{total_eps} "
                    f"arm={name} clamp_frac={a['null_target_clamped_frac']:.2f} "
                    f"n_distinct={int(a['null_series_n_distinct'])} "
                    f"n_unclamped={a['n_unclamped_sigmas']} "
                    f"ratio={_fmt(a['null_slope_ratio_unclamped'])} "
                    f"{'RAILED' if a['railed'] else 'DERAILED'}"
                    f"({a['rail_kind']})",
                    flush=True,
                )

            anchor = arms["NULL_ZERO"]
            unpaired = arms["NULL_UNPAIRED"]
            readiness = anchor["readiness_met"] and unpaired["readiness_met"]

            row: Dict[str, Any] = {
                "seed": seed,
                "arm": "REM_UNPAIRED_NULL",
                "arms_compared": list(ARMS),
                "hypothesis_id": HYPOTHESIS_ID,
                "sigmas": list(sigmas),
                "arms": arms,
                "anchor_null_zero_railed": bool(anchor["railed"]),
                "unpaired_derailed": bool(unpaired["derailed"]),
                "unpaired_content_contingent": bool(
                    unpaired["content_contingent_unclamped"]
                ),
                "unpaired_target_precision": unpaired["unpaired_target_precision"],
                "readiness_met": bool(readiness),
                # OFF/baseline (INJECTED) arm internals recorded as richly as the nulls.
                "integrity_injected": {str(s): inj_pr[s]["rem"] for s in sigmas},
            }
            cell_ctx.stamp(row)

        arm_results.append(row)
        print(
            f"  anchor_null_zero_railed={row['anchor_null_zero_railed']} "
            f"unpaired_derailed={row['unpaired_derailed']} "
            f"unpaired_content_contingent={row['unpaired_content_contingent']}",
            flush=True,
        )
        print(f"verdict: {'PASS' if row['unpaired_derailed'] else 'FAIL'}", flush=True)

    n = len(arm_results)
    n_anchor_railed = sum(1 for r in arm_results if r["anchor_null_zero_railed"])
    n_derail = sum(1 for r in arm_results if r["unpaired_derailed"])
    n_contingent = sum(1 for r in arm_results if r["unpaired_content_contingent"])
    anchor_frac = (n_anchor_railed / n) if n else 0.0
    anchor_ok = anchor_frac >= ANCHOR_MIN_RAILED_SEEDS_FRAC
    readiness_ok = all(r["readiness_met"] for r in arm_results)

    derail_majority = n_derail > n / 2
    contingent_majority = n_contingent > n / 2

    ratios = [
        r["arms"]["NULL_UNPAIRED"]["null_slope_ratio_unclamped"]
        for r in arm_results
        if r["arms"]["NULL_UNPAIRED"]["null_slope_ratio_unclamped"] != H.UNAVAILABLE
        and not math.isnan(r["arms"]["NULL_UNPAIRED"]["null_slope_ratio_unclamped"])
    ]
    mean_ratio = (sum(ratios) / len(ratios)) if ratios else H.UNAVAILABLE
    if len(ratios) >= 2:
        var = sum((v - mean_ratio) ** 2 for v in ratios) / (len(ratios) - 1)
        sd_ratio = math.sqrt(var)
        sem = sd_ratio / math.sqrt(len(ratios))
        ci_lo, ci_hi = mean_ratio - 1.96 * sem, mean_ratio + 1.96 * sem
    else:
        sd_ratio = ci_lo = ci_hi = H.UNAVAILABLE

    arm_summary = {
        "NULL_ZERO": {
            "n_seeds_railed": n_anchor_railed,
            "railed_frac": anchor_frac,
            "per_seed_clamp_frac": [
                r["arms"]["NULL_ZERO"]["null_target_clamped_frac"] for r in arm_results
            ],
            "per_seed_null_slope_ratio_full": [
                r["arms"]["NULL_ZERO"]["null_slope_ratio_full"] for r in arm_results
            ],
            # Which rail each seed hit -- the audit trail for the CHANGE-1 correction.
            # V3-EXQ-778d scored only "saturation" seeds railed; the
            # "positivity_floor_collapse" seeds are the 3/8 it wrongly called healthy.
            "per_seed_rail_kind": [
                r["arms"]["NULL_ZERO"]["rail_kind"] for r in arm_results
            ],
            "per_seed_min_corrupt_target_precision": [
                r["arms"]["NULL_ZERO"]["min_corrupt_target_precision"]
                for r in arm_results
            ],
            "n_seeds_saturation_railed": sum(
                1
                for r in arm_results
                if r["arms"]["NULL_ZERO"]["rail_kind"] in ("saturation", "both")
            ),
            "n_seeds_floor_collapsed": sum(
                1
                for r in arm_results
                if r["arms"]["NULL_ZERO"]["rail_kind"]
                in ("positivity_floor_collapse", "both")
            ),
        },
        "NULL_UNPAIRED": {
            "n_seeds_derailed": n_derail,
            "n_seeds_content_contingent": n_contingent,
            "mean_null_slope_ratio_unclamped": mean_ratio,
            "sd_null_slope_ratio_unclamped": sd_ratio,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
            "ceiling_inside_ci95": bool(
                ci_lo != H.UNAVAILABLE and ci_lo <= NULL_SLOPE_RATIO_CEILING <= ci_hi
            ),
            "per_seed_clamp_frac": [
                r["arms"]["NULL_UNPAIRED"]["null_target_clamped_frac"]
                for r in arm_results
            ],
            "per_seed_n_unclamped_sigmas": [
                r["arms"]["NULL_UNPAIRED"]["n_unclamped_sigmas"] for r in arm_results
            ],
            "per_seed_null_slope_ratio_unclamped": [
                r["arms"]["NULL_UNPAIRED"]["null_slope_ratio_unclamped"]
                for r in arm_results
            ],
            "per_seed_unpaired_target_precision": [
                r["unpaired_target_precision"] for r in arm_results
            ],
        },
    }

    # Self-routed label. Readiness or a failed anchor routes ONLY to requeue.
    if not readiness_ok or not anchor_ok:
        label = "substrate_not_ready_requeue"
    elif derail_majority and contingent_majority:
        label = "rem_clamp_artifact_confirmed_content_contingent_when_declamped"
    elif derail_majority:
        label = "rem_clamp_artifact_confirmed_but_readout_still_content_free"
    else:
        label = "rem_clamp_artifact_refuted_unpaired_null_also_railed"

    overall_pass = bool(readiness_ok and anchor_ok and derail_majority)

    min_inj_slope = min(
        abs(r["arms"]["NULL_ZERO"]["full_grid"]["injected_slope"])
        for r in arm_results
        if r["arms"]["NULL_ZERO"]["full_grid"]["injected_slope"] != H.UNAVAILABLE
        and not math.isnan(r["arms"]["NULL_ZERO"]["full_grid"]["injected_slope"])
    )

    interpretation = {
        "label": label,
        "hypothesis_id": HYPOTHESIS_ID,
        "hypothesis_axis": HYPOTHESIS_AXIS,
        "hypothesis_question": HYPOTHESIS_QUESTION,
        "probe_amended_pre_queue": {
            "retired_probe": "step ladder (0.1/0.25/0.5/1.0)",
            "reason": (
                "PROVABLY INERT. rv_after = (1-step)*rv_before + step*(1/(target+1e-6)) "
                "(e3_selector.recalibrate_precision_to), so (a) the 1e-3 positivity "
                "clamp is applied UPSTREAM of step, making the clamp fraction exactly "
                "step-invariant, and (b) both arms' sigma-slopes carry the same factor "
                "step, which CANCELS in the reported ratio. Verified numerically over "
                "step in {0.1,0.25,0.5,1.0}: ratio identical to 12 significant figures "
                "(14569.1833719598-...602, float round-off only), clamp fraction and "
                "distinct-value count bit-identical. Running it would have returned "
                "this leg's declared null BY CONSTRUCTION -- a false elimination in the "
                "frozen ledger, not a measurement."
            ),
            "replacement_probe": (
                "unpaired-target null: ARM_NULL_UNPAIRED points the recalibration at an "
                "INDEPENDENT positive target of the same magnitude class, unpaired with "
                "the clean target the error is scored against (the faithful Bar et al. "
                "2020 'same odour, no prior pairing' analog), contrasted against "
                "ARM_NULL_ZERO (the 778c zero-target null, retained as the replication "
                "anchor). The hypothesis adjudicated is unchanged; only the probe is."
            ),
            "ledger_amendment_required": True,
            "ledger_amendment_blocked_by": (
                "active TASK_CLAIMS entry 'axis-family convergence discriminator' "
                "(claimed 2026-07-18T08:05:00Z) holds "
                "REE_assembly/evidence/planning/hypothesis_space_registry.v1.json"
            ),
        },
        "declared_null": (
            "if ARM_NULL_UNPAIRED stays railed too -- still clamping, still constant -- "
            "H-rem-clamp-artifact is REFUTED: the both-rails degeneracy is NOT an "
            "artifact of how the null was operationalised, and the measurement axis is "
            "eliminated from question consolidation_readout_validity."
        ),
        "preconditions": [
            {
                "name": "injected_arm_sigma_slope_supra_floor",
                "description": (
                    "the ratio's DENOMINATOR -- the injected-arm sigma-slope, the same "
                    "statistic the content-contingency criterion routes on -- clears "
                    "the floor. Below floor means the sweep never damaged the readout, "
                    "so no ratio is interpretable."
                ),
                "measured": float(min_inj_slope),
                "threshold": INJECTED_SLOPE_FLOOR,
                "direction": "lower",
                "control": "injected arm swept to sigma=max (known-damaged positive control)",
                "met": bool(readiness_ok),
            },
            {
                "name": "null_zero_anchor_reproduces_778c_railed_signature",
                "description": (
                    "READINESS ANCHOR (both-rails predicate; CORRECTED in V3-EXQ-778h). "
                    "ARM_NULL_ZERO IS the V3-EXQ-778c rem condition on the same seeds "
                    "and the same RNG stream, so it must reproduce 778c's railed "
                    "signature. Measured as the FRACTION of seeds railed -- the SAME "
                    "railed/de-railed statistic the load-bearing criterion routes on, "
                    "asserted on the known-degenerate positive control. A seed counts as "
                    "railed if it hit EITHER rail: (a) SATURATION (reference clamped "
                    "across the grid -> constant series -> zero slope) or (b) "
                    "POSITIVITY-FLOOR COLLAPSE (reference touched the 1e-3 floor and the "
                    "fit went off-scale, |ratio| >= "
                    f"{FLOOR_COLLAPSE_RATIO_MIN}). V3-EXQ-778d's predicate tested rail "
                    "(a) ONLY, so the 3/8 floor-collapsed seeds (ratios 1801-9143) were "
                    "scored HEALTHY and the anchor could reach at most 5/8 = 0.625 "
                    "against a 0.75 gate -- unmeetable by construction, and the sole "
                    "cause of that run's degeneracy flag on the load-bearing criterion. "
                    "A reachability guard now asserts at setup that the frozen reference "
                    "clears this gate under the shipped predicate. If the anchor does "
                    "not reproduce, something other than the null-mode differs from 778c "
                    "and the contrast is uninterpretable -> requeue, never a verdict "
                    "about the clamp."
                ),
                "measured": float(anchor_frac),
                "threshold": ANCHOR_MIN_RAILED_SEEDS_FRAC,
                "direction": "lower",
                "control": "ARM_NULL_ZERO == the V3-EXQ-778c null (known-degenerate positive control)",
                "met": bool(anchor_ok),
            },
        ],
        "criteria_non_degenerate": {
            "C1_unpaired_null_derails": bool(readiness_ok and anchor_ok),
            # Content-contingency is vacuous unless the unpaired arm actually de-railed.
            "C2_unpaired_ratio_content_contingent": bool(derail_majority),
            "C3_anchor_reproduces_778c": bool(n >= 2),
        },
        "criteria": [
            {
                "name": "C1_unpaired_null_derails",
                "load_bearing": True,
                "passed": bool(derail_majority),
            },
            {
                "name": "C2_unpaired_ratio_content_contingent",
                "load_bearing": False,
                "passed": bool(contingent_majority),
            },
            {
                "name": "C3_anchor_reproduces_778c",
                "load_bearing": False,
                "passed": bool(anchor_ok),
            },
        ],
        "arm_summary": arm_summary,
        "anchor_railed_seed_frac": float(anchor_frac),
        # Provenance for the CHANGE-2 guard: proof, recorded in the shipped artifact,
        # that this run's readiness gate is reachable by its own reference.
        "anchor_reachability_guard": anchor_guard,
        "portfolio": {
            "gov_rule": "GOV-FANOUT-1",
            "question": HYPOTHESIS_QUESTION,
            "this_leg": f"{HYPOTHESIS_ID} (axis={HYPOTHESIS_AXIS})",
            "sibling_legs": [
                "V3-EXQ-778e H-rem-genuinely-content-free (axis=representation)",
                "V3-EXQ-778f H-gen-gain-content-free (axis=observation)",
            ],
            "note": (
                "Read the three legs JOINTLY. This leg alone cannot resolve the parent "
                "H-rem-content-contingent: de-railing establishes only that the "
                "zero-target null was mis-operationalised, not that the readout tracks "
                "content. Resolution via /failure-autopsy Step 9b Mode B against the "
                "frozen ledger, not by this self-route."
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
        f"anchor NULL_ZERO railed: {n_anchor_railed}/{n} "
        f"(need >= {ANCHOR_MIN_RAILED_SEEDS_FRAC:.0%}) | "
        f"NULL_UNPAIRED derailed: {n_derail}/{n} | "
        f"content_contingent: {n_contingent}/{n}",
        flush=True,
    )
    print(
        f"  NULL_UNPAIRED mean ratio (unclamped grid)={_fmt(mean_ratio)} "
        f"sd={_fmt(sd_ratio)} ci95=[{_fmt(ci_lo)}, {_fmt(ci_hi)}] "
        f"(ceiling {NULL_SLOPE_RATIO_CEILING})",
        flush=True,
    )
    print(f"self-route label: {label}", flush=True)

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": per_claim,
        "interpretation": interpretation,
        "arm_results": arm_results,
        "arm_summary": arm_summary,
        "config": config_slice,
        "seeds": seeds,
        "non_degenerate": bool(readiness_ok and anchor_ok),
        "degeneracy_reason": (
            ""
            if (readiness_ok and anchor_ok)
            else (
                "readiness below floor and/or ARM_NULL_ZERO failed to reproduce "
                "V3-EXQ-778c's railed signature; the null-mode contrast is "
                "uninterpretable"
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
        "supersedes": SUPERSEDES,
        "supersedes_run_id": SUPERSEDES_RUN_ID,
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
        "arm_summary": result["arm_summary"],
        "hypothesis_id": HYPOTHESIS_ID,
        "hypothesis_axis": HYPOTHESIS_AXIS,
        "hypothesis_question": HYPOTHESIS_QUESTION,
        "acceptance_criteria": {
            "C1_unpaired_null_derails": (
                f"on a majority of seeds ARM_NULL_UNPAIRED leaves the rails: clamp "
                f"fraction <= {DERAIL_CLAMP_CEILING}, null series has >= "
                f"{DERAIL_MIN_DISTINCT} distinct values, and >= {MIN_UNCLAMPED_SIGMAS} "
                "unclamped sigma points to fit through (LOAD-BEARING)"
            ),
            "C2_unpaired_ratio_content_contingent": (
                f"given a de-rail, null_slope_ratio <= {NULL_SLOPE_RATIO_CEILING} on the "
                "unclamped subgrid, on a majority of seeds (separates 'the null was "
                "mis-operationalised' from 'the readout tracks content' -- NOT the same "
                "finding)"
            ),
            "C3_anchor_reproduces_778c": (
                f"ARM_NULL_ZERO reproduces V3-EXQ-778c's railed signature on >= "
                f"{ANCHOR_MIN_RAILED_SEEDS_FRAC:.0%} of seeds, scored on the BOTH-RAILS "
                "predicate: saturation (clamped across the grid -> constant series) OR "
                f"positivity-floor collapse (reference on the {POSITIVITY_FLOOR} floor "
                f"AND |ratio| >= {FLOOR_COLLAPSE_RATIO_MIN}). Readiness anchor; below "
                "-> substrate_not_ready_requeue. Gate reachability is asserted against "
                "the frozen reference at setup by "
                "_lib/readiness_anchor.assert_anchor_reachable"
            ),
        },
        "arm_results": result["arm_results"],
        "notes": (
            "V3-EXQ-778h: SAME-QUESTION alphabetic-suffix re-run of V3-EXQ-778d whose "
            "ONLY change is fixing a readiness-anchor specification defect. LEDGER "
            "HYGIENE, not a blocker -- the science is already adjudicated. THE SCIENCE "
            "IS UNCHANGED: same arms, seeds, sigma grid, RNG streams, thresholds and "
            "criteria; the unpaired-target null (ARM_NULL_UNPAIRED vs the ARM_NULL_ZERO "
            "anchor) is correct and is the faithful Bar et al. 2020 'same odour, no "
            "prior pairing' analog; the retired step-ladder probe is NOT revisited (it "
            "was proven algebraically inert -- step cancels in the reported ratio, "
            "verified identical to 12 significant figures over step in "
            "{0.1,0.25,0.5,1.0} -- and that retirement is recorded at "
            "hypothesis_space_registry.v1.json -> consolidation_readout_validity -> "
            "H-rem-clamp-artifact -> probe_amendments[0]). "
            "CHANGE 1: the `railed` predicate now counts BOTH rails. 778d tested only "
            "the SATURATION rail, so the 3/8 POSITIVITY-FLOOR-COLLAPSE seeds (clamp_frac "
            "0.2, n_distinct 5, n_unclamped 4, but ratios 1801-9143 off the 1e-3 floor) "
            "were scored HEALTHY. Maximum achievable anchor fraction for a BIT-PERFECT "
            "replication of 778c was therefore exactly 5/8 = 0.625 against a 0.75 gate: "
            "UNMEETABLE BY CONSTRUCTION. Since criteria_non_degenerate."
            "C1_unpaired_null_derails = (readiness_ok and anchor_ok) with readiness_ok "
            "true on 8/8, that one mis-specified statistic accounted for the entire "
            "degeneracy flag on the load-bearing criterion, and the self-route "
            "mislabelled the cause `substrate_not_ready_requeue` when nothing about the "
            "substrate was unready -- correct classification "
            "`instrument_specification_gap`. Both rails are now derived from the "
            "RECORDED INTEGRITY FIELDS (min corrupt_target_precision on the positivity "
            "floor + off-scale full-grid ratio), never from hardcoded seed indices. "
            "CHANGE 2 (Learning 1 of the autopsy, generalisable): "
            "_lib/readiness_anchor.assert_anchor_reachable replays the frozen "
            "known-degenerate 778c reference through THE SHIPPED predicate at setup and "
            "raises if the gate exceeds what that reference can score -- a precondition "
            "a perfect replication cannot pass is a guaranteed false negative. Factored "
            "into _lib/ so other readiness anchors reuse it. "
            "EXPECTED: anchor clears 8/8, C1's non_degenerate flag becomes true, and "
            "C1_unpaired_null_derails still PASSes at 7/8 seeds de-railed -- the leg's "
            "existing verdict reproduced clean-as-shipped. "
            "LEDGER: H-rem-clamp-artifact is ALREADY resolved `confirmed` with "
            "`control_rehabilitated: true` on this corrected predicate; this run does "
            "NOT change that resolution, it makes the control provenance "
            "clean-as-shipped rather than clean-as-rehabilitated. A governance session "
            "may then clear `control_rehabilitated`. This experiment does NOT edit "
            "hypothesis_space_registry.v1.json (GOV-FROZEN-1 single-producer rule: only "
            "/failure-autopsy Step 9b writes it). Diagnosis: "
            "failure_autopsy_SD-068-rem-fanout-cluster_2026-07-18.{md,json}, REE_assembly "
            "master deda5e17ed. "
            "GOV-REUSE-1: the corrected anchor statistic IS computable post-hoc over "
            "778d's recorded arm_results -- the autopsy did exactly that to establish "
            "8/8, and that reanalysis is what `control_rehabilitated: true` records. "
            "This run is queued anyway, at LOW priority, because the deliverable is not "
            "the number but the PROVENANCE: a shipped artifact whose own readiness "
            "anchor passes under its own shipped predicate. Reanalysis cannot produce "
            "that. Experiment-layer only; zero ree_core change. "
            "--- inherited context from V3-EXQ-778d follows --- "
            "SD-068 GOV-FANOUT-1 discrimination portfolio, leg 1 of 3 (axis=MEASUREMENT, "
            "hypothesis H-rem-clamp-artifact, pre-registered 2026-07-18T08:41:15Z in "
            "hypothesis_space_registry.v1.json question consolidation_readout_validity). "
            "Routed by failure_autopsy_V3-EXQ-778c_2026-07-18.json "
            "targets[0].fanout_recommendation. PROBE AMENDED BEFORE QUEUING: the "
            "sketched `step` ladder is PROVABLY INERT (the 1e-3 clamp is applied "
            "upstream of step, and step cancels in the null/injected ratio -- verified "
            "identical to 12 significant figures over step in {0.1,0.25,0.5,1.0}), so it "
            "would have returned the declared null by construction and written a FALSE "
            "ELIMINATION into the frozen ledger. Replaced with an unpaired-target null: "
            "ARM_NULL_UNPAIRED points the recalibration at an INDEPENDENT positive "
            "target of the same magnitude class, unpaired with the scored clean target "
            "-- the faithful Bar et al. 2020 'same odour, no prior pairing' analog -- "
            "contrasted against ARM_NULL_ZERO (the 778c null, retained as replication "
            "anchor on the same 8 seeds and RNG stream). The ratio is fitted on the "
            "UNCLAMPED subgrid so a slope is never drawn through saturated points. The "
            "LEDGER entry still records the retired probe and needs amending; this "
            "session did not write it because an active TASK_CLAIMS entry ('axis-family "
            "convergence discriminator', 2026-07-18T08:05:00Z) holds that file. rem "
            "phase ONLY (the sws repair is a single unambiguous build, exempt from "
            "GOV-FANOUT-1; nrem is already confirmed content-contingent). DIAGNOSTIC: "
            "excluded from governance confidence/conflict scoring; PROMOTES/DEMOTES "
            "NOTHING. MECH-121 deliberately NOT tagged (held; NREM leg is "
            "substrate-plumbing-fidelity only). Siblings V3-EXQ-778e (representation) "
            "and V3-EXQ-778f (observation) run in PARALLEL and are read jointly. "
            "Experiment-layer only; zero ree_core change. GOV-REUSE-1: the decisive "
            "readout is the null-arm sigma-slope under an UNPAIRED (non-zero) target, "
            "which NO recorded manifest carries -- every SD-068 run (778, 778a, 778b, "
            "778c) operationalised the null as a ZERO target -- so it is not recoverable "
            "by reanalysis -> run."
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
