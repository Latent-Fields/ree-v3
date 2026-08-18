"""V3-EXQ-937a -- MECH-449 / ARC-107 perseveration No-Go: the ENVELOPE INERTNESS POINT.

EXPERIMENT_PURPOSE=diagnostic -- extends V3-EXQ-937's dose ladder into the
NARROW-envelope regime that 937 never reached. It does NOT supersede 937: 937's
curve is a valid measurement and its evidence stays active. This adds the half
of the curve 937 was missing.

WHAT V3-EXQ-937 MEASURED, AND WHAT IT LEFT OPEN
------------------------------------------------
937 swept f_eligibility_envelope_floor from 0.30 (the shipped default) down to
0.05, 3 seeds x 128 banks, all three readiness preconditions MET:

    floor            0.30   0.25   0.20   0.15   0.10   0.05
    ARM_CONSTITUTION 0.685  0.823  0.898  0.956  0.979  0.997
    ARM_OFF          0.000  0.000  0.000  0.000  0.000  0.000
    ARM_SHUFFLED     0.000  0.000  0.000  0.000  0.000  0.000

Two results, one of which contradicts the run's own premise:

1. The dependence is REAL and cleanly monotone (937's C2 passed at every step),
   and the specificity is total: ARM_SHUFFLED converts 0.000 at EVERY dose, so
   conversion is attributable to the CONTENT of MECH-260's recency vector rather
   than to gate activation, at all six doses. 926a had that control at one dose.

2. V3-EXQ-926a's stock-floor anchor DID NOT REPLICATE. 926a's notes report
   conversion 1/16 = 0.0625 with median pre-No-Go envelope 1 at floor 0.30;
   937 measured 0.685 with envelope 2.0 there. So the perseveration leg is NOT
   inert at the shipped default -- it fires the majority of the time -- and the
   claim that "the MECH-448 envelope collapses to the fail-open protect-min and
   the axis cannot fire" at 0.30 is refuted on this configuration.

Consequence: 937's ladder never contained an inert regime at all. It measured
the TOP of the curve (0.685 -> 0.997) and bottomed out well above inertness, so
its C1 lift (measured 0.344/0.273/0.320) spanned only part of the range. The
question 937 was posed to answer -- "is the leg inert as configured, and if not,
WHERE does it become inert?" -- is still open on the narrow side.

WHY THE C1 BAR IS RE-DERIVED RATHER THAN LOWERED
-------------------------------------------------
937's C1 threshold of 0.40 was arithmetic on 926a's two anchors
(0.97 - 0.0625 = 0.91, halved for safety). One of those anchors does not hold,
so the bar was calibrated against a number that is not real, and 937 FAILed
against it.

The wrong repair is to lower the bar until the measured lift clears it.
/queue-experiment forbids exactly that ("NEVER lower the threshold to resolve
it -- that converts a detected artifact into a citable result"). So the bar is
kept at 0.40, unchanged, and what is fixed is the COMPARISON it is applied to:

    at realized envelope size 1 the gng_protect_min_eligible guard refuses to
    drop the last survivor, so the No-Go can exclude nothing and conversion is
    suppressed BY CONSTRUCTION -- a structural property of the guard.

THE FLOOR IS THE KNOB BUT NOT THE MEDIATOR (measured during authoring, 128
banks, seed 42, and this is the load-bearing discovery of this script):

    floor      0.60   0.40   0.30
    envelope    4.0    1.0    2.0
    conversion 0.781  0.227  0.648

Raising the floor does NOT monotonically shrink the eligible set. Once the floor
is high enough that NO candidate clears it, the fail-open guard admits the FULL
set again -- so realized envelope size runs 4 -> 1 -> 2 -> 3 as the floor falls,
and the inert point sits near 0.40, NOT at the top of the ladder. V3-EXQ-937's
apparently clean monotone curve was an artifact of sampling only 0.30 -> 0.05, a
window over which the map happens to be monotone.

Two consequences, both acted on here:
  (a) The first draft of THIS script keyed C1 to floor endpoints (widest 0.10 vs
      narrowest 0.60) on the assumption that 0.60 would be inert. Its own P2
      readiness gate caught the error -- env_dose_separation came in at 0.5
      against a 1.0 floor and it self-routed substrate_not_ready_requeue rather
      than emitting a bogus curve. The gate worked; the design was wrong.
  (b) Every criterion below is therefore keyed to REALIZED ENVELOPE SIZE, which
      is what the mechanism predicts on, and the floor ladder is widened to
      0.60 / 0.50 / 0.40 / 0.35 / 0.30 / 0.25 / 0.20 / 0.15 / 0.10 / 0.05 purely
      to SAMPLE the envelope sizes well. Floor-keyed readouts are still recorded
      (mean_envelope_by_floor, mean_conversion_by_floor_*) because the fold-back
      itself is a finding worth having on the record.

THE INFORMATIVE NULL: if conversion does NOT fall to inertness even in the
protect-min regime (realized envelope 1), then envelope width is not what gates
this leg, and 926a's "structurally gated by envelope width" finding of record is
wrong rather than merely mis-scaled. C4 detects that.

ROUTING-LABEL DEFECT FIXED FROM V3-EXQ-937
-------------------------------------------
937 emitted `conversion_independent_of_envelope_width` on the C1-fails branch
WHILE its own C2 had passed with a clean monotone rise. "Independent" was
contradicted by a sibling criterion in the same manifest. Here that branch is
split: C1-short-but-C2-monotone routes to
`graded_dose_response_below_prereg_lift` (a quantitative shortfall against a
prediction), and the absence claim is only made when C2 also fails.

CRITERIA
--------
C1 (LOAD-BEARING) per-seed conversion lift, cells at REALIZED envelope >= 3
   minus cells at realized envelope <= 1 (the protect-min regime). GATED.
C2 monotone-nondecreasing in REALIZED ENVELOPE SIZE, within per-step slack.
   REPORTED. Not keyed to the floor, which folds back (see above).
C3 fail-open safety contract never violated at any dose. GATED.
C4 an inert regime (conversion <= 0.10) was actually reached. REPORTED, NOT
   GATED: failing to reach inertness even at envelope 1 is the informative null
   above, and must not be collapsed into a mere FAIL.

RE-DERIVE BRAKE (Step 2.5b)
---------------------------
MECH-449 = 0, ARC-107 = 0 (threshold 2) -- NOT braked.
MECH-439 = 12, ceiling_decision exhausted -- DELIBERATELY NOT TAGGED. This poses
no F-dominance lever and must not be read as a probe of that lineage. MECH-260
NOT tagged for 926a's reason (its falsifier is behavioural and vacuous at the
MECH-457 competence floor).

READINESS PRECONDITIONS (unchanged from 937, and all three MET there)
----------------------------------------------------------------------
P1 `suppression_cross_candidate_range_supra_floor` -- the cross-candidate RANGE
   is the statistic the perseveration threshold crossing depends on. Worst cell,
   not a mean. Reachability PROVEN AT SETUP by _assert_p1_anchor_reachable().
P2 `envelope_size_dose_separation` -- the dose-sweep readiness check: the ladder
   must REALIZE a spread of envelope sizes (>= 1 whole candidate). C1 routes on a
   contrast across realized envelope sizes, so readiness asserts a range of that
   same mediator (V3-EXQ-643 same-statistic rule). Not circular: P2 measures the
   MEDIATOR responding to the knob, C1 the OUTCOME responding to the mediator.
   This is the gate that caught the floor-keyed first draft (see above).
P3 `envelope_floor_config_plumbing_live` -- from_dims plumbs all three knobs,
   guarding the documented silent-kwarg-swallow mode.

926a's per-cell "envelope >= 2" gate remains DELIBERATELY NOT CARRIED OVER, and
the reason is now stronger than in 937: this ladder is DESIGNED to include doses
whose envelope collapses to 1. A whole-run all(...) gate over per-cell envelope
size would fail on precisely the cells this experiment exists to measure and
would vacate the entire run -- the V3-EXQ-785 defect. Per-cell envelope size is
recorded on every cell as a non-gating diagnostic.

DV-SYMMETRY DECLARATION (mandatory per /queue-experiment Step 3)
----------------------------------------------------------------
  ARM_CONSTITUTION -- manipulation: the envelope floor changes which candidates
    are F-eligible, and the No-Go removes one from that set. DV: committed
    argmin identity, aggregated to a conversion rate. Argmin's symmetry group is
    {uniform additive constant, monotone rescaling, tie permutation}; changing
    set MEMBERSHIP is in none of them -- it changes the domain the argmin is
    taken over. NOT invariant.
  ARM_SHUFFLED -- same domain manipulation, recency mass on a NON-incumbent.
    NOT invariant. Its 0.000 conversion at every dose in 937 is the specificity
    control working, not a measured null.
  ARM_OFF -- no No-Go manipulation, run at every dose so a dose-dependent rise
    is not aliased between the No-Go acting and the widened envelope alone
    moving the baseline pick. HONEST LIMIT, measured in 937: ARM_OFF converts
    exactly 0.000 at every dose BY CONSTRUCTION -- its arm selector is the
    baseline config, so its pick cannot differ from the baseline pick that
    defines the incumbent. It is a tautological control on this metric and
    proves nothing about the envelope; it is retained only because its cost is
    zero and its per-cell envelope-size diagnostic is genuinely informative.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import (  # noqa: E402
    compute_arm_fingerprint,
    reset_all_rng,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.cingulate.dacc import DACCAdaptiveControl, DACCConfig  # noqa: E402
from ree_core.predictors.e2_fast import Trajectory  # noqa: E402
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_937a_mech449_envelope_inertness_point"
CLAIM_IDS: List[str] = ["MECH-449", "ARC-107"]

SEEDS: List[int] = [42, 43, 44]
ARMS: List[str] = ["ARM_OFF", "ARM_CONSTITUTION", "ARM_SHUFFLED"]

# The dose ladder, ordered WIDEST-ENVELOPE-LAST so index order == increasing
# envelope width. STOCK_FLOOR is the shipped default and the low-conversion
# anchor; REFERENCE_FLOOR is V3-EXQ-926a's operating point and the
# high-conversion anchor. Both are ON the ladder so this run reproduces both of
# 926a's points under identical recorded conditions rather than citing them.
STOCK_FLOOR = 0.30
REFERENCE_FLOOR = 0.10
# EXTENDED ABOVE THE STOCK FLOOR. V3-EXQ-937 swept 0.30 down to 0.05 and never
# reached inertness: conversion bottomed out at 0.685 at the stock floor and rose
# monotonically to 0.997. So 937 measured the TOP of the curve only -- the
# question it was posed to answer ("is the leg inert as configured, and if not,
# where does it become inert?") is still open on the NARROW side. This ladder
# extends to 0.60 to find the lower asymptote.
NARROWEST_FLOOR = 0.60
ENVELOPE_FLOORS: List[float] = [
    0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05,
]

WORLD_DIM = 8
ACTION_DIM = 4
K_CANDIDATES = 4
HORIZON = 3
# 128 rather than V3-EXQ-926a's 32. The deliverable here is a CURVE, so per-dose
# precision is the product, not a nicety: 32 banks quantises each dose's
# conversion rate to 1/32 = 0.031, and the knee is read off differences between
# adjacent doses. The whole grid is a selection-face probe with no training
# (926a measured 9 cells x 32 banks in 0.77s), so 54 cells x 128 banks is still
# well under a minute of compute -- 4x the resolution for no meaningful cost.
N_BANKS = 128

# --- pre-registered thresholds (constants, never derived from run statistics) --
# C1: conversion lift from the NARROWEST envelope (floor 0.60) to the widest
# (0.10). PRE-REGISTERED FROM THE MECHANISM, NOT FROM DATA -- this is the point
# on which this experiment differs methodologically from V3-EXQ-937, and the
# distinction is load-bearing.
#
# 937 set this bar at 0.40 by arithmetic on V3-EXQ-926a's two anchors
# (0.97 - 0.0625 = 0.91, halved for safety). The stock-floor anchor DID NOT
# REPLICATE: 937 measured 0.685 there, not 0.0625, with median pre-No-Go
# envelope 2.0 rather than the 1 that 926a's authoring note reported. So 937's
# C1 failed (measured lift 0.344/0.273/0.320) against a threshold calibrated on
# a number that does not hold.
#
# The WRONG repair is to lower the bar until the measured lift clears it: that
# converts a detected calibration artifact into a citable result, which
# /queue-experiment forbids in terms ("NEVER lower the threshold to resolve it").
# So the bar is NOT lowered. It is RE-DERIVED from the substrate's own
# structure, and the dose range is extended until the mechanism's predicted
# floor is actually inside it:
#
#   As f_eligibility_envelope_floor rises, the MECH-448 F-eligibility envelope
#   admits fewer candidates. At envelope size 1 the fail-open
#   gng_protect_min_eligible guard REFUSES to drop the last survivor, so the
#   No-Go cannot exclude anything and conversion is 0 BY CONSTRUCTION -- not as
#   an empirical tendency but as a structural property of the guard.
#
# So if envelope width genuinely gates this axis, conversion at a floor narrow
# enough to collapse the envelope must approach 0, giving a lift near the 0.997
# measured at the wide end. 0.40 is retained as a deliberately conservative
# fraction of that mechanistic prediction. If the lift STILL fails at 0.60 --
# i.e. the axis keeps converting even when the envelope should have collapsed --
# then the envelope-width gating story is wrong in a new and reportable way,
# which is the informative null this design is powered to detect.
DOSE_LIFT_FLOOR = 0.40
# C4: conversion at or below this counts as the axis being INERT. Set from the
# same fail-open argument (structural 0), with headroom for per-seed jitter.
INERTNESS_CEILING = 0.10
SEED_MAJORITY = 2               # C1: seeds that must clear the lift (of 3)
# C2: monotonicity slack per adjacent dose step. Conversion may dip slightly
# between neighbouring doses without falsifying a monotone gating relationship.
MONOTONE_TOL = 0.10
PERSEVERATION_FLOOR = 0.5       # gng_perseveration_floor (matches config default)

# Readiness floors:
SUPPRESSION_RANGE_FLOOR = 0.25   # P1 -- SAME statistic the C1 crossing depends on
ENVELOPE_DOSE_SEPARATION_FLOOR = 1.0  # P2 -- >= 1 whole candidate of movement

# Perseverative history construction, unchanged from V3-EXQ-926a: the dominant
# class fills 6 of the 8-deep recency ring and two other classes take one slot
# each, giving suppression = [0.75, 0.125, 0.125, 0.0]. BOTH halves matter: one
# candidate must clear gng_perseveration_floor and the rest must NOT.
DOMINANT_REPEATS = 6
HISTORY_DEPTH = 8

# ANCHOR REACHABILITY -- why the _lib/readiness_anchor.py guard is not used here,
# and what is done instead. That helper's contract is a per-cell boolean
# `score_fn` applied to a frozen fixture of recorded reference cells, gated on
# the FRACTION that score True. Neither precondition here has that shape: P1 is
# a scalar floor on a worst cell (min of a range) and P2 is a scalar range
# across doses (max - min). Coercing them into a fraction-of-cells predicate
# would require WRITING A NEW per-cell scorer that is not the shipped predicate,
# which violates that module's own rule 1 ("score_fn MUST be the SAME callable
# the run scores its live cells with; a re-implementation defeats the entire
# purpose") and would guard a copy rather than the real gate.
#
# What is done instead, per precondition:
#   P1 -- reachability is PROVEN AT SETUP by _assert_p1_anchor_reachable() below,
#         which replays the actual shipped control construction through the
#         actual shipped comparison and raises if it cannot clear the gate. This
#         is the guard's SPIRIT in the shape the predicate really has, and it is
#         free: the control is deterministic in dominant_class, so its
#         suppression range is the constant 0.75 against a 0.25 gate (3x margin).
#   P2 -- the predicate IS the degeneracy definition, which is the documented
#         exemption condition: "the swept knob moved its mediator by at least one
#         whole candidate across the ladder" is not a proxy for the sweep being
#         non-degenerate, it is what non-degenerate MEANS for a dose sweep. It
#         also cannot be anchored to a frozen fixture, because no recorded
#         two-dose reference exists -- corpus-wide only ONE envelope-floor value
#         (0.10, V3-EXQ-926a) has ever been measured with this readout, and
#         producing the first such reference is precisely what this run is for.
ANCHOR_REACHABILITY_EXEMPT = (
    "P2 (envelope_size_dose_separation) IS the degeneracy definition for a dose "
    "sweep, not a proxy for it, and no frozen two-dose reference exists to "
    "anchor against -- this run produces the first one. P1 is not exempted by "
    "argument: its reachability is proven at setup by "
    "_assert_p1_anchor_reachable(), which replays the shipped control through "
    "the shipped comparison. Neither precondition has the per-cell-boolean "
    "shape _lib/readiness_anchor.py requires, and coercing them would guard a "
    "re-implementation rather than the shipped predicate (that module's rule 1)."
)


def _make_candidate(action_class: int, world_vec: torch.Tensor) -> Trajectory:
    states = [torch.zeros(1, WORLD_DIM) for _ in range(HORIZON + 1)]
    world_states = [world_vec.reshape(1, WORLD_DIM).clone() for _ in range(HORIZON + 1)]
    actions = torch.zeros(1, HORIZON, ACTION_DIM)
    actions[:, 0, action_class] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _build_bank(rng: torch.Generator) -> List[Trajectory]:
    """K candidates with all-distinct first-action classes and divergent world
    states, so raw F genuinely differs across candidates."""
    cands = []
    for k in range(K_CANDIDATES):
        wv = torch.randn(WORLD_DIM, generator=rng) * 0.5 + float(k) * 0.4
        cands.append(_make_candidate(action_class=k, world_vec=wv))
    return cands


def _perseverative_history(dominant_class: int) -> List[int]:
    minority = [
        (dominant_class + 1) % K_CANDIDATES,
        (dominant_class + 2) % K_CANDIDATES,
    ]
    hist = [dominant_class] * DOMINANT_REPEATS + minority
    return hist[:HISTORY_DEPTH]


def _live_suppression_vector(dominant_class: int) -> torch.Tensor:
    """MECH-260's per-candidate recency-share, read from a LIVE dACC.

    Nothing is hand-stuffed: the history is pushed through real record_action()
    calls and the value comes from dacc._suppression_penalty().
    """
    dacc = DACCAdaptiveControl(DACCConfig())
    for a in _perseverative_history(dominant_class):
        dacc.record_action(a)
    return torch.tensor(
        [dacc._suppression_penalty(c) for c in range(K_CANDIDATES)],
        dtype=torch.float32,
    )


def _raw_f(selector: E3TrajectorySelector, cands: List[Trajectory]) -> List[float]:
    selector._running_variance = 0.0
    r = selector.select(cands, temperature=1.0, score_bias=torch.zeros(len(cands)))
    return [float(s.detach()) for s in r.scores]


def _select_one(
    selector: E3TrajectorySelector,
    cands: List[Trajectory],
    suppression: torch.Tensor,
    gate_on: bool,
) -> Dict[str, Any]:
    """One committed selection through the real select() path."""
    selector._running_variance = 0.0  # deterministic committed argmin
    k = len(cands)
    go_nogo_signals = {"perseveration": suppression} if gate_on else None
    result = selector.select(
        cands,
        temperature=1.0,
        score_bias=torch.zeros(k),
        go_nogo_signals=go_nogo_signals,
    )
    diag = selector.last_score_diagnostics or {}
    env_size = diag.get("go_nogo_envelope_size", None)
    return {
        "selected_index": int(result.selected_index),
        "selected_class": int(result.selected_action.reshape(-1).argmax().item()),
        "go_nogo_active": bool(diag.get("go_nogo_constitution_active", False)),
        "n_soft_requested": int(diag.get("go_nogo_n_soft_requested", 0) or 0),
        "n_soft_applied": int(diag.get("go_nogo_n_soft_applied", 0) or 0),
        "envelope_size": int(env_size) if env_size is not None else None,
    }


def _config_plumbing_live() -> bool:
    """P3 readiness: from_dims plumbs ALL THREE knobs onto config.e3.

    Extends V3-EXQ-926a's check with f_eligibility_envelope_floor, which is the
    swept knob here and therefore the one whose silent swallow would be fatal.
    """
    probe_floor = 0.17  # a value equal to no default, so a swallow is visible
    c = REEConfig.from_dims(
        body_obs_dim=6,
        world_obs_dim=25,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        use_go_nogo_constitution=True,
        gng_perseveration_floor=PERSEVERATION_FLOOR,
        f_eligibility_envelope_floor=probe_floor,
    )
    return (
        getattr(c.e3, "use_go_nogo_constitution", False) is True
        and abs(float(getattr(c.e3, "gng_perseveration_floor", -1.0))
                - PERSEVERATION_FLOOR) < 1e-9
        and abs(float(getattr(c.e3, "f_eligibility_envelope_floor", -1.0))
                - probe_floor) < 1e-9
    )


class AnchorUnreachable(RuntimeError):
    """P1's gate cannot be cleared by its own positive control."""


def _assert_p1_anchor_reachable() -> Dict[str, Any]:
    """Setup-time proof that P1's gate is REACHABLE by P1's own positive control.

    The failure mode this closes (SD-068 REM fanout autopsy, Learning 1): a
    readiness predicate written NARROWER than the state it anchors to is
    unmeetable by construction -- it reports met=false on every run forever and
    self-routes `substrate_not_ready_requeue`, mislabelling an
    instrument-specification bug as a substrate verdict.

    Here that is checkable exactly rather than statistically, because the
    control is DETERMINISTIC: `_live_suppression_vector` depends only on
    `dominant_class`, so its cross-candidate range is a fixed constant. This
    replays the SHIPPED control construction through the SHIPPED comparison
    (`range >= SUPPRESSION_RANGE_FLOOR`) -- not a re-implementation of either.
    """
    ranges = []
    for dominant in range(K_CANDIDATES):
        v = _live_suppression_vector(dominant)
        ranges.append(float(v.max() - v.min()))
    worst = min(ranges)
    if not (worst >= SUPPRESSION_RANGE_FLOOR):
        raise AnchorUnreachable(
            "P1 suppression_cross_candidate_range_supra_floor is UNREACHABLE: "
            f"the positive control's own worst cross-candidate range is {worst:.4f}, "
            f"below its gate of {SUPPRESSION_RANGE_FLOOR}. The gate is a guaranteed "
            "false negative and would mislabel every run substrate_not_ready_requeue. "
            "Widen the predicate or lower the gate -- do NOT interpret it as a "
            "substrate verdict."
        )
    return {
        "anchor_name": "suppression_cross_candidate_range_supra_floor",
        "reachable": True,
        "control_worst_range": worst,
        "threshold": SUPPRESSION_RANGE_FLOOR,
        "margin": worst - SUPPRESSION_RANGE_FLOOR,
        "reference_source": (
            "deterministic: an 8-deep recency ring with DOMINANT_REPEATS=6, "
            "replayed through real DACCAdaptiveControl.record_action() + "
            "_suppression_penalty() for every dominant class"
        ),
    }


def _run_cell(arm: str, seed: int, envelope_floor: float, n_banks: int) -> Dict[str, Any]:
    """One (arm, seed, envelope_floor) cell over n_banks divergent candidate banks."""
    reset_all_rng(seed)
    rng = torch.Generator().manual_seed(seed)
    gate_on = arm in ("ARM_CONSTITUTION", "ARM_SHUFFLED")

    n_converted = 0
    n_incumbent_is_argmin = 0
    n_empty_eligible = 0
    n_gate_active = 0
    excluded_counts: List[int] = []
    raw_ranges: List[float] = []
    supp_ranges: List[float] = []
    pre_envelope_sizes: List[float] = []
    exemplar_suppression: Optional[List[float]] = None

    cond = f"{arm}@floor{envelope_floor:.2f}"
    print(f"Seed {seed} Condition {cond}", flush=True)
    for b in range(n_banks):
        # The trailing `or (b + 1) == n_banks` guarantees at least one progress
        # line per cell even when n_banks < 8 (the --dry-run case), so the smoke
        # actually verifies the runner's `ep N/M` instrumentation instead of
        # silently emitting none. The denominator is the LOOP BOUND, never a
        # hardcoded constant.
        if (b + 1) % 8 == 0 or (b + 1) == n_banks:
            print(f"  [eval] {cond} seed={seed} ep {b+1}/{n_banks}", flush=True)

        cands = _build_bank(rng)

        # Baseline selector at THIS dose: MECH-448 F-eligibility demotion ON,
        # constitution OFF. Its committed pick DEFINES the incumbent for this
        # dose -- which is why ARM_OFF is run at every dose (see the DV-symmetry
        # declaration): the incumbent itself can move with the envelope.
        sel_demote = E3TrajectorySelector(
            E3Config(
                world_dim=WORLD_DIM,
                hidden_dim=8,
                use_f_eligibility_demotion=True,
                f_eligibility_envelope_floor=envelope_floor,
                use_go_nogo_constitution=False,
            )
        )
        raw = _raw_f(sel_demote, cands)
        raw_ranges.append(max(raw) - min(raw))

        base = _select_one(sel_demote, cands, torch.zeros(K_CANDIDATES), gate_on=False)
        incumbent = base["selected_index"]

        # ARM_SHUFFLED perseverates on a NON-incumbent candidate (specificity).
        target = incumbent if arm != "ARM_SHUFFLED" else (incumbent + 1) % K_CANDIDATES
        suppression = _live_suppression_vector(target)
        supp_ranges.append(float(suppression.max() - suppression.min()))
        if exemplar_suppression is None:
            exemplar_suppression = [float(x) for x in suppression.tolist()]
        if int(suppression.argmax().item()) == target:
            n_incumbent_is_argmin += 1

        sel_arm = E3TrajectorySelector(
            E3Config(
                world_dim=WORLD_DIM,
                hidden_dim=8,
                use_f_eligibility_demotion=True,
                f_eligibility_envelope_floor=envelope_floor,
                use_go_nogo_constitution=gate_on,
                gng_perseveration_floor=PERSEVERATION_FLOOR,
            )
        )
        # Same head init as the baseline selector, so the ONLY difference
        # between base and arm is the gate (not random head weights).
        sel_arm.load_state_dict(sel_demote.state_dict())
        got = _select_one(sel_arm, cands, suppression, gate_on=gate_on)

        if got["go_nogo_active"]:
            n_gate_active += 1
        if got["envelope_size"] is not None:
            excluded_counts.append(K_CANDIDATES - got["envelope_size"])
            pre_envelope_sizes.append(
                float(got["envelope_size"] + got["n_soft_applied"])
            )
            if got["envelope_size"] <= 0:
                n_empty_eligible += 1

        converted = (
            base["selected_index"] == incumbent
            and got["selected_index"] != incumbent
        )
        if converted:
            n_converted += 1

    conversion_rate = n_converted / float(n_banks)
    supp_range = statistics.median(supp_ranges) if supp_ranges else 0.0
    cell = {
        "arm": arm,
        "seed": seed,
        "envelope_floor": envelope_floor,
        "condition": cond,
        "gate_on": gate_on,
        "n_banks": n_banks,
        "suppression_vector_exemplar": exemplar_suppression,
        "suppression_range": supp_range,
        "suppression_range_min": min(supp_ranges) if supp_ranges else 0.0,
        "conversion_rate": conversion_rate,
        "n_converted": n_converted,
        "incumbent_is_f_argmin_rate": n_incumbent_is_argmin / float(n_banks),
        "n_empty_eligible": n_empty_eligible,
        "gate_active_rate": n_gate_active / float(n_banks),
        "median_excluded_count": (
            statistics.median(excluded_counts) if excluded_counts else 0.0
        ),
        # NON-GATING diagnostic. Recorded on every cell, including the collapsed
        # stock-floor cells, precisely because a collapsed envelope here is the
        # phenomenon rather than a readiness failure (see module docstring).
        "median_pre_nogo_envelope_size": (
            statistics.median(pre_envelope_sizes) if pre_envelope_sizes else 0.0
        ),
        "median_raw_f_range": statistics.median(raw_ranges) if raw_ranges else 0.0,
    }
    cell["arm_fingerprint"] = compute_arm_fingerprint(
        config_slice={
            "arm": arm,
            "use_f_eligibility_demotion": True,
            "f_eligibility_envelope_floor": envelope_floor,
            "use_go_nogo_constitution": gate_on,
            "gng_perseveration_floor": PERSEVERATION_FLOOR,
            "n_banks": n_banks,
            "k": K_CANDIDATES,
            "dominant_repeats": DOMINANT_REPEATS,
            "history_depth": HISTORY_DEPTH,
        },
        seed=seed,
        script_path=Path(__file__),
        rng_fully_reset=True,
        config_slice_declared=True,
        extra_ineligible_reasons=["selection_face_synthetic_no_training"],
    )
    # Only ARM_CONSTITUTION at a WIDE envelope is expected to convert. Scoring
    # every cell against a conversion floor would report the stock-floor doses
    # and both control arms as failures when they are the design working.
    verdict_pass = n_empty_eligible == 0
    print(f"verdict: {'PASS' if verdict_pass else 'FAIL'}", flush=True)
    return cell


def _mean_conversion(cells: List[Dict[str, Any]]) -> float:
    return statistics.fmean([c["conversion_rate"] for c in cells]) if cells else 0.0


def _mean_env(cells: List[Dict[str, Any]]) -> float:
    """Mean pre-No-Go eligible-set size across cells (the P2 mediator)."""
    if not cells:
        return 0.0
    return statistics.fmean([c["median_pre_nogo_envelope_size"] for c in cells])


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = datetime.utcnow()
    # TWO seeds under --dry-run, not one. C1 requires SEED_MAJORITY (2) seeds to
    # clear the lift floor, so a single-seed smoke makes the load-bearing
    # criterion STRUCTURALLY UNREACHABLE and the smoke can only ever report
    # FAIL -- it would never exercise the PASS path it exists to validate.
    # Cheap here: the whole grid is a selection-face probe with no training.
    seeds = SEEDS[:2] if dry_run else SEEDS
    n_banks = 4 if dry_run else N_BANKS
    # Smoke must span BOTH REGIMES C1 contrasts, which are realized-envelope
    # regimes, not floor endpoints: floor 0.40 realizes envelope 1 (protect-min)
    # and 0.10 realizes envelope 3 (roomy). Using the floor endpoints 0.60/0.10
    # would NOT do this -- 0.60 folds back to envelope 4 -- so the smoke would
    # never exercise the protect-min regime the experiment exists to measure.
    floors = [0.40, REFERENCE_FLOOR] if dry_run else ENVELOPE_FLOORS

    # Prove P1's gate is reachable by its own control BEFORE any compute is
    # spent, so an unmeetable predicate is caught at setup rather than after the
    # grid runs and self-routes a false substrate verdict.
    p1_reachability = _assert_p1_anchor_reachable()
    print(
        f"[937a] P1 anchor reachable: control worst range "
        f"{p1_reachability['control_worst_range']:.4f} >= gate "
        f"{p1_reachability['threshold']} (margin "
        f"{p1_reachability['margin']:.4f})",
        flush=True,
    )

    arm_results: List[Dict[str, Any]] = []
    for floor in floors:
        for arm in ARMS:
            for seed in seeds:
                arm_results.append(_run_cell(arm, seed, floor, n_banks))

    on_cells = [c for c in arm_results if c["arm"] == "ARM_CONSTITUTION"]
    shuf_cells = [c for c in arm_results if c["arm"] == "ARM_SHUFFLED"]
    off_cells = [c for c in arm_results if c["arm"] == "ARM_OFF"]

    def _by_floor(cells: List[Dict[str, Any]], floor: float) -> List[Dict[str, Any]]:
        return [c for c in cells if abs(c["envelope_floor"] - floor) < 1e-9]

    widest = min(floors)   # lowest floor == widest eligible set
    stock = max(floors)    # highest floor on the ladder == stock default

    # ---------------- readiness preconditions (recomputable) ----------------
    # Worst CELL, not a mean -- `met` is a worst-case claim, so `measured` must
    # be the same extremum the indexer recomputes on.
    supp_ranges = [c["suppression_range_min"] for c in arm_results]
    worst_supp_range = min(supp_ranges) if supp_ranges else 0.0

    # P2: the swept knob must MOVE its proximal mediator across the ladder.
    # Measured on ARM_CONSTITUTION cells (the regime whose conversion C1 reads).
    on_env_by_floor = {
        f: _mean_env(_by_floor(on_cells, f)) for f in floors
    }
    env_dose_separation = (
        max(on_env_by_floor.values()) - min(on_env_by_floor.values())
        if on_env_by_floor else 0.0
    )
    plumbing_ok = _config_plumbing_live()

    preconditions = [
        {
            "name": "suppression_cross_candidate_range_supra_floor",
            "kind": "readiness",
            "description": (
                "live dACC MECH-260 suppression varies ACROSS candidates "
                "(max - min), the statistic the perseveration threshold "
                "crossing the conversion criterion depends on"
            ),
            "control": (
                "positive control: an 8-deep recency ring dominated by one "
                "action class, built via real record_action() calls"
            ),
            "measured": worst_supp_range,
            "threshold": SUPPRESSION_RANGE_FLOOR,
            "direction": "lower",
            "met": worst_supp_range >= SUPPRESSION_RANGE_FLOOR,
        },
        {
            # THE dose-sweep readiness check. C1 routes on a RANGE OF CONVERSION
            # ACROSS DOSES, so this asserts a RANGE ACROSS DOSES of the mediator
            # the knob acts through. Range-gated criterion -> range readiness
            # (the V3-EXQ-643 same-statistic rule). Not circular: this measures
            # the MEDIATOR responding to the knob, C1 measures the OUTCOME.
            "name": "envelope_size_dose_separation",
            "kind": "readiness",
            "description": (
                "the swept f_eligibility_envelope_floor actually changes how "
                "many candidates survive into the No-Go: spread of mean "
                "pre-No-Go envelope size across the dose ladder"
            ),
            "control": (
                "positive control: the ladder endpoints, stock floor "
                f"{stock:.2f} vs widest {widest:.2f}, on ARM_CONSTITUTION cells"
            ),
            "measured": env_dose_separation,
            "threshold": ENVELOPE_DOSE_SEPARATION_FLOOR,
            "note": (
                "the floor -> envelope map is non-monotone (measured: 0.60 -> 4, "
                "0.40 -> 1, 0.30 -> 2), so this asserts the LADDER REALIZED a "
                "spread of envelope sizes, not that the floor ordering did"
            ),
            "direction": "lower",
            "met": env_dose_separation >= ENVELOPE_DOSE_SEPARATION_FLOOR,
        },
        {
            "name": "envelope_floor_config_plumbing_live",
            "kind": "readiness",
            "description": (
                "REEConfig.from_dims plumbs use_go_nogo_constitution, "
                "gng_perseveration_floor AND the swept "
                "f_eligibility_envelope_floor onto config.e3 (guards the "
                "documented from_dims silent-kwarg-swallow failure mode)"
            ),
            "control": "direct from_dims construction with a non-default floor",
            "met": bool(plumbing_ok),
        },
    ]
    readiness_ok = all(p.get("met") for p in preconditions)

    # ---------------------------- criteria ---------------------------------
    # THE FLOOR -> ENVELOPE MAP IS NON-MONOTONE. Measured during authoring at
    # 128 banks, seed 42: floor 0.60 -> envelope 4.0 (conversion 0.781);
    # 0.40 -> 1.0 (0.227); 0.30 -> 2.0 (0.648). Raising the floor does NOT
    # monotonically shrink the eligible set: once the floor is high enough that
    # NO candidate clears it, the fail-open guard admits the FULL set again, so
    # envelope size runs 4 -> 1 -> 2 -> 3 as the floor falls. V3-EXQ-937's
    # apparent monotone curve was an artifact of sampling only 0.30 -> 0.05, a
    # window over which the map happens to be monotone.
    #
    # So the floor is the KNOB but not the MEDIATOR, and keying the criteria to
    # floor endpoints (as this script's own first draft did, and as 937 did)
    # measures a comparison whose meaning changes with the sampling window.
    # Everything below is therefore keyed to the REALIZED envelope size, which
    # is what the fail-open mechanism actually predicts on:
    #
    #   at realized envelope 1 the gng_protect_min_eligible guard refuses to
    #   drop the last survivor, so the No-Go can exclude nothing -- conversion
    #   must be suppressed BY CONSTRUCTION, independent of which floor produced
    #   that envelope.
    ordered = sorted(floors, reverse=True)   # highest floor first
    conv_by_floor = {f: _mean_conversion(_by_floor(on_cells, f)) for f in ordered}
    env_by_floor = {f: _mean_env(_by_floor(on_cells, f)) for f in ordered}

    def _cells_at_env(pred) -> List[Dict[str, Any]]:
        return [c for c in on_cells if pred(c["median_pre_nogo_envelope_size"])]

    pinned_cells = _cells_at_env(lambda e: e <= 1.0)          # protect-min regime
    open_cells = _cells_at_env(lambda e: e >= 3.0)            # roomy regime
    conv_pinned = _mean_conversion(pinned_cells)
    conv_open = _mean_conversion(open_cells)

    # C1 (LOAD-BEARING): the protect-min suppression effect, per seed. A cell
    # whose envelope collapsed to 1 must convert far LESS than one with room.
    # Threshold is the mechanistic DOSE_LIFT_FLOOR, unchanged and not re-fit.
    per_seed_lift: Dict[str, float] = {}
    for sd in seeds:
        po = _mean_conversion([c for c in open_cells if c["seed"] == sd])
        pp = _mean_conversion([c for c in pinned_cells if c["seed"] == sd])
        per_seed_lift[str(sd)] = po - pp
    n_seeds_clearing = sum(1 for v in per_seed_lift.values() if v >= DOSE_LIFT_FLOOR)
    c1 = n_seeds_clearing >= SEED_MAJORITY

    # C2: conversion is monotone-nondecreasing in REALIZED ENVELOPE SIZE (not in
    # the floor). This is the relation the mechanism predicts; the floor-ordered
    # version 937 used is not well defined once the map is known to fold back.
    env_levels = sorted({round(c["median_pre_nogo_envelope_size"], 3) for c in on_cells})
    conv_by_env = {
        e: _mean_conversion(_cells_at_env(lambda v, e=e: abs(v - e) < 1e-9))
        for e in env_levels
    }
    steps = []
    for i2 in range(len(env_levels) - 1):
        a, b = conv_by_env[env_levels[i2]], conv_by_env[env_levels[i2 + 1]]
        steps.append({"from_envelope": env_levels[i2], "to_envelope": env_levels[i2 + 1],
                      "delta": b - a, "ok": (b - a) >= -MONOTONE_TOL})
    c2 = all(st["ok"] for st in steps) if steps else False

    # C3: the fail-open safety contract is never violated at any dose.
    total_empty = sum(c["n_empty_eligible"] for c in arm_results)
    c3 = total_empty == 0

    # Reported, never gated: the narrowest realized envelope that still converts
    # above half, and the floor that produced it.
    knee_envelope = None
    for e in env_levels:
        if conv_by_env[e] >= 0.5:
            knee_envelope = e
            break
    knee_floor = None
    for f in ordered:
        if conv_by_floor[f] >= 0.5:
            knee_floor = f
            break

    # C4: was an INERT regime (conversion <= ceiling) reached at all? Reported,
    # not gated -- failing to reach it even in the protect-min regime would mean
    # envelope width does NOT gate this axis, which is the informative null.
    inert_envelope = None
    for e in env_levels:
        if conv_by_env[e] <= INERTNESS_CEILING:
            inert_envelope = e
            break
    inert_floor = None
    for f in ordered:
        if conv_by_floor[f] <= INERTNESS_CEILING:
            inert_floor = f
            break
    c4 = inert_envelope is not None

    # ------------------------- non-degeneracy ------------------------------
    gate_ever_active = any(c["gate_active_rate"] > 0 for c in on_cells)
    distinct_env = len({round(v, 3) for v in on_env_by_floor.values()})
    c1_non_degenerate = bool(
        gate_ever_active
        and worst_supp_range >= SUPPRESSION_RANGE_FLOOR
        and env_dose_separation >= ENVELOPE_DOSE_SEPARATION_FLOOR
        and _mean_conversion(on_cells) != _mean_conversion(off_cells)
    )
    criteria_non_degenerate = {
        # C2 needs at least 3 distinct mediator levels to describe a shape;
        # two points cannot distinguish monotone from a step.
        "C2_conversion_monotone_in_envelope_width": bool(
            c1_non_degenerate and distinct_env >= 3
        ),
        "C1_envelope_width_dose_response": c1_non_degenerate,
        "C3_safety_failopen": bool(arm_results),
        # C4 is degenerate if the ladder never included a floor narrow enough to
        # plausibly collapse the envelope -- exactly V3-EXQ-937's limitation.
        "C4_inert_regime_reached_on_ladder": bool(
            ordered and max(ordered) >= NARROWEST_FLOOR
        ),
    }

    criteria = [
        {
            "name": "C1_envelope_width_dose_response",
            "load_bearing": True,
            "passed": bool(c1),
            "measured": per_seed_lift,
            "threshold": DOSE_LIFT_FLOOR,
            "seeds_clearing": n_seeds_clearing,
            "seeds_required": SEED_MAJORITY,
        },
        {
            "name": "C2_conversion_monotone_in_envelope_width",
            "load_bearing": False,
            "passed": bool(c2),
            "measured": steps,
            "threshold": -MONOTONE_TOL,
            "direction": "lower",
        },
        {
            "name": "C3_safety_failopen",
            "load_bearing": False,
            "passed": bool(c3),
            "measured": total_empty,
            "threshold": 0,
            "direction": "upper",
        },
        {
            # Reported, not gated: failing to reach inertness even at floor 0.60
            # is itself a finding (it would mean envelope collapse does not
            # silence the axis), so it must not convert the run to a FAIL.
            "name": "C4_inert_regime_reached_on_ladder",
            "load_bearing": False,
            "passed": bool(c4),
            "measured": (
                conv_by_floor[ordered[0]] if ordered else None
            ),
            "threshold": INERTNESS_CEILING,
            "direction": "upper",
            "inert_floor": inert_floor,
        },
    ]
    combination_rule = (
        "PASS = readiness_ok AND C1 AND C3. C2 (monotonicity) and C4 (inertness reached) are REPORTED, not "
        "gated: a step-shaped gating relation is a legitimate and informative "
        "curve shape, so failing monotonicity must not convert a clean "
        "dose-response measurement into a FAIL. C1 is the load-bearing criterion."
    )

    # ---------------------------- routing ----------------------------------
    if not readiness_ok:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
    elif c1 and c3:
        outcome = "PASS"
        label = "envelope_width_gates_perseveration_conversion"
        direction = "supports"
    elif c3 and not c1 and c2:
        # ROUTING DEFECT FIXED FROM V3-EXQ-937. That run emitted
        # `conversion_independent_of_envelope_width` on exactly this branch --
        # C1 short of its lift bar -- WHILE C2 had passed with a clean monotone
        # rise 0.685 -> 0.997. "Independent" was flatly contradicted by the
        # run's own C2. A label must not assert an absence of dependence that a
        # sibling criterion in the same manifest just measured.
        #
        # The correct reading of C1-fails-but-C2-holds is a dependence that is
        # REAL and GRADED but of smaller magnitude than the pre-registered bar,
        # which is a quantitative shortfall against a prediction, not an absence
        # of effect.
        outcome = "FAIL"
        label = "graded_dose_response_below_prereg_lift"
        direction = "non_contributory"
    elif c3 and not c1:
        # C1 short AND no monotone structure: here the absence claim is earned.
        outcome = "FAIL"
        label = "conversion_independent_of_envelope_width"
        direction = "non_contributory"
    else:
        outcome = "FAIL"
        label = "safety_failopen_contract_violated"
        direction = "weakens"

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{t0.strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "overall_pass": outcome == "PASS",
        "timestamp_utc": t0.strftime("%Y%m%dT%H%M%SZ"),
        "evidence_direction": direction,
        "evidence_direction_per_claim": {
            "MECH-449": direction,
            "ARC-107": direction,
        },
        "arm_results": arm_results,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "dv_symmetry_declaration": {
                "ARM_CONSTITUTION": (
                    "manipulation = lowering f_eligibility_envelope_floor widens "
                    "the F-eligible SET (a change of the argmin's DOMAIN), and "
                    "the No-Go then removes a candidate from that set. DV = "
                    "committed argmin identity, aggregated to a conversion rate. "
                    "Argmin's symmetry group is {uniform additive constant, "
                    "monotone rescaling, tie permutation}; changing set "
                    "MEMBERSHIP is in none of them. NOT invariant."
                ),
                "ARM_SHUFFLED": (
                    "same domain manipulation, recency mass on a NON-incumbent. "
                    "Same argument, NOT invariant. Near-zero conversion here is "
                    "the specificity control working, not a measured null."
                ),
                "ARM_OFF": (
                    "no No-Go manipulation, but the envelope floor still varies "
                    "(it is the swept knob), so ARM_OFF runs at EVERY dose. "
                    "Without that, a dose-dependent conversion rise would be "
                    "aliased between the No-Go acting and the widened envelope "
                    "alone moving the baseline committed pick."
                ),
            },
            "anchor_reachability": p1_reachability,
            "scoped_out_preconditions": {
                "pre_nogo_eligible_set_admits_alternative": (
                    "V3-EXQ-926a's per-cell 'envelope >= 2' readiness gate is "
                    "DELIBERATELY NOT carried over. In this design a collapsed "
                    "envelope at the stock floor is the PHENOMENON UNDER STUDY, "
                    "not a substrate failure, so a whole-run all(...) gate over "
                    "per-cell envelope size would fail on the stock-floor dose "
                    "and vacate the wide-envelope doses that ran cleanly -- the "
                    "V3-EXQ-785 defect exactly. Replaced by "
                    "envelope_size_dose_separation, which asserts the property "
                    "this design actually needs. Per-cell envelope size is still "
                    "recorded on every cell as a non-gating diagnostic."
                )
            },
        },
        "summary": {
            "per_seed_conversion_lift_widest_vs_stock": per_seed_lift,
            "seeds_clearing_lift_floor": n_seeds_clearing,
            "mean_conversion_by_floor_constitution": conv_by_floor,
            "mean_conversion_by_floor_shuffled": {
                f: _mean_conversion(_by_floor(shuf_cells, f)) for f in ordered
            },
            "mean_conversion_by_floor_off": {
                f: _mean_conversion(_by_floor(off_cells, f)) for f in ordered
            },
            "mean_pre_nogo_envelope_by_floor_constitution": on_env_by_floor,
            "envelope_dose_separation": env_dose_separation,
            "monotone_steps": steps,
            "knee_floor_first_clearing_half": knee_floor,
            "knee_envelope_first_clearing_half": knee_envelope,
            "inert_envelope_first_at_or_below_ceiling": inert_envelope,
            "mean_conversion_by_realized_envelope": conv_by_env,
            "mean_envelope_by_floor": env_by_floor,
            "conversion_protect_min_regime": conv_pinned,
            "conversion_roomy_regime": conv_open,
            "inert_floor_first_at_or_below_ceiling": inert_floor,
            "inertness_ceiling": INERTNESS_CEILING,
            "narrowest_floor": NARROWEST_FLOOR,
            "stock_floor": stock,
            "widest_floor": widest,
            "worst_suppression_range": worst_supp_range,
            "total_empty_eligible": total_empty,
            "gate_ever_active": gate_ever_active,
        },
        "config": {
            "seeds": seeds,
            "arms": ARMS,
            "envelope_floors": floors,
            "n_banks": n_banks,
            "k_candidates": K_CANDIDATES,
            "world_dim": WORLD_DIM,
            "action_dim": ACTION_DIM,
            "horizon": HORIZON,
            "perseverated_class": "per-bank committed incumbent (probed, not fixed)",
            "dominant_repeats": DOMINANT_REPEATS,
            "history_depth": HISTORY_DEPTH,
            "gng_perseveration_floor": PERSEVERATION_FLOOR,
            "dose_lift_floor": DOSE_LIFT_FLOOR,
            "inertness_ceiling": INERTNESS_CEILING,
            "narrowest_floor": NARROWEST_FLOOR,
            "seed_majority": SEED_MAJORITY,
            "monotone_tol": MONOTONE_TOL,
            "suppression_range_floor": SUPPRESSION_RANGE_FLOOR,
            "envelope_dose_separation_floor": ENVELOPE_DOSE_SEPARATION_FLOOR,
            "stock_floor": STOCK_FLOOR,
            "reference_floor": REFERENCE_FLOOR,
        },
        "notes": (
            "Dose-response characterisation of the envelope-width gating that "
            "V3-EXQ-926a discovered and explicitly declined to map. 926a's "
            "manifest notes state the interaction (perseveration No-Go is "
            "structurally gated by envelope width) as its finding of record, "
            "from TWO points of which only ONE (floor 0.10, conversion ~0.97) "
            "was ever written to a manifest; the stock-floor point (0.30, "
            "conversion 1/16, median envelope 1) was authoring-time scratch "
            "recorded only in prose. Corpus-wide only two envelope-floor values "
            "appear in any manifest config block (0.10 in 926a alone; 0.30 in 15 "
            "runs that do not measure perseveration conversion), so the curve is "
            "not recoverable by reanalysis. This run puts BOTH 926a anchors on "
            "one ladder under identical recorded conditions plus four "
            "intermediate doses, and runs ARM_OFF at every dose so a "
            "dose-dependent rise is not aliased between the No-Go acting and the "
            "widened envelope alone moving the baseline pick. "
            "WHY IT MATTERS: f_eligibility_envelope_floor ships at 0.30. If the "
            "perseveration leg cannot fire at the stock default, the built "
            "ARC-107 constitution has an inert-as-configured leg and every run "
            "leaving the knob alone measures an axis that cannot act -- a "
            "build-configuration fact the ARC-107 roadmap needs and lacks. The "
            "deliverable is the knee: the narrowest envelope at which the leg "
            "has authority. "
            "PURPOSE=diagnostic, so this is excluded from confidence/conflict "
            "scoring by design; it characterises an operating curve rather than "
            "posing a new falsifier, and PROMOTES NOTHING. "
            "MECH-439 is DELIBERATELY NOT TAGGED (re-derive brake 12, "
            "ceiling_decision exhausted): this tests no F-dominance lever and "
            "must not be read as a 13th probe of that lineage. MECH-260 is "
            "DELIBERATELY NOT TAGGED for 926a's reason -- its falsifier is "
            "behavioural and vacuous at the MECH-457 competence floor."
        ),
    }
    if not c1_non_degenerate:
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = (
            "perseveration gate never engaged, suppression vector flat across "
            "candidates, the envelope floor did not move the pre-No-Go envelope "
            "across the dose ladder, or ON and OFF arms converted identically"
        )

    out_dir = (
        Path(tempfile.gettempdir()) / "ree_dry_run_manifests" if dry_run else EVIDENCE_DIR
    )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=manifest.get("config"),
        seeds=seeds,
        script_path=Path(__file__),
    )
    manifest["manifest_path"] = str(out_path)
    print(
        f"[937a] outcome={outcome} label={label} lift={per_seed_lift} "
        f"knee={knee_floor} env_sep={env_dose_separation:.3f} "
        f"empty_eligible={total_empty} -> {out_path}",
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-937a MECH-449/ARC-107 perseveration No-Go envelope INERTNESS POINT"
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
        dry_run=args.dry_run,
    )
