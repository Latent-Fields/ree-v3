"""V3-EXQ-937 -- MECH-449 / ARC-107 perseveration No-Go x envelope-width dose response.

EXPERIMENT_PURPOSE=diagnostic -- this MAPS an operating curve that V3-EXQ-926a
discovered and explicitly declined to characterise. It does not pose a new
falsifier for MECH-449; it converts 926a's two-point anecdote into a curve.

WHAT V3-EXQ-926a LEFT OPEN (its own stated "finding of record")
---------------------------------------------------------------
V3-EXQ-926a PASSed (supports MECH-449 + ARC-107; conversion 0.969/0.969/1.0 on
seeds 42/43/44) and its manifest `notes` state the honest scope limit verbatim:

    "... at the stock f_eligibility_envelope_floor of 0.30 the identical
     mechanism converted 1/16, because the MECH-448 envelope collapses to the
     fail-open protect-min and the axis cannot fire. That INTERACTION
     (perseveration No-Go is structurally gated by envelope width) is the
     finding of record here, and it is not documented anywhere in the MECH-449
     build notes or in V3-EXQ-689g."

So the interaction is asserted from TWO points, only ONE of which was ever
written to a manifest. The 0.30 datapoint (conversion 1/16, median envelope 1,
soft_applied 6/16) was an authoring-time scratch measurement recorded only in
prose. Corpus-wide, exactly two envelope-floor values appear in any manifest
`config` block: 0.10 (V3-EXQ-926a alone) and 0.30 (15 runs, none of which
measure perseveration conversion). There is no curve, and no recorded number
between the endpoints.

GOV-REUSE-1 (Step 2.4) -- decisive readout: per-dose ARM_CONSTITUTION
`conversion_rate` as a function of `f_eligibility_envelope_floor`. Checked every
manifest carrying `config.f_eligibility_envelope_floor` (17 runs: 485i/485j/485k,
654h/654i/654j, 689d/689e/689f/689i, 699/699b, 705/705b/706/706b, 926a). Only
926a measures the perseveration-conversion readout at all, and it measures it at
ONE dose. Not recoverable by reanalysis -> run.

WHY THIS MATTERS RATHER THAN BEING A TIDY-UP
--------------------------------------------
`f_eligibility_envelope_floor` ships with a STOCK default of 0.30. If the
perseveration leg of the MECH-449 constitution is structurally unable to fire at
the stock default -- which is what 926a's single scratch datapoint implies --
then the built ARC-107 constitution has a leg that is inert as configured, and
every future run that leaves the knob alone measures an axis that cannot act.
That is a build-configuration fact the ARC-107 roadmap needs and does not have.
The curve says where the knee is, i.e. what the knob must be set to for the leg
to have any authority.

RE-DERIVE BRAKE (Step 2.5b), stated because the neighbouring claim is braked
---------------------------------------------------------------------------
MECH-449 = 0, ARC-107 = 0 (threshold 2) -- NOT braked.
MECH-439 = 12 and `ceiling_decision: exhausted` -- heavily braked, and is
DELIBERATELY NOT TAGGED here. This experiment does not test F-dominance, does
not pose a conversion lever against F, and must not be read as a 13th probe of
that lineage. MECH-260 is likewise NOT tagged, for the reason 926a gives: its
falsifier is behavioural and vacuous at the MECH-457 competence floor.

WHY 926a's P2 IS DELIBERATELY *NOT* CARRIED OVER AS A GATE
-----------------------------------------------------------
V3-EXQ-926a's second readiness precondition was
`pre_nogo_eligible_set_admits_alternative`: worst gate-ON cell must have
median pre-No-Go envelope >= 2, on the correct reasoning that below 2 the
fail-open guard makes the axis structurally unable to fire.

Carrying that over here as a whole-run gate would be WRONG, and wrong in a
specific documented way. In THIS design a collapsed envelope at the stock floor
is the PHENOMENON UNDER STUDY, not a substrate failure -- the 0.30 dose is
expected to sit at envelope ~1 and convert ~0, and that cell is the most
informative point on the curve. A whole-run `all(...)` gate over per-cell
envelope size would therefore fail on the stock-floor dose and self-route the
ENTIRE run `substrate_not_ready_requeue`, vacating the wide-envelope doses that
ran cleanly. That is exactly the V3-EXQ-785 defect (`gate_green = all(arm...)`
letting one structurally-impossible arm silently vacate a clean, well-powered
arm), and the /queue-experiment rule derived from it says to condition every
precondition on the regimes it is meaningful for.

So the per-cell envelope floor is scoped OUT and replaced by P2 below, which
asserts the property this design actually needs: that the swept knob MOVES its
proximal mediator across the ladder. Per-cell envelope size is still recorded on
every cell as a non-gating diagnostic.

READINESS PRECONDITIONS
-----------------------
P1 `suppression_cross_candidate_range_supra_floor` -- carried over from 926a
   unchanged. C1 routes on a threshold CROSSING (some candidates over
   gng_perseveration_floor, others under), which is a property of the
   cross-candidate RANGE, not of any magnitude summary. Worst cell, not a mean.

P2 `envelope_size_dose_separation` -- the dose-sweep readiness check. C1 routes
   on a RANGE OF CONVERSION ACROSS DOSES, so the readiness check asserts a RANGE
   ACROSS DOSES of the proximal mediator the knob acts through (pre-No-Go
   envelope size). Range-gated criterion -> range readiness, per the V3-EXQ-643
   same-statistic rule. This is NOT circular: P2 measures the MEDIATOR responding
   to the knob, C1 measures the OUTCOME responding. If the floor knob does not
   change how many candidates survive into the No-Go, the sweep measured nothing
   and the run self-routes `substrate_not_ready_requeue` rather than reporting a
   flat curve as a scientific null.

P3 `envelope_floor_config_plumbing_live` -- from_dims plumbs all three knobs
   (use_go_nogo_constitution, gng_perseveration_floor, f_eligibility_envelope_floor)
   onto config.e3. Guards the documented from_dims silent-kwarg-swallow mode.

DV-SYMMETRY DECLARATION (mandatory per /queue-experiment Step 3)
----------------------------------------------------------------
  ARM_CONSTITUTION -- manipulation: lowering f_eligibility_envelope_floor widens
    the F-eligible SET, changing which candidates reach the No-Go, and the No-Go
    then removes one from that set. DV: per-bank committed argmin identity
    (aggregated to a conversion rate). The symmetry group of an argmin is
    {uniform additive constant, monotone rescaling, permutation of exact ties}.
    Changing set MEMBERSHIP is in none of those -- it does not translate or
    rescale any candidate score, it changes the domain the argmin is taken over,
    and specifically admits/deletes the argmin itself. NOT invariant.
  ARM_SHUFFLED -- same domain manipulation, recency mass placed on a
    NON-incumbent candidate. Same symmetry argument; NOT invariant. Its expected
    near-zero conversion is the specificity control working, not an effect.
  ARM_OFF -- no No-Go manipulation. The envelope floor still varies (it is the
    swept knob), which is why ARM_OFF is run at every dose rather than once: it
    separates "conversion rose because the No-Go could finally act" from
    "the committed pick moved because the envelope itself moved".

  The ARM_OFF-at-every-dose point is load-bearing and is an ADDITION to 926a,
  which ran ARM_OFF at a single floor. Without it, a dose-dependent conversion
  rise is aliased between the No-Go acting and the widened envelope alone
  changing the baseline pick.
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
EXPERIMENT_TYPE = "v3_exq_937_mech449_envelope_width_dose_response"
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
ENVELOPE_FLOORS: List[float] = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

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
# C1: conversion lift from the stock floor to the widest envelope. V3-EXQ-926a's
# two anchors give ~0.97 - 0.0625 = ~0.91; 0.40 is a deliberately conservative
# pre-registered margin, well inside that but far outside per-seed noise.
DOSE_LIFT_FLOOR = 0.40
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
    floors = [STOCK_FLOOR, REFERENCE_FLOOR] if dry_run else ENVELOPE_FLOORS

    # Prove P1's gate is reachable by its own control BEFORE any compute is
    # spent, so an unmeetable predicate is caught at setup rather than after the
    # grid runs and self-routes a false substrate verdict.
    p1_reachability = _assert_p1_anchor_reachable()
    print(
        f"[937] P1 anchor reachable: control worst range "
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
    # C1 (LOAD-BEARING): per-seed conversion lift, widest envelope vs stock.
    per_seed_lift: Dict[str, float] = {}
    for s in seeds:
        on_w = [c for c in _by_floor(on_cells, widest) if c["seed"] == s]
        on_s = [c for c in _by_floor(on_cells, stock) if c["seed"] == s]
        per_seed_lift[str(s)] = _mean_conversion(on_w) - _mean_conversion(on_s)
    n_seeds_clearing = sum(1 for v in per_seed_lift.values() if v >= DOSE_LIFT_FLOOR)
    c1 = n_seeds_clearing >= SEED_MAJORITY

    # C2: conversion is monotone-nondecreasing in envelope WIDTH across the
    # ladder, within per-step slack. Floors are swept high->low, i.e. narrow->wide.
    ordered = sorted(floors, reverse=True)   # widest LAST
    conv_by_floor = {f: _mean_conversion(_by_floor(on_cells, f)) for f in ordered}
    steps = []
    for i in range(len(ordered) - 1):
        a, b = conv_by_floor[ordered[i]], conv_by_floor[ordered[i + 1]]
        steps.append({"from_floor": ordered[i], "to_floor": ordered[i + 1],
                      "delta": b - a, "ok": (b - a) >= -MONOTONE_TOL})
    c2 = all(s["ok"] for s in steps) if steps else False

    # C3: the fail-open safety contract is never violated at any dose.
    total_empty = sum(c["n_empty_eligible"] for c in arm_results)
    c3 = total_empty == 0

    # The knee: the narrowest envelope (highest floor) at which mean conversion
    # first clears half. Reported, never gated -- its value IS the deliverable.
    knee_floor = None
    for f in ordered:
        if conv_by_floor[f] >= 0.5:
            knee_floor = f
            break

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
    ]
    combination_rule = (
        "PASS = readiness_ok AND C1 AND C3. C2 (monotonicity) is REPORTED, not "
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
    elif c3 and not c1:
        # No dose dependence: the axis's authority does NOT track envelope
        # width. That contradicts V3-EXQ-926a's stated finding of record and is
        # the informative null this run is powered to detect -- it is NOT a
        # falsification of MECH-449 (926a already showed the axis can act).
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
        f"[937] outcome={outcome} label={label} lift={per_seed_lift} "
        f"knee={knee_floor} env_sep={env_dose_separation:.3f} "
        f"empty_eligible={total_empty} -> {out_path}",
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-937 MECH-449/ARC-107 perseveration No-Go x envelope-width "
            "dose response"
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
