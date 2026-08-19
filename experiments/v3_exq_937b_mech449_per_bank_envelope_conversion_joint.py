"""V3-EXQ-937b -- MECH-449 / ARC-107: the PER-BANK (envelope size, converted) JOINT.

EXPERIMENT_PURPOSE=diagnostic -- RECORDING-ONLY successor to V3-EXQ-937a. It
changes no mechanism and poses no new hypothesis. It exists so that the
statistic which actually answers the question is EMITTED rather than computed
and discarded.

WHY THIS RUN EXISTS
-------------------
`failure_autopsy_V3-EXQ-937-937a-cluster_2026-08-18` (status CONFIRMED; ratified
and applied by the /governance cycle of 2026-08-18) found that V3-EXQ-937a
computes the per-bank (pre-No-Go envelope size, converted) joint at
v3_exq_937a_mech449_envelope_inertness_point.py:536-549 and then DISCARDS it,
reporting only cell-median aggregates.

Those aggregates are actively misleading. 937a's headline curve -- conversion
0.339 / 0.802 / 0.977 / 0.659 across realized envelope 1/2/3/4 -- is a MIXING
PROPORTION, not a conversion rate. Recovering the discarded joint over
11 floors x 3 seeds x 128 banks = 4,224 banks shows that conversion is a
DETERMINISTIC STEP FUNCTION of per-bank envelope size:

    per-bank envelope size    1        2        3       4
    banks                  1277     1196     1062     689
    conversions               0     1196     1062     689
    rate               0.000000 1.000000 1.000000 1.000000

stepping exactly where `gng_protect_min_eligible = 1` predicts
(n_can_drop = envelope - protect_min = 0 at envelope 1; see
ree_core/predictors/e3_selector.py:1596-1690). Consequently
manifest_conversion(floor) == 1 - P(envelope collapses to 1 | floor) to
0.000000 at ALL 11 doses: the reported curve is the MEDIATOR's own distribution
and carries NO independent information about the outcome given the mediator.

THE CLUSTER'S PLANNING DECISION, WHICH THIS RUN DISCHARGES
-----------------------------------------------------------
  "Any future ARC-107 / MECH-448 / MECH-449 readout keyed to a cell-level
   conversion rate over a discrete eligibility envelope is measuring the
   envelope-size distribution, not the gating. Key such criteria to the per-unit
   mediator and record the per-unit joint."

So every criterion below is keyed to the PER-BANK joint, and the joint itself is
recorded three ways that are mutually redundant on purpose (Experimental
Recording Standard sec 3c -- a false record is free, a false omission forces a
re-run):
  (a) `per_bank` -- a COLUMNAR raw record per cell: one entry per bank for
      pre-No-Go envelope size, F-eligibility envelope size, converted,
      incumbent-is-F-argmin and gate-active. Nothing is aggregated away.
  (b) `conversion_by_envelope_size` -- the per-cell cross-tab, which is a
      SUFFICIENT STATISTIC for (a)'s joint (both variables are discrete and
      small), in the shape a later reader or the indexer can consume directly.
  (c) `summary.per_bank_conversion_by_envelope_*` -- the pooled cross-tabs, per
      arm and per K.

WHAT IS DELIBERATELY NOT RE-LITIGATED
--------------------------------------
The MECHANISM. The substrate does exactly what MECH-449 and
`gng_protect_min_eligible` specify -- verified independently by source read
(e3_selector.py:1483-1575, :1596-1690) and by the autopsy's 4,224-bank
re-measurement. The autopsy minted NO substrate_queue entry
(`recommended_substrate_queue_entry.action = "none"`). Nothing here is a probe
of a suspected substrate gap, and no threshold is re-fit.

The chip that proposed `chip-20260818-mech449-envelope4-nonmonotonic` (a
supposed conversion DROP at envelope 4, and a supposedly non-inert axis at
envelope 1) was WITHDRAWN by the same governance cycle: both readings are
binning artifacts of exactly the aggregation this run replaces. They are not
revisited.

THE ONE ADDITION BEYOND PURE RE-RECORDING, AND WHY IT IS HERE
---------------------------------------------------------------
A K ladder (K_CANDIDATES in {4, 6, 8}), K=4 being the exact replication anchor.

The autopsy's Learning 6 states: "MECH-448's 'graded' envelope is functionally
binary for the MECH-449 leg at K=4. With four candidates and integer
protect_min = 1, the No-Go has authority or it does not; there is no graded
regime. Whether the constitution's gradation is real at larger K is UNTESTED and
should not be assumed from these runs."

At K=4 the envelope realizes only 4 states, so "the step sits at 2" is
consistent with both "the step is at protect_min + 1 for any K" and "the shape
is an artifact of a 4-state mediator". K = 6 and 8 realize 6 and 8 states and
separate those. The cost is measured, not assumed: the whole grid is a
selection-face probe with no training -- 128 banks time at 0.38s (K=4) and 0.48s
(K=8) on the authoring box, so 297 cells is ~3 minutes.

The K axis is REPORTED ONLY (C6). The load-bearing gate C1 stays on the K=4
anchor, so this run's PASS/FAIL is an exact per-bank re-measurement of 937a's
design and the generality axis is strictly additive. A K-generality failure
would be a finding, not a FAIL.

CRITERIA (all keyed to the PER-BANK joint, never to a cell aggregate)
---------------------------------------------------------------------
C1 (LOAD-BEARING, GATED) per-bank conversion rate at pre-No-Go envelope >= 2
   minus the rate at envelope == 1, ARM_CONSTITUTION, K=4 anchor, per seed.
   Threshold DOSE_LIFT_FLOOR = 0.40, carried UNCHANGED from 937/937a and NOT
   re-fit -- what changed is the statistic it is applied to.
C2 (REPORTED) per-bank conversion rate monotone-nondecreasing in envelope size.
C3 (GATED) fail-open safety contract never violated: no bank ever has an empty
   eligible set.
C4 (REPORTED) an inert regime was reached, with `measured` carrying the ACTUAL
   per-bank rate at envelope 1 -- not merely a verdict. (937a reported only a
   verdict here, which is half of why its "no inert regime reached" reading
   survived contact with its own data.)
C5 (REPORTED) CONTENT SPECIFICITY: ARM_SHUFFLED's per-bank conversion rate at
   EVERY envelope size, worst (max) cell. The autopsy's 0/4224 control is
   promoted to a first-class recorded criterion rather than a footnote.
C6 (REPORTED) K-GENERALITY: at EVERY K on the ladder the step sits at
   envelope 2 exactly, i.e. the per-bank rate at envelope == 1 is at or below
   INERTNESS_CEILING and the rate at envelope == 2 is at or above
   1 - INERTNESS_CEILING. This is the mechanism's own prediction
   (protect_min + 1 = 2, for any K), tested against two NAMED strata.
   It is deliberately NOT "the smallest envelope size whose rate reaches 0.5":
   that formulation reads the smallest POPULATED size, so an envelope size the
   dose ladder happens not to sample at some K reports as a K-dependence when
   it is a sampling gap -- which is the same binning artifact, one level up,
   that this entire run exists to remove. The per-K stratum counts are recorded
   so a gap is visible as a gap, and the smallest-populated-size reading is
   still reported alongside (`step_location_by_k`) as a descriptive, not a
   criterion.

READINESS PRECONDITIONS
------------------------
P1 `suppression_cross_candidate_range_supra_floor` -- worst cell, not a mean.
   Reachability PROVEN AT SETUP by _assert_p1_anchor_reachable(), FOR EVERY K on
   the ladder (the ladder is new, so its reachability is not inherited).
P2 `per_bank_envelope_strata_populated` -- C1 routes on a CONTRAST BETWEEN TWO
   PER-BANK STRATA, so readiness asserts that BOTH strata are populated at the
   K=4 anchor: min(n banks at envelope == 1, n banks at envelope >= 2). This is
   the V3-EXQ-643 same-statistic rule applied at the granularity 937a got wrong:
   937a's P2 asserted a spread of CELL-MEDIAN envelope sizes across the floor
   ladder, which is a different granularity from the per-bank strata its C1
   contrast actually consumed. Not circular -- P2 asserts the two strata EXIST,
   C1 measures the outcome DIFFERENCE between them.
P3 `envelope_floor_config_plumbing_live` -- from_dims plumbs all three knobs,
   guarding the documented silent-kwarg-swallow mode.

926a's per-cell "envelope >= 2" gate remains DELIBERATELY NOT CARRIED OVER, for
937a's reason and more strongly: this ladder is DESIGNED to include doses whose
envelope collapses to 1, and the envelope-1 stratum is now the load-bearing
half of C1's contrast. A whole-run all(...) gate over per-cell envelope size
would fail on precisely the banks this experiment exists to measure -- the
V3-EXQ-785 defect.

DV-SYMMETRY DECLARATION (mandatory per /queue-experiment Step 3)
----------------------------------------------------------------
  ARM_CONSTITUTION -- manipulation: the envelope floor changes which candidates
    are F-eligible, and the No-Go removes one from that set. DV: committed
    argmin identity, aggregated to a per-bank conversion indicator. Argmin's
    symmetry group is {uniform additive constant, monotone rescaling, tie
    permutation}; changing set MEMBERSHIP is in none of them -- it changes the
    domain the argmin is taken over. NOT invariant.
  ARM_SHUFFLED -- same domain manipulation, recency mass on a NON-incumbent.
    NOT invariant. Its ~0 conversion is the specificity control working, not a
    measured null.
  ARM_OFF -- HONEST LIMIT, carried here from 937a's docstring into the manifest
    (the autopsy's Learning 5: a limit recorded in the driver but omitted from
    the manifest is NOT recorded). ARM_OFF converts exactly 0.000 BY
    CONSTRUCTION: its arm selector IS the baseline config, so its pick cannot
    differ from the baseline pick that defines the incumbent. It is a
    TAUTOLOGICAL control on the conversion metric and proves nothing about the
    envelope. It is retained only because its cost is zero and its per-bank
    F-eligibility envelope column is genuinely informative -- it records the
    mediator's distribution under no No-Go at all.
  K LADDER -- K is not a manipulation of the DV's own symmetry group; it changes
    the ARITY of the candidate set, i.e. the size of the domain the argmin runs
    over and the number of states the mediator can realize. NOT invariant, and
    reported rather than gated.

THIRD CAVEAT, CARRIED FROM THE AUTOPSY AND NOT SOLVED HERE
------------------------------------------------------------
conversion == 1 at envelope >= 2 is partly a CONSTRUCTION property: the
incumbent is both the F-argmin and the sole suppression target, so once the
No-Go has room to drop it, the committed pick MUST move. This run records that
fact per bank (`incumbent_is_f_argmin` column) rather than asserting it away.
Breaking that coupling would be a DIFFERENT experiment with a different design,
not a recording change, and is out of scope here.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
EXPERIMENT_TYPE = "v3_exq_937b_mech449_per_bank_envelope_conversion_joint"
CLAIM_IDS: List[str] = ["MECH-449", "ARC-107"]

SEEDS: List[int] = [42, 43, 44]
ARMS: List[str] = ["ARM_OFF", "ARM_CONSTITUTION", "ARM_SHUFFLED"]

# The dose ladder, UNCHANGED from V3-EXQ-937a. It is retained verbatim so this
# run is a per-bank re-measurement of the same grid rather than a new design --
# the floors are a sampling device for realized envelope sizes, not the
# mediator (the floor -> envelope map folds back; see 937a).
STOCK_FLOOR = 0.30
REFERENCE_FLOOR = 0.10
NARROWEST_FLOOR = 0.60
ENVELOPE_FLOORS: List[float] = [
    0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05,
]

# THE K LADDER. K=4 is the replication anchor and the ONLY stratum C1 gates on.
# 6 and 8 exist to separate "the step sits at protect_min + 1 for any K" from
# "the step shape is an artifact of a 4-state mediator" (autopsy Learning 6).
K_ANCHOR = 4
K_LADDER: List[int] = [4, 6, 8]

WORLD_DIM = 8
HORIZON = 3
N_BANKS = 128

# --- pre-registered thresholds (constants, never derived from run statistics) --
# C1: CARRIED UNCHANGED from V3-EXQ-937/937a. The autopsy's finding is that the
# bar was being applied to the wrong statistic, NOT that the bar was wrong, so
# lowering or re-deriving it here would be exactly the move /queue-experiment
# forbids ("NEVER lower the threshold to resolve it"). The mechanism predicts
# the per-bank lift is 1.0 - 0.0 = 1.0 against this 0.40 bar.
DOSE_LIFT_FLOOR = 0.40
# C4: per-bank conversion rate at or below this counts as INERT.
INERTNESS_CEILING = 0.10
# C5: ARM_SHUFFLED's per-bank rate must stay at or below this at EVERY envelope
# size. Set from the mechanism (the suppression mass is on a non-incumbent, so
# the No-Go cannot drop the incumbent), with headroom for tie jitter.
SPECIFICITY_CEILING = 0.05
# C6 descriptive readout: the smallest envelope size whose per-bank rate
# reaches this. REPORTED ONLY -- see C6's own note for why it is not the
# criterion.
STEP_HALF_RATE = 0.5
# C6: per-K occupancy floor for the two strata the criterion names. Measured on
# the authoring box over the full 11-floor ladder at 64 banks x seed 42, the
# pooled (envelope==1, envelope==2) counts are K=4: 226/202, K=6: 109/106,
# K=8: 56/40 -- so at 3 seeds x 128 banks the full run clears 20 at every K with
# two orders of magnitude of margin. The smoke variant exists because the smoke
# grid is 2 floors x 2 seeds; see the run-mode note at `floors` below.
C6_STRATUM_MIN_BANKS = 20
C6_STRATUM_MIN_BANKS_SMOKE = 1
SEED_MAJORITY = 2               # C1: seeds that must clear the lift (of 3)
MONOTONE_TOL = 0.10             # C2: per-step slack in envelope size
PERSEVERATION_FLOOR = 0.5       # gng_perseveration_floor (matches config default)

# Readiness floors:
SUPPRESSION_RANGE_FLOOR = 0.25   # P1 -- SAME statistic the C1 crossing depends on
# P2 -- BOTH per-bank strata C1 contrasts must be populated, at the K=4 anchor.
# Two pre-registered constants, one per run mode, both declared here rather than
# derived from the run: the smoke grid is 2 floors x 2 seeds x 4 banks = 16
# banks per arm per K, so the full-run floor is structurally unreachable there.
# SATISFIABILITY, checked on paper at design time (Step 3.5): the autopsy's
# 4,224-bank K=4 table gives 1277 banks at envelope 1 and 2947 at envelope >= 2,
# so min(...) = 1277 >> 200. The smoke's floors [0.40, 0.10] realize envelope 1
# and envelope 3 respectively, giving ~8 banks per stratum >> 2.
STRATUM_MIN_BANKS = 200
STRATUM_MIN_BANKS_SMOKE = 2

# Perseverative history construction, unchanged from V3-EXQ-926a/937/937a: the
# dominant class fills 6 of the 8-deep recency ring and two other classes take
# one slot each. BOTH halves matter: one candidate must clear
# gng_perseveration_floor and the rest must NOT. At K > 4 the extra classes are
# simply absent from the ring (penalty 0.0), so the range is unchanged at 0.75 --
# verified for every K on the ladder by _assert_p1_anchor_reachable().
DOMINANT_REPEATS = 6
HISTORY_DEPTH = 8

# Sentinel for a per-bank envelope column entry that the substrate did not
# report. Recorded rather than dropped, so a later reader can tell "not measured"
# from "measured 0" -- absence of the block would mean "unmeasured", which is
# precisely the ambiguity this whole run exists to remove.
ENVELOPE_UNAVAILABLE = -1

ANCHOR_REACHABILITY_EXEMPT = (
    "P2 (per_bank_envelope_strata_populated) IS the degeneracy definition for a "
    "two-stratum per-bank contrast, not a proxy for it: 'both strata C1 "
    "contrasts are populated' is what non-degenerate MEANS here. It also cannot "
    "be anchored to a frozen fixture, because no manifest in the corpus records "
    "a per-bank envelope/conversion joint at all -- producing the first one is "
    "what this run is for (confirmed via reanalysis_query.py: 0 of 6 manifests "
    "on any MECH-449 substrate_hash carry the readout). P1 is NOT exempted by "
    "argument: its reachability is proven at setup by "
    "_assert_p1_anchor_reachable(), which replays the SHIPPED control "
    "construction through the SHIPPED comparison, for EVERY K on the ladder. "
    "Neither precondition has the per-cell-boolean shape "
    "_lib/readiness_anchor.py requires, and coercing them would guard a "
    "re-implementation rather than the shipped predicate (that module's rule 1)."
)


# --------------------------------------------------------------------------- #
# bank / history construction (parameterised in K; otherwise verbatim 937a)
# --------------------------------------------------------------------------- #
def _make_candidate(action_class: int, world_vec: torch.Tensor, action_dim: int) -> Trajectory:
    states = [torch.zeros(1, WORLD_DIM) for _ in range(HORIZON + 1)]
    world_states = [world_vec.reshape(1, WORLD_DIM).clone() for _ in range(HORIZON + 1)]
    actions = torch.zeros(1, HORIZON, action_dim)
    actions[:, 0, action_class] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _build_bank(rng: torch.Generator, k: int) -> List[Trajectory]:
    """K candidates with all-distinct first-action classes and divergent world
    states, so raw F genuinely differs across candidates."""
    cands = []
    for i in range(k):
        wv = torch.randn(WORLD_DIM, generator=rng) * 0.5 + float(i) * 0.4
        cands.append(_make_candidate(action_class=i, world_vec=wv, action_dim=k))
    return cands


def _perseverative_history(dominant_class: int, k: int) -> List[int]:
    minority = [(dominant_class + 1) % k, (dominant_class + 2) % k]
    hist = [dominant_class] * DOMINANT_REPEATS + minority
    return hist[:HISTORY_DEPTH]


def _live_suppression_vector(dominant_class: int, k: int) -> torch.Tensor:
    """MECH-260's per-candidate recency-share, read from a LIVE dACC.

    Nothing is hand-stuffed: the history is pushed through real record_action()
    calls and the value comes from dacc._suppression_penalty().
    """
    dacc = DACCAdaptiveControl(DACCConfig())
    for a in _perseverative_history(dominant_class, k):
        dacc.record_action(a)
    return torch.tensor(
        [dacc._suppression_penalty(c) for c in range(k)],
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
    """One committed selection through the real select() path.

    Returns BOTH envelope readouts the substrate exposes:
      * `f_eligibility_envelope_size` -- MECH-448's F-eligibility envelope,
        reported by the demotion path and therefore available even when the
        constitution is OFF. This is the PRIMARY per-bank mediator here.
      * `envelope_size` -- the POST-No-Go eligible-set size, reported only when
        the constitution is on. 937a reconstructed the pre-No-Go size as
        `envelope_size + n_soft_applied`; that reconstruction is kept and
        cross-checked against the primary readout above (see
        `pre_nogo_envelope_agreement_rate`), rather than trusted silently.
    """
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
    f_env = diag.get("f_eligibility_envelope_size", None)
    return {
        "selected_index": int(result.selected_index),
        "go_nogo_active": bool(diag.get("go_nogo_constitution_active", False)),
        "n_soft_requested": int(diag.get("go_nogo_n_soft_requested", 0) or 0),
        "n_soft_applied": int(diag.get("go_nogo_n_soft_applied", 0) or 0),
        "envelope_size": int(env_size) if env_size is not None else None,
        "f_eligibility_envelope_size": int(f_env) if f_env is not None else None,
    }


def _config_plumbing_live() -> bool:
    """P3 readiness: from_dims plumbs ALL THREE knobs onto config.e3."""
    probe_floor = 0.17  # a value equal to no default, so a swallow is visible
    c = REEConfig.from_dims(
        body_obs_dim=6,
        world_obs_dim=25,
        world_dim=WORLD_DIM,
        action_dim=K_ANCHOR,
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


def _assert_p1_anchor_reachable(ks: List[int]) -> Dict[str, Any]:
    """Setup-time proof that P1's gate is REACHABLE by P1's own positive control.

    Run for EVERY K on the ladder. The K ladder is new in this run, so 937a's
    reachability proof does not carry over: a larger K dilutes the recency ring
    across more classes, and it is exactly the kind of change that can silently
    push a control below its own gate.

    The control is DETERMINISTIC in (dominant_class, k), so this is an exact
    check, not a statistical one, and it replays the SHIPPED control
    construction through the SHIPPED comparison -- not a re-implementation.
    """
    per_k: Dict[str, float] = {}
    worst_overall = None
    for k in ks:
        ranges = [
            float(_live_suppression_vector(d, k).max() - _live_suppression_vector(d, k).min())
            for d in range(k)
        ]
        worst_k = min(ranges)
        per_k[str(k)] = worst_k
        worst_overall = worst_k if worst_overall is None else min(worst_overall, worst_k)
    worst = float(worst_overall if worst_overall is not None else 0.0)
    if not (worst >= SUPPRESSION_RANGE_FLOOR):
        raise AnchorUnreachable(
            "P1 suppression_cross_candidate_range_supra_floor is UNREACHABLE: "
            f"the positive control's own worst cross-candidate range is {worst:.4f}, "
            f"below its gate of {SUPPRESSION_RANGE_FLOOR}, over K ladder {ks}. The "
            "gate is a guaranteed false negative and would mislabel every run "
            "substrate_not_ready_requeue. Widen the predicate or lower the gate -- "
            "do NOT interpret it as a substrate verdict."
        )
    return {
        "anchor_name": "suppression_cross_candidate_range_supra_floor",
        "reachable": True,
        "control_worst_range": worst,
        "control_worst_range_by_k": per_k,
        "threshold": SUPPRESSION_RANGE_FLOOR,
        "margin": worst - SUPPRESSION_RANGE_FLOOR,
        "reference_source": (
            "deterministic: an 8-deep recency ring with DOMINANT_REPEATS=6, "
            "replayed through real DACCAdaptiveControl.record_action() + "
            "_suppression_penalty() for every dominant class, at every K on the "
            "ladder"
        ),
    }


# --------------------------------------------------------------------------- #
# the cell
# --------------------------------------------------------------------------- #
def _crosstab(envelopes: List[int], converted: List[int]) -> Dict[str, Dict[str, Any]]:
    """The per-bank (envelope size, converted) JOINT, as a cross-tab.

    A SUFFICIENT STATISTIC for the joint: both variables are discrete and small,
    so nothing about their relationship is lost. Keys are stringified ints
    because JSON object keys are strings; ENVELOPE_UNAVAILABLE is kept as its own
    bucket rather than dropped.
    """
    n_by: Dict[int, int] = defaultdict(int)
    c_by: Dict[int, int] = defaultdict(int)
    for e, c in zip(envelopes, converted):
        n_by[e] += 1
        c_by[e] += c
    out: Dict[str, Dict[str, Any]] = {}
    for e in sorted(n_by):
        n = n_by[e]
        out[str(e)] = {
            "n_banks": n,
            "n_converted": c_by[e],
            "conversion_rate": c_by[e] / float(n) if n else 0.0,
        }
    return out


def _run_cell(arm: str, seed: int, envelope_floor: float, k: int, n_banks: int) -> Dict[str, Any]:
    """One (arm, seed, envelope_floor, K) cell over n_banks divergent banks."""
    reset_all_rng(seed)
    rng = torch.Generator().manual_seed(seed)
    gate_on = arm in ("ARM_CONSTITUTION", "ARM_SHUFFLED")

    # PER-BANK COLUMNS -- the whole point of this run. Columnar rather than a
    # list of dicts purely for manifest size (128 small ints per column per cell
    # vs ~5x that as repeated JSON keys); the (arm, seed, floor, K, bank index)
    # key is fully recoverable, since arm/seed/floor/K are on the cell and the
    # bank index is the list position.
    col_pre_env: List[int] = []
    col_f_env: List[int] = []
    col_converted: List[int] = []
    col_incumbent_is_f_argmin: List[int] = []
    col_gate_active: List[int] = []

    n_empty_eligible = 0
    n_envelope_agree = 0
    n_envelope_comparable = 0
    raw_ranges: List[float] = []
    supp_ranges: List[float] = []
    exemplar_suppression: Optional[List[float]] = None

    cond = f"{arm}@K{k}@floor{envelope_floor:.2f}"
    print(f"Seed {seed} Condition {cond}", flush=True)
    for b in range(n_banks):
        # The trailing `or (b + 1) == n_banks` guarantees at least one progress
        # line per cell even when n_banks < 8 (the --dry-run case), so the smoke
        # actually verifies the runner's `ep N/M` instrumentation instead of
        # silently emitting none. The denominator is the LOOP BOUND, never a
        # hardcoded constant.
        if (b + 1) % 8 == 0 or (b + 1) == n_banks:
            print(f"  [eval] {cond} seed={seed} ep {b+1}/{n_banks}", flush=True)

        cands = _build_bank(rng, k)

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

        base = _select_one(sel_demote, cands, torch.zeros(k), gate_on=False)
        incumbent = base["selected_index"]
        # MECH-448's F-eligibility envelope, read DIRECTLY from the demotion
        # path. Available on every arm including ARM_OFF, which is why it is the
        # primary per-bank mediator here rather than 937a's reconstruction.
        f_env = base["f_eligibility_envelope_size"]

        # ARM_SHUFFLED perseverates on a NON-incumbent candidate (specificity).
        target = incumbent if arm != "ARM_SHUFFLED" else (incumbent + 1) % k
        suppression = _live_suppression_vector(target, k)
        supp_ranges.append(float(suppression.max() - suppression.min()))
        if exemplar_suppression is None:
            exemplar_suppression = [float(x) for x in suppression.tolist()]
        col_incumbent_is_f_argmin.append(
            1 if int(suppression.argmax().item()) == target else 0
        )

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

        col_gate_active.append(1 if got["go_nogo_active"] else 0)

        # PRE-No-Go envelope size. Prefer the reconstruction from the arm
        # selector when the constitution reported one (this is exactly the
        # quantity 937a computed and discarded); fall back to the direct
        # F-eligibility readout, which is what ARM_OFF has.
        if got["envelope_size"] is not None:
            pre_env = int(got["envelope_size"] + got["n_soft_applied"])
            if got["envelope_size"] <= 0:
                n_empty_eligible += 1
            if f_env is not None:
                n_envelope_comparable += 1
                if f_env == pre_env:
                    n_envelope_agree += 1
        elif f_env is not None:
            pre_env = int(f_env)
        else:
            pre_env = ENVELOPE_UNAVAILABLE
        col_pre_env.append(pre_env)
        col_f_env.append(int(f_env) if f_env is not None else ENVELOPE_UNAVAILABLE)

        converted = 1 if (base["selected_index"] == incumbent
                          and got["selected_index"] != incumbent) else 0
        col_converted.append(converted)

    n_converted = sum(col_converted)
    cell = {
        "arm": arm,
        "seed": seed,
        "envelope_floor": envelope_floor,
        "k_candidates": k,
        "condition": cond,
        "gate_on": gate_on,
        "n_banks": n_banks,
        "suppression_vector_exemplar": exemplar_suppression,
        "suppression_range": statistics.median(supp_ranges) if supp_ranges else 0.0,
        "suppression_range_min": min(supp_ranges) if supp_ranges else 0.0,
        # ---- THE JOINT: raw per-bank columns, plus its sufficient statistic ----
        "per_bank": {
            "pre_nogo_envelope_size": col_pre_env,
            "f_eligibility_envelope_size": col_f_env,
            "converted": col_converted,
            "incumbent_is_f_argmin": col_incumbent_is_f_argmin,
            "gate_active": col_gate_active,
        },
        "conversion_by_envelope_size": _crosstab(col_pre_env, col_converted),
        # ---- cell aggregates, EXPLICITLY LABELLED FOR WHAT THEY ARE ----
        # `conversion_rate` keeps its 937/937a name so the three runs stay
        # directly comparable, but it is a MIXING PROPORTION over the cell's
        # envelope-size distribution, NOT a conversion rate given the envelope.
        # The autopsy showed this exact number being read as a dose-response.
        "conversion_rate": n_converted / float(n_banks),
        "conversion_rate_semantics": "MIXING_PROPORTION_over_envelope_size_distribution",
        "n_converted": n_converted,
        "incumbent_is_f_argmin_rate": (
            sum(col_incumbent_is_f_argmin) / float(n_banks) if n_banks else 0.0
        ),
        "n_empty_eligible": n_empty_eligible,
        "gate_active_rate": sum(col_gate_active) / float(n_banks) if n_banks else 0.0,
        "median_pre_nogo_envelope_size": (
            statistics.median([e for e in col_pre_env if e != ENVELOPE_UNAVAILABLE])
            if any(e != ENVELOPE_UNAVAILABLE for e in col_pre_env) else 0.0
        ),
        "median_pre_nogo_envelope_size_semantics": (
            "CELL SUMMARY of the MEDIATOR's distribution. 937a keyed its criteria "
            "to this and thereby measured the mediator's distribution rather than "
            "the outcome given the mediator. Retained for comparability only -- "
            "every criterion in THIS run reads conversion_by_envelope_size."
        ),
        "median_raw_f_range": statistics.median(raw_ranges) if raw_ranges else 0.0,
        "pre_nogo_envelope_agreement_rate": (
            n_envelope_agree / float(n_envelope_comparable)
            if n_envelope_comparable else None
        ),
        "n_pre_nogo_envelope_comparable": n_envelope_comparable,
    }
    cell["arm_fingerprint"] = compute_arm_fingerprint(
        config_slice={
            "arm": arm,
            "use_f_eligibility_demotion": True,
            "f_eligibility_envelope_floor": envelope_floor,
            "use_go_nogo_constitution": gate_on,
            "gng_perseveration_floor": PERSEVERATION_FLOOR,
            "n_banks": n_banks,
            "k": k,
            "dominant_repeats": DOMINANT_REPEATS,
            "history_depth": HISTORY_DEPTH,
        },
        seed=seed,
        script_path=Path(__file__),
        rng_fully_reset=True,
        config_slice_declared=True,
        extra_ineligible_reasons=["selection_face_synthetic_no_training"],
    )
    # Per-cell verdict is the SAFETY contract only. Scoring every cell against a
    # conversion floor would report the collapsed-envelope doses and both control
    # arms as failures when they are the design working.
    verdict_pass = n_empty_eligible == 0
    print(f"verdict: {'PASS' if verdict_pass else 'FAIL'}", flush=True)
    return cell


# --------------------------------------------------------------------------- #
# pooled per-bank readouts
# --------------------------------------------------------------------------- #
def _pool(cells: List[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    """Concatenate the per-bank columns of a set of cells."""
    envs: List[int] = []
    convs: List[int] = []
    for c in cells:
        envs.extend(c["per_bank"]["pre_nogo_envelope_size"])
        convs.extend(c["per_bank"]["converted"])
    return envs, convs


def _rate_where(cells: List[Dict[str, Any]], pred) -> Tuple[float, int, int]:
    """PER-BANK conversion rate over banks whose envelope size satisfies `pred`.

    Returns (rate, n_banks, n_converted). This is the statistic every criterion
    in this run reads -- never a mean of cell-level rates, which is the
    aggregation the autopsy identified as the defect.
    """
    envs, convs = _pool(cells)
    n = 0
    c = 0
    for e, v in zip(envs, convs):
        if e != ENVELOPE_UNAVAILABLE and pred(e):
            n += 1
            c += v
    return (c / float(n) if n else 0.0), n, c


def _step_location(crosstab: Dict[str, Dict[str, Any]]) -> Optional[int]:
    """Smallest envelope size whose PER-BANK conversion rate reaches STEP_HALF_RATE."""
    for key in sorted((int(k) for k in crosstab), key=int):
        if key == ENVELOPE_UNAVAILABLE:
            continue
        if crosstab[str(key)]["conversion_rate"] >= STEP_HALF_RATE:
            return key
    return None


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = datetime.utcnow()
    perf0 = time.perf_counter()
    # TWO seeds under --dry-run, not one. C1 requires SEED_MAJORITY (2) seeds to
    # clear the lift floor, so a single-seed smoke makes the load-bearing
    # criterion STRUCTURALLY UNREACHABLE and the smoke can only ever report
    # FAIL -- it would never exercise the PASS path it exists to validate.
    seeds = SEEDS[:2] if dry_run else SEEDS
    # 16 rather than 4 under --dry-run. At 4 banks x 2 seeds the envelope-2
    # stratum is a coin flip at K=8 (see the floor table below), so C6 would
    # read degenerate in the smoke for a sampling reason rather than a real one.
    # The cost is trivial: the whole smoke grid is 576 banks, ~2s.
    n_banks = 16 if dry_run else N_BANKS
    # SMOKE FLOOR SELECTION IS LOAD-BEARING, and the obvious choice is wrong.
    # The smoke must populate BOTH strata that C1 and C6 name (envelope == 1 and
    # envelope == 2) AT EVERY K, or those criteria are structurally unreachable
    # in the smoke and it can never exercise the PASS path it exists to
    # validate -- the same defect the two-seed rule above avoids for C1.
    #
    # Measured on the authoring box (64 banks, seed 42, per floor, per K), the
    # realized F-eligibility envelope distribution is strongly K-dependent, so a
    # floor pair chosen at K=4 does NOT transfer:
    #   floor 0.40 -> K=4 {1:49, 2:6, 4:9}   K=6 {1:14, 6:50}   K=8 {8:64}
    #   floor 0.10 -> K=4 {1:1, 2:18, 3:45}  K=6 {2:1, 3:15, 4:30, 5:18}
    #                                        K=8 {3:2, 4:15, 5:28, 6:16, 7:3}
    # so V3-EXQ-937a's [0.40, 0.10] pair leaves envelope 2 essentially unsampled
    # at K=6 and completely unsampled at K=8. Confirmed live: the first smoke of
    # this script reported step_location_by_k {4:2, 6:3, 8:4}, which reads as a
    # K-dependence and is in fact a sampling gap.
    #   floor 0.20 -> K=4 {1:5, 2:38, 3:21}  K=6 {1:2, 2:43, 3:18, 4:1}
    #                                        K=8 {1:22, 2:28, 3:8, 8:6}
    # 0.20 is the one floor on the ladder that populates BOTH strata at ALL
    # THREE K, so it replaces 0.10 in the smoke pair. The full run uses the whole
    # 11-floor ladder and does not depend on this choice.
    floors = [0.40, 0.20] if dry_run else ENVELOPE_FLOORS
    # The smoke keeps the WHOLE K ladder: C6 is a cross-K criterion, so a
    # single-K smoke would make it structurally unreachable.
    ks = list(K_LADDER)
    stratum_floor = STRATUM_MIN_BANKS_SMOKE if dry_run else STRATUM_MIN_BANKS
    c6_stratum_floor = C6_STRATUM_MIN_BANKS_SMOKE if dry_run else C6_STRATUM_MIN_BANKS

    # Prove P1's gate is reachable by its own control, at EVERY K, BEFORE any
    # compute is spent -- so an unmeetable predicate is caught at setup rather
    # than after the grid runs and self-routes a false substrate verdict.
    p1_reachability = _assert_p1_anchor_reachable(ks)
    print(
        f"[937b] P1 anchor reachable: control worst range "
        f"{p1_reachability['control_worst_range']:.4f} >= gate "
        f"{p1_reachability['threshold']} (margin "
        f"{p1_reachability['margin']:.4f}); by K "
        f"{p1_reachability['control_worst_range_by_k']}",
        flush=True,
    )

    arm_results: List[Dict[str, Any]] = []
    for k in ks:
        for floor in floors:
            for arm in ARMS:
                for seed in seeds:
                    arm_results.append(_run_cell(arm, seed, floor, k, n_banks))

    def _sel(arm=None, k=None, seed=None) -> List[Dict[str, Any]]:
        out = arm_results
        if arm is not None:
            out = [c for c in out if c["arm"] == arm]
        if k is not None:
            out = [c for c in out if c["k_candidates"] == k]
        if seed is not None:
            out = [c for c in out if c["seed"] == seed]
        return out

    on_cells = _sel(arm="ARM_CONSTITUTION")
    shuf_cells = _sel(arm="ARM_SHUFFLED")
    off_cells = _sel(arm="ARM_OFF")
    anchor_on = _sel(arm="ARM_CONSTITUTION", k=K_ANCHOR)

    # ---------------- readiness preconditions (recomputable) ----------------
    # Worst CELL, not a mean -- `met` is a worst-case claim, so `measured` must
    # be the same extremum the indexer recomputes on.
    supp_ranges = [c["suppression_range_min"] for c in arm_results]
    worst_supp_range = min(supp_ranges) if supp_ranges else 0.0

    # P2: BOTH per-bank strata C1 contrasts must be populated, at the K anchor.
    # Same GRANULARITY as C1 (per-bank strata), which is the correction to
    # 937a's P2 -- that one asserted a spread of CELL-MEDIAN envelope sizes.
    _, n_pinned_anchor, _ = _rate_where(anchor_on, lambda e: e <= 1)
    _, n_open_anchor, _ = _rate_where(anchor_on, lambda e: e >= 2)
    strata_min = min(n_pinned_anchor, n_open_anchor)
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
                "action class, built via real record_action() calls, at every K "
                "on the ladder"
            ),
            "measured": worst_supp_range,
            "threshold": SUPPRESSION_RANGE_FLOOR,
            "direction": "lower",
            "met": worst_supp_range >= SUPPRESSION_RANGE_FLOOR,
        },
        {
            # C1 routes on a contrast between TWO PER-BANK STRATA, so readiness
            # asserts those strata are populated -- the same statistic at the
            # same granularity (the V3-EXQ-643 rule, applied at the granularity
            # 937a's own P2 missed). `measured` is the WORST (min) stratum, not
            # a mean, because `met` is a worst-case claim over the two.
            "name": "per_bank_envelope_strata_populated",
            "kind": "readiness",
            "description": (
                "both per-bank envelope strata that C1 contrasts are populated "
                "at the K=%d anchor: min(n banks at pre-No-Go envelope == 1, "
                "n banks at envelope >= 2) on ARM_CONSTITUTION" % K_ANCHOR
            ),
            "control": (
                "positive control: the dose ladder is designed to realize both "
                "strata -- floor 0.40 collapses the envelope to 1 and floor 0.10 "
                "leaves room; both are on the ladder in every run mode"
            ),
            "measured": strata_min,
            "threshold": stratum_floor,
            "offending_stratum": (
                "envelope==1" if n_pinned_anchor <= n_open_anchor else "envelope>=2"
            ),
            "n_banks_envelope_pinned": n_pinned_anchor,
            "n_banks_envelope_open": n_open_anchor,
            "direction": "lower",
            "met": strata_min >= stratum_floor,
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
    # EVERY criterion below reads the PER-BANK joint. None reads a cell-level
    # conversion rate -- that is the whole point of this run.
    anchor_crosstab = _crosstab(*_pool(anchor_on))
    rate_pinned, n_pinned, c_pinned = _rate_where(anchor_on, lambda e: e <= 1)
    rate_open, n_open, c_open = _rate_where(anchor_on, lambda e: e >= 2)

    # C1 (LOAD-BEARING): per-seed per-bank lift, K anchor.
    per_seed_lift: Dict[str, float] = {}
    per_seed_detail: Dict[str, Dict[str, Any]] = {}
    for sd in seeds:
        cs = _sel(arm="ARM_CONSTITUTION", k=K_ANCHOR, seed=sd)
        rp, np_, cp = _rate_where(cs, lambda e: e <= 1)
        ro, no_, co = _rate_where(cs, lambda e: e >= 2)
        per_seed_lift[str(sd)] = ro - rp
        per_seed_detail[str(sd)] = {
            "rate_envelope_pinned": rp, "n_banks_pinned": np_, "n_converted_pinned": cp,
            "rate_envelope_open": ro, "n_banks_open": no_, "n_converted_open": co,
        }
    n_seeds_clearing = sum(1 for v in per_seed_lift.values() if v >= DOSE_LIFT_FLOOR)
    c1 = n_seeds_clearing >= SEED_MAJORITY

    # C2: per-bank conversion rate monotone-nondecreasing in envelope size.
    env_levels = sorted(int(e) for e in anchor_crosstab if int(e) != ENVELOPE_UNAVAILABLE)
    steps = []
    for i2 in range(len(env_levels) - 1):
        a = anchor_crosstab[str(env_levels[i2])]["conversion_rate"]
        b = anchor_crosstab[str(env_levels[i2 + 1])]["conversion_rate"]
        steps.append({"from_envelope": env_levels[i2], "to_envelope": env_levels[i2 + 1],
                      "delta": b - a, "ok": (b - a) >= -MONOTONE_TOL})
    c2 = all(st["ok"] for st in steps) if steps else False

    # C3: the fail-open safety contract is never violated at any dose or K.
    total_empty = sum(c["n_empty_eligible"] for c in arm_results)
    c3 = total_empty == 0

    # C4: was an INERT regime reached, and AT WHAT RATE. `measured` carries the
    # actual per-bank rate at the narrowest realized envelope -- 937a reported
    # only a verdict here, which is half of why its "no inert regime reached"
    # reading survived contact with its own data.
    inert_envelope = None
    for e in env_levels:
        if anchor_crosstab[str(e)]["conversion_rate"] <= INERTNESS_CEILING:
            inert_envelope = e
            break
    c4 = inert_envelope is not None

    # C5: CONTENT SPECIFICITY. Worst (max) per-bank rate across every envelope
    # size on ARM_SHUFFLED, all K pooled.
    shuf_crosstab = _crosstab(*_pool(shuf_cells))
    shuf_worst_rate = 0.0
    shuf_worst_env = None
    for e_key, blk in shuf_crosstab.items():
        if int(e_key) == ENVELOPE_UNAVAILABLE:
            continue
        if blk["conversion_rate"] >= shuf_worst_rate:
            shuf_worst_rate = blk["conversion_rate"]
            shuf_worst_env = int(e_key)
    c5 = shuf_worst_rate <= SPECIFICITY_CEILING

    # C6: K-GENERALITY, tested against TWO NAMED STRATA rather than against the
    # smallest populated envelope size. The mechanism predicts the step sits at
    # protect_min + 1 = 2 for ANY K, so that is what is asserted: rate at
    # envelope == 1 is inert, rate at envelope == 2 is saturated, at every K.
    # See the criterion's own note for why the smallest-populated-size reading
    # is reported but not gated on.
    crosstab_by_k = {
        str(k): _crosstab(*_pool(_sel(arm="ARM_CONSTITUTION", k=k))) for k in ks
    }
    step_by_k = {k: _step_location(ct) for k, ct in crosstab_by_k.items()}
    k_strata: Dict[str, Dict[str, Any]] = {}
    for k in ks:
        cs = _sel(arm="ARM_CONSTITUTION", k=k)
        r1, n1, x1 = _rate_where(cs, lambda e: e == 1)
        r2, n2, x2 = _rate_where(cs, lambda e: e == 2)
        k_strata[str(k)] = {
            "rate_at_envelope_1": r1, "n_banks_at_envelope_1": n1,
            "n_converted_at_envelope_1": x1,
            "rate_at_envelope_2": r2, "n_banks_at_envelope_2": n2,
            "n_converted_at_envelope_2": x2,
            "min_stratum_n": min(n1, n2),
            "step_at_2": bool(
                r1 <= INERTNESS_CEILING and r2 >= (1.0 - INERTNESS_CEILING)
            ),
        }
    c6_worst_stratum_n = (
        min(v["min_stratum_n"] for v in k_strata.values()) if k_strata else 0
    )
    c6 = bool(k_strata) and all(v["step_at_2"] for v in k_strata.values())

    # ------------------------- non-degeneracy ------------------------------
    gate_ever_active = any(c["gate_active_rate"] > 0 for c in on_cells)
    off_envs, off_convs = _pool(off_cells)
    on_envs, on_convs = _pool(on_cells)
    on_off_differ = (
        (sum(on_convs) / float(len(on_convs)) if on_convs else 0.0)
        != (sum(off_convs) / float(len(off_convs)) if off_convs else 0.0)
    )
    core_non_degenerate = bool(
        gate_ever_active
        and worst_supp_range >= SUPPRESSION_RANGE_FLOOR
        and strata_min >= stratum_floor
        and on_off_differ
    )
    criteria_non_degenerate = {
        "C1_per_bank_envelope_conversion_lift": core_non_degenerate,
        # C2 needs at least 3 distinct mediator levels to describe a shape;
        # two points cannot distinguish monotone from a step.
        "C2_per_bank_conversion_monotone_in_envelope": bool(
            core_non_degenerate and len(env_levels) >= 3
        ),
        "C3_safety_failopen": bool(arm_results),
        # C4 is degenerate if the envelope-1 stratum was never populated -- i.e.
        # the ladder never realized the regime where inertness is predicted.
        "C4_inert_regime_reached_per_bank": bool(n_pinned > 0),
        # C5 is degenerate if the shuffled arm's gate never engaged: a control
        # that never fired cannot demonstrate specificity.
        "C5_content_specificity_shuffled": bool(
            any(c["gate_active_rate"] > 0 for c in shuf_cells)
        ),
        # C6 is degenerate unless BOTH named strata are populated at EVERY K:
        # an unsampled envelope size is a sampling gap, not a K-dependence.
        "C6_step_location_invariant_in_k": bool(
            len(k_strata) >= 2 and c6_worst_stratum_n >= c6_stratum_floor
        ),
    }

    criteria = [
        {
            "name": "C1_per_bank_envelope_conversion_lift",
            "load_bearing": True,
            "passed": bool(c1),
            "measured": per_seed_lift,
            "measured_detail": per_seed_detail,
            "threshold": DOSE_LIFT_FLOOR,
            "seeds_clearing": n_seeds_clearing,
            "seeds_required": SEED_MAJORITY,
            "note": (
                "PER-BANK rate at envelope >= 2 minus PER-BANK rate at envelope "
                "== 1, K=%d anchor. Threshold carried UNCHANGED from "
                "V3-EXQ-937/937a -- the autopsy found the bar was applied to the "
                "wrong statistic, not that the bar was wrong." % K_ANCHOR
            ),
        },
        {
            "name": "C2_per_bank_conversion_monotone_in_envelope",
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
            "name": "C4_inert_regime_reached_per_bank",
            "load_bearing": False,
            "passed": bool(c4),
            # The ACTUAL per-bank rate in the protect-min stratum, not a verdict.
            "measured": rate_pinned,
            "threshold": INERTNESS_CEILING,
            "direction": "upper",
            "inert_envelope_size": inert_envelope,
            "n_banks_in_pinned_stratum": n_pinned,
            "n_converted_in_pinned_stratum": c_pinned,
        },
        {
            "name": "C5_content_specificity_shuffled",
            "load_bearing": False,
            "passed": bool(c5),
            "measured": shuf_worst_rate,
            "threshold": SPECIFICITY_CEILING,
            "direction": "upper",
            "offending_envelope_size": shuf_worst_env,
            "note": (
                "worst (max) per-bank conversion rate across every envelope size "
                "on ARM_SHUFFLED, all K pooled -- worst cell, not a mean"
            ),
        },
        {
            "name": "C6_step_location_invariant_in_k",
            "load_bearing": False,
            "passed": bool(c6),
            # The WORST per-K stratum occupancy, so `measured` is the same
            # extremum a recompute would read and a thin stratum is visible.
            "measured": c6_worst_stratum_n,
            "threshold": c6_stratum_floor,
            "direction": "lower",
            "per_k_strata": k_strata,
            "step_location_by_k_descriptive": step_by_k,
            "note": (
                "The criterion is 'the step sits at envelope 2 at EVERY K' -- "
                "rate at envelope == 1 at or below %.2f AND rate at envelope == "
                "2 at or above %.2f -- which is the mechanism's own prediction "
                "(protect_min + 1 = 2, for any K), tested against two NAMED "
                "strata. `step_location_by_k_descriptive` (smallest POPULATED "
                "envelope size reaching %.2f) is reported alongside but is NOT "
                "the criterion: it reads a dose-ladder sampling gap as a "
                "K-dependence. Confirmed live during authoring -- this script's "
                "first smoke reported {4:2, 6:3, 8:4} purely because its floor "
                "pair left envelope 2 unsampled at K=6/8. `measured` here is the "
                "worst per-K occupancy of the two named strata, against the "
                "pre-registered floor, so a thin stratum is visible rather than "
                "silently decisive. "
                "REPORTED, NOT GATED, and deliberately so: a genuinely "
                "K-dependent step would be a FINDING (MECH-448's gradation is "
                "real at larger K, which the autopsy's Learning 6 records as "
                "untested) rather than a failure of this measurement. Gating it "
                "would let a generality result convert a clean per-bank "
                "re-recording into a FAIL."
                % (INERTNESS_CEILING, 1.0 - INERTNESS_CEILING, STEP_HALF_RATE)
            ),
        },
    ]
    combination_rule = (
        "PASS = readiness_ok AND C1 AND C3. C2, C4, C5 and C6 are REPORTED, not "
        "gated. C1 is the load-bearing criterion and is keyed to the K=%d anchor "
        "so this run's verdict is an exact per-bank re-measurement of "
        "V3-EXQ-937a's design; the K ladder (C6) is strictly additive. C4 is "
        "reported rather than gated because failing to reach inertness even in "
        "the protect-min stratum would be the informative null, not a FAIL; C6 "
        "for the symmetric reason." % K_ANCHOR
    )

    # ---------------------------- routing ----------------------------------
    step_is_sharp = bool(
        rate_pinned <= INERTNESS_CEILING and rate_open >= (1.0 - INERTNESS_CEILING)
    )
    if not readiness_ok:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
    elif c1 and c3 and step_is_sharp:
        outcome = "PASS"
        label = "conversion_is_a_deterministic_step_in_per_bank_envelope_size"
        direction = "supports"
    elif c1 and c3:
        outcome = "PASS"
        label = "per_bank_envelope_size_gates_perseveration_conversion"
        direction = "supports"
    elif c3 and not c1 and c2:
        # ROUTING DEFECT FIXED FROM V3-EXQ-937 and carried forward from 937a: a
        # label must not assert an absence of dependence that a sibling
        # criterion in the same manifest just measured.
        outcome = "FAIL"
        label = "graded_per_bank_dose_response_below_prereg_lift"
        direction = "non_contributory"
    elif c3 and not c1:
        outcome = "FAIL"
        label = "conversion_independent_of_per_bank_envelope_size"
        direction = "non_contributory"
    else:
        outcome = "FAIL"
        label = "safety_failopen_contract_violated"
        direction = "weakens"

    agree_rates = [
        c["pre_nogo_envelope_agreement_rate"] for c in arm_results
        if c["pre_nogo_envelope_agreement_rate"] is not None
    ]

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
            "aggregation_warning": (
                "READ THE PER-BANK JOINT, NOT THE CELL AGGREGATES. Every "
                "cell-level `conversion_rate` in this manifest is a MIXING "
                "PROPORTION over that cell's envelope-size distribution, and "
                "equals 1 - P(envelope collapses to 1) rather than a conversion "
                "rate given the envelope. V3-EXQ-937a's headline curve was "
                "exactly this quantity read as a dose-response "
                "(failure_autopsy_V3-EXQ-937-937a-cluster_2026-08-18, CONFIRMED). "
                "The load-bearing readouts here are "
                "`arm_results[].conversion_by_envelope_size` (per cell), "
                "`arm_results[].per_bank` (raw columns) and "
                "`summary.per_bank_conversion_by_envelope_constitution`."
            ),
            "dv_symmetry_declaration": {
                "ARM_CONSTITUTION": (
                    "manipulation = lowering f_eligibility_envelope_floor widens "
                    "the F-eligible SET (a change of the argmin's DOMAIN), and "
                    "the No-Go then removes a candidate from that set. DV = "
                    "committed argmin identity, recorded as a per-bank "
                    "conversion indicator. Argmin's symmetry group is {uniform "
                    "additive constant, monotone rescaling, tie permutation}; "
                    "changing set MEMBERSHIP is in none of them. NOT invariant."
                ),
                "ARM_SHUFFLED": (
                    "same domain manipulation, recency mass on a NON-incumbent. "
                    "Same argument, NOT invariant. Near-zero conversion here is "
                    "the specificity control working, not a measured null; it is "
                    "recorded as first-class criterion C5 rather than a footnote."
                ),
                "ARM_OFF": (
                    "HONEST LIMIT, carried from the driver docstring into the "
                    "manifest per the autopsy's Learning 5 (a limit recorded in "
                    "the driver but omitted from the manifest is NOT recorded). "
                    "ARM_OFF converts exactly 0.000 BY CONSTRUCTION: its arm "
                    "selector IS the baseline config, so its pick cannot differ "
                    "from the baseline pick that defines the incumbent. It is a "
                    "TAUTOLOGICAL control on the conversion metric and proves "
                    "NOTHING about the envelope. It is retained only because its "
                    "cost is zero and its per-bank f_eligibility_envelope_size "
                    "column records the mediator's distribution under no No-Go."
                ),
                "K_LADDER": (
                    "K is not a manipulation of the DV's own symmetry group: it "
                    "changes the ARITY of the candidate set, i.e. the size of the "
                    "domain the argmin runs over and the number of states the "
                    "mediator can realize. NOT invariant. Reported (C6), not "
                    "gated -- see combination_rule."
                ),
            },
            "anchor_reachability": p1_reachability,
            "construction_caveat": (
                "conversion == 1 at envelope >= 2 is PARTLY A CONSTRUCTION "
                "PROPERTY: the incumbent is both the F-argmin and the sole "
                "suppression target, so once the No-Go has room to drop it the "
                "committed pick MUST move. This run RECORDS that coupling per "
                "bank (`per_bank.incumbent_is_f_argmin`) rather than asserting "
                "it away. Breaking the coupling would be a DIFFERENT experiment "
                "with a different design, not a recording change, and is out of "
                "scope here. Carried verbatim from the autopsy's caveat (3)."
            ),
            "generality_caveat": (
                "Selection-face synthetic probe, NO TRAINING and NO agent loop "
                "(reuse_ineligible_reasons: selection_face_synthetic_no_training). "
                "This confirms the constitution's ARITHMETIC, not any behavioural "
                "competence. purpose=diagnostic, so it scores nothing and "
                "PROMOTES NOTHING. Carried from the autopsy's caveat (1)."
            ),
            "scoped_out_preconditions": {
                "pre_nogo_eligible_set_admits_alternative": (
                    "V3-EXQ-926a's per-cell 'envelope >= 2' readiness gate is "
                    "DELIBERATELY NOT carried over, for V3-EXQ-937a's reason and "
                    "more strongly here: the envelope-1 stratum is now the "
                    "load-bearing HALF of C1's contrast, so a whole-run all(...) "
                    "gate over per-cell envelope size would fail on precisely the "
                    "banks this experiment exists to measure and vacate the run "
                    "-- the V3-EXQ-785 defect exactly. Replaced by "
                    "per_bank_envelope_strata_populated, which asserts the "
                    "property this design actually needs, at the same "
                    "granularity C1 consumes."
                )
            },
        },
        "summary": {
            # ---- THE HEADLINE READOUTS: per-bank, conditioned on the mediator --
            "per_bank_conversion_by_envelope_constitution": anchor_crosstab,
            "per_bank_conversion_by_envelope_constitution_by_k": crosstab_by_k,
            "per_bank_conversion_by_envelope_shuffled": shuf_crosstab,
            "per_bank_conversion_by_envelope_off": _crosstab(*_pool(off_cells)),
            "per_bank_rate_envelope_pinned": rate_pinned,
            "n_banks_envelope_pinned": n_pinned,
            "n_converted_envelope_pinned": c_pinned,
            "per_bank_rate_envelope_open": rate_open,
            "n_banks_envelope_open": n_open,
            "n_converted_envelope_open": c_open,
            "per_seed_per_bank_lift": per_seed_lift,
            "per_seed_per_bank_detail": per_seed_detail,
            "seeds_clearing_lift_floor": n_seeds_clearing,
            "step_is_sharp": step_is_sharp,
            "step_location_by_k_descriptive": step_by_k,
            "step_at_envelope_2_by_k": k_strata,
            "step_at_envelope_2_at_every_k": c6,
            "c6_worst_per_k_stratum_n": c6_worst_stratum_n,
            "inert_envelope_first_at_or_below_ceiling": inert_envelope,
            "shuffled_worst_per_bank_rate": shuf_worst_rate,
            "shuffled_worst_envelope_size": shuf_worst_env,
            "monotone_steps": steps,
            # ---- integrity cross-check between the two envelope readouts ----
            "pre_nogo_envelope_agreement_rate_min": min(agree_rates) if agree_rates else None,
            "pre_nogo_envelope_agreement_cells": len(agree_rates),
            "pre_nogo_envelope_agreement_note": (
                "the reconstructed pre-No-Go size (go_nogo_envelope_size + "
                "n_soft_applied, which is what V3-EXQ-937a computed) is "
                "cross-checked per bank against MECH-448's directly-reported "
                "f_eligibility_envelope_size. Recorded rather than asserted."
            ),
            # ---- cell aggregates, kept ONLY for comparability with 937/937a ----
            "mixing_proportion_by_floor_constitution__NOT_a_conversion_rate": {
                str(f): (
                    statistics.fmean([
                        c["conversion_rate"] for c in anchor_on
                        if abs(c["envelope_floor"] - f) < 1e-9
                    ]) if any(abs(c["envelope_floor"] - f) < 1e-9 for c in anchor_on) else 0.0
                )
                for f in floors
            },
            "mixing_proportion_semantics": (
                "MIXING PROPORTION over the envelope-size distribution at each "
                "floor, i.e. 1 - P(envelope collapses to 1 | floor). It is NOT a "
                "conversion rate given the envelope and MUST NOT be read as a "
                "dose-response. Retained solely so this run is directly "
                "comparable with V3-EXQ-937 and V3-EXQ-937a, whose headline "
                "curves are this same quantity."
            ),
            "worst_suppression_range": worst_supp_range,
            "total_empty_eligible": total_empty,
            "gate_ever_active": gate_ever_active,
            "k_anchor": K_ANCHOR,
            "k_ladder": ks,
        },
        "config": {
            "seeds": seeds,
            "arms": ARMS,
            "envelope_floors": floors,
            "k_ladder": ks,
            "k_anchor": K_ANCHOR,
            "n_banks": n_banks,
            "world_dim": WORLD_DIM,
            "horizon": HORIZON,
            "action_dim": "= k_candidates per cell",
            "perseverated_class": "per-bank committed incumbent (probed, not fixed)",
            "dominant_repeats": DOMINANT_REPEATS,
            "history_depth": HISTORY_DEPTH,
            "gng_perseveration_floor": PERSEVERATION_FLOOR,
            "dose_lift_floor": DOSE_LIFT_FLOOR,
            "inertness_ceiling": INERTNESS_CEILING,
            "specificity_ceiling": SPECIFICITY_CEILING,
            "step_half_rate": STEP_HALF_RATE,
            "c6_stratum_min_banks": c6_stratum_floor,
            "c6_stratum_min_banks_full": C6_STRATUM_MIN_BANKS,
            "c6_stratum_min_banks_smoke": C6_STRATUM_MIN_BANKS_SMOKE,
            "narrowest_floor": NARROWEST_FLOOR,
            "seed_majority": SEED_MAJORITY,
            "monotone_tol": MONOTONE_TOL,
            "suppression_range_floor": SUPPRESSION_RANGE_FLOOR,
            "stratum_min_banks": stratum_floor,
            "stratum_min_banks_full": STRATUM_MIN_BANKS,
            "stratum_min_banks_smoke": STRATUM_MIN_BANKS_SMOKE,
            "stock_floor": STOCK_FLOOR,
            "reference_floor": REFERENCE_FLOOR,
            "envelope_unavailable_sentinel": ENVELOPE_UNAVAILABLE,
        },
        "notes": (
            "RECORDING-ONLY successor to V3-EXQ-937a. It changes no mechanism, "
            "poses no new hypothesis and re-fits no threshold. Routed by "
            "failure_autopsy_V3-EXQ-937-937a-cluster_2026-08-18 (status "
            "CONFIRMED), ratified and applied by the /governance cycle of "
            "2026-08-18. That autopsy found that V3-EXQ-937a computes the "
            "per-bank (pre-No-Go envelope size, converted) joint at "
            "v3_exq_937a_mech449_envelope_inertness_point.py:536-549 and "
            "DISCARDS it, reporting only cell-median aggregates, and that those "
            "aggregates are MIXING PROPORTIONS: recovering the joint over 11 "
            "floors x 3 seeds x 128 banks = 4,224 banks shows conversion is a "
            "DETERMINISTIC STEP FUNCTION of per-bank envelope size (envelope 1 "
            "-> 0/1277, rate 0.000000; envelope 2/3/4 -> 1196/1196, 1062/1062, "
            "689/689, rate 1.000000), stepping exactly where "
            "gng_protect_min_eligible=1 predicts. This run EMITS that joint as a "
            "first-class recorded artifact per the Experimental Recording "
            "Standard sec 3b/3c: raw per-bank columns, the per-cell cross-tab, "
            "and pooled cross-tabs per arm and per K. It also closes 937a's "
            "elapsed_seconds always-core gap and carries ARM_OFF's tautology "
            "limit from the driver docstring into the manifest's "
            "dv_symmetry_declaration (the autopsy's Learning 5). "
            "GOV-REUSE-1: the decisive readout is the per-bank (envelope size, "
            "converted) joint. reanalysis_query.py --readout "
            "conversion_by_envelope_size --claim MECH-449 returns 6 manifests "
            "across 6 substrate_hashes and 0 carry the readout (checked "
            "v3_exq_937, v3_exq_937a, v3_exq_699, v3_exq_699b x2, v3_exq_689g); "
            "937 and 937a record only cell medians and conversion counts. NOT "
            "RECOVERABLE by reanalysis -> run. "
            "THE ONE ADDITION beyond re-recording is a K ladder {4, 6, 8}, which "
            "the autopsy's Learning 6 names as untested: at K=4 with integer "
            "protect_min=1 the envelope realizes only 4 states, so the step shape "
            "may be an artifact of small K. K=4 is the replication anchor and the "
            "only stratum the load-bearing C1 gates on; C6 (step location "
            "invariant in K) is REPORTED, not gated. Measured cost on the "
            "authoring box: 128 banks in 0.38s (K=4) / 0.48s (K=8). "
            "SUBSTRATE-PATH OVERLAP (Step 2.5c): SD-E3-SCORER-COMPLETION is open "
            "with severity=degrading on ree_core/predictors/e3_selector.py -- "
            "this run executes that module, so it runs under that known "
            "limitation (its two untrained scorer heads are gated out of the E3 "
            "score by default, which does not bear on the eligibility-envelope or "
            "No-Go arithmetic measured here). The only open CORRUPTING entry "
            "touching anything this driver imports is mode-governance-engagement, "
            "via ree_core/utils/config.py; that entry's executed footprint is "
            "salience_coordinator.py / agent.py / regime_occupancy_gate.py, none "
            "of which this driver constructs or steps (it builds no agent and no "
            "salience coordinator), and config.py appears only as the declaration "
            "site of a knob this run never sets. Judged a declaration-site "
            "overlap, not an execution overlap; recorded here so the judgement is "
            "auditable rather than silent. "
            "PURPOSE=diagnostic, so this is excluded from confidence/conflict "
            "scoring by design and PROMOTES NOTHING. "
            "MECH-439 is DELIBERATELY NOT TAGGED (re-derive brake 12, "
            "ceiling_decision exhausted): this poses no F-dominance lever. "
            "MECH-260 is DELIBERATELY NOT TAGGED for V3-EXQ-926a's reason -- its "
            "falsifier is behavioural and vacuous at the MECH-457 competence "
            "floor. Re-derive brake NOT fired: MECH-449 = 0, ARC-107 = 0 against "
            "a threshold of 2, recomputed at queue time and matching the "
            "autopsy's own count. "
            "NOT marked `supersedes`: V3-EXQ-937a's PASS/supports stands with its "
            "basis corrected by the autopsy, and V3-EXQ-937's evidence stays "
            "active per its own driver's statement."
        ),
    }
    if not core_non_degenerate:
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = (
            "perseveration gate never engaged, suppression vector flat across "
            "candidates, one of the two per-bank envelope strata C1 contrasts "
            "was unpopulated, or ON and OFF arms converted identically"
        )

    # DRY-RUN OUTPUT DIR: a SUBDIRECTORY of the scratch dir, not the scratch dir
    # itself. This keeps V3-EXQ-937a's stronger guarantee -- a smoke manifest
    # NEVER lands under evidence/experiments/, so it can never reach the indexer
    # or pending_review.md even if the process dies before emit_outcome
    # (incident V3-EXQ-696) -- while working around a live defect in the
    # relocation helper it interacts with.
    #
    # THE DEFECT (found by this script's own smoke, 2026-08-19; pre-existing and
    # NOT introduced here): experiment_protocol._relocate_dry_run_manifest builds
    # `dest = <tempdir>/ree_dry_run_manifests/<name>` and runs
    # `if dest.exists(): dest.unlink()` before `shutil.move(src, dest)`. When the
    # driver has ALREADY written to that exact directory, src == dest, so the
    # unlink DELETES the manifest and the subsequent move fails on the missing
    # source -- swallowed by the helper's best-effort `except Exception`. The
    # smoke then reports a manifest path that does not exist. V3-EXQ-937a has the
    # same shape and the same silent loss. Writing one level down makes src and
    # dest genuinely distinct, so the helper's move does what it intends and
    # lands the manifest in the canonical scratch dir. Tracked separately as an
    # infrastructure chip; deliberately NOT fixed from this queue-experiment run,
    # since experiment_protocol.py is on every driver's emit path.
    out_dir = (
        Path(tempfile.gettempdir()) / "ree_dry_run_manifests" / EXPERIMENT_TYPE
        if dry_run else EVIDENCE_DIR
    )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=manifest.get("config"),
        seeds=seeds,
        script_path=Path(__file__),
        # Closes V3-EXQ-937a's missing always-core field: it omitted started_at
        # and its manifest carries elapsed_seconds: null.
        started_at=perf0,
    )
    manifest["manifest_path"] = str(out_path)
    print(
        f"[937b] outcome={outcome} label={label} "
        f"rate_env1={rate_pinned:.6f} (n={n_pinned}) "
        f"rate_env2plus={rate_open:.6f} (n={n_open}) "
        f"lift={per_seed_lift} step_at_2_by_k="
        f"{ {k: v['step_at_2'] for k, v in k_strata.items()} } "
        f"c6_worst_stratum_n={c6_worst_stratum_n} "
        f"shuffled_worst={shuf_worst_rate:.6f} empty_eligible={total_empty} "
        f"-> {out_path}",
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-937b MECH-449/ARC-107 per-bank (envelope size, converted) joint"
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
