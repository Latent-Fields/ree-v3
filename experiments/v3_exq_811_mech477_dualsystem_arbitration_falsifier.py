"""
V3-EXQ-811 -- MECH-477 dual-system uncertainty-arbitration falsifier (SD-081 OFF vs ON).

Claims: MECH-477 (primary), MECH-163 leg (1) (secondary)

WHAT MECH-477 ASSERTS. Differential recruitment between the habit pathway and the
hippocampally-planned pathway is produced by an explicit ARBITRATION element that reads
the two pathways' relative uncertainty and reallocates control -- NOT by the mere presence
of two pathways. Two pathways WITHOUT an arbitrator produce a FLAT recruitment response
regardless of context novelty.

THE SUBSTRATE EXISTS. SD-081 (e3.dualsystem_uncertainty_arbitration) landed in ree-v3 on
2026-07-22 (commit 4472811): an arbitration weight w = sigmoid(gain*(u_habit_n -
u_planned_n) + bias) over a depth-limited HABIT read and a full-horizon PLANNED read of
the same E3 scorer, applied upstream of raw_scores / score_bias / the commit gate / the
argmin. Before it, E3 scored candidates with an unconditional full-horizon J(zeta) -- no
weight existed anywhere between the myopic and the deep read, so nothing COULD respond to
novelty. This experiment is the two-arm OFF-vs-ON contrast MECH-477's what_would_answer
names. Design doc:
REE_assembly/docs/architecture/sd_081_dualsystem_uncertainty_arbitration.md

===========================================================================
WHY THE OFF ARM IS NOT V3-EXQ-786a AS-RUN
===========================================================================
MECH-477's what_would_answer says "the OFF arm is already measured: V3-EXQ-786a IS that
arm". That is no longer true, and the reason is recorded in SD-081's
evidence_quality_note: 786a's recruitment DV was itself DEGENERATE, confirmed by
measurement while SD-081 was being built.

786a computed recruitment = 1 - spearman(full, first) with first =
evaluate_trajectory(world_seq[:, :1, :]). Index 0 of the z_world sequence is the CURRENT
state, shared by every candidate, so that vector is CONSTANT -- measured n_unique = 1
across all 32 candidates, range exactly 0, on every scored tick under 786a's own config.
Its _spearman degeneracy guard could not fire, because it tests the std of the RANKS and
double-argsort of a constant vector is a permutation of 0..K-1 (std 9.23 at K=32). The DV
therefore measured stable-sort tie-break noise: simulated mean 1.0173 sd 0.1871 over 200
draws, against the manifest's reported seed-0 familiar recruitment_rate 1.01725 with
per-layout sds 0.149 / 0.207 / 0.165 / 0.190 -- a match to five significant figures.

So BOTH arms are freshly run here with a non-degenerate habit read at HABIT_DEPTH = 2
(the same floor SD-081 enforces inside _arbitrate_dual_system), and the readiness gate
below asserts the cross-candidate RANGE and the DISTINCT-VALUE FRACTION of the HABIT
vector -- not only the full-horizon vector. Gating only the latter is exactly why 786a
passed readiness with a degenerate DV. RANGE and distinctness, never magnitude (the
V3-EXQ-643 lesson).

Adjudicating 786a itself is /failure-autopsy and /governance work. Nothing here does it,
and this run marks nothing reviewed.

===========================================================================
WHAT IS HELD FIXED FROM 786a
===========================================================================
The familiarity manipulation (familiar size=10/hazards=3/resources=5/landmarks_b=2 vs
novel size=14/hazards=7/resources=2/landmarks_b=5, all locally observable in the agent's
5x5 field views), env_drift_prob = 0.0 in both conditions, the AUC manipulation-check bar
(substantive leg > 0.70 plus an above-chance SEM leg > 0.50), the pre-registered
familiarity query/update bandwidth pair (0.20 / 1.0), n = 8 seeds, and the power-derived
absolute margin 0.02 at SEM_K = 1.0. All of it lives in the canonical lineage module
experiments/_lib/baselines/mech477_dualsystem_arbitration.py, so the OFF arm is minted
reuse-eligible for a later different-driver iteration (include_driver_script_in_hash=False).

===========================================================================
THE MANDATORY MANIPULATION CHECK -- AND WHY ONE LEG OF IT IS AN IDENTITY
===========================================================================
MECH-477 makes it mandatory that the arbitration weight be shown to VARY WITH measured
uncertainty, otherwise a null is a readiness failure that scores nothing rather than a
refutation. `agent.e3.last_arbitration` is the paired series that check consumes, sampled
ONLY on a real E3 tick (it is None when the arbitrator is off and retains its previous
value between ticks; see the latch discipline in the lineage module).

The natural form -- rank correlation between w_planned and (u_habit_norm - u_planned_norm)
-- is measured and gated here at rho > 0.99. But it must be read for what it is:
w = sigmoid(gain * (u_habit_n - u_planned_n) + bias) is a STRICTLY MONOTONE function of
that difference, so the rank correlation is 1.0 BY ARITHMETIC. Measured on this config
during authoring: rho = 1.000000 exactly, in both conditions. It is a useful check -- it
would fall below 1.0 on a NaN, a clipped sigmoid, or a mixed-degenerate series -- but it
is a no-numerical-pathology check, NOT evidence that the arbitrator is responsive. This is
the V3-EXQ-604c DV-symmetry discipline applied to a readiness gate rather than to a DV.

The NON-VACUOUS legs, which carry the actual readiness claim:

  ARB-LIVE     -- >= 95% of ON-arm E3 ticks produce a non-degenerate arbitration.
  ARB-SOURCE   -- >= 95% of them resolve u_habit through the FAMILIARITY path, not the E1
                  novelty-EMA fallback. Not a stylistic preference: with
                  curiosity_weight = 0.0 the fallback was measured on this exact config
                  returning u_habit_raw = 0.0 on EVERY tick, so u_habit_norm is a constant
                  and the novelty channel -- the one MECH-477 is about -- never varies.
                  An ON arm running on the fallback would gate w through u_planned alone
                  and could not test the claim. (This is the "a gate whose statistic
                  annihilates a channel cannot speak for that channel" lesson, caught
                  before the run rather than after.)
  ARB-W-RANGE  -- the cross-tick RANGE of w_planned clears 0.05, i.e. at least a
                  5-percentage-point reallocation of control between the pathways.

WHY ARB-W-RANGE GATES THE NULL BRANCH ONLY (asymmetric, and deliberately so). A weight
that never moves cannot refute MECH-477: a flat DV under a frozen w is a readiness
failure, which is precisely what MECH-477's own wording attaches the mandatory check to.
But a weight range floor must NOT be allowed to void a POSITIVE result -- if the ON arm
demonstrably shifts the DV relative to OFF, the behaviour itself is stronger evidence of
liveness than any proxy on w, and voiding it on a proxy would be the V3-EXQ-785 vacating
failure in a new costume. So ARB-W-RANGE is declared with
applies_to = (ON arm) AND (C1 did not pass), and is recorded under `scoped_out` with its
reason when C1 passes. Every other precondition applies unconditionally.

MEASURED DURING AUTHORING, and why the floor is worth stating up front: at the registered
defaults the observed w sat near 0.83 with a cross-tick range of only ~0.002 over a short
probe. The per-pathway EMA normalisation (u / (u + ema), alpha 0.05) re-centres each
uncertainty at its own baseline on a ~20-tick timescale, so a SUSTAINED context change is
partly washed out and the arbitrator responds mostly to transients. If that holds at full
budget the run self-routes substrate_not_ready_requeue with the measured w range and the
u_habit / u_planned EMA dynamics in hand -- which hands the successor a
`puzzle (known rules)` (lower dualsystem_uncertainty_ema_alpha, or raise the gain) instead
of a bare "precondition unmet". The defaults are NOT re-tuned here: the claim under test
is about the registered substrate, and moving a threshold or a gain to reach a verdict is
what this discipline exists to prevent.

===========================================================================
DV-SYMMETRY DECLARATION (mandatory design-audit step, per arm)
===========================================================================
DV: recruitment = 1 - spearman(full_horizon_scores, habit_depth_scores), a rank
correlation. Its symmetry group is the MONOTONE TRANSFORMS of the candidate score vectors:
any uniform additive constant or positive monotone rescaling is invisible to it.

  arb_off -- the manipulation is the familiarity contrast (hazard / resource / landmark
             density). It changes the WORLD STATES candidates roll out through, hence
             their relative ordering under deep versus shallow scoring, which is not any
             monotone remap of a fixed score vector. NOT invariant. (786a's own
             declaration, inherited unchanged.)
  arb_on  -- the additional manipulation is the arbitration weight. NOTE the arbitrator
             does NOT enter the DV directly: the DV is computed from
             residue_field.evaluate_trajectory, not from E3's arbitrated scores. It acts
             on the DV SECOND-ORDER, through the committed action stream -> the states
             visited -> the candidate sets proposed on later ticks. That is a real causal
             path and not an arithmetic identity (a change in w changes the argmin, hence
             the trajectory, hence the candidate distribution), so the arm is not
             invariant -- but because the path is indirect, C2 below reads the allocation
             weight DIRECTLY as a secondary, non-load-bearing criterion. C1 remains the
             DID on recruitment that MECH-477's what_would_answer specifies.

===========================================================================
SCORING (pre-registered; MECH-477's own criteria)
===========================================================================
C1 (LOAD-BEARING, MECH-477 primary). SUPPORTED when BOTH:
    (a) the novel-minus-familiar recruitment delta is strictly greater ON than OFF on
        >= 2/3 of DIVERGENT seeds (a seed is divergent when |delta_on - delta_off| >
        0.005, a quarter of the absolute margin; >= 3 divergent seeds required, else the
        arms did not separate and (a) cannot be evaluated), AND
    (b) the ON-arm delta clears mean - 1.0*SEM > 0.02 (786a's power-derived absolute leg).
REFUTED when C1 fails, every unconditional precondition is green, ARB-W-RANGE is green,
and the ON arm reproduces a flat response: the ON-arm delta bar fails AND |Cohen's d| of
the ON-arm deltas < 0.1.
Anything else with green gates is `mixed` -- measured, but neither supporting nor cleanly
flat. Red gates route substrate_not_ready_requeue / non_contributory and weight NOTHING.

C2 (secondary, NOT load-bearing). The arbitrator's own allocation: mean w_planned on novel
layouts minus mean w_planned on familiar layouts, against a pre-registered floor of 0.01.
This is the most direct expression of MECH-477 available, and it is reported whatever C1
does.

MECH-163 leg (1) is read from the ON arm ONLY -- the substrate that CAN express
differential recruitment -- against 786a's own two legs (mean - SEM > 0.02 AND Cohen's
d >= 0.8). Its own gate is the ON arm's, not the run's, so a red OFF arm does not vacate
it (non_degenerate_per_claim). Legs (2) and (3) of MECH-163 are OUT OF SCOPE and unchanged
by anything here.

SLEEP: not used (no sleep flags set), so no SLEEP DRIVER line applies.
PHASED TRAINING: not required. SD-081 has NO learned parameters -- it is a read-only
weighting over existing scorers, deliberately not a learned gate (a learned gate would
confound "the arbitrator works" with "the gate trained"). Nothing here trains a head on
z_world / z_harm / any encoder output beyond warmup_train's own internal schedule.

ETHICS PREFLIGHT (documentation habit, non-enforced): involves_negative_valence false,
involves_suffering_like_state false, involves_self_model false,
involves_inescapability_or_helplessness false, involves_offline_replay_over_harm false,
involves_social_mind_or_language false, involves_human_data_or_clinical_context false,
decision: allow.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402

from _lib.arm_fingerprint import arm_cell  # noqa: E402
from _lib.baselines import mech477_dualsystem_arbitration as LIN  # noqa: E402
from _lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from _lib.robustness_bars import robust_by_sem  # noqa: E402
from pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_811_mech477_dualsystem_arbitration_falsifier"
CLAIM_IDS = ["MECH-477", "MECH-163"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ---------------------------------------------------------------------------
# Pre-registered constants. Thresholds are fixed HERE and never derived from the
# run's own statistics.
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

# --- C1, MECH-477 primary --------------------------------------------------
# Absolute leg inherited from 786a's power calculation: seed-level SD 0.0261
# (carried forward as a NOISE estimate only), n=8 -> SEM 0.0092, so the bar
# demands an ON-arm mean of 0.0292.
DIVERGENCE_MARGIN_ABS = 0.02
SEM_K = 1.0
SEM_MIN_N = 3
# A seed counts as DIVERGENT when the two arms' deltas actually differ. A quarter
# of the absolute margin: below this the arms produced the same number and the
# ON-greater-than-OFF question is not posed for that seed.
DIVERGENT_SEED_EPS = 0.005
MIN_DIVERGENT_SEEDS = 3
ON_GREATER_FRACTION_FLOOR = 2.0 / 3.0
# REFUTED leg: the ON arm reproduces the OFF arm's flat response.
FLAT_COHEN_D_CEIL = 0.1
# MECH-163 leg (1), read from the ON arm against 786a's own standardized leg.
COHEN_D_FLOOR = 0.8

# --- C2, the arbitrator's own allocation (secondary, not load-bearing) ------
W_NOVELTY_SHIFT_FLOOR = 0.01

# --- Readiness floors ------------------------------------------------------
FAMILIARITY_AUC_FLOOR = 0.70          # SUBSTANTIVE leg, d ~= 0.74
FAMILIARITY_AUC_CHANCE_FLOOR = 0.50   # ABOVE-CHANCE leg
AUC_SEM_K = 1.0
MIN_LAYOUTS_FOR_AUC = 3
SCORE_RANGE_FLOOR = LIN.SCORE_RANGE_FLOOR              # 1e-6
# The 786a defect stated as its own statistic: a constant vector has distinct
# fraction 1/K (0.031 at K=32), which NO rank-variance guard can see.
HABIT_DISTINCT_FRAC_FLOOR = 0.5
# CEILING. The SD-025 scorer lever must be inert: measured 0.0 exactly on this
# config (benefit_terrain_live_producer defaults False -> no benefit centers ->
# density 0 -> bonus 0). Fail-closed if it ever becomes unmeasurable.
CURIOSITY_LEVER_RANGE_CEIL = 1e-6
ARB_NONDEGENERATE_FRAC_FLOOR = 0.95
ARB_SOURCE_FAMILIARITY_FRAC_FLOOR = 0.95
ARB_W_UNCERTAINTY_RHO_FLOOR = 0.99
ARB_W_DYNAMIC_RANGE_FLOOR = 0.05

# Per-cell cap on the retained paired arbitration series (the summary statistics
# are computed over the FULL series; this is the auditable sample kept in the
# manifest so the file stays a few hundred KB rather than a few MB).
ARB_SAMPLE_CAP = 200


def _cohen_d(vals: List[float]) -> Optional[float]:
    """mean / SD of the per-seed deltas -- the standardized leg.

    Scales the effect on the SD of the DELTA itself, which is the quantity the
    between-seed bar is actually fighting.
    """
    xs = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if len(xs) < 2:
        return None
    sd = float(statistics.pstdev(xs))
    if sd == 0.0:
        return None
    return float(statistics.fmean(xs)) / sd


def _worst(vals: List[float], lo: bool = True) -> float:
    """The extremum, never a mean: the indexer recomputes `met` from the number
    reported, so a `measured` that averages while `met` quantifies over cells is
    not recomputable (a single out-of-band cell hides inside an in-band mean)."""
    xs = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not xs:
        return 0.0 if lo else float("inf")
    return min(xs) if lo else max(xs)


def _arb_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise one arm's paired arbitration series."""
    if not rows:
        return {
            "n": 0, "w_mean": None, "w_range": 0.0,
            "u_habit_norm_range": 0.0, "u_planned_norm_range": 0.0,
            "rho_w_vs_relative_uncertainty": 0.0,
            "rho_w_vs_u_habit_alone": None,
        }
    w = [r["w_planned"] for r in rows]
    uh = [r["u_habit_norm"] for r in rows]
    up = [r["u_planned_norm"] for r in rows]
    diff = [a - b for a, b in zip(uh, up)]
    rho = LIN.spearman(w, diff)
    rho_h = LIN.spearman(w, uh)
    return {
        "n": len(rows),
        "w_mean": float(np.mean(w)),
        "w_sd": float(np.std(w)),
        "w_range": float(max(w) - min(w)),
        "u_habit_norm_mean": float(np.mean(uh)),
        "u_habit_norm_range": float(max(uh) - min(uh)),
        "u_planned_norm_mean": float(np.mean(up)),
        "u_planned_norm_range": float(max(up) - min(up)),
        # 1.0 by arithmetic when the series is healthy -- see the module
        # docstring. A no-numerical-pathology check, not a responsiveness one.
        "rho_w_vs_relative_uncertainty": float(rho) if rho is not None else 0.0,
        # NON-GATING. The responsiveness of w to the NOVELTY channel specifically.
        "rho_w_vs_u_habit_alone": float(rho_h) if rho_h is not None else None,
    }


# ---------------------------------------------------------------------------
# Preconditions. Regime-conditioned via applies_to -- never ANDed whole-run
# (the V3-EXQ-785 vacating failure).
# ---------------------------------------------------------------------------
def _is_on(ctx: Dict[str, Any]) -> bool:
    return bool(ctx.get("is_on"))


def _is_on_null_branch(ctx: Dict[str, Any]) -> bool:
    return bool(ctx.get("is_on")) and not bool(ctx.get("c1_passed"))


PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="familiarity_discriminability_auc",
        description=(
            "MANIPULATION CHECK, substantive leg (V3-EXQ-786a's, held fixed). Rank "
            "discriminability (AUC) of practiced vs held-out layout familiarity, mean "
            "across seeds, read from the experiment-owned tracker at the pre-registered "
            "query bandwidth 0.20. AUC is invariant under any monotone transform of the "
            "readout, so it measures the manipulation and not the instrument's gain, and "
            "it has a principled null at 0.5."
        ),
        control=(
            "practiced vs held-out layouts differing in generating parameters "
            "(size 10->14, hazards 3->7, resources 5->2, landmarks_b 2->5), all locally "
            "observable in the agent's 5x5 field views, drift suppressed in both"
        ),
        threshold=FAMILIARITY_AUC_FLOOR,
        direction="lower",
        structural_max=lambda ctx: 1.0,
    ),
    PreconditionSpec(
        name="familiarity_discriminability_above_chance",
        description=(
            "MANIPULATION CHECK, above-chance leg. The SEM lower bound of the same AUC "
            "must beat the 0.5 null across seeds, so the discrimination is reliable and "
            "not a large point estimate carried by one seed."
        ),
        control="same layout populations as the substantive leg",
        threshold=FAMILIARITY_AUC_CHANCE_FLOOR,
        direction="lower",
        structural_max=lambda ctx: 1.0,
    ),
    PreconditionSpec(
        name="full_score_range_non_degenerate",
        description=(
            "Cross-candidate RANGE of the FULL-HORIZON score, worst scored tick (never a "
            "mean). The DV routes on a rank correlation over candidates, which is "
            "arbitrary noise unless the candidates are separable. RANGE, not magnitude "
            "(V3-EXQ-643): a uniform per-tick offset has large magnitude and ~0 range."
        ),
        control="candidate set on a real E3 tick after the practice phase",
        threshold=SCORE_RANGE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="habit_score_range_non_degenerate",
        description=(
            "THE GATE V3-EXQ-786a LACKED. Cross-candidate RANGE of the HABIT (depth-2) "
            "score, worst scored tick. 786a gated only the full-horizon vector, which is "
            "why it passed readiness while its depth-1 habit vector was constant (range "
            "exactly 0, n_unique 1 over 32 candidates on every tick)."
        ),
        control="candidate set on a real E3 tick after the practice phase",
        threshold=SCORE_RANGE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="habit_score_distinct_fraction",
        description=(
            "The 786a defect stated as its OWN statistic: the fraction of DISTINCT values "
            "in the habit score vector, worst scored tick. A constant vector reads 1/K "
            "(0.031 at K=32) here while its RANK std is 9.23 -- which is exactly why "
            "786a's _spearman rank-variance guard could not fire. Range and distinctness "
            "are gated together because they fail independently under float quantisation."
        ),
        control="candidate set on a real E3 tick after the practice phase",
        threshold=HABIT_DISTINCT_FRAC_FLOOR,
        direction="lower",
        structural_max=lambda ctx: 1.0,
    ),
    PreconditionSpec(
        name="curiosity_scorer_lever_inert",
        description=(
            "CEILING. curiosity_weight > 0 is required to build the familiarity tracker "
            "the arbitrator reads, but it also makes the CEM scorer novelty-sensitive "
            "(SD-025), which would make the within-arm MECH-163 leg-(1) reading circular. "
            "Measured rather than assumed: the bonus is proportional to the SD-024 "
            "representational density and benefit_terrain_live_producer defaults False, "
            "so the benefit map is unpopulated and the CROSS-CANDIDATE range is 0.0 "
            "exactly on this config. RANGE, not magnitude -- a uniform bonus is invariant "
            "under the CEM argmin and could not reorder anything anyway. Fail-closed: an "
            "unmeasurable bonus reports +inf and refuses the run."
        ),
        control="all candidates on the first scored E3 tick of each probe layout",
        threshold=CURIOSITY_LEVER_RANGE_CEIL,
        direction="upper",
        structural_min=lambda ctx: 0.0,
    ),
    PreconditionSpec(
        name="arbitration_live_non_degenerate",
        description=(
            "ARB-LIVE. Fraction of ON-arm E3 ticks producing a non-degenerate arbitration "
            "(E3Selector flags degenerate on no_habit_uncertainty / insufficient_candidates "
            "/ degenerate_score_spread / non_finite_scores and returns the planned scores "
            "unchanged). An ON arm that mostly declined to arbitrate is an OFF arm."
        ),
        control="ON arm only; e3.last_arbitration cleared before every select_action",
        threshold=ARB_NONDEGENERATE_FRAC_FLOOR,
        direction="lower",
        applies_to=_is_on,
        applies_note="the OFF arm performs no arbitration by construction (last_arbitration is None)",
        structural_max=lambda ctx: 1.0,
    ),
    PreconditionSpec(
        name="arbitration_habit_uncertainty_source_is_familiarity",
        description=(
            "ARB-SOURCE. Fraction of ON-arm arbitrations resolving u_habit through the "
            "FAMILIARITY path rather than the E1 novelty-EMA fallback. Not a preference: "
            "with curiosity_weight = 0.0 the fallback was measured on this exact config "
            "returning u_habit_raw = 0.0 on EVERY tick, so u_habit_norm is constant and "
            "the novelty channel MECH-477 is about never varies -- w would then be gated "
            "by u_planned alone and the claim would be untested."
        ),
        control="ON arm only; habit_uncertainty_source recorded per arbitration",
        threshold=ARB_SOURCE_FAMILIARITY_FRAC_FLOOR,
        direction="lower",
        applies_to=_is_on,
        applies_note="no u_habit is resolved in the OFF arm",
        structural_max=lambda ctx: 1.0,
    ),
    PreconditionSpec(
        name="arbitration_weight_uncertainty_coupling",
        description=(
            "MECH-477's mandated form of the manipulation check: rank correlation between "
            "w_planned and (u_habit_norm - u_planned_norm) across scored ON ticks. READ "
            "IT FOR WHAT IT IS -- w = sigmoid(gain*diff + bias) is strictly monotone in "
            "that difference, so rho is 1.0 BY ARITHMETIC (measured 1.000000 during "
            "authoring). It detects a NaN, a clipped sigmoid or a mixed-degenerate series; "
            "it is NOT evidence of responsiveness. ARB-LIVE / ARB-SOURCE / ARB-W-RANGE "
            "carry that."
        ),
        control="ON arm only; full paired series, not a sample",
        threshold=ARB_W_UNCERTAINTY_RHO_FLOOR,
        direction="lower",
        applies_to=_is_on,
        applies_note="no arbitration series exists in the OFF arm",
        structural_max=lambda ctx: 1.0,
    ),
    PreconditionSpec(
        name="arbitration_weight_dynamic_range",
        description=(
            "ARB-W-RANGE, NULL-QUALIFYING ONLY. Cross-tick range of w_planned: at least a "
            "5-percentage-point reallocation of control between the two pathways. A "
            "weight that never moves cannot REFUTE MECH-477 -- a flat DV under a frozen w "
            "is a readiness failure, which is what MECH-477's mandatory check attaches to. "
            "But it must not void a POSITIVE result either: if the ON arm demonstrably "
            "shifts the DV relative to OFF, the behaviour is stronger evidence of liveness "
            "than any proxy on w, and voiding that on a proxy would be the V3-EXQ-785 "
            "vacating failure in a new costume. Hence applies_to = ON arm AND C1 failed."
        ),
        control="ON arm only, and only when C1 did not pass",
        threshold=ARB_W_DYNAMIC_RANGE_FLOOR,
        direction="lower",
        applies_to=_is_on_null_branch,
        applies_note=(
            "scoped out because C1 passed: a demonstrated behavioural shift is a stronger "
            "liveness certificate than the weight-range proxy, and gating a positive "
            "result on the proxy would vacate a real finding"
        ),
        structural_max=lambda ctx: 1.0,
    ),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _arm_measured(cells: List[Dict[str, Any]], arb: Dict[str, Any]) -> Dict[str, float]:
    """Every precondition's measured value for one arm.

    Scoped-out preconditions still get a value (evaluate_arm_gate ignores them);
    supplying all of them keeps a coding slip from silently reading as a scoped-out
    precondition instead of raising.
    """
    aucs = [c["familiarity_auc"] for c in cells if c["familiarity_auc"] is not None]
    auc_bar = robust_by_sem(
        aucs, margin=FAMILIARITY_AUC_CHANCE_FLOOR, k=AUC_SEM_K, min_n=SEM_MIN_N
    )
    auc_mean = float(auc_bar.get("mean", 0.0)) if aucs else 0.0
    auc_lower = auc_mean - AUC_SEM_K * float(auc_bar.get("sem", 0.0)) if aucs else 0.0
    n_arb = int(arb.get("n_ticks", 0))
    return {
        "familiarity_discriminability_auc": auc_mean if len(aucs) >= SEM_MIN_N else 0.0,
        "familiarity_discriminability_above_chance": auc_lower if len(aucs) >= SEM_MIN_N else 0.0,
        "full_score_range_non_degenerate": _worst([c["min_full_score_range"] for c in cells]),
        "habit_score_range_non_degenerate": _worst([c["min_habit_score_range"] for c in cells]),
        "habit_score_distinct_fraction": _worst([c["min_habit_distinct_frac"] for c in cells]),
        "curiosity_scorer_lever_inert": _worst(
            [c["curiosity_bonus_range_max"] for c in cells], lo=False
        ),
        "arbitration_live_non_degenerate": (
            float(arb["n_nondegenerate"]) / float(n_arb) if n_arb else 0.0
        ),
        "arbitration_habit_uncertainty_source_is_familiarity": (
            float(arb["n_familiarity_source"]) / float(n_arb) if n_arb else 0.0
        ),
        "arbitration_weight_uncertainty_coupling": float(
            arb["stats"]["rho_w_vs_relative_uncertainty"]
        ),
        "arbitration_weight_dynamic_range": float(arb["stats"]["w_range"]),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    dims = LIN.assert_dims_match()

    seeds = SEEDS[:2] if dry_run else SEEDS
    practice = 2 if dry_run else LIN.PRACTICE_EPISODES_PER_LAYOUT
    probe = 1 if dry_run else LIN.PROBE_EPISODES_PER_LAYOUT
    accum = 2 if dry_run else LIN.ACCUM_EPISODES_PER_LAYOUT

    # Design-time refusal, BEFORE compute is spent: no precondition may be
    # unsatisfiable from this design's own pre-registered values.
    arm_contexts = [
        {"arm_id": a, "is_on": a == LIN.ARM_ON, "c1_passed": False} for a in LIN.ARMS
    ]
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_contexts)

    # --- measurement -------------------------------------------------------
    cells_by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in LIN.ARMS}
    arb_by_arm: Dict[str, Dict[str, Any]] = {}
    for arm_id in LIN.ARMS:
        arb_rows_all: List[Dict[str, Any]] = []
        n_ticks = n_nondeg = n_fam = n_latched = 0
        for seed in seeds:
            with arm_cell(
                seed,
                config_slice=LIN.arm_config_slice(arm_id),
                script_path=Path(__file__),
                config_slice_declared=True,
                # MINT AS YOU GO: excluding the driver from the hash is what lets a
                # later, different-driver iteration reuse this cell (plan 9.4/9.7).
                # The cell's whole computation lives in the lineage module under
                # experiments/_lib/**, which IS in the substrate glob, so any change
                # to what a cell computes still refuses a stale reuse.
                include_driver_script_in_hash=False,
            ) as cell:
                row = LIN.run_cell(
                    seed, arm_id,
                    practice_episodes=practice,
                    probe_episodes=probe,
                    accum_episodes=accum,
                )
                cell.stamp(row)

            for cond in ("familiar", "novel"):
                block = row["conditions"][cond]
                rows = block.pop("_arb_rows")
                arb_rows_all.extend(rows)
                n_nondeg += len(rows)
                n_ticks += len(rows) + int(block["arb_n_degenerate"])
                n_fam += int(block["arb_sources"].get("familiarity", 0))
                n_latched += int(block["n_latched_ticks"])
                block["arb_sample"] = rows[:ARB_SAMPLE_CAP]
                block["arb_stats"] = _arb_stats(rows)

            delta = row["recruitment_delta"]
            print(
                "verdict: {}".format(
                    "PASS" if (delta is not None and delta > DIVERGENCE_MARGIN_ABS) else "FAIL"
                ),
                flush=True,
            )
            cells_by_arm[arm_id].append(row)

        arb_by_arm[arm_id] = {
            "n_ticks": n_ticks,
            "n_nondegenerate": n_nondeg,
            "n_familiarity_source": n_fam,
            # The true denominator convention: a held/latched tick recorded NOTHING.
            "n_latched_ticks": n_latched,
            "stats": _arb_stats(arb_rows_all),
        }

    off_cells = cells_by_arm[LIN.ARM_OFF]
    on_cells = cells_by_arm[LIN.ARM_ON]

    # --- C1: the difference-in-differences MECH-477 specifies ---------------
    per_seed_pairs: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        d_off = off_cells[i]["recruitment_delta"]
        d_on = on_cells[i]["recruitment_delta"]
        divergent = (
            d_off is not None and d_on is not None
            and abs(d_on - d_off) > DIVERGENT_SEED_EPS
        )
        per_seed_pairs.append({
            "seed": seed,
            "delta_off": d_off,
            "delta_on": d_on,
            "delta_on_minus_off": (d_on - d_off) if (d_off is not None and d_on is not None) else None,
            "divergent": bool(divergent),
            "on_greater": bool(divergent and d_on > d_off),
        })

    divergent_pairs = [p for p in per_seed_pairs if p["divergent"]]
    n_divergent = len(divergent_pairs)
    n_on_greater = sum(1 for p in divergent_pairs if p["on_greater"])
    on_greater_frac = (float(n_on_greater) / float(n_divergent)) if n_divergent else 0.0

    deltas_on = [c["recruitment_delta"] for c in on_cells if c["recruitment_delta"] is not None]
    deltas_off = [c["recruitment_delta"] for c in off_cells if c["recruitment_delta"] is not None]
    on_bar = robust_by_sem(deltas_on, margin=DIVERGENCE_MARGIN_ABS, k=SEM_K, min_n=SEM_MIN_N)
    off_bar = robust_by_sem(deltas_off, margin=DIVERGENCE_MARGIN_ABS, k=SEM_K, min_n=SEM_MIN_N)
    d_on_cohen = _cohen_d(deltas_on)
    d_off_cohen = _cohen_d(deltas_off)

    c1_seed_leg = bool(n_divergent >= MIN_DIVERGENT_SEEDS
                       and on_greater_frac >= ON_GREATER_FRACTION_FLOOR)
    c1_magnitude_leg = bool(on_bar.get("passes", False))
    c1_passed = bool(c1_seed_leg and c1_magnitude_leg)

    # --- C2: the arbitrator's own allocation (secondary) --------------------
    w_fam = [c["conditions"]["familiar"]["arb_stats"]["w_mean"] for c in on_cells]
    w_nov = [c["conditions"]["novel"]["arb_stats"]["w_mean"] for c in on_cells]
    w_shifts = [
        (n - f) for f, n in zip(w_fam, w_nov) if f is not None and n is not None
    ]
    w_shift_mean = float(statistics.fmean(w_shifts)) if w_shifts else None
    c2_passed = bool(w_shift_mean is not None and w_shift_mean > W_NOVELTY_SHIFT_FLOOR)

    # --- per-arm gates ------------------------------------------------------
    arm_gates = []
    for arm_id in LIN.ARMS:
        ctx = {"arm_id": arm_id, "is_on": arm_id == LIN.ARM_ON, "c1_passed": c1_passed}
        arm_gates.append(evaluate_arm_gate(
            arm_id, ctx, PRECONDITION_SPECS,
            _arm_measured(cells_by_arm[arm_id], arb_by_arm[arm_id]),
        ))
    agg = aggregate_arm_gates(arm_gates)
    gate_by_arm = {g["arm"]: g for g in arm_gates}
    off_green = bool(gate_by_arm[LIN.ARM_OFF]["gate_green"])
    on_green = bool(gate_by_arm[LIN.ARM_ON]["gate_green"])
    both_green = bool(off_green and on_green)

    # --- routing ------------------------------------------------------------
    # C1's DV is a CROSS-ARM contrast, so unlike V3-EXQ-785's independent regimes
    # a red arm genuinely vacates it -- `any arm green` is the wrong aggregation
    # for a difference-in-differences. MECH-163's leg-(1) read needs the ON arm
    # only, so it is carried separately via non_degenerate_per_claim rather than
    # being dragged down with the run.
    refuted = bool(
        both_green
        and not c1_passed
        and not on_bar.get("passes", False)
        and d_on_cohen is not None
        and abs(d_on_cohen) < FLAT_COHEN_D_CEIL
    )
    mech163_supported = bool(
        on_green and on_bar.get("passes", False)
        and d_on_cohen is not None and d_on_cohen >= COHEN_D_FLOOR
    )
    mech163_weakened = bool(
        on_green and not on_bar.get("passes", False)
        and d_on_cohen is not None and abs(d_on_cohen) < FLAT_COHEN_D_CEIL
    )

    if not both_green:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        per_claim = {"MECH-477": "non_contributory", "MECH-163": "non_contributory"}
        if on_green:
            per_claim["MECH-163"] = (
                "supports" if mech163_supported
                else ("weakens" if mech163_weakened else "mixed")
            )
    elif c1_passed:
        outcome = "PASS"
        label = "arbitration_produces_differential_recruitment"
        direction = "supports"
        per_claim = {
            "MECH-477": "supports",
            "MECH-163": "supports" if mech163_supported else "mixed",
        }
    elif refuted:
        outcome = "FAIL"
        label = "arbitration_live_but_recruitment_flat"
        direction = "weakens"
        per_claim = {
            "MECH-477": "weakens",
            "MECH-163": "weakens" if mech163_weakened else "mixed",
        }
    else:
        outcome = "FAIL"
        label = "arbitration_live_effect_equivocal"
        direction = "mixed"
        per_claim = {
            "MECH-477": "mixed",
            "MECH-163": (
                "supports" if mech163_supported
                else ("weakens" if mech163_weakened else "mixed")
            ),
        }

    criteria_nd = arm_criteria_non_degenerate(
        {LIN.ARM_OFF: [], LIN.ARM_ON: ["C2_arbitration_weight_novelty_shift"]}, agg
    )
    criteria_nd["C1_recruitment_delta_greater_with_arbitrator"] = both_green

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": bool(dry_run),
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "arm_results": off_cells + on_cells,
        "per_seed_results": per_seed_pairs,
        "recruitment_deltas_off": deltas_off,
        "recruitment_deltas_on": deltas_on,
        "robustness_bar_on": on_bar,
        "robustness_bar_off": off_bar,
        "cohen_d_delta_on": d_on_cohen,
        "cohen_d_delta_off": d_off_cohen,
        "arbitration_summary": {a: arb_by_arm[a] for a in LIN.ARMS},
        "per_arm_gate": agg["per_arm_gate"],
        "observation_dims": dims,
        "criteria": [
            {
                "name": "C1_recruitment_delta_greater_with_arbitrator",
                "load_bearing": True,
                "passed": c1_passed,
                "seed_leg_passed": c1_seed_leg,
                "magnitude_leg_passed": c1_magnitude_leg,
                "n_divergent_seeds": n_divergent,
                "n_on_greater": n_on_greater,
                "on_greater_fraction": on_greater_frac,
            },
            {
                "name": "C2_arbitration_weight_novelty_shift",
                "load_bearing": False,
                "passed": c2_passed,
                "measured": w_shift_mean,
                "threshold": W_NOVELTY_SHIFT_FLOOR,
                "note": (
                    "the arbitrator's OWN allocation, novel minus familiar mean w_planned. "
                    "The most direct expression of MECH-477 available; secondary because "
                    "MECH-477's what_would_answer specifies the recruitment DID as the "
                    "criterion. Reported whatever C1 does."
                ),
            },
        ],
        "interpretation": {
            "label": label,
            "preconditions": agg["adjudication_preconditions"],
            "criteria_non_degenerate": criteria_nd,
            "preconditions_scope_note": agg["per_arm_gate"]["preconditions_scope_note"],
        },
        # Top-level keeps the module's `any arm green` semantics so a red OFF arm
        # cannot vacate the ON arm's own MECH-163 leg-(1) read. The cross-arm
        # vacating that C1 genuinely DOES require -- it is a difference-in-
        # differences, so unlike V3-EXQ-785's independent regimes a red arm really
        # does destroy it -- is expressed per-claim and per-criterion instead, which
        # is the channel the indexer honours without dropping the whole run.
        "non_degenerate": bool(agg["non_degenerate"]),
        "non_degenerate_per_claim": {"MECH-477": both_green, "MECH-163": on_green},
        "degeneracy_reason": (
            "" if both_green else (
                "MECH-477's C1 is a CROSS-ARM contrast, so a red arm vacates it even when "
                "the other arm is green; MECH-163 leg (1) needs the ON arm only. "
                + str(agg.get("degeneracy_reason", ""))
            )
        ),
        "diagnostics": {
            "familiarity_separation_raw_off": [
                c["familiarity_separation_raw"] for c in off_cells
            ],
            "familiarity_separation_raw_on": [
                c["familiarity_separation_raw"] for c in on_cells
            ],
            "v3_exq_786_achievable_separation": 0.049365,
            "worst_familiarity_pinned_high_frac": max(
                [c["worst_pinned_high_frac"] for c in off_cells + on_cells] or [0.0]
            ),
            "worst_familiarity_pinned_low_frac": max(
                [c["worst_pinned_low_frac"] for c in off_cells + on_cells] or [0.0]
            ),
            "arbitration_w_mean_familiar_per_seed": w_fam,
            "arbitration_w_mean_novel_per_seed": w_nov,
            "arbitration_w_novelty_shift_per_seed": w_shifts,
            "rho_w_vs_u_habit_alone": (
                arb_by_arm[LIN.ARM_ON]["stats"]["rho_w_vs_u_habit_alone"]
            ),
            "coupling_check_is_an_identity_note": (
                "arbitration_weight_uncertainty_coupling is expected to read 1.000000: "
                "w = sigmoid(gain*(u_habit_n - u_planned_n) + bias) is strictly monotone "
                "in its own argument, so the rank correlation is fixed by arithmetic. It "
                "is a no-numerical-pathology check. ARB-LIVE / ARB-SOURCE / ARB-W-RANGE "
                "carry the responsiveness claim."
            ),
            "e1_novelty_fallback_is_degenerate_note": (
                "Measured on this config during authoring: with curiosity_weight = 0.0 the "
                "arbitrator falls back to the E1 novelty EMA and u_habit_raw reads exactly "
                "0.0 on every tick, so u_habit_norm is constant and w is gated by "
                "u_planned alone. That is why curiosity_weight > 0 is a precondition for "
                "testing MECH-477 rather than a stylistic preference, and why ARB-SOURCE "
                "gates on the familiarity path."
            ),
        },
        "pre_registered": {
            "seeds": seeds,
            "arms": list(LIN.ARMS),
            "habit_depth": LIN.HABIT_DEPTH,
            "curiosity_weight": LIN.CURIOSITY_WEIGHT,
            "dualsystem_arbitration_gain": LIN.ARB_GAIN,
            "dualsystem_arbitration_bias": LIN.ARB_BIAS,
            "dualsystem_uncertainty_ema_alpha": LIN.ARB_UNCERTAINTY_EMA_ALPHA,
            "divergence_margin_abs": DIVERGENCE_MARGIN_ABS,
            "sem_k": SEM_K,
            "sem_min_n": SEM_MIN_N,
            "divergent_seed_eps": DIVERGENT_SEED_EPS,
            "min_divergent_seeds": MIN_DIVERGENT_SEEDS,
            "on_greater_fraction_floor": ON_GREATER_FRACTION_FLOOR,
            "flat_cohen_d_ceil": FLAT_COHEN_D_CEIL,
            "cohen_d_floor_mech163": COHEN_D_FLOOR,
            "w_novelty_shift_floor": W_NOVELTY_SHIFT_FLOOR,
            "familiarity_auc_floor": FAMILIARITY_AUC_FLOOR,
            "familiarity_auc_chance_floor": FAMILIARITY_AUC_CHANCE_FLOOR,
            "score_range_floor": SCORE_RANGE_FLOOR,
            "habit_distinct_frac_floor": HABIT_DISTINCT_FRAC_FLOOR,
            "curiosity_lever_range_ceil": CURIOSITY_LEVER_RANGE_CEIL,
            "arb_nondegenerate_frac_floor": ARB_NONDEGENERATE_FRAC_FLOOR,
            "arb_source_familiarity_frac_floor": ARB_SOURCE_FAMILIARITY_FRAC_FLOOR,
            "arb_w_uncertainty_rho_floor": ARB_W_UNCERTAINTY_RHO_FLOOR,
            "arb_w_dynamic_range_floor": ARB_W_DYNAMIC_RANGE_FLOOR,
            "familiarity_query_bandwidth": LIN.FAMILIARITY_QUERY_BANDWIDTH,
            "familiarity_update_bandwidth": LIN.FAMILIARITY_UPDATE_BANDWIDTH,
            "power_calculation": (
                "Inherited from V3-EXQ-786a: its seed-level SD = SEM 0.0117 * sqrt(5) = "
                "0.0261, carried forward as a NOISE estimate only. At n=8, SEM = 0.0092, "
                "so the absolute leg mean - 1.0*SEM > 0.02 demands an ON-arm mean of "
                "0.0292. 786a's own 0.0133 mean is NOT used as an effect-size estimate: "
                "it was measured on a degenerate DV and is a measurement of nothing."
            ),
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
        "scope_note": (
            "Primary claim MECH-477 (uncertainty-based control arbitration). MECH-163 is "
            "read for leg (1) ONLY, from the ON arm, and a result here says nothing about "
            "leg (2) long-horizon benefit accumulation (blocked by ARC-007 STRICT "
            "value-flat proposals, now MECH-478) or leg (3) prosocial planning (no V3 "
            "social substrate, now MECH-479). ARC-071 (planned->habit TRANSFER) is a "
            "distinct mechanism from this allocation one and is not exercised. This run "
            "does NOT adjudicate V3-EXQ-786a and marks nothing reviewed."
        ),
    }

    manifest["_full_config"] = {
        "familiar_env_kwargs": LIN.FAMILIAR_ENV_KWARGS,
        "novel_env_kwargs": LIN.NOVEL_ENV_KWARGS,
        "familiar_env_seeds": LIN.FAMILIAR_ENV_SEEDS,
        "novel_env_seeds": LIN.NOVEL_ENV_SEEDS,
        "steps_per_episode": LIN.STEPS_PER_EPISODE,
        "practice_episodes_per_layout": practice,
        "probe_episodes_per_layout": probe,
        "accum_episodes_per_layout": accum,
        "total_training_episodes": practice * len(LIN.FAMILIAR_ENV_SEEDS),
        "arm_config_slices": {a: LIN.arm_config_slice(a) for a in LIN.ARMS},
        "lineage": LIN.LINEAGE,
    }
    manifest["_seeds"] = seeds
    manifest["_started_at"] = t0
    return manifest


def _out_dir() -> Path:
    return (_ROOT.parent / "REE_assembly" / "evidence" / "experiments").resolve()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = run_experiment(dry_run=args.dry_run)

    full_config = manifest.pop("_full_config")
    seeds_used = manifest.pop("_seeds")
    started_at = manifest.pop("_started_at")
    # write_flat_manifest stamps the Recording Standard always-core
    # (recording_schema / substrate_hash / machine / machine_class /
    # elapsed_seconds / config / seeds) AFTER arm_results is assembled, so the
    # top-level substrate_hash HOISTS from the per-cell fingerprints.
    out_path = write_flat_manifest(
        manifest,
        _out_dir(),
        dry_run=args.dry_run,
        config=full_config,
        seeds=seeds_used,
        script_path=Path(__file__),
        started_at=started_at,
    )

    print(f"outcome: {manifest['outcome']}", flush=True)
    print(f"evidence_direction: {manifest.get('evidence_direction')}", flush=True)
    print(f"manifest: {out_path}", flush=True)

    _raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run=args.dry_run,
    )
