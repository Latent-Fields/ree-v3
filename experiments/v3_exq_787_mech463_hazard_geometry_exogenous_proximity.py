"""
V3-EXQ-787 -- MECH-463 / hazard-geometry: does EXOGENOUS hazard proximity drive
committed-selection candidate geometry?

TARGET: the hypothesis-space ledger leg
    question `arousal-variance-amplifier` :: `H-endogenous-hazard-geometry`
(REE_assembly/evidence/planning/hypothesis_space_registry.v1.json), the ONE hypothesis
left `alive` after V3-EXQ-785a eliminated the other two. Full adjudication:
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-785_2026-07-19.md section 9a.

WHY A NEW NUMBER, NOT A LETTER
  785 / 785a asked "does global scalar arousal amplify or concentrate selection
  variance?". This asks "does HAZARD PROXIMITY drive committed-selection candidate
  geometry?". The mechanism under test changed, so per the EXQ versioning policy this
  is a new number rather than a lettered iteration.

BACKGROUND (established, not re-derived here)
  V3-EXQ-785 measured urgency ENDOGENOUSLY and reported a striking profile: total
  cross-candidate score variance up 14.1x with urgency, incumbent share falling
  0.970 -> 0.831 (rho -0.83). V3-EXQ-785a re-ran with urgency set EXOGENOUSLY (i.i.d.
  uniform over a pre-registered grid) and BOTH halves vanished: var_total fold 0.970
  (rho -0.086), incumbent share 0.9375 -> 0.9411 (gap +0.0036, under 1 SE). So 785's
  entire profile was attributable to conditioning on an endogenous variable.

  Hazard proximity is the NAMED candidate for what actually drove it -- high-urgency
  ticks were near-hazard ticks -- but it is NOT confirmed, and the reason is the whole
  design driver for this run. Re-analysis of the 1757 committed rows embedded in the
  785a manifest at custom_information.per_tick_rows:

    POOLED across seeds, hazard proximity appears to reproduce 785 exactly:
        corr(hazard_prox_mean, incumbent share)  = -0.187
        corr(hazard_prox_mean, log10 var_total)  = +0.171     (same signs as 785)

    WITHIN each seed it vanishes and mostly REVERSES:
        per-seed r(hazard, share)  = [+0.089, +0.076, +0.041, -0.125, +0.135]
        per-seed tertile share gap = [+0.011, +0.012, +0.005, -0.014, +0.014]

    Across the 5 seed means: r(mean hazard, mean share) = -0.78 and
    r(mean hazard, mean var) = +0.77, carried largely by seed 0 (mean share 0.833,
    mean var 4.47e-05, ~7x the other seeds).

  That is Simpson's paradox on n=5. The pooled statistic is between-seed heterogeneity
  masquerading as a within-seed effect. Note also that seed 0's mean share of 0.833 is
  near-identical to 785's endpoint of 0.831 -- the ledger explicitly asks whoever picks
  this up to understand that coincidence, so D4 below tests it.

WHAT THIS RUN DOES DIFFERENTLY (both fixes the ledger named, not one)
  (a) It MANIPULATES hazard proximity instead of observing it, making the covariate
      exogenous the way 785a made urgency exogenous.
  (b) It uses a WITHIN-SEED design with 24 seeds instead of 5, so within- and
      between-seed variation are separable. The LOAD-BEARING statistic is the mean of
      per-seed within-seed slopes. The pooled tick-level correlation -- precisely the
      statistic that misleads here -- is recorded as an explicitly-labelled DECOY and
      routes nothing.

THE MANIPULATION
  Env: CausalGridWorldV2(use_proxy_fields=True, seed=seed, hazard_harm=0.5,
                         num_hazards=1, env_drift_prob=0.0)

  num_hazards=1 is REQUIRED, not incidental. hazard_field_view is normalised by the
  GLOBAL grid max (causal_grid_world.py:3146), so with more than one hazard the same
  true distance yields different observed values depending on how the hazards cluster.
  With exactly one hazard the max is always 1.0 at the hazard cell, which makes the
  view a clean absolute distance code and the levels comparable across conditions.

  env_drift_prob=0.0 freezes hazards. The default 0.3 drifts each hazard every
  env_drift_interval=5 steps (_drift_hazards, causal_grid_world.py:4435), which would
  let the manipulation decay inside its own assignment window.

  Assignment: a pre-registered Manhattan-distance grid, drawn i.i.d. per assignment
  window from np.random.default_rng(20_000 + seed), independent of state. The field
  falloff is 1/(1 + dist * hazard_field_decay) with decay 0.5, so the agent's own cell
  reads {0.222, 0.286, 0.400, 0.500, 0.667} across the grid -- a 3.0x exogenous span.
  The grid is ordered FAR -> NEAR so that level index increases with PROXIMITY; under
  that ordering H predicts a POSITIVE slope on log10 var_total and a NEGATIVE slope on
  incumbent share, matching the signs of the 785 profile. (Ordering it by distance
  instead would flip both and invite a sign error at adjudication.)

  Mechanics: at every env step inside a window the single hazard is RE-PINNED to
  Manhattan distance d from the agent's CURRENT cell, with the bearing drawn i.i.d. at
  each re-pin so no fixed direction is confounded with the level, followed by
  env._compute_proximity_fields() so the view reflects it immediately. This uses the
  in-place relocate pattern from v3_exq_495a_mech163_planned_system_gate.py:363 (clear
  the old grid cell, mutate the env.hazards entry, set the new cell) rather than
  reset_to() mid-episode, which would wipe health/energy and rebuild the grid.
  reset_to((5,5), [(hx,hy)]) is used only at episode start. Re-pinning holds the
  REALIZED distance at the assigned value throughout the window, so manipulation
  fidelity is near-deterministic by construction rather than hoped for.

  DESIGN CAVEAT, stated plainly: the hazard tracks the agent. That is ecologically odd
  and it is exactly what exogenous manipulation of proximity means here. It is the
  price of breaking the endogeneity, and it is the same price 785a paid on the urgency
  axis. What it buys is that proximity no longer co-varies with wherever the agent
  chose to go, which is the confound that made 785 uninterpretable.

PROXIMITY COVARIATE
  PRIMARY is hazard_prox_center -- index 12 of the 25-dim hazard_field_view, the
  agent's own cell. NOT hazard_prox_max, which 785a used: the 5x5 window spans +/-2, so
  the nearest window cell to the hazard sits at distance max(0, d-2) and hazard_prox_max
  SATURATES at 1.0 for every d <= 2. hazard_prox_center is monotone in d across the
  whole grid. max and mean are recorded as secondaries.

  This is the learner's OWN 5x5 local observable. A privileged global oracle distance is
  NOT used -- that is the 732a confound.

ARMS
  A1 `hazard_exog_urgency_clamped` -- PRIMARY IDENTIFICATION. Hazard assigned as above;
     urgency CLAMPED to a single constant 0.16 via 785a's injection
     (agent.e3.config.urgency_weight = clamp / sig_norm, read live at select()).
     Any dependence of share / var on assigned proximity here is pure hazard geometry,
     net of arousal.
  A2 `hazard_exog_urgency_free` -- ECOLOGICAL ARM AND MEDIATION LINK. Same hazard
     assignment; urgency left natural (urgency_weight = 0.12 baseline, no injection).
     Verifies the coupling H itself posits: corr(assigned proximity, realized urgency)
     should be strongly POSITIVE. If it is not, H's own premise ("high-urgency ticks are
     near-hazard ticks") fails at the first stage, and that is decisive independently of
     anything the geometry does.
  A3 `expressivity_positive_control` -- 8 seeds. Hazard fixed at d=3, urgency clamped at
     0.16; sweeps support_preserving_ao_std_floor over {0.2, 0.6, 1.2} (785a used 0.2).
     Widening the action-option support makes candidates differ more, so var_total must
     rise monotonically. This arm is the arbiter described next.

PRE-DECLARED NULL-vs-INEXPRESSIVE DISCRIMINATOR
  Live caveat carried forward from the 785a adjudication: z_world is under-differentiated
  (participation ratio ~1.06 at world_dim=128, absolute variances ~1.2e-05). So a null
  here invites the objection "the substrate cannot EXPRESS the effect" -- an objection
  which, if correct, bites the FIX rather than the MEASUREMENT. The two readings are
  pre-separated BEFORE the run, not argued afterwards:

    - Every arm records the z_world participation ratio and the absolute var_total scale.
    - A3 is the arbiter. If A3's ao_std_floor sweep DOES move var_total monotonically,
      the measurement channel is demonstrably expressive, and a flat A1 response is a
      GENUINE NULL -> H-endogenous-hazard-geometry is eliminated, self-route
      `hazard_geometry_inert_on_selection_variance`.
    - If A3 does NOT move var_total, the channel is inert regardless of input and
      NOTHING about hazard can be concluded -> self-route
      `measurement_channel_inexpressive`, outcome FAIL, evidence_direction
      non_contributory, non_degenerate false, and the ledger leg stays `alive` with no
      bit claimed.
    - Second tell, independent of A3: if A2 shows hazard strongly moves realized urgency
      (the input path is live) while cross-candidate variance geometry stays flat, the
      failure localises to the geometry stage rather than to the manipulation.

PRE-REGISTERED STATISTICS
  LOAD-BEARING, within-seed. For each seed s, the within-seed OLS slope beta_s of the DV
  on the assigned PROXIMITY level index, over that seed's fresh committed rows only.
      C1 amplification:  DV = log10(var_total)     H predicts beta > 0
      C2 redistribution: DV = incumbent share      H predicts beta < 0
  Each is tested with a two-sided one-sample t-test on {beta_s} against H0 mean = 0,
  using a pre-registered critical value (no post-hoc threshold). The effect must also
  clear max(ABS_FLOOR, 2.0 * sd(beta_s) / sqrt(S)) -- noise scaled on the SD of the
  per-seed effect plus an absolute floor, so a statistically-detectable but
  scientifically-empty slope does not read as support.
  A redundant robustness statistic must agree in SIGN: the within-seed level gap
  (nearest level minus farthest, computed within seed, then averaged), plus a per-seed
  sign test.

  EXPLICITLY NOT LOAD-BEARING -- recorded as decoys so the Simpson structure is legible
  in the manifest rather than reconstructible only by autopsy:
      D1 pooled tick-level r over all rows. THIS IS THE STATISTIC THAT MISLEADS HERE.
      D2 between-seed r across the S seed means.
      D3 a within/between covariance decomposition: the fraction of the pooled
         covariance attributable to between-seed heterogeneity.
      D4 between_seed_correlation_replication -- does 785a's r(mean hazard, mean share)
         = -0.78 / r(mean hazard, mean var) = +0.77 replicate at S=24, or was it a seed-0
         artifact? The per-seed distribution of mean share is reported and any seed near
         0.833 is flagged.

  FIRST STAGE / FIDELITY (preconditions, never criteria): |corr(assigned proximity level,
  realized hazard_prox_center)| near 1; per-level realized proximity means; the fraction
  of re-pins that missed the exact distance; corr(assigned level, realized urgency),
  expected ~0 in A1 (the clamp holds) and strongly positive in A2.

  REFERENCE MAGNITUDE: H predicts the 785 profile, i.e. near-hazard gives var_total up
  ~14.1x (1.149 in log10) and share down ~0.139. The observed within-seed effect is
  reported as a FRACTION of that reference, which is what makes the
  "directionally-right-but-small" interpretation cell decidable instead of a judgement
  call.

SCOPE AND WHAT A RESULT DOES NOT MEAN
  - Scope is the SD-011 commit-threshold route only. use_harm_variance_commit is OFF, so
    the gated quantity is the z_world RUNNING VARIANCE (world-model stability), not
    candidate separation. A result here is not a result about every hazard route.
  - CHANNEL-AGNOSTICISM IS OUT OF SCOPE. The incumbent is harm_weighted at a single
    regime. The entropy regime is dropped with recorded evidence (CH:mech341 absorbs
    ~99-115% of cross-candidate variance at every entropy_bias_scale scanned), so it is
    not resurrected here and no result should be read as covering a second incumbent
    identity.
  - A null on C1/C2 with A3 green eliminates H at THIS incumbent identity on THIS route.
    It does not establish what did drive 785; that would remain open.

RECORDING
  Per-tick rows are embedded IN the manifest at custom_information.per_tick_rows. A
  sidecar written next to the manifest on a cloud worker is NEVER transported under
  Phase 3 (PHASE3_DISABLE_RUNNER_RESULT_PUSH=1) -- that is how 785 lost its per-tick
  sink and with it the one reanalysis that would have tested its mechanism directly. A
  .jsonl sidecar is written in non-dry-run mode as an explicitly NON-AUTHORITATIVE
  convenience copy.

  agent.e3.last_score_diagnostics is cleared to None immediately BEFORE every
  select_action, and a row is recorded ONLY when the post-call diagnostics are non-None
  and carry urgency_applied. 785 omitted this clear and re-recorded the previous tick's
  diagnostics on skipped ticks (~9.0x pseudo-replication).

  CORRECTION to 785a's pseudo_replication_note, which this script does not repeat: the
  skip driver is the E3 CADENCE (heartbeat.e3_steps_per_tick, default 10 -- see the
  `if not ticks["e3_tick"]` early return at agent.py:5429), NOT the commitment latch.
  On a non-E3 tick select_action is never reached at all, so the diagnostics are stale
  for a structural reason rather than a behavioural one.

DRIVER CONSTRAINTS INHERITED FROM 785a (do not "simplify" these)
  - Do NOT use act_with_split_obs(); feed obs_harm_a explicitly and hand-roll
    clock.advance() -> _e1_tick -> generate_trajectories -> select_action.
  - Call agent.e3.update_running_variance(z_world_cur - pred) explicitly. Nothing in
    ree_core calls it, and without it the commit gate never fires.
  - agent.e3.e3_score_decomp_enabled = True (default False, e3_selector.py:360) is what
    populates last_score_decomp at all.
  - cfg.e3.use_finer_channel_gating = True gives NAMED channels in last_channel_terms.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)

EXPERIMENT_TYPE = "v3_exq_787_mech463_hazard_geometry_exogenous_proximity"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = ["MECH-463"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = list(range(24))
A3_SEEDS = SEEDS[:8]
TICKS_PER_SEED = 2000
# The expressivity control needs far less: its effect is ~1.0-1.6 orders of magnitude in
# log10 var_total (measured in the 2026-07-19 dry run at only 120 ticks/cell), against a
# 0.10 floor. Spending A1/A2's budget on an already-decisive control would nearly double
# the run for no information.
A3_TICKS_PER_SEED = 1000
DRY_RUN_TICKS = 120
DRY_RUN_SEEDS = SEEDS[:2]

# Assignment window, in env steps. The E3 cadence is heartbeat.e3_steps_per_tick
# (default 10, agent.py:5429), so a 10-step window carries ~1 scored E3 tick.
WINDOW_STEPS = 10

# EXOGENOUS hazard distances, ordered FAR -> NEAR so the level index increases with
# PROXIMITY. Under this ordering H predicts beta > 0 on log10 var_total and beta < 0 on
# incumbent share -- the signs of the 785 profile. Field value at the agent's own cell is
# 1/(1 + d*0.5) = {0.222, 0.286, 0.400, 0.500, 0.667}, a 3.0x span.
# d=7 (not 8) is the far end: on a size-10 grid the interior is 1..8, so from a central
# start only the corners realise d=8 and the re-pin would miss constantly.
HAZARD_DISTANCES = [7, 5, 3, 2, 1]
HAZARD_PROX_EXPECTED = [1.0 / (1.0 + d * 0.5) for d in HAZARD_DISTANCES]

A3_FIXED_DISTANCE = 3
A3_AO_STD_FLOORS = [0.2, 0.6, 1.2]

URGENCY_CLAMP = 0.16          # A1 and A3: urgency held constant
URGENCY_WEIGHT_BASELINE = 0.12  # A2: natural urgency, 785a's baseline
ALPHA_WORLD = 0.9
MODULATORY_AUTHORITY_GAIN = 0.5
EPISODE_START_CELL = (5, 5)

PRIMARY_COMPONENTS = [
    "f", "harm_weighted", "residue_weighted",
    "benefit_weighted", "novelty_weighted", "goal_weighted",
]
EXPECTED_INCUMBENT = "harm_weighted"
# 785a's measured mean share for harm_weighted at this regime. Well below 1.0, so the
# ">= 2 components above the |0.01| floor" gate is arithmetically SATISFIABLE here --
# unlike 785's entropy arm, whose committed expected_incumbent_share of 1.043 forced
# every other component to sum to -0.043 and made its own gate unmeetable by construction.
EXPECTED_INCUMBENT_SHARE = 0.9368
NONTRIVIAL_SHARE_FLOOR = 0.01

ARMS: List[Dict[str, Any]] = [
    {"id": "hazard_exog_urgency_clamped", "kind": "hazard_sweep",
     "urgency_mode": "clamped", "urgency_clamp": URGENCY_CLAMP,
     "ao_std_floor": 0.2, "seeds": SEEDS},
    {"id": "hazard_exog_urgency_free", "kind": "hazard_sweep",
     "urgency_mode": "free", "urgency_clamp": None,
     "ao_std_floor": 0.2, "seeds": SEEDS},
    {"id": "expressivity_positive_control", "kind": "ao_sweep",
     "urgency_mode": "clamped", "urgency_clamp": URGENCY_CLAMP,
     "ao_std_floor": None, "seeds": A3_SEEDS},
]
HAZARD_SWEEP_ARMS = [a["id"] for a in ARMS if a["kind"] == "hazard_sweep"]

DROPPED_REGIME_EVIDENCE = {
    "dropped": "entropy_incumbent (entropy_bias_scale sweep)",
    "reason": (
        "no viable second cross-candidate regime exists on the entropy axis -- "
        "CH:mech341 absorbs ~99-115% of cross-candidate variance at every "
        "entropy_bias_scale scanned in the 785a design pass {0.15,0.30,0.50,1.00}, so "
        "its share statistic is arithmetically forced and cannot measure redistribution"
    ),
    "consequence": (
        "MECH-463's CHANNEL-AGNOSTICISM half is UNTESTED by this run. A result here "
        "covers the harm_weighted incumbent identity only."
    ),
}

# --- power / adequacy floors ------------------------------------------------- #
MIN_ROWS_PER_SEED_LEVEL = 30
MIN_LEVELS_POPULATED = 4          # of 5
# Seed-count adequacy is pre-registered as a FRACTION of the arm's seed list, not as an
# absolute count. The scientific requirement is "most seeds contribute a usable slope";
# hardcoding the count would be correct at full scale and structurally unsatisfiable at
# any smaller one, which is exactly what assert_no_structurally_unsatisfiable_gate
# refuses. At full scale these resolve to >= 20 of 24 and >= 6 of 8.
MIN_SEEDS_WITH_SLOPE_FRAC = 0.83
MIN_A3_SEEDS_WITH_PROFILE_FRAC = 0.75
SCORE_VARIANCE_FLOOR = 1e-18
FIDELITY_CORR_FLOOR = 0.90
PIN_MISS_FRAC_CEILING = 0.15
INCUMBENT_MARGIN_FLOOR = 0.30

# --- effect floors, pre-registered against the 785 reference profile ---------- #
# 785 reference: var_total up 14.1x over the span (log10 1.149) and share down 0.139.
# The level index spans 4 units, so the reference slopes are 1.149/4 = 0.287 and
# -0.139/4 = -0.0348. The absolute floors are 10% of those reference slopes: large
# enough that pure noise cannot clear them, small enough that a real-but-modest effect
# still registers as the "directionally-right-but-small" cell rather than as flat.
REF_LOGVAR_FOLD = 1.149
REF_SHARE_GAP = -0.139
REF_LOGVAR_SLOPE = REF_LOGVAR_FOLD / 4.0
REF_SHARE_SLOPE = REF_SHARE_GAP / 4.0
LOGVAR_SLOPE_ABS_FLOOR = 0.0287
SHARE_SLOPE_ABS_FLOOR = 0.0035
SE_MULTIPLIER = 2.0
SMALL_EFFECT_FRACTION = 0.25      # below this fraction of reference -> "small" cell

# Two-sided t critical values at alpha=0.05, keyed by degrees of freedom. Pre-registered
# so no threshold is ever derived from the run's own statistics. df >= 30 uses 2.042.
T_CRIT_TWO_SIDED = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                    11: 2.201, 12: 2.179, 15: 2.131, 19: 2.093, 20: 2.086,
                    21: 2.080, 22: 2.074, 23: 2.069}
T_CRIT_DEFAULT = 2.042

A3_MONOTONE_RHO_FLOOR = 0.60
A3_MIN_LOGVAR_SPAN = 0.10         # the ao sweep must move log10 var_total by >= 0.1

PER_CANDIDATE_SAMPLE_PER_CELL = 25

REFERENCE_PROFILE_785 = {
    "var_total_fold": 14.1,
    "log10_var_fold": REF_LOGVAR_FOLD,
    "incumbent_share_gap": REF_SHARE_GAP,
    "note": (
        "the endogenous-urgency profile this hypothesis says was hazard geometry all "
        "along; observed within-seed effects are reported as a fraction of it"
    ),
}
SEED0_ANOMALY_785A = {
    "mean_share": 0.833,
    "mean_var_total": 4.47e-05,
    "note": (
        "785a seed 0 carried the between-seed signal almost alone; its mean share of "
        "0.833 is near-identical to 785's endpoint of 0.831. D4 tests whether any seed "
        "at S=24 reproduces that anomaly."
    ),
}
SEED0_FLAG_TOLERANCE = 0.02


# ------------------------------------------------------------------ #
# Statistical helpers                                                #
# ------------------------------------------------------------------ #
def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation. Returns 0.0 when undefined (n < 3 or a flat side)."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(v: List[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return float(num / (dx * dy))


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def _ols_slope(x: List[float], y: List[float]) -> Optional[float]:
    """Least-squares slope of y on x. None when undefined (n < 3 or x is flat)."""
    if len(x) < 3 or len(set(x)) < 2:
        return None
    ax, ay = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    vx = float(np.var(ax))
    if vx <= 0.0:
        return None
    return float(np.mean((ax - ax.mean()) * (ay - ay.mean())) / vx)


def _t_crit(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df in T_CRIT_TWO_SIDED:
        return T_CRIT_TWO_SIDED[df]
    if df >= 30:
        return T_CRIT_DEFAULT
    # Between tabulated points, take the nearest tabulated df below (conservative).
    below = [k for k in T_CRIT_TWO_SIDED if k <= df]
    return T_CRIT_TWO_SIDED[max(below)] if below else 2.571


def _one_sample_t(values: List[float]) -> Dict[str, Any]:
    """Two-sided one-sample t-test against H0 mean = 0, with a pre-registered critical
    value. No p-value machinery: the decision is |t| > t_crit(df), fixed in advance."""
    n = len(values)
    if n < 3:
        return {"n": n, "mean": 0.0, "sd": 0.0, "se": 0.0, "t": 0.0,
                "df": max(0, n - 1), "t_crit": float("inf"), "significant": False}
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0.0 else 0.0
    t = float(mean / se) if se > 0.0 else 0.0
    df = n - 1
    crit = _t_crit(df)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t, "df": df,
            "t_crit": crit, "significant": bool(abs(t) > crit)}


def _sign_test(values: List[float]) -> Dict[str, Any]:
    pos = sum(1 for v in values if v > 0.0)
    neg = sum(1 for v in values if v < 0.0)
    n = pos + neg
    return {"n_nonzero": n, "n_positive": pos, "n_negative": neg,
            "positive_frac": (pos / n) if n > 0 else 0.0}


def _within_between_decomposition(seed_ids: List[int], xs: List[float],
                                  ys: List[float]) -> Dict[str, Any]:
    """Split the pooled covariance of (x, y) into within-seed and between-seed parts.

    Cov_pooled = E_s[Cov_within(s)] + Cov_between(seed means). The between fraction is
    the formal Simpson diagnostic: 785a's apparent hazard effect was ~entirely between.
    """
    by_seed: Dict[int, Tuple[List[float], List[float]]] = {}
    for s, x, y in zip(seed_ids, xs, ys):
        bucket = by_seed.setdefault(s, ([], []))
        bucket[0].append(x)
        bucket[1].append(y)
    if len(by_seed) < 2:
        return {"pooled_cov": 0.0, "within_cov": 0.0, "between_cov": 0.0,
                "between_fraction": 0.0, "n_seeds": len(by_seed)}

    ax, ay = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    pooled = float(np.mean((ax - ax.mean()) * (ay - ay.mean())))

    within_terms, mx, my, weights = [], [], [], []
    total = float(len(xs))
    for _s, (sx, sy) in by_seed.items():
        a, b = np.asarray(sx, dtype=float), np.asarray(sy, dtype=float)
        w = len(sx) / total
        weights.append(w)
        within_terms.append(w * float(np.mean((a - a.mean()) * (b - b.mean()))))
        mx.append(float(a.mean()))
        my.append(float(b.mean()))
    within = float(sum(within_terms))
    wa = np.asarray(weights, dtype=float)
    amx, amy = np.asarray(mx, dtype=float), np.asarray(my, dtype=float)
    gx = float((wa * amx).sum())
    gy = float((wa * amy).sum())
    between = float((wa * (amx - gx) * (amy - gy)).sum())
    frac = float(between / pooled) if abs(pooled) > 1e-30 else 0.0
    return {"pooled_cov": pooled, "within_cov": within, "between_cov": between,
            "between_fraction": frac, "n_seeds": len(by_seed)}


def _component_shares(components: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
    """Covariance-correct cross-candidate variance share for EVERY component.

    score_k = sum_c C_ck. Attributing to component c its own variance plus its full
    covariance with the rest:
        share_c = (Var(C_c) + Cov(C_c, sum_{d != c} C_d)) / Var(score)
    These sum to EXACTLY 1, since
        sum_c [Var(C_c) + Cov(C_c, rest_c)] = Var(sum_c C_c).
    Retaining the per-candidate [K] tensors unreduced is what makes the covariance terms
    computable at all -- marginal variances alone cannot form them.
    """
    keys = [k for k, v in components.items() if v.size >= 2]
    if not keys:
        return None
    n = components[keys[0]].size
    total = np.zeros(n, dtype=float)
    for k in keys:
        if components[k].size != n:
            return None
        total = total + components[k]
    var_total = float(np.var(total))
    if not np.isfinite(var_total) or var_total <= SCORE_VARIANCE_FLOOR:
        return None
    out: Dict[str, float] = {}
    for k in keys:
        c = components[k]
        rest = total - c
        cov = float(np.mean((c - c.mean()) * (rest - rest.mean())))
        out[k] = (float(np.var(c)) + cov) / var_total
    out["__var_total__"] = var_total
    return out


def _participation_ratio(mat: np.ndarray) -> float:
    """PR = (sum eig)^2 / sum(eig^2) of the covariance. ~1.0 means one direction carries
    everything -- the z_world under-differentiation the 785a adjudication flagged."""
    if mat.ndim != 2 or mat.shape[0] < 3:
        return 0.0
    cov = np.cov(mat, rowvar=False)
    if cov.ndim == 0:
        return 1.0
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    s1 = float(eig.sum())
    s2 = float((eig ** 2).sum())
    if s2 <= 0.0:
        return 0.0
    return float(s1 * s1 / s2)


# ------------------------------------------------------------------ #
# Hazard manipulation                                                #
# ------------------------------------------------------------------ #
def _repin_hazard(env, target_dist: int, rng: np.random.Generator) -> Tuple[float, bool]:
    """Move the single hazard to Manhattan distance `target_dist` from the agent NOW.

    In-place relocate (clear old grid cell, mutate the env.hazards entry, set the new
    cell) following v3_exq_495a_mech163_planned_system_gate.py:363, then recompute the
    proximity fields so hazard_field_view reflects it on the very next observation.
    reset_to() is NOT used here: mid-episode it would wipe health/energy and rebuild the
    grid, destroying episode continuity.

    The bearing is drawn i.i.d. at every re-pin so no fixed direction is confounded with
    the assigned level. When no interior empty cell sits at exactly `target_dist` (the
    agent is near a corner), the largest achievable distance <= target_dist is used and
    the call reports a MISS, which the fidelity precondition bounds.

    Returns (realized_distance, missed).
    """
    ax, ay = int(env.agent_x), int(env.agent_y)
    empty = env.ENTITY_TYPES["empty"]
    lo, hi = 1, env.size - 2

    by_dist: Dict[int, List[Tuple[int, int]]] = {}
    for i in range(lo, hi + 1):
        for j in range(lo, hi + 1):
            d = abs(i - ax) + abs(j - ay)
            if d <= 0 or d > target_dist:
                continue
            if env.grid[i, j] != empty:
                continue
            by_dist.setdefault(d, []).append((i, j))
    if not by_dist:
        return float(target_dist), True

    realized = target_dist if target_dist in by_dist else max(by_dist)
    cells = by_dist[realized]
    tx, ty = cells[int(rng.integers(len(cells)))]

    if env.hazards:
        h = env.hazards[0]
        if 0 <= h[0] < env.size and 0 <= h[1] < env.size:
            if env.grid[h[0], h[1]] == env.ENTITY_TYPES["hazard"]:
                env.grid[h[0], h[1]] = empty
        h[0], h[1] = tx, ty
    else:
        env.hazards.append([tx, ty])
    env.grid[tx, ty] = env.ENTITY_TYPES["hazard"]
    env._compute_proximity_fields()
    return float(realized), bool(realized != target_dist)


def _episode_reset(env, target_dist: int, rng: np.random.Generator):
    """Deterministic episode start: agent at the fixed centre cell, hazard at the
    assigned distance. reset_to is the sanctioned scripted-placement API."""
    ax, ay = EPISODE_START_CELL
    cands = [(i, j) for i in range(1, env.size - 1) for j in range(1, env.size - 1)
             if abs(i - ax) + abs(j - ay) == target_dist]
    if not cands:
        cands = [(1, 1)]
    hx, hy = cands[int(rng.integers(len(cands)))]
    return env.reset_to((ax, ay), [(hx, hy)])


def _hazard_readouts(obs_dict) -> Tuple[float, float, float]:
    """Read the learner's OWN 5x5 hazard observable. Index 12 is the agent's own cell
    (the view is built as h_view[di+2, dj+2] then flattened, causal_grid_world.py:3146),
    so `center` is the monotone-in-distance code. `max` saturates at 1.0 for every
    d <= 2 because the window spans +/-2 -- that is why it is NOT the primary."""
    hz = obs_dict.get("hazard_field_view")
    if hz is None:
        return 0.0, 0.0, 0.0
    v = hz.detach().cpu().numpy().astype(float).reshape(-1)
    return float(v[v.size // 2]), float(v.max()), float(v.mean())


# ------------------------------------------------------------------ #
# Agent / env construction                                           #
# ------------------------------------------------------------------ #
def _urgency_signal(agent: REEAgent, latent) -> Optional[torch.Tensor]:
    """Mirrors SD-019a's redirect: the urgency drive is z_harm_a unless the unified harm
    stream is on, in which case it is z_harm_un."""
    sig = latent.z_harm_a
    if getattr(agent.config.latent, "use_harm_un", False) and latent.z_harm_un is not None:
        sig = latent.z_harm_un
    return sig


def _build_agent_and_env(seed: int, ao_std_floor: float):
    env = CausalGridWorldV2(
        use_proxy_fields=True,
        seed=seed,
        hazard_harm=0.5,
        num_hazards=1,        # keeps the hazard_field_view normaliser fixed at 1.0
        env_drift_prob=0.0,   # freeze hazards; the re-pin is the only mover
    )
    _obs, obs_dict = env.reset()

    kw = dict(
        body_obs_dim=obs_dict["body_state"].shape[-1],
        world_obs_dim=obs_dict["world_state"].shape[-1],
        action_dim=env.action_dim,
        alpha_world=ALPHA_WORLD,
        use_harm_stream=True,
        use_affective_harm_stream=True,   # both required or z_harm_a is None
        urgency_weight=URGENCY_WEIGHT_BASELINE,
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=ao_std_floor,
        use_per_stream_vs=True, use_per_region_vs=True,
        use_event_segmenter=True, use_invalidation_trigger=True, use_anchor_sets=True,
        e2_action_contrastive_enabled=True, e2_action_contrastive_weight=0.1,
        e2_rollout_output_norm_clamp_enabled=True, e2_rollout_output_norm_clamp_ratio=4.0,
        use_structured_curiosity=True, use_curiosity_novelty=True,
        curiosity_bias_scale=0.1, curiosity_novelty_weight=0.05,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_min_range_floor=1e-6,
        use_e3_score_diversity=False,
        use_e3_diversity_entropy_bonus=False,
    )
    cfg = REEConfig.from_dims(**kw)
    cfg.e3.use_finer_channel_gating = True   # NAMED channels in last_channel_terms
    agent = REEAgent(cfg)
    agent.eval()
    # MECH-463 instrumentation gate (ree-v3 435322f); default False at
    # e3_selector.py:360, and without it last_score_decomp is never populated.
    agent.e3.e3_score_decomp_enabled = True
    # use_harm_variance_commit is deliberately left OFF: the gated quantity is therefore
    # the z_world RUNNING VARIANCE (world-model stability), not candidate separation.
    return agent, env, obs_dict, kw


# ------------------------------------------------------------------ #
# Cell collection                                                    #
# ------------------------------------------------------------------ #
def _collect_cell(seed: int, arm: Dict[str, Any], ao_std_floor: float,
                  fixed_distance: Optional[int], n_ticks: int
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one (arm, seed[, ao level]) cell and return its per-tick rows + telemetry."""
    agent, env, obs_dict, _kw = _build_agent_and_env(seed, ao_std_floor)
    rng = np.random.default_rng(20_000 + seed)

    rows: List[Dict[str, Any]] = []
    z_world_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None

    n_fresh = 0
    n_skipped = 0
    n_committed = 0
    pin_attempts = 0
    pin_misses = 0
    authority_hits = 0
    channel_ranges: List[float] = []
    incumbent_ranges: List[float] = []
    zworld_samples: List[np.ndarray] = []

    level_idx = 0
    assigned_distance = fixed_distance if fixed_distance is not None else HAZARD_DISTANCES[0]
    _obs, obs_dict = _episode_reset(env, assigned_distance, rng)

    print(f"Seed {seed} Condition {arm['id']}_ao{ao_std_floor}", flush=True)

    for tick in range(n_ticks):
        # --- exogenous assignment, drawn BEFORE anything about this tick is observed --
        if tick % WINDOW_STEPS == 0:
            if fixed_distance is None:
                level_idx = int(rng.integers(len(HAZARD_DISTANCES)))
                assigned_distance = HAZARD_DISTANCES[level_idx]
            else:
                level_idx = 0
                assigned_distance = fixed_distance
        realized_distance, missed = _repin_hazard(env, assigned_distance, rng)
        pin_attempts += 1
        pin_misses += int(missed)
        obs_dict = env._get_observation_dict()

        hz_center, hz_max, hz_mean = _hazard_readouts(obs_dict)

        with torch.no_grad():
            latent = agent.sense(
                obs_dict["body_state"].unsqueeze(0),
                obs_dict["world_state"].unsqueeze(0),
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            z_world_cur = latent.z_world.detach()
            if len(zworld_samples) < 400:
                zworld_samples.append(z_world_cur.cpu().numpy().reshape(-1))
            if z_world_prev is not None and action_prev is not None:
                _pred = agent.e2.world_forward(z_world_prev, action_prev)
                # Nothing in ree_core calls this; without it the commit gate never fires.
                agent.e3.update_running_variance(z_world_cur - _pred.detach())

            sig = _urgency_signal(agent, latent)
            sig_norm = float(sig.norm(dim=-1).mean().item()) if sig is not None else 0.0
            natural_urgency = min(sig_norm * URGENCY_WEIGHT_BASELINE,
                                  agent.e3.config.urgency_max)
            if arm["urgency_mode"] == "clamped":
                # urgency_applied = min(sig_norm * urgency_weight, urgency_max); solving
                # for the weight lands it exactly on the clamp. Read live at select().
                clamp = float(arm["urgency_clamp"])
                agent.e3.config.urgency_weight = (
                    (clamp / sig_norm) if sig_norm > 1e-9 else 0.0)
            else:
                agent.e3.config.urgency_weight = URGENCY_WEIGHT_BASELINE

            # Freshness marker. select_action is not reached at all on a non-E3 tick
            # (the `if not ticks["e3_tick"]` early return at agent.py:5429, cadence
            # heartbeat.e3_steps_per_tick default 10). Without this clear a skipped tick
            # re-reads the PREVIOUS tick's diagnostics and the row is a duplicate -- the
            # ~9.0x pseudo-replication defect in 785.
            agent.e3.last_score_diagnostics = None

            ticks_d = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks_d["e1_tick"]
                else torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks_d)
            action = agent.select_action(candidates, ticks_d, 1.0)
        agent._step_count += 1

        diag = agent.e3.last_score_diagnostics
        if diag is None or "urgency_applied" not in diag:
            n_skipped += 1
        else:
            n_fresh += 1
            realized_urgency = float(diag["urgency_applied"])
            if float(diag.get("modulatory_authority_active", 0.0) or 0.0) > 0.0:
                authority_hits += 1

            decomp = agent.e3.last_score_decomp or {}
            chan = agent.e3.last_channel_terms or {}
            per_cand = decomp.get("per_candidate") or []
            comps: Dict[str, np.ndarray] = {}
            for name in PRIMARY_COMPONENTS:
                comps[name] = np.asarray(
                    [float(c.get(name, 0.0) or 0.0) for c in per_cand], dtype=float)
            for cname, cvec in chan.items():
                v = cvec.detach().cpu().numpy().astype(float)
                comps["CH:" + cname] = v
                if v.size >= 2:
                    channel_ranges.append(float(v.max() - v.min()))
            if EXPECTED_INCUMBENT in comps and comps[EXPECTED_INCUMBENT].size >= 2:
                iv = comps[EXPECTED_INCUMBENT]
                incumbent_ranges.append(float(iv.max() - iv.min()))

            shares = _component_shares(comps) if len(per_cand) >= 2 else None
            if shares is not None:
                var_total = shares.pop("__var_total__")
                committed = bool(diag.get("committed", False))
                n_committed += int(committed)
                row: Dict[str, Any] = {
                    "arm": arm["id"],
                    "ao_std_floor": float(ao_std_floor),
                    "seed": seed,
                    "tick": tick,
                    "assigned_distance": int(assigned_distance),
                    "assigned_level_idx": int(level_idx),
                    "realized_distance": float(realized_distance),
                    "pin_missed": bool(missed),
                    "hazard_prox_center": round(hz_center, 6),
                    "hazard_prox_max": round(hz_max, 6),
                    "hazard_prox_mean": round(hz_mean, 6),
                    "realized_urgency": round(realized_urgency, 6),
                    "natural_urgency": round(natural_urgency, 6),
                    "effective_threshold": float(diag.get("effective_threshold", 0.0)),
                    "commit_variance": float(diag.get("commit_variance", 0.0)),
                    "commit_gate_mode": str(diag.get("commit_gate_mode", "")),
                    "committed": committed,
                    "temperature": float(
                        diag.get("gap_scaled_commit_temperature_eff", 1.0) or 1.0),
                    "var_total": round(var_total, 14),
                    "shares": {k: round(v, 5) for k, v in shares.items()},
                }
                # Bulky per-candidate vectors: bounded sample only. Scalars are kept for
                # every row, so the manifest carries the full analysis set at a sane size.
                if len(rows) < PER_CANDIDATE_SAMPLE_PER_CELL:
                    row["per_candidate"] = {
                        k: [round(x, 6) for x in v.tolist()] for k, v in comps.items()}
                rows.append(row)

        act_idx = int(action.argmax().item()) if isinstance(action, torch.Tensor) else int(action)
        action_prev = torch.zeros(1, env.action_dim)
        action_prev[0, act_idx % env.action_dim] = 1.0
        z_world_prev = z_world_cur
        _obs, _r, done, _info, obs_dict = env.step(act_idx % env.action_dim)
        if done:
            _obs, obs_dict = _episode_reset(env, assigned_distance, rng)

        if (tick + 1) % 200 == 0:
            print(f"  [train] {arm['id']} seed={seed} ep {tick + 1}/{n_ticks} "
                  f"rows={len(rows)} committed={n_committed}", flush=True)

    pr = 0.0
    if len(zworld_samples) >= 8:
        pr = _participation_ratio(np.vstack(zworld_samples))

    telem = {
        "arm": arm["id"],
        "seed": seed,
        "ao_std_floor": float(ao_std_floor),
        "n_ticks": n_ticks,
        "n_fresh_select": n_fresh,
        "n_skipped_e3_cadence": n_skipped,
        "fresh_select_yield": (n_fresh / n_ticks) if n_ticks else 0.0,
        "n_rows": len(rows),
        "n_committed": n_committed,
        "pin_miss_frac": (pin_misses / pin_attempts) if pin_attempts else 0.0,
        "authority_active_frac": (authority_hits / n_fresh) if n_fresh else 0.0,
        "channel_range_mean": float(np.mean(channel_ranges)) if channel_ranges else 0.0,
        "incumbent_range_mean": float(np.mean(incumbent_ranges)) if incumbent_ranges else 0.0,
        "zworld_participation_ratio": round(pr, 4),
    }
    return rows, telem


# ------------------------------------------------------------------ #
# Analysis                                                           #
# ------------------------------------------------------------------ #
def _committed(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("committed")]


def _per_seed_slopes(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """THE LOAD-BEARING STATISTIC. Within-seed OLS slopes on the assigned PROXIMITY level
    index, one per seed, then a one-sample t-test across seeds.

    This is the statistic the ledger asks for. A pooled tick-level correlation over the
    same rows is the statistic that produced 785a's false positive, and it appears below
    only under `decoys`."""
    by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), []).append(r)

    slope_var: List[float] = []
    slope_share: List[float] = []
    gap_var: List[float] = []
    gap_share: List[float] = []
    seeds_used: List[int] = []
    per_seed: List[Dict[str, Any]] = []
    n_levels = len(HAZARD_DISTANCES)

    for seed in sorted(by_seed):
        srows = by_seed[seed]
        by_level: Dict[int, List[Dict[str, Any]]] = {}
        for r in srows:
            by_level.setdefault(int(r["assigned_level_idx"]), []).append(r)
        populated = [lv for lv, lr in by_level.items() if len(lr) >= MIN_ROWS_PER_SEED_LEVEL]
        if len(populated) < MIN_LEVELS_POPULATED:
            per_seed.append({"seed": seed, "usable": False,
                             "levels_populated": len(populated), "n_rows": len(srows)})
            continue

        xs = [float(r["assigned_level_idx"]) for r in srows]
        ylv = [math.log10(max(float(r["var_total"]), 1e-30)) for r in srows]
        ysh = [float(r["shares"].get(EXPECTED_INCUMBENT, 0.0)) for r in srows]
        b_var = _ols_slope(xs, ylv)
        b_sh = _ols_slope(xs, ysh)
        if b_var is None or b_sh is None:
            per_seed.append({"seed": seed, "usable": False,
                             "levels_populated": len(populated), "n_rows": len(srows)})
            continue

        lo, hi = min(populated), max(populated)
        near = by_level[hi]   # highest index = NEAREST hazard
        far = by_level[lo]
        g_var = (float(np.mean([math.log10(max(float(r["var_total"]), 1e-30)) for r in near]))
                 - float(np.mean([math.log10(max(float(r["var_total"]), 1e-30)) for r in far])))
        g_sh = (float(np.mean([float(r["shares"].get(EXPECTED_INCUMBENT, 0.0)) for r in near]))
                - float(np.mean([float(r["shares"].get(EXPECTED_INCUMBENT, 0.0)) for r in far])))

        seeds_used.append(seed)
        slope_var.append(b_var)
        slope_share.append(b_sh)
        gap_var.append(g_var)
        gap_share.append(g_sh)
        per_seed.append({
            "seed": seed, "usable": True, "n_rows": len(srows),
            "levels_populated": len(populated),
            "slope_log10_var": round(b_var, 6),
            "slope_incumbent_share": round(b_sh, 6),
            "gap_log10_var_near_minus_far": round(g_var, 6),
            "gap_incumbent_share_near_minus_far": round(g_sh, 6),
            "mean_incumbent_share": round(
                float(np.mean([float(r["shares"].get(EXPECTED_INCUMBENT, 0.0))
                               for r in srows])), 6),
            "mean_var_total": float(np.mean([float(r["var_total"]) for r in srows])),
            "mean_hazard_prox_center": round(
                float(np.mean([float(r["hazard_prox_center"]) for r in srows])), 6),
        })

    return {
        "n_levels": n_levels,
        "seeds_used": seeds_used,
        "n_seeds_with_slope": len(seeds_used),
        "per_seed": per_seed,
        "log10_var": {
            "slopes": [round(v, 6) for v in slope_var],
            "t_test": _one_sample_t(slope_var),
            "sign_test": _sign_test(slope_var),
            "gap_t_test": _one_sample_t(gap_var),
            "gap_sign_test": _sign_test(gap_var),
            "mean_gap_near_minus_far": float(np.mean(gap_var)) if gap_var else 0.0,
        },
        "incumbent_share": {
            "slopes": [round(v, 6) for v in slope_share],
            "t_test": _one_sample_t(slope_share),
            "sign_test": _sign_test(slope_share),
            "gap_t_test": _one_sample_t(gap_share),
            "gap_sign_test": _sign_test(gap_share),
            "mean_gap_near_minus_far": float(np.mean(gap_share)) if gap_share else 0.0,
        },
    }


def _decoy_statistics(rows: List[Dict[str, Any]],
                      per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recorded, labelled, and load-bearing on NOTHING.

    D1 is the pooled tick-level correlation -- exactly the statistic that made hazard
    proximity look confirmed in 785a. D2/D3 expose the between-seed channel that
    actually carried it. D4 asks whether 785a's seed-0 anomaly replicates at S=24."""
    seed_ids = [int(r["seed"]) for r in rows]
    hz = [float(r["hazard_prox_center"]) for r in rows]
    lv = [math.log10(max(float(r["var_total"]), 1e-30)) for r in rows]
    sh = [float(r["shares"].get(EXPECTED_INCUMBENT, 0.0)) for r in rows]

    usable = [p for p in per_seed if p.get("usable")]
    m_hz = [float(p["mean_hazard_prox_center"]) for p in usable]
    m_sh = [float(p["mean_incumbent_share"]) for p in usable]
    m_var = [math.log10(max(float(p["mean_var_total"]), 1e-30)) for p in usable]

    flagged = [p["seed"] for p in usable
               if abs(float(p["mean_incumbent_share"]) - SEED0_ANOMALY_785A["mean_share"])
               <= SEED0_FLAG_TOLERANCE]

    return {
        "load_bearing": False,
        "why_recorded": (
            "D1 is the statistic that misled in 785a: pooled across seeds it reproduced "
            "the 785 signs, while within seed the effect vanished and mostly reversed. "
            "It is reported here so the Simpson structure is legible in the manifest, "
            "and it routes NOTHING."
        ),
        "D1_pooled_tick_level": {
            "n_rows": len(rows),
            "r_hazard_vs_incumbent_share": round(_pearson(hz, sh), 6),
            "r_hazard_vs_log10_var": round(_pearson(hz, lv), 6),
            "reference_785a_pooled": {"share": -0.187, "log10_var": 0.171},
        },
        "D2_between_seed": {
            "n_seeds": len(usable),
            "r_mean_hazard_vs_mean_share": round(_pearson(m_hz, m_sh), 6),
            "r_mean_hazard_vs_mean_log10_var": round(_pearson(m_hz, m_var), 6),
            "reference_785a_between_seed": {"share": -0.78, "log10_var": 0.77},
        },
        "D3_covariance_decomposition": {
            "incumbent_share": _within_between_decomposition(seed_ids, hz, sh),
            "log10_var": _within_between_decomposition(seed_ids, hz, lv),
            "note": (
                "between_fraction near 1.0 reproduces the 785a Simpson signature: the "
                "pooled covariance is between-seed heterogeneity, not a within-seed effect"
            ),
        },
        "D4_between_seed_correlation_replication": {
            "seed0_anomaly_785a": SEED0_ANOMALY_785A,
            "per_seed_mean_incumbent_share": [
                {"seed": p["seed"], "mean_incumbent_share": p["mean_incumbent_share"]}
                for p in usable],
            "seeds_near_785a_seed0_share": flagged,
            "flag_tolerance": SEED0_FLAG_TOLERANCE,
            "replicates": bool(len(usable) >= 3
                               and abs(_pearson(m_hz, m_sh)) >= 0.5),
        },
    }


def _a3_profile(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The expressivity arbiter: does widening the action-option support move var_total?

    The verdict rests on the MEAN OF PER-SEED SPANS, not on the pooled profile -- the
    same within-seed discipline the load-bearing hazard criteria use. Pooling here would
    reproduce, inside the arbiter itself, exactly the between-seed contamination this run
    exists to rule out. The pooled profile is retained as a secondary readout only.
    """
    by_level: Dict[float, List[Dict[str, Any]]] = {}
    for r in rows:
        by_level.setdefault(float(r["ao_std_floor"]), []).append(r)

    n_seeds_total = len({int(r["seed"]) for r in rows}) or 1
    levels, means, per_seed_ok = [], [], 0
    for ao in sorted(by_level):
        lr = by_level[ao]
        # Pooled bucket spans every seed, so the floor scales with the seed count.
        # Applying the per-seed floor here would silently empty the pooled profile.
        if len(lr) < MIN_ROWS_PER_SEED_LEVEL * n_seeds_total:
            continue
        levels.append(ao)
        means.append(float(np.mean(
            [math.log10(max(float(r["var_total"]), 1e-30)) for r in lr])))

    by_seed: Dict[int, Dict[float, List[float]]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), {}).setdefault(
            float(r["ao_std_floor"]), []).append(
                math.log10(max(float(r["var_total"]), 1e-30)))
    per_seed_rows = []
    for seed in sorted(by_seed):
        lv = by_seed[seed]
        aos = sorted(lv)
        if len(aos) < 2:
            continue
        per_seed_ok += 1
        mv = [float(np.mean(lv[a])) for a in aos]
        per_seed_rows.append({"seed": seed, "ao_levels": aos,
                              "mean_log10_var": [round(v, 6) for v in mv],
                              "span": round(max(mv) - min(mv), 6)})

    rho = _spearman_rho(levels, means) if len(levels) >= 3 else 0.0
    span = (max(means) - min(means)) if means else 0.0

    # LOAD-BEARING for this arm: the within-seed spans and their per-seed monotonicity.
    seed_spans = [float(p["span"]) for p in per_seed_rows]
    seed_rhos = [_spearman_rho(p["ao_levels"], p["mean_log10_var"])
                 for p in per_seed_rows if len(p["ao_levels"]) >= 3]
    mean_seed_span = float(np.mean(seed_spans)) if seed_spans else 0.0
    min_seed_span = float(np.min(seed_spans)) if seed_spans else 0.0
    monotone_frac = (sum(1 for r in seed_rhos if r >= A3_MONOTONE_RHO_FLOOR)
                     / len(seed_rhos)) if seed_rhos else 0.0
    return {
        "mean_per_seed_log10_var_span_LOAD_BEARING": round(mean_seed_span, 6),
        "min_per_seed_log10_var_span": round(min_seed_span, 6),
        "per_seed_monotone_fraction": round(monotone_frac, 4),
        "per_seed_rhos": [round(r, 4) for r in seed_rhos],
        "pooled_ao_levels_SECONDARY": levels,
        "pooled_mean_log10_var_by_level_SECONDARY": [round(v, 6) for v in means],
        "pooled_rho_monotone_SECONDARY": round(rho, 6),
        "pooled_log10_var_span_SECONDARY": round(span, 6),
        "log10_var_span": round(mean_seed_span, 6),
        "n_seeds_with_profile": per_seed_ok,
        "per_seed": per_seed_rows,
        "expressive": bool(per_seed_ok >= 2
                           and mean_seed_span >= A3_MIN_LOGVAR_SPAN
                           and monotone_frac >= 0.5),
        "role": (
            "ARBITER for the null-vs-inexpressive discrimination. Expressive=True means "
            "the cross-candidate variance channel demonstrably responds to a known "
            "mover, so a flat hazard response in A1 is a GENUINE null. Expressive=False "
            "means the channel is inert regardless of input and nothing about hazard "
            "can be concluded from this run."
        ),
    }


def _fidelity(rows: List[Dict[str, Any]], telems: List[Dict[str, Any]]) -> Dict[str, Any]:
    lvl = [float(r["assigned_level_idx"]) for r in rows]
    cen = [float(r["hazard_prox_center"]) for r in rows]
    urg = [float(r["realized_urgency"]) for r in rows]

    by_level: Dict[int, List[float]] = {}
    for r in rows:
        by_level.setdefault(int(r["assigned_level_idx"]), []).append(
            float(r["hazard_prox_center"]))
    realized_means = {str(k): round(float(np.mean(v)), 6) for k, v in sorted(by_level.items())}

    miss = [float(t["pin_miss_frac"]) for t in telems]
    return {
        "corr_assigned_level_vs_realized_center": round(_pearson(lvl, cen), 6),
        "corr_assigned_level_vs_realized_urgency": round(_pearson(lvl, urg), 6),
        "realized_center_mean_by_level": realized_means,
        "expected_center_by_level": {
            str(i): round(HAZARD_PROX_EXPECTED[i], 6)
            for i in range(len(HAZARD_PROX_EXPECTED))},
        "pin_miss_frac_mean": round(float(np.mean(miss)), 6) if miss else 0.0,
        "pin_miss_frac_max": round(float(np.max(miss)), 6) if miss else 0.0,
        "urgency_spread": round(
            (float(np.max(urg)) - float(np.min(urg))) if urg else 0.0, 6),
    }


# ------------------------------------------------------------------ #
# Preconditions                                                      #
# ------------------------------------------------------------------ #
def _is_hazard_sweep(ctx: Dict[str, Any]) -> bool:
    return ctx.get("kind") == "hazard_sweep"


def _is_ao_sweep(ctx: Dict[str, Any]) -> bool:
    return ctx.get("kind") == "ao_sweep"


def _incumbent_is_channel(ctx: Dict[str, Any]) -> bool:
    """The >=2-nontrivial-components gate is meaningful ONLY where the incumbent is a
    modulatory CH: channel. Asserting it at a primary-component incumbent is the
    generalisation of the rule 785 derived for its P1 and failed to apply to its P7."""
    return str(ctx.get("expected_incumbent", "")).startswith("CH:")


def _count_floor(n_seeds: int, frac: float) -> float:
    """A '> threshold' floor demanding at least ceil(frac * n) of n seeds, and always at
    least 2. Returned as a half-integer so the indexer's `measured > threshold` recompute
    lands on the intended integer count."""
    need = max(2, math.ceil(frac * max(1, n_seeds)))
    return float(need) - 0.5


def _precondition_specs(n_hazard_seeds: int, n_a3_seeds: int) -> List[PreconditionSpec]:
    """Specs are built against the ACTUAL seed lists so the seed-count floors scale with
    the run (full vs dry-run) instead of being unsatisfiable at any smaller scale."""
    return [
        PreconditionSpec(
            name="manipulation_fidelity",
            description=("|corr(assigned proximity level, realized hazard_prox_center)| -- "
                         "the first stage. Near 1 by construction because the hazard is "
                         "re-pinned every env step."),
            control="exogenous re-pin at a pre-registered distance grid",
            threshold=FIDELITY_CORR_FLOOR,
            direction="lower",
            applies_to=_is_hazard_sweep,
            applies_note="A3 holds hazard distance fixed, so there is no level to correlate",
            structural_max=lambda ctx: 1.0,
        ),
        PreconditionSpec(
            name="pin_miss_frac",
            description=("fraction of re-pins that could not realise the exact assigned "
                         "distance (agent near a corner) -- a CEILING"),
            control="interior-cell enumeration at each re-pin",
            threshold=PIN_MISS_FRAC_CEILING,
            direction="upper",
            structural_min=lambda ctx: 0.0,
        ),
        PreconditionSpec(
            name="seeds_with_usable_slope",
            description=("seeds contributing a within-seed slope; the load-bearing t-test "
                         "is across these"),
            control="MIN_LEVELS_POPULATED levels each with >= MIN_ROWS_PER_SEED_LEVEL rows",
            threshold=_count_floor(n_hazard_seeds, MIN_SEEDS_WITH_SLOPE_FRAC),
            direction="lower",
            applies_to=_is_hazard_sweep,
            applies_note="A3 sweeps ao_std_floor, not proximity; it has no per-seed slope",
            structural_max=lambda ctx: float(len(ctx.get("seeds", []))),
        ),
        PreconditionSpec(
            name="a3_seeds_with_profile",
            description="seeds contributing an ao_std_floor profile in the expressivity arm",
            control="at least two ao levels populated per seed",
            threshold=_count_floor(n_a3_seeds, MIN_A3_SEEDS_WITH_PROFILE_FRAC),
            direction="lower",
            applies_to=_is_ao_sweep,
            applies_note="only the expressivity arm sweeps ao_std_floor",
            structural_max=lambda ctx: float(len(ctx.get("seeds", []))),
        ),
        PreconditionSpec(
            name="expressivity_control_log10_var_span",
            description=(
                "P0 READINESS-ASSERT. Movement in log10 var_total produced by the "
                "ao_std_floor positive control. This asserts the SAME statistic the "
                "load-bearing C1 criterion routes on -- movement in log10 var_total -- not a "
                "magnitude proxy for it, so a below-floor reading means 'the cross-candidate "
                "variance channel cannot express an effect', never 'hazard geometry "
                "falsified'."),
            control=(
                "widening support_preserving_ao_std_floor over {0.2,0.6,1.2} makes "
                "candidates differ more, a known-positive mover of cross-candidate variance"),
            threshold=A3_MIN_LOGVAR_SPAN,
            direction="lower",
            kind="readiness",
            applies_to=_is_ao_sweep,
            applies_note="only the expressivity positive control sweeps ao_std_floor",
            structural_max=lambda ctx: 10.0,
        ),
        PreconditionSpec(
            name="incumbent_identity_as_preregistered",
            description=("the measured incumbent is harm_weighted with a margin over the "
                         "runner-up"),
            control="pre-registered from 785a's surviving regime",
            threshold=INCUMBENT_MARGIN_FLOOR,
            direction="lower",
            structural_max=lambda ctx: 1.0,
        ),
        PreconditionSpec(
            name="cross_candidate_score_variance",
            description="mean cross-candidate total score variance is non-zero",
            control="a degenerate candidate set makes every share statistic meaningless",
            threshold=SCORE_VARIANCE_FLOOR,
            direction="lower",
            structural_max=lambda ctx: 1.0,
        ),
        PreconditionSpec(
            name="n_components_with_nontrivial_share",
            description=("components above the |0.01| share floor; a single-component "
                         "decomposition cannot measure redistribution"),
            control="covariance-correct shares summing to exactly 1",
            threshold=1.5,
            direction="lower",
            applies_to=_incumbent_is_channel,
            applies_note=(
                "meaningful only where the incumbent is a modulatory CH: channel. This is "
                "the 785 P7 lesson generalised: applying it at a primary-component "
                "incumbent would make the regime structurally un-passable and silently "
                "vacate a valid arm."
            ),
            structural_max=lambda ctx: float(len(PRIMARY_COMPONENTS)),
        ),
    ]


# ------------------------------------------------------------------ #
# Criteria and self-route                                            #
# ------------------------------------------------------------------ #
def _score_criterion(stats: Dict[str, Any], abs_floor: float, ref_slope: float,
                     predicted_sign: int) -> Dict[str, Any]:
    """Signed three-way verdict on one within-seed slope distribution."""
    t = stats["t_test"]
    slopes = stats["slopes"]
    mean = float(t["mean"])
    se = float(t["se"])
    floor = max(abs_floor, SE_MULTIPLIER * se)
    clears = bool(abs(mean) > floor) and bool(t["significant"])

    gap_t = stats["gap_t_test"]
    gap_mean = float(gap_t["mean"])
    # The robustness statistic must AGREE IN SIGN with the slope, or the effect is not
    # stable enough to route on.
    sign_agrees = bool(mean == 0.0 or gap_mean == 0.0 or (mean > 0) == (gap_mean > 0))

    if not clears or not sign_agrees:
        verdict = "flat"
    elif (mean > 0) == (predicted_sign > 0):
        verdict = "as_predicted"
    else:
        verdict = "sign_reversed"

    frac = abs(mean / ref_slope) if abs(ref_slope) > 0 else 0.0
    return {
        "verdict": verdict,
        "mean_slope": round(mean, 6),
        "sd_slope": round(float(t["sd"]), 6),
        "se_slope": round(se, 6),
        "t": round(float(t["t"]), 4),
        "df": t["df"],
        "t_crit": t["t_crit"],
        "significant": bool(t["significant"]),
        "effect_floor_applied": round(floor, 6),
        "abs_floor": abs_floor,
        "clears_floor": bool(abs(mean) > floor),
        "robustness_gap_mean": round(gap_mean, 6),
        "robustness_sign_agrees": sign_agrees,
        "n_seeds": t["n"],
        "per_seed_slopes": slopes,
        "reference_slope_785": round(ref_slope, 6),
        "fraction_of_785_reference": round(frac, 4),
        "small_effect": bool(verdict == "as_predicted" and frac < SMALL_EFFECT_FRACTION),
    }


def _self_route(gate_green: bool, expressive: bool, c1: Optional[Dict[str, Any]],
                c2: Optional[Dict[str, Any]], degeneracy_reason: str) -> Dict[str, Any]:
    """The interpretation grid. Every cell has a route, INCLUDING sign-reversed and
    directionally-right-but-small -- 785 had no cell for its observed monotone decrease
    and a strong contrary result got buried as 'substrate not ready'."""
    if not gate_green or c1 is None or c2 is None:
        return {"outcome": "FAIL", "evidence_direction": "non_contributory",
                "label": "substrate_not_ready_requeue", "non_degenerate": False,
                "degeneracy_reason": degeneracy_reason,
                "ledger_effect": "H-endogenous-hazard-geometry stays alive; no bit claimed",
                "cell": "gate_red"}

    if not expressive:
        # The pre-declared inexpressive branch. Checked BEFORE reading C1/C2, because a
        # flat reading on an inert channel is not evidence about hazard at all.
        return {
            "outcome": "FAIL", "evidence_direction": "non_contributory",
            "label": "measurement_channel_inexpressive", "non_degenerate": False,
            "degeneracy_reason": (
                "expressivity positive control (A3) did not move log10 var_total: the "
                "cross-candidate variance channel is inert regardless of input, so a "
                "flat hazard response carries no information about hazard geometry"),
            "ledger_effect": "H-endogenous-hazard-geometry stays alive; no bit claimed",
            "cell": "inexpressive",
        }

    amp = c1["verdict"]
    red = c2["verdict"]
    small = c1.get("small_effect") or c2.get("small_effect")

    if amp == "as_predicted" and red == "as_predicted":
        if small:
            return {"outcome": "PASS", "evidence_direction": "mixed",
                    "label": "hazard_geometry_directional_but_subreference",
                    "non_degenerate": True, "degeneracy_reason": None,
                    "ledger_effect": (
                        "H-endogenous-hazard-geometry SUPPORTED in direction but the "
                        "effect is under 25% of the 785 reference, so hazard geometry "
                        "cannot be the whole account of the 785 profile"),
                    "cell": "directionally_right_but_small"}
        return {"outcome": "PASS", "evidence_direction": "supports",
                "label": "hazard_geometry_reproduces_785_profile",
                "non_degenerate": True, "degeneracy_reason": None,
                "ledger_effect": (
                    "H-endogenous-hazard-geometry CONFIRMED under exogenous "
                    "manipulation with a within-seed load-bearing statistic"),
                "cell": "reproduces_785"}

    if amp == "as_predicted" and red == "flat":
        return {"outcome": "FAIL", "evidence_direction": "mixed",
                "label": "hazard_amplifies_without_redistributing",
                "non_degenerate": True, "degeneracy_reason": None,
                "ledger_effect": (
                    "partial: hazard drives the amplification half of the 785 profile "
                    "but not the redistribution half"),
                "cell": "mixed_amp_only"}

    if amp == "flat" and red == "as_predicted":
        return {"outcome": "FAIL", "evidence_direction": "mixed",
                "label": "hazard_redistributes_without_amplifying",
                "non_degenerate": True, "degeneracy_reason": None,
                "ledger_effect": (
                    "partial: hazard drives the redistribution half of the 785 profile "
                    "but not the amplification half"),
                "cell": "mixed_redist_only"}

    if "sign_reversed" in (amp, red):
        return {"outcome": "FAIL", "evidence_direction": "weakens",
                "label": "hazard_geometry_sign_reversed",
                "non_degenerate": True, "degeneracy_reason": None,
                "ledger_effect": (
                    "H-endogenous-hazard-geometry ELIMINATED: proximity moves the "
                    "geometry in the OPPOSITE direction to the 785 profile it was "
                    "invoked to explain"),
                "cell": "sign_reversed"}

    return {"outcome": "FAIL", "evidence_direction": "does_not_support",
            "label": "hazard_geometry_inert_on_selection_variance",
            "non_degenerate": True, "degeneracy_reason": None,
            "ledger_effect": (
                "H-endogenous-hazard-geometry ELIMINATED at the harm_weighted incumbent "
                "on the SD-011 commit-threshold route: exogenous proximity moves neither "
                "half of the profile, and the expressivity control confirms the channel "
                "could have shown it"),
            "cell": "flat_with_expressive_channel"}


# ------------------------------------------------------------------ #
# Run                                                                #
# ------------------------------------------------------------------ #
def _arm_contexts(dry_run: bool) -> List[Dict[str, Any]]:
    ctxs = []
    for arm in ARMS:
        seeds = DRY_RUN_SEEDS if dry_run else arm["seeds"]
        ctxs.append({"id": arm["id"], "kind": arm["kind"], "seeds": list(seeds),
                     "expected_incumbent": EXPECTED_INCUMBENT})
    return ctxs


def run_experiment(dry_run: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    t0 = time.perf_counter()
    n_ticks = DRY_RUN_TICKS if dry_run else TICKS_PER_SEED

    arm_contexts = _arm_contexts(dry_run)
    # Design-time proof that no arm's gate is unsatisfiable from its PRE-REGISTERED
    # config. This is the check that would have caught 785's expected_incumbent_share
    # = 1.043 against a >= 2-components gate for free, before any compute was spent.
    specs = _precondition_specs(
        n_hazard_seeds=len(arm_contexts[0]["seeds"]),
        n_a3_seeds=len(arm_contexts[2]["seeds"]),
    )
    audited = assert_no_structurally_unsatisfiable_gate(specs, arm_contexts)

    all_rows: List[Dict[str, Any]] = []
    arm_results: List[Dict[str, Any]] = []
    rows_by_arm: Dict[str, List[Dict[str, Any]]] = {a["id"]: [] for a in ARMS}
    telems_by_arm: Dict[str, List[Dict[str, Any]]] = {a["id"]: [] for a in ARMS}

    for arm in ARMS:
        seeds = DRY_RUN_SEEDS if dry_run else arm["seeds"]
        if arm["kind"] == "ao_sweep":
            cells = [(s, ao, A3_FIXED_DISTANCE) for s in seeds for ao in A3_AO_STD_FLOORS]
        else:
            cells = [(s, float(arm["ao_std_floor"]), None) for s in seeds]

        cell_ticks = n_ticks if arm["kind"] != "ao_sweep" else min(
            n_ticks, A3_TICKS_PER_SEED)
        for seed, ao, fixed_d in cells:
            probe_slice = _build_agent_and_env(seed, ao)[3]
            probe_slice = dict(probe_slice)
            probe_slice.update({
                "_arm": arm["id"], "_urgency_mode": arm["urgency_mode"],
                "_urgency_clamp": arm["urgency_clamp"],
                "_hazard_distances": HAZARD_DISTANCES,
                "_fixed_distance": fixed_d, "_window_steps": WINDOW_STEPS,
                "_n_ticks": cell_ticks,
            })
            with arm_cell(
                seed,
                config_slice=probe_slice,
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                rows, telem = _collect_cell(seed, arm, ao, fixed_d, cell_ticks)
                row = {"arm_id": arm["id"], "seed": seed, **telem}
                cell.stamp(row)   # must be INSIDE the cell, before RNG state is released
            rows_by_arm[arm["id"]].extend(rows)
            telems_by_arm[arm["id"]].append(telem)
            all_rows.extend(rows)
            arm_results.append(row)
            print(f"verdict: {'PASS' if telem['n_rows'] > 0 else 'FAIL'}", flush=True)

    # ---------------- analysis ---------------- #
    arm_analyses: List[Dict[str, Any]] = []
    arm_gates: List[Dict[str, Any]] = []
    slope_stats_by_arm: Dict[str, Dict[str, Any]] = {}
    a3_profile: Dict[str, Any] = {}

    for arm, ctx in zip(ARMS, arm_contexts):
        rows = _committed(rows_by_arm[arm["id"]])
        telems = telems_by_arm[arm["id"]]

        shares_mean: Dict[str, float] = {}
        if rows:
            keys = set()
            for r in rows:
                keys.update(r["shares"].keys())
            for k in keys:
                shares_mean[k] = float(np.mean(
                    [float(r["shares"].get(k, 0.0)) for r in rows]))
        ordered = sorted(shares_mean.items(), key=lambda kv: -abs(kv[1]))
        measured_incumbent = ordered[0][0] if ordered else ""
        runner_up = abs(ordered[1][1]) if len(ordered) > 1 else 0.0
        margin = (abs(ordered[0][1]) - runner_up) if ordered else 0.0
        n_nontrivial = sum(1 for _k, v in shares_mean.items()
                           if abs(v) >= NONTRIVIAL_SHARE_FLOOR)
        var_mean = float(np.mean([float(r["var_total"]) for r in rows])) if rows else 0.0

        fid = _fidelity(rows, telems) if rows else {}

        if arm["kind"] == "hazard_sweep":
            slopes = _per_seed_slopes(rows)
            slope_stats_by_arm[arm["id"]] = slopes
            n_slope_seeds = slopes["n_seeds_with_slope"]
            decoys = _decoy_statistics(rows, slopes["per_seed"]) if rows else {}
            a3_n = 0.0
        else:
            slopes = {}
            a3_profile = _a3_profile(rows) if rows else {}
            n_slope_seeds = 0
            decoys = {}
            a3_n = float(a3_profile.get("n_seeds_with_profile", 0))

        measured = {
            "pin_miss_frac": float(np.mean([t["pin_miss_frac"] for t in telems]))
            if telems else 1.0,
            "incumbent_identity_as_preregistered": float(margin),
            "cross_candidate_score_variance": var_mean,
        }
        met_overrides = {
            "incumbent_identity_as_preregistered": bool(
                measured_incumbent == EXPECTED_INCUMBENT
                and margin > INCUMBENT_MARGIN_FLOOR),
        }
        if arm["kind"] == "hazard_sweep":
            measured["manipulation_fidelity"] = abs(float(
                fid.get("corr_assigned_level_vs_realized_center", 0.0)))
            measured["seeds_with_usable_slope"] = float(n_slope_seeds)
        else:
            measured["a3_seeds_with_profile"] = a3_n
            measured["expressivity_control_log10_var_span"] = float(
                a3_profile.get("log10_var_span", 0.0))
        if _incumbent_is_channel(ctx):
            measured["n_components_with_nontrivial_share"] = float(n_nontrivial)

        gate = evaluate_arm_gate(arm["id"], ctx, specs, measured,
                                 met_overrides=met_overrides)
        arm_gates.append(gate)

        arm_analyses.append({
            "arm_id": arm["id"],
            "kind": arm["kind"],
            "urgency_mode": arm["urgency_mode"],
            "n_committed_rows": len(rows),
            "measured_incumbent": measured_incumbent,
            "incumbent_margin": round(margin, 6),
            "mean_shares": {k: round(v, 6) for k, v in shares_mean.items()},
            "n_components_with_nontrivial_share": n_nontrivial,
            "mean_var_total": var_mean,
            "zworld_participation_ratio_mean": round(float(np.mean(
                [t["zworld_participation_ratio"] for t in telems])), 4) if telems else 0.0,
            "fidelity": fid,
            "within_seed_slopes_LOAD_BEARING": slopes,
            "decoys_NOT_LOAD_BEARING": decoys,
            "gate_green": gate["gate_green"],
        })

    aggregate = aggregate_arm_gates(arm_gates)

    primary = "hazard_exog_urgency_clamped"
    primary_green = primary in aggregate["green_arms"]
    a3_green = "expressivity_positive_control" in aggregate["green_arms"]
    expressive = bool(a3_profile.get("expressive", False)) and a3_green

    c1 = c2 = None
    if primary_green and slope_stats_by_arm.get(primary):
        ps = slope_stats_by_arm[primary]
        c1 = _score_criterion(ps["log10_var"], LOGVAR_SLOPE_ABS_FLOOR,
                              REF_LOGVAR_SLOPE, predicted_sign=+1)
        c2 = _score_criterion(ps["incumbent_share"], SHARE_SLOPE_ABS_FLOOR,
                              REF_SHARE_SLOPE, predicted_sign=-1)

    route = _self_route(primary_green, expressive, c1, c2,
                        aggregate.get("degeneracy_reason") or "")

    criteria = [
        {"name": "C1_amplification_within_seed_slope", "load_bearing": True,
         "arm": primary, "predicted_sign": "+ (near hazard -> higher var_total)",
         "passed": bool(c1 and c1["verdict"] == "as_predicted"), "detail": c1},
        {"name": "C2_redistribution_within_seed_slope", "load_bearing": True,
         "arm": primary, "predicted_sign": "- (near hazard -> lower incumbent share)",
         "passed": bool(c2 and c2["verdict"] == "as_predicted"), "detail": c2},
        {"name": "C3_expressivity_positive_control", "load_bearing": False,
         "arm": "expressivity_positive_control",
         "passed": bool(expressive), "detail": a3_profile,
         "role": "arbiter for the null-vs-inexpressive discrimination"},
    ]
    criteria_non_degenerate = arm_criteria_non_degenerate(
        {primary: ["C1_amplification_within_seed_slope",
                   "C2_redistribution_within_seed_slope"],
         "expressivity_positive_control": ["C3_expressivity_positive_control"]},
        aggregate,
    )

    free_arm = next((a for a in arm_analyses
                     if a["arm_id"] == "hazard_exog_urgency_free"), None)
    mediation = {}
    if free_arm:
        r_urg = float(free_arm.get("fidelity", {}).get(
            "corr_assigned_level_vs_realized_urgency", 0.0))
        mediation = {
            "corr_assigned_proximity_vs_realized_urgency": round(r_urg, 6),
            "premise_holds": bool(r_urg > 0.2),
            "note": (
                "H's own premise is that high-urgency ticks are near-hazard ticks. A "
                "near-zero or negative correlation here falsifies that premise at the "
                "first stage, independently of what the geometry does. It is also the "
                "second null-vs-inexpressive tell: a live input path plus a flat "
                "geometry localises the failure to the geometry stage."),
        }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "seeds_hazard_arms": DRY_RUN_SEEDS if dry_run else SEEDS,
        "seeds_expressivity_arm": DRY_RUN_SEEDS if dry_run else A3_SEEDS,
        "ticks_per_seed": n_ticks,
        "a3_ticks_per_seed": min(n_ticks, A3_TICKS_PER_SEED),
        "window_steps": WINDOW_STEPS,
        "hazard_distances_far_to_near": HAZARD_DISTANCES,
        "hazard_prox_expected_by_level": [round(v, 6) for v in HAZARD_PROX_EXPECTED],
        "a3_fixed_distance": A3_FIXED_DISTANCE,
        "a3_ao_std_floors": A3_AO_STD_FLOORS,
        "urgency_clamp": URGENCY_CLAMP,
        "urgency_weight_baseline": URGENCY_WEIGHT_BASELINE,
        "alpha_world": ALPHA_WORLD,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "episode_start_cell": list(EPISODE_START_CELL),
        "env": {"use_proxy_fields": True, "hazard_harm": 0.5,
                "num_hazards": 1, "env_drift_prob": 0.0},
        "expected_incumbent": EXPECTED_INCUMBENT,
        "expected_incumbent_share": EXPECTED_INCUMBENT_SHARE,
        "arms": [{k: v for k, v in a.items() if k != "seeds"} for a in ARMS],
        "effect_floors": {
            "log10_var_slope_abs_floor": LOGVAR_SLOPE_ABS_FLOOR,
            "incumbent_share_slope_abs_floor": SHARE_SLOPE_ABS_FLOOR,
            "se_multiplier": SE_MULTIPLIER,
            "small_effect_fraction": SMALL_EFFECT_FRACTION,
        },
        "use_harm_variance_commit": False,
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": route["outcome"],
        "evidence_direction": route["evidence_direction"],
        "timestamp_utc": ts,
        "dry_run": dry_run,
        "non_degenerate": bool(route["non_degenerate"] and aggregate["non_degenerate"]),
        "degeneracy_reason": route["degeneracy_reason"] or aggregate.get("degeneracy_reason"),
        "config": full_config,
        "seeds": DRY_RUN_SEEDS if dry_run else SEEDS,
        "arm_results": arm_results,
        "arm_analyses": arm_analyses,
        "per_arm_gate": aggregate["per_arm_gate"],
        "criteria": criteria,
        "metrics": {
            "primary_arm": primary,
            "primary_arm_gate_green": primary_green,
            "expressivity_control_expressive": expressive,
            "C1_mean_slope_log10_var": (c1 or {}).get("mean_slope"),
            "C2_mean_slope_incumbent_share": (c2 or {}).get("mean_slope"),
            "C1_fraction_of_785_reference": (c1 or {}).get("fraction_of_785_reference"),
            "C2_fraction_of_785_reference": (c2 or {}).get("fraction_of_785_reference"),
            "mediation_premise": mediation,
        },
        "interpretation": {
            "label": route["label"],
            "cell": route["cell"],
            "ledger_effect": route["ledger_effect"],
            "preconditions": aggregate["adjudication_preconditions"],
            "preconditions_scope_note": aggregate["per_arm_gate"].get(
                "preconditions_scope_note"),
            "criteria_non_degenerate": criteria_non_degenerate,
            "hypothesis_target": {
                "registry_question": "arousal-variance-amplifier",
                "hypothesis": "H-endogenous-hazard-geometry",
                "registry_path": "REE_assembly/evidence/planning/hypothesis_space_registry.v1.json",
                "resolve_via": "/failure-autopsy Step 9b -- the registry's SINGLE producer",
            },
            "load_bearing_statistic": (
                "mean of per-seed WITHIN-SEED OLS slopes on the assigned proximity level "
                "index, tested with a one-sample t-test across seeds against a "
                "pre-registered critical value, with an SD-of-effect plus absolute floor. "
                "The pooled tick-level correlation is a DECOY and routes nothing: it is "
                "the statistic that made hazard proximity look confirmed in 785a."
            ),
            "null_vs_inexpressive_discriminator": {
                "caveat": (
                    "z_world is under-differentiated (participation ratio ~1.06 at "
                    "world_dim=128, absolute variances ~1.2e-05), so a null invites the "
                    "objection that the substrate cannot EXPRESS the effect -- which "
                    "bites the fix, not the measurement."),
                "arbiter": (
                    "A3 expressivity positive control. Expressive -> a flat A1 is a "
                    "GENUINE null and the leg is eliminated. Not expressive -> "
                    "measurement_channel_inexpressive, nothing concluded, leg stays alive."),
                "second_tell": (
                    "if A2 shows proximity strongly moves realized urgency (input path "
                    "live) while geometry stays flat, the failure localises to the "
                    "geometry stage rather than to the manipulation."),
                "a3_expressive": expressive,
                "measured_participation_ratios": {
                    a["arm_id"]: a["zworld_participation_ratio_mean"] for a in arm_analyses},
                "participation_ratio_note": (
                    "The ~1.06 figure in the 785a adjudication is the reference for the "
                    "inexpressive objection. This run measures PR over z_world sampled "
                    "across ticks within each cell; the 2026-07-19 dry run returned ~5.0, "
                    "materially higher. If the full run reproduces that, the "
                    "under-differentiation objection is WEAKER than the 785a caveat "
                    "assumed, and a null here is correspondingly harder to attribute to "
                    "the substrate being unable to express the effect. Reported, not "
                    "asserted: the two figures may be computed over different populations, "
                    "so this is a flag for adjudication rather than a correction."),
            },
            "scope": {
                "route": "SD-011 commit-threshold route only",
                "gated_quantity": (
                    "z_world RUNNING VARIANCE (world-model stability), not candidate "
                    "separation -- use_harm_variance_commit is OFF"),
                "channel_agnosticism": "OUT OF SCOPE",
                "dropped_regime_evidence": DROPPED_REGIME_EVIDENCE,
            },
            "design_note": (
                "Hazard proximity is MANIPULATED, not observed: the single hazard is "
                "re-pinned to the assigned Manhattan distance from the agent's current "
                "cell at every env step, with the bearing re-drawn each time. The hazard "
                "therefore tracks the agent, which is ecologically odd and is exactly "
                "what exogenous manipulation of proximity means -- the price of breaking "
                "the endogeneity, the same price 785a paid on the urgency axis."),
            "reference_profile_785": REFERENCE_PROFILE_785,
            "structural_gate_audit": audited,
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
        "custom_information": {
            "per_tick_rows_schema": {
                "note": (
                    "AUTHORITATIVE per-tick record, embedded IN the manifest. A sidecar "
                    "written next to the manifest on a cloud worker is NEVER transported "
                    "under Phase 3 (PHASE3_DISABLE_RUNNER_RESULT_PUSH=1) -- that is how "
                    "785 lost its per-tick sink. Rows are recorded ONLY on fresh E3 "
                    "selections (last_score_diagnostics cleared before every "
                    "select_action), so there is no pseudo-replication."),
                "fields": [
                    "arm", "ao_std_floor", "seed", "tick", "assigned_distance",
                    "assigned_level_idx", "realized_distance", "pin_missed",
                    "hazard_prox_center (PRIMARY covariate)", "hazard_prox_max",
                    "hazard_prox_mean", "realized_urgency", "natural_urgency",
                    "effective_threshold", "commit_variance", "commit_gate_mode",
                    "committed", "temperature", "var_total", "shares",
                    "per_candidate (sampled)",
                ],
                "primary_covariate_note": (
                    "hazard_prox_center is index 12 of the 25-dim hazard_field_view (the "
                    "agent's own cell) and is monotone in distance across the whole grid. "
                    "hazard_prox_max SATURATES at 1.0 for every d <= 2 because the 5x5 "
                    "window spans +/-2, which is why 785a's use of it was weaker."),
                "level_ordering_note": (
                    "assigned_level_idx increases with PROXIMITY (distances ordered far "
                    "-> near), so H predicts a POSITIVE slope on log10 var_total and a "
                    "NEGATIVE slope on incumbent share."),
            },
            "e3_cadence_correction": (
                "785a's pseudo_replication_note attributed skipped ticks to the "
                "commitment latch. The actual driver is the E3 CADENCE: on a non-E3 tick "
                "the agent returns early at agent.py:5429 and select_action is never "
                "reached at all (heartbeat.e3_steps_per_tick, default 10)."),
            "seed0_anomaly_785a": SEED0_ANOMALY_785A,
            "per_tick_rows": all_rows,
        },
    }

    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=DRY_RUN_SEEDS if dry_run else SEEDS,
        script_path=Path(__file__),
        started_at=t0,
    )
    return manifest, all_rows


# ------------------------------------------------------------------ #
def main() -> Tuple[Dict[str, Any], Any, bool]:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t_start = time.perf_counter()
    manifest, all_rows = run_experiment(dry_run=args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        OUT_DIR,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=manifest.get("seeds"),
        script_path=Path(__file__),
        started_at=t_start,
    )

    if not args.dry_run:
        # NON-AUTHORITATIVE convenience copy. The manifest-embedded rows are the record.
        sidecar = Path(out_path).with_name(Path(out_path).stem + "_per_tick.jsonl")
        try:
            with open(sidecar, "w", encoding="utf-8") as fh:
                for r in all_rows:
                    fh.write(json.dumps(r) + "\n")
        except OSError as exc:
            print(f"  [warn] non-authoritative sidecar not written: {exc}", flush=True)

    print("")
    print(f"run_id: {manifest['run_id']}")
    print(f"outcome: {manifest['outcome']}  direction: {manifest['evidence_direction']}")
    print(f"label: {manifest['interpretation']['label']}  "
          f"cell: {manifest['interpretation']['cell']}")
    print(f"per_tick_rows: {len(all_rows)}")
    for gate in manifest["per_arm_gate"].get("red", []) or []:
        print(f"  RED arm {gate.get('arm')}: {gate.get('failed_preconditions')}")
    for a in manifest["arm_analyses"]:
        fid = a.get("fidelity") or {}
        print(f"  arm={a['arm_id']} rows={a['n_committed_rows']} "
              f"green={a['gate_green']} "
              f"fidelity_corr={fid.get('corr_assigned_level_vs_realized_center')} "
              f"realized_center_by_level={fid.get('realized_center_mean_by_level')}")
        sl = a.get("within_seed_slopes_LOAD_BEARING") or {}
        if sl:
            print(f"    seeds_with_slope={sl.get('n_seeds_with_slope')} "
                  f"mean_slope_log10var="
                  f"{sl.get('log10_var', {}).get('t_test', {}).get('mean')} "
                  f"mean_slope_share="
                  f"{sl.get('incumbent_share', {}).get('t_test', {}).get('mean')}")
        dec = a.get("decoys_NOT_LOAD_BEARING") or {}
        if dec:
            d1 = dec.get("D1_pooled_tick_level", {})
            print(f"    [DECOY, routes nothing] pooled r(hazard,share)="
                  f"{d1.get('r_hazard_vs_incumbent_share')} "
                  f"r(hazard,log10var)={d1.get('r_hazard_vs_log10_var')}")

    return manifest, out_path, bool(args.dry_run)


if __name__ == "__main__":
    _manifest, _out_path, _dry_run = main()
    _outcome_raw = str(_manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_dry_run,
    )
