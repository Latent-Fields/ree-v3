"""V3-EXQ-848b: ARC-005 -- does the calibrated dACC goal-readout channel route precision,
measured on a FINER (7-level) ladder that breaks the 3-point Spearman-rho resolution ceiling?

Successor to V3-EXQ-848a (measurement / test-design redesign, same scientific question, same
substrate, SAME pre-registered acceptance rule -- only the ladder RESOLUTION changes).
claim_ids = [ARC-005] ONLY.

=== WHY THIS SUCCESSOR EXISTS (848a's resolution ceiling, not a substrate gap) ===
848a CLOSED the wiring/calibration line of inquiry: the calibrated dACC goal-readout channel
(dacc_goal_readout_weight=0.5, dacc_goal_readout_normalize=True) was confirmed genuinely
engaged in that run (25,289 dACC bias calls; the build-time guard held). C_precision_monotonicity
still cleared only 1/10 units against the |Spearman rho| >= 0.60 bar -- BUT the per-unit rho
distribution was NOT a clean null. On 848a's 3-level ladder [0.0, 0.5, 1.0], the observed rho
values were EXACTLY {+0.5 x6, +1.0 x1, -0.5 x2, 0.0 x1} -- 7/10 positive-signed, six piled at
exactly +0.5. That is the arithmetic SIGNATURE of a 3-point-ladder Spearman-rho resolution
ceiling, not of no effect:

  On n=3 distinct points, Spearman rho can only take the values {-1.0, -0.5, 0.0, +0.5, +1.0}.
  A genuine but weak monotone tendency whose ordering is correct on 2 of the 3 pairwise
  comparisons (one out-of-order point due to substrate wiggle) lands at EXACTLY rho=+0.5 --
  below the 0.60 bar -- and is arithmetically INDISTINGUISHABLE from a coin-flip ordering that
  also happens to reach +0.5. 802's genuinely-null channels, by contrast, gave bit-identical
  rho=0.0 across seeds -- a qualitatively different pattern. 848a's own docstring anticipated
  this: "With only 3 ladder levels per unit, Spearman rho is a coarse statistic that cannot
  cleanly distinguish a weak real effect from pure sampling noise -- a future redesign should
  add ladder levels, not just seeds."

The 2026-08-03 governance walk (failure_autopsy_V3-EXQ-848a_2026-08-03) confirmed this reading
(mixed/standard, NOT substrate_ceiling) and routed the redesign here explicitly: add ladder
levels and/or a magnitude-aware statistic so the test can distinguish a weak-but-real monotonic
effect from measurement-resolution noise -- NOT further substrate work. The re-derive brake was
checked and does NOT fire (0 substrate_ceiling autopsies tag ARC-005).

=== WHY A FINER LADDER, AND WHY NOT A MAGNITUDE-FLOOR STATISTIC ===
848a's raw per-cell readouts (read directly from its manifest during this redesign) settle the
choice of statistic. The ladder's effect on log10-precision is real in SIGN but minuscule in
MAGNITUDE: across the 10 units the full-ladder delta (L2 - L0) had median +0.00009 and max
+0.00194 log10-units, while the CROSS-SEED SD of log10-precision was ~0.18 -- two-to-three
orders of magnitude larger. So:

  - The signal lives in ORDERING CONSISTENCY (7/10 units positive-signed), not in magnitude.
    A magnitude-based statistic gated on an ABSOLUTE effect-size floor would REJECT the real
    effect (any principled floor >> 0.002 log10-units), which is the wrong outcome. The
    task-brief phrase "magnitude-based statistic" is honoured here as RESOLUTION-INDEPENDENT,
    CONTINUOUS trend readouts recorded ALONGSIDE the primary rank criterion (Theil-Sen slope +
    Mann-Kendall significance), NOT as an absolute-floor gate.
  - The between-seed offset (~0.18) dwarfs the within-unit ladder effect (~0.001), so the
    UNIT OF ANALYSIS must remain the (content, seed) pair -- each unit is one seed, which
    controls for that offset. This is unchanged from 848/848a and is correct.

The fix is therefore purely MEASUREMENT RESOLUTION: replace the 3-level ladder with a 7-level
ladder [0/6, 1/6, ..., 6/6]. At n=7, Spearman rho under the null H0 has SD ~= 1/sqrt(n-1) =
0.408, so the SAME |rho| >= 0.60 bar (kept verbatim from 848/848a) is now ~1.47 SD -- a genuine
signal threshold -- and is ACHIEVABLE by a consistent weak monotone trend (a 7-point curve with
one or two out-of-order points gives rho ~0.7-0.9), where the 3-point ladder capped such a trend
at +0.5. A true null now SCATTERS near 0 rather than piling at +0.5, so the null band
(|rho| <= 0.10 in all units -> non_contributory) becomes cleanly separable too.

=== WHAT THIS EXPERIMENT IS, AND IS NOT ===
This is a MEASUREMENT redesign of 848a, under the "same scientific question, corrected
instrument" convention. Every SUBSTRATE and CONFIG element is UNCHANGED from 848a -- the
calibration fix (dacc_goal_readout_weight=0.5, dacc_goal_readout_normalize=True), the
channel-1/2-only ladder with channels 3/4 held fixed at L0, the content sets, the arena, the
schedule (200 warmup + 1800 measure ticks), the seeds [0..4], the DV (E3 log10-precision via
the canonical update_residue()/update_z_goal() path), the build-time guard and the whole-run
dACC-engagement check. The ONLY changes vs 848a are:
    (1) LEVELS: [0.0, 0.5, 1.0] (3)  ->  [0/6, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0] (7)
    (2) The primary criterion's rho is computed over the 7-level ladder (threshold and >=7/10
        count UNCHANGED), and per-unit Mann-Kendall significance + Theil-Sen slope + Kendall
        tau are recorded as resolution-independent SECONDARY diagnostics.
    (3) A manifest-level RESOLUTION diagnostic (distinct rho values, achievable-grid note)
        documenting that the 3-point ceiling is broken.
Channel-1-sub-pathway isolation (dACC goal-readout alone vs serotonin alone), channel 2 under
uncommitted selection (TRACK B), and the mode-occupancy side of ARC-005 (covered by 802/846
GAP-B) all remain OUT OF SCOPE, exactly as in 848a.

=== DESIGN: SINGLE-FACTOR 7-LEVEL LADDER ON CHANNELS 1+2 ONLY ===
Factor    channel level (1+2 only)   [0/6 .. 6/6], laddered together (linear interp of BASE)
Held fixed EVERY cell                channels 3+4 at their L0 substrate-default values
Content   set A / set B (independent units for the monotonicity criterion, not a
                          dissociation factor)

  cells: L0..L6 x {A,B} x 5 seeds = 7 x 2 x 5 = 70 cells (vs 30 in 848a/848)

THE TWO CHANNELS UNDER TEST (channels 3+4 explicitly excluded, see SCOPE), laddered by BASE:
  1. 5-HT rigidity (+ calibrated dACC goal-readout)  serotonin.tonic_5ht_baseline path
  2. phasic gain                                     phasic_burst_temp_delta
CALIBRATION (unchanged from 848a; NOT a design delta here):
  dacc_goal_readout_weight=0.5, dacc_goal_readout_normalize=True
  (use_mech_consume=True, cfg.goal.z_goal_enabled=True, candidate_summary_source="e2_world_forward")

DV: E3 precision readout -- e3.current_precision = 1/(running_variance + 1e-6), recorded as
log10 precision, via the CANONICAL update_residue() / update_z_goal() path -- identical to 848a.

=== DV-SYMMETRY INVARIANCE DECLARATION (mandatory, per arm) ===
  channel 1 (5-HT / calibrated dACC goal-readout): reaches dACC's score_bias via
    candidate_goal_proximity, non-zero-weighted (0.5) AND per-candidate-set-normalised. This
    is NOT a broadcast constant and NOT a monotone reparameterisation applied after selection --
    the per-candidate-set normalisation IS the per-candidate signal, not a global rescale -- so
    the manipulation is NOT invariant under the log10-precision DV's symmetry group (neither the
    additive-constant/argmax symmetry nor the monotone-rescale/rank symmetry): a uniform shift
    of every candidate's proximity would leave the argmin unchanged, but the normalisation makes
    the channel's contribution genuinely per-candidate. So a measured monotone trend here is a
    genuine measurement, not an arithmetic identity. Channel 1 ALSO still carries its pre-existing
    serotonin-mediated pathway (unchanged from 848a), so a positive result cannot by itself
    attribute causation to the goal-readout sub-pathway vs. the serotonin sub-pathway -- see the
    pre-registered criterion note.
  channel 2 (phasic gain): a genuine E3 SOFTMAX TEMPERATURE -- a monotone reparameterisation,
    hence provably ARGMAX-INVARIANT under DETERMINISTIC (committed) selection, which this
    experiment runs under (~97% commitment on this config family). Its contribution to any
    observed rho here is architecturally expected to remain NULL, unchanged from 848a -- the
    finer ladder does not alter this. It is NOT invariant under an uncommitted/stochastic regime
    (TRACK B, not built here). The joint channel-1/2 ladder means a NULL channel-2 contribution
    is absorbed into the same criterion; the criterion measures the JOINT channel-1/2 response,
    dominated by channel 1 for the reason above.
  channels 3/4: excluded entirely (SCOPE) -- never varied, so no DV-symmetry claim here.

=== PRE-REGISTERED CRITERION (constant below; nothing derived from the run) ===
PRIMARY (load-bearing) -- UNCHANGED from 848/848a except the ladder is now 7-level:
  C_PRECISION_MONOTONICITY_FINER
    Per (content, seed) unit, over the L0<L1<...<L6 ladder of channels 1+2:
      |Spearman rho of log10-precision vs level| >= 0.60   [UNSIGNED -- channels 1 and 2 touch
        precision with opposite-signed intuitions, and channel 1 carries two sub-pathways with
        no shared sign convention pre-registered between them; same UNSIGNED convention as 802's
        C2 and 848/848a's C_PRECISION_MONOTONICITY]
      satisfied in >= 7 of the 10 (content x seed) units

  PASS  = C_PRECISION_MONOTONICITY_FINER                 -> evidence_direction supports
  FAIL, all 10 units |rho| <= 0.10 (near-zero band)       -> evidence_direction non_contributory
    (the calibrated channel fails to produce a detectable monotonicity signal even on a finer
     ladder -- an informative negative about this specific pathway; distinct from 848a's coarse
     +0.5-pile mixed result. At n=7 a true null scatters near 0, so this band is now cleanly
     separable from a weak-but-real trend, which is exactly what the 3-point ladder could not do.)
  FAIL, otherwise (some real but sub-threshold trend)     -> evidence_direction mixed

SECONDARY (RECORDED PER UNIT, NON-LOAD-BEARING -- resolution-independent characterisation):
  - Mann-Kendall two-sided p (normal approx w/ continuity correction; ties negligible since the
    log10-precision values are distinct deterministic floats) -- significance of monotone trend.
  - Theil-Sen slope (median pairwise slope, log10-precision per unit-of-level) -- effect
    magnitude. Expected TINY (~1e-3) per the 848a magnitude analysis above; recorded so a
    marginal PASS/FAIL can be read as weak-real vs null regardless of the primary threshold.
  - Kendall tau_b.
  These satisfy the task-brief's "and/or a magnitude-based statistic" as CONTINUOUS,
  resolution-independent readouts alongside the primary rank criterion -- NOT as a gate (an
  absolute magnitude floor would reject the genuinely-tiny real effect; see the magnitude note).

RESOLUTION DIAGNOSTIC (manifest-level): n_distinct_rho_values across the 10 units and the
achievable-rho-grid note -- direct evidence the 3-point ceiling (only 5 achievable rho values)
is broken by the 7-point ladder.

This experiment does NOT isolate the calibrated dACC goal-readout channel's marginal
contribution from channel 1's pre-existing serotonin-mediated contribution (that needs a further
decoupled design -- ladder channel 1's dACC-consume flags alone while holding serotonin fixed --
explicitly OUT OF SCOPE here, noted as candidate follow-on in the queue entry).

=== NON-DEGENERACY (else substrate_not_ready_requeue, NOT a verdict; unchanged from 848a) ===
Per-cell gates via experiments/_lib/precondition_gate.py:
  P1 channel_state_delta_vs_L0 > 0.05  applies_to PERTURBED CELLS ONLY (channels 1+2 moved vs the
       same-content L0 cell). Verifies the settings TOOK EFFECT. The finest perturbed level here
       (1/6) yields a channel delta ~0.167 > 0.05, so every perturbed cell clears P1 by design.
  P2 precision_cross_seed_sd > 1e-6    all cells. A precision readout permanently floor-pinned
       across DIFFERENT SEEDS would be a substrate-readiness failure prior to the monotonicity
       question. (848a's cross-seed SD was ~0.18, far above the floor.)
  P3 n_salience_ticks >= 150           all cells. The coordinator must have actually ticked enough
       for a genuine precision trajectory to accumulate.

Plus a LADDER-VARIATION degeneracy guard (new, evidence-run net per the multi-point-sweep
code-review rule): if log10-precision is bit-identical across ALL 7 levels in EVERY unit (a full
clamp / saturation fingerprint absorbing the manipulation), the run self-stamps non_degenerate=
False with a degeneracy_reason. This does NOT fire on the expected tiny-but-nonzero regime (848a
showed most cells vary by ~1e-3 across the ladder) -- only on a genuine clamp.

BUILD-TIME GUARD (unchanged from 848a): `_build()` asserts the constructed REEConfig carries
dacc_goal_readout_weight == 0.5 and dacc_goal_readout_normalize is True, and that the constructed
DACCtoE3Adapter's own config carries the weight -- raising immediately (ERROR) on 848's exact
silent-zero failure mode.

=== SCOPE (unchanged from 848a) ===
Channels 3 and 4 are DELIBERATELY held fixed at L0 in every cell and are not tested here.
TRACK B (channel 2 under uncommitted selection) and a channel-1-sub-pathway-isolating follow-up
are NOT built in this experiment.

=== NO GRADIENT TRAINING ===
Nothing is trained; no head reads a latent under a loss (DACCtoE3Adapter has no nn.Parameter).
The agent is driven in eval() exactly as the cells are constructed. A WARMUP phase precedes
measurement, unchanged from 848a.

=== SAMPLE-SIZE INTEGRITY (unchanged from 848a) ===
`current_mode`/`current_precision` are agent STATE, held between coordinator ticks; a per-env-step
read is a time-fraction/EMA-continuation, not pseudo-replication. The number of genuine
coordinator ticks is counted (`n_salience_ticks`) and gated (P3) rather than inferred from the
step count. UNIT OF ANALYSIS for the criterion is the SEED (n=5) x CONTENT (n=2) = 10 units.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.baselines import exq802_arc005_control_plane as BASE  # noqa: E402

# ------------------------------------------------------------------ #
# Identity                                                            #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_848b_arc005_precision_only_finer_ladder"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["ARC-005"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
BACKLOG_ID = None
SUPERSEDES_NOTE = (
    "Measurement / test-design lettered successor of V3-EXQ-848a (same scientific question -- "
    "does the control plane route precision, decoupled from occupancy, via the calibrated dACC "
    "goal-readout channel -- same substrate, same calibration, SAME pre-registered acceptance "
    "rule). 848a's manifest confirmed the channel is genuinely engaged (25,289 dACC bias calls) "
    "and is NOT invalidated by this run; it is superseded only because its 3-level ladder "
    "capped Spearman rho at a discrete grid ({-1,-0.5,0,+0.5,+1}) that could not distinguish a "
    "weak-but-real monotone trend (its 6/10 units at exactly +0.5, 7/10 positive-signed) from "
    "resolution noise. This run re-measures the SAME criterion (|rho| >= 0.60 in >=7/10 units, "
    "UNSIGNED) on a 7-level ladder where 0.60 is both meaningful (H0 SD ~0.408) and achievable "
    "by a consistent weak trend, and records Mann-Kendall significance + Theil-Sen slope as "
    "resolution-independent secondary readouts. Governance should read 848a and 848b together: "
    "848b's finer-ladder result is the resolution-corrected measurement of the same effect."
)

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = [0, 1, 2, 3, 4]
CONTENTS = ["A", "B"]

# --- THE ONLY DESIGN DELTA vs 848a: a 7-level ladder in place of the 3-level [0,0.5,1.0]. ---
# Evenly spaced i/6 for i in 0..6 so L0/L3/L6 coincide with 848a's L0/L1/L2 ladder settings
# (BASE.channel_settings interpolates linearly in [0,1]), keeping the endpoints comparable.
N_LEVELS = 7
LEVELS: List[float] = [round(i / (N_LEVELS - 1), 6) for i in range(N_LEVELS)]  # 0.0 .. 1.0

WARMUP_TICKS = BASE.WARMUP_TICKS                # 200
MEASURE_TICKS = BASE.MEASURE_TICKS              # 1800
TOTAL_TICKS = BASE.TOTAL_TICKS                  # 2000

DRY_WARMUP_TICKS = 10
DRY_MEASURE_TICKS = 40
DRY_SEEDS = [0, 1]

# --- calibration (unchanged from 848a; NOT a design delta here) ---
DACC_GOAL_READOUT_WEIGHT = 0.5      # V3-EXQ-637 phase2_on precedent value
DACC_GOAL_READOUT_NORMALIZE = True  # ree-v3 commit 4b79d18d44 (IGW-20260801-199)

# --- precision monotonicity criterion (threshold + count UNCHANGED from 848/848a) ---
C_RHO_ABS_FLOOR = 0.60            # UNSIGNED; now measured over the 7-level ladder
C_MIN_UNITS = 7                   # of 10 (content x seed)
C_NULL_BAND = 0.10                # |rho| <= this in ALL 10 units -> non_contributory, not mixed

# --- secondary diagnostic thresholds (NON-load-bearing; recorded, do not gate PASS/FAIL) ---
MK_P_SIGNIF = 0.10               # Mann-Kendall two-sided p below this = a significant monotone
                                 # trend at that unit (reported per unit; not part of the gate)

# --- non-degeneracy floors (unchanged from 848a) ---
P1_CHANNEL_DELTA_FLOOR = 0.05
P2_PRECISION_SD_FLOOR = 1e-6
P3_SALIENCE_TICK_FLOOR = 150.0
DRY_P3_SALIENCE_TICK_FLOOR = 5.0  # smoke only -- criteria are not scored on a smoke

# Ladder-variation degeneracy guard: a unit "varies" if its 7 log10-precision values are not
# all identical to within this epsilon. Only a FULL clamp (no unit varies) trips non_degenerate.
LADDER_VARIATION_EPS = 1e-9


# ------------------------------------------------------------------ #
# Statistics helpers                                                  #
# ------------------------------------------------------------------ #
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


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation. 0.0 when undefined (n < 3 or a flat side)."""
    n = len(x)
    if n < 3:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return float(num / (dx * dy))


def _kendall_S(y: List[float]) -> int:
    """Mann-Kendall S statistic for y ordered by ascending level (x strictly increasing).

    S = sum over i<j of sign(y_j - y_i). Positive => increasing trend.
    """
    n = len(y)
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = y[j] - y[i]
            if d > 0:
                s += 1
            elif d < 0:
                s -= 1
    return s


def _kendall_tau_b(x: List[float], y: List[float]) -> float:
    """Kendall tau-b (tie-corrected). x is the strictly-increasing level, so x-ties are absent;
    y-ties are corrected. 0.0 when undefined."""
    n = len(x)
    if n < 3:
        return 0.0
    conc = disc = 0
    ty = 0  # y-tie pairs
    for i in range(n):
        for j in range(i + 1, n):
            dy = y[j] - y[i]
            if dy > 0:
                conc += 1
            elif dy < 0:
                disc += 1
            else:
                ty += 1
    n0 = n * (n - 1) / 2.0
    denom = math.sqrt((n0 - 0.0) * (n0 - ty))  # x has no ties -> (n0 - tx) = n0
    if denom <= 0.0:
        return 0.0
    return float((conc - disc) / denom)


def _erf(z: float) -> float:
    return math.erf(z)


def _mann_kendall_p(y: List[float]) -> Tuple[int, float, float]:
    """Two-sided Mann-Kendall p-value (normal approximation, continuity-corrected).

    Returns (S, z, p_two_sided). Ties are negligible here (distinct deterministic floats), so
    no tie-variance correction is applied; the p is a SECONDARY, non-load-bearing readout and is
    labelled approximate. n=7 is small, so this is an approximation, not an exact test.
    """
    n = len(y)
    s = _kendall_S(y)
    if n < 3:
        return s, 0.0, 1.0
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s <= 0.0:
        return s, 0.0, 1.0
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p_two = 2.0 * (1.0 - 0.5 * (1.0 + _erf(abs(z) / math.sqrt(2.0))))
    p_two = max(0.0, min(1.0, p_two))
    return s, float(z), float(p_two)


def _theil_sen_slope(x: List[float], y: List[float]) -> float:
    """Median of pairwise slopes (y_j - y_i)/(x_j - x_i). Robust magnitude of the trend."""
    slopes: List[float] = []
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0.0:
                slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return 0.0
    return float(statistics.median(slopes))


# ------------------------------------------------------------------ #
# Cell identity + channel settings                                   #
# ------------------------------------------------------------------ #
def _arm_id(level: float, content: str) -> str:
    """L<idx>_<content> where idx is the position in the 7-level LEVELS ladder.

    BASE.arm_id can't be reused -- its index is against BASE.CHANNEL_LEVELS (3 levels).
    """
    idx = LEVELS.index(round(float(level), 6))
    return f"L{idx}_{content}"


def _precision_channel_settings(level: float) -> Dict[str, Any]:
    """Channels 1+2 laddered by `level`; channels 3+4 held at L0 (0.0) in EVERY cell.

    Unchanged from 848a -- the decoupling design and the per-channel maps are not what this
    successor redesigns; only the SET of levels sampled changes.
    """
    laddered = BASE.channel_settings(level)
    fixed = BASE.channel_settings(0.0)
    return {
        "serotonin_seeding_gain": laddered["serotonin_seeding_gain"],
        "serotonin_wanting_floor": laddered["serotonin_wanting_floor"],
        "phasic_burst_temp_delta": laddered["phasic_burst_temp_delta"],
        # channels 3+4 -- FIXED at L0 regardless of `level`. SCOPE (see docstring).
        "salience_external_task_bias": fixed["salience_external_task_bias"],
        "pcc_stability_baseline": fixed["pcc_stability_baseline"],
    }


def _agent_kwargs(level: float) -> Dict[str, Any]:
    """REEConfig.from_dims kwargs for a channel level (obs dims added by caller).

    Identical to 848a's _agent_kwargs() (including the two calibration-fix flags) -- only the
    `level` values passed in change.
    """
    ch = _precision_channel_settings(level)
    return dict(
        alpha_world=0.9,
        # --- channel 1: 5-HT (live at every level) ---
        tonic_5ht_enabled=True,
        # --- channel 2: phasic surprise burst (live at every level) ---
        use_phasic_burst=True,
        phasic_burst_temp_delta=ch["phasic_burst_temp_delta"],
        phasic_burst_signal_source="instantaneous_pe",
        phasic_burst_baseline_continuity="carry",
        # --- channels 3+4: salience coordinator + PCC-analog, FIXED at L0 ---
        use_salience_coordinator=True,
        use_dacc=True,
        dacc_weight=0.5,
        dacc_foraging_weight=0.5,
        use_aic_analog=True,
        salience_external_task_bias=ch["salience_external_task_bias"],
        use_pcc_analog=True,
        pcc_stability_baseline=ch["pcc_stability_baseline"],
        salience_use_stability_temperature=True,
        salience_temperature_mu_alpha=1.0,
        # --- substrate operating config, identical in every cell ---
        use_harm_stream=True,
        use_affective_harm_stream=True,
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=4.0,
        # --- channel-1 dACC-consume prerequisites (unchanged from 848a) ---
        use_mech_consume=True,
        z_goal_enabled=True,
        candidate_summary_source="e2_world_forward",
        # --- calibration fix (unchanged from 848a) ---
        dacc_goal_readout_weight=DACC_GOAL_READOUT_WEIGHT,
        dacc_goal_readout_normalize=DACC_GOAL_READOUT_NORMALIZE,
    )


def _build(seed: int, level: float, content: str):
    """Construct env + agent for one cell (env/content via the canonical baseline module,
    calibration-fixed agent config above). BUILD-TIME GUARD unchanged from 848a."""
    env_kw = BASE.content_env_kwargs(content, seed)
    env = CausalGridWorld(**env_kw)
    _obs, obs_dict = env.reset()

    kw: Dict[str, Any] = dict(_agent_kwargs(level))
    kw.update(
        body_obs_dim=obs_dict["body_state"].shape[-1],
        world_obs_dim=obs_dict["world_state"].shape[-1],
        action_dim=env.action_dim,
    )
    cfg = REEConfig.from_dims(**kw)
    ch = _precision_channel_settings(level)
    cfg.serotonin.gain_min = cfg.serotonin.gain_max = float(ch["serotonin_seeding_gain"])
    cfg.serotonin.floor_min = cfg.serotonin.floor_max = float(ch["serotonin_wanting_floor"])

    assert cfg.dacc_goal_readout_weight == DACC_GOAL_READOUT_WEIGHT, (
        f"BUILD-TIME GUARD FAILED: cfg.dacc_goal_readout_weight="
        f"{cfg.dacc_goal_readout_weight!r}, expected {DACC_GOAL_READOUT_WEIGHT!r} -- "
        f"this is 848's silent-zero failure mode; the calibration fix did not thread through "
        f"REEConfig.from_dims."
    )
    assert cfg.dacc_goal_readout_normalize is DACC_GOAL_READOUT_NORMALIZE, (
        f"BUILD-TIME GUARD FAILED: cfg.dacc_goal_readout_normalize="
        f"{cfg.dacc_goal_readout_normalize!r}, expected {DACC_GOAL_READOUT_NORMALIZE!r}."
    )

    agent = REEAgent(cfg)
    agent.eval()

    assert agent.dacc_adapter is not None, (
        "BUILD-TIME GUARD FAILED: agent.dacc_adapter is None -- use_dacc must be True for the "
        "goal-readout term to have any bias to attach to."
    )
    assert agent.dacc_adapter.config.dacc_goal_readout_weight == DACC_GOAL_READOUT_WEIGHT, (
        "BUILD-TIME GUARD FAILED: the constructed DACCtoE3Adapter's own config does not carry "
        "dacc_goal_readout_weight -- REEAgent's DACCConfig construction site did not propagate "
        "cfg.dacc_goal_readout_weight."
    )

    return agent, env, obs_dict


def _run_cell(
    seed: int, level: float, content: str, n_warmup: int, n_measure: int,
    zg_acc: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """Drive one (seed, level, content) cell and return its readouts.

    Uses the CANONICAL update_z_goal() / update_residue() calls -- verbatim from 848a's run-cell
    logic (a landed, validated driver that produced a real 187-min manifest 2026-08-02).
    """
    arm = _arm_id(level, content)
    agent, env, obs_dict = _build(seed, level, content)
    n_ticks = n_warmup + n_measure

    print(f"Seed {seed} Condition {arm}", flush=True)

    log_precisions: List[float] = []
    seeding_gains: List[float] = []
    tonic_5ht: List[float] = []
    burst_levels: List[float] = []
    n_salience_ticks = 0
    n_measured_steps = 0
    n_dacc_bias_calls_start = int(agent.dacc_adapter._n_bias_calls)
    prev_sal_tick: Any = None
    last_reward = 0.0

    for tick in range(n_ticks):
        measuring = tick >= n_warmup
        with torch.no_grad():
            latent = agent.sense(
                obs_dict["body_state"].unsqueeze(0),
                obs_dict["world_state"].unsqueeze(0),
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            ticks_d = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks_d["e1_tick"]
                else torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks_d)
            # Canonical z_goal writer, BEFORE select_action (StepHarness order).
            drive_level = REEAgent.compute_drive_level(obs_dict["body_state"].unsqueeze(0))
            agent.update_z_goal(
                benefit_exposure=max(0.0, float(last_reward)), drive_level=drive_level
            )
            action = agent.select_action(candidates, ticks_d, 1.0)
        agent._step_count += 1

        cur_sal_tick = getattr(agent, "_salience_last_tick", None)
        if cur_sal_tick is not None and cur_sal_tick is not prev_sal_tick:
            n_salience_ticks += 1
        prev_sal_tick = cur_sal_tick

        act_idx = (
            int(action.argmax().item()) if isinstance(action, torch.Tensor) else int(action)
        )
        _obs, reward, done, _info, obs_dict = env.step(act_idx % env.action_dim)
        last_reward = float(reward)

        # Keep the 5-HT channel LIVE (as 848a/848/802 did).
        agent.serotonin_step(max(0.0, float(reward)))

        # Canonical precision-update path (unchanged from 848a).
        with torch.no_grad():
            agent.sense(
                obs_dict["body_state"].unsqueeze(0),
                obs_dict["world_state"].unsqueeze(0),
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            agent.update_residue(
                harm_signal=float(reward), world_delta=None,
                hypothesis_tag=False, owned=True,
            )

        if measuring:
            n_measured_steps += 1
            log_precisions.append(math.log10(max(agent.e3.current_precision, 1e-12)))
            seeding_gains.append(float(agent.serotonin.current_seeding_gain()))
            tonic_5ht.append(float(agent.serotonin.tonic_5ht))
            burst_levels.append(
                float(agent.phasic_burst.burst_level)
                if agent.phasic_burst is not None else 0.0
            )

        if done:
            _obs, obs_dict = env.reset()

        if (tick + 1) % 250 == 0 or tick == n_ticks - 1:
            print(
                f"  [train] arc005finerladder {arm} seed={seed} ep {tick + 1}/{n_ticks} "
                f"sal_ticks={n_salience_ticks} steps={n_measured_steps}",
                flush=True,
            )

    zg_acc.observe(agent)
    n_dacc_bias_calls = int(agent.dacc_adapter._n_bias_calls) - n_dacc_bias_calls_start

    row: Dict[str, Any] = {
        "arm_id": arm,
        "channel_level": float(level),
        "content": content,
        "seed": int(seed),
        "n_ticks": n_ticks,
        "n_warmup_ticks": n_warmup,
        "n_measured_steps": n_measured_steps,
        "n_salience_ticks": n_salience_ticks,
        "n_dacc_bias_calls": n_dacc_bias_calls,
        "log10_precision_mean": round(float(np.mean(log_precisions)), 8)
        if log_precisions else 0.0,
        "log10_precision_sd": round(float(np.std(log_precisions)), 8)
        if len(log_precisions) > 1 else 0.0,
        "realised_channel_state": {
            "seeding_gain_mean": round(float(np.mean(seeding_gains)), 8)
            if seeding_gains else 0.0,
            "tonic_5ht_mean": round(float(np.mean(tonic_5ht)), 8) if tonic_5ht else 0.0,
            "burst_level_mean": round(float(np.mean(burst_levels)), 8)
            if burst_levels else 0.0,
            "phasic_temp_delta_cfg": float(
                agent.phasic_burst.config.temp_delta
                if agent.phasic_burst is not None else 0.0
            ),
            "salience_external_task_bias_cfg_fixed_at_L0": float(
                agent.salience.config.external_task_bias
                if agent.salience is not None else 0.0
            ),
            "dacc_goal_readout_weight_cfg": float(
                agent.dacc_adapter.config.dacc_goal_readout_weight
            ),
            "dacc_goal_readout_normalize_cfg": bool(
                agent.dacc_adapter.config.dacc_goal_readout_normalize
            ),
        },
        "goal_state_active_at_end": bool(
            agent.goal_state is not None and agent.goal_state.is_active()
        ),
    }
    return row


# ------------------------------------------------------------------ #
# Precondition specs (regime-conditioned, unchanged from 848a)        #
# ------------------------------------------------------------------ #
def _specs(salience_tick_floor: float) -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="channel_state_delta_vs_L0",
            description=(
                "normalised L1 distance of the cell's REALISED channel-1/2 state from the "
                "same-content L0 cell's"
            ),
            control="same-content level-0 cell",
            threshold=P1_CHANNEL_DELTA_FLOOR,
            applies_to=lambda ctx: float(ctx["channel_level"]) != LEVELS[0],
            applies_note=(
                "perturbed cells only -- an L0 cell compared with itself is 0 by construction, "
                "not by substrate failure"
            ),
        ),
        PreconditionSpec(
            name="precision_cross_seed_sd",
            description="cross-seed SD of the cell's log10-precision mean, at a fixed level",
            control="five independently seeded cells of the same level+content",
            threshold=P2_PRECISION_SD_FLOOR,
        ),
        PreconditionSpec(
            name="n_salience_ticks",
            description="genuine SalienceCoordinator ticks",
            control="coordinator ticking on the live selection path",
            threshold=salience_tick_floor,
        ),
    ]


def _cell_contexts() -> List[Dict[str, Any]]:
    out = []
    for content in CONTENTS:
        for level in LEVELS:
            out.append(
                {
                    "id": _arm_id(level, content),
                    "channel_level": float(level),
                    "content": content,
                }
            )
    return out


def _channel_state_vector(row: Dict[str, Any]) -> np.ndarray:
    s = row["realised_channel_state"]
    return np.array(
        [
            (float(s["seeding_gain_mean"]) - 0.90) / 0.60,
            (abs(float(s["phasic_temp_delta_cfg"])) - 0.10) / 0.90,
        ],
        dtype=float,
    )


# ------------------------------------------------------------------ #
# Analysis                                                            #
# ------------------------------------------------------------------ #
def _analyse(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    by: Dict[Tuple[float, str, int], Dict[str, Any]] = {
        (r["channel_level"], r["content"], r["seed"]): r for r in rows
    }

    units: List[Dict[str, Any]] = []
    n_units_with_ladder_variation = 0
    for c in CONTENTS:
        for s in seeds:
            try:
                series = [by[(lv, c, s)] for lv in LEVELS]
            except KeyError:
                continue
            prec = [float(r["log10_precision_mean"]) for r in series]
            levels = list(LEVELS)
            rho = _spearman_rho(levels, prec)
            ok = abs(rho) >= C_RHO_ABS_FLOOR
            # secondary, resolution-independent readouts
            mk_s, mk_z, mk_p = _mann_kendall_p(prec)
            tau_b = _kendall_tau_b(levels, prec)
            slope = _theil_sen_slope(levels, prec)
            prec_range = float(max(prec) - min(prec))
            varies = prec_range > LADDER_VARIATION_EPS
            if varies:
                n_units_with_ladder_variation += 1
            units.append(
                {
                    "content": c,
                    "seed": s,
                    "rho_log10_precision": round(rho, 4),
                    "satisfied": bool(ok),
                    # secondary diagnostics (non-load-bearing)
                    "mann_kendall_S": int(mk_s),
                    "mann_kendall_z": round(mk_z, 4),
                    "mann_kendall_p_two_sided_approx": round(mk_p, 5),
                    "mann_kendall_significant": bool(mk_p < MK_P_SIGNIF),
                    "kendall_tau_b": round(tau_b, 4),
                    "theil_sen_slope_log10prec_per_level": round(slope, 8),
                    "log10_precision_range_across_ladder": round(prec_range, 8),
                    "ladder_varies": bool(varies),
                }
            )
    n_satisfied = sum(1 for u in units if u["satisfied"])
    criterion_pass = n_satisfied >= C_MIN_UNITS
    all_near_zero = bool(units) and all(
        abs(u["rho_log10_precision"]) <= C_NULL_BAND for u in units
    )

    # resolution diagnostic -- direct evidence the 3-point rho grid ceiling is broken.
    distinct_rho = sorted(set(u["rho_log10_precision"] for u in units))
    n_signif_mk = sum(1 for u in units if u["mann_kendall_significant"])

    return {
        "units": units,
        "n_satisfied": n_satisfied,
        "n_units": len(units),
        "criterion_pass": criterion_pass,
        "all_near_zero_null": all_near_zero,
        # secondary aggregate readouts (non-load-bearing)
        "n_units_mann_kendall_significant": n_signif_mk,
        "n_units_with_ladder_variation": n_units_with_ladder_variation,
        "resolution_diagnostic": {
            "n_ladder_levels": N_LEVELS,
            "ladder_levels": list(LEVELS),
            "n_distinct_rho_values_observed": len(distinct_rho),
            "distinct_rho_values_observed": distinct_rho,
            "note": (
                "On a 3-level ladder Spearman rho can take only {-1.0,-0.5,0.0,+0.5,+1.0} (5 "
                "values), so a weak-but-real monotone trend piles at exactly +0.5 (848a: 6/10 "
                "units). A 7-level ladder resolves rho on a much finer grid, separating a "
                "consistent weak trend (rho ~0.7-0.9) from resolution noise (rho scattered near "
                "0). A large n_distinct_rho_values_observed here is direct evidence the ceiling "
                "is broken."
            ),
        },
    }


# ------------------------------------------------------------------ #
# Driver                                                              #
# ------------------------------------------------------------------ #
def run_experiment(dry_run: bool) -> Tuple[Dict[str, Any], ZGoalStreamAccumulator]:
    seeds = DRY_SEEDS if dry_run else SEEDS
    n_warmup = DRY_WARMUP_TICKS if dry_run else WARMUP_TICKS
    n_measure = DRY_MEASURE_TICKS if dry_run else MEASURE_TICKS
    tick_floor = DRY_P3_SALIENCE_TICK_FLOOR if dry_run else P3_SALIENCE_TICK_FLOOR

    specs = _specs(tick_floor)
    contexts = _cell_contexts()
    assert_no_structurally_unsatisfiable_gate(specs, contexts)

    zg_acc = ZGoalStreamAccumulator()
    rows: List[Dict[str, Any]] = []
    total_dacc_bias_calls = 0
    for content in CONTENTS:
        for level in LEVELS:
            for seed in seeds:
                slice_ = {
                    "lineage": "exq848b_arc005_precision_only_finer_ladder",
                    "arm_id": _arm_id(level, content),
                    "channel_level": float(level),
                    "content": content,
                    "env_kwargs": BASE.content_env_kwargs(content, seed),
                    "agent_kwargs": _agent_kwargs(level),
                    "schedule": {
                        "warmup_ticks": WARMUP_TICKS, "measure_ticks": MEASURE_TICKS,
                        "total_ticks": TOTAL_TICKS,
                    },
                }
                with arm_cell(
                    seed,
                    config_slice=slice_,
                    script_path=Path(__file__),
                    config_slice_declared=True,
                    include_driver_script_in_hash=True,
                ) as cell:
                    row = _run_cell(seed, level, content, n_warmup, n_measure, zg_acc)
                    cell.stamp(row)
                rows.append(row)
                total_dacc_bias_calls += int(row["n_dacc_bias_calls"])
                print(
                    f"verdict: {'PASS' if row['n_measured_steps'] > 0 else 'FAIL'}",
                    flush=True,
                )

    # Whole-run engagement check: the calibrated dACC goal-readout channel must be invoked at
    # least once. Unchanged from 848a -- decisive-readout engagement failure, not a result.
    if total_dacc_bias_calls == 0:
        raise RuntimeError(
            "ENGAGEMENT CHECK FAILED: agent.dacc_adapter._n_bias_calls never incremented across "
            "any cell -- the dACC adapter this experiment exists to test was never invoked. Do "
            "not interpret any rho computed alongside this condition."
        )

    by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault(r["arm_id"], []).append(r)

    l0_state: Dict[str, np.ndarray] = {}
    for content in CONTENTS:
        cell0 = _arm_id(LEVELS[0], content)
        vecs = [_channel_state_vector(r) for r in by_cell.get(cell0, [])]
        l0_state[content] = (
            np.mean(np.stack(vecs), axis=0) if vecs else np.zeros(2, dtype=float)
        )

    cell_gates = []
    for ctx in contexts:
        cell_rows = by_cell.get(ctx["id"], [])
        if not cell_rows:
            continue
        prec_means = [float(r["log10_precision_mean"]) for r in cell_rows]
        prec_sd = float(statistics.stdev(prec_means)) if len(prec_means) > 1 else 0.0
        ticks_worst = min(float(r["n_salience_ticks"]) for r in cell_rows)
        vec = np.mean(np.stack([_channel_state_vector(r) for r in cell_rows]), axis=0)
        chan_delta = float(np.mean(np.abs(vec - l0_state[ctx["content"]])))
        cell_gates.append(
            evaluate_arm_gate(
                ctx["id"], ctx, specs,
                measured={
                    "channel_state_delta_vs_L0": chan_delta,
                    "precision_cross_seed_sd": prec_sd,
                    "n_salience_ticks": ticks_worst,
                },
            )
        )
    gate = aggregate_arm_gates(cell_gates)
    green = set(gate["green_arms"])
    scorable = set(c["id"] for c in contexts).issubset(green)

    analysis = _analyse(rows, seeds)

    # Ladder-variation degeneracy guard (evidence-run net): a FULL clamp -- log10-precision
    # bit-identical across all 7 levels in EVERY unit -- means the manipulation was fully absorbed
    # (a saturation fingerprint), not a null result. Does not fire on the expected tiny-but-
    # nonzero regime (some units vary).
    full_clamp = bool(analysis["units"]) and analysis["n_units_with_ladder_variation"] == 0

    if not scorable:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            f"cell(s) failed non-degeneracy gate (green: {sorted(green)} of "
            f"{sorted(c['id'] for c in contexts)})"
        )
        overall_pass = False
    elif full_clamp:
        outcome = "FAIL"
        label = "precision_dv_clamped_across_ladder"
        evidence_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            "log10-precision bit-identical across all 7 ladder levels in EVERY unit -- the "
            "channel-1/2 manipulation was fully absorbed (saturation/clamp fingerprint), so no "
            "monotonicity is measurable. Not a null result about ARC-005."
        )
        overall_pass = False
    else:
        overall_pass = bool(analysis["criterion_pass"])
        outcome = "PASS" if overall_pass else "FAIL"
        if overall_pass:
            label = "control_plane_routes_precision_decoupled_finer_ladder"
            evidence_direction = "supports"
        elif analysis["all_near_zero_null"]:
            label = "precision_channel_authority_null_finer_ladder"
            evidence_direction = "non_contributory"
        else:
            label = "precision_channel_authority_weak_finer_ladder"
            evidence_direction = "mixed"
        non_degenerate = True
        degeneracy_reason = ""

    criteria = [
        {"name": "C_precision_monotonicity_finer", "load_bearing": True,
         "passed": bool(analysis["criterion_pass"])},
    ]
    criteria_nd = {"C_precision_monotonicity_finer": scorable and not full_clamp}

    manifest: Dict[str, Any] = {
        "run_id": None,  # filled by __main__
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "backlog_id": BACKLOG_ID,
        "supersedes_note": SUPERSEDES_NOTE,
        "outcome": outcome,
        "overall_pass": overall_pass,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "criteria": criteria,
        "analysis": analysis,
        "per_arm_gate": gate["per_arm_gate"],
        "diagnostics": {
            "n_distinct_argmax_modes_across_design": None,  # not tested here -- see SCOPE
            "total_dacc_bias_calls": total_dacc_bias_calls,
            "n_ladder_levels": N_LEVELS,
            "ladder_levels": list(LEVELS),
        },
        "scorable": scorable,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": gate["adjudication_preconditions"],
            "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
            "criteria_non_degenerate": criteria_nd,
        },
        "pre_registered_thresholds": {
            "c_rho_abs_floor": C_RHO_ABS_FLOOR,
            "c_min_units": C_MIN_UNITS,
            "c_null_band": C_NULL_BAND,
            "n_ladder_levels": N_LEVELS,
            "ladder_levels": list(LEVELS),
            "mk_p_significance_threshold_secondary": MK_P_SIGNIF,
            "p1_channel_delta_floor": P1_CHANNEL_DELTA_FLOOR,
            "p2_precision_sd_floor": P2_PRECISION_SD_FLOOR,
            "p3_salience_tick_floor": tick_floor,
            "dacc_goal_readout_weight": DACC_GOAL_READOUT_WEIGHT,
            "dacc_goal_readout_normalize": DACC_GOAL_READOUT_NORMALIZE,
        },
        "custom_information": {
            "channel_ladder": {str(lv): _precision_channel_settings(lv) for lv in LEVELS},
            "content_sets": BASE.CONTENT_SETS,
            "arena": BASE.ARENA,
            "predecessor_run_id": "v3_exq_848a_arc005_precision_only_decoupled_ladder_calibrated",
            "predecessor_finding": (
                "848a confirmed the calibrated dACC goal-readout channel is genuinely engaged "
                "(25,289 dACC bias calls) but scored only 1/10 units against |rho|>=0.60 on a "
                "3-level ladder, with a per-unit rho distribution ({+0.5 x6, +1.0 x1, -0.5 x2, "
                "0.0 x1}, 7/10 positive-signed) that is the arithmetic signature of a 3-point "
                "Spearman resolution ceiling, not a clean null. This run re-measures the same "
                "criterion on a 7-level ladder that breaks that ceiling."
            ),
            "measurement_redesign_note": (
                "The ONLY change vs 848a is measurement RESOLUTION: LEVELS 3->7. Substrate, "
                "calibration, content, arena, schedule, seeds, DV and the primary acceptance "
                "rule (|rho|>=0.60 in >=7/10 units, UNSIGNED) are all UNCHANGED. Mann-Kendall "
                "significance + Theil-Sen slope + Kendall tau are recorded per unit as "
                "resolution-independent SECONDARY diagnostics -- NOT a magnitude-floor gate, "
                "because 848a's raw readouts show the real effect is ~1e-3 log10-units (far "
                "below any principled absolute floor) and swamped ~180x by the cross-seed SD, so "
                "the signal is in ORDERING CONSISTENCY, which the per-(content,seed) unit design "
                "and a finer-ladder rank statistic are the correct tools for."
            ),
            "dv_symmetry_declaration": (
                "channel 1 (5-HT / calibrated dACC goal-readout): genuinely weighted (0.5) and "
                "per-candidate-set normalised -- not a broadcast constant, not a monotone "
                "reparameterisation applied after selection -- so NOT invariant under the "
                "log10-precision DV's additive-constant/argmax or monotone-rescale/rank "
                "symmetries; a measured monotone trend here is a genuine measurement. Channel 1 "
                "also still carries its pre-existing serotonin pathway, so a positive result "
                "cannot be attributed to the goal-readout sub-pathway alone. channel 2 (phasic "
                "gain) is a genuine E3 softmax TEMPERATURE -- a monotone reparameterisation, "
                "hence provably argmax-invariant under the DETERMINISTIC (committed) selection "
                "this experiment runs under (~97% commitment on this config family); its "
                "contribution to rho is architecturally expected to stay null, unchanged by the "
                "finer ladder. Channels 3/4 held fixed at L0 -- no DV-symmetry claim (SCOPE)."
            ),
            "scope_note": (
                "Channels 3 (mode prior) and 4 (pcc_stability) are DELIBERATELY held fixed at "
                "their L0 values in every cell and excluded from this experiment's criterion a "
                "priori, unchanged from 848a -- see 802/GAP-B (V3-EXQ-846) for their occupancy "
                "authority. TRACK B (channel 2 under uncommitted selection) and a follow-up "
                "isolating the calibrated dACC goal-readout channel's marginal contribution from "
                "channel 1's pre-existing serotonin-mediated pathway are explicitly OUT OF SCOPE."
            ),
            "supersedes_848a": SUPERSEDES_NOTE,
            "expected_outcome": (
                "Not pre-determined. If the weak positive tendency 848a's coarse ladder hinted "
                "at (7/10 positive-signed) is a genuine consistent monotone trend, the 7-level "
                "ladder should resolve rho above 0.60 in several units (-> supports or mixed). "
                "If instead it was resolution noise piling at +0.5, the finer ladder scatters "
                "rho near 0 (-> non_contributory, now cleanly separable from a weak-real trend). "
                "Either way the result is informative in a way 848a's 3-point ladder could not "
                "be. Channel 2's contribution remains architecturally expected null."
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
    }
    return manifest, zg_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    from datetime import datetime

    manifest, zg_acc = run_experiment(dry_run=args.dry_run)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest["timestamp_utc"] = ts
    manifest["dry_run"] = bool(args.dry_run)

    full_config = {
        "arena": BASE.ARENA,
        "content_sets": BASE.CONTENT_SETS,
        "channel_levels": LEVELS,
        "channel_ladder": {str(lv): _precision_channel_settings(lv) for lv in LEVELS},
        "agent_kwargs_by_level": {str(lv): _agent_kwargs(lv) for lv in LEVELS},
        "schedule": {
            "warmup_ticks": DRY_WARMUP_TICKS if args.dry_run else WARMUP_TICKS,
            "measure_ticks": DRY_MEASURE_TICKS if args.dry_run else MEASURE_TICKS,
        },
        "pre_registered_thresholds": manifest["pre_registered_thresholds"],
    }
    seeds_used = DRY_SEEDS if args.dry_run else SEEDS

    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds_used,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg_acc.stats(),
    )

    out_path = write_flat_manifest(
        manifest, OUT_DIR, dry_run=args.dry_run, stamp=False
    )

    print(json.dumps({
        "run_id": manifest["run_id"],
        "outcome": manifest["outcome"],
        "evidence_direction": manifest["evidence_direction"],
        "non_degenerate": manifest["non_degenerate"],
        "n_satisfied": manifest["analysis"]["n_satisfied"],
        "n_units": manifest["analysis"]["n_units"],
        "n_units_mann_kendall_significant": manifest["analysis"]["n_units_mann_kendall_significant"],
        "n_distinct_rho_values": manifest["analysis"]["resolution_diagnostic"][
            "n_distinct_rho_values_observed"],
        "all_near_zero_null": manifest["analysis"]["all_near_zero_null"],
        "label": manifest["interpretation"]["label"],
        "total_dacc_bias_calls": manifest["diagnostics"]["total_dacc_bias_calls"],
        "manifest": str(out_path),
    }, indent=2), flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
