#!/opt/local/bin/python3
"""
V3-EXQ-1066 -- ARC-029 (EXP-1394 / "V3-EXQ-063b"): within-run committed vs
uncommitted HARM contrast under threshold-modulated mode alternation, on the
variance-tracking commit bar. PRECISION-INVARIANT ABLATION.

Supersedes: V3-EXQ-063a (non_contributory by confirmed autopsy 2026-09-14).
Claims: ARC-029 (only).
Proposal: EXP-1394 (experiment_proposals.v1.json, claim_id ARC-029).
Pre-registration derivation (every number below, with the rule that produced it):
  REE_assembly/evidence/planning/arc029_exq1066_prereg_derivation_20260919.md
Substrate the design rests on:
  REE_assembly/evidence/planning/arc029_variance_tracking_commit_bar_build_20260918.md
red-team: see the queue entry note for the verdict and model.

WHY V3-EXQ-063a WAS UNINFORMATIVE, AND WHAT IS DIFFERENT HERE
---------------------------------------------------------------------------
063a's driver wrote `agent.e3._running_variance` directly to ablate commitment.
`current_precision = 1/(running_variance + 1e-6)` is the SAME SCALAR, so forcing
one pinned the other: its harm gap is unattributable between "the commitment
gate did work" and "the agent's precision was crushed". Its occupancy was also
forced -- gate_active recorded n_committed_active_stable = 1155.5 against
n_uncommitted_active_stable = 0.0, so its own non-degeneracy check passed
VACUOUSLY and the within-agent two-mode contrast was never realised.

This driver NEVER assigns to `e3._running_variance`. It is a source-level
invariant, asserted at runtime by `_assert_no_running_variance_writes()` below
and gated as G1. The mode driver is THRESHOLD-SIDE on both legs:

  (1) the BASE bar is `use_variance_tracking_commit_threshold` (ree-v3
      6bac3753 / 9862ed61 / 3bb81276) -- a q-quantile of the DETRENDED residuals
      of the run's own commit-gate-variance window. This is what makes both
      regimes occupiable at a trained operating point at all: with the absolute
      0.40 bar, rv collapses ~5 orders during training and commitment becomes an
      absorbing state (measured `committed_step_fraction` 1.0000, ONE run of
      1200 ticks).
  (2) the ALTERNATION is MECH-108's BreathOscillator, which multiplies that base
      bar by `(1 - sweep_amplitude)` during the sweep phase (clock.py:20-31).
      ARC-029's P2 names exactly this lever.

Neither touches `running_variance` or any precision consumer.

THE ARMS
---------------------------------------------------------------------------
ONE agent is trained per seed, with the quantile bar ARMED and the breath
oscillator OFF, so the trained operating point is the one the bar will see.
Both arms then run on that SAME trained agent, so G1's precision comparison
isolates the manipulation rather than training noise (cell isolation is by
window-clear + per-episode reset + counterbalanced arm order -- `REEAgent` is
not deepcopy-able; see `_run_cell`):

  ARM_STATIC     quantile bar armed, breath_period = 0. No alternation driver.
  ARM_ALTERNATE  identical, plus the MECH-108 sweep at the seed's CALIBRATED
                 amplitude (see below). This is the arm C1 is read in.

`use_natural_commit_latch_hold` is NOT armed, deliberately. ARC-029's own P1
text and 063a section 8 both route P1 there; the 2026-09-18 build record sec 5.2
shows it is INAPPLICABLE and why -- it SUSTAINS commitment (built for the
V3-EXQ-460i fragmentation failure, occupancy too SHORT) and the measured failure
is the opposite, saturation at 100% committed. GFLAG-0354 carries this.

PRE-REGISTERED OPERATING POINT -- read off the build record's measured surface
---------------------------------------------------------------------------
  commit_threshold_quantile        0.50   maximises the min distance to P1's
                                          [0.15, 0.85] band over all nine
                                          measured (q, W) cells (0.33/0.32/0.30
                                          vs 0.12-0.19 at q=0.25 and 0.09-0.16
                                          at q=0.75), AND is the only q whose
                                          companion surfaces (estimator controls
                                          sec 3C, amplitude response sec 3D)
                                          were measured -- any other q makes the
                                          amplitude ladder an extrapolation.
  commit_threshold_quantile_window 200    W does NOT control run length
                                          (config.py:1265: ~3.2 -> ~3.4 across a
                                          40x range). It sets the WARMUP (in
                                          SELECT CALLS) and how much drift one
                                          linear term describes. Occupancy
                                          margin is flat across the q=0.50 row,
                                          so W is fixed at the only window where
                                          both companion surfaces exist.
  e3_steps_per_tick                3      measured mean committed-run length at
                                          q=0.50 is 10.41 / 6.10 / 3.36 / 2.60
                                          at ratio 1 / 3 / 10 (agent default) /
                                          20. The default clears P1's >= 3 floor
                                          by 12% on a SYNTHETIC stream before any
                                          seed variation. Rule: the largest
                                          (least-perturbing) MEASURED ratio with
                                          >= 2x margin -> 3 (6.10).
  precision_ema_alpha              0.05   the DEFAULT, held and pre-registered as
                                          held. Its measured cells are 0.5/0.05/
                                          0.001 -> 2.0/3.4/14.1; nothing sits
                                          between 3.4 and 14.1, so meeting the
                                          >= 2x rule through alpha would require
                                          INTERPOLATION. It is also the EMA
                                          constant on running_variance, hence on
                                          current_precision -- the quantity G1
                                          gates on. The cadence lever is not.

SWEEP GEOMETRY, derived from P1's own run-length criterion:
  breath_sweep_duration 18 env steps = 6 select calls = 2x P1's >= 3 floor, so a
    sweep window cannot itself be the 1-tick-blip structure V3-EXQ-460i's finding
    makes P1 reject.
  breath_period 54 env steps = 3 x the sweep, so the inter-sweep phase is 12
    select calls ~= 2x the measured mean committed-run length (6.10) and most
    committed runs complete inside it rather than being truncated by the sweep.

`sweep_amplitude` is NOT a fixed number, because ARC-029's P2 asks for a rule
and not a number: "Set sweep_amplitude from the run's own measured rv
distribution, not from a guess." Its worked values (a > 0.18) are measured stale
in BOTH directions (GFLAG-0354; on the quantile bar a=0.20 drives occupancy to
0.0030). The build record's own caveat is that the measured band a in
[0.02, 0.10] is "a property of the synthetic stream here and should be
re-measured on a real trained agent before amplitudes are pre-registered". So
what is pre-registered here is the SEARCH and its failure branch:

  CALIBRATION (per seed, on episodes DISJOINT from the measured ones):
    ladder a in {0.02, 0.05, 0.10}  -- the measured band and its interior
    choose the LARGEST a satisfying ALL of
      (i)   committed_step_fraction in [0.20, 0.80]  (interior to P1's band)
      (ii)  mean committed-run length >= 4.5          (1.5x P1's floor)
      (iii) |mean log10(current_precision) - ARM_STATIC's| <= 0.20 decades
    LARGEST because the sweep IS the manipulation and a stronger drive gives a
    cleaner two-mode contrast; bounded by (iii) because a stronger threshold
    perturbation also has more room to mediate into precision via behaviour.
    If no rung qualifies on >= 4 of 5 seeds the run is NON_CONTRIBUTORY, per
    ARC-029: "if occupancy or precision-invariance fails to gate cleanly, the
    run is non_contributory, not evidence."
  The gates are then RE-EVALUATED on the held-out measured episodes. Choosing a
  on calibration data and testing on disjoint data is what stops (i)-(iii) from
  being satisfied by construction.

GATES (EXP-1394; ALL THREE must hold before C1 is read as evidence)
---------------------------------------------------------------------------
  G0_occupancy            per seed, in ARM_ALTERNATE: committed_step_fraction in
                          [0.15, 0.85] AND mean committed-run length >= 3, on
                          >= 4/5 seeds. The full run-length HISTOGRAM is
                          reported, not just the mean -- EXP-1394 says so
                          explicitly, because a 1-tick-blip distribution with a
                          long tail can hit a mean of 3.
  G1_precision_invariance per seed, |d mean log10 P| <= 0.20 AND
                          |d SD log10 P| <= 0.20 between ARM_ALTERNATE and
                          ARM_STATIC, on >= 4/5 seeds; PLUS a power condition
                          (the across-seed SD of the paired mean delta is itself
                          <= 0.20, so "equivalence" cannot be bought with seed
                          noise); PLUS the SOURCE-level assert that the driver
                          never writes `e3._running_variance`.
                          THE BOUND IS DERIVED, NOT INVENTED: V3-EXQ-063a forced
                          rv to 0.50 (precision 2.0) against an operating point
                          measured at rv ~ 0.00542 (precision ~184.5), a
                          1.97-decade fiat confound. 0.20 = one tenth of it, a
                          10x discrimination margin. The prior session's finding
                          was that a 0.50 ceiling against that separation could
                          NOT distinguish genuine mediation from 063a's confound.
  G2_harm_floor           EXP-1394 says ">= 10x the measurement floor" and
                          ARC-029 P3 "an order of magnitude above the measurement
                          floor"; neither defines the floor. For a rate estimated
                          from DISCRETE harm events over N steps the measurement
                          floor is one resolvable event, 1/N. So G2 is:
                          >= 10 harm events per seed per cell, on >= 4/5 seeds.
                          No constant is invented and it scales with the budget.
                          (ARC-029 P3's "063a's ~-0.055 harm/step is a
                          demonstrated workable operating point" is STALE: the
                          recalibrated env measures 0.0025-0.0068 and the prior
                          session's real-length smoke 0.0056-0.0136 against the
                          0.01 absolute floor it had written -- i.e. the DV sat
                          BELOW its own gate. That absolute floor was never in
                          EXP-1394 or the claim and is not used here.)

CRITERIA (transcribed; C1's bar is NOT re-expressed)
---------------------------------------------------------------------------
  C1_harm_gap_stable      LOAD-BEARING. Paired per-seed (uncommitted - committed)
                          harm-per-step delta in the STABLE env exceeds
                          `max(0.5 * SD(per-seed paired delta), 0.002 harm/step)`
                          with the same sign on >= 4/5 seeds. VERBATIM from
                          ARC-029's CONFIRMING branch and EXP-1394's
                          acceptance_checks. The 2026-09-18 build record sec 6
                          RECOMMENDS re-expressing this bar relatively; that is a
                          governance act on a pre-registered falsifier and is NOT
                          performed here. Instead the relative effect and the
                          across-seed SD of harm/step -- the quantity the build
                          record says nobody has measured and a re-queue must
                          supply -- are emitted as pre-registered DESCRIPTIVES so
                          governance can re-express C1 on measured ground.
  C2_context_dependence   gap_volatile < gap_stable on >= 4/5 seeds.
  C3_density_interaction  the gap is larger at LOW hazard density (3) than at
                          HIGH (7), on >= 4/5 seeds (Humphries 2012 sharpening,
                          already in ARC-029's own notes).
  C4_no_fatal_errors      >= 5 seeds completed, no cell errored.
  COMBINATION RULE: outcome PASS requires G0 AND G1 AND G2 AND C1. C2/C3 are
  recorded, select the interpretation label, and do NOT gate -- EXP-1394 marks
  only G0/G1/G2 as GATING and C1 as the claim's own load-bearing criterion.

FALSIFICATION BRANCH, verbatim from ARC-029's what_would_answer
---------------------------------------------------------------------------
"With P1-P3 all met and the precision-invariance gate P2 clean, the committed and
uncommitted windows show no harm difference ... Because the manipulation is
threshold-side, such a null CANNOT be explained away as a precision artifact,
which is the one exemption every prior ARC-029 test has been able to claim. This
would be a genuine falsification: the commitment gate would be shown to be
structurally real (ARC-016, `stable`) but behaviourally inert on harm, and
ARC-029 should be narrowed to the structural claim or retired in favour of
ARC-016. Distinguish this from 'still an instrument problem' strictly by P1/P2:
if occupancy or precision-invariance fails to gate cleanly, the run is
`non_contributory`, not evidence."

Prior record cited: EXQ-125 (weakens/mixed), V3-EXQ-227 / 630 / 063a
(non_contributory), EXQ-199 (BreathOscillator attempt, committed_step_count = 0
on both seeds -- amplitude uncalibrated to the lineage's operating point, which
is exactly what the calibration phase above exists to prevent).

DV-SYMMETRY INVARIANCE (mandatory declaration), per arm
---------------------------------------------------------------------------
ARM_ALTERNATE. Manipulation: a multiplicative reduction of the commit BAR during
the sweep phase. DV: mean harm-per-step in env steps binned by the prevailing
committed flag. The DV's symmetry group is (a) permutation of env steps WITHIN a
bin and (b) an affine rescaling of the harm channel. The manipulation is not
invariant under either: it changes WHICH steps fall in which bin (it moves the
gate's decision, which is a comparison against the bar), and it does not touch
the harm channel at all. It is specifically NOT a broadcast additive constant
across candidates -- it does not enter `score_trajectory` -- so the 604c
argmax-invariance class does not apply.
ARM_STATIC. No manipulation; it is the G1 reference and the C1 descriptive
control. Its own committed-vs-uncommitted contrast is reported but is NOT
pre-registered evidence, precisely because there commitment is ENDOGENOUS (low
rv = a predictable world = independently lower harm). Breaking that
correlational confound is what the exogenous sweep is for, and the
intention-to-treat contrast (sweep-phase vs inter-sweep harm, which conditions
on the CLOCK rather than on the realised state) is emitted alongside C1 as the
descriptive that carries no such confound.

MECH-131 NOTE. Not applicable: the DV is a behavioural outcome (harm per env
step), not a candidate-set statistic, so no `selected_action_entropy` reading is
required to license it. A later session adding a candidate-diversity secondary
DV would inherit that RED and must gate it separately.

INSTRUMENTATION -- three traps this driver is written around
---------------------------------------------------------------------------
1. Commitment is read from `agent.e3.last_score_diagnostics["committed"]` -- the
   select() gate's OWN flag. `get_commitment_state()` was a SECOND, disagreeing
   predicate (53.7% disagreement at q=0.50) until red-team fix 3; it agrees now,
   but it is evaluated at READ time against an rv that post_action_update has
   already moved, so it is still not the gate's decision AT select time. Its
   value is recorded for cross-checking, never routed on.
2. `last_score_diagnostics` LATCHES -- it is populated only inside select(), and
   on a non-E3 tick it still holds the previous selection. The driver CLEARS it
   in a `StepHooks.on_sense` hook (which fires after sense() and before
   select_action), so a non-None read is always a FRESH selection. Occupancy and
   run length are counted in SELECT CALLS (the unit P1's >= 3 is in, per the
   build record's own re-measurement); `n_latched_ticks` is emitted so the true
   denominator is auditable. The harm DV is per ENV step, attributed to the
   PREVAILING committed flag -- that is the definition of a committed window, not
   pseudo-replication, and both denominators are recorded separately.
   It also requires `agent.e3.e3_score_decomp_enabled = True` (a diagnostics-only
   gate; measured at runtime 2026-09-19 -- without it the dict is never populated
   and occupancy reads 0 selections out of ~120, silently).
3. A driver that never calls post_action_update pins rv at `precision_init`
   forever (V3-EXQ-925a), producing the ALL-EQUAL window red-team fix 1 guards --
   which returns None and silently restores the ABSOLUTE bar and its saturation.
   `StepHarness` calls it every step (its step() docstring, item 10), and the
   readiness preconditions below assert the window is FULL and the bar is
   actually IN FORCE rather than trusting that.

SLEEP DRIVER: not applicable (no sleep flag is set by this driver).
"""

import argparse
import ast
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments._harness import StepHarness, StepHooks
from experiments._lib.allon_training import _train_all_on_agent
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from experiment_protocol import emit_outcome

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_1066_arc029_commitment_mode_harm_variance_bar"
QUEUE_ID = "V3-EXQ-1066"
CLAIM_IDS = ["ARC-029"]
SUPERSEDES = "V3-EXQ-063a"

SEEDS = [0, 42, 100, 123, 200]          # 125a's set; seed 44 is not in it (reef instability)
MIN_SEEDS_PASSING = 4                    # ">= 4/5 seeds", verbatim from the claim

# --- pre-registered operating point (derivation doc sec D1) ------------------
COMMIT_QUANTILE = 0.50
COMMIT_WINDOW = 200
E3_STEPS_PER_TICK = 3
PRECISION_EMA_ALPHA = 0.05               # the DEFAULT, held deliberately

# --- pre-registered sweep geometry (derivation doc sec D2) -------------------
SWEEP_DURATION = 18                      # env steps = 6 select calls = 2x P1's floor
BREATH_PERIOD = 54                       # 3x the sweep -> 12 select calls between sweeps
AMPLITUDE_LADDER = (0.02, 0.05, 0.10)    # the measured workable band
CAL_OCCUPANCY_BAND = (0.20, 0.80)        # interior to P1's [0.15, 0.85]
CAL_RUNLEN_FLOOR = 4.5                   # 1.5x P1's >= 3

# --- pre-registered gates ----------------------------------------------------
G0_OCCUPANCY_BAND = (0.15, 0.85)         # ARC-029 P1, verbatim
G0_RUNLEN_FLOOR = 3.0                    # ARC-029 P1, verbatim
G1_LOG10_PRECISION_BOUND = 0.20          # derivation doc sec D4 (1.97 / 10)
G2_MIN_HARM_EVENTS = 10                  # "10x the measurement floor" with floor = 1/N

# --- pre-registered criteria -------------------------------------------------
C1_ABS_FLOOR = 0.002                     # verbatim: max(0.5*SD(delta), 0.002)
C1_SD_MULTIPLIER = 0.5                   # verbatim

# --- readiness floors (same statistic the gates route on) --------------------
RV_WINDOW_RELATIVE_SPREAD_FLOOR = 1e-6   # below this the all-equal guard fires and
                                         # the ABSOLUTE bar silently returns
BAR_IN_FORCE_FRACTION_FLOOR = 0.95       # fraction of measured select calls with a
                                         # non-None variance_tracking_commit_bar

# --- budgets -----------------------------------------------------------------
STEPS_PER_EPISODE = 120
P0_EPISODES = 30
P1_EPISODES = 60
ZWORLD_P0_EPISODES = 20                  # SD-070: without this z_world stays a frozen
                                         # random projection and rv never collapses
EPISODES_PER_RUN = P0_EPISODES + P1_EPISODES     # == the [train] ep N/M denominator
WARM_EPISODE_CAP = 30                    # cap on the per-cell window-fill prefix
CAL_EPISODES = 15
MEAS_EPISODES = 30

ARM_STATIC = "ARM_STATIC"
ARM_ALTERNATE = "ARM_ALTERNATE"
ARMS = (ARM_STATIC, ARM_ALTERNATE)

VOL_STABLE, VOL_VOLATILE = "stable", "volatile"
DENSITY_LOW, DENSITY_HIGH = 3, 7
TRAIN_DENSITY = 5                        # the midpoint: neither eval density is trained on

# Env parameters are the ARC-029 LINEAGE's own (V3-EXQ-125a / 063a), NOT raised.
# Continuity is what keeps C1's 0.002 bar interpretable -- it was registered
# against this lineage. The 2026-09-18 build record's recommendation (ii) is
# followed: do NOT raise `hazard_harm` / `proximity_harm_scale` to make the bar
# reachable, because more harm changes the prediction-error statistics, hence rv,
# hence the very distribution the quantile bar is taken over -- a
# confound-generating move made to fix a confound.
BASE_ENV = dict(
    size=12, num_resources=5, hazard_harm=0.05,
    proximity_harm_scale=0.15, proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15, hazard_field_decay=0.5,
    use_proxy_fields=True,
)
TRAIN_DRIFT = dict(env_drift_interval=5, env_drift_prob=0.1)
STABLE_DRIFT = dict(env_drift_interval=50, env_drift_prob=0.0)
VOLATILE_DRIFT = dict(env_drift_interval=3, env_drift_prob=0.4)

_ZG = ZGoalStreamAccumulator()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _flat_scalar(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _mean(xs: List[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _sd(xs: List[float]) -> Optional[float]:
    return statistics.pstdev(xs) if len(xs) >= 2 else None


def _make_env(seed: int, density: int, drift: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, num_hazards=density, **BASE_ENV, **drift)


def _config_slice(env: CausalGridWorldV2) -> Dict[str, Any]:
    """Everything the OFF/trained computation reads. Declared so the per-cell
    arm_fingerprint is honest about what it covers."""
    return dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32, world_dim=32,
        alpha_world=0.9,                 # SD-008: 0.3 (the default) starves z_world
        use_variance_tracking_commit_threshold=True,
        commit_threshold_quantile=COMMIT_QUANTILE,
        commit_threshold_quantile_window=COMMIT_WINDOW,
        breath_period=0,                 # OFF during training; set per ARM at eval
        breath_sweep_amplitude=0.0,
        breath_sweep_duration=SWEEP_DURATION,
    )


def _build_agent(env: CausalGridWorldV2) -> Tuple[REEAgent, Dict[str, Any]]:
    slice_ = _config_slice(env)
    config = REEConfig.from_dims(**slice_)
    # The three lever fields ARE from_dims parameters (config.py:8806-8808) and
    # the selector RAISES on the unset sentinels, so a silent from_dims swallow
    # cannot pass silently. Re-assert anyway -- it costs nothing and this is the
    # exact failure mode the sentinels were built for.
    assert config.e3.use_variance_tracking_commit_threshold is True
    assert config.e3.commit_threshold_quantile == COMMIT_QUANTILE
    assert config.e3.commit_threshold_quantile_window == COMMIT_WINDOW
    # These two are NOT from_dims parameters -- they live on the sub-configs.
    # getattr-check first so a rename turns into a loud failure rather than a new
    # attribute nobody reads.
    assert hasattr(config.e3, "precision_ema_alpha")
    assert hasattr(config.heartbeat, "e3_steps_per_tick")
    config.e3.precision_ema_alpha = PRECISION_EMA_ALPHA
    config.heartbeat.e3_steps_per_tick = E3_STEPS_PER_TICK
    agent = REEAgent(config)
    # MECH-463 instrumentation gate. Diagnostics-only, but WITHOUT it
    # last_score_diagnostics is never populated and every occupancy reads zero.
    agent.e3.e3_score_decomp_enabled = True
    assert agent.clock.e3_steps_per_tick == E3_STEPS_PER_TICK
    assert agent.e3.config.precision_ema_alpha == PRECISION_EMA_ALPHA
    assert agent.e3._commit_gate_variance_window is not None
    assert agent.e3._commit_gate_variance_window.maxlen == COMMIT_WINDOW
    return agent, slice_


def _set_breath(agent: REEAgent, period: int, amplitude: float) -> None:
    """Arm / disarm the MECH-108 oscillator on an ALREADY-CONSTRUCTED clock.

    The clock reads its breath parameters at construction, so these are the
    live attributes; asserting they exist first keeps a rename loud.
    """
    for attr in ("_breath_period", "_sweep_amplitude", "_sweep_duration"):
        assert hasattr(agent.clock, attr), attr
    agent.clock._breath_period = int(period)
    agent.clock._sweep_amplitude = float(amplitude)
    agent.clock._sweep_duration = int(SWEEP_DURATION)
    agent.clock._breath_phase_step = 0
    # phase reset to 0 puts the cycle at the START of the inter-sweep phase in
    # both arms (sweep fires in the LAST sweep_duration steps of each period),
    # so no cell begins mid-sweep.
    assert agent.clock.sweep_active is False


def _assert_no_running_variance_writes() -> Dict[str, Any]:
    """G1's SOURCE-level half: this driver must never ASSIGN to
    `e3._running_variance`. EXP-1394 makes it an explicit assert, and it is the
    one check V3-EXQ-063a would have failed. An AST walk, not a substring scan:
    this file's own docstring discusses `_running_variance` at length and a
    textual check would flag its own explanation of the defect.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    offenders: List[int] = []

    def _targets(node: ast.AST) -> List[ast.AST]:
        if isinstance(node, ast.Assign):
            return list(node.targets)
        if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            return [node.target]
        return []

    for node in ast.walk(tree):
        for tgt in _targets(node):
            for sub in ast.walk(tgt):
                if isinstance(sub, ast.Attribute) and sub.attr == "_running_variance":
                    offenders.append(getattr(node, "lineno", -1))
    return {"n_assignments": len(offenders), "lines": sorted(set(offenders)),
            "method": "ast_walk_over_assign_targets"}


class _Roll:
    """One (arm x env) rollout: steps the agent and records both denominators."""

    def __init__(self, agent: REEAgent, env: CausalGridWorldV2, seed: int):
        self.agent, self.env, self.seed = agent, env, seed
        self.hooks = StepHooks(on_sense=self._clear)
        self.reset_counters()

    def _clear(self, **_kw) -> None:
        # Fires after sense() and BEFORE select_action -- see step()'s docstring,
        # order item 6. A None read downstream therefore means "no fresh
        # selection this tick", never "stale copy of the last one".
        self.agent.e3.last_score_diagnostics = None

    def reset_counters(self) -> None:
        self.committed_flags: List[bool] = []      # one per SELECT CALL
        self.window_fill: List[int] = []
        self.bar_in_force: List[bool] = []
        self.rv_window_spread: List[float] = []
        self.log10_precision: List[float] = []
        self.harm_by_state: Dict[bool, List[float]] = {True: [], False: []}
        self.harm_by_sweep: Dict[bool, List[float]] = {True: [], False: []}
        self.n_env_steps = 0
        self.n_latched_ticks = 0
        self.n_harm_events = 0
        self.n_unattributed_steps = 0

    def run(self, episodes: int, steps: int, collect: bool,
            stop_when_window_full: bool = False) -> None:
        agent = self.agent
        committed_now: Optional[bool] = None
        for _ep in range(episodes):
            # Same contract as StepHarness.run_episode (_harness.py:455-457):
            # env.reset() + agent.reset() + harness.reset() BEFORE the first
            # step. step() is driven directly rather than through run_episode
            # because the committed flag has to be read between the hook's clear
            # and the next sense(), which on_step cannot see.
            _flat, obs = self.env.reset()
            agent.reset()
            harness = StepHarness(agent, self.env, train_mode=False,
                                  hooks=self.hooks, seed=self.seed)
            for _ in range(steps):
                sweep = bool(agent.clock.sweep_active)
                result = harness.step(obs)
                obs = result.next_obs_dict
                diag = agent.e3.last_score_diagnostics
                fresh = isinstance(diag, dict) and "committed" in diag
                if fresh:
                    committed_now = bool(diag["committed"])
                    if collect:
                        self.committed_flags.append(committed_now)
                        self.window_fill.append(int(diag.get("commit_gate_window_fill", 0)))
                        self.bar_in_force.append(
                            diag.get("variance_tracking_commit_bar") is not None)
                        self.log10_precision.append(
                            math.log10(max(agent.e3.current_precision, 1e-300)))
                        win = agent.e3._commit_gate_variance_window
                        if win is not None and len(win) >= 2:
                            hi, lo = max(win), min(win)
                            self.rv_window_spread.append(
                                (hi - lo) / abs(hi) if hi > 0 else 0.0)
                else:
                    self.n_latched_ticks += 1
                self.n_env_steps += 1
                harm = -float(result.harm_signal) if result.harm_signal < 0 else 0.0
                if harm > 0.0:
                    self.n_harm_events += 1
                if collect:
                    if committed_now is None:
                        self.n_unattributed_steps += 1
                    else:
                        self.harm_by_state[committed_now].append(harm)
                        self.harm_by_sweep[sweep].append(harm)
                if result.done:
                    break
            _ZG.observe(agent)
            if stop_when_window_full:
                win = agent.e3._commit_gate_variance_window
                if win is not None and len(win) >= COMMIT_WINDOW:
                    return

    # -- derived quantities ---------------------------------------------------
    def run_length_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        cur = 0
        for c in self.committed_flags:
            if c:
                cur += 1
            elif cur:
                hist[str(cur)] = hist.get(str(cur), 0) + 1
                cur = 0
        if cur:
            hist[str(cur)] = hist.get(str(cur), 0) + 1
        return hist

    def summary(self) -> Dict[str, Any]:
        hist = self.run_length_histogram()
        runs = [int(k) * v for k, v in hist.items()]
        n_runs = sum(hist.values())
        n_sel = len(self.committed_flags)
        harm_c, harm_u = self.harm_by_state[True], self.harm_by_state[False]
        sw, nosw = self.harm_by_sweep[True], self.harm_by_sweep[False]
        return {
            "n_select_calls": n_sel,
            "n_env_steps": self.n_env_steps,
            "n_latched_ticks": self.n_latched_ticks,
            "n_unattributed_steps": self.n_unattributed_steps,
            "committed_step_fraction": (sum(self.committed_flags) / n_sel) if n_sel else None,
            "mean_committed_run_length": (sum(runs) / n_runs) if n_runs else 0.0,
            "committed_run_length_histogram": hist,
            "n_committed_runs": n_runs,
            "commit_gate_window_fill_max": max(self.window_fill) if self.window_fill else 0,
            "bar_in_force_fraction": (
                sum(self.bar_in_force) / len(self.bar_in_force)) if self.bar_in_force else 0.0,
            "rv_window_relative_spread_min": (
                min(self.rv_window_spread) if self.rv_window_spread else 0.0),
            "log10_precision_mean": _mean(self.log10_precision),
            "log10_precision_sd": _sd(self.log10_precision),
            "n_harm_events": self.n_harm_events,
            "harm_per_step_committed": _mean(harm_c),
            "harm_per_step_uncommitted": _mean(harm_u),
            "n_steps_committed": len(harm_c),
            "n_steps_uncommitted": len(harm_u),
            "harm_per_step_overall": _mean(harm_c + harm_u),
            # intention-to-treat: conditions on the CLOCK, not on the realised
            # state, so it carries none of the endogenous-commitment confound.
            "harm_per_step_sweep_phase": _mean(sw),
            "harm_per_step_inter_sweep": _mean(nosw),
            "n_steps_sweep_phase": len(sw),
        }


def _warm_and_measure(agent: REEAgent, env: CausalGridWorldV2, seed: int,
                      episodes: int, steps: int) -> Dict[str, Any]:
    """Fill the commit-gate window (discarded), then measure on disjoint episodes.

    The warm prefix is what makes the measured portion comparable across cells:
    until the window is FULL the bar returns None and the ABSOLUTE 0.40 bar is in
    force -- i.e. the saturated pre-lever regime. Including that prefix would
    contaminate occupancy with exactly the defect this run exists to remove.
    """
    roll = _Roll(agent, env, seed)
    roll.run(WARM_EPISODE_CAP, steps, collect=False, stop_when_window_full=True)
    win = agent.e3._commit_gate_variance_window
    warm_fill = 0 if win is None else len(win)
    roll.reset_counters()
    roll.run(episodes, steps, collect=True)
    out = roll.summary()
    out["warm_prefix_window_fill"] = warm_fill
    return out


def _run_cell(agent: REEAgent, seed: int, arm: str, amplitude: float,
              density: int, volatility: str, episodes: int, steps: int,
              cfg_slice: Dict[str, Any], label: str) -> Dict[str, Any]:
    """One (arm x density x volatility) measurement cell.

    CELL ISOLATION, and why it is not a deepcopy. The intended design was a
    pristine `copy.deepcopy` of the seed's trained agent per cell, so no cell
    could inherit another's residue / rv state. MEASURED 2026-09-19: `REEAgent`
    is NOT deepcopy-able -- torch refuses with "Only Tensors created explicitly
    by the user (graph leaves) support the deepcopy protocol", it still refuses
    after `agent.reset()`, and after a no_grad pass it fails differently
    ("cannot pickle 'module' object"). Re-training per cell would cost 8x the
    training budget AND destroy G1's premise, which is that both arms are the
    SAME trained agent so the precision comparison isolates the manipulation.

    So one agent per seed is reused across its cells, and the two carry-over
    hazards are handled explicitly rather than hoped away:
      - the COMMIT-GATE WINDOW is cleared and re-warmed from scratch per cell,
        so no cell's bar is computed over another cell's rv samples;
      - `agent.reset()` runs per episode (the run_episode contract);
      - ARM ORDER IS COUNTERBALANCED ACROSS SEEDS (see `run()`), so the residue
        field -- which `agent.reset()` deliberately does NOT clear -- cannot
        confound arm with position. The realised order is recorded per cell.
    """
    drift = STABLE_DRIFT if volatility == VOL_STABLE else VOLATILE_DRIFT
    env = _make_env(seed * 1000 + density + (7 if volatility == VOL_VOLATILE else 0),
                    density, drift)
    cell_slice = dict(cfg_slice)
    cell_slice.update(arm=arm, sweep_amplitude=amplitude, density=density,
                      volatility=volatility, e3_steps_per_tick=E3_STEPS_PER_TICK,
                      precision_ema_alpha=PRECISION_EMA_ALPHA)
    with arm_cell(seed, config_slice=cell_slice, script_path=Path(__file__)) as cell:
        agent.e3.e3_score_decomp_enabled = True
        agent.reset()
        win = agent.e3._commit_gate_variance_window
        assert win is not None
        win.clear()
        _set_breath(agent, BREATH_PERIOD if arm == ARM_ALTERNATE else 0,
                    amplitude if arm == ARM_ALTERNATE else 0.0)
        row = _warm_and_measure(agent, env, seed, episodes, steps)
        row.update(seed=seed, arm=arm, sweep_amplitude=amplitude, density=density,
                   volatility=volatility, cell_label=label,
                   get_commitment_state_crosscheck=dict(agent.e3.get_commitment_state()))
        cell.stamp(row)
    return row


def _calibrate(trained: REEAgent, seed: int, cfg_slice: Dict[str, Any],
               steps: int, episodes: int) -> Dict[str, Any]:
    """Choose sweep_amplitude per ARC-029 P2's rule, on episodes disjoint from
    the measured ones. Returns the chosen rung plus every rung's readings."""
    ref = _run_cell(trained, seed, ARM_STATIC, 0.0, TRAIN_DENSITY, VOL_STABLE,
                    episodes, steps, cfg_slice, f"cal_ref_seed{seed}")
    ref_logp = ref["log10_precision_mean"]
    rungs: List[Dict[str, Any]] = []
    for a in AMPLITUDE_LADDER:
        r = _run_cell(trained, seed, ARM_ALTERNATE, a, TRAIN_DENSITY, VOL_STABLE,
                      episodes, steps, cfg_slice, f"cal_a{a}_seed{seed}")
        occ = r["committed_step_fraction"]
        runlen = r["mean_committed_run_length"]
        d_logp = (None if (ref_logp is None or r["log10_precision_mean"] is None)
                  else abs(r["log10_precision_mean"] - ref_logp))
        ok = (occ is not None
              and CAL_OCCUPANCY_BAND[0] <= occ <= CAL_OCCUPANCY_BAND[1]
              and runlen >= CAL_RUNLEN_FLOOR
              and d_logp is not None and d_logp <= G1_LOG10_PRECISION_BOUND)
        rungs.append({"sweep_amplitude": a, "committed_step_fraction": occ,
                      "mean_committed_run_length": runlen,
                      "abs_delta_log10_precision_vs_static": d_logp,
                      "qualifies": bool(ok),
                      "committed_run_length_histogram":
                          r["committed_run_length_histogram"]})
        print(f"  [cal] seed={seed} a={a} occ={occ} runlen={runlen} "
              f"dlog10P={d_logp} qualifies={ok}", flush=True)
    qualifying = [r for r in rungs if r["qualifies"]]
    chosen = max((r["sweep_amplitude"] for r in qualifying), default=None)
    return {"seed": seed, "reference_static_log10_precision_mean": ref_logp,
            "rungs": rungs, "chosen_sweep_amplitude": chosen,
            "calibration_succeeded": chosen is not None,
            "reference_cell": ref}


def _train_seed(seed: int, steps: int, p0: int, p1: int,
                zp0: int) -> Tuple[REEAgent, Dict[str, Any], Dict[str, Any]]:
    train_env = _make_env(seed, TRAIN_DENSITY, TRAIN_DRIFT)
    agent, cfg_slice = _build_agent(train_env)
    zw_env = _make_env(seed, TRAIN_DENSITY, TRAIN_DRIFT)  # dedicated: P0a consumes env RNG
    rv_before = float(agent.e3._running_variance)
    stats = _train_all_on_agent(
        agent, train_env, seed=seed, p0_episodes=p0, p1_episodes=p1,
        steps_per_episode=steps, rung_id=f"arc029_seed{seed}",
        total_denominator=max(1, p0 + p1),
        # SD-070: without this the world encoder is never stepped and z_world
        # stays a frozen random projection, so rv never collapses and the whole
        # premise (a trained operating point) is absent.
        zworld_p0_episodes=zp0, zworld_p0_env=zw_env,
    )
    stats = dict(stats)
    stats["running_variance_before_training"] = rv_before
    stats["running_variance_after_training"] = float(agent.e3._running_variance)
    return agent, cfg_slice, stats


def analyse(rows: List[Dict[str, Any]], cals: List[Dict[str, Any]],
            src_scan: Dict[str, Any], n_seeds_completed: int) -> Dict[str, Any]:
    def cell(seed: int, arm: str, density: int, vol: str) -> Optional[Dict[str, Any]]:
        for r in rows:
            if (r["seed"] == seed and r["arm"] == arm
                    and r["density"] == density and r["volatility"] == vol):
                return r
        return None

    seeds = sorted({r["seed"] for r in rows})

    # ---- G0: occupancy + run length, in ARM_ALTERNATE, stable env, low density
    g0_rows, g0_pass_seeds = [], []
    for s in seeds:
        c = cell(s, ARM_ALTERNATE, DENSITY_LOW, VOL_STABLE)
        if c is None:
            continue
        occ, rl = c["committed_step_fraction"], c["mean_committed_run_length"]
        ok = (occ is not None and G0_OCCUPANCY_BAND[0] <= occ <= G0_OCCUPANCY_BAND[1]
              and rl >= G0_RUNLEN_FLOOR)
        g0_rows.append({"seed": s, "committed_step_fraction": occ,
                        "mean_committed_run_length": rl,
                        "committed_run_length_histogram": c["committed_run_length_histogram"],
                        "passed": bool(ok)})
        if ok:
            g0_pass_seeds.append(s)
    g0_pass = len(g0_pass_seeds) >= MIN_SEEDS_PASSING

    # ---- G1: precision invariance, ALTERNATE vs STATIC, paired per seed
    g1_rows, g1_pass_seeds, g1_mean_deltas = [], [], []
    for s in seeds:
        a = cell(s, ARM_ALTERNATE, DENSITY_LOW, VOL_STABLE)
        b = cell(s, ARM_STATIC, DENSITY_LOW, VOL_STABLE)
        if a is None or b is None:
            continue
        dm = (None if (a["log10_precision_mean"] is None or b["log10_precision_mean"] is None)
              else a["log10_precision_mean"] - b["log10_precision_mean"])
        ds = (None if (a["log10_precision_sd"] is None or b["log10_precision_sd"] is None)
              else a["log10_precision_sd"] - b["log10_precision_sd"])
        ok = (dm is not None and ds is not None
              and abs(dm) <= G1_LOG10_PRECISION_BOUND
              and abs(ds) <= G1_LOG10_PRECISION_BOUND)
        g1_rows.append({"seed": s, "delta_mean_log10_precision": dm,
                        "delta_sd_log10_precision": ds, "passed": bool(ok)})
        if dm is not None:
            g1_mean_deltas.append(dm)
        if ok:
            g1_pass_seeds.append(s)
    g1_across_seed_sd = _sd(g1_mean_deltas)
    g1_power_ok = (g1_across_seed_sd is not None
                   and g1_across_seed_sd <= G1_LOG10_PRECISION_BOUND)
    g1_source_ok = src_scan["n_assignments"] == 0
    g1_pass = (len(g1_pass_seeds) >= MIN_SEEDS_PASSING and g1_power_ok and g1_source_ok)

    # ---- G2: harm floor, per seed per cell
    g2_rows, g2_pass_seeds = [], []
    for s in seeds:
        cells = [r for r in rows if r["seed"] == s]
        worst = min((r["n_harm_events"] for r in cells), default=0)
        worst_cell = min(cells, key=lambda r: r["n_harm_events"], default=None)
        ok = worst >= G2_MIN_HARM_EVENTS
        g2_rows.append({"seed": s, "min_harm_events_over_cells": worst,
                        "offending_cell": None if worst_cell is None
                        else worst_cell["cell_label"], "passed": bool(ok)})
        if ok:
            g2_pass_seeds.append(s)
    g2_pass = len(g2_pass_seeds) >= MIN_SEEDS_PASSING

    # ---- C1: paired committed-vs-uncommitted harm gap, ARM_ALTERNATE, stable
    def gap(seed: int, density: int, vol: str, arm: str = ARM_ALTERNATE):
        c = cell(seed, arm, density, vol)
        if c is None:
            return None
        hc, hu = c["harm_per_step_committed"], c["harm_per_step_uncommitted"]
        if hc is None or hu is None:
            return None
        return hu - hc            # positive = committed is SAFER (ARC-029's direction)

    gaps_stable = {s: gap(s, DENSITY_LOW, VOL_STABLE) for s in seeds}
    gaps_volatile = {s: gap(s, DENSITY_LOW, VOL_VOLATILE) for s in seeds}
    gaps_high_density = {s: gap(s, DENSITY_HIGH, VOL_STABLE) for s in seeds}
    gaps_static_stable = {s: gap(s, DENSITY_LOW, VOL_STABLE, ARM_STATIC) for s in seeds}

    vals = [v for v in gaps_stable.values() if v is not None]
    sd_delta = _sd(vals)
    c1_bar = max(C1_SD_MULTIPLIER * sd_delta, C1_ABS_FLOOR) if sd_delta is not None \
        else C1_ABS_FLOOR
    c1_seeds = [s for s, v in gaps_stable.items() if v is not None and v > c1_bar]
    c1_pass = len(c1_seeds) >= MIN_SEEDS_PASSING

    c2_seeds = [s for s in seeds
                if gaps_stable.get(s) is not None and gaps_volatile.get(s) is not None
                and gaps_volatile[s] < gaps_stable[s]]
    c2_pass = len(c2_seeds) >= MIN_SEEDS_PASSING

    c3_seeds = [s for s in seeds
                if gaps_stable.get(s) is not None and gaps_high_density.get(s) is not None
                and gaps_stable[s] > gaps_high_density[s]]
    c3_pass = len(c3_seeds) >= MIN_SEEDS_PASSING

    c4_pass = (n_seeds_completed >= 5
               and all(r.get("n_select_calls", 0) > 0 for r in rows))

    # ---- readiness preconditions (the SAME statistics the gates route on) ----
    fills = [r["commit_gate_window_fill_max"] for r in rows]
    in_force = [r["bar_in_force_fraction"] for r in rows]
    spreads = [r["rv_window_relative_spread_min"] for r in rows]
    worst_fill_row = min(rows, key=lambda r: r["commit_gate_window_fill_max"], default=None)
    worst_force_row = min(rows, key=lambda r: r["bar_in_force_fraction"], default=None)
    worst_spread_row = min(rows, key=lambda r: r["rv_window_relative_spread_min"], default=None)
    n_cal_ok = sum(1 for c in cals if c["calibration_succeeded"])

    preconditions = [
        {"name": "commit_gate_window_full_every_cell",
         "description": ("worst cell's commit_gate_window_fill. Below W the bar returns "
                         "None and the ABSOLUTE 0.40 bar is in force -- the saturated "
                         "pre-lever regime, not a measurement of this lever."),
         "measured": min(fills) if fills else 0, "threshold": COMMIT_WINDOW,
         "direction": "lower", "control": "worst cell over the whole seed x arm grid",
         "offending_cell": None if worst_fill_row is None else worst_fill_row["cell_label"],
         "met": bool(fills and min(fills) >= COMMIT_WINDOW)},
        {"name": "variance_tracking_bar_in_force",
         "description": ("worst cell's fraction of measured SELECT CALLS with a non-None "
                         "variance_tracking_commit_bar -- the gate actually using the "
                         "lever rather than falling back to the absolute bar."),
         "measured": min(in_force) if in_force else 0.0,
         "threshold": BAR_IN_FORCE_FRACTION_FLOOR, "direction": "lower",
         "control": "worst cell over the whole seed x arm grid",
         "offending_cell": None if worst_force_row is None else worst_force_row["cell_label"],
         "met": bool(in_force and min(in_force) >= BAR_IN_FORCE_FRACTION_FLOOR)},
        {"name": "rv_window_non_degenerate",
         "description": ("worst cell's minimum relative spread of the commit-gate-variance "
                         "window. At or below the all-equal guard's epsilon the estimator "
                         "returns None and saturation silently returns (red-team fix 1; "
                         "the V3-EXQ-925a no-post_action_update shape)."),
         "measured": min(spreads) if spreads else 0.0,
         "threshold": RV_WINDOW_RELATIVE_SPREAD_FLOOR, "direction": "lower",
         "control": "worst cell over the whole seed x arm grid",
         "offending_cell": None if worst_spread_row is None else worst_spread_row["cell_label"],
         "met": bool(spreads and min(spreads) > RV_WINDOW_RELATIVE_SPREAD_FLOOR)},
        {"name": "sweep_amplitude_calibrated",
         "description": ("seeds for which some rung of the pre-registered amplitude ladder "
                         "met all three calibration conditions. ARC-029 P2 requires the "
                         "amplitude be set from the run's own rv distribution; a ladder "
                         "with no qualifying rung means the manipulation has no workable "
                         "operating point on THIS agent."),
         "measured": n_cal_ok, "threshold": MIN_SEEDS_PASSING, "direction": "lower",
         "control": "calibration episodes, disjoint from the measured ones",
         "met": bool(n_cal_ok >= MIN_SEEDS_PASSING)},
        {"name": "driver_never_writes_running_variance",
         "description": ("EXP-1394's G1 source-level assert, and the one check "
                         "V3-EXQ-063a would have failed."),
         "measured": src_scan["n_assignments"], "threshold": 0,
         "direction": "upper", "control": "AST-free source scan of this file",
         "met": bool(g1_source_ok)},
    ]
    gate_green = all(p["met"] for p in preconditions)

    if not gate_green or not (g0_pass and g1_pass and g2_pass):
        label = "non_contributory_preconditions_unmet"
        routes_to = ("ARC-029 stays candidate; the run is NOT evidence. ARC-029's own "
                     "instruction: 'if occupancy or precision-invariance fails to gate "
                     "cleanly, the run is non_contributory, not evidence.'")
        direction = "unknown"
    elif c1_pass:
        label = "arc029_supported_committed_mode_lowers_harm"
        routes_to = "ARC-029 CONFIRMING branch; promote per governance."
        direction = "supports"
    else:
        label = "arc029_falsified_gate_structurally_real_behaviourally_inert"
        routes_to = ("ARC-029's own FALSIFYING branch: the commitment gate is "
                     "structurally real (ARC-016, stable) but behaviourally inert on "
                     "harm; narrow ARC-029 to the structural claim or retire it in "
                     "favour of ARC-016. This reading is available ONLY because the "
                     "manipulation is threshold-side and G1 is clean.")
        direction = "weakens"

    harm_overall = [r["harm_per_step_overall"] for r in rows
                    if r["harm_per_step_overall"] is not None]
    rel_effects: Dict[int, Optional[float]] = {}
    for s_ in seeds:
        c = cell(s_, ARM_ALTERNATE, DENSITY_LOW, VOL_STABLE)
        base = None if c is None else c["harm_per_step_overall"]
        g = gaps_stable.get(s_)
        rel_effects[s_] = (g / base) if (g is not None and base) else None

    return {
        "label": label, "routes_to": routes_to, "evidence_direction": direction,
        "gate_green": gate_green, "preconditions": preconditions,
        "g0_rows": g0_rows, "g0_pass": g0_pass, "g0_n_seeds": len(g0_pass_seeds),
        "g1_rows": g1_rows, "g1_pass": g1_pass, "g1_n_seeds": len(g1_pass_seeds),
        "g1_across_seed_sd_of_mean_delta": g1_across_seed_sd,
        "g1_power_ok": g1_power_ok, "g1_source_scan": src_scan,
        "g2_rows": g2_rows, "g2_pass": g2_pass, "g2_n_seeds": len(g2_pass_seeds),
        "c1_bar": c1_bar, "c1_sd_of_paired_delta": sd_delta,
        "c1_gaps_stable": gaps_stable, "c1_n_seeds": len(c1_seeds), "c1_pass": c1_pass,
        "c2_gaps_volatile": gaps_volatile, "c2_n_seeds": len(c2_seeds), "c2_pass": c2_pass,
        "c3_gaps_high_density": gaps_high_density, "c3_n_seeds": len(c3_seeds),
        "c3_pass": c3_pass, "c4_pass": c4_pass,
        # descriptives the build record asks a re-queue to supply, so governance
        # can re-express C1 relatively on measured ground rather than by argument
        "descriptive_harm_per_step_mean": _mean(harm_overall),
        "descriptive_harm_per_step_across_cell_sd": _sd(harm_overall),
        "descriptive_relative_gap_stable": rel_effects,
        "descriptive_static_arm_gaps_stable": gaps_static_stable,
        "calibrations": cals,
    }


def run(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:1] if dry_run else SEEDS
    steps = 24 if dry_run else STEPS_PER_EPISODE
    p0 = 1 if dry_run else P0_EPISODES
    p1 = 1 if dry_run else P1_EPISODES
    zp0 = 1 if dry_run else ZWORLD_P0_EPISODES
    cal_eps = 1 if dry_run else CAL_EPISODES
    meas_eps = 2 if dry_run else MEAS_EPISODES
    densities = (DENSITY_LOW, DENSITY_HIGH)
    vols = (VOL_STABLE, VOL_VOLATILE)

    src_scan = _assert_no_running_variance_writes()
    rows: List[Dict[str, Any]] = []
    cals: List[Dict[str, Any]] = []
    train_stats: List[Dict[str, Any]] = []
    n_done = 0

    for seed in seeds:
        agent, cfg_slice, stats = _train_seed(seed, steps, p0, p1, zp0)
        stats["seed"] = seed
        train_stats.append(stats)

        cal = _calibrate(agent, seed, cfg_slice, steps, cal_eps)
        cals.append(cal)
        amplitude = cal["chosen_sweep_amplitude"]
        if amplitude is None:
            # No rung has a workable operating point on this agent. Fall back to
            # the ladder's midpoint so the cells still RUN and record WHY they
            # cannot be read -- a silently skipped seed is a smaller n presented
            # as a clean one. G0 / the calibration precondition carry the verdict.
            amplitude = AMPLITUDE_LADDER[len(AMPLITUDE_LADDER) // 2]

        # COUNTERBALANCE: cells share one agent (see _run_cell), so a fixed arm
        # order would confound arm with position in the residue field's history.
        # Alternating by seed index makes ARM_ALTERNATE first on half the seeds.
        arm_order = ARMS if (seeds.index(seed) % 2 == 0) else tuple(reversed(ARMS))
        for arm in arm_order:
            for density in densities:
                for vol in vols:
                    label = f"{arm}_d{density}_{vol}_seed{seed}"
                    print(f"Seed {seed} Condition {arm}_d{density}_{vol}", flush=True)
                    row = _run_cell(agent, seed, arm, amplitude, density, vol,
                                    meas_eps, steps, cfg_slice, label)
                    row["calibration_succeeded"] = cal["calibration_succeeded"]
                    row["arm_order_this_seed"] = list(arm_order)
                    rows.append(row)
                    ok = (row["n_select_calls"] > 0
                          and row["n_harm_events"] >= (1 if dry_run else G2_MIN_HARM_EVENTS))
                    print(f"verdict: {'PASS' if ok else 'FAIL'}", flush=True)
        n_done += 1

    s = analyse(rows, cals, src_scan, n_done)
    outcome = "PASS" if (s["gate_green"] and s["g0_pass"] and s["g1_pass"]
                         and s["g2_pass"] and s["c1_pass"] and s["c4_pass"]) else "FAIL"

    criteria = [
        {"name": "G0_occupancy", "load_bearing": True, "gating": True,
         "passed": bool(s["g0_pass"]), "measured": s["g0_n_seeds"],
         "threshold": MIN_SEEDS_PASSING, "direction": "lower",
         "note": "per seed: occupancy in [0.15,0.85] AND mean committed-run length >= 3"},
        {"name": "G1_precision_invariance", "load_bearing": True, "gating": True,
         "passed": bool(s["g1_pass"]), "measured": s["g1_n_seeds"],
         "threshold": MIN_SEEDS_PASSING, "direction": "lower",
         "note": (f"per seed |d mean| and |d SD| of log10(precision) <= "
                  f"{G1_LOG10_PRECISION_BOUND} decades, plus the across-seed power "
                  f"condition and the source-level no-write assert")},
        {"name": "G2_harm_floor", "load_bearing": True, "gating": True,
         "passed": bool(s["g2_pass"]), "measured": s["g2_n_seeds"],
         "threshold": MIN_SEEDS_PASSING, "direction": "lower",
         "note": f">= {G2_MIN_HARM_EVENTS} harm events in every cell of the seed"},
        {"name": "C1_harm_gap_stable", "load_bearing": True, "gating": True,
         "passed": bool(s["c1_pass"]), "measured": s["c1_n_seeds"],
         "threshold": MIN_SEEDS_PASSING, "direction": "lower",
         "measured_bar": s["c1_bar"], "threshold_bar_floor": C1_ABS_FLOOR,
         "note": "paired (uncommitted - committed) harm/step > max(0.5*SD(delta), 0.002)"},
        {"name": "C2_context_dependence", "load_bearing": False, "gating": False,
         "passed": bool(s["c2_pass"]), "measured": s["c2_n_seeds"],
         "threshold": MIN_SEEDS_PASSING, "direction": "lower"},
        {"name": "C3_density_interaction", "load_bearing": False, "gating": False,
         "passed": bool(s["c3_pass"]), "measured": s["c3_n_seeds"],
         "threshold": MIN_SEEDS_PASSING, "direction": "lower"},
        {"name": "C4_no_fatal_errors", "load_bearing": False, "gating": True,
         "passed": bool(s["c4_pass"]), "measured": n_done, "threshold": 5,
         "direction": "lower"},
    ]

    readout: Dict[str, float] = {}
    for key, value in (
        ("g0_n_seeds", s["g0_n_seeds"]), ("g1_n_seeds", s["g1_n_seeds"]),
        ("g2_n_seeds", s["g2_n_seeds"]), ("c1_n_seeds", s["c1_n_seeds"]),
        ("c2_n_seeds", s["c2_n_seeds"]), ("c3_n_seeds", s["c3_n_seeds"]),
        ("c1_bar", s["c1_bar"]), ("c1_sd_of_paired_delta", s["c1_sd_of_paired_delta"]),
        ("g1_across_seed_sd_of_mean_delta", s["g1_across_seed_sd_of_mean_delta"]),
        ("harm_per_step_mean", s["descriptive_harm_per_step_mean"]),
        ("harm_per_step_across_cell_sd", s["descriptive_harm_per_step_across_cell_sd"]),
        ("gate_green", 1 if s["gate_green"] else 0),
        ("g0_pass", 1 if s["g0_pass"] else 0), ("g1_pass", 1 if s["g1_pass"] else 0),
        ("g2_pass", 1 if s["g2_pass"] else 0), ("c1_pass", 1 if s["c1_pass"] else 0),
        ("c2_pass", 1 if s["c2_pass"] else 0), ("c3_pass", 1 if s["c3_pass"] else 0),
        ("n_seeds_completed", n_done),
    ):
        coerced = _flat_scalar(value)
        if coerced is not None:
            readout[key] = coerced

    non_degenerate = bool(s["gate_green"])
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": s["evidence_direction"],
        "outcome": outcome,
        "timestamp_utc": _utc_stamp(),
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (None if non_degenerate else
                              "precondition gate not green: " + "; ".join(
                                  p["name"] for p in s["preconditions"] if not p["met"])),
        "arm_results": rows,
        "training_stats": train_stats,
        "criteria": criteria,
        "criteria_non_degenerate": {
            "G0_occupancy": bool(s["gate_green"]),
            "G1_precision_invariance": bool(s["gate_green"]),
            "G2_harm_floor": bool(s["gate_green"]),
            "C1_harm_gap_stable": bool(s["gate_green"] and s["g0_pass"]
                                       and s["g1_pass"] and s["g2_pass"]),
            "C2_context_dependence": bool(s["gate_green"]),
            "C3_density_interaction": bool(s["gate_green"]),
            "C4_no_fatal_errors": True,
        },
        "combination_rule": (
            "outcome PASS requires the readiness gate green AND G0 AND G1 AND G2 AND "
            "C1 AND C4. G0/G1/G2 are EXP-1394's GATING checks and C1 is ARC-029's own "
            "load-bearing criterion; C2 and C3 are recorded and select the "
            "interpretation label but do NOT gate. A gate failure is "
            "NON_CONTRIBUTORY, not weakening evidence -- ARC-029's what_would_answer "
            "says so explicitly."),
        "interpretation": {
            "label": s["label"],
            "preconditions": s["preconditions"],
            "criteria_non_degenerate": {
                "G0_occupancy": bool(s["gate_green"]),
                "G1_precision_invariance": bool(s["gate_green"]),
                "G2_harm_floor": bool(s["gate_green"]),
                "C1_harm_gap_stable": bool(s["gate_green"] and s["g0_pass"]
                                           and s["g1_pass"] and s["g2_pass"]),
            },
            "routes_to": s["routes_to"],
        },
        "readout": readout,
        "summary": s,
        "prereg_operating_point": {
            "commit_threshold_quantile": COMMIT_QUANTILE,
            "commit_threshold_quantile_window": COMMIT_WINDOW,
            "e3_steps_per_tick": E3_STEPS_PER_TICK,
            "precision_ema_alpha": PRECISION_EMA_ALPHA,
            "breath_period": BREATH_PERIOD,
            "breath_sweep_duration": SWEEP_DURATION,
            "sweep_amplitude_ladder": list(AMPLITUDE_LADDER),
            "derivation":
                "REE_assembly/evidence/planning/arc029_exq1066_prereg_derivation_20260919.md",
        },
    }
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t_start = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    manifest = run(dry_run=args.dry_run)

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run,
        config={
            "base_env": BASE_ENV, "train_drift": TRAIN_DRIFT,
            "stable_drift": STABLE_DRIFT, "volatile_drift": VOLATILE_DRIFT,
            "train_density": TRAIN_DENSITY, "densities": [DENSITY_LOW, DENSITY_HIGH],
            "arms": list(ARMS), "steps_per_episode": STEPS_PER_EPISODE,
            "p0_episodes": P0_EPISODES, "p1_episodes": P1_EPISODES,
            "zworld_p0_episodes": ZWORLD_P0_EPISODES,
            "cal_episodes": CAL_EPISODES, "meas_episodes": MEAS_EPISODES,
            "episodes_per_run": EPISODES_PER_RUN,
            "operating_point": manifest["prereg_operating_point"],
        },
        seeds=SEEDS, script_path=Path(__file__), started_at=_t_start,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)

    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
