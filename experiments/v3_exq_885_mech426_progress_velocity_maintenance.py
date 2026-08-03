#!/opt/local/bin/python3
"""
V3-EXQ-885 -- MECH-426 progress-velocity goal maintenance: 2x2 ablation of the
rate-of-progress signal crossed with confirmation density.

=======================================================================
!! BLOCKED -- NOT QUEUED. DO NOT QUEUE THIS WITHOUT RE-RUNNING THE GATE !!
=======================================================================
Authored + smoke-tested 2026-08-02/03 and then WITHHELD from the queue: the
design is sound and the SD-093 substrate it targets is complete, but the
BEHAVIOURAL substrate this design needs does not yet exist. Measured here, not
assumed:

  * A bare (P0a-warmed but policy-untrained) REEAgent on this env is
    BEHAVIOURALLY FROZEN. Over 300 steps it selected action 4 (STAY) 300/300
    times and visited 1 unique cell. Invariant to select_action temperature
    (0.5 / 1.0 / 2.0 all give 300/300 STAY), so it is not a sampling-sharpness
    issue.
  * Consequently 6 x 500-step episodes (3000 steps) produced ZERO resource
    confirmations, so the measurement window never opened, z_goal never
    activated, and every DV read exactly 0.0. A RANDOM walk on the same env
    reaches resources readily (2 events / 176 steps SPARSE; 11 / 232 DENSE) --
    the agent is worse than random, not merely weak.
  * Independently corroborated one day earlier by V3-EXQ-866
    (v3_exq_866_inv034_q021_goal_maintenance_agency_20260802T074409Z_v3,
    outcome FAIL): resource_visit_rate FULL 0.0046 vs RANDOM 0.0217 -- a
    trained agent foraging ~4.7x WORSE than random. This is the
    goal_pipeline:GAP-2 foraging / benefit-contact ceiling.

WHY THIS CANNOT BE WORKED AROUND IN THE DRIVER. Every acceptance criterion
below is behavioural (goal completion rate, z_goal retention across
inter-confirmation gaps). Supplying the trajectory exogenously -- a scripted
walk, as V3-EXQ-883 legitimately does for a credit-propagation primitive --
would decouple the commit gate from behaviour entirely, leaving only the commit
gate's OWN output as a DV. That is tautological: SD-093 multiplies the commit
threshold, so of course the commit rate moves, and the 18 contract tests in
tests/contracts/test_mech426_progress_velocity.py already establish that much.
It would produce a confident-looking PASS that is an arithmetic identity, which
is the DV-symmetry failure class this file's own declaration (below) exists to
refuse.

RESUME CONDITION. An agent that forages ABOVE random on this env family. The
live owner of that question is V3-EXQ-866a
(v3_exq_866a_inv034_q021_goal_maintenance_agency_onboarded.py), queued and not
yet run, which retests 866 on the scaffolded_sd054_onboarding curriculum
(substrate_queue: ready=True, "P1 survival 3/3, P2 contact 3/3" on V3-EXQ-603m).
On a 866a result establishing above-random goal-directed foraging, rebuild this
driver's cells on that onboarded agent and re-run the readiness probe before
queuing. If 866a does not establish it, this proposal stays blocked on
goal_pipeline:GAP-2 and the route is /implement-substrate, not another
/queue-experiment letter.

Everything below is the design as authored, retained because it is what a
successor needs: the regime construction, the readiness gates, the DV-symmetry
analysis, and one env finding worth keeping (see CONTAMINATED_HARM in
_lib/baselines/exq885_mech426_velocity_baseline.py -- num_hazards=0 does NOT
give a long horizon; the agent's own vacated-cell contamination kills episodes
at ~29 steps).
=======================================================================

Claim: MECH-426 (goal.progress_velocity_maintenance). Proposal: EXP-0384.
Substrate: SD-093, landed 2026-08-02 (this is the first experiment on it).

QUESTION. Does a rate-of-progress (velocity) signal -- the temporal derivative
of an on-path bootstrapped progress estimate -- sustain superordinate-goal
commitment BETWEEN sparse terminal confirmations? And does ablating it degrade
long-horizon maintenance SPECIFICALLY in the sparse-confirmation regime?

DESIGN. 2x2 x 3 seeds = 12 cells.
  Factor 1 VELOCITY: `goal.use_progress_velocity_effort_modulation` ON vs OFF.
    OFF is the SD-093 master switch's default and is bit-identical to the stock
    substrate (contract C2/C9): record_progress() no-ops, the modulation property
    returns exactly 0.0, and E3's modulation branch is never entered. The goal
    representation and the proxy-wanting landscape are untouched in both arms --
    `compute_goal_score()` / `goal_proximity()` are verified byte-identical
    regardless of the flag -- so this ablates the velocity READOUT alone, which
    is what the registered design asks for.
  Factor 2 CONFIRMATION: SPARSE (num_resources=2) vs DENSE (num_resources=8),
    both with resource_respawn_on_consume=True. A "terminal confirmation" is a
    resource CONSUMPTION event, because that is exactly what re-pulls z_goal
    toward z_world: GoalState.update() opens its benefit gate only above
    benefit_threshold (0.1), which the env's resource_benefit (0.3) clears and
    the continuous proximity_benefit_scale (0.03) does not.

WM-DECAY REGIME (the acceptance check's non-degeneracy (a)). z_goal decays
unconditionally at decay_goal=0.005 per update() tick, so across a 500-step
episode an unrefreshed attractor falls to ~8% of peak. That is the regime
INV-086 is scoped to, and it is why the sparse arm is a maintenance test rather
than a formality. It is NOT assumed: R5 below MEASURES that the SPARSE OFF arm
retains less than the DENSE OFF arm, and a failure self-routes
substrate_not_ready_requeue rather than a MECH-426 verdict.

DIRECTION (the registered caveat, and the trap it names). Carver & Scheier's
velocity loop is EFFORT REGULATION, not same-goal reinforcement: above-reference
progress produces positive affect that REDUCES effort and licenses redeployment
(coasting). SD-093 implements exactly that sign -- positive velocity LOWERS
E3's effective commit threshold (stricter bar, more readily kicked back into
deliberation), negative velocity RAISES it (lock in and push through). This
driver does not touch that sign; it reads PASS/FAIL in the effort-regulation
direction. The prediction follows from the sign, not from a bonus story: in the
SPARSE regime the agent spends most of its time NOT closing on the goal, so the
modulation sits positive and sustains commitment across the long gaps between
confirmations. In DENSE, confirmations do that work themselves and velocity adds
no information -- hence the interaction, not a main effect.

DV-SYMMETRY INVARIANCE DECLARATION (mandatory, per arm -- the V3-EXQ-604c class).
  The DV's symmetry group. The primary DV chain is a THRESHOLD CROSSING,
  `committed = commit_variance < effective_threshold`, and its behavioural
  downstream (commit-run length, z_goal retention, confirmation rate). A
  threshold comparison is invariant under a COMMON monotone rescaling of BOTH
  sides, and under any transform of quantities it does not read.
  The manipulation is not invariant under it. SD-093 multiplies ONE side --
  `effective_threshold *= (1 + velocity_effort)` -- leaving `commit_variance`
  untouched, so it moves the two sides relative to each other and can flip the
  comparison. It is therefore NOT the 604c failure shape: that was a BROADCAST
  SCALAR added uniformly across candidates and read by an ARGMAX, which cancels
  by arithmetic. This modulation never enters `scores` at all, so it provably
  cannot move the argmin over candidates -- and correspondingly this experiment
  does not use any candidate-selection DV.
  The residual risk that follows from exactly that reading, and how it is
  gated. Because the modulation touches only the commit gate, an arm in which
  the gate is SATURATED (commit_variance always far below, or far above, the
  threshold) is structurally inert no matter how large the modulation is. That
  is not detectable from the modulation's own magnitude OR its range, so R2
  measures the thing itself: the fraction of genuine E3 selections at which the
  commit decision actually DIFFERS between the modulated and the counterfactual
  unmodulated threshold. Below floor -> substrate_not_ready_requeue.
  Per arm: VEL_ON_SPARSE and VEL_ON_DENSE carry the manipulation and are gated
  by R1/R2 above. VEL_OFF_SPARSE and VEL_OFF_DENSE carry no manipulation by
  construction (modulation identically 0.0), so R1/R2 are scoped OUT of them --
  disposition (a), "not meaningful for the regime", never a failure.

READINESS GATES (all measured, none assumed; worst cell across seeds reported).
  R1 velocity_effort_RANGE > 1e-4                  [ON arms; floor]
     RANGE, not mean-abs: the load-bearing criteria route on a cross-arm
     DIFFERENCE produced by a signal that must actually VARY. A modulation
     pinned at one constant value is a uniform re-scaling of the threshold for
     the whole run, which is a different (and untested) manipulation. This is
     the V3-EXQ-643 same-statistic rule; velocity_effort_mean_abs is recorded
     alongside as a diagnostic precisely so the two can be compared, but it
     gates nothing.
  R2 commit_flip_frac > 0.01                        [ON arms; floor]
     see the DV-symmetry note above -- the manipulation must be able to move
     the DV, not merely be non-constant.
  R3 goal_active_frac > 0.2                         [all arms; floor]
     z_goal must be live; E3's modulation branch is guarded on
     goal_state.is_active().
  R4 frac_episodes_with_confirmation > 0.5          [all arms; floor]
     the measurement window opens at an episode's first confirmation; if the
     sparse regime rarely confirms, there is no window.
  R5 wm_decay_regime_delta < -0.02                  [SPARSE arms; CEILING]
     retention(SPARSE,OFF) - retention(DENSE,OFF), i.e. the bare-goal control
     must itself show MORE maintenance decay in the sparse regime. Declared
     direction "upper" because it must sit BELOW its threshold.

PRE-REGISTERED ACCEPTANCE CRITERIA (thresholds are constants below, fixed
before the run; nothing here is derived from the run's own statistics).
  d_sparse(seed) = goal_retention[ON,SPARSE] - goal_retention[OFF,SPARSE]
  d_dense(seed)  = goal_retention[ON,DENSE]  - goal_retention[OFF,DENSE]
  C1 [LOAD-BEARING] d_sparse > MIN_EFFECT (0.02) on >= 2/3 seeds.
  C2 [LOAD-BEARING] d_dense < SPECIFICITY_FRAC (0.5) * d_sparse on >= 2/3 seeds
     -- "no comparable drop in the dense regime".
  C3 [reported, NOT gated] the same interaction sign on the completion-rate DV
     (confirmations_per_1k_steps).
  COMBINATION RULE: PASS = gate_green(all four arms) AND C1 AND C2. A plain AND
  over the two load-bearing criteria; C3 is recorded for the autopsy trail and
  never rescues or vetoes a verdict. Stated explicitly (and emitted as
  `combination_rule`) so the per-criterion list cannot be misread downstream.

ROUTING.
  any arm gate red      -> FAIL / non_contributory / substrate_not_ready_requeue,
                           non_degenerate=false. NOT a MECH-426 verdict.
  gates green, C1 & C2  -> PASS / supports.
  gates green, otherwise-> FAIL / weakens (a genuine null: the signal was live,
                           could move the DV, the regime did decay, and ablating
                           it still did not degrade sparse-regime maintenance).

WHAT A NULL WOULD AND WOULD NOT MEAN. A green-gated FAIL means the velocity
readout is not load-bearing for maintenance AT THIS GAIN, on this env family,
with z_goal retention as the maintenance readout. It would not show that no
rate-of-progress signal can serve this role, nor speak to the on-path estimate
itself (MECH-426 folds `on_path_progress_inference`; this ablates the
DERIVATIVE readout, holding the estimate intact by design).

NO POLICY LEARNING. See `_lib/baselines/exq885_mech426_velocity_baseline.py`
`run_cell` for the rationale: the mechanism is not learned, so a learning phase
would let the contrast ride on divergent learning instead of on maintenance
dynamics. P0a DOES train the z_world encoder (SD-070 recipe, RNG-neutral),
because goal_proximity() is a z_world distance and an untrained encoder would
make "progress estimate" an overclaim.

SLEEP DRIVER: N/A -- no sleep loop, SWS, REM or sleep-aggregation flag is set.

ethics_preflight:
  involves_negative_valence: false          # num_hazards=0; no harm stream fed
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.baselines import exq885_mech426_velocity_baseline as B  # noqa: E402

# --------------------------------------------------------------------- #
# Experiment metadata
# --------------------------------------------------------------------- #
EXPERIMENT_TYPE = "v3_exq_885_mech426_progress_velocity_maintenance"
QUEUE_ID = "V3-EXQ-885"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
# Only MECH-426 is DIRECTLY tested: the ablation toggles exactly the SD-093
# velocity readout. INV-086 / INV-034 / MECH-217 / MECH-116 are the proposal's
# related claims (regime scoping, wanting landscape, WM maintenance) and are
# discussed above, but this implementation does not isolate any of them, so
# tagging them would corrupt their confidence scores.
CLAIM_IDS = ["MECH-426"]
EXPERIMENT_PURPOSE = "evidence"

# Seed 44 is deliberately skipped: a per-seed early-episode instability on this
# env family is documented across two independent autopsies (EXQ-539-540,
# V3-EXQ-538a). Substituting 45 costs nothing.
SEEDS = [42, 43, 45]

# --- SD-093 ON-arm knobs (OFF arms short-circuit ahead of all three) --------
# The substrate defaults are window=5, gain=1.0, max=0.3. The gain is an
# instrument SCALE, not an acceptance threshold: goal_proximity is 1/(1+MSE_sum)
# and its per-window derivative is small in absolute terms, so gain=1.0 would
# leave the modulation far inside the +-0.3 cap and very likely inert at the
# commit gate. GAIN IS CALIBRATED FROM THE SMOKE RUN's measured R1/R2, before
# any acceptance threshold is read -- never adjusted afterwards to obtain a
# PASS. If the calibrated gain still cannot clear R2, that is a genuine
# substrate_not_ready_requeue, not a knob to keep turning.
PROGRESS_VELOCITY_EFFORT_GAIN = 10.0
PROGRESS_VELOCITY_WINDOW = 5
PROGRESS_VELOCITY_EFFORT_MAX = 0.3

# --- Pre-registered readiness floors / ceiling ------------------------------
VELOCITY_EFFORT_RANGE_FLOOR = 1e-4
COMMIT_FLIP_FRAC_FLOOR = 0.01
GOAL_ACTIVE_FRAC_FLOOR = 0.2
EPISODE_CONFIRMATION_FRAC_FLOOR = 0.5
WM_DECAY_REGIME_DELTA_CEIL = -0.02

# --- Pre-registered acceptance thresholds ----------------------------------
MIN_EFFECT = 0.02          # C1: minimum sparse-regime retention lift
SPECIFICITY_FRAC = 0.5     # C2: dense delta must be under half the sparse delta
MIN_SEEDS_PASS = 2         # of 3 -- ">= 2/3 seeds"

# --- Arms -------------------------------------------------------------------
ARMS: List[Dict[str, Any]] = [
    {"id": "VEL_ON_SPARSE", "velocity_on": True, "regime": "SPARSE"},
    {"id": "VEL_OFF_SPARSE", "velocity_on": False, "regime": "SPARSE"},
    {"id": "VEL_ON_DENSE", "velocity_on": True, "regime": "DENSE"},
    {"id": "VEL_OFF_DENSE", "velocity_on": False, "regime": "DENSE"},
]
ARM_IDS = [a["id"] for a in ARMS]


# --------------------------------------------------------------------- #
# Readiness specs
# --------------------------------------------------------------------- #

def _is_on(ctx: Dict[str, Any]) -> bool:
    return bool(ctx["velocity_on"])


def _is_sparse(ctx: Dict[str, Any]) -> bool:
    return ctx["regime"] == "SPARSE"


SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="velocity_effort_range",
        description=(
            "max-min of progress_velocity_effort_modulation across genuine E3 "
            "selections in the measurement window. RANGE, not magnitude: the "
            "load-bearing criteria route on a difference produced by a signal "
            "that must actually vary (V3-EXQ-643 same-statistic rule)."),
        control=(
            "measured in the ON arms, where the SD-093 master switch is on and "
            "the modulation is by construction the only cross-arm difference"),
        threshold=VELOCITY_EFFORT_RANGE_FLOOR,
        direction="lower",
        applies_to=_is_on,
        applies_note=(
            "OFF arms: progress_velocity_effort_modulation returns exactly 0.0 "
            "with the master switch off (contract C2), so the range is "
            "structurally 0 and the precondition is not meaningful for them -- "
            "disposition (a), scoped out, not failed."),
    ),
    PreconditionSpec(
        name="commit_flip_frac",
        description=(
            "fraction of genuine E3 selections where committed differs between "
            "the modulated threshold and the counterfactual unmodulated one "
            "(eff_thr / (1 + velocity_effort)). Guards the saturated-commit-gate "
            "degeneracy: a large modulation on a gate whose variance never "
            "approaches the threshold moves nothing."),
        control=(
            "counterfactual computed per tick from the same commit_variance, so "
            "the two thresholds differ only by the SD-093 factor"),
        threshold=COMMIT_FLIP_FRAC_FLOOR,
        direction="lower",
        applies_to=_is_on,
        applies_note=(
            "OFF arms: the modulation factor is identically 1.0, so the "
            "modulated and unmodulated thresholds are the same number and the "
            "flip fraction is structurally 0 -- scoped out, not failed."),
    ),
    PreconditionSpec(
        name="goal_active_frac",
        description=(
            "fraction of measured E3 ticks with goal_state.is_active(). E3's "
            "modulation branch is guarded on it, and there is nothing to "
            "maintain when z_goal is dead."),
        control="measured over the post-first-confirmation window in every arm",
        threshold=GOAL_ACTIVE_FRAC_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="frac_episodes_with_confirmation",
        description=(
            "fraction of P2 episodes reaching a first confirmation, which is "
            "what opens the measurement window."),
        control="measured per arm; the sparse regime is the binding case",
        threshold=EPISODE_CONFIRMATION_FRAC_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="wm_decay_regime_delta",
        description=(
            "goal_retention(SPARSE,OFF) - goal_retention(DENSE,OFF). The "
            "acceptance check's non-degeneracy (a): the BARE-GOAL control must "
            "itself show more maintenance decay in the chosen sparse regime, "
            "else the ablation is tested where there is nothing to sustain."),
        control=(
            "the two OFF arms are the bare-goal control -- stock substrate, "
            "identical but for num_resources"),
        threshold=WM_DECAY_REGIME_DELTA_CEIL,
        direction="upper",
        applies_to=_is_sparse,
        applies_note=(
            "a property of the SPARSE regime's adequacy as a decay regime; "
            "recorded against the sparse arms, not the dense reference."),
    ),
]


# --------------------------------------------------------------------- #
# Worst-cell reducers (measured must be the SAME statistic `met` tests)
# --------------------------------------------------------------------- #

def _worst_floor(rows: List[Dict[str, Any]], key: str) -> Tuple[float, Any]:
    """Minimum across seeds (the worst cell for a FLOOR), plus its seed."""
    worst = min(rows, key=lambda r: float(r[key]))
    return float(worst[key]), worst["seed"]


# --------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------- #

def run(dry_run: bool = False) -> Tuple[Dict[str, Any], ZGoalStreamAccumulator]:
    print(f"\n[{QUEUE_ID}] MECH-426 progress-velocity goal maintenance "
          f"(VELOCITY ON/OFF x SPARSE/DENSE confirmation)", flush=True)

    arm_contexts = [
        {"id": a["id"], "velocity_on": a["velocity_on"], "regime": a["regime"]}
        for a in ARMS
    ]
    # Design-time refusal: no applicable precondition may be unsatisfiable from
    # the pre-registered config. Runs BEFORE any compute is spent (V3-EXQ-785).
    assert_no_structurally_unsatisfiable_gate(SPECS, arm_contexts)

    p0a_eps = B.P0A_EPISODES_DRY if dry_run else B.P0A_EPISODES
    n_p2 = B.N_EPISODES_P2_DRY if dry_run else B.N_EPISODES_P2
    episodes_per_run = p0a_eps + n_p2

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict[str, Any]] = []
    rows_by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARM_IDS}

    for seed in SEEDS:
        for arm in ARMS:
            arm_id = arm["id"]
            velocity_on = arm["velocity_on"]
            config_slice = B.arm_config_slice(
                velocity_on=velocity_on,
                regime=arm["regime"],
                dry_run=dry_run,
                velocity_gain=PROGRESS_VELOCITY_EFFORT_GAIN,
                velocity_window=PROGRESS_VELOCITY_WINDOW,
                velocity_max=PROGRESS_VELOCITY_EFFORT_MAX,
            )
            # MINT AS YOU GO: include_driver_script_in_hash=False on EVERY cell
            # so a later, different-driver sibling can cache-HIT the OFF cells
            # banked here. No extra_ineligible_reasons -- each cell builds a
            # fresh env and a fresh agent and shares no mutable state with any
            # other cell, so the cells are genuinely independent.
            with arm_cell(
                seed,
                config_slice=config_slice,
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                row = B.run_cell(
                    velocity_on=velocity_on,
                    regime=arm["regime"],
                    seed=seed,
                    velocity_gain=PROGRESS_VELOCITY_EFFORT_GAIN,
                    velocity_window=PROGRESS_VELOCITY_WINDOW,
                    velocity_max=PROGRESS_VELOCITY_EFFORT_MAX,
                    zg=zg,
                    dry_run=dry_run,
                    arm_label=arm_id,
                    episode_counter_base=0,
                    episodes_per_run=episodes_per_run,
                )
                cell.stamp(row)
            arm_results.append(row)
            rows_by_arm[arm_id].append(row)
            print("verdict: PASS", flush=True)  # cell completed; science is aggregate-level

    # ----------------------------------------------------------------- #
    # Readiness gates (worst cell across seeds)
    # ----------------------------------------------------------------- #
    def _retention_mean(arm_id: str) -> float:
        vals = [float(r["goal_retention"]) for r in rows_by_arm[arm_id]]
        return sum(vals) / len(vals)

    wm_decay_regime_delta = (
        _retention_mean("VEL_OFF_SPARSE") - _retention_mean("VEL_OFF_DENSE")
    )

    arm_gates = []
    offending_cells: Dict[str, Any] = {}
    for ctx in arm_contexts:
        arm_id = ctx["id"]
        rows = rows_by_arm[arm_id]
        measured: Dict[str, float] = {}
        if _is_on(ctx):
            v, s = _worst_floor(rows, "velocity_effort_range")
            measured["velocity_effort_range"] = v
            offending_cells[f"{arm_id}::velocity_effort_range"] = s
            v, s = _worst_floor(rows, "commit_flip_frac")
            measured["commit_flip_frac"] = v
            offending_cells[f"{arm_id}::commit_flip_frac"] = s
        v, s = _worst_floor(rows, "goal_active_frac")
        measured["goal_active_frac"] = v
        offending_cells[f"{arm_id}::goal_active_frac"] = s
        v, s = _worst_floor(rows, "frac_episodes_with_confirmation")
        measured["frac_episodes_with_confirmation"] = v
        offending_cells[f"{arm_id}::frac_episodes_with_confirmation"] = s
        if _is_sparse(ctx):
            # A cross-arm aggregate, identical for both sparse arms by
            # construction; no per-seed worst cell exists for it.
            measured["wm_decay_regime_delta"] = wm_decay_regime_delta
        arm_gates.append(evaluate_arm_gate(arm_id, ctx, SPECS, measured=measured))

    gate = aggregate_arm_gates(arm_gates)
    green_arms = {g["arm"] for g in arm_gates if g["gate_green"]}
    # The contrast needs BOTH arms of a regime AND both regimes, so this run's
    # scoring rule is stricter than aggregate_arm_gates' any-arm-green default:
    # a red arm cannot be scored around here, because every criterion is a
    # cross-arm difference. The per-arm block is still emitted in full so a
    # partial failure stays attributable rather than collapsing to one flag.
    all_gates_green = len(green_arms) == len(ARM_IDS)

    # ----------------------------------------------------------------- #
    # Pre-registered criteria
    # ----------------------------------------------------------------- #
    def _by_seed(arm_id: str, seed: int, key: str) -> float:
        return float(next(r for r in rows_by_arm[arm_id] if r["seed"] == seed)[key])

    d_sparse: Dict[int, float] = {}
    d_dense: Dict[int, float] = {}
    d_sparse_completion: Dict[int, float] = {}
    d_dense_completion: Dict[int, float] = {}
    for seed in SEEDS:
        d_sparse[seed] = (
            _by_seed("VEL_ON_SPARSE", seed, "goal_retention")
            - _by_seed("VEL_OFF_SPARSE", seed, "goal_retention")
        )
        d_dense[seed] = (
            _by_seed("VEL_ON_DENSE", seed, "goal_retention")
            - _by_seed("VEL_OFF_DENSE", seed, "goal_retention")
        )
        d_sparse_completion[seed] = (
            _by_seed("VEL_ON_SPARSE", seed, "confirmations_per_1k_steps")
            - _by_seed("VEL_OFF_SPARSE", seed, "confirmations_per_1k_steps")
        )
        d_dense_completion[seed] = (
            _by_seed("VEL_ON_DENSE", seed, "confirmations_per_1k_steps")
            - _by_seed("VEL_OFF_DENSE", seed, "confirmations_per_1k_steps")
        )

    c1_per_seed = [d_sparse[s] > MIN_EFFECT for s in SEEDS]
    c2_per_seed = [d_dense[s] < SPECIFICITY_FRAC * d_sparse[s] for s in SEEDS]
    c3_per_seed = [
        (d_sparse_completion[s] > 0.0)
        and (d_dense_completion[s] < SPECIFICITY_FRAC * d_sparse_completion[s])
        for s in SEEDS
    ]
    frac_threshold = MIN_SEEDS_PASS / len(SEEDS)
    c1_frac = sum(1 for c in c1_per_seed if c) / len(SEEDS)
    c2_frac = sum(1 for c in c2_per_seed if c) / len(SEEDS)
    c3_frac = sum(1 for c in c3_per_seed if c) / len(SEEDS)
    c1_pass = c1_frac >= frac_threshold
    c2_pass = c2_frac >= frac_threshold
    c3_pass = c3_frac >= frac_threshold

    combination_rule = (
        "PASS = all_gates_green AND C1 AND C2 (plain AND over the two "
        "LOAD-BEARING criteria). C3 (completion-rate interaction) is recorded "
        "and reported but is NOT part of the gate -- it can neither rescue nor "
        "veto the verdict."
    )

    # ----------------------------------------------------------------- #
    # Routing
    # ----------------------------------------------------------------- #
    non_degenerate = True
    degeneracy_reason = None
    if not all_gates_green:
        status = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        red = sorted(set(ARM_IDS) - green_arms)
        failed = {g["arm"]: g["failed_preconditions"] for g in arm_gates
                  if not g["gate_green"]}
        degeneracy_reason = (
            f"readiness gate red on arm(s) {red}: {failed}. Every criterion in "
            f"this design is a cross-arm difference, so a red arm makes its "
            f"regime's contrast unscoreable. This is a statement about the "
            f"instrument (signal range, commit-gate authority, goal liveness, "
            f"confirmation reachability, or the regime's decay adequacy), NOT "
            f"evidence about MECH-426."
        )
        interpretation_text = (
            "SUBSTRATE NOT READY -- re-queue. " + degeneracy_reason + " Route: "
            "if velocity_effort_range or commit_flip_frac is the failure, "
            "recalibrate PROGRESS_VELOCITY_EFFORT_GAIN from the measured "
            "distribution and re-queue as V3-EXQ-885a; if "
            "frac_episodes_with_confirmation or goal_active_frac is the "
            "failure, the sparse regime is too sparse to seed z_goal at all "
            "and num_resources / steps need re-tuning; if "
            "wm_decay_regime_delta is the failure, the sparse and dense "
            "regimes did not actually differ in maintenance decay and the "
            "regime contrast itself needs widening."
        )
    else:
        all_pass = c1_pass and c2_pass
        status = "PASS" if all_pass else "FAIL"
        evidence_direction = "supports" if all_pass else "weakens"
        label = ("velocity_sustains_sparse_regime_maintenance" if all_pass
                 else "velocity_not_load_bearing_for_maintenance")
        if all_pass:
            interpretation_text = (
                "MECH-426 SUPPORTED. Ablating the SD-093 rate-of-progress "
                "readout (holding the goal representation and the proxy-wanting "
                "landscape intact -- compute_goal_score/goal_proximity are "
                "byte-identical either way) degraded long-horizon z_goal "
                "maintenance in the SPARSE-confirmation regime and did not "
                "produce a comparable drop in the DENSE regime, where "
                "confirmations already carry the maintenance load and velocity "
                "adds no information. Read in the Carver & Scheier effort-"
                "regulation direction: the substrate raises E3's commit "
                "threshold when progress stalls, sustaining commitment across "
                "the long gaps between terminal confirmations. The interaction "
                "-- not a main effect -- is what distinguishes this from a "
                "generic commit-threshold perturbation."
            )
        else:
            interpretation_text = (
                "MECH-426 WEAKENED. The readiness gates were green in every arm "
                "-- the velocity signal varied, could flip commit decisions, "
                "z_goal was live, and the sparse regime did show more decay "
                "than the dense one -- so this is a genuine null, not a "
                "starved test. Ablating the velocity readout did not degrade "
                "sparse-regime maintenance on the pre-registered criteria "
                f"(C1 {c1_frac:.2f}, C2 {c2_frac:.2f} of seeds). Scope: at "
                f"gain {PROGRESS_VELOCITY_EFFORT_GAIN}, on this env family, "
                "with z_goal retention as the maintenance readout. It does not "
                "speak to the on-path progress ESTIMATE itself, only to the "
                "derivative readout taken over it."
            )

    # ----------------------------------------------------------------- #
    # Manifest
    # ----------------------------------------------------------------- #
    criteria = [
        {"name": "C1_sparse_regime_retention_lift", "load_bearing": True,
         "passed": bool(c1_pass), "frac_seeds": c1_frac,
         "rule": f"goal_retention(ON,SPARSE) - goal_retention(OFF,SPARSE) > {MIN_EFFECT}"},
        {"name": "C2_dense_regime_specificity", "load_bearing": True,
         "passed": bool(c2_pass), "frac_seeds": c2_frac,
         "rule": f"d_dense < {SPECIFICITY_FRAC} * d_sparse"},
        {"name": "C3_completion_rate_interaction", "load_bearing": False,
         "passed": bool(c3_pass), "frac_seeds": c3_frac,
         "rule": "same interaction sign on confirmations_per_1k_steps"},
    ]
    criteria_non_degenerate = arm_criteria_non_degenerate(
        {
            "VEL_ON_SPARSE": ["C1_sparse_regime_retention_lift"],
            "VEL_OFF_SPARSE": ["C1_sparse_regime_retention_lift"],
            "VEL_ON_DENSE": ["C2_dense_regime_specificity",
                             "C3_completion_rate_interaction"],
            "VEL_OFF_DENSE": ["C2_dense_regime_specificity",
                              "C3_completion_rate_interaction"],
        },
        gate,
    )

    metrics: Dict[str, float] = {
        "c1_frac_seeds": c1_frac,
        "c2_frac_seeds": c2_frac,
        "c3_frac_seeds": c3_frac,
        "c1_pass": 1.0 if c1_pass else 0.0,
        "c2_pass": 1.0 if c2_pass else 0.0,
        "c3_pass": 1.0 if c3_pass else 0.0,
        "all_gates_green": 1.0 if all_gates_green else 0.0,
        "wm_decay_regime_delta": wm_decay_regime_delta,
        "d_sparse_mean": sum(d_sparse.values()) / len(SEEDS),
        "d_dense_mean": sum(d_dense.values()) / len(SEEDS),
        "d_sparse_completion_mean": sum(d_sparse_completion.values()) / len(SEEDS),
        "d_dense_completion_mean": sum(d_dense_completion.values()) / len(SEEDS),
    }
    for arm_id in ARM_IDS:
        rows = rows_by_arm[arm_id]
        for key in ("goal_retention", "commit_run_len_mean",
                    "confirmations_per_1k_steps", "committed_frac",
                    "velocity_effort_range", "velocity_effort_mean_abs",
                    "commit_flip_frac", "goal_active_frac",
                    "frac_episodes_with_confirmation"):
            metrics[f"{key}_mean_{arm_id}"] = (
                sum(float(r[key]) for r in rows) / len(rows)
            )
        metrics[f"n_ticks_measured_total_{arm_id}"] = float(
            sum(int(r["n_ticks_measured"]) for r in rows))
        metrics[f"n_latched_ticks_total_{arm_id}"] = float(
            sum(int(r["n_latched_ticks"]) for r in rows))

    summary_markdown = (
        f"# {QUEUE_ID} -- MECH-426 progress-velocity goal maintenance\n\n"
        f"**Status:** {status}  **Evidence direction:** {evidence_direction}\n"
        f"**Label:** {label}\n"
        f"**Claims:** MECH-426 (substrate SD-093)\n\n"
        f"## Readiness gates\n\n"
        f"| Arm | green | failed |\n|---|---|---|\n"
        + "".join(
            f"| {g['arm']} | {g['gate_green']} | {g['failed_preconditions']} |\n"
            for g in arm_gates
        )
        + f"\nwm_decay_regime_delta = {wm_decay_regime_delta:.4f} "
          f"(ceiling {WM_DECAY_REGIME_DELTA_CEIL})\n\n"
        f"## Criteria\n\n"
        f"| Criterion | load-bearing | frac seeds | pass |\n|---|---|---|---|\n"
        + "".join(
            f"| {c['name']} | {c['load_bearing']} | {c['frac_seeds']:.2f} | "
            f"{c['passed']} |\n" for c in criteria
        )
        + f"\n**Combination rule:** {combination_rule}\n\n"
        f"## Effects (goal_retention)\n\n"
        f"d_sparse mean = {metrics['d_sparse_mean']:.4f}, "
        f"d_dense mean = {metrics['d_dense_mean']:.4f}\n\n"
        f"## Interpretation\n\n{interpretation_text}\n"
    )

    result: Dict[str, Any] = {
        "status": status,
        "outcome": status,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-426": evidence_direction},
        "metrics": metrics,
        "arm_results": arm_results,
        "per_seed_results": {
            arm_id: {r["seed"]: r for r in rows_by_arm[arm_id]}
            for arm_id in ARM_IDS
        },
        "per_seed_deltas": {
            "d_sparse": d_sparse,
            "d_dense": d_dense,
            "d_sparse_completion": d_sparse_completion,
            "d_dense_completion": d_dense_completion,
        },
        "combination_rule": combination_rule,
        "per_arm_gate": gate.get("per_arm_gate", gate),
        "interpretation": {
            "label": label,
            # Green arms only, per aggregate_arm_gates: the indexer reads this
            # list flat and arm-blind and would otherwise let one red arm vacate
            # the whole run (the V3-EXQ-785 adjudication-time vacating).
            "preconditions": gate.get("adjudication_preconditions",
                                      gate.get("preconditions", [])),
            "preconditions_scope_note": gate.get("preconditions_scope_note", ""),
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "combination_rule": combination_rule,
            "text": interpretation_text,
        },
        "diagnostics": {
            "offending_cells": offending_cells,
            "arm_gates": arm_gates,
            "wm_decay_regime_delta": wm_decay_regime_delta,
        },
        "custom_information": {
            "dv_symmetry_declaration": {
                "dv_symmetry_group": (
                    "threshold crossing (commit_variance < effective_threshold) "
                    "and its behavioural downstream -- invariant under a COMMON "
                    "monotone rescaling of both sides"),
                "manipulation_invariant_under_it": False,
                "why": (
                    "SD-093 rescales only effective_threshold, leaving "
                    "commit_variance untouched, so the two sides move relative "
                    "to each other. It never enters `scores`, so it provably "
                    "cannot move the argmin over candidates -- and no "
                    "candidate-selection DV is used."),
                "per_arm": {
                    "VEL_ON_SPARSE": "manipulation present; not invariant (above)",
                    "VEL_ON_DENSE": "manipulation present; not invariant (above)",
                    "VEL_OFF_SPARSE": "no manipulation by construction (control)",
                    "VEL_OFF_DENSE": "no manipulation by construction (control)",
                },
                "residual_risk_gated_by": "commit_flip_frac (R2)",
            },
            "sd093_config": {
                "progress_velocity_effort_gain": PROGRESS_VELOCITY_EFFORT_GAIN,
                "progress_velocity_window": PROGRESS_VELOCITY_WINDOW,
                "progress_velocity_effort_max": PROGRESS_VELOCITY_EFFORT_MAX,
            },
            "gov_reuse_1_check": (
                "Decisive readout = goal_retention under "
                "use_progress_velocity_effort_modulation ON vs OFF. Checked "
                "2026-08-02 with reanalysis_query over evidence/experiments/: "
                "807 manifests scanned, 0 carry any 'progress_velocity' readout "
                "(--require-readout), and 0 tag MECH-426 at all (--claim "
                "MECH-426 matched nothing). A raw grep for "
                "'use_progress_velocity_effort_modulation' across the whole "
                "evidence tree also returns nothing. Consistent with SD-093 "
                "having landed 2026-08-02 defaulting False -- no recorded run "
                "can contain a velocity-ON cell. NOT RECOVERABLE -> run."),
        },
        "sleep_driver_pattern": "N/A (no sleep loop)",
        "fatal_error_count": 0,
    }
    if not non_degenerate:
        result["non_degenerate"] = False
        result["degeneracy_reason"] = degeneracy_reason

    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = ARCHITECTURE_EPOCH

    full_config = {
        "seeds": SEEDS,
        "arms": ARM_IDS,
        "grid_size": B.GRID_SIZE,
        "num_hazards": B.NUM_HAZARDS,
        "env_regimes": B.ENV_REGIMES,
        "resource_respawn_on_consume": True,
        "world_dim": B.WORLD_DIM,
        "steps_per_ep": B.STEPS_PER_EP_DRY if args.dry_run else B.STEPS_PER_EP,
        "n_episodes_p2": B.N_EPISODES_P2_DRY if args.dry_run else B.N_EPISODES_P2,
        "p0a_episodes": B.P0A_EPISODES_DRY if args.dry_run else B.P0A_EPISODES,
        "p0a_steps_per_ep": (
            B.P0A_STEPS_PER_EP_DRY if args.dry_run else B.P0A_STEPS_PER_EP),
        "progress_velocity_effort_gain": PROGRESS_VELOCITY_EFFORT_GAIN,
        "progress_velocity_window": PROGRESS_VELOCITY_WINDOW,
        "progress_velocity_effort_max": PROGRESS_VELOCITY_EFFORT_MAX,
        "velocity_effort_range_floor": VELOCITY_EFFORT_RANGE_FLOOR,
        "commit_flip_frac_floor": COMMIT_FLIP_FRAC_FLOOR,
        "goal_active_frac_floor": GOAL_ACTIVE_FRAC_FLOOR,
        "episode_confirmation_frac_floor": EPISODE_CONFIRMATION_FRAC_FLOOR,
        "wm_decay_regime_delta_ceil": WM_DECAY_REGIME_DELTA_CEIL,
        "min_effect": MIN_EFFECT,
        "specificity_frac": SPECIFICITY_FRAC,
        "min_seeds_pass": MIN_SEEDS_PASS,
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        z_goal_stream_stats=zg_accumulator.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)

    emit_outcome(
        outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
