"""
V3-EXQ-833 -- Stage-H STRICT goal isolation: does silencing the E3 goal term
change LEARNED avoidance? The DV the 2026-07-27 scaffold goal-freeze triage
explicitly left unmeasured.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

THE QUESTION, AND WHY IT IS NOT ALREADY ANSWERED
------------------------------------------------
`_set_goal_pipeline_frozen(agent, frozen=True)` silences the goal WRITE paths
(MECH-295 liking bridge, MECH-307 conjunction) and NOTHING else. The two goal
READ paths are gated independently and stay live through the three stages the
scheduler describes as goal-frozen (Stage-0b, P0, Stage-H):

  * E3 -- `e3_selector.score_trajectory` subtracts `goal_weight * goal_proximity`
    under `E3Config.goal_weight > 0.0` AND `goal_state.is_active()`.
    `REEConfig.from_dims` sets goal_weight to 1.0 (NOT the E3Config dataclass
    default of 0.0), so it is live for essentially every scaffold-built agent.
  * E1 -- `GoalConfig.e1_goal_conditioned` (default True).

The triage (REE_assembly/evidence/planning/scaffold_goal_freeze_e3_read_path_triage_2026-07-27.md)
measured the consequence on the 460c dry-run config over 3 seeds: in Stage-H the
goal term fires on 100% of ticks, its candidate spread is 0.65x the harm term's
and ~17% of the summed component spread, and removing it counterfactually moves
the cost-argmin candidate on 38% of ticks. Stage-H's stated purpose is that "the
agent's E3 harm evaluation drives survival", so a goal term nearly two-thirds as
discriminative as the harm term is a live threat to that reading.

BUT THAT IS A SELECTION COUNTERFACTUAL, NOT A DV. A moved argmin at a near-tie
tick may change nothing about what the agent ends up learning; selection is a
multinomial over softmax(-score/T), so the flip rate over-states behavioural
weight. The triage says so in terms: "Still unmeasured, stated explicitly:
whether silencing the goal term changes learned avoidance." This experiment
measures exactly that, using the opt-in knob built for it
(`ScaffoldedSD054OnboardingConfig.scaffold_strict_goal_isolation`, ree-v3
bd66d67f77).

DESIGN
------
Two arms, 20 shared seeds, paired WITHIN seed:

  ARM_LEGACY = scaffold_strict_goal_isolation False (stock; bit-identical to
               every landed scaffold run -- the goal term fires throughout the
               frozen stages)
  ARM_STRICT = scaffold_strict_goal_isolation True (the frozen stages
               additionally zero e3.goal_weight so the E3 term's own `> 0.0`
               gate FAILS and the subtraction is SKIPPED, and clear
               e1_goal_conditioned so E1 receives z_goal=None)

That knob is the ONLY difference. Both arms build their scaffold config and
their REEConfig from the SAME lineage module
(`experiments/_lib/baselines/stageh_strict_goal_isolation.py`), so the
single-variable claim is true by construction rather than by inspection.

Curriculum per cell: Stage-0 nursery (20) -> Stage-0b consolidation (10) ->
P0 (100) -> Stage-H (40) = 170 episodes. P1/P2 are dropped: they run AFTER the
frozen stages and cannot inform a Stage-H DV, and their 65 episodes/cell are
spent on seeds instead -- see SEED COUNT below for why that is the right trade.
Note the knob bites from Stage-0b onward (not Stage-H alone), because that is
what the landed lever does; Stage-0 is identical across arms, and P0 readouts
are recorded so a reader can see where the arms start to diverge.

SEED COUNT: 20, AND WHY IT IS NOT 3
-----------------------------------
The family standard is 3 seeds. On this DV that is not defensible, and the
anchor run says so quantitatively. Decomposing V3-EXQ-603q's ARM_BASE_IA_ONLY
Stage-H episode lengths:

  between-seed SD of the per-seed mean       29.3 steps
  mean within-seed SEM of that mean (40 eps)  8.1 steps
  -> between-seed share of variance           93%

So episodes-per-seed is nearly worthless as a precision lever and SEEDS are the
only one; this is why the budget freed by dropping P1/P2 is spent on seeds and
the Stage-H budget stays at 603q's 40. Worse, 603q's own within-seed arm delta
(full bridge minus base) had SD 51.4 against a mean of 23.3 -- the arm x seed
interaction is LARGER than the arm effect it was measuring, so at n=3 the
standard error of a paired delta is ~30 steps on a ~38-step baseline. A 3-seed
version of this experiment could not detect the effect NOR license the null; it
would be unfalsifiable either way.

n=20 is chosen so the null is DECLARABLE at a scientifically meaningful
precision rather than merely affordable. Carrying 603q's SD_delta = 51.4:
SE_delta ~ 11.5 and the 95% half-width ~23.0 steps, which clears the
pre-registered 25-step ceiling. 23 steps is almost exactly the measured effect
of the FULL escape-affordance bridge (603q: +23.3 steps of mean survival), so a
null here rules out a goal-term effect as large as an entire validated
substrate mechanism -- which is the claim worth being able to make.

The margin is real but not generous: the ceiling holds while SD_delta <= 55.9,
i.e. up to 9% worse than 603q's. If this run's variance exceeds that, the
pre-registered route is `underpowered_null_more_seeds_required` (a FAIL that
asks for more seeds), NEVER a substrate verdict -- "we could not measure it" is
not "there is nothing there". Cost is the binding constraint on going higher:
one cell is ~26k env-steps, timed at ~69 min on a contended hub (Stage-0 alone
measured 626s for 4000 env-steps, 2026-07-27), and P0 is ~71% of it.

PRE-REGISTERED ACCEPTANCE (constants; NOT derived from the run's own statistics)
-------------------------------------------------------------------------------
Primary DV: mean Stage-H episode length (episodes terminate on hazard contact,
so length = steps survived). Paired within seed: delta_s = STRICT_s - LEGACY_s.

  mean_delta = mean_s(delta_s)
  sd_delta   = stdev_s(delta_s)          se_delta = sd_delta / sqrt(n_pairs)

  EFFECT DETECTED  <=>  |mean_delta| >= K_SE * se_delta  AND
                        |mean_delta| >= ABS_FLOOR_STEPS
      (K_SE = 2.0 -- noise scaled on the SD of the DELTA, per the standing
       effect-size convention; ABS_FLOOR_STEPS = 5.0 -- an absolute floor so a
       tiny-but-precise delta is not reported as behaviourally meaningful.)

  ADEQUATELY POWERED NULL  <=>  NOT effect_detected AND
                                2 * se_delta <= NULL_PRECISION_CEILING_STEPS
      (NULL_PRECISION_CEILING_STEPS = 25.0 -- an absolute pre-registered
       constant, ~2/3 of 603q's 37.7-step base survival. Reading: the null is
       declarable only if the data can rule out a goal-term effect larger than
       25 steps of mean survival.)

BOTH DIRECTIONS ARE INFORMATIVE, AND THE OUTCOME FIELD SAYS SO
--------------------------------------------------------------
outcome PASS = the run REACHED AN ANSWER, of either kind:

  * effect detected -> the goal term IS behaviourally load-bearing inside the
    "isolated" stage. Stage-H's isolation claim is not merely imprecise in
    wording but consequential, and any future work needing goal-free Stage-H
    must set the knob. The sign is reported, not assumed: strict isolation could
    plausibly HELP (less distraction from the harm signal) or HURT (loss of a
    useful navigational prior).
  * adequately powered null -> the 38% argmin-flip does NOT translate into
    changed learned avoidance, to the stated precision. This DISCHARGES the
    isolation worry and is the informative outcome, NOT a failure. It is
    recorded as a PASS deliberately, so a null cannot be misread downstream as
    a run that went wrong.

outcome FAIL = the run could NOT reach an answer:

  * a readiness precondition is unmet -> `substrate_not_ready_requeue`
  * an underpowered null -> `underpowered_null_more_seeds_required`. NEVER a
    substrate verdict: "we could not measure it" is not "there is nothing there".

DV-SYMMETRY DECLARATION (mandatory, per arm)
--------------------------------------------
DV = mean Stage-H episode length, a function of the realised action sequence
through env dynamics. Its symmetry group: it is invariant under any change to
the E3 cost that leaves the per-candidate ORDER unchanged -- a broadcast
additive constant across candidates, or a positive monotone rescaling -- and
under permutation of episodes within the mean.

The manipulation is NOT invariant under that group, and this is measured rather
than argued. `goal_proximity` is computed PER CANDIDATE TRAJECTORY, so the
removed term varies across candidates within a tick: the triage recorded a
Stage-H candidate spread of 0.1022 (0.65x the harm term's) and a 38% argmin flip
rate. A broadcast constant would give a spread of exactly 0 and a flip rate of
exactly 0. So neither arm is a disposition-(b) structurally vacuous arm, and the
delta is a measurement rather than an arithmetic identity fixed before the run.

READINESS PRECONDITIONS (regime-conditioned; see experiments/_lib/precondition_gate.py)
---------------------------------------------------------------------------------------
BOTH arms:
  z_goal_formation_frac        -- Stage-0 z_goal peak > 0.4 on >= 2/3 of seeds.
      THE load-bearing one. With no z_goal, `goal_state.is_active()` is False,
      the E3 goal term is skipped, and the comparison is vacuous by construction
      (the triage's five inert exceptions). The triage's own dry-run numbers
      (0.352/0.259/0.422) were mostly BELOW this gate, which is why it must be
      checked in the production config rather than assumed; 603q's full-budget
      run of this exact substrate recorded 0.497/0.430/0.369.
  harm_landscape_discriminative_frac -- harm_eval_range > 0.02 on >= 2/3 seeds.
      Without harm-pathway training the E3 harm cost is a random constant
      (603i: range [0.522, 0.524]); comparing two arms against a noise harm
      landscape would answer nothing.
  survival_above_floor         -- arm mean Stage-H survival > 8 steps, i.e. the
      agent is not dying instantly every episode with no avoidance to compare.

ARM_LEGACY only (scoped out of ARM_STRICT, where it is False by construction):
  goal_term_live_in_stageh_frac -- at Stage-H exit, e3.goal_weight > 0 AND
      goal_state.is_active() on >= 2/3 seeds. The manipulation-liveness check on
      the control arm: the thing being removed must actually have been PRESENT.

ARM_STRICT only (scoped out of ARM_LEGACY):
  strict_isolation_applied_frac -- at Stage-H exit, e3.goal_weight == 0.0 AND
      e1_goal_conditioned is False on >= 2/3 seeds. The manipulation genuinely
      bit rather than silently no-op'ing.

NOTE ON THE AGGREGATE. `aggregate_arm_gates` defines non_degenerate as ANY arm
green, which is right for a design whose arms carry independent results. This
design is PAIRED: a green ARM_STRICT with a red ARM_LEGACY answers nothing,
because the delta needs both halves. So the top-level `non_degenerate` here is
`aggregate.non_degenerate AND both arms green`. That is a tightening for a
paired estimand, NOT the V3-EXQ-785 defect of letting a red arm vacate another
arm's independently valid finding -- there is no independent per-arm finding to
vacate.

CROSS-PROCESS REPRODUCIBILITY CAVEAT
------------------------------------
The scaffold curriculum is NOT reproducible across processes: two byte-identical
checkouts diverge at Stage-0 with torch, numpy AND stdlib `random` seeded
(`ree_core/hippocampal/module.py` draws from unseeded stdlib random, plus a
residual unidentified source). Hence the comparison is WITHIN-SEED and
within-process: both arms of a seed run in the same process, each preceded by a
complete RNG reset at cell entry (`arm_cell`). Do not expect run-to-run identity
across invocations, and do not compare these numbers to another process's.

claim_ids: [] -- DELIBERATELY EMPTY. experiment_purpose is `diagnostic`, so this
run is excluded from governance confidence/conflict scoring. Tagging SD-054 /
MECH-295 / MECH-307 would let a null silently re-weight their evidence, when
what a null actually licenses is a scope statement about the scaffold's freeze
helper. Route the interpretation through governance instead.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.entropy_headroom import per_arm_headroom  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    stage_plan,
)

from experiments._lib.baselines.stageh_strict_goal_isolation import (  # noqa: E402
    DRY_EPISODES_PER_CELL,
    EPISODES_PER_CELL,
    HAZARD_STAGE_BUDGET,
    HAZARD_STAGE_STABILITY_WINDOW,
    HAZARD_STAGE_SURVIVAL_GATE_STEPS,
    STAGE0_ZGOAL_GATE,
    TRAIN_STEPS,
    arm_config_slice,
    build_agent_config,
    build_scaffold_cfg,
)

EXPERIMENT_TYPE = "v3_exq_833_stageh_strict_goal_isolation_dv"
QUEUE_ID = "V3-EXQ-833"
CLAIM_IDS: List[str] = []          # deliberately empty -- see the module docstring
EXPERIMENT_PURPOSE = "diagnostic"

# 20 seeds -- see SEED COUNT in the module docstring for the variance decomposition
# and the power calculation that fixes n at 20 rather than the family-standard 3.
SEEDS = list(range(42, 62))

ARMS = [
    {"label": "ARM_LEGACY", "strict": False},
    {"label": "ARM_STRICT", "strict": True},
]
LEGACY = "ARM_LEGACY"
STRICT = "ARM_STRICT"

# ---- Pre-registered constants (NOT derived from this run's statistics) ----
# STAGE0_ZGOAL_GATE is imported from the lineage module: it is read INSIDE the
# cell and stamped as `stage0_zgoal_formed`, so it is declared in the arm
# fingerprint's config_slice. See audit Addendum 3.
HARM_DISC_RANGE_FLOOR = 0.02     # 603i flat ~0.002; 603k/603q discriminative ~0.13-0.40
SURVIVAL_FLOOR_STEPS = 8.0       # below this the agent dies instantly; nothing to compare
MIN_FRACTION = 2.0 / 3.0         # per-arm fraction-of-seeds threshold
MIN_PAIRED_SEEDS = 12            # a paired delta below this cannot support the gate

K_SE = 2.0                       # noise scaled on the SD of the DELTA
ABS_FLOOR_STEPS = 5.0            # absolute floor of behavioural meaningfulness
NULL_PRECISION_CEILING_STEPS = 25.0  # 95% half-width ceiling for a declarable null

# Non-gating per-arm survival headroom band (diagnostic only -- see the
# "saturation guard scoped to the baseline" rule; reported for EVERY arm, on
# PASS runs too, because a diagnostic that appears only when something looks
# wrong cannot establish that anything was ever right).
HEADROOM_LOW_FRAC = 0.02         # of TRAIN_STEPS -> floor-pinned
HEADROOM_HIGH_FRAC = 0.95        # of TRAIN_STEPS -> ceiling-pinned


# --------------------------------------------------------------------------- #
# Precondition specs (regime-conditioned)
# --------------------------------------------------------------------------- #
def _is_legacy(ctx: Dict[str, Any]) -> bool:
    return ctx.get("arm") == LEGACY


def _is_strict(ctx: Dict[str, Any]) -> bool:
    return ctx.get("arm") == STRICT


PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="z_goal_formation_frac",
        description=(
            "Fraction of seeds whose Stage-0 z_goal peak clears the 0.4 formation "
            "gate. With no z_goal, goal_state.is_active() is False, the E3 goal "
            "term is skipped outright, and the ON/OFF comparison is vacuous."),
        control=(
            "Stage-0 forced-benefit nursery is the goal-formation positive "
            "control; 603q recorded 0.497/0.430/0.369 on this exact substrate."),
        threshold=MIN_FRACTION,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="harm_landscape_discriminative_frac",
        description=(
            "Fraction of seeds whose post-Stage-H harm_eval_range clears 0.02. "
            "Without a discriminative harm landscape the E3 harm cost is a random "
            "constant and survival is unreachable at any budget."),
        control=(
            "603i flat landscape ~0.002 (negative reference); 603k/603q trained "
            "landscape 0.13-0.40 (positive reference)."),
        threshold=MIN_FRACTION,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="survival_above_floor",
        description=(
            "Arm mean Stage-H episode length in steps. Below the floor the agent "
            "dies almost immediately every episode and there is no learned "
            "avoidance for the goal term to modulate."),
        control="603q base arm mean survival 37.7 steps of 200.",
        threshold=SURVIVAL_FLOOR_STEPS,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="goal_term_live_in_stageh_frac",
        description=(
            "Fraction of seeds where, at Stage-H exit, e3.goal_weight > 0 AND "
            "goal_state.is_active(). Manipulation-liveness on the CONTROL arm: "
            "the term being removed must actually have been present."),
        control=(
            "REEConfig.from_dims sets goal_weight to 1.0 and Stage-0 seeds "
            "z_goal, so this is the documented live state."),
        threshold=MIN_FRACTION,
        direction="lower",
        kind="readiness",
        applies_to=_is_legacy,
        applies_note=(
            "Not meaningful for ARM_STRICT: the knob sets goal_weight to 0.0 "
            "there by construction, so asserting the term is live would make "
            "that arm structurally un-passable."),
    ),
    PreconditionSpec(
        name="strict_isolation_applied_frac",
        description=(
            "Fraction of seeds where, at Stage-H exit, e3.goal_weight == 0.0 AND "
            "e1_goal_conditioned is False. Confirms the knob genuinely bit rather "
            "than silently no-op'ing."),
        control=(
            "_enter_strict_goal_isolation writes both fields at every freeze "
            "site; the LEGACY arm is the negative control for the same read."),
        threshold=MIN_FRACTION,
        direction="lower",
        kind="readiness",
        applies_to=_is_strict,
        applies_note=(
            "Not meaningful for ARM_LEGACY: the knob is off there by design, so "
            "the isolation is expected NOT to be applied."),
    ),
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _goal_read_path_state(agent) -> Dict[str, Any]:
    """Read the two goal READ paths + goal liveness. Pure inspection.

    Called immediately after run_hazard_avoidance returns. run_hazard_avoidance
    freezes on entry and does NOT unfreeze on exit (P1 does), so the Stage-H
    state is still in place at this point -- which is exactly what must be
    checked.
    """
    e3 = getattr(agent, "e3", None)
    e3_cfg = getattr(e3, "config", None) if e3 is not None else None
    goal_weight = float(getattr(e3_cfg, "goal_weight", 0.0)) if e3_cfg is not None else 0.0

    goal_cfg = getattr(getattr(agent, "config", None), "goal", None)
    e1_goal_conditioned = bool(getattr(goal_cfg, "e1_goal_conditioned", False))

    gs = getattr(agent, "goal_state", None)
    active = False
    if gs is not None:
        try:
            raw = gs.is_active
            active = bool(raw() if callable(raw) else raw)
        except Exception:
            active = False

    goal_norm = 0.0
    if gs is not None and hasattr(gs, "goal_norm"):
        try:
            gn = gs.goal_norm
            goal_norm = float(gn() if callable(gn) else gn)
        except Exception:
            goal_norm = 0.0

    return {
        "e3_goal_weight": goal_weight,
        "e1_goal_conditioned": e1_goal_conditioned,
        "goal_state_active": active,
        "goal_norm": goal_norm,
        # The E3 term fires only when BOTH gates admit it.
        "e3_goal_term_live": bool(goal_weight > 0.0 and active),
        "strict_isolation_applied": bool(goal_weight == 0.0 and not e1_goal_conditioned),
    }


def _contact_termination_rate(ep_lengths: List[int], max_steps: int) -> float:
    """Fraction of Stage-H episodes that ended early, i.e. on hazard contact."""
    if not ep_lengths:
        return 0.0
    return float(sum(1 for L in ep_lengths if L < max_steps)) / float(len(ep_lengths))


def _auc_survival(ep_lengths: List[int], max_steps: int) -> float:
    if not ep_lengths:
        return 0.0
    return float(sum(ep_lengths)) / float(len(ep_lengths) * max_steps)


def _curve_thirds(ep_lengths: List[int]) -> Dict[str, float]:
    """Mean episode length in the first / middle / last third of Stage-H.

    The secondary readout: do the arms' LEARNING CURVES separate at all, or only
    their endpoints?
    """
    n = len(ep_lengths)
    if n < 3:
        return {"first_third": 0.0, "middle_third": 0.0, "last_third": 0.0}
    k = n // 3
    return {
        "first_third": float(st.fmean(ep_lengths[:k])),
        "middle_third": float(st.fmean(ep_lengths[k:2 * k])),
        "last_third": float(st.fmean(ep_lengths[2 * k:])),
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _aborted_record(arm_label: str, seed: int, stage: str, reason: str,
                    s0_peak: float = 0.0) -> Dict[str, Any]:
    return {
        "arm": arm_label,
        "seed": seed,
        "aborted_at": stage,
        "abort_reason": reason,
        "stage0_z_goal_norm_peak": float(s0_peak),
        "stage0_zgoal_formed": bool(s0_peak > STAGE0_ZGOAL_GATE),
        "p0_mean_episode_length": 0.0,
        "hazard_stage_mean_episode_length": 0.0,
        "hazard_stage_median_last_window": 0.0,
        "hazard_stage_survival_pass": False,
        "hazard_stage_auc_survival": 0.0,
        "hazard_stage_contact_termination_rate": 0.0,
        "hazard_stage_episode_lengths": [],
        "hazard_stage_curve_thirds": {"first_third": 0.0, "middle_third": 0.0,
                                      "last_third": 0.0},
        "harm_eval_range": 0.0,
        "harm_discriminativeness": {},
        "goal_read_path_state": {},
        "e3_goal_term_live": False,
        "strict_isolation_applied": False,
        "avoidance_gate_state": {},
        "usable": False,
    }


# --------------------------------------------------------------------------- #
# One (arm, seed) cell
# --------------------------------------------------------------------------- #
def _run_seed_arm(arm: Dict[str, Any], seed: int, dry_run: bool, total_eps: int,
                  zg_acc: ZGoalStreamAccumulator) -> Dict[str, Any]:
    """Stage-0 -> Stage-0b -> P0 -> Stage-H for one cell.

    arm_cell resets ALL RNG on enter (so cell order cannot matter) and stamps the
    reuse fingerprint on the returned row. include_driver_script_in_hash=False so
    a successor experiment with a DIFFERENT driver can still reuse these cells --
    the MINT-AS-YOU-GO default; terminality is never knowable in advance.
    """
    label = arm["label"]
    with arm_cell(
        seed,
        config_slice=arm_config_slice(arm["strict"], dry_run),
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
    ) as cell:
        scaffold_cfg = build_scaffold_cfg(dry_run, arm["strict"])
        device = torch.device("cpu")
        probe_env = _build_env(scaffold_cfg, "hazard")
        probe_env.reset()
        agent = REEAgent(build_agent_config(probe_env)).to(device)
        scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

        # Boundary line: resets the runner's episodes_in_run for this cell.
        print(f"Seed {seed} Condition {label}", flush=True)

        # ---- Stage-0: forced-benefit nursery (z_goal formation) ----
        s0 = scheduler.run_stage0_nursery(agent, device)
        done = s0.n_episodes
        print(f"  [train] stage0 {label} seed={seed} ep {done}/{total_eps}"
              f" z_goal_peak={s0.z_goal_norm_peak:.4f}", flush=True)
        if s0.aborted:
            print(f"verdict: FAIL seed={seed} arm={label} aborted_at=stage0", flush=True)
            rec = _aborted_record(label, seed, "stage0", s0.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            zg_acc.observe(agent)
            cell.stamp(rec)
            return rec

        # ---- Stage-0b: consolidation (FROZEN stage -- knob bites from here) ----
        s0b = scheduler.run_stage0b_consolidation(
            agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
        done += s0b.n_episodes
        print(f"  [train] stage0b {label} seed={seed} ep {done}/{total_eps}", flush=True)
        if s0b.aborted:
            print(f"verdict: FAIL seed={seed} arm={label} aborted_at=stage0b", flush=True)
            rec = _aborted_record(label, seed, "stage0b", s0b.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            zg_acc.observe(agent)
            cell.stamp(rec)
            return rec

        # ---- P0: warm-up (FROZEN stage) ----
        p0 = scheduler.run_p0(agent, device)
        done += p0.n_episodes
        print(f"  [train] p0 {label} seed={seed} ep {done}/{total_eps}"
              f" mean_len={p0.mean_episode_length:.1f}"
              f" rv={p0.final_running_variance:.5f}", flush=True)
        if p0.aborted:
            print(f"verdict: FAIL seed={seed} arm={label} aborted_at=p0", flush=True)
            rec = _aborted_record(label, seed, "p0", p0.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            zg_acc.observe(agent)
            cell.stamp(rec)
            return rec

        # ---- Stage-H: the DV (FROZEN stage) ----
        hz = scheduler.run_hazard_avoidance(agent, device)
        done += hz.n_episodes

        # Manipulation-liveness read. run_hazard_avoidance freezes on entry and
        # does NOT unfreeze on exit (P1 does), so the Stage-H goal-read-path state
        # is still live here -- which is the state that must be verified.
        goal_state = _goal_read_path_state(agent)

        harm_disc = dict(hz.harm_discriminativeness or {})
        harm_eval_range = float(harm_disc.get("harm_eval_range", 0.0))
        ep_lengths = [int(x) for x in (hz.episode_lengths or [])]
        max_steps = int(scaffold_cfg.scaffold_steps_per_episode)
        auc = _auc_survival(ep_lengths, max_steps)
        contact_rate = _contact_termination_rate(ep_lengths, max_steps)
        thirds = _curve_thirds(ep_lengths)

        print(f"  [train] hazard {label} seed={seed} ep {done}/{total_eps}"
              f" mean_len={hz.mean_episode_length:.1f}"
              f" median_last={hz.median_last_window_episode_length:.1f}"
              f" auc={auc:.3f}"
              f" contact_term={contact_rate:.3f}"
              f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}"
              f" harm_range={harm_eval_range:.4f}"
              f" goal_w={goal_state['e3_goal_weight']:.3f}"
              f" e1_gc={goal_state['e1_goal_conditioned']}"
              f" goal_live={goal_state['e3_goal_term_live']}", flush=True)

        zg_acc.observe(agent)

        if hz.aborted:
            print(f"verdict: FAIL seed={seed} arm={label} aborted_at=hazard", flush=True)
            rec = _aborted_record(label, seed, "hazard", hz.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            rec["harm_eval_range"] = harm_eval_range
            rec["goal_read_path_state"] = goal_state
            cell.stamp(rec)
            return rec

        # A cell is USABLE for the paired delta when Stage-H actually produced
        # episodes. Readiness is judged at the ARM level (fraction of seeds), not
        # here -- a single seed missing the z_goal gate is a normal occurrence
        # (603q: 1 of 3), not grounds to drop the pair.
        usable = bool(ep_lengths)

        print(f"verdict: {'PASS' if usable else 'FAIL'} seed={seed} arm={label}"
              f" mean_surv={hz.mean_episode_length:.1f}"
              f" z0peak={s0.z_goal_norm_peak:.4f}"
              f" harm_range={harm_eval_range:.4f}", flush=True)

        rec = {
            "arm": label,
            "seed": seed,
            "aborted_at": None,
            "abort_reason": "",
            "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
            "stage0_zgoal_formed": bool(s0.z_goal_norm_peak > STAGE0_ZGOAL_GATE),
            "p0_mean_episode_length": float(p0.mean_episode_length),
            "p0_final_running_variance": float(p0.final_running_variance),
            # PRIMARY DV.
            "hazard_stage_mean_episode_length": float(hz.mean_episode_length),
            # Secondaries.
            "hazard_stage_median_last_window": float(
                hz.median_last_window_episode_length),
            "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
            "hazard_stage_auc_survival": auc,
            "hazard_stage_contact_termination_rate": contact_rate,
            "hazard_stage_n_episodes": len(ep_lengths),
            "hazard_stage_episode_lengths": ep_lengths,   # full curve, every seed
            "hazard_stage_curve_thirds": thirds,
            "hazard_stage_final_running_variance": float(hz.final_running_variance),
            "harm_eval_range": harm_eval_range,
            "harm_discriminativeness": harm_disc,
            "harm_pathway_enabled": bool(hz.harm_pathway_enabled),
            "harm_pathway_diag": dict(hz.harm_pathway_diag or {}),
            # Manipulation liveness.
            "goal_read_path_state": goal_state,
            "e3_goal_term_live": bool(goal_state["e3_goal_term_live"]),
            "strict_isolation_applied": bool(goal_state["strict_isolation_applied"]),
            "avoidance_driver_enabled": bool(hz.avoidance_driver_enabled),
            "avoidance_gate_state": dict(hz.avoidance_gate_state or {}),
            "usable": usable,
        }
        cell.stamp(rec)
        return rec


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _arm_gate(arm_label: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate one arm's readiness gate from its per-seed rows."""
    ctx = {"arm": arm_label}
    measured = {
        "z_goal_formation_frac": _frac([r["stage0_zgoal_formed"] for r in rows]),
        "harm_landscape_discriminative_frac": _frac(
            [r["harm_eval_range"] > HARM_DISC_RANGE_FLOOR for r in rows]),
        "survival_above_floor": (
            float(st.fmean([r["hazard_stage_mean_episode_length"] for r in rows]))
            if rows else 0.0),
    }
    if arm_label == LEGACY:
        measured["goal_term_live_in_stageh_frac"] = _frac(
            [r["e3_goal_term_live"] for r in rows])
    if arm_label == STRICT:
        measured["strict_isolation_applied_frac"] = _frac(
            [r["strict_isolation_applied"] for r in rows])
    return evaluate_arm_gate(arm_label, ctx, PRECONDITION_SPECS, measured)


def _paired_analysis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Within-seed paired delta on the primary DV (STRICT - LEGACY)."""
    by_arm = {LEGACY: {}, STRICT: {}}
    for r in rows:
        if r.get("usable"):
            by_arm[r["arm"]][r["seed"]] = r

    seeds = sorted(set(by_arm[LEGACY]) & set(by_arm[STRICT]))
    pairs = []
    for s in seeds:
        leg = by_arm[LEGACY][s]["hazard_stage_mean_episode_length"]
        stc = by_arm[STRICT][s]["hazard_stage_mean_episode_length"]
        pairs.append({
            "seed": s,
            "legacy_mean_survival": float(leg),
            "strict_mean_survival": float(stc),
            "delta": float(stc - leg),
        })

    n = len(pairs)
    deltas = [p["delta"] for p in pairs]
    mean_delta = float(st.fmean(deltas)) if n else 0.0
    sd_delta = float(st.stdev(deltas)) if n >= 2 else 0.0
    se_delta = float(sd_delta / (n ** 0.5)) if n >= 2 else 0.0
    ci_half = 2.0 * se_delta

    effect_threshold = max(K_SE * se_delta, ABS_FLOOR_STEPS)
    effect_detected = bool(n >= MIN_PAIRED_SEEDS and abs(mean_delta) >= effect_threshold)
    powered_null = bool(
        n >= MIN_PAIRED_SEEDS
        and not effect_detected
        and ci_half <= NULL_PRECISION_CEILING_STEPS
    )

    # Degeneracy: bit-identical arms would make the delta an arithmetic zero
    # rather than a measurement.
    all_zero = bool(n > 0 and all(abs(d) < 1e-12 for d in deltas))

    return {
        "n_pairs": n,
        "pairs": pairs,
        "mean_delta": mean_delta,
        "sd_delta": sd_delta,
        "se_delta": se_delta,
        "ci95_half_width": ci_half,
        "effect_threshold_applied": float(effect_threshold),
        "effect_detected": effect_detected,
        "adequately_powered_null": powered_null,
        "answer_reached": bool(effect_detected or powered_null),
        "direction": ("strict_higher_survival" if mean_delta > 0
                      else "strict_lower_survival" if mean_delta < 0
                      else "no_difference"),
        "arms_bit_identical": all_zero,
        "legacy_mean_survival": float(
            st.fmean([p["legacy_mean_survival"] for p in pairs])) if n else 0.0,
        "strict_mean_survival": float(
            st.fmean([p["strict_mean_survival"] for p in pairs])) if n else 0.0,
    }


def _secondary_readouts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-arm secondary DVs + whether the learning curves separate at all."""
    out: Dict[str, Any] = {}
    for label in (LEGACY, STRICT):
        arm_rows = [r for r in rows if r["arm"] == label and r.get("usable")]
        if not arm_rows:
            out[label] = {}
            continue
        out[label] = {
            "n_seeds": len(arm_rows),
            "mean_survival": float(st.fmean(
                [r["hazard_stage_mean_episode_length"] for r in arm_rows])),
            "median_last_window": float(st.fmean(
                [r["hazard_stage_median_last_window"] for r in arm_rows])),
            "survival_gate_pass_rate": _frac(
                [r["hazard_stage_survival_pass"] for r in arm_rows]),
            "contact_termination_rate": float(st.fmean(
                [r["hazard_stage_contact_termination_rate"] for r in arm_rows])),
            "auc_survival": float(st.fmean(
                [r["hazard_stage_auc_survival"] for r in arm_rows])),
            "curve_thirds": {
                k: float(st.fmean([r["hazard_stage_curve_thirds"][k] for r in arm_rows]))
                for k in ("first_third", "middle_third", "last_third")
            },
            "p0_mean_episode_length": float(st.fmean(
                [r["p0_mean_episode_length"] for r in arm_rows])),
            "stage0_z_goal_peak": float(st.fmean(
                [r["stage0_z_goal_norm_peak"] for r in arm_rows])),
        }
    # Do the curves separate, or only the endpoints?
    if out.get(LEGACY) and out.get(STRICT):
        out["curve_separation"] = {
            k: float(out[STRICT]["curve_thirds"][k] - out[LEGACY]["curve_thirds"][k])
            for k in ("first_third", "middle_third", "last_third")
        }
        out["p0_divergence_mean_episode_length"] = float(
            out[STRICT]["p0_mean_episode_length"] - out[LEGACY]["p0_mean_episode_length"])
    return out


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:2] if dry_run else SEEDS
    total_eps = DRY_EPISODES_PER_CELL if dry_run else EPISODES_PER_CELL

    # Design-time refusal: a precondition no arm could satisfy from its
    # PRE-REGISTERED config is a design bug, and this catches it before compute.
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITION_SPECS,
        [{"arm": a["label"], "strict": a["strict"]} for a in ARMS],
    )

    zg_acc = ZGoalStreamAccumulator()
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_seed_arm(arm, seed, dry_run, total_eps, zg_acc))

    # ---- Per-arm readiness gates (regime-conditioned) ----
    arm_gates = [_arm_gate(label, [r for r in rows if r["arm"] == label])
                 for label in (LEGACY, STRICT)]
    agg = aggregate_arm_gates(arm_gates)
    both_green = all(g["gate_green"] for g in arm_gates)

    # ---- Primary paired analysis ----
    paired = _paired_analysis(rows)
    secondary = _secondary_readouts(rows)

    # ---- Criteria ----
    c1 = both_green
    c2 = bool(paired["n_pairs"] >= MIN_PAIRED_SEEDS)
    c3 = bool(paired["answer_reached"])          # LOAD-BEARING
    outcome = "PASS" if (c1 and c2 and c3) else "FAIL"

    if not both_green:
        label_ = "substrate_not_ready_requeue"
    elif not c2:
        label_ = "insufficient_usable_pairs_requeue"
    elif paired["effect_detected"]:
        label_ = "goal_term_behaviourally_load_bearing_in_stageh"
    elif paired["adequately_powered_null"]:
        label_ = "goal_term_behaviourally_inert_in_stageh_within_stated_precision"
    else:
        label_ = "underpowered_null_more_seeds_required"

    # A paired estimand needs BOTH halves: a green STRICT with a red LEGACY
    # answers nothing. See the module docstring's NOTE ON THE AGGREGATE for why
    # this tightening is not the V3-EXQ-785 defect.
    non_degenerate = bool(agg["non_degenerate"] and both_green
                          and not paired["arms_bit_identical"])
    degeneracy_reason = ""
    if not non_degenerate:
        if paired["arms_bit_identical"]:
            degeneracy_reason = (
                "every within-seed delta is exactly 0.0 -- the arms produced "
                "bit-identical Stage-H behaviour, so the knob did not bite")
        elif not both_green:
            degeneracy_reason = (
                "paired design requires BOTH arms green; "
                + (agg.get("degeneracy_reason") or "")
                + f" (green={[g['arm'] for g in arm_gates if g['gate_green']]})")
        else:
            degeneracy_reason = agg.get("degeneracy_reason") or "aggregate not green"

    # Each arm owns its OWN gate criterion (a shared name would collide in the
    # per-arm map and silently take whichever arm iterated last).
    criteria_non_degenerate = arm_criteria_non_degenerate(
        {LEGACY: ["C1a_legacy_arm_gate_green"], STRICT: ["C1b_strict_arm_gate_green"]},
        agg,
    )
    # C2/C3 are run-level, not arm-owned: degenerate when there is no paired
    # comparison to make, or when the arms came out bit-identical (an arithmetic
    # zero rather than a measured null).
    criteria_non_degenerate["C2_sufficient_usable_pairs"] = bool(paired["n_pairs"] > 0)
    criteria_non_degenerate["C3_answer_reached"] = bool(
        not paired["arms_bit_identical"] and paired["n_pairs"] >= 2)

    # Non-gating per-arm survival headroom (reported on PASS runs too).
    headroom = per_arm_headroom(
        [r for r in rows if r.get("usable")],
        value_key="hazard_stage_mean_episode_length",
        low=HEADROOM_LOW_FRAC * TRAIN_STEPS,
        high=HEADROOM_HIGH_FRAC * TRAIN_STEPS,
    )

    return {
        "outcome": outcome,
        "evidence_direction": "non_contributory",  # diagnostic; claim_ids is empty
        "z_goal_stream_stats": zg_acc.stats(),
        "per_arm_gate": agg["per_arm_gate"],
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "paired_analysis": paired,
        "secondary_readouts": secondary,
        "per_seed": rows,
        "arm_results": rows,
        "diagnostics": {"survival_headroom_per_arm": headroom},
        "interpretation": {
            "label": label_,
            "preconditions": agg["adjudication_preconditions"],
            "preconditions_scope_note": agg["per_arm_gate"].get(
                "preconditions_scope_note", ""),
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": [
                {"name": "C1a_legacy_arm_gate_green", "load_bearing": False,
                 "passed": bool(arm_gates[0]["gate_green"])},
                {"name": "C1b_strict_arm_gate_green", "load_bearing": False,
                 "passed": bool(arm_gates[1]["gate_green"])},
                {"name": "C2_sufficient_usable_pairs", "load_bearing": False,
                 "passed": c2},
                {"name": "C3_answer_reached", "load_bearing": True, "passed": c3},
            ],
            "reading": {
                "goal_term_behaviourally_load_bearing_in_stageh": (
                    "The goal term measurably changes learned Stage-H avoidance. "
                    "The scaffold's isolation gap is consequential, not merely a "
                    "wording imprecision; work needing goal-free Stage-H must set "
                    "scaffold_strict_goal_isolation=True."),
                "goal_term_behaviourally_inert_in_stageh_within_stated_precision": (
                    "The 38% argmin-flip does NOT translate into changed learned "
                    "avoidance, ruling out effects above the pre-registered "
                    f"{NULL_PRECISION_CEILING_STEPS} steps. This DISCHARGES the "
                    "isolation worry and is why the outcome is PASS -- an "
                    "informative null, not a failed run."),
                "underpowered_null_more_seeds_required": (
                    "No effect met the gate AND the interval is too wide to "
                    "license the null. Not a substrate verdict: re-queue with "
                    "more seeds."),
                "substrate_not_ready_requeue": (
                    "A readiness precondition failed -- most likely z_goal did "
                    "not form (making the goal term inert and the comparison "
                    "vacuous) or the harm landscape stayed flat."),
            },
        },
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = run_experiment(dry_run=dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    config = {
        "seeds": SEEDS if not dry_run else SEEDS[:2],
        "arms": [a["label"] for a in ARMS],
        "manipulated_variable": "ScaffoldedSD054OnboardingConfig.scaffold_strict_goal_isolation",
        "arm_config_slice_legacy": arm_config_slice(False, dry_run),
        "arm_config_slice_strict": arm_config_slice(True, dry_run),
        "episodes_per_cell": DRY_EPISODES_PER_CELL if dry_run else EPISODES_PER_CELL,
        "curriculum": "stage0 -> stage0b -> p0 -> stage_h (P1/P2 dropped)",
        "train_steps_per_episode": TRAIN_STEPS,
        "hazard_stage_budget": HAZARD_STAGE_BUDGET,
        "hazard_stage_stability_window": HAZARD_STAGE_STABILITY_WINDOW,
        "hazard_stage_survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        "pre_registered_gates": {
            "STAGE0_ZGOAL_GATE": STAGE0_ZGOAL_GATE,
            "HARM_DISC_RANGE_FLOOR": HARM_DISC_RANGE_FLOOR,
            "SURVIVAL_FLOOR_STEPS": SURVIVAL_FLOOR_STEPS,
            "MIN_FRACTION": MIN_FRACTION,
            "MIN_PAIRED_SEEDS": MIN_PAIRED_SEEDS,
            "K_SE": K_SE,
            "ABS_FLOOR_STEPS": ABS_FLOOR_STEPS,
            "NULL_PRECISION_CEILING_STEPS": NULL_PRECISION_CEILING_STEPS,
        },
        "dry_run": bool(dry_run),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": ts,
        "sleep_driver_pattern": "N/A (no sleep loop)",
        "stage_plan": stage_plan(),
        "depends_on": {
            "knob_commit": "bd66d67f77",
            "knob": "ScaffoldedSD054OnboardingConfig.scaffold_strict_goal_isolation",
            "triage": ("REE_assembly/evidence/planning/"
                       "scaffold_goal_freeze_e3_read_path_triage_2026-07-27.md"),
            "config_anchor": "V3-EXQ-603q ARM_BASE_IA_ONLY (truncated after Stage-H)",
        },
        "dv_symmetry_declaration": {
            "dv": "mean Stage-H episode length (steps survived)",
            "symmetry_group": (
                "invariant under any change to the E3 cost preserving per-candidate "
                "ORDER (broadcast additive constant across candidates; positive "
                "monotone rescaling), and under permutation of episodes in the mean"),
            "manipulation_is_invariant": False,
            "evidence": (
                "goal_proximity is per-candidate: the 2026-07-27 triage measured a "
                "Stage-H candidate spread of 0.1022 (0.65x the harm term's) and a "
                "38% argmin flip rate. A broadcast constant would give spread 0.0 "
                "and flip rate 0.0, so neither arm is structurally vacuous."),
            "per_arm": {
                LEGACY: "goal term present and order-affecting -> scorable",
                STRICT: "goal term removed, changing per-candidate order -> scorable",
            },
        },
        "config": config,
    }
    manifest.update(result)

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=config,
        seeds=(SEEDS[:2] if dry_run else SEEDS),
        script_path=Path(__file__),
        started_at=t0,
        # Accumulated per cell (the agent is built inside _run_seed_arm and
        # dropped when it returns, so the run-level accumulator is the right
        # shape here rather than holding every agent alive for provenance).
        z_goal_stream_stats=result.get("z_goal_stream_stats"),
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. outcome={result['outcome']} label={result['interpretation']['label']}"
          f" n_pairs={result['paired_analysis']['n_pairs']}"
          f" mean_delta={result['paired_analysis']['mean_delta']:.2f}"
          f" ci95_half={result['paired_analysis']['ci95_half_width']:.2f}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    _outcome_raw = str(_res["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res["manifest_path"],
        dry_run=args.dry_run,
    )
