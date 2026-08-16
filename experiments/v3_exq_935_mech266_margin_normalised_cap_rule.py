"""
V3-EXQ-935 (MECH-266 / SD-032a, GOV-FANOUT-1 leg H1 successor): is the
external_task cap recalibration a SHIPPABLE RULE or SEED-IDIOSYNCRATIC?
DIAGNOSTIC.

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).

WHY THIS RUN EXISTS
-------------------
V3-EXQ-934 (PASS, 2026-08-15T01:52Z, route
`cap_recalibration_admits_mixed_regime`) resolved GOV-FANOUT-1 leg H1: a mixed
external_task occupancy regime IS reachable, so the 464e/467e bang-bang is cap
MIS-CALIBRATION, not structural discreteness (H2). It reported a
`winning_cap_band_symmetric` of [0.75, 1.75].

THAT BAND IS NOT A RECALIBRATION TARGET, AND READING IT AS ONE IS THE ERROR
THIS RUN EXISTS TO PREVENT. It is the min and max of the caps at which SOME
seed graded -- a per-seed best-point statistic, not a common operating point.
Re-reading 934's own banked cells (`per_seed[].cells`, ARM_SYMMETRIC), the
occupancy at each swept cap is:

    cap     seed42    seed43    seed44    #seeds graded (0.1 < occ < 0.9)
    0.75    0.5606*   1.0000    1.0000    1/3
    1.00    0.0317    1.0000    1.0000    0/3
    1.25    0.0000    1.0000    1.0000    0/3
    1.50    0.0000    1.0000    1.0000    0/3
    1.75    0.0000    0.4447*   0.9002    1/3
                                          (* = graded)

NO SINGLE ABSOLUTE CAP GRADES ON MORE THAN 1 OF 3 SEEDS. The 2/3 acceptance
fraction 934 cleared was computed per-seed over each seed's OWN best cap, which
is the right question for "is a mixed regime reachable at all" (H1) and the
WRONG question for "can we ship a cap". A per-seed hand-tuned constant is not a
substrate fix.

THE HYPOTHESIS THIS RUN TESTS (and where it comes from)
------------------------------------------------------
The seeds differ in the SCALE of their external_task drive. Reading 934's
banked `ext_margin_mean` on ARM_SYMMETRIC at the lowest swept cap (0.75), the
per-seed baseline margin is 0.3373 / 0.7681 / 0.7850 for seeds 42 / 43 / 44 --
seed 42's drive is less than half the size of the other two. A uniform cap is
therefore a different manipulation for each agent: it over-clamps the weak-drive
seed (occupancy -> 0) at values where the strong-drive seeds are still saturated
at 1.0. That is the textbook signature of a missing normalisation.

Normalising each seed's cap by its OWN baseline margin,
r = cap / ext_margin_mean(cap=0.75, ARM_SYMMETRIC), collapses all three seeds
onto one axis -- and every graded cell in 934 lands in a narrow r band:

    seed   baseline_margin   r at each swept cap                 occupancy
    42     0.3373            2.22*  2.96   3.71   4.45   5.19    0.5606* 0.0317 0 0 0
    43     0.7681            0.98   1.30   1.63   1.95   2.28*   1.0 1.0 1.0 1.0 0.4447*
    44     0.7850            0.96   1.27   1.59   1.91   2.23    1.0 1.0 1.0 1.0 0.9002

    r <~ 2.0   -> saturated at 1.0  (every seed, every such cell)
    r ~ 2.2-2.3 -> graded            (0.5606, 0.4447; and 0.9002, marginally out)
    r >~ 3.0   -> collapsed to ~0   (every seed, every such cell)

MECHANISM (why a scale normalisation is the RIGHT form, not a curve fit).
`SalienceCoordinator.tick()` (salience_coordinator.py:455-467) clamps each
affinity input signal to [-cap, +cap] BEFORE its per-mode weight is applied,
while `external_task_bias` is added to the external_task logit and is NOT
clamped. So the cap sets the ratio between the clamped signal contribution and
the unclamped bias -- a ratio that is only meaningful relative to how large that
seed's signal actually is. A cap expressed in absolute units is therefore
under-specified; a cap expressed as a multiple of the agent's own drive scale is
the quantity the arbitration actually responds to.

  H-RULE (this run): a SINGLE pre-registered r, applied per-seed as
      cap = r * baseline_margin(seed), yields a graded regime on >= 2/3 of seeds
      SIMULTANEOUSLY -- i.e. the recalibration is a shippable rule.
  H-IDIO (the null): no single r does better than the best single absolute cap.
      The required cap is seed-idiosyncratic; there is no rule to ship, and
      MECH-266's over-binding test cannot be run on a calibrated substrate
      without per-seed hand-tuning (which would itself be a confound).
  H-KNIFE (aliasing control): the rule is right but the graded window in r is so
      narrow that no fixed r is robust. Distinguished from H-IDIO by the r-sweep
      below -- WITHOUT the sweep, "C1 failed" would alias "wrong rule" with
      "right rule, slightly wrong r".

R_STAR IS PRE-REGISTERED AT 2.25, DERIVED BEFORE THIS RUN, FROM BANKED DATA
ONLY: the mean of the three observed graded/near-graded r values
(2.224, 2.278, 2.229) is 2.244, rounded to 2.25. It is deliberately the MEAN and
not a value hand-picked to make any particular seed clear the band -- that
choice is what keeps the pre-registration honest, and the r-sweep is what
reports whether a different r would have been better (information for the
successor, NOT a post-hoc rescue of this run's verdict).

CIRCULARITY, STATED PLAINLY, AND WHAT CONTROLS IT
-------------------------------------------------
R_STAR was derived FROM seeds 42/43/44. Testing the rule on those same seeds is
partly circular and CANNOT on its own establish that the rule generalises. So:
  - Seeds 45 and 46 are NEW and were not used to derive R_STAR. They are the
    genuine out-of-sample test, and C2 below is scored on them ALONE.
  - Seeds 42/43/44 are retained anyway, and deliberately: seed 44 is the only
    seed that graded at NO absolute cap, so dropping it would remove the hardest
    case and bias the comparison in the rule's favour. (Note the standing
    "substitute seed 45 for 44 on reef-config" caution: it does not apply here --
    934 ran seed 44 on this exact curriculum with guard_pass=True,
    contact_rate=0.3064 -- and seed 45 is present as a NEW seed regardless.)

DESIGN
------
Per seed, ONE trained curriculum agent (identical scaffold to 934), then
frozen-policy eval cells on clones. ALL cells use ARM_SYMMETRIC rails (934's
primary arm and the neutral place to ask a calibration question); the rail
contrast is not re-run here.

  1. CALIBRATION CELL at CAP_REF = 0.75 -> baseline_margin m_seed. CAP_REF is
     0.75 because that is exactly the cap/arm at which the banked margins that
     produced R_STAR were measured; changing it would silently invalidate the
     pre-registered R_STAR.
  2. ARM_NORM cells: one per r in R_SWEEP = [1.85, 2.05, 2.25, 2.45, 2.65],
     each at cap = r * m_seed. R_STAR = 2.25 is IN the sweep, so the primary
     criterion costs no extra compute and the window shape comes free.
  3. ARM_ABS control: cap = CAP_ABS_CONTROL = 1.75, fixed, identical for every
     seed. 1.75 is pre-registered as the STRONGEST absolute contender from 934
     (1/3 graded plus one near-miss at 0.9002; cap=0.75 also scored 1/3 but with
     two hard-saturated seeds), so the head-to-head is against the best
     available absolute cap, not a straw man.

DEPENDENT VARIABLE (per cell): `fraction_in_external_task`, the discrete
committed-mode occupancy -- the SAME statistic 464e/934 measured and the one the
nulls above are stated in. The continuous pre-argmax
`operating_mode['external_task']` margin is recorded alongside (it is also the
calibration statistic), plus mode-conditioned dwell, switch counts and the
coordinator's own tick count so every denominator is auditable.

CRITERIA (pre-registered; all thresholds are constants in this file)
  C1 RULE_GRADES_AT_R_STAR      [load-bearing] -- at the SINGLE r = R_STAR,
     ARM_NORM occupancy is in (0.1, 0.9) on >= MIN_FRACTION of guard-passing
     seeds. This is a COMMON-RULE test: one r for all seeds. It is deliberately
     NOT "some r grades per seed" -- that would reproduce exactly the per-seed
     best-point weakness this run exists to correct, and it is reported as
     `graded_at_some_r_fraction` for information ONLY, explicitly not
     load-bearing.
  C2 RULE_GENERALISES_OUT_OF_SAMPLE [load-bearing] -- C1's condition restricted
     to the out-of-sample seeds (45, 46) alone, at >= 1/2. SCOPED OUT (not
     failed) if no out-of-sample seed passes the readiness guards, with the
     reason recorded -- a starved criterion must never read as a falsification.
  C3 BEATS_BEST_ABSOLUTE_CAP    [load-bearing] -- strictly more guard-passing
     seeds grade under ARM_NORM at R_STAR than under ARM_ABS at 1.75, scored on
     the SAME seeds in the same run.

  PASS iff C1 AND C3, and C2 unless scoped out. The combination rule is a plain
  AND and is recorded as `combination_rule` in the manifest so a reader never
  has to infer the gate from the per-criterion booleans.

READINESS GATES (route a not-ready read to substrate_not_ready_requeue, NEVER a
false verdict). All three are SEED-scoped, apply identically to every cell of
that seed, and are aggregated as a FRACTION over seeds -- never AND-ed across
seeds, so one bad seed cannot vacate the run (the V3-EXQ-785 vacating defect, in
its seed-level form).
  G-contact -- 603n contact guard: P2 contact_rate > 0 AND
     z_goal_norm_at_contact_peak > 0.4, on >= 2/3 seeds. An agent that never
     became foraging-competent makes every occupancy reading meaningless.
  G-margin  -- the external_task drive must ENGAGE: per-seed max-over-cells of
     the continuous ext_margin_mean > MARGIN_FLOOR. If the margin is ~0
     everywhere the drive is not producing the signal at all (wiring/substrate),
     which must self-route not-ready and NOT be read as H2 structural.
  G-calib   -- NEW, and specific to this design: the calibration cell must yield
     m_seed > MARGIN_FLOOR. If m_seed ~ 0 then cap = r * m_seed ~ 0 for every r
     and the ENTIRE ARM_NORM manipulation is degenerate -- there is nothing to
     test. This is the readiness form of the same statistic C1 routes on: C1
     reads occupancy under a cap DERIVED from m_seed, so a dead m_seed starves
     C1 rather than falsifying it.

STRUCTURAL SATISFIABILITY (checked on paper before queuing, per the
pre-registered-value rule; recorded in the manifest as
`structural_satisfiability`). Every gate is satisfiable from values 934 actually
measured on this exact substrate: G-contact 3/3 vs a 2/3 floor; G-margin
0.337-0.785 vs a 0.05 floor; G-calib 0.3373/0.7681/0.7850 at exactly CAP_REF on
exactly ARM_SYMMETRIC, vs a 0.05 floor. C1 is satisfiable but NOT trivially so
(2 of the 3 derivation seeds graded at r ~ 2.2), and C3's bar is a real one --
ARM_ABS genuinely grades on some seeds, so the control is not a straw man and
is not structurally vacuous.

DV-SYMMETRY INVARIANCE (mandatory per-arm declaration)
  The DV is `fraction_in_external_task`, a function of argmax(operating_mode).
  Its symmetry group: invariant under (i) a constant added to ALL mode logits,
  and (ii) any strictly monotone transform applied identically to all mode
  logits.
  ARM_NORM -- manipulation is a symmetric clamp of each affinity input to
    [-cap, +cap] with cap = r * m_seed. NOT INVARIANT: the clamp is applied to
    the signal BEFORE per-mode weighting and `external_task_bias` is added
    afterwards and is never clamped, so clamping changes logit DIFFERENCES, not
    a common offset; and a clamp is not a monotone transform of the logits.
    Empirically non-invariant, not merely argued: 934 seed 42 moved
    0.5606 -> 0.0317 -> 0.0 across caps on this exact DV.
  ARM_ABS -- identical manipulation type at a fixed cap; same argument, and
    likewise empirically non-invariant (934 seed 43: 1.0 -> 0.4447).
  Neither arm is invariant under the DV's symmetry group, so neither delta is an
  arithmetic identity fixed before the run.

NON-DEGENERACY. `manipulation_landed` requires that occupancy OR the continuous
margin actually varies across the swept r values on some guard-passing seed. If
NEITHER moves, the per-seed cap override was inert and a C1 failure is an
instrument artefact -- routed `substrate_not_ready_requeue`, never a null.

PER-CLAIM DIRECTION: diagnostic, EXCLUDED from governance confidence/conflict
scoring. MECH-266 non_contributory (this run establishes whether the measurement
precondition can be met by a RULE; it does not itself measure over-binding).
SD-032a supports if a common normalised rule grades (the register is
recalibratable in a shippable way), weakens if the best rule cannot beat the
best absolute cap (recalibration is idiosyncratic), non_contributory if not
ready or the manipulation was inert.

RE-DERIVE BRAKE: RELEASED, on two independent routes, both recorded in the queue
entry. (1) The governing cluster autopsy
`failure_autopsy_mech266-464e-467e-cluster_2026-08-13` refuses "a naked
V3-EXQ-464f/467f re-queue -- another lettered iteration of the same question, at
the same AFFINITY_INPUT_CAP, behind the same min()-based gate" and explicitly
licenses the H1/H2/H3 portfolio; this run is a NEW number asking a NEW question
(is the recalibration a rule?), with a NEW manipulation (per-seed normalised
cap, never a fixed one), a NEW control arm, and the regime_occupancy_gate
primitive rather than any min()-across-the-sweep statistic. (2) The brake's own
named upstream substrate `mode-governance-engagement` has since LANDED
(salience_affinity_input_cap, ree-v3 9bcde4cb63; substrate_queue status
`implemented_pending_validation`), which is the documented brake-release
condition. This run is part of that entry's validation.

GOV-REUSE-1: the decisive readout is "occupancy at a per-seed cap DERIVED from
that seed's own baseline margin". Checked against the only compatible banked
source, V3-EXQ-934 (substrate_hash
f53db12dd0a7e00dcf351e3ba024c861173ac53d435d507d493288f8f138ddeb). It is
PARTIALLY recoverable and that is exactly why this run is scoped the way it is:
934's grid happens to place exactly ONE cell per seed near r ~ 2.2, which is what
made the hypothesis visible, but n=1 cell per seed at an uneven, undesigned r
grid, with the third seed at 0.9002 (0.0002 outside the band), is
hypothesis-GENERATING and cannot test a common rule -- and it contains no
out-of-sample seed at all. So this is the "downgrade to a minimal targeted run"
route: it adds only what is genuinely missing (a designed r axis, a matched
absolute control, and out-of-sample seeds), and re-derives nothing 934 banked.

claim_ids: MECH-266, SD-032a.
experiment_purpose: diagnostic
predecessor: V3-EXQ-934 (successor; NOT a supersede -- 934's H1 finding stands).
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _derive_env_seed,
    _sd049_kwargs,
    _sense_with_optional_harm,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.regime_occupancy_gate import (  # noqa: E402
    OccupancyCell,
    evaluate_regime_occupancy_gate,
)

EXPERIMENT_TYPE = "v3_exq_935_mech266_margin_normalised_cap_rule"
QUEUE_ID = "V3-EXQ-935"
CLAIM_IDS: List[str] = ["MECH-266", "SD-032a"]
EXPERIMENT_PURPOSE = "diagnostic"
PREDECESSOR = "V3-EXQ-934 (successor; NOT a supersede -- 934's H1 finding stands)"

# Same reasoning as 934's exemption, and it carries over unchanged because the
# readiness anchors here are the same two upstream gates plus one more of the
# same kind (G-calib). These are ordinary "did the curriculum train" /
# "does the drive engage" / "is the calibration statistic alive" gates, NOT the
# "positive control reproduces a known degenerate signature" pattern the
# V3-EXQ-778d reachability guard targets. Their thresholds are reachable BY
# CONSTRUCTION from values V3-EXQ-934 measured on this exact substrate (contact
# 3/3 vs a 2/3 floor; margins 0.337-0.785 and calibration margins
# 0.3373/0.7681/0.7850 vs a 0.05 floor). The 778d failure mode -- an unmeetable
# predicate that reports met=false forever and mislabels an instrument gap as a
# substrate verdict -- does not apply: when the gates PASS and the rule
# nonetheless fails to beat the best absolute cap, the run routes
# `cap_recalibration_is_seed_idiosyncratic`, which is a VALID finding on a test
# that genuinely ran, not a starved false-falsification.
ANCHOR_REACHABILITY_EXEMPT = (
    "Readiness anchors are ordinary upstream gates (curriculum-trained; "
    "drive-engages; calibration-statistic-alive), not degeneracy-reproduction "
    "anchors; every threshold was cleared by V3-EXQ-934 on this exact substrate "
    "(contact 3/3 vs 2/3; margins 0.337-0.785 and calibration margins "
    "0.3373/0.7681/0.7850 vs a 0.05 floor). A gates-pass-but-rule-fails read "
    "routes a VALID seed-idiosyncratic finding, never a starved "
    "false-falsification -- the 778d unmeetable-predicate mode does not apply."
)

# Seeds 42/43/44 are V3-EXQ-934's (R_STAR was derived from them; retained
# because 44 is the only seed that graded at NO absolute cap and dropping it
# would bias the comparison). 45/46 are NEW and carry the out-of-sample
# criterion C2 on their own.
SEEDS = [42, 43, 44, 45, 46]
DERIVATION_SEEDS = [42, 43, 44]          # R_STAR was fitted on these
OUT_OF_SAMPLE_SEEDS = [45, 46]           # C2 is scored on these ALONE
CONDITION_LABEL = "CURRICULUM_BUILT_NORMALISED_CAP_RULE"

# z_goal stream liveness (Experimental Recording Standard). goal_state.is_active()
# gating is exactly the mechanism the _clone_for_arm goal_state fix addresses.
_ZG = ZGoalStreamAccumulator()

MODE_NAMES = [
    "external_task",
    "internal_planning",
    "internal_replay",
    "offline_consolidation",
]
STICKY_MODE = "external_task"

# --- The calibration rule under test -------------------------------------
# CAP_REF must stay 0.75: the banked margins that produced R_STAR were measured
# at exactly this cap on exactly ARM_SYMMETRIC. Changing it silently invalidates
# the pre-registered R_STAR.
CAP_REF = 0.75
# r values swept per seed as cap = r * m_seed. R_STAR is IN this list, so the
# primary criterion costs no extra compute and the window shape comes free.
R_SWEEP: List[float] = [1.85, 2.05, 2.25, 2.45, 2.65]
# Pre-registered common rule point: the MEAN of the three graded/near-graded r
# values banked by 934 (2.224, 2.278, 2.229 -> 2.244), rounded. Deliberately the
# mean rather than a value picked to make a particular seed clear the band.
R_STAR = 2.25
# Best single absolute cap available from 934 (1/3 seeds graded plus a near-miss
# at 0.9002; cap=0.75 also scored 1/3 but with two hard-saturated seeds). The
# head-to-head control for C3.
CAP_ABS_CONTROL = 1.75
# Training-time cap: held at 464e/934's construction value so the trained
# substrate stays comparable to the banked reference. Only EVAL caps vary.
AFFINITY_INPUT_CAP_TRAIN = 2.0

ARM_NORM = "ARM_NORM"
ARM_ABS = "ARM_ABS"
ARM_CALIB = "ARM_CALIB"

# The null's mixed band: occupancy strictly inside (floor, ceiling).
OCCUPANCY_FLOOR = 0.10
OCCUPANCY_CEILING = 0.90
# G-margin / G-calib floor. Conservative, well below the argmax boundary.
MARGIN_FLOOR = 0.05

WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
MODE_EVAL_EPISODES = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0
# C2 (out-of-sample) bar. Lower than MIN_FRACTION because it is scored over only
# 2 seeds; >= 1/2 means at least one of the two new seeds must grade under the
# common rule. Scoped out entirely if neither out-of-sample seed is guard-passing.
OOS_MIN_FRACTION = 0.5


# --------------------------------------------------------------------------
# Harness below this line is REUSED VERBATIM from V3-EXQ-934's validated
# driver (scaffold config, substrate config, env build, arm clone, symmetric
# rails, quantile helper, and the per-cell frozen-policy eval). It is
# substrate plumbing, not science: keeping it byte-identical is what makes
# this run's occupancy readings directly comparable to 934's banked cells,
# which the pre-registered R_STAR is derived from.
# --------------------------------------------------------------------------
def _make_scaffold_cfg(dry_run: bool,
                       env_seed: Optional[int] = None) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_env_seed=env_seed,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """603n-validated foraging substrate + SalienceCoordinator + dACC + LateralPFC +
    bistable. use_closure_operator OFF (closure would inject a confounding
    mode-switch signal). salience_affinity_input_cap set to AFFINITY_INPUT_CAP_TRAIN
    (= 2.0, 464e's value) at TRAINING time; the EVAL cap is swept per cell by
    overriding coord.config.affinity_input_cap on the clone."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=False,
        use_external_task_drive=True,
        external_task_drive_affinity_weight=3.0,
        external_task_drive_salience_weight=2.0,
        external_task_drive_commit_weight=1.0,
        external_task_drive_proximity_weight=1.0,
        salience_affinity_input_cap=AFFINITY_INPUT_CAP_TRAIN,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_dual_cue_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig,
                       seed: Optional[int] = None) -> CausalGridWorldV2:
    """P2-config foraging env WITH the GAP-3 dual_cue primitive (competing goals),
    identical to 464e. `seed` default None passes through to CausalGridWorldV2's
    OS-entropy default -- bit-identical to the landed 464e/467e eval env layout, so
    the cap-sweep results stay comparable to the banked reference."""
    p2_hfa = (
        scaffold_cfg.scaffold_p2_hazard_food_attraction_guard
        if scaffold_cfg.scaffold_p2_hazard_food_attraction_guard >= 0.0
        else scaffold_cfg.scaffold_p2_hazard_food_attraction
    )
    return CausalGridWorldV2(
        seed=seed,
        size=scaffold_cfg.scaffold_env_size,
        num_hazards=scaffold_cfg.scaffold_p2_num_hazards,
        num_resources=scaffold_cfg.scaffold_p2_num_resources,
        hazard_food_attraction=p2_hfa,
        proximity_harm_scale=scaffold_cfg.scaffold_p2_proximity_harm_scale,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis=scaffold_cfg.scaffold_reef_bipartite_axis,
        reef_bipartite_agent_band_radius=scaffold_cfg.scaffold_reef_bipartite_agent_band_radius,
        reef_bipartite_agent_spawn_in_reef_half=False,
        dual_cue_enabled=True,
        dual_cue_min_active_ticks=10,
        dual_cue_replace_on_early_consume=False,
        dual_cue_type_tags=(1, 2),
        **_sd049_kwargs(scaffold_cfg),
    )


def _clone_for_arm(trained_agent: REEAgent, device: torch.device) -> REEAgent:
    """Clone the SAME trained weights into a fresh agent (rails + eval cap applied by
    the caller). Also clones goal_state -- the 464e fix: GoalState is a plain Python
    object, invisible to state_dict(), so a weights-only clone dropped its z_goal
    attractor and hard-gated external_task_drive engagement to 0.0 for the eval."""
    cfg = copy.deepcopy(trained_agent.config)
    agent = REEAgent(cfg).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent.load_state_dict(state)
    except RuntimeError:
        agent.load_state_dict(state, strict=False)
    agent.e3._running_variance = float(trained_agent.e3._running_variance)
    if trained_agent.goal_state is not None and agent.goal_state is not None:
        agent.goal_state.load_state_dict(trained_agent.goal_state.state_dict())
    return agent


def _apply_symmetric(coord) -> None:
    coord.config.enter_thresholds = {}
    coord.config.exit_thresholds = {}

def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)

def _eval_cap_cell(
    agent: REEAgent,
    env: CausalGridWorldV2,
    cap: float,
    arm_label: str,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_ep: int,
) -> Dict[str, Any]:
    """Frozen-policy eval for ONE (cap, arm) cell. Rails must already be applied by
    the caller; this sets the EVAL-time cap on the coordinator config (read live by
    tick()). Instruments BOTH the discrete committed-mode occupancy AND the
    continuous pre-argmax operating_mode['external_task'] margin, plus a
    MODE-CONDITIONED dwell in external_task specifically."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    coord = agent.salience
    # EVAL-time cap override (bit-live at tick(); no retraining). This is the sweep.
    coord.config.affinity_input_cap = float(cap)
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    coord_ticks_start = int(coord.diagnostics.get("n_ticks", 0))

    mode_step_counts = {m: 0 for m in MODE_NAMES}
    other_mode_steps = 0
    total_switches = 0
    total_steps = 0
    ext_margins: List[float] = []          # continuous operating_mode[external_task] per step
    # Mode-conditioned dwell: run-lengths measured only while current_mode ==
    # external_task (fixes 464e's M3 mode-agnostic dwell).
    ext_run_lengths: List[int] = []
    all_run_lengths: List[int] = []

    with torch.no_grad():
        for _ep in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            prev_mode = coord.current_mode
            current_run = 1
            ext_run = 1 if prev_mode == STICKY_MODE else 0

            for _ in range(steps_per_ep):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device, feed_harm
                )

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                # Continuous pre-argmax margin -- the probe's / H2's recommended
                # instrumentation. operating_mode is the last softmax vector.
                ext_margins.append(float(coord.operating_mode.get(STICKY_MODE, 0.0)))

                cur_mode = coord.current_mode
                if cur_mode in mode_step_counts:
                    mode_step_counts[cur_mode] += 1
                else:
                    other_mode_steps += 1

                if cur_mode != prev_mode:
                    all_run_lengths.append(current_run)
                    if prev_mode == STICKY_MODE and ext_run > 0:
                        ext_run_lengths.append(ext_run)
                    total_switches += 1
                    current_run = 1
                    ext_run = 1 if cur_mode == STICKY_MODE else 0
                    prev_mode = cur_mode
                else:
                    current_run += 1
                    if cur_mode == STICKY_MODE:
                        ext_run += 1

                total_steps += 1
                _, _harm, done, _info, obs_dict = env.step(action_idx)
                if done:
                    all_run_lengths.append(current_run)
                    if cur_mode == STICKY_MODE and ext_run > 0:
                        ext_run_lengths.append(ext_run)
                    current_run = 0
                    ext_run = 0
                    break

            if current_run > 0:
                all_run_lengths.append(current_run)
                if prev_mode == STICKY_MODE and ext_run > 0:
                    ext_run_lengths.append(ext_run)

    frac_task = mode_step_counts[STICKY_MODE] / total_steps if total_steps else 0.0
    mean_dwell = (
        float(sum(all_run_lengths)) / len(all_run_lengths)
        if all_run_lengths else float(steps_per_ep)
    )
    ext_dwell_mean = (
        float(sum(ext_run_lengths)) / len(ext_run_lengths)
        if ext_run_lengths else 0.0
    )
    margins_sorted = sorted(ext_margins)
    margin_mean = float(sum(ext_margins) / len(ext_margins)) if ext_margins else 0.0
    coord_ticks = int(coord.diagnostics.get("n_ticks", 0)) - coord_ticks_start

    return {
        "cap": float(cap),
        "arm": arm_label,
        "fraction_in_external_task": round(frac_task, 4),
        "ext_margin_mean": round(margin_mean, 4),
        "ext_margin_p10": round(_quantile(margins_sorted, 0.10), 4),
        "ext_margin_p50": round(_quantile(margins_sorted, 0.50), 4),
        "ext_margin_p90": round(_quantile(margins_sorted, 0.90), 4),
        "ext_margin_max": round(margins_sorted[-1], 4) if margins_sorted else 0.0,
        "n_switches": total_switches,
        "mean_dwell": round(mean_dwell, 3),
        "ext_dwell_mean": round(ext_dwell_mean, 3),
        "n_ext_runs": len(ext_run_lengths),
        "mode_step_counts": mode_step_counts,
        "other_mode_steps": other_mode_steps,
        "total_steps": total_steps,
        "coord_n_ticks": coord_ticks,
        "n_episodes": n_eps,
    }




def _graded(occ: float) -> bool:
    """The null's mixed band: occupancy STRICTLY inside (floor, ceiling)."""
    return bool(OCCUPANCY_FLOOR < float(occ) < OCCUPANCY_CEILING)


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "is_out_of_sample": bool(seed in OUT_OF_SAMPLE_SEEDS),
        "baseline_margin": 0.0,
        "calib_ok": False,
        "cells": [],
        "cap_at_r_star": 0.0,
        "occ_at_r_star": None,
        "graded_at_r_star": False,
        "occ_abs_control": None,
        "graded_abs_control": False,
        "graded_at_some_r": False,
        "graded_r_values": [],
        "r_regime": {},
        "max_margin_mean": 0.0,
        "margin_engaged": False,
        "rule_testable": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int,
              env_seed_base: Optional[int] = None) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_env_base = None if env_seed_base is None else int(env_seed_base) + int(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run, env_seed=seed_env_base)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else MODE_EVAL_EPISODES
    r_values = R_SWEEP[:2] if dry_run else R_SWEEP

    probe_env = _build_dual_cue_env(
        scaffold_cfg, seed=_derive_env_seed(seed_env_base, stream=2, idx=0)
    )
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0

    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
          f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(
        agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
          f" retention={s0b.retention_ratio:.3f}"
          f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" median_last={hz.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "hazard", hz.abort_reason)

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )
    _ZG.observe(agent)  # trained curriculum agent, after all training stages

    dual_env = _build_dual_cue_env(
        scaffold_cfg, seed=_derive_env_seed(seed_env_base, stream=2, idx=1)
    )
    dual_env.reset()

    cells: List[Dict[str, Any]] = []

    # -- 1. CALIBRATION CELL at CAP_REF -> this seed's baseline drive scale.
    # Same cap and same (symmetric) rails at which the banked margins that
    # produced R_STAR were measured; that identity is what makes R_STAR transfer.
    calib_agent = _clone_for_arm(agent, device)
    _apply_symmetric(calib_agent.salience)
    calib_cell = _eval_cap_cell(
        calib_agent, dual_env, CAP_REF, ARM_CALIB,
        scaffold_cfg, device, eval_eps, steps_per_ep,
    )
    calib_cell["r"] = None
    cells.append(calib_cell)
    done += eval_eps
    _ZG.observe(calib_agent)
    baseline_margin = float(calib_cell["ext_margin_mean"])
    calib_ok = bool(baseline_margin > MARGIN_FLOOR)
    print(f"  [calib] seed={seed} cap_ref={CAP_REF}"
          f" baseline_margin={baseline_margin:.4f} calib_ok={calib_ok}"
          f" occ={calib_cell['fraction_in_external_task']}", flush=True)

    # -- 2. ARM_NORM cells: cap = r * baseline_margin, one per swept r.
    # Run even when calib_ok is False: the cells are then degenerate BY
    # CONSTRUCTION (cap ~ 0), and recording that is more informative than a
    # silent skip -- G-calib is what routes the seed, not a missing row.
    for r in r_values:
        cap = float(r) * baseline_margin
        cell_agent = _clone_for_arm(agent, device)
        _apply_symmetric(cell_agent.salience)
        cell = _eval_cap_cell(
            cell_agent, dual_env, cap, ARM_NORM,
            scaffold_cfg, device, eval_eps, steps_per_ep,
        )
        cell["r"] = float(r)
        cell["is_r_star"] = bool(abs(float(r) - R_STAR) < 1e-9)
        cells.append(cell)
        done += eval_eps
        _ZG.observe(cell_agent)
        print(f"  [eval] seed={seed} {ARM_NORM} r={r} cap={cap:.4f}"
              f" occ={cell['fraction_in_external_task']}"
              f" graded={_graded(cell['fraction_in_external_task'])}"
              f" margin_mean={cell['ext_margin_mean']}"
              f" switches={cell['n_switches']}", flush=True)

    # -- 3. ARM_ABS control: the best single absolute cap from 934, identical
    # for every seed. This is C3's head-to-head comparator.
    abs_agent = _clone_for_arm(agent, device)
    _apply_symmetric(abs_agent.salience)
    abs_cell = _eval_cap_cell(
        abs_agent, dual_env, CAP_ABS_CONTROL, ARM_ABS,
        scaffold_cfg, device, eval_eps, steps_per_ep,
    )
    abs_cell["r"] = None
    cells.append(abs_cell)
    done += eval_eps
    _ZG.observe(abs_agent)
    print(f"  [eval] seed={seed} {ARM_ABS} cap={CAP_ABS_CONTROL}"
          f" occ={abs_cell['fraction_in_external_task']}"
          f" graded={_graded(abs_cell['fraction_in_external_task'])}"
          f" margin_mean={abs_cell['ext_margin_mean']}", flush=True)

    norm_cells = [c for c in cells if c["arm"] == ARM_NORM]
    star_cells = [c for c in norm_cells if c.get("is_r_star")]
    star_cell = star_cells[0] if star_cells else None

    occ_at_r_star = (
        float(star_cell["fraction_in_external_task"]) if star_cell is not None else None
    )
    graded_at_r_star = bool(star_cell is not None and _graded(occ_at_r_star))
    occ_abs = float(abs_cell["fraction_in_external_task"])
    graded_abs = _graded(occ_abs)

    graded_r_values = [
        float(c["r"]) for c in norm_cells
        if _graded(c["fraction_in_external_task"])
    ]
    # Information-only regime shape over the r axis. Uses the shared H3-fix
    # primitive for comparability with 934. Explicitly NOT load-bearing: a
    # per-seed best-r read is exactly the weakness this run exists to correct.
    r_gate = evaluate_regime_occupancy_gate(
        [
            OccupancyCell(label=f"r={c['r']}",
                          fraction=float(c["fraction_in_external_task"]))
            for c in norm_cells
        ],
        mode_label=STICKY_MODE,
        floor=OCCUPANCY_FLOOR, ceiling=OCCUPANCY_CEILING,
    )
    r_gate["graded_r_values"] = graded_r_values

    max_margin_mean = max((float(c["ext_margin_mean"]) for c in cells), default=0.0)
    margin_engaged = bool(max_margin_mean > MARGIN_FLOOR)
    rule_testable = bool(guard_pass and calib_ok)

    print(f"  [rule] seed={seed} baseline_margin={baseline_margin:.4f}"
          f" cap_at_r_star={R_STAR * baseline_margin:.4f}"
          f" occ_at_r_star={occ_at_r_star} graded_at_r_star={graded_at_r_star}"
          f" | abs_cap={CAP_ABS_CONTROL} occ={occ_abs} graded={graded_abs}"
          f" | r_shape={r_gate.get('regime_shape')}", flush=True)
    print(f"verdict: {'PASS' if (rule_testable and margin_engaged and graded_at_r_star) else 'FAIL'}"
          f" seed={seed} guard_pass={guard_pass} calib_ok={calib_ok}"
          f" margin_engaged={margin_engaged} graded_at_r_star={graded_at_r_star}"
          f" (contact_rate={p2.contact_rate:.4f}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f})", flush=True)

    return {
        "seed": seed,
        "aborted_at": None,
        "abort_reason": "",
        "guard_pass": guard_pass,
        "is_out_of_sample": bool(seed in OUT_OF_SAMPLE_SEEDS),
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_num_contact_events": int(p2.num_contact_events),
        "baseline_margin": round(baseline_margin, 4),
        "calib_ok": calib_ok,
        "rule_testable": rule_testable,
        "cells": cells,
        "cap_at_r_star": round(R_STAR * baseline_margin, 4),
        "occ_at_r_star": occ_at_r_star,
        "graded_at_r_star": graded_at_r_star,
        "occ_abs_control": occ_abs,
        "graded_abs_control": graded_abs,
        "graded_at_some_r": bool(graded_r_values),
        "graded_r_values": graded_r_values,
        "r_regime": r_gate,
        "max_margin_mean": round(max_margin_mean, 4),
        "margin_engaged": margin_engaged,
    }


def run_experiment(dry_run: bool = False,
                   env_seed_base: Optional[int] = None) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}, "
          f"env_seed_base={env_seed_base})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    r_values = R_SWEEP[:2] if dry_run else R_SWEEP
    # cells per seed = 1 calibration + len(r_values) ARM_NORM + 1 ARM_ABS
    n_cells = 1 + len(r_values) + 1
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + n_cells * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + n_cells * MODE_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps, env_seed_base=env_seed_base))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    # G-margin: the external_task drive must ENGAGE somewhere on this seed.
    margin_flags = [bool(r.get("margin_engaged", False)) for r in guard_passing]
    margin_frac = _frac(margin_flags)
    margin_ready_met = bool(margin_frac >= MIN_FRACTION)

    # G-calib: the calibration statistic must be alive, else cap = r * m_seed ~ 0
    # for every r and the whole ARM_NORM manipulation is degenerate.
    calib_flags = [bool(r.get("calib_ok", False)) for r in guard_passing]
    calib_frac = _frac(calib_flags)
    calib_ready_met = bool(calib_frac >= MIN_FRACTION)

    # Criteria are scored over seeds that are actually TESTABLE (guard-passing
    # AND calibration-alive). A seed whose calibration statistic is dead never
    # received the manipulation, so scoring it as "did not grade" would read a
    # starved cell as a falsification -- scope it out instead, and record n.
    testable = [r for r in guard_passing if r.get("calib_ok", False)]
    n_testable = len(testable)

    # -- C1: the COMMON RULE test. One r (R_STAR) for every seed.
    c1_flags = [bool(r.get("graded_at_r_star", False)) for r in testable]
    c1_frac = _frac(c1_flags)
    c1_passed = bool(n_testable > 0 and c1_frac >= MIN_FRACTION)
    n_graded_norm = sum(1 for f in c1_flags if f)

    # -- C2: out-of-sample generalisation, scored on seeds 45/46 ALONE.
    oos_testable = [r for r in testable if r.get("is_out_of_sample")]
    oos_flags = [bool(r.get("graded_at_r_star", False)) for r in oos_testable]
    oos_frac = _frac(oos_flags)
    c2_scoped_out = bool(len(oos_testable) == 0)
    c2_passed = bool((not c2_scoped_out) and oos_frac >= OOS_MIN_FRACTION)
    c2_scope_note = (
        "SCOPED OUT: no out-of-sample seed (45/46) was both guard-passing and "
        "calibration-alive, so the generalisation criterion was never exercised. "
        "A starved criterion is scoped out, never scored as a falsification; the "
        "PASS gate therefore does not require C2 in this run, and the "
        "circularity caveat on R_STAR remains UNDISCHARGED -- any successor "
        "must re-run the out-of-sample leg before the rule is treated as "
        "generalising."
        if c2_scoped_out else ""
    )

    # -- C3: strictly beat the best single absolute cap on the SAME seeds.
    abs_flags = [bool(r.get("graded_abs_control", False)) for r in testable]
    n_graded_abs = sum(1 for f in abs_flags if f)
    abs_frac = _frac(abs_flags)
    c3_passed = bool(n_testable > 0 and n_graded_norm > n_graded_abs)

    # Information ONLY -- the per-seed best-r read. Explicitly not load-bearing;
    # reported so a narrow C1 miss can be told apart from a wrong rule.
    some_r_flags = [bool(r.get("graded_at_some_r", False)) for r in testable]
    some_r_frac = _frac(some_r_flags)

    # Non-degeneracy: did the per-seed normalised cap manipulation LAND at all?
    # Occupancy OR the continuous margin must vary across the swept r on some
    # testable seed. If NEITHER moves, the override was inert and a C1 failure
    # is an instrument artefact, not a null.
    def _varies(key: str) -> bool:
        for r in testable:
            vals = [
                float(c[key]) for c in r.get("cells", []) if c["arm"] == ARM_NORM
            ]
            if len(vals) >= 2 and (max(vals) - min(vals)) > 1e-6:
                return True
        return False

    occupancy_varies = _varies("fraction_in_external_task")
    margin_varies = _varies("ext_margin_mean")
    manipulation_landed = bool(occupancy_varies or margin_varies)

    # PASS gate: plain AND of C1 and C3, plus C2 unless it was scoped out.
    combination_rule = (
        "PASS iff C1_rule_grades_at_r_star AND C3_beats_best_absolute_cap AND "
        "(C2_rule_generalises_out_of_sample OR C2 scoped out for want of a "
        "testable out-of-sample seed). Plain AND -- recorded explicitly so the "
        "gate is never inferred from the per-criterion booleans alone."
    )
    rule_supported = bool(c1_passed and c3_passed and (c2_passed or c2_scoped_out))

    if not contact_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
    elif not margin_ready_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "external_task_drive_not_engaging"
    elif not calib_ready_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "calibration_statistic_dead_cap_rule_degenerate"
    elif not manipulation_landed:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "normalised_cap_manipulation_inert_verify_instrument"
    elif rule_supported:
        outcome = "PASS"
        readiness_route = "normalised_cap_rule_admits_common_mixed_regime"
        route_reason = "single_pre_registered_r_grades_across_seeds_and_beats_best_absolute_cap"
    else:
        outcome = "FAIL"
        readiness_route = "cap_recalibration_is_seed_idiosyncratic"
        route_reason = "no_common_normalised_rule_outperformed_the_best_absolute_cap"

    # Diagnostic: excluded from confidence scoring. Directions are context only.
    if rule_supported:
        sd032a_dir = "supports"          # register recalibratable by a shippable rule
    elif contact_non_vacuity_met and margin_ready_met and calib_ready_met and manipulation_landed:
        sd032a_dir = "weakens"           # ready + manipulation landed, but no rule generalises
    else:
        sd032a_dir = "non_contributory"  # not ready / inert; says nothing about the register
    direction_map = {
        "MECH-266": "non_contributory",
        "SD-032a": sd032a_dir,
    }
    overall_direction = "non_contributory"

    print(f"[{EXPERIMENT_TYPE}] guard {sum(guard_flags)}/{n}"
          f" margin_ready={margin_ready_met} calib_ready={calib_ready_met}"
          f" n_testable={n_testable}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] C1 graded_at_r_star={n_graded_norm}/{n_testable}"
          f" (frac={c1_frac:.3f}) passed={c1_passed}"
          f" | C2 oos={sum(1 for f in oos_flags if f)}/{len(oos_testable)}"
          f" passed={c2_passed} scoped_out={c2_scoped_out}"
          f" | C3 norm={n_graded_norm} vs abs={n_graded_abs} passed={c3_passed}",
          flush=True)
    print(f"[{EXPERIMENT_TYPE}] (info only, NOT load-bearing)"
          f" graded_at_some_r_frac={some_r_frac:.3f}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] -> outcome={outcome} route={readiness_route}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "margin_ready_met": margin_ready_met,
        "margin_ready_fraction": margin_frac,
        "calib_ready_met": calib_ready_met,
        "calib_ready_fraction": calib_frac,
        "n_testable_seeds": n_testable,
        "c1_rule_grades_at_r_star": c1_passed,
        "c1_fraction": c1_frac,
        "c1_n_graded": n_graded_norm,
        "c2_rule_generalises_out_of_sample": c2_passed,
        "c2_scoped_out": c2_scoped_out,
        "c2_scope_note": c2_scope_note,
        "c2_fraction": oos_frac,
        "c2_n_out_of_sample_testable": len(oos_testable),
        "c3_beats_best_absolute_cap": c3_passed,
        "c3_n_graded_normalised": n_graded_norm,
        "c3_n_graded_absolute": n_graded_abs,
        "c3_absolute_fraction": abs_frac,
        "rule_supported": rule_supported,
        "combination_rule": combination_rule,
        "graded_at_some_r_fraction_INFO_ONLY": some_r_frac,
        "occupancy_varies_across_r": occupancy_varies,
        "margin_varies_across_r": margin_varies,
        "manipulation_landed": manipulation_landed,
        "route_reason": route_reason,
        "per_seed_guard_pass": guard_flags,
        "per_seed_calib_ok": [bool(r.get("calib_ok", False)) for r in per_seed],
        "per_seed_baseline_margin": [r.get("baseline_margin", 0.0) for r in per_seed],
        "per_seed_cap_at_r_star": [r.get("cap_at_r_star", 0.0) for r in per_seed],
        "per_seed_occ_at_r_star": [r.get("occ_at_r_star") for r in per_seed],
        "per_seed_graded_at_r_star": [bool(r.get("graded_at_r_star", False)) for r in per_seed],
        "per_seed_occ_abs_control": [r.get("occ_abs_control") for r in per_seed],
        "per_seed_graded_abs_control": [bool(r.get("graded_abs_control", False)) for r in per_seed],
    }

    preconditions = [
        {
            "name": "foraging_contact_guard",
            "kind": "readiness",
            "description": "603n G2+G3 contact guard on >= 2/3 seeds. A curriculum "
                           "that never became foraging-competent makes every "
                           "occupancy reading meaningless.",
            "control": "fraction of seeds with P2 contact_rate > 0 AND "
                       "z_goal_norm_at_contact_peak > 0.4. Cleared 3/3 by "
                       "V3-EXQ-934 on this exact curriculum.",
            "measured": round(guard_frac, 4),
            "threshold": MIN_FRACTION,
            "direction": "lower",
            "met": contact_non_vacuity_met,
        },
        {
            "name": "external_task_drive_engages",
            "kind": "readiness",
            "description": "the external_task drive must ENGAGE at SOME cell -- "
                           "per-seed max-over-cells of the CONTINUOUS "
                           "operating_mode['external_task'] margin > MARGIN_FLOOR "
                           "-- on >= 2/3 guard-passing seeds. If the margin is ~0 "
                           "everywhere the drive is not producing the signal at all "
                           "(substrate/wiring), which must self-route to "
                           "substrate_not_ready_requeue and NOT be read as a "
                           "structural or idiosyncratic finding.",
            "control": "fraction of guard-passing seeds whose best cell's "
                       "ext_margin_mean clears MARGIN_FLOOR. V3-EXQ-934 measured "
                       "0.337-0.785 on this substrate against a 0.05 floor.",
            "measured": round(margin_frac, 4),
            "threshold": MIN_FRACTION,
            "direction": "lower",
            "met": margin_ready_met,
        },
        {
            "name": "calibration_statistic_alive",
            "kind": "readiness",
            "description": "the calibration cell at CAP_REF must yield a live "
                           "baseline margin (m_seed > MARGIN_FLOOR) on >= 2/3 "
                           "guard-passing seeds. This is the readiness form of the "
                           "SAME quantity the load-bearing C1 criterion routes on: "
                           "C1 reads occupancy under cap = R_STAR * m_seed, so a "
                           "dead m_seed makes the cap ~0 for every r and STARVES C1 "
                           "rather than falsifying it. Below floor self-routes "
                           "substrate_not_ready_requeue, never "
                           "cap_recalibration_is_seed_idiosyncratic.",
            "control": "the calibration cell is a known-non-degenerate positive "
                       "control: it is the SAME cap (0.75) and SAME symmetric rails "
                       "at which V3-EXQ-934 measured m = 0.3373 / 0.7681 / 0.7850 "
                       "on seeds 42/43/44, all ~7-16x the 0.05 floor.",
            "measured": round(calib_frac, 4),
            "threshold": MIN_FRACTION,
            "direction": "lower",
            "met": calib_ready_met,
        },
    ]

    criteria = [
        {"name": "C1_rule_grades_at_r_star", "load_bearing": True, "passed": c1_passed},
        {"name": "C2_rule_generalises_out_of_sample", "load_bearing": True,
         "passed": c2_passed, "scoped_out": c2_scoped_out,
         "scope_note": c2_scope_note},
        {"name": "C3_beats_best_absolute_cap", "load_bearing": True, "passed": c3_passed},
    ]

    # A criterion is a meaningful test only if readiness held AND the normalised
    # cap manipulation actually landed. C2 is additionally non-degenerate only
    # when at least one out-of-sample seed was testable.
    base_non_degenerate = bool(
        contact_non_vacuity_met and margin_ready_met and calib_ready_met
        and manipulation_landed and n_testable > 0
    )
    criteria_non_degenerate = {
        "C1_rule_grades_at_r_star": base_non_degenerate,
        "C2_rule_generalises_out_of_sample": bool(base_non_degenerate and not c2_scoped_out),
        "C3_beats_best_absolute_cap": base_non_degenerate,
    }

    return {
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": direction_map,
        "acceptance": acceptance,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "route_reason": route_reason,
            "hypothesis": "H-RULE: a SINGLE pre-registered r, applied per-seed as "
                          "cap = r * baseline_margin(seed), yields a graded "
                          "external_task occupancy regime on >= 2/3 of seeds "
                          "simultaneously -- i.e. the V3-EXQ-934 cap recalibration "
                          "is a shippable RULE, not per-seed hand-tuning.",
            "null": "H-IDIO: no single normalised r grades on more seeds than the "
                    "best single absolute cap (1.75, which graded 1/3 in 934). The "
                    "required cap is seed-idiosyncratic and there is no rule to ship.",
            "aliasing_control": "H-KNIFE (the rule is right but the graded r window "
                                "is too narrow for any fixed r to be robust) is "
                                "separated from H-IDIO by the R_SWEEP: "
                                "graded_at_some_r_fraction and the per-seed "
                                "graded_r_values report the window, so a narrow C1 "
                                "miss is distinguishable from a wrong rule. Those "
                                "readouts are INFORMATION ONLY and deliberately not "
                                "load-bearing -- scoring on each seed's own best r "
                                "would reproduce exactly the per-seed best-point "
                                "weakness this run exists to correct.",
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
            "circularity": {
                "statement": "R_STAR = 2.25 was derived from V3-EXQ-934's banked "
                             "seeds 42/43/44 (mean of the observed graded r values "
                             "2.224 / 2.278 / 2.229 = 2.244, rounded). Scoring the "
                             "rule on those same seeds is partly circular.",
                "control": "C2 is scored on seeds 45/46 ALONE, which were not used "
                           "to derive R_STAR. C2 is load-bearing whenever at least "
                           "one of them is testable.",
                "derivation_seeds": DERIVATION_SEEDS,
                "out_of_sample_seeds": OUT_OF_SAMPLE_SEEDS,
            },
            "dv_symmetry": {
                "dv": "fraction_in_external_task -- a function of "
                      "argmax(operating_mode).",
                "symmetry_group": "invariant under (i) a constant added to ALL mode "
                                  "logits and (ii) any strictly monotone transform "
                                  "applied identically to all mode logits.",
                "ARM_NORM": "NOT INVARIANT. The manipulation is a symmetric clamp of "
                            "each affinity input to [-cap, +cap] with "
                            "cap = r * m_seed, applied BEFORE per-mode weighting, "
                            "while external_task_bias is added afterwards and is "
                            "never clamped -- so it changes logit DIFFERENCES, not a "
                            "common offset, and a clamp is not a monotone transform "
                            "of the logits. Empirically non-invariant on this exact "
                            "DV: V3-EXQ-934 seed 42 moved 0.5606 -> 0.0317 -> 0.0 "
                            "across caps.",
                "ARM_ABS": "NOT INVARIANT. Identical manipulation type at a fixed "
                           "cap; same argument, and likewise empirically "
                           "non-invariant (V3-EXQ-934 seed 43: 1.0 -> 0.4447).",
            },
            "structural_satisfiability": {
                "checked": "on paper before queuing, against values V3-EXQ-934 "
                           "measured on this exact substrate.",
                "foraging_contact_guard": "3/3 measured vs a 2/3 floor -- satisfiable.",
                "external_task_drive_engages": "0.337-0.785 measured vs a 0.05 floor "
                                               "-- satisfiable.",
                "calibration_statistic_alive": "0.3373 / 0.7681 / 0.7850 measured at "
                                               "exactly CAP_REF on exactly symmetric "
                                               "rails vs a 0.05 floor -- satisfiable.",
                "C1": "satisfiable but NOT trivially so -- 2 of the 3 derivation "
                      "seeds graded at r ~ 2.2 in the banked data.",
                "C3": "the control is not a straw man and not structurally vacuous "
                      "-- ARM_ABS at cap 1.75 genuinely graded on 1/3 seeds in 934, "
                      "so it CAN grade and C3 requires strictly beating it.",
                "no_arm_structurally_vacuous": True,
            },
            "regime_gate": {
                "definition": "occupancy is 'graded' iff strictly inside "
                              "(OCCUPANCY_FLOOR, OCCUPANCY_CEILING). The primary "
                              "criteria evaluate this at the SINGLE pre-registered "
                              "R_STAR -- a common-rule test. The shared primitive "
                              "experiments/_lib/regime_occupancy_gate.py is applied "
                              "over the r axis for the INFORMATION-ONLY window "
                              "shape; never a min-across-the-sweep statistic, and "
                              "never as the load-bearing read.",
                "occupancy_floor": OCCUPANCY_FLOOR,
                "occupancy_ceiling": OCCUPANCY_CEILING,
                "margin_floor": MARGIN_FLOOR,
                "cap_ref": CAP_REF,
                "r_sweep": R_SWEEP,
                "r_star": R_STAR,
                "cap_abs_control": CAP_ABS_CONTROL,
                "min_fraction": MIN_FRACTION,
                "oos_min_fraction": OOS_MIN_FRACTION,
            },
            "contact_guard": {
                "definition": "per-seed P2 contact_rate > 0 AND "
                              "z_goal_norm_at_contact_peak > 0.4; < 2/3 seeds -> "
                              "substrate_not_ready_requeue.",
                "min_fraction": MIN_FRACTION,
                "p2_zgoal_gate": P2_ZGOAL_GATE,
                "contact_gate": CONTACT_GATE,
            },
        },
        "per_seed": per_seed,
    }


def main(dry_run: bool = False,
         env_seed_base: Optional[int] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = run_experiment(dry_run=dry_run, env_seed_base=env_seed_base)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)

    full_config = {
        "cap_ref": CAP_REF,
        "r_sweep": R_SWEEP,
        "r_star": R_STAR,
        "cap_abs_control": CAP_ABS_CONTROL,
        "affinity_input_cap_train": AFFINITY_INPUT_CAP_TRAIN,
        "occupancy_floor": OCCUPANCY_FLOOR,
        "occupancy_ceiling": OCCUPANCY_CEILING,
        "margin_floor": MARGIN_FLOOR,
        "arms": [ARM_CALIB, ARM_NORM, ARM_ABS],
        "rails": "ARM_SYMMETRIC (legacy MECH-259, no per-mode rails) on EVERY cell",
        "mode_eval_episodes_per_cell": MODE_EVAL_EPISODES,
        "train_steps": TRAIN_STEPS,
        "seeds": SEEDS,
        "derivation_seeds": DERIVATION_SEEDS,
        "out_of_sample_seeds": OUT_OF_SAMPLE_SEEDS,
        "min_fraction": MIN_FRACTION,
        "oos_min_fraction": OOS_MIN_FRACTION,
        "p2_zgoal_gate": P2_ZGOAL_GATE,
        "contact_gate": CONTACT_GATE,
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "n_resource_types": N_RESOURCE_TYPES,
            "config_basis": "V3-EXQ-603n",
        },
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "env_seed_base": env_seed_base,
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum; 603n config) + "
                     "SalienceCoordinator (SD-032a) + mode-governance-engagement "
                     "external_task drive (use_external_task_drive=True) + GAP-3 "
                     "dual_cue competing-goal env + goal_state clone fix + "
                     "salience_affinity_input_cap (trained at 2.0; EVAL cap set "
                     "PER SEED as r * baseline_margin on clones, plus a fixed "
                     "absolute control cap of 1.75). Symmetric rails on every cell. "
                     "use_closure_operator OFF.",
        "condition": CONDITION_LABEL,
        "predecessor": PREDECESSOR,
        "gov_fanout": "Successor to GOV-FANOUT-1 leg H1 (V3-EXQ-934, PASS). H1 "
                      "established that a mixed regime is REACHABLE; this run asks "
                      "whether reaching it is a shippable RULE or per-seed "
                      "hand-tuning -- the question 934's winning_cap_band "
                      "[0.75, 1.75] does NOT answer, because no single absolute cap "
                      "in that band graded on more than 1 of 3 seeds.",
        "method_note": "Per seed: train ONE curriculum agent, then run frozen-policy "
                       "eval cells on clones. (1) A calibration cell at CAP_REF=0.75 "
                       "on symmetric rails yields that seed's baseline external_task "
                       "margin m_seed -- the same cap/rails at which the banked "
                       "margins that produced R_STAR were measured, which is what "
                       "makes R_STAR transfer. (2) ARM_NORM cells at "
                       "cap = r * m_seed for r in [1.85, 2.05, 2.25, 2.45, 2.65]. "
                       "(3) An ARM_ABS control at a fixed cap of 1.75, the strongest "
                       "single absolute cap available from 934. The cap is read live "
                       "at SalienceCoordinator.tick() so no retraining is needed "
                       "(the same train-once/sweep-on-clones pattern 934 and 467e "
                       "use). Primary criteria evaluate the SINGLE pre-registered "
                       "R_STAR across seeds -- a common-rule test -- never each "
                       "seed's own best r, which is the per-seed best-point weakness "
                       "this run exists to correct.",
        "pre_registered_thresholds": {
            "cap_ref": CAP_REF,
            "r_sweep": R_SWEEP,
            "r_star": R_STAR,
            "r_star_derivation": "mean of the graded/near-graded r values banked by "
                                 "V3-EXQ-934 (2.224 / 2.278 / 2.229 = 2.244), "
                                 "rounded to 2.25. Deliberately the mean, not a "
                                 "value picked to make any seed clear the band.",
            "cap_abs_control": CAP_ABS_CONTROL,
            "affinity_input_cap_train": AFFINITY_INPUT_CAP_TRAIN,
            "occupancy_floor": OCCUPANCY_FLOOR,
            "occupancy_ceiling": OCCUPANCY_CEILING,
            "margin_floor": MARGIN_FLOOR,
            "min_fraction": MIN_FRACTION,
            "oos_min_fraction": OOS_MIN_FRACTION,
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
        },
        "anchor_reachability_exempt": ANCHOR_REACHABILITY_EXEMPT,
        "config": full_config,
        "stage_plan": stage_plan(),
    }
    manifest.update(result)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
        started_at=t0,
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--env-seed", type=int, default=None,
        help="Opt-in env-seed base. Omitted (the default) reproduces V3-EXQ-934's "
             "OS-entropy env seeding behaviour. Set it and every env this run "
             "builds is deterministically seeded. A pinned run is NOT comparable to "
             "an unpinned one.",
    )
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run, env_seed_base=args.env_seed)
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_res["manifest_path"],
            dry_run=bool(args.dry_run),
        )
