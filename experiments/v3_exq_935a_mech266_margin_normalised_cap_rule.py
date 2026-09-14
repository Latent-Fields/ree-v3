"""
V3-EXQ-935a (MECH-266 / SD-032a, corrected re-test of V3-EXQ-935): is the
external_task cap recalibration a SHIPPABLE RULE or SEED-IDIOSYNCRATIC?
DIAGNOSTIC.

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).

WHY THIS RUN EXISTS
--------------------
V3-EXQ-935 (FAIL, 2026-08-17, self-route `cap_recalibration_is_seed_idiosyncratic`)
was CONFIRMED-AUTOPSIED (`REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-935_2026-08-18.md`,
also `.json`) as a pure MEASUREMENT / instrument-defect failure, not a substrate
ceiling. 935's own sweep -- R_SWEEP = [1.85, 2.05, 2.25, 2.45, 2.65], seeds
42-46 -- shows the rule DOES grade cleanly once the pre-registered r is moved:

    r     C1 n/5   C2 oos   C3
    1.85  1/5      0/2      1>0 pass
    2.05  1/5      0/2      1>0 pass
    2.25  2/5      1/2      2>0 pass   <- 935's R_STAR, FAIL
    2.45  4/5      2/2      4>0 pass   <- PASS
    2.65  4/5      2/2      4>0 pass   <- PASS

935's own R_STAR = 2.25 was imported from V3-EXQ-934's coarse 5-point absolute
cap grid (two graded caps at the grid's ends, divided by seed margins that
happened to cluster) and did not survive the 3-commit substrate move between
934 and 935 (935's own `substrate_hash` differs from 934's). 935's finer,
purpose-built sweep puts the graded window's LOWER edge at r ~ 2.45 -- above
the entire imported band. This run fixes the ONE thing that was actually
wrong (an imported, non-reproducing constant), and re-earns the result on
genuinely fresh seeds, per the autopsy's Section 9 routing (confirmed at the
Step 8 interactive gate, 2026-08-18).

THE FOUR MEASUREMENT DEFECTS THIS RUN FIXES (autopsy Section 5a; all four
are driver-only, not substrate)
-----------------------------------------------------------------------
1. An aliasing control (the r-sweep) was computed but never wired into the
   routing -- 935 had a bare `else` fall-through from C1-fail straight to
   H-IDIO, with no H-KNIFE branch, even though the sweep it built for exactly
   this purpose showed r=2.45/2.65 clearing the bar. FIXED below: an explicit
   H-KNIFE branch, evaluated with the CORRECTED (2026-09-11,
   `experiments/_lib/regime_occupancy_gate.py`, ree-v3 606aea2) shared
   primitive `classify_regime_shape` / `evaluate_regime_occupancy_gate`,
   called ONCE over every (seed, r) cell -- never per-seed (935's own
   per-seed info-only call was the exact call shape the module's docstring
   now names as the source of the 934 false-positive; this run uses the
   corrected call shape throughout, including for the informational
   whole-sweep read).
2. `route_reason` was hardcoded per branch and was factually false on 935
   (it asserted "no rule beat the best absolute cap" while C3 was TRUE).
   FIXED below: `route_reason` is built from the actual criterion booleans
   at verdict time, never a fixed string per branch.
3. `R_STAR` was imported from V3-934 without a substrate_hash check. FIXED:
   R_STAR = 2.45 is now pre-registered from 935's OWN sweep, and the
   readiness-anchor exemption below is gated on 935's actual recorded
   `substrate_hash` rather than a prose assertion of identity (item 6).
4. `ANCHOR_REACHABILITY_EXEMPT` asserted "this exact substrate" as prose;
   935 in fact ran 3 `ree_core` commits apart from 934 despite the claim.
   FIXED below: the exemption records 935's measured `substrate_hash`
   (`921d0af6b6b6c54b12baafb090984edd9fcc2c3f5bf4ed2ba84d36f6c040b99b`) as a
   literal comparison value and the manifest records whether THIS run's own
   substrate_hash matches it, rather than asserting a claim no code checks.

THE HYPOTHESIS THIS RUN TESTS (unchanged from 935 -- same scientific question)
-------------------------------------------------------------------------------
  H-RULE (this run): a SINGLE pre-registered r, applied per-seed as
      cap = r * baseline_margin(seed), yields a graded regime on >= 2/3 of
      FRESH seeds SIMULTANEOUSLY -- i.e. the recalibration is a shippable
      rule that generalises out of sample.
  H-IDIO (the null): no single r in the sweep grades on >= 2/3 of these
      fresh seeds. The required cap is seed-idiosyncratic; there is no rule
      to ship.
  H-KNIFE (aliasing control, now WIRED into the routing -- see fix (1)
      above): C1 fails at the single pre-registered R_STAR, but some OTHER
      common r in the sweep clears the bar on these same fresh seeds. This
      means the rule-vs-idiosyncrasy question is not yet answered and a
      further-corrected re-queue (not H-IDIO) is the right route.

R_STAR IS PRE-REGISTERED AT 2.45 -- the lower edge of the graded window
observed in V3-EXQ-935's own five-seed sweep (seeds 42-46). Stated openly:
this value is drawn FROM 935 and is therefore IN-SAMPLE for seeds 42-46,
which is exactly why this run uses ENTIRELY DIFFERENT, FRESH seeds (47-51)
-- see "FRESH SEEDS" below. R_STAR is the FIRST sweep point in this run at
which C1 passed in 935 (4/5 at 2.45, same as 2.65) -- the lower edge, chosen
over the higher-power 2.65 point because it is the more conservative
pre-registration (closer to the observed transition, per the same
"deliberately not the value picked to look best" discipline 935 used for its
own R_STAR).

R_SWEEP EXTENDED UPWARD -- [2.25, 2.45, 2.65, 2.85, 3.05]. 935's sweep topped
out at 2.65, which still graded 4/5 -- the upper edge of the graded window
was never measured, and 935's seed 42 (flat at occupancy 1.000 across the
ENTIRE 935 sweep, n_switches=0, margin falling monotonically but the
arbitration threshold never crossed) needs r > 2.65, untested by 935. This
sweep keeps r=2.25 (935's old, now-refuted R_STAR, retained as a direct
comparison point) and adds 2.85 / 3.05 so a seed-42-class agent gets a
tested r and the window's upper edge is actually measured.

FRESH, ENTIRELY OUT-OF-SAMPLE SEEDS -- 47, 48, 49, 50, 51
-----------------------------------------------------------
935's own seeds were 42-46: three (42/43/44) inherited from 934 (as R_STAR's
DERIVATION seeds for a different, refuted R_STAR) and two (45/46) that were
new relative to 934 but were themselves used to help characterise 935's own
sweep (R_STAR=2.45 for THIS run is drawn from all five of 935's seeds
42-46's occupancy-vs-r curve, not from 42-44 alone). So R_STAR=2.45 is
in-sample for the FULL 42-46 set, and the common-rule claim must be re-earned
on seeds that did not choose it in ANY way. Unlike 935 (which mixed
derivation and out-of-sample seeds within one run), this run uses seeds that
played NO role at all in deriving R_STAR=2.45 or in shaping the extended
sweep -- a clean, fully out-of-sample test, at higher power (5 fresh seeds
vs 935's 2 genuinely-fresh-relative-to-934 seeds).

DESIGN SIMPLIFICATION FROM 935 (stated openly): because every seed here is
equally, fully out-of-sample (there is no partial-derivation subset the way
942-46 split into 42-44 vs 45-46), the separate in-run "C2 out-of-sample"
criterion 935 scored on a 2-seed subset is now REDUNDANT with the whole-run
common-rule test -- scoring it on the identical 5-seed population as C1, at
a DIFFERENT (weaker) bar, would not test anything C1 does not already test
more strongly, and keeping it would just be a confusing echo. This run
therefore uses TWO load-bearing criteria (renamed C1, C2; 935's C3 becomes
this run's C2) rather than three. The out-of-sample generalisation question
935's C2 existed to ask is answered by this run's design as a whole (every
seed is fresh), not by a criterion nested inside it.

DESIGN (unchanged harness from 935, itself REUSED VERBATIM from V3-EXQ-934's
validated driver)
-------------------------------------------------------------------------------
Per seed, ONE trained curriculum agent (identical scaffold to 934/935), then
frozen-policy eval cells on clones. ALL cells use ARM_SYMMETRIC rails.

  1. CALIBRATION CELL at CAP_REF = 0.75 -> baseline_margin m_seed. Unchanged
     from 935 -- CAP_REF must stay 0.75 because 935's own r=2.45/2.65 graded
     window was measured with baseline margins taken at this exact cap.
  2. ARM_NORM cells: one per r in R_SWEEP = [2.25, 2.45, 2.65, 2.85, 3.05],
     each at cap = r * m_seed.
  3. ARM_ABS control: cap = CAP_ABS_CONTROL = 1.75, fixed, identical for
     every seed -- unchanged from 934/935, still the strongest single
     absolute cap on record.

DEPENDENT VARIABLE (per cell): `fraction_in_external_task`, unchanged from
934/935. Continuous pre-argmax margin, mode-conditioned dwell, switch
counts, and coordinator tick count all recorded per cell as before.

CRITERIA (pre-registered; all thresholds are constants in this file)
  C1 RULE_GRADES_AT_R_STAR   [load-bearing] -- at the SINGLE r = R_STAR
     (2.45), ARM_NORM occupancy is in (0.1, 0.9) on >= MIN_FRACTION (2/3) of
     guard-passing seeds, ALL of them fresh (47-51). A COMMON-RULE test on a
     genuinely out-of-sample population -- this is what 935's C1+C2 wanted
     to be jointly and could not, because 935's population was partly
     in-sample.
  C2 BEATS_BEST_ABSOLUTE_CAP [load-bearing] -- strictly more guard-passing
     seeds grade under ARM_NORM at R_STAR than under ARM_ABS at 1.75, same
     seeds, same run. (935's C3, renamed for the two-criterion design.)

  PASS iff C1 AND C2. The combination rule is a plain AND, recorded as
  `combination_rule` in the manifest.

H-KNIFE ROUTING (NEW -- fix (1) above). If C1 FAILS at R_STAR, the run does
NOT fall through to H-IDIO by default. It checks, over the SAME aggregate
call used for the informational whole-sweep read (see REGIME SHAPE below),
whether any OTHER r in R_SWEEP clears the seed-reproducibility bar (>= 2/3
of these fresh seeds mixed at that r). If yes: route `rule_right_r_wrong_requeue`
(H-KNIFE) -- the rule form is supported but R_STAR itself needs a further
correction, and this is NOT the same finding as "no rule exists" (H-IDIO).
If no r in the sweep clears the bar for ANY seed count: route
`cap_recalibration_is_seed_idiosyncratic` (H-IDIO) -- now a genuine finding,
because every r in an EXTENDED sweep (up to 3.05, past 935's top of 2.65)
was checked and none reproduced.

REGIME SHAPE (informational, using the CORRECTED shared primitive, called
ONCE over ALL (seed, r) ARM_NORM cells -- never per-seed, which is the
exact call shape that produced 934's false positive per the module's own
docstring). `experiments/_lib/regime_occupancy_gate.classify_regime_shape`
implements the ">= 2 consecutive swept values, each mixed on >= 2/3 of
seeds" reproducibility bar (ree-v3 606aea2, 2026-09-11) -- the SAME bar
`mode-governance-engagement`'s own open failure_record targets (V3-EXQ-467e,
V3-EXQ-934) specify. This informs the H-KNIFE check above and is recorded
as `interpretation.occupancy_gate`, never load-bearing on its own (C1/C2
remain the pre-registered, load-bearing criteria).

READINESS GATES (route a not-ready read to substrate_not_ready_requeue,
NEVER a false verdict) -- unchanged in KIND from 934/935 (G-contact,
G-margin, G-calib, G-rstar), all SEED-scoped and aggregated as a FRACTION
over seeds, never AND-ed across seeds (the V3-EXQ-785 vacating defect, in
its seed-level form).

STRUCTURAL SATISFIABILITY. Unlike 935 (whose seeds 42-44 had banked V3-934
data at the exact readiness values), this run's seeds (47-51) have NEVER
run before -- there is no banked per-seed data to cite. Satisfiability is
argued from the curriculum's track record instead: the IDENTICAL
`scaffolded_sd054_onboarding` curriculum + readiness thresholds cleared
G-contact/G-margin/G-calib on 5/5 distinct seeds across 934 (42-44) and 935
(45-46), with margins spanning 0.3217-0.8139 against a 0.05 floor and
contact 5/5 against a 2/3 floor -- there is no known seed-dependent failure
mode in this curriculum construction. This is NOT the same evidentiary
strength as 935's "cleared by 934 on this exact substrate" claim (which the
autopsy found to be factually imprecise about substrate identity) -- it is
explicitly a track-record argument, not a per-seed proof, and the readiness
gates below are exactly what catches a seed for which the argument does not
hold.

DV-SYMMETRY INVARIANCE. Unchanged from 935 -- the DV `fraction_in_external_task`
is a function of argmax(operating_mode); ARM_NORM's clamp-before-weighting,
unclamped-bias-after mechanism is empirically non-invariant under both of
the DV's symmetry operations (935's seed 42: 0.5606 -> 0.0317 -> 0.0 across
absolute caps in 934's own banked data; the mechanism is architecturally
identical in this run). ARM_ABS likewise non-invariant (935 cites 934 seed
43: 1.0 -> 0.4447).

TWO-MODE SCOPE (item 7 -- reported so the regime claim is bounded by what
was actually measured). 935's manifest recorded `internal_replay` and
`offline_consolidation` at EXACTLY 0.0 in all 35 cells -- the "mixed
regime" this lineage measures is strictly a two-mode split (external_task
vs internal_planning), never observed to engage the other two SD-032a
register modes. This run inherits the identical mode-register wiring and
therefore the identical scope limit; "graded occupancy regime" here means
external_task-vs-internal_planning only, and `two_mode_scope` is recorded
explicitly in the manifest rather than left to be inferred.

NON-DEGENERACY. Unchanged from 934/935 -- `manipulation_landed` requires
occupancy OR the continuous margin to vary across the swept r on some
guard-passing seed; if neither moves, a C1 failure routes
`substrate_not_ready_requeue`, never a null.

PER-CLAIM DIRECTION: diagnostic, EXCLUDED from governance confidence/conflict
scoring. MECH-266 non_contributory (this run still contains no
asymmetric-threshold arm; the Schmitt trigger is not instantiated here
either -- the autopsy's peripheral/non_contributory read of MECH-266 is
UPHELD for this run too, unchanged). SD-032a supports if the common
normalised rule grades on fresh seeds AND beats the best absolute cap;
weakens if ready + manipulation landed but the rule does not generalise;
non_contributory if not ready, the manipulation was inert, or the run
routes H-KNIFE (a further-correction finding, not a verdict on the register).

RE-DERIVE BRAKE: RELEASED, checked against the CLAUDE.md /queue-experiment
Step 2.5b recipe. Count for MECH-266 and SD-032a: 7 prior genuinely-braking
autopsies (464b/c/d, 467b/c/d, 797), all >= the threshold of 2. The named
upstream substrate is `mode-governance-engagement`
(`REE_assembly/evidence/planning/substrate_queue.json`), whose most recent
counted autopsy (`failure_autopsy_grandfathered-r5-batch23-mixed-findings_2026-08-08`)
requested exactly this build. `ree-v3/CLAUDE.md` records it LANDED
(`mode-governance-engagement-external-task-salience-source-for.md`,
2026-06-13; the further gate-instrument fix,
`mode-governance-engagement-regime-occupancy-reproducibility.md`,
2026-09-11). V3-EXQ-935 itself is independent confirmation the substrate now
engages: its own autopsy recommends `weakens` -> `supports` for SD-032a
because the register "performed better here than anywhere else in this
lineage" (graded, non-degenerate occupancy 0.124-0.925 with 15-26 genuine
switches per cell on 4/5 seeds). Per Step 2.5b item 1, the brake is
RELEASED and this re-test is meaningful.

Step 2.5c (substrate-path overlap gate): checked against every OPEN
`substrate_queue.json` entry with `substrate_paths` set, cross-referenced
against this driver's imports (`ree_core.agent`, `causal_grid_world`,
`ree_core.utils.config`, `experiments._lib.regime_occupancy_gate`,
`experiments._lib.z_goal_stream`). The one prior blocker,
`mode-governance-engagement`, carried `severity: corrupting` describing the
EXACT successor defect in `regime_occupancy_gate.py` that this run's H-KNIFE
wiring now consumes post-fix; governance cycle `gov-20260911-1612` /
GFLAG-0262 reassessed it `severity: cosmetic`, `ready: true` (no open defect
remains for Step 2.5c to protect other experiments from -- confirmed live in
`substrate_queue.json` at authoring time). No other open `corrupting`-severity
entry was found overlapping this driver's imports.

GOV-REUSE-1: the decisive readout is "occupancy at r=2.45 (and the extended
sweep) on seeds 47-51, which have never run." Not recoverable from any
existing manifest -- checked `experiment_queue.json` / `runner_status.json` /
`experiments/` tree on `origin/main` for any V3-EXQ-935[a-z] driver or queue
entry (none found) and confirmed seeds 47-51 appear in no prior manifest
this lineage. Proceeding to author + queue is the correct route.

claim_ids: MECH-266, SD-032a.
experiment_purpose: diagnostic
predecessor: V3-EXQ-935 (successor; NOT a supersede -- 935's diagnosis of its
own four measurement defects stands; this is the corrected re-test its own
autopsy routed to).
red-team: pending -- see Step 4.5 note appended after the adversarial design
review (recorded in the queue entry `note` and appended below this line once
run).
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
from experiments.pack_writer import write_flat_manifest, flat_readout  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.regime_occupancy_gate import (  # noqa: E402
    DEFAULT_MIN_SEEDS,
    OccupancyCell,
    evaluate_regime_occupancy_gate,
)

EXPERIMENT_TYPE = "v3_exq_935a_mech266_margin_normalised_cap_rule"
QUEUE_ID = "V3-EXQ-935a"
CLAIM_IDS: List[str] = ["MECH-266", "SD-032a"]
EXPERIMENT_PURPOSE = "diagnostic"
PREDECESSOR = (
    "V3-EXQ-935 (successor; NOT a supersede -- 935's diagnosis of its own "
    "four measurement defects stands; this is the corrected re-test its own "
    "autopsy routed to)"
)

# V3-EXQ-935's own recorded substrate_hash. Item 6 fix: rather than asserting
# "this exact substrate" in prose (935's own now-refuted claim, which ran 3
# ree_core commits apart from 934 despite an identical assertion), this run
# records the REFERENCE hash literally and compares its own measured
# substrate_hash against it at manifest-write time (see main()). The
# readiness-anchor thresholds below are unchanged in KIND from 935 (ordinary
# curriculum-readiness gates), not degeneracy-reproduction anchors, so the
# 778d unmeetable-predicate failure mode still does not apply regardless of
# the comparison's outcome -- the comparison is recorded for auditability,
# not as a gate on whether the run proceeds.
REFERENCE_SUBSTRATE_HASH_V3_935 = (
    "921d0af6b6b6c54b12baafb090984edd9fcc2c3f5bf4ed2ba84d36f6c040b99b"
)
ANCHOR_REACHABILITY_EXEMPT = (
    "Readiness anchors are ordinary upstream gates (curriculum-trained; "
    "drive-engages; calibration-statistic-alive), not degeneracy-reproduction "
    "anchors. Unlike 935, this run's seeds (47-51) have no banked per-seed "
    "readiness data of their own -- satisfiability is argued from the "
    "curriculum's track record on 5 DISTINCT prior seeds (934's 42-44, 935's "
    "45-46), not asserted as identical-substrate inheritance. The manifest "
    "records `substrate_hash_matches_v3_935` (True/False) as a literal "
    "comparison against 935's own recorded hash "
    f"({REFERENCE_SUBSTRATE_HASH_V3_935}), computed post-hoc from this run's "
    "own stamped substrate_hash -- an auditable fact, not a prose assertion "
    "no code checks. A gates-pass-but-rule-fails (or H-KNIFE) read routes a "
    "VALID finding either way; the 778d unmeetable-predicate mode does not "
    "apply to either branch."
)

# Entirely fresh seeds -- none contributed to deriving R_STAR=2.45 or to
# shaping the extended R_SWEEP (both are drawn from V3-EXQ-935's seeds
# 42-46 only). See "FRESH, ENTIRELY OUT-OF-SAMPLE SEEDS" in the module
# docstring for why this run does not split seeds into a derivation/OOS
# subset the way 935 did.
SEEDS = [47, 48, 49, 50, 51]
CONDITION_LABEL = "CURRICULUM_BUILT_NORMALISED_CAP_RULE_935A_FRESH_SEEDS"

# z_goal stream liveness (Experimental Recording Standard).
_ZG = ZGoalStreamAccumulator()

MODE_NAMES = [
    "external_task",
    "internal_planning",
    "internal_replay",
    "offline_consolidation",
]
STICKY_MODE = "external_task"

# --- The calibration rule under test -------------------------------------
# CAP_REF must stay 0.75 -- unchanged from 934/935; the graded window this
# run's R_STAR/R_SWEEP are drawn from was measured with baseline margins
# taken at exactly this cap.
CAP_REF = 0.75
# r values swept per seed as cap = r * m_seed. R_STAR is IN this list.
# Extended upward from 935's [1.85, 2.05, 2.25, 2.45, 2.65]: keeps 2.25
# (935's old, now-refuted R_STAR, as a direct comparison point), replaces
# the two sub-graded points (1.85, 2.05) with 2.85 / 3.05 so a seed-42-class
# agent (needed r > 2.65 in 935, untested there) gets a tested r and the
# graded window's upper edge is actually measured.
R_SWEEP: List[float] = [2.25, 2.45, 2.65, 2.85, 3.05]
# Pre-registered common rule point: the LOWER edge of the graded window
# V3-EXQ-935 itself observed (r=2.45 and r=2.65 both graded 4/5 on 935's
# seeds; 2.45 is chosen as the more conservative pre-registration, closer to
# the observed transition rather than deeper into the confirmed-graded
# region).
R_STAR = 2.45
# Best single absolute cap available from 934/935 -- unchanged.
CAP_ABS_CONTROL = 1.75
# Training-time cap -- unchanged from 464e/934/935.
AFFINITY_INPUT_CAP_TRAIN = 2.0

ARM_NORM = "ARM_NORM"
ARM_ABS = "ARM_ABS"
ARM_CALIB = "ARM_CALIB"

# The null's mixed band: occupancy strictly inside (floor, ceiling).
OCCUPANCY_FLOOR = 0.10
OCCUPANCY_CEILING = 0.90
# G-margin / G-calib floor. Unchanged.
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


# --------------------------------------------------------------------------
# Harness below this line is REUSED VERBATIM from V3-EXQ-934/935's validated
# driver (scaffold config, substrate config, env build, arm clone, symmetric
# rails, quantile helper, and the per-cell frozen-policy eval). It is
# substrate plumbing, not science: keeping it byte-identical is what makes
# this run's occupancy readings directly comparable to 934's and 935's
# banked cells.
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
    identical to 464e/934/935. `seed` default None passes through to
    CausalGridWorldV2's OS-entropy default -- bit-identical to the landed
    464e/467e eval env layout, so the cap-sweep results stay comparable to
    the banked reference."""
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


def _r_values(dry_run: bool) -> List[float]:
    """The r values swept per seed. Single source of truth so the cell-count
    arithmetic in run_experiment() and the loop in _run_seed() cannot drift.

    THE DRY-RUN SUBSET MUST CONTAIN R_STAR (kept in sync with R_STAR=2.45,
    R_SWEEP[1] in this run -- item from the 2026-09-10 refusal record: 935's
    first smoke took R_SWEEP[:2] and OMITTED R_STAR by accident, producing
    occ_at_r_star=None and still routing a verdict on a criterion that was
    never measured). C1 -- the load-bearing criterion -- is evaluated ONLY at
    R_STAR, so a smoke that omits it exercises nothing."""
    if dry_run:
        return sorted({R_SWEEP[0], R_STAR})
    return list(R_SWEEP)


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "baseline_margin": 0.0,
        "calib_ok": False,
        "r_star_measured": False,
        "cells": [],
        "cap_at_r_star": 0.0,
        "occ_at_r_star": None,
        "graded_at_r_star": False,
        "occ_abs_control": None,
        "graded_abs_control": False,
        "graded_at_some_r": False,
        "graded_r_values": [],
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
    r_values = _r_values(dry_run)

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
    # for every seed. This is C2's head-to-head comparator.
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

    # An ABSENT R_STAR cell means C1 was never measured on this seed. That is an
    # instrument condition, NOT evidence about the rule -- so it must never be
    # allowed to read as "did not grade" and feed a verdict.
    r_star_measured = bool(star_cell is not None)
    occ_at_r_star = (
        float(star_cell["fraction_in_external_task"]) if r_star_measured else None
    )
    graded_at_r_star = bool(r_star_measured and _graded(occ_at_r_star))
    occ_abs = float(abs_cell["fraction_in_external_task"])
    graded_abs = _graded(occ_abs)

    graded_r_values = [
        float(c["r"]) for c in norm_cells
        if _graded(c["fraction_in_external_task"])
    ]

    max_margin_mean = max((float(c["ext_margin_mean"]) for c in cells), default=0.0)
    margin_engaged = bool(max_margin_mean > MARGIN_FLOOR)
    rule_testable = bool(guard_pass and calib_ok and r_star_measured)

    print(f"  [rule] seed={seed} baseline_margin={baseline_margin:.4f}"
          f" cap_at_r_star={R_STAR * baseline_margin:.4f}"
          f" occ_at_r_star={occ_at_r_star} graded_at_r_star={graded_at_r_star}"
          f" | abs_cap={CAP_ABS_CONTROL} occ={occ_abs} graded={graded_abs}", flush=True)
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
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_num_contact_events": int(p2.num_contact_events),
        "baseline_margin": round(baseline_margin, 4),
        "calib_ok": calib_ok,
        "r_star_measured": r_star_measured,
        "rule_testable": rule_testable,
        "cells": cells,
        "cap_at_r_star": round(R_STAR * baseline_margin, 4),
        "occ_at_r_star": occ_at_r_star,
        "graded_at_r_star": graded_at_r_star,
        "occ_abs_control": occ_abs,
        "graded_abs_control": graded_abs,
        "graded_at_some_r": bool(graded_r_values),
        "graded_r_values": graded_r_values,
        "max_margin_mean": round(max_margin_mean, 4),
        "margin_engaged": margin_engaged,
    }


def run_experiment(dry_run: bool = False,
                   env_seed_base: Optional[int] = None) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}, "
          f"env_seed_base={env_seed_base})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    r_values = _r_values(dry_run)
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

    # G-calib: the calibration statistic must be alive.
    calib_flags = [bool(r.get("calib_ok", False)) for r in guard_passing]
    calib_frac = _frac(calib_flags)
    calib_ready_met = bool(calib_frac >= MIN_FRACTION)

    # G-rstar: the R_STAR cell must actually EXIST on every guard-passing seed.
    r_star_flags = [bool(r.get("r_star_measured", False)) for r in guard_passing]
    r_star_frac = _frac(r_star_flags)
    r_star_measured_met = bool(r_star_frac >= 1.0)

    # Criteria are scored over seeds that are actually TESTABLE (guard-passing
    # AND calibration-alive AND carrying a measured R_STAR cell).
    testable = [
        r for r in guard_passing
        if r.get("calib_ok", False) and r.get("r_star_measured", False)
    ]
    n_testable = len(testable)

    # -- C1: the COMMON RULE test on these ENTIRELY FRESH seeds.
    c1_flags = [bool(r.get("graded_at_r_star", False)) for r in testable]
    c1_frac = _frac(c1_flags)
    c1_passed = bool(n_testable > 0 and c1_frac >= MIN_FRACTION)
    n_graded_norm = sum(1 for f in c1_flags if f)

    # -- C2 (935's C3, renamed -- see DESIGN SIMPLIFICATION in the module
    # docstring): strictly beat the best single absolute cap, same seeds.
    abs_flags = [bool(r.get("graded_abs_control", False)) for r in testable]
    n_graded_abs = sum(1 for f in abs_flags if f)
    abs_frac = _frac(abs_flags)
    c2_passed = bool(n_testable > 0 and n_graded_norm > n_graded_abs)

    # Non-degeneracy: did the per-seed normalised cap manipulation LAND at all?
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

    # H-KNIFE aggregate: the CORRECTED shared primitive, called ONCE over ALL
    # (seed, r) ARM_NORM cells drawn from testable seeds -- the call shape the
    # module's own docstring requires (a per-seed call is precisely the shape
    # that produced 934's false positive). This both supplies the
    # informational whole-sweep regime_shape AND is what the H-KNIFE routing
    # branch below consults.
    all_norm_cells = [
        OccupancyCell(
            label=f"r={c['r']}",
            fraction=float(c["fraction_in_external_task"]),
            seed=int(r["seed"]),
            sweep_value=float(c["r"]),
        )
        for r in testable
        for c in r.get("cells", [])
        if c["arm"] == ARM_NORM
    ]
    occupancy_gate = evaluate_regime_occupancy_gate(
        all_norm_cells,
        mode_label=STICKY_MODE,
        floor=OCCUPANCY_FLOOR,
        ceiling=OCCUPANCY_CEILING,
        min_seed_fraction=MIN_FRACTION,
    )
    # H-KNIFE candidates: any OTHER r (not R_STAR itself -- if R_STAR
    # qualified here, C1 would already have passed, since both use the
    # identical testable population and threshold) that clears the
    # seed-reproducibility bar on its own, whether or not it is part of an
    # adjacent qualifying run. This is deliberately looser than
    # `occupancy_gate["graded"]` (which additionally requires >= 2 ADJACENT
    # qualifying r's): H-KNIFE only needs ONE other r to justify a
    # further-corrected re-queue, not a full graded regime.
    #
    # Guarded on the SAME min-seeds floor `classify_regime_shape` uses
    # (DEFAULT_MIN_SEEDS=2): `_per_value_reproducibility`'s per-r `qualifies`
    # flag does not itself check seed count, so on a run with only 1
    # testable seed (e.g. the --dry-run smoke) a single seed's own
    # 1-of-1 == 100% would spuriously "qualify" every r it graded at. That
    # is harmless in the smoke (n_testable==1 there is expected and C1 is
    # evaluated identically), but the SAME code path runs the real 5-seed
    # grid, where n_testable could in principle also fall to 1 if 4/5 seeds
    # fail readiness -- and an unguarded H-KNIFE claim there would be exactly
    # as unreliable as the per-cell existential this module was built to
    # replace. Below the floor, treat H-KNIFE as unavailable (falls through
    # to H-IDIO, which is the conservative direction: a spurious H-KNIFE
    # claim would wrongly avoid a genuine idiosyncrasy finding).
    n_testable_seeds_for_knife = len({int(c.seed) for c in all_norm_cells if c.seed is not None})
    if n_testable_seeds_for_knife >= DEFAULT_MIN_SEEDS:
        knife_qualifying_rs = sorted(
            rec["condition"] for rec in occupancy_gate.get("per_value", [])
            if rec.get("qualifies") and abs(float(rec["condition"]) - R_STAR) > 1e-9
        )
    else:
        knife_qualifying_rs = []
    h_knife_available = bool(knife_qualifying_rs)

    # PASS gate: plain AND of C1 and C2.
    combination_rule = (
        "PASS iff C1_rule_grades_at_r_star AND C2_beats_best_absolute_cap. "
        "Plain AND -- recorded explicitly so the gate is never inferred from "
        "the per-criterion booleans alone. (935's separate C2 'out-of-sample' "
        "criterion is not carried forward -- see DESIGN SIMPLIFICATION in the "
        "module docstring: every seed in this run is equally out-of-sample, "
        "so a nested OOS subset would test nothing C1 does not already test "
        "on the full, fresh population.)"
    )
    rule_supported = bool(c1_passed and c2_passed)

    # route_reason is DERIVED from the actual criterion states below, never a
    # fixed string per branch (item 5 -- 935's route_reason was hardcoded and
    # factually false on its own run).
    failed_criteria = []
    if not c1_passed:
        failed_criteria.append("C1_rule_grades_at_r_star")
    if not c2_passed:
        failed_criteria.append("C2_beats_best_absolute_cap")

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
    elif not r_star_measured_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "r_star_cell_missing_c1_never_evaluated"
    elif not manipulation_landed:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "normalised_cap_manipulation_inert_verify_instrument"
    elif rule_supported:
        outcome = "PASS"
        readiness_route = "normalised_cap_rule_admits_common_mixed_regime"
        route_reason = "criteria_passed:C1_rule_grades_at_r_star+C2_beats_best_absolute_cap"
    elif c1_passed and not c2_passed:
        # Red-team (fable) finding F1, verified against source and fixed here
        # rather than dismissed: this state (rule GRADES at R_STAR on fresh
        # seeds, but does not STRICTLY beat the best absolute cap) is
        # reachable and is NOT seed-idiosyncrasy -- R_STAR itself cleared the
        # seed-reproducibility bar, so "no_r_in_extended_sweep_clears_min_
        # fraction" would be factually false here, reproducing 935's own
        # defect #2 (a hardcoded route_reason false on its own run) inside
        # this run's H-IDIO branch. The honest finding is distinct from
        # H-IDIO: the rule generalises to fresh seeds, but a fixed absolute
        # cap does equally well or better for THIS seed set, so the
        # normalisation does not earn its keep as a shippable rule.
        outcome = "FAIL"
        readiness_route = "rule_grades_but_not_better_than_absolute_cap"
        route_reason = "criteria_failed:C2_beats_best_absolute_cap|c1_passed_r_star_not_idiosyncratic"
    elif (not c1_passed) and h_knife_available:
        outcome = "FAIL"
        readiness_route = "rule_right_r_wrong_requeue"
        route_reason = (
            "criteria_failed:" + "+".join(failed_criteria)
            + f"|other_r_clearing_min_fraction:{knife_qualifying_rs}"
        )
    else:
        outcome = "FAIL"
        readiness_route = "cap_recalibration_is_seed_idiosyncratic"
        route_reason = (
            "criteria_failed:" + "+".join(failed_criteria)
            + "|no_r_in_extended_sweep_clears_min_fraction"
        )

    # Diagnostic: excluded from confidence scoring. Directions are context only.
    if rule_supported:
        sd032a_dir = "supports"          # register recalibratable by a shippable rule, on fresh seeds
    elif readiness_route == "rule_right_r_wrong_requeue":
        sd032a_dir = "non_contributory"  # further-correction finding, not a verdict on the register
    elif (contact_non_vacuity_met and margin_ready_met and calib_ready_met
          and r_star_measured_met and manipulation_landed):
        sd032a_dir = "weakens"           # ready + manipulation landed, but no rule generalises anywhere in the extended sweep
    else:
        sd032a_dir = "non_contributory"  # not ready / inert; says nothing about the register
    direction_map = {
        "MECH-266": "non_contributory",
        "SD-032a": sd032a_dir,
    }
    overall_direction = "non_contributory"

    print(f"[{EXPERIMENT_TYPE}] guard {sum(guard_flags)}/{n}"
          f" margin_ready={margin_ready_met} calib_ready={calib_ready_met}"
          f" r_star_measured={r_star_measured_met}"
          f" n_testable={n_testable}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] C1 graded_at_r_star={n_graded_norm}/{n_testable}"
          f" (frac={c1_frac:.3f}) passed={c1_passed}"
          f" | C2 norm={n_graded_norm} vs abs={n_graded_abs} passed={c2_passed}",
          flush=True)
    print(f"[{EXPERIMENT_TYPE}] occupancy_gate regime_shape="
          f"{occupancy_gate.get('regime_shape')}"
          f" h_knife_available={h_knife_available}"
          f" knife_qualifying_rs={knife_qualifying_rs}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] -> outcome={outcome} route={readiness_route}"
          f" route_reason={route_reason}", flush=True)
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
        "r_star_measured_met": r_star_measured_met,
        "r_star_measured_fraction": r_star_frac,
        "n_testable_seeds": n_testable,
        "c1_rule_grades_at_r_star": c1_passed,
        "c1_fraction": c1_frac,
        "c1_n_graded": n_graded_norm,
        "c2_beats_best_absolute_cap": c2_passed,
        "c2_n_graded_normalised": n_graded_norm,
        "c2_n_graded_absolute": n_graded_abs,
        "c2_absolute_fraction": abs_frac,
        "rule_supported": rule_supported,
        "combination_rule": combination_rule,
        "occupancy_varies_across_r": occupancy_varies,
        "margin_varies_across_r": margin_varies,
        "manipulation_landed": manipulation_landed,
        "h_knife_available": h_knife_available,
        "h_knife_qualifying_rs": knife_qualifying_rs,
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
                       "z_goal_norm_at_contact_peak > 0.4. Cleared 5/5 by "
                       "V3-EXQ-934+935 on this curriculum family (seeds 42-46); "
                       "these are FRESH seeds (47-51) so this is a live check, "
                       "not a banked assertion.",
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
                       "ext_margin_mean clears MARGIN_FLOOR. V3-EXQ-934/935 "
                       "measured 0.3217-0.8139 on this curriculum family against "
                       "a 0.05 floor (fresh seeds here, live check).",
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
                       "control at the SAME cap (0.75) and SAME symmetric rails "
                       "at which V3-EXQ-934/935 measured m in [0.3217, 0.8139] "
                       "across seeds 42-46, all ~6-16x the 0.05 floor.",
            "measured": round(calib_frac, 4),
            "threshold": MIN_FRACTION,
            "direction": "lower",
            "met": calib_ready_met,
        },
        {
            "name": "r_star_cell_measured",
            "kind": "readiness",
            "description": "the R_STAR cell must EXIST on every guard-passing seed. "
                           "C1 -- the load-bearing criterion -- is evaluated ONLY at "
                           "R_STAR, so a missing cell means C1 was never measured at "
                           "all. This is an INSTRUMENT invariant, not a statistical "
                           "one, which is why the threshold is 1.0 rather than "
                           "MIN_FRACTION.",
            "control": "R_STAR is a member of R_SWEEP by construction and the "
                       "dry-run r subset is built by _r_values() to always contain "
                       "it (kept in sync with R_STAR this run -- see the "
                       "2026-09-10 refusal-record note in that function's "
                       "docstring).",
            "measured": round(r_star_frac, 4),
            "threshold": 1.0,
            "direction": "lower",
            "met": r_star_measured_met,
        },
    ]

    criteria = [
        {"name": "C1_rule_grades_at_r_star", "load_bearing": True, "passed": c1_passed,
         "measured": round(c1_frac, 4), "threshold": MIN_FRACTION, "comparator": ">="},
        {"name": "C2_beats_best_absolute_cap", "load_bearing": True, "passed": c2_passed,
         "measured": n_graded_norm, "threshold": n_graded_abs, "comparator": ">"},
    ]

    base_non_degenerate = bool(
        contact_non_vacuity_met and margin_ready_met and calib_ready_met
        and r_star_measured_met and manipulation_landed and n_testable > 0
    )
    criteria_non_degenerate = {
        "C1_rule_grades_at_r_star": base_non_degenerate,
        "C2_beats_best_absolute_cap": base_non_degenerate,
    }

    # Flat scalar readout -- the pack's metrics.values source (Experimental
    # Recording Standard sec 3b "Machine-readable verdict readout").
    readout = flat_readout({
        "C1_rule_grades_at_r_star": c1_passed,
        "C2_beats_best_absolute_cap": c2_passed,
        "n_criteria_passed": sum(1 for c in criteria if c["passed"]),
        "n_criteria_total": len(criteria),
        "rule_supported_flag": rule_supported,
        "base_non_degenerate_flag": base_non_degenerate,
        "h_knife_available": h_knife_available,
        # readiness: four preconditions, each against the shared seed bar
        "min_fraction": MIN_FRACTION,
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "margin_ready_met": margin_ready_met,
        "margin_ready_fraction": margin_frac,
        "margin_floor": MARGIN_FLOOR,
        "calib_ready_met": calib_ready_met,
        "calib_ready_fraction": calib_frac,
        "r_star_measured_met": r_star_measured_met,
        "r_star_measured_fraction": r_star_frac,
        "n_testable_seeds": n_testable,
        # C1 -- the rule at the pre-registered r, on fresh seeds
        "r_star": R_STAR,
        "c1_fraction": c1_frac,
        "c1_n_graded": n_graded_norm,
        # C2 -- normalised rule vs the best absolute cap
        "c2_n_graded_normalised": n_graded_norm,
        "c2_n_graded_absolute": n_graded_abs,
        "c2_absolute_fraction": abs_frac,
        "c2_normalised_minus_absolute_n_graded": n_graded_norm - n_graded_abs,
        # regime shape over the extended sweep (corrected primitive)
        "regime_shape_graded": bool(occupancy_gate.get("graded")),
        "regime_shape_reachable": bool(occupancy_gate.get("reachable")),
        "n_r_swept": len(R_SWEEP),
        "r_sweep_min": min(R_SWEEP),
        "r_sweep_max": max(R_SWEEP),
        # did the normalised cap manipulation land at all -- both disjuncts, separately
        "manipulation_landed": manipulation_landed,
        "occupancy_varies_across_r": occupancy_varies,
        "margin_varies_across_r": margin_varies,
        "n_seeds": len(per_seed),
    })

    return {
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": direction_map,
        "readout": readout,
        "acceptance": acceptance,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "route_reason": route_reason,
            "hypothesis": "H-RULE: a SINGLE pre-registered r, applied per-seed as "
                          "cap = r * baseline_margin(seed), yields a graded "
                          "external_task occupancy regime on >= 2/3 of ENTIRELY "
                          "FRESH seeds simultaneously -- i.e. the recalibration "
                          "is a shippable RULE that generalises out of sample.",
            "null": "H-IDIO: no r anywhere in the extended sweep clears the "
                    "seed-reproducibility bar on these fresh seeds. The "
                    "required cap is seed-idiosyncratic; there is no rule to "
                    "ship. Reached ONLY when C1 fails at R_STAR AND no other "
                    "r qualifies -- a C1-passes-but-C2-fails state routes to "
                    "the distinct rule_grades_but_not_better_than_absolute_cap "
                    "label below, never here (red-team fable finding F1, "
                    "2026-09-14: the naive else-fallthrough this run's own "
                    "predecessor 935 was autopsied for would otherwise "
                    "reproduce inside this run too).",
            "not_better_than_absolute": "rule_grades_but_not_better_than_absolute_cap: "
                                "C1 passes (the rule DOES grade on >= 2/3 of "
                                "fresh seeds at R_STAR) but C2 fails -- the "
                                "fixed absolute cap grades on as many or more "
                                "of the same seeds. This is NOT seed-idiosyncrasy "
                                "(R_STAR cleared the bar) and is NOT H-KNIFE "
                                "(C1 already passed) -- a genuinely distinct, "
                                "third outcome: the normalisation generalises "
                                "but does not earn its keep over a simple "
                                "fixed cap for this seed set.",
            "aliasing_control": "H-KNIFE (the rule is right, R_STAR itself was "
                                "wrong) is now WIRED into the routing (fix (1) "
                                "in the module docstring, the defect 935's own "
                                "autopsy found): if C1 fails at R_STAR but some "
                                "OTHER r in the extended sweep clears the "
                                "seed-reproducibility bar on these fresh seeds, "
                                "the run routes rule_right_r_wrong_requeue "
                                "instead of falling through to H-IDIO.",
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
            "occupancy_gate": occupancy_gate,
            "h_knife": {
                "checked": True,
                "available": h_knife_available,
                "qualifying_r_values": knife_qualifying_rs,
                "note": "Computed from the SAME aggregate call as occupancy_gate "
                        "(one call over all testable-seed ARM_NORM cells, never "
                        "per-seed). A qualifying r other than R_STAR means the "
                        "rule FORM is supported and only the pre-registered "
                        "operating point needs correction -- a distinct finding "
                        "from H-IDIO (no r anywhere reproduces).",
            },
            "fresh_seeds": {
                "statement": "R_STAR=2.45 and the extended R_SWEEP were both "
                             "derived entirely from V3-EXQ-935's seeds 42-46. "
                             "This run's seeds (47-51) played no role in either "
                             "derivation.",
                "prior_run_seeds": [42, 43, 44, 45, 46],
                "this_run_seeds": SEEDS,
            },
            "two_mode_scope": {
                "statement": "V3-EXQ-935's manifest recorded internal_replay and "
                             "offline_consolidation at EXACTLY 0.0 in all 35 "
                             "cells. This run's mode-register wiring is "
                             "identical, so the regime claim below is bounded "
                             "to a two-mode split (external_task vs "
                             "internal_planning); the other two SD-032a "
                             "register modes have never been observed to "
                             "engage in this lineage.",
                "modes_observed_in_prior_run": ["external_task", "internal_planning"],
                "modes_never_engaged_in_lineage": ["internal_replay", "offline_consolidation"],
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
                            "DV in the identical mechanism (V3-EXQ-934 seed 42 moved "
                            "0.5606 -> 0.0317 -> 0.0 across caps).",
                "ARM_ABS": "NOT INVARIANT. Identical manipulation type at a fixed "
                           "cap; same argument, and likewise empirically "
                           "non-invariant (V3-EXQ-934 seed 43: 1.0 -> 0.4447).",
            },
            "structural_satisfiability": {
                "checked": "on paper before queuing, argued from the curriculum's "
                           "track record on 5 DISTINCT prior seeds (42-46 across "
                           "934/935) since 47-51 have no banked data of their own "
                           "-- see STRUCTURAL SATISFIABILITY in the module "
                           "docstring for why this is a weaker evidentiary claim "
                           "than 935's own (and deliberately stated as such).",
                "foraging_contact_guard": "5/5 measured (934+935) vs a 2/3 floor "
                                         "-- no known seed-dependent failure mode.",
                "external_task_drive_engages": "0.3217-0.8139 measured (934+935) "
                                               "vs a 0.05 floor -- no known "
                                               "seed-dependent failure mode.",
                "calibration_statistic_alive": "same statistic as above, same "
                                               "argument.",
                "C1": "satisfiable in principle -- 935's own sweep graded 4/5 at "
                      "r=2.45 on seeds 42-46; NOT proven for 47-51 specifically, "
                      "which is exactly what this run tests.",
                "C2": "the control is not a straw man -- ARM_ABS at cap 1.75 "
                      "graded on seeds in 934/935's banked data, so it CAN grade "
                      "and C2 requires strictly beating it on fresh seeds too.",
                "no_arm_structurally_vacuous": True,
            },
            "regime_gate": {
                "definition": "occupancy is 'graded' iff strictly inside "
                              "(OCCUPANCY_FLOOR, OCCUPANCY_CEILING). C1 evaluates "
                              "this at the SINGLE pre-registered R_STAR -- a "
                              "common-rule test on fresh seeds. The shared "
                              "primitive experiments/_lib/regime_occupancy_gate.py "
                              "(corrected, 2026-09-11) is applied over the full r "
                              "axis for occupancy_gate / H-KNIFE, called ONCE over "
                              "all testable-seed cells -- never per-seed.",
                "occupancy_floor": OCCUPANCY_FLOOR,
                "occupancy_ceiling": OCCUPANCY_CEILING,
                "margin_floor": MARGIN_FLOOR,
                "cap_ref": CAP_REF,
                "r_sweep": R_SWEEP,
                "r_star": R_STAR,
                "cap_abs_control": CAP_ABS_CONTROL,
                "min_fraction": MIN_FRACTION,
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
        "min_fraction": MIN_FRACTION,
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
        "supersedes": None,
        "method_note": "Per seed: train ONE curriculum agent, then run frozen-policy "
                       "eval cells on clones, IDENTICAL to 934/935's harness. (1) A "
                       "calibration cell at CAP_REF=0.75 on symmetric rails yields "
                       "that seed's baseline external_task margin m_seed. (2) "
                       "ARM_NORM cells at cap = r * m_seed for r in "
                       "[2.25, 2.45, 2.65, 2.85, 3.05]. (3) An ARM_ABS control at "
                       "a fixed cap of 1.75. Primary criteria evaluate the SINGLE "
                       "pre-registered R_STAR=2.45 across FIVE ENTIRELY FRESH "
                       "seeds (47-51) -- a common-rule test that is, by "
                       "construction, also the out-of-sample generalisation test "
                       "935's own C2 tried to be on a 2-seed subset.",
        "pre_registered_thresholds": {
            "cap_ref": CAP_REF,
            "r_sweep": R_SWEEP,
            "r_star": R_STAR,
            "r_star_derivation": "the LOWER edge of the graded window observed in "
                                 "V3-EXQ-935's own five-seed sweep (r=2.45 and "
                                 "r=2.65 both graded 4/5 on seeds 42-46; 2.45 "
                                 "chosen as the more conservative pre-registration).",
            "cap_abs_control": CAP_ABS_CONTROL,
            "affinity_input_cap_train": AFFINITY_INPUT_CAP_TRAIN,
            "occupancy_floor": OCCUPANCY_FLOOR,
            "occupancy_ceiling": OCCUPANCY_CEILING,
            "margin_floor": MARGIN_FLOOR,
            "min_fraction": MIN_FRACTION,
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
        },
        "anchor_reachability_exempt": ANCHOR_REACHABILITY_EXEMPT,
        "reference_substrate_hash_v3_935": REFERENCE_SUBSTRATE_HASH_V3_935,
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
    # Item 6 fix: record the substrate_hash comparison as an auditable fact
    # rather than a prose assertion of identity. write_flat_manifest / the
    # recording-core stamper already wrote `substrate_hash` into the file at
    # `out_path`; re-read it to append the comparison rather than
    # recomputing the hash independently (avoids drift between the two).
    try:
        import json as _json
        with open(out_path, "r", encoding="utf-8") as _f:
            _written = _json.load(_f)
        _this_hash = _written.get("substrate_hash")
        _written["substrate_hash_matches_v3_935"] = bool(
            _this_hash == REFERENCE_SUBSTRATE_HASH_V3_935
        )
        with open(out_path, "w", encoding="utf-8") as _f:
            _json.dump(_written, _f, indent=2, sort_keys=True)
    except Exception as _e:  # pragma: no cover - never fail the run over this
        print(f"[{EXPERIMENT_TYPE}] WARNING: could not stamp "
              f"substrate_hash_matches_v3_935: {_e}", flush=True)

    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--env-seed", type=int, default=None,
        help="Opt-in env-seed base. Omitted (the default) reproduces V3-EXQ-934/935's "
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
