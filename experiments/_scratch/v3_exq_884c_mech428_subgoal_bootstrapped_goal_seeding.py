"""
PARKED -- REFUSED at /queue-experiment Step 4.5. NOT QUEUED. NO queue entry,
no coordinator row, no manifest. Do NOT resurrect this file as-is.

Red-team (fable, foreground, 2026-09-16): BLOCKING, four findings, ALL FOUR
independently re-confirmed against this file's own source before being
accepted. Full record, including the synthetic no-signal confirmation:
REE_assembly/evidence/planning/
exq884c_mech428_c1_ema_self_overlap_redteam_blocking_20260916.md

F1 BLOCKING  C1's statistic T = cos(parent, mean(attained)) - cos(parent,
             mean(non_attained)) is BIASED POSITIVE with zero content signal.
             The parent is an EMA built FROM the attained group, so
             cos(parent, mean(attained)) carries a self-overlap term that
             cos(parent, mean(non_attained)) does not; the label-permutation
             null holds the parent FIXED and therefore never re-runs the step
             that makes the observed labelling special, so it cannot absorb
             that term. CONFIRMED on synthetic data whose labels carry zero
             information by construction (reps = one shared unit direction +
             iid noise): T "separates" at the 100th percentile of its own null
             on 12/12 trials at sigma >= 0.05, at the SAME magnitudes
             (~0.003 at sigma=0.10) as the real measurement. The real
             measurement is therefore fully consistent with pure artifact.
F2 BLOCKING  C2, the negative control meant to catch exactly F1, scores the
             CREDIT_RANDOM_TICKS arm against the ATTAINMENT partition rather
             than against that arm's OWN credited/uncredited partition
             (_score(rand, ...) reads row["_content_attained_reprs"]). By the
             same self-overlap term its T is negative BY CONSTRUCTION, so the
             control is silent for a bookkeeping reason and the branch is
             dead.
F3 CONTESTED G3 samples the POST-arrival representation (z_now in the readiness
             probe) while C1 credits and scores the PRE-arrival one (prev_z),
             at 1/5 the sample size -- and this session's own measurement says
             the post-arrival statistic is AT-CHANCE on 2/3 seeds at
             alpha_world=0.9, so at real scale the run self-routes
             substrate_not_ready_requeue before any C criterion is read.
F4 CONTESTED CREDIT_RANDOM_TICKS is not random: random_tick_set =
             set(shuffled range(1, n_steps+1)) is the set of ALL ticks, so
             membership is always true and the shuffle is a no-op. It credits
             the EARLIEST non-attainment ticks until the budget is spent.

WHAT IS STILL WORTH KEEPING FROM THIS FILE (the reason it is parked, not
deleted): the two substrate levers it establishes are real and were measured
before any of the above was known -- see the measurement section below and the
three probes in this directory. Pre-arrival crediting through the real
substrate call and alpha_world=0.9 are JOINTLY necessary and individually
insufficient to make the attained/transit groups separable in encoded z_world
space (3/3 seeds). That finding survives F1-F4 untouched, because it is a
GROUP-MEAN result that never touches the parent EMA. What does NOT survive is
any parent-level content criterion built on statistic T.

The correctly-nulled parent-level statistic (S, which DOES re-run the
parent-building step inside the null) was measured AT-CHANCE on 3/3 seeds.
So the honest position after this session is: at the substrate's default
parent_goal_alpha / parent_goal_decay and this event rate, the parent-level
content signal is not detectable with a sound null. That is a finding about
the measurable range, not a failure of effort, and a successor should attack
parent_goal_alpha / event rate as the independent variable rather than search
for a fourth statistic.

--- the design as it stood when it was refused, for the record ---

V3-EXQ-884c -- MECH-428 Subgoal-Bootstrapped Goal Seeding: content-DV
criterion, PRE-ARRIVAL credited representation at alpha_world=0.9.
SUPERSEDES V3-EXQ-884b (parked at experiments/_scratch/, never queued --
red-team BLOCKING, four findings F1-F4).

Claim: MECH-428 (subgoal_bootstrapped_goal_seeding). Proposal: EXP-0390.

SLEEP DRIVER: N/A (no sleep loop; scripted-trajectory representational probe).

RED-TEAM (Step 4.5): __REDTEAM__

WHAT 884b LEFT BEHIND
---------------------------------------------------------------------------
884b was refused at /queue-experiment Step 4.5. Full record:
REE_assembly evidence/planning/
exq884b_mech428_c1_content_dv_redteam_blocking_20260914.md (064df491da).
  F1 BLOCKING  CONTENT_DELTA_ABS_FLOOR=0.05 is 70-250x the achievable
               ceiling; only FAIL is reachable.
  F2 BLOCKING  the criterion was computed over a REPLAY, so it tested
               z_world's group separation, not credit_subgoal_attainment.
  F3 CONFIRMED G3 certified noise: 96 percent of random label partitions
               cleared its 0.0002 floor.
  F4 CONTESTED the credited observation carries no waypoint signature,
               because the agent marker overwrites the waypoint cell on
               arrival and SD-094(b) restores it only on departure.

WHAT THIS SESSION MEASURED BEFORE CHANGING ANYTHING
---------------------------------------------------------------------------
The chip made an achievable-ceiling measurement a PRECONDITION of any
redesign ("measure the ceiling BEFORE fixing the floor, not after" -- the gap
884b's own dv_headroom block had, recording the bound [-1,1] rather than an
achievable estimate). Three scratch probes, kept at experiments/_scratch/,
all on the mean-direction cosine dissimilarity between the attained-tick and
transit-tick groups against a 200-draw label-permutation null:

(1) _probe_884c_credited_repr_ceiling.py -- RAW OBSERVATION space, seeds
    42/43/44. post-arrival local_view 0.0132/0.0182/0.0140 (100th percentile
    of its own null on all three); pre-arrival local_view
    0.0261/0.0242/0.0222 (100th on all three); SD-WAYPOINT-FIELD
    0.0018/0.0036/0.0023 (AT-CHANCE on seed 42, 99.5th on 44).
    => F4's STRONG form is FALSE. The raw observation DOES carry the
       signature, post-arrival included. And direction 2 of the chip's two
       candidates (credit the waypoint-proximity field) is REJECTED ON
       MEASUREMENT: it is the weakest of the three and not robust across
       seeds, so it is not implemented here.

(2) _probe_884c_encoded_ceiling.py -- ENCODED z_world space (the space the
    criterion actually routes on), the full 2x2 of credited tick x
    alpha_world, seeds 42/43/44, real P0 scale:

      alpha_world  credited tick   seed 42        seed 43        seed 44
      0.3 (884b)   post-arrival    AT-CHANCE      AT-CHANCE      AT-CHANCE
      0.3          pre-arrival     AT-CHANCE      AT-CHANCE      AT-CHANCE
      0.9          post-arrival    AT-CHANCE      SEPARATES      AT-CHANCE
      0.9          pre-arrival     SEPARATES      SEPARATES      SEPARATES
                                   0.002160       0.002288       0.001145
                                   (null p95      (null p95      (null p95
                                    0.000799)      0.000989)      0.000558)

    => The two levers are JOINTLY necessary and neither alone is sufficient.
       This is the finding that makes 884c different from 884b rather than a
       power bump of it. The second lever was not in the chip's candidate
       list and was found by reading the substrate:
       ree_core/latent/stack.py:1584 blends
         z_world = alpha_world * z_world + (1 - alpha_world) * prev.z_world
       and 884b never set alpha_world, so it ran at the REEConfig default
       0.3 -- i.e. every sensed z_world was 70 percent inherited from the
       PREVIOUS tick, a temporal EMA that smears an attainment tick into its
       transit neighbours before any crediting happens. stack.py:1537 calls
       0.3 backward-compat only and says "set alpha_world >= 0.9 to fix event
       suppression"; config.py:80-84 says the same. An event-suppression EMA
       is exactly the mechanism that maps a raw 0.024 separation onto an
       encoded 0.0007. F4 named the observation; the dominant term was
       actually the sense path.

(3) _probe_884c_parent_level.py + the part-3 statistic selection -- the
    PARENT-level ceiling at the chosen config, which is a DIFFERENT scale
    from (2) because the parent is a recency-weighted EMA, not a mean.
    Two candidate parent statistics were measured, and the first was
    REJECTED on measurement:
      S = 1 - cos(parent_true, parent_control), both built by the same
          decay+credit replay at the same credit-tick positions:
          AT-CHANCE on 3/3 seeds (78.0 / 63.0 / 86.5 percentile of its own
          null). The EMA's effective sample size is ~1/parent_goal_alpha =
          20, so a resampled control parent carries so much sampling
          variance that the null's spread swamps the achievable signal.
          Recording this so no successor re-derives it.
      T = cos(parent, mean(attained)) - cos(parent, mean(non_attained)),
          with the PARENT HELD FIXED and only the LABELS permuted for the
          null (group means over n=60 and n=340, so low variance):
            seed 42  T=0.003431  null_med=-0.000367  null_p95=0.001008  3.4x
            seed 43  T=0.000662  null_med=-0.000224  null_p95=0.000196  3.4x
            seed 44  T=0.001100  null_med=-0.000302  null_p95=0.000333  3.3x
          SEPARATES on 3/3. T is the criterion adopted below.

    The same probe measured C0 (below): cos(replayed parent, the REAL
    substrate parent) = 1.000000000 / 0.999999881 / 1.000000119.

HOW EACH RED-TEAM FINDING IS DISCHARGED
---------------------------------------------------------------------------
F4 (root cause). The credited representation is now the PRE-ARRIVAL z_world
   -- the latent sensed at tick t-1, before the agent marker overwrote the
   waypoint cell -- AND alpha_world is raised to 0.9 so that latent is not
   70 percent inherited from earlier transit ticks. Measurement (2) shows
   both are required.

F2 (the criterion tested a replay, not the substrate). The pre-arrival
   representation is credited THROUGH THE REAL SUBSTRATE CALL:
     agent.notify_subgoal_attainment(ttype, child_representation=prev_z)
   GoalState.credit_subgoal_attainment's own docstring sanctions this --
   "child_representation is caller-supplied (rather than implicitly
   self.z_goal) so this primitive stays agnostic about which representation
   counts as 'the attained subgoal' -- that is an experiment-design
   decision ... left to the call site, not baked into the substrate". So
   agent.goal_state.z_goal_parent IS the pre-arrival-credited parent, and
   C1' reads THAT, not a replay. The replay survives only as C0, a FIDELITY
   CONTROL: it checks that the driver's attained-tick set equals the
   substrate's credited-tick set and that the replay arithmetic matches, so
   a C0 failure routes to substrate_not_ready_requeue instead of being
   silently attributed to MECH-428. This is the control F2 asked for.

F1 (unpassable absolute floor). CONTENT_DELTA_ABS_FLOOR=0.05 is WITHDRAWN,
   with the measurement that shows why: the parent-level statistic's
   achievable magnitude is 6.6e-4 to 3.4e-3, two orders below it. The
   criterion is now NULL-REFERENCED -- T must exceed the 95th percentile of
   its OWN label-permutation null -- which is scale-free and therefore
   cannot inherit this defect. T_ABS_FLOOR (2e-4) is retained only as a
   DEGENERATE-NULL guard and is deliberately NON-BINDING: it sits 3.3x
   below the smallest T measured in (3). DISCLOSURE: (3) was measured on
   the same seeds this experiment scores, so T_ABS_FLOOR is a
   floor-of-convenience, not an independently calibrated bar; the
   discriminating test is the null reference, which is a pre-registered
   PROCEDURE computed from the scored run's own data rather than a fitted
   number.

F3 (G3 certified noise). G3 is now referenced against its own 200-draw
   label-permutation null instead of the absolute 0.0002 floor that 96
   percent of random partitions cleared. Same statistic, falsifiable bar.

THE NEW NEGATIVE-CONTROL ARM
---------------------------------------------------------------------------
CREDIT_RANDOM_TICKS credits the SAME NUMBER of pre-arrival representations,
through the SAME real substrate call, at randomly chosen NON-ATTAINMENT
ticks (seed-derived, drawn once per seed before the walk). It is the
between-arm answer to "the manipulation cannot reach the DV" and to "any
crediting whatsoever would produce this alignment": if T is equally positive
there, the SUBGOAL_BOOTSTRAP reading is not specific to attainment and the
run routes non_contributory rather than supports. C2 is that test.

DV-SYMMETRY INVARIANCE, per arm (Step 3.5 requirement)
---------------------------------------------------------------------------
NO_SUBGOAL: no manipulation (parent never credited); scored only as a
  readiness/parity reference, never as a DV-bearing arm.
SUBGOAL_BOOTSTRAP: the manipulation is WHICH ticks are credited. The DV is
  T, a difference of two cosines of a FIXED parent against two group means.
  Symmetry group of T: relabelling the attained/non-attained partition --
  which is exactly what the permutation null enumerates. The manipulation is
  NOT invariant under it: crediting a different tick set produces a
  different parent, hence a different T. T is not a set-aggregate over the
  credited multiset either (the parent is an order- and recency-weighted
  EMA), so a permutation of credit ORDER does move it.
CREDIT_RANDOM_TICKS: same DV, same symmetry group; the manipulation is the
  null condition by construction, so invariance here is the PREDICTION, not
  a defect.
FORCED_SEED: DV is parent_goal_norm, symmetric under nothing relevant;
  FORCED_CREDIT=20 saturates a=min(1, alpha*credit) to 1.0 so parent=z_world
  exactly on every credited tick -- so G1 tests ||z_world|| >= floor, not
  structure. Carried forward from 884b with that limitation restated rather
  than repaired (884b minor finding), and G1 is a readiness gate only.

DESIGN CHOICE (unchanged from 884a/884b, restated): scripted greedy walk,
not a learned navigation policy -- the basal-ganglia commitment substrate is
still under construction, MECH-428's own text frames the mechanism as
EMA-pull/decay arithmetic rather than navigation competence, and GAP-2
sparse-seeding is reproduced by the substrate fact that _z_goal_parent has
exactly one write path. This script is about the PARENT attractor's CONTENT
and makes no claim about downstream goal-directed behaviour. The scripted
walk never calls agent.select_action.

KNOWN LIMITATIONS THE RUN HAPPENS UNDER (Step 2.5c, severity=degrading)
---------------------------------------------------------------------------
SD-018 and SD-106 (ree_core/latent/stack.py, ree_core/latent/zworld_p0.py)
and SD-ZWORLD-SENSE-PATH-PARITY (ree_core/agent.py::REEAgent.sense,
zworld_p0.py::ZWorldP0Trainer._z_world_path,
experiments/_lib/zworld_p0_warmup.py) are open degrading entries on paths
this driver exercises. SD-ZWORLD-SENSE-PATH-PARITY is the one to weigh when
reading a result: it is about the z_world sense path differing between the
P0 trainer and the agent, which is the same seam alpha_world sits on. No
open severity=corrupting entry is reached -- measured, not assumed, by
experiments/_scratch/_probe_884c_reachcheck2.py over 80 steps of this exact
loop (tonic_vigor, e1_deep ContextMemory.write, affect/blocked_agency and
regulators/selection_entropy_floor are all reached 0 times).

Biological basis (unchanged): Bandura & Schunk (1981) -- proximal subgoals
CREATED intrinsic interest/goal pursuit in initially-uninterested learners.

PASS (supports MECH-428) = G0 AND G1 AND G2 AND G3 AND C0 AND C1 AND C2.
G0/G1/G2/G3/C0 fail -> non_contributory (readiness or instrument, never a
  MECH-428 verdict).
C2 fail (the random-tick control aligns too) -> non_contributory: the
  alignment is not specific to attainment, so nothing is attributable.
G0-G3, C0, C2 pass and C1 fails -> weakens: attainment occurred, the
  substrate carries content-differentiated representations, the replay is
  faithful, the negative control is silent -- and the parent built by the
  substrate's own crediting path still did not align with the attained
  group beyond chance labelling.

ethics_preflight:
  involves_negative_valence: false        # no hazards/resources in this config; harm stream unused
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
import random
import zlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from ree_core.environment.causal_grid_world import CausalGridWorld

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from ree_core.latent.zworld_p0 import ZWorldP0Config  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    assert_world_encoder_trained,
    zworld_precondition,
    ZWorldEncoderUntrainedError,
)

# --------------------------------------------------------------------- #
# Experiment metadata
# --------------------------------------------------------------------- #
EXPERIMENT_TYPE = "v3_exq_884c_mech428_subgoal_bootstrapped_goal_seeding"
QUEUE_ID = "V3-EXQ-884c"
SUPERSEDES = "V3-EXQ-884"  # 884a and 884b were both parked, never queued -- 884 is the last QUEUED predecessor
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-428"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
ARMS = ["NO_SUBGOAL", "SUBGOAL_BOOTSTRAP", "CREDIT_RANDOM_TICKS", "FORCED_SEED"]
# ORDER IS LOAD-BEARING, not cosmetic: CREDIT_RANDOM_TICKS sizes its credit
# budget from SUBGOAL_BOOTSTRAP's measured attainment count on the same seed,
# so it MUST run after it. If the order were reversed the control would credit
# zero times, C2 ("the control is silent") would pass VACUOUSLY, and a C1 PASS
# would look attributable when nothing had controlled it. Asserted at import
# rather than trusted.
assert ARMS.index("SUBGOAL_BOOTSTRAP") < ARMS.index("CREDIT_RANDOM_TICKS"), (
    "ARMS order is load-bearing: CREDIT_RANDOM_TICKS takes its credit budget "
    "from SUBGOAL_BOOTSTRAP's count and must run after it"
)
STAY_ACTION = 4  # (dx, dy) = (0, 0) in CausalGridWorld.ACTIONS -- a true no-op

# --- Env config (unchanged from 884a) ---------------------------------- #
GRID_SIZE = 12
NUM_WAYPOINTS = 3
N_STEPS = 400
N_STEPS_DRY = 80
MEASUREMENT_WINDOW = 60

WORLD_DIM = 32
# SD-008 (ree_core/latent/stack.py:1536-1539, ree_core/utils/config.py:80-84).
# 884b never set this and therefore ran at the REEConfig default 0.3, which
# blends 70 percent of the PREVIOUS tick into every sensed z_world. Measured
# in experiments/_scratch/_probe_884c_encoded_ceiling.py: at 0.3 the
# attained-vs-transit separation is AT-CHANCE for BOTH the post-arrival and
# the pre-arrival credited representation, on all three seeds. Fixed BEFORE
# the run, never tuned against it.
ALPHA_WORLD = 0.9
PARENT_GOAL_ALPHA = 0.05
PARENT_GOAL_DECAY = 0.005
FORCED_CREDIT = 20.0

# --- P0 warmup config (NEW) --------------------------------------------- #
P0_EPISODES = 20
P0_STEPS_PER_EPISODE = 50
P0_EPISODES_DRY = 2
P0_STEPS_PER_EPISODE_DRY = 10
P0_PROBE_STEPS = N_STEPS_DRY  # G3 dedicated readiness probe, real run and dry-run alike

# SD-106 generic bottleneck variance preservation (scale-normalised reconstruction,
# NOT waypoint-specific -- see module docstring point 1). Empirically required and
# empirically calibrated (2026-09-14 scratch probes, seed 42, real P0_EPISODES/
# P0_STEPS_PER_EPISODE scale, all against G3's FINAL statistic -- mean-direction
# dissimilarity between the waypoint-tick group and the transit-tick group, NOT the
# two earlier, empirically-refuted pairwise-dissimilarity versions; see module
# docstring point 2 for why those were wrong):
#   untrained (P0_EPISODES=0):        0.0000669
#   preservation_weight=200:          0.000242   (~3.6x untrained)
#   preservation_weight=1000:         0.000731   (~11x untrained)
# 1000 is adopted for real headroom over the untrained floor, not the module
# docstring's own cited "200 + skip" operating point (which is calibrated for a
# DIFFERENT statistic -- world_obs reconstruction R^2 -- and was a bare ~3.6x
# separation on THIS statistic, too thin a margin to certify readiness on).
P0A_PRESERVATION_WEIGHT = 1000.0

# --- Pre-registered readiness thresholds --------------------------------
# 884a's own precedent: thresholds are calibrated against a SEPARATE dry
# probe measurement of this harness, never against the scored run's own
# statistics. FORCED_STRUCTURED_FLOOR / PARENT_NORM_ABS_FLOOR are unchanged
# from 884a. ZWORLD_CONTENT_DISSIM_FLOOR is new, calibrated on G3's FINAL
# statistic (mean-direction dissimilarity -- see P0A_PRESERVATION_WEIGHT
# comment above for the full three-point calibration table: untrained
# 0.0000669, weight=200 0.000242, weight=1000 0.000731, all seed 42,
# 2026-09-14, real P0 scale, measured BEFORE this threshold was fixed). Set
# at ~3x the untrained baseline (0.0000669 * 3 ~= 0.0002), so a genuine
# regression toward the untrained/collapsed regime trips it, while the
# adopted preservation_weight=1000 operating point (0.000731) clears it with
# > 3.5x headroom -- a comfortable, not razor-thin, margin. This floor is a
# PROPERTY OF THE SUBSTRATE (trained vs. untrained separation), fixed BEFORE
# any scored run -- NOT tuned to make a scored result pass. Recorded for the
# honest case this settles: even at this calibration, the WITHIN-run
# separation between attained/non-attained z_world content is small in
# absolute terms (1e-4 to 1e-3 scale) -- if a future session needs a larger
# operating margin, that is a P0-recipe / environment-design question, not a
# reason to lower this floor post hoc.
FORCED_STRUCTURED_FLOOR = 0.05
PARENT_NORM_ABS_FLOOR = 0.01
ZWORLD_CONTENT_DISSIM_FLOOR = 0.0002

# --- C1/C2 (the content criterion) -------------------------------------
# 884b's CONTENT_DELTA_ABS_FLOOR = 0.05 is WITHDRAWN. Red-team F1 derived it
# to be 70-250x the achievable ceiling from first principles; this session
# measured the ceiling directly at the parent level and confirms it
# (6.6e-4 to 3.4e-3 across seeds 42/43/44 -- see module docstring
# measurement 3). An absolute bar at that scale is the defect, not the fix,
# so the criterion is NULL-REFERENCED instead: T must exceed the 95th
# percentile of its OWN label-permutation null. That bar is scale-free and
# is a pre-registered PROCEDURE, not a fitted number.
N_PERMUTATIONS = 400      # label-permutation draws for the C1/C2/G3 nulls
NULL_PERCENTILE = 0.95

# Retained ONLY as a degenerate-null guard: if the permutation null ever
# collapses to ~zero spread, a trivially small T would "separate". Set at
# 2e-4, which is 3.3x BELOW the smallest T measured in the part-3 probe
# (seed 43, 0.000662), so it is deliberately NON-BINDING relative to the
# null reference. DISCLOSURE: that probe ran on the same seeds this
# experiment scores, so this number is a floor-of-convenience -- the
# discriminating test is the null reference, not this.
T_ABS_FLOOR = 0.0002

# C0 fidelity: cos(replayed parent, the REAL substrate parent). Measured
# 1.000000000 / 0.999999881 / 1.000000119 on seeds 42/43/44 before this
# threshold was fixed. 0.999 leaves four orders of float headroom while
# still failing loudly if the driver's credited-tick set ever diverges from
# the substrate's.
C0_FIDELITY_FLOOR = 0.999

MIN_SEEDS_PASS = 2  # of 3 -- ">= 2/3 seeds"


# --------------------------------------------------------------------- #
# Env + agent builders (env builder unchanged from 884a)
# --------------------------------------------------------------------- #

def _build_env(seed: int) -> CausalGridWorld:
    env = CausalGridWorld(
        size=GRID_SIZE,
        num_hazards=0,
        num_resources=0,
        subgoal_mode=True,
        num_waypoints=NUM_WAYPOINTS,
        seed=seed,
        subgoal_arrival_position_check=True,
        hazard_free_contamination_gate=True,
    )
    assert env.subgoal_arrival_position_check is True, (
        "SD-094 (a) arrival-by-position check did not take effect"
    )
    assert getattr(env, "_contamination_gate_applied", False) is True, (
        "SD-094 (b) hazard-free contamination gate did not fire "
        f"(contamination_spread={env.contamination_spread})"
    )
    assert float(env.contamination_spread) == 0.0, (
        f"SD-094 (b) contamination_spread is {env.contamination_spread}, expected 0.0"
    )
    return env


def _build_agent(env: CausalGridWorld) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        world_dim=WORLD_DIM,
        z_goal_enabled=True,
        # SD-008 -- see ALPHA_WORLD above. Asserted below because
        # REEConfig.from_dims silently swallows unknown kwargs
        # ([memory] reference_reeconfig_from_dims_silent_kwargs), which
        # would leave this at the 0.3 default with no error at all.
        alpha_world=ALPHA_WORLD,
        # SD-106: zero-initialised linear bypass around the encoder's ReLU --
        # bit-identical to OFF until the P0 preservation head pulls on it (see
        # _run_p0_and_g3 / P0A_CONFIG below). Needed for the preservation term
        # to have somewhere to push; see module docstring point 1.
        use_world_encoder_skip=True,
    )
    got = float(getattr(cfg.latent, "alpha_world", -1.0))
    assert abs(got - ALPHA_WORLD) < 1e-9, (
        "SD-008 alpha_world did not reach cfg.latent (from_dims swallowed the "
        f"kwarg): got {got!r}, expected {ALPHA_WORLD!r}"
    )
    agent = REEAgent(cfg)
    agent.goal_state.config.use_hierarchical_goal_credit = True
    return agent


def _scripted_action(env: CausalGridWorld) -> int:
    """Ground-truth greedy walk toward the currently-targeted waypoint (verbatim from 884a)."""
    sub = env.get_subgoal_state()
    idx = sub["next_waypoint_idx"]
    if not env.waypoints or idx >= len(env.waypoints):
        return STAY_ACTION
    wx, wy = env.waypoints[idx]
    ax, ay = env.get_agent_position()
    if ax != wx:
        return 1 if wx > ax else 0
    if ay != wy:
        return 3 if wy > ay else 2
    return STAY_ACTION


def _cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b) / denom)


# --------------------------------------------------------------------- #
# P0 phase (NEW): once per seed, shared across all 3 arms
# --------------------------------------------------------------------- #

def _run_p0_and_g3(seed: int, dry_run: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Train z_world on a dedicated warmup env, then run the G3 readiness
    probe. Returns (trained_latent_stack_state_dict, p0_g3_report)."""
    n_ep = P0_EPISODES_DRY if dry_run else P0_EPISODES
    n_steps_ep = P0_STEPS_PER_EPISODE_DRY if dry_run else P0_STEPS_PER_EPISODE

    warmup_env = _build_env(seed)
    master_agent = _build_agent(warmup_env)
    before = latent_stack_snapshot(master_agent)

    p0a_report = run_zworld_p0(
        master_agent,
        warmup_env,
        seed=seed,
        episodes=n_ep,
        steps_per_episode=n_steps_ep,
        policy=RandomPolicy(seed),
        label="mech428c1_p0",
        dry_run=dry_run,
        # target_fn left None -- deliberately generic, never waypoint-specific;
        # see module docstring point 1. config= adds the SD-106 preservation
        # term on top of the SD-070 defaults (empirically required -- see the
        # P0A_PRESERVATION_WEIGHT comment above).
        config=ZWorldP0Config(preservation_weight=P0A_PRESERVATION_WEIGHT),
    )

    guard_report: Dict[str, Any]
    try:
        guard_report = assert_world_encoder_trained(
            master_agent, before, p0=n_ep, strict=False,
            context="V3-EXQ-884c P0 (generic recipe, hazard/resource-free env)",
        )
        g3b_untrained_error = ""
    except ZWorldEncoderUntrainedError as exc:  # pragma: no cover -- strict=False above
        guard_report = {"zworld_encoder_trained": False, "guard_checked": True}
        g3b_untrained_error = str(exc)

    trained_state_dict = {
        k: v.detach().clone() for k, v in master_agent.latent_stack.state_dict().items()
    }

    # --- G3: dedicated post-P0 readiness probe. SAME statistic C1' actually
    #     routes on: C1' asks whether crediting the ATTAINED group produces a
    #     DIFFERENT parent than crediting the NON-ATTAINED group, so G3 must
    #     gate on BETWEEN-group separation (attained-tick reps vs
    #     non-attained-tick reps), not within-group separation among
    #     waypoint reps alone. An earlier version of this gate measured only
    #     within-group dissimilarity (mean pairwise among waypoint reps) --
    #     it passed (0.011-0.015) while a real-scale end-to-end measurement
    #     (2026-09-14, seed 42) showed the PARENT built from either group is
    #     ~99.93% cosine-similar regardless of which group was credited
    #     (delta_group ~= -0.00005) -- i.e. the within-group statistic did
    #     not certify the channel C1' actually needs. Fresh env, fresh agent
    #     carrying the just-trained encoder weights, no further training, no
    #     crediting.
    probe_env = _build_env(seed)
    probe_agent = _build_agent(probe_env)
    probe_agent.latent_stack.load_state_dict(trained_state_dict)

    obs_flat, _obs_dict = probe_env.reset()
    probe_agent.act(obs_flat)
    waypoint_reps: List[torch.Tensor] = []
    transit_reps: List[torch.Tensor] = []
    n_probe_steps = P0_PROBE_STEPS
    for _step in range(n_probe_steps):
        action_idx = _scripted_action(probe_env)
        obs_flat, _harm, done, info, _obs_dict = probe_env.step(action_idx)
        probe_agent.act(obs_flat)
        ttype = info.get("transition_type", "none")
        z_now = probe_agent._current_latent.z_world.detach().clone()
        if ttype in ("waypoint", "sequence_complete"):
            waypoint_reps.append(z_now)
        else:
            transit_reps.append(z_now)
        if done:
            break

    # PAIRWISE dissimilarity (kept, recorded) measures per-SAMPLE noise, not
    # whether the two GROUPS' MEANS differ systematically -- and C1'/delta_group
    # is built from a (recency-weighted) AVERAGE over each group, so it is the
    # MEAN-DIRECTION separation that actually predicts whether crediting one
    # group's samples produces a different EMA than crediting the other's.
    # Two groups can have high pairwise dissimilarity (noisy individual
    # samples) while their means still coincide (the noise averages out) --
    # exactly what a real-scale end-to-end check found here (2026-09-14,
    # seed 42): pairwise between-group dissimilarity read 0.017 (comfortably
    # "ready" by that statistic) while the actual parent-level delta_group
    # measured -0.00005 (no discrimination at all). g3_measured below is
    # therefore the MEAN-DIRECTION dissimilarity, the SAME-STATISTIC gate for
    # C1'; pairwise dissimilarity is kept only as a secondary diagnostic.
    n_pairs = 0
    dissim_sum = 0.0
    for wr in waypoint_reps:
        for tr in transit_reps:
            dissim_sum += 1.0 - _cos_sim(wr, tr)
            n_pairs += 1
    g3_pairwise_dissim = (dissim_sum / n_pairs) if n_pairs > 0 else 0.0

    # G3, NULL-REFERENCED (red-team F3). The statistic is unchanged -- the
    # mean-direction dissimilarity between the waypoint-tick group and the
    # transit-tick group, which is the same statistic C1 routes on -- but the
    # BAR is now the 95th percentile of its own label-permutation null
    # instead of the absolute ZWORLD_CONTENT_DISSIM_FLOOR that 96 percent of
    # random partitions cleared. The old floor is still recorded, as a
    # non-gating diagnostic, so the two bars stay comparable across letters.
    if len(waypoint_reps) >= 2 and len(transit_reps) >= 2:
        g3_measured = _mean_direction_dissim(waypoint_reps, transit_reps)
        g3_null = _permutation_null(
            _mean_direction_dissim, waypoint_reps + transit_reps,
            len(waypoint_reps), seed, "g3",
        )
        g3_null_p95 = _null_p95(g3_null)
        g3_null_median = g3_null[len(g3_null) // 2]
        g3_null_pct = _null_pct(g3_measured, g3_null)
        g3_pass = bool(g3_measured > g3_null_p95)
    else:
        g3_measured = 0.0
        g3_null_p95 = 0.0
        g3_null_median = 0.0
        g3_null_pct = 0.0
        g3_pass = False

    report = {
        "p0a_report": p0a_report,
        "guard_report": {
            "zworld_encoder_trained": bool(guard_report.get("zworld_encoder_trained", False)),
            "world_encoder_max_abs_delta": float(guard_report.get("world_encoder_max_abs_delta", 0.0)),
            "n_world_encoder_changed": int(guard_report.get("n_world_encoder_changed", 0)),
            "guard_checked": bool(guard_report.get("guard_checked", False)),
        },
        "g3b_untrained_error": g3b_untrained_error,
        "g3_zworld_content_dissim_measured": g3_measured,
        "g3_zworld_content_dissim_threshold": g3_null_p95,
        "g3_null_p95": g3_null_p95,
        "g3_null_median": g3_null_median,
        "g3_null_percentile_of_observed": g3_null_pct,
        "g3_n_permutations": N_PERMUTATIONS,
        "g3_legacy_884b_abs_floor": ZWORLD_CONTENT_DISSIM_FLOOR,
        "g3_clears_legacy_884b_abs_floor": bool(g3_measured >= ZWORLD_CONTENT_DISSIM_FLOOR),
        "g3_pairwise_dissim_secondary": g3_pairwise_dissim,
        "g3_n_waypoint_reps_probed": len(waypoint_reps),
        "g3_n_transit_reps_probed": len(transit_reps),
        "g3_n_pairs": n_pairs,
        "g3_pass": g3_pass,
    }
    return trained_state_dict, report


# --------------------------------------------------------------------- #
# One seed x arm cell (per-tick z_world logging added for SUBGOAL_BOOTSTRAP)
# --------------------------------------------------------------------- #

def _run_cell(
    arm: str,
    seed: int,
    zg: ZGoalStreamAccumulator,
    dry_run: bool,
    trained_state_dict: Dict[str, torch.Tensor],
    n_target_credits: int = 0,
) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    n_steps = N_STEPS_DRY if dry_run else N_STEPS

    env = _build_env(seed)
    agent = _build_agent(env)
    agent.latent_stack.load_state_dict(trained_state_dict)  # NEW: shared P0-trained encoder

    obs_flat, _obs_dict = env.reset()
    agent.act(obs_flat)
    # PRE-ARRIVAL credited representation (red-team F4 + measurement 2): the
    # latent sensed at tick t-1, while the agent was still adjacent to -- not
    # on top of -- the target cell, so the waypoint marker is still in
    # local_view. Carried forward one tick and handed to the REAL substrate
    # call below, never replayed outside it (F2).
    prev_z = agent._current_latent.z_world.detach().clone()

    # CREDIT_RANDOM_TICKS: candidate ticks are drawn ONCE, BEFORE the walk,
    # from a seed-derived RNG, so the schedule is fixed independently of what
    # the walk does. Attainment ticks are skipped when reached (below), and
    # crediting stops once n_target_credits have been applied -- so this arm
    # applies EXACTLY as many credits as SUBGOAL_BOOTSTRAP did on the same
    # seed (n_target_credits is that arm's measured count, passed in by the
    # caller; the walk is scripted and seeded, so the two arms traverse an
    # identical trajectory and the count is exact, not an estimate).
    random_tick_rng = random.Random("randomticks884c-%d" % seed)
    random_tick_order = list(range(1, n_steps + 1))
    random_tick_rng.shuffle(random_tick_order)
    random_tick_set = set(random_tick_order)
    n_random_credits_applied = 0

    n_subgoal_credits = 0
    n_waypoint_events = 0
    n_sequence_complete_events = 0
    parent_norm_trace: List[float] = []
    steps_taken = 0
    done = False
    done_cause = ""

    # NEW: per-tick content logging, kept only for SUBGOAL_BOOTSTRAP (the arm
    # the content DV is computed on) to avoid unnecessary memory/compute on
    # the other two arms.
    # Both DV-bearing arms log content: SUBGOAL_BOOTSTRAP is the treatment,
    # CREDIT_RANDOM_TICKS is the negative control C2 reads.
    log_content = arm in ("SUBGOAL_BOOTSTRAP", "CREDIT_RANDOM_TICKS")
    attained_reprs: List[torch.Tensor] = []
    non_attained_reprs: List[torch.Tensor] = []
    credit_tick_positions: List[int] = []
    event_tick_positions: List[int] = []  # every waypoint/sequence_complete tick (== credit ticks here)

    for _step in range(n_steps):
        action_idx = _scripted_action(env)
        obs_flat, _harm_signal, done, info, _obs_dict = env.step(action_idx)
        steps_taken += 1
        done_cause = str(info.get("done_cause", "") or "")
        agent.act(obs_flat)
        agent.update_z_goal(benefit_exposure=0.0, drive_level=0.0)

        ttype = info.get("transition_type", "none")
        if ttype == "waypoint":
            n_waypoint_events += 1
        elif ttype == "sequence_complete":
            n_sequence_complete_events += 1

        if arm == "NO_SUBGOAL":
            credit_result = {}
        elif arm == "SUBGOAL_BOOTSTRAP":
            # child_representation=prev_z is the PRE-ARRIVAL latent. This is
            # a sanctioned degree of freedom, not a substrate change:
            # GoalState.credit_subgoal_attainment's docstring states that
            # "child_representation is caller-supplied ... which
            # representation counts as 'the attained subgoal' ... is an
            # experiment-design decision ... left to the call site".
            credit_result = agent.notify_subgoal_attainment(
                ttype, child_representation=prev_z
            )
        elif arm == "CREDIT_RANDOM_TICKS":
            assert int(n_target_credits) > 0, (
                "CREDIT_RANDOM_TICKS was handed a zero credit budget -- the "
                "negative control would credit nothing and C2 would pass "
                "vacuously. Check the ARMS ordering assert at the top of this "
                "module."
            )
            # NEGATIVE CONTROL: same substrate call, same pre-arrival
            # representation, same credit magnitude -- but at ticks chosen
            # WITHOUT REGARD TO ATTAINMENT. A tick that happens to BE an
            # attainment tick is skipped, so this arm credits strictly
            # non-attainment ticks and cannot borrow the treatment's signal.
            is_attainment = ttype in ("waypoint", "sequence_complete")
            budget_left = n_random_credits_applied < int(n_target_credits)
            if budget_left and steps_taken in random_tick_set and not is_attainment:
                credit_result = agent.notify_subgoal_attainment(
                    "waypoint", child_representation=prev_z
                )
                n_random_credits_applied += 1
            else:
                credit_result = {}
        else:  # FORCED_SEED
            credit_result = agent.notify_subgoal_attainment(
                "sequence_complete", credit=FORCED_CREDIT
            )

        if credit_result:
            n_subgoal_credits = credit_result["n_subgoal_credits"]

        if log_content:
            # PRE-ARRIVAL representation, matching what was actually credited.
            if ttype in ("waypoint", "sequence_complete"):
                attained_reprs.append(prev_z)
                event_tick_positions.append(steps_taken)
                if arm == "SUBGOAL_BOOTSTRAP":
                    credit_tick_positions.append(steps_taken)
            else:
                non_attained_reprs.append(prev_z)
                if arm == "CREDIT_RANDOM_TICKS" and steps_taken in random_tick_set:
                    credit_tick_positions.append(steps_taken)

        parent_norm_trace.append(float(agent.goal_state.parent_goal_norm()))
        # Advance the one-tick delay LAST, so prev_z is tick t-1 throughout
        # the body above.
        prev_z = agent._current_latent.z_world.detach().clone()

        if done:
            break

    window = parent_norm_trace[-MEASUREMENT_WINDOW:] if parent_norm_trace else [0.0]
    steady_state_norm = float(sorted(window)[len(window) // 2])
    final_parent_norm = float(agent.goal_state.parent_goal_norm())
    final_parent_vec = (
        agent.goal_state.z_goal_parent.detach().clone()
        if agent.goal_state.z_goal_parent is not None
        else None
    )
    zg.observe(agent)

    budget_frac = float(steps_taken) / float(n_steps) if n_steps else 0.0
    budget_reached = bool(steps_taken >= n_steps)

    print(
        f"  [eval] {arm} seed={seed} n_subgoal_credits={n_subgoal_credits} "
        f"n_waypoint_events={n_waypoint_events} "
        f"n_sequence_complete_events={n_sequence_complete_events} "
        f"steady_state_parent_norm={steady_state_norm:.4f} "
        f"steps_actual={steps_taken}/{n_steps} "
        f"done={done} done_cause='{done_cause or 'none'}' "
        f"agent_health={float(env.agent_health):.3f}",
        flush=True,
    )
    print(f"  [train] {arm} seed={seed} ep 1/1", flush=True)
    print("verdict: PASS", flush=True)

    row: Dict[str, Any] = {
        "seed": seed,
        "arm": arm,
        "n_subgoal_credits": n_subgoal_credits,
        "n_random_credits_applied": n_random_credits_applied,
        "n_target_credits": int(n_target_credits),
        "n_waypoint_events": n_waypoint_events,
        "n_sequence_complete_events": n_sequence_complete_events,
        "parent_goal_norm_final": final_parent_norm,
        "parent_goal_norm_steady_state": steady_state_norm,
        "parent_goal_norm_trace": parent_norm_trace,
        "n_steps": n_steps,
        "n_steps_configured": n_steps,
        "n_steps_actual": steps_taken,
        "episode_budget_frac": budget_frac,
        "episode_budget_reached": budget_reached,
        "episode_done": bool(done),
        "episode_done_cause": done_cause,
        "agent_health_final": float(env.agent_health),
    }
    if log_content:
        row["_content_final_parent_vec"] = final_parent_vec
        row["_content_attained_reprs"] = attained_reprs
        row["_content_non_attained_reprs"] = non_attained_reprs
        row["_content_credit_tick_positions"] = credit_tick_positions
        row["_content_event_tick_positions"] = event_tick_positions
        row["_content_n_steps"] = steps_taken
    return row


def _mean_vec(group: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack([g.reshape(-1).float() for g in group]).mean(dim=0)


def _mean_direction_dissim(a: List[torch.Tensor], b: List[torch.Tensor]) -> float:
    """1 - cos(mean(a), mean(b)). G3's statistic."""
    return 1.0 - _cos_sim(_mean_vec(a), _mean_vec(b))


def _t_statistic(parent: torch.Tensor,
                 attained: List[torch.Tensor],
                 non_attained: List[torch.Tensor]) -> float:
    """C1/C2's statistic: how much more the parent aligns with the ATTAINED
    group's mean than with the NON-ATTAINED group's mean.

    Both terms are cosines of the SAME fixed parent, so the large shared
    direction that 884b's raw parent-vs-mean comparison foundered on
    (alignment ~0.9995 against every control alike) cancels to first order in
    the DIFFERENCE, and whatever does not cancel is exactly what the
    label-permutation null below absorbs. No centering, and no resampled
    control parent -- the resampled form was measured AT-CHANCE on 3/3 seeds
    because an EMA's effective sample size (~1/parent_goal_alpha = 20) gives
    the null more spread than the signal (module docstring, measurement 3)."""
    return (_cos_sim(parent, _mean_vec(attained))
            - _cos_sim(parent, _mean_vec(non_attained)))


def _permutation_null(stat_fn,
                      pool: List[torch.Tensor],
                      n_a: int,
                      seed: int,
                      tag: str) -> List[float]:
    """N_PERMUTATIONS random relabellings of the SAME representations into
    groups of the SAME sizes. This is red-team F3's own test, promoted from a
    post-hoc diagnostic to the bar the criteria are read against: F3 showed
    884b's absolute G3 floor was cleared by 96 percent of random partitions,
    which is what 'certifies noise' means. A null-referenced bar cannot
    inherit that defect, and it is scale-free, so it also cannot inherit
    F1's unreachable-absolute-floor defect."""
    # zlib.crc32 over a stable string, NOT hash((tag, seed)): str.__hash__ is
    # salted per process by PYTHONHASHSEED, so a hash-derived generator seed
    # would make the nulls -- and therefore the PASS/FAIL verdict -- differ
    # between two runs of this script on identical data.
    gen = torch.Generator().manual_seed(
        zlib.crc32(("%s-%d" % (tag, seed)).encode("ascii")) % (2 ** 31)
    )
    n_total = len(pool)
    null: List[float] = []
    for _ in range(N_PERMUTATIONS):
        perm = torch.randperm(n_total, generator=gen).tolist()
        null.append(stat_fn([pool[i] for i in perm[:n_a]],
                            [pool[i] for i in perm[n_a:]]))
    null.sort()
    return null


def _null_p95(null: List[float]) -> float:
    return null[int(NULL_PERCENTILE * len(null))]


def _null_pct(value: float, null: List[float]) -> float:
    return 100.0 * sum(1 for v in null if v < value) / len(null)


def _replay_parent_from_reprs(
    reprs_to_credit: List[torch.Tensor],
    credit_tick_positions: List[int],
    n_steps: int,
    goal_dim: int,
) -> torch.Tensor:
    """Replay GoalState's own decay+credit recursion (verbatim from
    ree_core/goal.py: parent *= (1-decay) every tick; parent =
    (1-a)*parent + a*z on a credit tick) OUTSIDE the substrate, crediting
    `reprs_to_credit` IN ORDER at the given credit-tick positions. The
    caller controls both WHICH representations are credited and their
    ORDER; this function only replays the arithmetic faithfully."""
    parent = torch.zeros(goal_dim, dtype=torch.float32)
    credit_set = set(credit_tick_positions)
    k = 0
    a = min(1.0, PARENT_GOAL_ALPHA * 1.0)  # default credit=1.0, matches SUBGOAL_BOOTSTRAP
    for tick in range(1, n_steps + 1):
        parent = parent * (1.0 - PARENT_GOAL_DECAY)
        if tick in credit_set and k < len(reprs_to_credit):
            z = reprs_to_credit[k].reshape(-1).float()
            parent = (1.0 - a) * parent + a * z
            k += 1
    return parent


# --------------------------------------------------------------------- #
# Aggregate + acceptance criteria
# --------------------------------------------------------------------- #

def run(dry_run: bool = False) -> tuple:
    print(f"\n[{QUEUE_ID}] MECH-428 Subgoal-Bootstrapped Goal Seeding "
          f"(content-DV redesign, P0-conditioned)", flush=True)

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict] = []
    per_seed: Dict[str, Dict[int, Dict]] = {a: {} for a in ARMS}
    p0_g3_by_seed: Dict[int, Dict[str, Any]] = {}

    for seed in SEEDS:
        trained_state_dict, p0_g3_report = _run_p0_and_g3(seed, dry_run)
        p0_g3_by_seed[seed] = p0_g3_report
        n_target_credits_this_seed = 0

        for arm in ARMS:
            config_slice = {
                "arm": arm, "seed": seed, "dry_run": dry_run,
                "grid_size": GRID_SIZE, "num_waypoints": NUM_WAYPOINTS,
                "n_steps": N_STEPS_DRY if dry_run else N_STEPS,
                "world_dim": WORLD_DIM, "forced_credit": FORCED_CREDIT,
                "subgoal_arrival_position_check": True,
                "hazard_free_contamination_gate": True,
                "p0_trained": True,
                "p0_episodes": P0_EPISODES_DRY if dry_run else P0_EPISODES,
                "p0_steps_per_episode": P0_STEPS_PER_EPISODE_DRY if dry_run else P0_STEPS_PER_EPISODE,
            }
            with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
                row = _run_cell(
                    arm, seed, zg, dry_run=dry_run,
                    trained_state_dict=trained_state_dict,
                    n_target_credits=n_target_credits_this_seed,
                )
                cell.stamp({k: v for k, v in row.items() if not k.startswith("_content_")})
            if arm == "SUBGOAL_BOOTSTRAP":
                # ARMS is ordered so SUBGOAL_BOOTSTRAP runs BEFORE
                # CREDIT_RANDOM_TICKS; this is what matches the control arm's
                # credit count to the treatment's exactly.
                n_target_credits_this_seed = int(row["n_subgoal_credits"])
            arm_results.append(row)
            per_seed[arm][seed] = row

    # --- G2 (unchanged from 884a): every cell ran its full configured budget.
    g2_failures = [
        f"{r['arm']}/seed{r['seed']}: {r['n_steps_actual']}/{r['n_steps_configured']} "
        f"(done_cause={r['episode_done_cause'] or 'none'}, "
        f"health={r['agent_health_final']:.3f})"
        for r in arm_results
        if not r["episode_budget_reached"]
    ]
    g2_pass = not g2_failures
    g2_min_budget_frac = min(float(r["episode_budget_frac"]) for r in arm_results)

    # --- G3 (NEW): every seed's post-P0 readiness probe cleared the content-
    #     dissimilarity floor.
    g3_failures = [
        f"seed{seed}: dissim={rep['g3_zworld_content_dissim_measured']:.6f} "
        f"(null_p95={rep['g3_null_p95']:.6f}, "
        f"pct_of_own_null={rep['g3_null_percentile_of_observed']:.1f})"
        for seed, rep in p0_g3_by_seed.items()
        if not rep["g3_pass"]
    ]
    g3_pass = not g3_failures

    g0_per_seed, g1_per_seed = [], []
    c0_per_seed: List[bool] = []
    c1_per_seed: List[bool] = []
    c2_per_seed: List[bool] = []
    t_stat_by_seed: Dict[int, Dict[str, float]] = {}
    c0_fidelity_by_seed: Dict[int, float] = {}
    inter_event_interval_by_seed: Dict[int, Optional[float]] = {}

    for seed in SEEDS:
        boot = per_seed["SUBGOAL_BOOTSTRAP"][seed]
        rand = per_seed["CREDIT_RANDOM_TICKS"][seed]
        forced = per_seed["FORCED_SEED"][seed]

        g0_per_seed.append(boot["n_subgoal_credits"] > 0)
        # The negative control is only a control if it actually credited the
        # same number of times. A shortfall (e.g. the random schedule ran out
        # of non-attainment ticks) would make C2 silent for a bookkeeping
        # reason rather than a scientific one.
        assert rand["n_random_credits_applied"] == boot["n_subgoal_credits"], (
            f"seed {seed}: CREDIT_RANDOM_TICKS applied "
            f"{rand['n_random_credits_applied']} credits but SUBGOAL_BOOTSTRAP "
            f"applied {boot['n_subgoal_credits']} -- the C2 control is not "
            "credit-matched and cannot be read as a control"
        )
        g1_per_seed.append(forced["parent_goal_norm_steady_state"] >= FORCED_STRUCTURED_FLOOR)

        # ---- C0: FIDELITY CONTROL (red-team F2) -----------------------
        # The credit went through the REAL substrate call, so
        # boot["_content_final_parent_vec"] IS the pre-arrival-credited
        # parent. C0 replays GoalState's own decay+credit recursion outside
        # the substrate over the SAME representations at the SAME credit-tick
        # positions and asks whether it reproduces that parent. It therefore
        # tests two things at once: that the replay arithmetic matches
        # ree_core/goal.py, and -- the load-bearing half -- that the
        # DRIVER's attained-tick set is the SAME SET the substrate actually
        # credited. A mismatch means the DV is computed over the wrong
        # events, which is an INSTRUMENT failure and must never be read as a
        # MECH-428 verdict.
        boot_parent_vec = boot["_content_final_parent_vec"]
        boot_attained = boot["_content_attained_reprs"]
        boot_non_attained = boot["_content_non_attained_reprs"]
        boot_credit_pos = boot["_content_credit_tick_positions"]
        boot_n_steps = boot["_content_n_steps"]
        boot_events = boot["_content_event_tick_positions"]

        if boot_parent_vec is not None and len(boot_attained) >= 2:
            goal_dim = boot_parent_vec.reshape(-1).shape[0]
            replayed = _replay_parent_from_reprs(
                boot_attained, boot_credit_pos, boot_n_steps, goal_dim,
            )
            c0_fidelity = _cos_sim(boot_parent_vec.reshape(-1).float(), replayed)
        else:
            c0_fidelity = 0.0
        c0_fidelity_by_seed[seed] = c0_fidelity
        c0_per_seed.append(c0_fidelity >= C0_FIDELITY_FLOOR)

        # ---- C1 (treatment) and C2 (negative control) -------------------
        # T = cos(parent, mean(attained)) - cos(parent, mean(non_attained)),
        # each arm read against ITS OWN label-permutation null. C1 asks
        # whether the treatment arm's parent is attainment-specific; C2 asks
        # whether the SAME reading appears when the same number of the same
        # pre-arrival representations are credited at ticks chosen without
        # regard to attainment. C2 is what makes a C1 PASS attributable: if
        # both fire, the alignment is a property of crediting anything, not
        # of crediting ATTAINED subgoals.
        def _score(row: Dict[str, Any], tag: str) -> Dict[str, float]:
            parent_vec = row["_content_final_parent_vec"]
            att = row["_content_attained_reprs"]
            non = row["_content_non_attained_reprs"]
            if parent_vec is None or len(att) < 2 or len(non) < 2:
                return {"t": 0.0, "null_p95": 0.0, "null_median": 0.0,
                        "null_pct": 0.0, "separates": 0.0,
                        "cos_attained": 0.0, "cos_non_attained": 0.0,
                        "n_attained": float(len(att)), "n_non_attained": float(len(non))}
            parent = parent_vec.reshape(-1).float()
            t_obs = _t_statistic(parent, att, non)
            null = _permutation_null(
                lambda a, b: _t_statistic(parent, a, b),
                att + non, len(att), seed, tag,
            )
            p95 = _null_p95(null)
            return {
                "t": t_obs,
                "null_p95": p95,
                "null_median": null[len(null) // 2],
                "null_pct": _null_pct(t_obs, null),
                "separates": 1.0 if (t_obs > p95 and t_obs >= T_ABS_FLOOR) else 0.0,
                "cos_attained": _cos_sim(parent, _mean_vec(att)),
                "cos_non_attained": _cos_sim(parent, _mean_vec(non)),
                "n_attained": float(len(att)),
                "n_non_attained": float(len(non)),
            }

        boot_stat = _score(boot, "c1_boot")
        rand_stat = _score(rand, "c2_rand")
        t_stat_by_seed[seed] = {
            "bootstrap": boot_stat,
            "random_ticks": rand_stat,
        }
        c1_per_seed.append(bool(boot_stat["separates"]))
        # C2 PASSES when the control is SILENT -- note the inversion.
        c2_per_seed.append(not bool(rand_stat["separates"]))

        if len(boot_events) >= 2:
            intervals = [
                boot_events[i] - boot_events[i - 1]
                for i in range(1, len(boot_events))
            ]
            inter_event_interval_by_seed[seed] = sum(intervals) / len(intervals)
        else:
            inter_event_interval_by_seed[seed] = None

    threshold = MIN_SEEDS_PASS / len(SEEDS)
    g0_pass = all(g0_per_seed)
    g0_frac = sum(1 for g in g0_per_seed if g) / len(g0_per_seed)
    g1_frac = sum(1 for g in g1_per_seed if g) / len(g1_per_seed)
    g1_pass = g1_frac >= threshold
    c0_frac = sum(1 for c in c0_per_seed if c) / len(c0_per_seed)
    c0_pass = all(c0_per_seed)   # instrument fidelity: EVERY seed, not 2/3
    c1_frac = sum(1 for c in c1_per_seed if c) / len(c1_per_seed)
    c2_frac = sum(1 for c in c2_per_seed if c) / len(c2_per_seed)
    c1_pass = c1_frac >= threshold
    c2_pass = c2_frac >= threshold

    non_degenerate = True
    degeneracy_reason = None

    if not g2_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "non_degenerate_precondition_unmet"
        non_degenerate = False
        degeneracy_reason = (
            "G2 readiness gate failed: at least one cell's episode was "
            "terminated by the environment before its configured step budget. "
            "Starved cells: " + "; ".join(g2_failures) + ". Not evidence about "
            "MECH-428; diagnose the environment configuration and re-queue "
            "under a new letter."
        )
        interpretation = "Readiness (G2) failed. Not evidence against MECH-428."
    elif not g0_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "non_degenerate_precondition_unmet"
        non_degenerate = False
        degeneracy_reason = (
            "G0 readiness gate failed: ARM_SUBGOAL_BOOTSTRAP did not produce a "
            "subgoal-attainment credit on every seed. Harness/environment "
            "defect, not evidence about MECH-428."
        )
        interpretation = "Readiness (G0) failed. Not evidence against MECH-428."
    elif not g1_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "non_degenerate_precondition_unmet"
        non_degenerate = False
        degeneracy_reason = (
            "G1 readiness gate failed: ARM_FORCED_SEED did not itself produce "
            "a steady-state parent_goal_norm clearing FORCED_STRUCTURED_FLOOR "
            "on >= 2/3 seeds. The substrate, not the bootstrap hypothesis, is "
            "what is not ready."
        )
        interpretation = "Readiness (G1) failed. Not evidence against MECH-428."
    elif not g3_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "G3 readiness gate failed: the post-P0 probe's MEAN-DIRECTION "
            "cosine dissimilarity between the waypoint-tick and transit-tick "
            "z_world groups -- the SAME statistic C1 routes on -- did not "
            "exceed the 95th percentile of its own label-permutation null on "
            "every seed. This is the exact vacuity the P0 phase exists to "
            "prevent: an untrained or collapsed encoder cannot support a "
            "content verdict in either direction, regardless of MECH-428. "
            "Failures: " + "; ".join(g3_failures) + ". Not evidence about "
            "MECH-428; diagnose the P0 recipe (episode/step budget, "
            "preservation/anti-collapse weights, alpha_world) before "
            "re-queuing under a new letter."
        )
        interpretation = "Readiness (G3, P0 content-discriminability) failed. Not evidence against MECH-428."
    elif not c0_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "C0 fidelity control failed: the decay+credit replay over this "
            "driver's recorded attained representations did not reproduce the "
            "REAL substrate parent (agent.goal_state.z_goal_parent) to within "
            f"C0_FIDELITY_FLOOR={C0_FIDELITY_FLOOR} on every seed "
            f"(per-seed cosine: {[round(c0_fidelity_by_seed[s], 9) for s in SEEDS]}). "
            "That means the DV is being computed over a different event set "
            "than the substrate actually credited -- an INSTRUMENT defect, "
            "not evidence about MECH-428. Diagnose the credit-tick bookkeeping "
            "before re-queuing under a new letter."
        )
        interpretation = "Instrument fidelity (C0) failed. Not evidence about MECH-428."
    elif not c2_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "negative_control_fired"
        non_degenerate = False
        degeneracy_reason = (
            "C2 negative control FIRED: the CREDIT_RANDOM_TICKS arm -- same "
            "substrate call, same pre-arrival representations, same number of "
            "credits, at ticks chosen WITHOUT REGARD TO ATTAINMENT -- also "
            "produced a parent whose alignment with the attained-tick group "
            "exceeded its own label-permutation null, on more than 1/3 of "
            "seeds. Whatever the SUBGOAL_BOOTSTRAP arm shows is therefore NOT "
            "specific to subgoal attainment, so no reading of C1 is "
            "attributable to MECH-428 in either direction."
        )
        interpretation = (
            "Negative control (C2) fired -- the alignment is not specific to "
            "attainment. Not evidence about MECH-428 in either direction."
        )
    else:
        # C0 and C2 are already known to have passed on this branch, so the
        # remaining question is purely C1. COMBINATION RULE for the whole run:
        # PASS = G0 AND G1 AND G2 AND G3 AND C0 AND C2 AND C1, with the three
        # seed-quantified criteria (G1, C1, C2) each at >= 2/3 seeds and C0 at
        # 3/3. Recorded machine-readably as combination_rule below.
        all_pass = c1_pass
        status = "PASS" if all_pass else "FAIL"
        evidence_direction = "supports" if all_pass else "weakens"
        route_reason = "clean_scripted_bootstrap_content_scored"
        if all_pass:
            interpretation = (
                "MECH-428 SUPPORTED: the parent (superordinate) attractor "
                "built by the substrate's OWN credit_subgoal_attainment path "
                "aligns with the ATTAINED subgoals' representations more than "
                "with the non-attained ones, beyond the 95th percentile of a "
                "label-permutation null, on >= 2/3 seeds -- while the "
                "CREDIT_RANDOM_TICKS control, which credits the same number "
                "of the same representations at ticks chosen without regard "
                "to attainment, does NOT (C2). The replay used to build the "
                "comparison reproduces the real substrate parent (C0), so the "
                "reading is about the crediting path and not about an "
                "instrument. The parent's structure is specific to WHICH "
                "subgoals were actually attained."
            )
        else:
            interpretation = (
                "MECH-428 WEAKENED: attainment occurred (G0), the forced-seed "
                "positive control showed the wiring can sustain a structured "
                "parent (G1), the substrate carries content-differentiated "
                "representations post-P0 against a permutation null (G3), the "
                "replay reproduces the real substrate parent (C0), and the "
                "random-tick negative control was silent (C2) -- and the "
                "parent's alignment with the attained group STILL did not "
                "exceed its own label-permutation null on >= 2/3 seeds. With "
                "the observation-timing and sense-path defects that sank 884b "
                "both corrected (pre-arrival credit, alpha_world=0.9) and the "
                "achievable ceiling measured in advance, this is genuine "
                "evidence that bottom-up bootstrapping at this "
                "alpha/decay/event-rate combination does not produce "
                "content-specific parent structure."
            )

    metrics: Dict[str, float] = {
        "g2_pass": 1.0 if g2_pass else 0.0,
        "g2_min_episode_budget_frac": g2_min_budget_frac,
        "n_cells_starved": float(len(g2_failures)),
        "g3_pass": 1.0 if g3_pass else 0.0,
        "g0_frac_seeds": g0_frac,
        "g1_frac_seeds": g1_frac,
        "c0_frac_seeds": c0_frac,
        "c1_frac_seeds": c1_frac,
        "c2_frac_seeds": c2_frac,
        "g0_pass": 1.0 if g0_pass else 0.0,
        "g1_pass": 1.0 if g1_pass else 0.0,
        "c0_pass": 1.0 if c0_pass else 0.0,
        "c1_pass": 1.0 if c1_pass else 0.0,
        "c2_pass": 1.0 if c2_pass else 0.0,
        "c0_fidelity_min": min(c0_fidelity_by_seed[s] for s in SEEDS),
        "c0_fidelity_threshold": C0_FIDELITY_FLOOR,
        "t_bootstrap_mean": sum(
            t_stat_by_seed[s]["bootstrap"]["t"] for s in SEEDS) / len(SEEDS),
        "t_bootstrap_null_p95_mean": sum(
            t_stat_by_seed[s]["bootstrap"]["null_p95"] for s in SEEDS) / len(SEEDS),
        "t_random_ticks_mean": sum(
            t_stat_by_seed[s]["random_ticks"]["t"] for s in SEEDS) / len(SEEDS),
        "t_random_ticks_null_p95_mean": sum(
            t_stat_by_seed[s]["random_ticks"]["null_p95"] for s in SEEDS) / len(SEEDS),
        "t_abs_floor": T_ABS_FLOOR,
        "n_permutations": float(N_PERMUTATIONS),
        "g3_dissim_min": min(
            rep_["g3_zworld_content_dissim_measured"] for rep_ in p0_g3_by_seed.values()),
        "g3_null_p95_max": max(rep_["g3_null_p95"] for rep_ in p0_g3_by_seed.values()),
        "alpha_world": ALPHA_WORLD,
    }
    for arm in ARMS:
        pn = [per_seed[arm][s]["parent_goal_norm_steady_state"] for s in SEEDS]
        nc = [per_seed[arm][s]["n_subgoal_credits"] for s in SEEDS]
        sa = [per_seed[arm][s]["n_steps_actual"] for s in SEEDS]
        metrics[f"parent_goal_norm_steady_state_mean_{arm}"] = sum(pn) / len(pn)
        metrics[f"n_subgoal_credits_mean_{arm}"] = sum(nc) / len(nc)
        metrics[f"n_steps_actual_mean_{arm}"] = sum(sa) / len(sa)

    evidence_direction_per_claim = {"MECH-428": evidence_direction}

    _g3_worst_seed = min(
        p0_g3_by_seed,
        key=lambda sd: (p0_g3_by_seed[sd]["g3_zworld_content_dissim_measured"]
                        - p0_g3_by_seed[sd]["g3_null_p95"]),
    )
    _c0_worst_seed = min(SEEDS, key=lambda sd: c0_fidelity_by_seed[sd])
    criteria = [
        {
            # WORST CELL, not the mean: g3_pass is an all() over seeds, so the
            # indexer must recompute `met` against the seed that came closest
            # to failing.
            "name": "G3_zworld_content_dissim_exceeds_permutation_null",
            "kind": "readiness",
            "load_bearing": True,
            "control": (
                "post-P0 probe walk, waypoint-tick group vs transit-tick "
                "group, SAME statistic C1 routes on; bar is the 95th "
                "percentile of a %d-draw label-permutation null over the same "
                "representations (red-team F3: 884b's absolute 0.0002 floor "
                "was cleared by 96 percent of random partitions)" % N_PERMUTATIONS
            ),
            "measured": p0_g3_by_seed[_g3_worst_seed]["g3_zworld_content_dissim_measured"],
            "threshold": p0_g3_by_seed[_g3_worst_seed]["g3_null_p95"],
            "direction": "lower",
            "comparator": ">",
            "offending_cell": f"seed{_g3_worst_seed}",
            "combination_rule": "every seed",
            "passed": g3_pass,
        },
        {
            "name": "C0_replay_reproduces_real_substrate_parent",
            "kind": "readiness",
            "load_bearing": True,
            "control": (
                "cosine between the decay+credit replay over this driver's "
                "recorded attained representations and the REAL substrate "
                "parent agent.goal_state.z_goal_parent; a mismatch means the "
                "DV is computed over a different event set than the substrate "
                "credited (red-team F2)"
            ),
            "measured": c0_fidelity_by_seed[_c0_worst_seed],
            "threshold": C0_FIDELITY_FLOOR,
            "direction": "lower",
            "offending_cell": f"seed{_c0_worst_seed}",
            "measured_per_seed": [c0_fidelity_by_seed[s] for s in SEEDS],
            "combination_rule": "every seed",
            "passed": c0_pass,
        },
        {
            "name": "C1_parent_aligns_with_attained_group_beyond_null",
            "load_bearing": True,
            "measured_per_seed": [t_stat_by_seed[s]["bootstrap"]["t"] for s in SEEDS],
            "threshold_per_seed": [
                max(t_stat_by_seed[s]["bootstrap"]["null_p95"], T_ABS_FLOOR) for s in SEEDS
            ],
            "measured": min(
                t_stat_by_seed[s]["bootstrap"]["t"]
                - max(t_stat_by_seed[s]["bootstrap"]["null_p95"], T_ABS_FLOOR)
                for s in SEEDS
            ),
            "threshold": 0.0,
            "direction": "lower",
            "comparator": ">",
            "combination_rule": (
                ">= 2/3 seeds, each seed passing iff T > max(its own "
                "permutation-null p95, T_ABS_FLOOR)"
            ),
            "passed": c1_pass,
        },
        {
            "name": "C2_random_tick_control_does_not_fire",
            "load_bearing": True,
            "control": (
                "CREDIT_RANDOM_TICKS arm -- same substrate call, same "
                "pre-arrival representations, same credit count, ticks chosen "
                "without regard to attainment. PASSES when the control is "
                "SILENT; note the inversion."
            ),
            "measured_per_seed": [t_stat_by_seed[s]["random_ticks"]["t"] for s in SEEDS],
            "threshold_per_seed": [
                max(t_stat_by_seed[s]["random_ticks"]["null_p95"], T_ABS_FLOOR) for s in SEEDS
            ],
            "measured": max(
                t_stat_by_seed[s]["random_ticks"]["t"] for s in SEEDS
            ),
            "threshold": max(
                max(t_stat_by_seed[s]["random_ticks"]["null_p95"], T_ABS_FLOOR) for s in SEEDS
            ),
            "direction": "upper",
            "combination_rule": ">= 2/3 seeds SILENT",
            "passed": c2_pass,
        },
    ]
    combination_rule = (
        "PASS = G0 (every seed) AND G1 (>=2/3) AND G2 (every cell) AND G3 "
        "(every seed) AND C0 (every seed) AND C2 (>=2/3 silent) AND C1 "
        "(>=2/3). C0/C2 failures route non_contributory, not weakens."
    )

    # This block is the direct repair of 884b's own recorded gap: it recorded
    # the BOUND [-1,1] rather than an ACHIEVABLE estimate, which is how a
    # 70-250x-unreachable floor got past it. The number below was MEASURED, at
    # this exact config, BEFORE any threshold here was fixed -- see the module
    # docstring, measurement 3.
    dv_headroom = {
        "dv_name": "T_parent_attained_vs_non_attained_cosine_delta",
        "criterion": "C1_parent_aligns_with_attained_group_beyond_null",
        "criterion_threshold": "per-seed max(permutation-null p95, T_ABS_FLOOR)",
        "achievable": 0.00343,
        "achievable_basis": (
            "MEASURED, not bounded: experiments/_scratch/"
            "_probe_884c_parent_level.py and the part-3 statistic selection, "
            "at this config (pre-arrival credit, alpha_world=0.9, real P0 "
            "scale), seeds 42/43/44 -> T = 0.003431 / 0.000662 / 0.001100 "
            "against null p95 0.001008 / 0.000196 / 0.000333 (ratios 3.4 / "
            "3.4 / 3.3). 'achievable' is the largest of the three. The "
            "theoretical bound is [-2, 2] and is NOT the right number to "
            "record here -- recording it is exactly what 884b did wrong."
        ),
        "theoretical_bound": [-2.0, 2.0],
        "abs_floor_headroom_note": (
            "T_ABS_FLOOR=%g sits 3.3x below the SMALLEST measured T "
            "(0.000662), deliberately non-binding; the discriminating bar is "
            "the per-seed permutation null." % T_ABS_FLOOR
        ),
        "withdrawn_884b_floor": 0.05,
        "withdrawn_884b_floor_reason": (
            "red-team F1: 70-250x the achievable ceiling, now confirmed by "
            "direct measurement at the parent level (max 0.00343)"
        ),
        "rejected_statistic": {
            "name": "S = 1 - cos(parent_true, parent_control)",
            "verdict": "AT-CHANCE on 3/3 seeds (percentile 78.0 / 63.0 / 86.5)",
            "reason": (
                "a resampled control PARENT inherits the EMA's effective "
                "sample size (~1/parent_goal_alpha = 20), so the null's spread "
                "exceeds the achievable signal. Recorded so no successor "
                "re-derives it."
            ),
        },
        "statistic": "cos(parent, mean_attained) - cos(parent, mean_non_attained)",
        "t_per_seed_bootstrap": [t_stat_by_seed[s]["bootstrap"]["t"] for s in SEEDS],
        "t_per_seed_random_ticks": [t_stat_by_seed[s]["random_ticks"]["t"] for s in SEEDS],
        "gates": False,
        "gate_rationale": (
            "A collapsed/untrained substrate is already caught by G3 (now "
            "null-referenced); an instrument mismatch is caught by C0; a "
            "non-specific alignment is caught by C2. A fourth gate on the "
            "same conditions would double-count them."
        ),
    }

    diagnostics = {
        "dv_headroom": dv_headroom,
        "t_statistic_per_seed": t_stat_by_seed,
        "c0_fidelity_per_seed": c0_fidelity_by_seed,
        "combination_rule": combination_rule,
        "credit_count_parity_per_seed": {
            str(s): {
                "bootstrap_credits": per_seed["SUBGOAL_BOOTSTRAP"][s]["n_subgoal_credits"],
                "random_ticks_credits": per_seed["CREDIT_RANDOM_TICKS"][s]["n_subgoal_credits"],
                "random_ticks_applied": per_seed["CREDIT_RANDOM_TICKS"][s]["n_random_credits_applied"],
                "random_ticks_target": per_seed["CREDIT_RANDOM_TICKS"][s]["n_target_credits"],
            }
            for s in SEEDS
        },
        # Option 3 (T as IV) -- free, non-gating diagnostic per the
        # recommendation's own disposition for it.
        "inter_event_interval_ticks_per_seed": inter_event_interval_by_seed,
        "p0_g3_report_per_seed": {
            str(seed): {
                k: v for k, v in rep.items() if k != "p0a_report"
            }
            for seed, rep in p0_g3_by_seed.items()
        },
        "p0a_report_per_seed": {
            str(seed): rep["p0a_report"] for seed, rep in p0_g3_by_seed.items()
        },
    }

    summary_markdown = (
        f"# {QUEUE_ID} -- MECH-428 Subgoal-Bootstrapped Goal Seeding (content-DV redesign)\n\n"
        f"**Status:** {status}  **Evidence direction:** {evidence_direction}\n"
        f"**Route reason:** {route_reason}\n"
        f"**Claims:** MECH-428\n\n"
        f"## Gates and criteria\n\n"
        f"| Gate/Criterion | Value | Pass |\n|---|---|---|\n"
        f"| G2 readiness (episode budget, every cell) | {g2_min_budget_frac:.2f} (min) | {g2_pass} |\n"
        f"| G0 readiness (attainment, every seed) | {g0_frac:.2f} | {g0_pass} |\n"
        f"| G1 readiness (forced-seed structured, >=2/3) | {g1_frac:.2f} | {g1_pass} |\n"
        f"| G3 readiness (content-discriminability vs permutation null, every seed) | "
        f"{min(rep_['g3_zworld_content_dissim_measured'] for rep_ in p0_g3_by_seed.values()):.6f} "
        f"(worst null p95 "
        f"{max(rep_['g3_null_p95'] for rep_ in p0_g3_by_seed.values()):.6f}) | {g3_pass} |\n"
        f"| C0 replay fidelity vs real substrate parent (every seed) | "
        f"{min(c0_fidelity_by_seed[s] for s in SEEDS):.9f} "
        f"(floor {C0_FIDELITY_FLOOR}) | {c0_pass} |\n"
        f"| C2 random-tick negative control SILENT (>=2/3) | {c2_frac:.2f} | {c2_pass} |\n"
        f"| C1 parent aligns with attained group beyond null (>=2/3) | {c1_frac:.2f} | {c1_pass} |\n\n"
        f"**Combination rule:** {combination_rule}\n\n"
        f"## Interpretation\n\n{interpretation}\n"
    )

    result: Dict[str, Any] = {
        "status": status,
        "outcome": status,
        "route_reason": route_reason,
        "metrics": metrics,
        "readout": dict(metrics),
        "criteria": criteria,
        "arm_results": [
            {k: v for k, v in r.items() if not k.startswith("_content_")} for r in arm_results
        ],
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "per_seed_results": {
            arm: {
                s: {k: v for k, v in row.items() if not k.startswith("_content_")}
                for s, row in seeds.items()
            }
            for arm, seeds in per_seed.items()
        },
        "dv_headroom": dv_headroom,
        "combination_rule": combination_rule,
        "diagnostics": diagnostics,
        "episode_budget_audit": {
            "g2_pass": g2_pass,
            "min_episode_budget_frac": g2_min_budget_frac,
            "starved_cells": g2_failures,
            "per_cell": [
                {
                    "arm": r["arm"], "seed": r["seed"],
                    "n_steps_actual": r["n_steps_actual"],
                    "n_steps_configured": r["n_steps_configured"],
                    "done": r["episode_done"],
                    "done_cause": r["episode_done_cause"],
                    "agent_health_final": r["agent_health_final"],
                }
                for r in arm_results
            ],
        },
        "supersedes": SUPERSEDES,
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
        "arms": ARMS,
        "grid_size": GRID_SIZE,
        "num_waypoints": NUM_WAYPOINTS,
        "n_steps": N_STEPS,
        "measurement_window": MEASUREMENT_WINDOW,
        "world_dim": WORLD_DIM,
        "parent_goal_alpha": PARENT_GOAL_ALPHA,
        "parent_goal_decay": PARENT_GOAL_DECAY,
        "forced_credit": FORCED_CREDIT,
        "forced_structured_floor": FORCED_STRUCTURED_FLOOR,
        "parent_norm_abs_floor": PARENT_NORM_ABS_FLOOR,
        "zworld_content_dissim_floor_legacy_884b": ZWORLD_CONTENT_DISSIM_FLOOR,
        "alpha_world": ALPHA_WORLD,
        "n_permutations": N_PERMUTATIONS,
        "null_percentile": NULL_PERCENTILE,
        "t_abs_floor": T_ABS_FLOOR,
        "c0_fidelity_floor": C0_FIDELITY_FLOOR,
        "credited_representation": "pre_arrival_zworld_tick_minus_one",
        "p0_episodes": P0_EPISODES,
        "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
        "subgoal_arrival_position_check": True,
        "hazard_free_contamination_gate": True,
        "supersedes": SUPERSEDES,
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
