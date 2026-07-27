"""V3-EXQ-798a -- SD-MEL-PRODUCER validation, C4 ACTUALLY MEASURABLE: does a
non-converging world produce a GRADED, ABOVE-REFERENCE and genuinely LEARNABLE
waking prediction-error load?

SUPERSEDES V3-EXQ-798. Same scientific question, same instrument, same arms ladder.
The ONLY changes are the ones needed to make C4 -- the anti-artifact learnability
criterion, and the sole reason a noise control arm exists -- readable at all, plus
the routing guard that stops an unreadable C4 masquerading as a measured verdict.

PURPOSE. This validates the ENVIRONMENT TEST-BED landed as SD-MEL-PRODUCER
(2026-07-21), not a claim. claim_ids is deliberately EMPTY.

================================================================================
WHY 798a EXISTS -- three defects in V3-EXQ-798, all in or around C4
================================================================================
V3-EXQ-798 landed FAIL / non_contributory, label producer_graded_but_not_learnable
(manifest v3_exq_798_sdmelproducer_graded_nonconverging_world_20260723T081627Z_v3).
Readiness PASSED, C1 2/3, C2 2/3, C3 3/3, C4 fail. Adjudicated 2026-07-27.

DEFECT 1 -- C4 WAS UNREADABLE BY CONSTRUCTION, not by bad luck.
  `_decay_frac` returns None unless bin[0] and bin[N-1] each hold >= 5 samples.
  798 used SSL_BIN_EDGES=(2,5,12), so bin 3 required
  steps_since_world_rule_shift > 12. C4 read only ARM_3_HIGH and ARM_4_NOISE, and
  BOTH ran at world_rule_shift_interval=10. The counter cycles 0..interval-1
  (causal_grid_world._maybe_shift_world_rule resets it to 0 on each fire), so at
  interval 10 it never exceeds 9. Landed bin counts are {0:270, 1:270, 2:360, 3:0}
  on all three seeds of BOTH arms -- deterministic, not sparse sampling. The arms
  C4 did NOT read confirm the mechanism: interval 60 -> bin 3 holds 705
  (decay 0.260 / 0.103 / 0.361), interval 25 -> 432 (0.033 / 0.092 / 0.203).

DEFECT 2 -- THE NOISE CONTROL WAS NOT ACTUALLY ELEVATED.
  The `noise_control_at_least_as_elevated` precondition also failed:
  ARM_4_NOISE mel 2.36e-05 < ARM_3_HIGH mel 3.58e-05 at NOISE_SIGMA=0.05. Even a
  populated bin 3 would have given a comparison biased toward the graded arm.

DEFECT 3 -- THE ROUTING IGNORED READABILITY.
  `evaluate()` selected the producer_graded_but_not_learnable branch on
  `c1_pass and c2_pass and not c4_pass` WITHOUT consulting `c4_readable`, so the
  landed manifest asserts "ARTIFACT WARNING ... the elevation does NOT decay under
  training and is not separable from the matched noise control" about a quantity
  that was never measured. Its own `learnability_bins_populated` precondition
  correctly records met:false, so the manifest contradicts itself. Do NOT read
  798's evidence_direction_note as evidence about learnability.

  Latent FOURTH gap in the same grid, found while repairing it: 798's branches did
  not cover C1 pass + C2 pass + C3 fail + C4 pass -- that combination fell through
  to `producer_ineffective`, which flatly contradicts C1/C2 passing. 798a's grid is
  exhaustive over all sixteen (C1,C2,C3,C4) combinations.

NOT A DEFECT, DELIBERATELY UNCHANGED: the z_goal stream. 798 has a confirmed dead
z_goal stream (active_frac=0.000, writer_calls=0) and it was adjudicated
NON_CONTRIBUTORY on verified grounds -- the E3 goal term is gated on
`goal_state.is_active()` (e3_selector.py:1180-1188); the E1 goal-conditioning path
is gated on `is_active()` too (agent.py:4844-4859), so E1 receives z_goal=None,
identical to the flag being off; `update_residue`, which produces the DV
`e3_prediction_error`, has no goal reference; and `_make_agent(env)` is
arm-independent, so the inert knobs are structurally arm-symmetric. Wiring the
stream live would ACTIVATE the E3 goal term and E1 conditioning, changing action
selection and hence the trajectories the PE is computed over -- that would make
798a non-comparable to 798 / 701c / 718a and would confound the C4 repair with a
behaviour change. The two defects are kept separate. `_make_agent` below is
therefore byte-identical to 798's, and the dead-z_goal lint is opted out of
explicitly rather than silenced by a config change.

================================================================================
BACKGROUND -- the producer gap (unchanged from 798)
================================================================================
MECH-180 / INV-050 split into:
  link (i)  real graded novelty -> graded above-reference waking MEL
  link (ii) MEL -> graded offline-phase duration
Link (ii) is BUILT + PROVEN (SD-MEL-CONSUMER; V3-EXQ-718a's injection positive
control: injected MEL [0.6..2.5] -> offline duration [9,13,18,24,30,38], all seeds).
Link (i) has NEVER been demonstrated, and three runs establish that it cannot be
demonstrated with the env_drift knob:
  V3-EXQ-677   C1 manipulation check: high- vs low-novelty mean E1 prediction error
               differed by 8.8e-07 against a 0.01 threshold. NO novelty gradient.
  V3-EXQ-718   C1 novelty-label monotonicity 0/3 seeds.
  V3-EXQ-718a  ecological MEL ~1e-5 -- noise-level, and SCRAMBLED vs novelty level;
               conv_rel_drop ~0.98.
Both autopsies classify this measurement_gap (environment / test-bed producer gap),
explicitly NOT a substrate ceiling and NOT a falsification.

ROOT CAUSE. env_drift_interval fires _drift_hazards(), which only MOVES hazards. The
optimal prediction of a random walk is its mean, so the world-forward model learns
that fast and PE floors at the irreducible noise level. Drift adds sampling NOISE,
not learning LOAD.

MECHANISM UNDER TEST. SD-MEL-PRODUCER periodically re-permutes the action ->
displacement map. E2.world_forward(z_world, a) takes the action as an INPUT, so a
permutation makes the learned forward model systematically wrong until re-learned.
Between shifts the world is deterministic and therefore LEARNABLE; each shift
invalidates learned structure. Load is graded by shift RATE.

WHY THE NOISE ARM IS THE POINT OF THIS EXPERIMENT. Grading observation noise would
ALSO produce a monotone MEL ladder -- but by construction, on any substrate, whether
or not MECH-180 is true. That is the DV-symmetry artifact class
(failure_autopsy_V3-EXQ-604c; the defect that held V3-EXQ-683 on 2026-07-21).
Elevated PE is only learning LOAD if it is REDUCIBLE. So this experiment does not
merely measure whether MEL is graded -- it measures whether the gradient is made of
re-learnable structure, by comparing against an arm whose PE elevation is
definitionally un-learnable.

================================================================================
DESIGN CHANGE 1 -- CYCLE-QUARTILE BINNING (fixes defect 1)
================================================================================
798 binned PE by ABSOLUTE steps_since_world_rule_shift against fixed edges. That
makes the top bin's reachability an accident of which intervals happen to be in the
ladder, and it silently produced an empty top bin on exactly the two arms C4 read.

798a bins by POSITION WITHIN THE SHIFT CYCLE: quartile q = min(3, 4*counter //
interval). Reachability then follows from the design rather than from a coincidence:
every quartile of a cycle of length I is non-empty for any I >= 4, at EVERY interval
in the ladder, and `_assert_quartile_reachability()` proves it in-script from the
pre-registered ARMS table before any compute is spent (Step 3.5: a gate expressed
over quantities the design pre-registers is checkable on paper at design time).
Measured occupancy at 900 steps: interval 10 -> 270/180/270/180, interval 25 ->
~252/216/216/216, interval 60 -> 225 each.

Quartile binning also makes the bins mean the same PHASE at every shift rate, so the
decay-vs-rate profile is directly comparable across arms -- which the absolute
scheme could not deliver (at interval 60 the old bin 3 spanned steps 13-59; at
interval 25 it spanned 13-24; at interval 10 it was empty).

The absolute-edge binning is STILL RECORDED, at 798's own edges (2,5,12), as
`*_pe_by_ssl_bin` / `*_ssl_bin_counts`. It is a ROBUSTNESS DIAGNOSTIC ONLY and
enters NO criterion: pre-registered, C4 and C3 route on the QUARTILE binning and on
nothing else. Recording both is what lets a reader check that the repaired C4 is not
itself an artifact of the new binning, and it keeps 798's numbers comparable.

================================================================================
DESIGN CHANGE 2 -- C4 ANCHORED WHERE RE-LEARNING IS PHYSICALLY POSSIBLE
================================================================================
Making bin 3 reachable at interval 10 is necessary but not sufficient. At interval
10 the model gets at most 9 steps between shifts, so a null decay there ALIASES two
very different causes -- "the structure is not re-learnable" and "there was no time
to re-learn it". A criterion that cannot separate those cannot discharge the
anti-artifact question (verdict aliasing, GOV-FANOUT-1 step 4).

So C4 is ANCHORED on the SLOWEST graded shift arm, ARM_1_LOW at interval 60, which
has the most re-learning headroom: if the elevation is reducible at all, it is
reducible there. A null at the anchor is an unaliased negative. 798's own unread
arms are the reachability demonstration for this anchor: at interval 60 the top
absolute bin held 705 samples on every seed.

The anchor choice is PRE-REGISTERED as C4_ANCHOR_ARM below, fixed before the run,
and is NOT selected from this run's outcomes.

The decay of EVERY graded arm is reported as `learn_decay_by_arm` /
`learn_decay_by_interval`. That profile is the corroborating signature -- genuine
learning load should decay MORE as the inter-shift window lengthens, whereas an
un-learnable elevation is flat at every rate. It is a recorded diagnostic, not a
criterion; no threshold is fitted to it.

================================================================================
DESIGN CHANGE 3 -- NOISE CONTROL MATCHED IN INTERVAL AND CALIBRATED IN SIGMA
                   (fixes defect 2)
================================================================================
INTERVAL. The control must share the anchor's bin structure or the decay comparison
is not like-for-like, so the noise arms run at the ANCHOR's interval (60), not at
10. Depth 0 keeps the schedule firing while the rule never changes, so the counter
(and therefore the quartile binning) stays defined while there is by construction
nothing to re-learn.

SIGMA. 798's NOISE_SIGMA=0.05 produced a MEL elevation of only ~0.47e-05 over the
stable base, against ARM_3_HIGH's ~1.69e-05, so the control was LESS elevated than
the arm it controls for. Calibration is derived from 798's landed measurement under
an explicit model, stated here BEFORE the run rather than fitted after it:
observation noise enters an L2 residual quadratically, so the PE contribution scales
as sigma^2. Reaching the worst-seed ARM_1_LOW elevation (seed 456: 1.82e-05 against
the control's 0.65e-05 at sigma 0.05) needs a factor of ~2.8, i.e. sigma ~0.084;
NOISE_SIGMA_LO = 0.12 gives ~5.8x, which clears the anchor's worst seed with ~2x
headroom and also clears ARM_3_HIGH's worst seed (~1.2x).

TWO noise arms are run, not one, at NOISE_SIGMA_LO = 0.12 and NOISE_SIGMA_HI = 0.24
(4x the PE contribution). This is NOT post-hoc tuning: the selection rule is
pre-registered -- the C4 control is the LOWEST-sigma noise arm whose MEL is at least
the anchor's on EVERY seed, i.e. the most STRINGENT fair control available. Two
points remove the knife-edge left by a single-shot sigma extrapolated across a
substrate change, and they record a sigma -> MEL calibration curve that a successor
can read instead of re-deriving. Over-shooting is conservative for C4 (decay_frac is
a relative drop, so a larger noise level cannot manufacture decay in the control),
which is why the failure direction chosen here is "too elevated", never "not
elevated enough". The ACHIEVED ratios against both ARM_1_LOW and ARM_3_HIGH are
REPORTED (`noise_elevation_ratios`), not tuned.

================================================================================
DESIGN CHANGE 4 -- ROUTING GUARD (fixes defect 3)
================================================================================
`c4_measurable = c4_bins_readable AND c4_control_fair`. When it is false, NO
C4-dependent branch may be taken; the run routes to the explicit terminal label
`c4_unreadable_requeue` and says which of C1/C2/C3 nonetheless stand. An unreadable
criterion is never reported as a measured verdict.

The fairness leg gates the verdict as well as the readability leg. A C4 PASS won
against an under-elevated control is untrustworthy; a C4 FAIL against one is
arguably still informative, but conflating "the control was unfair" with "the load
is an artifact" is precisely the class of error 798 committed, so the conservative
route is taken in both directions and the C1/C2/C3 findings are stated explicitly in
the note so nothing is buried.

================================================================================
WHY C4 NEEDS A TRAINING PHASE (unchanged from 798)
================================================================================
A FROZEN model cannot re-learn, so PE after a shift would never decay regardless of
whether the structure is learnable. The MEL measurement (P1) stays frozen for
comparability with 701c / 718a; the learnability probe (P2) enables online
world-forward training so decay is meaningful.

MEASUREMENT NOTE (unchanged). Shift rate SHORTENS episodes (798 measured 33.5 ->
18.4 steps between ARM_0_NONE and ARM_3_HIGH), which would confound a fixed-episode
window by giving each arm a different episode-phase mix. So both measurement phases
use a fixed STEP budget and mean_episode_length is reported per arm.

ARMS (6 cells x SEEDS):
  ARM_0_NONE      interval 0   depth 2  -- no shifts (stable-base reference)
  ARM_1_LOW       interval 60  depth 2  -- C4 ANCHOR
  ARM_2_MED       interval 25  depth 2
  ARM_3_HIGH      interval 10  depth 2
  ARM_4_NOISE_LO  interval 60  depth 0  sigma 0.12  (NEGATIVE CONTROL, matched)
  ARM_5_NOISE_HI  interval 60  depth 0  sigma 0.24  (NEGATIVE CONTROL, matched)

DV-SYMMETRY DECLARATION (one line per arm, per the Step 3.5 requirement):
  ARM_0_NONE / ARM_1_LOW / ARM_2_MED / ARM_3_HIGH -- DV is mean per-step e3
    prediction_error, an L2 residual between E2's ACTION-CONDITIONED prediction and
    the observed next z_world. The manipulation re-permutes the action->displacement
    map, which changes that conditional mapping. The DV is NOT invariant under it:
    this is a conditional-prediction argument, not a distributional one -- even a
    uniform-random policy in a symmetric world leaves the trajectory DISTRIBUTION
    unchanged while breaking the learned per-action mapping the residual is computed
    against.
  ARM_4_NOISE_LO / ARM_5_NOISE_HI -- DV is invariant in the trivial sense: additive
    observation noise raises an L2 residual BY ARITHMETIC. That is deliberate and is
    precisely their role. They are NEGATIVE CONTROLS calibrating what an artifact
    looks like, and are NOT evidence for anything. They are excluded from the C1
    grading ladder.
  Learnability DV (PE binned by cycle quartile under a TRAINING model) -- NOT an
    arithmetic identity for either family: noise yields flat quartiles, re-learnable
    structure yields decaying ones. This is the discriminating DV.

CRITERIA (pre-registered):
  C1 (LOAD-BEARING) grading: MEL monotone NONE < LOW < MED < HIGH and relative
     spread (HIGH/NONE - 1) >= MIN_REL_MEL_SPREAD, on >= SEED_PASS_FRAC of seeds.
  C2 above-reference: every graded arm's MEL exceeds the NONE reference by at least
     MIN_ABS_MEL_SPREAD. This is the exact thing 718a failed (ecological HIGH DV
     72/74 BELOW the OFF baseline 90). The floor is 1e-6, matching the consumer's
     recalibrated mel_relative_floor -- NOT the 701c ABS_MEL_FLOOR=1e-4, which is
     ~5x the entire converged-base signal and structurally unreachable.
  C3 sustained non-convergence: the HIGH arm's PE stays ABOVE the stable-base
     reference (ARM_0_NONE's mean MEL) even in the LAST QUARTILE of its own shift
     cycle -- i.e. the world does not settle back before the next shift. Read
     against NONE's OVERALL mean, not NONE's late quartile: NONE has no shifts, so
     all of its PE lands in quartile 0 and that comparison would be vacuous. NOTE A
     SEMANTIC SHIFT FROM 798: under quartile binning "long after" means late in the
     arm's OWN cycle rather than an absolute step count, which is the meaningful
     reading for a rate-graded ladder. In 798 this criterion was never readable at
     all and silently fell back to an overall-mean comparison on 3/3 seeds
     (criteria_non_degenerate.C3 = false).
  C4 (LOAD-BEARING) learnability (ANTI-ARTIFACT): under training, the ANCHOR arm
     shows post-shift PE DECAY (quartile 0 -> quartile 3 relative drop >=
     MIN_LEARN_DECAY) while the matched, at-least-as-elevated noise control does not
     (<= MAX_NOISE_DECAY). Without this, a graded ladder is indistinguishable from
     graded noise.

INTERPRETATION GRID (exhaustive over the sixteen C1..C4 combinations):
  readiness unmet                     -> substrate_not_ready_requeue
  C4 not measurable                   -> c4_unreadable_requeue
  C1 fail, C2 fail                    -> producer_ineffective
  C1 fail, C2 pass                    -> producer_elevates_but_not_graded
  C1+C2 pass, C4 fail                 -> producer_graded_but_not_learnable (ARTIFACT)
  C1+C2+C4 pass, C3 fail              -> producer_graded_learnable_but_settles
  C1+C2+C3+C4 pass                    -> producer_validated_graded_learnable  (PASS)

PROMOTES / DEMOTES NOTHING. claim_ids=[]; experiment_purpose=diagnostic. A PASS
licenses the SEPARATE, still-gated ecological end-to-end MECH-180 run; it is not
itself evidence for MECH-180.

SLEEP DRIVER: not applicable (no sleep; the consumer is deliberately absent -- 718a's
learning_extracted[1] records that the consumer's DV is a deterministic function of
MEL, so involving it cannot validate a producer).
"""

import sys
import math
import time
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from _metrics import check_degeneracy
from experiments._lib.arm_fingerprint import arm_cell
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_798a_sdmelproducer_graded_nonconverging_world_c4readable"
QUEUE_ID = "V3-EXQ-798a"
SUPERSEDES = "V3-EXQ-798"
CLAIM_IDS: List[str] = []          # validates a TEST-BED, tags no claim. See docstring.
EXPERIMENT_PURPOSE = "diagnostic"

# The dead z_goal stream is INHERITED FROM 798 ON PURPOSE and was adjudicated
# non_contributory on verified grounds (2026-07-27; see the docstring's "NOT A
# DEFECT" block and the pinned comment in
# tests/contracts/test_dead_z_goal_stream_lint.py). Wiring update_z_goal would
# activate the E3 goal term and E1 goal-conditioning, changing action selection and
# hence the trajectories the DV is computed over -- that is a behaviour change that
# would make this run non-comparable to 798 / 701c / 718a and would confound the C4
# repair. The knobs are inert IDENTICALLY in every arm (_make_agent takes only env
# dims), so the inertness is structurally arm-symmetric.
DEAD_Z_GOAL_STREAM_EXEMPT = (
    "inherited verbatim from V3-EXQ-798 for comparability; the dead stream was "
    "adjudicated non_contributory 2026-07-27 (E3 goal term and E1 conditioning both "
    "gated on goal_state.is_active(); update_residue, which produces the DV, has no "
    "goal reference; knobs are arm-symmetric). Wiring it live would confound the C4 "
    "repair with a behaviour change -- see the docstring."
)

# The R1/R2 readiness predicates and their thresholds are inherited VERBATIM from
# V3-EXQ-701c, and were then MEASURED clearing comfortably by V3-EXQ-798 on this exact
# instrument and this exact recon-only base: conv_seed_fraction 1.0 (3/3 seeds,
# per-seed drops 0.986 / 0.982 / 0.989) against MIN_REL_CONV_DROP=0.10, and
# pe_response 0.452 against MIN_REL_PE_RESPONSE=0.25. So reachability is established
# by demonstration on TWO landed runs rather than by a synthetic in-script assertion
# -- these are not hand-written predicates narrower than the state they anchor to
# (the V3-EXQ-778d defect this check exists to catch). The two C4 preconditions are
# NOT covered by this exemption and are proved separately: quartile-bin reachability
# by _assert_quartile_reachability() from the pre-registered ARMS table, and control
# fairness by the pre-registered two-point sigma ladder (see the docstring).
ANCHOR_REACHABILITY_EXEMPT = (
    "R1/R2 predicates + thresholds inherited verbatim from V3-EXQ-701c and measured "
    "clearing on V3-EXQ-798 (conv_frac 1.0 vs 0.10 floor; pe_response 0.452 vs 0.25 "
    "floor) on the same instrument and same recon-only base. The C4-side "
    "preconditions are proved in-script instead: see _assert_quartile_reachability()."
)

# -- Design parameters -------------------------------------------------------
SEEDS = [42, 123, 456]
CONV_EPISODES = 60          # P0 world-model convergence on the STABLE base env
STEPS_PER_EPISODE = 90
MEAS_STEPS = 900            # P1 FROZEN MEL window -- a fixed STEP budget (see note)
LEARN_STEPS = 900           # P2 TRAINING learnability window -- also step-budgeted
PROBE_STEPS = 100
PROBE_BATTERY_SIZE = 64
# Progress denominator M. P1/P2 are step-budgeted, so their episode-equivalents are
# what the runner's ETA can meaningfully count.
MEAS_EPISODE_EQUIV = MEAS_STEPS // STEPS_PER_EPISODE
LEARN_EPISODE_EQUIV = LEARN_STEPS // STEPS_PER_EPISODE
EPISODES_PER_RUN = CONV_EPISODES + MEAS_EPISODE_EQUIV + LEARN_EPISODE_EQUIV

# E2 world-forward online training. PRIMARY = reconstruction MSE. RECON-ONLY: the
# SD-056 contrastive auxiliary is a CONFIRMED P0 destabiliser (V3-EXQ-701b ablation),
# so the loss term is omitted. SD056_WEIGHT is retained only for module parity with
# the proven-converging recon-only arm; it is never added to the loss here.
SD056_WEIGHT = 0.05
E2_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
TRANSITION_BUFFER_MAX = 256

# -- Thresholds (pre-registered constants, NOT derived from run stats) -------
MIN_REL_PE_RESPONSE = 0.25   # R1: pe_shock at least 25% above pe_stable
MIN_REL_CONV_DROP = 0.10     # R2: per-seed frozen-probe PE drops at least 10%
SEED_PASS_FRAC = 2.0 / 3.0
MIN_REL_MEL_SPREAD = 0.25    # C1: mel[HIGH] at least 25% above mel[NONE]
MONO_TOL = 0.02              # C1: monotonicity slack (relative to mel[NONE])
# C2 floor: 1e-6, matching SD-MEL-CONSUMER's recalibrated mel_relative_floor.
# NOT the 701c ABS_MEL_FLOOR=1e-4, which is ~5x the whole converged-base signal.
MIN_ABS_MEL_SPREAD = 1e-6
MIN_LEARN_DECAY = 0.15       # C4: anchor PE must drop >=15% from quartile 0 -> 3
MAX_NOISE_DECAY = 0.10       # C4: the matched noise control must show < 10% decay
MIN_BIN_SAMPLES = 5          # a quartile is readable only with this many samples

# Observation noise for the NEGATIVE CONTROLS. PRE-REGISTERED, derived from 798's
# landed measurement under a stated sigma^2 model (see the docstring's DESIGN
# CHANGE 3). NOT fitted to this run: the achieved elevation ratios are REPORTED.
NOISE_SIGMA_LO = 0.12
NOISE_SIGMA_HI = 0.24

# -- Binning -----------------------------------------------------------------
# PRIMARY (drives C3 and C4): position within the shift CYCLE, as a quartile.
# q = min(N_BINS-1, N_BINS * counter // interval). Every quartile of a cycle of
# length I is non-empty for any I >= 4, so the top bin's reachability is a property
# of the DESIGN, not an accident of which intervals are in the ladder -- which is
# exactly what failed in 798. Proved from the ARMS table by
# _assert_quartile_reachability() before any compute is spent.
N_BINS = 4
MIN_INTERVAL_FOR_QUARTILES = 8   # margin: >= 2 steps of the cycle per quartile

# SECONDARY (recorded, enters NO criterion): 798's absolute edges, kept so the new
# result can be checked against the old numbers and so the repaired C4 can be shown
# not to be an artifact of the new binning.
SSL_BIN_EDGES = (2, 5, 12)   # -> bins [0-2], [3-5], [6-12], [13+]
N_SSL_BINS = 4

# -- Environment base (functional config; identical to 701c / 798) -----------
ENV_BASE: Dict[str, Any] = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    use_proxy_fields=True,
)

# The stable base carries NO hazard drift, so the only non-stationarity in the graded
# arms is the rule shift itself.
STABLE_DRIFT = dict(env_drift_interval=999, env_drift_prob=0.0)
SHOCK_DRIFT = dict(env_drift_interval=1, env_drift_prob=0.8)

ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_NONE",  "level": 0, "interval": 0,  "depth": 2, "noise": 0.0,
     "graded": True},
    {"arm_id": "ARM_1_LOW",   "level": 1, "interval": 60, "depth": 2, "noise": 0.0,
     "graded": True},
    {"arm_id": "ARM_2_MED",   "level": 2, "interval": 25, "depth": 2, "noise": 0.0,
     "graded": True},
    {"arm_id": "ARM_3_HIGH",  "level": 3, "interval": 10, "depth": 2, "noise": 0.0,
     "graded": True},
    # NEGATIVE CONTROLS: the ANCHOR's interval (60) so the quartile bins are
    # structurally identical to the arm C4 compares them against, depth=0 so the rule
    # never changes (nothing to re-learn), plus observation noise to elevate PE.
    {"arm_id": "ARM_4_NOISE_LO", "level": 4, "interval": 60, "depth": 0,
     "noise": NOISE_SIGMA_LO, "graded": False},
    {"arm_id": "ARM_5_NOISE_HI", "level": 5, "interval": 60, "depth": 0,
     "noise": NOISE_SIGMA_HI, "graded": False},
]
GRADED_ORDER = ["ARM_0_NONE", "ARM_1_LOW", "ARM_2_MED", "ARM_3_HIGH"]

# PRE-REGISTERED, fixed before the run and NOT selected from its outcomes: C4 is
# anchored on the slowest graded shift arm, which has the most re-learning headroom,
# so a null decay there is an UNALIASED negative rather than "no time to re-learn".
C4_ANCHOR_ARM = "ARM_1_LOW"
# PRE-REGISTERED selection rule: the C4 control is the LOWEST-sigma noise arm whose
# MEL is at least the anchor's on EVERY seed -- the most STRINGENT fair control
# available. Ordered by ascending sigma.
NOISE_CONTROL_ORDER = ["ARM_4_NOISE_LO", "ARM_5_NOISE_HI"]


def _assert_quartile_reachability() -> Dict[str, Any]:
    """Design-time proof that every quartile is reachable on every arm C3/C4 read.

    This is the check 798 lacked. `_decay_frac` needs samples in BOTH the first and
    the last bin, and in 798 the last bin was empty BY CONSTRUCTION on both arms C4
    read. Rather than assume the repair works, prove it from the pre-registered ARMS
    table before any compute is spent: the counter cycles 0..interval-1, so with
    N_BINS quartiles every bin is non-empty exactly when interval >= N_BINS.
    Raises AssertionError instead of running a doomed 9-hour grid.
    """
    shifting = [a for a in ARMS if a["interval"] > 0]
    assert shifting, "no shift-enabled arm; the quartile binning would be vacuous"
    worst = min(a["interval"] for a in shifting)
    assert worst >= MIN_INTERVAL_FOR_QUARTILES, (
        f"quartile binning unreachable: arm interval {worst} < "
        f"{MIN_INTERVAL_FOR_QUARTILES}; every quartile of a cycle of length I is "
        f"non-empty only for I >= {N_BINS}")
    # The C4 pair must additionally SHARE a bin structure, or the decay comparison is
    # not like-for-like -- the second half of the 798 defect.
    by_id = {a["arm_id"]: a for a in ARMS}
    anchor = by_id[C4_ANCHOR_ARM]
    for ctrl_id in NOISE_CONTROL_ORDER:
        ctrl = by_id[ctrl_id]
        assert ctrl["interval"] == anchor["interval"], (
            f"noise control {ctrl_id} interval {ctrl['interval']} != anchor "
            f"{C4_ANCHOR_ARM} interval {anchor['interval']}; the quartile bins would "
            f"not be comparable")
        assert ctrl["depth"] == 0, f"{ctrl_id} must have depth 0 (nothing to re-learn)"
        assert ctrl["noise"] > 0.0, f"{ctrl_id} must carry observation noise"
    return {
        "min_shift_interval": int(worst),
        "n_bins": int(N_BINS),
        "min_interval_required": int(MIN_INTERVAL_FOR_QUARTILES),
        "anchor_interval": int(anchor["interval"]),
        "expected_min_quartile_steps_per_cycle": int(worst // N_BINS),
    }


# Fail loudly at import if the ladder and the binning ever drift apart again.
QUARTILE_REACHABILITY = _assert_quartile_reachability()


def _make_env(seed: int, interval: int, depth: int) -> CausalGridWorldV2:
    kw = dict(ENV_BASE)
    kw.update(STABLE_DRIFT)
    kw.update(
        world_rule_shift_enabled=(interval > 0),
        world_rule_shift_interval=interval,
        world_rule_shift_depth=depth,
    )
    return CausalGridWorldV2(seed=seed, **kw)


def _make_probe_env(seed: int, **drift) -> CausalGridWorldV2:
    kw = dict(ENV_BASE)
    kw.update(drift)
    return CausalGridWorldV2(seed=seed, **kw)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Functional waking agent with a trainable world-forward model. NO sleep.

    Config is byte-identical to V3-EXQ-701c and V3-EXQ-798 so the world_forward
    MODULE matches the proven-converging recon-only ablation arm exactly, and so 798a
    stays comparable to the run it supersedes. The encoder is frozen (only agent.e2
    trains), which is what makes the frozen-probe battery valid: the captured z0/z1
    live in a latent space that is CONSTANT across P0 checkpoints.

    The z_goal knobs are inert here and are DELIBERATELY LEFT AS-IS -- see
    DEAD_Z_GOAL_STREAM_EXEMPT above and the docstring's "NOT A DEFECT" block. They
    are identical in every arm, so the inertness is arm-symmetric.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
    )
    return REEAgent(cfg)


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    v = d.get(key)
    if v is None:
        return None
    return v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32)


def _apply_obs_noise(obs_dict: Dict[str, Any], sigma: float,
                     gen: Optional[torch.Generator]) -> Dict[str, Any]:
    """NEGATIVE-CONTROL perturbation: additive Gaussian noise on the exteroceptive
    channel. Deliberately UN-LEARNABLE -- there is no structure here to re-learn, so
    a model cannot reduce the PE it induces no matter how long it trains."""
    if sigma <= 0.0:
        return obs_dict
    out = dict(obs_dict)
    ws = _obs(obs_dict, "world_state")
    if ws is not None:
        noise = torch.randn(ws.shape, generator=gen, dtype=ws.dtype) * sigma
        out["world_state"] = ws + noise
    return out


def _sense_latent(agent: REEAgent, obs_dict: Dict[str, Any]):
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return agent.sense(
        obs_body=body, obs_world=world,
        obs_harm=_obs(obs_dict, "harm_obs"),
        obs_harm_a=_obs(obs_dict, "harm_obs_a"),
        obs_harm_history=_obs(obs_dict, "harm_history"),
    )


def _sample_class_diverse_batch(buffer: Deque, k: int, rng: random.Random):
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    if len(pool) <= k:
        return pool
    return rng.sample(pool, k)


def _e2_train_step(agent: REEAgent, buffer: Deque,
                   optimiser: torch.optim.Optimizer,
                   rng: random.Random) -> Optional[float]:
    """One world-forward training step. RECON-ONLY: reconstruction MSE on buffered
    one-step (z0, a, z1) transitions. The SD-056 contrastive auxiliary is omitted
    (confirmed P0 destabiliser, V3-EXQ-701b)."""
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)

    optimiser.zero_grad(set_to_none=True)
    z1_pred = agent.e2.world_forward(z0_K, actions_K)
    recon = F.mse_loss(z1_pred, z1_K)
    recon_val = float(recon.detach().item())
    if not math.isfinite(recon_val):
        return recon_val
    recon.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return recon_val


def _pe_from_metrics(metrics: Dict[str, Any]) -> Optional[float]:
    pe = metrics.get("e3_prediction_error")
    if pe is None:
        return None
    val = float(pe.detach().item()) if torch.is_tensor(pe) else float(pe)
    return val if math.isfinite(val) else None


def _step_cycle(agent: REEAgent, env: CausalGridWorldV2, obs_dict: Dict[str, Any],
                train: bool, buffer: Optional[Deque],
                e2_opt: Optional[torch.optim.Optimizer],
                sample_rng: Optional[random.Random],
                pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
                noise_sigma: float = 0.0,
                noise_gen: Optional[torch.Generator] = None,
                ) -> Tuple[Optional[float], Dict[str, Any], bool]:
    """One waking step. Returns (per_step_pe, next_obs_dict, done)."""
    # Observation noise (the NEGATIVE-CONTROL arms only) perturbs what the agent
    # PERCEIVES, so it enters at sense time. sigma=0.0 is a pass-through elsewhere.
    latent = _sense_latent(agent, _apply_obs_noise(obs_dict, noise_sigma, noise_gen))

    if train and buffer is not None:
        pend = pending_capture_ref[0]
        if pend is not None:
            z0_prev, a_prev = pend
            z1_obs = latent.z_world.detach().reshape(-1).clone()
            if (torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()):
                buffer.append((z0_prev, a_prev, z1_obs))
            pending_capture_ref[0] = None

    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device))
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)

    if action is None:
        idx = int(np.random.randint(0, env.action_dim))
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, idx] = 1.0
        agent._last_action = action
    if not torch.isfinite(action).all():
        return None, obs_dict, True

    if train and buffer is not None and torch.isfinite(latent.z_world).all():
        pending_capture_ref[0] = (
            latent.z_world.detach().reshape(-1).clone(),
            action.detach().reshape(-1).clone(),
        )
        if e2_opt is not None and sample_rng is not None:
            _e2_train_step(agent, buffer, e2_opt, sample_rng)

    _, harm_signal, done, info, next_obs_dict = env.step(action)
    with torch.no_grad():
        metrics = agent.update_residue(
            harm_signal=float(harm_signal), world_delta=None,
            hypothesis_tag=False, owned=True,
        )
    pe = _pe_from_metrics(metrics)
    return pe, next_obs_dict, bool(done)


def _cycle_quartile(steps_since: int, interval: int) -> int:
    """PRIMARY binning: position within the shift cycle, as a quartile.

    Reachable at every interval >= N_BINS by construction (the counter cycles
    0..interval-1), which is the property 798's absolute edges lacked. Quartile 0 is
    immediately post-shift (the model is maximally wrong); quartile N_BINS-1 is late
    in the cycle (the model has had as long as this arm's rate allows to re-learn).
    A non-shifting arm has no cycle, so all of its PE lands in quartile 0.
    """
    if interval <= 0:
        return 0
    q = (N_BINS * int(steps_since)) // int(interval)
    return int(min(N_BINS - 1, max(0, q)))


def _ssl_bin(steps_since: int) -> int:
    """SECONDARY binning, recorded for robustness only. 798's absolute edges."""
    lo, mid, hi = SSL_BIN_EDGES
    if steps_since <= lo:
        return 0
    if steps_since <= mid:
        return 1
    if steps_since <= hi:
        return 2
    return 3


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _run_step_budget(agent: REEAgent, env: CausalGridWorldV2, budget_steps: int,
                     steps_per_episode: int, train: bool,
                     buffer: Optional[Deque],
                     e2_opt: Optional[torch.optim.Optimizer],
                     sample_rng: Optional[random.Random],
                     arm_id: str, seed: int, phase: str, ep_offset: int,
                     interval: int = 0,
                     noise_sigma: float = 0.0,
                     noise_gen: Optional[torch.Generator] = None,
                     ) -> Dict[str, Any]:
    """Run until budget_steps env steps are consumed, across as many episodes as
    that takes.

    A fixed STEP budget (not a fixed episode count) is REQUIRED here: shift rate
    shortens episodes, so a fixed-episode window would give each arm a different
    total measurement length AND a different episode-phase mix. Returns per-step PE,
    PE binned BOTH by cycle quartile (primary) and by 798's absolute edges
    (secondary, recorded only), and the episode lengths (reported so the confound
    stays visible)."""
    all_pe: List[float] = []
    qbins: Dict[int, List[float]] = {i: [] for i in range(N_BINS)}
    sbins: Dict[int, List[float]] = {i: [] for i in range(N_SSL_BINS)}
    ep_lens: List[int] = []
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None]
    used = 0
    ep = 0
    while used < budget_steps:
        # Progress index is derived from STEPS consumed, not from the episode
        # counter. Episodes end early under a high shift rate, so an
        # episode-counted index would OVERSHOOT EPISODES_PER_RUN on exactly the
        # arms that matter most. Steps-derived, it tracks the phase budget
        # monotonically and cannot exceed it.
        glob_ep = ep_offset + min(used // max(1, steps_per_episode),
                                  max(0, (budget_steps // max(1, steps_per_episode)) - 1))
        print(f"  [train] {arm_id} seed={seed} {phase} ep {glob_ep+1}/"
              f"{EPISODES_PER_RUN}", flush=True)
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        pending_capture_ref[0] = None
        n_steps = 0
        for _s in range(steps_per_episode):
            if used >= budget_steps:
                break
            pe, obs_dict, done = _step_cycle(
                agent, env, obs_dict, train, buffer, e2_opt, sample_rng,
                pending_capture_ref, noise_sigma=noise_sigma, noise_gen=noise_gen,
            )
            used += 1
            n_steps += 1
            if pe is not None:
                ssl = int(env._steps_since_world_rule_shift)
                all_pe.append(pe)
                qbins[_cycle_quartile(ssl, interval)].append(pe)
                sbins[_ssl_bin(ssl)].append(pe)
            if done:
                break
        ep_lens.append(n_steps)
        ep += 1
    return {
        "all_pe": all_pe,
        "quartile_means": {i: _mean(qbins[i]) for i in range(N_BINS)},
        "quartile_counts": {i: len(qbins[i]) for i in range(N_BINS)},
        "bin_means": {i: _mean(sbins[i]) for i in range(N_SSL_BINS)},
        "bin_counts": {i: len(sbins[i]) for i in range(N_SSL_BINS)},
        "mean_episode_length": float(np.mean(ep_lens)) if ep_lens else 0.0,
        "n_episodes": len(ep_lens),
        "n_steps_used": used,
        "shift_count": int(env._world_rule_shift_count),
    }


def _sample_probe_battery(agent: REEAgent, seed: int, size: int,
                          steps: int) -> List[Tuple[torch.Tensor, torch.Tensor,
                                                    torch.Tensor]]:
    """FIXED held-out one-step transitions captured BEFORE training with the agent's
    own (frozen) encoder, so z0/z1 are CONSTANT across P0 checkpoints."""
    env = _make_probe_env(seed + 9973, **STABLE_DRIFT)
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    rng = np.random.default_rng(seed + 4242)
    for _t in range(steps):
        if len(battery) >= size:
            break
        latent = _sense_latent(agent, obs_dict)
        z_now = latent.z_world.detach().reshape(-1).clone()
        if prev is not None:
            z0_prev, a_prev = prev
            battery.append((z0_prev, a_prev, z_now))
        a_idx = int(rng.integers(0, env.action_dim))
        a_vec = torch.zeros(env.action_dim, dtype=torch.float32)
        a_vec[a_idx] = 1.0
        prev = (z_now, a_vec)
        _, _r, done, _info, obs_dict = env.step(a_idx)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            agent.e1.reset_hidden_state()
            prev = None
    return battery


def _frozen_probe_pe(agent: REEAgent, battery) -> float:
    """Same battery + frozen encoder -> reads ONLY world_forward improvement, i.e.
    the same one-step world mismatch the e3 prediction_error reads."""
    if not battery:
        return 0.0
    with torch.no_grad():
        z0 = torch.stack([b[0] for b in battery]).to(agent.device)
        a = torch.stack([b[1] for b in battery]).to(agent.device)
        z1 = torch.stack([b[2] for b in battery]).to(agent.device)
        pred = agent.e2.world_forward(z0, a)
        return float(F.mse_loss(pred, z1).item())


def _train_p0_and_probe(seed: int, conv_eps: int, steps: int, probe_size: int,
                        label: str) -> Dict[str, Any]:
    env_stable = _make_probe_env(seed, **STABLE_DRIFT)
    agent = _make_agent(env_stable)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    buffer: Deque = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    battery = _sample_probe_battery(agent, seed, probe_size, steps)
    pe_probe_init = _frozen_probe_pe(agent, battery)

    _run_step_budget(agent, env_stable, conv_eps * steps, steps, train=True,
                     buffer=buffer, e2_opt=e2_opt, sample_rng=sample_rng,
                     arm_id=label, seed=seed, phase="P0", ep_offset=0, interval=0)
    pe_probe_final = _frozen_probe_pe(agent, battery)
    conv_rel_drop = (((pe_probe_init - pe_probe_final) / pe_probe_init)
                     if pe_probe_init > 1e-12 else 0.0)
    return {
        "agent": agent, "battery": battery, "e2_opt": e2_opt, "buffer": buffer,
        "sample_rng": sample_rng,
        "pe_probe_init": pe_probe_init, "pe_probe_final": pe_probe_final,
        "conv_rel_drop": conv_rel_drop, "n_probe": len(battery),
    }


def _decay_frac(means: Dict[int, float], counts: Dict[int, int],
                n_bins: int) -> Optional[float]:
    """Relative PE drop from the immediately-post-shift bin to the late-cycle bin.
    Positive => the model re-learned => the load was REDUCIBLE (genuine learning
    load). ~0 => nothing to re-learn (noise). None => bins too sparse to read."""
    if (counts.get(0, 0) < MIN_BIN_SAMPLES
            or counts.get(n_bins - 1, 0) < MIN_BIN_SAMPLES):
        return None
    early = means.get(0, 0.0)
    late = means.get(n_bins - 1, 0.0)
    if early <= 1e-12:
        return None
    return float((early - late) / early)


# z_goal-stream liveness, pooled across the run's per-cell agents for the
# manifest block (read at end-of-cell, so no agent is retained for provenance).
_ZG = ZGoalStreamAccumulator()


def run_cell(arm: Dict[str, Any], seed: int, conv_eps: int, meas_steps: int,
             learn_steps: int, steps: int, probe_steps: int,
             probe_size: int) -> Dict[str, Any]:
    """One (arm, seed) cell: recon-only P0 on the STABLE base -> positive-control
    probes -> P1 FROZEN MEL measurement -> P2 TRAINING learnability probe."""
    arm_id = arm["arm_id"]
    interval = arm["interval"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    p0 = _train_p0_and_probe(seed, conv_eps, steps, probe_size, label=arm_id)
    agent = p0["agent"]

    # Positive-control probes on the FROZEN model (readiness R1).
    stable_pe = _run_step_budget(
        agent, _make_probe_env(seed, **STABLE_DRIFT), probe_steps, probe_steps,
        train=False, buffer=None, e2_opt=None, sample_rng=None, arm_id=arm_id,
        seed=seed, phase="PROBE_STABLE", ep_offset=conv_eps, interval=0)["all_pe"]
    shock_pe = _run_step_budget(
        agent, _make_probe_env(seed, **SHOCK_DRIFT), probe_steps, probe_steps,
        train=False, buffer=None, e2_opt=None, sample_rng=None, arm_id=arm_id,
        seed=seed, phase="PROBE_SHOCK", ep_offset=conv_eps, interval=0)["all_pe"]
    pe_stable = _mean(stable_pe)
    pe_shock = _mean(shock_pe)
    pe_response_rel = ((pe_shock / pe_stable) - 1.0) if pe_stable > 1e-12 else 0.0

    noise_gen = torch.Generator().manual_seed(seed + 31337)

    # P1 -- FROZEN MEL measurement (comparable to 701c / 718a / 798).
    meas = _run_step_budget(
        agent, _make_env(seed, interval, arm["depth"]), meas_steps, steps,
        train=False, buffer=None, e2_opt=None, sample_rng=None, arm_id=arm_id,
        seed=seed, phase="P1_MEL", ep_offset=conv_eps + 1, interval=interval,
        noise_sigma=arm["noise"], noise_gen=noise_gen)

    # P2 -- TRAINING learnability probe. A frozen model cannot re-learn, so decay is
    # only meaningful with the world-forward optimiser live.
    learn = _run_step_budget(
        agent, _make_env(seed + 1, interval, arm["depth"]), learn_steps, steps,
        train=True, buffer=p0["buffer"], e2_opt=p0["e2_opt"],
        sample_rng=p0["sample_rng"], arm_id=arm_id, seed=seed, phase="P2_LEARN",
        ep_offset=conv_eps + 1 + MEAS_EPISODE_EQUIV, interval=interval,
        noise_sigma=arm["noise"], noise_gen=noise_gen)

    print("verdict: PASS", flush=True)
    # z_goal liveness -- read AFTER this cell stepped; the agent is not retained.
    _ZG.observe(agent)
    return {
        "arm_id": arm_id,
        "level": arm["level"],
        "graded": bool(arm["graded"]),
        "seed": seed,
        "world_rule_shift_interval": interval,
        "world_rule_shift_depth": arm["depth"],
        "obs_noise_sigma": arm["noise"],
        "mel_mean_pe": _mean(meas["all_pe"]),
        "n_meas_pe": len(meas["all_pe"]),
        "meas_mean_episode_length": meas["mean_episode_length"],
        "meas_n_episodes": meas["n_episodes"],
        "meas_shift_count": meas["shift_count"],
        # PRIMARY binning (drives C3 / C4).
        "meas_pe_by_cycle_quartile": meas["quartile_means"],
        "meas_quartile_counts": meas["quartile_counts"],
        "learn_pe_by_cycle_quartile": learn["quartile_means"],
        "learn_quartile_counts": learn["quartile_counts"],
        "learn_decay_frac": _decay_frac(learn["quartile_means"],
                                        learn["quartile_counts"], N_BINS),
        # SECONDARY binning (798's absolute edges) -- recorded for robustness only,
        # enters NO criterion.
        "meas_pe_by_ssl_bin": meas["bin_means"],
        "meas_ssl_bin_counts": meas["bin_counts"],
        "learn_pe_by_ssl_bin": learn["bin_means"],
        "learn_ssl_bin_counts": learn["bin_counts"],
        "learn_decay_frac_abs_bins": _decay_frac(learn["bin_means"],
                                                 learn["bin_counts"], N_SSL_BINS),
        "learn_mean_pe": _mean(learn["all_pe"]),
        "learn_mean_episode_length": learn["mean_episode_length"],
        "learn_shift_count": learn["shift_count"],
        "pe_stable": pe_stable,
        "pe_shock": pe_shock,
        "pe_response_rel": pe_response_rel,
        "conv_metric": "frozen_probe_battery",
        "p0_recipe": "recon_only",
        "conv_pe_init": p0["pe_probe_init"],
        "conv_pe_final": p0["pe_probe_final"],
        "conv_rel_drop": p0["conv_rel_drop"],
        "n_probe": p0["n_probe"],
    }


def evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    seed_list = sorted({r["seed"] for r in rows})
    n_seeds = len(seed_list)

    def cell(arm_id: str, seed: int) -> Optional[Dict[str, Any]]:
        for r in rows:
            if r["arm_id"] == arm_id and r["seed"] == seed:
                return r
        return None

    def arm_mean(arm_id: str, key: str) -> float:
        return _mean([r[key] for r in rows if r["arm_id"] == arm_id])

    # -- Readiness (R1 / R2) on the recon-only converged base -----------------
    per_seed_conv = {s: _mean([r["conv_rel_drop"] for r in rows if r["seed"] == s])
                     for s in seed_list}
    per_seed_resp = {s: _mean([r["pe_response_rel"] for r in rows if r["seed"] == s])
                     for s in seed_list}
    conv_ready = [s for s in seed_list if per_seed_conv[s] >= MIN_REL_CONV_DROP]
    conv_seed_fraction = (len(conv_ready) / n_seeds) if n_seeds else 0.0
    r2_ok = conv_seed_fraction >= SEED_PASS_FRAC
    resp_ready = _mean([per_seed_resp[s] for s in conv_ready]) if conv_ready else 0.0
    r1_ok = resp_ready >= MIN_REL_PE_RESPONSE
    readiness_ok = bool(r2_ok and r1_ok)

    # -- C1 grading (graded arms only; the noise controls are EXCLUDED) -------
    c1_seed_pass: List[bool] = []
    per_seed_mel: Dict[int, Dict[str, float]] = {}
    for s in seed_list:
        vals = []
        rec = {}
        for a in GRADED_ORDER:
            c = cell(a, s)
            v = float(c["mel_mean_pe"]) if c else 0.0
            vals.append(v)
            rec[a] = v
        per_seed_mel[s] = rec
        base = vals[0]
        tol = MONO_TOL * base
        mono = all(vals[i] <= vals[i + 1] + tol for i in range(len(vals) - 1))
        spread = ((vals[-1] - base) / base) if base > 1e-12 else 0.0
        c1_seed_pass.append(bool(mono and spread >= MIN_REL_MEL_SPREAD))
    c1_frac = (sum(c1_seed_pass) / n_seeds) if n_seeds else 0.0
    c1_pass = c1_frac >= SEED_PASS_FRAC

    # -- C2 above-reference ---------------------------------------------------
    c2_seed_pass: List[bool] = []
    for s in seed_list:
        base = per_seed_mel[s][GRADED_ORDER[0]]
        c2_seed_pass.append(all(
            (per_seed_mel[s][a] - base) >= MIN_ABS_MEL_SPREAD
            for a in GRADED_ORDER[1:]))
    c2_frac = (sum(c2_seed_pass) / n_seeds) if n_seeds else 0.0
    c2_pass = c2_frac >= SEED_PASS_FRAC

    # -- C3 sustained non-convergence on HIGH --------------------------------
    # NOTE ON THE COMPARISON. ARM_0_NONE has NO shifts, so its
    # steps_since_world_rule_shift never advances and ALL of its PE lands in
    # quartile 0 -- its late quartile is empty (0.0). Comparing HIGH's late quartile
    # against NONE's late quartile would therefore be VACUOUS (any HIGH late-quartile
    # PE beats an empty 0.0). The meaningful reference is NONE's OVERALL mean, which
    # IS the stable-base level. C3 asks: does HIGH remain above the stable base even
    # LATE IN ITS OWN SHIFT CYCLE, i.e. does the world fail to settle back before the
    # next shift? Under quartile binning this is now genuinely readable on every seed
    # (in 798 the absolute late bin was empty and C3 silently fell back).
    c3_seed_pass = []
    c3_readable = []
    for s in seed_list:
        none_c, high_c = cell("ARM_0_NONE", s), cell("ARM_3_HIGH", s)
        if none_c is None or high_c is None:
            c3_seed_pass.append(False)
            c3_readable.append(False)
            continue
        stable_ref = none_c["mel_mean_pe"]
        late_high = high_c["meas_pe_by_cycle_quartile"].get(N_BINS - 1, 0.0)
        late_n = high_c["meas_quartile_counts"].get(N_BINS - 1, 0)
        # Fall back to HIGH's overall mean only if the late quartile is somehow too
        # sparse to read. _assert_quartile_reachability() makes that structurally
        # impossible for the pre-registered ladder; the branch is kept as a guard.
        if late_n >= MIN_BIN_SAMPLES:
            c3_readable.append(True)
            c3_seed_pass.append(bool(late_high > stable_ref))
        else:
            c3_readable.append(False)
            c3_seed_pass.append(bool(high_c["mel_mean_pe"] > stable_ref))
    c3_frac = (sum(c3_seed_pass) / n_seeds) if n_seeds else 0.0
    c3_pass = c3_frac >= SEED_PASS_FRAC

    # -- C4 learnability (THE anti-artifact criterion) -----------------------
    # PRE-REGISTERED control selection: the LOWEST-sigma noise arm that is at least
    # as elevated as the anchor on EVERY seed -- the most stringent FAIR control.
    # If none qualifies, the highest-sigma arm is reported as the best available and
    # the fairness precondition is recorded UNMET, which routes to
    # c4_unreadable_requeue rather than to any C4-dependent verdict.
    anchor_mel_by_seed = {s: (cell(C4_ANCHOR_ARM, s) or {}).get("mel_mean_pe", 0.0)
                          for s in seed_list}
    noise_elevation_ratios: Dict[str, Dict[str, Any]] = {}
    fair_controls: List[str] = []
    for ctrl_id in NOISE_CONTROL_ORDER:
        per_seed_ratio = {}
        for s in seed_list:
            c = cell(ctrl_id, s)
            a_mel = anchor_mel_by_seed.get(s, 0.0)
            per_seed_ratio[s] = ((float(c["mel_mean_pe"]) / a_mel)
                                 if (c is not None and a_mel > 1e-15) else 0.0)
        worst_seed = (min(per_seed_ratio, key=lambda k: per_seed_ratio[k])
                      if per_seed_ratio else None)
        worst_ratio = per_seed_ratio.get(worst_seed, 0.0) if worst_seed is not None else 0.0
        # Reported only (not a gate): how the control sits against the STEEPEST
        # graded arm, so a reader can see whether it also covers ARM_3_HIGH.
        vs_high = []
        for s in seed_list:
            c, h = cell(ctrl_id, s), cell("ARM_3_HIGH", s)
            if c is not None and h is not None and h["mel_mean_pe"] > 1e-15:
                vs_high.append(float(c["mel_mean_pe"]) / float(h["mel_mean_pe"]))
        noise_elevation_ratios[ctrl_id] = {
            "sigma": next(a["noise"] for a in ARMS if a["arm_id"] == ctrl_id),
            "mel_mean": arm_mean(ctrl_id, "mel_mean_pe"),
            "ratio_vs_anchor_per_seed": {str(k): v for k, v in per_seed_ratio.items()},
            "ratio_vs_anchor_worst_seed": worst_ratio,
            "worst_seed": (int(worst_seed) if worst_seed is not None else None),
            "ratio_vs_high_worst_seed": (min(vs_high) if vs_high else 0.0),
            "fair": bool(worst_ratio >= 1.0),
        }
        if worst_ratio >= 1.0:
            fair_controls.append(ctrl_id)

    c4_control_fair = bool(fair_controls)
    c4_control_arm = (fair_controls[0] if fair_controls
                      else NOISE_CONTROL_ORDER[-1])
    control_worst_ratio = noise_elevation_ratios[c4_control_arm][
        "ratio_vs_anchor_worst_seed"]

    anchor_decays = [c["learn_decay_frac"] for s in seed_list
                     if (c := cell(C4_ANCHOR_ARM, s)) is not None
                     and c["learn_decay_frac"] is not None]
    control_decays = [c["learn_decay_frac"] for s in seed_list
                      if (c := cell(c4_control_arm, s)) is not None
                      and c["learn_decay_frac"] is not None]
    anchor_decay_mean = _mean(anchor_decays) if anchor_decays else None
    control_decay_mean = _mean(control_decays) if control_decays else None

    # Readability is the SAMPLE-COUNT condition _decay_frac gates on. Report the
    # WORST cell across the C4 pair's first and last quartiles, so the number the
    # indexer recomputes is the same statistic the boolean quantifies over.
    c4_bin_counts: List[Tuple[str, int, int]] = []
    for arm_id in (C4_ANCHOR_ARM, c4_control_arm):
        for s in seed_list:
            c = cell(arm_id, s)
            if c is None:
                c4_bin_counts.append((f"{arm_id}|seed{s}", s, 0))
                continue
            q = c["learn_quartile_counts"]
            c4_bin_counts.append((f"{arm_id}|seed{s}", s,
                                  min(int(q.get(0, 0)), int(q.get(N_BINS - 1, 0)))))
    worst_bin = min(c4_bin_counts, key=lambda t: t[2]) if c4_bin_counts else ("none", -1, 0)
    c4_bins_readable = bool(anchor_decay_mean is not None
                            and control_decay_mean is not None)
    c4_measurable = bool(c4_bins_readable and c4_control_fair)
    c4_pass = bool(
        c4_measurable
        and anchor_decay_mean >= MIN_LEARN_DECAY
        and control_decay_mean <= MAX_NOISE_DECAY
    )

    # Reported diagnostic, NOT a criterion and NOT threshold-fitted: the decay of
    # every graded arm. Genuine learning load should decay MORE as the inter-shift
    # window lengthens; an un-learnable elevation is flat at every rate.
    learn_decay_by_arm = {}
    for a in ARMS:
        vals = [r["learn_decay_frac"] for r in rows
                if r["arm_id"] == a["arm_id"] and r["learn_decay_frac"] is not None]
        learn_decay_by_arm[a["arm_id"]] = {
            "interval": a["interval"],
            "obs_noise_sigma": a["noise"],
            "decay_mean": (_mean(vals) if vals else None),
            "n_seeds_readable": len(vals),
            "decay_mean_abs_bins": (
                _mean([r["learn_decay_frac_abs_bins"] for r in rows
                       if r["arm_id"] == a["arm_id"]
                       and r["learn_decay_frac_abs_bins"] is not None])
                if any(r["learn_decay_frac_abs_bins"] is not None for r in rows
                       if r["arm_id"] == a["arm_id"]) else None),
        }

    # -- Preconditions --------------------------------------------------------
    # Each `measured` is the SAME statistic its `met` quantifies over, so the
    # indexer's recompute agrees with the flag (the 798 mean-vs-quantifier trap).
    preconditions = [
        {"name": "world_model_converged_frozen_probe",
         "description": "recon-only P0 base converges on the FIXED frozen probe "
                        "battery on at least SEED_PASS_FRAC of seeds (readiness R2)",
         "measured": float(conv_seed_fraction),
         "threshold": float(SEED_PASS_FRAC), "direction": "lower",
         "control": "fraction of seeds whose stable-base P0 conv_rel_drop clears "
                    "MIN_REL_CONV_DROP; same seed-fraction statistic `met` tests",
         "met": bool(r2_ok)},
        {"name": "frozen_pe_responds_to_shock",
         "description": "on the converged frozen base, PE rises under a max-novelty "
                        "shock (readiness R1) -- the instrument can move at all",
         "measured": float(resp_ready),
         "threshold": MIN_REL_PE_RESPONSE, "direction": "lower",
         "control": "SHOCK_DRIFT positive control vs STABLE_DRIFT, ready seeds. This "
                    "is also the SAME KIND of relative-PE-reduction statistic C4's "
                    "decay_frac routes on, measured on a positive control",
         "met": bool(r1_ok)},
        {"name": "learnability_quartiles_populated",
         "description": "the first and last cycle quartiles both carry enough "
                        "samples for the C4 decay comparison to be readable, on "
                        "every cell of the anchor/control pair. THIS IS THE 798 "
                        "DEFECT: there its top bin held 0 samples by construction",
         "measured": float(worst_bin[2]),
         "threshold": float(MIN_BIN_SAMPLES), "direction": "lower",
         "control": f"worst cell of {C4_ANCHOR_ARM} and {c4_control_arm} over "
                    f"quartile 0 and quartile {N_BINS - 1}",
         "offending_cell": worst_bin[0],
         "met": bool(c4_bins_readable)},
        {"name": "noise_control_at_least_as_elevated",
         "description": "the selected negative control's MEL is at least the "
                        "anchor's on EVERY seed, so the C4 decay comparison is "
                        "fair-or-conservative rather than won by the graded arm "
                        "simply having more PE. 798 failed this at sigma 0.05",
         "measured": float(control_worst_ratio),
         "threshold": 1.0, "direction": "lower",
         "control": f"{c4_control_arm} mel / {C4_ANCHOR_ARM} mel, worst seed",
         "offending_cell": (f"{c4_control_arm}|seed"
                            f"{noise_elevation_ratios[c4_control_arm]['worst_seed']}"),
         "met": bool(c4_control_fair)},
        {"name": "cycle_quartiles_reachable_by_construction",
         "description": "design-time proof that every quartile is non-empty on "
                        "every shift-enabled arm: the counter cycles 0..interval-1, "
                        "so N_BINS quartiles are all reachable when interval >= "
                        "N_BINS. Asserted from the pre-registered ARMS table before "
                        "any compute is spent",
         "measured": float(QUARTILE_REACHABILITY["min_shift_interval"]),
         "threshold": float(MIN_INTERVAL_FOR_QUARTILES), "direction": "lower",
         "control": "minimum world_rule_shift_interval over shift-enabled arms",
         "met": True},
    ]

    # -- Routing (EXHAUSTIVE over the sixteen C1..C4 combinations) ------------
    c1c2 = bool(c1_pass and c2_pass)
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        note = ("Readiness unmet on the recon-only base (R2 conv_seed_fraction "
                f"{conv_seed_fraction:.2f}, R1 pe_response {resp_ready:.3f}). The "
                "graded-MEL readings are UNINTERPRETABLE without a converged, "
                "novelty-responsive base -- this is a re-queue at an adequate P0, "
                "NOT a verdict on the test-bed.")
    elif not c4_measurable:
        label = "c4_unreadable_requeue"
        outcome = "FAIL"
        why = []
        if not c4_bins_readable:
            why.append(f"the C4 quartiles are too sparse to read (worst cell "
                       f"{worst_bin[0]} holds {worst_bin[2]} samples against a floor "
                       f"of {MIN_BIN_SAMPLES})")
        if not c4_control_fair:
            why.append(f"no noise control reached the anchor's MEL on every seed "
                       f"(best available {c4_control_arm}, worst-seed ratio "
                       f"{control_worst_ratio:.3f} against 1.0) -- raise NOISE_SIGMA "
                       f"using the achieved sigma -> MEL points in "
                       f"noise_elevation_ratios")
        note = ("C4 -- the anti-artifact learnability criterion -- was NOT MEASURABLE: "
                + "; ".join(why) + ". NO learnability verdict is asserted, and in "
                "particular this run does NOT claim the elevation is an artifact "
                "(that is the V3-EXQ-798 defect this iteration exists to fix). The "
                f"other criteria stand on their own: C1 {c1_pass} "
                f"({c1_frac:.2f} of seeds), C2 {c2_pass} ({c2_frac:.2f}), C3 "
                f"{c3_pass} ({c3_frac:.2f}). Re-queue the C4 leg with the recorded "
                "calibration.")
    elif not c1c2:
        if c2_pass:
            label = "producer_elevates_but_not_graded"
            outcome = "FAIL"
            note = ("The rule shift lifts MEL above the stable-base reference (C2) "
                    "but not monotonically in shift rate (C1). The knob is a novelty "
                    "source but not yet a GRADED one; re-centre the interval ladder "
                    "before the ecological run.")
        else:
            label = "producer_ineffective"
            outcome = "FAIL"
            note = ("The rule shift did not lift MEL above the stable-base "
                    "reference. Same outcome class as the env_drift knob it was "
                    "built to replace.")
    elif not c4_pass:
        label = "producer_graded_but_not_learnable"
        outcome = "FAIL"
        note = ("ARTIFACT: MEL is graded and above reference, but on the anchor arm "
                f"{C4_ANCHOR_ARM} (interval "
                f"{QUARTILE_REACHABILITY['anchor_interval']}, the arm with the most "
                "re-learning headroom) the elevation does NOT decay under training "
                f"(anchor decay {anchor_decay_mean}, floor {MIN_LEARN_DECAY}) and is "
                f"not separable from the matched, at-least-as-elevated noise control "
                f"{c4_control_arm} (decay {control_decay_mean}, ceiling "
                f"{MAX_NOISE_DECAY}). This reading IS measured, unlike V3-EXQ-798's. "
                "A graded-but-unlearnable ladder is a DV-symmetry artifact, not a "
                "novelty producer -- do NOT use this test-bed for the ecological "
                "MECH-180 run on the strength of C1/C2.")
    elif not c3_pass:
        label = "producer_graded_learnable_but_settles"
        outcome = "FAIL"
        note = ("The load is graded, above reference, and genuinely REDUCIBLE (C1, "
                "C2, C4 all pass), but the HIGH arm settles back to the stable-base "
                "level late in its own shift cycle (C3 fail), so the world is not "
                "sustainedly non-converging at the fastest rate in the ladder. The "
                "producer mechanism is real; the RATE ladder needs re-centring "
                "upward before the ecological run.")
    else:
        label = "producer_validated_graded_learnable"
        outcome = "PASS"
        note = ("SD-MEL-PRODUCER produces a graded, above-reference waking MEL load "
                "that is REDUCIBLE (post-shift PE decays under training on the "
                f"{C4_ANCHOR_ARM} anchor) while the matched, at-least-as-elevated "
                "noise control does not. The test-bed is validated as a novelty "
                "producer. This licenses the SEPARATE, still-gated MECH-180 "
                "ecological end-to-end run; it is NOT itself MECH-180 evidence.")

    criteria = [
        {"name": "C1_mel_graded_in_shift_rate", "load_bearing": True,
         "passed": bool(c1_pass), "seed_fraction": c1_frac},
        {"name": "C2_mel_above_stable_reference", "load_bearing": False,
         "passed": bool(c2_pass), "seed_fraction": c2_frac},
        {"name": "C3_sustained_non_convergence", "load_bearing": False,
         "passed": bool(c3_pass), "seed_fraction": c3_frac,
         "late_quartile_readable_seeds": int(sum(c3_readable))},
        {"name": "C4_load_is_learnable_not_noise", "load_bearing": True,
         "passed": bool(c4_pass),
         "anchor_arm": C4_ANCHOR_ARM,
         "control_arm": c4_control_arm,
         "anchor_decay_mean": anchor_decay_mean,
         "control_decay_mean": control_decay_mean,
         "measurable": bool(c4_measurable)},
    ]

    # Non-degeneracy: a criterion is degenerate if its inputs carry no spread, or if
    # it was not measurable at all.
    mel_all = [r["mel_mean_pe"] for r in rows]
    mel_spread = bool(len(set(round(v, 12) for v in mel_all)) > 1)
    criteria_non_degenerate = {
        "C1": mel_spread,
        "C2": mel_spread,
        "C3": bool(any(c3_readable)),
        "C4": bool(c4_measurable),
    }

    degeneracy = check_degeneracy({
        "mel_mean_pe": mel_all,
        "learn_decay_frac": [r["learn_decay_frac"] for r in rows
                             if r["learn_decay_frac"] is not None],
    })

    return {
        "outcome": outcome, "label": label, "note": note,
        "evidence_direction": "non_contributory",
        "preconditions": preconditions,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "degeneracy": degeneracy,
        "per_seed_mel": per_seed_mel,
        "per_seed_conv_rel_drop": per_seed_conv,
        "per_seed_pe_response_rel": per_seed_resp,
        "conv_seed_fraction": conv_seed_fraction,
        "readiness_ok": readiness_ok, "r1_ok": r1_ok, "r2_ok": r2_ok,
        "c1_pass": c1_pass, "c1_frac": c1_frac,
        "c2_pass": c2_pass, "c2_frac": c2_frac,
        "c3_pass": c3_pass, "c3_frac": c3_frac,
        "c4_pass": c4_pass,
        "c4_measurable": c4_measurable,
        "c4_bins_readable": c4_bins_readable,
        "c4_control_fair": c4_control_fair,
        "c4_anchor_arm": C4_ANCHOR_ARM,
        "c4_control_arm": c4_control_arm,
        "c4_worst_quartile_cell": worst_bin[0],
        "c4_worst_quartile_count": worst_bin[2],
        "anchor_decay_mean": anchor_decay_mean,
        "control_decay_mean": control_decay_mean,
        "noise_elevation_ratios": noise_elevation_ratios,
        "learn_decay_by_arm": learn_decay_by_arm,
        "quartile_reachability": QUARTILE_REACHABILITY,
        "mel_per_arm": {a: arm_mean(a, "mel_mean_pe")
                        for a in [x["arm_id"] for x in ARMS]},
        "mean_episode_length_per_arm": {
            a: arm_mean(a, "meas_mean_episode_length")
            for a in [x["arm_id"] for x in ARMS]},
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    # Feeds the always-core `elapsed_seconds` via the stamper. V3-EXQ-798's landed
    # manifest recorded it as null (validate_recording --strict flags that as an
    # always-core gap); threading the clock through fixes it here.
    t_start = time.perf_counter()
    if dry_run:
        seeds = [42]
        # The smoke budget MUST span at least two full shift cycles of the ANCHOR
        # (interval 60), or the last quartile is empty and the smoke cannot exercise
        # the very thing this iteration repairs -- it would only ever demonstrate the
        # new c4_unreadable_requeue guard. 150 steps -> 2 cycles -> ~30 samples in
        # quartile 3, comfortably over MIN_BIN_SAMPLES.
        conv_eps, meas_steps, learn_steps = 2, 150, 150
        steps, probe_steps, probe_size = 12, 12, 8
    else:
        seeds = SEEDS
        conv_eps, meas_steps, learn_steps = CONV_EPISODES, MEAS_STEPS, LEARN_STEPS
        steps, probe_steps, probe_size = (STEPS_PER_EPISODE, PROBE_STEPS,
                                          PROBE_BATTERY_SIZE)

    cell_config = {
        "env_base": ENV_BASE, "conv_eps": conv_eps, "meas_steps": meas_steps,
        "learn_steps": learn_steps, "steps": steps, "probe_steps": probe_steps,
        "probe_size": probe_size, "sd056_weight": SD056_WEIGHT, "e2_lr": E2_LR,
        "p0_loss": "recon_only", "conv_metric": "frozen_probe_battery",
        # The BINNING is declared in the cell's config slice because the cell's
        # recorded readouts depend on it. Omitting it (as V3-EXQ-798's slice did)
        # would let a future consumer FALSE-HIT this cell's fingerprint and silently
        # read bin means computed under a different scheme.
        "binning": "cycle_quartile_primary+abs_edges_secondary",
        "n_bins": N_BINS, "ssl_bin_edges": list(SSL_BIN_EDGES),
    }

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            slice_cfg = dict(cell_config)
            slice_cfg.update({
                "arm_id": arm["arm_id"],
                "world_rule_shift_interval": arm["interval"],
                "world_rule_shift_depth": arm["depth"],
                "obs_noise_sigma": arm["noise"],
            })
            with arm_cell(seed, config_slice=slice_cfg,
                          script_path=Path(__file__),
                          include_driver_script_in_hash=False) as cell:
                row = run_cell(arm, seed, conv_eps, meas_steps, learn_steps, steps,
                               probe_steps, probe_size)
                cell.stamp(row)
            rows.append(row)

    ev = evaluate(rows)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": ev["outcome"],
        "result": ev["outcome"],
        "evidence_direction": ev["evidence_direction"],
        "evidence_direction_note": ev["note"],
        "sleep_driver_pattern": "not_applicable_no_sleep",
        "interpretation": {
            "label": ev["label"],
            "preconditions": ev["preconditions"],
            "criteria_non_degenerate": ev["criteria_non_degenerate"],
        },
        "criteria": ev["criteria"],
        "summary": {
            "mel_mean_pe_per_arm": ev["mel_per_arm"],
            "mean_episode_length_per_arm": ev["mean_episode_length_per_arm"],
            "per_seed_mel": ev["per_seed_mel"],
            "per_seed_conv_rel_drop": ev["per_seed_conv_rel_drop"],
            "per_seed_pe_response_rel": ev["per_seed_pe_response_rel"],
            "conv_seed_fraction": ev["conv_seed_fraction"],
            "readiness_ok": ev["readiness_ok"],
            "r1_ok": ev["r1_ok"], "r2_ok": ev["r2_ok"],
            "c1_pass": ev["c1_pass"], "c1_frac": ev["c1_frac"],
            "c2_pass": ev["c2_pass"], "c2_frac": ev["c2_frac"],
            "c3_pass": ev["c3_pass"], "c3_frac": ev["c3_frac"],
            "c4_pass": ev["c4_pass"],
            "c4_measurable": ev["c4_measurable"],
            "c4_bins_readable": ev["c4_bins_readable"],
            "c4_control_fair": ev["c4_control_fair"],
            "c4_anchor_arm": ev["c4_anchor_arm"],
            "c4_control_arm": ev["c4_control_arm"],
            "c4_worst_quartile_cell": ev["c4_worst_quartile_cell"],
            "c4_worst_quartile_count": ev["c4_worst_quartile_count"],
            "anchor_decay_mean": ev["anchor_decay_mean"],
            "control_decay_mean": ev["control_decay_mean"],
            "noise_elevation_ratios": ev["noise_elevation_ratios"],
            "learn_decay_by_arm": ev["learn_decay_by_arm"],
            "quartile_reachability": ev["quartile_reachability"],
            "conv_metric": "frozen_probe_battery",
            "p0_recipe": "recon_only",
            "thresholds": {
                "MIN_REL_PE_RESPONSE": MIN_REL_PE_RESPONSE,
                "MIN_REL_CONV_DROP": MIN_REL_CONV_DROP,
                "SEED_PASS_FRAC": SEED_PASS_FRAC,
                "MIN_REL_MEL_SPREAD": MIN_REL_MEL_SPREAD,
                "MIN_ABS_MEL_SPREAD": MIN_ABS_MEL_SPREAD,
                "MIN_LEARN_DECAY": MIN_LEARN_DECAY,
                "MAX_NOISE_DECAY": MAX_NOISE_DECAY,
                "MONO_TOL": MONO_TOL,
                "MIN_BIN_SAMPLES": MIN_BIN_SAMPLES,
                "NOISE_SIGMA_LO": NOISE_SIGMA_LO,
                "NOISE_SIGMA_HI": NOISE_SIGMA_HI,
            },
        },
        "config": {
            "seeds": seeds, "conv_episodes": conv_eps, "meas_steps": meas_steps,
            "learn_steps": learn_steps, "steps_per_episode": steps,
            "probe_steps": probe_steps, "probe_battery_size": probe_size,
            "episodes_per_run": EPISODES_PER_RUN, "arms": ARMS,
            "env_base": ENV_BASE, "stable_drift": STABLE_DRIFT,
            "shock_drift": SHOCK_DRIFT, "p0_loss": "recon_only",
            "sd056_weight": SD056_WEIGHT, "e2_lr": E2_LR,
            "conv_metric": "frozen_probe_battery", "sleep_used": False,
            "substrate": "SD-MEL-PRODUCER",
            "binning_primary": "cycle_quartile", "n_bins": N_BINS,
            "binning_secondary_abs_edges": list(SSL_BIN_EDGES),
            "c4_anchor_arm": C4_ANCHOR_ARM,
            "noise_control_order": NOISE_CONTROL_ORDER,
        },
        "arm_results": rows,
    }
    manifest.update(ev["degeneracy"])

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=manifest.get("config"),
        seeds=seeds, script_path=Path(__file__), started_at=t_start,
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"\nResults written to {out_path}")
    print(f"Outcome: {ev['outcome']}  Label: {ev['label']}")
    print(f"MEL per arm: {ev['mel_per_arm']}")
    print(f"Mean episode length per arm: {ev['mean_episode_length_per_arm']}")
    print(f"C1 {ev['c1_pass']} C2 {ev['c2_pass']} C3 {ev['c3_pass']} "
          f"C4 {ev['c4_pass']} (measurable {ev['c4_measurable']}: bins "
          f"{ev['c4_bins_readable']}, fair control {ev['c4_control_fair']})")
    print(f"C4 anchor {ev['c4_anchor_arm']} decay {ev['anchor_decay_mean']}  "
          f"control {ev['c4_control_arm']} decay {ev['control_decay_mean']}")
    print(f"Decay by arm: {ev['learn_decay_by_arm']}")
    return {"outcome": ev["outcome"], "out_path": str(out_path)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Minimal smoke test (1 seed, tiny budgets)")
    args = parser.parse_args()

    t0 = time.time()
    print(f"{QUEUE_ID} {EXPERIMENT_TYPE}")
    print(f"supersedes {SUPERSEDES}; quartile reachability {QUARTILE_REACHABILITY}")
    if args.dry_run:
        print("[DRY RUN] smoke test")

    result = run_experiment(dry_run=args.dry_run)
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    _oc = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_oc if _oc in ("PASS", "FAIL") else "FAIL",
        manifest_path=result["out_path"],
        dry_run=args.dry_run,
    )
