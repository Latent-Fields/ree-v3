"""V3-EXQ-939a: MECH-303 behavioural retest, SAME QUESTION as V3-EXQ-939, run under a
REPAIRED readiness gate.

SUPERSEDES v3_exq_939_mech303_proximity_gated_contextual_safety_vigilance_release_20260818T213039Z_v3.

DOUBLE-COUNTING GUARD (binding, from failure_autopsy_V3-EXQ-939_2026-08-20). 939 and 939a
answer the SAME question with the SAME design. They are NOT independent evidence and MUST NOT
both be counted toward MECH-303's supports. Governance re-scored 939's manifest as provisional
evidence so a completed result was not lost in the interim; this run is the confirming re-run
that the re-score defers to. `supersedes` is emitted in the manifest for exactly that reason.

WHY A RE-RUN AT ALL, AND WHY IT IS NOT A REPLAY (GOV-REUSE-1, Step 2.4). The decisive readout
-- release_rate under use_contextual_safety_terrain=True + gate_source='proximity_signal' with
LIVE accumulation -- IS already recorded, in 939's own manifest, on substrate_hash
a0abd50e06e278f7a7ed551428ffcca9f3e1511781d5a66dc9c6169b77f510d0. That is stated plainly here
rather than papered over: the answer is recoverable, and governance HAS recovered it. What 939
does not contain is that answer taken under a correct PRE-REGISTERED gate; its own gate
self-routed the run non_contributory, so the re-score rests on a post-hoc reinterpretation of a
manifest that disagrees with itself. This run supplies the pre-registered artifact.

It is also not a bit-identical replay. arm_cell() resets all RNG per cell, so a cell is a pure
function of (substrate, config_slice, seed) -- verified below to ~7 significant figures ACROSS
MACHINE CLASSES -- and re-running 939 unchanged would therefore reproduce 939's numbers and
carry no information. N_TEST_TRIALS 30 -> 90 changes the config_slice, so every cell here is a
fresh realisation, and it is the change the autopsy's own `scale: borderline` finding asks for.

THE FOUR REPAIRS (failure_autopsy_V3-EXQ-939_2026-08-20 requeue_spec).

(1) AGGREGATION ALIGNED, AND NOW DERIVED RATHER THAN DECLARED. 939's readiness control
    aggregated arm A's release_rate by MIN (all(r >= floor)) while the DVs it guards aggregate
    by MEAN. Same statistic, so the V3-EXQ-643 rule was met in letter, but the gate was
    strictly harsher than the criterion it guarded and voided a run whose mean cleared the
    floor by 0.099. Here the floor is not an independent constant at all:

        DV1 = mean(A) - mean(B),  DV3 = mean(A) - mean(C),  with mean(B), mean(C) >= 0
        =>  DV1 <= mean(A)  and  DV3 <= mean(A)
        =>  mean(A) >= DV_MARGIN is the EXACT necessary condition for either to clear.

    So READINESS_FLOOR_MEAN is DV_MARGIN itself -- same statistic, same aggregation, same
    threshold. The gate is provably never harsher than the criteria it guards, and it cannot
    drift out of alignment with them, because there is no second number to drift.

(2) THE FLOOR IS ON THE ACHIEVABLE LATTICE, BY CONSTRUCTION. release_rate is a count of
    crossings out of n trials, so it only takes values k/n. 939's 0.34 against n=30 lay
    strictly between 10/30 = 0.33333 and 11/30 = 0.36667, giving an undeclared effective floor
    of 0.36667 -- a silent 7.8% tightening -- and seed 1 missed it by 0.2 of one trial. Two
    fixes are applied together:
      * N_TEST_TRIALS 30 -> 90. The lattice goes from 1/30 = 0.0333 to 1/90 = 0.0111 and the
        per-seed binomial SE at p ~ 0.44 falls from ~0.091 to ~0.052, which is the autopsy's
        `scale: borderline` finding ("arm-A per-seed spread 0.3333-0.5000 makes a gate at the
        centre of that spread fragile by construction") answered directly. Cost is ~2% of
        runtime: the test phase is 90 of ~2730 agent steps per cell.
      * The per-seed floor is a FUNCTION of the live trial count, `_per_seed_floor()`, not a
        constant: floor(DV_MARGIN * n) / n. It is quantised to whatever n actually is, so it
        cannot go off-lattice if n is ever changed again -- including under --dry-run, where
        939's constant would have been off-lattice against n=6. It rounds DOWN, not up: the
        autopsy's headline finding is that a readiness gate must never be harsher than the
        criterion it guards, and floor() is the direction that keeps the guard from ever
        exceeding the DV_MARGIN it is derived from. (The autopsy offered the round-UP
        alternative, 11/30; it is rejected here for that reason, and this is a deliberate
        choice, not an oversight.)

(3) THE PRE-REGISTERED 2-OF-6 TOLERANCE NOW BINDS. 939 declared MIN_VALID_SEEDS = 4 of 6 and
    then vetoed it: readiness_ok already required EVERY arm-A cell to clear the same floor, so
    `n_valid_seeds >= 4` could never be the binding constraint. With the readiness gate moved
    to the MEAN, the tolerance is the only per-seed guard left and it binds. N_SEEDS stays 6
    and MIN_VALID_SEEDS stays 4 precisely so the tolerance is the one that was pre-registered,
    not a new one.
    WITNESS that it has a reachable binding state (the autopsy's learning #3): three arm-A
    seeds at 0.30 and three at 0.55 gives mean 0.425 >= 0.34, so readiness passes, while only
    3 of 6 seeds clear the per-seed floor -- 3 < 4, so the tolerance blocks. Reachable, and
    reachable in the region the observed data actually sits in.

(4) THE CALIBRATION GAP IS EXPLAINED, AND THE MECHANISM THAT PRODUCED IT IS REMOVED.
    939's queue entry reported a "Full-scale 1-seed calibration: A release 0.900 (terrain
    1.235), C 0.000 (terrain 0.071), D 1.000 (terrain 2.099)" against a realised A 0.4667
    (terrain 0.6155), C 0.0 (0.0027), D 0.4333 (0.8592) at the same seed 0 -- roughly 2x.
    The obvious hypothesis was cross-machine-class divergence (the calibration was run on the
    Mac, the run on ree-cloud-2). IT WAS TESTED AND IT IS FALSE. Re-running the committed 939
    driver at its committed constants on darwin-arm64 / torch 2.12.0, seed 0, reproduces the
    ree-cloud-2 (linux-x86_64 / torch 2.12.0+cpu) run EXACTLY on every discrete readout --
    release_rate 0.4666666666666667 / 0.0 / 0.0 / 0.43333333333333335, num_safety_steps
    240/0/6/240, total_safety 12.000027656555176 -- and to ~7 significant figures on the
    float32 terrain read (0.61553285 vs 0.61553312). This DV is machine-class portable, and it
    is portable for a structural reason: the agent's selected action is discarded (the env is
    stepped with an independent random.randint draw), so the one path that is known NOT to be
    portable, the torch.multinomial draw in e3_selector, cannot reach the trajectory.
    The actual finding is therefore stronger and worse than machine-class drift: THE
    CALIBRATION FIGURE IS NOT REPRODUCIBLE FROM THE COMMITTED SCRIPT ON EITHER MACHINE CLASS.
    It described a configuration that did not ship, and its provenance -- constants, commit,
    machine -- was never recorded, so which configuration cannot now be recovered.
    The removal is procedural plus structural. Procedural: this experiment's queue-entry
    calibration is produced BY THE SHIPPED SCRIPT AT ITS SHIPPED CONSTANTS, and states its
    machine, torch version and seed, so it is checkable. Structural: a hand-run release-rate
    figure is a weak artifact whatever its provenance, because release_rate is a threshold
    crossing count and says nothing about how close to the threshold the run was sitting. So
    the run now RECORDS ITS OWN OPERATING POINT -- per arm, the mean/min/max terrain read and
    its margin over the fixed release threshold, under
    custom_information.terrain_read_operating_point. That is the quantity that actually
    explains the release rate (939's arm-A read sat at 0.578 against a 0.5 threshold, a margin
    of only 0.078, which is why ~44% of trials crossed rather than ~90%), and it makes any
    future calibration-vs-run discrepancy diagnosable from the manifest alone instead of
    requiring a re-run to investigate. It is deliberately a NON-GATING diagnostic, emitted on
    PASS runs too: gating on it would assert a PROXY statistic against the criterion the DVs
    route on, which is exactly the V3-EXQ-643 defect, and its realised margin is thin enough
    that it would have become a second way to void a sound run -- the failure 939a exists to
    fix.

WHAT IS DELIBERATELY UNCHANGED. The scientific design is not touched: same 2x2, same arms,
same contexts, same gate sources and thresholds, same DV_MARGIN = 0.34, same DVs, same
load-bearing assignment, same 6 seeds, same substrate config, same phased training, same
recording spec. Only the readiness gate, the trial count, and the recorded diagnostics change.

DESIGN -- 2x2, EVERY ARM TESTED IN ITS OWN EXPOSURE CONTEXT.
Crossing {exposure context: SAFE (num_hazards=0) / HAZARD (num_hazards=8)} x {accumulation
gate: NATURAL (shipped default threshold 0.25) / FORCED (threshold overridden)}:

  A_safe_gate_natural         SAFE   context, prox_thresh=0.25 -> gate OPEN  -> ACCUMULATES
  B_safe_gate_forced_closed   SAFE   context, prox_thresh=0.0  -> gate CLOSED-> no accumulation
  C_hazard_gate_natural       HAZARD context, prox_thresh=0.25 -> gate CLOSED-> no accumulation
  D_hazard_gate_forced_open   HAZARD context, prox_thresh=2.0  -> gate OPEN  -> ACCUMULATES

A/B and C/D are WITHIN-CONTEXT contrasts, so neither depends on cross-context z_world keying.
D is the yoked-accumulation control: it asks whether the terrain CAN read above the release
threshold IN THE HAZARDOUS CONTEXT when accumulation is permitted there. If it can, C's
non-release is caused by the gate WITHHOLDING accumulation, not by the hazardous context
suppressing the read -- the confound that caps V3-EXQ-764.

NOTE ON THE ABSENT FLAG-OFF ARM. use_contextual_safety_terrain=False is deliberately NOT an
arm: that ablation is near-analytic (the release block is flag-gated, so it returns 0 by
construction) and 764 already measured it at a 0.748 gap. B -- terrain flag ON, read path live,
accumulation gate held shut -- is the informative ablation of the same thing.

RELATION TO THE TWO PREDECESSORS (unchanged from 939, restated because it governs tagging).
V3-EXQ-930 validated the DEDICATED SIGNAL at the SIGNAL LAYER only and NEVER ENABLED THE GATE
(use_contextual_safety_terrain stayed False, the agent was untrained, no vigilance readout was
taken); governance overrode its self-stamped `supports` to non_contributory. It is NOT cited as
support here. V3-EXQ-764 is the promote-to-active falsifier, BUILT + validated but HELD since
2026-07-15 on the z_world SAFE-vs-UNSAFE separability ceiling (SD-008, rank AUC ~0.83); that
blocker is UNCHANGED, because SD-MECH303-THRESHOLD-SOURCING fixed the WRITE gate, not the READ
keying. This design routes its load-bearing contrasts through ACCUMULATION HISTORY instead of
cross-context keying, so it is not 764 re-queued and must not be read as clearing 764. 764
stays HELD.

DVs (pre-registered constants below; PASS = readiness AND tolerance AND DV1 AND DV2 AND DV3,
also recorded verbatim as `combination_rule`):
  DV1 accumulation-necessity    (context held SAFE):   release_A - release_B >= DV_MARGIN
  DV2 accumulation-is-the-cause (context held HAZARD): release_D - release_C >= DV_MARGIN
  DV3 gate-natural context appropriateness:            release_A - release_C >= DV_MARGIN

LOAD-BEARING: DV2 and DV3. DV1 is reported and gates the PASS but is marked NOT load-bearing:
forcing the gate shut is a config override, so a large A-B gap is close to structural. DV3 is
the genuinely non-analytic one -- whether the SHIPPED default threshold (0.25) naturally
withholds accumulation in a hazardous context by enough to change the behavioural readout is an
empirical question about the proximity EMA's realized distribution under a moving agent, the
RBF read at the test z_world, and the release threshold. It could come out either way.

DV-SYMMETRY DECLARATION (one line per arm; the V3-EXQ-604c class). The DV, release_rate, is a
count of threshold crossings of `residue_field.evaluate_safety(z_world).mean()` against a FIXED
pre-registered contextual_safety_release_threshold. Its symmetry group is (i) any transform of
the safety scalar preserving each trial's position relative to that fixed threshold, and (ii)
permutation of test trials. Every arm's manipulation changes the accumulated terrain's
MAGNITUDE (by ~accum_weight x num_safety_steps at the test z_world), moving it ACROSS the fixed
threshold rather than rescaling it -- so no arm's manipulation is invariant under that group:
  A: accumulates ~EXPOSURE_STEPS increments -> terrain magnitude rises above threshold.
  B: accumulates zero -> terrain stays at its zero-init, structurally below threshold.
  C: accumulates only on the few ticks the agent strays far from hazards -> below threshold.
  D: accumulates ~EXPOSURE_STEPS increments IN THE HAZARD CONTEXT -> above threshold.
The manipulation is an additive change to the very quantity the threshold is applied to: not a
broadcast constant across candidates, not a monotone rescaling of a rank-based DV, and not a
permutation of interchangeable units.

READINESS / NON-VACUITY. Five readiness-kind preconditions, all measured, all reported with
numeric measured+threshold so the indexer recomputes `met` itself. P1 is the repaired gate;
P2-P5 are unchanged from 939 and their MIN/MAX aggregation is retained DELIBERATELY, which the
autopsy's requeue_spec item 1 explicitly permits ("keep min and justify it explicitly as a
deliberately harsher gate"):
  P1 arm_A_mean_release_rate_positive_control -- arm A's MEAN release_rate, the same statistic
     AND the same aggregation as the DVs it guards, against DV_MARGIN itself. See repair (1):
     this is the exact necessary condition for DV1 and DV3, so it cannot be harsher than them.
  P2 arm_A_accumulation_occurred -- arm A num_safety_steps, MIN over cells. Min is correct and
     is not the 939 defect: this is a structural did-the-manipulation-happen-at-all check (the
     V3-EXQ-916 catch, where an entire run executed at num_safety_steps=0), not a magnitude
     guard on a mean-aggregated DV. A single cell that accumulated nothing measured nothing,
     whatever the other cells did. Realised margin is 240 vs a floor of 20, ~12x, so it is not
     fragile in the way a min-gate at the centre of the arm-A spread was.
  P3 proximity_safe_below_gate / P4 proximity_hazard_above_gate -- MAX over safe cells and MIN
     over hazard cells respectively. Worst-case is correct for the same reason: these ask
     whether the dedicated channel STRADDLES the gate in every cell, which is a structural
     property of the manipulation, not an effect size. Realised 0.0 vs 0.25 and 0.887 vs 0.25.
  P5 world_encoder_trained -- SD-070 weight-delta on split_encoder.world_encoder, a count that
     must equal the cell count.
A below-floor readiness measure self-routes `substrate_not_ready_requeue` -> non_contributory,
NEVER a substrate-verdict label. A tolerance failure self-routes the DISTINCT label
`insufficient_valid_seeds_requeue` -> non_contributory: it is a robustness failure, not a
substrate-readiness failure, and 939's autopsy is explicit that imprecise labelling of what did
and did not happen is itself a defect.

DEGENERACY WORDING (the autopsy's learning #4). 939 recorded degeneracy_reason "the mechanism
could not express itself, so nothing was measured" while the same manifest contained a clean
2x2 dissociation refuting it. The reasons emitted here state only what the gate found and
explicitly point the reader at the arm-level numbers before drawing any conclusion about the
mechanism.

MANDATORY RECORDING (carried forward from failure_autopsy_V3-EXQ-930_2026-08-16's explicit
recording-gap finding, plus the operating-point block added by repair (4)):
  * per-tick z_harm_a norm SERIES per (arm, seed) retained in full, plus per-seed mean/std so
    the PER-SEED STANDARDISATION that autopsy requires is computable post hoc. The z_harm_a
    encoder is untrained and its between-seed init offset (sd ~6.4e-02) exceeds within-run
    modulation (~1e-03) by ~60x, so an ABSOLUTE-threshold read on the raw norm cannot
    discriminate; the standardised series keeps that question open rather than re-run blind.
  * damage-exposure readout -- per-tick sum(env.limb_damage) plus a damaged-tick count, the
    manipulation check that the SD-022 damage pathway was exercised under
    limb_damage_enabled=True. 930 recorded none, so its z_harm_a null was unfalsifiable.
  * per-tick proximity-signal series, num_safety_steps / total_safety, the terrain read at
    test, and the per-arm terrain-read operating point.

SUBSTRATE. Run under the PRODUCTION scenario limb_damage_enabled=True (SD-022 damage-sourced
z_harm_a) -- the exact scenario in which z_harm_a cannot serve MECH-303's gate and the
dedicated signal must. alpha_world=0.9 for z_world fidelity (SD-008; V3-EXQ-760 precedent).
safety_terrain_bandwidth=0.03 (SD-067): the shared kernel_bandwidth 1.0 is ~15x too wide for
the z_world residual scale and saturates evaluate_safety. Confounding release paths held off at
test: urgency_interrupt_threshold=1e9 (MECH-091 abort), no goal, MECH-304 conditioned store OFF
-- so the ONLY release path in every arm is the MECH-303 contextual gate.

PHASED TRAINING. P0a SD-070 z_world encoder warmup (run_zworld_p0, 60 episodes -- the validated
operating point) on a DEDICATED warmup env; then the encoder is not stepped again -- P1 exposure
writes the terrain under torch.no_grad() (accumulate_safety), and P2 reads it read-only. No head
is trained on a moving latent target.

GAP-I / COMMITMENT WALL. beta_gate.elevate() fires only from the F-selection/commitment path
this substrate cannot sustain (603h), so the commitment is HARNESS-INDUCED each test trial (764
option i) and the DV is whether the MECH-303 gate RELEASES it. The DV therefore never requires
the substrate to SUSTAIN multi-step commitment, which is what
`feedback_dont_queue_commitment_dependent_behavioural` warns against.

SUBSTRATE-PATH OVERLAP (Step 2.5c). Three OPEN `corrupting` substrate_queue entries name files
this driver imports; each is disclosed and each is argued non-firing on this DV, not waved past:
  * mode-governance-engagement (ree_core/{cingulate/salience_coordinator.py,utils/config.py,
    agent.py}, experiments/_lib/regime_occupancy_gate.py). use_salience_coordinator defaults
    False (config.py:3200) and this driver never sets it, so agent.py:682-683 leaves
    self.salience = None and the SalienceCoordinator is never instantiated; regime_occupancy_gate
    is not imported. The defect has no live code path here.
  * MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (ree_core/agent.py::run_sws_schema_pass). Sleep.
    This driver sets no sleep flag and never enters a sleep path.
  * contextmemory-write-path-addressing-degeneracy (ree_core/predictors/e1_deep.py). The DV
    reads ResidueField.evaluate_safety and the beta gate, not ContextMemory; the defect is
    scoped to SD-017/ARC-045/MECH-166.
Open `degrading` overlaps, disclosed and non-blocking: SD-MECH303-THRESHOLD-SOURCING (the
substrate under test, status_phase validated), SD-E3-SCORER-COMPLETION
(ree_core/predictors/e3_selector.py) and SD-ORIENTING-DECISION-SCALE
(ree_core/agent.py::select_action) -- both in the selection path this driver calls. 939 did not
disclose those last two; they are recorded here.

RE-DERIVE BRAKE (Step 2.5b). Recomputed for MECH-303 at authoring time: count 1, threshold 2,
NOT braked. The single counted hit is 939's own autopsy. Worth flagging to governance: that
autopsy is classified `failure_location.net_classification = "MEASURES FAILED"`, i.e. an
instrument defect with `recommended_substrate_queue_entry.action = "none"`, but it carries
`recommended_epistemic_category: "standard"`, so the MOVE-3 counter's instrument-defect
carve-out (which keys on the CATEGORY string) does not exclude it and it counts. It is harmless
at 1 of 2. It would not be harmless if a further MECH-303 letter also landed non_contributory:
the brake would then arm against the next retest on a reading 939's own autopsy calls false.
That autopsy already instructs governance not to stamp substrate_ceiling on 939; this note
records the adjacent counter-shape hazard.
"""
import argparse
import math
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402

from _lib.arm_fingerprint import arm_cell  # noqa: E402
from _lib.capability_eval import RandomPolicy  # noqa: E402
from _lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from _lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from _lib.manifest_core import enabled_default_off_flags  # noqa: E402
from _lib.zworld_encoder_guard import (  # noqa: E402
    ZWorldEncoderUntrainedError,
    assert_world_encoder_trained,
    latent_stack_snapshot,
)
from pack_writer import write_flat_manifest  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

_ZG = ZGoalStreamAccumulator()
# stamp_recording_core() only derives `enabled_default_off_flags` from a live `agent=`,
# and this driver builds a fresh agent per cell and deliberately does NOT retain them (24
# REEAgents held to the end of the run would not fit beside serve.py on the 3.8 GB hub).
# Walking each arm's config instead costs nothing and yields the same block, which the
# stamper then leaves alone because it only fills the field when absent. Without this the
# manifest silently records no default-off-knob provenance at all -- and this experiment
# turns on four of them (use_contextual_safety_terrain and the three gate knobs), so that
# is exactly the run where governance drift-detection needs it.
_DEFAULT_OFF_FLAGS = {}

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_939a_mech303_proximity_gated_contextual_safety_vigilance_release"
ARCH_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-303"]
# Same question, same design as V3-EXQ-939 -- see the DOUBLE-COUNTING GUARD in the module
# docstring. Emitted at manifest top level so governance cannot score both as supports.
SUPERSEDES = ("v3_exq_939_mech303_proximity_gated_contextual_safety_vigilance_release"
              "_20260818T213039Z_v3")

DEVICE = torch.device("cpu")

# --- pre-registered constants -------------------------------------------------------------
N_SEEDS = 6
SEED_BASE = 1000
TEST_SEED_OFFSET = 500        # held-out test-env layout, distinct from the exposure layout
P0_EPISODES = 60              # SD-070 validated operating point
P0_STEPS_PER_EPISODE = 40
EXPOSURE_STEPS = 240          # P1 accumulation walk, per (arm, seed)
TOTAL_DENOM = P0_EPISODES + EXPOSURE_STEPS   # progress denominator M == queue episodes_per_run
N_TEST_TRIALS = 90            # was 30 in V3-EXQ-939. Repair (2): the 1/30 lattice put the
                              # readiness floor 0.34 strictly between two achievable values
                              # and made the per-seed estimate (binomial SE ~0.091 at
                              # p~0.44) fragile. 1/90 lattice, SE ~0.052, ~2% more runtime.
SETTLE_STEPS = 8              # settle the recurrent z_world into the test context

DV_MARGIN = 0.34              # every DV gap must clear this (764's registered margin)
# Repair (1): the readiness floor is DERIVED from DV_MARGIN, not declared beside it.
#   DV1 = mean(A) - mean(B), DV3 = mean(A) - mean(C), and mean(B), mean(C) >= 0,
#   so DV1 <= mean(A) and DV3 <= mean(A). mean(A) >= DV_MARGIN is therefore the EXACT
#   necessary condition for either load-bearing-relevant gap to clear the margin -- same
#   statistic (release_rate), same aggregation (mean), same threshold. The gate is provably
#   never harsher than the criteria it guards, and there is no second number to drift.
READINESS_FLOOR_MEAN = DV_MARGIN
ACCUM_STEPS_FLOOR = 20        # arm A num_safety_steps floor (the V3-EXQ-916 zero-accumulation catch)
MIN_VALID_SEEDS = 4           # repair (3): the PRE-REGISTERED 2-of-6 tolerance, now binding

# Gate-source knobs (SD-MECH303-THRESHOLD-SOURCING, ree-v3 b257e7ad14).
GATE_SOURCE = "proximity_signal"
PROX_THRESH_NATURAL = 0.25    # the SHIPPED default -- deliberately not retuned here
PROX_THRESH_FORCED_CLOSED = 0.0   # signal is clipped to [0,1] and >= 0 -> gate never opens
PROX_THRESH_FORCED_OPEN = 2.0     # signal is clipped to [0,1] -> gate always open

CONTEXTUAL_ACCUM_WEIGHT = 0.05
CONTEXTUAL_RELEASE_THRESHOLD = 0.5   # 764's calibrated value, between the safe and unsafe reads
SAFETY_TERRAIN_BW = 0.03             # SD-067 dedicated (tighter) RBF bandwidth

GRID_SIZE = 10
NUM_RESOURCES = 3
NUM_HAZARDS_SAFE = 0
NUM_HAZARDS_HAZARD = 8
HARM_OBS_A_DIM = 7            # harm_obs_a is 7-dim under limb_damage_enabled=True (SD-022)

_SUBSTRATE_COMMON = dict(
    alpha_world=0.9,                      # SD-008 z_world fidelity (760 precedent)
    use_harm_stream=True, harm_obs_dim=51,
    use_affective_harm_stream=True, harm_obs_a_dim=HARM_OBS_A_DIM,
)

ARMS = [
    "A_safe_gate_natural",
    "B_safe_gate_forced_closed",
    "C_hazard_gate_natural",
    "D_hazard_gate_forced_open",
]
_ARM_SPEC = {
    # arm -> (num_hazards, proximity_threshold)
    "A_safe_gate_natural":       (NUM_HAZARDS_SAFE,   PROX_THRESH_NATURAL),
    "B_safe_gate_forced_closed": (NUM_HAZARDS_SAFE,   PROX_THRESH_FORCED_CLOSED),
    "C_hazard_gate_natural":     (NUM_HAZARDS_HAZARD, PROX_THRESH_NATURAL),
    "D_hazard_gate_forced_open": (NUM_HAZARDS_HAZARD, PROX_THRESH_FORCED_OPEN),
}
SAFE_ARMS = ("A_safe_gate_natural", "B_safe_gate_forced_closed")
HAZARD_ARMS = ("C_hazard_gate_natural", "D_hazard_gate_forced_open")


def _per_seed_floor(n_trials=None):
    """Repair (2). Largest achievable release_rate that does NOT EXCEED DV_MARGIN, on the
    live 1/n lattice.

    release_rate is a count of crossings out of n trials, so it only ever takes values k/n.
    A floor that is not itself a lattice point silently rounds UP to the next one:
    V3-EXQ-939's 0.34 against n=30 had an undeclared effective floor of 11/30 = 0.36667, a
    7.8% tightening, and seed 1 missed it by 0.2 of a single trial.

    Two properties matter and neither is incidental. It is a FUNCTION of the live
    N_TEST_TRIALS rather than a module constant, so it re-quantises if the trial count is
    ever changed again -- including under --dry-run, where a constant computed against 90
    would be off-lattice against 6. And it uses floor(), not ceil(): the autopsy's headline
    finding is that a readiness guard must never be harsher than the criterion it guards, so
    the rounding direction that can only ever RELAX relative to DV_MARGIN is the correct one.
    The autopsy offered the round-UP alternative (11/30); it is deliberately rejected.
    """
    n = int(n_trials if n_trials is not None else N_TEST_TRIALS)
    return math.floor(DV_MARGIN * n) / n


def _env_kwargs(num_hazards):
    return dict(
        size=GRID_SIZE, num_hazards=num_hazards, num_resources=NUM_RESOURCES,
        use_proxy_fields=True,
        limb_damage_enabled=True, heal_rate=0.4,     # production scenario (SD-022 damage-sourced z_harm_a)
        safety_proximity_signal_enabled=True,        # emit the dedicated MECH-303 channel
    )


def _sense(agent, obs):
    """Forward the dedicated proximity channel into sense(). Without obs_safety_proximity the
    proximity gate simply never fires (agent.py has NO silent fallback to z_harm_a, by
    design), so this argument is load-bearing, not cosmetic."""
    sp = obs.get("safety_proximity_harm")
    return agent.sense(
        obs["body_state"], obs["world_state"],
        obs_harm=obs.get("harm_obs"), obs_harm_a=obs.get("harm_obs_a"),
        obs_safety_proximity=float(sp.item()) if sp is not None else None,
    )


def _act(agent, latent):
    ticks = agent.clock.advance()
    world_dim = agent.config.latent.world_dim
    e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick") else torch.zeros(1, world_dim, device=DEVICE)
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    return int(action.argmax(dim=-1).item())


def _prox(obs):
    sp = obs.get("safety_proximity_harm")
    return float(sp.item()) if sp is not None else float("nan")


def _damage(env):
    return float(np.sum(env.limb_damage)) if getattr(env, "limb_damage_enabled", False) else 0.0


def _config_slice(arm):
    num_hazards, prox_thresh = _ARM_SPEC[arm]
    return {
        "arm": arm,
        "env_kwargs": _env_kwargs(num_hazards),
        # Declared explicitly even though env_kwargs already binds size/num_resources: this
        # fingerprint is CROSS-DRIVER reusable (include_driver_script_in_hash=False), so an
        # under-declared slice is a false-cache-HIT bug, not just a false MISS
        # (arm_reuse_fingerprint_plan.md 7b). SEED_BASE/TEST_SEED_OFFSET pick the exposure and
        # held-out test layouts and are otherwise invisible to the slice.
        "grid_size": GRID_SIZE,
        "num_resources": NUM_RESOURCES,
        "seed_base": SEED_BASE,
        "test_seed_offset": TEST_SEED_OFFSET,
        "substrate_common": dict(_SUBSTRATE_COMMON),
        "use_contextual_safety_terrain": True,
        "contextual_safety_gate_source": GATE_SOURCE,
        "contextual_safety_proximity_threshold": prox_thresh,
        "contextual_safety_accum_weight": CONTEXTUAL_ACCUM_WEIGHT,
        "contextual_safety_release_threshold": CONTEXTUAL_RELEASE_THRESHOLD,
        "safety_terrain_bandwidth": SAFETY_TERRAIN_BW,
        "urgency_interrupt_threshold": 1e9,
        "beta_gate_bistable": True,
        "p0_episodes": P0_EPISODES,
        "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
        "exposure_steps": EXPOSURE_STEPS,
        "n_test_trials": N_TEST_TRIALS,
        "settle_steps": SETTLE_STEPS,
    }


def _build(arm, seed):
    num_hazards, prox_thresh = _ARM_SPEC[arm]
    env = CausalGridWorldV2(seed=SEED_BASE + seed, **_env_kwargs(num_hazards))
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim, action_dim=5,
        use_contextual_safety_terrain=True,
        contextual_safety_gate_source=GATE_SOURCE,
        contextual_safety_proximity_threshold=prox_thresh,
        contextual_safety_accum_weight=CONTEXTUAL_ACCUM_WEIGHT,
        contextual_safety_release_threshold=CONTEXTUAL_RELEASE_THRESHOLD,
        safety_terrain_bandwidth=SAFETY_TERRAIN_BW,
        **_SUBSTRATE_COMMON,
    )
    # from_dims round-trip is verified at runtime below (from_dims silently swallows unknown
    # kwargs, so a renamed knob would otherwise leave the gate at its default -- the
    # reference-reeconfig-from-dims-silent-kwargs hazard).
    for _k, _want in (
        ("use_contextual_safety_terrain", True),
        ("contextual_safety_gate_source", GATE_SOURCE),
        ("contextual_safety_proximity_threshold", prox_thresh),
        ("contextual_safety_accum_weight", CONTEXTUAL_ACCUM_WEIGHT),
        ("contextual_safety_release_threshold", CONTEXTUAL_RELEASE_THRESHOLD),
    ):
        _got = getattr(cfg, _k, None)
        if _got != _want:
            raise RuntimeError(
                "REEConfig.from_dims did not take %s (got %r, want %r) -- the MECH-303 gate "
                "would run at its default. Do not proceed." % (_k, _got, _want)
            )
    cfg.heartbeat.beta_gate_bistable = True
    cfg.e3.urgency_interrupt_threshold = 1e9   # disable MECH-091 abort -> isolate the safety gate
    agent = REEAgent(cfg)
    return env, agent


def _p0_warmup(agent, arm, seed):
    """P0a: SD-070 z_world encoder warmup on a DEDICATED env (the rollout consumes env RNG,
    so reusing the exposure env would shift the layout sequence P1 then sees)."""
    num_hazards, _ = _ARM_SPEC[arm]
    warm_env = CausalGridWorldV2(seed=SEED_BASE + seed, **_env_kwargs(num_hazards))
    before = latent_stack_snapshot(agent)
    stats = run_zworld_p0(
        agent, warm_env, seed, P0_EPISODES, P0_STEPS_PER_EPISODE,
        policy=RandomPolicy(seed), label=f"exq939 arm={arm}", dry_run=_DRY_RUN,
    )
    guard = {"guard_checked": False}
    encoder_trained = False
    try:
        guard = assert_world_encoder_trained(
            agent, before, p0=P0_EPISODES, strict=True, context=f"exq939 arm={arm} seed={seed}",
        )
        encoder_trained = True
    except ZWorldEncoderUntrainedError as exc:
        # Do NOT crash: an untrained encoder is a substrate-readiness state, and the correct
        # route for it is substrate_not_ready_requeue (a recorded non_contributory), never an
        # ERROR that looks like a code bug or a FAIL that looks like a claim verdict.
        guard = {"guard_checked": True, "error": str(exc)[:400]}
        print(f"  [P0a-UNTRAINED] arm={arm} seed={seed}: {exc}", flush=True)
    stats = dict(stats)
    stats["encoder_trained"] = bool(encoder_trained)
    stats["weight_delta_guard"] = guard
    return stats


def _expose(env, agent, arm, seed):
    """P1: random-walk the exposure context. accumulate_safety runs inside sense(), gated on
    the DEDICATED proximity channel. Records the per-tick series the 930 autopsy found
    missing (z_harm_a norm, proximity, limb damage)."""
    agent.reset()
    _, obs = env.reset()
    zharm_a, prox, damage = [], [], []
    for step in range(EXPOSURE_STEPS):
        prox.append(_prox(obs))
        damage.append(_damage(env))
        latent = _sense(agent, obs)
        zharm_a.append(
            float(latent.z_harm_a.detach().norm().item()) if latent.z_harm_a is not None else float("nan")
        )
        _ = _act(agent, latent)
        _flat, _harm, done, _info, obs = env.step(random.randint(0, 4))
        if done:
            _, obs = env.reset()
        if (step + 1) % 50 == 0 or (step + 1) == EXPOSURE_STEPS:
            print(f"  [train] arm={arm} seed={seed} phase=P1 "
                  f"ep {P0_EPISODES + step + 1}/{TOTAL_DENOM}", flush=True)
    return {"zharm_a": zharm_a, "prox": prox, "damage": damage}


def _test_release(agent, test_env):
    """P2: induced-commitment release readout in the arm's OWN context. agent.reset() clears
    the recurrent latent so the test-context z_world is clean; the safety terrain lives in
    the residue field and SURVIVES the reset (documented invariant). The only release path
    live here is the MECH-303 contextual gate (urgency interrupt off, no goal, store off)."""
    agent.reset()
    _, obs = test_env.reset()
    for _ in range(SETTLE_STEPS):
        _sense(agent, obs)
        _flat, _harm, done, _info, obs = test_env.step(random.randint(0, 4))
        if done:
            _, obs = test_env.reset()
    released = 0
    preds, zharm_a, prox, damage = [], [], [], []
    for _t in range(N_TEST_TRIALS):
        prox.append(_prox(obs))
        damage.append(_damage(test_env))
        latent = _sense(agent, obs)
        zharm_a.append(
            float(latent.z_harm_a.detach().norm().item()) if latent.z_harm_a is not None else float("nan")
        )
        if latent.z_world is not None and hasattr(agent.residue_field, "evaluate_safety"):
            preds.append(float(agent.residue_field.evaluate_safety(latent.z_world.detach()).mean().detach()))
        agent.beta_gate.elevate()
        was = agent.beta_gate.is_elevated
        _ = _act(agent, latent)
        if was and not agent.beta_gate.is_elevated:
            released += 1
        _flat, _harm, done, _info, obs = test_env.step(random.randint(0, 4))
        if done:
            _, obs = test_env.reset()
    return {
        "release_rate": released / max(N_TEST_TRIALS, 1),
        "n_released": released,
        "n_test_trials": N_TEST_TRIALS,
        "mean_contextual_safety_pred": float(np.mean(preds)) if preds else 0.0,
        "series": {"zharm_a": zharm_a, "prox": prox, "damage": damage},
    }


def _series_stats(vals):
    """Per-seed summary of a per-tick series. `std` is what makes the PER-SEED
    STANDARDISATION the 930 autopsy requires computable post hoc from the recorded series."""
    clean = [v for v in vals if v is not None and np.isfinite(v)]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def _run_cell(arm, seed):
    with arm_cell(seed, config_slice=_config_slice(arm), script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        env, agent = _build(arm, seed)
        try:
            _DEFAULT_OFF_FLAGS.update(enabled_default_off_flags(agent.config) or {})
        except Exception as _exc:   # recording nicety -- never kill a 10-minute run for it
            _DEFAULT_OFF_FLAGS.setdefault("_error", str(_exc)[:200])
        p0 = _p0_warmup(agent, arm, seed)
        expo = _expose(env, agent, arm, seed)
        rf = agent.residue_field
        num_safety_steps = int(getattr(rf, "num_safety_steps", 0))
        total_safety = float(getattr(rf, "total_safety", 0.0))
        num_hazards, prox_thresh = _ARM_SPEC[arm]
        test_env = CausalGridWorldV2(seed=SEED_BASE + TEST_SEED_OFFSET + seed,
                                     **_env_kwargs(num_hazards))
        test = _test_release(agent, test_env)
        _ZG.observe(agent)

        all_zharm = list(expo["zharm_a"]) + list(test["series"]["zharm_a"])
        all_prox = list(expo["prox"]) + list(test["series"]["prox"])
        all_damage = list(expo["damage"]) + list(test["series"]["damage"])
        damaged_ticks = sum(1 for d in all_damage if d > 1e-9)
        row = {
            "arm": arm,
            "seed": seed,
            "num_hazards": num_hazards,
            "proximity_threshold": prox_thresh,
            "release_rate": test["release_rate"],
            "n_released": test["n_released"],
            "mean_contextual_safety_pred": test["mean_contextual_safety_pred"],
            "num_safety_steps": num_safety_steps,
            "total_safety": total_safety,
            "accumulation_frac": num_safety_steps / max(EXPOSURE_STEPS, 1),
            # --- 930 autopsy recording gap, closed -------------------------------------
            "zharm_a_stats": _series_stats(all_zharm),
            "proximity_stats": _series_stats(all_prox),
            "damage_stats": _series_stats(all_damage),
            "damaged_tick_count": damaged_ticks,
            "damaged_tick_frac": damaged_ticks / max(len(all_damage), 1),
            "series": {                      # retained in full -- see module docstring
                "zharm_a_norm": [float(v) for v in all_zharm],
                "safety_proximity_harm": [float(v) for v in all_prox],
                "limb_damage_sum": [float(v) for v in all_damage],
            },
            "p0a": p0,
            "encoder_trained": bool(p0.get("encoder_trained")),
        }
        cell.stamp(row)
    return row


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(statistics.fmean(vals)) if vals else 0.0


def _operating_point(rows):
    """Repair (4), NON-GATING. Where the accumulated terrain read actually sat relative to the
    FIXED release threshold, per arm.

    release_rate is a threshold-crossing COUNT, so on its own it cannot say whether a run was
    comfortably clear of the threshold or balanced on it -- which is exactly why V3-EXQ-939's
    prose calibration figure (A 0.900) and its realised value (A 0.4667) could differ by ~2x
    with nothing in the manifest to explain it. The margin does say: 939's arm-A read sat at
    0.578 against a 0.5 threshold, a margin of 0.078, so ~44% of trials crossed rather than
    ~90%. Recording it makes any future calibration-vs-run discrepancy diagnosable from the
    manifest alone instead of requiring a re-run to investigate.

    Deliberately NOT a precondition. It is a PROXY for the statistic the DVs route on, and
    gating on a proxy is the V3-EXQ-643 defect; its realised margin is also thin enough that a
    gate here would have become a second way to void a sound run -- the failure this experiment
    exists to fix. Emitted on PASS runs too: a diagnostic that only appears when something
    already looks wrong cannot establish that anything was ever right.
    """
    preds = [r.get("mean_contextual_safety_pred") for r in rows]
    preds = [float(p) for p in preds if p is not None]
    if not preds:
        return None
    mean_pred = float(statistics.fmean(preds))
    return {
        "n_cells": len(preds),
        "terrain_read_mean": mean_pred,
        "terrain_read_min": float(min(preds)),
        "terrain_read_max": float(max(preds)),
        "release_threshold": CONTEXTUAL_RELEASE_THRESHOLD,
        "margin_over_release_threshold_mean": mean_pred - CONTEXTUAL_RELEASE_THRESHOLD,
        "margin_over_release_threshold_min": float(min(preds)) - CONTEXTUAL_RELEASE_THRESHOLD,
    }


def _worst_cell(rows, key):
    """Extremum + the offending cell id, so a `measured` recomputes exactly against an
    all(...) style `met` instead of being masked by an in-band mean."""
    cand = [(r.get(key), r) for r in rows if r.get(key) is not None]
    if not cand:
        return None, None
    val, r = min(cand, key=lambda t: t[0])
    return float(val), f"{r['arm']}::seed{r['seed']}"


def run_experiment(seeds):
    t_start = datetime.utcnow()
    per_cell = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = _run_cell(arm, seed)
            print(f"  [{arm} seed={seed}] release_rate={row['release_rate']:.2f} "
                  f"num_safety_steps={row['num_safety_steps']} "
                  f"pred={row['mean_contextual_safety_pred']:.3f} "
                  f"prox_mean={(row['proximity_stats']['mean'] or 0.0):.3f}", flush=True)
            print("verdict: PASS", flush=True)   # per-cell run-completion marker
            per_cell.append(row)

    def rows(arm):
        return [r for r in per_cell if r["arm"] == arm]

    rel = {a: _mean([r["release_rate"] for r in rows(a)]) for a in ARMS}
    dv1 = rel["A_safe_gate_natural"] - rel["B_safe_gate_forced_closed"]
    dv2 = rel["D_hazard_gate_forced_open"] - rel["C_hazard_gate_natural"]
    dv3 = rel["A_safe_gate_natural"] - rel["C_hazard_gate_natural"]

    # ---- readiness preconditions (all numeric -> the indexer recomputes `met`) ----------
    a_rows = rows("A_safe_gate_natural")
    safe_rows = [r for r in per_cell if r["arm"] in SAFE_ARMS]
    hazard_rows = [r for r in per_cell if r["arm"] in HAZARD_ARMS]
    a_mean_release = _mean([r["release_rate"] for r in a_rows])
    a_worst_release, a_worst_release_cell = _worst_cell(a_rows, "release_rate")
    a_worst_accum, a_worst_accum_cell = _worst_cell(a_rows, "num_safety_steps")
    per_seed_floor = _per_seed_floor()
    safe_prox_max = max([(r["proximity_stats"]["mean"] or 0.0) for r in safe_rows] or [0.0])
    hazard_prox_min = min([(r["proximity_stats"]["mean"] or 0.0) for r in hazard_rows] or [0.0])
    n_encoder_trained = sum(1 for r in per_cell if r.get("encoder_trained"))

    preconditions = [
        {"name": "arm_A_mean_release_rate_positive_control", "kind": "readiness",
         "description": ("arm A's MEAN release_rate -- the same statistic AND the same "
                         "aggregation as the DVs this gate guards, against DV_MARGIN itself"),
         "control": (
             "arm A (safe context, gate at the shipped default) is the best case by "
             "construction. AGGREGATION IS THE MEAN, DELIBERATELY, and `met` is a "
             "central-tendency comparison rather than a worst-case claim -- so `measured` is "
             "the mean and recomputes exactly. DV1 = mean(A) - mean(B) and DV3 = mean(A) - "
             "mean(C) with mean(B), mean(C) >= 0, hence DV1 <= mean(A) and DV3 <= mean(A): "
             "mean(A) >= DV_MARGIN is the EXACT necessary condition for either to clear, so "
             "this gate cannot be harsher than the criteria it guards. V3-EXQ-939 aggregated "
             "this same statistic by MIN and voided a run whose mean cleared the floor by "
             "0.099 because one of six seeds sat 0.0067 under it. Per-seed robustness is not "
             "abandoned -- it moved to the pre-registered MIN_VALID_SEEDS tolerance below, "
             "which for the first time can actually bind."),
         "measured": float(a_mean_release),
         "threshold": READINESS_FLOOR_MEAN, "direction": "lower",
         "worst_cell": a_worst_release_cell,
         "worst_cell_release_rate": a_worst_release,
         "met": bool(a_rows) and a_mean_release >= READINESS_FLOOR_MEAN},
        {"name": "arm_A_accumulation_occurred", "kind": "readiness",
         "description": "arm A num_safety_steps -- the V3-EXQ-916 zero-accumulation catch",
         "control": ("arm A's proximity gate is open by construction in a hazard-free context. "
                     "MIN aggregation is RETAINED here deliberately (requeue_spec item 1 "
                     "permits a justified min): this is a structural did-the-manipulation-"
                     "happen-at-all check -- the V3-EXQ-916 catch, where an entire run "
                     "executed at num_safety_steps=0 -- not a magnitude guard on a "
                     "mean-aggregated DV. A cell that accumulated nothing measured nothing, "
                     "whatever the other cells did. Realised margin is ~12x (240 vs 20), so it "
                     "is not fragile the way a min-gate at the centre of the arm-A release "
                     "spread was."),
         "measured": float(a_worst_accum) if a_worst_accum is not None else 0.0,
         "threshold": float(ACCUM_STEPS_FLOOR), "direction": "lower",
         "offending_cell": a_worst_accum_cell,
         "met": all(r["num_safety_steps"] >= ACCUM_STEPS_FLOOR for r in a_rows) if a_rows else False},
        {"name": "proximity_safe_below_gate", "kind": "readiness",
         "description": "realized proximity mean in SAFE contexts must sit below the gate threshold",
         "control": ("num_hazards=0 emits a hazard-proximity EMA of ~0 by construction. MAX "
                     "over safe cells (worst case) is RETAINED deliberately: this asks whether "
                     "the dedicated channel straddles the gate in EVERY cell, a structural "
                     "property of the manipulation rather than an effect size."),
         "measured": float(safe_prox_max), "threshold": PROX_THRESH_NATURAL, "direction": "upper",
         "met": bool(safe_prox_max < PROX_THRESH_NATURAL)},
        {"name": "proximity_hazard_above_gate", "kind": "readiness",
         "description": "realized proximity mean in HAZARD contexts must sit above the gate threshold",
         "control": ("num_hazards=8 emits an elevated hazard-proximity EMA by construction. "
                     "MIN over hazard cells (worst case) is RETAINED deliberately, for the "
                     "same structural reason as the safe-side check above."),
         "measured": float(hazard_prox_min), "threshold": PROX_THRESH_NATURAL, "direction": "lower",
         "met": bool(hazard_prox_min > PROX_THRESH_NATURAL)},
        {"name": "world_encoder_trained", "kind": "readiness",
         "description": "SD-070 P0a warmup moved split_encoder.world_encoder in every cell",
         "control": "run_zworld_p0 weight-delta guard on a matched-distribution warmup env",
         "measured": float(n_encoder_trained), "threshold": float(len(per_cell)), "direction": "lower",
         "met": n_encoder_trained == len(per_cell)},
    ]
    readiness_ok = all(bool(p["met"]) for p in preconditions)
    # Repair (3). With the readiness gate now on the MEAN, this pre-registered 2-of-6 tolerance
    # is the only per-seed guard left and is therefore the binding constraint in a reachable
    # region: three arm-A seeds at 0.30 and three at 0.55 gives mean 0.425 >= 0.34 (readiness
    # passes) with only 3 of 6 seeds clearing the per-seed floor -- 3 < 4, tolerance blocks. In
    # V3-EXQ-939 readiness_ok already demanded every cell clear the same floor, so this could
    # never bind and the declared tolerance was vetoed by a zero-tolerance conjunction.
    n_valid_seeds = len({r["seed"] for r in a_rows if r["release_rate"] >= per_seed_floor})

    dv1_pass = dv1 >= DV_MARGIN
    dv2_pass = dv2 >= DV_MARGIN
    dv3_pass = dv3 >= DV_MARGIN
    overall_pass = bool(readiness_ok and n_valid_seeds >= MIN_VALID_SEEDS
                        and dv1_pass and dv2_pass and dv3_pass)
    tolerance_ok = n_valid_seeds >= MIN_VALID_SEEDS

    # Non-degeneracy: a DV whose two arms are bit-identical across every seed measured nothing.
    def spread(arm_a, arm_b):
        pair = [r["release_rate"] for r in rows(arm_a)] + [r["release_rate"] for r in rows(arm_b)]
        return float(max(pair) - min(pair)) if pair else 0.0

    criteria_non_degenerate = {
        "DV1_accumulation_necessity": spread("A_safe_gate_natural", "B_safe_gate_forced_closed") > 0.0,
        "DV2_accumulation_is_the_cause": spread("D_hazard_gate_forced_open", "C_hazard_gate_natural") > 0.0,
        "DV3_gate_natural_context_appropriate": spread("A_safe_gate_natural", "C_hazard_gate_natural") > 0.0,
    }
    non_degenerate = bool(readiness_ok and tolerance_ok and any(criteria_non_degenerate.values()))
    degeneracy_reason = None
    # Wording discharges the autopsy's learning #4: V3-EXQ-939 recorded "the mechanism could not
    # express itself, so nothing was measured" while the SAME manifest carried a clean 2x2
    # dissociation refuting it. These reasons state only what the guard found, and point the
    # reader at the arm-level numbers before any claim about the mechanism is drawn.
    if not readiness_ok:
        unmet = [p["name"] for p in preconditions if not p["met"]]
        degeneracy_reason = (
            "readiness gate unmet: %s. This states only that a pre-registered readiness "
            "condition did not hold; it does NOT assert that the mechanism failed to express "
            "itself. Read release_rate_by_arm, per_seed_release_rate, criteria and "
            "custom_information.terrain_read_operating_point in this manifest before drawing "
            "that conclusion." % (", ".join(unmet),))
    elif not tolerance_ok:
        degeneracy_reason = (
            "per-seed robustness tolerance unmet: only %d of %d arm-A seeds cleared the "
            "on-lattice per-seed floor %.5f (>= %d required). The arm means in this manifest "
            "ARE measured and the readiness gate passed; they are not adjudicated here because "
            "the effect is not robust across seeds."
            % (n_valid_seeds, len(a_rows), per_seed_floor, MIN_VALID_SEEDS))
    elif not any(criteria_non_degenerate.values()):
        degeneracy_reason = "every DV's arm pair is bit-identical across all seeds -- no contrast measured"

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
    elif not tolerance_ok:
        # A DISTINCT label: a robustness failure is not a substrate-readiness failure, and the
        # 939 autopsy is explicit that imprecise labelling of what did and did not happen is
        # itself a defect. Both route non_contributory.
        label = "insufficient_valid_seeds_requeue"
        direction = "non_contributory"
    elif overall_pass:
        label = "mech303_proximity_gated_accumulation_lowers_background_vigilance"
        direction = "supports"
    elif dv3_pass and not dv2_pass:
        label = "mech303_context_difference_not_attributable_to_accumulation"
        direction = "mixed"
    else:
        label = "mech303_accumulation_does_not_lower_background_vigilance"
        direction = "weakens"

    manifest = {
        "experiment_type": EXPERIMENT_TYPE,
        "enabled_default_off_flags": dict(_DEFAULT_OFF_FLAGS),
        "architecture_epoch": ARCH_EPOCH,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": t_start.strftime("%Y%m%dT%H%M%SZ"),
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": direction,
        "release_rate_by_arm": rel,
        "dv1_accumulation_necessity_gap": dv1,
        "dv2_accumulation_is_the_cause_gap": dv2,
        "dv3_gate_natural_context_gap": dv3,
        "dv_margin": DV_MARGIN,
        "n_valid_seeds": n_valid_seeds,
        "n_seeds": len(a_rows),
        "per_seed_release_rate_floor": per_seed_floor,
        "readiness_floor_mean": READINESS_FLOOR_MEAN,
        "arm_A_mean_release_rate": float(a_mean_release),
        "combination_rule": (
            "PASS = readiness_ok AND n_valid_seeds >= %d AND (DV1 >= %.2f) AND (DV2 >= %.2f) "
            "AND (DV3 >= %.2f) -- a plain conjunction of three gaps. DV2 and DV3 are the "
            "load-bearing criteria; DV1 is a config-forced ablation and is reported but is "
            "NOT load-bearing. readiness_ok gates arm A's MEAN release_rate against DV_MARGIN "
            "itself (the exact necessary condition for DV1 and DV3, so it can never be harsher "
            "than them); n_valid_seeds counts arm-A seeds clearing the on-lattice per-seed "
            "floor %.5f = floor(%.2f * %d)/%d, which is the pre-registered 2-of-%d tolerance "
            "and, unlike in V3-EXQ-939, can actually bind."
            % (MIN_VALID_SEEDS, DV_MARGIN, DV_MARGIN, DV_MARGIN,
               per_seed_floor, DV_MARGIN, N_TEST_TRIALS, N_TEST_TRIALS, len(a_rows))
        ),
        "criteria": [
            {"name": "DV1_accumulation_necessity", "load_bearing": False, "passed": bool(dv1_pass),
             "gap": dv1, "margin": DV_MARGIN},
            {"name": "DV2_accumulation_is_the_cause", "load_bearing": True, "passed": bool(dv2_pass),
             "gap": dv2, "margin": DV_MARGIN},
            {"name": "DV3_gate_natural_context_appropriate", "load_bearing": True, "passed": bool(dv3_pass),
             "gap": dv3, "margin": DV_MARGIN},
        ],
        "criteria_non_degenerate": criteria_non_degenerate,
        "non_degenerate": bool(non_degenerate),
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "arm_results": per_cell,
        "per_seed_release_rate": {a: [r["release_rate"] for r in rows(a)] for a in ARMS},
        "per_seed_num_safety_steps": {a: [r["num_safety_steps"] for r in rows(a)] for a in ARMS},
        "custom_information": {
            "gate_source": GATE_SOURCE,
            # Repair (4) -- see _operating_point()'s docstring. NON-GATING by design.
            "terrain_read_operating_point": {a: _operating_point(rows(a)) for a in ARMS},
            "terrain_read_operating_point_note": (
                "Where each arm's accumulated terrain read sat relative to the FIXED "
                "contextual_safety_release_threshold. release_rate is a crossing COUNT and "
                "cannot distinguish 'comfortably clear' from 'balanced on the threshold', which "
                "is why V3-EXQ-939's prose calibration figure (A 0.900) and its realised value "
                "(A 0.4667) differed ~2x with nothing in the manifest to explain it. Recorded "
                "so a future calibration-vs-run discrepancy is diagnosable from the manifest "
                "alone. NOT a gate: it is a proxy for the statistic the DVs route on, and "
                "gating on a proxy is the V3-EXQ-643 defect."
            ),
            "readiness_gate_repair_note": (
                "V3-EXQ-939's readiness control aggregated arm A's release_rate by MIN while "
                "the DVs aggregate by MEAN, used a floor (0.34) off the achievable 1/30 "
                "lattice (effective floor 11/30 = 0.36667), and declared a MIN_VALID_SEEDS "
                "tolerance that a zero-tolerance conjunction vetoed. Here the mean gate is "
                "DV_MARGIN itself (provably not harsher than the DVs), the per-seed floor is "
                "computed on the live 1/N_TEST_TRIALS lattice by floor() so it can never be "
                "stricter than DV_MARGIN, and the pre-registered 2-of-6 tolerance binds."
            ),
            "calibration_provenance_note": (
                "V3-EXQ-939's queue-entry calibration (A 0.900 / C 0.000 / D 1.000, terrain "
                "1.235 / 0.071 / 2.099) is NOT reproducible from its committed script at its "
                "committed constants on EITHER machine class: re-running that driver on "
                "darwin-arm64 / torch 2.12.0 at seed 0 reproduced the ree-cloud-2 "
                "(linux-x86_64 / torch 2.12.0+cpu) run exactly on every discrete readout "
                "(release_rate 0.4666666666666667 / 0.0 / 0.0 / 0.43333333333333335, "
                "num_safety_steps 240/0/6/240, total_safety 12.000027656555176) and to ~7 "
                "significant figures on the float32 terrain read. Cross-machine-class "
                "divergence is therefore REFUTED as the explanation -- and structurally so, "
                "since the agent's selected action is discarded (the env is stepped with an "
                "independent random.randint draw), so the non-portable torch.multinomial draw "
                "in e3_selector cannot reach the trajectory. The calibration described a "
                "configuration that did not ship and whose provenance was never recorded."
            ),
            "proximity_threshold_by_arm": {a: _ARM_SPEC[a][1] for a in ARMS},
            "zharm_a_recording_note": (
                "Per-tick z_harm_a norm series retained per cell under arm_results[].series. "
                "The encoder is UNTRAINED and its between-seed init offset (sd ~6.4e-02) "
                "exceeds within-run modulation (~1e-03) by ~60x, so an ABSOLUTE-threshold read "
                "on the raw norm cannot discriminate -- standardise PER SEED using "
                "arm_results[].zharm_a_stats.{mean,std} before any discrimination analysis. "
                "Recorded per the explicit recording-gap finding in "
                "failure_autopsy_V3-EXQ-930_2026-08-16."
            ),
            "damage_recording_note": (
                "Per-tick sum(limb_damage) retained per cell; damaged_tick_frac is the "
                "manipulation check that the SD-022 damage pathway was exercised under "
                "limb_damage_enabled=True. V3-EXQ-930 recorded none, which is why its "
                "damage-sourced z_harm_a null was unfalsifiable."
            ),
            "seed_44_avoided": True,
        },
    }
    if degeneracy_reason:
        manifest["degeneracy_reason"] = degeneracy_reason
    return manifest, overall_pass


_DRY_RUN = False


def build_and_run():
    global _DRY_RUN, EXPOSURE_STEPS, N_TEST_TRIALS, P0_EPISODES, TOTAL_DENOM, MIN_VALID_SEEDS
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()
    _DRY_RUN = bool(args.dry_run)

    if args.dry_run:
        EXPOSURE_STEPS = 30
        N_TEST_TRIALS = 6
        P0_EPISODES = 3
        TOTAL_DENOM = P0_EPISODES + EXPOSURE_STEPS
        MIN_VALID_SEEDS = 1

    seeds = list(range(2 if args.dry_run else args.seeds))
    random.seed(SEED_BASE)
    torch.manual_seed(SEED_BASE)
    np.random.seed(SEED_BASE)

    manifest, _overall = run_experiment(seeds)
    ts = manifest["timestamp_utc"]
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest["run_id"] = run_id

    full_config = {
        "arms": ARMS,
        "arm_spec": {a: {"num_hazards": _ARM_SPEC[a][0], "proximity_threshold": _ARM_SPEC[a][1]}
                     for a in ARMS},
        "env_kwargs_safe": _env_kwargs(NUM_HAZARDS_SAFE),
        "env_kwargs_hazard": _env_kwargs(NUM_HAZARDS_HAZARD),
        "substrate_common": dict(_SUBSTRATE_COMMON),
        "gate_source": GATE_SOURCE,
        "proximity_threshold_natural": PROX_THRESH_NATURAL,
        "proximity_threshold_forced_closed": PROX_THRESH_FORCED_CLOSED,
        "proximity_threshold_forced_open": PROX_THRESH_FORCED_OPEN,
        "contextual_safety_accum_weight": CONTEXTUAL_ACCUM_WEIGHT,
        "contextual_safety_release_threshold": CONTEXTUAL_RELEASE_THRESHOLD,
        "safety_terrain_bandwidth": SAFETY_TERRAIN_BW,
        "p0_episodes": P0_EPISODES,
        "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
        "exposure_steps": EXPOSURE_STEPS,
        "n_test_trials": N_TEST_TRIALS,
        "settle_steps": SETTLE_STEPS,
        "dv_margin": DV_MARGIN,
        "readiness_floor_mean": READINESS_FLOOR_MEAN,
        "per_seed_release_rate_floor": _per_seed_floor(),
        "accum_steps_floor": ACCUM_STEPS_FLOOR,
        "min_valid_seeds": MIN_VALID_SEEDS,
        "seed_base": SEED_BASE,
        "test_seed_offset": TEST_SEED_OFFSET,
    }

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"outcome={manifest['outcome']} evidence_direction={manifest['evidence_direction']} "
          f"label={manifest['interpretation']['label']} "
          f"dv1={manifest['dv1_accumulation_necessity_gap']:.3f} "
          f"dv2={manifest['dv2_accumulation_is_the_cause_gap']:.3f} "
          f"dv3={manifest['dv3_gate_natural_context_gap']:.3f}", flush=True)
    print(f"manifest={out_path}", flush=True)
    return manifest, out_path, run_id, args.dry_run


if __name__ == "__main__":
    _manifest, _out_path, _run_id, _dry = build_and_run()
    emit_outcome(
        outcome=_manifest["outcome"] if _manifest["outcome"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        dry_run=_dry,
    )
