"""
H3 leg (GOV-FANOUT-1, algorithm axis): does the V3-EXQ-861e seed-271 HIGH-arm MEL collapse survive on 861c's substrate? 861e protocol (CALIB_DRAWS=10 + R3) pinned to f810969

GOV-FANOUT-1 discrimination leg. Portfolio question (hypothesis_space_registry
qid): inv050_mech180_861e_producer_vs_intervention_isolation
Confirmed source autopsy: failure_autopsy_V3-EXQ-861e_2026-08-21
(status confirmed, Step 7c CONTESTED -> Step 8 interactive gate
"adopt_redteam_portfolio"; frozen hypothesis set H1 + H3, H2 a labelled
follow-on, count 2).

Hypothesis under test: H3   Design axis: algorithm
Substrate pinned to: f810969
Predecessors compared against: v3_exq_861e_inv050_mech180_calibration_power_raised_replication_20260820T214522Z_v3 (861e) and v3_exq_861c_inv050_mech180_calibration_fixed_replication_20260814T231404Z_v3 (861c).

================================================================================
WHAT 861e LEFT OPEN
================================================================================
861c and 861e are not only a CALIB_DRAWS contrast. They executed DIFFERENT
substrates on DIFFERENT boxes:

  861c: substrate_commit f810969, substrate_hash 5eaa59f5..., ree-cloud-4, 3.5h
  861e: substrate_commit 17befb8c, substrate_hash d1f4bdae..., ree-worker-1, 10.3h

and seed 271's ARM_3_HIGH_ON mean MEL went from 2.8999705e-05 (above its own
reference; factor 1.215) to 2.2980323e-05 (below it; factor 0.884). The
confirmed autopsy calls that a live H3, distinct from the within-run
substrate_stable_across_run flag (which is 798a's reuse-safety pattern, not a
mid-run code change).

The f810969 -> 17befb8c delta is real and NOT merely a default-off knob
addition: 1296 inserted lines across 8 ree_core files, including UNCONDITIONAL
changes in agent.py (extra clock.phase_reset() sites, the cem_elite modulatory
route backstop, orienting-decision tick decrements) and the whole
ContextMemory write-selection machinery in e1_deep.py. So "the substrate
changed under the comparison" is a hypothesis with a concrete mechanism behind
it, not a formality.

================================================================================
H3 -- WHAT THIS LEG DOES
================================================================================
IT VARIES EXACTLY ONE THING vs 861e: ree_core is pinned to f810969, 861c's own
substrate. CALIB_DRAWS stays at 10, R3 stays on, no reseed, legacy write path --
i.e. the 861e protocol, run on the old code. That fills the missing cell of the
2x2 the lineage has been reasoning across:

                     CALIB_DRAWS=5        CALIB_DRAWS=10
    f810969          861c (271 HIGH)      THIS LEG (primary grid)
    17befb8c         -- (that is H2) --   861e (271 collapsed)

Verified at authoring time on the pinned tree: all 43 REEConfig.from_dims
kwargs this driver passes exist at f810969, so none is silently swallowed
(cf. reference-reeconfig-from-dims-silent-kwargs, where a knob absent from
from_dims is dropped without error). The ContextMemory write-selection knobs do
NOT exist at f810969, which is itself informative: 861c could not have had the
repair, so both predecessors necessarily ran the legacy write path.

================================================================================
IN-RUN CONTROL -- a POSITIVE control that validates the pin behaviourally
================================================================================
After the 5x3 grid at CALIB_DRAWS=10, this leg re-runs ARM_3_HIGH_ON for all
three seeds at CALIB_DRAWS=5 -- i.e. 861c's exact decisive condition -- on the
same pinned substrate, in the same process, on the same machine.

This control does double duty:

 (i) It is a behavioural verification of the pin that no static check can give.
     861c recorded seed 271 ARM_3_HIGH_ON mean_mel 2.8999705e-05 against
     reference 2.3900e-05 (factor 1.215). If the pin is faithful, this control
     should land near that; if it lands near 861e's 2.2980e-05 instead, the pin
     did not take (or the machine delta dominates) and the primary grid must
     not be read as an H3 answer.
 (ii) It isolates the CALIB_DRAWS change ON THE OLD SUBSTRATE, same box --
      which is the contrast 861c-vs-861e was supposed to be but was not.

================================================================================
DECLARED NULL (pre-registered)
================================================================================
DECISIVE COMPARISON: seed 271, ARM_3_HIGH_ON, primary (f810969, n=10) against
861e's recorded cell (17befb8c, n=10), and against this run's own n=5 control.

 - H3 SUPPORTED if seed 271 stays HIGH-graded here (factor above 1.0) at n=10 on
   f810969, while 861e collapsed at n=10 on 17befb8c. The collapse is then a
   substrate (or machine) delta, and the 861c/861e comparison is confounded by
   code drift rather than by calibration power.
 - H3 NOT SUPPORTED if seed 271 collapses here too. The old substrate does not
   rescue it, so the collapse tracks CALIB_DRAWS or measurement RNG (H1) or the
   seed itself (H2) -- and the in-run n=5 control discriminates further: if n=5
   reproduces 861c's 1.215 while n=10 collapses on the SAME substrate and box,
   that is a clean, machine-free demonstration that CALIB_DRAWS alone moves the
   readout, which is H1's mechanism confirmed from the other side.
 - UNINFORMATIVE (non_contributory) if the n=5 positive control does NOT
   approximately reproduce 861c. Then the pin or the machine is dominating and
   the primary grid answers nothing about H3.

WHAT A NULL HERE DOES NOT MEAN: it is not evidence about INV-050 or MECH-180,
and it is not a substrate ceiling. Neither claim's status, confidence, or
v3_pending may move on this leg.

================================================================================
WHAT IS UNCHANGED FROM V3-EXQ-861e (deliberately -- this is an ISOLATION leg)
================================================================================
Env, ARMS, SEEDS [7, 271, 883], agent config, C1/C2 formulae, R1/R2/R3
readiness, MEAS_CYCLES, thresholds, the interpretation grid and the scored-DV
set are all byte-identical to V3-EXQ-861e. Each leg of this portfolio varies
EXACTLY ONE thing, so a difference in the readout is attributable.

MECH-122 content-selection stays OFF (USE_MECH122_SPINDLE_CONTENT_SELECTION =
False), as in 861b/861c/861e.

SLEEP DRIVER: manual-cycle-loop (force_cycle() called once per cycle in a
dedicated MEAS_CYCLES wake-sleep loop) -- unchanged from 861/861a/845/861b/
861c/861e.

================================================================================
SUBSTRATE PIN (experiments/_lib/substrate_pin.py)
================================================================================
ree_core/ is executed from a PINNED historical commit, extracted read-only with
`git archive` into a scratch dir placed first on sys.path. experiments/_lib/**,
experiment_protocol and pack_writer still come from the LIVE checkout. The pin
is proven by TWO fatal checks before any science runs -- a structural path check
on ree_core.__file__, and a behavioural source-marker check
(ree_core.predictors.e3_selector.authority_spread_ratio, which exists at
17befb8c and does NOT exist at f810969). A pin that cannot be proven raises
SubstratePinError, exits non-zero, and the runner classifies it ERROR. Running
a leg that cannot say which substrate it executed is the exact verdict-aliasing
failure this portfolio exists to avoid, so there is no degraded mode.

Per-cell arm_fingerprint uses repo_root=<pin dir> and substrate_scope=
("ree_core/**/*.py",), so the recorded substrate_hash describes the code that
ACTUALLY RAN. That scope is deliberately narrower than the default globs, so
every pinned cell is stamped reuse-INELIGIBLE
(substrate_pinned_to_historical_commit) and must never be minted as a baseline.

Note the pin also makes this leg independent of trunk drift while it sits in
the queue -- a substrate change landing between queue-time and run-time cannot
silently move the comparison.

================================================================================
EXPERIMENT_PURPOSE = "diagnostic" -- and why the directions are "unknown"
================================================================================
This is an instrument-isolation probe, not evidence for or against INV-050 /
MECH-180. It is tagged with both claim_ids because that is the lineage it
adjudicates, but evidence_direction is pinned non_contributory and
evidence_direction_per_claim is {"INV-050": "unknown", "MECH-180": "unknown"}
REGARDLESS of how the replicated C1/C2 grid comes out. The grid IS still
computed and recorded in full (under interpretation + replication_readout), so
it is directly comparable to 861c/861e -- it just does not vote. Diagnostics are
excluded from governance confidence/conflict scoring by construction; this
pinning makes that explicit rather than relying on it.

PROMOTES/DEMOTES NOTHING BY ITSELF: /governance applies the verdict.

================================================================================
STANDING SUBSTRATE-DEFECT DISCLOSURE (queue-experiment Step 2.5c)
================================================================================
`contextmemory-write-path-addressing-degeneracy` (substrate_queue, severity
CORRUPTING, status implemented_pending_validation, substrate_paths
[ree_core/predictors/e1_deep.py]) OVERLAPS this driver -- and it is not
hypothetical here. Measured directly from the recorded manifests of both
predecessors, seed 271 is write-address LOCKED in BOTH runs, on EVERY arm:

  861c (f810969, n=5)   seed 271 per_cycle_n_touched_slots ~ [1,1,1,1,2,1]
  861e (17befb8c, n=10) seed 271 per_cycle_n_touched_slots ~ [1,1,1,1,1,1]
                        (ARM_3_HIGH_ON: 6/6 cycles insufficient, new_div 0.0000)
  seeds 7 and 883 rotate normally in both runs (4-14 slots touched).

Seed 271 is the seed the whole H1-vs-H3 discrimination rests on. Two
consequences, both pre-registered here:

 (a) The lock is a CONSTANT across 861c and 861e, so it cannot be the cause of
     the DIFFERENCE between them. H1 vs H3 remains well-posed.
 (b) The lock still means seed 271's readouts are produced by an agent with a
     1-slot context bank, which is a documented corrupting condition. That is
     why V3-EXQ-861h exists: it repeats this protocol with the already-built
     non-degenerate write selection ON, so the portfolio measures the defect
     instead of inheriting it. 861f/861g deliberately keep the LEGACY argmin
     write path, because changing it would destroy the single-variable
     isolation that is their entire point.

Every cell records per_cycle_n_touched_slots and
n_cycles_insufficient_touched_slots, and the manifest carries a
contextmemory_write_lock_audit block, so any later autopsy can read the lock
state per seed without re-deriving it.

Also disclosed, unchanged from 861e's own Step 2.5c adjudication:
 - MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (corrupting): inert here --
   agent.run_sws_schema_pass gates the whole relative_novelty() consumption
   behind use_mech122_spindle_content_selection, False in this run.
 - mode-governance-engagement (corrupting): unrelated subsystem; this driver
   never imports salience_coordinator.py or regime_occupancy_gate.
 - SD-MECH267-CEM-SELECTION-FIX / SD-MECH303-THRESHOLD-SOURCING /
   SD-SLEEP-ENTRY-PRESSURE / SD-E3-SCORER-COMPLETION / SD-ORIENTING-DECISION-
   SCALE (degrading): arm-symmetric, noted not blocking, as in 861e.

================================================================================
RE-DERIVE BRAKE (queue-experiment Step 2.5b) -- examined, RELEASED
================================================================================
The literal run-keyed counter reads INV-050 = 7 and MECH-180 = 5, i.e. over the
threshold of 2. It is released on three independent grounds:

 1. The skill's own "Not braked" clause: this is a `diagnostic` whose purpose is
    to discriminate WHY the reading came out as it did -- not a lettered
    re-derive of the same claim at the same granularity.
 2. GOV-FANOUT-1 makes a diverse discrimination PORTFOLIO the prescribed route
    for a braked lineage carrying a fanout_recommendation, which is exactly what
    failure_autopsy_V3-EXQ-861e_2026-08-21 emitted.
 3. That autopsy's own producer half stamped re_derive_brake.fired = false,
    refused_requeue = false, route_to = "queue-experiment", and its Step 8
    interactive gate recorded verdict "adopt_redteam_portfolio".

The counter's hits are almost all `standard` / `non_contributory` targets that
count only through the non_contributory-direction proxy the skill itself warns
inverts the brake's purpose for instrument defects. 861e's category note says it
outright: "Stamping substrate_ceiling here would fire the re-derive brake and
forbid the H1/H3 isolation the data now need."

================================================================================
GOV-REUSE-1 (Step 2.4) -- not recoverable, must run
================================================================================
Decisive readout: ARM_3_HIGH_ON mean_mel and mean_duration_factor for seed 271,
each against that cell's own mel_reference, under this leg's single varied
condition. No recorded manifest carries it: 861c has (f810969, n=5, no reseed,
legacy write), 861e has (17befb8c, n=10, no reseed, legacy write), and every
cell in this portfolio is a condition neither ran. Not recoverable -> run.
"""

import sys
import math
import time
import random
import argparse
from collections import deque
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- SUBSTRATE PIN -- MUST run before the first `import ree_core` -----------
# A driver that imports ree_core first silently gets the live checkout and the
# pin becomes a no-op, which is exactly the verdict-aliasing failure this leg
# exists to avoid. pin_ree_core() raises if ree_core is already in sys.modules,
# and verify_pin() raises if the pin cannot be PROVEN structurally AND
# behaviourally. Both are fatal on purpose -- see the docstring.
from experiments._lib.substrate_pin import (            # noqa: E402
    pin_ree_core, verify_pin, pin_fingerprint_kwargs, pin_manifest_block,
)

SUBSTRATE_PIN_REF = "f810969089fa8193959f49072e8aa1c2de0cb193"
# Source marker that DIFFERS across the pinned/live boundary: added by
# ree-v3 commit 17befb8c ("modulatory-bias-selection-authority AMEND"), so it
# is present at 17befb8c and absent at f810969. A structural path check alone
# cannot catch a stale cache dir holding the wrong ref's content; this can.
SUBSTRATE_PIN_MARKER_MODULE = "ree_core.predictors.e3_selector"
SUBSTRATE_PIN_MARKER_ATTR = "authority_spread_ratio"
SUBSTRATE_PIN_MARKER_EXPECTED_PRESENT = False

_PIN = pin_ree_core(SUBSTRATE_PIN_REF)
verify_pin(
    _PIN,
    marker_module=SUBSTRATE_PIN_MARKER_MODULE,
    marker_attr=SUBSTRATE_PIN_MARKER_ATTR,
    marker_expected_present=SUBSTRATE_PIN_MARKER_EXPECTED_PRESENT,
)
# repo_root=<pin dir> + substrate_scope=ree_core/**  -> the recorded
# substrate_hash describes the code that ACTUALLY RAN, and every pinned cell is
# stamped reuse-INELIGIBLE.
_PIN_FP_KWARGS = pin_fingerprint_kwargs(_PIN)

import numpy as np
import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.readiness_anchor import assert_anchor_reachable
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_861g_inv050_mech180_h3_substrate_pin_f810969"
QUEUE_ID = "V3-EXQ-861g"
CLAIM_IDS = ["INV-050", "MECH-180"]
# diagnostic, NOT evidence: an instrument-isolation leg. Excluded from
# governance confidence/conflict scoring; directions pinned "unknown".
EXPERIMENT_PURPOSE = "diagnostic"

# -- GOV-FANOUT-1 leg identity (see docstring) ------------------------------
FANOUT_QID = "inv050_mech180_861e_producer_vs_intervention_isolation"
FANOUT_HYPOTHESIS = "H3"
FANOUT_AXIS = "algorithm"
FANOUT_SOURCE_AUTOPSY = "failure_autopsy_V3-EXQ-861e_2026-08-21"
PRIMARY_VARIANT = "n10"
CONTROL_VARIANT = "n5"
# The decisive cell of the whole 861 fan-out. Both the in-run control set and
# every declared-null comparison are scoped to it.
DECISIVE_ARM_ID = "ARM_3_HIGH_ON"
DECISIVE_SEED = 271
# Recorded predecessor values for the decisive cell, read from the manifests at
# authoring time and pinned here so the comparison is pre-registered rather than
# recomputed post hoc.
PREDECESSOR_DECISIVE = {
    # seed 271 / ARM_3_HIGH_ON, copied verbatim from the recorded manifests at
    # authoring time. These are the frozen reference cells the anchor-
    # reachability guards below score the SHIPPED predicates against.
    "861c": {
        "run_id": ("v3_exq_861c_inv050_mech180_calibration_fixed_replication"
                   "_20260814T231404Z_v3"),
        "substrate_commit": "f810969", "machine": "ree-cloud-4",
        "calib_draws": 5,
        "mean_mel": 2.8999705256751363e-05,
        "mel_reference": 2.3899629401276597e-05,
        "mean_duration_factor": 1.2145974718547234,
        "n_cycles_insufficient_touched_slots": 4, "meas_cycles": 6,
    },
    "861e": {
        "run_id": ("v3_exq_861e_inv050_mech180_calibration_power_raised_replication"
                   "_20260820T214522Z_v3"),
        "substrate_commit": "17befb8c", "machine": "ree-worker-1",
        "calib_draws": 10,
        "mean_mel": 2.2980323189226863e-05,
        "mel_reference": 2.5982329782371255e-05,
        "mean_duration_factor": 0.8844596840125855,
        "n_cycles_insufficient_touched_slots": 6, "meas_cycles": 6,
    },
}
# Seed 271 was write-address LOCKED in BOTH predecessors -- measured, not
# assumed. See the docstring's Step 2.5c disclosure.
PREDECESSOR_WRITE_LOCK = {
    "861c": {7: False, 271: True, 883: False},
    "861e": {7: False, 271: True, 883: False},
}
# A cycle counts as write-address locked when fewer than 2 slots moved, which is
# the same >=2-occupied-slots floor substrate_queue registered for
# contextmemory-write-path-addressing-degeneracy at V3-EXQ-436f/943.
WRITE_LOCK_INSUFFICIENT_FRAC = 0.5

# NOT a supersession of V3-EXQ-861c: 861c's own recorded result (non_contributory,
# C2 under-powered at 1/3) is the CORRECT, unaltered adjudication for its own
# CALIB_DRAWS=5 configuration -- unlike 861b, nothing about 861c's result was
# wrong or an artifact, it was simply inconclusive. This run supplies the
# properly-powered follow-up the autopsy's repair pathway named, alongside
# 861c rather than in place of it.
SUPERSEDES = None

# The V3-EXQ-861c run this follows up (byte-identical env/arms/seeds/C1
# readout/agent config/C2 formula; ONLY CALIB_DRAWS and the new R3
# precondition differ) -- recorded for audit.
# The run this leg ISOLATES. 861c's decisive value is also carried, in
# PREDECESSOR_DECISIVE below, because several declared-null branches read it.
COMPARES_AGAINST_RUN_ID = (
    "v3_exq_861e_inv050_mech180_calibration_power_raised_replication"
    "_20260820T214522Z_v3"
)

# -- INDEPENDENCE (unchanged from 861b/861c) ----------------------------------
PRIOR_LINEAGE_SEEDS: List[int] = [42, 123, 456]
PRIOR_LINEAGE_RUNS: List[str] = [
    "V3-EXQ-718", "V3-EXQ-718a", "V3-EXQ-845", "V3-EXQ-861", "V3-EXQ-861a",
    "V3-EXQ-861b", "V3-EXQ-861c", "V3-EXQ-861d", "V3-EXQ-901",
]
NON_DEGENERACY_LEVERS_SATISFIED = ["a_new_seeds", "c_consumer_absent_control_arm"]
NON_DEGENERACY_LEVERS_NOT_SATISFIED = ["b_held_out_environment"]

# -- DV3 descoping (pre-registered; see module docstring "DV3 IS DESCOPED") ---
SCORED_DVS = ["sws_power", "replay_rate"]
UNSCORED_DVS = ["spindle_density"]
DV3_SCORING_EXCLUDED_REASON = (
    "spindle_density (mean_sws_new_slot_diversity) is RECORDED but NOT SCORED, "
    "unchanged reasoning from V3-EXQ-861c: substrate_queue.json entry "
    "MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION's flag-OFF configuration is "
    "what this run uses; the flag-ON follow-up fix (mel_pe novelty reference) "
    "is confirmed to fire but its own gate operating point (861d) is a "
    "separate, not-yet-repaired defect, out of scope for this driver-side "
    "calibration fix. Gating on it would re-import a mechanism this run does "
    "not exercise."
)
DV3_BLOCKING_SD_ID = "MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION"

TOUCHED_SLOT_L2_EPS = 1e-6

# MECH-122 content-packaging: OFF in this run (861c/861b's value). See module
# docstring "SUBSTRATE-PATH OVERLAP GATE" for the empirical confirmation that
# this keeps the corrupting-flagged relative_novelty() consumption dead code.
USE_MECH122_SPINDLE_CONTENT_SELECTION = False
MECH122_SPINDLE_SELECTION_GAIN = 1.0   # inert while the flag is False

# z_goal_enabled=True inherited verbatim for architecture parity; see
# DEAD_Z_GOAL_STREAM_EXEMPT below (unchanged reasoning from 861c).
DEAD_Z_GOAL_STREAM_EXEMPT = (
    "inherited verbatim from V3-EXQ-718a/798a/845/861/861b/861c for "
    "architecture parity; wiring update_z_goal would activate the E3 goal "
    "term, E1 conditioning, and the SD-024 benefit-attractor producer, "
    "breaking the single-variable comparison against V3-EXQ-861c that this "
    "replication depends on. Knob is arm-symmetric (identical in every arm)."
)

# -- Design parameters (IDENTICAL to V3-EXQ-861c except the CALIBRATION block) -
SEEDS = [7, 271, 883]         # REUSED from 861b/861c, disjoint from PRIOR_LINEAGE_SEEDS
CONV_EPISODES = 60
STEPS_PER_EPISODE = 90
PROBE_BATTERY_SIZE = 64

# -- CALIBRATION POWER RAISE (the substantive change from V3-EXQ-861c) -------
# 861c used CALIB_DRAWS=5 (rel_sd 0.152-0.283), which left seed 271 failing C2
# by 0.5% purely on calibration sample size (confirmed
# failure_autopsy_861c-861d-mech180-cluster_2026-08-16 section 2c). Raising to
# 10 is the exact n that autopsy's own projection shows flips seed 271's
# margin to PASS, using SEM ~ rel_sd/sqrt(n_draws) (confirmed to hold in
# 861c's own manifest). CALIB_EPISODES_PER_DRAW unchanged at 6 (episodes per
# draw is not the lever the autopsy identified; more draws, not longer draws).
CALIB_DRAWS = 10               # independent repeated calibration draws (861c: 5)
CALIB_EPISODES_PER_DRAW = 6    # stable-base wake episodes per draw (unchanged)
CALIB_EPISODES = CALIB_DRAWS * CALIB_EPISODES_PER_DRAW   # total: 60 (861c: 30)
K_CALIB_MARGIN = 2.0           # unchanged from 861c

# -- NEW: calibration-precision readiness precondition (R3) -------------------
# See module docstring "THE FIX" item (2) for the full justification. At
# CALIB_DRAWS=10, the confirmed 861c rel_sd range (0.152-0.283) projects
# calib_rel_sd_of_mean in [0.048, 0.090]; 0.15 sits ~67% above the worst
# projected value. Pre-registered, NOT derived from this run's own stats.
MAX_CALIB_REL_SD_OF_MEAN = 0.15

MEAS_CYCLES = 6
WAKE_EPISODES_PER_CYCLE = 2
EPISODES_PER_RUN = (CONV_EPISODES + CALIB_EPISODES
                    + MEAS_CYCLES * WAKE_EPISODES_PER_CYCLE)

SWS_CONSOLIDATION_STEPS = 5
REM_ATTRIBUTION_STEPS = 10

MEL_GAIN = 1.0
FACTOR_MIN = 0.5
FACTOR_MAX = 3.0
MEL_RELATIVE_FLOOR = 1e-6

SD056_WEIGHT = 0.05
E2_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
TRANSITION_BUFFER_MAX = 256

# -- Thresholds (pre-registered constants, NOT derived from run stats). ------
# Byte-identical to V3-EXQ-845/861/861a/861b/861c EXCEPT MAX_CALIB_REL_SD_OF_MEAN
# (new). Deliberately NOT re-tuned otherwise: a replication that moved its
# thresholds would not be a replication.
MIN_REL_CONV_DROP = 0.10
SEED_PASS_FRAC = 2.0 / 3.0
MIN_MEL_SPREAD = 0.15
MIN_REL_DV_SPREAD = 0.15
MONO_TOL = 0.05
PINNED_ABS_VAR_ATOL = 1e-6

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

STABLE_DRIFT = dict(env_drift_interval=999, env_drift_prob=0.0)
WORLD_RULE_SHIFT_DEPTH = 2

ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_NONE_ON",  "level": 0, "interval": 0,  "mel_on": True},
    {"arm_id": "ARM_1_LOW_ON",   "level": 1, "interval": 60, "mel_on": True},
    {"arm_id": "ARM_2_MED_ON",   "level": 2, "interval": 25, "mel_on": True},
    {"arm_id": "ARM_3_HIGH_ON",  "level": 3, "interval": 10, "mel_on": True},
    {"arm_id": "ARM_4_HIGH_OFF", "level": 3, "interval": 10, "mel_on": False},
]
ON_ECO_ARMS = ["ARM_0_NONE_ON", "ARM_1_LOW_ON", "ARM_2_MED_ON", "ARM_3_HIGH_ON"]


def seed_overlap_with_prior_lineage(seeds: List[int]) -> List[int]:
    """UNCHANGED from 861c. Empty list = non-degeneracy precondition met."""
    return sorted(set(seeds) & set(PRIOR_LINEAGE_SEEDS))


def _make_env(seed: int, interval: int) -> CausalGridWorldV2:
    kw = dict(ENV_BASE)
    kw.update(STABLE_DRIFT)
    kw.update(
        world_rule_shift_enabled=(interval > 0),
        world_rule_shift_interval=interval,
        world_rule_shift_depth=WORLD_RULE_SHIFT_DEPTH if interval > 0 else 0,
    )
    return CausalGridWorldV2(seed=seed, **kw)


def _make_agent(env: CausalGridWorldV2, mel_on: bool, mel_reference: float) -> REEAgent:
    """UNCHANGED from 861c -- byte-identical config."""
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
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        surprise_gated_replay=True,
        use_sleep_loop=True,
        sleep_loop_episodes_K=10**9,
        sws_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        use_mech122_spindle_content_selection=USE_MECH122_SPINDLE_CONTENT_SELECTION,
        mech122_spindle_selection_gain=MECH122_SPINDLE_SELECTION_GAIN,
        rem_enabled=True,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        use_mel_consumer=bool(mel_on),
        mel_gain=MEL_GAIN,
        mel_reference=float(mel_reference),
        mel_reference_mode="fixed",
        mel_duration_factor_min=FACTOR_MIN,
        mel_duration_factor_max=FACTOR_MAX,
        mel_relative_floor=MEL_RELATIVE_FLOOR,
        mel_scale_sws=True,
        mel_scale_rem=True,
        use_mel_entry=False,
    )
    return REEAgent(cfg)


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


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


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int, rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen:
            seen[cls] = tup
        if len(seen) >= k:
            break
    if len(seen) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen.values())
    picked = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked:
            continue
        samples.append(tup)
        picked.add(id(tup))
    return samples


def _e2_train_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer, rng: random.Random,
) -> Optional[float]:
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


def _waking_step(
    agent: REEAgent, env: CausalGridWorldV2, obs_dict: Dict[str, Any],
    train: bool, buffer: Optional[Deque],
    e2_opt: Optional[torch.optim.Optimizer], sample_rng: Optional[random.Random],
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
) -> Tuple[Dict[str, Any], bool]:
    latent = _sense_latent(agent, obs_dict)

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
        return obs_dict, True

    if train and buffer is not None and torch.isfinite(latent.z_world).all():
        pending_capture_ref[0] = (
            latent.z_world.detach().reshape(-1).clone(),
            action.detach().reshape(-1).clone(),
        )
        if e2_opt is not None and sample_rng is not None:
            _e2_train_step(agent, buffer, e2_opt, sample_rng)

    _, harm_signal, done, info, next_obs_dict = env.step(action)
    with torch.no_grad():
        agent.update_residue(
            harm_signal=float(harm_signal), world_delta=None,
            hypothesis_tag=False, owned=True,
        )
    return next_obs_dict, bool(done)


def _run_wake_window(
    agent: REEAgent, env: CausalGridWorldV2, n_episodes: int, steps: int,
    train: bool, buffer: Optional[Deque],
    e2_opt: Optional[torch.optim.Optimizer], sample_rng: Optional[random.Random],
    ep_offset: int, arm_id: str, seed: int,
) -> None:
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None]
    for ep in range(n_episodes):
        glob_ep = ep_offset + ep
        if (glob_ep % 10 == 0) or (glob_ep == EPISODES_PER_RUN - 1):
            print(f"  [train] {arm_id} seed={seed} ep {glob_ep+1}/{EPISODES_PER_RUN}",
                  flush=True)
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        pending_capture_ref[0] = None
        for _step in range(steps):
            obs_dict, done = _waking_step(
                agent, env, obs_dict, train, buffer, e2_opt, sample_rng,
                pending_capture_ref,
            )
            if done:
                break


def _sample_probe_battery(
    agent: REEAgent, seed: int, n_transitions: int, steps: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    env = _make_env(seed, interval=0)
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    act_rng = random.Random(seed + 9973)
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    guard = 0
    max_guard = max(steps, 1) * 8
    while len(battery) < n_transitions and guard < max_guard:
        guard += 1
        latent = _sense_latent(agent, obs_dict)
        if not torch.isfinite(latent.z_world).all():
            break
        z_now = latent.z_world.detach().reshape(1, -1).clone()
        if prev is not None:
            z0, a = prev
            battery.append((z0, a, z_now))
        idx = act_rng.randrange(env.action_dim)
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, idx] = 1.0
        _, _, done, _, obs_dict = env.step(action)
        prev = (z_now, action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            agent.e1.reset_hidden_state()
            prev = None
    return battery


def _frozen_probe_pe(
    agent: REEAgent, battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    if not battery:
        return 0.0
    errs: List[float] = []
    with torch.no_grad():
        for z0, a, z1 in battery:
            pred = agent.e2.world_forward(z0.to(agent.device), a.to(agent.device))
            err = float((pred - z1.to(agent.device)).pow(2).mean().item())
            if math.isfinite(err):
                errs.append(err)
    return float(np.mean(errs)) if errs else 0.0


def _touched_slot_diversity(
    mem_before: torch.Tensor, mem_after: torch.Tensor, eps: float = TOUCHED_SLOT_L2_EPS,
) -> Tuple[float, int, bool]:
    """Byte-identical to 861/861a/861c."""
    diffs = (mem_after - mem_before).norm(dim=-1)
    touched_mask = diffs > eps
    n_touched = int(touched_mask.sum().item())
    if n_touched < 2:
        return 0.0, n_touched, True
    touched = mem_after[touched_mask]
    norms = touched.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normed = touched / norms
    sim_mat = torch.mm(normed, normed.t())
    mask = torch.eye(n_touched, device=sim_mat.device, dtype=torch.bool)
    off_diag = sim_mat[~mask]
    diversity = float((1.0 - off_diag).mean().item())
    return diversity, n_touched, False


_ZG = ZGoalStreamAccumulator()
_LAST_AGENT: List[Optional[REEAgent]] = [None]


def _base_config(arm: Dict[str, Any], conv_eps: int, calib_draws: int,
                 calib_eps_per_draw: int, meas_cycles: int, steps: int) -> Dict[str, Any]:
    """Config slice for one cell. Factored out of run_experiment (861e had it
    inline) so the in-run control set builds the SAME slice as the primary grid
    and the two fingerprints differ only by the varied knob."""
    return {
        "env_base": ENV_BASE,
        "arm": arm,
        "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH,
        "conv_episodes": conv_eps,
        "calib_draws": calib_draws,
        "calib_episodes_per_draw": calib_eps_per_draw,
        "k_calib_margin": K_CALIB_MARGIN,
        "max_calib_rel_sd_of_mean": MAX_CALIB_REL_SD_OF_MEAN,
        "meas_cycles": meas_cycles,
        "steps_per_episode": steps,
        "sws_steps": SWS_CONSOLIDATION_STEPS,
        "rem_steps": REM_ATTRIBUTION_STEPS,
        "mel_gain": MEL_GAIN,
        "factor_min": FACTOR_MIN,
        "factor_max": FACTOR_MAX,
        "mel_relative_floor": MEL_RELATIVE_FLOOR,
        "touched_slot_l2_eps": TOUCHED_SLOT_L2_EPS,
        "use_mech122_spindle_content_selection": USE_MECH122_SPINDLE_CONTENT_SELECTION,
        "mech122_spindle_selection_gain": MECH122_SPINDLE_SELECTION_GAIN,
        "substrate_pin_ref": SUBSTRATE_PIN_REF,
    }


def _variant_config(variant: str, calib_draws: int) -> Dict[str, Any]:
    """The ONE knob varied between primary and in-run control here is
    CALIB_DRAWS, which _base_config already carries -- so nothing extra."""
    return {}


def _variant_cell_kwargs(variant: str, calib_draws: int) -> Dict[str, Any]:
    return {}


def _control_calib_draws(calib_draws: int) -> int:
    """Half the primary draw count. At the production CALIB_DRAWS=10 this is
    exactly 5, i.e. V3-EXQ-861c's own decisive condition, which is what makes
    the control a POSITIVE control for the pin. Expressed as a ratio so the
    --dry-run smoke exercises the same code path at its own scale."""
    return max(1, calib_draws // 2)


# -- Pre-registered decision thresholds for this leg's declared null ---------
# A cell is "HIGH-graded" iff its mean measured MEL sits above its OWN
# calibrated reference. mean_duration_factor IS that ratio (clamped to
# [FACTOR_MIN, FACTOR_MAX]), so > 1.0 is exactly the autopsy's own criterion --
# not a threshold invented here.
FACTOR_GRADED_FLOOR = 1.0
# 861c's recorded decisive factor was 1.215; the pin positive control is
# required to land within ~10% below it. Pre-registered from that recorded
# value, NOT tuned against this run.
PIN_CONTROL_MIN_FACTOR = 1.10


def _cell(rows: List[Dict[str, Any]], seed: int, variant: str,
          arm_id: str = DECISIVE_ARM_ID) -> Optional[Dict[str, Any]]:
    return next((r for r in rows
                 if r["seed"] == seed and r["arm_id"] == arm_id
                 and r.get("variant") == variant), None)


def _insufficient_frac(row: Optional[Dict[str, Any]]) -> Optional[float]:
    """Fraction of measurement cycles in which fewer than 2 ContextMemory slots
    moved -- the write-address lock signature of substrate_queue
    `contextmemory-write-path-addressing-degeneracy`."""
    if not row:
        return None
    n = float(row.get("n_cycles_insufficient_touched_slots", 0) or 0)
    d = float(row.get("meas_cycles", 0) or 0)
    return (n / d) if d > 0 else None


def _cell_summary(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return {
        "arm_id": row["arm_id"], "seed": row["seed"], "variant": row.get("variant"),
        "mean_mel": row.get("mean_mel"),
        "mel_reference": row.get("mel_reference"),
        "mean_duration_factor": row.get("mean_duration_factor"),
        "high_graded": bool(float(row.get("mean_duration_factor", 0.0))
                            > FACTOR_GRADED_FLOOR),
        "calib_draws_this_cell": row.get("calib_draws_this_cell"),
        "mel_reference_calib_rel_sd_of_mean": row.get("mel_reference_calib_rel_sd_of_mean"),
        "per_cycle_n_touched_slots": row.get("per_cycle_n_touched_slots"),
        "n_cycles_insufficient_touched_slots": row.get("n_cycles_insufficient_touched_slots"),
        "write_address_locked": (
            (_insufficient_frac(row) or 0.0) >= WRITE_LOCK_INSUFFICIENT_FRAC),
        "contextmemory_write_selection": row.get("contextmemory_write_selection"),
        "reseed_before_measurement": row.get("reseed_before_measurement"),
    }


def _pin_control_reproduces_861c(cell: Dict[str, Any]) -> bool:
    """THE SHIPPED PREDICATE for this leg's readiness anchor. Used both by
    _discrimination() on the live n=5 control cell and by the setup-time
    reachability guard on 861c's frozen recorded cell -- one definition, never
    a copy, which is the whole point of the guard."""
    return float(cell.get("mean_duration_factor", 0.0)) >= PIN_CONTROL_MIN_FACTOR


_ANCHOR_REACHABILITY = [assert_anchor_reachable(
    anchor_name="pin_positive_control_reproduces_861c_decisive_cell",
    reference_cells=[PREDECESSOR_DECISIVE["861c"]],
    score_fn=_pin_control_reproduces_861c,
    threshold=1.0,
    reference_source=("V3-EXQ-861c recorded seed 271 / ARM_3_HIGH_ON cell "
                      "(factor 1.2146) on this leg's pinned substrate commit "
                      "-- the exact state the pin must reproduce"),
)]


def _write_lock_audit(arm_results: List[Dict[str, Any]],
                      seeds: List[int]) -> Dict[str, Any]:
    """Per-seed ContextMemory write-address lock state, recorded so a later
    autopsy never has to re-derive it from per_cycle_n_touched_slots.
    Disclosure context: queue-experiment Step 2.5c, substrate_queue entry
    `contextmemory-write-path-addressing-degeneracy` (severity corrupting)."""
    out: Dict[str, Any] = {
        "sd_id": "contextmemory-write-path-addressing-degeneracy",
        "severity": "corrupting",
        "substrate_paths": ["ree_core/predictors/e1_deep.py"],
        "lock_criterion": ("fraction of measurement cycles with < 2 touched "
                           "slots >= %.2f" % WRITE_LOCK_INSUFFICIENT_FRAC),
        "predecessor_lock_state_measured_from_manifests": PREDECESSOR_WRITE_LOCK,
        "per_seed": {},
    }
    for s in seeds:
        per_variant = {}
        for v in (PRIMARY_VARIANT, CONTROL_VARIANT):
            rows = [r for r in arm_results
                    if r["seed"] == s and r.get("variant") == v]
            if not rows:
                continue
            fracs = [f for f in (_insufficient_frac(r) for r in rows)
                     if f is not None]
            per_variant[v] = {
                "n_cells": len(rows),
                "mean_insufficient_frac": (float(sum(fracs) / len(fracs))
                                           if fracs else None),
                "locked_cells": sum(1 for f in fracs
                                    if f >= WRITE_LOCK_INSUFFICIENT_FRAC),
                "decisive_arm_insufficient_frac": _insufficient_frac(
                    _cell(arm_results, s, v)),
            }
        out["per_seed"][str(s)] = per_variant
    return out


def _discrimination(arm_results: List[Dict[str, Any]],
                    seeds: List[int]) -> Dict[str, Any]:
    """H3 verdict on the decisive cell.

    Primary = f810969 at CALIB_DRAWS=10 (the missing 2x2 cell). In-run control =
    f810969 at CALIB_DRAWS=5, i.e. 861c's own decisive condition, which doubles
    as the POSITIVE control that verifies the pin behaviourally. See the
    docstring's DECLARED NULL."""
    p = _cell(arm_results, DECISIVE_SEED, PRIMARY_VARIANT)
    c = _cell(arm_results, DECISIVE_SEED, CONTROL_VARIANT)
    ps, cs = _cell_summary(p), _cell_summary(c)
    pf = float(p.get("mean_duration_factor", 0.0)) if p else None
    cf = float(c.get("mean_duration_factor", 0.0)) if c else None

    # The pin is verified structurally + behaviourally at import time; this is
    # the third, empirical check: does the OLD substrate at 861c's own
    # CALIB_DRAWS reproduce 861c's own decisive number?
    pin_control_ok = (c is not None and _pin_control_reproduces_861c(c))
    primary_high_graded = (pf is not None and pf > FACTOR_GRADED_FLOOR)

    if not pin_control_ok:
        label = "uninformative_pin_positive_control_did_not_reproduce_861c"
        supported = None
    elif primary_high_graded:
        label = "h3_supported_old_substrate_retains_high_grading_at_n10"
        supported = True
    else:
        label = "h3_not_supported_collapse_present_on_old_substrate_at_n10"
        supported = False

    # Machine-free read on CALIB_DRAWS: n=10 vs n=5 on ONE substrate, ONE box.
    calib_only = None
    if pf is not None and cf is not None:
        calib_only = {
            "note": ("n=10 vs n=5 on the SAME pinned substrate and the SAME box. "
                     "861c-vs-861e could not isolate this because substrate and "
                     "machine moved with CALIB_DRAWS."),
            "factor_n10": pf, "factor_n5": cf, "delta": pf - cf,
            "calib_draws_alone_moves_readout": bool(
                (cf > FACTOR_GRADED_FLOOR) != (pf > FACTOR_GRADED_FLOOR)),
        }
    return {
        "verdict_label": label,
        "hypothesis_supported": supported,
        "primary_cell": ps, "control_cell": cs,
        "readings": {
            "pin_positive_control_reproduces_861c": bool(pin_control_ok),
            "pin_control_min_factor": PIN_CONTROL_MIN_FACTOR,
            "recorded_861c_factor": PREDECESSOR_DECISIVE["861c"]["mean_duration_factor"],
            "primary_high_graded_at_n10_on_f810969": bool(primary_high_graded),
        },
        "calibration_only_readout": calib_only,
    }


def _leg_preconditions(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The unstated assumption behind every H3 branch: the pin must reproduce
    861c's own decisive number at 861c's own CALIB_DRAWS. Structural and
    behavioural pin checks run at import; this is the empirical one, and it is
    a precondition rather than a finding (queue-experiment P0 readiness-assert)."""
    cs = d.get("control_cell") or {}
    return [{
        "name": "pin_positive_control_reproduces_861c_decisive_cell",
        "kind": "readiness",
        "description": ("the f810969-pinned ARM_3_HIGH_ON cell on seed 271 at "
                        "CALIB_DRAWS=5 must land at or above PIN_CONTROL_MIN_FACTOR, "
                        "i.e. near 861c's recorded 1.215, before the n=10 primary "
                        "grid can be read as an H3 answer."),
        "measured": cs.get("mean_duration_factor"),
        "threshold": PIN_CONTROL_MIN_FACTOR,
        "direction": "lower",
        "control": ("861c's own recorded decisive value (%.3f) on the same "
                    "substrate commit; threshold pre-registered at ~10%% below "
                    "it, not fitted to this run"
                    % PREDECESSOR_DECISIVE["861c"]["mean_duration_factor"]),
        "met": bool(d["readings"]["pin_positive_control_reproduces_861c"]),
    }]


def _run_cell(seed: int, arm: Dict[str, Any], steps: int, conv_eps: int,
              meas_cycles: int, calib_draws: int,
              calib_eps_per_draw: int, *,
              variant: str = PRIMARY_VARIANT,
              reseed_before_measurement: bool = False,
              write_selection: str = "argmin") -> Dict[str, Any]:
    """One (seed, arm, variant) cell.

    Cell logic is byte-identical to V3-EXQ-861e except for the ONE knob this
    leg varies, which is carried by the keyword-only arguments above and
    recorded on the returned row as `variant`. Every cell still resets all RNG
    at entry, so cells are independent of each other and of their order --
    which is what makes the in-run control set a valid same-machine,
    same-substrate replica rather than a sequence effect."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    arm_id = arm["arm_id"]
    mel_on = bool(arm["mel_on"])
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    stable_env = _make_env(seed, interval=0)
    agent = _make_agent(stable_env, mel_on=mel_on, mel_reference=0.0)
    battery = _sample_probe_battery(agent, seed, PROBE_BATTERY_SIZE, steps)
    probe_pe_init = _frozen_probe_pe(agent, battery)

    buffer: Deque = deque(maxlen=TRANSITION_BUFFER_MAX)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    sample_rng = random.Random(seed + 4242)

    _run_wake_window(
        agent, stable_env, conv_eps, steps, train=True, buffer=buffer,
        e2_opt=e2_opt, sample_rng=sample_rng, ep_offset=0, arm_id=arm_id, seed=seed,
    )

    probe_pe_final = _frozen_probe_pe(agent, battery)
    conv_rel_drop = (((probe_pe_init - probe_pe_final) / probe_pe_init)
                     if probe_pe_init > 1e-12 else 0.0)

    # -- Reference calibration (V3-EXQ-861c methodology, CALIB_DRAWS raised
    # to 10 in this run -- see module docstring "THE FIX" item (1)) --------
    calib_draws_mel: List[float] = []
    calib_ep_cursor = conv_eps
    if agent.mel_consumer is not None:
        for _draw in range(calib_draws):
            agent.mel_consumer.reset()
            _run_wake_window(
                agent, stable_env, calib_eps_per_draw, steps, train=False,
                buffer=None, e2_opt=None, sample_rng=None,
                ep_offset=calib_ep_cursor, arm_id=arm_id, seed=seed,
            )
            calib_ep_cursor += calib_eps_per_draw
            draw_mel = float(agent.mel_consumer.current_mel())
            if math.isfinite(draw_mel) and draw_mel > 0.0:
                calib_draws_mel.append(draw_mel)
    else:
        _run_wake_window(
            agent, stable_env, calib_draws * calib_eps_per_draw, steps,
            train=False, buffer=None, e2_opt=None, sample_rng=None,
            ep_offset=calib_ep_cursor, arm_id=arm_id, seed=seed,
        )
        calib_ep_cursor += calib_draws * calib_eps_per_draw

    n_calib_valid = len(calib_draws_mel)
    if mel_on and agent.mel_consumer is not None:
        if calib_draws_mel:
            base_ref = float(np.mean(calib_draws_mel))
            calib_sd = float(np.std(calib_draws_mel)) if n_calib_valid > 1 else 0.0
        else:
            base_ref = float(probe_pe_final)
            calib_sd = 0.0
        if not (base_ref > 0.0):
            base_ref = float(probe_pe_final)
            calib_sd = 0.0
        calib_rel_sd = (calib_sd / base_ref) if base_ref > 0.0 else 0.0
        calib_rel_sd_of_mean = calib_rel_sd / math.sqrt(max(1, n_calib_valid))
        agent.config.mel_reference = base_ref
        agent.mel_consumer.config.mel_reference = base_ref
        agent.mel_consumer.reset()
    else:
        base_ref = float(probe_pe_final)
        calib_sd = 0.0
        calib_rel_sd = 0.0
        calib_rel_sd_of_mean = 0.0

    meas_env = _make_env(seed, arm["interval"])
    cum_sws = 0.0
    cum_rem = 0.0
    per_cycle_sws: List[float] = []
    per_cycle_rem: List[float] = []
    per_cycle_diversity_legacy: List[float] = []
    per_cycle_new_diversity: List[float] = []
    per_cycle_n_touched: List[int] = []
    n_cycles_insufficient_touched = 0
    factors: List[float] = []
    mels: List[float] = []
    per_cycle_spindle_selection_applied: List[float] = []
    per_cycle_spindle_selection_mean_weight: List[float] = []
    ep_off = calib_ep_cursor
    for _cyc in range(meas_cycles):
        _run_wake_window(
            agent, meas_env, WAKE_EPISODES_PER_CYCLE, steps, train=False,
            buffer=None, e2_opt=None, sample_rng=None, ep_offset=ep_off,
            arm_id=arm_id, seed=seed,
        )
        ep_off += WAKE_EPISODES_PER_CYCLE

        mem_before = agent.e1.context_memory.memory.detach().clone()
        m = agent.sleep_loop.force_cycle(agent)
        mem_after = agent.e1.context_memory.memory.detach().clone()
        new_div, n_touched, insufficient = _touched_slot_diversity(mem_before, mem_after)
        if insufficient:
            n_cycles_insufficient_touched += 1

        sws = float(m.get("sws_n_writes", 0.0))
        rem = float(m.get("rem_n_rollouts", 0.0))
        diversity_legacy = float(m.get("sws_slot_diversity", 0.0))
        cum_sws += sws
        cum_rem += rem
        per_cycle_sws.append(sws)
        per_cycle_rem.append(rem)
        per_cycle_diversity_legacy.append(diversity_legacy)
        per_cycle_new_diversity.append(new_div)
        per_cycle_n_touched.append(n_touched)
        per_cycle_spindle_selection_applied.append(
            float(m.get("sws_spindle_selection_applied", 0.0))
        )
        per_cycle_spindle_selection_mean_weight.append(
            float(m.get("sws_spindle_selection_mean_weight", 0.0))
        )
        if mel_on:
            factors.append(float(m.get("mel_duration_factor", 1.0)))
            mels.append(float(m.get("mel_mean", 0.0)))

    valid_new_div = [v for v, n in zip(per_cycle_new_diversity, per_cycle_n_touched)
                     if n >= 2]
    mean_new_diversity = float(np.mean(valid_new_div)) if valid_new_div else 0.0
    mean_diversity_legacy = (float(np.mean(per_cycle_diversity_legacy))
                             if per_cycle_diversity_legacy else 0.0)
    sws_count_var = float(np.var(per_cycle_sws))
    rem_count_var = float(np.var(per_cycle_rem))
    new_diversity_var = float(np.var(valid_new_div)) if len(valid_new_div) > 1 else 0.0
    diversity_var_legacy = float(np.var(per_cycle_diversity_legacy))
    mean_factor = float(np.mean(factors)) if factors else 1.0
    mean_mel = float(np.mean(mels)) if mels else 0.0
    mean_spindle_selection_applied = float(np.mean(per_cycle_spindle_selection_applied))
    mean_spindle_selection_weight = float(np.mean(per_cycle_spindle_selection_mean_weight))

    _ZG.observe(agent)
    _LAST_AGENT[0] = agent

    print(f"    {arm_id} seed={seed}: conv_drop={conv_rel_drop:.3f} "
          f"ref={base_ref:.3e} (calib n={n_calib_valid} rel_sd={calib_rel_sd:.3f} "
          f"rel_sd_of_mean={calib_rel_sd_of_mean:.3f}) "
          f"mel={mean_mel:.3e} factor={mean_factor:.3f} "
          f"cum_sws={cum_sws:.0f} cum_rem={cum_rem:.0f} "
          f"[descoped dv3: mean_new_div={mean_new_diversity:.4f} "
          f"legacy={mean_diversity_legacy:.4f} "
          f"n_touched_mean={np.mean(per_cycle_n_touched):.1f} "
          f"insufficient_cycles={n_cycles_insufficient_touched}/{meas_cycles}] "
          f"spindle_sel_applied={mean_spindle_selection_applied:.2f} "
          f"spindle_sel_weight={mean_spindle_selection_weight:.4f}",
          flush=True)
    print(f"verdict: {'PASS' if conv_rel_drop >= MIN_REL_CONV_DROP else 'FAIL'}",
          flush=True)

    return {
        "arm_id": arm_id,
        "variant": variant,
        "reseed_before_measurement": bool(reseed_before_measurement),
        "contextmemory_write_selection": str(write_selection),
        "calib_draws_this_cell": int(calib_draws),
        "level": arm["level"],
        "mel_on": mel_on,
        "world_rule_shift_interval": arm["interval"],
        "seed": seed,
        "conv_rel_drop": conv_rel_drop,
        "probe_pe_init": probe_pe_init,
        "probe_pe_final": probe_pe_final,
        "mel_reference": base_ref,
        "mel_reference_calib_draws": list(calib_draws_mel),
        "mel_reference_calib_n_valid": n_calib_valid,
        "mel_reference_calib_sd": calib_sd,
        "mel_reference_calib_rel_sd": calib_rel_sd,
        "mel_reference_calib_rel_sd_of_mean": calib_rel_sd_of_mean,
        "mean_mel": mean_mel,
        "mean_duration_factor": mean_factor,
        "cumulative_sws_writes": cum_sws,
        "cumulative_rem_rollouts": cum_rem,
        "mean_sws_new_slot_diversity": mean_new_diversity,
        "per_cycle_new_diversity": per_cycle_new_diversity,
        "per_cycle_n_touched_slots": per_cycle_n_touched,
        "n_cycles_insufficient_touched_slots": n_cycles_insufficient_touched,
        "new_diversity_variance": new_diversity_var,
        "mean_sws_slot_diversity_wholebank_legacy": mean_diversity_legacy,
        "per_cycle_diversity_wholebank_legacy": per_cycle_diversity_legacy,
        "diversity_variance_wholebank_legacy": diversity_var_legacy,
        "per_cycle_sws": per_cycle_sws,
        "per_cycle_rem": per_cycle_rem,
        "per_cycle_mel": mels,
        "per_cycle_factor": factors,
        "sws_count_variance": sws_count_var,
        "rem_count_variance": rem_count_var,
        "meas_cycles": meas_cycles,
        "mean_spindle_selection_applied": mean_spindle_selection_applied,
        "mean_spindle_selection_weight": mean_spindle_selection_weight,
        "per_cycle_spindle_selection_applied": per_cycle_spindle_selection_applied,
        "per_cycle_spindle_selection_mean_weight": per_cycle_spindle_selection_mean_weight,
    }


def _dv_dose_response(values: List[float]) -> Dict[str, Any]:
    """UNCHANGED from 845/861/861c."""
    if len(values) < 2:
        return {"monotone_ok": False, "spread_ok": False, "pass_": False}
    lo = values[0]
    tol = MONO_TOL * max(lo, 1e-9)
    monotone_ok = all(values[i] <= values[i + 1] + tol for i in range(len(values) - 1))
    spread_ok = lo > 0 and values[-1] >= lo * (1 + MIN_REL_DV_SPREAD)
    return {"monotone_ok": bool(monotone_ok), "spread_ok": bool(spread_ok),
            "pass_": bool(monotone_ok and spread_ok)}


def _seed_readiness(on_eco_cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """R1 (world-model trained) AND R2 (ecological novelty->MEL link) --
    UNCHANGED from 845/861/861c. R3 (calibration precision on the decisive
    arm) is NEW in this run -- see module docstring "THE FIX" item (2)."""
    if len(on_eco_cells) != len(ON_ECO_ARMS):
        return {"r1_ok": False, "r2_ok": False, "r3_ok": False, "ready": False,
                "arm3_calib_rel_sd_of_mean": None}
    r1_ok = all(r["conv_rel_drop"] >= MIN_REL_CONV_DROP for r in on_eco_cells)
    arms_sorted = sorted(on_eco_cells, key=lambda r: r["mean_mel"])
    mels = [r["mean_mel"] for r in arms_sorted]
    r2_ok = mels[0] > 0 and mels[-1] >= mels[0] * (1 + MIN_MEL_SPREAD)
    arm3 = next((r for r in on_eco_cells if r["arm_id"] == "ARM_3_HIGH_ON"), None)
    arm3_calib_rel_sd_of_mean = (
        float(arm3["mel_reference_calib_rel_sd_of_mean"]) if arm3 is not None else None
    )
    r3_ok = (arm3_calib_rel_sd_of_mean is not None
             and arm3_calib_rel_sd_of_mean <= MAX_CALIB_REL_SD_OF_MEAN)
    return {
        "r1_ok": bool(r1_ok), "r2_ok": bool(r2_ok), "r3_ok": bool(r3_ok),
        "arm3_calib_rel_sd_of_mean": arm3_calib_rel_sd_of_mean,
        "ready": bool(r1_ok and r2_ok and r3_ok),
    }


def _seed_c1(on_eco_by_arm: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """UNCHANGED from 861c."""
    arms_sorted = sorted(on_eco_by_arm.values(), key=lambda r: r["mean_mel"])
    arm_order = [r["arm_id"] for r in arms_sorted]
    mels = [r["mean_mel"] for r in arms_sorted]
    sws = [r["cumulative_sws_writes"] for r in arms_sorted]
    div = [r["mean_sws_new_slot_diversity"] for r in arms_sorted]
    rem = [r["cumulative_rem_rollouts"] for r in arms_sorted]
    c1a = _dv_dose_response(sws)
    c1b = _dv_dose_response(div)
    c1c = _dv_dose_response(rem)
    n_scored_pass = sum(1 for c in (c1a, c1c) if c["pass_"])
    return {
        "arm_order_by_measured_mel": arm_order,
        "mel_by_measured_order": mels,
        "c1a_sws_power": {**c1a, "values": sws, "scored": True},
        "c1c_replay_rate": {**c1c, "values": rem, "scored": True},
        "c1b_spindle_density_decoupled": {
            **c1b, "values": div,
            "scored": False,
            "scoring_excluded": True,
            "scoring_excluded_reason": DV3_SCORING_EXCLUDED_REASON,
            "blocking_sd_id": DV3_BLOCKING_SD_ID,
        },
        "c1_scored_n_pass": n_scored_pass,
        "c1_scored_all_pass": bool(n_scored_pass == 2),
        "c1_scored_zero_pass": bool(n_scored_pass == 0),
    }


def _seed_c2(on_eco_by_arm: Dict[str, Dict[str, Any]],
             off_cell: Dict[str, Any]) -> Dict[str, Any]:
    """UNCHANGED formula from 861c -- now fed calib_rel_sd_of_mean computed
    with CALIB_DRAWS=10 rather than 5."""
    pinned_ok = (off_cell["sws_count_variance"] <= PINNED_ABS_VAR_ATOL
                 and off_cell["rem_count_variance"] <= PINNED_ABS_VAR_ATOL)
    on_high = on_eco_by_arm["ARM_3_HIGH_ON"]

    on_factor = float(on_high["mean_duration_factor"])
    calib_rel_sd_of_mean = float(
        on_high.get("mel_reference_calib_rel_sd_of_mean", 0.0))
    calib_margin = 1.0 + K_CALIB_MARGIN * calib_rel_sd_of_mean
    on_factor_clears_calib_noise = bool(on_factor >= calib_margin)
    on_gt_off = on_factor_clears_calib_noise

    on_gt_off_sws = on_high["cumulative_sws_writes"] > off_cell["cumulative_sws_writes"]
    on_gt_off_rem = on_high["cumulative_rem_rollouts"] > off_cell["cumulative_rem_rollouts"]
    on_gt_off_spindle = (on_high["mean_sws_new_slot_diversity"]
                         > off_cell["mean_sws_new_slot_diversity"])
    return {
        "pinned_ok": bool(pinned_ok),
        "on_gt_off": bool(on_gt_off),
        "on_factor": on_factor,
        "on_factor_calib_rel_sd_of_mean": calib_rel_sd_of_mean,
        "on_factor_calib_margin": calib_margin,
        "on_factor_clears_calib_noise": on_factor_clears_calib_noise,
        "k_calib_margin": K_CALIB_MARGIN,
        "on_gt_off_sws_reported_only": bool(on_gt_off_sws),
        "on_gt_off_rem_reported_only": bool(on_gt_off_rem),
        "on_gt_off_spindle_reported_only": bool(on_gt_off_spindle),
        "c2_pass": bool(pinned_ok and on_gt_off),
        "off_diversity_variance_reported_only": off_cell["new_diversity_variance"],
        "off_diversity_variance_wholebank_legacy_reported_only":
            off_cell["diversity_variance_wholebank_legacy"],
    }


def run_experiment(steps: int, conv_eps: int, meas_cycles: int,
                   seeds: List[int], arms: Optional[List[Dict[str, Any]]] = None,
                   calib_draws: int = CALIB_DRAWS,
                   calib_eps_per_draw: int = CALIB_EPISODES_PER_DRAW,
                   ) -> Dict[str, Any]:
    arms = arms if arms is not None else ARMS
    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            full_config = _base_config(arm, conv_eps, calib_draws,
                                       calib_eps_per_draw, meas_cycles, steps)
            full_config["variant"] = PRIMARY_VARIANT
            full_config.update(_variant_config(PRIMARY_VARIANT, calib_draws))
            with arm_cell(seed, config_slice=full_config,
                          script_path=Path(__file__),
                          **_PIN_FP_KWARGS) as cell:
                row = _run_cell(seed, arm, steps, conv_eps, meas_cycles,
                                calib_draws, calib_eps_per_draw,
                                variant=PRIMARY_VARIANT,
                                **_variant_cell_kwargs(PRIMARY_VARIANT, calib_draws))
                cell.stamp(row)
            arm_results.append(row)

    # -- IN-RUN CONTROL SET: the decisive arm only, re-run under the CONTROL
    # variant. Same process, same machine, same pinned substrate; cells reset
    # all RNG at entry so order is irrelevant. This is what lets the declared
    # null be decided WITHIN this run instead of against a cross-machine
    # comparison -- see the docstring's "IN-RUN CONTROL" section.
    control_arm = next((a for a in arms if a["arm_id"] == DECISIVE_ARM_ID), None)
    if control_arm is not None:
        for seed in seeds:
            c_draws = _control_calib_draws(calib_draws)
            full_config = _base_config(control_arm, conv_eps, c_draws,
                                       calib_eps_per_draw, meas_cycles, steps)
            full_config["variant"] = CONTROL_VARIANT
            full_config.update(_variant_config(CONTROL_VARIANT, c_draws))
            with arm_cell(seed, config_slice=full_config,
                          script_path=Path(__file__),
                          **_PIN_FP_KWARGS) as cell:
                row = _run_cell(seed, control_arm, steps, conv_eps, meas_cycles,
                                c_draws, calib_eps_per_draw,
                                variant=CONTROL_VARIANT,
                                **_variant_cell_kwargs(CONTROL_VARIANT, c_draws))
                cell.stamp(row)
            arm_results.append(row)

    # -- Readiness (R1/R2 unchanged, R3 NEW -- see _seed_readiness) --
    seed_ready: Dict[int, bool] = {}
    seed_readiness_detail: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        on_eco_cells = [r for r in arm_results
                        if r["seed"] == seed and r["mel_on"]
                        and r.get("variant") == PRIMARY_VARIANT]
        rd = _seed_readiness(on_eco_cells)
        seed_readiness_detail[seed] = rd
        seed_ready[seed] = rd["ready"]
    readiness_frac = sum(seed_ready.values()) / max(1, len(seeds))
    r1_frac = sum(1 for s in seeds if seed_readiness_detail[s]["r1_ok"]) / max(1, len(seeds))
    r2_frac = sum(1 for s in seeds if seed_readiness_detail[s]["r2_ok"]) / max(1, len(seeds))
    r3_frac = sum(1 for s in seeds if seed_readiness_detail[s]["r3_ok"]) / max(1, len(seeds))
    readiness_ok = readiness_frac >= SEED_PASS_FRAC

    c1_seed_pass = 0
    c2_seed_pass = 0
    c2_pinned_seed_pass = 0
    c2_factor_seed_pass = 0
    per_dv_seed_pass = {"sws_power": 0, "spindle_density": 0, "replay_rate": 0}
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        on_eco = {r["arm_id"]: r for r in arm_results
                  if r["seed"] == seed and r["mel_on"]
                  and r.get("variant") == PRIMARY_VARIANT}
        off_cell = next((r for r in arm_results
                         if r["seed"] == seed and r["arm_id"] == "ARM_4_HIGH_OFF"
                         and r.get("variant") == PRIMARY_VARIANT), None)
        rec: Dict[str, Any] = {
            "seed": seed, "ready": seed_ready[seed],
            "readiness_detail": seed_readiness_detail[seed],
        }
        if seed_ready[seed] and len(on_eco) == len(ON_ECO_ARMS) and off_cell:
            c1 = _seed_c1(on_eco)
            c2 = _seed_c2(on_eco, off_cell)
            rec.update({"c1": c1, "c2": c2})
            if c1["c1a_sws_power"]["pass_"]:
                per_dv_seed_pass["sws_power"] += 1
            if c1["c1b_spindle_density_decoupled"]["pass_"]:
                per_dv_seed_pass["spindle_density"] += 1
            if c1["c1c_replay_rate"]["pass_"]:
                per_dv_seed_pass["replay_rate"] += 1
            if c1["c1_scored_all_pass"]:
                c1_seed_pass += 1
            if c2["c2_pass"]:
                c2_seed_pass += 1
            if c2["pinned_ok"]:
                c2_pinned_seed_pass += 1
            if c2["on_factor_clears_calib_noise"]:
                c2_factor_seed_pass += 1
        else:
            rec.update({"c1": None, "c2": None, "skipped": "not_ready_or_missing_arms"})
        per_seed.append(rec)

    n_seeds = max(1, len(seeds))
    c1_frac = c1_seed_pass / n_seeds
    c2_frac = c2_seed_pass / n_seeds
    c2_pinned_frac = c2_pinned_seed_pass / n_seeds
    c2_factor_frac = c2_factor_seed_pass / n_seeds
    c1_all_pass = c1_frac >= SEED_PASS_FRAC
    c2_pass = c2_frac >= SEED_PASS_FRAC
    c2_pinned_ok = c2_pinned_frac >= SEED_PASS_FRAC
    per_dv_frac = {k: v / n_seeds for k, v in per_dv_seed_pass.items()}
    per_dv_pass = {k: (v >= SEED_PASS_FRAC) for k, v in per_dv_frac.items()}
    n_scored_dv_pass = sum(1 for k in SCORED_DVS if per_dv_pass[k])

    overlap = seed_overlap_with_prior_lineage(seeds)
    seeds_disjoint = (len(overlap) == 0)

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        inv050_direction = "non_contributory"
        mech180_direction = "non_contributory"
    elif not c2_pinned_ok:
        label = "mel_control_degenerate"
        outcome = "FAIL"
        inv050_direction = "non_contributory"
        mech180_direction = "non_contributory"
    elif not c2_pass:
        label = "mel_coupling_below_calibration_noise_floor"
        outcome = "FAIL"
        inv050_direction = "non_contributory"
        mech180_direction = "non_contributory"
    elif n_scored_dv_pass == 2:
        label = "third_drive_independent_seed_replication_confirmed"
        outcome = "PASS"
        inv050_direction = "supports"
        mech180_direction = "mixed"
    elif n_scored_dv_pass == 0:
        label = "third_drive_not_replicated_on_independent_seeds"
        outcome = "FAIL"
        inv050_direction = "weakens"
        mech180_direction = "weakens"
    else:
        label = "third_drive_partial_replication_independent_seeds"
        outcome = "FAIL"
        inv050_direction = "mixed"
        mech180_direction = "mixed"

    if inv050_direction == mech180_direction:
        direction = inv050_direction
    elif "weakens" in (inv050_direction, mech180_direction):
        direction = "weakens"
    else:
        direction = "mixed"

    ready_seeds = [s for s in seeds if seed_ready[s]]
    c1_gradient_present = readiness_ok and bool(ready_seeds)
    ready_on_dvs_sws = [r["cumulative_sws_writes"] for r in arm_results
                        if r["mel_on"] and seed_ready.get(r["seed"], False)]
    c1_dv_spread_nonzero = (
        len(set(round(v, 3) for v in ready_on_dvs_sws)) > 1
    ) if ready_on_dvs_sws else False
    off_cells = [r for r in arm_results if not r["mel_on"]]
    c2_off_present = len(off_cells) > 0

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "seeds_disjoint_from_prior_lineage",
                "description": "This run's seed set must share NO seed with the "
                               "ORIGINAL {42,123,456} configuration. Seeds "
                               "[7,271,883] REUSED from 861b/861c (disjoint).",
                "measured": float(len(overlap)),
                "threshold": 0.0,
                "direction": "upper",
                "control": "seed sets audited programmatically across all recorded "
                           "manifests tagging INV-050 and/or MECH-180",
                "met": bool(seeds_disjoint),
            },
            {
                "name": "world_forward_converged_frozen_probe",
                "description": "recon-only P0 converges on the fixed frozen-probe "
                               "battery (conv_rel_drop >= MIN_REL_CONV_DROP) on the "
                               "ecological ON arms, on >= 2/3 seeds (R1).",
                "measured": r1_frac,
                "threshold": SEED_PASS_FRAC,
                "direction": "lower",
                "met": bool(r1_frac >= SEED_PASS_FRAC),
            },
            {
                "name": "ecological_novelty_mel_gradient_present_this_config",
                "description": "measured mean_mel is non-degenerately graded across "
                               "the 4 ON arms, on >= 2/3 seeds (R2). Below-floor means "
                               "the IV never varied on this seed set (not-ready), NOT "
                               "an INV-050/MECH-180 falsification.",
                "measured": r2_frac,
                "threshold": SEED_PASS_FRAC,
                "direction": "lower",
                "met": bool(r2_frac >= SEED_PASS_FRAC),
            },
            {
                "name": "calibration_precision_adequate_on_decisive_arm",
                "description": "NEW in this run (repair for the 861c under-powered "
                               "C2 near-miss): ARM_3_HIGH_ON's own "
                               "mel_reference_calib_rel_sd_of_mean must not exceed "
                               "MAX_CALIB_REL_SD_OF_MEAN, on >= 2/3 seeds (R3). A "
                               "seed whose calibration is too imprecise to trust its "
                               "C2 margin self-routes to substrate_not_ready_requeue "
                               "rather than deciding C2 on an unreliable margin.",
                "measured": r3_frac,
                "threshold": SEED_PASS_FRAC,
                "direction": "lower",
                "control": "MAX_CALIB_REL_SD_OF_MEAN=0.15, pre-registered from the "
                           "confirmed failure_autopsy_861c-861d-mech180-cluster_"
                           "2026-08-16 projection at CALIB_DRAWS=10 "
                           "(rel_sd/sqrt(10) in [0.048, 0.090]), not derived from "
                           "this run's own stats.",
                "met": bool(r3_frac >= SEED_PASS_FRAC),
            },
        ],
        "criteria_non_degenerate": {
            "C1_measured_mel_gradient_present": bool(c1_gradient_present),
            "C1_dv_spread_nonzero": bool(c1_dv_spread_nonzero),
            "C2_off_control_present": bool(c2_off_present),
            "seeds_independent_of_prior_lineage": bool(seeds_disjoint),
        },
        "per_dv_seed_pass_fraction": per_dv_frac,
        "per_dv_pass": per_dv_pass,
        "scored_dvs": SCORED_DVS,
        "unscored_dvs": UNSCORED_DVS,
        "dv3_scoring_excluded_reason": DV3_SCORING_EXCLUDED_REASON,
        "dv3_blocking_sd_id": DV3_BLOCKING_SD_ID,
        "independence": {
            "this_run_seeds": list(seeds),
            "prior_lineage_seeds": PRIOR_LINEAGE_SEEDS,
            "prior_lineage_runs": PRIOR_LINEAGE_RUNS,
            "seed_overlap": overlap,
            "seeds_disjoint": bool(seeds_disjoint),
            "non_degeneracy_levers_satisfied": NON_DEGENERACY_LEVERS_SATISFIED,
            "non_degeneracy_levers_not_satisfied": NON_DEGENERACY_LEVERS_NOT_SATISFIED,
            "limitation": "Environment-independence is NOT established -- see "
                          "V3-EXQ-861c's own docstring for the unchanged reasoning.",
            "residual_lever_c_note": "ARM_4_HIGH_OFF shows the elevation is "
                          "consumer-driven rather than environmental stochasticity, "
                          "but does NOT separate genuine third-drive coupling from "
                          "the consumer's own deterministic MEL->duration "
                          "arithmetic. Unchanged from 861c.",
        },
        "replication_note": "Calibration-power-raised replication of "
                            "V3-EXQ-861c: identical env, arms, SEEDS [7,271,883], "
                            "C1 readout code, agent config, and C2 formula. The "
                            "ONLY changes are CALIB_DRAWS (5 -> 10) and the new R3 "
                            "calibration-precision readiness precondition. C1 "
                            "thresholds were deliberately NOT re-tuned.",
        "calibration_redesign": {
            "calib_draws": CALIB_DRAWS,
            "calib_episodes_per_draw": CALIB_EPISODES_PER_DRAW,
            "calib_episodes_total": CALIB_EPISODES,
            "prior_calib_draws_861c": 5,
            "k_calib_margin": K_CALIB_MARGIN,
            "max_calib_rel_sd_of_mean": MAX_CALIB_REL_SD_OF_MEAN,
            "r3_frac": r3_frac,
            "c2_pinned_frac": c2_pinned_frac,
            "c2_factor_clears_noise_frac": c2_factor_frac,
            "c2_pass_frac": c2_frac,
            "per_seed_calib_rel_sd_of_mean_arm3": [
                float(next((r for r in arm_results
                            if r["seed"] == s and r["arm_id"] == "ARM_3_HIGH_ON"
                            and r.get("variant") == PRIMARY_VARIANT),
                           {}).get("mel_reference_calib_rel_sd_of_mean", 0.0))
                for s in seeds
            ],
            "note": "c2_factor_clears_noise_frac is the fraction of seeds on which "
                    "ARM_3's mean_duration_factor cleared 1.0 + k*calib_rel_sd_of_mean "
                    "at CALIB_DRAWS=10. Compare against 861c (CALIB_DRAWS=5), which "
                    "passed on_gt_off on 1/3 seeds. r3_frac is the fraction of seeds "
                    "whose ARM_3 calibration precision cleared the new readiness bar; "
                    "a seed pre-registered to fail here (rather than at C2) means its "
                    "calibration was too imprecise to trust, not that coupling failed.",
        },
    }
    # -- Leg-specific: fan-out identity, the in-run discrimination, and the
    #    ContextMemory write-lock audit. Injected into the 861e interpretation
    #    block rather than replacing it, so the replicated C1/C2 grid stays
    #    directly comparable to 861c/861e.
    discrimination = _discrimination(arm_results, seeds)
    write_lock_audit = _write_lock_audit(arm_results, seeds)
    interpretation["fanout"] = {
        "qid": FANOUT_QID,
        "hypothesis": FANOUT_HYPOTHESIS,
        "axis": FANOUT_AXIS,
        "source_autopsy": FANOUT_SOURCE_AUTOPSY,
        "primary_variant": PRIMARY_VARIANT,
        "control_variant": CONTROL_VARIANT,
        "portfolio": ["V3-EXQ-861f (H1, measurement)",
                      "V3-EXQ-861g (H3, algorithm)",
                      "V3-EXQ-861h (substrate-defect control, representation)"],
        "frozen_set_note": ("The registry's frozen set for this qid is H1 + H3 "
                            "(count 2), H2 a labelled follow-on. V3-EXQ-861h is a "
                            "CONTROL added by the queue-experiment Step 2.5b(iv) "
                            "design audit and the Step 2.5c corrupting-overlap "
                            "gate, NOT a fourth frozen hypothesis."),
    }
    interpretation["discrimination"] = discrimination
    interpretation["anchor_reachability"] = _ANCHOR_REACHABILITY
    interpretation["contextmemory_write_lock_audit"] = write_lock_audit
    interpretation["preconditions"].extend(_leg_preconditions(discrimination))
    interpretation["criteria_non_degenerate"].update({
        "in_run_control_set_present": bool(discrimination.get("control_cell")),
        "primary_and_control_cells_differ": bool(
            discrimination.get("primary_cell") and discrimination.get("control_cell")
            and discrimination["primary_cell"].get("mean_duration_factor")
            != discrimination["control_cell"].get("mean_duration_factor")),
        "substrate_pin_verified": bool(_PIN.get("verified")),
    })
    interpretation["diagnostic_scope_note"] = (
        "EXPERIMENT_PURPOSE is diagnostic. The replicated C1/C2/readiness grid "
        "below is recorded for comparability with 861c/861e and does NOT vote on "
        "INV-050 or MECH-180; evidence_direction_per_claim is pinned unknown "
        "regardless of how it comes out.")

    criteria = [
        {"name": "C1a_sws_power_monotone_in_measured_mel", "load_bearing": True,
         "passed": bool(per_dv_pass["sws_power"])},
        {"name": "C1c_replay_rate_monotone_in_measured_mel", "load_bearing": True,
         "passed": bool(per_dv_pass["replay_rate"])},
        {"name": "C1b_spindle_density_decoupled_monotone_in_measured_mel",
         "load_bearing": False,
         "scoring_excluded": True,
         "scoring_excluded_reason": DV3_SCORING_EXCLUDED_REASON,
         "passed": bool(per_dv_pass["spindle_density"])},
        {"name": "C2_on_factor_clears_calibration_noise_floor", "load_bearing": True,
         "passed": bool(c2_pass)},
        {"name": "R3_calibration_precision_adequate_on_decisive_arm",
         "load_bearing": True,
         "passed": bool(r3_frac >= SEED_PASS_FRAC)},
        {"name": "SEEDS_disjoint_from_prior_lineage", "load_bearing": True,
         "passed": bool(seeds_disjoint)},
    ]
    combination_rule = (
        "PASS iff readiness_ok AND c2_pass AND (C1a AND C1c) each on >= 2/3 seeds, "
        "where readiness_ok now additionally requires R3 (calibration precision on "
        "ARM_3_HIGH_ON) on >= 2/3 seeds, alongside the unchanged R1/R2. c2_pass = "
        "pinned_ok AND (ARM_3 mean_duration_factor >= 1.0 + K_CALIB_MARGIN * "
        "calib_rel_sd_of_mean) on >= 2/3 seeds, computed at CALIB_DRAWS=10 (was 5 "
        "in V3-EXQ-861c). C1b (spindle_density) is COMPUTED AND RECORDED but "
        "EXCLUDED from the conjunction and from C2 -- see "
        "interpretation.dv3_scoring_excluded_reason. Pre-registered expectation: "
        "seed 7 is a genuine, disjoint producer-side failure (HIGH-arm MEL below "
        "the no-shift base) that no calibration fix addresses -- a 2/3 pass with "
        "seed 7 the lone C2 miss is this run's target CONFIRMED outcome, not a "
        "partial result."
    )

    # EXPERIMENT_PURPOSE is diagnostic: this leg answers a question about the
    # INSTRUMENT, so it must not vote on either claim no matter how the
    # replicated grid came out. The computed grid directions are preserved
    # verbatim under replication_readout for comparison with 861c/861e.
    return {
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": {
            "INV-050": "unknown",
            "MECH-180": "unknown",
        },
        "evidence_direction_note": (
            "Pinned non_contributory / unknown by construction: this is a "
            "GOV-FANOUT-1 instrument-isolation leg (" + FANOUT_HYPOTHESIS + ", "
            + FANOUT_AXIS + " axis) for qid " + FANOUT_QID + ", not evidence "
            "for or against INV-050 or MECH-180. Neither claim's status, "
            "confidence or v3_pending may move on it."),
        "replication_readout": {
            "note": ("What the unmodified 861e verdict machinery computed on this "
                     "leg's PRIMARY grid, recorded for comparability with "
                     "861c/861e. Does NOT vote -- see evidence_direction_note."),
            "grid_evidence_direction": direction,
            "grid_evidence_direction_per_claim": {
                "INV-050": inv050_direction,
                "MECH-180": mech180_direction,
            },
        },
        "discrimination": discrimination,
        "contextmemory_write_lock_audit": write_lock_audit,
        "substrate_pin": pin_manifest_block(_PIN),
        "interpretation": interpretation,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "readiness_ok": readiness_ok,
        "readiness_frac": readiness_frac,
        "r1_frac": r1_frac,
        "r2_frac": r2_frac,
        "r3_frac": r3_frac,
        "c1_all_pass": c1_all_pass, "c1_frac": c1_frac,
        "c2_pass": c2_pass, "c2_frac": c2_frac,
        "c2_pinned_frac": c2_pinned_frac,
        "c2_factor_clears_noise_frac": c2_factor_frac,
        "n_scored_dv_pass": n_scored_dv_pass,
        "per_dv_pass": per_dv_pass,
        "per_dv_frac": per_dv_frac,
        "seeds_disjoint_from_prior_lineage": bool(seeds_disjoint),
        "seed_overlap_with_prior_lineage": overlap,
        "per_seed": per_seed,
        "arm_results": arm_results,
        "thresholds": {
            "MIN_REL_CONV_DROP": MIN_REL_CONV_DROP,
            "SEED_PASS_FRAC": SEED_PASS_FRAC,
            "MIN_MEL_SPREAD": MIN_MEL_SPREAD,
            "MIN_REL_DV_SPREAD": MIN_REL_DV_SPREAD,
            "MONO_TOL": MONO_TOL,
            "PINNED_ABS_VAR_ATOL": PINNED_ABS_VAR_ATOL,
            "K_CALIB_MARGIN": K_CALIB_MARGIN,
            "CALIB_DRAWS": CALIB_DRAWS,
            "CALIB_EPISODES_PER_DRAW": CALIB_EPISODES_PER_DRAW,
            "MAX_CALIB_REL_SD_OF_MEAN": MAX_CALIB_REL_SD_OF_MEAN,
            "MEL_GAIN": MEL_GAIN,
            "FACTOR_MIN": FACTOR_MIN,
            "FACTOR_MAX": FACTOR_MAX,
            "MEL_RELATIVE_FLOOR": MEL_RELATIVE_FLOOR,
            "TOUCHED_SLOT_L2_EPS": TOUCHED_SLOT_L2_EPS,
        },
    }


def write_manifest(result: Dict[str, Any], *, dry_run: bool = False,
                   started_at: Optional[float] = None) -> str:
    ts = dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (Path(__file__).resolve().parents[2]
               / "REE_assembly" / "evidence" / "experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    full_config = {
        "env_base": ENV_BASE,
        "arms": ARMS,
        "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH,
        "conv_episodes": CONV_EPISODES,
        "calib_episodes": CALIB_EPISODES,
        "calib_draws": CALIB_DRAWS,
        "calib_episodes_per_draw": CALIB_EPISODES_PER_DRAW,
        "k_calib_margin": K_CALIB_MARGIN,
        "max_calib_rel_sd_of_mean": MAX_CALIB_REL_SD_OF_MEAN,
        "meas_cycles": MEAS_CYCLES,
        "wake_episodes_per_cycle": WAKE_EPISODES_PER_CYCLE,
        "steps_per_episode": STEPS_PER_EPISODE,
        "sws_steps": SWS_CONSOLIDATION_STEPS,
        "rem_steps": REM_ATTRIBUTION_STEPS,
        "mel_gain": MEL_GAIN,
        "factor_min": FACTOR_MIN,
        "factor_max": FACTOR_MAX,
        "mel_relative_floor": MEL_RELATIVE_FLOOR,
        "touched_slot_l2_eps": TOUCHED_SLOT_L2_EPS,
        "use_mech122_spindle_content_selection": USE_MECH122_SPINDLE_CONTENT_SELECTION,
        "mech122_spindle_selection_gain": MECH122_SPINDLE_SELECTION_GAIN,
        "prior_lineage_seeds": PRIOR_LINEAGE_SEEDS,
        "scored_dvs": SCORED_DVS,
        "unscored_dvs": UNSCORED_DVS,
    }
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "compares_against_run_id": COMPARES_AGAINST_RUN_ID,
        "sleep_driver_pattern": "manual-cycle-loop (force_cycle() once per cycle in a "
                                "MEAS_CYCLES wake-sleep loop; MEL consumer engages via "
                                "SleepLoopManager, novelty knob is SD-MEL-PRODUCER "
                                "world_rule_shift, not env_drift; UNCHANGED from "
                                "V3-EXQ-861/861a/845/861b/861c)",
        "timestamp_utc": ts,
        "seeds": SEEDS,
        "substrate_pin": pin_manifest_block(_PIN),
        "fanout_qid": FANOUT_QID,
        "fanout_hypothesis": FANOUT_HYPOTHESIS,
        "fanout_axis": FANOUT_AXIS,
        "fanout_source_autopsy": FANOUT_SOURCE_AUTOPSY,
        "discrimination": result["discrimination"],
        "contextmemory_write_lock_audit":
            result["contextmemory_write_lock_audit"],
        "replication_readout": result["replication_readout"],
        "evidence_direction_note": result["evidence_direction_note"],
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "interpretation": result["interpretation"],
        "criteria": result["criteria"],
        "combination_rule": result["combination_rule"],
        "readiness_ok": result["readiness_ok"],
        "readiness_frac": result["readiness_frac"],
        "r1_frac": result["r1_frac"],
        "r2_frac": result["r2_frac"],
        "r3_frac": result["r3_frac"],
        "c1_all_pass": result["c1_all_pass"], "c1_frac": result["c1_frac"],
        "c2_pass": result["c2_pass"], "c2_frac": result["c2_frac"],
        "c2_pinned_frac": result["c2_pinned_frac"],
        "c2_factor_clears_noise_frac": result["c2_factor_clears_noise_frac"],
        "n_scored_dv_pass": result["n_scored_dv_pass"],
        "per_dv_pass": result["per_dv_pass"],
        "per_dv_frac": result["per_dv_frac"],
        "seeds_disjoint_from_prior_lineage": result["seeds_disjoint_from_prior_lineage"],
        "seed_overlap_with_prior_lineage": result["seed_overlap_with_prior_lineage"],
        "per_seed": result["per_seed"],
        "arm_results": result["arm_results"],
        "thresholds": result["thresholds"],
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
        agent=_LAST_AGENT[0],
        started_at=started_at,
    )
    return str(out_path)


def main():
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="1 seed, tiny convergence + measurement (smoke)")
    args = ap.parse_args()

    if args.dry_run:
        steps = 12
        conv_eps = 4
        meas_cycles = 3
        # Small multi-draw calibration so the smoke still exercises the mean +
        # noise-estimate + R3 code paths (>1 draw so calib_sd is a real std).
        calib_draws = 2
        calib_eps_per_draw = 2
        # DECISIVE_SEED, not SEEDS[0]: the in-run control set and
        # _discrimination() are scoped to it, so smoking any other seed
        # would leave both paths structurally exercised but EMPTY -- the
        # first populated run would then be the multi-hour one.
        seeds = [DECISIVE_SEED]
        # DECISIVE_ARM_ID must stay in the smoke arm set: the in-run control
        # set is scoped to it, so dropping it would silently skip the control
        # path and _discrimination() entirely.
        smoke_ids = {"ARM_0_NONE_ON", DECISIVE_ARM_ID, "ARM_4_HIGH_OFF"}
        arms = [a for a in ARMS if a["arm_id"] in smoke_ids]
    else:
        steps = STEPS_PER_EPISODE
        conv_eps = CONV_EPISODES
        meas_cycles = MEAS_CYCLES
        calib_draws = CALIB_DRAWS
        calib_eps_per_draw = CALIB_EPISODES_PER_DRAW
        seeds = SEEDS
        arms = ARMS

    result = run_experiment(steps, conv_eps, meas_cycles, seeds, arms,
                            calib_draws=calib_draws,
                            calib_eps_per_draw=calib_eps_per_draw)
    out_path = write_manifest(result, dry_run=bool(args.dry_run), started_at=t0)
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"seeds={seeds} disjoint_from_prior_lineage="
          f"{result['seeds_disjoint_from_prior_lineage']} "
          f"overlap={result['seed_overlap_with_prior_lineage']}", flush=True)
    print(f"readiness_frac={result['readiness_frac']:.2f} "
          f"(r1={result['r1_frac']:.2f} r2={result['r2_frac']:.2f} "
          f"r3={result['r3_frac']:.2f}) "
          f"c1_frac={result['c1_frac']:.2f} c2_frac={result['c2_frac']:.2f} "
          f"(pinned={result['c2_pinned_frac']:.2f} "
          f"factor_clears_noise={result['c2_factor_clears_noise_frac']:.2f}) "
          f"n_scored_dv_pass={result['n_scored_dv_pass']}/2 "
          f"per_dv={result['per_dv_frac']}", flush=True)
    print(f"evidence_direction_per_claim={result['evidence_direction_per_claim']}",
          flush=True)
    _d = result["discrimination"]
    print(f"fanout: qid={FANOUT_QID} hypothesis={FANOUT_HYPOTHESIS} "
          f"axis={FANOUT_AXIS}", flush=True)
    print(f"substrate_pin: ref={SUBSTRATE_PIN_REF[:10]} verified={_PIN['verified']} "
          f"dir={_PIN['pin_dir']}", flush=True)
    print(f"discrimination: {_d['verdict_label']}", flush=True)
    print(f"  primary({PRIMARY_VARIANT})={_d['primary_cell']} ", flush=True)
    print(f"  control({CONTROL_VARIANT})={_d['control_cell']}", flush=True)
    print(f"  readings={_d['readings']}", flush=True)
    print(f"manifest: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_dry_run,
    )
