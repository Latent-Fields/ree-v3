"""
V3-EXQ-861b -- INDEPENDENT-SEED REPLICATION of INV-050's third-drive coupling
(and, on its two testable DVs, MECH-180's novelty-adaptive sleep upregulation).

Does NOT supersede V3-EXQ-861 or V3-EXQ-861a: both remain valid evidence for
their own configurations. This run changes exactly ONE experimental variable
relative to V3-EXQ-861 -- the SEED SET -- and narrows the pre-registered
scoring gate to the two DVs whose enabling substrate is actually built (see
"DV3 IS DESCOPED" below). Everything else (env, arms, thresholds, readiness
gates, readout code, agent config) is byte-identical to V3-EXQ-861.

SLEEP DRIVER: manual-cycle-loop (agent.sleep_loop.force_cycle() called once per
cycle in a dedicated MEAS_CYCLES wake-sleep loop). The MEL consumer engages
ONLY through the SleepLoopManager path (force_cycle), exactly as in
V3-EXQ-718a/845/861/861a; a driver calling agent.run_sleep_cycle() directly
would bypass it.

CLAIMS UNDER TEST:
  INV-050 (primary): "Sleep phase architecture is regulated by three distinct
    drives -- circadian timing, homeostatic synaptic pressure, and a
    learning/model-update demand drive proportional to daily prediction error
    accumulation (Model Error Load, MEL) -- and only the third drive determines
    whether the overnight update phase is sufficient for the error burden
    generated during waking."
  MECH-180 (secondary, PARTIAL -- 2 of its 3 named DVs; see CLAIM TAGGING):
    "Novel environments and high-MEL learning episodes adaptively upregulate
    the learning drive component of sleep (INV-050 third drive), producing
    measurable increases in slow-wave activity power, sleep spindle density,
    and hippocampal replay rate proportional to the novelty and prediction
    error load encountered during the preceding wake period."

================================================================================
WHY THIS RUN EXISTS -- the gate is INDEPENDENCE, not substrate
================================================================================
INV-050's core DVs (sws_power, replay_rate) have now passed 2/3 seeds in THREE
consecutive autopsy-confirmed runs with zero FAILs:
  V3-EXQ-845  (2026-08-01, confirmed failure_autopsy_V3-EXQ-845_2026-08-01)
  V3-EXQ-861  (2026-08-01, write-count-deconfounded redesign,
               confirmed failure_autopsy_V3-EXQ-861_2026-08-01)
  V3-EXQ-861a (2026-08-02/03, MECH-122 content-selection flag validation;
               DV1/DV2 unaffected and unchanged,
               confirmed failure_autopsy_V3-EXQ-861a_2026-08-03)

All three share the IDENTICAL environment (CausalGridWorldV2 / SD-MEL-PRODUCER
world_rule_shift knob) and the IDENTICAL 3 seeds (42, 123, 456). Per this
project's own documented lesson (failure_autopsy_V3-EXQ-718a_2026-07-08:
"DV-monotone-in-measured-MEL is near-tautological on a functional consumer"),
that is ONE CONFIGURATION CONFIRMED THREE TIMES, not three independent
replications.

/governance 2026-08-07 (GFLAG-0002, user-confirmed) therefore HELD INV-050 at
`candidate` despite the 3x clean-PASS record, and re-gated promotion on a
genuinely independent test. The claim's own what_would_answer NON-DEGENERACY
PRECONDITION names three admissible levers -- vary at least one of:
  (a) seeds genuinely uninvolved in 845/861/861a (NOT a superset that merely
      adds more),
  (b) a held-out environment / world-model instance distinct from the
      CausalGridWorldV2 instance SD-MEL-PRODUCER was built and validated
      against,
  (c) an explicit MEL-consumer-absent control arm within the same run.

THIS RUN SATISFIES (a) AND (c):
  (a) SEEDS = [7, 271, 883]. Audited against every manifest in
      REE_assembly/evidence/experiments/ carrying an INV-050 / MECH-180 /
      SD-MEL-* experiment_type: the ENTIRE lineage (718, 718a, 845, 861, 861a,
      and INV-051's V3-EXQ-901) has only ever used {42, 123, 456}. The
      intersection with this run's seed set is EMPTY -- asserted numerically at
      runtime and emitted as the `seeds_disjoint_from_prior_lineage`
      precondition, so the independence claim is machine-checkable rather than
      a prose assertion.
  (c) ARM_4_HIGH_OFF (use_mel_consumer=False at matched novelty) is retained
      unchanged from 845/861/861a and remains the C2 control.
  (b) is NOT satisfied -- a held-out environment does not exist; SD-MEL-PRODUCER
      was built and validated (V3-EXQ-798a) against this CausalGridWorldV2
      instance specifically. This is stated as a LIMITATION, not papered over:
      a PASS here establishes seed-independence, NOT environment-independence.
      Environment-independence remains open and needs its own test-bed build.

GOV-REUSE-1 CHECK (Step 2.4, this session): decisive readout = C1a
(cumulative_sws_writes) and C1c (cumulative_rem_rollouts) monotone in measured
mean_mel, ON SEEDS 7/271/883. Every recorded manifest in this lineage was
audited programmatically (10 manifests tagging INV-050 and/or MECH-180); ALL
carry seeds [42, 123, 456] (or, for the pre-standard 677/701* runs, no seeds
field at all). The decisive readout on the new seed set is therefore not
present in, and not derivable from, ANY recorded manifest under any
substrate_hash -- the readout is definitionally about seeds no run has
executed. NOT RECOVERABLE by reanalysis -> proceeds to a new run.

RE-DERIVE BRAKE (Step 2.5b, this session): re-ran the counting method over
REE_assembly/evidence/planning/failure_autopsy_*.json.
  INV-050:  count = 5 (701, 701a, 701b, 718, 718a)
  MECH-180: count = 3 (677, 718, 718a)
Both >= threshold 2, so the brake is CHECKED and RELEASED, not ignored. The
named upstream substrate in the most recent counted autopsy (718a) is a
graded-MEL / non-converging-world environment test-bed; that substrate was
subsequently BUILT as SD-MEL-PRODUCER and is VALIDATED --
  ree-v3/CLAUDE.md: "SD-MEL-PRODUCER: environment.non_converging_world_rule_
    shift -- IMPLEMENTED 2026-07-21"
  REE_assembly/docs/architecture/sd_mel_producer.md: "Status: VALIDATED
    (V3-EXQ-798a, 2026-07-29; confirmed failure_autopsy_V3-EXQ-798a_2026-07-30)"
-- which is precisely the Step 2.5b release condition. The brake was already
released for this lineage for 845/861/861a on the same grounds. Additionally,
this run is NOT the class of re-queue the brake refuses: the refusal is
scoped to "a same-environment re-GRADE re-queue" (re-posing the novelty->MEL
producer question that 718a root-caused). This run does not re-pose that
question at all -- it re-poses the ALREADY-POSITIVE end-to-end result on a
disjoint seed set, which is the specific test /governance re-gated promotion
on.

SUBSTRATE-PATH OVERLAP GATE (Step 2.5c, this session): no open `corrupting`
substrate_queue entry overlaps this driver's module footprint. Three open
`degrading` entries do overlap and are recorded here for the audit trail, per
the gate's own instruction:
  SD-MECH267-CEM-SELECTION-FIX      (ree_core/hippocampal/module.py) -- touches
    the hippocampal module, which backs the C1c replay-rate DV
    (rem_n_rollouts). Arm-symmetric: identical in every arm, so it cannot bias
    the cross-arm dose-response this run routes on.
  SD-MECH303-THRESHOLD-SOURCING     (ree_core/utils/config.py,
    ree_core/environment/causal_grid_world.py, ree_core/agent.py) --
    arm-symmetric for the same reason.
  SD-QUEUE-SEED-ENFORCEMENT         (experiment_runner.py, validate_queue.py) --
    runner infrastructure, not substrate. Directly relevant to THIS run
    nonetheless, because this run's entire scientific content is its seed set:
    the queue entry's declared `seeds` (3) matches len(SEEDS) exactly, and
    every seed is printed in a "Seed S Condition C" boundary line, so an
    under-run seed set is visible in the log rather than silent.

================================================================================
DV3 IS DESCOPED -- pre-registered, and why that is scoping, not goalpost-moving
================================================================================
MECH-180 names THREE DVs. Two are testable now; the third is not.

  DV1 sws_power      -> cumulative_sws_writes        SCORED (C1a)
  DV2 replay_rate    -> cumulative_rem_rollouts      SCORED (C1c)
  DV3 spindle_density-> mean_sws_new_slot_diversity  RECORDED, NOT SCORED (C1b)

DV3's enabling substrate is NOT BUILT. substrate_queue.json entry
MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION currently reads
  status: "implemented_validation_failed_needs_followup_fix"
Its wiring landed (ree-v3 a7d36429fd) and V3-EXQ-861a confirmed the mechanism
FIRES every cycle (mean_spindle_selection_applied = 1.0), but its
selection_weight is uniformly tiny and flat (~0.004-0.01, no MEL-tracking)
because ThetaBuffer.consolidation_summary() -- the novelty reference -- is a
10-tick recency average of the SAME short window that supplies the
schema-installation prototypes being compared against it, hence structurally
self-similar (novelty ~ 0) regardless of arm. That entry's own failure_record
names the required follow-up fix (re-source consolidation_ref from a signal
correlated with the world_rule_shift/MEL axis). THAT FIX IS NOT BUILT. Running
DV3 as a gating criterion would therefore re-measure a known-unbuilt mechanism
and is guaranteed to fail, exactly as it did at 0/3 seeds in BOTH 861
(flag OFF) and 861a (flag ON).

WHY THIS MATTERS MECHANICALLY, not just presentationally: in V3-EXQ-845 and
V3-EXQ-861a the pre-registered gate was CONJUNCTIVE, C1 = C1a AND C1b AND C1c,
and the C2 control's on_gt_off leg likewise required ON-HIGH > OFF-HIGH on ALL
THREE DVs. The single known-blocked DV therefore VETOED two clean, correctly
signed, monotone positives and collapsed the whole run's self-route to
`non_contributory`. 845's own confirmed autopsy says so explicitly: "The
driver's binary C2 control gate (ALL 3 DVs must show ON>OFF) collapses the
self-route to non_contributory despite 2/3 DVs being clean and positive."
Repeating that gate here would guarantee a fourth non_contributory on a run
whose entire purpose is to produce an interpretable independence verdict.

So BOTH gates are narrowed in this run, pre-registration, before any data:
  C1 = C1a AND C1c                    (C1b computed + recorded, NOT gated)
  C2.on_gt_off = sws AND rem          (spindle leg computed + recorded, NOT gated)

This is SCOPING, not goalpost-moving, on three independent grounds:
  1. It is pre-registered in this file, against an EXTERNALLY RECORDED blocker
     (substrate_queue.json status + failure_record), before the run executes --
     not chosen after seeing DV3 fail.
  2. INV-050's own what_would_answer names ONLY sws_power and replay_rate as
     its DVs. For the PRIMARY claim, DV3 was never in scope at all.
  3. DV3's exclusion is recorded in the manifest as an explicit
     `scoring_excluded` marker with its reason and blocking sd_id, so a later
     reader (or autopsy) sees the descoping rather than inferring a two-DV
     design was always intended.

Because MECH-180's own content IS the three-DV conjunction, this run CANNOT
return "supports" for MECH-180 -- its best available per-claim direction is
"mixed" (genuine partial support, third DV not under test). That asymmetry is
deliberate and is encoded in the per-claim direction mapping below.

================================================================================
V3 PROXY MAPPING -- IDENTICAL machinery and readout code to V3-EXQ-861
================================================================================
  "slow-wave activity power"   -> sws_n_writes / cumulative_sws_writes
  "hippocampal replay rate"    -> rem_n_rollouts / cumulative_rem_rollouts
  "sleep spindle density"      -> mean_sws_new_slot_diversity (the
    write-count-DECOUPLED touched-slot statistic V3-EXQ-861 introduced:
    before/after ContextMemory.memory snapshot around force_cycle(), mean
    pairwise cosine DISTANCE among ONLY the touched slots).
    _touched_slot_diversity() below is byte-identical to 861/861a.

MECH-122 CONTENT-SELECTION FLAG IS OFF in this run
(USE_MECH122_SPINDLE_CONTENT_SELECTION = False), which makes this a clean
seed-replication of V3-EXQ-861 specifically (the write-count-deconfounded
design), rather than of 861a. 861a established that the flag leaves DV1/DV2
unchanged, so this choice does not affect the scored DVs either way; OFF is
chosen because it is the configuration whose methodology was corrected and
because it keeps this run a genuine SINGLE-variable change (seeds only) from
861. 861a's flag instrumentation fields are RETAINED and still emitted -- with
the flag off they read ~0.0, which is a useful OFF-inert confirmation.

DV-SYMMETRY (604c check), stated per arm as Step 3.5 requires. The manipulation
is world_rule_shift_interval (novelty rate); the scored DVs are
cumulative_sws_writes and cumulative_rem_rollouts.
  ARM_0_NONE_ON / ARM_1_LOW_ON / ARM_2_MED_ON / ARM_3_HIGH_ON (consumer ON):
    causal chain is shift rate -> per-step e3 prediction error -> accumulated
    waking MEL -> mel_duration_factor -> sws_consolidation_steps /
    rem_attribution_steps -> write / rollout COUNTS. The DVs are integer counts
    (cardinalities), not rank statistics and not set-aggregates over
    interchangeable units, and the manipulation enters as a MULTIPLICATIVE
    duration scale, not as a broadcast additive constant across candidates.
    So the manipulation is NOT invariant under any of the three canonical
    symmetries: broadcast-additive (the factor multiplies a count, it is not
    added uniformly across candidates and does not cancel in a difference),
    monotone-rescaling (a count is not order-only; C1 gates on relative SPREAD
    >= 15%, which a rescaling changes), or permutation-of-units (the DV is a
    total over cycles, and the manipulation changes the total, not the
    ordering of interchangeable summands).
  ARM_4_HIGH_OFF (consumer OFF): the manipulation is deliberately DISCONNECTED
    from the DV here -- that is the arm's entire purpose. With
    use_mel_consumer=False the duration factor is pinned at 1.0, so counts are
    scheduler-determined and per-cycle variance is ~0. This arm is NOT scored
    for dose-response; it is the C2 control, and its expected invariance is the
    control's positive content, not a symmetry defect. Scoping it out of C1 is
    disposition (a) "not meaningful for the regime", not (b) structural
    vacuity: the arm is legitimately informative for what it IS used for.

RESIDUAL LIMITATION, stated so a PASS is not over-read (this is what_would_
answer's lever (c) concern, and it is NOT fully discharged by this run): with
the consumer ON, mel_duration_factor is a deterministic function of measured
MEL, and the counts are a deterministic function of that factor. So C1a/C1c
tracking measured MEL is partly guaranteed by the consumer's own arithmetic
once MEL is nonzero and graded. What this run genuinely adds is that the
ECOLOGICAL chain -- environmental novelty rate -> graded above-reference waking
MEL (R2) -> graded offline counts -- holds on seeds that played no part in
establishing it. ARM_4_HIGH_OFF shows the elevation is consumer-driven rather
than raw environmental stochasticity, but it does NOT by itself separate
"genuine third-drive coupling" from "consumer arithmetic". A full discharge of
lever (c) needs a MEL-injection-decoupled arm (consumer ON, fed a reference
decoupled from the environment, while novelty still varies), which is NOT run
here and is flagged as the natural successor.

CLAIM TAGGING (Step 3.5): claim_ids = ["INV-050", "MECH-180"].
  INV-050 is tagged because its two named DVs are BOTH fully under test and the
    run is purpose-built to satisfy the non-degeneracy precondition written
    into its own what_would_answer. Its direction follows the C1a+C1c grid
    directly.
  MECH-180 is tagged because sws_power and replay_rate are TWO OF ITS OWN THREE
    named DVs -- an independent-seed replication of them is real, interpretable
    signal for this claim, not merely instrumental. But its third DV is not
    under test (above), so its direction is capped at "mixed" on the full-pass
    branch. It is NOT tagged on the strength of the descoped DV3.

PRE-REGISTERED ACCEPTANCE (evidence; claim_ids=["INV-050","MECH-180"]).
Thresholds are IDENTICAL constants to V3-EXQ-845/861/861a; only the SCORED DV
set and the seed set differ.
--------------------------------------------------------------------------
READINESS (per seed, over the four ecological ON arms) -- UNCHANGED from 861:
  R1 world-model trained: frozen-probe conv_rel_drop >= MIN_REL_CONV_DROP.
  R2 ecological novelty->MEL link holds IN THIS CONFIG: measured mean_mel is
     non-degenerately graded (mel[HIGH] >= mel[NONE] * (1+MIN_MEL_SPREAD)).
  Below-floor on either, on >= SEED_PASS_FRAC of seeds, routes to
  substrate_not_ready_requeue, NEVER an INV-050/MECH-180 verdict. This matters
  especially here: R2 failing on NEW seeds means the IV never varied on those
  seeds, which is a not-ready condition, NOT a falsification of the claim.

C1 (LOAD-BEARING, conjunctive over the TWO SCORED sub-DVs; each on the 4 ON
    arms sorted ascending by MEASURED mean_mel, per seed):
  C1a SWS power:   cumulative_sws_writes monotone non-decreasing (+ MONO_TOL)
                    and relative spread >= MIN_REL_DV_SPREAD.
  C1c replay rate: cumulative_rem_rollouts monotone non-decreasing (+ MONO_TOL)
                    and relative spread >= MIN_REL_DV_SPREAD.
  C1 = C1a AND C1c, each on >= SEED_PASS_FRAC of seeds.
  C1b spindle density: COMPUTED AND RECORDED IDENTICALLY, but NOT part of the
                    gate (see "DV3 IS DESCOPED"). Emitted with
                    scoring_excluded=True + reason + blocking sd_id.

C2 (control, non-load-bearing):
  pinned:   ARM_4_HIGH_OFF shows near-zero per-cycle variance in sws_n_writes
            and rem_n_rollouts (factor pinned at 1.0 when the consumer is off).
            Diversity is NOT in the pinning check (same reasoning as 845/861:
            it is an emergent geometric quantity, not parameter-bounded).
  on_gt_off: ARM_3_HIGH_ON exceeds ARM_4_HIGH_OFF on the TWO SCORED DVs
            (sws, rem). The spindle leg is computed and reported but NOT gated
            -- gating it would re-import the descoped DV3 through the control
            gate, which is exactly how V3-EXQ-845 lost its two clean positives.

INTERPRETATION GRID (drives both claims' directions):
  readiness unmet                     -> substrate_not_ready_requeue      (non_contributory)
  C2 fail (control not pinned / not gt)-> mel_control_degenerate           (non_contributory)
  C2 pass, C1a AND C1c pass           -> third_drive_independent_seed_replication_confirmed (PASS)
  C2 pass, neither C1a nor C1c passes -> third_drive_not_replicated_on_independent_seeds    (FAIL)
  C2 pass, exactly one passes         -> third_drive_partial_replication_independent_seeds  (FAIL)

PER-CLAIM DIRECTION:
  label                                        INV-050          MECH-180
  substrate_not_ready_requeue                  non_contributory non_contributory
  mel_control_degenerate                       non_contributory non_contributory
  ..._independent_seed_replication_confirmed   supports         mixed   (2/3 DVs only)
  ..._not_replicated_on_independent_seeds      weakens          weakens
  ..._partial_replication_independent_seeds    mixed            mixed

The `weakens` branch is deliberate and is what makes this run genuinely
falsifiable rather than a confirmation exercise: INV-050's what_would_answer
FALSIFYING clause (a) states that failure to track measured MEL under a
non-degeneracy-satisfying design means "the positive 845/861/861a signal does
not survive leaving the exact configuration it was found in", which is real
negative evidence, not a null. Note the readiness gate sits upstream of it, so
a seed set where the IV simply never varied cannot reach this branch.

PROMOTES/DEMOTES NOTHING BY ITSELF: /governance applies the verdict. On the
confirmed branch this run supplies the independent evidence GFLAG-0002's
resolution re-gated INV-050's promotion on; it does not clear MECH-180's
v3_pending, which additionally requires the descoped DV3 once the MECH-122
write-gate follow-up fix is built.
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

import numpy as np
import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_861b_inv050_mech180_independent_seed_replication"
QUEUE_ID = "V3-EXQ-861b"
CLAIM_IDS = ["INV-050", "MECH-180"]
EXPERIMENT_PURPOSE = "evidence"

# Deliberately NOT superseding V3-EXQ-861/861a: both remain valid evidence for
# their own seed set / flag condition. This is an independent replication on a
# disjoint seed set, not a correction of either.
SUPERSEDES = None

# The V3-EXQ-861 run this replicates (same flag-OFF config, same thresholds,
# same readout code; only the seed set differs) -- recorded for audit.
COMPARES_AGAINST_RUN_ID = (
    "v3_exq_861_mech180_ecological_novelty_sleep_consolidation_decoupled_diversity_20260801T205600Z_v3"
)

# -- INDEPENDENCE (the whole point of this run) -------------------------------
# Every seed ever used by this lineage, audited programmatically this session
# across all 10 manifests in REE_assembly/evidence/experiments/ carrying an
# INV-050 / MECH-180 tag, plus INV-051's V3-EXQ-901 (which shares the design).
PRIOR_LINEAGE_SEEDS: List[int] = [42, 123, 456]
PRIOR_LINEAGE_RUNS: List[str] = [
    "V3-EXQ-718", "V3-EXQ-718a", "V3-EXQ-845", "V3-EXQ-861", "V3-EXQ-861a",
    "V3-EXQ-901",
]
# Which of what_would_answer's three non-degeneracy levers this run satisfies.
NON_DEGENERACY_LEVERS_SATISFIED = ["a_new_seeds", "c_consumer_absent_control_arm"]
NON_DEGENERACY_LEVERS_NOT_SATISFIED = ["b_held_out_environment"]

# -- DV3 descoping (pre-registered; see module docstring "DV3 IS DESCOPED") ---
SCORED_DVS = ["sws_power", "replay_rate"]
UNSCORED_DVS = ["spindle_density"]
DV3_SCORING_EXCLUDED_REASON = (
    "spindle_density (mean_sws_new_slot_diversity) is RECORDED but NOT SCORED: "
    "its enabling substrate is not built. substrate_queue.json entry "
    "MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION reads status="
    "implemented_validation_failed_needs_followup_fix -- V3-EXQ-861a confirmed "
    "the content-selection mechanism fires every cycle "
    "(mean_spindle_selection_applied=1.0) but its selection_weight is uniformly "
    "flat because ThetaBuffer.consolidation_summary() is a 10-tick recency "
    "average of the same window supplying the prototypes compared against it, "
    "hence self-similar regardless of arm. The named follow-up fix (re-source "
    "consolidation_ref from a signal correlated with the world_rule_shift/MEL "
    "axis) is NOT built. Gating on it would guarantee a fourth non_contributory "
    "and veto the two clean scored DVs, exactly as the conjunctive gate did in "
    "V3-EXQ-845 and V3-EXQ-861a."
)
DV3_BLOCKING_SD_ID = "MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION"

# Touched-slot detection threshold: an EMA-blended write (0.9*old + 0.1*new)
# always moves the written row by a non-trivial amount for real content; this
# guards only against float noise from the clone/detach round-trip, not
# against genuine small updates. UNCHANGED from 861/861a.
TOUCHED_SLOT_L2_EPS = 1e-6

# MECH-122 content-packaging: OFF in this run, making it a clean
# seed-replication of V3-EXQ-861 (flag OFF) rather than 861a (flag ON). 861a
# established the flag leaves DV1/DV2 unchanged, so this does not affect the
# scored DVs; OFF keeps this a genuine single-variable change (seeds only) from
# 861. The instrumentation fields are retained and still emitted -- with the
# flag off they read ~0.0, a useful OFF-inert confirmation.
USE_MECH122_SPINDLE_CONTENT_SELECTION = False
MECH122_SPINDLE_SELECTION_GAIN = 1.0   # inert while the flag is False

# z_goal_enabled=True is inherited verbatim from the V3-EXQ-718a/798a/845/861
# lineage for architecture parity, but no code path here calls update_z_goal --
# the stream is inert, identically, in every arm (goal_weight /
# e1_goal_conditioned are structural config, not exercised by this driver).
# Wiring it live would activate the E3 goal term + E1 goal-conditioning and the
# SD-024 benefit-attractor producer, changing action selection and hence the
# trajectories the MEL/DV readouts are computed over -- a behaviour change that
# would make this run non-comparable to the V3-EXQ-861 run it is a
# seed-replication OF, destroying the single-variable property that is this
# run's entire scientific value. The knob is arm-symmetric, so the inertness
# cannot bias the ON-vs-OFF or cross-arm comparisons this experiment routes on.
DEAD_Z_GOAL_STREAM_EXEMPT = (
    "inherited verbatim from V3-EXQ-718a/798a/845/861 for architecture parity; "
    "wiring update_z_goal would activate the E3 goal term, E1 conditioning, and "
    "the SD-024 benefit-attractor producer, breaking the single-variable "
    "(seeds-only) comparison against V3-EXQ-861 that this replication depends "
    "on. Knob is arm-symmetric (identical in every arm)."
)

# -- Design parameters (IDENTICAL to V3-EXQ-845/861/861a except SEEDS) --------
SEEDS = [7, 271, 883]         # DISJOINT from PRIOR_LINEAGE_SEEDS -- the manipulation
CONV_EPISODES = 60            # P0 world-model convergence on the STABLE base env
STEPS_PER_EPISODE = 90
PROBE_BATTERY_SIZE = 64       # FIXED held-out probe battery (frozen-encoder)
CALIB_EPISODES = 3            # stable-base MEL reference-calibration wake pass
MEAS_CYCLES = 6               # wake-sleep cycles per arm
WAKE_EPISODES_PER_CYCLE = 2   # wake episodes per cycle (populate buffers + MEL)
# Progress denominator M (per cell): P0 + calibration + measurement wake episodes.
EPISODES_PER_RUN = (CONV_EPISODES + CALIB_EPISODES
                    + MEAS_CYCLES * WAKE_EPISODES_PER_CYCLE)

# Sleep pass base durations (the scheduler-pinned counts V3-EXQ-677 measured;
# same base as V3-EXQ-718/718a/845/861 for lineage comparability).
SWS_CONSOLIDATION_STEPS = 5
REM_ATTRIBUTION_STEPS = 10

# MEL consumer config (identical to the validated V3-EXQ-718a/845/861 test-bed).
MEL_GAIN = 1.0
FACTOR_MIN = 0.5
FACTOR_MAX = 3.0
MEL_RELATIVE_FLOOR = 1e-6     # relative floor only guards mel/ref against ref ~ 0

# E2 world-forward online training (recon-only; SD-056 auxiliary OFF at train time).
SD056_WEIGHT = 0.05
E2_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
TRANSITION_BUFFER_MAX = 256

# -- Thresholds (pre-registered constants, NOT derived from run stats) -------
# Byte-identical to V3-EXQ-845/861/861a. Deliberately NOT re-tuned: a
# replication that moved its thresholds would not be a replication.
MIN_REL_CONV_DROP = 0.10      # R1: per-seed frozen-probe PE drops at least 10%
SEED_PASS_FRAC = 2.0 / 3.0    # R / C1 / C2: at least 2/3 of seeds
MIN_MEL_SPREAD = 0.15         # R2: measured mean_mel[max] at least 15% above [min]
MIN_REL_DV_SPREAD = 0.15      # C1: DV[max] at least 15% above DV[min]
MONO_TOL = 0.05               # C1: monotonicity slack (relative to DV[min])
PINNED_ABS_VAR_ATOL = 1e-6    # C2: OFF-arm per-cycle count variance ceiling

# -- Environment base (identical to V3-EXQ-798a / 718a / 701c / 845 / 861) ---
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

# The stable base carries NO hazard drift, so the only non-stationarity in the
# graded arms is the SD-MEL-PRODUCER rule shift itself (per its own docstring).
STABLE_DRIFT = dict(env_drift_interval=999, env_drift_prob=0.0)
WORLD_RULE_SHIFT_DEPTH = 2    # matches V3-EXQ-798a's validated ladder

# arm_id, novelty level (world_rule_shift_interval; 798a's validated ladder),
# consumer on/off. IDENTICAL to V3-EXQ-845/861/861a.
ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_NONE_ON",  "level": 0, "interval": 0,  "mel_on": True},
    {"arm_id": "ARM_1_LOW_ON",   "level": 1, "interval": 60, "mel_on": True},
    {"arm_id": "ARM_2_MED_ON",   "level": 2, "interval": 25, "mel_on": True},
    {"arm_id": "ARM_3_HIGH_ON",  "level": 3, "interval": 10, "mel_on": True},
    {"arm_id": "ARM_4_HIGH_OFF", "level": 3, "interval": 10, "mel_on": False},
]
# The 4 ecological ON arms (C1 is scored over these, sorted by MEASURED mean_mel).
ON_ECO_ARMS = ["ARM_0_NONE_ON", "ARM_1_LOW_ON", "ARM_2_MED_ON", "ARM_3_HIGH_ON"]


def seed_overlap_with_prior_lineage(seeds: List[int]) -> List[int]:
    """The independence manipulation, made machine-checkable.

    Returns the sorted intersection of this run's seed set with every seed the
    845/861/861a (and 718/718a/901) lineage has ever used. An EMPTY list is the
    non-degeneracy precondition INV-050's what_would_answer lever (a) requires:
    "seeds genuinely uninvolved in 845/861/861a (not a superset that merely
    adds more)". Emitted as a precondition so the claim is verifiable from the
    manifest alone rather than asserted in prose."""
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
    """Converged-base agent (recon-only e2 training; encoder frozen) + SD-017
    SWS/REM passes + the SleepLoopManager. When mel_on, the SD-MEL-CONSUMER is
    enabled with a FIXED reference set-point.

    Config is byte-identical to the validated V3-EXQ-861 recipe --
    use_mech122_spindle_content_selection is False here, which is 861's own
    value (861a is the run that set it True). sd016_writepath_mode is left at
    its "off" default and shy_enabled at its False default: both are the
    preconditions that make ContextMemory.memory touched ONLY by
    run_sws_schema_pass's write() loop during a measurement cycle (861's module
    docstring point (c), unchanged here)."""
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
        # SD-017 sleep passes + SleepLoopManager (no aggregation cluster needed).
        use_sleep_loop=True,
        sleep_loop_episodes_K=10**9,   # never auto-fire; we drive via force_cycle
        sws_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        # MECH-122 content-packaging: OFF (861's value). Retained explicitly
        # rather than omitted so the manifest config records the condition.
        use_mech122_spindle_content_selection=USE_MECH122_SPINDLE_CONTENT_SELECTION,
        mech122_spindle_selection_gain=MECH122_SPINDLE_SELECTION_GAIN,
        rem_enabled=True,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        # SD-MEL-CONSUMER (GAP-5b) -- fixed reference set-point.
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
    """One recon-only P0 world-forward training step (reconstruction MSE)."""
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
    """One waking step. Always calls agent.update_residue() (hypothesis_tag=False)
    so the MEL consumer accumulates per-step e3 prediction error. Returns
    (next_obs_dict, done)."""
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
    """Run n_episodes of waking on env. During P0 (train=True) trains e2 recon-only.
    During measurement (train=False) just drives the agent + accumulates MEL + warms
    the agent's experience buffers."""
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


# -- FROZEN held-out probe battery (the V3-EXQ-701b/c convergence instrument) --
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
    """Mean one-step world_forward reconstruction error over the FIXED battery.
    The world-model convergence metric (readiness only)."""
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
    """V3-EXQ-861's write-count-decoupled statistic -- byte-identical to 861/861a.

    Given a before/after snapshot of ContextMemory.memory around ONE
    force_cycle() call, identify the rows (slots) whose per-row L2 diff exceeds
    eps -- i.e. exactly the slots run_sws_schema_pass's write() loop wrote to
    this cycle.

    Returns (diversity, n_touched, insufficient) where diversity is the mean
    pairwise cosine DISTANCE among ONLY the touched rows, n_touched is the count
    of distinct touched slots, and insufficient is True when n_touched < 2.

    NOTE: this backs the DESCOPED DV3 in this run -- computed and recorded
    identically to 861, but NOT part of any pass gate (see module docstring
    "DV3 IS DESCOPED")."""
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


# z_goal-stream liveness, pooled across the run's per-cell agents.
_ZG = ZGoalStreamAccumulator()
# ONE representative agent, overwritten each cell, so write_flat_manifest can
# record `enabled_default_off_flags` off a live config without pinning every
# (seed x arm) agent's nets/optimiser/buffers in memory until the run ends.
# z_goal_stream_stats takes precedence over `agent` for the z_goal block
# (manifest_core.py), so the pooled accumulator above still supplies that.
_LAST_AGENT: List[Optional[REEAgent]] = [None]


def _run_cell(seed: int, arm: Dict[str, Any], steps: int, conv_eps: int,
              meas_cycles: int) -> Dict[str, Any]:
    """One (seed, arm) cell: build agent, converge P0 recon-only on the stable
    base, calibrate the MEL reference to the stable-base measurement-modality
    MEL, then run the wake-sleep measurement cycles on the arm's
    SD-MEL-PRODUCER env (world_rule_shift, not env_drift). Cell logic is
    IDENTICAL to V3-EXQ-861's."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    arm_id = arm["arm_id"]
    mel_on = bool(arm["mel_on"])
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # ONE agent per cell. Encoder FROZEN (only agent.e2 trains). Build with
    # mel_reference=0.0, then fix it to the calibrated stable-base MEL AFTER
    # P0 (mode="fixed", so the >0 reference is used and never auto-relocked).
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

    # Reference calibration: a stable-base wake pass measured through the SAME
    # path as the measurement MEL (agent.update_residue -> e3 PE -> consumer
    # accumulator). NONE arm (stable env, no shift) then yields factor ~1.0;
    # higher-shift-rate arms scale above it.
    if agent.mel_consumer is not None:
        agent.mel_consumer.reset()   # clear P0 accumulation
    _run_wake_window(
        agent, stable_env, CALIB_EPISODES, steps, train=False, buffer=None,
        e2_opt=None, sample_rng=None, ep_offset=conv_eps, arm_id=arm_id, seed=seed,
    )
    if mel_on and agent.mel_consumer is not None:
        base_ref = float(agent.mel_consumer.current_mel())
        if not (base_ref > 0.0):     # degenerate fallback (no PE accumulated)
            base_ref = float(probe_pe_final)
        agent.config.mel_reference = base_ref
        agent.mel_consumer.config.mel_reference = base_ref
        agent.mel_consumer.reset()   # clean slate for the first measurement cycle
    else:
        base_ref = float(probe_pe_final)   # OFF arm: reference unused

    # Measurement: MEAS_CYCLES wake-sleep cycles on the arm's SD-MEL-PRODUCER
    # env (world_rule_shift is the ONLY non-stationarity; env_drift stays at
    # the STABLE_DRIFT sentinel in every arm).
    meas_env = _make_env(seed, arm["interval"])
    cum_sws = 0.0
    cum_rem = 0.0
    per_cycle_sws: List[float] = []
    per_cycle_rem: List[float] = []
    per_cycle_diversity_legacy: List[float] = []      # whole-bank, non-gating
    per_cycle_new_diversity: List[float] = []         # 861 statistic; DESCOPED here
    per_cycle_n_touched: List[int] = []
    n_cycles_insufficient_touched = 0
    factors: List[float] = []
    mels: List[float] = []
    # MECH-122 content-packaging instrumentation (retained from 861a). With the
    # flag OFF these should read ~0.0 every cycle -- a useful OFF-inert
    # confirmation, and the direct cross-check against 861a's flag-ON values.
    per_cycle_spindle_selection_applied: List[float] = []
    per_cycle_spindle_selection_mean_weight: List[float] = []
    ep_off = conv_eps + CALIB_EPISODES
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

    # Mean over cycles that had >= 2 touched slots (an "insufficient" cycle
    # contributes no data point rather than a spurious 0.0 -- a 0.0 fallback
    # baked into the mean would itself be a write-count-correlated artifact
    # for low-MEL arms, exactly what 861's redesign exists to avoid).
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
          f"ref={base_ref:.3e} mel={mean_mel:.3e} factor={mean_factor:.3f} "
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
        "level": arm["level"],
        "mel_on": mel_on,
        "world_rule_shift_interval": arm["interval"],
        "seed": seed,
        "conv_rel_drop": conv_rel_drop,
        "probe_pe_init": probe_pe_init,
        "probe_pe_final": probe_pe_final,
        "mel_reference": base_ref,
        "mean_mel": mean_mel,
        "mean_duration_factor": mean_factor,
        # -- SCORED DVs --
        "cumulative_sws_writes": cum_sws,
        "cumulative_rem_rollouts": cum_rem,
        # -- DESCOPED DV3 (recorded, never gated) --
        "mean_sws_new_slot_diversity": mean_new_diversity,
        "per_cycle_new_diversity": per_cycle_new_diversity,
        "per_cycle_n_touched_slots": per_cycle_n_touched,
        "n_cycles_insufficient_touched_slots": n_cycles_insufficient_touched,
        "new_diversity_variance": new_diversity_var,
        "mean_sws_slot_diversity_wholebank_legacy": mean_diversity_legacy,
        "per_cycle_diversity_wholebank_legacy": per_cycle_diversity_legacy,
        "diversity_variance_wholebank_legacy": diversity_var_legacy,
        # -- per-cycle detail (generous recording) --
        "per_cycle_sws": per_cycle_sws,
        "per_cycle_rem": per_cycle_rem,
        "per_cycle_mel": mels,
        "per_cycle_factor": factors,
        "sws_count_variance": sws_count_var,
        "rem_count_variance": rem_count_var,
        "meas_cycles": meas_cycles,
        # MECH-122 instrumentation (flag OFF here -- expect ~0.0; retained for
        # direct comparison against 861a's flag-ON values).
        "mean_spindle_selection_applied": mean_spindle_selection_applied,
        "mean_spindle_selection_weight": mean_spindle_selection_weight,
        "per_cycle_spindle_selection_applied": per_cycle_spindle_selection_applied,
        "per_cycle_spindle_selection_mean_weight": per_cycle_spindle_selection_mean_weight,
    }


def _dv_dose_response(values: List[float]) -> Dict[str, Any]:
    """Given a DV in ascending-measured-MEL order, test monotone non-decreasing
    (+ tol) and relative spread. Shared by C1a/C1b/C1c. UNCHANGED from 845/861."""
    if len(values) < 2:
        return {"monotone_ok": False, "spread_ok": False, "pass_": False}
    lo = values[0]
    tol = MONO_TOL * max(lo, 1e-9)
    monotone_ok = all(values[i] <= values[i + 1] + tol for i in range(len(values) - 1))
    spread_ok = lo > 0 and values[-1] >= lo * (1 + MIN_REL_DV_SPREAD)
    return {"monotone_ok": bool(monotone_ok), "spread_ok": bool(spread_ok),
            "pass_": bool(monotone_ok and spread_ok)}


def _seed_readiness(on_eco_cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """R1 (world-model trained) AND R2 (ecological novelty->MEL link holds in
    THIS run's config, on THIS run's seeds) for one seed's 4 ecological ON arms.
    UNCHANGED from 845/861."""
    if len(on_eco_cells) != len(ON_ECO_ARMS):
        return {"r1_ok": False, "r2_ok": False, "ready": False}
    r1_ok = all(r["conv_rel_drop"] >= MIN_REL_CONV_DROP for r in on_eco_cells)
    arms_sorted = sorted(on_eco_cells, key=lambda r: r["mean_mel"])
    mels = [r["mean_mel"] for r in arms_sorted]
    r2_ok = mels[0] > 0 and mels[-1] >= mels[0] * (1 + MIN_MEL_SPREAD)
    return {"r1_ok": bool(r1_ok), "r2_ok": bool(r2_ok), "ready": bool(r1_ok and r2_ok)}


def _seed_c1(on_eco_by_arm: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """C1a/C1c (SCORED) and C1b (RECORDED, NOT SCORED) per seed on the 4
    ecological ON arms, sorted ascending by MEASURED mean_mel.

    The gate is C1a AND C1c only. C1b is computed identically to 861 and
    reported with an explicit scoring_excluded marker -- including it in the
    conjunction is precisely the defect that collapsed V3-EXQ-845 and
    V3-EXQ-861a to non_contributory despite two clean positives."""
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
    """C2 per seed: OFF control's step-bounded counts pinned (near-zero
    variance) AND ON-HIGH exceeds OFF-HIGH on the TWO SCORED DVs.

    The spindle-density leg is COMPUTED AND REPORTED but deliberately NOT part
    of the on_gt_off gate -- gating it would re-import the descoped DV3 through
    the control gate. failure_autopsy_V3-EXQ-845_2026-08-01: "The driver's
    binary C2 control gate (ALL 3 DVs must show ON>OFF) collapses the self-route
    to non_contributory despite 2/3 DVs being clean and positive." """
    pinned_ok = (off_cell["sws_count_variance"] <= PINNED_ABS_VAR_ATOL
                 and off_cell["rem_count_variance"] <= PINNED_ABS_VAR_ATOL)
    on_high = on_eco_by_arm["ARM_3_HIGH_ON"]
    on_gt_off_sws = on_high["cumulative_sws_writes"] > off_cell["cumulative_sws_writes"]
    on_gt_off_rem = on_high["cumulative_rem_rollouts"] > off_cell["cumulative_rem_rollouts"]
    on_gt_off = bool(on_gt_off_sws and on_gt_off_rem)
    # Reported only -- never gated.
    on_gt_off_spindle = (on_high["mean_sws_new_slot_diversity"]
                         > off_cell["mean_sws_new_slot_diversity"])
    return {
        "pinned_ok": bool(pinned_ok),
        "on_gt_off": on_gt_off,
        "on_gt_off_sws": bool(on_gt_off_sws),
        "on_gt_off_rem": bool(on_gt_off_rem),
        "on_gt_off_spindle_reported_only": bool(on_gt_off_spindle),
        "c2_pass": bool(pinned_ok and on_gt_off),
        "off_diversity_variance_reported_only": off_cell["new_diversity_variance"],
        "off_diversity_variance_wholebank_legacy_reported_only":
            off_cell["diversity_variance_wholebank_legacy"],
    }


def run_experiment(steps: int, conv_eps: int, meas_cycles: int,
                   seeds: List[int], arms: Optional[List[Dict[str, Any]]] = None,
                   ) -> Dict[str, Any]:
    arms = arms if arms is not None else ARMS
    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            full_config = {
                "env_base": ENV_BASE,
                "arm": arm,
                "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH,
                "conv_episodes": conv_eps,
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
            }
            with arm_cell(seed, config_slice=full_config,
                          script_path=Path(__file__)) as cell:
                row = _run_cell(seed, arm, steps, conv_eps, meas_cycles)
                cell.stamp(row)
            arm_results.append(row)

    # -- Readiness (UNCHANGED from 845/861) --
    seed_ready: Dict[int, bool] = {}
    seed_readiness_detail: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        on_eco_cells = [r for r in arm_results
                        if r["seed"] == seed and r["mel_on"]]
        rd = _seed_readiness(on_eco_cells)
        seed_readiness_detail[seed] = rd
        seed_ready[seed] = rd["ready"]
    readiness_frac = sum(seed_ready.values()) / max(1, len(seeds))
    r1_frac = sum(1 for s in seeds if seed_readiness_detail[s]["r1_ok"]) / max(1, len(seeds))
    r2_frac = sum(1 for s in seeds if seed_readiness_detail[s]["r2_ok"]) / max(1, len(seeds))
    readiness_ok = readiness_frac >= SEED_PASS_FRAC

    # -- C1 / C2 per seed, on ready seeds only --
    c1_seed_pass = 0
    c2_seed_pass = 0
    per_dv_seed_pass = {"sws_power": 0, "spindle_density": 0, "replay_rate": 0}
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        on_eco = {r["arm_id"]: r for r in arm_results
                  if r["seed"] == seed and r["mel_on"]}
        off_cell = next((r for r in arm_results
                         if r["seed"] == seed and r["arm_id"] == "ARM_4_HIGH_OFF"), None)
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
                per_dv_seed_pass["spindle_density"] += 1   # recorded, not gated
            if c1["c1c_replay_rate"]["pass_"]:
                per_dv_seed_pass["replay_rate"] += 1
            if c1["c1_scored_all_pass"]:
                c1_seed_pass += 1
            if c2["c2_pass"]:
                c2_seed_pass += 1
        else:
            rec.update({"c1": None, "c2": None, "skipped": "not_ready_or_missing_arms"})
        per_seed.append(rec)

    n_seeds = max(1, len(seeds))
    c1_frac = c1_seed_pass / n_seeds
    c2_frac = c2_seed_pass / n_seeds
    c1_all_pass = c1_frac >= SEED_PASS_FRAC
    c2_pass = c2_frac >= SEED_PASS_FRAC
    per_dv_frac = {k: v / n_seeds for k, v in per_dv_seed_pass.items()}
    per_dv_pass = {k: (v >= SEED_PASS_FRAC) for k, v in per_dv_frac.items()}
    # The gate counts ONLY the scored DVs.
    n_scored_dv_pass = sum(1 for k in SCORED_DVS if per_dv_pass[k])

    # -- Independence audit (this run's actual manipulation) --
    overlap = seed_overlap_with_prior_lineage(seeds)
    seeds_disjoint = (len(overlap) == 0)

    # -- Self-route --
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        inv050_direction = "non_contributory"
        mech180_direction = "non_contributory"
    elif not c2_pass:
        label = "mel_control_degenerate"
        outcome = "FAIL"
        inv050_direction = "non_contributory"
        mech180_direction = "non_contributory"
    elif n_scored_dv_pass == 2:
        label = "third_drive_independent_seed_replication_confirmed"
        outcome = "PASS"
        inv050_direction = "supports"
        # Capped at mixed: 2 of MECH-180's 3 named DVs are under test.
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

    # Summary direction across both claims.
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
                "description": "THE MANIPULATION. This run's seed set must share "
                               "NO seed with any prior INV-050/MECH-180 run "
                               "(718/718a/845/861/861a, and INV-051's 901), all of "
                               "which used exactly {42,123,456}. A non-empty "
                               "intersection would make this another "
                               "same-configuration re-run rather than the "
                               "independent replication GFLAG-0002 re-gated "
                               "promotion on. Measured = size of the intersection; "
                               "must be 0.",
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
                               "ecological ON arms, on >= 2/3 seeds -- the world "
                               "model must be trained for MEL to live at a "
                               "meaningful scale (R1).",
                "measured": r1_frac,
                "threshold": SEED_PASS_FRAC,
                "direction": "lower",
                "met": bool(r1_frac >= SEED_PASS_FRAC),
            },
            {
                "name": "ecological_novelty_mel_gradient_present_this_config",
                "description": "measured mean_mel is non-degenerately graded "
                               "(mel[HIGH] >= mel[NONE]*(1+MIN_MEL_SPREAD)) across "
                               "the 4 ON arms, on >= 2/3 seeds -- re-confirms the "
                               "V3-EXQ-798a/845 novelty->MEL producer link ON THE "
                               "NEW SEEDS. Below-floor means the IV never varied on "
                               "this seed set (not-ready), NOT an INV-050/MECH-180 "
                               "falsification (R2). This precondition sits UPSTREAM "
                               "of the weakens branch precisely so a seed set with "
                               "no IV variation cannot produce negative evidence.",
                "measured": r2_frac,
                "threshold": SEED_PASS_FRAC,
                "direction": "lower",
                "met": bool(r2_frac >= SEED_PASS_FRAC),
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
            "limitation": "Environment-independence is NOT established: "
                          "SD-MEL-PRODUCER was built and validated (V3-EXQ-798a) "
                          "against this same CausalGridWorldV2 instance, and no "
                          "held-out environment exists. A PASS here establishes "
                          "SEED-independence only.",
            "residual_lever_c_note": "ARM_4_HIGH_OFF shows the elevation is "
                          "consumer-driven rather than environmental stochasticity, "
                          "but does NOT separate genuine third-drive coupling from "
                          "the consumer's own deterministic MEL->duration "
                          "arithmetic. Full discharge needs a MEL-injection-"
                          "decoupled arm (consumer ON, reference decoupled from the "
                          "environment, novelty still varying), NOT run here.",
        },
        "replication_note": "Single-variable replication of V3-EXQ-861: identical "
                            "env, arms, thresholds, readout code and agent config "
                            "(use_mech122_spindle_content_selection=False, 861's own "
                            "value); ONLY the seed set differs. Thresholds were "
                            "deliberately NOT re-tuned -- a replication that moved "
                            "its thresholds would not be a replication.",
    }
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
        {"name": "C2_consumer_not_env_causes_variation", "load_bearing": False,
         "passed": bool(c2_pass)},
        {"name": "SEEDS_disjoint_from_prior_lineage", "load_bearing": True,
         "passed": bool(seeds_disjoint)},
    ]
    combination_rule = (
        "PASS iff readiness_ok AND c2_pass AND (C1a AND C1c) each on >= 2/3 seeds. "
        "C1b (spindle_density) is COMPUTED AND RECORDED but EXCLUDED from the "
        "conjunction and from C2's on_gt_off leg -- see interpretation."
        "dv3_scoring_excluded_reason. The gate is a 2-DV conjunction, NOT the 3-DV "
        "conjunction used by V3-EXQ-845/861/861a."
    )

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {
            "INV-050": inv050_direction,
            "MECH-180": mech180_direction,
        },
        "interpretation": interpretation,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "readiness_ok": readiness_ok,
        "readiness_frac": readiness_frac,
        "r1_frac": r1_frac,
        "r2_frac": r2_frac,
        "c1_all_pass": c1_all_pass, "c1_frac": c1_frac,
        "c2_pass": c2_pass, "c2_frac": c2_frac,
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
                                "V3-EXQ-861/861a/845)",
        "timestamp_utc": ts,
        "seeds": SEEDS,
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
        "c1_all_pass": result["c1_all_pass"], "c1_frac": result["c1_frac"],
        "c2_pass": result["c2_pass"], "c2_frac": result["c2_frac"],
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
        # Pooled z_goal liveness across every per-cell agent; takes precedence
        # over `agent` for the z_goal block. `agent` additionally supplies the
        # enabled_default_off_flags drift-detection record.
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
        seeds = [SEEDS[0]]
        # Smoke subset: exercise every distinct code path (ecological ON +
        # calibration, plus OFF pinning) with 3 agent builds instead of 5.
        smoke_ids = {"ARM_0_NONE_ON", "ARM_3_HIGH_ON", "ARM_4_HIGH_OFF"}
        arms = [a for a in ARMS if a["arm_id"] in smoke_ids]
    else:
        steps = STEPS_PER_EPISODE
        conv_eps = CONV_EPISODES
        meas_cycles = MEAS_CYCLES
        seeds = SEEDS
        arms = ARMS

    result = run_experiment(steps, conv_eps, meas_cycles, seeds, arms)
    out_path = write_manifest(result, dry_run=bool(args.dry_run), started_at=t0)
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"seeds={seeds} disjoint_from_prior_lineage="
          f"{result['seeds_disjoint_from_prior_lineage']} "
          f"overlap={result['seed_overlap_with_prior_lineage']}", flush=True)
    print(f"readiness_frac={result['readiness_frac']:.2f} "
          f"(r1={result['r1_frac']:.2f} r2={result['r2_frac']:.2f}) "
          f"c1_frac={result['c1_frac']:.2f} c2_frac={result['c2_frac']:.2f} "
          f"n_scored_dv_pass={result['n_scored_dv_pass']}/2 "
          f"per_dv={result['per_dv_frac']}", flush=True)
    print(f"evidence_direction_per_claim={result['evidence_direction_per_claim']}",
          flush=True)
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
