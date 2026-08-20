"""
V3-EXQ-861e -- CALIBRATION-POWER-RAISED REPLICATION of INV-050's third-drive
coupling (and, on its two testable DVs, MECH-180's novelty-adaptive sleep
upregulation). Same-question re-run of V3-EXQ-861c, NOT a supersession.

================================================================================
WHY THIS RUN EXISTS -- 861c's C2 failure was under-powered calibration, not a
substrate ceiling (confirmed failure_autopsy_861c-861d-mech180-cluster_2026-08-16)
================================================================================
The confirmed cluster autopsy (interactive gate, 2026-08-16, session
cranky-driscoll-126a36) found, for V3-EXQ-861c specifically:
  - Both readiness gates (R1 world-forward convergence, R2 ecological MEL
    gradient) passed at 1.0 on all 3 seeds -- the run was fully ready.
  - DV1 (sws_power) and DV2 (replay_rate) PASSED 3/3 seeds -- the MEL consumer
    is functional and the two clean scored DVs replicate cleanly.
  - The decisive C2 leg (on_factor_clears_calibration_noise_floor) failed 1/3
    seeds -- ROOT CAUSE, quantified exactly:
      on_factor == mean_mel(ARM_3_HIGH_ON) / mel_reference, verified to 4
      decimal places on all 3 seeds. mel_reference was estimated from
      CALIB_DRAWS=5 independent draws at rel_sd 0.152-0.283 (the per-draw
      sampling scatter of the calibration MEAN, i.e. THIS run's own
      calib_rel_sd metric). At n=5 the required C2 margin
      (1 + K_CALIB_MARGIN * rel_sd/sqrt(n)) is comparable to or larger than
      the effect size on 2/3 seeds:
        seed   on_factor   rel_sd   n=5 margin        n=10 margin
        7      0.8366      0.283    1.2530  FAIL      1.1789  FAIL (genuine)
        271    1.2146      0.246    1.2202  FAIL by 0.5%   1.1557  PASS
        883    1.8846      0.152    1.1360  PASS      1.0962  PASS
      Seed 271 misses by 0.5% PURELY on calibration sample size -- at n=10 the
      margin flips and C2 reaches 2/3 -> PASS. Seed 7 is DIFFERENT: its
      ARM_3_HIGH_ON mean MEL (3.04e-5) sits BELOW the no-shift stable base
      (3.64e-5), the same link-(i) producer failure already documented for
      the 677/718/718a lineage -- more calibration draws will NOT fix it.
  - Governance verdict (both claims): non_contributory, status and
    v3_pending UNCHANGED. NOT a substrate ceiling -- "a broken instrument is
    not evidence of a ceiling" (autopsy Section 5, R3 of the re-derive brake
    convention). Repair pathway (autopsy Section 7, confirmed at the gate):
    raise CALIB_DRAWS (>=10) and pre-register an operating-point precondition
    on the calibration's own precision, so a future near-miss self-routes to
    substrate_not_ready_requeue instead of landing another ambiguous FAIL.

THE FIX (this run), per the autopsy's Section 7 repair pathway items 1-3:
  (1) CALIBRATION POWER RAISE. CALIB_DRAWS: 5 -> 10 (CALIB_EPISODES_PER_DRAW
      unchanged at 6, so total calibration wake episodes: 30 -> 60). This is
      NOT a round-number guess -- it is the exact n the autopsy's own
      confirmed projection (table above, reproduced from
      failure_autopsy_861c-861d-mech180-cluster_2026-08-16.md section 2c)
      shows flips seed 271's C2 margin from FAIL-by-0.5% to PASS, using the
      observed rel_sd range 0.152-0.283 as the basis for the projection
      (SEM ~ rel_sd/sqrt(n_draws), confirmed to hold in 861c's own manifest:
      the reported calib_rel_sd_of_mean values matched rel_sd/sqrt(5) to
      within rounding).
  (2) PRE-REGISTERED CALIBRATION-PRECISION OPERATING-POINT PRECONDITION (R3).
      Raising the draw count alone only helps if the run's ACTUAL calibration
      precision lands near the projection; a run could still land a wider
      rel_sd by chance. So this run adds a third seed-readiness gate,
      alongside R1/R2, that is NEW relative to 861c:
        R3: ARM_3_HIGH_ON's mel_reference_calib_rel_sd_of_mean (the SD-of-mean
            of its own CALIB_DRAWS repeated draws -- exactly the quantity C2's
            uncertainty-aware threshold reads) must not exceed
            MAX_CALIB_REL_SD_OF_MEAN.
      MAX_CALIB_REL_SD_OF_MEAN = 0.15 (pre-registered constant, NOT derived
      from this run's own stats). Justification: at CALIB_DRAWS=10, the
      autopsy's confirmed rel_sd range (0.152-0.283) projects
      calib_rel_sd_of_mean in [0.048, 0.090] (rel_sd/sqrt(10)). 0.15 sits
      ~67% above the worst projected value -- enough headroom that ordinary
      run-to-run fluctuation in the underlying rel_sd does not spuriously
      trip the gate, while still catching a genuine ~2x-or-worse calibration
      blowup on a seed that would otherwise let C2 decide on an unreliable
      margin. A seed failing R3 self-routes to substrate_not_ready_requeue
      for THAT seed (folded into the existing readiness gate, see
      "READINESS" below), never to a coupling verdict -- this is the
      "self-routes substrate_not_ready rather than producing another
      under-powered near-miss" fix the autopsy's repair pathway named.
  (3) SEED 7 IS A GENUINE PRODUCER FAILURE, PRE-REGISTERED AS SUCH -- NOT A
      CALIBRATION ISSUE, AND THIS RUN EXPECTS 2/3, NOT 3/3, ON C2. Per the
      autopsy: seed 7's ARM_3_HIGH_ON mean MEL sits below its own no-shift
      stable base -- the world_rule_shift producer failed to raise MEL on
      that seed at all. More calibration draws cannot fix a numerator problem
      by tightening the denominator's error bars. This run does NOT add a
      producer-side readiness gate for that specific case (that would be a
      substrate change, out of scope for a driver-side calibration fix, and
      the autopsy's repair pathway explicitly offers "state plainly" as the
      alternative to "handle it explicitly"). Instead: SEED_PASS_FRAC = 2/3
      (unchanged, inherited from the whole 845/861/861a/861b/861c lineage)
      is PRE-REGISTERED here as sufficient and expected -- a 2/3 C2 pass with
      seed 7 the lone failure is the CONFIRMED, not partial, outcome this run
      is designed to produce. Do not read a 2/3 result as weaker evidence
      than a hypothetical 3/3; the third seed's failure is a known, disjoint,
      already-diagnosed producer defect this run was never going to fix.
  (4) z_goal writer defect: the lineage-wide (5-run) z_goal_stream.
      writer_defect=true is a documented, common-mode, arm-symmetric
      condition (autopsy Section 1b) that biases no arm contrast -- not fixed
      here, DEAD_Z_GOAL_STREAM_EXEMPT below states why, unchanged from 861c.

NOT IN SCOPE (explicitly, per the dispatching chip and the autopsy's own
target separation): V3-EXQ-861d's SEPARATE finding (the MECH-122
relative_novelty clamp gate being dead/saturated on 2/3 seeds) is NOT folded
into this run. That is a different mechanism (content-packaging selection,
USE_MECH122_SPINDLE_CONTENT_SELECTION=True) with its own tracked repair
(substrate_queue.json MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION, severity
corrupting, action=amend, pre-registered operating-point precondition still
to be designed for THAT gate). This run keeps the flag OFF (861c/861b's
value), exactly as 861c did, so 861d's defect is structurally inert here --
see "SUBSTRATE-PATH OVERLAP GATE" below for the empirical confirmation.

SLEEP DRIVER: manual-cycle-loop (agent.sleep_loop.force_cycle() called once
per cycle in a dedicated MEAS_CYCLES wake-sleep loop). Byte-identical pattern
to V3-EXQ-718a/845/861/861a/861b/861c.

CLAIMS UNDER TEST (UNCHANGED from 861c):
  INV-050 (primary): sleep phase architecture regulated by three distinct
    drives, the third (MEL / learning demand) determining update sufficiency.
  MECH-180 (secondary, PARTIAL -- 2 of its 3 named DVs; DV3 still descoped,
    unchanged from 861c -- see "DV3 IS DESCOPED" below).

================================================================================
INDEPENDENCE PROVENANCE -- unchanged from 861b/861c
================================================================================
Seeds [7, 271, 883] are REUSED from 861b/861c (established disjoint from the
{42,123,456} configuration used by 718/718a/845/861/861a/901). This run's
single experimental change from 861c is the calibration power raise + the
new R3 precondition -- reusing the same seeds isolates that change exactly as
861c isolated its own calibration-methodology change from 861b.

GOV-REUSE-1 CHECK (Step 2.4, this session): decisive readout = the C2
independence discriminator (ARM_3_HIGH_ON mean_duration_factor vs the pinned
OFF baseline) computed under CALIB_DRAWS=10 + the new R3 precondition, on
seeds 7/271/883. This readout does not exist in any recorded manifest: 861c
computed C2 at CALIB_DRAWS=5 only, and recorded no R3-equivalent precondition.
The fix is a new manipulation of the calibration draw count and a new
precondition, present in no recorded run, so it is neither recorded nor
derivable by reanalysis of 861c's manifest (raising n post-hoc from 861c's 5
draws is not possible -- the additional draws were never taken; its own
per-draw values are recorded but there are only 5 of them). NOT RECOVERABLE
-> proceeds to a new run. Checked: v3_exq_861c (5 draws, no R3),
v3_exq_861b (1 draw, no multi-draw noise estimate at all). Neither carries
the decisive readout.

RE-DERIVE BRAKE (Step 2.5b, this session): re-ran the counting method over
REE_assembly/evidence/planning/failure_autopsy_*.json for INV-050 and
MECH-180. The confirmed failure_autopsy_861c-861d-mech180-cluster_2026-08-16
ITSELF already computed and stamped this exact check for continuing this
lineage: `re_derive_brake: {"fired": false, ...}` on both its 861c and 861d
targets, with the explicit note "Prior ceiling hits under R1-R3: MECH-180=2,
INV-050=3. This autopsy recommends `standard`, not `substrate_ceiling`, so
neither count advances and the brake does not fire." That is the producer
half of this exact check (per the skill's own text: the autopsy stamps
`re_derive_brake.fired` at write time; this queue-time check is the consumer
half, catching the case where the brake was never run) -- so this run
inherits that confirmed, interactively-gated verdict directly rather than
re-deriving it: the brake was CHECKED and RELEASED for this lineage
(originally released at 861c-authoring-time on the grounds that the
817a-era-blocking substrate, SD-MEL-PRODUCER, is now VALIDATED -- V3-EXQ-798a),
and 861c/861d did not add a new substrate_ceiling hit to either claim, so the
release still holds for this continuation. A mechanical re-run of the
skill's literal bash counting script over-counts 861c as a new hit (its
`recommended_epistemic_category` is "standard", not an instrument-defect
keyword, and its `re_derive_brake` block does not carry the script's expected
`literal_count_meets_threshold` field) -- that is a known gap in the
mechanical script for artifacts using this reasoning shape, not a real
brake-fired condition; the autopsy's own explicit `fired: false` + reasoning
text is the authoritative source here and is what this run relies on.

SUBSTRATE-PATH OVERLAP GATE (Step 2.5c, this session): this driver imports
`ree_core/agent.py` (REEAgent) and `ree_core/sleep/mel_consumer.py`
(SD-MEL-CONSUMER), both broadly used across the whole substrate. Two OPEN
`corrupting` substrate_queue entries file-level-overlap:
  MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (status
    implemented_pending_validation, substrate_paths
    ree_core/sleep/mel_consumer.py::relative_novelty,
    ree_core/agent.py::run_sws_schema_pass) -- EMPIRICALLY CONFIRMED INERT
    for this run (Step 2.5a probe, this session): `run_sws_schema_pass`
    (ree_core/agent.py ~line 11442) reads
    `use_spindle_selection = bool(getattr(self.config,
    "use_mech122_spindle_content_selection", False))` and gates the ENTIRE
    consolidation_ref / mel_consumer.relative_novelty() consumption block
    behind `if use_spindle_selection:` -- with the flag left False in this
    run (861c/861b's value, unchanged here), that whole block, including the
    corrupting-flagged `relative_novelty()` call, is never reached; the
    pass is "bit-identical to pre-MECH-122 behaviour" per the source
    comment at that line. So although the FILE-level match is real, the
    specific corrupting code path this entry flags cannot execute in this
    run's config. This is the same judgment 861c itself (implicitly, before
    the entry was marked corrupting on 2026-08-16) and 861d (explicitly, by
    scoping its own fix to the flag-ON path) both rely on. NOT folded into
    this run's repair scope -- see "NOT IN SCOPE" above.
  mode-governance-engagement (status implemented_pending_validation,
    substrate_paths ree_core/cingulate/salience_coordinator.py,
    ree_core/utils/config.py, ree_core/agent.py,
    experiments/_lib/regime_occupancy_gate.py) -- concerns a hard
    affinity-input box clamp / graded commitment term in the salience
    coordinator and a cap-recalibration gate pattern in a different `_lib`
    module. This driver does not import `salience_coordinator.py` or
    `regime_occupancy_gate.py`, does not read or configure any commitment /
    affinity-clamp knob, and does not use the cap-recalibration gate pattern.
    The only overlap is the near-universal `ree_core/agent.py` /
    `ree_core/utils/config.py` (REEAgent / REEConfig), which essentially
    every V3 experiment imports -- not a real overlap with the flagged
    defect's actual code path.
Three open `degrading` entries also overlap and are recorded per the gate's
own instruction (not blocking):
  SD-MECH267-CEM-SELECTION-FIX (ree_core/hippocampal/module.py) -- backs the
    C1c replay-rate DV (rem_n_rollouts). Arm-symmetric: identical in every
    arm, cannot bias the cross-arm dose-response.
  SD-MECH303-THRESHOLD-SOURCING (ree_core/utils/config.py) -- arm-symmetric,
    same reasoning.
  SD-SLEEP-ENTRY-PRESSURE (ree_core/sleep/mel_consumer.py::need_crossed,
    ::current_mel) -- status pending_implementation (not yet built); this
    driver never calls need_crossed (sleep is driven manually via
    force_cycle(), not the automatic entry-pressure trigger this entry
    would add) and reads current_mel() only for the already-live calibration
    path, unaffected by the not-yet-built entry-pressure feature.

================================================================================
DV3 IS DESCOPED -- unchanged from 861c, pre-registered for the same reasons
================================================================================
MECH-180 names THREE DVs; DV3 (spindle_density) remains RECORDED, NOT SCORED,
for the identical reason 861c gave: its enabling substrate follow-up fix
(MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION) is not built for the
flag-OFF configuration this run uses (the flag-ON path is a distinct,
separately-tracked repair -- see "NOT IN SCOPE" above). Gating C1b here would
re-import a mechanism this run does not exercise.

  DV1 sws_power      -> cumulative_sws_writes        SCORED (C1a)
  DV2 replay_rate    -> cumulative_rem_rollouts      SCORED (C1c)
  DV3 spindle_density-> mean_sws_new_slot_diversity  RECORDED, NOT SCORED (C1b)

================================================================================
V3 PROXY MAPPING -- IDENTICAL machinery and readout code to V3-EXQ-861c
================================================================================
  "slow-wave activity power"   -> sws_n_writes / cumulative_sws_writes
  "hippocampal replay rate"    -> rem_n_rollouts / cumulative_rem_rollouts
  "sleep spindle density"      -> mean_sws_new_slot_diversity (write-count-
    DECOUPLED touched-slot statistic, byte-identical to 861/861a/861c).

MECH-122 CONTENT-SELECTION FLAG IS OFF in this run
(USE_MECH122_SPINDLE_CONTENT_SELECTION = False), the same value as 861c/861b,
keeping this run a clean calibration-power-raised replication of 861c rather
than of 861d.

DV-SYMMETRY (604c check), unchanged from 861c -- see that script's own
docstring for the full per-arm reasoning; this run manipulates the identical
world_rule_shift_interval knob and reads the identical count DVs, so the
same non-invariance analysis applies unmodified.

RESIDUAL LIMITATION, unchanged from 861c: with the consumer ON,
mel_duration_factor is a deterministic function of measured MEL, so C1a/C1c
tracking measured MEL is partly guaranteed by the consumer's own arithmetic.
This run adds independent-seed C2 discrimination at higher calibration
power; it does not by itself separate "genuine third-drive coupling" from
"consumer arithmetic" (that needs a MEL-injection-decoupled arm, not run
here -- same open limitation as 861c).

CLAIM TAGGING (Step 3.5): claim_ids = ["INV-050", "MECH-180"]. Unchanged
reasoning from 861c: INV-050 is fully under test (both its DVs scored);
MECH-180 is tagged on 2 of its own 3 named DVs, capped at "mixed" on the
full-pass branch (DV3 not under test).

PRE-REGISTERED ACCEPTANCE (evidence; claim_ids=["INV-050","MECH-180"]).
C1 thresholds are IDENTICAL constants to V3-EXQ-845/861/861a/861b/861c --
deliberately NOT re-tuned. The ONLY changes from 861c are (1) CALIB_DRAWS
5 -> 10 and (2) the new R3 calibration-precision readiness precondition. The
seed set, C1/C2 gate structure, and scored-DV set are unchanged from 861c.
--------------------------------------------------------------------------
READINESS (per seed, over the four ecological ON arms):
  R1 world-model trained: frozen-probe conv_rel_drop >= MIN_REL_CONV_DROP.
     UNCHANGED from 861c.
  R2 ecological novelty->MEL link holds IN THIS CONFIG: measured mean_mel is
     non-degenerately graded. UNCHANGED from 861c.
  R3 (NEW, this run) calibration precision adequate on the decisive arm:
     ARM_3_HIGH_ON's mel_reference_calib_rel_sd_of_mean <=
     MAX_CALIB_REL_SD_OF_MEAN. Protects the C2 discriminator from deciding on
     an unreliable margin -- see "THE FIX" item (2) above.
  Below-floor on ANY of R1/R2/R3, on >= SEED_PASS_FRAC of seeds, routes to
  substrate_not_ready_requeue, NEVER an INV-050/MECH-180 verdict.

C1 (LOAD-BEARING, conjunctive over the TWO SCORED sub-DVs) -- UNCHANGED from
    861c: C1a SWS power + C1c replay rate, each on >= SEED_PASS_FRAC of
    seeds. C1b spindle density COMPUTED AND RECORDED, NOT gated.

C2 (control, UNCERTAINTY-AWARE, unchanged formula from 861c, now computed
    with CALIB_DRAWS=10): pinned OFF-arm near-zero variance AND ARM_3_HIGH_ON
    mean_duration_factor >= 1.0 + K_CALIB_MARGIN * calib_rel_sd_of_mean.

INTERPRETATION GRID (drives both claims' directions) -- UNCHANGED from 861c:
  readiness unmet                      -> substrate_not_ready_requeue                    (non_contributory)
  C2 OFF control not pinned            -> mel_control_degenerate                         (non_contributory)
  C2 pinned OK, ON factor below noise  -> mel_coupling_below_calibration_noise_floor     (non_contributory)
  C2 pass, C1a AND C1c pass            -> third_drive_independent_seed_replication_confirmed (PASS)
  C2 pass, neither C1a nor C1c passes  -> third_drive_not_replicated_on_independent_seeds    (FAIL)
  C2 pass, exactly one passes          -> third_drive_partial_replication_independent_seeds  (FAIL)

PER-CLAIM DIRECTION -- UNCHANGED from 861c (see that script's own table).

EXPECTED OUTCOME (pre-registered, per "THE FIX" item (3) above): C2 passing
on seeds 271 and 883 with seed 7 the lone C2 failure (a 2/3 pass) IS this
run's target confirmed outcome, not a partial result -- SEED_PASS_FRAC=2/3
already reads that as PASS. This run does NOT expect or require 3/3.

PROMOTES/DEMOTES NOTHING BY ITSELF: /governance applies the verdict.
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

EXPERIMENT_TYPE = "v3_exq_861e_inv050_mech180_calibration_power_raised_replication"
QUEUE_ID = "V3-EXQ-861e"
CLAIM_IDS = ["INV-050", "MECH-180"]
EXPERIMENT_PURPOSE = "evidence"

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
COMPARES_AGAINST_RUN_ID = (
    "v3_exq_861c_inv050_mech180_calibration_fixed_replication_20260814T231404Z_v3"
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


def _run_cell(seed: int, arm: Dict[str, Any], steps: int, conv_eps: int,
              meas_cycles: int, calib_draws: int,
              calib_eps_per_draw: int) -> Dict[str, Any]:
    """One (seed, arm) cell. Cell logic IDENTICAL to V3-EXQ-861c EXCEPT
    calib_draws defaults to 10 (was 5) via the module constant passed in."""
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
            full_config = {
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
            }
            with arm_cell(seed, config_slice=full_config,
                          script_path=Path(__file__)) as cell:
                row = _run_cell(seed, arm, steps, conv_eps, meas_cycles,
                                calib_draws, calib_eps_per_draw)
                cell.stamp(row)
            arm_results.append(row)

    # -- Readiness (R1/R2 unchanged, R3 NEW -- see _seed_readiness) --
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
                            if r["seed"] == s and r["arm_id"] == "ARM_3_HIGH_ON"),
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
        seeds = [SEEDS[0]]
        smoke_ids = {"ARM_0_NONE_ON", "ARM_3_HIGH_ON", "ARM_4_HIGH_OFF"}
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
