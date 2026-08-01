"""V3-EXQ-860: MECH-204/SD-076 H2 discrimination probe, REDESIGNED -- does a longer
episode (more waking steps per episode) close the V3-EXQ-794a full-loop-vs-repair-smoke
gap, WITHOUT also raising the F1/REM recalibration firing count?

GOV-FANOUT-1 leg. Fanout source: failure_autopsy_V3-EXQ-794a_2026-07-31.json,
target run_id v3_exq_794a_mech204_phase7_sd076_calibration_loop_2x2_20260724T063301Z_v3,
hypothesis H2 ("insufficient training exposure") of a 3-way discrimination
(H1 F1-interaction damping / H2 insufficient exposure / H3 wrong mechanism form).
This is a CORRECTED REPLACEMENT for the just-failed V3-EXQ-853 leg (script
v3_exq_850_mech204_sd076_h2_exposure_budget_probe.py) -- not a new hypothesis, not a
lettered bugfix of 853's own numbers, but a redesign of H2's MANIPULATED AXIS per
failure_autopsy_V3-EXQ-853_2026-08-01.md section 6 ("The design confound"), because 853's
own manipulation (N_TRAIN_EPS) was discovered to be non-orthogonal to the sibling H1 axis.
See "THE CONFOUND THIS REDESIGN FIXES" below for the full mechanism and "WHY THIS
COUNTS AS THE SAME LEG, REDESIGNED" for why this is not itself a new fanout leg.

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every
episode) -- UNCHANGED from 794a AND from 853. The whole point of this redesign is that
K=1 now fires exactly N_TRAIN_EPS=30 times (794a's own budget), not 853's 150 -- see below.

THE CONFOUND THIS REDESIGN FIXES
---------------------------------
V3-EXQ-853 (the just-failed prior H2 leg) manipulated N_TRAIN_EPS (30 -> 150, 5x) to test
whether more total waking exposure closes the 794a-vs-repair-smoke rv_final gap. Its own
autopsy (failure_autopsy_V3-EXQ-853_2026-08-01.md section 6, "The design confound")
identified that this manipulation is NOT single-axis: SleepLoopManager fires the F1 REM
precision-recalibration WRITEBACK once per episode (K=1, sleep_loop_episodes_K=1), read at
the NEXT agent.reset() call (_read_recalib_metrics). So raising N_TRAIN_EPS from 30 to 150
simultaneously raises (a) total waking exposure -- H2's intended axis -- AND (b) the total
F1-firing count from 30 to 150 -- the axis the SIBLING H1 leg (V3-EXQ-850) manipulates
directly (by disabling F1 recalibration outright). V3-EXQ-853's result (dose separation
SHRANK monotonically as N_TRAIN_EPS rose: smoke 4.35e-4 -> 794a[30] 1.28e-4 ->
853[150] 2.48e-5) is therefore inseparable from "more F1 firings increasingly dominate and
suppress the dose signal" (favours H1) versus "more exposure alone would have done this"
(H2's own claim) -- the run cannot tell the two apart, per the autopsy's own section 5
reasoning.

THE FIX: hold N_TRAIN_EPS AT 794a's ORIGINAL VALUE (30) -- so F1 fires exactly 30 times,
identical to 794a and to the sibling H1 leg's own budget, completely unconfounded with
H1's axis -- and manipulate STEPS_PER_EP (the per-episode step count) instead. Raising
STEPS_PER_EP raises total waking exposure (N_TRAIN_EPS * STEPS_PER_EP) exactly as raising
N_TRAIN_EPS did, but LEAVES THE F1-FIRING COUNT UNTOUCHED, because K=1 fires once per
EPISODE regardless of how many steps that episode contains (SleepLoopManager's firing
condition is keyed to the episode-boundary counter, not the step counter -- verified by
direct code read of ree_core/agent.py's agent.reset() -> SleepLoopManager firing path,
which contains no reference to STEPS_PER_EP or any per-step counter). This decouples H2's
exposure axis from H1's recalibration-cycle axis cleanly, which is exactly the fix the
853 autopsy itself specified (section 6, last sentence: "A clean H2 probe should vary
STEPS_PER_EP (episode length) instead, which raises total waking exposure without
touching firing count") and the routing the user confirmed (autopsy section 9).

WHY THIS COUNTS AS THE SAME LEG, REDESIGNED (not a new fanout leg, not a lettered
bugfix). GOV-FANOUT-1's "never a power-bump of the braked design" rule (queue-experiment
skill Step 2.5b) forbids re-posing the SAME confounded manipulation at a larger dose --
that is exactly what 853 already braked against a hypothetical 10x/20x N_TRAIN_EPS
re-attempt. This script does not do that: it changes WHICH variable is manipulated, so
that H2's axis (exposure) is no longer entangled with H1's axis (recalibration count).
This is a discrimination-design correction (853's own diagnosis says "the environment is
confounded", not "the substrate is at ceiling") -- per the autopsy's own re-derive-brake
block (fired: false, 0 prior substrate_ceiling autopsies for SD-076) and per the queue-
experiment skill's own carve-out ("a diagnostic whose purpose is to discriminate WHY the
ceiling holds ... is not the re-derive loop"). No new EXQ number was assigned per the
"redesign of the SAME mechanism/claim" convention question -- but this IS in practice a
new number (V3-EXQ-860), because the prior leg's own manipulated variable (N_TRAIN_EPS)
is being replaced wholesale by a different one (STEPS_PER_EP): per CLAUDE.md's
EXQ-versioning rule, "bug fix" = same scientific question, implementation was WRONG;
"major redesign" = the experimental design is substantially different. Swapping the
entire manipulated axis is a design change, not an implementation bug -- 853's code ran
correctly and produced a real, informative (if unwanted) result; nothing in it was
"wrong" in the sense a lettered fix corrects. Hence a new number, not V3-EXQ-853b.

BUDGET CHOSEN: STEPS_PER_EP 200 -> 1000 (5x), N_TRAIN_EPS held at 30 (UNCHANGED from
794a -- this is what decouples F1-firing count from exposure). Reasoning: the redesign's
goal is a total-exposure increase "roughly comparable in magnitude" to 853's own 5x
N_TRAIN_EPS change, so that this run and 853 measure the SAME SIZE of exposure increase
via two different mechanisms (episode length vs episode count) and are directly
comparable. 853's total per-cell TRAINING exposure was N_TRAIN_EPS(150) *
STEPS_PER_EP(200) = 30000 waking steps. Matching that exactly: N_TRAIN_EPS(30) *
STEPS_PER_EP(1000) = 30000 waking steps -- the SAME total training exposure as 853,
reached via 30 long episodes instead of 150 short ones. This is not an arbitrary
multiplier: it is the largest STEPS_PER_EP increase that keeps this run's total training
exposure exactly parity with the just-failed run it replaces, so any DIFFERENCE in
outcome between 853 and this run is attributable to the confound (F1-firing count: 150
vs 30) rather than to a difference in how much exposure was tested. EVAL_STEPS_PER_EP is
kept at 794a's ORIGINAL 200 (NOT scaled) -- exactly 853's own precedent ("scaling only
the training phase ... since the decisive readout rv_final is read at the END of
training, before eval begins"), for the identical reason: eval-phase step count does not
touch the decisive readout, and keeping it fixed keeps the eval-derived overconfidence
score/calibration ratio comparable in scale across 794a / 853 / this run.

COMPUTE-TIME PARITY WITH 853 (see the proposed queue entry `note` for the full
derivation): per-cell TOTAL step count (train + eval) is IDENTICAL to 853's own:
train 30 * 1000 = 30000, eval 20 * 200 = 4000, total 34000 -- exactly 853's own
150 * 200 (train) + 20 * 200 (eval) = 30000 + 4000 = 34000. Same arm/seed count (2 arms
x 3 seeds = 6 cells) as 853. So this run's estimated wall time is parity with 853's own
140-minute estimate (if anything marginally FASTER: 30 episode-boundary sleep cycles
instead of 150, saving ~120 * (SWS_CONSOLIDATION_STEPS + REM_ATTRIBUTION_STEPS) = 120 *
14 = 1680 sleep-cycle substeps against a 34000-step main budget -- a ~5% reduction, small
enough that the same 140-minute estimate is kept as the safe (not optimistic) figure).

WHY ONLY TWO ARMS (ARM_INFL_LO / ARM_INFL_HI), NOT A 2x3 FACTORIAL.
Unchanged from 853/850-h2 and 850-h1: this is a single-axis discrimination probe
(GOV-FANOUT-1: "never a power-bump of the braked design"; each leg attacks ONE design
axis). The Phase 7 broadcast axis (MECH-204's correction, `use_rem_precision_broadcast`)
is not manipulated here at all -- it stays OFF in both arms, exactly as it is absent from
the repair's own isolated smoke, so the comparison against the smoke is apples-to-apples
on the SD-076 drift-source axis alone. F1/REM precision recalibration
(`use_rem_precision_recalibration`, rem_precision_recalibration_step=0.25) is
UNCONDITIONAL and UNCHANGED from 794a/853 -- it is the axis the SIBLING (H1) leg
manipulates, not this one; holding it ON and letting it fire exactly 30 times (matching
794a) is precisely the point of this redesign.

DECISIVE READOUT NAMED, GOV-REUSE-1 CHECK (queue-experiment Step 2.4). Decisive readout:
rv_final_after_training at the LO and HI inflation doses, from a full behavioural loop on
the SD-076-headroom-repaired substrate under a STEPS_PER_EP-varied exposure manipulation
that is decoupled from F1-firing count. Checked via
REE_assembly/scripts/reanalysis_query.py query --readout rv_final --claim SD-076
(2026-08-01): four manifests carry rv_final -- v3_exq_794 (substrate_hash
402e3f5a23a3a8e1..., pre-repair), v3_exq_794a (substrate_hash f569f39451e9746a...,
N_TRAIN_EPS=30, STEPS_PER_EP=200), v3_exq_850_..._h1_f1_damping_probe (substrate_hash
f5130fe8f287b555..., F1 disabled, N_TRAIN_EPS=30, STEPS_PER_EP=200), and
v3_exq_850_..._h2_exposure_budget_probe / V3-EXQ-853 (substrate_hash
d9acf1249f3eac28..., N_TRAIN_EPS=150, STEPS_PER_EP=200). NONE of the four vary
STEPS_PER_EP -- every recorded run in this lineage held it at 794a's original 200. The
decisive readout this leg exists to produce (rv_final under a STEPS_PER_EP-varied,
F1-count-matched-to-794a exposure manipulation) does not exist in any recorded manifest
and is not derivable by reanalysis of the existing four (none of them isolate the
exposure axis from the firing-count axis). Not recoverable -> run.

SUBSTRATE READINESS (queue-experiment Step 2.5). Every feature this script exercises is
already IMPLEMENTED and consumed identically to 794a/850-h1/853: the SD-076 headroom
repair (ree-v3 452f99e367, sd_waking_confidence_inflation_headroom, status "implemented"
in substrate_queue.json) and MECH-204's F1 REM precision recalibration
(rem_precision_recalibration_step, already load-bearing in 794a/853). This probe
manipulates NEITHER's implementation, only STEPS_PER_EP (train-phase episode length) --
no new substrate build is needed. claims.yaml: SD-076 is `candidate` /
`epistemic_category: standard`, no `v3_pending` / `implementation_phase: v3` gate.
Attribute surface re-verified directly against current ree-v3 source (2026-08-01):
`E3TrajectorySelector._running_variance` / `._wci_symmetric_rv_ref` (ree_core/predictors/
e3_selector.py:302,314), `agent.sleep_loop` (ree_core/agent.py:2336), the
`mech204_recalibration_fired` / `mech204_running_variance_{before,after}` writeback keys
(ree_core/sleep/phase_manager.py:442-453), and the `use_waking_confidence_inflation` /
`waking_confidence_inflation_asymmetry` / `waking_confidence_rv_floor*` REEConfig.e3
fields (ree_core/utils/config.py:600-658) all confirmed present and unchanged.

RE-DERIVE BRAKE (queue-experiment Step 2.5b). Standard grep-count method over
REE_assembly/evidence/planning/failure_autopsy_*.json for claim SD-076: 0 autopsies
count as genuinely braking (0 `substrate_ceiling`-category, 0 `non_contributory`-
direction with an owed build). The just-failed V3-EXQ-853 autopsy's own
`recommended_epistemic_category` is `measurement_test_design_defect` (not
`substrate_ceiling`) with `recommended_evidence_direction: weakens` (not
non_contributory), and its own `re_derive_brake` block records `fired: false`,
`prior_substrate_ceiling_autopsies: []`. Brake does not fire -> proceed to author. (This
is additionally exempt on its own terms regardless of count, being a GOV-FANOUT-1
discrimination leg explaining WHY a confounded design read the way it did, not a
same-axis re-test of an already-braked design -- see "WHY THIS COUNTS AS THE SAME LEG,
REDESIGNED" above.)

CLAIM TAGGING (queue-experiment Step 3 "claim_ids accuracy rule"). CLAIM_IDS = ["SD-076"]
ONLY, not MECH-204 -- identical reasoning to 853/850-h2: this run never engages the
Phase 7 broadcast (`use_rem_precision_broadcast=False` in every arm), so it produces no
evidence at all about MECH-204's corrective mechanism. F1/REM recalibration (nominally a
MECH-204 substrate component) is held constant/unmanipulated here (fires 30 times in
every arm, matching 794a), so even though active, its presence cannot yield MECH-204
evidence either way -- an unmanipulated factor cannot be attributed a direction.

EXPERIMENT_PURPOSE = "diagnostic" (matches 794/794a/850-h1/853): discriminates WHY 794a's
full-loop result fell short of the repair's own smoke, not a governance-scoring evidence
run. Excluded from confidence/conflict scoring per convention.

DV-SYMMETRY INVARIANCE DECLARATION (mandatory per-arm; queue-experiment Step 3).
DV = rv_final_after_training, a scalar level of `E3TrajectorySelector._running_variance`
read at the end of the training phase (before any eval-phase tick). It is a SNAPSHOT of
an EMA's current level, not a statistic computed over a permutable collection -- it has
no permutation symmetry to be invariant or non-invariant under. The only relevant
question is whether the manipulations change that level:
  ARM_INFL_LO / ARM_INFL_HI (the dose axis): the asymmetric EMA's update rule (and hence
      its settling level) changes with `inflation_asymmetry`. A level change, not a
      symmetry-preserving transform. NOT invariant. OK -- identical reasoning to
      794a/853's factor B declaration.
  STEPS_PER_EP (the exposure axis, THIS leg's manipulation, replacing 853's N_TRAIN_EPS):
      more update iterations before the snapshot is taken changes an EMA's level whenever
      it has not yet converged, exactly as N_TRAIN_EPS did in 853 -- the number of update
      calls before the snapshot is what matters to an EMA's level, not whether those
      calls are distributed across more episodes or more steps within fewer episodes.
      NOT invariant by construction -- this is the entire premise of the probe.
Both arms write the SAME scalar (rv, in precision units) the smoke, 794a, and 853 all
read, so all four readouts (smoke, 794a, 853, this run) are on a directly comparable
scale.

Both arms also carry the internal comparator `_wci_symmetric_rv_ref` (the
E3TrajectorySelector's own tracked un-inflated counterfactual EMA -- see
ree_core/predictors/e3_selector.py update_running_variance / _apply_wci_rv_floor), read
per seed at end of training. This lets the `inflation_lowers_rv` readiness precondition
be evaluated WITHOUT running an ARM_OFF_OFF comparator arm, unchanged from 853/850-h2.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

QUEUE_ID = "V3-EXQ-860"
EXPERIMENT_TYPE = "v3_exq_860_mech204_sd076_h2_steps_per_ep_probe"
CLAIM_IDS = ["SD-076"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SLEEP_DRIVER_PATTERN = (
    "K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)"
)
FANOUT_SOURCE = "failure_autopsy_V3-EXQ-794a_2026-07-31.json"
FANOUT_HYPOTHESIS = "H2-insufficient-training-exposure-redesigned-steps-per-ep"
FANOUT_TARGET_RUN_ID = "v3_exq_794a_mech204_phase7_sd076_calibration_loop_2x2_20260724T063301Z_v3"
REDESIGN_SOURCE = "failure_autopsy_V3-EXQ-853_2026-08-01.json"
SUPERSEDES_RUN_ID = "v3_exq_850_mech204_sd076_h2_exposure_budget_probe_20260801T005937Z_v3"
SUPERSEDES_QUEUE_ID = "V3-EXQ-853"

# ---- Run shape. STEPS_PER_EP (train) is now the manipulated axis; N_TRAIN_EPS is held
# at 794a's original value so F1-firing count is UNCHANGED (the confound fix). ----
N_TRAIN_EPS_794A = 30            # 794a's own budget
N_TRAIN_EPS = 30                 # UNCHANGED from 794a -- F1 fires exactly 30 times,
                                  # identical to 794a and the sibling H1 leg. THIS is what
                                  # decouples exposure (this axis) from F1-firing count
                                  # (853's confound). Do NOT raise this -- see design-time
                                  # assert below.
STEPS_PER_EP_794A = 200          # 794a's own per-episode step count (both phases)
STEPS_PER_EP = 1000              # 5x -- see module docstring "BUDGET CHOSEN". Total
                                  # training exposure (30 * 1000 = 30000) matches 853's own
                                  # total training exposure (150 * 200 = 30000) exactly.
EVAL_STEPS_PER_EP = 200          # UNCHANGED from 794a/853 -- decisive readout is read
                                  # BEFORE eval begins, so eval length does not affect it;
                                  # kept fixed for cross-run comparability of the
                                  # secondary eval-phase metrics (overconfidence_score,
                                  # calibration_ratio), mirroring 853's own precedent of
                                  # keeping N_EVAL_EPS fixed while scaling only the train
                                  # axis.
N_EVAL_EPS = 20
N_SEEDS = 3
GRID_SIZE = 12
LR = 5e-4

# ---- Substrate operating point (held constant, identical to 794a/853). ----
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1
REM_PRECISION_RECALIBRATION_STEP = 0.25
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3

# ---- The single factor under test: SD-076 asymmetry dose. Phase 7 broadcast is
# NOT a factor here -- it is held OFF in every arm (see module docstring). ----
INFLATION_ASYMMETRY_LO = 0.6
INFLATION_ASYMMETRY_HI = 0.8
INFLATION_RV_FLOOR = 0.01
INFLATION_RV_FLOOR_RELATIVE_FRAC = 0.2   # SD-076 headroom repair, identical to 794a
INFLATION_RV_FLOOR_MODE = "soft"
INFLATION_RV_FLOOR_SOFTNESS = 0.25

# ---- Pre-registered thresholds (NOT derived from this run's own statistics). ----
PRECISION_INIT_BASELINE = 0.5    # REEConfig precision_init default
RV_LIVE_FLOOR = 1e-6             # rv_final must differ from precision_init by more than this
RECALIB_MOVE_FLOOR = 1e-4        # F1 mean per-cycle |rv_after - rv_before| floor
INFLATION_MOVE_FLOOR = 1e-4      # inflation must push rv below the internal un-inflated ref
# Same statistic and same floor as 794a/853's dose_levels_separated gate -- the "still
# clamped" saturation signature this probe would ALSO need to rule out before its own H2
# readout can be trusted.
DOSE_SEPARATION_FLOOR = 1e-4

# ---- H2-specific pre-registered EXTERNAL reference values (not derived from this
# run). Cited precisely in the module docstring's "COMPUTE-TIME PARITY WITH 853" and
# "BUDGET CHOSEN" sections. Same 794a/smoke references as 853, PLUS 853's own rv_final as
# additional (non-load-bearing) context so a reader can place all three N_TRAIN_EPS/
# STEPS_PER_EP configurations side by side. ----
FULLLOOP_794A_RV_FINAL = {
    "LO": 0.003997733405264173,   # 794a manifest aggregates.per_level.LO.rv_final
    "HI": 0.003870367272153275,   # 794a manifest aggregates.per_level.HI.rv_final
}
SMOKE_RV_FINAL = {
    # ree-v3 452f99e367 sd_waking_confidence_inflation_headroom validation smoke
    # (test_sd076_rv_floor_headroom.py, REPAIRED config, at the 794-measured error
    # scale true_error_ref ~0.0037). 7-significant-figure values as recorded in
    # substrate_queue.json's implementation_note_update.
    "LO": 0.0025377,
    "HI": 0.0021031,
}
# CONTEXT ONLY (not load-bearing, not used by C1/C2/C3): the just-failed confounded H2
# leg's own rv_final, from V3-EXQ-853's manifest aggregates.per_level.{LO,HI}.rv_final.
# Comparing against this shows directly whether decoupling the F1-firing count from
# exposure changes the reading, holding total training exposure fixed at 30000 steps in
# both runs.
CONFOUNDED_853_RV_FINAL = {
    # V3-EXQ-853 manifest aggregates.per_level.{LO,HI}.rv_final (=
    # aggregates.arm_rv_final.ARM_INFL_{LO,HI}), from
    # v3_exq_850_mech204_sd076_h2_exposure_budget_probe_20260801T005937Z_v3.json.
    "LO": 0.0036801948616908275,
    "HI": 0.0036554310790144306,
}
# Fraction of the (794a -> smoke) gap this run's rv_final must close, at BOTH LO and HI,
# to read H2 as SUPPORTED (more exposure moves the substrate meaningfully toward the
# smoke's demonstrated regime).
H2_CLOSURE_SUPPORT_FLOOR = 0.30
# Fraction below which, at BOTH LO and HI, reads as a PLATEAU despite the 5x-equivalent
# budget -- H2 NOT supported (the gap is not an exposure/dose-duration artifact).
H2_CLOSURE_PLATEAU_CEILING = 0.10

# (arm_id, inflation_asymmetry). Both arms have Phase 7 broadcast OFF (see module
# docstring "WHY ONLY TWO ARMS").
ARMS: Tuple[Tuple[str, float], ...] = (
    ("ARM_INFL_LO", INFLATION_ASYMMETRY_LO),
    ("ARM_INFL_HI", INFLATION_ASYMMETRY_HI),
)

# Ascending order, mirroring 794a/853's operative-level convention (unused here for a
# capability gate since both arms are always evaluated, but kept so per-level reporting
# reads the same way as the sibling runs for a human cross-reading them).
INFLATION_LEVELS = (
    ("LO", INFLATION_ASYMMETRY_LO, "ARM_INFL_LO"),
    ("HI", INFLATION_ASYMMETRY_HI, "ARM_INFL_HI"),
)

# Each arm's sibling at the OTHER asymmetry level -- the matched positive control
# for the dose-separation precondition (differs ONLY in the dose).
DOSE_SIBLING = {
    "ARM_INFL_LO": "ARM_INFL_HI",
    "ARM_INFL_HI": "ARM_INFL_LO",
}

# arm_id -> level key, for the per-cell progress verdict below (queue-experiment skill
# Step 3 progress-instrumentation rule #3 -- the 850-h2/853 template this script is
# derived from OMITTED this per-cell verdict line entirely, a gap caught and fixed by
# this script's own Step 3.5 code-review pass; see 850-h1's own precedent, which DOES
# carry an analogous per-cell proxy verdict, for the pattern this follows).
ARM_TO_LEVEL = {"ARM_INFL_LO": "LO", "ARM_INFL_HI": "HI"}

# z_goal liveness accumulator (queue-experiment skill Step 3/4 "z_goal liveness
# recorded?" -- another gap present in the 850-h2/853 template, wired here per 850-h1's
# own precedent). This run does not manipulate goal state at all, so an
# all-zero/unmeasured reading is the EXPECTED result, not a defect -- recording it makes
# that explicit rather than silently absent.
_ZG = ZGoalStreamAccumulator()


# ---------------------------------------------------------------- preconditions --
# Both arms are inflation arms with no broadcast axis, so every precondition here
# applies unconditionally (no regime conditioning needed -- contrast 794a, where
# broadcast-scoped preconditions needed `applies_to`).
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="rv_live",
        description="rv_final differs from precision_init by more than the floor "
                    "(the Q-042/530c substrate-liveness contract). Worst cell "
                    "reported.",
        control="every seed of this arm; a dead rv makes the DV meaningless",
        threshold=RV_LIVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="f1_recalib_engaged",
        description="mean per-cycle |rv_after - rv_before| from the F1 WRITEBACK "
                    "recalibration exceeds the floor, i.e. REM was entered and the "
                    "MECH-204 lever moved rv at least once. F1 recalibration is ON "
                    "in every arm, unchanged from 794a/853 -- confirms the substrate "
                    "operating point this probe holds constant is actually live "
                    "across the LONGER episodes too (30 firings, matching 794a "
                    "exactly, NOT 853's 150).",
        control="F1 recalibration is ON in every arm of this design",
        threshold=RECALIB_MOVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="inflation_lowers_rv",
        description="mean over seeds of (wci_symmetric_rv_ref_final - "
                    "rv_final_after_training). SIGNED: SD-076 must push rv DOWN "
                    "relative to the SUBSTRATE'S OWN internally-tracked un-inflated "
                    "counterfactual (E3TrajectorySelector._wci_symmetric_rv_ref) "
                    "or it is not an inflation source. Same statistic the DV "
                    "routes on (rv level), measured against a positive control "
                    "computed by the substrate itself on every tick -- no separate "
                    "ARM_OFF_OFF arm is needed for this comparison.",
        control="each seed's own _wci_symmetric_rv_ref, tracked in parallel by the "
                "substrate on every update_running_variance call",
        threshold=INFLATION_MOVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="dose_levels_separated",
        description="|rv_final(this arm) - rv_final(sibling arm at the OTHER "
                    "asymmetry)|. THE 794/853 GATE, carried forward: two nominally "
                    "different doses producing the same rv is a SATURATION "
                    "signature, not a null. Must clear before this run's own H2 "
                    "closure readout can be trusted (a still-clamped lever would "
                    "read as 'no closure' for a reason unrelated to exposure).",
        control="sibling inflation arm at the other asymmetry level, same seeds "
                "-- differs only in the dose",
        threshold=DOSE_SEPARATION_FLOOR,
        direction="lower",
    ),
)


def _arm_ctx(arm_id: str, asym: float) -> Dict[str, object]:
    return {"arm_id": arm_id, "asymmetry": asym}


ARM_CONTEXTS = [_arm_ctx(a, x) for (a, x) in ARMS]


# ------------------------------------------------------------------ build helpers --
def _make_env(seed: int, dry_run: bool = False) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=(8 if dry_run else GRID_SIZE),
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.10,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, asym: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        sws_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        rem_enabled=True,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1,
        use_rem_precision_recalibration=True,
        precision_zero_point_ema_alpha=PRECISION_ZERO_POINT_EMA_ALPHA,
        rem_precision_recalibration_step=REM_PRECISION_RECALIBRATION_STEP,
        # Phase 7 broadcast is NOT a factor in this probe -- held OFF in every arm.
        use_rem_precision_broadcast=False,
        rem_precision_broadcast_gain=0.0,
    )
    # SD-076 waking confidence inflation -- the single manipulated dose axis.
    cfg.e3.use_waking_confidence_inflation = True
    cfg.e3.waking_confidence_inflation_asymmetry = float(asym)
    cfg.e3.waking_confidence_rv_floor = INFLATION_RV_FLOOR
    cfg.e3.waking_confidence_rv_floor_relative_frac = INFLATION_RV_FLOOR_RELATIVE_FRAC
    cfg.e3.waking_confidence_rv_floor_mode = INFLATION_RV_FLOOR_MODE
    cfg.e3.waking_confidence_rv_floor_softness = INFLATION_RV_FLOOR_SOFTNESS
    # Tonic 5-HT must be on for compute_recalibration_target() to be meaningful (the
    # F1 WRITEBACK reads it every recalibration cycle).
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def _arm_config_slice(asym: float, n_train: int, train_steps: int, eval_steps: int) -> Dict:
    """The config the cell's build+collect path actually reads."""
    return {
        "grid_size": GRID_SIZE,
        "train_steps_per_ep": train_steps,
        "eval_steps_per_ep": eval_steps,
        "n_train_eps": n_train,
        "n_eval_eps": N_EVAL_EPS,
        "lr": LR,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "sws_enabled": True,
        "rem_enabled": True,
        "use_rem_precision_recalibration": True,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "sleep_loop_episodes_K": 1,
        "tonic_5ht_enabled": True,
        "use_rem_precision_broadcast": False,
        "rem_precision_broadcast_gain": 0.0,
        "use_waking_confidence_inflation": True,
        "waking_confidence_inflation_asymmetry": float(asym),
        "waking_confidence_rv_floor": INFLATION_RV_FLOOR,
        "waking_confidence_rv_floor_relative_frac": INFLATION_RV_FLOOR_RELATIVE_FRAC,
        "waking_confidence_rv_floor_mode": INFLATION_RV_FLOOR_MODE,
        "waking_confidence_rv_floor_softness": INFLATION_RV_FLOOR_SOFTNESS,
        "precision_init_baseline": PRECISION_INIT_BASELINE,
    }


def _read_recalib_metrics(agent: REEAgent) -> Optional[Dict[str, float]]:
    """Sleep-cycle telemetry left in sleep_loop.state.last_metrics by agent.reset()."""
    if agent.sleep_loop is None:
        return None
    state = agent.sleep_loop.state
    if state is None or not state.last_metrics:
        return None
    m = dict(state.last_metrics)
    out: Dict[str, float] = {}
    if "mech204_recalibration_fired" in m:
        out["fired"] = float(m.get("mech204_recalibration_fired", 0.0))
    if "mech204_running_variance_before" in m and "mech204_running_variance_after" in m:
        out["rv_before"] = float(m["mech204_running_variance_before"])
        out["rv_after"] = float(m["mech204_running_variance_after"])
    return out or None


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------- one cell --
def _run_arm_seed(arm, seed, n_train, n_eval, train_steps, eval_steps, dry_run=False) -> Dict:
    arm_label, asym = arm

    with arm_cell(
        seed,
        config_slice=_arm_config_slice(asym, n_train, train_steps, eval_steps),
        script_path=Path(__file__),
        include_driver_script_in_hash=False,  # mint-as-you-go: cross-driver reusable
    ) as cell:
        env = _make_env(seed, dry_run=dry_run)
        agent = _make_agent(env, asym)
        optimizer = optim.Adam(agent.parameters(), lr=LR)

        print(f"Seed {seed} Condition {arm_label} n_train={n_train}", flush=True)

        # ---- Training: forward model learns; F1 recalibration fires each boundary
        # (exactly n_train times, K=1 -- UNCHANGED from 794a regardless of train_steps). ----
        recalib_moves: List[float] = []
        recalib_fired = 0
        train_harness = StepHarness(agent, env, train_mode=True, seed=seed)
        for ep in range(n_train):
            agent.reset()  # fires the sleep cycle for the prior episode (K=1)
            rec = _read_recalib_metrics(agent)
            if rec is not None:
                if rec.get("fired", 0.0) > 0.0:
                    recalib_fired += 1
                if "rv_before" in rec and "rv_after" in rec:
                    recalib_moves.append(abs(rec["rv_after"] - rec["rv_before"]))
            _, obs_dict = env.reset()
            train_harness.reset()
            for _ in range(train_steps):
                result = train_harness.step(obs_dict)
                optimizer.zero_grad()
                loss = agent.compute_prediction_loss()
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                obs_dict = result.next_obs_dict
                if result.done:
                    break
            if (ep + 1) % 5 == 0 or ep + 1 == n_train:
                print(
                    f"  [train] arm={arm_label} seed={seed} ep {ep + 1}/{n_train} "
                    f"rv={float(agent.e3._running_variance):.6f} "
                    f"ref={float(agent.e3._wci_symmetric_rv_ref):.6f} "
                    f"prec={float(agent.e3.current_precision):.4f}",
                    flush=True,
                )

        rv_after_training = float(agent.e3._running_variance)
        # The substrate's own internally-tracked un-inflated counterfactual -- the
        # inflation_lowers_rv precondition's positive control (see module docstring).
        wci_symmetric_rv_ref_after_training = float(agent.e3._wci_symmetric_rv_ref)

        # ---- Eval: capture confidence (rv) and accuracy (real forward-model error),
        # recorded for context/comparability with 794a/853 even though this probe's
        # decisive readout is rv_after_training, not the eval-phase overconfidence
        # score. eval_steps is UNCHANGED from 794a (fixed at 200) -- see module docstring
        # "BUDGET CHOSEN". ----
        eval_harness = StepHarness(agent, env, train_mode=False, seed=seed + 10000)
        rv_vals: List[float] = []
        pe_vals: List[float] = []
        for ep in range(n_eval):
            agent.reset()
            _, obs_dict = env.reset()
            eval_harness.reset()
            for _ in range(eval_steps):
                result = eval_harness.step(obs_dict)
                rv_vals.append(float(agent.e3._running_variance))
                pe = result.residue_metrics.get("e3_prediction_error")
                if pe is not None:
                    pe_vals.append(float(pe))
                obs_dict = result.next_obs_dict
                if result.done:
                    break

        mean_rv = _mean(rv_vals)
        true_error_ref = _mean(pe_vals)

        if true_error_ref > 1e-9 and mean_rv > 1e-9:
            calibration_ratio = mean_rv / true_error_ref
            overconfidence_score = float(np.log(true_error_ref / mean_rv))
        else:
            calibration_ratio = float("nan")
            overconfidence_score = 0.0

        absolutely_overconfident = overconfidence_score > 0.10
        print(
            f"  [eval] arm={arm_label} seed={seed} score={overconfidence_score:+.4f} "
            f"calib_ratio={calibration_ratio:.3f} true_err={true_error_ref:.6f} "
            f"mean_rv={mean_rv:.6f} rv_final={rv_after_training:.6f} "
            f"wci_ref={wci_symmetric_rv_ref_after_training:.6f}",
            flush=True,
        )

        # Per-cell progress verdict (runner display only -- NOT this run's scientific
        # verdict, which is level-level via C1/C2/C3 in _analyse). Proxy: did this cell's
        # rv_final move toward the smoke's reference relative to 794a's own reference at
        # the matching dose level? Mirrors the sibling H1 leg's own per-cell proxy
        # (v3_exq_850_mech204_sd076_h1_f1_damping_probe.py "closer_to_smoke").
        _level = ARM_TO_LEVEL[arm_label]
        _moved_toward_smoke = bool(
            (FULLLOOP_794A_RV_FINAL[_level] - rv_after_training) > 0.0)
        print(f"verdict: {'PASS' if _moved_toward_smoke else 'FAIL'}", flush=True)

        _ZG.observe(agent)  # AFTER stepping is complete for this cell

        row = {
            "arm_id": arm_label,
            "seed": seed,
            "inflation_asymmetry": float(asym),
            "n_train_eps": n_train,
            "train_steps_per_ep": train_steps,
            "eval_steps_per_ep": eval_steps,
            "overconfidence_score": overconfidence_score,
            "calibration_ratio": calibration_ratio,
            "true_error_ref": true_error_ref,
            "mean_running_variance": mean_rv,
            "rv_final_after_training": rv_after_training,
            "wci_symmetric_rv_ref_after_training": wci_symmetric_rv_ref_after_training,
            "rv_delta_from_precision_init": abs(rv_after_training - PRECISION_INIT_BASELINE),
            "recalib_cycles_fired": recalib_fired,
            "recalib_mean_abs_move": _mean(recalib_moves),
            "absolutely_overconfident": absolutely_overconfident,
            "n_eval_ticks": len(rv_vals),
            "n_pe_ticks": len(pe_vals),
        }
        cell.stamp(row)
    return row


# ---------------------------------------------------------------------- analysis --
def _closure_fraction(this_run_rv: float, level: str) -> float:
    """Fraction of the (794a -> smoke) gap this run's rv_final has closed.

    0.0 = exactly reproduces 794a (no movement); 1.0 = exactly reaches the smoke's
    value; >1.0 = overshoots past the smoke's value; negative = moved AWAY from the
    smoke's value (rv rose relative to 794a). rv is a "lower = more overconfident"
    scale, so the gap is (794a_rv - smoke_rv), both positive since 794a's rv sat
    above the smoke's.
    """
    baseline = FULLLOOP_794A_RV_FINAL[level]
    target = SMOKE_RV_FINAL[level]
    gap = baseline - target
    if abs(gap) < 1e-12:
        return 0.0
    return float((baseline - this_run_rv) / gap)


def _analyse(cells: List[Dict], seeds: List[int]) -> Dict:
    by_arm: Dict[str, Dict[int, Dict]] = {}
    for c in cells:
        by_arm.setdefault(c["arm_id"], {})[c["seed"]] = c

    arm_rv = {a: _mean([by_arm[a][s]["rv_final_after_training"] for s in seeds])
              for a in by_arm}
    arm_score = {a: _mean([by_arm[a][s]["overconfidence_score"] for s in seeds])
                 for a in by_arm}
    arm_ratio = {a: _mean([by_arm[a][s]["calibration_ratio"] for s in seeds])
                 for a in by_arm}
    arm_true_err = {a: _mean([by_arm[a][s]["true_error_ref"] for s in seeds])
                    for a in by_arm}

    # ---- readiness gates (both arms unconditionally in scope; no regime
    # conditioning needed -- see PRECONDITION_SPECS comment). ----
    arm_gates = []
    for (arm_id, asym) in ARMS:
        ctx = _arm_ctx(arm_id, asym)
        sibling = DOSE_SIBLING[arm_id]
        measured: Dict[str, float] = {
            "rv_live": min(by_arm[arm_id][s]["rv_delta_from_precision_init"] for s in seeds),
            "f1_recalib_engaged": _mean(
                [by_arm[arm_id][s]["recalib_mean_abs_move"] for s in seeds]),
            "inflation_lowers_rv": _mean(
                [by_arm[arm_id][s]["wci_symmetric_rv_ref_after_training"]
                 - by_arm[arm_id][s]["rv_final_after_training"] for s in seeds]),
            "dose_levels_separated": abs(
                _mean([by_arm[arm_id][s]["rv_final_after_training"] for s in seeds])
                - _mean([by_arm[sibling][s]["rv_final_after_training"] for s in seeds])),
        }
        arm_gates.append(
            evaluate_arm_gate(arm_id, ctx, list(PRECONDITION_SPECS), measured))

    gate = aggregate_arm_gates(arm_gates)

    # ---- per-level H2 closure readout ----
    per_level: Dict[str, Dict] = {}
    for (lvl, asym, infl_arm) in INFLATION_LEVELS:
        rv_this_run = arm_rv[infl_arm]
        closure = _closure_fraction(rv_this_run, lvl)
        per_level[lvl] = {
            "asymmetry": asym,
            "infl_arm": infl_arm,
            "rv_final": rv_this_run,
            "rv_final_794a": FULLLOOP_794A_RV_FINAL[lvl],
            "rv_final_smoke": SMOKE_RV_FINAL[lvl],
            "rv_final_853_confounded": CONFOUNDED_853_RV_FINAL[lvl],
            "closure_fraction": closure,
            "closes_meaningfully": bool(closure >= H2_CLOSURE_SUPPORT_FLOOR),
            "plateaus": bool(closure < H2_CLOSURE_PLATEAU_CEILING),
            "n_seeds_overconfident": sum(
                1 for s in seeds if by_arm[infl_arm][s]["absolutely_overconfident"]),
            "infl_score": arm_score[infl_arm],
        }

    # C1 (load-bearing): H2 is SUPPORTED iff the STEPS_PER_EP-based exposure increase
    # (parity total exposure with 853, but F1-firing count held at 794a's 30) closes a
    # meaningful fraction of the 794a-vs-smoke gap at BOTH doses.
    c1_h2_supported = bool(
        per_level["LO"]["closes_meaningfully"] and per_level["HI"]["closes_meaningfully"])
    # C2 (load-bearing): H2 is NOT SUPPORTED (a plateau) iff BOTH doses stay below the
    # plateau ceiling despite the parity-exposure budget -- the gap is not exposure-driven
    # (favours H1's F1-interaction explanation instead, consistent with 853's own reading
    # AND with this run's decoupled design confirming it independent of firing count).
    c2_h2_plateau = bool(
        per_level["LO"]["plateaus"] and per_level["HI"]["plateaus"])
    # C3 (diagnostic, non-load-bearing): dose-response direction preserved (more
    # asymmetry -> more overconfidence / lower rv), same check as 794a/853's own.
    c3_dose_response_monotone = bool(
        per_level["HI"]["rv_final"] < per_level["LO"]["rv_final"])

    criteria = [
        {"name": "C1_decoupled_exposure_closes_smoke_gap", "load_bearing": True,
         "passed": c1_h2_supported,
         "closure_lo": per_level["LO"]["closure_fraction"],
         "closure_hi": per_level["HI"]["closure_fraction"],
         "support_floor": H2_CLOSURE_SUPPORT_FLOOR},
        {"name": "C2_decoupled_exposure_plateaus", "load_bearing": True,
         "passed": c2_h2_plateau,
         "closure_lo": per_level["LO"]["closure_fraction"],
         "closure_hi": per_level["HI"]["closure_fraction"],
         "plateau_ceiling": H2_CLOSURE_PLATEAU_CEILING},
        {"name": "C3_dose_response_monotone", "load_bearing": False,
         "passed": c3_dose_response_monotone,
         "lo_rv": per_level["LO"]["rv_final"], "hi_rv": per_level["HI"]["rv_final"]},
    ]

    # ---- non-degeneracy, keyed to the owning arm's readiness gate ----
    criteria_by_arm = {
        "ARM_INFL_LO": ["C1_decoupled_exposure_closes_smoke_gap",
                        "C2_decoupled_exposure_plateaus",
                        "C3_dose_response_monotone"],
        "ARM_INFL_HI": ["C1_decoupled_exposure_closes_smoke_gap",
                        "C2_decoupled_exposure_plateaus",
                        "C3_dose_response_monotone"],
    }
    # C1/C2/C3 all read BOTH arms jointly, so they are only non-degenerate if BOTH arms
    # are green (a criterion owned by two arms is degenerate if either is red --
    # arm_criteria_non_degenerate keys per-arm, so intersect by hand).
    both_green = bool(gate["all_green"])
    raw_non_degen = {
        "C1_decoupled_exposure_closes_smoke_gap": both_green,
        "C2_decoupled_exposure_plateaus": both_green,
        "C3_dose_response_monotone": both_green,
    }
    criteria_non_degenerate = arm_criteria_non_degenerate(
        {"ARM_INFL_LO": list(raw_non_degen.keys())}, gate, raw_non_degen)

    # ---- self-route ----
    readiness_ok = bool(gate["non_degenerate"]) and both_green
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "inconclusive"
    elif c1_h2_supported:
        label = "h2_decoupled_exposure_closes_smoke_gap"
        outcome = "PASS"
        direction = "supports"
    elif c2_h2_plateau:
        label = "h2_decoupled_exposure_insufficient_explanation_plateau"
        outcome = "FAIL"
        direction = "inconclusive"
    else:
        label = "h2_decoupled_exposure_partial_ambiguous"
        outcome = "FAIL"
        direction = "inconclusive"

    per_claim = {"SD-076": direction if readiness_ok else "unknown"}

    return {
        "outcome": outcome,
        "label": label,
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "gate": gate,
        "arm_gates": arm_gates,
        "arm_overconfidence_score": arm_score,
        "arm_calibration_ratio": arm_ratio,
        "arm_true_error_ref": arm_true_err,
        "arm_rv_final": arm_rv,
        "per_level": per_level,
        "readiness_ok": readiness_ok,
        "thresholds": {
            "H2_CLOSURE_SUPPORT_FLOOR": H2_CLOSURE_SUPPORT_FLOOR,
            "H2_CLOSURE_PLATEAU_CEILING": H2_CLOSURE_PLATEAU_CEILING,
            "DOSE_SEPARATION_FLOOR": DOSE_SEPARATION_FLOOR,
            "RV_LIVE_FLOOR": RV_LIVE_FLOOR,
            "RECALIB_MOVE_FLOOR": RECALIB_MOVE_FLOOR,
            "INFLATION_MOVE_FLOOR": INFLATION_MOVE_FLOOR,
            "PRECISION_INIT_BASELINE": PRECISION_INIT_BASELINE,
        },
        "reference_values": {
            "fullloop_794a_rv_final": FULLLOOP_794A_RV_FINAL,
            "smoke_rv_final": SMOKE_RV_FINAL,
            "confounded_853_rv_final_context_only": CONFOUNDED_853_RV_FINAL,
        },
    }


# -------------------------------------------------------------------------- main --
def run_experiment(dry_run: bool = False) -> Dict:
    t0 = time.perf_counter()
    n_train = 4 if dry_run else N_TRAIN_EPS
    n_eval = 1 if dry_run else N_EVAL_EPS
    n_seeds = 2 if dry_run else N_SEEDS
    train_steps = 20 if dry_run else STEPS_PER_EP
    eval_steps = 10 if dry_run else EVAL_STEPS_PER_EP
    seeds = list(range(n_seeds))

    # Design-time proof #1: this probe's whole point is a substantially larger STEPS_PER_EP
    # than 794a/853 -- catch a copy-paste regression back to the old baseline before any
    # compute is spent.
    if not dry_run:
        assert STEPS_PER_EP >= 3 * STEPS_PER_EP_794A, (
            f"H2 redesign requires STEPS_PER_EP >= 3x 794a's per-episode step count "
            f"({3 * STEPS_PER_EP_794A}); got {STEPS_PER_EP}"
        )
        # Design-time proof #2: THE CONFOUND FIX ITSELF -- N_TRAIN_EPS must stay at 794a's
        # original value, or this script silently reintroduces 853's own confound (raising
        # the F1-firing count alongside exposure). This is the single assertion this whole
        # redesign exists to guarantee.
        assert N_TRAIN_EPS == N_TRAIN_EPS_794A, (
            f"H2 redesign requires N_TRAIN_EPS == 794a's original budget "
            f"({N_TRAIN_EPS_794A}) so F1-firing count stays unconfounded with exposure; "
            f"got {N_TRAIN_EPS}. Raising N_TRAIN_EPS here reproduces V3-EXQ-853's "
            f"confound -- see module docstring 'THE CONFOUND THIS REDESIGN FIXES'."
        )

    # Design-time proof: refuse before compute if any gate is structurally unsatisfiable.
    assert_no_structurally_unsatisfiable_gate(list(PRECONDITION_SPECS), ARM_CONTEXTS)

    cells: List[Dict] = []
    for arm in ARMS:
        for seed in seeds:
            cells.append(
                _run_arm_seed(arm, seed, n_train, n_eval, train_steps, eval_steps,
                              dry_run=dry_run))

    adj = _analyse(cells, seeds)
    adj["cells"] = cells
    adj["seeds"] = seeds
    adj["elapsed_seconds"] = time.perf_counter() - t0
    adj["t0_perf"] = t0
    adj["config_n"] = {"train_steps_per_ep": train_steps, "eval_steps_per_ep": eval_steps,
                       "n_train_eps": n_train, "n_eval_eps": n_eval, "n_seeds": n_seeds}
    return adj


def main(dry_run: bool = False) -> Dict:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    adj = run_experiment(dry_run=dry_run)
    outcome = adj["outcome"]

    print("", flush=True)
    print(f"label={adj['label']} outcome={outcome} readiness_ok={adj['readiness_ok']}",
          flush=True)
    for lvl in ("LO", "HI"):
        pl = adj["per_level"][lvl]
        print(f"  {lvl:<3} asym={pl['asymmetry']:.2f} rv_final={pl['rv_final']:.6f} "
              f"(794a={pl['rv_final_794a']:.6f} smoke={pl['rv_final_smoke']:.6f} "
              f"853_confounded={pl['rv_final_853_confounded']:.6f}) "
              f"closure={pl['closure_fraction']:+.3f}", flush=True)
    for c in adj["criteria"]:
        lb = " (load-bearing)" if c["load_bearing"] else ""
        print(f"  {c['name']}: {'PASS' if c['passed'] else 'FAIL'}{lb}", flush=True)

    if dry_run:
        print("DRY_RUN_COMPLETE", flush=True)
        return {"outcome": outcome, "manifest_path": None, "run_id": run_id}

    full_config = {
        "grid_size": GRID_SIZE,
        "train_steps_per_ep": adj["config_n"]["train_steps_per_ep"],
        "eval_steps_per_ep": adj["config_n"]["eval_steps_per_ep"],
        "n_train_eps": adj["config_n"]["n_train_eps"],
        "n_train_eps_794a_baseline": N_TRAIN_EPS_794A,
        "train_steps_per_ep_794a_baseline": STEPS_PER_EP_794A,
        "n_eval_eps": adj["config_n"]["n_eval_eps"],
        "n_seeds": adj["config_n"]["n_seeds"],
        "lr": LR,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "use_rem_precision_broadcast": False,
        "inflation_asymmetry_lo": INFLATION_ASYMMETRY_LO,
        "inflation_asymmetry_hi": INFLATION_ASYMMETRY_HI,
        "inflation_rv_floor": INFLATION_RV_FLOOR,
        "waking_confidence_rv_floor_relative_frac": INFLATION_RV_FLOOR_RELATIVE_FRAC,
        "waking_confidence_rv_floor_mode": INFLATION_RV_FLOOR_MODE,
        "waking_confidence_rv_floor_softness": INFLATION_RV_FLOOR_SOFTNESS,
        "sleep_loop_episodes_K": 1,
        "tonic_5ht_enabled": True,
        "arms": [{"arm_id": a[0], "inflation_asymmetry": float(a[1])} for a in ARMS],
        "env": {"num_hazards": 3, "num_resources": 3, "hazard_harm": 0.04,
                "proximity_harm_scale": 0.12, "proximity_benefit_scale": 0.10,
                "use_proxy_fields": True, "resource_respawn_on_consume": True},
        "seeds": adj["seeds"],
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": adj["evidence_direction"],
        "evidence_direction_per_claim": adj["evidence_direction_per_claim"],
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "dose_key": "asymmetry",
        "fanout_source": FANOUT_SOURCE,
        "fanout_hypothesis": FANOUT_HYPOTHESIS,
        "fanout_target_run_id": FANOUT_TARGET_RUN_ID,
        "redesign_source": REDESIGN_SOURCE,
        "supersedes": SUPERSEDES_RUN_ID,
        "supersedes_queue_id": SUPERSEDES_QUEUE_ID,
        "interpretation": {
            "label": adj["label"],
            "preconditions": adj["gate"]["adjudication_preconditions"],
            "criteria": adj["criteria"],
            "criteria_non_degenerate": adj["criteria_non_degenerate"],
        },
        "per_arm_gate": adj["gate"]["per_arm_gate"],
        "non_degenerate": adj["gate"]["non_degenerate"],
        "degeneracy_reason": adj["gate"]["degeneracy_reason"],
        "aggregates": {
            "arm_overconfidence_score": adj["arm_overconfidence_score"],
            "arm_calibration_ratio": adj["arm_calibration_ratio"],
            "arm_true_error_ref": adj["arm_true_error_ref"],
            "arm_rv_final": adj["arm_rv_final"],
            "per_level": adj["per_level"],
            "readiness_ok": adj["readiness_ok"],
        },
        "thresholds": adj["thresholds"],
        "reference_values": adj["reference_values"],
        "arm_results": adj["cells"],
        "per_seed_cells": adj["cells"],
        "elapsed_seconds": adj["elapsed_seconds"],
        "notes": (
            "DIAGNOSTIC GOV-FANOUT-1 H2 leg (exposure/world axis), REDESIGNED to fix a "
            "design confound identified in the just-failed V3-EXQ-853 leg's own autopsy "
            "(failure_autopsy_V3-EXQ-853_2026-08-01.md section 6): N_TRAIN_EPS is not a "
            "single-axis manipulation for H2 because SleepLoopManager fires the F1 REM "
            "recalibration cycle once per episode (K=1), so raising N_TRAIN_EPS also "
            "raises the F1-firing count in lockstep, confounding H2's exposure axis with "
            "the sibling H1 leg's recalibration-count axis. This run fixes that by "
            "holding N_TRAIN_EPS at 794a's original value (30, so F1 fires exactly 30 "
            "times, unchanged from 794a and the sibling H1 leg) and instead raising "
            "STEPS_PER_EP (200 -> 1000, 5x) to reach the SAME total training-step "
            "exposure as 853's own 5x N_TRAIN_EPS change (30000 waking steps in both "
            "runs), so this run and 853 are directly comparable at matched total "
            "exposure but with the confound removed. Tests whether that decoupled "
            "exposure increase closes the gap between V3-EXQ-794a's full-loop rv_final "
            "(0.003998 LO / 0.003870 HI) and the SD-076 headroom repair's own isolated "
            "validation smoke (rv_final 0.0025377 LO / 0.0021031 HI) at the identical "
            "measured error scale -- an exposure/dose-duration hypothesis (H2), distinct "
            "from the sibling leg's F1-interaction-damping hypothesis (H1, V3-EXQ-850) "
            "and the driver's own pre-registered wrong-mechanism-form fallback (H3, "
            "already resolved via /lit-pull, weakened not eliminated). Single-axis: only "
            "ARM_INFL_LO/ARM_INFL_HI run, Phase 7 broadcast held OFF throughout (not this "
            "leg's manipulated axis), F1/REM precision recalibration held ON, firing "
            "exactly 30 times per arm (matching 794a, NOT 853's 150). DIAGNOSTIC => "
            "excluded from governance confidence/conflict scoring. claim_ids=[SD-076] "
            "only -- MECH-204's broadcast correction is never engaged in this design, so "
            "it yields no MECH-204 evidence either way (see module docstring "
            "'CLAIM TAGGING')."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=full_config,
        seeds=adj["seeds"],
        script_path=Path(__file__),
        started_at=adj["t0_perf"],
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"Result written to: {out_path}", flush=True)
    return {"outcome": outcome, "manifest_path": str(out_path), "run_id": run_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true", help="Quick smoke test (2 seeds, tiny).")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    _outcome = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=result["manifest_path"],
        run_id=result["run_id"],
        dry_run=args.dry_run,
    )
    sys.exit(0)
