"""
V3-EXQ-1067 (MECH-266 / SD-032a): squash-vs-clamp affinity BOUNDING-OPERATOR cap
sweep for external_task mode-occupancy. DIAGNOSTIC.

RED-TEAM (/queue-experiment Step 4.5, model fable): **BLOCKING -> RESOLVED by
user decision 2026-09-19T23:52:40Z (OPTION A).** The finding was real and is
recorded at GFLAG-0370: the bounding operator is applied to EVERY
affinity_weights signal (salience_coordinator.py:616-636), and
`external_task_drive` is one of them (agent.py:2760-2765, weight 3.0). At
sigma = cap the squash is NOT an identity on sub-cap signals -- it returns 0.800x
at 0.25*cap, 0.667x at 0.5*cap and 0.500x AT the cap. Because the still-open
boolean commitment latch (agent.py:7870) pins that engagement signal at exactly
1.0, swapping clamp->squash moved the external_task logit by -0.96..-1.50 while
the dacc_pe degeneracy fix under test moved its logit by only -0.034..-0.173 --
an 8x to 28x confound, in the direction of LESS external_task occupancy. With
only two arms neither a PASS nor the pre-registered NULL was attributable.

THE FIX, as decided: a THIRD, GAIN-MATCHED CLAMP arm (see THREE ARMS below), so
the gain component and the gradedness component are measured separately. Two
lower-severity red-team findings are fixed in the same pass: cells are now
RNG- AND ENV-PAIRED across bound arms (so the paired-delta non-degeneracy guard
is no longer vacuous), and each cell builds its own env (so the bound arms no
longer share one stateful, never-rebuilt env, and the bound arm is nested
INNERMOST). Occupancy remains the LOAD-BEARING criterion exactly as ratified.

WHY THIS RUN EXISTS
-------------------
`mode-governance-engagement` (REE_assembly evidence/planning/substrate_queue.json)
is three coupled items. ITEM (1) -- the bounding operator -- COMPLETED on
2026-09-19: `SalienceCoordinator.tick()`'s hard box clamp now sits behind a mode
selector (ree-v3 956e213f81), and sigma defaults to the configured cap
(524ef52484). The entry's own words: "ITEM (1) IS NOW COMPLETE, operator AND its
sigma rule. Nothing further is owed on the bounding operator itself."

That entry then names THIS run as the validation it unblocks, verbatim:

    "NOW UNBLOCKED ... the cap/operator sweep in the V3-EXQ-934 / 935a lineage --
     a successor to experiments/v3_exq_934_mech266_cap_sweep_mode_occupancy.py
     that sweeps the cap with salience_affinity_bound_mode='squash' against the
     clamp arm, to test whether a bounded-but-GRADED input admits the mixed
     regime the bang-bang clamp could not. It must import
     experiments/_lib/regime_occupancy_gate.py rather than re-deriving min()."

THE PROBLEM THE SQUASH IS BUILT TO FIX (measured, not asserted)
---------------------------------------------------------------
The box clamp is DEGENERATE at the cap: two different large dacc_pe values map to
the identical bounded value, so the affinity logit cannot distinguish them. A
pre-authoring probe against the REAL SalienceCoordinator (sigma = cap default)
measured, at every cap this run sweeps:

  cap    clamp bound pe=16 / pe=17 (separation)   squash bound pe=16 / pe=17 (separation)
  0.75   0.750000000000 / 0.750000000000 (0.0)    0.716417910448 / 0.718309859155 (0.00189)
  1.00   1.000000000000 / 1.000000000000 (0.0)    0.941176470588 / 0.944444444444 (0.00327)
  1.25   1.250000000000 / 1.250000000000 (0.0)    1.159420289855 / 1.164383561644 (0.00496)
  1.50   1.500000000000 / 1.500000000000 (0.0)    1.371428571429 / 1.378378378378 (0.00695)
  1.75   1.750000000000 / 1.750000000000 (0.0)    1.577464788732 / 1.586666666667 (0.00920)

i.e. the clamp separation is EXACTLY ZERO at every swept cap and the squash
separation is not. On the live coordinator the same override moves the whole soft
mode vector (operating_mode['external_task'] differs between operators by up to
0.045 at cap 0.75). V3-EXQ-935a independently recorded the clamp-era signature of
this degeneracy: ext_margin_mean LINEAR in cap at R^2 0.9996-0.9999.

WHY sigma = cap (do NOT "improve" it)
-------------------------------------
sigma sets the SMALL-SIGNAL GAIN as well as where saturation bites: the derivative
at the origin is cap/sigma. At sigma == cap that slope is exactly 1 (measured
0.99999987-0.99999994 at the five swept caps), so every SUB-cap signal passes
through precisely as the legacy box clamp passed it and the ONLY behavioural
change is at the top end, where the clamp was degenerate. That is what lets this
sweep attribute an effect to the OPERATOR change rather than to a simultaneous
gain change. sigma is therefore left at the landed default (None -> cap) and is
NOT swept; an explicit sigma is what a CALIBRATION sweep varies, and an ADAPTIVE
sigma was considered and explicitly declined on the record ("IF IT IS EVER WANTED
IT IS A NEW substrate_queue ITEM").

THE THREE BOUND ARMS (the manipulation)
---------------------------------------
All three run at every swept cap, on both rail arms, on shared seeds, on clones
of the SAME trained agent, with RNG and env paired cell-for-cell.

  clamp_baseline      operator = clamp,  drive weight = 3.0 (unchanged)
                      The V3-EXQ-934 BASELINE. Degenerate at the cap; full drive.
  squash              operator = squash, drive weight = 3.0 (unchanged)
                      Graded at the cap; drive gain CUT as a side effect.
  clamp_gain_matched  operator = clamp,  drive weight = w'(cap)
                      Drive gain cut to match `squash`, but the bound is still
                      DEGENERATE at the cap. This is the control the red-team
                      finding requires: it carries the gain component WITHOUT the
                      gradedness component.

w'(cap) is set so the external_task_drive logit contribution MATCHES the squash
arm at the dominant operating point, engagement e = 1.0 (the value the boolean
commitment latch pins it to whenever beta is elevated -- see
et_drive_saturated_frac, which measures how dominant that point actually is):

    squash contributes  3.0 * cap*e/(cap+e)          -> 3.0 * cap/(cap+1) at e=1
    clamp  contributes  w'  * min(e, cap)            -> w'  * min(1, cap) at e=1
    =>  w'(cap) = 3.0 * (cap/(cap+1)) / min(1.0, cap)

For every cap >= 1.0 this is exactly the `3*cap/(cap+1)` the decision specified.
It differs ONLY at cap = 0.75, the one sub-1.0 point in the sweep, where the
CLAMP itself already attenuates e=1.0 to 0.75 and the literal formula would
therefore under-deliver the match by ~0.32 of logit (1.286 wanted vs 0.964
delivered) -- at precisely the cap where V3-EXQ-934's seed 42 produced its only
mixed cell. The min() term is a derivation-level correction in service of the
decision's stated purpose ("so the gain-cut component is measured separately from
the gradedness component"), not a change to it; both forms are recorded in the
manifest under config.gain_matched_drive_weight_by_cap so the choice is auditable.

RESIDUAL, STATED: the match is exact only at e = 1.0. Where engagement is
UNLATCHED (e < 1) the squash and the gain-matched clamp diverge slightly. That is
accepted rather than hidden -- et_drive_saturated_frac records what fraction of
ticks sat at the matched point, and it is expected near 1.0 precisely because
item (2) is still open.

HOW THE THREE ARMS ARE READ TOGETHER (pre-registered; this is the attribution)
------------------------------------------------------------------------------
Three occupancy gates are computed on the PRIMARY rail arm (ARM_SYMMETRIC), one
per bound arm. `squash` graded is the LOAD-BEARING criterion and decides the
OUTCOME, unchanged. The other two are ATTRIBUTION: they decide what a PASS or a
null MEANS, and they alone drive `evidence_direction_per_claim`. Every cell below
is entailed by something the run measures; no cell writes a direction the
measurement cannot support. (Rows marked [RT2] were added or corrected by the
second red-team pass -- see RED-TEAM PASS 2 below.)

  condition                                        attribution            SD-032a
  ---------------------------------------------------------------------------
  clamp_baseline graded (checked FIRST)            baseline_divergence    non_contributory
      -> divergence from V3-EXQ-934; the contrast is not interpretable
         until explained, so nothing else is read.
  squash graded + gain_matched graded              gain                   non_contributory
      -> the mixed regime is ALSO reachable with a DEGENERATE bound once
         the drive gain is cut, so the PASS does not evidence gradedness.
  squash graded + gain_match_valid FALSE   [RT2]   undetermined_control_  non_contributory
      -> the control did not actually reproduce      mismatched
         the gain cut on the realized engagement trace, so "the control
         failed to grade" cannot be read as "gradedness is what mattered".
  squash graded + gain_matched NOT graded          gradedness             SUPPORTS
      + gain_match_valid TRUE
      -> the ONLY path to `supports`. The graded bound is what buys the
         mixed regime, and the control demonstrably controlled for gain.
  gain_matched graded + squash NOT graded  [RT2]   gain                   non_contributory
      -> the mixed regime is reached by the gain cut alone. Previously
         UNROUTED: it fell through to a branch whose route_reason asserted
         "no_arm_graded" (false) and recorded `weakens` (wrong).
  none graded + occupancy static across all        item_2_commitment_     weakens
      three arms + et_sat_mean > 0                   latch
      -> residual discreteness that is cap-, operator- AND gain-independent,
         with the latch demonstrably firing. Isolates entry item (2).
  none graded + occupancy static + et_sat   [RT2]  static_but_latch_not_  non_contributory
      == 0                                           implicated
      -> static, but the boolean latch NEVER FIRED, so it cannot be the
         source. Naming it would be an unsupported attribution.
  none graded + occupancy shifts between arms      mixed_gain_and_        weakens
                                                     gradedness_insufficient
      -> occupancy responds to gain and/or gradedness but never lands a
         reproducible mixed band; bang-bang persists.

The two decomposition quantities are recorded as telemetry with NO threshold:
  occ_shift_gain        = mean |occ(clamp_gain_matched) - occ(clamp_baseline)|
  occ_shift_gradedness  = mean |occ(squash) - occ(clamp_gain_matched)|
  occ_shift_total       = mean |occ(squash) - occ(clamp_baseline)|
(each over the PRIMARY rail arm's (seed, cap) cells). They are meaningful ONLY
because the cells are RNG- and env-paired; see PAIRING below.

PAIRING (red-team findings 2 and 3)
------------------------------------
Every cell derives a per-cell RNG seed from (seed, cap, rail_arm) ONLY -- never
from the bound arm -- and calls `reset_all_rng(cell_seed)` at cell entry, then
builds its OWN env seeded with that same cell seed. So the three bound arms at a
given (seed, cap, rail_arm) start from bit-identical RNG state and an identical
env layout, and differ ONLY in the manipulated variables. Consequences:
  - the paired deltas above measure the manipulation, not run-to-run noise;
  - the 1e-6 `operator_manipulation_landed` guard is a genuine bit-level identity
    test rather than a threshold that any two stochastic cells would clear
    (banked V3-EXQ-934 cells differing only in rails already differed by 1.7e-3);
  - bound-arm ORDER cannot confound, since no env state carries between cells.
This DEPARTS from V3-EXQ-934's OS-entropy eval env on purpose: 934 shared one env
across its cells, which is exactly what made its cells unpairable. The clamp
baseline arm still reproduces 934's DESIGN (same operator, same weight, same cap
band, same seeds, same rails); it is not expected to reproduce 934's cell values
bit-for-bit, and nothing here compares against them numerically.

DESIGN
------
Cross the OPERATOR with the CAP on SHARED seeds, at EVAL time, on clones of ONE
trained curriculum agent per seed. `SalienceCoordinator.tick()` reads
`self.config.affinity_input_cap`, `.affinity_bound_mode` and
`.affinity_squash_sigma` LIVE at every tick, so overriding them on a clone changes
arbitration with no retraining -- the same train-once/sweep-on-clones pattern 934
and 467e use, and empirically confirmed before authoring.

  BOUND_ARMS -- the MANIPULATION; THREE arms, defined under THE THREE BOUND ARMS
      above. `clamp_baseline` is the V3-EXQ-934 BASELINE and must reproduce its
      DESIGN: the entry's own build constraint is "Keep the existing clamp
      reachable behind a mode selector so the V3-EXQ-934 baseline stays
      reproducible", and the operator contract file's C1 pins bit-identity of the
      default path. `clamp_gain_matched` is the attribution control added under
      option A; it is what separates the gain component from the gradedness
      component, and without it neither a PASS nor a null is attributable.
  CAP_SWEEP = [0.75, 1.0, 1.25, 1.5, 1.75]  -- 934's band, INHERITED unchanged.
      Transferability caveat, stated rather than papered over: this band was
      chosen from a CLAMP-era synthetic probe. sigma = cap fixes the small-signal
      gain at exactly 1, which is the property that makes it arguably
      transferable, but that is an argument, not a registration. A different band
      would be a new design decision and is not taken here.
  ARM_LABELS = ["ARM_SYMMETRIC", "ARM_ASYM_STICKY_TASK"]  -- 934's two rail arms,
      unchanged. The occupancy bar is defined on ARM_SYMMETRIC (primary); the
      sticky arm deliberately pushes toward saturation and is context only.
  SEEDS = [42, 43, 44]  -- 934's seeds, kept DELIBERATELY. The usual
      substitute-seed-44-on-reef caution is overridden here for a recorded reason:
      the clamp arm's whole job is to reproduce 934's banked cells, which requires
      the identical seeds, and 934's seed 44 guard-PASSED on this exact curriculum
      with the highest contact rate of the three (0.3064, z_goal 0.441).

Training is held at AFFINITY_INPUT_CAP_TRAIN = 2.0 under the CLAMP (934's / 464e's
construction) so the trained substrate is bit-comparable to the banked reference
and BOTH eval operators are applied to the same trained agent. use_closure_operator
OFF (closure injects a confounding mode-switch signal).

Cells = BOUND_MODE x CAP x ARM per seed = 2 x 5 x 2 = 20 (934 had 10).

RED-TEAM PASS 2 (model fable) -- CONTESTED, all four findings verified and fixed
--------------------------------------------------------------------------------
The revised three-arm design was re-reviewed once (licensed: the BLOCKING finding
had changed the manipulation). Verdict CONTESTED: the load-bearing criterion
discriminates and the three-arm fix works at its derivation point, but four
defects in the ATTRIBUTION half were found. All four were reproduced against
source before being fixed.

F1 ROUTING GAP. The (squash NOT graded, gain_matched graded) cell was unrouted:
   it fell to the final else, whose route_reason asserted "no_arm_graded" -- false
   -- and recorded SD-032a `weakens`, when the table's own logic makes it a
   GAIN reading. Confirmed by replaying the branch chain over all four gate
   combinations, and it is the shape the authoring smoke produced (cap 1.0
   ARM_SYMMETRIC: gain_matched 0.1944 mixed, squash 1.0 saturated). FIXED: it has
   its own branch and routes non_contributory.

F2 THE GAIN MATCH HOLDS ONLY AT e = 1.0. For every cap >= 1.0 at engagement below
   1, the control carries LESS external_task gain than the squash (at cap 1.0,
   e = 0.5: squash 1.00 vs control 0.75), so "the control did not grade" can be
   caused by the control OVER-cutting rather than by gradedness mattering -- a
   false-positive path to `supports`. An earlier version of this file called that
   residual "conservative"; that was WRONG and is corrected in
   gain_match_quality_note. FIXED: each cell now records the REALIZED drive
   contribution under all three arm definitions on its own engagement trace, and
   the gradedness attribution is gated on `gain_match_valid` -- the residual
   mismatch must be smaller than the gain cut the control exists to reproduce.
   The 1.0 boundary is DEFINITIONAL (a control whose error exceeds the effect it
   controls for has not controlled for it), not a tuned threshold on the DV.

F3 THE LATCH ATTRIBUTION WAS NOT ENTAILED. "occupancy static across all three
   arms" was routed to the item-(2) commitment latch unconditionally, but that
   reading is also produced by engagement sitting near ZERO (goal inactive), in
   which case the latch never fired and cannot be the source. FIXED: the latch
   attribution now requires et_sat_mean > 0; otherwise it routes
   `static_but_latch_not_implicated`, non_contributory.

F4 THE DRIVE-READINESS PRECONDITION COULD NOT SEE A DEAD DRIVE -- an INHERITED
   defect, carried verbatim from V3-EXQ-934. Measured on the real coordinator with
   the external_task_drive signal at EXACTLY 0.0, the external_task margin still
   reads 0.3287-0.4950 across the swept caps, 6.6x to 9.9x the 0.05 floor, because
   the margin is floored by `external_task_bias = 1.0` rather than by the drive.
   So the precondition passed through the very failure its own description names
   (a goal_state drop hard-gating engagement to 0.0 -- the V3-EXQ-464d signature).
   FIXED: a second precondition, `external_task_drive_signal_nonzero`, asserts the
   drive's OWN value (a zero-test, not a tuned threshold), and the margin
   precondition's description now states its real scope. Worth governance
   attention on its own account, since V3-EXQ-934 carries the margin precondition
   alone.

DEPENDENT VARIABLE (LOAD-BEARING) -- OCCUPANCY
----------------------------------------------
`fraction_in_external_task` (discrete committed-mode occupancy), read through
`experiments/_lib/regime_occupancy_gate.evaluate_regime_occupancy_gate`.

THE BAR IS TRANSCRIBED, NOT INVENTED -- from this entry's own failure_record
target for V3-EXQ-934: "A COMMON cap value yielding per-arm occupancy in
(0.1, 0.9) on >= 2/3 seeds on ARM_SYMMETRIC, with a mixed band at least 2 grid
steps wide", which the gate module implements as: GRADED iff a run of
>= min_adjacent (2) CONSECUTIVE swept values each reads mixed on
>= min_seed_fraction (2/3) of the seeds measured at that value.

CALL SHAPE (the trap that cost 934 its routing): the gate is called ONCE per
(bound_arm, rail_arm) over ALL (seed, cap) cells, with `seed` AND `sweep_value`
populated -- never per-seed with booleans counted afterwards. The entry: "Do NOT
call per-seed and count booleans afterwards -- that is precisely the 934 shape
whose collapse of 'which cap' produced the false routing." 934's own
`_regime_for_arm` is that defective shape (it builds OccupancyCells with neither
seed nor sweep_value) and is deliberately NOT inherited.

LOAD-BEARING CRITERION: the SQUASH x ARM_SYMMETRIC gate returns graded == True.
The clamp x ARM_SYMMETRIC gate is the recorded 934 BASELINE and is reported, NOT
load-bearing.

TELEMETRY, RECORDED WITH NO THRESHOLD (user decision 2026-09-19T22:12:50Z)
--------------------------------------------------------------------------
The continuous pre-argmax margin is RECORDED as telemetry and scored against NO
threshold. This is deliberate and is what makes a third occupancy failure
ATTRIBUTABLE rather than merely disappointing:

  ext_margin_mean/p10/p50/p90/max -- the continuous operating_mode['external_task'].
  margin_cap_linearity_r2 -- R^2 of a least-squares fit of ext_margin_mean vs cap,
      per (seed, arm, bound_arm). 935a recorded 0.9996-0.9999 under the CLAMP;
      whether the squash degrades it is REPORTED, never gated.
  operator margin deltas -- paired squash-minus-clamp ext_margin_mean per
      (seed, cap, arm).
  COMMITMENT-LATCH ATTRIBUTION (entry item (2), still OPEN):
      et_drive_saturated_frac -- fraction of eval steps where the injected
          external_task_drive engagement is EXACTLY 1.0. agent.py:7870 computes
          _et_commit = commit_w * float(beta_gate.is_elevated), a BOOLEAN latch,
          then :7881 clips _et_commit + _et_prox into [0,1] -- so at the default
          commit_weight 1.0 ANY beta elevation saturates engagement to exactly 1.0
          regardless of goal proximity. This is a cap- AND operator-INDEPENDENT
          discreteness source.
      dacc_pe_mean / dacc_pe_abs_max / dacc_pe_over_cap_frac -- whether the
          operator BITES at all (it acts only where |signal| is comparable to or
          above the cap). external_task_drive is already bounded to [0,1], so at
          any cap >= 1.0 BOTH operators are no-ops on it (934's own
          failure_record); the squash bites on dacc_pe (~16-17), i.e. on how hard
          internal_planning dominates.

WHAT A NULL MEANS (pre-registered, so a third failure is not a surprise)
------------------------------------------------------------------------
The entry predicts in writing that a graded operator ALONE may not be enough:
"item (2) commitment-term grading -- agent.py's _et_commit = commit_w *
float(beta_gate.is_elevated) remains a boolean latch and a cap-INDEPENDENT
discreteness source, so a graded bounding operator alone does NOT make the
register graded end-to-end". So a squash-not-graded read, WITH the operator
manipulation demonstrably landing on the continuous margin, is an INFORMATIVE
null: it isolates item (2) (and any other cap-independent source) as the residual
discreteness, rather than leaving the operator under suspicion. That is
pre-registered here as the point, not read out of the manifest afterwards.

PER-CLAIM DIRECTION: diagnostic, EXCLUDED from governance confidence/conflict
scoring. MECH-266 non_contributory (this establishes the measurement precondition
for the over-binding test; it does not itself measure over-binding). SD-032a
supports if the graded operator admits the mixed regime, weakens if it provably
does not while demonstrably landing (favouring a residual structural/latch
source), else non_contributory.

THIS PROMOTES NOTHING. The entry's governance line for this lineage is "PROMOTES
NOTHING. claims.yaml untouched; MECH-266 / SD-032a pending_retest_after_substrate
holds". It validates ITEM (1) ONLY; items (2) (commitment-term grading) and (3)
(production default) remain OPEN, and the production defaults
(salience_affinity_input_cap = None, use_external_task_drive = False) are NOT
flipped here -- that IS item (3).

claim_ids: MECH-266, SD-032a.
experiment_purpose: diagnostic
predecessor: V3-EXQ-934 (successor; NOT a supersede -- 934 measured the CLAMP
             across the same band and that reading stands).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
from ree_core.cingulate.salience_coordinator import (  # noqa: E402
    AFFINITY_BOUND_CLAMP,
    AFFINITY_BOUND_SQUASH,
    bound_affinity_input,
)
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
    OccupancyCell,
    evaluate_regime_occupancy_gate,
)
from experiments._lib.arm_fingerprint import reset_all_rng  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1067_mech266_squash_vs_clamp_cap_sweep"
QUEUE_ID = "V3-EXQ-1067"
CLAIM_IDS: List[str] = ["MECH-266", "SD-032a"]
EXPERIMENT_PURPOSE = "diagnostic"
PREDECESSOR = "V3-EXQ-934 (successor; NOT a supersede)"

# Same exemption 934 carries, for the same two readiness anchors. They are
# ordinary UPSTREAM gates -- "did the curriculum train" and "does the external_task
# drive engage at all" -- NOT the "positive control reproduces a known degenerate
# signature" pattern the V3-EXQ-778d readiness-anchor reachability guard targets.
# Reachable BY CONSTRUCTION: 934 cleared the 603n contact guard on this exact
# curriculum on 3/3 seeds, and the pre-authoring probe measured the continuous
# margin at 0.33-0.50 across the swept band under BOTH operators, ~7-10x the 0.05
# floor. The 778d failure mode (an unmeetable predicate that mislabels an
# instrument gap as a substrate verdict) does not apply: a margin-PASS with
# saturated occupancy under BOTH operators routes an informative,
# pre-registered null that names the residual discreteness source, never a
# starved false-falsification.
ANCHOR_REACHABILITY_EXEMPT = (
    "Readiness anchors are ordinary upstream gates (curriculum-trained; drive-engages), "
    "not degeneracy-reproduction anchors; thresholds reachable by construction (934 "
    "cleared the contact guard 3/3 on this curriculum; the 0.05 margin floor cleared "
    "~7-10x by the pre-authoring probe under BOTH operators). A margin-pass with "
    "saturated occupancy routes a VALID pre-registered null isolating the "
    "cap-independent commitment latch, never a starved false-falsification."
)

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_SQUASH_VS_CLAMP_CAP_SWEEP"

_ZG = ZGoalStreamAccumulator()

MODE_NAMES = [
    "external_task",
    "internal_planning",
    "internal_replay",
    "offline_consolidation",
]
STICKY_MODE = "external_task"
STICKY_EXIT = 0.05
LOOSE_EXIT = 0.90

# THE MANIPULATION -- three bound arms (see THE THREE BOUND ARMS in the docstring).
BOUND_ARM_CLAMP = "clamp_baseline"
BOUND_ARM_SQUASH = "squash"
BOUND_ARM_CLAMP_GAINMATCHED = "clamp_gain_matched"
BOUND_ARMS: List[str] = [
    BOUND_ARM_CLAMP, BOUND_ARM_SQUASH, BOUND_ARM_CLAMP_GAINMATCHED,
]
# Which coordinator operator each bound arm selects.
BOUND_ARM_OPERATOR: Dict[str, str] = {
    BOUND_ARM_CLAMP: AFFINITY_BOUND_CLAMP,
    BOUND_ARM_SQUASH: AFFINITY_BOUND_SQUASH,
    BOUND_ARM_CLAMP_GAINMATCHED: AFFINITY_BOUND_CLAMP,
}
# sigma stays at the landed DEFAULT (None -> sigma = cap). NOT a swept knob here.
AFFINITY_SQUASH_SIGMA: Optional[float] = None

# The external_task_drive affinity weight the substrate is TRAINED and evaluated
# with on the two unmatched arms.
EXTERNAL_TASK_DRIVE_AFFINITY_WEIGHT = 3.0


def gain_matched_drive_weight(cap: float) -> float:
    """The clamp-arm external_task_drive weight that MATCHES the squash arm's
    logit contribution at engagement e = 1.0 (the point the still-open boolean
    commitment latch pins engagement to; agent.py:7870/:7881).

        squash contributes  w * cap*e/(cap+e)   -> w * cap/(cap+1)   at e = 1
        clamp  contributes  w' * min(e, cap)    -> w' * min(1, cap)  at e = 1
        =>  w'(cap) = w * (cap/(cap+1)) / min(1.0, cap)

    Equals the decision's literal `3*cap/(cap+1)` at every cap >= 1.0. The min()
    term corrects ONLY the sub-1.0 case (cap = 0.75 here), where the clamp itself
    already attenuates e = 1.0 to 0.75 so the literal form would under-deliver the
    match -- at exactly the cap where V3-EXQ-934's seed 42 produced its only mixed
    cell. See the docstring; both forms are recorded in the manifest."""
    w = EXTERNAL_TASK_DRIVE_AFFINITY_WEIGHT
    return float(w * (float(cap) / (float(cap) + 1.0)) / min(1.0, float(cap)))


def bound_arm_drive_weight(bound_arm: str, cap: float) -> float:
    """Per-cell external_task_drive affinity weight for this bound arm."""
    if bound_arm == BOUND_ARM_CLAMP_GAINMATCHED:
        return gain_matched_drive_weight(cap)
    return EXTERNAL_TASK_DRIVE_AFFINITY_WEIGHT


def cell_rng_seed(seed: int, cap: float, rail_arm: str) -> int:
    """Per-cell RNG/env seed. DELIBERATELY EXCLUDES the bound arm, so the three
    bound arms at one (seed, cap, rail_arm) start from bit-identical RNG state and
    an identical env layout and differ ONLY in the manipulated variables. This is
    what makes the paired deltas and the 1e-6 non-degeneracy guard meaningful --
    red-team findings 2 and 3."""
    key = f"{int(seed)}|{float(cap):.6f}|{rail_arm}".encode("ascii")
    return int(hashlib.blake2b(key, digest_size=4).hexdigest(), 16) % (2 ** 31 - 1)

# 934's cap band, inherited unchanged (see the docstring's transferability caveat).
CAP_SWEEP: List[float] = [0.75, 1.0, 1.25, 1.5, 1.75]
# Training-time cap AND operator: 464e/934's construction, so the trained substrate
# is comparable to the banked reference and all three bound arms share one agent.
AFFINITY_INPUT_CAP_TRAIN = 2.0
AFFINITY_BOUND_MODE_TRAIN = AFFINITY_BOUND_CLAMP

# The mixed band, transcribed from the entry's own failure_record target.
OCCUPANCY_FLOOR = 0.10
OCCUPANCY_CEILING = 0.90
# Gate parameters -- the module defaults, named here so they are pre-registered in
# the manifest rather than implicit.
GATE_MIN_SEED_FRACTION = 2.0 / 3.0
GATE_MIN_ADJACENT = 2
GATE_MIN_SEEDS = 2

# G-margin readiness floor (934's value).
MARGIN_FLOOR = 0.05

# Bit-level identity epsilon for the two "did the manipulation land at all"
# non-degeneracy checks. This is a float-equality tolerance, NOT a contrast
# threshold on the DV -- no threshold is registered for the squash-vs-clamp
# contrast anywhere, by decision.
MANIPULATION_EPS = 1e-6

ARM_LABELS = ["ARM_SYMMETRIC", "ARM_ASYM_STICKY_TASK"]
PRIMARY_ARM = "ARM_SYMMETRIC"

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
    mode-switch signal). TRAINING holds salience_affinity_input_cap at
    AFFINITY_INPUT_CAP_TRAIN (= 2.0, 464e/934's value) under the CLAMP, so the
    trained substrate is bit-comparable to the banked 934 reference; the EVAL cap
    AND operator are swept per cell on clones."""
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
        external_task_drive_affinity_weight=EXTERNAL_TASK_DRIVE_AFFINITY_WEIGHT,
        external_task_drive_salience_weight=2.0,
        external_task_drive_commit_weight=1.0,
        external_task_drive_proximity_weight=1.0,
        salience_affinity_input_cap=AFFINITY_INPUT_CAP_TRAIN,
        salience_affinity_bound_mode=AFFINITY_BOUND_MODE_TRAIN,
        salience_affinity_squash_sigma=AFFINITY_SQUASH_SIGMA,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_dual_cue_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig,
                        seed: Optional[int] = None) -> CausalGridWorldV2:
    """P2-config foraging env WITH the GAP-3 dual_cue primitive (competing goals),
    identical to 934/464e."""
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
    """Clone the SAME trained weights into a fresh agent (rails, eval cap and eval
    bounding operator applied by the caller). INHERITED FROM V3-EXQ-934, NOT from
    464d/467d: GoalState is a plain Python object, invisible to state_dict(), so a
    weights-only clone dropped its z_goal attractor and
    external_task_drive_require_goal_active=True then hard-gated engagement to
    exactly 0.0 for the whole eval."""
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


def _apply_asymmetric_sticky_task(coord) -> None:
    coord.config.enter_thresholds = {}
    coord.config.exit_thresholds = {}
    for mode in MODE_NAMES:
        coord.set_exit_threshold(mode, STICKY_EXIT if mode == STICKY_MODE else LOOSE_EXIT)


def _apply_rails(coord, arm_label: str) -> None:
    if arm_label == "ARM_SYMMETRIC":
        _apply_symmetric(coord)
    elif arm_label == "ARM_ASYM_STICKY_TASK":
        _apply_asymmetric_sticky_task(coord)
    else:
        raise ValueError(f"unknown arm {arm_label}")


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


def _r2_vs_cap(caps: List[float], values: List[float]) -> Optional[float]:
    """R^2 of a least-squares line of `values` on `caps`. Telemetry only -- NO
    threshold is registered against it anywhere. Returns None when undefined
    (< 3 points, or a degenerate/constant axis), so an absent value reads as
    unmeasured rather than as 0.0."""
    if len(caps) < 3 or len(caps) != len(values):
        return None
    x = np.asarray(caps, dtype=float)
    y = np.asarray(values, dtype=float)
    if float(np.ptp(x)) <= 0.0:
        return None
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        # A perfectly constant y is the clamp's degenerate signature, not a fit.
        return None
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_res = float((resid ** 2).sum())
    return float(round(1.0 - ss_res / ss_tot, 6))


def _eval_cell(
    agent: REEAgent,
    env: CausalGridWorldV2,
    cap: float,
    bound_arm: str,
    arm_label: str,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_ep: int,
) -> Dict[str, Any]:
    """Frozen-policy eval for ONE (bound_arm, cap, rail_arm) cell. Rails must already
    be applied by the caller; this sets the EVAL-time cap, bounding operator and
    external_task_drive affinity weight on the coordinator config (all read live at
    tick()). Instruments the discrete occupancy DV, the continuous pre-argmax
    margin, a MODE-CONDITIONED dwell, and the commitment-latch / signal-magnitude
    attribution telemetry."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    coord = agent.salience
    # EVAL-time override -- THIS IS THE SWEEP. All read live at tick().
    coord.config.affinity_input_cap = float(cap)
    coord.config.affinity_bound_mode = BOUND_ARM_OPERATOR[bound_arm]
    coord.config.affinity_squash_sigma = AFFINITY_SQUASH_SIGMA
    # The gain-matched arm's whole point: clamp operator, squash-matched drive gain.
    drive_weight = bound_arm_drive_weight(bound_arm, cap)
    coord.config.affinity_weights["external_task_drive"] = {
        "external_task": float(drive_weight),
    }
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    coord_ticks_start = int(coord.diagnostics.get("n_ticks", 0))

    mode_step_counts = {m: 0 for m in MODE_NAMES}
    other_mode_steps = 0
    total_switches = 0
    total_steps = 0
    ext_margins: List[float] = []
    ext_run_lengths: List[int] = []
    all_run_lengths: List[int] = []
    # Attribution telemetry (recorded, never gated).
    et_drive_values: List[float] = []
    dacc_pe_values: List[float] = []

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

                # Continuous pre-argmax margin (telemetry; no threshold).
                ext_margins.append(float(coord.operating_mode.get(STICKY_MODE, 0.0)))
                # Commitment-latch + signal-magnitude attribution (telemetry).
                # Read AFTER select_action so the values are the ones this tick's
                # arbitration actually consumed.
                et_drive_values.append(
                    float(coord._input_signals.get("external_task_drive", 0.0))
                )
                dacc_pe_values.append(float(coord._input_signals.get("dacc_pe", 0.0)))

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

    n_et = len(et_drive_values)
    # F2 (red-team pass 2): the gain match is derived at engagement e = 1.0. Record
    # what each arm's drive contribution ACTUALLY was on this cell's own engagement
    # trace, so the control's validity is MEASURED rather than assumed.
    realized_contrib: Dict[str, float] = {}
    for _arm in BOUND_ARMS:
        _w = bound_arm_drive_weight(_arm, cap)
        _op = BOUND_ARM_OPERATOR[_arm]
        realized_contrib[_arm] = (
            float(sum(_w * bound_affinity_input(v, float(cap), _op,
                                                AFFINITY_SQUASH_SIGMA)
                      for v in et_drive_values) / n_et)
            if n_et else 0.0
        )
    # Item (2) attribution: engagement pinned at EXACTLY 1.0 is the boolean
    # commitment latch saturating (agent.py:7870/:7881).
    et_saturated = sum(1 for v in et_drive_values if v >= 1.0)
    et_zero = sum(1 for v in et_drive_values if v <= 0.0)
    n_pe = len(dacc_pe_values)
    # Does the bounding operator BITE at all on this signal at this cap?
    pe_over_cap = sum(1 for v in dacc_pe_values if abs(v) > float(cap))

    return {
        "cap": float(cap),
        "bound_arm": str(bound_arm),
        "bound_operator": BOUND_ARM_OPERATOR[bound_arm],
        "drive_affinity_weight": round(float(drive_weight), 6),
        "arm": arm_label,
        "cell_label": f"{bound_arm}|cap={cap}|{arm_label}",
        "fraction_in_external_task": round(frac_task, 4),
        # --- continuous margin telemetry (recorded, NO threshold) ---
        "ext_margin_mean": round(margin_mean, 6),
        "ext_margin_p10": round(_quantile(margins_sorted, 0.10), 6),
        "ext_margin_p50": round(_quantile(margins_sorted, 0.50), 6),
        "ext_margin_p90": round(_quantile(margins_sorted, 0.90), 6),
        "ext_margin_max": round(margins_sorted[-1], 6) if margins_sorted else 0.0,
        # --- commitment-latch attribution telemetry (entry item (2)) ---
        "et_drive_mean": round(float(sum(et_drive_values) / n_et), 6) if n_et else 0.0,
        # realized drive logit contribution under each arm definition, on THIS
        # cell's engagement trace (F2 -- measures whether the gain match held).
        "realized_drive_contrib": {k: round(v, 6) for k, v in realized_contrib.items()},
        "et_drive_saturated_frac": round(et_saturated / n_et, 4) if n_et else 0.0,
        "et_drive_zero_frac": round(et_zero / n_et, 4) if n_et else 0.0,
        # --- does the operator bite on the signal it acts on? ---
        "dacc_pe_mean": round(float(sum(dacc_pe_values) / n_pe), 4) if n_pe else 0.0,
        "dacc_pe_abs_max": round(max((abs(v) for v in dacc_pe_values), default=0.0), 4),
        "dacc_pe_over_cap_frac": round(pe_over_cap / n_pe, 4) if n_pe else 0.0,
        # --- switching / dwell ---
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


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "cells": [],
        "max_margin_mean": 0.0,
        "margin_engaged": False,
        "max_et_drive_mean": 0.0,
        "drive_engaged": False,
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
    caps = CAP_SWEEP[:2] if dry_run else CAP_SWEEP

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
    _ZG.observe(agent)

    # Sweep CAP x RAIL_ARM x BOUND_ARM on clones of the SAME trained agent, with the
    # BOUND ARM INNERMOST and every cell RNG- and ENV-paired (red-team findings 2/3):
    # the per-cell seed excludes the bound arm, so the three bound arms at one
    # (cap, rail_arm) start from bit-identical RNG state and an identical env.
    cells: List[Dict[str, Any]] = []
    for cap in caps:
        for arm_label in ARM_LABELS:
            c_seed = cell_rng_seed(seed, cap, arm_label)
            for bound_arm in BOUND_ARMS:
                # Identical starting RNG state for all three bound arms.
                reset_all_rng(c_seed)
                # Each cell gets its OWN env, seeded identically across bound arms,
                # so no stateful env carries between cells and order cannot confound.
                cell_env = _build_dual_cue_env(scaffold_cfg, seed=c_seed)
                cell_env.reset()
                agent_cell = _clone_for_arm(agent, device)
                _apply_rails(agent_cell.salience, arm_label)
                cell = _eval_cell(
                    agent_cell, cell_env, cap, bound_arm, arm_label,
                    scaffold_cfg, device, eval_eps, steps_per_ep,
                )
                cell["seed"] = int(seed)
                cell["cell_rng_seed"] = int(c_seed)
                cells.append(cell)
                done += eval_eps
                _ZG.observe(agent_cell)
                print(f"  [eval] seed={seed} bound_arm={bound_arm} cap={cap}"
                      f" {arm_label} w_drive={cell['drive_affinity_weight']}"
                      f" occ={cell['fraction_in_external_task']}"
                      f" margin_mean={cell['ext_margin_mean']}"
                      f" et_sat={cell['et_drive_saturated_frac']}"
                      f" pe_over_cap={cell['dacc_pe_over_cap_frac']}"
                      f" switches={cell['n_switches']}", flush=True)

    max_margin_mean = max((float(c["ext_margin_mean"]) for c in cells), default=0.0)
    margin_engaged = bool(max_margin_mean > MARGIN_FLOOR)
    # F4 (red-team pass 2): the margin floor CANNOT detect a dead drive -- with the
    # external_task_drive signal at exactly 0.0 the margin still reads 0.33-0.50
    # (measured), because it is floored by external_task_bias = 1.0, not by the
    # drive. So the drive's OWN engagement is asserted separately, as a zero-test.
    max_et_drive_mean = max((float(c["et_drive_mean"]) for c in cells), default=0.0)
    drive_engaged = bool(max_et_drive_mean > 0.0)

    print(f"  [seed] seed={seed} max_margin={max_margin_mean:.4f}"
          f" margin_engaged={margin_engaged} n_cells={len(cells)}", flush=True)
    print(f"verdict: {'PASS' if (guard_pass and margin_engaged and drive_engaged) else 'FAIL'}"
          f" seed={seed} guard_pass={guard_pass} margin_engaged={margin_engaged}"
          f" drive_engaged={drive_engaged}"
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
        "cells": cells,
        "max_margin_mean": round(max_margin_mean, 6),
        "margin_engaged": margin_engaged,
        "max_et_drive_mean": round(max_et_drive_mean, 6),
        "drive_engaged": drive_engaged,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _gate_for(cells: List[Dict[str, Any]], bound_arm: str,
              arm_label: str) -> Dict[str, Any]:
    """Run the regime-conditioned occupancy gate ONCE over ALL (seed, cap) cells of
    one (bound_arm, rail_arm) slice.

    THIS CALL SHAPE IS THE POINT. Both `seed` and `sweep_value` are populated, so
    the gate can evaluate the entry's actual bar -- >= 2 CONSECUTIVE cap values
    each mixed on >= 2/3 of the seeds measured at that value. V3-EXQ-934 called the
    gate PER SEED with neither field set and counted booleans afterwards, which
    discarded WHICH cap was mixed and produced the false `graded` routing on two
    seeds whose mixed bands were disjoint singletons at opposite ends of the sweep.
    The module FAILS CLOSED to `underdetermined` if the metadata cannot support the
    bar, so this is enforced, not merely intended."""
    slice_cells = [c for c in cells
                   if c["bound_arm"] == bound_arm and c["arm"] == arm_label]
    occ_cells = [
        OccupancyCell(
            label=f"cap={c['cap']}",
            fraction=float(c["fraction_in_external_task"]),
            seed=int(c["seed"]),
            sweep_value=float(c["cap"]),
        )
        for c in slice_cells
    ]
    gate = evaluate_regime_occupancy_gate(
        occ_cells, mode_label=STICKY_MODE,
        floor=OCCUPANCY_FLOOR, ceiling=OCCUPANCY_CEILING,
        min_seed_fraction=GATE_MIN_SEED_FRACTION,
        min_adjacent=GATE_MIN_ADJACENT,
        min_seeds=GATE_MIN_SEEDS,
    )
    gate["bound_arm"] = bound_arm
    gate["arm"] = arm_label
    gate["n_cells"] = len(occ_cells)
    return gate


def run_experiment(dry_run: bool = False,
                   env_seed_base: Optional[int] = None) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}, "
          f"env_seed_base={env_seed_base})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    caps = CAP_SWEEP[:2] if dry_run else CAP_SWEEP
    n_cells = len(BOUND_ARMS) * len(caps) * len(ARM_LABELS)
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

    margin_flags = [bool(r.get("margin_engaged", False)) for r in guard_passing]
    margin_frac = _frac(margin_flags)
    margin_ready_met = bool(margin_frac >= MIN_FRACTION)
    # F4: the DRIVE's own engagement, which the margin floor cannot see.
    drive_flags = [bool(r.get("drive_engaged", False)) for r in guard_passing]
    drive_frac = _frac(drive_flags)
    drive_ready_met = bool(drive_frac >= MIN_FRACTION)

    # All cells from guard-passing seeds -- the gate is called over these.
    all_cells: List[Dict[str, Any]] = [c for r in guard_passing for c in r.get("cells", [])]

    # ONE gate call per (bound_arm, rail_arm) over ALL (seed, cap) cells.
    gates: Dict[str, Dict[str, Any]] = {}
    for bound_arm in BOUND_ARMS:
        for arm_label in ARM_LABELS:
            gates[f"{bound_arm}|{arm_label}"] = _gate_for(all_cells, bound_arm, arm_label)

    squash_primary = gates[f"{BOUND_ARM_SQUASH}|{PRIMARY_ARM}"]
    clamp_primary = gates[f"{BOUND_ARM_CLAMP}|{PRIMARY_ARM}"]
    gain_primary = gates[f"{BOUND_ARM_CLAMP_GAINMATCHED}|{PRIMARY_ARM}"]

    # THE LOAD-BEARING CRITERION (user decision 2026-09-19T22:12:50Z): occupancy,
    # via the transcribed bar, on the SQUASH x ARM_SYMMETRIC slice.
    squash_graded = bool(squash_primary.get("graded", False))
    # Reported, NOT load-bearing: the V3-EXQ-934 clamp baseline reproduction.
    clamp_graded = bool(clamp_primary.get("graded", False))
    # Reported, NOT load-bearing: THE ATTRIBUTION CONTROL. A graded gain-matched
    # clamp arm means the mixed regime is reachable with a DEGENERATE bound once the
    # drive gain is cut -- i.e. the effect is GAIN, not gradedness.
    gain_graded = bool(gain_primary.get("graded", False))

    # --- NON-DEGENERACY (bit-level identity checks, NOT contrast thresholds) ---
    # (1) Did the OPERATOR manipulation land at all? Pair cells by (seed, cap, arm)
    #     and ask whether ANY pair's continuous margin differs. If the two operators
    #     are bit-identical everywhere, a not-graded read says nothing about the
    #     operator and is an instrument concern, not a finding.
    by_key: Dict[Any, Dict[str, float]] = {}
    for c in all_cells:
        key = (c["seed"], c["cap"], c["arm"])
        by_key.setdefault(key, {})[c["bound_arm"]] = float(c["ext_margin_mean"])
    operator_margin_deltas: List[Dict[str, Any]] = []
    for key, vals in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if BOUND_ARM_CLAMP in vals and BOUND_ARM_SQUASH in vals:
            delta = vals[BOUND_ARM_SQUASH] - vals[BOUND_ARM_CLAMP]
            row = {
                "seed": key[0], "cap": key[1], "arm": key[2],
                "clamp_margin_mean": round(vals[BOUND_ARM_CLAMP], 6),
                "squash_margin_mean": round(vals[BOUND_ARM_SQUASH], 6),
                "delta": round(delta, 6),
            }
            if BOUND_ARM_CLAMP_GAINMATCHED in vals:
                gm = vals[BOUND_ARM_CLAMP_GAINMATCHED]
                row["gain_matched_margin_mean"] = round(gm, 6)
                # The two-way decomposition of the total operator delta.
                row["delta_gain_component"] = round(gm - vals[BOUND_ARM_CLAMP], 6)
                row["delta_gradedness_component"] = round(
                    vals[BOUND_ARM_SQUASH] - gm, 6)
            operator_margin_deltas.append(row)
    max_abs_operator_delta = max(
        (abs(d["delta"]) for d in operator_margin_deltas), default=0.0)
    operator_manipulation_landed = bool(max_abs_operator_delta > MANIPULATION_EPS)

    # Same question on the discrete DV -- reported separately so a reader can see
    # which level the operator reached.
    occ_by_key: Dict[Any, Dict[str, float]] = {}
    for c in all_cells:
        occ_by_key.setdefault((c["seed"], c["cap"], c["arm"]), {})[c["bound_arm"]] = \
            float(c["fraction_in_external_task"])
    max_abs_operator_occ_delta = max(
        (abs(v[BOUND_ARM_SQUASH] - v[BOUND_ARM_CLAMP])
         for v in occ_by_key.values()
         if BOUND_ARM_CLAMP in v and BOUND_ARM_SQUASH in v),
        default=0.0)
    operator_moves_occupancy = bool(max_abs_operator_occ_delta > MANIPULATION_EPS)

    # --- THE THREE-ARM OCCUPANCY DECOMPOSITION (telemetry, NO threshold). Only
    #     meaningful because the cells are RNG- and env-paired; see PAIRING.
    def _mean_abs_occ_shift(arm_a: str, arm_b: str) -> float:
        vals = [abs(v[arm_a] - v[arm_b])
                for k, v in occ_by_key.items()
                if k[2] == PRIMARY_ARM and arm_a in v and arm_b in v]
        return round(float(sum(vals) / len(vals)), 6) if vals else 0.0

    occ_shift_gain = _mean_abs_occ_shift(BOUND_ARM_CLAMP_GAINMATCHED, BOUND_ARM_CLAMP)
    occ_shift_gradedness = _mean_abs_occ_shift(BOUND_ARM_SQUASH, BOUND_ARM_CLAMP_GAINMATCHED)
    occ_shift_total = _mean_abs_occ_shift(BOUND_ARM_SQUASH, BOUND_ARM_CLAMP)
    # "Occupancy does not move across ANY arm" -- the degenerate case the
    # pre-registered attribution table routes to the item-(2) latch.
    occupancy_static_across_arms = bool(
        max_abs_operator_occ_delta <= MANIPULATION_EPS
        and occ_shift_gain <= MANIPULATION_EPS
        and occ_shift_gradedness <= MANIPULATION_EPS
    )
    gain_arm_landed = bool(
        max((abs(v[BOUND_ARM_CLAMP_GAINMATCHED] - v[BOUND_ARM_CLAMP])
             for v in by_key.values()
             if BOUND_ARM_CLAMP in v and BOUND_ARM_CLAMP_GAINMATCHED in v),
            default=0.0) > MANIPULATION_EPS
    )

    # (2) Did the CAP manipulation land at all (934's check, per operator)?
    def _varies(key: str, bound_arm: str) -> bool:
        for r in guard_passing:
            for arm_label in ARM_LABELS:
                vals = [float(c[key]) for c in r.get("cells", [])
                        if c["arm"] == arm_label and c["bound_arm"] == bound_arm]
                if len(vals) >= 2 and (max(vals) - min(vals)) > MANIPULATION_EPS:
                    return True
        return False

    occupancy_varies = any(_varies("fraction_in_external_task", m) for m in BOUND_ARMS)
    margin_varies = any(_varies("ext_margin_mean", m) for m in BOUND_ARMS)
    cap_manipulation_landed = bool(occupancy_varies or margin_varies)

    # --- TELEMETRY: margin-vs-cap linearity per (seed, arm, bound_mode). NO
    #     threshold is registered against this anywhere; it is recorded so the
    #     935a clamp-era R^2 0.9996-0.9999 has a like-for-like successor value.
    linearity_rows: List[Dict[str, Any]] = []
    for r in guard_passing:
        for bound_arm in BOUND_ARMS:
            for arm_label in ARM_LABELS:
                sel = sorted(
                    [c for c in r.get("cells", [])
                     if c["bound_arm"] == bound_arm and c["arm"] == arm_label],
                    key=lambda c: c["cap"])
                r2 = _r2_vs_cap([float(c["cap"]) for c in sel],
                                [float(c["ext_margin_mean"]) for c in sel])
                linearity_rows.append({
                    "seed": r["seed"], "bound_arm": bound_arm, "arm": arm_label,
                    "margin_cap_linearity_r2": r2,
                    "n_points": len(sel),
                })

    def _mean_r2(bound_arm: str) -> Optional[float]:
        vals = [row["margin_cap_linearity_r2"] for row in linearity_rows
                if row["bound_arm"] == bound_arm
                and row["arm"] == PRIMARY_ARM
                and row["margin_cap_linearity_r2"] is not None]
        return round(float(sum(vals) / len(vals)), 6) if vals else None

    clamp_r2_primary = _mean_r2(BOUND_ARM_CLAMP)
    squash_r2_primary = _mean_r2(BOUND_ARM_SQUASH)
    gain_r2_primary = _mean_r2(BOUND_ARM_CLAMP_GAINMATCHED)

    # --- TELEMETRY: commitment-latch attribution (entry item (2), still OPEN) ---
    et_mean_vals = [float(c["et_drive_mean"]) for c in all_cells]
    et_drive_mean_overall = (round(float(sum(et_mean_vals) / len(et_mean_vals)), 6)
                             if et_mean_vals else 0.0)
    et_sat_vals = [float(c["et_drive_saturated_frac"]) for c in all_cells]
    et_sat_mean = round(float(sum(et_sat_vals) / len(et_sat_vals)), 4) if et_sat_vals else 0.0
    et_sat_max = round(max(et_sat_vals), 4) if et_sat_vals else 0.0
    pe_over_vals = [float(c["dacc_pe_over_cap_frac"]) for c in all_cells]
    pe_over_mean = round(float(sum(pe_over_vals) / len(pe_over_vals)), 4) if pe_over_vals else 0.0
    pe_abs_max = round(max((float(c["dacc_pe_abs_max"]) for c in all_cells), default=0.0), 4)

    # --- F2: DID THE GAIN-MATCHED CONTROL ACTUALLY REPRODUCE THE GAIN CUT? ---
    # Measured on the clamp_baseline cells' own engagement traces: the residual
    # mismatch between the control and the squash, as a FRACTION of the gain cut the
    # control exists to reproduce. The boundary at 1.0 is DEFINITIONAL, not tuned --
    # a control whose residual error exceeds the very effect it controls for has not
    # controlled for it, so the gradedness attribution is not licensed.
    _ref = [c for c in all_cells
            if c["bound_arm"] == BOUND_ARM_CLAMP and c["arm"] == PRIMARY_ARM]
    _resid, _effect = [], []
    for c in _ref:
        rc = c.get("realized_drive_contrib") or {}
        if not all(k in rc for k in BOUND_ARMS):
            continue
        _resid.append(abs(rc[BOUND_ARM_SQUASH] - rc[BOUND_ARM_CLAMP_GAINMATCHED]))
        _effect.append(abs(rc[BOUND_ARM_CLAMP] - rc[BOUND_ARM_SQUASH]))
    gain_match_residual_mean = (round(float(sum(_resid) / len(_resid)), 6)
                                if _resid else None)
    gain_cut_effect_mean = (round(float(sum(_effect) / len(_effect)), 6)
                            if _effect else None)
    gain_match_residual_fraction = (
        round(gain_match_residual_mean / gain_cut_effect_mean, 6)
        if (gain_match_residual_mean is not None and gain_cut_effect_mean)
        else None
    )
    # Licensed only when the control demonstrably reproduced the gain it controls for.
    gain_match_valid = bool(
        gain_match_residual_fraction is not None
        and gain_match_residual_fraction < 1.0
    )

    # --- ROUTING (the pre-registered three-arm attribution table; see docstring) ---
    attribution = None
    if not contact_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
    elif not margin_ready_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "external_task_drive_not_engaging"
    elif not drive_ready_met:
        # The external_task_drive signal itself is dead on too many seeds -- the
        # 464d goal_state-drop signature. Never read as gradedness evidence.
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "external_task_drive_signal_never_nonzero"
    elif not operator_manipulation_landed:
        # The clamp and squash arms produced bit-identical continuous margins on
        # PAIRED cells. Since the cells share RNG and env by construction, this is a
        # genuine identity test -- and the pre-authoring probe showed the override
        # lands, so this is an instrument concern, never a conclusion.
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "operator_manipulation_inert_verify_instrument"
    elif clamp_graded:
        # Baseline divergence: the clamp arm graded where V3-EXQ-934 recorded no
        # common cap. The three-arm contrast is not interpretable until explained.
        outcome = "FAIL"
        readiness_route = "clamp_baseline_diverges_from_934"
        route_reason = "clamp_baseline_arm_graded_contrast_not_interpretable"
        attribution = "baseline_divergence"
    elif squash_graded and gain_graded:
        # THE CRITICAL ATTRIBUTION BRANCH the gain-matched arm exists for. The mixed
        # regime is ALSO reachable with a DEGENERATE bound once the drive gain is
        # cut, so the ratified criterion passed but it does NOT evidence gradedness.
        outcome = "PASS"
        readiness_route = "mixed_regime_attributable_to_drive_gain_not_gradedness"
        route_reason = "squash_and_gain_matched_clamp_both_graded"
        attribution = "gain"
    elif squash_graded and not gain_match_valid:
        # Squash grades and the control does not -- but the control did NOT actually
        # reproduce the gain cut on the realized engagement trace (F2), so "the
        # control failed to grade" cannot be read as "gradedness is what mattered".
        outcome = "PASS"
        readiness_route = "squash_admits_mixed_regime_attribution_undetermined_control_mismatched"
        route_reason = "gain_match_residual_exceeds_the_gain_cut_it_controls_for"
        attribution = "undetermined_control_mismatched"
    elif squash_graded:
        # Squash grades, the gain-matched clamp does not, AND the control genuinely
        # reproduced the gain -> the graded bound is what buys the mixed regime.
        outcome = "PASS"
        readiness_route = "squash_operator_admits_mixed_regime_attributable_to_gradedness"
        route_reason = "squash_graded_gain_matched_clamp_not_graded_control_valid"
        attribution = "gradedness"
    elif gain_graded:
        # F1 (red-team pass 2): the gain-matched clamp grades while the squash does
        # NOT. The mixed regime is reachable with a DEGENERATE bound at reduced
        # drive gain, so the effect is GAIN. This cell was previously unrouted and
        # fell through to a branch whose route_reason asserted "no_arm_graded" and
        # recorded SD-032a weakens -- factually wrong on both counts.
        outcome = "FAIL"
        readiness_route = "mixed_regime_reached_by_gain_only_squash_did_not_grade"
        route_reason = "gain_matched_clamp_graded_squash_not_graded"
        attribution = "gain"
    elif occupancy_static_across_arms and et_sat_mean > 0.0:
        # Nothing moves the discrete occupancy -- not the cap, not the gain, not the
        # gradedness -- while the continuous margin demonstrably responds. That
        # isolates a cap-, operator- AND gain-INDEPENDENT residual, which is exactly
        # what the substrate_queue entry predicts for item (2)'s boolean latch.
        outcome = "FAIL"
        readiness_route = "residual_discreteness_cap_and_operator_independent_isolates_commitment_latch"
        route_reason = "occupancy_static_across_all_three_arms_margin_responds"
        attribution = "item_2_commitment_latch"
    elif occupancy_static_across_arms:
        # F3 (red-team pass 2): occupancy is static across all three arms, but the
        # boolean commitment latch DEMONSTRABLY NEVER FIRED (et_sat_mean == 0), so
        # the residual discreteness cannot be attributed to it. Something else --
        # e.g. an inactive goal leaving engagement near zero -- produced the static
        # reading. Naming the latch here would be an unsupported attribution.
        outcome = "FAIL"
        readiness_route = "occupancy_static_across_arms_but_commitment_latch_never_fired"
        route_reason = "occupancy_static_with_et_drive_never_saturated_source_unidentified"
        attribution = "static_but_latch_not_implicated"
    else:
        # Occupancy DOES respond to gain and/or gradedness, but never lands a
        # reproducible mixed band. Bang-bang persists for a reason this design has
        # now bounded but not identified.
        outcome = "FAIL"
        readiness_route = "occupancy_responds_but_never_grades_bang_bang_persists"
        route_reason = "no_arm_graded_but_occupancy_shifts_between_arms"
        attribution = "mixed_gain_and_gradedness_insufficient"

    # Per-claim direction follows the ATTRIBUTION, not the bare outcome: a PASS that
    # the gain-matched control shows is reachable without gradedness must NOT be
    # recorded as support for the graded-bound hypothesis.
    if attribution == "gradedness":
        sd032a_dir = "supports"
    elif attribution in ("item_2_commitment_latch", "mixed_gain_and_gradedness_insufficient"):
        sd032a_dir = "weakens"
    elif attribution in ("gain", "undetermined_control_mismatched",
                         "static_but_latch_not_implicated", "baseline_divergence"):
        sd032a_dir = "non_contributory"
    else:
        # "gain", "baseline_divergence", or any not-ready route.
        sd032a_dir = "non_contributory"
    direction_map = {
        "MECH-266": "non_contributory",
        "SD-032a": sd032a_dir,
    }
    overall_direction = "non_contributory"

    squash_band = squash_primary.get("reproducible_band")
    clamp_band = clamp_primary.get("reproducible_band")
    gain_band = gain_primary.get("reproducible_band")

    print(f"[{EXPERIMENT_TYPE}] contact_ready={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) margin_ready={margin_ready_met}"
          f" (frac={margin_frac:.3f})", flush=True)
    print(f"[{EXPERIMENT_TYPE}] SQUASH/{PRIMARY_ARM}"
          f" shape={squash_primary.get('regime_shape')}"
          f" graded={squash_graded} run={squash_primary.get('longest_adjacent_run')}"
          f" band={squash_band}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] CLAMP_BASELINE/{PRIMARY_ARM} (934 baseline)"
          f" shape={clamp_primary.get('regime_shape')}"
          f" graded={clamp_graded} run={clamp_primary.get('longest_adjacent_run')}"
          f" band={clamp_band}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] CLAMP_GAIN_MATCHED/{PRIMARY_ARM} (attribution control)"
          f" shape={gain_primary.get('regime_shape')}"
          f" graded={gain_graded} run={gain_primary.get('longest_adjacent_run')}"
          f" band={gain_band}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] occupancy decomposition"
          f" gain={occ_shift_gain} gradedness={occ_shift_gradedness}"
          f" total={occ_shift_total} static_across_arms={occupancy_static_across_arms}"
          f" gain_arm_landed={gain_arm_landed}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] ATTRIBUTION={attribution}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] operator_landed={operator_manipulation_landed}"
          f" max_abs_margin_delta={max_abs_operator_delta:.6f}"
          f" moves_occupancy={operator_moves_occupancy}"
          f" r2_clamp={clamp_r2_primary} r2_squash={squash_r2_primary}"
          f" r2_gain_matched={gain_r2_primary}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] latch_telemetry et_drive_mean={et_drive_mean_overall}"
          f" et_saturated_frac mean={et_sat_mean} max={et_sat_max}"
          f" dacc_pe_over_cap_frac_mean={pe_over_mean} dacc_pe_abs_max={pe_abs_max}",
          flush=True)
    print(f"[{EXPERIMENT_TYPE}] -> outcome={outcome} route={readiness_route}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "margin_ready_met": margin_ready_met,
        "margin_ready_fraction": margin_frac,
        "drive_ready_met": drive_ready_met,
        "drive_ready_fraction": drive_frac,
        "gain_match_valid": gain_match_valid,
        "gain_match_residual_fraction": gain_match_residual_fraction,
        "squash_symmetric_graded": squash_graded,
        "clamp_symmetric_graded_baseline": clamp_graded,
        "gain_matched_clamp_symmetric_graded": gain_graded,
        "attribution": attribution,
        "squash_reproducible_band": squash_band,
        "clamp_reproducible_band": clamp_band,
        "gain_matched_reproducible_band": gain_band,
        "occ_shift_gain": occ_shift_gain,
        "occ_shift_gradedness": occ_shift_gradedness,
        "occ_shift_total": occ_shift_total,
        "occupancy_static_across_arms": occupancy_static_across_arms,
        "gain_arm_landed": gain_arm_landed,
        "operator_manipulation_landed": operator_manipulation_landed,
        "max_abs_operator_margin_delta": round(max_abs_operator_delta, 6),
        "operator_moves_occupancy": operator_moves_occupancy,
        "max_abs_operator_occupancy_delta": round(max_abs_operator_occ_delta, 6),
        "cap_manipulation_landed": cap_manipulation_landed,
        "occupancy_varies_across_caps": occupancy_varies,
        "margin_varies_across_caps": margin_varies,
        "route_reason": route_reason,
        "per_seed_guard_pass": guard_flags,
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
            "description": "the external_task drive must ENGAGE at SOME swept "
                           "(operator, cap) cell -- per-seed max-over-cells of the "
                           "CONTINUOUS operating_mode['external_task'] margin > "
                           "MARGIN_FLOOR -- on >= 2/3 guard-passing seeds. This is "
                           "the readiness form of the SAME arbitration signal the "
                           "occupancy DV routes on: if the margin is ~0 in every "
                           "cell the drive is not producing the signal (a "
                           "substrate/wiring failure, e.g. a goal_state drop), "
                           "which must self-route substrate_not_ready_requeue and "
                           "NOT be read as gradedness evidence either way.",
            "control": "fraction of guard-passing seeds whose best cell's "
                       "ext_margin_mean clears MARGIN_FLOOR. NOTE its real scope: "
                       "this asserts that ARBITRATION produces an external_task "
                       "signal, NOT that the DRIVE does -- see the next "
                       "precondition, which is the one that can see a dead drive.",
            "measured": round(margin_frac, 4),
            "threshold": MIN_FRACTION,
            "direction": "lower",
            "met": margin_ready_met,
        },
        {
            "name": "external_task_drive_signal_nonzero",
            "kind": "readiness",
            "description": "The external_task_drive SIGNAL must itself be nonzero "
                           "on >= 2/3 guard-passing seeds. This exists because the "
                           "margin precondition above CANNOT detect a dead drive: "
                           "measured on the real coordinator, with the drive signal "
                           "at EXACTLY 0.0 the external_task margin still reads "
                           "0.3287-0.4950 across the swept caps -- 6.6x to 9.9x the "
                           "0.05 floor -- because the margin is floored by "
                           "external_task_bias = 1.0, not by the drive. The margin "
                           "precondition therefore passes through the very failure "
                           "its own description names (a goal_state drop hard-gating "
                           "engagement to 0.0, the V3-EXQ-464d signature). This is "
                           "the same-statistic fix: the drive's own value is what is "
                           "asserted. Inherited defect -- V3-EXQ-934 carries the "
                           "margin precondition alone.",
            "control": "fraction of guard-passing seeds whose best cell has "
                       "et_drive_mean > 0. A zero-test, not a tuned threshold: a "
                       "drive that is never nonzero anywhere cannot be engaging.",
            "measured": round(drive_frac, 4),
            "threshold": MIN_FRACTION,
            "direction": "lower",
            "met": drive_ready_met,
        },
    ]

    criteria = [
        {
            "name": "H_squash_symmetric_arm_graded_regime_reachable",
            "load_bearing": True,
            "passed": squash_graded,
            "description": "SQUASH x ARM_SYMMETRIC: >= 2 CONSECUTIVE swept cap "
                           "values each with occupancy in (0.1, 0.9) on >= 2/3 of "
                           "the seeds measured at that value. Bar transcribed from "
                           "the mode-governance-engagement failure_record target "
                           "for V3-EXQ-934; evaluated by "
                           "experiments/_lib/regime_occupancy_gate.py, not "
                           "re-derived here.",
            "measured": int(squash_primary.get("longest_adjacent_run", 0)),
            "threshold": GATE_MIN_ADJACENT,
            "direction": "lower",
        },
        {
            "name": "gain_matched_clamp_arm_not_graded_attribution_control",
            "load_bearing": False,
            "passed": not gain_graded,
            "description": "REPORTED, NOT LOAD-BEARING -- THE ATTRIBUTION CONTROL "
                           "(user decision 2026-09-19T23:52:40Z, option A). Clamp "
                           "operator with the external_task_drive weight cut to "
                           "match the squash arm's gain at engagement 1.0. If THIS "
                           "arm also grades, the mixed regime is reachable with a "
                           "DEGENERATE bound once the drive gain is cut, so a "
                           "squash PASS is attributable to GAIN, not gradedness -- "
                           "and the run records SD-032a non_contributory rather "
                           "than supports. See the pre-registered attribution "
                           "table in the module docstring.",
            "measured": int(gain_primary.get("longest_adjacent_run", 0)),
            "threshold": GATE_MIN_ADJACENT,
            "direction": "upper",
        },
        {
            "name": "clamp_symmetric_arm_baseline_934_reproduction",
            "load_bearing": False,
            "passed": not clamp_graded,
            "description": "REPORTED, NOT LOAD-BEARING. V3-EXQ-934 recorded no "
                           "common cap on the clamp; a graded clamp arm here would "
                           "be a baseline divergence that makes the contrast "
                           "uninterpretable, and is routed as such.",
            "measured": int(clamp_primary.get("longest_adjacent_run", 0)),
            "threshold": GATE_MIN_ADJACENT,
            "direction": "upper",
        },
    ]
    combination_rule = (
        "OUTCOME: PASS iff the single load-bearing criterion "
        "H_squash_symmetric_arm_graded_regime_reachable passes, AND both readiness "
        "preconditions are met, AND the operator manipulation demonstrably landed "
        "on the continuous margin of PAIRED cells. That criterion is unchanged from "
        "the ratified design. "
        "ATTRIBUTION (separate from the outcome, pre-registered in the module "
        "docstring's three-arm table): the two NON-load-bearing arms decide what a "
        "PASS or a null MEANS, and they drive evidence_direction_per_claim. A PASS "
        "with the gain-matched clamp control ALSO graded is attributed to drive "
        "GAIN, not gradedness, and records SD-032a non_contributory despite the "
        "PASS. A PASS with that control NOT graded is attributed to gradedness and "
        "records supports. A null with occupancy static across all three arms "
        "isolates the item-(2) commitment latch; a null with occupancy shifting but "
        "never grading records bang-bang persistence. A graded clamp BASELINE arm "
        "pre-empts every branch: the contrast is not interpretable, direction "
        "non_contributory."
    )

    crit_non_degenerate = bool(
        contact_non_vacuity_met and margin_ready_met and drive_ready_met
        and operator_manipulation_landed
    )

    readout = flat_readout({
        "H_squash_symmetric_arm_graded_regime_reachable": squash_graded,
        "n_criteria_passed": sum(1 for c in criteria if c["passed"]),
        "n_criteria_total": len(criteria),
        "n_load_bearing_criteria": sum(1 for c in criteria if c.get("load_bearing")),
        "criteria_non_degenerate_flag": crit_non_degenerate,
        # readiness
        "min_fraction": MIN_FRACTION,
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "margin_ready_met": margin_ready_met,
        "margin_ready_fraction": margin_frac,
        "margin_floor": MARGIN_FLOOR,
        "drive_ready_met": drive_ready_met,
        "drive_ready_fraction": drive_frac,
        # did the gain-matched control actually reproduce the gain cut? (F2)
        "gain_match_valid": gain_match_valid,
        "gain_match_residual_fraction": gain_match_residual_fraction,
        "gain_match_residual_mean": gain_match_residual_mean,
        "gain_cut_effect_mean": gain_cut_effect_mean,
        # the load-bearing DV, and the reported baseline
        "squash_symmetric_graded": squash_graded,
        "squash_longest_adjacent_run": int(squash_primary.get("longest_adjacent_run", 0)),
        "clamp_symmetric_graded": clamp_graded,
        "clamp_longest_adjacent_run": int(clamp_primary.get("longest_adjacent_run", 0)),
        # the attribution control arm (option A)
        "gain_matched_clamp_symmetric_graded": gain_graded,
        "gain_matched_longest_adjacent_run": int(gain_primary.get("longest_adjacent_run", 0)),
        "attribution_is_gradedness": bool(attribution == "gradedness"),
        "attribution_is_gain": bool(attribution == "gain"),
        "attribution_is_item2_latch": bool(attribution == "item_2_commitment_latch"),
        "attribution_is_undetermined_control_mismatched":
            bool(attribution == "undetermined_control_mismatched"),
        # three-arm occupancy decomposition (telemetry, no threshold)
        "occ_shift_gain": occ_shift_gain,
        "occ_shift_gradedness": occ_shift_gradedness,
        "occ_shift_total": occ_shift_total,
        "occupancy_static_across_arms": occupancy_static_across_arms,
        "gain_arm_landed": gain_arm_landed,
        "gate_min_adjacent": GATE_MIN_ADJACENT,
        "gate_min_seed_fraction": GATE_MIN_SEED_FRACTION,
        "gate_min_seeds": GATE_MIN_SEEDS,
        "occupancy_floor": OCCUPANCY_FLOOR,
        "occupancy_ceiling": OCCUPANCY_CEILING,
        # non-degeneracy (bit-level identity, not contrast thresholds)
        "operator_manipulation_landed": operator_manipulation_landed,
        "max_abs_operator_margin_delta": round(max_abs_operator_delta, 6),
        "operator_moves_occupancy": operator_moves_occupancy,
        "max_abs_operator_occupancy_delta": round(max_abs_operator_occ_delta, 6),
        "cap_manipulation_landed": cap_manipulation_landed,
        "occupancy_varies_across_caps": occupancy_varies,
        "margin_varies_across_caps": margin_varies,
        # continuous-margin telemetry -- RECORDED, NO THRESHOLD
        "margin_cap_linearity_r2_clamp_symmetric": clamp_r2_primary,
        "margin_cap_linearity_r2_squash_symmetric": squash_r2_primary,
        "margin_cap_linearity_r2_gain_matched_symmetric": gain_r2_primary,
        # commitment-latch attribution telemetry (entry item (2), OPEN)
        "et_drive_saturated_frac_mean": et_sat_mean,
        "et_drive_saturated_frac_max": et_sat_max,
        # Mean engagement -- HOW GOOD THE GAIN MATCH ACTUALLY WAS. The match is
        # exact at e = 1.0; the further this sits below 1.0, the more the
        # gain-matched clamp arm UNDER-delivers relative to the squash, and the
        # more conservative the gradedness component reads.
        "et_drive_mean": et_drive_mean_overall,
        "dacc_pe_over_cap_frac_mean": pe_over_mean,
        "dacc_pe_abs_max": pe_abs_max,
        # sweep shape
        "n_seeds": n,
        "n_caps_swept": len(CAP_SWEEP),
        "cap_sweep_min": min(CAP_SWEEP),
        "cap_sweep_max": max(CAP_SWEEP),
        "n_bound_arms": len(BOUND_ARMS),
        "n_preconditions_met": sum(1 for pc in preconditions if pc["met"]),
        "n_preconditions_total": len(preconditions),
    })

    return {
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": direction_map,
        "readout": readout,
        "acceptance": acceptance,
        "occupancy_gates": gates,
        "attribution": attribution,
        "occupancy_decomposition": {
            "note": "Mean |occupancy difference| over the PRIMARY rail arm's "
                    "(seed, cap) cells. Meaningful ONLY because cells are RNG- and "
                    "env-paired across bound arms. TELEMETRY -- no threshold.",
            "occ_shift_gain": occ_shift_gain,
            "occ_shift_gradedness": occ_shift_gradedness,
            "occ_shift_total": occ_shift_total,
            "occupancy_static_across_arms": occupancy_static_across_arms,
            "gain_arm_landed": gain_arm_landed,
        },
        "operator_margin_deltas": operator_margin_deltas,
        "margin_cap_linearity": linearity_rows,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "route_reason": route_reason,
            "hypothesis": "A bounded-but-GRADED affinity input (the saturating "
                          "squash, sigma = cap) admits the mixed external_task "
                          "occupancy regime that the bang-bang box clamp could not, "
                          "at some cap in the swept band.",
            "null": "No >= 2 CONSECUTIVE cap values yield ARM_SYMMETRIC occupancy "
                    "in (0.1, 0.9) on >= 2/3 seeds under the squash -- i.e. the "
                    "graded operator alone does not make the mode register graded.",
            "null_meaning": "A null here is INFORMATIVE and pre-registered as such. "
                            "The mode-governance-engagement entry predicts it in "
                            "writing: item (2)'s _et_commit = commit_w * "
                            "float(beta_gate.is_elevated) is a boolean latch and a "
                            "cap- AND operator-INDEPENDENT discreteness source, so "
                            "'a graded bounding operator alone does NOT make the "
                            "register graded end-to-end'. With the operator "
                            "manipulation demonstrably landing on the continuous "
                            "margin (operator_manipulation_landed), a null "
                            "ISOLATES that residual source rather than leaving the "
                            "operator under suspicion. It does NOT license any "
                            "claim promotion or demotion.",
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": {
                "H_squash_symmetric_arm_graded_regime_reachable": crit_non_degenerate,
                "clamp_symmetric_arm_baseline_934_reproduction": crit_non_degenerate,
            },
            "occupancy_gate": {
                "definition": "ONE evaluate_regime_occupancy_gate call per "
                              "(bound_arm, rail_arm) over ALL (seed, cap) cells, with "
                              "seed AND sweep_value populated. GRADED iff >= "
                              "min_adjacent (2) CONSECUTIVE swept cap values each "
                              "read mixed on >= min_seed_fraction (2/3) of the "
                              "seeds measured at that value. The bar is TRANSCRIBED "
                              "from mode-governance-engagement's own failure_record "
                              "target for V3-EXQ-934 and implemented in "
                              "experiments/_lib/regime_occupancy_gate.py -- never "
                              "re-derived, and never a min-across-the-sweep "
                              "statistic. V3-EXQ-934's per-seed call shape (no "
                              "seed, no sweep_value, booleans counted afterwards) "
                              "is deliberately NOT inherited: it is what produced "
                              "that run's false 'graded' routing.",
                "load_bearing_slice": f"{BOUND_ARM_SQUASH}|{PRIMARY_ARM}",
                "baseline_slice": f"{BOUND_ARM_CLAMP}|{PRIMARY_ARM}",
                "attribution_control_slice": f"{BOUND_ARM_CLAMP_GAINMATCHED}|{PRIMARY_ARM}",
                "occupancy_floor": OCCUPANCY_FLOOR,
                "occupancy_ceiling": OCCUPANCY_CEILING,
                "min_seed_fraction": GATE_MIN_SEED_FRACTION,
                "min_adjacent": GATE_MIN_ADJACENT,
                "min_seeds": GATE_MIN_SEEDS,
                "margin_floor": MARGIN_FLOOR,
                "cap_sweep": CAP_SWEEP,
                "bound_arms": BOUND_ARMS,
                "primary_arm": PRIMARY_ARM,
            },
            "three_arm_attribution": {
                "status": "PRE-REGISTERED (user decision 2026-09-19T23:52:40Z, "
                          "option A). Fixed BEFORE the run; the outcome follows the "
                          "ratified occupancy criterion, the ATTRIBUTION follows "
                          "this table and drives evidence_direction_per_claim.",
                "resolved_verdict": attribution,
                "table": [
                    "squash graded + gain_matched NOT graded -> GRADEDNESS "
                    "(SD-032a supports)",
                    "squash graded + gain_matched ALSO graded -> GAIN, not "
                    "gradedness; PASS but SD-032a non_contributory",
                    "neither graded + occupancy static across all three arms -> "
                    "cap/operator/gain-INDEPENDENT residual, isolates the item-(2) "
                    "commitment latch (SD-032a weakens)",
                    "neither graded + occupancy shifts between arms -> bang-bang "
                    "persists (SD-032a weakens)",
                    "clamp BASELINE graded -> divergence from V3-EXQ-934, contrast "
                    "not interpretable (SD-032a non_contributory)",
                ],
                "gain_matched_drive_weight_rule":
                    "w'(cap) = 3.0 * (cap/(cap+1)) / min(1.0, cap); matches the "
                    "squash arm's external_task_drive logit contribution at "
                    "engagement e = 1.0. Equals the decision's literal 3*cap/(cap+1) "
                    "at every cap >= 1.0; the min() term corrects only cap = 0.75, "
                    "where the clamp itself attenuates e = 1.0 to 0.75.",
                "gain_matched_drive_weight_by_cap": {
                    str(c): round(gain_matched_drive_weight(c), 6) for c in CAP_SWEEP
                },
                "literal_decision_formula_by_cap": {
                    str(c): round(EXTERNAL_TASK_DRIVE_AFFINITY_WEIGHT * c / (c + 1.0), 6)
                    for c in CAP_SWEEP
                },
                "residual": "The gain match is exact only at engagement e = 1.0. "
                            "Where engagement is UNLATCHED (e < 1) the squash and "
                            "the gain-matched clamp diverge slightly; "
                            "et_drive_saturated_frac records how dominant e = 1.0 "
                            "actually was.",
            },
            "cell_pairing": {
                "status": "Cells are RNG- AND ENV-PAIRED across bound arms (red-team "
                          "findings 2 and 3, fixed under option A).",
                "rule": "The per-cell seed is derived from (seed, cap, rail_arm) "
                        "ONLY -- never the bound arm -- and drives both "
                        "reset_all_rng() at cell entry and the cell's OWN freshly "
                        "built env. So the three bound arms at one (seed, cap, "
                        "rail_arm) start from bit-identical RNG state and an "
                        "identical env layout.",
                "why_it_matters": "Without it the 1e-6 operator_manipulation_landed "
                                  "guard is vacuous -- banked V3-EXQ-934 cells "
                                  "differing only in rails already differed by "
                                  "1.7e-3 -- and the occupancy decomposition would "
                                  "measure run-to-run noise. It also removes the "
                                  "bound-arm-order confound: V3-EXQ-934 shared one "
                                  "never-rebuilt stateful env across all cells.",
                "departs_from_934": "934 used an OS-entropy eval env shared across "
                                    "its cells. The clamp baseline arm here "
                                    "reproduces 934's DESIGN (same operator, "
                                    "weight, cap band, seeds, rails) but is NOT "
                                    "expected to reproduce its cell values "
                                    "bit-for-bit, and nothing compares against them "
                                    "numerically.",
            },
            "continuous_margin_telemetry": {
                "status": "RECORDED WITH NO THRESHOLD -- deliberate (user decision "
                          "2026-09-19T22:12:50Z). Occupancy is the load-bearing "
                          "criterion; the continuous margin is recorded so that a "
                          "null is ATTRIBUTABLE to the cap-independent commitment "
                          "latch rather than to the operator. No numeric threshold "
                          "for the squash-vs-clamp contrast is registered anywhere, "
                          "and none is invented here.",
                "clamp_era_reference": "V3-EXQ-935a recorded ext_margin_mean LINEAR "
                                       "in cap at R^2 0.9996-0.9999 under the clamp "
                                       "(the at-cap degeneracy signature).",
                "margin_cap_linearity_r2_clamp_symmetric": clamp_r2_primary,
                "margin_cap_linearity_r2_squash_symmetric": squash_r2_primary,
                "margin_cap_linearity_r2_gain_matched_symmetric": gain_r2_primary,
            },
            "commitment_latch_attribution": {
                "status": "RECORDED WITH NO THRESHOLD. mode-governance-engagement "
                          "item (2) is OPEN and untouched by this run.",
                "mechanism": "ree_core/agent.py:7870 _et_commit = _et_commit_w * "
                             "(1.0 if self.beta_gate.is_elevated else 0.0); :7881 "
                             "_et_engagement = max(0.0, min(1.0, _et_commit + "
                             "_et_prox)). At the default "
                             "external_task_drive_commit_weight = 1.0 ANY beta "
                             "elevation saturates engagement to exactly 1.0 "
                             "regardless of goal proximity.",
                "et_drive_saturated_frac_mean": et_sat_mean,
                "et_drive_saturated_frac_max": et_sat_max,
                "et_drive_mean": et_drive_mean_overall,
                "gain_match_quality_note":
                    "The gain match is EXACT at engagement e = 1.0 and degrades "
                    "below it (at cap 1.0 the squash delivers 3e/(1+e) while the "
                    "gain-matched clamp delivers 1.5e, so at e = 0.5 they are 1.00 "
                    "vs 0.75). CORRECTION, from the second red-team pass: an "
                    "earlier version of this note called that residual "
                    "'conservative'. It is not. For every cap >= 1.0 at e < 1 the "
                    "control carries LESS external_task gain than the squash, so "
                    "'the control did not grade' can be caused by the control "
                    "OVER-cutting rather than by gradedness mattering -- a "
                    "false-positive path to the supports branch, which is the "
                    "opposite of conservative. That is why the gradedness "
                    "attribution is now gated on gain_match_valid: the residual "
                    "mismatch, measured on the realized engagement trace, must be "
                    "smaller than the gain cut the control exists to reproduce. "
                    "The 1.0 boundary is definitional, not tuned.",
                "operator_bite_note": "external_task_drive is already bounded to "
                                      "[0,1], so at any cap >= 1.0 BOTH operators "
                                      "are no-ops on it (V3-EXQ-934 failure_record). "
                                      "The operator bites on dacc_pe; "
                                      "dacc_pe_over_cap_frac records how often.",
                "dacc_pe_over_cap_frac_mean": pe_over_mean,
                "dacc_pe_abs_max": pe_abs_max,
            },
            "contact_guard": {
                "definition": "per-seed P2 contact_rate > 0 AND "
                              "z_goal_norm_at_contact_peak > 0.4; < 2/3 seeds -> "
                              "substrate_not_ready_requeue.",
                "min_fraction": MIN_FRACTION,
                "p2_zgoal_gate": P2_ZGOAL_GATE,
                "contact_gate": CONTACT_GATE,
            },
            "scope": "Validates mode-governance-engagement ITEM (1) (the bounding "
                     "operator) ONLY. Items (2) (commitment-term grading) and (3) "
                     "(production default) remain OPEN and are untouched: "
                     "salience_affinity_input_cap and use_external_task_drive are "
                     "set explicitly by this driver and the PRODUCTION defaults are "
                     "NOT flipped. PROMOTES NOTHING -- MECH-266 / SD-032a "
                     "pending_retest_after_substrate holds, and the Stage-H "
                     "nav-competence dependency gating v3_exq_464d / v3_exq_467d is "
                     "untouched.",
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
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
        "bound_arms": BOUND_ARMS,
        "bound_arm_operator": BOUND_ARM_OPERATOR,
        "external_task_drive_affinity_weight": EXTERNAL_TASK_DRIVE_AFFINITY_WEIGHT,
        "gain_matched_drive_weight_by_cap": {
            str(c): round(gain_matched_drive_weight(c), 6) for c in CAP_SWEEP
        },
        "affinity_squash_sigma": AFFINITY_SQUASH_SIGMA,
        "affinity_squash_sigma_rule": "None MEANS sigma = affinity_input_cap (landed "
                                      "default, user decision 2026-09-19T09:45Z). "
                                      "Slope at the origin is then exactly 1, so "
                                      "every SUB-cap signal passes through as the "
                                      "legacy clamp passed it and the only "
                                      "behavioural change is at the top end.",
        "cap_sweep": CAP_SWEEP,
        "affinity_input_cap_train": AFFINITY_INPUT_CAP_TRAIN,
        "affinity_bound_mode_train": AFFINITY_BOUND_MODE_TRAIN,
        "occupancy_floor": OCCUPANCY_FLOOR,
        "occupancy_ceiling": OCCUPANCY_CEILING,
        "gate_min_seed_fraction": GATE_MIN_SEED_FRACTION,
        "gate_min_adjacent": GATE_MIN_ADJACENT,
        "gate_min_seeds": GATE_MIN_SEEDS,
        "margin_floor": MARGIN_FLOOR,
        "manipulation_eps": MANIPULATION_EPS,
        "arms": ARM_LABELS,
        "primary_arm": PRIMARY_ARM,
        "sticky_exit": STICKY_EXIT,
        "loose_exit": LOOSE_EXIT,
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
                     "salience_affinity_input_cap (trained at 2.0 under the CLAMP; "
                     "EVAL cap swept in [0.75,1.0,1.25,1.5,1.75] CROSSED with THREE "
                     "bound arms -- clamp_baseline, squash (sigma = cap), and "
                     "clamp_gain_matched (clamp operator with the "
                     "external_task_drive affinity weight cut to match the squash "
                     "arm's gain at engagement 1.0) -- on RNG- and env-paired "
                     "clones). use_closure_operator OFF.",
        "condition": CONDITION_LABEL,
        "predecessor": PREDECESSOR,
        "validates": "mode-governance-engagement ITEM (1) (bounding operator) ONLY; "
                     "items (2) commitment-term grading and (3) production default "
                     "remain OPEN.",
        "method_note": "Crosses the affinity BOUNDING OPERATOR (clamp vs squash) "
                       "with the cap at EVAL time on clones of a single trained "
                       "curriculum agent per seed -- cap, mode and sigma are all "
                       "read live at SalienceCoordinator.tick(), so no retraining "
                       "is needed (the same train-once/sweep-on-clones pattern 934 "
                       "and 467e use; confirmed by a pre-authoring one-tick probe "
                       "against the real coordinator). Training is held at cap 2.0 "
                       "under the CLAMP (464e/934 construction) so the clamp arm "
                       "reproduces the V3-EXQ-934 baseline design and all three "
                       "bound arms share one trained agent. Cells are RNG- and "
                       "env-paired across bound arms (per-cell seed excludes the "
                       "bound arm), with the bound arm nested innermost. Non-vacuity via "
                       "experiments/_lib/regime_occupancy_gate.py, called ONCE per "
                       "(bound_arm, rail_arm) over ALL (seed, cap) cells with seed AND "
                       "sweep_value populated -- never per-seed with booleans "
                       "counted afterwards (the V3-EXQ-934 shape that produced its "
                       "false routing), and never min-across-the-sweep.",
        "pre_registered_thresholds": {
            "cap_sweep": CAP_SWEEP,
            "bound_arms": BOUND_ARMS,
            "affinity_squash_sigma": AFFINITY_SQUASH_SIGMA,
            "affinity_input_cap_train": AFFINITY_INPUT_CAP_TRAIN,
            "occupancy_floor": OCCUPANCY_FLOOR,
            "occupancy_ceiling": OCCUPANCY_CEILING,
            "gate_min_seed_fraction": GATE_MIN_SEED_FRACTION,
            "gate_min_adjacent": GATE_MIN_ADJACENT,
            "gate_min_seeds": GATE_MIN_SEEDS,
            "margin_floor": MARGIN_FLOOR,
            "min_fraction": MIN_FRACTION,
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "sticky_exit": STICKY_EXIT,
            "loose_exit": LOOSE_EXIT,
            "manipulation_eps": MANIPULATION_EPS,
            "continuous_margin_threshold": None,
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
             "OS-entropy env seeding. Set it and every env this run builds is "
             "deterministically seeded. A pinned run is NOT comparable to a landed "
             "one.",
    )
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run, env_seed_base=args.env_seed)
    _outcome_raw = str(_res["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res.get("manifest_path"),
        dry_run=bool(args.dry_run),
    )
