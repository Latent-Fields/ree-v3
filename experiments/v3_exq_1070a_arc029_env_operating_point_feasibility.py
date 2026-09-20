#!/opt/local/bin/python3
"""
V3-EXQ-1070a -- ARC-029: is there ANY env operating point where P1 and P3 are
JOINTLY satisfiable on the variance-tracking commit bar? DIAGNOSTIC ONLY.

SUPERSEDES V3-EXQ-1070 (ran 2026-09-20T04:26Z, outcome FAIL, self-routed
`substrate_not_ready_requeue`). SAME scientific question, SAME pre-registered
criteria and thresholds, ONE instrument repair. Nothing else is changed.

WHAT 1070 ESTABLISHED -- keep this, it is not re-derived here
---------------------------------------------------------------------------
Three of its four readiness preconditions PASSED, and two of them settle questions
that were open when it was queued:

  * `variance_tracking_bar_in_force_somewhere` measured **1.000** and
    `commit_gate_window_filled_in_some_cell` measured **30 of 30 cells**. The bar
    engages at every rung. **So the predecessor's `bar_in_force = 0.000` /
    occupancy 1.0000 reading was a WARM-BUDGET artifact, not an env property** --
    exactly what this driver's own docstring predicted under "F5" and said would
    be a finding if it happened. It happened.
  * `training_collapsed_running_variance` measured **1.27 decades** at its worst
    cell against a 0.5 floor, so every cell ran on a genuinely trained agent.
  * The F1 repair mattered more than expected: mean committed-run length is
    **4.4-8.1 per episode** against **17.8-31.3** on the flat cross-episode
    sequence -- a 3-4x overstatement. Without that fix P1 would have passed
    everywhere trivially and the whole run would have been vacuous.
  * The health-budget conservation tightened on a trained agent:
    `episode_length x |mean harm/step|` = **1.159 / 1.159 / 1.209 / 1.259 / 1.203**
    across the five rungs (random policy gave 1.02-1.42).
  * The owed quantity was delivered: rv residual dispersion rises monotonically
    **0.0112 -> 0.0206 -> 0.0291 -> 0.0376 -> 0.0672** from L0 to L5.
  * And a caveat the eventual user decision needs: `window_span_episodes` is
    **15-39** at every rung, so the bar is a CROSS-EPISODE quantile everywhere, not
    the within-run one the estimator's drift premise describes. That is reported,
    not gated, and it is a property of the substrate rather than of this run.

NOT CARRIED FORWARD, DELIBERATELY. 1070's cells did populate a would-be feasibility
table, and it is NOT reproduced here as a result or an expectation. Its gate failed,
so by this design's own rule those numbers are not licensed -- and they were produced
under exactly the unequal training budgets this iteration exists to remove, which is
the one thing most likely to move which rung looks feasible. 1070a re-measures.

## THE ONE DEFECT, AND THE RULE THAT REPLACES IT

`training_tick_budget_equalised` measured **3.96** against its 2.0 ceiling, so the
run correctly refused to report a feasibility region.

ROOT CAUSE, measured. 1070 derived each rung's EPISODE count from `LADDER_EP_LEN`,
a table of episode lengths pre-measured under a RANDOM policy. Those lengths do not
transfer to training-time behaviour, and the error is systematic in the direction
that matters: realised/predicted episode length was 0.82-1.22 at L0/L1 but
**0.35-0.70 at L2, L3 and L5**. The soft rungs were therefore trained on far fewer
ticks than intended (L1 1736 vs L3 614 mean total ticks), which is precisely the
confound-along-the-manipulated-dimension that precondition exists to catch.

TWO CHEAPER FIXES WERE CHECKED ON 1070's OWN NUMBERS AND BOTH FAIL -- recorded so
they are not re-proposed:
  (a) *Re-derive the table from 1070's realised training-time lengths.* Applying
      that correction to 1070's own per-seed data gives a spread of **~2.0** --
      exactly at the ceiling, with no margin -- because it cannot touch the second
      error source: realised ticks vary up to **2x BETWEEN SEEDS at one rung**
      (L5: 461 vs 876). A feed-forward prediction is set before the seed is run.
  (b) *Gate on the per-rung MEAN instead of the per-cell spread,* on the argument
      that only between-rung variation confounds the manipulation. 1070's per-rung
      means are 1294 / 1736 / 685 / 614 / 668, a spread of **2.83** -- still over
      the ceiling. It also relaxes the precondition, which the repair brief forbids.

**THE RULE THIS RUN USES INSTEAD -- closed-loop, per (rung, seed):**

  For each (rung, seed) a PILOT runs the IDENTICAL trainer on a throwaway agent
  constructed from the same seed, for `PILOT_EPISODES` episodes of each phase, and
  measures the realised ticks-per-episode of that (rung, seed) under real training
  dynamics. The real agent is then built fresh and trained in exactly ONE
  `_train_all_on_agent` call with
      `p0_episodes = ceil(TARGET_P0_TICKS / pilot_p0_ticks_per_episode)`
      `p1_episodes = ceil(TARGET_P1_TICKS / pilot_p1_ticks_per_episode)`
  This is closed-loop over BOTH error sources -- the rung AND the seed -- which
  neither (a) nor (b) is.

  WHY ONE CALL AND NOT A CHUNKED TICK CAP. The obvious implementation of "train to
  a fixed tick budget" is to call the trainer repeatedly in small chunks until the
  tick target is met. That is WRONG here: `_train_all_on_agent` CONSTRUCTS its
  optimizers internally (`experiments/_lib/allon_training.py`, `e2_opt`,
  `bias_opt`, `ofc_deval_opt` are `torch.optim.Adam(...)` at the top of the
  function), so every chunk boundary would reset Adam's moment estimates -- and
  reset them MORE TIMES at the short-episode rungs, re-introducing a
  training-dynamics difference along the manipulated dimension in the act of
  removing a tick-count one. The pilot keeps a single optimizer lifetime.

  THE ZWORLD P0a PHASE IS BUDGETED FROM THE RANDOM-POLICY TABLE, DELIBERATELY:
  `run_zworld_p0` rolls out under `RandomPolicy`, so `LADDER_EP_LEN` is the CORRECT
  predictor for that phase and the wrong one for the others. That asymmetry was the
  bug in 1070 and is now explicit.

  WHAT THIS RULE DOES *NOT* FIX, STATED SO IT IS NOT MIS-READ (red-team F2/F4):
    * The pilot is an ESTIMATE, and the "1.029" figure obtainable by applying this
      rule to 1070's realised whole-phase lengths is arithmetic about `ceil`, NOT a
      prediction of the pilot's error -- it assumes the pilot is exact. The two-stage
      design above removes the largest known rung-correlated component of that error;
      whatever remains is bounded by the 2.0 ceiling, and
      `pilot_p1_ticks_per_episode` is recorded beside the realised
      `n_p1_ticks / episodes_p1` so the residual can be checked for rung-correlation
      post hoc rather than assumed absent.
    * EQUAL TICKS IS NOT EQUAL TRAINING ON EVERY AXIS, and cannot be. e2 contrastive
      steps and the `running_variance` EMA are per TICK, so those ARE equalised. The
      lateral-PFC and OFC REINFORCE heads step once per P1 EPISODE
      (`allon_training.py:815, :825`), so equalising ticks makes their update count
      vary INVERSELY with episode length -- on 1070's realised lengths, ~178 updates
      at L0 against ~33 at L5. This is structural: ticks = episodes x length, so with
      length varying by rung no budget can equalise both. It is NOT introduced by this
      repair (1070 had a 6.3x spread on the same axis) and it is not silently fixed
      here. It IS a live caveat for the deliverable: those heads feed E3 scoring, so a
      rung difference in P1 feasibility could in principle be mediated by policy-head
      dose rather than by the environment. Read the deliverable as "is there a
      region", not as "which rung is best". `episodes_p1` is recorded per cell.

  IF A RUNG CANNOT BE BUDGETED -- `ceil(target / pilot_len)` exceeding
  `MAX_TRAIN_EPISODES_PER_PHASE` -- the rung is DROPPED from the feasibility
  analysis and recorded as `budget_infeasible`, rather than silently truncated to
  the cap (which is what would quietly re-create the defect). The cap is set well
  above the worst case 1070 measured (L0 needs ~177 P1 episodes at 5.1 ticks each),
  so it is a guard and not an expected path. **A drop is also GATED**
  (`all_rungs_tick_budgetable`), and that is load-bearing rather than belt-and-braces:
  the budget is `ceil(target / length)`, so ONLY a SHORT-episode rung can overrun the
  cap -- which is the ladder's MINIMUM and, at L0, the lineage control. On 1070's own
  measured rung means, dropping L0 moves D1's episode-length range from 3.32 to
  **1.94**, i.e. below D1's own 2.0 floor. Ungated, a budget failure would therefore
  have presented as "the ladder did not sweep" while the interpretation label still
  reported a feasibility region. Gating routes it to `substrate_not_ready_requeue`,
  which is what it actually is.

Claims: ARC-029 (tagged for traceability; `diagnostic` runs are excluded from
        governance confidence/conflict scoring and this run returns NO claim
        verdict -- see DISPOSITION below, which the run RECORDS and does not
        APPLY).
Authority: user decision 2026-09-19T23:52:40Z (real AskUserQuestion via
        orchestrate-20260919-2125), OPTION C on decision chip
        `chip-20260919-arc029-p1p3-env-operating-point`: "ONE CHEAP DIAGNOSTIC
        FIRST, no evidence run."
Predecessor: the pre-registered ARC-029 evidence design (EXP-1394 /
        "V3-EXQ-063b") was REFUSED at /queue-experiment Step 4 and NOT queued.
        Its driver is `experiments/v3_exq_1066_arc029_commitment_mode_harm_variance_bar.py`
        (ree-v3 9bbe1b0dab, DO-NOT-QUEUE banner). Measured record:
        REE_assembly/evidence/planning/arc029_exq1066_prereg_derivation_20260919.md
        (addendum, c1f576c709). GFLAG-0371.
red-team: see the queue entry note for the verdict and model.

WHY THIS RUN EXISTS -- the bind, in closed form
---------------------------------------------------------------------------
ARC-029's non-degeneracy precondition has two halves that pull against each
other through ONE env quantity:

  P1  within a single run, `committed_step_fraction` in [0.15, 0.85] AND mean
      committed-run length >= 3. Needs LONG episodes.
  P3  the harm DV must sit an order of magnitude above the measurement floor.
      Needs a LARGE per-step harm signal.

`CausalGridWorldV2` drains `agent_health` by `abs(harm_signal)` per harm step
(`causal_grid_world.py:2642`) from a starting value HARD-CODED to 1.0
(`:1774, :2083, :2216`) -- there is NO starting-health constructor parameter, so
"lengthen episodes without shrinking the DV" is not reachable with existing
levers. Episodes end on `done_cause: health_depleted`, so

      episode_length  x  |mean harm per step|  ~  the health budget  ~  1.0

MEASURED 2026-09-19 over a 6x range of episode length (random policy, 30
episodes per rung): the product is 1.02 / 1.10 / 1.09 / 1.25 / 1.27 / 1.42. It
is a conserved quantity, not a tendency. **So P1 and P3 trade off EXACTLY, and
the only open question is whether the hyperbola passes through a region where
both are met.** That question has never been measured, and answering it is this
run's entire job.

It is also the measurement
`arc029_variance_tracking_commit_bar_build_20260918.md` sec 6a already says is
owed -- "the rv residual dispersion on a REAL trained agent, on which the whole
alternating-arm design's viability rests" -- which is emitted here per cell.

WHAT THE PREDECESSOR MEASURED, AND WHY IT IS NOT A REPEAT
---------------------------------------------------------------------------
On a trained agent at the ARC-029 lineage env (V3-EXQ-125a's kwargs), episodes
run 4-9 env steps = 4.9-7.3 SELECT CALLS, the 200-select-call commit-gate window
never filled, `bar_in_force` was 0.000 in every cell and
`committed_step_fraction` was 1.0000 with ONE committed run -- V3-EXQ-063a's
saturation defect reproduced WITH the new lever armed. That probe capped warm-up
at 12 episodes, which is why the window never filled; here the warm-up runs to
`WARM_EPISODE_CAP = 200` episodes and STOPS WHEN THE WINDOW IS FULL, so "the bar
cannot engage" and "the bar was not given enough episodes" are separated rather
than conflated. `warm_episodes_used` and `window_span_episodes` are recorded
precisely so a reader can tell them apart.

THE LADDER (one dimension, because the bind is one-dimensional)
---------------------------------------------------------------------------
A cross-product over four env knobs would be ~16x the cells for no extra
information: they all act on the SAME conserved product above. So the grid is a
LETHALITY LADDER, pre-measured (random policy, 30 eps/rung) to span the range
rather than cluster:

  rung  num_hazards  hazard_harm  proximity_harm_scale   measured ep_len  reward/step
  L0        5           0.05           0.15                   6.2          -0.177
  L1        4           0.04           0.10                  10.8          -0.095
  L2        3           0.03           0.06                  22.5          -0.048
  L3        3           0.02           0.035                 29.5          -0.042
  L5        1           0.005          0.005                  (extreme rung: tests
                                       whether the ~30-step ceiling L3 shows is
                                       STRUCTURAL -- i.e. set by harm sources other
                                       than hazards -- or just under-softened)

L0 is the ARC-029 LINEAGE point (V3-EXQ-125a / 063a) and is the ENV negative control.
**It is NOT expected to reproduce the predecessor's occupancy 1.0000, and an earlier
draft of this docstring said it was. That was wrong and is corrected here (Step 4.5
red-team finding F5):** the predecessor saturated because a 12-EPISODE warm cap never
filled a 200-SELECT-CALL window, and at L0's measured ~7 select calls per episode this
run's 250-episode cap fills it in ~29 episodes. So L0 here is expected to show the bar
IN FORCE with occupancy ~q. If it does, that is itself a finding -- the predecessor's
saturation was a warm-budget artifact, not an env property -- and it is reported as
such rather than read as an env result. What L0 genuinely controls is the ENV end of
the ladder: it is the shortest-episode, highest-harm rung, so it is where P1's
run-length half is expected to fail and P3 to pass.

P1 arithmetic that sets the target zone: at least one committed run of >= 3
requires `f * E >= 3` select calls, so E >= 3.5 at f = 0.85 and E >= 20 at
f = 0.15. At the measured 1.0-3.6 env steps per select call, that is roughly
18-60 env steps per episode -- which the ladder brackets from BOTH sides.

WHAT IS MEASURED PER CELL (rung x q x seed)
---------------------------------------------------------------------------
Exactly what the user's option C names, plus the two quantities that decide
whether the numbers are readable at all:
  - `mean_episode_length` (+ min/max) and `select_calls_per_episode`
  - `abs_mean_harm_per_step` and its `sem_harm_per_step`
  - `committed_step_fraction` and the committed-run-length HISTOGRAM
  - `bar_in_force_fraction` -- CONFIRMED in force (~1.0) or the cell is not
    readable as an occupancy measurement at all. This is the gate the
    predecessor lacked.
  - `rv_residual_dispersion_sd` / `_iqr` -- the SD and IQR of the residuals of a
    least-squares line through log(gate variance) vs tick index over the live
    window: the SAME estimator the selector uses
    (`e3_selector._variance_tracking_commit_bar`), recomputed here so the
    quantity the build record says is owed is reported rather than inferred.
  - `window_span_episodes` -- how many EPISODE BOUNDARIES the full window spans.
    The estimator is specified against a WITHIN-run drift ("rv drifts ~5x within
    a single run"); a window pooled across many resets is a different object and
    a reader must be able to see that, even when `bar_in_force` is 1.0.

PRE-REGISTERED FEASIBILITY RULES (thresholds are constants in this file)
---------------------------------------------------------------------------
  P1_FEASIBLE(cell) := bar_in_force_fraction >= 0.95
                       AND committed_step_fraction in [0.15, 0.85]
                       AND mean_committed_run_length >= 3.0
      The band and the floor are VERBATIM from ARC-029's P1. The
      `bar_in_force` conjunct is not from the claim: it is the readability
      precondition, because an occupancy measured while the ABSOLUTE bar is in
      force is a measurement of the old defect, not of this lever.
  P3_FEASIBLE(cell) := TWO conjuncts, on the HARM-ONLY channel (see below).
      (a) RESOLVABILITY: abs_mean_harm_per_step >= 10 * SEM, where the SEM is
          autocorrelation-corrected. EXP-1394 says ">= 10x the measurement floor"
          and ARC-029 P3 "an order of magnitude above the measurement floor";
          NEITHER defines the floor, and for a MEAN estimated from N steps the
          floor is the estimator's own resolution.
      (b) ABSOLUTE SCALE: abs_mean_harm_per_step >= 0.002 / 0.20 = 0.01.
          (a) ALONE IS NOT ENOUGH, and this is the Step 4.5 red-team's sharpest
          finding: (a) is SCALE-INVARIANT -- multiply every harm parameter by k
          and both the mean and the SD scale by k, so the ratio does not move. It
          would certify a soft rung at |mean harm| ~ 0.008/step, the evidence run
          would be re-queued there, C1's UNTOUCHED absolute bar (0.002) would be a
          25% relative effect, and the resulting null would read as FALSIFYING
          under the claim's own rule while actually being the EXQ-227 floor regime
          P3 exists to exclude. (b) is DERIVED from C1's own registered floor and
          the build record's own framing of it ("a null at that floor excludes
          relative effects >= 20%"), not invented.
      THE CHANNEL MATTERS. `harm_signal` is the NET of harm and benefit
      (causal_grid_world.py:2410), and this ladder drives proximity_harm_scale
      BELOW the fixed proximity_benefit_scale at its soft rungs, so a net mean
      near zero there is CANCELLATION, not "harm on the floor". P3 therefore
      routes on per-step deltas of `env.total_harm`; the net reward (V3-EXQ-125a's
      own `mean_harm_per_step`, kept for lineage comparability) and the benefit
      channel are reported beside it.
  RUNG JOINTLY FEASIBLE := some q in {0.25, 0.50, 0.75} is BOTH P1- and
      P3-feasible on BOTH seeds. Both seeds, not a majority: at n=2 a "majority"
      is not a thing, and a feasibility claim that holds on one seed of two is
      not a region.

DISPOSITION -- RECORDED, NOT APPLIED (user instruction, verbatim intent)
---------------------------------------------------------------------------
  no jointly-feasible rung -> ARC-029 converts to `substrate_conditional` on
      commitment-occupancy sustainment, per ARC-029's OWN P1 text ("If P1 still
      fails with those armed, ARC-029 converts to `substrate_conditional` on
      commitment-occupancy sustainment and this falsifier is not readable").
      **That goes to the user as a decision WITH THE NUMBERS. This run does not
      apply it, does not touch claims.yaml, and returns no claim verdict.**
  a region found -> the pre-registered V3-EXQ-063b design
      (`v3_exq_1066_...py`) is re-queued at that setting, with C1 re-expressed if
      needed. Also a further decision, also not taken here.

DV-SYMMETRY INVARIANCE (mandatory declaration)
---------------------------------------------------------------------------
Manipulation: the env harm parameters (hazard count, `hazard_harm`,
`proximity_harm_scale`) and the commit quantile q. DVs: mean episode length,
select calls per episode, |mean harm/step| and its SEM, committed_step_fraction,
committed-run-length histogram, rv residual dispersion.
The DVs' symmetry group is (a) permutation of episodes within a cell and (b)
affine rescaling of the reward channel. The manipulation is invariant under
NEITHER: the harm parameters change the magnitude of `harm_signal` itself, hence
both the survival time and the DV scale (that is the bind under test), and q
moves the gate's comparison point, hence which select calls are committed. There
is no broadcast-constant / monotone-rescale / permutation channel here, so the
V3-EXQ-604c argmax-invariance class does not apply.
NOTE the one genuine invariance, declared because it is real: `|mean harm/step|
x episode_length` is CONSERVED at ~1.0-1.4 by the health budget, so those two
DVs are NOT independent and must never be reported as two separate effects. That
conservation is the finding, not a confound -- it is why the ladder is
one-dimensional.

MECH-131: not applicable. No DV here is a candidate-set statistic, so no
`selected_action_entropy` reading is required to license it.

INSTRUMENTATION -- three traps, all measured on this substrate 2026-09-19
---------------------------------------------------------------------------
1. `last_score_diagnostics` is populated ONLY under
   `if self.e3_score_decomp_enabled:` (`e3_selector.py:4098`), an INSTANCE
   attribute defaulting to False (`:639`) -- NOT a config field. Without
   `agent.e3.e3_score_decomp_enabled = True` a driver reading `committed` from it
   measures ZERO selections and reports occupancy 0.0, silently. Set and asserted
   here.
2. It also LATCHES: populated only inside select(), and on a non-E3 tick it still
   holds the previous selection. Cleared in a `StepHooks.on_sense` hook (which
   fires after sense() and before select_action), so a non-None read is always a
   FRESH selection. `n_latched_ticks` is emitted so the true denominator is
   auditable.
3. `e3_steps_per_tick` does NOT set the effective select cadence: configured 3,
   MEASURED 1.0-3.6 env steps per select, because MECH-091 `phase_reset` forces
   an E3 tick on salient events and this env delivers a negative reward on nearly
   every step. So the cadence is MEASURED and REPORTED per cell
   (`env_steps_per_select_call`) rather than assumed from config.
This driver NEVER assigns to `e3._running_variance` (the V3-EXQ-063a defect);
asserted by AST walk over this file's own assignment targets.

STEP 4.5 RED-TEAM, THIS ITERATION -- CONTESTED (Fable; this session runs on Opus)
---------------------------------------------------------------------------
A second, independent pass, scoped to the one changed mechanism. Six findings,
every one verified against source or against V3-EXQ-1070's own manifest before
acting; all fixed, none dismissed.

  F1 BLOCKING-adjacent and the most valuable: the drop rule can only ever remove a
     SHORT-episode rung (the budget is ceil(target/length)), which is the ladder's
     MINIMUM and, at L0, the lineage control -- and D1 / the label / the degeneracy
     reason were all computed over the filtered ladder without being told a drop
     had happened. On 1070's own rung means, dropping L0 moves D1 from 3.32 to
     1.94, so a BUDGET failure would have been reported as "the ladder did not
     sweep" while the label still named a feasibility region. FIXED: a drop is now
     GATED (`all_rungs_tick_budgetable`) and routes to substrate_not_ready_requeue;
     the degeneracy reason names the dropped rungs; excluded rows are flagged
     `excluded_from_analysis`.
  F2 the pilot's own error is rung-correlated -- it measured P1 length only 14
     episodes into P0, while the real P1 starts a full TARGET_P0_TICKS in, and that
     gap is ~6x larger at L0 than at L5. FIXED: the pilot is now TWO-STAGE and
     measures P1 at the real P1 start point. Also corrected a claim: applying this
     rule to 1070's realised whole-phase lengths yields ~1.03, but that is
     arithmetic about `ceil`, not a prediction of pilot error, and the docstring
     now says so and records the fields needed to audit the residual.
  F3 "closed-loop over the seed" was only true for agent init: the pilot env used
     `seed + 90000`, a different layout sequence. Since `CausalGridWorldV2` seeds a
     PER-INSTANCE generator (`causal_grid_world.py:1604`), the offset bought
     nothing. FIXED: pilot envs are separate instances at the SAME seed.
  F4 equal TICKS equalises e2 and the rv EMA but NOT the REINFORCE heads, which
     step per P1 EPISODE and so vary ~5x inversely with episode length. Structural
     (ticks = episodes x length), pre-existing in 1070, and NOT silently fixed.
     DECLARED in full above with the numbers, and `episodes_p1` recorded per cell.
  F5 the pilot left the global RNG streams in a pilot-dependent state that the real
     training call inherited. FIXED: the pilot now runs BEFORE the real agent is
     constructed, so the real run always starts from seeded_construct's own
     reset_all_rng(seed), as 1070's did.
  F6 `EPISODES_PER_RUN` did not cover the capped path. FIXED: derived from
     MAX_TRAIN_EPISODES_PER_PHASE.

Not re-spawned: CONTESTED, not BLOCKING.

STEP 4.5 RED-TEAM, INHERITED FROM V3-EXQ-1070 -- CONTESTED (Fable)
---------------------------------------------------------------------------
One pass, foreground, not iterated to CLEAR. Six findings, ALL verified against
source or against this driver's own dry-run manifest before acting, ALL fixed:

  F1 The committed-run histogram walked the FLAT cross-episode sequence, so P1's
     run-length half was BLIND to episode length -- the quantity this ladder
     manipulates and the quantity the verdict text attributes its outcome to.
     Confirmed in this driver's own dry-run manifest: n_episodes 2,
     n_select_calls 14, histogram {'14': 1} -- ONE run spanning both episodes.
     FIXED: runs are cut at episode boundaries
     (`run_length_histogram_within_episode`, the P1 statistic); the flat view is
     retained as a descriptive so the size of the boundary effect stays visible.
  F2 P3's SNR test is scale-invariant and so could not discharge P3's job in the
     pipeline. FIXED: the second, absolute conjunct above, plus an
     autocorrelation-corrected SEM (proximity harm is autocorrelated, so the
     plain SEM understates the floor), plus the measured phase now stopping on a
     fixed ENV-STEP target -- stopping on select calls alone handed the softer
     rungs up to 3.6x more samples, and so a sqrt(3.6)x easier P3, for a cadence
     reason.
  F3 The harm DV was the NET signal, which carries benefit. FIXED: P3 routes on
     the harm-only channel; net and benefit reported beside it.
  F4 Training was budgeted in EPISODES while `_train_all_on_agent` breaks on
     `done`, so training depth scaled with the manipulated variable (~5x across
     the ladder; visible in the dry run at 20 vs 15 ticks for equal episode
     budgets). FIXED: per-rung episode counts derived from the pre-measured
     episode lengths to equalise TICKS, realised ticks recorded, and a
     `training_tick_budget_equalised` precondition bounding their spread.
  F5 D4's stated inference rule was foreclosed by this run's own warm budget.
     FIXED: corrected in the ladder section above.
  F6 The three q cells share one agent in a fixed order and `agent.reset()` does
     not clear residue. FIXED: q order counterbalanced by seed, order recorded.

Not re-spawned: the verdict was CONTESTED, not BLOCKING, and the skill's rule is
one pass, never iterated to CLEAR.

SLEEP DRIVER: not applicable (no sleep flag is set by this driver).
"""

import argparse
import ast
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments._harness import StepHarness, StepHooks
from experiments._lib.allon_training import _train_all_on_agent
from experiments._lib.arm_fingerprint import arm_cell, seeded_construct
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from experiment_protocol import emit_outcome

ANCHOR_REACHABILITY_EXEMPT = (
    "All four readiness preconditions are reachable BY CONSTRUCTION or by direct "
    "measurement, not by a hand-written rubric that could be narrower than the state "
    "it anchors to (the V3-EXQ-778d shape this lint guards). "
    "(1) variance_tracking_bar_in_force_somewhere reads the selector's OWN "
    "`variance_tracking_commit_bar is not None` flag: once the window is full the bar "
    "is non-None on every tick unless the documented degenerate-window guard fires, so "
    "the 0.95 floor is the flag's own definition, not a scoring threshold. "
    "(2) commit_gate_window_filled_in_some_cell is a count of cells reaching "
    "len(window) >= maxlen -- an identity on the deque. "
    "(3) driver_never_writes_running_variance is an AST count that is 0 by "
    "construction in this file and is re-checked at runtime. "
    "(4) training_collapsed_running_variance has a MEASURED reference on this exact "
    "recipe: 2026-09-19 on ree-cloud-4, rv 5.0e-01 -> 9.0e-03 = 1.75 decades against "
    "this precondition's 0.5-decade floor, a 3.5x margin."
)

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_1070a_arc029_env_operating_point_feasibility"
QUEUE_ID = "V3-EXQ-1070a"
SUPERSEDES = "V3-EXQ-1070"
CLAIM_IDS = ["ARC-029"]

SEEDS = [0, 42]
Q_LADDER = (0.25, 0.50, 0.75)
COMMIT_WINDOW = 200
E3_STEPS_PER_TICK = 3
PRECISION_EMA_ALPHA = 0.05

# (rung_id, num_hazards, hazard_harm, proximity_harm_scale). Pre-measured with a
# random policy over 30 episodes/rung, 2026-09-19 -- see the ladder table in the
# module docstring for the episode lengths each produces.
LADDER: Tuple[Tuple[str, int, float, float], ...] = (
    ("L0_lineage_control", 5, 0.05,  0.15),
    ("L1",                 4, 0.04,  0.10),
    ("L2",                 3, 0.03,  0.06),
    ("L3",                 3, 0.02,  0.035),
    ("L5_extreme",         1, 0.005, 0.005),
)

# Pre-measured mean episode length per rung (random policy, 30 episodes/rung,
# 2026-09-19 on ree-cloud-4). Used ONLY to derive per-rung episode counts that
# budget the RANDOM-POLICY zworld P0a phase only (see _episode_budgets_from_pilot);
# every other phase is pilot-budgeted per (rung, seed). Realised ticks are recorded
# and gated, so an error here shows up as a precondition failure, not as a silent
# confound. L5 is extrapolated from the L3/L4 plateau and is deliberately the
# rung whose episode-length ceiling the run is testing.
LADDER_EP_LEN: Dict[str, float] = {
    "L0_lineage_control": 6.2,
    "L1": 10.8,
    "L2": 22.5,
    "L3": 29.5,
    "L5_extreme": 40.0,
}

# --- pre-registered feasibility thresholds -----------------------------------
P1_OCCUPANCY_BAND = (0.15, 0.85)    # ARC-029 P1, verbatim
P1_RUNLEN_FLOOR = 3.0               # ARC-029 P1, verbatim
BAR_IN_FORCE_FLOOR = 0.95           # readability, not from the claim
P3_SNR_FLOOR = 10.0                 # ">= 10x the measurement floor", floor = SEM
# SECOND, INDEPENDENT P3 conjunct -- added after the Step 4.5 red-team (finding F2).
# The SNR test above is SCALE-INVARIANT: multiply every harm parameter by k and mean
# and SD both scale by k, so the ratio does not move. It correctly tests whether the
# MEAN is resolvable, but it cannot discharge the job P3 has in ARC-029's pipeline,
# which is to keep the DV large enough that C1's ABSOLUTE bar is reachable. Without
# this conjunct a soft rung passes P3 at |mean harm| ~ 0.008/step, the evidence run is
# re-queued there, C1's untouched `max(0.5*SD(delta), 0.002)` floor is 25% of the base
# rate, and the resulting null reads as FALSIFYING under the claim's own rule -- while
# actually being the EXQ-227 floor regime P3 exists to exclude.
# DERIVED, not invented: C1's registered absolute floor is 0.002 and the build record
# frames it as "a null at that floor excludes relative effects >= 20%", so the DV must
# satisfy 0.002 / |mean harm/step| <= 0.20.
C1_ABS_FLOOR = 0.002                # verbatim from ARC-029 CONFIRMING / EXP-1394
C1_MAX_RELATIVE_EFFECT = 0.20       # the build record's own framing of that floor
P3_ABS_HARM_FLOOR = C1_ABS_FLOOR / C1_MAX_RELATIVE_EFFECT   # = 0.01 harm/step

# --- budgets -----------------------------------------------------------------
STEPS_PER_EPISODE = 120
# TRAINING IS BUDGETED IN TICKS, NOT EPISODES -- red-team finding F4. `_train_all_on_agent`
# breaks each episode on `done`, so a fixed EPISODE budget hands a 6-step rung ~5x fewer
# training ticks than a 30-step rung. That confound runs along exactly the dimension the
# ladder manipulates, and it would corrupt BOTH the rv-residual-dispersion deliverable and
# the worst-cell readiness gate (which would then fail at L0 for a manipulation-caused
# reason and route the whole run to substrate_not_ready_requeue). Per-rung episode counts
# are derived from the pre-measured episode lengths so realised TICKS are ~equal; the
# realised counts are recorded and a readiness precondition bounds their spread.
TARGET_P0_TICKS = 400
TARGET_P1_TICKS = 900
TARGET_ZWORLD_P0_TICKS = 400        # SD-070: without it z_world stays a frozen
                                    # random projection and rv never collapses
PILOT_EPISODES = 14                 # per phase, per (rung, seed) -- the closed-loop
                                    # measurement that replaces the random-policy table
MAX_TRAIN_EPISODES_PER_PHASE = 400  # a GUARD, not an expected path: 1070's worst case
                                    # (L0, 5.1 ticks/episode) needs ~177 P1 episodes
TRAIN_TICK_SPREAD_CEILING = 2.0     # max/min realised total train ticks across rungs
# The [train] ep N/M denominator, held CONSTANT across rungs so it matches the queue
# entry's `episodes_per_run` exactly (the runner overwrites that field from the prints).
# Per-rung episode counts vary by design -- see _episode_budgets_from_pilot -- and are
# pilot-derived at run time, so no single true value exists. This is held CONSTANT so it
# matches the queue entry's `episodes_per_run` exactly (the runner overwrites that field
# from the prints), and is set above MAX_TRAIN_EPISODES_PER_PHASE so progress cannot
# exceed 100% at any rung -- INCLUDING the capped path, where p0 and p1 can BOTH sit
# at MAX_TRAIN_EPISODES_PER_PHASE and the trainer's own total is their sum.
EPISODES_PER_RUN = 2 * MAX_TRAIN_EPISODES_PER_PHASE + 20
WARM_EPISODE_CAP = 250              # generous ON PURPOSE -- the predecessor's 12-episode
                                    # cap is what made "the bar cannot engage"
                                    # indistinguishable from "the warm-up was too short"
MEAS_ENV_STEP_TARGET = 900          # the HARM DV's denominator, equalised across rungs.
                                    # Stopping on SELECT CALLS alone (red-team F2) would
                                    # hand the softer rungs up to 3.6x more env steps --
                                    # the measured cadence range -- and so a sqrt(3.6)
                                    # easier SNR test, for a cadence reason.
MEAS_SELECT_MIN = 250               # ...but occupancy and run length need select-call n,
                                    # so the measured phase runs until BOTH are met.
MEAS_EPISODE_CAP = 250

BASE_ENV = dict(
    size=12, num_resources=5, proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15, hazard_field_decay=0.5,
    use_proxy_fields=True,
)
TRAIN_DRIFT = dict(env_drift_interval=5, env_drift_prob=0.1)
EVAL_DRIFT = dict(env_drift_interval=50, env_drift_prob=0.0)

_ZG = ZGoalStreamAccumulator()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _flat_scalar(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _mean(xs: List[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _sd(xs: List[float]) -> Optional[float]:
    return statistics.pstdev(xs) if len(xs) >= 2 else None


def _assert_no_running_variance_writes() -> Dict[str, Any]:
    """The V3-EXQ-063a defect, as a source-level check: this driver must never
    ASSIGN to `e3._running_variance`. An AST walk over assignment TARGETS, not a
    substring scan -- this file's docstring discusses the attribute by name and a
    textual check would flag its own explanation."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    offenders: List[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for tgt in targets:
            for sub in ast.walk(tgt):
                if isinstance(sub, ast.Attribute) and sub.attr == "_running_variance":
                    offenders.append(getattr(node, "lineno", -1))
    return {"n_assignments": len(offenders), "lines": sorted(set(offenders)),
            "method": "ast_walk_over_assign_targets"}


def _residual_dispersion(window: List[float]) -> Dict[str, Optional[float]]:
    """SD and IQR of the residuals of a least-squares line through
    log(gate variance) vs tick index -- the SAME detrend
    `e3_selector._variance_tracking_commit_bar` applies before taking its
    quantile. Recomputed here so the 'rv residual dispersion on a real trained
    agent' the build record sec 6a says is owed is REPORTED, not inferred from
    occupancy. Returns Nones on a degenerate window rather than inventing a
    number (the same shape the selector's own guard refuses on)."""
    vals = [v for v in window if v is not None and math.isfinite(v) and v > 0.0]
    n = len(vals)
    if n < 8:
        return {"rv_residual_dispersion_sd": None, "rv_residual_dispersion_iqr": None,
                "rv_window_log_slope": None, "rv_window_drift_ratio": None,
                "rv_residual_n": n}
    hi, lo = max(vals), min(vals)
    if (hi - lo) <= abs(hi) * 1e-9:
        return {"rv_residual_dispersion_sd": 0.0, "rv_residual_dispersion_iqr": 0.0,
                "rv_window_log_slope": 0.0, "rv_window_drift_ratio": 1.0,
                "rv_residual_n": n}
    logs = [math.log(v) for v in vals]
    mean_i = (n - 1) / 2.0
    mean_y = sum(logs) / n
    sxx = sum((i - mean_i) ** 2 for i in range(n))
    slope = (sum((i - mean_i) * (logs[i] - mean_y) for i in range(n)) / sxx) if sxx > 0 else 0.0
    resid = sorted(logs[i] - (mean_y + slope * (i - mean_i)) for i in range(n))
    q1 = resid[int(0.25 * (n - 1))]
    q3 = resid[int(0.75 * (n - 1))]
    return {"rv_residual_dispersion_sd": float(statistics.pstdev(resid)),
            "rv_residual_dispersion_iqr": float(q3 - q1),
            "rv_window_log_slope": float(slope),
            "rv_window_drift_ratio": float(hi / lo),
            "rv_residual_n": n}


def _make_env(seed: int, nh: int, hh: float, ph: float,
              drift: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, num_hazards=nh, hazard_harm=hh,
                             proximity_harm_scale=ph, **BASE_ENV, **drift)


def _config_slice(env: CausalGridWorldV2, rung: str, q: float) -> Dict[str, Any]:
    return dict(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=32, world_dim=32,
        alpha_world=0.9,                 # SD-008: the 0.3 default starves z_world
        use_variance_tracking_commit_threshold=True,
        commit_threshold_quantile=q,
        commit_threshold_quantile_window=COMMIT_WINDOW,
        breath_period=0,                 # MECH-108 OFF: this run measures the BASE
                                         # bar's reachable occupancy, not the
                                         # alternation driver. from_dims defaults
                                         # breath_period to 50, so leaving it
                                         # unset would silently arm a sweep.
        rung=rung,
    )


def _build_agent(env: CausalGridWorldV2, q: float, rung: str) -> Tuple[REEAgent, Dict[str, Any]]:
    slice_ = _config_slice(env, rung, q)
    kwargs = {k: v for k, v in slice_.items() if k != "rung"}
    config = REEConfig.from_dims(**kwargs)
    assert config.e3.use_variance_tracking_commit_threshold is True
    assert config.e3.commit_threshold_quantile == q
    assert config.e3.commit_threshold_quantile_window == COMMIT_WINDOW
    assert config.heartbeat.breath_period == 0
    # Not from_dims parameters -- they live on the sub-configs. getattr-check
    # first so a rename is a loud failure, not a new attribute nobody reads.
    assert hasattr(config.e3, "precision_ema_alpha")
    assert hasattr(config.heartbeat, "e3_steps_per_tick")
    config.e3.precision_ema_alpha = PRECISION_EMA_ALPHA
    config.heartbeat.e3_steps_per_tick = E3_STEPS_PER_TICK
    agent = REEAgent(config)
    agent.e3.e3_score_decomp_enabled = True   # trap 1 -- see the docstring
    assert agent.clock.e3_steps_per_tick == E3_STEPS_PER_TICK
    assert agent.e3._commit_gate_variance_window is not None
    assert agent.e3._commit_gate_variance_window.maxlen == COMMIT_WINDOW
    return agent, slice_


class _Roll:
    """Steps (agent, env) and records BOTH denominators -- select calls for the
    occupancy/run-length DVs (the unit ARC-029's >= 3 is in) and env steps for
    the harm DV."""

    def __init__(self, agent: REEAgent, env: CausalGridWorldV2, seed: int):
        self.agent, self.env, self.seed = agent, env, seed
        self.hooks = StepHooks(on_sense=self._clear)
        self.reset_counters()

    def _clear(self, **_kw) -> None:
        # Fires after sense() and BEFORE select_action (step()'s order item 6),
        # so a non-None read downstream is always a FRESH selection.
        self.agent.e3.last_score_diagnostics = None

    def reset_counters(self) -> None:
        self.committed: List[bool] = []
        self.bar_in_force: List[bool] = []
        self.net_reward: List[float] = []    # SIGNED per-step reward -- V3-EXQ-125a's
                                             # `mean_harm_per_step`, kept for lineage
                                             # comparability
        self.harm_only: List[float] = []     # env.total_harm DELTA per step. The NET
                                             # signal also carries BENEFIT
                                             # (causal_grid_world.py:2410), and this
                                             # ladder drives proximity_harm_scale BELOW
                                             # proximity_benefit_scale at its soft rungs,
                                             # so a net mean near zero there is
                                             # CANCELLATION, not "harm on the floor"
                                             # (red-team finding F3). P3 routes on this
                                             # channel; the net one is reported beside it.
        self.benefit_only: List[float] = []
        self.ep_lengths: List[int] = []
        self.ep_index_at_select: List[int] = []
        self.committed_by_episode: List[List[bool]] = []
        self.log10_precision: List[float] = []
        self.n_env_steps = 0
        self.n_latched_ticks = 0
        self.n_episodes = 0
        self.window_snapshot: List[float] = []

    def run(self, episode_cap: int, steps: int, collect: bool,
            stop_env_step_target: Optional[int] = None,
            stop_select_min: Optional[int] = None,
            stop_when_window_full: bool = False) -> None:
        agent = self.agent
        for _ep in range(episode_cap):
            # Same contract as StepHarness.run_episode (_harness.py:455-457).
            _flat, obs = self.env.reset()
            agent.reset()
            harness = StepHarness(agent, self.env, train_mode=False,
                                  hooks=self.hooks, seed=self.seed)
            ep_len = 0
            ep_committed: List[bool] = []
            prev_harm = float(getattr(self.env, "total_harm", 0.0))
            prev_benefit = float(getattr(self.env, "total_benefit", 0.0))
            for _ in range(steps):
                result = harness.step(obs)
                obs = result.next_obs_dict
                ep_len += 1
                self.n_env_steps += 1
                diag = agent.e3.last_score_diagnostics
                if isinstance(diag, dict) and "committed" in diag:
                    if collect:
                        c = bool(diag["committed"])
                        self.committed.append(c)
                        ep_committed.append(c)
                        self.bar_in_force.append(
                            diag.get("variance_tracking_commit_bar") is not None)
                        self.log10_precision.append(
                            math.log10(max(agent.e3.current_precision, 1e-300)))
                        self.ep_index_at_select.append(self.n_episodes)
                else:
                    self.n_latched_ticks += 1
                if collect:
                    self.net_reward.append(float(result.harm_signal))
                    cur_h = float(getattr(self.env, "total_harm", 0.0))
                    cur_b = float(getattr(self.env, "total_benefit", 0.0))
                    self.harm_only.append(max(0.0, cur_h - prev_harm))
                    self.benefit_only.append(max(0.0, cur_b - prev_benefit))
                    prev_harm, prev_benefit = cur_h, cur_b
                if result.done:
                    break
            if collect:
                self.committed_by_episode.append(ep_committed)
            self.ep_lengths.append(ep_len)
            self.n_episodes += 1
            _ZG.observe(agent)
            win = agent.e3._commit_gate_variance_window
            if stop_when_window_full and win is not None and len(win) >= COMMIT_WINDOW:
                break
            if (stop_env_step_target is not None
                    and self.n_env_steps >= stop_env_step_target
                    and len(self.committed) >= (stop_select_min or 0)):
                break
        win = agent.e3._commit_gate_variance_window
        self.window_snapshot = list(win) if win is not None else []

    @staticmethod
    def _hist(seq: List[bool]) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        cur = 0
        for c in seq:
            if c:
                cur += 1
            elif cur:
                hist[str(cur)] = hist.get(str(cur), 0) + 1
                cur = 0
        if cur:
            hist[str(cur)] = hist.get(str(cur), 0) + 1
        return hist

    def run_length_histogram_within_episode(self) -> Dict[str, int]:
        """THE P1 STATISTIC. Committed runs are cut at every episode boundary.

        Red-team finding F1, confirmed against this driver's own dry-run manifest:
        computing the histogram over the FLAT cross-episode sequence reported ONE
        run of 14 select calls spanning 2 episodes (`n_episodes: 2`,
        `n_select_calls: 14`, `{'14': 1}`). Nothing resets commitment at a reset --
        `agent.reset()` (agent.py:3585) touches E3 only via
        `clear_learned_channel_eligibility()`, and `_running_variance` and the
        commit-gate window both persist by design. So the flat statistic is BLIND
        to episode length, which is the quantity this whole ladder manipulates: it
        would have made P1 insensitive to the manipulation while the verdict text
        attributed the outcome to it.
        """
        hist: Dict[str, int] = {}
        for ep in self.committed_by_episode:
            for k, v in self._hist(ep).items():
                hist[k] = hist.get(k, 0) + v
        return hist

    def run_length_histogram_flat(self) -> Dict[str, int]:
        """Descriptive only -- the cross-episode view, reported beside the P1
        statistic so the size of the boundary effect is visible rather than
        implicit."""
        return self._hist(self.committed)

    def summary(self) -> Dict[str, Any]:
        hist = self.run_length_histogram_within_episode()
        hist_flat = self.run_length_histogram_flat()
        n_runs = sum(hist.values())
        total_committed_in_runs = sum(int(k) * v for k, v in hist.items())
        n_runs_flat = sum(hist_flat.values())
        n_sel = len(self.committed)
        harm_mean = _mean(self.harm_only)
        harm_sd = _sd(self.harm_only)
        n_h = len(self.harm_only)
        sem = (harm_sd / math.sqrt(n_h)) if (harm_sd is not None and n_h) else None
        # The SEM above assumes independent steps. Proximity harm is autocorrelated
        # over consecutive steps near a hazard, so the plain SEM UNDERSTATES the
        # measurement floor (red-team F2). Report lag-1 autocorrelation and an
        # effective-n SEM beside it rather than silently using the optimistic one.
        rho1 = None
        if harm_sd and n_h >= 20 and harm_mean is not None and harm_sd > 0:
            num = sum((self.harm_only[i] - harm_mean) * (self.harm_only[i + 1] - harm_mean)
                      for i in range(n_h - 1))
            rho1 = float(num / ((n_h - 1) * harm_sd * harm_sd))
        sem_eff = sem
        if sem is not None and rho1 is not None and -0.99 < rho1 < 0.99:
            sem_eff = float(sem * math.sqrt(max(1e-6, (1.0 + rho1) / (1.0 - rho1))))
        net_mean = _mean(self.net_reward)
        benefit_mean = _mean(self.benefit_only)
        # how many EPISODE BOUNDARIES the window spans -- the "is this a
        # within-run quantile at all" question, separate from bar_in_force
        span = None
        if self.ep_index_at_select:
            tail = self.ep_index_at_select[-COMMIT_WINDOW:]
            span = int(tail[-1] - tail[0] + 1)
        out = {
            "n_select_calls": n_sel,
            "n_env_steps": self.n_env_steps,
            "n_episodes": self.n_episodes,
            "n_latched_ticks": self.n_latched_ticks,
            "mean_episode_length": _mean([float(x) for x in self.ep_lengths]),
            "min_episode_length": min(self.ep_lengths) if self.ep_lengths else None,
            "max_episode_length": max(self.ep_lengths) if self.ep_lengths else None,
            "select_calls_per_episode": (n_sel / self.n_episodes) if self.n_episodes else None,
            "env_steps_per_select_call": (self.n_env_steps / n_sel) if n_sel else None,
            "committed_step_fraction": (sum(self.committed) / n_sel) if n_sel else None,
            # THE P1 STATISTIC -- runs cut at episode boundaries (red-team F1)
            "mean_committed_run_length": (total_committed_in_runs / n_runs) if n_runs else 0.0,
            "committed_run_length_histogram": hist,
            "n_committed_runs": n_runs,
            # descriptive: the cross-episode view, so the boundary effect is visible
            "mean_committed_run_length_flat": (
                sum(int(k) * v for k, v in hist_flat.items()) / n_runs_flat)
                if n_runs_flat else 0.0,
            "committed_run_length_histogram_flat": hist_flat,
            "bar_in_force_fraction": (
                sum(self.bar_in_force) / len(self.bar_in_force)) if self.bar_in_force else 0.0,
            # HARM-ONLY channel (env.total_harm deltas) -- what P3 routes on
            "mean_harm_per_step": harm_mean,
            "abs_mean_harm_per_step": abs(harm_mean) if harm_mean is not None else None,
            "sd_harm_per_step": harm_sd,
            "sem_harm_per_step": sem,
            "sem_harm_per_step_autocorr_corrected": sem_eff,
            "harm_lag1_autocorrelation": rho1,
            "harm_snr": (abs(harm_mean) / sem_eff) if (harm_mean is not None and sem_eff)
                        else None,
            "harm_snr_uncorrected": (abs(harm_mean) / sem) if (harm_mean is not None and sem)
                                    else None,
            "n_harm_steps": n_h,
            # NET reward (V3-EXQ-125a's `mean_harm_per_step`), and the BENEFIT channel,
            # so a soft rung's near-zero NET is readable as cancellation rather than
            # as an absent harm signal (red-team F3)
            "mean_net_reward_per_step": net_mean,
            "mean_benefit_per_step": benefit_mean,
            "log10_precision_mean": _mean(self.log10_precision),
            "log10_precision_sd": _sd(self.log10_precision),
            "window_span_episodes": span,
            "window_fill": len(self.window_snapshot),
            # the conserved quantity that IS the bind
            "health_budget_product": (
                abs(harm_mean) * _mean([float(x) for x in self.ep_lengths])
                if (harm_mean is not None and self.ep_lengths) else None),
            "mean_committed_runs_per_episode": (
                n_runs / self.n_episodes) if self.n_episodes else None,
        }
        out.update(_residual_dispersion(self.window_snapshot))
        return out


def _pilot_ticks_per_episode(rung: str, nh: int, hh: float, ph: float,
                             seed: int, steps: int,
                             episodes: int = PILOT_EPISODES,
                             p0_tick_target: int = TARGET_P0_TICKS) -> Dict[str, Any]:
    """CLOSED-LOOP budget measurement for ONE (rung, seed), in TWO STAGES.

    Replaces V3-EXQ-1070's feed-forward `LADDER_EP_LEN` lookup, which measured a
    RANDOM policy and over-predicted training-time episode length by 1.4-2.9x at
    the soft rungs -- the defect that made 1070 self-route at a 3.96 tick spread.

    WHY TWO STAGES. Episode length is NON-STATIONARY across training: 1070's own
    realised P1/P0 length ratio ranges 0.66-1.62 within a single (rung, seed). A
    one-stage pilot would measure P1 length 14 episodes into P0, whereas the real
    P1 begins after a full TARGET_P0_TICKS of P0 -- and the size of that gap is
    itself rung-correlated (at L0 a 14-episode P0 is ~70 ticks against the real
    ~400; at L5 it is ~300), i.e. it would re-introduce a rung-correlated error in
    the act of removing one. So:
      stage A  measure P0 ticks/episode over `episodes` episodes, behind a
               realistically-sized zworld P0a warmup;
      stage B  continue the SAME pilot agent through a FULL-LENGTH P0 sized from
               stage A, then measure P1 ticks/episode over `episodes` episodes --
               i.e. at the point in training where the real P1 actually starts.
    The two stages are separate trainer calls, so Adam's moment estimates reset
    between them. That is harmless HERE and only here: the pilot agent is thrown
    away. It is exactly what must NOT be done to the real run, which is why the
    real training is a single call (see the module docstring).

    The pilot's envs are separate INSTANCES built from the SAME seed as the real
    training env. `CausalGridWorldV2` seeds a per-instance generator
    (`causal_grid_world.py:1604`), so a separate instance consumes none of the
    real env's RNG while still seeing the same layout sequence -- which is what
    makes the estimate closed-loop over the SEED and not only over the rung.
    """
    env_a = _make_env(seed, nh, hh, ph, TRAIN_DRIFT)
    zw = _make_env(seed, nh, hh, ph, TRAIN_DRIFT)
    agent, _slice = seeded_construct(
        seed, lambda: _build_agent(env_a, Q_LADDER[len(Q_LADDER) // 2], rung))
    zp0 = max(1, int(math.ceil(TARGET_ZWORLD_P0_TICKS
                               / max(1.0, LADDER_EP_LEN.get(rung, 10.0)))))
    st_a = _train_all_on_agent(
        agent, env_a, seed=seed, p0_episodes=episodes, p1_episodes=0,
        steps_per_episode=steps, rung_id=f"pilotA_{rung}_s{seed}",
        total_denominator=EPISODES_PER_RUN,
        zworld_p0_episodes=min(zp0, episodes * 4), zworld_p0_env=zw,
    )
    p0_len = max(1.0, (st_a.get("n_p0_ticks") or 0) / episodes)
    # bring the pilot agent up to the REAL P0 tick budget before measuring P1
    # `p0_tick_target` is scaled down under --dry-run: at the real target this
    # stage is ~80 episodes at L0, which would make the smoke cost as much as the
    # run it is smoking.
    p0_remaining = max(0, int(math.ceil(p0_tick_target / p0_len)) - episodes)
    p0_remaining = min(p0_remaining, MAX_TRAIN_EPISODES_PER_PHASE)
    env_b = _make_env(seed, nh, hh, ph, TRAIN_DRIFT)
    st_b = _train_all_on_agent(
        agent, env_b, seed=seed, p0_episodes=p0_remaining, p1_episodes=episodes,
        steps_per_episode=steps, rung_id=f"pilotB_{rung}_s{seed}",
        total_denominator=EPISODES_PER_RUN,
        zworld_p0_episodes=0, zworld_p0_env=None,
    )
    p1_len = max(1.0, (st_b.get("n_p1_ticks") or 0) / episodes)
    return {"rung": rung, "seed": seed,
            "pilot_p0_ticks_per_episode": p0_len,
            "pilot_p1_ticks_per_episode": p1_len,
            "pilot_stageA_n_p0_ticks": st_a.get("n_p0_ticks"),
            "pilot_stageB_n_p0_ticks": st_b.get("n_p0_ticks"),
            "pilot_stageB_n_p1_ticks": st_b.get("n_p1_ticks"),
            # recorded so the RESIDUAL pilot error can be audited post hoc against
            # the real run's realised n_p1_ticks / episodes_p1 -- the one thing the
            # 1.029 arithmetic below cannot tell you
            "pilot_n_e2_train_steps": (st_a.get("n_e2_train_steps") or 0)
                                      + (st_b.get("n_e2_train_steps") or 0),
            "pilot_p0_episodes_to_full_budget": p0_remaining + episodes,
            "pilot_episodes_per_phase": episodes}


def _episode_budgets_from_pilot(rung: str, pilot: Dict[str, Any],
                                dry_run: bool) -> Dict[str, Any]:
    """Episode counts that deliver the TARGET TICK budget for this (rung, seed).

    The zworld P0a phase is budgeted from the RANDOM-POLICY table on purpose --
    `run_zworld_p0` rolls out under `RandomPolicy`, so that table is the correct
    predictor for THAT phase and the wrong one for the others. Conflating the two
    is what 1070 got wrong.

    A budget exceeding MAX_TRAIN_EPISODES_PER_PHASE means the rung cannot be
    equalised at all; it is reported `budget_infeasible` and DROPPED from the
    feasibility analysis by the stated rule, never silently truncated to the cap.
    """
    if dry_run:
        return {"p0": 1, "p1": 1, "zp0": 1, "budget_infeasible": False,
                "budget_infeasible_reason": None}
    p0 = int(math.ceil(TARGET_P0_TICKS / pilot["pilot_p0_ticks_per_episode"]))
    p1 = int(math.ceil(TARGET_P1_TICKS / pilot["pilot_p1_ticks_per_episode"]))
    zp0 = int(math.ceil(TARGET_ZWORLD_P0_TICKS / max(1.0, LADDER_EP_LEN.get(rung, 10.0))))
    over = [n for n, v in (("p0", p0), ("p1", p1), ("zp0", zp0))
            if v > MAX_TRAIN_EPISODES_PER_PHASE]
    return {"p0": min(p0, MAX_TRAIN_EPISODES_PER_PHASE),
            "p1": min(p1, MAX_TRAIN_EPISODES_PER_PHASE),
            "zp0": min(zp0, MAX_TRAIN_EPISODES_PER_PHASE),
            "requested_p0": p0, "requested_p1": p1, "requested_zp0": zp0,
            "budget_infeasible": bool(over),
            "budget_infeasible_reason": (
                None if not over else
                f"{'/'.join(over)} budget exceeds MAX_TRAIN_EPISODES_PER_PHASE="
                f"{MAX_TRAIN_EPISODES_PER_PHASE}; this rung cannot be tick-equalised "
                f"and is dropped from the feasibility analysis")}


def _run_cell(agent: REEAgent, seed: int, rung: str, nh: int, hh: float, ph: float,
              q: float, cfg_slice: Dict[str, Any], steps: int, warm_cap: int,
              meas_env_target: int, meas_select_min: int, meas_cap: int,
              label: str) -> Dict[str, Any]:
    env = _make_env(seed * 1000 + 7, nh, hh, ph, EVAL_DRIFT)
    cell_slice = dict(cfg_slice)
    cell_slice.update(q=q, commit_threshold_quantile=q, rung=rung,
                      num_hazards=nh, hazard_harm=hh,
                      proximity_harm_scale=ph, e3_steps_per_tick=E3_STEPS_PER_TICK,
                      precision_ema_alpha=PRECISION_EMA_ALPHA)
    with arm_cell(seed, config_slice=cell_slice, script_path=Path(__file__)) as cell:
        agent.e3.e3_score_decomp_enabled = True
        agent.e3.config.commit_threshold_quantile = q
        assert agent.e3.config.commit_threshold_quantile == q
        win = agent.e3._commit_gate_variance_window
        assert win is not None
        win.clear()
        # WARM: fill the window. Discarded -- until it is full the estimator
        # returns None and the ABSOLUTE 0.40 bar is in force, which is the
        # pre-lever saturated regime and not a measurement of this lever.
        warm = _Roll(agent, env, seed)
        warm.run(warm_cap, steps, collect=False, stop_when_window_full=True)
        warm_fill = len(agent.e3._commit_gate_variance_window)
        roll = _Roll(agent, env, seed)
        roll.run(meas_cap, steps, collect=True,
                 stop_env_step_target=meas_env_target,
                 stop_select_min=meas_select_min)
        row = roll.summary()
        row.update(seed=seed, rung=rung, q=q, num_hazards=nh, hazard_harm=hh,
                   proximity_harm_scale=ph, cell_label=label,
                   warm_episodes_used=warm.n_episodes,
                   warm_env_steps=warm.n_env_steps,
                   warm_prefix_window_fill=warm_fill,
                   window_filled=bool(warm_fill >= COMMIT_WINDOW))
        cell.stamp(row)
    return row


def _p1_feasible(row: Dict[str, Any]) -> bool:
    occ = row.get("committed_step_fraction")
    return bool(row.get("bar_in_force_fraction", 0.0) >= BAR_IN_FORCE_FLOOR
                and occ is not None
                and P1_OCCUPANCY_BAND[0] <= occ <= P1_OCCUPANCY_BAND[1]
                and row.get("mean_committed_run_length", 0.0) >= P1_RUNLEN_FLOOR)


def _p3_feasible(row: Dict[str, Any]) -> bool:
    """TWO conjuncts, because one of them alone cannot do P3's job.

    (a) RESOLVABILITY: |mean| >= 10 x the autocorrelation-corrected SEM. This is
        EXP-1394's ">= 10x the measurement floor" with the floor operationalised
        as the mean estimator's own resolution.
    (b) ABSOLUTE SCALE: |mean| >= 0.002 / 0.20 = 0.01 harm/step. (a) is
        scale-INVARIANT -- multiply every harm parameter by k and both mean and SD
        scale by k -- so it would certify a rung at |mean harm| ~ 0.008/step where
        C1's untouched absolute bar (0.002) is a 25% relative effect, i.e. exactly
        the EXQ-227 floor regime P3 exists to exclude. Derived from C1's own
        registered floor and the build record's own framing of it, not invented.
    """
    snr = row.get("harm_snr")
    mag = row.get("abs_mean_harm_per_step")
    return bool(snr is not None and snr >= P3_SNR_FLOOR
                and mag is not None and mag >= P3_ABS_HARM_FLOOR)


def analyse(rows: List[Dict[str, Any]], src_scan: Dict[str, Any],
            pilots: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    # STATED RULE (the V3-EXQ-1070 repair): a rung whose tick budget cannot be met
    # within MAX_TRAIN_EPISODES_PER_PHASE is DROPPED from the feasibility analysis
    # and from the tick-spread statistic -- never silently truncated to the cap,
    # which would re-create 1070's defect while reporting a passing gate. Dropped
    # rungs are still reported in full so the drop is auditable.
    dropped_rungs = sorted({r["rung"] for r in rows if r.get("budget_infeasible")})
    all_rows = rows
    rows = [r for r in rows if not r.get("budget_infeasible")]
    for r in all_rows:
        r["excluded_from_analysis"] = bool(r.get("budget_infeasible"))
    for r in all_rows:
        r["p1_feasible"] = _p1_feasible(r)
        r["p3_feasible"] = _p3_feasible(r)
        r["jointly_feasible"] = bool(r["p1_feasible"] and r["p3_feasible"])

    rung_ids = [r[0] for r in LADDER if r[0] not in dropped_rungs]
    seeds = sorted({r["seed"] for r in rows})
    rung_table = []
    feasible_rungs = []
    for rung in rung_ids:
        per_q = {}
        for q in Q_LADDER:
            cells = [r for r in rows if r["rung"] == rung and r["q"] == q]
            both = (len(cells) == len(seeds)
                    and all(c["jointly_feasible"] for c in cells))
            per_q[str(q)] = {
                "n_cells": len(cells),
                "jointly_feasible_all_seeds": bool(both),
                "committed_step_fraction": [c.get("committed_step_fraction") for c in cells],
                "mean_committed_run_length": [c.get("mean_committed_run_length") for c in cells],
                "bar_in_force_fraction": [c.get("bar_in_force_fraction") for c in cells],
                "harm_snr": [c.get("harm_snr") for c in cells],
            }
        rung_cells = [r for r in rows if r["rung"] == rung]
        ok = any(v["jointly_feasible_all_seeds"] for v in per_q.values())
        if ok:
            feasible_rungs.append(rung)
        rung_table.append({
            "rung": rung, "jointly_feasible": bool(ok), "per_q": per_q,
            "mean_episode_length": _mean([c["mean_episode_length"] for c in rung_cells
                                          if c.get("mean_episode_length") is not None]),
            "abs_mean_harm_per_step": _mean([c["abs_mean_harm_per_step"] for c in rung_cells
                                             if c.get("abs_mean_harm_per_step") is not None]),
            "mean_net_reward_per_step": _mean([c["mean_net_reward_per_step"] for c in rung_cells
                                               if c.get("mean_net_reward_per_step") is not None]),
            "mean_benefit_per_step": _mean([c["mean_benefit_per_step"] for c in rung_cells
                                            if c.get("mean_benefit_per_step") is not None]),
            "mean_committed_run_length": _mean([c["mean_committed_run_length"] for c in rung_cells
                                                if c.get("mean_committed_run_length") is not None]),
            "mean_committed_run_length_flat": _mean(
                [c["mean_committed_run_length_flat"] for c in rung_cells
                 if c.get("mean_committed_run_length_flat") is not None]),
            "train_ticks_total": _mean([float(c["train_ticks_total"]) for c in rung_cells
                                        if c.get("train_ticks_total")]),
            "health_budget_product": _mean([c["health_budget_product"] for c in rung_cells
                                            if c.get("health_budget_product") is not None]),
            "rv_residual_dispersion_sd": _mean([c["rv_residual_dispersion_sd"] for c in rung_cells
                                                if c.get("rv_residual_dispersion_sd") is not None]),
            "window_span_episodes": _mean([float(c["window_span_episodes"]) for c in rung_cells
                                           if c.get("window_span_episodes") is not None]),
        })

    # spread over the RETAINED rungs only -- a dropped rung is not evidence about
    # whether the retained ones were equalised
    ticks = [r.get("train_ticks_total") for r in rows if r.get("train_ticks_total")]
    tick_spread = (max(ticks) / min(ticks)) if (len(ticks) >= 2 and min(ticks)) else None
    max_bar = max((r.get("bar_in_force_fraction", 0.0) for r in rows), default=0.0)
    best_row = max(rows, key=lambda r: r.get("bar_in_force_fraction", 0.0), default=None)
    n_window_filled = sum(1 for r in rows if r.get("window_filled"))
    rv_collapse = [r.get("rv_collapse_decades") for r in rows
                   if r.get("rv_collapse_decades") is not None]
    worst_collapse = min(rv_collapse) if rv_collapse else None
    l0 = [r for r in rows if r["rung"] == "L0_lineage_control"]
    l0_occ = [r.get("committed_step_fraction") for r in l0]

    preconditions = [
        {"name": "variance_tracking_bar_in_force_somewhere",
         "description": ("BEST cell's fraction of measured SELECT CALLS with a non-None "
                         "variance_tracking_commit_bar. This is the SAME statistic the "
                         "P1_FEASIBLE criterion routes on. If the bar engages NOWHERE on "
                         "the whole ladder, the instrument never ran and no env verdict "
                         "is licensed -- that is substrate_not_ready, not infeasibility."),
         "measured": max_bar, "threshold": BAR_IN_FORCE_FLOOR, "direction": "lower",
         "control": ("the softest rungs are the positive control: their episodes are "
                     "longest, so if the window cannot fill there it cannot fill anywhere"),
         # ANY-quantifier: `met` is max >= floor, so the reported cell is the BEST
         # one (the extremum the claim rests on), not a worst-case offender.
         "best_cell": None if best_row is None else best_row["cell_label"],
         "met": bool(max_bar >= BAR_IN_FORCE_FLOOR)},
        {"name": "commit_gate_window_filled_in_some_cell",
         "description": ("cells whose warm prefix reached the full 200-select-call "
                         "window. Separates 'the bar cannot engage' from 'the warm-up "
                         "was too short' -- the conflation that made the predecessor "
                         "probe unreadable."),
         "measured": n_window_filled, "threshold": 1, "direction": "lower",
         "control": "warm-up runs to WARM_EPISODE_CAP=200 episodes or a full window",
         "met": bool(n_window_filled >= 1)},
        {"name": "training_collapsed_running_variance",
         "description": ("worst cell's log10 collapse of running_variance across "
                         "training. The whole premise is a TRAINED world-forward model; "
                         "an untrained one sits at precision_init and the window goes "
                         "degenerate (V3-EXQ-925a)."),
         "measured": worst_collapse, "threshold": 0.5, "direction": "lower",
         "control": "rv at agent construction vs rv after P0+P1",
         "offending_cell": (None if not rv_collapse else
                            min((r for r in rows
                                 if r.get("rv_collapse_decades") is not None),
                                key=lambda r: r["rv_collapse_decades"])["cell_label"]),
         "met": bool(worst_collapse is not None and worst_collapse >= 0.5)},
        {"name": "training_tick_budget_equalised",
         "description": ("max/min REALISED total training ticks across rungs. Episode "
                         "counts are derived per rung to equalise ticks, because "
                         "`_train_all_on_agent` breaks on `done` and a fixed episode "
                         "budget would hand the short-episode rungs ~5x less training "
                         "along exactly the manipulated dimension -- corrupting both "
                         "the rv-dispersion deliverable and the worst-cell readiness "
                         "gate below. UPPER bound: a LARGE ratio is the failure."),
         "measured": tick_spread, "threshold": TRAIN_TICK_SPREAD_CEILING,
         "direction": "upper",
         "control": "realised n_p0_ticks + n_p1_ticks per (rung, seed)",
         "met": bool(tick_spread is not None and tick_spread <= TRAIN_TICK_SPREAD_CEILING)},
        {"name": "all_rungs_tick_budgetable",
         "description": ("rungs whose tick budget could NOT be met within "
                         "MAX_TRAIN_EPISODES_PER_PHASE and were therefore dropped. "
                         "GATED, and that is the point: the drop can only ever remove "
                         "a SHORT-episode rung (the budget is ceil(target/length), so "
                         "only a small length overruns the cap), which is the ladder's "
                         "MINIMUM and also the lineage control. On 1070's own measured "
                         "rung means, dropping L0 moves D1's episode-length range from "
                         "3.32 to 1.94 -- i.e. a budget failure would present as "
                         "'the ladder did not sweep'. Gating here keeps that "
                         "mis-attribution impossible: a drop routes to "
                         "substrate_not_ready_requeue, which is what it is."),
         "measured": len(dropped_rungs), "threshold": 0, "direction": "upper",
         "control": "per-(rung, seed) pilot budget vs MAX_TRAIN_EPISODES_PER_PHASE",
         "offending_cell": ", ".join(dropped_rungs) if dropped_rungs else None,
         "met": bool(not dropped_rungs)},
        {"name": "driver_never_writes_running_variance",
         "description": "the V3-EXQ-063a defect, as a source-level assert.",
         "measured": src_scan["n_assignments"], "threshold": 0, "direction": "upper",
         "control": "AST walk over this file's assignment targets",
         "met": bool(src_scan["n_assignments"] == 0)},
    ]
    gate_green = all(p["met"] for p in preconditions)

    # non-degeneracy: the ladder must actually MOVE the quantity it manipulates,
    # or "no feasible region" is a statement about a sweep that never swept.
    ep_lengths = [t["mean_episode_length"] for t in rung_table
                  if t["mean_episode_length"] is not None]
    ladder_range = (max(ep_lengths) / min(ep_lengths)) if (len(ep_lengths) >= 2
                                                           and min(ep_lengths)) else None
    ladder_non_degenerate = bool(ladder_range is not None and ladder_range >= 2.0)

    if not gate_green:
        label = "substrate_not_ready_requeue"
        _failed = [p["name"] for p in preconditions if not p["met"]]
        routes_to = ("NO env verdict is licensed; failed precondition(s): "
                     + ", ".join(_failed) + ". "
                     + ("A rung could not be tick-budgeted ("
                        + ", ".join(dropped_rungs) + "), so the ladder this run "
                        "actually swept is not the ladder it pre-registered -- raise "
                        "MAX_TRAIN_EPISODES_PER_PHASE or re-site that rung. "
                        if dropped_rungs else "")
                     + "Do NOT read any of this as ARC-029 infeasibility.")
    elif feasible_rungs:
        label = "arc029_env_joint_feasible_region_found"
        routes_to = ("A region exists where P1 and P3 both hold with the bar confirmed "
                     "in force. DECISION OWED TO THE USER (not applied here): re-queue "
                     "the pre-registered V3-EXQ-063b design "
                     "(experiments/v3_exq_1066_arc029_commitment_mode_harm_variance_bar.py) "
                     "at the named rung, with C1 re-expressed if the harm scale there "
                     "makes the absolute 0.002 bar unreachable.")
    else:
        label = "arc029_p1_p3_jointly_infeasible_substrate_conditional_candidate"
        routes_to = ("The bar engages, the ladder swept, and NO rung satisfies P1 and P3 "
                     "together. DECISION OWED TO THE USER (not applied here): ARC-029 "
                     "converts to `substrate_conditional` on commitment-occupancy "
                     "sustainment, per its OWN P1 text -- 'If P1 still fails with those "
                     "armed, ARC-029 converts to substrate_conditional on "
                     "commitment-occupancy sustainment and this falsifier is not "
                     "readable.' The mechanism is the conserved health budget: "
                     "episode_length x |mean harm/step| ~ 1, hard-coded at "
                     "agent_health = 1.0 with no constructor parameter.")

    return {
        "label": label, "routes_to": routes_to,
        "gate_green": gate_green, "preconditions": preconditions,
        "rung_table": rung_table, "feasible_rungs": feasible_rungs,
        "n_jointly_feasible_cells": sum(1 for r in rows if r["jointly_feasible"]),
        "n_p1_feasible_cells": sum(1 for r in rows if r["p1_feasible"]),
        "n_p3_feasible_cells": sum(1 for r in rows if r["p3_feasible"]),
        "max_bar_in_force_fraction": max_bar,
        "n_cells_window_filled": n_window_filled,
        "ladder_episode_length_range_ratio": ladder_range,
        "train_tick_spread": tick_spread,
        "ladder_non_degenerate": ladder_non_degenerate,
        "lineage_control_occupancy": l0_occ,
        "rv_residual_dispersion_by_rung": {
            t["rung"]: t["rv_residual_dispersion_sd"] for t in rung_table},
        "health_budget_product_by_rung": {
            t["rung"]: t["health_budget_product"] for t in rung_table},
        "source_scan": src_scan,
        "dropped_rungs_budget_infeasible": dropped_rungs,
        "n_rungs_retained": len(rung_ids),
        "pilots": pilots or [],
    }


def run(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:1] if dry_run else SEEDS
    ladder = LADDER[:2] if dry_run else LADDER
    qs = Q_LADDER[:1] if dry_run else Q_LADDER
    steps = 24 if dry_run else STEPS_PER_EPISODE
    warm_cap = 3 if dry_run else WARM_EPISODE_CAP
    meas_env_target = 60 if dry_run else MEAS_ENV_STEP_TARGET
    meas_select_min = 8 if dry_run else MEAS_SELECT_MIN
    meas_cap = 3 if dry_run else MEAS_EPISODE_CAP

    src_scan = _assert_no_running_variance_writes()
    rows: List[Dict[str, Any]] = []
    train_stats: List[Dict[str, Any]] = []
    pilots: List[Dict[str, Any]] = []

    for rung, nh, hh, ph in ladder:
        for seed in seeds:
            # One trained agent per (rung, seed). q is mutated between cells --
            # it is read live by the bar (`self.config.commit_threshold_quantile`)
            # and does NOT change the window's maxlen or anything set at
            # construction, so re-training per q would buy nothing and cost 3x.
            # The window is cleared and re-warmed per cell regardless.
            # PILOT FIRST, then the real agent. The pilot trains a throwaway agent
            # and so leaves the global torch/numpy streams in a (deterministic but)
            # pilot-dependent state; building the real agent AFTER it means the real
            # run always starts from `seeded_construct`'s own reset_all_rng(seed),
            # exactly as V3-EXQ-1070's did. The reverse order silently confounds any
            # per-cell comparison against 1070 at the same seed with the budget fix.
            # CLOSED-LOOP tick budgeting (the V3-EXQ-1070 repair): measure THIS
            # (rung, seed)'s realised ticks per episode with the real trainer, then
            # size the single real training call from that. See the module docstring
            # for why a chunked tick cap is NOT used.
            # The pilot RUNS under --dry-run too, at 1 episode per phase. Short-
            # circuiting it there would leave the one code path this iteration
            # exists to add completely un-smoked.
            pilot = _pilot_ticks_per_episode(
                rung, nh, hh, ph, seed, steps,
                episodes=1 if dry_run else PILOT_EPISODES,
                p0_tick_target=12 if dry_run else TARGET_P0_TICKS)
            train_env = _make_env(seed, nh, hh, ph, TRAIN_DRIFT)
            agent, cfg_slice = seeded_construct(
                seed, lambda: _build_agent(train_env, Q_LADDER[len(Q_LADDER) // 2], rung))
            rv_before = float(agent.e3._running_variance)
            zw_env = _make_env(seed, nh, hh, ph, TRAIN_DRIFT)  # P0a consumes env RNG
            budget = _episode_budgets_from_pilot(rung, pilot, dry_run)
            pilots.append({**pilot, **{k: v for k, v in budget.items()}})
            p0, p1, zp0 = budget["p0"], budget["p1"], budget["zp0"]
            print(f"  [budget] {rung} seed={seed} pilot_p0_len="
                  f"{pilot['pilot_p0_ticks_per_episode']:.2f} pilot_p1_len="
                  f"{pilot['pilot_p1_ticks_per_episode']:.2f} -> p0={p0} p1={p1} "
                  f"zp0={zp0} infeasible={budget['budget_infeasible']}", flush=True)
            stats = _train_all_on_agent(
                agent, train_env, seed=seed, p0_episodes=p0, p1_episodes=p1,
                steps_per_episode=steps, rung_id=f"{rung}_s{seed}",
                total_denominator=EPISODES_PER_RUN if not dry_run else max(1, p0 + p1),
                # SD-070: without this the world encoder is never stepped and
                # z_world stays a frozen random projection, so rv never collapses.
                zworld_p0_episodes=zp0,
                zworld_p0_env=(zw_env if zp0 > 0 else None),
            )
            rv_after = float(agent.e3._running_variance)
            collapse = (math.log10(rv_before / rv_after)
                        if (rv_before > 0 and rv_after > 0) else None)
            train_stats.append({"rung": rung, "seed": seed,
                                "budget_infeasible": budget["budget_infeasible"],
                                "budget_infeasible_reason": budget["budget_infeasible_reason"],
                                "episodes_p0": p0, "episodes_p1": p1, "episodes_zp0": zp0,
                                "pilot_p0_ticks_per_episode":
                                    pilot["pilot_p0_ticks_per_episode"],
                                "pilot_p1_ticks_per_episode":
                                    pilot["pilot_p1_ticks_per_episode"],
                                "running_variance_before": rv_before,
                                "running_variance_after": rv_after,
                                "rv_collapse_decades": collapse,
                                "n_p0_ticks": stats.get("n_p0_ticks"),
                                "n_p1_ticks": stats.get("n_p1_ticks")})

            # COUNTERBALANCE q ORDER BY SEED (red-team F6). The three q cells share
            # one trained agent, and `agent.reset()` deliberately does NOT clear the
            # residue field, so a fixed q order would confound q with position in the
            # residue history. Feasibility is an any-q predicate so this does not move
            # the deliverable, but it does contaminate the per-q rows a reader would
            # use to pick an operating q. The realised order is recorded per cell.
            q_order = tuple(qs) if (seeds.index(seed) % 2 == 0) else tuple(reversed(qs))
            for q in q_order:
                label = f"{rung}_q{q}_seed{seed}"
                print(f"Seed {seed} Condition {rung}_q{q}", flush=True)
                row = _run_cell(agent, seed, rung, nh, hh, ph, q, cfg_slice,
                                steps, warm_cap, meas_env_target, meas_select_min,
                                meas_cap, label)
                row["rv_collapse_decades"] = collapse
                row["q_order_this_seed"] = list(q_order)
                row["budget_infeasible"] = budget["budget_infeasible"]
                row["train_ticks_total"] = int((stats.get("n_p0_ticks") or 0)
                                               + (stats.get("n_p1_ticks") or 0))
                rows.append(row)
                readable = bool(row["n_select_calls"] > 0 and row["window_filled"])
                print(f"  [cell] {label} ep_len={row['mean_episode_length']} "
                      f"sel/ep={row['select_calls_per_episode']} "
                      f"bar_in_force={row['bar_in_force_fraction']} "
                      f"occ={row['committed_step_fraction']} "
                      f"runlen={row['mean_committed_run_length']} "
                      f"snr={row['harm_snr']}", flush=True)
                print(f"verdict: {'PASS' if readable else 'FAIL'}", flush=True)

    s = analyse(rows, src_scan, pilots)
    # A DIAGNOSTIC's outcome is about whether it MEASURED, not about ARC-029.
    outcome = "PASS" if (s["gate_green"] and s["ladder_non_degenerate"]) else "FAIL"

    criteria = [
        {"name": "D1_ladder_swept_episode_length", "load_bearing": True,
         "passed": bool(s["ladder_non_degenerate"]),
         "measured": s["ladder_episode_length_range_ratio"], "threshold": 2.0,
         "direction": "lower",
         "note": ("max/min mean episode length across rungs. If the ladder did not "
                  "move the quantity it manipulates, 'no feasible region' is a "
                  "statement about a sweep that never swept.")},
        {"name": "D2_bar_confirmed_in_force", "load_bearing": True,
         "passed": bool(s["max_bar_in_force_fraction"] >= BAR_IN_FORCE_FLOOR),
         "measured": s["max_bar_in_force_fraction"], "threshold": BAR_IN_FORCE_FLOOR,
         "direction": "lower"},
        {"name": "D3_joint_feasible_region_found", "load_bearing": False,
         "passed": bool(s["feasible_rungs"]),
         "measured": len(s["feasible_rungs"]), "threshold": 1, "direction": "lower",
         "note": ("REPORTED, NOT GATING: both outcomes are informative and each routes "
                  "to a different user decision. A 0 here is the substrate_conditional "
                  "candidate, not a failed run.")},
        {"name": "D4_lineage_control_reproduces_saturation", "load_bearing": False,
         "passed": bool(s["lineage_control_occupancy"]),
         "measured": len(s["lineage_control_occupancy"]), "threshold": 1,
         "direction": "lower",
         "note": ("L0 is the ARC-029 lineage point and the negative control. Its "
                  "occupancy is REPORTED for comparison against the predecessor's "
                  "measured 1.0000; it does not gate.")},
    ]

    readout: Dict[str, float] = {}
    for key, value in (
        ("n_jointly_feasible_cells", s["n_jointly_feasible_cells"]),
        ("n_p1_feasible_cells", s["n_p1_feasible_cells"]),
        ("n_p3_feasible_cells", s["n_p3_feasible_cells"]),
        ("n_feasible_rungs", len(s["feasible_rungs"])),
        ("max_bar_in_force_fraction", s["max_bar_in_force_fraction"]),
        ("n_cells_window_filled", s["n_cells_window_filled"]),
        ("ladder_episode_length_range_ratio", s["ladder_episode_length_range_ratio"]),
        ("gate_green", 1 if s["gate_green"] else 0),
        ("ladder_non_degenerate", 1 if s["ladder_non_degenerate"] else 0),
        ("n_cells", len(rows)),
        ("train_tick_spread", s["train_tick_spread"]),
    ):
        coerced = _flat_scalar(value)
        if coerced is not None:
            readout[key] = coerced

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        # A diagnostic returns NO claim verdict, by user instruction. It reports a
        # feasibility region and the disposition the user must then decide on.
        "evidence_direction": "unknown",
        "outcome": outcome,
        "timestamp_utc": _utc_stamp(),
        "non_degenerate": bool(s["gate_green"] and s["ladder_non_degenerate"]),
        "degeneracy_reason": (
            None if (s["gate_green"] and s["ladder_non_degenerate"]) else
            "; ".join([p["name"] for p in s["preconditions"] if not p["met"]]
                      + ([] if s["ladder_non_degenerate"] else ["ladder_did_not_sweep"])
                      + ([] if not s["dropped_rungs_budget_infeasible"] else
                         ["rungs_dropped_budget_infeasible="
                          + ",".join(s["dropped_rungs_budget_infeasible"])]))),
        "arm_results": rows,
        "training_stats": train_stats,
        "budget_pilots": pilots,
        "criteria": criteria,
        "criteria_non_degenerate": {
            "D1_ladder_swept_episode_length": bool(len(rows) >= 2),
            "D2_bar_confirmed_in_force": bool(s["n_cells_window_filled"] >= 1),
            "D3_joint_feasible_region_found": bool(s["gate_green"]
                                                   and s["ladder_non_degenerate"]),
            "D4_lineage_control_reproduces_saturation": bool(s["lineage_control_occupancy"]),
        },
        "combination_rule": (
            "outcome PASS requires the readiness gate green AND D1 (the ladder actually "
            "moved episode length by >= 2x). D2 is a readiness criterion and also a "
            "precondition. D3 is the DELIVERABLE and is deliberately NOT gating -- both "
            "of its outcomes are informative and each routes to a different user "
            "decision. D4 is the negative control, reported."),
        "interpretation": {
            "label": s["label"],
            "preconditions": s["preconditions"],
            "criteria_non_degenerate": {
                "D1_ladder_swept_episode_length": bool(len(rows) >= 2),
                "D2_bar_confirmed_in_force": bool(s["n_cells_window_filled"] >= 1),
                "D3_joint_feasible_region_found": bool(s["gate_green"]
                                                       and s["ladder_non_degenerate"]),
            },
            "routes_to": s["routes_to"],
        },
        "disposition_rule_preregistered": {
            "no_joint_feasible_region": (
                "ARC-029 converts to substrate_conditional on commitment-occupancy "
                "sustainment, per ARC-029's own P1 text. RECORDED, NOT APPLIED -- goes "
                "to the user as a decision with the numbers."),
            "region_found": (
                "the pre-registered V3-EXQ-063b design "
                "(experiments/v3_exq_1066_arc029_commitment_mode_harm_variance_bar.py) "
                "is re-queued at that setting, with C1 re-expressed if needed. Also a "
                "user decision, also not taken here."),
            "applied_by_this_run": False,
        },
        "readout": readout,
        "summary": s,
    }
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t_start = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    manifest = run(dry_run=args.dry_run)

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run,
        config={
            "base_env": BASE_ENV, "train_drift": TRAIN_DRIFT, "eval_drift": EVAL_DRIFT,
            "ladder": [list(r) for r in LADDER], "q_ladder": list(Q_LADDER),
            "commit_threshold_quantile_window": COMMIT_WINDOW,
            "e3_steps_per_tick": E3_STEPS_PER_TICK,
            "precision_ema_alpha": PRECISION_EMA_ALPHA,
            "steps_per_episode": STEPS_PER_EPISODE,
            "target_p0_ticks": TARGET_P0_TICKS,
            "target_p1_ticks": TARGET_P1_TICKS,
            "target_zworld_p0_ticks": TARGET_ZWORLD_P0_TICKS,
            # random-policy table, used ONLY for the zworld P0a phase (which rolls
            # out under RandomPolicy); every other phase is pilot-budgeted per
            # (rung, seed) at run time and recorded in `budget_pilots`
            "ladder_episode_lengths_premeasured_randompolicy": LADDER_EP_LEN,
            "pilot_episodes_per_phase": PILOT_EPISODES,
            "max_train_episodes_per_phase": MAX_TRAIN_EPISODES_PER_PHASE,
            "train_tick_spread_ceiling": TRAIN_TICK_SPREAD_CEILING,
            "episodes_per_run": EPISODES_PER_RUN,
            "warm_episode_cap": WARM_EPISODE_CAP,
            "meas_env_step_target": MEAS_ENV_STEP_TARGET,
            "meas_select_min": MEAS_SELECT_MIN,
            "meas_episode_cap": MEAS_EPISODE_CAP,
            "p3_abs_harm_floor": P3_ABS_HARM_FLOOR,
            "c1_abs_floor": C1_ABS_FLOOR,
            "p1_occupancy_band": list(P1_OCCUPANCY_BAND),
            "p1_runlen_floor": P1_RUNLEN_FLOOR,
            "bar_in_force_floor": BAR_IN_FORCE_FLOOR,
            "p3_snr_floor": P3_SNR_FLOOR,
        },
        seeds=SEEDS, script_path=Path(__file__), started_at=_t_start,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"feasible_rungs: {manifest['summary']['feasible_rungs']}", flush=True)

    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
