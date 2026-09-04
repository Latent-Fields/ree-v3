#!/opt/local/bin/python3
"""
V3-EXQ-983a -- EXT-002: does the residue field leave a persistent error record?
(headroom-derived bar + training-completion gate; supersedes V3-EXQ-983)

Claim: EXT-002 (external_failure_mode, subject llm.hallucination) -- "Hallucination:
no persistent error residue accumulates to shape future outputs." Proposal: EXP-0526
(backlog_id EVB-1247). Rider actually exercised: ARC-013 (residue is persistent
latent-space curvature) -- see RIDER TAGGING below for why ARC-005 / INV-008, the
riders EXT-002's own `ree_mechanism` field names, are deliberately NOT tagged.
ree_failure_mode_analog: moral_amnesia.

WHY THIS RE-RUN EXISTS (read this before changing any threshold)
-----------------------------------------------------------------
V3-EXQ-983 ran 9.4h on ree-cloud-4 and was adjudicated **non_contributory** by
`failure_autopsy_ext-claim-probe-cluster_2026-09-03.md` (confirmed; ratified by
governance-20260903T2013 with a red-team pass). All nine of its preconditions
passed. Two defects, both measurement, neither a substrate ceiling:

  (1) THE DV HAD NO ROOM. `decline_gap` realised a range of 0.0468 across all six
      cells against a C1 threshold of 0.15 -- 3.2x short. An effect of the
      registered size could not have been observed under any outcome.
  (2) THE SEED POPULATION WAS UNGATED. `steps_realized_frac` was 0.079 / 0.077 /
      0.965 on seeds 42 / 123 / 456 (a ~12x spread in realised training) and
      `harm_rate_train` 0.7296 / 0.7427 / 0.0236 (~31x). Seeds 42 and 123 died to
      `agent_health <= 0` inside ~16 steps of every episode and never trained;
      equal-weight pooling across that population produced the -0.45pp
      `decline_gap`.

WEAK-NULL RESIDUAL CARRIED, NOT ERASED (governance red-team amendment 4)
--------------------------------------------------------------------------
The downgrade to `non_contributory` does **not** mean V3-EXQ-983 measured nothing,
and this redesign does not treat its predecessor as unmeasured. Recorded verbatim
on EXT-002's `evidence_quality_note` and carried here as the prior this run tests
against:

  * The P3 positive control PASSED at a 51x margin (`e1_prediction_error_min`
    0.005121 against a 1e-4 floor): the substrate's world-model error signal was
    demonstrably non-degenerate.
  * The manipulation REACHED BEHAVIOUR: same-seed A0-vs-A1 executed-action-stream
    divergence 0.186 / 0.237 / 0.295. Freezing the residue field changed what the
    agent did.
  * And `decline_gap` still did not move (-0.0045, effect_sd -0.232).

So the prior going in is a WEAK NULL on the rider prediction, not an absence of
evidence. If this run also returns a null with the DV demonstrably in range and a
trained seed population, that is a *second, stronger* null and should be read as
such -- see "WHAT A NULL MEANS HERE" below. This experiment is powered to tell
those two readings apart; its predecessor was not.

WHAT IS ACTUALLY UNDER TEST, AND WHY "ERROR" = HARM HERE (not literal E1 loss)
-------------------------------------------------------------------------------
Unchanged from V3-EXQ-983. The claim's prose uses "E1 prediction error" as the
general concept of "an unexpected, costly outcome that should leave a lasting
trace." `ree_core/agent.py`'s `update_residue()` accumulates the residue field on
`harm_signal < 0` -- on HARM, not on a raw E1 world-model loss value. So "error"
is operationalised as a harm event: the substrate's actual wired implementation of
"a costly outcome the agent's world-model did not want", keeping the DV causally
wired to the manipulated mechanism. The literal E1/E3 `prediction_error` signal is
used instead as the mandatory POSITIVE CONTROL (P3), on a throwaway agent isolated
from both scored arms.

DESIGN -- two arms, residue read identically, accumulation the manipulation
-----------------------------------------------------------------------------
  A0_INTACT           residue accumulates from every harm event and is consulted at
                      action-selection time (E3 argmin over residue-scored
                      candidates).
  A1_RESIDUE_FROZEN   residue field frozen at initialisation -- never accumulates --
                      but is READ via the identical E3 argmin path. **A1 is the
                      CONTROL ARM, and its realised per-seed `decline` values are
                      what the dv_headroom gate below measures.**

Both arms are otherwise IDENTICAL: same env, same E1/E2 training, same CEM candidate
budget, same action-selection code path. The only difference is whether harm events
ever get written into `agent.residue_field`.

OPERATIONAL DEFINITION OF "THE SAME ERROR RECURRING" (unchanged from 983)
--------------------------------------------------------------------------
    key = (pre_action_grid_x, pre_action_grid_y, executed_action_class)

captured BEFORE `env.step()` and paired with the REALISED action class from
`agent.select_action`'s E3 output. `first_harm_at[key]` is the first step index at
which a step at `key` produced `harm_signal < 0`; a REVISIT of `key` is scored as a
REPEAT if `harm_signal < 0` again. `repeat_error_rate(period)` is the mean repeat
indicator over revisits in that period (EARLY / LATE halves of the REALISED step
budget). `decline := repeat_error_rate(early) - repeat_error_rate(late)`; positive =
pressure observed. `decline_gap := mean(decline_A0) - mean(decline_A1)` over POOLED
seeds only (see the completion gate).

WHAT CHANGED FROM V3-EXQ-983 -- four changes, each answering one autopsy finding
---------------------------------------------------------------------------------
1. **C1 IS NOW HEADROOM-DERIVED AND HEADROOM-GATED (autopsy finding 1).**
   `THRESH_C1_DECLINE_GAP` moves 0.15 -> 0.04. The derivation is the predecessor's
   OWN measured data, not a guess: A1_RESIDUE_FROZEN (the control arm) realised
   per-seed `decline` values of [0.005309, 0.029011, -0.017779], a RANGE of
   **0.046790** -- the exact 0.0468 figure the autopsy table cites. 0.04 sits
   INSIDE that demonstrated range (85.5% of it), so the bar is satisfiable on
   evidence already in hand rather than on an argument. Cross-check against noise:
   the predecessor's cross-seed paired-delta SD was 0.019321, so 0.04 is 2.07 SD --
   a bar above the seed-to-seed noise floor, not at it. C2 (>= 0.80 SD) therefore
   requires only 0.01546 and C1 remains the binding, load-bearing criterion.

   AND the bar is now GATED at runtime by the `dv_headroom` precondition kind
   landed in `experiments/_metrics.py` on ree-v3 8e133d26ed (substrate_queue entry
   `dv-dynamic-range-precondition-class`, minted by this same autopsy). The gate
   measures THIS run's control arm (`statistic="range"` over A1's pooled per-seed
   `decline` values -- `_metrics.dv_achievable`'s docstring names 983 as the
   worked case for exactly this statistic) and requires it to reach
   `THRESH_C1_DECLINE_GAP * DV_HEADROOM_MARGIN`. Below that the run self-routes to
   `substrate_not_ready_requeue` and NEVER to a verdict on EXT-002. `margin` is
   1.0 -- bare feasibility -- deliberately: a margin above 1.0 would require more
   headroom than the predecessor's own control arm demonstrated, i.e. it would
   re-create the unsatisfiable-on-known-data defect this campaign exists to kill.

2. **A TRAINING-COMPLETION GATE NOW RUNS BEFORE SEEDS ARE POOLED (autopsy
   finding 2).** A seed contributes to `decline_gap` only if BOTH its arms clear a
   declared band:
       steps_realized_frac >= 0.60   AND   harm_rate_train <= 0.35
   The seed is the unit because the statistic is PAIRED; admitting one arm of a
   seed and not the other would compare a trained A0 against an untrained A1.
   Band derivation, again from the predecessor's own cells: the trained seed (456)
   read 0.965 / 0.9691 and 0.0236 / 0.0236; the two untrained seeds read
   0.0792 / 0.0769 / 0.0770 / 0.0854 and 0.7296 / 0.7518 / 0.7427 / 0.6751. The
   0.60 floor and 0.35 ceiling each sit in the empty gap between those two
   populations, so the gate separates them decisively and is not tuned to a
   boundary case. Excluded seeds are RECORDED IN FULL (`completion_gate.per_seed`,
   and their rows stay in `arm_results`) -- they are excluded from POOLING, never
   deleted.

3. **THE SEED POPULATION IS 8, NOT 3, AT THE SAME TOTAL COMPUTE.** 8 seeds x 2
   arms x 110 warmup episodes x 200 steps = 352,000 worst-case planned steps,
   2.2% UNDER the predecessor's 3 x 2 x 300 x 200 = 360,000 -- eight independent
   draws of the board lottery instead of three, for no extra compute. Only 1 of 3
   predecessor seeds trained; `MIN_POOLED_SEEDS = 2` is the readiness floor (P6)
   and 8 draws at the predecessor's observed ~1/3 survival rate expects ~2.7.
   Shortening the per-cell budget does not change the survival rate: the dying
   seeds died inside ~16 steps of EVERY episode (4755 realised steps over 300
   episodes), so their `steps_realized_frac` is invariant to the episode count.
   Nor does it starve the DV: the trained seed produced 437 revisit events over
   57,900 steps, so 22,000 steps yields ~166 (~83 per half) against C4's floor of
   20.

4. **P5 (harm_rate_not_saturated) IS REPLACED BY P6 (pooled_seed_count).** P5's
   subject -- a saturated harm source -- is now handled per seed by the completion
   gate's 0.35 ceiling, which is strictly stricter than P5's 0.90. Retaining P5
   over the POOLED rows would make it tautological (it can never fail once the
   0.35 band has run), and a tautological precondition is exactly the red-team
   Family-4 defect the predecessor already had to fix once. P6 asks the question
   P5 can no longer ask: did ENOUGH seed-pairs survive the completion gate to
   support a paired comparison at all?

WHAT A NULL MEANS HERE, AND WHAT IT DOES NOT (mandatory null declaration)
---------------------------------------------------------------------------
A FAIL on C1 with all preconditions green and the dv_headroom gate MET means: on
this substrate, in this environment, with a demonstrably in-range DV and a trained
seed population, freezing the residue field did not measurably raise the rate at
which the agent repeats a known-costly (cell, action) pairing. Combined with the
predecessor's weak null (positive control 51x, manipulation reaching behaviour at
0.19-0.30 action divergence), that would be the SECOND null on the same prediction
and the first one powered to see the registered effect -- material evidence
against the EXT-002 rider as operationalised here, and a `weakens` direction.

It does NOT mean: (a) that residue leaves no trace at all -- P1 measures a live,
structured field and the manipulation demonstrably reaches behaviour; (b) that
INV-006's non-erasability invariant is false -- that claim is `derivational` and
takes no experimental evidence; or (c) that a different operationalisation of
"the same error recurring" (a residue-gradient-conditioned choice measure, a
cross-context transfer measure) would also return null. The scope is this DV.

A FAIL on the dv_headroom gate or on P6 means NOTHING about EXT-002 at all: the
run self-routes `substrate_not_ready_requeue` with direction `non_contributory`.

RIDER TAGGING -- a deliberate divergence from the chip text, recorded here
----------------------------------------------------------------------------
Governance asked that redesigns tag "the rider claim actually exercised (ARC-005
and/or INV-008)". Read against the substrate, neither is:

  ARC-005  "Control plane routes precision and modes."   Freezing
           `agent.residue_field` touches neither precision routing nor mode
           selection. Nothing in this design manipulates the control plane.
  INV-008  "Precision is routed and depth-specific, not global."   Same: no
           precision term is manipulated or read.
  INV-006  "Post-commit consequence traces cannot be erased, only integrated."
           This IS the semantic content of the claim under test, but it carries
           `epistemic_category: derivational` and its own notes record it as a
           "Resolved universal ... a non-erasability property of any agent in any
           world with persistent state", explicitly "mechanism-agnostic ... residue
           geometry remains one candidate architectural realization". A
           derivational universal does not take experimental confirmation from a
           single substrate.
  ARC-013  "Residue is persistent latent-space curvature; hippocampal paths form a
           cognitive map."   THIS is what the A1 manipulation ablates and what the
           E3 argmin over `residue_field.evaluate_trajectory` reads. It is
           `architectural_commitment`, status active, and is already EXT-004's
           named rider for the same residue mechanism.

So `claim_ids = ["EXT-002", "ARC-013"]`, per the CRITICAL claim_ids accuracy rule
("Tag only the claims the experiment directly tests with its actual
implementation... Erroneous tags corrupt governance confidence scores"). Flagged to
governance for ratification; if governance prefers the ARC-005/INV-008 tags, that
is a one-line change here, but it should be made deliberately and not inherited.

DV-SYMMETRY DECLARATION (mandatory; one line per arm) -- unchanged from 983
-----------------------------------------------------------------------------
DV = `decline`, a function of the harm-event sequence at matched (cell, action)
keys, itself a function of the executed action stream, produced by an E3 ARGMIN
over per-candidate residue trajectory scores. Symmetry group: (i) permutation of
candidate INDEX order, (ii) any uniform additive constant applied to all candidate
scores, (iii) any monotone rescaling of all candidate scores -- none can move an
argmin.
  A0  residue accumulates at a NEW RBF center per harm event (magnitude
      `abs(harm_signal) * accumulation_rate`, `ree_core/residue/field.py:665`), so
      later candidate-trajectory scores differ from earlier ones by an amount
      depending on which regions were actually harmed -- neither a broadcast
      constant (candidate trajectories pass through different z_world regions) nor
      a monotone map of the earlier score vector. NOT invariant.
  A1  freezes the field at init, so every step's candidate scores are computed
      against the SAME harm-history-blind field. The residual per-candidate spread
      from the untrained `neural_field` term is not a broadcast constant or a
      monotone rescaling either, so A1's argmin is not a constant function -- but
      it carries no memory of PAST harm, which is exactly the manipulation. NOT
      invariant, and not the mechanism the claim asserts.

NON-DEGENERACY PRECONDITIONS (breach -> substrate_not_ready_requeue, NOT a verdict)
------------------------------------------------------------------------------------
  P1 residue_structure_live      A0 only, POOLED rows: cross-center weight variance
                                 of the trained residue field > 1e-6. Scoped OUT of
                                 A1 (frozen BY DESIGN).
  P2 candidate_score_range       Both arms, POOLED rows: cross-candidate residue-
                                 score RANGE on a post-training probe -- the SAME
                                 statistic the E3 argmin routes on (V3-EXQ-643).
  P3 e1_prediction_error_floor   Global: E3 world-model prediction error on a
                                 genuine hazard collision, on a THROWAWAY agent.
                                 Passed at 51x in the predecessor.
  P4 revisit_denominator         Both arms, POOLED rows: minimum EARLY-half
                                 (cell,action) revisit count. Deliberately
                                 decoupled from C4 (red-team Family 4 fix, kept).
  P6 pooled_seed_count           NEW. Number of seed-pairs surviving the training-
                                 completion gate must reach MIN_POOLED_SEEDS.
                                 Replaces the predecessor's P5 (see change 4).
  DV dv_headroom_decline_gap     NEW, kind `dv_headroom`, evaluated by
                                 `_metrics.p0_readiness_gate` AFTER the cells run
                                 (a control arm's realised range is necessarily a
                                 post-training measurement). Merged into
                                 interpretation.preconditions alongside the per-arm
                                 entries.

Per-arm gates use experiments/_lib/precondition_gate.py. As in V3-EXQ-983 the
primary statistic is a PAIRED comparison and is only interpretable when BOTH arms
are green, so this script uses `gate["all_green"]`, not the module default
`any_green`; `per_arm_gate` is still recorded verbatim.

PRE-REGISTERED THRESHOLDS (constants below; never inferred post-hoc)
----------------------------------------------------------------------
  C1 (LOAD-BEARING) decline_gap >= 0.04    -- derived above from the predecessor's
     control-arm range 0.046790, and runtime-gated by dv_headroom.
  C2 effect >= 0.80 SD of the cross-seed paired delta.
  C3 direction consistent on >= 2 pooled seeds.
  C4 data quality: min revisit count over BOTH halves, both arms, POOLED rows
     >= 20. Stricter than, and decoupled from, P4.

SLEEP: not used (no sleep flags set) -- no SLEEP DRIVER line required.

RED-TEAM (Step 4.5, fable, 2026-09-04): CONTESTED -> six findings, each with a
written disposition. Every claim below was re-verified against source by the
authoring session before acting; two were confirmed by direct measurement.

  F1 (CONFIRMED BY MEASUREMENT, and the most serious) -- THE DV WAS CONDITIONED ON
     AN ENV-DETERMINED EVENT. `CausalGridWorldV2.reset()` re-places every hazard
     from the seeded RNG on every episode. Measured on this exact config: three
     successive resets gave hazard sets [(1,1),(2,1),(2,2),(3,3)],
     [(1,2),(2,2),(3,1),(4,4)], [(1,1),(1,4),(2,4),(3,1)]. So under V3-EXQ-983's
     build-once-then-reset loop a key entered in `first_harm_at` at episode 3 was
     scored against episode 50's entirely different layout, and `P(harm | revisit)`
     was largely the env's per-episode hazard lottery -- which no residue field can
     anticipate. That is a construct defect in the very experiment meant to repair
     a measurement defect, and it plausibly explains the predecessor's near-ceiling
     repeat rates (0.864-0.993) better than any account of the agent's behaviour.
     FIXED, two ways: (a) the env is rebuilt from the SAME seed at the top of every
     episode, which pins both the hazard layout and the start position (verified
     identical across fresh builds), and `env_drift_prob` goes 0.1 -> 0.0 so the
     layout does not move within an episode either; (b) a NEW precondition P7
     `revisit_outcome_heterogeneity` gates on the red-team's own proposed
     confirmer -- at least 10% of erred keys must have resolved BOTH ways across
     their revisits. If every revisited key gives the same answer every time the DV
     has no agent-side degree of freedom, and the run self-routes instead of
     reporting an unattributable null.
  F4-A (ACCEPTED) -- THE HEADROOM GATE MEASURED THE CONTROL ARM'S NOISE. The first
     draft took `range` over A1's per-seed declines alone. A real effect that is
     CONSISTENT across seeds produces a TIGHT control arm, so that gate is
     anti-correlated with the runs it should protect and would refuse the
     best-powered ones. FIXED: the range is measured over EVERY pooled cell, both
     arms -- which is also exactly what the cluster autopsy computed ("realised
     range of 0.0468 across all six cells"). A real effect now WIDENS the measured
     range, while a DV pinned in both arms (the V3-EXQ-983 pathology) is still
     refused.
  F4-B (CONFIRMED BY MEASUREMENT) -- OFF-BY-ONE AT THE FLOOR'S OWN BOUNDARY.
     `PreconditionSpec.met_for` is STRICT (`measured > threshold`,
     _lib/precondition_gate.py:199) while the REE_assembly indexer recomputes the
     same entry INCLUSIVELY, so an integer count sitting exactly on an integer
     threshold reads UNMET in the script and MET in the indexer -- the manifest
     disagreeing with itself precisely at the boundary the floor names. Verified:
     `met_for(2.0)` against `threshold=2.0` is False; against 1.5 it is True.
     FIXED: both integer-valued floors are now half-integers (P4 4.5 for ">= 5",
     P6 `MIN_POOLED_SEEDS - 0.5`), which no integer measurement can land on, so
     both readings agree everywhere.
  F2 (ACCEPTED) -- C4's FAILURE SIGNATURE IS THE SUCCESS SIGNATURE. A working
     residue effect suppresses late revisits of costly keys, which is what drives
     the late-half count under C4's floor -- and the first draft routed a C4 miss
     to `weakens`, recording the claim's own confirmation as evidence against it.
     This is the identical self-defeating structure the predecessor's red-team
     already had to fix once for P4, left standing on C4. FIXED: a C4 failure now
     self-routes to `substrate_not_ready_requeue` (it is a DATA-QUALITY control,
     not a scientific criterion), with a note that the requeue should raise the
     step budget rather than lower the floor.
  F3 (ACCEPTED) -- ARC-013 INHERITED `weakens` FROM A DV THAT DOES NOT BEAR ON IT.
     P1, in the same run, is direct evidence FOR ARC-013's persistence property.
     FIXED: `evidence_direction_per_claim` now branches -- on a null EXT-002 takes
     `weakens` and ARC-013 takes `mixed`.
  F4-C (ACCEPTED AS A DIAGNOSTIC, gate unchanged) -- THE COMPLETION GATE READS TWO
     STATISTICS THE MANIPULATION COULD ITSELF MOVE, and requires both arms of a
     seed in band, so an A0-survives/A1-dies seed -- the strongest possible
     evidence for the claim -- is discarded. The gate is nonetheless correct
     (comparing a trained A0 against an untrained A1 is exactly the artifact the
     predecessor produced), and the predecessor's own cells show no arm asymmetry
     on these statistics (0.7296/0.7518, 0.7427/0.6751, 0.0236/0.0236), so the bias
     is latent rather than demonstrated. RECORDED, not silenced:
     `completion_gate.exclusion_asymmetry` reports every seed excluded because
     exactly one arm cleared the band, and which arm survived. A run whose
     exclusions are systematically one-armed is itself a finding.
  F2-power (PARTLY ACCEPTED, stated rather than fixed) -- the reviewer notes C1 has
     ~50% power at an effect exactly equal to its own bar. That is true of any bar
     at any n and is not a defect; what IS worth stating plainly is that 0.04 is
     the largest bar demonstrably inside the DV's realised dynamic range, NOT an
     effect-size prior. This run is powered to resolve roughly a 2-SD effect
     against the predecessor's cross-seed paired SD of 0.0193, and says so here
     rather than claiming more.

One pass only, per the skill's "do NOT iterate to CLEAR" rule. F1 did change the
causal chain, but the verdict was CONTESTED rather than BLOCKING and every finding
above is either fixed or dismissed in writing with a source citation, so no second
spawn was made.
Findings inherited from V3-EXQ-983's own red-team pass and RETAINED unchanged:
contamination_spread=0.0 (2a-i), realised-step early/late split (2a-ii), the
recorded-not-gated action-stream divergence (Family 1), and the P4/C4 decoupling
(Family 4).
"""

import argparse
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import HippocampalConfig, REEConfig

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments._lib.manifest_core import stamp_recording_core
from _metrics import (
    P0NotReady,
    dv_achievable,
    dv_headroom_check,
    p0_readiness_gate,
)
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)

EXPERIMENT_TYPE = "v3_exq_983a_ext002_residue_error_persistence_headroom"
SUPERSEDES = "V3-EXQ-983"
# ARC-013 (not ARC-005 / INV-008) is the rider this design actually manipulates --
# see RIDER TAGGING in the module docstring for the full reasoning and the
# governance divergence it records.
CLAIM_IDS = ["EXT-002", "ARC-013"]
EXPERIMENT_PURPOSE = "evidence"
BACKLOG_ID = "EVB-1247"

# Action-object-selection gate (`validate_experiments.action_object_selection_lint`)
# does not apply: this script never calls `action_object_decoder` at all. Every
# executed action comes from `agent.select_action(candidates, ticks)` (E3's J(zeta)),
# per `_select_action` below -- the substrate path V3-EXQ-800 established as the only
# one that lets a residue manipulation reach behaviour on this substrate.

# Hold-weighted-E3-readout gate (`validate_experiments.e3_hold_weighted_readout_lint`).
# DISCHARGED BY A TICK GUARD, not by a blanket marker. This file gates its
# repeat-error accumulation on `ticks["e3_tick"]` in `run_cell` (the `is_fresh_select`
# gate, whose counters are emitted as `n_fresh_select` / `n_latched` /
# `fresh_select_yield`) -- one of the discharge paths that lint names for itself in
# its own FIX prescription and in the backlog counter's message. So the blanket
# opt-out marker this script originally carried was INERT -- the lint returned None on
# the tick guard whether or not the marker was present -- and an inert marker is
# exactly what `e3_exemption_backlog_lint` exists to surface, so it has been removed
# rather than left standing. Per-site triage of the three constructs the lint's own
# detector reaches in this file is preserved below.
#
# (1) `run_cell`'s (cell,action) repeat-error tracking -- GATED. Accumulates only on a
#     fresh E3 selection tick, precisely so a single commitment held across a
#     hazard-adjacent wall (agent pinned, cell unchanged for the whole hold) cannot be
#     double-counted as many identical revisits. This is a RATE, so hold weighting
#     would move it; see the `is_fresh_select` gate there.
#
# (2) `_probe_candidate_score_range`'s `executed.append(...)` -- SAFE BY CONSTRUCTION,
#     no gate needed. Copied unchanged from V3-EXQ-800's helper of the same name, which
#     carries the identical construct. That list's ONLY reader is
#     `float(len(set(executed)))` (`executed_action_classes`, fed into P2's readiness
#     gate). Set CARDINALITY is exactly invariant under hold-duration replication --
#     replicating a held class can neither add a class nor remove one -- so hold
#     weighting cannot move this statistic. No magnitude or distribution-shape quantity
#     (entropy, variance, histogram mass) is derived from it, and it feeds no C1-C4
#     criterion.
#
# (3) `run_cell`'s `current_episode_actions.append(action_idx)` -- SAFE, and correct at
#     env-step granularity by design. This stream is the REALISED behaviour (a held
#     action genuinely is the action taken at that step), and its only consumer,
#     `_action_stream_divergence`, compares the two arms POSITIONALLY at the same seed
#     and episode index. A held commitment replicates identically on both sides of that
#     comparison, so it can neither manufacture nor mask a divergence. The resulting
#     `family1_action_stream_divergence` is recorded-not-gated (no calibrated
#     threshold), feeds no C1-C4 criterion, and is no distribution-shape statistic.

ARM_INTACT = "A0_INTACT"
ARM_FROZEN = "A1_RESIDUE_FROZEN"
ARMS = [ARM_INTACT, ARM_FROZEN]

# --- pre-registered thresholds -------------------------------------------
# C1 RE-DERIVED FROM THE PREDECESSOR'S OWN MEASURED CONTROL ARM, not from an
# argument. V3-EXQ-983's A1_RESIDUE_FROZEN (control) realised per-seed `decline`
# values [0.0053094, 0.0290108, -0.0177789] -> RANGE 0.0467897, the exact figure
# the cluster autopsy tabulates against the old 0.15 bar (3.2x short). 0.04 sits
# inside that demonstrated range (85.5% of it) and at 2.07x the predecessor's
# cross-seed paired-delta SD (0.0193206), so it is above the seed-noise floor
# rather than at it. It is ALSO gated at runtime -- see DV_HEADROOM_* below.
THRESH_C1_DECLINE_GAP = 0.04     # percentage points (decline_A0 - decline_A1)
THRESH_C2_EFFECT_SD = 0.80       # effect in SD of the cross-seed paired delta
THRESH_C3_MIN_SEEDS = 2          # of 3
THRESH_C4_MIN_REVISITS = 20      # per period, worst case across POOLED arm/seed

# --- dv_headroom gate (ree-v3 8e133d26ed, substrate entry
#     `dv-dynamic-range-precondition-class`, minted by this run's own autopsy) ---
# The control arm is A1_RESIDUE_FROZEN and the statistic is "range": C1 reads a
# BETWEEN-ARM DIFFERENCE, which is exactly the case `_metrics.dv_achievable`'s
# docstring names for "range" (and it names V3-EXQ-983 as the worked example).
# MARGIN 1.0 asserts bare feasibility ON PURPOSE. A margin above 1.0 would demand
# more headroom than the predecessor's own control arm demonstrated (0.0467897 vs
# a would-be 0.08 at margin 2.0), i.e. it would re-create the
# unsatisfiable-on-known-data defect this whole redesign exists to remove.
DV_HEADROOM_MARGIN = 1.0
DV_HEADROOM_DV_NAME = "decline_gap"
DV_HEADROOM_CONTROL_ARM = "A1_RESIDUE_FROZEN"
# Recorded for the manifest so a later reader can check the bar against the number
# it was derived from without re-opening the predecessor's manifest.
PREDECESSOR_CONTROL_RANGE = 0.0467896501434286
PREDECESSOR_RUN_ID = "v3_exq_983_ext002_residue_error_persistence_20260903T150005Z_v3"

# --- training-completion gate (autopsy finding 2) --------------------------
# A seed contributes to the POOLED decline_gap only if BOTH its arms clear this
# band. The seed, not the cell, is the unit: decline_gap is a PAIRED statistic, so
# admitting a trained A0 against an untrained A1 would compare two populations.
# Both bounds sit in the empty gap between the predecessor's two seed populations:
#   trained   (seed 456): steps_realized_frac 0.9650 / 0.9691, harm_rate 0.0236 / 0.0236
#   untrained (42, 123):  steps_realized_frac 0.0769-0.0854, harm_rate 0.6751-0.7518
# so neither bound is tuned to a boundary case.
# P7 (NEW, red-team Family 1): the fraction of erred (cell,action) keys whose
# revisits are NOT all-identical in outcome. If every revisited key produces the
# same result every single time, `P(harm | revisit)` is fixed by the environment
# and carries no agent-side degree of freedom, so no residue effect could move it
# and a null would be unattributable. This is the red-team's own proposed cheap
# confirmer, promoted from a diagnostic to a gate: it is the check that makes the
# layout pin (see _make_env / run_cell) verifiable rather than merely intended.
FLOOR_REVISIT_HETEROGENEITY = 0.10

FLOOR_STEPS_REALIZED_FRAC = 0.60
CEILING_HARM_RATE_TRAINING_COMPLETE = 0.35
MIN_POOLED_SEEDS = 2

# --- non-degeneracy floors ------------------------------------------------
FLOOR_RESIDUE_WEIGHT_VAR = 1e-6      # P1: field must not be uniform-pinned
                                      # (same floor V3-EXQ-800 validated on this
                                      # identical env/residue config)
FLOOR_CANDIDATE_SCORE_RANGE = 1e-6   # P2: cross-candidate score spread
FLOOR_E1_PREDICTION_ERROR = 1e-4     # P3: world-model error on a known hazard hit

# P4 (readiness) is DELIBERATELY DECOUPLED from C4 (verdict) -- red-team Family 4
# finding. P4 was previously set literally equal to THRESH_C4_MIN_REVISITS, which
# is tautological (P4 can never independently fail once passed) and, worse, a run
# where residue's suppression genuinely WORKS (few late revisits, because errors
# are being avoided) drove P4 red and self-routed to substrate_not_ready_requeue,
# discarding the very outcome that would confirm the claim. P4 now checks only
# that the EARLY half (which residue has not yet had a chance to shape) has
# enough revisits to trust as a baseline rate at all -- a low floor, "a handful".
# C4 stays the stricter, both-halves floor for trusting the decline COMPARISON at
# verdict time, entirely independently of P4.
# HALF-INTEGER ON PURPOSE. `PreconditionSpec.met_for` is STRICT
# (`measured > threshold`, _lib/precondition_gate.py:199) while the REE_assembly
# indexer recomputes the same entry INCLUSIVELY (`measured >= threshold`), so an
# integer-valued measurement sitting exactly on an integer threshold is read as
# UNMET by the script and MET by the indexer -- the manifest then disagrees with
# itself at precisely the boundary the floor names. A half-integer threshold is
# unreachable by an integer count, so both readings agree everywhere.
# "at least 5 early-half revisits" is therefore encoded as > 4.5.
FLOOR_REVISIT_DENOMINATOR_P4 = 4.5   # P4: early-half-only readiness floor (>= 5)

# P6 REPLACES V3-EXQ-983's P5 (`harm_rate_not_saturated`, ceiling 0.90). P5's
# subject -- a harm source saturated independently of true hazards, which pins
# repeat_rate near 1.0 in BOTH arms -- is now handled PER SEED by the completion
# gate's strictly stricter 0.35 ceiling. Keeping P5 over the POOLED rows would
# make it tautological (it cannot fail once the 0.35 band has run), and a
# tautological precondition is precisely the red-team Family-4 defect the
# predecessor already had to fix once. P6 asks what P5 no longer can: did ENOUGH
# seed-pairs survive the completion gate to support a paired comparison at all?
# Half-integer for the reason given at FLOOR_REVISIT_DENOMINATOR_P4 above:
# "at least MIN_POOLED_SEEDS pooled seeds" encoded so a measurement of exactly
# MIN_POOLED_SEEDS is MET under both the script's strict test and the indexer's
# inclusive recompute.
FLOOR_POOLED_SEED_COUNT = float(MIN_POOLED_SEEDS) - 0.5

# CEM candidate budget. MUST keep num_elite = int(NUM_CANDIDATES * elite_fraction)
# at >= 2 -- see V3-EXQ-800's identical design-time audit below for why (a
# single-elite CEM refit NaNs every candidate rollout via std() of one element).
NUM_CANDIDATES = 32
MIN_REQUIRED_ELITES = 2
CONTROL_PROBE_STEPS = 25  # positive-control probe for P2, run after training

# P3 positive-control probe budget (throwaway agent, isolated from scored arms).
PROBE_HAZARD_WARMUP_EPISODES = 10
PROBE_HAZARD_MAX_STEPS = 300
PROBE_SEED = 999983


# =========================================================================
# precondition specs
# =========================================================================
PRECONDITION_SPECS = [
    PreconditionSpec(
        name="residue_structure_live",
        description=(
            "cross-center variance of active residue RBF weights after training -- "
            "a flat field means no persistent trace was ever written, and any "
            "decline-rate difference from A1 would be a comparison against nothing"
        ),
        control="active centers accumulated over the full training phase (POOLED seeds only)",
        threshold=FLOOR_RESIDUE_WEIGHT_VAR,
        direction="lower",
        applies_to=lambda ctx: ctx["id"] == ARM_INTACT,
        applies_note=(
            "A1 is frozen-at-init BY DESIGN (asserting structure there is "
            "unsatisfiable, not a substrate fact)"
        ),
    ),
    PreconditionSpec(
        name="candidate_score_range",
        description=(
            "cross-candidate RANGE of residue trajectory scores on a post-training "
            "positive-control probe -- the SAME statistic the E3 argmin routes on "
            "(V3-EXQ-643 rule)"
        ),
        control="CEM candidates that genuinely differ, probed after training (POOLED seeds only)",
        threshold=FLOOR_CANDIDATE_SCORE_RANGE,
        direction="lower",
        applies_to=lambda ctx: True,
        applies_note="both arms read the residue field via the E3 argmin selection path",
    ),
    PreconditionSpec(
        name="e1_prediction_error_floor",
        description=(
            "E3.post_action_update's world-model prediction-error metric on a "
            "genuine hazard collision, measured on a throwaway agent isolated from "
            "both scored arms -- confirms the substrate's error signal is "
            "non-degenerate before a later null repeat-rate result is trusted"
        ),
        control="throwaway agent forced through hazard collisions",
        threshold=FLOOR_E1_PREDICTION_ERROR,
        direction="lower",
        applies_to=lambda ctx: True,
        applies_note=(
            "a substrate-wide readiness check, not an arm-specific property -- "
            "measured once and shared by both arm gates"
        ),
    ),
    PreconditionSpec(
        name="revisit_denominator",
        description=(
            "minimum number of (cell,action) revisit events observed in the "
            "EARLY training half only (DECOUPLED from C4's stricter both-halves "
            "verdict floor -- red-team Family 4 fix: the old floor equalled C4's, "
            "which is tautological and self-defeatingly routed a WORKING residue "
            "effect, few late revisits by construction, to "
            "substrate_not_ready_requeue) -- the early-half repeat rate itself is "
            "undefined/noisy below this"
        ),
        control="revisits of a previously-erred (cell,action) key during the early half (POOLED seeds only)",
        threshold=FLOOR_REVISIT_DENOMINATOR_P4,
        direction="lower",
        applies_to=lambda ctx: True,
        applies_note="both arms must accumulate enough EARLY-half revisit events to trust a baseline rate",
    ),
    PreconditionSpec(
        name="revisit_outcome_heterogeneity",
        description=(
            "fraction of erred (cell,action) keys whose revisits are NOT all "
            "identical in outcome (i.e. the key produced BOTH a harm and a "
            "non-harm revisit at some point). If every revisited key resolves the "
            "same way every time, P(harm | revisit) is fixed by the environment "
            "and has no agent-side degree of freedom, so no residue effect could "
            "move the DV and a null would be unattributable to the mechanism"
        ),
        control=(
            "the env layout and start position are pinned across episodes (fresh "
            "env from the same seed each episode, env_drift_prob=0.0), so a key's "
            "outcome varies only with what the agent does"
        ),
        threshold=FLOOR_REVISIT_HETEROGENEITY,
        direction="lower",
        applies_to=lambda ctx: True,
        applies_note="both arms -- an env-determined DV is a run-level defect",
    ),
    PreconditionSpec(
        name="pooled_seed_count",
        description=(
            "number of SEED-PAIRS surviving the training-completion gate "
            "(steps_realized_frac >= 0.60 AND harm_rate_train <= 0.35 on BOTH "
            "arms of that seed). decline_gap is a PAIRED statistic over seeds, so "
            "below this floor there is no population to pool and the comparison "
            "is undefined rather than null. REPLACES V3-EXQ-983's P5 "
            "harm_rate_not_saturated, whose subject the per-seed 0.35 ceiling now "
            "covers strictly more tightly (0.35 < 0.90) and which over POOLED "
            "rows would be tautological -- the red-team Family-4 defect"
        ),
        control=(
            "V3-EXQ-983 pooled 3 seeds of which only seed 456 had trained "
            "(steps_realized_frac 0.965 vs 0.077/0.079); this run draws 8"
        ),
        threshold=FLOOR_POOLED_SEED_COUNT,
        direction="lower",
        applies_to=lambda ctx: True,
        applies_note=(
            "a run-level population property measured identically for both arms "
            "(a seed survives as a PAIR or not at all)"
        ),
    ),
]


def arm_contexts() -> List[Dict[str, Any]]:
    return [
        {"id": ARM_INTACT, "freeze_residue": False},
        {"id": ARM_FROZEN, "freeze_residue": True},
    ]


# =========================================================================
# substrate helpers (selection path + readiness probes mirror V3-EXQ-800)
# =========================================================================
def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _make_env(seed: int, full_config: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=3,
        hazard_harm=full_config["hazard_harm"],
        # RED-TEAM (983a pass) FIX, FAMILY 1 -- see the module docstring's
        # "WHAT CHANGED" item 5. Within-episode drift is off for the same reason
        # the layout is now pinned across episodes: a (cell, action) key whose
        # hazard content moves is not "the same error" on a later visit.
        env_drift_interval=5,
        env_drift_prob=0.0,
        proximity_harm_scale=full_config["proximity_harm_scale"],
        proximity_benefit_scale=full_config["proximity_harm_scale"] * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
        # RED-TEAM 2a-i FIX: contamination_spread defaults to 0.5, which the
        # env's OWN docstring (causal_grid_world.py:71-77) names as a trap --
        # every cell the agent revisits crosses contamination_threshold after a
        # handful of contacts and starts dealing contaminated_harm=0.4 per
        # contact regardless of hazards, independent of residue. That silent
        # default was the mechanical cause of the pre-fix dry-run's
        # harm_rate_train~1.0. The docstring's own named precedent
        # (V3-EXQ-513) is to pass contamination_spread=0.0 explicitly.
        contamination_spread=0.0,
    )


def _make_agent(env: CausalGridWorldV2, full_config: Dict[str, Any]) -> REEAgent:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=full_config["self_dim"],
        world_dim=full_config["world_dim"],
        alpha_world=full_config["alpha_world"],
        alpha_self=full_config["alpha_self"],
        reafference_action_dim=0,  # SD-007 off -- isolate the residue mechanism
    )
    config.latent.unified_latent_mode = False  # SD-005 split latents
    return REEAgent(config)


def _active_weight_stats(residue_field) -> Dict[str, float]:
    """Variance / count over ACTIVE residue RBF weights (P1 measurement)."""
    rbf = residue_field.rbf_field
    with torch.no_grad():
        mask = rbf.active_mask
        n_active = int(mask.sum().item())
        if n_active < 2:
            return {"weight_var": 0.0, "n_active": n_active}
        w = rbf.weights[mask].detach()
        return {"weight_var": float(w.var(unbiased=False).item()), "n_active": n_active}


def _candidate_scores(agent, candidates) -> List[Tuple[int, float]]:
    """FINITE residue trajectory scores as (candidate_index, score), lower = better.

    Non-finite scores are DROPPED rather than ranked (same rationale as
    V3-EXQ-800's identical helper: an early-training E2 rollout can diverge and
    yield inf/nan trajectory residue, and ranking that would silently make the
    argmin arbitrary).
    """
    out: List[Tuple[int, float]] = []
    for i, traj in enumerate(candidates):
        world_seq = traj.get_world_state_sequence()
        if world_seq is None:
            continue
        val = float(agent.residue_field.evaluate_trajectory(world_seq).sum().item())
        if math.isfinite(val):
            out.append((i, val))
    return out


def _select_action(
    agent, latent, ticks, n_actions: int, rng: random.Random,
) -> Tuple[torch.Tensor, Optional[float], bool]:
    """Select via the CANONICAL E3 path: e3 ranks the hippocampal candidates.

    Reused verbatim (selection rule and rationale) from V3-EXQ-800's `_select_action`:
    `agent.select_action(candidates, ticks)` routes through E3's J(zeta), which reads
    the residue field via Phi_R and returns the action DIRECTLY -- never the
    `action_object_decoder` round trip, which V3-EXQ-800 measured collapses to a
    single constant class on this substrate (see that script's docstring for the
    measurement). Using the collapsed decoder here would make the residue-freeze
    manipulation an arithmetically forced no-op.

    Returns (action_tensor, cross_candidate_residue_score_range, used_e3).
    """
    with torch.no_grad():
        candidates = agent.hippocampal.propose_trajectories(
            latent.z_world.detach(),
            z_self=latent.z_self.detach(),
            num_candidates=NUM_CANDIDATES,
        )
        if not candidates:
            return (
                _action_to_onehot(rng.randint(0, n_actions - 1), n_actions, agent.device),
                None,
                False,
            )
        scored = _candidate_scores(agent, candidates)
        vals = [v for _i, v in scored]
        score_range = float(max(vals) - min(vals)) if len(vals) >= 2 else None
        action = agent.select_action(candidates, ticks)
        return action, score_range, True


def _probe_candidate_score_range(agent, env, n_actions, rng, n_steps) -> Dict[str, float]:
    """P2 post-training positive control: worst-case (minimum) cross-candidate range.

    Reports the MINIMUM observed range (not the mean), matching V3-EXQ-800's
    convention: P2's `met` is the claim that residue COULD discriminate candidates
    at every probed tick, and a mean can hide ticks where it could not.
    """
    ranges: List[float] = []
    executed: List[int] = []
    _, obs_dict = env.reset()
    agent.reset()
    for _ in range(n_steps):
        with torch.no_grad():
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            ticks = agent.clock.advance()
            action, span, _used = _select_action(agent, latent, ticks, n_actions, rng)
        if span is not None and math.isfinite(span):
            ranges.append(span)
        executed.append(int(torch.argmax(action, dim=-1).item()))
        _, _harm, done, _info, obs_dict = env.step(action)
        if done:
            break
    return {
        "score_range_min": float(min(ranges)) if ranges else 0.0,
        "executed_action_classes": float(len(set(executed))),
    }


def _positive_control_e1_prediction_error(
    full_config: Dict[str, Any], dry_run: bool,
) -> Dict[str, float]:
    """P3 (mandatory contract item 13): confirm the world-model error signal named
    by the claim ("E1 prediction errors") clears a non-degenerate floor on a KNOWN-
    erroneous action, before trusting a later null on the scored arms as meaningful.

    Uses `agent.e3.post_action_update`'s `prediction_error` metric
    (`actual_z_world - predicted_world`, `e3_selector.py:4243-4247`, where
    `predicted_world` is the E3-selected candidate's own E1/E2-rollout prediction)
    -- the closest available reading of "the world-model's error on the action just
    taken" on this substrate at inference time.

    Runs on a FRESH, THROWAWAY agent+env, entirely separate from A0/A1's residue
    fields -- `post_action_update` itself would otherwise accumulate residue via its
    own commitment-gated write (`e3_selector.py:4253-4256`), which must NEVER touch
    either scored arm's field. Hunts for real hazard collisions within a bounded step
    budget and returns the MINIMUM (worst-case) reading observed across them, or 0.0
    if none occurred (which correctly fails the floor and routes the run to
    substrate_not_ready_requeue rather than a false verdict on the claim).
    """
    reset_all_rng(PROBE_SEED)
    env = _make_env(PROBE_SEED, full_config)
    n_actions = env.action_dim
    agent = _make_agent(env, full_config)
    rng = random.Random(PROBE_SEED)

    n_warmup = min(2, PROBE_HAZARD_WARMUP_EPISODES) if dry_run else PROBE_HAZARD_WARMUP_EPISODES
    max_steps = min(30, PROBE_HAZARD_MAX_STEPS) if dry_run else PROBE_HAZARD_MAX_STEPS

    readings: List[float] = []
    steps_done = 0
    agent.train()
    for _ep in range(n_warmup):
        _, obs_dict = env.reset()
        agent.reset()
        while steps_done < max_steps:
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            ticks = agent.clock.advance()
            action, _span, _used = _select_action(agent, latent, ticks, n_actions, rng)
            _, harm_signal, done, _info, obs_dict = env.step(action)
            steps_done += 1
            if float(harm_signal) < 0:
                with torch.no_grad():
                    next_latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                    e3_metrics = agent.e3.post_action_update(
                        actual_z_world=next_latent.z_world, harm_occurred=True,
                    )
                pe = e3_metrics.get("prediction_error")
                if pe is not None:
                    readings.append(float(pe.detach().item()))
            if done:
                break
        if steps_done >= max_steps:
            break

    return {
        "e1_prediction_error_min": float(min(readings)) if readings else 0.0,
        "e1_prediction_error_mean": (
            float(statistics.fmean(readings)) if readings else 0.0
        ),
        "n_harm_events_observed": len(readings),
        "steps_run": steps_done,
    }


# =========================================================================
# one (seed, arm) cell
# =========================================================================
def run_cell(
    arm_ctx: Dict[str, Any],
    seed: int,
    full_config: Dict[str, Any],
    warmup_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Tuple[Dict[str, Any], REEAgent, List[List[int]]]:
    arm_id = arm_ctx["id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    with arm_cell(
        seed,
        config_slice=full_config,
        script_path=Path(__file__),
        include_driver_script_in_hash=False,
    ) as cell:
        rng = random.Random(seed)
        env = _make_env(seed, full_config)
        n_actions = env.action_dim
        agent = _make_agent(env, full_config)
        optimizer = optim.Adam(list(agent.parameters()), lr=full_config["lr"])

        n_warmup = min(3, warmup_episodes) if dry_run else warmup_episodes
        n_steps = min(20, steps_per_episode) if dry_run else steps_per_episode

        planned_total_steps = max(1, n_warmup * n_steps)

        # ---------------- TRAIN (encoder + E1/E2; residue accumulates per-arm) ---
        agent.train()
        harm_train = 0
        steps_train = 0
        hippo_fallbacks = 0

        first_harm_at: Dict[Tuple[int, int, int], int] = {}
        # RED-TEAM 2a-ii FIX: revisit events are buffered with their step index
        # rather than classified into early/late DURING the loop. The old code
        # classified against `half_point` computed from the PLANNED budget
        # (n_warmup * n_steps); but `agent_health <= 0` can end an episode well
        # short of `n_steps`, so the REALIZED `step_counter` can undershoot the
        # planned `half_point` and leave `revisit_late` permanently empty
        # (decline = nan for every row, regardless of any residue effect). The
        # early/late split below is instead computed AFTER training from the
        # REALIZED total step count.
        revisit_events: List[Tuple[int, float]] = []  # (step_index, is_repeat_harm)
        # P7 (red-team Family 1): per-key revisit outcomes, so we can report what
        # fraction of erred keys ever resolved BOTH ways. An all-identical key
        # carries no agent-side signal.
        revisit_outcomes_by_key: Dict[Tuple[int, int, int], List[float]] = {}
        step_counter = 0
        n_fresh_select = 0
        n_latched = 0

        # Family-1 mitigation (recorded, not gated): full per-episode executed-
        # action stream, compared against the other arm's (same seed) after both
        # arms have run -- see `_action_stream_divergence` / run()'s
        # `family1_action_stream_divergence`.
        episode_actions: List[List[int]] = []

        for ep in range(n_warmup):
            # RED-TEAM (983a pass) FIX, FAMILY 1 -- THE CONSTRUCT-VALIDITY FIX.
            # `CausalGridWorldV2.reset()` RE-PLACES every hazard from the seeded
            # RNG (verified empirically 2026-09-04 on this exact config: three
            # successive reset()s gave hazard sets [(1,1),(2,1),(2,2),(3,3)],
            # [(1,2),(2,2),(3,1),(4,4)], [(1,1),(1,4),(2,4),(3,1)]). Under the
            # predecessor's single-env-then-reset loop, a key recorded in
            # `first_harm_at` during episode 3 was scored against episode 50's
            # completely different layout -- so `P(harm | revisit)` was largely
            # the env's per-episode hazard lottery, which no residue field can
            # anticipate, rather than the agent's learned avoidance. Rebuilding
            # the env from the SAME seed each episode pins the layout AND the
            # start position (both verified identical across fresh builds), which
            # is what makes "the same error recurring" a well-formed measurement.
            env = _make_env(seed, full_config)
            _, obs_dict = env.reset()
            agent.reset()
            current_episode_actions: List[int] = []
            for _ in range(n_steps):
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                ticks = agent.clock.advance()
                z_world_curr = latent.z_world.detach()
                pre_x, pre_y = int(env.agent_x), int(env.agent_y)

                action, _span, used = _select_action(agent, latent, ticks, n_actions, rng)
                if not used:
                    hippo_fallbacks += 1
                action_idx = int(torch.argmax(action, dim=-1).item())
                current_episode_actions.append(action_idx)

                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps_train += 1
                is_harm = float(harm_signal) < 0

                # Repeat-error bookkeeping is gated on a FRESH E3 selection tick
                # (`ticks["e3_tick"]`), never on a HELD tick (`agent.py:7138` returns
                # the committed action unchanged while `not ticks["e3_tick"]`). Without
                # this gate a single commitment held across a hazard-adjacent wall
                # (agent pinned, cell unchanged for the whole hold) would be
                # double-counted as many identical (cell,action) revisits from ONE
                # underlying decision -- exactly the hold-weighted-readout construct
                # defect `validate_experiments.e3_hold_weighted_readout_lint` exists
                # to catch (see the module-level hold-weighted-readout triage note
                # for why the OTHER sites in this file do not need this same gate:
                # their statistics are a cardinality and a positional same-seed
                # comparison, both invariant under hold duplication -- this one is a
                # RATE, which is not).
                is_fresh_select = bool(ticks.get("e3_tick", True))
                if is_fresh_select:
                    n_fresh_select += 1
                    key = (pre_x, pre_y, action_idx)
                    if key in first_harm_at:
                        revisit_events.append((step_counter, 1.0 if is_harm else 0.0))
                        revisit_outcomes_by_key.setdefault(key, []).append(
                            1.0 if is_harm else 0.0
                        )
                    elif is_harm:
                        first_harm_at[key] = step_counter
                else:
                    n_latched += 1

                if is_harm:
                    harm_train += 1
                    # A1 freezes the terrain at initialisation: never accumulate.
                    if not arm_ctx["freeze_residue"]:
                        agent.residue_field.accumulate(
                            z_world_curr, harm_magnitude=abs(float(harm_signal)),
                        )

                loss = agent.compute_prediction_loss() + agent.compute_e2_loss()
                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

                step_counter += 1
                if done:
                    break

            episode_actions.append(current_episode_actions)

            if (ep + 1) % 50 == 0 or ep == n_warmup - 1:
                print(
                    f"  [train] seed={seed} arm={arm_id} ep {ep+1}/{n_warmup}"
                    f" harm_events={harm_train}"
                    f" harm_rate={harm_train / max(1, steps_train):.4f}"
                    f" n_revisit_events={len(revisit_events)}"
                    f" step_counter={step_counter}/{planned_total_steps}",
                    flush=True,
                )

        # ---------------- RED-TEAM 2a-ii FIX: split by REALIZED steps, not the
        # planned budget -- see the comment at revisit_events above. -------------
        # P7: a key is HETEROGENEOUS if its revisits were not all the same
        # outcome. Keys revisited only once are counted in the denominator (they
        # cannot demonstrate heterogeneity) so the statistic is conservative.
        _keys = list(revisit_outcomes_by_key.values())
        _hetero = sum(1 for v in _keys if len(set(v)) > 1)
        revisit_heterogeneity = (
            float(_hetero) / float(len(_keys)) if _keys else 0.0
        )

        realized_total_steps = step_counter
        half_point_realized = max(1, realized_total_steps // 2)
        revisit_early = [v for (i, v) in revisit_events if i < half_point_realized]
        revisit_late = [v for (i, v) in revisit_events if i >= half_point_realized]

        # ---------------- readiness measurements (post-training) -----------------
        agent.eval()
        w_stats = _active_weight_stats(agent.residue_field)
        probe = _probe_candidate_score_range(
            agent, env, n_actions, rng,
            min(5, CONTROL_PROBE_STEPS) if dry_run else CONTROL_PROBE_STEPS,
        )

        repeat_rate_early = (
            float(statistics.fmean(revisit_early)) if revisit_early else float("nan")
        )
        repeat_rate_late = (
            float(statistics.fmean(revisit_late)) if revisit_late else float("nan")
        )
        decline = (
            float(repeat_rate_early - repeat_rate_late)
            if (revisit_early and revisit_late) else float("nan")
        )

        row: Dict[str, Any] = {
            "arm_id": arm_id,
            "seed": int(seed),
            "harm_rate_train": float(harm_train / max(1, steps_train)),
            "harm_events_train": int(harm_train),
            "total_steps_train": int(steps_train),
            "n_distinct_erred_keys": int(len(first_harm_at)),
            "revisit_outcome_heterogeneity": revisit_heterogeneity,
            "n_revisited_keys": int(len(revisit_outcomes_by_key)),
            "n_heterogeneous_keys": int(_hetero),
            "n_revisits_early": int(len(revisit_early)),
            "n_revisits_late": int(len(revisit_late)),
            "n_fresh_select": int(n_fresh_select),
            "n_latched": int(n_latched),
            "fresh_select_yield": float(
                n_fresh_select / max(1, n_fresh_select + n_latched)
            ),
            "repeat_rate_early": repeat_rate_early,
            "repeat_rate_late": repeat_rate_late,
            "decline": decline,
            "residue_weight_var": float(w_stats["weight_var"]),
            "residue_active_centers": int(w_stats["n_active"]),
            "candidate_score_range_min": float(probe["score_range_min"]),
            "executed_action_classes": float(probe["executed_action_classes"]),
            "hippo_fallback_steps": int(hippo_fallbacks),
            "residue_total": float(agent.residue_field.total_residue.item()),
            "residue_harm_events": int(agent.residue_field.num_harm_events.item()),
            "residue_coverage": agent.residue_field.get_coverage_telemetry(),
            # 2a-ii diagnostics: how much episode-death truncation actually
            # occurred, distinguishing "nan due to tiny dry-run scale" from "nan
            # due to the underlying defect" at read time.
            "planned_total_steps": int(planned_total_steps),
            "realized_total_steps": int(realized_total_steps),
            "steps_realized_frac": float(realized_total_steps / planned_total_steps),
        }
        cell.stamp(row)

    print(
        f"  [eval] seed={seed} arm={arm_id}"
        f" repeat_early={row['repeat_rate_early']:.4f}"
        f" repeat_late={row['repeat_rate_late']:.4f}"
        f" decline={row['decline']:.4f}"
        f" n_revisits=({row['n_revisits_early']},{row['n_revisits_late']})"
        f" steps_realized_frac={row['steps_realized_frac']:.4f}",
        flush=True,
    )
    # Cell-level sanity check, NOT the scientific verdict (that is computed at the
    # run level in analyse()): did this cell produce any usable harm/revisit data.
    cell_ok = row["harm_events_train"] > 0 and (
        row["n_revisits_early"] + row["n_revisits_late"] > 0
    )
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row, agent, episode_actions


# =========================================================================
# analysis
# =========================================================================
def _mean(vals: List[float]) -> float:
    return float(statistics.fmean(vals)) if vals else float("nan")


def completion_gate(rows: List[Dict[str, Any]],
                    seeds: List[int]) -> Dict[str, Any]:
    """Training-completion gate -- decide which SEEDS may be pooled.

    Autopsy finding 2: V3-EXQ-983 pooled a population in which two of three seeds
    had realised 7.7-8.5% of their planned training steps (episodes ending on
    `agent_health <= 0` inside ~16 steps) at a training harm_rate of 0.68-0.75,
    against one seed at 96.5% and 0.024. Equal-weight pooling across that
    population is what produced the -0.45pp decline_gap.

    The unit is the SEED, not the cell, because `decline_gap` is a PAIRED
    statistic: admitting a trained A0 against an untrained A1 would compare two
    different populations rather than one manipulation.

    Excluded seeds are RECORDED IN FULL (both in the returned `per_seed` block and
    as ordinary `arm_results` rows) -- excluded from POOLING, never deleted.
    """
    per_seed: List[Dict[str, Any]] = []
    pooled: List[int] = []
    for s_ in seeds:
        cells = [r for r in rows if r["seed"] == s_]
        detail = {
            "seed": int(s_),
            "n_cells": len(cells),
            "per_arm": {
                r["arm_id"]: {
                    "steps_realized_frac": float(r["steps_realized_frac"]),
                    "harm_rate_train": float(r["harm_rate_train"]),
                } for r in cells
            },
        }
        if len(cells) != len(ARMS):
            detail["pooled"] = False
            detail["reason"] = f"expected {len(ARMS)} cells, saw {len(cells)}"
        else:
            bad = [
                (r["arm_id"],
                 float(r["steps_realized_frac"]),
                 float(r["harm_rate_train"]))
                for r in cells
                if not (r["steps_realized_frac"] >= FLOOR_STEPS_REALIZED_FRAC
                        and r["harm_rate_train"] <= CEILING_HARM_RATE_TRAINING_COMPLETE)
            ]
            if bad:
                detail["pooled"] = False
                detail["reason"] = "; ".join(
                    f"{a}: steps_realized_frac={f:.4f} (floor {FLOOR_STEPS_REALIZED_FRAC})"
                    f", harm_rate_train={h:.4f} (ceiling "
                    f"{CEILING_HARM_RATE_TRAINING_COMPLETE})"
                    for a, f, h in bad
                )
            else:
                detail["pooled"] = True
                detail["reason"] = ""
                pooled.append(int(s_))
        per_seed.append(detail)
    # RED-TEAM (983a pass), FAMILY 4: the completion gate reads two statistics the
    # manipulation could itself move (survival and harm rate), and requires BOTH
    # arms of a seed in-band. If residue genuinely helps, a seed where A0 survives
    # and A1 dies is the strongest possible evidence -- and the gate discards it.
    # The gate is still correct (comparing a trained A0 against an untrained A1 is
    # exactly the artifact the predecessor produced), but the bias direction is
    # AGAINST the claim, so the asymmetry is recorded rather than left invisible:
    # a run whose exclusions are systematically one-armed is itself a finding.
    one_armed = [
        d for d in per_seed
        if not d.get("pooled")
        and isinstance(d.get("per_arm"), dict)
        and len(d["per_arm"]) == len(ARMS)
        and 0 < sum(
            1 for v in d["per_arm"].values()
            if v["steps_realized_frac"] >= FLOOR_STEPS_REALIZED_FRAC
            and v["harm_rate_train"] <= CEILING_HARM_RATE_TRAINING_COMPLETE
        ) < len(ARMS)
    ]
    return {
        "exclusion_asymmetry": {
            "n_one_armed_exclusions": len(one_armed),
            "seeds": [d["seed"] for d in one_armed],
            "surviving_arm_by_seed": {
                str(d["seed"]): [
                    a for a, v in d["per_arm"].items()
                    if v["steps_realized_frac"] >= FLOOR_STEPS_REALIZED_FRAC
                    and v["harm_rate_train"] <= CEILING_HARM_RATE_TRAINING_COMPLETE
                ] for d in one_armed
            },
            "note": (
                "seeds excluded because exactly ONE arm cleared the band. If these "
                "are systematically A0-survives/A1-dies, the exclusion is removing "
                "evidence FOR the claim and should be read as a finding, not as "
                "housekeeping."
            ),
        },
        "floor_steps_realized_frac": FLOOR_STEPS_REALIZED_FRAC,
        "ceiling_harm_rate_train": CEILING_HARM_RATE_TRAINING_COMPLETE,
        "min_pooled_seeds": MIN_POOLED_SEEDS,
        "pooled_seeds": pooled,
        "excluded_seeds": [int(x) for x in seeds if int(x) not in pooled],
        "n_pooled": len(pooled),
        "per_seed": per_seed,
    }


def dv_headroom_gate(rows: List[Dict[str, Any]],
                     pooled_seeds: List[int]) -> Dict[str, Any]:
    """The `dv_headroom` precondition for C1 -- the point of this whole redesign.

    Measures what `decline_gap` CAN reach in this configuration, from the CONTROL
    arm's own realised spread over the pooled seeds, and requires that to cover the
    registered C1 threshold. Built with `_metrics.dv_headroom_check` and evaluated
    by `_metrics.p0_readiness_gate`, so an unmet entry raises P0NotReady and the
    caller self-routes `substrate_not_ready_requeue` exactly like any other
    readiness failure -- never a verdict on EXT-002.

    Necessarily POST-training: a control arm's realised range is not knowable
    before the control arm has run. What it protects is the VERDICT, not the
    compute (`_metrics.p0_readiness_gate`'s docstring makes the same point for the
    dv_headroom kind).

    Returns {"available": bool, "reason": str, "check": <check dict or None>}.
    `available` is False only when there is nothing to measure -- fewer than two
    pooled control values, which the P6 gate has already caught and reported more
    legibly.
    """
    # RED-TEAM (983a pass) FIX, FAMILY 4. The first draft measured the range of
    # the CONTROL ARM's per-seed declines alone. That statistic is the control
    # arm's seed-to-seed NOISE, and it is anti-correlated with the thing the gate
    # should protect: a real effect that is CONSISTENT across seeds produces a
    # TIGHT control arm, so a control-only range gate refuses exactly the
    # best-powered runs. It is measured here over EVERY pooled cell, both arms --
    # which is also precisely what the cluster autopsy itself computed ("realised
    # range of 0.0468 ACROSS ALL SIX CELLS"), and what `_metrics.dv_achievable`'s
    # "range" docstring means by "the DV's demonstrated dynamic range". A real
    # effect WIDENS this range, so the gate now moves with the effect rather than
    # against it, and a DV pinned in both arms -- the V3-EXQ-983 pathology -- is
    # still refused.
    control = [
        float(r["decline"]) for r in rows
        if int(r["seed"]) in set(pooled_seeds) and math.isfinite(r["decline"])
    ]
    if len(control) < 2:
        return {
            "available": False,
            "reason": (
                f"only {len(control)} finite `decline` value(s) over pooled cells; "
                f"a RANGE needs at least 2. P6 (pooled_seed_count) is the "
                f"precondition that reports this."
            ),
            "check": None,
            "control_values": control,
        }
    check = dv_headroom_check(
        "dv_headroom_decline_gap",
        dv_name=DV_HEADROOM_DV_NAME,
        criterion_threshold=THRESH_C1_DECLINE_GAP,
        control_values=control,
        statistic="range",
        margin=DV_HEADROOM_MARGIN,
        description=(
            "can `decline_gap` reach its own registered C1 threshold in this "
            "configuration? Measured as the RANGE of `decline` over EVERY pooled "
            "cell (both arms) -- the DV's demonstrated dynamic range, which is the "
            "quantity the cluster autopsy itself tabulated (0.0468 across all six "
            "V3-EXQ-983 cells) and the case _metrics.dv_achievable names for the "
            "'range' statistic. Deliberately NOT the control arm alone: that is "
            "the control's seed-to-seed noise, and a consistent real effect makes "
            "it SMALL, so a control-only gate would refuse the best-powered runs"
        ),
        control=(
            "every training-complete cell of both arms; a real effect widens this "
            "range, a DV pinned in both arms (the V3-EXQ-983 pathology) does not"
        ),
        predecessor_run_id=PREDECESSOR_RUN_ID,
        predecessor_control_range=PREDECESSOR_CONTROL_RANGE,
        pooled_seeds=[int(x) for x in pooled_seeds],
        measured_over="all pooled cells, both arms",
    )
    return {
        "available": True,
        "reason": "",
        "check": check,
        "control_values": control,
        "achievable_recomputed": float(dv_achievable(control, statistic="range")),
    }


def analyse(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
    for r in rows:
        by_arm[r["arm_id"]].append(r)

    def _seed_row(arm: str, s: int) -> Optional[Dict[str, Any]]:
        return next((r for r in by_arm[arm] if r["seed"] == s), None)

    per_seed_diff: List[float] = []
    declines: Dict[str, List[float]] = {ARM_INTACT: [], ARM_FROZEN: []}
    for s in seeds:
        r0 = _seed_row(ARM_INTACT, s)
        r1 = _seed_row(ARM_FROZEN, s)
        if r0 and r1 and math.isfinite(r0["decline"]) and math.isfinite(r1["decline"]):
            per_seed_diff.append(r0["decline"] - r1["decline"])
            declines[ARM_INTACT].append(r0["decline"])
            declines[ARM_FROZEN].append(r1["decline"])

    mean_decline_a0 = _mean(declines[ARM_INTACT])
    mean_decline_a1 = _mean(declines[ARM_FROZEN])
    decline_gap = (
        float(mean_decline_a0 - mean_decline_a1)
        if math.isfinite(mean_decline_a0) and math.isfinite(mean_decline_a1)
        else float("nan")
    )

    delta_mean = _mean(per_seed_diff)
    delta_sd = (
        float(statistics.pstdev(per_seed_diff)) if len(per_seed_diff) > 1 else 0.0
    )
    effect_sd = (
        float(delta_mean / delta_sd) if delta_sd > 1e-12
        else (float("inf") if math.isfinite(delta_mean) and delta_mean > 0 else 0.0)
    )
    seeds_consistent = sum(1 for d in per_seed_diff if d > 0)
    # C4 ranges over POOLED cells only -- an excluded seed's revisit counts are
    # artifacts of a cell that never trained and must not decide data quality for
    # the comparison they are not part of.
    _pooled_set = {int(x) for x in seeds}
    min_revisits = min(
        (min(r["n_revisits_early"], r["n_revisits_late"])
         for r in rows if int(r["seed"]) in _pooled_set), default=0,
    )

    c1 = bool(math.isfinite(decline_gap) and decline_gap >= THRESH_C1_DECLINE_GAP)
    c2 = bool(math.isfinite(effect_sd) and effect_sd >= THRESH_C2_EFFECT_SD)
    c3 = bool(seeds_consistent >= THRESH_C3_MIN_SEEDS)
    c4 = bool(min_revisits >= THRESH_C4_MIN_REVISITS)

    return {
        "mean_decline_A0_intact": mean_decline_a0,
        "mean_decline_A1_frozen": mean_decline_a1,
        "decline_gap": decline_gap,
        "per_seed_diff": per_seed_diff,
        "delta_mean": float(delta_mean) if math.isfinite(delta_mean) else float("nan"),
        "delta_sd": delta_sd,
        "effect_sd": effect_sd,
        "seeds_consistent": int(seeds_consistent),
        "min_revisits": int(min_revisits),
        "c1_decline_gap_pass": c1,
        "c2_effect_sd_pass": c2,
        "c3_seed_consistency_pass": c3,
        "c4_data_quality_pass": c4,
        "all_pass": bool(c1 and c2 and c3 and c4),
    }


def _action_stream_divergence(
    episodes_a: List[List[int]], episodes_b: List[List[int]],
) -> Dict[str, float]:
    """Family-1 mitigation (red-team, contested -> confirmed live risk, non-blocking).

    P2 measures `residue_field.evaluate_trajectory` in ISOLATION, not the composite
    E3 score `f_weight*F + lambda_eff*M + rho_residue*phi` the argmin actually
    routes on (`e3_selector.py` `score_trajectory`, ~line 1386; `rho_residue`
    defaults to 0.5 against `f_weight`/`lambda_ethical` = 1.0,
    `ree_core/utils/config.py:911-925`). So a green P2 does not guarantee the
    residue-freeze manipulation (A0 vs A1) actually reaches BEHAVIOUR -- if F/M
    dominate the composite, A0 and A1 could select identical actions throughout
    despite a non-trivial isolated residue-score range.

    Not upgraded to a hard gate -- no calibrated threshold exists for "how much
    divergence is enough" -- but RECORDED so a null C1 result can be told apart
    from "the manipulation never reached behaviour" (near-zero divergence here)
    vs. a genuine null (non-trivial divergence, still no persistence effect).

    Compares same-seed A0/A1 executed-action streams EPISODE-BY-EPISODE (not as
    one flat concatenation) because `agent_health` death can end an episode at a
    different real length per arm once behaviour diverges -- comparing flat
    streams past that point would compare unrelated ticks.
    """
    n_compared = 0
    n_diverged = 0
    for ep_a, ep_b in zip(episodes_a, episodes_b):
        m = min(len(ep_a), len(ep_b))
        for i in range(m):
            n_compared += 1
            if ep_a[i] != ep_b[i]:
                n_diverged += 1
    frac = float(n_diverged / n_compared) if n_compared > 0 else float("nan")
    return {
        "n_episodes_compared": int(min(len(episodes_a), len(episodes_b))),
        "n_steps_compared": int(n_compared),
        "n_steps_diverged": int(n_diverged),
        "diverged_fraction": frac,
    }


def build_gate(rows: List[Dict[str, Any]], e1_pe_min: float,
               pooled_seeds: List[int]) -> Dict[str, Any]:
    """Per-arm precondition gate over the POOLED cells only.

    A red arm never vacates a green one (module default) -- but see run() for why
    THIS script additionally requires `all_green`, not `any_green`, as its own
    readiness verdict (the primary statistic is a PAIRED comparison).

    P1/P2/P4 are measured over cells whose seed survived the training-completion
    gate. Measuring them over an excluded seed's cells would gate the comparison on
    data that is not part of it -- and, for P4 in particular, an untrained cell's
    revisit counts are inflated by exactly the pathology the exclusion removes.
    P6 (`pooled_seed_count`) is a run-level population property and is supplied
    identically to both arms.
    """
    pooled_set = {int(x) for x in pooled_seeds}
    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
    for r in rows:
        if int(r["seed"]) in pooled_set:
            by_arm[r["arm_id"]].append(r)

    gates = []
    for ctx in arm_contexts():
        arm_rows = by_arm[ctx["id"]]
        # An EMPTY pooled population is not a coding error here -- it is the
        # outcome the completion gate exists to produce when no seed trained. Feed
        # the floor-kind measurements 0.0 so the arm gate goes RED with a readable
        # entry, rather than raising min() on an empty sequence and turning a
        # legitimate substrate_not_ready_requeue into an ERROR. P6 reports the
        # actual cause alongside.
        _min = (lambda key: float(min(r[key] for r in arm_rows))) if arm_rows \
            else (lambda key: 0.0)
        measured: Dict[str, float] = {
            "candidate_score_range": _min("candidate_score_range_min"),
            "e1_prediction_error_floor": float(e1_pe_min),
            # P4 (readiness): EARLY half only -- see FLOOR_REVISIT_DENOMINATOR_P4
            # and the PreconditionSpec note for why this is decoupled from C4.
            "revisit_denominator": _min("n_revisits_early"),
            # P7: worst (lowest) heterogeneity across this arm's pooled cells --
            # the worst-cell rule, since `met` is an all-cells claim.
            "revisit_outcome_heterogeneity": _min("revisit_outcome_heterogeneity"),
            # P6: run-level population size -- identical for both arms, because a
            # seed survives the completion gate as a PAIR or not at all.
            "pooled_seed_count": float(len(pooled_set)),
        }
        if ctx["id"] == ARM_INTACT:
            measured["residue_structure_live"] = _min("residue_weight_var")
        gates.append(
            evaluate_arm_gate(ctx["id"], ctx, PRECONDITION_SPECS, measured=measured)
        )
    return aggregate_arm_gates(gates)


# =========================================================================
# main
# =========================================================================
def run(
    seeds: Tuple[int, ...] = (42, 123, 456, 7, 11, 17, 23, 31),
    warmup_episodes: int = 110,
    steps_per_episode: int = 200,
    dry_run: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    full_config: Dict[str, Any] = {
        "env": "CausalGridWorldV2",
        "size": 6,
        "num_hazards": 4,
        "num_resources": 3,
        "hazard_harm": 0.02,
        "proximity_harm_scale": 0.05,
        "use_proxy_fields": True,
        "self_dim": 32,
        "world_dim": 32,
        "alpha_world": 0.9,   # SD-008: z_world fidelity needed for terrain reads
        "alpha_self": 0.3,
        "lr": 1e-3,
        "num_candidates": NUM_CANDIDATES,
        "warmup_episodes": warmup_episodes,
        "steps_per_episode": steps_per_episode,
        "unified_latent_mode": False,
        "reafference_action_dim": 0,
        "arms": ARMS,
        "selection_rule": "argmin_over_candidate_residue_scores",
        "repeat_error_key": "(pre_action_grid_x, pre_action_grid_y, executed_action_class)",
        "control_probe_steps": CONTROL_PROBE_STEPS,
        "supersedes": SUPERSEDES,
        "thresholds": {
            "C1_decline_gap": THRESH_C1_DECLINE_GAP,
            "C2_effect_sd": THRESH_C2_EFFECT_SD,
            "C3_min_seeds": THRESH_C3_MIN_SEEDS,
            "C4_min_revisits": THRESH_C4_MIN_REVISITS,
        },
        "dv_headroom": {
            "dv_name": DV_HEADROOM_DV_NAME,
            "control_arm": DV_HEADROOM_CONTROL_ARM,
            "statistic": "range",
            "margin": DV_HEADROOM_MARGIN,
            "criterion_threshold": THRESH_C1_DECLINE_GAP,
            "predecessor_run_id": PREDECESSOR_RUN_ID,
            "predecessor_control_range": PREDECESSOR_CONTROL_RANGE,
        },
        "training_completion_gate": {
            "floor_steps_realized_frac": FLOOR_STEPS_REALIZED_FRAC,
            "ceiling_harm_rate_train": CEILING_HARM_RATE_TRAINING_COMPLETE,
            "min_pooled_seeds": MIN_POOLED_SEEDS,
        },
    }

    # Design-time audit 1: refuse a run carrying a structurally unsatisfiable gate
    # BEFORE any compute is spent (the V3-EXQ-785 free catch).
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_contexts())

    # Design-time audit 2 (identical to V3-EXQ-800): a single-elite CEM refit makes
    # ao_std NaN and silently NaNs every candidate rollout, zeroing the cross-
    # candidate score range and turning the residue-freeze manipulation into a
    # guaranteed no-op.
    _elite_fraction = float(HippocampalConfig().elite_fraction)
    _num_elite = max(1, int(NUM_CANDIDATES * _elite_fraction))
    if _num_elite < MIN_REQUIRED_ELITES:
        raise ValueError(
            f"CEM refit would use num_elite={_num_elite} from "
            f"num_candidates={NUM_CANDIDATES} x elite_fraction={_elite_fraction}. "
            f"std() over a single elite is NaN and poisons ao_std, NaN-ing every "
            f"candidate rollout and identically zeroing the cross-candidate residue "
            f"score range -- the A1 freeze manipulation would become a guaranteed "
            f"no-op. Raise NUM_CANDIDATES so num_elite >= {MIN_REQUIRED_ELITES}."
        )
    # Design-time audit 3 (NEW -- the point of this redesign): C1's bar must lie
    # inside the dynamic range the PREDECESSOR's control arm actually demonstrated.
    # This is the paper arithmetic the cluster autopsy says nobody did for
    # V3-EXQ-983's 0.15 (which needed 3.2x more room than the DV ever had). It is
    # a design-time proof against a number already in hand; the runtime
    # `dv_headroom` gate re-asks it of THIS run's own control arm.
    _required_headroom = THRESH_C1_DECLINE_GAP * DV_HEADROOM_MARGIN
    if _required_headroom > PREDECESSOR_CONTROL_RANGE:
        raise ValueError(
            f"C1 requires headroom {_required_headroom:.6g} "
            f"(THRESH_C1_DECLINE_GAP {THRESH_C1_DECLINE_GAP} x margin "
            f"{DV_HEADROOM_MARGIN}) but the predecessor {PREDECESSOR_RUN_ID} "
            f"measured a control-arm range of only {PREDECESSOR_CONTROL_RANGE:.6g}. "
            f"Registering a bar outside the range the DV has been shown to reach is "
            f"exactly the defect this redesign exists to remove -- lower "
            f"THRESH_C1_DECLINE_GAP or change the DV, do not raise the margin."
        )
    print(
        f"[V3-EXQ-983a] design-audit OK: gate satisfiable, "
        f"num_elite={_num_elite} (>= {MIN_REQUIRED_ELITES}); "
        f"C1 headroom required {_required_headroom:.6g} <= predecessor control "
        f"range {PREDECESSOR_CONTROL_RANGE:.6g}",
        flush=True,
    )

    # P3 positive control -- global, isolated from both scored arms.
    pe_probe = _positive_control_e1_prediction_error(full_config, dry_run)
    print(
        f"[V3-EXQ-983a] P3 positive control: e1_prediction_error_min="
        f"{pe_probe['e1_prediction_error_min']:.6g} "
        f"(n_harm_events={pe_probe['n_harm_events_observed']}, "
        f"steps_run={pe_probe['steps_run']})",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    agents: List[REEAgent] = []
    family1_divergence: Dict[str, Dict[str, float]] = {}
    for seed in seeds:
        seed_episode_actions: Dict[str, List[List[int]]] = {}
        for ctx in arm_contexts():
            row, agent, episode_actions = run_cell(
                arm_ctx=ctx,
                seed=seed,
                full_config=full_config,
                warmup_episodes=warmup_episodes,
                steps_per_episode=steps_per_episode,
                dry_run=dry_run,
            )
            rows.append(row)
            agents.append(agent)
            seed_episode_actions[ctx["id"]] = episode_actions

        # Family-1 mitigation: same-seed A0-vs-A1 executed-action divergence,
        # recorded (not gated) -- see _action_stream_divergence.
        div = _action_stream_divergence(
            seed_episode_actions[ARM_INTACT], seed_episode_actions[ARM_FROZEN],
        )
        family1_divergence[str(seed)] = div
        print(
            f"[V3-EXQ-983a] Family-1 action-stream divergence seed={seed}: "
            f"diverged_fraction={div['diverged_fraction']:.4f} "
            f"({div['n_steps_diverged']}/{div['n_steps_compared']} steps, "
            f"{div['n_episodes_compared']} episodes compared)",
            flush=True,
        )

    # ---- training-completion gate: decide the pooled seed population --------
    completion = completion_gate(rows, list(seeds))
    pooled_seeds = completion["pooled_seeds"]
    print(
        f"[V3-EXQ-983a] training-completion gate: pooled {completion['n_pooled']}"
        f"/{len(seeds)} seeds {pooled_seeds} "
        f"(floor steps_realized_frac >= {FLOOR_STEPS_REALIZED_FRAC}, "
        f"ceiling harm_rate_train <= {CEILING_HARM_RATE_TRAINING_COMPLETE}); "
        f"excluded {completion['excluded_seeds']}",
        flush=True,
    )
    for d in completion["per_seed"]:
        if not d["pooled"]:
            print(f"  [excluded] seed={d['seed']}: {d['reason']}", flush=True)

    gate = build_gate(rows, pe_probe["e1_prediction_error_min"], pooled_seeds)
    analysis = analyse(rows, pooled_seeds)

    # ---- dv_headroom gate: can decline_gap reach its own C1 bar here? -------
    headroom = dv_headroom_gate(rows, pooled_seeds)
    headroom_preconditions: List[Dict[str, Any]] = []
    headroom_met: Optional[bool] = None
    headroom_reason = headroom["reason"]
    if headroom["available"]:
        try:
            headroom_preconditions = p0_readiness_gate([headroom["check"]])
            headroom_met = True
        except P0NotReady as exc:
            headroom_preconditions = list(exc.preconditions)
            headroom_met = False
            _c = headroom["check"]
            headroom_reason = (
                f"DV HEADROOM UNMET: {DV_HEADROOM_DV_NAME} can only reach "
                f"{_c['measured']:.6g} in this configuration (RANGE of the "
                f"{DV_HEADROOM_CONTROL_ARM} control arm over pooled seeds "
                f"{pooled_seeds}), against a required {_c['threshold']:.6g} "
                f"(C1 {THRESH_C1_DECLINE_GAP} x margin {DV_HEADROOM_MARGIN}) -- "
                f"a {1.0 / max(1e-12, _c.get('headroom_ratio', 0.0)):.1f}x "
                f"shortfall. No outcome of this run could have shown the "
                f"registered effect, so nothing is concluded about EXT-002."
            )
    print(
        f"[V3-EXQ-983a] dv_headroom: available={headroom['available']} "
        f"met={headroom_met} "
        f"achievable={headroom.get('achievable_recomputed')} "
        f"required={THRESH_C1_DECLINE_GAP * DV_HEADROOM_MARGIN:.6g}",
        flush=True,
    )

    # This design's primary statistic is a PAIRED comparison (decline_A0 -
    # decline_A1) and is only interpretable when BOTH arms cleared their gate --
    # unlike V3-EXQ-800's four-arm design where a manipulated arm's result stands
    # alone against a common reference. So `all_green`, not the module default
    # `any_green`, is this script's readiness verdict (see module docstring).
    # The dv_headroom gate is a readiness condition of the SAME class: an unmet
    # entry means the DV could not have shown the registered effect, so it routes
    # to substrate_not_ready_requeue and NEVER to a verdict label (the
    # /queue-experiment rule: below-floor readiness -> requeue, never
    # substrate_ceiling / does_not_support / *_nondiscriminative).
    non_degenerate = bool(gate["all_green"]) and headroom_met is True
    if not non_degenerate:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        _parts = []
        if not gate["all_green"]:
            _parts.append(gate["degeneracy_reason"])
        if headroom_met is not True:
            _parts.append(headroom_reason)
        interpretation_text = (
            "SUBSTRATE NOT READY -- not a verdict on EXT-002 or ARC-013. "
            + " | ".join(x for x in _parts if x)
        )
    elif not analysis["c4_data_quality_pass"]:
        # RED-TEAM (983a pass) FIX, FAMILY 2. C4 is a DATA-QUALITY control, not a
        # scientific criterion, and its failure signature is the SUCCESS
        # signature: a working residue effect suppresses late revisits of costly
        # keys, which is exactly what drives the late-half count under C4's floor.
        # Routing that to `weakens` records the claim's own confirmation as
        # evidence against it -- the identical self-defeating structure the
        # predecessor's red-team already had to fix once for P4 (Family 4), left
        # standing on C4. A data-quality failure now self-routes to requeue.
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        interpretation_text = (
            "DATA QUALITY BELOW FLOOR -- not a verdict on EXT-002 or ARC-013. C4 "
            f"requires at least {THRESH_C4_MIN_REVISITS} revisits in BOTH halves "
            f"of every pooled cell; the worst cell had {analysis['min_revisits']}. "
            "Note this can be the SUCCESS signature (residue suppressing late "
            "revisits), which is why it routes here and not to `weakens`; a "
            "requeue should raise the step budget rather than lower the floor."
        )
    else:
        outcome = "PASS" if analysis["all_pass"] else "FAIL"
        if analysis["all_pass"]:
            label = "ext002_residue_persistent_error_record_supported"
            direction = "supports"
            interpretation_text = (
                "EXT-002 SUPPORTED: with the residue field intact and accumulating, "
                "the repeat-error rate at matched (cell,action) pairs declined by "
                f"{analysis['decline_gap']*100:.1f} percentage points more than under "
                "the residue-frozen control, over "
                f"{completion['n_pooled']} training-complete seed pairs and with the "
                "DV demonstrably in range (dv_headroom met). Residue accumulation "
                "measurably suppresses repeating a known-costly action, structural "
                "pressure the hallucination analog (no persistent error cost) lacks. "
                "This also reverses V3-EXQ-983's weak null on the same prediction."
            )
        else:
            label = "ext002_residue_persistent_error_record_not_supported"
            direction = "weakens"
            interpretation_text = (
                "EXT-002 WEAKENED: with the residue field intact, the repeat-error "
                "rate did not decline materially more than under the residue-frozen "
                f"control (decline_gap {analysis['decline_gap']*100:.2f}pp < "
                f"{THRESH_C1_DECLINE_GAP*100:.0f}pp, or effect/consistency/data-"
                "quality criteria unmet) -- with the DV demonstrably in range "
                "(dv_headroom met, the gate V3-EXQ-983 lacked) and over "
                f"{completion['n_pooled']} training-complete seed pairs. This is the "
                "SECOND null on this prediction and the first one powered to see "
                "the registered effect: V3-EXQ-983's null was weak (positive "
                "control 51x, manipulation reaching behaviour at 0.19-0.30 action "
                "divergence, but the DV 3.2x short of its own bar). Residue "
                "accumulation is live here (P1) and reaches behaviour, but is not "
                "shown to create the structural pressure against repeat errors the "
                "EXT-002 rider asserts, as operationalised by this DV. It does NOT "
                "bear on INV-006, which is derivational."
            )

    if direction == "supports":
        per_claim_direction = {"EXT-002": "supports", "ARC-013": "supports"}
    elif direction == "weakens":
        per_claim_direction = {"EXT-002": "weakens", "ARC-013": "mixed"}
    else:
        per_claim_direction = {c: direction for c in CLAIM_IDS}

    criteria = [
        {"name": "C1_decline_gap", "load_bearing": True,
         "passed": analysis["c1_decline_gap_pass"]},
        {"name": "C2_effect_sd", "load_bearing": False,
         "passed": analysis["c2_effect_sd_pass"]},
        {"name": "C3_seed_consistency", "load_bearing": False,
         "passed": analysis["c3_seed_consistency_pass"]},
        {"name": "C4_data_quality", "load_bearing": False,
         "passed": analysis["c4_data_quality_pass"]},
    ]

    # All four criteria are a PAIRED property of (A0, A1) together -- see the
    # `non_degenerate` note above -- so their non-degeneracy is the paired-green
    # flag directly, rather than `precondition_gate.arm_criteria_non_degenerate`
    # (which attributes a criterion to a single owning arm; not the right shape
    # here, since neither arm's gate alone makes the paired decline_gap meaningful).
    # ...and a criterion whose DV could not reach its own bar is degenerate no
    # matter how green the arm gates are -- that is the whole autopsy finding.
    paired_green = bool(
        ARM_INTACT in gate["green_arms"] and ARM_FROZEN in gate["green_arms"]
        and headroom_met is True
    )
    criteria_non_degenerate = {c["name"]: paired_green for c in criteria}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "backlog_id": BACKLOG_ID,
        "outcome": outcome,
        "timestamp_utc": ts,
        "supersedes": SUPERSEDES,
        "evidence_direction": direction,
        # RED-TEAM (983a pass) FIX, FAMILY 3. The first draft applied the blanket
        # direction to both claims, so a null recorded `weakens` against ARC-013 --
        # while P1, in the SAME run, is direct evidence FOR ARC-013's persistence
        # property (a populated, structured field). ARC-013 asserts that residue IS
        # persistent latent-space curvature; this DV asks the narrower question of
        # whether that curvature produces one particular behavioural signature. A
        # null weakens the FUNCTIONAL reading without touching the persistence
        # property P1 confirms, so ARC-013 takes `mixed`, not `weakens`.
        "evidence_direction_per_claim": per_claim_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": gate["degeneracy_reason"],
        "per_arm_gate": gate["per_arm_gate"],
        "positive_control_e1_prediction_error": pe_probe,
        "completion_gate": completion,
        "dv_headroom_gate": {
            "available": headroom["available"],
            "met": headroom_met,
            "reason": headroom_reason,
            "control_arm": DV_HEADROOM_CONTROL_ARM,
            "control_values": headroom.get("control_values", []),
            "achievable": headroom.get("achievable_recomputed"),
            "required": THRESH_C1_DECLINE_GAP * DV_HEADROOM_MARGIN,
            "predecessor_run_id": PREDECESSOR_RUN_ID,
            "predecessor_control_range": PREDECESSOR_CONTROL_RANGE,
        },
        # The weak-null prior this redesign carries rather than erases -- see the
        # module docstring and EXT-002's evidence_quality_note.
        "predecessor_weak_null": {
            "run_id": PREDECESSOR_RUN_ID,
            "queue_id": SUPERSEDES,
            "adjudicated": "non_contributory (measurement grounds)",
            "p3_positive_control_margin_x": 51.2,
            "action_stream_divergence_range": [0.186, 0.295],
            "decline_gap": -0.0044813346981,
            "dv_realised_range": PREDECESSOR_CONTROL_RANGE,
            "registered_bar": 0.15,
            "shortfall_x": 3.2,
            "note": (
                "carried as a prior, not erased: the positive control passed and "
                "the manipulation reached behaviour while the DV did not move, but "
                "the DV had 3.2x too little room for the registered effect. A null "
                "HERE, with dv_headroom met, is a second and stronger null."
            ),
        },
        "interpretation": {
            "label": label,
            "text": interpretation_text,
            # The per-arm entries plus the run-level dv_headroom entry. The
            # indexer reads this list flat and kind-agnostically, recomputing `met`
            # from (measured, threshold, direction) -- a dv_headroom entry rides
            # that same single-bound path by construction (_metrics.py).
            "preconditions": list(gate["adjudication_preconditions"])
                             + list(headroom_preconditions),
            "preconditions_scope_note": gate.get("preconditions_scope_note", ""),
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "criteria": criteria,
        "analysis": analysis,
        "arm_results": rows,
        "per_seed_decline": {
            a: [r["decline"] for r in rows if r["arm_id"] == a] for a in ARMS
        },
        "registered_thresholds": full_config["thresholds"],
        "pooled_seeds": pooled_seeds,
        # Family-1 mitigation (recorded, non-gating) -- see
        # _action_stream_divergence for what this does and does not prove.
        "family1_action_stream_divergence": family1_divergence,
    }

    # stamp AFTER arm_results so substrate_hash HOISTS from the per-cell
    # fingerprints; agent= wires the z_goal_stream liveness block (this driver
    # steps agents; z_goal is not the mechanism under test but the block must be
    # recorded, not silently left "NOT RECORDED" -- see /queue-experiment Step 4).
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=list(seeds),
        script_path=Path(__file__),
        started_at=t0,
        agent=agents,
    )

    print("\n[V3-EXQ-983a] Results", flush=True)
    print(
        f"  mean_decline_A0={analysis['mean_decline_A0_intact']:.4f}"
        f"  mean_decline_A1={analysis['mean_decline_A1_frozen']:.4f}"
        f"  decline_gap={analysis['decline_gap']:.4f}",
        flush=True,
    )
    print(
        f"  pooled_seeds={pooled_seeds}  dv_headroom_met={headroom_met}", flush=True
    )
    print(f"  non_degenerate={non_degenerate}  outcome={outcome}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456, 7, 11, 17, 23, 31],
    )
    parser.add_argument("--warmup", type=int, default=110)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t_start = time.perf_counter()
    manifest = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        steps_per_episode=args.steps,
        dry_run=args.dry_run,
    )

    out_path = write_flat_manifest(
        manifest,
        None,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=list(args.seeds),
        script_path=Path(__file__),
        started_at=_t_start,
    )
    print(f"\nResult written to: {out_path}", flush=True)

    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
