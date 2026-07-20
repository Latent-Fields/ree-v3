#!/opt/local/bin/python3
"""
V3-EXQ-699b -- conversion_ceiling_campaign:P-comp -- SELECTION-FACE COMPOSITION
characterization: does the MECH-448 (ARC-107) RANK-PRESERVING F->ELIGIBILITY
DEMOTION lever and the MECH-449 (ARC-107) GO/NO-GO ELIGIBILITY CONSTITUTION
(active No-Go opponency leg) COMPOUND or CANCEL at the committed-class entropy
(C2) DV when co-armed?

INSTRUMENT REPAIR OF V3-EXQ-699 -- READ THIS FIRST
==================================================
Source: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-699_2026-07-20.json
(REE_assembly ac2fb64028). The scientific question is UNCHANGED; only the
instrumentation was wrong, which is why this takes an alphabetic SUFFIX per
CLAUDE.md EXQ versioning ("bug fix = the scientific question is unchanged but the
implementation was wrong (broken instrumentation)"; "when in doubt: new letter"),
matching the V3-EXQ-708 -> 708a precedent. The autopsy records a deliberate
dissent from an earlier chip that asserted a new EXQ number.

WHAT STANDS from 699: the PASS, and the entire C1 readiness battery. Its single
load-bearing criterion is readiness-only (C_READY: levers engaged and substrate
exercisable), and all seven preconditions survive on THRESHOLD-INVARIANCE grounds
-- the C1e/C1f non-vacuity floors are literally 0.0 (a strictly-positive test
cannot be manufactured from an all-zero record, in either direction), and the
active-frac and frac_pre_ge2 gates read EXACTLY 1.0, which given the per-select
wholesale reassignment of last_score_diagnostics (e3_selector.py:2452) requires
the gates to have fired on EVERY E3 tick. Both levers demonstrably did real work.

WHAT DOES NOT STAND: the `levers_compound` composition finding, WITHDRAWN.
699's primary DV, committed_class_entropy_nats, was accumulated at driver
:882/:899 from int(action[0].argmax()) on the action RETURNED by
agent.select_action -- once per ENV STEP. On a non-E3 tick, ree_core/agent.py:5430
returns the HELD action before e3.select() is ever reached. So the DV was
hold-duration-weighted OCCUPANCY entropy standing in for the per-COMMITMENT
construct the driver names (lines below: "the levers move WHICH CLASS is
committed") and that both claims act on -- MECH-448 and MECH-449 both operate at
SELECTION. This is a CONSTRUCT MISMATCH, not staleness: the diagnostics values
were fresh; the READ was replicated by hold duration.

WHY IT COULD NOT BE SALVAGED AS "DIRECTIONALLY RIGHT": the bias is ALIGNED with
the hypothesis. ARM_OFF is the reference arm for every delta, and an unarmed agent
-- one with both selection-perturbing levers removed -- perseverates longer, so
its holds are systematically longer, its occupancy entropy is systematically
depressed relative to its per-commitment entropy, and EVERY d_* is inflated,
d_both included. There is therefore no sign-check available, and the corrected
sign is UNKNOWN. THIS IS A WITHDRAWAL, NOT A REVERSAL: do not restate "the levers
compound", "the levers cancel", or "safe to co-arm" in EITHER direction pending
this run. Note also that 699's own CANCEL branch (the destructive Factor-A x
Factor-B 689a signature) was never ruled out by an uncontaminated measurement
either -- the composition question is genuinely open both ways.

WHAT 699b CHANGES (autopsy routing_detail.requirements 1-8)
-----------------------------------------------------------
 1. FRESHNESS GATE. The committed class -- and every other per-selection quantity
    (consumed-summary divergence, lateral_pfc bias, CRF state, and both lever
    diagnostics) -- is recorded ONLY on a tick where E3 genuinely re-selected.
    DELIBERATE DEVIATION from the autopsy's literal instruction and from the
    v3_exq_785a reference pattern: 699b does NOT null the two latches. Nulling
    `_last_selected_trajectory` CHANGES SUBSTRATE BEHAVIOUR -- post_action_update
    (e3_selector.py:3224) falls back to it when _committed_trajectory is None (the
    ARC-016 deadlock fix) and runs on EVERY step via update_residue
    (agent.py:8006), so nulling it silently skips running-variance updates and the
    prediction_error / dynamic_precision metrics on non-E3 ticks with no live
    commitment. In a COMPOSITION experiment that perturbs the very selection
    dynamics under test, which would make 699b a different experiment rather than
    a repaired instrument. Nulling `last_score_diagnostics` is None-safe but also
    not inert (agent.py:9660 would fall back to authority defaults on non-E3
    ticks). Instead a private SENTINEL KEY is stamped into the diagnostics dict
    before every select_action: e3_selector.select() reassigns that dict WHOLESALE
    at :2452 with no early return, so the key survives iff select() did not run.
    Its sole ree_core reader (agent.py:9660) uses .get() with defaults and nothing
    iterates its keys, so the marker is fully INERT while giving an exact per-tick
    freshness signal. Same guarantee, zero substrate perturbation.
 2. n_fresh_select / n_latched / fresh_select_yield / replication_factor emitted
    per arm-seed. This is the single field whose absence made 699 unrecoverable:
    n_p2_ticks counts ENV STEPS and varies with episode termination (3.8x within
    one seed), so the replication factor could not even be estimated from the
    record.
 3. BOTH readouts emitted and kept distinct: committed_class_entropy_nats (PRIMARY,
    per-COMMITMENT -- the DV the verdict reads) and occupancy_class_entropy_nats
    (699's DV verbatim, hold-weighted). The pre-registered composition rule is
    applied to BOTH via _composition_verdict_from(), a matched-replay comparison
    over identical cells/seeds/thresholds differing only in sampling unit -- the
    method used on the 663 driver at ree-v3 5433e3ab1c. This makes the defect's
    size AND SIGN a measured quantity for the first time. The 699-instrumented
    verdict is DIAGNOSTIC CONTEXT ONLY and is not a finding.
    (The 663 replay measured this defect at sub-1% and sign-varying, but that does
    NOT generalise here: 663's DV is a continuous magnitude whose mean is nearly
    unchanged by replication and whose replication is near-uniform across arms, so
    it CANCELS in the contrast. 699's DV is an ENTROPY over a class histogram --
    replication reweights the distribution itself -- and its arms differ in HOLD
    DURATION, the very quantity doing the weighting. 663 measured the artifact
    where it cancels; 699 sits where it does not.)
 4. HOLD DURATION recorded per arm-seed (mean / median / max / full histogram).
    This directly tests the perseveration mechanism above: if ARM_OFF's holds are
    systematically longest, that confirms the alignment argument and retro-explains
    699; if flat, it bounds the distortion.
 5. selected_class_entropy_nats DROPPED. Under 699's instrumentation it equalled
    the primary DV to 6dp on ALL 12 arm-seeds and carried zero independent
    information -- an internal cross-check that looked like corroboration and was
    the same number twice. (When two supposedly independent readouts agree to the
    last decimal on every cell, that is a defect signature, not a validation.)
 6. C1 readiness battery UNCHANGED in predicates and thresholds -- it is sound and
    reproducing it exactly is the one part of 699 worth keeping. The only change is
    the DENOMINATOR of the active-frac gates, now n_fresh_select rather than
    n_p2_ticks, which is the honest denominator for a per-SELECTION gate; under the
    autopsy's finding that these gates fired on every E3 tick both denominators
    give the same saturated reading, so this is a correctness fix that does not
    move the gate. ONE gate is ADDED: C1(g) fresh-selection sufficiency, which
    699 could not have had because it counted no selections.
 7. supersedes V3-EXQ-699 and V3-EXQ-699a (see SUPERSEDES_QUEUE_IDS).
 8. Recording core stamped via pack_writer -> manifest_core.stamp_recording_core.

699b also inherits the precondition-DECLARATION fix (ree-v3 0bfbb42, _min_arm_count)
that 699a existed to exercise, since that fix is prospective and already on main.

WHY THIS, WHY NOW (the prong-map P-comp gate, user decision 2026-06-22).
The conversion ceiling's SELECTION FACE is now exhausted lever-by-lever:
  - Factor A (conflict-graded shortlist width) INERT (V3-EXQ-689a 0/3).
  - Factor B (gap-scaled commit-temperature) REFUTED at the selection face
    (V3-EXQ-689c FAIL; C_PRIMARY 1/3, C_GAPBLIND_B 0/3; flat-hot 0.6684 >
    gap-scaled-a0b1 0.6646) -- DROPPED from the full-stack arm.
  - MECH-448 demotion FAILS the committed-class C2 entropy lift ALONE
    (V3-EXQ-654i FAIL non_contributory; the cleanest conversion-ceiling
    instance: all C1 readiness met, demotion LIVE excluding 18.4 candidates,
    yet C2 did not lift).
  - MECH-449 Go/No-Go FAILS C2 ALONE (V3-EXQ-654j FAIL non_contributory;
    conversion_ceiling_persists_despite_active_nogo).
BOTH levers are face-validated alone (689d PASS demotion 0.938 vs 0.371;
689g PASS 3/3 Go/No-Go) and built as no-op-default flags in
ree_core/predictors/e3_selector.py (use_f_eligibility_demotion,
use_go_nogo_constitution). The campaign's endgame is a co-armed FULL-STACK
arm. The composition-readiness gate requires that within-selection-face lever
INTERACTIONS be CHARACTERIZED before co-arming: we KNOW Factor A x Factor B was
destructive at 689a (cancelled), but we do NOT know whether MECH-448 demotion
and MECH-449 Go/No-Go COMPOUND or CANCEL at C2. This experiment measures that
interaction directly.

THIS IS NOT (the re-derive-brake exemption, made explicit):
  - NOT another same-lever GAP-B behavioural letter (it does not sweep
    use_candidate_rule_field; that is a matched CONSTANT here).
  - NOT a Factor-B test (Factor B is refuted; not represented).
  - NOT a falsifier of MECH-309 / ARC-062 (those are not its claim_ids).
It tests the LEVER COMPOSITION (MECH-448 x MECH-449 interaction at C2), a
DISTINCT scientific question, so it is exempt from the conversion-ceiling
re-derive brake. claim_ids = [MECH-448, MECH-449] (both 0 substrate_ceiling
autopsies; both newly built + face-validated). experiment_purpose = diagnostic
(it CHARACTERIZES a lever interaction feeding the full-stack composition-
readiness gate; it PROMOTES NOTHING and weights no claim's governance
confidence). RUNNABLE NOW (uses only built + face-validated levers).

DESIGN -- a 2x2 (demotion {OFF,ON} x Go/No-Go {OFF,ON}) over a MATCHED stack
--------------------------------------------------------------------------
The swept variable is the LEVER COMBINATION, NOT use_candidate_rule_field. The
rule-apprehension differentiated signal (use_candidate_rule_field=True + the
matured/maintained CRF pool) is a MATCHED CONSTANT on ALL FOUR arms -- it is
the per-candidate differentiated source whose conversion-to-committed-class-
diversity the two levers gate. The shared selection-face conversion machinery
(SP-CEM, GAP-A e2_world_forward candidate source, modulatory-bias-selection-
authority + channel routing, the 569i top_k shortlist-then-modulate scaffold,
MECH-341 stratified, MECH-313 noise floor, V_s minimal, use_gated_policy,
use_lateral_pfc_analog with the SD-033a bias head un-zeroed + TRAINED in a
frozen-encoder P1 REINFORCE window, SD-056 all levers, use_dacc -> the MECH-260
recency-share vector that feeds the Go/No-Go perseveration axis) is MATCHED on
all four arms. The channel-adaptive (mean-relative) eligibility floor
(use_f_eligibility_adaptive_floor=True; V3-EXQ-689e PASS) is ON as a CONSTANT
(it is inert when demotion is OFF, and auto-calibrates the envelope when ON).

The ONLY two toggled flags are the two levers under test:
  ARM_OFF  : demotion OFF, Go/No-Go OFF -- the matched top_k-eligible +
             modulatory-arbitration baseline (neither lever).
  ARM_DEM  : demotion ON, Go/No-Go OFF -- MECH-448 alone (f_demotion overrides
             the top_k eligible-set construction with the graded DN envelope;
             channel-adaptive floor live).
  ARM_GNG  : demotion OFF, Go/No-Go ON -- MECH-449 alone (the active No-Go
             governs the top_k eligible set; the perseveration No-Go fed by the
             MECH-260 recency-share vector drops perseverated candidates).
  ARM_BOTH : demotion ON, Go/No-Go ON -- co-armed (the active No-Go governs the
             f_demotion-built eligible set). The composition cell.

WHAT MECH-448 DOES (e3_selector.py _f_eligibility_envelope + the "f_demotion"
mode of the shortlist-then-modulate block). F decides who is ELIGIBLE, not who
wins: F is renormalised against the competing field by a divisive-normalisation
analog with a CHANNEL-ADAPTIVE mean-relative share floor; the eligible set is an
F-RANK PREFIX (rank-preserving), F is REMOVED from the final argmin, and the
within-eligible _modulatory_accum arbitration picks the committed action.

WHAT MECH-449 DOES (e3_selector.py _go_nogo_eligibility_gate). Runs AFTER the
F-built eligible set (eligible_idx) is computed and BEFORE the within-eligible
arbitration: a bounded Go/No-Go opponency drops a candidate from the eligible
set when a non-F axis crosses its floor (here the perseveration axis, fed
ecologically by the MECH-260 dACC recency-share `suppression` vector with
use_dacc=True). Fail-open (gng_protect_min_eligible) never empties the eligible
set on a soft No-Go. The selection-face PASS V3-EXQ-689g validated this gate.

Dependent variable -- COMMITTED-CLASS diversity (the C2 DV, matched to 654i/j)
------------------------------------------------------------------------------
CODE-CONFIRMED (agent._candidate_world_summaries + lateral_pfc.compute_bias):
the per-candidate summaries the bias channels consume are keyed on the
candidate's FIRST action; compute_bias broadcasts a single rule_state across all
K candidates, so within a first-action class every candidate receives an
IDENTICAL rule bias. The levers move WHICH CLASS is committed (the committed-
class axis), NOT within-class representative selection. The matched readout is
committed-class entropy (PRIMARY DV). Within-class-representative entropy is a
SECONDARY NEGATIVE CONTROL (expected ~null).

Phases / budget
---------------
P0 (200 ep, e2 TRAINED online via SD-056 contrastive; bias head NOT trained;
   the CRF field matures across episodes via crf_persist + the 666c-validated
   maintenance levers).
P1 (90 ep, encoder FROZEN, bias head TRAINED via outcome-coupled REINFORCE;
   field continues to mature + maintain): the GAP-D trained-bias-head window.
P2 (60 ep, all FROZEN; instrumentation ON incl the per-arm demotion + Go/No-Go
   diagnostics): the behavioural measurement window.
Budget: 4 arms x 3 seeds x 350 ep x 200 steps = 840k steps total. Prefer cloud
over the Mac (the 689/654 lineage; prefer-cloud).

Pre-registered acceptance criteria (the composition characterization)
---------------------------------------------------------------------
  C1 (READINESS / non-vacuity -- protects against a false interaction; ANY C1
      fail self-routes substrate_not_ready_requeue, NEVER a verdict):
     (a) committed-class axis exercisable: frac_pre_ge2 > FRAC_PRE_GE2_FLOOR on
         a majority (>= 2/3) of seeds in ALL FOUR arms.
     (b) GAP-A divergence real: consumed_summary_pairwise_dist_mean >
         CONSUMED_SPREAD_FLOOR (and bounded) on a majority of seeds in ALL arms.
     (c) the matched rule field MATURED: per-tick crf_n_active >=
         CRF_N_ACTIVE_FLOOR on >= CRF_FRAC_ACTIVE_FLOOR of P2 ticks AND >=
         CRF_MIN_MINTED distinct rules minted, on a majority of seeds in ALL
         arms (the differentiated source is present so the levers have something
         to convert).
     (d) PROPAGATION non-vacuity: the matched trained rule bias REACHES
         committed action -- mean_lateral_pfc_bias_abs > PROP_NONVAC_FLOOR on a
         majority of seeds in ALL arms (and a within-arm rule_state
         counterfactual confirms zeroing rule_state changes the bias).
     (e) MECH-448 DEMOTION non-vacuity, PER-ARM CONDITIONAL: on the
         DEMOTION-ARMED arms (ARM_DEM, ARM_BOTH) the f_demotion envelope is
         active on >= DEMOTION_ACTIVE_FRAC_FLOOR of P2 ticks AND actually
         EXCLUDES (mean f_eligibility_excluded_count > EXCLUDED_COUNT_FLOOR) on
         a majority of seeds. An all-admit envelope (excluded_count==0, a flat-F
         pool) means the demotion did nothing -> the composition cell is vacuous
         -> substrate_not_ready_requeue. (Demotion is OFF by design on ARM_OFF /
         ARM_GNG -- inactivity there is correct, not vacuous.)
     (f) MECH-449 ACTIVE NO-GO non-vacuity, PER-ARM CONDITIONAL: on the
         GO/NO-GO-ARMED arms (ARM_GNG, ARM_BOTH) the constitution is active on
         >= NOGO_ACTIVE_FRAC_FLOOR of P2 ticks AND actually SUPPRESSES (mean
         go_nogo_n_safety_nogo + go_nogo_n_soft_applied > NOGO_SUPPRESSED_FLOOR;
         the perseveration No-Go fed by the MECH-260 recency-share vector) on a
         majority of seeds. An inert No-Go means the opponency leg did nothing
         -> vacuous -> substrate_not_ready_requeue. (Go/No-Go is OFF by design on
         ARM_OFF / ARM_DEM.)
  C_INTERACTION (PRIMARY characterization; only read when C1 holds): per-seed
     paired committed-class entropy deltas vs ARM_OFF --
        d_dem  = E(ARM_DEM)  - E(ARM_OFF)
        d_gng  = E(ARM_GNG)  - E(ARM_OFF)
        d_both = E(ARM_BOTH) - E(ARM_OFF)
        interaction = d_both - (d_dem + d_gng)   (super-additive if > 0)
     Three-way verdict (>= COMPOSITION_MIN_SEEDS of 3 seeds; COMPOSITION_MARGIN
     guards noise):
        COMPOUND -> d_both >= max(d_dem, d_gng) + COMPOSITION_MARGIN AND
                    d_both >= COMPOSITION_LIFT_MARGIN (the combination converts
                    where each lever alone does not -> SAFE to co-arm; synergy).
        CANCEL   -> d_both <= min(d_dem, d_gng) - COMPOSITION_MARGIN OR
                    d_both <= -COMPOSITION_MARGIN (the combination is WORSE than
                    either lever alone / below baseline -- a DESTRUCTIVE
                    interaction, the Factor-A x Factor-B 689a signature ->
                    do NOT co-arm both in the full stack; drop one).
        NEUTRAL  -> otherwise (no collision; co-armable, composition adds nothing
                    on this bank -- SAFE to co-arm, no synergy).

Outcome map (diagnostic semantics; NO weakens -- characterizes, does not falsify)
---------------------------------------------------------------------------------
  PASS = C1 holds (the characterization is trustworthy -- both levers genuinely
         engaged in their arms, class axis + GAP-A + matured field present) AND
         the interaction is non-degenerate (arms not bit-identical). The
         interpretation_label carries the composition verdict
         (levers_compound / levers_cancel / levers_neutral). This is the
         composition-readiness CHARACTERIZATION the full-stack arm consumes:
            levers_compound / levers_neutral -> demotion + Go/No-Go are
              composition-ready (co-arm both in the full stack).
            levers_cancel -> the full-stack arm must NOT co-arm both (drop one,
              mirroring Factor A x Factor B).
  FAIL = C1 fails (a lever was vacuous on its armed arm, or the class axis /
         GAP-A / matured field was absent) OR the four arms are bit-identical
         (the interaction could not be measured) -> substrate_not_ready_requeue.
         NOT a falsification of anything. Re-queue at an adequate substrate.

claim_ids = [MECH-448, MECH-449] (the two levers whose interaction this
characterizes). experiment_purpose = diagnostic. PROMOTES NOTHING; weights no
claim's governance confidence (diagnostics are scoring-excluded). The result
feeds conversion_ceiling_campaign:FULLSTACK composition-readiness.

See REE_assembly/evidence/planning/conversion_ceiling_prong_map.md (node P-comp),
    REE_assembly/evidence/planning/conversion_ceiling_campaign_plan.md,
    REE_assembly/docs/architecture/mech_448_f_eligibility_demotion.md,
    REE_assembly/docs/architecture/mech_449_go_nogo_constitution.md,
    REE_assembly/evidence/planning/arc_107_selector_constitution_design_2026-06-20.md,
    experiments/v3_exq_689d_mech448_f_eligibility_demotion_falsifier.py (demotion face PASS),
    experiments/v3_exq_689g_mech449_go_nogo_conversion_falsifier.py (Go/No-Go face PASS),
    experiments/v3_exq_654i_arc062_gapb_rule_apprehension_behavioural_falsifier.py (demotion C2 FAIL),
    experiments/v3_exq_654j_arc062_gapb_rule_apprehension_nogo_behavioural_falsifier.py (Go/No-Go C2 FAIL; structure ported here),
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-689a_2026-06-20.json (Factor A x Factor B cancellation precedent).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import compute_arm_fingerprint, reset_all_rng
from experiments._lib.readiness_anchor import assert_anchor_reachable
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


# --- validate_experiments lint exemptions -------------------------------------
# BOTH lints below detect the EXACT defect this experiment exists to repair, and
# both fire here as FALSE POSITIVES: they pattern-match on a literal
# `agent.e3.<attr> = None` clear immediately preceding select_action(), and 699b
# deliberately uses a SENTINEL KEY instead (see the module docstring, item 1).
#
# The remedy each lint prescribes is implemented in full:
#   * accumulation is gated on a genuine fresh E3 selection (`fresh_select`),
#   * n_fresh_select / n_latched / fresh_select_yield / replication_factor are
#     emitted per arm-seed,
#   * the hold-weighted quantity is emitted TOO, kept distinct
#     (occupancy_class_entropy_nats vs committed_class_entropy_nats), with the
#     pre-registered verdict rule applied to both for a matched comparison.
# The ONLY deviation is the freshness MECHANISM, and it is deliberate: nulling
# `_last_selected_trajectory` would change substrate behaviour (post_action_update
# at e3_selector.py:3224 falls back to it for running-variance updates on every
# step via agent.py:8006), which is unacceptable in a composition experiment.
# The sentinel key is substrate-inert and gives the identical guarantee, because
# e3_selector.select() reassigns the diagnostics dict WHOLESALE at :2452.
#
# These markers must NOT be read as "699b still carries the 699 defect" -- the
# opposite is true. Re-audit if the freshness mechanism ever changes.
_FRESH_SELECT_EXEMPT_REASON = (
    "Freshness is enforced via a substrate-inert SENTINEL KEY stamped into "
    "agent.e3.last_score_diagnostics before every select_action(), not via a "
    "`= None` clear: e3_selector.select() reassigns that dict wholesale at :2452, "
    "so the key survives iff select() did not run. A `= None` clear of "
    "_last_selected_trajectory would suppress running-variance updates on non-E3 "
    "ticks (e3_selector.py:3224 via agent.py:8006) and perturb the selection "
    "dynamics under test. All accumulation is fresh-gated; n_fresh_select / "
    "n_latched / fresh_select_yield are emitted per arm-seed; both the "
    "per-commitment and the hold-weighted occupancy entropies are emitted and kept "
    "distinct. See failure_autopsy_V3-EXQ-699_2026-07-20.json."
)
E3_DIAGNOSTICS_STALENESS_EXEMPT = _FRESH_SELECT_EXEMPT_REASON
E3_HOLD_WEIGHTED_READOUT_EXEMPT = _FRESH_SELECT_EXEMPT_REASON

EXPERIMENT_TYPE = "v3_exq_699b_pcomp_demotion_x_gonogo_fresh_select"
QUEUE_ID = "V3-EXQ-699b"
# INSTRUMENT REPAIR of V3-EXQ-699 (failure_autopsy_V3-EXQ-699_2026-07-20.json,
# REE_assembly ac2fb64028). The 699 PASS and its C1 readiness battery STAND; the
# `levers_compound` composition finding is WITHDRAWN. 699's primary DV was
# accumulated from the action RETURNED by agent.select_action once per ENV STEP,
# and on a non-E3 tick ree_core/agent.py:5430 returns the HELD action before
# e3.select() is reached -- so the DV was hold-duration-weighted OCCUPANCY
# entropy standing in for the per-COMMITMENT construct MECH-448/449 act on. The
# bias is ALIGNED with the hypothesis (ARM_OFF is the reference for every delta
# and an unarmed agent perseverates longer), so no sign-check is available and
# the corrected sign is UNKNOWN. This is a WITHDRAWAL, not a reversal: 699b must
# not be read as confirming or refuting `levers_compound` in advance.
# Same scientific question, corrected instrumentation -> alphabetic suffix per
# CLAUDE.md EXQ versioning ("bug fix = ... implementation was wrong (broken
# instrumentation)"; "when in doubt: new letter"), matching the 708 -> 708a
# precedent. The autopsy records a deliberate dissent from an earlier chip that
# asserted a new EXQ number.
SUPERSEDES_RUN_ID = (
    "v3_exq_699_pcomp_demotion_x_gonogo_composition_20260623T053755Z_v3"
)
# 699a is a FAITHFUL re-run of the unchanged 699 driver and reproduces every
# defect above; it was in flight (claimed by ree-cloud-3) when 699b was authored.
# If it lands a manifest, it is superseded too -- set at runtime below.
SUPERSEDES_QUEUE_IDS: List[str] = ["V3-EXQ-699", "V3-EXQ-699a"]
CLAIM_IDS: List[str] = ["MECH-448", "MECH-449"]
EXPERIMENT_PURPOSE = "diagnostic"

# CRF-gate calibration amend levers (crf-availability-maintenance at the CRF locus;
# the now-working CRF stack; ported verbatim from V3-EXQ-654j, matched on all arms).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# Within-class-representative signature horizon (SECONDARY negative control).
H_SIGNATURE = 3

# Private key stamped into agent.e3.last_score_diagnostics before every
# select_action(). e3_selector.select() reassigns that dict wholesale
# (e3_selector.py:2452), so the key survives IFF select() did not run -- an exact,
# substrate-inert per-tick freshness signal. Nothing in ree_core iterates the dict
# (its sole reader, agent.py:9660, uses .get() with defaults), so the extra key is
# inert. Underscore-prefixed and never emitted to the manifest.
_STALE_MARKER_KEY = "_exq699b_stale_marker"

# C1 readiness floor on the fresh-selection yield. A run whose E3 cadence produced
# too few genuine selections cannot support a per-commitment entropy at all; this
# self-routes substrate_not_ready_requeue rather than a composition verdict.
MIN_FRESH_SELECT_PER_CELL = 100

# C_INTERACTION (PRIMARY characterization): composition verdict thresholds.
# COMPOSITION_MARGIN guards seed noise; COMPOSITION_LIFT_MARGIN is the absolute
# committed-class entropy lift the co-armed cell must clear to count as COMPOUND.
COMPOSITION_MARGIN = 0.05
COMPOSITION_LIFT_MARGIN = 0.05
COMPOSITION_MIN_SEEDS = 2  # of 3

# C1(a) readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# C1(b) readiness: GAP-A consumed-summary divergence (649 statistic + 643a ceiling).
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# C1(c) readiness: matured rule field (minted distinct rules + fired on matured P2 ticks).
CRF_MIN_MINTED = 2
CRF_N_ACTIVE_FLOOR = 1
CRF_FRAC_ACTIVE_FLOOR = 0.30
CRF_DIST_FLOOR = 1e-3

# C1(d) PROPAGATION non-vacuity: the matched trained rule bias reaches committed action.
PROP_NONVAC_FLOOR = 1e-3

# Only classes committed to at least this many P2 ticks feed the unweighted mean
# within-class entropy (secondary negative control).
MIN_TICKS_PER_CLASS = 5

MIN_SEEDS_FOR_PASS = 2  # of 3

# C1(e) MECH-448 DEMOTION non-vacuity: the demotion lever is LIVE on the
# demotion-ARMED arms (active + actually excludes on the divergent pool). An
# all-admit envelope (excluded_count == 0; flat-F) is vacuous.
DEMOTION_ACTIVE_FRAC_FLOOR = 0.8
EXCLUDED_COUNT_FLOOR = 0.0

# C1(f) MECH-449 ACTIVE NO-GO non-vacuity: the Go/No-Go constitution is LIVE on the
# Go/No-Go-ARMED arms (active + actually suppresses). An inert No-Go is vacuous.
NOGO_ACTIVE_FRAC_FLOOR = 0.8
NOGO_SUPPRESSED_FLOOR = 0.0


# =============================================================================
# THE SHIPPED C1 PREDICATES (single definition, used live AND by the guards)
# =============================================================================
# Hoisted to module level for ONE reason: `assert_anchor_reachable` requires
# `score_fn` to be THE SHIPPED PREDICATE, not a re-implementation -- the defect
# the guard exists to catch IS a mis-specified predicate, so a guard scoring a
# copy proves nothing (experiments/_lib/readiness_anchor.py rule 1).
#
# NOTHING about C1 changes here. Each function is the verbatim boolean that was
# previously written inline in `_run_seed_arm`, and every threshold is the same
# module constant. The autopsy requires 699's C1 battery be reproduced EXACTLY
# (it is the one part of 699 judged sound and threshold-invariant to the 699
# defect), so this is a pure code move.
#
# They take a MAPPING keyed by the manifest row's own field names, which is what
# makes the frozen 699 cells below usable as reference input without translation.
# `_run_seed_arm` calls them with the UNROUNDED locals (the row itself stores
# round(...,6) values); passing raw preserves the shipped comparison bit-for-bit.

def _pred_class_axis_exercisable(c: Dict[str, Any]) -> bool:
    """C1(a): committed-class axis exercisable on this cell."""
    return bool(c["frac_pre_ge2"] > FRAC_PRE_GE2_FLOOR)


def _pred_gapa_divergence(c: Dict[str, Any]) -> bool:
    """C1(b): GAP-A consumed-summary spread non-degenerate AND numerically bounded."""
    return bool(
        c["consumed_summary_pairwise_dist_mean"] > CONSUMED_SPREAD_FLOOR
        and c["consumed_summary_pairwise_dist_max"] < CONSUMED_MAGNITUDE_CEIL
    )


def _pred_gapa_bounded(c: Dict[str, Any]) -> bool:
    """`gapa_consumed_summary_bounded`, per cell.

    The shipped anchor is `max(dist_max over all cells) < CEIL`, which is
    identically `all(cells under CEIL)` -- so scoring per cell at threshold 1.0
    reproduces it exactly rather than approximating it.
    """
    return bool(c["consumed_summary_pairwise_dist_max"] < CONSUMED_MAGNITUDE_CEIL)


def _pred_crf_differentiated(c: Dict[str, Any]) -> bool:
    """C1(c): rule field minted distinct rules AND fired on enough matured P2 ticks."""
    return bool(
        c["crf_n_minted_total"] >= CRF_MIN_MINTED
        and c["crf_frac_active_ge_floor"] >= CRF_FRAC_ACTIVE_FLOOR
    )


def _pred_prop_non_vacuous(c: Dict[str, Any]) -> bool:
    """C1(d): the per-candidate bias reaches committed action."""
    return bool(c["mean_lateral_pfc_bias_abs"] > PROP_NONVAC_FLOOR)


def _pred_demotion_non_vacuous(c: Dict[str, Any]) -> bool:
    """C1(e): MECH-448 demotion envelope is active AND actually excludes."""
    return bool(
        c["f_eligibility_demotion_active_frac"] >= DEMOTION_ACTIVE_FRAC_FLOOR
        and c["f_eligibility_excluded_count_mean"] > EXCLUDED_COUNT_FLOOR
    )


def _pred_nogo_non_vacuous(c: Dict[str, Any]) -> bool:
    """C1(f): MECH-449 Go/No-Go constitution is active AND actually suppresses."""
    return bool(
        c["go_nogo_active_frac"] >= NOGO_ACTIVE_FRAC_FLOOR
        and c["go_nogo_suppressed_per_tick_mean"] > NOGO_SUPPRESSED_FLOOR
    )


def _pred_fresh_select_sufficient(c: Dict[str, Any]) -> bool:
    """C1(g), NEW IN 699b: enough genuine E3 selections to support a per-commitment DV."""
    return bool(c["n_fresh_select"] >= MIN_FRESH_SELECT_PER_CELL)


# =============================================================================
# ANCHOR-REACHABILITY GUARDS (discharging the warning INHERITED from 699)
# =============================================================================
# `validate_experiments.py --strict` warned that this driver declares anchor-kind
# readiness preconditions and self-routes on them to `substrate_not_ready_requeue`
# -- a consequential label -- without ever asserting those gates are REACHABLE by
# the control they claim to score. That warning came across from the 699 driver
# unchanged; it is NOT introduced by this repair, and the queueing session
# deliberately left it un-exempted because it had not been discharged. It is
# discharged HERE, with a replay rather than a marker.
#
# WHY NEITHER MARKER WAS THE RIGHT ANSWER (readiness_anchor.py, "ALREADY-RAN
# DEFECTS"): ANCHOR_REACHABILITY_EXEMPT is only for a predicate that IS the
# degeneracy definition, which none of these are -- they are hand-written floors
# over continuous statistics, exactly the class 778d belongs to.
# ANCHOR_REACHABILITY_SUPERSEDED is only for an ALREADY-RAN script whose repair
# must live in a successor letter; 699b has not run, so an in-place guard alters
# no recorded evidence.
#
# THE REFERENCE. V3-EXQ-699 ran these same seven anchors at full scale and MET
# all seven -- so they are demonstrably reachable rather than unmeetable by
# construction, and this block turns that empirical fact into a setup-time
# assertion instead of a note in a queue entry. Frozen verbatim from
# REE_assembly/evidence/experiments/
#   v3_exq_699_pcomp_demotion_x_gonogo_composition_20260623T053755Z_v3.json
# (12 arm-seed cells, 4 arms x seeds 42/43/44), so the guard needs no compute and
# cannot drift with the substrate.
#
# TWO RECORDED-VALUE CAVEATS, both in the SAFE direction:
#   * `f_eligibility_demotion_active_frac` / `go_nogo_active_frac` were computed
#     in 699 against an n_p2_ticks (env-step) denominator; 699b divides by
#     n_fresh_select, which is strictly smaller, so the same run yields a LARGER
#     frac. The recorded values are already 1.0 (the ceiling), so the replay is
#     conservative and the change cannot flip these anchors.
#   * The manifest stores round(...,6) values while `_run_seed_arm` scores the
#     unrounded locals. Every recorded margin here is orders of magnitude wider
#     than 1e-6, so rounding is immaterial to the replay.
#
# ENCODING. Each anchor's shipped `met` is a per-arm-majority rule -- min over
# arms of the count of satisfying seeds >= MIN_SEEDS_FOR_PASS -- so the reference
# cells are the ARMS, scored by `_arm_majority(<shipped leaf predicate>)`, at
# threshold 1.0 (every arm must hold). That reproduces the shipped rule EXACTLY
# rather than approximating it with a pooled per-seed fraction, which would be
# strictly weaker and could certify reachable a gate the real rule cannot meet.
# `margin_cells` is therefore 0 throughout and that zero is intended, not
# overlooked (readiness_anchor.py rule 4): at threshold 1.0 any positive margin is
# arithmetically unsatisfiable, so headroom has to be read at the SEED level. It
# is reported per anchor below.
#
# NOT GUARDED, DELIBERATELY: `fresh_e3_selection_sufficiency_all_arms` (C1(g)).
# It is NEW in 699b and 699 recorded NO fresh-selection count, so no frozen
# reference for it exists. The only quantity derivable from the 699 record is
# n_p2_ticks / e3_steps_per_tick, which is a strict UPPER bound on n_fresh_select
# -- a guard built on it could certify reachable a gate the true value misses, the
# precise failure `assert_anchor_reachable` exists to prevent. See the
# C1(g) headroom note below, which is reported to the operator rather than
# asserted.

_REFERENCE_699_C1_CELLS: Dict[str, Tuple[Dict[str, Any], ...]] = {
    "ARM_OFF": (
        {
            "seed": 42,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.086386,
            "consumed_summary_pairwise_dist_max": 0.129977,
            "crf_n_minted_total": 13,
            "crf_frac_active_ge_floor": 0.844635,
            "mean_lateral_pfc_bias_abs": 0.05777158,
            "f_eligibility_demotion_active_frac": 0.0,
            "f_eligibility_excluded_count_mean": 0.0,
            "go_nogo_active_frac": 0.0,
            "go_nogo_suppressed_per_tick_mean": 0.0,
            "n_p2_ticks": 1165,
        },
        {
            "seed": 43,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.074105,
            "consumed_summary_pairwise_dist_max": 0.141069,
            "crf_n_minted_total": 16,
            "crf_frac_active_ge_floor": 0.969228,
            "mean_lateral_pfc_bias_abs": 0.08846155,
            "f_eligibility_demotion_active_frac": 0.0,
            "f_eligibility_excluded_count_mean": 0.0,
            "go_nogo_active_frac": 0.0,
            "go_nogo_suppressed_per_tick_mean": 0.0,
            "n_p2_ticks": 10139,
        },
        {
            "seed": 44,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.080774,
            "consumed_summary_pairwise_dist_max": 0.122618,
            "crf_n_minted_total": 8,
            "crf_frac_active_ge_floor": 0.817143,
            "mean_lateral_pfc_bias_abs": 0.04061345,
            "f_eligibility_demotion_active_frac": 0.0,
            "f_eligibility_excluded_count_mean": 0.0,
            "go_nogo_active_frac": 0.0,
            "go_nogo_suppressed_per_tick_mean": 0.0,
            "n_p2_ticks": 525,
        },
    ),
    "ARM_DEM": (
        {
            "seed": 42,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.052343,
            "consumed_summary_pairwise_dist_max": 0.077569,
            "crf_n_minted_total": 16,
            "crf_frac_active_ge_floor": 0.87429,
            "mean_lateral_pfc_bias_abs": 0.10000001,
            "f_eligibility_demotion_active_frac": 1.0,
            "f_eligibility_excluded_count_mean": 16.68608,
            "go_nogo_active_frac": 0.0,
            "go_nogo_suppressed_per_tick_mean": 0.0,
            "n_p2_ticks": 1408,
        },
        {
            "seed": 43,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.004852,
            "consumed_summary_pairwise_dist_max": 0.007972,
            "crf_n_minted_total": 14,
            "crf_frac_active_ge_floor": 0.949167,
            "mean_lateral_pfc_bias_abs": 0.04298512,
            "f_eligibility_demotion_active_frac": 1.0,
            "f_eligibility_excluded_count_mean": 16.170917,
            "go_nogo_active_frac": 0.0,
            "go_nogo_suppressed_per_tick_mean": 0.0,
            "n_p2_ticks": 12000,
        },
        {
            "seed": 44,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.15336,
            "consumed_summary_pairwise_dist_max": 0.256346,
            "crf_n_minted_total": 15,
            "crf_frac_active_ge_floor": 0.879678,
            "mean_lateral_pfc_bias_abs": 0.09172368,
            "f_eligibility_demotion_active_frac": 1.0,
            "f_eligibility_excluded_count_mean": 14.814623,
            "go_nogo_active_frac": 0.0,
            "go_nogo_suppressed_per_tick_mean": 0.0,
            "n_p2_ticks": 1737,
        },
    ),
    "ARM_GNG": (
        {
            "seed": 42,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.102041,
            "consumed_summary_pairwise_dist_max": 0.150264,
            "crf_n_minted_total": 16,
            "crf_frac_active_ge_floor": 0.896194,
            "mean_lateral_pfc_bias_abs": 0.0501911,
            "f_eligibility_demotion_active_frac": 0.0,
            "f_eligibility_excluded_count_mean": 0.0,
            "go_nogo_active_frac": 1.0,
            "go_nogo_suppressed_per_tick_mean": 0.694637,
            "n_p2_ticks": 1156,
        },
        {
            "seed": 43,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.156846,
            "consumed_summary_pairwise_dist_max": 0.228805,
            "crf_n_minted_total": 8,
            "crf_frac_active_ge_floor": 0.986083,
            "mean_lateral_pfc_bias_abs": 0.09990293,
            "f_eligibility_demotion_active_frac": 0.0,
            "f_eligibility_excluded_count_mean": 0.0,
            "go_nogo_active_frac": 1.0,
            "go_nogo_suppressed_per_tick_mean": 1.9795,
            "n_p2_ticks": 12000,
        },
        {
            "seed": 44,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.246777,
            "consumed_summary_pairwise_dist_max": 0.371948,
            "crf_n_minted_total": 10,
            "crf_frac_active_ge_floor": 0.828633,
            "mean_lateral_pfc_bias_abs": 0.08216546,
            "f_eligibility_demotion_active_frac": 0.0,
            "f_eligibility_excluded_count_mean": 0.0,
            "go_nogo_active_frac": 1.0,
            "go_nogo_suppressed_per_tick_mean": 1.587852,
            "n_p2_ticks": 461,
        },
    ),
    "ARM_BOTH": (
        {
            "seed": 42,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.054398,
            "consumed_summary_pairwise_dist_max": 0.081006,
            "crf_n_minted_total": 16,
            "crf_frac_active_ge_floor": 0.804196,
            "mean_lateral_pfc_bias_abs": 0.09988336,
            "f_eligibility_demotion_active_frac": 1.0,
            "f_eligibility_excluded_count_mean": 18.615385,
            "go_nogo_active_frac": 1.0,
            "go_nogo_suppressed_per_tick_mean": 2.191808,
            "n_p2_ticks": 1001,
        },
        {
            "seed": 43,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.008482,
            "consumed_summary_pairwise_dist_max": 0.01315,
            "crf_n_minted_total": 10,
            "crf_frac_active_ge_floor": 0.928508,
            "mean_lateral_pfc_bias_abs": 0.10000001,
            "f_eligibility_demotion_active_frac": 1.0,
            "f_eligibility_excluded_count_mean": 28.446828,
            "go_nogo_active_frac": 1.0,
            "go_nogo_suppressed_per_tick_mean": 12.40563,
            "n_p2_ticks": 11190,
        },
        {
            "seed": 44,
            "frac_pre_ge2": 1.0,
            "consumed_summary_pairwise_dist_mean": 0.112685,
            "consumed_summary_pairwise_dist_max": 0.164194,
            "crf_n_minted_total": 13,
            "crf_frac_active_ge_floor": 0.824611,
            "mean_lateral_pfc_bias_abs": 0.05125697,
            "f_eligibility_demotion_active_frac": 1.0,
            "f_eligibility_excluded_count_mean": 20.652051,
            "go_nogo_active_frac": 1.0,
            "go_nogo_suppressed_per_tick_mean": 5.123055,
            "n_p2_ticks": 707,
        },
    ),
}

_REFERENCE_699_SOURCE = (
    "V3-EXQ-699 (v3_exq_699_pcomp_demotion_x_gonogo_composition_20260623T053755Z_v3), "
    "full-scale run that MET all seven of these anchors"
)

# Arm groupings, matching each anchor's shipped scope. C1(e)/C1(f) are declared
# over the ARMED arms only -- demotion is OFF by design on ARM_OFF/ARM_GNG and
# Go/No-Go on ARM_OFF/ARM_DEM, so inactivity there is correct, not vacuous.
_REF_ALL_ARMS = tuple(_REFERENCE_699_C1_CELLS[a]
                      for a in ("ARM_OFF", "ARM_DEM", "ARM_GNG", "ARM_BOTH"))
_REF_DEMOTION_ARMED = tuple(_REFERENCE_699_C1_CELLS[a] for a in ("ARM_DEM", "ARM_BOTH"))
_REF_GNG_ARMED = tuple(_REFERENCE_699_C1_CELLS[a] for a in ("ARM_GNG", "ARM_BOTH"))
_REF_ALL_CELLS = tuple(c for arm in _REF_ALL_ARMS for c in arm)


def _arm_majority(pred):
    """Lift a shipped per-seed predicate to the shipped per-arm-majority rule.

    A thin composition, NOT a re-implementation: the leaf is the same callable
    `_run_seed_arm` scores live cells with, and the count/threshold form is the
    one `_min_arm_count` applies in the analysis.
    """
    def _score(arm_cells) -> bool:
        return sum(1 for c in arm_cells if pred(c)) >= MIN_SEEDS_FOR_PASS
    return _score


def assert_c1_anchors_reachable() -> List[Dict[str, Any]]:
    """Replay the frozen 699 cells through the SHIPPED C1 predicates at setup.

    Raises AnchorUnreachable (from readiness_anchor) before any compute is spent
    if any gate is unmeetable by the control that already met it.
    """
    payloads: List[Dict[str, Any]] = []
    for anchor_name, reference_cells, score_fn in (
        ("committed_class_axis_exercisable_all_arms",
         _REF_ALL_ARMS, _arm_majority(_pred_class_axis_exercisable)),
        ("gapa_consumed_summary_divergence_all_arms",
         _REF_ALL_ARMS, _arm_majority(_pred_gapa_divergence)),
        ("gapa_consumed_summary_bounded",
         _REF_ALL_CELLS, _pred_gapa_bounded),
        ("rule_field_differentiated_and_matured_all_arms",
         _REF_ALL_ARMS, _arm_majority(_pred_crf_differentiated)),
        ("propagation_non_vacuity_bias_reaches_committed_action_all_arms",
         _REF_ALL_ARMS, _arm_majority(_pred_prop_non_vacuous)),
        ("mech448_demotion_lever_live_and_excluding_demotion_armed_arms",
         _REF_DEMOTION_ARMED, _arm_majority(_pred_demotion_non_vacuous)),
        ("mech449_active_nogo_live_and_suppressing_gng_armed_arms",
         _REF_GNG_ARMED, _arm_majority(_pred_nogo_non_vacuous)),
    ):
        payloads.append(assert_anchor_reachable(
            anchor_name=anchor_name,
            reference_cells=reference_cells,
            score_fn=score_fn,
            # 1.0 = EVERY arm must clear its own majority, which is exactly the
            # shipped `min(per-arm satisfying-seed counts) >= MIN_SEEDS_FOR_PASS`.
            # For `gapa_consumed_summary_bounded` the cells are seeds, not arms,
            # and 1.0 is exactly `max(dist_max) < CEIL`.
            threshold=1.0,
            reference_source=_REFERENCE_699_SOURCE,
            # See the encoding note above: at threshold 1.0 a positive
            # margin_cells is unsatisfiable by arithmetic, so seed-level headroom
            # is reported instead of asserted.
            margin_cells=0,
        ))

    # Seed-level headroom, printed because the arm-level guard cannot express it.
    # Two anchors clear by the BARE minimum and are the ones to watch if 699b ever
    # self-routes substrate_not_ready_requeue:
    #   * gapa_divergence: ARM_DEM and ARM_BOTH each had 2 of 3 seeds divergent on
    #     699 (seed 43 fell below CONSUMED_SPREAD_FLOOR in both), i.e. exactly
    #     MIN_SEEDS_FOR_PASS with zero spare seeds.
    #   * gapa_consumed_summary_bounded is the opposite shape -- a VACUOUS anchor
    #     in the readiness_anchor.py "mirror failure" sense: the recorded maximum
    #     is 0.371948 against a 1.0e6 ceiling. It is a numerical-explosion
    #     denominator guard (643a), so a wide margin is intended; recorded here so
    #     it is not mistaken for a readiness gate. NOT retuned: the autopsy
    #     requires 699's C1 battery be reproduced exactly.
    for p in payloads:
        print(
            f"anchor_reachable: {p['anchor_name']} "
            f"{p['n_reference_scored_true']}/{p['n_reference_cells']} "
            f"(gate {p['required_score']:.4f})",
            flush=True,
        )
    return payloads


SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 200
P1_BIAS_TRAIN_EPISODES = 90
P2_MEASUREMENT_EPISODES = 60
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 30

# Matched-stack lever constants (identical on ALL FOUR arms).
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
USE_MODULATORY_CHANNEL_ROUTING = True
MODULATORY_CHANNEL_ROUTE_SOURCE = "cand_world_summary"
MODULATORY_CHANNEL_ROUTE_WEIGHT = 1.0
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6
# 569i TOP-K SHORTLIST scaffold -- the eligible-set machinery the Go/No-Go gate
# governs and that f_demotion overrides when demotion is ON. MATCHED CONSTANT.
USE_MODULATORY_SHORTLIST_THEN_MODULATE = True
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_SHORTLIST_K = 3
# MECH-448 demotion envelope params (consulted only when demotion is ARMED).
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30   # legacy absolute floor (ignored under the adaptive floor)
F_ELIGIBILITY_DN_SIGMA = 0.0
# CHANNEL-ADAPTIVE (mean-relative) envelope floor -- ON as a MATCHED CONSTANT
# (inert when demotion OFF; auto-calibrates the envelope when demotion ON). 689e PASS.
USE_F_ELIGIBILITY_ADAPTIVE_FLOOR = True
F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR = 1.0
# MECH-449 Go/No-Go constitution params (consulted only when Go/No-Go is ARMED).
# use_dacc is a MATCHED CONSTANT: it feeds the MECH-260 perseveration No-Go axis
# ecologically AND its dACC score-bias is identical across arms.
USE_DACC = True
GNG_PERSEVERATION_FLOOR = 0.5
GNG_SAFETY_FLOOR = 0.5
GNG_PROTECT_MIN_ELIGIBLE = 1
MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

# use_candidate_rule_field is a MATCHED CONSTANT (the differentiated conversion source).
USE_CANDIDATE_RULE_FIELD = True

# SD-056 online e2 training (mirror V3-EXQ-649/654j harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
SD056_MULTISTEP_CONTRASTIVE = True
SD056_CONTRASTIVE_HORIZON = 5
SD056_OUTPUT_NORM_CLAMP = True
SD056_OUTPUT_NORM_CLAMP_RATIO = 2.0

# P1 bias-head REINFORCE training (mirror V3-EXQ-598b/654j).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9


# IDENTICAL env to V3-EXQ-654j / 654 / 614e (the arc_062 GAP-B bank).
ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)


# The 2x2: the ONLY toggled flags are demotion_on x gng_on. use_candidate_rule_field
# + the channel-adaptive floor + use_dacc + the whole conversion stack are matched.
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_OFF",
        "label": "neither_lever_topk_eligible_modulatory_baseline",
        "demotion_on": False,
        "gng_on": False,
    },
    {
        "arm_id": "ARM_DEM",
        "label": "mech448_demotion_only_graded_dn_envelope",
        "demotion_on": True,
        "gng_on": False,
    },
    {
        "arm_id": "ARM_GNG",
        "label": "mech449_go_nogo_only_active_opponency",
        "demotion_on": False,
        "gng_on": True,
    },
    {
        "arm_id": "ARM_BOTH",
        "label": "demotion_plus_go_nogo_co_armed_composition",
        "demotion_on": True,
        "gng_on": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, demotion_on: bool, gng_on: bool) -> REEAgent:
    """Matched-stack agent; the ONLY toggled flags are use_f_eligibility_demotion
    (demotion_on) and use_go_nogo_constitution (gng_on). Everything else --
    use_candidate_rule_field=True, use_f_eligibility_adaptive_floor=True, use_dacc,
    the top_k shortlist scaffold, the modulatory authority + routing, MECH-341,
    SD-056, the CRF maturity levers, the trainable lateral_pfc bias head -- is a
    matched constant on all four arms.
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
        # --- Matched stack (identical on all four arms) ---
        # Layer A: SP-CEM (candidate-pool first-action-class diversity).
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # GAP-A (V3-EXQ-649): shared per-candidate signal from e2.world_forward.
        candidate_summary_source="e2_world_forward",
        # modulatory-bias-selection-authority (V3-EXQ-643a float32 fix) + channel routing.
        use_modulatory_selection_authority=USE_MODULATORY_SELECTION_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_channel_routing=USE_MODULATORY_CHANNEL_ROUTING,
        modulatory_channel_route_source=MODULATORY_CHANNEL_ROUTE_SOURCE,
        modulatory_channel_route_weight=MODULATORY_CHANNEL_ROUTE_WEIGHT,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        # 569i TOP-K shortlist scaffold (the eligible-set machinery the Go/No-Go gate
        # governs; f_demotion overrides the eligible-set construction when demotion ON).
        # MATCHED CONSTANT on all arms.
        use_modulatory_shortlist_then_modulate=USE_MODULATORY_SHORTLIST_THEN_MODULATE,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        # MECH-448 RANK-PRESERVING F->ELIGIBILITY DEMOTION -- TOGGLED (demotion_on).
        use_f_eligibility_demotion=bool(demotion_on),
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        # CHANNEL-ADAPTIVE (mean-relative) floor -- MATCHED CONSTANT (inert when
        # demotion OFF; 689e PASS).
        use_f_eligibility_adaptive_floor=USE_F_ELIGIBILITY_ADAPTIVE_FLOOR,
        f_eligibility_adaptive_mean_factor=F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
        # MECH-449 GO/NO-GO ELIGIBILITY CONSTITUTION -- TOGGLED (gng_on). use_dacc is a
        # MATCHED CONSTANT feeding the MECH-260 perseveration No-Go axis ecologically.
        use_dacc=USE_DACC,
        use_go_nogo_constitution=bool(gng_on),
        gng_perseveration_floor=GNG_PERSEVERATION_FLOOR,
        gng_safety_floor=GNG_SAFETY_FLOOR,
        gng_protect_min_eligible=GNG_PROTECT_MIN_ELIGIBLE,
        # MECH-341 (stratified across-class; within-class temperature default).
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=MECH341_ENTROPY_BIAS_SCALE,
        e3_diversity_stratified_within_class_temperature=None,
        # MECH-313 noise floor.
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        # V_s minimal stack.
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        # ARC-062 GatedPolicy (matched; symmetry-broken bias).
        use_gated_policy=True,
        # SD-033a LateralPFCAnalog with the bias head UN-ZEROED + trainable (GAP-D).
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        # SD-056 (e2 action-conditional divergence; trained online in P0).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
        # --- CRF maturity + maintenance levers (MATCHED; the differentiated source) ---
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_context_from_e2_world_forward=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=CRF_MAINTENANCE_FLOOR,
        crf_maintenance_decay=CRF_MAINTENANCE_DECAY,
        crf_mature_context_match_threshold=CRF_MATURE_CONTEXT_MATCH_THRESHOLD,
        crf_tolerance_conflict_cap=CRF_TOLERANCE_CONFLICT_CAP,
        crf_maintenance_couple_to_theta=CRF_MAINTENANCE_COUPLE_TO_THETA,
        # --- MATCHED CONSTANT: the differentiated conversion source ---
        use_candidate_rule_field=USE_CANDIDATE_RULE_FIELD,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-649/654j)
# ---------------------------------------------------------------------------


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen_classes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen_classes:
            seen_classes[cls] = tup
        if len(seen_classes) >= k:
            break
    if len(seen_classes) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen_classes.values())
    picked_ids = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked_ids:
            continue
        samples.append(tup)
        picked_ids.add(id(tup))
    return samples


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K,
        actions=actions_K,
        z_world_1_targets=z1_K,
        simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    weighted = SD056_WEIGHT * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# Per-tick measurement helpers
# ---------------------------------------------------------------------------


def _traj_first_action_class(traj) -> int:
    return int(traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item())


def _traj_rep_signature(traj) -> Tuple[int, ...]:
    acts = traj.actions[0]  # [horizon, action_dim]
    h = min(H_SIGNATURE, int(acts.shape[0]))
    classes = acts[:h, :].argmax(dim=-1).detach().reshape(-1).tolist()
    return tuple(int(c) for c in classes)


def _consumed_summaries(agent: REEAgent, candidates) -> Optional[torch.Tensor]:
    """Per-candidate cand_world_summaries the bias channels consume (GAP-A
    e2.world_forward source; matched on all arms)."""
    summ = agent._candidate_world_summaries(candidates)
    if summ is not None:
        return summ.detach()
    rows: List[torch.Tensor] = []
    for c in candidates:
        if c.world_states is not None:
            rows.append(c.get_world_state_sequence()[0, 0, :].detach())
        elif agent._current_latent is not None:
            rows.append(agent._current_latent.z_world[0].detach())
        else:
            return None
    return torch.stack(rows, dim=0) if rows else None


def _mean_pairwise_l2(summ: torch.Tensor) -> float:
    summ = summ.detach()
    k = summ.shape[0]
    if k < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += float(torch.linalg.vector_norm(summ[i] - summ[j]))
            n += 1
    return total / max(n, 1)


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


def _entropy_from_counter(counter: Counter) -> float:
    n = sum(counter.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


def _entropy_from_int_counts(counts: Dict[int, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


# ---------------------------------------------------------------------------
# P1 bias-head REINFORCE training (mirror V3-EXQ-598b/654j)
# ---------------------------------------------------------------------------


def _lpfc_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
) -> torch.Tensor:
    if agent.lateral_pfc is None or len(outcome_buf) < 2:
        return torch.zeros(1, device=device)
    n = len(outcome_buf)
    idxs = np.random.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False)
    terms: List[torch.Tensor] = []
    for i in idxs:
        cand_features, sel_idx, ep_return = outcome_buf[int(i)]
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            continue
        bias = agent.lateral_pfc.compute_bias(cand_features.to(device))
        log_p = F.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
    if not terms:
        return torch.zeros(1, device=device)
    return torch.stack(terms).mean()


def _propagation_counterfactual_delta(
    agent: REEAgent, summaries: torch.Tensor
) -> Optional[float]:
    """Within-arm isolation: mean |bias(field rule_state) - bias(zeroed rule_state)|.
    Quantifies how much the field's rule_state shapes the per-candidate bias. Best-effort."""
    lpfc = getattr(agent, "lateral_pfc", None)
    if lpfc is None or summaries is None:
        return None
    try:
        with torch.no_grad():
            bias_field = lpfc.compute_bias(summaries).detach().clone()
            saved = lpfc.rule_state.detach().clone()
            lpfc.rule_state.zero_()
            bias_zero = lpfc.compute_bias(summaries).detach().clone()
            lpfc.rule_state.copy_(saved)
        return float((bias_field - bias_zero).abs().mean().item())
    except Exception:
        return None


def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)
    env = _make_env(seed)
    agent = _make_agent(env, bool(arm["demotion_on"]), bool(arm["gng_on"]))
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = torch.optim.Adam(
        list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS
    )
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes + p2_episodes
    p1_start = p0_episodes
    p2_start = p0_episodes + p1_episodes
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p2_ticks = 0
    n_p0_contrastive_steps = 0
    n_p1_bias_updates = 0

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    # PRIMARY DV (699b): PER-COMMITMENT committed-class counts -- accumulated ONLY
    # on a tick where E3 genuinely re-selected (the freshness marker repopulated).
    # This is the construct MECH-448/449 act on: both levers operate at SELECTION.
    committed_class_counts_fresh: Dict[int, int] = {}
    # 699's DV, retained VERBATIM as a named secondary so the size and direction of
    # the defect is measurable for the first time: per-ENV-STEP occupancy counts,
    # which weight each commitment by its hold duration.
    occupancy_class_counts: Dict[int, int] = {}
    # Freshness telemetry -- the single field whose absence made 699 unrecoverable.
    n_fresh_select = 0
    n_latched = 0
    # Hold-duration distribution: consecutive env steps spent on one fresh
    # selection. Directly tests the perseveration/alignment argument.
    hold_durations: List[int] = []
    _cur_hold = 0
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0

    # SECONDARY negative control (within-class-representative; P2).
    per_class_rep_sigs: Dict[int, Counter] = {}
    all_rep_sigs: Counter = Counter()

    # Rule-field differentiation + bias diagnostics (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_matched_per_tick: List[int] = []
    crf_max_pairwise_rule_dist_max = 0.0
    crf_n_minted_total_last = 0
    lateral_pfc_bias_abs_vals: List[float] = []
    prop_counterfactual_deltas: List[float] = []

    # MECH-448 f_eligibility-demotion non-vacuity readouts (LIVE; fires only on
    # demotion-armed arms). excluded_count == 0 == all-admit == vacuous.
    demotion_active_ticks = 0
    demotion_envelope_sizes: List[float] = []
    demotion_excluded_counts: List[float] = []
    demotion_winner_neq_f_argmin_ticks = 0
    demotion_rank_preserving_active_ticks = 0

    # MECH-449 Go/No-Go-constitution non-vacuity readouts (LIVE; fires only on
    # gng-armed arms). suppressed == safety + soft No-Go applied; 0 == inert == vacuous.
    nogo_active_ticks = 0
    nogo_suppressed_per_tick: List[int] = []
    nogo_envelope_sizes: List[float] = []

    for ep in range(total_train_eps):
        is_p1 = (p1_start <= ep < p2_start)
        is_p2 = (ep >= p2_start)
        phase_label = "P2" if is_p2 else ("P1" if is_p1 else "P0")

        _, obs_dict = env.reset()
        agent.reset()

        # Flush any hold open at the episode boundary -- agent.reset() clears the
        # commitment latch, so a hold cannot span episodes.
        if _cur_hold > 0:
            hold_durations.append(_cur_hold)
            _cur_hold = 0

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

        ep_reward = 0.0
        ep_buf: List[Tuple[torch.Tensor, int]] = []

        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs_harm(obs_dict),
                obs_harm_a=_obs_harm_a(obs_dict),
                obs_harm_history=_obs_harm_history(obs_dict),
            )

            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(
                    z_self_prev, action_prev, latent.z_self.detach()
                )

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            pre_e3_classes: List[int] = []
            if is_p2 and candidates:
                pre_e3_classes = sorted({
                    _traj_first_action_class(t) for t in candidates
                })

            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            # --- FRESHNESS MARKER (the 699 instrument repair) --------------------
            # e3_selector.select() reassigns last_score_diagnostics WHOLESALE to a
            # fresh dict (e3_selector.py:2452, unconditional, no early return), so a
            # private marker key survives iff select() did NOT run this tick. On a
            # non-E3 tick agent.py:5430 returns the HELD action before e3.select() is
            # reached, and the diagnostics dict is left untouched.
            #
            # WHY A SENTINEL KEY AND NOT `= None` (a deliberate, documented deviation
            # from the autopsy's routing_detail requirement 1 and the 785a reference):
            #   * Nulling `_last_selected_trajectory` CHANGES SUBSTRATE BEHAVIOUR.
            #     post_action_update (e3_selector.py:3224) falls back to it when
            #     _committed_trajectory is None -- the ARC-016 deadlock fix -- and it
            #     runs on EVERY step via update_residue (agent.py:8006), not only on
            #     E3 ticks. Nulling it silently SKIPS the running-variance update and
            #     the prediction_error / dynamic_precision metrics on non-E3 ticks
            #     with no live commitment. That perturbs selection dynamics in a
            #     COMPOSITION experiment, which would make 699b a different
            #     experiment rather than a repaired instrument.
            #   * Nulling `last_score_diagnostics` is None-SAFE but not inert either:
            #     _assemble_control_vector (agent.py:9660) reads it on non-E3 ticks
            #     and would fall back to authority defaults instead of the persisted
            #     E3 values.
            # The sentinel key preserves both attributes byte-for-byte -- the only
            # ree_core reader of the dict is agent.py:9660, which uses .get() with
            # defaults, and nothing in ree_core iterates its keys (verified) -- so
            # the marker is INERT while giving an exact per-tick freshness signal.
            _diag_prev = agent.e3.last_score_diagnostics
            if isinstance(_diag_prev, dict):
                _diag_prev[_STALE_MARKER_KEY] = True

            action = agent.select_action(candidates, ticks)

            _diag_now = agent.e3.last_score_diagnostics
            fresh_select = (
                isinstance(_diag_now, dict)
                and _STALE_MARKER_KEY not in _diag_now
            )

            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at arm={arm['arm_id']} seed={seed} "
                        f"phase={phase_label} ep={ep} step={_step}"
                    )
                break

            committed_class = int(action[0].argmax().item())

            if is_p1 and p1_snap_summaries is not None:
                sel = 0
                for ci, c in enumerate(candidates):
                    if (
                        getattr(c, "actions", None) is not None
                        and c.actions.shape[1] >= 1
                        and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                        == committed_class
                    ):
                        sel = min(ci, p1_snap_summaries.shape[0] - 1)
                        break
                ep_buf.append((p1_snap_summaries, sel))

            if is_p2:
                n_p2_ticks += 1

                # --- 699's DV, retained verbatim as a NAMED SECONDARY -------------
                # Per-ENV-STEP occupancy: every step contributes, so each commitment
                # is weighted by its hold duration. This is exactly what 699
                # measured and reported as `committed_class_entropy_nats`.
                occupancy_class_counts[committed_class] = (
                    occupancy_class_counts.get(committed_class, 0) + 1
                )

                # --- PRIMARY DV (699b): PER-COMMITMENT --------------------------
                if fresh_select:
                    n_fresh_select += 1
                    committed_class_counts_fresh[committed_class] = (
                        committed_class_counts_fresh.get(committed_class, 0) + 1
                    )
                    if _cur_hold > 0:
                        hold_durations.append(_cur_hold)
                    _cur_hold = 1
                    if len(pre_e3_classes) >= 2:
                        n_p2_pre_ge2 += 1
                else:
                    n_latched += 1
                    if _cur_hold > 0:
                        _cur_hold += 1

                # Candidate regeneration is itself e3_tick-gated (agent.py:4812), so
                # the consumed-summary divergence is ALSO a per-selection quantity --
                # a third exposure the autopsy names. Gate it on freshness too.
                if fresh_select and candidates and len(candidates) >= 2:
                    consumed = _consumed_summaries(agent, candidates)
                    if consumed is not None and torch.isfinite(consumed).all():
                        d = _mean_pairwise_l2(consumed)
                        if math.isfinite(d):
                            consumed_dists.append(d)
                            consumed_dist_max = max(consumed_dist_max, d)

                # SECONDARY negative control (within-class representative). NOTE:
                # `selected_class_counts` / `selected_class_entropy_nats` are DROPPED
                # in 699b per the autopsy: as instrumented they equalled the primary
                # DV to 6dp on all 12 arm-seeds and carried zero independent
                # information -- an internal cross-check that looked like
                # corroboration and was the same number twice. The rep-signature
                # control below is retained and is now fresh-gated.
                sel_traj = (
                    getattr(agent.e3, "_last_selected_trajectory", None)
                    if fresh_select else None
                )
                if sel_traj is not None:
                    sel_class = _traj_first_action_class(sel_traj)
                    rep_sig = _traj_rep_signature(sel_traj)
                    per_class_rep_sigs.setdefault(sel_class, Counter())[rep_sig] += 1
                    all_rep_sigs[rep_sig] += 1

                # lateral_pfc._last_bias_abs_mean is written inside compute_bias,
                # which is reached only from the E3 selection path -- another
                # per-selection quantity. Fresh-gated.
                lpfc = getattr(agent, "lateral_pfc", None) if fresh_select else None
                if lpfc is not None:
                    lb_mean = getattr(lpfc, "_last_bias_abs_mean", None)
                    if isinstance(lb_mean, (int, float)):
                        lateral_pfc_bias_abs_vals.append(float(lb_mean))

                # C1(e)/C1(f) lever non-vacuity diagnostics. The PREDICATES and
                # THRESHOLDS are UNCHANGED from 699 (the autopsy finds the C1
                # battery sound and threshold-invariant, and reproducing it exactly
                # is the one part of 699 worth keeping). What changes is only the
                # DENOMINATOR: the active-fracs are now over n_fresh_select rather
                # than n_p2_ticks, which is the honest denominator for a
                # per-selection gate. Under the autopsy's finding that these gates
                # fired on EVERY E3 tick (measured exactly 1.0), both denominators
                # yield the same saturated reading -- so this is a correctness fix
                # that does not move the gate.
                diag = (
                    (getattr(agent.e3, "last_score_diagnostics", {}) or {})
                    if fresh_select else {}
                )
                # MECH-448 f_eligibility-demotion diagnostics (fires only on
                # demotion-armed arms; C1e checks active AND actually excludes).
                if bool(diag.get("f_eligibility_demotion_active", False)):
                    demotion_active_ticks += 1
                    env_size = float(diag.get("f_eligibility_envelope_size", -1))
                    if math.isfinite(env_size) and env_size >= 0:
                        demotion_envelope_sizes.append(env_size)
                    excl = float(diag.get("f_eligibility_excluded_count", -1))
                    if math.isfinite(excl) and excl >= 0:
                        demotion_excluded_counts.append(excl)
                    if bool(diag.get("f_eligibility_winner_neq_f_argmin", False)):
                        demotion_winner_neq_f_argmin_ticks += 1
                    if bool(diag.get("f_eligibility_rank_preserving", True)):
                        demotion_rank_preserving_active_ticks += 1

                # MECH-449 Go/No-Go-constitution diagnostics (fires only on
                # gng-armed arms; C1f checks active AND actually suppresses).
                if bool(diag.get("go_nogo_constitution_active", False)):
                    nogo_active_ticks += 1
                    n_safety = int(diag.get("go_nogo_n_safety_nogo", 0) or 0)
                    n_soft = int(diag.get("go_nogo_n_soft_applied", 0) or 0)
                    nogo_suppressed_per_tick.append(n_safety + n_soft)
                    gng_env = float(diag.get("go_nogo_envelope_size", -1))
                    if math.isfinite(gng_env) and gng_env >= 0:
                        nogo_envelope_sizes.append(gng_env)

                # CRF state advances with candidate generation, which is e3_tick-gated
                # (agent.py:4812) -- per-selection. Fresh-gated so crf_frac_active and
                # the propagation counterfactual are per-selection rates, not
                # hold-duration-weighted ones.
                crf = getattr(agent, "candidate_rule_field", None) if fresh_select else None
                if crf is not None:
                    st = crf.get_state()
                    n_active = int(st.get("crf_n_active_last", 0))
                    crf_n_active_per_tick.append(n_active)
                    crf_n_matched_per_tick.append(int(st.get("crf_n_matched_last", 0)))
                    crf_max_pairwise_rule_dist_max = max(
                        crf_max_pairwise_rule_dist_max,
                        float(st.get("crf_max_pairwise_rule_dist", 0.0)),
                    )
                    crf_n_minted_total_last = int(st.get("crf_n_minted_total", 0))
                    if (
                        n_active >= CRF_N_ACTIVE_FLOOR
                        and candidates and len(candidates) >= 2
                    ):
                        cf_summ = _consumed_summaries(agent, candidates)
                        if cf_summ is not None and torch.isfinite(cf_summ).all():
                            d_cf = _propagation_counterfactual_delta(agent, cf_summ)
                            if d_cf is not None and math.isfinite(d_cf):
                                prop_counterfactual_deltas.append(d_cf)
            elif is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 ONLY (e2 frozen in P1/P2 for stable measurement).
            if (not is_p1) and (not is_p2) and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val):
                    n_p0_contrastive_steps += 1

            _, _harm_signal, done, info, obs_dict = env.step(action)
            if is_p1:
                ep_reward += float(_harm_signal)
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(_harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            tick_in_ep += 1
            if done:
                break

        if is_p1:
            reinforce_baseline = (
                EMA_DECAY * reinforce_baseline + (1.0 - EMA_DECAY) * ep_reward
            )
            for cand_features, sel in ep_buf:
                outcome_buf.append((cand_features, sel, ep_reward))
            if len(outcome_buf) > OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
            l_loss = _lpfc_reinforce_loss(
                agent, outcome_buf, reinforce_baseline, agent.device
            )
            if l_loss.requires_grad:
                bias_opt.zero_grad()
                l_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.lateral_pfc.bias_head_parameters(), 1.0
                )
                bias_opt.step()
                n_p1_bias_updates += 1

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

    # ----- Per-seed aggregation (over P2) -----
    # Flush a hold left open by the final episode.
    if _cur_hold > 0:
        hold_durations.append(_cur_hold)
        _cur_hold = 0

    # PRIMARY DV (699b): per-COMMITMENT class entropy -- one observation per genuine
    # E3 selection. This is the construct MECH-448/449 act on and the DV the
    # composition verdict reads.
    committed_class_entropy = _entropy_from_int_counts(committed_class_counts_fresh)
    # SECONDARY: 699's DV verbatim -- per-ENV-STEP occupancy, hold-duration-weighted.
    # Emitting BOTH is what makes the size and direction of the 699 defect a
    # MEASURED quantity for the first time (autopsy routing requirement 3).
    occupancy_class_entropy = _entropy_from_int_counts(occupancy_class_counts)
    entropy_defect_delta = occupancy_class_entropy - committed_class_entropy

    # Freshness telemetry (autopsy routing requirement 2) -- the single field whose
    # absence made 699 unrecoverable. n_p2_ticks counts ENV STEPS and varies with
    # episode termination, so it could not stand in for the replication factor.
    fresh_select_yield = (
        float(n_fresh_select) / float(n_p2_ticks) if n_p2_ticks > 0 else 0.0
    )
    replication_factor = (
        float(n_p2_ticks) / float(n_fresh_select) if n_fresh_select > 0 else 0.0
    )

    # Hold-duration distribution (autopsy routing requirement 4) -- directly tests
    # the perseveration/alignment argument: if ARM_OFF's holds are systematically
    # longer, that confirms the 699 bias was aligned with the hypothesis; if not,
    # it bounds the distortion.
    hold_duration_mean = (
        float(sum(hold_durations)) / float(len(hold_durations))
        if hold_durations else 0.0
    )
    _hd_sorted = sorted(hold_durations)
    hold_duration_median = (
        float(_hd_sorted[len(_hd_sorted) // 2]) if _hd_sorted else 0.0
    )
    hold_duration_max = int(_hd_sorted[-1]) if _hd_sorted else 0
    hold_duration_hist = {
        str(k): int(v) for k, v in sorted(Counter(hold_durations).items())
    }

    # C1(a) denominator is now the fresh-selection count: pre_e3_classes derives from
    # `candidates`, whose regeneration is itself e3_tick-gated (agent.py:4812).
    frac_pre_ge2 = (
        float(n_p2_pre_ge2 / n_fresh_select) if n_fresh_select > 0 else 0.0
    )

    consumed_spread_mean = (
        float(sum(consumed_dists) / len(consumed_dists)) if consumed_dists else 0.0
    )

    qualifying: List[float] = []
    within_class_entropies: Dict[int, float] = {}
    for cls, sig_counter in per_class_rep_sigs.items():
        ent = _entropy_from_counter(sig_counter)
        within_class_entropies[cls] = ent
        if sum(sig_counter.values()) >= MIN_TICKS_PER_CLASS:
            qualifying.append(ent)
    mean_within_class_rep_entropy = (
        float(sum(qualifying) / len(qualifying)) if qualifying else 0.0
    )

    if crf_n_active_per_tick:
        frac_crf_active_ge_floor = float(
            sum(1 for n in crf_n_active_per_tick if n >= CRF_N_ACTIVE_FLOOR)
            / len(crf_n_active_per_tick)
        )
        mean_crf_n_active = float(
            sum(crf_n_active_per_tick) / len(crf_n_active_per_tick)
        )
    else:
        frac_crf_active_ge_floor = 0.0
        mean_crf_n_active = 0.0

    if crf_n_matched_per_tick:
        mean_crf_n_matched = float(
            sum(crf_n_matched_per_tick) / len(crf_n_matched_per_tick)
        )
        max_crf_n_matched = int(max(crf_n_matched_per_tick))
    else:
        mean_crf_n_matched = 0.0
        max_crf_n_matched = 0

    # use_candidate_rule_field is a matched constant (True) on all arms.
    # Shipped predicate, called on the UNROUNDED locals (see the hoisted block).
    crf_differentiated = _pred_crf_differentiated({
        "crf_n_minted_total": crf_n_minted_total_last,
        "crf_frac_active_ge_floor": frac_crf_active_ge_floor,
    })

    mean_lateral_pfc_bias_abs = (
        float(sum(lateral_pfc_bias_abs_vals) / len(lateral_pfc_bias_abs_vals))
        if lateral_pfc_bias_abs_vals else 0.0
    )
    mean_prop_counterfactual_delta = (
        float(sum(prop_counterfactual_deltas) / len(prop_counterfactual_deltas))
        if prop_counterfactual_deltas else 0.0
    )

    # MECH-448 f_eligibility-demotion aggregates (C1e; armed only when demotion_on).
    # Denominator is n_fresh_select, not n_p2_ticks: the demotion envelope is a
    # per-SELECTION event, so a per-env-step denominator would divide genuine
    # firings by the hold-inflated tick count. Predicate and threshold unchanged.
    demotion_active_frac = (
        float(demotion_active_ticks) / float(n_fresh_select)
        if n_fresh_select > 0 else 0.0
    )
    demotion_excluded_count_mean = (
        float(sum(demotion_excluded_counts) / len(demotion_excluded_counts))
        if demotion_excluded_counts else 0.0
    )
    demotion_envelope_size_mean = (
        float(sum(demotion_envelope_sizes) / len(demotion_envelope_sizes))
        if demotion_envelope_sizes else 0.0
    )
    demotion_rank_preserving_frac = (
        float(demotion_rank_preserving_active_ticks) / float(demotion_active_ticks)
        if demotion_active_ticks > 0 else 0.0
    )
    seed_demotion_non_vacuous = _pred_demotion_non_vacuous({
        "f_eligibility_demotion_active_frac": demotion_active_frac,
        "f_eligibility_excluded_count_mean": demotion_excluded_count_mean,
    })

    # MECH-449 Go/No-Go-constitution aggregates (C1f; armed only when gng_on).
    # Per-SELECTION denominator, as for demotion above. Predicate/threshold unchanged.
    nogo_active_frac = (
        float(nogo_active_ticks) / float(n_fresh_select)
        if n_fresh_select > 0 else 0.0
    )
    nogo_suppressed_mean = (
        float(sum(nogo_suppressed_per_tick) / len(nogo_suppressed_per_tick))
        if nogo_suppressed_per_tick else 0.0
    )
    nogo_envelope_size_mean = (
        float(sum(nogo_envelope_sizes) / len(nogo_envelope_sizes))
        if nogo_envelope_sizes else 0.0
    )
    seed_nogo_non_vacuous = _pred_nogo_non_vacuous({
        "go_nogo_active_frac": nogo_active_frac,
        "go_nogo_suppressed_per_tick_mean": nogo_suppressed_mean,
    })

    seed_class_axis_exercisable = _pred_class_axis_exercisable(
        {"frac_pre_ge2": frac_pre_ge2})
    seed_gapa_divergence = _pred_gapa_divergence({
        "consumed_summary_pairwise_dist_mean": consumed_spread_mean,
        "consumed_summary_pairwise_dist_max": consumed_dist_max,
    })
    seed_prop_non_vacuous = _pred_prop_non_vacuous(
        {"mean_lateral_pfc_bias_abs": mean_lateral_pfc_bias_abs})

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "demotion_on": bool(arm["demotion_on"]),
        "gng_on": bool(arm["gng_on"]),
        "use_candidate_rule_field": True,
        "crf_persist_rules_across_episode_reset": True,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
        # ----- FRESHNESS TELEMETRY (autopsy routing requirement 2) -----
        # n_p2_ticks counts ENV STEPS; n_fresh_select counts genuine E3 selections.
        # Their ratio IS the replication factor 699 could not even estimate.
        "n_fresh_select": int(n_fresh_select),
        "n_latched": int(n_latched),
        "fresh_select_yield": round(fresh_select_yield, 6),
        "replication_factor": round(replication_factor, 6),
        "fresh_select_sufficient": _pred_fresh_select_sufficient(
            {"n_fresh_select": n_fresh_select}),
        # ----- PRIMARY DV: PER-COMMITMENT class entropy (699b) -----
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "n_unique_committed_classes": int(len(committed_class_counts_fresh)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts_fresh.items())
        },
        # ----- SECONDARY: 699's DV verbatim (hold-weighted OCCUPANCY) -----
        # Reported side-by-side with the primary so the 699 defect's magnitude and
        # SIGN become a measured quantity across the corpus (routing requirement 3).
        # A positive entropy_defect_delta on ARM_OFF larger than on the armed arms is
        # the signature of the hypothesis-aligned bias the autopsy describes.
        "occupancy_class_entropy_nats": round(occupancy_class_entropy, 6),
        "n_unique_occupancy_classes": int(len(occupancy_class_counts)),
        "occupancy_class_counts": {
            str(k): int(v) for k, v in sorted(occupancy_class_counts.items())
        },
        "entropy_defect_delta_nats": round(entropy_defect_delta, 6),
        # ----- HOLD DURATION (autopsy routing requirement 4) -----
        "hold_duration_mean": round(hold_duration_mean, 6),
        "hold_duration_median": round(hold_duration_median, 6),
        "hold_duration_max": int(hold_duration_max),
        "hold_duration_n": int(len(hold_durations)),
        "hold_duration_hist": hold_duration_hist,
        # ----- C1 readiness -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "class_axis_exercisable": seed_class_axis_exercisable,
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        "crf_mean_n_active": round(mean_crf_n_active, 6),
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_max_pairwise_rule_dist": round(crf_max_pairwise_rule_dist_max, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "crf_differentiated": crf_differentiated,
        "crf_mean_n_matched": round(mean_crf_n_matched, 6),
        "crf_max_n_matched": int(max_crf_n_matched),
        # ----- C1d propagation non-vacuity -----
        "mean_lateral_pfc_bias_abs": round(mean_lateral_pfc_bias_abs, 8),
        "mean_prop_counterfactual_delta": round(mean_prop_counterfactual_delta, 8),
        "prop_non_vacuous": seed_prop_non_vacuous,
        # ----- C1e MECH-448 demotion non-vacuity (armed only when demotion_on) -----
        "f_eligibility_demotion_active_ticks": int(demotion_active_ticks),
        "f_eligibility_demotion_active_frac": round(demotion_active_frac, 6),
        "f_eligibility_excluded_count_mean": round(demotion_excluded_count_mean, 6),
        "f_eligibility_envelope_size_mean": round(demotion_envelope_size_mean, 6),
        "f_eligibility_winner_neq_f_argmin_ticks": int(demotion_winner_neq_f_argmin_ticks),
        "f_eligibility_rank_preserving_frac": round(demotion_rank_preserving_frac, 6),
        "demotion_non_vacuous": seed_demotion_non_vacuous,
        # ----- C1f MECH-449 active No-Go non-vacuity (armed only when gng_on) -----
        "go_nogo_active_ticks": int(nogo_active_ticks),
        "go_nogo_active_frac": round(nogo_active_frac, 6),
        "go_nogo_suppressed_per_tick_mean": round(nogo_suppressed_mean, 6),
        "go_nogo_envelope_size_mean": round(nogo_envelope_size_mean, 6),
        "nogo_non_vacuous": seed_nogo_non_vacuous,
        # ----- C3 SECONDARY negative control (NOT load-bearing) -----
        "mean_within_class_rep_entropy_nats": round(mean_within_class_rep_entropy, 6),
        "n_distinct_rep_signatures_total": int(len(all_rep_sigs)),
        # NOTE: `selected_class_entropy_nats` is DROPPED in 699b (autopsy routing
        # requirement 5). Under 699's per-env-step instrumentation it equalled
        # committed_class_entropy_nats to 6dp on ALL 12 arm-seeds and carried zero
        # independent information -- the design's internal cross-check was vacuous.
        # Under fresh-gating it would be the same number by construction, since both
        # now derive from the identical set of genuine selections.
    }


def _arm_rows(arm_results: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [
        r for r in arm_results
        if r["arm_id"] == arm_id and r["error_note"] is None
    ]


def _composition_verdict_from(
    off_rows: List[Dict[str, Any]],
    dem_rows: List[Dict[str, Any]],
    gng_rows: List[Dict[str, Any]],
    both_rows: List[Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    """Pre-registered composition verdict computed over an arbitrary per-cell DV `key`.

    Factored out of the inline 699 computation for ONE reason: it lets the SAME
    pre-registered rule be applied to BOTH instrumentations of the same run --
    key="committed_class_entropy_nats" (699b, per-COMMITMENT) and
    key="occupancy_class_entropy_nats" (699's per-ENV-STEP DV, retained verbatim).

    That is the matched-replay method used on the 663 driver at ree-v3 5433e3ab1c,
    and it is what turns the 699 defect from an argued distortion into a measured
    one: both verdicts come from identical cells, identical seeds and identical
    thresholds, differing ONLY in the sampling unit of the DV.

    The 699b verdict (per-commitment) is the LOAD-BEARING one. The 699-instrumented
    verdict is DIAGNOSTIC CONTEXT ONLY and must not be reported as a finding -- the
    autopsy withdrew `levers_compound` and established that the corrected sign is
    UNKNOWN, so neither agreement nor disagreement between the two rehabilitates it.

    Thresholds are the module-level pre-registered constants, unchanged from 699.
    """
    off_by_seed = {int(r["seed"]): r[key] for r in off_rows}
    dem_by_seed = {int(r["seed"]): r[key] for r in dem_rows}
    gng_by_seed = {int(r["seed"]): r[key] for r in gng_rows}
    both_by_seed = {int(r["seed"]): r[key] for r in both_rows}

    shared_seeds = sorted(
        set(off_by_seed) & set(dem_by_seed) & set(gng_by_seed) & set(both_by_seed)
    )

    per_seed_deltas: Dict[int, Dict[str, Any]] = {}
    n_compound = 0
    n_cancel = 0
    n_neutral = 0
    for seed in shared_seeds:
        e_off = off_by_seed[seed]
        e_dem = dem_by_seed[seed]
        e_gng = gng_by_seed[seed]
        e_both = both_by_seed[seed]
        d_dem = e_dem - e_off
        d_gng = e_gng - e_off
        d_both = e_both - e_off
        interaction = d_both - (d_dem + d_gng)
        better_single = max(d_dem, d_gng)
        worse_single = min(d_dem, d_gng)
        if (
            d_both >= better_single + COMPOSITION_MARGIN
            and d_both >= COMPOSITION_LIFT_MARGIN
        ):
            verdict = "compound"
            n_compound += 1
        elif (
            d_both <= worse_single - COMPOSITION_MARGIN
            or d_both <= -COMPOSITION_MARGIN
        ):
            verdict = "cancel"
            n_cancel += 1
        else:
            verdict = "neutral"
            n_neutral += 1
        per_seed_deltas[seed] = {
            "e_off": round(e_off, 6),
            "e_dem": round(e_dem, 6),
            "e_gng": round(e_gng, 6),
            "e_both": round(e_both, 6),
            "d_dem": round(d_dem, 6),
            "d_gng": round(d_gng, 6),
            "d_both": round(d_both, 6),
            "interaction": round(interaction, 6),
            "verdict": verdict,
        }

    arm_spreads: List[float] = []
    for seed in shared_seeds:
        vals = [
            off_by_seed[seed], dem_by_seed[seed],
            gng_by_seed[seed], both_by_seed[seed],
        ]
        arm_spreads.append(max(vals) - min(vals))
    max_arm_spread = max(arm_spreads) if arm_spreads else 0.0

    composition_verdict = "undetermined"
    if shared_seeds:
        if n_compound >= COMPOSITION_MIN_SEEDS:
            composition_verdict = "levers_compound"
        elif n_cancel >= COMPOSITION_MIN_SEEDS:
            composition_verdict = "levers_cancel"
        else:
            composition_verdict = "levers_neutral"

    return {
        "dv_key": key,
        "composition_verdict": composition_verdict,
        "per_seed_deltas": {str(k): v for k, v in per_seed_deltas.items()},
        "n_seeds_compound": int(n_compound),
        "n_seeds_cancel": int(n_cancel),
        "n_seeds_neutral": int(n_neutral),
        "shared_seeds": [int(s) for s in shared_seeds],
        "max_arm_spread": round(max_arm_spread, 6),
        "arms_not_bit_identical": bool(max_arm_spread > 1e-9),
        "mean_dv_by_arm": {
            "ARM_OFF": round(_mean([r[key] for r in off_rows]), 6),
            "ARM_DEM": round(_mean([r[key] for r in dem_rows]), 6),
            "ARM_GNG": round(_mean([r[key] for r in gng_rows]), 6),
            "ARM_BOTH": round(_mean([r[key] for r in both_rows]), 6),
        },
    }


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arm_results: List[Dict[str, Any]] = []
    script_path = Path(__file__).resolve()

    for arm in ARMS:
        print(
            f"Arm {arm['arm_id']} ({arm['label']}) "
            f"demotion_on={arm['demotion_on']} gng_on={arm['gng_on']} "
            f"(P0={p0_episodes} ep e2-train, P1={p1_episodes} ep bias-train, "
            f"P2={p2_episodes} ep measure, steps_per_episode={steps_per_episode}, "
            f"dry_run={dry_run})",
            flush=True,
        )
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)
            row = _run_seed_arm(
                arm, s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
            )
            row["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm_id": arm["arm_id"],
                    "demotion_on": bool(arm["demotion_on"]),
                    "gng_on": bool(arm["gng_on"]),
                    "use_candidate_rule_field": True,
                    "crf_persist_rules_across_episode_reset": True,
                    "crf_mature_pool_dynamics": True,
                    "crf_context_from_e2_world_forward": True,
                    "crf_availability_maintenance": True,
                    "crf_maintenance_floor": float(CRF_MAINTENANCE_FLOOR),
                    "crf_maintenance_decay": float(CRF_MAINTENANCE_DECAY),
                    "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
                    "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
                    "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
                    "f_eligibility_envelope_floor": float(F_ELIGIBILITY_ENVELOPE_FLOOR),
                    "f_eligibility_dn_sigma": float(F_ELIGIBILITY_DN_SIGMA),
                    "use_f_eligibility_adaptive_floor": bool(USE_F_ELIGIBILITY_ADAPTIVE_FLOOR),
                    "f_eligibility_adaptive_mean_factor": float(F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR),
                    "use_dacc": bool(USE_DACC),
                    "gng_perseveration_floor": float(GNG_PERSEVERATION_FLOOR),
                    "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
                    "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
                    "modulatory_channel_route_source": str(MODULATORY_CHANNEL_ROUTE_SOURCE),
                    "env_kwargs": dict(ENV_KWARGS),
                    "sd056_weight": SD056_WEIGHT,
                    "lr_lpfc_bias": LR_LPFC_BIAS,
                    "p0_episodes": int(p0_episodes),
                    "p1_episodes": int(p1_episodes),
                    "p2_episodes": int(p2_episodes),
                    "steps_per_episode": int(steps_per_episode),
                },
                seed=s,
                script_path=script_path,
                rng_fully_reset=True,
                extra_ineligible_reasons=[
                    "online_e2_training_stateful_per_cell",
                    "p1_bias_head_reinforce_training_stateful_per_cell",
                ],
            )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    off_rows = _arm_rows(arm_results, "ARM_OFF")
    dem_rows = _arm_rows(arm_results, "ARM_DEM")
    gng_rows = _arm_rows(arm_results, "ARM_GNG")
    both_rows = _arm_rows(arm_results, "ARM_BOTH")
    all_rows = off_rows + dem_rows + gng_rows + both_rows
    demotion_armed_rows = dem_rows + both_rows
    gng_armed_rows = gng_rows + both_rows

    def _maj_all(pred) -> bool:
        # majority (>= MIN_SEEDS_FOR_PASS) of seeds in EACH of the four arms.
        return all(
            sum(1 for r in arm_rows if pred(r)) >= MIN_SEEDS_FOR_PASS
            for arm_rows in (off_rows, dem_rows, gng_rows, both_rows)
        )

    def _min_arm_count(pred, groups=None) -> int:
        """The WORST per-arm count of seeds satisfying `pred` -- the recomputable form of
        the `_maj_all` / per-arm-majority rule, for the precondition DECLARATIONS below.

        `min(per-arm counts) >= MIN_SEEDS_FOR_PASS` is EXACTLY `all(count >=
        MIN_SEEDS_FOR_PASS)`, so the adjudication gate
        (build_experiment_indexes._precondition_unmet), which RECOMPUTES each
        precondition's `met` from its own (measured, threshold) pair and treats the
        recompute as AUTHORITATIVE, reproduces the shipped boolean. The min-over-cells of
        the underlying CONTINUOUS statistic that these entries used to report does not:
        it is strictly HARSHER than "on a majority of seeds" (one bad seed sinks it), so
        the recompute wrongly flagged sound diagnostics `precondition_unmet`. And no
        single continuous statistic CAN reproduce these -- most of the per-seed flags are
        CONJUNCTIONS (e.g. gapa_divergence is `spread > floor AND dist_max < ceil`), and a
        count over a conjunction does not distribute into per-leg counts. The original
        continuous statistics are preserved on each entry as NON-BOUND `observed_*`
        diagnostics; extra keys are ignored by the recompute, so nothing is lost.
        """
        if groups is None:
            groups = (off_rows, dem_rows, gng_rows, both_rows)
        return int(min(
            [sum(1 for r in arm_rows if pred(r)) for arm_rows in groups] or [0]
        ))

    # C1(a): committed-class axis exercisable on majority of seeds in ALL arms.
    c1a_holds = _maj_all(lambda r: r["class_axis_exercisable"])
    # C1(b): GAP-A divergence real on majority of seeds in ALL arms.
    c1b_holds = _maj_all(lambda r: r["gapa_divergence"])
    # C1(c): matured rule field on majority of seeds in ALL arms.
    c1c_holds = _maj_all(lambda r: r["crf_differentiated"])
    # C1(d): propagation non-vacuity (bias reaches committed action) ALL arms.
    c1d_holds = _maj_all(lambda r: r["prop_non_vacuous"])
    n_within_arm_cf_nonzero = sum(
        1 for r in all_rows if r["mean_prop_counterfactual_delta"] > PROP_NONVAC_FLOOR
    )

    # C1(e): MECH-448 demotion non-vacuity on the DEMOTION-ARMED arms ONLY.
    c1e_holds = all(
        sum(1 for r in arm_rows if r["demotion_non_vacuous"]) >= MIN_SEEDS_FOR_PASS
        for arm_rows in (dem_rows, both_rows)
    )
    # C1(f): MECH-449 active No-Go non-vacuity on the GO/NO-GO-ARMED arms ONLY.
    c1f_holds = all(
        sum(1 for r in arm_rows if r["nogo_non_vacuous"]) >= MIN_SEEDS_FOR_PASS
        for arm_rows in (gng_rows, both_rows)
    )

    # C1(g) NEW IN 699b: fresh-selection sufficiency. The per-commitment DV is an
    # entropy over genuine E3 selections, so a cell with too few of them cannot
    # support it. This gate is NOT a tightening of the 699 battery -- 699 had no
    # fresh-selection count at all, which is precisely why its replication factor
    # could not even be estimated. Below-floor self-routes
    # substrate_not_ready_requeue, NEVER a composition verdict.
    c1g_holds = _maj_all(lambda r: r["fresh_select_sufficient"])

    c1_holds = bool(
        c1a_holds and c1b_holds and c1c_holds and c1d_holds
        and c1e_holds and c1f_holds and c1g_holds
    )

    # ----- C_INTERACTION (PRIMARY characterization) -----
    # LOAD-BEARING verdict: the per-COMMITMENT DV. Both claims act at SELECTION, so
    # the readout's sampling unit must be the selection, not the env step.
    _primary = _composition_verdict_from(
        off_rows, dem_rows, gng_rows, both_rows,
        key="committed_class_entropy_nats",
    )
    per_seed_deltas = _primary["per_seed_deltas"]
    n_compound = _primary["n_seeds_compound"]
    n_cancel = _primary["n_seeds_cancel"]
    n_neutral = _primary["n_seeds_neutral"]
    shared_seeds = _primary["shared_seeds"]
    composition_verdict = _primary["composition_verdict"]

    # Non-degeneracy: the four arms must not be bit-identical (otherwise the
    # interaction is vacuous, the V3-EXQ-514m class).
    max_arm_spread = _primary["max_arm_spread"]
    arms_not_bit_identical = _primary["arms_not_bit_identical"]

    off_mean_dv = _primary["mean_dv_by_arm"]["ARM_OFF"]
    dem_mean_dv = _primary["mean_dv_by_arm"]["ARM_DEM"]
    gng_mean_dv = _primary["mean_dv_by_arm"]["ARM_GNG"]
    both_mean_dv = _primary["mean_dv_by_arm"]["ARM_BOTH"]

    # ----- Outcome map (DIAGNOSTIC semantics; NO weakens -- it CHARACTERIZES) -----
    if not c1_holds:
        # The levers were not genuinely engaged in their armed arms, or the class
        # axis / GAP-A / matured field was absent -> the interaction could not be
        # measured -> re-queue. NOT a falsification.
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif not arms_not_bit_identical:
        # The four arms produced bit-identical committed-class entropy -> the
        # interaction is vacuous (no lever moved committed selection on this run).
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue_arms_bit_identical"
    else:
        # C1 holds AND arms differ -> the composition CHARACTERIZATION succeeded.
        # The diagnostic PASS carries the composition verdict; it weights no claim.
        outcome = "PASS"
        direction = "non_contributory"
        label = composition_verdict

    # Diagnostic: claim_ids are CONTEXT only (scoring-excluded). Both levers were
    # characterized for interaction; neither is supported/weakened by a diagnostic.
    evidence_direction_per_claim = {"MECH-448": "non_contributory", "MECH-449": "non_contributory"}

    # ----- interpretation block (preconditions + non-degeneracy + load-bearing) -----
    interpretation = {
        "label": label,
        "composition_verdict": composition_verdict,
        "preconditions": [
            {
                "name": "fresh_e3_selection_sufficiency_all_arms",
                "kind": "readiness",
                "description": (
                    "Each arm has a majority of seeds with >= MIN_FRESH_SELECT_PER_CELL "
                    "GENUINE E3 selections in P2. NEW IN 699b: the primary DV is a "
                    "per-COMMITMENT class entropy, so its sampling unit is the E3 "
                    "selection, not the env step. V3-EXQ-699 recorded no fresh-selection "
                    "count at all, which is why its replication factor could not even be "
                    "estimated from the manifest (n_p2_ticks counts ENV STEPS and varies "
                    "with episode termination). Below-floor self-routes "
                    "substrate_not_ready_requeue, never a composition verdict."
                ),
                "control": (
                    "sentinel-key freshness marker on agent.e3.last_score_diagnostics, "
                    "which e3_selector.select() reassigns wholesale at :2452"
                ),
                "measured": float(
                    _min_arm_count(lambda r: r["fresh_select_sufficient"])),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                # COUNT-shaped, INCLUSIVE floor -- same per-arm-majority form as the
                # entries below (see _min_arm_count).
                "comparator": ">=",
                "direction": "lower",
                "observed_min_n_fresh_select": int(
                    min([r["n_fresh_select"] for r in all_rows] or [0])),
                "observed_min_fresh_select_yield": float(
                    min([r["fresh_select_yield"] for r in all_rows] or [0.0])),
                "observed_fresh_select_floor": int(MIN_FRESH_SELECT_PER_CELL),
                "met": bool(c1g_holds),
            },
            {
                "name": "committed_class_axis_exercisable_all_arms",
                "kind": "readiness",
                "description": (
                    "frac of P2 ticks with >= 2 candidate first-action classes exceeds "
                    "floor on a majority of seeds in ALL FOUR arms. SAME-statistic family "
                    "as the committed-class entropy DV (class multiplicity bounds class "
                    "entropy)."
                ),
                "control": "SP-CEM multi-class candidate pool, all arms",
                "measured": float(_min_arm_count(lambda r: r["class_axis_exercisable"])),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                # COUNT-shaped, INCLUSIVE floor: `met` is a per-arm-majority rule, i.e.
                # `min(per-arm satisfying-seed counts) >= MIN_SEEDS_FOR_PASS`. See
                # _min_arm_count above for why the previous min-over-cells continuous
                # statistic could not reproduce it.
                "comparator": ">=",
                "direction": "lower",
                "observed_min_frac_pre_ge2": float(
                    min([r["frac_pre_ge2"] for r in all_rows] or [0.0])),
                "observed_frac_pre_ge2_floor": float(FRAC_PRE_GE2_FLOOR),
                "met": bool(c1a_holds),
            },
            {
                "name": "gapa_consumed_summary_divergence_all_arms",
                "kind": "readiness",
                "description": (
                    "consumed cand_world_summaries (e2.world_forward) per-candidate SPREAD "
                    "clears the floor on a majority of seeds in ALL arms -- the class bias "
                    "is non-degenerate. Same range statistic the 649 GAP-A readiness asserts."
                ),
                "control": "SD-056 e2 trained online in P0; candidate_summary_source=e2_world_forward",
                "measured": float(_min_arm_count(lambda r: r["gapa_divergence"])),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                # COUNT-shaped, INCLUSIVE floor: `met` is a per-arm-majority rule, i.e.
                # `min(per-arm satisfying-seed counts) >= MIN_SEEDS_FOR_PASS`. See
                # _min_arm_count above for why the previous min-over-cells continuous
                # statistic could not reproduce it.
                "comparator": ">=",
                "direction": "lower",
                "observed_min_consumed_summary_pairwise_dist_mean": float(
                    min([r["consumed_summary_pairwise_dist_mean"] for r in all_rows] or [0.0])
                ),
                "observed_consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
                "met": bool(c1b_holds),
            },
            {
                "name": "gapa_consumed_summary_bounded",
                "kind": "readiness",
                "description": (
                    "consumed-summary spread stayed below the 643a explosion ceiling "
                    "(SD-056 online-training numerical stability; rollout-norm clamp ON)."
                ),
                "control": "max consumed_summary_pairwise_dist across all cells",
                "measured": float(
                    max([r["consumed_summary_pairwise_dist_max"] for r in all_rows] or [0.0])
                ),
                "threshold": float(CONSUMED_MAGNITUDE_CEIL),
                # CEILING-shaped and STRICT: `met` below is `max(...) < CEIL`. The
                # `direction` was already declared; the `comparator` is added so the
                # recompute mirrors the shipped predicate's strictness rather than
                # defaulting to an inclusive bound.
                "direction": "upper",
                "comparator": "<",
                "met": bool(
                    max([r["consumed_summary_pairwise_dist_max"] for r in all_rows] or [0.0])
                    < CONSUMED_MAGNITUDE_CEIL
                ),
            },
            {
                "name": "rule_field_differentiated_and_matured_all_arms",
                "kind": "readiness",
                "description": (
                    "the matched CandidateRuleField (use_candidate_rule_field=True, "
                    "constant) minted >= CRF_MIN_MINTED distinct rules AND fired a non-zero "
                    "rule_state on >= CRF_FRAC_ACTIVE_FLOOR of P2 ticks, on a majority of "
                    "seeds in ALL arms. Below-floor => the differentiated source the levers "
                    "convert is absent => substrate_not_ready_requeue."
                ),
                "control": "crf frac-active (matured pool) + crf_n_minted_total, all arms",
                "measured": float(_min_arm_count(lambda r: r["crf_differentiated"])),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                # COUNT-shaped, INCLUSIVE floor: `met` is a per-arm-majority rule, i.e.
                # `min(per-arm satisfying-seed counts) >= MIN_SEEDS_FOR_PASS`. See
                # _min_arm_count above for why the previous min-over-cells continuous
                # statistic could not reproduce it.
                "comparator": ">=",
                "direction": "lower",
                "observed_min_crf_frac_active": float(
                    min([r["crf_frac_active_ge_floor"] for r in all_rows] or [0.0])),
                "observed_crf_frac_active_floor": float(CRF_FRAC_ACTIVE_FLOOR),
                "met": bool(c1c_holds),
            },
            {
                "name": "propagation_non_vacuity_bias_reaches_committed_action_all_arms",
                "kind": "readiness",
                "description": (
                    "mean_lateral_pfc_bias_abs > floor on a majority of seeds in ALL arms "
                    "(the trained head + matured field produce a non-zero per-candidate "
                    "bias that reaches committed action). Below-floor => vacuous propagation "
                    "=> substrate_not_ready_requeue. Supported by the within-arm rule_state "
                    "counterfactual delta (zeroing rule_state changes the bias)."
                ),
                "control": "mean lateral_pfc bias on the matched trained-head stack, all arms",
                "measured": float(_min_arm_count(lambda r: r["prop_non_vacuous"])),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                # COUNT-shaped, INCLUSIVE floor: `met` is a per-arm-majority rule, i.e.
                # `min(per-arm satisfying-seed counts) >= MIN_SEEDS_FOR_PASS`. See
                # _min_arm_count above for why the previous min-over-cells continuous
                # statistic could not reproduce it.
                "comparator": ">=",
                "direction": "lower",
                "observed_min_mean_lateral_pfc_bias_abs": float(
                    min([r["mean_lateral_pfc_bias_abs"] for r in all_rows] or [0.0])),
                "observed_prop_nonvac_floor": float(PROP_NONVAC_FLOOR),
                "met": bool(c1d_holds),
            },
            {
                "name": "mech448_demotion_lever_live_and_excluding_demotion_armed_arms",
                "kind": "readiness",
                "description": (
                    "On the DEMOTION-ARMED arms (ARM_DEM, ARM_BOTH) the f_demotion envelope "
                    "is ACTIVE on >= DEMOTION_ACTIVE_FRAC_FLOOR of P2 ticks AND actually "
                    "EXCLUDES (mean f_eligibility_excluded_count > EXCLUDED_COUNT_FLOOR) on a "
                    "majority of seeds. An all-admit envelope (excluded_count==0; flat-F pool) "
                    "means demotion did nothing -> the composition cell is vacuous => "
                    "substrate_not_ready_requeue. Demotion is OFF by design on ARM_OFF/ARM_GNG "
                    "(inactivity there is correct, NOT vacuous). Mirrors V3-EXQ-689d readiness."
                ),
                "control": "f_demotion active-frac + excluded_count on the demotion-armed arms",
                "measured": float(_min_arm_count(
                    lambda r: r["demotion_non_vacuous"], groups=(dem_rows, both_rows))),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                # COUNT-shaped, INCLUSIVE floor: `met` is a per-arm-majority rule, i.e.
                # `min(per-arm satisfying-seed counts) >= MIN_SEEDS_FOR_PASS`. See
                # _min_arm_count above for why the previous min-over-cells continuous
                # statistic could not reproduce it.
                "comparator": ">=",
                "direction": "lower",
                "observed_min_f_eligibility_excluded_count_mean": float(
                    min([r["f_eligibility_excluded_count_mean"] for r in demotion_armed_rows] or [0.0])
                ),
                "observed_excluded_count_floor": float(EXCLUDED_COUNT_FLOOR),
                "met": bool(c1e_holds),
            },
            {
                "name": "mech449_active_nogo_live_and_suppressing_gng_armed_arms",
                "kind": "readiness",
                "description": (
                    "On the GO/NO-GO-ARMED arms (ARM_GNG, ARM_BOTH) the Go/No-Go constitution "
                    "is ACTIVE on >= NOGO_ACTIVE_FRAC_FLOOR of P2 ticks AND actually SUPPRESSES "
                    "(mean go_nogo_n_safety_nogo + go_nogo_n_soft_applied > NOGO_SUPPRESSED_FLOOR; "
                    "the perseveration No-Go fed by the MECH-260 recency-share vector with "
                    "use_dacc=True) on a majority of seeds. An inert No-Go means the opponency "
                    "leg did nothing -> vacuous => substrate_not_ready_requeue. Go/No-Go is OFF "
                    "by design on ARM_OFF/ARM_DEM. Mirrors the V3-EXQ-689g non-degeneracy gate."
                ),
                "control": "Go/No-Go active-frac + suppressed-count on the gng-armed arms",
                "measured": float(_min_arm_count(
                    lambda r: r["nogo_non_vacuous"], groups=(gng_rows, both_rows))),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                # COUNT-shaped, INCLUSIVE floor: `met` is a per-arm-majority rule, i.e.
                # `min(per-arm satisfying-seed counts) >= MIN_SEEDS_FOR_PASS`. See
                # _min_arm_count above for why the previous min-over-cells continuous
                # statistic could not reproduce it.
                "comparator": ">=",
                "direction": "lower",
                "observed_min_go_nogo_suppressed_per_tick_mean": float(
                    min([r["go_nogo_suppressed_per_tick_mean"] for r in gng_armed_rows] or [0.0])
                ),
                "observed_nogo_suppressed_floor": float(NOGO_SUPPRESSED_FLOOR),
                "met": bool(c1f_holds),
            },
        ],
        "criteria": [
            {
                "name": "C_READY_levers_engaged_and_substrate_exercisable",
                "load_bearing": True,
                "passed": bool(c1_holds and arms_not_bit_identical),
            },
        ],
        "criteria_non_degenerate": {
            "C1a_class_axis_exercisable_all_arms": bool(c1a_holds),
            "C1b_gapa_divergence_all_arms": bool(c1b_holds),
            "C1c_rule_field_differentiated_matured_all_arms": bool(c1c_holds),
            "C1d_propagation_non_vacuity_all_arms": bool(c1d_holds),
            "C1d_within_arm_rule_state_counterfactual_nonzero": bool(
                n_within_arm_cf_nonzero >= MIN_SEEDS_FOR_PASS
            ),
            "C1e_mech448_demotion_live_demotion_armed_arms": bool(c1e_holds),
            "C1f_mech449_active_nogo_live_gng_armed_arms": bool(c1f_holds),
            "C1g_fresh_e3_selection_sufficiency_all_arms": bool(c1g_holds),
            "arms_not_bit_identical": bool(arms_not_bit_identical),
        },
        "instrumentation_repair_note": (
            "V3-EXQ-699b repairs the V3-EXQ-699 instrument. The 699 PASS and its C1 "
            "readiness battery STAND (autopsy: the C1e/C1f non-vacuity floors are "
            "literally 0.0 and the active-frac gates saturate at exactly 1.0, both "
            "threshold-invariant to the defect); the 699 `levers_compound` finding is "
            "WITHDRAWN. 699's DV was accumulated once per ENV STEP from the action "
            "RETURNED by agent.select_action, and on a non-E3 tick agent.py:5430 "
            "returns the HELD action -- so it was hold-duration-weighted OCCUPANCY "
            "entropy standing in for the per-COMMITMENT construct MECH-448/449 act on. "
            "The bias is ALIGNED with the hypothesis (ARM_OFF is the reference for "
            "every delta and an unarmed agent perseverates longer, depressing its "
            "occupancy entropy and inflating every delta), so no sign-check was "
            "available and the corrected sign was UNKNOWN going in. This run must NOT "
            "be read as confirming or refuting `levers_compound` in advance; whatever "
            "it returns is the FIRST valid measurement of this composition. "
            "See defect_measurement_699_hold_weighting for the matched-instrumentation "
            "comparison, which is diagnostic context and not a finding. "
            "BEFORE ACTING ON THIS RUN, read `parked_successor` below: a further "
            "instrument repair (V3-EXQ-699c) is already written and committed but "
            "deliberately UNQUEUED, pending exactly the numbers this run measures."
        ),
        # ---------------------------------------------------------------------
        # PARKED SUCCESSOR -- read this before drawing conclusions from this run.
        # ---------------------------------------------------------------------
        # This pointer lives in the MANIFEST rather than only in the successor's
        # docstring because it must reach whoever reviews THIS run's result. The
        # successor is not in experiment_queue.json, so nothing else would surface
        # it: a reviewer has no reason to open a driver they do not know exists.
        "parked_successor": {
            "queue_id": "V3-EXQ-699c",
            "script": "experiments/v3_exq_699c_pcomp_demotion_x_gonogo_fixed_n.py",
            "status": "written, committed (ree-v3 cb35a38211), validated, NOT queued",
            "fixes": (
                "P2 sample size N is set by SURVIVAL -- CausalGridWorldV2 ends an "
                "episode on death (causal_grid_world.py:2626) and steps_per_episode "
                "caps below its 500-step limit -- so N differs across arms WITHIN a "
                "seed and the plug-in entropy estimator's ~(K-1)/(2N) downward bias "
                "becomes ARM-DEPENDENT and signed: an arm dying sooner reads as lower "
                "committed-class entropy for a purely statistical reason. 699c runs P2 "
                "to a FIXED N (400 genuine E3 selections per cell, capped at 800 "
                "episodes) so the bias is common-mode by construction, and applies a "
                "Miller-Madow correction for the residual K_obs term."
            ),
            "why_not_queued": (
                "The MAGNITUDE is unknown and THIS RUN measures it. E3 firing is "
                "event-driven (clock.advance() fires on a pending phase reset; "
                "agent.py:5430 runs select() whenever _last_action is None), so an "
                "episode RESET buys a free selection -- which means a death-prone cell "
                "running many short episodes earns a HIGHER yield and PARTIALLY "
                "COMPENSATES the very spread 699c removes. Estimated at ~0.0136 nats "
                "(27% of COMPOSITION_LIFT_MARGIN=0.05) without that compensation and "
                "~0.005 (~10%) with it. Not derivable from any recorded manifest: "
                "V3-EXQ-699 recorded no fresh-selection counts at all."
            ),
            "decision_rule": (
                "From this run's arm_results, compute the within-seed spread in "
                "n_fresh_select across the four arms, then the implied bias "
                "differential (K_obs-1)/2 * (1/N_min_arm - 1/N_max_arm) using each "
                "cell's own n_unique_committed_classes for K_obs. Queue 699c only if "
                "that differential is a material fraction of "
                "COMPOSITION_LIFT_MARGIN=0.05 nats. If it is small, leave it parked "
                "and record that it was considered."
            ),
            "fields_to_read": [
                "n_fresh_select", "n_latched", "fresh_select_yield",
                "replication_factor", "n_p2_ticks", "n_unique_committed_classes",
            ],
            "caveat_if_queued": (
                "Re-check 699c's TARGET_FRESH_SELECT_PER_CELL=400 against this run's "
                "measured yield before queueing -- 400 was sized assuming ~4000 P2 env "
                "steps per cell at yield ~0.1 (supported by V3-EXQ-785a's measured "
                "0.111/0.143/0.111, but not yet confirmed for THIS config). Note the "
                "tradeoff: equal-N trades unequal SAMPLE SIZE for unequal DURATION."
            ),
            "full_rationale": (
                "the STATUS: PARKED banner at the top of the 699c driver's module "
                "docstring"
            ),
        },
    }

    total_seeds = len(ARMS) * len(seeds)
    total_completed = len(all_rows)

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "interpretation_label": label,
        "composition_verdict": composition_verdict,
        "interpretation": interpretation,
        "seeds": seeds,
        "n_arms": len(ARMS),
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "h_signature": int(H_SIGNATURE),
            "composition_margin": float(COMPOSITION_MARGIN),
            "composition_lift_margin": float(COMPOSITION_LIFT_MARGIN),
            "composition_min_seeds": int(COMPOSITION_MIN_SEEDS),
            "frac_pre_ge2_floor": float(FRAC_PRE_GE2_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "consumed_magnitude_ceil": float(CONSUMED_MAGNITUDE_CEIL),
            "crf_min_minted": int(CRF_MIN_MINTED),
            "crf_n_active_floor": int(CRF_N_ACTIVE_FLOOR),
            "crf_frac_active_floor": float(CRF_FRAC_ACTIVE_FLOOR),
            "crf_dist_floor": float(CRF_DIST_FLOOR),
            "prop_nonvac_floor": float(PROP_NONVAC_FLOOR),
            "min_ticks_per_class": int(MIN_TICKS_PER_CLASS),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "lr_lpfc_bias": float(LR_LPFC_BIAS),
            "use_candidate_rule_field": bool(USE_CANDIDATE_RULE_FIELD),
            "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
            "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
            "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
            "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
            "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
            "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
            "f_eligibility_envelope_floor": float(F_ELIGIBILITY_ENVELOPE_FLOOR),
            "f_eligibility_dn_sigma": float(F_ELIGIBILITY_DN_SIGMA),
            "demotion_active_frac_floor": float(DEMOTION_ACTIVE_FRAC_FLOOR),
            "excluded_count_floor": float(EXCLUDED_COUNT_FLOOR),
            "use_f_eligibility_adaptive_floor": bool(USE_F_ELIGIBILITY_ADAPTIVE_FLOOR),
            "f_eligibility_adaptive_mean_factor": float(F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR),
            "use_dacc": bool(USE_DACC),
            "gng_perseveration_floor": float(GNG_PERSEVERATION_FLOOR),
            "gng_safety_floor": float(GNG_SAFETY_FLOOR),
            "gng_protect_min_eligible": int(GNG_PROTECT_MIN_ELIGIBLE),
            "nogo_active_frac_floor": float(NOGO_ACTIVE_FRAC_FLOOR),
            "nogo_suppressed_floor": float(NOGO_SUPPRESSED_FLOOR),
            "sd056_weight": float(SD056_WEIGHT),
            "crf_persist_rules_across_episode_reset": True,
        },
        "acceptance_criteria": {
            "C1_substrate_exercisable_and_levers_live": c1_holds,
            "C1a_class_axis_exercisable_all_arms": c1a_holds,
            "C1b_gapa_divergence_all_arms": c1b_holds,
            "C1c_rule_field_differentiated_matured_all_arms": c1c_holds,
            "C1d_propagation_non_vacuity_all_arms": c1d_holds,
            "C1d_n_within_arm_counterfactual_nonzero": int(n_within_arm_cf_nonzero),
            "C1e_mech448_demotion_live_demotion_armed_arms": c1e_holds,
            "C1f_mech449_active_nogo_live_gng_armed_arms": c1f_holds,
            "arms_not_bit_identical": arms_not_bit_identical,
            "max_arm_committed_class_entropy_spread": round(max_arm_spread, 6),
            "composition_verdict": composition_verdict,
            "n_seeds_compound": int(n_compound),
            "n_seeds_cancel": int(n_cancel),
            "n_seeds_neutral": int(n_neutral),
            "per_seed_deltas": per_seed_deltas,
            "mean_committed_class_entropy_off": round(off_mean_dv, 6),
            "mean_committed_class_entropy_dem": round(dem_mean_dv, 6),
            "mean_committed_class_entropy_gng": round(gng_mean_dv, 6),
            "mean_committed_class_entropy_both": round(both_mean_dv, 6),
            "mean_interaction_term": round(
                _mean([v["interaction"] for v in per_seed_deltas.values()]), 6
            ),
        },
        "secondary_negative_control_not_load_bearing": {
            "note": (
                "Within-class-representative entropy is a NEGATIVE CONTROL: the rule bias "
                "is class-keyed (compute_bias broadcasts one rule_state across K), so it "
                "cannot move within-class selection -> ~null across arms is EXPECTED, "
                "confirming the levers act on the committed-class axis (the load-bearing DV)."
            ),
            "within_class_rep_entropy_mean_by_arm": {
                "ARM_OFF": round(_mean([r["mean_within_class_rep_entropy_nats"] for r in off_rows]), 6),
                "ARM_DEM": round(_mean([r["mean_within_class_rep_entropy_nats"] for r in dem_rows]), 6),
                "ARM_GNG": round(_mean([r["mean_within_class_rep_entropy_nats"] for r in gng_rows]), 6),
                "ARM_BOTH": round(_mean([r["mean_within_class_rep_entropy_nats"] for r in both_rows]), 6),
            },
            # `selected_class_entropy_mean_by_arm` is DROPPED (autopsy requirement 5):
            # it duplicated the primary DV to 6dp on all 12 arm-seeds in 699.
            # Replaced by the DEFECT-MEASUREMENT block below, which is the first
            # direct read of the 699 contamination's magnitude AND sign.
            "lateral_pfc_bias_abs_mean_by_arm": {
                "ARM_OFF": round(_mean([r["mean_lateral_pfc_bias_abs"] for r in off_rows]), 8),
                "ARM_DEM": round(_mean([r["mean_lateral_pfc_bias_abs"] for r in dem_rows]), 8),
                "ARM_GNG": round(_mean([r["mean_lateral_pfc_bias_abs"] for r in gng_rows]), 8),
                "ARM_BOTH": round(_mean([r["mean_lateral_pfc_bias_abs"] for r in both_rows]), 8),
            },
        },
        # ----- THE 699 DEFECT, MEASURED (autopsy routing requirements 2/3/4) -------
        # NOT load-bearing for the composition verdict. This block exists so the
        # hold-duration-weighting defect's magnitude and DIRECTION become measured
        # rather than argued, for this run and as a corpus reference point.
        #
        # The autopsy's alignment argument predicts: ARM_OFF (the reference arm for
        # every delta, and the only arm with neither selection-perturbing lever)
        # perseverates MOST, so it carries the LONGEST holds and the LARGEST
        # entropy_defect_delta -- which is what inflated every d_* in 699. If
        # hold_duration_mean is instead flat across arms, the 699 distortion is
        # bounded rather than confirmed. Either way it is now on the record.
        "defect_measurement_699_hold_weighting": {
            "note": (
                "Per-commitment (PRIMARY) vs per-env-step occupancy (699's DV) "
                "entropy, side by side, plus the hold durations that separate them. "
                "entropy_defect_delta = occupancy - per_commitment. Diagnostic only; "
                "no acceptance criterion reads this block."
            ),
            "fresh_select_yield_by_arm": {
                a: round(_mean([r["fresh_select_yield"] for r in rows]), 6)
                for a, rows in (
                    ("ARM_OFF", off_rows), ("ARM_DEM", dem_rows),
                    ("ARM_GNG", gng_rows), ("ARM_BOTH", both_rows),
                )
            },
            "replication_factor_by_arm": {
                a: round(_mean([r["replication_factor"] for r in rows]), 6)
                for a, rows in (
                    ("ARM_OFF", off_rows), ("ARM_DEM", dem_rows),
                    ("ARM_GNG", gng_rows), ("ARM_BOTH", both_rows),
                )
            },
            "hold_duration_mean_by_arm": {
                a: round(_mean([r["hold_duration_mean"] for r in rows]), 6)
                for a, rows in (
                    ("ARM_OFF", off_rows), ("ARM_DEM", dem_rows),
                    ("ARM_GNG", gng_rows), ("ARM_BOTH", both_rows),
                )
            },
            "occupancy_class_entropy_mean_by_arm": {
                a: round(_mean([r["occupancy_class_entropy_nats"] for r in rows]), 6)
                for a, rows in (
                    ("ARM_OFF", off_rows), ("ARM_DEM", dem_rows),
                    ("ARM_GNG", gng_rows), ("ARM_BOTH", both_rows),
                )
            },
            "entropy_defect_delta_mean_by_arm": {
                a: round(_mean([r["entropy_defect_delta_nats"] for r in rows]), 6)
                for a, rows in (
                    ("ARM_OFF", off_rows), ("ARM_DEM", dem_rows),
                    ("ARM_GNG", gng_rows), ("ARM_BOTH", both_rows),
                )
            },
            "verdict_under_699_instrumentation": _composition_verdict_from(
                off_rows, dem_rows, gng_rows, both_rows,
                key="occupancy_class_entropy_nats",
            ),
            "verdict_under_699b_instrumentation": _composition_verdict_from(
                off_rows, dem_rows, gng_rows, both_rows,
                key="committed_class_entropy_nats",
            ),
        },
        "interpretation_grid": {
            "PASS_levers_compound": (
                "C1 holds (both levers genuinely engaged in their armed arms, class axis + "
                "GAP-A + matured field present) AND the co-armed committed-class entropy "
                "lift beats the better single lever by margin on a majority of seeds. The "
                "MECH-448 demotion and MECH-449 Go/No-Go COMPOUND at C2: they convert where "
                "each alone (654i / 654j) could not -> demotion + Go/No-Go are "
                "COMPOSITION-READY (co-arm both in the conversion_ceiling_campaign:FULLSTACK "
                "arm). Route to /governance: feed the full-stack composition-readiness gate."
            ),
            "PASS_levers_neutral": (
                "C1 holds AND the co-armed cell is within margin of the better single lever "
                "(neither compounds nor cancels). The two levers do NOT collide -> SAFE to "
                "co-arm in the full stack, but the composition adds no marginal committed-"
                "class diversity on the GAP-B bank (the conversion ceiling is not lifted by "
                "this pair alone). Composition-readiness: co-armable; the full-stack arm's "
                "lift, if any, must come from the other faces (P2 root-C de-commit, P3 OFC) "
                "or the assembled stack. Feed the full-stack composition-readiness gate."
            ),
            "PASS_levers_cancel": (
                "C1 holds AND the co-armed committed-class entropy is WORSE than the better "
                "single lever (or below the ARM_OFF baseline) by margin on a majority of "
                "seeds. The MECH-448 demotion and MECH-449 Go/No-Go CANCEL at C2 -- a "
                "DESTRUCTIVE within-selection-face interaction (the Factor-A x Factor-B 689a "
                "signature). The conversion_ceiling_campaign:FULLSTACK arm must NOT co-arm "
                "BOTH levers; drop one (mirroring how Factor B was dropped). Route to "
                "/governance: amend the full-stack composition matrix to exclude the "
                "co-arming; feed the composition-readiness gate."
            ),
            "FAIL_substrate_not_ready_requeue": (
                "A lever was vacuous on its armed arm (demotion all-admit excluded_count==0, "
                "or Go/No-Go inert nothing-suppressed), and/or the committed-class axis was "
                "not exercisable, and/or GAP-A divergence absent, and/or the matched rule "
                "field did not mature, and/or propagation vacuous, and/or the four arms were "
                "bit-identical (no lever moved committed selection). The interaction could "
                "NOT be measured -- NOT a falsification of MECH-448 / MECH-449. Route to "
                "substrate enrichment / re-queue at an adequate substrate; do NOT weaken."
            ),
        },
        "arm_results": arm_results,
    }


def _build_manifest(
    result: Dict[str, Any],
    timestamp_utc: str,
    dry_run: bool,
) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES_RUN_ID,
        # Both the original 699 run and the in-flight 699a faithful re-run are
        # superseded: 699a runs the UNCHANGED driver and reproduces the same
        # hold-duration-weighted DV, so its composition verdict is not adjudicable
        # either. Whichever manifests exist at governance time, 699b is the valid
        # measurement of this composition.
        "supersedes_queue_ids": list(SUPERSEDES_QUEUE_IDS),
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "interpretation_label": result["interpretation_label"],
        "composition_verdict": result["composition_verdict"],
        "interpretation": result["interpretation"],
        "evidence_direction_note": (
            f"V3-EXQ-699b INSTRUMENT REPAIR of V3-EXQ-699 (supersedes "
            f"{SUPERSEDES_RUN_ID}; also supersedes V3-EXQ-699a, a faithful re-run of the "
            f"unchanged 699 driver that reproduces the same defect). Per "
            f"failure_autopsy_V3-EXQ-699_2026-07-20 (REE_assembly ac2fb64028): the 699 "
            f"PASS and its C1 readiness battery STAND, but the `levers_compound` finding "
            f"is WITHDRAWN -- 699's DV was accumulated once per ENV STEP from the action "
            f"RETURNED by agent.select_action, and on a non-E3 tick agent.py:5430 returns "
            f"the HELD action, so it measured hold-duration-weighted OCCUPANCY entropy in "
            f"place of the per-COMMITMENT construct MECH-448/449 act on. The bias was "
            f"ALIGNED with the hypothesis, so the corrected sign was UNKNOWN: this is a "
            f"WITHDRAWAL, not a reversal, and 699b's result must not be read as "
            f"confirming or refuting `levers_compound` in advance. 699b records the "
            f"committed class ONLY on a genuine E3 selection, emits n_fresh_select / "
            f"n_latched / fresh_select_yield so the replication factor is MEASURED rather "
            f"than inferred, emits hold-duration distributions per arm-seed, and reports "
            f"BOTH readouts (per-commitment PRIMARY and 699's occupancy DV verbatim) so "
            f"the defect's magnitude and SIGN are on the record. "
            f"V3-EXQ-699 conversion_ceiling_campaign:P-comp SELECTION-FACE COMPOSITION "
            f"characterization (DIAGNOSTIC; claim_ids=[MECH-448, MECH-449] CONTEXT only -- "
            f"diagnostics are scoring-excluded, this PROMOTES NOTHING and weights no claim's "
            f"governance confidence). A 2x2 (MECH-448 demotion {{OFF,ON}} x MECH-449 Go/No-Go "
            f"{{OFF,ON}}) over a MATCHED stack: the swept variable is the LEVER COMBINATION, "
            f"NOT use_candidate_rule_field (which is a matched CONSTANT True on all four arms, "
            f"the differentiated conversion source). use_f_eligibility_adaptive_floor=True "
            f"(689e PASS) + use_dacc + the 569i top_k shortlist scaffold + the modulatory "
            f"authority/routing + MECH-341 + SD-056 + the trained lateral_pfc bias head + the "
            f"matured CRF pool are all MATCHED CONSTANTS; ONLY use_f_eligibility_demotion and "
            f"use_go_nogo_constitution toggle. PRIMARY DV = committed-class entropy (C2). The "
            f"selection face is exhausted lever-by-lever (Factor A inert 689a; Factor B "
            f"REFUTED 689c; demotion fails C2 alone 654i; Go/No-Go fails C2 alone 654j) and "
            f"both levers are face-validated alone (689d demotion 0.938 vs 0.371; 689g "
            f"Go/No-Go 3/3) -- this characterizes whether they COMPOUND or CANCEL when "
            f"co-armed (we KNOW Factor A x Factor B CANCELLED at 689a; the demotion x Go/No-Go "
            f"interaction is UNKNOWN). C1 (readiness/non-vacuity) = class axis exercisable AND "
            f"GAP-A divergence AND matured rule field AND propagation non-vacuous (ALL arms) "
            f"AND -- PER-ARM CONDITIONAL -- the MECH-448 demotion LIVE-and-excluding on the "
            f"demotion-armed arms (ARM_DEM, ARM_BOTH) AND the MECH-449 active No-Go LIVE-and-"
            f"suppressing on the gng-armed arms (ARM_GNG, ARM_BOTH). C_INTERACTION (PRIMARY) = "
            f"per-seed paired deltas vs ARM_OFF -> three-way verdict {{levers_compound | "
            f"levers_cancel | levers_neutral}} on >= {COMPOSITION_MIN_SEEDS}/3 seeds "
            f"(margin {COMPOSITION_MARGIN}). outcome=PASS = characterization succeeded (C1 met "
            f"+ arms not bit-identical), interpretation_label carries the composition verdict; "
            f"outcome=FAIL = substrate_not_ready_requeue (a lever vacuous on its armed arm, or "
            f"readiness unmet, or arms bit-identical). NO weakens branch -- it CHARACTERIZES "
            f"the lever interaction, it does not falsify a claim. Result feeds "
            f"conversion_ceiling_campaign:FULLSTACK composition-readiness "
            f"(label={result['interpretation_label']}; "
            f"C1={result['acceptance_criteria']['C1_substrate_exercisable_and_levers_live']}; "
            f"verdict={result['composition_verdict']})."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "2x2 (MECH-448 demotion {OFF,ON} x MECH-449 Go/No-Go {OFF,ON})",
            "arms": "ARM_OFF / ARM_DEM (demotion only) / ARM_GNG (Go/No-Go only) / ARM_BOTH (co-armed)",
            "swept_variable": "lever combination (use_f_eligibility_demotion x use_go_nogo_constitution)",
            "matched_constants": (
                "use_candidate_rule_field=True (differentiated conversion source) + "
                "use_f_eligibility_adaptive_floor=True (689e; inert when demotion OFF) + "
                "use_dacc=True (MECH-260 perseveration No-Go feed) + top_k shortlist scaffold "
                "+ modulatory authority/routing (643a) + MECH-341 + MECH-313 + V_s minimal + "
                "use_gated_policy + use_lateral_pfc_analog (lateral_pfc_train_rule_bias_head=True, "
                "TRAINED in P1 REINFORCE) + SD-056 all levers + the matured/maintained CRF pool"
            ),
            "primary_dv": "committed-class entropy (the C2 DV, matched to 654i/654j)",
            "secondary_negative_control": "within-class-representative entropy (expected ~null)",
            "phases": "P0 e2-train (CRF matures) -> P1 frozen-encoder bias-head REINFORCE -> P2 frozen measurement",
            "per_arm_conditional_readiness": (
                "C1e (MECH-448 demotion live + excluding) checked ONLY on the demotion-armed "
                "arms (ARM_DEM, ARM_BOTH); C1f (MECH-449 active No-Go live + suppressing) "
                "checked ONLY on the gng-armed arms (ARM_GNG, ARM_BOTH); ARM_BOTH must satisfy both"
            ),
            "demotion_validated_by": "V3-EXQ-689d PASS 2026-06-20 (committed entropy 0.938 vs hard-top-k 0.371); MECH-448 provisional",
            "nogo_validated_by": "V3-EXQ-689g PASS 3/3 2026-06-21 (go_nogo_converts_gated_channel; MECH-449/ARC-107); MECH-449 provisional",
            "adaptive_floor_validated_by": "V3-EXQ-689e PASS 2026-06-21 (channel-adaptive envelope excludes on the arc_062 bank)",
            "cancellation_precedent": "Factor A x Factor B CANCELLED at V3-EXQ-689a (the destructive within-selection-face interaction this characterization guards the full-stack arm against)",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "sd056_output_norm_clamp": SD056_OUTPUT_NORM_CLAMP,
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-699 P-comp MECH-448 demotion x MECH-449 Go/No-Go composition at committed-class entropy (2x2 diagnostic)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        p2 = DRY_RUN_P2
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1 = P1_BIAS_TRAIN_EPISODES
        p2 = P2_MEASUREMENT_EPISODES
        steps = STEPS_PER_EPISODE

    # SETUP-TIME anchor-reachability replay: raises AnchorUnreachable before any
    # compute is spent if a C1 gate is unmeetable by the 699 control that already
    # met it. Runs in dry-run too -- the frozen reference is scale-independent, so
    # a --dry-run smoke exercises the guard exactly as a full run does.
    assert_c1_anchors_reachable()

    # C1(g) has no frozen reference (see the guard block). Report its headroom
    # from the shipped run shape instead, so an operator sees the risk BEFORE the
    # run rather than in a substrate_not_ready_requeue manifest.
    print(
        "anchor_note: fresh_e3_selection_sufficiency_all_arms is UNGUARDED -- no "
        "frozen reference exists, because V3-EXQ-699 recorded NO fresh-selection "
        f"count at all. floor={MIN_FRESH_SELECT_PER_CELL} genuine E3 selections per "
        f"cell; P2={p2} ep x {steps} steps = {p2 * steps} env steps per cell IF "
        "episodes run to full length (they do not -- CausalGridWorldV2 ends an "
        "episode on death, and 699's recorded P2 env-step counts ranged 461..12000).\n"
        "  n_p2_ticks/10 is a reasonable SCALE for the fresh-selection count but is a "
        "slight UNDER-estimate, not an upper bound. E3 firing is EVENT-DRIVEN: "
        "heartbeat/clock.py advance() fires an E3 tick IMMEDIATELY on a pending phase "
        "reset (a salient event) and resets the phase counter, and agent.py:5430 also "
        "runs select() on a non-E3 tick whenever _last_action is None (so an episode "
        "RESET buys a free selection). e3_steps_per_tick=10 is therefore the MAXIMUM "
        "interval between ticks. Measured on a real full-scale run, V3-EXQ-785a "
        "recorded fresh_select_yield 0.111 / 0.143 / 0.111 across its arms -- just "
        "above 1/10, consistent with that.\n"
        "  DO NOT calibrate this from a --dry-run. Short DRY_RUN_STEPS episodes end in "
        "~10-17 ticks and E3 fires near episode start, so toy yields run 0.5-1.0 and "
        "OVERSTATE the real value by 4-8x (independently observed in the V3-EXQ-707c "
        "smoke, 0.67-1.00).\n"
        "  ONE EFFECT THAT IS NOT QUANTIFIED, and which this run settles: because an "
        "episode reset yields a free selection, a DEATH-PRONE cell running many SHORT "
        "episodes gets a higher yield than a long-surviving one. That partially "
        "COMPENSATES the survival-driven spread in sample size (on 699's seed 44 the "
        "raw P2-tick ratio between ARM_DEM and ARM_GNG is ~3.8x, but if every one of "
        "the 60 P2 episodes contributes a free selection the N ratio falls to ~2.2x). "
        "Whether the residual spread matters against COMPOSITION_LIFT_MARGIN=0.05 is "
        "exactly what the emitted per-cell counts below decide.\n"
        "  That is exactly why n_fresh_select / n_latched / fresh_select_yield / "
        "replication_factor are emitted per arm-seed here: the factor is MEASURED for "
        "the first time rather than inferred. Read those fields before drawing any "
        "conclusion about sample adequacy. Treat a C1(g) substrate_not_ready_requeue "
        "as an instrument-scale finding, not a substrate verdict.",
        flush=True,
    )

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_episodes=p1,
        p2_episodes=p2,
        steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"C1={result['acceptance_criteria']['C1_substrate_exercisable_and_levers_live']} "
        f"verdict={result['composition_verdict']} "
        f"label={result['interpretation_label']}",
        flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
    sys.exit(0)
