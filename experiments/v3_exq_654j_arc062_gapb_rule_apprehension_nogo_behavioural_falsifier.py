#!/opt/local/bin/python3
"""
V3-EXQ-654j -- arc_062_rule_apprehension:GAP-B behavioural falsifier for
MECH-309 / ARC-062, engaging the MECH-449 (ARC-107) GO/NO-GO ELIGIBILITY
CONSTITUTION (the ACTIVE No-Go OPPONENCY leg) on top of the MECH-448 (ARC-107)
RANK-PRESERVING F->ELIGIBILITY DEMOTION lever.

SUPERSEDES V3-EXQ-654i (FAIL non_contributory, 2026-06-22; run_id
v3_exq_654i_arc062_gapb_rule_apprehension_behavioural_falsifier_20260622T014706Z_v3;
self_route_label conversion_ceiling_persists_despite_demotion_route_mech449; CONFIRMED via
failure_autopsy_V3-EXQ-654i_2026-06-22). 654i was the CLEANEST instance of the conversion
ceiling: ALL C1 readiness/non-vacuity gates MET and non-degenerate (rule field matured 0.913,
propagation non-vacuous 0.027, the MECH-448 demotion lever LIVE and excluding 18.4 candidates)
-- yet the load-bearing C2 committed-class entropy lift FAILED. The differentiated
rule-apprehension bias REACHES committed action but does NOT convert to committed-class
diversity EVEN UNDER the live MECH-448 demotion lever. Four-layer diagnosis = prerequisites
missing: conversion needs the ACTIVE No-Go WITHDRAWAL that rank-preserving demotion is
order-preserving over F and structurally CANNOT express (MECH-448 demotes F to an eligibility
envelope, but the within-eligible argmin has no mechanism to suppress the F-eligible-but-
perseverative winner). Cross-substrate corroboration of V3-EXQ-485k (OFC devaluation face):
demotion != active No-Go; two faces, one fix (MECH-449).

RE-DERIVE BRAKE RELEASED (this test is the brake's named off-ramp, NOT a same-substrate
re-queue). The 654i autopsy FIRED the re-derive brake (16-17 prior substrate_ceiling /
non_contributory autopsies on MECH-309 / ARC-062) and REFUSED a same-substrate 654j re-running
the MECH-448-only (demotion) stack -- the only permitted next test must engage MECH-449 as a
DIFFERENT mechanism. MECH-449 (Go/No-Go eligibility constitution, ARC-107 opponency leg) was
BUILT 2026-06-21 (ree_core/predictors/e3_selector.py._go_nogo_eligibility_gate;
E3Config.use_go_nogo_constitution) and VALIDATED: the selection-face falsifier V3-EXQ-689g
PASSed 3/3 (go_nogo_converts_gated_channel; claim_ids=[MECH-449, ARC-107]); MECH-449 promoted
candidate->provisional 2026-06-22. The brake's named upstream substrate is therefore
IMPLEMENTED + VALIDATED and this is a different mechanism -> brake released.

THE 654j CHANGE vs 654i (everything else MATCHED -- the now-working CRF stack, the GAP-A
e2_world_forward channel source, the C1a-d preconditions, the committed-class entropy PRIMARY
DV, the MECH-448 demotion conversion constant): the MECH-449 Go/No-Go constitution is ARMED as
a matched-stack CONSTANT on BOTH arms (use_go_nogo_constitution=True), engaging the ACTIVE
No-Go. The No-Go is fed ECOLOGICALLY (NOT synthetic injection) via the BUILT default-loop
MECH-260 reuse: with use_dacc=True the dACC bundle's recency-share `suppression` vector
populates the gate's `perseveration` axis (the autopsy's named dependency "active No-Go
withdrawal (MECH-260 generalised -> MECH-449)"), so a candidate whose first-action class is
perseverated above gng_perseveration_floor is DROPPED from the F-eligible set BEFORE the
within-eligible _modulatory_accum arbitration -- the active withdrawal demotion-alone could not
express. The MECH-448 demotion floor is now the CHANNEL-ADAPTIVE (mean-relative) floor
(use_f_eligibility_adaptive_floor=True; V3-EXQ-689e PASS on the arc_062 bank), which
auto-calibrates per channel and REPLACES 654i's per-(arm,seed) _calibrate_envelope_floor manual
recalibration. use_dacc + use_go_nogo_constitution + use_f_eligibility_demotion +
use_f_eligibility_adaptive_floor are ALL matched-stack constants on BOTH arms; the ONLY swept
variable remains use_candidate_rule_field. A NEW C1f readiness gate requires the active No-Go to
be LIVE and ACTUALLY SUPPRESSING (non-vacuous) on both arms; an inert No-Go self-routes
substrate_not_ready_requeue, NEVER a false result.

PREDECESSOR LINEAGE (V3-EXQ-654g, FAIL non_contributory, 2026-06-19; run_id
v3_exq_654g_arc062_gapb_rule_apprehension_behavioural_falsifier_20260619T213118Z_v3;
interpretation_label shared_selection_authority_conversion_ceiling_route_implement_substrate).
654g was the CRF lineage TERMINUS: it proved the CRF stack is DONE (crf_frac_active
cleared the 0.30 floor, conflict-gate lockout gone, propagation non-vacuous) AND it
wired the 569i-validated TOP-K SHORTLIST conversion -- yet C1 held while C2 (committed-
class entropy lift) FAILED. 654g self-routed the SHARED selection-authority CONVERSION
CEILING (behavioral_diversity_isolation:GAP-A/GAP-I; MECH-439 F-dominance live root):
the differentiated rule_state's bias REACHES committed action but does NOT move the
F-dominated committed argmax (F monopolises ~88-89%% of E3 committed-selection variance,
V3-EXQ-571), even under the thin-margin top-k shortlist. The fix is NOT another CRF amend
and NOT a parametric near-tie lever -- it is the constitutional one.

WHY 654h NOW (the ceiling has operationally lifted). The 2026-06-21 governance cycle
promoted MECH-448 to PROVISIONAL on the V3-EXQ-689d PASS: the rank-preserving
F->eligibility demotion converts committed-action diversity on the GAP-A foraging
substrate (committed entropy 0.938 vs ARM_OFF hard-top-k 0.371; all four load-bearing
acceptance criteria met -- readiness, rank-preserving, C_PRIMARY strict-above both
collapsed controls, C_SAFETY). That demotion lever operationally LIFTS the shared
MECH-439 F-dominance conversion ceiling (closure node
behavioral_diversity_isolation:GAP-I) that 654g's C2 committed-class-entropy DV was
stuck under. The substrate-ceiling audit (scripts/check_substrate_ceiling_audit.py)
flags MECH-309 and ARC-062 as "ceiling-may-have-lifted." 654h is the FIRST DOWNSTREAM
confirmation that the demotion lever generalises off the GAP-A foraging substrate onto
the GAP-B rule-apprehension composite.

WHAT MECH-448 DOES (e3_selector.py: _f_eligibility_envelope + the "f_demotion" mode of
the shortlist-then-modulate block). F decides who is ELIGIBLE to compete, not who wins.
F is renormalised against the COMPETING FIELD by a divisive-normalisation analog with an
ABSOLUTE share floor:
    merit[i] = clamp(raw_scores.max() - raw_scores[i], min=0)   # best (lowest F) = highest merit
    pooled   = f_eligibility_dn_sigma + merit.sum()
    elig[i]  = merit[i] / pooled                                # share of the competing field
    eligible = { i : elig[i] >= f_eligibility_envelope_floor }  # ABSOLUTE share floor
A decisive F-winner commands most of the merit share so others fall below the floor
(NARROW envelope); a near-tie spreads the share (WIDE envelope) -- the BG hyperdirect
conflict-grade emerging from the field structure, env-general (NOT the hard top-k count
569i used, which was env-conditional on the reef-bipartite guarantee and thinly cleared
2/3). elig is monotone in -F so the eligible set is an F-RANK PREFIX (rank-preserving;
F still RANKS within-eligible). Within the eligible set the EXISTING _modulatory_accum
arbitration (the lateral_pfc CRF rule-bias + the routed cand_world_summary range) picks
the committed action (argmin committed) with F REMOVED from the final argmin.

THE 654h CHANGE vs 654g (everything else MATCHED -- the now-working CRF stack, the GAP-A
e2_world_forward channel source, the C1a-d preconditions, the committed-class entropy
PRIMARY DV, and the NO-weakens routing are UNCHANGED): the MECH-448 demotion lever is
ARMED as a matched-stack CONSTANT on BOTH arms (use_f_eligibility_demotion=True +
f_eligibility_envelope_floor=0.30 + f_eligibility_dn_sigma=0.0). Per e3_selector.py the
f_demotion mode TAKES PRECEDENCE over the retained top_k shortlist keys (which stay
present documenting the 569i lineage but no longer construct the eligible set), so the
eligibility-set construction becomes the graded DN envelope on BOTH arms identically;
the within-eligible _modulatory_accum arbitration is identical to 654g. This is exactly
the V3-EXQ-689d ARM_ON composition (shortlist_mode=top_k kept + use_f_eligibility_demotion
=True). 654h adds NO further CRF amend; CRF is DONE.

Single-variable arm contrast
----------------------------
Both arms run the SAME matched stack -- SP-CEM (Layer A), GAP-A shared candidate-summary
source candidate_summary_source="e2_world_forward" (V3-EXQ-649) with e2 TRAINED ONLINE in
P0 (SD-056 contrastive; rollout-norm clamp ON per the 643a numerical-stability lesson),
the modulatory-bias-selection-authority gate (V3-EXQ-643a float32 fix) + channel routing
(cand_world_summary) + the MECH-448 RANK-PRESERVING F->ELIGIBILITY DEMOTION conversion
(use_f_eligibility_demotion=True, graded DN envelope floor 0.30), MECH-341
stratified-select, MECH-313 noise floor, the V_s minimal stack, AND
use_lateral_pfc_analog=True + use_gated_policy=True with the SD-033a bias head UN-ZEROED
(lateral_pfc_train_rule_bias_head=True) so the rule_state reaches a non-zero per-candidate
E3 score-bias.

The ONLY swept variable is use_candidate_rule_field (which auto-sets
use_candidate_rule_source on the agent):
  ARM_OFF: use_candidate_rule_field=False -> LateralPFCAnalog rule_state is the
           LEGACY delta_proj(z_delta) + world_pool_weight * world_proj(z_world)
           EMA source -- the 543/598b COLLAPSED rule_state. (Demotion lever ON, matched.)
  ARM_ON:  use_candidate_rule_field=True (+ crf_persist=True) -> LateralPFCAnalog
           consumes crf_source -- the field's DIFFERENTIATED active-rule-stack
           rule_state, MATURED across episodes. (Demotion lever ON, matched.)

Dependent variable -- COMMITTED-CLASS diversity (mechanistically matched)
-------------------------------------------------------------------------
CODE-CONFIRMED (agent._candidate_world_summaries + lateral_pfc.compute_bias):
the per-candidate summaries the bias channels consume are keyed on the candidate's
FIRST action (e2.world_forward(z0, a_first)), and compute_bias broadcasts a single
rule_state across all K candidates. So within a first-action class every candidate
receives an IDENTICAL rule bias. Therefore the rule-creator moves WHICH CLASS is
committed (the committed-class axis), NOT within-class representative selection.
The matched readout is committed-class entropy (PRIMARY DV). Within-class-representative
entropy is retained as a SECONDARY NEGATIVE CONTROL (expected ~null).

Phases / budget
---------------
P0 (200 ep, e2 TRAINED online via SD-056 contrastive; bias head NOT trained; field
   matures across episodes via crf_persist + the 666c-validated maintenance levers).
P1 (90 ep, encoder FROZEN, bias head TRAINED via outcome-coupled REINFORCE; field
   continues to mature + maintain): the GAP-D trained-bias-head window. No measurement.
P2 (60 ep, all FROZEN -- e2 + bias head; field persists + maintains; instrumentation
   ON, including the MECH-448 demotion-lever diagnostics): the behavioural measurement
   window.
Budget: 2 arms x 3 seeds x 350 ep x 200 steps = 420k steps total. The C2 paired lift +
the C1d propagation + C1e demotion-non-vacuity comparisons are within-seed -> a cloud
worker (689-lineage; prefer cloud over the Mac).

Pre-registered acceptance criteria
----------------------------------
  C1 (READINESS / non-vacuity -- protects against a false weakens; ANY C1 fail
      self-routes substrate_not_ready_requeue, NOT a falsification):
     (a) committed-class axis exercisable: frac_pre_ge2 > FRAC_PRE_GE2_FLOOR on a
         majority (>= 2/3) of seeds in BOTH arms.
     (b) GAP-A divergence real: consumed_summary_pairwise_dist_mean >
         CONSUMED_SPREAD_FLOOR (and bounded) on a majority of seeds in BOTH arms.
     (c) ARM_ON manipulation live AND MATURED: per-tick crf_n_active >=
         CRF_N_ACTIVE_FLOOR on >= CRF_FRAC_ACTIVE_FLOOR of P2 ticks AND >=
         CRF_MIN_MINTED distinct rules minted, on a majority of ARM_ON seeds.
     (d) PROPAGATION non-vacuity: paired-by-seed |mean_lateral_pfc_bias_abs(ARM_ON)
         - mean_lateral_pfc_bias_abs(ARM_OFF)| > PROP_NONVAC_FLOOR on a majority of
         seeds.
     (e) MECH-448 DEMOTION non-vacuity (the matched conversion constant is LIVE): the
         f_demotion envelope is active on >= DEMOTION_ACTIVE_FRAC_FLOOR of P2 ticks AND
         ACTUALLY EXCLUDES (mean f_eligibility_excluded_count > EXCLUDED_COUNT_FLOOR) on
         the divergent e2_world_forward pool, on a majority of seeds in BOTH arms. An
         all-admit envelope (excluded_count==0; a flat-F pool) means the demotion did
         nothing -> vacuous -> substrate_not_ready_requeue, NEVER a false weakens.
     (f) MECH-449 ACTIVE NO-GO non-vacuity (the new matched opponency constant is LIVE
         AND SUPPRESSING): the Go/No-Go constitution is active (go_nogo_constitution_active)
         on >= NOGO_ACTIVE_FRAC_FLOOR of P2 ticks AND it ACTUALLY removes candidates from
         the F-eligible set (mean go_nogo_n_safety_nogo + go_nogo_n_soft_applied >
         NOGO_SUPPRESSED_FLOOR; the perseveration No-Go fed by the MECH-260 recency-share
         vector), on a majority of seeds in BOTH arms. An inert No-Go (nothing suppressed;
         the perseveration axis never crosses gng_perseveration_floor) means the active
         opponency leg did nothing -> vacuous -> substrate_not_ready_requeue, NEVER a false
         weakens. Mirrors the V3-EXQ-689g go_nogo-actually-suppresses non-degeneracy gate.
  C2 (PRIMARY -- the falsifier): paired-by-seed lift in committed_class_entropy_nats
     of ARM_ON over ARM_OFF of at least C2_LIFT_MARGIN_NATS (strict-above-by-margin) on
     a majority (>= 2/3) of seeds.
  C3 (SECONDARY negative control): within-class-representative entropy + selected-class
     entropy + lateral_pfc bias range. Reported, not load-bearing.

Overall outcome (THREE branches; NO weakens -- the constitution is the deep fix)
--------------------------------------------------------------------------------
  PASS  = C1 (non-vacuous, incl the demotion lever live AND the active No-Go live and
          suppressing) AND C2 (committed-class lift) -> supports MECH-309 + ARC-062. The
          MECH-449 active Go/No-Go opponency leg CONVERTS the differentiated rule-apprehension
          bias into committed-class diversity where the MECH-448 demotion-alone stack
          (654i) could not -> closes behavioral_diversity_isolation:GAP-I.
  FAIL (C1 holds, C2 fails) = the matured + maintained + differentiated rule pool's bias
          REACHES committed action (C1d non-vacuous), the demotion lever is LIVE (C1e), AND
          the MECH-449 active No-Go is LIVE and ACTUALLY SUPPRESSING (C1f) -- but the
          differentiated rule_state STILL does not lift committed-class diversity EVEN UNDER
          active Go/No-Go opponency. The conversion ceiling persists WITH active No-Go: a
          GENUINE SIGNAL that the Go/No-Go opponency leg ALONE is insufficient for
          behavioural conversion on the GAP-B composite ->
          conversion_ceiling_persists_despite_active_nogo; non_contributory; ROUTE to the
          NEXT ARC-107 selector-constitution ledger component (the open components 2/4/5 of
          the BG constitution; commitment-dependent behavioural DVs need more of the ARC-107
          stack, NOT a falsification) / V4 directions. NOT a MECH-309 / ARC-062
          falsification. Do NOT weaken the claims.
  FAIL (C1 fails) = substrate not exercisable / not matured (C1c crf_frac_active < 0.30)
          / propagation vacuous (C1d) / demotion vacuous (C1e excluded_count==0, all-admit)
          / active No-Go vacuous (C1f: nothing suppressed) / class axis or GAP-A divergence
          absent -> substrate_not_ready_requeue; re-queue. Do NOT weaken.

Claims: [MECH-309, ARC-062] (only the PASS branch weights them, as supports; both FAIL
branches are pre-registered non_contributory). experiment_purpose = evidence. PROMOTES
NOTHING by itself. supersedes V3-EXQ-654i (conversion ceiling persisted despite the live
MECH-448 demotion lever; 654j engages the newly-built+validated MECH-449 active No-Go leg
as the distinct next mechanism, with the channel-adaptive MECH-448 floor replacing 654i's
manual per-(arm,seed) envelope-floor calibration).

STANDING-GUIDANCE NOTE (commitment-dependent behavioural DV on an incomplete BG layer): the
REE memory guidance warns that a commitment-dependent behavioural DV tested while the ARC-107
constitution is incomplete (ledger components 2/4/5 still open) risks re-deriving the
F-dominance conversion ceiling rather than returning a verdict. This test is adjudicated
runnable because it engages the newly-built+validated MECH-449 (a DIFFERENT mechanism than
the 654i demotion-only stack), and its three-branch NO-weakens map handles the incomplete-
layer case explicitly: a C1-holds/C2-fails outcome is routed to the next ARC-107 component
(NOT a weakens), and the C1f non-vacuity gate guarantees the No-Go genuinely engaged before
any conclusion is drawn.

See REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md (GAP-B),
REE_assembly/docs/architecture/mech_449_go_nogo_constitution.md (the active No-Go leg this engages),
REE_assembly/docs/architecture/mech_448_f_eligibility_demotion.md (the demotion lever + channel-adaptive floor),
REE_assembly/evidence/planning/arc_107_selector_constitution_design_2026-06-20.md,
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-654i_2026-06-22.json (the FAIL this supersedes; re-derive brake off-ramp),
experiments/v3_exq_689g_mech449_go_nogo_conversion_falsifier.py (the MECH-449 selection-face PASS this engages behaviourally),
experiments/v3_exq_689d_mech448_f_eligibility_demotion_falsifier.py (the MECH-448 PASS
this generalises; ARM_ON config ported here),
experiments/v3_exq_654i_arc062_gapb_rule_apprehension_behavioural_falsifier.py (the demotion-only predecessor),
experiments/v3_exq_569i_gapa_conversion_topk_shortlist_falsifier.py (the retained top-k
lineage the f_demotion mode now overrides),
ree-v3/CLAUDE.md (MECH-448/ARC-107 + modulatory-bias-selection-authority TOP-K shortlist +
ARC-063 crf-availability-maintenance + ARC-062 GAP-A/B/C/D + SD-056 entries),
experiments/v3_exq_598b_gap1_sd033a_bias_head_trainable_ablation.py (P1 bias-head
REINFORCE pattern mirrored here).
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
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_654j_arc062_gapb_rule_apprehension_nogo_behavioural_falsifier"
QUEUE_ID = "V3-EXQ-654j"
SUPERSEDES = "V3-EXQ-654i"
CLAIM_IDS: List[str] = ["MECH-309", "ARC-062"]
EXPERIMENT_PURPOSE = "evidence"

# CRF-gate calibration amend levers (crf-availability-maintenance at the CRF locus;
# landed ree-v3 main 42895f6, 2026-06-17). The 654d FIX: 654d proved GAP-A
# de-collapse (ARM_STD_G2, armed below + kept as a matched-stack constant) was the
# WRONG lever -- crf_frac_active stayed EXACTLY 0.0 on all 3 ARM_ON seeds even where
# the consumed_summary spread cleared the 0.05 floor, because the real GAP-B blocker
# is the CRF conflict-gate lockout (7-8 rules co-match -> theta(7)=0.15+0.25*6=1.65
# >> maintenance_floor 0.45 -> every matched rule gated out), INDEPENDENT of GAP-A.
# These three levers (consulted only under crf_mature_pool_dynamics; inert in ARM_OFF
# where no field is built) clear that lockout:
#   FAULT 1 -- sharpen the gate match cutoff so fewer of the clustered context_tags
#     co-match a per-tick context (n_matched 7-8 -> ~2-3).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
#   FAULT 2a -- cap n_competing in theta so it stays reachable (theta_floor +
#     theta_gain*cap = 0.15 + 0.25*3 = 0.90 < 1.0).
CRF_TOLERANCE_CONFLICT_CAP = 3
#   FAULT 2b -- couple maintained availability to the (capped) per-tick theta so the
#     maintained differentiated pool CLEARS the gate (max(maintenance_floor, theta+eps)).
CRF_MAINTENANCE_COUPLE_TO_THETA = True

# crf-availability-maintenance substrate levers (the 666c ARM_2 flags, validated
# V3-EXQ-666c PASS 2026-06-15; substrate_queue crf-availability-maintenance ready=True).
CRF_MAINTENANCE_FLOOR = 0.45   # 666c MAINTENANCE_FLOOR
CRF_MAINTENANCE_DECAY = 0.0    # 666c MAINTENANCE_DECAY (pure activity-silent hold)

# Within-class-representative signature horizon (SECONDARY negative control).
H_SIGNATURE = 3

# C2 (PRIMARY): paired-by-seed committed-class entropy lift of ARM_ON over ARM_OFF.
C2_LIFT_MARGIN_NATS = 0.05
C2_MIN_LIFT_SEEDS = 2  # of 3

# C1(a) readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# C1(b) readiness: GAP-A consumed-summary divergence (649 statistic + 643a ceiling).
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# C1(c) readiness: ARM_ON rule field minted distinct rules AND -- the 654a fix --
# fired a non-zero differentiated rule_state on a meaningful fraction of MATURED P2
# ticks. The 654 cold-start kept crf_frac_active ~0.12; with crf_persist the pool
# persists across episodes so the floor is reachable.
CRF_MIN_MINTED = 2              # distinct rules created over the run
CRF_N_ACTIVE_FLOOR = 1          # >= 1 active rule => non-zero rule_state this tick
CRF_FRAC_ACTIVE_FLOOR = 0.30    # fraction of P2 ticks the field fired a rule_state
CRF_DIST_FLOOR = 1e-3           # reported diagnostic (pinned-distinct separability)

# C1(d) PROPAGATION non-vacuity: ARM_ON mean lateral_pfc bias must differ from
# ARM_OFF (the 654 seed-42 byte-identical washout had a ~4.4e-5 difference).
PROP_NONVAC_FLOOR = 1e-3

# Only classes committed to at least this many P2 ticks feed the unweighted mean
# within-class entropy (secondary negative control).
MIN_TICKS_PER_CLASS = 5

MIN_SEEDS_FOR_PASS = 2  # of 3

# C1(e) MECH-448 DEMOTION non-vacuity: the matched f_eligibility-demotion conversion
# constant must be LIVE. The f_demotion envelope is active on a meaningful fraction of
# P2 ticks AND it ACTUALLY excludes on the divergent e2_world_forward pool. An all-admit
# envelope (excluded_count == 0; a flat-F pool) means the demotion did nothing -> the run
# is vacuous -> substrate_not_ready_requeue, NEVER a false weakens. Mirrors V3-EXQ-689d
# READINESS(c) (the MECH-448 PASS this 654h generalises).
DEMOTION_ACTIVE_FRAC_FLOOR = 0.8   # fraction of P2 ticks the f_demotion envelope is active
EXCLUDED_COUNT_FLOOR = 0.0         # mean f_eligibility_excluded_count strictly above this

# C1(f) MECH-449 ACTIVE NO-GO non-vacuity: the matched Go/No-Go constitution constant must be
# LIVE (go_nogo_constitution_active) on a meaningful fraction of P2 ticks AND it must ACTUALLY
# SUPPRESS -- remove candidates from the F-eligible set (mean of go_nogo_n_safety_nogo +
# go_nogo_n_soft_applied > floor; the soft path is the MECH-260 perseveration No-Go fed by the
# dACC recency-share vector with use_dacc=True). An inert No-Go (nothing suppressed, the
# perseveration axis never crosses gng_perseveration_floor) means the active opponency leg did
# nothing -> the run is vacuous -> substrate_not_ready_requeue, NEVER a false weakens. Mirrors
# the V3-EXQ-689g go_nogo-actually-suppresses (excluded_count>0) non-degeneracy gate.
NOGO_ACTIVE_FRAC_FLOOR = 0.8       # fraction of P2 ticks the Go/No-Go constitution is active
NOGO_SUPPRESSED_FLOOR = 0.0        # mean per-tick (safety + soft) No-Go suppressions strictly above this

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 200         # 654c ENLARGED maturation (was 150): matches the 666c maturation budget; field matures + maintains
P1_BIAS_TRAIN_EPISODES = 90      # frozen-encoder bias-head REINFORCE (GAP-D); field keeps maturing + maintaining
P2_MEASUREMENT_EPISODES = 60     # all frozen; behavioural measurement
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 30

# Matched-stack lever constants (identical on BOTH arms).
#
# 654h: arm the MECH-448 (ARC-107) RANK-PRESERVING F->ELIGIBILITY DEMOTION as the
# conversion constant (below). The std-basis authority + channel routing build the
# modulatory accumulator (_modulatory_accum); the demotion lever then constructs the
# eligible set (graded DN envelope, F removed from the final argmin) and the modulatory
# channel arbitrates WITHIN that set. The retained 569i top_k keys (kept for lineage)
# are OVERRIDDEN by the f_demotion mode in e3_selector.py, so the top-k count is no
# longer the conversion -- the graded DN envelope is. 654g (the predecessor) used the
# 569i top-k as the conversion and FAILED C2 under the F-dominance ceiling; 654h tests
# the MECH-448 demotion lever that LIFTED that ceiling on GAP-A (V3-EXQ-689d PASS).
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 2.0               # 689d-matched (authority builds _modulatory_accum)
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"  # 689d-matched std basis
USE_MODULATORY_CHANNEL_ROUTING = True
MODULATORY_CHANNEL_ROUTE_SOURCE = "cand_world_summary"
MODULATORY_CHANNEL_ROUTE_WEIGHT = 1.0          # 689d-matched routed-vs-legacy accumulator proportion
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6        # substrate numerical active/inactive floor
# 569i TOP-K SHORTLIST keys -- RETAINED FOR LINEAGE ONLY; the MECH-448 f_demotion mode
# (below) takes precedence in e3_selector.py and constructs the eligible set instead.
USE_MODULATORY_SHORTLIST_THEN_MODULATE = True
MODULATORY_SHORTLIST_MODE = "top_k"            # 569i lineage (KEPT; f_demotion overrides it)
MODULATORY_SHORTLIST_K = 3                      # 569i lineage (KEPT; f_demotion overrides it)
# MECH-448 (ARC-107) RANK-PRESERVING F->ELIGIBILITY DEMOTION -- the 654h conversion
# constant (matched on BOTH arms). Per e3_selector.py the "f_demotion" mode TAKES
# PRECEDENCE over the retained top_k shortlist keys above: the eligible set becomes the
# graded divisive-normalisation envelope (absolute share floor), F is REMOVED from the
# final argmin, and the within-eligible _modulatory_accum arbitration (lateral_pfc CRF
# rule-bias + routed cand_world_summary) is identical. = the V3-EXQ-689d ARM_ON config
# (the MECH-448 PASS this run generalises off the GAP-A foraging substrate).
USE_F_ELIGIBILITY_DEMOTION = True
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30            # legacy absolute DN-share floor (ignored under the adaptive floor below; kept for the bit-identical fallback path)
F_ELIGIBILITY_DN_SIGMA = 0.0                   # 689d / substrate default: DN semi-saturation (>0 narrows)
# MECH-448 CHANNEL-ADAPTIVE (mean-relative) envelope floor -- the built replacement for 654i's
# per-(arm,seed) _calibrate_envelope_floor manual recalibration dance (the 485i/485j hand-floor).
# floor = f_eligibility_adaptive_mean_factor * elig.mean(): a candidate is eligible iff its share
# of the competing merit exceeds the field's OWN mean share. SCALE-INVARIANT -> auto-calibrates to
# the arc_062 bank's spread F distribution (V3-EXQ-689e PASS: excludes a non-empty tail where the
# 689d-tuned absolute 0.30 floor no-op'd in 654h). Still a threshold on elig (monotone in merit) so
# the eligible set stays an F-RANK PREFIX (rank-preserving). For mean_factor >= 1.0 on any
# non-uniform field at least one candidate is below the mean share -> excludes by construction; the
# C1e (excluded_count>0) readiness gate still guards a flat-F vacuous self-route. Matched on BOTH arms.
USE_F_ELIGIBILITY_ADAPTIVE_FLOOR = True
F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR = 1.0       # "above-average merit share" eligibility threshold multiple (689e default)
# MECH-449 (ARC-107) GO/NO-GO ELIGIBILITY CONSTITUTION -- the active No-Go OPPONENCY leg, the 654j
# conversion constant (matched on BOTH arms). use_dacc=True populates the dACC recency-share
# `suppression` vector that the default loop wires into the gate's `perseveration` axis (MECH-260
# reuse = the autopsy's named "active No-Go withdrawal" dependency); a candidate perseverated above
# gng_perseveration_floor is DROPPED from the F-eligible set BEFORE the within-eligible
# _modulatory_accum arbitration. The selection-face PASS V3-EXQ-689g validated this gate end-to-end.
USE_GO_NOGO_CONSTITUTION = True
USE_DACC = True                                # feeds the MECH-260 perseveration No-Go axis (ecological, not synthetic injection)
GNG_PERSEVERATION_FLOOR = 0.5                  # No-Go a candidate whose first-action class recency-share >= floor (substrate default)
GNG_SAFETY_FLOOR = 0.5                         # substrate default (no safety signal supplied here; perseveration is the live axis)
GNG_PROTECT_MIN_ELIGIBLE = 1                   # fail-open: never empty the eligible set on soft No-Go (substrate default)
MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

# SD-056 online e2 training (mirror V3-EXQ-649 harness).
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

# P1 bias-head REINFORCE training (mirror V3-EXQ-598b).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9


# IDENTICAL env to V3-EXQ-654 / 614e (SD-054 reef + hazard_food_attraction
# + bipartite layout) -- the behavioural falsifier substrate.
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


ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_OFF",
        "label": "rule_creator_absent_legacy_collapsed_rule_state",
        "use_candidate_rule_field": False,
    },
    {
        "arm_id": "ARM_ON",
        "label": "rule_creator_present_matured_persisted_crf_source",
        "use_candidate_rule_field": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


# NOTE (654j): 654i's _calibrate_envelope_floor per-(arm,seed) manual recalibration is REMOVED.
# The MECH-448 channel-adaptive (mean-relative) floor (use_f_eligibility_adaptive_floor=True) is
# the built replacement -- it auto-calibrates the eligibility envelope per channel inside
# e3_selector._f_eligibility_envelope, so the lagged per-tick floor rewrite is no longer needed.


def _make_agent(env: CausalGridWorldV2, use_candidate_rule_field: bool) -> REEAgent:
    """Matched-stack agent; the ONLY varied flag is use_candidate_rule_field.

    Both arms enable use_lateral_pfc_analog + use_gated_policy with the bias head
    UN-ZEROED (lateral_pfc_train_rule_bias_head=True) so the rule_state reaches a
    non-zero per-candidate E3 score-bias AND the head can be trained in P1.
    crf_persist_rules_across_episode_reset=True (the 654a fix; inert in ARM_OFF
    where no field is built) so the ARM_ON rule pool matures across episodes.
    candidate_summary_source = e2_world_forward on BOTH arms (GAP-A; e2 trained
    online in P0).
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
        # --- Matched stack (identical on both arms) ---
        # Layer A: SP-CEM (candidate-pool first-action-class diversity).
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # GAP-A (V3-EXQ-649): shared per-candidate signal from e2.world_forward.
        candidate_summary_source="e2_world_forward",
        # modulatory-bias-selection-authority (V3-EXQ-643a float32 fix) ARMED with
        # the 569i-VALIDATED TOP-K SHORTLIST conversion: std-basis authority + channel
        # routing build _modulatory_accum; the top-k shortlist (below) arbitrates
        # within F's small rotating near-tie set (and OVERRIDES the additive selection).
        use_modulatory_selection_authority=USE_MODULATORY_SELECTION_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_channel_routing=USE_MODULATORY_CHANNEL_ROUTING,
        modulatory_channel_route_source=MODULATORY_CHANNEL_ROUTE_SOURCE,
        modulatory_channel_route_weight=MODULATORY_CHANNEL_ROUTE_WEIGHT,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        # 569i TOP-K SHORTLIST keys retained for lineage; the MECH-448 f_demotion mode
        # (below) TAKES PRECEDENCE and constructs the eligible set instead (e3_selector.py).
        use_modulatory_shortlist_then_modulate=USE_MODULATORY_SHORTLIST_THEN_MODULATE,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        # MECH-448 (ARC-107) RANK-PRESERVING F->ELIGIBILITY DEMOTION -- the 654h
        # conversion constant, matched on BOTH arms. f_demotion mode overrides top_k:
        # the eligible set is the graded DN-share-floor envelope (F removed from the
        # final argmin), the within-eligible _modulatory_accum arbitration is identical.
        # = the V3-EXQ-689d ARM_ON config (the MECH-448 PASS this generalises).
        use_f_eligibility_demotion=USE_F_ELIGIBILITY_DEMOTION,
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        # MECH-448 channel-adaptive (mean-relative) floor (the 654j built replacement for the
        # 654i manual per-(arm,seed) envelope-floor calibration; V3-EXQ-689e PASS on arc_062).
        # Matched on BOTH arms.
        use_f_eligibility_adaptive_floor=USE_F_ELIGIBILITY_ADAPTIVE_FLOOR,
        f_eligibility_adaptive_mean_factor=F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
        # MECH-449 (ARC-107) GO/NO-GO ELIGIBILITY CONSTITUTION -- the active No-Go opponency leg,
        # the 654j conversion constant (matched on BOTH arms). use_dacc=True populates the dACC
        # recency-share `suppression` vector that the default select loop wires into the gate's
        # `perseveration` axis (MECH-260 reuse); a perseverated candidate is dropped from the
        # F-eligible set BEFORE the within-eligible _modulatory_accum arbitration. Validated by
        # V3-EXQ-689g (selection-face PASS).
        use_dacc=USE_DACC,
        use_go_nogo_constitution=USE_GO_NOGO_CONSTITUTION,
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
        # SD-033a LateralPFCAnalog with the bias head UN-ZEROED + trainable (GAP-D)
        # so the head can be trained in P1 and maps rule_state -> a non-zero
        # per-candidate bias.
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        # SD-056 (e2 action-conditional divergence; trained online in P0).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
        # --- CRF maturity + maintenance levers (inert in ARM_OFF -- no field built) ---
        # 654a fix: cross-episode pool carry-over.
        crf_persist_rules_across_episode_reset=True,
        # 654b mature-pool amend: deadlock-free conflict gate + slow decay +
        # mint-youth protection + e2-world-forward CRF context (differentiation source).
        crf_mature_pool_dynamics=True,
        crf_context_from_e2_world_forward=True,
        # 654c: the 666c-validated availability-maintenance trace (activity-silent
        # hold of a differentiated rule across sparse-matching ticks). This is what
        # 654b lacked -- it is the lever V3-EXQ-666c PASS validated.
        crf_availability_maintenance=True,
        crf_maintenance_floor=CRF_MAINTENANCE_FLOOR,
        crf_maintenance_decay=CRF_MAINTENANCE_DECAY,
        # 654e: the CRF-gate calibration amend -- the actual GAP-B fix (654d proved
        # GAP-A de-collapse alone leaves crf_frac_active=0.0; the gate locks out the
        # whole maintained pool). Sharpen the gate match cutoff (FAULT 1) + cap theta
        # (FAULT 2a) + couple maintained availability to theta (FAULT 2b) so the
        # maintained differentiated pool CLEARS the conflict gate. Inert in ARM_OFF.
        crf_mature_context_match_threshold=CRF_MATURE_CONTEXT_MATCH_THRESHOLD,
        crf_tolerance_conflict_cap=CRF_TOLERANCE_CONFLICT_CAP,
        crf_maintenance_couple_to_theta=CRF_MAINTENANCE_COUPLE_TO_THETA,
        # --- The ONLY swept variable ---
        use_candidate_rule_field=bool(use_candidate_rule_field),
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-649)
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
    e2.world_forward source; both arms)."""
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
# P1 bias-head REINFORCE training (mirror V3-EXQ-598b _lpfc_reinforce_loss)
# ---------------------------------------------------------------------------


def _lpfc_reinforce_loss(
    agent: REEAgent,
    outcome_buf: List[Tuple[torch.Tensor, int, float]],
    baseline: float,
    device,
) -> torch.Tensor:
    """REINFORCE on the SD-033a bias head over stored (candidate_features, sel, return).

    Re-runs compute_bias (differentiable w.r.t. rule_bias_head weights) with the
    CURRENT rule_state on stored candidate summaries, REINFORCE-weighted by the
    episode-return advantage. Mirrors v3_exq_598b._lpfc_reinforce_loss.
    """
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
    """Within-ARM_ON isolation: mean |bias(field rule_state) - bias(zeroed rule_state)|.

    Quantifies how much the field's rule_state actually shapes the per-candidate
    bias (independent of the trained head). Returned as a supporting diagnostic
    alongside the load-bearing cross-arm C1d precondition. Best-effort.
    """
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
        # Restore best-effort if anything went wrong mid-swap.
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
    agent = _make_agent(env, bool(arm["use_candidate_rule_field"]))
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

    # P1 REINFORCE state.
    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    # PRIMARY DV + readiness accumulators (P2).
    committed_class_counts: Dict[int, int] = {}
    selected_class_counts: Dict[int, int] = {}
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0

    # SECONDARY negative control (within-class-representative; P2).
    per_class_rep_sigs: Dict[int, Counter] = {}
    all_rep_sigs: Counter = Counter()

    # ARM_ON differentiation + bias diagnostics (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_matched_per_tick: List[int] = []  # 654d: the 654c-autopsy discriminator
    crf_max_pairwise_rule_dist_max = 0.0
    crf_n_minted_total_last = 0
    lateral_pfc_bias_abs_vals: List[float] = []
    prop_counterfactual_deltas: List[float] = []

    # MECH-448 f_eligibility-demotion non-vacuity readouts (the matched conversion
    # constant; read LIVE from e3.last_score_diagnostics at the P2 select tick). Fires
    # on BOTH arms (the lever is matched). excluded_count == 0 == all-admit == vacuous.
    demotion_active_ticks = 0
    demotion_envelope_sizes: List[float] = []
    demotion_excluded_counts: List[float] = []
    demotion_winner_neq_f_argmin_ticks = 0
    demotion_rank_preserving_active_ticks = 0

    # MECH-449 Go/No-Go-constitution non-vacuity readouts (the matched active-No-Go
    # opponency constant; read LIVE from e3.last_score_diagnostics at the P2 select tick).
    # Fires on BOTH arms (matched). go_nogo suppressed == safety + soft No-Go applied; 0
    # suppressions on every tick == inert No-Go == vacuous.
    nogo_active_ticks = 0
    nogo_suppressed_per_tick: List[int] = []
    nogo_envelope_sizes: List[float] = []

    for ep in range(total_train_eps):
        is_p1 = (p1_start <= ep < p2_start)
        is_p2 = (ep >= p2_start)
        phase_label = "P2" if is_p2 else ("P1" if is_p1 else "P0")

        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

        # P1 per-episode REINFORCE buffers.
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

            # SD-056 transition capture (z0, a) this tick -> z1 next tick.
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

            # Capture candidate summaries BEFORE select_action for P1 REINFORCE
            # snap (the same e2_world_forward source compute_bias consumes).
            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            action = agent.select_action(candidates, ticks)
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

            # 654j: NO manual envelope-floor calibration. The MECH-448 channel-adaptive
            # (mean-relative) floor (use_f_eligibility_adaptive_floor=True) auto-calibrates the
            # eligibility envelope per channel inside e3_selector._f_eligibility_envelope, so the
            # 654i lagged per-tick floor rewrite is removed (replaced by the built lever).

            # P1: record (candidate_features, selected-candidate-index) snap.
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
                committed_class_counts[committed_class] = (
                    committed_class_counts.get(committed_class, 0) + 1
                )
                if len(pre_e3_classes) >= 2:
                    n_p2_pre_ge2 += 1

                if candidates and len(candidates) >= 2:
                    consumed = _consumed_summaries(agent, candidates)
                    if consumed is not None and torch.isfinite(consumed).all():
                        d = _mean_pairwise_l2(consumed)
                        if math.isfinite(d):
                            consumed_dists.append(d)
                            consumed_dist_max = max(consumed_dist_max, d)

                # SECONDARY negative control: within-class representative.
                sel_traj = getattr(agent.e3, "_last_selected_trajectory", None)
                if sel_traj is not None:
                    sel_class = _traj_first_action_class(sel_traj)
                    rep_sig = _traj_rep_signature(sel_traj)
                    selected_class_counts[sel_class] = (
                        selected_class_counts.get(sel_class, 0) + 1
                    )
                    per_class_rep_sigs.setdefault(sel_class, Counter())[rep_sig] += 1
                    all_rep_sigs[rep_sig] += 1

                # lateral_pfc bias magnitude (manipulation-reached-E3 context).
                lpfc = getattr(agent, "lateral_pfc", None)
                if lpfc is not None:
                    lb_mean = getattr(lpfc, "_last_bias_abs_mean", None)
                    if isinstance(lb_mean, (int, float)):
                        lateral_pfc_bias_abs_vals.append(float(lb_mean))

                # MECH-448 f_eligibility-demotion diagnostics (read LIVE at the select
                # tick; the matched conversion constant fires on BOTH arms). C1e checks
                # the envelope is active AND actually excludes on the divergent pool.
                diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
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

                # MECH-449 Go/No-Go-constitution diagnostics (read LIVE at the select tick;
                # the matched active-No-Go opponency constant fires on BOTH arms). C1f checks
                # the constitution is active AND actually SUPPRESSES (safety + soft No-Go
                # removed candidates from the F-eligible set). Same diag dict as the demotion.
                if bool(diag.get("go_nogo_constitution_active", False)):
                    nogo_active_ticks += 1
                    n_safety = int(diag.get("go_nogo_n_safety_nogo", 0) or 0)
                    n_soft = int(diag.get("go_nogo_n_soft_applied", 0) or 0)
                    nogo_suppressed_per_tick.append(n_safety + n_soft)
                    gng_env = float(diag.get("go_nogo_envelope_size", -1))
                    if math.isfinite(gng_env) and gng_env >= 0:
                        nogo_envelope_sizes.append(gng_env)

                # CandidateRuleField differentiation (ARM_ON).
                crf = getattr(agent, "candidate_rule_field", None)
                if crf is not None:
                    st = crf.get_state()
                    n_active = int(st.get("crf_n_active_last", 0))
                    crf_n_active_per_tick.append(n_active)
                    # 654d: record n_matched -- distinguishes "never matched" from
                    # "matched but gated out" if C1c fails despite de-collapse.
                    crf_n_matched_per_tick.append(
                        int(st.get("crf_n_matched_last", 0))
                    )
                    crf_max_pairwise_rule_dist_max = max(
                        crf_max_pairwise_rule_dist_max,
                        float(st.get("crf_max_pairwise_rule_dist", 0.0)),
                    )
                    crf_n_minted_total_last = int(st.get("crf_n_minted_total", 0))
                    # Within-ARM_ON propagation counterfactual on firing ticks only.
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

            # Capture (z0, a) for the next-tick SD-056 transition.
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

        # P1 end-of-episode: REINFORCE update on the SD-033a bias head.
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
    committed_class_entropy = _entropy_from_int_counts(committed_class_counts)
    selected_class_entropy = _entropy_from_int_counts(selected_class_counts)

    frac_pre_ge2 = float(n_p2_pre_ge2 / n_p2_ticks) if n_p2_ticks > 0 else 0.0

    consumed_spread_mean = (
        float(sum(consumed_dists) / len(consumed_dists)) if consumed_dists else 0.0
    )

    # SECONDARY within-class-representative entropy (negative control).
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

    # ARM_ON differentiation readiness.
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

    # 654d discriminator: n_matched aggregates. High mean_n_matched with
    # frac_active=0 == matched-but-gated-out (route to the maintenance-theta amend);
    # mean_n_matched ~0 == never-matched (a different fault).
    if crf_n_matched_per_tick:
        mean_crf_n_matched = float(
            sum(crf_n_matched_per_tick) / len(crf_n_matched_per_tick)
        )
        max_crf_n_matched = int(max(crf_n_matched_per_tick))
    else:
        mean_crf_n_matched = 0.0
        max_crf_n_matched = 0

    crf_present = bool(arm["use_candidate_rule_field"])
    crf_differentiated = bool(
        crf_present
        and crf_n_minted_total_last >= CRF_MIN_MINTED
        and frac_crf_active_ge_floor >= CRF_FRAC_ACTIVE_FLOOR
    )

    mean_lateral_pfc_bias_abs = (
        float(sum(lateral_pfc_bias_abs_vals) / len(lateral_pfc_bias_abs_vals))
        if lateral_pfc_bias_abs_vals else 0.0
    )
    mean_prop_counterfactual_delta = (
        float(sum(prop_counterfactual_deltas) / len(prop_counterfactual_deltas))
        if prop_counterfactual_deltas else 0.0
    )

    # MECH-448 f_eligibility-demotion aggregates (C1e non-vacuity).
    demotion_active_frac = (
        float(demotion_active_ticks) / float(n_p2_ticks) if n_p2_ticks > 0 else 0.0
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
    # C1e: the matched f_demotion envelope is ACTIVE on a meaningful fraction of P2
    # ticks AND actually EXCLUDES on the divergent pool (excluded_count > floor). An
    # all-admit envelope (flat-F) is vacuous -> substrate_not_ready_requeue.
    seed_demotion_non_vacuous = bool(
        demotion_active_frac >= DEMOTION_ACTIVE_FRAC_FLOOR
        and demotion_excluded_count_mean > EXCLUDED_COUNT_FLOOR
    )

    # MECH-449 Go/No-Go-constitution aggregates (C1f non-vacuity).
    nogo_active_frac = (
        float(nogo_active_ticks) / float(n_p2_ticks) if n_p2_ticks > 0 else 0.0
    )
    nogo_suppressed_mean = (
        float(sum(nogo_suppressed_per_tick) / len(nogo_suppressed_per_tick))
        if nogo_suppressed_per_tick else 0.0
    )
    nogo_envelope_size_mean = (
        float(sum(nogo_envelope_sizes) / len(nogo_envelope_sizes))
        if nogo_envelope_sizes else 0.0
    )
    # C1f: the matched Go/No-Go constitution is ACTIVE on a meaningful fraction of P2
    # ticks AND it actually SUPPRESSES (mean per-tick safety+soft No-Go removals > floor).
    # An inert No-Go (perseveration axis never crosses gng_perseveration_floor; nothing
    # removed) is vacuous -> substrate_not_ready_requeue, NEVER a false weakens.
    seed_nogo_non_vacuous = bool(
        nogo_active_frac >= NOGO_ACTIVE_FRAC_FLOOR
        and nogo_suppressed_mean > NOGO_SUPPRESSED_FLOOR
    )

    # Per-seed readiness flags.
    seed_class_axis_exercisable = bool(frac_pre_ge2 > FRAC_PRE_GE2_FLOOR)
    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "use_candidate_rule_field": crf_present,
        "crf_persist_rules_across_episode_reset": True,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
        # ----- PRIMARY DV -----
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        # ----- C1 readiness -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "class_axis_exercisable": seed_class_axis_exercisable,
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        "crf_present": crf_present,
        "crf_mean_n_active": round(mean_crf_n_active, 6),
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_max_pairwise_rule_dist": round(crf_max_pairwise_rule_dist_max, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "crf_differentiated": crf_differentiated,
        # 654d discriminator (matched-but-gated-out vs never-matched).
        "crf_mean_n_matched": round(mean_crf_n_matched, 6),
        "crf_max_n_matched": int(max_crf_n_matched),
        # ----- C1d propagation non-vacuity -----
        "mean_lateral_pfc_bias_abs": round(mean_lateral_pfc_bias_abs, 8),
        "mean_prop_counterfactual_delta": round(mean_prop_counterfactual_delta, 8),
        # ----- C1e MECH-448 demotion non-vacuity (matched conversion constant) -----
        "f_eligibility_demotion_active_ticks": int(demotion_active_ticks),
        "f_eligibility_demotion_active_frac": round(demotion_active_frac, 6),
        "f_eligibility_excluded_count_mean": round(demotion_excluded_count_mean, 6),
        "f_eligibility_envelope_size_mean": round(demotion_envelope_size_mean, 6),
        "f_eligibility_winner_neq_f_argmin_ticks": int(demotion_winner_neq_f_argmin_ticks),
        "f_eligibility_rank_preserving_frac": round(demotion_rank_preserving_frac, 6),
        "demotion_non_vacuous": seed_demotion_non_vacuous,
        # ----- C1f MECH-449 active No-Go non-vacuity (matched opponency constant) -----
        "go_nogo_active_ticks": int(nogo_active_ticks),
        "go_nogo_active_frac": round(nogo_active_frac, 6),
        "go_nogo_suppressed_per_tick_mean": round(nogo_suppressed_mean, 6),
        "go_nogo_envelope_size_mean": round(nogo_envelope_size_mean, 6),
        "nogo_non_vacuous": seed_nogo_non_vacuous,
        # ----- C3 SECONDARY negative control (NOT load-bearing) -----
        "mean_within_class_rep_entropy_nats": round(mean_within_class_rep_entropy, 6),
        "n_distinct_rep_signatures_total": int(len(all_rep_sigs)),
        "selected_class_entropy_nats": round(selected_class_entropy, 6),
    }


def _arm_rows(arm_results: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [
        r for r in arm_results
        if r["arm_id"] == arm_id and r["error_note"] is None
    ]


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
                    "use_candidate_rule_field": bool(arm["use_candidate_rule_field"]),
                    "crf_persist_rules_across_episode_reset": True,
                    "crf_mature_pool_dynamics": True,
                    "crf_context_from_e2_world_forward": True,
                    "crf_availability_maintenance": True,
                    "crf_maintenance_floor": float(CRF_MAINTENANCE_FLOOR),
                    "crf_maintenance_decay": float(CRF_MAINTENANCE_DECAY),
                    "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
                    "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
                    "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
                    "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
                    "f_eligibility_envelope_floor": float(F_ELIGIBILITY_ENVELOPE_FLOOR),
                    "f_eligibility_dn_sigma": float(F_ELIGIBILITY_DN_SIGMA),
                    "use_f_eligibility_adaptive_floor": bool(USE_F_ELIGIBILITY_ADAPTIVE_FLOOR),
                    "f_eligibility_adaptive_mean_factor": float(F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR),
                    "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
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
    on_rows = _arm_rows(arm_results, "ARM_ON")

    # C1(a): committed-class axis exercisable on majority of seeds in BOTH arms.
    n_off_axis = sum(1 for r in off_rows if r["class_axis_exercisable"])
    n_on_axis = sum(1 for r in on_rows if r["class_axis_exercisable"])
    c1a_holds = bool(n_off_axis >= MIN_SEEDS_FOR_PASS and n_on_axis >= MIN_SEEDS_FOR_PASS)

    # C1(b): GAP-A divergence real on majority of seeds in BOTH arms.
    n_off_gapa = sum(1 for r in off_rows if r["gapa_divergence"])
    n_on_gapa = sum(1 for r in on_rows if r["gapa_divergence"])
    c1b_holds = bool(n_off_gapa >= MIN_SEEDS_FOR_PASS and n_on_gapa >= MIN_SEEDS_FOR_PASS)

    # C1(c): ARM_ON field minted distinct rules AND matured on majority of ARM_ON seeds.
    n_on_differentiated = sum(1 for r in on_rows if r["crf_differentiated"])
    c1c_holds = bool(n_on_differentiated >= MIN_SEEDS_FOR_PASS)

    # C1(d): propagation non-vacuity -- paired |bias_ON - bias_OFF| > floor.
    off_bias_by_seed = {int(r["seed"]): r["mean_lateral_pfc_bias_abs"] for r in off_rows}
    on_bias_by_seed = {int(r["seed"]): r["mean_lateral_pfc_bias_abs"] for r in on_rows}
    prop_diff_by_seed: Dict[int, float] = {}
    n_prop_nonvac_seeds = 0
    for seed in sorted(set(off_bias_by_seed) & set(on_bias_by_seed)):
        diff = abs(on_bias_by_seed[seed] - off_bias_by_seed[seed])
        prop_diff_by_seed[seed] = round(diff, 8)
        if diff > PROP_NONVAC_FLOOR:
            n_prop_nonvac_seeds += 1
    c1d_holds = bool(n_prop_nonvac_seeds >= MIN_SEEDS_FOR_PASS)

    # C1(e): MECH-448 DEMOTION non-vacuity -- the matched conversion constant is LIVE
    # (envelope active AND actually excluding) on a majority of seeds in BOTH arms. An
    # all-admit envelope (excluded_count==0; flat-F pool) is a vacuous self-route.
    n_off_demotion = sum(1 for r in off_rows if r["demotion_non_vacuous"])
    n_on_demotion = sum(1 for r in on_rows if r["demotion_non_vacuous"])
    c1e_holds = bool(
        n_off_demotion >= MIN_SEEDS_FOR_PASS and n_on_demotion >= MIN_SEEDS_FOR_PASS
    )

    # C1(f): MECH-449 ACTIVE NO-GO non-vacuity -- the matched opponency constant is LIVE
    # (constitution active AND actually SUPPRESSING candidates from the F-eligible set) on a
    # majority of seeds in BOTH arms. An inert No-Go (nothing suppressed) is a vacuous
    # self-route -> substrate_not_ready_requeue, NEVER a false weakens.
    n_off_nogo = sum(1 for r in off_rows if r["nogo_non_vacuous"])
    n_on_nogo = sum(1 for r in on_rows if r["nogo_non_vacuous"])
    c1f_holds = bool(
        n_off_nogo >= MIN_SEEDS_FOR_PASS and n_on_nogo >= MIN_SEEDS_FOR_PASS
    )

    c1_holds = bool(
        c1a_holds and c1b_holds and c1c_holds and c1d_holds and c1e_holds and c1f_holds
    )

    # C2 (PRIMARY): paired-by-seed committed-class entropy lift.
    off_by_seed = {int(r["seed"]): r["committed_class_entropy_nats"] for r in off_rows}
    on_by_seed = {int(r["seed"]): r["committed_class_entropy_nats"] for r in on_rows}
    paired_lifts: Dict[int, float] = {}
    n_lift_seeds = 0
    for seed in sorted(set(off_by_seed) & set(on_by_seed)):
        lift = on_by_seed[seed] - off_by_seed[seed]
        paired_lifts[seed] = round(lift, 6)
        if lift >= C2_LIFT_MARGIN_NATS:
            n_lift_seeds += 1
    c2_holds = bool(n_lift_seeds >= C2_MIN_LIFT_SEEDS)

    off_mean_dv = _mean([r["committed_class_entropy_nats"] for r in off_rows])
    on_mean_dv = _mean([r["committed_class_entropy_nats"] for r in on_rows])

    # within-ARM_ON propagation counterfactual (supporting diagnostic).
    on_prop_cf = [r["mean_prop_counterfactual_delta"] for r in on_rows]
    n_on_prop_cf_nonzero = sum(1 for d in on_prop_cf if d > PROP_NONVAC_FLOOR)

    # ----- Outcome map (THREE branches; NO weakens -- the constitution is the deep fix) -----
    if not c1_holds:
        # C1 fail: substrate not exercisable / not matured (C1c) / propagation vacuous
        # (C1d) / DEMOTION vacuous (C1e) / ACTIVE NO-GO vacuous (C1f: nothing suppressed)
        # / class axis or GAP-A divergence absent -> re-queue, NOT a falsification.
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif c2_holds:
        outcome = "PASS"
        direction = "supports"
        label = "PASS_C1_C2_active_nogo_converts_rule_creator_committed_class_diversity"
    else:
        # C1 holds (matured + maintained pool, propagation non-vacuous, the MECH-448
        # demotion lever LIVE, AND the MECH-449 active No-Go LIVE and actually suppressing)
        # but C2 fails. The conversion ceiling persists EVEN UNDER active Go/No-Go opponency
        # -> a GENUINE SIGNAL that the Go/No-Go opponency leg ALONE is insufficient for
        # behavioural conversion on the GAP-B composite. NOT a MECH-309/ARC-062 falsification.
        # non_contributory; ROUTE to the NEXT ARC-107 selector-constitution ledger component
        # (open components 2/4/5) / V4 directions. PRE-REGISTERED off-ramp (see docstring).
        outcome = "FAIL"
        direction = "non_contributory"
        label = "conversion_ceiling_persists_despite_active_nogo_route_next_arc107_ledger_component"

    evidence_direction_per_claim = {"MECH-309": direction, "ARC-062": direction}

    # ----- interpretation block (preconditions + non-degeneracy) -----
    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "committed_class_axis_exercisable_both_arms",
                "kind": "readiness",
                "description": (
                    "frac of P2 ticks with >= 2 candidate first-action classes "
                    "exceeds floor on a majority of seeds in BOTH arms. SAME-statistic "
                    "family as C2 (class multiplicity bounds class entropy)."
                ),
                "control": "SP-CEM multi-class candidate pool, both arms",
                "measured": float(
                    min([r["frac_pre_ge2"] for r in (off_rows + on_rows)] or [0.0])
                ),
                "threshold": float(FRAC_PRE_GE2_FLOOR),
                "met": bool(c1a_holds),
            },
            {
                "name": "gapa_consumed_summary_divergence_both_arms",
                "kind": "readiness",
                "description": (
                    "consumed cand_world_summaries (e2.world_forward) per-candidate "
                    "SPREAD clears the floor on a majority of seeds in BOTH arms -- "
                    "the class bias is non-degenerate. Same range statistic the 649 "
                    "GAP-A readiness asserts."
                ),
                "control": "ARM_OFF + ARM_ON: SD-056 e2 trained online in P0; candidate_summary_source=e2_world_forward",
                "measured": float(
                    min(
                        [r["consumed_summary_pairwise_dist_mean"] for r in (off_rows + on_rows)]
                        or [0.0]
                    )
                ),
                "threshold": float(CONSUMED_SPREAD_FLOOR),
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
                    max(
                        [r["consumed_summary_pairwise_dist_max"] for r in (off_rows + on_rows)]
                        or [0.0]
                    )
                ),
                "threshold": float(CONSUMED_MAGNITUDE_CEIL),
                "direction": "upper",
                "met": bool(
                    max(
                        [r["consumed_summary_pairwise_dist_max"] for r in (off_rows + on_rows)]
                        or [0.0]
                    ) < CONSUMED_MAGNITUDE_CEIL
                ),
            },
            {
                "name": "arm_on_rule_field_differentiated_and_matured",
                "kind": "readiness",
                "description": (
                    "ARM_ON CandidateRuleField minted >= CRF_MIN_MINTED distinct "
                    "rules AND fired a non-zero rule_state on >= CRF_FRAC_ACTIVE_FLOOR "
                    "of P2 ticks (the crf_persist-matured pool clears the 0.30 floor "
                    "the 654 per-episode cold-start could not), on a majority of "
                    "ARM_ON seeds. Below-floor => substrate not matured => "
                    "substrate_not_ready_requeue."
                ),
                "control": "ARM_ON crf frac-active (matured pool) + crf_n_minted_total",
                "measured": float(
                    max([r["crf_frac_active_ge_floor"] for r in on_rows] or [0.0])
                ),
                "threshold": float(CRF_FRAC_ACTIVE_FLOOR),
                "met": bool(c1c_holds),
            },
            {
                "name": "propagation_non_vacuity_arm_on_bias_differs_from_arm_off",
                "kind": "readiness",
                "description": (
                    "paired-by-seed |mean_lateral_pfc_bias_abs(ARM_ON) - "
                    "mean_lateral_pfc_bias_abs(ARM_OFF)| > floor on a majority of "
                    "seeds. The trained head + matured field produce a DIFFERENT bias "
                    "than the legacy collapsed source -- the 654 seed-42 byte-identical "
                    "(~4.4e-5) propagation washout did not recur. Below-floor => "
                    "vacuous propagation => substrate_not_ready_requeue (do NOT score "
                    "the falsifier). Supported by the within-ARM_ON rule_state "
                    "counterfactual delta (zeroing rule_state changes the bias)."
                ),
                "control": "paired ARM_ON vs ARM_OFF mean lateral_pfc bias on the matched stack",
                "measured": float(
                    max(list(prop_diff_by_seed.values()) or [0.0])
                ),
                "threshold": float(PROP_NONVAC_FLOOR),
                "met": bool(c1d_holds),
            },
            {
                "name": "mech448_demotion_lever_live_and_excluding_both_arms",
                "kind": "readiness",
                "description": (
                    "The MECH-448 f_eligibility-demotion conversion constant (matched "
                    "on BOTH arms) is ACTIVE on >= DEMOTION_ACTIVE_FRAC_FLOOR of P2 "
                    "ticks AND it ACTUALLY EXCLUDES on the divergent e2_world_forward "
                    "pool (mean f_eligibility_excluded_count > EXCLUDED_COUNT_FLOOR), on "
                    "a majority of seeds in BOTH arms. An all-admit envelope "
                    "(excluded_count==0; a flat-F pool) means the demotion did nothing "
                    "-> the conversion constant is vacuous => substrate_not_ready_requeue "
                    "(do NOT score the falsifier), NEVER a false weakens. Mirrors the "
                    "V3-EXQ-689d READINESS(c) non-degeneracy gate."
                ),
                "control": "f_demotion envelope active-frac + excluded_count on both arms (matched lever)",
                "measured": float(
                    min(
                        [r["f_eligibility_excluded_count_mean"] for r in (off_rows + on_rows)]
                        or [0.0]
                    )
                ),
                "threshold": float(EXCLUDED_COUNT_FLOOR),
                "met": bool(c1e_holds),
            },
            {
                "name": "mech449_active_nogo_live_and_suppressing_both_arms",
                "kind": "readiness",
                "description": (
                    "The MECH-449 Go/No-Go eligibility constitution (matched on BOTH arms) "
                    "is ACTIVE on >= NOGO_ACTIVE_FRAC_FLOOR of P2 ticks AND it ACTUALLY "
                    "SUPPRESSES -- removes candidates from the F-eligible set (mean per-tick "
                    "go_nogo_n_safety_nogo + go_nogo_n_soft_applied > NOGO_SUPPRESSED_FLOOR; "
                    "the soft path is the MECH-260 perseveration No-Go fed by the dACC "
                    "recency-share vector with use_dacc=True), on a majority of seeds in BOTH "
                    "arms. An inert No-Go (nothing suppressed; the perseveration axis never "
                    "crosses gng_perseveration_floor) means the active opponency leg did "
                    "nothing -> the conversion constant is vacuous => "
                    "substrate_not_ready_requeue (do NOT score the falsifier), NEVER a false "
                    "weakens. Mirrors the V3-EXQ-689g go_nogo-actually-suppresses "
                    "non-degeneracy gate."
                ),
                "control": "Go/No-Go constitution active-frac + suppressed-count on both arms (matched lever)",
                "measured": float(
                    min(
                        [r["go_nogo_suppressed_per_tick_mean"] for r in (off_rows + on_rows)]
                        or [0.0]
                    )
                ),
                "threshold": float(NOGO_SUPPRESSED_FLOOR),
                "met": bool(c1f_holds),
            },
        ],
        "criteria": [
            {
                "name": "C2_committed_class_entropy_lift",
                "load_bearing": True,
                "passed": bool(c2_holds),
            },
        ],
        "criteria_non_degenerate": {
            "C1a_class_axis_exercisable": bool(c1a_holds),
            "C1b_gapa_divergence": bool(c1b_holds),
            "C1c_arm_on_differentiated_matured": bool(c1c_holds),
            "C1d_propagation_non_vacuity": bool(c1d_holds),
            "C1d_within_arm_on_rule_state_counterfactual_nonzero": bool(
                n_on_prop_cf_nonzero >= MIN_SEEDS_FOR_PASS
            ),
            "C1e_mech448_demotion_live_and_excluding": bool(c1e_holds),
            "C1f_mech449_active_nogo_live_and_suppressing": bool(c1f_holds),
            "C2_paired_lift": bool(c2_holds),
        },
    }

    total_seeds = len(ARMS) * len(seeds)
    total_completed = len(off_rows) + len(on_rows)

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "interpretation_label": label,
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
            "c2_lift_margin_nats": float(C2_LIFT_MARGIN_NATS),
            "c2_min_lift_seeds": int(C2_MIN_LIFT_SEEDS),
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
            "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
            "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
            "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
            "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
            "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
            "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
            "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
            "f_eligibility_envelope_floor": float(F_ELIGIBILITY_ENVELOPE_FLOOR),
            "f_eligibility_dn_sigma": float(F_ELIGIBILITY_DN_SIGMA),
            "demotion_active_frac_floor": float(DEMOTION_ACTIVE_FRAC_FLOOR),
            "excluded_count_floor": float(EXCLUDED_COUNT_FLOOR),
            "use_f_eligibility_adaptive_floor": bool(USE_F_ELIGIBILITY_ADAPTIVE_FLOOR),
            "f_eligibility_adaptive_mean_factor": float(F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR),
            "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
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
            "C1_substrate_exercisable_and_manipulation_live": c1_holds,
            "C1a_class_axis_exercisable_both_arms": c1a_holds,
            "C1a_n_off_axis": int(n_off_axis),
            "C1a_n_on_axis": int(n_on_axis),
            "C1b_gapa_divergence_both_arms": c1b_holds,
            "C1b_n_off_gapa": int(n_off_gapa),
            "C1b_n_on_gapa": int(n_on_gapa),
            "C1c_arm_on_rule_field_differentiated_matured": c1c_holds,
            "C1c_n_on_differentiated": int(n_on_differentiated),
            "C1d_propagation_non_vacuity": c1d_holds,
            "C1d_n_prop_nonvac_seeds": int(n_prop_nonvac_seeds),
            "C1d_prop_diff_by_seed": prop_diff_by_seed,
            "C1d_n_on_within_arm_counterfactual_nonzero": int(n_on_prop_cf_nonzero),
            "C1e_mech448_demotion_live_both_arms": c1e_holds,
            "C1e_n_off_demotion_non_vacuous": int(n_off_demotion),
            "C1e_n_on_demotion_non_vacuous": int(n_on_demotion),
            "C1e_off_excluded_count_mean": round(
                _mean([r["f_eligibility_excluded_count_mean"] for r in off_rows]), 6
            ),
            "C1e_on_excluded_count_mean": round(
                _mean([r["f_eligibility_excluded_count_mean"] for r in on_rows]), 6
            ),
            "C1e_on_demotion_active_frac_mean": round(
                _mean([r["f_eligibility_demotion_active_frac"] for r in on_rows]), 6
            ),
            "C1f_mech449_active_nogo_live_both_arms": c1f_holds,
            "C1f_n_off_nogo_non_vacuous": int(n_off_nogo),
            "C1f_n_on_nogo_non_vacuous": int(n_on_nogo),
            "C1f_off_nogo_suppressed_per_tick_mean": round(
                _mean([r["go_nogo_suppressed_per_tick_mean"] for r in off_rows]), 6
            ),
            "C1f_on_nogo_suppressed_per_tick_mean": round(
                _mean([r["go_nogo_suppressed_per_tick_mean"] for r in on_rows]), 6
            ),
            "C1f_on_nogo_active_frac_mean": round(
                _mean([r["go_nogo_active_frac"] for r in on_rows]), 6
            ),
            "C2_committed_class_lift": c2_holds,
            "C2_n_lift_seeds": int(n_lift_seeds),
            "C2_paired_lifts_by_seed": paired_lifts,
            "C2_off_mean_committed_class_entropy": round(off_mean_dv, 6),
            "C2_on_mean_committed_class_entropy": round(on_mean_dv, 6),
        },
        "secondary_negative_control_not_load_bearing": {
            "note": (
                "Within-class-representative entropy is a NEGATIVE CONTROL: the rule "
                "bias is class-keyed (per-candidate summary first-action-keyed; "
                "compute_bias broadcasts one rule_state across K), so it cannot move "
                "within-class selection -> ARM_ON ~ ARM_OFF is EXPECTED here, "
                "confirming the rule-creator's signal lives in the committed-class "
                "axis (the load-bearing C2 DV)."
            ),
            "arm_off_within_class_rep_entropy_mean": round(
                _mean([r["mean_within_class_rep_entropy_nats"] for r in off_rows]), 6
            ),
            "arm_on_within_class_rep_entropy_mean": round(
                _mean([r["mean_within_class_rep_entropy_nats"] for r in on_rows]), 6
            ),
            "arm_off_selected_class_entropy_mean": round(
                _mean([r["selected_class_entropy_nats"] for r in off_rows]), 6
            ),
            "arm_on_selected_class_entropy_mean": round(
                _mean([r["selected_class_entropy_nats"] for r in on_rows]), 6
            ),
            "arm_off_lateral_pfc_bias_abs_mean": round(
                _mean([r["mean_lateral_pfc_bias_abs"] for r in off_rows]), 8
            ),
            "arm_on_lateral_pfc_bias_abs_mean": round(
                _mean([r["mean_lateral_pfc_bias_abs"] for r in on_rows]), 8
            ),
            "arm_on_within_arm_prop_counterfactual_delta_mean": round(
                _mean(on_prop_cf), 8
            ),
        },
        "interpretation_grid": {
            "PASS_C1_C2": (
                "The non-Bayesian rule-creator's DIFFERENTIATED + MATURED rule_state "
                "propagates (via the trained bias head) to greater committed-class "
                "diversity than the collapsed legacy baseline -- AND it CONVERTS under "
                "the MECH-449 active Go/No-Go opponency leg (on top of the matched MECH-448 "
                "demotion lever). The active No-Go withdrawal (the MECH-260 perseveration "
                "axis dropping the F-eligible-but-perseverative winner) CONVERTS the "
                "differentiated bias into committed-class diversity where demotion-alone "
                "(654i) could not -> closes behavioral_diversity_isolation:GAP-I. supports "
                "MECH-309 + ARC-062. Route to /governance: MECH-309 / ARC-062 supports "
                "evidence; consider GAP-B + GAP-I closure."
            ),
            "FAIL_C1_holds_C2_fails": (
                "Class axis exercisable, GAP-A divergence real, ARM_ON minted distinct "
                "rules + matured + MAINTAINED (frac_active >= 0.30), propagation "
                "non-vacuous (the bias REACHES committed action), the MECH-448 demotion "
                "lever is LIVE on both arms (envelope active + actually excluding; F "
                "demoted from the argmin), AND the MECH-449 active No-Go is LIVE and "
                "ACTUALLY SUPPRESSING on both arms (the perseveration No-Go removes "
                "candidates from the F-eligible set) -- but the differentiated rule_state "
                "STILL adds no marginal committed-class diversity EVEN UNDER active Go/No-Go "
                "opponency. The conversion ceiling persists WITH active No-Go: a GENUINE "
                "SIGNAL that the Go/No-Go opponency leg ALONE is insufficient for "
                "behavioural conversion on the GAP-B composite. NOT a MECH-309 / ARC-062 "
                "falsification. non_contributory; ROUTE to the NEXT ARC-107 "
                "selector-constitution ledger component (open components 2/4/5; the "
                "commitment-dependent behavioural DV needs more of the ARC-107 stack) / V4 "
                "directions. Do NOT weaken the claims."
            ),
            "FAIL_C1_substrate_not_ready_requeue": (
                "The committed-class axis was not exercisable, and/or GAP-A "
                "consumed-summary divergence was absent, and/or ARM_ON did not mature "
                "a differentiated pool (frac_active < 0.30), and/or propagation was "
                "vacuous (ARM_ON bias == ARM_OFF), and/or the MECH-448 demotion lever "
                "was vacuous (envelope inactive or all-admit excluded_count==0 on a "
                "flat-F pool), and/or the MECH-449 active No-Go was vacuous (constitution "
                "inactive or nothing suppressed -- the perseveration axis never crossed "
                "gng_perseveration_floor). The falsifier could not run -- NOT an MECH-309 "
                "/ ARC-062 falsification. Route to substrate enrichment / re-queue; do NOT "
                "weaken the claims."
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
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        # Mirror the self-route block at the manifest top level so the indexer's
        # _compute_adjudication (build_experiment_indexes.py: reads
        # manifest.get("interpretation") at the TOP level) surfaces the label.
        # The block is also nested under result[...]; these two keys make it
        # visible to top-level readers (governance / pending_review) without
        # diving into the nested result dict. Emit-hygiene only (the adjudication
        # FLAG stays "n/a" for experiment_purpose="evidence"); per
        # failure_autopsy_V3-EXQ-654f_2026-06-18 Section 4.
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "evidence_direction_note": (
            f"V3-EXQ-654j arc_062 GAP-B behavioural falsifier (MECH-309 / ARC-062), "
            f"SUPERSEDES V3-EXQ-654i (FAIL non_contributory 2026-06-22; "
            f"conversion_ceiling_persists_despite_demotion_route_mech449, CONFIRMED via "
            f"failure_autopsy_V3-EXQ-654i_2026-06-22). 654i was the CLEANEST conversion-"
            f"ceiling instance: ALL C1 readiness gates MET and non-degenerate (rule field "
            f"matured 0.913, propagation 0.027, MECH-448 demotion LIVE excluding 18.4 "
            f"candidates) yet the load-bearing C2 committed-class entropy lift FAILED -- the "
            f"differentiated rule-apprehension bias reaches committed action but does not "
            f"convert EVEN UNDER the live MECH-448 demotion lever (rank-preserving demotion "
            f"is order-preserving over F and structurally cannot express the active No-Go "
            f"WITHDRAWAL conversion needs). The 654i autopsy FIRED the re-derive brake "
            f"(16-17 prior substrate_ceiling/non_contributory autopsies on MECH-309/ARC-062) "
            f"and REFUSED a same-substrate 654j re-running the MECH-448-only stack; the brake "
            f"is RELEASED here because 654j engages a DIFFERENT mechanism -- the newly-built "
            f"MECH-449 (ARC-107) GO/NO-GO ELIGIBILITY CONSTITUTION (active No-Go opponency "
            f"leg; built 2026-06-21, selection-face falsifier V3-EXQ-689g PASS 3/3, MECH-449 "
            f"promoted candidate->provisional 2026-06-22). 654j arms the active No-Go as a "
            f"matched-stack CONSTANT on BOTH arms (use_go_nogo_constitution=True + "
            f"use_dacc=True), fed ECOLOGICALLY via the BUILT default-loop MECH-260 reuse: the "
            f"dACC recency-share `suppression` vector populates the gate's `perseveration` "
            f"axis (the autopsy's named 'active No-Go withdrawal (MECH-260 generalised -> "
            f"MECH-449)' dependency), dropping the F-eligible-but-perseverative winner from "
            f"the eligible set BEFORE the within-eligible _modulatory_accum arbitration. The "
            f"MECH-448 demotion floor is the CHANNEL-ADAPTIVE (mean-relative) floor "
            f"(use_f_eligibility_adaptive_floor=True, mean_factor="
            f"{F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR}; V3-EXQ-689e PASS on arc_062), the built "
            f"replacement for 654i's per-(arm,seed) _calibrate_envelope_floor manual "
            f"recalibration. use_dacc + use_go_nogo_constitution + use_f_eligibility_demotion "
            f"+ use_f_eligibility_adaptive_floor are ALL matched-stack constants on BOTH "
            f"arms; the ONLY swept variable is use_candidate_rule_field (ARM_OFF legacy "
            f"collapsed rule_state vs ARM_ON matured+active differentiated crf_source), on "
            f"the matched 649/643a/SD-056/SP-CEM/MECH-341/CRF stack with the SD-033a bias "
            f"head un-zeroed AND TRAINED in a frozen-encoder P1 REINFORCE window (GAP-D); P0 "
            f"200 ep. PRIMARY DV = COMMITTED-CLASS entropy. C1 (non-vacuity) = class axis "
            f"exercisable AND GAP-A divergence (both arms) AND ARM_ON matured (crf_frac_active "
            f">= {CRF_FRAC_ACTIVE_FLOOR}) AND propagation non-vacuous AND the MECH-448 "
            f"demotion lever LIVE (active >= {DEMOTION_ACTIVE_FRAC_FLOOR} of P2 ticks AND "
            f"excluded_count > {EXCLUDED_COUNT_FLOOR}, both arms) AND the MECH-449 active "
            f"No-Go LIVE and SUPPRESSING (active >= {NOGO_ACTIVE_FRAC_FLOOR} of P2 ticks AND "
            f"mean per-tick safety+soft suppressions > {NOGO_SUPPRESSED_FLOOR}, both arms; "
            f"C1f); C2 (PRIMARY) = paired-by-seed ARM_ON > ARM_OFF committed-class entropy "
            f"strict-above-by-margin (>= {C2_MIN_LIFT_SEEDS}/3 seeds). interpretation_label="
            f"{result['interpretation_label']}. "
            f"C1={result['acceptance_criteria']['C1_substrate_exercisable_and_manipulation_live']}, "
            f"C2={result['acceptance_criteria']['C2_committed_class_lift']}. ONLY the PASS "
            f"branch weights MECH-309/ARC-062 (as supports; closes GAP-I); C1-fail self-"
            f"routes substrate_not_ready_requeue AND C1-holds-C2-fail self-routes "
            f"conversion_ceiling_persists_despite_active_nogo (a genuine signal the Go/No-Go "
            f"opponency leg ALONE is insufficient -> route to the NEXT ARC-107 ledger "
            f"component / V4) -- BOTH non_contributory, NEITHER a falsification (NO weakens "
            f"branch). PROMOTES NOTHING by itself."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "arms": "ARM_OFF (rule field off) vs ARM_ON (rule field on + crf_persist + 666c maintenance levers)",
            "swept_variable": "use_candidate_rule_field",
            "crf_persist_rules_across_episode_reset": True,
            "crf_mature_pool_dynamics": True,
            "crf_context_from_e2_world_forward": True,
            "crf_availability_maintenance": True,
            "crf_maintenance_floor": float(CRF_MAINTENANCE_FLOOR),
            "crf_maintenance_decay": float(CRF_MAINTENANCE_DECAY),
            "crf_substrate_validated_by": "V3-EXQ-666c PASS 2026-06-15 (ready=True)",
            # CRF-gate calibration amend (the now-working CRF stack; 654f C1 fully met):
            "crf_mature_context_match_threshold": float(CRF_MATURE_CONTEXT_MATCH_THRESHOLD),
            "crf_tolerance_conflict_cap": int(CRF_TOLERANCE_CONFLICT_CAP),
            "crf_maintenance_couple_to_theta": bool(CRF_MAINTENANCE_COUPLE_TO_THETA),
            "crf_gate_amend_landed": "ree-v3 main 42895f6 (2026-06-17)",
            "matched_stack": (
                "SP-CEM + candidate_summary_source=e2_world_forward (GAP-A/649, e2 "
                "trained online in P0) + use_modulatory_selection_authority (643a) + "
                "channel routing (cand_world_summary) + MECH-448 RANK-PRESERVING "
                "F->ELIGIBILITY DEMOTION (use_f_eligibility_demotion + CHANNEL-ADAPTIVE "
                "mean-relative floor use_f_eligibility_adaptive_floor) + MECH-449 GO/NO-GO "
                "CONSTITUTION (use_go_nogo_constitution + use_dacc -> MECH-260 perseveration "
                "No-Go axis) + MECH-341 stratified + MECH-313 noise floor + V_s minimal "
                "+ use_gated_policy + use_lateral_pfc_analog "
                "(lateral_pfc_train_rule_bias_head=True, TRAINED in P1) + SD-056 all levers"
            ),
            "primary_dv": "committed-class entropy (class-keyed rule bias)",
            "secondary_negative_control": "within-class-representative entropy (expected ~null)",
            "phases": "P0 e2-train (field matures) -> P1 frozen-encoder bias-head REINFORCE -> P2 frozen measurement",
            "p1_bias_head_trained_via_reinforce": True,
            "propagation_non_vacuity_precondition": True,
            "demotion_non_vacuity_precondition": True,
            "nogo_non_vacuity_precondition": True,
            "e2_trained_in_p0_frozen_in_p1_p2": True,
            "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
            "use_modulatory_channel_routing": USE_MODULATORY_CHANNEL_ROUTING,
            "modulatory_channel_route_source": MODULATORY_CHANNEL_ROUTE_SOURCE,
            "use_modulatory_shortlist_then_modulate": USE_MODULATORY_SHORTLIST_THEN_MODULATE,
            "modulatory_shortlist_mode": MODULATORY_SHORTLIST_MODE,
            "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
            "use_f_eligibility_demotion": USE_F_ELIGIBILITY_DEMOTION,
            "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
            "f_eligibility_dn_sigma": F_ELIGIBILITY_DN_SIGMA,
            "use_f_eligibility_adaptive_floor": USE_F_ELIGIBILITY_ADAPTIVE_FLOOR,
            "f_eligibility_adaptive_mean_factor": F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
            "use_go_nogo_constitution": USE_GO_NOGO_CONSTITUTION,
            "use_dacc": USE_DACC,
            "gng_perseveration_floor": GNG_PERSEVERATION_FLOOR,
            "conversion_lever": "MECH-449 (ARC-107) Go/No-Go eligibility constitution (active No-Go opponency; perseveration No-Go via MECH-260 reuse) on top of the matched MECH-448 rank-preserving F->eligibility demotion (channel-adaptive floor); MECH-449 selection-face PASS V3-EXQ-689g",
            "demotion_validated_by": "V3-EXQ-689d PASS 2026-06-20 (committed entropy 0.938 vs hard-top-k 0.371); MECH-448 provisional 2026-06-21",
            "nogo_validated_by": "V3-EXQ-689g PASS 3/3 2026-06-21 (go_nogo_converts_gated_channel; MECH-449/ARC-107); MECH-449 promoted candidate->provisional 2026-06-22",
            "adaptive_floor_validated_by": "V3-EXQ-689e PASS 2026-06-21 (channel-adaptive envelope excludes on the arc_062 bank)",
            "matched_on_both_arms": "use_f_eligibility_demotion + use_f_eligibility_adaptive_floor + use_go_nogo_constitution + use_dacc (the conversion constants); ONLY swept variable is use_candidate_rule_field",
            "active_nogo_engagement": "654j engages the MECH-449 active No-Go (the demotion-alone 654i conversion ceiling's named off-ramp); the perseveration No-Go is fed ecologically by the MECH-260 recency-share vector (use_dacc=True), NOT synthetic injection",
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
    parser = argparse.ArgumentParser(description="V3-EXQ-654j GAP-B falsifier (MECH-449 active Go/No-Go constitution + MECH-448 demotion w/ channel-adaptive floor; supersedes 654i)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    if args.dry_run:
        out_path = out_dir / f"_dry_{manifest['run_id']}.json"

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"C1={result['acceptance_criteria']['C1_substrate_exercisable_and_manipulation_live']} "
        f"C2={result['acceptance_criteria']['C2_committed_class_lift']} "
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
