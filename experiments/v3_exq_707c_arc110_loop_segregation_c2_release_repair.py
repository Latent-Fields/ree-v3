#!/opt/local/bin/python3
"""
V3-EXQ-707c -- ARC-110 LOOP-SEGREGATION C2 RELEASE VALIDATION, INSTRUMENT REPAIR (diagnostic).

SUPERSEDES V3-EXQ-707b. This is INSTRUMENT REPAIR, not a substrate build and not another
letter circling a ceiling: the loop substrate is BUILT and was demonstrated LIVE by 707b
(in_layer_null_live and frac_pre_ge2 both SURVIVE its re-adjudication). What 707b never
validly measured is its own DV.

ROUTED BY the CONFIRMED, user-gated re-adjudication
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-707b_2026-07-20.{md,json}
(REE_assembly fc0ceff52d + 544a0633ab), which WITHDREW 707b's evidence_direction=weakens on
ARC-110 as a measurement_test_design_defect, AND withdrew the narrowing derived from it
("loop segregation is necessary-but-not-sufficient; the conversion route requires learned/
DA-gated cross-loop arbitration"). ARC-110 therefore returns to UNTESTED on the single-
arena-artefact question -- neither weakened nor narrowed. 707c is the venue that recovers it.

THE DEFECT 707c REPAIRS (autopsy sec 1, defect FORM 2 -- hold-weighted E3 readout).
707b accumulated `committed_class_counts` on EVERY P2 env step gated only by `is_p2`
(707b:1023-1035) with no `ticks["e3_tick"]` guard, while `ree_core/agent.py:5429` returns
the HELD / trajectory-stepped action before `e3.select()` is ever reached. The E3 cadence
defaults to 10 (`utils/config.py:2017`) and is MECH-093-modulated 5-20
(`heartbeat/clock.py:52-70`). The primary DV `committed_class_entropy_nats` was therefore an
entropy over a HOLD-WEIGHTED histogram -- DISQUALIFYING, because an entropy is a
distribution-SHAPE statistic and replication reweights the very distribution it measures.
No diagnostics latch is touched, so `e3_diagnostics_staleness_lint` (form 1) is structurally
blind to it. NOT a mere inflated n: hold duration is CLASS-dependent and arm exposure spread
reached +97.6% (s45) / +49.8% (s42) / -23.8% (s47), so it does not cancel, and the 663
matched-replay calibration explicitly does not cover this shape (autopsy 699 sec 4d).
DECISIVE: the A1-vs-null contrast the pre-registered DECISIVE branch depends on was
SIGN-INCONSISTENT across all three divergent seeds (-0.1173 s42, -0.0678 s44, +0.1687 s46)
and its pooled +0.0153 nats is EIGHT TIMES below the demonstrated contamination floor
(0.115-0.134 nats). The branch 707b took was UNREACHABLE with that instrument, whichever way
the bias ran. NOT recoverable by reanalysis (no per-tick sink, no fresh-select telemetry).

THE EIGHT REPAIRS (autopsy sec 8, all implemented here):
 (1) FRESH-SELECT-ONLY DV. `agent.e3.last_score_diagnostics` is cleared to None immediately
     before EVERY `select_action`; the DV is accumulated ONLY if it repopulated (pattern:
     v3_exq_785a:525-543). A latched tick records NOTHING. The random-fallback action
     (707b:993-997, taken when select_action returns None) is ALSO excluded -- 707b counted
     it into the primary DV, an independent inflation path.
 (2) REPLICATION TELEMETRY. n_fresh_select / n_latched / fresh_select_yield per arm-seed.
 (3) EXPOSURE ON THE RECORD. n_p2_ticks and exposure_imbalance_vs_a0 per arm-seed --
     REPORTED, NEVER GATING. The +98% spread is the mechanism of the distortion.
 (4) fresh_selects_sufficient READINESS GUARD (>= MIN_FRESH_SELECTS genuine fresh selects on
     EVERY arm-seed, measured on the WORST cell) so an honest-but-underpowered DV self-routes
     substrate_not_ready_requeue, NEVER a false weakens. (708a's pattern.)
 (5) DIVERGENCE-HEADROOM GUARD (NEW, autopsy sec 3b). 707b's enough_divergent_seeds cleared
     at EXACTLY 3.0/3.0 with seed 47 missing the 0.05 floor by 0.00064 (1.3%) -- zero
     headroom, on a statistic itself drawn from the contaminated cache path
     (agent.py:4812 returns CACHED candidates on a non-E3 tick). 707c (a) recomputes that
     statistic on FRESH TICKS ONLY, (b) raises MIN_DIVERGENT_SEEDS 3 -> 4, (c) widens SEEDS
     6 -> 8 so 4 is reachable rather than nominal, and (d) reports the per-seed margin to the
     floor so a near-miss is visible instead of silent. The identical 3.0/3.0 condition is
     flagged in 708 -- this is a LINEAGE-WIDE fragility, not a 707b quirk.
 (6) GRADED LIMBIC-ROUTING READOUT (NEW, autopsy sec 3a). 707b's named_channel_routing_live
     read EXACTLY sqrt(2) on every arm and every seed (zero variance across all 24 cells).
     That is a STRUCTURAL CONSTANT of project_channel_range (`e3_selector.py:124-177`): the
     projection is onto the leading right-singular vector of the CENTERED feature matrix, so
     a channel routed as orthonormal one-hot rows yields ||e_i - e_j|| = sqrt(2) whenever
     >= 2 candidates occupy distinct categories -- a BINARY liveness indicator (0 vs sqrt(2)),
     carrying NO magnitude. It is retained here for liveness ONLY, with its framing corrected.
     The C2 non-degeneracy claim now rests instead on `loop_limbic_pref_range`
     (`e3_selector.py:1881-1883`, the z-scored limbic-loop per-candidate preference range),
     which is genuinely GRADED in the 707b record: 0.013 -> 3.12 across cells, hard ZERO on
     A0 (routing off), and it COLLAPSES on ARM_DROP_LIMBIC exactly where the limbic channels
     are remapped away (s43: A1 1.0931 -> DROP 0.0133) while assoc_pref_range RISES to
     receive them. That dissociation is the measured per-candidate competition C2 needs.
     NO ree_core CHANGE: both diagnostics already exist and 707b already collected them.
 (7) supersedes: "V3-EXQ-707b" on the queue entry AND the manifest.
 (8) stamp_recording_core via write_flat_manifest, with `arm_results` hoisted to the manifest
     TOP LEVEL so the multi-arm substrate_hash hoist actually fires. 707b was missing SIX
     always-core fields (recording_schema, substrate_hash, machine_class, elapsed_seconds,
     config, seeds), so its provenance is unpinned and no arm reuse is possible from it.

WHAT SURVIVES 707b and is RE-CONFIRMED, not re-established (autopsy sec 3): in_layer_null_live
(the S2 same-layer null is genuinely live -- a strict >0 tick count, 1700-18072 on ARM_NOISE
against a HARD ZERO on A0/A1; replication cannot manufacture a positive from an all-zero
record) and frac_pre_ge2 (exactly 1.0 on all 24 cells, saturated). The same-layer null that
the single arena could NOT construct -- the 704b/706b binding constraint -- was genuinely
constructed. That is a RETAINED POSITIVE result about the v4_loop_segregation build.

RE-DERIVE BRAKE: NOT FIRED (autopsy targets[0].re_derive_brake.fired=false). The recommended
category is measurement_test_design_defect, so the counting rule (substrate_ceiling /
non_contributory-as-ceiling) is not met; the 2026-06-29 autopsy's `weakens` is WITHDRAWN by
this re-adjudication rather than joined by a second, so ARC-110's ceiling count moves to 0,
not 2. The brake stops a claim being re-tested at the same granularity against the same
ceiling letter after letter; 707c is instrument repair of a run that never validly measured
its DV -- the sanctioned V3-EXQ-785 -> 785a and 708 -> 708a shape.

--- 707b's own header follows (the design 707c inherits unchanged) ---

SUPERSEDES V3-EXQ-707a (and 707). 707/707a left C2 (limbic-loop load-bearing) SUBSTRATE-
BLOCKED: ARM_DROP_LIMBIC was BYTE-IDENTICAL to A1_LOOPS on every seed, so the per-loop
ablation produced ZERO change in the committed readout and C2 could not be tested (707a's
additive c2_drop_differs_from_a1 diagnostic flagged exactly this byte-identity). The 707
code autopsy (2026-06-28) traced it to a stack of three defects, the deepest being the
MECH-191 PHASIC-EXTERNALISATION GAP: even with the named limbic input modules enabled, the
named cortical bias HEADS (OFC/dACC/lPFC/vigour/liking) emit per-candidate-FLAT output --
the OFC head's INPUT carries real per-candidate range (~0.028) but its OUTPUT range is
exactly 0.0 -- so under per-loop zscore a flat channel is INERT and the limbic loop carried
no per-candidate competition. The only channels carrying per-candidate range were the lumped
residual/mech341/route (the GAP-A project_channel_range path), all in the associative loop.

THE C2 RELEASE (landed 2026-06-28 via /implement-substrate, ree-v3 ree_core): a no-op-default
E3Config flag `use_named_channel_routing` routes each named channel's per-candidate
REPRESENTATION (OFC/lPFC world-summaries [K,D], liking goal-proximity [K], vigour first-action
one-hots [K,A], dACC payoff/effort [K,2], gated_policy summaries [K,D]) through the SAME
parameter-free, range-preserving project_channel_range projection that keeps the lumped
`route` channel phasic, and substitutes that routed term for the flat bias-head scalar in the
segregated-loop accumulation ONLY (the _lcg_terms eligibility traces / authority recompose /
F-score commit path are UNCHANGED; byte-identical OFF). So the limbic loop (ofc/liking/vigour)
now carries REAL per-candidate range and ARM_DROP_LIMBIC can genuinely differ from A1_LOOPS --
making C2 testable. Regression guard: ree-v3/tests/test_arc110_loop_segregation.py
(TestNamedChannelRoutingC2Release + TestRoutedRepsReachSelectorThroughAgent).

CHANGES vs 707a: (1) the loop arms enable `use_named_channel_routing=True` AND the limbic
input modules (use_ofc_analog / use_mech295_liking_bridge / use_tonic_vigor) -- matched as a
CONSTANT on ALL arms incl. A0 so the only swept factor stays loop segregation; (2) a new
per-NAMED-channel non-degeneracy PRECONDITION (`named_channel_routing_live`): on A1_LOOPS at
least one LIMBIC channel's routed per-candidate range must clear a substantive floor on a
strict-majority of divergent seeds, evaluated BEFORE C2 is scored -- so the vacuous DROP==A1
condition self-routes substrate_not_ready_requeue, NEVER a false weakens; (3) C1 (A1 strict-
above max(A0, valid in-layer null ARM_NOISE)+margin) UNCHANGED, C2 (A1 strict-above
ARM_DROP_LIMBIC) re-tested now that DROP can genuinely differ. The c2_drop_differs_from_a1
diagnostic is retained (it should now be TRUE when the gate passes). The 707a in-layer-null
LIVENESS gate (null perturbs committed selection, not null>A0) is also UNCHANGED.

Substrate-readiness validation for the v4_loop_segregation substrate (ARC-110 + the
coupled ARC-109 D1/D2 split + MECH-452 loop-local traces + the S2 in-layer null) built
2026-06-27 via /implement-substrate. EXPERIMENT_PURPOSE=diagnostic; PROMOTES NOTHING.
Routed by the CONFIRMED cluster autopsy
REE_assembly/evidence/planning/failure_autopsy_704b-706b-conversion-ceiling_2026-06-27.{md,json}:
the conversion-ceiling lineage converged from four angles (700b/700c learned-gating+settling,
704b finer-channel, 706b double-gated curiosity) on the conclusion that the V3 SINGLE E3
foraging arena structurally denies (a) committed-action-class conversion from non-motor
channels and (b) a VALID same-layer null. The gate cleared by V3-EXQ-704b FAIL-to-convert
(the pre-registered positive-evidence-FOR-ARC-110 outcome: representational compression is
NOT the binding constraint).

THE QUESTION (ARC-110)
----------------------
Is the F-dominance conversion ceiling (MECH-439) partly an ARTEFACT of the single-arena
collapse? The hypothesis: with N>=3 parallel segregated cortico-BG-thalamic loops (motor /
associative / limbic), each running within-loop competition FIRST and cross-loop arbitration
AFTER, F dominates only the MOTOR loop and cannot drown the limbic "is this worth committing
to" value -- so committed-action diversity converts where the single F-dominated arena
plateaued. AND the in-layer (eligibility/settling-field) null can now VERIFY-LIFT, the valid
same-layer null the single arena could not construct (the 704b/706b binding constraint).

THE 4 ARMS (all carry the SAME landed arithmetic envelope + finer-channel gating + learned
settling as a MATCHED CONSTANT; the ONLY swept factor is loop segregation):
  A0_SINGLE_ARENA  : the single-arena baseline -- use_loop_segregation=False; finer-channel
                     gating + learned settling ON in ONE shared E3 arena (= the V3-EXQ-704b
                     A2 lineage that plateaued). The conversion baseline.
  A1_LOOPS         : use_loop_segregation=True (motor=F / associative=dACC+lPFC /
                     limbic=OFC+liking+vigour) + use_d1_d2_population_split + use_loop_local_
                     eligibility_traces. Within-loop competition first, Haber ascending-spiral
                     arbitration after, per-loop zscore normalisation (strips F's magnitude
                     advantage). The conversion arm.
  ARM_NOISE        : A1_LOOPS + loop_segregation_noise_on=True -- the S2 IN-LAYER same-layer
                     null: each non-motor loop accumulator is replaced by a magnitude-matched
                     random-structure perturbation at the SAME layer the loops settle on (NOT
                     policy temperature, the decoupled 700-lineage null). This is the VALID
                     same-layer null the single arena could not construct; a lift is
                     attributable to learned loop STRUCTURE only if it beats this null.
  ARM_DROP_LIMBIC  : A1_LOOPS with the LIMBIC loop ablated (its channels remapped to the
                     associative loop, so the limbic loop is empty) -- the ARC-106 per-loop
                     load-bearing falsifier: if dropping the limbic loop does NOT remove the
                     A1 lift, the limbic loop is DECORATIVE.
6 seeds. PRIMARY DV = committed-action-class entropy (nats), measured over P2.
claim_ids = [ARC-110]. experiment_purpose = diagnostic (substrate-readiness; PROMOTES NOTHING).

PRE-REGISTERED OUTCOME MAP (decisive either way)
------------------------------------------------
  PASS / supports ARC-110 (single-arena collapse WAS a binding constraint):
    A1_LOOPS lifts committed-action-class entropy strict-above A0_SINGLE_ARENA AND
    strict-above the VALID in-layer null ARM_NOISE on a strict-majority of DIVERGENT seeds,
    with live cross-loop variance (a non-motor loop actually FLIPPED the within-eligible
    winner). If ARM_DROP_LIMBIC does NOT reproduce the lift, the limbic loop is load-bearing.

  WEAKENED / route-elsewhere (collapse was NOT the binding constraint):
    a VALID same-layer null can now be constructed (ARM_NOISE is a LIVE perturbation of the
    committed selection) and loops carry live cross-loop variance, BUT A1_LOOPS does NOT
    convert committed diversity strict-above A0/the null. The conversion ceiling is INTRINSIC,
    not an artefact of single-arena collapse -> weakens ARC-110.

NON-VACUITY READINESS GATES (self-route substrate_not_ready_requeue, NEVER a false weakens):
  (1) candidate pool DIVERGENT: GAP-A guard cand_world_pairwise_dist > floor (per seed).
  (2) LOOPS carry live CROSS-LOOP VARIANCE on A1: loop_committed_neq_motor_winner OR
      loop_cross_loop_winner_disagreement on a majority of divergent seeds, AND per-loop
      loop_assoc_pref_range / loop_limbic_pref_range > 0. A "segregated" loop pinned to the
      motor winner is a vacuous split -> requeue (the ARC-110 what_would_answer guard).
  (3) the VALID in-layer null (ARM_NOISE) is a LIVE same-layer perturbation -- on a strict-
      majority of divergent seeds it actively perturbs the committed selection
      (loop_noise_active_ticks > 0 AND loop_frac_committed_neq_motor or loop_frac_disagree >
      the cross-loop-variance floor). This is the SAME-LAYER null the single arena could not
      construct; if even the in-layer null is INERT (does not reach the committed-class
      readout) the null-validity problem persists -> requeue, not weakens. NOTE (707->707a):
      the RETIRED proxy required the null to LIFT entropy strict-above A0 -- the 700-lineage
      TEMPERATURE-null liveness test, INVALID for a structured-accumulator null (random
      magnitude-matched loop content lands at baseline, not above it; structured beats random,
      random ~ A0). null-LIVENESS, not null-LIFT, is the correct non-vacuity guard, and it
      keeps the WEAKENS branch reachable. The path-artifact hypothesis (A1's lift is a zscore-
      path artefact) is ruled out by C1's "A1 strict-above the null", NOT by null-lift.
  (4) signed delta_t NON-FLAT + finer w_chan_finer entries actually MOVE (learning engaged).

ARC-106 load-bearing-vs-decorative: ARM_DROP_LIMBIC is the per-loop ablation. If dropping the
limbic loop leaves the A1 lift unchanged, the limbic loop is DECORATIVE (recorded; C2).
Phased training is NOT required in the encoder sense (reuses trained valuation heads; learned
objects use the ARC-108 LOCAL three-factor update), but the P0/P1/P2 phasing is kept for a
fair comparison with the 704b baseline. MECH-094: learning writes are waking-only (inherited);
the in-layer null is selection-only (no memory write).

See REE_assembly/docs/architecture/sd_v4_loop_segregation.md (ARC-110 design-of-record + IMPLEMENTED block),
    REE_assembly/evidence/planning/failure_autopsy_704b-706b-conversion-ceiling_2026-06-27.{md,json} (routes this build),
    REE_assembly/evidence/planning/substrate_queue.json (sd_id v4_loop_segregation),
    ree-v3/ree_core/predictors/e3_selector.py (_segregated_loop_arbitrate / _d1_d2_split / _loop_inlayer_null),
    experiments/v3_exq_704b_mech451_finer_channel_granularity_falsifier.py (matched-substrate sibling).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import deque
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
from experiments._lib.arm_reuse import try_reuse_cell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_707c_arc110_loop_segregation_c2_release_repair"
QUEUE_ID = "V3-EXQ-707c"
SUPERSEDES = "V3-EXQ-707b"   # instrument repair: 707b's hold-weighted committed-class entropy DV was DISQUALIFYING (autopsy 2026-07-20)
BACKLOG_ID = None   # no proposal; routed by failure_autopsy_704b-706b-conversion-ceiling_2026-06-27
CLAIM_IDS: List[str] = ["ARC-110"]
EXPERIMENT_PURPOSE = "diagnostic"   # substrate-readiness validation; PROMOTES NOTHING

# softplus-unity init for w_chan_finer (softplus(_FCG_W_INIT) == 1.0).
_FCG_W_INIT = math.log(math.e - 1.0)

# CRF-gate calibration levers (matured CRF stack; ported verbatim from 700c,
# matched on all arms -- the differentiated conversion source).
CRF_MATURE_CONTEXT_MATCH_THRESHOLD = 0.7
CRF_TOLERANCE_CONFLICT_CAP = 3
CRF_MAINTENANCE_COUPLE_TO_THETA = True
CRF_MAINTENANCE_FLOOR = 0.45
CRF_MAINTENANCE_DECAY = 0.0

# ----- Acceptance thresholds (pre-registered) -----
# C1 conversion: strict-above margin on committed-class entropy (nats).
CONVERSION_MARGIN = 0.05
# C2 grows-over-training: second-half P2 entropy strict-above first-half margin.
GROWTH_MARGIN = 0.02

# ----- Per-seed-divergent gating (701a-style) -----
# 707c REPAIR 5 (autopsy sec 3b -- DIVERGENCE HEADROOM). 707b set this to 3 and cleared it at
# EXACTLY 3.0/3.0 -- zero headroom -- with seed 47 missing the CONSUMED_SPREAD_FLOOR by
# 0.00064 (1.3% of the floor). One seed crossing flips n_divergent 3 -> 4 and changes both the
# C1 and C2 denominators, so the design's own admission gate had no slack. Worse, the
# statistic behind it (consumed_summary_pairwise_dist_mean) was accumulated on EVERY P2 env
# step from agent.py:4812, which returns CACHED candidates on a non-E3 tick -- the same
# defect family as the primary DV. 707c fixes BOTH ends: the statistic is recomputed on
# FRESH TICKS ONLY (see the cell loop), and the threshold is raised to 4 so clearing it means
# genuine headroom rather than a nominal pass. SEEDS is widened 6 -> 8 in the same move so 4
# is REACHABLE and not merely nominal -- raising the bar without widening the pool would just
# convert 707b's zero-headroom pass into a near-certain requeue. The per-seed margin to the
# floor is reported either way (divergence_margins), so a near-miss is visible, never silent.
# The identical 3.0/3.0 zero-headroom condition is independently flagged in 708 -- it is a
# LINEAGE-WIDE fragility, which is why this is a threshold change and not a 707b-local patch.
MIN_DIVERGENT_SEEDS = 4          # of 8 (707b: 3 of 6, cleared at EXACTLY threshold)
DIVERGENT_PASS_FRACTION = 0.5    # strict-majority-ish gate within the divergent seeds
MIN_SEEDS_FOR_PASS = 2           # absolute floor of divergent seeds clearing a criterion

# ----- 707c REPAIR 1/2/4: fresh-select-only DV, telemetry, and readiness floor -----
# The E3 cadence (heartbeat.e3_steps_per_tick) defaults to 10 and is MECH-093-modulated 5-20,
# so the honest denominator is ~1/10 of 707b's per-env-step count.
#
# REACHABILITY, checked against the FROZEN 707b RECORD rather than assumed. This matters:
# V3-EXQ-699b (ree-v3 62b3f43) set the same gate at 100 and found on inspection that its
# headroom was THIN -- not the ~12x assumed but ~46..1200 against the floor, because episodes
# terminate early and P2 env-step counts vary by >25x. Running the same check here, over
# 707b's 24 recorded cells (n_p2_ticks 1409 .. 18658, worst cell ARM_DROP_LIMBIC seed 42):
#     fastest cadence 5  (upper bound)   ~282 fresh   -> 9.4x this floor
#     785a measured real-run yield 0.12  ~169 fresh   -> 5.6x
#     cadence 10 (default)               ~141 fresh   -> 4.7x
#     slowest cadence 20 (lower bound)    ~70 fresh   -> 2.3x
# So even the PESSIMISTIC bound clears with 2.3x margin, and a cell failing this gate
# signals a real substrate problem (commitment latch pinned open / E3 never re-selecting),
# not ordinary sampling. Deliberately kept at 708a's 30 rather than raised toward 699b's 100:
# the gate exists to catch a starved DV, and a floor set near the lower bound would convert
# ordinary early-episode-termination variance into spurious requeues.
#
# Measured on the WORST cell (min across arm-seeds) to match the all(...) quantifier the
# `met` expression uses, so the indexer's authoritative recompute agrees with the flag we set.
MIN_FRESH_SELECTS = 30
# Exposure imbalance is REPORTED, NEVER GATING (autopsy sec 8 property 3). This floor only
# marks a cell as beyond-report in the manifest; it gates nothing. 707b's spread reached
# +97.6%, which is the mechanism of the distortion and must be on the record either way.
EXPOSURE_IMBALANCE_REPORT_FLOOR = 0.05

# ARC-110 non-degeneracy: loops must carry LIVE cross-loop variance on the loop arms.
# A non-motor loop must FLIP the within-eligible winner (or loops disagree) on at least this
# fraction of P2 select ticks, AND a non-motor loop must carry per-loop preference spread > 0.
LOOP_CROSS_VARIANCE_FRAC_FLOOR = 0.05
LOOP_PREF_RANGE_FLOOR = 1e-6
# ARC-110 C2 RELEASE non-degeneracy gate (707b): on A1_LOOPS, at least one LIMBIC channel
# (ofc/liking/vigour) must reach the segregated-loop arbitration carrying a per-candidate
# routed range above this SUBSTANTIVE floor (peak over P2 ticks), on a strict-majority of
# divergent seeds, BEFORE C2 (limbic load-bearing) is scored. This is the precondition the
# 707 vacuous DROP==A1 lacked: if the named limbic channels are still per-candidate-FLAT
# (the MECH-191 phasic gap), the limbic loop carries no competition and the per-loop ablation
# is a no-op -> self-route substrate_not_ready_requeue, NEVER a false weakens. Set well above
# the 1e-6 inert floor (so a collapsed/flat channel fails) but below the routed range a
# genuine world-summary / proximity / action-class representation yields (the OFC input range
# was ~0.028; project_channel_range preserves that order). 1e-3 ~= 36x the inert floor.
LIMBIC_ROUTED_RANGE_FLOOR = 1e-3
# Named channels assigned to the limbic loop (must match e3_selector _LOOP_DEFAULT_CHANNEL_MAP).
LIMBIC_NAMED_CHANNELS = ("ofc", "liking", "vigour")

# ----- 707c REPAIR 6: GRADED limbic-loop competition (autopsy sec 3a) -----
# WHY THE 707b GATE CANNOT CARRY C2's MAGNITUDE CLAIM. `named_channel_routing_live` above is
# carried entirely by the `vigour` channel at EXACTLY 1.414214 = sqrt(2), on every arm and
# every seed -- ZERO variance across the whole 24-cell design -- against {} / 0.0 when routing
# is off. That is a STRUCTURAL CONSTANT of project_channel_range (e3_selector.py:124-177), not
# a measured substrate property: the projection centers the [K, D] feature matrix and projects
# onto its leading right-singular vector, so a channel routed as ORTHONORMAL one-hot rows
# yields ||e_i - e_j|| = sqrt(2) whenever >= 2 candidates occupy distinct categories --
# independent of K, of the seed, and of the arm. It is therefore a BINARY liveness indicator
# (0 vs sqrt(2)) and is RETAINED here as exactly that, with its framing corrected: it
# certifies "routing is ON and non-degenerate" (and it does cleanly resolve 707's vacuous
# byte-identical DROP == A1), but it does NOT evidence that the limbic channel carries
# SUBSTANTIAL per-candidate competition. The 2026-06-29 reading "1.414 >> 0.001" reads as a
# 1414x margin on a graded magnitude and is not one.
#
# THE GRADED REPLACEMENT. `loop_limbic_pref_range` (e3_selector.py:1881-1883) is the range of
# the z-scored limbic-loop accumulation across the eligible candidates -- i.e. the actual
# per-candidate competition the limbic loop contributes to selection. It is genuinely graded
# in the 707b record (already collected there, never gated on):
#     A0  (routing off)     0.000000 on all 6 seeds        <- hard zero, correct
#     A1_LOOPS              1.0931 .. 2.4319
#     ARM_DROP_LIMBIC       0.0133 .. 2.1337
# and it DISSOCIATES exactly where C2 requires: on s43 the limbic range COLLAPSES A1 1.0931 ->
# DROP 0.0133 (a 98.8% loss) while assoc_pref_range RISES 1.1802 -> receives the remapped
# channels. That is the measured mechanism C2's non-degeneracy claim needs, and it is what
# 707c gates on. NO ree_core CHANGE IS REQUIRED -- both diagnostics already exist and 707b
# already summed them; only the gate is new.
#
# FLOOR. Set at 0.05, ~5x above the collapsed DROP s43 value (0.0133) so a genuinely
# collapsed limbic loop fails, and ~20x below the observed A1 median (~1.9) so a live loop
# clears it with room. Deliberately NOT set from 707c's own statistics (pre-registered).
LIMBIC_GRADED_PREF_RANGE_FLOOR = 0.05

# C1(a) readiness: committed-class axis exercisable (>= 2 candidate first-action classes).
FRAC_PRE_GE2_FLOOR = 0.30
# Non-vacuity (b): GAP-A consumed-summary divergence (649 statistic + 643a ceiling).
CONSUMED_SPREAD_FLOOR = 0.05
CONSUMED_MAGNITUDE_CEIL = 1.0e6
# Non-vacuity (c): delta_t carries cross-tick variance (outcome variance to learn from).
DELTA_T_STD_FLOOR = 1e-4
# Non-vacuity (1): the finer w_chan_finer entries MOVED + are DISSOCIABLE (cross-channel
# range above floor -- finer entries that move identically are the blend re-labelled).
W_CHAN_FINER_RANGE_FLOOR = 1e-4   # softplus-unity init => range 0; >floor == reorganised + dissociable
# 704b near-vacuity guard (autopsy Step 7): the A2 LEARNED finer-channel range must be
# SUBSTANTIVE, not merely floor-clearing. V3-EXQ-704's A2 dissociation was 0.00148 (~15x the
# bare 1e-4 floor) -- the autopsy flagged that as near the "compressed blend re-labelled"
# vacuity boundary, where even a correctly magnitude-matched strict-above-noise bar is a
# knife-edge tiny-structure-vs-tiny-matched-noise non-test. A dissociation below this
# SUBSTANTIVE floor on a strict-majority of seeds self-routes substrate_not_ready_requeue
# (NEVER a false weakens). 5e-3 ~= 3.4x the prior near-vacuous 704 realised range (0.00148);
# erring high only yields an honest "not ready", never a misleading weakens.
W_CHAN_FINER_SUBSTANTIVE_RANGE_FLOOR = 5e-3
# A1 load-bearing-ablation non-vacuity: the single global w_chan also genuinely learned.
W_CHAN_RANGE_FLOOR = 1e-4

# CRF maturity readiness (matched constant; the differentiated source must be present).
CRF_MIN_MINTED = 2
CRF_N_ACTIVE_FLOOR = 1
CRF_FRAC_ACTIVE_FLOOR = 0.30

# 707c REPAIR 5: widened 6 -> 8 so MIN_DIVERGENT_SEEDS=4 is REACHABLE rather than nominal.
# 707b returned 3 divergent of 6 (50%) on the CONTAMINATED cache-path statistic, with a 4th
# (s47) missing by 1.3%; recomputing that statistic on fresh ticks only may move which seeds
# qualify in either direction, so the pool is widened rather than the bar simply raised.
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
P0_WARMUP_EPISODES = 100
P1_BIAS_TRAIN_EPISODES = 50
P2_MEASUREMENT_EPISODES = 100   # binned first/second half for C2 grows-over-training
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 4   # >= 2 so the P2 first/second-half split is exercised in the smoke
DRY_RUN_STEPS = 30

# ARC-110 S2 same-layer null (ARM_NOISE): the magnitude-matched random-structure perturbation
# is generated IN-SELECTOR per-tick at the non-motor loop accumulators (config.e3.
# loop_segregation_noise_on -> _loop_inlayer_null), NOT injected here. No manual w_chan_finer
# seeding is needed -- the null is a property of the loop substrate, not the experiment driver.

# Mint run_id to cite for arm-reuse; None => run all arms fresh (the safe default for a
# fresh lineage). A future sibling sets this to a prior 704 run_id after one lands.
REUSE_BASELINE_FROM = None

# Stable arm metric keys the acceptance logic reads from a reusable row (a reused cell
# MUST have recorded all of them -- the section-9.2 correctness trap).
REUSE_NEEDED_KEYS = [
    "committed_class_entropy_nats",
    "committed_class_entropy_p2_first_half_nats",
    "committed_class_entropy_p2_second_half_nats",
    "gapa_divergence",
    "frac_pre_ge2",
    "consumed_summary_pairwise_dist_mean",
    "consumed_summary_pairwise_dist_max",
    "crf_differentiated",
    "crf_frac_active_ge_floor",
    "crf_n_minted_total",
    "fcg_moved",
    "fcg_delta_nonflat",
    "fcg_w_chan_finer_range_max",
    "fcg_w_chan_finer_std_max",
    "fcg_delta_t_std",
    "lcg_moved",
    "lcg_delta_nonflat",
    "lcg_w_chan_range_max",
    "lcg_delta_t_std",
    "n_unique_committed_classes",
    "error_note",
]

# --- Matched-stack lever constants (identical on ALL arms; the landed envelope) ---
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
USE_MODULATORY_CHANNEL_ROUTING = True
MODULATORY_CHANNEL_ROUTE_SOURCE = "cand_world_summary"
MODULATORY_CHANNEL_ROUTE_WEIGHT = 1.0
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6
USE_MODULATORY_SHORTLIST_THEN_MODULATE = True
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_SHORTLIST_K = 3
USE_F_ELIGIBILITY_DEMOTION = True
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30
F_ELIGIBILITY_DN_SIGMA = 0.0
USE_F_ELIGIBILITY_ADAPTIVE_FLOOR = True
F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR = 1.0
USE_GO_NOGO_CONSTITUTION = True
USE_DACC = True
GNG_PERSEVERATION_FLOOR = 0.5
GNG_SAFETY_FLOOR = 0.5
GNG_PROTECT_MIN_ELIGIBLE = 1
MECH341_ENTROPY_BIAS_SCALE = 2.0
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4

USE_CANDIDATE_RULE_FIELD = True

# ----- ARC-108 JOB-1 step-1 learned-gating knobs (substrate defaults; matched when armed) -----
# Reused VERBATIM by the MECH-451 finer path (same eta/elig/baseline/asym), so
# A1_GLOBAL_WCHAN vs A2_FINER differ ONLY in channel granularity (single-variable design).
LCG_ETA = 0.01
LCG_ELIG_DECAY = 0.9
LCG_VALUE_BASELINE_BETA = 0.05
LCG_ASYM_POTENTIATION = 1.0
LCG_ASYM_DEPRESSION = 0.5

# ----- MECH-450 settling (ON on all arms; the within-loop settling each loop runs) -----
LEARNED_SETTLING_ROUNDS = 3
LEARNED_SETTLING_TEMPERATURE = 1.0
LEARNED_SETTLING_ETA = 0.01
LEARNED_SETTLING_ELIG_DECAY = 0.9

# ----- ARC-110 loop-segregation knobs (matched on all loop arms) -----
LOOP_SEGREGATION_NORMALIZE = "zscore"   # per-loop preference normalisation (strips F's magnitude)
LOOP_SEGREGATION_NOISE_ALPHA = 1.0      # S2 in-layer null: range == alpha x the real loop range

# SD-056 online e2 training (mirror 700c).
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

# P1 bias-head REINFORCE training (mirror 700c).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9


# IDENTICAL env to 700c (the GAP-A reef-bipartite foraging bank).
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


# The 4 arms. ALL carry finer-channel gating + learned settling as a MATCHED CONSTANT;
# the ONLY swept factor is loop segregation (loop_seg) + its noise / drop-limbic variants.
# A0_SINGLE_ARENA is the single-arena baseline (= the 704b A2 lineage). ARM_NOISE adds the
# S2 in-layer null; ARM_DROP_LIMBIC ablates the limbic loop (per-loop load-bearing falsifier).
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "A0_SINGLE_ARENA",
        "label": "single_arena_baseline_finer_gating_settling_no_loop_segregation",
        "finer_on": True,
        "loop_seg": False,
        "loop_noise": False,
        "drop_limbic": False,
    },
    {
        "arm_id": "A1_LOOPS",
        "label": "arc110_segregated_loops_motor_assoc_limbic_d1d2_loop_local_traces",
        "finer_on": True,
        "loop_seg": True,
        "loop_noise": False,
        "drop_limbic": False,
    },
    {
        "arm_id": "ARM_NOISE",
        "label": "arc110_s2_in_layer_same_layer_null_magnitude_matched_random_structure",
        "finer_on": True,
        "loop_seg": True,
        "loop_noise": True,
        "drop_limbic": False,
    },
    {
        "arm_id": "ARM_DROP_LIMBIC",
        "label": "arc110_limbic_loop_ablated_per_loop_load_bearing_falsifier",
        "finer_on": True,
        "loop_seg": True,
        "loop_noise": False,
        "drop_limbic": True,
    },
]

# A0_SINGLE_ARENA is a stable single-arena baseline (self-mint eligible). The loop arms ride
# the just-built loop substrate (in flux for this lineage) -- not minted as reusable baselines.
REUSABLE_ARM_IDS_LOCAL = ("A0_SINGLE_ARENA",)

# ARM_DROP_LIMBIC remaps the limbic channels into the associative loop so the limbic loop is
# empty (the ARC-106 per-loop ablation). Motor stays F; associative absorbs the rest.
_DROP_LIMBIC_CHANNEL_MAP = {
    "dacc": "associative",
    "lpfc": "associative",
    "ofc": "associative",
    "liking": "associative",
    "vigour": "associative",
}


def _arm_config_slice(
    arm: Dict[str, Any],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Declared reuse fingerprint slice: ONLY what an arm's computation reads -- the
    swept arm flags + the matched arithmetic envelope every arm runs + the env + the
    schedule. NEVER acceptance thresholds. Same slice for the consumer fingerprint and
    a future mint, so they match by construction (settling ON, finer gating ON on all arms;
    loop segregation is the swept factor)."""
    return {
        "arm_id": arm["arm_id"],
        "finer_on": bool(arm["finer_on"]),
        "loop_seg": bool(arm.get("loop_seg", False)),
        "loop_noise": bool(arm.get("loop_noise", False)),
        "drop_limbic": bool(arm.get("drop_limbic", False)),
        "use_learned_settling_step": True,
        "use_d1_d2_population_split": bool(arm.get("loop_seg", False)),
        "use_loop_local_eligibility_traces": bool(arm.get("loop_seg", False)),
        # ARC-110 C2 RELEASE (707b): routing (swept with loop_seg) + limbic input modules
        # (matched constant on all arms) -- both change the arm's computation, so the
        # fingerprint must declare them (else a future consumer mis-matches the mint).
        "use_named_channel_routing": bool(arm.get("loop_seg", False)),
        "use_ofc_analog": True,
        "use_mech295_liking_bridge": True,
        "use_tonic_vigor": True,
        "learned_channel_gating_eta": LCG_ETA,
        "lcg_elig_decay": LCG_ELIG_DECAY,
        "lcg_value_baseline_beta": LCG_VALUE_BASELINE_BETA,
        "lcg_asym_potentiation": LCG_ASYM_POTENTIATION,
        "lcg_asym_depression": LCG_ASYM_DEPRESSION,
        "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
        "use_f_eligibility_adaptive_floor": bool(USE_F_ELIGIBILITY_ADAPTIVE_FLOOR),
        "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
        "use_modulatory_selection_authority": bool(USE_MODULATORY_SELECTION_AUTHORITY),
        "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
        "modulatory_authority_normalize_basis": str(MODULATORY_AUTHORITY_NORMALIZE_BASIS),
        "use_modulatory_shortlist_then_modulate": bool(USE_MODULATORY_SHORTLIST_THEN_MODULATE),
        "modulatory_shortlist_mode": str(MODULATORY_SHORTLIST_MODE),
        "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
        "modulatory_channel_route_source": str(MODULATORY_CHANNEL_ROUTE_SOURCE),
        "use_candidate_rule_field": bool(USE_CANDIDATE_RULE_FIELD),
        "use_dacc": bool(USE_DACC),
        "env_kwargs": dict(ENV_KWARGS),
        "sd056_weight": float(SD056_WEIGHT),
        "lr_lpfc_bias": float(LR_LPFC_BIAS),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
    }


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Matched-stack agent. The landed arithmetic envelope (demotion + adaptive floor +
    Go/No-Go + authority + routing + top_k shortlist) + the diversity stack (MECH-341,
    SD-056, CRF, trained lateral_pfc bias head, use_dacc) + finer-channel gating + learned
    settling are MATCHED CONSTANTS on ALL arms. The ONLY swept factor is loop segregation:
    A0_SINGLE_ARENA runs the legacy single-arena within-eligible argmin (use_loop_segregation
    =False); A1_LOOPS / ARM_NOISE / ARM_DROP_LIMBIC run the ARC-110 segregated loops
    (+ ARC-109 D1/D2 + MECH-452 loop-local traces). ARM_NOISE adds loop_segregation_noise_on
    (the S2 in-layer same-layer null); ARM_DROP_LIMBIC remaps the limbic channels into the
    associative loop (the per-loop ablation)."""
    finer_on = bool(arm["finer_on"])
    loop_seg = bool(arm.get("loop_seg", False))
    loop_noise = bool(arm.get("loop_noise", False))
    drop_limbic = bool(arm.get("drop_limbic", False))
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
        # --- Matched stack (identical on all arms) ---
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # GAP-A (V3-EXQ-649): shared per-candidate signal from e2.world_forward.
        candidate_summary_source="e2_world_forward",
        # modulatory-bias-selection-authority (643a) + channel routing.
        use_modulatory_selection_authority=USE_MODULATORY_SELECTION_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_channel_routing=USE_MODULATORY_CHANNEL_ROUTING,
        modulatory_channel_route_source=MODULATORY_CHANNEL_ROUTE_SOURCE,
        modulatory_channel_route_weight=MODULATORY_CHANNEL_ROUTE_WEIGHT,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        # 569i TOP-K shortlist scaffold (the eligible set the gating acts inside).
        use_modulatory_shortlist_then_modulate=USE_MODULATORY_SHORTLIST_THEN_MODULATE,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        # MECH-448 demotion envelope (CONSTANT ON) + channel-adaptive floor (689e).
        use_f_eligibility_demotion=USE_F_ELIGIBILITY_DEMOTION,
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        use_f_eligibility_adaptive_floor=USE_F_ELIGIBILITY_ADAPTIVE_FLOOR,
        f_eligibility_adaptive_mean_factor=F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
        # MECH-449 Go/No-Go constitution (CONSTANT ON). use_dacc feeds the MECH-260
        # perseveration No-Go axis ecologically.
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
        # MECH-313 noise floor -- OFF on EVERY arm (the ARM_NOISE null is a same-layer
        # finer-gating null, NOT a policy-temperature lift).
        use_noise_floor=False,
        # V_s minimal stack.
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        # ARC-062 GatedPolicy (matched; symmetry-broken bias) -- a finer channel source.
        use_gated_policy=True,
        # SD-033a LateralPFCAnalog with the bias head UN-ZEROED + trainable (GAP-D) -- a
        # finer channel source.
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
        use_candidate_rule_field=USE_CANDIDATE_RULE_FIELD,
        # --- MECH-451: FINER separately-learnable channels (ON on ALL arms; the loops
        # partition these finer channels into motor/associative/limbic). ---
        use_finer_channel_gating=finer_on,
        use_learned_channel_gating=False,
        # Shared three-factor knobs (used by the finer w_chan_finer path on all arms).
        learned_channel_gating_eta=LCG_ETA,
        learned_channel_gating_elig_decay=LCG_ELIG_DECAY,
        learned_channel_value_baseline_beta=LCG_VALUE_BASELINE_BETA,
        learned_channel_asym_potentiation=LCG_ASYM_POTENTIATION,
        learned_channel_asym_depression=LCG_ASYM_DEPRESSION,
        # signed RPE (no C3 unsigned ablation in this experiment).
        learned_channel_rpe_mode="signed",
        # --- MECH-450 recurrent settling: ON on ALL arms (the within-loop settling each
        # segregated loop runs; on A0 it is the legacy single-arena settling). ---
        use_learned_settling_step=True,
        learned_settling_rounds=LEARNED_SETTLING_ROUNDS,
        learned_settling_temperature=LEARNED_SETTLING_TEMPERATURE,
        learned_settling_eta=LEARNED_SETTLING_ETA,
        learned_settling_elig_decay=LEARNED_SETTLING_ELIG_DECAY,
        # --- ARC-110 parallel segregated loops (the SWEPT factor) + coupled ARC-109 /
        # MECH-452 / S2 in-layer null. Default-off on A0_SINGLE_ARENA -> legacy single
        # arena. ARM_DROP_LIMBIC remaps the limbic channels into the associative loop. ---
        use_loop_segregation=loop_seg,
        loop_segregation_channel_map=(dict(_DROP_LIMBIC_CHANNEL_MAP) if drop_limbic else {}),
        loop_segregation_normalize=LOOP_SEGREGATION_NORMALIZE,
        loop_segregation_noise_on=loop_noise,
        loop_segregation_noise_alpha=LOOP_SEGREGATION_NOISE_ALPHA,
        use_d1_d2_population_split=loop_seg,
        use_loop_local_eligibility_traces=loop_seg,
        # --- ARC-110 C2 RELEASE (707b): per-named-channel range-preserving routing into the
        # segregated loops, so the named limbic channels carry per-candidate range (the flat
        # bias-head scalars are inert under per-loop zscore -- the MECH-191 phasic gap). ON on
        # the loop arms only (A0 has loop_seg=False so the override is built but never consumed
        # by the legacy single-arena path -> A0 stays the true single-arena baseline). ---
        use_named_channel_routing=loop_seg,
        # --- Limbic-loop INPUT modules: MATCHED CONSTANT on ALL arms (incl. A0) so the only
        # swept factor stays loop segregation. Without these the limbic channels carry NO live
        # representation to route. OFC-devaluation value / MECH-295 drive->liking->approach /
        # MECH-320 tonic vigour -- the three limbic-loop value sources (ofc/liking/vigour). ---
        use_ofc_analog=True,
        use_mech295_liking_bridge=True,
        use_tonic_vigor=True,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# SD-056 online e2 training (verbatim from 700c)
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
# P1 bias-head REINFORCE training (verbatim from 700c)
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
    agent = _make_agent(env, arm)

    # ARC-110 S2 in-layer null is IN-SELECTOR (config.e3.loop_segregation_noise_on), so
    # ARM_NOISE needs no manual buffer injection -- the magnitude-matched random structure
    # is generated per-tick at the non-motor loop accumulators inside _segregated_loop_arbitrate.

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
    p2_mid = p2_start + (p2_episodes // 2)  # P2 first/second-half split (by episode)
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p2_ticks = 0
    # ----- 707c REPAIR 2: replication telemetry (per arm-seed, emitted to the manifest) -----
    # n_p2_ticks     : RAW P2 env steps (the 707b denominator; exposure, gates nothing)
    # n_fresh_select : P2 ticks on which e3.select() genuinely ran (the HONEST denominator)
    # n_latched      : P2 ticks on which the commitment latch held (nothing recorded)
    # n_dv_ticks     : P2 ticks actually contributing to the primary DV
    #                  (= fresh AND not the random-fallback path)
    # These are in the MANIFEST ITSELF, not a per_tick_sink: Phase 3 cloud workers POST only
    # manifest_bytes, which is twice-confirmed (V3-EXQ-785 and 708) as the reason the
    # contamination was not recoverable by reanalysis. n_fresh_select == n_p2_ticks on any
    # arm-seed would mean the guard is not working (that exact equality was 708's signature).
    n_fresh_select = 0
    n_latched = 0
    n_dv_ticks = 0
    n_p2_fallback_actions = 0
    # 707c REPAIR 6: peak GRADED limbic-loop per-candidate competition over FRESH ticks.
    loop_limbic_pref_range_peak = 0.0
    n_p0_contrastive_steps = 0
    n_p1_bias_updates = 0

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    # PRIMARY DV: committed first-action class counts over P2, split first/second half.
    committed_class_counts: Dict[int, int] = {}
    committed_class_counts_p2a: Dict[int, int] = {}  # P2 first half
    committed_class_counts_p2b: Dict[int, int] = {}  # P2 second half
    n_p2_pre_ge2 = 0
    consumed_dists: List[float] = []
    consumed_dist_max = 0.0

    # CRF maturity readiness (P2).
    crf_n_active_per_tick: List[int] = []
    crf_n_minted_total_last = 0

    # ----- ARC-108 (A1) / MECH-451 (A2) learning diagnostics (accumulated all phases) -----
    lcg_delta_ts: List[float] = []
    lcg_w_chan_range_max = 0.0
    fcg_delta_ts: List[float] = []
    fcg_w_chan_finer_range_max = 0.0
    fcg_w_chan_finer_std_max = 0.0
    n_select_ticks = 0

    # ----- ARC-110 loop-segregation diagnostics (P2 select ticks; the non-degeneracy net) -----
    loop_active_ticks = 0
    loop_noise_active_ticks = 0
    loop_d1d2_active_ticks = 0
    loop_committed_neq_motor_ticks = 0   # a non-motor loop flipped the within-eligible winner
    loop_disagree_ticks = 0              # any loop's within-loop winner != motor winner
    loop_assoc_range_sum = 0.0
    loop_limbic_range_sum = 0.0
    loop_d1d2_conflict_sum = 0.0
    loop_local_credited_sum = 0.0
    loop_local_credited_n = 0
    n_loop_diag_ticks = 0
    # ARC-110 C2 RELEASE (707b) per-named-channel routing diagnostics (peak over P2 ticks).
    loop_named_routing_active_ticks = 0
    loop_limbic_routed_range_peak = 0.0          # peak limbic-loop routed per-candidate range
    loop_named_routed_range_peaks: Dict[str, float] = {}   # per-named-channel peak routed range

    for ep in range(total_train_eps):
        is_p1 = (p1_start <= ep < p2_start)
        is_p2 = (ep >= p2_start)
        is_p2_second_half = (ep >= p2_mid)
        phase_label = "P2" if is_p2 else ("P1" if is_p1 else "P0")

        _, obs_dict = env.reset()
        agent.reset()

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

            # ---------------- 707c REPAIR 1: FRESHNESS GUARD (the 707b defect) -------------
            # `agent.e3.last_score_diagnostics` is populated ONLY inside e3.select(), and it
            # LATCHES: on a tick where select() did not run it still holds the PREVIOUS
            # selection's values. agent.py:5429 returns the held / trajectory-stepped action
            # before the e3.select() call is ever reached whenever not ticks["e3_tick"], at a
            # cadence defaulting to 10 and MECH-093-modulated 5-20. So clearing to None
            # immediately before every select_action and requiring it to REPOPULATE is the
            # only sound freshness test -- and it is the ONLY thing that makes the primary DV
            # an entropy over genuine selections rather than over hold durations.
            #
            # NOT a beta-gate question: `beta_gate.is_elevated` merely chooses step-vs-hold
            # WITHIN an already-skipped tick, so "commitment was effectively disabled for this
            # run" is not a defence and must not be used as one.
            agent.e3.last_score_diagnostics = None

            action = agent.select_action(candidates, ticks)
            n_select_ticks += 1

            _diag_fresh = getattr(agent.e3, "last_score_diagnostics", None)
            is_fresh_select = _diag_fresh is not None
            if is_p2:
                if is_fresh_select:
                    n_fresh_select += 1
                else:
                    # Commitment latch held / non-E3 tick: NO fresh selection. Record NOTHING.
                    n_latched += 1
            # ARC-110: read the segregated-loop diagnostics from the last e3 select (P2 only,
            # when the loop path actually ran). These are the non-degeneracy net: a loop that
            # FLIPPED the within-eligible winner (loop_committed_neq_motor_winner) / disagreed
            # with the motor loop (loop_cross_loop_winner_disagreement) carries live cross-loop
            # variance; a "segregated" loop pinned to the motor winner is a vacuous split.
            # 707c: gated on is_fresh_select. Under 707b this block re-read a LATCHED
            # diagnostics dict on every held tick, so every one of these counters and sums was
            # itself hold-weighted. (With the clear above, `diag` is None on a latched tick and
            # the loop_segregation_active test would already short-circuit -- the explicit
            # freshness conjunct is kept so the invariant is stated, not merely implied.)
            if is_p2 and is_fresh_select:
                diag = _diag_fresh or {}
                if diag.get("loop_segregation_active", False):
                    n_loop_diag_ticks += 1
                    # 707c REPAIR 6: per-fresh-tick GRADED limbic-loop competition. Summed for
                    # a mean and tracked as a peak; both over FRESH selections only. This is
                    # the magnitude readout the C2 non-degeneracy claim now rests on, in place
                    # of the sqrt(2)-pinned structural constant (see the header, repair 6).
                    _lpr = float(diag.get("loop_limbic_pref_range", 0.0) or 0.0)
                    loop_limbic_pref_range_peak = max(loop_limbic_pref_range_peak, _lpr)
                    loop_active_ticks += 1
                    if diag.get("loop_segregation_noise_active", False):
                        loop_noise_active_ticks += 1
                    if diag.get("loop_d1_d2_active", False):
                        loop_d1d2_active_ticks += 1
                    if diag.get("loop_committed_neq_motor_winner", False):
                        loop_committed_neq_motor_ticks += 1
                    if diag.get("loop_cross_loop_winner_disagreement", False):
                        loop_disagree_ticks += 1
                    loop_assoc_range_sum += float(diag.get("loop_assoc_pref_range", 0.0) or 0.0)
                    loop_limbic_range_sum += float(diag.get("loop_limbic_pref_range", 0.0) or 0.0)
                    loop_d1d2_conflict_sum += float(diag.get("loop_d1_d2_conflict_signal", 0.0) or 0.0)
                    _lc = diag.get("loop_local_credited_channels", -1)
                    if _lc is not None and int(_lc) >= 0:
                        loop_local_credited_sum += float(_lc)
                        loop_local_credited_n += 1
                    # ARC-110 C2 RELEASE (707b): per-named-channel routed per-candidate range
                    # (the non-degeneracy gate reads this). Peak over P2 ticks per channel +
                    # the limbic-loop max. A flat (~0) limbic routed range == the MECH-191
                    # phasic gap unfixed -> the gate self-routes substrate_not_ready_requeue.
                    if diag.get("loop_named_channel_routing_active", False):
                        loop_named_routing_active_ticks += 1
                        loop_limbic_routed_range_peak = max(
                            loop_limbic_routed_range_peak,
                            float(diag.get("loop_limbic_routed_max_range", 0.0) or 0.0),
                        )
                        for _nm, _rg in (diag.get("loop_named_channel_routed_ranges", {}) or {}).items():
                            loop_named_routed_range_peaks[_nm] = max(
                                loop_named_routed_range_peaks.get(_nm, 0.0), float(_rg or 0.0)
                            )
            # 707c REPAIR 1 (second inflation path). 707b counted this RANDOM FALLBACK action
            # into the primary DV: `committed_class` is taken from the EXECUTED action one-hot,
            # so a tick where select_action returned None contributed a uniformly-random class
            # to committed_class_counts as though it were a committed selection. It is excluded
            # here and counted separately so the exclusion is auditable rather than invisible.
            action_was_fallback = False
            if action is None:
                action_was_fallback = True
                if is_p2:
                    n_p2_fallback_actions += 1
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
                # n_p2_ticks is the RAW env-step denominator. It is deliberately NOT gated on
                # freshness: it is the exposure quantity (repair 3), reported so the +98% arm
                # spread that drove the 707b distortion stays on the record. It gates nothing.
                n_p2_ticks += 1

            # ------------------- 707c REPAIR 1: THE PRIMARY DV, FRESH-SELECT ONLY -----------
            # THIS is the 707b defect. 707b incremented committed_class_counts inside the plain
            # `if is_p2:` above, on EVERY env step, so each committed class was replicated by
            # its HOLD DURATION and committed_class_entropy_nats became an entropy over a
            # hold-weighted histogram. Because hold duration is CLASS-dependent (_ncl_hold_ticks,
            # beta-gate elevation, and the _committed_trajectory horizon all vary with the
            # committed program) and exposure is ARM-dependent by up to +98%, the distortion
            # does NOT cancel under normalisation -- which is why the 663 matched-replay
            # calibration does not bound it and why no post-hoc division could recover it.
            #
            # A class is now counted once per GENUINE E3 selection, and never on the random
            # fallback path. Everything downstream of committed_class_counts (the p2a/p2b split,
            # n_unique_committed_classes, and the C1/C2 entropies themselves) inherits the
            # corrected denominator by construction.
            if is_p2 and is_fresh_select and not action_was_fallback:
                n_dv_ticks += 1
                committed_class_counts[committed_class] = (
                    committed_class_counts.get(committed_class, 0) + 1
                )
                if is_p2_second_half:
                    committed_class_counts_p2b[committed_class] = (
                        committed_class_counts_p2b.get(committed_class, 0) + 1
                    )
                else:
                    committed_class_counts_p2a[committed_class] = (
                        committed_class_counts_p2a.get(committed_class, 0) + 1
                    )
                if len(pre_e3_classes) >= 2:
                    n_p2_pre_ge2 += 1

                # 707c REPAIR 5 (the admission gate's OTHER contaminated end). The divergence
                # statistic consumed_summary_pairwise_dist_mean gates enough_divergent_seeds,
                # and 707b accumulated it on every P2 env step from agent.py:4812 -- which
                # returns CACHED candidates on a non-E3 tick. So the seed-47-misses-by-1.3%
                # near-miss that gave the design zero headroom was itself measured on the same
                # defect family as the primary DV. It is now fresh-select-only too.
                if candidates and len(candidates) >= 2:
                    consumed = _consumed_summaries(agent, candidates)
                    if consumed is not None and torch.isfinite(consumed).all():
                        d = _mean_pairwise_l2(consumed)
                        if math.isfinite(d):
                            consumed_dists.append(d)
                            consumed_dist_max = max(consumed_dist_max, d)

            if is_p2:

                crf = getattr(agent, "candidate_rule_field", None)
                if crf is not None:
                    st = crf.get_state()
                    crf_n_active_per_tick.append(int(st.get("crf_n_active_last", 0)))
                    crf_n_minted_total_last = int(st.get("crf_n_minted_total", 0))
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
            # update_residue drives e3.post_action_update -> the ARC-108 (A1 w_chan) /
            # MECH-451 (A2 w_chan_finer) three-factor learning fires here on EVERY waking
            # tick (all phases). On ARM_NOISE learned_channel_gating_eta=0.0 -> w_chan_finer
            # unchanged (frozen at the injected random vector).
            with torch.no_grad():
                resid_metrics = agent.update_residue(
                    harm_signal=float(_harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )
            # Capture the learning diagnostics surfaced by post_action_update (e3_ prefix).
            ldt = resid_metrics.get("e3_lcg_delta_t")
            if ldt is not None:
                lcg_delta_ts.append(float(ldt.item()))
            lwr = resid_metrics.get("e3_lcg_w_chan_range")
            if lwr is not None:
                lcg_w_chan_range_max = max(lcg_w_chan_range_max, float(lwr.item()))
            fdt = resid_metrics.get("e3_fcg_delta_t")
            if fdt is not None:
                fcg_delta_ts.append(float(fdt.item()))
            fwr = resid_metrics.get("e3_fcg_w_chan_finer_range")
            if fwr is not None:
                fcg_w_chan_finer_range_max = max(
                    fcg_w_chan_finer_range_max, float(fwr.item())
                )
            fws = resid_metrics.get("e3_fcg_w_chan_finer_std")
            if fws is not None:
                fcg_w_chan_finer_std_max = max(
                    fcg_w_chan_finer_std_max, float(fws.item())
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
    committed_class_entropy = _entropy_from_int_counts(committed_class_counts)
    committed_class_entropy_p2a = _entropy_from_int_counts(committed_class_counts_p2a)
    committed_class_entropy_p2b = _entropy_from_int_counts(committed_class_counts_p2b)

    # 707c: DENOMINATOR MATCHED TO THE NUMERATOR. n_p2_pre_ge2 is now incremented only on DV
    # ticks (fresh, non-fallback), so dividing by the RAW n_p2_ticks -- as 707b did -- would
    # deflate this ratio by the hold factor (~10x) and make the saturated 1.0 that 707b
    # legitimately observed look like ~0.1. Both ends must move together.
    frac_pre_ge2 = float(n_p2_pre_ge2 / n_dv_ticks) if n_dv_ticks > 0 else 0.0
    consumed_spread_mean = (
        float(sum(consumed_dists) / len(consumed_dists)) if consumed_dists else 0.0
    )

    if crf_n_active_per_tick:
        frac_crf_active_ge_floor = float(
            sum(1 for n in crf_n_active_per_tick if n >= CRF_N_ACTIVE_FLOOR)
            / len(crf_n_active_per_tick)
        )
    else:
        frac_crf_active_ge_floor = 0.0
    crf_differentiated = bool(
        crf_n_minted_total_last >= CRF_MIN_MINTED
        and frac_crf_active_ge_floor >= CRF_FRAC_ACTIVE_FLOOR
    )

    # Learning non-vacuity per-seed.
    lcg_delta_t_std = float(statistics.pstdev(lcg_delta_ts)) if len(lcg_delta_ts) >= 2 else 0.0
    fcg_delta_t_std = float(statistics.pstdev(fcg_delta_ts)) if len(fcg_delta_ts) >= 2 else 0.0

    seed_class_axis_exercisable = bool(frac_pre_ge2 > FRAC_PRE_GE2_FLOOR)
    seed_gapa_divergence = bool(
        consumed_spread_mean > CONSUMED_SPREAD_FLOOR
        and consumed_dist_max < CONSUMED_MAGNITUDE_CEIL
    )

    # Per-arm learning-engaged non-vacuity (only meaningful on the armed arms).
    lcg_moved = bool(lcg_w_chan_range_max > W_CHAN_RANGE_FLOOR)
    lcg_delta_nonflat = bool(lcg_delta_t_std > DELTA_T_STD_FLOOR)
    # MECH-451 gate (1): finer channels MOVED + DISSOCIABLE (range above floor).
    fcg_moved = bool(fcg_w_chan_finer_range_max > W_CHAN_FINER_RANGE_FLOOR)
    fcg_delta_nonflat = bool(fcg_delta_t_std > DELTA_T_STD_FLOOR)

    # ----- ARC-110 loop-segregation per-seed aggregation (over P2 select ticks) -----
    loop_n = max(n_loop_diag_ticks, 1)
    loop_frac_committed_neq_motor = float(loop_committed_neq_motor_ticks / loop_n) if n_loop_diag_ticks else 0.0
    loop_frac_disagree = float(loop_disagree_ticks / loop_n) if n_loop_diag_ticks else 0.0
    loop_assoc_range_mean = float(loop_assoc_range_sum / loop_n) if n_loop_diag_ticks else 0.0
    loop_limbic_range_mean = float(loop_limbic_range_sum / loop_n) if n_loop_diag_ticks else 0.0
    loop_d1d2_conflict_mean = float(loop_d1d2_conflict_sum / loop_n) if n_loop_diag_ticks else 0.0
    loop_local_credited_mean = (
        float(loop_local_credited_sum / loop_local_credited_n) if loop_local_credited_n else -1.0
    )
    # Live cross-loop variance: a non-motor loop FLIPPED the commit OR loops disagreed on a
    # non-trivial fraction of ticks, AND at least one non-motor loop carries pref range > 0.
    seed_loop_cross_variance = bool(
        n_loop_diag_ticks > 0
        and (loop_frac_committed_neq_motor > LOOP_CROSS_VARIANCE_FRAC_FLOOR
             or loop_frac_disagree > LOOP_CROSS_VARIANCE_FRAC_FLOOR)
        and (loop_assoc_range_mean > LOOP_PREF_RANGE_FLOOR
             or loop_limbic_range_mean > LOOP_PREF_RANGE_FLOOR)
    )
    # ARC-110 C2 RELEASE (707b) per-seed non-degeneracy: at least one LIMBIC channel reached
    # the arbitration carrying a routed per-candidate range above the substantive floor (peak
    # over P2 ticks). This is the C2-specific precondition the 707 vacuous DROP==A1 lacked.
    limbic_routed_peaks = [
        loop_named_routed_range_peaks.get(nm, 0.0) for nm in LIMBIC_NAMED_CHANNELS
    ]
    loop_limbic_routed_range_max = float(max(limbic_routed_peaks)) if limbic_routed_peaks else 0.0
    seed_named_channel_routing_live = bool(
        loop_named_routing_active_ticks > 0
        and loop_limbic_routed_range_max > LIMBIC_ROUTED_RANGE_FLOOR
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "finer_on": bool(arm["finer_on"]),
        "loop_seg": bool(arm.get("loop_seg", False)),
        "loop_noise": bool(arm.get("loop_noise", False)),
        "drop_limbic": bool(arm.get("drop_limbic", False)),
        # ----- ARC-110 loop diagnostics -----
        "loop_active_ticks": int(loop_active_ticks),
        "loop_noise_active_ticks": int(loop_noise_active_ticks),
        "loop_d1d2_active_ticks": int(loop_d1d2_active_ticks),
        "loop_frac_committed_neq_motor": round(loop_frac_committed_neq_motor, 6),
        "loop_frac_disagree": round(loop_frac_disagree, 6),
        "loop_assoc_pref_range": round(loop_assoc_range_mean, 6),
        "loop_limbic_pref_range": round(loop_limbic_range_mean, 6),
        "loop_d1d2_conflict_signal": round(loop_d1d2_conflict_mean, 6),
        "loop_local_credited_channels_mean": round(loop_local_credited_mean, 4),
        "loop_cross_variance": seed_loop_cross_variance,
        # ----- ARC-110 C2 RELEASE (707b) per-named-channel routing diagnostics -----
        "loop_named_routing_active_ticks": int(loop_named_routing_active_ticks),
        "loop_limbic_routed_range_max": round(loop_limbic_routed_range_max, 6),
        "loop_named_routed_range_peaks": {
            str(k): round(float(v), 6) for k, v in sorted(loop_named_routed_range_peaks.items())
        },
        "named_channel_routing_live": seed_named_channel_routing_live,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p2_ticks": int(n_p2_ticks),
        # ----- 707c REPAIR 2: replication telemetry -----
        "n_fresh_select": int(n_fresh_select),
        "n_latched": int(n_latched),
        "n_dv_ticks": int(n_dv_ticks),
        "n_p2_fallback_actions": int(n_p2_fallback_actions),
        "fresh_select_yield": round(
            float(n_fresh_select / n_p2_ticks) if n_p2_ticks > 0 else 0.0, 6
        ),
        # 707c REPAIR 6: GRADED limbic-loop per-candidate competition (peak over fresh ticks).
        # The mean is emitted above as loop_limbic_pref_range; the peak is the non-degeneracy
        # readout (a loop that ever carried real competition is live; a mean can be dragged
        # down by ticks where the eligible set collapsed to a single candidate).
        "loop_limbic_pref_range_peak": round(float(loop_limbic_pref_range_peak), 6),
        "n_p0_contrastive_steps": int(n_p0_contrastive_steps),
        "n_p1_bias_updates": int(n_p1_bias_updates),
        "error_note": error_note,
        # ----- PRIMARY DV (committed-class entropy) -----
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "committed_class_entropy_p2_first_half_nats": round(committed_class_entropy_p2a, 6),
        "committed_class_entropy_p2_second_half_nats": round(committed_class_entropy_p2b, 6),
        "n_unique_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": {
            str(k): int(v) for k, v in sorted(committed_class_counts.items())
        },
        # ----- Readiness / non-vacuity -----
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "class_axis_exercisable": seed_class_axis_exercisable,
        "consumed_summary_pairwise_dist_mean": round(consumed_spread_mean, 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        "gapa_divergence": seed_gapa_divergence,
        "crf_frac_active_ge_floor": round(frac_crf_active_ge_floor, 6),
        "crf_n_minted_total": int(crf_n_minted_total_last),
        "crf_differentiated": crf_differentiated,
        # ----- A1 global-w_chan learning diagnostics -----
        "lcg_n_updates": int(len(lcg_delta_ts)),
        "lcg_delta_t_std": round(lcg_delta_t_std, 8),
        "lcg_w_chan_range_max": round(lcg_w_chan_range_max, 8),
        "lcg_moved": lcg_moved,
        "lcg_delta_nonflat": lcg_delta_nonflat,
        # ----- A2 / ARM_NOISE finer-channel learning diagnostics (MECH-451) -----
        "fcg_n_updates": int(len(fcg_delta_ts)),
        "fcg_delta_t_std": round(fcg_delta_t_std, 8),
        "fcg_w_chan_finer_range_max": round(fcg_w_chan_finer_range_max, 8),
        "fcg_w_chan_finer_std_max": round(fcg_w_chan_finer_std_max, 8),
        "fcg_moved": fcg_moved,
        "fcg_delta_nonflat": fcg_delta_nonflat,
    }


def _arm_rows(arm_results: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [
        r for r in arm_results
        if r["arm_id"] == arm_id and r["error_note"] is None
    ]


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _median(vals: List[float]) -> float:
    return float(statistics.median(vals)) if vals else 0.0


def _by_seed(rows: List[Dict[str, Any]], key: str) -> Dict[int, float]:
    return {int(r["seed"]): float(r[key]) for r in rows}


def _gap_by_seed(rows: List[Dict[str, Any]]) -> Dict[int, bool]:
    return {int(r["seed"]): bool(r["gapa_divergence"]) for r in rows}


def _div_pass(n_ok: int, n_div: int) -> bool:
    """A criterion PASSES on the divergent seeds iff there are >= MIN_DIVERGENT_SEEDS
    divergent seeds AND the criterion holds on a strict-majority-ish fraction of them
    (>= ceil(DIVERGENT_PASS_FRACTION * n_div)), with an absolute floor of
    MIN_SEEDS_FOR_PASS divergent seeds."""
    if n_div < MIN_DIVERGENT_SEEDS:
        return False
    needed = max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * n_div)))
    return n_ok >= needed


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
    n_reuse_hits = 0

    for arm in ARMS:
        print(
            f"Arm {arm['arm_id']} ({arm['label']}) loop_seg={arm.get('loop_seg', False)} "
            f"loop_noise={arm.get('loop_noise', False)} drop_limbic={arm.get('drop_limbic', False)} "
            f"finer_on={arm['finer_on']} "
            f"(P0={p0_episodes} ep e2-train, P1={p1_episodes} ep bias-train, "
            f"P2={p2_episodes} ep measure, steps_per_episode={steps_per_episode}, "
            f"dry_run={dry_run})",
            flush=True,
        )
        reusable = arm["arm_id"] in REUSABLE_ARM_IDS_LOCAL
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)

            # ----- ARM-REUSE (consumer side), gated + safe-by-default -----
            # Only A0_SINGLE_ARENA is reuse-eligible, and only IFF a mint is cited. With
            # REUSE_BASELINE_FROM=None (the default) this is skipped; every arm runs fresh.
            row: Optional[Dict[str, Any]] = None
            if REUSE_BASELINE_FROM is not None and reusable:
                cell = try_reuse_cell(
                    config_slice=_arm_config_slice(
                        arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                    ),
                    seed=s,
                    script_path=script_path,
                    needed_keys=REUSE_NEEDED_KEYS,
                    cite_run_id=REUSE_BASELINE_FROM,
                    include_driver_script_in_hash=False,
                )
                if cell is not None:
                    row = dict(cell)  # reuse hit; provenance stamped by try_reuse_cell
                    n_reuse_hits += 1

            if row is None:
                row = _run_seed_arm(
                    arm, s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode,
                )

            # Per-cell fingerprint. A0_SINGLE_ARENA emits a REUSE-ELIGIBLE fingerprint
            # (MINT-AS-YOU-GO: config slice declared, include_driver_script_in_hash=False).
            # The loop arms ride the just-built loop substrate (in flux) -- not reusable.
            if reusable:
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(
                        arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                    ),
                    seed=s,
                    script_path=script_path,
                    rng_fully_reset=True,
                    config_slice_declared=True,
                    include_driver_script_in_hash=False,
                )
            else:
                row["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=_arm_config_slice(
                        arm, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
                    ),
                    seed=s,
                    script_path=script_path,
                    rng_fully_reset=True,
                    extra_ineligible_reasons=[
                        "arc110_loop_substrate_just_built_in_flux_not_a_reusable_baseline",
                    ],
                )
            arm_results.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)

    a0_rows = _arm_rows(arm_results, "A0_SINGLE_ARENA")
    a1_rows = _arm_rows(arm_results, "A1_LOOPS")
    noise_rows = _arm_rows(arm_results, "ARM_NOISE")
    drop_rows = _arm_rows(arm_results, "ARM_DROP_LIMBIC")
    all_rows = a0_rows + a1_rows + noise_rows + drop_rows

    def _maj(rows: List[Dict[str, Any]], pred) -> bool:
        return sum(1 for r in rows if pred(r)) >= MIN_SEEDS_FOR_PASS

    a0_ent = _by_seed(a0_rows, "committed_class_entropy_nats")
    a1_ent = _by_seed(a1_rows, "committed_class_entropy_nats")
    noise_ent = _by_seed(noise_rows, "committed_class_entropy_nats")
    drop_ent = _by_seed(drop_rows, "committed_class_entropy_nats")

    a0_gap = _gap_by_seed(a0_rows)
    a1_gap = _gap_by_seed(a1_rows)
    noise_gap = _gap_by_seed(noise_rows)
    drop_gap = _gap_by_seed(drop_rows)

    # ----- Per-seed-divergent gating: seeds whose pool is divergent on the C1 comparison
    # arms (A0 + A1 + ARM_NOISE all divergent). A degenerate cell self-excludes. -----
    primary_div = [
        s for s in sorted(set(a0_gap) & set(a1_gap) & set(noise_gap))
        if a0_gap.get(s) and a1_gap.get(s) and noise_gap.get(s)
    ]
    n_primary_div = len(primary_div)
    enough_divergent = n_primary_div >= MIN_DIVERGENT_SEEDS

    # ----- Precondition (ARC-110 non-degeneracy): loops carry LIVE cross-loop variance on
    # A1 (a non-motor loop flipped the within-eligible winner / loops disagreed, AND a
    # non-motor loop carries pref range > 0) on a majority of seeds. A "segregated" loop
    # pinned to the motor winner is a vacuous split -> requeue. -----
    loop_cross_variance_ok = _maj(a1_rows, lambda r: r.get("loop_cross_variance", False))
    # The COUNT behind the _maj above. Reported by the precondition entry so the
    # indexer's authoritative recompute can reproduce `met`; see that entry for why
    # the min-fraction statistic it used to report could not.
    n_a1_loop_cross_variance = sum(1 for r in a1_rows if r.get("loop_cross_variance", False))
    a1_loop_flip_min = float(min([r.get("loop_frac_committed_neq_motor", 0.0) for r in a1_rows] or [0.0]))
    a1_loop_disagree_min = float(min([r.get("loop_frac_disagree", 0.0) for r in a1_rows] or [0.0]))

    # ----- Precondition (707a FIX): the S2 in-layer null (ARM_NOISE) is a LIVE same-layer
    # perturbation -- on a strict-majority of divergent seeds it actively perturbs the
    # committed selection (loop_noise_active_ticks > 0 AND loop_frac_committed_neq_motor or
    # loop_frac_disagree > the cross-loop-variance floor). THIS is the same-layer null the
    # single arena could not construct (the 704b/706b binding constraint); if even the in-layer
    # null is INERT (does not reach the committed-class readout) the null-validity problem
    # persists -> requeue, NOT a weakens.
    #
    # WHY null-LIVENESS, not null-LIFT (the 707->707a fix): 707 required the null to LIFT
    # committed-class entropy strict-ABOVE A0 (n_noise_lifts = noise_ent > a0_ent + margin).
    # That is the 700-lineage TEMPERATURE-null liveness proxy (raising softmax temperature
    # trivially raises entropy) and is INVALID for a structured-accumulator null: random
    # magnitude-matched content routed through the SAME loop path (per-loop zscore + Haber
    # spiral, motor never nulled) lands committed-class entropy near BASELINE (structured loops
    # beat random; random ~ A0), NOT above it. In 707 the null was demonstrably live (it
    # perturbed selection on 4/4 divergent seeds) yet sat at/below A0 on 3/4, so the run was
    # spuriously requeued and the pre-registered WEAKENS branch was unreachable. The non-vacuity
    # the null actually has to provide is that "A1 strict-above the null" (C1) is a real test --
    # i.e. the null is a LIVE magnitude-matched alternative, not a dead control. The path-
    # artifact hypothesis (A1's lift is a zscore-path artefact) is ruled out by C1 itself (a
    # live null on the SAME zscore path would match A1 and fail C1), NOT by null-lift.
    # n_noise_lifts is still RECORDED for continuity but no longer gates.
    noise_by_seed = {int(r["seed"]): r for r in noise_rows}

    def _noise_live(s: int) -> bool:
        r = noise_by_seed.get(s)
        if not r:
            return False
        return bool(
            int(r.get("loop_noise_active_ticks", 0)) > 0
            and (
                float(r.get("loop_frac_committed_neq_motor", 0.0)) > LOOP_CROSS_VARIANCE_FRAC_FLOOR
                or float(r.get("loop_frac_disagree", 0.0)) > LOOP_CROSS_VARIANCE_FRAC_FLOOR
            )
        )

    n_noise_live = sum(1 for s in primary_div if _noise_live(s))
    n_noise_lifts = sum(
        1 for s in primary_div if noise_ent.get(s, 0.0) > a0_ent.get(s, 0.0) + CONVERSION_MARGIN
    )  # recorded for continuity with 707; NOT a gate in 707a
    noise_verified_lifting = bool(enough_divergent and _div_pass(n_noise_live, n_primary_div))

    # ----- Precondition (ARC-110 C2 RELEASE, 707b): per-NAMED-channel routing is LIVE on A1.
    # At least one LIMBIC channel (ofc/liking/vigour) reached the arbitration carrying a routed
    # per-candidate range above the substantive floor, on a strict-majority of DIVERGENT seeds.
    # This is evaluated BEFORE C2 is scored (it is part of preconditions_met, and C2 is only
    # read in the c1_holds PASS branch). If the named limbic channels are still per-candidate-
    # FLAT (MECH-191 phasic gap unfixed), the limbic loop carries no competition, ARM_DROP_LIMBIC
    # is a no-op (the 707 vacuous DROP==A1), and the run self-routes substrate_not_ready_requeue
    # -- NEVER a false weakens. -----
    named_routing_live_div = [
        s for s in primary_div
        if next((r for r in a1_rows if int(r["seed"]) == s), {}).get("named_channel_routing_live", False)
    ]
    n_named_routing_live = len(named_routing_live_div)
    named_channel_routing_live = bool(enough_divergent and _div_pass(n_named_routing_live, n_primary_div))
    a1_limbic_routed_range_max = float(
        max([r.get("loop_limbic_routed_range_max", 0.0) for r in a1_rows] or [0.0])
    )

    # ----- Precondition: learning engaged on A1 (finer channels dissociable + delta_t nonflat) -----
    fcg_moved_ok = _maj(a1_rows, lambda r: r.get("fcg_moved", False))
    fcg_delta_nonflat_ok = _maj(a1_rows, lambda r: r.get("fcg_delta_nonflat", False))
    # Per-leg COUNTS behind the _maj calls above. The two legs are declared as SEPARATE
    # recomputable preconditions because the single entry that used to carry both had
    # `met = fcg_moved_ok and fcg_delta_nonflat_ok` -- an AND of two independent counts,
    # which NO single (measured, threshold) pair can reproduce. Their conjunction is the
    # same predicate, and the label routing still reads the two booleans directly.
    n_fcg_moved = sum(1 for r in a1_rows if r.get("fcg_moved", False))
    n_fcg_delta_nonflat = sum(1 for r in a1_rows if r.get("fcg_delta_nonflat", False))

    # CRF maturity (matched constant; majority of seeds on all arms).
    crf_matured = all(
        _maj(rows, lambda r: r["crf_differentiated"]) for rows in
        (a0_rows, a1_rows, noise_rows, drop_rows)
    )
    # The per-arm COUNTS behind the _maj calls above, reported by the crf_matured
    # precondition entry so the indexer's authoritative recompute can reproduce `met`.
    # `crf_matured` is an all()/AND over per-arm k-of-n seed counts and _maj is
    # `count >= MIN_SEEDS_FOR_PASS`, so min(per-arm count) >= MIN_SEEDS_FOR_PASS is
    # EXACT: min(counts) >= k iff every count >= k. NOT split into one entry per arm --
    # a k-of-n COUNT does not distribute over the conjunction the way all() does (two
    # arms each cleared by k DIFFERENT seeds is not the conjunction), so a per-leg split
    # would be strictly LOOSER than the shipped gate.
    n_crf_differentiated_per_arm = [
        sum(1 for r in rows if r["crf_differentiated"])
        for rows in (a0_rows, a1_rows, noise_rows, drop_rows)
    ]
    n_crf_differentiated_min = int(min(n_crf_differentiated_per_arm))

    # ================= 707c REPAIR 4: fresh_selects_sufficient (readiness) =================
    # An honest-but-underpowered DV must self-route substrate_not_ready_requeue, NEVER a false
    # weakens -- which is exactly the branch 707b took on an instrument that could not resolve
    # it. `measured` is the WORST cell (min over arm-seeds) to match the all(...) quantifier
    # the `met` expression uses, so the indexer's authoritative recompute agrees with the flag
    # set here rather than adjudicating a mean that could mask a single starved cell.
    _fresh_counts = [
        (int(r.get("n_fresh_select", 0)), str(r.get("arm_id", "?")), int(r.get("seed", -1)))
        for r in all_rows
    ]
    _worst_fresh = min(_fresh_counts) if _fresh_counts else (0, "none", -1)
    fresh_selects_sufficient = bool(
        _fresh_counts and all(n >= MIN_FRESH_SELECTS for n, _a, _s in _fresh_counts)
    )

    # ---- REPAIR 3: exposure imbalance vs A0 -- REPORTED, NEVER GATING ----
    # This does not appear in preconditions_met and must not: it is the record of the +98%
    # arm-exposure spread that made 707b's hold-weighting directional rather than merely noisy.
    _a0_fresh_by_seed = {int(r["seed"]): int(r.get("n_fresh_select", 0)) for r in a0_rows}
    exposure_imbalance: List[Dict[str, Any]] = []
    for r in all_rows:
        _base = _a0_fresh_by_seed.get(int(r.get("seed", -1)), 0)
        _nf = int(r.get("n_fresh_select", 0))
        _frac = float((_nf - _base) / _base) if _base > 0 else 0.0
        exposure_imbalance.append({
            "arm_id": r.get("arm_id"),
            "seed": int(r.get("seed", -1)),
            "n_p2_ticks": int(r.get("n_p2_ticks", 0)),
            "n_fresh_select": _nf,
            "n_latched": int(r.get("n_latched", 0)),
            "n_dv_ticks": int(r.get("n_dv_ticks", 0)),
            "fresh_select_yield": r.get("fresh_select_yield", 0.0),
            "a0_n_fresh_select": _base,
            "exposure_imbalance_vs_a0": round(_frac, 6),
            "beyond_report_floor": bool(abs(_frac) > EXPOSURE_IMBALANCE_REPORT_FLOOR),
        })
    max_exposure_imbalance = max(
        [abs(float(e["exposure_imbalance_vs_a0"])) for e in exposure_imbalance] or [0.0]
    )

    # ============ 707c REPAIR 5: divergence HEADROOM (autopsy sec 3b) ============
    # 707b cleared enough_divergent_seeds at EXACTLY 3.0/3.0. `enough_divergent` above already
    # tests against the raised MIN_DIVERGENT_SEEDS=4, so this block's job is to make the
    # MARGIN visible: the per-seed distance from CONSUMED_SPREAD_FLOOR, so a 1.3%-style
    # near-miss is on the record instead of silently deciding the run. Reported for EVERY seed,
    # including the ones that cleared -- a diagnostic that appears only when something already
    # looks wrong cannot establish that anything was ever right.
    _spread_by_seed: Dict[int, Dict[str, float]] = {}
    for r in all_rows:
        _s = int(r.get("seed", -1))
        _spread_by_seed.setdefault(_s, {})[str(r.get("arm_id"))] = float(
            r.get("consumed_summary_pairwise_dist_mean", 0.0) or 0.0
        )
    divergence_margins: List[Dict[str, Any]] = []
    for _s in sorted(_spread_by_seed):
        _arms = _spread_by_seed[_s]
        # The C1 admission set is A0 + A1 + ARM_NOISE (DROP is gated separately inside C2).
        _c1_arms = {k: v for k, v in _arms.items()
                    if k in ("A0_SINGLE_ARENA", "A1_LOOPS", "ARM_NOISE")}
        _binding_arm, _binding_val = (
            min(_c1_arms.items(), key=lambda kv: kv[1]) if _c1_arms else ("none", 0.0)
        )
        divergence_margins.append({
            "seed": _s,
            "binding_arm": _binding_arm,          # the arm that decides this seed's divergence
            "binding_consumed_spread": round(_binding_val, 8),
            "floor": float(CONSUMED_SPREAD_FLOOR),
            "margin_to_floor": round(_binding_val - float(CONSUMED_SPREAD_FLOOR), 8),
            "margin_frac_of_floor": round(
                (_binding_val - float(CONSUMED_SPREAD_FLOOR)) / float(CONSUMED_SPREAD_FLOOR), 6
            ),
            "divergent": bool(_s in primary_div),
        })
    divergence_headroom = int(n_primary_div - MIN_DIVERGENT_SEEDS)
    # The RUN's seed count, not the module-level SEEDS -- same class of bug as the
    # write_flat_manifest(seeds=SEEDS) slip 707b shipped: a --dry-run manifest would
    # otherwise report 8 seeds attempted while having executed 1.
    n_seeds_attempted = len({int(r.get("seed", -1)) for r in all_rows})

    # ============ 707c REPAIR 6: GRADED limbic-loop competition (autopsy sec 3a) ============
    # Replaces the sqrt(2)-pinned structural constant as the C2 non-degeneracy basis. Gated on
    # A1_LOOPS (the arm whose limbic loop must be live for a limbic ABLATION to mean anything)
    # over the divergent seeds, using the same strict-majority rule as the other gates.
    _a1_by_seed = {int(r["seed"]): r for r in a1_rows}

    def _limbic_graded_live(s: int) -> bool:
        r = _a1_by_seed.get(s)
        if not r:
            return False
        return bool(
            float(r.get("loop_limbic_pref_range_peak", 0.0) or 0.0)
            > LIMBIC_GRADED_PREF_RANGE_FLOOR
        )

    n_limbic_graded_live = sum(1 for s in primary_div if _limbic_graded_live(s))
    limbic_graded_competition_live = bool(
        enough_divergent and _div_pass(n_limbic_graded_live, n_primary_div)
    )
    # Worst A1 cell over the divergent seeds -- the recomputable measured value for the
    # precondition entry, and the offending cell when it fails.
    _a1_graded = [
        (float(_a1_by_seed[s].get("loop_limbic_pref_range_peak", 0.0) or 0.0), s)
        for s in primary_div if s in _a1_by_seed
    ]
    _worst_a1_graded = min(_a1_graded) if _a1_graded else (0.0, -1)
    # Non-gating C2 MECHANISM diagnostic: does the limbic ablation actually move the graded
    # quantity, and do the remapped channels show up in the associative loop? This is the
    # dissociation 707b's binary gate could not see (s43: A1 1.0931 -> DROP 0.0133 while
    # assoc ROSE). Reported on PASS runs too.
    _drop_by_seed = {int(r["seed"]): r for r in drop_rows}
    limbic_ablation_effect: List[Dict[str, Any]] = []
    for s in primary_div:
        _a1r, _dr = _a1_by_seed.get(s), _drop_by_seed.get(s)
        if not _a1r or not _dr:
            continue
        _a1l = float(_a1r.get("loop_limbic_pref_range", 0.0) or 0.0)
        _drl = float(_dr.get("loop_limbic_pref_range", 0.0) or 0.0)
        limbic_ablation_effect.append({
            "seed": s,
            "a1_limbic_pref_range": round(_a1l, 6),
            "drop_limbic_pref_range": round(_drl, 6),
            "limbic_loss_frac": round((_a1l - _drl) / _a1l, 6) if _a1l > 0 else 0.0,
            "a1_assoc_pref_range": round(float(_a1r.get("loop_assoc_pref_range", 0.0) or 0.0), 6),
            "drop_assoc_pref_range": round(float(_dr.get("loop_assoc_pref_range", 0.0) or 0.0), 6),
        })

    preconditions_met = bool(
        enough_divergent
        and loop_cross_variance_ok
        and named_channel_routing_live    # ARC-110 C2 RELEASE (707b): routing LIVE (binary, sqrt(2) -- liveness only)
        and limbic_graded_competition_live  # 707c REPAIR 6: routing carries GRADED magnitude
        and fresh_selects_sufficient      # 707c REPAIR 4: the DV has an honest denominator
        and noise_verified_lifting
        and fcg_moved_ok and fcg_delta_nonflat_ok
        and crf_matured
    )

    # ----- C1 (loop-segregated conversion): A1_LOOPS committed-class entropy strict-above
    # BOTH A0_SINGLE_ARENA AND the VALID in-layer null ARM_NOISE, on a strict-majority of
    # divergent seeds. -----
    c1_seeds: List[int] = []
    for s in primary_div:
        bar = max(a0_ent.get(s, 0.0), noise_ent.get(s, 0.0)) + CONVERSION_MARGIN
        if a1_ent.get(s, 0.0) > bar:
            c1_seeds.append(s)
    n_c1 = len(c1_seeds)
    c1_holds = _div_pass(n_c1, n_primary_div)

    # ----- C2 (ARC-106 per-loop load-bearing): A1_LOOPS strict-above ARM_DROP_LIMBIC on a
    # strict-majority of divergent seeds where DROP is divergent. If dropping the limbic loop
    # does NOT remove the lift, the limbic loop is DECORATIVE. -----
    c2_div = [s for s in primary_div if drop_gap.get(s)]
    c2_seeds = [
        s for s in c2_div
        if a1_ent.get(s, 0.0) > drop_ent.get(s, 0.0) + CONVERSION_MARGIN
    ]
    c2_holds = bool(c1_holds and len(c2_div) >= MIN_SEEDS_FOR_PASS and _div_pass(len(c2_seeds), len(c2_div)))

    # ----- C2 NON-DEGENERACY DIAGNOSTIC (707a, additive; does NOT gate any verdict). 707
    # showed ARM_DROP_LIMBIC byte-identical to A1_LOOPS on every seed (same committed-class
    # entropy + same flip/disagree fractions), i.e. the limbic-loop ablation produced ZERO
    # change in the committed readout, so C2 (limbic load-bearing) was untestable -- it could
    # only ever read passed=false for a degenerate reason, not because the limbic loop is
    # decorative. Record per-seed whether DROP actually DIFFERS from A1 so a future read /
    # autopsy can tell "limbic decorative" (DROP differs, A1 not above it) from "ablation
    # inert / untestable" (DROP == A1). C2 is load_bearing=false and this flag changes NO
    # outcome; it only annotates the manifest. -----
    DROP_DIFF_EPS = 1e-6
    c2_drop_differs_seeds = [
        s for s in primary_div
        if abs(a1_ent.get(s, 0.0) - drop_ent.get(s, 0.0)) > DROP_DIFF_EPS
    ]
    c2_drop_differs_from_a1 = bool(c2_drop_differs_seeds)

    # ----- Outcome map (decisive either way) -----
    if not preconditions_met:
        outcome = "FAIL"
        overall_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "ARC-110 loop-segregated conversion could NOT be validly measured: a precondition "
            "is unmet (too few divergent seeds / loops carry NO live cross-loop variance = a "
            "vacuous split pinned to the motor winner / NAMED limbic channels carry NO routed "
            "per-candidate range = the MECH-191 phasic gap unfixed, so DROP_LIMBIC is a no-op "
            "(the 707 vacuous DROP==A1) / the in-layer null is INERT = does not perturb the "
            "committed selection / finer channels not dissociable / delta_t flat / 707c: the "
            "limbic loop carried no GRADED per-candidate competition on A1, so a limbic "
            "ablation is vacuous / 707c: an arm-seed recorded fewer than MIN_FRESH_SELECTS "
            "genuine E3 selections, so the DV has no honest denominator). NOT a falsification. "
            "This is the branch 707b should have taken and did not: its instrument could not "
            "resolve the contrast it reported, and it returned a weakens instead."
        )
        per_claim = {"ARC-110": "non_contributory"}
    elif c1_holds:
        outcome = "PASS"
        overall_direction = "supports"
        non_degenerate = True
        degeneracy_reason = ""
        if c2_holds:
            label = "loop_segregation_converts_limbic_loop_load_bearing_supports_arc110"
        else:
            label = "loop_segregation_converts_limbic_not_sole_driver_supports_arc110"
        per_claim = {"ARC-110": "supports"}
    else:
        outcome = "FAIL"
        overall_direction = "weakens"
        non_degenerate = True
        degeneracy_reason = ""
        label = "valid_null_loops_vary_but_no_conversion_ceiling_intrinsic_weakens_arc110"
        per_claim = {"ARC-110": "weakens"}

    a0_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a0_rows])
    a1_mean_dv = _mean([r["committed_class_entropy_nats"] for r in a1_rows])
    noise_mean_dv = _mean([r["committed_class_entropy_nats"] for r in noise_rows])
    drop_mean_dv = _mean([r["committed_class_entropy_nats"] for r in drop_rows])

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "crf_matured",
                "kind": "readiness",
                "description": (
                    "CRF maturity: the consumed matched constant must be DIFFERENTIATED "
                    "(per-seed crf_differentiated) on at least MIN_SEEDS_FOR_PASS seeds within "
                    "EVERY arm -- an undifferentiated constant makes the manipulation vacuous "
                    "=> substrate_not_ready_requeue. measured = the SMALLEST per-arm count of "
                    "seeds carrying crf_differentiated, over arms (a0_rows, a1_rows, noise_rows, drop_rows)."
                ),
                "control": "per-seed crf_differentiated, counted within each arm",
                # COUNT-shaped, INCLUSIVE floor, and EXACT for the shipped predicate:
                # `met` is all(_maj(rows, crf_differentiated) for rows in arms) with _maj ==
                # `count >= MIN_SEEDS_FOR_PASS`, and min(counts) >= k iff every count >= k.
                "measured": float(n_crf_differentiated_min),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                # Non-bound observable (inert to the recompute): the full per-arm counts,
                # so a reader can see WHICH arm failed, not merely that one did.
                "observed_crf_differentiated_counts_per_arm": [int(c) for c in n_crf_differentiated_per_arm],
                "met": bool(crf_matured),
            },
            {
                "name": "enough_divergent_seeds",
                "kind": "readiness",
                "description": (
                    "number of seeds whose candidate pool is DIVERGENT on ALL C1 comparison "
                    "arms (A0 + A1 + ARM_NOISE) >= MIN_DIVERGENT_SEEDS. Per-seed-divergent "
                    "gating; too few => substrate_not_ready_requeue (pool too collapsed to "
                    "test conversion)."
                ),
                "control": "consumed cand_world_summary pairwise spread > floor (GAP-A); per-seed",
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                # COUNT-shaped, INCLUSIVE floor: `met` is exactly
                # `n_primary_div >= MIN_DIVERGENT_SEEDS`. Declared rather than left to the
                # indexer's default so the boundary case is explicit.
                "comparator": ">=",
                "direction": "lower",
                # ---- 707c REPAIR 5: HEADROOM, on the record (autopsy sec 3b) ----
                # 707b cleared this at EXACTLY 3.0/3.0 with seed 47 missing the 0.05 floor by
                # 0.00064 (1.3%), on a statistic drawn from the contaminated cache path. Here
                # the threshold is raised 3 -> 4 (with SEEDS widened 6 -> 8 so 4 is reachable)
                # AND the statistic is recomputed on FRESH ticks only. These non-bound
                # observables make the remaining slack legible instead of implicit: a future
                # reader can see whether this run cleared with room or scraped through.
                "observed_divergence_headroom": int(divergence_headroom),
                "observed_divergence_margins": divergence_margins,
                "observed_n_seeds_attempted": int(n_seeds_attempted),
                "met": bool(enough_divergent),
            },
            {
                "name": "loops_carry_live_cross_loop_variance",
                "kind": "readiness",
                "description": (
                    "ARC-110 NON-DEGENERACY guard (what_would_answer): on A1_LOOPS a non-motor "
                    "loop must FLIP the within-eligible winner (loop_committed_neq_motor_winner) "
                    "or the loops must DISAGREE (loop_cross_loop_winner_disagreement) on a "
                    "non-trivial fraction of P2 ticks, AND a non-motor loop must carry per-loop "
                    "preference RANGE > 0 -- on a majority of seeds. A loop pinned to the motor "
                    "winner is a vacuous split => substrate_not_ready_requeue. measured = the "
                    "NUMBER of A1 seeds carrying live cross-loop variance."
                ),
                "control": "A1_LOOPS loop_frac_committed_neq_motor / loop_frac_disagree + per-loop pref range",
                # COUNT-shaped, INCLUSIVE floor: `met` is `_maj(a1_rows, loop_cross_variance)`,
                # i.e. `n_a1_loop_cross_variance >= MIN_SEEDS_FOR_PASS` -- a COUNT of seeds.
                # This entry previously reported min(flip/disagree fraction) across A1 seeds
                # against LOOP_CROSS_VARIANCE_FRAC_FLOOR, which is strictly HARSHER than the
                # shipped predicate (a majority of seeds can clear the floor while the min does
                # not), so the indexer's authoritative recompute wrongly flagged sound
                # diagnostics precondition_unmet. No single fraction statistic CAN reproduce
                # `met`: the per-seed boolean is a CONJUNCTION ((flip frac > floor OR disagree
                # frac > floor) AND a non-motor pref range > floor), and a count over a
                # conjunction does not distribute into per-leg counts. The min-fraction number
                # is preserved below as a NON-BOUND diagnostic (extra keys are ignored by the
                # recompute) so no information is lost.
                "measured": float(n_a1_loop_cross_variance),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_a1_flip_or_disagree_frac": float(
                    round(max(a1_loop_flip_min, a1_loop_disagree_min), 6)
                ),
                "observed_cross_loop_variance_frac_floor": float(LOOP_CROSS_VARIANCE_FRAC_FLOOR),
                "met": bool(loop_cross_variance_ok),
            },
            {
                "name": "named_channel_routing_live",
                "kind": "readiness",
                "description": (
                    "ARC-110 C2 RELEASE non-degeneracy gate (707b): on A1_LOOPS at least one "
                    "LIMBIC channel (ofc/liking/vigour) must reach the segregated-loop "
                    "arbitration carrying a routed per-candidate RANGE > LIMBIC_ROUTED_RANGE_FLOOR "
                    "(peak over P2 ticks), on a strict-majority of DIVERGENT seeds. This is the "
                    "MECH-191 phasic-externalisation gate: the named bias HEADS emit "
                    "per-candidate-FLAT output, so without the use_named_channel_routing release "
                    "the limbic loop is inert and ARM_DROP_LIMBIC is byte-identical to A1 (the "
                    "707 vacuous DROP==A1). Evaluated BEFORE C2 is scored. measured = the NUMBER "
                    "of DIVERGENT seeds on which a limbic channel is routing-live. "
                    "707c FRAMING CORRECTION (autopsy 2026-07-20 sec 3a): this gate is BINARY "
                    "LIVENESS ONLY and carries NO magnitude. In 707b it read EXACTLY 1.414214 = "
                    "sqrt(2) on every arm and every seed -- zero variance across all 24 cells -- "
                    "because it is a STRUCTURAL CONSTANT of project_channel_range "
                    "(e3_selector.py:124-177): the projection centers the [K, D] feature matrix "
                    "and projects onto its leading right-singular vector, so a channel routed as "
                    "ORTHONORMAL one-hot rows yields ||e_i - e_j|| = sqrt(2) whenever >= 2 "
                    "candidates occupy distinct categories, independent of K, seed and arm. The "
                    "707b reading '1.414 >> 0.001' therefore does NOT denote a 1414x margin on a "
                    "graded magnitude. It certifies routing is ON and non-degenerate (and it does "
                    "cleanly resolve 707's byte-identical DROP == A1), and nothing more. The "
                    "MAGNITUDE half of the C2 non-degeneracy claim is carried by the separate "
                    "limbic_graded_competition_live entry below."
                ),
                "control": "A1_LOOPS loop_limbic_routed_range_max (peak limbic routed per-candidate range over P2)",
                # COUNT-shaped, INCLUSIVE floor: `met` is
                # `enough_divergent and _div_pass(n_named_routing_live, n_primary_div)`, and
                # _div_pass is `n_ok >= max(MIN_SEEDS_FOR_PASS, ceil(FRACTION * n_div))` --
                # the threshold reported here -- guarded by `n_div >= MIN_DIVERGENT_SEEDS`,
                # which is the same leg as `enough_divergent` and is declared separately as
                # `enough_divergent_seeds` (an AND of two counts is not reproducible from one
                # (measured, threshold) pair). This entry previously reported MAX across A1
                # seeds of the limbic routed range against LIMBIC_ROUTED_RANGE_FLOOR, which is
                # strictly LOOSER than the shipped strict-majority count -- one seed clearing
                # the floor would have satisfied the recompute while `met` was False -- and in
                # any case the per-seed boolean is a CONJUNCTION (routing_active_ticks > 0 AND
                # range > floor), which no single range statistic can reproduce. The
                # max-range number is preserved as a NON-BOUND diagnostic.
                "measured": float(n_named_routing_live),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "comparator": ">=",
                "direction": "lower",
                "observed_a1_limbic_routed_range_max": float(round(a1_limbic_routed_range_max, 6)),
                "observed_limbic_routed_range_floor": float(LIMBIC_ROUTED_RANGE_FLOOR),
                # The sqrt(2) pin itself, on the record as a non-bound observable so a reader
                # can SEE the zero-variance structural constant rather than take it on trust.
                "observed_a1_limbic_routed_range_distinct_values": sorted({
                    round(float(r.get("loop_limbic_routed_range_max", 0.0) or 0.0), 6)
                    for r in a1_rows
                }),
                "met": bool(named_channel_routing_live),
            },
            {
                # ---------------- 707c REPAIR 6 (autopsy sec 3a) ----------------
                "name": "limbic_graded_competition_live",
                "kind": "readiness",
                "description": (
                    "GRADED magnitude half of the C2 non-degeneracy claim, replacing the "
                    "sqrt(2)-pinned structural constant above as the basis on which C2 rests. "
                    "On A1_LOOPS the limbic loop must carry MEASURED per-candidate competition: "
                    "loop_limbic_pref_range_peak (e3_selector.py:1881-1883, the range of the "
                    "z-scored limbic-loop accumulation across the ELIGIBLE candidates, peaked "
                    "over FRESH E3 selections) > LIMBIC_GRADED_PREF_RANGE_FLOOR, on a "
                    "strict-majority of DIVERGENT seeds. Unlike the routed-range gate this "
                    "quantity is genuinely graded in the 707b record (0.013 .. 3.12 across "
                    "cells, hard ZERO on A0 where routing is off) and it DISSOCIATES exactly "
                    "where C2 requires: on seed 43 it collapses A1 1.0931 -> DROP 0.0133 while "
                    "assoc_pref_range RISES to receive the remapped channels. A limbic ABLATION "
                    "is only interpretable if the limbic loop was carrying competition to begin "
                    "with; if it was not, C2 is vacuous and the run must self-route "
                    "substrate_not_ready_requeue, NEVER a weakens. measured = the WORST "
                    "(minimum) A1 cell over the divergent seeds, matching the per-seed "
                    "majority-count semantics rather than a mean that could mask a dead cell."
                ),
                "control": (
                    f"A1_LOOPS loop_limbic_pref_range_peak over fresh E3 selections; worst "
                    f"divergent-seed cell seed={_worst_a1_graded[1]}"
                ),
                "offending_cell": f"A1_LOOPS:seed{_worst_a1_graded[1]}",
                # COUNT-shaped like its siblings: `met` is
                # `enough_divergent and _div_pass(n_limbic_graded_live, n_primary_div)`.
                "measured": float(n_limbic_graded_live),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                "comparator": ">=",
                "direction": "lower",
                "observed_worst_a1_limbic_pref_range_peak": round(float(_worst_a1_graded[0]), 6),
                "observed_limbic_graded_pref_range_floor": float(LIMBIC_GRADED_PREF_RANGE_FLOOR),
                "met": bool(limbic_graded_competition_live),
            },
            {
                # ---------------- 707c REPAIR 4 (autopsy sec 8 property 4) ----------------
                "name": "fresh_selects_sufficient",
                "kind": "readiness",
                "description": (
                    "EVERY arm-seed recorded at least MIN_FRESH_SELECTS genuinely FRESH E3 "
                    "selections in P2. 707c accumulates the primary DV ONLY on verified-fresh "
                    "ticks (agent.e3.last_score_diagnostics cleared to None before every "
                    "select_action and required to repopulate), so the honest denominator is "
                    "~1/e3_steps_per_tick of 707b's, whose per-env-step read made "
                    "committed_class_entropy_nats an entropy over a HOLD-WEIGHTED histogram. An "
                    "honest-but-underpowered DV must route substrate_not_ready_requeue, NEVER a "
                    "weakens -- which is precisely the branch 707b took on an instrument whose "
                    "A1-vs-null contrast was sign-inconsistent across all three divergent seeds "
                    "and 8x below the demonstrated contamination floor. measured = the WORST "
                    "cell (min across arm-seeds), matching the all(...) quantifier this `met` "
                    "expression uses, so the indexer's recompute agrees. DIAGNOSTIC TRIPWIRE: "
                    "n_fresh_select == n_p2_ticks on any arm-seed would mean the guard is not "
                    "working -- that exact equality was the 708 defect signature."
                ),
                "control": (
                    f"worst arm-seed n_fresh_select over P2; offending cell "
                    f"arm={_worst_fresh[1]} seed={_worst_fresh[2]}"
                ),
                "offending_cell": f"{_worst_fresh[1]}:seed{_worst_fresh[2]}",
                "measured": float(_worst_fresh[0]),
                "threshold": float(MIN_FRESH_SELECTS),
                "comparator": ">=",
                "direction": "lower",   # FLOOR
                "met": bool(fresh_selects_sufficient),
            },
            {
                "name": "in_layer_null_live",
                "kind": "readiness",
                "description": (
                    "707a FIX (supersedes in_layer_null_verified_lifting): the S2 IN-LAYER "
                    "same-layer null (ARM_NOISE: magnitude-matched random structure at the "
                    "non-motor loop accumulators) is a LIVE perturbation -- it actively perturbs "
                    "the committed selection (loop_noise_active_ticks > 0 AND "
                    "loop_frac_committed_neq_motor or loop_frac_disagree > the cross-loop-variance "
                    "floor) on a strict-majority of DIVERGENT seeds. This is the SAME-LAYER null "
                    "the single arena could NOT construct (704b/706b binding constraint); a null "
                    "that does NOT reach the committed-class readout makes C1's strict-above-null "
                    "bar meaningless => substrate_not_ready_requeue (NOT a weakens). The RETIRED "
                    "707 proxy required the null to LIFT entropy strict-above A0 -- the 700-lineage "
                    "TEMPERATURE-null liveness test, INVALID for a structured-accumulator null: "
                    "random magnitude-matched loop content lands at baseline, not above it "
                    "(structured beats random; random ~ A0), so it spuriously requeued a valid "
                    "result and made the WEAKENS branch unreachable. The path-artifact hypothesis "
                    "is ruled out by C1 (A1 strict-above the live null), NOT by null-lift. "
                    "measured = n divergent seeds the null is LIVE on (n_noise_lifts recorded "
                    "separately for continuity, no longer gates)."
                ),
                "control": "ARM_NOISE loop_frac_committed_neq_motor / loop_frac_disagree + loop_noise_active_ticks, divergent seeds",
                "measured": float(n_noise_live),
                "threshold": float(max(MIN_SEEDS_FOR_PASS, int(math.ceil(DIVERGENT_PASS_FRACTION * max(n_primary_div, 1))))),
                # COUNT-shaped, INCLUSIVE floor: `met` is
                # `enough_divergent and _div_pass(n_noise_live, n_primary_div)`, and _div_pass
                # is `n_ok >= max(MIN_SEEDS_FOR_PASS, ceil(FRACTION * n_div))` -- exactly the
                # threshold reported here -- guarded by `n_div >= MIN_DIVERGENT_SEEDS`, which
                # is the SAME leg as `enough_divergent`. That leg is declared separately as
                # `enough_divergent_seeds`, so this entry declares the _div_pass leg alone
                # (an AND of two counts is not reproducible from one (measured, threshold)
                # pair; the two entries conjoined are the same predicate).
                "comparator": ">=",
                "direction": "lower",
                "met": bool(noise_verified_lifting),
            },
            {
                "name": "learning_engaged_finer_channels_dissociable",
                "kind": "readiness",
                "description": (
                    "on A1_LOOPS the finer w_chan_finer entries MOVED + carry cross-channel "
                    "range above floor, on a majority of seeds -- learning is engaged (else "
                    "the loops are reading an un-trained gate). measured = the NUMBER of A1 "
                    "seeds with fcg_moved. Below floor => substrate_not_ready_requeue."
                ),
                "control": "A1 fcg_w_chan_finer_range_max",
                # COUNT-shaped, INCLUSIVE floor: `met` is `_maj(a1_rows, fcg_moved)`, i.e.
                # `n_fcg_moved >= MIN_SEEDS_FOR_PASS`. SPLIT from the delta_t leg below: the
                # single entry that used to carry both had `met = fcg_moved_ok and
                # fcg_delta_nonflat_ok`, an AND of two independent counts that NO single
                # (measured, threshold) pair can reproduce -- and it reported min over A1 seeds
                # of fcg_w_chan_finer_range_max, a per-seed magnitude that is strictly harsher
                # than a majority count and says nothing at all about the delta_t leg. The
                # min-magnitude number is preserved as a NON-BOUND diagnostic key.
                "measured": float(n_fcg_moved),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_a1_w_chan_finer_range_max": float(
                    min([r["fcg_w_chan_finer_range_max"] for r in a1_rows] or [0.0])
                ),
                "observed_w_chan_finer_range_floor": float(W_CHAN_FINER_RANGE_FLOOR),
                "met": bool(fcg_moved_ok),
            },
            {
                "name": "learning_engaged_delta_nonflat",
                "kind": "readiness",
                "description": (
                    "on A1_LOOPS the signed-RPE delta_t carries cross-tick variance above "
                    "floor on a majority of seeds -- the second leg of the learning-engaged "
                    "guard. measured = the NUMBER of A1 seeds with fcg_delta_nonflat."
                ),
                "control": "A1 fcg_delta_t_std",
                # COUNT-shaped, INCLUSIVE floor: `met` is `_maj(a1_rows, fcg_delta_nonflat)`,
                # i.e. `n_fcg_delta_nonflat >= MIN_SEEDS_FOR_PASS`. See the entry above for why
                # the two legs are declared separately; their conjunction is the same predicate
                # the shipped code applies, and the label routing still reads the two booleans.
                "measured": float(n_fcg_delta_nonflat),
                "threshold": float(MIN_SEEDS_FOR_PASS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_a1_delta_t_std": float(
                    min([r["fcg_delta_t_std"] for r in a1_rows] or [0.0])
                ),
                "observed_delta_t_std_floor": float(DELTA_T_STD_FLOOR),
                "met": bool(fcg_delta_nonflat_ok),
            },
            {
                "name": "candidate_pool_divergent",
                "kind": "readiness",
                "description": (
                    "consumed cand_world_summaries (e2.world_forward) per-candidate SPREAD "
                    "clears the GAP-A non-vacuity floor on enough seeds: measured = the NUMBER "
                    "of seeds DIVERGENT on all C1 comparison arms, threshold = "
                    "MIN_DIVERGENT_SEEDS."
                ),
                "control": "SD-056 e2 trained online in P0; candidate_summary_source=e2_world_forward",
                # COUNT-shaped, INCLUSIVE floor: `met` is `enough_divergent`, i.e.
                # `n_primary_div >= MIN_DIVERGENT_SEEDS` -- a COUNT of seeds divergent on ALL
                # C1 comparison arms. This entry previously reported min over all_rows of
                # consumed_summary_pairwise_dist_mean against CONSUMED_SPREAD_FLOOR, which is
                # strictly HARSHER than the shipped count predicate (a majority of seeds can
                # clear the floor while the min does not), so the indexer's authoritative
                # recompute wrongly flagged sound diagnostics precondition_unmet. No spread
                # statistic CAN reproduce `met`: the per-seed divergence boolean is itself a
                # CONJUNCTION (spread > CONSUMED_SPREAD_FLOOR and dist_max <
                # CONSUMED_MAGNITUDE_CEIL) evaluated on EACH C1 arm, and a count over a
                # conjunction does not distribute into per-leg counts. The min-spread number is
                # preserved as a NON-BOUND diagnostic (extra keys are ignored by the recompute)
                # so no information is lost.
                "measured": float(n_primary_div),
                "threshold": float(MIN_DIVERGENT_SEEDS),
                "comparator": ">=",
                "direction": "lower",
                "observed_min_consumed_spread": float(
                    min([r["consumed_summary_pairwise_dist_mean"] for r in all_rows] or [0.0])
                ),
                "observed_consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
                "met": bool(enough_divergent),
            },
        ],
        "criteria": [
            {
                "name": "C1_A1_loops_strict_above_A0_and_in_layer_null",
                "load_bearing": True,
                "passed": bool(c1_holds),
            },
            {
                "name": "C2_limbic_loop_load_bearing_A1_above_drop_limbic",
                "load_bearing": False,
                "passed": bool(c2_holds),
            },
        ],
        "criteria_non_degenerate": {
            "preconditions_met": bool(preconditions_met),
            "enough_divergent_seeds": bool(enough_divergent),
            "loops_carry_live_cross_loop_variance": bool(loop_cross_variance_ok),
            "named_channel_routing_live": bool(named_channel_routing_live),
            # 707c REPAIR 4/6: the two new gates are non-degeneracy conditions in their own
            # right. A DV without an honest denominator, or a C2 ablation of a loop that
            # carried no graded competition, is exactly a vacuous criterion.
            "limbic_graded_competition_live": bool(limbic_graded_competition_live),
            "fresh_selects_sufficient": bool(fresh_selects_sufficient),
            "in_layer_null_live": bool(noise_verified_lifting),
            "learning_engaged": bool(fcg_moved_ok and fcg_delta_nonflat_ok),
            "crf_matured": bool(crf_matured),
            "c2_drop_differs_from_a1": bool(c2_drop_differs_from_a1),
        },
        # ---- 707c: REPORTED, NEVER GATING (autopsy sec 8 property 3) ----
        # Emitted on PASS runs too: a diagnostic that appears only when something already
        # looks wrong cannot establish that anything was ever right.
        "replication_telemetry": {
            "exposure_imbalance": exposure_imbalance,
            "max_exposure_imbalance_vs_a0": round(float(max_exposure_imbalance), 6),
            "exposure_imbalance_report_floor": float(EXPOSURE_IMBALANCE_REPORT_FLOOR),
            "worst_cell_n_fresh_select": int(_worst_fresh[0]),
            "worst_cell": f"{_worst_fresh[1]}:seed{_worst_fresh[2]}",
            "note": (
                "707b's arm exposure spread reached +97.6% (s45) / +49.8% (s42) / -23.8% "
                "(s47) on the RAW per-env-step denominator. Recorded here so the mechanism "
                "of that distortion stays auditable. GATES NOTHING."
            ),
        },
        "divergence_headroom": {
            "n_primary_div": int(n_primary_div),
            "min_divergent_seeds": int(MIN_DIVERGENT_SEEDS),
            "headroom": int(divergence_headroom),
            "n_seeds_attempted": int(n_seeds_attempted),
            "margins": divergence_margins,
            "note": (
                "707b cleared enough_divergent_seeds at EXACTLY 3.0/3.0 (zero headroom) with "
                "seed 47 missing by 1.3%. GATES NOTHING here -- enough_divergent_seeds is the "
                "gate; this is the margin record that makes a near-miss visible."
            ),
        },
        "limbic_ablation_effect": limbic_ablation_effect,
    }

    total_seeds = len(ARMS) * len(seeds)
    total_completed = len(all_rows)

    manifest_core = {
        "outcome": outcome,
        "overall_direction": overall_direction,
        "evidence_direction_per_claim": per_claim,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": seeds,
        "n_arms": len(ARMS),
        "n_reuse_hits": int(n_reuse_hits),
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "conversion_margin": float(CONVERSION_MARGIN),
            "growth_margin": float(GROWTH_MARGIN),
            "min_divergent_seeds": int(MIN_DIVERGENT_SEEDS),
            "divergent_pass_fraction": float(DIVERGENT_PASS_FRACTION),
            "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
            "loop_cross_variance_frac_floor": float(LOOP_CROSS_VARIANCE_FRAC_FLOOR),
            "loop_pref_range_floor": float(LOOP_PREF_RANGE_FLOOR),
            "limbic_routed_range_floor": float(LIMBIC_ROUTED_RANGE_FLOOR),
            "consumed_spread_floor": float(CONSUMED_SPREAD_FLOOR),
            "delta_t_std_floor": float(DELTA_T_STD_FLOOR),
            "w_chan_finer_range_floor": float(W_CHAN_FINER_RANGE_FLOOR),
            "loop_segregation_normalize": str(LOOP_SEGREGATION_NORMALIZE),
            "loop_segregation_noise_alpha": float(LOOP_SEGREGATION_NOISE_ALPHA),
            "use_f_eligibility_demotion": bool(USE_F_ELIGIBILITY_DEMOTION),
            "use_go_nogo_constitution": bool(USE_GO_NOGO_CONSTITUTION),
            "modulatory_shortlist_k": int(MODULATORY_SHORTLIST_K),
        },
        "acceptance_criteria": {
            "preconditions_met": preconditions_met,
            "n_divergent_seeds": int(n_primary_div),
            "enough_divergent_seeds": enough_divergent,
            "crf_matured": crf_matured,
            "loops_carry_live_cross_loop_variance": loop_cross_variance_ok,
            "a1_loop_flip_frac_min": round(a1_loop_flip_min, 6),
            "a1_loop_disagree_frac_min": round(a1_loop_disagree_min, 6),
            "named_channel_routing_live": named_channel_routing_live,
            "n_named_routing_live_over_divergent": int(n_named_routing_live),
            "a1_limbic_routed_range_max": round(a1_limbic_routed_range_max, 6),
            "limbic_routed_range_floor": float(LIMBIC_ROUTED_RANGE_FLOOR),
            "in_layer_null_live": noise_verified_lifting,
            "n_noise_live_over_divergent": int(n_noise_live),
            "n_noise_lifts_over_a0": int(n_noise_lifts),
            "learning_engaged_fcg_moved": fcg_moved_ok,
            "learning_engaged_fcg_delta_nonflat": fcg_delta_nonflat_ok,
            "C1_loop_conversion_a1_above_a0_and_null": c1_holds,
            "C1_a1_n_seeds": int(n_c1),
            "C1_n_divergent": int(n_primary_div),
            "C2_limbic_load_bearing_a1_above_drop": c2_holds,
            "C2_n_seeds": int(len(c2_seeds)),
            "C2_n_divergent": int(len(c2_div)),
            "C2_drop_differs_from_a1": c2_drop_differs_from_a1,
            "C2_n_drop_differs_seeds": int(len(c2_drop_differs_seeds)),
            "mean_committed_class_entropy_a0_single_arena": round(a0_mean_dv, 6),
            "mean_committed_class_entropy_a1_loops": round(a1_mean_dv, 6),
            "mean_committed_class_entropy_arm_noise_in_layer_null": round(noise_mean_dv, 6),
            "mean_committed_class_entropy_arm_drop_limbic": round(drop_mean_dv, 6),
            "mean_loop_d1d2_conflict_signal_a1": round(_mean([r.get("loop_d1d2_conflict_signal", 0.0) for r in a1_rows]), 6),
            "mean_loop_local_credited_channels_a1": round(_mean([r.get("loop_local_credited_channels_mean", -1.0) for r in a1_rows]), 4),
        },
        "interpretation_grid": {
            "PASS_loop_segregation_converts_limbic_loop_load_bearing_supports_arc110": (
                "preconditions met (ENOUGH divergent seeds + loops carry LIVE cross-loop "
                "variance + the in-layer null is LIVE + learning engaged) AND C1 (A1_LOOPS "
                "committed-class entropy strict-above A0_SINGLE_ARENA AND the valid in-layer null "
                "on a strict-majority of divergent seeds) AND C2 (A1 strict-above ARM_DROP_LIMBIC "
                "-- the limbic loop is load-bearing). Loop segregation CONVERTS committed-action "
                "diversity where the single F-dominated arena plateaued -> the single-arena "
                "collapse WAS a binding constraint on the F-dominance conversion ceiling "
                "(MECH-439) -> supports ARC-110."
            ),
            "PASS_loop_segregation_converts_limbic_not_sole_driver_supports_arc110": (
                "preconditions + C1 met BUT NOT C2 (dropping the limbic loop does NOT remove the "
                "lift). Loop segregation converts, but the associative loop (or the loop structure "
                "per se) carries it -- the limbic loop is not the sole driver. Still supports "
                "ARC-110 (segregation converts); the per-loop load-bearing attribution is partial."
            ),
            "FAIL_valid_null_loops_vary_but_no_conversion_ceiling_intrinsic_weakens_arc110": (
                "DECISIVE. preconditions met (a VALID in-layer null can now be constructed -- "
                "ARM_NOISE is a LIVE perturbation of the committed selection -- and the loops "
                "carry live cross-loop variance) BUT A1_LOOPS does NOT lift committed-class "
                "entropy strict-above A0/the null. The single-arena collapse was NOT the binding "
                "constraint -> the F-dominance conversion ceiling is INTRINSIC, not an artefact "
                "of collapse -> weakens ARC-110 (the loop-segregation-as-artefact hypothesis)."
            ),
            "FAIL_substrate_not_ready_requeue": (
                "A precondition is unmet: too FEW divergent seeds (pool collapsed), OR the loops "
                "carry NO live cross-loop variance (a vacuous split pinned to the motor winner -- "
                "the ARC-110 what_would_answer guard), OR the NAMED limbic channels carry NO routed "
                "per-candidate range (named_channel_routing_live=false -- the MECH-191 phasic gap "
                "unfixed, so ARM_DROP_LIMBIC is a no-op = the 707 vacuous DROP==A1, and C2 cannot "
                "be validly scored), OR the S2 in-layer null is INERT (does NOT perturb the "
                "committed selection even with loops -> null-validity problem persists), OR "
                "learning was not engaged (finer channels not dissociable / delta_t flat). The "
                "conversion question could NOT be measured -- NOT a falsification."
            ),
        },
        "arm_results": arm_results,
    }
    return manifest_core

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
        "backlog_id": BACKLOG_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "non_degenerate": bool(result["non_degenerate"]),
        "degeneracy_reason": result["degeneracy_reason"],
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "same_layer_null": {
            "layer": (
                "ARC-110 S2: the eligibility/settling field at the non-motor loop accumulators "
                "(the SAME layer the per-loop settling acts on; NOT policy softmax temperature, "
                "the decoupled 700-lineage null)"
            ),
            "mechanism": (
                "config.e3.loop_segregation_noise_on=True -> _loop_inlayer_null replaces each "
                "non-motor loop accumulator with a magnitude-matched random-structure (gaussian) "
                "perturbation, range == loop_segregation_noise_alpha x the real loop accumulator "
                "range. Motor (F) is never nulled. Selection-only (no memory write)."
            ),
            "noise_alpha": LOOP_SEGREGATION_NOISE_ALPHA,
            "match_mode": "in_selector_per_tick_range_matched_to_real_loop_accumulator",
        },
        "reuse_baseline_from": REUSE_BASELINE_FROM,
        "evidence_direction_note": (
            f"V3-EXQ-707c ARC-110 LOOP-SEGREGATION C2 RELEASE VALIDATION -- INSTRUMENT REPAIR "
            f"(supersedes V3-EXQ-707b; experiment_purpose=diagnostic; claim_ids=[ARC-110]; "
            f"PROMOTES NOTHING). "
            f"WHY THIS RUN EXISTS: failure_autopsy_V3-EXQ-707b_2026-07-20 (CONFIRMED, "
            f"user-gated) WITHDREW 707b's evidence_direction=weakens on ARC-110 as a "
            f"measurement_test_design_defect, AND withdrew the narrowing derived from it "
            f"(necessary-but-not-sufficient / requires learned DA-gated cross-loop arbitration). "
            f"ARC-110 is therefore UNTESTED on the single-arena-artefact question -- neither "
            f"weakened nor narrowed -- and 707c is the venue that recovers it. DEFECT REPAIRED: "
            f"707b accumulated committed_class_counts on EVERY P2 env step (707b:1023-1035) with "
            f"no e3_tick guard while agent.py:5429 returns the HELD action before e3.select() is "
            f"reached, so committed_class_entropy_nats was an entropy over a HOLD-WEIGHTED "
            f"histogram (defect form 2; the form-1 staleness lint is structurally blind to it). "
            f"DISQUALIFYING: an entropy is a distribution-SHAPE statistic and arm exposure "
            f"differed by up to +97.6%, so it does not cancel; the A1-vs-null contrast the "
            f"DECISIVE branch depended on was SIGN-INCONSISTENT across all three divergent seeds "
            f"and 8x below the demonstrated contamination floor. 707c accumulates the DV ONLY on "
            f"verified-fresh E3 selections (and never on the random-fallback path 707b also "
            f"counted), recomputes the divergence admission statistic on fresh ticks too, adds "
            f"fresh_selects_sufficient + divergence-headroom + GRADED limbic-competition guards, "
            f"and records exposure telemetry. RETAINED FROM 707b, re-confirmed not "
            f"re-established: in_layer_null_live (strict >0 tick count, hard zero on A0/A1 -- the "
            f"same-layer null the single arena could NOT construct was genuinely built) and "
            f"frac_pre_ge2 (saturated at 1.0). The loop substrate is BUILT and demonstrably LIVE; "
            f"this is instrument repair, not a substrate build. "
            f"--- The 707b design 707c inherits unchanged follows. --- "
            f"707b LANDED THE C2 RELEASE: a no-op-default flag "
            f"use_named_channel_routing routes each named channel's per-candidate REPRESENTATION "
            f"through project_channel_range (range-preserving, the GAP-A path) into the segregated "
            f"loops, so the limbic loop (ofc/liking/vigour) carries REAL per-candidate range and "
            f"ARM_DROP_LIMBIC can differ from A1 -- making C2 (limbic load-bearing) testable. 707/"
            f"707a left C2 substrate-blocked: the named bias HEADS emit per-candidate-FLAT output "
            f"(MECH-191 phasic gap), so DROP_LIMBIC was byte-identical to A1 (vacuous). A NEW "
            f"per-named-channel non-degeneracy precondition (named_channel_routing_live: a limbic "
            f"channel's routed range > floor on a strict-majority of divergent seeds) is evaluated "
            f"BEFORE C2 is scored, so the vacuous DROP==A1 self-routes substrate_not_ready_requeue "
            f"(NEVER a false weakens). The 707a in-layer-null LIVENESS gate (null must PERTURB "
            f"committed selection, not raise entropy above A0) and C1 (A1 strict-above "
            f"max(A0, the live null)+margin) are UNCHANGED. "
            f"Substrate-readiness validation for the v4_loop_segregation substrate (ARC-110 + "
            f"ARC-109 D1/D2 + MECH-452 loop-local traces + the S2 in-layer null) built "
            f"2026-06-27 via /implement-substrate. Routed by the confirmed cluster autopsy "
            f"failure_autopsy_704b-706b-conversion-ceiling_2026-06-27: the conversion-ceiling "
            f"lineage converged from four angles (700b/700c learned-gating+settling, 704b finer-"
            f"channel, 706b double-gated curiosity) on the conclusion that the V3 SINGLE E3 "
            f"foraging arena structurally denies (a) committed-action-class conversion from non-"
            f"motor channels and (b) a VALID same-layer null; the gate cleared by 704b FAIL-to-"
            f"convert (positive-evidence-FOR-ARC-110). 4 arms on the SAME GAP-A reef-bipartite "
            f"foraging substrate + the SAME landed arithmetic envelope + finer-channel gating + "
            f"learned settling as a MATCHED CONSTANT; the ONLY swept factor is loop segregation: "
            f"A0_SINGLE_ARENA (use_loop_segregation=False -- the 704b-A2 single-arena baseline) / "
            f"A1_LOOPS (ARC-110 motor/assoc/limbic + ARC-109 D1/D2 + MECH-452 loop-local traces) "
            f"/ ARM_NOISE (A1 + the S2 in-layer same-layer null) / ARM_DROP_LIMBIC (A1 with the "
            f"limbic loop ablated -- the ARC-106 per-loop load-bearing falsifier). PRE-REGISTERED "
            f"decisive either way: A1 committed-class entropy strict-above A0 AND the VALID in-"
            f"layer null on a strict-majority of divergent seeds, with live cross-loop variance "
            f"=> single-arena collapse WAS a binding constraint -> supports ARC-110 (limbic loop "
            f"load-bearing if A1 also strict-above ARM_DROP_LIMBIC); a LIVE null + "
            f"loops vary BUT A1 does NOT convert => the ceiling is INTRINSIC -> weakens ARC-110. "
            f"Non-vacuity self-route substrate_not_ready_requeue (NEVER a false weakens): loops "
            f"must carry LIVE cross-loop variance (not a vacuous split pinned to the motor "
            f"winner), the NAMED limbic channels must carry routed per-candidate range > floor "
            f"(named_channel_routing_live -- else DROP_LIMBIC is a no-op = the 707 vacuous "
            f"DROP==A1 and C2 cannot be scored), the in-layer null must be LIVE (perturb the "
            f"committed selection, else the null-validity problem persists), the pool divergent "
            f"(GAP-A), finer channels dissociable + delta_t non-"
            f"flat. PROMOTES NOTHING (ARC-110/ARC-109/MECH-452 candidate/substrate_conditional/v3). "
            f"outcome={result['outcome']}; label={result['interpretation_label']}; "
            f"per_claim={result['evidence_direction_per_claim']}."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": "4-arm ARC-110 loop-segregation validation (A0_SINGLE_ARENA / A1_LOOPS / ARM_NOISE in-layer null / ARM_DROP_LIMBIC) + per-seed-divergent gating + live-cross-loop-variance non-degeneracy gate + LIVE in-layer same-layer null gate (707a: null-liveness, not null-lift)",
            "arms": "A0_SINGLE_ARENA (use_loop_segregation=False) / A1_LOOPS (ARC-110 motor/assoc/limbic + ARC-109 D1/D2 + MECH-452 loop-local traces) / ARM_NOISE (A1 + loop_segregation_noise_on) / ARM_DROP_LIMBIC (A1, limbic loop remapped into associative)",
            "swept_variables": "use_loop_segregation (A0 off, loop arms on) + loop_segregation_noise_on (ARM_NOISE) + limbic-loop channel-map ablation (ARM_DROP_LIMBIC). Finer-channel gating + learned settling ON on ALL arms.",
            "the_isolated_factor": (
                "loop segregation only: A0 runs the single-arena within-eligible argmin over the "
                "finer channels; A1 partitions the SAME finer channels into motor=F / "
                "associative=dACC+lPFC / limbic=OFC+liking+vigour loops with within-loop "
                "competition first + Haber ascending-spiral arbitration after + per-loop zscore "
                "normalisation (strips F's magnitude advantage). D1/D2 + loop-local traces are "
                "inert without loops, so the functional swept variable IS loop segregation."
            ),
            "matched_constant_arithmetic_envelope": (
                "use_f_eligibility_demotion=True + use_f_eligibility_adaptive_floor=True (689e) + "
                "use_go_nogo_constitution=True (689g) + use_modulatory_selection_authority=True (643a) + "
                "use_modulatory_channel_routing (cand_world_summary) + top_k shortlist (k=3, 569i)"
            ),
            "matched_diversity_stack": (
                "MECH-341 stratified + use_dacc (MECH-260 perseveration No-Go feed) + use_gated_policy + "
                "use_lateral_pfc_analog (lateral_pfc_train_rule_bias_head=True, TRAINED in P1 REINFORCE) + "
                "SD-056 all levers + the matured/maintained CRF pool + use_candidate_rule_field + "
                "use_finer_channel_gating=True + use_learned_settling_step=True"
            ),
            "settling_W_lat": "ON on ALL arms (the within-loop settling each segregated loop runs; on A0 the legacy single-arena settling)",
            "primary_dv": "committed-action-class entropy (nats), interpreted on divergent seeds only",
            "phases": "P0 e2-train (CRF matures, finer gating ON) -> P1 frozen-encoder bias-head REINFORCE -> P2 e2+bias frozen, gating + loops KEEP adapting",
            "learning_wiring": "w_chan_finer learns via e3.post_action_update driven by agent.update_residue every waking tick; on the loop arms credit is loop-local (MECH-452); the S2 null is selection-only (no learning write)",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "loop_segregation_normalize": LOOP_SEGREGATION_NORMALIZE,
            "loop_segregation_noise_alpha": LOOP_SEGREGATION_NOISE_ALPHA,
            "loop_default_channel_map": "motor=F; associative=dACC+lPFC; limbic=OFC+liking+vigour (built-in _LOOP_DEFAULT_CHANNEL_MAP)",
            "drop_limbic_channel_map": "limbic channels remapped into associative -> limbic loop empty (per-loop ablation)",
            "reuse_baseline_from": REUSE_BASELINE_FROM,
            "reusable_arm_ids": list(REUSABLE_ARM_IDS_LOCAL),
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
            "arc110_relationship": "this IS the ARC-110 validation -- the gate cleared by 704b FAIL-to-convert; supports => single-arena collapse was a binding constraint, weakens => ceiling intrinsic",
        },
        # ---------------- 707c REPAIR 8 (autopsy sec 4) ----------------
        # `arm_results` MUST sit at the manifest TOP LEVEL. manifest_core's
        # _hoist_multi_arm_substrate_hash (manifest_core.py:117-131) reads
        # manifest["arm_results"] and nothing else; 707b nested it under "result", so the
        # hoist silently returned None, the stamper fell back to a driver-INCLUSIVE
        # single-arm hash, and the landed 707b manifest carries NO substrate_hash at all --
        # confirmed by inspection (validate_recording reports six missing always-core
        # fields). Provenance was therefore unpinned and no arm reuse was possible from it,
        # even though every one of its 24 cells had a perfectly good fingerprint.
        # The full result body is still emitted under "result" for continuity; this is an
        # additional top-level reference to the SAME list object, not a second copy.
        "arm_results": result.get("arm_results", []),
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-707 ARC-110 parallel segregated loop-segregation validation"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    # 707c: retain the --dry-run manifest for INSPECTION instead of deleting it.
    # 707b unlinked its smoke manifest unconditionally, which is precisely why its own
    # smoke could never be audited: a dry run that merely does not crash cannot show that
    # n_fresh_select < n_p2_ticks, and their EQUALITY is the 708 defect signature (i.e. the
    # freshness guard silently not working). Since the whole point of this letter is the
    # telemetry, the smoke has to be able to produce the artifact that proves it. Use with
    # --out-dir pointing OUTSIDE evidence/ so the indexer never sees the toy manifest.
    parser.add_argument("--keep-dry-manifest", action="store_true")
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
        # 707c: the RUN's seeds, not the module-level SEEDS. 707b passed SEEDS here, so a
        # --dry-run manifest recorded all six seeds while having executed one -- an
        # always-core field that misdescribes the run it stamps.
        seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    _ac = result["acceptance_criteria"]
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"reuse_hits={result['n_reuse_hits']} "
        f"preconditions_met={_ac['preconditions_met']} "
        f"n_divergent={_ac['n_divergent_seeds']} "
        f"loop_cross_variance={_ac['loops_carry_live_cross_loop_variance']} "
        f"in_layer_null_live={_ac['in_layer_null_live']} "
        f"n_noise_live={_ac['n_noise_live_over_divergent']} n_noise_lifts={_ac['n_noise_lifts_over_a0']} "
        f"C1_loop_conversion={_ac['C1_loop_conversion_a1_above_a0_and_null']} "
        f"C2_limbic_load_bearing={_ac['C2_limbic_load_bearing_a1_above_drop']} "
        f"C2_drop_differs={_ac['C2_drop_differs_from_a1']} "
        f"label={result['interpretation_label']}",
        flush=True,
    )

    if args.dry_run and not args.keep_dry_manifest:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
    elif args.dry_run:
        print(f"dry-run manifest RETAINED for inspection: {out_path}", flush=True)

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
    sys.exit(0)
