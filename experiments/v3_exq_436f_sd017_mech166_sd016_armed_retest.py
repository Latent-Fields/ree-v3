#!/opt/local/bin/python3
"""
V3-EXQ-436f -- SD-017/ARC-045/MECH-166 slot-differentiation retest, on the
436e OCCUPIED-SLOTS-ONLY DV machinery, with SD-016 CUE-CONDITIONING ARMED
SLEEP DRIVER: manual-multi (run_sleep_cycle() called directly every SLEEP_INTERVAL episodes in training loop)

experiment_purpose: evidence

Discharges the SD-016-armed retest recorded on SD-017/ARC-045/MECH-166 by
failure_autopsy_V3-EXQ-436e_2026-08-13 (confirmed, user-adjudicated at the
Step 8 gate). That autopsy found 436e's DV repair SOUND (all six items of the
2026-08-07 methodology-check Recommendation #4 implemented and empirically
confirmed) but its result non_contributory at the P0 readiness layer: only
2/5 seeds carried >=2 occupied ContextMemory slots (need 3), because seeds
7/13/100 occupied exactly ONE slot of 16 in BOTH arms despite 3,019-4,884
write() calls each. Root cause established by probe: ContextMemory.write()
addresses by argmin(query_proj(state) . memory) and has a deterministic
single-slot fixed point when the query stream is near-constant -- the
long-registered SD-016 z_world entropy bottleneck (cross-batch cosine 0.998,
substrate_queue SD-016 entry, 2026-04-28). The SD-016 selection-mechanism fix
LANDED 2026-08-11 (ree-v3 110a2785b6, in 436e's own substrate commit
871c221933, verified by merge-base) with SD-017 named in its unblocks_claims
-- and V3-EXQ-436e armed NONE of it (sd016_cue_slot_tagger and
sd016_context_divergence_weight both absent from its config). This letter
arms it. Lineage: 436 -> 436a -> 436b -> 436c -> 436d -> 436e -> 436f.

WHAT CHANGES vs 436e (the SINGLE scientific change -- arm SD-016; the DV
machinery, conditions, seeds, sleep manipulation, C1/C4 gates and thresholds
are otherwise inherited from 436e UNCHANGED):

  The SD-016 production combination validated by V3-EXQ-922 (which CONFIRMED
  MECH-150 and SUPPORTS MECH-151/152/ARC-041 on this substrate) is now armed
  and HELD CONSTANT across all three conditions (a substrate property, exactly
  like 436e's writepath_mode and requires_grad freeze -- sleep remains the sole
  manipulation):
    sd016_enabled                    = True   (constructs the SD-016 apparatus;
                                                the whole cue-tagger/ctxdiv path
                                                is gated on this -- 436e left it
                                                default False, so its config-only
                                                knob list could not have armed
                                                anything)
    sd016_cue_slot_tagger            = True
    sd016_cue_slot_tagger_selection  = 'gumbel'   (NOT 'topk' -- eliminated by 922)
    sd016_context_divergence_weight  = 0.5

  ARMING IS NOT A CONFIG-ONLY FLIP (this is why 436f is real driver surgery,
  not the three-line change the autopsy's Routing sketch implied). SD-016's
  H1 context-divergence auxiliary loss is INERT as config alone:
  E1DeepPredictor.compute_context_divergence_loss is NEVER called by the agent
  automatically -- only ever by a driver's own training loop (V3-EXQ-907/922/
  922a), which must (a) collect fixed detached safe/dangerous z_world batches
  and (b) ADD the loss to its training total. 436e's training loop computes
  only e1_loss + e2_loss and calls neither, so setting the weight alone would
  leave the tagger MLP untrained (no terrain/cue/divergence gradient reaches
  it) and the saddle-breaking drive absent. This driver therefore ADDS 922's
  validated ctxdiv wiring to 436e's loop: it collects a fixed detached
  safe-context and dangerous-context z_world batch per (seed, condition) right
  after agent construction, and adds
  agent.e1.compute_context_divergence_loss(z_safe_div, z_dang_div) to the
  per-step training total in every condition. The existing 436e optimizer
  already includes every e1 parameter except the frozen context_memory.memory
  (standard_params), so the cue_slot_tagger it now constructs is trained by
  this loss with no optimizer change.

  A SIXTH P0 readiness check is added -- sd016_arming_engaged -- gating on the
  pooled magnitude of the applied ctxdiv loss across training being > 0, so an
  inert arming self-reports DISTINCTLY from an occupancy collapse.

READ-PATH vs WRITE-PATH CAVEAT (recorded so governance interprets a possible
re-failure correctly; see the governance flag raised alongside this queue
entry). The occupancy collapse that failed 436e is on the WRITE path:
ContextMemory.write() addresses by argmin over query_proj(STATE), where STATE
is the 64-dim [z_self, z_world] concatenation (verified: query_proj expects
64 dims, not z_world's 32). SD-016's cue_slot_tagger + ctxdiv loss operate on
the READ path (extract_cue_context) via a SEPARATE tagger MLP consuming 32-dim
z_world, and the ctxdiv loss trains that tagger on DETACHED z_world batches
(922's validated recipe), so its gradient does NOT reach the encoder that
feeds query_proj. It is therefore NOT mechanically guaranteed that arming the
read-path selection lifts write-slot occupancy: 436f may reproduce 436e's
insufficient_occupancy_for_c1 P0 failure. This is not a reason to withhold the
run -- the autopsy is user-confirmed, the sufficient_occupancy_for_c1 P0 gate
makes a re-failure self-reporting rather than false science, and the run is
~32 min of cloud compute -- but the two new P0 signals are designed to
DISCRIMINATE the outcome: sd016_arming_engaged=yes WITH
sufficient_occupancy_for_c1=no would be direct evidence that the read-path fix
does not reach write-path occupancy, routing the lineage to a WRITE-path
successor (query_proj / z_world-entropy) rather than another read-path letter.

--- (inherited 436e context follows) ---

Discharges the DV-repair obligation recorded on SD-017/ARC-045/MECH-166 by
failure_autopsy_V3-EXQ-436d-methodology-check_2026-08-07 (confirmed,
user-confirmed): "pending_retest_after_substrate stays true, now re-scoped
from 'after-substrate' to 'after-DV-repair' -- a V3-EXQ-436e re-run on an
occupied-slots-only DV is owed before this lineage's slot-differentiation
question can be tested at all." Same lineage: 436 -> 436a -> 436b -> 436c ->
436d -> 436e -> 436f.

WHY THIS RUN EXISTS (the DV, not the substrate, was the defect). 436d was
the first run of the lineage with a working write path (P0 clearance, no
known instrumentation defect) and its C1 criterion FAILED 2/5 seeds (need
3/5). Governance HELD that result pending a methodology check: the
WAKING_ONLY baseline cosine ranged 0.0009-0.47 across seeds instead of
clustering near ARC-045's predicted ~1.0, and one seed showed sleep
INCREASING similarity. The methodology check (confirmed 2026-08-07) found
FIVE independent, jointly-fatal defects in slot_cosine_sim itself, not in
the substrate:

  F1. WAKING_ONLY performed ZERO ContextMemory writes (sd016_writepath_mode
      was held at "off" for BOTH conditions in 436d) -- C1 compared a bank
      holding 160 content writes against a bank holding no context at all.
  F2. slot_cosine_sim masks only the diagonal -- an UNWRITTEN slot bank
      reads ~0 (mean +0.000145, sd 0.008121 over 2000 random inits), not
      ~1.0 as ARC-045's prediction assumes. That near-1.0 baseline was
      reachable only under the LEGACY write_gate-as-payload defect 436d
      itself repaired -- C1 was calibrated against a substrate that no
      longer exists.
  F3. The whole-bank statistic is a PRODUCT of content-similarity (c) and
      occupancy-fraction (k/n): value ~ c*k(k-1)/(n(n-1)). Occupancy alone
      (orthogonal content) does not move it; only occupancy*similarity does.
  F4. Direction-reversing consequence of F3: an arm recruiting 12
      near-orthogonal slots (k=12, c~+0.113 -- EXCELLENT differentiation)
      scores 8x WORSE (+0.0763) than an arm using a single slot (k=1,
      +0.0092). The criterion is anti-correlated with the mechanism it was
      written to detect.
  F5. WAKING_ONLY's observed 0.0009-0.47 spread is fully reproduced by Adam
      optimizer drift alone on context_memory.memory (an nn.Parameter
      reachable through read()'s shared softmax attention and included in
      the driver's own optimizer), with ZERO write() calls, at the exact
      five experiment seeds -- non-monotone in training time, so the
      baseline's readout value is an arbitrary function of seed and of
      where training stopped.
  F6. Seeds 13 and 200 sat INSIDE the untouched-bank null (z=+1.21, +0.09),
      making C1 unsatisfiable there by construction -- effective denominator
      <=3, not 5, so "2/5 vs 3/5" was never an interpretable null.

verdict: "C1's status as evidence" does not survive; P0 clearance (the write
path works) does. evidence_direction weakens -> non_contributory,
epistemic_category -> measurement_test_design_defect (applied to SD-017,
ARC-045, MECH-166 uniformly, 2026-08-07 governance).

WHAT CHANGES vs 436d (per the autopsy's Recommendation #4, all six items
implemented here -- the ONLY changes; conditions' underlying training loop,
env, action-selection wiring, C4/behavioural DVs, seeds, and thresholds are
otherwise UNCHANGED from 436d):

1. STATISTIC OVER OCCUPIED SLOTS ONLY, tracking write()'s min_idx directly
   (F2/F3/F4 fix). `ContextMemory.write()` (ree_core/predictors/e1_deep.py)
   computes `min_idx = torch.mm(query_proj(state), memory.t()).mean(0).argmin()`
   under torch.no_grad() and writes there -- this driver installs a
   thin instance-level wrapper around the SAME bound `write()` method (the
   ONE call site all three of write()'s callers funnel through: sense()'s
   per-tick hook at agent.py:4847, E1.update_from_observation at
   e1_deep.py:690, and run_sws_schema_pass's replay writes at
   agent.py:11313) that recomputes the identical min_idx expression
   READ-ONLY (under no_grad, before delegating to the original write()) and
   records it into a per-cell occupancy set. At read-time, the corrected DV
   (`slot_cosine_sim_occupied_only`) computes the mean off-diagonal cosine
   over ONLY the rows in that occupancy set -- masking every never-written
   slot out, not just the diagonal. The OLD whole-bank statistic is ALSO
   still computed and recorded verbatim as `slot_cosine_sim_raw_whole_bank`
   (audit-only, non-gating), so the confound documented by the methodology
   check stays auditable in every future manifest of this lineage, per the
   autopsy's own fan-out note that the identical defect recurs in
   sws_slot_diversity and the v3_exq_242/243/245/245a/245b/246 family.

2. SIMILARITY (c) AND OCCUPANCY (k) REPORTED SEPARATELY, NEVER AS THE
   PRODUCT (F3/F4 fix). `n_occupied_slots` (k) and `n_write_calls` (total
   write() invocations, which can exceed k since the same slot can be
   rewritten) are recorded as their own top-level per-cell fields.
   `slot_cosine_sim_occupied_only` is c, computed on ONLY the occupied
   subset -- it is not weighted or scaled by k. C1 gates on c alone
   (a directional paired comparison, per point 6 below), never on the
   c*k(k-1)/(n(n-1)) shape that produced F4's reversal.

3. WAKING_ONLY GIVEN A REAL WRITE PATH (F1 fix). Both WAKING_ONLY and
   SWS_THEN_REM now run with `sd016_writepath_mode="sense_only"` (the SD-016
   Part B2 per-tick hook at agent.py:4847, gated on
   `not self.e1._offline_mode` -- which run_sws_schema_pass /
   run_rem_attribution_pass set True only for the DURATION of the sleep
   pass itself via enter_offline_mode(), so per-tick waking writes fire
   identically across BOTH conditions during every ordinary waking step;
   SWS_THEN_REM additionally receives the offline consolidation writes
   during its periodic sleep cycles -- exactly the manipulation under
   test). This is a SUBSTRATE property held CONSTANT across the two
   comparison conditions (like contextmemory_gated_content_write in 436d),
   not itself the manipulation. WAKING_ONLY is now a genuine
   online-writes-only control instead of an empty bank. A THIRD condition,
   NO_WRITES (`sd016_writepath_mode="off"`, no sleep -- structurally
   identical to 436d's WAKING_ONLY), is added purely as a calibration
   negative control for point 4 below; it is NOT part of the C1 comparison.
   Per-arm write counts (n_write_calls, n_occupied_slots) are recorded for
   every condition so the three arms' write engagement is auditable
   side-by-side rather than assumed equal.

4. ADAM DRIFT NEUTRALIZED (F5 fix). `context_memory.memory.requires_grad_(
   False)` is called on every agent immediately after construction, for
   ALL THREE conditions (a substrate property held constant, exactly like
   point 3). write() still mutates `.memory.data` directly under its own
   torch.no_grad() block (unaffected by requires_grad), so writes work
   identically; but the parameter no longer receives a gradient from
   read()'s backward pass, so Adam can no longer perturb it between writes.
   The NO_WRITES arm (point 3) is the empirical check that this neutralization
   actually held THIS run: with zero write() calls and no gradient reaching
   memory, its raw whole-bank cosine should sit inside the untouched-bank
   null (point 5) at every seed -- if it drifts outside, something is
   writing to or perturbing the bank that the calibration model does not
   account for, and the run's C1 reading is not trustworthy. This is a P0
   readiness gate (see ACCEPTANCE CRITERIA), not a post-hoc note.

5. PREDICTED BASELINE RE-DERIVED AGAINST THE REPAIRED PATH, NOT REUSED FROM
   ARC-045'S NEAR-1.0 FIGURE (F2 fix). Rather than hard-coding the autopsy's
   own measured null (mean +0.000145, sd 0.008121 over 2000 draws, itself
   measured at these exact dims), this driver RE-DERIVES the untouched-bank
   null empirically at run time from a dedicated, seeded generator sampling
   `torch.randn(num_slots, memory_dim) * 0.01` (ContextMemory's own init
   recipe) -- reproducible, self-contained, and independent of the
   experiment's own per-seed RNG state. This null is used ONLY as the
   reference distribution for point 4's calibration check and as a
   descriptive diagnostic; C1 itself remains a PAIRED, directional
   comparison (SWS_THEN_REM's occupied cosine vs WAKING_ONLY's, same seed)
   exactly as in 436d, so it was never actually anchored to the absolute
   near-1.0 figure -- only the descriptive `arc045_abs_reference_legacy`
   secondary readout was, and that field is retained here relabelled as
   explicitly historical/non-predictive (it is NOT recomputed against the
   repaired write path, because ARC-045's own claims.yaml notes do not
   specify what the repaired-path absolute reference should be -- reporting
   a stale number under an honest label is safer than inventing a new one
   the claim was never pre-registered against).

6. NEGATIVE CONTROL ADDED (F5 verification, ties points 3+4+5 together).
   The NO_WRITES condition's raw whole-bank cosine, per seed, is compared
   against the freshly-derived null (point 5): |z-score| must stay under
   ADAM_DRIFT_NULL_TOLERANCE_SIGMA for the run to be treated as having a
   validated instrument this time. This is the direct empirical answer to
   "did the fix actually work this run", not an assumption.

CLAIM SUBSTRATE UNDER TEST (unchanged from 436d/436c/436b/436a -- the
pre-registered comparison itself is unchanged; only the statistic computing
each side of it is repaired):
  SD-017    (sleep_phase.minimal_sleep_infrastructure_v3): "context
            representations remain globally undifferentiated" without the
            SWS/REM-analog phases -- slot_cosine_sim -> 1.0 without them.
  ARC-045   (hippocampus.bidirectional_information_flow): "an agent with
            bidirectional offline flow should show cosine_sim < 0.95
            (differentiated contexts) after sleep phases; one with only
            waking online encoding remains at cosine_sim -> 1.0 regardless
            of training duration" (claims.yaml notes, verbatim experimental
            implication).
  MECH-166  (hippocampus.slot_formation_filling_temporal_separation): "Slot
            structure must be consolidated during an SWS-analog phase...
            A direct test requires implementing the SWS-analog pass and
            comparing attribution map quality (context cosine_sim...) with
            vs without it." This experiment IS that direct test, on a
            repaired instrument.

DV-SYMMETRY INVARIANCE DECLARATION (mandatory per 604c net, unchanged from
436b/436c/436d): the manipulation under test is "ran an SWS-analog + REM-
analog consolidation pass" vs "did not" (both conditions otherwise share the
SAME per-tick sense_only write path -- point 3 above). run_sleep_cycle()
directly mutates agent.e1.context_memory.memory via ContextMemory.write()
during sleep, and slot_cosine_sim_occupied_only reads that same memory
tensor, restricted to the write-touched rows. This is not a uniform
additive constant, not a monotone rescaling, and not a permutation of
interchangeable units -- the manipulation adds an entirely separate class of
write events (offline consolidation replay) whose targets and content
differ from the shared online writes, so the occupied-slot set and the
content within it can both differ between conditions, and the DV is not
invariant under any of the three flagged symmetry classes.

CONDITIONS (3): NO_WRITES (calibration-only negative control, sd016_writepath_
mode="off", no sleep -- NOT part of the C1 comparison), WAKING_ONLY
(baseline for C1, sd016_writepath_mode="sense_only", no sleep), SWS_THEN_REM
(sd016_writepath_mode="sense_only" PLUS full SWS-then-REM cycle every
SLEEP_INTERVAL episodes, plus the DR-6 context-conditioned harm threshold in
action selection -- unchanged from 436d). use_noise_floor=True on ALL THREE
conditions (a waking-phase substrate property, not part of the manipulation).
contextmemory_gated_content_write=True on ALL THREE (the 436d write-path
repair, held constant -- still required, since without it writes homogenize
per the 436c defect regardless of the occupancy mask).

ACCEPTANCE CRITERIA:
  P0 (gates C1's interpretability; five checks, ALL must clear):
      sws_context_memory_writes_occur / rem_attribution_rollouts_occur:
      pooled sleep-pass write/rollout counters (SWS_THEN_REM seeds) > 0 --
      unchanged from 436d, confirms the offline path fired.
      waking_writepath_engaged: pooled per-tick sense_only write() calls
      (WAKING_ONLY seeds) > 0 -- NEW, confirms point 3's fix actually gave
      WAKING_ONLY a real write path this run.
      sufficient_occupancy_for_c1: count of seeds where BOTH WAKING_ONLY and
      SWS_THEN_REM have >=2 occupied slots (the minimum for an off-diagonal
      mean to exist) must be >= C1_N_SEEDS_REQUIRED -- NEW, the direct fix
      for F6 (an unscoreable seed no longer silently deflates the
      denominator; the run self-routes to FAIL/non_contributory instead if
      too few seeds are scoreable).
      adam_drift_neutralized: max |z-score| of the NO_WRITES arm's raw
      whole-bank cosine vs the freshly-derived untouched-bank null, across
      all 5 seeds, must stay under ADAM_DRIFT_NULL_TOLERANCE_SIGMA -- NEW,
      the empirical check for point 4/6.
      Any unmet -> outcome FAIL, evidence_direction non_contributory, C1/C4
      NOT scored as evidence either way; interpretation label names the
      specific unmet gate.
  C1 (PRIMARY, SOLE GATE when P0 is met -- SD-017 + ARC-045 + MECH-166,
      wall-independent, PAIRED per seed, directional per ARC-045's own
      pre-registered experimental implication -- unchanged in SHAPE from
      436d, only the statistic is repaired):
      slot_cosine_sim_occupied_only(SWS_THEN_REM) <
      slot_cosine_sim_occupied_only(WAKING_ONLY) in >= 3/5 seeds.
  C4 (SECONDARY, ARC-045 slot_separation, non-gating, carried unchanged from
      436d -- a read-side visitation-distribution statistic, independent of
      the write-occupancy cosine confound):
      slot_separation(SWS_THEN_REM) > 0.3 in >= 3/5 seeds.
  Secondary / exploratory (recorded, NEVER gating, unchanged from 436d):
      harm_rate_dangerous, harm_rate_safe signed diffs.

PASS: P0 met AND C1 (n_seeds_passed >= 3/5).

INTERPRETATION GRID:
  P0 UNMET (any of the SIX gates -- five inherited from 436e, plus
              sd016_arming_engaged new in 436f) -> label names the specific
              gate (sleep_cycle_recording_gap_still_present /
              waking_writepath_not_engaged / insufficient_occupancy_for_c1 /
              adam_drift_neutralization_failed / sd016_arming_not_engaged).
              Route to /failure-autopsy on the named mechanism -- C1/C4 are
              not evidence either way. NOTE (436f): the joint reading
              sd016_arming_engaged=MET with insufficient_occupancy_for_c1=UNMET
              is the diagnostic signal that the read-path SD-016 fix did not
              lift write-path slot occupancy -> route to a WRITE-path successor
              (query_proj / z_world entropy), not another read-path letter.
  P0 met, C1 PASS  -> SWS-analog consolidation differentiates OCCUPIED
              context slots relative to online-writes-only, on the
              write-occupancy-corrected instrument. Supports SD-017,
              ARC-045, MECH-166. First genuine, non-confounded support for
              the slot-differentiation prediction in this lineage.
  P0 met, C1 FAIL  -> Genuine weakens for all three claims -- with P0 met,
              the write path, the waking control's write engagement, the
              occupancy floor, and the drift-neutralization calibration are
              ALL confirmed this time, so a FAIL here is not attributable to
              any of the five previously-identified confounds. Check
              per-seed action-class entropy as in 436d/436c before reading
              further (a still-degenerate waking entropy would point at the
              noise-floor magnitude, not at slot differentiation itself).

claim_ids: ["SD-017", "ARC-045", "MECH-166"]
experiment_purpose: "evidence"

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow
"""

import argparse
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_436f_sd017_mech166_sd016_armed_retest"
QUEUE_ID = "V3-EXQ-436f"
SUPERSEDES = "V3-EXQ-436e"
CLAIM_IDS = ["SD-017", "ARC-045", "MECH-166"]
EXPERIMENT_PURPOSE = "evidence"

# 436d substrate repair, held constant (unchanged -- still required so writes
# are content-bearing rather than homogenizing per the 436c defect).
CONTEXTMEMORY_GATED_CONTENT_WRITE = True

# --- 436f: SD-016 cue-conditioning production combination (V3-EXQ-922) -------
# Armed and HELD CONSTANT across all three conditions (a substrate property,
# not the manipulation -- sleep remains the manipulation). See module docstring
# "WHAT CHANGES vs 436e". sd016_enabled=True is a PREREQUISITE: the entire
# cue-tagger / ctxdiv apparatus in E1DeepPredictor is gated on it.
SD016_ENABLED = True
SD016_CUE_SLOT_TAGGER = True
SD016_CUE_SLOT_TAGGER_SELECTION = "gumbel"   # NOT "topk" -- eliminated by V3-EXQ-922
SD016_CONTEXT_DIVERGENCE_WEIGHT = 0.5
# Fixed detached divergence batch size per (seed, condition), matching the
# scale V3-EXQ-907/922 used for the ctxdiv training batches.
SD016_CTXDIV_BATCH_SIZE = 64
# P0: pooled |applied ctxdiv loss| across training must exceed this for the
# arming to count as engaged (an inert arming self-reports distinctly from an
# occupancy collapse). |loss| = weight * divergence; a genuinely engaged tagger
# produces divergence well above float noise, so a tiny floor separates "fired"
# from "never called".
SD016_CTXDIV_ENGAGED_FLOOR = 1e-9

# Pre-registered thresholds (unchanged from 436d/436c/436b/436a unless noted).
BASE_HARM_THRESHOLD = 0.05       # filter actions whose predicted harm exceeds this
CONTEXT_BETA = 0.8                # danger-score modulation strength
SLOT_DANGER_EMA_ALPHA = 0.05      # slot_danger_score EMA update rate

# Phase 2 substrate template (validated by V3-EXQ-265a; reused verbatim).
SD016_DIVERSIFICATION_WEIGHT = 0.5

# MECH-313 / ARC-065 noise floor -- unchanged.
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0        # matches REEAgent.select_action's own default

# Acceptance thresholds.
C1_N_SEEDS_REQUIRED = 3           # >= 3/5 seeds, sole PASS/FAIL gate (once P0 met)
C4_SLOT_SEPARATION_THRESHOLD = 0.3
C4_N_SEEDS_REQUIRED = 3
# Historical / legacy only -- calibrated against the PRE-436d write-gate
# defect (see docstring point 5). NOT recomputed against the repaired path
# and NOT used for any gating decision; retained purely so a reader
# comparing across 436a-436e sees where the old descriptive reference came
# from and why it is not treated as a live expectation here.
ARC045_ABS_COSINE_REFERENCE_LEGACY = 0.95

# P0 sleep-pass write-counter gate (unchanged concept from 436c/436d).
SWS_WRITES_FLOOR = 1.0
REM_ROLLOUTS_FLOOR = 1.0
# P0 NEW gates (436e DV repair).
WAKING_WRITEPATH_CALLS_FLOOR = 1.0
ADAM_DRIFT_NULL_TOLERANCE_SIGMA = 4.0
UNTOUCHED_BANK_NULL_DRAWS = 500
UNTOUCHED_BANK_NULL_SEED = 20260812  # fixed, reproducible; independent of experiment seeds

SLEEP_INTERVAL = 10
CONTEXT_SWITCH_EVERY = 5
TRAINING_EPISODES = 200
STEPS_PER_EPISODE = 150
EVAL_EPISODES_EACH = 30
MAX_HARM_BUF = 4000
SEEDS = [42, 7, 13, 100, 200]      # unchanged from 436/436a/436b/436c/436d for cross-lineage comparability

# (label, sws_enabled, rem_enabled, sd016_writepath_mode). NO_WRITES is a
# calibration-only negative control (docstring points 3/6), not part of the
# C1 comparison. WAKING_ONLY and SWS_THEN_REM share writepath_mode
# "sense_only" (docstring point 3) -- the write-path repair is held constant
# across the two conditions actually compared; sleep is the manipulation.
CONDITIONS_SPEC: List[Tuple[str, bool, bool, str]] = [
    ("NO_WRITES", False, False, "off"),
    ("WAKING_ONLY", False, False, "sense_only"),
    ("SWS_THEN_REM", True, True, "sense_only"),
]
CONDITIONS: List[str] = [c[0] for c in CONDITIONS_SPEC]
_COND_PARAMS: Dict[str, Tuple[bool, bool, str]] = {
    name: (sws_en, rem_en, wp_mode) for name, sws_en, rem_en, wp_mode in CONDITIONS_SPEC
}


# ------------------------------------------------------------------ #
# Env / agent helpers (unchanged from 436d)                                #
# ------------------------------------------------------------------ #

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=1,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.10,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000,
        size=10,
        num_hazards=8,
        num_resources=4,
        hazard_harm=0.05,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.15,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, sws_enabled: bool, rem_enabled: bool,
                use_sleep_loop: bool, writepath_mode: str) -> REEAgent:
    """Phase 2 substrate stack (unchanged from 436d) + MECH-313/ARC-065
    noise floor (unchanged) + 436e's two DV-repair substrate properties:
    writepath_mode (docstring point 3, per-condition per CONDITIONS_SPEC)
    and Adam-drift neutralization (docstring point 4, applied by the caller
    right after construction -- see _run_condition).
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        # Phase 2 substrate template (mechanically applied; constant across
        # all three conditions except sd016_writepath_mode, which is the
        # 436e DV-repair change -- see docstring point 3).
        sd016_writepath_mode=writepath_mode,
        sd016_diversification_weight=SD016_DIVERSIFICATION_WEIGHT,
        # 436f: SD-016 cue-conditioning armed, held constant across all three
        # conditions (see module docstring "WHAT CHANGES vs 436e"). sd016_enabled
        # is the prerequisite gating the whole apparatus; the ctxdiv weight is
        # inert without the training-loop wiring added in _run_episode.
        sd016_enabled=SD016_ENABLED,
        sd016_cue_slot_tagger=SD016_CUE_SLOT_TAGGER,
        sd016_cue_slot_tagger_selection=SD016_CUE_SLOT_TAGGER_SELECTION,
        sd016_context_divergence_weight=SD016_CONTEXT_DIVERGENCE_WEIGHT,
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_sd039_anchor_payload=True,
        # 436d write-path repair -- restores ContextMemory write_gate to a
        # modulator so writes are content-bearing. Constant across all three
        # conditions. Still required alongside 436e's occupancy fix.
        contextmemory_gated_content_write=CONTEXTMEMORY_GATED_CONTENT_WRITE,
        # SD-017 sleep phases (toggle per condition).
        sws_enabled=sws_enabled,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=rem_enabled,
        rem_attribution_steps=6,
        use_sleep_loop=use_sleep_loop,
        # MECH-313 / ARC-065 stochastic noise floor -- ON in every condition.
        use_noise_floor=USE_NOISE_FLOOR,
        noise_floor_alpha=NOISE_FLOOR_ALPHA,
        noise_floor_min_temperature=NOISE_FLOOR_MIN_TEMPERATURE,
    )
    agent = REEAgent(cfg)
    assert agent.noise_floor is not None, (
        "use_noise_floor=True did not construct agent.noise_floor -- "
        "REEConfig/REEAgent wiring regression; the diversity fix this "
        "experiment depends on is not live."
    )
    return agent


def _action_onehot(a_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, a_idx] = 1.0
    return v


def _effective_temperature(agent: REEAgent) -> float:
    """The same tonic-lift computation REEAgent.select_action() applies
    before e3.select() (ree_core/agent.py:7438-7444), applied here so this
    driver's own harm-based action scores get the same MECH-313 noise-floor
    diversity injection the substrate's own selection path would give them.
    """
    if agent.noise_floor is not None:
        return agent.noise_floor.compute_effective_temperature(
            baseline_temperature=BASELINE_TEMPERATURE, simulation_mode=False,
        )
    return BASELINE_TEMPERATURE


# ------------------------------------------------------------------ #
# Context-slot detection: read-side (visitation, unchanged) + write-side  #
# occupancy tracking (NEW -- 436e's core DV repair)                       #
# ------------------------------------------------------------------ #

def _active_slot_idx(agent: REEAgent, z_self: torch.Tensor,
                     z_world: torch.Tensor) -> int:
    """Determine which ContextMemory slot is most strongly ACTIVATED
    (read-side argmax over ContextMemory.read()'s soft-attention scores) by
    (z_self, z_world). Used only for the C4 slot_separation visitation
    statistic -- unrelated to, and a DIFFERENT index than, the write-side
    occupancy tracked below (write() selects by argMIN of query.memory, not
    this argmax). Unchanged from 436d.
    """
    with torch.no_grad():
        cm = agent.e1.context_memory
        state = torch.cat([z_self, z_world], dim=-1)
        query = cm.query_proj(state)
        keys = cm.key_proj(cm.memory)
        scores = torch.mm(query, keys.t()) / (cm.memory_dim ** 0.5)
        idx = int(scores.argmax(dim=-1).item())
    return idx


def _collect_z_world_batch(agent: REEAgent, env: CausalGridWorldV2,
                           n: int) -> torch.Tensor:
    """Collect n z_world vectors by stepping the agent through `env` with a
    fixed action, for use as a FIXED, DETACHED context-divergence batch (922's
    recipe). Returns [n, world_dim]. Called BEFORE the write-occupancy tracker
    is installed (see _run_condition), so the sense_only writes it triggers are
    a small, fixed pre-training warm-up and are NOT counted as run occupancy --
    the occupancy DV measures only training-time writes.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    zs: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()
            zs.append(latent.z_world.detach().clone())
            action_oh = _action_onehot(0, env.action_dim, agent.device)
            agent._last_action = action_oh
            _, _harm, done, _info, obs_dict = env.step(action_oh)
            if done:
                _, obs_dict = env.reset()
                agent.reset()
                agent.e1.reset_hidden_state()
    return torch.cat(zs, dim=0).detach()


class _WriteOccupancyTracker:
    """Accumulates the set of ContextMemory slot indices actually written to
    during a cell's run, plus a total write() call count. Populated by
    _install_write_tracker's instrumentation, which mirrors
    ContextMemory.write()'s own min_idx computation read-only.
    """

    def __init__(self) -> None:
        self.occupied: set = set()
        self.n_write_calls: int = 0

    def record(self, idx: int) -> None:
        self.occupied.add(idx)
        self.n_write_calls += 1


def _install_write_tracker(agent: REEAgent) -> _WriteOccupancyTracker:
    """Wrap agent.e1.context_memory's bound write() method with a thin,
    read-only occupancy tracker, without modifying ree_core source. All
    THREE of write()'s call sites in the substrate (sense()'s per-tick
    sense_only/both hook at agent.py:4847, E1.update_from_observation's
    train_only/both hook at e1_deep.py:690, and run_sws_schema_pass's
    replay writes at agent.py:11313) funnel through this SAME bound method,
    so wrapping it once here captures every write() invocation regardless
    of which caller triggered it.

    The wrapper recomputes min_idx via the IDENTICAL expression
    ContextMemory.write() itself uses (torch.no_grad(), read-only -- this
    duplicates a cheap [1, memory_dim] @ [memory_dim, num_slots] matmul, not
    a meaningful cost) so the tracked index is guaranteed to be the same one
    write() is about to mutate, then delegates to the original bound method
    unchanged.
    """
    cm = agent.e1.context_memory
    tracker = _WriteOccupancyTracker()
    orig_write = cm.write

    def _tracked_write(state: torch.Tensor) -> None:
        with torch.no_grad():
            query = cm.query_proj(state)
            scores = torch.mm(query, cm.memory.t())
            min_idx = int(scores.mean(0).argmin().item())
        tracker.record(min_idx)
        return orig_write(state)

    cm.write = _tracked_write  # instance-level override; class method untouched
    return tracker


def _compute_slot_cosine_sim_raw(agent: REEAgent) -> float:
    """OLD, CONFOUNDED whole-bank statistic (unchanged formula from
    436a-436d) -- averages over ALL slots including never-written ones, so
    it tracks content-similarity x occupancy-fraction rather than
    differentiation alone (failure_autopsy_V3-EXQ-436d-methodology-check_
    2026-08-07, F2-F4). Retained here ONLY for cross-letter audit
    continuity; NEVER used for gating in this experiment.
    """
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        n = mem.shape[0]
        normed = F.normalize(mem, dim=-1)
        sim = torch.mm(normed, normed.t())
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        return float(sim[mask].mean().item())


def _compute_slot_cosine_sim_occupied(agent: REEAgent,
                                       occupied_idx: List[int]
                                       ) -> Optional[float]:
    """CORRECTED primary DV (V3-EXQ-436e). Masks to slots that received
    >=1 write() call this run (per _install_write_tracker) before computing
    the mean off-diagonal cosine -- closes F2/F3/F4 of the methodology-check
    autopsy: never-written slots (whose pairwise cosine reflects only random
    init, ~0) can no longer dilute or invert the statistic, and the reading
    is no longer a function of HOW MANY slots got touched, only of how
    similar the touched ones are.

    Returns None (undefined) when fewer than 2 slots are occupied -- an
    off-diagonal pairwise mean needs at least a pair to exist. Callers must
    treat None as "not scoreable this seed", never as a numeric 0.
    """
    idx = sorted(occupied_idx)
    if len(idx) < 2:
        return None
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        sub = mem[idx]
        normed = F.normalize(sub, dim=-1)
        sim = torch.mm(normed, normed.t())
        n = sub.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        return float(sim[mask].mean().item())


def _derive_untouched_bank_null(num_slots: int, memory_dim: int,
                                 n_draws: int = UNTOUCHED_BANK_NULL_DRAWS,
                                 seed: int = UNTOUCHED_BANK_NULL_SEED
                                 ) -> Dict[str, float]:
    """Re-derive the untouched-bank whole-bank-cosine null distribution
    against THIS run's actual (num_slots, memory_dim) at run time, rather
    than hard-coding the methodology-check autopsy's own borrowed figure
    (docstring point 5). Draws n_draws fresh
    torch.randn(num_slots, memory_dim) * 0.01 initialisations -- matching
    ContextMemory.__init__'s own init recipe exactly -- and computes the
    whole-bank off-diagonal cosine mean for each, using a dedicated,
    explicitly-seeded generator so this derivation neither consumes nor is
    affected by the experiment's own per-seed RNG state.
    """
    gen = torch.Generator().manual_seed(seed)
    mask = ~torch.eye(num_slots, dtype=torch.bool)
    vals: List[float] = []
    for _ in range(n_draws):
        mem = torch.randn(num_slots, memory_dim, generator=gen) * 0.01
        normed = F.normalize(mem, dim=-1)
        sim = torch.mm(normed, normed.t())
        vals.append(float(sim[mask].mean().item()))
    return {
        "n_draws": n_draws,
        "mean": statistics.fmean(vals),
        "sd": statistics.pstdev(vals),
        "min": min(vals),
        "max": max(vals),
    }


# ------------------------------------------------------------------ #
# Action selection -- STOCHASTIC (noise-floor temperature-graded sample,   #
# unchanged from 436d)                                                     #
# ------------------------------------------------------------------ #

def _select_action_baseline(agent: REEAgent, z_world: torch.Tensor,
                             num_actions: int) -> Tuple[int, float]:
    """Temperature-graded softmax sample over predicted harm (low harm ->
    high selection probability), using the MECH-313 noise-floor effective
    temperature. Unchanged from 436d.
    """
    with torch.no_grad():
        harms: List[float] = []
        for a in range(num_actions):
            a_oh = _action_onehot(a, num_actions, z_world.device)
            zw_next = agent.e2.world_forward(z_world, a_oh)
            h = agent.e3.harm_eval(zw_next).mean().item()
            harms.append(h)
        eff_t = _effective_temperature(agent)
        harms_t = torch.tensor(harms, dtype=torch.float32)
        probs = F.softmax(-harms_t / eff_t, dim=0)
        best_a = int(torch.multinomial(probs, 1).item())
        best_h = harms[best_a]
    return best_a, best_h


def _select_action_context_cond(agent: REEAgent, z_world: torch.Tensor,
                                 num_actions: int, slot_danger_score: float,
                                 base_thresh: float, context_beta: float
                                 ) -> Tuple[int, float, float]:
    """Context-conditioned harm threshold action selection (DR-6 pathway),
    unchanged from 436d: effective threshold =
    base_thresh * (1 - context_beta * slot_danger_score); higher danger ->
    lower threshold -> more candidates filtered -> more cautious. Selection
    WITHIN the filtered (or, on empty filter, the full) candidate set is a
    noise-floor temperature-graded softmax sample.
    Returns (action_idx, chosen_harm, effective_threshold).
    """
    eff_thresh = base_thresh * max(0.1, 1.0 - context_beta * slot_danger_score)
    with torch.no_grad():
        harms: List[float] = []
        for a in range(num_actions):
            a_oh = _action_onehot(a, num_actions, z_world.device)
            zw_next = agent.e2.world_forward(z_world, a_oh)
            h = agent.e3.harm_eval(zw_next).mean().item()
            harms.append(h)
        eff_t = _effective_temperature(agent)
        filtered_idx = [a for a, h in enumerate(harms) if h <= eff_thresh]
        if filtered_idx:
            sub_harms = torch.tensor([harms[a] for a in filtered_idx], dtype=torch.float32)
            probs = F.softmax(-sub_harms / eff_t, dim=0)
            sel = int(torch.multinomial(probs, 1).item())
            best_a = filtered_idx[sel]
        else:
            harms_t = torch.tensor(harms, dtype=torch.float32)
            probs = F.softmax(-harms_t / eff_t, dim=0)
            best_a = int(torch.multinomial(probs, 1).item())
        best_h = harms[best_a]
    return best_a, float(best_h), float(eff_thresh)


# ------------------------------------------------------------------ #
# Episode runner (unchanged control flow from 436d)                        #
# ------------------------------------------------------------------ #

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    train: bool,
    is_dangerous_ep: bool,
    optimizer,
    harm_eval_opt,
    harm_buf_pos: List,
    harm_buf_neg: List,
    slot_danger_ema: List[float],
    use_context_cond: bool,
    ctxdiv_active: bool = False,
    z_safe_div: Optional[torch.Tensor] = None,
    z_dang_div: Optional[torch.Tensor] = None,
    ctxdiv_accum: Optional[List[float]] = None,
) -> Tuple[float, List[torch.Tensor], List[int], List[int]]:
    """Run single episode. Returns (harm_sum, z_world_list, slot_visits,
    action_seq). action_seq is recorded to compute a per-episode action-class
    entropy diagnostic. Updates slot_danger_ema in place when train=True.
    Control flow unchanged from 436d/436e; 436f ADDS the SD-016 H1
    context-divergence loss to the training total when ctxdiv_active (see module
    docstring "WHAT CHANGES vs 436e"). ctxdiv_accum[0] accumulates |applied
    ctxdiv loss| across steps for the sd016_arming_engaged P0 check.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    ep_harm = 0.0
    z_world_list: List[torch.Tensor] = []
    slot_visits: List[int] = []
    action_seq: List[int] = []

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        agent.clock.advance()
        z_self = latent.z_self.detach().clone()
        z_world = latent.z_world.detach().clone()
        z_world_list.append(z_world)

        # Mirrors _e1_tick's append+trim (ree_core/agent.py:4900-4904) and
        # theta_buffer push (ree_core/agent.py:4945) directly -- this
        # driver's manual harm-based action-selection loop calls
        # agent.sense() directly and never agent.act(), so both stores
        # (which run_sws_schema_pass / run_rem_attribution_pass sample from)
        # would otherwise stay permanently empty. Unchanged from 436b/436c/
        # 436d; see 436b's docstring for full derivation.
        agent._self_experience_buffer.append(z_self)
        agent._world_experience_buffer.append(z_world)
        if len(agent._self_experience_buffer) > 1000:
            del agent._self_experience_buffer[:-1000]
        if len(agent._world_experience_buffer) > 1000:
            del agent._world_experience_buffer[:-1000]
        agent.theta_buffer.update(z_world, z_self)

        slot_idx = _active_slot_idx(agent, z_self, z_world)
        slot_visits.append(slot_idx)

        if use_context_cond:
            danger = slot_danger_ema[slot_idx]
            action_idx, _, _ = _select_action_context_cond(
                agent, z_world, env.action_dim, danger,
                BASE_HARM_THRESHOLD, CONTEXT_BETA,
            )
        else:
            action_idx, _ = _select_action_baseline(agent, z_world, env.action_dim)
        action_seq.append(action_idx)

        action_oh = _action_onehot(action_idx, env.action_dim, agent.device)
        agent._last_action = action_oh

        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0
        if is_harm:
            ep_harm += abs(float(harm_signal))

        if train:
            target = 1.0 if is_dangerous_ep else 0.0
            slot_danger_ema[slot_idx] = (
                (1.0 - SLOT_DANGER_EMA_ALPHA) * slot_danger_ema[slot_idx]
                + SLOT_DANGER_EMA_ALPHA * target
            )

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss
            # 436f: SD-016 H1 context-divergence auxiliary loss (V3-EXQ-907/922).
            # compute_context_divergence_loss returns -weight*divergence (already
            # weighted and sign-flipped to REWARD divergence), ready to ADD.
            # z_safe_div/z_dang_div are fixed detached batches collected once per
            # (seed, condition) before training. Held constant across all three
            # conditions -- a substrate property, not the sleep manipulation.
            if ctxdiv_active and z_safe_div is not None and z_dang_div is not None:
                div_loss = agent.e1.compute_context_divergence_loss(z_safe_div, z_dang_div)
                total = total + div_loss
                if ctxdiv_accum is not None:
                    ctxdiv_accum[0] += abs(float(div_loss.item()))
            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if is_harm:
                harm_buf_pos.append(z_world)
            else:
                harm_buf_neg.append(z_world)

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target_t = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval_head(zw_b)
                h_loss = F.binary_cross_entropy_with_logits(pred, target_t)
                harm_eval_opt.zero_grad()
                h_loss.backward()
                harm_eval_opt.step()

        if done:
            break

    return ep_harm, z_world_list, slot_visits, action_seq


def _action_class_entropy(action_seq: List[int], num_actions: int) -> float:
    """Shannon entropy (nats) of the realized action-class distribution over
    one episode. Unchanged from 436d.
    """
    if not action_seq:
        return 0.0
    counts = [0] * num_actions
    for a in action_seq:
        counts[a % num_actions] += 1
    total = float(len(action_seq))
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p)
    return float(ent)


# ------------------------------------------------------------------ #
# Condition runner                                                     #
# ------------------------------------------------------------------ #

def _run_condition(
    seed: int,
    condition: str,
    training_episodes: int,
    steps_per_episode: int,
    eval_episodes_each: int,
    zg: ZGoalStreamAccumulator,
    verbose: bool = True,
) -> Dict:
    sws_en, rem_en, writepath_mode = _COND_PARAMS[condition]
    use_sleep_loop = sws_en or rem_en  # ON for SWS_THEN_REM; OFF otherwise
    use_context_cond = condition == "SWS_THEN_REM"   # DR-6 pathway only here (unchanged from 436d)

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe, sws_en, rem_en, use_sleep_loop, writepath_mode)

    # 436f: SD-016 H1 context-divergence arming. Collect FIXED, DETACHED
    # safe-context and dangerous-context z_world batches BEFORE the write-
    # occupancy tracker is installed, so the small sense_only warm-up writes
    # this triggers are not counted as run occupancy (they warm the encoder;
    # the DV counts training-time writes only). Held constant across all three
    # conditions. compute_context_divergence_loss is a no-op (zero, no graph)
    # unless the tagger is on and the weight > 0, so gathering the batches
    # unconditionally is safe -- ctxdiv_active gates whether the loss is added.
    ctxdiv_active = (
        SD016_ENABLED and SD016_CUE_SLOT_TAGGER and SD016_CONTEXT_DIVERGENCE_WEIGHT > 0.0
    )
    z_safe_div: Optional[torch.Tensor] = None
    z_dang_div: Optional[torch.Tensor] = None
    if ctxdiv_active:
        z_safe_div = _collect_z_world_batch(agent, env_safe, SD016_CTXDIV_BATCH_SIZE)
        z_dang_div = _collect_z_world_batch(agent, env_dang, SD016_CTXDIV_BATCH_SIZE)
    ctxdiv_accum: List[float] = [0.0]

    # 436e DV-repair point 4: Adam-drift neutralization, held constant across
    # ALL THREE conditions. write() still mutates memory.data directly under
    # its own torch.no_grad() block regardless of requires_grad, so writes
    # are unaffected -- only gradient-descent perturbation between writes is
    # suppressed.
    agent.e1.context_memory.memory.requires_grad_(False)

    # 436e DV-repair point 1: install the write-occupancy tracker BEFORE any
    # training so it captures every write() call for the whole cell.
    write_tracker = _install_write_tracker(agent)

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n and "context_memory.memory" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=1e-3)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    num_slots = agent.e1.context_memory.num_slots
    slot_danger_ema: List[float] = [0.5] * num_slots

    per_ep_harm_safe: List[float] = []
    per_ep_harm_dang: List[float] = []
    per_ep_action_entropy: List[float] = []
    slot_visit_safe_count: List[int] = [0] * num_slots
    slot_visit_dang_count: List[int] = [0] * num_slots
    sleep_passes = 0
    cum_train_pos = 0  # cumulative harm_eval_head TRAINING label counts
    cum_train_neg = 0  # (pre-MAX_HARM_BUF-trim, so not diluted by the cap)

    cum_sws_n_writes = 0.0
    cum_rem_n_rollouts = 0.0
    last_sws_slot_diversity = 0.0
    last_rem_mean_harm_terrain = 0.0

    agent.train()
    for ep in range(training_episodes):
        block = ep // CONTEXT_SWITCH_EVERY
        is_safe_ep = (block % 2 == 0)
        env = env_safe if is_safe_ep else env_dang

        _len_pos_before, _len_neg_before = len(harm_buf_pos), len(harm_buf_neg)
        ep_harm, _z_list, slot_visits, action_seq = _run_episode(
            agent, env, steps_per_episode,
            train=True,
            is_dangerous_ep=(not is_safe_ep),
            optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
            ctxdiv_active=ctxdiv_active,
            z_safe_div=z_safe_div,
            z_dang_div=z_dang_div,
            ctxdiv_accum=ctxdiv_accum,
        )
        harm_rate = ep_harm / steps_per_episode
        per_ep_action_entropy.append(_action_class_entropy(action_seq, env.action_dim))
        cum_train_pos += len(harm_buf_pos) - _len_pos_before
        cum_train_neg += len(harm_buf_neg) - _len_neg_before
        if is_safe_ep:
            per_ep_harm_safe.append(harm_rate)
            for s in slot_visits:
                slot_visit_safe_count[s] += 1
        else:
            per_ep_harm_dang.append(harm_rate)
            for s in slot_visits:
                slot_visit_dang_count[s] += 1

        if len(harm_buf_pos) > MAX_HARM_BUF:
            harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
        if len(harm_buf_neg) > MAX_HARM_BUF:
            harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

        if (sws_en or rem_en) and (ep + 1) % SLEEP_INTERVAL == 0 and ep > 0:
            if rem_en:
                sleep_metrics = agent.run_sleep_cycle()
            else:
                # Dead branch in this design (sws_en == rem_en always for
                # the CONDITIONS_SPEC above), kept for robustness against a
                # future asymmetric-phase variant. Unchanged from 436d.
                sleep_metrics = agent.run_sws_schema_pass()
            cum_sws_n_writes += float(sleep_metrics.get("sws_n_writes", 0.0))
            cum_rem_n_rollouts += float(sleep_metrics.get("rem_n_rollouts", 0.0))
            last_sws_slot_diversity = float(
                sleep_metrics.get("sws_slot_diversity", last_sws_slot_diversity)
            )
            last_rem_mean_harm_terrain = float(
                sleep_metrics.get("rem_mean_harm_terrain", last_rem_mean_harm_terrain)
            )
            sleep_passes += 1

        if (ep + 1) % 50 == 0:
            print(f"  [train] label seed={seed} cond={condition} "
                  f"ep {ep+1}/{training_episodes} "
                  f"harm_safe_ema={(sum(per_ep_harm_safe[-10:])/max(len(per_ep_harm_safe[-10:]),1)):.4f} "
                  f"harm_dang_ema={(sum(per_ep_harm_dang[-10:])/max(len(per_ep_harm_dang[-10:]),1)):.4f} "
                  f"action_entropy_ema={(sum(per_ep_action_entropy[-10:])/max(len(per_ep_action_entropy[-10:]),1)):.4f}",
                  flush=True)

    safe_tot = float(sum(slot_visit_safe_count))
    dang_tot = float(sum(slot_visit_dang_count))
    if safe_tot > 0 and dang_tot > 0:
        safe_dist = [c / safe_tot for c in slot_visit_safe_count]
        dang_dist = [c / dang_tot for c in slot_visit_dang_count]
        slot_separation = float(sum(abs(s - d) for s, d in zip(safe_dist, dang_dist)))
    else:
        slot_separation = 0.0

    # 436e DV repair: read BOTH statistics at the same point 436d read its
    # single one (right after training, before agent.eval()'s eval episodes
    # continue writing).
    final_slot_sim_raw = _compute_slot_cosine_sim_raw(agent)
    occupied_idx = sorted(write_tracker.occupied)
    n_occupied_slots = len(occupied_idx)
    n_write_calls = write_tracker.n_write_calls
    final_slot_sim_occupied = _compute_slot_cosine_sim_occupied(agent, occupied_idx)

    train_action_entropy_mean = (
        sum(per_ep_action_entropy) / max(1, len(per_ep_action_entropy))
    )

    zg.observe(agent)

    agent.eval()
    eval_harm_safe: List[float] = []
    eval_harm_dang: List[float] = []
    eval_z_safe: List[torch.Tensor] = []
    eval_z_dang: List[torch.Tensor] = []

    for _ in range(eval_episodes_each):
        h_s, zs, _, _ = _run_episode(
            agent, env_safe, steps_per_episode,
            train=False, is_dangerous_ep=False,
            optimizer=optimizer, harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos, harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
        )
        eval_harm_safe.append(h_s / steps_per_episode)
        eval_z_safe.extend(zs)

    for _ in range(eval_episodes_each):
        h_d, zd, _, _ = _run_episode(
            agent, env_dang, steps_per_episode,
            train=False, is_dangerous_ep=True,
            optimizer=optimizer, harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos, harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
        )
        eval_harm_dang.append(h_d / steps_per_episode)
        eval_z_dang.extend(zd)

    with torch.no_grad():
        n_samp = min(len(eval_z_safe), len(eval_z_dang), 200)
        if n_samp > 0:
            zs_s = torch.cat(eval_z_safe[:n_samp], dim=0)
            zd_s = torch.cat(eval_z_dang[:n_samp], dim=0)
            he_safe = float(agent.e3.harm_eval(zs_s).mean().item())
            he_dang = float(agent.e3.harm_eval(zd_s).mean().item())
        else:
            he_safe = 0.0
            he_dang = 0.0
    harm_discrim = he_dang - he_safe

    harm_safe = sum(eval_harm_safe) / max(1, len(eval_harm_safe))
    harm_dang = sum(eval_harm_dang) / max(1, len(eval_harm_dang))

    # 436f: SD-016 arming diagnostics. sd016_ctxdiv_applied_total is the pooled
    # |applied ctxdiv loss| across every training step (the P0
    # sd016_arming_engaged signal that arming actually fired). The final raw
    # divergence and cue-slot selection entropy describe how differentiated the
    # trained read-side selection became -- read-side, so NOT a guarantee of
    # lifted write-slot occupancy (see module docstring caveat).
    sd016_ctxdiv_applied_total = float(ctxdiv_accum[0])
    sd016_ctxdiv_final_divergence = None
    sd016_cue_slot_selection_entropy = None
    if ctxdiv_active and z_safe_div is not None and z_dang_div is not None:
        with torch.no_grad():
            _dl = float(agent.e1.compute_context_divergence_loss(z_safe_div, z_dang_div).item())
            # loss = -weight * divergence; recover raw unweighted divergence.
            sd016_ctxdiv_final_divergence = (
                (-_dl / SD016_CONTEXT_DIVERGENCE_WEIGHT)
                if SD016_CONTEXT_DIVERGENCE_WEIGHT > 0 else 0.0
            )
            # Mean Shannon entropy (nats) of the tagger's per-slot selection over
            # the safe batch; a fully-collapsed selection -> ~0, uniform -> ln(16).
            _ = agent.e1.extract_cue_context(z_safe_div)
            _w = getattr(agent.e1, "_last_cue_slot_weights", None)
            if _w is not None:
                _p = _w.clamp(min=1e-12)
                sd016_cue_slot_selection_entropy = float(
                    (-(_p * _p.log()).sum(dim=-1)).mean().item()
                )

    if verbose:
        print(f"  [seed={seed} {condition}] "
              f"slot_sim_occupied={final_slot_sim_occupied} "
              f"slot_sim_raw={final_slot_sim_raw:.4f} "
              f"n_occupied={n_occupied_slots} "
              f"n_write_calls={n_write_calls} "
              f"slot_sep={slot_separation:.3f} "
              f"harm_safe={harm_safe:.4f} "
              f"harm_dang={harm_dang:.4f} "
              f"discrim={harm_discrim:.4f} "
              f"action_entropy_mean={train_action_entropy_mean:.4f} "
              f"ctxdiv_applied={sd016_ctxdiv_applied_total:.6f} "
              f"ctxdiv_final={sd016_ctxdiv_final_divergence} "
              f"cue_sel_entropy={sd016_cue_slot_selection_entropy} "
              f"sleep_passes={sleep_passes} "
              f"sws_n_writes_total={cum_sws_n_writes:.1f} "
              f"rem_n_rollouts_total={cum_rem_n_rollouts:.1f}",
              flush=True)

    # Per-condition verdict (progress-instrumentation / runner-ETA purposes
    # only; the experiment-level PASS/FAIL is the aggregate C1 gate (plus
    # the P0 readiness gate) computed once across all seeds in __main__).
    verdict = "PASS" if (harm_dang < 0.04 and harm_safe < 0.04) else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "condition": condition,
        "sd016_writepath_mode": writepath_mode,
        # NEW (436e) -- corrected primary DV. None when <2 occupied slots.
        "slot_cosine_sim_occupied_only": final_slot_sim_occupied,
        "n_occupied_slots": n_occupied_slots,
        "n_write_calls": n_write_calls,
        # OLD, confounded whole-bank stat -- audit-only, never gating.
        "slot_cosine_sim_raw_whole_bank": float(final_slot_sim_raw),
        "slot_separation": float(slot_separation),
        "harm_rate_safe": float(harm_safe),
        "harm_rate_dangerous": float(harm_dang),
        "harm_discrimination": float(harm_discrim),
        "harm_eval_safe": float(he_safe),
        "harm_eval_dangerous": float(he_dang),
        "slot_danger_ema": [float(x) for x in slot_danger_ema],
        "slot_visit_safe_count": slot_visit_safe_count,
        "slot_visit_dang_count": slot_visit_dang_count,
        "train_harm_safe_final": float(sum(per_ep_harm_safe[-20:]) / max(1, len(per_ep_harm_safe[-20:]))),
        "train_harm_dang_final": float(sum(per_ep_harm_dang[-20:]) / max(1, len(per_ep_harm_dang[-20:]))),
        "train_action_class_entropy_mean": float(train_action_entropy_mean),
        "sleep_passes": sleep_passes,
        "sleep_cycle_sws_n_writes_total": float(cum_sws_n_writes),
        "sleep_cycle_rem_n_rollouts_total": float(cum_rem_n_rollouts),
        "sleep_cycle_sws_slot_diversity_last": float(last_sws_slot_diversity),
        "sleep_cycle_rem_mean_harm_terrain_last": float(last_rem_mean_harm_terrain),
        # 436f: SD-016 arming engagement + read-side differentiation diagnostics.
        "sd016_ctxdiv_active": bool(ctxdiv_active),
        "sd016_ctxdiv_applied_total": sd016_ctxdiv_applied_total,
        "sd016_ctxdiv_final_divergence": sd016_ctxdiv_final_divergence,
        "sd016_cue_slot_selection_entropy": sd016_cue_slot_selection_entropy,
        "effective_temperature_last": float(_effective_temperature(agent)),
        "noise_floor_state": agent.noise_floor.get_state() if agent.noise_floor is not None else None,
        "label_balance": {
            "harm_eval_head_train_pos_frac": (
                float(cum_train_pos) / max(1, cum_train_pos + cum_train_neg)
            ),
            "harm_eval_head_train_n_pos": cum_train_pos,
            "harm_eval_head_train_n_neg": cum_train_neg,
        },
    }


# ------------------------------------------------------------------ #
# Run                                                                   #
# ------------------------------------------------------------------ #

def run(dry_run: bool = False) -> Tuple[dict, ZGoalStreamAccumulator]:
    zg = ZGoalStreamAccumulator()

    if dry_run:
        print("[DRY RUN] MECH-166 occupied-slot DV-repair smoke "
              "(seed=42, 3 conditions, enough episodes for >=1 sleep pass "
              "and >=2 occupied slots)", flush=True)
        smoke_ok = True
        smoke_results = []
        # NOTE: training_episodes must clear SLEEP_INTERVAL (10) so
        # SWS_THEN_REM actually triggers >=1 sleep pass during the smoke --
        # unchanged value from 436d, which already validated this margin.
        smoke_training_episodes = 11
        smoke_steps_per_episode = 20
        try:
            for cond in CONDITIONS:
                print(f"Seed 42 Condition {cond}", flush=True)
                _sws_en, _rem_en, _wp_mode = _COND_PARAMS[cond]
                config_slice = {
                    "seed": 42, "condition": cond,
                    "training_episodes": smoke_training_episodes,
                    "steps_per_episode": smoke_steps_per_episode,
                    "eval_episodes_each": 2,
                    "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
                    "sd016_writepath_mode": _wp_mode,
                    "context_memory_memory_requires_grad": False,
                    "sd016_enabled": SD016_ENABLED,
                    "sd016_cue_slot_tagger": SD016_CUE_SLOT_TAGGER,
                    "sd016_cue_slot_tagger_selection": SD016_CUE_SLOT_TAGGER_SELECTION,
                    "sd016_context_divergence_weight": SD016_CONTEXT_DIVERGENCE_WEIGHT,
                }
                with arm_cell(42, config_slice=config_slice, script_path=Path(__file__)) as cell:
                    r = _run_condition(
                        seed=42, condition=cond,
                        training_episodes=smoke_training_episodes,
                        steps_per_episode=smoke_steps_per_episode,
                        eval_episodes_each=2,
                        zg=zg,
                        verbose=False,
                    )
                    cell.stamp(r)
                smoke_results.append(r)
                print(f"  {cond}: slot_sim_occupied={r['slot_cosine_sim_occupied_only']} "
                      f"slot_sim_raw={r['slot_cosine_sim_raw_whole_bank']:.4f} "
                      f"n_occupied={r['n_occupied_slots']} "
                      f"n_write_calls={r['n_write_calls']} "
                      f"slot_sep={r['slot_separation']:.3f} "
                      f"harm_safe={r['harm_rate_safe']:.4f} "
                      f"harm_dang={r['harm_rate_dangerous']:.4f} "
                      f"action_entropy={r['train_action_class_entropy_mean']:.4f} "
                      f"ctxdiv_active={r['sd016_ctxdiv_active']} "
                      f"ctxdiv_applied={r['sd016_ctxdiv_applied_total']:.6f} "
                      f"sleep_passes={r['sleep_passes']} "
                      f"sws_n_writes_total={r['sleep_cycle_sws_n_writes_total']:.1f} "
                      f"rem_n_rollouts_total={r['sleep_cycle_rem_n_rollouts_total']:.1f}")

            required_keys = {"slot_cosine_sim_occupied_only", "slot_cosine_sim_raw_whole_bank",
                             "n_occupied_slots", "n_write_calls",
                             "slot_separation", "harm_rate_safe", "harm_rate_dangerous",
                             "train_action_class_entropy_mean",
                             "sleep_cycle_sws_n_writes_total",
                             "sleep_cycle_rem_n_rollouts_total",
                             "sd016_ctxdiv_active", "sd016_ctxdiv_applied_total",
                             "arm_fingerprint"}
            for r in smoke_results:
                missing = required_keys - set(r.keys())
                if missing:
                    print(f"  [SMOKE] FAIL: condition {r['condition']} missing keys {missing}")
                    smoke_ok = False

            by_c = {r["condition"]: r for r in smoke_results}
            no_writes = by_c.get("NO_WRITES")
            waking = by_c.get("WAKING_ONLY")
            sws_then_rem = by_c.get("SWS_THEN_REM")

            if sws_then_rem is None or waking is None or no_writes is None:
                print("  [SMOKE] FAIL: missing one of NO_WRITES/WAKING_ONLY/SWS_THEN_REM")
                smoke_ok = False
            else:
                if sws_then_rem["sleep_passes"] < 1:
                    print("  [SMOKE] FAIL: SWS_THEN_REM triggered 0 sleep passes -- "
                          "smoke config does not exercise the sleep path at all")
                    smoke_ok = False
                if sws_then_rem["sleep_cycle_sws_n_writes_total"] <= 0.0:
                    print("  [SMOKE] FAIL: sleep_cycle_sws_n_writes_total="
                          f"{sws_then_rem['sleep_cycle_sws_n_writes_total']} -- still zero")
                    smoke_ok = False
                if sws_then_rem["sleep_cycle_rem_n_rollouts_total"] <= 0.0:
                    print("  [SMOKE] FAIL: sleep_cycle_rem_n_rollouts_total="
                          f"{sws_then_rem['sleep_cycle_rem_n_rollouts_total']} -- still zero")
                    smoke_ok = False
                # THE key NEW verification: waking writepath must actually engage.
                if waking["n_write_calls"] <= 0:
                    print("  [SMOKE] FAIL: WAKING_ONLY n_write_calls=0 -- "
                          "sd016_writepath_mode='sense_only' did not fire; "
                          "the F1 fix (docstring point 3) is not live")
                    smoke_ok = False
                # Occupancy sufficiency (F6 fix check).
                if waking["n_occupied_slots"] < 2 or sws_then_rem["n_occupied_slots"] < 2:
                    print("  [SMOKE] FAIL: fewer than 2 occupied slots in "
                          f"WAKING_ONLY ({waking['n_occupied_slots']}) or "
                          f"SWS_THEN_REM ({sws_then_rem['n_occupied_slots']}) -- "
                          "occupied-only cosine would be undefined this seed")
                    smoke_ok = False
                if waking["slot_cosine_sim_occupied_only"] is None or \
                   sws_then_rem["slot_cosine_sim_occupied_only"] is None:
                    print("  [SMOKE] FAIL: slot_cosine_sim_occupied_only is None "
                          "for WAKING_ONLY or SWS_THEN_REM despite reported n_occupied>=2")
                    smoke_ok = False
                # THE key 436f verification: SD-016 ctxdiv arming must engage on
                # the two C1 comparison arms (pooled |applied ctxdiv loss| > 0).
                if not waking["sd016_ctxdiv_active"] or not sws_then_rem["sd016_ctxdiv_active"]:
                    print("  [SMOKE] FAIL: sd016_ctxdiv_active is False for a C1 arm -- "
                          "SD-016 arming (enabled/tagger/weight) not live")
                    smoke_ok = False
                if waking["sd016_ctxdiv_applied_total"] <= 0.0 or \
                   sws_then_rem["sd016_ctxdiv_applied_total"] <= 0.0:
                    print("  [SMOKE] FAIL: sd016_ctxdiv_applied_total<=0 for a C1 arm "
                          f"(WAKING_ONLY={waking['sd016_ctxdiv_applied_total']}, "
                          f"SWS_THEN_REM={sws_then_rem['sd016_ctxdiv_applied_total']}) -- "
                          "compute_context_divergence_loss never fired; arming is inert")
                    smoke_ok = False
                # NO_WRITES calibration sanity: with writes truly disabled and
                # requires_grad_(False), n_write_calls must be exactly 0.
                if no_writes["n_write_calls"] != 0:
                    print("  [SMOKE] FAIL: NO_WRITES n_write_calls="
                          f"{no_writes['n_write_calls']} -- expected exactly 0 "
                          "(sd016_writepath_mode='off', no sleep)")
                    smoke_ok = False

            if smoke_ok:
                print("[DRY RUN] PASS - all three conditions wire correctly; "
                      "WAKING_ONLY now has a real write path, occupied-slot "
                      "cosine is computable for both comparison arms, and "
                      "NO_WRITES stays at zero write calls as expected")
            else:
                print("[DRY RUN] FAIL - check above for missing keys, unfired "
                      "write paths, or insufficient occupancy")
        except Exception as exc:
            print(f"[DRY RUN] FAIL - exception during smoke: {exc!r}")
            smoke_ok = False

        return {
            "outcome": "PASS" if smoke_ok else "FAIL",
            "status": "PASS" if smoke_ok else "FAIL",
        }, zg

    t0 = time.time()
    print(f"{QUEUE_ID} {EXPERIMENT_TYPE}", flush=True)

    arm_results: List[Dict] = []
    for seed in SEEDS:
        print(f"Seed {seed}")
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}")
            _sws_en, _rem_en, _wp_mode = _COND_PARAMS[cond]
            config_slice = {
                "seed": seed, "condition": cond,
                "training_episodes": TRAINING_EPISODES,
                "steps_per_episode": STEPS_PER_EPISODE,
                "eval_episodes_each": EVAL_EPISODES_EACH,
                "use_noise_floor": USE_NOISE_FLOOR,
                "noise_floor_alpha": NOISE_FLOOR_ALPHA,
                "noise_floor_min_temperature": NOISE_FLOOR_MIN_TEMPERATURE,
                "sd016_diversification_weight": SD016_DIVERSIFICATION_WEIGHT,
                "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
                "sd016_writepath_mode": _wp_mode,
                "context_memory_memory_requires_grad": False,
                "sd016_enabled": SD016_ENABLED,
                "sd016_cue_slot_tagger": SD016_CUE_SLOT_TAGGER,
                "sd016_cue_slot_tagger_selection": SD016_CUE_SLOT_TAGGER_SELECTION,
                "sd016_context_divergence_weight": SD016_CONTEXT_DIVERGENCE_WEIGHT,
                "sd016_ctxdiv_batch_size": SD016_CTXDIV_BATCH_SIZE,
            }
            with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
                r = _run_condition(
                    seed=seed, condition=cond,
                    training_episodes=TRAINING_EPISODES,
                    steps_per_episode=STEPS_PER_EPISODE,
                    eval_episodes_each=EVAL_EPISODES_EACH,
                    zg=zg,
                )
                cell.stamp(r)
            arm_results.append(r)

    elapsed = time.time() - t0

    def by_cond(c):
        return [r for r in arm_results if r["condition"] == c]

    no_writes_r = by_cond("NO_WRITES")
    waking = by_cond("WAKING_ONLY")
    sws_r = by_cond("SWS_THEN_REM")

    # --- Re-derived untouched-bank null (docstring point 5) -----------------
    _num_slots = arm_results[0]["slot_danger_ema"].__len__() if arm_results else 16
    # memory_dim is not directly in the row; derive from CausalGridWorldV2's
    # fixed E1 config (world_dim=32 -> memory_dim defaults to 128 in
    # ContextMemory's own signature, matching every prior letter's usage --
    # no config in this driver ever overrides memory_dim).
    _memory_dim = 128
    untouched_bank_null = _derive_untouched_bank_null(_num_slots, _memory_dim)

    # --- P0 READINESS GATE (6 checks; all must clear) ------------------------
    # 5 inherited from 436e; +1 (sd016_arming_engaged) new in 436f.
    pooled_sws_n_writes = sum(r["sleep_cycle_sws_n_writes_total"] for r in sws_r)
    pooled_rem_n_rollouts = sum(r["sleep_cycle_rem_n_rollouts_total"] for r in sws_r)
    pooled_waking_write_calls = sum(r["n_write_calls"] for r in waking)
    # 436f: pooled |applied ctxdiv loss| across the two C1 comparison arms
    # (WAKING_ONLY + SWS_THEN_REM) over all seeds. > floor confirms SD-016 H1
    # arming actually engaged this run -- so an inert arming self-reports as a
    # distinct P0 failure rather than masquerading as an occupancy collapse.
    pooled_sd016_ctxdiv_applied = sum(
        r["sd016_ctxdiv_applied_total"] for r in (waking + sws_r)
    )

    n_scoreable_seeds = sum(
        1 for w_r, s_r in zip(waking, sws_r)
        if w_r["n_occupied_slots"] >= 2 and s_r["n_occupied_slots"] >= 2
    )

    _null_mean = untouched_bank_null["mean"]
    _null_sd = max(untouched_bank_null["sd"], 1e-9)
    no_writes_abs_z = [
        abs(r["slot_cosine_sim_raw_whole_bank"] - _null_mean) / _null_sd
        for r in no_writes_r
    ]
    max_no_writes_abs_z = max(no_writes_abs_z) if no_writes_abs_z else float("inf")

    readiness_checks = [
        {
            "name": "sws_context_memory_writes_occur",
            "measured": pooled_sws_n_writes,
            "threshold": SWS_WRITES_FLOOR,
            "direction": "lower",
            "control": (
                "sum of ContextMemory.write() calls (sws_n_writes, returned "
                "by agent.run_sws_schema_pass() via run_sleep_cycle()) "
                "across every SWS_THEN_REM sleep pass, pooled across all "
                f"{len(SEEDS)} seeds -- must be > 0 for the C1 comparison "
                "to test the mechanism rather than an unrecorded no-op. "
                "Unchanged from 436c/436d."
            ),
        },
        {
            "name": "rem_attribution_rollouts_occur",
            "measured": pooled_rem_n_rollouts,
            "threshold": REM_ROLLOUTS_FLOOR,
            "direction": "lower",
            "control": (
                "sum of hippocampal replay rollouts (rem_n_rollouts) across "
                "every SWS_THEN_REM sleep pass, pooled across all "
                f"{len(SEEDS)} seeds. Unchanged from 436c/436d."
            ),
        },
        {
            "name": "waking_writepath_engaged",
            "measured": pooled_waking_write_calls,
            "threshold": WAKING_WRITEPATH_CALLS_FLOOR,
            "direction": "lower",
            "control": (
                "sum of ContextMemory.write() calls fired by the per-tick "
                "sense_only hook (agent.py:4847) across all WAKING_ONLY "
                f"seeds, pooled across all {len(SEEDS)} seeds -- NEW in "
                "436e. Confirms the docstring point-3 fix (giving "
                "WAKING_ONLY a real write path via sd016_writepath_mode="
                "'sense_only') actually engaged this run; F1 of the "
                "methodology-check autopsy found this pooled to exactly 0 "
                "in 436d."
            ),
        },
        {
            "name": "sufficient_occupancy_for_c1",
            "measured": n_scoreable_seeds,
            "threshold": C1_N_SEEDS_REQUIRED,
            "direction": "lower",
            "control": (
                "count of seeds (of "
                f"{len(SEEDS)}) where BOTH WAKING_ONLY and SWS_THEN_REM "
                "have >=2 occupied slots (the minimum for an off-diagonal "
                "pairwise mean to exist) -- NEW in 436e, the direct fix for "
                "F6 (seeds 13/200 sat inside the untouched-bank null in "
                "436d, making C1 unsatisfiable there by construction and "
                "silently shrinking the effective denominator below the "
                "registered 3/5)."
            ),
        },
        {
            "name": "adam_drift_neutralized",
            "measured": max_no_writes_abs_z,
            "threshold": ADAM_DRIFT_NULL_TOLERANCE_SIGMA,
            "direction": "upper",
            "control": (
                "max |z-score| of the NO_WRITES calibration arm's raw "
                "whole-bank cosine (slot_cosine_sim_raw_whole_bank) against "
                "the freshly-derived untouched-bank null "
                f"(mean={_null_mean:.6f}, sd={_null_sd:.6f}, "
                f"n_draws={untouched_bank_null['n_draws']}), across all "
                f"{len(SEEDS)} seeds -- NEW in 436e, the empirical check "
                "for docstring points 4/5/6 (F5: WAKING_ONLY's 436d spread "
                "0.0009-0.47 was fully reproduced by Adam drift alone on "
                "context_memory.memory with zero write() calls). "
                "context_memory.memory.requires_grad_(False) plus zero "
                "write() calls on this arm should hold its raw cosine "
                "inside the null; a large z here means something is "
                "perturbing the bank that the calibration model does not "
                "account for."
            ),
        },
        {
            "name": "sd016_arming_engaged",
            "measured": pooled_sd016_ctxdiv_applied,
            "threshold": SD016_CTXDIV_ENGAGED_FLOOR,
            "direction": "lower",
            "control": (
                "pooled |applied SD-016 H1 context-divergence loss| "
                "(sd016_ctxdiv_applied_total) across the two C1 comparison "
                "arms (WAKING_ONLY + SWS_THEN_REM) over all "
                f"{len(SEEDS)} seeds -- NEW in 436f. Confirms the SD-016 "
                "arming (sd016_enabled + cue_slot_tagger='gumbel' + "
                "context_divergence_weight=0.5, with the ctxdiv training-loop "
                "wiring this driver adds) actually ENGAGED this run. > floor "
                "means compute_context_divergence_loss was called and "
                "returned a non-trivial weighted divergence every step; a "
                "zero here means the arming was inert (weight or tagger "
                "silently off), which must self-report DISTINCTLY from an "
                "occupancy collapse (sufficient_occupancy_for_c1) so "
                "governance can tell 'armed-but-occupancy-still-collapsed' "
                "(read-path fix did not reach write-path occupancy -> route "
                "to a write-path successor) from 'arming never engaged'."
            ),
        },
    ]
    ready = True
    preconditions: List[Dict] = []
    try:
        preconditions = p0_readiness_gate(readiness_checks)
    except P0NotReady as e:
        preconditions = e.preconditions
        ready = False

    def _first_unmet_label(precs: List[Dict]) -> str:
        _LABELS = {
            "sws_context_memory_writes_occur": "sleep_cycle_recording_gap_still_present",
            "rem_attribution_rollouts_occur": "sleep_cycle_recording_gap_still_present",
            "waking_writepath_engaged": "waking_writepath_not_engaged",
            "sufficient_occupancy_for_c1": "insufficient_occupancy_for_c1",
            "adam_drift_neutralized": "adam_drift_neutralization_failed",
            "sd016_arming_engaged": "sd016_arming_not_engaged",
        }
        for p in precs:
            if not p.get("met", True):
                return _LABELS.get(p.get("name"), "p0_readiness_unmet")
        return "p0_readiness_unmet"

    per_seed_diff: Dict[str, Dict] = {}
    for w_r, s_r in zip(waking, sws_r):
        seed = w_r["seed"]
        occ_w = w_r["slot_cosine_sim_occupied_only"]
        occ_s = s_r["slot_cosine_sim_occupied_only"]
        c1_scoreable = occ_w is not None and occ_s is not None
        c1_passes = bool(c1_scoreable and occ_s < occ_w)
        slot_sep_diff = s_r["slot_separation"] - w_r["slot_separation"]
        harm_dang_diff = s_r["harm_rate_dangerous"] - w_r["harm_rate_dangerous"]
        harm_safe_diff = s_r["harm_rate_safe"] - w_r["harm_rate_safe"]
        per_seed_diff[str(seed)] = {
            "seed": seed,
            "waking_slot_cosine_sim_occupied_only": occ_w,
            "sws_then_rem_slot_cosine_sim_occupied_only": occ_s,
            "occupied_only_signed_diff": (occ_s - occ_w) if c1_scoreable else None,
            "waking_n_occupied_slots": w_r["n_occupied_slots"],
            "sws_then_rem_n_occupied_slots": s_r["n_occupied_slots"],
            "waking_n_write_calls": w_r["n_write_calls"],
            "sws_then_rem_n_write_calls": s_r["n_write_calls"],
            "waking_slot_cosine_sim_raw_whole_bank": w_r["slot_cosine_sim_raw_whole_bank"],
            "sws_then_rem_slot_cosine_sim_raw_whole_bank": s_r["slot_cosine_sim_raw_whole_bank"],
            "c1_scoreable": c1_scoreable,
            # C1: PRIMARY GATE (once P0 is met). Directional (per ARC-045's
            # own experimental implication) -- SWS_THEN_REM strictly lower
            # than WAKING_ONLY, on the CORRECTED occupied-only statistic.
            "slot_cosine_sim_occupied_only_passes_C1": c1_passes,
            "waking_harm_rate_dangerous": w_r["harm_rate_dangerous"],
            "sws_then_rem_harm_rate_dangerous": s_r["harm_rate_dangerous"],
            "harm_rate_dangerous_signed_diff": harm_dang_diff,
            "waking_harm_rate_safe": w_r["harm_rate_safe"],
            "sws_then_rem_harm_rate_safe": s_r["harm_rate_safe"],
            "harm_rate_safe_signed_diff": harm_safe_diff,
            "waking_slot_separation": w_r["slot_separation"],
            "sws_then_rem_slot_separation": s_r["slot_separation"],
            "slot_separation_signed_diff": slot_sep_diff,
            "sws_then_rem_slot_separation_passes_C4": s_r["slot_separation"] > C4_SLOT_SEPARATION_THRESHOLD,
            "waking_action_class_entropy": w_r["train_action_class_entropy_mean"],
            "sws_then_rem_action_class_entropy": s_r["train_action_class_entropy_mean"],
            "sws_then_rem_sws_n_writes_total": s_r["sleep_cycle_sws_n_writes_total"],
            "sws_then_rem_rem_n_rollouts_total": s_r["sleep_cycle_rem_n_rollouts_total"],
        }

    c1_count = sum(1 for d in per_seed_diff.values() if d["slot_cosine_sim_occupied_only_passes_C1"])
    c1_pass = ready and c1_count >= C1_N_SEEDS_REQUIRED

    c4_count = sum(1 for d in per_seed_diff.values() if d["sws_then_rem_slot_separation_passes_C4"])
    c4_pass = c4_count >= C4_N_SEEDS_REQUIRED

    # Non-degeneracy self-report over the CORRECTED, gating DV (occupied-only
    # cosine) plus action entropy. Filters out None (unscoreable) entries --
    # a scoreability shortfall is caught by the sufficient_occupancy_for_c1
    # P0 gate above, not by degeneracy.
    all_occupied_cosine_values = [
        r["slot_cosine_sim_occupied_only"] for r in (waking + sws_r)
        if r["slot_cosine_sim_occupied_only"] is not None
    ]
    all_action_entropy = [r["train_action_class_entropy_mean"] for r in arm_results]
    degeneracy = check_degeneracy({
        "slot_cosine_sim_occupied_only": all_occupied_cosine_values or [0.0],
        "train_action_class_entropy_mean": {"values": all_action_entropy, "floor": 1e-6},
    })

    def _direction(passed: bool) -> str:
        return "supports" if passed else "weakens"

    if not ready:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        label = _first_unmet_label(preconditions)
        evidence_direction_per_claim = {cid: "non_contributory" for cid in CLAIM_IDS}
    else:
        outcome = "PASS" if c1_pass else "FAIL"
        evidence_direction = _direction(c1_pass)
        label = (
            "sws_then_rem_differentiates_occupied_slots" if c1_pass
            else "sws_then_rem_does_not_differentiate_occupied_slots"
        )
        evidence_direction_per_claim = {
            "SD-017": _direction(c1_pass),
            "ARC-045": _direction(c1_pass),
            "MECH-166": _direction(c1_pass),
        }

    summary = {
        "P0_sleep_cycle_write_counters": {
            "sws_n_writes_pooled": pooled_sws_n_writes,
            "sws_n_writes_floor": SWS_WRITES_FLOOR,
            "rem_n_rollouts_pooled": pooled_rem_n_rollouts,
            "rem_n_rollouts_floor": REM_ROLLOUTS_FLOOR,
            "waking_write_calls_pooled": pooled_waking_write_calls,
            "waking_write_calls_floor": WAKING_WRITEPATH_CALLS_FLOOR,
            "n_scoreable_seeds": n_scoreable_seeds,
            "n_scoreable_seeds_required": C1_N_SEEDS_REQUIRED,
            "adam_drift_max_abs_z": max_no_writes_abs_z,
            "adam_drift_tolerance_sigma": ADAM_DRIFT_NULL_TOLERANCE_SIGMA,
            "sd016_ctxdiv_applied_pooled": pooled_sd016_ctxdiv_applied,
            "sd016_ctxdiv_engaged_floor": SD016_CTXDIV_ENGAGED_FLOOR,
            "ready": ready,
            "desc": ("P0 GATE (6 checks, all must clear). 2 unchanged from "
                     "436c/436d (sleep-pass write/rollout counters); 3 from "
                     "436e (waking writepath engagement, occupancy "
                     "sufficiency, Adam-drift-neutralization calibration, per "
                     "failure_autopsy_V3-EXQ-436d-methodology-check_2026-08-07 "
                     "Recommendation #4); 1 NEW in 436f (sd016_arming_engaged "
                     "-- pooled |applied ctxdiv loss| > 0, per "
                     "failure_autopsy_V3-EXQ-436e_2026-08-13, so an inert "
                     "SD-016 arming self-reports distinctly from an occupancy "
                     "collapse)."),
        },
        "C1_primary_slot_cosine_sim_occupied_only_directional": {
            "n_seeds_required": C1_N_SEEDS_REQUIRED,
            "n_seeds_passed": c1_count,
            "pass": c1_pass,
            "scored": ready,
            "desc": ("SOLE PASS/FAIL GATE once P0 is met. "
                     "slot_cosine_sim_occupied_only(SWS_THEN_REM) < "
                     "slot_cosine_sim_occupied_only(WAKING_ONLY) in >= 3/5 "
                     "seeds. Computed over ONLY slots that received >=1 "
                     "ContextMemory.write() call this run (tracks write()'s "
                     "own min_idx directly) -- the corrected instrument."),
        },
        "C4_arc045_slot_separation_threshold": {
            "threshold": C4_SLOT_SEPARATION_THRESHOLD,
            "n_seeds_required": C4_N_SEEDS_REQUIRED,
            "n_seeds_passed": c4_count,
            "pass": c4_pass,
            "scored": ready,
            "desc": ("SECONDARY, non-gating. slot_separation in SWS_THEN_REM "
                     "> 0.3 in >= 3/5 seeds. Unchanged formula from 436d -- "
                     "a read-side visitation statistic, independent of the "
                     "write-occupancy cosine confound."),
        },
    }

    print(f"\nOutcome: {outcome}  label={label}  P0_ready={ready}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  Per-claim direction: {evidence_direction_per_claim}")
    print(f"  Non-degenerate: {degeneracy.get('non_degenerate')} "
          f"({degeneracy.get('degeneracy_reason', '')})")
    print(f"  Untouched-bank null (re-derived): {untouched_bank_null}")

    result = {
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": "manual-multi (run_sleep_cycle() called directly every SLEEP_INTERVAL episodes in training loop)",
        "outcome": outcome,
        "status": outcome,
        "result": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "pass_criteria_summary": summary,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1": degeneracy.get("non_degenerate"),
            },
        },
        "non_degenerate": degeneracy.get("non_degenerate"),
        "degeneracy_reason": degeneracy.get("degeneracy_reason"),
        "untouched_bank_null_re_derived": untouched_bank_null,
        "arc045_abs_reference_legacy": {
            "value": ARC045_ABS_COSINE_REFERENCE_LEGACY,
            "note": ("Calibrated against the PRE-436d write-gate defect. "
                     "NOT recomputed against the repaired path, NOT used "
                     "for gating. See module docstring point 5."),
        },
        "aggregated": {
            "per_seed": per_seed_diff,
            "n_seeds_passed": {"C1": c1_count, "C4": c4_count},
            "sleep_cycle_write_counters_pooled": {
                "sws_n_writes": pooled_sws_n_writes,
                "rem_n_rollouts": pooled_rem_n_rollouts,
                "waking_write_calls": pooled_waking_write_calls,
            },
            "no_writes_calibration_arm": {
                "per_seed_raw_whole_bank_cosine": [
                    r["slot_cosine_sim_raw_whole_bank"] for r in no_writes_r
                ],
                "per_seed_abs_z_vs_null": no_writes_abs_z,
                "per_seed_n_write_calls": [r["n_write_calls"] for r in no_writes_r],
            },
        },
        "arm_results": arm_results,
        "per_seed_results": arm_results,
        "registered_thresholds": {
            "C1_N_SEEDS_REQUIRED": C1_N_SEEDS_REQUIRED,
            "C4_SLOT_SEPARATION_THRESHOLD": C4_SLOT_SEPARATION_THRESHOLD,
            "C4_N_SEEDS_REQUIRED": C4_N_SEEDS_REQUIRED,
            "ARC045_ABS_COSINE_REFERENCE_LEGACY": ARC045_ABS_COSINE_REFERENCE_LEGACY,
            "BASE_HARM_THRESHOLD": BASE_HARM_THRESHOLD,
            "CONTEXT_BETA": CONTEXT_BETA,
            "SLOT_DANGER_EMA_ALPHA": SLOT_DANGER_EMA_ALPHA,
            "USE_NOISE_FLOOR": USE_NOISE_FLOOR,
            "NOISE_FLOOR_ALPHA": NOISE_FLOOR_ALPHA,
            "NOISE_FLOOR_MIN_TEMPERATURE": NOISE_FLOOR_MIN_TEMPERATURE,
            "BASELINE_TEMPERATURE": BASELINE_TEMPERATURE,
            "SWS_WRITES_FLOOR": SWS_WRITES_FLOOR,
            "REM_ROLLOUTS_FLOOR": REM_ROLLOUTS_FLOOR,
            "WAKING_WRITEPATH_CALLS_FLOOR": WAKING_WRITEPATH_CALLS_FLOOR,
            "ADAM_DRIFT_NULL_TOLERANCE_SIGMA": ADAM_DRIFT_NULL_TOLERANCE_SIGMA,
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
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-017/ARC-045/MECH-166 slot-differentiation retest on 436e's "
            "OCCUPIED-SLOTS-ONLY slot_cosine_sim DV machinery, with SD-016 "
            "cue-conditioning ARMED. Discharges the SD-016-armed retest "
            "recorded by failure_autopsy_V3-EXQ-436e_2026-08-13 (confirmed, "
            "user-adjudicated), which found 436e's DV repair SOUND (all six "
            "methodology-check items empirically confirmed) but its result "
            "non_contributory at the P0 layer: only 2/5 seeds carried >=2 "
            "occupied slots (need 3) because ContextMemory.write() argmin-"
            "addresses to a single-slot fixed point under the near-constant "
            "z_world query stream (cross-batch cosine 0.998) -- the "
            "long-registered SD-016 z_world bottleneck whose selection-"
            "mechanism fix (ree-v3 110a2785b6) 436e left switched off. This "
            "letter arms the V3-EXQ-922 production combination (sd016_enabled "
            "+ cue_slot_tagger + selection='gumbel' + "
            "context_divergence_weight=0.5, plus 922's ctxdiv training-loop "
            "wiring -- the config alone is inert, since "
            "compute_context_divergence_loss is never called automatically), "
            "held CONSTANT across all three conditions like 436e's writepath "
            "and drift-freeze. Sleep (SWS_THEN_REM vs WAKING_ONLY) remains "
            "the sole manipulation; the 436e DV machinery, gates, seeds and "
            "C1 shape are inherited unchanged. A sixth P0 gate "
            "(sd016_arming_engaged: pooled |applied ctxdiv loss| > 0) is "
            "added so an inert arming self-reports distinctly. CAVEAT: the "
            "cue-tagger + ctxdiv loss act on the READ path (extract_cue_"
            "context, on z_world via a separate tagger MLP, trained on "
            "DETACHED z_world), while the occupancy collapse is on the WRITE "
            "path (query_proj on the 64-dim [z_self,z_world] state), so "
            "arming is NOT mechanically guaranteed to lift write occupancy; "
            "sd016_arming_engaged=MET with insufficient_occupancy_for_c1=UNMET "
            "is the signal to route to a WRITE-path successor. The OLD "
            "whole-bank statistic is still recorded verbatim "
            "(slot_cosine_sim_raw_whole_bank) for audit continuity, but "
            "never gates."
        ),
    }
    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["timestamp_utc"] = ts
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config={
            "conditions": CONDITIONS,
            "sleep_interval": SLEEP_INTERVAL,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_episodes_each": EVAL_EPISODES_EACH,
            "seeds": SEEDS,
            "env_safe_num_hazards": 1,
            "env_dangerous_num_hazards": 8,
            "sd016_writepath_mode_by_condition": {
                name: wp for name, _s, _r, wp in CONDITIONS_SPEC
            },
            "sd016_diversification_weight": SD016_DIVERSIFICATION_WEIGHT,
            "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
            "context_memory_memory_requires_grad": False,
            # 436f: SD-016 cue-conditioning armed (held constant across all
            # three conditions). See module docstring "WHAT CHANGES vs 436e".
            "sd016_enabled": SD016_ENABLED,
            "sd016_cue_slot_tagger": SD016_CUE_SLOT_TAGGER,
            "sd016_cue_slot_tagger_selection": SD016_CUE_SLOT_TAGGER_SELECTION,
            "sd016_context_divergence_weight": SD016_CONTEXT_DIVERGENCE_WEIGHT,
            "sd016_ctxdiv_batch_size": SD016_CTXDIV_BATCH_SIZE,
            "use_per_stream_vs": True,
            "use_anchor_sets": True,
            "use_sd039_anchor_payload": True,
            "use_sleep_loop_in_sleep_arms": True,
            "use_noise_floor": USE_NOISE_FLOOR,
            "noise_floor_alpha": NOISE_FLOOR_ALPHA,
            "noise_floor_min_temperature": NOISE_FLOOR_MIN_TEMPERATURE,
        },
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        z_goal_stream_stats=zg_accumulator.stats(),
    )
    print(f"Output written to: {out_path}", flush=True)

    _outcome_clean = str(result.get("outcome", "FAIL")).upper()
    if _outcome_clean not in ("PASS", "FAIL"):
        _outcome_clean = "FAIL"
    emit_outcome(
        outcome=_outcome_clean,
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
