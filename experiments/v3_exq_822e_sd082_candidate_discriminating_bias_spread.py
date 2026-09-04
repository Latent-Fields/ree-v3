"""V3-EXQ-822e -- SD-082 with a SPREAD-VALUED DV. Successor to V3-EXQ-822d.

supersedes: V3-EXQ-822d

=============================== 822e HEADER ===============================
WHAT CHANGED, AND WHAT DELIBERATELY DID NOT.

THE BUILD IS NOT IN DOUBT AND IS NOT RE-OPENED HERE. V3-EXQ-822d confirmed the
SD-082 per-candidate-summary AMEND is ENGAGED: post-centering cross-candidate
spread ratio 2.70e-3 against 2.2e-9 for the legacy path, 0 fallbacks, 0
degenerate ticks. The SD-082 substrate_queue entry stays
implemented_pending_validation. This run does not re-validate or rebuild the
substrate; it replaces the DEPENDENT VARIABLE.

THE DEPENDENT VARIABLE WAS THE DEFECT. Confirmed autopsy
failure_autopsy_966-436g-951-959-822d-cluster_2026-08-30 (REE_assembly
40dd17331e), governance-ratified 54dbe477be, withdrew BOTH of 822d's per-claim
directions to non_contributory on two grounds:

  (1) THE SHARED CLUSTER DEFECT -- present in 3 of the cluster's 5 members. 822d's
      C1 was an ON-ARM-ONLY ABSOLUTE FLOOR (on_prop_delta_mean 0.001556 >= 1e-3)
      which the NEGATIVE CONTROL cleared MORE STRONGLY (off_prop_delta_mean
      0.001912) and to which the floor was never applied. The negative control was
      measured, it sat in the manifest, and it was never applied to the criterion
      it bears on.
  (2) THE SUBSTANTIVE ONE. prop_delta is an ablation MAGNITUDE -- the mean over
      candidates of |bias(rule_state) - bias(0)|. SD-082's whole substance is
      whether the propagated bias is candidate-DISCRIMINATING or candidate-
      UNIFORM, and a magnitude provably cannot separate those: a bias that is the
      same nonzero value on every candidate has a large mean-abs, a cross-
      candidate range of exactly zero, and is invisible to every argmax-derived
      downstream reader. 822d's C2 showed an ON-over-OFF contrast on 1/5 seeds
      with a NEGATIVE mean contrast (-3.56e-4).

THE FIX, IN FOUR PARTS. (Parts (a)-(c) are the second revision of this letter;
the first draft of 822e was red-teamed at Step 4.5, got CONTESTED at the severe
end on two findings, both were independently re-verified at source, and it was NOT
queued. Those two findings are (a) and (b) below.)

  (a) THE DV IS READ AT THE RAW PRE-TANH STAGE. The consumer's output stage is
      bias = bias_scale*tanh(bias_raw/bias_scale) with bias_scale 0.1. tanh is
      monotone but NOT affine, so its local slope depends on the base raw value
      each candidate sits at, and a PERFECTLY UNIFORM raw shift therefore emerges
      as a NON-uniform post-tanh delta: +0.01 applied at base raws 0.0 and 0.05
      gives 9.9668e-3 and 7.4932e-3, a "cross-candidate spread" of 2.4736e-3
      manufactured entirely by the squasher -- the same order as the first draft's
      own smoke numbers. That state is candidate-UNIFORM (a uniform shift through
      a monotone map preserves argmax order and is invisible to every downstream
      reader), which is exactly the state the DV exists to EXCLUDE. Measuring
      post-tanh cannot exclude it. The raw stage is recomputed from the
      substrate's own head -- not by editing the module under validation -- and
      cross-checked every tick against the substrate's real output inverted
      through raw = bias_scale*atanh(bias/bias_scale), with the residual GATED as
      a readiness precondition so the replica cannot silently diverge.
      The post-tanh spread is retained as a recorded diagnostic (C5).

  (b) THE ATTRIBUTION IS STRUCTURAL. The arm axis is crf_cue_centering, which is
      SD-078's rule-POOL knob; SD-082's own fix
      (candidate_summary_source="proposer_post_action") is passed UNCONDITIONALLY
      in BOTH arms. So no ON/OFF contrast can attribute to SD-082 -- both arms
      carry its fix and ARM_OFF is a pool control, not a summary-fix-off control.
      The first draft routed SD-082's direction off that contrast, which is 822d's
      withdrawn misattribution shape with the claim roles swapped. Here SD-082's
      question is treated as what it is -- an ABSOLUTE property of the propagated
      bias, measurable in either arm -- so C1 and C2 are ABSOLUTE criteria applied
      IDENTICALLY IN BOTH ARMS, and dir_082 is computed with no contrast quantity
      in it at all. The ON-minus-OFF contrast is C4 and belongs to SD-078 alone.

  (c) THE STATISTIC IS DIMENSIONLESS, SO THE BAR IS NOT A GUESS ON AN UNMEASURED
      SCALE. prop_ratio_raw = (max-min)/(mean-abs) over candidates of the raw
      propagated bias. Its null is analytic (candidate-uniform -> 0) and so is its
      ceiling (all mass on one candidate -> K, the candidate count). It was then
      MEASURED on the real substrate before any bar was written (2 seeds x both
      arms, P0/P1/P2 = 10/25/10, K=32): per-cell medians 0.59 / 1.14 / 2.81 / 4.11.
      The 1.0 floor sits inside that range and is two-sided -- 3 of 4 probed cells
      clear it, 1 does not. C1's discriminating power is additionally DEMONSTRATED
      in-run by a negative control (candidate-constant summaries through the real
      head) that must read BELOW the bar, and its feasibility by dv_headroom
      entries measured from an in-loop positive control.

  (d) EVERY NON-FINDING ROUTES TO INCONCLUSIVE. A criterion that can only support
      is not a test, and a null a 5-seed design cannot separate from a small true
      effect is not a falsification. Both per-claim directions are three-valued;
      "weakens" for SD-082 requires the index to sit near its analytic null in
      BOTH arms (a positive finding of candidate-uniformity), not merely to miss
      C1's bar.

WHAT THE FIRST DRAFT'S C2 WAS, AND WHY IT IS GONE. It required the MEAN per-seed
contrast to be positive, while C1 required ALL FIVE per-seed contrasts to be
positive -- which implies it arithmetically. It was not a second gate and its
failure branch was dead code. C2 is now the POOLED argmax-flip fraction, which is
genuinely independent of C1 in both directions: an order-preserving propagated
bias has a large index and flips nothing.

PRE-REGISTERED ACCEPTANCE (822e, second revision). Constants in the module:
  C1  raw discrimination index, ARM_ON  : per-seed median >= 1.0 on >= 3/5 seeds
                                                                          [SD-082]
  C2  pooled argmax-flip fraction       : >= 0.02 over all cells          [SD-082]
  C4  SD-078 index contrast             : ON-OFF >= 0.25 on >= 4/5 seeds
                                          and mean >= 0.25                [SD-078]
  C1b same floor on ARM_OFF             : APPLIED and REPORTED, gates nothing
  C3  legacy magnitude floor, both arms : DIAGNOSTIC, gates nothing
  C5  post-tanh spread                  : DIAGNOSTIC, gates nothing
overall PASS = readiness AND C1 AND C2. C4 is load-bearing for SD-078's direction
only and is out of BOTH that conjunction and run-level readiness.

WHAT THE Step 4.5 RED-TEAM CHANGED (model: fable; verdict CONTESTED; six findings,
all verified against source before acting, three of them structural):
  * C1 originally required the floor in BOTH arms. That handed SD-078's knob a veto
    over SD-082's direction -- the same misattribution this letter exists to remove,
    in mirror image (ARM_ON 5/5 with ARM_OFF 2/5 would have recorded SD-082
    "inconclusive"), and ARM_OFF was already a coin flip at probe scale (2.81 and
    0.59 against a 1.0 floor). C1 now gates on ARM_ON; the identical floor is
    evaluated on ARM_OFF and reported as non-gating C1b.
  * C4's dv_headroom achievable was the range over ALL cells, i.e. the treatment
    effect certifying its own headroom -- and it sat in run-level readiness, so a
    genuine SD-078 null (ON ~ OFF) would have read as "the DV cannot reach the bar"
    and voided a confirmed SD-082 result as substrate_not_ready_requeue. It is now
    measured on the CONTROL arm's seed-to-seed spread and gates SD-078 only.
  * "the bias head actually trained" was a diagnostic flag and is now a readiness
    gate: the index is scale-free, so the cross-candidate structure a random first
    layer plus ReLU already carries would otherwise be reportable as a trained
    coupling.
  * Dismissed with a recorded reason rather than fixed: the index's scale-freeness
    itself (C2 is the functional backstop and re-adding a magnitude gate would
    reinstate the DV the ratified autopsy withdrew); hr_c1's insensitivity (it
    certifies reachability, and its description now says so); and the
    upstream-mask-starvation route into the `weakens` branch (caveat recorded at
    reads_uniform, pointing at the recorded summary-spread statistic).

READINESS, BEYOND 822a/822c's. Four new gates, all self-routing to
substrate_not_ready_requeue rather than to a claim verdict:
  raw_stage_replica_matches_substrate   the raw replica IS the substrate's path
  uniform_control_reads_as_uniform      the negative control reads below C1's bar
  control_probes_sampled                both in-loop controls actually ran
  dv_headroom_*                         each load-bearing bar is reachable
Both controls now run INSIDE the measurement loop at live-rule_state ticks, at the
real candidate count and rescaled to the real summary norm. The first draft ran
its positive control ONCE after the episode loop, on a rule_state the final
agent.reset() had just zeroed, and fed it randn-scale (~8-norm) input to a head
that sees ~0.4-norm input -- so a rule-inactive final episode read 0 and
spuriously routed the whole run substrate_not_ready.

SD-078 IS A CO-TAG, NOT THE SUBJECT. Its rule-pool differentiation passed
decisively as a readiness precondition in 822d (16 rules, max pairwise distance
1.711, rule_state_diff 0.650 against a 3.24e-10 control). Both SD-078 and SD-082
had pending_retest_after_substrate CLEARED to false by the 2026-08-30 governance
cycle; 822e is a NEW QUESTION, not the retest they were standing on.

red-team (Step 4.5): CONTESTED (model: claude-fable) -- 6 findings, all verified
at source; 3 structural fixes applied (C1 arm scoping, C4 headroom control + scope,
head-trained gate), 3 dismissed in writing. See "WHAT THE Step 4.5 RED-TEAM CHANGED"
above and the queue entry note for V3-EXQ-822e.
========================= END 822e HEADER; 822d text follows =========================

The remainder of this docstring is INHERITED FROM 822d and describes the shared
harness, the readiness gates, and the substrate history, all of which 822e keeps
unchanged. Where it states 822d's C1/C2 acceptance, the 822e header above
supersedes it.

WHAT THE 822d RUN WAS. The designated `validation_experiment` for the SD-082
substrate_queue entry, whose per-candidate-summary AMEND landed
2026-08-30T11:32:08Z (ree-v3 merge ef88faa, from integration/sd082-percandidate-
summary 3aa45de). Both the substrate_queue entry and ree-v3/CLAUDE.md's
"SD-082 AMEND: per-candidate summary was a shared constant" section name
V3-EXQ-822d explicitly and record it as "PLANNED but NOT YET QUEUED ... a
/queue-experiment follow-on session is needed". This is that session's output.

THE DEFECT THIS RUN TESTS THE FIX FOR. Confirmed by
failure_autopsy_V3-EXQ-822c_2026-08-29.md (user-adjudicated). Under the default
candidate_summary_source='proposer', every caller's manual fallback read
trajectory.world_states[:, 0, :] -- but E2FastPredictor.rollout_with_world seeds
world_states=[initial_z_world], so index 0 is the rollout's SHARED initial world
state. Candidates differ only in the actions applied from t>=1, so that read is
bit-identical across all K candidates BY CONSTRUCTION and carries zero
candidate-discriminating information. SD-082's own centering step
(summaries - summaries.mean(dim=0)) then annihilates that constant to float32
cancellation noise. Measured consequence in V3-EXQ-822c:
rule_summary_magnitude_ratio 2.8e6-4.5e6 in all six cells (~4000x its 1e3
ceiling), while prop_delta nonetheless cleared the 1e-3 non-vacuity floor --
an authentic-looking, meaningless number. Severity: corrupting.

THE FIX, AND THE ONE LINE THAT CHANGES HERE. candidate_summary_source gains a
third value 'proposer_post_action' (default stays 'proposer', bit-identical);
agent._proposer_post_action_summaries() reads world_states[:, 1:, :].mean(0) --
the POST-ACTION states, reflecting each candidate's own action sequence -- at
zero extra model calls. compute_bias additionally gained a diagnostic-only
centering-degeneracy guard (LateralPFCConfig.candidate_summary_degeneracy_floor,
default 1e-4) that records candidate_summary_norm_pre/post_centering and flags
candidate_summary_degenerate, never raising and never changing the bias.

STEP 2.5a EMPIRICAL PROBE (this session, before authoring; 40 real ticks on the
822c config at seed 101). Plumbing reaches: REEConfig.from_dims ->
agent.config.candidate_summary_source == 'proposer_post_action'.
_candidate_world_summaries() returned NON-None on 40/40 ticks (the dispatch
bypasses every manual fallback, so the 822/822a/822b starvation route is closed
at source). And, directly measuring the defect and its repair on the same
candidates:

    post-centering cross-candidate summary norm
      LEGACY  ws[0, 0, :]           = 5.66e-08      <- float32 cancellation noise
      FIXED   proposer_post_action  = 4.01e-01      <- 7.1e6 x more signal

with candidate_summary_degenerate = False under the fix (pre 25.23 -> post 0.44).
That 7.1-million-fold separation is why the summary-spread readiness gate below
is set at a ratio floor of 1e-3: four orders of magnitude above the legacy
reading and ~17x below the observed fixed reading.

EXPERIMENT_PURPOSE = evidence (CHANGED from 822c's diagnostic, deliberately).
822b/822c were diagnostics asking WHY propagation read zero. That question is
answered and the substrate is fixed. This run asks SD-082's own stated
acceptance question, which the autopsy is explicit was never actually put:
"SD-082's centering is NOT falsified -- it is UNTESTED, because its input never
carried the cross-candidate variance it was designed to preserve." Both claims
are pending_retest_after_substrate=true. So this run is scored.

822d's PRE-REGISTERED ACCEPTANCE -- SUPERSEDED BY THE 822e HEADER ABOVE, kept
here so the change is legible rather than silent:
  C1 propagation non-vacuous : on_prop_delta_mean >= 1e-3   [WITHDRAWN in 822e:
      an ON-arm-only absolute floor the negative control cleared more strongly]
  C2 ON>OFF contrast         : (on - off) > 1e-3 on a MAJORITY of seeds
      [REPLACED in 822e: same contrast shape, but on the SPREAD DV, at margin 0,
       requiring unanimity plus a positive mean rather than a bare majority]

SEEDS RAISED 3 -> 5, ON THE AUTOPSY'S OWN FINDING. Section 4 grades Scale
"adequate for the primary finding; INSUFFICIENT for the ON-vs-OFF read (3 seeds,
direction inverts)" -- 822c's per-seed contrast was +0.001771 / +0.000033 /
-0.002036, i.e. carried entirely by one seed. C2 is exactly that read, so
running it at n=3 would reproduce the same under-powering the autopsy flagged.
Five seeds make "majority" 3/5 rather than 2/3. Seeds 101/202/303 are retained
verbatim from 822a/822c for continuity; 404/505 are the new draw.

READINESS (gating; unmet -> outcome FAIL, non_degenerate=false,
interpretation.label = substrate_not_ready_requeue, never a substrate verdict):
  (a) z_world common-mode cone present               [== 822a/822c, floor 0.90]
  (b) ON pool differentiated / OFF pool pinned       [== 822a/822c]
  (c) ON rule active on >= frac floor of P2 ticks    [== 822a/822c]
  (d) NEW -- prop-sample sufficiency: EVERY cell's n_prop_samples clears a floor
      of 20. This is the gate the substrate_queue entry names in so many words
      ("asserts n_prop_samples > 0 as a readiness gate BEFORE trusting any
      prop_delta aggregate -- the exact measurement-starvation gap 822/822a/822b
      fell into"), set at 20 rather than the literal 1 because 822c observed
      145-200 per cell, so 20 is comfortably non-binding while still strictly
      implying > 0. The lesson it encodes is autopsy Section 5's: a no-data
      default (`fmean(xs) if xs else 0.0`) that shares the numeric value of a
      meaningful result is the most expensive shape in this corpus.
  (e) NEW -- fix engaged: cross-candidate summary spread survives centering.
      Worst-cell median of (candidate_summary_norm_post_centering /
      _pre_centering) clears 1e-3, AND no cell ever set
      candidate_summary_degenerate, AND the manual ws[0,0,:] fallback was taken
      ZERO times (a nonzero count means the dispatch did not engage).
  (f) capture positive control: the head-diagnostics path returns finite,
      in-range values on a synthetic pre-training forward pass, every cell
      [== 822c].

SAME-STATISTIC DISCIPLINE for gate (e) (the V3-EXQ-643 rule). C1/C2 route on
prop_delta, a rule-state-ablation delta whose MEANINGFULNESS depends entirely on
the candidate summaries carrying cross-candidate variance. Gate (e) therefore
asserts a cross-candidate SPREAD (the post-centering norm IS the deviation from
the cross-candidate mean, since centering removes that mean), scale-normalised
by the pre-centering norm -- NOT a magnitude, mean_abs, or max_abs, any of which
can be large while the spread is ~0. That is precisely the failure mode here:
the legacy read had pre-centering norm 25.23 (large) and post-centering spread
5.66e-08 (nil). A magnitude-shaped gate would have passed it.

DV-SYMMETRY INVARIANCE (Step 3 mandatory declaration, per arm). The manipulation
is crf_cue_centering (OFF -> collapsed rule pool, ON -> differentiated).

  * prop_delta_mean -- mean over K candidates of |bias(rule_state) - bias(0)|.
    Symmetry group: permutation of candidates (it is a symmetric aggregate), and
    addition of a candidate-independent constant to the per-candidate DIFFERENCE.
    The manipulation is not a candidate permutation. It is NOT invariant under
    the second either -- but ONLY because of the fix, and this is the whole
    point. rule_state is broadcast identically across candidates
    (rule_repeated = rule_state.expand(k, -1)); the head is nonlinear
    (Linear->ReLU->Linear), so changing rule_state moves each candidate's bias by
    a DIFFERENT amount only when `summaries` actually differ across candidates.
    Under the pre-fix 'proposer' path summaries were constant, so the per-
    candidate difference was one shared constant -- the manipulation WAS
    invariant in the relevant sense, and 822c's prop_delta was an arithmetic
    artifact rather than a measurement (autopsy disposition (b)). Gate (e) is
    what certifies the invariance is broken before C1/C2 are read at all.
  * rule_flip_frac -- fraction of ticks where the argmax over the bias vector
    moves under rule-state ablation. Invariant under adding a broadcast constant
    to the bias vector (a constant cannot move an argmax). Under the pre-fix path
    the ablation delta WAS exactly such a constant, so flip was structurally
    pinned at zero -- and V3-EXQ-822c measured rule_flip_frac = 0.0000 in all six
    cells, which is an independent arithmetic confirmation of the autopsy's
    structural finding. Under proposer_post_action the delta is per-candidate and
    a flip is no longer forbidden. Recorded as a DIAGNOSTIC, NOT gating and NOT
    load-bearing: flip > 0 is confirmatory, but flip == 0 is not decisive, since
    a genuinely near-invariant head can produce no flip without any symmetry
    pinning it. Reading a zero here as a verdict would repeat this lineage's
    founding error.
  * rule_state_diff -- mean pairwise (1 - cosine) over sampled rule_states. The
    manipulation acts directly on rule-pool differentiation, which is what this
    measures; no invariance. [== 822a/822c]
  * candidate_summary_norm_post/pre_centering -- crf_cue_centering does not touch
    candidate world summaries at all, so this readout is EXPECTED to be
    arm-independent. That is by design, not a null finding: it is a readiness
    measure of the fix, not an effect measure of the manipulation.

GOV-REUSE-1 (Step 2.4). Decisive readout: on_prop_delta_mean and the per-seed
ON-OFF contrast, measured under candidate_summary_source='proposer_post_action'.
That config value did not exist before ree-v3 3aa45de (2026-08-30); a grep of
REE_assembly/evidence/experiments/ finds 0 manifests mentioning
'proposer_post_action' and 0 mentioning 'candidate_summary_degenerate'. The
readout is not recorded and not derivable post-hoc from any prior manifest --
822/822a/822b recorded n_prop_samples = 0 in all 18 cells (no data at all), and
822c measured the DEFECTIVE path. Not recoverable -> run.

RE-DERIVE BRAKE (Step 2.5b) -- FIRES, AND IS RELEASED. Counted under the skill's
own predicate (substrate_ceiling category OR non_contributory direction, keyed by
run at most-recent adjudication): SD-078 = 4 hits, SD-082 = 2 hits, both at or
over the threshold of 2. Braking autopsies: failure_autopsy_816c-822_2026-07-26,
failure_autopsy_batch-822a-826-817a-827_2026-07-26,
failure_autopsy_2026-07-28-sweep, failure_autopsy_V3-EXQ-822c_2026-08-29.
RELEASED because the named upstream substrate is now BUILT: the most recent
counted autopsy's recommended_substrate_queue_entry is
{action: amend, target_sd_id: SD-082}, and that amend is IMPLEMENTED
(ree-v3 ef88faa, 2026-08-30; ree-v3/CLAUDE.md "SD-082 AMEND: per-candidate
summary ..."). Per the skill, a released brake makes the re-test meaningful --
this is not a fourth blind lettered iteration around the same ceiling, it is the
first test of the claim on substrate that can carry the signal.
(Note the 822c autopsy recorded "brake does not fire" using the narrower R1-R3
predicate, which counts only literal substrate_ceiling readings. Both readings
are stated so a later reader is not left reconciling them.)

STEP 2.5c substrate-path overlap gate. Resolved by call-trace over 30 real ticks
of this exact driver path rather than by module-name matching. Of the OPEN
corrupting substrate_queue entries: salience_coordinator.py
(mode-governance-engagement) NOT executed; blocked_agency.py
(sd_blocked_agency_mismatch_floor_calibration) NOT executed;
ContextMemory.write / compute_write_addressing_loss / agent.compute_prediction_loss
(contextmemory-write-path-addressing-degeneracy) NOT executed -- this driver is a
pure forward/select path and never trains E1, so the degenerate WRITE path is
never taken (only e1_deep read/forward/generate_prior/predict_long_horizon/
get_schema_salience/reset_hidden_state are reached); probe_warmup.py
(SD-PROBE-WARMUP) not imported. The ONLY overlap is SD-082 itself, whose entry
this run exists to validate and whose corrupting failure_record was marked
resolved by the fix above.

ETHICS PREFLIGHT (Step 2.6). All involvement flags false, decision allow. No
negative-valence drive, no suffering-like accumulator, no self-model, no
inescapability manipulation, no offline replay over harm, no social/language, no
human or clinical data. Pre-ethical V3 instrumentation only (SENT-0).

SUPERSEDES: nothing. 822c is NOT superseded -- it is the diagnostic that measured
the defect, its failure_record is marked `resolved` (not `superseded`) by the fix,
and its finding stands. Stated explicitly so a later reader does not infer a
supersession from the shared lineage.

ARM REUSE: in-line emit only (arm_cell defaults, driver folded into the hash). No
canonical baseline module and no separate mint job -- this lineage re-tunes its
OFF path at every letter (822 -> 822a consumer flags -> 822b capture flag ->
822c measurement fix -> 822d summary source), which is exactly the
substrate-in-flux calibration family the skill's carve-out names: the fingerprint
correctly refuses a drifted mint, so there is nothing to skip.

See SD-078, SD-082, SD-033a, ARC-063, SD-008;
REE_assembly/docs/architecture/sd_082_rule_selection_action_consumer.md;
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-822c_2026-08-29.md;
ree-v3/CLAUDE.md "SD-082 AMEND: per-candidate summary was a shared constant";
experiments/v3_exq_822c_sd082_candidate_summary_fallback_fix.py (parent design).
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import dv_headroom_check  # noqa: E402

_ZG = ZGoalStreamAccumulator()
# One representative agent per arm, kept alive purely so stamp_recording_core can
# read enabled_default_off_flags off a live .config. Bounded at 2 (not 10) -- the
# accumulator above, not this dict, is what measures z_goal liveness across all
# cells, and z_goal_stream_stats takes precedence over agent= for that block.
_ARM_AGENTS: Dict[str, Any] = {}

EXPERIMENT_TYPE = "v3_exq_822e_sd082_candidate_discriminating_bias_spread"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["SD-078", "SD-082"]

# `head_diagnostics_capture_active` and `summary_dispatch_engaged` are config-echo
# plus finiteness/count checks on deterministic forward passes, not hand-tuned
# numeric scoring predicates: capture_head_diagnostics=True and
# candidate_summary_source="proposer_post_action" are passed explicitly in
# _make_agent (both arms), so both are true by construction unless the three-site
# plumbing itself is broken -- which the SD-082 amend's own 22 contracts, not a
# precondition here, are the right place to catch.
ANCHOR_REACHABILITY_EXEMPT = (
    "head_diagnostics_capture_active / summary_dispatch_engaged are config echoes "
    "(capture_head_diagnostics and candidate_summary_source are set explicitly, not "
    "learned or tuned) plus finiteness/count checks on deterministic forward passes; "
    "reachable by construction, no scoring predicate involved."
)

SEEDS = [101, 202, 303, 404, 505]  # 101/202/303 == 822a/822c; 404/505 the new draw.
ARMS = ["ARM_OFF", "ARM_ON"]
P0_WARMUP_EPISODES = 60
P1_BIAS_TRAIN_EPISODES = 70
P2_MEASUREMENT_EPISODES = 40
STEPS_PER_EPISODE = 48

# --- Pre-registered readiness thresholds (a)-(c) identical to 822a/822c. ---
CONE_MIN_COSINE_FLOOR = 0.90
DIST_FLOOR = 0.05
MIN_LIVE_RULES = 2
OFF_MAX_LIVE_CEIL = 1
FRAC_ACTIVE_FLOOR = 0.10
SEED_PASS_FRACTION = 3.0 / 5.0  # majority of 5 (822a/822c used 2/3 of 3).

# --- Pre-registered acceptance (SD-082's own, per the autopsy). ---
# --- 822e's DV and its bars -------------------------------------------------
# The load-bearing DV is prop_spread: the CROSS-CANDIDATE RANGE of the propagated
# bias (max - min over candidates of bias(rule_state) - bias(0)), NOT 822d's
# prop_delta magnitude (mean over candidates of |bias(rule_state) - bias(0)|).
#
# WHY THE MARGIN IS EXACTLY ZERO, AND WHY THAT IS THE CAREFUL CHOICE RATHER THAN
# THE LAZY ONE. Requirement (b) of this redesign is to do the threshold
# arithmetic against measured magnitudes before pre-registering any bar. That
# arithmetic CANNOT be done for prop_spread: 822d recorded the prop_delta
# MAGNITUDE (on 1.556e-3, off 1.912e-3) and the SUMMARY spread ratio (2.70e-3),
# but never the BIAS spread, so its scale is unmeasured. Any nonzero absolute bar
# would therefore be a guess -- and a guessed bar on an unmeasured scale is
# precisely the defect class that produced V3-EXQ-936a's vacuous verdict (a 0.05
# bar sitting ~7,900x above the maximum attainable value), 642a's, and 964's.
# A SIGNED CONTRAST at margin 0.0 is scale-free and attainable in BOTH directions
# by construction, so no branch can fire by construction.
# Statistical content comes from UNANIMITY rather than from the bar's size: under
# a null of no effect the per-seed sign is a fair coin, so 5/5 is a one-sided sign
# test at p = 1/32 = 0.031. A 4/5 outcome is therefore NOT a PASS -- it is
# recorded as mixed, with the count kept in the manifest.
SPREAD_MEASURABLE_FLOOR = 1e-9 # readiness: synthetic positive control must move.
SYNTH_PROBE_SEED = 8221        # dedicated generator -- must NOT perturb run RNG.

# --- 822e's LOAD-BEARING DV: the RAW-STAGE DISCRIMINATION INDEX -------------
# prop_ratio_raw = (max - min over candidates) / (mean |.| over candidates) of the
# RAW PRE-TANH propagated bias, bias_raw(rule_state) - bias_raw(0).
#
# WHY A RATIO AND NOT THE BARE SPREAD. The bare spread is measured in the raw
# head's own arbitrary output units, so any absolute bar on it is a guess on an
# unmeasured scale -- the defect class that produced 936a, 642a, 964 and 822d. The
# ratio is DIMENSIONLESS and its two reference points are both known a priori:
#   candidate-UNIFORM bias (the state the autopsy says a magnitude DV cannot
#     exclude, and the state this DV exists to exclude) --> ratio = 0 exactly;
#   bias concentrated on a single candidate --> ratio = K, the candidate count,
#     which is therefore the DV's ANALYTIC ceiling.
# So the criterion is scale-free AND its achievable range is bounded by
# construction, not by a guess.
#
# MEASURED ON THE ACTUAL SUBSTRATE before this bar was written (probe, 2 seeds x
# both arms, P0/P1/P2 = 10/25/10, K = 32 candidates, 40-50 sampled P2 ticks/cell):
#   ARM_OFF seed 101  median 2.81   (per-tick 1.87 - 3.57)
#   ARM_ON  seed 101  median 4.11   (per-tick 0.83 - 7.57)
#   ARM_OFF seed 202  median 0.59   (per-tick 0.33 - 0.85)
#   ARM_ON  seed 202  median 1.14   (per-tick 0.29 - 4.21)
# The bar 1.0 -- "the cross-candidate range is at least as large as the typical
# per-candidate magnitude" -- therefore sits INSIDE the measured range and is
# genuinely two-sided: 3 of the 4 probed cells clear it and one does not. It is
# neither unreachable (the 936a defect) nor cleared by everything (which would
# make the criterion decorative).
RAW_RATIO_FLOOR = 1.0
# C2's bar, on the POOLED argmax-flip fraction over all cells of both arms.
# Under a candidate-uniform bias the flip rate is 0 EXACTLY (a uniform raw shift
# through a monotone squash preserves argmax order), so 0 is the null. Probe
# pooled rate: 14 flips / 181 ticks = 0.077. POOLED, not worst-cell: the per-cell
# rates (0.043 / 0.100 / 0.140 / 0.023) rest on 1-7 flips each and a worst-cell
# gate on them would be a coin flip -- piloted, found under-powered, and pooled
# for that reason rather than shipped as a gate this run cannot resolve.
FLIP_FRAC_FLOOR = 0.02
# C2's bar is a PROPORTION, so its achievable range is arithmetic ([0,1]) rather
# than sampled -- and the real feasibility constraint on a 0.02 floor is
# RESOLUTION, not ceiling: with n pooled ticks the smallest non-zero fraction
# expressible is 1/n, so a 0.02 bar is unresolvable below 50 ticks. That is the
# schedule-derived quantity dv_achievable's analytic route exists for (951c's
# "zero reachable ticks"), and it is gated separately as
# `flip_resolution_sufficient`. 4x the bare resolution requirement.
FLIP_TICKS_FLOOR = 200
# NOT measured from the synthetic positive control, deliberately, and this is
# worth stating because it was tried and rejected on measurement: the synthetic
# control uses randn-DIFFERENTIATED summaries, whereas the real candidates sit
# inside the SD-008 ~0.98-cosine cone and are therefore near-tied under the head.
# A small rule_state contribution flips a near-tie readily and a randn spread
# almost never, so the synthetic control's own flip rate measured 0.0 while the
# real pooled rate was 0.138 in the same smoke -- i.e. it is a HARDER test than
# the measurement, and using it as the achievable would have aborted the run as
# "not ready" precisely when the DV was working. The positive control remains the
# right achievable measurement for C1's index, where no such geometry mismatch
# applies.
# Seeds (of 5) that must clear RAW_RATIO_FLOOR, IN EACH ARM SEPARATELY.
RATIO_SEED_MAJORITY = 3
# SD-078's own criterion (C4) -- and ONLY SD-078's. See the attribution note at
# CLAIM_IDS. Probed per-seed (ON - OFF) median-ratio contrast: +1.30, +0.55.
SD078_CONTRAST_MARGIN = 0.25
SD078_CONTRAST_SEEDS = 4          # of 5; not unanimity (see C4's detail text).
# Below this the propagated bias reads as candidate-UNIFORM and SD-082 is actively
# WEAKENED rather than merely unsupported. Between this and RAW_RATIO_FLOOR the
# run is INCONCLUSIVE and must not emit a claim direction at all.
UNIFORM_VERDICT_CEIL = 0.25
# Headroom margin for every dv_headroom entry: the bar must sit at most half the
# achievable range away, not merely touch it.
DV_HEADROOM_MARGIN = 2.0
# Readiness: the driver-side raw-stage replica must reproduce the substrate's own
# output inverted through atanh. Probe residual 7.5e-09 - 6.0e-08 (float32 dust).
RAW_REPLICA_TOL = 1e-4
ATANH_CLAMP = 0.999999

# Retained from 822d as RECORDED DIAGNOSTICS ONLY -- they gate nothing here. The
# whole point of 822e is that this floor could not discriminate: 822d's C1 was an
# ON-arm-only absolute floor that the NEGATIVE CONTROL cleared MORE strongly
# (off 1.912e-3 > on 1.556e-3) and was never applied to. 822e keeps measuring it
# and now APPLIES IT TO BOTH ARMS, reporting the result, which is the discipline
# 822d omitted.
PROP_NONVAC_FLOOR = 1e-3   # DIAGNOSTIC ONLY in 822e (822d's C1 bar).
CONTRAST_MARGIN = 1e-3     # DIAGNOSTIC ONLY in 822e (822d's C2 bar).
# CONTRAST_MARGIN is set equal to PROP_NONVAC_FLOOR deliberately and named
# separately so the choice is explicit rather than an accidental reuse: it demands
# the ON-over-OFF effect be at least as large as the smallest propagation the
# claim calls meaningful at all.

# --- Pre-registered readiness thresholds (d)-(f), new in this letter. ---
PROP_SAMPLE_FLOOR = 20              # (d) worst-cell n_prop_samples; 822c saw 145-200.
SUMMARY_SPREAD_RATIO_FLOOR = 1e-3   # (e) post/pre centering norm; legacy ~2.2e-9, fixed ~1.8e-2.
HEAD_DIAG_SAMPLE_FLOOR = 5          # (f) == 822c.

# --- Diagnostic (non-gating) interpretation bands, carried from 822c. ---
MAGNITUDE_RATIO_LOW = 1e-3
MAGNITUDE_RATIO_HIGH = 1e3
DEAD_RELU_HIGH_FLOOR = 0.90
DEAD_RELU_MED_FLOOR = 0.50
LAST_LAYER_WEIGHT_DELTA_FLOOR = 1e-3

# P1 REINFORCE (mirror V3-EXQ-598b / 654f / 822a / 822c).
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9

ENV_KWARGS = dict(size=8, num_hazards=2, num_resources=6, use_proxy_fields=True)


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, centering: bool) -> REEAgent:
    """Identical to 822c's _make_agent plus candidate_summary_source=
    "proposer_post_action" on BOTH arms -- the one substrate change this letter
    validates. Everything else (schedule, env, consumer flags, capture flag) is
    held fixed against 822c so the comparison is controlled."""
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=0.9,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        lateral_pfc_rule_readout_consumer=True,
        lateral_pfc_capture_head_diagnostics=True,
        candidate_summary_source="proposer_post_action",  # <-- the only new flag vs 822c.
        use_gated_policy=True,
        use_candidate_rule_field=True,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=0.45,
        crf_maintenance_couple_to_theta=True,
        crf_tolerance_conflict_cap=3,
        crf_cue_centering=centering,
        crf_cue_baseline_alpha=0.02,
    ))


def _config_slice(centering: bool) -> Dict[str, Any]:
    return {
        "env": dict(ENV_KWARGS),
        "schedule": {"p0": P0_WARMUP_EPISODES, "p1": P1_BIAS_TRAIN_EPISODES,
                     "p2": P2_MEASUREMENT_EPISODES, "steps": STEPS_PER_EPISODE},
        "alpha_world": 0.9,
        "use_lateral_pfc_analog": True,
        "lateral_pfc_train_rule_bias_head": True,
        "lateral_pfc_rule_readout_consumer": True,
        "lateral_pfc_capture_head_diagnostics": True,
        "candidate_summary_source": "proposer_post_action",
        "use_gated_policy": True,
        "use_candidate_rule_field": True,
        "crf_persist_rules_across_episode_reset": True,
        "crf_mature_pool_dynamics": True,
        "crf_availability_maintenance": True,
        "crf_maintenance_floor": 0.45,
        "crf_maintenance_couple_to_theta": True,
        "crf_tolerance_conflict_cap": 3,
        "crf_cue_centering": centering,
        "crf_cue_baseline_alpha": 0.02,
    }


def _cone_min_cosine(vecs: List[torch.Tensor]) -> float:
    if len(vecs) < 2:
        return 1.0
    m = torch.nn.functional.normalize(torch.stack(vecs).float(), dim=-1)
    c = m @ m.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(c[iu[0], iu[1]].min())


def _mean_pairwise_one_minus_cosine(vecs: List[torch.Tensor]) -> float:
    vecs = [v for v in vecs if v is not None and float(v.norm()) > 1e-9]
    if len(vecs) < 2:
        return 0.0
    m = torch.nn.functional.normalize(torch.stack(vecs).float(), dim=-1)
    c = m @ m.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float((1.0 - c[iu[0], iu[1]]).mean())


def _candidate_summaries(
    agent: REEAgent, candidates, counters: Dict[str, int]
) -> Optional[torch.Tensor]:
    """With candidate_summary_source="proposer_post_action" the dispatch in
    agent._candidate_world_summaries() returns a real [K, world_dim] tensor, so
    the manual ws[0, 0, :] fallback below should NEVER be taken. It is retained
    only so a plumbing regression degrades to 822c's behaviour instead of
    crashing -- and every fallback is COUNTED, because a nonzero count means
    'proposer_post_action' did not engage and readiness gate (e) must fail rather
    than the run silently measuring the defective path again."""
    summ = agent._candidate_world_summaries(candidates)
    if summ is not None:
        counters["dispatch"] += 1
        return summ.detach()
    if not candidates:
        return None
    counters["fallback"] += 1
    cand_world_list: List[torch.Tensor] = []
    for c in candidates:
        if c.world_states is not None:
            ws = c.get_world_state_sequence()  # [batch, horizon+1, world_dim]
            cand_world_list.append(ws[0, 0, :])
        elif agent._current_latent is not None:
            cand_world_list.append(agent._current_latent.z_world[0].detach())
        else:
            return None
    return torch.stack(cand_world_list, dim=0).detach()


def _head_diag_snapshot(agent: REEAgent) -> Dict[str, float]:
    """Unconditional (always-fresh, never-latched) weight-norm read."""
    st = agent.lateral_pfc.get_state()
    return {
        "capture_head_diagnostics": bool(st.get("capture_head_diagnostics", False)),
        "first_linear_weight_norm": float(st["rule_bias_head_first_linear_weight_norm"]),
        "first_linear_bias_norm": float(st["rule_bias_head_first_linear_bias_norm"]),
        "last_linear_weight_norm": float(st["rule_bias_head_last_linear_weight_norm"]),
        "last_linear_bias_norm": float(st["rule_bias_head_last_linear_bias_norm"]),
    }


def _init_positive_control(agent: REEAgent) -> Dict[str, Any]:
    """Readiness (f) positive control: force one real compute_bias() call with
    capture_head_diagnostics=True on a synthetic 3-candidate input BEFORE any
    training, confirming the capture path returns finite, in-range values
    independently of whether training later moves anything. The synthetic input
    is torch.randn, i.e. genuinely differentiated across candidates -- so this
    also confirms the degeneracy guard does NOT false-positive on a
    non-degenerate summary."""
    lpfc = agent.lateral_pfc
    wd = agent.config.latent.world_dim
    synth = torch.randn(3, wd, device=agent.device)
    with torch.no_grad():
        lpfc.compute_bias(synth)
        st = lpfc.get_state()
    snap = _head_diag_snapshot(agent)
    snap["dead_relu_frac"] = float(st["hidden_dead_relu_frac"])
    snap["magnitude_ratio"] = float(st["rule_summary_magnitude_ratio"])
    snap["summary_norm_pre_centering"] = float(st["candidate_summary_norm_pre_centering"])
    snap["summary_norm_post_centering"] = float(st["candidate_summary_norm_post_centering"])
    snap["summary_degenerate"] = bool(st["candidate_summary_degenerate"])
    snap["dead_relu_frac_finite_in_range"] = (
        math.isfinite(snap["dead_relu_frac"]) and 0.0 <= snap["dead_relu_frac"] <= 1.0)
    snap["magnitude_ratio_finite"] = math.isfinite(snap["magnitude_ratio"])
    return snap


def _raw_stage_prop(lpfc, summaries: torch.Tensor
                    ) -> Optional[Tuple[float, float, float, float, float]]:
    """THE RAW PRE-TANH PROPAGATION. Returns
    (raw_spread, raw_delta, raw_ratio, replica_atanh_abs_err, max_bias_over_scale).

    WHY THIS EXISTS -- THE DEFECT IT REMOVES (the blocking finding against the
    first 822e draft, verified twice at source and once by arithmetic). The
    consumer's output stage is
        bias = bias_scale * tanh(bias_raw / bias_scale)
    (lateral_pfc_analog.py, under `if self.config.rule_readout_consumer:`), and
    bias_scale defaults to 0.1. tanh is monotone but NOT affine, so its local
    slope depends on the BASE raw value each candidate happens to sit at. A
    PERFECTLY UNIFORM raw shift therefore emerges as a NON-uniform post-tanh
    delta: a uniform +0.01 applied at base raws 0.0 and 0.05 gives output deltas
    9.9668e-3 and 7.4932e-3 -- a post-tanh "cross-candidate spread" of 2.4736e-3
    manufactured entirely by the squasher, on the SAME order as the first draft's
    own smoke numbers (ON 2.33e-3, OFF 2.80e-3). That state is candidate-UNIFORM:
    a uniform raw shift through a monotone map preserves argmax order and is
    invisible to every downstream reader -- precisely the state this DV was
    introduced to EXCLUDE. Measuring the spread post-tanh cannot exclude it.
    The load-bearing statistic is therefore read at the RAW stage, and the
    post-tanh spread is retained alongside as a diagnostic only.

    WHY A REPLICA RATHER THAN A SUBSTRATE LATCH. Capturing bias_raw inside
    compute_bias would mean editing the module under validation, which this run
    is explicitly not permitted to re-open. Instead the pre-tanh stage is
    recomputed here from the module's own head and its own centering rule, and
    then CROSS-CHECKED against the module's real output inverted through the
    exact analytic inverse raw = bias_scale * atanh(bias / bias_scale). The
    residual is returned and gated as a readiness precondition
    (`raw_stage_replica_matches_substrate`), so if the module's output stage ever
    changes the replica cannot silently diverge -- the run self-routes to
    substrate_not_ready_requeue instead of reporting a statistic from a path the
    substrate no longer takes. Measured residual at probe scale: 7.45e-09 (float32
    dust) in 4/4 cells.

    A NOTE ON SATURATION, since it is the one condition that would degrade the
    cross-check: atanh is ill-conditioned as |bias|/bias_scale -> 1. The probe
    measured a maximum of 0.576, where d(atanh)/du = 1.5, so the inversion is well
    conditioned here; `max_bias_over_scale` is returned and recorded so a later
    reader can see whether that stayed true.
    """
    cfg = lpfc.config
    k = int(summaries.shape[0])
    if k < 2:
        return None
    s = summaries
    # Same centering rule the module applies, same guard.
    if cfg.rule_readout_consumer and k >= 2:
        s = s - s.mean(dim=0, keepdim=True)

    def _raw(rule_state: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([rule_state.expand(k, -1), s], dim=-1)
        return lpfc.rule_bias_head(joined).squeeze(-1)

    with torch.no_grad():
        r1 = _raw(lpfc.rule_state).detach().clone().reshape(-1)
        b1 = lpfc.compute_bias(summaries).detach().clone().reshape(-1)
        saved = lpfc.rule_state.detach().clone()
        lpfc.rule_state.zero_()
        r0 = _raw(lpfc.rule_state).detach().clone().reshape(-1)
        lpfc.rule_state.copy_(saved)
    scale = float(cfg.bias_scale)
    if not (scale > 0.0):
        return None
    u = (b1 / scale).clamp(-ATANH_CLAMP, ATANH_CLAMP)
    err = float((scale * torch.atanh(u) - r1).abs().max().item())
    sat = float((b1.abs() / scale).max().item())
    prop = r1 - r0
    raw_spread = float((prop.max() - prop.min()).item())
    raw_delta = float(prop.abs().mean().item())
    # DIMENSIONLESS DISCRIMINATION INDEX. spread / mean-abs separates exactly the
    # two states the autopsy names: a candidate-UNIFORM bias (same nonzero value on
    # every candidate) has mean-abs large and range 0, so the ratio is 0; a bias
    # concentrated on one candidate has ratio K. Scale-free, and its ceiling is
    # ANALYTIC (K, the candidate count) rather than sampled.
    raw_ratio = (raw_spread / raw_delta) if raw_delta > 0.0 else 0.0
    vals = (raw_spread, raw_delta, raw_ratio, err, sat)
    if not all(math.isfinite(v) for v in vals):
        return None
    return vals


def _prop_delta_and_flip_with_diag(
    agent: REEAgent, summaries: torch.Tensor
) -> Optional[Tuple[float, bool, Dict[str, Any]]]:
    """822c's latch-safe measurement, extended with the SD-082 amend's three new
    degeneracy fields plus rule_state_norm (the autopsy Section 3 "cheap
    confirmer": without it the magnitude ratio cannot be decomposed into its
    numerator and denominator, which is the one premise that autopsy could not
    verify from its own artifact).

    LATCH ORDERING (unchanged from 822c and load-bearing). compute_bias() is
    called TWICE per active tick -- once on the real rule_state, once on a zeroed
    one -- and hidden_dead_relu_frac / rule_summary_magnitude_ratio are LATCHED
    (get_state() returns whatever the most recent call wrote). get_state() is
    therefore read immediately after the FIRST call and BEFORE rule_state is
    zeroed, so the diagnostics and rule_state_norm both belong to the real call.
    The two centering-degeneracy norms depend only on candidate_world_summaries
    and so are call-invariant, but they are read at the same point anyway."""
    lpfc = getattr(agent, "lateral_pfc", None)
    if lpfc is None or summaries is None:
        return None
    try:
        with torch.no_grad():
            bias_field = lpfc.compute_bias(summaries).detach().clone()
            diag_state = lpfc.get_state()
            head_diag = {
                "dead_relu_frac": float(diag_state["hidden_dead_relu_frac"]),
                "magnitude_ratio": float(diag_state["rule_summary_magnitude_ratio"]),
                "rule_state_norm": float(diag_state["rule_state_norm"]),
                "summary_norm_pre_centering": float(
                    diag_state["candidate_summary_norm_pre_centering"]),
                "summary_norm_post_centering": float(
                    diag_state["candidate_summary_norm_post_centering"]),
                "summary_degenerate": bool(diag_state["candidate_summary_degenerate"]),
            }
            saved = lpfc.rule_state.detach().clone()
            lpfc.rule_state.zero_()
            bias_zero = lpfc.compute_bias(summaries).detach().clone()
            lpfc.rule_state.copy_(saved)
        prop = (bias_field - bias_zero).reshape(-1)
        # 822d's DV: an ablation MAGNITUDE. Kept as a diagnostic.
        delta = float(prop.abs().mean().item())
        # 822e's DV: the CROSS-CANDIDATE RANGE of the propagated bias. This is the
        # statistic SD-082's substance actually turns on -- whether the propagated
        # bias is candidate-DISCRIMINATING or candidate-UNIFORM -- and a magnitude
        # provably cannot separate those: a bias that is the SAME nonzero value on
        # every candidate has a large mean-abs and a range of exactly 0, and is
        # invisible to any argmax-derived downstream reader. That is the documented
        # magnitude-vs-range failure (V3-EXQ-643) and the broadcast-scalar
        # DV-symmetry failure (V3-EXQ-604c), arriving here as 822d's C1.
        spread = float((prop.max() - prop.min()).item()) if prop.numel() >= 2 else 0.0
        # ARGMAX FLIP is stage-invariant: bias = scale*tanh(raw/scale) is strictly
        # monotone, so argmax(bias) == argmax(bias_raw) for both the real and the
        # ablated rule_state. Reading it post-tanh is therefore identical to
        # reading it raw, and no separate raw flip statistic is needed.
        flip = int(bias_field.argmax().item()) != int(bias_zero.argmax().item())
        raw = _raw_stage_prop(lpfc, summaries)
        if raw is None:
            return None
        head_diag["raw_spread"] = raw[0]
        head_diag["raw_delta"] = raw[1]
        head_diag["raw_ratio"] = raw[2]
        head_diag["raw_replica_atanh_abs_err"] = raw[3]
        head_diag["max_bias_over_scale"] = raw[4]
        if not (math.isfinite(delta) and math.isfinite(spread)):
            return None
        return delta, spread, bool(flip), head_diag
    except Exception:
        return None


def _control_probes(lpfc, summaries: torch.Tensor) -> Optional[Dict[str, float]]:
    """The TWO in-loop controls that make the raw-stage DV interpretable.

    Both are read AT A LIVE MEASUREMENT TICK, on the real trained head, with the
    real rule_state in place -- which is the defect this replaces. The first draft
    ran its positive control ONCE after the episode loop, on the end-of-run
    rule_state that the final `agent.reset()` has just zeroed, so a rule-inactive
    final episode read 0 and spuriously routed the WHOLE run to
    substrate_not_ready_requeue; it also fed randn-scale (~8-norm) summaries to a
    head that sees ~0.4-norm ones in the run. Here both controls are built at the
    tick, at the SAME candidate count K and rescaled to the SAME mean row norm as
    the real summaries, so the head is probed in the regime it actually operates.

    POSITIVE CONTROL (`synth_*`): genuinely differentiated candidate summaries.
    Establishes that the statistic CAN register discrimination on this head --
    below floor means the measurement path is dead (last layer still at its zero
    init), NOT that the propagated bias is uniform. This is what supplies the
    ACHIEVABLE range for the dv_headroom preconditions on C1 and C2.

    NEGATIVE CONTROL (`uniform_*`): summaries held CONSTANT across candidates --
    the provably zero-information input, the exact shape the legacy
    `ws[0, 0, :]` read produced and the state SD-082's centering step annihilates.
    A statistic that cannot read this as uniform cannot exclude candidate-
    uniformity, which is the one thing 822d's magnitude DV failed at. Requiring
    this to fall BELOW the C1 bar is therefore a demonstration, in-run and on the
    real head, that C1 has discriminating power -- rather than an assertion that
    it does.

    Own torch.Generator, so neither control consumes run RNG or perturbs the
    per-cell arm_fingerprint.
    """
    try:
        k = int(summaries.shape[0])
        wd = int(summaries.shape[-1])
        if k < 2:
            return None
        real_norm = float(summaries.norm(dim=-1).mean().item())
        g = torch.Generator()
        g.manual_seed(SYNTH_PROBE_SEED)
        synth = torch.randn(k, wd, generator=g)
        sn = float(synth.norm(dim=-1).mean().item())
        if sn > 0.0 and real_norm > 0.0:
            synth = synth * (real_norm / sn)
        synth = synth.to(summaries.device).to(summaries.dtype)
        # Constant across candidates, at the real row norm: row 0 repeated.
        uniform = summaries[0:1].detach().clone().expand(k, -1).contiguous()

        out: Dict[str, float] = {}
        for tag, inp in (("synth", synth), ("uniform", uniform)):
            r = _raw_stage_prop(lpfc, inp)
            if r is None:
                return None
            out[f"{tag}_raw_spread"] = r[0]
            out[f"{tag}_raw_ratio"] = r[2]
        # Positive control's argmax flip -- the achievable side of C2's bar.
        with torch.no_grad():
            b1 = lpfc.compute_bias(synth).detach().reshape(-1)
            saved = lpfc.rule_state.detach().clone()
            lpfc.rule_state.zero_()
            b0 = lpfc.compute_bias(synth).detach().reshape(-1)
            lpfc.rule_state.copy_(saved)
        out["synth_flip"] = float(
            int(b1.argmax().item()) != int(b0.argmax().item()))
        if not all(math.isfinite(v) for v in out.values()):
            return None
        return out
    except Exception:
        return None


def _lpfc_reinforce_loss(agent: REEAgent,
                         outcome_buf: List[Tuple[torch.Tensor, int, float]],
                         baseline: float, device) -> torch.Tensor:
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


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    centering = (arm == "ARM_ON")
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(centering),
                  script_path=Path(__file__)) as cell:
        env = _build_env(seed)
        agent = _make_agent(env, centering)
        crf = agent.candidate_rule_field
        wd = agent.config.latent.world_dim
        bias_opt = torch.optim.Adam(
            list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS)

        head_diag_by_phase: Dict[str, Any] = {"init": _init_positive_control(agent)}
        summary_counters = {"dispatch": 0, "fallback": 0}

        total_eps = P0_WARMUP_EPISODES + P1_BIAS_TRAIN_EPISODES + P2_MEASUREMENT_EPISODES
        p2_start = P0_WARMUP_EPISODES + P1_BIAS_TRAIN_EPISODES

        reinforce_baseline = 0.0
        outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

        zworlds: List[torch.Tensor] = []
        rule_state_samples: List[torch.Tensor] = []
        prop_deltas: List[float] = []
        prop_spreads: List[float] = []          # POST-tanh: diagnostic only in 822e.
        prop_spreads_raw: List[float] = []      # raw pre-tanh spread (diagnostic).
        prop_ratios_raw: List[float] = []       # raw pre-tanh DISCRIMINATION INDEX (DV).
        raw_replica_errs: List[float] = []
        bias_over_scale: List[float] = []
        synth_ratios: List[float] = []
        synth_spreads: List[float] = []
        synth_flips: List[float] = []
        uniform_ratios: List[float] = []
        flips: List[int] = []
        p2_dead_relu_samples: List[float] = []
        p2_magnitude_ratio_samples: List[float] = []
        p2_rule_state_norms: List[float] = []
        p2_spread_ratios: List[float] = []
        n_summary_degenerate_ticks = 0
        n_magnitude_ratio_inf = 0
        p2_ticks = 0
        p2_active_ticks = 0
        max_live = 0

        for ep in range(total_eps):
            is_p1 = (P0_WARMUP_EPISODES <= ep < p2_start)
            is_p2 = (ep >= p2_start)
            phase = "P2" if is_p2 else ("P1" if is_p1 else "P0")
            _, obs = env.reset()
            agent.reset()
            ep_reward = 0.0
            ep_buf: List[Tuple[torch.Tensor, int]] = []

            for _step in range(STEPS_PER_EPISODE):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=agent.device))
                candidates = agent.generate_trajectories(latent, e1, ticks)

                p1_snap: Optional[torch.Tensor] = None
                if is_p1 and candidates and len(candidates) >= 2:
                    cs = _candidate_summaries(agent, candidates, summary_counters)
                    if cs is not None and torch.isfinite(cs).all():
                        p1_snap = cs.clone()

                action = agent.select_action(candidates, ticks)
                if action is None:
                    idx = int(np.random.randint(0, 4))
                    action = torch.zeros(1, 4, device=agent.device)
                    action[0, idx] = 1.0
                    agent._last_action = action
                committed_class = int(action[0].argmax().item())

                if is_p1 and p1_snap is not None:
                    sel = 0
                    for ci, c in enumerate(candidates):
                        if (getattr(c, "actions", None) is not None
                                and c.actions.shape[1] >= 1
                                and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                                == committed_class):
                            sel = min(ci, p1_snap.shape[0] - 1)
                            break
                    ep_buf.append((p1_snap, sel))

                if is_p2 and ticks.get("e3_tick", False):
                    p2_ticks += 1
                    if len(zworlds) < 200:
                        zworlds.append(latent.z_world.detach().reshape(-1).clone())
                    st = crf.get_state()
                    n_active = int(st.get("crf_n_active_last", 0))
                    max_live = max(max_live, int(st.get("crf_n_live", len(crf._rules))))
                    if n_active >= 1:
                        p2_active_ticks += 1
                        rs = agent.lateral_pfc.rule_state.detach().reshape(-1).clone()
                        if float(rs.norm()) > 1e-9:
                            rule_state_samples.append(rs)
                        if candidates and len(candidates) >= 2:
                            summ = _candidate_summaries(agent, candidates, summary_counters)
                            if summ is not None and torch.isfinite(summ).all():
                                pf = _prop_delta_and_flip_with_diag(agent, summ)
                                if pf is not None:
                                    delta, spread, flip, hd = pf
                                    prop_deltas.append(delta)
                                    prop_spreads.append(spread)
                                    prop_spreads_raw.append(hd["raw_spread"])
                                    prop_ratios_raw.append(hd["raw_ratio"])
                                    raw_replica_errs.append(
                                        hd["raw_replica_atanh_abs_err"])
                                    bias_over_scale.append(hd["max_bias_over_scale"])
                                    cp = _control_probes(agent.lateral_pfc, summ)
                                    if cp is not None:
                                        synth_ratios.append(cp["synth_raw_ratio"])
                                        synth_spreads.append(cp["synth_raw_spread"])
                                        synth_flips.append(cp["synth_flip"])
                                        uniform_ratios.append(cp["uniform_raw_ratio"])
                                    flips.append(1 if flip else 0)
                                    p2_dead_relu_samples.append(hd["dead_relu_frac"])
                                    p2_rule_state_norms.append(hd["rule_state_norm"])
                                    if hd["summary_degenerate"]:
                                        n_summary_degenerate_ticks += 1
                                    pre = hd["summary_norm_pre_centering"]
                                    if pre > 0.0:
                                        p2_spread_ratios.append(
                                            hd["summary_norm_post_centering"] / pre)
                                    # The magnitude ratio is rule_norm/summary_norm and is
                                    # 0.0 by construction on a tick where rule_state is
                                    # still zero -- recording those would drag the median
                                    # to 0 and manufacture a spurious LOW-band flag. Only
                                    # ticks with a genuinely live rule_state are recorded.
                                    mr = hd["magnitude_ratio"]
                                    if hd["rule_state_norm"] > 1e-9:
                                        if math.isfinite(mr):
                                            p2_magnitude_ratio_samples.append(mr)
                                        else:
                                            n_magnitude_ratio_inf += 1

                _, _h, done, _info, obs = env.step(int(action.argmax(dim=-1).item()))
                if is_p1:
                    ep_reward += float(_h)
                if done:
                    break

            if is_p1:
                reinforce_baseline = (EMA_DECAY * reinforce_baseline
                                      + (1.0 - EMA_DECAY) * ep_reward)
                for cand_features, sel in ep_buf:
                    outcome_buf.append((cand_features, sel, ep_reward))
                if len(outcome_buf) > OUTCOME_BUF_MAX:
                    outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
                l_loss = _lpfc_reinforce_loss(
                    agent, outcome_buf, reinforce_baseline, agent.device)
                if l_loss.requires_grad:
                    bias_opt.zero_grad()
                    l_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.lateral_pfc.bias_head_parameters(), 1.0)
                    bias_opt.step()

            if (ep + 1) == P0_WARMUP_EPISODES:
                head_diag_by_phase["post_p0"] = _head_diag_snapshot(agent)
            elif (ep + 1) == p2_start:
                head_diag_by_phase["post_p1"] = _head_diag_snapshot(agent)
            elif (ep + 1) == total_eps:
                head_diag_by_phase["post_p2"] = _head_diag_snapshot(agent)

            if (ep + 1) % 20 == 0 or (ep + 1) == total_eps:
                print(f"  [train] crf seed={seed} arm={arm} phase={phase} "
                      f"ep {ep+1}/{total_eps} live={len(crf._rules)} "
                      f"minted={crf._n_minted}", flush=True)

        rule_state_diff = _mean_pairwise_one_minus_cosine(rule_state_samples)
        prop_delta_mean = statistics.fmean(prop_deltas) if prop_deltas else 0.0
        prop_spread_mean = statistics.fmean(prop_spreads) if prop_spreads else 0.0
        prop_spread_max = max(prop_spreads) if prop_spreads else 0.0
        # The LOAD-BEARING per-cell statistic is the MEDIAN of the per-tick raw
        # discrimination index, not the mean: the index is a ratio and its
        # per-tick distribution is right-skewed (probe per-tick max 7.57 against a
        # median of 4.11), so a mean is dragged by a handful of ticks while the
        # criterion's claim is about the typical tick.
        prop_ratio_raw_median = (statistics.median(prop_ratios_raw)
                                 if prop_ratios_raw else 0.0)
        prop_ratio_raw_mean = (statistics.fmean(prop_ratios_raw)
                               if prop_ratios_raw else 0.0)
        prop_spread_raw_mean = (statistics.fmean(prop_spreads_raw)
                                if prop_spreads_raw else 0.0)
        synth_ratio_median = statistics.median(synth_ratios) if synth_ratios else 0.0
        synth_spread = statistics.fmean(synth_spreads) if synth_spreads else 0.0
        synth_flip_frac = statistics.fmean(synth_flips) if synth_flips else 0.0
        # Worst = MAXIMUM for the uniform control: the readiness gate asserts it
        # stays BELOW the C1 bar on every tick, so the extremum the gate tests is
        # the one that must be reported (a mean can hide an out-of-band tick).
        uniform_ratio_max = max(uniform_ratios) if uniform_ratios else 0.0
        raw_replica_err_max = max(raw_replica_errs) if raw_replica_errs else 0.0
        bias_over_scale_max = max(bias_over_scale) if bias_over_scale else 0.0
        rule_flip_frac = (sum(flips) / len(flips)) if flips else 0.0
        frac_active = (p2_active_ticks / p2_ticks) if p2_ticks else 0.0

        dead_relu_frac_p2_mean = (statistics.fmean(p2_dead_relu_samples)
                                  if p2_dead_relu_samples else 0.0)
        dead_relu_frac_p2_worst = max(p2_dead_relu_samples) if p2_dead_relu_samples else 0.0
        magnitude_ratio_p2_median = (statistics.median(p2_magnitude_ratio_samples)
                                     if p2_magnitude_ratio_samples else 0.0)
        # Worst (MINIMUM) spread ratio, not the mean -- the readiness gate quantifies
        # over ticks ("centering never annihilated the summary"), so the reported
        # statistic must be the extremum the gate tests, else an in-band mean can
        # mask an out-of-band tick and the indexer's recompute silently passes it.
        spread_ratio_p2_min = min(p2_spread_ratios) if p2_spread_ratios else 0.0
        spread_ratio_p2_median = (statistics.median(p2_spread_ratios)
                                  if p2_spread_ratios else 0.0)
        rule_state_norm_p2_median = (statistics.median(p2_rule_state_norms)
                                     if p2_rule_state_norms else 0.0)
        last_layer_weight_delta_init_to_p1 = float(
            head_diag_by_phase.get("post_p1", {}).get("last_linear_weight_norm", 0.0)
            - head_diag_by_phase["init"]["last_linear_weight_norm"])

        row = {
            "arm": arm,
            "seed": seed,
            "rule_state_diff": float(rule_state_diff),
            "prop_delta_mean": float(prop_delta_mean),
            "prop_spread_mean": float(prop_spread_mean),
            "prop_spread_max": float(prop_spread_max),
            "n_prop_spread_samples": len(prop_spreads),
            # --- 822e's LOAD-BEARING raw-stage statistics ---
            "prop_ratio_raw_median": float(prop_ratio_raw_median),
            "prop_ratio_raw_mean": float(prop_ratio_raw_mean),
            "prop_spread_raw_mean": float(prop_spread_raw_mean),
            "n_prop_ratio_raw_samples": len(prop_ratios_raw),
            "raw_replica_atanh_err_max": float(raw_replica_err_max),
            "max_bias_over_scale": float(bias_over_scale_max),
            "n_flips": int(sum(flips)),
            "n_flip_ticks": len(flips),
            # --- in-loop controls (positive + negative) ---
            "synth_ratio_median": float(synth_ratio_median),
            "synth_flip_frac": float(synth_flip_frac),
            "uniform_ratio_max": float(uniform_ratio_max),
            "n_control_probe_samples": len(synth_ratios),
            "synthetic_spread_probe": float(synth_spread),
            "rule_flip_frac": float(rule_flip_frac),
            "crf_max_pairwise_rule_dist": float(crf.max_pairwise_rule_distance()),
            "crf_max_live_rules": int(max_live),
            "crf_live_rules_final": int(len(crf._rules)),
            "crf_n_minted": int(crf._n_minted),
            "crf_frac_active_p2": float(frac_active),
            "crf_baseline_allocated": crf._baseline is not None,
            "n_rule_state_samples": len(rule_state_samples),
            "n_prop_samples": len(prop_deltas),
            "zworld_cone_min_cosine": _cone_min_cosine(zworlds),
            "n_zworld_sampled": len(zworlds),
            # --- SD-082 amend: fix-engagement readouts (new in 822d) ---
            "n_summary_dispatch": int(summary_counters["dispatch"]),
            "n_summary_fallback": int(summary_counters["fallback"]),
            "summary_spread_ratio_p2_min": float(spread_ratio_p2_min),
            "summary_spread_ratio_p2_median": float(spread_ratio_p2_median),
            "n_summary_degenerate_ticks": int(n_summary_degenerate_ticks),
            "n_spread_ratio_samples": len(p2_spread_ratios),
            "rule_state_norm_p2_median": float(rule_state_norm_p2_median),
            "n_rule_state_norm_samples": len(p2_rule_state_norms),
            # --- head-internals diagnostics, carried from 822c ---
            "n_head_diag_samples": len(p2_dead_relu_samples),
            "hidden_dead_relu_frac_p2_mean": float(dead_relu_frac_p2_mean),
            "hidden_dead_relu_frac_p2_worst": float(dead_relu_frac_p2_worst),
            "rule_summary_magnitude_ratio_p2_median": float(magnitude_ratio_p2_median),
            "n_magnitude_ratio_live_samples": len(p2_magnitude_ratio_samples),
            "n_magnitude_ratio_inf": int(n_magnitude_ratio_inf),
            "head_diag_by_phase": head_diag_by_phase,
            "last_layer_weight_delta_init_to_p1": last_layer_weight_delta_init_to_p1,
        }
        cell.stamp(row)
    _ZG.observe(agent)
    _ARM_AGENTS[arm] = agent
    passed = (row["rule_state_diff"] > 0.05) if centering else \
             (row["crf_max_live_rules"] <= OFF_MAX_LIVE_CEIL)
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min"):
    fn = min if mode == "min" else max
    r = fn(rows, key=lambda x: x[key])
    return float(r[key]), f"{r['arm']}/seed{r['seed']}"


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(_run_cell(arm, seed))

    off = [r for r in rows if r["arm"] == "ARM_OFF"]
    on = [r for r in rows if r["arm"] == "ARM_ON"]

    # --- READINESS (a)-(c): identical in form to 822a/822c. ---
    cone_worst, cone_cell = _worst_cell(rows, "zworld_cone_min_cosine", mode="min")
    cone_present = cone_worst >= CONE_MIN_COSINE_FLOOR
    n_on_diff = sum(1 for r in on if r["crf_max_pairwise_rule_dist"] > DIST_FLOOR
                    and r["crf_max_live_rules"] >= MIN_LIVE_RULES)
    on_diff_pool = (n_on_diff / len(on)) >= SEED_PASS_FRACTION
    n_off_pinned = sum(1 for r in off if r["crf_max_live_rules"] <= OFF_MAX_LIVE_CEIL)
    off_pinned_pool = (n_off_pinned / len(off)) >= SEED_PASS_FRACTION
    n_on_active = sum(1 for r in on if r["crf_frac_active_p2"] >= FRAC_ACTIVE_FLOOR)
    on_active = (n_on_active / len(on)) >= SEED_PASS_FRACTION

    # --- READINESS (d): prop-sample sufficiency, EVERY cell (worst-cell). ---
    n_prop_worst, n_prop_cell = _worst_cell(rows, "n_prop_samples", mode="min")
    prop_samples_sufficient = n_prop_worst >= PROP_SAMPLE_FLOOR

    # --- READINESS (e): the fix engaged -- cross-candidate spread survives centering. ---
    spread_worst, spread_cell = _worst_cell(rows, "summary_spread_ratio_p2_min", mode="min")
    spread_ok = spread_worst >= SUMMARY_SPREAD_RATIO_FLOOR
    n_degen_worst, degen_cell = _worst_cell(rows, "n_summary_degenerate_ticks", mode="max")
    no_degenerate_tick = n_degen_worst <= 0.0
    n_fallback_worst, fallback_cell = _worst_cell(rows, "n_summary_fallback", mode="max")
    dispatch_engaged = n_fallback_worst <= 0.0
    fix_engaged = spread_ok and no_degenerate_tick and dispatch_engaged

    # --- READINESS (f): capture positive control (worst cell). ---
    init_flags = [r["head_diag_by_phase"]["init"] for r in rows]
    def _ctrl_ok(f):
        return bool(f["capture_head_diagnostics"] and f["dead_relu_frac_finite_in_range"]
                    and f["magnitude_ratio_finite"])
    capture_worst = min(1.0 if _ctrl_ok(f) else 0.0 for f in init_flags)
    capture_active = capture_worst >= 1.0
    capture_offending = next(
        (f"{r['arm']}/seed{r['seed']}" for r, f in zip(rows, init_flags) if not _ctrl_ok(f)),
        "none")
    n_diag_worst, n_diag_cell = _worst_cell(rows, "n_head_diag_samples", mode="min")
    head_diag_sufficient = n_diag_worst >= HEAD_DIAG_SAMPLE_FLOOR

    synth_worst, synth_cell = _worst_cell(rows, "synthetic_spread_probe", mode="min")
    spread_measurable = synth_worst > SPREAD_MEASURABLE_FLOOR

    # --- 822e readiness additions -------------------------------------------
    # (i) The raw-stage replica must still BE the substrate's own pre-tanh path.
    replica_worst, replica_cell = _worst_cell(
        rows, "raw_replica_atanh_err_max", mode="max")
    replica_ok = bool(replica_worst <= RAW_REPLICA_TOL)
    # (ii) The NEGATIVE control must read as uniform, i.e. below C1's own bar.
    #      This is what gives C1 demonstrated discriminating power on this head
    #      rather than asserted power.
    uniform_worst, uniform_cell = _worst_cell(rows, "uniform_ratio_max", mode="max")
    uniform_reads_uniform = bool(uniform_worst < RAW_RATIO_FLOOR)
    ctrl_n_worst, ctrl_n_cell = _worst_cell(
        rows, "n_control_probe_samples", mode="min")
    controls_sampled = bool(ctrl_n_worst >= HEAD_DIAG_SAMPLE_FLOOR)

    # --- dv_headroom: one entry per LOAD-BEARING criterion --------------------
    # Each certifies that the DV can REACH the bar the criterion registers, using
    # the in-loop POSITIVE control as the achievable measurement -- an independent
    # demonstration on the real head, not this run's own treatment values.
    synth_ratio_values = [float(r["synth_ratio_median"]) for r in rows]
    synth_flip_values = [float(r["synth_flip_frac"]) for r in rows]
    # C4's achievable must be measured on the CONTROL arm alone. Using the range
    # over ALL cells would make the measurement the treatment effect itself: a
    # large ON-vs-OFF difference would certify its own headroom, and a genuine
    # SD-078 null (ON ~ OFF, low seed variance) would read as "the DV cannot reach
    # the bar" and abort the run. The honest control quantity is how far this index
    # moves across seeds ABSENT the manipulation.
    off_ratio_values = [float(r["prop_ratio_raw_median"]) for r in off]

    def _headroom(name, **kw):
        e = dv_headroom_check(name, margin=DV_HEADROOM_MARGIN, **kw)
        # This driver hand-rolls its preconditions[] rather than calling
        # p0_readiness_gate (it adjudicates AFTER the cells, not before P1), so
        # `met` is computed here with the same single-bound floor semantics that
        # gate uses: kind dv_headroom rides the ordinary lower-bound path.
        e["met"] = bool(e["measured"] >= e["threshold"])
        return e

    hr_c1 = _headroom(
        "dv_headroom_raw_discrimination_index",
        dv_name="prop_ratio_raw_median",
        criterion_threshold=RAW_RATIO_FLOOR,
        control_values=synth_ratio_values,
        statistic="max_abs",
        description=("C1's bar on the raw-stage discrimination index must lie "
                     "inside what that index can reach on this head. Achievable "
                     "is measured from the IN-LOOP POSITIVE CONTROL (synthetic "
                     "genuinely-differentiated summaries, real K, real row norm, "
                     "real trained head, live rule_state), not from this run's own "
                     "treatment cells. The index also has an ANALYTIC ceiling of K "
                     "(the candidate count, 32 here) and an analytic null of 0 "
                     "(candidate-uniform), so both ends of its range are known. "
                     "WHAT THIS ENTRY CAN AND CANNOT CERTIFY: because the index is "
                     "scale-free, randn-differentiated summaries through any head "
                     "with a live rule pathway read close to the iid reference "
                     "(~5.15 median at K=32), so in practice this entry certifies "
                     "that the measurement path is LIVE and the bar is reachable -- "
                     "it is not a sensitive test and is expected to pass. The bar's "
                     "two-sidedness rests on the MEASURED cell values (0.59-4.11 at "
                     "probe scale), not on this entry."),
        control="in-loop synthetic differentiated-summary positive control",
    )
    hr_c2 = _headroom(
        "dv_headroom_argmax_flip_fraction",
        dv_name="pooled_rule_flip_frac",
        criterion_threshold=FLIP_FRAC_FLOOR,
        achievable=1.0,
        dv_bounds=(0.0, 1.0),
        description=("C2's DV is a PROPORTION, so its achievable range is "
                     "ARITHMETIC ([0,1]) rather than sampled -- dv_achievable's "
                     "analytic route. Its null is 0 EXACTLY: a candidate-uniform "
                     "raw shift through a monotone squash cannot change an argmax. "
                     "The binding feasibility constraint on a 0.02 floor is "
                     "RESOLUTION (1/n with n pooled ticks), which is gated "
                     "separately as flip_resolution_sufficient. The synthetic "
                     "positive control is deliberately NOT the achievable "
                     "measurement here -- see FLIP_TICKS_FLOOR's comment for the "
                     "geometry mismatch that makes it a harder test than the "
                     "measurement itself."),
        control="arithmetic bound of a proportion, plus a separate resolution gate",
    )
    hr_c4 = _headroom(
        "dv_headroom_sd078_ratio_contrast",
        dv_name="sd078_ratio_contrast_on_minus_off",
        criterion_threshold=SD078_CONTRAST_MARGIN,
        control_values=off_ratio_values,
        statistic="range",
        description=("C4 reads a BETWEEN-ARM DIFFERENCE, so its achievable "
                     "measurement is the realised seed-to-seed RANGE of the index "
                     "within the CONTROL ARM ONLY (dv_achievable's documented "
                     "'range' case). Deliberately not the across-ALL-cells range: "
                     "that is the treatment effect, so a large effect would certify "
                     "its own headroom and a genuine SD-078 null would read as an "
                     "unreachable DV. A margin larger than the index's control-arm "
                     "spread could not be resolved from seed noise."),
        control="ARM_OFF seed-to-seed range of prop_ratio_raw_median",
    )
    # hr_c4 is SD-078-SCOPED and is deliberately NOT in the run-level readiness
    # conjunction. Folding it in would let an SD-078 null void an SD-082 result via
    # substrate_not_ready_requeue -- the same veto the C1 change above removes, by
    # a different route. It gates SD-078's direction only.
    headroom_entries = [hr_c1, hr_c2, hr_c4]
    headroom_ok = all(bool(e["met"]) for e in (hr_c1, hr_c2))
    sd078_headroom_ok = bool(hr_c4["met"])

    # C2's RESOLUTION gate: with n pooled ticks the smallest expressible non-zero
    # flip fraction is 1/n, so a FLIP_FRAC_FLOOR of 0.02 is meaningless below 50
    # ticks. Gated here rather than folded into the headroom entry because it is a
    # property of the SCHEDULE, not of the DV's range.
    total_flip_ticks_pre = sum(int(r["n_flip_ticks"]) for r in rows)
    flip_resolution_ok = bool(total_flip_ticks_pre >= FLIP_TICKS_FLOOR)

    # (iii) SD-082 is a claim about a TRAINED consumer, so "the head actually
    # trained" must gate the verdict rather than sit in diagnostic_flags. Without
    # it, the cross-candidate structure a random first layer + ReLU already
    # possesses could be reported as a trained coupling. Probe last-layer weight
    # deltas were 0.0064-0.0118 against this 1e-3 floor (6-12x), so it is a real
    # gate that the intended schedule clears comfortably.
    last_layer_worst, last_layer_cell = _worst_cell(
        rows, "last_layer_weight_delta_init_to_p1", mode="min")
    head_trained = bool(last_layer_worst >= LAST_LAYER_WEIGHT_DELTA_FLOOR)

    ready = (cone_present and on_diff_pool and off_pinned_pool and on_active
             and prop_samples_sufficient and fix_engaged and capture_active
             and head_diag_sufficient and spread_measurable
             and replica_ok and uniform_reads_uniform and controls_sampled
             and flip_resolution_ok and head_trained and headroom_ok)

    # --- PRE-REGISTERED ACCEPTANCE (SD-082's own). ---
    # --- LEGACY MAGNITUDE DV: measured, and APPLIED TO BOTH ARMS. --------------
    # 822d computed exactly these two numbers, gated on the ON one alone, and never
    # applied the floor to the OFF arm that its own manifest showed clearing it more
    # strongly. Here both are computed, the floor is evaluated on BOTH, and the
    # resulting "can this floor discriminate at all" verdict is recorded. It gates
    # NOTHING -- it is the negative-control application that 822d omitted, kept
    # visible so the defect cannot silently recur.
    on_prop = statistics.fmean(r["prop_delta_mean"] for r in on)
    off_prop = statistics.fmean(r["prop_delta_mean"] for r in off)
    legacy_floor_on = bool(on_prop >= PROP_NONVAC_FLOOR)
    legacy_floor_off = bool(off_prop >= PROP_NONVAC_FLOOR)
    legacy_floor_discriminates = bool(legacy_floor_on and not legacy_floor_off)

    # --- 822e's LOAD-BEARING DV: the RAW-STAGE DISCRIMINATION INDEX -----------
    # THE ATTRIBUTION RULE, AND WHY IT IS BUILT INTO THE SHAPE OF THE CRITERIA.
    # The arm axis of this experiment is `crf_cue_centering`, which is SD-078's
    # rule-POOL differentiation knob. SD-082's own fix
    # (candidate_summary_source="proposer_post_action") is passed UNCONDITIONALLY
    # in BOTH arms -- see _make_agent. So the ON-vs-OFF contrast cannot attribute
    # to SD-082 under any reading: both arms carry SD-082's fix, and ARM_OFF is a
    # rule-pool control, not a summary-fix-off control. An earlier draft of this
    # letter routed SD-082's direction off exactly that contrast, which is 822d's
    # own withdrawn misattribution shape with the claim roles swapped.
    #
    # The fix is structural, not a caveat in prose:
    #   * SD-082's question -- is the propagated bias candidate-DISCRIMINATING or
    #     candidate-UNIFORM -- is an ABSOLUTE property of the propagated bias,
    #     measurable in EITHER arm. C1 and C2 are therefore ABSOLUTE criteria
    #     evaluated IDENTICALLY IN BOTH ARMS, and SD-082's direction reads from
    #     them and from nothing else.
    #   * The ON-minus-OFF contrast is a legitimate SD-078 criterion. It is C4,
    #     and SD-078's direction reads from C4 and from nothing else.
    # dir_082 below is computed WITHOUT reference to any contrast quantity, and
    # dir_078 WITHOUT reference to C1/C2, so no pool-contrast result can set an
    # SD-082 direction even if a later edit reorders the label ladder.
    per_arm_ratio: Dict[str, Any] = {}
    for label_arm, group in (("ARM_ON", on), ("ARM_OFF", off)):
        vals = [r["prop_ratio_raw_median"] for r in group]
        n_clear = sum(1 for v in vals if v >= RAW_RATIO_FLOOR)
        per_arm_ratio[label_arm] = {
            "per_seed_median_ratio": {r["seed"]: r["prop_ratio_raw_median"]
                                      for r in sorted(group, key=lambda x: x["seed"])},
            "n_seeds_clearing_floor": int(n_clear),
            "n_seeds": len(vals),
            "worst_seed_median_ratio": min(vals) if vals else 0.0,
            "mean_median_ratio": statistics.fmean(vals) if vals else 0.0,
            "clears": bool(n_clear >= RATIO_SEED_MAJORITY),
        }
    # C1: the absolute floor, gating on the arm where the substrate is in its
    # INTENDED OPERATING REGIME (ARM_ON, differentiated pool). The IDENTICAL floor
    # is evaluated on ARM_OFF and reported as C1b -- applied, never hidden -- but
    # it does NOT gate.
    #
    # WHY NOT REQUIRE BOTH (a red-team finding this design originally had wrong).
    # Requiring the pool-control arm to clear an absolute SD-082 bar hands SD-078's
    # knob a veto over SD-082's direction, which is the very coupling the
    # attribution block above exists to break -- in mirror image. Concretely: with
    # ARM_ON clearing 5/5 and ARM_OFF 2/5 the run would record SD-082 as
    # "inconclusive" when its own ON-arm evidence is unambiguous, and at probe
    # scale ARM_OFF was already a coin flip (medians 2.81 and 0.59 against this 1.0
    # floor). So the conjunction is not conservatism, it is a second
    # misattribution.
    #
    # AND THIS IS NOT 822d's DEFECT RETURNING. 822d's C1 was ON-only AND its
    # negative control was never evaluated against it AND the result was presented
    # as evidence about the pool contrast. Here the floor IS evaluated on the
    # control arm and recorded (C1b), and the pool contrast has its own separate
    # criterion (C4). Note also what the control means here: BOTH arms carry
    # SD-082's own fix, so ARM_OFF clearing the floor is CONFIRMATORY for SD-082
    # rather than contaminating, and ARM_OFF failing it is a fact about the
    # rule-pool arm, not about the consumer.
    c1 = bool(per_arm_ratio["ARM_ON"]["clears"])
    c1b_off_clears = bool(per_arm_ratio["ARM_OFF"]["clears"])

    # C2: POOLED argmax-flip fraction over every cell of BOTH arms. Genuinely
    # INDEPENDENT of C1 rather than implied by it: a propagated bias with a large
    # cross-candidate range that nonetheless preserves the ordering of bias(0)
    # flips nothing (high C1, C2 = 0), and that ordering-preserving state is
    # precisely the "invisible to every argmax-derived downstream reader" case the
    # autopsy names. C1 does not imply C2 and C2 does not imply C1 at a nonzero
    # floor. (The first draft's C2 -- "the mean contrast is positive" -- was
    # implied ARITHMETICALLY by its C1 "all five per-seed contrasts are positive",
    # so it was not a second gate at all and its failure branch was dead code.)
    total_flips = sum(int(r["n_flips"]) for r in rows)
    total_flip_ticks = sum(int(r["n_flip_ticks"]) for r in rows)
    pooled_flip_frac = (total_flips / total_flip_ticks) if total_flip_ticks else 0.0
    c2 = bool(total_flip_ticks > 0 and pooled_flip_frac >= FLIP_FRAC_FLOOR)

    # C4: SD-078's ONLY criterion -- the between-arm contrast on the same index.
    off_by_seed = {r["seed"]: r for r in off}
    per_seed_contrast: List[Dict[str, Any]] = []
    for r in sorted(on, key=lambda x: x["seed"]):
        o = off_by_seed.get(r["seed"])
        if o is None:
            continue
        d = r["prop_ratio_raw_median"] - o["prop_ratio_raw_median"]
        per_seed_contrast.append({
            "seed": r["seed"],
            "on_ratio_median": r["prop_ratio_raw_median"],
            "off_ratio_median": o["prop_ratio_raw_median"],
            "on_minus_off": d,
            "passes_margin": bool(d >= SD078_CONTRAST_MARGIN),
            # raw and post-tanh spreads carried alongside, per seed, both arms
            "on_prop_spread_raw_mean": r["prop_spread_raw_mean"],
            "off_prop_spread_raw_mean": o["prop_spread_raw_mean"],
            "on_prop_spread_post_tanh_mean": r["prop_spread_mean"],
            "off_prop_spread_post_tanh_mean": o["prop_spread_mean"],
            "on_prop_delta_mean": r["prop_delta_mean"],
            "off_prop_delta_mean": o["prop_delta_mean"],
            "legacy_on_minus_off": r["prop_delta_mean"] - o["prop_delta_mean"],
        })
    n_contrast_seeds = sum(1 for c in per_seed_contrast if c["passes_margin"])
    deltas = [c["on_minus_off"] for c in per_seed_contrast]
    contrast_mean = statistics.fmean(deltas) if deltas else 0.0
    contrast_sd = statistics.stdev(deltas) if len(deltas) >= 2 else 0.0
    contrast_t_like = (contrast_mean / (contrast_sd / math.sqrt(len(deltas)))
                       if contrast_sd > 0.0 and deltas else 0.0)
    legacy_deltas = [c["legacy_on_minus_off"] for c in per_seed_contrast]
    legacy_contrast_mean = statistics.fmean(legacy_deltas) if legacy_deltas else 0.0
    c4 = bool(len(per_seed_contrast) == len(SEEDS)
              and n_contrast_seeds >= SD078_CONTRAST_SEEDS
              and contrast_mean >= SD078_CONTRAST_MARGIN)
    # The mirror, for the WEAKENS direction -- so C4 is two-sided rather than a
    # gate that can only ever support (see the per-claim direction block).
    n_contrast_seeds_neg = sum(1 for c in per_seed_contrast
                               if c["on_minus_off"] <= -SD078_CONTRAST_MARGIN)
    c4_negative = bool(len(per_seed_contrast) == len(SEEDS)
                       and n_contrast_seeds_neg >= SD078_CONTRAST_SEEDS
                       and contrast_mean <= -SD078_CONTRAST_MARGIN)

    # OVERALL PASS is SD-082's question only -- C4 is deliberately NOT in this
    # conjunction. SD-078 is a CO-TAG here, not the subject: its own
    # differentiation is already verified as a readiness precondition, and letting
    # an SD-078 null turn an SD-082 success into a run-level FAIL would reproduce
    # exactly the blurring this letter exists to remove.
    overall = bool(ready and c1 and c2)

    # --- Non-degeneracy (applies to evidence runs -- the V3-EXQ-514m net). ---
    base_non_degenerate = bool(fix_engaged and prop_samples_sufficient
                               and spread_measurable and replica_ok
                               and uniform_reads_uniform and controls_sampled
                               and flip_resolution_ok and headroom_ok
                               and len(per_seed_contrast) == len(SEEDS))
    c1_non_degenerate = bool(
        base_non_degenerate
        and all(r["n_prop_ratio_raw_samples"] > 0 for r in rows))
    c2_non_degenerate = bool(base_non_degenerate and total_flip_ticks > 0)
    c4_non_degenerate = bool(
        base_non_degenerate and len(set(round(d, 12) for d in deltas)) > 1)

    # --- Diagnostic flags (informational; not gating, all components recorded). ---
    on_dead_relu = statistics.fmean(r["hidden_dead_relu_frac_p2_mean"] for r in on)
    off_dead_relu = statistics.fmean(r["hidden_dead_relu_frac_p2_mean"] for r in off)
    on_last_layer_delta = statistics.fmean(
        r["last_layer_weight_delta_init_to_p1"] for r in on)
    live_ratios = [r["rule_summary_magnitude_ratio_p2_median"] for r in on
                   if r["n_magnitude_ratio_live_samples"] > 0]
    magnitude_in_band = all(MAGNITUDE_RATIO_LOW <= x <= MAGNITUDE_RATIO_HIGH
                            for x in live_ratios) if live_ratios else False
    on_flip = statistics.fmean(r["rule_flip_frac"] for r in on)
    off_flip = statistics.fmean(r["rule_flip_frac"] for r in off)

    diagnostic_flags = {
        "candidate_summary_fix_engaged": fix_engaged,
        "magnitude_ratio_in_band": magnitude_in_band,
        "rule_flip_observed": bool(on_flip > 0.0 or off_flip > 0.0),
        "dead_relu_confirmed": bool(on_dead_relu >= DEAD_RELU_HIGH_FLOOR),
        "dead_relu_partial_contributor": bool(
            DEAD_RELU_MED_FLOOR <= on_dead_relu < DEAD_RELU_HIGH_FLOOR),
        "head_untrained_last_layer_static": bool(
            abs(on_last_layer_delta) < LAST_LAYER_WEIGHT_DELTA_FLOOR),
    }

    # Label ladder ordered by CAUSAL DEPTH, not by the order the hypotheses were
    # written -- autopsy Section 5's third lesson (822c's own ladder suppressed its
    # best finding by testing a proxy measure before a direct one).
    worst_arm_ratio = min(per_arm_ratio["ARM_ON"]["worst_seed_median_ratio"],
                          per_arm_ratio["ARM_OFF"]["worst_seed_median_ratio"])
    best_arm_ratio = max(per_arm_ratio["ARM_ON"]["mean_median_ratio"],
                         per_arm_ratio["ARM_OFF"]["mean_median_ratio"])
    # "Reads as candidate-UNIFORM" is a POSITIVE finding about the substrate and is
    # the only state that may weaken SD-082. It requires the index to sit near its
    # analytic null (0) in BOTH arms -- not merely to miss C1's bar.
    #
    # CAVEAT A LATER READER MUST APPLY, recorded here rather than left implicit.
    # The index also reads ~0 when the hidden ReLU mask is IDENTICAL across
    # candidates, which is an UPSTREAM property (the centered candidate summaries
    # were too similar to flip any hidden unit) rather than a property of SD-082's
    # consumer. The `candidate_summary_spread_survives_centering` readiness gate
    # excludes the severe form of that (post/pre centering norm ratio below 1e-3),
    # so a `weakens` here is already conditional on the input not being starved --
    # but the gate is a floor, not a guarantee of mask heterogeneity. Any autopsy
    # reading a `weakens` from this run MUST check
    # metrics.summary_spread_ratio_worst against that floor before attributing the
    # uniformity to the consumer.
    reads_uniform = bool(best_arm_ratio < UNIFORM_VERDICT_CEIL)

    if not ready:
        label = "substrate_not_ready_requeue"
    elif c1 and c2:
        label = "sd082_candidate_discriminating_bias_confirmed"
    elif reads_uniform:
        label = "propagated_bias_candidate_uniform_sd082_not_supported"
    elif c1 and not c2:
        label = "spread_present_but_not_argmax_consequential"
    else:
        label = "sd082_discrimination_inconclusive_underpowered"

    # --- PER-CLAIM DIRECTIONS -------------------------------------------------
    # dir_082 is a function of C1/C2 and the uniform-verdict floor ONLY -- no
    # contrast quantity appears in it. dir_078 is a function of C4 ONLY -- no
    # absolute criterion appears in it. That separation is the structural fix for
    # the misattribution described at the top of this block, and it must survive
    # any later edit: a pool-contrast result must remain incapable of setting an
    # SD-082 direction.
    #
    # BOTH directions are three-valued. A criterion that can only ever SUPPORT is
    # not a test, and a null that the design lacks the power to distinguish from a
    # small true effect must not be recorded as "weakens" -- so the middle band
    # routes to "inconclusive" rather than to a claim verdict in either direction.
    if not ready:
        dir_082 = dir_078 = overall_dir = "unknown"
    else:
        if c1 and c2:
            dir_082 = "supports"
        elif reads_uniform:
            dir_082 = "weakens"
        else:
            dir_082 = "inconclusive"
        if not sd078_headroom_ok:
            # C4's own bar is out of reach of the control arm's seed noise, so this
            # run cannot adjudicate SD-078 either way. It says nothing about SD-082.
            dir_078 = "unknown"
        elif c4:
            dir_078 = "supports"
        elif c4_negative:
            dir_078 = "weakens"
        else:
            dir_078 = "inconclusive"
        # The run's own question is SD-082's, so the overall direction is SD-082's.
        overall_dir = dir_082

    metrics = {
        # --- LOAD-BEARING: the raw pre-tanh discrimination index --------------
        "raw_ratio_per_arm": per_arm_ratio,
        "on_prop_ratio_raw_mean": statistics.fmean(
            r["prop_ratio_raw_median"] for r in on),
        "off_prop_ratio_raw_mean": statistics.fmean(
            r["prop_ratio_raw_median"] for r in off),
        "worst_cell_prop_ratio_raw_median": float(worst_arm_ratio),
        "pooled_rule_flip_frac": float(pooled_flip_frac),
        "n_flips_pooled": int(total_flips),
        "n_flip_ticks_pooled": int(total_flip_ticks),
        "reads_candidate_uniform": bool(reads_uniform),
        "c1b_off_arm_clears_same_floor": c1b_off_clears,
        # INTERPRETIVE REFERENCE, computed analytically, not a bar. For K
        # independent zero-mean values the index concentrates near a known value
        # (5.15 median, 4.12-6.57 5th-95th at K=32). Cell medians BELOW it indicate
        # a common-mode (candidate-uniform) component in the propagated bias;
        # values at or above it indicate cross-candidate structure as strong as
        # independent noise. Recorded so a reader can situate the measured index
        # instead of reading 1.0 as though it were the natural scale.
        "iid_reference_index_median_k32": 5.15,
        "iid_reference_index_p5_p95_k32": [4.12, 6.57],
        "last_layer_weight_delta_worst": float(last_layer_worst),
        "last_layer_weight_delta_worst_cell": last_layer_cell,
        "readiness_head_trained": head_trained,
        "sd078_headroom_ok": sd078_headroom_ok,
        # --- SD-078's contrast (C4), kept strictly separate -------------------
        "sd078_ratio_contrast_mean": contrast_mean,
        "sd078_ratio_contrast_sd": contrast_sd,
        "n_sd078_contrast_seeds_positive": int(n_contrast_seeds),
        "n_sd078_contrast_seeds_negative": int(n_contrast_seeds_neg),
        # --- controls ---------------------------------------------------------
        "synth_ratio_median_values": synth_ratio_values,
        "synth_flip_frac_values": synth_flip_values,
        "uniform_control_ratio_worst": float(uniform_worst),
        "uniform_control_worst_cell": uniform_cell,
        "raw_replica_atanh_err_worst": float(replica_worst),
        "raw_replica_worst_cell": replica_cell,
        "max_bias_over_scale_worst": max(
            (r["max_bias_over_scale"] for r in rows), default=0.0),
        "dv_headroom": {e["name"]: {"measured": e["measured"],
                                    "threshold": e["threshold"],
                                    "headroom_ratio": e.get("headroom_ratio"),
                                    "met": e["met"]} for e in headroom_entries},
        # --- POST-TANH spread: DIAGNOSTIC ONLY in 822e (see _raw_stage_prop) ---
        "on_prop_spread_post_tanh_mean": statistics.fmean(
            r["prop_spread_mean"] for r in on),
        "off_prop_spread_post_tanh_mean": statistics.fmean(
            r["prop_spread_mean"] for r in off),
        "on_prop_spread_raw_mean": statistics.fmean(
            r["prop_spread_raw_mean"] for r in on),
        "off_prop_spread_raw_mean": statistics.fmean(
            r["prop_spread_raw_mean"] for r in off),
        "synthetic_spread_probe_worst": float(synth_worst),
        "synthetic_spread_probe_worst_cell": synth_cell,
        # --- legacy magnitude DV, APPLIED TO BOTH ARMS (822d gated on ON alone) ---
        "on_prop_delta_mean": on_prop,
        "off_prop_delta_mean": off_prop,
        "legacy_floor_cleared_by_on": legacy_floor_on,
        "legacy_floor_cleared_by_off": legacy_floor_off,
        "legacy_floor_discriminates": legacy_floor_discriminates,
        "legacy_contrast_mean": legacy_contrast_mean,
        "on_rule_state_diff_mean": statistics.fmean(r["rule_state_diff"] for r in on),
        "off_rule_state_diff_mean": statistics.fmean(r["rule_state_diff"] for r in off),
        "n_contrast_seeds_passing": n_contrast_seeds,
        "n_contrast_seeds_total": len(per_seed_contrast),
        "contrast_mean_on_minus_off": contrast_mean,
        "contrast_sd_on_minus_off": contrast_sd,
        "contrast_t_like_diagnostic_only": contrast_t_like,
        "per_seed_contrast": per_seed_contrast,
        "readiness_raw_replica_matches_substrate": replica_ok,
        "readiness_uniform_control_reads_as_uniform": uniform_reads_uniform,
        "readiness_control_probes_sampled": controls_sampled,
        "readiness_dv_headroom_ok": headroom_ok,
        "readiness_flip_resolution_ok": flip_resolution_ok,
        "n_flip_ticks_pooled_pre": int(total_flip_ticks_pre),
        "readiness_cone_present": cone_present,
        "readiness_on_diff_pool": on_diff_pool,
        "readiness_off_pinned_pool": off_pinned_pool,
        "readiness_on_active": on_active,
        "readiness_prop_samples_sufficient": prop_samples_sufficient,
        "readiness_fix_engaged": fix_engaged,
        "readiness_spread_ok": spread_ok,
        "readiness_no_degenerate_tick": no_degenerate_tick,
        "readiness_dispatch_engaged": dispatch_engaged,
        "readiness_capture_active": capture_active,
        "readiness_head_diag_sufficient": head_diag_sufficient,
        "zworld_cone_min_cosine_worst": cone_worst,
        "zworld_cone_worst_cell": cone_cell,
        "n_prop_samples_worst": n_prop_worst,
        "n_prop_samples_worst_cell": n_prop_cell,
        "summary_spread_ratio_worst": spread_worst,
        "summary_spread_ratio_worst_cell": spread_cell,
        "n_summary_fallback_worst": n_fallback_worst,
        "n_summary_degenerate_ticks_worst": n_degen_worst,
        "on_rule_state_norm_p2_median_values": [
            r["rule_state_norm_p2_median"] for r in on],
        "on_rule_summary_magnitude_ratio_p2_median_values": live_ratios,
        "on_hidden_dead_relu_frac_p2_mean": on_dead_relu,
        "off_hidden_dead_relu_frac_p2_mean": off_dead_relu,
        "on_last_layer_weight_delta_init_to_p1_mean": on_last_layer_delta,
        "on_rule_flip_frac_mean": on_flip,
        "off_rule_flip_frac_mean": off_flip,
        "c1_pass": c1,
        "c2_pass": c2,
        "c4_pass": c4,
        "c4_negative": c4_negative,
        "diagnostic_flags": diagnostic_flags,
    }

    result: Dict[str, Any] = {
        "outcome": "PASS" if overall else "FAIL",
        "evidence_direction": overall_dir,
        "evidence_direction_per_claim": {"SD-078": dir_078, "SD-082": dir_082},
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "zworld_common_mode_cone_present",
                 "description": "min pairwise z_world cosine clears the cone floor (== 822a/822c).",
                 "measured": cone_worst, "threshold": CONE_MIN_COSINE_FLOOR,
                 "direction": "lower", "offending_cell": cone_cell, "met": cone_present},
                {"name": "on_pool_differentiated",
                 "description": "ARM_ON differentiated on >= majority of seeds (== 822a/822c).",
                 "measured": float(n_on_diff),
                 "threshold": float(math.ceil(SEED_PASS_FRACTION * len(on))),
                 "direction": "lower", "met": on_diff_pool},
                {"name": "off_pool_pinned",
                 "description": "ARM_OFF pool pinned on >= majority of seeds (== 822a/822c).",
                 "measured": float(n_off_pinned),
                 "threshold": float(math.ceil(SEED_PASS_FRACTION * len(off))),
                 "direction": "lower", "met": off_pinned_pool},
                {"name": "on_rule_active_p2",
                 "description": ("count of ARM_ON seeds whose P2 rule-active fraction "
                                 "clears FRAC_ACTIVE_FLOOR, over a majority of seeds "
                                 "(== 822a/822c in substance). `measured` is the SEED "
                                 "COUNT, not the mean fraction: `met` quantifies over "
                                 "seeds, and a mean clearing the floor can coexist with "
                                 "a minority of seeds clearing it, which the indexer's "
                                 "recompute would silently pass. Per-seed fractions are "
                                 "in per_seed_rows[].crf_frac_active_p2."),
                 "measured": float(n_on_active),
                 "threshold": float(math.ceil(SEED_PASS_FRACTION * len(on))),
                 "direction": "lower", "met": on_active},
                {"name": "spread_dv_measurable",
                 "description": ("POSITIVE CONTROL for the statistic C1/C2 route on. "
                                 "The REAL trained head is fed synthetic, genuinely "
                                 "differentiated candidate summaries and the SAME "
                                 "cross-candidate range is read off it. A flat zero "
                                 "contrast on real candidates has two causes -- the "
                                 "propagated bias is candidate-uniform (the finding), or "
                                 "nothing could have registered a spread at all (head "
                                 "last layer still at its zero init, dead measurement "
                                 "path). Without this control they are indistinguishable "
                                 "and a starved run self-routes a falsification. Worst "
                                 "cell, not the mean, because `met` quantifies over "
                                 "cells."),
                 "measured": float(synth_worst), "threshold": SPREAD_MEASURABLE_FLOOR,
                 "direction": "lower", "control": "synthetic differentiated summaries "
                                                  "through the real trained head",
                 "offending_cell": synth_cell, "met": spread_measurable},
                {"name": "prop_samples_sufficient",
                 "description": ("worst-cell n_prop_samples clears the floor -- the gate "
                                 "822/822a/822b lacked, where n_prop_samples was 0 in all "
                                 "18 cells and fmean's empty-list default of 0.0 was read "
                                 "as a measured structural zero for a month."),
                 "measured": n_prop_worst, "threshold": float(PROP_SAMPLE_FLOOR),
                 "direction": "lower", "offending_cell": n_prop_cell,
                 "met": prop_samples_sufficient},
                {"name": "candidate_summary_spread_survives_centering",
                 "description": ("worst-cell MINIMUM over P2 ticks of "
                                 "post_centering_norm / pre_centering_norm -- the "
                                 "cross-candidate SPREAD the load-bearing prop_delta "
                                 "criteria depend on, not a magnitude. Legacy ws[0,0,:] "
                                 "measures ~2.2e-9 here; proposer_post_action ~1.8e-2."),
                 "measured": spread_worst, "threshold": SUMMARY_SPREAD_RATIO_FLOOR,
                 "direction": "lower", "offending_cell": spread_cell,
                 "control": ("Step 2.5a probe, 40 real ticks at seed 101: legacy "
                             "post-centering norm 5.66e-08 vs proposer_post_action "
                             "4.01e-01 on the same candidates"),
                 "met": spread_ok},
                {"name": "no_degenerate_centering_tick",
                 "description": ("the substrate's own candidate_summary_degenerate guard "
                                 "never fired on any P2 tick in any cell."),
                 "measured": n_degen_worst, "threshold": 0.0, "direction": "upper",
                 "offending_cell": degen_cell, "met": no_degenerate_tick},
                {"name": "summary_dispatch_engaged",
                 "description": ("_candidate_world_summaries() returned non-None on every "
                                 "read; the manual ws[0,0,:] fallback (the 822c path) was "
                                 "taken zero times."),
                 "measured": n_fallback_worst, "threshold": 0.0, "direction": "upper",
                 "offending_cell": fallback_cell, "met": dispatch_engaged},
                {"name": "head_diagnostics_capture_active",
                 "description": ("capture_head_diagnostics echoes True and the synthetic "
                                 "init positive control returns finite, in-range values "
                                 "in every cell (== 822c)."),
                 "measured": capture_worst, "threshold": 1.0, "direction": "lower",
                 "control": ("synthetic differentiated 3-candidate forward pass "
                             "immediately post-construction, pre-training"),
                 "offending_cell": capture_offending, "met": capture_active},
                {"name": "head_diag_samples_sufficient",
                 "description": "worst-cell count of P2 head-diagnostic reads clears the floor (== 822c).",
                 "measured": n_diag_worst, "threshold": float(HEAD_DIAG_SAMPLE_FLOOR),
                 "direction": "lower", "offending_cell": n_diag_cell,
                 "met": head_diag_sufficient},
                {"name": "raw_stage_replica_matches_substrate",
                 "description": (
                     "The load-bearing DV is read at the RAW PRE-TANH stage, which "
                     "the driver recomputes from the substrate's own head rather "
                     "than by editing the module under validation. This gate "
                     "confirms the replica IS that path: it compares the replica "
                     "against the substrate's real output inverted through the "
                     "exact analytic inverse raw = bias_scale*atanh(bias/bias_scale). "
                     "If the module's output stage ever changes, the replica cannot "
                     "silently diverge -- the run self-routes to "
                     "substrate_not_ready_requeue instead of reporting a statistic "
                     "from a path the substrate no longer takes. Worst cell, since "
                     "`met` quantifies over cells."),
                 "measured": float(replica_worst), "threshold": RAW_REPLICA_TOL,
                 "direction": "upper", "offending_cell": replica_cell,
                 "control": ("substrate's own compute_bias output, analytically "
                             "inverted through atanh"),
                 "met": replica_ok},
                {"name": "uniform_control_reads_as_uniform",
                 "description": (
                     "NEGATIVE CONTROL for C1's statistic. Candidate summaries held "
                     "CONSTANT across candidates -- the provably zero-information "
                     "input, the exact shape the legacy ws[0,0,:] read produced -- "
                     "are pushed through the real trained head at a live tick and "
                     "the SAME discrimination index is read off. It must fall BELOW "
                     "C1's own bar. WHAT IT DOES AND DOES NOT SHOW, stated plainly "
                     "rather than overclaimed: given the consumer's centering step a "
                     "candidate-constant input is annihilated to zero, so this "
                     "control reads exactly 0 BY CONSTRUCTION, and the gate is a "
                     "construction check -- it fires only if the index is "
                     "mis-implemented (wrong axis, a fallback returning a non-zero "
                     "constant, or centering removed from the substrate), not if the "
                     "propagated bias merely happens to be uniform. It therefore "
                     "does NOT on its own separate the raw stage from the post-tanh "
                     "one, since both read 0 here; the side-by-side raw-vs-post-tanh "
                     "numbers recorded under C5 are what do that. Worst (maximum) "
                     "cell."),
                 "measured": float(uniform_worst), "threshold": RAW_RATIO_FLOOR,
                 "direction": "upper", "offending_cell": uniform_cell,
                 "control": "candidate-constant summaries through the real trained head",
                 "met": uniform_reads_uniform},
                {"name": "control_probes_sampled",
                 "description": ("worst-cell count of in-loop control-probe reads "
                                 "(positive and negative) clears the floor -- both "
                                 "controls now run INSIDE the measurement loop at "
                                 "live-rule_state ticks, so a cell with none of them "
                                 "has no control at all."),
                 "measured": ctrl_n_worst, "threshold": float(HEAD_DIAG_SAMPLE_FLOOR),
                 "direction": "lower", "offending_cell": ctrl_n_cell,
                 "met": controls_sampled},
                {"name": "bias_head_actually_trained",
                 "description": (
                     "SD-082 is a claim about a TRAINED consumer, so this gates the "
                     "verdict rather than sitting in diagnostic_flags. Without it, "
                     "the cross-candidate structure that a random first layer plus "
                     "ReLU already possesses could be reported as a trained "
                     "coupling -- the load-bearing index is scale-free and would not "
                     "distinguish the two. Worst cell of the P1 last-layer weight "
                     "delta; probe measured 0.0064-0.0118 against this floor."),
                 "measured": float(last_layer_worst),
                 "threshold": float(LAST_LAYER_WEIGHT_DELTA_FLOOR),
                 "direction": "lower", "offending_cell": last_layer_cell,
                 "met": head_trained},
                {"name": "flip_resolution_sufficient",
                 "description": (
                     "C2's floor is a fraction of pooled ticks, and with n ticks the "
                     "smallest non-zero fraction expressible is 1/n -- so a 0.02 bar "
                     "is unresolvable below 50 ticks regardless of how the substrate "
                     "behaves. This gates the SCHEDULE, not the DV's range, which is "
                     "why it sits beside the dv_headroom entries rather than inside "
                     "C2's. Floor is 4x the bare resolution requirement."),
                 "measured": float(total_flip_ticks_pre),
                 "threshold": float(FLIP_TICKS_FLOOR),
                 "direction": "lower", "met": flip_resolution_ok},
                hr_c1,
                hr_c2,
                hr_c4,
            ],
            "criteria": [
                {"name": "C1_raw_discrimination_index_on_arm",
                 "load_bearing": True, "passed": c1,
                 "detail": (
                     f"ABSOLUTE floor {RAW_RATIO_FLOOR} on the RAW pre-tanh "
                     f"discrimination index (cross-candidate range / mean-abs of "
                     f"bias_raw(rule_state)-bias_raw(0)), on the arm in the intended "
                     f"operating regime: ARM_ON "
                     f"{per_arm_ratio['ARM_ON']['n_seeds_clearing_floor']}"
                     f"/{per_arm_ratio['ARM_ON']['n_seeds']} seeds clear, majority "
                     f"required {RATIO_SEED_MAJORITY}. ABSOLUTE, not a contrast: "
                     f"SD-082's question is a property of the propagated bias. The "
                     f"index is SCALE-FREE by construction, so it measures "
                     f"spread-per-magnitude and NOT magnitude -- C2 is the "
                     f"functional-consequence backstop, and C3 carries the magnitude "
                     f"reading. Reference: for K independent zero-mean values the "
                     f"index concentrates at ~5.15 (K=32), so a cell below that has "
                     f"a common-mode component.")},
                {"name": "C1b_same_floor_on_pool_control_arm",
                 "load_bearing": False, "passed": c1b_off_clears,
                 "detail": (
                     f"The IDENTICAL floor {RAW_RATIO_FLOOR}, evaluated on ARM_OFF: "
                     f"{per_arm_ratio['ARM_OFF']['n_seeds_clearing_floor']}"
                     f"/{per_arm_ratio['ARM_OFF']['n_seeds']} seeds clear. Applied "
                     f"and reported -- which is the negative-control application "
                     f"822d omitted -- but NOT gating, and that is deliberate. BOTH "
                     f"arms carry SD-082's own fix, so ARM_OFF clearing is "
                     f"CONFIRMATORY rather than contaminating, and ARM_OFF failing "
                     f"is a fact about the rule-pool arm. Making it a conjunct would "
                     f"hand SD-078's knob a veto over SD-082's direction -- the same "
                     f"misattribution this letter exists to remove, in mirror "
                     f"image.")},
                {"name": "C2_argmax_flip_pooled", "load_bearing": True, "passed": c2,
                 "detail": (
                     f"POOLED argmax-flip fraction over all cells of both arms: "
                     f"{total_flips}/{total_flip_ticks} = {pooled_flip_frac:.4g} "
                     f"against floor {FLIP_FRAC_FLOOR}. Independent of C1: an "
                     f"order-preserving propagated bias has a large index and flips "
                     f"nothing. Null is 0 exactly. Pooled rather than worst-cell "
                     f"because the per-cell rates rest on single-digit flip counts "
                     f"-- piloted, found under-powered at cell granularity.")},
                {"name": "C4_sd078_ratio_contrast", "load_bearing": True,
                 "passed": c4,
                 "detail": (
                     f"ARM_ON minus ARM_OFF on the SAME index, per seed: "
                     f"{n_contrast_seeds}/{len(per_seed_contrast)} seeds clear "
                     f"margin {SD078_CONTRAST_MARGIN} (required "
                     f"{SD078_CONTRAST_SEEDS}), mean contrast {contrast_mean:.4g}. "
                     f"THIS CRITERION BELONGS TO SD-078 ALONE -- the arm axis is "
                     f"crf_cue_centering, SD-078's rule-pool knob, while SD-082's "
                     f"own fix is ON in BOTH arms. It is load-bearing for SD-078's "
                     f"direction and DELIBERATELY EXCLUDED from the overall PASS "
                     f"conjunction, which is SD-082's question. Two-sided: "
                     f"{n_contrast_seeds_neg} seeds clear the mirrored negative "
                     f"margin (c4_negative={c4_negative}). Its dv_headroom entry "
                     f"is measured on the CONTROL ARM's seed spread only and gates "
                     f"SD-078's direction alone -- never run-level readiness, which "
                     f"would let an SD-078 null void an SD-082 result.")},
                {"name": "C3_legacy_magnitude_floor_applied_to_both_arms",
                 "load_bearing": False,
                 "passed": legacy_floor_discriminates,
                 "detail": (f"822d's ON-only floor {PROP_NONVAC_FLOOR}: ON={on_prop:.6g} "
                            f"cleared={legacy_floor_on}, OFF={off_prop:.6g} "
                            f"cleared={legacy_floor_off}. DIAGNOSTIC, gates nothing. "
                            f"passed=False here means the floor cannot discriminate -- "
                            f"which is exactly what 822d's own manifest showed and never "
                            f"applied to its negative control.")},
                {"name": "C5_post_tanh_spread_diagnostic", "load_bearing": False,
                 "passed": True,
                 "detail": (
                     "The POST-TANH cross-candidate spread -- the statistic the "
                     "first 822e draft made load-bearing -- is measured and "
                     "recorded here but gates NOTHING. tanh is monotone but not "
                     "affine, so a PERFECTLY UNIFORM raw shift emerges as a "
                     "non-uniform post-tanh delta (a uniform +0.01 at base raws 0.0 "
                     "and 0.05 gives 9.9668e-3 and 7.4932e-3, a manufactured spread "
                     "of 2.4736e-3), which is the candidate-uniform state this DV "
                     "exists to exclude. Kept visible so the two stages can be "
                     "compared in the manifest rather than argued about later.")},
            ],
            "combination_rule": (
                "overall PASS = readiness AND C1 AND C2. Both load-bearing SD-082 "
                "criteria are ABSOLUTE rather than contrasts, because SD-082's "
                "question -- is the propagated bias candidate-DISCRIMINATING or "
                "candidate-UNIFORM -- is an absolute property, and because both arms "
                "carry SD-082's own fix (candidate_summary_source="
                "'proposer_post_action' is passed unconditionally, so no ON/OFF "
                "contrast can attribute to it). C1 gates on ARM_ON, the intended "
                "operating regime; the IDENTICAL floor is evaluated on ARM_OFF and "
                "reported as C1b, applied but not gating -- see C1b's detail for why "
                "a conjunction over both arms would be a second misattribution "
                "rather than conservatism. C1 and C2 are independent of each other: "
                "an order-preserving bias clears C1 and fails C2. C1 is SCALE-FREE, "
                "so 'confirmed' means the propagated bias has cross-candidate "
                "STRUCTURE, not that it is large; C2 (a flip of the real committed "
                "selection) is what makes the structure functionally consequential, "
                "and C3 carries the magnitude reading alongside. A magnitude gate is "
                "deliberately NOT re-added -- the ratified autopsy withdrew exactly "
                "that DV as unable to separate uniform from discriminating. C4 is "
                "load-bearing for SD-078's direction ONLY: it is out of this "
                "conjunction AND out of run-level readiness, so neither an SD-078 "
                "null nor an unreachable C4 bar can void an SD-082 result. C1b, C3 "
                "and C5 gate nothing. Every failure branch that is not a positive "
                "finding routes to inconclusive, never to a claim direction."),
            "criteria_non_degenerate": {
                "C1_raw_discrimination_index_on_arm": c1_non_degenerate,
                "C1b_same_floor_on_pool_control_arm": c1_non_degenerate,
                "C2_argmax_flip_pooled": c2_non_degenerate,
                "C4_sd078_ratio_contrast": c4_non_degenerate,
                "C3_legacy_magnitude_floor_applied_to_both_arms": c1_non_degenerate,
                "C5_post_tanh_spread_diagnostic": c1_non_degenerate,
            },
            "claim_criterion_map": {
                "SD-082": {
                    "criteria": ["C1_raw_discrimination_index_on_arm",
                                 "C2_argmax_flip_pooled"],
                    "role": "primary",
                    "note": ("SD-082's direction is a function of C1, C2 and the "
                             "uniform-verdict floor ONLY. No between-arm contrast "
                             "quantity enters it. That is structural, not a caveat: "
                             "the arm axis (crf_cue_centering) is SD-078's rule-pool "
                             "knob and SD-082's own fix is ON IN BOTH ARMS, so an "
                             "ON-vs-OFF contrast cannot attribute to SD-082 under "
                             "any reading -- which is 822d's withdrawn "
                             "misattribution shape with the claim roles swapped. The "
                             "DV is read at the RAW PRE-TANH stage because the "
                             "consumer's tanh output stage manufactures apparent "
                             "cross-candidate range out of a uniform raw shift."),
                },
                "SD-078": {
                    "criteria": ["C4_sd078_ratio_contrast"],
                    "role": "secondary_mediated",
                    "note": ("SD-078's direction is a function of C4 ONLY -- the "
                             "between-arm contrast on the rule-pool axis, which is "
                             "the one comparison this design can attribute to it. "
                             "SD-078's own differentiation is additionally verified "
                             "as a readiness precondition (on_pool_differentiated / "
                             "off_pool_pinned / rule_state_diff), not as a DV. C4 "
                             "does not enter the overall PASS conjunction. A later "
                             "autopsy must NOT read a blanket verdict on this run as "
                             "a direct test of SD-078: it is a co-tag, and its "
                             "contrast is mediated entirely by SD-082's consumer."),
                },
            },
            "diagnostic_flags": diagnostic_flags,
        },
    }

    reasons: List[str] = []
    if not cone_present:
        reasons.append("z_world common-mode cone absent")
    if not on_diff_pool:
        reasons.append("ON pool not differentiated")
    if not off_pinned_pool:
        reasons.append("OFF pool not pinned")
    if not on_active:
        reasons.append("ON rule not active enough in P2")
    if not prop_samples_sufficient:
        reasons.append(f"prop samples starved (worst cell {n_prop_cell}: {n_prop_worst:g})")
    if not spread_ok:
        reasons.append(f"candidate-summary spread annihilated by centering "
                       f"(worst cell {spread_cell}: {spread_worst:g})")
    if not no_degenerate_tick:
        reasons.append(f"substrate degeneracy guard fired ({degen_cell})")
    if not dispatch_engaged:
        reasons.append(f"proposer_post_action dispatch did not engage; manual "
                       f"ws[0,0,:] fallback taken ({fallback_cell})")
    if not capture_active:
        reasons.append("head-diagnostics capture positive control failed")
    if not head_diag_sufficient:
        reasons.append("P2 head-diagnostic samples starved")
    if not replica_ok:
        reasons.append(f"raw-stage replica diverged from the substrate's own "
                       f"pre-tanh path (worst {replica_cell}: {replica_worst:g} "
                       f"> {RAW_REPLICA_TOL:g})")
    if not uniform_reads_uniform:
        reasons.append(f"uniform negative control did NOT read as uniform "
                       f"(worst {uniform_cell}: {uniform_worst:g} >= "
                       f"{RAW_RATIO_FLOOR:g}) -- C1's statistic cannot exclude "
                       f"candidate-uniformity on this substrate")
    if not controls_sampled:
        reasons.append(f"in-loop control probes starved ({ctrl_n_cell}: "
                       f"{ctrl_n_worst:g})")
    if not flip_resolution_ok:
        reasons.append(f"C2 unresolvable: {total_flip_ticks_pre} pooled flip ticks "
                       f"< {FLIP_TICKS_FLOOR} (a {FLIP_FRAC_FLOOR:g} bar needs "
                       f"{int(round(1.0 / FLIP_FRAC_FLOOR))}+ ticks to express)")
    if not head_trained:
        reasons.append(f"bias head did not train (worst cell {last_layer_cell}: "
                       f"last-layer weight delta {last_layer_worst:g} < "
                       f"{LAST_LAYER_WEIGHT_DELTA_FLOOR:g}) -- SD-082 is a claim "
                       f"about a TRAINED consumer")
    if not headroom_ok:
        unmet = [e["name"] for e in (hr_c1, hr_c2) if not e["met"]]
        reasons.append("dv_headroom unmet -- a load-bearing bar lies outside the "
                       "range its DV can reach: " + ", ".join(unmet))
    if reasons or not (c1_non_degenerate and c2_non_degenerate and c4_non_degenerate):
        if not c1_non_degenerate and "prop samples starved" not in " ".join(reasons):
            reasons.append("C1 degenerate: the raw-stage discrimination index was "
                           "not measurable, not sampled, or a seed is missing an "
                           "arm -- so a low index is starvation, not a finding")
        if not c2_non_degenerate:
            reasons.append("C2 degenerate: no argmax-flip ticks were sampled at all")
        if not c4_non_degenerate:
            reasons.append("C4 degenerate: the per-seed ON-OFF index contrast has no "
                           "variation across seeds")
        result["non_degenerate"] = False
        result["degeneracy_reason"] = "; ".join(reasons)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    global SEEDS, P0_WARMUP_EPISODES, P1_BIAS_TRAIN_EPISODES, P2_MEASUREMENT_EPISODES
    if args.dry_run:
        SEEDS = [101]
        P0_WARMUP_EPISODES = 3
        P1_BIAS_TRAIN_EPISODES = 3
        P2_MEASUREMENT_EPISODES = 3

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS,
        "p0": P0_WARMUP_EPISODES, "p1": P1_BIAS_TRAIN_EPISODES,
        "p2": P2_MEASUREMENT_EPISODES, "steps_per_episode": STEPS_PER_EPISODE,
        "candidate_summary_source": "proposer_post_action",
        "cone_min_cosine_floor": CONE_MIN_COSINE_FLOOR,
        "dist_floor": DIST_FLOOR, "min_live_rules": MIN_LIVE_RULES,
        "off_max_live_ceil": OFF_MAX_LIVE_CEIL, "frac_active_floor": FRAC_ACTIVE_FLOOR,
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "prop_nonvac_floor": PROP_NONVAC_FLOOR,
        "contrast_margin": CONTRAST_MARGIN,
        "raw_ratio_floor": RAW_RATIO_FLOOR,
        "flip_frac_floor": FLIP_FRAC_FLOOR,
        "flip_ticks_floor": FLIP_TICKS_FLOOR,
        "ratio_seed_majority": RATIO_SEED_MAJORITY,
        "sd078_contrast_margin": SD078_CONTRAST_MARGIN,
        "sd078_contrast_seeds": SD078_CONTRAST_SEEDS,
        "uniform_verdict_ceil": UNIFORM_VERDICT_CEIL,
        "dv_headroom_margin": DV_HEADROOM_MARGIN,
        "raw_replica_tol": RAW_REPLICA_TOL,
        "prop_sample_floor": PROP_SAMPLE_FLOOR,
        "summary_spread_ratio_floor": SUMMARY_SPREAD_RATIO_FLOOR,
        "head_diag_sample_floor": HEAD_DIAG_SAMPLE_FLOOR,
        "magnitude_ratio_low": MAGNITUDE_RATIO_LOW,
        "magnitude_ratio_high": MAGNITUDE_RATIO_HIGH,
        "dead_relu_high_floor": DEAD_RELU_HIGH_FLOOR,
        "dead_relu_med_floor": DEAD_RELU_MED_FLOOR,
        "last_layer_weight_delta_floor": LAST_LAYER_WEIGHT_DELTA_FLOOR,
        "lr_lpfc_bias": LR_LPFC_BIAS,
        "reinforce_batch_size": REINFORCE_BATCH_SIZE,
        "adv_min_threshold": ADV_MIN_THRESHOLD,
        "arm_config_slice_off": _config_slice(False),
        "arm_config_slice_on": _config_slice(True),
        "validates_substrate": ("SD-082 per-candidate-summary amend, ree-v3 ef88faa "
                                "(2026-08-30); routed by "
                                "failure_autopsy_V3-EXQ-822c_2026-08-29 Section 6"),
        "supersedes_note": ("SUPERSEDES V3-EXQ-822d, whose DV (prop_delta, an ablation "
                            "magnitude) could not separate a candidate-DISCRIMINATING "
                            "propagated bias from a candidate-UNIFORM one, and whose C1 "
                            "was an ON-arm-only absolute floor the negative control "
                            "cleared more strongly. Does NOT supersede v3_exq_822c -- "
                            "822c is the diagnostic that measured the original defect and "
                            "its failure_record is marked resolved (not superseded) by "
                            "the substrate fix 822d validated. The SD-082 BUILD is not "
                            "re-opened by this run."),
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "supersedes": "V3-EXQ-822d",
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
    }
    if "non_degenerate" in result:
        manifest["non_degenerate"] = result["non_degenerate"]
        manifest["degeneracy_reason"] = result["degeneracy_reason"]

    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        agent=list(_ARM_AGENTS.values()) or None,
        z_goal_stream_stats=_ZG.stats(),
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"RAW index: ON={m['on_prop_ratio_raw_mean']:.6g} "
          f"OFF={m['off_prop_ratio_raw_mean']:.6g} "
          f"worst_cell={m['worst_cell_prop_ratio_raw_median']:.6g} "
          f"(floor {RAW_RATIO_FLOOR:g}) C1={m['c1_pass']}", flush=True)
    print(f"pooled flip={m['pooled_rule_flip_frac']:.6g} "
          f"({m['n_flips_pooled']}/{m['n_flip_ticks_pooled']}) "
          f"(floor {FLIP_FRAC_FLOOR:g}) C2={m['c2_pass']}", flush=True)
    print(f"SD-078 contrast (C4 only): mean={m['sd078_ratio_contrast_mean']:.6g} "
          f"pos={m['n_sd078_contrast_seeds_positive']}/{m['n_contrast_seeds_total']} "
          f"neg={m['n_sd078_contrast_seeds_negative']} C4={m['c4_pass']}", flush=True)
    print(f"POST-TANH spread (DIAGNOSTIC ONLY): "
          f"ON={m['on_prop_spread_post_tanh_mean']:.6g} "
          f"OFF={m['off_prop_spread_post_tanh_mean']:.6g}", flush=True)
    print(f"controls: uniform_worst={m['uniform_control_ratio_worst']:.6g} "
          f"replica_err_worst={m['raw_replica_atanh_err_worst']:.3g} "
          f"bias/scale_worst={m['max_bias_over_scale_worst']:.3g}", flush=True)
    for _hn, _hv in m["dv_headroom"].items():
        print(f"  dv_headroom {_hn}: measured={_hv['measured']:.6g} "
              f"required={_hv['threshold']:.6g} met={_hv['met']}", flush=True)
    print(f"LEGACY (diagnostic, both arms): ON prop={m['on_prop_delta_mean']:.6g} "
          f"OFF prop={m['off_prop_delta_mean']:.6g} "
          f"floor_on={m['legacy_floor_cleared_by_on']} "
          f"floor_off={m['legacy_floor_cleared_by_off']} "
          f"discriminates={m['legacy_floor_discriminates']}", flush=True)
    print(f"synthetic_spread_probe_worst={m['synthetic_spread_probe_worst']:.6g}",
          flush=True)
    print(f"fix_engaged={m['readiness_fix_engaged']} "
          f"spread_worst={m['summary_spread_ratio_worst']:.3g} "
          f"fallbacks={m['n_summary_fallback_worst']:g} "
          f"degen_ticks={m['n_summary_degenerate_ticks_worst']:g} "
          f"n_prop_worst={m['n_prop_samples_worst']:g}", flush=True)
    print(f"ready(cone={m['readiness_cone_present']} diffpool={m['readiness_on_diff_pool']} "
          f"pinned={m['readiness_off_pinned_pool']} active={m['readiness_on_active']} "
          f"propsamp={m['readiness_prop_samples_sufficient']} "
          f"capture={m['readiness_capture_active']} "
          f"headdiag={m['readiness_head_diag_sufficient']})", flush=True)
    print(f"flip: ON={m['on_rule_flip_frac_mean']:.4f} OFF={m['off_rule_flip_frac_mean']:.4f} "
          f"(822c measured 0.0000 in all 6 cells -- broadcast-constant symmetry)", flush=True)
    print(f"diagnostic_flags: {m['diagnostic_flags']}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
