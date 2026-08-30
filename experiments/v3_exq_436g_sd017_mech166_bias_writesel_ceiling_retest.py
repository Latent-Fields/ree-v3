#!/opt/local/bin/python3
"""
V3-EXQ-436g -- SD-017/ARC-045/MECH-166 slot-differentiation ceiling retest on
the 436e occupied-slots-only DV, with the write-path addressing degeneracy
FIXED via the BIAS (conscience-bias usage-balancing) write-selection
mechanism
SLEEP DRIVER: manual-multi (run_sleep_cycle() called directly every SLEEP_INTERVAL episodes in training loop)

experiment_purpose: evidence

Discharges the follow-on adjudicated DUE NOW by the 2026-08-30 A.6 due-ness
assessment (chip-20260830-sd017-ceiling-retest, resolved -- this session
executes the ratified decision, it does not re-decide it). Basis (verbatim
from the assessment, retained for the audit trail): write-path occupancy is
now decisively met by all three landed write-path mechanisms (BIAS,
REFRACTORY, gumbel_learned); SD-016 has been fully armed since 2026-08-16
(V3-EXQ-436f); and the still-failing content-discrimination criterion (C2,
2-cluster occupied-slot-SET Jaccard) was diagnosed by the 2026-08-29 scoping
spike (contextmemory_write_c2_criterion_reposed_20260829.md) as a mis-posed,
near-binary, aliasing-prone instrument that must NOT gate this retest.

Same lineage: 436 -> 436a -> 436b -> 436c -> 436d -> 436e -> 436f -> 436g.

WHY THIS RUN EXISTS NOW (the occupancy blocker is resolved; the FIX that
resolves it is what changed, not the DV or the claim substrate). 436e/436f
both established, independently and by a closed-form sign discriminator,
that ContextMemory.write()'s hard `scores.mean(0).argmin()` addressing has a
deterministic single-slot fixed point under the near-constant z_world query
stream this substrate produces -- seeds 7/13/100 occupied exactly 1 of 16
slots in BOTH the WAKING_ONLY and SWS_THEN_REM arms of 436f, DESPITE the full
SD-016 production combination (cue_slot_tagger + gumbel read-side selection +
context_divergence_weight=0.5 + the 922 ctxdiv training-loop wiring) being
ARMED and CONFIRMED ENGAGED (sd016_arming_engaged: pooled applied ctxdiv loss
25,796 against a 1e-9 floor). That confirmed the occupancy collapse is a
WRITE-path defect, structurally untouched by any READ-path (SD-016) fix --
436f's own docstring pre-registered this exact joint reading as the signal to
route to a write-path successor.

The write-path fix landed 2026-08-19 as
substrate_queue.json:contextmemory-write-path-addressing-degeneracy
(`status: implemented_pending_validation`, `ready: true`). Two orthogonal
mechanisms shipped: BIAS (E1Config.contextmemory_write_usage_balancing, a
frequency-sensitive competitive-learning "conscience" penalty on the write
selection SCORE) and REFRACTORY (E1Config.contextmemory_write_selection=
"refractory", which instead restricts the ELIGIBLE SET). Both independently
CLEAR the occupancy floor -- V3-EXQ-943 (the substrate's own validation run,
run_id v3_exq_943_contextmemory_write_selection_validation_20260820T115815Z_v3)
measured BIAS at 16/16 occupied slots on 5/5 seeds (round-robin agreement
0.993-0.996, entropy ~4.0 bits, self-repeat 0) against LEGACY argmin's
still-2/5. This run uses BIAS (the task's own sketch names it explicitly);
REFRACTORY's structural k+1 floor is a distinct, un-scored alternative left
for a future letter if this one's C1 result needs a robustness check against
the write-selection mechanism choice.

WHAT CHANGES vs 436e (the ONLY changes; 436e's DV machinery -- write-occupancy
tracking, occupied-slots-only cosine, Adam-drift neutralization, the
re-derived untouched-bank null, the three-condition NO_WRITES/WAKING_ONLY/
SWS_THEN_REM design, seeds, C4 -- is otherwise INHERITED UNCHANGED. This
letter is closer to 436e than to 436f: it does NOT arm the SD-016 read-path
production combination 436f used, because 436f already demonstrated that
combination does not move write-path occupancy by even one seed. The write
addressing mechanism, not the retrieval-cue mechanism, is this letter's
manipulation-holding-constant substrate change):

1. BIAS WRITE-SELECTION ARMED, held constant across all three conditions (a
   substrate property exactly like 436e's writepath_mode and Adam-drift
   freeze -- sleep remains the sole manipulation).
   E1Config.contextmemory_write_usage_balancing = True (the "conscience-bias"
   frequency-penalty mechanism; E1Config.contextmemory_write_selection is
   LEFT AT ITS DEFAULT "argmin" -- the two mechanisms are orthogonal by
   design, usage_balancing adjusts the selection SCORE while
   write_selection="refractory" would instead restrict the ELIGIBLE SET; this
   letter uses ONLY the score-adjustment mechanism, per the task's own
   sketch). contextmemory_write_usage_bias_weight and
   contextmemory_write_usage_decay are left at their registered defaults
   (1.0, 0.99) -- the same defaults V3-EXQ-943 validated at 16/16 occupied
   slots on 5/5 seeds.
   A SEVENTH P0 gate is added: bias_occupancy_confirms_fix, the direct
   empirical readout of whether BIAS actually cleared the occupancy floor
   THIS run (rather than assuming V3-EXQ-943's substrate-validation numbers
   transfer unchanged to this driver's own env/training-loop configuration).
   Distinct from -- and NOT a duplicate of -- the existing
   sufficient_occupancy_for_c1 gate below: that gate counts PAIRED seeds
   (both WAKING_ONLY and SWS_THEN_REM scoreable); this gate reports the RAW
   per-seed occupancy achieved by BIAS across ALL non-NO_WRITES cells, so a
   partial failure (e.g. BIAS clears occupancy in WAKING_ONLY but not
   SWS_THEN_REM on some seed) is separately diagnosable rather than folded
   into a single pass/fail count.

2. THE MIS-POSED C2 CONTENT-DISCRIMINATION CRITERION IS NOT LOAD-BEARING AND
   IS NOT COMPUTED AT ALL in this run. The 2-cluster occupied-slot-SET
   Jaccard instrument (contextmemory_write_c2_criterion_reposed_20260829.md)
   is a DIFFERENT lineage's criterion (V3-EXQ-956's write-address-selection
   validation, testing whether the write mechanism ITSELF is content-
   conditioned in isolation) and was never part of the 436-family's own C1/C4
   design. It is diagnosed as near-binary and aliasing-prone at n=5 seeds --
   an untrained tagger clears its bar ~8%% of the time by chance alone -- and
   its own recommendation for the SD-017/ARC-045/MECH-166 retest specifically
   is: "proceed on the occupancy floor already met... treat content-
   discrimination as a descriptive readout inside the end-to-end retest
   (using a non-aliasing DV -- occupancy, self-repeat, round-robin index, or
   a contingency-table statistic -- never 2-cluster set-Jaccard at n=5), not
   as a precondition gate on the isolated write mechanism." This run follows
   that recommendation literally by omission: no Jaccard statistic of any
   kind is computed. The occupied-slots-only cosine DV (inherited from 436e)
   already IS a non-aliasing, graded, non-binary content-discrimination
   readout over the write-touched slots -- it is the C1 criterion itself, not
   a separate gate layered on top of it.

3. EFFECT-SIZE GATE ON C1 (no bare sign test -- 2026-08-30 standing design
   lesson from failure_autopsy_V3-EXQ-936a_2026-08-30 Section 7). The 436e/
   436f C1 criterion was a bare per-seed sign count (n_seeds_passed >= 3/5).
   This letter ADDS an aggregate effect-size requirement on top of the
   inherited sign test, scaled on the SD of the per-seed signed delta plus an
   absolute floor, per this week's standing constraint. C1 now requires BOTH:
     (a) [inherited, unchanged] the per-seed directional sign count
         (slot_cosine_sim_occupied_only(SWS_THEN_REM) <
         slot_cosine_sim_occupied_only(WAKING_ONLY)) in >= 3/5 seeds; AND
     (b) [NEW] the mean signed delta (SWS_THEN_REM - WAKING_ONLY, pooled over
         scoreable seeds) is more negative than -EFFECT_SIZE_MARGIN, where
         EFFECT_SIZE_MARGIN = max(EFFECT_SIZE_ABS_FLOOR,
         EFFECT_SIZE_SD_MULT * pooled_sd_of_delta,
         NULL_SD_MULT * untouched_bank_null_sd).
   The margin is FLOORED from three independent sources (an absolute
   constant, the run's OWN observed delta spread, and the empirically re-
   derived untouched-bank null's own spread) so a numerically tiny but
   directionally consistent effect cannot pass on sign alone -- "both tails
   floored" per this week's design constraint, in the sense that the margin
   is the MAX (not the min) of three lower bounds, so no single source can be
   gamed down to zero. Every threshold in this margin is a MEASURED quantity
   (the run's own delta SD; the empirically re-derived null's SD) or a small
   fixed constant, never inferred post-hoc from the observed C1 result
   itself.

4. THE UNTOUCHED-BANK NULL IS RE-DERIVED EMPIRICALLY AT RUN TIME, unchanged
   in mechanism from 436e (a dedicated, seeded generator sampling
   torch.randn(num_slots, memory_dim) * 0.01, matching ContextMemory's own
   init recipe, independent of the experiment's own per-seed RNG state), and
   is now used for TWO purposes rather than one: (a) [inherited] the P0
   adam_drift_neutralized calibration check on the NO_WRITES arm; (b) [NEW]
   contributing NULL_SD_MULT * null_sd as one of the three floors composing
   the point-3 effect-size margin.

CLAIM SUBSTRATE UNDER TEST (unchanged from every prior letter -- the pre-
registered comparison itself is unchanged; only the write-selection
mechanism underlying it, and the C1 effect-size gate, are repaired/tightened):
  SD-017    (sleep_phase.minimal_sleep_infrastructure_v3): "context
            representations remain globally undifferentiated" without the
            SWS/REM-analog phases -- slot_cosine_sim -> 1.0 without them.
  ARC-045   (hippocampus.bidirectional_information_flow): "an agent with
            bidirectional offline flow should show cosine_sim < 0.95
            (differentiated contexts) after sleep phases; one with only
            waking online encoding remains at cosine_sim -> 1.0 regardless
            of training duration."
  MECH-166  (hippocampus.slot_formation_filling_temporal_separation): "Slot
            structure must be consolidated during an SWS-analog phase...
            A direct test requires implementing the SWS-analog pass and
            comparing attribution map quality (context cosine_sim...) with
            vs without it." This experiment IS that direct test, on a
            write-path-repaired substrate.

DV-SYMMETRY INVARIANCE DECLARATION (mandatory per 604c net, unchanged from
436b-436f): the manipulation under test is "ran an SWS-analog + REM-analog
consolidation pass" vs "did not" (both conditions otherwise share the SAME
per-tick sense_only write path and the SAME BIAS write-selection mechanism --
points 1 above, held constant). run_sleep_cycle() directly mutates
agent.e1.context_memory.memory via ContextMemory.write() during sleep, and
slot_cosine_sim_occupied_only reads that same memory tensor, restricted to
the write-touched rows. This is not a uniform additive constant, not a
monotone rescaling, and not a permutation of interchangeable units -- the
manipulation adds an entirely separate class of write events (offline
consolidation replay) whose targets and content differ from the shared
online writes, so the occupied-slot set and the content within it can both
differ between conditions, and the DV is not invariant under any of the
three flagged symmetry classes. BIAS write-selection is a SUBSTRATE property
held CONSTANT across all three conditions, not part of the manipulation
under test -- it changes WHICH slot each write lands in, not WHETHER sleep
differentiates the resulting occupied set, so arming it uniformly cannot by
itself manufacture the C1 direction; the manipulation (sleep vs no-sleep)
still has to do the differentiating work.

CONDITIONS (3, inherited unchanged from 436e): NO_WRITES (calibration-only
negative control, sd016_writepath_mode="off", no sleep -- NOT part of the C1
comparison), WAKING_ONLY (baseline for C1, sd016_writepath_mode="sense_only",
no sleep), SWS_THEN_REM (sd016_writepath_mode="sense_only" PLUS full
SWS-then-REM cycle every SLEEP_INTERVAL episodes, plus the DR-6 context-
conditioned harm threshold in action selection -- unchanged from 436d/436e).
use_noise_floor=True on ALL THREE conditions. contextmemory_gated_content_
write=True on ALL THREE (the 436d write-path repair, held constant).
contextmemory_write_usage_balancing=True (BIAS, point 1 above) on ALL THREE.

ACCEPTANCE CRITERIA:
  P0 (gates C1's interpretability; SEVEN checks, ALL must clear):
      sws_context_memory_writes_occur / rem_attribution_rollouts_occur:
      pooled sleep-pass write/rollout counters (SWS_THEN_REM seeds) > 0 --
      unchanged from 436c-436f.
      waking_writepath_engaged: pooled per-tick sense_only write() calls
      (WAKING_ONLY seeds) > 0 -- unchanged from 436e/436f.
      sufficient_occupancy_for_c1: count of seeds where BOTH WAKING_ONLY and
      SWS_THEN_REM have >=2 occupied slots must be >= C1_N_SEEDS_REQUIRED --
      unchanged from 436e/436f, but expected to clear decisively this letter
      given V3-EXQ-943's 16/16-on-5/5-seeds BIAS validation.
      adam_drift_neutralized: max |z-score| of the NO_WRITES arm's raw
      whole-bank cosine vs the freshly-derived untouched-bank null, across
      all 5 seeds, must stay under ADAM_DRIFT_NULL_TOLERANCE_SIGMA --
      unchanged from 436e/436f.
      bias_occupancy_confirms_fix: min per-seed n_occupied_slots across ALL
      non-NO_WRITES cells (WAKING_ONLY + SWS_THEN_REM, all 5 seeds each) must
      be >= BIAS_OCCUPANCY_FLOOR -- NEW this letter, the direct empirical
      confirmation that BIAS write-selection cleared occupancy in THIS
      driver's own env/config, not merely in V3-EXQ-943's substrate-
      validation harness.
      Any unmet -> outcome FAIL, evidence_direction non_contributory, C1/C4
      NOT scored as evidence either way; interpretation label names the
      specific unmet gate.
  C1 (PRIMARY, SOLE GATE when P0 is met -- SD-017 + ARC-045 + MECH-166,
      wall-independent, PAIRED per seed, directional per ARC-045's own
      pre-registered experimental implication, PLUS the point-3 effect-size
      floor):
      (a) slot_cosine_sim_occupied_only(SWS_THEN_REM) <
          slot_cosine_sim_occupied_only(WAKING_ONLY) in >= 3/5 seeds, AND
      (b) mean signed delta (pooled over scoreable seeds) <=
          -EFFECT_SIZE_MARGIN (point 3 above).
  C4 (SECONDARY, ARC-045 slot_separation, non-gating, carried unchanged from
      436d-436f -- a read-side visitation-distribution statistic, independent
      of the write-occupancy cosine confound):
      slot_separation(SWS_THEN_REM) > 0.3 in >= 3/5 seeds.
  Secondary / exploratory (recorded, NEVER gating, unchanged from prior
  letters): harm_rate_dangerous, harm_rate_safe signed diffs.

PASS: P0 met AND C1 (both (a) and (b)).

INTERPRETATION GRID:
  P0 UNMET (any of the seven gates) -> label names the specific gate
              (sleep_cycle_recording_gap_still_present /
              waking_writepath_not_engaged / insufficient_occupancy_for_c1 /
              adam_drift_neutralization_failed /
              bias_occupancy_fix_did_not_transfer). Route to
              /failure-autopsy on the named mechanism -- C1/C4 are not
              evidence either way.
  P0 met, C1 PASS (both a and b) -> SWS-analog consolidation differentiates
              OCCUPIED context slots relative to online-writes-only, on the
              write-occupancy-corrected instrument and a write-selection
              mechanism that decisively clears occupancy. First genuine,
              non-confounded, non-vacuous support for the slot-
              differentiation prediction in this lineage. Supports SD-017,
              ARC-045, MECH-166.
  P0 met, C1(a) PASS, C1(b) FAIL -> directionally consistent but below the
              measured-noise-floor effect-size margin; genuine but WEAK
              signal, not strong enough to distinguish from the run's own
              seed-to-seed delta variance or the untouched-bank null's
              spread. Reported as weakens (the pre-registered gate is
              conjunctive), with the raw margin numbers surfaced for a
              human read rather than silently rounding to a clean PASS/FAIL.
  P0 met, C1(a) FAIL -> Genuine weakens for all three claims -- with P0 met,
              the write path, the waking control's write engagement, the
              occupancy floor (now cleared decisively by BIAS), and the
              drift-neutralization calibration are ALL confirmed this time,
              so a FAIL here is not attributable to any of the previously-
              identified confounds (occupancy collapse, SD-016 read-path
              non-transfer, Adam drift, unwritten WAKING_ONLY bank). Check
              per-seed action-class entropy before reading further (a
              degenerate waking entropy would point at the noise-floor
              magnitude, not at slot differentiation itself).

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


EXPERIMENT_TYPE = "v3_exq_436g_sd017_mech166_bias_writesel_ceiling_retest"
QUEUE_ID = "V3-EXQ-436g"
SUPERSEDES = "V3-EXQ-436f"
CLAIM_IDS = ["SD-017", "ARC-045", "MECH-166"]
EXPERIMENT_PURPOSE = "evidence"

# 436d substrate repair, held constant (unchanged -- still required so writes
# are content-bearing rather than homogenizing per the 436c defect).
CONTEXTMEMORY_GATED_CONTENT_WRITE = True

# --- 436g: BIAS write-selection (contextmemory-write-path-addressing- ------
# degeneracy substrate, IMPLEMENTED 2026-08-19; V3-EXQ-943 validated BIAS at
# 16/16 occupied slots on 5/5 seeds). Held constant across all three
# conditions -- a substrate property, not the manipulation (sleep is).
# contextmemory_write_selection is deliberately LEFT at its default "argmin";
# usage_balancing (BIAS) and write_selection="refractory" are orthogonal
# mechanisms (score-adjustment vs eligible-set-restriction) and this letter
# uses only the former, per the task's own sketch.
CONTEXTMEMORY_WRITE_USAGE_BALANCING = True
# Registered defaults (unchanged from the substrate landing / V3-EXQ-943).
CONTEXTMEMORY_WRITE_USAGE_BIAS_WEIGHT = 1.0
CONTEXTMEMORY_WRITE_USAGE_DECAY = 0.99

# Pre-registered thresholds (unchanged from 436d-436f unless noted).
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
C1_N_SEEDS_REQUIRED = 3           # >= 3/5 seeds, sign-test half of C1
C4_SLOT_SEPARATION_THRESHOLD = 0.3
C4_N_SEEDS_REQUIRED = 3
# Historical / legacy only -- calibrated against the PRE-436d write-gate
# defect. NOT recomputed against the repaired path and NOT used for gating.
ARC045_ABS_COSINE_REFERENCE_LEGACY = 0.95

# --- 436g: C1 effect-size margin (no bare sign test) ------------------------
# EFFECT_SIZE_MARGIN = max(ABS_FLOOR, SD_MULT * pooled_sd_of_delta,
#                          NULL_SD_MULT * untouched_bank_null_sd)
# All three sources are MEASURED (the run's own delta spread; the
# empirically re-derived null's spread) or a small fixed constant -- never
# inferred post-hoc from the observed C1 sign-test result.
EFFECT_SIZE_ABS_FLOOR = 0.02
EFFECT_SIZE_SD_MULT = 1.0
NULL_SD_MULT = 2.0

# P0 sleep-pass write-counter gate (unchanged concept from 436c-436f).
SWS_WRITES_FLOOR = 1.0
REM_ROLLOUTS_FLOOR = 1.0
# P0 gates inherited from 436e/436f.
WAKING_WRITEPATH_CALLS_FLOOR = 1.0
ADAM_DRIFT_NULL_TOLERANCE_SIGMA = 4.0
UNTOUCHED_BANK_NULL_DRAWS = 500
UNTOUCHED_BANK_NULL_SEED = 20260812  # fixed, reproducible; independent of experiment seeds
# P0 gate NEW this letter: BIAS must clear occupancy in THIS driver's config.
BIAS_OCCUPANCY_FLOOR = 2            # min n_occupied_slots per non-NO_WRITES cell

SLEEP_INTERVAL = 10
CONTEXT_SWITCH_EVERY = 5
TRAINING_EPISODES = 200
STEPS_PER_EPISODE = 150
EVAL_EPISODES_EACH = 30
MAX_HARM_BUF = 4000
SEEDS = [42, 7, 13, 100, 200]      # unchanged from 436/436a-436f for cross-lineage comparability

# (label, sws_enabled, rem_enabled, sd016_writepath_mode). NO_WRITES is a
# calibration-only negative control, not part of the C1 comparison.
# WAKING_ONLY and SWS_THEN_REM share writepath_mode "sense_only" -- the
# write-path repair (both content-gated writes AND BIAS write-selection) is
# held constant across the two conditions actually compared; sleep is the
# manipulation.
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
# Env / agent helpers (unchanged from 436d/436e except BIAS wiring)         #
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
    """Phase 2 substrate stack (unchanged from 436d/436e) + MECH-313/ARC-065
    noise floor (unchanged) + 436e's occupancy-DV-repair substrate
    properties (writepath_mode per-condition; Adam-drift neutralization
    applied by the caller right after construction) + 436g's BIAS
    write-selection (contextmemory_write_usage_balancing=True), held
    constant across all three conditions.
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
        # 436e DV-repair change).
        sd016_writepath_mode=writepath_mode,
        sd016_diversification_weight=SD016_DIVERSIFICATION_WEIGHT,
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_sd039_anchor_payload=True,
        # 436d write-path repair -- restores ContextMemory write_gate to a
        # modulator so writes are content-bearing. Constant across all three
        # conditions. Still required alongside 436e's occupancy fix.
        contextmemory_gated_content_write=CONTEXTMEMORY_GATED_CONTENT_WRITE,
        # 436g: BIAS write-selection, held constant across all three
        # conditions (a substrate property, not the manipulation). Resolves
        # the write-path addressing degeneracy 436e/436f both localized.
        contextmemory_write_usage_balancing=CONTEXTMEMORY_WRITE_USAGE_BALANCING,
        contextmemory_write_usage_bias_weight=CONTEXTMEMORY_WRITE_USAGE_BIAS_WEIGHT,
        contextmemory_write_usage_decay=CONTEXTMEMORY_WRITE_USAGE_DECAY,
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
    assert getattr(agent.e1.context_memory, "usage_ema", None) is not None or \
        bool(getattr(agent.e1.config, "contextmemory_write_usage_balancing", False)), (
        "contextmemory_write_usage_balancing=True did not wire onto "
        "agent.e1.config -- BIAS write-selection is not live for this "
        "experiment; REEConfig/REEAgent wiring regression."
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
# occupancy tracking (inherited verbatim from 436e)                       #
# ------------------------------------------------------------------ #

def _active_slot_idx(agent: REEAgent, z_self: torch.Tensor,
                     z_world: torch.Tensor) -> int:
    """Determine which ContextMemory slot is most strongly ACTIVATED
    (read-side argmax over ContextMemory.read()'s soft-attention scores) by
    (z_self, z_world). Used only for the C4 slot_separation visitation
    statistic -- unrelated to, and a DIFFERENT index than, the write-side
    occupancy tracked below (write() selects by the BIAS-adjusted argmin
    over query.memory, not this argmax). Unchanged from 436d-436f.
    """
    with torch.no_grad():
        cm = agent.e1.context_memory
        state = torch.cat([z_self, z_world], dim=-1)
        query = cm.query_proj(state)
        keys = cm.key_proj(cm.memory)
        scores = torch.mm(query, keys.t()) / (cm.memory_dim ** 0.5)
        idx = int(scores.argmax(dim=-1).item())
    return idx


class _WriteOccupancyTracker:
    """Accumulates the set of ContextMemory slot indices actually written to
    during a cell's run, plus a total write() call count. Populated by
    _install_write_tracker's instrumentation, which reads
    ContextMemory.write()'s own recorded outcome directly rather than
    recomputing the (now BIAS-adjusted) selection score itself -- see
    _install_write_tracker's docstring for why this is the load-bearing
    change vs 436e's read-only min_idx recomputation.
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
    sense_only/both hook, E1.update_from_observation's train_only/both hook,
    and run_sws_schema_pass's replay writes) funnel through this SAME bound
    method, so wrapping it once here captures every write() invocation
    regardless of which caller triggered it.

    UNLIKE 436e's tracker (which recomputed the plain
    scores.mean(0).argmin() expression read-only, matching write()'s LEGACY
    selection exactly), this tracker instead delegates to the ORIGINAL bound
    write() first and then reads back
    agent.e1.context_memory.last_write_index -- the substrate's own
    instrumentation field (contextmemory-write-path-addressing-degeneracy,
    ContextMemory._record_write()), which is correct for EVERY selection
    mode (argmin / refractory / gumbel_learned / BIAS-adjusted argmin) by
    construction. Re-deriving 436e's plain-argmin expression under BIAS
    would report the WRONG slot -- the substrate's own repair note is
    explicit that a stale reimplementation of the pre-fix selection rule
    disagrees with the true write() outcome under any of the three fixed
    selection modes. Reading last_write_index instead sidesteps that
    class of bug entirely.
    """
    cm = agent.e1.context_memory
    tracker = _WriteOccupancyTracker()
    orig_write = cm.write

    def _tracked_write(state: torch.Tensor) -> None:
        result = orig_write(state)
        idx = getattr(cm, "last_write_index", None)
        if idx is not None:
            tracker.record(int(idx))
        return result

    cm.write = _tracked_write  # instance-level override; class method untouched
    return tracker


def _compute_slot_cosine_sim_raw(agent: REEAgent) -> float:
    """OLD, CONFOUNDED whole-bank statistic (unchanged formula from
    436a-436f) -- averages over ALL slots including never-written ones, so
    it tracks content-similarity x occupancy-fraction rather than
    differentiation alone. Retained here ONLY for cross-letter audit
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
    """CORRECTED primary DV (inherited verbatim from 436e). Masks to slots
    that received >=1 write() call this run (per _install_write_tracker)
    before computing the mean off-diagonal cosine -- never-written slots
    (whose pairwise cosine reflects only random init, ~0) can no longer
    dilute or invert the statistic, and the reading is no longer a function
    of HOW MANY slots got touched, only of how similar the touched ones are.

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
    against THIS run's actual (num_slots, memory_dim) at run time (inherited
    verbatim from 436e). Draws n_draws fresh
    torch.randn(num_slots, memory_dim) * 0.01 initialisations -- matching
    ContextMemory.__init__'s own init recipe exactly -- and computes the
    whole-bank off-diagonal cosine mean for each, using a dedicated,
    explicitly-seeded generator so this derivation neither consumes nor is
    affected by the experiment's own per-seed RNG state. Used both for the
    P0 adam_drift_neutralized calibration check AND (436g NEW) as one of the
    three floors composing the C1 effect-size margin.
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
# unchanged from 436d-436f)                                                #
# ------------------------------------------------------------------ #

def _select_action_baseline(agent: REEAgent, z_world: torch.Tensor,
                             num_actions: int) -> Tuple[int, float]:
    """Temperature-graded softmax sample over predicted harm (low harm ->
    high selection probability), using the MECH-313 noise-floor effective
    temperature. Unchanged from 436d-436f.
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
    unchanged from 436d-436f: effective threshold =
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
# Episode runner (unchanged control flow from 436d-436f)                   #
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
) -> Tuple[float, List[torch.Tensor], List[int], List[int]]:
    """Run single episode. Returns (harm_sum, z_world_list, slot_visits,
    action_seq). action_seq is recorded to compute a per-episode action-class
    entropy diagnostic. Updates slot_danger_ema in place when train=True.
    Unchanged from 436d-436f.
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

        # Mirrors _e1_tick's append+trim and theta_buffer push directly --
        # this driver's manual harm-based action-selection loop calls
        # agent.sense() directly and never agent.act(), so both stores
        # (which run_sws_schema_pass / run_rem_attribution_pass sample from)
        # would otherwise stay permanently empty. Unchanged from 436b-436f.
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
    one episode. Unchanged from 436d-436f.
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
    use_context_cond = condition == "SWS_THEN_REM"   # DR-6 pathway only here (unchanged)

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe, sws_en, rem_en, use_sleep_loop, writepath_mode)

    # 436e DV-repair point 4: Adam-drift neutralization, held constant across
    # ALL THREE conditions. write() still mutates memory.data directly under
    # its own torch.no_grad() block regardless of requires_grad, so writes
    # are unaffected -- only gradient-descent perturbation between writes is
    # suppressed.
    agent.e1.context_memory.memory.requires_grad_(False)

    # 436e DV-repair point 1 (436g: reads write()'s own recorded outcome
    # rather than recomputing a stale selection expression -- see
    # _install_write_tracker's docstring). Installed BEFORE any training so
    # it captures every write() call for the whole cell.
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
    cum_train_pos = 0
    cum_train_neg = 0

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
                # future asymmetric-phase variant. Unchanged from 436d-436f.
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

    # 436e DV repair: read BOTH statistics right after training, before
    # agent.eval()'s eval episodes continue writing.
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
              f"sleep_passes={sleep_passes} "
              f"sws_n_writes_total={cum_sws_n_writes:.1f} "
              f"rem_n_rollouts_total={cum_rem_n_rollouts:.1f}",
              flush=True)

    verdict = "PASS" if (harm_dang < 0.04 and harm_safe < 0.04) else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "condition": condition,
        "sd016_writepath_mode": writepath_mode,
        "slot_cosine_sim_occupied_only": final_slot_sim_occupied,
        "n_occupied_slots": n_occupied_slots,
        "n_write_calls": n_write_calls,
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
        print("[DRY RUN] MECH-166/SD-017 BIAS write-selection ceiling retest "
              "smoke (seed=42, 3 conditions, enough episodes for >=1 sleep "
              "pass and >=2 occupied slots)", flush=True)
        smoke_ok = True
        smoke_results = []
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
                    "contextmemory_write_usage_balancing": CONTEXTMEMORY_WRITE_USAGE_BALANCING,
                    "sd016_writepath_mode": _wp_mode,
                    "context_memory_memory_requires_grad": False,
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
                      f"sleep_passes={r['sleep_passes']} "
                      f"sws_n_writes_total={r['sleep_cycle_sws_n_writes_total']:.1f} "
                      f"rem_n_rollouts_total={r['sleep_cycle_rem_n_rollouts_total']:.1f}")

            required_keys = {"slot_cosine_sim_occupied_only", "slot_cosine_sim_raw_whole_bank",
                             "n_occupied_slots", "n_write_calls",
                             "slot_separation", "harm_rate_safe", "harm_rate_dangerous",
                             "train_action_class_entropy_mean",
                             "sleep_cycle_sws_n_writes_total",
                             "sleep_cycle_rem_n_rollouts_total",
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
                if waking["n_write_calls"] <= 0:
                    print("  [SMOKE] FAIL: WAKING_ONLY n_write_calls=0 -- "
                          "sd016_writepath_mode='sense_only' did not fire")
                    smoke_ok = False
                # THE key NEW verification: BIAS must clear occupancy in
                # BOTH non-NO_WRITES arms, not merely reach >=2.
                if waking["n_occupied_slots"] < BIAS_OCCUPANCY_FLOOR or \
                   sws_then_rem["n_occupied_slots"] < BIAS_OCCUPANCY_FLOOR:
                    print("  [SMOKE] FAIL: BIAS write-selection did not clear "
                          f"the occupancy floor ({BIAS_OCCUPANCY_FLOOR}) in "
                          f"WAKING_ONLY ({waking['n_occupied_slots']}) or "
                          f"SWS_THEN_REM ({sws_then_rem['n_occupied_slots']}) -- "
                          "contextmemory_write_usage_balancing is not live "
                          "or not effective at smoke scale")
                    smoke_ok = False
                if waking["slot_cosine_sim_occupied_only"] is None or \
                   sws_then_rem["slot_cosine_sim_occupied_only"] is None:
                    print("  [SMOKE] FAIL: slot_cosine_sim_occupied_only is None "
                          "for WAKING_ONLY or SWS_THEN_REM despite reported n_occupied>=2")
                    smoke_ok = False
                if no_writes["n_write_calls"] != 0:
                    print("  [SMOKE] FAIL: NO_WRITES n_write_calls="
                          f"{no_writes['n_write_calls']} -- expected exactly 0 "
                          "(sd016_writepath_mode='off', no sleep)")
                    smoke_ok = False

            if smoke_ok:
                print("[DRY RUN] PASS - all three conditions wire correctly; "
                      "BIAS write-selection clears the occupancy floor in "
                      "both comparison arms even at smoke scale, "
                      "occupied-slot cosine is computable, and NO_WRITES "
                      "stays at zero write calls as expected")
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
                "contextmemory_write_usage_balancing": CONTEXTMEMORY_WRITE_USAGE_BALANCING,
                "contextmemory_write_usage_bias_weight": CONTEXTMEMORY_WRITE_USAGE_BIAS_WEIGHT,
                "contextmemory_write_usage_decay": CONTEXTMEMORY_WRITE_USAGE_DECAY,
                "sd016_writepath_mode": _wp_mode,
                "context_memory_memory_requires_grad": False,
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

    # --- Re-derived untouched-bank null (inherited from 436e) ---------------
    _num_slots = arm_results[0]["slot_danger_ema"].__len__() if arm_results else 16
    _memory_dim = 128
    untouched_bank_null = _derive_untouched_bank_null(_num_slots, _memory_dim)
    _null_mean = untouched_bank_null["mean"]
    _null_sd = max(untouched_bank_null["sd"], 1e-9)

    # --- P0 READINESS GATE (7 checks; all must clear) ------------------------
    pooled_sws_n_writes = sum(r["sleep_cycle_sws_n_writes_total"] for r in sws_r)
    pooled_rem_n_rollouts = sum(r["sleep_cycle_rem_n_rollouts_total"] for r in sws_r)
    pooled_waking_write_calls = sum(r["n_write_calls"] for r in waking)

    n_scoreable_seeds = sum(
        1 for w_r, s_r in zip(waking, sws_r)
        if w_r["n_occupied_slots"] >= 2 and s_r["n_occupied_slots"] >= 2
    )

    no_writes_abs_z = [
        abs(r["slot_cosine_sim_raw_whole_bank"] - _null_mean) / _null_sd
        for r in no_writes_r
    ]
    max_no_writes_abs_z = max(no_writes_abs_z) if no_writes_abs_z else float("inf")

    # 436g NEW: BIAS occupancy floor across every non-NO_WRITES cell.
    non_no_writes_occupancy = [r["n_occupied_slots"] for r in (waking + sws_r)]
    min_bias_occupancy = min(non_no_writes_occupancy) if non_no_writes_occupancy else 0

    readiness_checks = [
        {
            "name": "sws_context_memory_writes_occur",
            "measured": pooled_sws_n_writes,
            "threshold": SWS_WRITES_FLOOR,
            "direction": "lower",
            "control": (
                "sum of ContextMemory.write() calls (sws_n_writes) across "
                "every SWS_THEN_REM sleep pass, pooled across all "
                f"{len(SEEDS)} seeds. Unchanged from 436c-436f."
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
                f"{len(SEEDS)} seeds. Unchanged from 436c-436f."
            ),
        },
        {
            "name": "waking_writepath_engaged",
            "measured": pooled_waking_write_calls,
            "threshold": WAKING_WRITEPATH_CALLS_FLOOR,
            "direction": "lower",
            "control": (
                "sum of ContextMemory.write() calls fired by the per-tick "
                "sense_only hook across all WAKING_ONLY seeds, pooled "
                f"across all {len(SEEDS)} seeds. Unchanged from 436e/436f."
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
                "pairwise mean to exist). Unchanged from 436e/436f -- "
                "expected to clear decisively under BIAS write-selection "
                "per V3-EXQ-943's substrate-validation reading (16/16 "
                "occupied slots on 5/5 seeds)."
            ),
        },
        {
            "name": "adam_drift_neutralized",
            "measured": max_no_writes_abs_z,
            "threshold": ADAM_DRIFT_NULL_TOLERANCE_SIGMA,
            "direction": "upper",
            "control": (
                "max |z-score| of the NO_WRITES calibration arm's raw "
                "whole-bank cosine against the freshly-derived untouched-"
                f"bank null (mean={_null_mean:.6f}, sd={_null_sd:.6f}, "
                f"n_draws={untouched_bank_null['n_draws']}), across all "
                f"{len(SEEDS)} seeds. Unchanged from 436e/436f."
            ),
        },
        {
            "name": "bias_occupancy_confirms_fix",
            "measured": min_bias_occupancy,
            "threshold": BIAS_OCCUPANCY_FLOOR,
            "direction": "lower",
            "control": (
                "min per-seed n_occupied_slots across ALL non-NO_WRITES "
                f"cells (WAKING_ONLY + SWS_THEN_REM, {len(SEEDS)} seeds "
                "each) -- NEW in 436g, the direct empirical confirmation "
                "that BIAS write-selection (contextmemory_write_usage_"
                "balancing=True) actually cleared the occupancy floor in "
                "THIS driver's own env/training-loop configuration, not "
                "merely in V3-EXQ-943's substrate-validation harness."
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
            "bias_occupancy_confirms_fix": "bias_occupancy_fix_did_not_transfer",
        }
        for p in precs:
            if not p.get("met", True):
                return _LABELS.get(p.get("name"), "p0_readiness_unmet")
        return "p0_readiness_unmet"

    per_seed_diff: Dict[str, Dict] = {}
    scoreable_deltas: List[float] = []
    for w_r, s_r in zip(waking, sws_r):
        seed = w_r["seed"]
        occ_w = w_r["slot_cosine_sim_occupied_only"]
        occ_s = s_r["slot_cosine_sim_occupied_only"]
        c1_scoreable = occ_w is not None and occ_s is not None
        signed_diff = (occ_s - occ_w) if c1_scoreable else None
        if c1_scoreable:
            scoreable_deltas.append(signed_diff)
        c1a_sign_passes = bool(c1_scoreable and occ_s < occ_w)
        slot_sep_diff = s_r["slot_separation"] - w_r["slot_separation"]
        harm_dang_diff = s_r["harm_rate_dangerous"] - w_r["harm_rate_dangerous"]
        harm_safe_diff = s_r["harm_rate_safe"] - w_r["harm_rate_safe"]
        per_seed_diff[str(seed)] = {
            "seed": seed,
            "waking_slot_cosine_sim_occupied_only": occ_w,
            "sws_then_rem_slot_cosine_sim_occupied_only": occ_s,
            "occupied_only_signed_diff": signed_diff,
            "waking_n_occupied_slots": w_r["n_occupied_slots"],
            "sws_then_rem_n_occupied_slots": s_r["n_occupied_slots"],
            "waking_n_write_calls": w_r["n_write_calls"],
            "sws_then_rem_n_write_calls": s_r["n_write_calls"],
            "waking_slot_cosine_sim_raw_whole_bank": w_r["slot_cosine_sim_raw_whole_bank"],
            "sws_then_rem_slot_cosine_sim_raw_whole_bank": s_r["slot_cosine_sim_raw_whole_bank"],
            "c1_scoreable": c1_scoreable,
            # C1(a): PRIMARY sign-test half (once P0 is met). Directional
            # (per ARC-045's own experimental implication) -- SWS_THEN_REM
            # strictly lower than WAKING_ONLY, on the corrected occupied-
            # only statistic.
            "slot_cosine_sim_occupied_only_passes_C1a_sign": c1a_sign_passes,
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

    c1a_count = sum(1 for d in per_seed_diff.values() if d["slot_cosine_sim_occupied_only_passes_C1a_sign"])
    c1a_pass = c1a_count >= C1_N_SEEDS_REQUIRED

    # --- 436g NEW: C1(b) effect-size margin (both tails floored) ------------
    pooled_delta_sd = statistics.pstdev(scoreable_deltas) if len(scoreable_deltas) >= 2 else 0.0
    effect_size_margin = max(
        EFFECT_SIZE_ABS_FLOOR,
        EFFECT_SIZE_SD_MULT * pooled_delta_sd,
        NULL_SD_MULT * _null_sd,
    )
    mean_signed_delta = (
        statistics.fmean(scoreable_deltas) if scoreable_deltas else float("nan")
    )
    c1b_effect_size_pass = bool(
        scoreable_deltas and mean_signed_delta <= -effect_size_margin
    )

    c1_pass = ready and c1a_pass and c1b_effect_size_pass

    c4_count = sum(1 for d in per_seed_diff.values() if d["sws_then_rem_slot_separation_passes_C4"])
    c4_pass = c4_count >= C4_N_SEEDS_REQUIRED

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
        if c1_pass:
            label = "sws_then_rem_differentiates_occupied_slots"
        elif c1a_pass and not c1b_effect_size_pass:
            label = "sws_then_rem_direction_consistent_but_below_effect_size_floor"
        else:
            label = "sws_then_rem_does_not_differentiate_occupied_slots"
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
            "min_bias_occupancy_across_non_no_writes_cells": min_bias_occupancy,
            "bias_occupancy_floor": BIAS_OCCUPANCY_FLOOR,
            "ready": ready,
            "desc": ("P0 GATE (7 checks, all must clear). 5 unchanged from "
                     "436e/436f (sleep-pass write/rollout counters, waking "
                     "writepath engagement, occupancy sufficiency, Adam-"
                     "drift-neutralization calibration); 1 NEW in 436g "
                     "(bias_occupancy_confirms_fix -- direct empirical "
                     "confirmation that BIAS write-selection cleared "
                     "occupancy in this driver's own configuration)."),
        },
        "C1_primary_slot_cosine_sim_occupied_only_directional": {
            "n_seeds_required": C1_N_SEEDS_REQUIRED,
            "n_seeds_passed_sign_test": c1a_count,
            "sign_test_pass": c1a_pass,
            "mean_signed_delta": mean_signed_delta,
            "pooled_delta_sd": pooled_delta_sd,
            "untouched_bank_null_sd": _null_sd,
            "effect_size_margin": effect_size_margin,
            "effect_size_margin_components": {
                "abs_floor": EFFECT_SIZE_ABS_FLOOR,
                "sd_mult_x_pooled_delta_sd": EFFECT_SIZE_SD_MULT * pooled_delta_sd,
                "null_sd_mult_x_null_sd": NULL_SD_MULT * _null_sd,
            },
            "effect_size_pass": c1b_effect_size_pass,
            "pass": c1_pass,
            "scored": ready,
            "desc": ("SOLE PASS/FAIL GATE once P0 is met. TWO conjunctive "
                     "sub-criteria (no bare sign test, per this week's "
                     "standing design constraint): (a) sign test -- "
                     "slot_cosine_sim_occupied_only(SWS_THEN_REM) < "
                     "slot_cosine_sim_occupied_only(WAKING_ONLY) in >= 3/5 "
                     "seeds [inherited from 436e/436f]; (b) effect-size "
                     "margin -- mean signed delta (pooled over scoreable "
                     "seeds) more negative than -max(abs_floor, sd_mult * "
                     "pooled_delta_sd, null_sd_mult * untouched_bank_null_"
                     "sd) [NEW in 436g]. Computed over ONLY slots that "
                     "received >=1 ContextMemory.write() call this run."),
        },
        "C4_arc045_slot_separation_threshold": {
            "threshold": C4_SLOT_SEPARATION_THRESHOLD,
            "n_seeds_required": C4_N_SEEDS_REQUIRED,
            "n_seeds_passed": c4_count,
            "pass": c4_pass,
            "scored": ready,
            "desc": ("SECONDARY, non-gating. slot_separation in SWS_THEN_REM "
                     "> 0.3 in >= 3/5 seeds. Unchanged formula from 436d-"
                     "436f -- a read-side visitation statistic, independent "
                     "of the write-occupancy cosine confound."),
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
                     "for gating. See module docstring."),
        },
        "aggregated": {
            "per_seed": per_seed_diff,
            "n_seeds_passed": {"C1_sign_test": c1a_count, "C4": c4_count},
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
            "EFFECT_SIZE_ABS_FLOOR": EFFECT_SIZE_ABS_FLOOR,
            "EFFECT_SIZE_SD_MULT": EFFECT_SIZE_SD_MULT,
            "NULL_SD_MULT": NULL_SD_MULT,
            "BIAS_OCCUPANCY_FLOOR": BIAS_OCCUPANCY_FLOOR,
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
            "SD-017/ARC-045/MECH-166 slot-differentiation CEILING RETEST, "
            "executing the 2026-08-30 A.6 due-ness assessment "
            "(chip-20260830-sd017-ceiling-retest, already ratified -- this "
            "run does not re-decide the due-ness question). Inherits 436e's "
            "occupied-slots-only slot_cosine_sim DV machinery verbatim "
            "(write-tracker, Adam-drift neutralization, re-derived "
            "untouched-bank null, 3-condition NO_WRITES/WAKING_ONLY/"
            "SWS_THEN_REM design). Changes vs 436e/436f: (1) arms BIAS "
            "write-selection (E1Config.contextmemory_write_usage_"
            "balancing=True) instead of 436f's SD-016 read-path production "
            "combination -- 436f already demonstrated the read-path "
            "combination does not move write-path occupancy by even one "
            "seed (n_occupied_slots identical to 436e on all 5 seeds "
            "despite sd016_arming_engaged confirmed); BIAS resolves the "
            "underlying write-path addressing degeneracy directly, "
            "validated at 16/16 occupied slots on 5/5 seeds by the "
            "substrate's own V3-EXQ-943 validation run. (2) adds a 7th P0 "
            "gate (bias_occupancy_confirms_fix) confirming BIAS clears "
            "occupancy in THIS driver's configuration specifically. (3) "
            "adds an effect-size margin to C1 on top of the inherited sign "
            "test (no bare sign test), floored on the max of an absolute "
            "constant, the run's own observed delta SD, and the re-derived "
            "untouched-bank null's own SD -- both tails floored so no "
            "single source can be gamed down to zero. (4) deliberately does "
            "NOT compute or gate on the 2-cluster occupied-slot-set Jaccard "
            "content-discrimination statistic from the V3-EXQ-956 lineage "
            "-- the 2026-08-29 scoping spike diagnosed that instrument as "
            "mis-posed (near-binary, aliasing-prone at n=5) and explicitly "
            "recommended against gating this retest on it; the occupied-"
            "slots-only cosine DV inherited from 436e is itself a graded, "
            "non-aliasing content-discrimination readout, so nothing "
            "additional is needed or added. RE-DERIVE BRAKE STATUS: SD-017 "
            "had a 3rd substrate_ceiling hit recorded against it as of "
            "436f (failure_autopsy_436f-603u-precondition-blocked-cluster_"
            "2026-08-16), which explicitly REFUSED a same-question 436g "
            "re-queue until the write-path build landed. That build "
            "(substrate_queue.json:contextmemory-write-path-addressing-"
            "degeneracy) landed 2026-08-19 (status: implemented_pending_"
            "validation, ready: true; BIAS occupancy floor independently "
            "confirmed by V3-EXQ-943) -- the Step 2.5b release condition -- "
            "so the brake is RELEASED for this run. Manipulation under test "
            "is unchanged from every prior letter: SWS_THEN_REM (offline "
            "consolidation) vs WAKING_ONLY (online writes only)."
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
            "contextmemory_write_usage_balancing": CONTEXTMEMORY_WRITE_USAGE_BALANCING,
            "contextmemory_write_usage_bias_weight": CONTEXTMEMORY_WRITE_USAGE_BIAS_WEIGHT,
            "contextmemory_write_usage_decay": CONTEXTMEMORY_WRITE_USAGE_DECAY,
            "context_memory_memory_requires_grad": False,
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
